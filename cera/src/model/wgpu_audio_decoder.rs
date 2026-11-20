//! WGPU-accelerated audio detokenizer (spectrum).
//!
//! Mirrors `metal_audio_decoder`, but drives the batched WGPU
//! kernels the LFM2 prefill path already ships (`mul_mat_reg_tile` f32,
//! `rmsnorm_batch`, `add_rmsnorm_batch`, `qk_norm_rope_batch`,
//! `conv1d_fused_batch`, `flash_attention`, `silu_mul`, `bias_add`) rather than
//! the per-token Metal helpers. No new kernels.
//!
//! The 6 upsampled tokens of one frame are run as a single 6-token batch
//! (LFM2's shortconv is sequential per channel inside `conv1d_fused_batch`; the
//! detokenizer attention is non-causal over the sliding window, so each token's
//! query is attended with the decode `flash_attention` kernel that reads the
//! whole live window). Weights are dequantized to f32 and uploaded, matching the
//! Metal decoder's precision path. KV caches are f32 via the dedicated
//! `FLASH_ATTENTION_F32` twin (the main `flash_attention` kernel reads packed
//! f16 halves for the LLM path), so there is no f16 cast and the path is a
//! touch more accurate than Metal's f16 cache.
//!
//! Scope: both the detokenizer and depthformer run on WGPU. The depthformer
//! samples codebooks with a WGPU compute pipeline while fallback CPU decoding
//! remains available via `AudioOutputDecoder`.
//!
//! The final ISTFT (`istft_to_pcm`) runs on the GPU too: `exp_polar` maps the
//! polar half-spectrum to complex, a reg-tile GEMM against a precomputed real
//! inverse-DFT basis does the iDFT, and `overlap_add` windows and folds the
//! frames into PCM. Only the startup-pad strip stays on the CPU after readback.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result};
use wgpu::Buffer;

use crate::backend::wgpu::{GpuContext, shaders};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{
    DecoderConfig, DepthformerConfig, DetokenizerConfig, DetokenizerWeights,
};
use crate::model::gpu_lfm2::{
    MUL_MAT_TILE_K, MUL_MAT_TILE_M, MUL_MAT_TILE_N, MUL_MAT_TILE_WG_M, MUL_MAT_TILE_WG_N,
};

/// Tokens per detokenizer frame (fixed upsample factor, mirrors the CPU and
/// Metal paths).
const N_FRAMES: usize = 6;

/// F32 weight matrix on the GPU, row-major `[m rows x k cols]` (an `[out, in]`
/// projection), plus the reg-tile geometry needed to dispatch it.
struct GpuWeight {
    buf: Buffer,
    m: u32,
    k: u32,
}

/// Per-layer depthformer weights and pre-baked bind groups on GPU.
struct DfLayerGpu {
    operator_norm_bg: wgpu::BindGroup,
    gemm_qkv_bg: wgpu::BindGroup,
    rope_bgs: Vec<wgpu::BindGroup>,    // 8 entries for pos 0..8
    kv_copy_bgs: Vec<wgpu::BindGroup>, // 8 entries for pos 0..8
    attn_bgs: Vec<wgpu::BindGroup>,    // 8 entries for pos 0..8
    gemm_wo_accum_bg: wgpu::BindGroup,
    ffn_norm_bg: wgpu::BindGroup,
    ffn_gate_up_bg: wgpu::BindGroup,
    gemm_w2_accum_bg: wgpu::BindGroup,
}

/// Per-layer detokenizer weights on the GPU (conv XOR attention, plus FFN).
struct DetokLayerGpu {
    operator_norm: Buffer,
    ffn_norm: Buffer,
    ffn_w1: GpuWeight,
    ffn_w2: GpuWeight,
    ffn_w3: GpuWeight,
    // Conv-only.
    conv_in_proj: Option<GpuWeight>,
    conv_out_proj: Option<GpuWeight>,
    conv_weight: Option<Buffer>,
    // Attention-only.
    wq: Option<GpuWeight>,
    wk: Option<GpuWeight>,
    wv: Option<GpuWeight>,
    wo: Option<GpuWeight>,
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
}

#[derive(Clone)]
pub(crate) struct DepthformerPipelines {
    gemv_f32: wgpu::ComputePipeline,
    gemv_f32_accum: wgpu::ComputePipeline,
    argmax_f32: wgpu::ComputePipeline,
    df_embed_add: wgpu::ComputePipeline,
    df_kv_copy: wgpu::ComputePipeline,
    df_sample_logits: wgpu::ComputePipeline,
    df_ffn_gate_up: wgpu::ComputePipeline,
    df_qkv_gemv: wgpu::ComputePipeline,
}

#[derive(Clone)]
pub(crate) struct Pipelines {
    gemm_f32: wgpu::ComputePipeline,
    rmsnorm_batch: wgpu::ComputePipeline,
    add_rmsnorm_batch: wgpu::ComputePipeline,
    qk_norm_rope_batch: wgpu::ComputePipeline,
    conv1d_fused_batch: wgpu::ComputePipeline,
    flash_attention: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
    add_inplace: wgpu::ComputePipeline,
    bias_add: wgpu::ComputePipeline,
    exp_polar: wgpu::ComputePipeline,
    overlap_add: wgpu::ComputePipeline,
    df: Option<DepthformerPipelines>,
}

/// Orchestration seam behind [`WgpuAudioDecoder::from_ggufs_with_context`] pipeline
/// validation so pre-weight validation, optional depthformer pipeline alignment, and
/// lost-device aborts are testable on host without a GPU.
fn init_audio_decoder_pipelines_with<Core, Df>(
    try_df: bool,
    mut device_lost_detail: impl FnMut() -> Option<String>,
    build_core: impl FnOnce() -> Result<Core>,
    build_df: impl FnOnce() -> Result<Option<Df>>,
) -> Result<(Core, Option<Df>)> {
    if let Some(detail) = device_lost_detail() {
        anyhow::bail!("audio_decoder pipelines: GPU device lost ({detail})");
    }
    let core = build_core()?;
    let df = if try_df { build_df()? } else { None };
    Ok((core, df))
}

/// Orchestration seam behind the post-upload tail of [`WgpuAudioDecoder::from_ggufs_with_context`]
/// so device-loss checks before depthformer upload, on depthformer error, and after all uploads
/// stay unit-testable on host without a GPU.
fn finish_audio_decoder_load_with<Df>(
    has_df_pipes: bool,
    mut ensure_alive: impl FnMut(&str) -> Result<()>,
    mut is_device_lost: impl FnMut() -> bool,
    load_df: impl FnOnce() -> Result<Df>,
) -> Result<Option<Df>> {
    ensure_alive("audio_decoder weight upload")?;
    let depthformer = if has_df_pipes {
        match load_df() {
            Ok(df) => {
                tracing::info!("[cera:wgpu_audio_decoder] WebGPU depthformer loaded successfully");
                Some(df)
            }
            Err(e) => {
                if is_device_lost() {
                    return Err(e);
                }
                tracing::warn!(
                    "[cera:wgpu_audio_decoder] WebGPU depthformer unavailable ({e:#}), using CPU depthformer"
                );
                eprintln!(
                    "[cera:wgpu_audio_decoder] WebGPU depthformer unavailable ({e:#}), using CPU depthformer"
                );
                None
            }
        }
    } else {
        None
    };
    ensure_alive("audio_decoder post-upload")?;
    Ok(depthformer)
}

impl Pipelines {
    /// Compile and validate the mandatory audio detokenizer/ISTFT pipelines, plus
    /// optional depthformer pipelines when `try_df` is true (degrading only the
    /// depthformer pipelines on non-device-loss validation failure).
    pub(crate) fn build_validated(ctx: &GpuContext, try_df: bool) -> Result<Self> {
        let (mut pipes, df_pipes) = init_audio_decoder_pipelines_with(
            try_df,
            || ctx.device_lost_detail(),
            || {
                ctx.check_pipelines("audio_decoder pipelines", || {
                    // f32 reg-tile GEMM matches the LFM2 prefill's dense fallback kernel.
                    let gemm_f32 = ctx.create_pipeline_with_defines(
                        shaders::MUL_MAT_REG_TILE,
                        "main",
                        "audio_detok_gemm_f32",
                        &[
                            ("SRC0_INNER_TYPE", "f32"),
                            ("INIT_SRC0_SHMEM_FLOAT", ""),
                            ("WORKGROUP_SIZE_M", &format!("{MUL_MAT_TILE_WG_M}u")),
                            ("WORKGROUP_SIZE_N", &format!("{MUL_MAT_TILE_WG_N}u")),
                            ("TILE_M", &format!("{MUL_MAT_TILE_M}u")),
                            ("TILE_N", &format!("{MUL_MAT_TILE_N}u")),
                            ("TILE_K", &format!("{MUL_MAT_TILE_K}u")),
                        ],
                    );
                    Pipelines {
                        gemm_f32,
                        rmsnorm_batch: ctx.create_pipeline(
                            shaders::RMSNORM_BATCH,
                            "rmsnorm_batch",
                            "audio_detok_rmsnorm_batch",
                        ),
                        add_rmsnorm_batch: ctx.create_pipeline(
                            shaders::RMSNORM_BATCH,
                            "add_rmsnorm_batch",
                            "audio_detok_add_rmsnorm_batch",
                        ),
                        qk_norm_rope_batch: ctx.create_pipeline(
                            shaders::QK_NORM_ROPE_BATCH,
                            "qk_norm_rope_batch",
                            "audio_detok_qk_norm_rope_batch",
                        ),
                        conv1d_fused_batch: ctx.create_pipeline(
                            shaders::CONV1D_FUSED_BATCH,
                            "conv1d_fused_batch",
                            "audio_detok_conv1d_fused_batch",
                        ),
                        flash_attention: ctx.create_pipeline(
                            shaders::FLASH_ATTENTION_F32,
                            "flash_attention",
                            "audio_detok_flash_attention",
                        ),
                        silu_mul: ctx.create_pipeline(
                            shaders::ELEMENTWISE,
                            "silu_mul_inplace",
                            "audio_detok_silu_mul",
                        ),
                        add_inplace: ctx.create_pipeline(
                            shaders::ELEMENTWISE,
                            "add_inplace",
                            "audio_detok_add_inplace",
                        ),
                        bias_add: ctx.create_pipeline(
                            shaders::BIAS_ADD,
                            "bias_add",
                            "audio_detok_bias_add",
                        ),
                        exp_polar: ctx.create_pipeline(
                            shaders::EXP_POLAR,
                            "exp_polar",
                            "audio_istft_exp_polar",
                        ),
                        overlap_add: ctx.create_pipeline(
                            shaders::OVERLAP_ADD,
                            "overlap_add",
                            "audio_istft_overlap_add",
                        ),
                        df: None,
                    }
                })
            },
            || {
                ctx.degrade_pipeline_err(
                    "audio_depthformer",
                    "using CPU depthformer",
                    ctx.check_pipelines("audio_depthformer pipelines", || DepthformerPipelines {
                        gemv_f32: ctx.create_pipeline(
                            shaders::GEMV_F32,
                            "gemv_f32",
                            "audio_df_gemv_f32",
                        ),
                        gemv_f32_accum: ctx.create_pipeline(
                            shaders::GEMV_F32,
                            "gemv_f32_accum",
                            "audio_df_gemv_f32_accum",
                        ),
                        argmax_f32: ctx.create_pipeline(
                            shaders::ARGMAX_F32,
                            "argmax_f32",
                            "audio_df_argmax_f32",
                        ),
                        df_embed_add: ctx.create_pipeline(
                            include_str!("../backend/shaders/df_embed_add.wgsl"),
                            "df_embed_add",
                            "audio_df_embed_add",
                        ),
                        df_kv_copy: ctx.create_pipeline(
                            include_str!("../backend/shaders/df_kv_cache_copy.wgsl"),
                            "df_kv_cache_copy",
                            "audio_df_kv_copy",
                        ),
                        df_sample_logits: ctx.create_pipeline(
                            include_str!("../backend/shaders/df_sample_logits.wgsl"),
                            "df_sample_logits",
                            "audio_df_sample_logits",
                        ),
                        df_ffn_gate_up: ctx.create_pipeline(
                            include_str!("../backend/shaders/df_ffn_gate_up.wgsl"),
                            "df_ffn_gate_up",
                            "audio_df_ffn_gate_up",
                        ),
                        df_qkv_gemv: ctx.create_pipeline(
                            include_str!("../backend/shaders/df_qkv_gemv.wgsl"),
                            "df_qkv_gemv",
                            "audio_df_qkv_gemv",
                        ),
                    }),
                )
            },
        )?;
        pipes.df = df_pipes;
        Ok(pipes)
    }
}

pub struct WgpuAudioDecoder {
    ctx: GpuContext,
    cfg: DetokenizerConfig,
    pipes: Arc<Pipelines>,
    depthformer: Option<WgpuDepthformer>,

    layers: Vec<DetokLayerGpu>,
    output_norm: Buffer,
    lin_w: GpuWeight,
    lin_b: Buffer,

    // GPU ISTFT: real inverse-DFT basis `[n_fft x 2·n_fft_bins]` and the Hann
    // window `[n_fft]`, both derived from the config and uploaded once.
    idft_basis: GpuWeight,
    hann: Buffer,

    // Scratch (all sized for N_FRAMES tokens).
    hidden_buf: Buffer,       // residual stream [N x hs]
    normed_buf: Buffer,       // rmsnorm out / attn out [N x max(hs, q_dim)]
    proj_buf: Buffer,         // conv in_proj [N x 3hs] / attn Q [N x q_dim]
    gate_buf: Buffer,         // attn K / op residual / ffn gate [N x max(hs, kv_dim, ffn)]
    up_buf: Buffer,           // attn V / ffn up / ffn down [N x max(hs, kv_dim, ffn)]
    q_single: Buffer,         // one token's Q [q_dim]
    attn_single: Buffer,      // one token's attn out [q_dim]
    spectrum_buf: Buffer,     // [N x (n_fft_bins*2)]
    rope_freqs_dummy: Buffer, // 1-element buffer bound at qk_norm_rope binding 5

    // Persistent state.
    conv_bufs: Vec<Option<Buffer>>, // [d_conv x hs] per conv layer
    kv_k: Vec<Option<Buffer>>,      // [swa x kv_dim] f32 per attn layer
    kv_v: Vec<Option<Buffer>>,
    n_past: AtomicUsize,
    session_active: AtomicBool,
    /// Sticky record of a failed sync depthformer sample (see
    /// `sample_audio_frame`): the async error has no readback slot to
    /// land in, so the sync wrapper records it here for
    /// [`crate::model::audio_decoder::AudioAccelerator::take_audio_error`].
    sample_error: std::sync::Mutex<Option<crate::CeraError>>,
}

fn get_detok_tensor(gguf: &GgufFile, name1: &str, name2: &str) -> Result<crate::tensor::Tensor> {
    if gguf.tensors.contains_key(name1) {
        gguf.get_tensor(name1)
            .with_context(|| format!("failed to load tensor `{name1}`"))
    } else if gguf.tensors.contains_key(name2) {
        gguf.get_tensor(name2)
            .with_context(|| format!("failed to load tensor `{name2}`"))
    } else {
        anyhow::bail!("tensor not found: neither `{name1}` nor `{name2}`")
    }
}

impl WgpuAudioDecoder {
    /// Pure-host metadata, probe-dtype, geometry, and tensor validation ahead of opening a WebGPU
    /// adapter (`GpuContext::new()`) or building compute pipelines.
    pub(crate) fn prevalidate_detokenizer_gguf(gguf: &GgufFile) -> Result<DetokenizerConfig> {
        let lookup_probe = |name1: &'static str,
                            name2: &'static str|
         -> Result<(&'static str, &crate::gguf::TensorInfo)> {
            let (label, info) = if let Some(info) = gguf.tensors.get(name1) {
                (name1, info)
            } else if let Some(info) = gguf.tensors.get(name2) {
                (name2, info)
            } else {
                anyhow::bail!("tensor not found: neither `{name1}` nor `{name2}`");
            };
            crate::model::audio_decoder::check_gguf_tensor_decodable_to_f32(info, label)?;
            Ok((label, info))
        };

        let (conv_lbl, conv_info) = lookup_probe(
            "lfm.layers.0.conv.in_proj.weight",
            "blk.0.shortconv.in_proj.weight",
        )?;
        let (n_embd, _) =
            crate::model::audio_decoder::tensor_2d_cols_rows(&conv_info.shape, conv_lbl)?;
        let (_qn_lbl, qn_info) = lookup_probe(
            "lfm.layers.2.self_attn.q_layernorm.weight",
            "blk.2.attn_q_norm.weight",
        )?;
        // An empty shape is rejected cleanly below by validate_detokenizer_dims.
        let head_dim = qn_info.shape.first().copied().unwrap_or(0);
        let (q_lbl, q_info) = lookup_probe(
            "lfm.layers.2.self_attn.q_proj.weight",
            "blk.2.attn_q.weight",
        )?;
        let (_, q_rows) = crate::model::audio_decoder::tensor_2d_cols_rows(&q_info.shape, q_lbl)?;
        let (k_lbl, k_info) = lookup_probe(
            "lfm.layers.2.self_attn.k_proj.weight",
            "blk.2.attn_k.weight",
        )?;
        let (_, k_rows) = crate::model::audio_decoder::tensor_2d_cols_rows(&k_info.shape, k_lbl)?;
        let (w1_lbl, w1_info) = lookup_probe(
            "lfm.layers.0.feed_forward.w1.weight",
            "blk.0.ffn_gate.weight",
        )?;
        let (_, ffn_dim) =
            crate::model::audio_decoder::tensor_2d_cols_rows(&w1_info.shape, w1_lbl)?;

        let (n_head, n_kv) = crate::model::audio_decoder::validate_detokenizer_dims(
            n_embd,
            head_dim,
            q_rows,
            k_rows,
            ffn_dim,
            crate::model::audio_decoder::HeadDimBounds::WGPU,
        )?;

        let layer_is_conv = vec![true, true, false, true, false, true, false, true];
        let n_layer = layer_is_conv.len();
        let cfg = DetokenizerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            n_embd_head: head_dim,
            ffn_dim,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 8,
            n_fft: 1280,
            hop_length: 320,
            sample_rate: 24000,
            layer_is_conv,
        };
        anyhow::ensure!(
            cfg.swa_window_size.is_multiple_of(N_FRAMES),
            "detokenizer swa_window_size {} must be a multiple of {N_FRAMES} for the single-chunk KV ring write",
            cfg.swa_window_size
        );
        crate::model::audio_decoder::validate_detokenizer_gguf_tensors(gguf, &cfg, true)?;
        Ok(cfg)
    }

    pub fn from_gguf(gguf: &Arc<GgufFile>, _vocoder_path: &Path) -> Result<Self> {
        let _ = Self::prevalidate_detokenizer_gguf(gguf)?;
        let ctx = GpuContext::new()?;
        Self::from_gguf_with_context(ctx, gguf)
    }

    pub fn from_gguf_with_context(ctx: GpuContext, gguf: &Arc<GgufFile>) -> Result<Self> {
        Self::from_ggufs_with_context(ctx, gguf, None)
    }

    pub fn from_ggufs_with_context(
        ctx: GpuContext,
        detok_gguf: &Arc<GgufFile>,
        depthformer_gguf: Option<&Arc<GgufFile>>,
    ) -> Result<Self> {
        let gguf = detok_gguf;
        let cfg = Self::prevalidate_detokenizer_gguf(gguf)?;
        let n_layer = cfg.n_layer;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_kv = cfg.n_head_kv;
        let head_dim = cfg.n_embd_head;
        let ffn_dim = cfg.ffn_dim;
        let kv_dim = n_kv * head_dim;
        let q_dim = n_head * head_dim;

        let df_src = depthformer_gguf.unwrap_or(detok_gguf);
        let try_df = depthformer_gguf.is_some() || df_src.get_u32("depthformer_n_layer").is_some();
        let pipes = Pipelines::build_validated(&ctx, try_df)?;

        // Dequantize each weight to f32 (CPU-matching precision) and upload.
        let make_weight = |name1: &str, name2: &str| -> Result<GpuWeight> {
            let t = get_detok_tensor(gguf, name1, name2)?;
            let f32_data = t.try_to_f32_vec()?;
            let shape = t.shape();
            let (rows, cols) = match shape.len() {
                1 => (1, shape[0]),
                2 => (shape[1], shape[0]),
                _ => anyhow::bail!("unexpected rank for {name1} / {name2}"),
            };
            let buf = ctx.upload_f32(&f32_data, name1);
            Ok(GpuWeight {
                buf,
                m: rows as u32,
                k: cols as u32,
            })
        };
        let upload_vec = |name1: &str, name2: &str| -> Result<Buffer> {
            Ok(ctx.upload_f32(
                &get_detok_tensor(gguf, name1, name2)?.try_to_f32_vec()?,
                name1,
            ))
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let pfx_lfm = format!("lfm.layers.{i}");
            let pfx_blk = format!("blk.{i}");
            let is_conv = cfg.layer_is_conv[i];
            let (cin, cop, cw) = if is_conv {
                (
                    Some(make_weight(
                        &format!("{pfx_lfm}.conv.in_proj.weight"),
                        &format!("{pfx_blk}.shortconv.in_proj.weight"),
                    )?),
                    Some(make_weight(
                        &format!("{pfx_lfm}.conv.out_proj.weight"),
                        &format!("{pfx_blk}.shortconv.out_proj.weight"),
                    )?),
                    Some(upload_vec(
                        &format!("{pfx_lfm}.conv.conv.weight"),
                        &format!("{pfx_blk}.shortconv.conv.weight"),
                    )?),
                )
            } else {
                (None, None, None)
            };
            let (wq, wk, wv, wo, qn, kn) = if !is_conv {
                (
                    Some(make_weight(
                        &format!("{pfx_lfm}.self_attn.q_proj.weight"),
                        &format!("{pfx_blk}.attn_q.weight"),
                    )?),
                    Some(make_weight(
                        &format!("{pfx_lfm}.self_attn.k_proj.weight"),
                        &format!("{pfx_blk}.attn_k.weight"),
                    )?),
                    Some(make_weight(
                        &format!("{pfx_lfm}.self_attn.v_proj.weight"),
                        &format!("{pfx_blk}.attn_v.weight"),
                    )?),
                    Some(make_weight(
                        &format!("{pfx_lfm}.self_attn.out_proj.weight"),
                        &format!("{pfx_blk}.attn_output.weight"),
                    )?),
                    Some(upload_vec(
                        &format!("{pfx_lfm}.self_attn.q_layernorm.weight"),
                        &format!("{pfx_blk}.attn_q_norm.weight"),
                    )?),
                    Some(upload_vec(
                        &format!("{pfx_lfm}.self_attn.k_layernorm.weight"),
                        &format!("{pfx_blk}.attn_k_norm.weight"),
                    )?),
                )
            } else {
                (None, None, None, None, None, None)
            };
            layers.push(DetokLayerGpu {
                operator_norm: upload_vec(
                    &format!("{pfx_lfm}.operator_norm.weight"),
                    &format!("{pfx_blk}.attn_norm.weight"),
                )?,
                ffn_norm: upload_vec(
                    &format!("{pfx_lfm}.ffn_norm.weight"),
                    &format!("{pfx_blk}.ffn_norm.weight"),
                )?,
                ffn_w1: make_weight(
                    &format!("{pfx_lfm}.feed_forward.w1.weight"),
                    &format!("{pfx_blk}.ffn_gate.weight"),
                )?,
                ffn_w2: make_weight(
                    &format!("{pfx_lfm}.feed_forward.w2.weight"),
                    &format!("{pfx_blk}.ffn_down.weight"),
                )?,
                ffn_w3: make_weight(
                    &format!("{pfx_lfm}.feed_forward.w3.weight"),
                    &format!("{pfx_blk}.ffn_up.weight"),
                )?,
                conv_in_proj: cin,
                conv_out_proj: cop,
                conv_weight: cw,
                wq,
                wk,
                wv,
                wo,
                q_norm: qn,
                k_norm: kn,
            });
        }

        let output_norm = upload_vec("lfm.embedding_norm.weight", "token_embd_norm.weight")?;
        let lin_w = make_weight("lin.weight", "dense_2.weight")?;
        let lin_b = upload_vec("lin.bias", "dense_2.bias")?;
        // The lin head GEMM writes `lin_w.m` rows at a `spec_per_frame` stride into
        // `spectrum_buf`. Both are pinned to `n_fft`, but assert the shared
        // assumption so a vocoder with a different head width fails at load rather
        // than silently overrunning the spectrum buffer.
        anyhow::ensure!(
            lin_w.m as usize == n_embd_bins(&cfg),
            "detokenizer lin.weight out-dim {} != spectrum width {}",
            lin_w.m,
            n_embd_bins(&cfg)
        );

        // GPU ISTFT tables, built once from the config. `idft_basis` is an
        // `[n_fft x 2·n_fft_bins]` weight consumed by the reg-tile GEMM exactly
        // like `lin_w`; `hann` is the length-`n_fft` window read by `overlap_add`.
        let basis = crate::model::audio_decoder::build_idft_basis(cfg.n_fft);
        let idft_basis = GpuWeight {
            buf: ctx.upload_f32(&basis, "audio_istft_idft_basis"),
            m: cfg.n_fft as u32,
            k: n_embd_bins(&cfg) as u32,
        };
        let hann = ctx.upload_f32(
            &crate::model::audio_decoder::build_hann(cfg.n_fft),
            "audio_istft_hann",
        );

        let pipes = Arc::new(pipes);

        let alloc = |n: usize, label: &str| ctx.create_storage_rw((n * 4) as u64, label);
        let big = n_embd.max(kv_dim).max(ffn_dim);
        let hidden_buf = alloc(N_FRAMES * n_embd, "audio_detok_hidden");
        let normed_buf = alloc(N_FRAMES * n_embd.max(q_dim), "audio_detok_normed");
        let proj_buf = alloc(N_FRAMES * (3 * n_embd).max(q_dim), "audio_detok_proj");
        let gate_buf = alloc(N_FRAMES * big, "audio_detok_gate");
        let up_buf = alloc(N_FRAMES * big, "audio_detok_up");
        let q_single = alloc(q_dim, "audio_detok_q_single");
        let attn_single = alloc(q_dim, "audio_detok_attn_single");
        let spec_len = N_FRAMES * (n_embd_bins(&cfg));
        let spectrum_buf = alloc(spec_len, "audio_detok_spectrum");
        let rope_freqs_dummy = ctx.upload_f32(&[1.0f32], "audio_detok_rope_dummy");

        let mut conv_bufs = vec![None; n_layer];
        let mut kv_k = vec![None; n_layer];
        let mut kv_v = vec![None; n_layer];
        for i in 0..n_layer {
            if cfg.layer_is_conv[i] {
                conv_bufs[i] = Some(alloc(cfg.d_conv * n_embd, "audio_detok_conv_rbuf"));
            } else {
                kv_k[i] = Some(alloc(cfg.swa_window_size * kv_dim, "audio_detok_kv_k"));
                kv_v[i] = Some(alloc(cfg.swa_window_size * kv_dim, "audio_detok_kv_v"));
            }
        }

        let depthformer = finish_audio_decoder_load_with(
            pipes.df.is_some(),
            |stage| ctx.ensure_alive(stage),
            || ctx.is_device_lost(),
            || WgpuDepthformer::try_from_gguf(&ctx, &pipes, df_src),
        )?;

        ctx.save_pipeline_cache();
        Ok(Self {
            ctx,
            cfg,
            pipes,
            depthformer,
            layers,
            output_norm,
            lin_w,
            lin_b,
            idft_basis,
            hann,
            hidden_buf,
            normed_buf,
            proj_buf,
            gate_buf,
            up_buf,
            q_single,
            attn_single,
            spectrum_buf,
            rope_freqs_dummy,
            conv_bufs,
            kv_k,
            kv_v,
            n_past: AtomicUsize::new(0),
            session_active: AtomicBool::new(false),
            sample_error: std::sync::Mutex::new(None),
        })
    }

    pub fn config(&self) -> &DetokenizerConfig {
        &self.cfg
    }

    pub fn reset(&self) {
        self.n_past.store(0, Ordering::Relaxed);
        // Zero the conv rolling buffers. The KV caches need no clearing: n_past=0
        // bounds `flash_attention` to the entries this generation writes.
        let zeros_len = self.cfg.d_conv * self.cfg.n_embd;
        let zeros = vec![0.0f32; zeros_len];
        for b in self.conv_bufs.iter().flatten() {
            self.ctx
                .queue
                .write_buffer(b, 0, bytemuck::cast_slice(&zeros));
        }
    }

    // ── encode helpers (add passes to a shared encoder) ─────────────────────

    fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bufs: &[&Buffer],
        workgroups: (u32, u32, u32),
        label: &str,
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        let mut pass = self.ctx.begin_pass(enc, label);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    fn params(&self, data: &[u32], label: &str) -> Buffer {
        self.ctx.upload_storage(bytemuck::cast_slice(data), label)
    }

    /// `y[n, m] = x[n, k] * wᵀ` via the f32 reg-tile GEMM. `x_stride` / `y_stride`
    /// are f32 elements between consecutive token rows.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &GpuWeight,
        x: &Buffer,
        y: &Buffer,
        n: u32,
        x_stride: u32,
        y_stride: u32,
    ) {
        let params = self.params(
            &[w.m, w.k, n, x_stride, y_stride],
            "audio_detok_gemm_params",
        );
        let wg_m = w.m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
        let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);
        self.encode(
            enc,
            &self.pipes.gemm_f32,
            &[&w.buf, x, y, &params],
            (wg_m, wg_n, 1),
            "audio_detok_gemm",
        );
    }

    fn rmsnorm_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &Buffer,
        dst: &Buffer,
        weight: &Buffer,
        n: u32,
    ) {
        let hs = self.cfg.n_embd as u32;
        let p = self.params(
            &[
                hs,
                self.cfg.rms_norm_eps.to_bits(),
                hs,
                hs,
                1.0f32.to_bits(),
            ],
            "audio_detok_rmsnorm_params",
        );
        self.encode(
            enc,
            &self.pipes.rmsnorm_batch,
            &[src, dst, weight, &p],
            (n, 1, 1),
            "audio_detok_rmsnorm",
        );
    }

    /// `src += residual; dst = rmsnorm(src, weight)`. `dst` and `residual` must
    /// be distinct buffers (WGPU rejects binding-aliasing).
    fn add_rmsnorm_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &Buffer,
        dst: &Buffer,
        weight: &Buffer,
        residual: &Buffer,
        n: u32,
    ) {
        let hs = self.cfg.n_embd as u32;
        let p = self.params(
            &[
                hs,
                self.cfg.rms_norm_eps.to_bits(),
                hs,
                hs,
                1.0f32.to_bits(),
            ],
            "audio_detok_add_rmsnorm_params",
        );
        self.encode(
            enc,
            &self.pipes.add_rmsnorm_batch,
            &[src, dst, weight, &p, residual],
            (n, 1, 1),
            "audio_detok_add_rmsnorm",
        );
    }

    fn silu_mul(&self, enc: &mut wgpu::CommandEncoder, gate: &Buffer, up: &Buffer, total: u32) {
        let p = self.params(&[total, 0], "audio_detok_silu_params");
        self.encode(
            enc,
            &self.pipes.silu_mul,
            &[gate, up, &p],
            (total.div_ceil(256), 1, 1),
            "audio_detok_silu_mul",
        );
    }

    fn add_inplace(&self, enc: &mut wgpu::CommandEncoder, a: &Buffer, b: &Buffer, n: u32) {
        let p = self.params(&[n, 0], "audio_detok_add_params");
        self.encode(
            enc,
            &self.pipes.add_inplace,
            &[a, b, &p],
            (n.div_ceil(256), 1, 1),
            "audio_detok_add",
        );
    }

    // ── public forward ──────────────────────────────────────────────────────

    fn encode_detokenize_to_spectrum(
        &self,
        enc: &mut wgpu::CommandEncoder,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> (usize, usize, usize) {
        let hs = self.cfg.n_embd;
        let hd = self.cfg.n_embd_head;
        let n_head = self.cfg.n_head as u32;
        let n_kv = self.cfg.n_head_kv as u32;
        let q_dim = n_head * hd as u32;
        let kv_dim = n_kv * hd as u32;
        let ffn = self.cfg.ffn_dim as u32;
        let n = N_FRAMES as u32;
        let hs_u = hs as u32;
        let spec_per_frame = n_embd_bins(&self.cfg);

        // 1. Embed + upsample on CPU, upload as the residual stream.
        let tokens = {
            use crate::model::audio_decoder::{detok_embed_codes, upsample};
            let emb = detok_embed_codes(cpu_weights, codes);
            upsample(&emb, hs, N_FRAMES)
        };
        self.ctx
            .queue
            .write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(&tokens));

        let n_past = self.n_past.load(Ordering::Relaxed);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let seq_len = (n_past + N_FRAMES).min(self.cfg.swa_window_size) as u32;

        for (il, lw) in self.layers.iter().enumerate() {
            // Phase 1: (add prev-layer residual then) rmsnorm with operator_norm.
            if il == 0 {
                self.rmsnorm_batch(
                    enc,
                    &self.hidden_buf,
                    &self.normed_buf,
                    &lw.operator_norm,
                    n,
                );
            } else {
                self.add_rmsnorm_batch(
                    enc,
                    &self.hidden_buf,
                    &self.normed_buf,
                    &lw.operator_norm,
                    &self.up_buf,
                    n,
                );
            }

            if self.cfg.layer_is_conv[il] {
                let cin = lw.conv_in_proj.as_ref().unwrap();
                let cop = lw.conv_out_proj.as_ref().unwrap();
                let cw = lw.conv_weight.as_ref().unwrap();
                let rbuf = self.conv_bufs[il].as_ref().unwrap();

                // in_proj: normed → proj [n x 3hs].
                self.gemm(
                    enc,
                    cin,
                    &self.normed_buf,
                    &self.proj_buf,
                    n,
                    hs_u,
                    3 * hs_u,
                );
                // fused conv1d (b*x, depthwise conv, c-gate) → normed [n x hs].
                let cp = self.params(
                    &[
                        hs_u,
                        (self.cfg.d_conv + 1) as u32,
                        self.cfg.d_conv as u32,
                        n,
                        3 * hs_u,
                        hs_u,
                    ],
                    "audio_detok_conv_params",
                );
                self.encode(
                    enc,
                    &self.pipes.conv1d_fused_batch,
                    &[&self.proj_buf, rbuf, cw, &self.normed_buf, &cp],
                    (hs_u.div_ceil(256), 1, 1),
                    "audio_detok_conv1d",
                );
                // out_proj: normed → gate (op residual scratch) [n x hs].
                self.gemm(enc, cop, &self.normed_buf, &self.gate_buf, n, hs_u, hs_u);
            } else {
                let wq = lw.wq.as_ref().unwrap();
                let wk = lw.wk.as_ref().unwrap();
                let wv = lw.wv.as_ref().unwrap();
                let wo = lw.wo.as_ref().unwrap();
                let qn = lw.q_norm.as_ref().unwrap();
                let kn = lw.k_norm.as_ref().unwrap();
                let k_cache = self.kv_k[il].as_ref().unwrap();
                let v_cache = self.kv_v[il].as_ref().unwrap();

                // Q → proj, K → gate, V → up.
                self.gemm(enc, wq, &self.normed_buf, &self.proj_buf, n, hs_u, q_dim);
                self.gemm(enc, wk, &self.normed_buf, &self.gate_buf, n, hs_u, kv_dim);
                self.gemm(enc, wv, &self.normed_buf, &self.up_buf, n, hs_u, kv_dim);

                // Per-head QK-norm + NeoX RoPE on Q (proj) and K (gate).
                let p = self.params(
                    &[
                        n_past as u32,
                        n,
                        n_head,
                        n_kv,
                        hd as u32,
                        self.cfg.rms_norm_eps.to_bits(),
                        self.cfg.rope_freq_base.to_bits(),
                        0, // rope_type = NeoX
                        q_dim,
                        kv_dim,
                        0, // has_freq_factors
                        1, // has_qk_norm
                    ],
                    "audio_detok_rope_params",
                );
                let tg = n * (n_head + n_kv);
                self.encode(
                    enc,
                    &self.pipes.qk_norm_rope_batch,
                    &[
                        &self.proj_buf,
                        &self.gate_buf,
                        qn,
                        kn,
                        &p,
                        &self.rope_freqs_dummy,
                    ],
                    (tg, 1, 1),
                    "audio_detok_qk_rope",
                );

                // Write this frame's K/V into the ring cache. swa (30) is a
                // multiple of N_FRAMES (6), so the 6-slot write never straddles
                // the wrap; one contiguous copy suffices.
                let ring = (n_past % self.cfg.swa_window_size) as u64;
                let f32_bytes = std::mem::size_of::<f32>() as u64;
                let dst_off = ring * kv_dim as u64 * f32_bytes;
                let chunk = (N_FRAMES as u64) * kv_dim as u64 * f32_bytes;
                enc.copy_buffer_to_buffer(&self.gate_buf, 0, k_cache, dst_off, chunk);
                enc.copy_buffer_to_buffer(&self.up_buf, 0, v_cache, dst_off, chunk);

                // Non-causal attention: each token's query attends the whole live
                // window. `flash_attention` reads the cache linearly; the live set
                // is order-independent under softmax, so the ring order is fine.
                let attn_params = self.params(
                    &[
                        n_head,
                        n_kv,
                        hd as u32,
                        kv_dim,
                        seq_len,
                        scale.to_bits(),
                        0,
                        0,
                    ],
                    "audio_detok_attn_params",
                );
                for t in 0..N_FRAMES {
                    let q_off = (t * q_dim as usize) as u64 * 4;
                    enc.copy_buffer_to_buffer(
                        &self.proj_buf,
                        q_off,
                        &self.q_single,
                        0,
                        q_dim as u64 * 4,
                    );
                    self.encode(
                        enc,
                        &self.pipes.flash_attention,
                        &[
                            &self.q_single,
                            k_cache,
                            v_cache,
                            &self.attn_single,
                            &attn_params,
                        ],
                        (n_head, 1, 1),
                        "audio_detok_flash",
                    );
                    enc.copy_buffer_to_buffer(
                        &self.attn_single,
                        0,
                        &self.normed_buf,
                        q_off,
                        q_dim as u64 * 4,
                    );
                }

                // out_proj: attn out (normed, stride q_dim) → gate (op residual).
                self.gemm(enc, wo, &self.normed_buf, &self.gate_buf, n, q_dim, hs_u);
            }

            // FFN: fused (hidden += op residual) + ffn_norm → normed.
            self.add_rmsnorm_batch(
                enc,
                &self.hidden_buf,
                &self.normed_buf,
                &lw.ffn_norm,
                &self.gate_buf,
                n,
            );
            self.gemm(
                enc,
                &lw.ffn_w1,
                &self.normed_buf,
                &self.gate_buf,
                n,
                hs_u,
                ffn,
            );
            self.gemm(
                enc,
                &lw.ffn_w3,
                &self.normed_buf,
                &self.up_buf,
                n,
                hs_u,
                ffn,
            );
            self.silu_mul(enc, &self.gate_buf, &self.up_buf, n * ffn);
            // down → up (next layer's residual scratch).
            self.gemm(enc, &lw.ffn_w2, &self.gate_buf, &self.up_buf, n, ffn, hs_u);
        }

        // Final residual add (last layer's FFN down lives in up_buf).
        self.add_inplace(enc, &self.hidden_buf, &self.up_buf, n * hs_u);

        // Output norm + linear head + bias per frame.
        self.rmsnorm_batch(
            enc,
            &self.hidden_buf,
            &self.normed_buf,
            &self.output_norm,
            n,
        );
        self.gemm(
            enc,
            &self.lin_w,
            &self.normed_buf,
            &self.spectrum_buf,
            n,
            hs_u,
            spec_per_frame as u32,
        );
        let bias_params = self.params(
            &[(N_FRAMES * spec_per_frame) as u32, spec_per_frame as u32],
            "audio_detok_bias_params",
        );
        self.encode(
            enc,
            &self.pipes.bias_add,
            &[&self.spectrum_buf, &self.lin_b, &bias_params],
            (((N_FRAMES * spec_per_frame) as u32).div_ceil(256), 1, 1),
            "audio_detok_bias",
        );

        (N_FRAMES, spec_per_frame, n_past + N_FRAMES)
    }

    pub fn detokenize_to_spectrum(
        &self,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Vec<f32> {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let (n_frames, spec_per_frame, next_n_past) =
            self.encode_detokenize_to_spectrum(&mut enc, cpu_weights, codes);
        self.ctx.submit_encoder(enc);
        let out = self
            .ctx
            .download_f32(&self.spectrum_buf, n_frames * spec_per_frame);
        if !self.ctx.has_readback_fault() {
            self.n_past.store(next_n_past, Ordering::Relaxed);
        }
        out
    }

    pub async fn detokenize_to_spectrum_async(
        &self,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Result<Vec<f32>> {
        detokenize_to_spectrum_with(
            &self.n_past,
            |stage| self.ctx.ensure_alive(stage),
            || {
                let mut enc = self.ctx.device.create_command_encoder(&Default::default());
                let (n_frames, spec_per_frame, next_n_past) =
                    self.encode_detokenize_to_spectrum(&mut enc, cpu_weights, codes);
                let expected_floats = n_frames * spec_per_frame;
                let size = (expected_floats * std::mem::size_of::<f32>()) as u64;
                let pending = self
                    .ctx
                    .begin_download_with_encoder(enc, &self.spectrum_buf, size);
                (expected_floats, next_n_past, pending.recv())
            },
        )
        .await
    }

    fn encode_istft_to_pcm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        spectrum: &[f32],
        n_fft: usize,
        hop_length: usize,
    ) -> Option<(Buffer, usize)> {
        let bins = n_fft / 2 + 1;
        let frame_size = bins * 2;
        let n_frames = spectrum.len() / frame_size;
        if n_frames == 0 {
            return None;
        }

        let spec_buf = self.ctx.upload_f32(spectrum, "audio_istft_spectrum_in");
        let n_spec_floats = n_frames * frame_size;
        let complex_buf = self.ctx.create_storage_rw(
            (n_spec_floats * std::mem::size_of::<f32>()) as u64,
            "audio_istft_complex_half_spectrum",
        );
        let ep_params = self.params(
            &[n_frames as u32, bins as u32],
            "audio_istft_exp_polar_params",
        );
        self.encode(
            enc,
            &self.pipes.exp_polar,
            &[&spec_buf, &complex_buf, &ep_params],
            (((n_frames * bins) as u32).div_ceil(256), 1, 1),
            "audio_istft_exp_polar",
        );

        let windowed_buf = self.ctx.create_storage_rw(
            (n_frames * n_fft * std::mem::size_of::<f32>()) as u64,
            "audio_istft_windowed_frames",
        );
        self.gemm(
            enc,
            &self.idft_basis,
            &complex_buf,
            &windowed_buf,
            n_frames as u32,
            frame_size as u32,
            n_fft as u32,
        );

        let total_samples = n_frames * hop_length;
        let pcm_buf = self.ctx.create_storage_rw(
            (total_samples * std::mem::size_of::<f32>()) as u64,
            "audio_istft_pcm_out",
        );
        let oa_params = self.params(
            &[n_frames as u32, n_fft as u32, hop_length as u32, 0u32],
            "audio_istft_overlap_add_params",
        );
        self.encode(
            enc,
            &self.pipes.overlap_add,
            &[&windowed_buf, &self.hann, &pcm_buf, &oa_params],
            ((total_samples as u32).div_ceil(256), 1, 1),
            "audio_istft_overlap_add",
        );

        Some((pcm_buf, total_samples))
    }

    /// Convert the accumulated spectrum to PCM on the GPU: `exp_polar` →
    /// iDFT-matmul → windowed `overlap_add`. Numerically mirrors the CPU
    /// `istft_to_pcm` (the iDFT basis folds the Hermitian mirror and the
    /// `1/n_fft` scale), down to the startup-pad strip, which stays on the CPU
    /// after readback. Runs once at end-of-generation, so the frame-sized
    /// scratch is allocated per call rather than persisted.
    pub fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        // `idft_basis` and `hann` are built once from `self.cfg`, but every
        // buffer below is sized from these arguments, so a mismatch does not
        // compute a different transform: it indexes a basis that is the wrong
        // shape for the buffers around it. This was a `debug_assert`, which is
        // compiled out of exactly the builds that run this. The CPU
        // implementation takes both as parameters and is tied to neither, so
        // hand the call over rather than refuse it.
        if n_fft != self.cfg.n_fft || hop_length != self.cfg.hop_length {
            return crate::model::audio_decoder::istft_to_pcm(spectrum, n_fft, hop_length);
        }
        let bins = n_fft / 2 + 1;
        let frame_size = bins * 2;
        let n_frames = spectrum.len() / frame_size;
        if n_frames == 0 {
            return vec![];
        }

        // A single 1D dispatch caps the frame count at the device's
        // `max_compute_workgroups_per_dimension` (65535 by default); the CPU
        // `istft_to_pcm` has no such limit, so fall back to it for very long
        // audio rather than truncating.
        if n_frames > (u16::MAX as usize) {
            return crate::model::audio_decoder::istft_to_pcm(spectrum, n_fft, hop_length);
        }

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let (pcm_buf, total_samples) =
            match self.encode_istft_to_pcm(&mut enc, spectrum, n_fft, hop_length) {
                Some(res) => res,
                None => return vec![],
            };

        self.ctx.submit_encoder(enc);

        let mut pcm = self.ctx.download_f32(&pcm_buf, total_samples);
        let padding = (n_fft - hop_length) / 2;
        if pcm.len() > padding {
            pcm.drain(..padding);
        }
        pcm
    }

    pub async fn istft_to_pcm_async(
        &self,
        spectrum: &[f32],
        n_fft: usize,
        hop_length: usize,
    ) -> Result<Vec<f32>> {
        if n_fft != self.cfg.n_fft || hop_length != self.cfg.hop_length {
            return Ok(crate::model::audio_decoder::istft_to_pcm(
                spectrum, n_fft, hop_length,
            ));
        }
        let bins = n_fft / 2 + 1;
        let frame_size = bins * 2;
        let n_frames = spectrum.len() / frame_size;
        if n_frames == 0 {
            return Ok(vec![]);
        }

        if n_frames > (u16::MAX as usize) {
            return Ok(crate::model::audio_decoder::istft_to_pcm(
                spectrum, n_fft, hop_length,
            ));
        }

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let (pcm_buf, total_samples) =
            match self.encode_istft_to_pcm(&mut enc, spectrum, n_fft, hop_length) {
                Some(res) => res,
                None => return Ok(vec![]),
            };

        let pending = self.ctx.begin_download_with_encoder(
            enc,
            &pcm_buf,
            (total_samples * std::mem::size_of::<f32>()) as u64,
        );
        let bytes = pending.recv().await?;
        let expected_bytes = total_samples * std::mem::size_of::<f32>();
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU audio readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        let mut pcm = vec![0f32; total_samples];
        bytemuck::cast_slice_mut(&mut pcm).copy_from_slice(&bytes[..expected_bytes]);
        let padding = (n_fft - hop_length) / 2;
        if pcm.len() > padding {
            pcm.drain(..padding);
        }
        Ok(pcm)
    }

    /// Whether this GPU audio decoder has an active GPU depthformer.
    pub fn supports_depthformer(&self) -> bool {
        self.depthformer.is_some()
    }
}

/// Spectrum floats per frame: `(n_fft/2 + 1) * 2` (log-magnitude, angle).
fn n_embd_bins(cfg: &DetokenizerConfig) -> usize {
    (cfg.n_fft / 2 + 1) * 2
}

impl crate::model::audio_decoder::AudioAccelerator for WgpuAudioDecoder {
    fn supports_depthformer(&self) -> bool {
        self.depthformer.is_some()
    }

    fn take_audio_error(&self) -> Option<crate::CeraError> {
        // Drain every context this decoder owns (detokenizer plus the
        // depthformer's private context) and the sync-sample slot. All
        // three are cleared; the first `Some` in this fixed order wins.
        // Callers discard before each bare call, so a take here holds at
        // most the current attempt's fault.
        let a = self.ctx.take_readback_fault();
        let b = self
            .depthformer
            .as_ref()
            .and_then(|df| df.ctx.take_readback_fault());
        let c = crate::model::take_fault(&self.sample_error);
        a.or(b).or(c)
    }

    fn sample_audio_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8] {
        if let Some(ref _df) = self.depthformer {
            #[cfg(not(target_arch = "wasm32"))]
            {
                pollster::block_on(_df.sample_frame_async(embedding, temperature, top_k))
                    .unwrap_or_else(|e| {
                        tracing::error!(
                            "[cera::wgpu_audio_decoder] sample_frame_async failed: {e:#}"
                        );
                        eprintln!("[cera::wgpu_audio_decoder] sample_frame_async failed: {e:#}");
                        crate::model::record_first_fault(
                            &self.sample_error,
                            crate::CeraError::Backend(format!("sample_frame_async: {e:#}")),
                        );
                        [0; 8]
                    })
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = (embedding, temperature, top_k);
                [0; 8]
            }
        } else {
            [0; 8]
        }
    }

    fn sample_audio_frame_async<'a>(
        &'a self,
        embedding: &'a [f32],
        temperature: f32,
        top_k: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<[i32; 8]>> + Send + 'a>> {
        if let Some(ref df) = self.depthformer {
            Box::pin(df.sample_frame_async(embedding, temperature, top_k))
        } else {
            Box::pin(async move {
                anyhow::bail!("WGPU depthformer not available");
            })
        }
    }

    #[cfg(feature = "gpu")]
    fn sample_audio_frame_from_gpu_hidden_async<'a>(
        &'a self,
        hidden_buf: &'a wgpu::Buffer,
        temperature: f32,
        top_k: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<[i32; 8]>> + Send + 'a>> {
        if let Some(ref df) = self.depthformer {
            Box::pin(df.sample_frame_from_gpu_hidden_async(hidden_buf, temperature, top_k))
        } else {
            Box::pin(async move {
                anyhow::bail!("WGPU depthformer not available");
            })
        }
    }

    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
        self.detokenize_to_spectrum(cpu_weights, codes)
    }

    fn detokenize_to_spectrum_async<'a>(
        &'a self,
        cpu_weights: &'a DetokenizerWeights,
        codes: &'a [i32],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(self.detokenize_to_spectrum_async(cpu_weights, codes))
    }

    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        self.istft_to_pcm(spectrum, n_fft, hop_length)
    }

    fn istft_to_pcm_async<'a>(
        &'a self,
        spectrum: &'a [f32],
        n_fft: usize,
        hop_length: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(self.istft_to_pcm_async(spectrum, n_fft, hop_length))
    }

    fn reset_depthformer(&self) {
        if let Some(ref df) = self.depthformer {
            df.reset();
        }
    }

    fn seed_rng(&self, seed: u64) {
        if let Some(ref df) = self.depthformer {
            df.seed_rng(seed);
        }
    }

    fn reset_detokenizer(&self) {
        self.reset();
    }

    fn try_acquire_session(&self) -> bool {
        self.session_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    fn release_session(&self) {
        self.session_active.store(false, Ordering::Release);
    }
}

async fn detokenize_to_spectrum_with<Fut>(
    n_past: &AtomicUsize,
    ensure_alive: impl FnOnce(&str) -> Result<()>,
    prepare: impl FnOnce() -> (usize, usize, Fut),
) -> Result<Vec<f32>>
where
    Fut: std::future::Future<Output = Result<Vec<u8>>>,
{
    ensure_alive("detokenize_to_spectrum_async")?;
    let (expected_floats, next_n_past, readback) = prepare();
    let bytes = readback.await?;
    let expected_bytes = expected_floats * std::mem::size_of::<f32>();
    anyhow::ensure!(
        bytes.len() >= expected_bytes,
        "GPU spectrum staging readback buffer truncated (expected {expected_bytes} bytes, got {})",
        bytes.len()
    );
    let mut result = vec![0f32; expected_floats];
    bytemuck::cast_slice_mut(&mut result).copy_from_slice(&bytes[..expected_bytes]);
    n_past.store(next_n_past, Ordering::Relaxed);
    Ok(result)
}

async fn finish_depthformer_sample_with<Fut>(
    ensure_alive: impl FnOnce(&str) -> Result<()>,
    dispatch_and_readback: impl FnOnce() -> Fut,
) -> Result<[i32; 8]>
where
    Fut: std::future::Future<Output = Result<Vec<u8>>>,
{
    ensure_alive("sample_frame_internal_async")?;
    let bytes = dispatch_and_readback().await?;
    let expected_bytes = 8 * std::mem::size_of::<u32>();
    anyhow::ensure!(
        bytes.len() >= expected_bytes,
        "GPU staging readback buffer truncated (expected {expected_bytes} bytes, got {})",
        bytes.len()
    );
    let mut codes = [0i32; 8];
    for i in 0..8 {
        codes[i] = bytemuck::pod_read_unaligned::<u32>(&bytes[i * 4..(i + 1) * 4]) as i32;
    }
    Ok(codes)
}

// ── WgpuDepthformer ────────────────────────────────────────────────────────
//
// F32 dequantized weights + F32 KV cache. Runs all 8 codebook steps of an audio
// frame on WebGPU compute shaders, reading back only the 2049 logits per step.

#[allow(dead_code)]
pub struct WgpuDepthformer {
    ctx: GpuContext,
    df_cfg: DepthformerConfig,
    dec_cfg: DecoderConfig,
    pipes: Arc<Pipelines>,
    df_pipes: DepthformerPipelines,
    layers: Vec<DfLayerGpu>,
    // Pre-baked bind groups:
    depth_linear_bgs: Vec<wgpu::BindGroup>,   // 8 entries
    bias_add_bgs: Vec<wgpu::BindGroup>,       // 8 entries
    embed_add_bgs: Vec<wgpu::BindGroup>,      // 7 entries (j=0..6 for prev codebooks 0..6)
    to_logits_norm_bgs: Vec<wgpu::BindGroup>, // 8 entries
    to_logits_gemm_bgs: Vec<wgpu::BindGroup>, // 8 entries
    argmax_bgs: Vec<wgpu::BindGroup>,         // 8 entries
    sample_bgs: Vec<wgpu::BindGroup>,         // 8 entries
    sample_params_bufs: Vec<Buffer>,          // 8 entries
    // Grid dispatch dimensions:
    depth_linear_wg_m: u32,
    dl_cols: usize,
    qkv_wg_m: u32,
    n_embd_wg_m: u32,
    ffn_dim_wg_m: u32,
    n_vocab_wg_m: u32,
    // Storage buffers:
    embedding_in_buf: Buffer,
    hidden_buf: Buffer,
    normed_buf: Buffer,
    q_buf: Buffer,
    k_buf: Buffer,
    v_buf: Buffer,
    attn_out_buf: Buffer,
    ffn_act_buf: Buffer,
    sampled_codes_buf: Buffer,
    logits_buf: Buffer,
    kv_k: Vec<Buffer>,
    kv_v: Vec<Buffer>,
    rng_state: std::sync::atomic::AtomicU64,
}

struct PreparedDepthformerGguf {
    df_cfg: DepthformerConfig,
    dec_cfg: DecoderConfig,
    n_embd_d: usize,
    dl_cols: usize,
    dl_f32: Vec<f32>,
    depth_linear_b: Vec<f32>,
}

/// Validate all depthformer GGUF metadata, dimensions, and per-layer/codebook shapes upfront
/// before dequantizing (`try_to_f32_vec()`) or allocating GPU buffers, so malformed files
/// fail with `Result::Err` cleanly on both host and GPU.
fn prepare_depthformer_gguf(
    gguf: &GgufFile,
    head_dim_bounds: crate::model::audio_decoder::HeadDimBounds,
) -> Result<PreparedDepthformerGguf> {
    use crate::model::audio_decoder::{
        checked_tensor_numel, tensor_2d_cols_rows, validate_depthformer_config_geometry,
        validate_depthformer_dims, validate_depthformer_gguf_tensors,
    };

    let n_layer = gguf
        .get_u32("depthformer_n_layer")
        .context("missing depthformer_n_layer")? as usize;
    anyhow::ensure!(
        (1..=256).contains(&n_layer),
        "depthformer_n_layer ({n_layer}) must be in 1..=256"
    );
    let n_embd = gguf
        .get_u32("depthformer_n_embd")
        .context("missing depthformer_n_embd")? as usize;

    let qkv_info = gguf
        .tensors
        .get("depthformer.layers.0.operator.qkv_proj.weight")
        .context("tensor not found: depthformer.layers.0.operator.qkv_proj.weight")?;
    let (qkv_cols, qkv_out) = tensor_2d_cols_rows(
        &qkv_info.shape,
        "depthformer.layers.0.operator.qkv_proj.weight",
    )?;
    anyhow::ensure!(
        n_embd > 0 && qkv_cols == n_embd,
        "depthformer_n_embd ({n_embd}) disagrees with qkv_proj ({qkv_cols} columns)"
    );
    let q_norm_info = gguf
        .tensors
        .get("depthformer.layers.0.operator.attention.q_layernorm.weight")
        .context("tensor not found: depthformer.layers.0.operator.attention.q_layernorm.weight")?;
    let n_embd_head = q_norm_info
        .shape
        .first()
        .copied()
        .filter(|&d| d > 0)
        .context("depthformer q_layernorm has no usable shape")?;
    let n_head = 32usize;
    let n_head_kv = 8usize;
    let expected_qkv_out = (n_head + 2 * n_head_kv)
        .checked_mul(n_embd_head)
        .context("qkv_proj expected rows overflow")?;
    anyhow::ensure!(qkv_out == expected_qkv_out, "qkv_proj shape mismatch");

    let w1_info = gguf
        .tensors
        .get("depthformer.layers.0.feed_forward.w1.weight")
        .context("tensor not found: depthformer.layers.0.feed_forward.w1.weight")?;
    let (w1_cols, ffn_dim) = tensor_2d_cols_rows(
        &w1_info.shape,
        "depthformer.layers.0.feed_forward.w1.weight",
    )?;
    anyhow::ensure!(
        ffn_dim > 0 && w1_cols == n_embd,
        "depthformer ffn_dim ({ffn_dim}) / w1 cols ({w1_cols}) invalid for depthformer_n_embd ({n_embd})"
    );

    let df_cfg = DepthformerConfig {
        n_layer,
        n_embd,
        n_head,
        n_head_kv,
        n_embd_head,
        ffn_dim,
        rms_norm_eps: 1e-5,
        rope_freq_base: 1_000_000.0,
        max_seq_len: 8,
    };

    let n_codebook = 8;
    let dl_info = gguf
        .tensors
        .get("depth_linear.weight")
        .context("tensor not found: depth_linear.weight")?;
    let dl_bias_info = gguf
        .tensors
        .get("depth_linear.bias")
        .context("tensor not found: depth_linear.bias")?;
    let to_logits_0_info = gguf
        .tensors
        .get("depth_embeddings.0.to_logits.weight")
        .context("tensor not found: depth_embeddings.0.to_logits.weight")?;
    let (_, n_vocab) = tensor_2d_cols_rows(
        &to_logits_0_info.shape,
        "depth_embeddings.0.to_logits.weight",
    )?;

    let (n_embd_d, dl_cols, _) = validate_depthformer_dims(
        n_embd,
        n_embd_head,
        ffn_dim,
        n_codebook,
        n_vocab,
        &dl_info.shape,
        checked_tensor_numel(&dl_info.shape, "depth_linear.weight")?,
        checked_tensor_numel(&dl_bias_info.shape, "depth_linear.bias")?,
        head_dim_bounds,
    )?;

    let dec_cfg = DecoderConfig {
        n_codebook,
        n_vocab,
        n_embd: dl_cols,
        rms_norm_eps: 1e-5,
    };
    validate_depthformer_config_geometry(&df_cfg, &dec_cfg, head_dim_bounds)?;
    validate_depthformer_gguf_tensors(gguf, &df_cfg, &dec_cfg, false)?;

    let dl_f32 = gguf.get_tensor("depth_linear.weight")?.try_to_f32_vec()?;
    let depth_linear_b = gguf.get_tensor("depth_linear.bias")?.try_to_f32_vec()?;
    Ok(PreparedDepthformerGguf {
        df_cfg,
        dec_cfg,
        n_embd_d,
        dl_cols,
        dl_f32,
        depth_linear_b,
    })
}

impl WgpuDepthformer {
    pub(crate) fn try_from_gguf(
        ctx: &GpuContext,
        pipes: &Arc<Pipelines>,
        gguf: &Arc<GgufFile>,
    ) -> Result<Self> {
        let prepared =
            prepare_depthformer_gguf(gguf, crate::model::audio_decoder::HeadDimBounds::WGPU)?;
        Self::from_configs(ctx, pipes, gguf, prepared)
    }

    fn from_configs(
        ctx: &GpuContext,
        pipes: &Arc<Pipelines>,
        gguf: &Arc<GgufFile>,
        prepared: PreparedDepthformerGguf,
    ) -> Result<Self> {
        let df_pipes = pipes
            .df
            .as_ref()
            .context("audio_depthformer pipelines unavailable")?
            .clone();
        let PreparedDepthformerGguf {
            df_cfg,
            dec_cfg,
            n_embd_d,
            dl_cols,
            dl_f32,
            depth_linear_b,
        } = prepared;
        crate::model::audio_decoder::validate_depthformer_config_geometry(
            &df_cfg,
            &dec_cfg,
            crate::model::audio_decoder::HeadDimBounds::WGPU,
        )?;
        let n_embd = df_cfg.n_embd;
        let hd = df_cfg.n_embd_head;
        let n_head = df_cfg.n_head;
        let n_kv = df_cfg.n_head_kv;
        let q_dim = n_head * hd;
        let kv_dim = n_kv * hd;
        let ffn_dim = df_cfg.ffn_dim;

        let upload_vec = |name: &str| -> Result<Buffer> {
            Ok(ctx.upload_f32(&gguf.get_tensor(name)?.try_to_f32_vec()?, name))
        };

        let alloc = |n: usize, label: &str| ctx.create_storage_rw((n * 4) as u64, label);
        let hidden_buf = alloc(n_embd, "df_hidden");
        let normed_buf = alloc(n_embd, "df_normed");
        let q_buf = alloc(q_dim, "df_q");
        let k_buf = alloc(kv_dim, "df_k");
        let v_buf = alloc(kv_dim, "df_v");
        let attn_out_buf = alloc(q_dim, "df_attn_out");
        let ffn_act_buf = alloc(ffn_dim, "df_ffn_act");

        let embedding_in_buf = alloc(dl_cols, "df_embedding_in");
        let logits_buf = alloc(dec_cfg.n_vocab, "df_logits");
        let sampled_codes_buf = alloc(dec_cfg.n_codebook, "df_sampled_codes");
        let rope_freqs_dummy = ctx.upload_f32(&[1.0f32], "df_rope_dummy");

        let mut kv_k = Vec::with_capacity(df_cfg.n_layer);
        let mut kv_v = Vec::with_capacity(df_cfg.n_layer);
        for i in 0..df_cfg.n_layer {
            kv_k.push(alloc(df_cfg.max_seq_len * kv_dim, &format!("df_kv_k_{i}")));
            kv_v.push(alloc(df_cfg.max_seq_len * kv_dim, &format!("df_kv_v_{i}")));
        }

        let make_bg =
            |pipeline: &wgpu::ComputePipeline, bufs: &[&Buffer], label: &str| -> wgpu::BindGroup {
                let entries: Vec<wgpu::BindGroupEntry> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect();
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                })
            };

        // Shared param buffers
        let hs = n_embd as u32;
        let rmsnorm_params_1 = ctx.upload_storage(
            bytemuck::cast_slice(&[hs, df_cfg.rms_norm_eps.to_bits(), hs, hs, 1.0f32.to_bits()]),
            "df_rmsnorm_params_1",
        );
        let add_params_n_embd = ctx.upload_storage(
            bytemuck::cast_slice(&[n_embd as u32, 0u32]),
            "df_add_params_n_embd",
        );

        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut rope_params = Vec::with_capacity(df_cfg.max_seq_len);
        let mut attn_params = Vec::with_capacity(df_cfg.max_seq_len);
        let mut kv_copy_params = Vec::with_capacity(df_cfg.max_seq_len);
        for pos in 0..df_cfg.max_seq_len {
            rope_params.push(ctx.upload_storage(
                bytemuck::cast_slice(&[
                    pos as u32,
                    1,
                    n_head as u32,
                    n_kv as u32,
                    hd as u32,
                    df_cfg.rms_norm_eps.to_bits(),
                    df_cfg.rope_freq_base.to_bits(),
                    1, // rope_type = interleaved / NORM
                    q_dim as u32,
                    kv_dim as u32,
                    0,
                    1,
                ]),
                &format!("df_rope_params_{pos}"),
            ));
            attn_params.push(ctx.upload_storage(
                bytemuck::cast_slice(&[
                    n_head as u32,
                    n_kv as u32,
                    hd as u32,
                    kv_dim as u32,
                    (pos + 1) as u32,
                    scale.to_bits(),
                    0,
                    0,
                ]),
                &format!("df_attn_params_{pos}"),
            ));
            kv_copy_params.push(ctx.upload_storage(
                bytemuck::cast_slice(&[(pos * kv_dim) as u32, kv_dim as u32]),
                &format!("df_kv_copy_params_{pos}"),
            ));
        }

        // Layer GEMV params: vec4<u32>(m, k, row_base, 0)
        let qkv_params = ctx.upload_storage(
            bytemuck::cast_slice(&[q_dim as u32, kv_dim as u32, n_embd as u32, 0u32]),
            "df_qkv_params",
        );
        let wo_params = ctx.upload_storage(
            bytemuck::cast_slice(&[n_embd as u32, q_dim as u32, 0u32, 0u32]),
            "df_wo_params",
        );
        let w13_params = ctx.upload_storage(
            bytemuck::cast_slice(&[ffn_dim as u32, n_embd as u32, 0u32, 0u32]),
            "df_w13_params",
        );
        let w2_params = ctx.upload_storage(
            bytemuck::cast_slice(&[n_embd as u32, ffn_dim as u32, 0u32, 0u32]),
            "df_w2_params",
        );

        let mut layers = Vec::with_capacity(df_cfg.n_layer);
        for i in 0..df_cfg.n_layer {
            let pfx = format!("depthformer.layers.{i}");
            let qkv_t = gguf.get_tensor(&format!("{pfx}.operator.qkv_proj.weight"))?;
            let qkv_f32 = qkv_t.try_to_f32_vec()?;
            let qkv_buf = ctx.upload_f32(&qkv_f32, &format!("{pfx}.qkv"));

            let op_norm = upload_vec(&format!("{pfx}.operator_norm.weight"))?;
            let q_norm = upload_vec(&format!("{pfx}.operator.attention.q_layernorm.weight"))?;
            let k_norm = upload_vec(&format!("{pfx}.operator.attention.k_layernorm.weight"))?;
            let wo_buf = ctx.upload_f32(
                &gguf
                    .get_tensor(&format!("{pfx}.operator.out_proj.weight"))?
                    .try_to_f32_vec()?,
                &format!("{pfx}.wo"),
            );
            let ffn_norm = upload_vec(&format!("{pfx}.ffn_norm.weight"))?;

            let w1_vec = gguf
                .get_tensor(&format!("{pfx}.feed_forward.w1.weight"))?
                .try_to_f32_vec()?;
            let w3_vec = gguf
                .get_tensor(&format!("{pfx}.feed_forward.w3.weight"))?
                .try_to_f32_vec()?;
            let mut w13_vec = Vec::with_capacity(w1_vec.len() + w3_vec.len());
            w13_vec.extend_from_slice(&w1_vec);
            w13_vec.extend_from_slice(&w3_vec);
            let w13_buf = ctx.upload_f32(&w13_vec, &format!("{pfx}.w13"));
            let w2_buf = ctx.upload_f32(
                &gguf
                    .get_tensor(&format!("{pfx}.feed_forward.w2.weight"))?
                    .try_to_f32_vec()?,
                &format!("{pfx}.w2"),
            );

            let operator_norm_bg = make_bg(
                &pipes.rmsnorm_batch,
                &[&hidden_buf, &normed_buf, &op_norm, &rmsnorm_params_1],
                &format!("df_op_norm_bg_{i}"),
            );
            let gemm_qkv_bg = make_bg(
                &df_pipes.df_qkv_gemv,
                &[&qkv_buf, &normed_buf, &q_buf, &k_buf, &v_buf, &qkv_params],
                &format!("df_qkv_bg_{i}"),
            );

            let mut rope_bgs = Vec::with_capacity(df_cfg.max_seq_len);
            let mut kv_copy_bgs = Vec::with_capacity(df_cfg.max_seq_len);
            let mut attn_bgs = Vec::with_capacity(df_cfg.max_seq_len);
            for pos in 0..df_cfg.max_seq_len {
                rope_bgs.push(make_bg(
                    &pipes.qk_norm_rope_batch,
                    &[
                        &q_buf,
                        &k_buf,
                        &q_norm,
                        &k_norm,
                        &rope_params[pos],
                        &rope_freqs_dummy,
                    ],
                    &format!("df_rope_bg_{i}_{pos}"),
                ));
                kv_copy_bgs.push(make_bg(
                    &df_pipes.df_kv_copy,
                    &[&k_buf, &v_buf, &kv_k[i], &kv_v[i], &kv_copy_params[pos]],
                    &format!("df_kv_copy_bg_{i}_{pos}"),
                ));
                attn_bgs.push(make_bg(
                    &pipes.flash_attention,
                    &[&q_buf, &kv_k[i], &kv_v[i], &attn_out_buf, &attn_params[pos]],
                    &format!("df_attn_bg_{i}_{pos}"),
                ));
            }

            let gemm_wo_accum_bg = make_bg(
                &df_pipes.gemv_f32_accum,
                &[&wo_buf, &attn_out_buf, &hidden_buf, &wo_params],
                &format!("df_wo_accum_bg_{i}"),
            );
            let ffn_norm_bg = make_bg(
                &pipes.rmsnorm_batch,
                &[&hidden_buf, &normed_buf, &ffn_norm, &rmsnorm_params_1],
                &format!("df_ffn_norm_bg_{i}"),
            );
            let ffn_gate_up_bg = make_bg(
                &df_pipes.df_ffn_gate_up,
                &[&w13_buf, &normed_buf, &ffn_act_buf, &w13_params],
                &format!("df_ffn_gate_up_bg_{i}"),
            );
            let gemm_w2_accum_bg = make_bg(
                &df_pipes.gemv_f32_accum,
                &[&w2_buf, &ffn_act_buf, &hidden_buf, &w2_params],
                &format!("df_w2_accum_bg_{i}"),
            );

            layers.push(DfLayerGpu {
                operator_norm_bg,
                gemm_qkv_bg,
                rope_bgs,
                kv_copy_bgs,
                attn_bgs,
                gemm_wo_accum_bg,
                ffn_norm_bg,
                ffn_gate_up_bg,
                gemm_w2_accum_bg,
            });
        }

        // Slice depth_linear.weight and pre-bake depth_linear GEMM and bias_add
        let dl_params = ctx.upload_storage(
            bytemuck::cast_slice(&[n_embd_d as u32, dl_cols as u32, 0u32, 0u32]),
            "df_dl_params",
        );
        let mut depth_linear_bgs = Vec::with_capacity(dec_cfg.n_codebook);
        let mut bias_add_bgs = Vec::with_capacity(dec_cfg.n_codebook);

        for j in 0..dec_cfg.n_codebook {
            let start = j * n_embd_d * dl_cols;
            let end = start + n_embd_d * dl_cols;
            let slice_data = &dl_f32[start..end];
            let buf = ctx.upload_f32(slice_data, &format!("depth_linear_slice_{j}"));
            depth_linear_bgs.push(make_bg(
                &df_pipes.gemv_f32,
                &[&buf, &embedding_in_buf, &hidden_buf, &dl_params],
                &format!("df_dl_bg_{j}"),
            ));

            let b_start = j * n_embd_d;
            let b_end = b_start + n_embd_d;
            let b_slice = &depth_linear_b[b_start..b_end];
            let b_buf = ctx.upload_f32(b_slice, &format!("depth_linear_b_{j}"));
            bias_add_bgs.push(make_bg(
                &pipes.add_inplace,
                &[&hidden_buf, &b_buf, &add_params_n_embd],
                &format!("df_bias_add_bg_{j}"),
            ));
        }

        let tl_params = ctx.upload_storage(
            bytemuck::cast_slice(&[dec_cfg.n_vocab as u32, n_embd as u32, 0u32, 0u32]),
            "df_tl_params",
        );
        let mut embed_add_bgs = Vec::with_capacity(dec_cfg.n_codebook.saturating_sub(1));
        let mut to_logits_norm_bgs = Vec::with_capacity(dec_cfg.n_codebook);
        let mut to_logits_gemm_bgs = Vec::with_capacity(dec_cfg.n_codebook);
        let mut argmax_bgs = Vec::with_capacity(dec_cfg.n_codebook);

        for j in 0..dec_cfg.n_codebook {
            let pfx = format!("depth_embeddings.{j}");
            let norm_buf = upload_vec(&format!("{pfx}.embedding_norm.weight"))?;
            to_logits_norm_bgs.push(make_bg(
                &pipes.rmsnorm_batch,
                &[&hidden_buf, &normed_buf, &norm_buf, &rmsnorm_params_1],
                &format!("df_cb_norm_bg_{j}"),
            ));

            let tl_t = gguf.get_tensor(&format!("{pfx}.to_logits.weight"))?;
            let tl_f32 = tl_t.try_to_f32_vec()?;
            let tl_buf = ctx.upload_f32(&tl_f32, &format!("{pfx}.to_logits"));
            to_logits_gemm_bgs.push(make_bg(
                &df_pipes.gemv_f32,
                &[&tl_buf, &normed_buf, &logits_buf, &tl_params],
                &format!("df_tl_bg_{j}"),
            ));

            let am_params = ctx.upload_storage(
                bytemuck::cast_slice(&[dec_cfg.n_vocab as u32, j as u32]),
                &format!("df_argmax_params_{j}"),
            );
            argmax_bgs.push(make_bg(
                &df_pipes.argmax_f32,
                &[&logits_buf, &sampled_codes_buf, &am_params],
                &format!("df_argmax_bg_{j}"),
            ));

            if j < dec_cfg.n_codebook - 1 {
                let emb_t = gguf.get_tensor(&format!("{pfx}.embedding.weight"))?;
                let emb_buf = ctx.upload_f32(&emb_t.try_to_f32_vec()?, &format!("{pfx}.emb"));
                let em_params = ctx.upload_storage(
                    bytemuck::cast_slice(&[n_embd as u32, j as u32]),
                    &format!("df_embed_params_{j}"),
                );
                embed_add_bgs.push(make_bg(
                    &df_pipes.df_embed_add,
                    &[&emb_buf, &sampled_codes_buf, &hidden_buf, &em_params],
                    &format!("df_embed_add_bg_{j}"),
                ));
            }
        }

        let mut sample_params_bufs = Vec::with_capacity(dec_cfg.n_codebook);
        let mut sample_bgs = Vec::with_capacity(dec_cfg.n_codebook);
        for j in 0..dec_cfg.n_codebook {
            let sm_params = alloc(4, &format!("df_sample_params_{j}"));
            sample_bgs.push(make_bg(
                &df_pipes.df_sample_logits,
                &[&logits_buf, &sampled_codes_buf, &sm_params],
                &format!("df_sample_bg_{j}"),
            ));
            sample_params_bufs.push(sm_params);
        }

        let depth_linear_wg_m = (n_embd_d as u32).div_ceil(8);
        let qkv_wg_m = ((q_dim + 2 * kv_dim) as u32).div_ceil(8);
        let n_embd_wg_m = (n_embd as u32).div_ceil(8);
        let ffn_dim_wg_m = (ffn_dim as u32).div_ceil(8);
        let n_vocab_wg_m = (dec_cfg.n_vocab as u32).div_ceil(8);

        Ok(Self {
            ctx: ctx.clone(),
            df_cfg: df_cfg.clone(),
            dec_cfg: dec_cfg.clone(),
            pipes: pipes.clone(),
            df_pipes,
            layers,
            depth_linear_bgs,
            bias_add_bgs,
            embed_add_bgs,
            to_logits_norm_bgs,
            to_logits_gemm_bgs,
            argmax_bgs,
            sample_bgs,
            sample_params_bufs,
            depth_linear_wg_m,
            dl_cols,
            qkv_wg_m,
            n_embd_wg_m,
            ffn_dim_wg_m,
            n_vocab_wg_m,
            embedding_in_buf,
            hidden_buf,
            normed_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            ffn_act_buf,
            sampled_codes_buf,
            logits_buf,
            kv_k,
            kv_v,
            rng_state: std::sync::atomic::AtomicU64::new(
                crate::model::audio_decoder::non_zero_rng_seed(rand::random::<u64>()),
            ),
        })
    }

    pub fn seed_rng(&self, seed: u64) {
        self.rng_state.store(
            crate::model::audio_decoder::seed_to_rng_state(seed),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    pub fn reset(&self) {}

    pub async fn sample_frame_async(
        &self,
        embedding: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<[i32; 8]> {
        self.ctx.ensure_alive("sample_frame_async")?;
        let mut padded = vec![0.0f32; self.dl_cols];
        let copy_len = embedding.len().min(self.dl_cols);
        padded[..copy_len].copy_from_slice(&embedding[..copy_len]);
        self.ctx
            .queue
            .write_buffer(&self.embedding_in_buf, 0, bytemuck::cast_slice(&padded));
        self.sample_frame_internal_async(temperature, top_k, None)
            .await
    }

    pub async fn sample_frame_from_gpu_hidden_async(
        &self,
        hidden_buf: &wgpu::Buffer,
        temperature: f32,
        top_k: usize,
    ) -> Result<[i32; 8]> {
        self.sample_frame_internal_async(temperature, top_k, Some(hidden_buf))
            .await
    }

    async fn sample_frame_internal_async(
        &self,
        temperature: f32,
        top_k: usize,
        gpu_hidden_src: Option<&wgpu::Buffer>,
    ) -> Result<[i32; 8]> {
        finish_depthformer_sample_with(
            |stage| self.ctx.ensure_alive(stage),
            || {
                let n_embd = self.df_cfg.n_embd;
                let hd = self.df_cfg.n_embd_head;
                let n_head = self.df_cfg.n_head as u32;
                let n_kv = self.df_cfg.n_head_kv as u32;
                let kv_dim = n_kv * hd as u32;

                let use_sampling = temperature > 0.0 && temperature.is_finite() && top_k > 1;
                let inv_temp = if use_sampling { 1.0 / temperature } else { 1.0 };

                if use_sampling {
                    for j in 0..self.dec_cfg.n_codebook {
                        let rand_val = crate::model::audio_decoder::next_atomic_xorshift64_f32_01(
                            &self.rng_state,
                        );
                        let params = [self.dec_cfg.n_vocab as f32, j as f32, inv_temp, rand_val];
                        self.ctx.queue.write_buffer(
                            &self.sample_params_bufs[j],
                            0,
                            bytemuck::cast_slice(&params),
                        );
                    }
                }

                let mut enc =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("wgpu_depthformer_all_codebooks"),
                        });

                if let Some(src) = gpu_hidden_src {
                    let size = (self.dl_cols * 4) as u64;
                    enc.copy_buffer_to_buffer(src, 0, &self.embedding_in_buf, 0, size);
                }

                let dispatch_pass = |enc: &mut wgpu::CommandEncoder,
                                     pipeline: &wgpu::ComputePipeline,
                                     bg: &wgpu::BindGroup,
                                     workgroups: (u32, u32, u32),
                                     label: &str| {
                    let mut pass = self.ctx.begin_pass(enc, label);
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
                };

                for j in 0..self.dec_cfg.n_codebook {
                    let pos = j;

                    // Initial projections for this codebook
                    dispatch_pass(
                        &mut enc,
                        &self.df_pipes.gemv_f32,
                        &self.depth_linear_bgs[j],
                        (self.depth_linear_wg_m, 1, 1),
                        "df_depth_linear",
                    );

                    dispatch_pass(
                        &mut enc,
                        &self.pipes.add_inplace,
                        &self.bias_add_bgs[j],
                        ((n_embd as u32).div_ceil(256), 1, 1),
                        "df_bias_add",
                    );

                    if j > 0 {
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.df_embed_add,
                            &self.embed_add_bgs[j - 1],
                            ((n_embd as u32).div_ceil(256), 1, 1),
                            "df_embed_add",
                        );
                    }

                    // Transformer layers with fused QKV, fused FFN gate+up with inline SiLU, and direct residual accumulators
                    for lw in &self.layers {
                        // RMSnorm -> normed_buf
                        dispatch_pass(
                            &mut enc,
                            &self.pipes.rmsnorm_batch,
                            &lw.operator_norm_bg,
                            (1, 1, 1),
                            "df_op_norm",
                        );

                        // Fused QKV GEMV -> q_buf, k_buf, v_buf
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.df_qkv_gemv,
                            &lw.gemm_qkv_bg,
                            (self.qkv_wg_m, 1, 1),
                            "df_qkv_gemv",
                        );

                        // QK norm + RoPE (interleaved / NORM)
                        dispatch_pass(
                            &mut enc,
                            &self.pipes.qk_norm_rope_batch,
                            &lw.rope_bgs[pos],
                            (n_head + n_kv, 1, 1),
                            "df_qk_norm_rope",
                        );

                        // GPU-side KV cache slot copy inside the compute pass
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.df_kv_copy,
                            &lw.kv_copy_bgs[pos],
                            (kv_dim.div_ceil(256), 1, 1),
                            "df_kv_copy",
                        );

                        // Flash Attention -> attn_out_buf
                        dispatch_pass(
                            &mut enc,
                            &self.pipes.flash_attention,
                            &lw.attn_bgs[pos],
                            (n_head, 1, 1),
                            "df_flash_attention",
                        );

                        // Out projection directly accumulated into hidden_buf (hidden_buf += Wo · attn_out)
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.gemv_f32_accum,
                            &lw.gemm_wo_accum_bg,
                            (self.n_embd_wg_m, 1, 1),
                            "df_wo_accum",
                        );

                        // FFN: RMSnorm -> normed_buf
                        dispatch_pass(
                            &mut enc,
                            &self.pipes.rmsnorm_batch,
                            &lw.ffn_norm_bg,
                            (1, 1, 1),
                            "df_ffn_norm",
                        );

                        // FFN Fused Gate + Up with inline SiLU -> ffn_act_buf
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.df_ffn_gate_up,
                            &lw.ffn_gate_up_bg,
                            (self.ffn_dim_wg_m, 1, 1),
                            "df_ffn_gate_up",
                        );

                        // FFN W2 Down projection directly accumulated into hidden_buf (hidden_buf += W2 · ffn_act)
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.gemv_f32_accum,
                            &lw.gemm_w2_accum_bg,
                            (self.n_embd_wg_m, 1, 1),
                            "df_w2_accum",
                        );
                    }

                    // to_logits: RMSnorm -> GEMV -> logits_buf
                    dispatch_pass(
                        &mut enc,
                        &self.pipes.rmsnorm_batch,
                        &self.to_logits_norm_bgs[j],
                        (1, 1, 1),
                        "df_to_logits_norm",
                    );

                    dispatch_pass(
                        &mut enc,
                        &self.df_pipes.gemv_f32,
                        &self.to_logits_gemm_bgs[j],
                        (self.n_vocab_wg_m, 1, 1),
                        "df_to_logits_gemm",
                    );

                    // Sample on GPU (with temperature) or Argmax on GPU -> sampled_codes_buf[j]
                    if use_sampling {
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.df_sample_logits,
                            &self.sample_bgs[j],
                            (1, 1, 1),
                            "df_sample_logits",
                        );
                    } else {
                        dispatch_pass(
                            &mut enc,
                            &self.df_pipes.argmax_f32,
                            &self.argmax_bgs[j],
                            (1, 1, 1),
                            "df_argmax",
                        );
                    }
                }

                // Single 32-byte download of all 8 codes through `PendingReadback`
                let size = (self.dec_cfg.n_codebook * 4) as u64;
                self.ctx
                    .begin_download_with_encoder(enc, &self.sampled_codes_buf, size)
                    .recv()
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        detokenize_to_spectrum_with, finish_audio_decoder_load_with,
        finish_depthformer_sample_with, init_audio_decoder_pipelines_with,
    };

    #[test]
    fn init_audio_decoder_pipelines_with_validates_core_and_aligns_df_optionality() {
        // 1. Pre-existing device loss aborts before building core or df pipelines.
        let mut core_calls = 0usize;
        let err_pre = init_audio_decoder_pipelines_with(
            true,
            || Some("pre-lost".to_string()),
            || {
                core_calls += 1;
                Ok(1u32)
            },
            || Ok(Some(2u32)),
        )
        .unwrap_err();
        assert_eq!(core_calls, 0);
        assert!(err_pre.to_string().contains("pre-lost"));

        // 2. Core pipeline error aborts before df validation.
        let mut df_calls = 0usize;
        let err_core = init_audio_decoder_pipelines_with::<u32, u32>(
            true,
            || None,
            || anyhow::bail!("core detok failed"),
            || {
                df_calls += 1;
                Ok(Some(2))
            },
        )
        .unwrap_err();
        assert_eq!(df_calls, 0);
        assert!(err_core.to_string().contains("core detok failed"));

        // 3. When try_df is false, df builder is never invoked and returns None.
        let mut df_calls = 0usize;
        let (core, df) = init_audio_decoder_pipelines_with(
            false,
            || None,
            || Ok(11u32),
            || {
                df_calls += 1;
                Ok(Some(22u32))
            },
        )
        .unwrap();
        assert_eq!((core, df, df_calls), (11, None, 0));

        // 4. When try_df is true and df degrades non-fatally to None, detokenizer succeeds.
        let (core, df) =
            init_audio_decoder_pipelines_with(true, || None, || Ok(11u32), || Ok(None::<u32>))
                .unwrap();
        assert_eq!((core, df), (11, None));

        // 5. When try_df is true and df aborts with device-loss Err, whole load aborts.
        let err_df = init_audio_decoder_pipelines_with::<u32, u32>(
            true,
            || None,
            || Ok(11),
            || anyhow::bail!("df device lost"),
        )
        .unwrap_err();
        assert!(err_df.to_string().contains("df device lost"));
    }

    #[test]
    fn finish_audio_decoder_load_with_aborts_on_post_upload_device_loss() {
        // 1. Device lost after detokenizer upload with has_df_pipes=true skips depthformer upload.
        let mut df_calls = 0usize;
        let err_detok_with_df = finish_audio_decoder_load_with(
            true,
            |_| anyhow::bail!("audio_decoder weight upload: GPU device lost (detok lost)"),
            || true,
            || {
                df_calls += 1;
                Ok(1u32)
            },
        )
        .unwrap_err();
        assert_eq!(df_calls, 0);
        assert!(format!("{err_detok_with_df:#}").contains("detok lost"));

        // 2. Device lost after detokenizer upload with has_df_pipes=false aborts instead of Ok(None).
        let err_detok_no_df = finish_audio_decoder_load_with(
            false,
            |_| anyhow::bail!("audio_decoder weight upload: GPU device lost (detok no-df lost)"),
            || true,
            || Ok(1u32),
        )
        .unwrap_err();
        assert!(format!("{err_detok_no_df:#}").contains("detok no-df lost"));

        // 3. Depthformer upload returns Ok(df), but device died silently mid-upload -> post-upload abort.
        let mut stage_idx = 0usize;
        let err_df_silent = finish_audio_decoder_load_with(
            true,
            |stage| {
                stage_idx += 1;
                if stage_idx == 1 {
                    Ok(())
                } else {
                    anyhow::bail!("{stage}: GPU device lost (df silent loss)")
                }
            },
            || true,
            || Ok(99u32),
        )
        .unwrap_err();
        assert_eq!(stage_idx, 2);
        assert!(format!("{err_df_silent:#}").contains("df silent loss"));

        // 4. Depthformer load returns Err and device is lost -> aborts with Err instead of CPU fallback.
        let err_df_lost = finish_audio_decoder_load_with::<u32>(
            true,
            |_| Ok(()),
            || true,
            || anyhow::bail!("df load error on dead device"),
        )
        .unwrap_err();
        assert!(format!("{err_df_lost:#}").contains("df load error on dead device"));

        // 5. Depthformer load returns Err while device is alive -> falls back to CPU (Ok(None)).
        let fallback = finish_audio_decoder_load_with::<u32>(
            true,
            |_| Ok(()),
            || false,
            || anyhow::bail!("non-fatal df GGUF mismatch"),
        )
        .unwrap();
        assert_eq!(fallback, None);

        // 6. Both uploads succeed on live device -> Ok(Some(df)).
        let ok = finish_audio_decoder_load_with(true, |_| Ok(()), || false, || Ok(42u32)).unwrap();
        assert_eq!(ok, Some(42));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn pipelines_build_validated_succeeds_and_aborts_on_lost_device() {
        use super::Pipelines;
        let Some(ctx) = crate::backend::wgpu::gpu_or_skip() else {
            return;
        };
        let core_only = Pipelines::build_validated(&ctx, false)
            .expect("core audio_decoder pipelines must build on a live device");
        assert!(core_only.df.is_none());

        let with_df = Pipelines::build_validated(&ctx, true)
            .expect("audio_decoder pipelines with depthformer must build on a live device");
        assert!(with_df.df.is_some());

        ctx.simulate_device_lost("test: audio decoder lost");
        let err = match Pipelines::build_validated(&ctx, true) {
            Ok(_) => panic!("Pipelines::build_validated must abort on a lost device"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("audio_decoder pipelines"), "{msg}");
        assert!(msg.contains("test: audio decoder lost"), "{msg}");
    }

    fn noop_waker() -> std::task::Waker {
        use std::task::{RawWaker, RawWakerVTable, Waker};
        unsafe fn clone(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        unsafe fn noop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    fn poll_once<F: std::future::Future>(fut: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
        let waker = noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        fut.poll(&mut cx)
    }

    #[test]
    fn detokenize_and_depthformer_async_readbacks_defer_commit_and_abort_on_loss() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // 1. Pre-flight device loss skips encoding and keeps n_past unchanged.
        let n_past = AtomicUsize::new(0);
        let mut prepared = false;
        let err_pre = {
            let mut fut = std::pin::pin!(detokenize_to_spectrum_with(
                &n_past,
                |stage| anyhow::bail!("{stage}: GPU device lost (pre-lost)"),
                || {
                    prepared = true;
                    (2, 6, async { Ok(vec![0u8; 8]) })
                },
            ));
            match poll_once(fut.as_mut()) {
                std::task::Poll::Ready(Err(e)) => e,
                other => panic!("expected Ready(Err), got {other:?}"),
            }
        };
        assert!(!prepared);
        assert_eq!(n_past.load(Ordering::Relaxed), 0);
        assert!(format!("{err_pre:#}").contains("pre-lost"));

        // 2. Mid-flight readback failure propagates Err and does NOT advance n_past.
        let mut fut = std::pin::pin!(detokenize_to_spectrum_with(
            &n_past,
            |_| Ok(()),
            || (2, 6, async {
                anyhow::bail!("GPU device lost (mid-flight); readback discarded")
            }),
        ));
        let err_mid = match poll_once(fut.as_mut()) {
            std::task::Poll::Ready(Err(e)) => e,
            other => panic!("expected Ready(Err), got {other:?}"),
        };
        assert_eq!(n_past.load(Ordering::Relaxed), 0);
        assert!(format!("{err_mid:#}").contains("mid-flight"));

        // 3. Dropping/cancelling a pending readback future does NOT advance n_past.
        {
            let (_tx, rx) = futures_channel::oneshot::channel::<anyhow::Result<Vec<u8>>>();
            let mut fut = std::pin::pin!(detokenize_to_spectrum_with(
                &n_past,
                |_| Ok(()),
                || (2, 6, async move {
                    rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("closed")))
                }),
            ));
            assert!(matches!(poll_once(fut.as_mut()), std::task::Poll::Pending));
        }
        assert_eq!(n_past.load(Ordering::Relaxed), 0);

        // 4. Truncated spectrum buffer fails without advancing n_past.
        let mut fut = std::pin::pin!(detokenize_to_spectrum_with(
            &n_past,
            |_| Ok(()),
            || (2, 6, async { Ok(vec![0u8; 4]) }),
        ));
        assert!(matches!(
            poll_once(fut.as_mut()),
            std::task::Poll::Ready(Err(_))
        ));
        assert_eq!(n_past.load(Ordering::Relaxed), 0);

        // 5. Valid spectrum readback decodes floats and commits n_past.
        let expected = [1.5f32, -2.25f32];
        let valid_bytes = bytemuck::cast_slice(&expected).to_vec();
        let mut fut = std::pin::pin!(detokenize_to_spectrum_with(
            &n_past,
            |_| Ok(()),
            || (2, 6, async move { Ok(valid_bytes) }),
        ));
        let got = match poll_once(fut.as_mut()) {
            std::task::Poll::Ready(Ok(v)) => v,
            other => panic!("expected Ready(Ok), got {other:?}"),
        };
        assert_eq!(got, expected);
        assert_eq!(n_past.load(Ordering::Relaxed), 6);

        // 6. Depthformer sample helper: pre-flight loss skips dispatch; mid-flight loss &
        //    truncated bytes abort; valid 32-byte readback decodes all 8 codes.
        let mut dispatched = false;
        {
            let mut fut = std::pin::pin!(finish_depthformer_sample_with(
                |stage| anyhow::bail!("{stage}: GPU device lost (df pre-lost)"),
                || {
                    dispatched = true;
                    async { Ok(vec![0u8; 32]) }
                },
            ));
            assert!(matches!(
                poll_once(fut.as_mut()),
                std::task::Poll::Ready(Err(_))
            ));
        }
        assert!(!dispatched);

        let mut fut = std::pin::pin!(finish_depthformer_sample_with(
            |_| Ok(()),
            || async { anyhow::bail!("GPU device lost (df mid-flight); readback discarded") },
        ));
        let df_mid = match poll_once(fut.as_mut()) {
            std::task::Poll::Ready(Err(e)) => e,
            other => panic!("expected Ready(Err), got {other:?}"),
        };
        assert!(format!("{df_mid:#}").contains("df mid-flight"));

        let mut fut = std::pin::pin!(finish_depthformer_sample_with(
            |_| Ok(()),
            || async { Ok(vec![0u8; 16]) },
        ));
        assert!(matches!(
            poll_once(fut.as_mut()),
            std::task::Poll::Ready(Err(_))
        ));

        let codes_u32 = [11u32, 22, 33, 44, 55, 66, 77, 88];
        let code_bytes = bytemuck::cast_slice(&codes_u32).to_vec();
        let mut fut = std::pin::pin!(finish_depthformer_sample_with(
            |_| Ok(()),
            || async move { Ok(code_bytes) },
        ));
        let got_codes = match poll_once(fut.as_mut()) {
            std::task::Poll::Ready(Ok(c)) => c,
            other => panic!("expected Ready(Ok), got {other:?}"),
        };
        assert_eq!(got_codes, [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn prepare_depthformer_gguf_rejects_bad_dtypes_and_shapes_upfront() {
        fn build(
            n_embd: usize,
            head_dim: usize,
            ffn_dim: usize,
            dl_cols: usize,
            n_vocab: usize,
            mut overrides: impl FnMut(&str, &mut Vec<usize>, &mut u32),
        ) -> crate::gguf::GgufFile {
            let mut b = Some(
                crate::gguf::GgufBuilder::new()
                    .kv_u32("depthformer_n_layer", 1)
                    .kv_u32("depthformer_n_embd", n_embd as u32),
            );
            {
                let mut add = |name: &str, mut dims: Vec<usize>| {
                    let mut ty = 0u32; // F32
                    overrides(name, &mut dims, &mut ty);
                    let numel: usize = dims.iter().product();
                    if let Some(builder) = b.take() {
                        b = Some(builder.tensor(name, &dims, ty, vec![0u8; numel * 4]));
                    }
                };
                let qkv_out = (32 + 2 * 8) * head_dim;
                add("depthformer.layers.0.operator_norm.weight", vec![n_embd]);
                add(
                    "depthformer.layers.0.operator.qkv_proj.weight",
                    vec![n_embd, qkv_out],
                );
                add(
                    "depthformer.layers.0.operator.attention.q_layernorm.weight",
                    vec![head_dim],
                );
                add(
                    "depthformer.layers.0.operator.attention.k_layernorm.weight",
                    vec![head_dim],
                );
                add(
                    "depthformer.layers.0.operator.out_proj.weight",
                    vec![32 * head_dim, n_embd],
                );
                add("depthformer.layers.0.ffn_norm.weight", vec![n_embd]);
                add(
                    "depthformer.layers.0.feed_forward.w1.weight",
                    vec![n_embd, ffn_dim],
                );
                add(
                    "depthformer.layers.0.feed_forward.w2.weight",
                    vec![ffn_dim, n_embd],
                );
                add(
                    "depthformer.layers.0.feed_forward.w3.weight",
                    vec![n_embd, ffn_dim],
                );
                add("depth_linear.weight", vec![dl_cols, 8 * n_embd]);
                add("depth_linear.bias", vec![8 * n_embd]);
                for j in 0..8 {
                    add(
                        &format!("depth_embeddings.{j}.embedding_norm.weight"),
                        vec![n_embd],
                    );
                    add(
                        &format!("depth_embeddings.{j}.to_logits.weight"),
                        vec![n_embd, n_vocab],
                    );
                    if j < 7 {
                        add(
                            &format!("depth_embeddings.{j}.embedding.weight"),
                            vec![n_embd, n_vocab],
                        );
                    }
                }
            }
            b.unwrap().build()
        }

        use crate::model::audio_decoder::HeadDimBounds;
        // Valid baseline passes cleanly (head_dim=4 and head_dim=6 both valid on WGPU).
        let valid = build(8, 4, 16, 6, 12, |_, _, _| {});
        let prep = super::prepare_depthformer_gguf(&valid, HeadDimBounds::WGPU).unwrap();
        assert_eq!(prep.n_embd_d, 8);
        assert_eq!(prep.dl_cols, 6);
        let valid_hd6 = build(8, 6, 16, 6, 12, |_, _, _| {});
        assert!(super::prepare_depthformer_gguf(&valid_hd6, HeadDimBounds::WGPU).is_ok());

        // GGML_TYPE_I32 (26) fails cleanly via try_to_f32_vec() instead of panicking.
        let i32_dl_w = build(8, 4, 16, 6, 12, |name, _, ty| {
            if name == "depth_linear.weight" {
                *ty = 26;
            }
        });
        assert!(super::prepare_depthformer_gguf(&i32_dl_w, HeadDimBounds::WGPU).is_err());

        let i32_dl_b = build(8, 4, 16, 6, 12, |name, _, ty| {
            if name == "depth_linear.bias" {
                *ty = 26;
            }
        });
        assert!(super::prepare_depthformer_gguf(&i32_dl_b, HeadDimBounds::WGPU).is_err());

        // Invalid dl_shape (1D) and dl_cols == 0 fail upfront.
        let bad_dl_rank = build(8, 4, 16, 6, 12, |name, dims, _| {
            if name == "depth_linear.weight" {
                *dims = vec![6 * 8 * 8];
            }
        });
        assert!(super::prepare_depthformer_gguf(&bad_dl_rank, HeadDimBounds::WGPU).is_err());

        let bad_dl_cols = build(8, 4, 16, 0, 12, |_, _, _| {});
        assert!(super::prepare_depthformer_gguf(&bad_dl_cols, HeadDimBounds::WGPU).is_err());

        // head_dim > 128 and odd head_dim rejected on WGPU.
        let bad_hd = build(8, 130, 16, 6, 12, |_, _, _| {});
        assert!(super::prepare_depthformer_gguf(&bad_hd, HeadDimBounds::WGPU).is_err());
        let odd_hd = build(8, 3, 16, 6, 12, |_, _, _| {});
        assert!(super::prepare_depthformer_gguf(&odd_hd, HeadDimBounds::WGPU).is_err());

        // ffn_dim == 0 and n_vocab == 0 rejected upfront.
        let bad_ffn = build(8, 4, 0, 6, 12, |_, _, _| {});
        assert!(super::prepare_depthformer_gguf(&bad_ffn, HeadDimBounds::WGPU).is_err());
        let bad_vocab = build(8, 4, 16, 6, 0, |_, _, _| {});
        assert!(super::prepare_depthformer_gguf(&bad_vocab, HeadDimBounds::WGPU).is_err());

        // Bad layer matrix dims rejected before weight dequantization.
        let bad_w3 = build(8, 4, 16, 6, 12, |name, dims, _| {
            if name == "depthformer.layers.0.feed_forward.w3.weight" {
                *dims = vec![8, 4];
            }
        });
        assert!(super::prepare_depthformer_gguf(&bad_w3, HeadDimBounds::WGPU).is_err());

        // Unsupported GGML placeholder type Q2_K (10 -> size_bytes=0) rejected upfront.
        let q2k_w1 = build(8, 4, 16, 6, 12, |name, _, ty| {
            if name == "depthformer.layers.0.feed_forward.w1.weight" {
                *ty = 10;
            }
        });
        let q2k_err = format!(
            "{:#}",
            super::prepare_depthformer_gguf(&q2k_w1, HeadDimBounds::WGPU)
                .err()
                .expect("Q2_K must fail upfront")
        );
        assert!(q2k_err.contains("unsupported GGML type 10"), "{q2k_err}");
    }

    #[test]
    fn prevalidate_detokenizer_gguf_and_probe_helpers_reject_bad_dtypes_and_geometry_before_gpu() {
        use crate::model::audio_decoder::tests::build_minimal_detokenizer_gguf;

        // 1. Valid lfm.* and blk.* detokenizer GGUFs (including head_dim=6 on WGPU) pass pure-host
        //    prevalidation without touching GpuContext.
        let valid_lfm = build_minimal_detokenizer_gguf(8, 4, 2, 1, 16, false, |_, _, _| {});
        let cfg = super::WgpuAudioDecoder::prevalidate_detokenizer_gguf(&valid_lfm).unwrap();
        assert_eq!(
            (cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.n_embd_head),
            (8, 2, 1, 4)
        );

        let valid_blk = build_minimal_detokenizer_gguf(8, 4, 2, 1, 16, true, |_, _, _| {});
        let cfg_blk = super::WgpuAudioDecoder::prevalidate_detokenizer_gguf(&valid_blk).unwrap();
        assert_eq!(cfg_blk.ffn_dim, 16);

        let valid_hd6 = build_minimal_detokenizer_gguf(12, 6, 2, 1, 16, false, |_, _, _| {});
        assert!(super::WgpuAudioDecoder::prevalidate_detokenizer_gguf(&valid_hd6).is_ok());

        let assert_fails_before_gpu =
            |gguf: std::sync::Arc<crate::gguf::GgufFile>, needle: &str, label: &str| {
                let helper_err = format!(
                    "{:#}",
                    super::WgpuAudioDecoder::prevalidate_detokenizer_gguf(&gguf).unwrap_err()
                );
                assert!(
                    helper_err.contains(needle),
                    "{label} helper: expected {needle:?}, got {helper_err:?}"
                );
                let msg = format!(
                    "{:#}",
                    super::WgpuAudioDecoder::from_gguf(&gguf, std::path::Path::new(""))
                        .err()
                        .unwrap_or_else(|| panic!("{label}: expected Err"))
                );
                assert!(
                    !msg.contains("No suitable GPU adapter"),
                    "{label}: skipped prevalidation and reached GpuContext::new(): {msg}"
                );
                assert!(
                    msg.contains(needle),
                    "{label} from_gguf: expected {needle:?}, got {msg:?}"
                );
            };

        // 2. Q2_K (10 -> size_bytes=0) on an lfm.* probe tensor must surface unsupported type 10
        //    instead of falling back to absent blk.* and reporting "not found" (R8-G2).
        let q2k_probe_conv = std::sync::Arc::new(build_minimal_detokenizer_gguf(
            8,
            4,
            2,
            1,
            16,
            false,
            |name, _, ty| {
                if name == "lfm.layers.0.conv.in_proj.weight" {
                    *ty = 10;
                }
            },
        ));
        let direct_conv_err = format!(
            "{:#}",
            super::get_detok_tensor(
                &q2k_probe_conv,
                "lfm.layers.0.conv.in_proj.weight",
                "blk.0.shortconv.in_proj.weight",
            )
            .err()
            .expect("get_detok_tensor must fail on Q2_K conv probe")
        );
        assert!(
            direct_conv_err.contains("unsupported GGML type 10")
                && !direct_conv_err.contains("not found"),
            "get_detok_tensor masked conv error: {direct_conv_err}"
        );
        assert_fails_before_gpu(q2k_probe_conv, "unsupported GGML type 10", "q2k probe conv");

        let q2k_probe_tensor = std::sync::Arc::new(build_minimal_detokenizer_gguf(
            8,
            4,
            2,
            1,
            16,
            false,
            |name, _, ty| {
                if name == "lfm.layers.2.self_attn.q_layernorm.weight" {
                    *ty = 10;
                }
            },
        ));
        let direct_tensor_err = format!(
            "{:#}",
            super::get_detok_tensor(
                &q2k_probe_tensor,
                "lfm.layers.2.self_attn.q_layernorm.weight",
                "blk.2.attn_q_norm.weight",
            )
            .err()
            .expect("get_detok_tensor must fail on Q2_K probe tensor")
        );
        assert!(
            direct_tensor_err.contains("unsupported GGML type 10")
                && !direct_tensor_err.contains("not found"),
            "get_detok_tensor masked error: {direct_tensor_err}"
        );
        assert_fails_before_gpu(
            q2k_probe_tensor,
            "unsupported GGML type 10",
            "q2k probe tensor",
        );

        // 3. Q2_K and I32 on non-probe detokenizer layer weights also fail upfront before GPU init.
        let i32_layer = std::sync::Arc::new(build_minimal_detokenizer_gguf(
            8,
            4,
            2,
            1,
            16,
            false,
            |name, _, ty| {
                if name == "lin.weight" {
                    *ty = 26;
                }
            },
        ));
        assert_fails_before_gpu(i32_layer, "I32", "i32 lin.weight");

        let q2k_layer = std::sync::Arc::new(build_minimal_detokenizer_gguf(
            8,
            4,
            2,
            1,
            16,
            true,
            |name, _, ty| {
                if name == "blk.1.shortconv.conv.weight" {
                    *ty = 10;
                }
            },
        ));
        assert_fails_before_gpu(q2k_layer, "unsupported GGML type 10", "q2k blk layer");

        // 4. HeadDimBounds::WGPU (odd head_dim=3 and oversized head_dim=130) rejected upfront.
        let odd_hd = std::sync::Arc::new(build_minimal_detokenizer_gguf(
            6,
            3,
            2,
            1,
            16,
            false,
            |_, _, _| {},
        ));
        assert_fails_before_gpu(odd_hd, "multiple of 2", "head_dim=3 WGPU");

        let big_hd = std::sync::Arc::new(build_minimal_detokenizer_gguf(
            260,
            130,
            2,
            1,
            16,
            false,
            |_, _, _| {},
        ));
        assert_fails_before_gpu(big_hd, "head_dim 130 > 128", "head_dim=130 WGPU");
    }
}

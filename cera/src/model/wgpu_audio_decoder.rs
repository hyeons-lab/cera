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
//! Metal decoder's precision path. KV caches are f32 (the WGPU `flash_attention`
//! kernel reads `array<f32>`), so there is no f16 cast and the path is a touch
//! more accurate than Metal's f16 cache.
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
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use wgpu::Buffer;

#[cfg(not(target_arch = "wasm32"))]
use crate::backend::wgpu::DevicePollExt;
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
pub(crate) struct Pipelines {
    gemm_f32: wgpu::ComputePipeline,
    gemv_f32: wgpu::ComputePipeline,
    gemv_f32_accum: wgpu::ComputePipeline,
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
    argmax_f32: wgpu::ComputePipeline,
    df_embed_add: wgpu::ComputePipeline,
    df_kv_copy: wgpu::ComputePipeline,
    df_sample_logits: wgpu::ComputePipeline,
    df_ffn_gate_up: wgpu::ComputePipeline,
    df_qkv_gemv: wgpu::ComputePipeline,
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
    hidden_buf: Buffer,           // residual stream [N x hs]
    normed_buf: Buffer,           // rmsnorm out / attn out [N x max(hs, q_dim)]
    proj_buf: Buffer,             // conv in_proj [N x 3hs] / attn Q [N x q_dim]
    gate_buf: Buffer,             // attn K / op residual / ffn gate [N x max(hs, kv_dim, ffn)]
    up_buf: Buffer,               // attn V / ffn up / ffn down [N x max(hs, kv_dim, ffn)]
    q_single: Buffer,             // one token's Q [q_dim]
    attn_single: Buffer,          // one token's attn out [q_dim]
    spectrum_buf: Buffer,         // [N x (n_fft_bins*2)]
    spectrum_staging_buf: Buffer, // pre-allocated staging buffer for zero-alloc spectrum readback
    rope_freqs_dummy: Buffer,     // 1-element buffer bound at qk_norm_rope binding 5

    // Persistent state.
    conv_bufs: Vec<Option<Buffer>>, // [d_conv x hs] per conv layer
    kv_k: Vec<Option<Buffer>>,      // [swa x kv_dim] f32 per attn layer
    kv_v: Vec<Option<Buffer>>,
    n_past: AtomicUsize,
}

fn get_detok_tensor(gguf: &GgufFile, name1: &str, name2: &str) -> Result<crate::tensor::Tensor> {
    gguf.get_tensor(name1)
        .or_else(|_| gguf.get_tensor(name2))
        .with_context(|| format!("tensor not found: neither `{name1}` nor `{name2}`"))
}

fn get_detok_mmap_weight(
    gguf: &Arc<GgufFile>,
    name1: &str,
    name2: &str,
) -> Result<crate::model::weights::MmapWeight> {
    crate::model::weights::MmapWeight::from_gguf(gguf, name1)
        .or_else(|_| crate::model::weights::MmapWeight::from_gguf(gguf, name2))
        .with_context(|| format!("tensor weight not found: neither `{name1}` nor `{name2}`"))
}

impl WgpuAudioDecoder {
    pub fn from_gguf(gguf: &Arc<GgufFile>, _vocoder_path: &Path) -> Result<Self> {
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
        // Config from tensor shapes (same derivation as DetokenizerWeights).
        let conv_in = get_detok_mmap_weight(
            gguf,
            "lfm.layers.0.conv.in_proj.weight",
            "blk.0.shortconv.in_proj.weight",
        )?;
        let n_embd = conv_in.cols;
        let q_norm_t = get_detok_tensor(
            gguf,
            "lfm.layers.2.self_attn.q_layernorm.weight",
            "blk.2.attn_q_norm.weight",
        )?;
        let head_dim = q_norm_t.shape()[0];
        // `n_head` and `n_kv` below both divide by this, so a corrupt vocoder
        // GGUF reporting an empty q_layernorm shape would be a div-by-zero
        // panic. Same wording as the CPU loader's check in `audio_decoder.rs`,
        // since the two read the same tensor and fail for the same reason.
        anyhow::ensure!(
            head_dim > 0,
            "detokenizer n_embd_head must be > 0 (q_layernorm shape was empty)"
        );
        let q_w = get_detok_mmap_weight(
            gguf,
            "lfm.layers.2.self_attn.q_proj.weight",
            "blk.2.attn_q.weight",
        )?;
        let n_head = q_w.rows / head_dim;
        let k_w = get_detok_mmap_weight(
            gguf,
            "lfm.layers.2.self_attn.k_proj.weight",
            "blk.2.attn_k.weight",
        )?;
        let n_kv = k_w.rows / head_dim;
        let ffn_w1_0 = get_detok_mmap_weight(
            gguf,
            "lfm.layers.0.feed_forward.w1.weight",
            "blk.0.ffn_gate.weight",
        )?;
        let ffn_dim = ffn_w1_0.rows;

        let layer_is_conv = vec![true, true, false, true, false, true, false, true];
        let n_layer = layer_is_conv.len();
        let kv_dim = n_kv * head_dim;
        let q_dim = n_head * head_dim;

        // Kernel contracts, checked once at construction so a future model or
        // config drift fails loudly here instead of silently corrupting output:
        //   - `flash_attention` sizes `q_shared`/`acc` at MAX_HEAD_DIM (128).
        //   - GQA: the kernel derives `group_size = n_head / n_kv`.
        //   - the single-chunk ring write below assumes `swa % N_FRAMES == 0`
        //     (so 6 consecutive positions never straddle the ring wrap).
        anyhow::ensure!(
            head_dim <= 128,
            "detokenizer head_dim {head_dim} > 128; wgpu flash_attention cannot size q_shared/acc"
        );
        anyhow::ensure!(
            n_kv > 0 && n_head.is_multiple_of(n_kv),
            "detokenizer GQA requires n_kv > 0 and n_head divisible by n_kv (n_head={n_head}, n_kv={n_kv})"
        );

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
        let pipes = Pipelines {
            gemm_f32,
            gemv_f32: ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "audio_df_gemv_f32"),
            gemv_f32_accum: ctx.create_pipeline(
                shaders::GEMV_F32,
                "gemv_f32_accum",
                "audio_df_gemv_f32_accum",
            ),
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
                shaders::FLASH_ATTENTION,
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
            bias_add: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add", "audio_detok_bias_add"),
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
        };

        // Dequantize each weight to f32 (CPU-matching precision) and upload.
        let make_weight = |name1: &str, name2: &str| -> Result<GpuWeight> {
            let t = get_detok_tensor(gguf, name1, name2)?;
            let f32_data = t.to_f32_vec();
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
            Ok(ctx.upload_f32(&get_detok_tensor(gguf, name1, name2)?.to_f32_vec(), name1))
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
        let spectrum_staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audio_spectrum_staging"),
            size: (spec_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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

        let df_src = depthformer_gguf.unwrap_or(detok_gguf);
        let depthformer = match WgpuDepthformer::try_from_gguf(&ctx, &pipes, df_src) {
            Ok(df) => {
                tracing::info!("[cera:wgpu_audio_decoder] WebGPU depthformer loaded successfully");
                Some(df)
            }
            Err(e) => {
                tracing::warn!(
                    "[cera:wgpu_audio_decoder] WebGPU depthformer unavailable ({e:#}), using CPU depthformer"
                );
                None
            }
        };

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
            spectrum_staging_buf,
            rope_freqs_dummy,
            conv_bufs,
            kv_k,
            kv_v,
            n_past: AtomicUsize::new(0),
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
    ) -> (usize, usize) {
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

        self.n_past.store(n_past + N_FRAMES, Ordering::Relaxed);
        (N_FRAMES, spec_per_frame)
    }

    pub fn detokenize_to_spectrum(
        &self,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Vec<f32> {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let (n_frames, spec_per_frame) =
            self.encode_detokenize_to_spectrum(&mut enc, cpu_weights, codes);
        self.ctx.submit_encoder(enc);
        self.ctx
            .download_f32(&self.spectrum_buf, n_frames * spec_per_frame)
    }

    pub async fn detokenize_to_spectrum_async(
        &self,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Result<Vec<f32>> {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let (n_frames, spec_per_frame) =
            self.encode_detokenize_to_spectrum(&mut enc, cpu_weights, codes);
        let size = (n_frames * spec_per_frame * std::mem::size_of::<f32>()) as u64;
        enc.copy_buffer_to_buffer(&self.spectrum_buf, 0, &self.spectrum_staging_buf, 0, size);
        self.ctx.submit_encoder(enc);

        let (tx, rx) = futures_channel::oneshot::channel();
        self.spectrum_staging_buf
            .slice(0..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        #[cfg(not(target_arch = "wasm32"))]
        self.ctx.device.poll_wait();

        let res = rx.await;
        let mut result = vec![0f32; n_frames * spec_per_frame];
        match res {
            Ok(Ok(())) => {
                let parse_res: Result<Vec<f32>> = {
                    let slice = self.spectrum_staging_buf.slice(0..size);
                    match slice.get_mapped_range() {
                        Ok(data) => {
                            let expected_bytes =
                                n_frames * spec_per_frame * std::mem::size_of::<f32>();
                            if data.len() < expected_bytes {
                                Err(anyhow::anyhow!(
                                    "GPU spectrum staging readback buffer truncated (expected {expected_bytes} bytes, got {})",
                                    data.len()
                                ))
                            } else {
                                bytemuck::cast_slice_mut(&mut result)
                                    .copy_from_slice(&data[..expected_bytes]);
                                Ok(result)
                            }
                        }
                        Err(e) => Err(anyhow::anyhow!("get_mapped_range failed: {e:?}")),
                    }
                };
                self.spectrum_staging_buf.unmap();
                parse_res
            }
            Ok(Err(e)) => anyhow::bail!("GPU readback failed: {e:?}"),
            Err(_) => anyhow::bail!("GPU readback channel closed"),
        }
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

impl crate::model::audio_decoder::AudioGpu for WgpuAudioDecoder {
    fn supports_depthformer(&self) -> bool {
        self.depthformer.is_some()
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

    fn reset_detokenizer(&self) {
        self.reset();
    }
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
    staging_readback_buf: Buffer,
    logits_buf: Buffer,
    kv_k: Vec<Buffer>,
    kv_v: Vec<Buffer>,
}

impl WgpuDepthformer {
    pub(crate) fn try_from_gguf(
        ctx: &GpuContext,
        pipes: &Arc<Pipelines>,
        gguf: &Arc<GgufFile>,
    ) -> Result<Self> {
        // Hyperparameters from GGUF metadata.
        let n_layer = gguf
            .get_u32("depthformer_n_layer")
            .context("missing depthformer_n_layer")? as usize;
        let n_embd = gguf
            .get_u32("depthformer_n_embd")
            .context("missing depthformer_n_embd")? as usize;

        // Derive head config from qkv_proj shape.
        let qkv = crate::model::weights::MmapWeight::from_gguf(
            gguf,
            "depthformer.layers.0.operator.qkv_proj.weight",
        )?;
        let q_norm_w =
            gguf.get_tensor("depthformer.layers.0.operator.attention.q_layernorm.weight")?;
        let n_embd_head = q_norm_w.shape()[0];
        let qkv_out = qkv.rows;
        let n_head = 32;
        let n_head_kv = 8;
        anyhow::ensure!(
            qkv_out == (n_head + 2 * n_head_kv) * n_embd_head,
            "qkv_proj shape mismatch"
        );

        let w1_0 = crate::model::weights::MmapWeight::from_gguf(
            gguf,
            "depthformer.layers.0.feed_forward.w1.weight",
        )?;
        let ffn_dim = w1_0.rows;

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

        let depth_linear_w =
            crate::model::weights::MmapWeight::from_gguf(gguf, "depth_linear.weight")?;
        let depth_linear_b = gguf.get_tensor("depth_linear.bias")?.to_f32_vec();
        let n_codebook = 8;
        let to_logits_0 = crate::model::weights::MmapWeight::from_gguf(
            gguf,
            "depth_embeddings.0.to_logits.weight",
        )?;
        let n_vocab = to_logits_0.rows;
        let n_embd_llm = depth_linear_w.cols;

        let dec_cfg = DecoderConfig {
            n_codebook,
            n_vocab,
            n_embd: n_embd_llm,
            rms_norm_eps: 1e-5,
        };

        Self::from_configs(ctx, pipes, gguf, &df_cfg, &dec_cfg, depth_linear_b)
    }

    fn from_configs(
        ctx: &GpuContext,
        pipes: &Arc<Pipelines>,
        gguf: &Arc<GgufFile>,
        df_cfg: &DepthformerConfig,
        dec_cfg: &DecoderConfig,
        depth_linear_b: Vec<f32>,
    ) -> Result<Self> {
        let n_embd = df_cfg.n_embd;
        let hd = df_cfg.n_embd_head;
        let n_head = df_cfg.n_head;
        let n_kv = df_cfg.n_head_kv;
        let q_dim = n_head * hd;
        let kv_dim = n_kv * hd;
        let ffn_dim = df_cfg.ffn_dim;

        let upload_vec = |name: &str| -> Result<Buffer> {
            Ok(ctx.upload_f32(&gguf.get_tensor(name)?.to_f32_vec(), name))
        };

        let alloc = |n: usize, label: &str| ctx.create_storage_rw((n * 4) as u64, label);
        let hidden_buf = alloc(n_embd, "df_hidden");
        let normed_buf = alloc(n_embd, "df_normed");
        let q_buf = alloc(q_dim, "df_q");
        let k_buf = alloc(kv_dim, "df_k");
        let v_buf = alloc(kv_dim, "df_v");
        let attn_out_buf = alloc(q_dim, "df_attn_out");
        let ffn_act_buf = alloc(ffn_dim, "df_ffn_act");
        let dl_t = gguf.get_tensor("depth_linear.weight")?;
        let dl_f32 = dl_t.to_f32_vec();
        let dl_cols = dl_t.shape()[0]; // e.g. 2048 (n_embd_llm)
        let dl_rows = dl_t.shape()[1]; // e.g. 8 * 1024
        let n_embd_d = dl_rows / dec_cfg.n_codebook;

        let embedding_in_buf = alloc(dl_cols, "df_embedding_in");
        let logits_buf = alloc(dec_cfg.n_vocab, "df_logits");
        let sampled_codes_buf = alloc(dec_cfg.n_codebook, "df_sampled_codes");
        let staging_readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("df_staging_readback"),
            size: (dec_cfg.n_codebook * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
            let qkv_f32 = qkv_t.to_f32_vec();
            let qkv_buf = ctx.upload_f32(&qkv_f32, &format!("{pfx}.qkv"));

            let op_norm = upload_vec(&format!("{pfx}.operator_norm.weight"))?;
            let q_norm = upload_vec(&format!("{pfx}.operator.attention.q_layernorm.weight"))?;
            let k_norm = upload_vec(&format!("{pfx}.operator.attention.k_layernorm.weight"))?;
            let wo_buf = ctx.upload_f32(
                &gguf
                    .get_tensor(&format!("{pfx}.operator.out_proj.weight"))?
                    .to_f32_vec(),
                &format!("{pfx}.wo"),
            );
            let ffn_norm = upload_vec(&format!("{pfx}.ffn_norm.weight"))?;

            let w1_vec = gguf
                .get_tensor(&format!("{pfx}.feed_forward.w1.weight"))?
                .to_f32_vec();
            let w3_vec = gguf
                .get_tensor(&format!("{pfx}.feed_forward.w3.weight"))?
                .to_f32_vec();
            let mut w13_vec = Vec::with_capacity(w1_vec.len() + w3_vec.len());
            w13_vec.extend_from_slice(&w1_vec);
            w13_vec.extend_from_slice(&w3_vec);
            let w13_buf = ctx.upload_f32(&w13_vec, &format!("{pfx}.w13"));
            let w2_buf = ctx.upload_f32(
                &gguf
                    .get_tensor(&format!("{pfx}.feed_forward.w2.weight"))?
                    .to_f32_vec(),
                &format!("{pfx}.w2"),
            );

            let operator_norm_bg = make_bg(
                &pipes.rmsnorm_batch,
                &[&hidden_buf, &normed_buf, &op_norm, &rmsnorm_params_1],
                &format!("df_op_norm_bg_{i}"),
            );
            let gemm_qkv_bg = make_bg(
                &pipes.df_qkv_gemv,
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
                    &pipes.df_kv_copy,
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
                &pipes.gemv_f32_accum,
                &[&wo_buf, &attn_out_buf, &hidden_buf, &wo_params],
                &format!("df_wo_accum_bg_{i}"),
            );
            let ffn_norm_bg = make_bg(
                &pipes.rmsnorm_batch,
                &[&hidden_buf, &normed_buf, &ffn_norm, &rmsnorm_params_1],
                &format!("df_ffn_norm_bg_{i}"),
            );
            let ffn_gate_up_bg = make_bg(
                &pipes.df_ffn_gate_up,
                &[&w13_buf, &normed_buf, &ffn_act_buf, &w13_params],
                &format!("df_ffn_gate_up_bg_{i}"),
            );
            let gemm_w2_accum_bg = make_bg(
                &pipes.gemv_f32_accum,
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
                &pipes.gemv_f32,
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
            let tl_f32 = tl_t.to_f32_vec();
            let tl_buf = ctx.upload_f32(&tl_f32, &format!("{pfx}.to_logits"));
            to_logits_gemm_bgs.push(make_bg(
                &pipes.gemv_f32,
                &[&tl_buf, &normed_buf, &logits_buf, &tl_params],
                &format!("df_tl_bg_{j}"),
            ));

            let am_params = ctx.upload_storage(
                bytemuck::cast_slice(&[dec_cfg.n_vocab as u32, j as u32]),
                &format!("df_argmax_params_{j}"),
            );
            argmax_bgs.push(make_bg(
                &pipes.argmax_f32,
                &[&logits_buf, &sampled_codes_buf, &am_params],
                &format!("df_argmax_bg_{j}"),
            ));

            if j < dec_cfg.n_codebook - 1 {
                let emb_t = gguf.get_tensor(&format!("{pfx}.embedding.weight"))?;
                let emb_buf = ctx.upload_f32(&emb_t.to_f32_vec(), &format!("{pfx}.emb"));
                let em_params = ctx.upload_storage(
                    bytemuck::cast_slice(&[n_embd as u32, j as u32]),
                    &format!("df_embed_params_{j}"),
                );
                embed_add_bgs.push(make_bg(
                    &pipes.df_embed_add,
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
                &pipes.df_sample_logits,
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
            staging_readback_buf,
            logits_buf,
            kv_k,
            kv_v,
        })
    }

    pub fn reset(&self) {}

    pub async fn sample_frame_async(
        &self,
        embedding: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<[i32; 8]> {
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
        let n_embd = self.df_cfg.n_embd;
        let hd = self.df_cfg.n_embd_head;
        let n_head = self.df_cfg.n_head as u32;
        let n_kv = self.df_cfg.n_head_kv as u32;
        let kv_dim = n_kv * hd as u32;

        let use_sampling = temperature > 0.0 && temperature.is_finite() && top_k > 1;
        let inv_temp = if use_sampling { 1.0 / temperature } else { 1.0 };

        if use_sampling {
            for j in 0..self.dec_cfg.n_codebook {
                let rand_val = rand::random::<f32>();
                let params = [self.dec_cfg.n_vocab as f32, j as f32, inv_temp, rand_val];
                self.ctx.queue.write_buffer(
                    &self.sample_params_bufs[j],
                    0,
                    bytemuck::cast_slice(&params),
                );
            }
        }

        let mut enc = self
            .ctx
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
                &self.pipes.gemv_f32,
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
                    &self.pipes.df_embed_add,
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
                    &self.pipes.df_qkv_gemv,
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
                    &self.pipes.df_kv_copy,
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
                    &self.pipes.gemv_f32_accum,
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
                    &self.pipes.df_ffn_gate_up,
                    &lw.ffn_gate_up_bg,
                    (self.ffn_dim_wg_m, 1, 1),
                    "df_ffn_gate_up",
                );

                // FFN W2 Down projection directly accumulated into hidden_buf (hidden_buf += W2 · ffn_act)
                dispatch_pass(
                    &mut enc,
                    &self.pipes.gemv_f32_accum,
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
                &self.pipes.gemv_f32,
                &self.to_logits_gemm_bgs[j],
                (self.n_vocab_wg_m, 1, 1),
                "df_to_logits_gemm",
            );

            // Sample on GPU (with temperature) or Argmax on GPU -> sampled_codes_buf[j]
            if use_sampling {
                dispatch_pass(
                    &mut enc,
                    &self.pipes.df_sample_logits,
                    &self.sample_bgs[j],
                    (1, 1, 1),
                    "df_sample_logits",
                );
            } else {
                dispatch_pass(
                    &mut enc,
                    &self.pipes.argmax_f32,
                    &self.argmax_bgs[j],
                    (1, 1, 1),
                    "df_argmax",
                );
            }
        }

        // Single 32-byte download of all 8 codes via pre-allocated staging buffer
        let size = (self.dec_cfg.n_codebook * 4) as u64;
        enc.copy_buffer_to_buffer(
            &self.sampled_codes_buf,
            0,
            &self.staging_readback_buf,
            0,
            size,
        );
        self.ctx.submit_encoder(enc);

        let (tx, rx) = futures_channel::oneshot::channel();
        self.staging_readback_buf
            .slice(0..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        #[cfg(not(target_arch = "wasm32"))]
        self.ctx.device.poll_wait();

        let res = rx.await;
        let mut codes = [0i32; 8];
        match res {
            Ok(Ok(())) => {
                let parse_res: Result<[i32; 8]> = {
                    let slice = self.staging_readback_buf.slice(0..size);
                    match slice.get_mapped_range() {
                        Ok(data) => {
                            let expected_bytes = 8 * std::mem::size_of::<u32>();
                            if data.len() < expected_bytes {
                                Err(anyhow::anyhow!(
                                    "GPU staging readback buffer truncated (expected {expected_bytes} bytes, got {})",
                                    data.len()
                                ))
                            } else {
                                for i in 0..8 {
                                    codes[i] = bytemuck::pod_read_unaligned::<u32>(
                                        &data[i * 4..(i + 1) * 4],
                                    ) as i32;
                                }
                                Ok(codes)
                            }
                        }
                        Err(e) => Err(anyhow::anyhow!("get_mapped_range failed: {e:?}")),
                    }
                };
                self.staging_readback_buf.unmap();
                parse_res
            }
            Ok(Err(e)) => anyhow::bail!("GPU readback failed: {e:?}"),
            Err(_) => anyhow::bail!("GPU readback channel closed"),
        }
    }
}

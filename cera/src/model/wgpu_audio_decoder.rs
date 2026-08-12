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
//! Scope (PR1): the detokenizer only, which is the validated win. The
//! depthformer (code sampling) stays on CPU; `supports_depthformer` reports
//! that, so `CERA_GPU_DF=1` keeps sampling on the CPU rather than reaching the
//! `sample_audio_frame` panic here. A WGPU depthformer is a follow-up.
//!
//! The final ISTFT (`istft_to_pcm`) runs on the GPU too: `exp_polar` maps the
//! polar half-spectrum to complex, a reg-tile GEMM against a precomputed real
//! inverse-DFT basis does the iDFT, and `overlap_add` windows and folds the
//! frames into PCM. Only the startup-pad strip stays on the CPU after readback.

use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use wgpu::Buffer;

use crate::backend::wgpu::{GpuContext, shaders};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{DetokenizerConfig, DetokenizerWeights};
use crate::model::gpu_lfm2::{
    MUL_MAT_TILE_M, MUL_MAT_TILE_N, MUL_MAT_TILE_WG_M, MUL_MAT_TILE_WG_N,
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

struct Pipelines {
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
}

pub struct WgpuAudioDecoder {
    ctx: GpuContext,
    cfg: DetokenizerConfig,
    pipes: Pipelines,

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
    n_past: Cell<usize>,
}

impl WgpuAudioDecoder {
    pub fn from_gguf(gguf: &Arc<GgufFile>, _vocoder_path: &Path) -> Result<Self> {
        let ctx = GpuContext::new()?;

        // Config from tensor shapes (same derivation as DetokenizerWeights).
        let conv_in =
            crate::model::weights::MmapWeight::from_gguf(gguf, "lfm.layers.0.conv.in_proj.weight")?;
        let n_embd = conv_in.cols;
        let q_norm_t = gguf.get_tensor("lfm.layers.2.self_attn.q_layernorm.weight")?;
        let head_dim = q_norm_t.shape()[0];
        // `n_head` and `n_kv` below both divide by this, so a corrupt vocoder
        // GGUF reporting an empty q_layernorm shape would be a div-by-zero
        // panic. Same wording as the CPU loader's check in `audio_decoder.rs`,
        // since the two read the same tensor and fail for the same reason.
        anyhow::ensure!(
            head_dim > 0,
            "detokenizer n_embd_head must be > 0 (q_layernorm shape was empty)"
        );
        let q_w = crate::model::weights::MmapWeight::from_gguf(
            gguf,
            "lfm.layers.2.self_attn.q_proj.weight",
        )?;
        let n_head = q_w.rows / head_dim;
        let k_w = crate::model::weights::MmapWeight::from_gguf(
            gguf,
            "lfm.layers.2.self_attn.k_proj.weight",
        )?;
        let n_kv = k_w.rows / head_dim;
        let ffn_w1_0 = crate::model::weights::MmapWeight::from_gguf(
            gguf,
            "lfm.layers.0.feed_forward.w1.weight",
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
                (
                    "TILE_K",
                    &format!("{tile_k}u", tile_k = crate::model::gpu_lfm2::MUL_MAT_TILE_K),
                ),
            ],
        );
        let pipes = Pipelines {
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
        };

        // Dequantize each weight to f32 (CPU-matching precision) and upload.
        let make_weight = |name: &str| -> Result<GpuWeight> {
            let t = gguf.get_tensor(name)?;
            let f32_data = t.to_f32_vec();
            let shape = t.shape();
            let (rows, cols) = match shape.len() {
                1 => (1, shape[0]),
                2 => (shape[1], shape[0]),
                _ => anyhow::bail!("unexpected rank for {name}"),
            };
            let buf = ctx.upload_f32(&f32_data, name);
            Ok(GpuWeight {
                buf,
                m: rows as u32,
                k: cols as u32,
            })
        };
        let upload_vec = |name: &str| -> Result<Buffer> {
            Ok(ctx.upload_f32(&gguf.get_tensor(name)?.to_f32_vec(), name))
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let pfx = format!("lfm.layers.{i}");
            let is_conv = cfg.layer_is_conv[i];
            let (cin, cop, cw) = if is_conv {
                (
                    Some(make_weight(&format!("{pfx}.conv.in_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.conv.out_proj.weight"))?),
                    Some(upload_vec(&format!("{pfx}.conv.conv.weight"))?),
                )
            } else {
                (None, None, None)
            };
            let (wq, wk, wv, wo, qn, kn) = if !is_conv {
                (
                    Some(make_weight(&format!("{pfx}.self_attn.q_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.self_attn.k_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.self_attn.v_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.self_attn.out_proj.weight"))?),
                    Some(upload_vec(&format!("{pfx}.self_attn.q_layernorm.weight"))?),
                    Some(upload_vec(&format!("{pfx}.self_attn.k_layernorm.weight"))?),
                )
            } else {
                (None, None, None, None, None, None)
            };
            layers.push(DetokLayerGpu {
                operator_norm: upload_vec(&format!("{pfx}.operator_norm.weight"))?,
                ffn_norm: upload_vec(&format!("{pfx}.ffn_norm.weight"))?,
                ffn_w1: make_weight(&format!("{pfx}.feed_forward.w1.weight"))?,
                ffn_w2: make_weight(&format!("{pfx}.feed_forward.w2.weight"))?,
                ffn_w3: make_weight(&format!("{pfx}.feed_forward.w3.weight"))?,
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

        let output_norm = upload_vec("lfm.embedding_norm.weight")?;
        let lin_w = make_weight("lin.weight")?;
        let lin_b = upload_vec("lin.bias")?;
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

        let alloc = |n: usize, label: &str| ctx.create_storage_rw((n * 4) as u64, label);
        let big = n_embd.max(kv_dim).max(ffn_dim);
        let hidden_buf = alloc(N_FRAMES * n_embd, "audio_detok_hidden");
        let normed_buf = alloc(N_FRAMES * n_embd.max(q_dim), "audio_detok_normed");
        let proj_buf = alloc(N_FRAMES * (3 * n_embd).max(q_dim), "audio_detok_proj");
        let gate_buf = alloc(N_FRAMES * big, "audio_detok_gate");
        let up_buf = alloc(N_FRAMES * big, "audio_detok_up");
        let q_single = alloc(q_dim, "audio_detok_q_single");
        let attn_single = alloc(q_dim, "audio_detok_attn_single");
        let spectrum_buf = alloc(N_FRAMES * (n_embd_bins(&cfg)), "audio_detok_spectrum");
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

        Ok(Self {
            ctx,
            cfg,
            pipes,
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
            n_past: Cell::new(0),
        })
    }

    pub fn config(&self) -> &DetokenizerConfig {
        &self.cfg
    }

    pub fn reset(&self) {
        self.n_past.set(0);
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

    pub fn detokenize_to_spectrum(
        &self,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Vec<f32> {
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

        let n_past = self.n_past.get();
        let scale = 1.0f32 / (hd as f32).sqrt();
        let seq_len = (n_past + N_FRAMES).min(self.cfg.swa_window_size) as u32;

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());

        for (il, lw) in self.layers.iter().enumerate() {
            // Phase 1: (add prev-layer residual then) rmsnorm with operator_norm.
            if il == 0 {
                self.rmsnorm_batch(
                    &mut enc,
                    &self.hidden_buf,
                    &self.normed_buf,
                    &lw.operator_norm,
                    n,
                );
            } else {
                self.add_rmsnorm_batch(
                    &mut enc,
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
                    &mut enc,
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
                    &mut enc,
                    &self.pipes.conv1d_fused_batch,
                    &[&self.proj_buf, rbuf, cw, &self.normed_buf, &cp],
                    (hs_u.div_ceil(256), 1, 1),
                    "audio_detok_conv1d",
                );
                // out_proj: normed → gate (op residual scratch) [n x hs].
                self.gemm(
                    &mut enc,
                    cop,
                    &self.normed_buf,
                    &self.gate_buf,
                    n,
                    hs_u,
                    hs_u,
                );
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
                self.gemm(
                    &mut enc,
                    wq,
                    &self.normed_buf,
                    &self.proj_buf,
                    n,
                    hs_u,
                    q_dim,
                );
                self.gemm(
                    &mut enc,
                    wk,
                    &self.normed_buf,
                    &self.gate_buf,
                    n,
                    hs_u,
                    kv_dim,
                );
                self.gemm(
                    &mut enc,
                    wv,
                    &self.normed_buf,
                    &self.up_buf,
                    n,
                    hs_u,
                    kv_dim,
                );

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
                    &mut enc,
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
                        &mut enc,
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
                self.gemm(
                    &mut enc,
                    wo,
                    &self.normed_buf,
                    &self.gate_buf,
                    n,
                    q_dim,
                    hs_u,
                );
            }

            // FFN: fused (hidden += op residual) + ffn_norm → normed.
            self.add_rmsnorm_batch(
                &mut enc,
                &self.hidden_buf,
                &self.normed_buf,
                &lw.ffn_norm,
                &self.gate_buf,
                n,
            );
            self.gemm(
                &mut enc,
                &lw.ffn_w1,
                &self.normed_buf,
                &self.gate_buf,
                n,
                hs_u,
                ffn,
            );
            self.gemm(
                &mut enc,
                &lw.ffn_w3,
                &self.normed_buf,
                &self.up_buf,
                n,
                hs_u,
                ffn,
            );
            self.silu_mul(&mut enc, &self.gate_buf, &self.up_buf, n * ffn);
            // down → up (next layer's residual scratch).
            self.gemm(
                &mut enc,
                &lw.ffn_w2,
                &self.gate_buf,
                &self.up_buf,
                n,
                ffn,
                hs_u,
            );
        }

        // Final residual add (last layer's FFN down lives in up_buf).
        self.add_inplace(&mut enc, &self.hidden_buf, &self.up_buf, n * hs_u);

        // Output norm + linear head + bias per frame.
        self.rmsnorm_batch(
            &mut enc,
            &self.hidden_buf,
            &self.normed_buf,
            &self.output_norm,
            n,
        );
        self.gemm(
            &mut enc,
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
            &mut enc,
            &self.pipes.bias_add,
            &[&self.spectrum_buf, &self.lin_b, &bias_params],
            (((N_FRAMES * spec_per_frame) as u32).div_ceil(256), 1, 1),
            "audio_detok_bias",
        );

        self.ctx.submit_encoder(enc);
        self.n_past.set(n_past + N_FRAMES);

        self.ctx
            .download_f32(&self.spectrum_buf, N_FRAMES * spec_per_frame)
    }

    /// Convert the accumulated spectrum to PCM on the GPU: `exp_polar` →
    /// iDFT-matmul → windowed `overlap_add`. Numerically mirrors the CPU
    /// `istft_to_pcm` (the iDFT basis folds the Hermitian mirror and the
    /// `1/n_fft` scale), down to the startup-pad strip, which stays on the CPU
    /// after readback. Runs once at end-of-generation, so the frame-sized
    /// scratch is allocated per call rather than persisted.
    pub fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        debug_assert_eq!(n_fft, self.cfg.n_fft, "istft n_fft differs from config");
        debug_assert_eq!(
            hop_length, self.cfg.hop_length,
            "istft hop differs from config"
        );
        let bins = n_fft / 2 + 1;
        let frame_size = bins * 2;
        let n_frames = spectrum.len() / frame_size;
        if n_frames == 0 {
            return vec![];
        }

        // A single 1D dispatch caps the frame count at the device's
        // `max_compute_workgroups_per_dimension` (65535 by default); the CPU
        // `istft_to_pcm` has no such limit, so fall back to it for very long
        // utterances (many minutes) rather than tripping a wgpu validation panic.
        let max_wg = self
            .ctx
            .device
            .limits()
            .max_compute_workgroups_per_dimension as usize;
        let exp_wg = (n_frames * bins).div_ceil(256);
        let oa_wg = (n_frames * hop_length).div_ceil(256);
        let gemm_wg_n = n_frames.div_ceil((MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N) as usize);
        if exp_wg > max_wg || oa_wg > max_wg || gemm_wg_n > max_wg {
            return crate::model::audio_decoder::istft_to_pcm(spectrum, n_fft, hop_length);
        }

        let spec_buf = self.ctx.upload_f32(spectrum, "audio_istft_spectrum_in");
        let halfspec = self
            .ctx
            .create_storage_rw((n_frames * frame_size * 4) as u64, "audio_istft_halfspec");
        let time_domain = self
            .ctx
            .create_storage_rw((n_frames * n_fft * 4) as u64, "audio_istft_time");
        let pcm_buf = self
            .ctx
            .create_storage_rw((n_frames * hop_length * 4) as u64, "audio_istft_pcm");

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());

        // 1. Polar half-spectrum (log-mag, angle) → interleaved [re | im].
        let ep = self.params(
            &[n_frames as u32, bins as u32],
            "audio_istft_exp_polar_params",
        );
        self.encode(
            &mut enc,
            &self.pipes.exp_polar,
            &[&spec_buf, &halfspec, &ep],
            (((n_frames * bins) as u32).div_ceil(256), 1, 1),
            "audio_istft_exp_polar",
        );

        // 2. iDFT as a matmul: time_domain[frame, t] = Σ_j halfspec[frame, j]·B[t, j].
        self.gemm(
            &mut enc,
            &self.idft_basis,
            &halfspec,
            &time_domain,
            n_frames as u32,
            frame_size as u32,
            n_fft as u32,
        );

        // 3. Windowed overlap-add → PCM.
        let oa = self.params(
            &[n_frames as u32, n_fft as u32, hop_length as u32, 0],
            "audio_istft_overlap_params",
        );
        self.encode(
            &mut enc,
            &self.pipes.overlap_add,
            &[&time_domain, &self.hann, &pcm_buf, &oa],
            (((n_frames * hop_length) as u32).div_ceil(256), 1, 1),
            "audio_istft_overlap_add",
        );

        self.ctx.submit_encoder(enc);

        let mut pcm = self.ctx.download_f32(&pcm_buf, n_frames * hop_length);
        // Strip the startup-padding artifacts, matching the CPU `istft_to_pcm`
        // tail (leave the buffer untouched when it is shorter than the pad).
        let padding = (n_fft - hop_length) / 2;
        if pcm.len() > padding {
            pcm.drain(..padding);
        }
        pcm
    }
}

/// Spectrum floats per frame: `(n_fft/2 + 1) * 2` (log-magnitude, angle).
fn n_embd_bins(cfg: &DetokenizerConfig) -> usize {
    (cfg.n_fft / 2 + 1) * 2
}

impl crate::model::audio_decoder::AudioGpu for WgpuAudioDecoder {
    // PR1 ships the detokenizer only; the depthformer stays on CPU and is a
    // follow-up.
    fn supports_depthformer(&self) -> bool {
        false
    }

    fn sample_audio_frame(&self, _embedding: &[f32], _temperature: f32, _top_k: usize) -> [i32; 8] {
        // Unreachable through the CLI, which routes the depthformer by
        // `supports_depthformer` above rather than by `CERA_GPU_DF` alone. Kept
        // as a backstop for a caller that wires the trait up itself.
        panic!("WGPU depthformer not implemented; unset CERA_GPU_DF to use the CPU sampler");
    }

    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
        self.detokenize_to_spectrum(cpu_weights, codes)
    }

    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        self.istft_to_pcm(spectrum, n_fft, hop_length)
    }

    fn reset_depthformer(&self) {}

    fn reset_detokenizer(&self) {
        self.reset();
    }
}

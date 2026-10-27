//! GPU forward pass for the LFM2-VL ViT vision encoder.
//!
//! The CPU encoder ([`super::vision_encoder::VisionEncoderWeights::encode_image`])
//! runs every linear layer through a per-token `gemv` — the slowest part of the
//! VL pipeline. This module batches the whole forward pass on the GPU.
//!
//! To target both wgpu and native Metal without duplicating the forward pass,
//! the math is written once against the [`VitGpuOps`] trait (opaque buffer
//! handle + a small op set: linear / layernorm / gelu / bias_add / attention /
//! residual-add). Each backend implements the trait; [`encode_image_gpu`] is
//! backend-agnostic.
//!
//! Numerical reference is the CPU encoder: see `tests` for the parity check.
//!
//! What stays on the CPU (tiny, data-dependent rearrangement): the patch
//! im2col, position-embedding interpolation, and pixel-shuffle. Everything with
//! real arithmetic (matmuls, norms, attention, activations) runs on the GPU.

use anyhow::Result;

#[cfg(feature = "gpu")]
use crate::backend::wgpu::DevicePollExt;

use super::vision_encoder::{
    VisionEncoderConfig, VisionEncoderWeights, extract_patch, interpolate_pos_embed_2d,
    pixel_shuffle,
};
use crate::model::weights::MmapWeight;

/// Largest patch count `vit_attention` supports (its score buffer is
/// workgroup-resident, sized MAX_TOKENS). LFM2-VL's `image_max_pixels` caps the
/// patch grid well under this, but [`encode_image_gpu`] guards it so a future
/// config can fall back to CPU instead of producing garbage.
///
/// This value MUST match the `MAX_TOKENS` literal sizing the `scores` scratch
/// array in both `vit_attention.wgsl` and `vit_attention.metal` — raising it
/// here without updating the shaders would let the guards admit grids larger
/// than the scratch array and silently write out of bounds. The Metal pipeline
/// can't take a runtime define (its source is a `&'static str` keyed by
/// pointer), so the three literals are duplicated and kept in lockstep by
/// `const_sync_tests::max_vit_tokens_matches_attention_shader_scratch`.
pub const MAX_VIT_TOKENS: usize = 1024;

// Co-located with `MAX_VIT_TOKENS` on purpose: this const-sync check must run in
// default CI (`#[cfg(test)]` only, no GPU/feature gate), unlike the feature-gated
// `tests` module at the end of the file, so it can't be folded into it.
#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod const_sync_tests {
    use super::MAX_VIT_TOKENS;

    /// Fails loudly if [`MAX_VIT_TOKENS`] is bumped without updating the
    /// `scores` scratch-array size in both attention shaders — the missing
    /// compile-time link the shaders' `MAX_TOKENS` literals would otherwise
    /// lack. Runs in default CI (no GPU/feature needed): it only reads source.
    #[test]
    fn max_vit_tokens_matches_attention_shader_scratch() {
        let wgsl = include_str!("../backend/shaders/vit_attention.wgsl");
        let metal = include_str!("../backend/shaders/vit_attention.metal");
        let wgsl_decl = format!("const MAX_TOKENS: u32 = {MAX_VIT_TOKENS}u;");
        let metal_decl = format!("constant uint MAX_TOKENS = {MAX_VIT_TOKENS}u;");
        assert!(
            wgsl.contains(&wgsl_decl),
            "vit_attention.wgsl MAX_TOKENS != MAX_VIT_TOKENS ({MAX_VIT_TOKENS}); \
             update the shader's `scores` array size to match"
        );
        assert!(
            metal.contains(&metal_decl),
            "vit_attention.metal MAX_TOKENS != MAX_VIT_TOKENS ({MAX_VIT_TOKENS}); \
             update the shader's `scores` array size to match"
        );
    }
}

/// Backend-agnostic GPU op interface for the ViT forward pass.
///
/// All ops operate on row-major f32 buffers. In-place ops (`bias_add`, `gelu`,
/// `add`) mutate the GPU contents behind `&Self::Buf`; producing ops (`linear`,
/// `layernorm`, `attention`) allocate and return a fresh buffer.
pub trait VitGpuOps {
    /// Opaque GPU buffer handle (e.g. `wgpu::Buffer`, `metal::Buffer`).
    type Buf;

    /// A linear-layer weight, ready for [`Self::linear`]. Both GPU backends keep
    /// Q8_0/Q4_0 weights packed and run a quantized GEMM straight from the bytes
    /// (Metal a simdgroup GEMM, wgpu the register-tiled `mul_mat_reg_tile` kernel
    /// with in-kernel Q8_0/Q4_0 decode); other dtypes are dequantized to f32.
    /// Distinct from [`Self::Buf`] so backends can carry the dtype/packing
    /// alongside the GPU buffer.
    type Weight;

    /// Upload `data` to a new GPU buffer.
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// Read `len` f32s back from a GPU buffer (blocking).
    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32>;

    /// Upload a (possibly quantized) linear weight `[out_dim, in_dim]` row-major
    /// (the `MmapWeight` layout). Backends may keep it packed or dequantize.
    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight;

    /// Upload a dense f32 linear weight `[out_dim, in_dim]` row-major (for
    /// weights that aren't `MmapWeight`-backed, e.g. the transposed patch conv).
    fn upload_weight_f32(&self, data: &[f32], out_dim: usize, in_dim: usize) -> Self::Weight;

    /// `y[tokens, out_dim] = x[tokens, in_dim] · wᵀ` where `w` is the
    /// `[out_dim, in_dim]` linear weight uploaded via [`Self::upload_weight`].
    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf;

    /// In-place broadcast bias: `x[t*dim + j] += bias[j]` for all `rows` rows.
    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize);

    /// Out-of-place affine LayerNorm over the last dim, returning a new buffer.
    /// `(src - mean) * inv_std * weight + bias` per row.
    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf;

    /// In-place tanh-approximation GELU over `len` elements.
    fn gelu(&self, x: &Self::Buf, len: usize);

    /// Bidirectional multi-head self-attention. Q/K/V are
    /// `[tokens, n_head*head_dim]` row-major; returns the same shape.
    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf;

    /// In-place ReLU over `len` elements.
    ///
    /// Default: panics; backends that run a ReLU stack must override.
    fn relu(&self, x: &Self::Buf, len: usize) {
        let _ = (x, len);
        unimplemented!("relu not supported by this backend")
    }

    /// Whether this backend implements [`Self::relu`]. `GpuStack::build` consults this for
    /// `Relu` specs so a non-overriding backend refuses instead of panicking mid-run.
    fn supports_relu(&self) -> bool {
        false
    }

    /// The most tokens [`Self::attention`] takes at this `head_dim`. The flash kernels have no
    /// limit; the scalar fallbacks keep one query's scores in workgroup memory.
    fn attention_token_limit(&self, _head_dim: usize) -> usize {
        MAX_VIT_TOKENS
    }

    /// The most rows [`Self::layernorm`] takes in one dispatch. Backends whose norm kernel
    /// spends one workgroup per row override this with the grid limit; the default admits.
    fn max_rows(&self) -> usize {
        usize::MAX
    }

    /// The largest single buffer a pass may ask for, in bytes.
    ///
    /// Default 0 refuses every GPU pass, so an impl without a real limit falls back to the
    /// CPU instead of over-allocating; backends must override.
    fn max_buffer_bytes(&self) -> u64 {
        0
    }

    /// In-place residual add: `dst[i] += src[i]` over `len` elements.
    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize);

    /// Block until all previously submitted GPU work has completed.
    ///
    /// Default no-op: Metal's ops each block on `wait_until_completed`, so the
    /// GPU is already idle between calls. wgpu's `dispatch` only *submits* (work
    /// runs asynchronously and is synced lazily at `download`), so it overrides
    /// this with `device.poll(Wait)`. Only the env-gated `VitProfiler` calls
    /// `sync` — the normal forward path never does, so wgpu keeps its pipelining.
    fn sync(&self) {}

    /// Called before and after one image's forward pass. The wgpu ops use them to reset and resolve
    /// the per-kernel GPU timestamps (`CERA_GPU_PROFILE=1`); default no-ops.
    fn begin_encode(&self) {}
    fn end_encode(&self) {}
}

/// One ViT block's weights, uploaded to GPU buffers. Linear weights are
/// `O::Weight` (possibly quantized); norm/bias vectors are plain f32 `O::Buf`.
pub struct GpuVitBlock<O: VitGpuOps> {
    ln1_w: O::Buf,
    ln1_b: O::Buf,
    q_w: O::Weight,
    q_b: O::Buf,
    k_w: O::Weight,
    k_b: O::Buf,
    v_w: O::Weight,
    v_b: O::Buf,
    o_w: O::Weight,
    o_b: O::Buf,
    ln2_w: O::Buf,
    ln2_b: O::Buf,
    ffn_up_w: O::Weight,
    ffn_up_b: O::Buf,
    ffn_down_w: O::Weight,
    ffn_down_b: O::Buf,
}

/// All vision-encoder weights uploaded to GPU buffers, plus the small CPU-side
/// state the per-call rearrangements need (config + trained position embedding).
///
/// Built once via [`GpuVitWeights::build`] and reused across images — the
/// upload (the LFM2-VL mmproj dequantized to f32) is the expensive part and
/// must not happen per image.
pub struct GpuVitWeights<O: VitGpuOps> {
    cfg: VisionEncoderConfig,
    /// Trained position embedding `[n_trained_patches * n_embd]`, kept on CPU
    /// for per-call bilinear interpolation to the dynamic grid.
    position_embed: Vec<f32>,
    /// Patch-embed kernel transposed to `[n_embd, in_dim]` (the `linear`
    /// layout); the CPU encoder stores it as `[in_dim, n_embd]` for its
    /// `C = A·B` matmul, but `linear` computes `A·Bᵀ`.
    patch_conv_wt: O::Weight,
    patch_conv_b: O::Buf,
    blocks: Vec<GpuVitBlock<O>>,
    post_ln_w: O::Buf,
    post_ln_b: O::Buf,
    mm1_w: O::Weight,
    mm1_b: O::Buf,
    mm2_w: O::Weight,
    mm2_b: O::Buf,
    /// Projector intermediate width (`mm.1` rows). Derived from the tensor
    /// shape, not the LFM2 `projection_dim·2` convention, to stay robust to
    /// variants — matching `vision_encoder`'s loader.
    proj_intermediate: usize,
}

impl<O: VitGpuOps> GpuVitWeights<O> {
    /// Upload every encoder weight via `ops`. Run once per loaded model.
    pub fn build(ops: &O, w: &VisionEncoderWeights) -> Self {
        let cfg = w.config.clone();
        let p = cfg.patch_size;
        let in_dim = 3 * p * p;
        let out_dim = cfg.n_embd;

        // Transpose conv_w [in_dim, out_dim] → [out_dim, in_dim].
        let src = &w.patch_embed.conv_w;
        let mut convt = vec![0f32; in_dim * out_dim];
        for i in 0..in_dim {
            for o in 0..out_dim {
                convt[o * in_dim + i] = src[i * out_dim + o];
            }
        }

        let blocks = w
            .blocks
            .iter()
            .map(|b| GpuVitBlock {
                ln1_w: ops.upload(&b.ln1_w),
                ln1_b: ops.upload(&b.ln1_b),
                q_w: ops.upload_weight(&b.q_w),
                q_b: ops.upload(&b.q_b),
                k_w: ops.upload_weight(&b.k_w),
                k_b: ops.upload(&b.k_b),
                v_w: ops.upload_weight(&b.v_w),
                v_b: ops.upload(&b.v_b),
                o_w: ops.upload_weight(&b.o_w),
                o_b: ops.upload(&b.o_b),
                ln2_w: ops.upload(&b.ln2_w),
                ln2_b: ops.upload(&b.ln2_b),
                ffn_up_w: ops.upload_weight(&b.ffn_up_w),
                ffn_up_b: ops.upload(&b.ffn_up_b),
                ffn_down_w: ops.upload_weight(&b.ffn_down_w),
                ffn_down_b: ops.upload(&b.ffn_down_b),
            })
            .collect();

        GpuVitWeights {
            position_embed: w.position_embed.clone(),
            patch_conv_wt: ops.upload_weight_f32(&convt, out_dim, in_dim),
            patch_conv_b: ops.upload(&w.patch_embed.conv_b),
            blocks,
            post_ln_w: ops.upload(&w.post_ln_w),
            post_ln_b: ops.upload(&w.post_ln_b),
            mm1_w: ops.upload_weight(&w.projector.mm1_w),
            mm1_b: ops.upload(&w.projector.mm1_b),
            mm2_w: ops.upload_weight(&w.projector.mm2_w),
            mm2_b: ops.upload(&w.projector.mm2_b),
            proj_intermediate: w.projector.mm1_w.rows,
            cfg,
        }
    }
}

/// Build the `[n_patches, in_dim]` im2col matrix the patch-embed linear consumes.
/// Mirrors the extraction in `vision_encoder::patch_embed_compute` (minus the
/// matmul, which moves to the GPU). `image` is `[3, target_h, target_w]` NCHW.
fn im2col_patches(
    image: &[f32],
    cfg: &VisionEncoderConfig,
    grid_w: usize,
    grid_h: usize,
) -> Vec<f32> {
    let p = cfg.patch_size;
    let in_dim = 3 * p * p;
    let target_w = grid_w * p;
    let target_h = grid_h * p;
    let h_stride = target_w;
    let c_stride = target_h * target_w;
    let n_patches = grid_w * grid_h;

    let mut patches = vec![0f32; n_patches * in_dim];
    for patch_idx in 0..n_patches {
        let base = patch_idx * in_dim;
        extract_patch(
            image,
            &mut patches[base..base + in_dim],
            patch_idx,
            grid_w,
            p,
            h_stride,
            c_stride,
        );
    }
    patches
}

/// Env-gated per-op wall-clock profiler for [`encode_image_gpu`].
///
/// Enabled by setting `CERA_VIT_PROFILE` to a non-empty value other than `0`, `false` or `off`. When
/// unset the profiler is `None` and the forward pass runs with zero overhead
/// (no timers, no `sync` calls). When set, each GPU op is followed by
/// `ops.sync()` so its wall-clock isolates that op — accurate on Metal (ops
/// already block) and on wgpu (the forced sync drains the otherwise-async
/// dispatch).
///
/// NOTE: those per-op syncs serialize wgpu, so the *sum* reported here exceeds
/// wgpu's real pipelined total — use the `vit_encode_bench` benchmark (profiling
/// off) for end-to-end numbers. The per-op *breakdown* is the bottleneck signal.
struct VitProfiler {
    /// `(label, summed duration, call count)`, in first-seen order.
    spans: std::cell::RefCell<Vec<(&'static str, std::time::Duration, u32)>>,
}

impl VitProfiler {
    /// Whether `CERA_VIT_PROFILE` is set to a non-empty value other than `0`, `false` or `off`. Read
    /// from the environment once and cached, so the disabled hot path is a
    /// single atomic load (the env var is a process-lifetime toggle anyway).
    fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| crate::backend::cpu_features::env_enabled("CERA_VIT_PROFILE"))
    }

    /// `Some` iff profiling is [`enabled`](Self::enabled).
    fn from_env() -> Option<Self> {
        Self::enabled().then(|| Self {
            spans: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// Add `d` to the running total for `label` (creating it on first use).
    fn record(&self, label: &'static str, d: std::time::Duration) {
        let mut spans = self.spans.borrow_mut();
        match spans.iter_mut().find(|s| s.0 == label) {
            Some(s) => {
                s.1 += d;
                s.2 += 1;
            }
            None => spans.push((label, d, 1)),
        }
    }

    /// Print the breakdown to stderr, sorted by descending total time, with each
    /// op's share of the summed total, call count, and per-call mean.
    fn report(&self, n_patches: usize, n_layer: usize) {
        let mut spans = self.spans.borrow().clone();
        spans.sort_by_key(|s| std::cmp::Reverse(s.1));
        let total_ms: f64 = spans.iter().map(|s| s.1.as_secs_f64() * 1e3).sum();
        eprintln!(
            "\nViT GPU profile — {n_patches} patches, {n_layer} layers \
             (per-op sync; wgpu sum is serialized, not the pipelined total)"
        );
        eprintln!(
            "  {:<16} {:>10} {:>7} {:>6} {:>10}",
            "stage", "total ms", "%", "calls", "mean ms"
        );
        for (label, dur, count) in &spans {
            let ms = dur.as_secs_f64() * 1e3;
            let pct = if total_ms > 0.0 {
                ms / total_ms * 100.0
            } else {
                0.0
            };
            let mean = ms / *count as f64;
            eprintln!("  {label:<16} {ms:>10.2} {pct:>6.1}% {count:>6} {mean:>10.3}");
        }
        eprintln!("  {:<16} {total_ms:>10.2} {:>6.1}%", "TOTAL", 100.0);
    }
}

/// Run the ViT encoder + projector on the GPU. Backend-agnostic: `ops` provides
/// the kernels, `gpu_w` the uploaded weights. Output is identical in shape to
/// [`VisionEncoderWeights::encode_image`]: `[n_image_tokens * projection_dim]`.
///
/// Set `CERA_VIT_PROFILE=1` to print a per-op timing breakdown (see
/// `VitProfiler`); unset, this runs with zero profiling overhead.
pub fn encode_image_gpu<O: VitGpuOps>(
    ops: &O,
    gpu_w: &GpuVitWeights<O>,
    pixels: &[f32],
    grid_w: usize,
    grid_h: usize,
) -> Result<Vec<f32>> {
    let cfg = &gpu_w.cfg;
    anyhow::ensure!(grid_w > 0 && grid_h > 0, "grid dims must be > 0");
    anyhow::ensure!(
        cfg.scale_factor > 0,
        "vision encoder config has scale_factor=0"
    );
    anyhow::ensure!(
        grid_w.is_multiple_of(cfg.scale_factor) && grid_h.is_multiple_of(cfg.scale_factor),
        "grid {grid_w}×{grid_h} not divisible by scale_factor ({})",
        cfg.scale_factor,
    );

    let p = cfg.patch_size;
    let in_dim = 3 * p * p;
    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;
    let n_head = cfg.n_head;
    let head_dim = n_embd / n_head;
    let n_patches = grid_w * grid_h;
    let eps = cfg.eps;

    anyhow::ensure!(
        pixels.len() == 3 * grid_w * p * grid_h * p,
        "encode_image_gpu: pixels.len() {} != 3·target_w·target_h",
        pixels.len()
    );
    anyhow::ensure!(
        n_patches <= MAX_VIT_TOKENS,
        "encode_image_gpu: {n_patches} patches exceeds GPU MAX_VIT_TOKENS ({MAX_VIT_TOKENS}); \
         caller should fall back to CPU",
    );
    // Only now touch the GPU context (flush the pending encoder, reset the profiler): a rejected
    // input must leave it as it was.
    ops.begin_encode();

    // Env-gated profiler (`CERA_VIT_PROFILE`). `None` → zero overhead.
    let prof = VitProfiler::from_env();
    // Time a GPU op: run it, force the GPU to finish (so async wgpu dispatches
    // are attributed correctly), then record. Compiles to bare `$e` when off.
    macro_rules! timed {
        ($label:literal, $e:expr) => {{
            match &prof {
                Some(p) => {
                    let __t = crate::time::Instant::now();
                    let __r = $e;
                    ops.sync();
                    p.record($label, __t.elapsed());
                    __r
                }
                None => $e,
            }
        }};
    }
    // Time a CPU-side stage (no GPU sync — these don't submit GPU work).
    macro_rules! timed_cpu {
        ($label:literal, $e:expr) => {{
            match &prof {
                Some(p) => {
                    let __t = crate::time::Instant::now();
                    let __r = $e;
                    p.record($label, __t.elapsed());
                    __r
                }
                None => $e,
            }
        }};
    }

    // 1. Patch embed: im2col on CPU, batched matmul + bias on GPU.
    let patches = timed_cpu!("im2col", im2col_patches(pixels, cfg, grid_w, grid_h));
    let patches_buf = timed!("upload", ops.upload(&patches));
    let tokens = timed!(
        "linear",
        ops.linear(
            &patches_buf,
            &gpu_w.patch_conv_wt,
            n_patches,
            n_embd,
            in_dim
        )
    );
    timed!(
        "bias_add",
        ops.bias_add(&tokens, &gpu_w.patch_conv_b, n_patches, n_embd)
    );

    // 2. Add (interpolated) position embeddings. The trained grid is square;
    // guard that in release too (the CPU encoder only `debug_assert`s it), since
    // a non-square `n_trained_patches` would make `interpolate_pos_embed_2d`
    // index out of bounds. Borrow (not clone) the trained embedding on the
    // common matching-grid path.
    let trained_side = (cfg.n_trained_patches as f64).sqrt().round() as usize;
    anyhow::ensure!(
        trained_side * trained_side == cfg.n_trained_patches,
        "non-square trained pos-embed grid ({} patches) is not supported",
        cfg.n_trained_patches,
    );
    let pos: std::borrow::Cow<[f32]> = if grid_w == trained_side && grid_h == trained_side {
        std::borrow::Cow::Borrowed(&gpu_w.position_embed)
    } else {
        timed_cpu!(
            "posembed_interp",
            std::borrow::Cow::Owned(interpolate_pos_embed_2d(
                &gpu_w.position_embed,
                trained_side,
                trained_side,
                grid_h,
                grid_w,
                n_embd,
            ))
        )
    };
    let pos_buf = timed!("upload", ops.upload(&pos));
    timed!("add", ops.add(&tokens, &pos_buf, n_patches * n_embd));

    // 3. ViT blocks.
    for blk in &gpu_w.blocks {
        // Pre-attention LN → Q/K/V (+bias) → attention → O proj (+bias) → residual.
        let normed = timed!(
            "layernorm",
            ops.layernorm(&tokens, &blk.ln1_w, &blk.ln1_b, eps, n_patches, n_embd)
        );
        let q = timed!(
            "linear",
            ops.linear(&normed, &blk.q_w, n_patches, n_embd, n_embd)
        );
        timed!("bias_add", ops.bias_add(&q, &blk.q_b, n_patches, n_embd));
        let k = timed!(
            "linear",
            ops.linear(&normed, &blk.k_w, n_patches, n_embd, n_embd)
        );
        timed!("bias_add", ops.bias_add(&k, &blk.k_b, n_patches, n_embd));
        let v = timed!(
            "linear",
            ops.linear(&normed, &blk.v_w, n_patches, n_embd, n_embd)
        );
        timed!("bias_add", ops.bias_add(&v, &blk.v_b, n_patches, n_embd));
        let attn = timed!(
            "attention",
            ops.attention(&q, &k, &v, n_patches, n_head, head_dim)
        );
        let proj = timed!(
            "linear",
            ops.linear(&attn, &blk.o_w, n_patches, n_embd, n_embd)
        );
        timed!("bias_add", ops.bias_add(&proj, &blk.o_b, n_patches, n_embd));
        timed!("add", ops.add(&tokens, &proj, n_patches * n_embd));

        // Pre-MLP LN → FFN up (+bias) → GELU → FFN down (+bias) → residual.
        let normed2 = timed!(
            "layernorm",
            ops.layernorm(&tokens, &blk.ln2_w, &blk.ln2_b, eps, n_patches, n_embd)
        );
        let mid = timed!(
            "linear",
            ops.linear(&normed2, &blk.ffn_up_w, n_patches, n_ff, n_embd)
        );
        timed!(
            "bias_add",
            ops.bias_add(&mid, &blk.ffn_up_b, n_patches, n_ff)
        );
        timed!("gelu", ops.gelu(&mid, n_patches * n_ff));
        let down = timed!(
            "linear",
            ops.linear(&mid, &blk.ffn_down_w, n_patches, n_embd, n_ff)
        );
        timed!(
            "bias_add",
            ops.bias_add(&down, &blk.ffn_down_b, n_patches, n_embd)
        );
        timed!("add", ops.add(&tokens, &down, n_patches * n_embd));
    }

    // 4. Post-LN.
    let tokens = timed!(
        "layernorm",
        ops.layernorm(
            &tokens,
            &gpu_w.post_ln_w,
            &gpu_w.post_ln_b,
            eps,
            n_patches,
            n_embd
        )
    );

    // 5. Pixel-shuffle on CPU (pure rearrangement).
    let tok_cpu = timed!("download", ops.download(&tokens, n_patches * n_embd));
    let pooled = timed_cpu!(
        "pixel_shuffle",
        pixel_shuffle(&tok_cpu, cfg, grid_w, grid_h)
    );
    let pooled_in_dim = n_embd * cfg.scale_factor * cfg.scale_factor;
    let n_out = pooled.len() / pooled_in_dim;

    // 6. Projector: mm.1 (+bias) + GELU → mm.2 (+bias).
    let mid_dim = gpu_w.proj_intermediate;
    let pooled_buf = timed!("upload", ops.upload(&pooled));
    let proj_dim = cfg.projection_dim;
    let mid = timed!(
        "linear",
        ops.linear(&pooled_buf, &gpu_w.mm1_w, n_out, mid_dim, pooled_in_dim)
    );
    timed!("bias_add", ops.bias_add(&mid, &gpu_w.mm1_b, n_out, mid_dim));
    timed!("gelu", ops.gelu(&mid, n_out * mid_dim));
    let out = timed!(
        "linear",
        ops.linear(&mid, &gpu_w.mm2_w, n_out, proj_dim, mid_dim)
    );
    timed!(
        "bias_add",
        ops.bias_add(&out, &gpu_w.mm2_b, n_out, proj_dim)
    );

    let result = timed!("download", ops.download(&out, n_out * proj_dim));
    ops.end_encode();
    if let Some(p) = &prof {
        p.report(n_patches, gpu_w.blocks.len());
    }
    Ok(result)
}

/// Async variant of [`encode_image_gpu`] for wgpu, using non-blocking readbacks
/// (`PendingReadback` / `download_f32_async`). Essential for wasm32 / WebGPU where
/// synchronous `poll_wait` + `mpsc::recv` deadlocks against the JS event loop.
#[cfg(feature = "gpu")]
pub async fn encode_image_gpu_wgpu_async(
    ops: &WgpuVitOps,
    gpu_w: &GpuVitWeights<WgpuVitOps>,
    pixels: &[f32],
    grid_w: usize,
    grid_h: usize,
) -> Result<Vec<f32>> {
    let cfg = &gpu_w.cfg;
    anyhow::ensure!(grid_w > 0 && grid_h > 0, "grid dims must be > 0");
    anyhow::ensure!(
        cfg.scale_factor > 0,
        "vision encoder config has scale_factor=0"
    );
    anyhow::ensure!(
        grid_w.is_multiple_of(cfg.scale_factor) && grid_h.is_multiple_of(cfg.scale_factor),
        "grid {grid_w}×{grid_h} not divisible by scale_factor ({})",
        cfg.scale_factor,
    );

    let p = cfg.patch_size;
    let in_dim = 3 * p * p;
    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;
    let n_head = cfg.n_head;
    let head_dim = n_embd / n_head;
    let n_patches = grid_w * grid_h;
    let eps = cfg.eps;

    anyhow::ensure!(
        pixels.len() == 3 * grid_w * p * grid_h * p,
        "encode_image_gpu: pixels.len() {} != 3·target_w·target_h",
        pixels.len()
    );
    anyhow::ensure!(
        n_patches <= MAX_VIT_TOKENS,
        "encode_image_gpu: {n_patches} patches exceeds GPU MAX_VIT_TOKENS ({MAX_VIT_TOKENS}); \
         caller should fall back to CPU",
    );

    // Only now touch the GPU context (flush the pending encoder, reset the profiler): a rejected
    // input must leave it as it was.
    ops.begin_encode();

    // 1. Patch embed: im2col on CPU, batched matmul + bias on GPU.
    let patches = im2col_patches(pixels, cfg, grid_w, grid_h);
    let patches_buf = ops.upload(&patches);
    let tokens = ops.linear(
        &patches_buf,
        &gpu_w.patch_conv_wt,
        n_patches,
        n_embd,
        in_dim,
    );
    ops.bias_add(&tokens, &gpu_w.patch_conv_b, n_patches, n_embd);

    // 2. Add (interpolated) position embeddings.
    anyhow::ensure!(
        cfg.n_trained_patches > 0,
        "n_trained_patches must be greater than 0"
    );
    let trained_side = (cfg.n_trained_patches as f64).sqrt().round() as usize;
    anyhow::ensure!(
        trained_side * trained_side == cfg.n_trained_patches,
        "non-square trained pos-embed grid ({} patches) is not supported",
        cfg.n_trained_patches,
    );
    let pos: std::borrow::Cow<[f32]> = if grid_w == trained_side && grid_h == trained_side {
        std::borrow::Cow::Borrowed(&gpu_w.position_embed)
    } else {
        std::borrow::Cow::Owned(interpolate_pos_embed_2d(
            &gpu_w.position_embed,
            trained_side,
            trained_side,
            grid_h,
            grid_w,
            n_embd,
        ))
    };
    anyhow::ensure!(
        !pos.is_empty(),
        "failed to interpolate 2D position embeddings (dimension mismatch)"
    );
    let pos_buf = ops.upload(&pos);
    ops.add(&tokens, &pos_buf, n_patches * n_embd);

    // 3. ViT blocks.
    for blk in &gpu_w.blocks {
        let normed = ops.layernorm(&tokens, &blk.ln1_w, &blk.ln1_b, eps, n_patches, n_embd);
        let q = ops.linear(&normed, &blk.q_w, n_patches, n_embd, n_embd);
        ops.bias_add(&q, &blk.q_b, n_patches, n_embd);
        let k = ops.linear(&normed, &blk.k_w, n_patches, n_embd, n_embd);
        ops.bias_add(&k, &blk.k_b, n_patches, n_embd);
        let v = ops.linear(&normed, &blk.v_w, n_patches, n_embd, n_embd);
        ops.bias_add(&v, &blk.v_b, n_patches, n_embd);

        let attn_out = ops.attention(&q, &k, &v, n_patches, n_head, head_dim);
        let o = ops.linear(&attn_out, &blk.o_w, n_patches, n_embd, n_embd);
        ops.bias_add(&o, &blk.o_b, n_patches, n_embd);
        ops.add(&tokens, &o, n_patches * n_embd);

        // Pre-FFN LN → FFN up (+bias) → GELU → FFN down (+bias) → residual.
        let normed = ops.layernorm(&tokens, &blk.ln2_w, &blk.ln2_b, eps, n_patches, n_embd);
        let mid = ops.linear(&normed, &blk.ffn_up_w, n_patches, n_ff, n_embd);
        ops.bias_add(&mid, &blk.ffn_up_b, n_patches, n_ff);
        ops.gelu(&mid, n_patches * n_ff);
        let down = ops.linear(&mid, &blk.ffn_down_w, n_patches, n_embd, n_ff);
        ops.bias_add(&down, &blk.ffn_down_b, n_patches, n_embd);
        ops.add(&tokens, &down, n_patches * n_embd);
    }

    // 4. Post-LN.
    let tokens = ops.layernorm(
        &tokens,
        &gpu_w.post_ln_w,
        &gpu_w.post_ln_b,
        eps,
        n_patches,
        n_embd,
    );

    // 5. Pixel-shuffle on CPU (async readback).
    ops.flush();
    let tok_cpu = ops
        .ctx
        .download_f32_async(&tokens, n_patches * n_embd)
        .await?;
    let pooled = pixel_shuffle(&tok_cpu, cfg, grid_w, grid_h);
    let pooled_in_dim = n_embd * cfg.scale_factor * cfg.scale_factor;
    let n_out = pooled.len() / pooled_in_dim;

    // 6. Projector: mm.1 (+bias) + GELU → mm.2 (+bias).
    let mid_dim = gpu_w.proj_intermediate;
    let pooled_buf = ops.upload(&pooled);
    let proj_dim = cfg.projection_dim;
    let mid = ops.linear(&pooled_buf, &gpu_w.mm1_w, n_out, mid_dim, pooled_in_dim);
    ops.bias_add(&mid, &gpu_w.mm1_b, n_out, mid_dim);
    ops.gelu(&mid, n_out * mid_dim);
    let out = ops.linear(&mid, &gpu_w.mm2_w, n_out, proj_dim, mid_dim);
    ops.bias_add(&out, &gpu_w.mm2_b, n_out, proj_dim);

    ops.flush();
    let result = ops.ctx.download_f32_async(&out, n_out * proj_dim).await?;
    ops.end_encode();
    Ok(result)
}

// ── wgpu backend implementation ──────────────────────────────────────────────

// Register-tiled `mul_mat_reg_tile` config for the ViT GEMMs. Each workgroup
// computes a (WG_M·TILE_M)×(WG_N·TILE_N) = 64×64 output tile with WG_M·WG_N =
// 256 threads, staging a TILE_K=16 slice of decoded weights + activations into
// shared memory per step (so weights are reused across the token tile instead
// of re-read per token like the batched-GEMV kernels).
//
// The shader hand-unrolls a 4×4 thread tile, so TILE_M/TILE_N are not free (see
// `mul_mat_reg_tile.wgsl`). A previous 32×32 tile here used TILE_N=1 because
// higher TILE_N measured slower; that was an artifact of the accumulator array
// spilling out of registers, which the shader rewrite fixed.
//
// Identical to the text path's geometry (`MUL_MAT_TILE_*` in gpu_lfm2.rs). It
// was briefly narrower — 16×8 — purely to stay under the 16 KiB WebGPU
// workgroup-storage floor while the text path used TILE_K=32 and needed 17.0
// KiB. At TILE_K=16 both are 8.5 KiB, so the constraint that forced them apart
// is gone.
//
// Staying under the 16 KiB floor matters more here than on the text path:
// `WgpuVitOps::new` catches a pipeline-creation panic and
// `try_wgpu_vision_encoder`'s `.ok()?` then drops vision to the CPU encoder
// silently, as a perf cliff rather than an error.
//
// UNMEASURED on the ViT specifically, and taken for uniformity rather than for
// throughput: widening WG_N 8 -> 16 doubles the output tile to 64×64 on token
// counts no sweep covered, and a projector input below 64 columns is then mostly
// padding. The measured TILE_K trade-off recorded against `MUL_MAT_TILE_K` in
// gpu_lfm2.rs is the text path's; do not read this widening as having the same
// backing.
#[cfg(feature = "gpu")]
const VIT_MM_WG_M: u32 = 16;
#[cfg(feature = "gpu")]
const VIT_MM_WG_N: u32 = 16;
#[cfg(feature = "gpu")]
const VIT_MM_TILE_M: u32 = 4;
#[cfg(feature = "gpu")]
const VIT_MM_TILE_N: u32 = 4;
#[cfg(feature = "gpu")]
const VIT_MM_TILE_K: u32 = 16;

// Same invariants the text path asserts (gpu_lfm2.rs), plus the workgroup-storage
// bound and an explicit tie to the text constants — the comment above says the
// two geometries are identical, so make that enforced rather than aspirational.
// They can be retuned together, but not apart: the ViT's failure mode is a
// swallowed pipeline-creation panic that silently drops vision to CPU.
//
// Asserted equal rather than *defined* as `= gpu_lfm2::MUL_MAT_TILE_*`, which
// would look tidier and is a standing review suggestion. Aliasing would make a
// text-path retune silently retune the ViT, and that is the one thing this
// block exists to prevent: the geometry here is UNMEASURED on the ViT (see
// above), so a value chosen from the text path's sweep is a value nobody has
// justified for this kernel. The assert turns such a change into a compile
// error naming both sites, which is the point where someone has to decide
// whether the ViT should follow. Keep them separate.
#[cfg(feature = "gpu")]
const _: () = assert!(
    VIT_MM_TILE_M == 4 && VIT_MM_TILE_N == 4,
    "mul_mat_reg_tile.wgsl hand-unrolls a 4x4 thread tile"
);
#[cfg(feature = "gpu")]
const _: () = assert!(
    VIT_MM_WG_M == crate::model::gpu_lfm2::MUL_MAT_TILE_WG_M
        && VIT_MM_WG_N == crate::model::gpu_lfm2::MUL_MAT_TILE_WG_N
        && VIT_MM_TILE_M == crate::model::gpu_lfm2::MUL_MAT_TILE_M
        && VIT_MM_TILE_N == crate::model::gpu_lfm2::MUL_MAT_TILE_N
        && VIT_MM_TILE_K == crate::model::gpu_lfm2::MUL_MAT_TILE_K,
    "the ViT and text reg-tile geometries are documented as identical; retune \
     them together or split the comment too"
);
#[cfg(feature = "gpu")]
const _: () = assert!(
    VIT_MM_TILE_K.is_multiple_of(8),
    "the Q4_0 shmem loader stages 8 consecutive k per thread"
);
// 16384 B = the 16 KiB WebGPU `max_compute_workgroup_storage_size` floor. At
// 16x16 / TILE_K=16 this evaluates to 8704 B. A compile error is the right
// failure mode: at runtime `WgpuVitOps::new` swallows the pipeline-creation
// panic and vision silently drops to the CPU encoder.
#[cfg(feature = "gpu")]
const _: () = assert!(
    (VIT_MM_TILE_K * (VIT_MM_WG_M * VIT_MM_TILE_M + 4)
        + VIT_MM_TILE_K * (VIT_MM_WG_N * VIT_MM_TILE_N + 4))
        * 4
        <= 16384,
    "ViT reg-tile shmem must stay within the 16 KiB WebGPU floor so a \
     spec-minimum adapter keeps the GPU vision encoder"
);

/// A wgpu linear weight `[out_dim, in_dim]` row-major. Mirrors
/// `MetalVitWeight`: quantized weights keep their packed bytes and run the
/// register-tiled `mul_mat_reg_tile` GEMM (decoding Q8_0/Q4_0 into shared
/// memory in-kernel); f32 weights (or quant dtypes without a tiled decoder)
/// fall back to the same register-tiled `mul_mat_reg_tile` matmul on a
/// dequantized f32 buffer (its f32 `INIT_SRC0_SHMEM_FLOAT` variant).
#[cfg(feature = "gpu")]
pub enum WgpuVitWeight {
    /// Dequantized f32 buffer, `[out_dim, in_dim]` row-major.
    Dense(wgpu::Buffer),
    /// Packed quantized bytes (`MmapWeight::data()` layout) + dtype.
    Quant {
        buf: wgpu::Buffer,
        dtype: crate::tensor::DType,
    },
    /// Q8_0 repacked into the streaming-GEMM layout (see `repack_q8_0_stream`): int8 words `q`
    /// and paired f16 scales `d`. Runs on the fp16 streaming kernel, which is an order of
    /// magnitude faster than the generic register-tile kernel on Adreno.
    QuantStream { q: wgpu::Buffer, d: wgpu::Buffer },
}

/// Repack Q8_0 GGUF bytes into the streaming-GEMM layout of `gemm_stream_q8_0_k64.slang`.
/// Returns `(q, d)`: `q[(k/4) * m + row]` is the row's weights `k..k+3` as four little-endian int8
/// bytes, and `d[(k/64) * m + row]` is `half(scale of block 2s) | half(scale of block 2s+1) << 16`.
/// Requires `k % 32 == 0`; when `k % 64 == 32`, the trailing 32-element half-slice is zero-padded.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn repack_q8_0_stream(data: &[u8], m: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    assert!(
        m > 0 && k > 0 && k.is_multiple_of(32),
        "q8_0 streaming repack needs m > 0, k > 0, and k % 32 == 0, got m={m}, k={k}"
    );
    let k_pad = k.next_multiple_of(64);
    let blocks_per_row = k / 32;
    assert_eq!(
        data.len(),
        m * blocks_per_row * 34,
        "q8_0 repack: wrong byte count"
    );
    let mut q = vec![0u32; m * k_pad / 4];
    let mut d = vec![0u32; m * (k_pad / 64)];
    for row in 0..m {
        for b in 0..blocks_per_row {
            let base = (row * blocks_per_row + b) * 34;
            let scale_bits = u16::from_le_bytes([data[base], data[base + 1]]) as u32;
            let pair = b / 2;
            if b % 2 == 0 {
                d[pair * m + row] = scale_bits;
            } else {
                d[pair * m + row] |= scale_bits << 16;
            }
            for w in 0..8 {
                let src = &data[base + 2 + w * 4..base + 2 + w * 4 + 4];
                q[(b * 8 + w) * m + row] = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
            }
        }
    }
    (q, d)
}

/// Quantize and repack row-major F16 `[m, k]` weights into the Q8_0 streaming layout of
/// `gemm_stream_q8_0_k64.slang`, zero-padding each row from `k` to `k.next_multiple_of(64)`.
/// Used for ViT linear layers stored as F16 in GGUF because `k` is not a multiple of 32 (such as
/// LFM2.5-VL-1.6B's `ffn_down` with `k = 4304`).
#[cfg(any(feature = "gpu", test))]
pub(crate) fn repack_f16_as_q8_0_stream(data: &[u8], m: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    assert!(
        m > 0 && k > 0,
        "f16 streaming repack needs m > 0 and k > 0, got m={m}, k={k}"
    );
    assert_eq!(
        data.len(),
        m * k * 2,
        "f16 streaming repack: wrong byte count"
    );
    let k_pad = k.next_multiple_of(64);
    let blocks_per_row = k.div_ceil(32);
    let mut q = vec![0u32; m * k_pad / 4];
    let mut d = vec![0u32; m * (k_pad / 64)];
    for row in 0..m {
        let row_bytes = &data[row * k * 2..(row + 1) * k * 2];
        for b in 0..blocks_per_row {
            let col_start = b * 32;
            let valid = (k - col_start).min(32);
            let mut block = [0.0f32; 32];
            let mut amax = 0.0f32;
            for (i, dst) in block[..valid].iter_mut().enumerate() {
                let off = (col_start + i) * 2;
                let bits = u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
                let val = half::f16::from_bits(bits).to_f32();
                *dst = val;
                let a = val.abs();
                if a > amax {
                    amax = a;
                }
            }
            let scale_f32 = amax / 127.0;
            let scale_f16 = half::f16::from_f32(scale_f32);
            let scale_bits = scale_f16.to_bits() as u32;
            let inv_scale = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            let pair = b / 2;
            if b % 2 == 0 {
                d[pair * m + row] = scale_bits;
            } else {
                d[pair * m + row] |= scale_bits << 16;
            }
            for w in 0..8 {
                let mut bytes = [0u8; 4];
                for j in 0..4 {
                    let qi = (block[w * 4 + j] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    bytes[j] = qi as u8;
                }
                q[(b * 8 + w) * m + row] = u32::from_le_bytes(bytes);
            }
        }
    }
    (q, d)
}

/// How [`WgpuVitOps::dispatch_elements`] binds one buffer of an elementwise kernel.
#[cfg(feature = "gpu")]
enum ElementBind<'a> {
    /// Bound whole, every chunk.
    Whole(&'a wgpu::Buffer),
    /// Bound at the chunk's element offset: a buffer the kernel walks with the thread index.
    Ranged(&'a wgpu::Buffer),
    /// The chunk's parameters.
    Params,
}

/// `(offset, len)` chunks covering `len` elements for [`WgpuVitOps::dispatch_elements`]:
/// the largest chunk that is a whole number of row-preserving steps and fits the dispatch
/// limit (`MAX_GROUPS` workgroups of 256 threads), with every boundary on the 64-element
/// storage-offset alignment. Pure so a host test can pin it without a device.
#[cfg(feature = "gpu")]
fn chunk_bounds(len: usize, unit: usize) -> Vec<(usize, usize)> {
    const MAX_GROUPS: usize = crate::backend::wgpu::MAX_WG as usize;
    const OFFSET_ALIGN_ELEMS: usize = 64;
    let unit = unit.max(1);
    let step = crate::model::gpu_lfm2::lcm_u64(unit as u64, OFFSET_ALIGN_ELEMS as u64) as usize;
    // the largest chunk that is a whole number of steps and fits the dispatch limit
    let chunk = ((MAX_GROUPS * 256) / step).max(1) * step;
    let mut bounds = Vec::new();
    let mut offset = 0usize;
    while offset < len {
        let n = chunk.min(len - offset);
        bounds.push((offset, n));
        offset += n;
    }
    bounds
}

/// wgpu implementation of [`VitGpuOps`]. Owns the [`GpuContext`](crate::backend::wgpu::GpuContext) and the compute
/// pipelines (compiled once) so it can be cached for the session's lifetime.
/// Bind groups are created per dispatch (cheap relative to the kernel work).
#[cfg(feature = "gpu")]
pub struct WgpuVitOps {
    pub(crate) ctx: crate::backend::wgpu::GpuContext,
    p_linear: wgpu::ComputePipeline,
    p_mul_mat_q8_0: wgpu::ComputePipeline,
    p_mul_mat_q4_0: wgpu::ComputePipeline,
    p_bias: wgpu::ComputePipeline,
    p_layernorm: wgpu::ComputePipeline,
    p_gelu: wgpu::ComputePipeline,
    p_relu: wgpu::ComputePipeline,
    p_attn: wgpu::ComputePipeline,
    p_attn_tiled: wgpu::ComputePipeline,
    /// Register-tiled flash attention, `None` when its build failed: the kernel needs
    /// `FLASH_ATTENTION_WORKGROUP_BYTES` of workgroup storage, past some adapters' limit,
    /// and losing it must fall back to the tiled/scalar kernels rather than fail the
    /// whole encoder.
    p_attn_flash: Option<wgpu::ComputePipeline>,
    p_add: wgpu::ComputePipeline,
    /// Flash-attention dispatches issued (test observability: proves the flash kernel ran,
    /// since every fallback produces correct output too and would otherwise hide a dead path).
    #[cfg(test)]
    flash_dispatches: std::sync::atomic::AtomicU32,
    /// Streaming Q8_0 GEMM and the activation transpose-cast it reads through; `None` where SPIR-V
    /// passthrough is unavailable (anything but Vulkan) or `CERA_VIT_STREAM=0`.
    p_gemm_q8_stream: Option<wgpu::ComputePipeline>,
    p_transpose_f16: Option<wgpu::ComputePipeline>,
    /// The fp16 GEMM attention (scores, column softmax, P·V) and the V cast it needs; same gating.
    p_attn_scores: Option<wgpu::ComputePipeline>,
    p_attn_softmax: Option<wgpu::ComputePipeline>,
    p_attn_pv: Option<wgpu::ComputePipeline>,
    p_cast_f16: Option<wgpu::ComputePipeline>,
    /// Passes recorded since the last submit. Every op used to build and submit its own encoder
    /// (about 700 submits per image, which left the GPU idle for half the tower's wall time); ops now
    /// append passes here and [`Self::flush`] submits them in groups.
    pending: std::sync::Mutex<PendingPasses>,
    /// Read-only parameter buffers keyed by their bytes. The same shapes recur in every layer, so
    /// this turns about 700 buffer creations per image into a few dozen.
    param_cache: std::sync::Mutex<std::collections::HashMap<Vec<u8>, wgpu::Buffer>>,
    /// Recycled intermediate buffers (see [`BufPool`]).
    pool: std::sync::Arc<BufPool>,
}

/// Free storage buffers by `(size in bytes, purpose)`. Allocating a fresh Vulkan buffer costs about
/// 0.45 ms on Adreno 830 (600 of them per image was 280 ms of host time, with the GPU idle), while
/// every layer asks for the same shapes, so buffers are recycled instead. Sizes are rounded up to a
/// power of two ([`pool_bucket`]), so images of similar size, and a warm-up encode, share buffers.
#[cfg(feature = "gpu")]
#[derive(Default)]
struct BufPool {
    free: std::sync::Mutex<std::collections::HashMap<(u64, &'static str), Vec<wgpu::Buffer>>>,
}

/// The size a pooled buffer is allocated at: `len` rounded up to a power of two (at least 256 bytes).
/// Shaders index by their own parameters, never by buffer length, so a larger buffer is harmless, and
/// the zeroed variant clears the whole of it.
#[cfg(feature = "gpu")]
fn pool_bucket(len: u64) -> u64 {
    len.max(256).next_power_of_two()
}

/// A storage buffer that goes back to its `BufPool` when dropped. Dereferences to the raw buffer.
///
/// Recycling is safe against in-flight GPU work: commands run in submission order and wgpu inserts the
/// hazard barriers between passes that touch the same buffer, so a later pass may reuse a buffer an
/// earlier, still-queued pass wrote.
#[cfg(feature = "gpu")]
pub struct VitBuf {
    buf: Option<wgpu::Buffer>,
    key: (u64, &'static str),
    pool: Option<std::sync::Arc<BufPool>>,
}

#[cfg(feature = "gpu")]
impl VitBuf {
    /// A buffer that is not pooled (inputs and one-off uploads).
    fn owned(buf: wgpu::Buffer) -> Self {
        Self {
            buf: Some(buf),
            key: (0, ""),
            pool: None,
        }
    }

    fn raw(&self) -> &wgpu::Buffer {
        self.buf.as_ref().expect("VitBuf used after drop")
    }
}

#[cfg(feature = "gpu")]
impl std::ops::Deref for VitBuf {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.raw()
    }
}

#[cfg(feature = "gpu")]
impl Drop for VitBuf {
    fn drop(&mut self) {
        if let (Some(buf), Some(pool)) = (self.buf.take(), self.pool.as_ref()) {
            pool.free
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(self.key)
                .or_default()
                .push(buf);
        }
    }
}

/// An encoder under construction and the number of compute passes already recorded into it.
#[cfg(feature = "gpu")]
#[derive(Default)]
struct PendingPasses {
    enc: Option<wgpu::CommandEncoder>,
    passes: u32,
}

/// Passes recorded before the pending encoder is submitted: about one layer's worth, so the GPU can
/// start layer i while the CPU encodes layer i+1 (batching a whole image idles the GPU through the
/// encode, which the decode path measured as a loss).
#[cfg(feature = "gpu")]
const VIT_PASSES_PER_SUBMIT: u32 = 24;

#[cfg(feature = "gpu")]
impl WgpuVitOps {
    /// The head-dim ceiling of the query-tiled attention kernel.
    const VIT_ATTN_TILED_MAX_HEAD_DIM: usize = 64;

    /// Whether the query-tiled attention kernel takes this head dim (and so streams K/V
    /// without a token cap). Single source for the dispatch branch and the token limit.
    ///
    /// On Android / Vulkan (Adreno, Mali), vit_attention_tiled.wgsl allocates
    /// 128 dynamic registers per thread across 256 threads that spill to DRAM,
    /// hanging the Qualcomm driver watchdog. vit_attention.wgsl uses workgroup
    /// shared memory (spill-free) and executes safely on mobile GPUs.
    fn streams_kv(head_dim: usize) -> bool {
        cfg!(not(target_os = "android")) && head_dim <= Self::VIT_ATTN_TILED_MAX_HEAD_DIM
    }

    pub fn new(ctx: crate::backend::wgpu::GpuContext) -> Result<Self> {
        use crate::backend::wgpu::shaders;
        // `create_pipeline*` return the pipeline directly and panic only on host-side
        // shader preprocessing failure (pinned wgpu 30 reports adapter-side
        // validation/compile failure to the error sink and hands back an invalid
        // pipeline instead of panicking). Catch the host panic here and surface an `Err`
        // so the caller's `?` degrades to the CPU encoder, mirroring
        // `MetalVitOps::new`'s `.ok()?`, instead of aborting `CeraEngine`
        // construction. The flash build below checks its error scope explicitly because
        // it is the one kernel that can exceed an adapter's workgroup limit.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let (wg_m, wg_n) = (format!("{VIT_MM_WG_M}u"), format!("{VIT_MM_WG_N}u"));
            let (tile_m, tile_n) = (format!("{VIT_MM_TILE_M}u"), format!("{VIT_MM_TILE_N}u"));
            let tile_k = format!("{VIT_MM_TILE_K}u");
            let mk_mul_mat = |label: &str, src0_ty: &str, init_src0: &str| {
                ctx.create_pipeline_with_defines(
                    shaders::MUL_MAT_REG_TILE,
                    "main",
                    label,
                    &[
                        ("SRC0_INNER_TYPE", src0_ty),
                        (init_src0, ""),
                        ("WORKGROUP_SIZE_M", &wg_m),
                        ("WORKGROUP_SIZE_N", &wg_n),
                        ("TILE_M", &tile_m),
                        ("TILE_N", &tile_n),
                        ("TILE_K", &tile_k),
                    ],
                )
            };
            // f32 batched matmul — the dequantized-weight fallback path.
            let p_linear = mk_mul_mat("vit_linear", "f32", "INIT_SRC0_SHMEM_FLOAT");
            // Register-tiled quantized GEMM: decode Q8_0/Q4_0 weight blocks into
            // shared memory in-kernel and reuse them across the token tile.
            let mk_quant = |label: &str, init_src0: &str| mk_mul_mat(label, "u32", init_src0);
            let p_mul_mat_q8_0 = mk_quant("vit_mul_mat_q8_0", "INIT_SRC0_SHMEM_Q8_0");
            let p_mul_mat_q4_0 = mk_quant("vit_mul_mat_q4_0", "INIT_SRC0_SHMEM_Q4_0");
            let stream = ctx.supports_spirv_passthrough()
                && !crate::backend::cpu_features::env_disabled("CERA_VIT_STREAM");
            let p_gemm_q8_stream = stream.then(|| ctx.gemm_stream_q8_0_k64_passthrough());
            let p_transpose_f16 = stream.then(|| ctx.transpose_cast_f16_passthrough());
            let p_attn_scores = stream.then(|| ctx.vit_attn_scores_f16_passthrough());
            let p_attn_softmax = stream.then(|| ctx.vit_attn_softmax_t_passthrough());
            let p_attn_pv = stream.then(|| ctx.vit_attn_pv_f16_passthrough());
            let p_cast_f16 = stream.then(|| ctx.cast_f32_f16_passthrough());
            Self {
                pending: Default::default(),
                param_cache: Default::default(),
                pool: Default::default(),
                p_gemm_q8_stream,
                p_transpose_f16,
                p_attn_scores,
                p_attn_softmax,
                p_attn_pv,
                p_cast_f16,
                p_bias: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add", "vit_bias_add"),
                p_layernorm: ctx.create_pipeline(
                    shaders::LAYERNORM_BATCH,
                    "layernorm_batch",
                    "vit_layernorm",
                ),
                p_gelu: ctx.create_pipeline(shaders::GELU, "gelu_inplace", "vit_gelu"),
                p_relu: ctx.create_pipeline(shaders::ACTIVATIONS, "relu_inplace", "vit_relu"),
                p_attn: ctx.create_pipeline(
                    shaders::VIT_ATTENTION,
                    "vit_attention",
                    "vit_attention",
                ),
                p_attn_tiled: ctx.create_pipeline(
                    shaders::VIT_ATTENTION_TILED,
                    "vit_attention_tiled",
                    "vit_attention_tiled",
                ),
                // Built only where the flash gate passes (the kernel needs
                // `FLASH_ATTENTION_WORKGROUP_BYTES`, past some adapters' limit,
                // and the checked build cannot run on wasm32, so the gate keeps
                // flash off there); a flash-only failure falls back, far slower,
                // instead of failing the encoder (shared tail in
                // `build_flash_pipeline_or_log`).
                p_attn_flash: if ctx.uses_flash_attention(64) {
                    ctx.build_flash_pipeline_or_log(
                        "ViT attention falls back to the tiled/scalar kernels",
                    )
                } else {
                    None
                },
                p_add: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "vit_add"),
                p_mul_mat_q8_0,
                p_mul_mat_q4_0,
                p_linear,
                ctx,
                #[cfg(test)]
                flash_dispatches: std::sync::atomic::AtomicU32::new(0),
            }
        }))
        .map_err(|_| {
            anyhow::anyhow!("wgpu ViT pipeline creation failed (shader compile/validation)")
        })
    }

    /// Whether `attention` with `head_dim` runs on the register-tiled flash kernel: head_dim 64
    /// on an adapter that runs it, with the pipeline built.
    fn flash_ready(&self, head_dim: usize) -> bool {
        self.p_attn_flash.is_some() && self.ctx.uses_flash_attention(head_dim)
    }

    /// Attend with the flash kernel in f32 instead of the fp16 GEMM pipeline the streaming path
    /// prefers. The fp16 pipeline rounds Q, K, V and the probabilities to half precision, which an
    /// image tower tolerates and the decision head (whose scores are compared to the host's to
    /// 2e-3) does not.
    #[must_use]
    pub fn with_f32_attention(mut self) -> Self {
        self.p_attn_scores = None;
        self.p_attn_softmax = None;
        self.p_attn_pv = None;
        self
    }

    /// Like [`Self::dispatch`], binding each buffer over `(offset_bytes, size_bytes)` of it
    /// (`size_bytes == 0` binds from the offset to the end). Offsets must be multiples of the
    /// storage-offset alignment (256 bytes).
    pub(crate) fn dispatch_ranges(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bufs: &[(&wgpu::Buffer, u64, u64)],
        workgroups: (u32, u32, u32),
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, &(buffer, offset, size))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size: wgpu::BufferSize::new(if size == 0 {
                        buffer.size().saturating_sub(offset)
                    } else {
                        size
                    }),
                }),
            })
            .collect();
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        // recorded into the pending encoder like every other pass, so it runs after the passes
        // already queued and before the ones that follow
        self.record(1, |enc| {
            let mut pass = self.ctx.begin_pass(enc, label);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        });
    }

    /// Encode one bind group from `bufs` (in binding order) and dispatch.
    pub(crate) fn dispatch(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
    ) {
        self.dispatch_seq(&[(label, pipeline, bufs, workgroups)]);
    }

    /// Append `passes` compute passes to the pending encoder, submitting it once it holds a layer's
    /// worth.
    fn record(&self, passes: u32, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let full = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let enc = pending.enc.get_or_insert_with(|| {
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
            });
            f(enc);
            pending.passes += passes;
            pending.passes >= VIT_PASSES_PER_SUBMIT
        };
        if full {
            self.flush();
        }
    }

    /// Submit whatever has been recorded. Must run before anything reads results back or waits.
    pub(crate) fn flush(&self) {
        let enc = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            pending.passes = 0;
            pending.enc.take()
        };
        if let Some(enc) = enc {
            self.ctx.submit_encoder(enc);
        }
    }

    /// A storage buffer of `len` bytes for `label`'s purpose, recycled when one is free. Contents are
    /// whatever the last user left; use [`Self::rw_buf_zeroed`] where padding must read as zero.
    pub(crate) fn rw_buf(&self, len: u64, label: &'static str) -> VitBuf {
        self.rw_buf_reused(len, label).0
    }

    /// [`Self::rw_buf`] and whether the buffer came out of the pool (dirty) rather than being created
    /// (all zeros). The pop and the answer come from one lock acquisition, so a buffer another thread
    /// returns in between cannot be handed out as fresh.
    fn rw_buf_reused(&self, len: u64, label: &'static str) -> (VitBuf, bool) {
        let len = pool_bucket(len);
        let reused = self
            .pool
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&(len, label))
            .and_then(|v| v.pop());
        let was_reused = reused.is_some();
        let buf = VitBuf {
            buf: Some(reused.unwrap_or_else(|| self.ctx.create_storage_rw(len, label))),
            key: (len, label),
            pool: Some(std::sync::Arc::clone(&self.pool)),
        };
        (buf, was_reused)
    }

    /// [`Self::rw_buf`] that is all zeros on return: fresh buffers already are, a recycled one is
    /// cleared by a command in the pending encoder, ahead of the passes that use it.
    fn rw_buf_zeroed(&self, len: u64, label: &'static str) -> VitBuf {
        let (buf, was_reused) = self.rw_buf_reused(len, label);
        if was_reused {
            self.record(0, |enc| enc.clear_buffer(buf.raw(), 0, None));
        }
        buf
    }

    /// A read-only parameter buffer holding `params`, created once per distinct content.
    fn params_buf(&self, params: &[u32], label: &str) -> wgpu::Buffer {
        let key: Vec<u8> = bytemuck::cast_slice(params).to_vec();
        let mut cache = self.param_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= 4096 {
            cache.clear();
        }
        cache
            .entry(key)
            .or_insert_with_key(|k| self.ctx.upload_storage(k, label))
            .clone()
    }

    /// Several dependent dispatches in one encoder and one submit, as consecutive compute passes
    /// (wgpu orders them, so a later pass reads an earlier one's output).
    #[allow(clippy::type_complexity)]
    fn dispatch_seq(
        &self,
        steps: &[(
            &str,
            &wgpu::ComputePipeline,
            &[&wgpu::Buffer],
            (u32, u32, u32),
        )],
    ) {
        let groups: Vec<wgpu::BindGroup> = steps
            .iter()
            .map(|(_, pipeline, bufs, _)| {
                let entries: Vec<wgpu::BindGroupEntry> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect();
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &entries,
                    })
            })
            .collect();
        self.record(steps.len() as u32, |enc| {
            for ((label, pipeline, _, workgroups), bind_group) in steps.iter().zip(&groups) {
                let mut pass = self.ctx.begin_pass(enc, label);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            }
        });
    }

    /// An elementwise kernel over `len` elements, one thread each, split into dispatches of at
    /// most 65535 workgroups (the per-dimension limit) when it is longer than 16.7M elements:
    /// the feed-forward intermediate of a few thousand tokens already is. `binds` are in binding
    /// order; a `Ranged` buffer is bound at the chunk's element offset (a multiple of 64
    /// elements, the storage-offset alignment), a `Whole` one as is, and `Params` takes
    /// `params(chunk_elements)`. `unit` keeps chunk boundaries on whole rows for kernels that
    /// index by `i % dim`.
    fn dispatch_elements(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        binds: &[ElementBind<'_>],
        len: usize,
        unit: usize,
        params: impl Fn(u32) -> Vec<u32>,
    ) {
        for (offset, n) in chunk_bounds(len, unit) {
            let p_buf = self.params_buf(&params(n as u32), "vit_elementwise_params");
            let entries: Vec<wgpu::BindGroupEntry> = binds
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: match b {
                        ElementBind::Whole(buf) => buf.as_entire_binding(),
                        ElementBind::Ranged(buf) => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: buf,
                                offset: (offset * 4) as u64,
                                size: wgpu::BufferSize::new(((n * 4) as u64).max(4)),
                            })
                        }
                        ElementBind::Params => p_buf.as_entire_binding(),
                    },
                })
                .collect();
            let bind_group = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
            self.record(1, |enc| {
                let mut pass = self.ctx.begin_pass(enc, label);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
            });
        }
    }

    /// Quantized `y[tokens, out_dim] = x[tokens, in_dim] · wᵀ` via the
    /// register-tiled `mul_mat_reg_tile` kernel (`pipe` carries the Q8_0/Q4_0
    /// in-kernel decoder). `wq` is the packed weight `[out_dim, in_dim]`; `x` is
    /// f32 `[tokens, in_dim]`; `y` is f32 `[tokens, out_dim]`. `MulMatParams`
    /// is `[m, k, n, x_stride, y_stride]`; the grid covers one
    /// (WG_M·TILE_M)×(WG_N·TILE_N) output tile per workgroup (dispatched 2D and
    /// linearized in-shader to dodge the 65535 `num_wg.x` cap).
    #[allow(clippy::too_many_arguments)]
    fn run_mul_mat_tiled(
        &self,
        pipe: &wgpu::ComputePipeline,
        wq: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) {
        // MulMatParams { m, k, n, x_stride, y_stride }.
        let params: [u32; 5] = [
            out_dim as u32,
            in_dim as u32,
            tokens as u32,
            in_dim as u32,
            out_dim as u32,
        ];
        let p_buf = self.params_buf(&params, "vit_mul_mat_params");
        let wg_m = (out_dim as u32).div_ceil(VIT_MM_WG_M * VIT_MM_TILE_M);
        let wg_n = (tokens as u32).div_ceil(VIT_MM_WG_N * VIT_MM_TILE_N);
        self.dispatch(
            "vit_linear_tiled",
            pipe,
            &[wq, x, y, &p_buf],
            (wg_m, wg_n, 1),
        );
    }
}

#[cfg(feature = "gpu")]
impl VitGpuOps for WgpuVitOps {
    type Buf = VitBuf;
    type Weight = WgpuVitWeight;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        let pad_len = data.len().next_multiple_of(32);
        if pad_len == data.len() {
            VitBuf::owned(self.ctx.upload_f32(data, "vit"))
        } else {
            let mut padded = data.to_vec();
            padded.resize(pad_len, 0.0);
            VitBuf::owned(self.ctx.upload_f32(&padded, "vit"))
        }
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        self.flush();
        self.ctx.download_f32(buf, len)
    }

    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight {
        use crate::tensor::DType;
        match w.dtype {
            // Streaming layout when the fp16 kernel is available (k padded to a 64-slice).
            DType::Q8_0
                if self.p_gemm_q8_stream.is_some()
                    && w.rows > 0
                    && w.cols > 0
                    && w.cols.is_multiple_of(32) =>
            {
                let (q, d) = repack_q8_0_stream(w.data(), w.rows, w.cols);
                WgpuVitWeight::QuantStream {
                    q: self
                        .ctx
                        .upload_storage(bytemuck::cast_slice(&q), "vit_wq_stream"),
                    d: self
                        .ctx
                        .upload_storage(bytemuck::cast_slice(&d), "vit_wd_stream"),
                }
            }
            DType::F16 if self.p_gemm_q8_stream.is_some() && w.rows > 0 && w.cols > 0 => {
                let (q, d) = repack_f16_as_q8_0_stream(w.data(), w.rows, w.cols);
                WgpuVitWeight::QuantStream {
                    q: self
                        .ctx
                        .upload_storage(bytemuck::cast_slice(&q), "vit_wq_stream"),
                    d: self
                        .ctx
                        .upload_storage(bytemuck::cast_slice(&d), "vit_wd_stream"),
                }
            }
            // Keep packed → quantized GEMM straight from the bytes.
            DType::Q8_0 | DType::Q4_0 => WgpuVitWeight::Quant {
                buf: self.ctx.upload_storage(w.data(), "vit_wq"),
                dtype: w.dtype,
            },
            // Dense or a quant dtype without a ViT GEMM kernel: dequantize.
            _ => WgpuVitWeight::Dense(self.ctx.upload_f32(&w.to_dense_f32(), "vit_w")),
        }
    }

    fn upload_weight_f32(&self, data: &[f32], _out_dim: usize, _in_dim: usize) -> Self::Weight {
        WgpuVitWeight::Dense(self.ctx.upload_f32(data, "vit_w"))
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        let pad_tokens = tokens.next_multiple_of(32);
        let pad_out = out_dim.next_multiple_of(32);
        let y = self.rw_buf((pad_tokens * pad_out * 4) as u64, "vit_linear_out");
        match w {
            WgpuVitWeight::Quant { buf, dtype } => {
                // `upload_weight` only builds `Quant` for Q8_0/Q4_0, so those
                // are the only dtypes reachable here.
                let pipe = match dtype {
                    crate::tensor::DType::Q8_0 => &self.p_mul_mat_q8_0,
                    crate::tensor::DType::Q4_0 => &self.p_mul_mat_q4_0,
                    other => {
                        unreachable!("WgpuVitWeight::Quant holds only Q8_0/Q4_0, got {other:?}")
                    }
                };
                self.run_mul_mat_tiled(pipe, buf, x, &y, tokens, out_dim, in_dim);
            }
            WgpuVitWeight::QuantStream { q, d } => {
                let (gemm, transpose) = (
                    self.p_gemm_q8_stream
                        .as_ref()
                        .expect("stream weight without a pipeline"),
                    self.p_transpose_f16
                        .as_ref()
                        .expect("stream weight without a pipeline"),
                );
                let k_pad = in_dim.next_multiple_of(64);
                // Activations become f16 k-major `[k_pad][pad_tokens]` for the GEMM's 32-wide
                // fibers; zero-clear when `k_pad > in_dim` so the padded rows hold 0.0h rather
                // than recycled non-finite bits.
                let b16 = if k_pad == in_dim {
                    self.rw_buf((k_pad * pad_tokens * 2) as u64, "vit_b16")
                } else {
                    self.rw_buf_zeroed((k_pad * pad_tokens * 2) as u64, "vit_b16")
                };
                let t_params: [u32; 4] = [tokens as u32, pad_tokens as u32, in_dim as u32, 0];
                let g_params: [u32; 5] = [
                    out_dim as u32,
                    k_pad as u32,
                    tokens as u32,
                    pad_tokens as u32,
                    out_dim as u32,
                ];
                let t_buf = self.params_buf(&t_params, "vit_t_params");
                let g_buf = self.params_buf(&g_params, "vit_g_params");
                self.dispatch_seq(&[
                    (
                        "vit_transpose_x",
                        transpose,
                        &[x, &b16, &t_buf][..],
                        (pad_tokens as u32 / 32, (in_dim as u32).div_ceil(32), 1),
                    ),
                    (
                        "vit_gemm_q8_stream",
                        gemm,
                        &[q, d, &b16, &y, &g_buf][..],
                        (
                            (out_dim as u32).div_ceil(256),
                            (tokens as u32).div_ceil(32),
                            1,
                        ),
                    ),
                ]);
            }
            WgpuVitWeight::Dense(buf) => {
                // MulMatParams: m, k, n, x_stride, y_stride.
                let params: [u32; 5] = [
                    out_dim as u32,
                    in_dim as u32,
                    tokens as u32,
                    in_dim as u32,
                    out_dim as u32,
                ];
                let p_buf = self.params_buf(&params, "vit_linear_params");
                // Derived from the constants, not a hardcoded tile size: this
                // dispatch and `mul_mat`'s below share one pipeline geometry.
                let wg_m = (out_dim as u32).div_ceil(VIT_MM_WG_M * VIT_MM_TILE_M);
                let wg_n = (tokens as u32).div_ceil(VIT_MM_WG_N * VIT_MM_TILE_N);
                self.dispatch(
                    "vit_linear_dense",
                    &self.p_linear,
                    &[buf, x, &y, &p_buf],
                    (wg_m, wg_n, 1),
                );
            }
        }
        y
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        self.dispatch_elements(
            "vit_bias",
            &self.p_bias,
            &[
                ElementBind::Ranged(x),
                ElementBind::Whole(bias),
                ElementBind::Params,
            ],
            rows * dim,
            dim,
            |n| vec![n, dim as u32],
        );
    }

    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        let pad_rows = rows.next_multiple_of(32);
        let pad_dim = dim.next_multiple_of(32);
        let dst = self.rw_buf((pad_rows * pad_dim * 4) as u64, "vit_ln_out");
        let params: [u32; 4] = [dim as u32, eps.to_bits(), dim as u32, dim as u32];
        let p_buf = self.params_buf(&params, "vit_ln_params");
        self.dispatch(
            "vit_layernorm",
            &self.p_layernorm,
            &[src, &dst, weight, bias, &p_buf],
            (rows as u32, 1, 1),
        );
        dst
    }

    fn gelu(&self, x: &Self::Buf, len: usize) {
        self.dispatch_elements(
            "vit_gelu",
            &self.p_gelu,
            &[ElementBind::Ranged(x), ElementBind::Params],
            len,
            1,
            |n| vec![n, 0],
        );
    }

    fn relu(&self, x: &Self::Buf, len: usize) {
        self.dispatch_elements(
            "vit_relu",
            &self.p_relu,
            &[ElementBind::Ranged(x), ElementBind::Params],
            len,
            1,
            |n| vec![n, 0],
        );
    }

    fn supports_relu(&self) -> bool {
        true
    }

    fn attention_token_limit(&self, head_dim: usize) -> usize {
        // the flash kernel where built, and the query-tiled kernel on desktop, stream K/V;
        // the scalar one does not
        if self.flash_ready(head_dim) || Self::streams_kv(head_dim) {
            usize::MAX
        } else {
            MAX_VIT_TOKENS
        }
    }

    fn max_rows(&self) -> usize {
        // the norm kernel spends one workgroup per row with no chunking (Metal's grid is u64,
        // so only wgpu overrides)
        crate::backend::wgpu::MAX_WG as usize
    }

    fn max_buffer_bytes(&self) -> u64 {
        self.ctx.max_storage_buffer_binding_size
    }

    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf {
        let dim = n_head * head_dim;
        let pad_tokens = tokens.next_multiple_of(32).max(32);
        let pad_dim = dim.next_multiple_of(32).max(32);
        let out = self.rw_buf((pad_tokens * pad_dim * 4) as u64, "vit_attn_out");
        if tokens == 0 {
            return out;
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [u32; 4] = [
            tokens as u32,
            n_head as u32,
            head_dim as u32,
            scale.to_bits(),
        ];
        let p_buf = self.params_buf(&params, "vit_attn_params");
        if let (Some(scores), Some(softmax), Some(pv), Some(cast), Some(transpose)) = (
            &self.p_attn_scores,
            &self.p_attn_softmax,
            &self.p_attn_pv,
            &self.p_cast_f16,
            &self.p_transpose_f16,
        ) && head_dim == 64
        {
            // Attention as fp16 GEMMs (see `vit_attn_*.slang`): the generic kernels below run at
            // about 19 GFLOP/s on Adreno 830, 58% of the whole tower.
            let key_pad = tokens.next_multiple_of(64);
            let plain = |len: usize, label: &'static str| self.rw_buf(len as u64, label);
            let zeroed = |len: usize, label: &'static str| self.rw_buf_zeroed(len as u64, label);
            let qt = plain(dim * pad_tokens * 2, "vit_attn_qt");
            let kt = plain(dim * pad_tokens * 2, "vit_attn_kt");
            let v16 = zeroed(key_pad * dim * 2, "vit_attn_v16");
            let st = plain(n_head * tokens * pad_tokens * 4, "vit_attn_st");
            let pt = zeroed(n_head * key_pad * pad_tokens * 2, "vit_attn_pt");
            let lsum = plain(n_head * pad_tokens * 4, "vit_attn_l");
            let up = |p: &[u32], label: &str| self.params_buf(p, label);
            let t_params = up(
                &[tokens as u32, pad_tokens as u32, dim as u32, 0],
                "vit_attn_tp",
            );
            let cast_params = up(&[(tokens * dim) as u32], "vit_attn_cp");
            let scale_log2e = scale * std::f32::consts::LOG2_E;
            let s_params = up(
                &[tokens as u32, pad_tokens as u32, scale_log2e.to_bits()],
                "vit_attn_sp",
            );
            let m_params = up(
                &[tokens as u32, pad_tokens as u32, key_pad as u32],
                "vit_attn_mp",
            );
            let pv_params = up(
                &[tokens as u32, pad_tokens as u32, key_pad as u32, dim as u32],
                "vit_attn_vp",
            );
            let tile_cols = (pad_tokens / 32) as u32;
            let steps_all = [
                (
                    "attn_transpose_q",
                    transpose,
                    &[q, &qt, &t_params][..],
                    (tile_cols, (dim as u32).div_ceil(32), 1),
                ),
                (
                    "attn_transpose_k",
                    transpose,
                    &[k, &kt, &t_params][..],
                    (tile_cols, (dim as u32).div_ceil(32), 1),
                ),
                (
                    "attn_cast_v",
                    cast,
                    &[v, &v16, &cast_params][..],
                    (((tokens * dim) as u32).div_ceil(256), 1, 1),
                ),
                (
                    "attn_scores",
                    scores,
                    &[&qt, &kt, &st, &s_params][..],
                    ((tokens as u32).div_ceil(256), tile_cols, n_head as u32),
                ),
                (
                    "attn_softmax",
                    softmax,
                    &[&st, &pt, &lsum, &m_params][..],
                    ((tokens as u32).div_ceil(256), n_head as u32, 1),
                ),
                (
                    "attn_pv",
                    pv,
                    &[&v16, &pt, &lsum, &out, &pv_params][..],
                    (n_head as u32, tile_cols, 1),
                ),
            ];
            self.dispatch_seq(&steps_all);
            return out;
        }
        // Query-tiled flash attention (one workgroup per Q_TILE=256 queries,
        // reusing K/V tiles in shared memory) when head_dim fits its shared/
        // register sizing; else the scalar per-query kernel.
        const VIT_ATTN_TILED_Q: u32 = 256;
        let use_tiled = Self::streams_kv(head_dim);
        if self.flash_ready(head_dim) {
            // the gate holds head_dim at 64, which the kernel hardwires (its params[2] is
            // informational and never read). Fail fast on a malformed call, like the trunk.
            assert!(
                head_dim == 64,
                "vit attention_flash_hd64 hardcodes head_dim 64; got {head_dim}"
            );
            let flash_pipe = self
                .p_attn_flash
                .as_ref()
                .expect("flash pipeline is built when flash_ready() holds");
            // tokens, n_head, head_dim, scale, n_kv_head (every head has its own), no window, and
            // the first query tile of the dispatch; a long call is several short dispatches
            for (first, count) in crate::backend::wgpu::GpuContext::flash_dispatch_runs(
                (tokens as u32).div_ceil(32),
                tokens,
                n_head,
            ) {
                #[cfg(test)]
                self.flash_dispatches
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let flash: [u32; 8] = [
                    tokens as u32,
                    n_head as u32,
                    head_dim as u32,
                    scale.to_bits(),
                    n_head as u32,
                    0,
                    0,
                    first,
                ];
                let flash_buf = self
                    .ctx
                    .upload_storage(bytemuck::cast_slice(&flash), "vit_attn_flash_params");
                self.dispatch(
                    "vit_attn_flash",
                    flash_pipe,
                    &[q, k, v, &out, &flash_buf],
                    (count, n_head as u32, 1),
                );
            }
        } else if use_tiled {
            self.dispatch(
                "vit_attn_tiled",
                &self.p_attn_tiled,
                &[q, k, v, &out, &p_buf],
                ((tokens as u32).div_ceil(VIT_ATTN_TILED_Q), n_head as u32, 1),
            );
        } else {
            self.dispatch(
                "vit_attn_scalar",
                &self.p_attn,
                &[q, k, v, &out, &p_buf],
                (tokens as u32, n_head as u32, 1),
            );
        }
        out
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        self.dispatch_elements(
            "vit_add",
            &self.p_add,
            &[
                ElementBind::Ranged(dst),
                ElementBind::Ranged(src),
                ElementBind::Params,
            ],
            len,
            1,
            |n| vec![n, 0],
        );
    }

    /// wgpu's `dispatch` only submits — block here so the profiler can attribute
    /// per-op GPU time. Off the profiled path this is never called.
    fn sync(&self) {
        self.flush();
        self.ctx.device.poll_wait();
    }

    fn begin_encode(&self) {
        self.flush();
        self.ctx.reset_profiler();
    }

    fn end_encode(&self) {
        self.flush();
        self.ctx.finish_profiler();
    }
}

// ── native Metal backend implementation ──────────────────────────────────────

/// A Metal linear weight `[out_dim, in_dim]` row-major. Quantized weights keep
/// their packed bytes and run the simdgroup `gemm_q8_0`/`gemm_q4_0` kernels;
/// f32 weights (or quant dtypes without a GEMM kernel) fall back to the scalar
/// `vit_linear` gemv on a dequantized f32 buffer.
///
/// Shared with the audio encoder, which makes the identical packed-versus-dense
/// decision. Alias rather than a rename so the ViT-facing name still reads at
/// the `VitGpuOps::Weight` binding below.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub type MetalVitWeight = crate::backend::metal::MetalLinearWeight;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use crate::backend::metal::MetalLinear;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use crate::backend::metal::params::{
    BiasAddParams, ElementwiseParams, LayerNormBatchParams, VitAttnParams,
};

/// Native-Metal implementation of [`VitGpuOps`]. Mirrors `WgpuVitOps` using
/// MSL kernels. Each op runs in its own command buffer and blocks on
/// `wait_until_completed`, so `download` always sees current data (unified
/// memory on Apple Silicon).
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub struct MetalVitOps {
    ctx: crate::backend::metal::MetalContext,
    linear: MetalLinear,
    p_bias: metal::ComputePipelineState,
    p_layernorm: metal::ComputePipelineState,
    p_gelu: metal::ComputePipelineState,
    p_relu: metal::ComputePipelineState,
    p_attn: metal::ComputePipelineState,
    p_attn_mma: metal::ComputePipelineState,
    p_attn_mma_hd64: metal::ComputePipelineState,
    p_add: metal::ComputePipelineState,
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl MetalVitOps {
    /// Whether the flash-attention MMA kernel takes this head dim (a multiple of 8, at most
    /// 128). Single source for the dispatch branch and the token limit.
    ///
    /// The kernel's threadgroup memory is 176·head_dim + 2144 bytes, so head_dim=256
    /// would need ~46 KB and blow the 32 KB threadgroup limit on Apple M1.
    /// 128 keeps it at ~24 KB and covers every current LFM2 ViT (hd ∈ {64,
    /// 128}); larger head dims fall back to the scalar kernel.
    fn streams_kv(head_dim: usize) -> bool {
        head_dim.is_multiple_of(8) && head_dim <= 128
    }

    pub fn new(ctx: crate::backend::metal::MetalContext) -> Result<Self> {
        use crate::backend::metal::shaders;
        Ok(Self {
            linear: MetalLinear::new(&ctx)?,
            p_bias: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add")?,
            p_layernorm: ctx.create_pipeline(shaders::LAYERNORM_BATCH, "layernorm_batch")?,
            p_gelu: ctx.create_pipeline(shaders::GELU, "gelu_inplace")?,
            p_relu: ctx.create_pipeline(shaders::ACTIVATIONS, "relu_inplace")?,
            p_attn: ctx.create_pipeline(shaders::VIT_ATTENTION, "vit_attention")?,
            p_attn_mma: ctx.create_pipeline(shaders::VIT_ATTENTION_MMA, "vit_attention_mma")?,
            p_attn_mma_hd64: ctx
                .create_pipeline(shaders::VIT_ATTENTION_MMA, "vit_attention_mma_hd64")?,
            p_add: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace")?,
            ctx,
        })
    }

    /// Bidirectional attention via the flash-attention MMA kernel
    /// (`vit_attention_mma`). Requires `head_dim % 8 == 0`; the caller falls
    /// back to the scalar `vit_attention` otherwise. `q`/`k`/`v`/`out` are
    /// `[tokens, n_head*head_dim]` f32. Threadgroup memory and grid mirror
    /// `attention_prefill`'s host dispatch (Q_PER_TG=8 queries/threadgroup).
    #[allow(clippy::too_many_arguments)]
    fn run_attn_mma(
        &self,
        q: &metal::Buffer,
        k: &metal::Buffer,
        v: &metal::Buffer,
        out: &metal::Buffer,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) {
        const Q_PER_TG: u64 = 8;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params = VitAttnParams {
            tokens: tokens as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            scale_bits: scale.to_bits(),
        };
        // q_tg + kv_tile (half) + scores + out_tg + state + rescales (f32)
        // = 2·(8+64)·hd + 4·(8·64 + 8·hd + 8·2 + 8) bytes = 176·hd + 2144.
        let shmem = 176 * head_dim as u64 + 2144;
        // hd=64 gets the constant-propagated variant; others use the runtime one.
        let pipe = if head_dim == 64 {
            &self.p_attn_mma_hd64
        } else {
            &self.p_attn_mma
        };
        self.ctx.run_kernel_shmem(
            pipe,
            &[q, k, v, out],
            &params,
            metal::MTLSize::new(n_head as u64 * (tokens as u64).div_ceil(Q_PER_TG), 1, 1),
            metal::MTLSize::new(256, 1, 1),
            Some(shmem),
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl VitGpuOps for MetalVitOps {
    type Buf = metal::Buffer;
    type Weight = MetalVitWeight;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        self.ctx.upload_f32(data)
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        self.ctx.read_f32(buf, len)
    }

    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight {
        self.ctx.upload_linear_weight(w)
    }

    fn upload_weight_f32(&self, data: &[f32], _out_dim: usize, _in_dim: usize) -> Self::Weight {
        MetalVitWeight::Dense(self.ctx.upload_f32(data))
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        self.linear
            .forward(&self.ctx, x, w, tokens, out_dim, in_dim)
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        let total = (rows * dim) as u32;
        let params = BiasAddParams {
            total,
            dim: dim as u32,
        };
        self.ctx.run_kernel(
            &self.p_bias,
            &[x, bias],
            &params,
            metal::MTLSize::new(total.div_ceil(256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        let dst = self.ctx.create_buffer((rows * dim * 4) as u64);
        let params = LayerNormBatchParams {
            n: dim as u32,
            eps_bits: eps.to_bits(),
            src_stride: dim as u32,
            dst_stride: dim as u32,
        };
        self.ctx.run_kernel(
            &self.p_layernorm,
            &[src, &dst, weight, bias],
            &params,
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        dst
    }

    fn gelu(&self, x: &Self::Buf, len: usize) {
        self.ctx.run_kernel(
            &self.p_gelu,
            &[x],
            &ElementwiseParams::new(len as u32),
            metal::MTLSize::new((len as u64).div_ceil(256), 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    fn relu(&self, x: &Self::Buf, len: usize) {
        self.ctx.run_kernel(
            &self.p_relu,
            &[x],
            &ElementwiseParams::new(len as u32),
            metal::MTLSize::new((len as u64).div_ceil(256), 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    fn supports_relu(&self) -> bool {
        true
    }

    fn attention_token_limit(&self, head_dim: usize) -> usize {
        // the flash MMA kernel streams K/V; the scalar fallback does not
        if Self::streams_kv(head_dim) {
            usize::MAX
        } else {
            MAX_VIT_TOKENS
        }
    }

    fn max_buffer_bytes(&self) -> u64 {
        self.ctx.device.max_buffer_length()
    }

    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf {
        let dim = n_head * head_dim;
        let out = self.ctx.create_buffer((tokens * dim * 4) as u64);
        if Self::streams_kv(head_dim) {
            self.run_attn_mma(q, k, v, &out, tokens, n_head, head_dim);
        } else {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let params = VitAttnParams {
                tokens: tokens as u32,
                n_head: n_head as u32,
                head_dim: head_dim as u32,
                scale_bits: scale.to_bits(),
            };
            self.ctx.run_kernel(
                &self.p_attn,
                &[q, k, v, &out],
                &params,
                metal::MTLSize::new(tokens as u64, n_head as u64, 1),
                metal::MTLSize::new(256, 1, 1),
            );
        }
        out
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        self.ctx.run_kernel(
            &self.p_add,
            &[dst, src],
            &ElementwiseParams::new(len as u32),
            metal::MTLSize::new((len as u64).div_ceil(256), 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }
}

// ── Cached, object-safe encoder for the live session path ────────────────────

/// Object-safe GPU vision encoder cached in a [`crate::session::Session`].
/// Wraps a backend's ops + uploaded weights so the whole ViT runs on the GPU.
/// Implementors are `Send + Sync` so the engine can share them across sessions.
pub trait VisionGpuEncode: Send + Sync {
    /// The backend this encoder runs on: `metal`, `wgpu` or `hexagon`.
    fn backend_label(&self) -> &'static str {
        "gpu"
    }

    /// Encode preprocessed pixels (`[3·H·W]` NCHW, normalized) at the given
    /// patch grid. Output matches [`VisionEncoderWeights::encode_image`].
    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>>;

    /// Async variant for environments (like browser WebGPU on wasm32) where
    /// GPU readbacks cannot block the main/worker thread. It is single-threaded by contract: only the
    /// wasm worker calls it, and unlike [`Self::encode_image`] the wgpu implementation does not take the
    /// encoder's serialization lock, so a native caller must not drive two at once.
    fn encode_image_async<'a>(
        &'a self,
        pixels: &'a [f32],
        grid_w: usize,
        grid_h: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move { self.encode_image(pixels, grid_w, grid_h) })
    }
}

#[cfg(feature = "gpu")]
struct WgpuVisionEncoder {
    ops: WgpuVitOps,
    weights: GpuVitWeights<WgpuVitOps>,
    /// Serializes `encode_image`. The engine shares one encoder across every session, and an encode
    /// records its passes into the encoder-wide `pending` encoder and draws from the encoder-wide buffer
    /// pool: two concurrent encodes would flush each other's half-recorded passes and reuse buffers the
    /// other has not finished with, returning wrong embeddings as `Ok`. The wasm async path is
    /// single-threaded and flushes before every await, so it does not take this lock.
    encode_lock: std::sync::Mutex<()>,
}

#[cfg(feature = "gpu")]
impl VisionGpuEncode for WgpuVisionEncoder {
    fn backend_label(&self) -> &'static str {
        "wgpu"
    }

    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        let _serial = self.encode_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Drain the readback slot around the encode (mirrors Session's
        // discard-at-entry/drain-after-work discipline): a map/range fault
        // mid-encode otherwise returns Ok(zero embeddings) that the
        // session prefills as valid, while the Err arm's CPU fallback
        // (built for exactly this fault class) never fires.
        let _ = self.ops.ctx.take_readback_fault();
        let out = encode_image_gpu(&self.ops, &self.weights, pixels, grid_w, grid_h)?;
        if let Some(e) = self.ops.ctx.take_readback_fault() {
            return Err(e.into());
        }
        Ok(out)
    }

    fn encode_image_async<'a>(
        &'a self,
        pixels: &'a [f32],
        grid_w: usize,
        grid_h: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move {
            let _ = self.ops.ctx.take_readback_fault();
            let res =
                encode_image_gpu_wgpu_async(&self.ops, &self.weights, pixels, grid_w, grid_h).await;
            if let Some(e) = self.ops.ctx.take_readback_fault() {
                return Err(e.into());
            }
            res
        })
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
struct MetalVisionEncoder {
    ops: MetalVitOps,
    weights: GpuVitWeights<MetalVitOps>,
    /// One encode at a time, as in `WgpuVisionEncoder`: the engine shares this encoder across sessions
    /// and the command-error slot drained around each encode belongs to the whole context, so a second
    /// encode's entry drain could swallow the first one's commit fault.
    encode_lock: std::sync::Mutex<()>,
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl VisionGpuEncode for MetalVisionEncoder {
    fn backend_label(&self) -> &'static str {
        "metal"
    }

    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        let _serial = self.encode_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Drain the command-error slot around the encode (mirrors
        // Session's discard-at-entry/drain-after-work discipline): a
        // commit fault mid-encode otherwise returns Ok(stale embeddings)
        // that the session prefills as valid, while the Err arm's CPU fallback (built for
        // exactly this fault class) never fires.
        let _ = self.ops.ctx.take_cmd_error();
        let out = encode_image_gpu(&self.ops, &self.weights, pixels, grid_w, grid_h)?;
        if let Some(e) = self.ops.ctx.take_cmd_error() {
            return Err(e.into());
        }
        Ok(out)
    }
}

// ── A ViT block stack with caller-supplied weights ───────────────────────────

/// One pre-norm ViT block as the caller holds it: the norm and bias vectors on the host, the
/// linear weights as tensors of the model file (kept packed on the GPU when they are Q8_0 or
/// Q4_0).
pub struct VitStackBlock {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub q: MmapWeight,
    pub q_b: Vec<f32>,
    pub k: MmapWeight,
    pub k_b: Vec<f32>,
    pub v: MmapWeight,
    pub v_b: Vec<f32>,
    pub o: MmapWeight,
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub up: MmapWeight,
    pub up_b: Vec<f32>,
    pub down: MmapWeight,
    pub down_b: Vec<f32>,
}

/// A stack of pre-norm transformer blocks and an optional final LayerNorm: for the ViT, everything
/// between the position-embedded patch tokens and the tokens the projector reads. LayerNorm and
/// unmasked attention are those of [`encode_image_gpu`]; only where the weights come from and the
/// feed-forward activation differ, so a model whose patch embedding, positions or projector
/// differ from LFM2-VL's can still use it, and so can a decision head with ReLU blocks.
pub struct VitStackSpec {
    pub width: usize,
    pub heads: usize,
    pub ffn: usize,
    pub eps: f32,
    /// What sits between a block's two feed-forward projections.
    pub activation: VitStackActivation,
    pub blocks: Vec<VitStackBlock>,
    /// The final LayerNorm's weight and bias, when the stack ends in one.
    pub post: Option<(Vec<f32>, Vec<f32>)>,
    /// A projector for [`VitStack::run_projector`], for the backends that can run one beside the
    /// blocks; the others ignore it and the caller projects.
    pub projector: Option<VitStackProjector>,
}

/// The projector of a [`VitStackSpec`]: `Linear(GELU(Linear(x)))` per row.
///
/// The GELU is the exact (erf) form of `torch.nn.GELU()`, on every backend: the Hexagon stack
/// runs the skels' native erf op (measured on a device against the exact GELU: worst error
/// 4.8e-7 over +-12). A backend that cannot be exact returns `Ok(None)` from
/// [`VitStack::run_projector`] and the caller projects on the host.
pub struct VitStackProjector {
    pub mm1: MmapWeight,
    pub mm1_b: Vec<f32>,
    pub mm2: MmapWeight,
    pub mm2_b: Vec<f32>,
}

/// The feed-forward activation of a [`VitStackSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VitStackActivation {
    /// The tanh approximation SigLIP and the other ViTs use.
    GeluTanh,
    /// The plain rectifier of `torch.nn.TransformerEncoderLayer`'s default.
    Relu,
}

/// A [`VitStackSpec`] uploaded to a GPU, ready to run.
pub trait VitStack: Send + Sync {
    /// Run the blocks (and the final LayerNorm, if any) over `tokens` rows of `x` (`[tokens, width]`).
    ///
    /// # Errors
    ///
    /// Fails when `tokens` is over what the attention kernel holds or a pass needs a buffer the
    /// device cannot make, or the device faults; the caller falls back to the CPU.
    fn run(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>>;

    /// Run the spec's projector (two linears with a GELU between, see
    /// [`VitStackSpec::projector`]) over `pooled` rows (`[n, first linear's input width]`), when
    /// this backend holds one. `Ok(None)` leaves the projector to the caller.
    ///
    /// # Errors
    ///
    /// Fails when the device faults; the caller then projects on the host.
    fn run_projector(&self, pooled: &[f32]) -> Result<Option<Vec<f32>>> {
        let _ = pooled;
        Ok(None)
    }
}

#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
struct StackBlock<O: VitGpuOps> {
    ln1_w: O::Buf,
    ln1_b: O::Buf,
    q: O::Weight,
    q_b: O::Buf,
    k: O::Weight,
    k_b: O::Buf,
    v: O::Weight,
    v_b: O::Buf,
    o: O::Weight,
    o_b: O::Buf,
    ln2_w: O::Buf,
    ln2_b: O::Buf,
    up: O::Weight,
    up_b: O::Buf,
    down: O::Weight,
    down_b: O::Buf,
}

#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
struct GpuStack<O: VitGpuOps> {
    ops: O,
    width: usize,
    heads: usize,
    ffn: usize,
    eps: f32,
    activation: VitStackActivation,
    blocks: Vec<StackBlock<O>>,
    post: Option<(O::Buf, O::Buf)>,
}

#[cfg(any(
    feature = "gpu",
    feature = "hexagon",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
/// The caller-supplied public spec is enforced before uploading: a zero head count would
/// divide by zero in `run`, and a ragged geometry or short vector would misread on the
/// device. Pure so a host test can pin it without a device. Every device builder runs it,
/// the Hexagon stack included.
pub(crate) fn check_spec(spec: &VitStackSpec) -> Result<()> {
    anyhow::ensure!(
        spec.width > 0 && spec.ffn > 0,
        "width {} and ffn {} must be positive",
        spec.width,
        spec.ffn
    );
    anyhow::ensure!(
        spec.heads > 0 && spec.width.is_multiple_of(spec.heads),
        "width {} is not a whole number of {} heads",
        spec.width,
        spec.heads
    );
    if let Some((w, b)) = &spec.post {
        anyhow::ensure!(
            w.len() == spec.width && b.len() == spec.width,
            "the final norm is not {} wide",
            spec.width
        );
    }
    for (i, b) in spec.blocks.iter().enumerate() {
        for (name, w, rows, cols) in [
            ("q", &b.q, spec.width, spec.width),
            ("k", &b.k, spec.width, spec.width),
            ("v", &b.v, spec.width, spec.width),
            ("o", &b.o, spec.width, spec.width),
            ("up", &b.up, spec.ffn, spec.width),
            ("down", &b.down, spec.width, spec.ffn),
        ] {
            anyhow::ensure!(
                w.rows == rows && w.cols == cols,
                "block {i} {name} is {} x {}, expected {rows} x {cols}",
                w.rows,
                w.cols
            );
            anyhow::ensure!(
                w.cols.is_multiple_of(w.dtype.block_size()),
                "block {i} {name}: {} columns are not whole {:?} blocks",
                w.cols,
                w.dtype
            );
            // the uploads trust the payload blindly (packed weights go whole, the dense path
            // short-copies), so a short buffer must refuse here rather than misread on-device
            let row_bytes = w.cols / w.dtype.block_size() * w.dtype.block_bytes();
            let want = (w.rows as u64) * (row_bytes as u64);
            anyhow::ensure!(
                w.data().len() as u64 == want,
                "block {i} {name} holds {} bytes, expected {want} for {} x {} {:?}",
                w.data().len(),
                w.rows,
                w.cols,
                w.dtype
            );
        }
        for (name, v, want) in [
            ("ln1_w", &b.ln1_w, spec.width),
            ("ln1_b", &b.ln1_b, spec.width),
            ("q_b", &b.q_b, spec.width),
            ("k_b", &b.k_b, spec.width),
            ("v_b", &b.v_b, spec.width),
            ("o_b", &b.o_b, spec.width),
            ("ln2_w", &b.ln2_w, spec.width),
            ("ln2_b", &b.ln2_b, spec.width),
            ("up_b", &b.up_b, spec.ffn),
            ("down_b", &b.down_b, spec.width),
        ] {
            anyhow::ensure!(
                v.len() == want,
                "block {i} {name} has {} values, expected {want}",
                v.len()
            );
        }
    }
    Ok(())
}

#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl<O: VitGpuOps> GpuStack<O> {
    fn build(ops: O, spec: &VitStackSpec) -> Result<Self> {
        check_spec(spec)?;
        if spec.activation == VitStackActivation::Relu {
            anyhow::ensure!(
                ops.supports_relu(),
                "this backend has no ReLU kernel; caller should fall back to CPU"
            );
        }
        let blocks = spec
            .blocks
            .iter()
            .map(|b| StackBlock {
                ln1_w: ops.upload(&b.ln1_w),
                ln1_b: ops.upload(&b.ln1_b),
                q: ops.upload_weight(&b.q),
                q_b: ops.upload(&b.q_b),
                k: ops.upload_weight(&b.k),
                k_b: ops.upload(&b.k_b),
                v: ops.upload_weight(&b.v),
                v_b: ops.upload(&b.v_b),
                o: ops.upload_weight(&b.o),
                o_b: ops.upload(&b.o_b),
                ln2_w: ops.upload(&b.ln2_w),
                ln2_b: ops.upload(&b.ln2_b),
                up: ops.upload_weight(&b.up),
                up_b: ops.upload(&b.up_b),
                down: ops.upload_weight(&b.down),
                down_b: ops.upload(&b.down_b),
            })
            .collect();
        Ok(Self {
            width: spec.width,
            heads: spec.heads,
            ffn: spec.ffn,
            eps: spec.eps,
            activation: spec.activation,
            blocks,
            post: spec
                .post
                .as_ref()
                .map(|(w, b)| (ops.upload(w), ops.upload(b))),
            ops,
        })
    }

    fn run(&self, x: &[f32], n: usize) -> Result<Vec<f32>> {
        let ops = &self.ops;
        let (d, ff) = (self.width, self.ffn);
        anyhow::ensure!(
            n > 0 && x.len() == n * d,
            "{} values are not {n} tokens of {d}",
            x.len()
        );
        let limit = ops.attention_token_limit(d / self.heads);
        anyhow::ensure!(
            n <= limit,
            "{n} tokens exceeds the GPU attention limit ({limit}); caller should fall back to CPU"
        );
        let max_rows = ops.max_rows();
        anyhow::ensure!(
            n <= max_rows,
            "{n} tokens exceeds this backend's per-dispatch row limit ({max_rows}); caller should fall back to CPU"
        );
        // the widest tensor of a pass is the feed-forward's intermediate or one [n, d]
        // activation, padded the way upload/linear/attention pad (multiples of 32)
        let widest = (n.next_multiple_of(32) * ff.max(d).next_multiple_of(32) * 4) as u64;
        anyhow::ensure!(
            widest <= ops.max_buffer_bytes(),
            "a {n}-token pass needs a {widest} byte buffer, more than the device allows ({}); \
             caller should fall back to CPU",
            ops.max_buffer_bytes()
        );
        let tokens = ops.upload(x);
        for b in &self.blocks {
            let normed = ops.layernorm(&tokens, &b.ln1_w, &b.ln1_b, self.eps, n, d);
            let q = ops.linear(&normed, &b.q, n, d, d);
            ops.bias_add(&q, &b.q_b, n, d);
            let k = ops.linear(&normed, &b.k, n, d, d);
            ops.bias_add(&k, &b.k_b, n, d);
            let v = ops.linear(&normed, &b.v, n, d, d);
            ops.bias_add(&v, &b.v_b, n, d);
            let attn = ops.attention(&q, &k, &v, n, self.heads, d / self.heads);
            let proj = ops.linear(&attn, &b.o, n, d, d);
            ops.bias_add(&proj, &b.o_b, n, d);
            ops.add(&tokens, &proj, n * d);
            let normed = ops.layernorm(&tokens, &b.ln2_w, &b.ln2_b, self.eps, n, d);
            let mid = ops.linear(&normed, &b.up, n, ff, d);
            ops.bias_add(&mid, &b.up_b, n, ff);
            match self.activation {
                VitStackActivation::GeluTanh => ops.gelu(&mid, n * ff),
                VitStackActivation::Relu => ops.relu(&mid, n * ff),
            }
            let down = ops.linear(&mid, &b.down, n, d, ff);
            ops.bias_add(&down, &b.down_b, n, d);
            ops.add(&tokens, &down, n * d);
        }
        let out = match &self.post {
            Some((w, b)) => ops.layernorm(&tokens, w, b, self.eps, n, d),
            None => tokens,
        };
        Ok(ops.download(&out, n * d))
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl VitStack for GpuStack<MetalVitOps> {
    fn run(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        // Same fault discipline as `MetalVisionEncoder`: a faulted commit must not return
        // stale tokens as if they were the result.
        let _ = self.ops.ctx.take_cmd_error();
        let out = GpuStack::run(self, x, tokens)?;
        if let Some(e) = self.ops.ctx.take_cmd_error() {
            return Err(e.into());
        }
        Ok(out)
    }
}

#[cfg(feature = "gpu")]
impl VitStack for GpuStack<WgpuVitOps> {
    fn run(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        let _ = self.ops.ctx.take_readback_fault();
        let out = GpuStack::run(self, x, tokens)?;
        if let Some(e) = self.ops.ctx.take_readback_fault() {
            return Err(e.into());
        }
        Ok(out)
    }
}

/// Upload `spec` to the GPU that `backend` names: `Metal`, `Gpu`, or `Auto` (Metal, then
/// wgpu). Returns `None` for `Cpu`, `Hexagon`, and `Npu`, when the named feature is not
/// compiled, or when no device could be opened.
pub fn build_vit_stack(
    spec: &VitStackSpec,
    backend: crate::engine::BackendPreference,
) -> Option<std::sync::Arc<dyn VitStack>> {
    use crate::engine::BackendPreference as BP;
    match backend {
        BP::Metal => try_metal_stack(spec),
        BP::Gpu => try_wgpu_stack(spec, false),
        BP::Auto => try_metal_stack(spec)
            .or_else(|| try_hexagon_stack(spec, MAX_VIT_TOKENS))
            .or_else(|| try_wgpu_stack(spec, false)),
        BP::Hexagon | BP::Npu => try_hexagon_stack(spec, MAX_VIT_TOKENS),
        BP::Cpu => None,
    }
}

/// [`build_vit_stack`] for a stack that sees long sequences, which only pays on a GPU whose
/// attention is fast: Metal, or wgpu where the adapter runs the register-tiled flash kernel
/// (head_dim 64, see `crate::backend::wgpu::GpuContext::uses_flash_attention`). The scalar
/// wgpu attention kernels are slower than the host's blocked kernel on a long sequence, so
/// elsewhere this returns `None` and the caller stays on the host.
pub fn build_vit_stack_fast(
    spec: &VitStackSpec,
    backend: crate::engine::BackendPreference,
) -> Option<std::sync::Arc<dyn VitStack>> {
    use crate::engine::BackendPreference as BP;
    match backend {
        BP::Metal => try_metal_stack(spec),
        BP::Gpu => try_wgpu_stack(spec, true),
        BP::Auto => try_metal_stack(spec)
            .or_else(|| try_hexagon_stack(spec, LONG_STACK_ROWS))
            .or_else(|| try_wgpu_stack(spec, true)),
        BP::Hexagon | BP::Npu => try_hexagon_stack(spec, LONG_STACK_ROWS),
        BP::Cpu => None,
    }
}

/// A stack backend that opens but then fails (pipelines, spec) must be visible, since the
/// only other symptom is "the tower is slow". `tracing` alone is not enough: no subscriber
/// exists on the shipping mobile/FFI platforms, so the warning is paired with a stderr line.
/// (An absent device is ordinary Auto fallthrough and stays at debug at the call site.)
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
fn warn_stack_unavailable(what: &str, e: impl std::fmt::Display) {
    tracing::warn!("vit stack: {what} unavailable: {e:#}");
    eprintln!("vit stack: {what} unavailable: {e:#}");
}

/// Renamed to [`build_vit_stack_fast`]: `Gpu` now returns a stack when the adapter runs the
/// flash kernel instead of always returning `None`.
#[deprecated(
    since = "0.7.0",
    note = "renamed to build_vit_stack_fast; Gpu now returns a stack when the adapter runs the flash kernel"
)]
pub fn build_vit_stack_native(
    spec: &VitStackSpec,
    backend: crate::engine::BackendPreference,
) -> Option<std::sync::Arc<dyn VitStack>> {
    build_vit_stack_fast(spec, backend)
}

/// Tokens per pass of a stack that sees long sequences on the NPU (the d1 decision head over an
/// image prefix): its scratch is about 52 KB a row at the head's widths, so this is about 210 MB.
pub(crate) const LONG_STACK_ROWS: usize = 4096;

#[cfg(feature = "hexagon")]
fn try_hexagon_stack(spec: &VitStackSpec, rows: usize) -> Option<std::sync::Arc<dyn VitStack>> {
    crate::model::vision_encoder_hexagon::try_hexagon_vit_stack(spec, rows)
}

#[cfg(not(feature = "hexagon"))]
fn try_hexagon_stack(_spec: &VitStackSpec, _rows: usize) -> Option<std::sync::Arc<dyn VitStack>> {
    None
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn try_metal_stack(spec: &VitStackSpec) -> Option<std::sync::Arc<dyn VitStack>> {
    // an absent device is ordinary Auto fallthrough (the trunk logs it at debug); only a
    // backend that opens but then fails (pipelines, spec) is a warn-worthy refusal
    let ctx = match crate::backend::metal::MetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            tracing::debug!("vit stack: Metal backend unavailable ({e:#}); falling back");
            return None;
        }
    };
    let ops = match MetalVitOps::new(ctx) {
        Ok(ops) => ops,
        Err(e) => {
            warn_stack_unavailable("Metal pipelines", e);
            return None;
        }
    };
    match GpuStack::build(ops, spec) {
        Ok(stack) => {
            tracing::info!("vit stack: using native Metal backend");
            Some(std::sync::Arc::new(stack))
        }
        Err(e) => {
            warn_stack_unavailable("Metal stack spec", e);
            None
        }
    }
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn try_metal_stack(_spec: &VitStackSpec) -> Option<std::sync::Arc<dyn VitStack>> {
    None
}

/// A wgpu stack; with `need_flash`, only on an adapter that attends with the flash kernel.
#[cfg(feature = "gpu")]
fn try_wgpu_stack(spec: &VitStackSpec, need_flash: bool) -> Option<std::sync::Arc<dyn VitStack>> {
    // an absent device is ordinary Auto fallthrough (the trunk logs it at debug); only a
    // backend that opens but then fails (pipelines, spec) is a warn-worthy refusal
    let ctx = match crate::backend::wgpu::GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            tracing::debug!("vit stack: wgpu backend unavailable ({e:#}); falling back");
            return None;
        }
    };
    let head_dim = spec.width / spec.heads.max(1);
    if need_flash && !ctx.uses_flash_attention(head_dim) {
        return None;
    }
    let ops = match WgpuVitOps::new(ctx) {
        Ok(ops) => ops,
        Err(e) => {
            warn_stack_unavailable("wgpu pipelines", e);
            return None;
        }
    };
    if need_flash && !ops.flash_ready(head_dim) {
        // the gate passed but the flash pipeline failed to build: the slow path is not what
        // this stack is for, so decline and let the caller stay on the host.
        return None;
    }
    // A VitStack evaluates layers in f32 (the decision head scores are compared to host
    // layers to 2e-3); the fp16 streaming pipeline is for WgpuVisionEncoder image encoding.
    let ops = ops.with_f32_attention();
    match GpuStack::build(ops, spec) {
        Ok(stack) => {
            tracing::info!("vit stack: using wgpu GPU backend");
            Some(std::sync::Arc::new(stack))
        }
        Err(e) => {
            warn_stack_unavailable("wgpu stack spec", e);
            None
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn try_wgpu_stack(_spec: &VitStackSpec, _need_flash: bool) -> Option<std::sync::Arc<dyn VitStack>> {
    None
}

/// Build a cached GPU vision encoder for `weights`, honoring `backend`.
/// Returns `None` for `Cpu`, when the chosen backend's feature isn't compiled,
/// when the device/context can't be created, or (wgpu, native only) when the blank-image warm-up
/// encode fails: the caller then falls back to the CPU encoder. `Auto` prefers Metal, then Hexagon, then wgpu.
pub fn build_gpu_vision_encoder(
    weights: &VisionEncoderWeights,
    backend: crate::engine::BackendPreference,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    use crate::engine::BackendPreference as BP;
    match backend {
        BP::Cpu => None,
        // Hexagon is the only NPU with a vision encoder today; `Npu` takes the
        // same path and gains other vendors here as they grow one.
        BP::Hexagon | BP::Npu => {
            #[cfg(feature = "hexagon")]
            {
                crate::model::vision_encoder_hexagon::try_hexagon_vision_encoder(weights)
            }
            #[cfg(not(feature = "hexagon"))]
            {
                None
            }
        }
        BP::Metal => try_metal_vision_encoder(weights),
        BP::Gpu => try_wgpu_vision_encoder(weights),
        BP::Auto => try_metal_vision_encoder(weights)
            .or_else(|| {
                #[cfg(feature = "hexagon")]
                {
                    crate::model::vision_encoder_hexagon::try_hexagon_vision_encoder(weights)
                }
                #[cfg(not(feature = "hexagon"))]
                {
                    None
                }
            })
            .or_else(|| try_wgpu_vision_encoder(weights)),
    }
}

#[cfg(feature = "gpu")]
pub fn build_wgpu_vision_encoder_with_context(
    ctx: crate::backend::wgpu::GpuContext,
    weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    let ops = WgpuVitOps::new(ctx).ok()?;
    let gpu_w = GpuVitWeights::build(&ops, weights);
    let encoder = WgpuVisionEncoder {
        ops,
        weights: gpu_w,
        encode_lock: std::sync::Mutex::new(()),
    };
    // Not on wasm32: the warm-up encode ends in a blocking `download_f32`, which waits on a map
    // callback that only the JS event loop can run, so it would hang the worker at load.
    #[cfg(not(target_arch = "wasm32"))]
    if !warm_up(&encoder, &weights.config) {
        return None;
    }
    tracing::info!("vision encoder: using wgpu GPU backend");
    Some(std::sync::Arc::new(encoder))
}

/// Encode a blank 512x384 image once, so the first real image does not pay for one-time setup: driver
/// work on first use of each pipeline and the pooled buffers' first allocation, together about 80 ms on
/// Adreno 830. llama.cpp's mtmd runs a warm-up encode for the same reason. Costs about 190 ms once, at
/// load; `CERA_VIT_WARMUP=0` skips it. Under `CERA_VIT_PROFILE` its own profile report is printed
/// first. Returns `false` when the encode fails: a tower that cannot encode a blank image will fail every
/// real one, so the caller drops the GPU encoder and the session uses the CPU one, instead of paying a
/// failed GPU attempt (and a warning) for each tile of each image.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn warm_up(encoder: &WgpuVisionEncoder, cfg: &VisionEncoderConfig) -> bool {
    if crate::backend::cpu_features::env_disabled("CERA_VIT_WARMUP") {
        return true;
    }
    warm_up_with(cfg, |pixels, grid_w, grid_h| {
        encoder.encode_image(pixels, grid_w, grid_h)
    })
}

/// The decision behind [`warm_up`], with the encode passed in so it can be tested without a GPU:
/// `true` to keep the encoder (the blank image encoded, or the grid does not fit this config so no
/// warm-up ran), `false` to drop it (the encode failed).
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn warm_up_with(
    cfg: &VisionEncoderConfig,
    encode: impl FnOnce(&[f32], usize, usize) -> Result<Vec<f32>>,
) -> bool {
    let (grid_w, grid_h) = (32usize, 24usize);
    if !grid_w.is_multiple_of(cfg.scale_factor) || !grid_h.is_multiple_of(cfg.scale_factor) {
        return true;
    }
    let pixels = vec![0.0f32; 3 * grid_w * cfg.patch_size * grid_h * cfg.patch_size];
    let start = crate::time::Instant::now();
    match encode(&pixels, grid_w, grid_h) {
        Ok(_) => {
            tracing::debug!(
                "vision encoder warm-up took {:.0} ms",
                start.elapsed().as_secs_f64() * 1e3
            );
            true
        }
        Err(e) => {
            tracing::warn!("vision encoder warm-up failed, using the CPU vision encoder: {e:#}");
            false
        }
    }
}

#[cfg(feature = "gpu")]
fn try_wgpu_vision_encoder(
    weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    let ctx = crate::backend::wgpu::GpuContext::new().ok()?;
    build_wgpu_vision_encoder_with_context(ctx, weights)
}

#[cfg(not(feature = "gpu"))]
fn try_wgpu_vision_encoder(
    _weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    None
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn try_metal_vision_encoder(
    weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    let ctx = crate::backend::metal::MetalContext::new().ok()?;
    let ops = MetalVitOps::new(ctx).ok()?;
    let gpu_w = GpuVitWeights::build(&ops, weights);
    tracing::info!("vision encoder: using native Metal backend");
    Some(std::sync::Arc::new(MetalVisionEncoder {
        ops,
        weights: gpu_w,
        encode_lock: std::sync::Mutex::new(()),
    }))
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn try_metal_vision_encoder(
    _weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    None
}

#[cfg(all(
    test,
    any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    )
))]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    use crate::backend::wgpu::{gpu_or_skip, require_flash_or_skip};
    use crate::model::vision_encoder::{PatchEmbedWeights, ProjectorWeights, VitBlockWeights};
    use crate::model::weights::MmapWeight;
    use crate::tensor::DType;

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    fn require_metal(err: impl std::fmt::Display) {
        assert!(
            std::env::var("CERA_REQUIRE_METAL").as_deref() != Ok("1"),
            "CERA_REQUIRE_METAL=1 but no Metal device is available: {err}"
        );
    }

    /// Deterministic pseudo-random f32s in roughly [-0.5, 0.5].
    fn rnd(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i + seed) * 1103515245 + 12345) % 1000) as f32 / 1000.0 - 0.5)
            .collect()
    }

    fn f32_weight(rows: usize, cols: usize, seed: usize) -> MmapWeight {
        let data = rnd(rows * cols, seed);
        MmapWeight::from_owned_bytes(bytemuck::cast_slice(&data).to_vec(), DType::F32, rows, cols)
    }

    /// Quantize a `[rows, cols]` row-major f32 weight to packed Q8_0 (34 bytes
    /// per 32-element block: f16 scale + 32 int8) and wrap as an `MmapWeight` —
    /// exercises the quantized GEMM path. `cols` must be a multiple of 32.
    fn q8_0_weight(rows: usize, cols: usize, seed: usize) -> MmapWeight {
        assert_eq!(cols % 32, 0, "Q8_0 cols must be a multiple of 32");
        let data = rnd(rows * cols, seed);
        let mut bytes = Vec::with_capacity(rows * (cols / 32) * 34);
        for block in data.as_chunks::<32>().0 {
            let amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for &x in block {
                bytes.push((x * id).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        MmapWeight::from_owned_bytes(bytes, DType::Q8_0, rows, cols)
    }

    /// Quantize a `[rows, cols]` row-major f32 weight to packed Q4_0 (18 bytes
    /// per 32-element block: f16 scale + 16 packed bytes, low nibbles → lanes
    /// 0..16, high nibbles → 16..32, offset −8) and wrap as an `MmapWeight` —
    /// exercises the `gemm_q4_0` GEMM path. `cols` must be a multiple of 32.
    fn q4_0_weight(rows: usize, cols: usize, seed: usize) -> MmapWeight {
        assert_eq!(cols % 32, 0, "Q4_0 cols must be a multiple of 32");
        let data = rnd(rows * cols, seed);
        let mut bytes = Vec::with_capacity(rows * (cols / 32) * 18);
        for block in data.as_chunks::<32>().0 {
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 7.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for qi in 0..16 {
                let lo = ((block[qi] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                let hi = ((block[qi + 16] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                bytes.push(lo | (hi << 4));
            }
        }
        MmapWeight::from_owned_bytes(bytes, DType::Q4_0, rows, cols)
    }

    /// Build a tiny synthetic VL encoder for CPU↔GPU parity. With `quant`, the
    /// linear weights (q/k/v/o, ffn, projector) are quantized to that dtype to
    /// exercise the quantized GEMM; norms/biases/conv stay f32. `None` keeps
    /// everything f32.
    fn synth_encoder() -> VisionEncoderWeights {
        synth_encoder_quant(None)
    }

    /// A failed warm-up encode drops the encoder (the session then uses the CPU one); a successful one
    /// keeps it, and a config whose scale factor does not divide the warm-up grid skips the encode and
    /// keeps it. The encode is a stand-in, so no GPU is needed.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[test]
    fn warm_up_drops_the_encoder_only_when_the_encode_fails() {
        let mut cfg = synth_encoder().config;
        let calls = std::cell::Cell::new(0usize);
        let seen = std::cell::Cell::new((0usize, 0usize, 0usize));
        let ok = |p: &[f32], w: usize, h: usize| {
            calls.set(calls.get() + 1);
            seen.set((p.len(), w, h));
            Ok(vec![0.0f32])
        };
        assert!(warm_up_with(&cfg, ok));
        assert_eq!(calls.get(), 1);
        let (len, w, h) = seen.get();
        assert_eq!(
            (w, h, len),
            (32, 24, 3 * 32 * cfg.patch_size * 24 * cfg.patch_size)
        );

        let fail = |_: &[f32], _: usize, _: usize| Err(anyhow::anyhow!("device lost"));
        assert!(
            !warm_up_with(&cfg, fail),
            "a failed encode must drop the encoder"
        );

        // 5 divides neither 32 nor 24: no warm-up runs and the encoder is kept.
        cfg.scale_factor = 5;
        calls.set(0);
        assert!(warm_up_with(&cfg, ok));
        assert_eq!(calls.get(), 0);
    }

    fn synth_encoder_quant(quant: Option<DType>) -> VisionEncoderWeights {
        let lin = |rows: usize, cols: usize, seed: usize| match quant {
            Some(DType::Q8_0) => q8_0_weight(rows, cols, seed),
            Some(DType::Q4_0) => q4_0_weight(rows, cols, seed),
            Some(d) => panic!("synth_encoder_quant: unsupported dtype {d:?}"),
            None => f32_weight(rows, cols, seed),
        };
        // All linear in_dims (k) must be multiples of 32 — the quant block size
        // asserted by `q8_0_weight`/`q4_0_weight` above, NOT the matmul's TILE_K (the
        // kernel handles a ragged final k-tile; `test_gpu_mul_mat_tile_f32_ragged_k_parity`
        // covers that):
        //   patch in_dim = 3·patch_size² = 192; q/k/v/o/ffn_up = n_embd = 32;
        //   ffn_down = n_ff = 64; mm.1 = n_embd·sf² = 128; mm.2 = intermediate = 64.
        let patch_size = 8;
        let n_embd = 32;
        let n_head = 4;
        let n_ff = 64;
        let n_layer = 2;
        let scale_factor = 2;
        let projection_dim = 16;
        let intermediate = 64;
        // 8×8 = 64 patches so attention spans >1 K_TILE (32) block, exercising
        // the tiled flash-attention kernel's cross-block online-softmax path.
        let trained_side = 8;
        let n_trained_patches = trained_side * trained_side;
        let image_size = trained_side * patch_size;
        let in_dim = 3 * patch_size * patch_size;
        let ppt = (patch_size * scale_factor) * (patch_size * scale_factor);

        let cfg = VisionEncoderConfig {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            eps: 1e-5,
            image_size,
            patch_size,
            n_trained_patches,
            projection_dim,
            scale_factor,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: ppt,
            image_max_pixels: ppt * n_trained_patches,
        };

        let blocks = (0..n_layer)
            .map(|l| {
                let s = l * 100 + 1;
                VitBlockWeights {
                    ln1_w: rnd(n_embd, s + 1),
                    ln1_b: rnd(n_embd, s + 2),
                    q_w: lin(n_embd, n_embd, s + 3),
                    q_b: rnd(n_embd, s + 4),
                    k_w: lin(n_embd, n_embd, s + 5),
                    k_b: rnd(n_embd, s + 6),
                    v_w: lin(n_embd, n_embd, s + 7),
                    v_b: rnd(n_embd, s + 8),
                    o_w: lin(n_embd, n_embd, s + 9),
                    o_b: rnd(n_embd, s + 10),
                    ln2_w: rnd(n_embd, s + 11),
                    ln2_b: rnd(n_embd, s + 12),
                    ffn_up_w: lin(n_ff, n_embd, s + 13),
                    ffn_up_b: rnd(n_ff, s + 14),
                    ffn_down_w: lin(n_embd, n_ff, s + 15),
                    ffn_down_b: rnd(n_embd, s + 16),
                }
            })
            .collect();

        VisionEncoderWeights {
            patch_embed: PatchEmbedWeights {
                conv_w: rnd(in_dim * n_embd, 50),
                conv_b: rnd(n_embd, 51),
            },
            position_embed: rnd(n_trained_patches * n_embd, 52),
            blocks,
            post_ln_w: rnd(n_embd, 53),
            post_ln_b: rnd(n_embd, 54),
            projector: ProjectorWeights {
                mm1_w: lin(intermediate, n_embd * scale_factor * scale_factor, 55),
                mm1_b: rnd(intermediate, 56),
                mm2_w: lin(projection_dim, intermediate, 57),
                mm2_b: rnd(projection_dim, 58),
            },
            config: cfg,
        }
    }

    /// Shared CPU↔GPU parity check, generic over the backend ops. Compares the
    /// GPU forward against the CPU encoder on a synthetic 2-layer ViT with the
    /// dynamic grid == trained grid (no pos-embed interpolation).
    ///
    /// Tolerance follows numpy-`allclose` semantics: each element must satisfy
    /// `|cpu - gpu| <= atol + rtol * |cpu|`. A pure absolute bound is the wrong
    /// metric for the quantized GEMM paths, whose kernels store dequantized
    /// weights as f16 — half-precision accumulation error scales with the output
    /// magnitude, so a single large-magnitude element (e.g. ~4.5) can drift
    /// ~1% (~0.05 abs) and tip a fixed absolute bound while every other element
    /// is fine. `rtol` absorbs that magnitude-proportional noise; `atol` keeps
    /// small-magnitude elements honest. A real GEMM bug (wrong index/scale/
    /// transpose) produces errors of order the output itself, far past either
    /// bound, so this does not mask regressions. Pass `rtol = 0.0` for the
    /// all-f32 paths, where the absolute bound alone is already tight.
    fn run_parity<O: VitGpuOps>(ops: &O, enc: &VisionEncoderWeights, atol: f32, rtol: f32) {
        let cfg = &enc.config;
        // Match the 8×8 trained grid (no pos-embed interpolation) and span
        // multiple attention K_TILE blocks (64 tokens > 32).
        let grid_w = 8;
        let grid_h = 8;
        let target_w = grid_w * cfg.patch_size;
        let target_h = grid_h * cfg.patch_size;
        let pixels = rnd(3 * target_h * target_w, 999);

        let cpu_out = enc.encode_image(&pixels, grid_w, grid_h).unwrap();

        let gpu_w = GpuVitWeights::build(ops, enc);
        let gpu_out = encode_image_gpu(ops, &gpu_w, &pixels, grid_w, grid_h).unwrap();

        assert_eq!(cpu_out.len(), gpu_out.len(), "output length mismatch");
        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (c, g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let d = (c - g).abs();
            max_diff = max_diff.max(d);
            max_rel = max_rel.max(d / (c.abs() + 1e-6));
            let limit = atol + rtol * c.abs();
            assert!(
                d <= limit,
                "encode_image parity mismatch at {i}: cpu={c}, gpu={g}, diff={d} \
                 (limit={limit}, atol={atol}, rtol={rtol})"
            );
        }
        println!(
            "ViT encode parity: max_diff={max_diff:.6}, max_rel={max_rel:.6}, {} values",
            cpu_out.len()
        );
    }

    /// The register-tiled flash kernel (head_dim 64) against a direct f64 softmax-attention, over
    /// token counts that are and are not a multiple of its 32-query and 32-key tiles.
    #[cfg(feature = "gpu")]
    #[test]
    fn wgpu_flash_attention_matches_the_definition() {
        let Some(ctx) = gpu_or_skip() else { return };
        // The dispatch predicate, not the adapter-only one: the `runs > 0` assert below
        // needs `flash_ready`, which carries the head_dim and wasm32 conjuncts.
        let flash_capable = ctx.uses_flash_attention(64);
        require_flash_or_skip(
            &ctx,
            flash_capable,
            "wgpu_flash_attention_matches_the_definition",
        );
        if !flash_capable {
            return;
        }
        let ops = WgpuVitOps::new(ctx)
            .expect("build wgpu vit ops")
            .with_f32_attention();
        let (heads, hd) = (4usize, 64usize);
        let dim = heads * hd;
        // 2100 tokens is 66 query tiles: past the 64 one dispatch covers, so the call splits
        for n in [1usize, 5, 31, 32, 33, 257, 1100, 2100] {
            let before = ops
                .flash_dispatches
                .load(std::sync::atomic::Ordering::Relaxed);
            let (q, k, v) = (rnd(n * dim, 1), rnd(n * dim, 2), rnd(n * dim, 3));
            let want = {
                let scale = (hd as f64).powf(-0.5);
                let mut out = vec![0f32; n * dim];
                let mut s = vec![0.0f64; n];
                for r in 0..n {
                    for h in 0..heads {
                        let q_off = r * dim + h * hd;
                        for (j, sj) in s.iter_mut().enumerate() {
                            let k_off = j * dim + h * hd;
                            let mut dot = 0.0f64;
                            for d in 0..hd {
                                dot += f64::from(q[q_off + d]) * f64::from(k[k_off + d]);
                            }
                            *sj = dot * scale;
                        }
                        let max = s.iter().cloned().fold(f64::MIN, f64::max);
                        let mut sum = 0.0;
                        for x in s.iter_mut() {
                            *x = (*x - max).exp();
                            sum += *x;
                        }
                        let inv_sum = 1.0 / sum;
                        let mut acc = [0.0f64; 64];
                        for (j, &sj) in s.iter().enumerate() {
                            let v_off = j * dim + h * hd;
                            for d in 0..hd {
                                acc[d] += sj * f64::from(v[v_off + d]);
                            }
                        }
                        for d in 0..hd {
                            out[q_off + d] = (acc[d] * inv_sum) as f32;
                        }
                    }
                }
                out
            };
            let got = ops.attention(
                &ops.upload(&q),
                &ops.upload(&k),
                &ops.upload(&v),
                n,
                heads,
                hd,
            );
            let got = ops.download(&got, n * dim);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "n={n}: worst {worst}");
            // output alone cannot tell flash from the tiled fallback, so pin the dispatch:
            // every leg here must run the flash kernel, and the 2100-token leg splits in two.
            let runs = ops
                .flash_dispatches
                .load(std::sync::atomic::Ordering::Relaxed)
                - before;
            assert!(runs > 0, "n={n}: the flash kernel ran zero dispatches");
            if n == 2100 {
                assert_eq!(runs, 2, "2100 tokens are 66 tiles over 64 per dispatch");
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_encode_image_parity() {
        let Some(ctx) = gpu_or_skip() else { return };
        run_parity(
            &WgpuVitOps::new(ctx).expect("build wgpu vit ops"),
            &synth_encoder(),
            2e-3,
            0.0, // all-f32 path: absolute bound is already tight
        );
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_metal_encode_image_parity() {
        let ctx = match crate::backend::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                require_metal(e);
                return;
            }
        };
        run_parity(&MetalVitOps::new(ctx).unwrap(), &synth_encoder(), 2e-3, 0.0);
    }

    /// Encode through the `VisionGpuEncode` impl (not the inner
    /// `encode_image_gpu` the parity tests call): pins that the
    /// discard/drain wrapper neither breaks normal encodes nor reports a
    /// stray fault. The fault-conversion half (mid-encode fault becomes
    /// `Err` so the session's CPU fallback fires) needs a real device
    /// fault and is not injectable here; there is no poison seam for the
    /// readback/command slots.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_wgpu_encode_wrapper_happy_path() {
        let enc = synth_encoder();
        let Some(ctx) = gpu_or_skip() else { return };
        let encoder =
            build_wgpu_vision_encoder_with_context(ctx, &enc).expect("build wgpu vision encoder");
        let cfg = &enc.config;
        let pixels = rnd(3 * 8 * cfg.patch_size * 8 * cfg.patch_size, 999);
        let cpu_out = enc.encode_image(&pixels, 8, 8).unwrap();
        let gpu_out = encoder.encode_image(&pixels, 8, 8).unwrap();
        assert_eq!(gpu_out.len(), cpu_out.len());
        assert!(!gpu_out.iter().all(|&x| x == 0.0));
    }

    /// The engine hands one GPU vision encoder to every session, so concurrent encodes must not see
    /// each other: the pending-pass encoder and the buffer pool are per encoder, not per call. Without
    /// serialization a second thread flushes while the first still has passes recorded, or pops a
    /// buffer the first has not finished with, and gets plausible garbage back as `Ok`. Each thread
    /// encodes its own input many times against a sequential reference.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_wgpu_encoder_is_safe_to_share_across_threads() {
        use crate::engine::BackendPreference as BP;
        let enc = synth_encoder();
        let Some(encoder) = build_gpu_vision_encoder(&enc, BP::Gpu) else {
            // No GPU, or the warm-up encode failed: a leg that requires a GPU must not pass silently.
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but no GPU vision encoder was built"
            );
            return;
        };
        let cfg = &enc.config;
        let (gw, gh) = (8usize, 8usize);
        let len = 3 * gw * cfg.patch_size * gh * cfg.patch_size;
        let inputs: Vec<Vec<f32>> = (0..4).map(|t| rnd(len, 500 + t)).collect();
        let want: Vec<Vec<f32>> = inputs
            .iter()
            .map(|p| encoder.encode_image(p, gw, gh).unwrap())
            .collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..inputs.len())
                .map(|t| {
                    let (encoder, inputs, want) = (&encoder, &inputs, &want);
                    scope.spawn(move || {
                        for i in 0..40 {
                            let got = encoder.encode_image(&inputs[t], gw, gh).unwrap();
                            assert_eq!(got.len(), want[t].len());
                            let diff = got
                                .iter()
                                .zip(&want[t])
                                .map(|(a, b)| {
                                    // A NaN difference is a failure: `f32::max` would drop it.
                                    let d = (a - b).abs();
                                    if d.is_nan() { f32::INFINITY } else { d }
                                })
                                .fold(0.0f32, f32::max);
                            assert!(diff < 1e-3, "thread {t} encode {i}: max abs diff {diff}");
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    /// Metal twin of `test_wgpu_encode_wrapper_happy_path` (same
    /// discard/drain wrapper shape over `take_cmd_error`).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_metal_encode_wrapper_happy_path() {
        use crate::engine::BackendPreference as BP;
        let enc = synth_encoder();
        let encoder = match build_gpu_vision_encoder(&enc, BP::Metal) {
            Some(e) => e,
            None => {
                require_metal("no Metal vision encoder built");
                return;
            }
        };
        let cfg = &enc.config;
        let pixels = rnd(3 * 8 * cfg.patch_size * 8 * cfg.patch_size, 999);
        let cpu_out = enc.encode_image(&pixels, 8, 8).unwrap();
        let gpu_out = encoder.encode_image(&pixels, 8, 8).unwrap();
        assert_eq!(gpu_out.len(), cpu_out.len());
        assert!(!gpu_out.iter().all(|&x| x == 0.0));
    }

    /// Q8_0 linear weights → exercises the Metal simdgroup `gemm_q8_0` path.
    /// The GEMM stores dequantized weights as f16, so its error scales with the
    /// output magnitude — hence the relative tolerance (see `run_parity`). A
    /// pure 5e-2 absolute bound was flaky here: the largest-magnitude element
    /// (~4.5) drifts ~0.0505 (1.1% rel), tipping the bound on some Metal
    /// hardware while every other element is well within it.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_metal_encode_image_parity_q8_0() {
        let ctx = match crate::backend::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                require_metal(e);
                return;
            }
        };
        run_parity(
            &MetalVitOps::new(ctx).unwrap(),
            &synth_encoder_quant(Some(DType::Q8_0)),
            5e-2, // atol: proven floor for small-magnitude elements
            2e-2, // rtol: f16-GEMM noise on large-magnitude elements
        );
    }

    /// Q8_0 linear weights on wgpu → exercises the `gemm_q8_0` / `QuantStream`
    /// GEMM path (packed bytes, no dequant-to-f32 upload). On Vulkan passthrough
    /// (`lavapipe` / Adreno), all 14 `k % 32 == 0` linears (including `n_embd = 32`)
    /// run `transpose_cast_f16` + `gemm_stream_q8_0_k64` (fp16 activations and fp16
    /// accumulators) against the CPU's int8-activation `Q8Act` reference.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_encode_image_parity_q8_0() {
        let Some(ctx) = gpu_or_skip() else { return };
        run_parity(
            &WgpuVitOps::new(ctx).expect("build wgpu vit ops"),
            &synth_encoder_quant(Some(DType::Q8_0)),
            6e-2,
            2.5e-2, // fp16-GEMM noise scales with magnitude (see run_parity)
        );
    }

    /// Q4_0 linear weights on wgpu → exercises the `gemm_q4_0` GEMM path.
    /// Looser tolerance than Q8_0: 4-bit quantization is far coarser.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_encode_image_parity_q4_0() {
        let Some(ctx) = gpu_or_skip() else { return };
        run_parity(
            &WgpuVitOps::new(ctx).expect("build wgpu vit ops"),
            &synth_encoder_quant(Some(DType::Q4_0)),
            2e-1,
            4e-2, // 4-bit GEMM noise scales with magnitude (see run_parity)
        );
    }
    #[cfg(feature = "gpu")]
    #[test]
    fn test_wgpu_vit_ops_attention_zero_tokens() {
        let Some(ctx) = gpu_or_skip() else { return };
        let ops = WgpuVitOps::new(ctx).expect("build wgpu vit ops");
        let empty = ops.upload(&[]);
        let out = ops.attention(&empty, &empty, &empty, 0, 4, 64);
        assert_eq!(out.size(), (32 * 256 * 4) as u64);
    }

    #[test]
    fn test_build_gpu_vision_encoder_hexagon_preference_fallback() {
        let enc = synth_encoder();
        let result = build_gpu_vision_encoder(&enc, crate::engine::BackendPreference::Hexagon);
        #[cfg(not(target_os = "android"))]
        assert!(result.is_none());
        let _ = result;
    }

    /// A spec-spoiling function for the per-entry negative legs below.
    type Spoil = fn(&mut VitStackBlock);

    /// Device-free ops for pinning the `GpuStack` guards: buffers are host shadows, every
    /// producing op returns zeros of the promised shape, and the limits are knobs.
    struct TinyOps {
        token_limit: usize,
        max_bytes: u64,
        max_rows: usize,
    }

    impl VitGpuOps for TinyOps {
        type Buf = Vec<f32>;
        type Weight = ();

        fn upload(&self, data: &[f32]) -> Self::Buf {
            data.to_vec()
        }

        fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
            buf.iter()
                .copied()
                .chain(std::iter::repeat(0.0))
                .take(len)
                .collect()
        }

        fn upload_weight(&self, _w: &MmapWeight) -> Self::Weight {}

        fn upload_weight_f32(
            &self,
            _data: &[f32],
            _out_dim: usize,
            _in_dim: usize,
        ) -> Self::Weight {
        }

        fn linear(
            &self,
            _x: &Self::Buf,
            _w: &Self::Weight,
            tokens: usize,
            out_dim: usize,
            _in_dim: usize,
        ) -> Self::Buf {
            vec![0.0; tokens * out_dim]
        }

        fn bias_add(&self, _x: &Self::Buf, _bias: &Self::Buf, _rows: usize, _dim: usize) {}

        fn layernorm(
            &self,
            _src: &Self::Buf,
            _weight: &Self::Buf,
            _bias: &Self::Buf,
            _eps: f32,
            rows: usize,
            dim: usize,
        ) -> Self::Buf {
            vec![0.0; rows * dim]
        }

        fn gelu(&self, _x: &Self::Buf, _len: usize) {}

        fn attention(
            &self,
            _q: &Self::Buf,
            _k: &Self::Buf,
            _v: &Self::Buf,
            tokens: usize,
            n_head: usize,
            head_dim: usize,
        ) -> Self::Buf {
            vec![0.0; tokens * n_head * head_dim]
        }

        fn attention_token_limit(&self, _head_dim: usize) -> usize {
            self.token_limit
        }

        fn max_buffer_bytes(&self) -> u64 {
            self.max_bytes
        }

        fn max_rows(&self) -> usize {
            self.max_rows
        }

        fn add(&self, _dst: &Self::Buf, _src: &Self::Buf, _len: usize) {}
    }

    /// A valid one-block stack spec: `d` wide, 2 heads, `ff` intermediate.
    fn tiny_spec(d: usize, ff: usize) -> VitStackSpec {
        let vec = |n: usize| vec![0.0f32; n];
        let mat =
            |rows: usize, cols: usize| MmapWeight::from_owned_f32(vec(rows * cols), rows, cols);
        VitStackSpec {
            width: d,
            heads: 2,
            ffn: ff,
            eps: 1e-5,
            activation: VitStackActivation::GeluTanh,
            blocks: vec![VitStackBlock {
                ln1_w: vec(d),
                ln1_b: vec(d),
                q: mat(d, d),
                q_b: vec(d),
                k: mat(d, d),
                k_b: vec(d),
                v: mat(d, d),
                v_b: vec(d),
                o: mat(d, d),
                o_b: vec(d),
                ln2_w: vec(d),
                ln2_b: vec(d),
                up: mat(ff, d),
                up_b: vec(ff),
                down: mat(d, ff),
                down_b: vec(d),
            }],
            post: Some((vec(d), vec(d))),
            projector: None,
        }
    }

    /// The `GpuStack` guards refuse violating inputs: a ragged spec fails `build` (each of the
    /// 16 guarded entries pinned by a leg naming it, plus a short-payload leg), and an
    /// over-limit or misshapen pass fails `run` (attention, byte, and row gates each pinned),
    /// while a valid pass succeeds through the fake.
    #[test]
    fn stack_guards_refuse_violating_inputs() {
        let ops = || TinyOps {
            token_limit: 4,
            max_bytes: u64::MAX,
            max_rows: usize::MAX,
        };
        let (d, ff) = (8usize, 16usize);
        let stack = GpuStack::build(ops(), &tiny_spec(d, ff)).unwrap();
        assert_eq!(stack.run(&vec![0.0; 2 * d], 2).unwrap().len(), 2 * d);
        // `GpuStack` has no `Debug`, so `expect_err` is unavailable; match instead
        let expect_refusal = |name: &str, spec: &VitStackSpec| match GpuStack::build(ops(), spec) {
            Ok(_) => panic!("{name}: build must refuse"),
            Err(e) => e,
        };

        let mut bad_heads = tiny_spec(d, ff);
        bad_heads.heads = 0;
        assert!(GpuStack::build(ops(), &bad_heads).is_err());
        // every guarded entry gets its own negative leg naming it: dropping one tuple from
        // the spec loops must fail this test, not sail through on a later guard
        let bad_vectors: &[(&str, Spoil)] = &[
            ("ln1_w", |b| {
                b.ln1_w.pop();
            }),
            ("ln1_b", |b| {
                b.ln1_b.pop();
            }),
            ("q_b", |b| {
                b.q_b.pop();
            }),
            ("k_b", |b| {
                b.k_b.pop();
            }),
            ("v_b", |b| {
                b.v_b.pop();
            }),
            ("o_b", |b| {
                b.o_b.pop();
            }),
            ("ln2_w", |b| {
                b.ln2_w.pop();
            }),
            ("ln2_b", |b| {
                b.ln2_b.pop();
            }),
            ("up_b", |b| {
                b.up_b.pop();
            }),
            ("down_b", |b| {
                b.down_b.pop();
            }),
        ];
        for (name, spoil) in bad_vectors {
            let mut bad = tiny_spec(d, ff);
            spoil(&mut bad.blocks[0]);
            let err = expect_refusal(name, &bad);
            assert!(err.to_string().contains(&format!(" {name} has ")), "{err}");
        }
        let bad_matrices: &[(&str, Spoil)] = &[
            ("q", |b| {
                b.q.rows -= 1;
            }),
            ("k", |b| {
                b.k.rows -= 1;
            }),
            ("v", |b| {
                b.v.rows -= 1;
            }),
            ("o", |b| {
                b.o.rows -= 1;
            }),
            ("up", |b| {
                b.up.rows -= 1;
            }),
            ("down", |b| {
                b.down.rows -= 1;
            }),
        ];
        for (name, spoil) in bad_matrices {
            let mut bad = tiny_spec(d, ff);
            spoil(&mut bad.blocks[0]);
            let err = expect_refusal(name, &bad);
            assert!(err.to_string().contains(&format!(" {name} is ")), "{err}");
        }
        let mut bad_post = tiny_spec(d, ff);
        bad_post.post.as_mut().unwrap().0.pop();
        assert!(GpuStack::build(ops(), &bad_post).is_err());
        let mut bad_width = tiny_spec(d, ff);
        bad_width.width = 0;
        let err = expect_refusal("width 0", &bad_width);
        assert!(err.to_string().contains("must be positive"), "{err}");
        let mut bad_ffn = tiny_spec(d, ff);
        bad_ffn.ffn = 0;
        let err = expect_refusal("ffn 0", &bad_ffn);
        assert!(err.to_string().contains("must be positive"), "{err}");
        let mut bad_ratio = tiny_spec(d, ff);
        bad_ratio.heads = 3;
        assert!(GpuStack::build(ops(), &bad_ratio).is_err());
        let mut bad_post_b = tiny_spec(d, ff);
        bad_post_b.post.as_mut().unwrap().1.pop();
        assert!(GpuStack::build(ops(), &bad_post_b).is_err());
        // right shape (48 x 48) but 48 columns are not whole Q8_0 blocks
        let mut bad_blocks = tiny_spec(48, 64);
        bad_blocks.blocks[0].q =
            MmapWeight::from_owned_bytes(vec![0u8; 48 * 34], DType::Q8_0, 48, 48);
        assert!(GpuStack::build(ops(), &bad_blocks).is_err());
        // right dims and dtype but a short payload: the byte check refuses what the shape
        // checks admit
        let mut bad_short = tiny_spec(d, ff);
        bad_short.blocks[0].q =
            MmapWeight::from_owned_bytes(vec![0u8; d * d * 4 - 4], DType::F32, d, d);
        let err = expect_refusal("short payload", &bad_short);
        assert!(err.to_string().contains("bytes, expected"), "{err}");
        // a `Relu` spec through a backend without the kernel refuses rather than panicking
        // mid-run (`TinyOps` keeps the trait default)
        let mut bad_relu = tiny_spec(d, ff);
        bad_relu.activation = VitStackActivation::Relu;
        let err = expect_refusal("relu without kernel", &bad_relu);
        assert!(err.to_string().contains("no ReLU kernel"), "{err}");

        let err = stack
            .run(&vec![0.0; 5 * d], 5)
            .expect_err("5 tokens exceeds the limit of 4");
        assert!(err.to_string().contains("attention limit"), "{err}");
        assert!(stack.run(&vec![0.0; 2 * d + 1], 2).is_err());
        let tight = GpuStack::build(
            TinyOps {
                token_limit: 4,
                max_bytes: 100,
                max_rows: usize::MAX,
            },
            &tiny_spec(d, ff),
        )
        .unwrap();
        let err = tight
            .run(&vec![0.0; 2 * d], 2)
            .expect_err("the padded pass needs more than 100 bytes");
        assert!(err.to_string().contains("byte buffer"), "{err}");
        // a row limit tighter than the token and byte gates refuses a pass both admit
        let narrow = GpuStack::build(
            TinyOps {
                token_limit: 4,
                max_bytes: u64::MAX,
                max_rows: 1,
            },
            &tiny_spec(d, ff),
        )
        .unwrap();
        let err = narrow
            .run(&vec![0.0; 2 * d], 2)
            .expect_err("2 tokens exceeds the row limit of 1");
        assert!(err.to_string().contains("row limit"), "{err}");
    }

    /// The kernel-selection helper agrees with itself at the boundary each backend retunes:
    /// the tiled ceiling admits, one past it refuses. (Android always takes the scalar path,
    /// so this leg is host-only.)
    #[cfg(all(feature = "gpu", not(target_os = "android")))]
    #[test]
    fn wgpu_streams_kv_matches_the_tiled_ceiling() {
        assert!(super::WgpuVitOps::streams_kv(64));
        assert!(!super::WgpuVitOps::streams_kv(65));
    }

    /// The MMA window admits a multiple of 8 through 128 and refuses past it or off-alignment.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_streams_kv_matches_the_mma_window() {
        assert!(super::MetalVitOps::streams_kv(64));
        assert!(super::MetalVitOps::streams_kv(128));
        assert!(!super::MetalVitOps::streams_kv(136));
        assert!(!super::MetalVitOps::streams_kv(100));
    }

    /// `chunk_bounds` covers short inputs in one chunk and splits long ones on the
    /// storage-offset alignment and whole rows, with every chunk fitting one dispatch.
    #[cfg(feature = "gpu")]
    #[test]
    fn chunk_bounds_respects_alignment_and_limit() {
        assert_eq!(chunk_bounds(100, 1), vec![(0, 100)]);
        assert!(chunk_bounds(0, 128).is_empty());
        assert!(chunk_bounds(0, 0).is_empty());
        let big = chunk_bounds(20_000_000, 128);
        assert!(big.len() > 1);
        let mut covered = 0usize;
        for (i, (off, n)) in big.iter().enumerate() {
            assert_eq!(*off, covered);
            assert_eq!(off % 64, 0, "chunk {i} offset");
            assert_eq!(off % 128, 0, "chunk {i} row alignment");
            // every chunk but the last is a whole number of steps; the last is the remainder
            if i + 1 < big.len() {
                assert_eq!(n % 128, 0, "chunk {i} length");
            }
            assert!(
                n.div_ceil(256) <= crate::backend::wgpu::MAX_WG as usize,
                "chunk {i} fits one dispatch"
            );
            covered += n;
        }
        assert_eq!(covered, 20_000_000);
    }
}

#[cfg(test)]
mod q8_stream_repack_tests {
    use super::repack_q8_0_stream;

    /// Every weight must be recoverable from the packed words and scales exactly as the kernel reads
    /// them: byte `j` of `q[(k/4) * m + row]` times the half scale of its 32-block.
    #[test]
    fn repack_q8_0_stream_round_trips_through_the_kernel_layout() {
        let (m, k) = (7usize, 192usize);
        let blocks = k / 32;
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as u32
        };
        let mut data = Vec::new();
        for _ in 0..m * blocks {
            let scale = half::f16::from_f32(((next() % 1000) as f32 + 1.0) / 4000.0);
            data.extend_from_slice(&scale.to_bits().to_le_bytes());
            for _ in 0..32 {
                data.push(next() as u8);
            }
        }
        let (q, d) = repack_q8_0_stream(&data, m, k);
        assert_eq!(q.len(), m * k / 4);
        assert_eq!(d.len(), m * (k / 64));

        let mut want = vec![0.0f32; k];
        for row in 0..m {
            crate::quant::dequantize_q8_0_row(
                &data[row * blocks * 34..(row + 1) * blocks * 34],
                &mut want,
            );
            for kk in 0..k {
                let word = q[(kk / 4) * m + row];
                let w = ((word >> (8 * (kk % 4))) as u8 as i8) as f32;
                let pair = d[(kk / 64) * m + row];
                let bits = if (kk / 32) % 2 == 0 {
                    pair & 0xFFFF
                } else {
                    pair >> 16
                };
                let got = w * half::f16::from_bits(bits as u16).to_f32();
                assert_eq!(got, want[kk], "row {row} k {kk}");
            }
        }
    }

    #[test]
    fn repack_q8_0_stream_pads_odd_block_to_64() {
        let (m, k) = (5usize, 96usize);
        let k_pad = 128usize;
        let blocks = k / 32;
        let mut data = Vec::with_capacity(m * blocks * 34);
        for b in 0..m * blocks {
            let scale = half::f16::from_f32(0.05 * (b as f32 + 1.0));
            data.extend_from_slice(&scale.to_bits().to_le_bytes());
            for j in 0..32 {
                data.push((j as i8 - 16) as u8);
            }
        }
        let (q, d) = repack_q8_0_stream(&data, m, k);
        assert_eq!(q.len(), m * k_pad / 4);
        assert_eq!(d.len(), m * (k_pad / 64));
        for row in 0..m {
            for kk in k..k_pad {
                assert_eq!(
                    q[(kk / 4) * m + row],
                    0,
                    "padded q word at row {row} k {kk}"
                );
            }
            assert_eq!(
                d[(k / 64) * m + row] >> 16,
                0,
                "padded high scale at row {row}"
            );
        }
    }

    #[test]
    fn repack_f16_as_q8_0_stream_quantizes_and_zero_pads() {
        use super::repack_f16_as_q8_0_stream;
        let (m, k) = (4usize, 80usize);
        let k_pad = 128usize;
        let mut f16_bytes = Vec::with_capacity(m * k * 2);
        let mut want = Vec::with_capacity(m * k);
        for i in 0..m * k {
            let v = ((i as f32 * 0.13).sin()) * 2.5;
            let h = half::f16::from_f32(v);
            want.push(h.to_f32());
            f16_bytes.extend_from_slice(&h.to_bits().to_le_bytes());
        }
        let (q, d) = repack_f16_as_q8_0_stream(&f16_bytes, m, k);
        assert_eq!(q.len(), m * k_pad / 4);
        assert_eq!(d.len(), m * (k_pad / 64));
        for row in 0..m {
            for kk in 0..k_pad {
                let word = q[(kk / 4) * m + row];
                let w = ((word >> (8 * (kk % 4))) as u8 as i8) as f32;
                let pair = d[(kk / 64) * m + row];
                let bits = if (kk / 32) % 2 == 0 {
                    pair & 0xFFFF
                } else {
                    pair >> 16
                };
                let got = w * half::f16::from_bits(bits as u16).to_f32();
                if kk < k {
                    let ref_v = want[row * k + kk];
                    assert!(
                        (got - ref_v).abs() < 0.03,
                        "row {row} k {kk}: got {got} vs {ref_v}"
                    );
                } else {
                    assert_eq!(got, 0.0, "padded tail at row {row} k {kk} must be 0.0");
                }
            }
        }
    }
}

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

#[cfg(test)]
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

    /// Upload `data` to a new GPU buffer.
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// Read `len` f32s back from a GPU buffer (blocking).
    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32>;

    /// `y[tokens, out_dim] = x[tokens, in_dim] · wᵀ` where `w` is
    /// `[out_dim, in_dim]` row-major (the `MmapWeight` linear-layer layout).
    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
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

    /// In-place residual add: `dst[i] += src[i]` over `len` elements.
    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize);
}

/// Dequantize a linear weight to a contiguous `[rows*cols]` f32 vector for
/// upload. F32 weights are copied directly; quantized dtypes go row-by-row.
fn dequant_weight(w: &MmapWeight) -> Vec<f32> {
    if let Some(f) = w.try_as_f32() {
        return f.to_vec();
    }
    let mut out = vec![0f32; w.rows * w.cols];
    for r in 0..w.rows {
        w.dequantize_row(r, &mut out[r * w.cols..(r + 1) * w.cols]);
    }
    out
}

/// One ViT block's weights, uploaded to GPU buffers.
pub struct GpuVitBlock<B> {
    ln1_w: B,
    ln1_b: B,
    q_w: B,
    q_b: B,
    k_w: B,
    k_b: B,
    v_w: B,
    v_b: B,
    o_w: B,
    o_b: B,
    ln2_w: B,
    ln2_b: B,
    ffn_up_w: B,
    ffn_up_b: B,
    ffn_down_w: B,
    ffn_down_b: B,
}

/// All vision-encoder weights uploaded to GPU buffers, plus the small CPU-side
/// state the per-call rearrangements need (config + trained position embedding).
///
/// Built once via [`GpuVitWeights::build`] and reused across images — the
/// upload (the LFM2-VL mmproj dequantized to f32) is the expensive part and
/// must not happen per image.
pub struct GpuVitWeights<B> {
    cfg: VisionEncoderConfig,
    /// Trained position embedding `[n_trained_patches * n_embd]`, kept on CPU
    /// for per-call bilinear interpolation to the dynamic grid.
    position_embed: Vec<f32>,
    /// Patch-embed kernel transposed to `[n_embd, in_dim]` (the `linear`
    /// layout); the CPU encoder stores it as `[in_dim, n_embd]` for its
    /// `C = A·B` matmul, but `linear` computes `A·Bᵀ`.
    patch_conv_wt: B,
    patch_conv_b: B,
    blocks: Vec<GpuVitBlock<B>>,
    post_ln_w: B,
    post_ln_b: B,
    mm1_w: B,
    mm1_b: B,
    mm2_w: B,
    mm2_b: B,
    /// Projector intermediate width (`mm.1` rows). Derived from the tensor
    /// shape, not the LFM2 `projection_dim·2` convention, to stay robust to
    /// variants — matching `vision_encoder`'s loader.
    proj_intermediate: usize,
}

impl<B> GpuVitWeights<B> {
    /// Upload every encoder weight via `ops`. Run once per loaded model.
    pub fn build<O: VitGpuOps<Buf = B>>(ops: &O, w: &VisionEncoderWeights) -> Self {
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
                q_w: ops.upload(&dequant_weight(&b.q_w)),
                q_b: ops.upload(&b.q_b),
                k_w: ops.upload(&dequant_weight(&b.k_w)),
                k_b: ops.upload(&b.k_b),
                v_w: ops.upload(&dequant_weight(&b.v_w)),
                v_b: ops.upload(&b.v_b),
                o_w: ops.upload(&dequant_weight(&b.o_w)),
                o_b: ops.upload(&b.o_b),
                ln2_w: ops.upload(&b.ln2_w),
                ln2_b: ops.upload(&b.ln2_b),
                ffn_up_w: ops.upload(&dequant_weight(&b.ffn_up_w)),
                ffn_up_b: ops.upload(&b.ffn_up_b),
                ffn_down_w: ops.upload(&dequant_weight(&b.ffn_down_w)),
                ffn_down_b: ops.upload(&b.ffn_down_b),
            })
            .collect();

        GpuVitWeights {
            position_embed: w.position_embed.clone(),
            patch_conv_wt: ops.upload(&convt),
            patch_conv_b: ops.upload(&w.patch_embed.conv_b),
            blocks,
            post_ln_w: ops.upload(&w.post_ln_w),
            post_ln_b: ops.upload(&w.post_ln_b),
            mm1_w: ops.upload(&dequant_weight(&w.projector.mm1_w)),
            mm1_b: ops.upload(&w.projector.mm1_b),
            mm2_w: ops.upload(&dequant_weight(&w.projector.mm2_w)),
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

/// Run the ViT encoder + projector on the GPU. Backend-agnostic: `ops` provides
/// the kernels, `gpu_w` the uploaded weights. Output is identical in shape to
/// [`VisionEncoderWeights::encode_image`]: `[n_image_tokens * projection_dim]`.
pub fn encode_image_gpu<O: VitGpuOps>(
    ops: &O,
    gpu_w: &GpuVitWeights<O::Buf>,
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
        grid_w % cfg.scale_factor == 0 && grid_h % cfg.scale_factor == 0,
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
        std::borrow::Cow::Owned(interpolate_pos_embed_2d(
            &gpu_w.position_embed,
            trained_side,
            trained_side,
            grid_h,
            grid_w,
            n_embd,
        ))
    };
    let pos_buf = ops.upload(&pos);
    ops.add(&tokens, &pos_buf, n_patches * n_embd);

    // 3. ViT blocks.
    for blk in &gpu_w.blocks {
        // Pre-attention LN → Q/K/V (+bias) → attention → O proj (+bias) → residual.
        let normed = ops.layernorm(&tokens, &blk.ln1_w, &blk.ln1_b, eps, n_patches, n_embd);
        let q = ops.linear(&normed, &blk.q_w, n_patches, n_embd, n_embd);
        ops.bias_add(&q, &blk.q_b, n_patches, n_embd);
        let k = ops.linear(&normed, &blk.k_w, n_patches, n_embd, n_embd);
        ops.bias_add(&k, &blk.k_b, n_patches, n_embd);
        let v = ops.linear(&normed, &blk.v_w, n_patches, n_embd, n_embd);
        ops.bias_add(&v, &blk.v_b, n_patches, n_embd);
        let attn = ops.attention(&q, &k, &v, n_patches, n_head, head_dim);
        let proj = ops.linear(&attn, &blk.o_w, n_patches, n_embd, n_embd);
        ops.bias_add(&proj, &blk.o_b, n_patches, n_embd);
        ops.add(&tokens, &proj, n_patches * n_embd);

        // Pre-MLP LN → FFN up (+bias) → GELU → FFN down (+bias) → residual.
        let normed2 = ops.layernorm(&tokens, &blk.ln2_w, &blk.ln2_b, eps, n_patches, n_embd);
        let mid = ops.linear(&normed2, &blk.ffn_up_w, n_patches, n_ff, n_embd);
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

    // 5. Pixel-shuffle on CPU (pure rearrangement).
    let tok_cpu = ops.download(&tokens, n_patches * n_embd);
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

    Ok(ops.download(&out, n_out * proj_dim))
}

// ── wgpu backend implementation ──────────────────────────────────────────────

/// wgpu implementation of [`VitGpuOps`]. Owns the [`GpuContext`] and the compute
/// pipelines (compiled once) so it can be cached for the session's lifetime.
/// Bind groups are created per dispatch (cheap relative to the kernel work).
#[cfg(feature = "gpu")]
pub struct WgpuVitOps {
    ctx: crate::backend::wgpu::GpuContext,
    p_linear: wgpu::ComputePipeline,
    p_bias: wgpu::ComputePipeline,
    p_layernorm: wgpu::ComputePipeline,
    p_gelu: wgpu::ComputePipeline,
    p_attn: wgpu::ComputePipeline,
    p_add: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
impl WgpuVitOps {
    pub fn new(ctx: crate::backend::wgpu::GpuContext) -> Result<Self> {
        use crate::backend::wgpu::shaders;
        // `create_pipeline*` return the pipeline directly and panic on shader
        // preprocessing or adapter-side validation/compile failure (they have
        // no `Result`). Catch that here and surface an `Err` so the caller's
        // `?` degrades to the CPU encoder — mirroring `MetalVitOps::new`'s
        // `.ok()?` — instead of aborting `CeraEngine` construction on a weak or
        // non-conformant adapter.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            // f32 batched matmul via the SCALAR mul_mat variant (handles any m).
            let p_linear = ctx.create_pipeline_with_defines(
                shaders::MUL_MAT_REG_TILE,
                "main",
                "vit_linear",
                &[
                    ("SCALAR", ""),
                    ("SRC0_INNER_TYPE", "f32"),
                    ("SRC1_INNER_TYPE", "f32"),
                    ("INIT_SRC0_SHMEM_FLOAT", ""),
                    ("INIT_SRC1_SHMEM_FLOAT", ""),
                    ("WORKGROUP_SIZE_M", "8u"),
                    ("WORKGROUP_SIZE_N", "8u"),
                    ("TILE_M", "4u"),
                    ("TILE_N", "4u"),
                    ("TILE_K", "32u"),
                ],
            );
            Self {
                p_bias: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add", "vit_bias_add"),
                p_layernorm: ctx.create_pipeline(
                    shaders::LAYERNORM_BATCH,
                    "layernorm_batch",
                    "vit_layernorm",
                ),
                p_gelu: ctx.create_pipeline(shaders::GELU, "gelu_inplace", "vit_gelu"),
                p_attn: ctx.create_pipeline(
                    shaders::VIT_ATTENTION,
                    "vit_attention",
                    "vit_attention",
                ),
                p_add: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "vit_add"),
                p_linear,
                ctx,
            }
        }))
        .map_err(|_| {
            anyhow::anyhow!("wgpu ViT pipeline creation failed (shader compile/validation)")
        })
    }

    /// Encode one bind group from `bufs` (in binding order) and dispatch.
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
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
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }
}

#[cfg(feature = "gpu")]
impl VitGpuOps for WgpuVitOps {
    type Buf = wgpu::Buffer;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        self.ctx.upload_f32(data, "vit")
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        self.ctx.download_f32(buf, len)
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        let y = self
            .ctx
            .create_storage_rw((tokens * out_dim * 4) as u64, "vit_linear_out");
        // MulMatParams: m, k, n, x_stride, y_stride.
        let params: [u32; 5] = [
            out_dim as u32,
            in_dim as u32,
            tokens as u32,
            in_dim as u32,
            out_dim as u32,
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "vit_linear_params");
        let wg_m = (out_dim as u32).div_ceil(32);
        let wg_n = (tokens as u32).div_ceil(32);
        self.dispatch(&self.p_linear, &[w, x, &y, &p_buf], (wg_m, wg_n, 1));
        y
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        let total = (rows * dim) as u32;
        let params: [u32; 2] = [total, dim as u32];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "vit_bias_params");
        self.dispatch(
            &self.p_bias,
            &[x, bias, &p_buf],
            (total.div_ceil(256), 1, 1),
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
        let dst = self
            .ctx
            .create_storage_rw((rows * dim * 4) as u64, "vit_ln_out");
        let params: [u32; 4] = [dim as u32, eps.to_bits(), dim as u32, dim as u32];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "vit_ln_params");
        self.dispatch(
            &self.p_layernorm,
            &[src, &dst, weight, bias, &p_buf],
            (rows as u32, 1, 1),
        );
        dst
    }

    fn gelu(&self, x: &Self::Buf, len: usize) {
        let params: [u32; 2] = [len as u32, 0];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "vit_gelu_params");
        self.dispatch(
            &self.p_gelu,
            &[x, &p_buf],
            ((len as u32).div_ceil(256), 1, 1),
        );
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
        let out = self
            .ctx
            .create_storage_rw((tokens * dim * 4) as u64, "vit_attn_out");
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [u32; 4] = [
            tokens as u32,
            n_head as u32,
            head_dim as u32,
            scale.to_bits(),
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "vit_attn_params");
        self.dispatch(
            &self.p_attn,
            &[q, k, v, &out, &p_buf],
            (tokens as u32, n_head as u32, 1),
        );
        out
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        let params: [u32; 2] = [len as u32, 0];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "vit_add_params");
        self.dispatch(
            &self.p_add,
            &[dst, src, &p_buf],
            ((len as u32).div_ceil(256), 1, 1),
        );
    }
}

// ── native Metal backend implementation ──────────────────────────────────────

/// Native-Metal implementation of [`VitGpuOps`]. Mirrors [`WgpuVitOps`] using
/// MSL kernels. Each op runs in its own command buffer and blocks on
/// `wait_until_completed`, so `download` always sees current data (unified
/// memory on Apple Silicon).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalVitOps {
    ctx: crate::backend::metal::MetalContext,
    p_linear: metal::ComputePipelineState,
    p_bias: metal::ComputePipelineState,
    p_layernorm: metal::ComputePipelineState,
    p_gelu: metal::ComputePipelineState,
    p_attn: metal::ComputePipelineState,
    p_add: metal::ComputePipelineState,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalVitOps {
    pub fn new(ctx: crate::backend::metal::MetalContext) -> Result<Self> {
        use crate::backend::metal::shaders;
        Ok(Self {
            p_linear: ctx.create_pipeline(shaders::VIT_LINEAR, "vit_linear")?,
            p_bias: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add")?,
            p_layernorm: ctx.create_pipeline(shaders::LAYERNORM_BATCH, "layernorm_batch")?,
            p_gelu: ctx.create_pipeline(shaders::GELU, "gelu_inplace")?,
            p_attn: ctx.create_pipeline(shaders::VIT_ATTENTION, "vit_attention")?,
            p_add: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace")?,
            ctx,
        })
    }

    /// Run one kernel in its own command buffer: bind `bufs` at slots 0.. then
    /// `params` (as bytes) at the next slot, dispatch, and block until done.
    fn run(
        &self,
        pipe: &metal::ComputePipelineState,
        bufs: &[&metal::Buffer],
        params: &[u8],
        grid: metal::MTLSize,
        threads: metal::MTLSize,
    ) {
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_bytes(
            bufs.len() as u64,
            params.len() as u64,
            params.as_ptr() as *const _,
        );
        enc.dispatch_thread_groups(grid, threads);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl VitGpuOps for MetalVitOps {
    type Buf = metal::Buffer;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        self.ctx.upload_f32(data)
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        self.ctx.read_f32(buf, len)
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        let y = self.ctx.create_buffer((tokens * out_dim * 4) as u64);
        let params: [u32; 4] = [out_dim as u32, in_dim as u32, tokens as u32, 0];
        self.run(
            &self.p_linear,
            &[w, x, &y],
            bytemuck::cast_slice(&params),
            metal::MTLSize::new(out_dim as u64, tokens as u64, 1),
            metal::MTLSize::new(32, 1, 1),
        );
        y
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        let total = (rows * dim) as u32;
        let params: [u32; 2] = [total, dim as u32];
        self.run(
            &self.p_bias,
            &[x, bias],
            bytemuck::cast_slice(&params),
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
        let params: [u32; 4] = [dim as u32, eps.to_bits(), dim as u32, dim as u32];
        self.run(
            &self.p_layernorm,
            &[src, &dst, weight, bias],
            bytemuck::cast_slice(&params),
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        dst
    }

    fn gelu(&self, x: &Self::Buf, len: usize) {
        let params: [u32; 2] = [len as u32, 0];
        self.run(
            &self.p_gelu,
            &[x],
            bytemuck::cast_slice(&params),
            metal::MTLSize::new((len as u64).div_ceil(256), 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
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
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [u32; 4] = [
            tokens as u32,
            n_head as u32,
            head_dim as u32,
            scale.to_bits(),
        ];
        self.run(
            &self.p_attn,
            &[q, k, v, &out],
            bytemuck::cast_slice(&params),
            metal::MTLSize::new(tokens as u64, n_head as u64, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        out
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        let params: [u32; 2] = [len as u32, 0];
        self.run(
            &self.p_add,
            &[dst, src],
            bytemuck::cast_slice(&params),
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
    /// Encode preprocessed pixels (`[3·H·W]` NCHW, normalized) at the given
    /// patch grid. Output matches [`VisionEncoderWeights::encode_image`].
    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>>;
}

#[cfg(feature = "gpu")]
struct WgpuVisionEncoder {
    ops: WgpuVitOps,
    weights: GpuVitWeights<wgpu::Buffer>,
}

#[cfg(feature = "gpu")]
impl VisionGpuEncode for WgpuVisionEncoder {
    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        encode_image_gpu(&self.ops, &self.weights, pixels, grid_w, grid_h)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
struct MetalVisionEncoder {
    ops: MetalVitOps,
    weights: GpuVitWeights<metal::Buffer>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl VisionGpuEncode for MetalVisionEncoder {
    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        encode_image_gpu(&self.ops, &self.weights, pixels, grid_w, grid_h)
    }
}

/// Build a cached GPU vision encoder for `weights`, honoring `backend`.
/// Returns `None` for `Cpu`, when the chosen backend's feature isn't compiled,
/// or when the device/context can't be created — the caller then falls back to
/// the CPU encoder. `Auto` prefers Metal, then wgpu.
pub fn build_gpu_vision_encoder(
    weights: &VisionEncoderWeights,
    backend: crate::engine::BackendPreference,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    use crate::engine::BackendPreference as BP;
    match backend {
        BP::Cpu => None,
        BP::Metal => try_metal_vision_encoder(weights),
        BP::Gpu => try_wgpu_vision_encoder(weights),
        BP::Auto => try_metal_vision_encoder(weights).or_else(|| try_wgpu_vision_encoder(weights)),
    }
}

#[cfg(feature = "gpu")]
fn try_wgpu_vision_encoder(
    weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    let ctx = crate::backend::wgpu::GpuContext::new().ok()?;
    let ops = WgpuVitOps::new(ctx).ok()?;
    let gpu_w = GpuVitWeights::build(&ops, weights);
    tracing::info!("vision encoder: using wgpu GPU backend");
    Some(std::sync::Arc::new(WgpuVisionEncoder {
        ops,
        weights: gpu_w,
    }))
}

#[cfg(not(feature = "gpu"))]
fn try_wgpu_vision_encoder(
    _weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    None
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    }))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn try_metal_vision_encoder(
    _weights: &VisionEncoderWeights,
) -> Option<std::sync::Arc<dyn VisionGpuEncode>> {
    None
}

#[cfg(all(
    test,
    any(feature = "gpu", all(feature = "metal", target_os = "macos"))
))]
mod tests {
    use super::*;
    use crate::model::vision_encoder::{PatchEmbedWeights, ProjectorWeights, VitBlockWeights};
    use crate::model::weights::MmapWeight;
    use crate::tensor::DType;

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

    /// Build a tiny synthetic VL encoder (all F32) for CPU↔GPU parity.
    fn synth_encoder() -> VisionEncoderWeights {
        // All linear in_dims (k) must be multiples of the matmul's TILE_K=32:
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
        let trained_side = 4;
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
                    q_w: f32_weight(n_embd, n_embd, s + 3),
                    q_b: rnd(n_embd, s + 4),
                    k_w: f32_weight(n_embd, n_embd, s + 5),
                    k_b: rnd(n_embd, s + 6),
                    v_w: f32_weight(n_embd, n_embd, s + 7),
                    v_b: rnd(n_embd, s + 8),
                    o_w: f32_weight(n_embd, n_embd, s + 9),
                    o_b: rnd(n_embd, s + 10),
                    ln2_w: rnd(n_embd, s + 11),
                    ln2_b: rnd(n_embd, s + 12),
                    ffn_up_w: f32_weight(n_ff, n_embd, s + 13),
                    ffn_up_b: rnd(n_ff, s + 14),
                    ffn_down_w: f32_weight(n_embd, n_ff, s + 15),
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
                mm1_w: f32_weight(intermediate, n_embd * scale_factor * scale_factor, 55),
                mm1_b: rnd(intermediate, 56),
                mm2_w: f32_weight(projection_dim, intermediate, 57),
                mm2_b: rnd(projection_dim, 58),
            },
            config: cfg,
        }
    }

    /// Shared CPU↔GPU parity check, generic over the backend ops. Compares the
    /// GPU forward against the CPU encoder on a synthetic 2-layer ViT with the
    /// dynamic grid == trained grid (no pos-embed interpolation).
    fn run_parity<O: VitGpuOps>(ops: &O) {
        let enc = synth_encoder();
        let cfg = &enc.config;
        let grid_w = 4;
        let grid_h = 4;
        let target_w = grid_w * cfg.patch_size;
        let target_h = grid_h * cfg.patch_size;
        let pixels = rnd(3 * target_h * target_w, 999);

        let cpu_out = enc.encode_image(&pixels, grid_w, grid_h).unwrap();

        let gpu_w = GpuVitWeights::build(ops, &enc);
        let gpu_out = encode_image_gpu(ops, &gpu_w, &pixels, grid_w, grid_h).unwrap();

        assert_eq!(cpu_out.len(), gpu_out.len(), "output length mismatch");
        let mut max_diff = 0.0f32;
        for (i, (c, g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let d = (c - g).abs();
            max_diff = max_diff.max(d);
            assert!(
                d < 2e-3,
                "encode_image parity mismatch at {i}: cpu={c}, gpu={g}, diff={d}"
            );
        }
        println!(
            "ViT encode parity: max_diff={max_diff:.6}, {} values",
            cpu_out.len()
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_encode_image_parity() {
        let ctx = match crate::backend::wgpu::GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return, // no GPU (CI)
        };
        run_parity(&WgpuVitOps::new(ctx).expect("build wgpu vit ops"));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn test_metal_encode_image_parity() {
        let ctx = match crate::backend::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return, // no Metal device (CI)
        };
        run_parity(&MetalVitOps::new(ctx).unwrap());
    }
}

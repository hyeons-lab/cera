//! LFM2-VL vision encoder (image → continuous embeddings) — weights
//! loader, config, and tensor-name mapping.
//!
//! Loaded from the `multimodal_projector` GGUF in a LeapBundles VL
//! manifest (e.g. `mmproj-LFM2.5-VL-450m-Q8_0.gguf`). The encoder is
//! a CLIP-family ViT with a 2-layer MLP projector
//! (`PROJECTOR_TYPE_LFM2` in llama.cpp's `mtmd` / `clip` code).
//!
//! High-level shape (verified against LFM2.5-VL-450M-Q4_0; spec in
//! the per-topic memory `project_vl_architecture.md`):
//!
//! ```text
//! image [3 × 256 × 256] (RGB, normalised by mean/std)
//!   → patch_embd Conv2D(kernel=16, stride=16) + bias  → [256, 768]
//!   → + position_embd                                  → [256, 768]
//!   → 12 × ViT block (LN1 → MHA → +residual → LN2 → GELU MLP → +residual)
//!   → post_ln                                          → [256, 768]
//!   → pixel-shuffle 2×2 pool                            → [64, 3072]
//!   → mm.1 (3072 → 2048) + GELU                        → [64, 2048]
//!   → mm.2 (2048 → 1024)                                → [64, 1024]  (LLM embed dim)
//! ```
//!
//! This module is the **loader only** — config + weight structs +
//! `from_gguf`. The ViT forward pass + pixel-shuffle pool +
//! projector forward land in follow-up PRs (the "VL pipeline" plan
//! in `devlog/`).
//!
//! Tensor name conventions are taken from llama.cpp's
//! `tools/mtmd/clip-impl.h` (`TN_*` macros), substituted with
//! `prefix = "v"` (vision). All weight strings the loader reaches
//! for are listed up-front in this module's source so a future
//! schema drift on the upstream side surfaces as a single grep
//! target.

use anyhow::{Context, Result};

use crate::gguf::GgufFile;
use std::sync::Arc;

use crate::model::weights::MmapWeight;

// ── GGUF metadata keys ────────────────────────────────────────────

const KEY_HAS_VISION: &str = "clip.has_vision_encoder";
const KEY_N_LAYER: &str = "clip.vision.block_count";
const KEY_N_EMBD: &str = "clip.vision.embedding_length";
const KEY_N_FF: &str = "clip.vision.feed_forward_length";
const KEY_N_HEAD: &str = "clip.vision.attention.head_count";
const KEY_LN_EPS: &str = "clip.vision.attention.layer_norm_epsilon";
const KEY_IMAGE_SIZE: &str = "clip.vision.image_size";
const KEY_PATCH_SIZE: &str = "clip.vision.patch_size";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";
const KEY_SCALE_FACTOR: &str = "clip.vision.projector.scale_factor";
const KEY_IMAGE_MEAN: &str = "clip.vision.image_mean";
const KEY_IMAGE_STD: &str = "clip.vision.image_std";

/// Configuration for the LFM2-VL ViT vision encoder. Read from
/// the `clip.vision.*` metadata block of the multimodal_projector
/// GGUF.
#[derive(Debug, Clone)]
pub struct VisionEncoderConfig {
    /// Number of ViT transformer blocks.
    pub n_layer: usize,
    /// Encoder hidden dimension (`clip.vision.embedding_length`).
    pub n_embd: usize,
    /// FFN intermediate dimension inside each ViT block.
    pub n_ff: usize,
    /// Number of attention heads per block.
    pub n_head: usize,
    /// LayerNorm epsilon. Read from
    /// `clip.vision.attention.layer_norm_epsilon` — the metadata
    /// key is named after attention but the value applies to
    /// **every** norm in the encoder (per-block ln1 + ln2 + the
    /// final post_ln), matching llama.cpp's `clip.cpp` which
    /// uses the single key for all of them.
    pub eps: f32,
    /// Square input image side length in pixels (typically 256).
    pub image_size: usize,
    /// Square patch side length in pixels (typically 16).
    pub patch_size: usize,
    /// Number of patches in the **trained** position-grid:
    /// `(image_size / patch_size)²`. This is the row count of the
    /// `position_embd.weight` tensor — the size at which the model
    /// was trained. The **runtime** patch count is `grid_w *
    /// grid_h` after dynamic-resolution resize and is computed
    /// per-call by the preprocessor; when those don't match the
    /// trained grid, position embeddings are bilinearly
    /// interpolated to the dynamic dims (mirrors llama.cpp's
    /// `clip_graph::resize_position_embeddings`).
    pub n_trained_patches: usize,
    /// Projector output dimension; matches the LLM's
    /// `embedding_length` so projected image tokens drop straight
    /// into the LFM2 stream.
    pub projection_dim: usize,
    /// Pixel-shuffle pooling factor between the ViT output and the
    /// projector input. `scale_factor=2` means a 16×16 patch grid
    /// becomes 8×8 tokens with 4× channel inflation
    /// (768 → 768·4 = 3072).
    pub scale_factor: usize,
    /// Per-channel mean for image normalisation. RGB order, matches
    /// CLIP family preprocessing conventions.
    pub image_mean: [f32; 3],
    /// Per-channel std for image normalisation.
    pub image_std: [f32; 3],
    /// Lower bound on resized image area in pixels for the
    /// dynamic-resolution preprocessor. Equal to
    /// `n_tokens_min · (patch_size · scale_factor)²`. For LFM2-VL:
    /// `64 · 32² = 65 536` (= 256² square baseline).
    /// Source: llama.cpp's
    /// `clip_model.h::set_limit_image_tokens(64, 256)` for
    /// `PROJECTOR_TYPE_LFM2`. Hardcoded since no GGUF metadata
    /// key surfaces it.
    pub image_min_pixels: usize,
    /// Upper bound on resized image area in pixels.
    /// `n_tokens_max · (patch_size · scale_factor)² = 256 · 32² =
    /// 262 144` (= 512² square baseline) for LFM2-VL. Inputs above
    /// this band are scaled down preserving aspect ratio.
    pub image_max_pixels: usize,
}

impl VisionEncoderConfig {
    /// Read the vision-encoder config from a multimodal_projector
    /// GGUF's `clip.vision.*` metadata. Errors on any missing
    /// required key — the LFM2.5-VL bundles all carry the full
    /// set, so a missing key indicates a corrupt or
    /// non-vision-encoder mmproj.
    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        let has_vision = gguf.get_bool(KEY_HAS_VISION).unwrap_or(false);
        anyhow::ensure!(
            has_vision,
            "mmproj GGUF missing or false `{KEY_HAS_VISION}`; \
             not a vision encoder"
        );
        let n_layer = gguf
            .get_u32(KEY_N_LAYER)
            .with_context(|| format!("missing `{KEY_N_LAYER}`"))? as usize;
        let n_embd = gguf
            .get_u32(KEY_N_EMBD)
            .with_context(|| format!("missing `{KEY_N_EMBD}`"))? as usize;
        let n_ff = gguf
            .get_u32(KEY_N_FF)
            .with_context(|| format!("missing `{KEY_N_FF}`"))? as usize;
        let n_head = gguf
            .get_u32(KEY_N_HEAD)
            .with_context(|| format!("missing `{KEY_N_HEAD}`"))? as usize;
        let eps = gguf
            .get_f32(KEY_LN_EPS)
            .with_context(|| format!("missing `{KEY_LN_EPS}`"))?;
        let image_size =
            gguf.get_u32(KEY_IMAGE_SIZE)
                .with_context(|| format!("missing `{KEY_IMAGE_SIZE}`"))? as usize;
        let patch_size =
            gguf.get_u32(KEY_PATCH_SIZE)
                .with_context(|| format!("missing `{KEY_PATCH_SIZE}`"))? as usize;
        anyhow::ensure!(
            n_head > 0 && n_embd.is_multiple_of(n_head),
            "n_embd ({n_embd}) must be a positive multiple of n_head ({n_head})"
        );
        anyhow::ensure!(
            patch_size > 0 && image_size.is_multiple_of(patch_size),
            "image_size ({image_size}) must be a positive multiple of patch_size ({patch_size})"
        );
        let n_trained_patches = (image_size / patch_size).pow(2);

        let projection_dim =
            gguf.get_u32(KEY_PROJECTION_DIM)
                .with_context(|| format!("missing `{KEY_PROJECTION_DIM}`"))? as usize;
        let scale_factor =
            gguf.get_u32(KEY_SCALE_FACTOR)
                .with_context(|| format!("missing `{KEY_SCALE_FACTOR}`"))? as usize;
        anyhow::ensure!(
            scale_factor > 0,
            "scale_factor ({scale_factor}) must be > 0; a zero or missing value would \
             collapse the pixel band and trip a divide-by-zero in `pixel_shuffle`",
        );
        let image_mean = read_rgb_array(gguf, KEY_IMAGE_MEAN)?;
        let image_std = read_rgb_array(gguf, KEY_IMAGE_STD)?;

        // LFM2-VL dynamic-resolution bounds, hardcoded per
        // `clip_model.h::set_limit_image_tokens(64, 256)`. The
        // `(min, max) = (64, 256) tokens` band converts to pixel
        // bounds via `n_tokens · (patch_size · scale_factor)²`.
        let n_tokens_min: usize = 64;
        let n_tokens_max: usize = 256;
        let pixels_per_token = (patch_size * scale_factor).pow(2);
        let image_min_pixels = n_tokens_min * pixels_per_token;
        let image_max_pixels = n_tokens_max * pixels_per_token;

        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            eps,
            image_size,
            patch_size,
            n_trained_patches,
            projection_dim,
            scale_factor,
            image_mean,
            image_std,
            image_min_pixels,
            image_max_pixels,
        })
    }
}

/// Patch-embed Conv2D weights, pre-transposed at load time into
/// row-major `[in_dim × n_embd]` where `in_dim = 3·patch_size²`.
/// The conv-with-stride-equal-to-kernel-size collapses to a per-
/// patch matmul; storing the kernel in `[in × out]` row-major
/// matches the layout `cpu::matmul_f32(a, b, c, m, n, k)` reads
/// for `B` (the function does standard `C = A · B`, NOT `A · B^T`),
/// so the forward pass calls it directly with no per-image
/// transpose.
#[derive(Clone)]
pub struct PatchEmbedWeights {
    /// Row-major `[in_dim × n_embd]` kernel ready for matmul. Built
    /// from `v.patch_embd.weight` once at load time.
    pub conv_w: Vec<f32>,
    /// `v.patch_embd.bias` — `[n_embd]`.
    pub conv_b: Vec<f32>,
}

/// One ViT block's weight set. Pre-norm self-attention + GELU MLP
/// with residual connections; matches llama.cpp's
/// `clip.cpp` ViT block.
pub struct VitBlockWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub q_w: MmapWeight,
    pub q_b: Vec<f32>,
    pub k_w: MmapWeight,
    pub k_b: Vec<f32>,
    pub v_w: MmapWeight,
    pub v_b: Vec<f32>,
    pub o_w: MmapWeight,
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub ffn_up_w: MmapWeight,
    pub ffn_up_b: Vec<f32>,
    pub ffn_down_w: MmapWeight,
    pub ffn_down_b: Vec<f32>,
}

/// 2-layer MLP projector that maps the pixel-shuffled ViT output
/// into the LLM embedding dim. `mm.1` is `[n_embd·scale_factor² →
/// projection_dim·2]`, GELU, `mm.2` is `[projection_dim·2 →
/// projection_dim]` per llama.cpp's LFM2 projector layout.
#[derive(Clone)]
pub struct ProjectorWeights {
    pub mm1_w: MmapWeight,
    pub mm1_b: Vec<f32>,
    pub mm2_w: MmapWeight,
    pub mm2_b: Vec<f32>,
}

impl ProjectorWeights {
    pub(crate) fn forward(&self, pooled: &[f32], projection_dim: usize) -> Vec<f32> {
        let in_dim = self.mm1_w.cols;
        let mid_dim = self.mm1_w.rows;
        let out_dim = projection_dim;
        if in_dim == 0 || pooled.is_empty() || !pooled.len().is_multiple_of(in_dim) {
            return Vec::new();
        }
        let n_tokens = pooled.len() / in_dim;

        let mut mid = vec![0f32; n_tokens * mid_dim];
        let mut out = vec![0f32; n_tokens * out_dim];

        // The same int8 GEMM the ViT blocks use; the f32 path stays as the fallback.
        let mut q8 = Q8Act::for_dims(&[in_dim, mid_dim], n_tokens);
        q8.quantize(pooled, in_dim, n_tokens);
        if !q8.matmul(&self.mm1_w, &mut mid) {
            self.mm1_w.batched_matmul(pooled, &mut mid, n_tokens);
        }
        par_rows_n(&mut mid, mid_dim, |_, mid_row| {
            crate::backend::cpu::add_inplace(mid_row, &self.mm1_b);
            crate::backend::cpu::gelu_inplace(mid_row);
        });
        q8.quantize(&mid, mid_dim, n_tokens);
        if !q8.matmul(&self.mm2_w, &mut out) {
            self.mm2_w.batched_matmul(&mid, &mut out, n_tokens);
        }
        par_rows_n(&mut out, out_dim, |_, out_row| {
            crate::backend::cpu::add_inplace(out_row, &self.mm2_b);
        });
        out
    }
}

/// Intermediates of the first ViT block, in the layouts both encoders share
/// (`[tokens, width]` row-major), for comparing the Hexagon encoder with this
/// one stage by stage (see `examples/hexagon_vit_probe.rs`).
#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct VitStageDump {
    /// The tokens going in (patch embedding plus position embedding).
    pub x0: Vec<f32>,
    /// Q, K and V with their biases.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// The attention output, and the output projection with its bias (both
    /// before the residual add).
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    /// The second LayerNorm's output, the feed-forward's activation (after
    /// the up projection, its bias and GELU) and its down projection with bias
    /// (before the residual add).
    pub ln2: Vec<f32>,
    pub ffn_mid: Vec<f32>,
    pub ffn_out: Vec<f32>,
    /// The tokens leaving the block.
    pub tokens: Vec<f32>,
}

/// All vision-encoder weights, loaded from a multimodal_projector
/// GGUF in one shot. Mirrors `audio_encoder::AudioEncoderWeights`
/// for the audio counterpart.
///
/// **Memory note.** Every linear weight is dequantised to f32 at
/// load time (same trade-off `audio_encoder.rs` makes). For a
/// LFM2.5-VL-450M Q8_0 mmproj this means ~94 MB → ~376 MB
/// resident. Acceptable on desktop / server but a concern on
/// mobile (Android/iOS) where the eventual VL consumers live.
/// **TODO(VL perf):** when memory pressure shows up, swap the
/// per-block `MmapWeight` for `QuantWeight` (already exists in
/// `audio_decoder.rs`) and route the forward pass through the
/// quantised GEMV path. The public accessor
/// `CeraEngine::vision_encoder()` doesn't change shape — only
/// internal field types — so the swap is internal.
pub struct VisionEncoderWeights {
    pub config: VisionEncoderConfig,
    pub patch_embed: PatchEmbedWeights,
    /// `v.position_embd.weight` — `[n_patches × n_embd]` flattened
    /// row-major. Learnable absolute position embeddings; added
    /// to the patch tokens before block 0. GGUF reports the
    /// shape as `[n_embd, n_patches]` (innermost-first
    /// convention); `to_f32_vec` returns the data row-major over
    /// `[n_patches × n_embd]` so `position_embed[p * n_embd + i]`
    /// indexes patch `p`'s embedding dim `i`.
    pub position_embed: Vec<f32>,
    pub blocks: Vec<VitBlockWeights>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    pub projector: ProjectorWeights,
}

impl VisionEncoderWeights {
    /// Load every vision-encoder tensor from a multimodal_projector
    /// GGUF. Errors if any required tensor or metadata key is
    /// missing — no silent defaults. Per-tensor `with_context`
    /// surfaces the first missing name at the top of the error
    /// chain.
    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        let config = VisionEncoderConfig::from_gguf(gguf)?;

        // Patch embed Conv2D kernel `[kw, kh, ic, oc]` in GGUF
        // layout — `kw` innermost (stride 1), `oc` outermost
        // (stride `kw·kh·ic`), matching the same convention
        // `F32Weight::from_tensor` uses for 2D shapes (rightmost-
        // in-shape is the outermost axis).
        //
        // The forward pass treats Conv2D-with-stride-equal-to-
        // kernel-size as a per-patch GEMV against a
        // `[n_embd × (3·patch_size²)]` matrix; transpose into
        // that row-major layout once here so `encode_image`
        // doesn't have to rebuild it on every call.
        let patch_t = gguf
            .get_tensor("v.patch_embd.weight")
            .context("loading v.patch_embd.weight")?;
        let patch_shape = patch_t.shape().to_vec();
        anyhow::ensure!(
            patch_shape.len() == 4
                && patch_shape[0] == config.patch_size
                && patch_shape[1] == config.patch_size
                && patch_shape[2] == 3
                && patch_shape[3] == config.n_embd,
            "v.patch_embd.weight shape {patch_shape:?} != [patch_size={}, patch_size={}, 3, n_embd={}]",
            config.patch_size,
            config.patch_size,
            config.n_embd,
        );
        let raw = patch_t.to_f32_vec();
        let p = config.patch_size;
        let in_dim = 3 * p * p;
        let out_dim = config.n_embd;
        // Row-major destination `[in_dim × out_dim]` where rows are
        // input flat-index (`c*p² + kh*p + kw`) and cols are output
        // channels (oc). This is the layout `cpu::matmul_f32` reads
        // for its `B` argument when computing `C = A · B` standard
        // (NOT `A · B^T` — the function name is plain `matmul_f32`,
        // its docs/test confirm `c[i*n+j] = Σ_k a[i*k+k]·b[k*n+j]`).
        // Building the kernel directly in this layout means the
        // forward pass can call `matmul_f32(patches, conv_w, …)` with
        // no per-image transpose. Source GGUF stride:
        // `linear(kw, kh, c, oc) = kw + p·kh + p²·c + p²·3·oc`.
        let mut conv_w = vec![0f32; in_dim * out_dim];
        for oc in 0..out_dim {
            for c in 0..3 {
                for kh in 0..p {
                    for kw in 0..p {
                        let src = kw + p * kh + p * p * c + p * p * 3 * oc;
                        let in_idx = c * p * p + kh * p + kw;
                        conv_w[in_idx * out_dim + oc] = raw[src];
                    }
                }
            }
        }
        let conv_b = load_vec_f32(gguf, "v.patch_embd.bias")?;
        anyhow::ensure!(
            conv_b.len() == config.n_embd,
            "v.patch_embd.bias len ({}) != n_embd ({})",
            conv_b.len(),
            config.n_embd,
        );
        let patch_embed = PatchEmbedWeights { conv_w, conv_b };

        // Position embedding `[n_trained_patches × n_embd]`. The
        // tensor stores embeddings for the *trained* grid only;
        // dynamic-resolution inputs interpolate this into a
        // per-call dynamic grid via [`interpolate_pos_embed_2d`].
        // MmapWeight would also work but treating it as a flat
        // Vec<f32> keeps the indexing readable
        // (`position_embed[p * n_embd + i]`).
        let pos_t = gguf
            .get_tensor("v.position_embd.weight")
            .context("loading v.position_embd.weight")?;
        let pos_shape = pos_t.shape();
        anyhow::ensure!(
            pos_shape.len() == 2
                && pos_shape[0] == config.n_embd
                && pos_shape[1] == config.n_trained_patches,
            "v.position_embd.weight shape {pos_shape:?} != [n_embd={}, n_trained_patches={}]",
            config.n_embd,
            config.n_trained_patches,
        );
        let position_embed = pos_t.to_f32_vec();

        // ── ViT blocks ──
        let mut blocks = Vec::with_capacity(config.n_layer);
        for il in 0..config.n_layer {
            blocks.push(load_vit_block(gguf, il, &config)?);
        }

        // Post-final-block layer norm.
        let post_ln_w = load_vec_f32(gguf, "v.post_ln.weight")?;
        let post_ln_b = load_vec_f32(gguf, "v.post_ln.bias")?;
        anyhow::ensure!(
            post_ln_w.len() == config.n_embd && post_ln_b.len() == config.n_embd,
            "v.post_ln {{weight,bias}} len ({}, {}) != n_embd ({})",
            post_ln_w.len(),
            post_ln_b.len(),
            config.n_embd,
        );

        // ── Projector (mm.1, mm.2) ──
        // Shape relationships we encode:
        //   mm.1: [intermediate_dim, n_embd * scale_factor²]
        //   mm.2: [projection_dim,   intermediate_dim]
        // The `intermediate_dim` is **derived from mm.1.weight.rows**
        // rather than hardcoded as `projection_dim * 2`. The "× 2"
        // is llama.cpp's LFM2 projector convention but isn't
        // surfaced in any GGUF metadata key, so deriving from the
        // actual tensor shape keeps the loader robust to a future
        // LFM2-VL variant that picks a different intermediate
        // width while still asserting the mm.1 → mm.2 size match.
        let mm1_w = wt_f32(gguf, "mm.1.weight")?;
        let mm1_b = load_vec_f32(gguf, "mm.1.bias")?;
        let mm2_w = wt_f32(gguf, "mm.2.weight")?;
        let mm2_b = load_vec_f32(gguf, "mm.2.bias")?;
        let mm1_in_dim = config.n_embd * config.scale_factor.pow(2);
        let intermediate_dim = mm1_w.rows;
        anyhow::ensure!(
            mm1_w.cols == mm1_in_dim,
            "mm.1.weight cols ({}) != n_embd*sf² ({mm1_in_dim})",
            mm1_w.cols,
        );
        anyhow::ensure!(
            mm1_b.len() == intermediate_dim,
            "mm.1.bias len ({}) != mm.1.weight.rows ({intermediate_dim})",
            mm1_b.len(),
        );
        anyhow::ensure!(
            mm2_w.cols == intermediate_dim,
            "mm.2.weight cols ({}) != mm.1.weight.rows ({intermediate_dim}) — \
             projector mm.1→mm.2 dimensions don't line up",
            mm2_w.cols,
        );
        anyhow::ensure!(
            mm2_w.rows == config.projection_dim,
            "mm.2.weight rows ({}) != projection_dim ({})",
            mm2_w.rows,
            config.projection_dim,
        );
        anyhow::ensure!(
            mm2_b.len() == config.projection_dim,
            "mm.2.bias len ({}) != projection_dim ({})",
            mm2_b.len(),
            config.projection_dim,
        );

        let projector = ProjectorWeights {
            mm1_w,
            mm1_b,
            mm2_w,
            mm2_b,
        };

        let weights = Self {
            config,
            patch_embed,
            position_embed,
            blocks,
            post_ln_w,
            post_ln_b,
            projector,
        };
        weights.prepare_int8();
        Ok(weights)
    }

    /// Build the smmla-repacked copy of every Q8_0 linear now, so the first image does not pay for
    /// it. A no-op unless the int8 path is on and the host has i8mm.
    fn prepare_int8(&self) {
        let cfg = &self.config;
        let dims = [cfg.n_embd, cfg.n_ff];
        if !Q8Act::for_dims(&dims, 0).enabled {
            return;
        }
        for b in &self.blocks {
            for w in [&b.q_w, &b.k_w, &b.v_w, &b.o_w, &b.ffn_up_w, &b.ffn_down_w] {
                let _ = w.smmla_repacked();
            }
        }
        let _ = self.projector.mm1_w.smmla_repacked();
        let _ = self.projector.mm2_w.smmla_repacked();
    }

    /// Run the vision encoder + projector on a preprocessed image
    /// at the dynamic patch grid the preprocessor selected.
    ///
    /// `pixels` is `[3 × target_h × target_w]` f32 NCHW
    /// (RGB, normalised via `image_mean` / `image_std`).
    /// `(grid_w, grid_h)` are the patch-grid dimensions:
    /// `target_w / patch_size` × `target_h / patch_size`. The
    /// preprocessor returns these as part of `PreprocessedImage`.
    ///
    /// Output: `[n_image_tokens × projection_dim]` f32 where
    /// `n_image_tokens = (grid_w / scale_factor) · (grid_h /
    /// scale_factor)`. For a 1024×771 input on LFM2.5-VL-450M
    /// (target 576×416 → grid 36×26): 18·13 = 234 image tokens
    /// of 1024 dims each.
    ///
    /// Pipeline:
    /// `pixels → patch_embed → +interpolated_pos_embed → 12 × ViT
    /// block → post_ln → pixel-shuffle 2×2 → mm.1+GELU → mm.2`.
    pub fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        anyhow::ensure!(grid_w > 0 && grid_h > 0, "grid dims must be > 0");
        anyhow::ensure!(
            cfg.scale_factor > 0,
            "vision encoder config has scale_factor=0; refusing to divide by zero"
        );
        anyhow::ensure!(
            grid_w.is_multiple_of(cfg.scale_factor) && grid_h.is_multiple_of(cfg.scale_factor),
            "grid {grid_w}×{grid_h} not divisible by scale_factor ({})",
            cfg.scale_factor,
        );
        // Compute target image / patch counts with overflow checks —
        // a malformed caller (or 32-bit target like wasm32 with a
        // huge grid) shouldn't silently wrap and bypass the
        // pixels.len() invariant below.
        let target_w = grid_w
            .checked_mul(cfg.patch_size)
            .ok_or_else(|| anyhow::anyhow!("grid_w·patch_size overflow"))?;
        let target_h = grid_h
            .checked_mul(cfg.patch_size)
            .ok_or_else(|| anyhow::anyhow!("grid_h·patch_size overflow"))?;
        let n_pix = target_w
            .checked_mul(target_h)
            .and_then(|x| x.checked_mul(3))
            .ok_or_else(|| anyhow::anyhow!("3·target_w·target_h overflow"))?;
        anyhow::ensure!(
            pixels.len() == n_pix,
            "encode_image: pixels.len() {} != 3·target_w·target_h ({n_pix})",
            pixels.len()
        );
        let n_patches = grid_w
            .checked_mul(grid_h)
            .ok_or_else(|| anyhow::anyhow!("grid_w·grid_h overflow"))?;
        let mut prof = Prof::new();
        // 1. Patch embed: [3, H, W] → [n_patches, n_embd]
        let t = prof.start();
        let mut tokens = patch_embed_compute(pixels, &self.patch_embed, cfg, grid_w, grid_h);
        prof.stop("patch_embed", t);
        // 2. Add position embeddings — interpolate from the trained
        //    grid when the dynamic grid differs.
        let pos = self.resolved_position_embed(grid_w, grid_h);
        debug_assert_eq!(pos.len(), n_patches * cfg.n_embd);
        for (t, p) in tokens.iter_mut().zip(pos.iter()) {
            *t += *p;
        }

        // 3. 12 × ViT block. Allocate scratch sized to the dynamic
        //    grid and reuse across blocks.
        let mut scratch = VitScratch::new(cfg, n_patches);
        for block in &self.blocks {
            self.vit_block_forward(&mut tokens, block, &mut scratch, n_patches, &mut prof);
        }

        // 4. post_ln (per token).
        let t = prof.start();
        par_rows_n(&mut tokens, cfg.n_embd, |_, row| {
            crate::backend::cpu::layer_norm_inplace(row, &self.post_ln_w, &self.post_ln_b, cfg.eps);
        });

        // 5. Pixel-shuffle scale_factor² over the dynamic grid →
        //    `(grid_w/sf) · (grid_h/sf)` tokens with sf² channel
        //    inflation.
        let pooled = pixel_shuffle(&tokens, cfg, grid_w, grid_h);
        prof.stop("post_ln_shuffle", t);

        // 6. Projector: mm.1 + GELU + mm.2.
        let t = prof.start();
        let out = self.projector_forward(&pooled, cfg);
        prof.stop("projector", t);
        prof.report();
        Ok(out)
    }

    /// The `[patches, n_embd]` tokens after patch embedding, the position
    /// embeddings and the first `n_blocks` ViT blocks (before the final
    /// LayerNorm, pooling and projector): the reference the Hexagon encoder is
    /// compared with block by block.
    #[doc(hidden)]
    pub fn debug_blocks(
        &self,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
        n_blocks: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_patches = grid_w * grid_h;
        anyhow::ensure!(
            pixels.len() == n_patches * cfg.patch_size * cfg.patch_size * 3,
            "debug_blocks: {} pixels for a {grid_w}x{grid_h} grid",
            pixels.len()
        );
        let mut tokens = patch_embed_compute(pixels, &self.patch_embed, cfg, grid_w, grid_h);
        let pos = self.resolved_position_embed(grid_w, grid_h);
        for (t, p) in tokens.iter_mut().zip(pos.iter()) {
            *t += *p;
        }
        let mut scratch = VitScratch::new(cfg, n_patches);
        for block in self.blocks.iter().take(n_blocks) {
            self.vit_block_forward(
                &mut tokens,
                block,
                &mut scratch,
                n_patches,
                &mut Prof::off(),
            );
        }
        Ok(tokens)
    }

    /// Run only the first ViT block and return its intermediates.
    #[doc(hidden)]
    pub fn debug_first_block(
        &self,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
    ) -> Result<VitStageDump> {
        let cfg = &self.config;
        let n_patches = grid_w * grid_h;
        anyhow::ensure!(
            pixels.len() == n_patches * cfg.patch_size * cfg.patch_size * 3,
            "debug_first_block: {} pixels for a {grid_w}x{grid_h} grid",
            pixels.len()
        );
        let mut tokens = patch_embed_compute(pixels, &self.patch_embed, cfg, grid_w, grid_h);
        let pos = self.resolved_position_embed(grid_w, grid_h);
        for (t, p) in tokens.iter_mut().zip(pos.iter()) {
            *t += *p;
        }
        let x0 = tokens.clone();
        let mut scratch = VitScratch::new(cfg, n_patches);
        let block = self
            .blocks
            .first()
            .ok_or_else(|| anyhow::anyhow!("no ViT blocks"))?;
        self.vit_block_forward(
            &mut tokens,
            block,
            &mut scratch,
            n_patches,
            &mut Prof::off(),
        );
        Ok(VitStageDump {
            x0,
            q: scratch.q,
            k: scratch.k,
            v: scratch.v,
            attn_out: scratch.attn_out,
            attn_proj: scratch.attn_proj,
            ln2: scratch.pre_norm,
            ffn_mid: scratch.ffn_mid,
            ffn_out: scratch.ffn_out,
            tokens,
        })
    }

    /// Return position embeddings for the requested dynamic patch
    /// grid, interpolating from the trained square grid when
    /// needed. Borrows the trained tensor directly when it matches
    /// to avoid the f32 copy.
    fn resolved_position_embed(&self, grid_w: usize, grid_h: usize) -> std::borrow::Cow<'_, [f32]> {
        let cfg = &self.config;
        let trained_side = (cfg.n_trained_patches as f64).sqrt().round() as usize;
        debug_assert_eq!(
            trained_side * trained_side,
            cfg.n_trained_patches,
            "non-square trained pos-embed grid is not currently supported"
        );
        if grid_w == trained_side && grid_h == trained_side {
            std::borrow::Cow::Borrowed(self.position_embed.as_slice())
        } else {
            std::borrow::Cow::Owned(interpolate_pos_embed_2d(
                &self.position_embed,
                trained_side,
                trained_side,
                grid_h,
                grid_w,
                cfg.n_embd,
            ))
        }
    }

    /// Patch embed forward: a Conv2D with kernel=patch_size and
    /// stride=patch_size is mathematically equivalent to a per-
    /// patch matmul — we extract each `patch_size × patch_size × 3`
    /// chunk as a `3·patch_size²`-dim vector and matmul against
    /// the kernel pre-transposed at load time into row-major
    /// `[in_dim × n_embd]`. The patch-input layout `(c, kh, kw)`
    /// matches the kernel's input flat-index, and the
    /// `[in × out]` kernel layout matches what `cpu::matmul_f32`
    /// reads for its `B` argument (standard `C = A · B`).
    /// One ViT transformer block: pre-norm self-attention with
    /// residual + pre-norm GELU MLP with residual. In-place on
    /// `tokens` so the residual chain stays in one buffer.
    /// `n_tokens` is the dynamic patch count (`grid_w · grid_h`).
    fn vit_block_forward(
        &self,
        tokens: &mut [f32],
        block: &VitBlockWeights,
        scratch: &mut VitScratch,
        n_tokens: usize,
        prof: &mut Prof,
    ) {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let head_dim = n_embd / n_head;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Every parallel section below runs on the `RowPool` the int8 GEMM uses (`par_rows_n`), not
        // rayon: the pool's workers spin between dispatches, and a rayon fork-join issued right after
        // one pays a park/unpark against them (about 35 ms of extra tower time when only the
        // transpose was moved).

        // ── Pre-attention ──
        // Copy tokens into `pre_norm` so we can LN it without clobbering the residual.
        let t = prof.start();
        scratch.pre_norm.copy_from_slice(tokens);
        par_rows_n(&mut scratch.pre_norm, n_embd, |_, row| {
            crate::backend::cpu::layer_norm_inplace(row, &block.ln1_w, &block.ln1_b, cfg.eps);
        });
        prof.stop("ln", t);
        // Q/K/V projections batched across all tokens. The three share one input, so it is
        // quantized to Q8_0 once.
        let t = prof.start();
        scratch.q8.quantize(&scratch.pre_norm, n_embd, n_tokens);
        prof.stop("quant", t);
        let t = prof.start();
        vit_linear(
            &block.q_w,
            &scratch.pre_norm,
            &mut scratch.q,
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        vit_linear(
            &block.k_w,
            &scratch.pre_norm,
            &mut scratch.k,
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        vit_linear(
            &block.v_w,
            &scratch.pre_norm,
            &mut scratch.v,
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        prof.stop("gemm_qkv", t);

        let t = prof.start();
        {
            // Bias for all three; k and v are reached through raw pointers because `par_rows_n`
            // splits one slice, and each task touches only its own token row of each.
            let k_ptr = scratch.k.as_mut_ptr() as usize;
            let v_ptr = scratch.v.as_mut_ptr() as usize;
            par_rows_n(&mut scratch.q, n_embd, |t, q_row| {
                // SAFETY: row `t` of `k` and `v` is touched by this task alone.
                let (k_row, v_row) = unsafe {
                    (
                        core::slice::from_raw_parts_mut(
                            (k_ptr as *mut f32).add(t * n_embd),
                            n_embd,
                        ),
                        core::slice::from_raw_parts_mut(
                            (v_ptr as *mut f32).add(t * n_embd),
                            n_embd,
                        ),
                    )
                };
                crate::backend::cpu::add_inplace(q_row, &block.q_b);
                crate::backend::cpu::add_inplace(k_row, &block.k_b);
                crate::backend::cpu::add_inplace(v_row, &block.v_b);
            });
        }
        prof.stop("qkv_bias", t);

        // Multi-head scaled dot-product attention.
        let t = prof.start();
        vit_attention(
            &scratch.q,
            &scratch.k,
            &scratch.v,
            &mut scratch.attn_out,
            &mut scratch.attn_heads,
            &mut scratch.attn_kt,
            n_tokens,
            n_head,
            head_dim,
            scale,
        );
        prof.stop("attention", t);

        // Output projection + bias + residual add.
        let t = prof.start();
        scratch.q8.quantize(&scratch.attn_out, n_embd, n_tokens);
        prof.stop("quant", t);
        let t = prof.start();
        vit_linear(
            &block.o_w,
            &scratch.attn_out,
            &mut scratch.attn_proj,
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        prof.stop("gemm_o", t);
        let t = prof.start();
        add_bias_residual(tokens, &mut scratch.attn_proj, &block.o_b, n_embd);
        prof.stop("bias_res", t);

        // ── MLP ──
        let t = prof.start();
        scratch.pre_norm.copy_from_slice(tokens);
        par_rows_n(&mut scratch.pre_norm, n_embd, |_, row| {
            crate::backend::cpu::layer_norm_inplace(row, &block.ln2_w, &block.ln2_b, cfg.eps);
        });
        prof.stop("ln", t);

        let n_ff = cfg.n_ff;
        let t = prof.start();
        scratch.q8.quantize(&scratch.pre_norm, n_embd, n_tokens);
        prof.stop("quant", t);
        let t = prof.start();
        vit_linear(
            &block.ffn_up_w,
            &scratch.pre_norm,
            &mut scratch.ffn_mid,
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        prof.stop("gemm_up", t);
        let t = prof.start();
        par_rows_n(&mut scratch.ffn_mid, n_ff, |_, ff_row| {
            crate::backend::cpu::add_inplace(ff_row, &block.ffn_up_b);
            crate::backend::cpu::gelu_inplace(ff_row);
        });
        prof.stop("gelu", t);
        let t = prof.start();
        scratch.q8.quantize(&scratch.ffn_mid, n_ff, n_tokens);
        prof.stop("quant", t);
        let t = prof.start();
        vit_linear(
            &block.ffn_down_w,
            &scratch.ffn_mid,
            &mut scratch.ffn_out,
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        prof.stop("gemm_down", t);
        let t = prof.start();
        add_bias_residual(tokens, &mut scratch.ffn_out, &block.ffn_down_b, n_embd);
        prof.stop("bias_res", t);
    }

    /// Projector forward: pixel-shuffled `[64, 3072]` → `mm.1`
    /// (3072 → 2048) + GELU → `mm.2` (2048 → 1024). Output:
    /// `[64, 1024]` flattened.
    fn projector_forward(&self, pooled: &[f32], cfg: &VisionEncoderConfig) -> Vec<f32> {
        self.projector.forward(pooled, cfg.projection_dim)
    }
}

/// Per-phase wall time of one `encode_image` call, on with `CERA_VIT_PROFILE=1` and printed to stderr
/// when the call ends. Off, `start` returns `None` and costs nothing.
struct Prof {
    on: bool,
    acc: Vec<(&'static str, std::time::Duration)>,
}

impl Prof {
    fn new() -> Self {
        Self {
            on: crate::backend::cpu_features::env_enabled("CERA_VIT_PROFILE"),
            acc: Vec::new(),
        }
    }

    fn off() -> Self {
        Self {
            on: false,
            acc: Vec::new(),
        }
    }

    fn start(&self) -> Option<crate::time::Instant> {
        self.on.then(crate::time::Instant::now)
    }

    fn stop(&mut self, key: &'static str, t: Option<crate::time::Instant>) {
        let Some(t) = t else { return };
        let d = t.elapsed();
        match self.acc.iter_mut().find(|(k, _)| *k == key) {
            Some((_, a)) => *a += d,
            None => self.acc.push((key, d)),
        }
    }

    fn report(&self) {
        if !self.on {
            return;
        }
        let total: std::time::Duration = self.acc.iter().map(|(_, d)| *d).sum();
        let mut line = format!("vit profile (sum {:.1} ms):", total.as_secs_f64() * 1e3);
        for (k, d) in &self.acc {
            line.push_str(&format!(" {k} {:.1}", d.as_secs_f64() * 1e3));
        }
        eprintln!("{line}");
    }
}

/// Per-block scratch buffers for the ViT forward pass. Allocated
/// once per `encode_image` and reused across all 12 blocks; sized
/// to the **dynamic** patch count picked by the preprocessor
/// (`grid_w · grid_h`), not the trained-grid count.
struct VitScratch {
    pre_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Attention output `[n_tokens × n_embd]` — reused across all
    /// 12 blocks (was previously allocated per block).
    attn_out: Vec<f32>,
    /// Per-head attention output in the flash kernel's `[head][token][head_dim]` layout.
    attn_heads: Vec<f32>,
    /// The keys of every head transposed to `[head][head_dim][token]` (tokens padded to a multiple of 8)
    /// for `cpu::vit_attention_chunk_neon`; empty where that kernel is not used.
    attn_kt: Vec<f32>,
    attn_proj: Vec<f32>,
    ffn_mid: Vec<f32>,
    ffn_out: Vec<f32>,
    /// Reusable row dequantization buffer sized to `max(n_embd, n_ff)`
    /// to avoid per-matmul heap allocations across all blocks.
    dequant_scratch: Vec<f32>,
    /// Int8 activation buffers for the Q8_0 linears.
    q8: Q8Act,
}

impl VitScratch {
    fn new(cfg: &VisionEncoderConfig, n_tokens: usize) -> Self {
        let n_pe = n_tokens * cfg.n_embd;
        let n_pf = n_tokens * cfg.n_ff;
        let max_dim = cfg.n_embd.max(cfg.n_ff);
        Self {
            pre_norm: vec![0.0; n_pe],
            q: vec![0.0; n_pe],
            k: vec![0.0; n_pe],
            v: vec![0.0; n_pe],
            attn_out: vec![0.0; n_pe],
            attn_heads: vec![0.0; attn_heads_len(n_tokens, cfg.n_head, cfg.n_embd / cfg.n_head)],
            attn_kt: vec![0.0; attn_kt_len(n_tokens, cfg.n_head, cfg.n_embd / cfg.n_head)],
            attn_proj: vec![0.0; n_pe],
            ffn_mid: vec![0.0; n_pf],
            ffn_out: vec![0.0; n_pe],
            dequant_scratch: vec![0.0; max_dim],
            q8: Q8Act::new(cfg, n_tokens),
        }
    }
}

/// Int8 activations for the tower's Q8_0 linears.
///
/// Every linear in the tower is a Q8_0 weight applied to f32 activations. The scalar path
/// dequantizes each weight row to f32 and runs one `dot_f32` per (row, token) pair, which is
/// compute-bound at about 130 GFLOP/s on the phone. This quantizes the activations to Q8_0 once per
/// distinct input (the same step llama.cpp's CPU backend takes for a Q8_0 weight) and runs the
/// batched int8 GEMM the LLM prefill uses (`i8mm` on cores that have it).
///
/// The kernel writes `[rows][tokens]`; the tower wants `[tokens][rows]`, so the result is
/// transposed on the way out. `CERA_VIT_INT8=0` forces the f32 path, for A/B runs.
struct Q8Act {
    enabled: bool,
    /// Feature dim and token count of the activations last passed to [`Self::quantize`].
    dim: usize,
    n: usize,
    scales: Vec<f32>,
    quants: Vec<i8>,
    /// GEMM output `[rows][tokens]`, before the transpose into the caller's `[tokens][rows]`.
    out_t: Vec<f32>,
}

impl Q8Act {
    fn new(cfg: &VisionEncoderConfig, n_tokens: usize) -> Self {
        Self::for_dims(&[cfg.n_embd, cfg.n_ff], n_tokens)
    }

    /// An int8 activation buffer for token rows of any of `dims` floats.
    fn for_dims(dims: &[usize], n_tokens: usize) -> Self {
        let max_dim = dims.iter().copied().max().unwrap_or(0);
        let enabled = crate::backend::cpu::int8_gemm_available()
            && dims.iter().all(|d| d.is_multiple_of(32))
            && !crate::backend::cpu_features::env_disabled("CERA_VIT_INT8");
        let (n_scales, n_quants, n_out) = if enabled {
            (
                n_tokens * (max_dim / 32),
                n_tokens * max_dim,
                n_tokens * max_dim,
            )
        } else {
            (0, 0, 0)
        };
        Self {
            enabled,
            dim: 0,
            n: 0,
            scales: vec![0.0; n_scales],
            quants: vec![0; n_quants],
            out_t: vec![0.0; n_out],
        }
    }

    /// Quantize `n` token rows of `dim` floats each. Call once per distinct input, before the
    /// [`vit_linear`] calls that consume it.
    fn quantize(&mut self, x: &[f32], dim: usize, n: usize) {
        if !self.enabled {
            return;
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            crate::model::transformer::quantize_rows(x, dim, n, &mut self.scales, &mut self.quants);
            self.dim = dim;
            self.n = n;
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = (x, dim, n);
    }

    /// `y[tokens][rows] = x · wᵀ` from the activations last quantized. Returns `false` (and leaves
    /// `y` untouched) when this weight or host cannot take the int8 path.
    fn matmul(&mut self, w: &MmapWeight, y: &mut [f32]) -> bool {
        if !self.enabled
            || w.dtype != crate::tensor::DType::Q8_0
            || w.cols != self.dim
            || self.n == 0
        {
            return false;
        }
        let (rows, n, dim) = (w.rows, self.n, self.dim);
        debug_assert_eq!(y.len(), rows * n);
        // Repacked smmla kernel (i8mm hosts): 8 weight rows by 4 tokens per tile, written straight
        // into `y` row-major, so there is no transpose pass either.
        #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(has_blas)))]
        if let Some((packed, scales)) = w.smmla_repacked()
            && crate::backend::cpu::gemm_preq_repacked_q8_0_smmla_rowmajor_dispatch(
                packed,
                scales,
                &self.scales[..n * (dim / 32)],
                &self.quants[..n * dim],
                y,
                n,
                rows,
                dim,
            )
        {
            return true;
        }
        let out_t = &mut self.out_t[..rows * n];
        let ran = crate::backend::cpu::gemm_preq_dispatch(
            crate::tensor::DType::Q8_0,
            w.data(),
            &self.scales[..n * (dim / 32)],
            &self.quants[..n * dim],
            out_t,
            rows,
            n,
            dim,
        );
        if ran {
            transpose_rows_tokens(out_t, y, rows, n);
        }
        ran
    }
}

/// `dst[t][r] = src[r][t]` for `src` `[rows][n]`, one destination row (token) per task on the
/// same `RowPool` the GEMM runs on. A rayon fan-out here costs a park/unpark per call while the
/// GEMM's workers are still spinning, which measured about 0.7 ms for a 2 MB transpose.
fn transpose_rows_tokens(src: &[f32], dst: &mut [f32], rows: usize, n: usize) {
    crate::backend::cpu::par_rows_n(dst, rows, 16, |(t, dst_row)| {
        for (r, d) in dst_row.iter_mut().enumerate() {
            *d = src[r * n + t];
        }
    });
}

/// Queries per attention task: the flash kernel's own query block of 32, so each task runs whole
/// blocks. The kernel reads Q as stride-`n_queries` columns whenever `q_stride == n_queries`, and Q
/// here is token-major with `q_stride == n_embd`, so a chunk must never hold exactly `n_embd`
/// queries; a (test-sized) embedding of 32 or fewer gets a chunk one smaller than it.
fn attn_q_chunk(n_embd: usize) -> usize {
    assert!(n_embd > 1, "vit_attention: n_embd must be at least 2");
    n_embd.saturating_sub(1).min(32)
}

/// Size of the transposed-keys scratch the NEON attention kernel needs, or 0 where it is not used.
fn attn_kt_len(n_tokens: usize, n_head: usize, head_dim: usize) -> usize {
    if cfg!(target_arch = "aarch64") && head_dim.is_multiple_of(8) && head_dim <= 128 {
        n_head * head_dim * n_tokens.next_multiple_of(8)
    } else {
        0
    }
}

/// `[heads][padded tokens][head_dim]` scratch length for [`vit_attention`].
fn attn_heads_len(n_tokens: usize, n_head: usize, head_dim: usize) -> usize {
    n_head * n_tokens.next_multiple_of(attn_q_chunk(n_head * head_dim)) * head_dim
}

/// Bidirectional multi-head attention over `n_tokens` tokens. `q`, `k`, `v` and `out` are
/// `[n_tokens][n_head * head_dim]`; head `h` of token `t` is the `head_dim` slice at
/// `t * n_embd + h * head_dim`.
///
/// Runs the CPU flash-attention kernel (queries in blocks that share one pass over K and V, online
/// softmax) once per (head, 32-query chunk) on the `RowPool`. The previous per-query loop re-read a
/// head's whole K and V (about 400 KB) from L2 for every query, which made attention memory-bound at
/// about 90 GFLOP/s. `heads` receives each head's output in the kernel's `[head][token][head_dim]`
/// layout before it is scattered into `out`.
#[allow(clippy::too_many_arguments)]
fn vit_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: &mut [f32],
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))] kt: &mut [f32],
    n_tokens: usize,
    n_head: usize,
    head_dim: usize,
    scale: f32,
) {
    let n_embd = n_head * head_dim;
    let chunk_q = attn_q_chunk(n_embd);
    let padded = n_tokens.next_multiple_of(chunk_q);
    let n_chunks = padded / chunk_q;
    let heads = &mut heads[..attn_heads_len(n_tokens, n_head, head_dim)];

    // The ViT-shaped NEON kernel where it applies: keys transposed once per head, then 8-query tiles.
    #[cfg(target_arch = "aarch64")]
    if attn_kt_len(n_tokens, n_head, head_dim) > 0 {
        let npad = n_tokens.next_multiple_of(8);
        let kt = &mut kt[..n_head * head_dim * npad];
        par_rows_n(kt, head_dim * npad, |h, kt_h| {
            crate::backend::cpu::vit_transpose_keys(
                k,
                kt_h,
                n_tokens,
                npad,
                n_embd,
                h * head_dim,
                head_dim,
            );
        });
        let kt = &*kt;
        par_rows_n(heads, chunk_q * head_dim, |r, chunk| {
            let (h, c) = (r / n_chunks, r % n_chunks);
            let t0 = c * chunk_q;
            let nq = (n_tokens - t0).min(chunk_q);
            crate::backend::cpu::vit_attention_chunk_neon(
                q,
                &kt[h * head_dim * npad..(h + 1) * head_dim * npad],
                v,
                &mut chunk[..nq * head_dim],
                t0,
                nq,
                n_tokens,
                npad,
                n_embd,
                h * head_dim,
                head_dim,
                scale,
            );
        });
        scatter_heads(heads, out, padded, n_tokens, n_head, head_dim);
        return;
    }

    par_rows_n(heads, chunk_q * head_dim, |r, chunk| {
        let (h, c) = (r / n_chunks, r % n_chunks);
        let t0 = c * chunk_q;
        let nq = (n_tokens - t0).min(chunk_q);
        // Non-causal: the kernel attends to `start_pos + n_queries` keys, so this chunk sees all of them.
        #[cfg(target_arch = "aarch64")]
        crate::backend::cpu::flash_attention_gqa_cpu_opt(
            &q[t0 * n_embd..],
            k,
            v,
            &mut chunk[..nq * head_dim],
            h,
            1,
            nq,
            n_embd,
            n_embd,
            h * head_dim,
            head_dim,
            scale,
            n_tokens - nq,
            false,
        );
        // Only the NEON kernel reads a row-major Q (`q[token * q_stride + dim]`). The scalar, AVX2 and
        // AVX-512 kernels take the column layout `q[dim * q_stride + token]` with `q_stride == nq`, so
        // hand them this head's queries transposed, with the head at row 0 and K and V still offset.
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut q_t = vec![0.0f32; head_dim * nq];
            for j in 0..nq {
                let src = &q[(t0 + j) * n_embd + h * head_dim..][..head_dim];
                for (d, &x) in src.iter().enumerate() {
                    q_t[d * nq + j] = x;
                }
            }
            crate::backend::cpu::flash_attention_gqa_cpu_opt(
                &q_t,
                k,
                v,
                &mut chunk[..nq * head_dim],
                0,
                1,
                nq,
                nq,
                n_embd,
                h * head_dim,
                head_dim,
                scale,
                n_tokens - nq,
                false,
            );
        }
    });

    scatter_heads(heads, out, padded, n_tokens, n_head, head_dim);
}

/// Interleave the per-head attention results (`[head][padded token][head_dim]`) into token-major rows of
/// `n_head * head_dim`. The one definition of that layout, shared by the NEON and the generic paths.
fn scatter_heads(
    heads: &[f32],
    out: &mut [f32],
    padded: usize,
    n_tokens: usize,
    n_head: usize,
    head_dim: usize,
) {
    let n_embd = n_head * head_dim;
    par_rows_n(&mut out[..n_tokens * n_embd], n_embd, |t, row| {
        for h in 0..n_head {
            let src = (h * padded + t) * head_dim;
            row[h * head_dim..(h + 1) * head_dim].copy_from_slice(&heads[src..src + head_dim]);
        }
    });
}

/// `par_rows_n` with the closure shaped `(row, slice)` and a floor of 8 rows per task.
fn par_rows_n(y: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    crate::backend::cpu::par_rows_n(y, width, 8, |(i, row)| f(i, row));
}

/// `tokens[t] += proj[t] + bias` for every token row; `proj` is clobbered (it holds the bias-added
/// projection afterwards), as it was when this was written inline.
fn add_bias_residual(tokens: &mut [f32], proj: &mut [f32], bias: &[f32], width: usize) {
    let proj_ptr = proj.as_mut_ptr() as usize;
    par_rows_n(tokens, width, |t, tok_row| {
        // SAFETY: row `t` of `proj` is touched by this task alone.
        let proj_row = unsafe {
            core::slice::from_raw_parts_mut((proj_ptr as *mut f32).add(t * width), width)
        };
        crate::backend::cpu::add_inplace(proj_row, bias);
        crate::backend::cpu::add_inplace(tok_row, proj_row);
    });
}

/// One tower linear: the int8 GEMM when `q8` holds this input's quantization and the weight
/// qualifies, else the f32 path.
fn vit_linear(
    w: &MmapWeight,
    x: &[f32],
    y: &mut [f32],
    n_tokens: usize,
    q8: &mut Q8Act,
    dequant_scratch: &mut [f32],
) {
    if !q8.matmul(w, y) {
        w.batched_matmul_with_scratch(x, y, n_tokens, Some(dequant_scratch));
    }
}

/// Free-function patch-embed math, factored out of
/// [`VisionEncoderWeights::patch_embed_forward`] so unit tests can
/// drive it with hand-built kernels (no GGUF needed). Produces
/// `[n_patches × n_embd]` row-major.
///
/// Each patch's output is independent (different output rows of the
/// per-patch matmul), so under the `parallel` feature the per-patch
/// pixel-extract + matvec + bias-add are fanned across rayon's
/// worker pool. For LFM2-VL-450M (256 patches × 768 in × 768 out =
/// ~150M ops) this brings encode latency from ~30 ms to ~6 ms on an
/// 8-core M-class CPU. Without the feature, falls through to the
/// scalar single-thread path so embedded targets that disable
/// `parallel` still build.
pub(crate) fn patch_embed_compute(
    image: &[f32],
    patch_embed: &PatchEmbedWeights,
    cfg: &VisionEncoderConfig,
    grid_w: usize,
    grid_h: usize,
) -> Vec<f32> {
    let p = cfg.patch_size;
    let in_dim = 3 * p * p;
    let out_dim = cfg.n_embd;
    let target_w = grid_w * p;
    let target_h = grid_h * p;
    debug_assert_eq!(
        patch_embed.conv_w.len(),
        in_dim * out_dim,
        "patch_embed.conv_w length should be in_dim*out_dim after load-time transpose"
    );
    debug_assert_eq!(image.len(), 3 * target_h * target_w);

    let h_stride = target_w;
    let c_stride = target_h * target_w;
    let n_patches = grid_w * grid_h;

    // Per-patch math: extract the patch_size² × 3 chunk into a
    // pre-allocated `patch` scratch, run the matmul-as-matvec
    // against `conv_w` viewed as `[in_dim × out_dim]` row-major,
    // then add the per-channel bias. The scratch can be reused
    // across patches — every patch writes the same `in_dim`
    // entries so no zero-init is needed between calls.
    //
    // out_row[oc] = Σ_in patch[in] · conv_w[in × out_dim + oc] + conv_b[oc]
    //
    // matmul_f32 *accumulates* (`c[i,j] += …`); pre-filling
    // out_row with the bias lands the bias-add for free, mirroring
    // the trick `cpu::conv2d`'s pointwise fast path uses.
    let conv_w = &patch_embed.conv_w;
    let conv_b = &patch_embed.conv_b;
    let compute_patch = |patch: &mut [f32], patch_idx: usize, out_row: &mut [f32]| {
        extract_patch(image, patch, patch_idx, grid_w, p, h_stride, c_stride);
        out_row.copy_from_slice(conv_b);
        crate::backend::cpu::matmul_f32(patch, conv_w, out_row, 1, out_dim, in_dim);
    };

    let mut out = vec![0f32; n_patches * out_dim];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // `for_each_init` gives each rayon worker its own
        // long-lived `patch` buffer, so the parallel path also
        // amortises the allocation across patches handled by the
        // same worker (instead of per-patch alloc).
        out.par_chunks_mut(out_dim).enumerate().for_each_init(
            || vec![0f32; in_dim],
            |patch, (patch_idx, out_row)| compute_patch(patch, patch_idx, out_row),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        // Hoisted out of the loop to avoid `n_patches` small
        // allocations in the serial path.
        let mut patch = vec![0f32; in_dim];
        for (patch_idx, out_row) in out.chunks_mut(out_dim).enumerate() {
            compute_patch(&mut patch, patch_idx, out_row);
        }
    }
    out
}

/// Extract one `[3·patch_size²]` patch (channel-major `(c, kh, kw)` layout)
/// from an NCHW `image` into `patch`. Shared by the CPU `patch_embed_compute`
/// and the GPU `im2col_patches` so the two patch extractions stay identical.
/// `h_stride = target_w`, `c_stride = target_h · target_w`.
#[inline]
pub(crate) fn extract_patch(
    image: &[f32],
    patch: &mut [f32],
    patch_idx: usize,
    grid_w: usize,
    p: usize,
    h_stride: usize,
    c_stride: usize,
) {
    let gr = patch_idx / grid_w;
    let gc = patch_idx % grid_w;
    for c in 0..3 {
        for kh in 0..p {
            for kw in 0..p {
                let pixel_r = gr * p + kh;
                let pixel_c = gc * p + kw;
                let in_idx = c * p * p + kh * p + kw;
                let img_idx = c * c_stride + pixel_r * h_stride + pixel_c;
                patch[in_idx] = image[img_idx];
            }
        }
    }
}

/// Bilinearly interpolate position embeddings from the trained
/// `(in_h, in_w)` square grid to a dynamic `(out_h, out_w)` grid.
/// Mirrors llama.cpp's `clip_graph::resize_position_embeddings`
/// (`tools/mtmd/clip.cpp:272-292`) using
/// `ggml_interpolate(GGML_SCALE_MODE_BILINEAR)` semantics: the
/// pixel-centre convention is `out_pixel(r, c)` samples source
/// fractional coords
/// `((r + 0.5) · in_h / out_h - 0.5, (c + 0.5) · in_w / out_w
/// - 0.5)`, clamped to `[0, in_dim - 1]`. Output layout is
/// `[out_h × out_w, n_embd]` row-major, matching the patch order
/// `encode_image` adds against.
pub(crate) fn interpolate_pos_embed_2d(
    pos: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    n_embd: usize,
) -> Vec<f32> {
    if in_h == 0
        || in_w == 0
        || out_h == 0
        || out_w == 0
        || n_embd == 0
        || pos.len() != in_h * in_w * n_embd
    {
        return Vec::new();
    }
    let mut out = vec![0f32; out_h * out_w * n_embd];
    let scale_y = in_h as f32 / out_h as f32;
    let scale_x = in_w as f32 / out_w as f32;
    let in_h_max = (in_h - 1) as f32;
    let in_w_max = (in_w - 1) as f32;
    for or in 0..out_h {
        for oc in 0..out_w {
            // Source fractional coords (centre-of-pixel convention).
            let sy = ((or as f32 + 0.5) * scale_y - 0.5).clamp(0.0, in_h_max);
            let sx = ((oc as f32 + 0.5) * scale_x - 0.5).clamp(0.0, in_w_max);
            let y0 = sy.floor() as usize;
            let x0 = sx.floor() as usize;
            let y1 = (y0 + 1).min(in_h - 1);
            let x1 = (x0 + 1).min(in_w - 1);
            let dy = sy - y0 as f32;
            let dx = sx - x0 as f32;
            let w00 = (1.0 - dy) * (1.0 - dx);
            let w01 = (1.0 - dy) * dx;
            let w10 = dy * (1.0 - dx);
            let w11 = dy * dx;
            let dst = (or * out_w + oc) * n_embd;
            let src00 = (y0 * in_w + x0) * n_embd;
            let src01 = (y0 * in_w + x1) * n_embd;
            let src10 = (y1 * in_w + x0) * n_embd;
            let src11 = (y1 * in_w + x1) * n_embd;
            for k in 0..n_embd {
                out[dst + k] = w00 * pos[src00 + k]
                    + w01 * pos[src01 + k]
                    + w10 * pos[src10 + k]
                    + w11 * pos[src11 + k];
            }
        }
    }
    out
}

/// `cfg.scale_factor`× pixel-shuffle (space-to-depth) over a
/// non-square patch grid.
/// Reshapes `[grid_h · grid_w, n_embd]` →
/// `[(grid_h/sf) · (grid_w/sf), n_embd · sf²]` by concatenating
/// each sf×sf patch group's channels in row-major (sr, sc) order.
/// Matches llama.cpp's `clip_graph::build_patch_merge_permute`
/// LFM2 projector convention; both grid dims must be divisible
/// by `cfg.scale_factor` (the preprocessor's
/// `align_size = patch_size · scale_factor` guarantees that).
pub(crate) fn pixel_shuffle(
    tokens: &[f32],
    cfg: &VisionEncoderConfig,
    grid_w: usize,
    grid_h: usize,
) -> Vec<f32> {
    let sf = cfg.scale_factor;
    debug_assert!(
        sf > 0 && grid_w.is_multiple_of(sf) && grid_h.is_multiple_of(sf),
        "patch grid {grid_w}×{grid_h} must be divisible by scale_factor ({sf})"
    );
    let new_w = grid_w / sf;
    let new_h = grid_h / sf;
    let n_embd = cfg.n_embd;
    let new_dim = n_embd * sf * sf;
    let n_out = new_w * new_h;
    let mut out = vec![0f32; n_out * new_dim];
    for or in 0..new_h {
        for oc in 0..new_w {
            let dst_off = (or * new_w + oc) * new_dim;
            // Walk the sf×sf source patches in (row-major) order
            // — must match clip.cpp's traversal.
            for sr in 0..sf {
                for sc in 0..sf {
                    let in_r = or * sf + sr;
                    let in_c = oc * sf + sc;
                    let src_off = (in_r * grid_w + in_c) * n_embd;
                    let chan_base = (sr * sf + sc) * n_embd;
                    out[dst_off + chan_base..dst_off + chan_base + n_embd]
                        .copy_from_slice(&tokens[src_off..src_off + n_embd]);
                }
            }
        }
    }
    out
}

/// Read `[f32; 3]` from a GGUF f32-array metadata key. Errors if
/// the key is missing or the array length isn't 3.
fn read_rgb_array(gguf: &Arc<GgufFile>, key: &str) -> Result<[f32; 3]> {
    let arr = gguf
        .get_f32_array(key)
        .with_context(|| format!("missing `{key}`"))?;
    anyhow::ensure!(
        arr.len() == 3,
        "`{key}` length {} != 3 (RGB triple expected)",
        arr.len()
    );
    Ok([arr[0], arr[1], arr[2]])
}

/// Load one ViT block's full weight set + cross-check shapes
/// against `n_embd` / `n_ff` / `n_head` from config.
fn load_vit_block(
    gguf: &Arc<GgufFile>,
    il: usize,
    cfg: &VisionEncoderConfig,
) -> Result<VitBlockWeights> {
    let pfx = format!("v.blk.{il}");
    let vec_f32 = |suffix: &str| load_vec_f32(gguf, &format!("{pfx}.{suffix}"));
    let weight_f32 = |suffix: &str| -> Result<MmapWeight> {
        let name = format!("{pfx}.{suffix}");
        MmapWeight::from_gguf(gguf, &name).with_context(|| format!("loading {name}"))
    };

    // Pre-attn layer norm.
    let ln1_w = vec_f32("ln1.weight")?;
    let ln1_b = vec_f32("ln1.bias")?;

    // Multi-head self-attention (no RoPE — ViT uses absolute
    // position embeddings added at the patch level).
    let q_w = weight_f32("attn_q.weight")?;
    let q_b = vec_f32("attn_q.bias")?;
    let k_w = weight_f32("attn_k.weight")?;
    let k_b = vec_f32("attn_k.bias")?;
    let v_w = weight_f32("attn_v.weight")?;
    let v_b = vec_f32("attn_v.bias")?;
    let o_w = weight_f32("attn_out.weight")?;
    let o_b = vec_f32("attn_out.bias")?;

    // Post-attn / pre-FFN layer norm.
    let ln2_w = vec_f32("ln2.weight")?;
    let ln2_b = vec_f32("ln2.bias")?;

    // FFN (tanh-approx GELU activation between up and down — the
    // mmproj GGUF carries `clip.use_gelu = true`, which in
    // llama.cpp's `clip.cpp` selects `FFN_GELU` → `ggml_gelu`,
    // i.e. the tanh approximation. Matching that exactly matters:
    // the erf-form GELU drifts by ~1e-3 per call which compounds
    // over 12 layers + the projector and silently degrades
    // semantic-image-token quality.
    let ffn_up_w = weight_f32("ffn_up.weight")?;
    let ffn_up_b = vec_f32("ffn_up.bias")?;
    let ffn_down_w = weight_f32("ffn_down.weight")?;
    let ffn_down_b = vec_f32("ffn_down.bias")?;

    // Shape sanity: every linear is [n_embd × n_embd] except FFN
    // which is [n_ff × n_embd] (up) / [n_embd × n_ff] (down).
    // Loud assertion at load time beats a corrupted forward.
    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;
    for (name, w) in [
        ("attn_q.weight", &q_w),
        ("attn_k.weight", &k_w),
        ("attn_v.weight", &v_w),
        ("attn_out.weight", &o_w),
    ] {
        anyhow::ensure!(
            w.rows == n_embd && w.cols == n_embd,
            "block {il} {name} shape ({}, {}) != ({n_embd}, {n_embd})",
            w.rows,
            w.cols,
        );
    }
    anyhow::ensure!(
        ffn_up_w.rows == n_ff && ffn_up_w.cols == n_embd,
        "block {il} ffn_up.weight shape ({}, {}) != ({n_ff}, {n_embd})",
        ffn_up_w.rows,
        ffn_up_w.cols,
    );
    anyhow::ensure!(
        ffn_down_w.rows == n_embd && ffn_down_w.cols == n_ff,
        "block {il} ffn_down.weight shape ({}, {}) != ({n_embd}, {n_ff})",
        ffn_down_w.rows,
        ffn_down_w.cols,
    );

    // Bias / norm length checks.
    for (name, v) in [
        ("ln1.weight", &ln1_w),
        ("ln1.bias", &ln1_b),
        ("attn_q.bias", &q_b),
        ("attn_k.bias", &k_b),
        ("attn_v.bias", &v_b),
        ("attn_out.bias", &o_b),
        ("ln2.weight", &ln2_w),
        ("ln2.bias", &ln2_b),
        ("ffn_down.bias", &ffn_down_b),
    ] {
        anyhow::ensure!(
            v.len() == n_embd,
            "block {il} {name} len ({}) != n_embd ({n_embd})",
            v.len(),
        );
    }
    anyhow::ensure!(
        ffn_up_b.len() == n_ff,
        "block {il} ffn_up.bias len ({}) != n_ff ({n_ff})",
        ffn_up_b.len(),
    );
    // `n_head > 0` first to avoid the modulo's div-by-zero on a
    // corrupt mmproj reporting `n_head = 0`.
    anyhow::ensure!(
        cfg.n_head > 0 && n_embd.is_multiple_of(cfg.n_head),
        "n_embd ({n_embd}) not divisible by n_head ({})",
        cfg.n_head,
    );

    Ok(VitBlockWeights {
        ln1_w,
        ln1_b,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        o_w,
        o_b,
        ln2_w,
        ln2_b,
        ffn_up_w,
        ffn_up_b,
        ffn_down_w,
        ffn_down_b,
    })
}

/// `MmapWeight::from_gguf` with a `with_context` wrapping the
/// tensor name into the error chain.
fn wt_f32(gguf: &Arc<GgufFile>, name: &str) -> Result<MmapWeight> {
    MmapWeight::from_gguf(gguf, name).with_context(|| format!("loading {name}"))
}

/// Read a 1D `Vec<f32>` from a GGUF tensor by name. Validates the
/// rank — a hypothetical schema drift turning a vector tensor into
/// a 2D matrix would otherwise pass through unnoticed and trip the
/// downstream length checks at a less actionable site.
fn load_vec_f32(gguf: &Arc<GgufFile>, name: &str) -> Result<Vec<f32>> {
    let tensor = gguf
        .get_tensor(name)
        .with_context(|| format!("loading {name}"))?;
    anyhow::ensure!(
        tensor.shape().len() == 1,
        "tensor {name} must be 1D, got rank {}",
        tensor.shape().len()
    );
    Ok(tensor.to_f32_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_cfg(grid: usize, n_embd: usize, scale_factor: usize) -> VisionEncoderConfig {
        let patch_size = 16;
        let image_size = grid * patch_size;
        let pixels_per_token = (patch_size * scale_factor).pow(2);
        VisionEncoderConfig {
            n_layer: 1,
            n_embd,
            n_ff: n_embd * 4,
            n_head: 1,
            eps: 1e-6,
            image_size,
            patch_size,
            n_trained_patches: grid * grid,
            projection_dim: n_embd,
            scale_factor,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            image_min_pixels: 64 * pixels_per_token,
            image_max_pixels: 256 * pixels_per_token,
        }
    }

    /// `pixel_shuffle` reshapes a 4×4 grid of single-channel
    /// "tokens" into a 2×2 grid of 4-channel tokens, with the
    /// per-token channels concatenated in row-major order over
    /// the source 2×2 group. Verifying via a tagged input
    /// (`patch_idx + 0.1·channel`) catches any mis-orderings
    /// of the inner 2×2 traversal — exactly the bug-class that
    /// would silently produce visually-OK but semantically-
    /// wrong projector input.
    #[test]
    fn pixel_shuffle_2x2_reshapes_grid_in_row_major_order() {
        // 4×4 patch grid, n_embd=1, scale_factor=2.
        // Output should be a 2×2 grid with n_embd=4.
        let cfg = synth_cfg(4, 1, 2);
        let n_in = cfg.n_trained_patches * cfg.n_embd; // 16 * 1 = 16
        let mut tokens = vec![0f32; n_in];
        for (i, t) in tokens.iter_mut().enumerate() {
            *t = i as f32; // patch i has value i
        }
        let pooled = pixel_shuffle(&tokens, &cfg, 4, 4);
        // 4 output tokens, each with 4 channels.
        assert_eq!(pooled.len(), 4 * 4);

        // Source patch indices grouped by 2×2 in the 4×4 grid:
        //   [0]=(0,0) [1]=(0,1) | [2]=(0,2) [3]=(0,3)
        //   [4]=(1,0) [5]=(1,1) | [6]=(1,2) [7]=(1,3)
        //   ─────────────────────┼─────────────────────
        //   [8]=(2,0) [9]=(2,1) | [10]=(2,2) [11]=(2,3)
        //   [12]=(3,0)[13]=(3,1)| [14]=(3,2)[15]=(3,3)
        // Output token (0,0) = patches [0,1,4,5] in (sr*sf+sc) order.
        // Output token (0,1) = patches [2,3,6,7].
        // Output token (1,0) = patches [8,9,12,13].
        // Output token (1,1) = patches [10,11,14,15].
        assert_eq!(&pooled[0..4], &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(&pooled[4..8], &[2.0, 3.0, 6.0, 7.0]);
        assert_eq!(&pooled[8..12], &[8.0, 9.0, 12.0, 13.0]);
        assert_eq!(&pooled[12..16], &[10.0, 11.0, 14.0, 15.0]);
    }

    /// Multi-channel pixel-shuffle: each source patch carries
    /// `n_embd` values that survive in-order in the output.
    #[test]
    fn pixel_shuffle_2x2_preserves_per_patch_channels() {
        // 2×2 patch grid, n_embd=3, scale_factor=2 → single
        // output token with 12 channels.
        let cfg = synth_cfg(2, 3, 2);
        // Patch i has channels [i*10, i*10+1, i*10+2].
        let mut tokens = vec![0f32; 4 * 3];
        for i in 0..4 {
            for c in 0..3 {
                tokens[i * 3 + c] = (i * 10 + c) as f32;
            }
        }
        let pooled = pixel_shuffle(&tokens, &cfg, 2, 2);
        assert_eq!(pooled.len(), 12);
        // Output token gets patches in (sr=0,sc=0), (0,1), (1,0),
        // (1,1) order — i.e. patches 0, 1, 2, 3 in the 2×2 grid.
        // Each contributes its 3 channels in order.
        assert_eq!(
            pooled,
            vec![
                0.0, 1.0, 2.0, // patch 0
                10.0, 11.0, 12.0, // patch 1
                20.0, 21.0, 22.0, // patch 2
                30.0, 31.0, 32.0, // patch 3
            ]
        );
    }

    /// Loader transpose check: simulate the GGUF source layout
    /// `linear(kw, kh, c, oc) = kw + p·kh + p²·c + p²·3·oc` with a
    /// tagged value scheme, then exercise the same index-mapping
    /// loop the real loader uses (lines 290–300 of this file) and
    /// verify the resulting `[in_dim × n_embd]` row-major buffer
    /// reads back the original tagged values at every `(oc, c, kh,
    /// kw)`. Catches the bug class where someone swaps
    /// `oc * in_dim + in_idx` ↔ `in_idx * out_dim + oc` (the exact
    /// transpose mistake the patch-embed parity fix corrected).
    #[test]
    fn patch_embed_loader_transpose_round_trip() {
        let p = 2usize; // patch_size — small for clarity
        let in_dim = 3 * p * p; // 12
        let out_dim = 5; // arbitrary; deliberately ≠ in_dim
        // Synthetic source: tag every element so a transposed read
        // produces visibly wrong values.
        let mut raw = vec![0f32; p * p * 3 * out_dim];
        let tag = |oc: usize, c: usize, kh: usize, kw: usize| -> f32 {
            (oc * 1000 + c * 100 + kh * 10 + kw) as f32
        };
        for oc in 0..out_dim {
            for c in 0..3 {
                for kh in 0..p {
                    for kw in 0..p {
                        let src = kw + p * kh + p * p * c + p * p * 3 * oc;
                        raw[src] = tag(oc, c, kh, kw);
                    }
                }
            }
        }
        // Apply the loader's transpose into `[in_dim × out_dim]`.
        let mut conv_w = vec![0f32; in_dim * out_dim];
        for oc in 0..out_dim {
            for c in 0..3 {
                for kh in 0..p {
                    for kw in 0..p {
                        let src = kw + p * kh + p * p * c + p * p * 3 * oc;
                        let in_idx = c * p * p + kh * p + kw;
                        conv_w[in_idx * out_dim + oc] = raw[src];
                    }
                }
            }
        }
        // Verify every (oc, c, kh, kw) round-trips through the
        // `[in_dim × out_dim]` layout.
        for oc in 0..out_dim {
            for c in 0..3 {
                for kh in 0..p {
                    for kw in 0..p {
                        let in_idx = c * p * p + kh * p + kw;
                        let got = conv_w[in_idx * out_dim + oc];
                        let expected = tag(oc, c, kh, kw);
                        assert_eq!(
                            got, expected,
                            "conv_w[in_idx={in_idx}, oc={oc}] = {got} != {expected} \
                             (oc={oc}, c={c}, kh={kh}, kw={kw})"
                        );
                    }
                }
            }
        }
    }

    /// End-to-end patch-embed math: build a known kernel + image,
    /// run `patch_embed_compute`, and assert the output matches an
    /// explicit per-patch dot product. This is the test that would
    /// have caught the matmul-transpose bug — it doesn't need a
    /// GGUF, it doesn't need llama.cpp, just direct matmul math.
    #[test]
    fn patch_embed_compute_matches_explicit_dot_product() {
        // 2×2 patch grid → 4 patches, patch_size=2, n_embd=3.
        let grid = 2usize;
        let cfg = synth_cfg(grid, /* n_embd */ 3, /* scale_factor */ 1);
        let p = cfg.patch_size;
        let in_dim = 3 * p * p; // 12
        let out_dim = cfg.n_embd; // 3
        let n_patches = cfg.n_trained_patches; // 4

        // Synthetic kernel in the [in_dim × out_dim] row-major
        // layout the loader produces. Pick small values so the
        // explicit reference dot product (~12 terms summed in f32)
        // stays inside f32's precision floor — but vary in both
        // axes so a transposed read produces visibly wrong dot
        // products.
        let mut conv_w = vec![0f32; in_dim * out_dim];
        for in_idx in 0..in_dim {
            for oc in 0..out_dim {
                conv_w[in_idx * out_dim + oc] = (in_idx as f32) * 0.01 + (oc as f32) * 0.1;
            }
        }
        // Bias: per-channel offsets so a swapped bias-add is
        // visible too.
        let conv_b = vec![0.5, 1.0, 1.5];
        let patch_embed = PatchEmbedWeights {
            conv_w: conv_w.clone(),
            conv_b: conv_b.clone(),
        };

        // Synthetic image: CHW, image[c, y, x] uses small values so
        // the per-patch dot stays well below f32 mantissa overflow.
        let n_pix = 3 * cfg.image_size * cfg.image_size;
        let mut image = vec![0f32; n_pix];
        for c in 0..3 {
            for y in 0..cfg.image_size {
                for x in 0..cfg.image_size {
                    image[c * cfg.image_size * cfg.image_size + y * cfg.image_size + x] =
                        (c as f32) * 0.3 + (y as f32) * 0.05 + (x as f32) * 0.02;
                }
            }
        }

        let actual = patch_embed_compute(&image, &patch_embed, &cfg, grid, grid);
        assert_eq!(actual.len(), n_patches * out_dim);

        // Compute the expected output explicitly per-patch,
        // per-output-channel.
        for gr in 0..grid {
            for gc in 0..grid {
                let patch_idx = gr * grid + gc;
                for oc in 0..out_dim {
                    let mut expected = conv_b[oc];
                    for c in 0..3 {
                        for kh in 0..p {
                            for kw in 0..p {
                                let in_idx = c * p * p + kh * p + kw;
                                let pixel_r = gr * p + kh;
                                let pixel_c = gc * p + kw;
                                let img_v = image[c * cfg.image_size * cfg.image_size
                                    + pixel_r * cfg.image_size
                                    + pixel_c];
                                expected += img_v * conv_w[in_idx * out_dim + oc];
                            }
                        }
                    }
                    let got = actual[patch_idx * out_dim + oc];
                    assert!(
                        (got - expected).abs() < 1e-5,
                        "patch_idx={patch_idx} oc={oc}: got {got}, expected {expected}"
                    );
                }
            }
        }
    }

    /// Identity check: interpolate from a 4×4 grid to itself
    /// reproduces the input.
    #[test]
    fn interpolate_pos_embed_2d_identity() {
        let n_embd = 3;
        let mut input = vec![0f32; 4 * 4 * n_embd];
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as f32;
        }
        let out = interpolate_pos_embed_2d(&input, 4, 4, 4, 4, n_embd);
        for (i, (a, b)) in out.iter().zip(input.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "diff at {i}: {a} vs {b}");
        }
    }

    /// Non-square interpolation: 2×2 grid → 4×6 grid. The four
    /// corners of the trained grid must appear unchanged at the
    /// four corners of the output (centre-of-pixel convention
    /// clamps the fractional source coords to `[0, in_dim - 1]`).
    /// Catches axis swaps (out_w / out_h transposition) and
    /// channel shuffling.
    #[test]
    fn interpolate_pos_embed_2d_corners_match() {
        // Single channel, 2×2 grid with corner values 1/2/3/4.
        let n_embd = 1;
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let out = interpolate_pos_embed_2d(&input, 2, 2, 4, 6, n_embd);
        // out[0, 0] = top-left corner of input (1.0).
        assert!((out[0] - 1.0).abs() < 1e-5);
        // out[0, 5] = top-right corner of input (2.0).
        assert!((out[5] - 2.0).abs() < 1e-5);
        // out[3, 0] = bottom-left of input (3.0).
        assert!((out[3 * 6] - 3.0).abs() < 1e-5);
        // out[3, 5] = bottom-right (4.0).
        assert!((out[3 * 6 + 5] - 4.0).abs() < 1e-5);
    }

    /// Deterministic pseudo-random f32 in `[-1, 1)`.
    fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((st >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    /// A `rows x cols` Q8_0 `MmapWeight` quantized from deterministic random f32.
    fn q8_weight(rows: usize, cols: usize, seed: u64) -> MmapWeight {
        let mut bytes = Vec::new();
        for row in lcg_vec(seed, rows * cols).chunks(cols) {
            let mut scales = vec![0.0f32; row.len() / 32];
            let mut quants = vec![0i8; row.len()];
            crate::backend::cpu::quantize_f32_to_q8_0_into(row, &mut scales, &mut quants);
            for (b, d) in scales.iter().enumerate() {
                bytes.extend_from_slice(&half::f16::from_f32(*d).to_bits().to_le_bytes());
                bytes.extend_from_slice(bytemuck::cast_slice::<i8, u8>(
                    &quants[b * 32..(b + 1) * 32],
                ));
            }
        }
        MmapWeight::from_owned_bytes(bytes, crate::tensor::DType::Q8_0, rows, cols)
    }

    fn enabled_q8act() -> Q8Act {
        Q8Act {
            enabled: crate::backend::cpu::int8_gemm_available(),
            dim: 0,
            n: 0,
            scales: vec![0.0; 4096 * 8],
            quants: vec![0; 4096 * 8 * 32],
            out_t: vec![0.0; 4096 * 256],
        }
    }

    /// The int8 path must agree with the f32 path to within activation-quantization error, including
    /// odd row and token counts (the GEMM's remainder paths), and must write `[token][row]`.
    #[test]
    fn int8_linear_matches_f32_path() {
        if !crate::backend::cpu::int8_gemm_available() {
            return;
        }
        for &(rows, cols, n) in &[(72usize, 96usize, 40usize), (71, 64, 37), (96, 128, 16)] {
            let w = q8_weight(rows, cols, 7);
            let x = lcg_vec(11, n * cols);
            let mut want = vec![0.0f32; n * rows];
            w.batched_matmul_with_scratch(&x, &mut want, n, None);

            let mut q8 = enabled_q8act();
            q8.quantize(&x, cols, n);
            let mut got = vec![0.0f32; n * rows];
            let mut dq = vec![0.0f32; cols];
            vit_linear(&w, &x, &mut got, n, &mut q8, &mut dq);
            assert!(
                q8.matmul(&w, &mut vec![0.0; n * rows]),
                "int8 path declined a Q8_0 weight at {rows}x{cols}x{n}"
            );

            let num: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|b| b * b).sum();
            let rel = (num / den).sqrt();
            assert!(
                rel < 0.01,
                "int8 vs f32 relative RMS error {rel} at {rows}x{cols}x{n}"
            );
            // Layout: a transposed result would be far off in RMS, but pin one element exactly.
            let (t, r) = (n - 1, rows - 1);
            assert!(
                (got[t * rows + r] - want[t * rows + r]).abs()
                    < 0.05 * want[t * rows + r].abs().max(1.0)
            );
        }
    }

    /// The smmla-repacked kernel and the standard Q8_0 kernel compute the same integer dot per block, so
    /// they differ only in float summation order. Runs where the host has i8mm and the weight repacks.
    #[test]
    fn smmla_repacked_matches_standard_gemm() {
        let mut ran = false;
        for &(rows, cols, n) in &[
            (64usize, 96usize, 40usize),
            (72, 128, 37),
            (24, 64, 4),
            (16, 32, 1),
        ] {
            let w = q8_weight(rows, cols, 3);
            if w.smmla_repacked().is_none() {
                continue;
            }
            ran = true;
            let x = lcg_vec(5, n * cols);
            let mut q8 = enabled_q8act();
            q8.quantize(&x, cols, n);
            let mut got = vec![0.0f32; n * rows];
            assert!(q8.matmul(&w, &mut got));

            let mut want_t = vec![0.0f32; rows * n];
            assert!(crate::backend::cpu::gemm_preq_dispatch(
                crate::tensor::DType::Q8_0,
                w.data(),
                &q8.scales[..n * (cols / 32)],
                &q8.quants[..n * cols],
                &mut want_t,
                rows,
                n,
                cols,
            ));
            for t in 0..n {
                for r in 0..rows {
                    let (g, e) = (got[t * rows + r], want_t[r * n + t]);
                    assert!(
                        (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                        "smmla {g} vs standard {e} at token {t} row {r} ({rows}x{cols}x{n})"
                    );
                }
            }
        }
        if !ran {
            // A host without the repack skips quietly, unless the CI leg that is supposed to have
            // i8mm says so: then a never-run comparison is a failure, not a pass (the message names
            // i8mm, the usual cause; a BLAS build also returns no repack).
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            crate::backend::simd::require_simd_or_skip("i8mm", false);
            eprintln!("smmla_repacked_matches_standard_gemm: no i8mm on this host, skipped");
        }
    }

    /// Large score magnitudes (a softmax that is nearly one-hot) must not overflow or lose the winner:
    /// the exact softmax subtracts each row's maximum before exponentiating.
    #[test]
    fn vit_attention_is_stable_for_large_scores() {
        let (n_tokens, n_head, head_dim) = (40usize, 2usize, 32usize);
        let n_embd = n_head * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = lcg_vec(5, n_tokens * n_embd)
            .iter()
            .map(|x| x * 60.0)
            .collect();
        let k = lcg_vec(6, n_tokens * n_embd);
        let v = lcg_vec(7, n_tokens * n_embd);
        let mut got = vec![f32::NAN; n_tokens * n_embd];
        let mut heads = vec![0.0f32; attn_heads_len(n_tokens, n_head, head_dim)];
        let mut kt = vec![0.0f32; attn_kt_len(n_tokens, n_head, head_dim)];
        vit_attention(
            &q, &k, &v, &mut got, &mut heads, &mut kt, n_tokens, n_head, head_dim, scale,
        );
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite attention output"
        );
        // Each output row is a convex combination of value rows, so it stays within their range.
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        assert!(
            got.iter().all(|&x| x >= lo - 1e-4 && x <= hi + 1e-4),
            "outside the value range"
        );
    }

    /// A weight the int8 path cannot take (here F32) must fall back to the f32 kernel unchanged.
    #[test]
    fn int8_linear_falls_back_for_non_q8_weights() {
        let (rows, cols, n) = (24usize, 64usize, 20usize);
        let w = MmapWeight::from_owned_f32(lcg_vec(3, rows * cols), rows, cols);
        let x = lcg_vec(5, n * cols);
        let mut want = vec![0.0f32; n * rows];
        w.batched_matmul_with_scratch(&x, &mut want, n, None);

        let mut q8 = enabled_q8act();
        q8.quantize(&x, cols, n);
        let mut got = vec![0.0f32; n * rows];
        let mut dq = vec![0.0f32; cols];
        vit_linear(&w, &x, &mut got, n, &mut q8, &mut dq);
        assert_eq!(got, want, "fallback must be the f32 kernel's exact result");
    }

    /// The flash attention path must match the per-query softmax loop it replaced, including a
    /// token count that is not a multiple of the 32-query chunk.
    #[test]
    fn vit_attention_matches_naive() {
        for &(n_tokens, n_head, head_dim) in &[
            (70usize, 3usize, 32usize),
            (96, 4, 16),
            (33, 2, 64),
            (40, 4, 8),
            (32, 2, 16),
            // Token counts that are not a multiple of 4 or 8, a single token, a head_dim of 24, and
            // enough queries for a ragged last 8-query tile.
            (37, 3, 16),
            (1, 1, 8),
            (9, 2, 24),
            // A head_dim that is not a multiple of 8 takes the generic flash-attention path on every
            // target (the NEON ViT kernel needs a multiple of 8), so that arm runs on aarch64 too.
            (21, 2, 12),
            (101, 2, 64),
        ] {
            let n_embd = n_head * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (q, k, v) = (
                lcg_vec(1, n_tokens * n_embd),
                lcg_vec(2, n_tokens * n_embd),
                lcg_vec(3, n_tokens * n_embd),
            );
            let mut want = vec![0.0f32; n_tokens * n_embd];
            for t in 0..n_tokens {
                for h in 0..n_head {
                    let qh = &q[t * n_embd + h * head_dim..][..head_dim];
                    let mut scores: Vec<f32> = (0..n_tokens)
                        .map(|j| {
                            crate::backend::cpu::dot_f32(
                                qh,
                                &k[j * n_embd + h * head_dim..][..head_dim],
                            ) * scale
                        })
                        .collect();
                    crate::backend::cpu::softmax_inplace(&mut scores);
                    for (j, &sc) in scores.iter().enumerate() {
                        let vh = &v[j * n_embd + h * head_dim..][..head_dim];
                        for (o, vv) in want[t * n_embd + h * head_dim..][..head_dim]
                            .iter_mut()
                            .zip(vh)
                        {
                            *o += sc * vv;
                        }
                    }
                }
            }

            let mut got = vec![f32::NAN; n_tokens * n_embd];
            let mut heads = vec![0.0f32; attn_heads_len(n_tokens, n_head, head_dim)];
            let mut kt = vec![0.0f32; attn_kt_len(n_tokens, n_head, head_dim)];
            vit_attention(
                &q, &k, &v, &mut got, &mut heads, &mut kt, n_tokens, n_head, head_dim, scale,
            );
            let max_err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-4,
                "attention max abs error {max_err} at n={n_tokens} heads={n_head} hd={head_dim}"
            );
        }
    }

    /// Non-square pixel-shuffle: 4×6 grid → 2×3 token grid with
    /// scale_factor=2. Verifies the non-square traversal matches
    /// the row-major (sr, sc) order the LFM2 projector expects.
    #[test]
    fn pixel_shuffle_non_square() {
        let cfg = synth_cfg(
            /* placeholder */ 4, /* n_embd */ 1, /* sf */ 2,
        );
        // 4 rows × 6 cols of single-channel tokens, value = patch_idx.
        let grid_w = 6;
        let grid_h = 4;
        let n_in = grid_w * grid_h;
        let mut tokens = vec![0f32; n_in];
        for (i, t) in tokens.iter_mut().enumerate() {
            *t = i as f32;
        }
        let pooled = pixel_shuffle(&tokens, &cfg, grid_w, grid_h);
        // 2 × 3 = 6 output tokens, each with 4 channels.
        assert_eq!(pooled.len(), 6 * 4);
        // Output (0, 0) covers source patches at rows 0..2, cols 0..2:
        //   row 0: indices [0, 1] (cols 0, 1)
        //   row 1: indices [6, 7]
        // (sr*sf+sc) order: (0,0)=0, (0,1)=1, (1,0)=6, (1,1)=7.
        assert_eq!(&pooled[0..4], &[0.0, 1.0, 6.0, 7.0]);
        // Output (0, 1) covers cols 2..4: (0,2)=2, (0,3)=3, (1,2)=8, (1,3)=9.
        assert_eq!(&pooled[4..8], &[2.0, 3.0, 8.0, 9.0]);
        // Output (0, 2): cols 4..6: 4, 5, 10, 11.
        assert_eq!(&pooled[8..12], &[4.0, 5.0, 10.0, 11.0]);
        // Output (1, 0): rows 2..4 cols 0..2: 12, 13, 18, 19.
        assert_eq!(&pooled[12..16], &[12.0, 13.0, 18.0, 19.0]);
    }
}

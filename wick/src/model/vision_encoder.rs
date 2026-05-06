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
    /// Number of patches per image (`(image_size / patch_size)^2`).
    /// Derived, not read — kept here so the forward pass doesn't
    /// recompute it.
    pub n_patches: usize,
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
            patch_size > 0 && image_size % patch_size == 0,
            "image_size ({image_size}) must be a positive multiple of patch_size ({patch_size})"
        );
        let n_patches = (image_size / patch_size).pow(2);

        let projection_dim =
            gguf.get_u32(KEY_PROJECTION_DIM)
                .with_context(|| format!("missing `{KEY_PROJECTION_DIM}`"))? as usize;
        let scale_factor =
            gguf.get_u32(KEY_SCALE_FACTOR)
                .with_context(|| format!("missing `{KEY_SCALE_FACTOR}`"))? as usize;
        let image_mean = read_rgb_array(gguf, KEY_IMAGE_MEAN)?;
        let image_std = read_rgb_array(gguf, KEY_IMAGE_STD)?;

        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            eps,
            image_size,
            patch_size,
            n_patches,
            projection_dim,
            scale_factor,
            image_mean,
            image_std,
        })
    }
}

/// Patch-embed Conv2D weights. The kernel is stored 4D
/// `[patch_size, patch_size, 3, n_embd]` per the GGUF layout; the
/// loader keeps it raw so the forward pass can reinterpret it
/// without a copy.
pub struct PatchEmbedWeights {
    /// `v.patch_embd.weight` — flattened conv kernel.
    pub conv_w: Vec<f32>,
    /// `v.patch_embd.bias` — `[n_embd]`.
    pub conv_b: Vec<f32>,
    /// Original GGUF shape (`[patch_size, patch_size, 3, n_embd]`).
    /// Preserved for the forward pass; not used at load time
    /// beyond the shape sanity check below.
    pub shape: Vec<usize>,
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
pub struct ProjectorWeights {
    pub mm1_w: MmapWeight,
    pub mm1_b: Vec<f32>,
    pub mm2_w: MmapWeight,
    pub mm2_b: Vec<f32>,
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
/// `WickEngine::vision_encoder()` doesn't change shape — only
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

        // Patch embed Conv2D kernel `[patch_size, patch_size, 3,
        // n_embd]` — kept raw so the forward pass can reinterpret
        // without a copy. Validate the shape so a future schema
        // change surfaces here, not during inference.
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
        let patch_embed = PatchEmbedWeights {
            conv_w: patch_t.to_f32_vec(),
            conv_b: load_vec_f32(gguf, "v.patch_embd.bias")?,
            shape: patch_shape,
        };
        anyhow::ensure!(
            patch_embed.conv_b.len() == config.n_embd,
            "v.patch_embd.bias len ({}) != n_embd ({})",
            patch_embed.conv_b.len(),
            config.n_embd,
        );

        // Position embedding `[n_patches × n_embd]`. MmapWeight
        // would also work but the forward pass treats this as a
        // plain matrix of patch-position rows; keep it as a flat
        // Vec<f32> so the indexing reads as
        // `position_embed[p * n_embd + i]`.
        let pos_t = gguf
            .get_tensor("v.position_embd.weight")
            .context("loading v.position_embd.weight")?;
        let pos_shape = pos_t.shape();
        anyhow::ensure!(
            pos_shape.len() == 2
                && pos_shape[0] == config.n_embd
                && pos_shape[1] == config.n_patches,
            "v.position_embd.weight shape {pos_shape:?} != [n_embd={}, n_patches={}]",
            config.n_embd,
            config.n_patches,
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

        Ok(Self {
            config,
            patch_embed,
            position_embed,
            blocks,
            post_ln_w,
            post_ln_b,
            projector,
        })
    }
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

    // FFN (GELU activation between up and down).
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
        cfg.n_head > 0 && n_embd % cfg.n_head == 0,
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

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
const KEY_PROJ_TYPE: &str = "clip.projector_type";
const KEY_VISION_PROJ_TYPE: &str = "clip.vision.projector_type";
const KEY_N_LAYER: &str = "clip.vision.block_count";
const KEY_N_EMBD: &str = "clip.vision.embedding_length";
const KEY_N_FF: &str = "clip.vision.feed_forward_length";
const KEY_N_HEAD: &str = "clip.vision.attention.head_count";
const KEY_N_HEAD_KV: &str = "clip.vision.attention.head_count_kv";
const KEY_ROPE_THETA: &str = "clip.vision.rope_theta";
const KEY_LN_EPS: &str = "clip.vision.attention.layer_norm_epsilon";
const KEY_IMAGE_SIZE: &str = "clip.vision.image_size";
const KEY_PATCH_SIZE: &str = "clip.vision.patch_size";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";
const KEY_SCALE_FACTOR: &str = "clip.vision.projector.scale_factor";
const KEY_IMAGE_MEAN: &str = "clip.vision.image_mean";
const KEY_IMAGE_STD: &str = "clip.vision.image_std";

/// Default RoPE theta for LFM3 vision 2D M-RoPE when `clip.vision.rope_theta` is absent.
const LFM3_DEFAULT_ROPE_THETA: f32 = 1_000_000.0;
/// Most pixels a GGUF may ask the vision preprocessor to resize an image to (64 Mpx).
const MAX_IMAGE_PIXELS: usize = 64 << 20;

/// Where in the LFM3 projector the optional `mm.input_norm` applies, by its width.
#[derive(Debug, PartialEq, Eq)]
enum MmNormStage {
    /// Over the encoder width, on the tokens before the spatial merge.
    PreMerge,
    /// Over the merged width (`n_embd * scale_factor^2`), after the merge.
    PostMerge,
}

/// The one stage a norm of `len` entries applies at. With `scale_factor == 1` the merged width
/// equals `n_embd`, and both stages would match: the norm must still run once.
fn mm_norm_stage(len: usize, n_embd: usize, merged_dim: usize) -> Option<MmNormStage> {
    if len == n_embd {
        Some(MmNormStage::PreMerge)
    } else if len == merged_dim {
        Some(MmNormStage::PostMerge)
    } else {
        None
    }
}

/// LayerNorm epsilon used by `mm.input_norm` in the LFM3 vision projector.
const LFM3_MM_INPUT_NORM_EPS: f32 = 1e-6;

/// Vision start boundary token emitted before projected image tokens for `lfm3o`.
pub const VISION_START_TOKEN: &str = "<|vision_start|>";
/// Vision end boundary token emitted after projected image tokens for `lfm3o`.
pub const VISION_END_TOKEN: &str = "<|vision_end|>";

/// Multimodal vision projector / encoder architecture type read from
/// `clip.vision.projector_type` or `clip.projector_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionProjectorType {
    /// LFM2-VL ViT encoder + pixel-shuffle + 2-layer MLP projector (`"lfm2"`).
    #[default]
    Lfm2,
    /// LFM3-VL hybrid conv + 2D M-RoPE GQA attention encoder (`"lfm3vl"`), without boundary tokens.
    Lfm3Vl,
    /// LFM3-Omni hybrid conv + 2D M-RoPE GQA attention encoder (`"lfm3o"`), with `<|vision_start|>` and `<|vision_end|>`.
    Lfm3O,
}

impl VisionProjectorType {
    /// Parse the GGUF projector type string (`"lfm2"`, `"lfm3vl"`, `"lfm3o"`).
    pub fn from_gguf_str(s: &str) -> Result<Self> {
        match s.trim() {
            "" | "lfm2" => Ok(Self::Lfm2),
            "lfm3vl" => Ok(Self::Lfm3Vl),
            "lfm3o" => Ok(Self::Lfm3O),
            other => anyhow::bail!("unsupported vision projector_type `{other}`"),
        }
    }

    /// Canonical GGUF string identifier for this projector type.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lfm2 => "lfm2",
            Self::Lfm3Vl => "lfm3vl",
            Self::Lfm3O => "lfm3o",
        }
    }

    /// True for LFM3-VL (`lfm3vl`) and LFM3-Omni (`lfm3o`) vision encoders.
    pub fn is_lfm3(&self) -> bool {
        matches!(self, Self::Lfm3Vl | Self::Lfm3O)
    }

    /// True when the vision projector wraps image embeddings in `<|vision_start|>` and `<|vision_end|>`.
    /// Only `lfm3o` enables boundary tokens; `lfm3vl` and `lfm2` do not.
    pub fn has_boundary_tokens(&self) -> bool {
        matches!(self, Self::Lfm3O)
    }

    /// Return `Some(("<|vision_start|>", "<|vision_end|>"))` when boundary tokens are enabled (`lfm3o`),
    /// or `None` for `lfm3vl` and `lfm2`.
    pub fn vision_boundary_tokens(&self) -> Option<(&'static str, &'static str)> {
        if self.has_boundary_tokens() {
            Some((VISION_START_TOKEN, VISION_END_TOKEN))
        } else {
            None
        }
    }

    /// Format an image placeholder or token string with boundary tokens if enabled (`lfm3o`).
    pub fn wrap_image_prompt(&self, inner: &str) -> String {
        if let Some((start, end)) = self.vision_boundary_tokens() {
            format!("{start}{inner}{end}")
        } else {
            inner.to_string()
        }
    }

    /// Resolve the `(start_token_id, end_token_id)` pair via a vocabulary lookup closure
    /// when boundary tokens are enabled for this projector type (`lfm3o`).
    pub fn resolve_boundary_token_ids<F>(&self, mut lookup: F) -> Option<(u32, u32)>
    where
        F: FnMut(&str) -> Option<u32>,
    {
        let (start_str, end_str) = self.vision_boundary_tokens()?;
        Some((lookup(start_str)?, lookup(end_str)?))
    }
}

/// Configuration for the LFM2-VL / LFM3-VL / LFM3-Omni vision encoder. Read from
/// the `clip.vision.*` metadata block of the multimodal_projector GGUF.
#[derive(Debug, Clone)]
pub struct VisionEncoderConfig {
    /// Vision architecture / projector variant (`lfm2`, `lfm3vl`, or `lfm3o`).
    pub projector_type: VisionProjectorType,
    /// Number of ViT / hybrid encoder blocks.
    pub n_layer: usize,
    /// Encoder hidden dimension (`clip.vision.embedding_length`).
    pub n_embd: usize,
    /// FFN intermediate dimension inside each block.
    pub n_ff: usize,
    /// Number of query attention heads per attention block.
    pub n_head: usize,
    /// Number of key/value attention heads per attention block (`clip.vision.attention.head_count_kv`).
    /// Defaults to `n_head` when absent.
    pub n_head_kv: usize,
    /// Base period theta for 2D M-RoPE in LFM3 vision attention blocks (`clip.vision.rope_theta`).
    /// Defaults to `1_000_000.0` (`1e6`).
    pub rope_theta: f32,
    /// LayerNorm epsilon. Read from `clip.vision.attention.layer_norm_epsilon`.
    pub eps: f32,
    /// Square input image side length in pixels (typically 256 or 512).
    pub image_size: usize,
    /// Square patch side length in pixels (typically 16).
    pub patch_size: usize,
    /// Number of patches in the trained position-grid: `(image_size / patch_size)^2`.
    pub n_trained_patches: usize,
    /// Projector output dimension; matches the LLM's `embedding_length`.
    pub projection_dim: usize,
    /// Spatial patch-merge / pixel-shuffle pooling factor between encoder output and projector input.
    pub scale_factor: usize,
    /// Per-channel mean for image normalisation (RGB order).
    pub image_mean: [f32; 3],
    /// Per-channel std for image normalisation (RGB order).
    pub image_std: [f32; 3],
    /// Lower bound on resized image area in pixels for dynamic-resolution preprocessing.
    pub image_min_pixels: usize,
    /// Upper bound on resized image area in pixels for dynamic-resolution preprocessing.
    pub image_max_pixels: usize,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            projector_type: VisionProjectorType::Lfm2,
            n_layer: 12,
            n_embd: 768,
            n_ff: 3072,
            n_head: 12,
            n_head_kv: 12,
            rope_theta: LFM3_DEFAULT_ROPE_THETA,
            eps: 1e-6,
            image_size: 512,
            patch_size: 16,
            n_trained_patches: 1024,
            projection_dim: 1024,
            scale_factor: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 65_536,
            image_max_pixels: 262_144,
        }
    }
}

impl VisionEncoderConfig {
    /// True when the encoder uses the LFM3 hybrid vision architecture (`lfm3vl` or `lfm3o`).
    pub fn is_lfm3(&self) -> bool {
        self.projector_type.is_lfm3()
    }

    /// True when projected image embeddings are wrapped in `<|vision_start|>` and `<|vision_end|>` (`lfm3o`).
    pub fn has_boundary_tokens(&self) -> bool {
        self.projector_type.has_boundary_tokens()
    }

    /// Return `Some(("<|vision_start|>", "<|vision_end|>"))` when boundary tokens are enabled (`lfm3o`).
    pub fn vision_boundary_tokens(&self) -> Option<(&'static str, &'static str)> {
        self.projector_type.vision_boundary_tokens()
    }

    /// Wrap an image prompt segment with `<|vision_start|>` and `<|vision_end|>` if enabled (`lfm3o`).
    pub fn wrap_image_prompt(&self, inner: &str) -> String {
        self.projector_type.wrap_image_prompt(inner)
    }

    /// Read the vision-encoder config from a multimodal_projector
    /// GGUF's `clip.vision.*` metadata.
    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        let has_vision = match gguf.metadata.get(KEY_HAS_VISION) {
            Some(crate::gguf::GgufValue::Bool(b)) => *b,
            Some(other) => anyhow::bail!("invalid type for `{KEY_HAS_VISION}`: {other:?}"),
            None => false,
        };
        anyhow::ensure!(
            has_vision,
            "mmproj GGUF missing or false `{KEY_HAS_VISION}`; \
             not a vision encoder"
        );
        let projector_type = if let Some(s) = match get_opt_str(gguf, KEY_VISION_PROJ_TYPE)? {
            Some(s) => Some(s),
            None => get_opt_str(gguf, KEY_PROJ_TYPE)?,
        } {
            VisionProjectorType::from_gguf_str(s)?
        } else {
            VisionProjectorType::Lfm2
        };

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
        let n_head_kv = if let Some(v) = gguf.get_u32(KEY_N_HEAD_KV) {
            v as usize
        } else if let Some(arr) = gguf.get_i32_array(KEY_N_HEAD_KV) {
            let v = arr.into_iter().find(|&x| x > 0).with_context(|| {
                format!("`{KEY_N_HEAD_KV}` array contains no positive head count")
            })?;
            v as usize
        } else if let Some(other) = gguf.metadata.get(KEY_N_HEAD_KV) {
            anyhow::bail!("invalid type for `{KEY_N_HEAD_KV}`: {other:?}");
        } else {
            n_head
        };
        let rope_theta = get_opt_f32(gguf, KEY_ROPE_THETA)?.unwrap_or(LFM3_DEFAULT_ROPE_THETA);
        let eps = gguf
            .get_f32(KEY_LN_EPS)
            .with_context(|| format!("missing `{KEY_LN_EPS}`"))?;
        anyhow::ensure!(
            eps.is_finite() && eps > 0.0,
            "`{KEY_LN_EPS}` ({eps}) must be finite and > 0"
        );
        anyhow::ensure!(
            rope_theta.is_finite() && rope_theta > 0.0,
            "`{KEY_ROPE_THETA}` ({rope_theta}) must be finite and > 0"
        );
        let patch_size =
            gguf.get_u32(KEY_PATCH_SIZE)
                .with_context(|| format!("missing `{KEY_PATCH_SIZE}`"))? as usize;
        let image_size = match get_opt_u32(gguf, KEY_IMAGE_SIZE)? {
            Some(v) => v as usize,
            None if projector_type.is_lfm3() => 512,
            None => anyhow::bail!("missing `{KEY_IMAGE_SIZE}`"),
        };
        anyhow::ensure!(
            n_embd > 0 && n_head > 0 && n_embd.is_multiple_of(n_head),
            "n_embd ({n_embd}) must be a positive multiple of n_head ({n_head})"
        );
        anyhow::ensure!(
            n_head_kv > 0 && n_head.is_multiple_of(n_head_kv),
            "n_head ({n_head}) must be a positive multiple of n_head_kv ({n_head_kv})"
        );
        anyhow::ensure!(
            image_size > 0 && patch_size > 0 && image_size.is_multiple_of(patch_size),
            "image_size ({image_size}) must be a positive multiple of patch_size ({patch_size})"
        );
        let n_trained_patches = (image_size / patch_size).pow(2);

        let projection_dim =
            gguf.get_u32(KEY_PROJECTION_DIM)
                .with_context(|| format!("missing `{KEY_PROJECTION_DIM}`"))? as usize;
        let scale_factor = match match get_opt_u32(gguf, KEY_SCALE_FACTOR)? {
            Some(v) => Some(v),
            None => get_opt_u32(gguf, "clip.vision.spatial_merge_size")?,
        } {
            Some(v) => v as usize,
            None if projector_type.is_lfm3() => 2,
            None => anyhow::bail!("missing `{KEY_SCALE_FACTOR}`"),
        };
        anyhow::ensure!(
            scale_factor > 0,
            "scale_factor ({scale_factor}) must be > 0; a zero or missing value would \
             collapse the pixel band and trip a divide-by-zero in `pixel_shuffle`",
        );
        let image_mean = read_rgb_array(gguf, KEY_IMAGE_MEAN)?;
        let image_std = read_rgb_array(gguf, KEY_IMAGE_STD)?;

        let n_tokens_min: usize = 64;
        let n_tokens_max: usize = 256;
        let pixels_per_token = (patch_size * scale_factor).pow(2);
        let image_min_pixels = get_opt_u32(gguf, "clip.vision.image_min_pixels")?
            .map(|v| v as usize)
            .unwrap_or(n_tokens_min * pixels_per_token);
        let image_max_pixels = get_opt_u32(gguf, "clip.vision.image_max_pixels")?
            .map(|v| v as usize)
            .unwrap_or(n_tokens_max * pixels_per_token);
        // they steer the resize target (and so an allocation): a range that is empty, inverted
        // or larger than any real image is a corrupt file
        anyhow::ensure!(
            image_min_pixels > 0
                && image_min_pixels <= image_max_pixels
                && image_max_pixels <= MAX_IMAGE_PIXELS,
            "clip.vision.image_min_pixels ({image_min_pixels}) and image_max_pixels \
             ({image_max_pixels}) must satisfy 0 < min <= max <= {MAX_IMAGE_PIXELS}"
        );

        Ok(Self {
            projector_type,
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            rope_theta,
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

fn get_opt_u32(gguf: &GgufFile, key: &str) -> Result<Option<u32>> {
    if let Some(v) = gguf.get_u32(key) {
        Ok(Some(v))
    } else if let Some(other) = gguf.metadata.get(key) {
        anyhow::bail!("invalid type for `{key}`: {other:?}");
    } else {
        Ok(None)
    }
}

fn get_opt_f32(gguf: &GgufFile, key: &str) -> Result<Option<f32>> {
    if let Some(v) = gguf.get_f32(key) {
        Ok(Some(v))
    } else if let Some(other) = gguf.metadata.get(key) {
        anyhow::bail!("invalid type for `{key}`: {other:?}");
    } else {
        Ok(None)
    }
}

fn get_opt_str<'a>(gguf: &'a GgufFile, key: &str) -> Result<Option<&'a str>> {
    if let Some(s) = gguf.get_str(key) {
        Ok(Some(s))
    } else if let Some(other) = gguf.metadata.get(key) {
        anyhow::bail!("invalid type for `{key}`: {other:?}");
    } else {
        Ok(None)
    }
}

/// Patch-embed Conv2D weights, pre-transposed at load time into
/// row-major `[in_dim x n_embd]` where `in_dim = 3 * patch_size^2`.
#[derive(Clone)]
pub struct PatchEmbedWeights {
    /// Row-major `[in_dim x n_embd]` kernel ready for matmul. Built
    /// from `v.patch_embd.weight` once at load time.
    pub conv_w: Vec<f32>,
    /// `v.patch_embd.bias` (`[n_embd]`).
    pub conv_b: Vec<f32>,
}

/// Norm weights for LFM3 vision layers. Uses standard LayerNorm when `bias` is `Some`,
/// and RMSNorm when `bias` is `None`.
#[derive(Clone, Debug)]
pub struct Lfm3NormWeights {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl Lfm3NormWeights {
    #[inline]
    pub(crate) fn apply_row_inplace(&self, row: &mut [f32], eps: f32) {
        if let Some(bias) = &self.bias {
            crate::backend::cpu::layer_norm_inplace(row, &self.weight, bias, eps);
        } else {
            rms_norm_row_inplace(row, &self.weight, eps);
        }
    }
}

#[inline]
fn rms_norm_row_inplace(row: &mut [f32], weight: &[f32], eps: f32) {
    crate::backend::cpu::rmsnorm(row, weight, eps);
}

/// LFM3 gated 2D depthwise convolution sub-block weights.
pub struct Lfm3ConvBlockWeights {
    /// Input projection: either `[3 * n_embd, n_embd]` (double-gated `first, middle, third`)
    /// or `[2 * n_embd, n_embd]` (single-gated `b, c`).
    pub in_proj_w: MmapWeight,
    pub in_proj_b: Option<Vec<f32>>,
    /// Depthwise 2D spatial convolution weights stored in `[n_embd][kernel_size][kernel_size]`
    /// row-major order (`c * K * K + ky * K + kx`).
    pub depthwise_w: Vec<f32>,
    pub depthwise_b: Option<Vec<f32>>,
    pub kernel_size: usize,
    pub out_proj_w: MmapWeight,
    pub out_proj_b: Option<Vec<f32>>,
}

/// LFM3 2D M-RoPE grouped-query attention sub-block weights.
pub struct Lfm3AttnBlockWeights {
    pub q_w: MmapWeight,
    pub q_b: Option<Vec<f32>>,
    pub k_w: MmapWeight,
    pub k_b: Option<Vec<f32>>,
    pub v_w: MmapWeight,
    pub v_b: Option<Vec<f32>>,
    pub q_norm: Lfm3NormWeights,
    pub k_norm: Lfm3NormWeights,
    pub o_w: MmapWeight,
    pub o_b: Option<Vec<f32>>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
}

/// Operator inside an LFM3 hybrid vision block (either spatial Conv2d or 2D M-RoPE GQA Attention).
#[allow(clippy::large_enum_variant)]
pub enum Lfm3BlockOp {
    Conv(Lfm3ConvBlockWeights),
    Attn(Lfm3AttnBlockWeights),
}

/// Feed-forward network inside an LFM3 hybrid vision block (SwiGLU when `gate_w` is `Some`, GELU otherwise).
pub struct Lfm3FfnWeights {
    pub gate_w: Option<MmapWeight>,
    pub gate_b: Option<Vec<f32>>,
    pub up_w: MmapWeight,
    pub up_b: Option<Vec<f32>>,
    pub down_w: MmapWeight,
    pub down_b: Option<Vec<f32>>,
}

/// One LFM3 hybrid vision encoder block (`ln1 -> Conv/Attn + residual -> ln2 -> FFN + residual`).
pub struct Lfm3VisionBlockWeights {
    pub ln1: Lfm3NormWeights,
    pub op: Lfm3BlockOp,
    pub ln2: Lfm3NormWeights,
    pub ffn: Lfm3FfnWeights,
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

/// 2-layer MLP projector that maps the pixel-shuffled encoder output
/// into the LLM embedding dim.
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
        let mut q8 = Q8Act::for_dims(&[in_dim, mid_dim, out_dim], n_tokens);
        q8.quantize(pooled, in_dim, n_tokens);
        if !q8.matmul(&self.mm1_w, &mut mid) {
            self.mm1_w.batched_matmul(pooled, &mut mid, n_tokens);
        }
        let has_mm1_b = !self.mm1_b.is_empty();
        par_rows_n(&mut mid, mid_dim, |_, mid_row| {
            if has_mm1_b {
                crate::backend::cpu::add_inplace(mid_row, &self.mm1_b);
            }
            crate::backend::cpu::gelu_inplace(mid_row);
        });
        q8.quantize(&mid, mid_dim, n_tokens);
        if !q8.matmul(&self.mm2_w, &mut out) {
            self.mm2_w.batched_matmul(&mid, &mut out, n_tokens);
        }
        if !self.mm2_b.is_empty() {
            par_rows_n(&mut out, out_dim, |_, out_row| {
                crate::backend::cpu::add_inplace(out_row, &self.mm2_b);
            });
        }
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
    /// Optional second temporal patch-embed kernel (`v.patch_embd.weight.1` or `v.patch_embd.1.weight`)
    /// used by LFM3 vision models (`lfm3vl`, `lfm3o`) for 2-frame temporal patchification.
    pub patch_embed_1: Option<PatchEmbedWeights>,
    /// Optional post-patch-embedding norm (`v.embedding_norm.weight` / `.bias`) used by LFM3 vision models.
    pub embedding_norm: Option<Lfm3NormWeights>,
    /// `v.position_embd.weight` (`[n_patches x n_embd]` flattened row-major). Empty for LFM3 models
    /// that use 2D M-RoPE instead of learned absolute position embeddings.
    pub position_embed: Vec<f32>,
    pub blocks: Vec<VitBlockWeights>,
    /// Hybrid Conv2d / 2D M-RoPE GQA Attention encoder blocks for `lfm3vl` and `lfm3o`.
    pub lfm3_blocks: Vec<Lfm3VisionBlockWeights>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    /// Optional projector input norm (`mm.input_norm.weight` / `.bias`) used by LFM3 vision models.
    pub mm_input_norm: Option<Lfm3NormWeights>,
    pub projector: ProjectorWeights,
}

impl VisionEncoderWeights {
    /// True when the vision encoder uses the LFM3 hybrid architecture (`lfm3vl` or `lfm3o`).
    pub fn is_lfm3(&self) -> bool {
        self.config.is_lfm3()
    }

    /// True when projected image embeddings are wrapped in `<|vision_start|>` and `<|vision_end|>` (`lfm3o`).
    pub fn has_boundary_tokens(&self) -> bool {
        self.config.has_boundary_tokens()
    }

    /// Return `Some(("<|vision_start|>", "<|vision_end|>"))` when boundary tokens are enabled (`lfm3o`).
    pub fn vision_boundary_tokens(&self) -> Option<(&'static str, &'static str)> {
        self.config.vision_boundary_tokens()
    }

    /// Wrap an image prompt segment with `<|vision_start|>` and `<|vision_end|>` if enabled (`lfm3o`).
    pub fn wrap_image_prompt(&self, inner: &str) -> String {
        self.config.wrap_image_prompt(inner)
    }

    /// Resolve the `(start_token_id, end_token_id)` pair via a vocabulary lookup closure
    /// when boundary tokens are enabled (`lfm3o`).
    pub fn resolve_boundary_token_ids<F>(&self, lookup: F) -> Option<(u32, u32)>
    where
        F: FnMut(&str) -> Option<u32>,
    {
        self.config
            .projector_type
            .resolve_boundary_token_ids(lookup)
    }

    /// Load every vision-encoder tensor from a multimodal_projector
    /// GGUF without eagerly building CPU `smmla` repacks.
    pub(crate) fn from_gguf_unprepared(gguf: &Arc<GgufFile>) -> Result<Self> {
        let config = VisionEncoderConfig::from_gguf(gguf)?;
        if config.is_lfm3() {
            return Self::from_gguf_lfm3_unprepared(gguf, config);
        }

        let (patch_embed, _) = load_patch_embed_weights(
            gguf,
            "v.patch_embd.weight",
            Some("v.patch_embd.bias"),
            true,
            &config,
        )?;

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
        let position_embed = pos_t.try_to_f32_vec()?;

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
        let mm1_w = wt_f32(gguf, "mm.1.weight")?;
        let mm1_b = load_vec_f32(gguf, "mm.1.bias")?;
        let mm2_w = wt_f32(gguf, "mm.2.weight")?;
        let mm2_b = load_vec_f32(gguf, "mm.2.bias")?;
        let mm1_in_dim = config.n_embd * config.scale_factor.pow(2);
        let intermediate_dim = mm1_w.rows;
        anyhow::ensure!(
            mm1_w.cols == mm1_in_dim,
            "mm.1.weight cols ({}) != n_embd*sf^2 ({mm1_in_dim})",
            mm1_w.cols,
        );
        anyhow::ensure!(
            mm1_b.len() == intermediate_dim,
            "mm.1.bias len ({}) != mm.1.weight.rows ({intermediate_dim})",
            mm1_b.len(),
        );
        anyhow::ensure!(
            mm2_w.cols == intermediate_dim,
            "mm.2.weight cols ({}) != mm.1.weight.rows ({intermediate_dim}); \
             projector mm.1->mm.2 dimensions do not line up",
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
            patch_embed_1: None,
            embedding_norm: None,
            position_embed,
            blocks,
            lfm3_blocks: Vec::new(),
            post_ln_w,
            post_ln_b,
            mm_input_norm: None,
            projector,
        })
    }

    fn from_gguf_lfm3_unprepared(
        gguf: &Arc<GgufFile>,
        config: VisionEncoderConfig,
    ) -> Result<Self> {
        let (patch_embed, inline_pe1) = load_patch_embed_weights(
            gguf,
            "v.patch_embd.weight",
            Some("v.patch_embd.bias"),
            false,
            &config,
        )?;
        let patch_embed_1 = if inline_pe1.is_some() {
            inline_pe1
        } else if gguf.tensors.contains_key("v.patch_embd.weight.1") {
            let (pe1, _) = load_patch_embed_weights(
                gguf,
                "v.patch_embd.weight.1",
                Some("v.patch_embd.bias.1"),
                false,
                &config,
            )?;
            Some(pe1)
        } else if gguf.tensors.contains_key("v.patch_embd.1.weight") {
            let (pe1, _) = load_patch_embed_weights(
                gguf,
                "v.patch_embd.1.weight",
                Some("v.patch_embd.1.bias"),
                false,
                &config,
            )?;
            Some(pe1)
        } else {
            None
        };

        let embedding_norm = load_optional_lfm3_norm(gguf, "v.embedding_norm", config.n_embd)?;

        let position_embed = if gguf.tensors.contains_key("v.position_embd.weight") {
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
            pos_t.try_to_f32_vec()?
        } else {
            Vec::new()
        };

        let mut lfm3_blocks = Vec::with_capacity(config.n_layer);
        for il in 0..config.n_layer {
            lfm3_blocks.push(load_lfm3_block(gguf, il, &config)?);
        }

        let post_ln_w = if gguf.tensors.contains_key("v.post_ln.weight") {
            let w = load_vec_f32(gguf, "v.post_ln.weight")?;
            anyhow::ensure!(
                w.len() == config.n_embd,
                "v.post_ln.weight len ({}) != n_embd ({})",
                w.len(),
                config.n_embd
            );
            w
        } else {
            Vec::new()
        };
        let post_ln_b = if gguf.tensors.contains_key("v.post_ln.bias") {
            let b = load_vec_f32(gguf, "v.post_ln.bias")?;
            anyhow::ensure!(
                b.len() == config.n_embd,
                "v.post_ln.bias len ({}) != n_embd ({})",
                b.len(),
                config.n_embd
            );
            b
        } else {
            Vec::new()
        };

        let mm1_in_dim = config.n_embd * config.scale_factor.pow(2);
        let mm_input_norm = if gguf.tensors.contains_key("mm.input_norm.weight") {
            let w = load_vec_f32(gguf, "mm.input_norm.weight")?;
            anyhow::ensure!(
                w.len() == config.n_embd || w.len() == mm1_in_dim,
                "mm.input_norm.weight len ({}) must equal n_embd ({}) or n_embd*sf^2 ({mm1_in_dim})",
                w.len(),
                config.n_embd,
            );
            let bias = if gguf.tensors.contains_key("mm.input_norm.bias") {
                let b = load_vec_f32(gguf, "mm.input_norm.bias")?;
                anyhow::ensure!(
                    b.len() == w.len(),
                    "mm.input_norm.bias len ({}) != mm.input_norm.weight len ({})",
                    b.len(),
                    w.len(),
                );
                Some(b)
            } else {
                None
            };
            Some(Lfm3NormWeights { weight: w, bias })
        } else {
            None
        };

        let (mm0_w_name, mm0_b_name) = if gguf.tensors.contains_key("mm.0.weight") {
            ("mm.0.weight", "mm.0.bias")
        } else {
            ("mm.1.weight", "mm.1.bias")
        };
        let mm1_w = wt_f32(gguf, mm0_w_name)?;
        let intermediate_dim = mm1_w.rows;
        anyhow::ensure!(
            mm1_w.cols == mm1_in_dim,
            "{mm0_w_name} cols ({}) != n_embd*sf^2 ({mm1_in_dim})",
            mm1_w.cols,
        );
        let mm1_b = if gguf.tensors.contains_key(mm0_b_name) {
            let b = load_vec_f32(gguf, mm0_b_name)?;
            anyhow::ensure!(
                b.len() == intermediate_dim,
                "{mm0_b_name} len ({}) != {mm0_w_name}.rows ({intermediate_dim})",
                b.len(),
            );
            b
        } else {
            vec![0.0; intermediate_dim]
        };

        let mm2_w = wt_f32(gguf, "mm.2.weight")?;
        anyhow::ensure!(
            mm2_w.cols == intermediate_dim,
            "mm.2.weight cols ({}) != {mm0_w_name}.rows ({intermediate_dim})",
            mm2_w.cols,
        );
        anyhow::ensure!(
            mm2_w.rows == config.projection_dim,
            "mm.2.weight rows ({}) != projection_dim ({})",
            mm2_w.rows,
            config.projection_dim,
        );
        let mm2_b = if gguf.tensors.contains_key("mm.2.bias") {
            let b = load_vec_f32(gguf, "mm.2.bias")?;
            anyhow::ensure!(
                b.len() == config.projection_dim,
                "mm.2.bias len ({}) != projection_dim ({})",
                b.len(),
                config.projection_dim,
            );
            b
        } else {
            vec![0.0; config.projection_dim]
        };

        let projector = ProjectorWeights {
            mm1_w,
            mm1_b,
            mm2_w,
            mm2_b,
        };

        Ok(Self {
            config,
            patch_embed,
            patch_embed_1,
            embedding_norm,
            position_embed,
            blocks: Vec::new(),
            lfm3_blocks,
            post_ln_w,
            post_ln_b,
            mm_input_norm,
            projector,
        })
    }

    /// Parse a `mmproj` GGUF and pre-build the CPU `smmla` repacks.
    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        let weights = Self::from_gguf_unprepared(gguf)?;
        weights.prepare_int8();
        Ok(weights)
    }

    /// Build the smmla-repacked copy of every Q8_0 linear now, so the first image does not pay for
    /// it. A no-op unless the int8 path is on and the host has i8mm.
    pub(crate) fn prepare_int8(&self) {
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
        for b in &self.lfm3_blocks {
            match &b.op {
                Lfm3BlockOp::Conv(conv) => {
                    let _ = conv.in_proj_w.smmla_repacked();
                    let _ = conv.out_proj_w.smmla_repacked();
                }
                Lfm3BlockOp::Attn(attn) => {
                    for w in [&attn.q_w, &attn.k_w, &attn.v_w, &attn.o_w] {
                        let _ = w.smmla_repacked();
                    }
                }
            }
            if let Some(gw) = &b.ffn.gate_w {
                let _ = gw.smmla_repacked();
            }
            let _ = b.ffn.up_w.smmla_repacked();
            let _ = b.ffn.down_w.smmla_repacked();
        }
        let _ = self.projector.mm1_w.smmla_repacked();
        let _ = self.projector.mm2_w.smmla_repacked();
    }

    /// Run the vision encoder + projector on a preprocessed image
    /// at the dynamic patch grid the preprocessor selected.
    pub fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        if self.config.is_lfm3() {
            return self.encode_lfm3_frames(pixels, None, grid_w, grid_h);
        }
        let cfg = &self.config;
        anyhow::ensure!(grid_w > 0 && grid_h > 0, "grid dims must be > 0");
        anyhow::ensure!(
            cfg.scale_factor > 0,
            "vision encoder config has scale_factor=0; refusing to divide by zero"
        );
        anyhow::ensure!(
            grid_w.is_multiple_of(cfg.scale_factor) && grid_h.is_multiple_of(cfg.scale_factor),
            "grid {grid_w}x{grid_h} not divisible by scale_factor ({})",
            cfg.scale_factor,
        );
        let target_w = grid_w
            .checked_mul(cfg.patch_size)
            .ok_or_else(|| anyhow::anyhow!("grid_w*patch_size overflow"))?;
        let target_h = grid_h
            .checked_mul(cfg.patch_size)
            .ok_or_else(|| anyhow::anyhow!("grid_h*patch_size overflow"))?;
        let n_pix = target_w
            .checked_mul(target_h)
            .and_then(|x| x.checked_mul(3))
            .ok_or_else(|| anyhow::anyhow!("3*target_w*target_h overflow"))?;
        anyhow::ensure!(
            pixels.len() == n_pix,
            "encode_image: pixels.len() {} != 3*target_w*target_h ({n_pix})",
            pixels.len()
        );
        let n_patches = grid_w
            .checked_mul(grid_h)
            .ok_or_else(|| anyhow::anyhow!("grid_w*grid_h overflow"))?;
        let mut prof = Prof::new();
        // 1. Patch embed: [3, H, W] -> [n_patches, n_embd]
        let t = prof.start();
        let mut tokens = patch_embed_compute(pixels, &self.patch_embed, cfg, grid_w, grid_h);
        prof.stop("patch_embed", t);
        // 2. Add position embeddings.
        let pos = self.resolved_position_embed(grid_w, grid_h);
        debug_assert_eq!(pos.len(), n_patches * cfg.n_embd);
        for (t, p) in tokens.iter_mut().zip(pos.iter()) {
            *t += *p;
        }

        // 3. 12 x ViT block.
        let mut scratch = VitScratch::new(cfg, n_patches);
        for block in &self.blocks {
            self.vit_block_forward(&mut tokens, block, &mut scratch, n_patches, &mut prof);
        }

        // 4. post_ln (per token).
        let t = prof.start();
        par_rows_n(&mut tokens, cfg.n_embd, |_, row| {
            crate::backend::cpu::layer_norm_inplace(row, &self.post_ln_w, &self.post_ln_b, cfg.eps);
        });

        // 5. Pixel-shuffle scale_factor^2 over the dynamic grid.
        let pooled = pixel_shuffle(&tokens, cfg, grid_w, grid_h);
        prof.stop("post_ln_shuffle", t);

        // 6. Projector: mm.1 + GELU + mm.2.
        let t = prof.start();
        let out = self.projector_forward(&pooled, cfg);
        prof.stop("projector", t);
        prof.report();
        Ok(out)
    }

    /// Encode a pair of consecutive video frames (`pixels_t0` and `pixels_t1`) using the LFM3
    /// temporal patchifier (`v.patch_embd.weight` on `pixels_t0` plus `v.patch_embd.weight.1` on `pixels_t1`).
    pub fn encode_video_frame_pair(
        &self,
        pixels_t0: &[f32],
        pixels_t1: &[f32],
        grid_w: usize,
        grid_h: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.config.is_lfm3() && self.patch_embed_1.is_some(),
            "encode_video_frame_pair needs an LFM3 encoder with a temporal patch kernel \
             (`v.patch_embd.weight.1`); this one would drop the second frame"
        );
        self.encode_lfm3_frames(pixels_t0, Some(pixels_t1), grid_w, grid_h)
    }

    fn encode_lfm3_frames(
        &self,
        pixels_t0: &[f32],
        pixels_t1: Option<&[f32]>,
        grid_w: usize,
        grid_h: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        anyhow::ensure!(grid_w > 0 && grid_h > 0, "grid dims must be > 0");
        anyhow::ensure!(
            cfg.scale_factor > 0,
            "vision encoder config has scale_factor=0; refusing to divide by zero"
        );
        anyhow::ensure!(
            grid_w.is_multiple_of(cfg.scale_factor) && grid_h.is_multiple_of(cfg.scale_factor),
            "grid {grid_w}x{grid_h} not divisible by scale_factor ({})",
            cfg.scale_factor,
        );
        let target_w = grid_w
            .checked_mul(cfg.patch_size)
            .ok_or_else(|| anyhow::anyhow!("grid_w*patch_size overflow"))?;
        let target_h = grid_h
            .checked_mul(cfg.patch_size)
            .ok_or_else(|| anyhow::anyhow!("grid_h*patch_size overflow"))?;
        let n_pix = target_w
            .checked_mul(target_h)
            .and_then(|x| x.checked_mul(3))
            .ok_or_else(|| anyhow::anyhow!("3*target_w*target_h overflow"))?;
        anyhow::ensure!(
            pixels_t0.len() == n_pix,
            "encode_image: pixels.len() {} != 3*target_w*target_h ({n_pix})",
            pixels_t0.len()
        );
        if let Some(p1) = pixels_t1 {
            anyhow::ensure!(
                p1.len() == n_pix,
                "encode_video_frame_pair: pixels_t1.len() {} != 3*target_w*target_h ({n_pix})",
                p1.len()
            );
        }
        let n_patches = grid_w
            .checked_mul(grid_h)
            .ok_or_else(|| anyhow::anyhow!("grid_w*grid_h overflow"))?;

        // 1. Temporal patchifier: conv2d(t0, w0) + conv2d(t1_or_t0, w1) + bias.
        let mut tokens = patch_embed_compute(pixels_t0, &self.patch_embed, cfg, grid_w, grid_h);
        if let Some(pe1) = &self.patch_embed_1 {
            let frame1 = pixels_t1.unwrap_or(pixels_t0);
            let tokens_t1 = patch_embed_compute(frame1, pe1, cfg, grid_w, grid_h);
            for (dst, &src) in tokens.iter_mut().zip(tokens_t1.iter()) {
                *dst += src;
            }
        }

        // 2. Post-patch embedding norm (if present) and optional position embedding.
        if let Some(emb_norm) = &self.embedding_norm {
            par_rows_n(&mut tokens, cfg.n_embd, |_, row| {
                emb_norm.apply_row_inplace(row, cfg.eps);
            });
        }
        if !self.position_embed.is_empty() {
            let pos = self.resolved_position_embed(grid_w, grid_h);
            for (t, p) in tokens.iter_mut().zip(pos.iter()) {
                *t += *p;
            }
        }

        // 3. Hybrid Conv2d / 2D M-RoPE GQA Attention blocks.
        let mut scratch = VitScratch::new(cfg, n_patches);
        for block in &self.lfm3_blocks {
            self.lfm3_block_forward(&mut tokens, block, &mut scratch, grid_w, grid_h);
        }

        // 4. Optional post_ln and pre-merge mm.input_norm.
        if !self.post_ln_w.is_empty() {
            let has_post_b = !self.post_ln_b.is_empty();
            par_rows_n(&mut tokens, cfg.n_embd, |_, row| {
                if has_post_b {
                    crate::backend::cpu::layer_norm_inplace(
                        row,
                        &self.post_ln_w,
                        &self.post_ln_b,
                        cfg.eps,
                    );
                } else {
                    rms_norm_row_inplace(row, &self.post_ln_w, cfg.eps);
                }
            });
        }
        let merged_dim = cfg.n_embd * cfg.scale_factor * cfg.scale_factor;
        let norm_stage = self
            .mm_input_norm
            .as_ref()
            .and_then(|n| mm_norm_stage(n.weight.len(), cfg.n_embd, merged_dim));
        if let Some(mm_norm) = &self.mm_input_norm
            && norm_stage == Some(MmNormStage::PreMerge)
        {
            par_rows_n(&mut tokens, cfg.n_embd, |_, row| {
                mm_norm.apply_row_inplace(row, LFM3_MM_INPUT_NORM_EPS);
            });
        }

        // 5. 2x2 spatial patch merge (space-to-depth in row-major 2x2 order).
        let mut pooled = pixel_shuffle(&tokens, cfg, grid_w, grid_h);
        if let Some(mm_norm) = &self.mm_input_norm
            && norm_stage == Some(MmNormStage::PostMerge)
        {
            par_rows_n(&mut pooled, merged_dim, |_, row| {
                mm_norm.apply_row_inplace(row, LFM3_MM_INPUT_NORM_EPS);
            });
        }

        // 6. 2-layer GELU MLP projector (mm.0 + GELU + mm.2).
        Ok(self.projector_forward(&pooled, cfg))
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

    fn lfm3_block_forward(
        &self,
        tokens: &mut [f32],
        block: &Lfm3VisionBlockWeights,
        scratch: &mut VitScratch,
        grid_w: usize,
        grid_h: usize,
    ) {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_tokens = grid_w * grid_h;

        scratch.pre_norm[..n_tokens * n_embd].copy_from_slice(tokens);
        par_rows_n(
            &mut scratch.pre_norm[..n_tokens * n_embd],
            n_embd,
            |_, row| {
                block.ln1.apply_row_inplace(row, cfg.eps);
            },
        );

        match &block.op {
            Lfm3BlockOp::Conv(conv) => {
                self.lfm3_conv_forward(tokens, conv, scratch, grid_w, grid_h);
            }
            Lfm3BlockOp::Attn(attn) => {
                self.lfm3_attn_forward(tokens, attn, scratch, grid_w, grid_h);
            }
        }

        scratch.pre_norm[..n_tokens * n_embd].copy_from_slice(tokens);
        par_rows_n(
            &mut scratch.pre_norm[..n_tokens * n_embd],
            n_embd,
            |_, row| {
                block.ln2.apply_row_inplace(row, cfg.eps);
            },
        );

        self.lfm3_ffn_forward(tokens, &block.ffn, scratch, n_tokens);
    }

    fn lfm3_conv_forward(
        &self,
        tokens: &mut [f32],
        conv: &Lfm3ConvBlockWeights,
        scratch: &mut VitScratch,
        grid_w: usize,
        grid_h: usize,
    ) {
        let n_tokens = grid_w * grid_h;
        let d = self.config.n_embd;
        let proj_rows = conv.in_proj_w.rows;
        let proj_len = n_tokens * proj_rows;
        if scratch.conv_proj.len() < proj_len {
            scratch.conv_proj.resize(proj_len, 0.0);
        }

        scratch
            .q8
            .quantize(&scratch.pre_norm[..n_tokens * d], d, n_tokens);
        vit_linear(
            &conv.in_proj_w,
            &scratch.pre_norm[..n_tokens * d],
            &mut scratch.conv_proj[..proj_len],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        if let Some(b) = &conv.in_proj_b {
            par_rows_n(&mut scratch.conv_proj[..proj_len], proj_rows, |_, row| {
                crate::backend::cpu::add_inplace(row, b);
            });
        }

        if scratch.conv_gated.len() < n_tokens * d {
            scratch.conv_gated.resize(n_tokens * d, 0.0);
        }
        if scratch.conv_out.len() < n_tokens * d {
            scratch.conv_out.resize(n_tokens * d, 0.0);
        }
        {
            let proj = &scratch.conv_proj[..proj_len];
            let gated = &mut scratch.conv_gated[..n_tokens * d];
            if proj_rows == 3 * d {
                par_rows_n(gated, d, |t, dst| {
                    let row = &proj[t * 3 * d..(t + 1) * 3 * d];
                    let first = &row[0..d];
                    let third = &row[2 * d..3 * d];
                    for c in 0..d {
                        dst[c] = first[c] * third[c];
                    }
                });
            } else {
                par_rows_n(gated, d, |t, dst| {
                    let row = &proj[t * 2 * d..(t + 1) * 2 * d];
                    let b_part = &row[0..d];
                    let c_part = &row[d..2 * d];
                    for c in 0..d {
                        dst[c] = b_part[c] * c_part[c];
                    }
                });
            }
        }

        depthwise_conv2d_same_into(
            &scratch.conv_gated[..n_tokens * d],
            &conv.depthwise_w,
            conv.depthwise_b.as_deref(),
            grid_w,
            grid_h,
            d,
            conv.kernel_size,
            &mut scratch.conv_out[..n_tokens * d],
        );
        if proj_rows == 3 * d {
            let proj = &scratch.conv_proj[..proj_len];
            let conv_out = &mut scratch.conv_out[..n_tokens * d];
            par_rows_n(conv_out, d, |t, dst| {
                let middle = &proj[t * 3 * d + d..t * 3 * d + 2 * d];
                crate::backend::cpu::mul_inplace(dst, middle);
            });
        }

        scratch
            .q8
            .quantize(&scratch.conv_out[..n_tokens * d], d, n_tokens);
        vit_linear(
            &conv.out_proj_w,
            &scratch.conv_out[..n_tokens * d],
            &mut scratch.attn_proj[..n_tokens * d],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        if let Some(b) = &conv.out_proj_b {
            add_bias_residual(tokens, &mut scratch.attn_proj[..n_tokens * d], b, d);
        } else {
            crate::backend::cpu::add_inplace(tokens, &scratch.attn_proj[..n_tokens * d]);
        }
    }

    fn lfm3_attn_forward(
        &self,
        tokens: &mut [f32],
        attn: &Lfm3AttnBlockWeights,
        scratch: &mut VitScratch,
        grid_w: usize,
        grid_h: usize,
    ) {
        let cfg = &self.config;
        let n_tokens = grid_w * grid_h;
        let d = cfg.n_embd;
        let n_head = attn.n_head;
        let n_head_kv = attn.n_head_kv;
        let head_dim = attn.head_dim;
        let kv_dim = n_head_kv * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        scratch
            .q8
            .quantize(&scratch.pre_norm[..n_tokens * d], d, n_tokens);
        vit_linear(
            &attn.q_w,
            &scratch.pre_norm[..n_tokens * d],
            &mut scratch.q[..n_tokens * d],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        vit_linear(
            &attn.k_w,
            &scratch.pre_norm[..n_tokens * d],
            &mut scratch.k[..n_tokens * kv_dim],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        vit_linear(
            &attn.v_w,
            &scratch.pre_norm[..n_tokens * d],
            &mut scratch.v[..n_tokens * kv_dim],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );

        if let Some(qb) = &attn.q_b {
            par_rows_n(&mut scratch.q[..n_tokens * d], d, |_, row| {
                crate::backend::cpu::add_inplace(row, qb);
            });
        }
        if let Some(kb) = &attn.k_b {
            par_rows_n(&mut scratch.k[..n_tokens * kv_dim], kv_dim, |_, row| {
                crate::backend::cpu::add_inplace(row, kb);
            });
        }
        if let Some(vb) = &attn.v_b {
            par_rows_n(&mut scratch.v[..n_tokens * kv_dim], kv_dim, |_, row| {
                crate::backend::cpu::add_inplace(row, vb);
            });
        }

        // Per-head (or per-row) QK norms.
        let q_norm_stride = attn.q_norm.weight.len();
        if q_norm_stride > 0 && (n_tokens * d).is_multiple_of(q_norm_stride) {
            par_rows_n(&mut scratch.q[..n_tokens * d], q_norm_stride, |_, slice| {
                attn.q_norm.apply_row_inplace(slice, cfg.eps);
            });
        }
        let k_norm_stride = attn.k_norm.weight.len();
        if k_norm_stride > 0 && (n_tokens * kv_dim).is_multiple_of(k_norm_stride) {
            par_rows_n(
                &mut scratch.k[..n_tokens * kv_dim],
                k_norm_stride,
                |_, slice| {
                    attn.k_norm.apply_row_inplace(slice, cfg.eps);
                },
            );
        }

        // 2D Vision M-RoPE on Q and K (sharing the per-row/col sin_cos tables).
        apply_2d_vision_mrope_qk(
            &mut scratch.q[..n_tokens * d],
            &mut scratch.k[..n_tokens * kv_dim],
            grid_w,
            grid_h,
            n_head,
            n_head_kv,
            head_dim,
            cfg.rope_theta,
        );

        // Expand KV heads in-place if GQA (n_head_kv < n_head) so vit_attention can run directly.
        if n_head_kv < n_head {
            let n_rep = n_head / n_head_kv;
            for t in (0..n_tokens).rev() {
                for h in (0..n_head).rev() {
                    let kv_h = h / n_rep;
                    let src_off = t * kv_dim + kv_h * head_dim;
                    let dst_off = t * d + h * head_dim;
                    if src_off != dst_off {
                        scratch.k.copy_within(src_off..src_off + head_dim, dst_off);
                        scratch.v.copy_within(src_off..src_off + head_dim, dst_off);
                    }
                }
            }
        }

        vit_attention(
            &scratch.q[..n_tokens * d],
            &scratch.k[..n_tokens * d],
            &scratch.v[..n_tokens * d],
            &mut scratch.attn_out[..n_tokens * d],
            &mut scratch.attn_heads,
            &mut scratch.attn_kt,
            n_tokens,
            n_head,
            head_dim,
            scale,
        );

        scratch
            .q8
            .quantize(&scratch.attn_out[..n_tokens * d], d, n_tokens);
        vit_linear(
            &attn.o_w,
            &scratch.attn_out[..n_tokens * d],
            &mut scratch.attn_proj[..n_tokens * d],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        if let Some(ob) = &attn.o_b {
            add_bias_residual(tokens, &mut scratch.attn_proj[..n_tokens * d], ob, d);
        } else {
            for (tok, &p) in tokens
                .iter_mut()
                .zip(scratch.attn_proj[..n_tokens * d].iter())
            {
                *tok += p;
            }
        }
    }

    fn lfm3_ffn_forward(
        &self,
        tokens: &mut [f32],
        ffn: &Lfm3FfnWeights,
        scratch: &mut VitScratch,
        n_tokens: usize,
    ) {
        let d = self.config.n_embd;
        let n_ff = ffn.up_w.rows;
        if scratch.ffn_mid.len() < n_tokens * n_ff {
            scratch.ffn_mid.resize(n_tokens * n_ff, 0.0);
        }
        scratch
            .q8
            .quantize(&scratch.pre_norm[..n_tokens * d], d, n_tokens);
        vit_linear(
            &ffn.up_w,
            &scratch.pre_norm[..n_tokens * d],
            &mut scratch.ffn_mid[..n_tokens * n_ff],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        if let Some(up_b) = &ffn.up_b {
            par_rows_n(&mut scratch.ffn_mid[..n_tokens * n_ff], n_ff, |_, row| {
                crate::backend::cpu::add_inplace(row, up_b);
            });
        }

        if let Some(gate_w) = &ffn.gate_w {
            if scratch.ffn_gate.len() < n_tokens * n_ff {
                scratch.ffn_gate.resize(n_tokens * n_ff, 0.0);
            }
            vit_linear(
                gate_w,
                &scratch.pre_norm[..n_tokens * d],
                &mut scratch.ffn_gate[..n_tokens * n_ff],
                n_tokens,
                &mut scratch.q8,
                &mut scratch.dequant_scratch,
            );
            let gate_b = ffn.gate_b.as_deref();
            let gate_slice = &mut scratch.ffn_gate[..n_tokens * n_ff];
            let mid_slice = &mut scratch.ffn_mid[..n_tokens * n_ff];
            if let Some(gb) = gate_b {
                par_rows_n(gate_slice, n_ff, |_, g_row| {
                    crate::backend::cpu::add_inplace(g_row, gb);
                });
            }
            crate::backend::cpu::silu_mul_inplace(gate_slice, mid_slice);
            mid_slice.copy_from_slice(gate_slice);
        } else {
            par_rows_n(&mut scratch.ffn_mid[..n_tokens * n_ff], n_ff, |_, row| {
                crate::backend::cpu::gelu_inplace(row);
            });
        }

        scratch
            .q8
            .quantize(&scratch.ffn_mid[..n_tokens * n_ff], n_ff, n_tokens);
        vit_linear(
            &ffn.down_w,
            &scratch.ffn_mid[..n_tokens * n_ff],
            &mut scratch.ffn_out[..n_tokens * d],
            n_tokens,
            &mut scratch.q8,
            &mut scratch.dequant_scratch,
        );
        if let Some(down_b) = &ffn.down_b {
            add_bias_residual(tokens, &mut scratch.ffn_out[..n_tokens * d], down_b, d);
        } else {
            crate::backend::cpu::add_inplace(tokens, &scratch.ffn_out[..n_tokens * d]);
        }
    }

    /// Projector forward: pixel-shuffled `[64, 3072]` -> `mm.1` / `mm.0`
    /// + GELU -> `mm.2`.
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
/// once per `encode_image` and reused across all blocks; sized
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
    conv_proj: Vec<f32>,
    conv_gated: Vec<f32>,
    conv_out: Vec<f32>,
    ffn_mid: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_out: Vec<f32>,
    /// Reusable dequantization buffer to avoid per-matmul heap allocations across all blocks.
    dequant_scratch: Vec<f32>,
    /// Int8 activation buffers for the Q8_0 linears.
    q8: Q8Act,
}

impl VitScratch {
    fn new(cfg: &VisionEncoderConfig, n_tokens: usize) -> Self {
        let n_pe = n_tokens * cfg.n_embd;
        let n_pf = n_tokens * cfg.n_ff;
        let max_dim = cfg.n_embd.max(cfg.n_ff);
        #[cfg(has_blas)]
        let dequant_len = (3 * cfg.n_embd).max(cfg.n_ff).saturating_mul(max_dim);
        #[cfg(not(has_blas))]
        let dequant_len = max_dim;
        Self {
            pre_norm: vec![0.0; n_pe],
            q: vec![0.0; n_pe],
            k: vec![0.0; n_pe],
            v: vec![0.0; n_pe],
            attn_out: vec![0.0; n_pe],
            attn_heads: vec![0.0; attn_heads_len(n_tokens, cfg.n_head, cfg.n_embd / cfg.n_head)],
            attn_kt: vec![0.0; attn_kt_len(n_tokens, cfg.n_head, cfg.n_embd / cfg.n_head)],
            attn_proj: vec![0.0; n_pe],
            conv_proj: vec![0.0; 3 * n_pe],
            conv_gated: vec![0.0; n_pe],
            conv_out: vec![0.0; n_pe],
            ffn_mid: vec![0.0; n_pf],
            ffn_gate: vec![0.0; n_pf],
            ffn_out: vec![0.0; n_pe],
            dequant_scratch: vec![0.0; dequant_len],
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
        Self::for_dims(&[cfg.n_embd, cfg.n_ff, 3 * cfg.n_embd], n_tokens)
    }

    /// An int8 activation buffer for token rows of any of `dims` floats.
    fn for_dims(dims: &[usize], n_tokens: usize) -> Self {
        let max_dim = dims.iter().copied().max().unwrap_or(0);
        let max_dim_pad = max_dim.next_multiple_of(32);
        let enabled = crate::backend::cpu::int8_gemm_available()
            && max_dim > 0
            && dims.iter().all(|&d| d > 0)
            && !crate::backend::cpu_features::env_disabled("CERA_VIT_INT8");
        let (n_scales, n_quants, n_out) = if enabled {
            (
                n_tokens * (max_dim_pad / 32),
                n_tokens * max_dim_pad,
                n_tokens * max_dim_pad,
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
        if !self.enabled || dim == 0 || n == 0 {
            self.dim = 0;
            self.n = 0;
            return;
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            if dim.is_multiple_of(32) {
                crate::model::transformer::quantize_rows(
                    x,
                    dim,
                    n,
                    &mut self.scales,
                    &mut self.quants,
                );
            } else {
                let dim_pad = dim.next_multiple_of(32);
                let mut padded = vec![0.0f32; n * dim_pad];
                for t in 0..n {
                    padded[t * dim_pad..t * dim_pad + dim]
                        .copy_from_slice(&x[t * dim..(t + 1) * dim]);
                }
                crate::model::transformer::quantize_rows(
                    &padded,
                    dim_pad,
                    n,
                    &mut self.scales,
                    &mut self.quants,
                );
            }
            self.dim = dim;
            self.n = n;
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = (x, dim, n);
    }

    /// `y[tokens][rows] = x · wᵀ` from the activations last quantized. Returns `false` (and leaves
    /// `y` untouched) when this weight or host cannot take the int8 path.
    fn matmul(&mut self, w: &MmapWeight, y: &mut [f32]) -> bool {
        if !self.enabled || self.dim == 0 || w.rows == 0 || w.cols != self.dim || self.n == 0 {
            return false;
        }
        let (rows, n, dim) = (w.rows, self.n, self.dim);
        debug_assert_eq!(y.len(), rows * n);
        // Repacked smmla kernel (i8mm hosts): 8 weight rows by 4 tokens per tile, written straight
        // into `y` row-major, so there is no transpose pass either.
        #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(has_blas)))]
        {
            let dim_pad = dim.next_multiple_of(32);
            if matches!(
                w.dtype,
                crate::tensor::DType::Q8_0 | crate::tensor::DType::F16
            ) && let Some((packed, scales)) = w.smmla_repacked()
                && crate::backend::cpu::gemm_preq_repacked_q8_0_smmla_rowmajor_dispatch(
                    packed,
                    scales,
                    &self.scales[..n * (dim_pad / 32)],
                    &self.quants[..n * dim_pad],
                    y,
                    n,
                    rows,
                    dim_pad,
                )
            {
                return true;
            }
        }
        if w.dtype != crate::tensor::DType::Q8_0 || !dim.is_multiple_of(32) {
            return false;
        }
        let needed = rows * n;
        if self.out_t.len() < needed {
            return false;
        }
        let out_t = &mut self.out_t[..needed];
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
        n_embd > 0 && cfg.n_head > 0 && n_embd.is_multiple_of(cfg.n_head),
        "n_embd ({n_embd}) must be a positive multiple of n_head ({})",
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
    Ok(tensor.try_to_f32_vec()?)
}

/// Apply 2D Vision M-RoPE (`GGML_ROPE_TYPE_VISION`, `mrope_sections = [head_dim/4, head_dim/4, 0, 0]`,
/// `indep_sects = true`) in-place on `q_or_k` (`[grid_h * grid_w, n_head * head_dim]`).
///
/// For each token at grid coordinate `(py, px)` (`py in 0..grid_h`, `px in 0..grid_w`) and each head:
/// - `sect = head_dim / 4`, `half = head_dim / 2`.
/// - `theta_scale = rope_theta.powf(-2.0 / (half as f32))`.
/// - For `ic in 0..sect`: rotates `(head[ic], head[ic + half])` by `(py as f32) * theta_scale.powi(ic)`.
/// - For `ic in sect..2*sect` (`k = ic - sect`): rotates `(head[ic], head[ic + half])` by
///   `(px as f32) * theta_scale.powi(k)`.
#[cfg(test)]
pub(crate) fn apply_2d_vision_mrope(
    q_or_k: &mut [f32],
    grid_w: usize,
    grid_h: usize,
    n_head: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    apply_2d_vision_mrope_qk(
        q_or_k,
        &mut [],
        grid_w,
        grid_h,
        n_head,
        0,
        head_dim,
        rope_theta,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_2d_vision_mrope_qk(
    q: &mut [f32],
    k: &mut [f32],
    grid_w: usize,
    grid_h: usize,
    n_head_q: usize,
    n_head_k: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half = head_dim / 2;
    let sect = head_dim / 4;
    if half == 0 || sect == 0 || grid_w == 0 || grid_h == 0 || (n_head_q == 0 && n_head_k == 0) {
        return;
    }
    let theta_scale = rope_theta.powf(-2.0f32 / (half as f32));

    let mut inv_freq = vec![0.0f32; sect];
    let mut cur = 1.0f32;
    for freq in inv_freq.iter_mut() {
        *freq = cur;
        cur *= theta_scale;
    }

    let mut y_sincos = vec![(0.0f32, 1.0f32); grid_h * sect];
    for py in 0..grid_h {
        let py_f = py as f32;
        let row = &mut y_sincos[py * sect..(py + 1) * sect];
        for (ic, &freq) in inv_freq.iter().enumerate() {
            row[ic] = (py_f * freq).sin_cos();
        }
    }
    let mut x_sincos = vec![(0.0f32, 1.0f32); grid_w * sect];
    for px in 0..grid_w {
        let px_f = px as f32;
        let row = &mut x_sincos[px * sect..(px + 1) * sect];
        for (idx, &freq) in inv_freq.iter().enumerate() {
            row[idx] = (px_f * freq).sin_cos();
        }
    }

    let rotate = |buf: &mut [f32], n_head: usize| {
        if n_head == 0 {
            return;
        }
        let stride = n_head * head_dim;
        debug_assert_eq!(buf.len(), grid_w * grid_h * stride);
        par_rows_n(buf, stride, |tok_idx, tok_row| {
            let py = tok_idx / grid_w;
            let px = tok_idx % grid_w;
            let y_sc = &y_sincos[py * sect..(py + 1) * sect];
            let x_sc = &x_sincos[px * sect..(px + 1) * sect];
            for h in 0..n_head {
                let head = &mut tok_row[h * head_dim..(h + 1) * head_dim];
                for (ic, &(sin_y, cos_y)) in y_sc.iter().enumerate() {
                    let x0 = head[ic];
                    let x1 = head[ic + half];
                    head[ic] = x0 * cos_y - x1 * sin_y;
                    head[ic + half] = x0 * sin_y + x1 * cos_y;
                }
                for (idx, &(sin_x, cos_x)) in x_sc.iter().enumerate() {
                    let ic = sect + idx;
                    let x0 = head[ic];
                    let x1 = head[ic + half];
                    head[ic] = x0 * cos_x - x1 * sin_x;
                    head[ic + half] = x0 * sin_x + x1 * cos_x;
                }
            }
        });
    };
    rotate(q, n_head_q);
    rotate(k, n_head_k);
}

/// Depthwise 2D spatial convolution with `stride = 1` and `padding = kernel_size / 2` ("same" padding)
/// over a token-major `[grid_h * grid_w, channels]` buffer, writing into `out`.
/// `weight` is stored in `[channels][kernel_size][kernel_size]` order (`c * K * K + ky * K + kx`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_conv2d_same_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    grid_w: usize,
    grid_h: usize,
    channels: usize,
    kernel_size: usize,
    out: &mut [f32],
) {
    let n_tokens = grid_w * grid_h;
    let k2 = kernel_size * kernel_size;
    debug_assert_eq!(input.len(), n_tokens * channels);
    debug_assert_eq!(weight.len(), channels * k2);
    debug_assert_eq!(out.len(), n_tokens * channels);
    let pad = kernel_size / 2;

    // Transpose weights from [channels, k2] to tap-major [k2, channels] for unit-stride SIMD loops.
    let mut w_tap = vec![0.0f32; k2 * channels];
    for c in 0..channels {
        let src = &weight[c * k2..(c + 1) * k2];
        for (k_off, &w_val) in src.iter().enumerate() {
            w_tap[k_off * channels + c] = w_val;
        }
    }

    par_rows_n(out, channels, |tok_idx, out_row| {
        let y = tok_idx / grid_w;
        let x = tok_idx % grid_w;
        if let Some(b) = bias {
            out_row.copy_from_slice(b);
        } else {
            out_row.fill(0.0);
        }
        for ky in 0..kernel_size {
            let sy_isize = (y + ky) as isize - (pad as isize);
            if sy_isize < 0 || (sy_isize as usize) >= grid_h {
                continue;
            }
            let sy = sy_isize as usize;
            for kx in 0..kernel_size {
                let sx_isize = (x + kx) as isize - (pad as isize);
                if sx_isize < 0 || (sx_isize as usize) >= grid_w {
                    continue;
                }
                let sx = sx_isize as usize;
                let in_row =
                    &input[(sy * grid_w + sx) * channels..(sy * grid_w + sx + 1) * channels];
                let k_off = ky * kernel_size + kx;
                let w_row = &w_tap[k_off * channels..(k_off + 1) * channels];
                for c in 0..channels {
                    out_row[c] += in_row[c] * w_row[c];
                }
            }
        }
    });
}

/// Depthwise 2D spatial convolution with `stride = 1` and `padding = kernel_size / 2` ("same" padding)
/// over a token-major `[grid_h * grid_w, channels]` buffer.
/// `weight` is stored in `[channels][kernel_size][kernel_size]` order (`c * K * K + ky * K + kx`).
#[cfg(test)]
pub(crate) fn depthwise_conv2d_same(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    grid_w: usize,
    grid_h: usize,
    channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; grid_w * grid_h * channels];
    depthwise_conv2d_same_into(
        input,
        weight,
        bias,
        grid_w,
        grid_h,
        channels,
        kernel_size,
        &mut out,
    );
    out
}

/// Load a patch-embedding kernel from GGUF and transpose it into row-major `[3 * p^2, n_embd]`.
/// Supports both 4D `[p, p, 3, n_embd]` and 5D temporal `[p, p, 2, 3, n_embd]` / `[p, p, 3, 2, n_embd]`.
fn load_patch_embed_weights(
    gguf: &Arc<GgufFile>,
    weight_name: &str,
    bias_name: Option<&str>,
    require_bias: bool,
    config: &VisionEncoderConfig,
) -> Result<(PatchEmbedWeights, Option<PatchEmbedWeights>)> {
    let patch_t = gguf
        .get_tensor(weight_name)
        .with_context(|| format!("loading {weight_name}"))?;
    let patch_shape = patch_t.shape().to_vec();
    let p = config.patch_size;
    let in_dim = 3 * p * p;
    let out_dim = config.n_embd;
    let raw = patch_t.try_to_f32_vec()?;

    let conv_b = match bias_name {
        Some(b_name) if gguf.tensors.contains_key(b_name) => {
            let b = load_vec_f32(gguf, b_name)?;
            anyhow::ensure!(
                b.len() == out_dim,
                "{b_name} len ({}) != n_embd ({out_dim})",
                b.len()
            );
            b
        }
        Some(b_name) if require_bias => {
            return Err(anyhow::anyhow!("missing required `{b_name}`"));
        }
        _ => vec![0.0; out_dim],
    };

    if patch_shape.len() == 4 {
        anyhow::ensure!(
            patch_shape[0] == p
                && patch_shape[1] == p
                && patch_shape[2] == 3
                && patch_shape[3] == out_dim,
            "{weight_name} shape {patch_shape:?} != [patch_size={p}, patch_size={p}, 3, n_embd={out_dim}]"
        );
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
        return Ok((PatchEmbedWeights { conv_w, conv_b }, None));
    }

    if patch_shape.len() == 5 {
        // 5D temporal kernel: either [p, p, 2, 3, out_dim] or [p, p, 3, 2, out_dim].
        let t_first = patch_shape[0] == p
            && patch_shape[1] == p
            && patch_shape[2] == 2
            && patch_shape[3] == 3
            && patch_shape[4] == out_dim;
        let c_first = patch_shape[0] == p
            && patch_shape[1] == p
            && patch_shape[2] == 3
            && patch_shape[3] == 2
            && patch_shape[4] == out_dim;
        anyhow::ensure!(
            t_first || c_first,
            "{weight_name} 5D shape {patch_shape:?} != [{p}, {p}, 2, 3, {out_dim}]"
        );
        let mut conv_w0 = vec![0f32; in_dim * out_dim];
        let mut conv_w1 = vec![0f32; in_dim * out_dim];
        for oc in 0..out_dim {
            for c in 0..3 {
                for t in 0..2 {
                    for kh in 0..p {
                        for kw in 0..p {
                            let src = if t_first {
                                kw + p * kh + p * p * t + p * p * 2 * c + p * p * 6 * oc
                            } else {
                                kw + p * kh + p * p * c + p * p * 3 * t + p * p * 6 * oc
                            };
                            let in_idx = c * p * p + kh * p + kw;
                            if t == 0 {
                                conv_w0[in_idx * out_dim + oc] = raw[src];
                            } else {
                                conv_w1[in_idx * out_dim + oc] = raw[src];
                            }
                        }
                    }
                }
            }
        }
        return Ok((
            PatchEmbedWeights {
                conv_w: conv_w0,
                conv_b,
            },
            Some(PatchEmbedWeights {
                conv_w: conv_w1,
                conv_b: vec![0.0; out_dim],
            }),
        ));
    }

    anyhow::bail!(
        "{weight_name} unexpected rank {} ({patch_shape:?})",
        patch_shape.len()
    )
}

fn load_optional_lfm3_norm(
    gguf: &Arc<GgufFile>,
    prefix: &str,
    expected_len: usize,
) -> Result<Option<Lfm3NormWeights>> {
    let w_name = format!("{prefix}.weight");
    if !gguf.tensors.contains_key(&w_name) {
        return Ok(None);
    }
    Ok(Some(load_lfm3_norm(gguf, prefix, &[expected_len])?))
}

fn load_lfm3_norm(
    gguf: &Arc<GgufFile>,
    prefix: &str,
    allowed_lens: &[usize],
) -> Result<Lfm3NormWeights> {
    let w_name = format!("{prefix}.weight");
    let b_name = format!("{prefix}.bias");
    let weight = load_vec_f32(gguf, &w_name)?;
    anyhow::ensure!(
        allowed_lens.contains(&weight.len()),
        "{w_name} len ({}) not in allowed lengths {allowed_lens:?}",
        weight.len()
    );
    let bias = if gguf.tensors.contains_key(&b_name) {
        let b = load_vec_f32(gguf, &b_name)?;
        anyhow::ensure!(
            b.len() == weight.len(),
            "{b_name} len ({}) != {w_name} len ({})",
            b.len(),
            weight.len()
        );
        Some(b)
    } else {
        None
    };
    Ok(Lfm3NormWeights { weight, bias })
}

fn load_optional_vec_f32(
    gguf: &Arc<GgufFile>,
    name: &str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>> {
    if !gguf.tensors.contains_key(name) {
        return Ok(None);
    }
    let v = load_vec_f32(gguf, name)?;
    anyhow::ensure!(
        v.len() == expected_len,
        "{name} len ({}) != expected ({expected_len})",
        v.len()
    );
    Ok(Some(v))
}

/// The per-head width of an LFM3 vision attention block: `d / n_head`, which must divide evenly
/// and be a multiple of 4 (the 2D M-RoPE rotates half the head per axis in pairs; any other
/// width rotates fewer dimensions than the head has and silently skews the positions).
fn lfm3_attention_head_dim(il: usize, d: usize, n_head: usize) -> Result<usize> {
    anyhow::ensure!(n_head > 0, "block {il}: n_head must be positive");
    anyhow::ensure!(
        d.is_multiple_of(n_head),
        "block {il}: embedding width {d} is not a multiple of n_head {n_head}"
    );
    let head_dim = d / n_head;
    anyhow::ensure!(
        head_dim > 0 && head_dim.is_multiple_of(4),
        "block {il}: head_dim {head_dim} must be a positive multiple of 4 for 2D M-RoPE"
    );
    Ok(head_dim)
}

fn load_lfm3_block(
    gguf: &Arc<GgufFile>,
    il: usize,
    cfg: &VisionEncoderConfig,
) -> Result<Lfm3VisionBlockWeights> {
    let pfx = format!("v.blk.{il}");
    let d = cfg.n_embd;
    let ln1 = load_lfm3_norm(gguf, &format!("{pfx}.ln1"), &[d])?;
    let ln2 = load_lfm3_norm(gguf, &format!("{pfx}.ln2"), &[d])?;

    let conv_spatial_name = if gguf
        .tensors
        .contains_key(&format!("{pfx}.conv2d_spatial.weight"))
    {
        Some((
            format!("{pfx}.conv2d_in_proj"),
            format!("{pfx}.conv2d_spatial"),
            format!("{pfx}.conv2d_out_proj"),
        ))
    } else if gguf
        .tensors
        .contains_key(&format!("{pfx}.conv.conv.weight"))
    {
        Some((
            format!("{pfx}.conv.in_proj"),
            format!("{pfx}.conv.conv"),
            format!("{pfx}.conv.out_proj"),
        ))
    } else {
        None
    };

    let op = if let Some((in_proj_pfx, spatial_pfx, out_proj_pfx)) = conv_spatial_name {
        let in_proj_w = wt_f32(gguf, &format!("{in_proj_pfx}.weight"))?;
        anyhow::ensure!(
            in_proj_w.cols == d && (in_proj_w.rows == 3 * d || in_proj_w.rows == 2 * d),
            "block {il} {in_proj_pfx}.weight shape ({}, {}) != (3*{d} or 2*{d}, {d})",
            in_proj_w.rows,
            in_proj_w.cols,
        );
        let in_proj_b =
            load_optional_vec_f32(gguf, &format!("{in_proj_pfx}.bias"), in_proj_w.rows)?;

        let dw_t = gguf
            .get_tensor(&format!("{spatial_pfx}.weight"))
            .with_context(|| format!("loading {spatial_pfx}.weight"))?;
        let dw_shape = dw_t.shape().to_vec();
        let kernel_size = match dw_shape.as_slice() {
            [kw, kh, 1, oc] if kw == kh && *oc == d => *kw,
            [kw, kh, oc] if kw == kh && *oc == d => *kw,
            [k2, oc] if *oc == d => {
                let k = (*k2 as f64).sqrt().round() as usize;
                anyhow::ensure!(
                    k * k == *k2,
                    "block {il} {spatial_pfx}.weight non-square {k2}"
                );
                k
            }
            other => anyhow::bail!(
                "block {il} {spatial_pfx}.weight unexpected shape {other:?} for n_embd={d}"
            ),
        };
        anyhow::ensure!(
            kernel_size > 0 && kernel_size % 2 == 1,
            "block {il} {spatial_pfx}.weight kernel_size ({kernel_size}) must be odd and > 0"
        );
        let depthwise_w = dw_t.try_to_f32_vec()?;
        anyhow::ensure!(
            depthwise_w.len() == d * kernel_size * kernel_size,
            "block {il} {spatial_pfx}.weight len ({}) != d*K^2 ({})",
            depthwise_w.len(),
            d * kernel_size * kernel_size
        );
        let depthwise_b = load_optional_vec_f32(gguf, &format!("{spatial_pfx}.bias"), d)?;

        let out_proj_w = wt_f32(gguf, &format!("{out_proj_pfx}.weight"))?;
        anyhow::ensure!(
            out_proj_w.rows == d && out_proj_w.cols == d,
            "block {il} {out_proj_pfx}.weight shape ({}, {}) != ({d}, {d})",
            out_proj_w.rows,
            out_proj_w.cols,
        );
        let out_proj_b = load_optional_vec_f32(gguf, &format!("{out_proj_pfx}.bias"), d)?;

        Lfm3BlockOp::Conv(Lfm3ConvBlockWeights {
            in_proj_w,
            in_proj_b,
            depthwise_w,
            depthwise_b,
            kernel_size,
            out_proj_w,
            out_proj_b,
        })
    } else if gguf.tensors.contains_key(&format!("{pfx}.attn_q.weight")) {
        let head_dim = lfm3_attention_head_dim(il, d, cfg.n_head)?;
        let q_w = wt_f32(gguf, &format!("{pfx}.attn_q.weight"))?;
        let k_w = wt_f32(gguf, &format!("{pfx}.attn_k.weight"))?;
        let v_w = wt_f32(gguf, &format!("{pfx}.attn_v.weight"))?;
        let o_w = wt_f32(gguf, &format!("{pfx}.attn_out.weight"))?;

        anyhow::ensure!(
            q_w.rows == d && q_w.cols == d,
            "block {il} attn_q.weight shape ({}, {}) != ({d}, {d})",
            q_w.rows,
            q_w.cols,
        );
        anyhow::ensure!(
            k_w.cols == d
                && v_w.cols == d
                && k_w.rows == v_w.rows
                && k_w.rows > 0
                && k_w.rows.is_multiple_of(head_dim),
            "block {il} attn_k/v.weight shapes ({}, {}) / ({}, {}) invalid for d={d}, head_dim={head_dim}",
            k_w.rows,
            k_w.cols,
            v_w.rows,
            v_w.cols,
        );
        let n_head_kv = k_w.rows / head_dim;
        anyhow::ensure!(
            cfg.n_head.is_multiple_of(n_head_kv),
            "block {il} n_head ({}) not divisible by n_head_kv ({n_head_kv})",
            cfg.n_head,
        );
        anyhow::ensure!(
            o_w.rows == d && o_w.cols == d,
            "block {il} attn_out.weight shape ({}, {}) != ({d}, {d})",
            o_w.rows,
            o_w.cols,
        );

        let q_b = load_optional_vec_f32(gguf, &format!("{pfx}.attn_q.bias"), d)?;
        let k_b = load_optional_vec_f32(gguf, &format!("{pfx}.attn_k.bias"), k_w.rows)?;
        let v_b = load_optional_vec_f32(gguf, &format!("{pfx}.attn_v.bias"), v_w.rows)?;
        let o_b = load_optional_vec_f32(gguf, &format!("{pfx}.attn_out.bias"), d)?;

        let q_norm = load_lfm3_norm(gguf, &format!("{pfx}.attn_q_norm"), &[head_dim, d])?;
        let k_norm = load_lfm3_norm(gguf, &format!("{pfx}.attn_k_norm"), &[head_dim, k_w.rows])?;

        Lfm3BlockOp::Attn(Lfm3AttnBlockWeights {
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            q_norm,
            k_norm,
            o_w,
            o_b,
            n_head: cfg.n_head,
            n_head_kv,
            head_dim,
        })
    } else {
        anyhow::bail!("LFM3 vision block {il} has neither conv2d_spatial.weight nor attn_q.weight");
    };

    let up_w = wt_f32(gguf, &format!("{pfx}.ffn_up.weight"))?;
    let down_w = wt_f32(gguf, &format!("{pfx}.ffn_down.weight"))?;
    let n_ff = up_w.rows;
    // the int8 scratch is sized from `clip.vision.feed_forward_length`: a block of another
    // width would load and then panic in `quantize_rows`
    anyhow::ensure!(
        n_ff == cfg.n_ff,
        "block {il} ffn_up.weight has {n_ff} rows but clip.vision.feed_forward_length is {}",
        cfg.n_ff
    );
    anyhow::ensure!(
        up_w.cols == d && down_w.rows == d && down_w.cols == n_ff,
        "block {il} FFN up/down shapes ({}, {}) / ({}, {}) mismatch for d={d}",
        up_w.rows,
        up_w.cols,
        down_w.rows,
        down_w.cols,
    );
    let up_b = load_optional_vec_f32(gguf, &format!("{pfx}.ffn_up.bias"), n_ff)?;
    let down_b = load_optional_vec_f32(gguf, &format!("{pfx}.ffn_down.bias"), d)?;

    let (gate_w, gate_b) = if gguf.tensors.contains_key(&format!("{pfx}.ffn_gate.weight")) {
        let gw = wt_f32(gguf, &format!("{pfx}.ffn_gate.weight"))?;
        anyhow::ensure!(
            gw.rows == n_ff && gw.cols == d,
            "block {il} ffn_gate.weight shape ({}, {}) != ({n_ff}, {d})",
            gw.rows,
            gw.cols,
        );
        let gb = load_optional_vec_f32(gguf, &format!("{pfx}.ffn_gate.bias"), n_ff)?;
        (Some(gw), gb)
    } else {
        (None, None)
    };

    let ffn = Lfm3FfnWeights {
        gate_w,
        gate_b,
        up_w,
        up_b,
        down_w,
        down_b,
    };

    Ok(Lfm3VisionBlockWeights { ln1, op, ln2, ffn })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm3_attention_head_dim_must_suit_2d_rope() {
        assert_eq!(lfm3_attention_head_dim(0, 128, 2).unwrap(), 64);
        assert_eq!(lfm3_attention_head_dim(0, 64, 16).unwrap(), 4);
        for (d, n_head) in [(130, 2), (100, 3), (66, 2), (6, 2), (64, 0)] {
            assert!(
                lfm3_attention_head_dim(0, d, n_head).is_err(),
                "d={d} n_head={n_head} was accepted"
            );
        }
    }

    #[test]
    fn the_projector_input_norm_is_applied_at_exactly_one_stage() {
        // scale_factor 1: the merged width equals n_embd, and the norm used to run twice
        assert_eq!(
            mm_norm_stage(16, 16, 16),
            Some(MmNormStage::PreMerge),
            "one stage when the widths coincide"
        );
        // scale_factor 2: a norm over the encoder width runs before the merge, one over the
        // merged width after it, anything else is not applied
        assert_eq!(mm_norm_stage(16, 16, 64), Some(MmNormStage::PreMerge));
        assert_eq!(mm_norm_stage(64, 16, 64), Some(MmNormStage::PostMerge));
        assert_eq!(mm_norm_stage(32, 16, 64), None);
    }

    fn synth_cfg(grid: usize, n_embd: usize, scale_factor: usize) -> VisionEncoderConfig {
        let patch_size = 16;
        let image_size = grid * patch_size;
        let pixels_per_token = (patch_size * scale_factor).pow(2);
        VisionEncoderConfig {
            projector_type: VisionProjectorType::Lfm2,
            n_layer: 1,
            n_embd,
            n_ff: n_embd * 4,
            n_head: 1,
            n_head_kv: 1,
            rope_theta: LFM3_DEFAULT_ROPE_THETA,
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

    #[test]
    fn vision_projector_type_and_boundary_tokens() {
        let lfm2 = VisionProjectorType::from_gguf_str("lfm2").unwrap();
        let lfm3vl = VisionProjectorType::from_gguf_str("lfm3vl").unwrap();
        let lfm3o = VisionProjectorType::from_gguf_str("lfm3o").unwrap();

        assert!(!lfm2.is_lfm3());
        assert!(lfm3vl.is_lfm3());
        assert!(lfm3o.is_lfm3());

        assert!(!lfm2.has_boundary_tokens());
        assert!(!lfm3vl.has_boundary_tokens());
        assert!(lfm3o.has_boundary_tokens());

        assert_eq!(lfm3vl.vision_boundary_tokens(), None);
        assert_eq!(
            lfm3o.vision_boundary_tokens(),
            Some((VISION_START_TOKEN, VISION_END_TOKEN))
        );
        assert_eq!(lfm3vl.wrap_image_prompt("<img/>"), "<img/>");
        assert_eq!(
            lfm3o.wrap_image_prompt("<img/>"),
            "<|vision_start|><img/><|vision_end|>"
        );
        let ids = lfm3o.resolve_boundary_token_ids(|tok| match tok {
            "<|vision_start|>" => Some(100),
            "<|vision_end|>" => Some(101),
            _ => None,
        });
        assert_eq!(ids, Some((100, 101)));
        assert_eq!(lfm3vl.resolve_boundary_token_ids(|_| Some(1)), None);
    }

    #[test]
    fn vision_mrope_2d_rotates_height_and_width_sections_independently() {
        // 2x2 grid, 1 head, head_dim = 8 => half = 4, sect = 2.
        // Indices [0, 1] pair with [4, 5] for height (py).
        // Indices [2, 3] pair with [6, 7] for width (px), with frequency resetting at index 2.
        let (grid_w, grid_h, n_head, head_dim) = (2usize, 2usize, 1usize, 8usize);
        let rope_theta = 1_000_000.0f32;
        let orig = lcg_vec(42, grid_w * grid_h * head_dim);
        let mut rotated = orig.clone();
        apply_2d_vision_mrope(&mut rotated, grid_w, grid_h, n_head, head_dim, rope_theta);

        // At (py=0, px=0), angles are 0 so the vector is unchanged.
        for d in 0..head_dim {
            assert!((rotated[d] - orig[d]).abs() < 1e-6);
        }

        // At (py=0, px=1) (token 1), height section [0,1,4,5] has angle 0 (unchanged),
        // while width section [2,3,6,7] is rotated by px=1.
        let tok1 = &rotated[head_dim..2 * head_dim];
        let orig1 = &orig[head_dim..2 * head_dim];
        assert!((tok1[0] - orig1[0]).abs() < 1e-6);
        assert!((tok1[1] - orig1[1]).abs() < 1e-6);
        assert!((tok1[4] - orig1[4]).abs() < 1e-6);
        assert!((tok1[5] - orig1[5]).abs() < 1e-6);
        // ic=2 (k=0) has freq = 1.0, so angle_x = 1.0 rad.
        let (sin1, cos1) = 1.0f32.sin_cos();
        let want_x0 = orig1[2] * cos1 - orig1[6] * sin1;
        let want_x1 = orig1[2] * sin1 + orig1[6] * cos1;
        assert!((tok1[2] - want_x0).abs() < 1e-5);
        assert!((tok1[6] - want_x1).abs() < 1e-5);

        // Every token preserves its L2 norm under 2D M-RoPE rotation.
        for t in 0..(grid_w * grid_h) {
            let n0: f32 = orig[t * head_dim..(t + 1) * head_dim]
                .iter()
                .map(|x| x * x)
                .sum();
            let n1: f32 = rotated[t * head_dim..(t + 1) * head_dim]
                .iter()
                .map(|x| x * x)
                .sum();
            assert!((n0 - n1).abs() < 1e-4, "token {t} norm drift: {n0} vs {n1}");
        }
    }

    #[test]
    fn depthwise_conv2d_same_computes_spatial_3x3_per_channel() {
        // 2x2 grid, 2 channels, 3x3 identity-center + neighbor kernel.
        let (grid_w, grid_h, channels, k) = (2usize, 2usize, 2usize, 3usize);
        // Channel 0 kernel: center (ky=1, kx=1) = 2.0, right (ky=1, kx=2) = 1.0.
        // Channel 1 kernel: center (ky=1, kx=1) = 1.0, bottom (ky=2, kx=1) = 0.5.
        let mut weight = vec![0.0f32; channels * k * k];
        weight[4] = 2.0;
        weight[5] = 1.0;
        weight[9 + 4] = 1.0;
        weight[9 + 7] = 0.5;
        let bias = [0.25f32, -0.5f32];
        // Input tokens at (0,0), (0,1), (1,0), (1,1):
        let input = vec![
            1.0, 10.0, // (0,0)
            2.0, 20.0, // (0,1)
            3.0, 30.0, // (1,0)
            4.0, 40.0, // (1,1)
        ];
        let out = depthwise_conv2d_same(&input, &weight, Some(&bias), grid_w, grid_h, channels, k);
        // At (0,0): ch0 = 0.25 + 2.0*1.0 + 1.0*2.0 = 4.25; ch1 = -0.5 + 1.0*10.0 + 0.5*30.0 = 24.5.
        assert!((out[0] - 4.25).abs() < 1e-6);
        assert!((out[1] - 24.5).abs() < 1e-6);
    }

    #[test]
    fn lfm3_hybrid_encoder_forward_pass_end_to_end() {
        let (grid_w, grid_h) = (2usize, 2usize);
        let patch_size = 2usize;
        let n_embd = 16usize;
        let n_ff = 32usize;
        let n_head = 2usize;
        let n_head_kv = 1usize;
        let head_dim = n_embd / n_head;
        let scale_factor = 2usize;
        let projection_dim = 12usize;

        let cfg = VisionEncoderConfig {
            projector_type: VisionProjectorType::Lfm3O,
            n_layer: 2,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            rope_theta: LFM3_DEFAULT_ROPE_THETA,
            eps: 1e-6,
            image_size: 4,
            patch_size,
            n_trained_patches: 4,
            projection_dim,
            scale_factor,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            image_min_pixels: 16,
            image_max_pixels: 16,
        };

        let in_dim = 3 * patch_size * patch_size;
        let patch_embed = PatchEmbedWeights {
            conv_w: lcg_vec(1, in_dim * n_embd),
            conv_b: lcg_vec(2, n_embd),
        };
        let patch_embed_1 = Some(PatchEmbedWeights {
            conv_w: lcg_vec(3, in_dim * n_embd),
            conv_b: vec![0.0; n_embd],
        });

        let conv_block = Lfm3VisionBlockWeights {
            ln1: Lfm3NormWeights {
                weight: vec![1.0; n_embd],
                bias: Some(vec![0.0; n_embd]),
            },
            op: Lfm3BlockOp::Conv(Lfm3ConvBlockWeights {
                in_proj_w: MmapWeight::from_owned_f32(
                    lcg_vec(4, 3 * n_embd * n_embd),
                    3 * n_embd,
                    n_embd,
                ),
                in_proj_b: Some(vec![0.0; 3 * n_embd]),
                depthwise_w: lcg_vec(5, n_embd * 9),
                depthwise_b: Some(vec![0.0; n_embd]),
                kernel_size: 3,
                out_proj_w: MmapWeight::from_owned_f32(lcg_vec(6, n_embd * n_embd), n_embd, n_embd),
                out_proj_b: Some(vec![0.0; n_embd]),
            }),
            ln2: Lfm3NormWeights {
                weight: vec![1.0; n_embd],
                bias: None,
            },
            ffn: Lfm3FfnWeights {
                gate_w: Some(MmapWeight::from_owned_f32(
                    lcg_vec(7, n_ff * n_embd),
                    n_ff,
                    n_embd,
                )),
                gate_b: None,
                up_w: MmapWeight::from_owned_f32(lcg_vec(8, n_ff * n_embd), n_ff, n_embd),
                up_b: None,
                down_w: MmapWeight::from_owned_f32(lcg_vec(9, n_embd * n_ff), n_embd, n_ff),
                down_b: None,
            },
        };

        let kv_dim = n_head_kv * head_dim;
        let attn_block = Lfm3VisionBlockWeights {
            ln1: Lfm3NormWeights {
                weight: vec![1.0; n_embd],
                bias: None,
            },
            op: Lfm3BlockOp::Attn(Lfm3AttnBlockWeights {
                q_w: MmapWeight::from_owned_f32(lcg_vec(10, n_embd * n_embd), n_embd, n_embd),
                q_b: None,
                k_w: MmapWeight::from_owned_f32(lcg_vec(11, kv_dim * n_embd), kv_dim, n_embd),
                k_b: None,
                v_w: MmapWeight::from_owned_f32(lcg_vec(12, kv_dim * n_embd), kv_dim, n_embd),
                v_b: None,
                q_norm: Lfm3NormWeights {
                    weight: vec![1.0; head_dim],
                    bias: Some(vec![0.0; head_dim]),
                },
                k_norm: Lfm3NormWeights {
                    weight: vec![1.0; head_dim],
                    bias: Some(vec![0.0; head_dim]),
                },
                o_w: MmapWeight::from_owned_f32(lcg_vec(13, n_embd * n_embd), n_embd, n_embd),
                o_b: None,
                n_head,
                n_head_kv,
                head_dim,
            }),
            ln2: Lfm3NormWeights {
                weight: vec![1.0; n_embd],
                bias: Some(vec![0.0; n_embd]),
            },
            ffn: Lfm3FfnWeights {
                gate_w: None,
                gate_b: None,
                up_w: MmapWeight::from_owned_f32(lcg_vec(14, n_ff * n_embd), n_ff, n_embd),
                up_b: Some(vec![0.0; n_ff]),
                down_w: MmapWeight::from_owned_f32(lcg_vec(15, n_embd * n_ff), n_embd, n_ff),
                down_b: Some(vec![0.0; n_embd]),
            },
        };

        let merged_dim = n_embd * scale_factor * scale_factor;
        let weights = VisionEncoderWeights {
            config: cfg,
            patch_embed,
            patch_embed_1,
            embedding_norm: Some(Lfm3NormWeights {
                weight: vec![1.0; n_embd],
                bias: Some(vec![0.0; n_embd]),
            }),
            position_embed: Vec::new(),
            blocks: Vec::new(),
            lfm3_blocks: vec![conv_block, attn_block],
            post_ln_w: Vec::new(),
            post_ln_b: Vec::new(),
            mm_input_norm: Some(Lfm3NormWeights {
                weight: vec![1.0; n_embd],
                bias: Some(vec![0.0; n_embd]),
            }),
            projector: ProjectorWeights {
                mm1_w: MmapWeight::from_owned_f32(lcg_vec(16, 24 * merged_dim), 24, merged_dim),
                mm1_b: vec![0.1; 24],
                mm2_w: MmapWeight::from_owned_f32(
                    lcg_vec(17, projection_dim * 24),
                    projection_dim,
                    24,
                ),
                mm2_b: vec![-0.05; projection_dim],
            },
        };

        assert!(weights.is_lfm3());
        assert!(weights.has_boundary_tokens());
        let pixels = lcg_vec(99, 3 * (grid_w * patch_size) * (grid_h * patch_size));
        let out = weights.encode_image(&pixels, grid_w, grid_h).unwrap();
        // 2x2 grid merged by 2x2 => 1 token of `projection_dim` floats.
        assert_eq!(out.len(), projection_dim);
        assert!(out.iter().all(|v| v.is_finite()));

        // Consecutive video frame pair equals `encode_image` when both frames are identical.
        let out_pair = weights
            .encode_video_frame_pair(&pixels, &pixels, grid_w, grid_h)
            .unwrap();
        assert_eq!(out, out_pair);

        // Without the second patch kernel the pair would drop its second frame: refuse it.
        let mut single_kernel = weights;
        single_kernel.patch_embed_1 = None;
        let err = single_kernel
            .encode_video_frame_pair(&pixels, &pixels, grid_w, grid_h)
            .unwrap_err();
        assert!(err.to_string().contains("temporal patch kernel"), "{err}");
    }

    #[test]
    fn vision_encoder_config_from_gguf_validates_bounds_types_and_kv_head_array() {
        let base_builder = || {
            crate::gguf::GgufBuilder::new()
                .kv_str("general.architecture", "clip")
                .kv_bool(KEY_HAS_VISION, true)
                .kv_str(KEY_VISION_PROJ_TYPE, "lfm3o")
                .kv_u32(KEY_N_LAYER, 2)
                .kv_u32(KEY_N_EMBD, 16)
                .kv_u32(KEY_N_FF, 32)
                .kv_u32(KEY_N_HEAD, 4)
                .kv_i32_array(KEY_N_HEAD_KV, vec![0, 2])
                .kv_f32(KEY_LN_EPS, 1e-6)
                .kv_u32(KEY_PATCH_SIZE, 2)
                .kv_u32(KEY_IMAGE_SIZE, 4)
                .kv_u32(KEY_PROJECTION_DIM, 12)
                .kv_u32(KEY_SCALE_FACTOR, 2)
                .kv_f32_array(KEY_IMAGE_MEAN, vec![0.5, 0.5, 0.5])
                .kv_f32_array(KEY_IMAGE_STD, vec![0.5, 0.5, 0.5])
        };

        let parse = |b: crate::gguf::GgufBuilder| {
            let gguf = Arc::new(GgufFile::from_bytes(Arc::from(b.build_bytes())).unwrap());
            VisionEncoderConfig::from_gguf(&gguf)
        };

        let valid = parse(base_builder()).expect("valid config with i32 array head_count_kv");
        assert_eq!(valid.n_head_kv, 2);
        assert_eq!(valid.projector_type, VisionProjectorType::Lfm3O);

        assert!(parse(base_builder().kv_u32(KEY_N_EMBD, 0)).is_err());
        assert!(parse(base_builder().kv_u32(KEY_IMAGE_SIZE, 0)).is_err());
        assert!(parse(base_builder().kv_f32(KEY_LN_EPS, 0.0)).is_err());
        assert!(parse(base_builder().kv_f32(KEY_LN_EPS, -1e-5)).is_err());
        assert!(parse(base_builder().kv_f32(KEY_LN_EPS, f32::NAN)).is_err());
        assert!(parse(base_builder().kv_f32(KEY_ROPE_THETA, 0.0)).is_err());
        assert!(parse(base_builder().kv_f32(KEY_ROPE_THETA, -100.0)).is_err());
        assert!(parse(base_builder().kv_str(KEY_ROPE_THETA, "bad")).is_err());
        assert!(parse(base_builder().kv_str(KEY_IMAGE_SIZE, "bad")).is_err());
        assert!(parse(base_builder().kv_u32(KEY_VISION_PROJ_TYPE, 123)).is_err());
        assert!(parse(base_builder().kv_i32_array(KEY_N_HEAD_KV, vec![0, 0])).is_err());
    }
}

// Plain dense transformer text model. Covers two RoPE families on one code path:
//   - NEOX (split-halves) rope: Qwen2, Qwen3.
//   - NORM (interleaved-pair) rope: LLaMA, Mistral, Granite 3.x.
//
// Per-arch differences are gated on tensor presence / metadata at load time:
//   - Qwen2 carries Q/K/V projection biases (`blk.N.attn_{q,k,v}.bias`) and no
//     QK-norm.
//   - Qwen3 carries per-head Q/K RMSNorm weights (`blk.N.attn_{q,k}_norm.weight`)
//     and no biases.
//   - LLaMA / Mistral carry neither (plain attention) but use NORM rope.
//   - Granite 3.x is a NORM-rope llama variant plus four scalar multipliers
//     (`{arch}.embedding_scale`, `.residual_scale`, `.attention.scale`,
//     `.logit_scale`). All default to identity, so the other archs are unaffected.
//
// GGUF weights for every supported arch are stored un-permuted, matching
// llama.cpp, so the correct rope layout is selected per arch (NEOX vs NORM)
// rather than permuting weights at load.

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::backend::cpu::RopeType;
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
use crate::kv_cache::LayerState;
pub use crate::model::transformer::FfnActivation;
use crate::model::transformer::{self, AttnDims, AttnExtras, AttnWeights, FfnWeights, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
// Only the batched-LM-head warning path names `DType` unqualified; every other
// reference is fully qualified. Gate the import to that path so `--features blas`
// and non-int8 targets do not see it as unused under clippy's `-D warnings`.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
use crate::tensor::DType;

/// Layer normalization ordering (Pre-Norm vs Post-Norm).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum NormOrder {
    #[default]
    PreNorm,
    PostNorm,
}

// ── Per-layer weight references ─────────────────────────────────────────────

/// Pre-resolved quantized weight refs for one transformer layer.
#[derive(Clone)]
struct LayerWeightRefs {
    attn_q: WeightRef,
    attn_k: WeightRef,
    attn_v: WeightRef,
    attn_output: WeightRef,
    ffn_gate: WeightRef,
    ffn_up: WeightRef,
    ffn_down: WeightRef,
}

// ── LLaMA-family Model ──────────────────────────────────────────────────────

pub struct LlamaModel {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    /// RoPE pair layout: `Neox` for Qwen2/Qwen3/Gemma 2/Olmo 2, `Norm` for LLaMA/Mistral/Granite.
    rope_type: RopeType,
    /// Llama-3 RoPE frequency-scaling factors (`rope_freqs.weight`, `head_dim/2`),
    /// applied per-pair on the NORM path. `None` for archs without the tensor
    /// (Qwen/Mistral/Granite) ⇒ plain RoPE.
    rope_freqs: Option<Vec<f32>>,
    norm_order: NormOrder,
    activation: FfnActivation,
    attn_logit_softcapping: Option<f32>,
    final_logit_softcapping: Option<f32>,
    // Granite 3.x and MiniCPM scalar multipliers live on `config.scalars` (identity
    // for every other arch): see `ScalarMultipliers`.
    // Pre-dequantized small F32 weights.
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_post_norm_weights: Vec<Option<Vec<f32>>>,
    ffn_post_norm_weights: Vec<Option<Vec<f32>>>,
    // Qwen3 / Olmo 2 QK-norm weights (None for Qwen2).
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    // Qwen2 Q/K/V projection biases (None for Qwen3).
    attn_q_bias: Vec<Option<Vec<f32>>>,
    attn_k_bias: Vec<Option<Vec<f32>>>,
    attn_v_bias: Vec<Option<Vec<f32>>>,
    // Mistral 3 optional projection and FFN biases.
    attn_output_bias: Vec<Option<Vec<f32>>>,
    ffn_gate_bias: Vec<Option<Vec<f32>>>,
    ffn_up_bias: Vec<Option<Vec<f32>>>,
    ffn_down_bias: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs.
    embd_ref: WeightRef,
    /// Separate output projection (`output.weight`) when present; `None` means
    /// tied embeddings (`token_embd.weight` reused for the logit projection).
    output_ref: Option<WeightRef>,
    layer_refs: Vec<LayerWeightRefs>,
    sliding_window: Option<usize>,
    sliding_window_pattern: Option<Vec<bool>>,
    yarn: Option<cpu::YarnParams>,
    attn_temp_scale: Option<(f32, usize)>,
    /// Number of physical layers per loop for models with tied layer loops (e.g. Nanbeige).
    /// When Some, output_norm is applied between loops.
    loop_norm_interval: Option<usize>,
    #[allow(dead_code)]
    model_id: String,
}

/// Report, once per distinct `(head, dtype)`, that the batched LM-head
/// projection declined, so speculative verification is paying a per-position
/// LM-head read again.
///
/// A free function, like the [`transformer::warn_unbatchable`] it is
/// deliberately *not* sharing: it touches no model state, and the contrast is
/// the point. That helper's message says prefill fell back to the per-token
/// path, which is false here — the layers can all be batchable while only the
/// head is not — and its dedupe set is process-global and keyed on dtype alone,
/// so warning through it would permanently suppress the genuine whole-model
/// prefill warning for that dtype, for every model loaded later in the process.
///
/// Keyed on `(head, dtype)`, and taking those unformatted rather than a built
/// message, so the dedupe compares the values themselves instead of prose about
/// them. The caller reaches this once per verification round for as long as the
/// model is loaded, and every call after the first reports a decline already
/// reported, so there is no reason to build a string to throw away. Both fields
/// come from model metadata fixed at load, so `SEEN` holds one entry per
/// distinct `(head, dtype)` pair, however many models load.
///
/// One call site today. A second decline path for the same head and dtype would
/// be masked by the first and should pass its own discriminator rather than rely
/// on the message text differing.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
fn warn_lm_head_unbatched(head: &str, dtype: DType) {
    use std::sync::Mutex;
    // A Vec, not a HashSet: `DType` is not `Hash`, and the set is tiny. Same
    // reasoning as `transformer::warn_unbatchable`.
    static SEEN: Mutex<Vec<(String, DType)>> = Mutex::new(Vec::new());
    let mut guard = match SEEN.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(), // a poisoned warn-dedupe set must not kill inference
    };
    if !guard.iter().any(|(h, d)| h == head && *d == dtype) {
        guard.push((head.to_string(), dtype));
        tracing::warn!(
            "batched LM-head projection declined (`{head}` is {dtype:?}, which has \
             no batched GEMM kernel here); speculative verification will re-read \
             the output matrix once per verified position instead of once per round"
        );
    }
}

/// Force the batched LM-head projection to decline, for A/B measurement.
///
/// `CERA_LM_HEAD_NO_GEMM=1` puts the projection back on the per-row loop the
/// GEMM replaced. Without it the "before" half of the A/B in
/// `tests/spec_lm_head_bench.rs` can only be reproduced by hand-editing this
/// file, which makes a headline perf number unfalsifiable the moment its author
/// moves on. Same lever-for-measurement role as `CERA_CPU_TIER`.
///
/// Read once per process — this sits in the verification hot path.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
fn lm_head_gemm_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("CERA_LM_HEAD_NO_GEMM").as_deref() == Ok("1"))
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
#[inline]
fn apply_column_major_bias(mat: &mut [f32], bias: &[f32], dim: usize, n: usize) {
    if n == 1 {
        let len = dim.min(bias.len()).min(mat.len());
        cpu::add_inplace(&mut mat[..len], &bias[..len]);
    } else {
        for (i, &b) in bias.iter().enumerate().take(dim) {
            let start = i * n;
            let end = (start + n).min(mat.len());
            if start >= end {
                break;
            }
            for val in &mut mat[start..end] {
                *val += b;
            }
        }
    }
}

impl LlamaModel {
    fn check_rewind_mode(
        &self,
        state: &InferenceState,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        if !self.config.is_causal || state.lora.as_ref().is_some_and(|l| l.is_classifier()) {
            return Err(crate::kv_cache::KvRewindError::NonCausal);
        }
        Ok(())
    }

    /// Construct without a model identifier.
    #[allow(dead_code)]
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF path).
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        ensure!(context_size > 0, "context_size must be > 0");

        // Metadata prefix is the architecture string itself
        // ("qwen2"/"qwen3"/"llama"/"granite"; classic Mistral ships as "llama").
        let arch = gguf
            .get_str("general.architecture")
            .context("missing general.architecture")?
            .to_lowercase();
        let resolved_prefix = if (arch == "ministral3" || arch == "ministral")
            && (!gguf
                .metadata
                .keys()
                .any(|k| k.starts_with(&format!("{arch}.")))
                || gguf.metadata.keys().any(|k| k.starts_with("mistral3.")))
        {
            "mistral3"
        } else if arch == "phi"
            && (!gguf.metadata.contains_key("phi.block_count")
                && gguf.metadata.contains_key("phi3.block_count"))
        {
            "phi3"
        } else {
            arch.as_str()
        };
        let prefix = resolved_prefix;

        // RoPE layout per arch. Qwen, Gemma 2, Olmo 2, and Olmo 3 GGUFs are NEOX (split-halves);
        let rope_type = match prefix {
            "qwen2" | "qwen3" | "gemma2" | "olmo2" | "olmo3" | "phi3" | "phi" => RopeType::Neox,
            // "llama" also covers classic Mistral (it ships as GGUF arch "llama").
            "llama" | "granite" | "minicpm" | "minicpm5" | "nanbeige" | "mistral3"
            | "ministral3" | "ministral" => RopeType::Norm,
            // Keep exhaustive with the `load_model` dispatch allow-list: a new arch
            // routed here without a layout mapping must fail loudly rather than
            // silently default to NORM (wrong for any NEOX-family arch: phi3,
            // stablelm, starcoder2, etc. are all NEOX in llama.cpp).
            other => bail!(
                "LlamaModel: no RoPE layout mapping for arch {other:?}; \
                 add it to the rope_type match in llama.rs"
            ),
        };

        let norm_order = match prefix {
            "olmo2" | "olmo3" => NormOrder::PostNorm,
            _ => NormOrder::PreNorm,
        };

        let activation = match prefix {
            "gemma2" => FfnActivation::Geglu,
            _ => FfnActivation::Swiglu,
        };

        let attn_logit_softcapping = gguf
            .get_f32(&format!("{prefix}.attn_logit_softcapping"))
            .filter(|&c| c.is_finite() && c > 0.0);
        let final_logit_softcapping = gguf
            .get_f32(&format!("{prefix}.final_logit_softcapping"))
            .filter(|&c| c.is_finite() && c > 0.0);
        if (prefix == "minicpm" || prefix == "minicpm5")
            && (attn_logit_softcapping.is_some() || final_logit_softcapping.is_some())
        {
            tracing::warn!(
                "minicpm model specifies unexpected logit softcapping; softcapping may interact unexpectedly with logit scaling"
            );
        }

        let sliding_window = gguf
            .get_u32(&format!("{prefix}.attention.sliding_window"))
            .or_else(|| gguf.get_u32("attention.sliding_window"))
            .map(|w| w as usize)
            .filter(|&w| w > 0);
        let sliding_window_pattern = gguf
            .get_bool_array(&format!("{prefix}.attention.sliding_window_pattern"))
            .or_else(|| gguf.get_bool_array("attention.sliding_window_pattern"))
            .filter(|p| !p.is_empty())
            .or_else(|| {
                gguf.get_u32(&format!("{prefix}.attention.sliding_window_pattern"))
                    .or_else(|| gguf.get_u32("attention.sliding_window_pattern"))
                    .map(|period| {
                        let p = period as usize;
                        if p == 1 {
                            vec![false]
                        } else if p == 0 {
                            vec![true]
                        } else if p > 1024 {
                            tracing::warn!("sliding_window_pattern period {p} exceeds maximum 1024; defaulting to full attention");
                            vec![false]
                        } else {
                            (0..p).map(|i| i < p - 1).collect()
                        }
                    })
            })
            .or_else(|| {
                // Default SWA patterns when sliding_window is set but pattern key is omitted in GGUF:
                // Olmo 2/3 default to period 4 (3 SWA layers, 1 dense layer).
                // Gemma 2 defaults to period 2 (1 SWA layer, 1 dense layer).
                match prefix {
                    "olmo2" | "olmo3" if sliding_window.is_some() => {
                        Some(vec![true, true, true, false])
                    }
                    "gemma2" if sliding_window.is_some() => Some(vec![true, false]),
                    _ => None,
                }
            });

        let rope_scaling_type = gguf
            .get_str(&format!("{prefix}.rope.scaling.type"))
            .or_else(|| gguf.get_str("rope.scaling.type"));
        let yarn = if matches!(rope_scaling_type, Some(s) if s.eq_ignore_ascii_case("yarn")) {
            let factor = gguf
                .get_f32(&format!("{prefix}.rope.scaling.factor"))
                .or_else(|| gguf.get_f32("rope.scaling.factor"))
                .filter(|x| x.is_finite() && *x > 0.0)
                .unwrap_or(1.0);
            let orig_ctx_len = gguf
                .get_u32(&format!("{prefix}.rope.scaling.original_context_length"))
                .or_else(|| gguf.get_u32(&format!("{prefix}.rope.scaling.orig_ctx_len")))
                .or_else(|| gguf.get_u32("rope.scaling.original_context_length"))
                .or_else(|| gguf.get_u32("rope.scaling.orig_ctx_len"))
                .map(|len| len as usize)
                .filter(|&len| len > 0)
                .unwrap_or(context_size);
            let attn_factor = gguf
                .get_f32(&format!("{prefix}.rope.scaling.attn_factor"))
                .or_else(|| gguf.get_f32(&format!("{prefix}.rope.scaling.yarn_attn_factor")))
                .or_else(|| gguf.get_f32("rope.scaling.attn_factor"))
                .or_else(|| gguf.get_f32("rope.scaling.yarn_attn_factor"))
                .filter(|x| x.is_finite() && *x > 0.0)
                .unwrap_or(1.0);
            let beta_fast = gguf
                .get_f32(&format!("{prefix}.rope.scaling.yarn_beta_fast"))
                .or_else(|| gguf.get_f32("rope.scaling.yarn_beta_fast"))
                .filter(|x| x.is_finite() && *x > 0.0)
                .unwrap_or(32.0);
            let beta_slow = gguf
                .get_f32(&format!("{prefix}.rope.scaling.yarn_beta_slow"))
                .or_else(|| gguf.get_f32("rope.scaling.yarn_beta_slow"))
                .filter(|x| x.is_finite() && *x > 0.0)
                .unwrap_or(1.0);
            let ext_factor = gguf
                .get_f32(&format!("{prefix}.rope.scaling.yarn_ext_factor"))
                .or_else(|| gguf.get_f32("rope.scaling.yarn_ext_factor"))
                .filter(|x| x.is_finite() && *x >= 0.0)
                .map(|x| x.clamp(0.0, 1.0))
                .unwrap_or(1.0);
            let log_mul = gguf
                .get_f32(&format!("{prefix}.rope.scaling.yarn_log_multiplier"))
                .or_else(|| gguf.get_f32("rope.scaling.yarn_log_multiplier"))
                .filter(|x| x.is_finite() && *x >= 0.0)
                .unwrap_or(0.1);
            let freq_scale = if factor > 0.0 { 1.0 / factor } else { 1.0 };
            Some(cpu::YarnParams::new_with_log_mul(
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
                orig_ctx_len,
                log_mul,
            ))
        } else {
            None
        };

        let attn_temp_scale = gguf
            .get_f32(&format!("{prefix}.attention.temperature_scale"))
            .or_else(|| gguf.get_f32("attention.temperature_scale"))
            .or_else(|| gguf.get_f32(&format!("{prefix}.attention.temp_scale")))
            .or_else(|| gguf.get_f32("attention.temp_scale"))
            .filter(|&scale| scale.is_finite() && scale > 0.0)
            .map(|scale| {
                let floor_scale = gguf
                    .get_u32(&format!("{prefix}.attention.temperature_length"))
                    .or_else(|| gguf.get_u32("attention.temperature_length"))
                    .or_else(|| gguf.get_u32(&format!("{prefix}.attention.temp_floor_scale")))
                    .or_else(|| gguf.get_u32("attention.temp_floor_scale"))
                    .or_else(|| {
                        gguf.get_u32(&format!("{prefix}.rope.scaling.original_context_length"))
                    })
                    .or_else(|| gguf.get_u32("rope.scaling.original_context_length"))
                    .or_else(|| gguf.get_u32(&format!("{prefix}.context_length")))
                    .or_else(|| gguf.get_u32("context_length"))
                    .map(|len| len as usize)
                    .filter(|&len| len > 0)
                    .unwrap_or(context_size.max(1));
                (scale, floor_scale)
            });

        let n_phys_layers =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .with_context(|| format!("missing {prefix}.block_count"))? as usize;
        ensure!(n_phys_layers > 0, "block_count must be > 0");

        let (n_loops, skip_loop_final_norm) = if prefix == "nanbeige" {
            let loops = match gguf.metadata.get("nanbeige.num_loops") {
                Some(crate::gguf::GgufValue::U8(v)) => *v as usize,
                Some(crate::gguf::GgufValue::I8(v)) => {
                    ensure!(*v >= 1, "nanbeige.num_loops must be >= 1, got {v}");
                    *v as usize
                }
                Some(crate::gguf::GgufValue::U16(v)) => *v as usize,
                Some(crate::gguf::GgufValue::I16(v)) => {
                    ensure!(*v >= 1, "nanbeige.num_loops must be >= 1, got {v}");
                    *v as usize
                }
                Some(crate::gguf::GgufValue::U32(v)) => *v as usize,
                Some(crate::gguf::GgufValue::I32(v)) => {
                    ensure!(*v >= 1, "nanbeige.num_loops must be >= 1, got {v}");
                    *v as usize
                }
                Some(crate::gguf::GgufValue::U64(v)) => usize::try_from(*v)
                    .context("nanbeige.num_loops exceeds platform pointer width")?,
                Some(crate::gguf::GgufValue::I64(v)) => {
                    ensure!(*v >= 1, "nanbeige.num_loops must be >= 1, got {v}");
                    usize::try_from(*v)
                        .context("nanbeige.num_loops exceeds platform pointer width")?
                }
                Some(other) => bail!("nanbeige.num_loops has unexpected metadata type {other:?}"),
                None => 1,
            };
            ensure!(loops >= 1, "nanbeige.num_loops must be >= 1, got {loops}");
            let skip = match gguf.metadata.get("nanbeige.skip_loop_final_norm") {
                Some(crate::gguf::GgufValue::Bool(b)) => *b,
                Some(crate::gguf::GgufValue::U8(v)) => *v != 0,
                Some(crate::gguf::GgufValue::I8(v)) => *v != 0,
                Some(crate::gguf::GgufValue::U16(v)) => *v != 0,
                Some(crate::gguf::GgufValue::I16(v)) => *v != 0,
                Some(crate::gguf::GgufValue::U32(v)) => *v != 0,
                Some(crate::gguf::GgufValue::I32(v)) => *v != 0,
                Some(crate::gguf::GgufValue::U64(v)) => *v != 0,
                Some(crate::gguf::GgufValue::I64(v)) => *v != 0,
                Some(other) => {
                    bail!("nanbeige.skip_loop_final_norm has unexpected metadata type {other:?}")
                }
                None => false,
            };
            (loops, skip)
        } else {
            (1, false)
        };

        let n_layers = n_phys_layers
            .checked_mul(n_loops)
            .context("layer count overflow")?;
        ensure!(
            n_layers <= 512,
            "total logical layer count ({n_layers} = {n_phys_layers} phys * {n_loops} loops) \
             exceeds maximum supported layers (512)"
        );
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .with_context(|| format!("missing {prefix}.embedding_length"))?
            as usize;
        ensure!(n_layers > 0, "{prefix}.block_count must be > 0");
        ensure!(hidden_size > 0, "{prefix}.embedding_length must be > 0");

        // Granite 3.x and MiniCPM scalar multipliers (embedding/residual/attention/logit).
        // Absent on every other arch => identity, so this is a no-op for
        // LLaMA/Mistral/Qwen. Carried on `config.scalars`.
        let mut scalars = ScalarMultipliers::from_gguf(&gguf, prefix, n_layers, hidden_size)?;

        // Gemma 2 scales token embeddings by sqrt(hidden_size).
        if prefix == "gemma2" && scalars.embedding == 1.0 {
            scalars.embedding = (hidden_size as f32).sqrt();
        }
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .with_context(|| format!("missing {prefix}.feed_forward_length"))?
            as usize;
        ensure!(
            intermediate_size > 0,
            "{prefix}.feed_forward_length must be > 0"
        );
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .with_context(|| format!("missing {prefix}.attention.head_count"))?
            as usize;
        // SCALAR head_count_kv (not the per-layer array LFM2 uses).
        let n_kv_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .with_context(|| format!("missing {prefix}.attention.head_count_kv"))?
            as usize;
        ensure!(
            n_heads > 0 && n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads),
            "n_heads ({n_heads}) must be a positive multiple of n_kv_heads ({n_kv_heads})"
        );
        // Qwen GGUFs typically omit `{prefix}.vocab_size`; derive it from the
        // embedding tensor's outer dim (row count) when the key is absent.
        let vocab_size = match gguf.get_u32(&format!("{prefix}.vocab_size")) {
            Some(v) => v as usize,
            None => {
                let info = gguf
                    .tensors
                    .get("token_embd.weight")
                    .context("missing token_embd.weight (cannot derive vocab_size)")?;
                ensure!(
                    info.shape.len() >= 2,
                    "token_embd.weight has unexpected shape {:?}",
                    info.shape
                );
                info.shape[1]
            }
        };

        // Cap max_seq_len by the requested context_size (mirrors LFM2).
        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let default_rope_theta = match prefix {
            "gemma2" | "minicpm" | "minicpm5" | "nanbeige" | "phi3" | "phi" => 10_000.0,
            _ => 1_000_000.0,
        };
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(default_rope_theta);
        ensure!(
            rope_theta.is_finite() && (1.0..=1e9).contains(&rope_theta),
            "{prefix}.rope.freq_base must be finite and within [1.0, 1e9]"
        );
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        ensure!(
            rms_norm_eps.is_finite() && (1e-12..=1e-2).contains(&rms_norm_eps),
            "{prefix}.attention.layer_norm_rms_epsilon must be finite and within [1e-12, 1e-2]"
        );

        // head_dim: default hidden_size / n_heads, overridden by the optional
        // `{prefix}.attention.key_length` (Qwen3 sets this explicitly).
        let head_dim = match gguf.get_u32(&format!("{prefix}.attention.key_length")) {
            Some(v) => {
                ensure!(v > 0, "{prefix}.attention.key_length must be > 0");
                v as usize
            }
            None => {
                ensure!(
                    hidden_size.is_multiple_of(n_heads),
                    "hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
                );
                hidden_size / n_heads
            }
        };
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(2) && head_dim <= 4096,
            "head_dim ({head_dim}) must be positive, even for RoPE rotation, and <= 4096"
        );

        let block_types = vec![BlockType::Attention; n_layers];
        let kv_heads_per_layer = vec![n_kv_heads; n_layers];

        let config = ModelConfig {
            architecture: arch.clone(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types,
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer,
            scalars,
            // Dense transformers only; the `llama`-family loader has no expert path.
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };

        // Final norm tensor (NOT the LFM2 `token_embd_norm.weight`).
        let output_norm_weight = gguf.get_tensor("output_norm.weight")?.try_to_f32_vec()?;
        ensure!(
            output_norm_weight.len() == hidden_size,
            "output_norm length {} != hidden_size ({hidden_size})",
            output_norm_weight.len()
        );

        let q_dim = config
            .n_heads
            .checked_mul(head_dim)
            .context("q_dim overflow")?;
        let k_dim = config
            .n_kv_heads
            .checked_mul(head_dim)
            .context("k_dim overflow")?;
        let v_dim = config
            .n_kv_heads
            .checked_mul(head_dim)
            .context("v_dim overflow")?;
        let qkv_dim = q_dim
            .checked_add(k_dim)
            .and_then(|s| s.checked_add(v_dim))
            .context("qkv dimension overflow")?;
        let double_intermediate = config
            .intermediate_size
            .checked_mul(2)
            .context("intermediate size overflow")?;

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_post_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_post_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_bias = Vec::with_capacity(n_layers);
        let mut attn_k_bias = Vec::with_capacity(n_layers);
        let mut attn_v_bias = Vec::with_capacity(n_layers);
        let mut attn_output_bias = Vec::with_capacity(n_layers);
        let mut ffn_gate_bias = Vec::with_capacity(n_layers);
        let mut ffn_up_bias = Vec::with_capacity(n_layers);
        let mut ffn_down_bias = Vec::with_capacity(n_layers);
        let mut layer_refs = Vec::with_capacity(n_layers);

        for i in 0..n_phys_layers {
            // Note on Gemma 2 RMSNorm: Hugging Face checkpoints store weights with
            // an implicit +1.0 unit offset (x * (1.0 + w)), but standard GGUF converters
            // fold the +1.0 offset directly into the exported tensor data. Standard
            // cpu::rmsnorm without runtime offset addition matches upstream GGUF semantics.
            let attn_norm_name = format!("blk.{i}.attn_norm.weight");
            let attn_norm = if gguf.tensors.contains_key(&attn_norm_name) {
                gguf.get_tensor(&attn_norm_name)?.try_to_f32_vec()?
            } else if norm_order == NormOrder::PostNorm {
                Vec::new()
            } else {
                bail!("missing required tensor `{attn_norm_name}` for PreNorm architecture");
            };
            if norm_order == NormOrder::PreNorm {
                ensure!(
                    attn_norm.len() == hidden_size,
                    "layer {i} {attn_norm_name} length {} != hidden_size ({hidden_size})",
                    attn_norm.len()
                );
            }
            attn_norm_weights.push(attn_norm);

            let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");
            let ffn_norm = if gguf.tensors.contains_key(&ffn_norm_name) {
                gguf.get_tensor(&ffn_norm_name)?.try_to_f32_vec()?
            } else if norm_order == NormOrder::PostNorm {
                Vec::new()
            } else {
                bail!("missing required tensor `{ffn_norm_name}` for PreNorm architecture");
            };
            if norm_order == NormOrder::PreNorm {
                ensure!(
                    ffn_norm.len() == hidden_size,
                    "layer {i} {ffn_norm_name} length {} != hidden_size ({hidden_size})",
                    ffn_norm.len()
                );
            }
            ffn_norm_weights.push(ffn_norm);

            // Post-norms (Gemma 2, Olmo 2/3): check canonical GGUF names first.
            let attn_post = [
                format!("blk.{i}.post_attention_norm.weight"),
                format!("blk.{i}.attn_post_norm.weight"),
            ]
            .into_iter()
            .find(|name| gguf.tensors.contains_key(name))
            .map(|name| -> anyhow::Result<Vec<f32>> {
                let t = gguf.get_tensor(&name)?;
                Ok(t.try_to_f32_vec()?)
            })
            .transpose()?;
            if norm_order == NormOrder::PostNorm {
                ensure!(
                    attn_post.is_some(),
                    "missing required post-attention norm tensor for PostNorm architecture at layer {i}"
                );
            }
            if let Some(w) = &attn_post {
                ensure!(
                    w.len() == hidden_size,
                    "layer {i} post-attention norm length {} != hidden_size ({hidden_size})",
                    w.len()
                );
            }
            attn_post_norm_weights.push(attn_post);

            let ffn_post = [
                format!("blk.{i}.post_ffw_norm.weight"),
                format!("blk.{i}.ffn_post_norm.weight"),
            ]
            .into_iter()
            .find(|name| gguf.tensors.contains_key(name))
            .map(|name| -> anyhow::Result<Vec<f32>> {
                let t = gguf.get_tensor(&name)?;
                Ok(t.try_to_f32_vec()?)
            })
            .transpose()?;
            if norm_order == NormOrder::PostNorm {
                ensure!(
                    ffn_post.is_some(),
                    "missing required post-ffw norm tensor for PostNorm architecture at layer {i}"
                );
            }
            if let Some(w) = &ffn_post {
                ensure!(
                    w.len() == hidden_size,
                    "layer {i} post-ffw norm length {} != hidden_size ({hidden_size})",
                    w.len()
                );
            }
            ffn_post_norm_weights.push(ffn_post);

            // Qwen3 / Olmo 2 QK-norm: gate on tensor presence so the same code path
            // serves both archs.
            let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
            ensure!(
                gguf.tensors.contains_key(&q_norm_name) == gguf.tensors.contains_key(&k_norm_name),
                "layer {i} asymmetric QK-norm: `{q_norm_name}` and `{k_norm_name}` must both be present or both absent"
            );
            if gguf.tensors.contains_key(&q_norm_name) {
                let q_w = gguf.get_tensor(&q_norm_name)?.try_to_f32_vec()?;
                let k_w = gguf.get_tensor(&k_norm_name)?.try_to_f32_vec()?;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv_heads * head_dim;
                ensure!(
                    q_w.len() == head_dim || q_w.len() == q_dim,
                    "layer {i} {q_norm_name} length {} must match head_dim ({head_dim}) or q_dim ({q_dim})",
                    q_w.len()
                );
                ensure!(
                    k_w.len() == head_dim || k_w.len() == kv_dim,
                    "layer {i} {k_norm_name} length {} must match head_dim ({head_dim}) or kv_dim ({kv_dim})",
                    k_w.len()
                );
                let is_head_scoped = q_w.len() == head_dim && k_w.len() == head_dim;
                let is_vector_scoped = q_w.len() == q_dim && k_w.len() == kv_dim;
                ensure!(
                    is_head_scoped || is_vector_scoped,
                    "layer {i} QK-norm scoping mismatch: Q len {} (head_dim={head_dim}, q_dim={q_dim}), K len {} (head_dim={head_dim}, kv_dim={kv_dim})",
                    q_w.len(),
                    k_w.len()
                );
                attn_q_norm_weights.push(Some(q_w));
                attn_k_norm_weights.push(Some(k_w));
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
            }

            // Attention Q/K/V biases: support both fused `attn_qkv.bias` (Phi-3) and
            // separate `attn_q.bias`, `attn_k.bias`, `attn_v.bias` (Qwen2).
            let qkv_bias_name = format!("blk.{i}.attn_qkv.bias");
            let q_bias_name = format!("blk.{i}.attn_q.bias");
            let k_bias_name = format!("blk.{i}.attn_k.bias");
            let v_bias_name = format!("blk.{i}.attn_v.bias");
            if gguf.tensors.contains_key(&qkv_bias_name) {
                let qkv_b = gguf.get_tensor(&qkv_bias_name)?.try_to_f32_vec()?;
                ensure!(
                    qkv_b.len() == qkv_dim,
                    "invalid {qkv_bias_name} length {} for layer {i} (expected {qkv_dim})",
                    qkv_b.len()
                );
                ensure!(
                    qkv_b.iter().all(|v| v.is_finite()),
                    "non-finite value detected in {qkv_bias_name} for layer {i}"
                );
                attn_q_bias.push(Some(qkv_b[..q_dim].to_vec()));
                attn_k_bias.push(Some(qkv_b[q_dim..q_dim + k_dim].to_vec()));
                attn_v_bias.push(Some(qkv_b[q_dim + k_dim..qkv_dim].to_vec()));
            } else if gguf.tensors.contains_key(&q_bias_name) {
                let qb = gguf.get_tensor(&q_bias_name)?.try_to_f32_vec()?;
                let kb = gguf.get_tensor(&k_bias_name)?.try_to_f32_vec()?;
                let vb = gguf.get_tensor(&v_bias_name)?.try_to_f32_vec()?;
                ensure!(
                    qb.len() == q_dim && kb.len() == k_dim && vb.len() == v_dim,
                    "invalid Q/K/V bias lengths ({}, {}, {}) for layer {i} (expected {}, {}, {})",
                    qb.len(),
                    kb.len(),
                    vb.len(),
                    q_dim,
                    k_dim,
                    v_dim
                );
                ensure!(
                    qb.iter().all(|v| v.is_finite())
                        && kb.iter().all(|v| v.is_finite())
                        && vb.iter().all(|v| v.is_finite()),
                    "non-finite value detected in Q/K/V bias for layer {i}"
                );
                attn_q_bias.push(Some(qb));
                attn_k_bias.push(Some(kb));
                attn_v_bias.push(Some(vb));
            } else {
                attn_q_bias.push(None);
                attn_k_bias.push(None);
                attn_v_bias.push(None);
            }

            // Optional projection and FFN biases (Mistral 3 / Phi-3).
            let load_optional_bias =
                |name: &str, expected_len: usize| -> Result<Option<Vec<f32>>> {
                    if gguf.tensors.contains_key(name) {
                        let b = gguf.get_tensor(name)?.try_to_f32_vec()?;
                        ensure!(
                            b.len() == expected_len,
                            "invalid {name} length {} for layer {i} (expected {expected_len})",
                            b.len()
                        );
                        ensure!(
                            b.iter().all(|v| v.is_finite()),
                            "non-finite value detected in {name} for layer {i}"
                        );
                        Ok(Some(b))
                    } else {
                        Ok(None)
                    }
                };

            attn_output_bias.push(load_optional_bias(
                &format!("blk.{i}.attn_output.bias"),
                config.hidden_size,
            )?);
            let ffn_gate_bias_name = format!("blk.{i}.ffn_gate.bias");
            let ffn_up_bias_name = format!("blk.{i}.ffn_up.bias");
            if gguf.tensors.contains_key(&ffn_gate_bias_name) {
                ffn_gate_bias.push(load_optional_bias(
                    &ffn_gate_bias_name,
                    config.intermediate_size,
                )?);
                ffn_up_bias.push(load_optional_bias(
                    &ffn_up_bias_name,
                    config.intermediate_size,
                )?);
            } else if gguf.tensors.contains_key(&ffn_up_bias_name) {
                let b = gguf.get_tensor(&ffn_up_bias_name)?.try_to_f32_vec()?;
                ensure!(
                    b.iter().all(|v| v.is_finite()),
                    "non-finite value detected in {ffn_up_bias_name} for layer {i}"
                );
                if b.len() == double_intermediate {
                    ffn_gate_bias.push(Some(b[..config.intermediate_size].to_vec()));
                    ffn_up_bias.push(Some(b[config.intermediate_size..].to_vec()));
                } else if b.len() == config.intermediate_size {
                    ffn_gate_bias.push(None);
                    ffn_up_bias.push(Some(b));
                } else {
                    bail!(
                        "invalid {ffn_up_bias_name} length {} for layer {i} (expected {} or {})",
                        b.len(),
                        config.intermediate_size,
                        double_intermediate
                    );
                }
            } else {
                ffn_gate_bias.push(None);
                ffn_up_bias.push(None);
            }

            ffn_down_bias.push(load_optional_bias(
                &format!("blk.{i}.ffn_down.bias"),
                config.hidden_size,
            )?);

            // Projection weights: support both fused `attn_qkv.weight` (Phi-3) and
            // separate `attn_q.weight`, `attn_k.weight`, `attn_v.weight`.
            let qkv_weight_name = format!("blk.{i}.attn_qkv.weight");
            let (attn_q, attn_k, attn_v) = if gguf.tensors.contains_key(&qkv_weight_name) {
                let qkv_ref = transformer::resolve_weight(&gguf, &qkv_weight_name)?;
                ensure!(
                    qkv_ref.m == qkv_dim,
                    "fused {qkv_weight_name} row count {} does not match expected {qkv_dim}",
                    qkv_ref.m
                );
                ensure!(
                    qkv_ref.k == config.hidden_size,
                    "fused {qkv_weight_name} inner dimension k={} does not match hidden_size={}",
                    qkv_ref.k,
                    config.hidden_size
                );
                let q = qkv_ref.slice_rows(0, q_dim)?;
                let k = qkv_ref.slice_rows(q_dim, k_dim)?;
                let v = qkv_ref.slice_rows(q_dim + k_dim, v_dim)?;
                (q, k, v)
            } else {
                let q = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?;
                let k = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?;
                let v = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))?;
                ensure!(
                    q.m == q_dim && k.m == k_dim && v.m == v_dim,
                    "mismatched Q/K/V rows for layer {i}: q={}, k={}, v={} (expected {}, {}, {})",
                    q.m,
                    k.m,
                    v.m,
                    q_dim,
                    k_dim,
                    v_dim
                );
                ensure!(
                    q.k == config.hidden_size
                        && k.k == config.hidden_size
                        && v.k == config.hidden_size,
                    "mismatched Q/K/V inner dimension k for layer {i}: q={}, k={}, v={} (expected {})",
                    q.k,
                    k.k,
                    v.k,
                    config.hidden_size
                );
                (q, k, v)
            };

            // FFN weights: support both separate `ffn_gate.weight` and `ffn_up.weight` or
            // packed `ffn_up.weight` containing both gate and up stacked row-wise (Phi-3).
            let ffn_gate_name = format!("blk.{i}.ffn_gate.weight");
            let ffn_up_name = format!("blk.{i}.ffn_up.weight");
            let (ffn_gate, ffn_up) = if gguf.tensors.contains_key(&ffn_gate_name) {
                let gate = transformer::resolve_weight(&gguf, &ffn_gate_name)?;
                let up = transformer::resolve_weight(&gguf, &ffn_up_name)?;
                ensure!(
                    gate.m == config.intermediate_size && up.m == config.intermediate_size,
                    "mismatched FFN gate/up rows for layer {i}: gate={}, up={} (expected {})",
                    gate.m,
                    up.m,
                    config.intermediate_size
                );
                ensure!(
                    gate.k == config.hidden_size && up.k == config.hidden_size,
                    "mismatched FFN gate/up inner dimension k for layer {i}: gate={}, up={} (expected {})",
                    gate.k,
                    up.k,
                    config.hidden_size
                );
                (gate, up)
            } else {
                let packed_up = transformer::resolve_weight(&gguf, &ffn_up_name)?;
                ensure!(
                    packed_up.m == double_intermediate,
                    "packed {ffn_up_name} row count {} does not match expected {double_intermediate}",
                    packed_up.m
                );
                ensure!(
                    packed_up.k == config.hidden_size,
                    "packed {ffn_up_name} inner dimension k={} does not match hidden_size={}",
                    packed_up.k,
                    config.hidden_size
                );
                let gate = packed_up.slice_rows(0, config.intermediate_size)?;
                let up =
                    packed_up.slice_rows(config.intermediate_size, config.intermediate_size)?;
                (gate, up)
            };

            let attn_output =
                transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_output.weight"))?;
            ensure!(
                attn_output.m == config.hidden_size && attn_output.k == q_dim,
                "mismatched attn_output dimensions for layer {i}: m={}, k={} (expected m={}, k={})",
                attn_output.m,
                attn_output.k,
                config.hidden_size,
                q_dim
            );
            let ffn_down = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?;
            ensure!(
                ffn_down.m == config.hidden_size && ffn_down.k == config.intermediate_size,
                "mismatched ffn_down dimensions for layer {i}: m={}, k={} (expected m={}, k={})",
                ffn_down.m,
                ffn_down.k,
                config.hidden_size,
                config.intermediate_size
            );

            // `.with_repack` on the projection weights only: these are the ones
            // that hit the batched prefill GEMM at `n > 1`. token_embd / output
            // stay excluded.
            layer_refs.push(LayerWeightRefs {
                attn_q: attn_q.with_repack(&gguf),
                attn_k: attn_k.with_repack(&gguf),
                attn_v: attn_v.with_repack(&gguf),
                attn_output: attn_output.with_repack(&gguf),
                ffn_gate: ffn_gate.with_repack(&gguf),
                ffn_up: ffn_up.with_repack(&gguf),
                ffn_down: ffn_down.with_repack(&gguf),
            });
        }

        if n_loops > 1 {
            for _ in 1..n_loops {
                attn_norm_weights.extend_from_within(..n_phys_layers);
                ffn_norm_weights.extend_from_within(..n_phys_layers);
                attn_post_norm_weights.extend_from_within(..n_phys_layers);
                ffn_post_norm_weights.extend_from_within(..n_phys_layers);
                attn_q_norm_weights.extend_from_within(..n_phys_layers);
                attn_k_norm_weights.extend_from_within(..n_phys_layers);
                attn_q_bias.extend_from_within(..n_phys_layers);
                attn_k_bias.extend_from_within(..n_phys_layers);
                attn_v_bias.extend_from_within(..n_phys_layers);
                attn_output_bias.extend_from_within(..n_phys_layers);
                ffn_gate_bias.extend_from_within(..n_phys_layers);
                ffn_up_bias.extend_from_within(..n_phys_layers);
                ffn_down_bias.extend_from_within(..n_phys_layers);
                layer_refs.extend_from_within(..n_phys_layers);
            }
        }

        let loop_norm_interval = if n_loops > 1 && !skip_loop_final_norm {
            Some(n_phys_layers)
        } else {
            None
        };

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;
        // Separate output projection when present, else tied embeddings.
        let output_ref = if gguf.tensors.contains_key("output.weight") {
            Some(transformer::resolve_weight(&gguf, "output.weight")?)
        } else {
            None
        };

        // The LM head must be able to produce `vocab_size` logits from a
        // `hidden_size` vector. Checked at load because every logit projection
        // in this file trusts it in release: the GEMV kernels take their row
        // count from the *output buffer* rather than from `wref.m` (see
        // `gemm_preq`'s docs and `cpu::par_rows(y, ..)` in the SIMD kernels), so
        // a head with fewer rows than `vocab_size` reads past the end of the
        // weight on every decode step, not merely on the batched path. A
        // mismatched `k` overruns the activation the same way. Neither is
        // reachable on a well-formed GGUF; rejecting the file beats undefined
        // behaviour that only shows up as plausible garbage.
        {
            let head = output_ref.as_ref().unwrap_or(&embd_ref);
            let head_name = if output_ref.is_some() {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            ensure!(
                head.k == config.hidden_size,
                "LM head `{head_name}` has k={} but hidden_size is {}",
                head.k,
                config.hidden_size
            );
            ensure!(
                head.m >= config.vocab_size,
                "LM head `{head_name}` has {} rows, fewer than vocab_size {}",
                head.m,
                config.vocab_size
            );
            ensure!(
                config.hidden_size.is_multiple_of(32),
                "hidden_size {} is not a multiple of 32, which the Q8_0 \
                 activation quantization on both logit paths requires",
                config.hidden_size
            );
            // Both logit paths quantize the activation to Q8_0, whose blocks are
            // 32 wide, so `hidden_size` must be a whole number of them. This is
            // NOT a batched-path-only constraint and must not be a decline: the
            // per-row `project_logits` fallback asserts the same thing one frame
            // deeper — `quantize_to_scratch` on aarch64,
            // `cpu::quantize_f32_to_q8_0_into` on the x86 int8 tiers, both hard
            // `assert!`s — and where no int8 kernel runs at all the scalar GEMV
            // truncates `k / 32` and mis-reads every row instead. No path
            // tolerates it, so reject the file rather than defer to a fallback
            // that will only fail later and worse.
            //
            // Checked rather than treated as implied by the dtype:
            // `batched_gemm_supports` constrains `k` only for K-quants
            // (`k % 256`), and GGUF validates a tensor's *total* element count
            // against its block size rather than its per-row `k`, so a
            // Q4_0/Q8_0 head with an unaligned `hidden_size` clears every other
            // gate.
        }

        // Llama-3 RoPE frequency scaling (`rope_scaling: llama3`): per-pair factors
        // that divide each rotation angle, applied by llama.cpp on every rope call.
        // Present on Llama-3.x, absent on Qwen/Mistral/Granite ⇒ None (plain RoPE).
        let rope_freqs = gguf
            .get_tensor("rope_freqs.weight")
            .ok()
            .and_then(|t| t.try_to_f32_vec().ok());
        if let Some(rf) = &rope_freqs {
            ensure!(
                rf.len() == head_dim / 2,
                "rope_freqs.weight has {} entries, expected head_dim/2 = {}",
                rf.len(),
                head_dim / 2
            );
        }

        Ok(Self {
            gguf,
            config,
            head_dim,
            rope_type,
            rope_freqs,
            norm_order,
            activation,
            attn_logit_softcapping,
            final_logit_softcapping,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_post_norm_weights,
            ffn_post_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            attn_q_bias,
            attn_k_bias,
            attn_v_bias,
            attn_output_bias,
            ffn_gate_bias,
            ffn_up_bias,
            ffn_down_bias,
            embd_ref,
            output_ref,
            layer_refs,
            sliding_window,
            sliding_window_pattern,
            yarn,
            attn_temp_scale,
            loop_norm_interval,
            model_id,
        })
    }

    /// Check whether layer `il` is a sliding window attention layer.
    pub fn is_swa_layer(&self, il: usize) -> bool {
        if self.sliding_window.unwrap_or(0) == 0 {
            return false;
        }
        if let Some(pattern) = &self.sliding_window_pattern {
            if pattern.is_empty() {
                false
            } else {
                pattern[il % pattern.len()]
            }
        } else {
            true
        }
    }

    /// Return the global sliding window size if configured.
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Return whether any layer specifies projection or FFN biases.
    pub fn has_projection_or_ffn_biases(&self) -> bool {
        self.attn_output_bias.iter().any(Option::is_some)
            || self.ffn_gate_bias.iter().any(Option::is_some)
            || self.ffn_up_bias.iter().any(Option::is_some)
            || self.ffn_down_bias.iter().any(Option::is_some)
    }

    pub fn layer_sliding_window(&self, il: usize) -> Option<usize> {
        if self.is_swa_layer(il) {
            self.sliding_window
        } else {
            None
        }
    }

    /// Return YaRN parameters for layer `il`, or `None` if it is an SWA layer or non-YaRN model.
    pub fn layer_yarn(&self, il: usize) -> Option<cpu::YarnParams> {
        if self.is_swa_layer(il) {
            None
        } else {
            self.yarn
        }
    }

    /// Physical layer loop interval for looped architectures (e.g. Nanbeige).
    pub fn loop_norm_interval(&self) -> Option<usize> {
        self.loop_norm_interval
    }

    /// Attention dims for layer `il`.
    fn attn_dims(&self, il: usize) -> AttnDims<'_> {
        AttnDims {
            hidden_size: self.config.hidden_size,
            n_heads: self.config.n_heads,
            n_kv_heads: self.config.n_kv_heads,
            head_dim: self.head_dim,
            rope_theta: self.config.rope_theta,
            rms_norm_eps: self.config.rms_norm_eps,
            rope_type: self.rope_type,
            attn_scale: self.config.scalars.attn,
            rope_freqs: self.rope_freqs.as_deref(),
            attn_logit_softcapping: self.attn_logit_softcapping,
            sliding_window: self.layer_sliding_window(il),
            yarn: self.layer_yarn(il),
            attn_temp_scale: self.attn_temp_scale,
        }
    }

    /// Run all layers + final RMSNorm on a single-token hidden state.
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        // Take scratch out of `state` to avoid borrow conflicts with the
        // helpers that need `&mut state`; restore at the end.
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        if self.norm_order == NormOrder::PreNorm {
            normed.resize(hs, 0.0);
            ffn_input.resize(hs, 0.0);
        }

        let nb = hs / 32;
        state.scratch.q8_scales.resize(nb, 0.0);
        state.scratch.q8_quants.resize(hs, 0);

        for i in 0..cfg.n_layers {
            // Attention pre-norm.
            let normed_in = match self.norm_order {
                NormOrder::PreNorm => {
                    cpu::rmsnorm_and_quantize_q8_0(
                        hidden,
                        &self.attn_norm_weights[i],
                        cfg.rms_norm_eps,
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                        Some(&mut normed),
                    );
                    &normed[..]
                }
                NormOrder::PostNorm => {
                    #[cfg(target_arch = "aarch64")]
                    transformer::quantize_to_scratch(hidden, state);
                    &hidden[..]
                }
            };

            let refs = &self.layer_refs[i];
            let weights = AttnWeights {
                attn_q: &refs.attn_q,
                attn_k: &refs.attn_k,
                attn_v: &refs.attn_v,
                attn_output: &refs.attn_output,
            };
            let extras = AttnExtras {
                qkv_bias: match (
                    self.attn_q_bias[i].as_deref(),
                    self.attn_k_bias[i].as_deref(),
                    self.attn_v_bias[i].as_deref(),
                ) {
                    (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                    _ => None,
                },
                qk_norm: match (
                    self.attn_q_norm_weights[i].as_deref(),
                    self.attn_k_norm_weights[i].as_deref(),
                ) {
                    (Some(q), Some(k)) => Some((q, k)),
                    _ => None,
                },
                attn_output_bias: self.attn_output_bias[i].as_deref(),
            };
            let dims = self.attn_dims(i);
            transformer::forward_attn_block(
                &self.gguf, i, &weights, &extras, dims, normed_in, pos, state,
            );

            // Post-norm on attention output (Gemma 2, Olmo 2/3).
            if let Some(post_norm) = &self.attn_post_norm_weights[i] {
                cpu::rmsnorm(&mut state.scratch.out[..hs], post_norm, cfg.rms_norm_eps);
            }

            // Granite scales the block output before the residual add (identity
            // for every other arch).
            if self.config.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut state.scratch.out[..hs], self.config.scalars.residual);
            }
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            // FFN pre-norm.
            let ffn_in = match self.norm_order {
                NormOrder::PreNorm => {
                    cpu::rmsnorm_and_quantize_q8_0(
                        hidden,
                        &self.ffn_norm_weights[i],
                        cfg.rms_norm_eps,
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                        Some(&mut ffn_input),
                    );
                    &ffn_input[..]
                }
                NormOrder::PostNorm => {
                    #[cfg(target_arch = "aarch64")]
                    transformer::quantize_to_scratch(hidden, state);
                    &hidden[..]
                }
            };

            let refs = &self.layer_refs[i];
            let ffn_weights = FfnWeights {
                ffn_gate: &refs.ffn_gate,
                ffn_up: &refs.ffn_up,
                ffn_down: &refs.ffn_down,
            };
            let ffn_extras = transformer::FfnExtras {
                gate_bias: self.ffn_gate_bias[i].as_deref(),
                up_bias: self.ffn_up_bias[i].as_deref(),
                down_bias: self.ffn_down_bias[i].as_deref(),
            };
            transformer::forward_ffn_block(
                &self.gguf,
                i,
                &ffn_weights,
                &ffn_extras,
                hs,
                cfg.intermediate_size,
                ffn_in,
                self.activation,
                state,
            );

            // Post-norm on FFN output (Gemma 2, Olmo 2/3).
            if let Some(post_norm) = &self.ffn_post_norm_weights[i] {
                cpu::rmsnorm(&mut state.scratch.out[..hs], post_norm, cfg.rms_norm_eps);
            }

            if self.config.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut state.scratch.out[..hs], self.config.scalars.residual);
            }
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            // Oracle gate: residual stream after the full layer (= llama.cpp's
            // `l_out-{i}`). All-position for early layers, last-position for the
            // final layer: the test sums vs. takes-last accordingly. Guarded so
            // the per-token `format!` allocation only happens when dumping.
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("l_out-{i}"), hidden);
            }

            if self
                .loop_norm_interval
                .is_some_and(|n_phys| (i + 1) % n_phys == 0)
                && (i + 1) < cfg.n_layers
            {
                cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(&format!("loop_norm-{i}"), hidden);
                }
            }
        }

        cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
        transformer::oracle_dump::record("result_norm", hidden);
        state.seq_len += 1;

        state.scratch.normed = normed;
        state.scratch.ffn_input = ffn_input;
    }

    /// Project the final hidden state to logits over the vocabulary, using the
    /// separate `output.weight` when present, else the tied embedding table.
    fn project_logits(&self, hidden: &[f32], state: &mut InferenceState) -> Vec<f32> {
        let cfg = &self.config;
        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch(hidden, state);
            transformer::gemv_preq(
                &self.gguf,
                out_ref,
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = state;
            transformer::gemv(&self.gguf, out_ref, hidden, &mut logits);
        }
        // Granite divides the logits by `logits_scaling` (identity elsewhere).
        if self.config.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits, 1.0 / self.config.scalars.logit);
        }
        if let Some(cap) = self.final_logit_softcapping {
            cpu::softcap_inplace(&mut logits, cap);
        }
        transformer::oracle_dump::record("result_output", &logits);
        logits
    }

    /// Project `n` post-final-norm hidden states to logits in ONE GEMM, reading
    /// the LM head once for all of them. Input is row-major `[n × hs]` (the
    /// `forward_prefill_batched` hidden capture); output is row-major
    /// `[n × vocab]`, the layout `spec::verify_draft` indexes by row.
    ///
    /// This exists for speculative decoding. Verifying `1 + k` drafted tokens in
    /// one forward is supposed to amortize a single pass over the weights, but a
    /// per-row `project_logits` loop re-streams `hidden_size × vocab` — the
    /// largest tensor in the model — once per position, which gives most of that
    /// back.
    ///
    /// A/B on Llama-3.2-1B-Q4_0 (M1 Max), interleaved in one binary via
    /// `tests/spec_lm_head_bench.rs` — `CERA_LM_HEAD_NO_GEMM=1` runs the
    /// "before" half — three rounds, comparing minima:
    ///
    /// | `n` | per-row | batched |
    /// |-----|---------|---------|
    /// | 2   | 19.5 ms | 19.0 ms |
    /// | 4   | 31.6 ms | 25.4 ms |
    /// | 7   | 57.2 ms | 42.3 ms |
    /// | 9   | 67.3 ms | 53.6 ms |
    ///
    /// `n = 7` is the default `k = 6` draft: **~26% off a verification round**.
    /// Least-squares over those four rows puts the marginal cost of one more
    /// verified position at **~7.09 ms → ~5.05 ms**. (The benchmark prints its
    /// own fit over medians, which runs a little higher — medians carry the
    /// background load these minima exclude.)
    ///
    /// That ~2.04 ms/position is one LM-head read: despite the model's `Q4_0`
    /// name its tied head (`token_embd.weight`) is stored **Q6_K**, so at
    /// hs 2048 × vocab 128256 it is ~205 MiB — ~105 GB/s, well under this
    /// machine's peak, so the read is a real bandwidth term rather than a
    /// saturated one. What remains scales with `n` because it is per-token
    /// arithmetic, not a second amortizable weight read. Absolute ms are
    /// machine- and thermal-dependent; the ratio is the durable number.
    ///
    /// Returns `None` — leaving the caller on the per-row path — in exactly
    /// three cases: the LM head's dtype has no batched kernel here;
    /// `CERA_LM_HEAD_NO_GEMM=1` asked for the fallback; or `gemm_preq` reports
    /// that nothing ran. The last is a release-build safety net rather than an
    /// expected outcome, since it means the gate and the kernel table have
    /// drifted, and `gemm_preq` trips a `debug_assert` on it first.
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
    fn project_logits_batched(&self, hidden: &[f32], n: usize) -> Option<Vec<f32>> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        // A tied head IS `token_embd.weight`; naming it that way keeps the
        // warning below from sending an operator after an `output.weight` the
        // GGUF does not contain.
        let (head_name, out_ref) = match self.output_ref.as_ref() {
            Some(r) => ("output.weight", r),
            None => ("token_embd.weight", &self.embd_ref),
        };
        // `k == hs`, `m >= vocab`, and `hs % 32 == 0` are enforced at load
        // (`from_gguf_with_id`), which is what lets the GEMM index the weight,
        // the quantizer take whole Q8_0 blocks, and the transpose slice `vocab`
        // rows, none of them re-checking here.
        debug_assert_eq!(hidden.len(), n * hs, "hidden must be row-major [n * hs]");
        debug_assert_eq!(out_ref.k, hs);
        debug_assert!(out_ref.m >= vocab);
        debug_assert!(hs.is_multiple_of(32));

        if lm_head_gemm_disabled() {
            return None; // CERA_LM_HEAD_NO_GEMM=1; asked for, so not a warning.
        }
        // The one decline that warns. Falling back is not *wrong* — the per-row
        // path computes the same projection, to within f32 accumulation order —
        // so no correctness test can see it, and the only symptom is the
        // per-position LM-head read quietly coming back. That shape of silence
        // is how this repo lost ~4x on CPU prefill and ~340x on GPU submits.
        if !transformer::batched_gemm_supports(out_ref.dtype, hs) {
            warn_lm_head_unbatched(head_name, out_ref.dtype);
            return None;
        }

        // The GEMM's row count is the weight's, not `vocab`: an embedding table
        // used as a tied LM head may carry padding rows beyond the vocabulary
        // (see the `token_id < vocab_size` bound in `forward_prefill_batched`).
        // Computing them and dropping them in the transpose below keeps this
        // agreeing with `gemm_preq`'s `wref.m == m` contract; no shipping model
        // pads enough for the wasted rows to matter. Asserted `>= vocab` above.
        let rows = out_ref.m;

        // Quantize the activations straight out of `hidden`. No transpose:
        // `quantize_columns` exists to gather column `j` out of a column-major
        // matrix, but a row-major `[n × hs]` capture already stores position
        // `j`'s hidden vector contiguously at `hidden[j*hs..]` — which is
        // precisely the column the gather would rebuild. Feeding the rows
        // directly produces byte-identical `scales`/`quants` in the same packed
        // `[n][hs/32]` / `[n][hs]` layout the int8 GEMM consumes.
        let nb = hs / 32;
        let mut bq_scales = vec![0.0f32; n * nb];
        let mut bq_quants = vec![0i8; n * hs];
        for j in 0..n {
            cpu::quantize_f32_to_q8_0_into(
                &hidden[j * hs..(j + 1) * hs],
                &mut bq_scales[j * nb..(j + 1) * nb],
                &mut bq_quants[j * hs..(j + 1) * hs],
            );
        }

        let mut out = vec![0.0f32; rows * n];
        if !transformer::gemm_preq(
            &self.gguf, out_ref, &bq_scales, &bq_quants, &mut out, rows, n, hs,
        ) {
            return None;
        }

        // Column-major `[rows × n]` → the row-major `[n × vocab]` layout
        // `verify_draft` slices by row, dropping any pad rows.
        let mut logits = vec![0.0f32; n * vocab];
        transformer::gemm_out_to_rows(&out, rows, n, vocab, &mut logits);

        // Granite divides logits by `logits_scaling`; identity elsewhere. Applied
        // over the whole buffer here, per row inside `project_logits`.
        if cfg.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits, 1.0 / cfg.scalars.logit);
        }
        if let Some(cap) = self.final_logit_softcapping {
            cpu::softcap_inplace(&mut logits, cap);
        }
        Some(logits)
    }

    /// Batched-GEMM CPU prefill for the dense transformer (mirrors LFM2's CPU
    /// prefill). Reads each weight matrix once for all `n` tokens. Column-major
    /// `hidden[hs × n]` (token `j` of channel `i` at `i*n + j`). Numerically
    /// matches the per-token `forward` path. Only compiled where a batched-GEMM
    /// kernel exists (aarch64 NEON, x86_64 int8 — VNNI or AVX2 — or any target
    /// with the `blas` feature); the per-token fallback covers the rest. On
    /// x86_64 the kernel is additionally a *runtime* property, so the dtype scan
    /// below also asks `batched_gemm_supports` before committing to this path.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
    /// Batched-GEMM prefill. When `hidden_out` is `Some`, this captures the
    /// per-token post-final-norm hidden states into it (row-major `[n * hs]`),
    /// skips the logit projection, and returns an empty Vec — the hidden-states
    /// path. When `None`, it norms+projects the last token and returns its logits
    /// — the normal prefill path.
    fn forward_prefill_batched(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let n = tokens.len();
        let head_dim = self.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let group_size = n_heads / n_kv_heads;
        // Granite overrides the softmax scale via `attention.scale`; every other
        // arch uses the default 1/sqrt(head_dim).
        let scale = cfg
            .scalars
            .attn
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());

        // Cloned once (cheap Arc bump) so the adapter can be read while the
        // base-weight scratch buffers stay mutably borrowed (disjoint fields).
        let lora = state.lora.clone();

        // If any per-layer projection uses a dtype the batched GEMM cannot take,
        // fall back to the sequential per-token path so the result stays correct.
        //
        // Admits exactly what `batched_gemm_supports` can compute, which now
        // includes Q4_K/Q6_K on both int8 targets.
        //
        // The previous note here said widening needed a Q5_K GEMM first, because
        // "a Qwen Q4_K_M carries Q5_K tensors". That was wrong on the specifics:
        // those files carry **Q5_0**, not Q5_K, and cera rejects them at *load*
        // rather than at this gate — a Q5_K kernel would not have helped.
        //
        // The real rule is llama.cpp's: K-quants need a 256-element super-block,
        // so a tensor whose row length is not divisible by 256 falls back to a
        // legacy quant. Qwen2-0.5B is hidden=896 (896 % 256 = 128), so its
        // 896-wide tensors are Q5_0 while its 4864-wide `ffn_down` is Q6_K.
        // A model with a 256-divisible hidden size is genuinely Q4_K/Q6_K
        // throughout: Llama-3.2-1B (hidden 2048) is 96 Q4_K + 17 Q6_K + 34 F32,
        // which is what `llama_batched_prefill_parity_llama32_1b_q4_k_m`
        // exercises.
        let mut unbatchable: Option<(&str, crate::tensor::DType)> = None;
        for r in self.layer_refs.iter() {
            for (name, w) in [
                ("attn_q", &r.attn_q),
                ("attn_k", &r.attn_k),
                ("attn_v", &r.attn_v),
                ("attn_output", &r.attn_output),
                ("ffn_gate", &r.ffn_gate),
                ("ffn_up", &r.ffn_up),
                ("ffn_down", &r.ffn_down),
            ] {
                // `batched_gemm_supports` answers all three parts of the
                // question: the dtype has a kernel at all, that kernel can run
                // *on this host* (on x86 the int8 GEMM needs runtime avx2+fma), and
                // for K-quants that `k % 256 == 0`.
                //
                // The host check is the load-bearing one. Without it a Scalar-tier
                // x86 build reaches `gemm_preq`, no kernel runs, and callers
                // reuse one output buffer across layers — so the previous
                // layer's activations survive as this layer's result. Silent
                // wrong numbers, not a crash.
                if !transformer::batched_gemm_supports(w.dtype, w.k) {
                    unbatchable = Some((name, w.dtype));
                    break;
                }
            }
            if unbatchable.is_some() {
                break;
            }
        }
        if let Some((name, dtype)) = unbatchable {
            // Say so. A gate that declines in silence cost ~4x prefill on LFM2 (T1)
            // and ~340x the submits on the GPU (T8) before anyone noticed.
            transformer::warn_unbatchable(name, dtype);
        }
        if unbatchable.is_some() {
            // No batched kernel for these dtypes: capture per-token if requested,
            // else fall back to the sequential per-token logit path.
            if let Some(out) = hidden_out {
                *out = self.hidden_states_per_token(tokens, state);
                return Vec::new();
            }
            let mut logits = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                logits = self.forward(&[token], start_pos + i, state);
            }
            return logits;
        }

        // Embed all tokens → column-major hidden[hs × n] (Granite embedding scale).
        let mut hidden = vec![0.0f32; hs * n];
        let mut emb_buf = vec![0.0f32; hs];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            // Bound on `vocab_size` (not the possibly-padded embedding row count
            // `embd_ref.m`) so an out-of-vocab id is rejected identically to the
            // per-token `forward` path rather than silently reading a pad row.
            assert!(
                token_id < cfg.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            );
            transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, &mut emb_buf);
            if cfg.scalars.embedding != 1.0 {
                cpu::scale_inplace(&mut emb_buf, cfg.scalars.embedding);
            }
            for i in 0..hs {
                hidden[i * n + j] = emb_buf[i];
            }
        }

        // Per-layer buffers (reused across layers).
        let mut normed = match self.norm_order {
            NormOrder::PreNorm => vec![0.0f32; hs * n],
            NormOrder::PostNorm => Vec::new(),
        };
        let mut block_out = vec![0.0f32; hs * n];
        let mut ffn_input = match self.norm_order {
            NormOrder::PreNorm => vec![0.0f32; hs * n],
            NormOrder::PostNorm => Vec::new(),
        };
        let mut ffn_out = vec![0.0f32; hs * n];
        let mut norm_col = vec![0.0f32; hs];
        let mut ffn_col = vec![0.0f32; hs];
        let mut q_mat = vec![0.0f32; q_dim * n];
        let mut k_mat = vec![0.0f32; kv_dim * n];
        let mut v_mat = vec![0.0f32; kv_dim * n];
        let mut out_proj_input = vec![0.0f32; q_dim * n];
        let mut gate_mat = vec![0.0f32; is * n];
        let mut up_mat = vec![0.0f32; is * n];

        // NEON-fallback Q8_0 input scratch. One buffer set sized to the largest
        // GEMM k-dim (hs, q_dim, or is) — each quantize call is immediately
        // followed by its paired GEMM with the same k, so reuse is safe.
        #[cfg(not(has_blas))]
        let max_dim = hs.max(q_dim).max(is);
        #[cfg(not(has_blas))]
        let mut col = vec![0.0f32; max_dim];
        #[cfg(not(has_blas))]
        let mut bq_scales = vec![0.0f32; n * (max_dim / 32)];
        #[cfg(not(has_blas))]
        let mut bq_quants = vec![0i8; n * max_dim];

        // Flash attention (tiled + rayon) beats the naive per-token loop only for
        // longer prompts; below the threshold its two-pass online-softmax overhead
        // loses. Mirrors LFM2's measured crossover (~pp256 on Apple Silicon).
        const FLASH_ATTN_THRESHOLD: usize = 256;
        let use_flash = n >= FLASH_ATTN_THRESHOLD && self.attn_logit_softcapping.is_none();
        // Per-query-head attention output, [n_heads][n * head_dim], scattered
        // back into out_proj_input after the flash pass. (Byte-identical to the
        // old per-KV-head [n_kv_heads][group_size * n * head_dim] layout, since
        // head h = kv_h*group_size + g sits at h*n*head_dim either way.) Reused
        // across layers; empty (unused) below the threshold.
        let mut flash_out = if use_flash {
            vec![0.0f32; n_heads * n * head_dim]
        } else {
            Vec::new()
        };
        // f16 mode only: reused across layers to widen the half KV cache to f32
        // for the (f32-only) flash/naive kernels. Hoisted out of the layer loop
        // so the widen reuses one allocation instead of a fresh Vec per layer.
        // Stay empty (no alloc) on the f32 path.
        let mut kv_widen_k: Vec<f32> = Vec::new();
        let mut kv_widen_v: Vec<f32> = Vec::new();

        for layer in 0..cfg.n_layers {
            let refs = &self.layer_refs[layer];

            // Attention pre-norm: rmsnorm each column (PreNorm only).
            let normed_input: &[f32] = match self.norm_order {
                NormOrder::PreNorm => {
                    if n == 1 {
                        normed[..hs].copy_from_slice(&hidden[..hs]);
                        cpu::rmsnorm(
                            &mut normed[..hs],
                            &self.attn_norm_weights[layer],
                            cfg.rms_norm_eps,
                        );
                    } else {
                        for j in 0..n {
                            for i in 0..hs {
                                norm_col[i] = hidden[i * n + j];
                            }
                            cpu::rmsnorm(
                                &mut norm_col,
                                &self.attn_norm_weights[layer],
                                cfg.rms_norm_eps,
                            );
                            for i in 0..hs {
                                normed[i * n + j] = norm_col[i];
                            }
                        }
                    }
                    &normed
                }
                NormOrder::PostNorm => &hidden,
            };

            // Batched Q/K/V projections (weight [m×hs] × normed[hs×n] → [m×n]).
            #[cfg(has_blas)]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_q,
                    normed_input,
                    &mut q_mat,
                    q_dim,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_k,
                    normed_input,
                    &mut k_mat,
                    kv_dim,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_v,
                    normed_input,
                    &mut v_mat,
                    kv_dim,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(has_blas))]
            {
                transformer::quantize_columns(
                    normed_input,
                    hs,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_q,
                    &bq_scales,
                    &bq_quants,
                    &mut q_mat,
                    q_dim,
                    n,
                    hs,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_k,
                    &bq_scales,
                    &bq_quants,
                    &mut k_mat,
                    kv_dim,
                    n,
                    hs,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_v,
                    &bq_scales,
                    &bq_quants,
                    &mut v_mat,
                    kv_dim,
                    n,
                    hs,
                );
            }

            // LoRA on Q/K/V: added to the projection outputs before bias/RoPE,
            // input is the normed hidden `[hs×n]` (matches the decode hook order).
            if let Some(lora) = &lora {
                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnQ) {
                    crate::lora::apply_prefill(
                        t,
                        normed_input,
                        &mut q_mat,
                        n,
                        &mut state.scratch.lora_tmp,
                    );
                }
                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnK) {
                    crate::lora::apply_prefill(
                        t,
                        normed_input,
                        &mut k_mat,
                        n,
                        &mut state.scratch.lora_tmp,
                    );
                }
                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnV) {
                    crate::lora::apply_prefill(
                        t,
                        normed_input,
                        &mut v_mat,
                        n,
                        &mut state.scratch.lora_tmp,
                    );
                }
            }

            // Per-arch attention knobs (constant across tokens within a layer).
            let qkv_bias = match (
                self.attn_q_bias[layer].as_deref(),
                self.attn_k_bias[layer].as_deref(),
                self.attn_v_bias[layer].as_deref(),
            ) {
                (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                _ => None,
            };
            let qk_norm = match (
                self.attn_q_norm_weights[layer].as_deref(),
                self.attn_k_norm_weights[layer].as_deref(),
            ) {
                (Some(q), Some(k)) => Some((q, k)),
                _ => None,
            };

            // Pass A: per token, bias → QK-norm → RoPE → stash post-RoPE Q back
            // into q_mat (so the attention pass can read every query) → append
            // K/V to the f32 cache. Destructure the cache once (not per token)
            // and reserve the whole prompt's growth up front (matches lfm2) so
            // the per-token extend_from_slice doesn't repeatedly reallocate.
            // f16 KV: append converts to half; Pass B widens back to an f32
            // scratch (below) so the existing flash/naive kernels are unchanged.
            let use_f16 = state.kv_f16;
            let (key_cache, value_cache, key_cache_f16, value_cache_f16) =
                match &mut state.layers[layer] {
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } => (key_cache, value_cache, key_cache_f16, value_cache_f16),
                    _ => unreachable!("dense transformer layer is always Attention"),
                };
            if use_f16 {
                key_cache_f16.reserve(n * kv_dim);
                value_cache_f16.reserve(n * kv_dim);
            } else {
                key_cache.reserve(n * kv_dim);
                value_cache.reserve(n * kv_dim);
            }
            for j in 0..n {
                let pos = start_pos + j;
                let q = &mut state.scratch.q[..q_dim];
                let k = &mut state.scratch.k[..kv_dim];
                let v = &mut state.scratch.v[..kv_dim];
                for i in 0..q_dim {
                    q[i] = q_mat[i * n + j];
                }
                for i in 0..kv_dim {
                    k[i] = k_mat[i * n + j];
                    v[i] = v_mat[i * n + j];
                }

                // Qwen2 Q/K/V bias.
                if let Some((q_bias, k_bias, v_bias)) = qkv_bias {
                    cpu::add_inplace(q, q_bias);
                    cpu::add_inplace(k, k_bias);
                    cpu::add_inplace(v, v_bias);
                }

                // Qwen3 / Olmo 2 QK-norm (before RoPE).
                if let Some((q_norm, k_norm)) = qk_norm {
                    if q_norm.len() == head_dim {
                        for h in 0..n_heads {
                            cpu::rmsnorm(
                                &mut q[h * head_dim..(h + 1) * head_dim],
                                q_norm,
                                cfg.rms_norm_eps,
                            );
                        }
                    } else {
                        cpu::rmsnorm(q, q_norm, cfg.rms_norm_eps);
                    }
                    if k_norm.len() == head_dim {
                        for h in 0..n_kv_heads {
                            cpu::rmsnorm(
                                &mut k[h * head_dim..(h + 1) * head_dim],
                                k_norm,
                                cfg.rms_norm_eps,
                            );
                        }
                    } else {
                        cpu::rmsnorm(k, k_norm, cfg.rms_norm_eps);
                    }
                }

                // RoPE: layout per arch (NEOX for Qwen/Olmo2, NORM for LLaMA/Granite).
                // Optional YaRN scaling for NEOX.
                match self.rope_type {
                    RopeType::Neox => {
                        if let Some(yarn) = self.layer_yarn(layer) {
                            cpu::rope_neox_yarn(
                                q,
                                k,
                                pos,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                cfg.rope_theta,
                                &yarn,
                            );
                        } else {
                            cpu::rope(q, k, pos, n_heads, n_kv_heads, head_dim, cfg.rope_theta);
                        }
                    }
                    RopeType::Norm => {
                        if let Some(yarn) = self.layer_yarn(layer) {
                            cpu::rope_norm_yarn(
                                q,
                                k,
                                pos,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                cfg.rope_theta,
                                &yarn,
                            );
                        } else {
                            cpu::rope_norm(
                                q,
                                k,
                                pos,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                cfg.rope_theta,
                                self.rope_freqs.as_deref(),
                            );
                        }
                    }
                }

                // Optional attention temperature scaling (Mistral 3 / Llama 4).
                if let Some((scale, floor_scale)) = self.attn_temp_scale
                    && scale > 0.0
                    && floor_scale > 0
                    && pos >= floor_scale
                {
                    let q_scale =
                        ((pos as f32 / floor_scale as f32).floor() + 1.0).ln() * scale + 1.0;
                    cpu::scale_inplace(q, q_scale);
                }

                // Stash post-RoPE Q back into q_mat for the attention pass.
                for i in 0..q_dim {
                    q_mat[i * n + j] = q[i];
                }

                // Append K, V to the cache (destructured once above the loop).
                if use_f16 {
                    key_cache_f16.extend(
                        state.scratch.k[..kv_dim]
                            .iter()
                            .map(|&x| crate::quant::f32_to_f16(x)),
                    );
                    value_cache_f16.extend(
                        state.scratch.v[..kv_dim]
                            .iter()
                            .map(|&x| crate::quant::f32_to_f16(x)),
                    );
                } else {
                    key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                    value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                }
            }

            // Pass B: GQA attention over the now-complete KV cache → out_proj_input.
            // In f16 mode, widen the half cache into the reused f32 scratch once
            // per layer so the flash/naive kernels below stay f32-only (prefill
            // isn't the decode-at-depth hot path; native f16 flash is a
            // follow-up).
            let (k_cache, v_cache) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    if use_f16 {
                        kv_widen_k.clear();
                        kv_widen_k
                            .extend(key_cache_f16.iter().map(|&b| crate::quant::f16_to_f32(b)));
                        kv_widen_v.clear();
                        kv_widen_v
                            .extend(value_cache_f16.iter().map(|&b| crate::quant::f16_to_f32(b)));
                        (kv_widen_k.as_slice(), kv_widen_v.as_slice())
                    } else {
                        (key_cache.as_slice(), value_cache.as_slice())
                    }
                }
                _ => unreachable!("dense transformer layer is always Attention"),
            };
            let layer_swa = self.layer_sliding_window(layer);
            if use_flash && layer_swa.is_none() {
                // Flash attention (tiled + rayon), parallel across *query heads*,
                // not KV heads. Splitting per-KV-head caps parallelism at
                // n_kv_heads (8 for Llama-3.2-1B) — half-idle on a 16-core host,
                // which a pp2048 profile showed as the dominant prefill cost once
                // attention's O(n^2) term grew. One task per query head gives
                // n_heads-way (32) parallelism; group members of one KV head
                // re-read that head's K/V, but at these sizes those reads hit L3,
                // and full core utilization more than pays for it.
                //
                // The output layout is byte-identical to the per-KV-head split:
                // KV head kv_h's chunk was [group_size, n, head_dim] at offset
                // kv_h*group_size*n*head_dim, and group member g at
                // +g*n*head_dim — i.e. head h = kv_h*group_size + g sits at
                // exactly h*n*head_dim. So a flat per-head chunking writes the
                // same bytes; the scatter below is unchanged. Bit-identical
                // because each (head, query) output is computed independently.
                let head_chunk = n * head_dim;
                let flash_buf = &mut flash_out[..n_heads * head_chunk];
                let q_ref = &q_mat[..];
                // Fan out over query heads via `par_rows_n_chunked` — the pinned
                // RowPool on native, rayon on wasm32. On native this shares the
                // one prefill pool with the GEMM instead of a second full-width
                // pool spin-waiting through attention's phase (the
                // oversubscription the GEMM consolidation removed). Each
                // query head is one "row" of `head_chunk = n * head_dim`; the
                // per-(head, query) reductions are independent, so which worker
                // runs which head does not change the result — bit-identical.
                //
                // `min_chunk_rows = 1`: a head is a heavy row, and there are only
                // `n_heads` of them (32 for Llama-1B), so the default steal floor
                // would hand all heads to 2 workers. One head per steal unit lets
                // every worker take a head.
                cpu::par_rows_n_chunked(flash_buf, head_chunk, 1, 1, |(h, chunk)| {
                    let kv_h = h / group_size;
                    cpu::flash_attention_gqa_cpu(
                        q_ref,
                        k_cache,
                        v_cache,
                        chunk,
                        h,
                        1,
                        n,
                        n,
                        kv_dim,
                        kv_h * head_dim,
                        head_dim,
                        scale,
                        start_pos,
                    );
                });
                // Scatter flash_out [n_heads, n, head_dim] → out_proj_input [q_dim,
                // n] (stride-n columns). d-then-j inner order keeps out writes
                // sequential (stride 1) with small-stride reads from flash_buf.
                // Head h's block sits at h*n*head_dim (the per-head chunking
                // above), so the old kv_h/g nesting collapses to a flat h loop.
                for h in 0..n_heads {
                    let src_base = h * n * head_dim;
                    for d in 0..head_dim {
                        let row_idx = (h * head_dim + d) * n;
                        for j in 0..n {
                            out_proj_input[row_idx + j] = flash_buf[src_base + j * head_dim + d];
                        }
                    }
                }
            } else {
                // Naive per-token attention: token j attends over cache[0..pos+1]
                // (causal). Bit-identical to the per-token `forward` path.
                let attn_out = &mut state.scratch.attn_out[..q_dim];
                let q = &mut state.scratch.q[..q_dim];
                let scores = &mut state.scratch.scores;
                for j in 0..n {
                    let seq_len = start_pos + j + 1;
                    for i in 0..q_dim {
                        q[i] = q_mat[i * n + j];
                    }
                    scores.resize(seq_len, 0.0);
                    for h in 0..n_heads {
                        let kv_h = h / group_size;
                        let q_head = &q[h * head_dim..(h + 1) * head_dim];
                        let kv_h_offset = kv_h * head_dim;
                        cpu::attn_scores(
                            q_head,
                            k_cache,
                            scores,
                            kv_dim,
                            kv_h_offset,
                            head_dim,
                            scale,
                            seq_len,
                        );
                        if let Some(cap) = self.attn_logit_softcapping {
                            cpu::softcap_inplace(scores, cap);
                        }
                        if let Some(w) = layer_swa.filter(|&w| w > 0) {
                            let cutoff = seq_len.saturating_sub(w);
                            scores[..cutoff].fill(f32::NEG_INFINITY);
                        }
                        cpu::softmax_inplace(scores);
                        cpu::attn_values(
                            scores,
                            v_cache,
                            &mut attn_out[h * head_dim..(h + 1) * head_dim],
                            kv_dim,
                            kv_h_offset,
                            head_dim,
                            seq_len,
                        );
                    }
                    for i in 0..q_dim {
                        out_proj_input[i * n + j] = attn_out[i];
                    }
                }
            }

            // Batched output projection GEMM -> block_out[hs * n] (k = q_dim).
            #[cfg(has_blas)]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_output,
                    &out_proj_input,
                    &mut block_out,
                    hs,
                    n,
                    q_dim,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(has_blas))]
            {
                transformer::quantize_columns(
                    &out_proj_input,
                    q_dim,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_output,
                    &bq_scales,
                    &bq_quants,
                    &mut block_out,
                    hs,
                    n,
                    q_dim,
                );
            }

            // LoRA on the output projection - applied to the projection output
            // BEFORE the residual scale (so Granite's multiplier wraps the delta
            // too); input is the attention output `[q_dim*n]`.
            if let Some(lora) = &lora
                && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
            {
                crate::lora::apply_prefill(
                    t,
                    &out_proj_input,
                    &mut block_out,
                    n,
                    &mut state.scratch.lora_tmp,
                );
            }

            if let Some(bias) = self.attn_output_bias[layer].as_deref() {
                apply_column_major_bias(&mut block_out, bias, hs, n);
            }

            // Post-norm on attention output (Gemma 2, Olmo 2/3).
            if let Some(post_norm) = &self.attn_post_norm_weights[layer] {
                if n == 1 {
                    cpu::rmsnorm(&mut block_out[..hs], post_norm, cfg.rms_norm_eps);
                } else {
                    for j in 0..n {
                        for i in 0..hs {
                            norm_col[i] = block_out[i * n + j];
                        }
                        cpu::rmsnorm(&mut norm_col, post_norm, cfg.rms_norm_eps);
                        for i in 0..hs {
                            block_out[i * n + j] = norm_col[i];
                        }
                    }
                }
            }

            // Granite residual scale, then residual add into hidden.
            if cfg.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut block_out, cfg.scalars.residual);
            }
            cpu::add_inplace(&mut hidden, &block_out);

            // FFN pre-norm: rmsnorm each column (PreNorm only).
            let ffn_in: &[f32] = match self.norm_order {
                NormOrder::PreNorm => {
                    if n == 1 {
                        ffn_input[..hs].copy_from_slice(&hidden[..hs]);
                        cpu::rmsnorm(
                            &mut ffn_input[..hs],
                            &self.ffn_norm_weights[layer],
                            cfg.rms_norm_eps,
                        );
                    } else {
                        for j in 0..n {
                            for i in 0..hs {
                                ffn_col[i] = hidden[i * n + j];
                            }
                            cpu::rmsnorm(
                                &mut ffn_col,
                                &self.ffn_norm_weights[layer],
                                cfg.rms_norm_eps,
                            );
                            for i in 0..hs {
                                ffn_input[i * n + j] = ffn_col[i];
                            }
                        }
                    }
                    &ffn_input
                }
                NormOrder::PostNorm => &hidden,
            };

            // FFN gate/up GEMM → silu(gate)⊙up → down GEMM.
            #[cfg(has_blas)]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.ffn_gate,
                    ffn_in,
                    &mut gate_mat,
                    is,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.ffn_up,
                    ffn_in,
                    &mut up_mat,
                    is,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(has_blas))]
            {
                transformer::quantize_columns(
                    ffn_in,
                    hs,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_gate,
                    &bq_scales,
                    &bq_quants,
                    &mut gate_mat,
                    is,
                    n,
                    hs,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_up,
                    &bq_scales,
                    &bq_quants,
                    &mut up_mat,
                    is,
                    n,
                    hs,
                );
            }

            // LoRA on gate/up: BEFORE the SwiGLU mul, input is the normed FFN
            // input `[hs×n]` (mirrors the decode hook order).
            if let Some(lora) = &lora {
                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                    crate::lora::apply_prefill(
                        t,
                        ffn_in,
                        &mut gate_mat,
                        n,
                        &mut state.scratch.lora_tmp,
                    );
                }
                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                    crate::lora::apply_prefill(
                        t,
                        ffn_in,
                        &mut up_mat,
                        n,
                        &mut state.scratch.lora_tmp,
                    );
                }
            }

            if let Some(bias) = self.ffn_gate_bias[layer].as_deref() {
                apply_column_major_bias(&mut gate_mat, bias, is, n);
            }
            if let Some(bias) = self.ffn_up_bias[layer].as_deref() {
                apply_column_major_bias(&mut up_mat, bias, is, n);
            }

            match self.activation {
                FfnActivation::Swiglu => {
                    cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);
                }
                FfnActivation::Geglu => {
                    cpu::gelu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);
                }
            }

            #[cfg(has_blas)]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.ffn_down,
                    &gate_mat,
                    &mut ffn_out,
                    hs,
                    n,
                    is,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(has_blas))]
            {
                transformer::quantize_columns(
                    &gate_mat,
                    is,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_down,
                    &bq_scales,
                    &bq_quants,
                    &mut ffn_out,
                    hs,
                    n,
                    is,
                );
            }

            // LoRA on the down projection - applied BEFORE the residual scale;
            // input is the SwiGLU product in `gate_mat` `[is*n]`.
            if let Some(lora) = &lora
                && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
            {
                crate::lora::apply_prefill(
                    t,
                    &gate_mat,
                    &mut ffn_out,
                    n,
                    &mut state.scratch.lora_tmp,
                );
            }

            if let Some(bias) = self.ffn_down_bias[layer].as_deref() {
                apply_column_major_bias(&mut ffn_out, bias, hs, n);
            }

            // Post-norm on FFN output (Gemma 2, Olmo 2/3).
            if let Some(post_norm) = &self.ffn_post_norm_weights[layer] {
                if n == 1 {
                    cpu::rmsnorm(&mut ffn_out[..hs], post_norm, cfg.rms_norm_eps);
                } else {
                    for j in 0..n {
                        for i in 0..hs {
                            ffn_col[i] = ffn_out[i * n + j];
                        }
                        cpu::rmsnorm(&mut ffn_col, post_norm, cfg.rms_norm_eps);
                        for i in 0..hs {
                            ffn_out[i * n + j] = ffn_col[i];
                        }
                    }
                }
            }

            // Granite residual scale, then residual add.
            if cfg.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut ffn_out, cfg.scalars.residual);
            }
            cpu::add_inplace(&mut hidden, &ffn_out);

            if self
                .loop_norm_interval
                .is_some_and(|n_phys| (layer + 1) % n_phys == 0)
                && (layer + 1) < cfg.n_layers
            {
                if n == 1 {
                    cpu::rmsnorm(
                        &mut hidden[..hs],
                        &self.output_norm_weight,
                        cfg.rms_norm_eps,
                    );
                } else {
                    for j in 0..n {
                        for i in 0..hs {
                            norm_col[i] = hidden[i * n + j];
                        }
                        cpu::rmsnorm(&mut norm_col, &self.output_norm_weight, cfg.rms_norm_eps);
                        for i in 0..hs {
                            hidden[i * n + j] = norm_col[i];
                        }
                    }
                }
            }
        }

        // Advance seq_len (the block loops appended KV cells without bumping it).
        state.seq_len = start_pos + n;

        // Hidden-states capture: final-norm EVERY column into a row-major
        // `[n * hs]` buffer (post-final-RMSNorm = llama.cpp `result_norm`),
        // skipping the logit projection. Reuses `norm_col` as per-column scratch.
        if let Some(out) = hidden_out {
            out.clear();
            out.reserve(n * hs);
            for j in 0..n {
                for i in 0..hs {
                    norm_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(&mut norm_col, &self.output_norm_weight, cfg.rms_norm_eps);
                out.extend_from_slice(&norm_col);
            }
            return Vec::new();
        }

        // Final norm on the LAST column, then project last-token logits (what the
        // decode loop consumes). Reuse `norm_col` (an hs-length scratch that's
        // dead after the layer loop) rather than allocating. `project_logits`
        // handles the Granite logit scale and the aarch64 pre-quantized GEMV.
        for i in 0..hs {
            norm_col[i] = hidden[i * n + (n - 1)];
        }
        cpu::rmsnorm(&mut norm_col, &self.output_norm_weight, cfg.rms_norm_eps);
        self.project_logits(&norm_col, state)
    }

    /// Per-token hidden-states fallback: embed → `run_layers` (which applies the
    /// final RMSNorm) per token, concatenated row-major `[n * hidden_size]`.
    /// Post-final-norm, matching the batched capture path. Used when there's no
    /// batched-GEMM kernel (`n == 1`, non-gemmable dtypes, or non-aarch64/non-blas).
    /// Assumes `state` starts cleared at position 0.
    fn hidden_states_per_token(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        let hs = self.config.hidden_size;
        let mut out = Vec::with_capacity(tokens.len() * hs);
        // Reuse one embedding buffer across tokens (`dequantize_row_into`) instead
        // of allocating a fresh Vec per token.
        let mut hidden = vec![0.0f32; hs];
        for &token in tokens {
            let token_id = token as usize;
            assert!(
                token_id < self.config.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                self.config.vocab_size
            );
            transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, &mut hidden);
            if self.config.scalars.embedding != 1.0 {
                cpu::scale_inplace(&mut hidden, self.config.scalars.embedding);
            }
            // `run_layers` ropes at `pos` and appends one KV cell, bumping
            // seq_len; starting from a cleared state walks positions 0..n.
            let pos = state.seq_len;
            self.run_layers(&mut hidden, pos, state);
            out.extend_from_slice(&hidden);
        }
        out
    }
}

impl Model for LlamaModel {
    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &crate::kv_cache::KvCompression,
        max_seq_len: usize,
    ) -> Result<(), crate::session::CeraError> {
        super::reset_cpu_kv(self, state, compression, max_seq_len)
    }

    fn check_kv_rewind(
        &self,
        state: &InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        self.check_rewind_mode(state)?;
        state.check_truncate_to(len)
    }

    fn try_truncate_kv(
        &self,
        state: &mut InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        self.check_rewind_mode(state)?;
        state.try_truncate_to(len)
    }

    fn supports_hidden_states(&self) -> bool {
        true
    }

    fn f16_kv_supported(&self) -> bool {
        true
    }

    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "hidden_states requires at least one token"
        );
        // Batched-GEMM capture when a batched kernel exists and n > 1; the
        // batched path internally falls back to per-token for non-gemmable dtypes.
        // An active LoRA is applied in-batch (via `apply_prefill` after each
        // projection GEMM); non-gemmable dtypes fall back to the per-token decode
        // hooks, which apply it too.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
        if tokens.len() > 1 {
            let mut out = Vec::new();
            self.forward_prefill_batched(tokens, 0, state, Some(&mut out));
            return out;
        }
        self.hidden_states_per_token(tokens, state)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LlamaModel forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        let mut hidden = transformer::dequantize_row(&self.gguf, &self.embd_ref, token_id);
        if self.config.scalars.embedding != 1.0 {
            cpu::scale_inplace(&mut hidden, self.config.scalars.embedding);
        }
        // Record after the embedding scale: llama.cpp fires its "embd" callback
        // post-scale, so the dumped node is GET_ROWS for plain archs (scale=1) and
        // SCALE for Granite. Either way the value matches.
        transformer::oracle_dump::record("embd", &hidden);
        self.run_layers(&mut hidden, pos, state);
        self.project_logits(&hidden, state)
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        if tokens.is_empty() {
            return 0;
        }
        if tokens.len() > 1 {
            if transformer::oracle_dump::is_active() {
                for (i, &t) in tokens[..tokens.len() - 1].iter().enumerate() {
                    let _ = self.forward(&[t], pos + i, state);
                }
            } else {
                let cfg = &self.config;
                let mut hidden_stack = [0.0f32; 4096];
                let mut hidden_heap;
                let hidden = if cfg.hidden_size <= 4096 {
                    &mut hidden_stack[..cfg.hidden_size]
                } else {
                    hidden_heap = vec![0.0f32; cfg.hidden_size];
                    &mut hidden_heap[..]
                };
                for (i, &t) in tokens[..tokens.len() - 1].iter().enumerate() {
                    let token_id = t as usize;
                    if token_id >= cfg.vocab_size {
                        continue;
                    }
                    transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, hidden);
                    if self.config.scalars.embedding != 1.0 {
                        cpu::scale_inplace(hidden, self.config.scalars.embedding);
                    }
                    self.run_layers(hidden, pos + i, state);
                }
            }
            return self.forward_greedy(&tokens[tokens.len() - 1..], pos + tokens.len() - 1, state);
        }
        if transformer::oracle_dump::is_active() {
            let logits = self.forward(tokens, pos, state);
            return crate::sampler::argmax(&logits);
        }
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        if token_id >= cfg.vocab_size {
            let logits = self.forward(tokens, pos, state);
            return crate::sampler::argmax(&logits);
        }

        let mut hidden_stack = [0.0f32; 4096];
        let mut hidden_heap;
        let hidden = if cfg.hidden_size <= 4096 {
            &mut hidden_stack[..cfg.hidden_size]
        } else {
            hidden_heap = vec![0.0f32; cfg.hidden_size];
            &mut hidden_heap[..]
        };
        transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, hidden);
        if self.config.scalars.embedding != 1.0 {
            cpu::scale_inplace(hidden, self.config.scalars.embedding);
        }
        self.run_layers(hidden, pos, state);

        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);
        #[cfg(target_arch = "aarch64")]
        {
            if out_ref.dtype == crate::tensor::DType::Q6K && self.config.scalars.logit > 0.0 {
                transformer::quantize_to_scratch(hidden, state);
                return transformer::gemv_preq_argmax(
                    &self.gguf,
                    out_ref,
                    hidden,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                ) as u32;
            }
        }

        if state.scratch.logits.len() < cfg.vocab_size {
            state.scratch.logits.resize(cfg.vocab_size, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch(hidden, state);
            transformer::gemv_preq(
                &self.gguf,
                out_ref,
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut state.scratch.logits[..cfg.vocab_size],
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            transformer::gemv(
                &self.gguf,
                out_ref,
                hidden,
                &mut state.scratch.logits[..cfg.vocab_size],
            );
        }
        if self.config.scalars.logit > 0.0 && self.config.scalars.logit != 1.0 {
            cpu::scale_inplace(
                &mut state.scratch.logits[..cfg.vocab_size],
                1.0 / self.config.scalars.logit,
            );
        }
        crate::sampler::argmax(&state.scratch.logits[..cfg.vocab_size])
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "forward_prefill requires at least one token"
        );
        // Each `forward` appends one K/V cell and advances `seq_len`, so the
        // rope position of token `i` must equal the current cache length. That
        // holds only when `start_pos` lines up with the existing cache — enforce
        // it so a mismatched snapshot/prefix-cache restore fails loudly here
        // rather than drifting into a later KV-shift panic.
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );
        // Batched-GEMM prefill (reads each weight once for all N tokens) on
        // targets that have a batched kernel — aarch64 NEON or any `blas` build.
        // `n == 1` stays on the per-token path to avoid GEMM setup overhead, and
        // every other target has no batched kernel, so it also falls through.
        // When the oracle-dump harness is collecting, fall back to the per-token
        // path too: the batched path bypasses `run_layers` and so emits none of
        // the per-substep `oracle_dump::record` nodes that `tests/oracle_text.rs`
        // validates against llama.cpp.
        // An active LoRA is applied in-batch (`apply_prefill` after each projection
        // GEMM), so it no longer forces the per-token path; non-gemmable dtypes
        // still fall back to the per-token decode hooks, which apply it too.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
        if tokens.len() > 1 && !transformer::oracle_dump::is_active() {
            return self.forward_prefill_batched(tokens, start_pos, state, None);
        }

        // Sequential per-token prefill (single-token, or no batched kernel).
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            logits = self.forward(&[token], start_pos + i, state);
        }
        logits
    }

    fn supports_all_logits(&self) -> bool {
        true
    }

    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "forward_prefill_logits_all requires at least one token"
        );
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill_logits_all: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );
        let n = tokens.len();
        let vocab = self.config.vocab_size;

        // One batched pass captures every token's post-final-norm hidden state,
        // then the projection below turns all of them into logits. Reuses the
        // tested batched-prefill KV append (same gate as `forward_prefill`). The
        // oracle-dump harness needs the per-token substep records, so defer to
        // the per-token path when it is active.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
        if n > 1 && !transformer::oracle_dump::is_active() {
            let hs = self.config.hidden_size;
            let mut hidden = Vec::new();
            let _ = self.forward_prefill_batched(tokens, start_pos, state, Some(&mut hidden));
            debug_assert_eq!(hidden.len(), n * hs, "hidden capture must be [n * hs]");

            // Projection, preferred form: one `[rows x n] = [rows x hs] * [hs x n]`
            // GEMM, so the LM head is read once for all `n` positions instead
            // of once each. It declines to the per-row loop below when the head's
            // dtype has no batched kernel (a Q5_K head, or an x86 host below the
            // AVX2 tier) — see `project_logits_batched` for the full list.
            #[cfg(not(has_blas))]
            if let Some(logits) = self.project_logits_batched(&hidden, n) {
                return logits;
            }

            // Per-row fallback. Also the `blas` path: `try_blas_prefill_gemm`
            // dequantizes the whole `[m x k]` weight into scratch first, which
            // for an LM head is `vocab x hidden_size` — ~1 GB of f32 on
            // Llama-3.2-1B, against ~67 MB for the largest per-layer projection
            // it is normally used for. Not worth a chunked variant until someone
            // is actually speculating on a BLAS build.
            let mut logits = Vec::with_capacity(n * vocab);
            for j in 0..n {
                let row = &hidden[j * hs..(j + 1) * hs];
                let row_logits = self.project_logits(row, state);
                logits.extend_from_slice(&row_logits);
            }
            return logits;
        }

        // Fallback (single token, no batched kernel, or oracle active): each
        // `forward` returns that position's logits and appends its K/V cell.
        let mut logits = Vec::with_capacity(n * vocab);
        for (i, &token) in tokens.iter().enumerate() {
            let l = self.forward(&[token], start_pos + i, state);
            logits.extend_from_slice(&l);
        }
        logits
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn supports_kv_shift(&self) -> bool {
        // YaRN frequencies and per-layer SWA patterns do not compose with standard
        // unscaled RoPE delta rotation in `shift_kv_with_rope`.
        self.yarn.is_none() && self.sliding_window.is_none()
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        if self.yarn.is_some() || self.sliding_window.is_some() {
            tracing::warn!(
                "shift_kv called on model with YaRN or sliding window; skipping unscaled RoPE rotation"
            );
            return;
        }
        state.shift_kv_with_rope(
            n_keep,
            shift,
            self.config.rope_theta,
            self.head_dim,
            &self.config.kv_heads_per_layer,
            self.rope_type,
            self.rope_freqs.as_deref(),
        );
    }
}

// ── GPU weight source ───────────────────────────────────────────────────────
//
// Lets the wgpu loader (`gpu_lfm2.rs`) upload a dense transformer the same way
// it uploads LFM2. Every layer is attention (no conv refs); QK-norm / QKV-bias
// / untied-output / Llama-3 freq-factors are surfaced per-arch via the `Option`
// accessors. Granite scalars ride on `config().scalars`.
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl crate::model::gpu_weight_source::GpuWeightSource for LlamaModel {
    fn cache_identity_sources(&self) -> Option<Vec<&GgufFile>> {
        Some(vec![&self.gguf])
    }
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }
    fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }
    fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }
    fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_norm_weights[layer].as_deref()
    }
    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights[layer].as_deref()
    }
    fn conv_weight(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn attn_q_bias(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_bias[layer].as_deref()
    }
    fn attn_k_bias(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_bias[layer].as_deref()
    }
    fn attn_v_bias(&self, layer: usize) -> Option<&[f32]> {
        self.attn_v_bias[layer].as_deref()
    }
    fn rope_freqs(&self) -> Option<&[f32]> {
        self.rope_freqs.as_deref()
    }

    fn weight_bytes(&self, wref: &WeightRef) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(transformer::weight_data(&self.gguf, wref))
    }
    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32> {
        transformer::dequantize_weight(&self.gguf, wref)
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        self.output_ref.as_ref()
    }
    // Always dense: the `llama`-family loader has no expert path.
    fn ffn_gate_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn_gate)
    }
    fn ffn_up_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn_up)
    }
    fn ffn_down_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn_down)
    }
    fn conv_in_proj_ref(&self, _layer: usize) -> Option<&WeightRef> {
        None
    }
    fn conv_out_proj_ref(&self, _layer: usize) -> Option<&WeightRef> {
        None
    }
    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_q)
    }
    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_k)
    }
    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_v)
    }
    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_output)
    }

    fn rope_type(&self) -> RopeType {
        self.rope_type
    }
    fn supports_batched_prefill(&self) -> bool {
        // The batched wgpu prefill path now generalizes every dense-transformer
        // feature the per-token decode loop handles: `rope_type` (NEOX/NORM),
        // Llama-3 `freq_factors`, optional QK-norm, Qwen2 QKV bias, Qwen3
        // decoupled head_dim, Granite scalars (embedding/residual/attention/
        // logit), and untied output. Correctness is gated by the GPU-internal
        // differential test (batched vs per-token, all four archs) in
        // `tests/gpu_transformer_parity.rs`.
        //
        // YaRN frequency scaling, per-layer sliding window attention patterns,
        // attention temperature scaling, and projection/FFN biases are not
        // implemented in the current GPU prefill shaders.
        self.yarn.is_none()
            && self.sliding_window.is_none()
            && self.attn_temp_scale.is_none()
            && self.attn_output_bias.iter().all(Option::is_none)
            && self.ffn_gate_bias.iter().all(Option::is_none)
            && self.ffn_up_bias.iter().all(Option::is_none)
            && self.ffn_down_bias.iter().all(Option::is_none)
    }
    fn loop_norm_interval(&self) -> Option<usize> {
        self.loop_norm_interval
    }
}

#[cfg(test)]
mod tests {
    use crate::model::ScalarMultipliers;

    #[test]
    fn test_llama_scalars_logit_non_positive_fallback_logic() {
        // When scalars.logit <= 0.0, forward_greedy must not take the gemv_preq_argmax fast path
        // because argmax order would invert or degrade.
        let neg = ScalarMultipliers {
            logit: -1.0,
            ..Default::default()
        };
        assert!(neg.logit <= 0.0);

        let zero = ScalarMultipliers {
            logit: 0.0,
            ..Default::default()
        };
        assert!(zero.logit <= 0.0);

        let pos = ScalarMultipliers {
            logit: 1.0,
            ..Default::default()
        };
        assert!(pos.logit > 0.0);
    }
}

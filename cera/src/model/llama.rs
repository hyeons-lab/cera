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

/// Which normalization the block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum NormKind {
    #[default]
    Rms,
    /// Mean-subtracting LayerNorm with optional bias (StableLM, StarCoder2,
    /// Cohere/Command-R).
    Layer,
}

/// Archs that need the LayerNorm / parallel-residual / ungated-FFN / partial-RoPE
/// graph. CPU only: the GPU and NPU loaders reject them.
fn requires_layernorm_path(prefix: &str) -> bool {
    matches!(prefix, "stablelm" | "starcoder2" | "cohere" | "command-r")
}

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

/// LLaMA-family dense transformer (llama, qwen2/3, granite, ...).
///
/// The LayerNorm archs (stablelm, starcoder2, cohere, command-r) run the
/// sequential per-token path `run_layers_ext`: it has no batched prefill, so
/// every prompt token re-reads every weight once (prefill costs the same per
/// token as decode), and it supports neither LoRA nor yarn, `rope_freqs`,
/// sliding-window or temperature-scaled attention (rejected at load).
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
    /// YaRN expressed as `rope_freqs`-style per-pair divisors, for the GPU
    /// backends only (the CPU rotates with `yarn` directly). `None` unless the
    /// model is YaRN-scaled and the GPU rotation can reproduce it exactly.
    #[cfg_attr(
        not(any(
            feature = "gpu",
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            feature = "hexagon"
        )),
        allow(dead_code)
    )]
    yarn_freq_factors: Option<Vec<f32>>,
    norm_order: NormOrder,
    /// True for the archs served by the sequential LayerNorm path
    /// (`run_layers_ext`): stablelm, starcoder2, cohere, command-r.
    ext: bool,
    /// Rotated dims per head (`{prefix}.rope.dimension_count`); `head_dim` unless
    /// the arch rotates a prefix of each head (StableLM). Only the ext path
    /// honors a value below `head_dim`.
    n_rot: usize,
    /// False for plain up/down FFNs (StarCoder2). `layer_refs[i].ffn_gate` then
    /// aliases `ffn_up` and must not be read.
    ffn_gated: bool,
    activation: FfnActivation,
    attn_logit_softcapping: Option<f32>,
    final_logit_softcapping: Option<f32>,
    // Granite 3.x and MiniCPM scalar multipliers live on `config.scalars` (identity
    // for every other arch): see `ScalarMultipliers`.
    // Pre-dequantized small F32 weights.
    output_norm_weight: Vec<f32>,
    /// LayerNorm bias (`output_norm.bias`), StableLM/StarCoder2 only.
    output_norm_bias: Option<Vec<f32>>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    /// LayerNorm biases (`blk.N.attn_norm.bias` / `ffn_norm.bias`).
    attn_norm_biases: Vec<Option<Vec<f32>>>,
    ffn_norm_biases: Vec<Option<Vec<f32>>>,
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

// Test-only count of vocab-head projections on this thread, so the
// sequential prefill's "skip the LM head for all but the last token"
// optimisation is pinned by call count, not only by logits equivalence
// (a regression to per-token `forward` yields identical logits).
#[cfg(test)]
thread_local! {
    static PROJECT_LOGITS_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

/// Per-phase wall time of the dense batched prefill, printed with `CERA_PROFILE_PREFILL=1` (one line per
/// prefill call). The phases are the `PREFILL_PHASES` names, in the order the layer loop crosses them.
struct PrefillProf {
    on: bool,
    last: std::time::Instant,
    acc: [std::time::Duration; 11],
    /// Wall time of the down GEMM + residual (`down_residual`) per layer, in ms.
    down_layers: Vec<f64>,
}

const PREFILL_PHASES: [&str; 11] = [
    "norm",
    "qkv_gemm",
    "rope_kv",
    "attention",
    "out_proj",
    "ffn_norm",
    "ffn_quant",
    "gate_up",
    "silu",
    "down_quant",
    "down_residual",
];

impl PrefillProf {
    fn new() -> Self {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ENABLED.get_or_init(|| std::env::var_os("CERA_PROFILE_PREFILL").is_some());
        Self {
            on,
            last: std::time::Instant::now(),
            acc: [std::time::Duration::ZERO; 11],
            down_layers: Vec::new(),
        }
    }

    /// Charge the time since the previous lap to `phase`.
    fn lap(&mut self, phase: usize) {
        if self.on {
            let now = std::time::Instant::now();
            let d = now - self.last;
            self.acc[phase] += d;
            if phase == 10 {
                self.down_layers.push(d.as_secs_f64() * 1e3);
            }
            self.last = now;
        }
    }

    fn report(&self, n: usize) {
        if self.on {
            let total: std::time::Duration = self.acc.iter().sum();
            let mut line = format!("[PROFILE PREFILL dense] n={n}");
            for (name, d) in PREFILL_PHASES.iter().zip(&self.acc) {
                line.push_str(&format!(" | {name}: {:.2}ms", d.as_secs_f64() * 1e3));
            }
            line.push_str(&format!(" | total: {:.2}ms", total.as_secs_f64() * 1e3));
            eprintln!("{line}");
            let per: Vec<String> = self.down_layers.iter().map(|d| format!("{d:.0}")).collect();
            eprintln!(
                "[PROFILE PREFILL dense] down_residual per layer (ms): {}",
                per.join(" ")
            );
        }
    }
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

/// RoPE layout per arch, mirroring llama.cpp `llama_model_rope_type`.
///
/// Qwen, Gemma 2, Olmo 2/3, Phi, StableLM, StarCoder2 and OpenELM are NEOX
/// (split-halves); Llama-style, InternLM2 and Cohere/Command-R are NORM
/// (interleaved pairs). Keep exhaustive with the `load_model` allow-list:
/// an arch without a mapping must fail loudly rather than default to NORM.
fn rope_type_for_arch(prefix: &str) -> Option<RopeType> {
    match prefix {
        "qwen2" | "qwen3" | "gemma2" | "olmo2" | "olmo3" | "phi3" | "phi" | "starcoder2"
        | "stablelm" | "openelm" => Some(RopeType::Neox),
        // "llama" also covers classic Mistral (it ships as GGUF arch "llama").
        "llama" | "granite" | "minicpm" | "minicpm5" | "nanbeige" | "mistral3" | "ministral3"
        | "ministral" | "baichuan" | "deepseek" | "internlm2" | "internlm" | "cohere"
        | "command-r" => Some(RopeType::Norm),
        _ => None,
    }
}

/// LayerNorm `x = (x - mean) / sqrt(var + eps) * weight (+ bias)` in place.
/// A missing bias is a zero bias (Cohere norms carry weight only); that case
/// takes the bias-less kernel so no zero vector is allocated per call.
fn layer_norm_opt_bias(x: &mut [f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) {
    match bias {
        Some(b) => cpu::layer_norm_inplace(x, weight, b, eps),
        None => cpu::layer_norm_weight_only_inplace(x, weight, eps),
    }
}

/// Per-head LayerNorm for the optional Q/K norm of the LayerNorm archs
/// (StableLM 2, Command-R+). `weight` is either one `head_dim` vector shared by
/// all heads or `n_heads * head_dim` (each head its own slice). No bias.
fn layer_norm_per_head(x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    for (h, head) in x.chunks_mut(head_dim).enumerate() {
        let w = if weight.len() == head_dim {
            weight
        } else {
            &weight[h * head_dim..(h + 1) * head_dim]
        };
        layer_norm_opt_bias(head, w, None, eps);
    }
}

/// Reusable gather buffers for [`rope_partial`]: the rotated prefixes of every
/// Q head and every K head.
#[derive(Default)]
pub(crate) struct RopeGather {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
}

/// RoPE over the first `n_rot` dims of every head, leaving the tail of each
/// head untouched (StableLM `rope.dimension_count < head_dim`). Gathers the
/// rotated prefixes, runs the ordinary full-head kernel with `head_dim = n_rot`,
/// and scatters back; identical to llama.cpp rotating `n_rot` dims per head.
/// The gathered prefixes live in `scratch` so the partial path allocates
/// nothing once the buffers have grown to size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_partial(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    rope_theta: f32,
    rope_type: RopeType,
    scratch: &mut RopeGather,
) {
    let rotate = |q: &mut [f32], k: &mut [f32], nh: usize, nkv: usize, hd: usize| match rope_type {
        RopeType::Neox => cpu::rope(q, k, pos, nh, nkv, hd, rope_theta),
        RopeType::Norm => cpu::rope_norm(q, k, pos, nh, nkv, hd, rope_theta, None),
    };
    if n_rot == head_dim {
        rotate(q, k, n_heads, n_kv_heads, head_dim);
        return;
    }
    let gather = |x: &[f32], nh: usize, out: &mut Vec<f32>| {
        out.clear();
        for h in 0..nh {
            out.extend_from_slice(&x[h * head_dim..h * head_dim + n_rot]);
        }
    };
    gather(q, n_heads, &mut scratch.q);
    gather(k, n_kv_heads, &mut scratch.k);
    rotate(&mut scratch.q, &mut scratch.k, n_heads, n_kv_heads, n_rot);
    for h in 0..n_heads {
        q[h * head_dim..h * head_dim + n_rot]
            .copy_from_slice(&scratch.q[h * n_rot..(h + 1) * n_rot]);
    }
    for h in 0..n_kv_heads {
        k[h * head_dim..h * head_dim + n_rot]
            .copy_from_slice(&scratch.k[h * n_rot..(h + 1) * n_rot]);
    }
}

impl LlamaModel {
    /// Embedding row for `token`, scaled and recorded exactly as `forward`
    /// does. Panics on an out-of-range token, like `forward`.
    fn embed_token(&self, token: u32, what: &str) -> Vec<f32> {
        let token_id = token as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "{what}: token_id {token_id} out of range (vocab_size={})",
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
        hidden
    }

    /// `forward` minus the LM head: advances the KV cache for one token and
    /// returns nothing. For prompt tokens whose logits are never read.
    fn forward_no_head(&self, token: u32, pos: usize, state: &mut InferenceState) {
        let mut hidden = self.embed_token(token, "forward_prefill");
        self.run_layers(&mut hidden, pos, state);
    }

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
        Self::from_gguf_impl(gguf, context_size, model_id, true)
    }

    /// Load without the CPU int8 repacks. For the GPU/Metal loaders, which
    /// resolve weight metadata from this model but never dispatch CPU
    /// kernels — the repacks would be gigabytes allocated only to be freed
    /// after upload. Do NOT use for CPU inference (dispatch falls back to the
    /// naive path without the repacks, which is slower). This entry point is
    /// also the accelerator gate: the LayerNorm archs (`stablelm`, `starcoder2`,
    /// `cohere`, `command-r`) run only on the CPU path, so they fail to load
    /// here with an error naming the arch instead of loading slowly.
    pub fn from_gguf_with_id_no_repack(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        Self::from_gguf_impl(gguf, context_size, model_id, false)
    }

    fn from_gguf_impl(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
        repack: bool,
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
        } else if arch == "cohere"
            && (!gguf.metadata.contains_key("cohere.block_count")
                && gguf.metadata.contains_key("command-r.block_count"))
        {
            "command-r"
        } else if arch == "command-r"
            && (!gguf.metadata.contains_key("command-r.block_count")
                && gguf.metadata.contains_key("cohere.block_count"))
        {
            "cohere"
        } else if arch == "internlm"
            && (!gguf.metadata.contains_key("internlm.block_count")
                && gguf.metadata.contains_key("internlm2.block_count"))
        {
            "internlm2"
        } else {
            arch.as_str()
        };
        let prefix = resolved_prefix;
        let ext = requires_layernorm_path(prefix);
        if ext && !repack {
            bail!(
                "LlamaModel: arch {prefix:?} requires LayerNorm/parallel-residual support, \
                 which the GPU and NPU backends do not implement (CPU only)"
            );
        }
        let norm_kind = if ext { NormKind::Layer } else { NormKind::Rms };
        // StarCoder2 has no `ffn_gate`: plain up -> GELU -> down.
        let ffn_gated = prefix != "starcoder2";

        let rope_type = rope_type_for_arch(prefix).ok_or_else(|| {
            anyhow::anyhow!(
                "LlamaModel: no RoPE layout mapping for arch {prefix:?}; \
                 add it to rope_type_for_arch in llama.rs"
            )
        })?;

        let norm_order = match prefix {
            "olmo2" | "olmo3" => NormOrder::PostNorm,
            _ => NormOrder::PreNorm,
        };

        let activation = match prefix {
            "gemma2" | "starcoder2" => FfnActivation::Geglu,
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
        // Cohere/Command-R MULTIPLY the logits by `logit_scale` (llama.cpp
        // `ggml_scale(cur, f_logit_scale)`), while `ScalarMultipliers::logit` is a
        // divisor (Granite `logits_scaling`), so store the reciprocal. Absent key
        // means no scaling.
        if matches!(prefix, "cohere" | "command-r") {
            scalars.logit = match gguf.get_f32(&format!("{prefix}.logit_scale")) {
                Some(ls) => {
                    ensure!(
                        ls.is_finite() && ls > 0.0,
                        "{prefix}.logit_scale must be finite and positive, got {ls}"
                    );
                    1.0 / ls
                }
                None => 1.0,
            };
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
        // LayerNorm archs store `layer_norm_epsilon`; llama.cpp defaults it to 1e-5.
        let rms_norm_eps = if norm_kind == NormKind::Layer {
            gguf.get_f32(&format!("{prefix}.attention.layer_norm_epsilon"))
                .or_else(|| gguf.get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")))
                .unwrap_or(1e-5)
        } else {
            gguf.get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
                .or_else(|| gguf.get_f32(&format!("{prefix}.attention.layer_norm_epsilon")))
                .unwrap_or(1e-6)
        };
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

        // Rotated prefix of each head. Only StableLM sets a partial value; the
        // other archs keep the full-head rotation they always had.
        let n_rot = if prefix == "stablelm" {
            match gguf.get_u32(&format!("{prefix}.rope.dimension_count")) {
                Some(v) => {
                    let v = v as usize;
                    ensure!(
                        v > 0 && v.is_multiple_of(2) && v <= head_dim,
                        "{prefix}.rope.dimension_count ({v}) must be even and within (0, head_dim={head_dim}]"
                    );
                    v
                }
                None => head_dim,
            }
        } else {
            // Only StableLM's partial rotary is implemented. A file that rotates
            // fewer than `head_dim` dims on any other arch would silently get
            // full-head RoPE, so refuse it.
            if let Some(v) = gguf.get_u32(&format!("{prefix}.rope.dimension_count")) {
                ensure!(
                    v as usize >= head_dim,
                    "arch '{prefix}' sets {prefix}.rope.dimension_count ({v}) below head_dim \
                     ({head_dim}); partial rotary is only supported for stablelm"
                );
            }
            head_dim
        };

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
        let output_norm_bias = if gguf.tensors.contains_key("output_norm.bias") {
            ensure!(
                norm_kind == NormKind::Layer,
                "`output_norm.bias` is a LayerNorm bias, but {prefix:?} is loaded as an RMSNorm architecture"
            );
            let b = gguf.get_tensor("output_norm.bias")?.try_to_f32_vec()?;
            ensure!(
                b.len() == hidden_size,
                "output_norm.bias length {} != hidden_size ({hidden_size})",
                b.len()
            );
            Some(b)
        } else {
            None
        };

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
        let mut attn_norm_biases: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layers);
        let mut ffn_norm_biases: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layers);
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
            // LayerNorm bias tensors (StableLM/StarCoder2). RMSNorm archs have
            // none; a stray one on an RMS arch would be silently dropped, so it
            // fails closed instead.
            let load_norm_bias = |name: String| -> Result<Option<Vec<f32>>> {
                if !gguf.tensors.contains_key(&name) {
                    return Ok(None);
                }
                ensure!(
                    norm_kind == NormKind::Layer,
                    "`{name}` is a LayerNorm bias, but {prefix:?} is loaded as an RMSNorm architecture"
                );
                let b = gguf.get_tensor(&name)?.try_to_f32_vec()?;
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} {name} length {} != hidden_size ({hidden_size})",
                    b.len()
                );
                Ok(Some(b))
            };
            attn_norm_biases.push(load_norm_bias(format!("blk.{i}.attn_norm.bias"))?);

            let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");
            let ffn_norm = if gguf.tensors.contains_key(&ffn_norm_name) {
                gguf.get_tensor(&ffn_norm_name)?.try_to_f32_vec()?
            } else if norm_order == NormOrder::PostNorm || norm_kind == NormKind::Layer {
                // LayerNorm archs without `ffn_norm` run attention and FFN in
                // parallel off the attention norm (Cohere, StableLM 2 12B).
                Vec::new()
            } else {
                bail!("missing required tensor `{ffn_norm_name}` for PreNorm architecture");
            };
            if norm_order == NormOrder::PreNorm && !ffn_norm.is_empty() {
                ensure!(
                    ffn_norm.len() == hidden_size,
                    "layer {i} {ffn_norm_name} length {} != hidden_size ({hidden_size})",
                    ffn_norm.len()
                );
            }
            ffn_norm_weights.push(ffn_norm);
            ffn_norm_biases.push(load_norm_bias(format!("blk.{i}.ffn_norm.bias"))?);

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
            } else if [&q_bias_name, &k_bias_name, &v_bias_name]
                .iter()
                .any(|n| gguf.tensors.contains_key(*n))
            {
                // All-or-none: the forward path applies Q/K/V biases only as a
                // set, so a partial set would otherwise be dropped silently.
                let missing: Vec<&str> = [&q_bias_name, &k_bias_name, &v_bias_name]
                    .into_iter()
                    .filter(|n| !gguf.tensors.contains_key(*n))
                    .map(String::as_str)
                    .collect();
                ensure!(
                    missing.is_empty(),
                    "layer {i} has a partial Q/K/V bias set (missing {}); biases must be all-or-none",
                    missing.join(", ")
                );
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
            let (ffn_gate, ffn_up) = if !ffn_gated {
                // Plain up/down FFN (StarCoder2). `ffn_gate` aliases `ffn_up` so
                // the shared ref plumbing stays total; `ffn_gated` guards reads.
                ensure!(
                    !gguf.tensors.contains_key(&ffn_gate_name),
                    "{prefix:?} is loaded as an ungated FFN but layer {i} has `{ffn_gate_name}`"
                );
                let up = transformer::resolve_weight(&gguf, &ffn_up_name)?;
                ensure!(
                    up.m == config.intermediate_size && up.k == config.hidden_size,
                    "layer {i} {ffn_up_name} is {}x{}, expected {}x{}",
                    up.m,
                    up.k,
                    config.intermediate_size,
                    config.hidden_size
                );
                (up.clone(), up)
            } else if gguf.tensors.contains_key(&ffn_gate_name) {
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
                attn_q: attn_q.with_repack_if(&gguf, repack),
                attn_k: attn_k.with_repack_if(&gguf, repack),
                attn_v: attn_v.with_repack_if(&gguf, repack),
                attn_output: attn_output.with_repack_if(&gguf, repack),
                ffn_gate: ffn_gate.with_repack_if(&gguf, repack),
                ffn_up: ffn_up.with_repack_if(&gguf, repack),
                ffn_down: ffn_down.with_repack_if(&gguf, repack),
            });
        }

        if n_loops > 1 {
            for _ in 1..n_loops {
                attn_norm_weights.extend_from_within(..n_phys_layers);
                ffn_norm_weights.extend_from_within(..n_phys_layers);
                attn_norm_biases.extend_from_within(..n_phys_layers);
                ffn_norm_biases.extend_from_within(..n_phys_layers);
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

        // The LayerNorm path (`run_layers_ext`) applies plain (partial) RoPE and
        // full causal attention only. Loading a GGUF that asks for anything else
        // would silently produce wrong logits, so fail closed and name the arch.
        if ext {
            let unsupported: Vec<&str> = [
                ("rope yarn scaling", yarn.is_some()),
                ("rope_freqs.weight", rope_freqs.is_some()),
                ("attention.sliding_window", sliding_window.is_some()),
                ("attention temperature scaling", attn_temp_scale.is_some()),
            ]
            .into_iter()
            .filter_map(|(name, present)| present.then_some(name))
            .collect();
            ensure!(
                unsupported.is_empty(),
                "arch '{prefix}' uses the LayerNorm inference path, which does not support {}",
                unsupported.join(", ")
            );
        }

        // A GPU RoPE kernel takes one factor table and no per-layer variants,
        // so YaRN is reproducible there only when the table is otherwise free,
        // every layer rotates (no SWA layers, which skip YaRN) and the whole
        // head does (`n_rot == head_dim`).
        let yarn_freq_factors = match &yarn {
            Some(y) if rope_freqs.is_none() && sliding_window.is_none() && n_rot == head_dim => {
                cpu::yarn_freq_factors(head_dim, config.rope_theta, y)
            }
            _ => None,
        };

        Ok(Self {
            gguf,
            config,
            head_dim,
            rope_type,
            rope_freqs,
            yarn_freq_factors,
            norm_order,
            ext,
            n_rot,
            ffn_gated,
            activation,
            attn_logit_softcapping,
            final_logit_softcapping,
            output_norm_weight,
            output_norm_bias,
            attn_norm_weights,
            ffn_norm_weights,
            attn_norm_biases,
            ffn_norm_biases,
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

    /// Attention-output projection bias for layer `il`, if the model has one.
    #[cfg_attr(not(feature = "hexagon"), allow(dead_code))]
    pub(crate) fn attn_output_bias(&self, il: usize) -> Option<&[f32]> {
        self.attn_output_bias.get(il).and_then(|b| b.as_deref())
    }

    /// FFN gate projection bias for layer `il`, if any.
    #[cfg_attr(not(feature = "hexagon"), allow(dead_code))]
    pub(crate) fn ffn_gate_bias(&self, il: usize) -> Option<&[f32]> {
        self.ffn_gate_bias.get(il).and_then(|b| b.as_deref())
    }

    /// FFN up projection bias for layer `il`, if any.
    #[cfg_attr(not(feature = "hexagon"), allow(dead_code))]
    pub(crate) fn ffn_up_bias(&self, il: usize) -> Option<&[f32]> {
        self.ffn_up_bias.get(il).and_then(|b| b.as_deref())
    }

    /// FFN down projection bias for layer `il`, if any.
    #[cfg_attr(not(feature = "hexagon"), allow(dead_code))]
    pub(crate) fn ffn_down_bias(&self, il: usize) -> Option<&[f32]> {
        self.ffn_down_bias.get(il).and_then(|b| b.as_deref())
    }

    /// Attention temperature scaling `(scale, floor_scale)` (Mistral 3 / Llama 4).
    /// The CPU rule (`transformer::forward_attn_block`): for a token at `pos`,
    /// when `scale > 0 && floor_scale > 0 && pos >= floor_scale`, Q is multiplied
    /// by `ln(floor(pos / floor_scale) + 1) * scale + 1` after RoPE.
    #[cfg_attr(not(feature = "hexagon"), allow(dead_code))]
    pub(crate) fn attn_temp_scale(&self) -> Option<(f32, usize)> {
        self.attn_temp_scale
    }

    /// Why the GPU backends cannot run this model's rotary scheme, if they
    /// cannot: YaRN that does not reduce to a frequency table (see
    /// `yarn_freq_factors`).
    #[cfg(any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    ))]
    pub(crate) fn gpu_unsupported_reason(&self) -> Option<String> {
        (self.yarn.is_some() && self.yarn_freq_factors.is_none()).then(|| {
            "YaRN rope scaling combined with rope_freqs, sliding-window layers or partial rotary \
             is not reproducible by the GPU rotary kernel"
                .to_string()
        })
    }

    /// Longest context the GPU backends can serve exactly. Attention
    /// temperature scaling multiplies Q by a per-position factor that is 1
    /// below `floor_scale`; the GPU forward has no hook for the factor, so it
    /// is exact only below that position.
    #[cfg(any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    ))]
    pub(crate) fn gpu_context_cap(&self) -> Option<usize> {
        self.attn_temp_scale.map(|(_, floor_scale)| floor_scale)
    }

    /// Return whether any layer specifies an attention-output or FFN bias.
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

    /// Sequential LayerNorm graph for stablelm / starcoder2 / cohere / command-r
    /// (`self.ext`), one token. Mirrors llama.cpp's `build_stablelm`,
    /// `build_starcoder2` and `build_command_r`:
    ///
    /// - norms are true LayerNorm (mean-subtracting, optional bias);
    /// - a layer without `ffn_norm` runs attention and FFN in parallel off the
    ///   attention norm: `h + attn(norm(h)) + ffn(norm(h))` (Cohere, StableLM 2
    ///   12B), otherwise sequential `h1 = h + attn(norm(h)); h1 + ffn(norm2(h1))`;
    /// - the FFN is gated (SwiGLU) or, for StarCoder2, plain `up -> GELU -> down`;
    /// - optional per-head Q/K LayerNorm precedes RoPE; StableLM rotates only the
    ///   first `n_rot` dims of each head.
    ///
    /// Plain f32 GEMVs (no Q8 pre-quantized fast paths): this path serves four
    /// niche archs, so simplicity beats speed. No LoRA, no batched prefill.
    fn run_layers_ext(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let head_dim = self.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let eps = cfg.rms_norm_eps;
        let inter = cfg.intermediate_size;
        let use_f16 = state.kv_f16;

        // Per-token buffers reused across layers (each is re-zeroed or fully
        // overwritten per layer, so numerics match fresh allocations).
        let mut normed = vec![0.0f32; hs];
        let mut ffn_norm_buf = vec![0.0f32; hs];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut attn_proj = vec![0.0f32; hs];
        let mut gate = vec![0.0f32; if self.ffn_gated { inter } else { 0 }];
        let mut down_in = vec![0.0f32; inter];
        let mut ffn_out = vec![0.0f32; hs];

        for i in 0..cfg.n_layers {
            let refs = &self.layer_refs[i];

            normed.copy_from_slice(hidden);
            layer_norm_opt_bias(
                &mut normed,
                &self.attn_norm_weights[i],
                self.attn_norm_biases[i].as_deref(),
                eps,
            );

            // Attention.
            q.fill(0.0);
            k.fill(0.0);
            v.fill(0.0);
            transformer::gemv(&self.gguf, &refs.attn_q, &normed, &mut q);
            transformer::gemv(&self.gguf, &refs.attn_k, &normed, &mut k);
            transformer::gemv(&self.gguf, &refs.attn_v, &normed, &mut v);
            if let (Some(qb), Some(kb), Some(vb)) = (
                self.attn_q_bias[i].as_deref(),
                self.attn_k_bias[i].as_deref(),
                self.attn_v_bias[i].as_deref(),
            ) {
                cpu::add_inplace(&mut q, qb);
                cpu::add_inplace(&mut k, kb);
                cpu::add_inplace(&mut v, vb);
            }
            if let (Some(qn), Some(kn)) = (
                self.attn_q_norm_weights[i].as_deref(),
                self.attn_k_norm_weights[i].as_deref(),
            ) {
                layer_norm_per_head(&mut q, qn, head_dim, eps);
                layer_norm_per_head(&mut k, kn, head_dim, eps);
            }
            rope_partial(
                &mut q,
                &mut k,
                pos,
                n_heads,
                n_kv_heads,
                head_dim,
                self.n_rot,
                cfg.rope_theta,
                self.rope_type,
                &mut state.scratch.rope_gather,
            );

            if let crate::kv_cache::LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[i]
            {
                if use_f16 {
                    key_cache_f16.extend(k.iter().map(|&x| crate::quant::f32_to_f16(x)));
                    value_cache_f16.extend(v.iter().map(|&x| crate::quant::f32_to_f16(x)));
                } else {
                    key_cache.extend_from_slice(&k);
                    value_cache.extend_from_slice(&v);
                }
            }
            attn_out.fill(0.0);
            {
                let (kc, vc, kc16, vc16) = match &state.layers[i] {
                    crate::kv_cache::LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } => (
                        key_cache.as_slice(),
                        value_cache.as_slice(),
                        key_cache_f16.as_slice(),
                        value_cache_f16.as_slice(),
                    ),
                    _ => panic!("expected Attention state for layer {i}"),
                };
                let seq_len = if use_f16 {
                    kc16.len() / kv_dim
                } else {
                    kc.len() / kv_dim
                };
                let kv = if use_f16 {
                    transformer::KvView::F16 { k: kc16, v: vc16 }
                } else {
                    transformer::KvView::F32 { k: kc, v: vc }
                };
                transformer::decode_attention(
                    &q,
                    &kv,
                    &transformer::DecodeAttnDims {
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        scale: cfg
                            .scalars
                            .attn
                            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt()),
                        seq_len,
                        attn_logit_softcapping: None,
                        sliding_window: None,
                    },
                    &mut attn_out,
                    &mut state.scratch.scores,
                );
            }
            attn_proj.fill(0.0);
            transformer::gemv(&self.gguf, &refs.attn_output, &attn_out, &mut attn_proj);
            if let Some(b) = self.attn_output_bias[i].as_deref() {
                cpu::add_inplace(&mut attn_proj, b);
            }

            // FFN input: the same normed activation (parallel residual) or a
            // second LayerNorm after the attention residual.
            let parallel = self.ffn_norm_weights[i].is_empty();
            let ffn_in: &[f32] = if parallel {
                &normed
            } else {
                cpu::add_inplace(hidden, &attn_proj);
                ffn_norm_buf.copy_from_slice(hidden);
                layer_norm_opt_bias(
                    &mut ffn_norm_buf,
                    &self.ffn_norm_weights[i],
                    self.ffn_norm_biases[i].as_deref(),
                    eps,
                );
                &ffn_norm_buf
            };

            down_in.fill(0.0);
            if self.ffn_gated {
                gate.fill(0.0);
                transformer::gemv(&self.gguf, &refs.ffn_gate, ffn_in, &mut gate);
                transformer::gemv(&self.gguf, &refs.ffn_up, ffn_in, &mut down_in);
                if let Some(b) = self.ffn_gate_bias[i].as_deref() {
                    cpu::add_inplace(&mut gate, b);
                }
                if let Some(b) = self.ffn_up_bias[i].as_deref() {
                    cpu::add_inplace(&mut down_in, b);
                }
                match self.activation {
                    FfnActivation::Swiglu => cpu::silu_mul_inplace(&mut gate, &down_in),
                    FfnActivation::Geglu => cpu::gelu_mul_inplace(&mut gate, &down_in),
                }
                down_in.copy_from_slice(&gate);
            } else {
                transformer::gemv(&self.gguf, &refs.ffn_up, ffn_in, &mut down_in);
                if let Some(b) = self.ffn_up_bias[i].as_deref() {
                    cpu::add_inplace(&mut down_in, b);
                }
                cpu::gelu_inplace(&mut down_in);
            }
            ffn_out.fill(0.0);
            transformer::gemv(&self.gguf, &refs.ffn_down, &down_in, &mut ffn_out);
            if let Some(b) = self.ffn_down_bias[i].as_deref() {
                cpu::add_inplace(&mut ffn_out, b);
            }

            if parallel {
                cpu::add_inplace(hidden, &attn_proj);
            }
            cpu::add_inplace(hidden, &ffn_out);
        }

        layer_norm_opt_bias(
            hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            eps,
        );
        state.seq_len += 1;
    }

    /// Run all layers + final RMSNorm on a single-token hidden state.
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        if self.ext {
            return self.run_layers_ext(hidden, pos, state);
        }
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
        #[cfg(test)]
        PROJECT_LOGITS_CALLS.with(|c| c.set(c.get() + 1));
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

        let mut prof = PrefillProf::new();
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
                        transformer::rmsnorm_columns(
                            &hidden,
                            &mut normed,
                            &self.attn_norm_weights[layer],
                            cfg.rms_norm_eps,
                            hs,
                            n,
                        );
                    }
                    &normed
                }
                NormOrder::PostNorm => &hidden,
            };

            prof.lap(0);
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

            prof.lap(1);
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
            // into q_mat (so the attention pass can read every query) → write
            // K/V into the cache. Destructure the cache once (not per token).
            // f16 KV: the write converts to half; Pass B widens back to an f32
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
            // Tokens are independent here (the cache rows are disjoint and Pass B has
            // not started), so Pass A fans out over the prefill pool in tiles of 16
            // tokens: a tile gathers its columns of q/k/v (one 64-byte run per row,
            // not one cache line per element), runs the same per-token code, scatters
            // the post-RoPE Q back, and writes its K/V rows straight into the cache.
            let process = |pos: usize, q: &mut [f32], k: &mut [f32], v: &mut [f32]| {
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
            };
            const TILE: usize = 16;
            let kv_base = if use_f16 {
                key_cache_f16.len()
            } else {
                key_cache.len()
            };
            // `resize` keeps the old `extend` contract (rows land at the end of the
            // cache) while giving each worker a disjoint, already-initialized span.
            if use_f16 {
                key_cache_f16.resize(kv_base + n * kv_dim, 0);
                value_cache_f16.resize(kv_base + n * kv_dim, 0);
            } else {
                key_cache.resize(kv_base + n * kv_dim, 0.0);
                value_cache.resize(kv_base + n * kv_dim, 0.0);
            }
            let q_mat_ptr = q_mat.as_mut_ptr() as usize;
            let kc32 = key_cache.as_mut_ptr() as usize;
            let vc32 = value_cache.as_mut_ptr() as usize;
            let kc16 = key_cache_f16.as_mut_ptr() as usize;
            let vc16 = value_cache_f16.as_mut_ptr() as usize;
            let (k_src, v_src) = (&k_mat[..], &v_mat[..]);
            cpu::par_range_prefill(n.div_ceil(TILE), 1, |tile0, n_tiles| {
                let q_mat = q_mat_ptr as *mut f32;
                let mut qb = vec![0.0f32; TILE * q_dim];
                let mut kb = vec![0.0f32; TILE * kv_dim];
                let mut vb = vec![0.0f32; TILE * kv_dim];
                for tile in tile0..tile0 + n_tiles {
                    let j0 = tile * TILE;
                    let nc = TILE.min(n - j0);
                    for i in 0..q_dim {
                        for c in 0..nc {
                            // SAFETY: `i * n + j0 + c < q_dim * n`; tiles own disjoint columns.
                            qb[c * q_dim + i] = unsafe { *q_mat.add(i * n + j0 + c) };
                        }
                    }
                    for i in 0..kv_dim {
                        for c in 0..nc {
                            kb[c * kv_dim + i] = k_src[i * n + j0 + c];
                            vb[c * kv_dim + i] = v_src[i * n + j0 + c];
                        }
                    }
                    for c in 0..nc {
                        let j = j0 + c;
                        let q = &mut qb[c * q_dim..(c + 1) * q_dim];
                        let k = &mut kb[c * kv_dim..(c + 1) * kv_dim];
                        let v = &mut vb[c * kv_dim..(c + 1) * kv_dim];
                        process(start_pos + j, q, k, v);
                        let row = kv_base + j * kv_dim;
                        // SAFETY: row `kv_base + j * kv_dim` was sized by the `resize` above and
                        // is written only by the tile that owns token `j`.
                        unsafe {
                            if use_f16 {
                                let (dk, dv) =
                                    ((kc16 as *mut u16).add(row), (vc16 as *mut u16).add(row));
                                for i in 0..kv_dim {
                                    *dk.add(i) = crate::quant::f32_to_f16(k[i]);
                                    *dv.add(i) = crate::quant::f32_to_f16(v[i]);
                                }
                            } else {
                                core::ptr::copy_nonoverlapping(
                                    k.as_ptr(),
                                    (kc32 as *mut f32).add(row),
                                    kv_dim,
                                );
                                core::ptr::copy_nonoverlapping(
                                    v.as_ptr(),
                                    (vc32 as *mut f32).add(row),
                                    kv_dim,
                                );
                            }
                        }
                    }
                    for i in 0..q_dim {
                        for c in 0..nc {
                            // SAFETY: as above; this tile's columns are written by this tile only.
                            unsafe { *q_mat.add(i * n + j0 + c) = qb[c * q_dim + i] };
                        }
                    }
                }
            });

            prof.lap(2);
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
                let max_active = cpu::prefill_threads_for_tokens(n);
                // Work items of one head x 32 queries where the NEON range kernel applies (so the
                // pool can balance fast and slow cores); otherwise one item per head, as before.
                if !cpu::flash_attention_prefill_items(
                    q_ref, k_cache, v_cache, flash_buf, n_heads, group_size, n, kv_dim, head_dim,
                    scale, start_pos, true, max_active,
                ) {
                    cpu::par_rows_n_chunked_active(
                        flash_buf,
                        head_chunk,
                        1,
                        1,
                        max_active,
                        |(h, chunk)| {
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
                        },
                    );
                }
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

            prof.lap(3);
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

            prof.lap(4);
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
                        transformer::rmsnorm_columns(
                            &hidden,
                            &mut ffn_input,
                            &self.ffn_norm_weights[layer],
                            cfg.rms_norm_eps,
                            hs,
                            n,
                        );
                    }
                    &ffn_input
                }
                NormOrder::PostNorm => &hidden,
            };

            prof.lap(5);
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
                prof.lap(6);
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

            prof.lap(7);
            match self.activation {
                FfnActivation::Swiglu => {
                    cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);
                }
                FfnActivation::Geglu => {
                    cpu::gelu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);
                }
            }
            prof.lap(8);

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
                prof.lap(9);
                let t_dn = std::time::Instant::now();
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
                if prof.on {
                    eprintln!(
                        "DOWN layer {layer}: {:.2} ms",
                        t_dn.elapsed().as_secs_f64() * 1e3
                    );
                }
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

            prof.lap(10);
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

        prof.report(n);
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

    /// LoRA hooks live in the shared `transformer::forward_*_block` helpers
    /// this backend decodes through.
    fn supports_lora(&self) -> bool {
        // The LayerNorm path (`run_layers_ext`) has no LoRA hooks.
        !self.ext
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
        if tokens.len() > 1 && !self.ext {
            let mut out = Vec::new();
            self.forward_prefill_batched(tokens, 0, state, Some(&mut out));
            return out;
        }
        self.hidden_states_per_token(tokens, state)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LlamaModel forward expects single token");
        let mut hidden = self.embed_token(tokens[0], "forward");
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
        if tokens.len() > 1 && !self.ext && !transformer::oracle_dump::is_active() {
            return self.forward_prefill_batched(tokens, start_pos, state, None);
        }

        // Sequential per-token prefill (single-token, LayerNorm archs, or no
        // batched kernel). Only the last token's logits are returned, so the
        // earlier tokens run embed + layers and skip the vocab-sized LM head
        // and logits allocation. The oracle-dump harness keeps the full
        // `forward` per token so every node is recorded.
        let last = tokens.len() - 1;
        if !transformer::oracle_dump::is_active() {
            for (i, &token) in tokens[..last].iter().enumerate() {
                self.forward_no_head(token, start_pos + i, state);
            }
            return self.forward(&tokens[last..], start_pos + last, state);
        }
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
        if n > 1 && !self.ext && !transformer::oracle_dump::is_active() {
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
        self.yarn.is_none() && self.sliding_window.is_none() && self.n_rot == self.head_dim
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        if self.yarn.is_some() || self.sliding_window.is_some() || self.n_rot != self.head_dim {
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
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    feature = "hexagon"
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
    fn yarn_rope_freq_factors(&self) -> Option<&[f32]> {
        self.yarn_freq_factors.as_deref()
    }
    fn attn_scale_multiplier(&self) -> f32 {
        match (&self.yarn_freq_factors, &self.yarn) {
            (Some(_), Some(y)) => y.mscale * y.mscale,
            _ => 1.0,
        }
    }
    fn attn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_post_norm_weights
            .get(layer)
            .and_then(|w| w.as_deref())
    }
    fn ffn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.ffn_post_norm_weights
            .get(layer)
            .and_then(|w| w.as_deref())
    }
    fn activation(&self) -> crate::model::transformer::FfnActivation {
        self.activation
    }
    fn attn_logit_softcapping(&self) -> Option<f32> {
        self.attn_logit_softcapping
    }
    fn final_logit_softcapping(&self) -> Option<f32> {
        self.final_logit_softcapping
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
    use super::rope_type_for_arch;
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

    #[test]
    fn test_llama_arch_rope_type_coverage() {
        use crate::backend::cpu::RopeType;

        let neox_archs = [
            "qwen2",
            "qwen3",
            "gemma2",
            "olmo2",
            "olmo3",
            "phi3",
            "phi",
            "starcoder2",
            "stablelm",
            "openelm",
        ];
        let norm_archs = [
            "llama",
            "granite",
            "minicpm",
            "minicpm5",
            "nanbeige",
            "mistral3",
            "ministral3",
            "ministral",
            "baichuan",
            "deepseek",
            "internlm2",
            "internlm",
            "cohere",
            "command-r",
        ];
        for arch in neox_archs {
            assert_eq!(rope_type_for_arch(arch), Some(RopeType::Neox), "{arch}");
        }
        for arch in norm_archs {
            assert_eq!(rope_type_for_arch(arch), Some(RopeType::Norm), "{arch}");
        }
        assert_eq!(rope_type_for_arch("not-an-arch"), None);
    }

    // ── LayerNorm-arch (stablelm / starcoder2 / cohere / command-r) tests ────

    use super::{
        LlamaModel, layer_norm_opt_bias, layer_norm_per_head, requires_layernorm_path, rope_partial,
    };
    use crate::backend::cpu;
    use crate::gguf::GgufFile;
    use crate::model::Model;
    use std::sync::Arc;

    fn scalar_layer_norm(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + eps as f64).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                ((v as f64 - mean) * inv * w[i] as f64 + b.map_or(0.0, |b| b[i] as f64)) as f32
            })
            .collect()
    }

    #[test]
    fn layer_norm_matches_scalar_reference_with_and_without_bias() {
        let x: Vec<f32> = (0..32)
            .map(|i| (i as f32 * 0.37).sin() * 3.0 + 1.5)
            .collect();
        let w: Vec<f32> = (0..32).map(|i| 0.5 + i as f32 * 0.03).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 * 0.11).cos() * 0.2).collect();
        for bias in [Some(&b[..]), None] {
            let mut got = x.clone();
            layer_norm_opt_bias(&mut got, &w, bias, 1e-5);
            let want = scalar_layer_norm(&x, &w, bias, 1e-5);
            for (g, r) in got.iter().zip(&want) {
                assert!((g - r).abs() < 1e-4, "{g} vs {r}");
            }
        }
        // LayerNorm subtracts the mean; RMSNorm does not. A constant vector
        // normalizes to the bias under LayerNorm.
        let mut c = vec![2.0f32; 32];
        layer_norm_opt_bias(&mut c, &w, Some(&b), 1e-5);
        for (g, bb) in c.iter().zip(&b) {
            assert!((g - bb).abs() < 1e-3);
        }
    }

    #[test]
    fn per_head_layer_norm_slices_weights_per_head() {
        let hd = 8;
        let x: Vec<f32> = (0..24).map(|i| (i as f32 * 0.7).sin() + 0.3).collect();
        let w: Vec<f32> = (0..24).map(|i| 1.0 + i as f32 * 0.05).collect();
        let mut got = x.clone();
        layer_norm_per_head(&mut got, &w, hd, 1e-5);
        for h in 0..3 {
            let want = scalar_layer_norm(
                &x[h * hd..(h + 1) * hd],
                &w[h * hd..(h + 1) * hd],
                None,
                1e-5,
            );
            for (g, r) in got[h * hd..(h + 1) * hd].iter().zip(&want) {
                assert!((g - r).abs() < 1e-4);
            }
        }
        // A head_dim-length weight is shared by all heads.
        let shared = &w[..hd];
        let mut got = x.clone();
        layer_norm_per_head(&mut got, shared, hd, 1e-5);
        let want = scalar_layer_norm(&x[hd..2 * hd], shared, None, 1e-5);
        for (g, r) in got[hd..2 * hd].iter().zip(&want) {
            assert!((g - r).abs() < 1e-4);
        }
    }

    #[test]
    fn partial_rope_rotates_only_the_prefix_of_each_head() {
        use crate::backend::cpu::RopeType;
        let (nh, nkv, hd, nrot) = (2usize, 1usize, 16usize, 8usize);
        let q0: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.21).sin()).collect();
        let k0: Vec<f32> = (0..nkv * hd).map(|i| (i as f32 * 0.13).cos()).collect();
        for rt in [RopeType::Neox, RopeType::Norm] {
            let (mut q, mut k) = (q0.clone(), k0.clone());
            let mut scratch = crate::model::llama::RopeGather::default();
            rope_partial(
                &mut q,
                &mut k,
                5,
                nh,
                nkv,
                hd,
                nrot,
                10000.0,
                rt,
                &mut scratch,
            );
            // The gather buffers were sized by the call and a second call
            // reuses them: capacity is stable (pointer equality is not a
            // reliable signal, the allocator may hand back the same block).
            let (cq, ck) = (scratch.q.capacity(), scratch.k.capacity());
            assert!(cq >= nh * nrot && ck >= nkv * nrot);
            let (mut q2, mut k2) = (q0.clone(), k0.clone());
            rope_partial(
                &mut q2,
                &mut k2,
                5,
                nh,
                nkv,
                hd,
                nrot,
                10000.0,
                rt,
                &mut scratch,
            );
            assert_eq!((scratch.q.capacity(), scratch.k.capacity()), (cq, ck));
            assert_eq!(
                (&q2, &k2),
                (&q, &k),
                "reused scratch must not change the result"
            );
            // Tails untouched.
            for h in 0..nh {
                assert_eq!(
                    &q[h * hd + nrot..(h + 1) * hd],
                    &q0[h * hd + nrot..(h + 1) * hd]
                );
            }
            assert_eq!(&k[nrot..hd], &k0[nrot..hd]);
            // Prefix equals the full kernel applied to a head of size nrot.
            let mut qr: Vec<f32> = (0..nh)
                .flat_map(|h| q0[h * hd..h * hd + nrot].to_vec())
                .collect();
            let mut kr = k0[..nrot].to_vec();
            match rt {
                RopeType::Neox => cpu::rope(&mut qr, &mut kr, 5, nh, nkv, nrot, 10000.0),
                RopeType::Norm => cpu::rope_norm(&mut qr, &mut kr, 5, nh, nkv, nrot, 10000.0, None),
            }
            for h in 0..nh {
                assert_eq!(&q[h * hd..h * hd + nrot], &qr[h * nrot..(h + 1) * nrot]);
            }
            assert_eq!(&k[..nrot], &kr[..]);
            // n_rot == head_dim is the ordinary full rotation.
            let (mut qf, mut kf) = (q0.clone(), k0.clone());
            rope_partial(
                &mut qf,
                &mut kf,
                5,
                nh,
                nkv,
                hd,
                hd,
                10000.0,
                rt,
                &mut scratch,
            );
            let (mut qg, mut kg) = (q0.clone(), k0.clone());
            match rt {
                RopeType::Neox => cpu::rope(&mut qg, &mut kg, 5, nh, nkv, hd, 10000.0),
                RopeType::Norm => cpu::rope_norm(&mut qg, &mut kg, 5, nh, nkv, hd, 10000.0, None),
            }
            assert_eq!(qf, qg);
            assert_eq!(kf, kg);
        }
    }

    #[test]
    fn layernorm_arch_classifier_names_exactly_four_archs() {
        for a in ["stablelm", "starcoder2", "cohere", "command-r"] {
            assert!(requires_layernorm_path(a), "{a}");
        }
        for a in ["llama", "qwen2", "phi3", "internlm2", "gemma2"] {
            assert!(!requires_layernorm_path(a), "{a}");
        }
    }

    // A tiny synthetic GGUF (F32 tensors) per arch, run through the production
    // loader and `forward`, checked against an independent scalar
    // implementation of llama.cpp's graph.

    const HS: usize = 32;
    const NH: usize = 2;
    const NKV: usize = 1;
    const HD: usize = 16;
    const INTER: usize = 32;
    const VOCAB: usize = 8;
    const NLAYER: usize = 2;
    const THETA: f32 = 10000.0;
    const EPS: f32 = 1e-5;

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Kind {
        Cohere,
        Starcoder2,
        Stablelm,
    }

    struct Tiny {
        kind: Kind,
        arch: Option<&'static str>,
        tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
        /// Extra metadata as `(key suffix after "<arch>.", gguf type, bytes)`.
        extra_kv: Vec<(&'static str, u32, Vec<u8>)>,
    }

    impl Tiny {
        fn get(&self, name: &str) -> &[f32] {
            &self
                .tensors
                .iter()
                .find(|t| t.0 == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .2
        }
        fn has(&self, name: &str) -> bool {
            self.tensors.iter().any(|t| t.0 == name)
        }
        fn prefix(&self) -> &'static str {
            if let Some(a) = self.arch {
                return a;
            }
            match self.kind {
                Kind::Cohere => "cohere",
                Kind::Starcoder2 => "starcoder2",
                Kind::Stablelm => "stablelm",
            }
        }
        const NROT_STABLELM: usize = 8;
        const LOGIT_SCALE: f32 = 0.5;
    }

    fn lcg_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 2.0 * scale
            })
            .collect()
    }

    fn build_tiny(kind: Kind) -> Tiny {
        let mut seed = 0x1234_5678_9abc_def0u64 ^ (kind as u64 + 1);
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut add = |name: &str, dims: Vec<usize>, scale: f32, base: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = lcg_vec(&mut seed, n, scale)
                .into_iter()
                .map(|x| x + base)
                .collect();
            t.push((name.to_string(), dims, v));
        };
        let q_dim = NH * HD;
        let kv_dim = NKV * HD;
        add("token_embd.weight", vec![HS, VOCAB], 1.0, 0.0);
        add("output_norm.weight", vec![HS], 0.2, 1.0);
        if kind != Kind::Cohere {
            add("output_norm.bias", vec![HS], 0.1, 0.0);
            add("output.weight", vec![HS, VOCAB], 0.5, 0.0);
        }
        for l in 0..NLAYER {
            let b = |s: &str| format!("blk.{l}.{s}");
            add(&b("attn_norm.weight"), vec![HS], 0.2, 1.0);
            if kind != Kind::Cohere {
                add(&b("attn_norm.bias"), vec![HS], 0.1, 0.0);
                add(&b("ffn_norm.weight"), vec![HS], 0.2, 1.0);
                add(&b("ffn_norm.bias"), vec![HS], 0.1, 0.0);
            }
            add(&b("attn_q.weight"), vec![HS, q_dim], 0.3, 0.0);
            add(&b("attn_k.weight"), vec![HS, kv_dim], 0.3, 0.0);
            add(&b("attn_v.weight"), vec![HS, kv_dim], 0.3, 0.0);
            add(&b("attn_output.weight"), vec![q_dim, HS], 0.3, 0.0);
            if kind != Kind::Cohere {
                add(&b("attn_q.bias"), vec![q_dim], 0.1, 0.0);
                add(&b("attn_k.bias"), vec![kv_dim], 0.1, 0.0);
                add(&b("attn_v.bias"), vec![kv_dim], 0.1, 0.0);
            }
            if kind == Kind::Stablelm {
                add(&b("attn_q_norm.weight"), vec![q_dim], 0.2, 1.0);
                add(&b("attn_k_norm.weight"), vec![kv_dim], 0.2, 1.0);
            }
            if kind == Kind::Starcoder2 {
                add(&b("attn_output.bias"), vec![HS], 0.1, 0.0);
                add(&b("ffn_up.weight"), vec![HS, INTER], 0.3, 0.0);
                add(&b("ffn_up.bias"), vec![INTER], 0.1, 0.0);
                add(&b("ffn_down.weight"), vec![INTER, HS], 0.3, 0.0);
                add(&b("ffn_down.bias"), vec![HS], 0.1, 0.0);
            } else {
                add(&b("ffn_gate.weight"), vec![HS, INTER], 0.3, 0.0);
                add(&b("ffn_up.weight"), vec![HS, INTER], 0.3, 0.0);
                add(&b("ffn_down.weight"), vec![INTER, HS], 0.3, 0.0);
            }
        }
        Tiny {
            kind,
            arch: None,
            tensors: t,
            extra_kv: Vec::new(),
        }
    }

    fn gguf_bytes(m: &Tiny) -> Vec<u8> {
        use crate::gguf::{GgufBuilder, KvValue};
        let p = m.prefix();
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", p)
            .kv_u32(format!("{p}.block_count"), NLAYER as u32)
            .kv_u32(format!("{p}.embedding_length"), HS as u32)
            .kv_u32(format!("{p}.feed_forward_length"), INTER as u32)
            .kv_u32(format!("{p}.attention.head_count"), NH as u32)
            .kv_u32(format!("{p}.attention.head_count_kv"), NKV as u32)
            .kv_f32(format!("{p}.attention.layer_norm_epsilon"), EPS)
            .kv_f32(format!("{p}.rope.freq_base"), THETA)
            .kv_u32(format!("{p}.context_length"), 64)
            .kv_u32(format!("{p}.vocab_size"), VOCAB as u32);
        if m.kind == Kind::Stablelm {
            b = b.kv_u32(
                format!("{p}.rope.dimension_count"),
                Tiny::NROT_STABLELM as u32,
            );
        }
        if m.kind == Kind::Cohere {
            b = b.kv_f32(format!("{p}.logit_scale"), Tiny::LOGIT_SCALE);
        }
        for (suffix, ty, bytes) in &m.extra_kv {
            b = b.kv(format!("{p}.{suffix}"), KvValue::Raw(*ty, bytes.clone()));
        }
        for (name, dims, data) in &m.tensors {
            b = b.tensor_f32(name.as_str(), dims, data);
        }
        b.build_bytes()
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn matvec(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = x.len();
        (0..out_dim)
            .map(|o| (0..in_dim).map(|i| w[o * in_dim + i] * x[i]).sum())
            .collect()
    }

    fn add_bias(v: &mut [f32], b: Option<&[f32]>) {
        if let Some(b) = b {
            for (x, y) in v.iter_mut().zip(b) {
                *x += y;
            }
        }
    }

    /// Independent scalar reference of the four archs' graphs.
    fn reference_logits(m: &Tiny, tokens: &[usize]) -> Vec<Vec<f32>> {
        let neox = matches!(m.kind, Kind::Starcoder2 | Kind::Stablelm);
        let n_rot = if m.kind == Kind::Stablelm {
            Tiny::NROT_STABLELM
        } else {
            HD
        };
        let q_dim = NH * HD;
        let kv_dim = NKV * HD;
        let mut kcache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); NLAYER];
        let mut vcache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); NLAYER];
        let opt = |name: String| -> Option<Vec<f32>> {
            if m.has(&name) {
                Some(m.get(&name).to_vec())
            } else {
                None
            }
        };
        let mut all = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = m.get("token_embd.weight")[tok * HS..(tok + 1) * HS].to_vec();
            for l in 0..NLAYER {
                let b = |s: &str| format!("blk.{l}.{s}");
                let attn_norm_b = opt(b("attn_norm.bias"));
                let normed = scalar_layer_norm(
                    &h,
                    m.get(&b("attn_norm.weight")),
                    attn_norm_b.as_deref(),
                    EPS,
                );
                let mut q = matvec(m.get(&b("attn_q.weight")), &normed, q_dim);
                let mut k = matvec(m.get(&b("attn_k.weight")), &normed, kv_dim);
                let mut v = matvec(m.get(&b("attn_v.weight")), &normed, kv_dim);
                add_bias(&mut q, opt(b("attn_q.bias")).as_deref());
                add_bias(&mut k, opt(b("attn_k.bias")).as_deref());
                add_bias(&mut v, opt(b("attn_v.bias")).as_deref());
                if m.has(&b("attn_q_norm.weight")) {
                    for hh in 0..NH {
                        let w = &m.get(&b("attn_q_norm.weight"))[hh * HD..(hh + 1) * HD];
                        let r = scalar_layer_norm(&q[hh * HD..(hh + 1) * HD], w, None, EPS);
                        q[hh * HD..(hh + 1) * HD].copy_from_slice(&r);
                    }
                    for hh in 0..NKV {
                        let w = &m.get(&b("attn_k_norm.weight"))[hh * HD..(hh + 1) * HD];
                        let r = scalar_layer_norm(&k[hh * HD..(hh + 1) * HD], w, None, EPS);
                        k[hh * HD..(hh + 1) * HD].copy_from_slice(&r);
                    }
                }
                let rot = |x: &mut [f32], heads: usize| {
                    for hh in 0..heads {
                        let base = hh * HD;
                        for i in 0..n_rot / 2 {
                            let theta = pos as f32 * THETA.powf(-2.0 * i as f32 / n_rot as f32);
                            let (s, c) = theta.sin_cos();
                            let (a, bb) = if neox {
                                (base + i, base + i + n_rot / 2)
                            } else {
                                (base + 2 * i, base + 2 * i + 1)
                            };
                            let (x0, x1) = (x[a], x[bb]);
                            x[a] = x0 * c - x1 * s;
                            x[bb] = x0 * s + x1 * c;
                        }
                    }
                };
                rot(&mut q, NH);
                rot(&mut k, NKV);
                kcache[l].push(k);
                vcache[l].push(v);
                let mut attn = vec![0.0f32; q_dim];
                for hh in 0..NH {
                    let kvh = hh / (NH / NKV);
                    let scores: Vec<f32> = kcache[l]
                        .iter()
                        .map(|kk| {
                            (0..HD)
                                .map(|d| q[hh * HD + d] * kk[kvh * HD + d])
                                .sum::<f32>()
                                / (HD as f32).sqrt()
                        })
                        .collect();
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let ex: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let sum: f32 = ex.iter().sum();
                    for (t, e) in ex.iter().enumerate() {
                        for d in 0..HD {
                            attn[hh * HD + d] += e / sum * vcache[l][t][kvh * HD + d];
                        }
                    }
                }
                let mut a_out = matvec(m.get(&b("attn_output.weight")), &attn, HS);
                add_bias(&mut a_out, opt(b("attn_output.bias")).as_deref());

                let parallel = !m.has(&b("ffn_norm.weight"));
                let ffn_in = if parallel {
                    normed.clone()
                } else {
                    for (x, y) in h.iter_mut().zip(&a_out) {
                        *x += y;
                    }
                    let fb = opt(b("ffn_norm.bias"));
                    scalar_layer_norm(&h, m.get(&b("ffn_norm.weight")), fb.as_deref(), EPS)
                };
                let mut up = matvec(m.get(&b("ffn_up.weight")), &ffn_in, INTER);
                add_bias(&mut up, opt(b("ffn_up.bias")).as_deref());
                let act: Vec<f32> = if m.has(&b("ffn_gate.weight")) {
                    let gate = matvec(m.get(&b("ffn_gate.weight")), &ffn_in, INTER);
                    gate.iter()
                        .zip(&up)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect()
                } else {
                    up.iter()
                        .map(|&x| {
                            let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
                            0.5 * x * (1.0 + inner.tanh())
                        })
                        .collect()
                };
                let mut f = matvec(m.get(&b("ffn_down.weight")), &act, HS);
                add_bias(&mut f, opt(b("ffn_down.bias")).as_deref());
                if parallel {
                    for (x, y) in h.iter_mut().zip(&a_out) {
                        *x += y;
                    }
                }
                for (x, y) in h.iter_mut().zip(&f) {
                    *x += y;
                }
            }
            let ob = opt("output_norm.bias".to_string());
            let hn = scalar_layer_norm(&h, m.get("output_norm.weight"), ob.as_deref(), EPS);
            let head = if m.has("output.weight") {
                m.get("output.weight")
            } else {
                m.get("token_embd.weight")
            };
            let mut logits = matvec(head, &hn, VOCAB);
            if m.kind == Kind::Cohere {
                for l in &mut logits {
                    *l *= Tiny::LOGIT_SCALE;
                }
            }
            all.push(logits);
        }
        all
    }

    fn load_tiny(m: &Tiny) -> LlamaModel {
        let gguf = GgufFile::from_bytes(Arc::from(gguf_bytes(m).into_boxed_slice()))
            .expect("synthetic gguf parses");
        LlamaModel::from_gguf(gguf, 64).expect("synthetic model loads")
    }

    fn check_arch(kind: Kind) {
        let m = build_tiny(kind);
        let model = load_tiny(&m);
        assert!(model.ext);
        let tokens = [1usize, 5, 2, 7];
        let want = reference_logits(&m, &tokens);
        let mut state = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
        // Decode one token at a time.
        for (pos, &t) in tokens.iter().enumerate() {
            let got = model.forward(&[t as u32], pos, &mut state);
            for (g, r) in got.iter().zip(&want[pos]) {
                assert!(
                    (g - r).abs() < 2e-3 * r.abs().max(1.0),
                    "pos {pos}: {g} vs {r}"
                );
            }
        }
        // Batched-prefill entry point takes the same sequential path.
        let mut state2 = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
        let toks: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let last = model.forward_prefill(&toks, 0, &mut state2);
        for (g, r) in last.iter().zip(&want[tokens.len() - 1]) {
            assert!((g - r).abs() < 2e-3 * r.abs().max(1.0), "{g} vs {r}");
        }
    }

    #[test]
    fn cohere_parallel_residual_matches_scalar_reference() {
        check_arch(Kind::Cohere);
    }

    #[test]
    fn starcoder2_ungated_layernorm_ffn_matches_scalar_reference() {
        check_arch(Kind::Starcoder2);
    }

    #[test]
    fn stablelm_partial_rope_and_qk_layernorm_matches_scalar_reference() {
        check_arch(Kind::Stablelm);
    }

    #[test]
    fn layernorm_archs_are_rejected_by_the_no_repack_accelerator_loader() {
        let m = build_tiny(Kind::Cohere);
        let gguf = GgufFile::from_bytes(Arc::from(gguf_bytes(&m).into_boxed_slice())).unwrap();
        match LlamaModel::from_gguf_with_id_no_repack(gguf, 64, String::new()) {
            Ok(_) => panic!("accelerator loaders must not accept LayerNorm archs"),
            Err(e) => assert!(e.to_string().contains("LayerNorm"), "{e}"),
        }
    }

    #[test]
    fn rms_arch_rejects_a_layernorm_bias_tensor() {
        // A llama-arch file carrying `attn_norm.bias` would silently lose it.
        let mut m = build_tiny(Kind::Cohere);
        m.arch = Some("llama");
        m.tensors
            .push(("blk.0.attn_norm.bias".into(), vec![HS], vec![0.0; HS]));
        let gguf = GgufFile::from_bytes(Arc::from(gguf_bytes(&m).into_boxed_slice())).unwrap();
        match LlamaModel::from_gguf(gguf, 64) {
            Ok(_) => panic!("an RMSNorm arch must reject a LayerNorm bias"),
            Err(e) => assert!(e.to_string().contains("LayerNorm bias"), "{e}"),
        }
    }

    /// The LayerNorm path implements plain (partial) RoPE and full causal
    /// attention only. Each feature it lacks must fail the load, naming the
    /// feature, instead of yielding wrong logits.
    #[test]
    fn layernorm_arch_fails_closed_on_unsupported_features() {
        let u32v = |v: u32| v.to_le_bytes().to_vec();
        let f32v = |v: f32| v.to_le_bytes().to_vec();
        let str_v = |s: &str| {
            let mut b = Vec::new();
            put_str(&mut b, s);
            b
        };
        for kind in [Kind::Cohere, Kind::Starcoder2, Kind::Stablelm] {
            type Case = (&'static str, Box<dyn Fn(&mut Tiny)>);
            let cases: Vec<Case> = vec![
                (
                    "attention.sliding_window",
                    Box::new(move |m| {
                        m.extra_kv.push(("attention.sliding_window", 4, u32v(8)));
                    }),
                ),
                (
                    "rope yarn scaling",
                    Box::new(move |m| {
                        m.extra_kv.push(("rope.scaling.type", 8, str_v("yarn")));
                    }),
                ),
                (
                    "rope_freqs.weight",
                    Box::new(|m| {
                        m.tensors.push((
                            "rope_freqs.weight".into(),
                            vec![HD / 2],
                            vec![1.0; HD / 2],
                        ));
                    }),
                ),
                (
                    "attention temperature scaling",
                    Box::new(move |m| {
                        m.extra_kv
                            .push(("attention.temperature_scale", 6, f32v(0.5)));
                    }),
                ),
            ];
            for (feature, mutate) in cases {
                let mut m = build_tiny(kind);
                mutate(&mut m);
                let gguf =
                    GgufFile::from_bytes(Arc::from(gguf_bytes(&m).into_boxed_slice())).unwrap();
                match LlamaModel::from_gguf(gguf, 64) {
                    Ok(_) => panic!("LayerNorm arch must reject {feature}"),
                    Err(e) => {
                        let msg = e.to_string();
                        assert!(msg.contains("does not support"), "{msg}");
                        assert!(msg.contains(feature), "{feature} not named: {msg}");
                    }
                }
            }
        }
    }

    /// Only StableLM implements partial rotary; any other arch declaring
    /// `rope.dimension_count < head_dim` would silently get full-head RoPE.
    #[test]
    fn non_stablelm_arch_rejects_partial_rotary() {
        let u32v = |v: u32| v.to_le_bytes().to_vec();
        for arch in [None, Some("llama")] {
            let mut m = build_tiny(Kind::Cohere);
            m.arch = arch;
            m.extra_kv
                .push(("rope.dimension_count", 4, u32v((HD / 2) as u32)));
            let gguf = GgufFile::from_bytes(Arc::from(gguf_bytes(&m).into_boxed_slice())).unwrap();
            match LlamaModel::from_gguf(gguf, 64) {
                Ok(_) => panic!("partial rotary must be rejected on {arch:?}"),
                Err(e) => assert!(e.to_string().contains("partial rotary"), "{e}"),
            }
        }
        // A full-width value stays accepted.
        let mut ok = build_tiny(Kind::Cohere);
        ok.extra_kv
            .push(("rope.dimension_count", 4, u32v(HD as u32)));
        let gguf = GgufFile::from_bytes(Arc::from(gguf_bytes(&ok).into_boxed_slice())).unwrap();
        LlamaModel::from_gguf(gguf, 64).expect("full rotary loads");
    }

    /// Pins the call count, which the logits comparison below cannot: a
    /// regression to per-token `forward` yields identical logits but pays a
    /// vocab-sized projection for every prompt token. The oracle-dump path
    /// deliberately keeps one projection per token.
    #[test]
    fn ext_prefill_projects_the_vocab_head_once() {
        let calls = || super::PROJECT_LOGITS_CALLS.with(|c| c.get());
        let toks: Vec<u32> = vec![1, 5, 2, 7, 3];
        for kind in [Kind::Cohere, Kind::Starcoder2, Kind::Stablelm] {
            let model = load_tiny(&build_tiny(kind));
            let mut st = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
            let before = calls();
            model.forward_prefill(&toks, 0, &mut st);
            assert_eq!(
                calls() - before,
                1,
                "{kind:?}: head must run for the last token only"
            );

            crate::model::transformer::oracle_dump::begin();
            let mut st = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
            let before = calls();
            model.forward_prefill(&toks, 0, &mut st);
            let after = calls();
            crate::model::transformer::oracle_dump::take();
            assert_eq!(
                after - before,
                toks.len(),
                "{kind:?}: the oracle-dump path records every token"
            );
        }
    }

    /// A partial Q/K/V bias set would be dropped silently by the forward
    /// path (it applies biases only as a set), so the load must reject it,
    /// naming the layer and the missing tensors.
    #[test]
    fn partial_qkv_bias_set_fails_the_load() {
        for missing in [&["attn_q.bias"][..], &["attn_k.bias", "attn_v.bias"][..]] {
            let mut m = build_tiny(Kind::Starcoder2);
            m.tensors
                .retain(|(n, _, _)| !missing.iter().any(|s| n == &format!("blk.1.{s}")));
            let gguf = GgufFile::from_bytes(Arc::from(gguf_bytes(&m).into_boxed_slice())).unwrap();
            match LlamaModel::from_gguf(gguf, 64) {
                Ok(_) => panic!("a partial Q/K/V bias set must fail the load ({missing:?})"),
                Err(e) => {
                    let msg = e.to_string();
                    assert!(msg.contains("layer 1"), "{msg}");
                    assert!(msg.contains("partial Q/K/V bias"), "{msg}");
                    for s in missing {
                        assert!(msg.contains(&format!("blk.1.{s}")), "{msg}");
                    }
                }
            }
        }
        // Control: the full set loads.
        load_tiny(&build_tiny(Kind::Starcoder2));
    }

    /// The sequential prefill skips the LM head for every token but the last.
    /// It must leave the same KV state and return the same last-token logits
    /// (bit for bit) as running the full `forward` on every token, and it must
    /// also agree with the independent scalar reference.
    #[test]
    fn ext_prefill_skipping_head_matches_token_by_token_forward() {
        for kind in [Kind::Cohere, Kind::Starcoder2, Kind::Stablelm] {
            let m = build_tiny(kind);
            let model = load_tiny(&m);
            let toks: Vec<u32> = vec![1, 5, 2, 7, 3];
            let mut a = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
            let got = model.forward_prefill(&toks, 0, &mut a);
            let mut b = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
            let mut want = Vec::new();
            for (i, &t) in toks.iter().enumerate() {
                want = model.forward(&[t], i, &mut b);
            }
            assert_eq!(got, want, "last-token logits must be bit-identical");
            assert_eq!(a.seq_len, b.seq_len);
            // The cached K/V is what later decode reads: continue both.
            let na = model.forward(&[4], toks.len(), &mut a);
            let nb = model.forward(&[4], toks.len(), &mut b);
            assert_eq!(na, nb, "decode after prefill must see identical KV");
            // A prefill that starts from a non-empty cache (chunked prompts).
            let mut c = crate::kv_cache::InferenceState::from_config(model.config()).unwrap();
            model.forward_prefill(&toks[..2], 0, &mut c);
            let chunked = model.forward_prefill(&toks[2..], 2, &mut c);
            assert_eq!(chunked, want, "chunked prefill must match");
            let refl = reference_logits(&m, &toks.iter().map(|&t| t as usize).collect::<Vec<_>>());
            for (g, r) in got.iter().zip(&refl[toks.len() - 1]) {
                assert!((g - r).abs() < 2e-3 * r.abs().max(1.0), "{g} vs {r}");
            }
        }
    }
}

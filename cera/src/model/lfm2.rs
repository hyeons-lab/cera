// LFM2 / LFM2.5 hybrid conv+attention model.

use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvPrefixCache, LayerState};
use crate::model::transformer::{self, FfnWeights};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
// DType's only remaining uses here are the aarch64 per-token GEMV dispatch
// (`forward_conv_block`'s Q4_0/Q8_0 checks). The prefill gates used to name dtypes
// directly too, but they now ask `transformer::batched_gemm_supports`, which takes
// the dtype as a parameter — so on x86_64 + `blas` this import became unused. Gate
// it to where it is actually referenced.
#[cfg(target_arch = "aarch64")]
use crate::tensor::DType;
use crate::time::{Duration, Instant};
use crate::turboquant;

#[cfg(test)]
thread_local! {
    pub(crate) static PREFILLED_TOKENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(has_blas)]
thread_local! {
    static PAR_EXPERT_GATE_UP_ROWS: std::cell::Cell<Option<Vec<f32>>> =
        const { std::cell::Cell::new(None) };
    static PAR_EXPERT_DEQUANT_SCRATCH: std::cell::Cell<Option<Vec<f32>>> =
        const { std::cell::Cell::new(None) };
}

/// Run `f` with the TLS expert scratch buffer held by `tls`.
///
/// `Cell::take()`, not `with_borrow_mut`: the expert closure calls the
/// `dequantize_*_matrix` helpers, which fan out over rayon themselves, and
/// a worker waiting in the inner join steals outer expert tasks — so the
/// expert closure re-enters on one thread and a `RefCell` borrow panics
/// ("already borrowed", flaky). A nested call finds the slot empty and
/// degrades to one temporary allocation. Mirrors `with_gemv_scratch`.
#[cfg(has_blas)]
#[inline]
fn with_par_expert_scratch<R>(
    tls: &'static std::thread::LocalKey<std::cell::Cell<Option<Vec<f32>>>>,
    f: impl FnOnce(&mut Vec<f32>) -> R,
) -> R {
    tls.with(|cell| {
        let mut buf = cell.take().unwrap_or_default();
        let res = f(&mut buf);
        cell.set(Some(buf));
        res
    })
}

// ── Pre-resolved weight reference ───────────────────────────────────────────

// The pre-resolved mmap weight reference is the arch-agnostic one from
// `transformer.rs`; LFM2 shares it (and the weight-plumbing helpers below) so the
// type and kernels have a single definition. `gpu_lfm2.rs` / `metal_lfm2.rs` keep
// referring to `lfm2::WeightRef` via this re-export.
pub(crate) use transformer::WeightRef;

/// The three SwiGLU projections of a dense feed-forward block.
#[derive(Debug, Clone)]
pub struct DenseFfnRefs {
    pub gate: WeightRef,
    pub up: WeightRef,
    pub down: WeightRef,
    #[allow(dead_code)]
    pub gate_up_f32: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
}

/// Weight references for one mixture-of-experts feed-forward block.
///
/// The GGUF stores each layer's experts stacked into rank-3 tensors; these are
/// pre-split into one ordinary 2D [`WeightRef`] per expert at load time (see
/// [`crate::gguf::GgufFile::tensor_meta_expert`]) so the existing GEMV kernels
/// run against an expert unchanged. `gate`, `up` and `down` are each
/// `n_expert` long and index-aligned.
#[derive(Debug, Clone)]
pub struct MoeFfnRefs {
    /// Routing parameters for this layer, copied out of
    /// [`crate::model::MoeConfig`] at load.
    ///
    /// Carried here rather than read from `ModelConfig::moe` at the call site so
    /// that having expert weights and having the numbers to drive them is one
    /// fact instead of two. The alternative left the hot path with an
    /// `if let Some(cfg)` whose `else` would silently skip the whole FFN and add
    /// the *previous* block's stale scratch into the residual.
    ///
    /// All three are flattened rather than held as a `MoeConfig`: that struct
    /// also carries the per-model `is_moe_layer` map, which is both meaningless
    /// inside a per-layer struct (this layer is routed by virtue of being an
    /// `FfnRefs::Moe`) and a heap allocation per routed layer to clone.
    pub n_expert: usize,
    /// Experts activated per token. See [`Self::n_expert`].
    pub n_expert_used: usize,
    /// Per-expert feed-forward width. See [`Self::n_expert`].
    pub expert_ff_len: usize,
    /// Router projection (`ffn_gate_inp.weight`, F32, `[hidden, n_expert]`)
    /// producing one logit per expert.
    pub router: WeightRef,
    /// Per-expert selection bias (`exp_probs_b.bias`, F32, `n_expert` long).
    ///
    /// Added to the gate probabilities **only** to pick the top-k experts; the
    /// weights those experts are then combined with are the *unbiased*
    /// probabilities. This is the DeepSeek-V3 convention llama.cpp implements
    /// ("leave probs unbiased as it's later used to get expert weights"), and
    /// conflating the two still yields fluent output, so it is pinned by the
    /// routing unit tests rather than left to inspection.
    pub exp_probs_b: Vec<f32>,
    pub gate: Vec<WeightRef>,
    pub up: Vec<WeightRef>,
    pub down: Vec<WeightRef>,
    #[allow(dead_code)]
    pub gate_up_f32: Vec<std::sync::Arc<std::sync::OnceLock<Vec<f32>>>>,
}

/// A layer's feed-forward block: one dense SwiGLU, or a routed expert set.
///
/// `lfm2moe` mixes both (its leading blocks are dense, the rest MoE),
/// so this is per-layer rather than per-model. Modelled as a sum type instead
/// of optional dense fields plus optional expert fields so that "exactly one of
/// the two is populated" is a type-level fact rather than a load-time
/// invariant the hot paths would have to re-check.
#[derive(Debug, Clone)]
pub enum FfnRefs {
    Dense(DenseFfnRefs),
    /// Boxed so the enum's size is set by the dense variant rather than by the
    /// larger MoE one, which every dense layer of every model would otherwise
    /// carry as padding.
    Moe(Box<MoeFfnRefs>),
}

impl FfnRefs {
    /// The dense projections, or an error naming the layer when it is MoE.
    ///
    /// Every backend now implements experts, so this is no longer a
    /// not-yet-supported path: it is the accessor a caller reaches for once it
    /// has established the layer is dense, and the error is what keeps a caller
    /// that has *not* from silently falling back onto the wrong weights. Both
    /// GPU loaders ask `GpuWeightSource::moe_refs` first for exactly that
    /// reason.
    pub fn dense(&self) -> Result<&DenseFfnRefs> {
        match self {
            Self::Dense(d) => Ok(d),
            Self::Moe(_) => bail!("expected a dense FFN block, found a mixture-of-experts block"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FusedAttnProjections {
    Q40All {
        packed: Vec<u8>,
        scales: Vec<f32>,
        smmla: bool,
    },
    Q40QkOnly {
        packed: Vec<u8>,
        scales: Vec<f32>,
        smmla: bool,
    },
    Q4KAll {
        packed: Vec<u8>,
        dsc: Vec<f32>,
        dmn: Vec<f32>,
        smmla: bool,
    },
    Q4KQkOnly {
        packed: Vec<u8>,
        dsc: Vec<f32>,
        dmn: Vec<f32>,
        smmla: bool,
    },
    Q6KAll {
        packed: Vec<u8>,
        scales: Vec<f32>,
        smmla: bool,
    },
    Q6KQkOnly {
        packed: Vec<u8>,
        scales: Vec<f32>,
        smmla: bool,
    },
}

/// Per-layer weight references for quantized tensors.
#[derive(Debug, Clone)]
pub struct LayerWeightRefs {
    pub ffn: FfnRefs,
    pub shortconv_in_proj: Option<WeightRef>,
    pub shortconv_out_proj: Option<WeightRef>,
    pub attn_q: Option<WeightRef>,
    pub attn_k: Option<WeightRef>,
    pub attn_v: Option<WeightRef>,
    pub attn_output: Option<WeightRef>,
    #[allow(dead_code)]
    pub qkv_f32: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
    #[allow(dead_code, clippy::type_complexity)]
    pub qkv_repacked: std::sync::Arc<std::sync::OnceLock<Option<FusedAttnProjections>>>,
}

/// Upper bound on attention heads for `lfm3`, so the per-head gate lives on the
/// stack in the decode path. The checkpoint has 16.
pub(crate) const MAX_LFM3_HEADS: usize = 128;

/// A `u32` metadata value that must be present as exactly that type or absent: `get_u32`
/// reads another integer type (llama.cpp's reader accepts any) as absent, which would let a
/// mistyped head width, value width or rotary width pass as the default.
fn lfm3_u32(gguf: &GgufFile, key: &str) -> Result<Option<u32>> {
    match gguf.metadata.get(key) {
        None => Ok(None),
        Some(crate::gguf::GgufValue::U32(v)) => Ok(Some(*v)),
        Some(_) => bail!("{key} must be stored as a u32"),
    }
}

/// The `f32` twin of [`lfm3_u32`]: a mistyped rope base must fail, not read as its default.
fn lfm3_f32(gguf: &GgufFile, key: &str) -> Result<Option<f32>> {
    match gguf.metadata.get(key) {
        None => Ok(None),
        Some(crate::gguf::GgufValue::F32(v)) => Ok(Some(*v)),
        Some(_) => bail!("{key} must be stored as an f32"),
    }
}

/// A `bool` metadata value that must be present as exactly that type or absent: `get_bool`
/// reads another type as absent, which would let a mistyped flag load as its default.
fn lfm3_bool(gguf: &GgufFile, key: &str) -> Result<Option<bool>> {
    match gguf.metadata.get(key) {
        None => Ok(None),
        Some(crate::gguf::GgufValue::Bool(v)) => Ok(Some(*v)),
        Some(_) => bail!("{key} must be stored as a bool"),
    }
}

/// `acc + a * b` as decode's short-conv accumulates it: fused on aarch64 (the NEON
/// kernel uses `vfmaq_f32`), a separate multiply and add elsewhere (the scalar loop).
/// The multi-tap prefill conv uses it so both paths round alike.
// Its callers (the multi-tap batched prefill) are compiled only where a batched GEMM
// exists: aarch64, x86_64 or BLAS.
#[cfg_attr(
    not(any(target_arch = "aarch64", target_arch = "x86_64", has_blas)),
    allow(dead_code)
)]
#[inline(always)]
fn conv_mac(acc: f32, a: f32, b: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, acc)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        acc + a * b
    }
}

/// What `lfm3` adds to the LFM2 graph, per layer.
///
/// Every layer may carry sandwich norms (an RMSNorm over the conv or attention
/// output, and another over the FFN output, each applied before its residual
/// add when `use_sandwich_norm` is enabled). Attention layers also carry a
/// per-head sigmoid gate on the attention output, a sliding window (or none,
/// for global layers), and their own RoPE schedule.
#[derive(Debug)]
struct Lfm3Layer {
    post_operator_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
    attn: Option<Lfm3Attn>,
}

#[derive(Debug)]
struct Lfm3Attn {
    /// `[n_heads, hidden_size]` projection of the normed block input; its
    /// sigmoid scales each head's attention output.
    gate: WeightRef,
    /// Keys visible to a query, itself included. `None` on a global layer.
    window: Option<usize>,
    /// Frequency base for this layer: `rope.freq_base_swa` on a sliding-window
    /// layer, `rope.freq_base` on a global one.
    rope_theta: f32,
    /// `rope_freqs.weight` for a global layer (partial rotary: the unrotated
    /// pairs carry a huge factor). Sliding-window layers rotate the whole head.
    rope_freq_factors: Option<Vec<f32>>,
}

/// Which `lfm3` layers slide and how each rotates, read from the GGUF metadata.
#[derive(Debug)]
pub(crate) struct Lfm3Layout {
    /// Per layer: sliding-window attention (`true`) or global (`false`).
    pub(crate) swa_pattern: Vec<bool>,
    /// Keys visible to a query on a sliding layer, itself included.
    pub(crate) window: Option<usize>,
    /// RoPE base of the sliding layers (`rope.freq_base_swa`).
    pub(crate) theta_swa: f32,
    /// `rope_freqs.weight`: the partial-rotary mask of the global layers
    /// (the unrotated pairs carry a huge factor). `None` with no global layer.
    pub(crate) rope_freq_factors: Option<Vec<f32>>,
}

/// The `lfm3` architecture's deltas over `lfm2`. `None` on every other arch.
#[derive(Debug)]
struct Lfm3Extras {
    layers: Vec<Lfm3Layer>,
}

// ── LFM Model ──────────────────────────────────────────────────────────────

/// The CPU model of the LFM family: `lfm2`, the routed `lfm2moe`, and any later
/// architecture that shares the hybrid attention + gated-convolution block layout.
pub struct LfmModel {
    gguf: GgufFile,
    config: ModelConfig,
    /// Whether a bidirectional pass must take
    /// [`Self::hidden_states_float_bidirectional`]. Decided once at load (tensor
    /// dtypes never change), so the per-call gate stays a field read.
    needs_bidirectional_float_path: bool,
    // Pre-dequantized small F32 weights
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    #[allow(dead_code)]
    conv_weights: Vec<Option<Vec<f32>>>,
    #[allow(dead_code)]
    conv_weights_transposed: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs
    embd_ref: WeightRef,
    /// The logit projection: for `lfm3`, `output.weight` when the file carries one
    /// (a checkpoint converted with an explicit `lm_head` does, even when it equals
    /// the embedding); otherwise the tied embedding.
    logits_ref: WeightRef,
    layer_refs: Vec<LayerWeightRefs>,
    /// `Some` only for the `lfm3` architecture.
    lfm3: Option<Lfm3Extras>,
    /// Caller namespace, typically a GGUF path. Cold use also binds it to the
    /// loaded bytes; CPU and KV-format prefixes separate incompatible snapshots.
    /// An empty ID disables the cold tier while keeping the private warm cache.
    model_id: String,
    /// Resolved only when the caller enables disk caching; never on decode.
    #[cfg(feature = "disk-cache")]
    persistent_id: std::sync::OnceLock<String>,
    /// Two-tier prefix cache (warm in-memory + cold on-disk via
    /// FlatBuffers). Replaced wholesale by `Model::configure_cache`.
    /// Defaults to `KvCacheConfig::default()` (warm-only) at
    /// construction time so warm hits work without explicit config.
    prefix_cache: Mutex<KvPrefixCache>,
    /// Prefix-cache namespace tag for the KV mode this model's sessions use
    /// (`KvCompression::cache_tag`). `None` until `configure_kv_compression` runs;
    /// the empty string is the tag for the f32 default, which is why this is an
    /// `Option` rather than just `""`.
    ///
    /// The model owns this, but the mode is a *per-session* knob — so the two only
    /// stay consistent because configuration is first-call-wins. See
    /// `configure_kv_compression`.
    kv_cache_tag: Mutex<Option<String>>,
    classifier_weight: Option<Vec<f32>>,
    classifier_bias: Option<Vec<f32>>,
}

/// Choose the top-`n_used` experts and their combining weights.
///
/// `probs` are the sigmoid gate probabilities and `biases` the per-expert
/// selection bias (`exp_probs_b`). Experts are *ranked* by `probs + biases` but
/// *weighted* by `probs` alone, then renormalized to sum to 1. This is the DeepSeek-V3
/// convention llama.cpp implements in `build_moe_ffn`.
///
/// Split out of [`LfmModel::route_experts`] as a pure function purely so this
/// rule is directly testable: the bias affects rank only, which means the
/// returned weights are *not* necessarily descending, and a version that
/// weights by the biased score produces plausible text while being wrong.
pub(crate) fn select_experts(
    probs: &[f32],
    biases: &[f32],
    n_used: usize,
    selected: &mut Vec<(usize, f32)>,
) {
    selected.clear();
    let n_expert = probs.len().min(biases.len());
    let mut stack_biased = [0.0f32; 256];
    let heap_biased;
    let biased: &[f32] = if n_expert <= 256 {
        for i in 0..n_expert {
            stack_biased[i] = probs[i] + biases[i];
        }
        &stack_biased[..n_expert]
    } else {
        heap_biased = probs
            .iter()
            .zip(biases)
            .map(|(&p, &b)| p + b)
            .collect::<Vec<_>>();
        &heap_biased[..]
    };
    (0..n_used.min(n_expert)).for_each(|_| {
        let best = (0..n_expert)
            .filter(|e| !selected.iter().any(|(taken, _)| taken == e))
            .max_by(|&a, &b| {
                // `total_cmp`, not `partial_cmp`: a NaN logit would otherwise
                // make `max_by` silently return an arbitrary expert.
                biased[a]
                    .total_cmp(&biased[b])
                    // Ties go to the lower index, matching the stable order of
                    // `ggml_argsort_top_k`. `max_by` keeps the *last* maximum,
                    // so the reversed index comparison is what makes it the
                    // first.
                    .then(b.cmp(&a))
            });
        if let Some(e) = best {
            selected.push((e, probs[e]));
        }
    });

    // Renormalize the unbiased probabilities over the chosen experts, clamping
    // the divisor to f16's smallest positive normal (2^-14) exactly as
    // llama.cpp does so an all-but-zero gate cannot divide by zero. Spelled as
    // an exponent rather than the decimal expansion because 6.103515625e-5
    // carries more digits than an f32 literal keeps, which clippy rejects.
    const MIN_POSITIVE_F16: f32 = 1.0 / 16384.0;
    let denom = selected
        .iter()
        .map(|&(_, w)| w)
        .sum::<f32>()
        .max(MIN_POSITIVE_F16);
    selected.iter_mut().for_each(|(_, w)| *w /= denom);
}

/// Render a cache tag for the `KvCompressionConflict` message — the empty tag is
/// the f32 default, which reads better spelled out.
fn describe_cache_tag(tag: &str) -> String {
    if tag.is_empty() {
        "f32".to_string()
    } else {
        tag.trim_end_matches(':').to_string()
    }
}

/// Range-check `{arch}.shortconv.l_cache` at the point it enters from GGUF
/// metadata, so every downstream consumer can trust it.
///
/// The GPU short-conv kernels stage a channel's weights and rolling state in
/// fixed-size registers sized for `kernel_size <= 4`
/// (`conv1d_fused_batch.{wgsl,metal,slang}`), and `d_conv = kernel_size - 1`
/// underflows for 0. Both used to be handled per-kernel: the batched shader
/// silently returned on out-of-range params, which turned a malformed model into
/// wrong prefill logits rather than an error, and the CPU cache asserted the
/// lower bound only. Checking the range once here means a bad value is a load
/// failure with a clear message, and the kernels can index on the invariant.
///
/// `max` is [`GPU_CONV_KERNEL_MAX`] for `lfm2` / `lfm2moe`, whose files the GPU
/// loaders also read through this parser. `lfm3` uses [`CPU_CONV_KERNEL_MAX`]:
/// its checkpoint has `l_cache = 7`, the CPU conv loops are generic in the tap
/// count, and the GPU loaders refuse the arch before they get here.
///
/// Every shipped LFM2 GGUF sets `l_cache = 3`.
fn validate_conv_kernel_size(
    arch: &str,
    v: Option<usize>,
    max: usize,
) -> anyhow::Result<Option<usize>> {
    if let Some(k) = v {
        anyhow::ensure!(
            (2..=max).contains(&k),
            "{arch}.shortconv.l_cache must be in 2..={max}, got {k}"
        );
    }
    Ok(v)
}

/// Whether a GGUF's attention is causal: `is_causal` is cera's key,
/// `attention.causal` the one llama.cpp's converter writes for the bidirectional
/// encoders (Embedding, ColBERT), and a token classification model without either
/// key defaults to bidirectional attention.
pub(crate) fn gguf_is_causal(gguf: &GgufFile, prefix: &str) -> bool {
    if let Some(causal) = gguf
        .get_bool(&format!("{prefix}.is_causal"))
        .or_else(|| gguf.get_bool(&format!("{prefix}.attention.causal")))
    {
        causal
    } else {
        !gguf
            .tensors
            .get("classifier.weight")
            .is_some_and(|t| t.size_bytes > 0)
    }
}

/// Widest short-conv kernel the GPU kernels' fixed registers hold.
const GPU_CONV_KERNEL_MAX: usize = 4;

/// Widest short-conv kernel the CPU path accepts (a sanity bound, not a limit of
/// the loops).
pub(crate) const CPU_CONV_KERNEL_MAX: usize = 16;

impl LfmModel {
    /// Record a node's sum for the llama.cpp parity oracle, as llama.cpp's
    /// `llama-eval-callback` names it (`<name>-<layer>`). A no-op unless a dump
    /// is active, so the name is only formatted for the oracle tests. The names keep
    /// llama.cpp's literal `{}` placeholder (`model.layers.{}.post_ffn_norm`).
    #[inline]
    fn dump_node(name: &str, layer: usize, data: &[f32]) {
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record(&format!("{name}-{layer}"), data);
        }
    }

    /// Whether this call attends causally. The encoder and classifier branches (selected by a
    /// non-causal file or a classifier adapter on the state) are written for LFM2: three conv
    /// taps and none of lfm3's sandwich norms, gate, windows or rope factors. `lfm3` is always
    /// causal whatever the state holds, because `InferenceState.lora` is a public field that
    /// a direct caller can set past every entry guard; the loader refuses non-causal files.
    fn is_causal_for(&self, state: &InferenceState) -> bool {
        self.lfm3.is_some()
            || (self.config.is_causal && !state.lora.as_ref().is_some_and(|l| l.is_classifier()))
    }

    /// Whether `layer` uses bidirectional (non-causal) prefill attention.
    /// For LFM2 `d1`, every attention layer is bidirectional when `!config.is_causal`.
    /// For LFM3 `d1`, only global attention layers (`window.is_none()`) are bidirectional;
    /// sliding-window attention layers remain causal with their configured window.
    pub(crate) fn layer_is_non_causal(&self, layer: usize) -> bool {
        if self.config.is_causal {
            return false;
        }
        if let Some(lfm3) = &self.lfm3 {
            return lfm3.layers[layer]
                .attn
                .as_ref()
                .is_some_and(|a| a.window.is_none());
        }
        self.config.block_types[layer] == BlockType::Attention
    }

    fn check_rewind_mode(
        &self,
        state: &InferenceState,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        if !self.is_causal_for(state) || !self.config.is_causal {
            return Err(crate::kv_cache::KvRewindError::NonCausal);
        }
        Ok(())
    }

    /// Warm-only use needs no content scan. Resolve the loaded-byte identity
    /// once when disk caching is enabled, reusing the model's retained GGUF.
    fn cache_model_id(&self, cold: bool) -> &str {
        #[cfg(feature = "disk-cache")]
        if cold && !self.model_id.is_empty() {
            return self.persistent_id.get_or_init(|| {
                super::cache_identity::for_loaded_weights(&self.gguf, &self.model_id)
            });
        }
        let _ = cold;
        &self.model_id
    }

    /// The namespace string itself, taking the tag by value so a caller already
    /// holding the tag lock doesn't have to re-acquire it (see
    /// `configure_kv_compression`, which updates the tag and the cache atomically).
    fn namespace_for(tag: &str, model_id: &str) -> String {
        format!("cpu:{tag}{model_id}")
    }

    /// Construct without a model identifier. Equivalent to
    /// `from_gguf_with_id(gguf, context_size, "")`. Warm prefix caching works;
    /// cold caching is disabled even when a directory is configured.
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Parse the [`ModelConfig`] for an LFM2 / LFM2MoE / LFM3 GGUF file from its header metadata.
    pub fn parse_config(gguf: &GgufFile, context_size: usize) -> Result<ModelConfig> {
        ensure!(context_size > 0, "context_size must be > 0");
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("lfm2")
            .to_string();
        ensure!(
            matches!(arch.as_str(), "lfm2" | "lfm2moe" | "lfm3"),
            "LfmModel: unsupported architecture {arch}"
        );
        let prefix = arch.as_str();

        let stored_blocks =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .with_context(|| format!("missing {prefix}.block_count"))? as usize;
        // a public d1-omni file stores its head blocks after the trunk's, inside block_count
        let head_blocks = crate::d1::public::stored_head_blocks(gguf, prefix);
        ensure!(
            head_blocks < stored_blocks,
            "{prefix}.decision.block_count ({head_blocks}) leaves no trunk in {prefix}.block_count ({stored_blocks})"
        );
        let n_layers = stored_blocks - head_blocks;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .with_context(|| format!("missing {prefix}.embedding_length"))?
            as usize;
        ensure!(hidden_size > 0, "{prefix}.embedding_length must be > 0");
        let intermediate_size = if let Some(info) = gguf.tensors.get("blk.0.ffn_gate.weight") {
            if info.shape.len() > 1 {
                info.shape[1]
            } else {
                gguf.get_u32(&format!("{prefix}.feed_forward_length"))
                    .with_context(|| format!("missing {prefix}.feed_forward_length"))?
                    as usize
            }
        } else {
            gguf.get_u32(&format!("{prefix}.feed_forward_length"))
                .with_context(|| format!("missing {prefix}.feed_forward_length"))?
                as usize
        };
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .with_context(|| format!("missing {prefix}.attention.head_count"))?
            as usize;
        let vocab_size = if arch == "lfm3" {
            match lfm3_u32(gguf, &format!("{prefix}.vocab_size"))? {
                Some(v) => v as usize,
                None => {
                    let info = gguf
                        .tensors
                        .get("token_embd.weight")
                        .context("missing token_embd.weight (cannot derive lfm3 vocab_size)")?;
                    ensure!(
                        info.shape.len() == 2 && info.shape[1] > 0,
                        "token_embd.weight has unexpected shape {:?}",
                        info.shape
                    );
                    info.shape[1]
                }
            }
        } else {
            gguf.get_u32(&format!("{prefix}.vocab_size"))
                .with_context(|| format!("missing {prefix}.vocab_size"))? as usize
        };
        ensure!(vocab_size > 0, "{prefix}.vocab_size must be > 0");
        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let rope_theta_key = format!("{prefix}.rope.freq_base");
        let rope_theta = if arch == "lfm3" {
            lfm3_f32(gguf, &rope_theta_key)?
        } else {
            gguf.get_f32(&rope_theta_key)
        }
        .unwrap_or(1_000_000.0);
        let eps_key = format!("{prefix}.attention.layer_norm_rms_epsilon");
        let rms_norm_eps = if arch == "lfm3" {
            lfm3_f32(gguf, &eps_key)?
        } else {
            gguf.get_f32(&eps_key)
        }
        .unwrap_or(1e-5);
        let conv_kernel_size = validate_conv_kernel_size(
            prefix,
            gguf.get_u32(&format!("{prefix}.shortconv.l_cache"))
                .map(|v| v as usize),
            if arch == "lfm3" {
                CPU_CONV_KERNEL_MAX
            } else {
                GPU_CONV_KERNEL_MAX
            },
        )?;

        let kv_heads_array = gguf
            .get_i32_array(&format!("{prefix}.attention.head_count_kv"))
            .with_context(|| format!("missing {prefix}.attention.head_count_kv"))?;

        anyhow::ensure!(
            kv_heads_array.len() >= n_layers,
            "head_count_kv array length ({}) < block_count ({n_layers})",
            kv_heads_array.len()
        );

        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for (i, &kv_heads) in kv_heads_array.iter().enumerate().take(n_layers) {
            let is_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
            if is_attn {
                let n_kv = kv_heads as usize;
                anyhow::ensure!(
                    n_kv > 0 && n_heads.is_multiple_of(n_kv),
                    "layer {i}: n_kv_heads ({n_kv}) must be > 0 and divide n_heads ({n_heads})"
                );
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(n_kv);
            } else {
                block_types.push(BlockType::GatedConv);
                kv_heads_per_layer.push(0);
            }
        }

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0);

        // LFM2 derives the head width from the hidden size. `lfm3` states it via
        // `attention.key_length`, which may decouple `head_dim * n_heads` from `hidden_size`.
        ensure!(
            n_heads > 0,
            "{prefix}: attention.head_count must be positive"
        );
        let head_dim = if arch == "lfm3" {
            ensure!(
                n_heads <= MAX_LFM3_HEADS,
                "{prefix}: head_count ({n_heads}) exceeds the supported {MAX_LFM3_HEADS}"
            );
            let key_len = lfm3_u32(gguf, &format!("{prefix}.attention.key_length"))?
                .map_or(hidden_size / n_heads, |v| v as usize);
            ensure!(
                key_len > 0
                    && key_len.is_multiple_of(2)
                    && key_len
                        .checked_mul(n_heads)
                        .is_some_and(|q_dim| q_dim.is_multiple_of(32)),
                "{prefix}: attention.key_length ({key_len}) must be even and \
                 key_length * head_count ({n_heads}) must be a multiple of 32"
            );
            // the attention weights are sized by one head width for q, k, v and o
            ensure!(
                lfm3_u32(gguf, &format!("{prefix}.attention.value_length"))?
                    .is_none_or(|v| v as usize == key_len),
                "{prefix}: attention.value_length must equal attention.key_length ({key_len})"
            );
            let n_rot = lfm3_u32(gguf, &format!("{prefix}.rope.dimension_count"))?
                .map_or(key_len, |v| v as usize);
            ensure!(
                n_rot == key_len,
                "{prefix}: rope.dimension_count ({n_rot}) must equal attention.key_length \
                 ({key_len}); partial rotary is expressed through rope_freqs.weight"
            );
            key_len
        } else {
            hidden_size / n_heads
        };

        let moe_layers: Vec<bool> = (0..n_layers)
            .map(|i| {
                gguf.tensors
                    .contains_key(&format!("blk.{i}.ffn_gate_exps.weight"))
            })
            .collect();

        let moe = if moe_layers.iter().any(|&is_moe| is_moe) {
            if arch == "lfm3" {
                if let Some(n_dense_lead) =
                    lfm3_u32(gguf, &format!("{prefix}.leading_dense_block_count"))?
                {
                    let n_dense_lead = n_dense_lead as usize;
                    ensure!(
                        n_dense_lead <= n_layers,
                        "{prefix}.leading_dense_block_count ({n_dense_lead}) exceeds block_count ({n_layers})"
                    );
                    for (i, &is_moe) in moe_layers.iter().enumerate() {
                        let expected_moe = i >= n_dense_lead;
                        ensure!(
                            is_moe == expected_moe,
                            "layer {i}: expected is_moe={expected_moe} from leading_dense_block_count={n_dense_lead}, found {is_moe}"
                        );
                    }
                }
                if let Some(scale) = lfm3_f32(gguf, &format!("{prefix}.expert_weights_scale"))? {
                    ensure!(
                        scale.is_finite() && (scale - 1.0).abs() <= 1e-6,
                        "{prefix}.expert_weights_scale ({scale}) is not supported (only 1.0)"
                    );
                }
                if let Some(norm) = lfm3_bool(gguf, &format!("{prefix}.expert_weights_norm"))? {
                    ensure!(
                        norm,
                        "{prefix}.expert_weights_norm=false is not supported (only true)"
                    );
                }
            }
            let gating_func = gguf
                .get_u32(&format!("{prefix}.expert_gating_func"))
                .with_context(|| format!("missing {prefix}.expert_gating_func"))?;
            ensure!(
                gating_func == 2,
                "{prefix}: expert_gating_func {gating_func} is not supported (only 2 = sigmoid)"
            );
            let n_expert = gguf
                .get_u32(&format!("{prefix}.expert_count"))
                .with_context(|| format!("missing {prefix}.expert_count"))?
                as usize;
            let n_expert_used = gguf
                .get_u32(&format!("{prefix}.expert_used_count"))
                .with_context(|| format!("missing {prefix}.expert_used_count"))?
                as usize;
            let expert_ff_len = gguf
                .get_u32(&format!("{prefix}.expert_feed_forward_length"))
                .with_context(|| format!("missing {prefix}.expert_feed_forward_length"))?
                as usize;
            ensure!(
                n_expert_used > 0 && n_expert_used <= n_expert,
                "{prefix}: expert_used_count ({n_expert_used}) must be in 1..={n_expert}"
            );
            ensure!(
                expert_ff_len <= intermediate_size,
                "{prefix}: expert_feed_forward_length ({expert_ff_len}) exceeds \
                 feed_forward_length ({intermediate_size}), which sizes the FFN scratch"
            );
            ensure!(
                expert_ff_len > 0 && expert_ff_len.is_multiple_of(32),
                "{prefix}: expert_feed_forward_length ({expert_ff_len}) must be a positive \
                 multiple of 32"
            );
            Some(crate::model::MoeConfig {
                n_expert,
                n_expert_used,
                expert_ff_len,
                is_moe_layer: moe_layers,
            })
        } else {
            None
        };

        let is_causal = gguf_is_causal(gguf, prefix);
        // The bidirectional (encoder / classifier) branches are written for LFM2: they take
        // three conv taps and know nothing of lfm3's sandwich norms, gate or windows, so a
        // non-causal lfm3 file would run them and return plausible, wrong output.
        if arch == "lfm3" {
            // Each cause is refused on its own, so the message names the one that fired.
            // LFM3 d1 decision models (`lfm3.decision.type` = "lfm3-d1" | "lfm3-d1-omni")
            // legitimately set `lfm3.attention.causal = false` to run bidirectional global
            // attention while keeping ShortConv and SWA layers causal.
            let is_lfm3_d1 = matches!(
                gguf.get_str("lfm3.decision.type"),
                Some("lfm3-d1" | "lfm3-d1-omni")
            );
            ensure!(
                is_causal || is_lfm3_d1,
                "{prefix}: lfm3 supports causal language-model files only (this one is bidirectional)"
            );
            ensure!(
                !gguf.tensors.contains_key("classifier.weight"),
                "{prefix}: lfm3 supports causal language-model files only (this one carries \
                 classifier.weight)"
            );
            // A causal key stored as a non-bool type reads as absent above, so check the raw
            // metadata too.
            for key in ["is_causal", "attention.causal"] {
                ensure!(
                    !matches!(
                        gguf.metadata.get(&format!("{prefix}.{key}")),
                        Some(v) if !matches!(v, crate::gguf::GgufValue::Bool(_))
                    ),
                    "{prefix}.{key} must be stored as a bool"
                );
            }
            // `is_causal` wins over `attention.causal` above; a file that states both and
            // disagrees is ambiguous (llama.cpp reads only `attention.causal`), so refuse it.
            if let (Some(own), Some(llama)) = (
                gguf.get_bool(&format!("{prefix}.is_causal")),
                gguf.get_bool(&format!("{prefix}.attention.causal")),
            ) {
                ensure!(
                    own == llama,
                    "{prefix}: is_causal and attention.causal disagree"
                );
            }
        }

        let class_labels: Vec<String> = gguf
            .get_string_array("token_classifier.labels")
            .map(|arr| arr.into_iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        Ok(ModelConfig {
            architecture: arch,
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
            conv_kernel_size,
            ssm: None,
            kv_heads_per_layer,
            scalars: ScalarMultipliers::default(),
            moe,
            is_causal,
            class_labels,
        })
    }

    /// Resolve all layer weight references for an LFM2 model from its GGUF tensor index,
    /// enabling CPU int8 repacking by default.
    pub fn resolve_all_layer_refs(
        gguf: &GgufFile,
        config: &ModelConfig,
    ) -> Result<Vec<LayerWeightRefs>> {
        Self::resolve_all_layer_refs_with_repack(gguf, config, true)
    }

    /// Resolve all layer weight references for an LFM2 model from its GGUF tensor index.
    /// When `repack` is false the CPU int8 repacks are skipped (see
    /// `with_repack_if`): for loaders that only resolve metadata.
    pub fn resolve_all_layer_refs_with_repack(
        gguf: &GgufFile,
        config: &ModelConfig,
        repack: bool,
    ) -> Result<Vec<LayerWeightRefs>> {
        Self::resolve_all_layer_refs_with_loader(gguf, config, repack, |g, name| {
            Ok(g.get_tensor(name)?.try_to_f32_vec()?)
        })
    }

    /// Resolve all layer weight references for an LFM2 model using a custom `load_f32`
    /// callback for small 1D host tensors (such as `blk.{i}.exp_probs_b.bias` on `lfm2moe`),
    /// allowing header-only GGUF sources (like OPFS streaming in WebAssembly) to read
    /// small tensor payloads on demand.
    pub fn resolve_all_layer_refs_with_loader(
        gguf: &GgufFile,
        config: &ModelConfig,
        repack: bool,
        mut load_f32: impl FnMut(&GgufFile, &str) -> Result<Vec<f32>>,
    ) -> Result<Vec<LayerWeightRefs>> {
        let mut layer_refs = Vec::with_capacity(config.n_layers);
        for (i, bt) in config.block_types.iter().enumerate() {
            let is_moe = config
                .moe
                .as_ref()
                .and_then(|m| m.is_moe_layer.get(i).copied())
                .unwrap_or(false);

            let ffn = if is_moe {
                let moe_cfg = config.moe.as_ref().unwrap();
                let n_expert = moe_cfg.n_expert;
                let expert_refs = |suffix: &str| -> Result<Vec<WeightRef>> {
                    let name = format!("blk.{i}.{suffix}");
                    (0..n_expert)
                        .map(|e| Self::resolve_expert_weight(gguf, &name, e))
                        .collect()
                };

                let exp_probs_b = load_f32(gguf, &format!("blk.{i}.exp_probs_b.bias"))?;
                ensure!(
                    exp_probs_b.len() == n_expert,
                    "layer {i}: exp_probs_b has {} entries, expected {n_expert}",
                    exp_probs_b.len()
                );

                FfnRefs::Moe(Box::new(MoeFfnRefs {
                    n_expert,
                    n_expert_used: moe_cfg.n_expert_used,
                    expert_ff_len: moe_cfg.expert_ff_len,
                    router: Self::resolve_weight(gguf, &format!("blk.{i}.ffn_gate_inp.weight"))?,
                    exp_probs_b,
                    gate: expert_refs("ffn_gate_exps.weight")?,
                    up: expert_refs("ffn_up_exps.weight")?,
                    down: expert_refs("ffn_down_exps.weight")?,
                    gate_up_f32: (0..n_expert)
                        .map(|_| std::sync::Arc::new(std::sync::OnceLock::new()))
                        .collect(),
                }))
            } else {
                FfnRefs::Dense(DenseFfnRefs {
                    gate: Self::resolve_weight(gguf, &format!("blk.{i}.ffn_gate.weight"))?
                        .with_repack_if(gguf, repack),
                    up: Self::resolve_weight(gguf, &format!("blk.{i}.ffn_up.weight"))?
                        .with_repack_if(gguf, repack),
                    down: Self::resolve_weight(gguf, &format!("blk.{i}.ffn_down.weight"))?
                        .with_repack_if(gguf, repack),
                    gate_up_f32: std::sync::Arc::new(std::sync::OnceLock::new()),
                })
            };

            let (shortconv_in_proj, shortconv_out_proj, attn_q, attn_k, attn_v, attn_output) =
                if *bt == BlockType::GatedConv {
                    (
                        Some(
                            Self::resolve_weight(
                                gguf,
                                &format!("blk.{i}.shortconv.in_proj.weight"),
                            )?
                            .with_repack_if(gguf, repack),
                        ),
                        Some(
                            Self::resolve_weight(
                                gguf,
                                &format!("blk.{i}.shortconv.out_proj.weight"),
                            )?
                            .with_repack_if(gguf, repack),
                        ),
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        None,
                        None,
                        Some(
                            Self::resolve_weight(gguf, &format!("blk.{i}.attn_q.weight"))?
                                .with_repack_if(gguf, repack),
                        ),
                        Some(
                            Self::resolve_weight(gguf, &format!("blk.{i}.attn_k.weight"))?
                                .with_repack_if(gguf, repack),
                        ),
                        Some(
                            Self::resolve_weight(gguf, &format!("blk.{i}.attn_v.weight"))?
                                .with_repack_if(gguf, repack),
                        ),
                        Some(
                            Self::resolve_weight(gguf, &format!("blk.{i}.attn_output.weight"))?
                                .with_repack_if(gguf, repack),
                        ),
                    )
                };

            layer_refs.push(LayerWeightRefs {
                ffn,
                shortconv_in_proj,
                shortconv_out_proj,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                qkv_f32: std::sync::Arc::new(std::sync::OnceLock::new()),
                qkv_repacked: std::sync::Arc::new(std::sync::OnceLock::new()),
            });
        }
        Ok(layer_refs)
    }

    /// Construct with a caller-supplied identity for persistent prefix caching.
    /// A nonempty ID names a cache namespace, bound to the loaded GGUF bytes when
    /// disk caching is enabled. An empty ID keeps only warm caching.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        Self::from_gguf_impl(gguf, context_size, model_id, true, |g, name| {
            Ok(g.get_tensor(name)?.try_to_f32_vec()?)
        })
    }

    /// Load without the CPU int8 repacks. For the GPU/Metal loaders, which
    /// resolve weight metadata from this model but never dispatch CPU
    /// kernels — the repacks would be gigabytes allocated only to be freed
    /// after upload. Do NOT use for CPU inference (dispatch falls back to
    /// the slow path without them... it stays correct, just slower).
    pub fn from_gguf_with_id_no_repack(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        Self::from_gguf_impl(gguf, context_size, model_id, false, |g, name| {
            Ok(g.get_tensor(name)?.try_to_f32_vec()?)
        })
    }

    /// Load without the CPU int8 repacks (see
    /// [`from_gguf_with_id_no_repack`](Self::from_gguf_with_id_no_repack)).
    pub fn from_gguf_no_repack(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id_no_repack(gguf, context_size, String::new())
    }

    /// Same as [`Self::from_gguf_with_id_no_repack`], but uses a custom `load_f32`
    /// callback for small 1D norm, conv, MoE bias, and classifier tensors so
    /// header-only GGUF sources (such as OPFS streaming in WebAssembly) share the
    /// exact same validation and metadata resolution path.
    pub fn from_gguf_no_repack_with_loader(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
        load_f32: impl FnMut(&GgufFile, &str) -> Result<Vec<f32>>,
    ) -> Result<Self> {
        Self::from_gguf_impl(gguf, context_size, model_id, false, load_f32)
    }

    fn from_gguf_impl(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
        repack: bool,
        mut load_f32: impl FnMut(&GgufFile, &str) -> Result<Vec<f32>>,
    ) -> Result<Self> {
        let config = Self::parse_config(&gguf, context_size)?;
        let n_layers = config.n_layers;
        let hs = config.hidden_size;
        let head_dim = config.head_dim;
        let block_types = config.block_types.clone();

        if config.architecture == "lfm3" {
            Self::validate_lfm3_tensor_shapes(&gguf, &config)?;
        }

        // Pre-extract small F32 weights. The final norm is stored as
        // `token_embd_norm` by LFM2 (its HF name is `embedding_norm`) but as
        // `output_norm` by `lfm3`.
        let output_norm_name = if config.architecture == "lfm3" {
            "output_norm.weight"
        } else if gguf.tensors.contains_key("token_embd_norm.weight")
            || !gguf.tensors.contains_key("output_norm.weight")
        {
            "token_embd_norm.weight"
        } else {
            "output_norm.weight"
        };
        let output_norm_weight = load_f32(&gguf, output_norm_name)?;
        ensure!(
            output_norm_weight.len() == hs,
            "{output_norm_name} length {} != hidden_size ({hs})",
            output_norm_weight.len()
        );

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut conv_weights = Vec::with_capacity(n_layers);
        let mut conv_weights_transposed = Vec::with_capacity(n_layers);

        for (i, bt) in block_types.iter().enumerate() {
            let attn_norm_name = format!("blk.{i}.attn_norm.weight");
            let attn_norm = load_f32(&gguf, &attn_norm_name)?;
            ensure!(
                attn_norm.len() == hs,
                "{attn_norm_name} length {} != hidden_size ({hs})",
                attn_norm.len()
            );
            attn_norm_weights.push(attn_norm);

            let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");
            let ffn_norm = load_f32(&gguf, &ffn_norm_name)?;
            ensure!(
                ffn_norm.len() == hs,
                "{ffn_norm_name} length {} != hidden_size ({hs})",
                ffn_norm.len()
            );
            ffn_norm_weights.push(ffn_norm);

            if *bt == BlockType::Attention {
                let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
                let q_norm = load_f32(&gguf, &q_norm_name)?;
                ensure!(
                    q_norm.len() == head_dim,
                    "{q_norm_name} length {} != head_dim ({head_dim})",
                    q_norm.len()
                );
                let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
                let k_norm = load_f32(&gguf, &k_norm_name)?;
                ensure!(
                    k_norm.len() == head_dim,
                    "{k_norm_name} length {} != head_dim ({head_dim})",
                    k_norm.len()
                );
                attn_q_norm_weights.push(Some(q_norm));
                attn_k_norm_weights.push(Some(k_norm));
                conv_weights.push(None);
                conv_weights_transposed.push(None);
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                let w = load_f32(&gguf, &format!("blk.{i}.shortconv.conv.weight"))?;
                let kernel_size = config.conv_kernel_size.unwrap_or(3);
                ensure!(
                    w.len() == hs * kernel_size,
                    "blk.{i}.shortconv.conv.weight size {} != hidden_size * conv_kernel_size ({} * {})",
                    w.len(),
                    hs,
                    kernel_size
                );
                #[cfg(target_arch = "aarch64")]
                ensure!(
                    hs.is_multiple_of(4),
                    "blk.{i}.shortconv: hidden_size ({hs}) must be a multiple of 4 to support vectorized AArch64 NEON convolution ops",
                );
                // In GGUF, ne[0] is kernel_size and ne[1] is hidden_size, so `w` is channel-first:
                // `w[col * kernel_size + row]` is tap `row` of channel `col`.
                // `conv_weights_transposed` stores channel-first layout [hs, kernel_size].
                // `conv_weights` stores tap-first layout [kernel_size, hs] for prefill slicing.
                let mut tap_first = vec![0.0f32; hs * kernel_size];
                for row in 0..kernel_size {
                    for col in 0..hs {
                        tap_first[row * hs + col] = w[col * kernel_size + row];
                    }
                }
                conv_weights.push(Some(tap_first));
                conv_weights_transposed.push(Some(w));
            }
        }

        let layer_refs =
            Self::resolve_all_layer_refs_with_loader(&gguf, &config, repack, &mut load_f32)?;
        let embd_ref =
            Self::resolve_weight(&gguf, "token_embd.weight")?.with_dec4_if(&gguf, repack);
        ensure!(
            embd_ref.k == hs,
            "token_embd.weight has k={} but hidden_size is {hs}",
            embd_ref.k
        );
        ensure!(
            embd_ref.m >= config.vocab_size,
            "token_embd.weight has {} rows, fewer than vocab_size {}",
            embd_ref.m,
            config.vocab_size
        );
        // Only `lfm3` reads `output.weight`: LFM2 files are tied, and the GPU loaders
        // take the embedding as the logit weight, so honouring it here for lfm2 could
        // make CPU and GPU disagree on a file that carried one.
        let logits_ref =
            if config.architecture == "lfm3" && gguf.tensors.contains_key("output.weight") {
                // its shape against `token_embd.weight` is checked by `validate_lfm3_tensor_shapes`
                Self::resolve_weight(&gguf, "output.weight")?
            } else {
                embd_ref.clone()
            };
        let lfm3 = if config.architecture == "lfm3" {
            Some(Self::load_lfm3_extras(&gguf, &config)?)
        } else {
            None
        };

        let prefix_cache = Mutex::new(KvPrefixCache::for_model(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &model_id,
            &format!("cpu:{model_id}"),
        ));

        let (classifier_weight, classifier_bias) = if gguf.tensors.contains_key("classifier.weight")
        {
            let w = load_f32(&gguf, "classifier.weight")?;
            let b = if gguf.tensors.contains_key("classifier.bias") {
                Some(load_f32(&gguf, "classifier.bias")?)
            } else {
                None
            };
            if hs == 0 || !w.len().is_multiple_of(hs) {
                bail!(
                    "classifier.weight length {} is not a multiple of hidden_size {}",
                    w.len(),
                    hs
                );
            }
            if let Some(ref bias) = b {
                let implied_classes = w.len() / hs;
                if bias.len() != implied_classes {
                    bail!(
                        "classifier.bias length {} does not match implied num_classes {}",
                        bias.len(),
                        implied_classes
                    );
                }
            }
            (Some(w), b)
        } else {
            (None, None)
        };

        // Tensor dtypes never change after load, so the bidirectional routing is
        // decided once here instead of scanning the tensor index on every call.
        let needs_bidirectional_float_path = Self::bidirectional_float_path_needed(&gguf, &config);

        Ok(Self {
            gguf,
            config,
            needs_bidirectional_float_path,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights,
            conv_weights_transposed,
            embd_ref,
            logits_ref,
            layer_refs,
            lfm3,
            model_id,
            #[cfg(feature = "disk-cache")]
            persistent_id: std::sync::OnceLock::new(),
            prefix_cache,
            kv_cache_tag: Mutex::new(None),
            classifier_weight,
            classifier_bias,
        })
    }

    /// The per-layer attention layout of an `lfm3` GGUF, as llama.cpp reads it:
    /// `sliding_window_pattern` is a bool array, `rope.freq_base_swa` applies to the
    /// `true` layers, and the partial-rotary mask is the `rope_freqs.weight` tensor
    /// shared by every global attention layer. Shared by the CPU and NPU loaders so
    /// they cannot disagree about which layer slides or which base it rotates with.
    pub(crate) fn lfm3_layout(gguf: &GgufFile, config: &ModelConfig) -> Result<Lfm3Layout> {
        let n_layers = config.n_layers;
        // A stated window with no readable per-layer pattern would otherwise run every
        // layer as global attention: plausible text, wrong. llama.cpp treats an absent
        // pattern as "no window", so the pattern is only optional when no window is set.
        let swa_pattern = match gguf.get_bool_array("lfm3.attention.sliding_window_pattern") {
            Some(pattern) => pattern,
            None => {
                ensure!(
                    lfm3_u32(gguf, "lfm3.attention.sliding_window")?.is_none(),
                    "lfm3.attention.sliding_window is set but \
                     lfm3.attention.sliding_window_pattern is missing or not a bool array"
                );
                vec![false; n_layers]
            }
        };
        ensure!(
            swa_pattern.len() >= n_layers,
            "lfm3.attention.sliding_window_pattern has {} entries, block_count is {n_layers}",
            swa_pattern.len()
        );
        let window = if swa_pattern.iter().take(n_layers).any(|&swa| swa) {
            let w = lfm3_u32(gguf, "lfm3.attention.sliding_window")?
                .context("missing lfm3.attention.sliding_window")? as usize;
            ensure!(w > 0, "lfm3.attention.sliding_window must be > 0");
            Some(w)
        } else {
            None
        };
        ensure!(
            config.rope_theta.is_finite() && config.rope_theta > 1.0,
            "lfm3.rope.freq_base must be finite and > 1"
        );
        let theta_swa = lfm3_f32(gguf, "lfm3.rope.freq_base_swa")?.unwrap_or(10_000.0);
        ensure!(
            theta_swa.is_finite() && theta_swa > 1.0,
            "lfm3.rope.freq_base_swa must be finite and > 1"
        );

        let global_layers =
            (0..n_layers).any(|i| config.block_types[i] == BlockType::Attention && !swa_pattern[i]);
        let rope_freq_factors = if global_layers {
            let ff = gguf
                .get_tensor("rope_freqs.weight")
                .context("lfm3 global attention layers need rope_freqs.weight")?
                .try_to_f32_vec()
                .context("rope_freqs.weight cannot be read as f32")?;
            ensure!(
                ff.len() == config.head_dim / 2,
                "rope_freqs.weight has {} entries, expected {}",
                ff.len(),
                config.head_dim / 2
            );
            ensure!(
                ff.iter().all(|f| f.is_finite() && *f >= 1e-9),
                "rope_freqs.weight holds a non-finite factor or one below 1e-9 (position / factor \
                 would overflow); an F16 table overflows the 1e30 mask factor to infinity, so \
                 store it as F32"
            );
            Some(ff)
        } else {
            None
        };
        Ok(Lfm3Layout {
            swa_pattern,
            window,
            theta_swa,
            rope_freq_factors,
        })
    }

    /// Checks every lfm3 tensor whose size the graph assumes against the validated config
    /// (embedding, norms, FFN, conv projections, attention, the optional untied head).
    /// Shared by the CPU and Hexagon loaders (the NPU constructor sizes its planes from
    /// "at least" lengths and would otherwise read a short norm or a wide projection on the
    /// device with no error). A short norm makes `cpu::rmsnorm` return early (plausible,
    /// wrong logits); a wide projection or a long norm panics in the first forward.
    pub(crate) fn validate_lfm3_tensor_shapes(gguf: &GgufFile, config: &ModelConfig) -> Result<()> {
        let hs = config.hidden_size;
        let ff = config.intermediate_size;
        let vector = |name: &str, expected: usize| -> Result<()> {
            let len = if gguf.mmap_data().is_empty() {
                let info = gguf
                    .tensors
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;
                info.shape.iter().copied().product::<usize>()
            } else {
                gguf.get_tensor(name)?
                    .try_to_f32_vec()
                    .with_context(|| format!("{name} cannot be read as f32"))?
                    .len()
            };
            ensure!(
                len == expected,
                "{name} has {len} entries, expected {expected}"
            );
            Ok(())
        };
        let matrix = |name: String, rows: usize, cols: usize| -> Result<()> {
            let w = Self::resolve_weight(gguf, &name)?;
            ensure!(
                w.m == rows && w.k == cols,
                "{name} is {}x{}, expected {rows}x{cols}",
                w.m,
                w.k
            );
            Ok(())
        };
        matrix("token_embd.weight".into(), config.vocab_size, hs)?;
        vector("output_norm.weight", hs)?;
        for i in 0..config.n_layers {
            for name in ["attn_norm", "ffn_norm"] {
                vector(&format!("blk.{i}.{name}.weight"), hs)?;
            }
            let post_attn_name = format!("blk.{i}.post_attention_norm.weight");
            let post_ffn_name = format!("blk.{i}.post_ffw_norm.weight");
            let has_post_attn = gguf.tensors.contains_key(&post_attn_name);
            let has_post_ffn = gguf.tensors.contains_key(&post_ffn_name);
            ensure!(
                has_post_attn == has_post_ffn,
                "layer {i}: post_attention_norm.weight and post_ffw_norm.weight must both be present or both absent"
            );
            if has_post_attn {
                vector(&post_attn_name, hs)?;
                vector(&post_ffn_name, hs)?;
            }
            let is_moe = config
                .moe
                .as_ref()
                .and_then(|m| m.is_moe_layer.get(i).copied())
                .unwrap_or(false);
            if is_moe {
                let moe = config.moe.as_ref().unwrap();
                matrix(format!("blk.{i}.ffn_gate_inp.weight"), moe.n_expert, hs)?;
                vector(&format!("blk.{i}.exp_probs_b.bias"), moe.n_expert)?;
                for (name, rows, cols) in [
                    ("ffn_gate_exps", moe.expert_ff_len, hs),
                    ("ffn_up_exps", moe.expert_ff_len, hs),
                    ("ffn_down_exps", hs, moe.expert_ff_len),
                ] {
                    let full_name = format!("blk.{i}.{name}.weight");
                    for e in 0..moe.n_expert {
                        let w = Self::resolve_expert_weight(gguf, &full_name, e)?;
                        ensure!(
                            w.m == rows && w.k == cols,
                            "{full_name} expert {e} is {}x{}, expected {rows}x{cols}",
                            w.m,
                            w.k
                        );
                    }
                }
            } else {
                for (name, rows, cols) in [
                    ("ffn_gate", ff, hs),
                    ("ffn_up", ff, hs),
                    ("ffn_down", hs, ff),
                ] {
                    matrix(format!("blk.{i}.{name}.weight"), rows, cols)?;
                }
            }
            if config.block_types[i] != BlockType::Attention {
                for (name, rows) in [("in_proj", 3 * hs), ("out_proj", hs)] {
                    matrix(format!("blk.{i}.shortconv.{name}.weight"), rows, hs)?;
                }
                continue;
            }
            for name in ["attn_q_norm", "attn_k_norm"] {
                vector(&format!("blk.{i}.{name}.weight"), config.head_dim)?;
            }
            let q_dim = config.n_heads * config.head_dim;
            let kv_dim = config.kv_heads_per_layer[i] * config.head_dim;
            // q, k, v and o are sized by the one head width validated at load
            for (name, rows, cols) in [
                ("attn_gate", config.n_heads, hs),
                ("attn_q", q_dim, hs),
                ("attn_k", kv_dim, hs),
                ("attn_v", kv_dim, hs),
                ("attn_output", hs, q_dim),
            ] {
                matrix(format!("blk.{i}.{name}.weight"), rows, cols)?;
            }
        }
        if gguf.tensors.contains_key("output.weight") {
            let output = Self::resolve_weight(gguf, "output.weight")?;
            let embd = Self::resolve_weight(gguf, "token_embd.weight")?;
            ensure!(
                output.m == embd.m && output.k == embd.k,
                "output.weight is {}x{} but token_embd.weight is {}x{}",
                output.m,
                output.k,
                embd.m,
                embd.k
            );
        }
        Ok(())
    }

    /// Load what `lfm3` adds to the LFM2 graph (see [`Lfm3Extras`]).
    fn load_lfm3_extras(gguf: &GgufFile, config: &ModelConfig) -> Result<Lfm3Extras> {
        let n_layers = config.n_layers;
        let Lfm3Layout {
            swa_pattern,
            window,
            theta_swa,
            rope_freq_factors,
        } = Self::lfm3_layout(gguf, config)?;

        let opt_norm = |name: String| -> Result<Option<Vec<f32>>> {
            gguf.tensors
                .contains_key(&name)
                .then(|| {
                    gguf.get_tensor(&name)?
                        .try_to_f32_vec()
                        .with_context(|| format!("{name} cannot be read as f32"))
                })
                .transpose()
        };

        let layers = (0..n_layers)
            .map(|i| {
                let attn = (config.block_types[i] == BlockType::Attention)
                    .then(|| -> Result<Lfm3Attn> {
                        let gate =
                            Self::resolve_weight(gguf, &format!("blk.{i}.attn_gate.weight"))?;
                        let is_swa = swa_pattern[i];
                        Ok(Lfm3Attn {
                            gate,
                            window: window.filter(|_| is_swa),
                            rope_theta: if is_swa { theta_swa } else { config.rope_theta },
                            rope_freq_factors: rope_freq_factors.clone().filter(|_| !is_swa),
                        })
                    })
                    .transpose()?;
                Ok(Lfm3Layer {
                    post_operator_norm: opt_norm(format!("blk.{i}.post_attention_norm.weight"))?,
                    post_ffn_norm: opt_norm(format!("blk.{i}.post_ffw_norm.weight"))?,
                    attn,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lfm3Extras { layers })
    }

    /// Resolve a tensor name to a pre-computed byte range in the mmap.
    /// Thin wrapper over the shared `transformer::resolve_weight`.
    fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<WeightRef> {
        transformer::resolve_weight(gguf, name)
    }

    /// Resolve expert `expert` of a stacked MoE tensor to a 2D weight ref.
    fn resolve_expert_weight(gguf: &GgufFile, name: &str, expert: usize) -> Result<WeightRef> {
        transformer::resolve_expert_weight(gguf, name, expert)
    }

    /// Pick the experts for one token and their combining weights, leaving both
    /// in `state.scratch.moe_selected`.
    ///
    /// Mirrors llama.cpp's `build_moe_ffn` for `expert_gating_func = 2`:
    ///
    /// 1. `probs = sigmoid(router · x)`
    /// 2. select the top `n_expert_used` by `probs + exp_probs_b`
    /// 3. weight them by the **unbiased** `probs`, renormalized to sum to 1
    ///
    /// Step 3 is the part that is easy to get wrong: `exp_probs_b` is a
    /// DeepSeek-V3 style load-balancing bias that steers *selection* only. Using
    /// the biased score as the combining weight changes every output subtly and
    /// still reads as fluent text, so the `moe_routing_tests` module pins this
    /// against llama.cpp's own values rather than trusting inspection.
    fn route_experts(
        &self,
        layer: usize,
        moe: &MoeFfnRefs,
        lora: Option<&crate::lora::LoraAdapterWeights>,
        ffn_input: &[f32],
        state: &mut InferenceState,
    ) {
        let n_expert = moe.n_expert;
        let n_used = moe.n_expert_used;

        state.scratch.moe_probs.resize(n_expert, 0.0);
        transformer::gemv(
            &self.gguf,
            &moe.router,
            ffn_input,
            &mut state.scratch.moe_probs[..n_expert],
        );
        // The router is a LoRA target in its own right (llama.cpp builds it with
        // `build_lora_mm`), and it is adapted on the *logits*, before the
        // sigmoid. An adapter that moved a token across a selection boundary
        // would otherwise be silently ignored while its expert deltas applied.
        if let Some(lora) = lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGateInp)
        {
            crate::lora::apply_decode(
                t,
                ffn_input,
                &mut state.scratch.moe_probs[..n_expert],
                &mut state.scratch.lora_tmp,
            );
        }
        state.scratch.moe_probs[..n_expert]
            .iter_mut()
            .for_each(|p| *p = 1.0 / (1.0 + (-*p).exp()));

        select_experts(
            &state.scratch.moe_probs[..n_expert],
            &moe.exp_probs_b,
            n_used,
            &mut state.scratch.moe_selected,
        );
    }

    /// Routed feed-forward for one MoE layer, writing the combined expert
    /// output into `state.scratch.out[..hidden_size]`.
    ///
    /// On aarch64 the caller must have quantized `ffn_input` into
    /// `state.scratch.q8_*` already, matching the dense
    /// [`transformer::forward_ffn_block`] contract; each expert's
    /// down-projection re-quantizes its own SwiGLU product.
    ///
    /// The body deliberately duplicates the ~60 lines of
    /// [`transformer::forward_ffn_block`] (the same gate/up GEMV, SwiGLU,
    /// requantize, down GEMV, LoRA sequence) rather than calling it. Sharing
    /// would mean parameterizing the dense helper over per-expert weights, a
    /// per-expert LoRA lookup and an output buffer that is accumulated rather
    /// than written, which puts three new branches in the FFN hot path of every
    /// dense model cera runs so that one architecture can reuse it. Nothing
    /// pins the two copies to each other, so a change to the dense block's
    /// arithmetic has to be mirrored here by hand; if a third routed
    /// architecture lands, that trade stops paying and this should be extracted.
    fn forward_moe_ffn(
        &self,
        layer: usize,
        moe: &MoeFfnRefs,
        hidden_size: usize,
        ffn_input: &[f32],
        state: &mut InferenceState,
    ) {
        let ff = moe.expert_ff_len;
        // Cloned out of `state` up front: the `Arc` bumps a refcount, and the
        // alternative is holding a borrow of `state` across the GEMVs that need
        // it mutably. Same shape as `transformer::forward_ffn_block`.
        let lora = state.lora.clone();
        self.route_experts(layer, moe, lora.as_deref(), ffn_input, state);

        state.scratch.moe_expert_out.resize(hidden_size, 0.0);
        state.scratch.out[..hidden_size].fill(0.0);

        // `moe_selected` is scratch that the expert GEMVs below also borrow
        // from `state`, so take the routing decision out first. It is
        // `n_expert_used` pairs (4 here), not per-expert data.
        let selected = std::mem::take(&mut state.scratch.moe_selected);

        for &(expert, weight) in &selected {
            #[cfg(target_arch = "aarch64")]
            {
                let gate_data = self.weight_data(&moe.gate[expert]);
                let up_data = self.weight_data(&moe.up[expert]);
                cpu::gemv_q4_0_fused2_with_q8(
                    gate_data,
                    up_data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.gate[..ff],
                    &mut state.scratch.up[..ff],
                    ff,
                    hidden_size,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                transformer::gemv(
                    &self.gguf,
                    &moe.gate[expert],
                    ffn_input,
                    &mut state.scratch.gate[..ff],
                );
                transformer::gemv(
                    &self.gguf,
                    &moe.up[expert],
                    ffn_input,
                    &mut state.scratch.up[..ff],
                );
            }

            // Per-expert LoRA on gate/up, before the SwiGLU mul reads both. The
            // adapter's factors are indexed by the *same* expert id as the base
            // weight, matching llama.cpp's `build_lora_mm_id`, which feeds one
            // `ids` tensor to the base and both factors alike.
            if let Some(lora) = &lora {
                if let Some(t) =
                    lora.get_expert(layer, crate::lora::LoraTarget::FfnGateExps, expert)
                {
                    crate::lora::apply_decode(
                        t,
                        ffn_input,
                        &mut state.scratch.gate[..ff],
                        &mut state.scratch.lora_tmp,
                    );
                }
                if let Some(t) = lora.get_expert(layer, crate::lora::LoraTarget::FfnUpExps, expert)
                {
                    crate::lora::apply_decode(
                        t,
                        ffn_input,
                        &mut state.scratch.up[..ff],
                        &mut state.scratch.lora_tmp,
                    );
                }
            }

            cpu::silu_mul_inplace(&mut state.scratch.gate[..ff], &state.scratch.up[..ff]);

            #[cfg(target_arch = "aarch64")]
            {
                // The down projection consumes the SwiGLU product, not the
                // layer input, so it needs its own Q8_0 quantization into down buffers.
                let nb = ff / 32;
                state.scratch.q8_scales_down.resize(nb, 0.0);
                state.scratch.q8_quants_down.resize(ff, 0);
                unsafe {
                    crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                        &state.scratch.gate[..ff],
                        &mut state.scratch.q8_scales_down,
                        &mut state.scratch.q8_quants_down,
                    );
                }
                let down_data = self.weight_data(&moe.down[expert]);
                cpu::gemv_q4_0_with_q8(
                    down_data,
                    &state.scratch.q8_scales_down,
                    &state.scratch.q8_quants_down,
                    &mut state.scratch.moe_expert_out[..hidden_size],
                    hidden_size,
                    ff,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            transformer::gemv(
                &self.gguf,
                &moe.down[expert],
                &state.scratch.gate[..ff],
                &mut state.scratch.moe_expert_out[..hidden_size],
            );

            // LoRA on this expert's down projection. Its input is the SwiGLU
            // product in `gate`, not the layer input, so it must read `gate`
            // *before* the accumulate below consumes `moe_expert_out`.
            if let Some(lora) = &lora
                && let Some(t) =
                    lora.get_expert(layer, crate::lora::LoraTarget::FfnDownExps, expert)
            {
                crate::lora::apply_decode(
                    t,
                    &state.scratch.gate[..ff],
                    &mut state.scratch.moe_expert_out[..hidden_size],
                    &mut state.scratch.lora_tmp,
                );
            }

            let (out, expert_out) = (
                &mut state.scratch.out[..hidden_size],
                &state.scratch.moe_expert_out[..hidden_size],
            );
            out.iter_mut()
                .zip(expert_out)
                .for_each(|(acc, &v)| *acc += weight * v);
        }

        state.scratch.moe_selected = selected;
    }

    /// Prefill feed-forward for a MoE layer: route and run every token
    /// individually, reading `ffn_input` and writing `ffn_out` in the caller's
    /// column-major (`hs × n`) layout.
    ///
    /// Deliberately unbatched. The dense path batches because all `n` tokens
    /// share one weight matrix; here each token picks its own 4-of-32 experts,
    /// so a batched GEMM would first have to group tokens by expert and scatter
    /// the results back. That grouping is the next optimization (plan
    /// 000312-01, phase 1 step 6) and is kept separate from making the
    /// arithmetic right, so the batched version has a reference to match.
    #[allow(clippy::too_many_arguments)]
    fn prefill_moe_ffn(
        &self,
        layer: usize,
        moe: &MoeFfnRefs,
        hs: usize,
        n: usize,
        ffn_input: &[f32],
        ffn_out: &mut [f32],
        col: &mut [f32],
        state: &mut InferenceState,
    ) {
        if n <= 1 || state.lora.is_some() {
            for j in 0..n {
                (0..hs).for_each(|i| col[i] = ffn_input[j * hs + i]);

                #[cfg(target_arch = "aarch64")]
                Self::quantize_to_scratch(col, state);

                self.forward_moe_ffn(layer, moe, hs, col, state);

                (0..hs).for_each(|i| ffn_out[j * hs + i] = state.scratch.out[i]);
            }
            return;
        }

        let n_expert = moe.n_expert;
        let n_used = moe.n_expert_used;
        let ff = moe.expert_ff_len;

        let mut all_router_logits = vec![0.0f32; n * n_expert];
        // Gate first: `try_blas_prefill_gemm_rowmajor` reports (and debug-panics
        // on) a dtype it cannot run, e.g. an F32 router, even though this caller
        // checks the bool and falls back to GEMV. The dense prefill blocks gate
        // the same way; the gate, not the bool, is the decline path.
        #[cfg(has_blas)]
        let blas_ok = transformer::batched_gemm_supports(moe.router.dtype, hs)
            && transformer::try_blas_prefill_gemm_rowmajor(
                &self.gguf,
                &moe.router,
                ffn_input,
                &mut all_router_logits,
                n,
                n_expert,
                hs,
                &mut state.scratch.dequant_weight_scratch,
            );
        #[cfg(not(has_blas))]
        let blas_ok = false;

        if !blas_ok {
            for j in 0..n {
                let x = &ffn_input[j * hs..(j + 1) * hs];
                let out_slice = &mut all_router_logits[j * n_expert..(j + 1) * n_expert];
                self.gemv(&moe.router, x, out_slice);
            }
        }

        let mut selected = Vec::with_capacity(n_used);
        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];
        // Each token's picks in router-rank order, which is the order decode
        // accumulates its experts in.
        #[cfg(not(has_blas))]
        let mut token_picks: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);

        for j in 0..n {
            let router_probs = &mut all_router_logits[j * n_expert..(j + 1) * n_expert];
            router_probs
                .iter_mut()
                .for_each(|p| *p = 1.0 / (1.0 + (-*p).exp()));

            select_experts(router_probs, &moe.exp_probs_b, n_used, &mut selected);
            #[cfg(not(has_blas))]
            token_picks.push(selected.clone());

            for &(e, weight) in &selected {
                if weight > 0.0 {
                    expert_assignments[e].push((j, weight));
                }
            }
        }

        ffn_out[..hs * n].fill(0.0);

        #[cfg(has_blas)]
        {
            use crate::par::{IntoParallelRefIterator, ParallelIterator};

            struct ExpertTask {
                expert: usize,
                assigned: Vec<(usize, f32)>,
            }

            let active_tasks: Vec<ExpertTask> = (0..n_expert)
                .filter(|&e| !expert_assignments[e].is_empty())
                .map(|e| ExpertTask {
                    expert: e,
                    assigned: expert_assignments[e].clone(),
                })
                .collect();

            let expert_results: Vec<(usize, Vec<f32>)> = active_tasks
                .par_iter()
                .map(|task| {
                    let e = task.expert;
                    let assigned = &task.assigned;
                    let k_e = assigned.len();

                    let mut exp_in = vec![0.0f32; k_e * hs];
                    for (slot, &(token_j, _)) in assigned.iter().enumerate() {
                        exp_in[slot * hs..(slot + 1) * hs]
                            .copy_from_slice(&ffn_input[token_j * hs..(token_j + 1) * hs]);
                    }

                    let mut exp_gate_up = vec![0.0f32; k_e * 2 * ff];
                    let mut exp_gate = vec![0.0f32; k_e * ff];
                    let mut exp_down = vec![0.0f32; k_e * hs];

                    let dq_g = transformer::blas_dequantizer(moe.gate[e].dtype);
                    let dq_u = transformer::blas_dequantizer(moe.up[e].dtype);
                    let gate_up_ok = if let (Some(dq_g), Some(dq_u)) = (dq_g, dq_u) {
                        with_par_expert_scratch(&PAR_EXPERT_GATE_UP_ROWS, |rows| {
                            if rows.len() < 2 * ff * hs {
                                rows.resize(2 * ff * hs, 0.0);
                            }
                            let g_data = transformer::weight_data(&self.gguf, &moe.gate[e]);
                            let u_data = transformer::weight_data(&self.gguf, &moe.up[e]);
                            dq_g(g_data, ff, hs, &mut rows[..ff * hs]);
                            dq_u(u_data, ff, hs, &mut rows[ff * hs..2 * ff * hs]);

                            crate::backend::blas::sgemm_rowmajor_nt(
                                k_e,
                                2 * ff,
                                hs,
                                &exp_in,
                                &rows[..2 * ff * hs],
                                &mut exp_gate_up,
                            );
                        });
                        true
                    } else {
                        if dq_g.is_none() {
                            transformer::warn_unbatchable("moe_gate", moe.gate[e].dtype);
                        }
                        if dq_u.is_none() {
                            transformer::warn_unbatchable("moe_up", moe.up[e].dtype);
                        }
                        false
                    };

                    if gate_up_ok {
                        for t in 0..k_e {
                            let gate_slice = &mut exp_gate[t * ff..(t + 1) * ff];
                            let g_src = &exp_gate_up[t * 2 * ff..t * 2 * ff + ff];
                            let u_src = &exp_gate_up[t * 2 * ff + ff..(t + 1) * 2 * ff];
                            gate_slice.copy_from_slice(g_src);
                            cpu::silu_mul_inplace(gate_slice, u_src);
                        }
                    } else {
                        for (slot, &(token_j, _)) in assigned.iter().enumerate() {
                            let tok_in = &ffn_input[token_j * hs..(token_j + 1) * hs];
                            let gate_slice = &mut exp_gate[slot * ff..(slot + 1) * ff];
                            let up_slice = &mut exp_gate_up[slot * ff..(slot + 1) * ff];
                            self.gemv(&moe.gate[e], tok_in, gate_slice);
                            self.gemv(&moe.up[e], tok_in, up_slice);
                            cpu::silu_mul_inplace(gate_slice, up_slice);
                        }
                    }

                    // Gate first, as for the router above: calling the try-path
                    // with a dtype it cannot run debug-panics instead of
                    // declining to the GEMV fallback below.
                    let computed = transformer::batched_gemm_supports(moe.down[e].dtype, ff)
                        && with_par_expert_scratch(&PAR_EXPERT_DEQUANT_SCRATCH, |scratch| {
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                &moe.down[e],
                                &exp_gate,
                                &mut exp_down,
                                k_e,
                                hs,
                                ff,
                                scratch,
                            )
                        });
                    if !computed {
                        transformer::warn_unbatchable("moe_down", moe.down[e].dtype);
                        for slot in 0..k_e {
                            let g_in = &exp_gate[slot * ff..(slot + 1) * ff];
                            let d_out = &mut exp_down[slot * hs..(slot + 1) * hs];
                            self.gemv(&moe.down[e], g_in, d_out);
                        }
                    }

                    (e, exp_down)
                })
                .collect();

            for (e, exp_down) in expert_results {
                let assigned = &expert_assignments[e];
                for (slot, &(token_j, weight)) in assigned.iter().enumerate() {
                    let ed = &exp_down[slot * hs..(slot + 1) * hs];
                    let out_tok = &mut ffn_out[token_j * hs..(token_j + 1) * hs];

                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            use core::arch::aarch64::*;
                            let n_chunks = hs / 4;
                            let ed_ptr = ed.as_ptr();
                            let out_ptr = out_tok.as_mut_ptr();
                            let w_vec = vdupq_n_f32(weight);
                            for i in 0..n_chunks {
                                let ed_v = vld1q_f32(ed_ptr.add(i * 4));
                                let out_v = vld1q_f32(out_ptr.add(i * 4));
                                let res_v = vfmaq_f32(out_v, ed_v, w_vec);
                                vst1q_f32(out_ptr.add(i * 4), res_v);
                            }
                        }
                        for i in (hs / 4 * 4)..hs {
                            out_tok[i] += weight * ed[i];
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..hs {
                        out_tok[i] += weight * ed[i];
                    }
                }
            }
        }

        // Token by token, each token's experts in router-rank order, accumulating
        // `acc += weight * v` exactly as `forward_moe_ffn` does. This path reads each
        // expert's weights once per routed token either way, so walking tokens
        // costs nothing, and a different order (or a fused multiply-add) rounds the
        // combined output differently from decode. Expert routing is discrete, so
        // that last-bit difference grows into different experts a few layers on.
        #[cfg(not(has_blas))]
        {
            let mut exp_gate = vec![0.0f32; ff];
            let mut exp_up = vec![0.0f32; ff];
            let mut exp_down = vec![0.0f32; hs];

            #[cfg(target_arch = "aarch64")]
            let mut q8_scales_down = vec![0.0f32; ff / 32];
            #[cfg(target_arch = "aarch64")]
            let mut q8_quants_down = vec![0i8; ff];

            for (token_j, picks) in token_picks.iter().enumerate() {
                let tok_in = &ffn_input[token_j * hs..(token_j + 1) * hs];
                #[cfg(target_arch = "aarch64")]
                Self::quantize_to_scratch(tok_in, state);

                for &(e, weight) in picks {
                    #[cfg(target_arch = "aarch64")]
                    {
                        let gate_data = self.weight_data(&moe.gate[e]);
                        let up_data = self.weight_data(&moe.up[e]);
                        cpu::gemv_q4_0_fused2_with_q8(
                            gate_data,
                            up_data,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut exp_gate[..ff],
                            &mut exp_up[..ff],
                            ff,
                            hs,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.gemv(&moe.gate[e], tok_in, &mut exp_gate[..ff]);
                        self.gemv(&moe.up[e], tok_in, &mut exp_up[..ff]);
                    }

                    cpu::silu_mul_inplace(&mut exp_gate[..ff], &exp_up[..ff]);

                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                                &exp_gate[..ff],
                                &mut q8_scales_down,
                                &mut q8_quants_down,
                            );
                        }
                        let down_data = self.weight_data(&moe.down[e]);
                        cpu::gemv_q4_0_with_q8(
                            down_data,
                            &q8_scales_down,
                            &q8_quants_down,
                            &mut exp_down[..hs],
                            hs,
                            ff,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    self.gemv(&moe.down[e], &exp_gate[..ff], &mut exp_down[..hs]);

                    ffn_out[token_j * hs..(token_j + 1) * hs]
                        .iter_mut()
                        .zip(&exp_down[..hs])
                        .for_each(|(acc, &v)| *acc += weight * v);
                }
            }
        }
    }

    // ── Public accessors for GPU model construction ───────────────────────

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    pub fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }

    pub fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }

    pub fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }

    pub fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_norm_weights[layer].as_deref()
    }

    pub fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights[layer].as_deref()
    }

    pub fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.conv_weights_transposed[layer].as_deref()
    }

    /// Dequantize a token embedding row to f32.
    pub fn dequantize_embedding(&self, token_id: usize) -> Vec<f32> {
        self.dequantize_row(&self.embd_ref, token_id)
    }

    /// Conv in_proj GEMV for a layer.
    pub fn conv_in_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_in_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// Conv out_proj GEMV for a layer.
    pub fn conv_out_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_out_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// FFN gate GEMV for a layer. Errors on a MoE layer, whose gate weight is
    /// per-expert and only meaningful after routing.
    pub fn ffn_gate_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.gemv(&self.layer_refs[layer].ffn.dense()?.gate, x, y);
        Ok(())
    }

    /// FFN up GEMV for a layer. Errors on a MoE layer (see
    /// [`Self::ffn_gate_gemv`]).
    pub fn ffn_up_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.gemv(&self.layer_refs[layer].ffn.dense()?.up, x, y);
        Ok(())
    }

    /// FFN down GEMV for a layer. Errors on a MoE layer (see
    /// [`Self::ffn_gate_gemv`]).
    pub fn ffn_down_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.gemv(&self.layer_refs[layer].ffn.dense()?.down, x, y);
        Ok(())
    }

    /// Get raw weight bytes for a WeightRef (for GPU quantized upload).
    #[allow(dead_code)] // used by metal_lfm2/gpu_lfm2 behind feature gates
    pub(crate) fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        self.weight_data(wref)
    }

    // Full-matrix dequant lives in `transformer::dequantize_weight`; the LFM2
    // `GpuWeightSource` impl delegates to it. (The old inherent duplicate +
    // `dequantize_row_into_slice` helper were removed — single implementation.)

    /// Access the per-layer weight refs (for GPU model construction).
    #[allow(dead_code)]
    pub(crate) fn layer_refs(&self) -> &[LayerWeightRefs] {
        &self.layer_refs
    }

    /// Access the embedding weight ref.
    #[allow(dead_code)]
    pub(crate) fn embd_ref(&self) -> &WeightRef {
        &self.embd_ref
    }

    // ── Internal methods ────────────────────────────────────────────────

    /// Get the raw bytes for a pre-resolved weight.
    #[inline]
    fn weight_data(&self, wref: &WeightRef) -> &[u8] {
        transformer::weight_data(&self.gguf, wref)
    }

    /// GEMV dispatch without scratch buffers (shared `transformer::gemv`).
    fn gemv(&self, wref: &WeightRef, x: &[f32], y: &mut [f32]) {
        transformer::gemv(&self.gguf, wref, x, y);
    }

    /// GEMV with pre-quantized Q8_0 input (shared `transformer::gemv_preq`).
    #[cfg(target_arch = "aarch64")]
    fn gemv_preq(&self, wref: &WeightRef, x_f32: &[f32], q8s: &[f32], q8q: &[i8], y: &mut [f32]) {
        transformer::gemv_preq(&self.gguf, wref, x_f32, q8s, q8q, y);
    }

    // The batched-prefill GEMM helpers (`try_blas_prefill_gemm`, `gemm_preq`,
    // `quantize_columns`) are shared with the dense-transformer path and now
    // live as free functions in `transformer` — call sites use
    // `transformer::{try_blas_prefill_gemm, gemm_preq, quantize_columns}`.

    /// Quantize x to Q8_0 into scratch buffers (shared
    /// `transformer::quantize_to_scratch`).
    #[cfg(target_arch = "aarch64")]
    fn quantize_to_scratch(x: &[f32], state: &mut InferenceState) {
        transformer::quantize_to_scratch(x, state);
    }

    /// Dequantize a single row from a quantized matrix (for embedding lookup).
    fn dequantize_row(&self, wref: &WeightRef, row_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; wref.k];
        self.dequantize_row_into(wref, row_idx, &mut out);
        out
    }

    /// Dequantize a single row into `out` (shared
    /// `transformer::dequantize_row_into`).
    fn dequantize_row_into(&self, wref: &WeightRef, row_idx: usize, out: &mut [f32]) {
        transformer::dequantize_row_into(&self.gguf, wref, row_idx, out);
    }

    /// Whether a bidirectional pass over this model must take
    /// [`Self::hidden_states_float_bidirectional`]: some projection is a dtype the batched
    /// GEMM cannot run (F16, F32), and the per-token path that would take it instead is
    /// causal, so it would answer a different question.
    fn bidirectional_needs_float_path(&self) -> bool {
        self.needs_bidirectional_float_path
    }

    /// The load-time computation behind [`Self::bidirectional_needs_float_path`].
    ///
    /// A target with no batched GEMM at all (wasm32, 32-bit ARM and x86) has only that
    /// per-token path, so there every bidirectional model takes the float path.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
    fn bidirectional_float_path_needed(gguf: &GgufFile, config: &ModelConfig) -> bool {
        const PROJECTIONS: [&str; 9] = [
            "shortconv.in_proj",
            "shortconv.out_proj",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ];
        (0..config.n_layers).any(|layer| {
            PROJECTIONS.iter().any(|name| {
                gguf.tensors
                    .get(&format!("blk.{layer}.{name}.weight"))
                    .is_some_and(|t| {
                        !transformer::batched_gemm_supports(
                            t.dtype,
                            t.shape.first().copied().unwrap_or(0),
                        )
                    })
            })
        })
    }

    /// See the batched-target version: with no batched GEMM on this target, always.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", has_blas)))]
    fn bidirectional_float_path_needed(_gguf: &GgufFile, _config: &ModelConfig) -> bool {
        true
    }

    /// The bidirectional trunk in plain f32, for weights the batched GEMM cannot run.
    ///
    /// Takes the embedded hidden states, one row per token, and reuses that buffer as its
    /// working state: the caller has already dequantized the embeddings, so taking the
    /// tokens instead would allocate and embed a second `n * hidden` buffer. Each
    /// projection is read and widened to f32 as it is used, so only one layer's weights
    /// are resident at a time and a model is as large as its file, not four times it. It
    /// is the textbook form of the block, which makes it the reference the quantized path
    /// is measured against: pre-norm, a centred 3-tap gated convolution or full
    /// (non-causal) grouped-query attention, and a SwiGLU feed-forward.
    ///
    /// Returns the post-final-norm hidden state of every token, `[tokens, hidden]`.
    fn hidden_states_float_bidirectional(&self, h: Vec<f32>) -> Vec<f32> {
        self.float_trunk(h, 0)
    }

    /// The f32 bidirectional trunk over embeddings `h` (`[n, hidden]`) whose first
    /// `prefix` rows are media embeddings.
    ///
    /// A media prefix is a function of the media alone: its rows attend only to each other,
    /// and the convolution at its last row does not read the first text row. Text rows read
    /// everything. With `prefix == 0` this is the plain bidirectional trunk.
    fn float_trunk(&self, mut h: Vec<f32>, prefix: usize) -> Vec<f32> {
        use crate::model::pii::matmul_nt_f32;
        use crate::par::*;
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        debug_assert!(
            !h.is_empty() && h.len().is_multiple_of(hs),
            "the float trunk takes embedded rows, got {} f32 for hidden_size {hs}",
            h.len()
        );
        let n = h.len() / hs;
        let weight = |layer: usize, name: &str| -> Vec<f32> {
            self.gguf
                .get_tensor(&format!("blk.{layer}.{name}.weight"))
                .unwrap_or_else(|e| panic!("blk.{layer}.{name}.weight: {e}"))
                .to_f32_vec()
        };
        let project = |x: &[f32], w: &[f32], rows: usize, k: usize| -> Vec<f32> {
            let mut out = vec![0f32; n * rows];
            matmul_nt_f32(n, rows, k, x, w, None, &mut out);
            out
        };

        // One norm scratch for all layers: cloning `h` per norm would allocate
        // `2 * n_layers` full-size buffers per pass.
        let mut normed = vec![0f32; n * hs];
        for layer in 0..cfg.n_layers {
            normed.copy_from_slice(&h);
            for row in normed.chunks_exact_mut(hs) {
                cpu::rmsnorm(row, &self.attn_norm_weights[layer], cfg.rms_norm_eps);
            }
            let operator = if cfg.block_types[layer] == BlockType::GatedConv {
                // B, C and x side by side; y = C * conv3(B * x), centred, zero padded
                let proj = project(&normed, &weight(layer, "shortconv.in_proj"), 3 * hs, hs);
                let taps = self.conv_weights[layer].as_ref().expect("conv taps");
                let (w0, rest) = taps.split_at(hs);
                let (w1, w2) = rest.split_at(hs);
                let bx = |t: usize, i: usize| proj[t * 3 * hs + i] * proj[t * 3 * hs + 2 * hs + i];
                let mut gated = vec![0f32; n * hs];
                for t in 0..n {
                    for i in 0..hs {
                        let prev = if t > 0 { bx(t - 1, i) } else { 0.0 };
                        let reads_right = t + 1 < n && t + 1 != prefix;
                        let next = if reads_right { bx(t + 1, i) } else { 0.0 };
                        gated[t * hs + i] = proj[t * 3 * hs + hs + i]
                            * (prev * w0[i] + bx(t, i) * w1[i] + next * w2[i]);
                    }
                }
                project(&gated, &weight(layer, "shortconv.out_proj"), hs, hs)
            } else {
                let n_heads = cfg.n_heads;
                let n_kv = cfg.kv_heads_per_layer[layer];
                let head_dim = hs / n_heads;
                let kv_dim = n_kv * head_dim;
                let mut q = project(&normed, &weight(layer, "attn_q"), hs, hs);
                let mut k = project(&normed, &weight(layer, "attn_k"), kv_dim, hs);
                let v = project(&normed, &weight(layer, "attn_v"), kv_dim, hs);
                let q_norm = self.attn_q_norm_weights[layer].as_ref().expect("q norm");
                let k_norm = self.attn_k_norm_weights[layer].as_ref().expect("k norm");
                for t in 0..n {
                    let (qr, kr) = (
                        &mut q[t * hs..(t + 1) * hs],
                        &mut k[t * kv_dim..(t + 1) * kv_dim],
                    );
                    for head in qr.chunks_exact_mut(head_dim) {
                        cpu::rmsnorm(head, q_norm, cfg.rms_norm_eps);
                    }
                    for head in kr.chunks_exact_mut(head_dim) {
                        cpu::rmsnorm(head, k_norm, cfg.rms_norm_eps);
                    }
                    cpu::rope(qr, kr, t, n_heads, n_kv, head_dim, cfg.rope_theta);
                }
                let scale = (head_dim as f32).powf(-0.5);
                let group = n_heads / n_kv;
                let mut ctx = vec![0f32; n * hs];
                ctx.par_chunks_mut(hs).enumerate().for_each(|(t, out)| {
                    // a prefix row attends to the prefix only; a text row to everything
                    let keys = if t < prefix { prefix } else { n };
                    let mut scores = vec![0f32; keys];
                    for head in 0..n_heads {
                        let kv = head / group;
                        let qh = &q[t * hs + head * head_dim..][..head_dim];
                        for (j, s) in scores.iter_mut().enumerate() {
                            *s = cpu::dot_f32(qh, &k[j * kv_dim + kv * head_dim..][..head_dim])
                                * scale;
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for s in scores.iter_mut() {
                            *s = (*s - max).exp();
                            sum += *s;
                        }
                        let dst = &mut out[head * head_dim..(head + 1) * head_dim];
                        for (j, s) in scores.iter().enumerate() {
                            let w = s / sum;
                            let vh = &v[j * kv_dim + kv * head_dim..][..head_dim];
                            for (o, x) in dst.iter_mut().zip(vh) {
                                *o += w * x;
                            }
                        }
                    }
                });
                project(&ctx, &weight(layer, "attn_output"), hs, hs)
            };
            for (a, b) in h.iter_mut().zip(&operator) {
                *a += b;
            }

            // SwiGLU feed-forward
            normed.copy_from_slice(&h);
            for row in normed.chunks_exact_mut(hs) {
                cpu::rmsnorm(row, &self.ffn_norm_weights[layer], cfg.rms_norm_eps);
            }
            let gate_w = weight(layer, "ffn_gate");
            let ff = gate_w.len() / hs;
            let mut gate = project(&normed, &gate_w, ff, hs);
            let up = project(&normed, &weight(layer, "ffn_up"), ff, hs);
            cpu::silu_mul_inplace(&mut gate, &up);
            let down = project(&gate, &weight(layer, "ffn_down"), hs, ff);
            for (a, b) in h.iter_mut().zip(&down) {
                *a += b;
            }
        }
        for row in h.chunks_exact_mut(hs) {
            cpu::rmsnorm(row, &self.output_norm_weight, cfg.rms_norm_eps);
        }
        h
    }

    /// Process a single conv (recurrent) block using pre-allocated scratch buffers.
    fn forward_conv_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let hidden_size = self.config.hidden_size;
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
        let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
        let conv_weight = self.conv_weights_transposed[layer].as_ref().unwrap();
        #[cfg(target_arch = "aarch64")]
        let conv_tap_first = self.conv_weights[layer].as_ref().unwrap();

        // Cloned once (cheap Arc bump) so the adapter can be read while the base
        // scratch buffers stay mutably borrowed — same pattern as
        // `forward_attn_block`.
        let lora = state.lora.clone();

        // in_proj: hidden → 3*hidden (uses pre-quantized Q8_0 data when available)
        let proj = &mut state.scratch.conv_proj[..3 * hidden_size];
        #[cfg(target_arch = "aarch64")]
        self.gemv_preq(
            in_proj,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            proj,
        );
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(in_proj, hidden, proj);

        // LoRA on the conv in_proj — `proj += scale·B·(A·hidden)`, applied to the
        // full 3·hidden output before it is split into the B/C/x gates. Matches
        // llama.cpp applying the adapter to `shortconv.in_proj` on conv layers.
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvInProj)
        {
            crate::lora::apply_decode(t, hidden, proj, &mut state.scratch.lora_tmp);
        }

        // Split: b, c, x
        let (b, rest) = proj.split_at(hidden_size);
        let (c, x) = rest.split_at(hidden_size);

        // bx = b ⊙ x (element-wise gate before conv)
        let conv_scratch = &mut state.scratch.conv_scratch[..hidden_size];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let mut i = 0usize;
            while i + 3 < hidden_size {
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vx = vld1q_f32(x.as_ptr().add(i));
                vst1q_f32(conv_scratch.as_mut_ptr().add(i), vmulq_f32(vb, vx));
                i += 4;
            }
            while i < hidden_size {
                conv_scratch[i] = b[i] * x[i];
                i += 1;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (out, (bi, xi)) in conv_scratch.iter_mut().zip(b.iter().zip(x.iter())) {
            *out = bi * xi;
        }
        // conv_scratch now holds bx

        // Depthwise conv1d with valid convolution using rolling buffer.
        //
        // Fused multiply-add (`vfmaq_f32`), like the batched-prefill conv kernel.
        // `vmlaq_f32` lowers to a separate multiply and add, so decode and prefill
        // rounded the conv output differently. The next Q8_0 quantization hides a
        // 1-ulp difference, but anything reading it as f32 (the LoRA out_proj hook)
        // sees it.
        let LayerState::Conv { buffer, history } = &mut state.layers[layer] else {
            panic!("expected Conv state for layer {layer}");
        };

        {
            let out_buf = &mut state.scratch.out[..hidden_size];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let mut ch = 0usize;
                while ch + 3 < hidden_size {
                    let mut vsum = vdupq_n_f32(0.0);
                    for k in 0..d_conv {
                        let vw = vld1q_f32(conv_tap_first.as_ptr().add(k * hidden_size + ch));
                        let vbuf = vld1q_f32(buffer.as_ptr().add(k * hidden_size + ch));
                        vsum = vfmaq_f32(vsum, vbuf, vw);
                    }
                    let vw_last = vld1q_f32(conv_tap_first.as_ptr().add(d_conv * hidden_size + ch));
                    let vcur = vld1q_f32(conv_scratch.as_ptr().add(ch));
                    vsum = vfmaq_f32(vsum, vcur, vw_last);
                    vst1q_f32(out_buf.as_mut_ptr().add(ch), vsum);
                    ch += 4;
                }
                while ch < hidden_size {
                    let mut sum = 0.0f32;
                    for k in 0..d_conv {
                        sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
                    }
                    sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                    out_buf[ch] = sum;
                    ch += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for ch in 0..hidden_size {
                let mut sum = 0.0f32;
                for k in 0..d_conv {
                    sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
                }
                sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                out_buf[ch] = sum;
            }

            // Update rolling buffer: shift left by one slot, append bx
            if d_conv > 0 {
                if d_conv > 1 {
                    buffer.copy_within(hidden_size.., 0);
                }
                let last_slot = (d_conv - 1) * hidden_size;
                buffer[last_slot..last_slot + hidden_size].copy_from_slice(conv_scratch);
                history.push(pos + 1, buffer);
            }

            // o = c ⊙ conv_out (second gate), reuse conv_scratch
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let mut i = 0usize;
                while i + 3 < hidden_size {
                    let vc = vld1q_f32(c.as_ptr().add(i));
                    let vout = vld1q_f32(out_buf.as_ptr().add(i));
                    vst1q_f32(conv_scratch.as_mut_ptr().add(i), vmulq_f32(vc, vout));
                    i += 4;
                }
                while i < hidden_size {
                    conv_scratch[i] = c[i] * out_buf[i];
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (o, (ci, co)) in conv_scratch.iter_mut().zip(c.iter().zip(out_buf.iter())) {
                *o = ci * co;
            }
        }

        // out_proj: hidden → hidden, write result into out_buf
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch_bufs(
                conv_scratch,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
            self.gemv_preq(
                out_proj,
                conv_scratch,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut state.scratch.out[..hidden_size],
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(
            out_proj,
            conv_scratch,
            &mut state.scratch.out[..hidden_size],
        );
        // LoRA on the conv out_proj (input = the gated conv output, before residual).
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
        {
            let out = &mut state.scratch.out[..hidden_size];
            crate::lora::apply_decode(t, conv_scratch, out, &mut state.scratch.lora_tmp);
        }
        // Result is now in state.scratch.out[..hidden_size]
    }

    /// Process a single attention block using pre-allocated scratch buffers.
    fn forward_attn_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.n_heads * head_dim;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;

        // Cloned once (cheap Arc bump) so the base scratch buffers can stay
        // mutably borrowed while the adapter (a disjoint field) is read.
        let lora = state.lora.clone();

        // Q, K, V projections using pre-quantized hidden state
        let q = &mut state.scratch.q[..q_dim];
        let k = &mut state.scratch.k[..kv_dim];
        let v = &mut state.scratch.v[..kv_dim];

        // hidden was pre-quantized at layer level — use integer path
        #[cfg(target_arch = "aarch64")]
        {
            let q_ref = refs.attn_q.as_ref().unwrap();
            let k_ref = refs.attn_k.as_ref().unwrap();
            let v_ref = refs.attn_v.as_ref().unwrap();
            let valid_dims = q_ref.k == cfg.hidden_size
                && k_ref.k == cfg.hidden_size
                && v_ref.k == cfg.hidden_size
                && q_ref.m == q_dim
                && k_ref.m == kv_dim
                && v_ref.m == kv_dim;
            if valid_dims
                && q_ref.dtype == DType::Q4_0
                && k_ref.dtype == DType::Q4_0
                && v_ref.dtype == DType::Q4_0
            {
                if let (Some(q4), Some(k4), Some(v4)) =
                    (q_ref.dec4_bytes(), k_ref.dec4_bytes(), v_ref.dec4_bytes())
                {
                    cpu::gemv_q4_0_dec4_concat3_with_q8(
                        q4,
                        k4,
                        v4,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        q,
                        k,
                        v,
                        q_dim,
                        kv_dim,
                        kv_dim,
                        cfg.hidden_size,
                    );
                } else {
                    let q_data = self.weight_data(q_ref);
                    let k_data = self.weight_data(k_ref);
                    let v_data = self.weight_data(v_ref);
                    cpu::gemv_q4_0_concat3_with_q8(
                        q_data,
                        k_data,
                        v_data,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        q,
                        k,
                        v,
                        q_dim,
                        kv_dim,
                        kv_dim,
                        cfg.hidden_size,
                    );
                }
            } else if valid_dims
                && q_ref.dtype == DType::Q4KM
                && k_ref.dtype == DType::Q4KM
                && v_ref.dtype == DType::Q4KM
            {
                let q_data = self.weight_data(q_ref);
                let k_data = self.weight_data(k_ref);
                let v_data = self.weight_data(v_ref);
                cpu::gemv_q4k_concat3_with_q8(
                    q_data,
                    k_data,
                    v_data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    q,
                    k,
                    v,
                    q_dim,
                    kv_dim,
                    kv_dim,
                    cfg.hidden_size,
                );
            } else if valid_dims
                && q_ref.dtype == DType::Q5KM
                && k_ref.dtype == DType::Q5KM
                && v_ref.dtype == DType::Q5KM
            {
                let q_data = self.weight_data(q_ref);
                let k_data = self.weight_data(k_ref);
                let v_data = self.weight_data(v_ref);
                cpu::gemv_q5k_concat3_with_q8(
                    q_data,
                    k_data,
                    v_data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    q,
                    k,
                    v,
                    q_dim,
                    kv_dim,
                    kv_dim,
                    cfg.hidden_size,
                );
            } else {
                #[cfg(not(has_blas))]
                let smmla_done = if valid_dims
                    && q_ref.dec4_bytes().is_none()
                    && let (Some(q_rp), Some(k_rp), Some(v_rp)) =
                        (&q_ref.repacked, &k_ref.repacked, &v_ref.repacked)
                    && let (
                        transformer::Repacked::Q40Smmla {
                            packed: qp,
                            scales: qs,
                        },
                        transformer::Repacked::Q40Smmla {
                            packed: kp,
                            scales: ks,
                        },
                        transformer::Repacked::Q40Smmla {
                            packed: vp,
                            scales: vs,
                        },
                    ) = (&q_rp.kind, &k_rp.kind, &v_rp.kind)
                {
                    cpu::gemv_q4_0_smmla_concat3_with_q8(
                        qp,
                        qs,
                        kp,
                        ks,
                        vp,
                        vs,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        q,
                        k,
                        v,
                        q_dim,
                        kv_dim,
                        kv_dim,
                        cfg.hidden_size,
                    )
                } else {
                    false
                };
                #[cfg(has_blas)]
                let smmla_done = false;
                if !smmla_done {
                    self.gemv_preq(
                        q_ref,
                        hidden,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        q,
                    );
                    self.gemv_preq(
                        k_ref,
                        hidden,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        k,
                    );
                    self.gemv_preq(
                        v_ref,
                        hidden,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        v,
                    );
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.gemv(refs.attn_q.as_ref().unwrap(), hidden, q);
            self.gemv(refs.attn_k.as_ref().unwrap(), hidden, k);
            self.gemv(refs.attn_v.as_ref().unwrap(), hidden, v);
        }

        // LoRA on Q/K/V — input is the normed hidden, applied before QK-norm/RoPE.
        if let Some(lora) = &lora {
            crate::lora::apply_attn_qkv(lora, layer, hidden, q, k, v, &mut state.scratch.lora_tmp);
        }

        // `lfm3`: per-head sigmoid gate, from the same normed input as Q/K/V, that
        // scales the attention output before the output projection.
        let lfm3_attn = self
            .lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].attn.as_ref());
        let mut gate_buf = [0.0f32; MAX_LFM3_HEADS];
        let gate = lfm3_attn.map(|a| {
            let gate = &mut gate_buf[..cfg.n_heads];
            #[cfg(target_arch = "aarch64")]
            self.gemv_preq(
                &a.gate,
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                gate,
            );
            #[cfg(not(target_arch = "aarch64"))]
            self.gemv(&a.gate, hidden, gate);
            cpu::sigmoid_inplace(gate);
            Self::dump_node("model.layers.{}.self_attn.gate", layer, gate);
            &*gate
        });

        // Per-head QK norm (RMSnorm each head slice with shared weights)
        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
        for h in 0..cfg.n_heads {
            cpu::rmsnorm(
                &mut q[h * head_dim..(h + 1) * head_dim],
                q_norm,
                cfg.rms_norm_eps,
            );
        }
        for h in 0..n_kv_heads {
            cpu::rmsnorm(
                &mut k[h * head_dim..(h + 1) * head_dim],
                k_norm,
                cfg.rms_norm_eps,
            );
        }

        // RoPE. `lfm3` carries its own base and partial-rotary mask per layer.
        match lfm3_attn {
            Some(a) => cpu::rope_neox_freq_factors(
                q,
                k,
                pos,
                (cfg.n_heads, n_kv_heads),
                head_dim,
                a.rope_theta,
                a.rope_freq_factors.as_deref(),
            ),
            None => cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta),
        }

        // Grab per-model TurboQuant state once (None when disabled)
        // TurboQuant rotation state lives on InferenceState now (since PR #12
        // refactor). A single KvCompression::TurboQuant { seed, ... } config
        // on the state is enough — no separate model-side enable needed.
        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
        let tq_config = state.tq_config.as_ref();

        // f16 KV: append converts to half + the naive read path widens on read.
        // Mutually exclusive with TurboQuant (a distinct KvCompression variant),
        // so `use_f16` is never true alongside the TQ branches below.
        let use_f16 = state.kv_f16;

        // Append K, V to cache. Keys and values are compressed independently —
        // whichever side has a CompressedKvCache present gets the TurboQuant
        // path; the other side falls through to the f32 (or f16) cache.
        if let LayerState::Attention {
            key_cache,
            value_cache,
            key_cache_f16,
            value_cache_f16,
            compressed_keys,
            compressed_values,
        } = &mut state.layers[layer]
        {
            let tq_ok =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_encode_scratch.is_some();
            match (tq_ok, compressed_keys.as_mut()) {
                (true, Some(k_cache_tq)) => {
                    turboquant::compress_and_append_keys(
                        &state.scratch.k[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        k_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ if use_f16 => {
                    key_cache_f16.extend(
                        state.scratch.k[..kv_dim]
                            .iter()
                            .map(|&x| crate::quant::f32_to_f16(x)),
                    );
                }
                _ => {
                    key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                }
            }
            match (tq_ok, compressed_values.as_mut()) {
                (true, Some(v_cache_tq)) => {
                    turboquant::compress_and_append_values(
                        &state.scratch.v[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        v_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ if use_f16 => {
                    value_cache_f16.extend(
                        state.scratch.v[..kv_dim]
                            .iter()
                            .map(|&x| crate::quant::f32_to_f16(x)),
                    );
                }
                _ => {
                    value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                }
            }
        }

        // GQA: grouped query attention
        let group_size = cfg.n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        {
            // Access layers and scratch as disjoint fields to avoid whole-state borrow
            let (ck, cv, k_cache, v_cache, k_cache_f16, v_cache_f16) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    compressed_keys,
                    compressed_values,
                } => (
                    compressed_keys.as_ref(),
                    compressed_values.as_ref(),
                    key_cache.as_slice(),
                    value_cache.as_slice(),
                    key_cache_f16.as_slice(),
                    value_cache_f16.as_slice(),
                ),
                _ => panic!("expected Attention state for layer {layer}"),
            };

            // Keys and values are compressed independently — determine which
            // side of the attention read path uses TurboQuant.
            let tq_prereq =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_query_scratch.is_some();
            let use_tq_keys = tq_prereq && ck.is_some();
            let use_tq_values = tq_prereq && cv.is_some();

            // seq_len comes from whichever cache is populated. All four
            // combinations agree on seq_len per layer because encode appends
            // to one cache per side per token.
            let seq_len = if use_tq_keys {
                ck.unwrap().seq_len()
            } else if use_tq_values {
                cv.unwrap().seq_len()
            } else if use_f16 {
                k_cache_f16.len() / kv_dim
            } else {
                k_cache.len() / kv_dim
            };
            let attn_out = &mut state.scratch.attn_out[..q_dim];
            let q = &state.scratch.q[..q_dim];

            if use_tq_keys || use_tq_values {
                // GQA batched path — one score buffer per group, shared
                // between the key score and value weighted-sum stages.
                let scores = &mut state.scratch.scores;
                let rotation = tq_rotation.unwrap();
                let cfg_tq = tq_config.unwrap();
                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                if use_tq_keys {
                    turboquant::rotate_queries(q, cfg.n_heads, head_dim, rotation, qr_scratch);
                }
                scores.resize(seq_len * group_size, 0.0);
                for kv_h in 0..n_kv_heads {
                    let group_start = kv_h * group_size;
                    let kv_h_offset = kv_h * head_dim;

                    // Scores: TurboQuant or f32.
                    if use_tq_keys {
                        turboquant::attn_scores_turboquant_gqa(
                            ck.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            head_dim,
                            scale,
                            seq_len,
                            cfg_tq,
                            qr_scratch,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                            let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_scores(
                                q_head,
                                k_cache,
                                head_scores,
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                scale,
                                seq_len,
                            );
                        }
                    }

                    // Softmax each head's scores in place.
                    for g in 0..group_size {
                        let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                        cpu::softmax_inplace(head_scores);
                    }

                    // Values: TurboQuant or f32.
                    if use_tq_values {
                        turboquant::attn_values_turboquant_gqa(
                            cv.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            attn_out,
                            head_dim,
                            seq_len,
                            rotation,
                            cfg_tq,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let head_scores = &scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_values(
                                head_scores,
                                v_cache,
                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                seq_len,
                            );
                        }
                    }
                }
            } else {
                // Non-TQ path: f16 (widen-on-read) or f32. Only the active
                // representation is non-empty. Shared with the dense
                // transformers, which run the identical head loop.
                let kv = if use_f16 {
                    transformer::KvView::F16 {
                        k: k_cache_f16,
                        v: v_cache_f16,
                    }
                } else {
                    transformer::KvView::F32 {
                        k: k_cache,
                        v: v_cache,
                    }
                };
                transformer::decode_attention(
                    q,
                    &kv,
                    &transformer::DecodeAttnDims {
                        n_heads: cfg.n_heads,
                        n_kv_heads,
                        head_dim,
                        scale,
                        seq_len,
                        attn_logit_softcapping: None,
                        sliding_window: lfm3_attn.and_then(|a| a.window),
                    },
                    attn_out,
                    &mut state.scratch.scores,
                );
            }
        }

        // `lfm3`: scale each head's output by its gate.
        if let Some(gate) = gate {
            state.scratch.attn_out[..q_dim]
                .chunks_exact_mut(head_dim)
                .zip(gate)
                .for_each(|(head, &g)| head.iter_mut().for_each(|x| *x *= g));
        }

        // Output projection
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch_bufs(
                &state.scratch.attn_out[..q_dim],
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
            self.gemv_preq(
                refs.attn_output.as_ref().unwrap(),
                &state.scratch.attn_out[..q_dim],
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut state.scratch.out[..cfg.hidden_size],
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let out = &mut state.scratch.out[..cfg.hidden_size];
            self.gemv(
                refs.attn_output.as_ref().unwrap(),
                &state.scratch.attn_out[..q_dim],
                out,
            );
        }
        // LoRA on the output projection (input = the attention output).
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
        {
            let out = &mut state.scratch.out[..cfg.hidden_size];
            crate::lora::apply_decode(
                t,
                &state.scratch.attn_out[..q_dim],
                out,
                &mut state.scratch.lora_tmp,
            );
        }
    }

    /// Fallback multi-token bidirectional attention block for non-causal layers when
    /// the batched GEMM path is unavailable (for example, F32/F16 weights in tests).
    /// Appends K and V for all `n` tokens before computing attention so every token
    /// attends across the full `[0..start_pos + n)` span.
    fn forward_attn_block_bidir_batch(
        &self,
        layer: usize,
        normed: &[f32],
        start_pos: usize,
        n: usize,
        block_out: &mut [f32],
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.n_heads * head_dim;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;
        let group_size = cfg.n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let lora = state.lora.clone();
        let lfm3_attn = self
            .lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].attn.as_ref());
        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
        let q_ref = refs.attn_q.as_ref().unwrap();
        let k_ref = refs.attn_k.as_ref().unwrap();
        let v_ref = refs.attn_v.as_ref().unwrap();
        let out_ref = refs.attn_output.as_ref().unwrap();

        let mut q_all = vec![0.0f32; n * q_dim];
        let mut gate_all = if lfm3_attn.is_some() {
            vec![0.0f32; n * cfg.n_heads]
        } else {
            Vec::new()
        };

        for j in 0..n {
            let hidden = &normed[j * hs..(j + 1) * hs];
            let q = &mut q_all[j * q_dim..(j + 1) * q_dim];
            let k = &mut state.scratch.k[..kv_dim];
            let v = &mut state.scratch.v[..kv_dim];
            self.gemv(q_ref, hidden, q);
            self.gemv(k_ref, hidden, k);
            self.gemv(v_ref, hidden, v);
            if let Some(lora) = &lora {
                crate::lora::apply_attn_qkv(
                    lora,
                    layer,
                    hidden,
                    q,
                    k,
                    v,
                    &mut state.scratch.lora_tmp,
                );
            }
            if let Some(a) = lfm3_attn {
                let gate = &mut gate_all[j * cfg.n_heads..(j + 1) * cfg.n_heads];
                self.gemv(&a.gate, hidden, gate);
                cpu::sigmoid_inplace(gate);
            }
            for h in 0..cfg.n_heads {
                cpu::rmsnorm(
                    &mut q[h * head_dim..(h + 1) * head_dim],
                    q_norm,
                    cfg.rms_norm_eps,
                );
            }
            for h in 0..n_kv_heads {
                cpu::rmsnorm(
                    &mut k[h * head_dim..(h + 1) * head_dim],
                    k_norm,
                    cfg.rms_norm_eps,
                );
            }
            match lfm3_attn {
                Some(a) => cpu::rope_neox_freq_factors(
                    q,
                    k,
                    start_pos + j,
                    (cfg.n_heads, n_kv_heads),
                    head_dim,
                    a.rope_theta,
                    a.rope_freq_factors.as_deref(),
                ),
                None => cpu::rope(
                    q,
                    k,
                    start_pos + j,
                    cfg.n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rope_theta,
                ),
            }
            if let LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[layer]
            {
                if state.kv_f16 {
                    key_cache_f16.extend(k.iter().map(|&x| crate::quant::f32_to_f16(x)));
                    value_cache_f16.extend(v.iter().map(|&x| crate::quant::f32_to_f16(x)));
                } else {
                    key_cache.extend_from_slice(k);
                    value_cache.extend_from_slice(v);
                }
            }
        }

        let (k_cache_vec, v_cache_vec);
        let (k_cache, v_cache) = match &state.layers[layer] {
            LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } => {
                if state.kv_f16 {
                    k_cache_vec = key_cache_f16
                        .iter()
                        .map(|&b| crate::quant::f16_to_f32(b))
                        .collect::<Vec<_>>();
                    v_cache_vec = value_cache_f16
                        .iter()
                        .map(|&b| crate::quant::f16_to_f32(b))
                        .collect::<Vec<_>>();
                    (k_cache_vec.as_slice(), v_cache_vec.as_slice())
                } else {
                    (key_cache.as_slice(), value_cache.as_slice())
                }
            }
            _ => unreachable!(),
        };
        let seq_len = (start_pos + n).min(k_cache.len() / kv_dim);
        for j in 0..n {
            let q = &q_all[j * q_dim..(j + 1) * q_dim];
            let attn_out = &mut state.scratch.attn_out[..q_dim];
            let scores = &mut state.scratch.scores;
            scores.resize(seq_len, 0.0);
            for h in 0..cfg.n_heads {
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
            if lfm3_attn.is_some() {
                let gate = &gate_all[j * cfg.n_heads..(j + 1) * cfg.n_heads];
                attn_out
                    .chunks_exact_mut(head_dim)
                    .zip(gate)
                    .for_each(|(head, &g)| head.iter_mut().for_each(|x| *x *= g));
            }
            let out_row = &mut block_out[j * hs..(j + 1) * hs];
            self.gemv(out_ref, attn_out, out_row);
            if let Some(lora) = &lora
                && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
            {
                crate::lora::apply_decode(t, attn_out, out_row, &mut state.scratch.lora_tmp);
            }
        }
    }
    /// Run all layers + output norm on a hidden state vector. Shared by
    /// forward(), forward_embedding(), and forward_hidden_from_embedding().
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        // Decode has started: a large prefill working set is dead weight now.
        state.prefill_scratch.release_if_large();
        let lfm3 = self.lfm3.as_ref();
        // Reuse pre-allocated scratch from InferenceState instead of allocating
        // fresh Vecs on every call. Take them out of `state.scratch` to avoid
        // borrow-checker conflicts with the mutable `state` passed to
        // forward_attn_block / forward_conv_block below; put them back at the end.
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        normed.resize(hs, 0.0);
        ffn_input.resize(hs, 0.0);

        let nb = hs / 32;
        state.scratch.q8_scales.resize(nb, 0.0);
        state.scratch.q8_quants.resize(hs, 0);

        let profile = Self::profile_decode_enabled();
        let mut t_norm = std::time::Duration::ZERO;
        let mut t_conv = std::time::Duration::ZERO;
        let mut t_attn = std::time::Duration::ZERO;
        let mut t_ffn = std::time::Duration::ZERO;

        for i in 0..cfg.n_layers {
            let t0 = if profile {
                Some(crate::time::Instant::now())
            } else {
                None
            };
            state.scratch.q8_scales.resize(nb, 0.0);
            state.scratch.q8_quants.resize(hs, 0);
            cpu::rmsnorm_and_quantize_q8_0(
                hidden,
                &self.attn_norm_weights[i],
                cfg.rms_norm_eps,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
                Some(&mut normed),
            );
            if let Some(t) = t0 {
                t_norm += t.elapsed();
            }
            if lfm3.is_some() {
                Self::dump_node("model.layers.{}.operator_norm", i, &normed);
            }

            let t1 = if profile {
                Some(crate::time::Instant::now())
            } else {
                None
            };
            if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &normed, pos, state);
                if let Some(t) = t1 {
                    t_conv += t.elapsed();
                }
            } else {
                self.forward_attn_block(i, &normed, pos, state);
                if let Some(t) = t1 {
                    t_attn += t.elapsed();
                }
            }

            if let Some(norm_w) = lfm3.and_then(|l| l.layers[i].post_operator_norm.as_ref()) {
                cpu::rmsnorm(&mut state.scratch.out[..hs], norm_w, cfg.rms_norm_eps);
                Self::dump_node(
                    "model.layers.{}.post_operator_norm",
                    i,
                    &state.scratch.out[..hs],
                );
            }
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            let t2 = if profile {
                Some(crate::time::Instant::now())
            } else {
                None
            };
            state.scratch.q8_scales.resize(nb, 0.0);
            state.scratch.q8_quants.resize(hs, 0);
            cpu::rmsnorm_and_quantize_q8_0(
                hidden,
                &self.ffn_norm_weights[i],
                cfg.rms_norm_eps,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
                Some(&mut ffn_input),
            );
            if let Some(t) = t2 {
                t_norm += t.elapsed();
            }

            let t3 = if profile {
                Some(crate::time::Instant::now())
            } else {
                None
            };
            match &self.layer_refs[i].ffn {
                FfnRefs::Dense(dense) => {
                    let ffn_weights = FfnWeights {
                        ffn_gate: &dense.gate,
                        ffn_up: &dense.up,
                        ffn_down: &dense.down,
                    };
                    transformer::forward_ffn_block(
                        &self.gguf,
                        i,
                        &ffn_weights,
                        &transformer::FfnExtras::default(),
                        hs,
                        cfg.intermediate_size,
                        &ffn_input,
                        transformer::FfnActivation::Swiglu,
                        state,
                    );
                }
                FfnRefs::Moe(moe) => self.forward_moe_ffn(i, moe, hs, &ffn_input, state),
            }
            if let Some(t) = t3 {
                t_ffn += t.elapsed();
            }

            if let Some(norm_w) = lfm3.and_then(|l| l.layers[i].post_ffn_norm.as_ref()) {
                cpu::rmsnorm(&mut state.scratch.out[..hs], norm_w, cfg.rms_norm_eps);
                Self::dump_node("model.layers.{}.post_ffn_norm", i, &state.scratch.out[..hs]);
            }
            cpu::add_inplace(hidden, &state.scratch.out[..cfg.hidden_size]);
            if lfm3.is_some() {
                Self::dump_node("l_out", i, hidden);
            }
        }

        if profile {
            eprintln!(
                "[PROFILE DECODE] norm: {:.2}ms | conv: {:.2}ms | attn: {:.2}ms | ffn: {:.2}ms",
                t_norm.as_secs_f64() * 1000.0,
                t_conv.as_secs_f64() * 1000.0,
                t_attn.as_secs_f64() * 1000.0,
                t_ffn.as_secs_f64() * 1000.0,
            );
        }

        cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
        if lfm3.is_some() && transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_norm", hidden);
        }
        #[cfg(target_arch = "aarch64")]
        Self::quantize_to_scratch(hidden, state);
        state.seq_len += 1;

        // Return the scratch buffers for the next call.
        state.scratch.normed = normed;
        state.scratch.ffn_input = ffn_input;
    }

    /// Layer loop + output norm + tied logit projection for batched
    /// prefill. Takes a column-major hidden buffer (`hs × n`, channel
    /// `i` of token `j` at index `i * n + j`) already populated by the
    /// caller, runs the per-layer attn / conv + FFN passes, advances
    /// `state.seq_len` to `start_pos + n`, and projects the last
    /// frame's hidden state to logits over the vocabulary.
    ///
    /// Shared between [`Self::forward_prefill`] (token-id input,
    /// embedding-table lookup) and [`Self::forward_prefill_from_embeddings`]
    /// (raw embedding input, copied + transposed into the column-major
    /// `hidden` layout). The two entry points differ only in how they
    /// fill `hidden`; everything from the first RMSnorm onward is
    /// identical and lives here.
    fn debug_hidden_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("CERA_DEBUG_HIDDEN").as_deref() == Ok("1"))
    }

    fn profile_prefill_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("CERA_PROFILE_PREFILL").is_ok_and(|v| v != "0" && !v.is_empty())
        })
    }

    fn profile_decode_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("CERA_PROFILE_DECODE").is_some())
    }

    /// Shared layer loop for prefill passes. Runs all layers and updates `hidden` in place.
    fn prefill_layers_loop(
        &self,
        hidden: &mut [f32],
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        // Cloned once (cheap Arc bump) so the adapter can be read after each
        // projection GEMM while the base-weight scratch buffers stay borrowed.
        let lora = state.lora.clone();

        // Per-layer hidden-state diagnostic. Set `CERA_DEBUG_HIDDEN=1`
        // to print the last token's RMS at each layer entry, after
        // attn/conv (post 1st residual), and after FFN (post 2nd
        // residual). Used to find the layer where cera's hidden
        // states diverge from llama.cpp's reference. Cached in a
        // OnceLock so hot-path prefill performs zero syscalls.
        let debug_hidden = Self::debug_hidden_enabled();
        let log_rms = |label: &str, hidden: &[f32]| {
            if !debug_hidden {
                return;
            }
            // Last token's hidden vector lives at `hidden[(n-1)*hs..n*hs]` (row-major).
            // RMS of those `hs` values is what feeds the next layer / output norm.
            let last_tok = &hidden[(n - 1) * hs..n * hs];
            let mut sum_sq = 0.0f64;
            let mut max_abs = 0.0f64;
            for &v in last_tok {
                let v = v as f64;
                sum_sq += v * v;
                let abs_v = v.abs();
                if abs_v > max_abs {
                    max_abs = abs_v;
                }
            }
            let rms = (sum_sq / hs as f64).sqrt();
            eprintln!("[cera.hidden] {label}: rms={rms:.6e} max_abs={max_abs:.6e}");
        };
        log_rms("input (pre-layer-0)", hidden);

        // Per-layer loop: reuse prefill scratch buffers across layers and turns
        let mut scratch = std::mem::take(&mut state.prefill_scratch);
        scratch.ensure(n, cfg);
        #[allow(unused_variables)]
        let crate::kv_cache::PrefillScratch {
            normed,
            block_out,
            ffn_input,
            ffn_out,
            col,
            gate_col,
            up_col,
            out_col,
            proj_mat,
            out_proj_input,
            q_mat,
            k_mat,
            v_mat,
            bq_scales,
            bq_quants,
            gate_mat,
            up_mat,
            gate_up_mat,
            dq_scales,
            dq_quants,
            flash_out,
            q_col,
            kv_widen_k,
            kv_widen_v,
        } = &mut scratch;
        #[allow(unused_variables)]
        let bq_scales = &mut bq_scales[..n * hs.div_ceil(32)];
        #[allow(unused_variables)]
        let bq_quants = &mut bq_quants[..n * hs];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
        let is = cfg.intermediate_size;

        let profile_prefill = Self::profile_prefill_enabled();
        #[allow(unused_mut)]
        let mut t_norm = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_conv_in = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_conv_core = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_conv_out = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_attn_qkv = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_attn_out = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_ffn_norm = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_ffn_gate_up = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_ffn_down = Duration::ZERO;
        #[allow(unused_mut)]
        let mut t_residuals = Duration::ZERO;
        // B-activation quantization for every in-layer GEMM (attn/conv/FFN).
        // Split out of the GEMM phases so the profile line separates kernel
        // time from packing time.
        #[allow(unused_mut)]
        let mut t_quant = Duration::ZERO;
        // Attention scores (Q transpose + flash over heads + output
        // transpose-back) on the uncompressed path; TQ debug modes excluded.
        #[allow(unused_mut)]
        let mut t_attn_scores = Duration::ZERO;

        for layer in 0..cfg.n_layers {
            let t0 = Instant::now();
            // RMSnorm each token row
            let w_attn = &self.attn_norm_weights[layer];
            let eps = cfg.rms_norm_eps;
            if n >= 512 {
                let hidden_ptr = hidden.as_ptr() as usize;
                cpu::par_rows_n(&mut normed[..n * hs], hs, 64, move |(j, row)| unsafe {
                    let tok_h =
                        core::slice::from_raw_parts((hidden_ptr as *const f32).add(j * hs), hs);
                    cpu::rmsnorm_into(tok_h, row, w_attn, eps);
                });
            } else {
                for j in 0..n {
                    let tok_h = &hidden[j * hs..(j + 1) * hs];
                    let tok_norm = &mut normed[j * hs..(j + 1) * hs];
                    cpu::rmsnorm_into(tok_h, tok_norm, w_attn, eps);
                }
            }
            if profile_prefill {
                t_norm += t0.elapsed();
            }

            // Operator: conv or attention: batch projections via GEMM, sequential core
            let is_conv = cfg.block_types[layer] == BlockType::GatedConv;

            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
            let used_block_gemm = {
                let refs = &self.layer_refs[layer];
                if is_conv {
                    // --- Conv: batch in_proj + out_proj via GEMM ---
                    let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
                    let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
                    let blas_ok = [
                        ("shortconv.in_proj", in_proj),
                        ("shortconv.out_proj", out_proj),
                    ]
                    .into_iter()
                    .fold(true, |ok, (name, w)| {
                        if transformer::batched_gemm_supports(w.dtype, w.k) {
                            ok
                        } else {
                            transformer::warn_unbatchable(name, w.dtype);
                            false
                        }
                    });
                    if blas_ok {
                        #[cfg(not(has_blas))]
                        {
                            let t_q = Instant::now();
                            transformer::quantize_rows(normed, hs, n, bq_scales, bq_quants);
                            if profile_prefill {
                                t_quant += t_q.elapsed();
                            }
                            let t = Instant::now();
                            transformer::gemm_preq_rowmajor(
                                &self.gguf,
                                in_proj,
                                bq_scales,
                                bq_quants,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                3 * hs,
                                hs,
                            );
                            if profile_prefill {
                                t_conv_in += t.elapsed();
                            }
                        }
                        #[cfg(has_blas)]
                        {
                            let t = Instant::now();
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                in_proj,
                                normed,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                3 * hs,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            if profile_prefill {
                                t_conv_in += t.elapsed();
                            }
                        }

                        // LoRA on the conv in_proj
                        if let Some(lora) = &lora
                            && let Some(t) =
                                lora.get(layer, crate::lora::LoraTarget::ShortconvInProj)
                        {
                            crate::lora::apply_prefill_rowmajor(
                                t,
                                normed,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }

                        // Phase 2: Per-token sequential conv using pre-computed projections
                        let t_c = Instant::now();
                        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
                        let d_conv = kernel_size - 1;
                        let conv_wt = self.conv_weights[layer].as_ref().unwrap();
                        let (conv_w0, rest_w) = conv_wt.split_at(hs);
                        let (conv_w1, conv_w2) = rest_w.split_at(hs);

                        let is_causal = self.is_causal_for(state);

                        if !is_causal {
                            // Reuse pre-allocated ffn_input as scratch for bx_all;
                            // ffn_input is completely idle during the conv phase.
                            let bx_all = &mut ffn_input[..n * hs];
                            for j in 0..n {
                                let proj = &proj_mat[j * 3 * hs..(j + 1) * 3 * hs];
                                let b_slice = &proj[..hs];
                                let x_slice = &proj[2 * hs..3 * hs];
                                let bx_slice = &mut bx_all[j * hs..(j + 1) * hs];
                                for i in 0..hs {
                                    bx_slice[i] = b_slice[i] * x_slice[i];
                                }
                            }
                            for j in 0..n {
                                let proj = &proj_mat[j * 3 * hs..(j + 1) * 3 * hs];
                                let c_slice = &proj[hs..2 * hs];
                                let out_in = &mut out_proj_input[j * hs..(j + 1) * hs];

                                let prev_bx = if j > 0 {
                                    Some(&bx_all[(j - 1) * hs..j * hs])
                                } else {
                                    None
                                };
                                let curr_bx = &bx_all[j * hs..(j + 1) * hs];
                                let next_bx = if j + 1 < n {
                                    Some(&bx_all[(j + 1) * hs..(j + 2) * hs])
                                } else {
                                    None
                                };

                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    use core::arch::aarch64::*;
                                    let n_chunks = hs / 4;
                                    let c_ptr = c_slice.as_ptr();
                                    let curr_ptr = curr_bx.as_ptr();
                                    let w0_ptr = conv_w0.as_ptr();
                                    let w1_ptr = conv_w1.as_ptr();
                                    let w2_ptr = conv_w2.as_ptr();
                                    let out_ptr = out_in.as_mut_ptr();

                                    match (prev_bx, next_bx) {
                                        (Some(prev), Some(next)) => {
                                            let prev_ptr = prev.as_ptr();
                                            let next_ptr = next.as_ptr();
                                            for i in 0..n_chunks {
                                                let offset = i * 4;
                                                let curr_v = vld1q_f32(curr_ptr.add(offset));
                                                let w1_v = vld1q_f32(w1_ptr.add(offset));
                                                let prev_v = vld1q_f32(prev_ptr.add(offset));
                                                let w0_v = vld1q_f32(w0_ptr.add(offset));
                                                let next_v = vld1q_f32(next_ptr.add(offset));
                                                let w2_v = vld1q_f32(w2_ptr.add(offset));

                                                let conv_v = vfmaq_f32(
                                                    vfmaq_f32(
                                                        vmulq_f32(curr_v, w1_v),
                                                        prev_v,
                                                        w0_v,
                                                    ),
                                                    next_v,
                                                    w2_v,
                                                );
                                                let c_v = vld1q_f32(c_ptr.add(offset));
                                                vst1q_f32(
                                                    out_ptr.add(offset),
                                                    vmulq_f32(c_v, conv_v),
                                                );
                                            }
                                        }
                                        (None, Some(next)) => {
                                            let next_ptr = next.as_ptr();
                                            for i in 0..n_chunks {
                                                let offset = i * 4;
                                                let curr_v = vld1q_f32(curr_ptr.add(offset));
                                                let w1_v = vld1q_f32(w1_ptr.add(offset));
                                                let next_v = vld1q_f32(next_ptr.add(offset));
                                                let w2_v = vld1q_f32(w2_ptr.add(offset));

                                                let conv_v = vfmaq_f32(
                                                    vmulq_f32(curr_v, w1_v),
                                                    next_v,
                                                    w2_v,
                                                );
                                                let c_v = vld1q_f32(c_ptr.add(offset));
                                                vst1q_f32(
                                                    out_ptr.add(offset),
                                                    vmulq_f32(c_v, conv_v),
                                                );
                                            }
                                        }
                                        (Some(prev), None) => {
                                            let prev_ptr = prev.as_ptr();
                                            for i in 0..n_chunks {
                                                let offset = i * 4;
                                                let curr_v = vld1q_f32(curr_ptr.add(offset));
                                                let w1_v = vld1q_f32(w1_ptr.add(offset));
                                                let prev_v = vld1q_f32(prev_ptr.add(offset));
                                                let w0_v = vld1q_f32(w0_ptr.add(offset));

                                                let conv_v = vfmaq_f32(
                                                    vmulq_f32(curr_v, w1_v),
                                                    prev_v,
                                                    w0_v,
                                                );
                                                let c_v = vld1q_f32(c_ptr.add(offset));
                                                vst1q_f32(
                                                    out_ptr.add(offset),
                                                    vmulq_f32(c_v, conv_v),
                                                );
                                            }
                                        }
                                        (None, None) => {
                                            for i in 0..n_chunks {
                                                let offset = i * 4;
                                                let curr_v = vld1q_f32(curr_ptr.add(offset));
                                                let w1_v = vld1q_f32(w1_ptr.add(offset));
                                                let conv_v = vmulq_f32(curr_v, w1_v);
                                                let c_v = vld1q_f32(c_ptr.add(offset));
                                                vst1q_f32(
                                                    out_ptr.add(offset),
                                                    vmulq_f32(c_v, conv_v),
                                                );
                                            }
                                        }
                                    }
                                }

                                #[cfg(not(target_arch = "aarch64"))]
                                match (prev_bx, next_bx) {
                                    (Some(prev), Some(next)) => {
                                        for i in 0..hs {
                                            out_in[i] = c_slice[i]
                                                * (curr_bx[i] * conv_w1[i]
                                                    + prev[i] * conv_w0[i]
                                                    + next[i] * conv_w2[i]);
                                        }
                                    }
                                    (None, Some(next)) => {
                                        for i in 0..hs {
                                            out_in[i] = c_slice[i]
                                                * (curr_bx[i] * conv_w1[i] + next[i] * conv_w2[i]);
                                        }
                                    }
                                    (Some(prev), None) => {
                                        for i in 0..hs {
                                            out_in[i] = c_slice[i]
                                                * (curr_bx[i] * conv_w1[i] + prev[i] * conv_w0[i]);
                                        }
                                    }
                                    (None, None) => {
                                        for i in 0..hs {
                                            out_in[i] = c_slice[i] * (curr_bx[i] * conv_w1[i]);
                                        }
                                    }
                                }
                            }
                        } else if kernel_size != 3 {
                            // Any other tap count (`lfm3` has 7): the loops below are written
                            // for exactly three taps, so walk the taps. Same summation order and
                            // rounding as the decode kernel (buffered taps oldest first, then the
                            // new bx; see `conv_mac`).
                            for j in 0..n {
                                let proj = &proj_mat[j * 3 * hs..(j + 1) * 3 * hs];
                                let (b_slice, c_slice, x_slice) =
                                    (&proj[..hs], &proj[hs..2 * hs], &proj[2 * hs..3 * hs]);

                                let LayerState::Conv { buffer, history } = &mut state.layers[layer]
                                else {
                                    panic!("expected Conv state for layer {layer}");
                                };
                                let out_in = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let bx = &mut state.scratch.conv_scratch[..hs];

                                bx.iter_mut()
                                    .zip(b_slice.iter().zip(x_slice))
                                    .for_each(|(o, (&b, &x))| *o = b * x);
                                out_in
                                    .iter_mut()
                                    .zip(buffer[..hs].iter().zip(&conv_wt[..hs]))
                                    .for_each(|(o, (&s, &w))| *o = s * w);
                                (1..d_conv).for_each(|t| {
                                    out_in
                                        .iter_mut()
                                        .zip(buffer[t * hs..(t + 1) * hs].iter())
                                        .zip(&conv_wt[t * hs..(t + 1) * hs])
                                        .for_each(|((o, &s), &w)| *o = conv_mac(*o, s, w));
                                });
                                out_in
                                    .iter_mut()
                                    .zip(bx.iter().zip(&conv_wt[d_conv * hs..(d_conv + 1) * hs]))
                                    .zip(c_slice)
                                    .for_each(|((o, (&b, &w)), &c)| *o = c * conv_mac(*o, b, w));

                                buffer.copy_within(hs.., 0);
                                buffer[(d_conv - 1) * hs..d_conv * hs].copy_from_slice(bx);
                                history.push(start_pos + j + 1, buffer);
                            }
                        } else {
                            for j in 0..n {
                                let proj = &proj_mat[j * 3 * hs..(j + 1) * 3 * hs];
                                let b_slice = &proj[..hs];
                                let c_slice = &proj[hs..2 * hs];
                                let x_slice = &proj[2 * hs..3 * hs];

                                let LayerState::Conv { buffer, history } = &mut state.layers[layer]
                                else {
                                    panic!("expected Conv state for layer {layer}");
                                };

                                let out_in = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let conv_scratch = &mut state.scratch.conv_scratch[..hs];

                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    use core::arch::aarch64::*;
                                    let n_chunks = hs / 4;
                                    let b_ptr = b_slice.as_ptr();
                                    let x_ptr = x_slice.as_ptr();
                                    let c_ptr = c_slice.as_ptr();
                                    let b0_ptr = buffer.as_ptr();
                                    let b1_ptr = buffer.as_ptr().add(hs);
                                    let w0_ptr = conv_w0.as_ptr();
                                    let w1_ptr = conv_w1.as_ptr();
                                    let w2_ptr = conv_w2.as_ptr();
                                    let out_in_ptr = out_in.as_mut_ptr();
                                    let cs_ptr = conv_scratch.as_mut_ptr();

                                    for i in 0..n_chunks {
                                        let b_v = vld1q_f32(b_ptr.add(i * 4));
                                        let x_v = vld1q_f32(x_ptr.add(i * 4));
                                        let bx_v = vmulq_f32(b_v, x_v);
                                        vst1q_f32(cs_ptr.add(i * 4), bx_v);

                                        let b0_v = vld1q_f32(b0_ptr.add(i * 4));
                                        let b1_v = vld1q_f32(b1_ptr.add(i * 4));
                                        let w0_v = vld1q_f32(w0_ptr.add(i * 4));
                                        let w1_v = vld1q_f32(w1_ptr.add(i * 4));
                                        let w2_v = vld1q_f32(w2_ptr.add(i * 4));

                                        let out_v = vfmaq_f32(
                                            vfmaq_f32(vmulq_f32(b0_v, w0_v), b1_v, w1_v),
                                            bx_v,
                                            w2_v,
                                        );
                                        let c_v = vld1q_f32(c_ptr.add(i * 4));
                                        let final_v = vmulq_f32(c_v, out_v);
                                        vst1q_f32(out_in_ptr.add(i * 4), final_v);
                                    }
                                }

                                #[cfg(not(target_arch = "aarch64"))]
                                {
                                    for i in 0..hs {
                                        conv_scratch[i] = b_slice[i] * x_slice[i];
                                        let conv_out = buffer[i] * conv_w0[i]
                                            + buffer[hs + i] * conv_w1[i]
                                            + conv_scratch[i] * conv_w2[i];
                                        out_in[i] = c_slice[i] * conv_out;
                                    }
                                }

                                if d_conv > 0 {
                                    if d_conv > 1 {
                                        buffer.copy_within(hs.., 0);
                                    }
                                    let last_slot = (d_conv - 1) * hs;
                                    buffer[last_slot..last_slot + hs].copy_from_slice(conv_scratch);
                                    history.push(start_pos + j + 1, buffer);
                                }
                            }
                        }
                        if profile_prefill {
                            t_conv_core += t_c.elapsed();
                        }

                        // Phase 3: Batch out_proj GEMM
                        #[cfg(not(has_blas))]
                        {
                            let t_q = Instant::now();
                            transformer::quantize_rows(out_proj_input, hs, n, bq_scales, bq_quants);
                            if profile_prefill {
                                t_quant += t_q.elapsed();
                            }
                            let t_o = Instant::now();
                            transformer::gemm_preq_rowmajor(
                                &self.gguf,
                                out_proj,
                                bq_scales,
                                bq_quants,
                                &mut block_out[..hs * n],
                                n,
                                hs,
                                hs,
                            );
                            if profile_prefill {
                                t_conv_out += t_o.elapsed();
                            }
                        }
                        #[cfg(has_blas)]
                        {
                            let t_o = Instant::now();
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                out_proj,
                                out_proj_input,
                                block_out,
                                n,
                                hs,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            if profile_prefill {
                                t_conv_out += t_o.elapsed();
                            }
                        }

                        // LoRA on the conv out_proj
                        if let Some(lora) = &lora
                            && let Some(t) =
                                lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
                        {
                            crate::lora::apply_prefill_rowmajor(
                                t,
                                out_proj_input,
                                block_out,
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    // --- Attention: batch Q/K/V + output projections via GEMM ---
                    let attn_q_ref = refs.attn_q.as_ref().unwrap();
                    let attn_k_ref = refs.attn_k.as_ref().unwrap();
                    let attn_v_ref = refs.attn_v.as_ref().unwrap();
                    let attn_output_ref = refs.attn_output.as_ref().unwrap();
                    let lfm3_attn = self
                        .lfm3
                        .as_ref()
                        .and_then(|l| l.layers[layer].attn.as_ref());
                    // Require ALL projections (including LFM3's per-head gate when present)
                    // to be batchable. A mixed-dtype attention block would leave later matrices
                    // silently uncomputed in the batched path and produce wrong outputs.
                    let blas_ok = [
                        ("attn_q", attn_q_ref),
                        ("attn_k", attn_k_ref),
                        ("attn_v", attn_v_ref),
                        ("attn_output", attn_output_ref),
                    ]
                    .into_iter()
                    .fold(true, |ok, (name, w)| {
                        if transformer::batched_gemm_supports(w.dtype, w.k) {
                            ok
                        } else {
                            if self.lfm3.is_none() {
                                transformer::warn_unbatchable(name, w.dtype);
                            }
                            false
                        }
                    });
                    let gate_ok = lfm3_attn
                        .is_none_or(|a| transformer::batched_gemm_supports(a.gate.dtype, a.gate.k));
                    let dims_ok = cfg.n_heads * cfg.head_dim == hs;
                    let tq_sliding_ok =
                        lfm3_attn.and_then(|a| a.window).is_none() || state.tq_config.is_none();
                    if blas_ok
                        && gate_ok
                        && dims_ok
                        && tq_sliding_ok
                        && !transformer::oracle_dump::is_active()
                    {
                        let head_dim = cfg.head_dim;
                        let n_kv_heads = cfg.kv_heads_per_layer[layer];
                        let kv_dim = n_kv_heads * head_dim;

                        // Phase 1: Batch Q/K/V GEMM (Single fused AMX dispatch)
                        #[cfg(has_blas)]
                        {
                            let t_qkv = Instant::now();
                            let qkv_dim = hs + 2 * kv_dim;
                            let mut qkv_rows = vec![0.0f32; qkv_dim * hs];
                            let q_data = transformer::weight_data(&self.gguf, attn_q_ref);
                            let k_data = transformer::weight_data(&self.gguf, attn_k_ref);
                            let v_data = transformer::weight_data(&self.gguf, attn_v_ref);
                            if let (Some(dq_q), Some(dq_k), Some(dq_v)) = (
                                transformer::blas_dequantizer(attn_q_ref.dtype),
                                transformer::blas_dequantizer(attn_k_ref.dtype),
                                transformer::blas_dequantizer(attn_v_ref.dtype),
                            ) {
                                dq_q(q_data, hs, hs, &mut qkv_rows[..hs * hs]);
                                dq_k(
                                    k_data,
                                    kv_dim,
                                    hs,
                                    &mut qkv_rows[hs * hs..(hs + kv_dim) * hs],
                                );
                                dq_v(v_data, kv_dim, hs, &mut qkv_rows[(hs + kv_dim) * hs..]);
                            }

                            crate::backend::blas::sgemm_rowmajor_nt_parallel(
                                n,
                                qkv_dim,
                                hs,
                                normed,
                                &qkv_rows,
                                &mut proj_mat[..qkv_dim * n],
                            );

                            for j in 0..n {
                                let row = &proj_mat[j * qkv_dim..(j + 1) * qkv_dim];
                                q_mat[j * hs..(j + 1) * hs].copy_from_slice(&row[..hs]);
                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                    .copy_from_slice(&row[hs..hs + kv_dim]);
                                v_mat[j * kv_dim..(j + 1) * kv_dim]
                                    .copy_from_slice(&row[hs + kv_dim..qkv_dim]);
                            }
                            if let Some(a) = lfm3_attn {
                                let gate_buf = &mut ffn_input[..n * cfg.n_heads];
                                transformer::try_blas_prefill_gemm_rowmajor(
                                    &self.gguf,
                                    &a.gate,
                                    normed,
                                    gate_buf,
                                    n,
                                    cfg.n_heads,
                                    hs,
                                    &mut state.scratch.dequant_weight_scratch,
                                );
                                cpu::sigmoid_inplace(gate_buf);
                            }
                            if profile_prefill {
                                t_attn_qkv += t_qkv.elapsed();
                            }
                        }
                        #[cfg(not(has_blas))]
                        {
                            let t_q = Instant::now();
                            transformer::quantize_rows(normed, hs, n, bq_scales, bq_quants);
                            if profile_prefill {
                                t_quant += t_q.elapsed();
                            }
                            let t_qkv = Instant::now();

                            let qkv_dim = hs + 2 * kv_dim;
                            // (concatenated packed, concatenated scales, smmla layout?).
                            // Concatenation is layout-agnostic (per-super-row blocks
                            // in both repacks), but the fused dispatch must match the
                            // layout all three shards share.
                            let qkv_repacked = refs.qkv_repacked.get_or_init(|| {
                                #[allow(clippy::collapsible_if)]
                                if let (Some(q_rp), Some(k_rp)) =
                                    (&attn_q_ref.repacked, &attn_k_ref.repacked)
                                {
                                    // Check for 3-way fusion if V is also available with repacked weights
                                    if let Some(v_rp) = &attn_v_ref.repacked {
                                        match (&q_rp.kind, &k_rp.kind, &v_rp.kind) {
                                            (
                                                transformer::Repacked::Q40Smmla {
                                                    packed: q_p,
                                                    scales: q_s,
                                                },
                                                transformer::Repacked::Q40Smmla {
                                                    packed: k_p,
                                                    scales: k_s,
                                                },
                                                transformer::Repacked::Q40Smmla {
                                                    packed: v_p,
                                                    scales: v_s,
                                                },
                                            ) => {
                                                let mut p = Vec::with_capacity(
                                                    q_p.len() + k_p.len() + v_p.len(),
                                                );
                                                p.extend_from_slice(q_p);
                                                p.extend_from_slice(k_p);
                                                p.extend_from_slice(v_p);

                                                let mut s = Vec::with_capacity(
                                                    q_s.len() + k_s.len() + v_s.len(),
                                                );
                                                s.extend_from_slice(q_s);
                                                s.extend_from_slice(k_s);
                                                s.extend_from_slice(v_s);
                                                return Some(FusedAttnProjections::Q40All {
                                                    packed: p,
                                                    scales: s,
                                                    smmla: true,
                                                });
                                            }
                                            (
                                                transformer::Repacked::Q40 {
                                                    packed: q_p,
                                                    scales: q_s,
                                                },
                                                transformer::Repacked::Q40 {
                                                    packed: k_p,
                                                    scales: k_s,
                                                },
                                                transformer::Repacked::Q40 {
                                                    packed: v_p,
                                                    scales: v_s,
                                                },
                                            ) => {
                                                let mut p = Vec::with_capacity(
                                                    q_p.len() + k_p.len() + v_p.len(),
                                                );
                                                p.extend_from_slice(q_p);
                                                p.extend_from_slice(k_p);
                                                p.extend_from_slice(v_p);

                                                let mut s = Vec::with_capacity(
                                                    q_s.len() + k_s.len() + v_s.len(),
                                                );
                                                s.extend_from_slice(q_s);
                                                s.extend_from_slice(k_s);
                                                s.extend_from_slice(v_s);
                                                return Some(FusedAttnProjections::Q40All {
                                                    packed: p,
                                                    scales: s,
                                                    smmla: false,
                                                });
                                            }
                                            (
                                                transformer::Repacked::Q4KSmmla {
                                                    packed: q_p,
                                                    dsc: q_dsc,
                                                    dmn: q_dmn,
                                                },
                                                transformer::Repacked::Q4KSmmla {
                                                    packed: k_p,
                                                    dsc: k_dsc,
                                                    dmn: k_dmn,
                                                },
                                                transformer::Repacked::Q4KSmmla {
                                                    packed: v_p,
                                                    dsc: v_dsc,
                                                    dmn: v_dmn,
                                                },
                                            ) => {
                                                let mut p = Vec::with_capacity(
                                                    q_p.len() + k_p.len() + v_p.len(),
                                                );
                                                p.extend_from_slice(q_p);
                                                p.extend_from_slice(k_p);
                                                p.extend_from_slice(v_p);

                                                let mut dsc = Vec::with_capacity(
                                                    q_dsc.len() + k_dsc.len() + v_dsc.len(),
                                                );
                                                dsc.extend_from_slice(q_dsc);
                                                dsc.extend_from_slice(k_dsc);
                                                dsc.extend_from_slice(v_dsc);

                                                let mut dmn = Vec::with_capacity(
                                                    q_dmn.len() + k_dmn.len() + v_dmn.len(),
                                                );
                                                dmn.extend_from_slice(q_dmn);
                                                dmn.extend_from_slice(k_dmn);
                                                dmn.extend_from_slice(v_dmn);
                                                return Some(FusedAttnProjections::Q4KAll {
                                                    packed: p,
                                                    dsc,
                                                    dmn,
                                                    smmla: true,
                                                });
                                            }
                                            (
                                                transformer::Repacked::Q4K {
                                                    packed: q_p,
                                                    dsc: q_dsc,
                                                    dmn: q_dmn,
                                                },
                                                transformer::Repacked::Q4K {
                                                    packed: k_p,
                                                    dsc: k_dsc,
                                                    dmn: k_dmn,
                                                },
                                                transformer::Repacked::Q4K {
                                                    packed: v_p,
                                                    dsc: v_dsc,
                                                    dmn: v_dmn,
                                                },
                                            ) => {
                                                let mut p = Vec::with_capacity(
                                                    q_p.len() + k_p.len() + v_p.len(),
                                                );
                                                p.extend_from_slice(q_p);
                                                p.extend_from_slice(k_p);
                                                p.extend_from_slice(v_p);

                                                let mut dsc = Vec::with_capacity(
                                                    q_dsc.len() + k_dsc.len() + v_dsc.len(),
                                                );
                                                dsc.extend_from_slice(q_dsc);
                                                dsc.extend_from_slice(k_dsc);
                                                dsc.extend_from_slice(v_dsc);

                                                let mut dmn = Vec::with_capacity(
                                                    q_dmn.len() + k_dmn.len() + v_dmn.len(),
                                                );
                                                dmn.extend_from_slice(q_dmn);
                                                dmn.extend_from_slice(k_dmn);
                                                dmn.extend_from_slice(v_dmn);
                                                return Some(FusedAttnProjections::Q4KAll {
                                                    packed: p,
                                                    dsc,
                                                    dmn,
                                                    smmla: false,
                                                });
                                            }
                                            (
                                                transformer::Repacked::Q6KSmmla {
                                                    packed: q_p,
                                                    scales: q_s,
                                                },
                                                transformer::Repacked::Q6KSmmla {
                                                    packed: k_p,
                                                    scales: k_s,
                                                },
                                                transformer::Repacked::Q6KSmmla {
                                                    packed: v_p,
                                                    scales: v_s,
                                                },
                                            ) => {
                                                let mut p = Vec::with_capacity(
                                                    q_p.len() + k_p.len() + v_p.len(),
                                                );
                                                p.extend_from_slice(q_p);
                                                p.extend_from_slice(k_p);
                                                p.extend_from_slice(v_p);

                                                let mut s = Vec::with_capacity(
                                                    q_s.len() + k_s.len() + v_s.len(),
                                                );
                                                s.extend_from_slice(q_s);
                                                s.extend_from_slice(k_s);
                                                s.extend_from_slice(v_s);
                                                return Some(FusedAttnProjections::Q6KAll {
                                                    packed: p,
                                                    scales: s,
                                                    smmla: true,
                                                });
                                            }
                                            (
                                                transformer::Repacked::Q6K {
                                                    packed: q_p,
                                                    scales: q_s,
                                                },
                                                transformer::Repacked::Q6K {
                                                    packed: k_p,
                                                    scales: k_s,
                                                },
                                                transformer::Repacked::Q6K {
                                                    packed: v_p,
                                                    scales: v_s,
                                                },
                                            ) => {
                                                let mut p = Vec::with_capacity(
                                                    q_p.len() + k_p.len() + v_p.len(),
                                                );
                                                p.extend_from_slice(q_p);
                                                p.extend_from_slice(k_p);
                                                p.extend_from_slice(v_p);

                                                let mut s = Vec::with_capacity(
                                                    q_s.len() + k_s.len() + v_s.len(),
                                                );
                                                s.extend_from_slice(q_s);
                                                s.extend_from_slice(k_s);
                                                s.extend_from_slice(v_s);
                                                return Some(FusedAttnProjections::Q6KAll {
                                                    packed: p,
                                                    scales: s,
                                                    smmla: false,
                                                });
                                            }
                                            _ => {}
                                        }
                                    }

                                    // Fall back to 2-way QK fusion if Q and K share the same repack format
                                    match (&q_rp.kind, &k_rp.kind) {
                                        (
                                            transformer::Repacked::Q4KSmmla {
                                                packed: q_p,
                                                dsc: q_dsc,
                                                dmn: q_dmn,
                                            },
                                            transformer::Repacked::Q4KSmmla {
                                                packed: k_p,
                                                dsc: k_dsc,
                                                dmn: k_dmn,
                                            },
                                        ) => {
                                            let mut p = Vec::with_capacity(q_p.len() + k_p.len());
                                            p.extend_from_slice(q_p);
                                            p.extend_from_slice(k_p);

                                            let mut dsc =
                                                Vec::with_capacity(q_dsc.len() + k_dsc.len());
                                            dsc.extend_from_slice(q_dsc);
                                            dsc.extend_from_slice(k_dsc);

                                            let mut dmn =
                                                Vec::with_capacity(q_dmn.len() + k_dmn.len());
                                            dmn.extend_from_slice(q_dmn);
                                            dmn.extend_from_slice(k_dmn);
                                            return Some(FusedAttnProjections::Q4KQkOnly {
                                                packed: p,
                                                dsc,
                                                dmn,
                                                smmla: true,
                                            });
                                        }
                                        (
                                            transformer::Repacked::Q4K {
                                                packed: q_p,
                                                dsc: q_dsc,
                                                dmn: q_dmn,
                                            },
                                            transformer::Repacked::Q4K {
                                                packed: k_p,
                                                dsc: k_dsc,
                                                dmn: k_dmn,
                                            },
                                        ) => {
                                            let mut p = Vec::with_capacity(q_p.len() + k_p.len());
                                            p.extend_from_slice(q_p);
                                            p.extend_from_slice(k_p);

                                            let mut dsc =
                                                Vec::with_capacity(q_dsc.len() + k_dsc.len());
                                            dsc.extend_from_slice(q_dsc);
                                            dsc.extend_from_slice(k_dsc);

                                            let mut dmn =
                                                Vec::with_capacity(q_dmn.len() + k_dmn.len());
                                            dmn.extend_from_slice(q_dmn);
                                            dmn.extend_from_slice(k_dmn);
                                            return Some(FusedAttnProjections::Q4KQkOnly {
                                                packed: p,
                                                dsc,
                                                dmn,
                                                smmla: false,
                                            });
                                        }
                                        (
                                            transformer::Repacked::Q40Smmla {
                                                packed: q_p,
                                                scales: q_s,
                                            },
                                            transformer::Repacked::Q40Smmla {
                                                packed: k_p,
                                                scales: k_s,
                                            },
                                        ) => {
                                            let mut p = Vec::with_capacity(q_p.len() + k_p.len());
                                            p.extend_from_slice(q_p);
                                            p.extend_from_slice(k_p);

                                            let mut s = Vec::with_capacity(q_s.len() + k_s.len());
                                            s.extend_from_slice(q_s);
                                            s.extend_from_slice(k_s);
                                            return Some(FusedAttnProjections::Q40QkOnly {
                                                packed: p,
                                                scales: s,
                                                smmla: true,
                                            });
                                        }
                                        (
                                            transformer::Repacked::Q40 {
                                                packed: q_p,
                                                scales: q_s,
                                            },
                                            transformer::Repacked::Q40 {
                                                packed: k_p,
                                                scales: k_s,
                                            },
                                        ) => {
                                            let mut p = Vec::with_capacity(q_p.len() + k_p.len());
                                            p.extend_from_slice(q_p);
                                            p.extend_from_slice(k_p);

                                            let mut s = Vec::with_capacity(q_s.len() + k_s.len());
                                            s.extend_from_slice(q_s);
                                            s.extend_from_slice(k_s);
                                            return Some(FusedAttnProjections::Q40QkOnly {
                                                packed: p,
                                                scales: s,
                                                smmla: false,
                                            });
                                        }
                                        (
                                            transformer::Repacked::Q6KSmmla {
                                                packed: q_p,
                                                scales: q_s,
                                            },
                                            transformer::Repacked::Q6KSmmla {
                                                packed: k_p,
                                                scales: k_s,
                                            },
                                        ) => {
                                            let mut p = Vec::with_capacity(q_p.len() + k_p.len());
                                            p.extend_from_slice(q_p);
                                            p.extend_from_slice(k_p);

                                            let mut s = Vec::with_capacity(q_s.len() + k_s.len());
                                            s.extend_from_slice(q_s);
                                            s.extend_from_slice(k_s);
                                            return Some(FusedAttnProjections::Q6KQkOnly {
                                                packed: p,
                                                scales: s,
                                                smmla: true,
                                            });
                                        }
                                        (
                                            transformer::Repacked::Q6K {
                                                packed: q_p,
                                                scales: q_s,
                                            },
                                            transformer::Repacked::Q6K {
                                                packed: k_p,
                                                scales: k_s,
                                            },
                                        ) => {
                                            let mut p = Vec::with_capacity(q_p.len() + k_p.len());
                                            p.extend_from_slice(q_p);
                                            p.extend_from_slice(k_p);

                                            let mut s = Vec::with_capacity(q_s.len() + k_s.len());
                                            s.extend_from_slice(q_s);
                                            s.extend_from_slice(k_s);
                                            return Some(FusedAttnProjections::Q6KQkOnly {
                                                packed: p,
                                                scales: s,
                                                smmla: false,
                                            });
                                        }
                                        _ => {}
                                    }
                                }
                                None
                            });

                            let mut ran_fused = false;
                            if let Some(fused) = qkv_repacked.as_ref() {
                                match fused {
                                    FusedAttnProjections::Q40All {
                                        packed,
                                        scales,
                                        smmla,
                                    } => {
                                        let ok = if *smmla {
                                            crate::backend::cpu::gemm_preq_repacked_q4_0_smmla_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qkv_dim * n],
                                                n,
                                                qkv_dim,
                                                hs,
                                            )
                                        } else {
                                            crate::backend::cpu::gemm_preq_repacked_q4_0_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qkv_dim * n],
                                                n,
                                                qkv_dim,
                                                hs,
                                            )
                                        };
                                        if ok {
                                            ran_fused = true;
                                            for j in 0..n {
                                                let row = &proj_mat[j * qkv_dim..(j + 1) * qkv_dim];
                                                q_mat[j * hs..(j + 1) * hs]
                                                    .copy_from_slice(&row[..hs]);
                                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs..hs + kv_dim]);
                                                v_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs + kv_dim..qkv_dim]);
                                            }
                                        }
                                    }
                                    FusedAttnProjections::Q4KAll {
                                        packed,
                                        dsc,
                                        dmn,
                                        smmla,
                                    } => {
                                        let ok = if *smmla {
                                            crate::backend::cpu::gemm_preq_repacked_q4_k_smmla_rowmajor_dispatch(
                                                packed,
                                                dsc,
                                                dmn,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qkv_dim * n],
                                                n,
                                                qkv_dim,
                                                hs,
                                            )
                                        } else {
                                            crate::backend::cpu::gemm_preq_repacked_q4_k_rowmajor_dispatch(
                                                packed,
                                                dsc,
                                                dmn,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qkv_dim * n],
                                                n,
                                                qkv_dim,
                                                hs,
                                            )
                                        };
                                        if ok {
                                            ran_fused = true;
                                            for j in 0..n {
                                                let row = &proj_mat[j * qkv_dim..(j + 1) * qkv_dim];
                                                q_mat[j * hs..(j + 1) * hs]
                                                    .copy_from_slice(&row[..hs]);
                                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs..hs + kv_dim]);
                                                v_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs + kv_dim..qkv_dim]);
                                            }
                                        }
                                    }
                                    FusedAttnProjections::Q6KAll {
                                        packed,
                                        scales,
                                        smmla,
                                    } => {
                                        let ok = if *smmla {
                                            crate::backend::cpu::gemm_preq_repacked_q6_k_smmla_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qkv_dim * n],
                                                n,
                                                qkv_dim,
                                                hs,
                                            )
                                        } else {
                                            crate::backend::cpu::gemm_preq_repacked_q6_k_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qkv_dim * n],
                                                n,
                                                qkv_dim,
                                                hs,
                                            )
                                        };
                                        if ok {
                                            ran_fused = true;
                                            for j in 0..n {
                                                let row = &proj_mat[j * qkv_dim..(j + 1) * qkv_dim];
                                                q_mat[j * hs..(j + 1) * hs]
                                                    .copy_from_slice(&row[..hs]);
                                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs..hs + kv_dim]);
                                                v_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs + kv_dim..qkv_dim]);
                                            }
                                        }
                                    }
                                    FusedAttnProjections::Q4KQkOnly {
                                        packed,
                                        dsc,
                                        dmn,
                                        smmla,
                                    } => {
                                        let qk_dim = hs + kv_dim;
                                        let ok = if *smmla {
                                            crate::backend::cpu::gemm_preq_repacked_q4_k_smmla_rowmajor_dispatch(
                                                packed,
                                                dsc,
                                                dmn,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qk_dim * n],
                                                n,
                                                qk_dim,
                                                hs,
                                            )
                                        } else {
                                            crate::backend::cpu::gemm_preq_repacked_q4_k_rowmajor_dispatch(
                                                packed,
                                                dsc,
                                                dmn,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qk_dim * n],
                                                n,
                                                qk_dim,
                                                hs,
                                            )
                                        };
                                        if ok {
                                            ran_fused = true;
                                            for j in 0..n {
                                                let row = &proj_mat[j * qk_dim..(j + 1) * qk_dim];
                                                q_mat[j * hs..(j + 1) * hs]
                                                    .copy_from_slice(&row[..hs]);
                                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs..qk_dim]);
                                            }
                                            transformer::gemm_preq_rowmajor(
                                                &self.gguf,
                                                attn_v_ref,
                                                bq_scales,
                                                bq_quants,
                                                &mut v_mat[..kv_dim * n],
                                                n,
                                                kv_dim,
                                                hs,
                                            );
                                        }
                                    }
                                    FusedAttnProjections::Q40QkOnly {
                                        packed,
                                        scales,
                                        smmla,
                                    } => {
                                        let qk_dim = hs + kv_dim;
                                        let ok = if *smmla {
                                            crate::backend::cpu::gemm_preq_repacked_q4_0_smmla_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qk_dim * n],
                                                n,
                                                qk_dim,
                                                hs,
                                            )
                                        } else {
                                            crate::backend::cpu::gemm_preq_repacked_q4_0_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qk_dim * n],
                                                n,
                                                qk_dim,
                                                hs,
                                            )
                                        };
                                        if ok {
                                            ran_fused = true;
                                            for j in 0..n {
                                                let row = &proj_mat[j * qk_dim..(j + 1) * qk_dim];
                                                q_mat[j * hs..(j + 1) * hs]
                                                    .copy_from_slice(&row[..hs]);
                                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs..qk_dim]);
                                            }
                                            transformer::gemm_preq_rowmajor(
                                                &self.gguf,
                                                attn_v_ref,
                                                bq_scales,
                                                bq_quants,
                                                &mut v_mat[..kv_dim * n],
                                                n,
                                                kv_dim,
                                                hs,
                                            );
                                        }
                                    }
                                    FusedAttnProjections::Q6KQkOnly {
                                        packed,
                                        scales,
                                        smmla,
                                    } => {
                                        let qk_dim = hs + kv_dim;
                                        let ok = if *smmla {
                                            crate::backend::cpu::gemm_preq_repacked_q6_k_smmla_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qk_dim * n],
                                                n,
                                                qk_dim,
                                                hs,
                                            )
                                        } else {
                                            crate::backend::cpu::gemm_preq_repacked_q6_k_rowmajor_dispatch(
                                                packed,
                                                scales,
                                                bq_scales,
                                                bq_quants,
                                                &mut proj_mat[..qk_dim * n],
                                                n,
                                                qk_dim,
                                                hs,
                                            )
                                        };
                                        if ok {
                                            ran_fused = true;
                                            for j in 0..n {
                                                let row = &proj_mat[j * qk_dim..(j + 1) * qk_dim];
                                                q_mat[j * hs..(j + 1) * hs]
                                                    .copy_from_slice(&row[..hs]);
                                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                                    .copy_from_slice(&row[hs..qk_dim]);
                                            }
                                            transformer::gemm_preq_rowmajor(
                                                &self.gguf,
                                                attn_v_ref,
                                                bq_scales,
                                                bq_quants,
                                                &mut v_mat[..kv_dim * n],
                                                n,
                                                kv_dim,
                                                hs,
                                            );
                                        }
                                    }
                                }
                            }

                            if !ran_fused {
                                transformer::gemm_preq_rowmajor(
                                    &self.gguf,
                                    attn_q_ref,
                                    bq_scales,
                                    bq_quants,
                                    &mut q_mat[..hs * n],
                                    n,
                                    hs,
                                    hs,
                                );
                                transformer::gemm_preq_rowmajor(
                                    &self.gguf,
                                    attn_k_ref,
                                    bq_scales,
                                    bq_quants,
                                    &mut k_mat[..kv_dim * n],
                                    n,
                                    kv_dim,
                                    hs,
                                );
                                transformer::gemm_preq_rowmajor(
                                    &self.gguf,
                                    attn_v_ref,
                                    bq_scales,
                                    bq_quants,
                                    &mut v_mat[..kv_dim * n],
                                    n,
                                    kv_dim,
                                    hs,
                                );
                            }
                            if let Some(a) = lfm3_attn {
                                let gate_buf = &mut ffn_input[..n * cfg.n_heads];
                                transformer::gemm_preq_rowmajor(
                                    &self.gguf,
                                    &a.gate,
                                    bq_scales,
                                    bq_quants,
                                    gate_buf,
                                    n,
                                    cfg.n_heads,
                                    hs,
                                );
                                cpu::sigmoid_inplace(gate_buf);
                            }
                            if profile_prefill {
                                t_attn_qkv += t_qkv.elapsed();
                            }
                        }

                        // LoRA on Q/K/V — added to the projection outputs before
                        // QK-norm/RoPE, input is the normed hidden `[hs×n]`.
                        if let Some(lora) = &lora {
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnQ) {
                                crate::lora::apply_prefill_rowmajor(
                                    t,
                                    normed,
                                    &mut q_mat[..hs * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnK) {
                                crate::lora::apply_prefill_rowmajor(
                                    t,
                                    normed,
                                    &mut k_mat[..kv_dim * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnV) {
                                crate::lora::apply_prefill_rowmajor(
                                    t,
                                    normed,
                                    &mut v_mat[..kv_dim * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                        }

                        // Phase 2: Per-token attention (QK norm, RoPE, KV cache, scores)
                        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
                        let tq_config = state.tq_config.as_ref();
                        let will_compress_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_encode_scratch.is_some();
                        let will_read_compressed_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_query_scratch.is_some();
                        let use_f16 = state.kv_f16;

                        if let LayerState::Attention {
                            key_cache,
                            value_cache,
                            key_cache_f16,
                            value_cache_f16,
                            compressed_keys,
                            compressed_values,
                        } = &mut state.layers[layer]
                        {
                            match (will_compress_kv, compressed_keys.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.jl_data.iter_mut() {
                                        v.reserve(n * head_dim / 8);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ if use_f16 => {
                                    key_cache_f16.reserve(n * kv_dim);
                                }
                                _ => {
                                    key_cache.reserve(n * kv_dim);
                                }
                            }
                            match (will_compress_kv, compressed_values.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ if use_f16 => {
                                    value_cache_f16.reserve(n * kv_dim);
                                }
                                _ => {
                                    value_cache.reserve(n * kv_dim);
                                }
                            }
                        }
                        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
                        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
                        let group_size = cfg.n_heads / n_kv_heads;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        // ── Pass A: QK-norm + RoPE + KV cache append ──────────
                        if n >= 4 {
                            let q_ptr = q_mat.as_mut_ptr() as usize;
                            let k_ptr = k_mat.as_mut_ptr() as usize;
                            let n_heads = cfg.n_heads;
                            let rope_theta = cfg.rope_theta;
                            let lfm3_rope_theta = lfm3_attn.map(|a| a.rope_theta);
                            let lfm3_freq_factors_ptr = lfm3_attn
                                .and_then(|a| a.rope_freq_factors.as_deref())
                                .map(|s| (s.as_ptr() as usize, s.len()));
                            let eps = cfg.rms_norm_eps;
                            let max_active = cpu::prefill_threads_for_tokens(n);
                            cpu::par_rows_n_active(
                                &mut q_mat[..n * hs],
                                hs,
                                4,
                                max_active,
                                move |(j, _)| unsafe {
                                    let pos = start_pos + j;
                                    let q = core::slice::from_raw_parts_mut(
                                        (q_ptr as *mut f32).add(j * hs),
                                        hs,
                                    );
                                    let k = core::slice::from_raw_parts_mut(
                                        (k_ptr as *mut f32).add(j * kv_dim),
                                        kv_dim,
                                    );
                                    for h in 0..n_heads {
                                        cpu::rmsnorm(
                                            &mut q[h * head_dim..(h + 1) * head_dim],
                                            q_norm,
                                            eps,
                                        );
                                    }
                                    for h in 0..n_kv_heads {
                                        cpu::rmsnorm(
                                            &mut k[h * head_dim..(h + 1) * head_dim],
                                            k_norm,
                                            eps,
                                        );
                                    }
                                    if let Some(theta) = lfm3_rope_theta {
                                        let factors = lfm3_freq_factors_ptr.map(|(ptr, len)| {
                                            core::slice::from_raw_parts(ptr as *const f32, len)
                                        });
                                        cpu::rope_neox_freq_factors(
                                            q,
                                            k,
                                            pos,
                                            (n_heads, n_kv_heads),
                                            head_dim,
                                            theta,
                                            factors,
                                        );
                                    } else {
                                        cpu::rope(
                                            q, k, pos, n_heads, n_kv_heads, head_dim, rope_theta,
                                        );
                                    }
                                },
                            );
                        } else {
                            for j in 0..n {
                                let pos = start_pos + j;
                                let q = &mut q_mat[j * hs..(j + 1) * hs];
                                let k = &mut k_mat[j * kv_dim..(j + 1) * kv_dim];
                                for h in 0..cfg.n_heads {
                                    cpu::rmsnorm(
                                        &mut q[h * head_dim..(h + 1) * head_dim],
                                        q_norm,
                                        cfg.rms_norm_eps,
                                    );
                                }
                                for h in 0..n_kv_heads {
                                    cpu::rmsnorm(
                                        &mut k[h * head_dim..(h + 1) * head_dim],
                                        k_norm,
                                        cfg.rms_norm_eps,
                                    );
                                }
                                match lfm3_attn {
                                    Some(a) => cpu::rope_neox_freq_factors(
                                        q,
                                        k,
                                        pos,
                                        (cfg.n_heads, n_kv_heads),
                                        head_dim,
                                        a.rope_theta,
                                        a.rope_freq_factors.as_deref(),
                                    ),
                                    None => cpu::rope(
                                        q,
                                        k,
                                        pos,
                                        cfg.n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        cfg.rope_theta,
                                    ),
                                }
                            }
                        }

                        for j in 0..n {
                            let k = &k_mat[j * kv_dim..(j + 1) * kv_dim];
                            let v = &v_mat[j * kv_dim..(j + 1) * kv_dim];

                            // Append K, V to cache (f32 or TurboQuant-compressed).
                            if let LayerState::Attention {
                                key_cache,
                                value_cache,
                                key_cache_f16,
                                value_cache_f16,
                                compressed_keys,
                                compressed_values,
                            } = &mut state.layers[layer]
                            {
                                match (will_compress_kv, compressed_keys.as_mut()) {
                                    (true, Some(k_cache_tq)) => {
                                        turboquant::compress_and_append_keys(
                                            k,
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            k_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ if use_f16 => {
                                        key_cache_f16
                                            .extend(k.iter().map(|&x| crate::quant::f32_to_f16(x)));
                                    }
                                    _ => {
                                        key_cache.extend_from_slice(k);
                                    }
                                }
                                match (will_compress_kv, compressed_values.as_mut()) {
                                    (true, Some(v_cache_tq)) => {
                                        turboquant::compress_and_append_values(
                                            v,
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            v_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ if use_f16 => {
                                        value_cache_f16
                                            .extend(v.iter().map(|&x| crate::quant::f32_to_f16(x)));
                                    }
                                    _ => {
                                        value_cache.extend_from_slice(v);
                                    }
                                }
                            }
                        }

                        // ── Pass B: attention ────────────────────────────────
                        let use_tq = will_read_compressed_kv
                            && match &state.layers[layer] {
                                LayerState::Attention {
                                    compressed_keys,
                                    compressed_values,
                                    ..
                                } => compressed_keys.is_some() || compressed_values.is_some(),
                                _ => false,
                            };

                        let sliding_window = lfm3_attn.and_then(|a| a.window);
                        const FLASH_ATTN_THRESHOLD: usize = 16;
                        let use_flash = !use_tq
                            && n >= FLASH_ATTN_THRESHOLD
                            && sliding_window.is_none_or(|w| start_pos + n <= w);

                        if use_f16
                            && let LayerState::Attention {
                                key_cache_f16,
                                value_cache_f16,
                                ..
                            } = &state.layers[layer]
                        {
                            kv_widen_k.clear();
                            kv_widen_k
                                .extend(key_cache_f16.iter().map(|&b| crate::quant::f16_to_f32(b)));
                            kv_widen_v.clear();
                            kv_widen_v.extend(
                                value_cache_f16.iter().map(|&b| crate::quant::f16_to_f32(b)),
                            );
                        }

                        if use_flash {
                            let (k_cache, v_cache) = if use_f16 {
                                (kv_widen_k.as_slice(), kv_widen_v.as_slice())
                            } else {
                                match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        ..
                                    } => (key_cache.as_slice(), value_cache.as_slice()),
                                    _ => unreachable!(),
                                }
                            };
                            let n_heads = cfg.n_heads;
                            let head_chunk = n * head_dim;
                            let flash_buf = &mut flash_out[..n_heads * head_chunk];
                            let t_scores = Instant::now();
                            // Q transpose to head-major columns, row-parallel using 4x4 vector tiles.
                            let max_active = cpu::prefill_threads_for_tokens(n);
                            cpu::transpose_matrix_f32_parallel(
                                &q_mat[..n * hs],
                                &mut q_col[..hs * n],
                                n,
                                hs,
                                max_active,
                            );
                            let q_ref = &q_col[..hs * n];

                            let is_causal =
                                self.is_causal_for(state) && !self.layer_is_non_causal(layer);
                            if !cpu::flash_attention_prefill_items(
                                q_ref, k_cache, v_cache, flash_buf, n_heads, group_size, n, kv_dim,
                                head_dim, scale, start_pos, is_causal, max_active,
                            ) {
                                cpu::par_rows_n_chunked_active(
                                    flash_buf,
                                    head_chunk,
                                    1,
                                    1,
                                    max_active,
                                    |(h, chunk)| {
                                        let kv_h = h / group_size;
                                        cpu::flash_attention_gqa_cpu_opt(
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
                                            is_causal,
                                        );
                                    },
                                );
                            }

                            // Output transpose back to token-major, row-parallel over
                            // tokens (each row gathers one head-block per head).
                            let flash_ptr = flash_buf.as_ptr() as usize;
                            cpu::par_rows_n_active(
                                &mut out_proj_input[..n * hs],
                                hs,
                                64,
                                max_active,
                                move |(j, row)| unsafe {
                                    let src_all = core::slice::from_raw_parts(
                                        flash_ptr as *const f32,
                                        n_heads * n * head_dim,
                                    );
                                    for h in 0..n_heads {
                                        let src_base = h * n * head_dim + j * head_dim;
                                        let dst_base = h * head_dim;
                                        row[dst_base..dst_base + head_dim].copy_from_slice(
                                            &src_all[src_base..src_base + head_dim],
                                        );
                                    }
                                },
                            );
                            if profile_prefill {
                                t_attn_scores += t_scores.elapsed();
                            }
                        } else if use_tq {
                            state.scratch.scores.clear();
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let q = &q_mat[j * hs..(j + 1) * hs];
                                let (ck, cv, k_cache, v_cache) = match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        compressed_keys,
                                        compressed_values,
                                        ..
                                    } => (
                                        compressed_keys.as_ref(),
                                        compressed_values.as_ref(),
                                        key_cache.as_slice(),
                                        value_cache.as_slice(),
                                    ),
                                    _ => unreachable!(),
                                };

                                let use_tq_keys = will_read_compressed_kv && ck.is_some();
                                let use_tq_values = will_read_compressed_kv && cv.is_some();

                                let seq_len = if use_tq_keys {
                                    ck.unwrap().seq_len()
                                } else if use_tq_values {
                                    cv.unwrap().seq_len()
                                } else {
                                    k_cache.len() / kv_dim
                                };
                                let attn_out = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let scores = &mut state.scratch.scores;

                                let rotation = tq_rotation.unwrap();
                                let cfg_tq = tq_config.unwrap();
                                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                                if use_tq_keys {
                                    turboquant::rotate_queries(
                                        q,
                                        cfg.n_heads,
                                        head_dim,
                                        rotation,
                                        qr_scratch,
                                    );
                                }
                                scores.resize(seq_len * group_size, 0.0);
                                for kv_h in 0..n_kv_heads {
                                    let group_start = kv_h * group_size;
                                    let kv_h_offset = kv_h * head_dim;

                                    if use_tq_keys {
                                        turboquant::attn_scores_turboquant_gqa(
                                            ck.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            head_dim,
                                            scale,
                                            seq_len,
                                            cfg_tq,
                                            qr_scratch,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                            let head_scores =
                                                &mut scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_scores(
                                                q_head,
                                                k_cache,
                                                head_scores,
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                scale,
                                                seq_len,
                                            );
                                        }
                                    }

                                    for g in 0..group_size {
                                        let head_scores =
                                            &mut scores[g * seq_len..(g + 1) * seq_len];
                                        cpu::softmax_inplace(head_scores);
                                    }

                                    if use_tq_values {
                                        turboquant::attn_values_turboquant_gqa(
                                            cv.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            attn_out,
                                            head_dim,
                                            seq_len,
                                            rotation,
                                            cfg_tq,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let head_scores =
                                                &scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_values(
                                                head_scores,
                                                v_cache,
                                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                seq_len,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            let (k_cache, v_cache) = if use_f16 {
                                (kv_widen_k.as_slice(), kv_widen_v.as_slice())
                            } else {
                                match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        ..
                                    } => (key_cache.as_slice(), value_cache.as_slice()),
                                    _ => unreachable!(),
                                }
                            };
                            state.scratch.scores.clear();
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let is_causal =
                                    self.is_causal_for(state) && !self.layer_is_non_causal(layer);
                                let seq_len = if is_causal {
                                    (start_pos + j + 1).min(k_cache.len() / kv_dim)
                                } else {
                                    (start_pos + n).min(k_cache.len() / kv_dim)
                                };
                                let win_start =
                                    sliding_window.map_or(0, |w| seq_len.saturating_sub(w));
                                let win_len = seq_len - win_start;
                                let k_win = &k_cache[win_start * kv_dim..seq_len * kv_dim];
                                let v_win = &v_cache[win_start * kv_dim..seq_len * kv_dim];
                                let q = &q_mat[j * hs..(j + 1) * hs];
                                let attn_out = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let scores = &mut state.scratch.scores;
                                scores.resize(win_len, 0.0);
                                for h in 0..cfg.n_heads {
                                    let kv_h = h / group_size;
                                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                    let kv_h_offset = kv_h * head_dim;
                                    cpu::attn_scores(
                                        q_head,
                                        k_win,
                                        scores,
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        scale,
                                        win_len,
                                    );
                                    cpu::softmax_inplace(scores);
                                    cpu::attn_values(
                                        scores,
                                        v_win,
                                        &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        win_len,
                                    );
                                }
                            }
                        }

                        // `lfm3`: scale each head's output by its sigmoid gate before output projection.
                        if lfm3_attn.is_some() {
                            let gate_buf = &ffn_input[..n * cfg.n_heads];
                            for j in 0..n {
                                let tok_out = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let tok_gate = &gate_buf[j * cfg.n_heads..(j + 1) * cfg.n_heads];
                                for (head, &g) in tok_out.chunks_exact_mut(head_dim).zip(tok_gate) {
                                    for x in head {
                                        *x *= g;
                                    }
                                }
                            }
                        }

                        // Phase 3: Batch output projection GEMM
                        #[cfg(not(has_blas))]
                        {
                            let t_q = Instant::now();
                            transformer::quantize_rows(out_proj_input, hs, n, bq_scales, bq_quants);
                            if profile_prefill {
                                t_quant += t_q.elapsed();
                            }
                            let t_ao = Instant::now();
                            transformer::gemm_preq_rowmajor(
                                &self.gguf,
                                attn_output_ref,
                                bq_scales,
                                bq_quants,
                                &mut block_out[..hs * n],
                                n,
                                hs,
                                hs,
                            );
                            if profile_prefill {
                                t_attn_out += t_ao.elapsed();
                            }
                        }
                        #[cfg(has_blas)]
                        {
                            let t_ao = Instant::now();
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                attn_output_ref,
                                out_proj_input,
                                block_out,
                                n,
                                hs,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            if profile_prefill {
                                t_attn_out += t_ao.elapsed();
                            }
                        }

                        // LoRA on the output projection: applied to `block_out`
                        // BEFORE the residual add; input is the attention output
                        // `[hs×n]`.
                        if let Some(lora) = &lora
                            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
                        {
                            crate::lora::apply_prefill_rowmajor(
                                t,
                                out_proj_input,
                                block_out,
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        true
                    } else {
                        false
                    }
                }
            };

            // Fallback: per-token sequential path. Used on x86_64-no-blas
            // (no batched path compiled), and on any target where the
            // batched path saw mixed dtypes and bailed (`used_block_gemm
            // = false`).
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
            let need_block_fallback = !used_block_gemm;
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", has_blas)))]
            let need_block_fallback = true;
            if need_block_fallback {
                // The whole per-token block (projections and attention) is timed as one
                // lump under the conv-core or attention-scores column: `lfm3` attention
                // always lands here, and would otherwise read as free in the profile.
                let t_fallback = profile_prefill.then(Instant::now);
                block_out.fill(0.0);
                if !is_conv && self.layer_is_non_causal(layer) && n > 1 {
                    self.forward_attn_block_bidir_batch(
                        layer, normed, start_pos, n, block_out, state,
                    );
                } else {
                    for j in 0..n {
                        col.copy_from_slice(&normed[j * hs..(j + 1) * hs]);
                        #[cfg(target_arch = "aarch64")]
                        Self::quantize_to_scratch(col, state);

                        if is_conv {
                            self.forward_conv_block(layer, col, start_pos + j, state);
                        } else {
                            self.forward_attn_block(layer, col, start_pos + j, state);
                        }

                        block_out[j * hs..(j + 1) * hs].copy_from_slice(&state.scratch.out[..hs]);
                    }
                }
                if let Some(t) = t_fallback {
                    if is_conv {
                        t_conv_core += t.elapsed();
                    } else {
                        t_attn_scores += t.elapsed();
                    }
                }
            }

            // Log the BLOCK OUTPUT magnitude (pre-residual) so we
            // can see how much each layer contributes vs the prior
            // residual. Together with the post-block log this lets
            // us identify whether the magnitude growth is from a
            // misbehaving block, an oversize residual, or both.
            // The `if debug_hidden` guard keeps the per-layer
            // `format!` allocations off the hot path when the env
            // var isn't set.
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(
                    &format!("layer {layer} ({block_kind}) block-out"),
                    block_out,
                );
            }

            // `lfm3` sandwich norm: the block output is normalized before the residual add.
            if let Some(norm_w) = self
                .lfm3
                .as_ref()
                .and_then(|l3| l3.layers[layer].post_operator_norm.as_ref())
            {
                block_out[..n * hs]
                    .chunks_exact_mut(hs)
                    .for_each(|row| cpu::rmsnorm(row, norm_w, cfg.rms_norm_eps));
            }

            let t_res1 = Instant::now();
            // Residual: hidden += block_out
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let total = hs * n;
                let n_chunks = total / 4;
                let h_ptr = hidden.as_mut_ptr();
                let bo_ptr = block_out.as_ptr();
                for i in 0..n_chunks {
                    let h_v = vld1q_f32(h_ptr.add(i * 4));
                    let bo_v = vld1q_f32(bo_ptr.add(i * 4));
                    vst1q_f32(h_ptr.add(i * 4), vaddq_f32(h_v, bo_v));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for i in 0..hs * n {
                hidden[i] += block_out[i];
            }
            if profile_prefill {
                t_residuals += t_res1.elapsed();
            }
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(&format!("layer {layer} ({block_kind}) post-block"), hidden);
            }

            let t_fn = Instant::now();
            // FFN pre-norm each row
            let w_ffn = &self.ffn_norm_weights[layer];
            let eps = cfg.rms_norm_eps;
            if n >= 512 {
                let hidden_ptr = hidden.as_ptr() as usize;
                cpu::par_rows_n(&mut ffn_input[..n * hs], hs, 64, move |(j, row)| unsafe {
                    let tok_h =
                        core::slice::from_raw_parts((hidden_ptr as *const f32).add(j * hs), hs);
                    cpu::rmsnorm_into(tok_h, row, w_ffn, eps);
                });
            } else {
                for j in 0..n {
                    let tok_h = &hidden[j * hs..(j + 1) * hs];
                    let tok_in = &mut ffn_input[j * hs..(j + 1) * hs];
                    cpu::rmsnorm_into(tok_h, tok_in, w_ffn, eps);
                }
            }
            if profile_prefill {
                t_ffn_norm += t_fn.elapsed();
            }

            // FFN: batched GEMM (reads weights once for all n tokens) for the
            // dtypes `batched_gemm_supports` admits. Available on aarch64 (NEON
            // `gemm_preq`), on x86_64 with runtime avx2+fma (int8
            // `gemm_preq`), and on any target with `feature = "blas"` (BLAS
            // SGEMM via `try_blas_prefill_gemm`). Require all three projections
            // (gate/up/down) to be batchable — a mixed-dtype FFN block would
            // leave later matrices silently uncomputed in the batched path and
            // produce wrong outputs.
            let refs = &self.layer_refs[layer];
            // A MoE layer has no single FFN weight to batch: each token picks
            // its own experts, so the dense path below does not apply. Routed
            // per token: correct, but not yet grouped by expert (plan
            // 000312-01, phase 1 step 6).
            'dense_ffn: {
                let dense = match &refs.ffn {
                    FfnRefs::Dense(d) => d,
                    FfnRefs::Moe(moe) => {
                        self.prefill_moe_ffn(layer, moe, hs, n, ffn_input, ffn_out, col, state);
                        break 'dense_ffn;
                    }
                };
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
                let used_gemm = if [
                    ("ffn_gate", &dense.gate),
                    ("ffn_up", &dense.up),
                    ("ffn_down", &dense.down),
                ]
                .into_iter()
                .fold(true, |ok, (name, w)| {
                    if transformer::batched_gemm_supports(w.dtype, w.k) {
                        ok
                    } else {
                        transformer::warn_unbatchable(name, w.dtype);
                        false
                    }
                }) {
                    // Pre-quantize all n columns to Q8_0: only needed for the NEON fallback.
                    #[cfg(not(has_blas))]
                    {
                        let t_q = Instant::now();
                        transformer::quantize_rows(ffn_input, hs, n, bq_scales, bq_quants);
                        if profile_prefill {
                            t_quant += t_q.elapsed();
                        }
                    }

                    // Gate + Up via Single-Dispatch Fused AMX GEMM (AMX Gate + Up)
                    #[cfg(has_blas)]
                    {
                        let t_gu = Instant::now();
                        let mut gu_rows = vec![0.0f32; 2 * is * hs];
                        let g_data = transformer::weight_data(&self.gguf, &dense.gate);
                        let u_data = transformer::weight_data(&self.gguf, &dense.up);
                        if let (Some(dq_g), Some(dq_u)) = (
                            transformer::blas_dequantizer(dense.gate.dtype),
                            transformer::blas_dequantizer(dense.up.dtype),
                        ) {
                            dq_g(g_data, is, hs, &mut gu_rows[..is * hs]);
                            dq_u(u_data, is, hs, &mut gu_rows[is * hs..]);
                        }

                        crate::backend::blas::sgemm_rowmajor_nt_parallel(
                            n,
                            2 * is,
                            hs,
                            ffn_input,
                            &gu_rows,
                            &mut gate_up_mat[..2 * is * n],
                        );

                        // LoRA on gate/up. The fused GEMM leaves each token's gate then up
                        // in one `2 * is` row, so the deltas go on the two halves.
                        if let Some(lora) = &lora {
                            [
                                (crate::lora::LoraTarget::FfnGate, 0),
                                (crate::lora::LoraTarget::FfnUp, is),
                            ]
                            .into_iter()
                            .filter_map(|(target, off)| lora.get(layer, target).map(|t| (t, off)))
                            .for_each(|(t, off)| {
                                ffn_input[..n * hs]
                                    .chunks_exact(hs)
                                    .zip(gate_up_mat[..2 * is * n].chunks_exact_mut(2 * is))
                                    .for_each(|(x, row)| {
                                        crate::lora::apply_decode(
                                            t,
                                            x,
                                            &mut row[off..off + is],
                                            &mut state.scratch.lora_tmp,
                                        )
                                    });
                            });
                        }

                        for j in 0..n {
                            let (g_slice, u_slice) =
                                gate_up_mat[j * 2 * is..(j + 1) * 2 * is].split_at_mut(is);
                            cpu::silu_mul_inplace(g_slice, u_slice);
                        }

                        if profile_prefill {
                            t_ffn_gate_up += t_gu.elapsed();
                        }
                    }
                    #[cfg(not(has_blas))]
                    {
                        let t_gu = Instant::now();
                        let fused_ok = lora.is_none()
                            && transformer::try_repacked_gate_up_silu_rowmajor(
                                &dense.gate,
                                &dense.up,
                                bq_scales,
                                bq_quants,
                                &mut gate_mat[..is * n],
                                n,
                                is,
                                hs,
                            );
                        if !fused_ok {
                            transformer::gemm_preq_rowmajor(
                                &self.gguf,
                                &dense.gate,
                                bq_scales,
                                bq_quants,
                                &mut gate_mat[..is * n],
                                n,
                                is,
                                hs,
                            );
                            transformer::gemm_preq_rowmajor(
                                &self.gguf,
                                &dense.up,
                                bq_scales,
                                bq_quants,
                                &mut up_mat[..is * n],
                                n,
                                is,
                                hs,
                            );
                            if let Some(lora) = &lora {
                                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                                    crate::lora::apply_prefill_rowmajor(
                                        t,
                                        ffn_input,
                                        &mut gate_mat[..is * n],
                                        n,
                                        &mut state.scratch.lora_tmp,
                                    );
                                }
                                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                                    crate::lora::apply_prefill_rowmajor(
                                        t,
                                        ffn_input,
                                        &mut up_mat[..is * n],
                                        n,
                                        &mut state.scratch.lora_tmp,
                                    );
                                }
                            }
                            // Elementwise over the whole `[n x is]` block: one pool dispatch, not
                            // one 9-row dispatch per token (which ran on a single worker).
                            cpu::silu_mul_inplace(&mut gate_mat[..n * is], &up_mat[..n * is]);
                        }
                        if profile_prefill {
                            t_ffn_gate_up += t_gu.elapsed();
                        }
                    }

                    // Re-quantize gate_mat columns for down projection: only needed for NEON fallback.
                    #[cfg(not(has_blas))]
                    {
                        let t_q = Instant::now();
                        transformer::quantize_rows(gate_mat, is, n, dq_scales, dq_quants);
                        if profile_prefill {
                            t_quant += t_q.elapsed();
                        }
                    }

                    // Down via batched GEMM
                    #[cfg(has_blas)]
                    {
                        let t_d = Instant::now();
                        let mut down_rows = vec![0.0f32; hs * is];
                        let d_data = transformer::weight_data(&self.gguf, &dense.down);
                        if let Some(dq_d) = transformer::blas_dequantizer(dense.down.dtype) {
                            dq_d(d_data, hs, is, &mut down_rows);
                        }

                        unsafe {
                            crate::backend::blas::sgemm_rowmajor_nt_ld_parallel(
                                n,
                                hs,
                                is,
                                gate_up_mat.as_ptr(),
                                2 * is,
                                down_rows.as_ptr(),
                                is,
                                ffn_out.as_mut_ptr(),
                                hs,
                            );
                        }
                        if profile_prefill {
                            t_ffn_down += t_d.elapsed();
                        }
                    }
                    #[cfg(not(has_blas))]
                    {
                        let t_d = Instant::now();
                        transformer::gemm_preq_rowmajor(
                            &self.gguf,
                            &dense.down,
                            dq_scales,
                            dq_quants,
                            &mut ffn_out[..hs * n],
                            n,
                            hs,
                            is,
                        );
                        if profile_prefill {
                            t_ffn_down += t_d.elapsed();
                        }
                    }

                    // LoRA on the down projection: applied to `ffn_out` BEFORE the
                    // residual add; input is the SiLU⊙up product, in `gate_mat` `[n×is]`
                    // (or, under BLAS, the gate half of each `gate_up_mat` row).
                    if let Some(lora) = &lora
                        && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
                    {
                        #[cfg(not(has_blas))]
                        crate::lora::apply_prefill_rowmajor(
                            t,
                            &gate_mat[..is * n],
                            ffn_out,
                            n,
                            &mut state.scratch.lora_tmp,
                        );
                        #[cfg(has_blas)]
                        gate_up_mat[..2 * is * n]
                            .chunks_exact(2 * is)
                            .zip(ffn_out[..hs * n].chunks_exact_mut(hs))
                            .for_each(|(row, y)| {
                                crate::lora::apply_decode(
                                    t,
                                    &row[..is],
                                    y,
                                    &mut state.scratch.lora_tmp,
                                )
                            });
                    }
                    true
                } else {
                    false
                };

                // Fallback: per-token GEMV. Used on x86_64-no-blas (no batched
                // path compiled), and on any target where the FFN weights
                // weren't all batchable (`used_gemm = false`).
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
                let need_fallback = !used_gemm;
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", has_blas)))]
                let need_fallback = true;
                if need_fallback {
                    ffn_out.fill(0.0);
                    for j in 0..n {
                        col.copy_from_slice(&ffn_input[j * hs..(j + 1) * hs]);

                        #[cfg(target_arch = "aarch64")]
                        {
                            Self::quantize_to_scratch(col, state);
                            self.gemv_preq(
                                &dense.gate,
                                col,
                                &state.scratch.q8_scales,
                                &state.scratch.q8_quants,
                                gate_col,
                            );
                            self.gemv_preq(
                                &dense.up,
                                col,
                                &state.scratch.q8_scales,
                                &state.scratch.q8_quants,
                                up_col,
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            self.gemv(&dense.gate, col, gate_col);
                            self.gemv(&dense.up, col, up_col);
                        }

                        // LoRA on gate/up (per-token decode hook): this fallback loop
                        // doesn't route through `forward_ffn_block`, so apply it here.
                        if let Some(lora) = &lora {
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                                crate::lora::apply_decode(
                                    t,
                                    col,
                                    gate_col,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                                crate::lora::apply_decode(
                                    t,
                                    col,
                                    up_col,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                        }

                        cpu::silu_mul_inplace(gate_col, up_col);

                        #[cfg(target_arch = "aarch64")]
                        {
                            Self::quantize_to_scratch(gate_col, state);
                            self.gemv_preq(
                                &dense.down,
                                gate_col,
                                &state.scratch.q8_scales,
                                &state.scratch.q8_quants,
                                out_col,
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        self.gemv(&dense.down, gate_col, out_col);

                        // LoRA on the down projection (per-token decode hook): input is
                        // the SiLU⊙up product in `gate_col`.
                        if let Some(lora) = &lora
                            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
                        {
                            crate::lora::apply_decode(
                                t,
                                gate_col,
                                out_col,
                                &mut state.scratch.lora_tmp,
                            );
                        }

                        ffn_out[j * hs..(j + 1) * hs].copy_from_slice(&out_col[..hs]);
                    }
                }
            }

            if debug_hidden {
                log_rms(&format!("layer {layer} ffn-out"), ffn_out);
            }
            // `lfm3` sandwich norm on the FFN output, before its residual add.
            if let Some(norm_w) = self
                .lfm3
                .as_ref()
                .and_then(|l3| l3.layers[layer].post_ffn_norm.as_ref())
            {
                ffn_out[..n * hs]
                    .chunks_exact_mut(hs)
                    .for_each(|row| cpu::rmsnorm(row, norm_w, cfg.rms_norm_eps));
            }
            let t_res2 = Instant::now();
            // Second residual: hidden += ffn_out
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let total = hs * n;
                let n_chunks = total / 4;
                let h_ptr = hidden.as_mut_ptr();
                let fo_ptr = ffn_out.as_ptr();
                for i in 0..n_chunks {
                    let h_v = vld1q_f32(h_ptr.add(i * 4));
                    let fo_v = vld1q_f32(fo_ptr.add(i * 4));
                    vst1q_f32(h_ptr.add(i * 4), vaddq_f32(h_v, fo_v));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for i in 0..hs * n {
                hidden[i] += ffn_out[i];
            }
            if profile_prefill {
                t_residuals += t_res2.elapsed();
            }
            if debug_hidden {
                log_rms(&format!("layer {layer} post-ffn"), hidden);
            }
        }

        if profile_prefill {
            eprintln!(
                "[PROFILE PREFILL] n={n} | norm: {:.2}ms | conv_in: {:.2}ms | conv_core: {:.2}ms | conv_out: {:.2}ms | attn_qkv: {:.2}ms | attn_out: {:.2}ms | ffn_norm: {:.2}ms | ffn_gate_up: {:.2}ms | ffn_down: {:.2}ms | residuals: {:.2}ms | quant: {:.2}ms | attn_scores: {:.2}ms",
                t_norm.as_secs_f64() * 1000.0,
                t_conv_in.as_secs_f64() * 1000.0,
                t_conv_core.as_secs_f64() * 1000.0,
                t_conv_out.as_secs_f64() * 1000.0,
                t_attn_qkv.as_secs_f64() * 1000.0,
                t_attn_out.as_secs_f64() * 1000.0,
                t_ffn_norm.as_secs_f64() * 1000.0,
                t_ffn_gate_up.as_secs_f64() * 1000.0,
                t_ffn_down.as_secs_f64() * 1000.0,
                t_residuals.as_secs_f64() * 1000.0,
                t_quant.as_secs_f64() * 1000.0,
                t_attn_scores.as_secs_f64() * 1000.0,
            );
        }

        // seq_len tracks total tokens processed. The conv/attn blocks handle
        // per-token KV cache growth internally. We need seq_len = start_pos + n
        // at the end for the decode phase to continue from the right position.
        // Note: seq_len was NOT incremented inside the block functions: only
        // the single-token forward() does that. So set it here:
        state.seq_len = start_pos + n;
        state.prefill_scratch = scratch;
    }

    /// Run the prefill layer loop and project logits for the final (n - 1) token.
    fn prefill_layers_and_logits(
        &self,
        mut hidden: Vec<f32>,
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        self.prefill_layers_loop(&mut hidden, n, start_pos, state);

        // Extract last token, apply output norm + projection
        let mut last_hidden = vec![0.0f32; hs];
        last_hidden.copy_from_slice(&hidden[(n - 1) * hs..n * hs]);
        cpu::rmsnorm(&mut last_hidden, &self.output_norm_weight, cfg.rms_norm_eps);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&last_hidden, state);
            self.gemv_preq(
                &self.logits_ref,
                &last_hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.logits_ref, &last_hidden, &mut logits);

        logits
    }

    /// Batched projection of post-RMSnorm row-major activations `[n × hs]`
    /// against `token_embd.weight`, yielding row-major `[n × vocab_size]` logits.
    /// Returns `None` if the embedding weight dtype isn't supported by batched
    /// GEMM. Available on aarch64 (NEON `gemm_preq`) and on x86_64 with
    /// runtime avx2+fma (int8 `gemm_preq`).
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
    fn project_logits_batched(&self, normed_hidden: &[f32], n: usize) -> Option<Vec<f32>> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        if !transformer::batched_gemm_supports(self.logits_ref.dtype, hs) {
            return None;
        }

        let mut b_scales = vec![0.0f32; n * (hs / 32)];
        let mut b_quants = vec![0i8; n * hs];
        transformer::quantize_rows(normed_hidden, hs, n, &mut b_scales, &mut b_quants);

        let rows = self.logits_ref.m;
        let mut out = vec![0.0f32; rows * n];
        if !transformer::gemm_preq(
            &self.gguf,
            &self.logits_ref,
            &b_scales,
            &b_quants,
            &mut out,
            rows,
            n,
            hs,
        ) {
            return None;
        }

        let mut logits = vec![0.0f32; n * vocab];
        transformer::gemm_out_to_rows(&out, rows, n, vocab, &mut logits);
        Some(logits)
    }

    /// Run the prefill layer loop and project logits for ALL n tokens.
    fn prefill_layers_and_all_logits(
        &self,
        mut hidden: Vec<f32>,
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        self.prefill_layers_loop(&mut hidden, n, start_pos, state);

        for j in 0..n {
            cpu::rmsnorm(
                &mut hidden[j * hs..(j + 1) * hs],
                &self.output_norm_weight,
                cfg.rms_norm_eps,
            );
        }

        #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
        if let Some(logits) = self.project_logits_batched(&hidden, n) {
            return logits;
        }

        let mut all_logits = vec![0.0f32; n * cfg.vocab_size];
        for j in 0..n {
            let tok_h = &hidden[j * hs..(j + 1) * hs];
            let tok_l = &mut all_logits[j * cfg.vocab_size..(j + 1) * cfg.vocab_size];
            #[cfg(target_arch = "aarch64")]
            {
                Self::quantize_to_scratch(tok_h, state);
                self.gemv_preq(
                    &self.logits_ref,
                    tok_h,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    tok_l,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            self.gemv(&self.logits_ref, tok_h, tok_l);
        }
        all_logits
    }

    /// Lock-free body of `Model::forward_prefill` — does the actual
    /// embed + layer loop without consulting the prefix cache.
    /// `forward_prefill` wraps this with cache lookup/insert; cache
    /// hits bypass embedding the prefix tokens entirely and re-enter
    /// here with `start_pos = prefix_len` to prefill only the suffix.
    pub(crate) fn forward_prefill_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        assert!(
            !tokens.is_empty(),
            "forward_prefill_inner requires at least one token"
        );
        #[cfg(test)]
        PREFILLED_TOKENS.with(|count| count.set(count.get() + n));

        let mut hidden = vec![0.0f32; hs * n];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden[j * hs..(j + 1) * hs]);
        }

        self.prefill_layers_and_logits(hidden, n, start_pos, state)
    }

    /// Largest `forward_prefill_logits_all` batch treated as a speculative
    /// verify that releases a large earlier prefill working set.
    const VERIFY_RELEASE_MAX_TOKENS: usize = 16;

    /// Multi-token prefill producing all `[n × vocab_size]` logits (used for speculative verification).
    pub(crate) fn forward_prefill_logits_all_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        assert!(
            !tokens.is_empty(),
            "forward_prefill_logits_all_inner requires at least one token"
        );
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill_logits_all: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );

        // A small batch is a speculative verify: it never goes through
        // `run_layers`, so a large earlier prefill working set would be pinned
        // for as long as decoding stays verify-only. Same release rule as the
        // first decode token; it only fires above the threshold.
        if n <= Self::VERIFY_RELEASE_MAX_TOKENS {
            state.prefill_scratch.release_if_large();
        }

        let mut hidden = vec![0.0f32; hs * n];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden[j * hs..(j + 1) * hs]);
        }

        self.prefill_layers_and_all_logits(hidden, n, start_pos, state)
    }
}

impl Model for LfmModel {
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

    fn supports_all_logits(&self) -> bool {
        true
    }

    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.forward_prefill_logits_all_inner(tokens, start_pos, state)
    }

    fn supports_hidden_states(&self) -> bool {
        true
    }

    /// Only a bidirectional model reads a media prefix this way.
    fn supports_media_prefix(&self) -> bool {
        !self.config.is_causal
    }

    /// The CPU extractor branches on the config: full-context prefill for a
    /// bidirectional model, a causal loop otherwise.
    fn supports_bidirectional_hidden_states(&self) -> bool {
        !self.config.is_causal
    }

    fn hidden_states_with_prefix(
        &self,
        prefix: &[f32],
        tokens: &[u32],
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        let hs = self.config.hidden_size;
        assert!(
            !self.config.is_causal,
            "hidden_states_with_prefix needs a bidirectional model"
        );
        assert!(
            prefix.len().is_multiple_of(hs),
            "the media prefix is not a whole number of rows"
        );
        if prefix.is_empty() && tokens.is_empty() {
            return Vec::new();
        }
        let prefix_rows = prefix.len() / hs;
        let mut h = Vec::with_capacity(prefix.len() + tokens.len() * hs);
        h.extend_from_slice(prefix);
        h.resize(prefix.len() + tokens.len() * hs, 0.0);
        for (i, &token) in tokens.iter().enumerate() {
            let row = prefix.len() + i * hs;
            assert!(
                (token as usize) < self.config.vocab_size,
                "token_id {token} out of range (vocab_size={})",
                self.config.vocab_size
            );
            self.dequantize_row_into(&self.embd_ref, token as usize, &mut h[row..row + hs]);
        }
        let mut out = self.float_trunk(h, prefix_rows);
        out.drain(..prefix.len());
        out
    }

    /// The CPU path is the one backend with routed-FFN LoRA hooks: the router
    /// delta feeds `select_experts` and the per-expert factors are applied to
    /// the selected expert's projections, both pinned by `moe_lora_parity`.
    fn supports_moe_lora(&self) -> bool {
        true
    }

    /// Dense-target LoRA hooks in the attention, FFN, and gated-conv blocks.
    /// Not `lfm3`: its sandwich norms and attention gate sit between the hooked
    /// projections, and no adapter has been validated against them.
    fn supports_lora(&self) -> bool {
        self.lfm3.is_none()
    }

    fn f16_kv_supported(&self) -> bool {
        true
    }

    /// Per-token post-final-norm hidden states, row-major `[n * hidden_size]`.
    /// Mirrors [`Self::forward`]'s embedding path (dequantize → `run_layers`,
    /// which applies the output norm) minus the logit projection — so the result
    /// is the same post-`output_norm` vector, matching llama.cpp `--pooling none`.
    /// Per-token (not batched): LFM2's batched prefill is entangled with the
    /// prefix cache, and this stateless one-shot path must not touch it; batched
    /// capture is a possible perf follow-up. `state` must start cleared at pos 0.
    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "hidden_states requires at least one token"
        );
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        let is_causal = self.is_causal_for(state) && self.config.is_causal;
        if !is_causal {
            let n = tokens.len();
            let mut hidden = vec![0.0f32; n * hs];
            for (i, &token) in tokens.iter().enumerate() {
                let token_id = token as usize;
                assert!(
                    token_id < cfg.vocab_size,
                    "token_id {token_id} out of range (vocab_size={})",
                    cfg.vocab_size
                );
                self.dequantize_row_into(
                    &self.embd_ref,
                    token_id,
                    &mut hidden[i * hs..(i + 1) * hs],
                );
            }
            // F16 and F32 projections cannot take the batched path, and the per-token path
            // that would take them is causal: run the trunk in plain f32 instead, handing
            // it the embedded rows rather than embedding them a second time
            if self.lfm3.is_none() && state.lora.is_none() && self.bidirectional_needs_float_path()
            {
                return self.hidden_states_float_bidirectional(hidden);
            }
            self.prefill_layers_loop(&mut hidden, n, 0, state);
            for j in 0..n {
                cpu::rmsnorm(
                    &mut hidden[j * hs..(j + 1) * hs],
                    &self.output_norm_weight,
                    cfg.rms_norm_eps,
                );
            }
            return hidden;
        }

        let mut out = Vec::with_capacity(tokens.len() * hs);
        // Reuse one embedding buffer across tokens instead of a per-token Vec.
        let mut hidden = vec![0.0f32; hs];
        for &token in tokens {
            let token_id = token as usize;
            assert!(
                token_id < cfg.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden);
            // `run_layers` ropes at `pos == seq_len` and appends one cell,
            // bumping seq_len; a cleared state walks positions 0..n.
            let pos = state.seq_len;
            self.run_layers(&mut hidden, pos, state);
            out.extend_from_slice(&hidden);
        }
        out
    }

    fn is_classifier(&self) -> bool {
        self.classifier_weight.is_some()
    }

    fn num_classes(&self) -> usize {
        self.classifier_weight
            .as_ref()
            .map(|w| w.len() / self.config.hidden_size)
            .unwrap_or(0)
    }

    fn class_labels(&self) -> &[String] {
        &self.config.class_labels
    }

    fn classify_tokens(
        &self,
        tokens: &[u32],
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, crate::CeraError> {
        if tokens.len() > self.config.max_seq_len {
            return Err(crate::CeraError::Backend(format!(
                "sequence length {} exceeds maximum model sequence limit {}",
                tokens.len(),
                self.config.max_seq_len
            )));
        }

        let hidden = self.hidden_states(tokens, state);

        let (classifier_w, classifier_b) = if let Some(ref lora) = state.lora {
            if let Some(ref w) = lora.classifier_weight {
                (w.as_slice(), lora.classifier_bias.as_deref())
            } else if let Some(ref w) = self.classifier_weight {
                (w.as_slice(), self.classifier_bias.as_deref())
            } else {
                return Err(crate::CeraError::Backend(
                    "neither model nor attached LoRA adapter has a classifier head".into(),
                ));
            }
        } else if let Some(ref w) = self.classifier_weight {
            (w.as_slice(), self.classifier_bias.as_deref())
        } else {
            return Err(crate::CeraError::Backend(
                "model has no classifier head".into(),
            ));
        };
        let hs = self.config.hidden_size;
        if hs == 0 || !classifier_w.len().is_multiple_of(hs) {
            return Err(crate::CeraError::Backend(format!(
                "classifier.weight length {} is not a multiple of hidden_size {}",
                classifier_w.len(),
                hs
            )));
        }
        let num_classes = classifier_w.len() / hs;
        if let Some(bias) = classifier_b
            && bias.len() != num_classes
        {
            return Err(crate::CeraError::Backend(format!(
                "classifier.bias length {} does not match num_classes {}",
                bias.len(),
                num_classes
            )));
        }
        let n = tokens.len();

        let mut logits = vec![0.0f32; n * num_classes];
        for j in 0..n {
            let h = &hidden[j * hs..(j + 1) * hs];
            let out_slice = &mut logits[j * num_classes..(j + 1) * num_classes];
            for c in 0..num_classes {
                let w_row = &classifier_w[c * hs..(c + 1) * hs];
                assert_eq!(h.len(), hs);
                assert_eq!(w_row.len(), hs);
                let mut dot = crate::backend::cpu::dot_f32(h, w_row);
                if let Some(bias) = classifier_b {
                    dot += bias[c];
                }
                out_slice[c] = dot;
            }
        }
        Ok(logits)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LFM2 forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        // 1. Embedding lookup → layers → output norm
        let mut hidden_stack = [0.0f32; 4096];
        let mut hidden_heap;
        let hidden = if cfg.hidden_size <= 4096 {
            &mut hidden_stack[..cfg.hidden_size]
        } else {
            hidden_heap = vec![0.0f32; cfg.hidden_size];
            &mut hidden_heap[..]
        };
        self.dequantize_row_into(&self.embd_ref, token_id, hidden);
        if self.lfm3.is_some() && transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("model.embed_tokens", hidden);
        }
        self.run_layers(hidden, pos, state);

        // 2. Output projection (the tied embedding, or an lfm3 file's output.weight)
        if state.scratch.logits.len() < cfg.vocab_size {
            state.scratch.logits.resize(cfg.vocab_size, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        self.gemv_preq(
            &self.logits_ref,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.logits[..cfg.vocab_size],
        );
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(
            &self.logits_ref,
            hidden,
            &mut state.scratch.logits[..cfg.vocab_size],
        );
        if self.lfm3.is_some() && transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record(
                "result_output",
                &state.scratch.logits[..cfg.vocab_size],
            );
        }

        state.scratch.logits[..cfg.vocab_size].to_vec()
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
                    self.dequantize_row_into(&self.embd_ref, token_id, hidden);
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
        self.dequantize_row_into(&self.embd_ref, token_id, hidden);
        self.run_layers(hidden, pos, state);

        #[cfg(target_arch = "aarch64")]
        if self.logits_ref.dtype == DType::Q6K && self.config.scalars.logit > 0.0 {
            let profile = Self::profile_decode_enabled();
            let t_lm = std::time::Instant::now();
            let res = transformer::gemv_preq_argmax(
                &self.gguf,
                &self.logits_ref,
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
            ) as u32;
            if profile {
                eprintln!(
                    "[PROFILE DECODE] lm_head (argmax): {:.2}ms",
                    t_lm.elapsed().as_secs_f64() * 1000.0
                );
            }
            return res;
        }

        if state.scratch.logits.len() < cfg.vocab_size {
            state.scratch.logits.resize(cfg.vocab_size, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        self.gemv_preq(
            &self.logits_ref,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.logits[..cfg.vocab_size],
        );
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(
            &self.logits_ref,
            hidden,
            &mut state.scratch.logits[..cfg.vocab_size],
        );

        crate::sampler::argmax(&state.scratch.logits[..cfg.vocab_size])
    }

    fn supports_embedding_input(&self) -> bool {
        true
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);

        // Output projection (the tied embedding, or an lfm3 file's output.weight)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&hidden, state);
            self.gemv_preq(
                &self.logits_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.logits_ref, &hidden, &mut logits);

        logits
    }

    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert_eq!(tokens.len(), 1);
        let token_id = tokens[0] as usize;
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
    }

    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
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

        // Cache participation gate: only on a fresh prefill
        // (`start_pos == 0`). Continuation prefills (chunked /
        // mid-sequence) carry KV state from the prior chunk so a
        // cache restore would clobber it. TurboQuant-compressed
        // states are now supported via `LayerSnapshot::AttentionCompressed`
        // (the `!is_compressed()` exclusion was lifted in this PR).
        //
        // An active LoRA also disables the prefix cache: the batched
        // `forward_prefill_inner` still runs and applies the adapter in-batch (via
        // `apply_prefill`), but the resulting KV is adapter-specific, and the cache
        // key doesn't include the adapter — caching it would let a later base-model
        // (or different-adapter) prefill restore adapted KV. Skip both the lookup
        // and the insert while LoRA is attached.
        let cache_eligible = start_pos == 0 && state.lora.is_none();

        if cache_eligible {
            let hit = self
                .prefix_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .find_longest_prefix(tokens);
            if let Some((snapshot, prefix_len)) = hit {
                // Compatibility gate: snapshot's KV representation
                // must match the live state's. Cross-mode restores
                // would panic or silently corrupt in
                // `InferenceState::restore` (compressed snapshot into
                // `None` slots, an f16 snapshot's u16 bytes decoded as
                // f32, or an uncompressed snapshot into a TurboQuant-
                // configured state with mismatched scratch / rotation
                // shape). Four live-state modes:
                //
                // - fully uncompressed f32 → match `Attention` snapshots.
                // - fully compressed       → match `AttentionCompressed`.
                // - f16                    → match `AttentionF16`.
                // - mixed-mode (one side compressed, the other not):
                //   `snapshot()` returns `None` so the cache never
                //   holds an entry that matches; every branch rejects.
                //
                // `state.is_compressed()` (any-side-compressed) is
                // too loose for the compressed branch: a mixed-mode
                // state would erroneously match a fully-compressed
                // snapshot and panic in `restore`. Use
                // `is_fully_compressed` for the compressed branch,
                // `kv_f16` for the f16 branch, and `!is_compressed &&
                // !kv_f16` for the uncompressed f32 branch — the last
                // guards against an f32 snapshot being restored into an
                // f16 state (whose byte widths differ). Today
                // `model_fingerprint` doesn't include the compression
                // flags, so a `--cache-dir` shared between TurboQuant,
                // f16, and uncompressed runs of the same model file
                // relies on this gate; v2 could fold the KV mode into
                // the fingerprint.
                let compatible = if snapshot.is_compressed() {
                    state.is_fully_compressed() && state.is_compressed()
                } else if snapshot.is_f16() {
                    state.kv_f16
                } else {
                    !state.is_compressed() && !state.kv_f16
                };
                if !compatible {
                    // skip; fall through to cold prefill.
                } else if prefix_len < tokens.len() && prefix_len > 0 {
                    // Strict-prefix-only: a `prefix_len == tokens.len()`
                    // hit would force `use_len = tokens.len() - 1`, but
                    // the restored conv rolling buffer reflects "after
                    // all tokens" — re-running the last token would
                    // advance the conv buffer one position past where
                    // it should be (conv layers don't gate on
                    // seq_len). Skip full hits + fall through to cold
                    // prefill. Same fix wgpu got in PR #120; tracked
                    // as 8f on the punch list.
                    let use_len = prefix_len;
                    state.restore(&snapshot);
                    let logits = self.forward_prefill_inner(&tokens[use_len..], use_len, state);
                    if let Some(snap) = state.snapshot() {
                        self.prefix_cache
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(tokens, snap);
                    }
                    return logits;
                }
            }
        }

        let logits = self.forward_prefill_inner(tokens, start_pos, state);
        if cache_eligible && let Some(snap) = state.snapshot() {
            self.prefix_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(tokens, snap);
        }
        logits
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        // Content identity is independent of cache/compression state. Resolve
        // it before locking so a first scan does not block other sessions.
        let model_id = self.cache_model_id(config.cache_dir.is_some());
        // Use the same lock order as compression configuration. The tag must
        // remain stable while resolving the identity and replacing the cache.
        let mut cache = self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
        let tag = self.kv_cache_tag.lock().unwrap_or_else(|e| e.into_inner());
        let id = Self::namespace_for(tag.as_deref().unwrap_or(""), model_id);
        *cache = KvPrefixCache::for_model(config, &self.config, &self.model_id, &id);
    }

    fn clear_warm_cache(&self) {
        self.prefix_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear_warm();
    }

    fn clear_cache(&self) {
        self.prefix_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// The CPU backend allocates its KV from `InferenceState`, so there is nothing
    /// to build here — but the prefix cache's namespace still has to reflect the
    /// mode, or snapshots from different KV modes collide on disk. See
    /// [`KvCompression::cache_tag`] for what that collision costs.
    ///
    /// **First call wins**, matching the trait contract and the GPU backends: a
    /// later call with a *different* mode returns
    /// [`crate::CeraError::KvCompressionConflict`].
    ///
    /// That restriction is not about allocation here — nothing is sized by the mode
    /// on CPU. It is because the tag lives on the *model* while `KvCompression` is a
    /// *per-session* knob, and one model owns one prefix cache. Letting a second
    /// session re-tag would leave the first still writing snapshots, now into the
    /// second's namespace. For two TurboQuant seeds that is silent corruption rather
    /// than a miss: the compatibility gate only checks that both sides are
    /// compressed, and `InferenceState::restore` never validates the seed, so the
    /// first session's KV would be decoded in the wrong Hadamard basis. Two sessions
    /// in different KV modes need two model instances.
    fn configure_kv_compression(
        &self,
        compression: &KvCompression,
    ) -> Result<(), crate::CeraError> {
        // A request the model cannot serve (lfm3, or an incompatible `head_dim`) resolves to the
        // uncompressed cache here, and `Session::new` warns about it, so this stays a no-op.
        let resolved = compression.resolved_for(&self.config).cache_tag();

        // Cache lock outside, tag lock inside, so the tag and the cache's namespace
        // move as one unit. Dropping the tag guard before rebuilding would let two
        // concurrent calls land tag=Y with the cache fingerprinted X: the
        // cross-mode collision this exists to prevent. `configure_cache` takes
        // these locks in the same order.
        let mut cache = self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
        let mut tag = self.kv_cache_tag.lock().unwrap_or_else(|e| e.into_inner());
        match tag.as_deref() {
            Some(existing) if existing == resolved => return Ok(()),
            Some(existing) => {
                return Err(crate::CeraError::KvCompressionConflict {
                    configured: describe_cache_tag(existing),
                    requested: describe_cache_tag(&resolved),
                });
            }
            None => {}
        }
        let id = Self::namespace_for(
            &resolved,
            self.cache_model_id(cache.config.cache_dir.is_some()),
        );
        *tag = Some(resolved);
        let cache_config = cache.config.clone();
        *cache = KvPrefixCache::for_model(cache_config, &self.config, &self.model_id, &id);
        Ok(())
    }

    /// The rows the token prefill feeds the first layer: a dequantized embedding-table row per token,
    /// with no scaling (see `forward_prefill_inner`).
    fn embed_token_rows(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        let hs = self.config.hidden_size;
        if tokens.iter().any(|&t| t as usize >= self.embd_ref.m) {
            return None;
        }
        let mut rows = vec![0.0f32; tokens.len() * hs];
        for (j, &t) in tokens.iter().enumerate() {
            self.dequantize_row_into(&self.embd_ref, t as usize, &mut rows[j * hs..(j + 1) * hs]);
        }
        Some(rows)
    }

    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.forward_prefill_from_embeddings_chunk(embeddings, n_tokens, start_pos, state, true)
    }

    fn forward_prefill_from_embeddings_chunk(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = n_tokens;
        assert!(
            n > 0,
            "forward_prefill_from_embeddings requires at least one frame"
        );
        assert_eq!(
            embeddings.len(),
            n * hs,
            "embeddings.len() ({}) != n_tokens ({}) * hidden_size ({})",
            embeddings.len(),
            n,
            hs
        );

        let mut hidden = embeddings.to_vec();
        if need_logits {
            self.prefill_layers_and_logits(hidden, n, start_pos, state)
        } else {
            self.prefill_layers_loop(&mut hidden, n, start_pos, state);
            Vec::new()
        }
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Not `lfm3`: the TurboQuant attention read has no sliding-window mask.
    fn turboquant_supported(&self) -> bool {
        crate::kv_cache::turboquant_capable(&self.config)
    }

    fn supports_kv_shift(&self) -> bool {
        // CPU LFM2 implements shift with RoPE re-rotation. The wgpu and
        // Metal overrides mirror this with a GPU-side shift shader, and
        // report `false` only while a TurboQuant cache is active. `lfm3` rotates
        // each layer on its own base and mask, which the single-schedule
        // re-rotation in `shift_kv` does not know.
        self.lfm3.is_none()
    }

    fn shift_kv(&self, state: &mut crate::kv_cache::InferenceState, n_keep: usize, shift: usize) {
        let head_dim = self.config.hidden_size / self.config.n_heads;
        state.shift_kv_with_rope(
            n_keep,
            shift,
            self.config.rope_theta,
            head_dim,
            &self.config.kv_heads_per_layer,
            crate::backend::cpu::RopeType::Neox,
            None,
        );
    }
}

// ── GPU weight source ───────────────────────────────────────────────────────
//
// Drives the wgpu loader (`gpu_lfm2.rs`) for LFM2. Conv layers expose
// `conv_*` refs; attention layers expose `attn_*` refs + QK-norm. LFM2 has no
// QKV bias / untied output / Llama-3 freq-factors, uses NEOX RoPE, identity
// scalars, and supports the batched-prefill GPU path.
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    feature = "hexagon"
))]
impl crate::model::gpu_weight_source::GpuWeightSource for LfmModel {
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
        LfmModel::attn_q_norm_weight(self, layer)
    }
    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        LfmModel::attn_k_norm_weight(self, layer)
    }
    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        LfmModel::conv_weight(self, layer)
    }
    fn attn_q_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn attn_k_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn attn_v_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn rope_freqs(&self) -> Option<&[f32]> {
        self.lfm3.as_ref().and_then(|l| {
            l.layers
                .iter()
                .find_map(|ly| ly.attn.as_ref()?.rope_freq_factors.as_deref())
        })
    }
    fn attn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].post_operator_norm.as_deref())
    }
    fn ffn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].post_ffn_norm.as_deref())
    }

    fn weight_bytes(&self, wref: &WeightRef) -> Result<std::borrow::Cow<'_, [u8]>> {
        Ok(std::borrow::Cow::Borrowed(transformer::weight_data(
            &self.gguf, wref,
        )))
    }
    fn dequantize_weight(&self, wref: &WeightRef) -> Result<Vec<f32>> {
        Ok(transformer::dequantize_weight(&self.gguf, wref))
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        (self.config.architecture == "lfm3" && self.gguf.tensors.contains_key("output.weight"))
            .then_some(&self.logits_ref)
    }
    fn ffn_gate_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.gate)
    }
    fn ffn_up_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.up)
    }
    fn ffn_down_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.down)
    }
    fn moe_refs(&self, layer: usize) -> Option<&MoeFfnRefs> {
        match &self.layer_refs[layer].ffn {
            FfnRefs::Dense(_) => None,
            FfnRefs::Moe(m) => Some(m),
        }
    }
    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].shortconv_in_proj.as_ref()
    }
    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].shortconv_out_proj.as_ref()
    }
    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_q.as_ref()
    }
    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_k.as_ref()
    }
    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_v.as_ref()
    }
    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_output.as_ref()
    }
    fn attn_gate_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].attn.as_ref().map(|a| &a.gate))
    }
    fn layer_rope_theta(&self, layer: usize) -> f32 {
        self.lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].attn.as_ref().map(|a| a.rope_theta))
            .unwrap_or(self.config.rope_theta)
    }
    fn layer_has_rope_freqs(&self, layer: usize) -> bool {
        self.lfm3
            .as_ref()
            .and_then(|l| {
                l.layers[layer]
                    .attn
                    .as_ref()
                    .map(|a| a.rope_freq_factors.is_some())
            })
            .unwrap_or(false)
    }
    fn layer_sliding_window(&self, layer: usize) -> Option<usize> {
        self.lfm3
            .as_ref()
            .and_then(|l| l.layers[layer].attn.as_ref().and_then(|a| a.window))
    }
    fn layer_is_non_causal(&self, layer: usize) -> bool {
        LfmModel::layer_is_non_causal(self, layer)
    }

    fn gpu_unsupported_reason(&self) -> Option<String> {
        if !self.config.is_causal
            && (self.config.moe.is_some() || self.config.architecture == "lfm2moe")
        {
            return Some(format!(
                "the GPU backends run {} with causal attention only, and this model is bidirectional (CPU only)",
                self.config.architecture
            ));
        }
        if let Some(k) = self.config.conv_kernel_size
            && k > 8
        {
            return Some(format!(
                "the GPU backends support shortconv.l_cache in 2..=8, got {k}"
            ));
        }
        None
    }
    fn rope_type(&self) -> crate::backend::cpu::RopeType {
        crate::backend::cpu::RopeType::Neox
    }
    fn supports_batched_prefill(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod conv_kernel_size_tests {
    use super::{CPU_CONV_KERNEL_MAX, GPU_CONV_KERNEL_MAX, validate_conv_kernel_size};

    fn lfm2(v: Option<usize>) -> anyhow::Result<Option<usize>> {
        validate_conv_kernel_size("lfm2", v, GPU_CONV_KERNEL_MAX)
    }

    #[test]
    fn accepts_the_shipped_value_and_absence() {
        assert_eq!(lfm2(Some(3)).unwrap(), Some(3));
        // Absent metadata falls back to 3 downstream, so it stays None here.
        assert_eq!(lfm2(None).unwrap(), None);
    }

    #[test]
    fn accepts_the_register_array_bounds() {
        for k in [2usize, 4] {
            assert_eq!(lfm2(Some(k)).unwrap(), Some(k));
        }
    }

    #[test]
    fn rejects_values_the_kernels_cannot_honor() {
        // 0 and 1 underflow `d_conv = kernel_size - 1`; anything above 4 overruns
        // the batched kernel's fixed-size `w_local` / `rb` registers, which used
        // to make it silently skip the dispatch and emit wrong prefill logits.
        for k in [0usize, 1, 5, 64, usize::MAX] {
            let err = lfm2(Some(k)).unwrap_err().to_string();
            assert!(
                err.contains("l_cache must be in 2..=4"),
                "unexpected error for k={k}: {err}"
            );
        }
    }

    #[test]
    fn lfm3_takes_its_wider_kernel_but_not_an_unbounded_one() {
        let lfm3 = |k| validate_conv_kernel_size("lfm3", k, CPU_CONV_KERNEL_MAX);
        assert_eq!(lfm3(Some(7)).unwrap(), Some(7));
        for k in [0usize, 1, CPU_CONV_KERNEL_MAX + 1] {
            let err = lfm3(Some(k)).unwrap_err().to_string();
            assert!(err.starts_with("lfm3.shortconv.l_cache"), "{err}");
        }
    }
}

#[cfg(test)]
mod moe_routing_tests {
    use super::select_experts;

    /// The bias steers *selection*; the weights come from the unbiased
    /// probabilities. Both halves are asserted, because getting only the first
    /// half right is the failure mode that still produces fluent text.
    #[test]
    fn bias_ranks_experts_but_does_not_weight_them() {
        // Expert 1 has the lower probability but a large enough bias to be
        // ranked first; expert 0 wins on raw probability.
        let probs = [0.40, 0.30, 0.05, 0.01];
        let biases = [0.00, 0.50, 0.00, 0.00];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 2, &mut selected);

        let picked: Vec<usize> = selected.iter().map(|&(e, _)| e).collect();
        assert_eq!(picked, vec![1, 0], "bias must decide the ranking");

        // Weights are the unbiased probs renormalized: 0.30/0.70, 0.40/0.70.
        // Note they *ascend*; a biased-weight implementation would emit
        // descending values and pass a "weights are sorted" check.
        let weights: Vec<f32> = selected.iter().map(|&(_, w)| w).collect();
        assert!((weights[0] - 0.30 / 0.70).abs() < 1e-6, "got {weights:?}");
        assert!((weights[1] - 0.40 / 0.70).abs() < 1e-6, "got {weights:?}");
        assert!(
            weights[0] < weights[1],
            "unbiased weights must not be forced into descending order: {weights:?}"
        );
    }

    /// Regression against real llama.cpp output: `ffn_moe_topk-2` /
    /// `ffn_moe_weights_norm-2` for token 3 of "The capital of France is"
    /// (BOS-prefixed) on LFM2.5-8B-A1B-Q4_0. This token is the useful one
    /// precisely because its normalized weights are non-monotonic
    /// (0.3474 < 0.3556), which only happens when ranking and weighting use
    /// different scores.
    #[test]
    fn matches_llama_cpp_non_monotonic_token() {
        // Reconstructed from the reference dump: expert 21 ranks first on
        // biased score despite expert 17 having the higher raw probability.
        let mut probs = [0.0f32; 32];
        let mut biases = [0.0f32; 32];
        probs[21] = 0.3474;
        probs[17] = 0.3556;
        probs[6] = 0.1866;
        probs[12] = 0.1103;
        biases[21] = 0.10;

        let mut selected = Vec::new();
        select_experts(&probs, &biases, 4, &mut selected);

        assert_eq!(
            selected.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
            vec![21, 17, 6, 12]
        );
        let weights: Vec<f32> = selected.iter().map(|&(_, w)| w).collect();
        // The reference weights already sum to 1, so renormalizing is a no-op
        // and they should come back unchanged.
        for (got, want) in weights.iter().zip([0.3474, 0.3556, 0.1866, 0.1103]) {
            assert!((got - want).abs() < 1e-3, "got {weights:?}");
        }
    }

    /// Ties resolve to the lower expert index, matching `ggml_argsort_top_k`.
    /// Without the reversed index tiebreak `max_by` keeps the last maximum and
    /// silently picks the higher index.
    #[test]
    fn ties_break_toward_the_lower_index() {
        let probs = [0.25, 0.25, 0.25, 0.25];
        let biases = [0.0; 4];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 2, &mut selected);
        assert_eq!(
            selected.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    /// An all-zero gate must not divide by zero. llama.cpp clamps the divisor
    /// to f16's smallest positive normal; the weights stay finite.
    #[test]
    fn zero_gate_does_not_divide_by_zero() {
        let probs = [0.0f32; 4];
        let biases = [0.0f32; 4];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 2, &mut selected);
        assert!(selected.iter().all(|&(_, w)| w.is_finite()), "{selected:?}");
    }

    /// Asking for more experts than exist must stop at the expert count rather
    /// than spin or emit duplicates.
    #[test]
    fn n_used_saturates_at_the_expert_count() {
        let probs = [0.5, 0.3];
        let biases = [0.0, 0.0];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 8, &mut selected);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 0);
        assert_eq!(selected[1].0, 1);
    }
}

#[cfg(test)]
mod loader_tests {
    use super::LfmModel;
    use crate::gguf::GgufFile;

    #[test]
    fn zero_embedding_length_is_rejected_at_load() {
        // `hidden_size == 0` would divide by zero in mean-pooling and size
        // every buffer at zero; fail at load with a typed error instead.
        let mut writer = crate::convert::writer::GgufWriter::new();
        writer.add_string("general.architecture", "lfm2");
        writer.add_u32("lfm2.block_count", 1);
        writer.add_u32("lfm2.embedding_length", 0);
        let mut bytes = Vec::new();
        writer.write_header_and_tensor_info(&mut bytes).unwrap();
        let gguf = GgufFile::from_bytes(bytes.into()).unwrap();
        let err = LfmModel::parse_config(&gguf, 32).unwrap_err();
        assert!(
            err.to_string()
                .contains("lfm2.embedding_length must be > 0"),
            "unexpected error: {err}"
        );
    }
}

#[cfg(all(test, feature = "mmap"))]
mod no_repack_tests {
    use super::*;

    /// The no-repack loader resolves identical metadata without the CPU int8
    /// repacks (gigabytes the GPU/Metal loaders would allocate only to free
    /// after upload). Needs the 230M model locally; skips without it (fails under `CERA_REQUIRE_MODEL`).
    #[test]
    fn no_repack_skips_cpu_repacks() {
        let home = std::env::var("HOME").expect("HOME unset");
        let path = std::path::PathBuf::from(home)
            .join(".leap/models/LFM2.5-230M-Q4_0/LFM2.5-230M-Q4_0.gguf");
        if !crate::model::transformer::require_model_or_skip(&path) {
            return;
        }
        let full = LfmModel::from_gguf(GgufFile::open(&path).unwrap(), 64).unwrap();
        let lite = LfmModel::from_gguf_no_repack(GgufFile::open(&path).unwrap(), 64).unwrap();
        assert_eq!(full.config.n_layers, lite.config.n_layers);
        assert_eq!(full.config.hidden_size, lite.config.hidden_size);
        assert_eq!(full.layer_refs.len(), lite.layer_refs.len());
        assert!(lite.config.moe.is_none(), "test model must be dense");

        #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(has_blas)))]
        {
            fn projection_refs(refs: &[LayerWeightRefs]) -> Vec<&WeightRef> {
                let mut out = Vec::new();
                for layer in refs {
                    let FfnRefs::Dense(dense) = &layer.ffn else {
                        unreachable!("test model must be dense");
                    };
                    out.extend([&dense.gate, &dense.up, &dense.down]);
                    out.extend(layer.shortconv_in_proj.iter());
                    out.extend(layer.shortconv_out_proj.iter());
                    out.extend(layer.attn_q.iter());
                    out.extend(layer.attn_k.iter());
                    out.extend(layer.attn_v.iter());
                    out.extend(layer.attn_output.iter());
                }
                out
            }
            let full_refs = projection_refs(&full.layer_refs);
            let lite_refs = projection_refs(&lite.layer_refs);
            // Same metadata resolved...
            assert_eq!(full_refs.len(), lite_refs.len());
            assert!(
                full_refs
                    .iter()
                    .all(|w| w.dtype == crate::tensor::DType::Q4_0),
                "test model must be all-Q4_0 projections"
            );
            // ...repacks present in the full load, absent in the lite one.
            assert!(
                full_refs.iter().any(|w| w.repacked.is_some()),
                "full load repacked nothing; test exercises no path"
            );
            assert!(
                lite_refs.iter().all(|w| w.repacked.is_none()),
                "no-repack load still repacked weights"
            );
        }
    }
}

/// The routed-MoE prefill must produce, bit for bit, what running each token
/// through the decode FFN produces.
///
/// It used to walk experts by index and accumulate with a fused multiply-add, while
/// decode accumulates in router-rank order with an unfused `acc += weight * v`, so
/// the combined output differed in the last bits. A logits comparison on a tiny model
/// hides that (the next quantization absorbs it), but the real model amplifies it
/// through discrete expert choice: cosine 0.97 on LFM2.5-8B-A1B-Q4_0. Comparing the FFN
/// output itself over many tokens does not hide it.
///
/// Loaded without the CPU repacks (their prefill GEMMs reorder sums on purpose).
/// Not run under BLAS, whose expert GEMMs are f32 SGEMMs.
#[cfg(all(test, not(has_blas), feature = "mmap"))]
mod moe_prefill_identity_tests {
    use super::*;

    fn generate_model() -> Option<std::path::PathBuf> {
        let strict = std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
        let has_deps = std::process::Command::new("python3")
            .args(["-c", "import numpy, gguf"])
            .output()
            .is_ok_and(|o| o.status.success());
        if !has_deps {
            assert!(
                !strict,
                "python3 lacks numpy/gguf and CERA_REQUIRE_ORACLE=1"
            );
            eprintln!("skipping: python3 lacks numpy/gguf for the synthetic lfm2moe model");
            return None;
        }
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../scripts/oracle/create_lfm2_test_model.py");
        let out =
            std::env::temp_dir().join(format!("test_lfm2moe_unit_{}.gguf", std::process::id()));
        let status = std::process::Command::new("python3")
            .arg(&script)
            .arg(&out)
            .arg("--moe")
            .status()
            .expect("run python3");
        assert!(
            status.success(),
            "generating the synthetic lfm2moe model failed"
        );
        Some(out)
    }

    #[test]
    fn prefill_moe_ffn_equals_per_token_decode_bit_for_bit() {
        let Some(path) = generate_model() else {
            return;
        };
        let model = LfmModel::from_gguf_no_repack(GgufFile::open(&path).unwrap(), 128).unwrap();
        let _ = std::fs::remove_file(&path);

        let (hs, n) = (model.config.hidden_size, 64usize);
        let mut state = InferenceState::for_prefill(&model.config, n).unwrap();
        state.scratch.q8_scales.resize(hs / 32, 0.0);
        state.scratch.q8_quants.resize(hs, 0);

        // Deterministic inputs of a normalized-activation size.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let ffn_input: Vec<f32> = (0..n * hs)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect();

        let routed: Vec<_> = model
            .layer_refs
            .iter()
            .enumerate()
            .filter_map(|(layer, refs)| match &refs.ffn {
                FfnRefs::Moe(moe) => Some((layer, moe)),
                FfnRefs::Dense(_) => None,
            })
            .collect();
        assert!(
            routed.len() >= 3,
            "the synthetic model must have routed layers"
        );

        routed.into_iter().for_each(|(layer, moe)| {
            let mut col = vec![0.0f32; hs];
            let mut want = vec![0.0f32; n * hs];
            (0..n).for_each(|j| {
                col.copy_from_slice(&ffn_input[j * hs..(j + 1) * hs]);
                #[cfg(target_arch = "aarch64")]
                LfmModel::quantize_to_scratch(&col, &mut state);
                model.forward_moe_ffn(layer, moe, hs, &col, &mut state);
                want[j * hs..(j + 1) * hs].copy_from_slice(&state.scratch.out[..hs]);
            });

            let mut got = vec![0.0f32; n * hs];
            model.prefill_moe_ffn(layer, moe, hs, n, &ffn_input, &mut got, &mut col, &mut state);

            let mismatches = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                mismatches, 0,
                "layer {layer}: {mismatches} of {} MoE FFN outputs differ between prefill and decode",
                n * hs
            );
        });
    }
}

#[cfg(test)]
pub(crate) mod release_tests {
    use super::*;

    /// Minimal synthetic LFM2 GGUF: one short-conv layer and one attention
    /// layer, Q4_0 projections, F32 norms and embedding, sized so a 512-token
    /// prefill puts the batched working set above the release threshold.
    /// Runs anywhere, no model file needed. Shared with the GPU lost-device
    /// test, which needs a loadable model on every CI leg.
    pub(crate) fn synthetic_lfm2_gguf() -> GgufFile {
        const HS: usize = 512;
        const INTER: usize = 1024;
        const VOCAB: usize = 64;
        const N_HEADS: usize = 8;
        const HEAD_DIM: usize = 64;
        struct T {
            name: String,
            dims: Vec<usize>,
            ty: u32,
            data: Vec<u8>,
        }
        let mut seed = 0x9e37_79b9u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut f32t = |name: &str, dims: &[usize], scale: f32, base: f32| {
            let n: usize = dims.iter().product();
            T {
                name: name.into(),
                dims: dims.to_vec(),
                ty: 0,
                data: (0..n)
                    .flat_map(|_| (base + next() * 2.0 * scale).to_le_bytes())
                    .collect(),
            }
        };
        let mut q4 = {
            let mut seed = 0x1357_9bdfu64;
            move |name: &str, dims: &[usize]| {
                let blocks: usize = dims.iter().product::<usize>() / 32;
                let mut data = Vec::with_capacity(blocks * 18);
                for _ in 0..blocks {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let scale = 0.01 + ((seed >> 40) as f32 / (1u64 << 24) as f32) * 0.02;
                    data.extend_from_slice(&crate::quant::f32_to_f16(scale).to_le_bytes());
                    for _ in 0..16 {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        data.push((seed >> 56) as u8);
                    }
                }
                T {
                    name: name.into(),
                    dims: dims.to_vec(),
                    ty: 2,
                    data,
                }
            }
        };
        let q_dim = N_HEADS * HEAD_DIM;
        let mut t = vec![
            f32t("token_embd.weight", &[HS, VOCAB], 1.0, 0.0),
            f32t("token_embd_norm.weight", &[HS], 0.1, 1.0),
        ];
        for l in 0..2 {
            let n = |s: &str| format!("blk.{l}.{s}");
            t.push(f32t(&n("attn_norm.weight"), &[HS], 0.1, 1.0));
            t.push(f32t(&n("ffn_norm.weight"), &[HS], 0.1, 1.0));
            if l == 1 {
                t.push(q4(&n("attn_q.weight"), &[HS, q_dim]));
                t.push(q4(&n("attn_k.weight"), &[HS, HEAD_DIM]));
                t.push(q4(&n("attn_v.weight"), &[HS, HEAD_DIM]));
                t.push(q4(&n("attn_output.weight"), &[q_dim, HS]));
                t.push(f32t(&n("attn_q_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
                t.push(f32t(&n("attn_k_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
            } else {
                t.push(q4(&n("shortconv.in_proj.weight"), &[HS, 3 * HS]));
                t.push(q4(&n("shortconv.out_proj.weight"), &[HS, HS]));
                t.push(f32t(&n("shortconv.conv.weight"), &[3, HS], 0.3, 0.0));
            }
            t.push(q4(&n("ffn_gate.weight"), &[HS, INTER]));
            t.push(q4(&n("ffn_up.weight"), &[HS, INTER]));
            t.push(q4(&n("ffn_down.weight"), &[INTER, HS]));
        }

        use crate::gguf::GgufBuilder;
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "lfm2")
            .kv_u32("lfm2.block_count", 2)
            .kv_u32("lfm2.embedding_length", HS as u32)
            .kv_u32("lfm2.feed_forward_length", INTER as u32)
            .kv_u32("lfm2.attention.head_count", N_HEADS as u32)
            // Per-layer KV heads: 0 marks the short-conv layer.
            .kv_i32_array("lfm2.attention.head_count_kv", vec![0, 1])
            .kv_u32("lfm2.shortconv.l_cache", 3)
            .kv_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_f32("lfm2.rope.freq_base", 10_000.0)
            .kv_u32("lfm2.context_length", 1024)
            .kv_u32("lfm2.vocab_size", VOCAB as u32);
        for x in t {
            b = b.tensor(x.name, &x.dims, x.ty, x.data);
        }
        b.build()
    }

    /// The first decode token after a large prefill drops the batched
    /// working set (`run_layers` calls `release_if_large`). Pins that
    /// wiring: without the call the scratch pins its prefill high-water
    /// mark for the rest of the session. Uses a synthetic model, so it runs
    /// on every CI leg.
    #[test]
    fn decode_after_large_prefill_releases_prefill_scratch() {
        use crate::kv_cache::PrefillScratch;
        use crate::model::Model;
        let model = LfmModel::from_gguf(synthetic_lfm2_gguf(), 1024).unwrap();
        let mut state = InferenceState::from_config(model.config()).unwrap();
        let prompt: Vec<u32> = (0..512).map(|i| 1 + (i % 60) as u32).collect();
        model.forward_prefill(&prompt, 0, &mut state);
        assert!(
            state.prefill_scratch.capacity_bytes() > PrefillScratch::RELEASE_THRESHOLD_BYTES,
            "a 512-token prefill must exceed the release threshold, got {} bytes",
            state.prefill_scratch.capacity_bytes()
        );
        model.forward(&[1], prompt.len(), &mut state);
        assert_eq!(
            state.prefill_scratch.capacity_bytes(),
            0,
            "the first decode step must release the prefill working set"
        );
    }

    /// The marker rows a tiled image splices into its prompt are the rows a token prefill would stage:
    /// feeding them through the embedding-seeded prefill gives the logits of prefilling the token
    /// ids. A scaling or table-lookup slip here still yields a fluent but wrong tiled prompt, and no
    /// other host test would notice. Out-of-vocabulary ids are refused, not looked up.
    #[test]
    fn marker_rows_reproduce_the_token_prefill() {
        use crate::model::Model;
        let model = LfmModel::from_gguf(synthetic_lfm2_gguf(), 256).unwrap();
        let tokens: Vec<u32> = vec![3, 17, 42, 5];
        let rows = model
            .embed_image_marker_rows(&tokens)
            .expect("in-vocabulary ids must give rows");
        assert_eq!(rows.len(), tokens.len() * model.config().hidden_size);
        assert!(rows.iter().all(|v| v.is_finite()));
        assert!(model.embed_image_marker_rows(&[u32::MAX]).is_none());

        let mut by_tokens = InferenceState::from_config(model.config()).unwrap();
        let want = model.forward_prefill(&tokens, 0, &mut by_tokens);
        let mut by_rows = InferenceState::from_config(model.config()).unwrap();
        let got = model.forward_prefill_from_embeddings(&rows, tokens.len(), 0, &mut by_rows);
        assert_eq!(want.len(), got.len());
        let max_abs = want
            .iter()
            .zip(&got)
            .map(|(a, b)| {
                // A NaN difference is a failure: `f32::max` would drop it.
                let d = (a - b).abs();
                if d.is_nan() { f32::INFINITY } else { d }
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "marker rows diverge from the token prefill by {max_abs:.2e}"
        );
    }

    /// Speculative verification calls `forward_prefill_logits_all` with a
    /// handful of tokens and never decodes a single token in between, so it
    /// must release a large earlier prefill working set itself.
    #[test]
    fn small_verify_batch_after_large_prefill_releases_prefill_scratch() {
        use crate::kv_cache::PrefillScratch;
        use crate::model::Model;
        let model = LfmModel::from_gguf(synthetic_lfm2_gguf(), 1024).unwrap();
        let mut state = InferenceState::from_config(model.config()).unwrap();
        let prompt: Vec<u32> = (0..512).map(|i| 1 + (i % 60) as u32).collect();
        model.forward_prefill(&prompt, 0, &mut state);
        assert!(state.prefill_scratch.capacity_bytes() > PrefillScratch::RELEASE_THRESHOLD_BYTES);
        let logits = model.forward_prefill_logits_all(&[1, 2, 3, 4], prompt.len(), &mut state);
        assert_eq!(logits.len(), 4 * model.config().vocab_size);
        assert!(
            state.prefill_scratch.capacity_bytes() <= PrefillScratch::RELEASE_THRESHOLD_BYTES,
            "a small verify batch must not keep the 512-token working set, got {} bytes",
            state.prefill_scratch.capacity_bytes()
        );
    }
}

/// A bidirectional model whose projections are F16 or F32 cannot take the batched GEMM path,
/// and the per-token path that used to take them is causal. These tests pin the float path
/// that runs instead.
#[cfg(test)]
pub(crate) mod bidirectional_float_tests {
    use super::*;
    use crate::convert::TargetQuant;
    use crate::convert::writer::GGML_TYPE_Q8_0;
    use crate::model::Model;

    // 128 wide: the batched Q8_0 GEMV on Metal reads in steps of 128 values
    pub(crate) const HS: usize = 128;
    const INTER: usize = 256;
    const VOCAB: usize = 32;
    const N_HEADS: usize = 4;
    const HEAD_DIM: usize = 32;

    /// A three-layer non-causal LFM2 (a short-conv block, then two attention blocks, so that one
    /// layer's attention feeds the next layer's) with deterministic weights; the projections are
    /// F32, or Q8_0 of the same values.
    /// [`bidirectional_gguf_with`] with 4 heads of 32.
    pub(crate) fn bidirectional_gguf(q8_projections: bool) -> GgufFile {
        bidirectional_gguf_with(q8_projections, N_HEADS, HEAD_DIM)
    }

    /// 4 heads of 32 and 2 of 64: the second takes the head_dim 64 attention kernels.
    /// Shared by the Metal and wgpu bidirectional tests so a fixture shape change touches one
    /// line.
    #[cfg(all(any(feature = "gpu", feature = "metal"), not(target_arch = "wasm32")))]
    pub(crate) const BIDIRECTIONAL_SHAPES: [(usize, usize); 2] = [(4, 32), (2, 64)];

    pub(crate) fn bidirectional_gguf_with(
        q8_projections: bool,
        n_heads: usize,
        head_dim: usize,
    ) -> GgufFile {
        assert_eq!(
            n_heads * head_dim,
            HS,
            "the heads must fill the hidden width"
        );
        use crate::gguf::GgufBuilder;
        let mut seed = 0x2545_f491u64;
        let mut values = move |n: usize, scale: f32, base: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    base + (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 2.0 * scale
                })
                .collect()
        };
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "lfm2")
            .kv_u32("lfm2.block_count", 3)
            .kv_u32("lfm2.embedding_length", HS as u32)
            .kv_u32("lfm2.feed_forward_length", INTER as u32)
            .kv_u32("lfm2.attention.head_count", n_heads as u32)
            .kv_i32_array("lfm2.attention.head_count_kv", vec![0, 1, 1])
            .kv_u32("lfm2.shortconv.l_cache", 3)
            .kv_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_f32("lfm2.rope.freq_base", 10_000.0)
            .kv_u32("lfm2.context_length", 4096)
            .kv_u32("lfm2.vocab_size", VOCAB as u32)
            .kv_raw("lfm2.attention.causal", 7, vec![0]);
        let embd = values(VOCAB * HS, 1.0, 0.0);
        b = b.tensor_f32("token_embd.weight", &[HS, VOCAB], &embd);
        let norm = values(HS, 0.1, 1.0);
        b = b.tensor_f32("token_embd_norm.weight", &[HS], &norm);
        let projection = |b: GgufBuilder, name: String, dims: [usize; 2], w: Vec<f32>| {
            if q8_projections {
                let mut out = vec![0u8; TargetQuant::compute_tensor_bytes(GGML_TYPE_Q8_0, w.len())];
                crate::convert::quantize_tensor_data(&w, GGML_TYPE_Q8_0, &mut out).unwrap();
                b.tensor(name, &dims, GGML_TYPE_Q8_0, out)
            } else {
                b.tensor_f32(name, &dims, &w)
            }
        };
        for layer in 0..3 {
            let n = |s: &str| format!("blk.{layer}.{s}");
            b = b.tensor_f32(n("attn_norm.weight"), &[HS], &values(HS, 0.1, 1.0));
            b = b.tensor_f32(n("ffn_norm.weight"), &[HS], &values(HS, 0.1, 1.0));
            if layer >= 1 {
                b = projection(
                    b,
                    n("attn_q.weight"),
                    [HS, n_heads * head_dim],
                    values(
                        [HS, n_heads * head_dim][0] * [HS, n_heads * head_dim][1],
                        0.12,
                        0.0,
                    ),
                );
                b = projection(
                    b,
                    n("attn_k.weight"),
                    [HS, head_dim],
                    values([HS, head_dim][0] * [HS, head_dim][1], 0.12, 0.0),
                );
                b = projection(
                    b,
                    n("attn_v.weight"),
                    [HS, head_dim],
                    values([HS, head_dim][0] * [HS, head_dim][1], 0.12, 0.0),
                );
                b = projection(
                    b,
                    n("attn_output.weight"),
                    [n_heads * head_dim, HS],
                    values(
                        [n_heads * head_dim, HS][0] * [n_heads * head_dim, HS][1],
                        0.12,
                        0.0,
                    ),
                );
                b = b.tensor_f32(
                    n("attn_q_norm.weight"),
                    &[head_dim],
                    &values(head_dim, 0.1, 1.0),
                );
                b = b.tensor_f32(
                    n("attn_k_norm.weight"),
                    &[head_dim],
                    &values(head_dim, 0.1, 1.0),
                );
            } else {
                b = projection(
                    b,
                    n("shortconv.in_proj.weight"),
                    [HS, 3 * HS],
                    values([HS, 3 * HS][0] * [HS, 3 * HS][1], 0.12, 0.0),
                );
                b = projection(
                    b,
                    n("shortconv.out_proj.weight"),
                    [HS, HS],
                    values([HS, HS][0] * [HS, HS][1], 0.12, 0.0),
                );
                b = b.tensor_f32(
                    n("shortconv.conv.weight"),
                    &[3, HS],
                    &values(3 * HS, 0.3, 0.0),
                );
            }
            b = projection(
                b,
                n("ffn_gate.weight"),
                [HS, INTER],
                values([HS, INTER][0] * [HS, INTER][1], 0.12, 0.0),
            );
            b = projection(
                b,
                n("ffn_up.weight"),
                [HS, INTER],
                values([HS, INTER][0] * [HS, INTER][1], 0.12, 0.0),
            );
            b = projection(
                b,
                n("ffn_down.weight"),
                [INTER, HS],
                values([INTER, HS][0] * [INTER, HS][1], 0.12, 0.0),
            );
        }
        b.build()
    }

    fn hidden(gguf: GgufFile, tokens: &[u32]) -> Vec<f32> {
        let model = LfmModel::from_gguf(gguf, 256).unwrap();
        assert!(
            !model.config().is_causal,
            "the fixture must be bidirectional"
        );
        let mut state = InferenceState::for_prefill(model.config(), tokens.len()).unwrap();
        model.hidden_states(tokens, &mut state)
    }

    #[test]
    fn f32_weights_take_the_float_path() {
        let model = LfmModel::from_gguf(bidirectional_gguf(false), 256).unwrap();
        assert!(model.bidirectional_needs_float_path());
    }

    /// The gate's other arm: a batchable model must stay off the float path. Without
    /// this, a gate stuck at true would route the Q8_0 fixture below onto the float
    /// path too, and the agreement test would compare the float path against itself.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
    #[test]
    fn q8_weights_stay_off_the_float_path() {
        if !transformer::batched_gemm_supports(crate::tensor::DType::Q8_0, HS) {
            eprintln!("SKIPPED: this host has no batched Q8_0 GEMM, so Q8_0 takes the float path");
            return;
        }
        let model = LfmModel::from_gguf(bidirectional_gguf(true), 256).unwrap();
        assert!(!model.bidirectional_needs_float_path());
    }

    #[test]
    fn a_bidirectional_cpu_model_reports_bidirectional_hidden_states() {
        // the d1 gate refuses causal-only extractors; without this opt-in a
        // flipped default would refuse every CPU question with no test failing
        let model = LfmModel::from_gguf(bidirectional_gguf(false), 256).unwrap();
        assert!(model.supports_bidirectional_hidden_states());
    }

    #[test]
    fn a_later_token_changes_an_earlier_hidden_state() {
        let a = hidden(bidirectional_gguf(false), &[3, 7, 9, 12, 5]);
        let b = hidden(bidirectional_gguf(false), &[3, 7, 9, 12, 20]);
        let first = |h: &[f32]| h[..HS].to_vec();
        let moved = first(&a)
            .iter()
            .zip(first(&b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(moved > 1e-4, "token 0 ignored the last token ({moved})");
    }

    /// Deterministic embeddings, `rows` of `HS`, for a seed.
    pub(crate) fn embeddings(rows: usize, seed: f32) -> Vec<f32> {
        (0..rows * HS)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.8)
            .collect()
    }

    fn trunk(prefix_rows: usize, prefix_seed: f32, text_seed: f32, text_rows: usize) -> Vec<f32> {
        let model = LfmModel::from_gguf(bidirectional_gguf(false), 256).unwrap();
        let mut h = embeddings(prefix_rows, prefix_seed);
        h.extend(embeddings(text_rows, text_seed));
        model.float_trunk(h, prefix_rows)
    }

    /// An empty prefix with empty tokens yields no rows instead of tripping the trunk's
    /// non-empty precondition: the contract promises `tokens.len()` rows unconditionally.
    #[test]
    fn empty_prefix_and_tokens_yield_no_rows() {
        use crate::kv_cache::InferenceState;
        let model = LfmModel::from_gguf(bidirectional_gguf(false), 256).unwrap();
        let mut state = InferenceState::for_prefill(model.config(), 0).unwrap();
        assert!(
            model
                .hidden_states_with_prefix(&[], &[], &mut state)
                .is_empty()
        );
    }

    /// A media prefix is a function of the media alone: its rows attend only to each other and
    /// the convolution at its last row does not read the first text row.
    #[test]
    fn a_media_prefix_does_not_depend_on_the_text() {
        let a = trunk(4, 0.1, 1.0, 5);
        let b = trunk(4, 0.1, 7.0, 5);
        assert_eq!(
            a[..4 * HS],
            b[..4 * HS],
            "the prefix rows moved with the text"
        );
        let moved = a[4 * HS..]
            .iter()
            .zip(&b[4 * HS..])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(moved > 1e-3, "the text rows ignored their own text");
    }

    /// Text rows read everything, the prefix included.
    #[test]
    fn text_rows_read_the_media_prefix() {
        let a = trunk(4, 0.1, 1.0, 5);
        let b = trunk(4, 3.0, 1.0, 5);
        let moved = a[4 * HS..]
            .iter()
            .zip(&b[4 * HS..])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(moved > 1e-3, "the text rows ignored the media prefix");
    }

    /// Without a prefix the trunk is the plain bidirectional one, and a prefix of rows that
    /// are the embeddings of tokens is not the same thing as those tokens being text.
    #[test]
    fn a_prefix_of_zero_rows_is_the_plain_trunk() {
        let model = LfmModel::from_gguf(bidirectional_gguf(false), 256).unwrap();
        let tokens = [3u32, 7, 9, 12];
        let mut embedded = vec![0f32; tokens.len() * HS];
        for (i, &t) in tokens.iter().enumerate() {
            model.dequantize_row_into(
                &model.embd_ref,
                t as usize,
                &mut embedded[i * HS..(i + 1) * HS],
            );
        }
        assert_eq!(
            model.float_trunk(embedded.clone(), 0),
            model.hidden_states_float_bidirectional(embedded)
        );
    }

    /// Both text and media forms of the same positions differ: with a prefix, the last prefix
    /// row's convolution cannot see the first text row, so the result is not the plain trunk.
    #[test]
    fn a_prefix_changes_how_the_same_rows_are_read() {
        let model = LfmModel::from_gguf(bidirectional_gguf(false), 256).unwrap();
        let h = embeddings(6, 0.5);
        let plain = model.float_trunk(h.clone(), 0);
        let split = model.float_trunk(h, 3);
        assert_ne!(plain, split);
    }

    /// The float path computes the function the batched quantized path does: the same
    /// weights, once as F32 and once as Q8_0, agree to the quantization error.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", has_blas))]
    #[test]
    fn the_float_path_agrees_with_the_batched_quantized_path() {
        if !transformer::batched_gemm_supports(crate::tensor::DType::Q8_0, HS) {
            eprintln!("SKIPPED: this host has no batched Q8_0 GEMM to compare against");
            return;
        }
        let tokens = [3u32, 7, 9, 12, 5, 21, 8];
        let float = hidden(bidirectional_gguf(false), &tokens);
        let quant_model = LfmModel::from_gguf(bidirectional_gguf(true), 256).unwrap();
        assert!(
            !quant_model.bidirectional_needs_float_path(),
            "the Q8_0 fixture must take the batched path, or this compares the float path against itself"
        );
        let mut state = InferenceState::for_prefill(quant_model.config(), tokens.len()).unwrap();
        let quant = quant_model.hidden_states(&tokens, &mut state);
        assert_eq!(float.len(), tokens.len() * HS);
        let (float_rows, _) = float.as_chunks::<HS>();
        let (quant_rows, _) = quant.as_chunks::<HS>();
        for (t, (f, q)) in float_rows.iter().zip(quant_rows).enumerate() {
            let dot: f32 = f.iter().zip(q).map(|(a, b)| a * b).sum();
            let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cosine = dot / (norm(f) * norm(q));
            assert!(cosine > 0.995, "token {t}: cosine {cosine}");
        }
    }

    /// A public d1-omni file counts its head blocks in `block_count`; the trunk is the rest.
    #[test]
    fn a_public_d1_omni_trunk_excludes_the_head_blocks() {
        use crate::gguf::GgufBuilder;
        let base = || {
            GgufBuilder::new()
                .kv_str("general.architecture", "lfm2")
                .kv_u32("lfm2.block_count", 3)
                .kv_u32("lfm2.embedding_length", 128)
                .kv_u32("lfm2.feed_forward_length", 256)
                .kv_u32("lfm2.attention.head_count", 2)
                .kv_i32_array("lfm2.attention.head_count_kv", vec![0, 0, 2])
                .kv_u32("lfm2.shortconv.l_cache", 3)
                .kv_u32("lfm2.vocab_size", 64)
                .kv_u32("lfm2.decision.block_count", 1)
                .kv_bool("lfm2.attention.causal", false)
        };
        let omni = base().kv_str("lfm2.decision.type", "lfm2-d1-omni").build();
        let config = LfmModel::parse_config(&omni, 64).unwrap();
        assert_eq!(config.n_layers, 2);
        assert_eq!(config.block_types.len(), 2);
        assert!(!config.is_causal);
        // the same settings without the public type are an ordinary three-block model
        let plain = base().build();
        assert_eq!(LfmModel::parse_config(&plain, 64).unwrap().n_layers, 3);
        // a head count that leaves no trunk is refused
        let all_head = base()
            .kv_str("lfm2.decision.type", "lfm2-d1-omni")
            .kv_u32("lfm2.decision.block_count", 3)
            .build();
        let err = LfmModel::parse_config(&all_head, 64).unwrap_err();
        assert!(err.to_string().contains("leaves no trunk"), "{err}");
    }
}

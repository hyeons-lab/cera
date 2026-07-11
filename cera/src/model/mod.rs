pub mod lfm2;
pub mod llama;
pub mod transformer;

#[cfg(feature = "gpu")]
pub mod gpu_lfm2;

#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
pub(crate) mod gpu_weight_source;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_lfm2;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_audio_decoder;

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail, ensure};

use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;

/// Per-layer block type (for hybrid architectures like LFM2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    Attention,
    GatedConv,
}

/// Architecture scalar multipliers (Granite 3.x; HF names in parens). Every
/// other arch leaves all of these absent ⇒ [`ScalarMultipliers::default`]
/// (identity), so they are a no-op for LLaMA/Mistral/Qwen.
///
/// These travel on [`ModelConfig`] alongside the other GGUF-derived scalars
/// (`rope_theta`, `rms_norm_eps`, …) so a new multiplier-bearing arch or
/// back-end consumes them from config instead of re-deriving the four keys.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarMultipliers {
    /// `embedding_multiplier` — scale embeddings right after the token lookup.
    /// `1.0` ⇒ no-op.
    pub embedding: f32,
    /// `residual_multiplier` — scale each attention/FFN block output before its
    /// residual add. `1.0` ⇒ no-op.
    pub residual: f32,
    /// `attention_multiplier` — softmax scale that *replaces* `1/sqrt(head_dim)`.
    /// `None` ⇒ use the default `1/sqrt(head_dim)` (it is a replacement, not a
    /// multiplier, so it can't share the `1.0`-identity representation).
    pub attn: Option<f32>,
    /// `logits_scaling` — divide the final logits by this. `1.0` ⇒ no-op.
    pub logit: f32,
}

impl Default for ScalarMultipliers {
    fn default() -> Self {
        Self {
            embedding: 1.0,
            residual: 1.0,
            attn: None,
            logit: 1.0,
        }
    }
}

impl ScalarMultipliers {
    /// Load the four Granite scalars from GGUF metadata under `{prefix}.*`.
    /// Absent keys map to identity, so this returns [`Self::default`] for every
    /// non-Granite arch.
    pub fn from_gguf(gguf: &GgufFile, prefix: &str) -> Result<Self> {
        let embedding = gguf
            .get_f32(&format!("{prefix}.embedding_scale"))
            .unwrap_or(1.0);
        let residual = gguf
            .get_f32(&format!("{prefix}.residual_scale"))
            .unwrap_or(1.0);
        // llama.cpp treats a stored `attention.scale == 0.0` as "absent ⇒ use
        // 1/sqrt(head_dim)", so map Some(0.0) → None to match (a literal 0.0
        // would otherwise zero every attention score).
        let attn = gguf
            .get_f32(&format!("{prefix}.attention.scale"))
            .filter(|&s| s != 0.0);
        let logit = gguf
            .get_f32(&format!("{prefix}.logit_scale"))
            .unwrap_or(1.0);
        ensure!(logit != 0.0, "{prefix}.logit_scale must be non-zero");
        Ok(Self {
            embedding,
            residual,
            attn,
            logit,
        })
    }
}

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Attention head dimension. Usually `hidden_size / n_heads`, but some
    /// architectures (e.g. Qwen3) decouple it via `*.attention.key_length`, so
    /// it is carried explicitly: Q is `n_heads * head_dim`, KV is
    /// `n_kv_heads * head_dim`, either of which can exceed `hidden_size`.
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Per-layer block types. Empty for pure-transformer models.
    pub block_types: Vec<BlockType>,
    /// Convolution kernel size (LFM2-specific).
    pub conv_kernel_size: Option<usize>,
    /// Per-layer KV head counts. Length = n_layers. 0 for conv layers.
    pub kv_heads_per_layer: Vec<usize>,
    /// Architecture scalar multipliers (Granite 3.x). Identity for every other
    /// arch — see [`ScalarMultipliers`].
    pub scalars: ScalarMultipliers,
}

/// Trait for loaded models that can run forward passes.
///
/// `Send + Sync` is required so `std::sync::Arc<dyn Model>` is itself
/// `Send + Sync`, which is the prerequisite for exposing `Session`
/// through UniFFI's foreign-function boundary (the bindgen'd
/// Kotlin/Swift wrappers move the `Arc` between threads and require
/// both bounds).
///
/// **GPU backends keep per-instance scratch buffers + GPU-resident
/// KV caches in their own state.** `MetalLfm2Model` self-defends with
/// an internal `Mutex<()>` (`infer_lock`) that serializes every Model
/// trait call: two threads cloning the same `Arc<dyn Model>` and
/// running `forward()` / `forward_prefill()` concurrently are safe —
/// the second call blocks until the first releases. The lock is
/// uncontended in the single-Session-per-Model case, costing ~50 ns
/// per call (negligible vs Metal dispatch). For genuine throughput
/// across concurrent Sessions, prefer one `MetalLfm2Model` per
/// Session: their KV caches and scratch are still shared and the
/// lock just turns a races-to-corruption into a serial bottleneck.
///
/// `GpuLfm2Model` (wgpu) carries the same `infer_lock` for the same
/// reason — its per-instance scratch buffers and GPU KV caches share
/// the same shape. CPU `Lfm2Model` has no such shared state and is
/// safely shareable across concurrent Sessions without any lock.
pub trait Model: Send + Sync {
    /// Run a forward pass for a single token and return logits over the vocabulary.
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32>;

    /// Batched forward pass for prefill: process all prompt tokens at once.
    /// Implementations may use GEMM for linear projections. Returns logits for the LAST token only.
    /// Default: falls back to sequential single-token `forward()` calls.
    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        // Default: fall back to sequential single-token forward
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            logits = self.forward(&[token], start_pos + i, state);
        }
        logits
    }

    /// Cancelable chunked prefill. Splits `tokens` into `ubatch`-sized slices,
    /// calls [`Self::forward_prefill`] per chunk, and polls `cancel` between
    /// chunks so long prompts can be interrupted without blocking the
    /// caller for the full monolithic duration.
    ///
    /// Returns `(tokens_processed, last_logits)`:
    /// - `tokens_processed <= tokens.len()`; when cancel fires, equals the
    ///   number of tokens that made it into KV before the flag was
    ///   observed (granularity: one ubatch).
    /// - `last_logits` holds the logits from the final processed chunk —
    ///   `Some` whenever any chunk ran. `None` only for the empty-input
    ///   edge case (`tokens.is_empty()`).
    ///
    /// Default impl is correctness-preserving; backend-specific overrides
    /// are free to batch across chunks (none do in v1 — Phase 1.4's
    /// deliberate "probably not in v1" scope). `ubatch == 0` means "no
    /// chunking" (one chunk covering the whole input); this matches the
    /// CLI `--ubatch-size 0` convention for disabling chunking.
    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        ubatch: usize,
        cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        // `ubatch == 0` → one chunk covering everything (no chunking).
        // Otherwise keep the caller-supplied size.
        let ubatch = if ubatch == 0 {
            tokens.len().max(1)
        } else {
            ubatch
        };
        let mut consumed = 0usize;
        let mut last_logits: Option<Vec<f32>> = None;
        for chunk in tokens.chunks(ubatch) {
            let logits = self.forward_prefill(chunk, start_pos + consumed, state);
            consumed += chunk.len();
            last_logits = Some(logits);
            // Check *after* each chunk so we always make progress on at
            // least one ubatch — avoids the "cancel-before-start leaves
            // the session wedged with no position advance" corner.
            if cancel.load(Ordering::Relaxed) && consumed < tokens.len() {
                break;
            }
        }
        (consumed, last_logits)
    }

    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;

    /// Does this backend support `n_keep` context shift? Static
    /// capability probe — callers MUST check this before invoking
    /// [`Self::shift_kv`].
    ///
    /// The default is `false` so new backends opt in deliberately.
    /// RoPE-based models override to `true` across their backends: the
    /// CPU path re-rotates the KV cache on-CPU (`shift_kv_with_rope`,
    /// used by both `Lfm2Model` and `LlamaModel`), while the LFM2 GPU
    /// backends do a shader-based GPU-side shift (Metal `kv_shift.metal`,
    /// wgpu `kv_shift.wgsl`). Non-RoPE architectures stay `false` — the
    /// shift semantics differ per positional-encoding scheme.
    fn supports_kv_shift(&self) -> bool {
        false
    }

    /// Execute a `n_keep` context shift on this model's state. Drops
    /// attention KV cells `[n_keep .. n_keep + shift)` and re-rotates
    /// remaining K vectors so their RoPE-encoded position matches
    /// their new index. Implemented by overriding; the default is a
    /// no-op, consistent with the default `false` from
    /// [`Self::supports_kv_shift`].
    ///
    /// Callers (today: `Session::append_tokens`) MUST verify
    /// `supports_kv_shift()` is `true` before invoking this. Calling
    /// the default no-op on an overflowed state would leave
    /// `InferenceState` unchanged while the caller proceeds as if a
    /// shift happened — a silent corruption bug.
    fn shift_kv(&self, _state: &mut InferenceState, _n_keep: usize, _shift: usize) {}

    /// Run a forward pass and return the hidden state BEFORE logit projection.
    /// Used by the audio decoder to extract the LLM embedding for audio frame sampling.
    /// Default: panics (must be overridden by backends that support audio).
    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        let _ = tokens;
        unimplemented!("forward_embedding not supported by this backend")
    }

    /// Static capability probe: does this backend implement
    /// [`Self::forward_from_embedding`] (and the related
    /// `forward_*_from_embedding` family)? Default `false` so new
    /// backends opt in deliberately. Callers (today:
    /// `Session::append_embeddings`) MUST consult this before
    /// invoking the embedding-input methods so unsupported backends
    /// surface a typed error instead of the default `unimplemented!`
    /// panic.
    fn supports_embedding_input(&self) -> bool {
        false
    }

    /// Forward pass with a float embedding as input (instead of a token ID).
    /// Used to feed audio codec embeddings back into the LLM after an audio frame.
    /// Default: panics (must be overridden by backends that support audio).
    fn forward_from_embedding(
        &self,
        _embedding: &[f32],
        _pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        unimplemented!("forward_from_embedding not supported by this backend")
    }

    /// Forward pass with embedding input, returning hidden state (not logits).
    /// Used in audio mode: embedding → layers → hidden state → sample audio → embed → loop.
    fn forward_hidden_from_embedding(
        &self,
        _embedding: &[f32],
        _pos: usize,
        _state: &mut InferenceState,
    ) -> Vec<f32> {
        unimplemented!("forward_hidden_from_embedding not supported by this backend")
    }

    /// Batched forward pass for prefill from raw embeddings (instead of token
    /// IDs). Mirrors [`Self::forward_prefill`] but accepts a row-major embedding
    /// buffer (`embeddings.len() == n_tokens * hidden_size`, frame `j` at
    /// `[j * hs .. (j + 1) * hs]`) so audio / vision / soft-token inputs avoid
    /// the per-frame `forward_from_embedding` loop. Returns logits for the LAST
    /// frame only.
    ///
    /// Capability: gated by [`Self::supports_embedding_input`] (same probe as
    /// `forward_from_embedding`). Callers MUST consult that probe; the default
    /// impl below relies on `forward_from_embedding`, which itself panics when
    /// unsupported.
    ///
    /// Default impl loops [`Self::forward_from_embedding`] per frame —
    /// preserves correctness for backends that haven't overridden but
    /// gives no perf win. Backends with a true batched path (CPU
    /// `Lfm2Model`) override to share their `forward_prefill` layer
    /// loop.
    ///
    /// Panics on `n_tokens == 0` or shape mismatch (`embeddings.len()
    /// != n_tokens * hidden_size`). The Session-level caller pre-validates
    /// both, so panics here indicate a bug in a non-Session caller.
    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let hidden_size = self.config().hidden_size;
        assert!(
            n_tokens > 0,
            "forward_prefill_from_embeddings requires at least one frame"
        );
        assert_eq!(
            embeddings.len(),
            n_tokens * hidden_size,
            "embeddings.len() ({}) != n_tokens ({}) * hidden_size ({})",
            embeddings.len(),
            n_tokens,
            hidden_size
        );
        let mut logits = Vec::new();
        for i in 0..n_tokens {
            let frame = &embeddings[i * hidden_size..(i + 1) * hidden_size];
            logits = self.forward_from_embedding(frame, start_pos + i, state);
        }
        logits
    }

    /// Static capability probe: does this backend implement
    /// [`Self::hidden_states`]? Default `false`; text backends opt in so an
    /// unsupported backend surfaces a typed error instead of the default panic.
    fn supports_hidden_states(&self) -> bool {
        false
    }

    /// Run a forward pass over `tokens` and return the **per-token** last-layer
    /// hidden state AFTER the final RMSNorm — the exact vector fed to the LM
    /// head, matching llama.cpp `llama_get_embeddings_ith` with pooling `NONE`.
    /// Downstream classifiers mean-pool this and run their own head.
    ///
    /// Output is flattened row-major `[n_tokens * hidden_size]` (token `t`,
    /// channel `c` at `t * hidden_size + c`). Logits are NOT computed.
    ///
    /// `state` is a caller-owned throwaway scratch (the Session hands in a
    /// reused, prompt-sized [`InferenceState::for_prefill`], cleared before the
    /// call); this method starts from position 0 and does not touch any
    /// generation KV.
    ///
    /// Default: panics; gated by [`Self::supports_hidden_states`].
    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        let _ = (tokens, state);
        unimplemented!("hidden_states not supported by this backend")
    }

    /// Greedy (argmax) fast path. Returns just the selected token id,
    /// avoiding a full logits readback when the caller only needs argmax.
    ///
    /// Default impl falls back to `forward()` + CPU argmax. Backends with
    /// a GPU argmax kernel should override to skip the vocab-sized readback.
    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let logits = self.forward(tokens, pos, state);
        crate::sampler::argmax(&logits)
    }

    /// GPU memory allocated by this model (bytes). 0 for CPU-only backends.
    fn gpu_memory_bytes(&self) -> u64 {
        0
    }

    /// Configure the KV prefix cache. No-op for backends without caching.
    fn configure_cache(&self, _config: crate::kv_cache::KvCacheConfig) {}

    /// Snapshot the current KV and conv state for prefix caching.
    ///
    /// Implemented by GPU backends whose state lives on the model
    /// instance (`MetalLfm2Model`, `GpuLfm2Model`) — they take
    /// `infer_lock` then delegate to a private `_locked` body that
    /// reads GPU buffers into byte vectors.
    ///
    /// **Not implemented by CPU `Lfm2Model`** — its state lives on
    /// the caller's `InferenceState`, not on the model, so the
    /// argument-less trait signature can't be honored. CPU
    /// consumers should call `InferenceState::snapshot` directly
    /// (added in PR #119); the prefix cache integration inside
    /// `Lfm2Model::forward_prefill` does this internally without
    /// going through the trait.
    fn snapshot_state(&self) -> crate::kv_cache::StateSnapshot {
        unimplemented!("snapshot_state not supported by this backend")
    }

    /// Restore a previously snapshotted state. Sets internal seq_len.
    ///
    /// Same backend asymmetry as [`Self::snapshot_state`]: GPU
    /// backends override + lock internally; CPU's
    /// `InferenceState::restore` is the equivalent caller-side
    /// API.
    fn restore_state(&self, _snapshot: &crate::kv_cache::StateSnapshot) {
        unimplemented!("restore_state not supported by this backend")
    }

    /// Whether this model/backend supports TurboQuant KV cache compression.
    /// Used by the CLI to decide whether to request compression or fall back
    /// to f32. TurboQuant is fully driven by `KvCompression` on the
    /// `InferenceState`; models just need to honor the compressed buffers in
    /// their forward pass. Currently only the CPU `Lfm2Model` does.
    fn turboquant_supported(&self) -> bool {
        false
    }
}

/// Load a model from a GGUF file, dispatching on the architecture.
///
/// `context_size` caps the model's `max_seq_len` and determines KV cache
/// pre-allocation in `InferenceState::from_config_with_compression`. Smaller
/// values reduce startup memory; larger values allow longer prompts/decodes.
///
/// `path` (when supplied) is used as the model identifier for prefix-cache
/// namespacing. `None` is the path-less `from_bytes` case — warm cache works
/// but disk-cache files would namespace-collide between distinct models.
pub fn load_model(
    gguf: GgufFile,
    path: Option<&std::path::Path>,
    context_size: usize,
) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let model_id = path
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(lfm2::Lfm2Model::from_gguf_with_id(
            gguf,
            context_size,
            model_id,
        )?)),
        // Classic Mistral ships as arch "llama" (the `"mistral"` GGUF arch
        // string does not exist in llama.cpp; Mistral 3.x/4.x are the distinct
        // "mistral3"/"mistral4" archs with different layouts, not served here).
        "qwen2" | "qwen3" | "llama" | "granite" => Ok(Box::new(
            llama::LlamaModel::from_gguf_with_id(gguf, context_size, model_id)?,
        )),
        other => bail!("unsupported architecture: {other}"),
    }
}

/// Load a model with GPU acceleration.
///
/// `path` (when supplied) is used as the model identifier for prefix-cache
/// namespacing. `None` is the path-less from_bytes case — warm cache works
/// but disk-cache files would namespace-collide between distinct models.
#[cfg(feature = "gpu")]
pub fn load_model_gpu(
    gguf: GgufFile,
    path: Option<&std::path::Path>,
    context_size: usize,
) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let model_id = path
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
            gguf,
            context_size,
            model_id,
        )?)),
        // Dense transformers share the generalized wgpu loader (per-arch rope /
        // QK-norm / QKV-bias / untied-output / Granite scalars are driven by the
        // GpuWeightSource accessors). Mirrors the CPU `load_model` allow-list.
        "qwen2" | "qwen3" | "llama" | "granite" => Ok(Box::new(
            gpu_lfm2::GpuLfm2Model::from_llama_with_id(gguf, context_size, model_id)?,
        )),
        other => bail!("unsupported architecture for GPU: {other}"),
    }
}

/// Load a model with native Metal acceleration.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub fn load_model_metal(
    gguf: GgufFile,
    path: &std::path::Path,
    context_size: usize,
) -> Result<Box<dyn Model>> {
    let arch = gguf
        .get_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    match arch.as_str() {
        "lfm2" => Ok(Box::new(metal_lfm2::MetalLfm2Model::from_gguf(
            gguf,
            path,
            context_size,
        )?)),
        // Dense transformers share the generalized Metal forward path.
        "qwen2" | "qwen3" | "llama" | "granite" => Ok(Box::new(
            metal_lfm2::MetalLfm2Model::from_llama(gguf, path, context_size)?,
        )),
        other => bail!("unsupported architecture for Metal: {other}"),
    }
}
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_saturating_arithmetic,
    unused_variables
)]
pub mod audio_decoder;
pub mod audio_encoder;
pub mod audio_preprocessor;
pub mod vision_encoder;
pub mod vision_encoder_gpu;
#[cfg(feature = "vl-preprocess")]
pub mod vision_preprocessor;
pub mod weights;

// Compile-time proof that `Arc<dyn Model>` is `Send + Sync`. If a new
// backend impl introduces a non-`Sync` field (e.g. a `RefCell` / `Cell`),
// this assertion fires at lib-build time with a clear pointer at the
// invariant, instead of the regression surfacing at a downstream FFI
// crate's build that doesn't have enough context to explain the error.
#[allow(dead_code)]
fn _assert_arc_dyn_model_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<std::sync::Arc<dyn Model>>();
}

use std::cell::Cell;
use std::collections::HashMap;
#[cfg(feature = "disk-cache")]
use std::path::Path;
use std::path::PathBuf;

use crate::CeraError;
use crate::model::{BlockType, ModelConfig};
use crate::time::Instant;
use crate::turboquant::{
    CompressedKeyCache, CompressedValueCache, EncodeScratch, QueryRotationScratch, RotationState,
    TurboQuantConfig,
};

/// Reserve capacity for `len` values of `T`, returning [`CeraError::OutOfMemory`]
/// instead of aborting the process when the allocation can't be satisfied. Used
/// for the config-driven KV-cache buffers — the uncompressed per-layer f32
/// caches (`capacity * kv_dim`) and the per-head compressed buffers under
/// TurboQuant — the dominant allocations that OOM when a model's context is too
/// large for the device. Returns an empty `Vec` with the capacity reserved
/// (filled during inference), matching `Vec::with_capacity`.
///
/// `TryReserveError::CapacityOverflow` (a request larger than the allocator can
/// ever satisfy) is intentionally folded into `OutOfMemory` alongside a true
/// allocation failure: both mean "this KV won't fit," the caller's recovery is
/// identical (skip the model), and `requested_bytes` reports the attempted size.
/// Callers that derive `len` from a multiplication guard that multiply first
/// via [`checked_elems`] (both the f32 KV path and the compressed buffers) so a
/// `usize` wrap can't silently under-reserve; `len` here is a valid element
/// count, and `try_reserve_exact` itself covers the `len * size_of::<T>()`
/// byte-size overflow.
pub(crate) fn try_alloc<T>(len: usize) -> Result<Vec<T>, CeraError> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| CeraError::OutOfMemory {
            requested_bytes: (len as u64).saturating_mul(std::mem::size_of::<T>() as u64),
        })?;
    Ok(v)
}

/// `count * per` (an element count) with overflow guarded — a `usize` wrap would
/// silently under-reserve the buffer and reintroduce an infallible realloc, so
/// map overflow to `OutOfMemory` (the intended size is absurd) rather than let
/// it slip past [`try_alloc`]. Used where a buffer length is `capacity * per`.
///
/// `T` is the element type the resulting length feeds into `try_alloc::<T>`, so
/// the `OutOfMemory` diagnostic reports the intended **byte** size
/// (`count * per * size_of::<T>()`, saturating), consistent with `try_alloc`'s
/// own `requested_bytes` — not a bare element count.
pub(crate) fn checked_elems<T>(count: usize, per: usize) -> Result<usize, CeraError> {
    count.checked_mul(per).ok_or(CeraError::OutOfMemory {
        requested_bytes: (count as u64)
            .saturating_mul(per as u64)
            .saturating_mul(std::mem::size_of::<T>() as u64),
    })
}

/// A `fill`-initialized buffer of length `len`, reserved fallibly. Reserves via
/// [`try_alloc`] (→ [`CeraError::OutOfMemory`] instead of aborting) and then
/// `resize`s to fill within that reservation, so no further allocation occurs.
/// Used for the fixed-size scratch/conv buffers so every allocation in the
/// constructor is recoverable, not just the context-scaled KV caches.
pub(crate) fn zeroed<T: Clone>(len: usize, fill: T) -> Result<Vec<T>, CeraError> {
    let mut v = try_alloc::<T>(len)?;
    v.resize(len, fill);
    Ok(v)
}

/// [`zeroed`] specialized to zero-filled `f32` scratch buffers.
fn zeroed_f32(len: usize) -> Result<Vec<f32>, CeraError> {
    zeroed(len, 0.0)
}

/// KV cache compression mode. Passed to `InferenceState::from_config_with_compression`
/// (or via `GenerateConfig::kv_compression`) — that single call sets up everything
/// TurboQuant needs: the per-layer rotation states, the compressed key/value
/// caches, and the scratch buffers. **No separate `enable_turboquant` call on
/// the model is required.**
///
/// TurboQuant is honored by the CPU backend (`Lfm2Model`) and by both GPU
/// backends (`GpuLfm2Model` and `MetalLfm2Model`). The GPU paths additionally
/// need [`crate::model::Model::configure_kv_compression`] — which `Session`
/// calls — to build their GPU-resident compressed caches, and they only
/// implement the both-sides mode: a single-sided (debug) request, or a
/// `head_dim` their kernels can't handle, warns and falls back to the backend's
/// uncompressed KV (f32 on wgpu, f16 on Metal).
#[derive(Clone, Debug, Default)]
pub enum KvCompression {
    /// No compression — the backend's uncompressed KV: f32 on CPU and wgpu,
    /// f16 on native Metal, whose cache has always been half precision.
    #[default]
    None,
    /// f16 KV cache — keys and values stored as IEEE-754 half precision
    /// (2 bytes/elem instead of 4), halving the KV bytes streamed per decode
    /// token. Near-lossless (f16 has 10 mantissa bits; attention is robust to
    /// it — this is what llama.cpp uses by default). CPU LFM2 and dense
    /// transformer paths; accumulation stays f32 for softmax stability.
    F16,
    /// TurboQuant compression. Keys and values can be toggled independently
    /// for debugging (e.g. to isolate how much drift each side contributes).
    /// The common production configuration sets both `keys` and `values` to
    /// `true`.
    ///
    /// - Keys: 2-bit PolarQuant + 1-bit QJL residual (3 bits/elem + f16 norms).
    /// - Values: 2-bit PolarQuant only (2 bits/elem + f16 norms).
    ///
    /// `seed` drives the per-layer randomized Hadamard rotations — the same
    /// seed reproduces the same rotations deterministically.
    TurboQuant { seed: u64, keys: bool, values: bool },
}

impl KvCompression {
    /// Shortcut for the common "compress everything" configuration.
    pub fn turboquant(seed: u64) -> Self {
        Self::TurboQuant {
            seed,
            keys: true,
            values: true,
        }
    }

    /// Returns `(compress_keys, compress_values)` for the current mode.
    /// f16 is not TurboQuant "compression" in this sense — it reports `false`
    /// so the TurboQuant setup paths stay off; the f16 slots are driven by
    /// [`Self::is_f16`] instead.
    pub fn flags(&self) -> (bool, bool) {
        match self {
            Self::None | Self::F16 => (false, false),
            Self::TurboQuant { keys, values, .. } => (*keys, *values),
        }
    }

    /// Whether the KV cache is stored in f16 (half precision).
    pub fn is_f16(&self) -> bool {
        matches!(self, Self::F16)
    }

    /// The mode a state built from `config` will *actually* use.
    ///
    /// `from_config_capped` silently downgrades TurboQuant to plain f32 when
    /// `head_dim` isn't a power of two (the Walsh-Hadamard transform needs it), so
    /// anything deriving identity from the mode — the prefix-cache tag especially —
    /// has to ask what was resolved, not what was requested. Keeping that rule here
    /// beside the allocator that applies it stops the two from drifting.
    pub fn resolved_for(&self, config: &ModelConfig) -> Self {
        let (keys, values) = self.flags();
        if (keys || values) && !config.head_dim.is_power_of_two() {
            Self::None
        } else {
            self.clone()
        }
    }

    /// Prefix-cache namespace tag for this mode — `""` for plain f32, else a
    /// `"…:"`-terminated discriminator to prepend to a backend's cache id.
    ///
    /// The KV prefix cache's disk tier is keyed by model path (plus a backend
    /// prefix), so without this every KV mode over the same GGUF and
    /// `--cache-dir` shares one namespace. A snapshot from one mode then
    /// permanently shadows another's for the same prefix: the restore-time
    /// compatibility gate turns the longest match into a miss and does *not* fall
    /// back to a shorter compatible entry, so the other mode stays cold on every
    /// subsequent run, not just once. (Measured on wgpu: 134.7 → 10.3 tok/s
    /// prefill, sticky.)
    ///
    /// Two things beyond the mode name are load-bearing:
    /// - **The seed**, because it drives the per-layer randomized Hadamard
    ///   rotations. Restore validates only shape (`head_dim` / `n_kv_heads` /
    ///   `seq_len`), which a different-seed blob passes — so a shared namespace
    ///   would decode a prefix in the wrong basis and silently corrupt attention
    ///   rather than miss.
    /// - **Which sides are compressed**, since keys-only and values-only produce
    ///   genuinely different cache contents from both-sides and from f32.
    ///
    /// A TurboQuant config compressing neither side is *not* tagged: it degrades
    /// to an f32 cache in `from_config_capped`, so its contents really are f32's
    /// and separating them would only waste entries.
    ///
    /// Callers must tag with the mode their state actually ended up in, not the one
    /// requested — a backend that falls back to its uncompressed KV (unsupported
    /// `head_dim`, or a single-sided request on GPU) must use the uncompressed
    /// tag so it shares the namespace it is now writing into. Metal's
    /// uncompressed cache is f16 but still takes `None`'s tag; see the note in
    /// `MetalLfm2Model::configure_kv_compression`.
    pub fn cache_tag(&self) -> String {
        match self {
            Self::None => String::new(),
            Self::F16 => "f16:".to_string(),
            Self::TurboQuant { seed, keys, values } => match (keys, values) {
                (false, false) => String::new(),
                (true, true) => format!("tq3kv-s{seed}:"),
                (true, false) => format!("tq3k-s{seed}:"),
                (false, true) => format!("tq3v-s{seed}:"),
            },
        }
    }
}

/// Per-layer inference state.
/// Capacity for the Conv rollback ring buffer (in number of tokens).
pub const CONV_HISTORY_CAPACITY: usize = 64;

/// Zero-allocation flat ring buffer storing recent convolution state snapshots
/// for speculative decoding rollback.
#[derive(Clone, Debug)]
pub struct ConvHistory {
    snapshots: Vec<f32>,
    positions: [usize; CONV_HISTORY_CAPACITY],
    head: usize,
    count: usize,
    buf_len: usize,
}

impl ConvHistory {
    /// Create a new pre-allocated history ring buffer for a convolution layer with `buf_len` elements.
    pub fn new(buf_len: usize) -> Self {
        Self {
            snapshots: vec![0.0f32; CONV_HISTORY_CAPACITY * buf_len],
            positions: [0; CONV_HISTORY_CAPACITY],
            head: 0,
            count: 0,
            buf_len,
        }
    }

    /// Record a snapshot of `buffer` at sequence position `pos`.
    /// Zero heap allocation: copies into the pre-allocated flat storage.
    pub fn push(&mut self, pos: usize, buffer: &[f32]) {
        if self.buf_len == 0 {
            return;
        }
        debug_assert_eq!(buffer.len(), self.buf_len);
        let offset = self.head * self.buf_len;
        self.snapshots[offset..offset + self.buf_len].copy_from_slice(buffer);
        self.positions[self.head] = pos;
        self.head = (self.head + 1) % CONV_HISTORY_CAPACITY;
        if self.count < CONV_HISTORY_CAPACITY {
            self.count += 1;
        }
    }

    /// Roll back the convolution `buffer` to the state at `target_pos`.
    /// Returns `true` if the state was successfully found and restored.
    pub fn rollback_to(&mut self, target_pos: usize, buffer: &mut [f32]) -> bool {
        if target_pos == 0 {
            buffer.fill(0.0);
            self.clear();
            return true;
        }
        for i in 0..self.count {
            let idx = (self.head + CONV_HISTORY_CAPACITY - 1 - i) % CONV_HISTORY_CAPACITY;
            if self.positions[idx] == target_pos {
                let offset = idx * self.buf_len;
                buffer.copy_from_slice(&self.snapshots[offset..offset + self.buf_len]);
                self.head = (idx + 1) % CONV_HISTORY_CAPACITY;
                self.count = self.count.saturating_sub(i);
                return true;
            }
        }
        false
    }

    /// Reset all snapshot tracking without deallocating.
    pub fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}

#[allow(clippy::large_enum_variant)]
pub enum LayerState {
    /// KV cache for attention layers.
    Attention {
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
        /// f16 key cache (IEEE-754 half bits). Populated when `KvCompression::F16`
        /// is active; `key_cache` stays empty in that mode. Time-major
        /// `[seq_len × kv_dim]`, same layout as `key_cache`.
        key_cache_f16: Vec<u16>,
        /// f16 value cache (IEEE-754 half bits). Populated when `KvCompression::F16`
        /// is active; `value_cache` stays empty in that mode.
        value_cache_f16: Vec<u16>,
        /// Compressed key cache (populated when TurboQuant is active; key_cache stays empty).
        compressed_keys: Option<CompressedKeyCache>,
        /// Compressed value cache (populated when TurboQuant is active; value_cache stays empty).
        compressed_values: Option<CompressedValueCache>,
    },
    /// Rolling buffer for convolution layers.
    /// Stores previous `d_conv` pre-conv activations (bx values), time-major.
    Conv {
        buffer: Vec<f32>,
        /// Pre-allocated ring buffer history snapshots for speculative decoding rollback.
        history: ConvHistory,
    },
}

/// Pre-allocated scratch buffers reused across layers and tokens.
pub struct ScratchBuffers {
    /// Scratch for the normed hidden state (hidden_size).
    pub normed: Vec<f32>,
    /// Scratch for FFN input (hidden_size).
    pub ffn_input: Vec<f32>,
    /// Scratch for shortconv in_proj output (3 * hidden_size).
    pub conv_proj: Vec<f32>,
    /// Scratch for shortconv bx / conv output (hidden_size).
    pub conv_scratch: Vec<f32>,
    /// Scratch for Q projection (hidden_size = n_heads * head_dim).
    pub q: Vec<f32>,
    /// Scratch for K projection (max kv_dim).
    pub k: Vec<f32>,
    /// Scratch for V projection (max kv_dim).
    pub v: Vec<f32>,
    /// Scratch for attention output (hidden_size).
    pub attn_out: Vec<f32>,
    /// Scratch for FFN gate (intermediate_size).
    pub gate: Vec<f32>,
    /// Scratch for FFN up (intermediate_size).
    pub up: Vec<f32>,
    /// Scratch for block/FFN output (hidden_size).
    pub out: Vec<f32>,
    /// Scratch for attention scores (grows with seq_len). Reused across heads
    /// when the decode head loop runs serially; when it runs on the pool,
    /// `transformer::decode_attention` re-lays it out as one
    /// `seq_len + head_dim` row per head so the heads don't share a buffer.
    pub scores: Vec<f32>,
    /// Q8_0 quantization scratch: scales for the input vector (max_k / 32 entries).
    pub q8_scales: Vec<f32>,
    /// Q8_0 quantization scratch: quants for the input vector (max_k entries).
    pub q8_quants: Vec<i8>,
    /// Q8_0 quantization scratch for MoE down-projections.
    pub q8_scales_down: Vec<f32>,
    /// Q8_0 quantization scratch for MoE down-projections.
    pub q8_quants_down: Vec<i8>,
    /// Dequantized weight scratch for BLAS prefill. Grown lazily on first use
    /// to the largest weight matrix the BLAS path encounters; reused across
    /// every subsequent GEMM call within and between forward passes. Stays
    /// empty when the `blas` feature is off — the NEON fallback never touches it.
    pub dequant_weight_scratch: Vec<f32>,
    /// Scratch for the LoRA down-projection intermediate (`A·x`). Reused across
    /// apply calls so the LoRA hot path allocates nothing.
    pub lora_tmp: Vec<f32>,
    /// Router logits, then gate probabilities, for one MoE layer (`n_expert`).
    /// Stays empty on dense models, which never touch it.
    pub moe_probs: Vec<f32>,
    /// One expert's feed-forward output (`hidden_size`), before it is scaled by
    /// its routing weight and accumulated. Needed because the down-projection
    /// GEMV overwrites its destination rather than accumulating, so the running
    /// sum cannot live in `out`. Empty on dense models.
    pub moe_expert_out: Vec<f32>,
    /// The `n_expert_used` selected expert indices and their normalized
    /// combining weights, for one MoE layer. Held in scratch so routing
    /// allocates nothing per layer per token. Empty on dense models.
    pub moe_selected: Vec<(usize, f32)>,
    /// Scratch for LM Head output logits (vocab_size). Reused across tokens to eliminate per-token heap allocation.
    pub logits: Vec<f32>,
    /// Scratch for token embedding lookup / hidden state input (hidden_size).
    pub hidden_in: Vec<f32>,
}

/// Inference state across all layers.
pub struct InferenceState {
    pub layers: Vec<LayerState>,
    pub seq_len: usize,
    pub scratch: ScratchBuffers,
    /// Active LoRA adapter for this pass, if any. The Session copies its
    /// attached adapter here before each forward; the CPU projection helpers
    /// read `lora.get(layer, target)` and add `scale·B·(A·x)` after each base
    /// GEMV. `None` ⇒ base model only.
    pub lora: Option<std::sync::Arc<crate::lora::LoraAdapterWeights>>,
    /// TurboQuant encode scratch (None when disabled). Owned by InferenceState
    /// rather than Model so the model remains Sync for concurrent inference.
    pub tq_encode_scratch: Option<EncodeScratch>,
    /// TurboQuant query rotation scratch (None when disabled).
    pub tq_query_scratch: Option<QueryRotationScratch>,
    /// Per-layer TurboQuant rotation state (None for conv layers or when
    /// compression is disabled). Constructed from the `seed` in `KvCompression`
    /// at `from_config_with_compression` time.
    pub tq_rotations: Vec<Option<RotationState>>,
    /// Shared TurboQuant config (Lloyd-Max centroids, derived from head_dim).
    pub tq_config: Option<TurboQuantConfig>,
    /// Whether the attention KV caches are stored in f16 (`KvCompression::F16`).
    /// Set once at construction; the CPU forward paths read it to choose the
    /// f16 append + f16-widening attention kernels over the f32 path. Reliable
    /// even before the first token (when both cache vecs are empty).
    pub kv_f16: bool,
}

impl InferenceState {
    /// Create a new empty inference state.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState::Attention {
                    key_cache: Vec::new(),
                    value_cache: Vec::new(),
                    key_cache_f16: Vec::new(),
                    value_cache_f16: Vec::new(),
                    compressed_keys: None,
                    compressed_values: None,
                })
                .collect(),
            seq_len: 0,
            scratch: ScratchBuffers {
                normed: Vec::new(),
                ffn_input: Vec::new(),
                conv_proj: Vec::new(),
                conv_scratch: Vec::new(),
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                attn_out: Vec::new(),
                gate: Vec::new(),
                up: Vec::new(),
                out: Vec::new(),
                scores: Vec::new(),
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
                q8_scales_down: Vec::new(),
                q8_quants_down: Vec::new(),
                dequant_weight_scratch: Vec::new(),
                lora_tmp: Vec::new(),
                moe_probs: Vec::new(),
                moe_expert_out: Vec::new(),
                moe_selected: Vec::new(),
                logits: Vec::new(),
                hidden_in: Vec::new(),
            },
            tq_encode_scratch: None,
            tq_query_scratch: None,
            tq_rotations: Vec::new(),
            tq_config: None,
            lora: None,
            kv_f16: false,
        }
    }

    /// Create inference state matching a model config.
    /// Attention layers get empty KV caches; conv layers get zero-filled rolling buffers.
    /// Scratch buffers are pre-allocated to avoid per-token allocations.
    pub fn from_config(config: &ModelConfig) -> Result<Self, CeraError> {
        Self::from_config_with_compression(config, &KvCompression::None)
    }

    /// Build a throwaway prefill state whose KV capacity is capped to
    /// `n_tokens` cells (never the full `max_seq_len`), for one-shot passes
    /// like hidden-state extraction. Uncompressed. Capacity is clamped to
    /// `[1, max_seq_len]`, so a per-chunk classifier call reserves ~O(T·kv_dim)
    /// rather than the hundreds of MB a full-context cache would.
    pub fn for_prefill(config: &ModelConfig, n_tokens: usize) -> Result<Self, CeraError> {
        let capacity = n_tokens.clamp(1, config.max_seq_len);
        Self::from_config_capped(config, &KvCompression::None, capacity)
    }

    /// Reset an existing state for reuse as a throwaway prefill scratch: zero
    /// `seq_len`, truncate the (uncompressed) KV caches to empty, and zero the
    /// conv rolling buffers in place (kept at full length) — all while KEEPING
    /// allocated capacity, so a reused scratch does no allocation. Intended for
    /// the [`Self::for_prefill`] path (no compression); working scratch buffers
    /// are resized on demand by the forward pass.
    pub fn clear_for_reuse(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.layers {
            match layer {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    key_cache.clear();
                    value_cache.clear();
                    key_cache_f16.clear();
                    value_cache_f16.clear();
                }
                LayerState::Conv { buffer, history } => {
                    buffer.fill(0.0);
                    history.clear();
                }
            }
        }
    }

    /// Create inference state with optional KV cache compression.
    ///
    /// When `compression` is `KvCompression::TurboQuant`, this single call
    /// sets up everything TurboQuant needs: the per-layer rotation states,
    /// the compressed caches (keys and/or values), the encode scratch, and
    /// the query rotation scratch. The model itself doesn't need to be
    /// "enabled" separately — it reads all TurboQuant state from here.
    pub fn from_config_with_compression(
        config: &ModelConfig,
        compression: &KvCompression,
    ) -> Result<Self, CeraError> {
        Self::from_config_capped(config, compression, config.max_seq_len)
    }

    /// Like [`Self::from_config_with_compression`] but caps the pre-allocated
    /// KV-cache capacity to `capacity` cells instead of `config.max_seq_len`.
    /// Backs [`Self::for_prefill`]. `capacity` should be pre-clamped to
    /// `[1, max_seq_len]`; appending beyond it just triggers normal Vec growth.
    pub(crate) fn from_config_capped(
        config: &ModelConfig,
        compression: &KvCompression,
        capacity: usize,
    ) -> Result<Self, CeraError> {
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        assert!(
            kernel_size >= 2,
            "conv_kernel_size must be at least 2, got {kernel_size}"
        );
        let d_conv = kernel_size - 1;
        // Carried explicitly (≠ hidden_size / n_heads for head_dim-decoupled
        // models like Qwen3). `q_dim` (Q projection width = attention output
        // width) can exceed hidden_size.
        let head_dim = config.head_dim;
        // Guard the config-derived scratch/conv buffer-length multiplies: a
        // `usize` wrap from a malformed GGUF would size a too-small buffer and
        // panic on a later index, so map overflow to a recoverable OutOfMemory
        // (same policy as the KV path). All of these size f32 buffers.
        let q_dim = checked_elems::<f32>(config.n_heads, head_dim)?;
        let max_kv_dim = checked_elems::<f32>(
            config.kv_heads_per_layer.iter().copied().max().unwrap_or(0),
            head_dim,
        )?;

        // Compressed (TurboQuant) caches start at the same per-layer cap as the
        // f32 path, so the compressed-side Vecs also avoid mid-decode reallocs.
        let initial_capacity = capacity;
        let (compress_keys, compress_values) = compression.flags();
        // f16 KV: the f32 slots stay empty and the `*_f16` slots hold the cache
        // (half the bytes). Independent of the TurboQuant `compress_*` axis.
        let use_f16 = compression.is_f16();

        // TurboQuant requires power-of-2 head_dim for the Walsh-Hadamard Transform.
        // If the requirement isn't met, silently fall back to uncompressed f32.
        let tq_enabled = (compress_keys || compress_values) && head_dim.is_power_of_two();
        let (compress_keys, compress_values) = if tq_enabled {
            (compress_keys, compress_values)
        } else {
            (false, false)
        };

        let (tq_rotations, tq_config) = if tq_enabled {
            let seed = match compression {
                KvCompression::TurboQuant { seed, .. } => *seed,
                KvCompression::None | KvCompression::F16 => 0,
            };
            // Reserve the outer Vec fallibly too (layer-count-scaled), so no
            // allocation on this path can abort — each RotationState is already
            // built via the fallible try_from_seed.
            let mut rotations = try_alloc::<Option<RotationState>>(config.block_types.len())?;
            for (layer_idx, bt) in config.block_types.iter().enumerate() {
                rotations.push(match bt {
                    BlockType::Attention => Some(RotationState::try_from_seed(
                        seed ^ layer_idx as u64,
                        head_dim,
                    )?),
                    BlockType::GatedConv => None,
                });
            }
            (rotations, Some(TurboQuantConfig::for_head_dim(head_dim)))
        } else {
            (Vec::new(), None)
        };
        // Reserve the outer Vec<LayerState> fallibly (layer-count-scaled), then
        // push each layer, so every allocation on this path is recoverable.
        let mut layers = try_alloc::<LayerState>(config.block_types.len())?;
        for layer in config.block_types.iter().enumerate().map(
            |(layer_idx, bt)| -> Result<LayerState, CeraError> {
                match bt {
                    BlockType::Attention => {
                        let n_kv_heads = config.kv_heads_per_layer[layer_idx];
                        // Guard both multiplies (a usize wrap would silently
                        // under-reserve). A config bug (e.g. wildly large
                        // max_seq_len, n_kv_heads, or head_dim from a malformed
                        // GGUF) surfaces as a recoverable OutOfMemory — same as
                        // a genuinely too-large KV — rather than aborting the
                        // process. kv_dim counts f32 slots per token, so its
                        // overflow guard uses the same f32-sized helper.
                        let kv_dim = checked_elems::<f32>(n_kv_heads, head_dim)?;
                        let kv_capacity = checked_elems::<f32>(capacity, kv_dim)?;
                        let compressed_keys = if compress_keys && n_kv_heads > 0 {
                            Some(CompressedKeyCache::try_new(
                                n_kv_heads,
                                head_dim,
                                initial_capacity,
                            )?)
                        } else {
                            None
                        };
                        let compressed_values = if compress_values && n_kv_heads > 0 {
                            Some(CompressedValueCache::try_new(
                                n_kv_heads,
                                head_dim,
                                initial_capacity,
                            )?)
                        } else {
                            None
                        };
                        // Pre-allocate the f32 KV cache to exactly
                        // `capacity * kv_dim` floats (capacity = the session's
                        // requested context, ≤ model max_seq_len) so writes never
                        // trigger Vec doubling/reallocation. Like every other
                        // allocation in this constructor it's fallible
                        // (`try_alloc`) — an over-large context returns
                        // `OutOfMemory` instead of aborting the process — but this
                        // is the dominant, context-scaled one. When TurboQuant
                        // compression is active for that side, the f32 vec stays
                        // empty and the compressed cache stores it.
                        // In f16 mode the f32 vecs stay empty and the `*_f16`
                        // vecs are pre-allocated to `capacity * kv_dim` u16 slots
                        // (half the bytes of the f32 path), same anti-realloc
                        // reservation. Otherwise the f16 vecs stay empty.
                        let key_cache = if (compress_keys && n_kv_heads > 0) || use_f16 {
                            Vec::new()
                        } else {
                            try_alloc::<f32>(kv_capacity)?
                        };
                        let value_cache = if (compress_values && n_kv_heads > 0) || use_f16 {
                            Vec::new()
                        } else {
                            try_alloc::<f32>(kv_capacity)?
                        };
                        let key_cache_f16 = if use_f16 && n_kv_heads > 0 {
                            try_alloc::<u16>(kv_capacity)?
                        } else {
                            Vec::new()
                        };
                        let value_cache_f16 = if use_f16 && n_kv_heads > 0 {
                            try_alloc::<u16>(kv_capacity)?
                        } else {
                            Vec::new()
                        };
                        Ok(LayerState::Attention {
                            key_cache,
                            value_cache,
                            key_cache_f16,
                            value_cache_f16,
                            compressed_keys,
                            compressed_values,
                        })
                    }
                    BlockType::GatedConv => {
                        let buf_len = checked_elems::<f32>(d_conv, config.hidden_size)?;
                        Ok(LayerState::Conv {
                            buffer: zeroed_f32(buf_len)?,
                            history: ConvHistory::new(buf_len),
                        })
                    }
                }
            },
        ) {
            layers.push(layer?);
        }

        Ok(Self {
            layers,
            seq_len: 0,
            scratch: ScratchBuffers {
                normed: zeroed_f32(config.hidden_size)?,
                ffn_input: zeroed_f32(config.hidden_size)?,
                conv_proj: zeroed_f32(checked_elems::<f32>(3, config.hidden_size)?)?,
                conv_scratch: zeroed_f32(config.hidden_size)?,
                q: zeroed_f32(q_dim)?,
                k: zeroed_f32(max_kv_dim)?,
                v: zeroed_f32(max_kv_dim)?,
                attn_out: zeroed_f32(q_dim)?,
                gate: zeroed_f32(config.intermediate_size)?,
                up: zeroed_f32(config.intermediate_size)?,
                out: zeroed_f32(config.hidden_size)?,
                scores: Vec::new(),    // grows with seq_len during inference
                q8_scales: Vec::new(), // resized per GEMV input dimension (max of hidden/intermediate)
                q8_quants: Vec::new(), // resized per GEMV input dimension
                q8_scales_down: Vec::new(),
                q8_quants_down: Vec::new(),
                // Grown lazily to max(3*hs*hs, is*hs) on the first BLAS GEMM
                // call. Stays empty if the `blas` feature is off.
                dequant_weight_scratch: Vec::new(),
                lora_tmp: Vec::new(),
                moe_probs: config
                    .moe
                    .as_ref()
                    .map(|m| zeroed_f32(m.n_expert))
                    .transpose()?
                    .unwrap_or_default(),
                moe_expert_out: config
                    .moe
                    .as_ref()
                    .map(|_| zeroed_f32(config.hidden_size))
                    .transpose()?
                    .unwrap_or_default(),
                moe_selected: config
                    .moe
                    .as_ref()
                    .map(|m| Vec::with_capacity(m.n_expert_used))
                    .unwrap_or_default(),
                logits: zeroed_f32(config.vocab_size)?,
                hidden_in: zeroed_f32(config.hidden_size)?,
            },
            // Scratch is needed whenever either side is compressed. The
            // EncodeScratch `rot` buffer is shared between key and value
            // encode; QueryRotationScratch is shared between key score
            // computation and value weighted-sum reconstruction.
            tq_encode_scratch: if tq_enabled {
                Some(EncodeScratch::try_new(head_dim)?)
            } else {
                None
            },
            tq_query_scratch: if tq_enabled {
                Some(QueryRotationScratch::try_new(config.n_heads, head_dim)?)
            } else {
                None
            },
            tq_rotations,
            tq_config,
            lora: None,
            kv_f16: use_f16,
        })
    }

    /// Append K and V vectors to an attention layer's cache (uncompressed path).
    pub fn append_kv(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut self.layers[layer]
        {
            key_cache.extend_from_slice(k);
            value_cache.extend_from_slice(v);
        }
    }

    /// Append K and V to an attention layer's f16 cache, converting each f32 to
    /// IEEE-754 half on the way in. Used when `KvCompression::F16` is active.
    pub fn append_kv_f16(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if let LayerState::Attention {
            key_cache_f16,
            value_cache_f16,
            ..
        } = &mut self.layers[layer]
        {
            key_cache_f16.extend(k.iter().map(|&x| crate::quant::f32_to_f16(x)));
            value_cache_f16.extend(v.iter().map(|&x| crate::quant::f32_to_f16(x)));
        }
    }

    /// Borrow the f16 key and value caches for an attention layer (IEEE-754 half
    /// bits, time-major `[seq_len × kv_dim]`). Panics on a non-attention layer.
    pub fn kv_cache_f16(&self, layer: usize) -> (&[u16], &[u16]) {
        if let LayerState::Attention {
            key_cache_f16,
            value_cache_f16,
            ..
        } = &self.layers[layer]
        {
            (key_cache_f16, value_cache_f16)
        } else {
            panic!("kv_cache_f16 called on non-attention layer {layer}");
        }
    }

    /// Borrow the key and value caches for an attention layer.
    /// The returned slices are laid out as [seq_len, kv_dim] (time-major).
    pub fn kv_cache(&self, layer: usize) -> (&[f32], &[f32]) {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &self.layers[layer]
        {
            (key_cache, value_cache)
        } else {
            panic!("kv_cache called on non-attention layer {layer}");
        }
    }

    /// Borrow the compressed key cache for an attention layer, if present.
    pub fn compressed_keys(&self, layer: usize) -> Option<&CompressedKeyCache> {
        if let LayerState::Attention {
            compressed_keys, ..
        } = &self.layers[layer]
        {
            compressed_keys.as_ref()
        } else {
            None
        }
    }

    /// Mutably borrow the compressed key cache for an attention layer, if present.
    pub fn compressed_keys_mut(&mut self, layer: usize) -> Option<&mut CompressedKeyCache> {
        if let LayerState::Attention {
            compressed_keys, ..
        } = &mut self.layers[layer]
        {
            compressed_keys.as_mut()
        } else {
            None
        }
    }

    /// Is any attention layer's KV currently backed by a compressed
    /// (TurboQuant) cache? Used by `Session::append_tokens` to decide
    /// whether `n_keep` shift is supported for this state — v1 gates
    /// shift on uncompressed caches only.
    /// `true` iff *every* attention layer has BOTH
    /// `compressed_keys` and `compressed_values` populated. Used by
    /// the prefix-cache lookup gate to distinguish a fully-
    /// TurboQuant state (matchable against
    /// `LayerSnapshot::AttentionCompressed`) from a mixed-mode one
    /// (no snapshot variant fits — `snapshot()` returns `None`).
    pub fn is_fully_compressed(&self) -> bool {
        self.layers.iter().all(|l| match l {
            LayerState::Attention {
                compressed_keys,
                compressed_values,
                ..
            } => compressed_keys.is_some() && compressed_values.is_some(),
            // Conv layers are never compressed; they don't impact
            // the "fully compressed" determination.
            LayerState::Conv { .. } => true,
        })
    }

    pub fn is_compressed(&self) -> bool {
        self.layers.iter().any(|l| {
            matches!(
                l,
                LayerState::Attention {
                    compressed_keys: Some(_),
                    ..
                } | LayerState::Attention {
                    compressed_values: Some(_),
                    ..
                }
            )
        })
    }

    /// Capture the current inference state as a `StateSnapshot` suitable
    /// for the KV prefix cache. CPU-flavored: f32 KV caches are byte-cast
    /// via `bytemuck`; conv buffers are byte-cast wholesale. Compressed
    /// (TurboQuant) attention layers are encoded into the
    /// `LayerSnapshot::AttentionCompressed { keys, values }` byte slots
    /// via `turboquant::encode_compressed_*`.
    ///
    /// Returns `None` when an attention layer's compression is
    /// **mixed** — i.e. exactly one of `compressed_keys` /
    /// `compressed_values` is `Some`. The on-disk
    /// `AttentionCompressed { keys, values }` shape models both
    /// sides as encoded blobs; a single-side-compressed layer
    /// would lose the uncompressed side's f32 data on snapshot.
    /// `KvCompression::TurboQuant { keys: bool, values: bool }`
    /// has both bools as debug knobs; the production config sets
    /// both to `true`. Treating mixed as not-snapshotted is
    /// conservative + correct — caller falls back to a cold prefill
    /// for that turn.
    pub fn snapshot(&self) -> Option<StateSnapshot> {
        let mut layers = Vec::with_capacity(self.layers.len());
        // `kv_f16` is the authoritative mode flag (set once at construction),
        // reliable even for a zero-token layer whose `*_f16` slots are still
        // empty — so every attention layer of an f16 state snapshots as
        // `AttentionF16`, matching the `is_f16()` restore gate.
        let kv_f16 = self.kv_f16;
        for l in &self.layers {
            match l {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    compressed_keys,
                    compressed_values,
                } => {
                    let snap = if kv_f16 {
                        // f16 KV: serialize the u16 half-bits **little-endian**
                        // to match the documented on-disk format and `restore`'s
                        // `u16::from_le_bytes` decode (a native-endian
                        // `bytemuck::cast_slice` would byte-swap on a big-endian
                        // host). f16 and TurboQuant are mutually exclusive
                        // (`from_config_with_compression` picks one), so the
                        // compressed slots are always `None` here.
                        LayerSnapshot::AttentionF16 {
                            k_data: key_cache_f16.iter().flat_map(|h| h.to_le_bytes()).collect(),
                            v_data: value_cache_f16
                                .iter()
                                .flat_map(|h| h.to_le_bytes())
                                .collect(),
                        }
                    } else {
                        match (compressed_keys, compressed_values) {
                            (None, None) => LayerSnapshot::Attention {
                                k_data: bytemuck::cast_slice(key_cache).to_vec(),
                                v_data: bytemuck::cast_slice(value_cache).to_vec(),
                            },
                            (Some(k), Some(v)) => LayerSnapshot::AttentionCompressed {
                                keys: crate::turboquant::encode_compressed_keys(k),
                                values: crate::turboquant::encode_compressed_values(v),
                            },
                            // Mixed-mode: refuse the whole snapshot.
                            (Some(_), None) | (None, Some(_)) => return None,
                        }
                    };
                    layers.push(snap);
                }
                LayerState::Conv { buffer, .. } => layers.push(LayerSnapshot::Conv {
                    buffer: bytemuck::cast_slice(buffer).to_vec(),
                }),
            }
        }
        Some(StateSnapshot::new(layers, self.seq_len))
    }

    /// Restore a previously captured `StateSnapshot` into this state's
    /// f32 caches (or compressed caches for TurboQuant layers). Inverse
    /// of [`Self::snapshot`]. Asserts that the snapshot's layer count
    /// matches; f32 byte-length must be a multiple of 4 (one f32 per
    /// 4 bytes); compressed blobs are validated via the magic bytes
    /// in their headers.
    pub fn restore(&mut self, snapshot: &StateSnapshot) {
        assert_eq!(
            snapshot.layers.len(),
            self.layers.len(),
            "snapshot layer count {} doesn't match state layer count {}",
            snapshot.layers.len(),
            self.layers.len()
        );
        // `bytemuck::cast_slice::<u8, f32>` requires 4-byte alignment of
        // the source, which `Vec<u8>` doesn't guarantee. Decode element-
        // wise so we don't depend on the snapshot's allocator alignment.
        fn decode_f32_into(dst: &mut Vec<f32>, src: &[u8]) {
            assert!(
                src.len().is_multiple_of(4),
                "snapshot byte length {} not a multiple of 4",
                src.len()
            );
            dst.clear();
            dst.reserve(src.len() / 4);
            for chunk in src.as_chunks::<4>().0 {
                dst.push(f32::from_le_bytes(*chunk));
            }
        }

        // f16 sibling of `decode_f32_into`: the source is raw IEEE-754 half
        // bits (2 bytes/elem), decoded element-wise so we don't depend on the
        // snapshot allocator's alignment for a `u16` cast.
        fn decode_u16_into(dst: &mut Vec<u16>, src: &[u8]) {
            assert!(
                src.len().is_multiple_of(2),
                "f16 snapshot byte length {} not a multiple of 2",
                src.len()
            );
            dst.clear();
            dst.reserve(src.len() / 2);
            for chunk in src.as_chunks::<2>().0 {
                dst.push(u16::from_le_bytes(*chunk));
            }
        }

        // Captured before the mutable layer borrow; the `AttentionF16` arm
        // asserts on it (symmetric with the compressed arm's slot assert) so a
        // mode-mismatched snapshot panics loudly instead of silently writing
        // f16 bytes into an f32-configured state. The lfm2.rs compatibility gate
        // is the primary guard; this is the backstop.
        let kv_f16 = self.kv_f16;
        for (layer, snap) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            match (layer, snap) {
                (
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    },
                    LayerSnapshot::Attention { k_data, v_data },
                ) => {
                    // Symmetric backstop to the AttentionF16 arm: an f32
                    // snapshot must not restore into an f16 state (it would
                    // populate the f32 slots while the model reads the empty
                    // f16 slots → silent garbage). The lfm2 gate is the primary
                    // guard; panic loudly if it's ever bypassed.
                    assert!(
                        !kv_f16,
                        "f32 Attention snapshot restored into an f16 state — \
                         caller must gate on the snapshot/live compression mode"
                    );
                    decode_f32_into(key_cache, k_data);
                    decode_f32_into(value_cache, v_data);
                    // Keep the two representations from ever coexisting (f16
                    // slots stay empty on the f32 path).
                    key_cache_f16.clear();
                    value_cache_f16.clear();
                }
                (
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    },
                    LayerSnapshot::AttentionF16 { k_data, v_data },
                ) => {
                    assert!(
                        kv_f16,
                        "AttentionF16 snapshot restored into a non-f16 state — \
                         caller must gate on `LayerSnapshot::is_f16()` matching \
                         the live `kv_f16` mode"
                    );
                    decode_u16_into(key_cache_f16, k_data);
                    decode_u16_into(value_cache_f16, v_data);
                    // f32 caches are unused under f16; clear them so stale data
                    // from a prior uncompressed restore can't leak.
                    key_cache.clear();
                    value_cache.clear();
                }
                (
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        compressed_keys,
                        compressed_values,
                        ..
                    },
                    LayerSnapshot::AttentionCompressed { keys, values },
                ) => {
                    // The live state's compressed slots must already
                    // be allocated by `from_config_with_compression`;
                    // we don't allocate them here because the
                    // associated rotation/scratch state lives
                    // elsewhere on `InferenceState` and would be
                    // missing. A `None` slot means the live state
                    // isn't TurboQuant-configured; the caller
                    // should have detected the incompatibility
                    // before calling restore (see `LayerSnapshot::is_compressed`
                    *compressed_keys = Some(
                        crate::turboquant::decode_compressed_keys(keys)
                            .expect("invalid TQK1 blob in snapshot"),
                    );
                    *compressed_values = Some(
                        crate::turboquant::decode_compressed_values(values)
                            .expect("invalid TQV1 blob in snapshot"),
                    );
                    // f32 caches are unused under compression; clear
                    // them so stale data from a prior uncompressed
                    // restore can't leak.
                    key_cache.clear();
                    value_cache.clear();
                }
                (
                    LayerState::Conv { buffer, history },
                    LayerSnapshot::Conv { buffer: snap_buf },
                ) => {
                    decode_f32_into(buffer, snap_buf);
                    history.clear();
                }
                _ => panic!("snapshot layer kind doesn't match state layer kind"),
            }
        }
        self.seq_len = snapshot.seq_len;
    }

    /// Truncate the KV cache to its first `len` positions, dropping the tail
    /// `[len .. seq_len)` from every attention layer, and set `seq_len = len`.
    ///
    /// This is the speculative-decoding rollback: after verifying K drafted
    /// tokens, the rejected ones' K/V cells (appended at the tail) are dropped so
    /// the next forward continues from the accepted boundary. Unlike
    /// [`Self::shift_kv_with_rope`], **no RoPE fixup is needed** — surviving cells
    /// keep their original absolute positions, so this is a plain `Vec::truncate`.
    /// For convolution layers, rolling buffer state is rewound using recorded history snapshots.
    ///
    /// Preconditions (enforced in all builds):
    /// - `len <= seq_len`
    /// - `!self.is_compressed()` — TurboQuant caches have no tail-truncate.
    pub fn truncate_to(&mut self, len: usize) {
        assert!(
            len <= self.seq_len,
            "truncate_to({len}) exceeds seq_len {}",
            self.seq_len
        );
        assert!(
            !self.is_compressed(),
            "truncate_to called on a TurboQuant-compressed state; not supported"
        );
        if len == self.seq_len {
            return;
        }
        let seq_len = self.seq_len;
        // Truncate one time-major `[seq_len × kv_dim]` cache to `len` positions.
        // `kv_dim = cache.len() / seq_len` (exact — the cache is a whole multiple
        // of `seq_len`); an empty cache (the inactive f32/f16 slot) is a no-op.
        fn trunc<T>(cache: &mut Vec<T>, seq_len: usize, len: usize) {
            if cache.is_empty() {
                return;
            }
            // `kv_dim` is recovered by division, so a cache that is not a whole
            // multiple of `seq_len` would yield a wrong stride and truncate to a
            // boundary mid-vector — leaving a cache that still looks well-formed
            // and decodes to plausible garbage. Assert the invariant instead of
            // propagating it silently.
            assert_eq!(
                cache.len() % seq_len,
                0,
                "KV cache length {} is not a multiple of seq_len {seq_len}; \
                 cannot recover kv_dim to truncate on a vector boundary",
                cache.len()
            );
            let kv_dim = cache.len() / seq_len;
            cache.truncate(len * kv_dim);
        }
        for layer in &mut self.layers {
            match layer {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    trunc(key_cache, seq_len, len);
                    trunc(value_cache, seq_len, len);
                    trunc(key_cache_f16, seq_len, len);
                    trunc(value_cache_f16, seq_len, len);
                }
                LayerState::Conv { buffer, history } => {
                    if !history.rollback_to(len, buffer) {
                        tracing::warn!(
                            target: "cera::kv_cache",
                            target_len = len,
                            "truncate_to target pos not in ConvHistory ring buffer window; zeroing Conv buffer"
                        );
                        buffer.fill(0.0);
                        history.clear();
                    }
                }
            }
        }
        self.seq_len = len;
    }

    /// Drop KV cells `[n_keep .. n_keep + shift)` from every attention
    /// layer, slide the tail down, and re-apply RoPE so each shifted
    /// cell's stored K encodes its new absolute position rather than
    /// its old one. Implements the core of `n_keep` context shift
    /// (Phase 1.5).
    ///
    /// Attention cache layout is time-major `[seq_len × kv_dim]`
    /// where `kv_dim = n_kv_heads × head_dim`, so the drop maps to
    /// `Vec::drain` of a contiguous range — one memmove per layer.
    /// After the drain, cells originally at old position `p_old`
    /// (with `p_old >= n_keep + shift`) now sit at new position
    /// `p_new = p_old - shift`, but their stored K was rotated via
    /// RoPE for `p_old`. We fix this by applying `R(-shift)` (a
    /// constant delta rotation across the whole shifted region) to
    /// each cell's K via [`crate::backend::cpu::apply_rope_delta_to_head`].
    /// Rotations compose additively in each dim-pair plane, so the
    /// result is identical to freshly rotating the raw K for `p_new`.
    /// V is NOT rotated by RoPE — only the two drain calls touch V.
    ///
    /// f16 KV states (`kv_f16`) shift the `*_f16` half-precision slots the same
    /// way: the drain is over the same element range, and each K head is
    /// **widened to f32, delta-rotated with the same helper, then narrowed back
    /// to f16** (`head_dim ≤ 128`). f16 rounding makes this near-exact, not
    /// bit-exact, vs a fresh f16 encode at `p_new`.
    ///
    /// `seq_len` is decremented by `shift`. Compressed (TurboQuant)
    /// layers are **not** shifted — callers must check
    /// [`Self::is_compressed`] first; this method panics otherwise.
    ///
    /// Conv layers (LFM2's `GatedConv`) are intentionally left
    /// untouched: their buffer holds only the last `d_conv`
    /// activations, which are post-shift-valid as soon as the next
    /// forward pass runs. The quality transient decays to zero within
    /// `d_conv` forward passes (typically 3 tokens for LFM2). See
    /// `devlog/000034-feat-n-keep-shift.md` for the analysis.
    ///
    /// Preconditions (enforced in all builds — violating them silently
    /// corrupts state, so we use `assert!` rather than `debug_assert!`):
    /// - `shift > 0`
    /// - `n_keep + shift <= seq_len`
    /// - `!self.is_compressed()`
    /// - `n_kv_heads_per_layer.len() == self.layers.len()`
    #[allow(clippy::too_many_arguments)]
    pub fn shift_kv_with_rope(
        &mut self,
        n_keep: usize,
        shift: usize,
        rope_theta: f32,
        head_dim: usize,
        n_kv_heads_per_layer: &[usize],
        rope_type: crate::backend::cpu::RopeType,
        // Llama-3 RoPE frequency-scaling factors (`rope_freqs.weight`); must match
        // the forward pass so the delta-rotation composes correctly. Only used on
        // the NORM path; `None` ⇒ plain RoPE.
        freq_factors: Option<&[f32]>,
    ) {
        // Hard preconditions — keep them in release builds. Silently
        // decrementing `seq_len` without actually shifting the
        // compressed caches would hand the next forward pass a KV that
        // disagrees with `seq_len`, producing garbage output, and the
        // bounds asserts catch config errors that would otherwise
        // panic deep inside `Vec::drain`.
        assert!(shift > 0, "shift must be > 0");
        assert!(
            n_keep + shift <= self.seq_len,
            "shift range out of bounds: n_keep={n_keep} + shift={shift} > seq_len={}",
            self.seq_len
        );
        assert!(
            !self.is_compressed(),
            "shift_kv_with_rope called on a TurboQuant-compressed state; \
             shifting compressed caches is not supported"
        );
        assert_eq!(
            n_kv_heads_per_layer.len(),
            self.layers.len(),
            "n_kv_heads_per_layer length {} doesn't match layer count {}",
            n_kv_heads_per_layer.len(),
            self.layers.len(),
        );

        // `kv_f16` is a whole-state flag captured before the mutable layer
        // iteration; in f16 mode the `*_f16` slots hold the cache and the f32
        // slots are empty (and vice versa). The delta RoPE re-encoding is
        // identical for both — only the storage width (and the widen/narrow
        // around the rotation) differs.
        let kv_f16 = self.kv_f16;
        // The f16 path widens each K head into a fixed stack buffer before
        // rotating; `head_dim` is bounded by attention hardware (≤ 128 for
        // every supported model). Assert eagerly (f16 only) so a malformed
        // config can't index past the buffer; the f32 path never allocates it.
        if kv_f16 {
            assert!(
                head_dim <= 128,
                "head_dim {head_dim} exceeds the f16 shift widen buffer (128)"
            );
        }
        let new_seq_len = self.seq_len - shift;
        let seq_len = self.seq_len;
        let delta = -(shift as i32);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = layer
            {
                if kv_f16 {
                    // ── f16 path ──────────────────────────────────────────
                    // Layer has no KV yet — nothing to shift.
                    if key_cache_f16.is_empty() && value_cache_f16.is_empty() {
                        continue;
                    }
                    // Same invariants as the f32 path, on the u16 caches:
                    // equal lengths, a clean multiple of `seq_len` (in u16
                    // units — the element count is identical to f32).
                    assert_eq!(
                        key_cache_f16.len(),
                        value_cache_f16.len(),
                        "f16 KV cache length mismatch: key={} value={}",
                        key_cache_f16.len(),
                        value_cache_f16.len()
                    );
                    assert!(seq_len > 0, "attention layer has KV but seq_len is 0");
                    assert_eq!(
                        key_cache_f16.len() % seq_len,
                        0,
                        "f16 KV cache length {} not a multiple of seq_len {}",
                        key_cache_f16.len(),
                        seq_len
                    );
                    let n_kv_heads = n_kv_heads_per_layer[layer_idx];
                    let kv_dim = key_cache_f16.len() / seq_len;
                    assert_eq!(
                        n_kv_heads * head_dim,
                        kv_dim,
                        "layer {layer_idx}: n_kv_heads*head_dim ({}) != cached kv_dim ({})",
                        n_kv_heads * head_dim,
                        kv_dim
                    );
                    let drop_start = n_keep * kv_dim;
                    let drop_end = (n_keep + shift) * kv_dim;
                    key_cache_f16.drain(drop_start..drop_end);
                    value_cache_f16.drain(drop_start..drop_end);

                    // Re-rotate K cells now at [n_keep, new_seq_len) by
                    // R(-shift): widen the f16 head to f32, apply the SAME
                    // delta-RoPE helper as the f32 path, narrow back to f16.
                    // V is not RoPE'd.
                    let mut head_buf = [0.0f32; 128];
                    for t in n_keep..new_seq_len {
                        let row_base = t * kv_dim;
                        for h in 0..n_kv_heads {
                            let head_start = row_base + h * head_dim;
                            let head = &mut key_cache_f16[head_start..head_start + head_dim];
                            let buf = &mut head_buf[..head_dim];
                            for (dst, &bits) in buf.iter_mut().zip(head.iter()) {
                                *dst = crate::quant::f16_to_f32(bits);
                            }
                            match rope_type {
                                crate::backend::cpu::RopeType::Neox => {
                                    crate::backend::cpu::apply_rope_delta_to_head(
                                        buf, delta, head_dim, rope_theta,
                                    )
                                }
                                crate::backend::cpu::RopeType::Norm => {
                                    crate::backend::cpu::apply_rope_norm_delta_to_head(
                                        buf,
                                        delta,
                                        head_dim,
                                        rope_theta,
                                        freq_factors,
                                    )
                                }
                            }
                            for (dst, &x) in head.iter_mut().zip(buf.iter()) {
                                *dst = crate::quant::f32_to_f16(x);
                            }
                        }
                    }
                    continue;
                }

                // ── f32 path ──────────────────────────────────────────────
                // Layer has no KV yet — nothing to shift. Reaches here
                // for models whose first `n_layers - 1` layers were
                // populated but the last one wasn't; guard defensively.
                if key_cache.is_empty() && value_cache.is_empty() {
                    continue;
                }
                // Invariants we rely on: both caches the same length,
                // that length a clean multiple of `seq_len`. Asserting
                // here catches cache-corruption bugs with a clear
                // message instead of an opaque `Vec::drain` panic.
                assert_eq!(
                    key_cache.len(),
                    value_cache.len(),
                    "KV cache length mismatch: key={} value={}",
                    key_cache.len(),
                    value_cache.len()
                );
                assert!(seq_len > 0, "attention layer has KV but seq_len is 0");
                assert_eq!(
                    key_cache.len() % seq_len,
                    0,
                    "KV cache length {} not a multiple of seq_len {}",
                    key_cache.len(),
                    seq_len
                );
                let n_kv_heads = n_kv_heads_per_layer[layer_idx];
                let kv_dim = key_cache.len() / seq_len;
                // Sanity: declared kv_dim matches what's actually stored.
                // An off-by-one here (wrong head count passed in) would
                // silently corrupt the per-head RoPE application below,
                // so assert eagerly.
                assert_eq!(
                    n_kv_heads * head_dim,
                    kv_dim,
                    "layer {layer_idx}: n_kv_heads*head_dim ({}) != cached kv_dim ({})",
                    n_kv_heads * head_dim,
                    kv_dim
                );
                let drop_start = n_keep * kv_dim;
                let drop_end = (n_keep + shift) * kv_dim;
                // `Vec::drain` on a contiguous range is a memmove of
                // the tail — no reallocation, one pass.
                key_cache.drain(drop_start..drop_end);
                value_cache.drain(drop_start..drop_end);

                // Re-rotate K cells now at positions [n_keep, new_seq_len).
                // Their stored K was rotated for (new_pos + shift); apply
                // R(-shift) to re-encode as new_pos. V is not RoPE'd.
                for t in n_keep..new_seq_len {
                    let row_base = t * kv_dim;
                    for h in 0..n_kv_heads {
                        let head_start = row_base + h * head_dim;
                        let head_end = head_start + head_dim;
                        let head = &mut key_cache[head_start..head_end];
                        match rope_type {
                            crate::backend::cpu::RopeType::Neox => {
                                crate::backend::cpu::apply_rope_delta_to_head(
                                    head, delta, head_dim, rope_theta,
                                )
                            }
                            crate::backend::cpu::RopeType::Norm => {
                                crate::backend::cpu::apply_rope_norm_delta_to_head(
                                    head,
                                    delta,
                                    head_dim,
                                    rope_theta,
                                    freq_factors,
                                )
                            }
                        }
                    }
                }
            }
        }

        self.seq_len = new_seq_len;
    }
}

#[cfg(test)]
mod cache_tag_tests {
    use super::KvCompression;

    /// Every mode whose cache *contents* differ must get a distinct tag, and modes
    /// whose contents are identical must share one. Without this, snapshots from
    /// different modes collide in the prefix cache's disk namespace and one
    /// permanently shadows the other for the same prefix.
    #[test]
    fn distinct_modes_get_distinct_tags() {
        let f32_tag = KvCompression::None.cache_tag();
        let f16_tag = KvCompression::F16.cache_tag();
        let both = KvCompression::turboquant(42).cache_tag();
        let keys_only = KvCompression::TurboQuant {
            seed: 42,
            keys: true,
            values: false,
        }
        .cache_tag();
        let values_only = KvCompression::TurboQuant {
            seed: 42,
            keys: false,
            values: true,
        }
        .cache_tag();
        // A different seed means different rotations, and restore validates only
        // shape — so sharing a namespace would decode a prefix in the wrong basis
        // rather than miss.
        let other_seed = KvCompression::turboquant(7).cache_tag();

        let tags = [
            &f32_tag,
            &f16_tag,
            &both,
            &keys_only,
            &values_only,
            &other_seed,
        ];
        for (i, a) in tags.iter().enumerate() {
            for b in tags.iter().skip(i + 1) {
                assert_ne!(a, b, "two modes share a cache tag: {a:?} vs {b:?}");
            }
        }
        assert_eq!(f32_tag, "", "plain f32 must be the untagged default");
    }

    /// Compressing neither side degrades to an f32 cache in
    /// `from_config_capped`, so its contents really are f32's — separating the two
    /// namespaces would only waste entries.
    #[test]
    fn turboquant_with_no_sides_shares_the_f32_namespace() {
        assert_eq!(
            KvCompression::TurboQuant {
                seed: 42,
                keys: false,
                values: false,
            }
            .cache_tag(),
            KvCompression::None.cache_tag()
        );
    }
}

// ── KV Prefix Cache ─────────────────────────────────────────────────────

/// Classification of semantic boundaries in agentic and multimodal workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum SemanticBoundaryKind {
    Unspecified = 0,
    Turn = 1,
    ToolCall = 2,
    ToolOutput = 3,
    Thinking = 4,
    ImageTokens = 5,
    SystemPrompt = 6,
}

impl From<u8> for SemanticBoundaryKind {
    fn from(val: u8) -> Self {
        match val {
            1 => Self::Turn,
            2 => Self::ToolCall,
            3 => Self::ToolOutput,
            4 => Self::Thinking,
            5 => Self::ImageTokens,
            6 => Self::SystemPrompt,
            _ => Self::Unspecified,
        }
    }
}

/// Snapshot of model KV + conv state after prefilling a token sequence.
/// Backend-agnostic: stores raw bytes that the backend knows how to restore.
#[derive(Clone)]
pub struct StateSnapshot {
    pub layers: Vec<LayerSnapshot>,
    pub seq_len: usize,
    pub anchor_depth: u32,
    pub boundary_kind: u8,
    pub semantic_hash: u64,
    pub shift_offset: u32,
}

impl StateSnapshot {
    pub fn new(layers: Vec<LayerSnapshot>, seq_len: usize) -> Self {
        Self {
            layers,
            seq_len,
            anchor_depth: 0,
            boundary_kind: 0,
            semantic_hash: 0,
            shift_offset: 0,
        }
    }

    pub fn with_anchor(mut self, depth: u32, kind: SemanticBoundaryKind, semantic_hash: u64) -> Self {
        self.anchor_depth = depth;
        self.boundary_kind = kind as u8;
        self.semantic_hash = semantic_hash;
        self
    }

    pub fn with_shift_offset(mut self, shift: u32) -> Self {
        self.shift_offset = shift;
        self
    }
}

#[derive(Clone)]
pub enum LayerSnapshot {
    /// Raw f32 KV bytes (CPU / wgpu) or raw f16 (Metal). Backend
    /// chooses the element width; the byte length implicitly carries
    /// it. The `model_fingerprint` plus the `"cpu:"` / `"wgpu:"` /
    /// `"metal:"` model_id prefix prevent cross-backend loads.
    Attention {
        k_data: Vec<u8>,
        v_data: Vec<u8>,
    },
    /// TurboQuant-compressed attention layer with **both** sides
    /// compressed. `keys` is the encoded `CompressedKeyCache` (magic
    /// "TQK1"); `values` is the encoded `CompressedValueCache` (magic
    /// "TQV1"). Mixed-mode (only one side compressed) is not modeled
    /// here — `InferenceState::snapshot` returns `None` for such
    /// states because the alternate side's f32 data has no slot in
    /// this variant. v2 could add per-side variants if a real
    /// mixed-mode workload turns up; today the production config
    /// (`KvCompression::TurboQuant { keys: true, values: true }`)
    /// is uniform.
    AttentionCompressed {
        keys: Vec<u8>,
        values: Vec<u8>,
    },
    /// f16 (IEEE-754 half) KV bytes. Captured from an attention layer whose
    /// `key_cache_f16`/`value_cache_f16` slots hold the cache
    /// (`KvCompression::F16`). `k_data`/`v_data` are the raw little-endian u16
    /// half-bits — half the width of the f32 `Attention` variant. Distinct from
    /// `Attention` so the restore-time cross-mode gate can reject an f16 snapshot
    /// against an f32 state (and vice versa) before a byte-width mismatch
    /// corrupts the cache.
    AttentionF16 {
        k_data: Vec<u8>,
        v_data: Vec<u8>,
    },
    Conv {
        buffer: Vec<u8>,
    },
}

impl LayerSnapshot {
    /// `true` iff this snapshot was captured from a TurboQuant-
    /// compressed attention layer. Callers about to invoke
    /// [`InferenceState::restore`] should check that this matches
    /// the target's per-layer compression mode — `restore` panics
    /// on a compression-mode mismatch (e.g. compressed snapshot
    /// into an uncompressed live state).
    pub fn is_compressed(&self) -> bool {
        matches!(self, LayerSnapshot::AttentionCompressed { .. })
    }

    /// `true` iff this snapshot was captured from an f16 (`KvCompression::F16`)
    /// attention layer. Mirrors [`Self::is_compressed`]; callers gate `restore`
    /// on this matching the target state's `kv_f16` flag so an f16 snapshot is
    /// never restored into an f32 state (or vice versa) — the byte widths
    /// differ, so a cross-mode restore would corrupt the cache.
    pub fn is_f16(&self) -> bool {
        matches!(self, LayerSnapshot::AttentionF16 { .. })
    }
}

impl StateSnapshot {
    pub fn byte_size(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerSnapshot::Attention { k_data, v_data } => k_data.len() + v_data.len(),
                LayerSnapshot::AttentionCompressed { keys, values } => keys.len() + values.len(),
                LayerSnapshot::AttentionF16 { k_data, v_data } => k_data.len() + v_data.len(),
                LayerSnapshot::Conv { buffer } => buffer.len(),
            })
            .sum()
    }

    /// `true` iff any attention layer in this snapshot is
    /// compressed. Used by `Lfm2Model::forward_prefill` to skip
    /// snapshots whose compression mode doesn't match the live
    /// state — treat as cache miss instead of panicking on
    /// `restore`.
    pub fn is_compressed(&self) -> bool {
        self.layers.iter().any(LayerSnapshot::is_compressed)
    }

    /// `true` iff any attention layer in this snapshot is f16
    /// (`AttentionF16`). Used by `Lfm2Model::forward_prefill`'s
    /// cross-mode gate to reject an f16 snapshot against an f32
    /// (or compressed) live state — their byte widths differ, so a
    /// cross-mode restore would corrupt the cache.
    pub fn is_f16(&self) -> bool {
        self.layers.iter().any(LayerSnapshot::is_f16)
    }
}

/// Configuration for the KV prefix cache.
///
/// `Clone` so a backend can rebuild its cache under a new namespace without the
/// caller re-supplying the config — the wgpu backend does this when a session
/// configures KV compression, to keep compressed and f32 disk entries apart.
#[derive(Clone)]
pub struct KvCacheConfig {
    /// Directory for cold-tier (disk) cache files. None = disk caching disabled.
    pub cache_dir: Option<PathBuf>,
    /// Max warm-tier (memory) entries.
    pub max_warm_entries: usize,
    /// Max warm-tier total bytes.
    pub max_warm_bytes: u64,
    /// Max cold-tier (disk) total size in bytes.
    pub max_cold_bytes: u64,
    /// Max warm-tier semantic anchor snapshots retained.
    pub max_warm_anchors: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            max_warm_entries: 32,
            max_warm_bytes: 256 * 1024 * 1024,
            max_cold_bytes: 10 * 1024 * 1024 * 1024,
            max_warm_anchors: 16,
        }
    }
}

struct CacheEntry {
    tokens: Vec<u32>,
    snapshot: StateSnapshot,
    last_used: Cell<Instant>,
}

/// Two-tier KV prefix cache: warm (memory) + cold (disk via FlatBuffers).
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub struct KvPrefixCache {
    warm: HashMap<u64, CacheEntry>,
    pub config: KvCacheConfig,
    model_fingerprint: u64,
    warm_bytes: u64,
}

impl KvPrefixCache {
    pub fn new(config: KvCacheConfig, model_config: &ModelConfig, model_id: &str) -> Self {
        Self {
            warm: HashMap::new(),
            model_fingerprint: model_fingerprint(model_config, model_id),
            config,
            warm_bytes: 0,
        }
    }

    /// Find the longest cached **strict** prefix of `tokens` — an entry covering
    /// `[0, len)` with `len < tokens.len()`. Checks both warm and cold tiers and
    /// returns whichever has the longer match.
    ///
    /// Strictness is not a detail, it is the contract every consumer already
    /// enforces: a full-length hit would leave `use_len == tokens.len()`, and the
    /// restored state already reflects "after all tokens", so re-running the last
    /// token would advance the conv rolling buffer one position past where it
    /// belongs (conv layers don't gate on `seq_len`). `Lfm2Model::forward_prefill`
    /// and both GPU backends therefore skip full hits.
    ///
    /// Returning them anyway made the cache **effectively single-use per token
    /// sequence**: run a prompt once and `insert` stores a full-length entry for
    /// it; ask the same question again and that entry is the longest match, so the
    /// caller rejects it and falls through to a cold prefill *without* falling back
    /// to the shorter, perfectly usable prefix sitting right there. Measured on
    /// LFM2-VL-450M: a genuine prefix hit prefills at ~17k tok/s (f32) / ~20k
    /// (tq3), versus ~840 / ~367 cold — so the shadowed lookup was giving up a
    /// 20-55x speedup on every repeat query.
    ///
    /// Filtering here rather than in each backend keeps warm and cold consistent
    /// and fixes all three backends at once.
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<(StateSnapshot, usize)> {
        let warm_hit = self
            .warm
            .values()
            .filter(|e| e.tokens.len() < tokens.len() && tokens.starts_with(&e.tokens))
            .max_by_key(|e| e.tokens.len())
            .map(|e| {
                e.last_used.set(Instant::now());
                (e.snapshot.clone(), e.tokens.len())
            });

        // Check cold tier too — it may have a longer prefix than the warm hit.
        // `disk-cache` off → cold tier compiles out; only warm hits matter.
        #[cfg(feature = "disk-cache")]
        let cold_hit = self
            .config
            .cache_dir
            .clone()
            .and_then(|dir| self.find_cold_prefix(&dir, tokens))
            .map(|snapshot| {
                let len = snapshot.seq_len;
                (snapshot, len)
            });
        #[cfg(not(feature = "disk-cache"))]
        let cold_hit: Option<(StateSnapshot, usize)> = None;

        let best = match (warm_hit, cold_hit) {
            (Some(w), Some(c)) if c.1 > w.1 => Some(c),
            (Some(w), _) => Some(w),
            (None, c) => c,
        };

        // If the best hit came from the cold tier, promote it to warm.
        // The `< tokens.len()` bound matches the lookup filter above: a full-length
        // warm entry can never be returned, so treating it as "we already have
        // something at least as good" would block promotion forever and re-read the
        // multi-MB cold file on every call.
        if let Some((snapshot, len)) = &best
            && !self.warm.values().any(|e| {
                e.tokens.len() >= *len
                    && e.tokens.len() < tokens.len()
                    && tokens.starts_with(&e.tokens)
            })
        {
            let hash = hash_tokens(&tokens[..*len]);
            let snap_bytes = snapshot.byte_size() as u64;
            self.evict_warm_if_needed(snap_bytes);
            if let Some(old) = self.warm.insert(
                hash,
                CacheEntry {
                    tokens: tokens[..*len].to_vec(),
                    snapshot: snapshot.clone(),
                    last_used: Cell::new(Instant::now()),
                },
            ) {
                self.warm_bytes -= old.snapshot.byte_size() as u64;
            }
            self.warm_bytes += snap_bytes;
        }

        best
    }

    /// Find the deepest cached semantic anchor that is a strict prefix of `tokens`.
    /// Semantic anchors are snapshots saved at turn, tool, or thinking boundaries.
    /// If no anchor is tagged, falls back to [`find_longest_prefix`].
    pub fn find_deepest_semantic_anchor(&mut self, tokens: &[u32]) -> Option<(StateSnapshot, usize)> {
        let warm_anchor = self
            .warm
            .values()
            .filter(|e| {
                e.tokens.len() < tokens.len()
                    && tokens.starts_with(&e.tokens)
                    && (e.snapshot.anchor_depth > 0 || e.snapshot.boundary_kind > 0)
            })
            .max_by_key(|e| e.tokens.len())
            .map(|e| {
                e.last_used.set(Instant::now());
                (e.snapshot.clone(), e.tokens.len())
            });

        #[cfg(feature = "disk-cache")]
        let cold_anchor = self
            .config
            .cache_dir
            .clone()
            .and_then(|dir| self.find_cold_anchor(&dir, tokens))
            .map(|snapshot| {
                let len = snapshot.seq_len;
                (snapshot, len)
            });
        #[cfg(not(feature = "disk-cache"))]
        let cold_anchor: Option<(StateSnapshot, usize)> = None;

        let best = match (warm_anchor, cold_anchor) {
            (Some(w), Some(c)) if c.1 > w.1 => Some(c),
            (Some(w), _) => Some(w),
            (None, c) => c,
        };

        if let Some((snapshot, len)) = &best
            && !self.warm.values().any(|e| {
                e.tokens.len() >= *len
                    && e.tokens.len() < tokens.len()
                    && tokens.starts_with(&e.tokens)
            })
        {
            let hash = hash_tokens(&tokens[..*len]);
            let snap_bytes = snapshot.byte_size() as u64;
            self.evict_warm_if_needed(snap_bytes);
            if let Some(old) = self.warm.insert(
                hash,
                CacheEntry {
                    tokens: tokens[..*len].to_vec(),
                    snapshot: snapshot.clone(),
                    last_used: Cell::new(Instant::now()),
                },
            ) {
                self.warm_bytes -= old.snapshot.byte_size() as u64;
            }
            self.warm_bytes += snap_bytes;
        }

        best.or_else(|| self.find_longest_prefix(tokens))
    }

    /// Cache a semantic anchor prefix's state with boundary metadata.
    pub fn insert_anchor(
        &mut self,
        tokens: &[u32],
        snapshot: StateSnapshot,
        anchor_depth: u32,
        boundary_kind: SemanticBoundaryKind,
        semantic_hash: u64,
    ) {
        let snapshot = snapshot.with_anchor(anchor_depth, boundary_kind, semantic_hash);
        self.insert(tokens, snapshot);
    }

    /// Cache a prefix's state. Stores in warm tier; optionally persists to cold.
    pub fn insert(&mut self, tokens: &[u32], snapshot: StateSnapshot) {
        // Skip if cache is disabled (max_warm_entries == 0 and no disk).
        if self.config.max_warm_entries == 0 && self.config.cache_dir.is_none() {
            return;
        }
        let hash = hash_tokens(tokens);
        let snap_bytes = snapshot.byte_size() as u64;

        // Evict from warm if needed.
        self.evict_warm_if_needed(snap_bytes);

        // Save to cold tier (if `disk-cache` feature on; otherwise no-op).
        #[cfg(feature = "disk-cache")]
        if let Some(dir) = &self.config.cache_dir {
            self.save_cold(dir, tokens, &snapshot);
        }

        if let Some(old) = self.warm.insert(
            hash,
            CacheEntry {
                tokens: tokens.to_vec(),
                snapshot,
                last_used: Cell::new(Instant::now()),
            },
        ) {
            self.warm_bytes -= old.snapshot.byte_size() as u64;
        }
        self.warm_bytes += snap_bytes;
    }

    /// Total bytes in warm tier.
    pub fn warm_bytes(&self) -> u64 {
        self.warm_bytes
    }

    /// Number of warm entries.
    pub fn warm_count(&self) -> usize {
        self.warm.len()
    }

    fn evict_warm_if_needed(&mut self, new_bytes: u64) {
        while (self.warm.len() >= self.config.max_warm_entries
            || self.warm_bytes + new_bytes > self.config.max_warm_bytes)
            && !self.warm.is_empty()
        {
            let oldest = self
                .warm
                .iter()
                .min_by_key(|(_, e)| e.last_used.get())
                .map(|(k, _)| *k);
            if let Some(key) = oldest
                && let Some(removed) = self.warm.remove(&key)
            {
                self.warm_bytes -= removed.snapshot.byte_size() as u64;
            }
        }
    }

    // ── Cold tier (FlatBuffers) ─────────────────────────────────────
    //
    // All cold-tier helpers live behind `disk-cache` (default-on).
    // Builds without `disk-cache` compile only the warm (memory) tier;
    // `cache_dir` on `KvCacheConfig` is retained so consumers don't
    // need to conditionally construct the config, but it's ignored.

    #[cfg(feature = "disk-cache")]
    fn cold_filename(&self, token_hash: u64) -> String {
        format!(
            "{:016x}_{:016x}.kvcache",
            self.model_fingerprint, token_hash
        )
    }

    #[cfg(feature = "disk-cache")]
    fn save_cold(&self, dir: &Path, tokens: &[u32], snapshot: &StateSnapshot) {
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }

        let mut builder =
            flatbuffers::FlatBufferBuilder::with_capacity(snapshot.byte_size() + 1024);

        // Build layers. type_tag taxonomy:
        //   0 = Attention (raw KV bytes at the BACKEND-NATIVE width — f32 on
        //       CPU/wgpu, f16 on Metal; the byte length implicitly carries it).
        //   1 = Conv (raw f32 rolling buffer bytes).
        //   2 = AttentionCompressed (TurboQuant; k_data=encoded keys
        //       blob "TQK1...", v_data=encoded values blob "TQV1...").
        //   3 = AttentionF16 (CPU `KvCompression::F16`; raw IEEE-754 half bits,
        //       k_data/v_data are little-endian u16 half-widths — no schema
        //       change, the generic LayerData {type_tag, k_data, v_data} carries
        //       it. Distinct from Metal's f16-in-tag-0 because a CPU f16 session
        //       must not restore a tag-0 f32 snapshot; the lfm2 gate enforces it).
        let mut layer_offsets = Vec::with_capacity(snapshot.layers.len());
        for layer in &snapshot.layers {
            let (tag, k_off, v_off) = match layer {
                LayerSnapshot::Attention { k_data, v_data } => {
                    let k = builder.create_vector(k_data);
                    let v = builder.create_vector(v_data);
                    (0u8, Some(k), Some(v))
                }
                LayerSnapshot::Conv { buffer } => {
                    let k = builder.create_vector(buffer);
                    (1u8, Some(k), None)
                }
                LayerSnapshot::AttentionCompressed { keys, values } => {
                    let k = builder.create_vector(keys);
                    let v = builder.create_vector(values);
                    (2u8, Some(k), Some(v))
                }
                LayerSnapshot::AttentionF16 { k_data, v_data } => {
                    let k = builder.create_vector(k_data);
                    let v = builder.create_vector(v_data);
                    (3u8, Some(k), Some(v))
                }
            };
            let ld = crate::generated::cera::cache::LayerData::create(
                &mut builder,
                &crate::generated::cera::cache::LayerDataArgs {
                    type_tag: tag,
                    k_data: k_off,
                    v_data: v_off,
                },
            );
            layer_offsets.push(ld);
        }

        let layers_vec = builder.create_vector(&layer_offsets);
        let tokens_vec = builder.create_vector(tokens);

        let entry = crate::generated::cera::cache::KvCacheEntry::create(
            &mut builder,
            &crate::generated::cera::cache::KvCacheEntryArgs {
                model_fingerprint: self.model_fingerprint,
                seq_len: snapshot.seq_len as u32,
                tokens: Some(tokens_vec),
                layers: Some(layers_vec),
                format_version: 2,
                anchor_depth: snapshot.anchor_depth,
                boundary_kind: snapshot.boundary_kind,
                semantic_hash: snapshot.semantic_hash,
                shift_offset: snapshot.shift_offset,
            },
        );
        builder.finish(entry, None);

        let data = builder.finished_data();
        let hash = hash_tokens(tokens);
        let target_path = dir.join(self.cold_filename(hash));
        let tmp_path = dir.join(format!(
            "{:016x}_{:016x}.kvcache.tmp.{}",
            self.model_fingerprint,
            hash,
            std::process::id()
        ));
        if std::fs::write(&tmp_path, data).is_ok() {
            let _ = std::fs::rename(&tmp_path, &target_path);
        }

        self.evict_cold_if_needed(dir);
    }

    #[cfg(feature = "disk-cache")]
    fn find_cold_prefix(&self, dir: &Path, tokens: &[u32]) -> Option<StateSnapshot> {
        // Check specific filenames by pre-computing hashes for all prefixes,
        // longest first. This avoids reading the entire directory.
        //
        // The range stops at `tokens.len() - 1`: only STRICT prefixes are useful
        // (see `find_longest_prefix`). Including the full length would match the
        // entry this very query wrote last time, which every caller then rejects —
        // and because the scan `break`s on its first match, it would never fall
        // back to the shorter usable one. A 1-token input has no strict prefix, so
        // the loop is correctly empty.
        let mut best: Option<StateSnapshot> = None;

        for prefix_len in (1..tokens.len()).rev() {
            let prefix = &tokens[..prefix_len];
            let hash = hash_tokens(prefix);
            let path = dir.join(self.cold_filename(hash));
            if path.exists()
                && let Some(snapshot) = self.load_cold_file(&path, tokens)
            {
                best = Some(snapshot);
                break; // longest prefix first, so first match is best
            }
        }

        best
    }

    #[cfg(feature = "disk-cache")]
    fn find_cold_anchor(&self, dir: &Path, tokens: &[u32]) -> Option<StateSnapshot> {
        let mut best: Option<StateSnapshot> = None;
        for prefix_len in (1..tokens.len()).rev() {
            let prefix = &tokens[..prefix_len];
            let hash = hash_tokens(prefix);
            let path = dir.join(self.cold_filename(hash));
            if path.exists()
                && let Some(snapshot) = self.load_cold_file(&path, tokens)
                && (snapshot.anchor_depth > 0 || snapshot.boundary_kind > 0)
            {
                best = Some(snapshot);
                break;
            }
        }
        best
    }

    #[cfg(feature = "disk-cache")]
    fn load_cold_file(&self, path: &Path, expected_prefix: &[u32]) -> Option<StateSnapshot> {
        let data = std::fs::read(path).ok()?;
        let entry = flatbuffers::root::<crate::generated::cera::cache::KvCacheEntry>(&data).ok()?;

        // Validate format version: must be v2 (2). Legacy/invalid format triggers graceful cache miss.
        if entry.format_version() != 2 {
            return None;
        }

        // Validate model fingerprint.
        if entry.model_fingerprint() != self.model_fingerprint {
            return None;
        }

        let cached_tokens = entry.tokens()?;
        let seq_len = entry.seq_len() as usize;

        // Validate seq_len matches token count.
        if seq_len != cached_tokens.len() {
            return None;
        }

        // Check that cached tokens are a prefix of expected tokens.
        if cached_tokens.len() > expected_prefix.len() {
            return None;
        }
        for (i, ct) in cached_tokens.iter().enumerate() {
            if ct != expected_prefix[i] {
                return None;
            }
        }

        // Reconstruct snapshot.
        let layers_fb = entry.layers()?;
        let mut layers = Vec::with_capacity(layers_fb.len());
        for l in layers_fb {
            match l.type_tag() {
                0 => {
                    layers.push(LayerSnapshot::Attention {
                        k_data: l.k_data()?.bytes().to_vec(),
                        v_data: l.v_data()?.bytes().to_vec(),
                    });
                }
                1 => {
                    layers.push(LayerSnapshot::Conv {
                        buffer: l.k_data()?.bytes().to_vec(),
                    });
                }
                2 => {
                    let keys = l.k_data()?.bytes().to_vec();
                    let values = l.v_data()?.bytes().to_vec();
                    // Validate the encoded blob shape at load time so a
                    // corrupted disk entry surfaces as a cache miss
                    // (return None) rather than a panic later in
                    // `InferenceState::restore`'s `decode_*().expect(...)`.
                    if crate::turboquant::decode_compressed_keys(&keys).is_none()
                        || crate::turboquant::decode_compressed_values(&values).is_none()
                    {
                        return None;
                    }
                    layers.push(LayerSnapshot::AttentionCompressed { keys, values });
                }
                3 => {
                    let k_data = l.k_data()?.bytes().to_vec();
                    let v_data = l.v_data()?.bytes().to_vec();
                    // Validate the f16 blob shape at load time (mirrors the
                    // tag-2 TurboQuant validation): u16 half-bits are 2 bytes
                    // each and K/V are the same length, so a corrupted entry
                    // surfaces as a cache miss (None) here instead of panicking
                    // later in `restore`'s `decode_u16_into` (`len % 2 == 0`).
                    if !k_data.len().is_multiple_of(2)
                        || !v_data.len().is_multiple_of(2)
                        || k_data.len() != v_data.len()
                    {
                        return None;
                    }
                    layers.push(LayerSnapshot::AttentionF16 { k_data, v_data });
                }
                _ => return None,
            }
        }

        Some(StateSnapshot {
            layers,
            seq_len,
            anchor_depth: entry.anchor_depth(),
            boundary_kind: entry.boundary_kind(),
            semantic_hash: entry.semantic_hash(),
            shift_offset: entry.shift_offset(),
        })
    }

    #[cfg(feature = "disk-cache")]
    fn evict_cold_if_needed(&self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let fp_prefix = format!("{:016x}_", self.model_fingerprint);
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();

        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.contains(".kvcache.tmp.") {
                let _ = std::fs::remove_file(entry.path());
                continue;
            }
            if name_str.starts_with(&fp_prefix)
                && name_str.ends_with(".kvcache")
                && let Ok(meta) = entry.metadata()
                && let Ok(modified) = meta.modified()
            {
                files.push((entry.path(), meta.len(), modified));
            }
        }

        let total: u64 = files.iter().map(|(_, sz, _)| sz).sum();
        if total <= self.config.max_cold_bytes {
            return;
        }

        // Sort by mtime ascending (oldest first).
        files.sort_by_key(|(_, _, t)| *t);
        let mut remaining = total;
        for (path, sz, _) in &files {
            if remaining <= self.config.max_cold_bytes {
                break;
            }
            let _ = std::fs::remove_file(path);
            remaining -= sz;
        }
    }
}

/// Stable 64-bit FNV-1a hash. Unlike `DefaultHasher`, the output is guaranteed
/// to be identical across Rust versions and platforms — required for the
/// on-disk cold cache where filenames embed the token hash.
fn fnv1a_u64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    let bytes: &[u8] = bytemuck::cast_slice(tokens);
    fnv1a_u64(bytes)
}

/// Compute a fingerprint for a model configuration.
/// Two models with different fingerprints have incompatible KV cache layouts.
/// Callers should pass a `model_id` that uniquely identifies the specific
/// model weights (e.g. a hash of the GGUF file or the model name from metadata),
/// so different models with the same architecture don't share cache entries.
pub fn model_fingerprint(config: &ModelConfig, model_id: &str) -> u64 {
    // Build a stable byte representation and hash it via FNV-1a. Using
    // DefaultHasher would make the fingerprint non-stable across Rust versions,
    // invalidating on-disk cache files at every toolchain bump.
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(model_id.as_bytes());
    buf.push(0);
    buf.extend_from_slice(config.architecture.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&(config.n_layers as u64).to_le_bytes());
    buf.extend_from_slice(&(config.hidden_size as u64).to_le_bytes());
    buf.extend_from_slice(&(config.n_heads as u64).to_le_bytes());
    for bt in &config.block_types {
        buf.push(match bt {
            crate::model::BlockType::Attention => 0,
            crate::model::BlockType::GatedConv => 1,
        });
    }
    for k in &config.kv_heads_per_layer {
        buf.extend_from_slice(&(*k as u64).to_le_bytes());
    }
    fnv1a_u64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(n_layers: usize, hidden_size: usize) -> ModelConfig {
        ModelConfig {
            architecture: "lfm2".into(),
            n_layers,
            hidden_size,
            intermediate_size: hidden_size * 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: hidden_size / 4,
            vocab_size: 256,
            max_seq_len: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            block_types: (0..n_layers)
                .map(|i| {
                    if i % 2 == 0 {
                        BlockType::Attention
                    } else {
                        BlockType::GatedConv
                    }
                })
                .collect(),
            conv_kernel_size: Some(3),
            kv_heads_per_layer: (0..n_layers)
                .map(|i| if i % 2 == 0 { 2 } else { 0 })
                .collect(),
            scalars: crate::model::ScalarMultipliers::default(),
            moe: None,
        }
    }

    /// The COLD tier's strict-prefix range, pinned directly.
    ///
    /// Reverting `find_cold_prefix`'s `1..tokens.len()` back to `1..=tokens.len()`
    /// left the whole suite green before this test existed — the warm-tier test
    /// below doesn't touch the disk path, and `f16_snapshot_disk_round_trips` hits
    /// under either range. This asserts the boundary itself.
    #[cfg(feature = "disk-cache")]
    #[test]
    fn cold_tier_ignores_a_full_length_entry() {
        let cfg = tiny_config(2, 16);
        let dir = std::env::temp_dir().join(format!("cera_cold_strict_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KvPrefixCache::new(KvCacheConfig::default(), &cfg, "cpu:test");
        let snapshot = |seq_len: usize| StateSnapshot::new(Vec::new(), seq_len);
        let tokens = [7u32, 8, 9];

        // Saved at exactly the query length → unusable, must not be returned.
        cache.save_cold(&dir, &tokens, &snapshot(tokens.len()));
        assert!(
            cache.find_cold_prefix(&dir, &tokens).is_none(),
            "cold tier returned a full-length entry; every caller rejects it, and \
             the scan breaks on its first match so a shorter one would be skipped"
        );

        // A strict prefix of the same query must still be found.
        cache.save_cold(&dir, &tokens[..2], &snapshot(2));
        assert_eq!(
            cache
                .find_cold_prefix(&dir, &tokens)
                .expect("strict prefix must be found")
                .seq_len,
            2
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cached entry covering *all* of `tokens` must not shadow a shorter, usable
    /// one.
    ///
    /// Every consumer rejects a full-length hit (the restored state already
    /// reflects "after all tokens", so re-running the last token would over-advance
    /// the conv rolling buffer). But `insert` stores a full-length entry after each
    /// prefill, so returning them made the cache effectively single-use per token
    /// sequence: the second time the same prompt arrived, the full-length entry was
    /// the longest match, the caller rejected it, and the lookup never fell back to
    /// the strict prefix sitting right there. That cost a 20-55x prefill speedup on
    /// every repeat query.
    #[test]
    fn full_length_entry_does_not_shadow_a_shorter_prefix() {
        let cfg = tiny_config(2, 8);
        let mut cache = KvPrefixCache::new(
            KvCacheConfig {
                cache_dir: None,
                ..KvCacheConfig::default()
            },
            &cfg,
            "test",
        );
        let snapshot = |seq_len: usize| StateSnapshot::new(Vec::new(), seq_len);
        let tokens = [1u32, 2, 3, 4];

        // Only a full-length entry exists → no usable hit.
        cache.insert(&tokens, snapshot(tokens.len()));
        assert!(
            cache.find_longest_prefix(&tokens).is_none(),
            "a full-length entry was returned; the caller can only reject it"
        );

        // Add a strict prefix. It must now be found even though the (longer)
        // full-length entry is still present — this is the regression.
        cache.insert(&tokens[..2], snapshot(2));
        let (_, len) = cache
            .find_longest_prefix(&tokens)
            .expect("strict prefix must be found past the full-length entry");
        assert_eq!(len, 2);

        // A longer strict prefix still wins over a shorter one.
        cache.insert(&tokens[..3], snapshot(3));
        assert_eq!(cache.find_longest_prefix(&tokens).expect("hit").1, 3);
    }

    /// `for_prefill` must cap the pre-allocated KV cache to the prompt length,
    /// NOT `max_seq_len` — the whole point of the hidden-states scratch path.
    #[test]
    fn for_prefill_caps_kv_capacity() {
        let cfg = tiny_config(4, 16); // max_seq_len=64; attn kv_dim = 2*4 = 8
        let n = 5;
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim;
        let state = InferenceState::for_prefill(&cfg, n).unwrap();
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &state.layers[0]
        {
            assert!(key_cache.capacity() >= n * kv_dim);
            assert!(value_cache.capacity() >= n * kv_dim);
            // Far below a full-context reservation (5*8=40 « 64*8=512).
            assert!(key_cache.capacity() < cfg.max_seq_len * kv_dim);
        } else {
            panic!("layer 0 should be attention");
        }
        // Clamp: a prompt longer than max_seq_len caps at max_seq_len.
        let big = InferenceState::for_prefill(&cfg, cfg.max_seq_len + 100).unwrap();
        if let LayerState::Attention { key_cache, .. } = &big.layers[0] {
            assert!(key_cache.capacity() <= cfg.max_seq_len * kv_dim + kv_dim);
        }
    }

    /// Model-free coverage for `truncate_to` (used by speculative decoding to
    /// roll back rejected drafts): it must reset `seq_len` and cut every
    /// attention layer's cache to exactly `len * kv_dim`, byte-for-byte equal to
    /// the prefix present at that length. The `#[ignore]` model test proves the
    /// *logits* round-trip; this pins the pure `Vec` math in CI (which the model
    /// tests can't, per the repo's ignore-gated-oracle pattern).
    #[test]
    fn truncate_to_cuts_caches_to_prefix() {
        // Dense (all-attention) config — `truncate_to` panics on Conv layers.
        let mut cfg = tiny_config(2, 16);
        cfg.architecture = "llama".into();
        cfg.block_types = vec![BlockType::Attention; 2];
        cfg.kv_heads_per_layer = vec![2; 2];
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim; // 2 * 4 = 8

        let mut state = InferenceState::from_config(&cfg).unwrap();
        // Append R rows of distinct KV to every attention layer (value = a
        // function of (token, dim) so any mis-slice is caught).
        let r = 6usize;
        for layer in &mut state.layers {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = layer
            {
                for t in 0..r {
                    for d in 0..kv_dim {
                        key_cache.push((t * 100 + d) as f32);
                        value_cache.push(-((t * 100 + d) as f32));
                    }
                }
            }
        }
        state.seq_len = r;

        // Record the length-L prefix that must survive the truncation.
        let l = 4usize;
        let mut expect_k = Vec::new();
        let mut expect_v = Vec::new();
        for layer in &state.layers {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = layer
            {
                expect_k.push(key_cache[..l * kv_dim].to_vec());
                expect_v.push(value_cache[..l * kv_dim].to_vec());
            }
        }

        state.truncate_to(l);
        assert_eq!(
            state.seq_len, l,
            "seq_len must drop to the truncation length"
        );
        let mut li = 0;
        for layer in &state.layers {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = layer
            {
                assert_eq!(key_cache.len(), l * kv_dim, "key cache cut to len*kv_dim");
                assert_eq!(
                    value_cache.len(),
                    l * kv_dim,
                    "value cache cut to len*kv_dim"
                );
                assert_eq!(
                    key_cache, &expect_k[li],
                    "surviving keys must be the prefix"
                );
                assert_eq!(
                    value_cache, &expect_v[li],
                    "surviving values must be the prefix"
                );
                li += 1;
            }
        }
        assert_eq!(li, 2, "both attention layers must have been checked");

        // Truncating to the current length is a no-op.
        state.truncate_to(l);
        assert_eq!(state.seq_len, l);
    }

    /// f16 companion to `truncate_to_cuts_caches_to_prefix`: an `F16` KV state
    /// (uncompressed, so spec-decode `truncate_to` is legal) must cut the
    /// `key_cache_f16`/`value_cache_f16` half-precision slots to the length-L
    /// prefix. Covers the `trunc` calls on the f16 caches that the f32 test
    /// leaves on their `is_empty()` early-return.
    #[test]
    fn truncate_to_cuts_f16_caches_to_prefix() {
        let mut cfg = tiny_config(2, 16);
        cfg.architecture = "llama".into();
        cfg.block_types = vec![BlockType::Attention; 2];
        cfg.kv_heads_per_layer = vec![2; 2];
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim; // 8

        let mut state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
        assert!(!state.is_compressed(), "F16 KV is not `is_compressed`");

        let r = 6usize;
        for layer in &mut state.layers {
            if let LayerState::Attention {
                key_cache_f16,
                value_cache_f16,
                ..
            } = layer
            {
                for t in 0..r {
                    for d in 0..kv_dim {
                        key_cache_f16.push((t * 100 + d) as u16);
                        value_cache_f16.push((t * 10 + d) as u16);
                    }
                }
            }
        }
        state.seq_len = r;

        let l = 4usize;
        state.truncate_to(l);
        assert_eq!(state.seq_len, l);
        let mut li = 0;
        for layer in &state.layers {
            if let LayerState::Attention {
                key_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = layer
            {
                assert!(key_cache.is_empty(), "f32 slot stays empty under f16");
                assert_eq!(
                    key_cache_f16.len(),
                    l * kv_dim,
                    "f16 keys cut to len*kv_dim"
                );
                assert_eq!(
                    value_cache_f16.len(),
                    l * kv_dim,
                    "f16 values cut to len*kv_dim"
                );
                // Prefix values are (t*100+d) / (t*10+d); confirm the first and
                // last surviving rows are intact (no shifted slice).
                assert_eq!(key_cache_f16[0], 0);
                assert_eq!(
                    key_cache_f16[(l - 1) * kv_dim + (kv_dim - 1)],
                    (300 + 7) as u16
                );
                li += 1;
            }
        }
        assert_eq!(li, 2, "both attention layers must have been checked");
    }

    /// `KvCompression::F16` allocates the f16 slots (f32 slots empty), append
    /// converts to half and round-trips within f16 precision, and the snapshot
    /// path now emits `AttentionF16` for every attention layer.
    #[test]
    fn f16_kv_stores_half_and_snapshots() {
        let cfg = tiny_config(4, 16); // attn layers 0,2; kv_dim = 2*4 = 8
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim;
        let mut state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
        assert!(state.kv_f16, "F16 compression must set kv_f16");
        if let LayerState::Attention {
            key_cache,
            key_cache_f16,
            ..
        } = &state.layers[0]
        {
            assert!(key_cache.is_empty(), "f32 slot stays empty under f16");
            assert!(
                key_cache_f16.capacity() >= cfg.max_seq_len * kv_dim,
                "f16 slot pre-allocated to full context"
            );
        } else {
            panic!("layer 0 should be attention");
        }

        let k: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.1 - 0.3).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| i as f32 * -0.05 + 0.2).collect();
        state.append_kv_f16(0, &k, &v);
        let (k16, v16) = state.kv_cache_f16(0);
        assert_eq!(k16.len(), kv_dim);
        assert_eq!(v16.len(), kv_dim);
        for (i, &b) in k16.iter().enumerate() {
            let got = crate::quant::f16_to_f32(b);
            assert!(
                (got - k[i]).abs() < 1e-2,
                "f16 roundtrip drift at {i}: {got} vs {}",
                k[i]
            );
        }

        state.seq_len = 1;
        let snap = state.snapshot().expect("f16 KV must snapshot");
        // Every attention layer of an f16 state snapshots as AttentionF16,
        // even the one we didn't append to (empty u16 bytes) — the mode flag,
        // not the slot contents, drives the variant.
        for (i, l) in snap.layers.iter().enumerate() {
            match &state.layers[i] {
                LayerState::Attention { .. } => assert!(
                    l.is_f16(),
                    "attention layer {i} must snapshot as AttentionF16"
                ),
                LayerState::Conv { .. } => {
                    assert!(matches!(l, LayerSnapshot::Conv { .. }))
                }
            }
        }
    }

    /// An f16 `InferenceState` snapshots to `AttentionF16` and restores back
    /// into a fresh f16 state byte-exactly (the u16 half-bits and seq_len must
    /// round-trip losslessly through the prefix-cache path).
    #[test]
    fn f16_snapshot_restore_round_trips() {
        let cfg = tiny_config(4, 16); // attn layers 0,2; kv_dim = 2*4 = 8
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim;
        let mut state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();

        // Populate both attention layers with two tokens of deterministic KV.
        for layer in [0usize, 2] {
            for t in 0..2 {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| (layer as f32) + 0.1 * t as f32 + 0.01 * i as f32)
                    .collect();
                let v: Vec<f32> = k.iter().map(|x| -x).collect();
                state.append_kv_f16(layer, &k, &v);
            }
        }
        state.seq_len = 2;

        let snap = state.snapshot().expect("f16 state must snapshot");
        for layer in [0usize, 2] {
            assert!(
                snap.layers[layer].is_f16(),
                "attention layer {layer} must snapshot as AttentionF16"
            );
        }

        // Restore into a fresh f16 state and assert byte-exact round-trip.
        let mut fresh =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
        fresh.restore(&snap);
        assert_eq!(fresh.seq_len, state.seq_len);
        for layer in [0usize, 2] {
            let (ko, vo) = state.kv_cache_f16(layer);
            let (kr, vr) = fresh.kv_cache_f16(layer);
            assert_eq!(
                kr, ko,
                "f16 key cache must round-trip exactly (layer {layer})"
            );
            assert_eq!(
                vr, vo,
                "f16 value cache must round-trip exactly (layer {layer})"
            );
        }
    }

    /// Corruption backstop: restoring an `AttentionF16` snapshot into an
    /// f32-configured state must panic (not silently write u16 bytes into the
    /// f32 slots). The lfm2 compatibility gate is the primary guard; this
    /// asserts the `restore()` backstop behind it.
    #[test]
    #[should_panic(expected = "AttentionF16 snapshot restored into a non-f16 state")]
    fn restore_f16_snapshot_into_f32_state_panics() {
        let cfg = tiny_config(2, 16); // layer 0 attention, layer 1 conv
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim;
        let mut f16_state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
        f16_state.append_kv_f16(0, &vec![0.5f32; kv_dim], &vec![-0.5f32; kv_dim]);
        f16_state.seq_len = 1;
        let snap = f16_state.snapshot().expect("f16 snapshots");

        let mut f32_state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::None).unwrap();
        f32_state.restore(&snap); // must panic on the AttentionF16 arm's kv_f16 assert
    }

    /// Reverse backstop: restoring an f32 `Attention` snapshot into an
    /// f16-configured state must panic (else the f16 session gets f32 slots
    /// populated while it reads the empty f16 slots — silent garbage).
    #[test]
    #[should_panic(expected = "f32 Attention snapshot restored into an f16 state")]
    fn restore_f32_snapshot_into_f16_state_panics() {
        let cfg = tiny_config(2, 16);
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim;
        let mut f32_state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::None).unwrap();
        f32_state.append_kv(0, &vec![0.5f32; kv_dim], &vec![-0.5f32; kv_dim]);
        f32_state.seq_len = 1;
        let snap = f32_state.snapshot().expect("f32 snapshots");

        let mut f16_state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
        f16_state.restore(&snap); // must panic on the Attention arm's !kv_f16 assert
    }

    /// f16 snapshot survives the disk (flatbuffer `type_tag = 3`) round-trip:
    /// `save_cold` → `find_cold_prefix` returns the same `AttentionF16` bytes.
    #[cfg(feature = "disk-cache")]
    #[test]
    fn f16_snapshot_disk_round_trips() {
        let cfg = tiny_config(2, 16);
        let kv_dim = cfg.kv_heads_per_layer[0] * cfg.head_dim;
        let mut state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
        let k: Vec<f32> = (0..kv_dim).map(|i| 0.1 * i as f32 - 0.3).collect();
        let v: Vec<f32> = k.iter().map(|x| -x).collect();
        state.append_kv_f16(0, &k, &v);
        state.seq_len = 1;
        let snap = state.snapshot().expect("f16 snapshots");
        let (k0, v0) = match &snap.layers[0] {
            LayerSnapshot::AttentionF16 { k_data, v_data } => (k_data.clone(), v_data.clone()),
            _ => panic!("source layer 0 should be AttentionF16"),
        };

        let dir = std::env::temp_dir().join(format!("cera_f16_disk_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KvPrefixCache::new(KvCacheConfig::default(), &cfg, "cpu:test");
        // The cache requires seq_len == token count; we appended one token.
        let tokens = [42u32];
        cache.save_cold(&dir, &tokens, &snap);
        // Look up a LONGER sequence: `find_cold_prefix` only returns strict
        // prefixes (a full-length hit is unusable — see `find_longest_prefix`), so
        // the saved 1-token entry has to be read back as the prefix of something
        // longer. This test is about the `type_tag = 3` serialization, not the
        // matching policy.
        let query = [42u32, 43];
        let restored = cache
            .find_cold_prefix(&dir, &query)
            .expect("disk round-trip must find the f16 entry");
        assert_eq!(restored.seq_len, snap.seq_len);
        match &restored.layers[0] {
            LayerSnapshot::AttentionF16 { k_data, v_data } => {
                assert_eq!(*k_data, k0, "f16 key bytes must survive disk round-trip");
                assert_eq!(*v_data, v0, "f16 value bytes must survive disk round-trip");
            }
            _ => panic!("restored layer 0 should be AttentionF16 (type_tag 3)"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// After an f16 `n_keep` shift, a shifted K head decodes (within f16
    /// tolerance) to the value a fresh RoPE at the new position would produce
    /// from the same raw K — the widen→rotate→narrow path composes the delta
    /// rotation exactly like the f32 shift does. V is drained but unrotated.
    #[test]
    fn f16_shift_reencodes_like_fresh() {
        let head_dim = 16usize;
        let n_kv_heads = 2usize;
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 12usize;
        let n_keep = 3usize;
        let shift = 4usize;
        let freq_base = 10_000.0f32;

        // A 2-layer config: layer 0 attention, layer 1 conv. head_dim=16 →
        // hidden_size = n_heads(4) * head_dim = 64.
        let mut cfg = tiny_config(2, 64);
        cfg.head_dim = head_dim;
        cfg.n_kv_heads = n_kv_heads;
        cfg.kv_heads_per_layer = vec![n_kv_heads, 0];
        cfg.max_seq_len = 64;

        let mut state =
            InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();

        // Raw (pre-RoPE) K per (head, dim); the same raw is used at every
        // position so the oracle is a pure function of the new position.
        let raw = |h: usize, d: usize| 0.1 * (h as f32) + 0.01 * (d as f32);

        // Fill each position's K with the raw values rotated for that position
        // (mirroring the forward pass), and V with the unrotated raw.
        for t in 0..seq_len {
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            for h in 0..n_kv_heads {
                let mut head: Vec<f32> = (0..head_dim).map(|d| raw(h, d)).collect();
                crate::backend::cpu::apply_rope_delta_to_head(
                    &mut head, t as i32, head_dim, freq_base,
                );
                for d in 0..head_dim {
                    k[h * head_dim + d] = head[d];
                    v[h * head_dim + d] = raw(h, d);
                }
            }
            state.append_kv_f16(0, &k, &v);
        }
        state.seq_len = seq_len;

        state.shift_kv_with_rope(
            n_keep,
            shift,
            freq_base,
            head_dim,
            &cfg.kv_heads_per_layer,
            crate::backend::cpu::RopeType::Neox,
            None,
        );

        let new_seq_len = seq_len - shift;
        assert_eq!(state.seq_len, new_seq_len);
        let (k16, v16) = state.kv_cache_f16(0);
        assert_eq!(k16.len(), new_seq_len * kv_dim);

        // Tail cells: the cell now at t_new was at t_old = t_new + shift; its
        // stored K must decode to a fresh RoPE for t_new. V stays unrotated.
        for t_new in n_keep..new_seq_len {
            for h in 0..n_kv_heads {
                let mut oracle: Vec<f32> = (0..head_dim).map(|d| raw(h, d)).collect();
                crate::backend::cpu::apply_rope_delta_to_head(
                    &mut oracle,
                    t_new as i32,
                    head_dim,
                    freq_base,
                );
                for (d, &oracle_d) in oracle.iter().enumerate() {
                    let idx = t_new * kv_dim + h * head_dim + d;
                    let got_k = crate::quant::f16_to_f32(k16[idx]);
                    assert!(
                        (got_k - oracle_d).abs() < 1e-2,
                        "K reencode drift at t_new={t_new} h={h} d={d}: {got_k} vs {oracle_d}"
                    );
                    let got_v = crate::quant::f16_to_f32(v16[idx]);
                    assert!(
                        (got_v - raw(h, d)).abs() < 1e-2,
                        "V must stay unrotated at t_new={t_new} h={h} d={d}: {got_v} vs {}",
                        raw(h, d)
                    );
                }
            }
        }
    }

    /// `clear_for_reuse` zeroes seq_len and empties KV/conv buffers while KEEPING
    /// capacity, so a reused hidden-states scratch does no allocation.
    #[test]
    fn clear_for_reuse_resets_but_keeps_capacity() {
        let cfg = tiny_config(4, 16);
        let mut state = InferenceState::for_prefill(&cfg, 8).unwrap();
        state.seq_len = 3;
        let (cap_k, cap_v) = if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[0]
        {
            for i in 0..16 {
                key_cache.push(i as f32);
                value_cache.push(i as f32);
            }
            (key_cache.capacity(), value_cache.capacity())
        } else {
            panic!("layer 0 should be attention");
        };
        if let LayerState::Conv { buffer, .. } = &mut state.layers[1] {
            buffer.iter_mut().for_each(|x| *x = 1.0);
        }

        state.clear_for_reuse();

        assert_eq!(state.seq_len, 0);
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &state.layers[0]
        {
            assert!(key_cache.is_empty() && value_cache.is_empty());
            assert_eq!(key_cache.capacity(), cap_k, "capacity must be retained");
            assert_eq!(value_cache.capacity(), cap_v);
        }
        if let LayerState::Conv { buffer, .. } = &state.layers[1] {
            assert!(
                buffer.iter().all(|&x| x == 0.0),
                "conv buffer must be zeroed"
            );
        }
    }

    /// Snapshot then restore on a populated `InferenceState` must
    /// reproduce the exact same byte-level contents — the prefix
    /// cache's correctness depends on this round-trip being lossless.
    #[test]
    fn snapshot_restore_round_trip_attention_and_conv() {
        let cfg = tiny_config(4, 16);
        let mut state = InferenceState::from_config(&cfg).unwrap();

        // Populate attention layer 0's KV with deterministic values
        // that fit kv_dim = n_kv_heads * head_dim = 2 * 4 = 8.
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[0]
        {
            for i in 0..16 {
                key_cache.push(i as f32 * 0.5);
                value_cache.push(-(i as f32) * 0.25);
            }
        }
        if let LayerState::Conv { buffer, .. } = &mut state.layers[1] {
            for v in buffer.iter_mut() {
                *v = 0.7;
            }
        }
        state.seq_len = 2;

        let snap = state.snapshot().expect("uncompressed state must snapshot");

        // Drop the existing KV by recreating, then restore.
        let mut fresh = InferenceState::from_config(&cfg).unwrap();
        fresh.restore(&snap);

        match (&fresh.layers[0], &state.layers[0]) {
            (
                LayerState::Attention {
                    key_cache: kr,
                    value_cache: vr,
                    ..
                },
                LayerState::Attention {
                    key_cache: ko,
                    value_cache: vo,
                    ..
                },
            ) => {
                assert_eq!(kr, ko, "key_cache must round-trip exactly");
                assert_eq!(vr, vo, "value_cache must round-trip exactly");
            }
            _ => panic!("expected attention layer 0"),
        }
        match (&fresh.layers[1], &state.layers[1]) {
            (LayerState::Conv { buffer: br, .. }, LayerState::Conv { buffer: bo, .. }) => {
                assert_eq!(br, bo, "conv buffer must round-trip exactly")
            }
            _ => panic!("expected conv layer 1"),
        }
        assert_eq!(fresh.seq_len, state.seq_len);
    }

    /// Empty (no-prefill-yet) state must still round-trip — Vec<u8>
    /// length zero on both sides.
    #[test]
    fn snapshot_restore_round_trip_empty_state() {
        let cfg = tiny_config(2, 8);
        let state = InferenceState::from_config(&cfg).unwrap();
        let snap = state.snapshot().expect("empty state still snapshots");
        let mut fresh = InferenceState::from_config(&cfg).unwrap();
        fresh.restore(&snap);
        assert_eq!(fresh.seq_len, 0);
    }

    /// Compressed state now snapshots into the
    /// `LayerSnapshot::AttentionCompressed { keys, values }` variant
    /// (was `None` before this PR). The encoded blobs round-trip
    /// through `restore` byte-equal to the source cache.
    #[test]
    fn snapshot_compressed_state_emits_attention_compressed() {
        let cfg = tiny_config(2, 8);
        let mut state = InferenceState::from_config(&cfg).unwrap();
        // Layer 0 is an attention layer per `tiny_config`. Populate
        // both compressed_keys and compressed_values with a synthetic
        // 1-token compressed cache.
        if let LayerState::Attention {
            compressed_keys,
            compressed_values,
            ..
        } = &mut state.layers[0]
        {
            let mut keys = CompressedKeyCache::new(2, 8, 4);
            let mut values = CompressedValueCache::new(2, 8, 4);
            for h in 0..2 {
                keys.append(h, &[0xAB, 0xCD], &[0x55], 0x1234, 0x5678);
                values.append(h, &[0xEF, 0x01], 0x9ABC);
            }
            *compressed_keys = Some(keys);
            *compressed_values = Some(values);
        }
        state.seq_len = 1;
        assert!(state.is_compressed());

        let snap = state.snapshot().expect("compressed state must snapshot");
        match &snap.layers[0] {
            LayerSnapshot::AttentionCompressed { keys, values } => {
                assert!(keys.starts_with(b"TQK1"));
                assert!(values.starts_with(b"TQV1"));
            }
            _ => panic!("layer 0 should be AttentionCompressed"),
        }

        // Restore into a fresh state and assert the polar/jl bytes
        // and norms match the original.
        let mut fresh = InferenceState::from_config(&cfg).unwrap();
        // `from_config` (uncompressed) doesn't allocate
        // `compressed_keys` slots; manually wire empty caches so
        // `restore`'s `Some(_)` write target exists.
        if let LayerState::Attention {
            compressed_keys,
            compressed_values,
            ..
        } = &mut fresh.layers[0]
        {
            *compressed_keys = Some(CompressedKeyCache::new(2, 8, 4));
            *compressed_values = Some(CompressedValueCache::new(2, 8, 4));
        }
        fresh.restore(&snap);

        match (&state.layers[0], &fresh.layers[0]) {
            (
                LayerState::Attention {
                    compressed_keys: Some(orig_k),
                    compressed_values: Some(orig_v),
                    ..
                },
                LayerState::Attention {
                    compressed_keys: Some(restored_k),
                    compressed_values: Some(restored_v),
                    ..
                },
            ) => {
                assert_eq!(restored_k.polar_data, orig_k.polar_data);
                assert_eq!(restored_k.jl_data, orig_k.jl_data);
                assert_eq!(restored_k.norms, orig_k.norms);
                assert_eq!(restored_k.residual_norms, orig_k.residual_norms);
                assert_eq!(restored_v.polar_data, orig_v.polar_data);
                assert_eq!(restored_v.norms, orig_v.norms);
                // f32 caches are recomputed at decode — must match.
                assert_eq!(restored_k.norms_f32, orig_k.norms_f32);
                assert_eq!(restored_v.norms_f32, orig_v.norms_f32);
            }
            _ => panic!("expected both states to have populated compressed caches"),
        }
        assert_eq!(fresh.seq_len, state.seq_len);
    }

    #[test]
    fn conv_history_ring_buffer_and_rollback() {
        let buf_len = 16;
        let mut history = ConvHistory::new(buf_len);
        let mut buf = vec![0.0f32; buf_len];

        // Push 10 snapshots with recognizable values
        for pos in 1..=10 {
            buf.fill(pos as f32);
            history.push(pos, &buf);
        }

        // Roll back to pos = 5
        let mut restored = vec![0.0f32; buf_len];
        assert!(history.rollback_to(5, &mut restored));
        assert_eq!(restored, vec![5.0f32; buf_len]);

        // Roll back to pos = 0 (clears buffer)
        assert!(history.rollback_to(0, &mut restored));
        assert_eq!(restored, vec![0.0f32; buf_len]);

        // Roll back to non-existent position fails cleanly
        assert!(!history.rollback_to(99, &mut restored));
    }

    #[test]
    fn conv_history_ring_buffer_wraparound() {
        let buf_len = 8;
        let mut history = ConvHistory::new(buf_len);
        let mut buf = vec![0.0f32; buf_len];

        // Push 100 snapshots (exceeding CONV_HISTORY_CAPACITY = 64)
        for pos in 1..=100 {
            buf.fill(pos as f32);
            history.push(pos, &buf);
        }

        let mut restored = vec![0.0f32; buf_len];
        // Positions within the latest 64 steps (37..=100) must be retrievable
        assert!(history.rollback_to(100, &mut restored));
        assert_eq!(restored, vec![100.0f32; buf_len]);

        assert!(history.rollback_to(50, &mut restored));
        assert_eq!(restored, vec![50.0f32; buf_len]);

        // Older positions that fell out of the ring buffer (> 64 steps old) return false
        assert!(!history.rollback_to(10, &mut restored));
    }

    #[test]
    fn semantic_anchor_lookup_and_boundary_tags() {
        let cfg = tiny_config(2, 8);
        let mut cache = KvPrefixCache::new(
            KvCacheConfig {
                cache_dir: None,
                ..KvCacheConfig::default()
            },
            &cfg,
            "test_anchor",
        );
        let snapshot = |seq_len: usize| StateSnapshot::new(Vec::new(), seq_len);
        let tokens = [10u32, 20, 30, 40, 50, 60, 70];

        // Insert non-anchor prefix at len 5
        cache.insert(&tokens[..5], snapshot(5));

        // Insert semantic anchor at len 3 (e.g. Turn boundary)
        cache.insert_anchor(
            &tokens[..3],
            snapshot(3),
            1,
            SemanticBoundaryKind::Turn,
            0x12345678,
        );

        // find_deepest_semantic_anchor finds the anchor at len 3
        let (snap, len) = cache
            .find_deepest_semantic_anchor(&tokens)
            .expect("must find anchor");
        assert_eq!(len, 3);
        assert_eq!(snap.anchor_depth, 1);
        assert_eq!(snap.boundary_kind, SemanticBoundaryKind::Turn as u8);
        assert_eq!(snap.semantic_hash, 0x12345678);

        // Standard find_longest_prefix still prefers the longest prefix (len 5)
        let (_, longest_len) = cache
            .find_longest_prefix(&tokens)
            .expect("must find longest prefix");
        assert_eq!(longest_len, 5);
    }

    #[cfg(feature = "disk-cache")]
    #[test]
    fn flatbuffers_v2_disk_roundtrip_with_anchors_and_atomic_tmp() {
        let cfg = tiny_config(2, 16);
        let dir = std::env::temp_dir().join(format!("cera_cold_v2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KvPrefixCache::new(KvCacheConfig::default(), &cfg, "cpu:test_v2");

        let snap = StateSnapshot::new(
            vec![
                LayerSnapshot::Attention {
                    k_data: vec![1, 2, 3, 4],
                    v_data: vec![5, 6, 7, 8],
                },
                LayerSnapshot::Conv {
                    buffer: vec![9, 10, 11, 12],
                },
            ],
            4,
        )
        .with_anchor(2, SemanticBoundaryKind::ToolCall, 0xdeadbeef)
        .with_shift_offset(0);

        let tokens = [100u32, 101, 102, 103];
        cache.save_cold(&dir, &tokens, &snap);

        // Verify the file was written atomically and no orphan .tmp remains
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0]
                .file_name()
                .to_string_lossy()
                .ends_with(".kvcache")
        );

        // Load from disk and verify v2 schema fields
        let loaded = cache
            .find_cold_prefix(&dir, &[100, 101, 102, 103, 104])
            .expect("must load v2 snapshot from disk");
        assert_eq!(loaded.seq_len, 4);
        assert_eq!(loaded.anchor_depth, 2);
        assert_eq!(loaded.boundary_kind, SemanticBoundaryKind::ToolCall as u8);
        assert_eq!(loaded.semantic_hash, 0xdeadbeef);
        assert_eq!(loaded.shift_offset, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}

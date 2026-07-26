//! GPU-resident TurboQuant KV cache for the native Metal backend.
//!
//! The Metal twin of `model::gpu_turboquant` (a plain code span, not a link: that
//! module is behind the `gpu` feature and a link would break `cargo doc
//! --features metal`): same packed layout, same per-layer rotations, same
//! `TQK1`/`TQV1` snapshot blobs. Built lazily by
//! `Model::configure_kv_compression` so a session that never asks for TurboQuant
//! pays neither the buffers nor the MSL compilation.
//!
//! The kernels and the packed layout are documented in
//! `backend/shaders/turboquant.metal`; this module is the host mirror of that
//! layout and the single place the two are kept in sync.
//!
//! ## Where this differs from the wgpu mirror
//!
//! Two pieces of wgpu machinery have no counterpart here, both because Metal is
//! the friendlier API:
//!
//! - **No params slab.** wgpu has to stage every layer's params into one buffer
//!   ahead of the encoders, because `Queue::write_buffer` writes land in
//!   submission order and a per-dispatch write into a shared range would let
//!   every dispatch observe only the last value. Metal's `set_bytes` copies the
//!   bytes into the command buffer at encode time, so each dispatch carries its
//!   own params and they are built inline.
//! - **No gather kernel for snapshots.** wgpu needs a GPU-side compaction pass
//!   plus a staging-buffer readback. Metal buffers are `StorageModeShared`, so a
//!   snapshot is a plain memcpy out of the mapped contents.
//!
//! Compression rate, `head_dim = 128`: keys 52 bytes/vector (32 polar + 16 JL +
//! 2 + 2 norms) vs 512 as f32, values 36 vs 512 — about 11.6× less KV overall.
//! Against this backend's f16 KV the win is half that, ~5.8×.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use crate::CeraError;
use crate::backend::metal::{MetalContext, MetalParams, TqAttnParams, TqParams, shaders};
use crate::kv_cache::checked_elems;
use crate::model::{BlockType, ModelConfig};
use crate::turboquant::{
    CompressedKeyCache, CompressedValueCache, RotationState, TurboQuantConfig,
    decode_compressed_keys, decode_compressed_values, encode_compressed_keys,
    encode_compressed_values,
};
// The packed-layout and mode types are backend-agnostic and live with the
// algorithm; re-exported so `model::metal_turboquant::{TqLayout, TqMode}` stays a
// valid path for callers and tests, matching the wgpu mirror.
pub use crate::turboquant::{TqLayout, TqMode, head_dim_supported};

/// Threads per threadgroup for the encode / rotate kernels. Must equal `TQ_WG`
/// in `turboquant.metal`, which is also what caps `head_dim` — one thread covers
/// one element.
pub const TQ_THREADS: u64 = 128;

/// Threads per threadgroup for `flash_attention_tq`. Must equal its `TQA_TILE`,
/// which the kernel also uses as its KV tile width.
pub const TQ_ATTN_THREADS: u64 = 256;

/// The four compressed-path pipelines.
struct TqPipelines {
    encode_keys: ComputePipelineState,
    encode_values: ComputePipelineState,
    rotate_q: ComputePipelineState,
    attention: ComputePipelineState,
}

/// One attention layer's packed caches.
pub struct TqLayerCache {
    /// `[polar | jl | norms]`, `n_kv_heads * max_seq_len` slots per region.
    pub keys: Buffer,
    /// `[polar | norms]`.
    pub values: Buffer,
    pub n_kv_heads: usize,
}

/// GPU-resident compressed KV cache plus the machinery to drive it.
pub struct TqMetalCache {
    pub mode: TqMode,
    pub layout: TqLayout,
    /// Per model layer; `None` for conv (non-attention) layers.
    pub layers: Vec<Option<TqLayerCache>>,
    /// All layers' RHT sign flips, `[polar | jl]` per layer (`2 * head_dim` f32
    /// each). One buffer keeps the kernels at a single sign binding; the
    /// per-layer offset rides in the params.
    signs: Buffer,
    /// Rotated queries: `[q_rot | q_jl | sums]`, `q_cap` rows per region.
    qrot: Buffer,
    /// Query rows the `qrot` scratch can hold — the prefill chunk size.
    q_cap: usize,
    /// Lloyd-Max centroids + boundaries for this `head_dim`.
    pub config: TurboQuantConfig,
    pipelines: TqPipelines,
    max_seq_len: usize,
}

impl TqMetalCache {
    /// Allocate the compressed caches and compile the pipelines.
    ///
    /// `q_cap` is the largest query batch the attention kernel will see (the
    /// prefill chunk size); the rotated-query scratch is sized for it.
    ///
    /// The per-layer rotations come from the same
    /// `RotationState::try_from_seed(seed ^ layer_idx, head_dim)` the CPU and
    /// wgpu backends use, so a cache written on one backend is readable by the
    /// others — which is what makes a cross-backend prefix-cache snapshot
    /// meaningful.
    pub fn new(
        ctx: &MetalContext,
        config: &ModelConfig,
        max_seq_len: usize,
        q_cap: usize,
        mode: TqMode,
    ) -> Result<Self, CeraError> {
        let head_dim = config.head_dim;
        // The in-crate caller gates on `TqMode::from_compression`, but this type
        // is `pub` (the oracle tests construct it directly), so surface a typed
        // error rather than aborting the caller's process.
        if !head_dim_supported(head_dim) {
            return Err(crate::turboquant::unsupported_head_dim(head_dim));
        }
        let layout = TqLayout::new(head_dim);
        let n_layers = config.block_types.len();

        // Per-layer sign flips, laid out [polar | jl] so one binding serves both.
        let mut signs = Vec::with_capacity(n_layers * 2 * head_dim);
        for layer_idx in 0..n_layers {
            let rot = RotationState::try_from_seed(mode.seed ^ layer_idx as u64, head_dim)?;
            signs.extend_from_slice(&rot.polar_signs);
            signs.extend_from_slice(&rot.jl_signs);
        }

        let mut layers = Vec::with_capacity(n_layers);
        for (i, bt) in config.block_types.iter().enumerate() {
            match bt {
                BlockType::Attention => {
                    let n_kv_heads = config.kv_heads_per_layer[i];
                    // `TqLayout` owns the region formula and guards its own
                    // multiplies, so production and the oracle tests can't drift
                    // apart on the sizing.
                    let vecs = checked_elems::<u32>(n_kv_heads, max_seq_len)?;
                    let k_bytes = TqLayout::words_to_bytes(layout.key_words(vecs)?);
                    let v_bytes = TqLayout::words_to_bytes(layout.value_words(vecs)?);
                    layers.push(Some(TqLayerCache {
                        keys: ctx.create_buffer(k_bytes),
                        values: ctx.create_buffer(v_bytes),
                        n_kv_heads,
                    }));
                }
                BlockType::GatedConv => layers.push(None),
            }
        }

        // Rotated-query scratch: two head-wide regions plus one sum per (row, head).
        let q_rows = checked_elems::<f32>(q_cap, config.n_heads)?;
        let qrot_floats = checked_elems::<f32>(q_rows, 2 * head_dim + 1)?;

        let pipelines = TqPipelines {
            encode_keys: pipeline(ctx, shaders::TURBOQUANT, "tq_encode_keys")?,
            encode_values: pipeline(ctx, shaders::TURBOQUANT, "tq_encode_values")?,
            rotate_q: pipeline(ctx, shaders::TURBOQUANT, "tq_rotate_q")?,
            attention: pipeline(ctx, shaders::FLASH_ATTENTION_TQ, "flash_attention_tq")?,
        };

        Ok(Self {
            mode,
            layout,
            layers,
            signs: ctx.upload_f32(&signs),
            qrot: ctx.create_buffer(TqLayout::words_to_bytes(qrot_floats)),
            q_cap,
            config: TurboQuantConfig::for_head_dim(head_dim),
            pipelines,
            max_seq_len,
        })
    }

    /// This layer's cache, or `None` for a conv layer.
    pub fn layer(&self, layer: usize) -> Option<&TqLayerCache> {
        self.layers[layer].as_ref()
    }

    /// Base params shared by the three encode/rotate dispatches. `n_heads` and
    /// `src_stride` are the two fields that differ between the KV encoders (KV
    /// heads, KV row) and the query rotation (query heads, Q row), so they stay
    /// caller-supplied; everything else is derived here.
    fn base_params(
        &self,
        n_tokens: usize,
        n_heads: usize,
        layer: usize,
        start_pos: usize,
    ) -> TqParams {
        let head_dim = self.layout.head_dim;
        let c = &self.config.centroids;
        let b = &self.config.boundaries;
        TqParams {
            n_tokens: n_tokens as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            src_stride: (n_heads * head_dim) as u32,
            dst_pos: start_pos as u32,
            max_seq_len: self.max_seq_len as u32,
            sign_off: (layer * 2 * head_dim) as u32,
            q_cap: self.q_cap as u32,
            c0: c[0],
            c1: c[1],
            c2: c[2],
            c3: c[3],
            b0: b[0],
            b1: b[1],
            b2: b[2],
            _pad: 0,
        }
    }

    /// Compress `n_tokens` rows of K and V for `layer` into the packed cache,
    /// replacing the f16 path's two cast dispatches.
    ///
    /// `k_src` / `v_src` are row-major `[n_tokens × n_kv_heads × head_dim]`
    /// post-RoPE f32 projections; rows land at cache timesteps `[start_pos,
    /// start_pos + n_tokens)`.
    pub fn encode_kv(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
        k_src: &Buffer,
        v_src: &Buffer,
        n_tokens: usize,
        start_pos: usize,
    ) {
        let l = self
            .layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .expect("encode_kv on a layer without a compressed cache");
        assert!(
            start_pos + n_tokens <= self.max_seq_len,
            "encode_kv writes timesteps [{start_pos}, {}) past the cache capacity {}",
            start_pos + n_tokens,
            self.max_seq_len,
        );
        // The encoders walk KV heads and read KV-strided rows.
        let params = self.base_params(n_tokens, l.n_kv_heads, layer, start_pos);
        let groups = (n_tokens * l.n_kv_heads) as u64;
        self.dispatch(
            enc,
            &self.pipelines.encode_keys,
            k_src,
            &l.keys,
            &params,
            groups,
        );
        self.dispatch(
            enc,
            &self.pipelines.encode_values,
            v_src,
            &l.values,
            &params,
            groups,
        );
    }

    /// Pre-rotate `n_tokens × n_heads` query heads into the `qrot` scratch.
    /// Must run before [`Self::attention`] for the same layer.
    pub fn rotate_queries(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
        q_src: &Buffer,
        n_tokens: usize,
        n_heads: usize,
    ) {
        assert!(
            n_tokens <= self.q_cap,
            "rotate_queries: {n_tokens} rows exceeds the qrot scratch capacity {}",
            self.q_cap
        );
        // The rotation walks QUERY heads and reads Q-strided rows. `dst_pos` is
        // unused by `tq_rotate_q` (the scratch is indexed by batch row).
        let params = self.base_params(n_tokens, n_heads, layer, 0);
        self.dispatch(
            enc,
            &self.pipelines.rotate_q,
            q_src,
            &self.qrot,
            &params,
            (n_tokens * n_heads) as u64,
        );
    }

    /// Shared body for the three (src → dst) encode/rotate dispatches: same
    /// binding shape, same 1-D grid, different pipeline and params.
    fn dispatch(
        &self,
        enc: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        src: &Buffer,
        dst: &Buffer,
        params: &TqParams,
        groups: u64,
    ) {
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_buffer(2, Some(&self.signs), 0);
        params.set(enc, 3);
        // One threadgroup per (row, head). Metal's per-dimension grid limit is
        // far above any batch this sees, so unlike WGSL there is no spill into Y
        // and no `get_wid` flattening to mirror.
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(TQ_THREADS, 1, 1));
    }

    /// Causal attention for `n_tokens` query rows against the compressed cache.
    /// `out` receives `n_tokens × (n_heads * head_dim)` floats (heads
    /// concatenated per row).
    ///
    /// `scale` is the softmax scale — `1/sqrt(head_dim)` for every arch except
    /// Granite, which overrides it via `scalars.attn`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
        out: &Buffer,
        n_tokens: usize,
        n_heads: usize,
        start_pos: usize,
        scale: f32,
    ) {
        let l = self
            .layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .expect("attention on a layer without a compressed cache");
        let head_dim = self.layout.head_dim;
        let params = TqAttnParams {
            n_heads: n_heads as u32,
            n_kv_heads: l.n_kv_heads as u32,
            head_dim: head_dim as u32,
            // Causal clamp — the live length after this batch, NOT the region
            // stride. `cache_cap` below is the stride; conflating the two reads
            // the wrong head's slots.
            max_seq: (start_pos + n_tokens) as u32,
            start_pos: start_pos as u32,
            scale,
            q_cap: self.q_cap as u32,
            out_stride: (n_heads * head_dim) as u32,
            qjl_scale: crate::turboquant::qjl_scale(head_dim),
            sign_off: (layer * 2 * head_dim) as u32,
            c0: self.config.centroids[0],
            c1: self.config.centroids[1],
            c2: self.config.centroids[2],
            c3: self.config.centroids[3],
            // The whole batch is dispatched at once, so row 0 of the
            // rotated-query scratch is row 0 of this batch.
            q_base: 0,
            cache_cap: self.max_seq_len as u32,
        };
        enc.set_compute_pipeline_state(&self.pipelines.attention);
        enc.set_buffer(0, Some(&self.qrot), 0);
        enc.set_buffer(1, Some(&l.keys), 0);
        enc.set_buffer(2, Some(&l.values), 0);
        enc.set_buffer(3, Some(out), 0);
        params.set(enc, 4);
        enc.set_buffer(5, Some(&self.signs), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, n_tokens as u64, 1),
            MTLSize::new(TQ_ATTN_THREADS, 1, 1),
        );
    }
}

// ── Prefix-cache snapshot / restore ────────────────────────────────────────
//
// The packed layout was chosen so this is cheap: the 2-bit and 1-bit words are
// LSB-first, which makes their little-endian bytes identical to the CPU's
// `pack_2bit` / `pack_1bit` output, and each region is contiguous across
// timesteps within a head. The blob itself is produced by the same
// `encode_compressed_*` the CPU uses — one format, one implementation, so the
// blobs a Metal cache produces decode exactly on CPU or wgpu.
//
// Note what that does and does not mean: the *format* is shared, and the oracle
// suite pins it by asserting byte equality against the CPU encoder. Actual
// cross-backend prefix-cache *reuse* does not happen — `cache_namespace` puts each
// backend in its own namespace on purpose — so this is a property of the encoding,
// not a feature anyone can rely on to warm a Metal session from a CPU one.
//
// Unlike the wgpu mirror there is no gather pass and no staging buffer: these
// are `StorageModeShared` allocations, so the mapped contents are the cache.

impl TqMetalCache {
    /// The `[0, seq_len)` slice of one region, as `u32` words.
    ///
    /// Bounds-checked with a real `assert!`, matching [`write_words`]: reading
    /// past a `StorageModeShared` mapping is as much UB as writing past it, and a
    /// snapshot runs exactly as rarely as a restore (once per prefix-cache
    /// insert), so the check costs nothing measurable. `debug_assert!` here would
    /// leave release — the build that ships — as the unvalidated one.
    ///
    /// # Safety contract
    ///
    /// The caller must have waited on every command buffer that writes `buf`.
    /// Both call sites run under `infer_lock` after the forward pass's
    /// `wait_until_completed`, the same contract the f16 path's `contents()`
    /// reads already rely on.
    fn region(buf: &Buffer, word_off: usize, words: usize) -> &[u32] {
        let end = (word_off + words) * 4;
        assert!(
            end as u64 <= buf.length(),
            "snapshot would read to byte {end} of a {}-byte buffer",
            buf.length()
        );
        unsafe { std::slice::from_raw_parts((buf.contents() as *const u32).add(word_off), words) }
    }

    /// Read one layer's live `[0, seq_len)` cache back and encode it into the
    /// `(TQK1, TQV1)` blob pair that `LayerSnapshot::AttentionCompressed`
    /// carries.
    pub fn snapshot_layer(&self, layer: usize, seq_len: usize) -> (Vec<u8>, Vec<u8>) {
        let l = self
            .layer(layer)
            .expect("snapshot_layer on a layer without a compressed cache");
        // Mirrors `restore_layer`'s `seq_len > max_seq_len` rejection. Without it a
        // caller passing a stale longer length would read past the live region and,
        // for the last head, past the allocation.
        assert!(
            seq_len <= self.max_seq_len,
            "snapshot_layer: seq_len {seq_len} exceeds the cache capacity {}",
            self.max_seq_len
        );
        let head_dim = self.layout.head_dim;
        let n = l.n_kv_heads;
        let mut keys = CompressedKeyCache::new(n, head_dim, seq_len);
        let mut values = CompressedValueCache::new(n, head_dim, seq_len);

        // An empty cache still needs a well-formed (header-only) blob.
        if seq_len > 0 {
            let pw = self.layout.polar_words;
            let jw = self.layout.jl_words;
            let cap = self.max_seq_len;
            let (jl_off, norm_off) = self.layout.key_regions(n * cap);
            let v_norm_off = self.layout.value_norm_offset(n * cap);

            for h in 0..n {
                let k_polar = Self::region(&l.keys, h * cap * pw, seq_len * pw);
                let k_jl = Self::region(&l.keys, jl_off + h * cap * jw, seq_len * jw);
                let k_norms = Self::region(&l.keys, norm_off + h * cap, seq_len);
                let v_polar = Self::region(&l.values, h * cap * pw, seq_len * pw);
                let v_norms = Self::region(&l.values, v_norm_off + h * cap, seq_len);
                for t in 0..seq_len {
                    // Both key norms ride in one word: the kernel packs
                    // `(norm, residual_norm)` as two f16 with `norm` in the low
                    // half.
                    let nw = k_norms[t];
                    keys.append(
                        h,
                        bytemuck::cast_slice(&k_polar[t * pw..(t + 1) * pw]),
                        bytemuck::cast_slice(&k_jl[t * jw..(t + 1) * jw]),
                        (nw & 0xFFFF) as u16,
                        (nw >> 16) as u16,
                    );
                    values.append(
                        h,
                        bytemuck::cast_slice(&v_polar[t * pw..(t + 1) * pw]),
                        (v_norms[t] & 0xFFFF) as u16,
                    );
                }
            }
        }
        (
            encode_compressed_keys(&keys),
            encode_compressed_values(&values),
        )
    }

    /// Upload a `(TQK1, TQV1)` blob pair into one layer's cache. Inverse of
    /// [`Self::snapshot_layer`].
    ///
    /// Returns `None` — without touching the cache — when the blobs don't decode
    /// or don't match this layer's shape, so the caller can treat it as a
    /// prefix-cache miss instead of restoring a cache the kernels will misread.
    /// On success returns the restored sequence length, which the caller should
    /// cross-check against the snapshot's own `seq_len`.
    pub fn restore_layer(
        &self,
        layer: usize,
        keys_blob: &[u8],
        values_blob: &[u8],
    ) -> Option<usize> {
        let l = self.layer(layer)?;
        let keys = decode_compressed_keys(keys_blob)?;
        let values = decode_compressed_values(values_blob)?;
        // Shape gate lives on `TqLayout` so both GPU backends apply the same one.
        let seq_len = self
            .layout
            .blobs_match(&keys, &values, l.n_kv_heads, self.max_seq_len)?;
        if seq_len == 0 {
            return Some(0);
        }

        let pw = self.layout.polar_words;
        let jw = self.layout.jl_words;
        let cap = self.max_seq_len;
        let n = l.n_kv_heads;
        let (jl_off, norm_off) = self.layout.key_regions(n * cap);
        let v_norm_off = self.layout.value_norm_offset(n * cap);

        // Each region is contiguous across timesteps within a head, so one copy
        // per (head, region) covers it. Words past `seq_len` are left as-is; the
        // kernels only read up to the restored seq_len.
        let mut norm_words = Vec::with_capacity(seq_len);
        for h in 0..n {
            write_words(&l.keys, h * cap * pw, &keys.polar_data[h]);
            write_words(&l.keys, jl_off + h * cap * jw, &keys.jl_data[h]);
            norm_words.clear();
            for t in 0..seq_len {
                norm_words.push(
                    u32::from(keys.norms[h][t]) | (u32::from(keys.residual_norms[h][t]) << 16),
                );
            }
            write_words(
                &l.keys,
                norm_off + h * cap,
                bytemuck::cast_slice(&norm_words),
            );

            write_words(&l.values, h * cap * pw, &values.polar_data[h]);
            norm_words.clear();
            norm_words.extend(values.norms[h].iter().map(|&b| u32::from(b)));
            write_words(
                &l.values,
                v_norm_off + h * cap,
                bytemuck::cast_slice(&norm_words),
            );
        }
        Some(seq_len)
    }
}

/// Compile one entry point, mapping the backend's `anyhow` error into the typed
/// [`CeraError`] `configure_kv_compression` returns.
fn pipeline(
    ctx: &MetalContext,
    src: &'static str,
    entry: &str,
) -> Result<ComputePipelineState, CeraError> {
    ctx.create_pipeline(src, entry)
        .map_err(|e| CeraError::Backend(format!("TurboQuant kernel '{entry}': {e}")))
}

/// Copy `bytes` into `buf` at a `u32`-word offset.
///
/// Bounds-checked with a real `assert!`: this writes through a raw pointer into
/// a shared allocation, so an off-by-one in a region offset would corrupt
/// unrelated cache slots (or heap past the buffer) rather than fail. Restore
/// runs once per prefix-cache hit, so the check is free.
fn write_words(buf: &Buffer, word_off: usize, bytes: &[u8]) {
    let end = word_off * 4 + bytes.len();
    assert!(
        end as u64 <= buf.length(),
        // `end` is the exclusive end offset, not a length — say so, or a triggered
        // assert misstates the size by the region offset.
        "restore would write to byte {end} of a {}-byte buffer",
        buf.length()
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buf.contents() as *mut u8).add(word_off * 4),
            bytes.len(),
        );
    }
}

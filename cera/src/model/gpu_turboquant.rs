//! GPU-resident TurboQuant KV cache for the wgpu backend.
//!
//! Holds everything the compressed path needs that isn't already on
//! [`GpuLfm2Model`](super::gpu_lfm2::GpuLfm2Model): the packed per-layer key and
//! value buffers, the per-layer randomized-Hadamard sign flips, the rotated-query
//! scratch, the shader params slab, and the four pipelines. Built lazily by
//! `Model::configure_kv_compression` so a session that never asks for TurboQuant
//! pays neither the buffers nor the shader compilation.
//!
//! The kernels and the packed layout are documented in
//! `backend/shaders/turboquant.wgsl`; this module is the host mirror of that
//! layout and the single place the two are kept in sync.
//!
//! Compression rate, `head_dim = 128`: keys 52 bytes/vector (32 polar + 16 JL +
//! 2 + 2 norms) vs 512 as f32, values 36 vs 512 — about 11.6× less KV overall.

use crate::CeraError;
use crate::backend::wgpu::GpuContext;
use crate::kv_cache::checked_elems;
use crate::model::{BlockType, ModelConfig};
use crate::turboquant::{
    CompressedKeyCache, CompressedValueCache, RotationState, TurboQuantConfig,
    decode_compressed_keys, decode_compressed_values, encode_compressed_keys,
    encode_compressed_values,
};
// The packed-layout and mode types are backend-agnostic and live with the
// algorithm; re-exported so `model::gpu_turboquant::{TqLayout, TqMode}` stays a
// valid path for callers and tests. `describe_kv_mode` is crate-internal (an
// error-message formatter, not API) so it is imported, not re-exported.
pub(crate) use crate::turboquant::describe_kv_mode;
pub use crate::turboquant::{TqLayout, TqMode, head_dim_supported};

/// Params slots per attention layer in the shared slab: encode-keys,
/// encode-values, rotate-q, attention.
const SLOTS_PER_LAYER: usize = 4;
const SLOT_ENCODE_KEYS: usize = 0;
const SLOT_ENCODE_VALUES: usize = 1;
const SLOT_ROTATE_Q: usize = 2;
const SLOT_ATTENTION: usize = 3;

/// Bytes actually consumed by either params struct. The slab pads each slot out
/// to the device's storage-binding offset alignment; this is the size the bind
/// group binds, so it must equal the struct the shader reads. Asserted against
/// `TqParams` below (the `TqAttnParams` `[u32; 16]` encoding is the same width),
/// mirroring the `size_of` guards in `backend/metal/params.rs`.
const PARAMS_BYTES: usize = 64;

/// Encode / rotate params. Mirrors the `TqParams` struct in `turboquant.wgsl` —
/// all 16 fields are 4-byte scalars, so the WGSL and Rust layouts agree without
/// padding surprises. This struct is the single source of truth for that layout;
/// the shader-oracle test drives the kernels through it so a field reorder can't
/// pass the test while breaking production.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TqParams {
    /// Rows in this batch (1 for decode).
    pub n_tokens: u32,
    /// KV heads for the encode kernels, query heads for `tq_rotate_q`.
    pub n_heads: u32,
    pub head_dim: u32,
    /// Elements per row in the source buffer.
    pub src_stride: u32,
    /// Cache timestep the first row lands at. Unused by `tq_rotate_q`.
    pub dst_pos: u32,
    /// Cache capacity in timesteps — the per-head stride of every region.
    pub max_seq_len: u32,
    /// Offset of this layer's polar signs; the JL signs follow `head_dim` later.
    pub sign_off: u32,
    /// Row capacity of the rotated-query scratch. `tq_rotate_q` only.
    pub q_cap: u32,
    /// Lloyd-Max centroids, ascending. Unused by `tq_rotate_q`.
    pub c0: f32,
    pub c1: f32,
    pub c2: f32,
    pub c3: f32,
    /// The 3 decision boundaries between centroids. Unused by `tq_rotate_q`.
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<TqParams>() == PARAMS_BYTES);

impl TqParams {
    /// Fill the centroid + boundary fields from a [`TurboQuantConfig`].
    pub fn with_quant_config(mut self, config: &TurboQuantConfig) -> Self {
        let c = &config.centroids;
        let b = &config.boundaries;
        (self.c0, self.c1, self.c2, self.c3) = (c[0], c[1], c[2], c[3]);
        (self.b0, self.b1, self.b2) = (b[0], b[1], b[2]);
        self
    }
}

/// Params for `flash_attention_tq`. Mirrors the `params: array<u32, 16>` the
/// kernel reads; [`Self::to_u32_array`] is the single place that ordering lives.
#[derive(Clone, Copy)]
pub struct TqAttnParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Causal clamp: `start_pos + n_queries`. NOT the cache stride.
    pub max_seq: u32,
    /// Absolute position of query row 0.
    pub start_pos: u32,
    pub scale: f32,
    /// Row capacity of the rotated-query scratch.
    pub q_cap: u32,
    /// Elements per row in the output buffer.
    pub out_stride: u32,
    /// QJL inner-product estimator scale, `sqrt(pi/2) / head_dim`.
    pub qjl_scale: f32,
    pub sign_off: u32,
    pub centroids: [f32; 4],
    /// Index of the first query row of this dispatch within the batch.
    pub q_base: u32,
    /// Cache capacity in timesteps — the per-head stride of every region.
    pub cache_cap: u32,
}

impl TqAttnParams {
    pub fn to_u32_array(self) -> [u32; 16] {
        [
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq,
            self.start_pos,
            self.scale.to_bits(),
            self.q_cap,
            self.out_stride,
            self.qjl_scale.to_bits(),
            self.sign_off,
            self.centroids[0].to_bits(),
            self.centroids[1].to_bits(),
            self.centroids[2].to_bits(),
            self.centroids[3].to_bits(),
            self.q_base,
            self.cache_cap,
        ]
    }

    /// The QJL estimator scale for a given `head_dim`. Delegates to
    /// [`crate::turboquant::qjl_scale`] so there is one definition.
    pub fn qjl_scale_for(head_dim: usize) -> f32 {
        crate::turboquant::qjl_scale(head_dim)
    }
}

/// The four compressed-path pipelines.
struct TqPipelines {
    encode_keys: wgpu::ComputePipeline,
    encode_values: wgpu::ComputePipeline,
    rotate_q: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
}

/// One attention layer's packed caches.
pub struct TqLayerCache {
    /// `[polar | jl | norms]`, `n_kv_heads * max_seq_len` slots per region.
    pub keys: wgpu::Buffer,
    /// `[polar | norms]`.
    pub values: wgpu::Buffer,
    pub n_kv_heads: usize,
}

/// GPU-resident compressed KV cache plus the machinery to drive it.
pub struct TqGpuCache {
    pub mode: TqMode,
    pub layout: TqLayout,
    /// Per model layer; `None` for conv (non-attention) layers.
    pub layers: Vec<Option<TqLayerCache>>,
    /// All layers' RHT sign flips, `[polar | jl]` per layer (`2 * head_dim`
    /// f32 each). One buffer keeps the kernels at a single sign binding; the
    /// per-layer offset rides in the params.
    signs: wgpu::Buffer,
    /// Rotated queries: `[q_rot | q_jl | sums]`, `q_cap` rows per region.
    qrot: wgpu::Buffer,
    /// Query rows the `qrot` scratch can hold — the prefill chunk size.
    q_cap: usize,
    /// Params for every (layer, slot), padded to the device's binding alignment.
    params: wgpu::Buffer,
    slot_stride: usize,
    /// Lloyd-Max centroids + boundaries for this `head_dim`.
    pub config: TurboQuantConfig,
    pipelines: TqPipelines,
    max_seq_len: usize,
}

impl TqGpuCache {
    /// Allocate the compressed caches and compile the pipelines.
    ///
    /// `q_cap` is the largest query batch the attention kernel will see (the
    /// prefill chunk size); the rotated-query scratch is sized for it.
    ///
    /// The per-layer rotations come from the same
    /// `RotationState::try_from_seed(seed ^ layer_idx, head_dim)` the CPU uses,
    /// so a cache written on one backend is readable by the other — which is what
    /// makes the cross-backend prefix-cache snapshot meaningful.
    pub fn new(
        ctx: &GpuContext,
        config: &ModelConfig,
        max_seq_len: usize,
        q_cap: usize,
        mode: TqMode,
    ) -> Result<Self, CeraError> {
        let head_dim = config.head_dim;
        // The in-crate caller gates on `TqMode::from_compression`, but this type is
        // `pub` (the oracle tests construct it directly), so surface a typed error
        // rather than aborting the caller's process.
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
                        keys: ctx.create_storage_rw(k_bytes, &format!("l{i}.tq_keys")),
                        values: ctx.create_storage_rw(v_bytes, &format!("l{i}.tq_values")),
                        n_kv_heads,
                    }));
                }
                BlockType::GatedConv => layers.push(None),
            }
        }

        // Rotated-query scratch: two head-wide regions plus one sum per (row, head).
        let q_rows = checked_elems::<f32>(q_cap, config.n_heads)?;
        let qrot_floats = checked_elems::<f32>(q_rows, 2 * head_dim + 1)?;
        let qrot = ctx.create_storage_rw(TqLayout::words_to_bytes(qrot_floats), "tq_qrot");

        // One params slot per (layer, purpose), each padded out to the device's
        // storage-binding offset alignment so a bind group can address it.
        let align = ctx.min_storage_buffer_offset_alignment.max(4) as usize;
        let slot_stride = PARAMS_BYTES.next_multiple_of(align);
        let params = ctx.create_storage_rw(
            (n_layers * SLOTS_PER_LAYER * slot_stride) as u64,
            "tq_params",
        );

        let pipelines = TqPipelines {
            encode_keys: ctx.create_pipeline(
                crate::backend::wgpu::shaders::TURBOQUANT,
                "tq_encode_keys",
                "tq_encode_keys",
            ),
            encode_values: ctx.create_pipeline(
                crate::backend::wgpu::shaders::TURBOQUANT,
                "tq_encode_values",
                "tq_encode_values",
            ),
            rotate_q: ctx.create_pipeline(
                crate::backend::wgpu::shaders::TURBOQUANT,
                "tq_rotate_q",
                "tq_rotate_q",
            ),
            attention: ctx.create_pipeline(
                crate::backend::wgpu::shaders::FLASH_ATTENTION_TQ,
                "flash_attention_tq",
                "flash_attention_tq",
            ),
        };

        Ok(Self {
            mode,
            layout,
            layers,
            signs: ctx.upload_f32(&signs, "tq_signs"),
            qrot,
            q_cap,
            params,
            slot_stride,
            config: TurboQuantConfig::for_head_dim(head_dim),
            pipelines,
            max_seq_len,
        })
    }

    /// This layer's cache, or `None` for a conv layer.
    pub fn layer(&self, layer: usize) -> Option<&TqLayerCache> {
        self.layers[layer].as_ref()
    }

    /// Stage the params for every layer of one forward pass, in a single
    /// `write_buffer`.
    ///
    /// Must be called BEFORE the command encoder carrying the dispatches is
    /// submitted, and once per forward: `write_buffer` writes are applied in
    /// submission order, so a per-dispatch write into a shared range would let
    /// every dispatch in the encoder observe only the last value. Distinct slots
    /// per (layer, purpose) is what makes one write sufficient.
    ///
    /// - `n_tokens` — rows in this batch (1 for decode).
    /// - `start_pos` — absolute position of row 0.
    /// - `scale` — softmax scale (Granite overrides `1/sqrt(head_dim)`).
    ///
    /// Every row stride is derived rather than passed: the Q source and the
    /// attention output are both `q_dim = n_heads * head_dim` wide (heads
    /// concatenated per row), and each layer's K/V rows are `n_kv_heads *
    /// head_dim`. Deriving them keeps a caller from pairing this batch's params
    /// with another layout's strides, and is correct on models whose KV head count
    /// varies by layer.
    pub fn write_params(
        &self,
        ctx: &GpuContext,
        config: &ModelConfig,
        n_tokens: usize,
        start_pos: usize,
        scale: f32,
    ) {
        let head_dim = self.layout.head_dim;
        let q_dim = config.n_heads * head_dim;
        let mut slab = vec![0u8; self.layers.len() * SLOTS_PER_LAYER * self.slot_stride];

        for (i, layer) in self.layers.iter().enumerate() {
            let Some(l) = layer else { continue };
            let sign_off = (i * 2 * head_dim) as u32;
            let base = TqParams {
                n_tokens: n_tokens as u32,
                n_heads: l.n_kv_heads as u32,
                head_dim: head_dim as u32,
                src_stride: (l.n_kv_heads * head_dim) as u32,
                dst_pos: start_pos as u32,
                max_seq_len: self.max_seq_len as u32,
                sign_off,
                q_cap: self.q_cap as u32,
                ..Default::default()
            }
            .with_quant_config(&self.config);
            self.put(&mut slab, i, SLOT_ENCODE_KEYS, bytemuck::bytes_of(&base));
            self.put(&mut slab, i, SLOT_ENCODE_VALUES, bytemuck::bytes_of(&base));
            // Rotation runs over QUERY heads and reads the Q buffer.
            let rot = TqParams {
                n_heads: config.n_heads as u32,
                src_stride: q_dim as u32,
                ..base
            };
            self.put(&mut slab, i, SLOT_ROTATE_Q, bytemuck::bytes_of(&rot));
            let attn = TqAttnParams {
                n_heads: config.n_heads as u32,
                n_kv_heads: l.n_kv_heads as u32,
                head_dim: head_dim as u32,
                max_seq: (start_pos + n_tokens) as u32,
                start_pos: start_pos as u32,
                scale,
                q_cap: self.q_cap as u32,
                out_stride: q_dim as u32,
                qjl_scale: TqAttnParams::qjl_scale_for(head_dim),
                sign_off,
                centroids: self.config.centroids,
                // The whole batch is dispatched at once, so row 0 of the
                // rotated-query scratch is row 0 of this batch.
                q_base: 0,
                cache_cap: self.max_seq_len as u32,
            };
            self.put(
                &mut slab,
                i,
                SLOT_ATTENTION,
                bytemuck::cast_slice(&attn.to_u32_array()),
            );
        }
        ctx.queue.write_buffer(&self.params, 0, &slab);
    }

    fn put(&self, slab: &mut [u8], layer: usize, slot: usize, bytes: &[u8]) {
        let off = (layer * SLOTS_PER_LAYER + slot) * self.slot_stride;
        slab[off..off + bytes.len()].copy_from_slice(bytes);
    }

    fn params_binding(&self, layer: usize, slot: usize) -> wgpu::BindingResource<'_> {
        let off = ((layer * SLOTS_PER_LAYER + slot) * self.slot_stride) as u64;
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.params,
            offset: off,
            size: Some(std::num::NonZeroU64::new(PARAMS_BYTES as u64).unwrap()),
        })
    }

    /// Compress `n_tokens` rows of K and V for `layer` into the packed cache,
    /// replacing the f32 path's two `copy_buffer_to_buffer` writes.
    ///
    /// `k_src` / `v_src` are row-major `[n_tokens × kv_src_stride]` post-RoPE
    /// projections; rows land at cache timesteps `[start_pos, start_pos +
    /// n_tokens)` as staged by [`Self::write_params`].
    pub fn encode_kv(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        layer: usize,
        k_src: &wgpu::Buffer,
        v_src: &wgpu::Buffer,
        n_tokens: usize,
    ) {
        let l = self
            .layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .expect("encode_kv on a layer without a compressed cache");
        let groups = n_tokens * l.n_kv_heads;
        self.dispatch(
            ctx,
            enc,
            &self.pipelines.encode_keys,
            k_src,
            &l.keys,
            layer,
            SLOT_ENCODE_KEYS,
            groups,
            "tq_encode_keys",
        );
        self.dispatch(
            ctx,
            enc,
            &self.pipelines.encode_values,
            v_src,
            &l.values,
            layer,
            SLOT_ENCODE_VALUES,
            groups,
            "tq_encode_values",
        );
    }

    /// Pre-rotate `n_tokens × n_heads` query heads into the `qrot` scratch.
    /// Must run before [`Self::attention`] for the same layer.
    pub fn rotate_queries(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        layer: usize,
        q_src: &wgpu::Buffer,
        n_tokens: usize,
        n_heads: usize,
    ) {
        assert!(
            n_tokens <= self.q_cap,
            "rotate_queries: {n_tokens} rows exceeds the qrot scratch capacity {}",
            self.q_cap
        );
        self.dispatch(
            ctx,
            enc,
            &self.pipelines.rotate_q,
            q_src,
            &self.qrot,
            layer,
            SLOT_ROTATE_Q,
            n_tokens * n_heads,
            "tq_rotate_q",
        );
    }

    /// Shared body for the three (src → dst) encode/rotate dispatches: same
    /// bind-group shape, same 2-D workgroup grid, different pipeline and slot.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        layer: usize,
        slot: usize,
        groups: usize,
        label: &str,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.signs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_binding(layer, slot),
                },
            ],
        });
        // One workgroup per (row, head); the grid spills into Y past MAX_WG and
        // the kernel recovers the flat index with `get_wid`.
        // Shared with the GEMV kernels' `get_wid` contract — one helper so the
        // host-side flattening can't drift from what the shaders decode.
        let grid = crate::backend::wgpu::gemv_row_workgroups(groups as u32);
        let ts = ctx.begin_profile_span(label);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: ts,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
    }

    /// Causal attention for `n_tokens` query rows against the compressed cache.
    /// `out` receives `n_tokens × out_stride` floats (heads concatenated per row).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        layer: usize,
        out: &wgpu::Buffer,
        n_tokens: usize,
        n_heads: usize,
    ) {
        let l = self
            .layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .expect("attention on a layer without a compressed cache");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flash_attention_tq"),
            layout: &self.pipelines.attention.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.qrot.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: l.keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: l.values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_binding(layer, SLOT_ATTENTION),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.signs.as_entire_binding(),
                },
            ],
        });
        let ts = ctx.begin_profile_span("flash_attention_tq");
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flash_attention_tq"),
            timestamp_writes: ts,
        });
        pass.set_pipeline(&self.pipelines.attention);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_heads as u32, n_tokens as u32, 1);
    }
}

// ── Prefix-cache snapshot / restore ────────────────────────────────────────
//
// The packed GPU layout was chosen so this is cheap: the 2-bit and 1-bit words
// are LSB-first, which makes their little-endian bytes identical to the CPU's
// `pack_2bit` / `pack_1bit` output, and each region is contiguous across
// timesteps within a head. So a snapshot is a gather of `3 × n_kv_heads`
// sub-ranges (for keys; 2 for values) plus an f16-unpack of the norm words, and
// the blob itself is produced by the same `encode_compressed_*` the CPU uses —
// one format, one implementation, so a snapshot written on GPU is loadable on CPU
// and vice versa.

impl TqGpuCache {
    /// Read one layer's live `[0, seq_len)` cache back and encode it into the
    /// `(TQK1, TQV1)` blob pair that `LayerSnapshot::AttentionCompressed` carries.
    ///
    /// The live slice of every region is first compacted GPU-side into one
    /// scratch buffer, so this costs a single readback per layer — the same as
    /// the f32 path's two, over ~11× fewer bytes.
    pub fn snapshot_layer(
        &self,
        ctx: &GpuContext,
        layer: usize,
        seq_len: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let l = self
            .layer(layer)
            .expect("snapshot_layer on a layer without a compressed cache");
        let head_dim = self.layout.head_dim;
        let n = l.n_kv_heads;

        // An empty cache still needs a well-formed (header-only) blob.
        if seq_len == 0 {
            let keys = CompressedKeyCache::new(n, head_dim, 0);
            let values = CompressedValueCache::new(n, head_dim, 0);
            return (
                encode_compressed_keys(&keys),
                encode_compressed_values(&values),
            );
        }

        let pw = self.layout.polar_words;
        let jw = self.layout.jl_words;
        let cap = self.max_seq_len;
        let (jl_off, norm_off) = self.layout.key_regions(n * cap);
        let v_norm_off = self.layout.value_norm_offset(n * cap);

        // Scratch layout mirrors the read order below: key polar, key jl, key
        // norms, value polar, value norms — every head's live slice back to back.
        let k_polar = seq_len * pw;
        let k_jl = seq_len * jw;
        let total = n * (k_polar + k_jl + seq_len) + n * (k_polar + seq_len);
        let scratch = ctx.create_storage_rw((total * 4) as u64, "tq_snapshot_gather");

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tq_snapshot_gather"),
            });
        let mut dst = 0usize;
        let mut gather = |src: &wgpu::Buffer, src_word: usize, words: usize, dst: &mut usize| {
            enc.copy_buffer_to_buffer(
                src,
                (src_word * 4) as u64,
                &scratch,
                (*dst * 4) as u64,
                (words * 4) as u64,
            );
            *dst += words;
        };
        for h in 0..n {
            gather(&l.keys, h * cap * pw, k_polar, &mut dst);
        }
        for h in 0..n {
            gather(&l.keys, jl_off + h * cap * jw, k_jl, &mut dst);
        }
        for h in 0..n {
            gather(&l.keys, norm_off + h * cap, seq_len, &mut dst);
        }
        for h in 0..n {
            gather(&l.values, h * cap * pw, k_polar, &mut dst);
        }
        for h in 0..n {
            gather(&l.values, v_norm_off + h * cap, seq_len, &mut dst);
        }
        debug_assert_eq!(dst, total, "snapshot gather did not fill the scratch");
        ctx.queue.submit(Some(enc.finish()));

        let words = ctx.download_u32(&scratch, total);

        // Reassemble into the CPU cache types via their append API, so the blob
        // encoders stay the single source of truth for the on-disk format.
        let mut keys = CompressedKeyCache::new(n, head_dim, seq_len);
        let mut values = CompressedValueCache::new(n, head_dim, seq_len);
        let kp = 0;
        let kj = kp + n * k_polar;
        let kn = kj + n * k_jl;
        let vp = kn + n * seq_len;
        let vn = vp + n * k_polar;
        for h in 0..n {
            for t in 0..seq_len {
                let polar = &words[kp + h * k_polar + t * pw..][..pw];
                let jl = &words[kj + h * k_jl + t * jw..][..jw];
                // Both norms ride in one word: `pack2x16float(vec2(norm,
                // residual_norm))` puts `norm` in the low half.
                let nw = words[kn + h * seq_len + t];
                keys.append(
                    h,
                    bytemuck::cast_slice(polar),
                    bytemuck::cast_slice(jl),
                    (nw & 0xFFFF) as u16,
                    (nw >> 16) as u16,
                );
                let v_polar = &words[vp + h * k_polar + t * pw..][..pw];
                values.append(
                    h,
                    bytemuck::cast_slice(v_polar),
                    (words[vn + h * seq_len + t] & 0xFFFF) as u16,
                );
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
        ctx: &GpuContext,
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

        // Each region is contiguous across timesteps within a head, so one write
        // per (head, region) covers it. Bytes past `seq_len` are left as-is; the
        // kernels only read up to the restored seq_len.
        let mut norm_words = Vec::with_capacity(seq_len);
        for h in 0..n {
            ctx.queue
                .write_buffer(&l.keys, ((h * cap * pw) * 4) as u64, &keys.polar_data[h]);
            ctx.queue.write_buffer(
                &l.keys,
                ((jl_off + h * cap * jw) * 4) as u64,
                &keys.jl_data[h],
            );
            norm_words.clear();
            for t in 0..seq_len {
                norm_words.push(
                    u32::from(keys.norms[h][t]) | (u32::from(keys.residual_norms[h][t]) << 16),
                );
            }
            ctx.queue.write_buffer(
                &l.keys,
                ((norm_off + h * cap) * 4) as u64,
                bytemuck::cast_slice(&norm_words),
            );

            ctx.queue.write_buffer(
                &l.values,
                ((h * cap * pw) * 4) as u64,
                &values.polar_data[h],
            );
            norm_words.clear();
            norm_words.extend(values.norms[h].iter().map(|&b| u32::from(b)));
            ctx.queue.write_buffer(
                &l.values,
                ((v_norm_off + h * cap) * 4) as u64,
                bytemuck::cast_slice(&norm_words),
            );
        }
        Some(seq_len)
    }
}

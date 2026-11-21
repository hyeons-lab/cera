//! GPU-resident TurboQuant KV cache for the wgpu backend.
//!
//! Holds everything the compressed path needs that isn't already on
//! [`GpuLfmModel`](super::gpu_lfm2::GpuLfmModel): the packed per-layer key and
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

/// Orchestration seam behind [`TqGpuCache::new`] so host tests can verify pipeline
/// validation runs before VRAM allocations, post-allocation lost-device checks abort, and
/// errors map cleanly to [`CeraError::Backend`].
fn init_tq_cache_with<P, B>(
    build_pipelines: impl FnOnce() -> anyhow::Result<P>,
    alloc_buffers: impl FnOnce(P) -> Result<B, CeraError>,
    ensure_alive: impl FnOnce(&str) -> anyhow::Result<()>,
) -> Result<B, CeraError> {
    let pipelines = build_pipelines().map_err(|e| CeraError::Backend(format!("{e:#}")))?;
    let cache = alloc_buffers(pipelines)?;
    ensure_alive("tq allocation").map_err(|e| CeraError::Backend(format!("{e:#}")))?;
    Ok(cache)
}

struct TqCachePlan {
    layout: TqLayout,
    sign_elems: usize,
    layer_specs: Vec<Option<(usize, u64, u64)>>,
    qrot_bytes: u64,
    slot_stride: usize,
    params_bytes: u64,
}

fn validate_tq_buffer_limit(
    bytes: u64,
    label: &str,
    max_buffer_size: u64,
    max_storage_binding_size: u64,
) -> Result<(), CeraError> {
    if bytes == 0 {
        return Err(CeraError::Backend(format!(
            "turboquant {label} buffer size must be > 0"
        )));
    }
    let effective_limit = max_buffer_size.min(max_storage_binding_size);
    if bytes > effective_limit {
        return Err(CeraError::Backend(format!(
            "turboquant {label} buffer ({bytes} bytes) exceeds GPU limit ({effective_limit} bytes; max_buffer_size={max_buffer_size}, max_storage_buffer_binding_size={max_storage_binding_size})"
        )));
    }
    Ok(())
}

fn tq_checked_elems<T>(a: usize, b: usize, label: &str) -> Result<usize, CeraError> {
    checked_elems::<T>(a, b).map_err(|_| {
        CeraError::Backend(format!(
            "turboquant {label} element count overflow ({a} * {b})"
        ))
    })
}

fn checked_words_to_bytes(words: usize, label: &str) -> Result<u64, CeraError> {
    u64::try_from(words)
        .ok()
        .and_then(|w| w.checked_mul(4))
        .ok_or_else(|| CeraError::Backend(format!("turboquant {label} byte count overflow")))
}

/// Pre-validate geometry, u32 shader parameter ranges, overflow bounds, and GPU buffer
/// limits before compiling pipelines, generating signs, or allocating VRAM.
fn prevalidate_tq_cache_specs(
    config: &ModelConfig,
    max_seq_len: usize,
    q_cap: usize,
    min_storage_align: u64,
    max_buffer_size: u64,
    max_storage_binding_size: u64,
) -> Result<TqCachePlan, CeraError> {
    let head_dim = config.head_dim;
    // The in-crate caller gates on `TqMode::from_compression`, but this type is
    // `pub` (the oracle tests construct it directly), so surface a typed error
    // rather than aborting the caller's process.
    if !head_dim_supported(head_dim) {
        return Err(crate::turboquant::unsupported_head_dim(head_dim));
    }
    let n_layers = config.block_types.len();
    if n_layers == 0 {
        return Err(CeraError::Backend(
            "turboquant requires n_layers > 0".into(),
        ));
    }
    if max_seq_len == 0 {
        return Err(CeraError::Backend(
            "turboquant requires max_seq_len > 0".into(),
        ));
    }
    u32::try_from(max_seq_len).map_err(|_| {
        CeraError::Backend(format!("turboquant max_seq_len {max_seq_len} exceeds u32"))
    })?;
    if q_cap == 0 {
        return Err(CeraError::Backend("turboquant requires q_cap > 0".into()));
    }
    u32::try_from(q_cap)
        .map_err(|_| CeraError::Backend(format!("turboquant q_cap {q_cap} exceeds u32")))?;
    if config.n_heads == 0 {
        return Err(CeraError::Backend("turboquant requires n_heads > 0".into()));
    }
    let q_dim = config
        .n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| CeraError::Backend("turboquant q_dim overflow".into()))?;
    u32::try_from(q_dim)
        .map_err(|_| CeraError::Backend(format!("turboquant q_dim {q_dim} exceeds u32")))?;

    let signs_per_layer = head_dim
        .checked_mul(2)
        .ok_or_else(|| CeraError::Backend("turboquant signs_per_layer overflow".into()))?;
    let sign_elems = n_layers
        .checked_mul(signs_per_layer)
        .ok_or_else(|| CeraError::Backend("turboquant sign element count overflow".into()))?;
    u32::try_from(sign_elems).map_err(|_| {
        CeraError::Backend(format!(
            "turboquant sign element count {sign_elems} exceeds u32"
        ))
    })?;
    let signs_bytes = checked_words_to_bytes(sign_elems, "signs")?;
    validate_tq_buffer_limit(
        signs_bytes,
        "tq_signs",
        max_buffer_size,
        max_storage_binding_size,
    )?;

    let layout = TqLayout::new(head_dim);
    let mut layer_specs = Vec::with_capacity(n_layers);
    for (i, bt) in config.block_types.iter().enumerate() {
        match bt {
            BlockType::Attention | BlockType::ParallelAttentionMamba2 => {
                let n_kv_heads = *config.kv_heads_per_layer.get(i).ok_or_else(|| {
                    CeraError::Backend(format!(
                        "turboquant layer {i} missing kv_heads_per_layer entry (len={})",
                        config.kv_heads_per_layer.len()
                    ))
                })?;
                if n_kv_heads == 0 {
                    return Err(CeraError::Backend(format!(
                        "turboquant layer {i} n_kv_heads must be > 0"
                    )));
                }
                if !config.n_heads.is_multiple_of(n_kv_heads) {
                    return Err(CeraError::Backend(format!(
                        "turboquant layer {i} n_heads ({}) must be a multiple of n_kv_heads ({n_kv_heads})",
                        config.n_heads
                    )));
                }
                let kv_stride = n_kv_heads.checked_mul(head_dim).ok_or_else(|| {
                    CeraError::Backend(format!(
                        "turboquant layer {i} kv_stride overflow ({n_kv_heads} * {head_dim})"
                    ))
                })?;
                u32::try_from(kv_stride).map_err(|_| {
                    CeraError::Backend(format!(
                        "turboquant layer {i} kv_stride {kv_stride} exceeds u32"
                    ))
                })?;
                // `TqLayout` owns the region formula and guards its own
                // multiplies, so production and the oracle tests can't drift
                // apart on the sizing.
                let vecs =
                    tq_checked_elems::<u32>(n_kv_heads, max_seq_len, &format!("layer {i} vecs"))?;
                let k_words = layout.key_words(vecs).map_err(|_| {
                    CeraError::Backend(format!("turboquant layer {i} key word count overflow"))
                })?;
                let v_words = layout.value_words(vecs).map_err(|_| {
                    CeraError::Backend(format!("turboquant layer {i} value word count overflow"))
                })?;
                let k_bytes = checked_words_to_bytes(k_words, &format!("layer {i} keys"))?;
                let v_bytes = checked_words_to_bytes(v_words, &format!("layer {i} values"))?;
                validate_tq_buffer_limit(
                    k_bytes,
                    &format!("l{i}.tq_keys"),
                    max_buffer_size,
                    max_storage_binding_size,
                )?;
                validate_tq_buffer_limit(
                    v_bytes,
                    &format!("l{i}.tq_values"),
                    max_buffer_size,
                    max_storage_binding_size,
                )?;
                layer_specs.push(Some((n_kv_heads, k_bytes, v_bytes)));
            }
            BlockType::GatedConv | BlockType::Mamba2 | BlockType::DeltaNet => {
                layer_specs.push(None);
            }
        }
    }

    let q_rows = tq_checked_elems::<f32>(q_cap, config.n_heads, "qrot rows")?;
    let qrot_stride = head_dim
        .checked_mul(2)
        .and_then(|d| d.checked_add(1))
        .ok_or_else(|| CeraError::Backend("turboquant qrot stride overflow".into()))?;
    let qrot_floats = tq_checked_elems::<f32>(q_rows, qrot_stride, "qrot")?;
    let qrot_bytes = checked_words_to_bytes(qrot_floats, "qrot")?;
    validate_tq_buffer_limit(
        qrot_bytes,
        "tq_qrot",
        max_buffer_size,
        max_storage_binding_size,
    )?;

    let align = usize::try_from(min_storage_align.max(4))
        .map_err(|_| CeraError::Backend("turboquant storage alignment exceeds usize".into()))?;
    let slot_stride = PARAMS_BYTES
        .checked_next_multiple_of(align)
        .ok_or_else(|| CeraError::Backend("turboquant slot_stride alignment overflow".into()))?;
    let total_slots = n_layers
        .checked_mul(SLOTS_PER_LAYER)
        .ok_or_else(|| CeraError::Backend("turboquant params slot count overflow".into()))?;
    let params_bytes_usize = total_slots
        .checked_mul(slot_stride)
        .ok_or_else(|| CeraError::Backend("turboquant params byte count overflow".into()))?;
    let params_bytes = u64::try_from(params_bytes_usize)
        .map_err(|_| CeraError::Backend("turboquant params byte count overflow u64".into()))?;
    validate_tq_buffer_limit(
        params_bytes,
        "tq_params",
        max_buffer_size,
        max_storage_binding_size,
    )?;

    Ok(TqCachePlan {
        layout,
        sign_elems,
        layer_specs,
        qrot_bytes,
        slot_stride,
        params_bytes,
    })
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
        // Pre-validate all geometry, overflows, and GPU buffer limits before compiling
        // pipelines or generating sign tables.
        let plan = prevalidate_tq_cache_specs(
            config,
            max_seq_len,
            q_cap,
            ctx.min_storage_buffer_offset_alignment,
            ctx.max_buffer_size,
            ctx.max_storage_buffer_binding_size,
        )?;
        let n_layers = plan.layer_specs.len();

        // Per-layer sign flips, laid out [polar | jl] so one binding serves both.
        let mut signs = Vec::with_capacity(plan.sign_elems);
        for layer_idx in 0..n_layers {
            let rot = RotationState::try_from_seed(mode.seed ^ layer_idx as u64, head_dim)?;
            signs.extend_from_slice(&rot.polar_signs);
            signs.extend_from_slice(&rot.jl_signs);
        }

        let TqCachePlan {
            layout,
            sign_elems: _,
            layer_specs,
            qrot_bytes,
            slot_stride,
            params_bytes,
        } = plan;

        let cache = init_tq_cache_with(
            || {
                ctx.check_pipelines("tq pipelines", || TqPipelines {
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
                })
            },
            |pipelines| {
                let mut layers = Vec::with_capacity(n_layers);
                for (i, spec) in layer_specs.into_iter().enumerate() {
                    layers.push(spec.map(|(n_kv_heads, k_bytes, v_bytes)| TqLayerCache {
                        keys: ctx.create_storage_rw(k_bytes, &format!("l{i}.tq_keys")),
                        values: ctx.create_storage_rw(v_bytes, &format!("l{i}.tq_values")),
                        n_kv_heads,
                    }));
                }

                // Rotated-query scratch: two head-wide regions plus one sum per (row, head).
                let qrot = ctx.create_storage_rw(qrot_bytes, "tq_qrot");

                // One params slot per (layer, purpose), each padded out to the device's
                // storage-binding offset alignment so a bind group can address it.
                let params = ctx.create_storage_rw(params_bytes, "tq_params");

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
            },
            |stage| ctx.ensure_alive(stage),
        )?;
        ctx.save_pipeline_cache();
        Ok(cache)
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
        let mut pass = ctx.begin_pass(enc, label);
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
        let mut pass = ctx.begin_pass(enc, "flash_attention_tq");
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
        // Mirrors `MetalTqCache::snapshot_layer` and `restore_layer`'s rejection.
        // Without it a caller passing a stale longer length would read past the live
        // region and, for the last head, past the allocation.
        assert!(
            seq_len <= self.max_seq_len,
            "snapshot_layer: seq_len {seq_len} exceeds the cache capacity {}",
            self.max_seq_len
        );
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
        let scratch = ctx.create_storage_rw(TqLayout::words_to_bytes(total), "tq_snapshot_gather");

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tq_snapshot_gather"),
            });
        let mut dst = 0usize;
        let mut gather = |src: &wgpu::Buffer, src_word: usize, words: usize, dst: &mut usize| {
            enc.copy_buffer_to_buffer(
                src,
                TqLayout::words_to_bytes(src_word),
                &scratch,
                TqLayout::words_to_bytes(*dst),
                TqLayout::words_to_bytes(words),
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
            ctx.queue.write_buffer(
                &l.keys,
                TqLayout::words_to_bytes(h * cap * pw),
                &keys.polar_data[h],
            );
            ctx.queue.write_buffer(
                &l.keys,
                TqLayout::words_to_bytes(jl_off + h * cap * jw),
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
                TqLayout::words_to_bytes(norm_off + h * cap),
                bytemuck::cast_slice(&norm_words),
            );

            ctx.queue.write_buffer(
                &l.values,
                TqLayout::words_to_bytes(h * cap * pw),
                &values.polar_data[h],
            );
            norm_words.clear();
            norm_words.extend(values.norms[h].iter().map(|&b| u32::from(b)));
            ctx.queue.write_buffer(
                &l.values,
                TqLayout::words_to_bytes(v_norm_off + h * cap),
                bytemuck::cast_slice(&norm_words),
            );
        }
        Some(seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::init_tq_cache_with;
    use crate::CeraError;
    use crate::model::{BlockType, ModelConfig};

    #[test]
    fn init_tq_cache_with_validates_pipelines_before_allocating_buffers() {
        let mut allocated = false;
        let mut checked_alive = false;
        let err = init_tq_cache_with(
            || -> anyhow::Result<u32> { anyhow::bail!("tq validation failed") },
            |_| -> Result<u32, CeraError> {
                allocated = true;
                Ok(1)
            },
            |_| {
                checked_alive = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            !allocated && !checked_alive,
            "buffer allocation and post-upload check must not run after pipeline failure"
        );
        match err {
            CeraError::Backend(msg) => assert!(msg.contains("tq validation failed"), "{msg}"),
            other => panic!("expected CeraError::Backend, got {other:?}"),
        }

        // Allocation error aborts without running post-upload alive check.
        let mut checked_alive = false;
        let err_alloc = init_tq_cache_with(
            || Ok(42u32),
            |_| -> Result<u32, CeraError> { Err(CeraError::Backend("oom".into())) },
            |_| {
                checked_alive = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(!checked_alive);
        match err_alloc {
            CeraError::Backend(msg) => assert_eq!(msg, "oom"),
            other => panic!("expected CeraError::Backend, got {other:?}"),
        }

        // Post-allocation device loss aborts with CeraError::Backend instead of returning Ok.
        let err_lost = init_tq_cache_with(
            || Ok(42u32),
            |p| Ok(p + 1),
            |stage| anyhow::bail!("{stage}: GPU device lost (signs upload lost)"),
        )
        .unwrap_err();
        match err_lost {
            CeraError::Backend(msg) => {
                assert!(msg.contains("tq allocation"), "{msg}");
                assert!(msg.contains("signs upload lost"), "{msg}");
            }
            other => panic!("expected CeraError::Backend, got {other:?}"),
        }

        let ok = init_tq_cache_with(|| Ok(42u32), |p| Ok(p + 1), |_| Ok(())).unwrap();
        assert_eq!(ok, 43);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn tq_gpu_cache_aborts_on_lost_device() {
        use super::{TqGpuCache, TqMode};
        let Some(ctx) = crate::backend::wgpu::gpu_or_skip() else {
            return;
        };
        ctx.simulate_device_lost("test: tq lost");
        let cfg = ModelConfig {
            architecture: "lfm2".to_string(),
            n_layers: 1,
            hidden_size: 64,
            intermediate_size: 128,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 32,
            vocab_size: 32,
            max_seq_len: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            block_types: vec![BlockType::Attention],
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer: vec![2],
            scalars: Default::default(),
            moe: None,
            is_causal: true,
            class_labels: vec![],
        };
        let err = match TqGpuCache::new(&ctx, &cfg, 16, 4, TqMode { seed: 1 }) {
            Ok(_) => panic!("TqGpuCache::new must fail on a lost device"),
            Err(e) => e,
        };
        match err {
            CeraError::Backend(msg) => assert!(msg.contains("test: tq lost"), "{msg}"),
            other => panic!("expected CeraError::Backend, got {other:?}"),
        }
    }

    #[test]
    fn prevalidate_tq_cache_specs_rejects_degenerate_and_oversized_configs() {
        use super::prevalidate_tq_cache_specs;
        fn base_cfg() -> ModelConfig {
            ModelConfig {
                architecture: "lfm2".to_string(),
                n_layers: 1,
                hidden_size: 64,
                intermediate_size: 128,
                n_heads: 2,
                n_kv_heads: 2,
                head_dim: 32,
                vocab_size: 32,
                max_seq_len: 16,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                block_types: vec![BlockType::Attention],
                conv_kernel_size: None,
                ssm: None,
                kv_heads_per_layer: vec![2],
                scalars: Default::default(),
                moe: None,
                is_causal: true,
                class_labels: vec![],
            }
        }

        const LIMIT: u64 = 1024 * 1024;
        // Valid baseline passes cleanly.
        assert!(prevalidate_tq_cache_specs(&base_cfg(), 16, 4, 256, LIMIT, LIMIT).is_ok());

        // n_layers == 0
        let mut cfg = base_cfg();
        cfg.block_types.clear();
        cfg.kv_heads_per_layer.clear();
        assert!(matches!(
            prevalidate_tq_cache_specs(&cfg, 16, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("n_layers > 0")
        ));

        // max_seq_len == 0
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), 0, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("max_seq_len > 0")
        ));

        // q_cap == 0
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), 16, 0, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("q_cap > 0")
        ));

        // n_heads == 0
        let mut cfg = base_cfg();
        cfg.n_heads = 0;
        assert!(matches!(
            prevalidate_tq_cache_specs(&cfg, 16, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("n_heads > 0")
        ));

        // Short kv_heads_per_layer
        let mut cfg = base_cfg();
        cfg.kv_heads_per_layer.clear();
        assert!(matches!(
            prevalidate_tq_cache_specs(&cfg, 16, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("kv_heads_per_layer")
        ));

        // Zero n_kv_heads on attention layer
        let mut cfg = base_cfg();
        cfg.kv_heads_per_layer = vec![0];
        assert!(matches!(
            prevalidate_tq_cache_specs(&cfg, 16, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("n_kv_heads must be > 0")
        ));

        // n_kv_heads > n_heads (shader divides n_heads / n_kv_heads -> 0) and non-divisible
        // GQA head counts must return CeraError::Backend upfront.
        let mut cfg = base_cfg();
        cfg.n_heads = 2;
        cfg.kv_heads_per_layer = vec![4];
        assert!(matches!(
            prevalidate_tq_cache_specs(&cfg, 16, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("multiple of n_kv_heads")
        ));
        let mut cfg = base_cfg();
        cfg.n_heads = 10;
        cfg.kv_heads_per_layer = vec![4];
        assert!(matches!(
            prevalidate_tq_cache_specs(&cfg, 16, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("multiple of n_kv_heads")
        ));

        // Integer overflow in signs / qrot / params / vecs (both 32-bit and 64-bit targets return
        // CeraError::Backend, never OutOfMemory or wrapped u64 byte counts).
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), usize::MAX, 4, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(_))
        ));
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), 16, usize::MAX, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(_))
        ));
        let mut portable_u32_overflow = base_cfg();
        portable_u32_overflow.head_dim = 32;
        portable_u32_overflow.n_heads = 67_108_872;
        portable_u32_overflow.kv_heads_per_layer = vec![1];
        assert!(matches!(
            prevalidate_tq_cache_specs(
                &portable_u32_overflow,
                1,
                u32::MAX as usize,
                256,
                u64::MAX,
                u64::MAX
            ),
            Err(CeraError::Backend(_))
        ));
        // Exact u64 byte-scaling overflow on 64-bit (`qrot_floats = 4.61e18 <= usize::MAX` while
        // `qrot_floats * 4 > u64::MAX`, wrapping mod 2^64 to ~256 MiB < LIMIT if unguarded).
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            prevalidate_tq_cache_specs(&portable_u32_overflow, 1, 1_057_222_593, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("byte count overflow")
        ));
        #[cfg(not(target_pointer_width = "64"))]
        assert!(matches!(
            prevalidate_tq_cache_specs(&portable_u32_overflow, 1, 1_057_222_593, 256, LIMIT, LIMIT),
            Err(CeraError::Backend(_))
        ));
        let mut huge_layers_cfg = base_cfg();
        huge_layers_cfg.block_types = vec![BlockType::GatedConv; 128];
        huge_layers_cfg.kv_heads_per_layer = vec![0; 128];
        // Tiny storage alignment or huge alignment overflowing params
        assert!(matches!(
            prevalidate_tq_cache_specs(
                &huge_layers_cfg,
                16,
                4,
                usize::MAX as u64 / 2,
                LIMIT,
                LIMIT
            ),
            Err(CeraError::Backend(_))
        ));

        // Each of the 5 buffers exceeding max_buffer_size or max_storage_buffer_binding_size
        // returns CeraError::Backend naming the offending buffer.
        // Baseline buffer sizes for head_dim=32, n_layers=1, kv_heads=2, max_seq_len=16, q_cap=4,
        // n_heads=2, align=256:
        // tq_signs = 256 B, tq_params = 1024 B, l0.tq_keys = 512 B, l0.tq_values = 384 B,
        // tq_qrot = 2080 B.
        let plan = prevalidate_tq_cache_specs(&base_cfg(), 16, 4, 256, LIMIT, LIMIT).unwrap();
        let (n_kv, k_bytes, v_bytes) = plan.layer_specs[0].unwrap();
        assert_eq!(n_kv, 2);

        // Fail specifically on tq_signs via tiny limit < 256 B
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), 16, 4, 256, 128, LIMIT),
            Err(CeraError::Backend(msg)) if msg.contains("tq_signs")
        ));
        // Fail on max_storage_buffer_binding_size for k_bytes/v_bytes using large max_seq_len
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), 1_000_000, 1, 256, u64::MAX, 1024 * 1024),
            Err(CeraError::Backend(msg)) if msg.contains("tq_keys") || msg.contains("tq_values")
        ));
        // Fail specifically on tq_qrot via q_cap making qrot > limit while k/v stay <= limit
        assert!(matches!(
            prevalidate_tq_cache_specs(&base_cfg(), 1, 1024, 256, 1024 * 1024, 16 * 1024),
            Err(CeraError::Backend(msg)) if msg.contains("tq_qrot")
        ));
        // Fail specifically on tq_params when params_bytes > limit and all other buffers fit
        let many_conv_cfg = ModelConfig {
            block_types: vec![BlockType::GatedConv; 8],
            kv_heads_per_layer: vec![0; 8],
            ..base_cfg()
        };
        assert!(matches!(
            prevalidate_tq_cache_specs(&many_conv_cfg, 1, 1, 256, 4096, 4096),
            Err(CeraError::Backend(msg)) if msg.contains("tq_params")
        ));
        // Sanity check our baseline v_bytes < k_bytes
        assert!(v_bytes < k_bytes);
    }
}

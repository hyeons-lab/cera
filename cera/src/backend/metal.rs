// Native Metal compute backend for macOS.
//
// Bypasses wgpu's WGSL→MSL translation and per-dispatch validation overhead.
// Uses the `metal` crate directly for access to MTL APIs.

use std::collections::HashMap;

use anyhow::{Context, Result};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, CounterSampleBuffer, CounterSampleBufferDescriptor,
    Device, Library, MTLResourceOptions, MTLStorageMode,
};

pub mod params;
pub use params::{
    ArgmaxParams, BiasAddParams, Conv1dBatchParams, Conv1dParams, ElementwiseParams,
    FlashAttnParams, GemmF32Params, GemvBatchParams, GemvQkvParams, GemvRmsParams,
    GemvSplitKParams, KvCopyParams, KvShiftKParams, MetalParams, MoeCombineParams, MoeGemvParams,
    MoeRouteParams, NormParams, PrefillAttnParams, QkNormRopeBatchParams, QkNormRopeParams,
    QuantGemmParams, RmsNormBatchParams, RopeParams, ScaleParams, SplitAttnParams, TqAttnParams,
    TqParams,
};

/// Metal compute context: device, command queue, compiled shader library cache.
///
/// `library_cache` uses `Mutex` rather than `RefCell` so `MetalContext`
/// (and transitively `MetalLfm2Model`, `Arc<dyn Model>`, `Session`) is
/// `Sync`, which UniFFI requires on every type it exposes. Contention
/// is negligible — MSL libraries are only looked up during pipeline
/// creation, not on the per-token hot path.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub device_name: String,
    /// Cache compiled MSL libraries by source pointer address.
    /// Since sources are `include_str!` statics, pointer identity = source identity.
    library_cache: std::sync::Mutex<HashMap<usize, Library>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().context("no Metal device found")?;
        let queue = device.new_command_queue();
        let device_name = device.name().to_string();
        tracing::info!(device = %device_name, "Metal context initialized");
        Ok(Self {
            device,
            queue,
            device_name,
            library_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Upload f32 data to a GPU buffer (shared storage, unified memory).
    pub fn upload_f32(&self, data: &[f32]) -> Buffer {
        let size = std::mem::size_of_val(data) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// The placeholder `freq_factors` buffer for models with no Llama-3 RoPE scaling.
    ///
    /// [`QkNormRopeParams::bind`] and [`KvShiftKParams::bind`] must always bind slot 5 /
    /// slot 3: the kernels declare the binding unconditionally, and leaving it unbound is
    /// what produced NaN. When `has_freq_factors == 0` the contents are never read, so one
    /// element suffices — but it is `1.0`, not `0.0`, so that flipping the flag on can
    /// only be wrong, never a divide-by-zero. Owning that invariant in one place beats
    /// restating it at every construction site.
    pub fn freq_factors_dummy(&self) -> Buffer {
        self.upload_f32(&[1.0f32])
    }

    /// Upload raw bytes to a GPU buffer.
    pub fn upload_bytes(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a zeroed buffer.
    pub fn create_buffer(&self, size: u64) -> Buffer {
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    /// Compile an MSL source string into a compute pipeline.
    /// Libraries are cached by source pointer — multiple entry points from the
    /// same `include_str!` source share one compilation.
    ///
    /// `src` is `&'static str` by contract: the cache is keyed on the
    /// string's pointer address, and non-'static strings can be dropped
    /// and have their allocation reused for a different source, which
    /// would cause the cache to return the wrong compiled library. All
    /// real callers pass `include_str!` statics; this bound makes the
    /// invariant enforceable at the type level.
    pub fn create_pipeline(&self, src: &'static str, entry: &str) -> Result<ComputePipelineState> {
        let key = src.as_ptr() as usize;
        // Fast path: look up under the lock, release before compiling.
        // Compiling MSL can take tens of ms — holding the mutex across
        // that would serialize concurrent pipeline creation and, if the
        // compile panics, poison the mutex for every other pipeline
        // creation that follows. Cloning the cached `Library` (cheap,
        // it's an NSObject handle) lets us drop the lock immediately.
        {
            let cache = self
                .library_cache
                .lock()
                .expect("library_cache mutex poisoned");
            if let Some(lib) = cache.get(&key) {
                let library = lib.clone();
                drop(cache);
                return build_pipeline(&self.device, &library, entry);
            }
        }
        // Slow path: compile without holding the lock. A second caller
        // racing in between the drop above and the insert below will
        // compile again — wasted work but correctness-preserving (both
        // `Library`s reference the same underlying MSL source; last
        // writer wins into the cache).
        let opts = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(src, &opts)
            .map_err(|e| anyhow::anyhow!("MSL compile failed: {e}"))?;
        self.library_cache
            .lock()
            .expect("library_cache mutex poisoned")
            .entry(key)
            .or_insert_with(|| library.clone());
        build_pipeline(&self.device, &library, entry)
    }
}

/// A linear-layer weight on the Metal backend: packed for the dtypes with a
/// simdgroup GEMM kernel, dequantized to f32 otherwise.
///
/// Shared by the vision and audio encoders. They previously carried identical
/// private copies of this enum and of the `Q8_0 | Q4_0 => packed, _ => dense`
/// rule, which meant adding a packed-GEMM dtype had two places to land and one
/// of them would silently keep dequantizing.
pub enum MetalLinearWeight {
    Dense(Buffer),
    Quant {
        buf: Buffer,
        dtype: crate::tensor::DType,
    },
}

/// The batched linear layer both GPU encoders run: `y[rows, out_dim] =
/// x[rows, in_dim] · wᵀ`, dispatching to a simdgroup GEMM for packed weights and
/// to the dense `vit_linear` kernel otherwise.
///
/// Holds the three pipelines because which one runs is a property of the
/// uploaded weight, not of the caller.
pub struct MetalLinear {
    p_dense: ComputePipelineState,
    p_q8_0: ComputePipelineState,
    p_q4_0: ComputePipelineState,
}

impl MetalLinear {
    pub fn new(ctx: &MetalContext) -> Result<Self> {
        Ok(Self {
            p_dense: ctx.create_pipeline(shaders::VIT_LINEAR, "vit_linear")?,
            p_q8_0: ctx.create_pipeline(shaders::GEMM_Q8_0, "gemm_q8_0")?,
            p_q4_0: ctx.create_pipeline(shaders::GEMM_Q4_0, "gemm_q4_0")?,
        })
    }

    /// `y[rows, out_dim] = x[rows, in_dim] · wᵀ`, into a fresh buffer.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        x: &Buffer,
        w: &MetalLinearWeight,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Buffer {
        let y = ctx.create_buffer((rows * out_dim * 4) as u64);
        match w {
            MetalLinearWeight::Quant { buf, dtype } => {
                let pipe = match dtype {
                    crate::tensor::DType::Q8_0 => &self.p_q8_0,
                    crate::tensor::DType::Q4_0 => &self.p_q4_0,
                    other => panic!("unsupported MetalLinear weight dtype: {other:?}"),
                };
                // These GEMM kernels never read `_pad`, so they always plain-store.
                let p = params::QuantGemmParams {
                    m: out_dim as u32,
                    k: in_dim as u32,
                    n: rows as u32,
                    x_stride: in_dim as u32,
                    y_stride: out_dim as u32,
                    _pad: 0,
                };
                ctx.run_kernel_shmem(
                    pipe,
                    &[buf, x, &y],
                    &p,
                    // (ceil(rows/32), ceil(out_dim/64)) tiles x 128 threads.
                    metal::MTLSize::new(
                        (rows as u64).div_ceil(32),
                        (out_dim as u64).div_ceil(64),
                        1,
                    ),
                    metal::MTLSize::new(128, 1, 1),
                    Some(8192), // 4 KB weights + 4 KB input
                );
            }
            MetalLinearWeight::Dense(wbuf) => {
                let p = params::VitLinearParams {
                    m: out_dim as u32,
                    k: in_dim as u32,
                    n: rows as u32,
                    _pad: 0,
                };
                ctx.run_kernel(
                    &self.p_dense,
                    &[wbuf, x, &y],
                    &p,
                    metal::MTLSize::new(out_dim as u64, rows as u64, 1),
                    metal::MTLSize::new(32, 1, 1),
                );
            }
        }
        y
    }
}

fn build_pipeline(device: &Device, library: &Library, entry: &str) -> Result<ComputePipelineState> {
    let function = library
        .get_function(entry, None)
        .map_err(|e| anyhow::anyhow!("entry point '{entry}' not found: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("pipeline creation failed: {e}"))
}

impl MetalContext {
    /// Read f32 data back from a shared buffer (unified memory = zero copy).
    pub fn read_f32(&self, buf: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }

    /// Run one kernel in its own command buffer and block until it completes.
    ///
    /// Binds `bufs` at slots `0..`, then `params` at the next slot with a
    /// `size_of_val`-derived length (never a literal, per [`params`]), dispatches
    /// `grid` threadgroups of `threads`, and waits.
    ///
    /// The ViT encoder had a private copy of this, the quantized GEMM and the
    /// ViT's flash-attention dispatch each had another, and the audio encoder
    /// would have added a fourth. It is command-buffer plumbing rather than
    /// anything an encoder owns, and a divergence between copies (a missed
    /// `wait_until_completed`, a params slot off by one) would show up as
    /// nondeterministic wrong output, so the library now has exactly one copy and
    /// it lives with the queue it submits to. Kernels needing threadgroup memory
    /// go through [`Self::run_kernel_shmem`], which shares this body.
    ///
    /// `cera/tests/` still hand-rolls the same sequence in a dozen places. Those
    /// predate this helper and are a follow-up, not a contradiction: it is `pub`
    /// precisely so they can adopt it.
    pub fn run_kernel<P: params::MetalParams>(
        &self,
        pipe: &ComputePipelineState,
        bufs: &[&Buffer],
        params: &P,
        grid: metal::MTLSize,
        threads: metal::MTLSize,
    ) {
        self.run_kernel_shmem(pipe, bufs, params, grid, threads, None);
    }

    /// [`Self::run_kernel`] for a kernel that also needs `shmem` bytes of
    /// threadgroup memory at index 0 (the simdgroup GEMMs and the ViT's
    /// flash-attention kernel).
    ///
    /// `None` and `Some(0)` are not the same thing: a kernel that declares no
    /// threadgroup memory should not have a length set at all, which is why this
    /// takes an `Option` rather than defaulting to zero.
    pub fn run_kernel_shmem<P: params::MetalParams>(
        &self,
        pipe: &ComputePipelineState,
        bufs: &[&Buffer],
        params: &P,
        grid: metal::MTLSize,
        threads: metal::MTLSize,
        shmem: Option<u64>,
    ) {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        params.set(enc, bufs.len() as u64);
        if let Some(bytes) = shmem {
            enc.set_threadgroup_memory_length(0, bytes);
        }
        enc.dispatch_thread_groups(grid, threads);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Upload a `[out_dim, in_dim]` row-major linear weight, keeping it packed
    /// when a simdgroup GEMM kernel can read the quantized bytes directly.
    ///
    /// On the context rather than on [`MetalLinear`] because uploading a weight
    /// needs the device, not the pipelines.
    pub fn upload_linear_weight(&self, w: &crate::model::weights::MmapWeight) -> MetalLinearWeight {
        use crate::tensor::DType;
        match w.dtype {
            DType::Q8_0 | DType::Q4_0 => MetalLinearWeight::Quant {
                buf: self.upload_bytes(w.data()),
                dtype: w.dtype,
            },
            _ => MetalLinearWeight::Dense(self.upload_f32(&w.to_dense_f32())),
        }
    }

    /// Read `count` u32 values back from a shared buffer (e.g. an argmax index
    /// output), the integer analog of [`Self::read_f32`].
    pub fn read_u32(&self, buf: &Buffer, count: usize) -> Vec<u32> {
        let ptr = buf.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }

    /// Create a MTLCounterSampleBuffer backed by the device's hardware timestamp
    /// counter. Used for GPU-timestamped per-dispatch profiling. Returns None if
    /// the device doesn't expose timestamp counters.
    pub fn new_timestamp_sample_buffer(&self, sample_count: usize) -> Option<CounterSampleBuffer> {
        // Find the timestamp counter set (name == "timestamp").
        let counter_sets = self.device.counter_sets();
        let ts_set = counter_sets
            .iter()
            .find(|cs| cs.name().eq_ignore_ascii_case("timestamp"))?;
        let desc = CounterSampleBufferDescriptor::new();
        desc.set_counter_set(ts_set);
        desc.set_storage_mode(MTLStorageMode::Shared);
        desc.set_sample_count(sample_count as u64);
        self.device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .ok()
    }

    /// Sample CPU + GPU timestamps simultaneously. Returns (cpu_mach_ticks, gpu_ticks).
    pub fn sample_timestamps(&self) -> (u64, u64) {
        let mut cpu = 0u64;
        let mut gpu = 0u64;
        self.device.sample_timestamps(&mut cpu, &mut gpu);
        (cpu, gpu)
    }
}

// ── Native MSL Shaders ────────────────────────────────────────────────

pub mod shaders {
    pub const GEMV_Q4_0: &str = include_str!("shaders/gemv_q4_0.metal");
    pub const GEMV_Q4_0_FAST: &str = include_str!("shaders/gemv_q4_0_fast.metal");
    pub const GEMV_F32: &str = include_str!("shaders/gemv_f32.metal");
    pub const GEMV_F16: &str = include_str!("shaders/gemv_f16.metal");
    pub const GEMV_Q6_K: &str = include_str!("shaders/gemv_q6_k.metal");
    pub const GEMV_Q4_K: &str = include_str!("shaders/gemv_q4_k.metal");
    pub const GEMV_Q5_K: &str = include_str!("shaders/gemv_q5_k.metal");
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.metal");

    /// The four elementwise entry points with identical WGSL+MSL twins
    /// (`add_inplace`, `scaled_add_inplace`, `mul_inplace`, `silu_mul_inplace`),
    /// generated from `shaders/slang/elementwise.slang` by build.rs and shared
    /// with the wgpu backend's `wgpu::shaders::ELEMENTWISE_SLANG`.
    ///
    /// **Not** the production constant: the port covers 4 of the 8 entry points
    /// [`ELEMENTWISE`] exposes, missing `memcpy_f32`, `scale_f32`, `mul_out`
    /// and `cast_f32_to_f16`, which are Metal-only and so have no WGSL twin to
    /// share a body with. Swapping it in would fail at pipeline creation on the
    /// first missing entry point. Kept so the bench can compare the four it
    /// does cover.
    pub const ELEMENTWISE_SLANG: &str =
        include_str!(concat!(env!("OUT_DIR"), "/elementwise.metal"));
    /// Generated from `shaders/slang/rmsnorm.slang` by build.rs and shared with
    /// the wgpu backend's `wgpu::shaders::RMSNORM`. A
    /// `__target_switch` port that diverges in both reduction and I/O model: the
    /// metal branch is out-of-place (src -> dst, 4 buffers) with a two-stage
    /// `simd_sum`; the wgsl branch is in-place (3 bindings) with a shared-memory
    /// tree. Each branch's binding set is dropped for the other target.
    /// `tests/slang_multitarget_parity.rs` pins it against the CPU reference.
    pub const RMSNORM: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsnorm.metal"));
    /// Mixture-of-experts routing (`lfm2moe`), generated from
    /// `shaders/slang/moe_route.slang` and shared with the wgpu backend.
    /// Sigmoid + top-k over the router logits; see the source for why the
    /// ranking and weighting score sets differ.
    pub const MOE_ROUTE: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_route.metal"));
    /// Expert-indexed Q4_0 GEMV, generated from
    /// `shaders/slang/moe_gemv_q4_0.slang`. Reads the routed expert id from a
    /// device buffer and does the weight-slice arithmetic in-shader, which is
    /// what keeps routing off the host and out of a per-layer readback stall.
    pub const MOE_GEMV_Q4_0: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_gemv_q4_0.metal"));
    /// Weighted sum of a token's expert outputs, generated from
    /// `shaders/slang/moe_combine.slang`.
    pub const MOE_COMBINE: &str = include_str!(concat!(env!("OUT_DIR"), "/moe_combine.metal"));
    /// Generated from `shaders/slang/per_head_rmsnorm.slang` by build.rs and
    /// shared with the wgpu backend's
    /// `wgpu::shaders::PER_HEAD_RMSNORM`. A `__target_switch`
    /// port: the metal branch keeps the two-stage `simd_sum`, the wgsl branch the
    /// shared-memory tree. `tests/slang_multitarget_parity.rs` pins it against
    /// the CPU reference.
    pub const PER_HEAD_RMSNORM: &str =
        include_str!(concat!(env!("OUT_DIR"), "/per_head_rmsnorm.metal"));
    /// Generated from `shaders/slang/softmax.slang` by build.rs rather than
    /// hand-written, sharing that source with the wgpu backend's
    /// `wgpu::shaders::SOFTMAX`. Contract is unchanged (buffer
    /// 0 = x in-place, buffer 1 = params) and the two-stage `simd_max`/`simd_sum`
    /// reduction is preserved via `__target_switch`, so this is not the
    /// portable-tree fallback. `tests/slang_multitarget_parity.rs` pins it
    /// against the CPU reference.
    pub const SOFTMAX: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.metal"));
    /// Capability probe, not a kernel: nothing dispatches this. Pins that Slang
    /// reaches Metal's `simdgroup_matrix` hardware through `linalg::CoopMat`,
    /// which is what decides whether the eight hand-tuned `simdgroup_matrix`
    /// GEMMs are portable at all. See `shaders/slang/coopmat_probe.slang`;
    /// asserted in `tests/slang_multitarget_parity.rs`.
    pub const COOPMAT_PROBE_SLANG: &str =
        include_str!(concat!(env!("OUT_DIR"), "/coopmat_probe.metal"));
    /// NEOX-only RoPE kernel, generated from `shaders/slang/rope.slang` by
    /// build.rs. Unlike gelu/bias_add/elementwise this is a `__target_switch`
    /// port: the metal branch is this minimal NEOX kernel, the wgsl branch
    /// (shared with `wgpu::shaders::ROPE`) carries the fuller one, with the
    /// interleaved and freq_factors paths. Slang omits the freq_factors binding
    /// from the MSL since only the wgsl branch uses it.
    /// `tests/slang_multitarget_parity.rs` pins it against the CPU reference.
    pub const ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.metal"));
    pub const QK_NORM_ROPE: &str = include_str!("shaders/qk_norm_rope.metal");
    /// Generated from `shaders/slang/conv1d.slang` by build.rs and shared with
    /// the wgpu backend's `wgpu::shaders::CONV1D`. Unlike the
    /// norm tier this is a clean single-body port: the two handwritten twins
    /// already agreed on element type, bindings and entry name, and there is no
    /// reduction or subgroup op, so there is no `__target_switch`.
    /// `tests/slang_multitarget_parity.rs` pins it against the CPU reference.
    pub const CONV1D: &str = include_str!(concat!(env!("OUT_DIR"), "/conv1d.metal"));
    pub const ATTENTION: &str = include_str!("shaders/attention.metal");
    pub const FLASH_ATTENTION: &str = include_str!("shaders/flash_attention.metal");
    pub const ATTENTION_GQA: &str = include_str!("shaders/attention_gqa.metal");
    pub const ATTENTION_SPLITK: &str = include_str!("shaders/attention_splitk.metal");
    /// Generated from `shaders/slang/argmax_f32.slang` by build.rs and shared
    /// with the wgpu backend's `wgpu::shaders::ARGMAX_F32`. A
    /// `__target_switch` port: the metal branch keeps the two-stage
    /// `simd_shuffle_down` value+index reduction, the wgsl branch the
    /// shared-memory tree. `tests/slang_multitarget_parity.rs` pins it against
    /// the CPU reference.
    pub const ARGMAX_F32: &str = include_str!(concat!(env!("OUT_DIR"), "/argmax_f32.metal"));
    pub const GEMV_Q4_0_BATCH: &str = include_str!("shaders/gemv_q4_0_batch.metal");
    /// Two kernels (`rmsnorm_batch` + `add_rmsnorm_batch`), generated from
    /// `shaders/slang/rmsnorm_batch.slang` by build.rs and shared with the wgpu
    /// backend's `wgpu::shaders::RMSNORM_BATCH`. A
    /// `__target_switch` port: the metal branch keeps the two-stage `simd_sum`,
    /// the wgsl branch the shared-memory tree.
    /// `tests/slang_multitarget_parity.rs` pins both entry points against the CPU
    /// reference.
    pub const RMSNORM_BATCH: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsnorm_batch.metal"));
    /// Generated from `shaders/slang/conv1d_fused.slang` by build.rs and shared
    /// with the wgpu backend's `wgpu::shaders::CONV1D_FUSED`. A
    /// clean single-body port with no `__target_switch`, made possible by first
    /// consolidating the Metal twin onto the WGSL twin's single packed `proj`
    /// binding (it used to take x, b and c as three separate buffers that every
    /// caller filled from one buffer at three offsets). No kernel-size guard on
    /// the port: nothing here indexes a fixed-size array, so a bound could only
    /// convert a correct result into a skipped write. ([`CONV1D_FUSED_BATCH`]
    /// does constrain its params, because it stages the weights and rolling
    /// state in fixed-size registers.) `tests/slang_multitarget_parity.rs` pins
    /// it against the CPU reference.
    pub const CONV1D_FUSED: &str = include_str!(concat!(env!("OUT_DIR"), "/conv1d_fused.metal"));
    pub const GEMM_Q4_0: &str = include_str!("shaders/gemm_q4_0.metal");
    pub const GEMM_Q4_1: &str = include_str!("shaders/gemm_q4_1.metal");
    pub const GEMV_Q4_1: &str = include_str!("shaders/gemv_q4_1.metal");
    pub const GEMM_Q4_K: &str = include_str!("shaders/gemm_q4_k.metal");
    pub const GEMM_Q5_K: &str = include_str!("shaders/gemm_q5_k.metal");
    pub const GEMM_Q8_0: &str = include_str!("shaders/gemm_q8_0.metal");
    /// Slang port of [`GEMM_Q8_0`], generated from `shaders/slang/gemm_q8_0.slang`.
    /// Same buffer bindings, same 64x32 tile, same 8 KB threadgroup budget, and
    /// the same mixed `simdgroup_matrix<float>` x `simdgroup_matrix<half>` MMA,
    /// reached through `linalg::CoopMat`. Two deliberate divergences (no
    /// simdgroup-scoped barrier, and a two-round ragged epilogue) are documented
    /// at the top of the .slang. Not on the production path: dispatched only by
    /// `tests/slang_multitarget_parity.rs` and `examples/slang_gemm_bench.rs`,
    /// the latter being the one that can tell whether the divergences cost
    /// anything.
    ///
    /// **Callers must bind 8 KB of threadgroup memory at index 0**, exactly like
    /// [`GEMM_Q8_0`]. Slang declares the staging arrays statically, but
    /// `build_support/msl_postpass.rs` rewrites them into slices of a
    /// `[[threadgroup(0)]]` parameter, because static groupshared is one of
    /// three things that stop the native AGX compiler folding load
    /// displacements. That rewrite is worth ~5% and declines with a
    /// `cargo:warning` if slangc moves its anchors, so binding the memory is
    /// correct either way.
    pub const GEMM_Q8_0_SLANG: &str = include_str!(concat!(env!("OUT_DIR"), "/gemm_q8_0.metal"));
    pub const GEMM_Q6_K: &str = include_str!("shaders/gemm_q6_k.metal");
    pub const GEMM_F32: &str = include_str!("shaders/gemm_f32.metal");
    pub const GEMV_Q8_0: &str = include_str!("shaders/gemv_q8_0.metal");
    pub const GEMV_Q8_0_BATCH: &str = include_str!("shaders/gemv_q8_0_batch.metal");
    pub const ATTENTION_PREFILL: &str = include_str!("shaders/attention_prefill.metal");
    pub const QK_NORM_ROPE_BATCH: &str = include_str!("shaders/qk_norm_rope_batch.metal");
    /// Generated from `shaders/slang/conv1d_fused_batch.slang` by build.rs and
    /// shared with the wgpu backend's
    /// `wgpu::shaders::CONV1D_FUSED_BATCH`. A clean single-body
    /// port with no `__target_switch`: the two handwritten twins shared an
    /// element type, a binding contract and an entry name, and neither had a
    /// reduction. They did differ in loop spelling, in the weight preload bound,
    /// and in whether they carried the `ks > 4 || d_conv > 3` early-out (the
    /// Metal twin did not, and was unguarded on `w[d_conv]` as a result); the
    /// `.slang` header explains how the shared body reconciles those.
    /// `tests/slang_multitarget_parity.rs` pins it against the CPU reference.
    pub const CONV1D_FUSED_BATCH: &str =
        include_str!(concat!(env!("OUT_DIR"), "/conv1d_fused_batch.metal"));
    pub const KV_SHIFT: &str = include_str!("shaders/kv_shift.metal");
    /// TurboQuant KV compression: `tq_encode_keys`, `tq_encode_values`,
    /// `tq_rotate_q` (three kernels in one source).
    pub const TURBOQUANT: &str = include_str!("shaders/turboquant.metal");
    /// FlashAttention over a TurboQuant-compressed cache — serves decode
    /// (`n_queries = 1`) and chunked prefill from one kernel.
    pub const FLASH_ATTENTION_TQ: &str = include_str!("shaders/flash_attention_tq.metal");
    // Vision-encoder (ViT) kernels.
    pub const VIT_LINEAR: &str = include_str!("shaders/vit_linear.metal");
    /// Generated from `shaders/slang/layernorm_batch.slang` by build.rs and
    /// shared with the wgpu backend's
    /// `wgpu::shaders::LAYERNORM_BATCH`. A `__target_switch`
    /// port: the metal branch keeps the two-stage `simd_sum`, the wgsl branch the
    /// shared-memory tree. `tests/slang_multitarget_parity.rs` pins it against
    /// the CPU reference.
    pub const LAYERNORM_BATCH: &str =
        include_str!(concat!(env!("OUT_DIR"), "/layernorm_batch.metal"));
    /// Generated from `shaders/slang/gelu.slang` by build.rs, sharing that
    /// source with the wgpu backend's `wgpu::shaders::GELU`.
    /// Contract is buffer 0 = x in-place, buffer 1 = params. No per-target
    /// divergence, so the whole body is shared with no `__target_switch`.
    /// `tests/slang_multitarget_parity.rs` pins it against the CPU reference.
    pub const GELU: &str = include_str!(concat!(env!("OUT_DIR"), "/gelu.metal"));
    /// Generated from `shaders/slang/bias_add.slang` by build.rs, sharing that
    /// source with the wgpu backend's `wgpu::shaders::BIAS_ADD`.
    /// Contract is buffer 0 = x in-place, buffer 1 = bias, buffer 2 = params. No
    /// per-target divergence, so the whole body is shared with no
    /// `__target_switch`. `tests/slang_multitarget_parity.rs` pins it against the
    /// CPU reference.
    pub const BIAS_ADD: &str = include_str!(concat!(env!("OUT_DIR"), "/bias_add.metal"));
    pub const VIT_ATTENTION: &str = include_str!("shaders/vit_attention.metal");
    pub const VIT_ATTENTION_MMA: &str = include_str!("shaders/vit_attention_mma.metal");
    /// Generated from `shaders/slang/exp_polar.slang` by build.rs and shared with
    /// the wgpu backend's `wgpu::shaders::EXP_POLAR`. First GPU ISTFT stage:
    /// maps the detokenizer's polar half-spectrum (log-magnitude, angle) to the
    /// interleaved real/imag half-spectrum the iDFT matmul consumes. A portable
    /// element-wise map, no `__target_switch`.
    pub const EXP_POLAR: &str = include_str!(concat!(env!("OUT_DIR"), "/exp_polar.metal"));
    /// Generated from `shaders/slang/overlap_add.slang` by build.rs and shared
    /// with the wgpu backend's `wgpu::shaders::OVERLAP_ADD`. Final GPU ISTFT
    /// stage: windowed overlap-add of the iDFT frames into PCM, one thread per
    /// output sample. A portable position-indexed reduction, no `__target_switch`.
    pub const OVERLAP_ADD: &str = include_str!(concat!(env!("OUT_DIR"), "/overlap_add.metal"));
    // LFM2A audio kernels: the Conformer encoder body and the log-mel
    // front-end below it. All generated from `shaders/slang/*.slang` by
    // build.rs and shared with the wgpu backend's
    // `wgpu::shaders::*` twins. These MSL halves are pinned numerically against
    // the CPU encoder by `tests/audio_encoder_metal_parity.rs`;
    // `tests/slang_multitarget_parity.rs` adds only an entry-point-presence check
    // for them (its no-subgroup and no-f16 guards are WGSL-only, being
    // constraints on the wgpu emission).
    /// `relu_inplace` / `silu_inplace` / `gelu_erf_inplace`, sharing `GELU`'s
    /// contract (buffer 0 = x in-place, buffer 1 = params). The GELU here is the
    /// **erf** form the audio adapter was trained against, not `GELU`'s tanh
    /// approximation.
    pub const ACTIVATIONS: &str = include_str!(concat!(env!("OUT_DIR"), "/activations.metal"));
    /// Direct (im2col-free) conv2d covering the conv subsampling stem's regular,
    /// depthwise and pointwise layers *and* each Conformer block's depthwise
    /// conv1d (`kh = 1`).
    pub const CONV2D_DIRECT: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d_direct.metal"));
    /// Outer-axis swap with a contiguous inner block: the conv stem's
    /// channel/time permute (`K = f_out`) and the conv module's time-major ↔
    /// channel-major transposes (`K = 1`).
    pub const TRANSPOSE_BLOCKED: &str =
        include_str!(concat!(env!("OUT_DIR"), "/transpose_blocked.metal"));
    /// Batched GLU split for the Conformer convolution module.
    pub const GLU_SPLIT: &str = include_str!(concat!(env!("OUT_DIR"), "/glu_split.metal"));
    /// Per-channel affine + SiLU over a channel-major buffer (the conv module's
    /// `conv_norm`, which is a scale/shift and not a LayerNorm).
    pub const CHAN_AFFINE_SILU: &str =
        include_str!(concat!(env!("OUT_DIR"), "/chan_affine_silu.metal"));
    /// Conformer self-attention with Transformer-XL relative-position bias.
    /// Portable (non-MMA), so unlike `VIT_ATTENTION` it is generated rather than
    /// handwritten; only its softmax reduction has a `__target_switch`.
    pub const AUDIO_XL_ATTENTION: &str =
        include_str!(concat!(env!("OUT_DIR"), "/audio_xl_attention.metal"));
    /// Framing pass of the log-mel front-end: center padding, pre-emphasis and
    /// the Hann window folded into one gather.
    pub const STFT_FRAME: &str = include_str!(concat!(env!("OUT_DIR"), "/stft_frame.metal"));
    /// Per-frame power spectrum as a direct DFT against a host-built twiddle
    /// table, standing in for the CPU path's rustfft.
    pub const POWER_SPEC: &str = include_str!(concat!(env!("OUT_DIR"), "/power_spec.metal"));
    /// Mel filterbank projection plus the natural-log floor, emitting mel-major.
    pub const MEL_PROJECT: &str = include_str!(concat!(env!("OUT_DIR"), "/mel_project.metal"));
    /// Per-feature (per-mel-bin) normalization, transposing back to time-major.
    pub const MEL_NORM: &str = include_str!(concat!(env!("OUT_DIR"), "/mel_norm.metal"));
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `CERA_REQUIRE_METAL=1` makes a missing device a failure rather than a
    /// pass, so the CI leg that targets known-capable hardware proves this ran.
    /// Same skip-vs-fail convention as `CERA_REQUIRE_SIMD` / `CERA_REQUIRE_GPU`;
    /// the integration suites get it from `tests/common::metal_context`, which a
    /// library test cannot name.
    fn require_metal(err: &anyhow::Error) {
        assert!(
            std::env::var("CERA_REQUIRE_METAL").as_deref() != Ok("1"),
            "CERA_REQUIRE_METAL=1 but no Metal device is available ({err})"
        );
    }

    #[test]
    fn test_metal_context_init() {
        let ctx = MetalContext::new();
        match ctx {
            Ok(ctx) => {
                println!("Metal device: {}", ctx.device_name);
                assert!(!ctx.device_name.is_empty());
            }
            Err(e) => {
                require_metal(&e);
                println!("No Metal device available: {e}");
            }
        }
    }

    #[test]
    fn test_metal_buffer_roundtrip() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                require_metal(&e);
                return;
            }
        };
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data);
        let result = ctx.read_f32(&buf, data.len());
        assert_eq!(data, result);
    }
}

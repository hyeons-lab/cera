// Native Metal compute backend for macOS.
//
// Bypasses wgpu's WGSL→MSL translation and per-dispatch validation overhead.
// Uses the `metal` crate directly for access to MTL APIs.

use std::collections::HashMap;

use anyhow::{Context, Result};
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, CounterSampleBuffer,
    CounterSampleBufferDescriptor, Device, Library, MTLResourceOptions, MTLStorageMode,
};

/// CPU mirror of the `Params` struct in `shaders/qk_norm_rope.metal` (binding 4).
///
/// This is a named `#[repr(C)]` type rather than an ad-hoc `[u32; N]` on purpose.
/// The array form is what allowed a real NaN bug: the kernel's struct grew from 7 to
/// 9 fields (adding `has_freq_factors` / `has_qk_norm`) and gained a `freq_factors`
/// buffer at binding 5, but the audio decoder's three call sites kept uploading the old
/// 7 fields with a *hardcoded* byte length. Nothing failed to compile — the kernel simply
/// read the two new flags past the end of the upload and, on garbage `has_freq_factors`,
/// divided by an unbound buffer, producing NaN/Inf and silent audio.
///
/// Use [`Self::bind`] to encode it: it sets binding 4 with a `size_of_val`-derived
/// length and binding 5 in the same call, so neither "wrong length" nor "forgot the
/// freq_factors buffer" is expressible at a call site.
///
/// **Keep the field order and count identical to the MSL struct.** Rust cannot see the
/// shader, so the `size_of` assert below is the only mechanical link: it turns a change
/// to *this* struct into a build break, forcing whoever edits it to look at the `.metal`
/// file.
///
/// It does **not** catch a field added on the *shader* side — the exact direction that
/// caused the NaN. `tests/metal_shaders_parity.rs` dispatches these kernels through this
/// same type and would catch it, but only where it actually runs: it is gated on
/// `--features metal`, which no CI job currently passes. Until that job exists, the
/// shader-side direction is guarded by review, not by the build.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkNormRopeParams {
    pub pos: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps_bits: u32,
    pub freq_base_bits: u32,
    /// 0 = NeoX (pairs at `[i, i+half]`); 1 = interleaved/NORM (pairs at `[2i, 2i+1]`).
    pub rope_type: u32,
    /// 1 ⇒ divide each pair's angle by `freq_factors[d]` (Llama-3 long-context scaling).
    /// When 0, the buffer passed to [`Self::bind`] is never read — a 1-element dummy is fine.
    pub has_freq_factors: u32,
    /// 1 ⇒ per-head RMS-norm of Q/K before RoPE (LFM2 / Qwen3 / the audio decoder);
    /// 0 ⇒ RoPE only (LLaMA / Qwen2 / Mistral / Granite).
    pub has_qk_norm: u32,
}

const _: () = assert!(size_of::<QkNormRopeParams>() == 36); // 9 × uint, qk_norm_rope.metal

impl QkNormRopeParams {
    /// Bind the params (buffer 4) and the `freq_factors` array (buffer 5).
    ///
    /// `freq_factors` must always be a live buffer even when `has_freq_factors == 0`:
    /// the kernel declares the binding unconditionally, and leaving slot 5 unbound is
    /// what produced NaN. Pass a 1-element `[1.0]` dummy in that case — `1.0`, not `0.0`,
    /// so that flipping the flag on can't divide by zero.
    pub fn bind(&self, enc: &ComputeCommandEncoderRef, freq_factors: &Buffer) {
        enc.set_bytes(
            4,
            std::mem::size_of_val(self) as u64,
            self as *const Self as *const _,
        );
        enc.set_buffer(5, Some(freq_factors), 0);
    }
}

/// CPU mirror of the `BatchParams` struct in `shaders/qk_norm_rope_batch.metal`
/// (binding 4).
///
/// The batched prefill sibling of [`QkNormRopeParams`]: same kernel body, but over `n`
/// tokens with per-token Q/K strides. Kept as its own type because the layouts genuinely
/// differ — see the field order below. Named for the same reason as its sibling: an
/// untyped `[u32; N]` lets the shader's struct drift away from the upload silently.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkNormRopeBatchParams {
    pub start_pos: u32,
    pub n_tokens: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps_bits: u32,
    pub freq_base_bits: u32,
    pub rope_type: u32,
    pub q_stride: u32,
    pub k_stride: u32,
    pub has_freq_factors: u32,
    pub has_qk_norm: u32,
}

const _: () = assert!(size_of::<QkNormRopeBatchParams>() == 48); // 12 × uint

impl QkNormRopeBatchParams {
    /// Bind the params (buffer 4) and the `freq_factors` array (buffer 5).
    /// See [`QkNormRopeParams::bind`] — slot 5 must always be live.
    pub fn bind(&self, enc: &ComputeCommandEncoderRef, freq_factors: &Buffer) {
        enc.set_bytes(
            4,
            std::mem::size_of_val(self) as u64,
            self as *const Self as *const _,
        );
        enc.set_buffer(5, Some(freq_factors), 0);
    }
}

/// CPU mirror of the `KParams` struct in `shaders/kv_shift.metal` (binding 2).
///
/// Same reasoning as [`QkNormRopeParams`], and the same bug: `kv_shift.metal` also grew
/// `rope_type` / `has_freq_factors` and a `freq_factors` buffer at binding 3. The
/// production dispatch was updated; `tests/metal_kv_shift_oracle.rs` kept its own private
/// copy of the old 8-field layout and never bound slot 3 — so the oracle, the test whose
/// entire job is to police this kernel, was itself dispatching it wrong and comparing the
/// resulting NaN against the CPU reference.
///
/// Worth being precise about what an under-sized upload does, because it is the reason
/// this class is dangerous: it is undefined behaviour, not a reliable crash. Here it
/// happened to produce NaN (an unbound buffer 3 divided into the angle) and the test went
/// red. Read a different garbage value and the kernel returns *plausible* numbers instead,
/// and the test goes green over a real miscompute.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvShiftKParams {
    pub n_keep: u32,
    pub shift: u32,
    pub new_seq_len: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub freq_base_bits: u32,
    /// `-(shift as i32)`: the rotation delta applied to each retained cell.
    pub delta_pos: i32,
    /// 0 = NeoX, 1 = NORM/interleaved. Must match the layout the forward pass used —
    /// shifting a NORM model with the NeoX layout pairs the wrong elements.
    pub rope_type: u32,
    /// 1 ⇒ divide each pair's angle by `freq_factors[d]`. See [`QkNormRopeParams`].
    pub has_freq_factors: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<KvShiftKParams>() == 40); // 10 × 4B (incl. _pad), kv_shift.metal

impl KvShiftKParams {
    /// Bind the params (buffer 2) and the `freq_factors` array (buffer 3).
    ///
    /// Slot 3 must always be live even when `has_freq_factors == 0` — see
    /// [`QkNormRopeParams::bind`] for why, and pass a `[1.0]` dummy.
    pub fn bind(&self, enc: &ComputeCommandEncoderRef, freq_factors: &Buffer) {
        enc.set_bytes(
            2,
            std::mem::size_of_val(self) as u64,
            self as *const Self as *const _,
        );
        enc.set_buffer(3, Some(freq_factors), 0);
    }
}

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
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.metal");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.metal");
    pub const PER_HEAD_RMSNORM: &str = include_str!("shaders/per_head_rmsnorm.metal");
    pub const SOFTMAX: &str = include_str!("shaders/softmax.metal");
    pub const ROPE: &str = include_str!("shaders/rope.metal");
    pub const QK_NORM_ROPE: &str = include_str!("shaders/qk_norm_rope.metal");
    pub const CONV1D: &str = include_str!("shaders/conv1d.metal");
    pub const ATTENTION: &str = include_str!("shaders/attention.metal");
    pub const FLASH_ATTENTION: &str = include_str!("shaders/flash_attention.metal");
    pub const ATTENTION_GQA: &str = include_str!("shaders/attention_gqa.metal");
    pub const ATTENTION_SPLITK: &str = include_str!("shaders/attention_splitk.metal");
    pub const ARGMAX_F32: &str = include_str!("shaders/argmax_f32.metal");
    pub const GEMV_Q4_0_BATCH: &str = include_str!("shaders/gemv_q4_0_batch.metal");
    pub const RMSNORM_BATCH: &str = include_str!("shaders/rmsnorm_batch.metal");
    pub const CONV1D_FUSED: &str = include_str!("shaders/conv1d_fused.metal");
    pub const GEMM_Q4_0: &str = include_str!("shaders/gemm_q4_0.metal");
    pub const GEMM_Q4_K: &str = include_str!("shaders/gemm_q4_k.metal");
    pub const GEMM_Q8_0: &str = include_str!("shaders/gemm_q8_0.metal");
    pub const GEMM_Q6_K: &str = include_str!("shaders/gemm_q6_k.metal");
    pub const GEMM_F32: &str = include_str!("shaders/gemm_f32.metal");
    pub const GEMV_Q8_0: &str = include_str!("shaders/gemv_q8_0.metal");
    pub const GEMV_Q8_0_BATCH: &str = include_str!("shaders/gemv_q8_0_batch.metal");
    pub const ATTENTION_PREFILL: &str = include_str!("shaders/attention_prefill.metal");
    pub const QK_NORM_ROPE_BATCH: &str = include_str!("shaders/qk_norm_rope_batch.metal");
    pub const CONV1D_FUSED_BATCH: &str = include_str!("shaders/conv1d_fused_batch.metal");
    pub const KV_SHIFT: &str = include_str!("shaders/kv_shift.metal");
    // Vision-encoder (ViT) kernels.
    pub const VIT_LINEAR: &str = include_str!("shaders/vit_linear.metal");
    pub const LAYERNORM_BATCH: &str = include_str!("shaders/layernorm_batch.metal");
    pub const GELU: &str = include_str!("shaders/gelu.metal");
    pub const BIAS_ADD: &str = include_str!("shaders/bias_add.metal");
    pub const VIT_ATTENTION: &str = include_str!("shaders/vit_attention.metal");
    pub const VIT_ATTENTION_MMA: &str = include_str!("shaders/vit_attention_mma.metal");
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_context_init() {
        let ctx = MetalContext::new();
        match ctx {
            Ok(ctx) => {
                println!("Metal device: {}", ctx.device_name);
                assert!(!ctx.device_name.is_empty());
            }
            Err(e) => {
                println!("No Metal device available: {e}");
            }
        }
    }

    #[test]
    fn test_metal_buffer_roundtrip() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data);
        let result = ctx.read_f32(&buf, data.len());
        assert_eq!(data, result);
    }
}

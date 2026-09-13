// Native CUDA compute backend for Cera.
//
// Bypasses generic graphics abstractions for direct access to NVIDIA Ampere (sm_87)
// and Turing+ hardware via the CUDA driver API.
// Features:
// - Blocking driver synchronization (CU_CTX_SCHED_BLOCKING_SYNC) to protect host audio DSP.
// - Zero-copy mapped pinned memory for Jetson Orin unified memory (UMA).
// - Sub-microsecond 1-token-per-launch CUDA Graphs with instant TTS audio streaming.
// - Dynamic driver loading without compile-time toolkit dependencies.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

pub mod buffer;
pub mod device;
pub mod kernels;
pub mod module;
pub mod stream;

pub use buffer::{CudaBuffer, CudaPinnedBuffer, CudaUnifiedBuffer};
pub use cudarc::driver::{LaunchConfig, PushKernelArg, sys};
pub use device::CudaDevice;
pub use kernels::*;
pub use module::{CudaKernel, CudaModule};
pub use stream::{CudaEvent, CudaGraph, CudaStream};

/// CUDA compute context managing device, default execution stream, and JIT module cache.
pub struct CudaContext {
    pub device: Arc<CudaDevice>,
    pub stream: CudaStream,
    /// Cache compiled PTX modules by static source pointer address.
    module_cache: Mutex<HashMap<usize, CudaModule>>,
}

impl CudaContext {
    /// Check if the CUDA driver dynamic library is present and loadable on this system.
    pub fn is_available() -> bool {
        CudaDevice::is_available()
    }

    /// Initialize CUDA context on the specified device ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = Arc::new(CudaDevice::new(ordinal)?);
        let stream = CudaStream::new(&device.ctx)?;
        Ok(Self {
            device,
            stream,
            module_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Initialize CUDA context on the default device (ordinal 0).
    pub fn default_device() -> Result<Self> {
        Self::new(0)
    }

    /// Allocate a device-resident buffer of the specified size in bytes.
    pub fn create_buffer(&self, size_bytes: usize) -> Result<CudaBuffer> {
        self.device
            .ctx
            .bind_to_thread()
            .context("failed to bind CUDA context to thread")?;
        let slice = self
            .stream
            .inner()
            .alloc_zeros::<u8>(size_bytes)
            .context("failed to allocate device buffer")?;
        Ok(CudaBuffer::new(slice))
    }

    /// Upload host byte slice into a newly allocated device-resident buffer.
    pub fn upload_bytes(&self, data: &[u8]) -> Result<CudaBuffer> {
        self.device
            .ctx
            .bind_to_thread()
            .context("failed to bind CUDA context to thread")?;
        let slice = self
            .stream
            .inner()
            .clone_htod(data)
            .context("failed to upload bytes to device buffer")?;
        Ok(CudaBuffer::new(slice))
    }

    /// Upload host f32 slice into a newly allocated device-resident buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Result<CudaBuffer> {
        let bytes = bytemuck::cast_slice(data);
        self.upload_bytes(bytes)
    }

    /// Download bytes from a device-resident buffer into host memory.
    pub fn download_bytes(&self, buf: &CudaBuffer, dst: &mut [u8]) -> Result<()> {
        buf.copy_to_host(dst)
    }

    /// Download f32 elements from a device-resident buffer into host memory.
    pub fn download_f32(&self, buf: &CudaBuffer, dst: &mut [f32]) -> Result<()> {
        let bytes = bytemuck::cast_slice_mut(dst);
        self.download_bytes(buf, bytes)
    }

    /// Allocate page-locked host memory mapped into GPU address space for zero-copy UMA access.
    pub fn create_pinned_buffer(&self, size_bytes: usize) -> Result<CudaPinnedBuffer> {
        CudaPinnedBuffer::new(self.device.ctx.clone(), size_bytes)
    }

    /// Upload host bytes into a page-locked zero-copy mapped buffer.
    pub fn upload_pinned_bytes(&self, data: &[u8]) -> Result<CudaPinnedBuffer> {
        let mut pinned = self.create_pinned_buffer(data.len())?;
        pinned.as_mut_slice().copy_from_slice(data);
        Ok(pinned)
    }

    /// Allocate unified / managed memory buffer accessible from both CPU and GPU.
    pub fn create_unified_buffer(&self, size_bytes: usize) -> Result<CudaUnifiedBuffer> {
        self.device
            .ctx
            .bind_to_thread()
            .context("failed to bind CUDA context to thread")?;
        let slice = unsafe {
            self.device
                .ctx
                .alloc_unified::<u8>(size_bytes, true)
                .context("failed to allocate unified memory buffer")?
        };
        Ok(CudaUnifiedBuffer::new(slice))
    }

    /// Compile or retrieve cached PTX module by static source pointer.
    pub fn load_ptx(&self, ptx_src: &'static str) -> Result<CudaModule> {
        let key = ptx_src.as_ptr() as usize;
        let mut cache = self
            .module_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("CUDA module cache lock poisoned: {e}"))?;

        if let Some(module) = cache.get(&key) {
            return Ok(module.clone());
        }

        let module = CudaModule::from_ptx(&self.device.ctx, ptx_src)?;
        cache.insert(key, module.clone());
        Ok(module)
    }

    /// Determine target architecture string for NVRTC compilation based on device capability.
    pub fn nvrtc_arch(&self) -> &'static str {
        match self.device.compute_capability {
            (8, 7) => "compute_87", // Jetson Orin (Tegra Ampere)
            (8, 6) => "compute_86", // Ampere consumer (RTX 30xx)
            (8, 0) => "compute_80", // Ampere datacenter (A100)
            (8, 9) => "compute_89", // Ada Lovelace (RTX 40xx)
            (9, 0) => "compute_90", // Hopper (H100)
            (7, 5) => "compute_75", // Turing (RTX 20xx / T4)
            (7, 0) => "compute_70", // Volta (V100)
            _ => "compute_75",      // Baseline fallback
        }
    }

    /// Compile or retrieve cached CUDA C++ source module.
    pub fn load_cuda(&self, cu_src: &'static str, name: &str) -> Result<CudaModule> {
        let key = cu_src.as_ptr() as usize;
        let mut cache = self
            .module_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("CUDA module cache lock poisoned: {e}"))?;

        if let Some(module) = cache.get(&key) {
            return Ok(module.clone());
        }

        let arch = self.nvrtc_arch();
        let module = CudaModule::from_cuda_src(&self.device.ctx, cu_src, Some(name), Some(arch))?;
        cache.insert(key, module.clone());
        Ok(module)
    }

    /// Load or retrieve cached kernel function by static CUDA source and entry point name.
    pub fn load_kernel(
        &self,
        cu_src: &'static str,
        module_name: &str,
        func_name: &str,
    ) -> Result<CudaKernel> {
        let module = self.load_cuda(cu_src, module_name)?;
        module.get_kernel(func_name)
    }

    /// Execute single-token Q4_0 matrix-vector multiplication (y = A * x).
    pub fn gemv_q4_0(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMV_Q4_0_SRC, "gemv_q4_0", "gemv_q4_0")?;
        let warps_per_block = 8u32; // 256 threads / 32 = 8 warps
        let num_blocks = m.div_ceil(warps_per_block * 4);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (warps_per_block * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = GemvParams { m, k };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemv_q4_0 kernel")?;
        Ok(())
    }

    /// Execute single-token Q4_0 matrix-vector multiplication with residual accumulation (y += A * x).
    pub fn gemv_q4_0_accum(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMV_Q4_0_SRC, "gemv_q4_0", "gemv_q4_0_accum")?;
        let warps_per_block = 8u32;
        let num_blocks = m.div_ceil(warps_per_block * 4);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (warps_per_block * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = GemvParams { m, k };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemv_q4_0_accum kernel")?;
        Ok(())
    }

    /// Execute single-token Q8_0 matrix-vector multiplication (y = A * x).
    pub fn gemv_q8_0(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMV_Q8_0_SRC, "gemv_q8_0", "gemv_q8_0")?;
        let warps_per_block = 8u32;
        let num_blocks = m.div_ceil(warps_per_block * 4);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (warps_per_block * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = GemvParams { m, k };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemv_q8_0 kernel")?;
        Ok(())
    }

    /// Execute single-token Q8_0 matrix-vector multiplication with residual accumulation (y += A * x).
    pub fn gemv_q8_0_accum(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMV_Q8_0_SRC, "gemv_q8_0", "gemv_q8_0_accum")?;
        let warps_per_block = 8u32;
        let num_blocks = m.div_ceil(warps_per_block * 4);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (warps_per_block * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = GemvParams { m, k };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemv_q8_0_accum kernel")?;
        Ok(())
    }

    /// Execute batched Q4_0 matrix-matrix multiplication (Y = X * A^T).
    pub fn gemm_q4_0(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMM_Q4_0_SRC, "gemm_q4_0", "gemm_q4_0")?;
        let tile_m = 16u32;
        let tile_n = 16u32;
        let grid_x = n.div_ceil(tile_n);
        let grid_y = m.div_ceil(tile_m);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile_n, tile_m, 1),
            shared_mem_bytes: 0,
        };
        let params = GemmParams { m, n, k, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemm_q4_0 kernel")?;
        Ok(())
    }

    /// Execute batched Q4_0 matrix-matrix multiplication with residual accumulation (Y += X * A^T).
    pub fn gemm_q4_0_accum(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMM_Q4_0_SRC, "gemm_q4_0", "gemm_q4_0_accum")?;
        let tile_m = 16u32;
        let tile_n = 16u32;
        let grid_x = n.div_ceil(tile_n);
        let grid_y = m.div_ceil(tile_m);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile_n, tile_m, 1),
            shared_mem_bytes: 0,
        };
        let params = GemmParams { m, n, k, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemm_q4_0_accum kernel")?;
        Ok(())
    }

    /// Execute batched Q8_0 matrix-matrix multiplication (Y[M, N] = X[M, K] * A[N, K]^T).
    pub fn gemm_q8_0(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMM_Q8_0_SRC, "gemm_q8_0", "gemm_q8_0")?;
        let tile_m = 16u32;
        let tile_n = 16u32;
        let grid_x = n.div_ceil(tile_n);
        let grid_y = m.div_ceil(tile_m);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile_n, tile_m, 1),
            shared_mem_bytes: 0,
        };
        let params = GemmParams { m, n, k, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemm_q8_0 kernel")?;
        Ok(())
    }

    /// Execute batched Q8_0 matrix-matrix multiplication with residual accumulation (Y += X * A^T).
    pub fn gemm_q8_0_accum(
        &self,
        out: &mut CudaBuffer,
        weights: &CudaBuffer,
        x: &CudaBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(GEMM_Q8_0_SRC, "gemm_q8_0", "gemm_q8_0_accum")?;
        let tile_m = 16u32;
        let tile_n = 16u32;
        let grid_x = n.div_ceil(tile_n);
        let grid_y = m.div_ceil(tile_m);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile_n, tile_m, 1),
            shared_mem_bytes: 0,
        };
        let params = GemmParams { m, n, k, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(weights);
        builder.arg(x);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gemm_q8_0_accum kernel")?;
        Ok(())
    }

    /// Dequantize token embedding row directly on GPU into activation buffer.
    pub fn gather_embedding_q8_0(
        &self,
        out: &mut CudaBuffer,
        table: &CudaBuffer,
        token_id: u32,
        hidden_size: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(
            GATHER_EMBEDDING_SRC,
            "gather_embedding",
            "gather_embedding_q8_0",
        )?;
        let threads = 128u32.min(hidden_size / 32).max(32);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = GatherParams {
            token_id,
            hidden_size,
        };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(out);
        builder.arg(table);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gather_embedding_q8_0 kernel")?;
        Ok(())
    }

    /// Dequantize Q4_0 token embedding row directly on GPU into activation buffer.
    pub fn gather_embedding_q4_0(
        &self,
        out: &mut CudaBuffer,
        table: &CudaBuffer,
        token_id: u32,
        hidden_size: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(
            GATHER_EMBEDDING_SRC,
            "gather_embedding",
            "gather_embedding_q4_0",
        )?;
        let threads = 128u32.min(hidden_size / 32).max(32);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = GatherParams {
            token_id,
            hidden_size,
        };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(out);
        builder.arg(table);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch gather_embedding_q4_0 kernel")?;
        Ok(())
    }

    /// Execute Root Mean Square Normalization (RMSNorm).
    pub fn rmsnorm(
        &self,
        out: &mut CudaBuffer,
        x: &CudaBuffer,
        weight: &CudaBuffer,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        let kernel = self.load_kernel(RMSNORM_SRC, "rmsnorm", "rmsnorm")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = RmsNormParams { n, eps };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(x);
        builder.arg(weight);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch rmsnorm kernel")?;
        Ok(())
    }

    /// Execute fused in-place residual addition (x += residual) and RMSNorm (out = norm(x, weight)).
    pub fn fused_add_rmsnorm(
        &self,
        out: &mut CudaBuffer,
        x: &mut CudaBuffer,
        residual: &CudaBuffer,
        weight: &CudaBuffer,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        let kernel = self.load_kernel(RMSNORM_SRC, "rmsnorm", "fused_add_rmsnorm")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = RmsNormParams { n, eps };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(x);
        builder.arg(residual);
        builder.arg(weight);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch fused_add_rmsnorm kernel")?;
        Ok(())
    }

    /// Execute fused per-head RMSNorm + RoPE for Query and Key tensors.
    pub fn qk_norm_rope(
        &self,
        q: &mut CudaBuffer,
        k_cache: &mut CudaBuffer,
        q_norm_w: Option<&CudaBuffer>,
        k_norm_w: Option<&CudaBuffer>,
        freq_factors: Option<&CudaBuffer>,
        params: QkNormRopeParams,
    ) -> Result<()> {
        let kernel = self.load_kernel(QK_NORM_ROPE_SRC, "qk_norm_rope", "qk_norm_rope")?;
        let num_blocks = params.n_heads.max(params.n_kv_heads);
        // Shared memory for per-head reduction: head_dim floats + 8 floats for warp sums
        let shared_mem_bytes = (params.head_dim + 8) * std::mem::size_of::<f32>() as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };

        let null_ptr: sys::CUdeviceptr = 0;
        let q_norm_ptr = q_norm_w.map(|b| b.cu_device_ptr()).unwrap_or(null_ptr);
        let k_norm_ptr = k_norm_w.map(|b| b.cu_device_ptr()).unwrap_or(null_ptr);
        let freq_ptr = freq_factors.map(|b| b.cu_device_ptr()).unwrap_or(null_ptr);

        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(q);
        builder.arg(k_cache);
        builder.arg(&q_norm_ptr);
        builder.arg(&k_norm_ptr);
        builder.arg(&freq_ptr);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch qk_norm_rope kernel")?;
        Ok(())
    }

    /// Execute SwiGLU elementwise operation in-place: `a[i] = silu(a[i]) * b[i]`.
    pub fn silu_mul_inplace(&self, a: &mut CudaBuffer, b: &CudaBuffer, n: u32) -> Result<()> {
        let kernel = self.load_kernel(ELEMENTWISE_SRC, "elementwise", "silu_mul_inplace")?;
        let cfg = LaunchConfig::for_num_elems(n);
        let params = ElementwiseParams { n, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(a);
        builder.arg(b);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch silu_mul_inplace kernel")?;
        Ok(())
    }

    /// Execute elementwise vector addition in-place: `a[i] += b[i]`.
    pub fn add_inplace(&self, a: &mut CudaBuffer, b: &CudaBuffer, n: u32) -> Result<()> {
        let kernel = self.load_kernel(ELEMENTWISE_SRC, "elementwise", "add_inplace")?;
        let cfg = LaunchConfig::for_num_elems(n);
        let params = ElementwiseParams { n, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(a);
        builder.arg(b);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch add_inplace kernel")?;
        Ok(())
    }

    /// Execute scaled vector addition in-place: `a[i] += scale * b[i]`.
    pub fn scaled_add_inplace(
        &self,
        a: &mut CudaBuffer,
        b: &CudaBuffer,
        scale: f32,
        n: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(ELEMENTWISE_SRC, "elementwise", "scaled_add_inplace")?;
        let cfg = LaunchConfig::for_num_elems(n);
        let params = ScaleParams { n, scale };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(a);
        builder.arg(b);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch scaled_add_inplace kernel")?;
        Ok(())
    }

    /// Execute scalar scaling in-place: `a[i] *= scale`.
    pub fn scale_inplace(&self, a: &mut CudaBuffer, scale: f32, n: u32) -> Result<()> {
        let kernel = self.load_kernel(ELEMENTWISE_SRC, "elementwise", "scale_f32")?;
        let cfg = LaunchConfig::for_num_elems(n);
        let params = ScaleParams { n, scale };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(a);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch scale_f32 kernel")?;
        Ok(())
    }

    /// Cast f32 buffer into f16 representation at a specific element offset in dst.
    pub fn cast_f32_to_f16_offset(
        &self,
        src: &CudaBuffer,
        dst: &CudaBuffer,
        dst_element_offset: usize,
        n: u32,
    ) -> Result<()> {
        let kernel = self.load_kernel(ELEMENTWISE_SRC, "elementwise", "cast_f32_to_f16")?;
        let cfg = LaunchConfig::for_num_elems(n);
        let params = ElementwiseParams { n, _pad: 0 };
        // SAFETY: dst_ptr evaluates to a 64-bit unsigned device virtual address (CUdeviceptr = u64).
        // The cast_f32_to_f16 kernel accepts uint16_t* __restrict__ dst, which matches the standard
        // 64-bit CUDA driver execution stack pointer layout.
        let dst_ptr =
            dst.cu_device_ptr() + (dst_element_offset * std::mem::size_of::<u16>()) as u64;
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(src);
        builder.arg(&dst_ptr);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch cast_f32_to_f16 kernel")?;
        Ok(())
    }

    /// Execute numerically stable softmax in-place on a single row of logits.
    pub fn softmax(&self, x: &mut CudaBuffer, n: u32) -> Result<()> {
        let kernel = self.load_kernel(SOFTMAX_SRC, "softmax", "softmax")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = SoftmaxParams { n, _pad: 0 };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(x);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch softmax kernel")?;
        Ok(())
    }

    /// Execute single-token FlashAttention decode across all query heads.
    pub fn flash_attention(
        &self,
        out: &mut CudaBuffer,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        params: AttentionParams,
    ) -> Result<()> {
        let kernel = self.load_kernel(ATTENTION_SRC, "attention", "flash_attention")?;
        let cfg = LaunchConfig {
            grid_dim: (params.n_heads, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(q);
        builder.arg(k_cache);
        builder.arg(v_cache);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch flash_attention kernel")?;
        Ok(())
    }

    /// Execute fused 1D short convolution for LFM2 GatedConv blocks.
    pub fn conv1d_fused(
        &self,
        out: &mut CudaBuffer,
        proj: &CudaBuffer,
        rbuffer: &mut CudaBuffer,
        weight: &CudaBuffer,
        params: Conv1dParams,
    ) -> Result<()> {
        let kernel = self.load_kernel(CONV1D_FUSED_SRC, "conv1d_fused", "conv1d_fused")?;
        let cfg = LaunchConfig::for_num_elems(params.hs);
        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(proj);
        builder.arg(rbuffer);
        builder.arg(weight);
        builder.arg(out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.context("failed to launch conv1d_fused kernel")?;
        Ok(())
    }

    /// Synchronize the context's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Create an auxiliary non-blocking CUDA stream in this context.
    pub fn new_stream(&self) -> Result<CudaStream> {
        CudaStream::new(&self.device.ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_context_error_handling() {
        // On hosts without NVIDIA drivers, initialization must return an Err cleanly without panicking.
        let result = CudaContext::default_device();
        match result {
            Ok(ctx) => {
                assert_eq!(ctx.device.ordinal, 0);
            }
            Err(err) => {
                assert!(!err.to_string().is_empty());
            }
        }
    }
}

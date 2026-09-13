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
pub mod module;
pub mod stream;

pub use buffer::{CudaBuffer, CudaPinnedBuffer, CudaUnifiedBuffer};
pub use cudarc::driver::LaunchConfig;
pub use device::CudaDevice;
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

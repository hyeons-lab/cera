// Native CUDA device and primary context initialization.
//
// Manages device discovery, compute capability querying, and primary context setup.
// Applies CU_CTX_SCHED_BLOCKING_SYNC to eliminate driver spin-wait polling,
// protecting host ARM CPU cores for realtime audio DSP in automotive setups.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, result, sys};

/// CUDA device handle encapsulating primary context and hardware attributes.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub ctx: Arc<CudaContext>,
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory_bytes: usize,
    pub is_integrated: bool,
    pub supports_managed_memory: bool,
    pub supports_unified_addressing: bool,
}

impl CudaDevice {
    /// Check if the CUDA driver dynamic library is present and loadable on this system.
    pub fn is_available() -> bool {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(|| result::init().is_ok()).unwrap_or(false);
        std::panic::set_hook(prev_hook);
        res
    }

    /// Initialize a CUDA device by ordinal with blocking driver synchronization.
    pub fn new(ordinal: usize) -> Result<Self> {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let init_res = std::panic::catch_unwind(result::init);
        std::panic::set_hook(prev_hook);

        match init_res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e).context("failed to initialize CUDA driver"),
            Err(_) => anyhow::bail!("CUDA driver shared library not found on this system"),
        }

        let cu_device = result::device::get(ordinal as i32)
            .with_context(|| format!("failed to get CUDA device ordinal {ordinal}"))?;

        // Configure CU_CTX_SCHED_BLOCKING_SYNC on the primary context before retaining it.
        // This causes cudaStreamSynchronize to sleep the CPU thread on an OS wait condition
        // rather than burning 100% CPU in a spin-wait loop, which is critical for preventing
        // scheduling jitter on the 20 ms Tier 1 audio pipeline (AEC, beamforming, VAD).
        unsafe {
            let res = sys::cuDevicePrimaryCtxSetFlags_v2(
                cu_device,
                sys::CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC as u32,
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS
                && res != sys::cudaError_enum::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
            {
                return Err(cudarc::driver::DriverError(res)).context(
                    "failed to configure CU_CTX_SCHED_BLOCKING_SYNC on CUDA primary context",
                );
            }
        }

        let ctx = CudaContext::new(ordinal).with_context(|| {
            format!("failed to retain CUDA primary context for device {ordinal}")
        })?;

        let major = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .context("failed to query compute capability major")?
        };

        let minor = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .context("failed to query compute capability minor")?
        };

        let is_integrated = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED,
            )
            .unwrap_or(0)
                > 0
        };

        let supports_managed_memory = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
            )
            .unwrap_or(0)
                > 0
        };

        let supports_unified_addressing = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            )
            .unwrap_or(0)
                > 0
        };

        let total_memory_bytes = unsafe {
            result::device::total_mem(cu_device).context("failed to query total device memory")?
        };

        let name = result::device::get_name(cu_device).context("failed to query device name")?;

        tracing::info!(
            ordinal,
            %name,
            sm = %format!("{major}.{minor}"),
            vram_mb = total_memory_bytes / (1024 * 1024),
            is_integrated,
            supports_managed_memory,
            supports_unified_addressing,
            "CUDA device initialized with blocking synchronization"
        );

        Ok(Self {
            ctx,
            ordinal,
            name,
            compute_capability: (major, minor),
            total_memory_bytes,
            is_integrated,
            supports_managed_memory,
            supports_unified_addressing,
        })
    }

    /// Check if the device matches Ampere sm_87 (Jetson AGX Orin / Orin NX).
    pub fn is_sm87(&self) -> bool {
        self.compute_capability == (8, 7)
    }

    /// Check if the compute capability is at least sm_75 (Turing).
    pub fn is_sm75_or_newer(&self) -> bool {
        self.compute_capability.0 > 7
            || (self.compute_capability.0 == 7 && self.compute_capability.1 >= 5)
    }
}

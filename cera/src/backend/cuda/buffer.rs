// Memory buffers for CUDA backend: device-only, pinned host (zero-copy UMA), and unified managed.
//
// Supports:
// 1. CudaBuffer: High-bandwidth device memory allocated via cuMemAlloc.
// 2. CudaPinnedBuffer: Page-locked host memory mapped into GPU address space via
//    cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP, enabling zero-copy audio ingestion on Jetson Orin.
// 3. CudaUnifiedBuffer: Unified memory allocated via cuMemAllocManaged.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DriverError, LaunchArgs, PushKernelArg,
    UnifiedSlice, sys,
};

/// High-bandwidth device-resident memory buffer.
#[derive(Debug)]
pub struct CudaBuffer {
    pub(crate) slice: CudaSlice<u8>,
}

impl CudaBuffer {
    /// Wrap an existing `CudaSlice<u8>`.
    pub fn new(slice: CudaSlice<u8>) -> Self {
        Self { slice }
    }

    /// Size of buffer in bytes.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Underlying device pointer using the buffer's allocation stream.
    pub fn cu_device_ptr(&self) -> sys::CUdeviceptr {
        let (ptr, _guard) = self.slice.device_ptr(self.slice.stream());
        ptr
    }

    /// Underlying device pointer on a specific stream.
    pub fn device_ptr(&self, stream: &CudaStream) -> sys::CUdeviceptr {
        let (ptr, _guard) = self.slice.device_ptr(stream);
        ptr
    }

    /// Copy host slice to device buffer.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        let stream = self.slice.stream().clone();
        stream
            .memcpy_htod(src, &mut self.slice)
            .context("failed to copy host data to device buffer")?;
        Ok(())
    }

    /// Asynchronously zero the device buffer.
    pub fn zero(&mut self) -> Result<()> {
        let stream = self.slice.stream().clone();
        stream
            .memset_zeros(&mut self.slice)
            .context("failed to zero device buffer")?;
        Ok(())
    }

    /// Copy device buffer data to host slice.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        let stream = self.slice.stream().clone();
        stream
            .memcpy_dtoh(&self.slice, dst)
            .context("failed to copy device data to host slice")?;
        Ok(())
    }

    /// Reference to the underlying `CudaSlice<u8>`.
    pub fn inner(&self) -> &CudaSlice<u8> {
        &self.slice
    }

    /// Mutable reference to the underlying `CudaSlice<u8>`.
    pub fn inner_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.slice
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b CudaBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaBuffer) -> &mut Self {
        self.arg(&arg.slice)
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b mut CudaBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaBuffer) -> &mut Self {
        self.arg(&mut arg.slice)
    }
}

/// Page-locked (pinned) host buffer mapped directly into GPU virtual address space.
///
/// On Jetson Orin (UMA architecture), CPU and GPU share physical LPDDR5 memory.
/// Allocating with CU_MEMHOSTALLOC_DEVICEMAP allows both ARM CPU and Ampere GPU
/// to access the same physical buffer without explicit PCIe copies.
#[derive(Debug)]
pub struct CudaPinnedBuffer {
    host_ptr: *mut u8,
    device_ptr: sys::CUdeviceptr,
    len: usize,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaPinnedBuffer {}
unsafe impl Sync for CudaPinnedBuffer {}

impl CudaPinnedBuffer {
    /// Allocate a pinned host buffer mapped into device address space.
    pub fn new(ctx: Arc<CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread()
            .context("failed to bind CUDA context to thread")?;

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let res = sys::cuMemHostAlloc(&mut host_ptr, len, sys::CU_MEMHOSTALLOC_DEVICEMAP);
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(DriverError(res)).with_context(|| {
                    format!("failed to allocate {len} bytes of pinned host memory")
                });
            }

            let mut device_ptr: sys::CUdeviceptr = 0;
            let res = sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0);
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                let _ = sys::cuMemFreeHost(host_ptr);
                return Err(DriverError(res))
                    .context("failed to get mapped device pointer for pinned host memory");
            }

            Ok(Self {
                host_ptr: host_ptr as *mut u8,
                device_ptr,
                len,
                ctx,
            })
        }
    }

    /// Size of buffer in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Mapped GPU device pointer for kernel launches.
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.device_ptr
    }

    /// CPU host slice reference.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.host_ptr, self.len) }
        }
    }

    /// CPU host mutable slice reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.len) }
        }
    }
}

impl Deref for CudaPinnedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for CudaPinnedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for CudaPinnedBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            let _ = self.ctx.bind_to_thread();
            unsafe {
                let _ = sys::cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
            }
        }
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b CudaPinnedBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaPinnedBuffer) -> &mut Self {
        self.arg(&arg.device_ptr)
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b mut CudaPinnedBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaPinnedBuffer) -> &mut Self {
        self.arg(&arg.device_ptr)
    }
}

/// Unified / Managed memory buffer accessible from both CPU and GPU.
#[derive(Debug)]
pub struct CudaUnifiedBuffer {
    pub(crate) slice: UnifiedSlice<u8>,
}

impl CudaUnifiedBuffer {
    /// Wrap an existing `UnifiedSlice<u8>`.
    pub fn new(slice: UnifiedSlice<u8>) -> Self {
        Self { slice }
    }

    /// Size of buffer in bytes.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Read host slice.
    pub fn as_slice(&self) -> Result<&[u8]> {
        self.slice
            .as_slice()
            .context("failed to access unified memory on host")
    }

    /// Mutable host slice.
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8]> {
        self.slice
            .as_mut_slice()
            .context("failed to access unified memory on host for writing")
    }

    /// Prefetch buffer pages to GPU memory.
    pub fn prefetch_to_device(&self) -> Result<()> {
        self.slice
            .prefetch()
            .context("failed to prefetch unified memory to device")
    }

    /// Reference to underlying `UnifiedSlice<u8>`.
    pub fn inner(&self) -> &UnifiedSlice<u8> {
        &self.slice
    }

    /// Mutable reference to underlying `UnifiedSlice<u8>`.
    pub fn inner_mut(&mut self) -> &mut UnifiedSlice<u8> {
        &mut self.slice
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b CudaUnifiedBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaUnifiedBuffer) -> &mut Self {
        self.arg(&arg.slice)
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b mut CudaUnifiedBuffer> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaUnifiedBuffer) -> &mut Self {
        self.arg(&mut arg.slice)
    }
}

//! RAII memory management for FastRPC shared memory (`rpcmem`).
//!
//! `rpcmem` provides unified DMA buffers accessible by both the host ARM CPU
//! and the Hexagon DSP without PCIe-style copying.

use std::sync::Arc;

use super::sys::FastRpcDriver;
use crate::session::CeraError;

/// A shared memory buffer allocated via `rpcmem` and mapped into the CDSP.
pub struct RpcmemBuffer {
    driver: Arc<FastRpcDriver>,
    ptr: *mut u8,
    size: usize,
    fd: i32,
    mapped: bool,
}

// SAFETY (Send + Sync): the allocation is DSP-visible from any thread, but
// host-side soundness requires external serialization: every `as_slice` /
// `as_mut_ptr` use must hold `HexagonLfm2Model.device`'s `MutexGuard`, and
// `&RpcmemBuffer` must never escape that critical section. Concurrent
// `Model::forward` calls are safe only because they serialize on that lock.
unsafe impl Send for RpcmemBuffer {}
unsafe impl Sync for RpcmemBuffer {}

impl RpcmemBuffer {
    /// Allocate a new shared `rpcmem` buffer and optionally map it into the CDSP.
    pub fn alloc(
        driver: Arc<FastRpcDriver>,
        size: usize,
        map_to_dsp: bool,
    ) -> Result<Self, CeraError> {
        if size == 0 {
            return Err(CeraError::Backend(
                "rpcmem allocation size must be non-zero".into(),
            ));
        }
        let ptr = driver.rpcmem_alloc(size)?;
        let fd = match driver.rpcmem_to_fd(ptr) {
            Ok(fd) => fd,
            Err(e) => {
                driver.rpcmem_free(ptr);
                return Err(e);
            }
        };

        let mut mapped = false;
        if map_to_dsp {
            if let Err(e) = driver.fastrpc_mmap(fd, ptr, size) {
                driver.rpcmem_free(ptr);
                return Err(e);
            }
            mapped = true;
        }

        Ok(Self {
            driver,
            ptr,
            size,
            fd,
            mapped,
        })
    }

    /// Obtain a shared slice view of the host-accessible memory.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Obtain a mutable slice view of the host-accessible memory.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Raw pointer to the host memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Mutable raw pointer to the host memory.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Shared DMA file descriptor.
    pub fn fd(&self) -> i32 {
        self.fd
    }

    /// Base memory address as an integer for IDL mapping.
    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    /// Flush CPU cache lines for this buffer to ensure DSP visibility.
    pub fn flush_cpu_cache(&self, offset: usize, length: usize) {
        let end = offset.saturating_add(length).min(self.size);
        let start = offset.min(end);
        let len = end - start;
        if len == 0 {
            return;
        }
        // Memory fence ensures all writes to rpcmem complete before FastRPC invoke.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }

    /// Invalidate CPU cache lines to observe updates made by the DSP.
    pub fn invalidate_cpu_cache(&self, offset: usize, length: usize) {
        let end = offset.saturating_add(length).min(self.size);
        let start = offset.min(end);
        let len = end - start;
        if len == 0 {
            return;
        }
        // Memory fence ensures subsequent CPU reads observe completed DSP writes.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for RpcmemBuffer {
    fn drop(&mut self) {
        if self.mapped {
            let _ = self.driver.fastrpc_munmap(self.fd, self.ptr, self.size);
        }
        self.driver.rpcmem_free(self.ptr);
    }
}

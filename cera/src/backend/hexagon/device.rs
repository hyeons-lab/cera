//! Hexagon device management and skeleton session lifecycle.
//!
//! Handles opening FastRPC skeleton handles into an Unsigned Process Domain (domain 3),
//! querying hardware capabilities (`hwinfo`), and establishing command queue sessions.

use std::sync::Arc;

use super::queue::HexagonQueueSession;
use super::sys::{FastRpcDriver, RemoteArg, RemoteBuf, RemoteHandle64, remote_scalars_make};
use super::types::*;
use crate::session::CeraError;

/// Supported Hexagon architecture versions for compiled skel libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HexagonArch {
    V73 = 73, // Snapdragon 8 Gen 2 / 8s Gen 3 / X Elite (SM8550)
    V75 = 75, // Snapdragon 8 Gen 3 (SM8650)
    V79 = 79, // Snapdragon 8 Elite (SM8750)
    V81 = 81, // Snapdragon 8 Elite Gen 5 (SM8850)
}

impl HexagonArch {
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            73 => Some(Self::V73),
            75 => Some(Self::V75),
            79 => Some(Self::V79),
            81 => Some(Self::V81),
            _ => None,
        }
    }

    pub fn skel_filename(&self) -> String {
        format!("libggml-htp-v{}.so", *self as u32)
    }
}

/// Active Hexagon device session owning a FastRPC handle and command queue.
pub struct HexagonDevice {
    driver: Arc<FastRpcDriver>,
    handle: RemoteHandle64,
    arch: HexagonArch,
    hw_info: HtpHwInfo,
    queue_session: HexagonQueueSession,
}

// Device session is Send when guarded under model locks (such as Mutex<HexagonDevice>).
unsafe impl Send for HexagonDevice {}

#[repr(C)]
struct HtpHwinfoPayload {
    n_threads: u32,
    n_hvx: u32,
    n_hmx: u32,
    _pad: u32,
    vtcm_size: u64,
}

#[repr(C)]
struct HtpStartPayload {
    sess_id: u32,
    _pad: u32,
    dsp_queue_id: u64,
    n_hvx: u32,
    n_hmx: u32,
    max_vmem: u64,
}

#[repr(C)]
struct HtpMmapPayload {
    fd: u32,
    _pad: u32,
    size: u64,
}

#[repr(C)]
struct HtpMunmapPayload {
    fd: u32,
}

impl HexagonDevice {
    pub fn new(driver: Arc<FastRpcDriver>, arch: HexagonArch) -> Result<Self, CeraError> {
        // Enable Unsigned Process Domain (domain 3) for standard Android APK deployment
        driver.enable_unsigned_pd(3)?;

        // FastRPC URI pointing to the architecture skel library in CDSP Unsigned PD
        let skel_uri = format!(
            "file:///{}?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp",
            arch.skel_filename()
        );
        let handle = driver.open_skel_handle(&skel_uri)?;

        // Query hardware capabilities via htp_iface_hwinfo (Method 8: 0 in, 1 out)
        let mut hw_payload = HtpHwinfoPayload {
            n_threads: 0,
            n_hvx: 0,
            n_hmx: 0,
            _pad: 0,
            vtcm_size: 0,
        };
        let mut out_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &mut hw_payload as *mut _ as *mut std::ffi::c_void,
                len: std::mem::size_of::<HtpHwinfoPayload>(),
            },
        }];

        let hwinfo_scalars = remote_scalars_make(8, 0, 1);
        let hw_info = match driver.invoke_skel(handle, hwinfo_scalars, &mut out_args) {
            Ok(()) => HtpHwInfo {
                n_threads: hw_payload.n_threads,
                n_hvx: hw_payload.n_hvx,
                n_hmx: hw_payload.n_hmx,
                vtcm_size: hw_payload.vtcm_size,
            },
            Err(e) => {
                tracing::debug!("Failed to query HTP hwinfo ({e}), using default capabilities");
                HtpHwInfo {
                    n_threads: 8,
                    n_hvx: 8,
                    n_hmx: 1,
                    vtcm_size: 8 * 1024 * 1024,
                }
            }
        };

        // Create command queue session with 1 MiB staging buffer to accommodate full model layer batches
        let queue_session = match HexagonQueueSession::new(Arc::clone(&driver), 1024 * 1024) {
            Ok(qs) => qs,
            Err(e) => {
                driver.close_skel_handle(handle);
                return Err(e);
            }
        };

        // Start session on DSP via htp_iface_start (Method 2: 1 in, 0 out)
        let max_vmem: u64 = 3 * 1024 * 1024 * 1024; // 3 GB memory cap for single domain
        let start_payload = HtpStartPayload {
            sess_id: 0,
            _pad: 0,
            dsp_queue_id: queue_session.queue_id(),
            n_hvx: hw_info.n_hvx,
            n_hmx: hw_info.n_hmx,
            max_vmem,
        };

        let mut in_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &start_payload as *const _ as *mut std::ffi::c_void,
                len: std::mem::size_of::<HtpStartPayload>(),
            },
        }];

        let start_scalars = remote_scalars_make(2, 1, 0);
        if let Err(e) = driver.invoke_skel(handle, start_scalars, &mut in_args) {
            driver.close_skel_handle(handle);
            return Err(e);
        }

        Ok(Self {
            driver,
            handle,
            arch,
            hw_info,
            queue_session,
        })
    }

    /// Query probed hardware information for this Hexagon device.
    pub fn hw_info(&self) -> &HtpHwInfo {
        &self.hw_info
    }

    /// Hexagon architecture version.
    pub fn arch(&self) -> HexagonArch {
        self.arch
    }

    /// Register a mapped buffer with the DSP skeleton (Method 4: 1 in, 0 out).
    pub fn mmap_buffer(&self, fd: u32, size: u64) -> Result<(), CeraError> {
        let payload = HtpMmapPayload { fd, _pad: 0, size };
        let mut in_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &payload as *const _ as *mut std::ffi::c_void,
                len: std::mem::size_of::<HtpMmapPayload>(),
            },
        }];
        let mmap_scalars = remote_scalars_make(4, 1, 0);
        self.driver
            .invoke_skel(self.handle, mmap_scalars, &mut in_args)
    }

    /// Unregister a buffer from the DSP skeleton (Method 5: 1 in, 0 out).
    pub fn munmap_buffer(&self, fd: u32) -> Result<(), CeraError> {
        let payload = HtpMunmapPayload { fd };
        let mut in_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &payload as *const _ as *mut std::ffi::c_void,
                len: std::mem::size_of::<HtpMunmapPayload>(),
            },
        }];
        let munmap_scalars = remote_scalars_make(5, 1, 0);
        self.driver
            .invoke_skel(self.handle, munmap_scalars, &mut in_args)
    }

    /// Mutable reference to the command queue session.
    pub fn queue_session_mut(&mut self) -> &mut HexagonQueueSession {
        &mut self.queue_session
    }
}

impl Drop for HexagonDevice {
    fn drop(&mut self) {
        // Stop session on DSP (Method 3: 0 in, 0 out)
        let stop_scalars = remote_scalars_make(3, 0, 0);
        let _ = self.driver.invoke_skel(self.handle, stop_scalars, &mut []);
        self.driver.close_skel_handle(self.handle);
    }
}

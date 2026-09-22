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

impl HexagonDevice {
    /// Initialize a Hexagon device session for the specified architecture in Unsigned PD.
    pub fn new(driver: Arc<FastRpcDriver>, arch: HexagonArch) -> Result<Self, CeraError> {
        // FastRPC URI pointing to the architecture skel library in Unsigned PD (domain 3)
        let skel_uri = format!("file:///{}?domain=3", arch.skel_filename());
        let handle = driver.open_skel_handle(&skel_uri)?;

        // Query hardware capabilities via htp_iface_hwinfo (Method 6: 0 in, 4 out)
        let mut n_threads: u32 = 0;
        let mut n_hvx: u32 = 0;
        let mut n_hmx: u32 = 0;
        let mut vtcm_size: u64 = 0;

        let mut out_args = [
            RemoteArg {
                buf: RemoteBuf {
                    buf: &mut n_threads as *mut u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &mut n_hvx as *mut u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &mut n_hmx as *mut u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &mut vtcm_size as *mut u64 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u64>(),
                },
            },
        ];

        let hwinfo_scalars = remote_scalars_make(6, 0, 4);
        let hw_res = driver.invoke_skel(handle, hwinfo_scalars, &mut out_args);
        if let Err(e) = hw_res {
            driver.close_skel_handle(handle);
            return Err(e);
        }

        let hw_info = HtpHwInfo {
            n_threads,
            n_hvx,
            n_hmx,
            vtcm_size,
        };

        // Create command queue session with 1 MiB staging buffer to accommodate full model layer batches
        let queue_session = match HexagonQueueSession::new(Arc::clone(&driver), 1024 * 1024) {
            Ok(qs) => qs,
            Err(e) => {
                driver.close_skel_handle(handle);
                return Err(e);
            }
        };

        // Start session on DSP via htp_iface_start (Method 0: 5 in, 0 out)
        let sess_id: u32 = 0;
        let dsp_queue_id = queue_session.queue_id();
        let max_vmem: u64 = 3 * 1024 * 1024 * 1024; // 3 GB memory cap for single domain

        let mut in_args = [
            RemoteArg {
                buf: RemoteBuf {
                    buf: &sess_id as *const u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &dsp_queue_id as *const u64 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u64>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &n_hvx as *const u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &n_hmx as *const u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &max_vmem as *const u64 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u64>(),
                },
            },
        ];

        let start_scalars = remote_scalars_make(0, 5, 0);
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

    /// Register a mapped buffer with the DSP skeleton (Method 2: 2 in, 0 out).
    pub fn mmap_buffer(&self, fd: u32, size: u64) -> Result<(), CeraError> {
        let mut in_args = [
            RemoteArg {
                buf: RemoteBuf {
                    buf: &fd as *const u32 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u32>(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    buf: &size as *const u64 as *mut std::ffi::c_void,
                    len: std::mem::size_of::<u64>(),
                },
            },
        ];
        let mmap_scalars = remote_scalars_make(2, 2, 0);
        self.driver
            .invoke_skel(self.handle, mmap_scalars, &mut in_args)
    }

    /// Unregister a buffer from the DSP skeleton (Method 3: 1 in, 0 out).
    pub fn munmap_buffer(&self, fd: u32) -> Result<(), CeraError> {
        let mut in_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &fd as *const u32 as *mut std::ffi::c_void,
                len: std::mem::size_of::<u32>(),
            },
        }];
        let munmap_scalars = remote_scalars_make(3, 1, 0);
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
        // Stop session on DSP (Method 1: 0 in, 0 out)
        let stop_scalars = remote_scalars_make(1, 0, 0);
        let _ = self.driver.invoke_skel(self.handle, stop_scalars, &mut []);
        self.driver.close_skel_handle(self.handle);
    }
}

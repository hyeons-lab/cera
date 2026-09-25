//! Device management for Qualcomm Hexagon NPU.
//!
//! Handles device lifecycle, Unsigned PD session initiation, and
//! hardware resource registration via FastRPC.

use std::sync::Arc;

use super::queue::HexagonQueueSession;
use super::sys::{FastRpcDriver, RemoteArg, RemoteBuf, RemoteHandle64, remote_scalars_make};
use super::types::HtpHwInfo;
use crate::session::CeraError;

/// Target Hexagon architecture generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HexagonArch {
    V73, // Snapdragon 8 Gen 2
    V75, // Snapdragon 8 Gen 3
    V79, // Snapdragon 8 Elite
    V81, // Next-gen Snapdragon
}

impl HexagonArch {
    pub fn skel_filename(&self) -> &'static str {
        match self {
            Self::V73 => "libggml-htp-v73.so",
            Self::V75 => "libggml-htp-v75.so",
            Self::V79 => "libggml-htp-v79.so",
            Self::V81 => "libggml-htp-v81.so",
        }
    }

    /// Cross-language wire name (`HexagonProbeInfo.arch`): a pinned literal,
    /// not `Debug`, so a variant rename cannot silently change what mobile
    /// clients match on.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::V73 => "V73",
            Self::V75 => "V75",
            Self::V79 => "V79",
            Self::V81 => "V81",
        }
    }

    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            73 => Some(Self::V73),
            75 => Some(Self::V75),
            79 => Some(Self::V79),
            81 => Some(Self::V81),
            _ => None,
        }
    }
}

/// Represents an active connection to a Hexagon NPU execution domain.
pub struct HexagonDevice {
    driver: Arc<FastRpcDriver>,
    handle: RemoteHandle64,
    arch: HexagonArch,
    hw_info: HtpHwInfo,
    queue_session: HexagonQueueSession,
    profiler_on: bool,
}

#[repr(C)]
struct HtpStartPayload {
    sess_id: u32,
    _pad: u32,
    dsp_queue_id: u64,
    n_hvx: u32,
    use_hmx: u32,
    max_vmem: u64,
}

/// `htp_iface_profiler` (IDL Method 6) payload: mode + 8 PMU event ids.
/// Mode 1 (`HTP_PROF_BASIC`) fills the per-op profile descriptors the batch
/// response carries; without it the DSP leaves them zero and only the batch
/// totals are valid. PMU events are unused outside mode 2.
#[repr(C)]
struct HtpProfilerPayload {
    mode: u32,
    events: [u32; 8],
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

        // Create command queue session
        let mut queue_session = match HexagonQueueSession::new(Arc::clone(&driver)) {
            Ok(qs) => qs,
            Err(e) => {
                driver.close_skel_handle(handle);
                return Err(e);
            }
        };

        // Query DSP capabilities via `htp_iface_hwinfo` (IDL Method 8).
        // Like the start payload, scalars pack as one C struct — here a
        // single out-buffer. kparams thread counts derive from this
        // (llama's `sess->n_threads`); on failure take llama's
        // hwinfo-failure defaults.
        #[repr(C)]
        struct HwInfoOut {
            n_threads: u32,
            n_hvx: u32,
            n_hmx: u32,
            _pad: u32,
            vtcm_size: u64,
        }
        let mut hw_out = HwInfoOut {
            n_threads: 0,
            n_hvx: 0,
            n_hmx: 0,
            _pad: 0,
            vtcm_size: 0,
        };
        let mut hw_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &mut hw_out as *mut _ as *mut std::ffi::c_void,
                len: std::mem::size_of::<HwInfoOut>(),
            },
        }];
        let hw_info = if driver
            .invoke_skel(handle, remote_scalars_make(8, 0, 1), &mut hw_args)
            .is_ok()
            && hw_out.n_threads > 0
        {
            HtpHwInfo {
                n_threads: hw_out.n_threads.min(10), // HTP_MAX_NTHREADS
                n_hvx: hw_out.n_hvx,
                n_hmx: hw_out.n_hmx,
                vtcm_size: hw_out.vtcm_size,
            }
        } else {
            if std::env::var_os("CERA_HEXAGON_DEBUG").is_some() {
                eprintln!("[cera-hexagon] hwinfo query failed; using fallback threads=8");
            }
            HtpHwInfo {
                n_threads: 8,
                n_hvx: 8,
                n_hmx: 1,
                vtcm_size: 8 * 1024 * 1024,
            }
        };
        queue_session.set_dsp_threads(hw_info.n_threads);
        if std::env::var_os("CERA_HEXAGON_DEBUG").is_some() {
            eprintln!(
                "[cera-hexagon] hwinfo: threads={} hvx={} hmx={} vtcm={}MB",
                hw_info.n_threads,
                hw_info.n_hvx,
                hw_info.n_hmx,
                hw_info.vtcm_size / (1024 * 1024),
            );
        }

        // Start session on DSP via htp_iface_start (Method 2: 1 in, 0 out)
        let start_payload = HtpStartPayload {
            sess_id: 0,
            _pad: 0,
            dsp_queue_id: queue_session.queue_id(),
            n_hvx: 0,
            use_hmx: 1,
            max_vmem: 0xc800_0000,
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

        // Enable the DSP profiler alongside `CERA_HEXAGON_PROFILE` so the
        // per-op descriptors carry timings (llama enables `htp_iface_profiler`
        // when `GGML_HEXAGON_PROFILE` is set). A failed enable is non-fatal:
        // batch totals stay valid either way.
        let profiler_on = if std::env::var_os("CERA_HEXAGON_PROFILE").is_some() {
            let payload = HtpProfilerPayload {
                mode: 1, // HTP_PROF_BASIC
                events: [0; 8],
            };
            let mut args = [RemoteArg {
                buf: RemoteBuf {
                    buf: &payload as *const _ as *mut std::ffi::c_void,
                    len: std::mem::size_of::<HtpProfilerPayload>(),
                },
            }];
            driver
                .invoke_skel(handle, remote_scalars_make(6, 1, 0), &mut args)
                .is_ok()
        } else {
            false
        };

        Ok(Self {
            driver,
            handle,
            arch,
            hw_info,
            queue_session,
            profiler_on,
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

    /// Register a mapped buffer with the DSP skeleton.
    pub fn mmap_buffer(&self, _fd: u32, _size: u64) -> Result<(), CeraError> {
        Ok(())
    }

    /// Unregister a buffer from the DSP skeleton.
    pub fn munmap_buffer(&self, _fd: u32) -> Result<(), CeraError> {
        Ok(())
    }

    /// Mutable reference to the command queue session.
    pub fn queue_session_mut(&mut self) -> &mut HexagonQueueSession {
        &mut self.queue_session
    }
}

impl Drop for HexagonDevice {
    fn drop(&mut self) {
        if self.profiler_on {
            let payload = HtpProfilerPayload {
                mode: 0, // HTP_PROF_DISABLED
                events: [0; 8],
            };
            let mut args = [RemoteArg {
                buf: RemoteBuf {
                    buf: &payload as *const _ as *mut std::ffi::c_void,
                    len: std::mem::size_of::<HtpProfilerPayload>(),
                },
            }];
            let _ = self
                .driver
                .invoke_skel(self.handle, remote_scalars_make(6, 1, 0), &mut args);
        }
        // Stop session on DSP (Method 3: 0 in, 0 out)
        let stop_scalars = remote_scalars_make(3, 0, 0);
        let _ = self.driver.invoke_skel(self.handle, stop_scalars, &mut []);
        self.driver.close_skel_handle(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FFI `arch` wire strings are pinned literals: mobile clients
    /// match on them, so any rename must be a deliberate breaking change.
    /// A new variant fails here twice (exhaustive match, list length) until
    /// its wire string is deliberately pinned.
    #[test]
    fn arch_short_names_pinned() {
        for arch in super::super::skels::PROBE_ARCHS {
            let expected = match arch {
                HexagonArch::V73 => "V73",
                HexagonArch::V75 => "V75",
                HexagonArch::V79 => "V79",
                HexagonArch::V81 => "V81",
            };
            assert_eq!(arch.short_name(), expected);
        }
        assert_eq!(super::super::skels::PROBE_ARCHS.len(), 4);
        // Membership + order: order decides the winning DSP on multi-arch
        // devices and the install set, so a dup or reorder must fail loudly.
        assert_eq!(
            super::super::skels::PROBE_ARCHS,
            [
                HexagonArch::V79,
                HexagonArch::V75,
                HexagonArch::V73,
                HexagonArch::V81
            ]
        );
    }
}

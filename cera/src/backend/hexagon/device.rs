//! Device management for Qualcomm Hexagon NPU.
//!
//! Handles device lifecycle, Unsigned PD session initiation, and
//! hardware resource registration via FastRPC.

use std::sync::{Arc, Mutex};

use super::LockOrRecover;
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
    V85, // Snapdragon 8 Elite Gen 6
}

impl HexagonArch {
    pub fn skel_filename(&self) -> &'static str {
        match self {
            Self::V73 => "libggml-htp-v73.so",
            Self::V75 => "libggml-htp-v75.so",
            Self::V79 => "libggml-htp-v79.so",
            Self::V81 => "libggml-htp-v81.so",
            Self::V85 => "libggml-htp-v85.so",
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
            Self::V85 => "V85",
        }
    }

    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            73 => Some(Self::V73),
            75 => Some(Self::V75),
            79 => Some(Self::V79),
            81 => Some(Self::V81),
            85 => Some(Self::V85),
            _ => None,
        }
    }
}

/// Process-wide count of live [`PowerVote`] holders. The FastRPC latency QoS
/// and wakelock are process-wide and not counted by the driver, so the votes
/// are taken on the 0 to 1 transition and released on 1 to 0; one device
/// dropping must not cancel the votes another still decoding relies on.
struct VoteCounter(Mutex<usize>);

impl VoteCounter {
    const fn new() -> Self {
        Self(Mutex::new(0))
    }

    /// Register a holder, running `on_first` under the lock when it is the
    /// first (so the vote calls of concurrent enter/leave cannot interleave).
    fn enter(&self, on_first: impl FnOnce()) {
        let mut n = self.0.lock_or_recover();
        if *n == 0 {
            on_first();
        }
        *n += 1;
    }

    /// Unregister a holder, running `on_last` under the lock when it was the
    /// last. Saturates at zero rather than underflowing.
    fn leave(&self, on_last: impl FnOnce()) {
        let mut n = self.0.lock_or_recover();
        *n = n.saturating_sub(1);
        if *n == 0 {
            on_last();
        }
    }
}

static POWER_VOTES: VoteCounter = VoteCounter::new();

/// RAII holder of the CDSP latency QoS and wakelock votes. Held from the top
/// of [`HexagonDevice::new`], so every early-return path releases its share.
struct PowerVote(Arc<FastRpcDriver>);

impl PowerVote {
    fn acquire(driver: &Arc<FastRpcDriver>) -> Self {
        POWER_VOTES.enter(|| {
            // Prevent DSP power collapse during active inference. Best
            // effort: a refused vote costs performance, not correctness.
            if let Err(e) = driver.set_latency_qos(100) {
                tracing::warn!("Hexagon latency QoS vote failed: {e}");
            }
            if let Err(e) = driver.set_wakelock(true) {
                tracing::warn!("Hexagon wakelock vote failed: {e}");
            }
        });
        Self(Arc::clone(driver))
    }
}

impl Drop for PowerVote {
    fn drop(&mut self) {
        POWER_VOTES.leave(|| {
            let _ = self.0.set_wakelock(false);
            let _ = self.0.set_latency_qos(0);
        });
    }
}

/// Represents an active connection to a Hexagon NPU execution domain.
pub struct HexagonDevice {
    driver: Arc<FastRpcDriver>,
    handle: RemoteHandle64,
    arch: HexagonArch,
    hw_info: HtpHwInfo,
    /// Torn down explicitly in `Drop` (taken, then the skel handle closed).
    queue_session: Option<HexagonQueueSession>,
    profiler_on: bool,
    /// Field order matters: struct fields drop top to bottom after `Drop::drop`
    /// returns, so declaring this last keeps the votes held until the queue
    /// session and the skel handle are closed.
    _power_vote: PowerVote,
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

        // Held across every early return below, so a failed arch probe never
        // leaves the process-wide votes on.
        let power_vote = PowerVote::acquire(&driver);

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

        queue_session.set_skel_handle(handle);

        // Query DSP capabilities via `htp_iface_hwinfo` (IDL Method 8).
        // Like the start payload, scalars pack as one C struct, here a
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
            super::hexagon_warn!("hwinfo query failed; using fallback threads=8");
            HtpHwInfo {
                n_threads: 8,
                n_hvx: 8,
                n_hmx: 1,
                vtcm_size: 8 * 1024 * 1024,
            }
        };
        queue_session.set_dsp_threads(hw_info.n_threads);
        tracing::debug!(
            target: "cera::hexagon",
            "hwinfo: threads={} hvx={} hmx={} vtcm={}MB",
            hw_info.n_threads,
            hw_info.n_hvx,
            hw_info.n_hmx,
            hw_info.vtcm_size / (1024 * 1024),
        );

        // Start session on DSP via htp_iface_start (Method 2: 1 in, 0 out)
        let mut start_payload = HtpStartPayload {
            sess_id: 0,
            _pad: 0,
            dsp_queue_id: queue_session.queue_id(),
            n_hvx: 0,
            use_hmx: 1,
            max_vmem: 0xc800_0000,
        };

        let mut in_args = [RemoteArg {
            buf: RemoteBuf {
                buf: &mut start_payload as *mut _ as *mut std::ffi::c_void,
                len: std::mem::size_of::<HtpStartPayload>(),
            },
        }];

        let start_scalars = remote_scalars_make(2, 1, 0);
        if let Err(e) = driver.invoke_skel(handle, start_scalars, &mut in_args) {
            // Same order as `Drop`: queue session first, then the handle.
            drop(queue_session);
            driver.close_skel_handle(handle);
            return Err(e);
        }

        // Enable the DSP profiler alongside `CERA_HEXAGON_PROFILE` so the
        // per-op descriptors carry timings (llama enables `htp_iface_profiler`
        // when `GGML_HEXAGON_PROFILE` is set). A failed enable is non-fatal:
        // batch totals stay valid either way.
        let profiler_on = if std::env::var_os("CERA_HEXAGON_PROFILE").is_some() {
            let mut payload = HtpProfilerPayload {
                mode: 1, // HTP_PROF_BASIC
                events: [0; 8],
            };
            let mut args = [RemoteArg {
                buf: RemoteBuf {
                    buf: &mut payload as *mut _ as *mut std::ffi::c_void,
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
            queue_session: Some(queue_session),
            profiler_on,
            _power_vote: power_vote,
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

    /// Mutable reference to the command queue session.
    pub fn queue_session_mut(&mut self) -> &mut HexagonQueueSession {
        self.queue_session
            .as_mut()
            .expect("HexagonDevice: queue_session is active")
    }
}

impl Drop for HexagonDevice {
    fn drop(&mut self) {
        // Drop queue session before shutting down DSP hardware session.
        drop(self.queue_session.take());

        if self.profiler_on {
            let mut payload = HtpProfilerPayload {
                mode: 0, // HTP_PROF_DISABLED
                events: [0; 8],
            };
            let mut args = [RemoteArg {
                buf: RemoteBuf {
                    buf: &mut payload as *mut _ as *mut std::ffi::c_void,
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

/// Probe and initialize a Hexagon device, respecting optional architecture override.
pub fn probe_device(
    driver: &Arc<FastRpcDriver>,
    arch_override: Option<HexagonArch>,
) -> Result<HexagonDevice, CeraError> {
    probe_device_with(driver, arch_override, |k| std::env::var_os(k))
}

/// [`probe_device`] with the kill-switch env lookup injected (tests).
pub(crate) fn probe_device_with(
    driver: &Arc<FastRpcDriver>,
    arch_override: Option<HexagonArch>,
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<HexagonDevice, CeraError> {
    super::ensure_not_disabled_with(get)?;
    let probe_archs: &[HexagonArch] = match arch_override {
        Some(ref arch) => std::slice::from_ref(arch),
        None => &super::skels::PROBE_ARCHS,
    };
    let mut device_opt = None;
    let mut probed_errors = Vec::new();
    for &arch in probe_archs {
        match HexagonDevice::new(Arc::clone(driver), arch) {
            Ok(dev) => {
                tracing::info!(arch = ?arch, "initialized Hexagon NPU device");
                device_opt = Some(dev);
                break;
            }
            Err(e) => {
                probed_errors.push(format!("{arch:?}: {e}"));
            }
        }
    }
    match device_opt {
        Some(d) => Ok(d),
        None => Err(CeraError::Backend(format!(
            "no compatible Hexagon skeleton library found. Errors: {}",
            probed_errors.join("; ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::super::sys::fake;
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
                HexagonArch::V85 => "V85",
            };
            assert_eq!(arch.short_name(), expected);
        }
        assert_eq!(super::super::skels::PROBE_ARCHS.len(), 5);
        // Membership + order: order decides the winning DSP on multi-arch
        // devices and the install set, so a dup or reorder must fail loudly.
        assert_eq!(
            super::super::skels::PROBE_ARCHS,
            [
                HexagonArch::V79,
                HexagonArch::V75,
                HexagonArch::V73,
                HexagonArch::V81,
                HexagonArch::V85,
            ]
        );
    }

    /// A failed start closes in `Drop` order: the queue session first, then
    /// the skel handle.
    #[test]
    fn failed_start_closes_queue_before_handle() {
        fake::reset();
        fake::with(|s| s.fail_invoke_method = Some(2));
        let r = HexagonDevice::new(fake::driver(), HexagonArch::V75);
        assert!(r.is_err());
        let ev = fake::events();
        let q = ev.iter().position(|e| *e == fake::Event::QueueClose);
        let h = ev.iter().position(|e| *e == fake::Event::CloseSkel);
        assert!(q.is_some() && h.is_some() && q < h, "{ev:?}");
    }

    #[test]
    fn test_hexagon_arch_from_u32() {
        assert_eq!(HexagonArch::from_u32(73), Some(HexagonArch::V73));
        assert_eq!(HexagonArch::from_u32(75), Some(HexagonArch::V75));
        assert_eq!(HexagonArch::from_u32(79), Some(HexagonArch::V79));
        assert_eq!(HexagonArch::from_u32(81), Some(HexagonArch::V81));
        assert_eq!(HexagonArch::from_u32(85), Some(HexagonArch::V85));
        assert_eq!(HexagonArch::from_u32(0), None);
        assert_eq!(HexagonArch::from_u32(72), None);
        assert_eq!(HexagonArch::from_u32(74), None);
        assert_eq!(HexagonArch::from_u32(100), None);
    }
}

#[cfg(test)]
mod vote_tests {
    use super::VoteCounter;
    use std::cell::Cell;

    #[test]
    fn votes_toggle_only_on_first_enter_and_last_leave() {
        let c = VoteCounter::new();
        let (on, off) = (Cell::new(0), Cell::new(0));
        c.enter(|| on.set(on.get() + 1));
        c.enter(|| on.set(on.get() + 1));
        assert_eq!(on.get(), 1, "second holder must not re-vote");
        c.leave(|| off.set(off.get() + 1));
        assert_eq!(off.get(), 0, "one holder still live: votes must stay on");
        c.leave(|| off.set(off.get() + 1));
        assert_eq!(off.get(), 1);
        // Re-entering after a full release votes again.
        c.enter(|| on.set(on.get() + 1));
        assert_eq!(on.get(), 2);
    }

    #[test]
    fn leave_at_zero_saturates_and_survives_poison() {
        let c = std::sync::Arc::new(VoteCounter::new());
        let off = Cell::new(0);
        c.leave(|| off.set(off.get() + 1));
        c.leave(|| off.set(off.get() + 1));
        // Saturating: each leave at zero reports "last" but never underflows.
        assert_eq!(off.get(), 2);
        let c2 = std::sync::Arc::clone(&c);
        let _ = std::thread::spawn(move || {
            c2.enter(|| panic!("poison the counter mutex"));
        })
        .join();
        let on = Cell::new(0);
        c.enter(|| on.set(1));
        assert_eq!(on.get(), 1, "a poisoned counter still counts");
    }
}

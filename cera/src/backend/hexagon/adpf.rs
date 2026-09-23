//! Android ADPF CPU performance hints for Hexagon NPU sessions.
//!
//! NPU inference is DSP-bound: the host thread sleeps ~8ms per token inside
//! `dspqueue_read`. Left alone, the CPU governor sees an idle thread and
//! collapses host clocks (observed 4.47GHz -> 1.02GHz on Snapdragon 8 Elite),
//! stretching every RPC round trip and cache op into a ~2.9ms/token tax.
//!
//! The fix is Android's Application-Responsive Performance Framework (ADPF):
//! a hint session telling the power HAL this thread has periodic work with a
//! ~10ms budget, so it holds adequate CPU. Symbols resolve at runtime from
//! `libandroid.so` (API 33+); anything missing degrades to `None` and the
//! backend runs unhinted. Set `CERA_HEXAGON_ADPF=0` to disable explicitly,
//! `CERA_HEXAGON_ADPF_TARGET_MS` (default 10) to retune the work budget.

use std::ffi::{CString, c_void};

type GetManagerFn = unsafe extern "C" fn() -> *mut c_void;
type CreateSessionFn = unsafe extern "C" fn(*mut c_void, *const i32, usize, i64) -> *mut c_void;
type ReportActualFn = unsafe extern "C" fn(*mut c_void, i64) -> i32;
type CloseSessionFn = unsafe extern "C" fn(*mut c_void);

/// Current kernel thread id for ADPF session binding.
fn current_tid() -> Option<i32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        // Raw syscall: no Bionic version dependency (unlike the gettid wrapper).
        Some(unsafe { libc::syscall(libc::SYS_gettid) as i32 })
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

/// An open ADPF hint session bound to one forward thread.
///
/// Best-effort by design: every failure degrades to `None` (unhinted) rather
/// than erroring model load. Recreated automatically if the forward thread
/// changes under a multi-threaded host.
pub struct AdpfSession {
    _lib: *mut c_void,
    manager: *mut c_void,
    session: *mut c_void,
    create: CreateSessionFn,
    report_actual: ReportActualFn,
    close: CloseSessionFn,
    target_nanos: i64,
    tid: i32,
}

// Raw ADPF handles are used from one thread at a time (behind a Mutex).
unsafe impl Send for AdpfSession {}

impl AdpfSession {
    /// Open a hint session for the calling thread. Returns `None` when
    /// disabled, unavailable (non-Android, API < 33, missing symbols), or
    /// rejected by the power HAL.
    pub fn try_open(target_nanos: i64) -> Option<Self> {
        if std::env::var("CERA_HEXAGON_ADPF")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            return None;
        }
        #[cfg(unix)]
        {
            Self::open_unix(target_nanos)
        }
        #[cfg(not(unix))]
        {
            let _ = target_nanos;
            None
        }
    }

    #[cfg(unix)]
    fn open_unix(target_nanos: i64) -> Option<Self> {
        let tid = current_tid()?;
        unsafe {
            let lib_name = CString::new("libandroid.so").unwrap();
            let lib = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if lib.is_null() {
                return None;
            }
            macro_rules! resolve {
                ($name:expr, $type:ty) => {{
                    let c_sym = CString::new($name).unwrap();
                    let sym = libc::dlsym(lib, c_sym.as_ptr());
                    if sym.is_null() {
                        libc::dlclose(lib);
                        return None;
                    }
                    std::mem::transmute::<*mut c_void, $type>(sym)
                }};
            }
            let get_manager: GetManagerFn = resolve!("APerformanceHint_getManager", GetManagerFn);
            let create: CreateSessionFn =
                resolve!("APerformanceHint_createSession", CreateSessionFn);
            let report_actual: ReportActualFn =
                resolve!("APerformanceHint_reportActualWorkDuration", ReportActualFn);
            let close: CloseSessionFn = resolve!("APerformanceHint_closeSession", CloseSessionFn);

            let manager = get_manager();
            if manager.is_null() {
                libc::dlclose(lib);
                return None;
            }
            let session = create(manager, &tid, 1, target_nanos);
            if session.is_null() {
                libc::dlclose(lib);
                tracing::debug!("cera-hexagon: ADPF session rejected by power HAL");
                return None;
            }
            tracing::info!(
                target_ms = target_nanos / 1_000_000,
                "cera-hexagon: ADPF hint session active"
            );
            // lib intentionally leaked: process-lifetime, keeps symbols valid.
            Some(Self {
                _lib: lib,
                manager,
                session,
                create,
                report_actual,
                close,
                target_nanos,
                tid,
            })
        }
    }

    /// Report one token's wall time. Rebinds the session if the calling
    /// thread changed since it was opened.
    pub fn report(&mut self, actual_nanos: i64) {
        if let Some(tid) = current_tid()
            && tid != self.tid
        {
            // Create-before-close: a failed rebind keeps the old session.
            let session = unsafe { (self.create)(self.manager, &tid, 1, self.target_nanos) };
            if !session.is_null() {
                unsafe { (self.close)(self.session) };
                self.session = session;
                self.tid = tid;
            }
        }
        unsafe {
            (self.report_actual)(self.session, actual_nanos.max(0));
        }
    }
}

impl Drop for AdpfSession {
    fn drop(&mut self) {
        unsafe { (self.close)(self.session) };
    }
}

#[cfg(test)]
mod tests {
    use super::AdpfSession;

    #[test]
    fn adpf_degrades_gracefully_off_android() {
        // libandroid.so only exists on Android; everywhere else open must
        // degrade to None rather than crash or hang.
        #[cfg(not(target_os = "android"))]
        assert!(AdpfSession::try_open(10_000_000).is_none());
        // On-device this only asserts the open path is total: Some iff the
        // platform provides ADPF (API 33+ with power-HAL support).
        #[cfg(target_os = "android")]
        let _ = AdpfSession::try_open(10_000_000);
    }
}

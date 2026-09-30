//! Native Qualcomm Hexagon NPU backend for Cera.
//!
//! Provides hardware-accelerated tensor operations on Snapdragon Hexagon Tensor Processors (HTP)
//! through FastRPC shared memory (`rpcmem`) and asynchronous command queues (`dspqueue`).

#[cfg(not(unix))]
compile_error!(
    "the `hexagon` feature needs a unix target (FastRPC via libc dlopen); \
     it cannot be built for wasm32 or Windows"
);

pub mod adpf;
pub mod device;
pub mod params;
pub mod queue;
pub mod repack;
pub mod rpcmem;
pub mod skels;
pub mod sys;
pub mod types;

pub use adpf::AdpfSession;
pub use device::{HexagonArch, HexagonDevice, probe_device};
pub use params::*;
pub use queue::{BufferIndexMap, HexagonQueueSession, StagedBatch};
pub use repack::*;
pub use rpcmem::RpcmemBuffer;
pub use skels::{HexagonProbe, PROBE_ARCHS, embedded_skel, install_skels, probe};
pub use sys::FastRpcDriver;
pub use types::*;

use crate::session::CeraError;
use std::sync::{Arc, Mutex, MutexGuard};

/// Report a recovered poison to stderr and `tracing`. Hosts on Android and iOS
/// often install no tracing subscriber, so the stderr line keeps the panic's
/// aftermath visible there.
pub(crate) fn report_poison(what: &str, action: &str) {
    eprintln!("[cera-hexagon] {what} was poisoned by a panic; {action}");
    tracing::warn!("Hexagon {what} was poisoned by a panic; {action}");
}

/// Lock a mutex, recovering from poison. For state that is rewritten or
/// discarded on entry to every call (scratch buffers, the whisper and audio
/// device locks, whose pending batch each entry point drops), so a panic
/// mid-call leaves nothing torn. A single scalar that cannot tear is also
/// fine. State that persists across calls needs `lock_reporting_poison` and
/// an explicit reset (the LFM2 device lock fails closed instead). Logs the
/// recovery, since the original panic would otherwise leave no trace once the
/// lock error is swallowed.
pub(crate) trait LockOrRecover<T> {
    fn lock_or_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> LockOrRecover<T> for Mutex<T> {
    fn lock_or_recover(&self) -> MutexGuard<'_, T> {
        let (guard, poisoned) = lock_reporting_poison(self);
        if poisoned {
            report_poison(
                &format!("{} lock", std::any::type_name::<T>()),
                "recovering",
            );
        }
        guard
    }
}

/// Lock a mutex whose contents persist across calls, reporting whether it was
/// poisoned. On poison the flag is cleared so recovery happens once: a std
/// `Mutex` stays poisoned after `into_inner`, which would otherwise make every
/// later call look like a fresh failure. The caller must repair the contents
/// when the flag is set.
pub(crate) fn lock_reporting_poison<T>(m: &Mutex<T>) -> (MutexGuard<'_, T>, bool) {
    match m.lock() {
        Ok(g) => (g, false),
        Err(e) => {
            m.clear_poison();
            (e.into_inner(), true)
        }
    }
}

/// Lock a mutex without clearing its poison flag: for paths that touch only
/// counters and must not turn a failed-closed device back into a usable one.
pub(crate) fn lock_keeping_poison<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Lock a cache slot that is patched in place, discarding its contents when
/// the lock was poisoned: a panic mid-patch may leave the value half-updated,
/// and `None` sends the caller down its rebuild path.
pub(crate) fn lock_or_discard<T>(m: &Mutex<Option<T>>) -> MutexGuard<'_, Option<T>> {
    let (mut guard, poisoned) = lock_reporting_poison(m);
    if poisoned {
        report_poison(
            &format!("cached {}", std::any::type_name::<T>()),
            "discarding",
        );
        *guard = None;
    }
    guard
}

/// Shared Hexagon compute context managing the FastRPC driver and device session.
pub struct HexagonContext {
    driver: Arc<FastRpcDriver>,
}

impl HexagonContext {
    /// Initialize the Hexagon backend by loading the FastRPC userspace driver.
    pub fn new() -> Result<Arc<Self>, CeraError> {
        let driver = FastRpcDriver::load()?;
        if std::env::var("CERA_HEXAGON_SPIN")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            // Debug knob: park a spinner to hold CPU clocks across DSP-bound
            // waits (validates governor effects; burns a core, whereas ADPF is the
            // production answer). Detached: runs until process exit. `OnceLock`:
            // without it every model reload parks another thread and the
            // knob skews the runs it was meant to stabilize; the stored
            // `Result` keeps a spawn failure visible to every caller instead
            // of only the first.
            static SPIN_RESULT: std::sync::OnceLock<Result<(), CeraError>> =
                std::sync::OnceLock::new();
            SPIN_RESULT
                .get_or_init(|| {
                    tracing::warn!(
                        "cera-hexagon: CERA_HEXAGON_SPIN=1 parking a keepalive spinner thread"
                    );
                    std::thread::Builder::new()
                        .name("cera-hex-spin".into())
                        .spawn(|| {
                            loop {
                                std::hint::spin_loop();
                            }
                        })
                        .map(|_| ())
                        .map_err(|e| CeraError::Backend(format!("spinner spawn failed: {e}")))
                })
                .as_ref()
                .map_err(|e| CeraError::Backend(e.to_string()))?;
        }
        Ok(Arc::new(Self { driver }))
    }

    /// Access the underlying FastRPC driver handle.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }
}

#[cfg(test)]
mod poison_tests {
    use super::*;

    #[test]
    fn lock_reporting_poison_reports_once_then_clears() {
        let m = Arc::new(Mutex::new(7u32));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison the lock");
        })
        .join();
        assert!(m.is_poisoned());

        let (g, poisoned) = lock_reporting_poison(&m);
        assert!(poisoned, "first lock after the panic reports poison");
        assert_eq!(*g, 7);
        drop(g);

        let (_g, poisoned) = lock_reporting_poison(&m);
        assert!(!poisoned, "recovery must not repeat on later calls");
    }

    #[test]
    fn lock_keeping_poison_leaves_the_flag_set() {
        let m = Arc::new(Mutex::new(3u32));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison the lock");
        })
        .join();
        assert_eq!(*lock_keeping_poison(&m), 3);
        assert!(
            m.is_poisoned(),
            "must stay poisoned until an explicit reset"
        );
    }

    #[test]
    fn lock_or_discard_drops_a_poisoned_value_once() {
        let m = Arc::new(Mutex::new(Some(5u32)));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison the lock");
        })
        .join();
        assert_eq!(*lock_or_discard(&m), None, "poisoned value is dropped");
        *lock_or_discard(&m) = Some(6);
        assert_eq!(*lock_or_discard(&m), Some(6), "later locks keep the value");
    }

    #[test]
    fn lock_or_recover_clears_poison() {
        let m = Arc::new(Mutex::new(1u32));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison the lock");
        })
        .join();
        assert_eq!(*m.lock_or_recover(), 1);
        assert!(!m.is_poisoned(), "recovery must clear the poison flag");
    }
}

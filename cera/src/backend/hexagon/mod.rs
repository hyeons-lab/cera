//! Native Qualcomm Hexagon NPU backend for Cera.
//!
//! Provides hardware-accelerated tensor operations on Snapdragon Hexagon Tensor Processors (HTP)
//! through FastRPC shared memory (`rpcmem`) and asynchronous command queues (`dspqueue`).

#[cfg(not(unix))]
compile_error!(
    "the `hexagon` feature needs a unix target (FastRPC via libc dlopen); \
     it cannot be built for wasm32 or Windows"
);

/// Warning on the Hexagon path, emitted through `tracing::warn!` and
/// `eprintln!("cera-hexagon: ...")` with one formatted message. `cera-ffi`
/// installs no tracing subscriber, so on shipping platforms (Android logcat)
/// stderr is the only channel; hence every warning and failure on the model
/// path goes through this pair, and never through `tracing` alone.
macro_rules! hexagon_warn {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        tracing::warn!(target: "cera::hexagon", "{msg}");
        eprintln!("cera-hexagon: {msg}");
    }};
}

/// [`hexagon_warn!`] at error level, for failures (a loader that could not
/// bring the NPU path up, a lost device).
macro_rules! hexagon_error {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        tracing::error!(target: "cera::hexagon", "{msg}");
        eprintln!("cera-hexagon: {msg}");
    }};
}

pub(crate) use {hexagon_error, hexagon_warn};

pub mod adpf;
pub mod device;
pub(crate) mod dispatch;
#[cfg(test)]
pub(crate) mod op_capture;
pub mod params;
pub mod queue;
pub mod repack;
pub mod rpcmem;
pub mod skels;
pub mod sys;
pub mod types;

pub use adpf::AdpfSession;
pub use device::{HexagonArch, HexagonDevice, probe_device};
pub(crate) use device::{arch_override, arch_override_with};
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

/// Whether the `CERA_DISABLE_HEXAGON` / `CERA_NO_HEXAGON` kill switches are
/// set. Fail-safe parse: any value except empty, `0` or `false` disables, so
/// an unexpected spelling (`yes`, `on`) still turns the NPU off. `get` is the
/// env lookup, injected so the rule is unit-testable without process env.
fn disabled_by(get: impl Fn(&str) -> Option<std::ffi::OsString>) -> bool {
    ["CERA_DISABLE_HEXAGON", "CERA_NO_HEXAGON"]
        .iter()
        .filter_map(|k| get(k))
        .any(|v| {
            let v = v.to_string_lossy();
            let v = v.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        })
}

/// Error text of the kill switch (see [`ensure_not_disabled_with`]).
const MSG_DISABLED: &str = "Hexagon backend disabled via CERA_DISABLE_HEXAGON / CERA_NO_HEXAGON";
/// Error text of a device with no FastRPC userspace driver at all.
pub(crate) const MSG_DRIVER_ABSENT: &str =
    "Qualcomm FastRPC driver (libcdsprpc.so) not found on this system";

/// True for the two failures that just mean "no NPU here" (kill switch set,
/// FastRPC driver absent), as opposed to a driver that is present but broken.
fn is_expected_absence(e: &CeraError) -> bool {
    let text = e.to_string();
    text.contains(MSG_DISABLED) || text.contains(MSG_DRIVER_ABSENT)
}

/// Log why [`HexagonContext::new`] failed for an optional accelerator (`who`
/// names it). The two expected cases (kill switch set, no FastRPC driver on
/// this device) are info-level; anything else is a real driver-load failure on
/// a device that has an NPU and goes through [`hexagon_warn!`] so it reaches
/// logcat, where `cera-ffi` installs no tracing subscriber.
pub(crate) fn log_context_unavailable(who: &str, e: &CeraError) {
    if is_expected_absence(e) {
        tracing::info!("{who}: Hexagon backend unavailable ({e}), falling back");
    } else {
        hexagon_warn!("{who}: Hexagon driver failed to load ({e}), falling back");
    }
}

/// Central kill-switch gate. Every path that can open the FastRPC driver or
/// a skel goes through one of these entries, so the switch holds for all of
/// them: [`HexagonContext::new`] (every model loader: LFM2, audio, vision,
/// whisper), [`probe_device`] (pre-built driver handles), and
/// [`skels::probe`] (the public `hexagon_probe` FFI capability check, which
/// must not report a working NPU or take a power vote while loaders refuse).
///
/// `get` is the env lookup, injected so tests can drive every gated entry
/// without touching the process environment; production entries pass
/// `std::env::var_os`.
pub(crate) fn ensure_not_disabled_with(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<(), CeraError> {
    if disabled_by(get) {
        return Err(CeraError::Backend(MSG_DISABLED.into()));
    }
    Ok(())
}

/// The production env lookup every kill-switch entry uses; a named fn so a
/// test can pin that it reads the real process environment.
fn env_lookup(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(key)
}

/// Shared Hexagon compute context managing the FastRPC driver and device session.
pub struct HexagonContext {
    driver: Arc<FastRpcDriver>,
}

impl HexagonContext {
    /// Initialize the Hexagon backend by loading the FastRPC userspace driver.
    ///
    /// Honors the `CERA_DISABLE_HEXAGON` / `CERA_NO_HEXAGON` kill switches for
    /// every model that goes through this context.
    pub fn new() -> Result<Arc<Self>, CeraError> {
        Self::new_with(env_lookup, FastRpcDriver::load)
    }

    /// [`Self::new`] with the env lookup and driver loader injected (tests).
    fn new_with(
        get: impl Fn(&str) -> Option<std::ffi::OsString>,
        load: impl FnOnce() -> Result<Arc<FastRpcDriver>, CeraError>,
    ) -> Result<Arc<Self>, CeraError> {
        ensure_not_disabled_with(get)?;
        let driver = load()?;
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
            static SPIN_RESULT: std::sync::OnceLock<Result<(), String>> =
                std::sync::OnceLock::new();
            SPIN_RESULT
                .get_or_init(|| {
                    hexagon_warn!("CERA_HEXAGON_SPIN=1 parking a keepalive spinner thread");
                    std::thread::Builder::new()
                        .name("cera-hex-spin".into())
                        .spawn(|| {
                            loop {
                                std::hint::spin_loop();
                            }
                        })
                        .map(|_| ())
                        .map_err(|e| format!("spinner spawn failed: {e}"))
                })
                .as_ref()
                .map_err(|msg| CeraError::Backend(msg.clone()))?;
        }
        Ok(Arc::new(Self { driver }))
    }

    /// Access the underlying FastRPC driver handle.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }
}

/// Stage a diarizer accelerator on the Hexagon NPU and install it on the model
/// (`stage` builds the accelerator in DSP memory, `set` installs it). The two
/// diarizers shared this body line for line, so it lives here; `name` is the
/// model name (`"Sortformer"`, `"Nemotron3"`), from which the log tags derive.
///
/// Returns `None` (with a log line naming the reason) when the model already
/// has an accelerator, the DSP is unavailable, staging fails, or installing
/// fails: the caller keeps the CPU. On success returns the staged accelerator
/// (the same `Arc` the model holds).
///
/// Call once during setup, from one thread; concurrent staging of two
/// accelerators on the same model is not supported. A losing race would
/// double-stage DSP memory and report `None` while another staging won.
pub(crate) fn stage_diarizer_accelerator<T>(
    name: &'static str,
    has_accelerator: bool,
    max_frames: usize,
    stage: impl FnOnce(Arc<FastRpcDriver>, Arc<Mutex<HexagonDevice>>) -> Result<T, CeraError>,
    set: impl FnOnce(Arc<T>) -> anyhow::Result<()>,
) -> Option<Arc<T>> {
    if has_accelerator {
        tracing::info!("Hexagon{name}: the model already has an accelerator; keeping it");
        return None;
    }
    let context = HexagonContext::new()
        .inspect_err(|e| {
            log_context_unavailable(&format!("Hexagon{name}"), e);
        })
        .ok()?;
    let arch_override = arch_override();
    let dev = match probe_device(context.driver(), arch_override) {
        Ok(d) => d,
        Err(e) => {
            tracing::info!("Hexagon{name}: DSP device unavailable ({e}), using the CPU");
            return None;
        }
    };
    let device = Arc::new(Mutex::new(dev));
    let staged = match stage(Arc::clone(context.driver()), device) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            hexagon_error!("failed to stage {name} on the NPU: {e}");
            return None;
        }
    };
    if let Err(e) = set(staged.clone()) {
        hexagon_warn!("Hexagon{name}: {e:#}");
        return None;
    }
    let lower = name.to_lowercase();
    tracing::info!("{lower}: using the Hexagon NPU ({max_frames} encoder frames)");
    Some(staged)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| OsString::from(v))
        }
    }

    /// Only "no NPU here" (kill switch, absent driver) is logged quietly; a
    /// present-but-broken driver must reach the warn channel.
    #[test]
    fn expected_absence_is_only_kill_switch_and_missing_driver() {
        let err = |m: &str| CeraError::Backend(m.into());
        assert!(is_expected_absence(&err(MSG_DISABLED)));
        assert!(is_expected_absence(&err(MSG_DRIVER_ABSENT)));
        assert!(!is_expected_absence(&err(
            "failed to resolve required FastRPC symbol: remote_handle64_open"
        )));
    }

    /// Every gated entry refuses under the kill switch, names the variable,
    /// and never reaches the driver (no `OpenSkel`); the same entry with an
    /// empty env does open one, so the assertion is not vacuous.
    #[test]
    fn kill_switch_gates_every_entry() {
        use super::device::probe_device_with;
        use super::sys::fake;
        let on = [("CERA_DISABLE_HEXAGON", "1")];
        let opened = || {
            fake::events()
                .iter()
                .any(|e| matches!(e, fake::Event::OpenSkel))
        };

        fake::reset();
        let Err(err) = HexagonContext::new_with(env(&on), || Ok(fake::driver())) else {
            panic!("context must refuse")
        };
        assert!(err.to_string().contains("CERA_DISABLE_HEXAGON"), "{err}");
        assert!(!opened());

        let Err(err) = probe_device_with(&fake::driver(), Some(HexagonArch::V75), env(&on)) else {
            panic!("probe_device must refuse")
        };
        assert!(err.to_string().contains("CERA_DISABLE_HEXAGON"), "{err}");
        assert!(!opened());

        let Err(err) = skels::probe_with(env(&on), || Ok(fake::driver())) else {
            panic!("probe must refuse")
        };
        assert!(err.to_string().contains("CERA_DISABLE_HEXAGON"), "{err}");
        assert!(!opened());

        // Control: the same entries with the switch off do reach the driver.
        assert!(probe_device_with(&fake::driver(), Some(HexagonArch::V75), env(&[])).is_ok());
        assert!(opened(), "control run must open a skel");
        fake::reset();
        assert!(skels::probe_with(env(&[]), || Ok(fake::driver())).is_ok());
        assert!(opened());
        fake::reset();
        assert!(HexagonContext::new_with(env(&[]), || Ok(fake::driver())).is_ok());
    }

    /// `HexagonContext::new` passes `env_lookup`, which must read the real
    /// process environment (a stub returning `None` would silently disable
    /// the kill switch in production while every injected test stayed green).
    #[test]
    fn production_env_lookup_reads_process_env() {
        assert_eq!(env_lookup("PATH"), std::env::var_os("PATH"));
        assert!(env_lookup("PATH").is_some(), "PATH is set under cargo test");
        assert_eq!(env_lookup("CERA_NO_SUCH_VAR_FOR_TEST"), None);
    }

    #[test]
    fn kill_switch_parse_rule() {
        assert!(!disabled_by(env(&[])));
        for off in ["", "0", "false", "FALSE", " 0 "] {
            assert!(
                !disabled_by(env(&[("CERA_DISABLE_HEXAGON", off)])),
                "{off:?}"
            );
        }
        for on in ["1", "true", "yes", "on"] {
            assert!(disabled_by(env(&[("CERA_DISABLE_HEXAGON", on)])), "{on:?}");
            assert!(disabled_by(env(&[("CERA_NO_HEXAGON", on)])), "{on:?}");
        }
        // Either variable alone suffices, even if the other says off.
        assert!(disabled_by(env(&[
            ("CERA_DISABLE_HEXAGON", "0"),
            ("CERA_NO_HEXAGON", "1"),
        ])));
    }
}

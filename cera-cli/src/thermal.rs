//! Android thermal-headroom sampling for benchmark hygiene.
//!
//! Wraps the NDK thermal API (`AThermal_getThermalHeadroom`, API 30+) so
//! `cera bench` can annotate each run with the device's thermal state. Headroom
//! is a forecast in `[0.0, 1.0+]`: `0.0` is cool, `1.0` is the throttling
//! threshold, and `> 1.0` means the SoC is already throttling. Without it,
//! on-device throughput numbers are unreadable — a sustained CPU benchmark heats
//! the SoC within seconds and later runs silently drop, so a raw tok/s figure
//! conflates the change under test with thermal drift.
//!
//! Resolved at runtime via `dlopen("libandroid.so")` (through `libc`, which the
//! workspace already pulls for these targets) rather than link-time, so the
//! binary neither raises its min-SDK nor fails to load on a device without
//! the symbol; [`ThermalMonitor::new`] just returns `None` there (and on every
//! non-Android target).

pub use imp::ThermalMonitor;

#[cfg(target_os = "android")]
mod imp {
    use libc::{RTLD_NOW, dlclose, dlopen, dlsym};
    use std::ffi::c_void;
    use std::os::raw::c_int;

    type AcquireFn = unsafe extern "C" fn() -> *mut c_void;
    type ReleaseFn = unsafe extern "C" fn(*mut c_void);
    type HeadroomFn = unsafe extern "C" fn(*mut c_void, c_int) -> f32;

    /// A live handle to the platform thermal service. Sampling is a cheap,
    /// non-blocking forecast query.
    pub struct ThermalMonitor {
        lib: *mut c_void,
        manager: *mut c_void,
        release: ReleaseFn,
        headroom: HeadroomFn,
    }

    impl ThermalMonitor {
        /// Acquire the thermal manager, or `None` if the platform lacks the API
        /// (API < 30) or the service is unavailable.
        pub fn new() -> Option<Self> {
            // SAFETY: standard dlopen/dlsym probing. The resolved symbols have
            // the AThermal ABI declared above; a `data*`→`fn*` transmute is the
            // documented dlsym pattern and sound on every Android ABI. Every
            // early return `dlclose`s the handle so a failed probe doesn't leak
            // it; the success path stores it and closes it in `Drop`.
            unsafe {
                let lib = dlopen(c"libandroid.so".as_ptr().cast(), RTLD_NOW);
                if lib.is_null() {
                    return None;
                }
                let acquire = dlsym(lib, c"AThermal_acquireManager".as_ptr().cast());
                let release = dlsym(lib, c"AThermal_releaseManager".as_ptr().cast());
                let headroom = dlsym(lib, c"AThermal_getThermalHeadroom".as_ptr().cast());
                if acquire.is_null() || release.is_null() || headroom.is_null() {
                    dlclose(lib);
                    return None;
                }
                let acquire: AcquireFn = std::mem::transmute(acquire);
                let release: ReleaseFn = std::mem::transmute(release);
                let headroom: HeadroomFn = std::mem::transmute(headroom);
                let manager = acquire();
                if manager.is_null() {
                    dlclose(lib);
                    return None;
                }
                Some(Self {
                    lib,
                    manager,
                    release,
                    headroom,
                })
            }
        }

        /// Thermal headroom forecast `forecast_secs` into the future (`0` = now).
        /// `0.0` cool → `1.0` throttling threshold; `None` if the value is NaN
        /// (not yet available — the service needs a moment after acquisition).
        pub fn headroom(&self, forecast_secs: i32) -> Option<f32> {
            // SAFETY: `manager` is a valid handle for this monitor's lifetime.
            let h = unsafe { (self.headroom)(self.manager, forecast_secs as c_int) };
            if h.is_nan() { None } else { Some(h) }
        }
    }

    impl Drop for ThermalMonitor {
        fn drop(&mut self) {
            // SAFETY: `manager` was acquired in `new` and is released exactly
            // once here; `lib` came from the matching `dlopen` and is closed
            // after the manager so the release fn stays valid through the call.
            unsafe {
                (self.release)(self.manager);
                dlclose(self.lib);
            }
        }
    }
}

#[cfg(not(target_os = "android"))]
mod imp {
    /// Non-Android stub: no thermal API, so sampling is always unavailable.
    pub struct ThermalMonitor;

    impl ThermalMonitor {
        pub fn new() -> Option<Self> {
            None
        }
        pub fn headroom(&self, _forecast_secs: i32) -> Option<f32> {
            None
        }
    }
}

//! Runtime loading of the embedded XDNA shim DLL and its C ABI.
//!
//! On Windows x86_64 MSVC, `build.rs` compiles `shim/cera_xdna_shim.cpp` into a
//! DLL (`has_xdna_shim`) that links `xrt_coreutil.dll`. `LoadLibraryW` fails on
//! machines without the AMD NPU driver; that failure (and non-Windows targets,
//! where `has_xdna_shim` is unset) is the backend's "unavailable" signal and is
//! reported as an error string, never a panic.

#[cfg(has_xdna_shim)]
use std::ffi::CString;
use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

#[cfg(has_xdna_shim)]
const SHIM_DLL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cera_xdna_shim.dll"));

/// ABI version the shim reports from `cera_xdna_shim_abi`; must match the
/// `#define CERA_XDNA_SHIM_ABI` in `shim/cera_xdna_shim.cpp`.
pub const EXPECTED_ABI: u32 = 2;

pub type ErrBuf = [c_char; 512];

fn err_string(buf: &ErrBuf) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Resolve one shim symbol as the function-pointer type the caller expects
/// (inferred from the `Shim` field it initializes).
#[cfg(has_xdna_shim)]
unsafe fn get<T: Copy>(module: isize, name: &str) -> Result<T, String> {
    let cname = CString::new(name).expect("static symbol name");
    let p = unsafe { GetProcAddress(module, cname.as_ptr()) };
    if p.is_null() {
        return Err(format!("xdna shim is missing symbol {name}"));
    }
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

/// Function table for the shim's C ABI. Handles are opaque pointers owned by
/// the caller, freed in reverse order of creation.
pub struct Shim {
    #[allow(dead_code)]
    pub(crate) module: isize,
    pub abi: unsafe extern "C" fn() -> u32,
    pub set_wakelock: unsafe extern "C" fn(i32),
    pub device_open: unsafe extern "C" fn(u32, *mut *mut c_void, *mut c_char, usize) -> i32,
    pub device_free: unsafe extern "C" fn(*mut c_void),
    pub device_bdf:
        unsafe extern "C" fn(*mut c_void, *mut c_char, usize, *mut c_char, usize) -> i32,
    pub kernel_load: unsafe extern "C" fn(
        *mut c_void,
        *const c_void,
        usize,
        *const u32,
        usize,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> i32,
    pub kernel_free: unsafe extern "C" fn(*mut c_void),
    pub bo_alloc:
        unsafe extern "C" fn(*mut c_void, usize, u32, *mut *mut c_void, *mut c_char, usize) -> i32,
    pub bo_map: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub bo_sync: unsafe extern "C" fn(*mut c_void, i32, *mut c_char, usize) -> i32,
    pub bo_sync_range:
        unsafe extern "C" fn(*mut c_void, i32, usize, usize, *mut c_char, usize) -> i32,
    pub bo_free: unsafe extern "C" fn(*mut c_void),
    pub run_create: unsafe extern "C" fn(
        *mut c_void,
        *const *mut c_void,
        usize,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> i32,
    pub run_start: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32,
    pub run_wait: unsafe extern "C" fn(*mut c_void, u32, *mut c_char, usize) -> i32,
    pub run_free: unsafe extern "C" fn(*mut c_void),
    pub runlist_create:
        unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut c_char, usize) -> i32,
    pub runlist_add: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_char, usize) -> i32,
    pub runlist_execute: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32,
    pub runlist_wait: unsafe extern "C" fn(*mut c_void, u32, *mut c_char, usize) -> i32,
    pub runlist_free: unsafe extern "C" fn(*mut c_void),
}

static SHIM: OnceLock<Result<Shim, String>> = OnceLock::new();

/// Load the shim once per process. On Windows x86_64 MSVC, the DLL bytes are
/// written to a per-process temp file because `LoadLibraryW` needs a path.
pub fn shim() -> Result<&'static Shim, String> {
    #[cfg(test)]
    if let Some(fake) = fake::active_shim() {
        return Ok(fake);
    }
    SHIM.get_or_init(load).as_ref().map_err(|e| e.clone())
}

#[cfg(has_xdna_shim)]
fn load() -> Result<Shim, String> {
    let dir = std::env::temp_dir().join(format!("cera_xdna_shim_{}.dll", std::process::id()));
    std::fs::write(&dir, SHIM_DLL)
        .map_err(|e| format!("cannot write xdna shim to {}: {e}", dir.display()))?;
    let wide: Vec<u16> = dir
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    if module == 0 {
        let code = unsafe { GetLastError() };
        return Err(format!(
            "cannot load the xdna shim ({}); the AMD NPU driver (xrt_coreutil.dll) is probably not installed [Win32 error {code}]",
            dir.display()
        ));
    }
    let shim = Shim {
        module,
        abi: unsafe { get(module, "cera_xdna_shim_abi")? },
        set_wakelock: unsafe { get(module, "cera_xdna_set_wakelock")? },
        device_open: unsafe { get(module, "cera_xdna_device_open")? },
        device_free: unsafe { get(module, "cera_xdna_device_free")? },
        device_bdf: unsafe { get(module, "cera_xdna_device_bdf")? },
        kernel_load: unsafe { get(module, "cera_xdna_kernel_load")? },
        kernel_free: unsafe { get(module, "cera_xdna_kernel_free")? },
        bo_alloc: unsafe { get(module, "cera_xdna_bo_alloc")? },
        bo_map: unsafe { get(module, "cera_xdna_bo_map")? },
        bo_sync: unsafe { get(module, "cera_xdna_bo_sync")? },
        bo_sync_range: unsafe { get(module, "cera_xdna_bo_sync_range")? },
        bo_free: unsafe { get(module, "cera_xdna_bo_free")? },
        run_create: unsafe { get(module, "cera_xdna_run_create")? },
        run_start: unsafe { get(module, "cera_xdna_run_start")? },
        run_wait: unsafe { get(module, "cera_xdna_run_wait")? },
        run_free: unsafe { get(module, "cera_xdna_run_free")? },
        runlist_create: unsafe { get(module, "cera_xdna_runlist_create")? },
        runlist_add: unsafe { get(module, "cera_xdna_runlist_add")? },
        runlist_execute: unsafe { get(module, "cera_xdna_runlist_execute")? },
        runlist_wait: unsafe { get(module, "cera_xdna_runlist_wait")? },
        runlist_free: unsafe { get(module, "cera_xdna_runlist_free")? },
    };
    let abi = unsafe { (shim.abi)() };
    if abi != EXPECTED_ABI {
        return Err(format!(
            "xdna shim ABI mismatch: shim reports {abi}, this build expects {EXPECTED_ABI}"
        ));
    }
    Ok(shim)
}

#[cfg(not(has_xdna_shim))]
fn load() -> Result<Shim, String> {
    Err("AMD XDNA NPU backend requires Windows x86_64 (MSVC)".to_string())
}

/// Check a shim return code, turning the error buffer into the message.
pub fn check(rc: i32, buf: &ErrBuf, what: &str) -> Result<i32, String> {
    if rc < 0 {
        Err(format!("{what}: {}", err_string(buf)))
    } else {
        Ok(rc)
    }
}

pub fn new_err_buf() -> ErrBuf {
    [0; 512]
}

#[cfg(has_xdna_shim)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(path: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const c_char) -> *mut c_void;
    fn GetLastError() -> u32;
}

/// Host-side fake XDNA shim for unit tests: exercises device/kernel/BO
/// lifecycles, wakelock refcounting, and error cleanup on any OS without an NPU
/// driver.
#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum Event {
        Wakelock(bool),
        DeviceOpen {
            index: u32,
            handle: usize,
        },
        DeviceFree(usize),
        KernelLoad(usize),
        KernelFree(usize),
        BoAlloc {
            bytes: usize,
            arg: u32,
            handle: usize,
        },
        BoMap(usize),
        BoFree(usize),
    }

    #[derive(Default)]
    pub(crate) struct Config {
        pub(crate) fail_open: bool,
        pub(crate) fail_kernel_load: bool,
        pub(crate) fail_bo_map: bool,
    }

    struct State {
        cfg: Config,
        next_handle: usize,
        events: Vec<Event>,
        backing: Vec<Vec<u8>>,
    }

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static STATE: RefCell<State> = RefCell::new(State {
            cfg: Config::default(),
            next_handle: 0x1000,
            events: Vec::new(),
            backing: Vec::new(),
        });
    }

    fn write_err(err: *mut c_char, err_len: usize, msg: &str) {
        if err.is_null() || err_len == 0 {
            return;
        }
        let bytes = msg.as_bytes();
        let n = bytes.len().min(err_len - 1);
        for (i, &b) in bytes.iter().enumerate().take(n) {
            unsafe { *err.add(i) = b as c_char };
        }
        unsafe { *err.add(n) = 0 };
    }

    unsafe extern "C" fn f_abi() -> u32 {
        EXPECTED_ABI
    }

    unsafe extern "C" fn f_wakelock(enable: i32) {
        STATE.with(|s| s.borrow_mut().events.push(Event::Wakelock(enable != 0)));
    }

    unsafe extern "C" fn f_device_open(
        index: u32,
        out: *mut *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            if st.cfg.fail_open {
                write_err(err, err_len, "synthetic open refusal");
                return -1;
            }
            let h = st.next_handle;
            st.next_handle += 1;
            st.events.push(Event::DeviceOpen { index, handle: h });
            unsafe { *out = h as *mut c_void };
            0
        })
    }

    unsafe extern "C" fn f_device_free(d: *mut c_void) {
        STATE.with(|s| s.borrow_mut().events.push(Event::DeviceFree(d as usize)));
    }

    unsafe extern "C" fn f_device_bdf(
        _d: *mut c_void,
        buf: *mut c_char,
        len: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        write_err(buf, len, "0000:c5:00.1");
        0
    }

    unsafe extern "C" fn f_kernel_load(
        _d: *mut c_void,
        _xclbin: *const c_void,
        _xclbin_len: usize,
        _instr: *const u32,
        n_instr: usize,
        out: *mut *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            if st.cfg.fail_kernel_load || n_instr == 0 {
                write_err(err, err_len, "synthetic kernel load refusal");
                return -1;
            }
            let h = st.next_handle;
            st.next_handle += 1;
            st.events.push(Event::KernelLoad(h));
            unsafe { *out = h as *mut c_void };
            0
        })
    }

    unsafe extern "C" fn f_kernel_free(k: *mut c_void) {
        STATE.with(|s| s.borrow_mut().events.push(Event::KernelFree(k as usize)));
    }

    unsafe extern "C" fn f_bo_alloc(
        _k: *mut c_void,
        bytes: usize,
        arg: u32,
        out: *mut *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        if arg < 3 {
            write_err(err, err_len, "data arguments start at index 3");
            return -1;
        }
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.backing.push(vec![0u8; bytes.max(1)]);
            let idx = st.backing.len(); // 1-based handle
            st.events.push(Event::BoAlloc {
                bytes,
                arg,
                handle: idx,
            });
            unsafe { *out = idx as *mut c_void };
            0
        })
    }

    unsafe extern "C" fn f_bo_map(b: *mut c_void) -> *mut c_void {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            let h = b as usize;
            st.events.push(Event::BoMap(h));
            if st.cfg.fail_bo_map {
                return std::ptr::null_mut();
            }
            st.backing[h - 1].as_mut_ptr() as *mut c_void
        })
    }

    unsafe extern "C" fn f_bo_sync(
        _b: *mut c_void,
        _to_dev: i32,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn f_bo_sync_range(
        _b: *mut c_void,
        _to_dev: i32,
        _size: usize,
        _off: usize,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn f_bo_free(b: *mut c_void) {
        STATE.with(|s| s.borrow_mut().events.push(Event::BoFree(b as usize)));
    }

    unsafe extern "C" fn f_run_create(
        _k: *mut c_void,
        _bos: *const *mut c_void,
        _n_bos: usize,
        out: *mut *mut c_void,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            let h = st.next_handle;
            st.next_handle += 1;
            unsafe { *out = h as *mut c_void };
            0
        })
    }

    unsafe extern "C" fn f_run_start(_r: *mut c_void, _err: *mut c_char, _err_len: usize) -> i32 {
        0
    }

    unsafe extern "C" fn f_run_wait(
        _r: *mut c_void,
        _timeout_ms: u32,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        4
    }

    unsafe extern "C" fn f_run_free(_r: *mut c_void) {}

    unsafe extern "C" fn f_runlist_create(
        _k: *mut c_void,
        out: *mut *mut c_void,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            let h = st.next_handle;
            st.next_handle += 1;
            unsafe { *out = h as *mut c_void };
            0
        })
    }

    unsafe extern "C" fn f_runlist_add(
        _l: *mut c_void,
        _r: *mut c_void,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn f_runlist_execute(
        _l: *mut c_void,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn f_runlist_wait(
        _l: *mut c_void,
        _timeout_ms: u32,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn f_runlist_free(_l: *mut c_void) {}

    static FAKE_SHIM: Shim = Shim {
        module: 1,
        abi: f_abi,
        set_wakelock: f_wakelock,
        device_open: f_device_open,
        device_free: f_device_free,
        device_bdf: f_device_bdf,
        kernel_load: f_kernel_load,
        kernel_free: f_kernel_free,
        bo_alloc: f_bo_alloc,
        bo_map: f_bo_map,
        bo_sync: f_bo_sync,
        bo_sync_range: f_bo_sync_range,
        bo_free: f_bo_free,
        run_create: f_run_create,
        run_start: f_run_start,
        run_wait: f_run_wait,
        run_free: f_run_free,
        runlist_create: f_runlist_create,
        runlist_add: f_runlist_add,
        runlist_execute: f_runlist_execute,
        runlist_wait: f_runlist_wait,
        runlist_free: f_runlist_free,
    };

    pub(crate) fn active_shim() -> Option<&'static Shim> {
        ACTIVE.with(|a| a.get().then_some(&FAKE_SHIM))
    }

    pub(crate) fn with_shim<R>(cfg: Config, f: impl FnOnce() -> R) -> (R, Vec<Event>) {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.cfg = cfg;
            st.next_handle = 0x1000;
            st.events.clear();
            st.backing.clear();
        });
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                ACTIVE.with(|a| a.set(false));
            }
        }
        ACTIVE.with(|a| a.set(true));
        let _guard = Guard;
        let out = f();
        let events = STATE.with(|s| std::mem::take(&mut s.borrow_mut().events));
        (out, events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shim_loads_with_expected_abi() {
        // Driverless or non-Windows machines skip: loading the shim is the availability probe.
        let Ok(shim) = shim() else { return };
        assert_eq!(unsafe { (shim.abi)() }, EXPECTED_ABI);
    }
}

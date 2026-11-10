//! Runtime loading of the embedded XDNA shim DLL and its C ABI.
//!
//! The shim DLL itself links `xrt_coreutil.dll`, so `LoadLibraryW` fails on
//! machines without the AMD NPU driver; that failure is the backend's
//! "unavailable" signal and is reported as an error string, never a panic.

use std::ffi::{CString, c_char, c_void};
use std::sync::OnceLock;

const SHIM_DLL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cera_xdna_shim.dll"));

/// ABI version the shim reports from `cera_xdna_shim_abi`; must match the
/// `#define CERA_XDNA_SHIM_ABI` in `shim/cera_xdna_shim.cpp`.
pub const EXPECTED_ABI: u32 = 1;

type ErrBuf = [c_char; 512];

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
    module: isize,
    pub abi: unsafe extern "C" fn() -> u32,
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

/// Load the shim once per process. The DLL bytes are written to a per-process
/// temp file because `LoadLibraryW` needs a path.
pub fn shim() -> Result<&'static Shim, String> {
    SHIM.get_or_init(|| unsafe { load() })
        .as_ref()
        .map_err(|e| e.clone())
}

unsafe fn load() -> Result<Shim, String> {
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

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(path: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const c_char) -> *mut c_void;
    fn GetLastError() -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shim_loads_with_expected_abi() {
        // Driverless machines skip: loading the shim is the availability probe.
        let Ok(shim) = shim() else { return };
        assert_eq!(unsafe { (shim.abi)() }, EXPECTED_ABI);
    }
}

//! Runtime loading of the embedded XDNA shim DLL and its C ABI.
//!
//! On Windows x86_64 MSVC, `build.rs` compiles `shim/cera_xdna_shim.cpp` into a
//! DLL (`has_xdna_shim`) that links `xrt_coreutil.dll`. `LoadLibraryExW` fails
//! on machines without the AMD NPU driver; that failure (and non-Windows targets,
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
/// staged into a user-private directory with a CSPRNG nonce and exclusive write
/// sharing because `LoadLibraryExW` needs a path on disk.
pub fn shim() -> Result<&'static Shim, String> {
    #[cfg(test)]
    if let Some(fake) = fake::active_shim() {
        return Ok(fake);
    }
    SHIM.get_or_init(load).as_ref().map_err(|e| e.clone())
}

#[cfg(any(has_xdna_shim, test))]
fn reject_reparse_point(path: &std::path::Path) -> Result<(), String> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| format!("cannot inspect xdna shim directory {}: {e}", path.display()))?;
    if meta.file_type().is_symlink() {
        return Err(format!(
            "refusing symlinked xdna shim directory {}",
            path.display()
        ));
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "refusing reparse-point xdna shim directory {}",
                path.display()
            ));
        }
    }
    Ok(())
}

#[cfg(any(has_xdna_shim, test))]
fn ensure_verified_dir(path: &std::path::Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => match std::fs::create_dir(path) {
            Ok(()) => {}
            Err(ce) if ce.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(ce) => {
                return Err(format!(
                    "cannot create xdna shim directory {}: {ce}",
                    path.display()
                ));
            }
        },
        Err(e) => {
            return Err(format!(
                "cannot inspect xdna shim directory {}: {e}",
                path.display()
            ));
        }
    }
    reject_reparse_point(path)?;
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| format!("cannot inspect xdna shim directory {}: {e}", path.display()))?;
    if !meta.is_dir() {
        return Err(format!(
            "xdna shim staging path is not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(any(has_xdna_shim, test))]
fn reject_reparse_file_handle(file: &std::fs::File, path: &std::path::Path) -> Result<(), String> {
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot inspect xdna shim file {}: {e}", path.display()))?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "refusing reparse-point xdna shim file {}",
                path.display()
            ));
        }
    }
    if !meta.is_file() {
        return Err(format!(
            "refusing non-regular xdna shim file {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(any(has_xdna_shim, test))]
fn configure_nofollow(opts: &mut std::fs::OpenOptions) {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        opts.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = opts;
    }
}

#[cfg(any(has_xdna_shim, test))]
fn resolve_shim_stage_dir_with(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<std::path::PathBuf, String> {
    let abs_local = if let Some(local) = lookup("LOCALAPPDATA").filter(|v| !v.is_empty()) {
        let base = std::path::PathBuf::from(local);
        if !base.is_absolute() {
            return Err(format!(
                "refusing non-absolute LOCALAPPDATA path for xdna shim staging: {}",
                base.display()
            ));
        }
        let abs_base = std::path::absolute(&base)
            .map_err(|e| format!("cannot normalize xdna shim base {}: {e}", base.display()))?;
        ensure_verified_dir(&abs_base)?;
        abs_base
    } else if let Some(profile) = lookup("USERPROFILE").filter(|v| !v.is_empty()) {
        let profile_path = std::path::PathBuf::from(profile);
        if !profile_path.is_absolute() {
            return Err(format!(
                "refusing non-absolute USERPROFILE path for xdna shim staging: {}",
                profile_path.display()
            ));
        }
        let abs_profile = std::path::absolute(&profile_path).map_err(|e| {
            format!(
                "cannot normalize xdna shim profile {}: {e}",
                profile_path.display()
            )
        })?;
        ensure_verified_dir(&abs_profile)?;
        let appdata = abs_profile.join("AppData");
        ensure_verified_dir(&appdata)?;
        let local = appdata.join("Local");
        ensure_verified_dir(&local)?;
        local
    } else {
        return Err(
            "cannot resolve user-private xdna shim directory: neither LOCALAPPDATA nor USERPROFILE is set"
                .to_string(),
        );
    };
    let cera_dir = abs_local.join("cera");
    ensure_verified_dir(&cera_dir)?;
    let dir = cera_dir.join("xdna");
    ensure_verified_dir(&dir)?;
    Ok(dir)
}

#[cfg(has_xdna_shim)]
fn shim_stage_dir() -> Result<std::path::PathBuf, String> {
    resolve_shim_stage_dir_with(|k| std::env::var_os(k))
}

#[cfg(any(has_xdna_shim, test))]
fn acquire_stage_lock(dir: &std::path::Path) -> Result<std::fs::File, String> {
    let lock_path = dir.join("cera_xdna_shim.lock");
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true).write(true).create(true).truncate(false);
    configure_nofollow(&mut opts);
    let lock_file = opts
        .open(&lock_path)
        .map_err(|e| format!("cannot open xdna shim lock {}: {e}", lock_path.display()))?;
    reject_reparse_file_handle(&lock_file, &lock_path)?;
    lock_file
        .lock()
        .map_err(|e| format!("cannot lock xdna shim directory {}: {e}", dir.display()))?;
    Ok(lock_file)
}

#[cfg(any(has_xdna_shim, test))]
fn sweep_stale_shims(dir: &std::path::Path) {
    // Best-effort sweep of prior exited processes' staged DLLs; any DLL mapped
    // by a live process has its image section locked by Windows and fails with
    // a sharing violation, which is ignored. Callers must hold `acquire_stage_lock`
    // so an in-progress `stage_shim_dll_in` in another process cannot be swept
    // between its write close and its read-lock reopen.
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str()
                && name.starts_with("cera_xdna_shim_")
                && name.ends_with(".dll")
            {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

#[cfg(any(has_xdna_shim, test))]
fn stage_shim_dll_in(
    dir: &std::path::Path,
    bytes: &[u8],
) -> Result<(std::path::PathBuf, std::fs::File), String> {
    use std::fs::OpenOptions;
    use std::io::{Read, Write};

    let pid = std::process::id();
    for _ in 0..16 {
        let nonce: u128 = rand::random();
        let path = dir.join(format!("cera_xdna_shim_{pid}_{nonce:032x}.dll"));
        let mut create_opts = OpenOptions::new();
        create_opts.write(true).create_new(true);
        configure_nofollow(&mut create_opts);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Exclusive access (`share_mode(0)`) while writing initial bytes.
            create_opts.share_mode(0);
        }
        match create_opts.open(&path) {
            Ok(mut file) => {
                if let Err(e) = reject_reparse_file_handle(&file, &path) {
                    drop(file);
                    return Err(e);
                }
                if let Err(e) = file.write_all(bytes).and_then(|()| file.sync_all()) {
                    drop(file);
                    let _ = std::fs::remove_file(&path);
                    return Err(format!("cannot write xdna shim to {}: {e}", path.display()));
                }
                drop(file);

                // Re-open read-only with `FILE_SHARE_READ` (no write/delete
                // sharing) and `FILE_FLAG_OPEN_REPARSE_POINT` / `O_NOFOLLOW`, then
                // re-verify handle attributes and payload bytes while holding that
                // read lock. `LoadLibraryExW` fails with `ERROR_SHARING_VIOLATION`
                // if an open handle still holds `GENERIC_WRITE`, whereas a
                // read-only lock lets `LoadLibraryExW` map `SEC_IMAGE` while
                // blocking any concurrent write, rename, or symlink swap (CWE-377).
                let mut lock_opts = OpenOptions::new();
                lock_opts.read(true);
                configure_nofollow(&mut lock_opts);
                #[cfg(windows)]
                {
                    use std::os::windows::fs::OpenOptionsExt;
                    lock_opts.share_mode(0x0000_0001); // FILE_SHARE_READ
                }
                let mut read_lock = match lock_opts.open(&path) {
                    Ok(f) => f,
                    Err(e) => {
                        return Err(format!(
                            "cannot lock staged xdna shim {}: {e}",
                            path.display()
                        ));
                    }
                };
                if let Err(e) = reject_reparse_file_handle(&read_lock, &path) {
                    drop(read_lock);
                    return Err(e);
                }
                let mut verify = Vec::with_capacity(bytes.len());
                if let Err(e) = read_lock.read_to_end(&mut verify) {
                    drop(read_lock);
                    let _ = std::fs::remove_file(&path);
                    return Err(format!(
                        "cannot verify staged xdna shim {}: {e}",
                        path.display()
                    ));
                }
                if verify != bytes {
                    drop(read_lock);
                    let _ = std::fs::remove_file(&path);
                    return Err(format!(
                        "staged xdna shim {} failed integrity verification",
                        path.display()
                    ));
                }
                return Ok((path, read_lock));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!("cannot create xdna shim in {}: {e}", dir.display()));
            }
        }
    }
    Err(format!(
        "cannot create unique xdna shim file in {} after 16 attempts",
        dir.display()
    ))
}

#[cfg(has_xdna_shim)]
unsafe fn load_symbols(module: isize) -> Result<Shim, String> {
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

#[cfg(has_xdna_shim)]
fn load() -> Result<Shim, String> {
    use std::os::windows::ffi::OsStrExt;

    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

    let stage_dir = shim_stage_dir()?;
    let stage_lock = acquire_stage_lock(&stage_dir)?;
    sweep_stale_shims(&stage_dir);
    let (dll_path, lock_file) = stage_shim_dll_in(&stage_dir, SHIM_DLL)?;
    let wide: Vec<u16> = dll_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // Hold both `stage_lock` and `lock_file` across `LoadLibraryExW` so no
    // concurrent sweeper or writer can touch the staged DLL before Windows
    // maps its image section, and restrict dependent DLL lookup
    // (`xrt_coreutil.dll`, MSVC runtime) strictly to `%SystemRoot%\System32`.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), 0, LOAD_LIBRARY_SEARCH_SYSTEM32) };
    let load_err = if module == 0 {
        unsafe { GetLastError() }
    } else {
        0
    };
    drop(lock_file);
    if module == 0 {
        let _ = std::fs::remove_file(&dll_path);
        drop(stage_lock);
        return Err(format!(
            "cannot load the xdna shim ({}); the AMD NPU driver (xrt_coreutil.dll) is probably not installed [Win32 error {load_err}]",
            dll_path.display()
        ));
    }
    drop(stage_lock);
    match unsafe { load_symbols(module) } {
        Ok(shim) => Ok(shim),
        Err(e) => {
            unsafe {
                let _ = FreeLibrary(module);
            }
            let _ = std::fs::remove_file(&dll_path);
            Err(e)
        }
    }
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
    fn LoadLibraryExW(path: *const u16, file: isize, flags: u32) -> isize;
    fn FreeLibrary(module: isize) -> i32;
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
        BoSyncRange {
            handle: usize,
            to_device: bool,
            size: usize,
            offset: usize,
        },
        BoFree(usize),
        RunListExecute(usize),
        RunListWait {
            handle: usize,
            timeout_ms: u32,
        },
    }

    #[derive(Default)]
    pub(crate) struct Config {
        pub(crate) fail_open: bool,
        pub(crate) fail_kernel_load: bool,
        pub(crate) fail_bo_map: bool,
        pub(crate) run_wait_state: Option<i32>,
        pub(crate) fail_runlist_wait: bool,
    }

    struct State {
        cfg: Config,
        next_handle: usize,
        next_bo_handle: usize,
        events: Vec<Event>,
        backing: std::collections::HashMap<usize, Box<[u8]>>,
    }

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static STATE: RefCell<State> = RefCell::new(State {
            cfg: Config::default(),
            next_handle: 0x1000,
            next_bo_handle: 1,
            events: Vec::new(),
            backing: std::collections::HashMap::new(),
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
        xclbin: *const c_void,
        xclbin_len: usize,
        instr: *const u32,
        n_instr: usize,
        out: *mut *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            if st.cfg.fail_kernel_load
                || xclbin.is_null()
                || xclbin_len == 0
                || instr.is_null()
                || n_instr == 0
                || n_instr > (u32::MAX as usize) / std::mem::size_of::<u32>()
                || out.is_null()
            {
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
        if bytes == 0 || bytes > isize::MAX as usize {
            write_err(err, err_len, "buffer size out of range");
            return -1;
        }
        if arg < 3 {
            write_err(err, err_len, "data arguments start at index 3");
            return -1;
        }
        if arg > i32::MAX as u32 {
            write_err(err, err_len, "argument index exceeds INT_MAX");
            return -1;
        }
        if out.is_null() {
            write_err(err, err_len, "null output handle");
            return -1;
        }
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            let idx = st.next_bo_handle;
            st.next_bo_handle += 1;
            st.backing.insert(idx, vec![0u8; bytes].into_boxed_slice());
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
            st.backing
                .get_mut(&h)
                .map_or(std::ptr::null_mut(), |buf| buf.as_mut_ptr() as *mut c_void)
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
        b: *mut c_void,
        to_dev: i32,
        size: usize,
        off: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            let h = b as usize;
            let Some(buf) = st.backing.get(&h) else {
                write_err(err, err_len, "invalid buffer handle");
                return -1;
            };
            let bo_len = buf.len();
            if size == 0 || off > bo_len || size > bo_len - off {
                write_err(err, err_len, "buffer sync range invalid");
                return -1;
            }
            st.events.push(Event::BoSyncRange {
                handle: h,
                to_device: to_dev != 0,
                size,
                offset: off,
            });
            0
        })
    }

    unsafe extern "C" fn f_bo_free(b: *mut c_void) {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            let h = b as usize;
            st.backing.remove(&h);
            st.events.push(Event::BoFree(h));
        });
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
        STATE.with(|s| s.borrow().cfg.run_wait_state.unwrap_or(4))
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
        l: *mut c_void,
        _err: *mut c_char,
        _err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            s.borrow_mut()
                .events
                .push(Event::RunListExecute(l as usize))
        });
        0
    }

    unsafe extern "C" fn f_runlist_wait(
        l: *mut c_void,
        timeout_ms: u32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32 {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.events.push(Event::RunListWait {
                handle: l as usize,
                timeout_ms,
            });
            if st.cfg.fail_runlist_wait {
                write_err(err, err_len, "synthetic runlist wait timeout");
                -1
            } else {
                0
            }
        })
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
            if st.backing.is_empty() {
                st.next_bo_handle = 1;
            }
            st.events.clear();
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

    pub(crate) fn live_bo_count() -> usize {
        STATE.with(|s| s.borrow().backing.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn shim_loads_with_expected_abi() {
        #[cfg(has_xdna_shim)]
        if std::env::var_os("SystemRoot")
            .map(|r| {
                std::path::PathBuf::from(r)
                    .join("System32")
                    .join("xrt_coreutil.dll")
            })
            .is_some_and(|p| p.exists())
        {
            let shim = shim().expect("shim must load when xrt_coreutil.dll is installed");
            assert_eq!(unsafe { (shim.abi)() }, EXPECTED_ABI);
            return;
        }
        // Driverless or non-Windows machines skip: loading the shim is the availability probe.
        let Ok(shim) = shim() else { return };
        assert_eq!(unsafe { (shim.abi)() }, EXPECTED_ABI);
    }

    #[test]
    fn stage_shim_dll_creates_unique_locked_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let payload = b"cera-xdna-shim-test-payload";
        let (p1, f1) = stage_shim_dll_in(tmp.path(), payload).expect("stage 1");
        let (p2, f2) = stage_shim_dll_in(tmp.path(), payload).expect("stage 2");
        assert_ne!(p1, p2, "staged filenames must carry fresh CSPRNG nonces");
        assert_eq!(std::fs::read(&p1).expect("read p1"), payload);
        #[cfg(windows)]
        {
            // While `f1` is open with `FILE_SHARE_READ` only, any concurrent
            // write or delete attempt must fail with a sharing violation.
            assert!(
                std::fs::OpenOptions::new().write(true).open(&p1).is_err(),
                "staged DLL must reject concurrent write handles while locked"
            );
            assert!(
                std::fs::remove_file(&p1).is_err(),
                "staged DLL must reject unlink/swap while locked"
            );
        }
        drop(f1);
        drop(f2);
        std::fs::remove_file(&p1).expect("cleanup p1 after unlock");
        std::fs::remove_file(&p2).expect("cleanup p2 after unlock");
    }

    #[test]
    fn resolve_shim_stage_dir_rejects_missing_and_relative_env_and_accepts_userprofile() {
        let err = resolve_shim_stage_dir_with(|_| None).expect_err("missing env must fail");
        assert!(
            err.contains("neither LOCALAPPDATA nor USERPROFILE"),
            "{err}"
        );

        let err_rel = resolve_shim_stage_dir_with(|k| {
            (k == "LOCALAPPDATA").then(|| OsString::from("relative-appdata"))
        })
        .expect_err("relative LOCALAPPDATA must fail");
        assert!(err_rel.contains("non-absolute LOCALAPPDATA"), "{err_rel}");

        let err_rel_profile = resolve_shim_stage_dir_with(|k| {
            (k == "USERPROFILE").then(|| OsString::from("relative-profile"))
        })
        .expect_err("relative USERPROFILE must fail");
        assert!(
            err_rel_profile.contains("non-absolute USERPROFILE"),
            "{err_rel_profile}"
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        let resolved = resolve_shim_stage_dir_with(|k| {
            (k == "USERPROFILE").then(|| tmp.path().as_os_str().to_os_string())
        })
        .expect("USERPROFILE fallback must succeed");
        assert_eq!(
            resolved,
            tmp.path()
                .join("AppData")
                .join("Local")
                .join("cera")
                .join("xdna")
        );
        assert!(resolved.is_dir());
    }

    #[test]
    fn resolve_shim_stage_dir_refuses_reparse_component_before_creating_child() {
        let base_tmp = tempfile::tempdir().expect("base tempdir");
        let target_tmp = tempfile::tempdir().expect("target tempdir");
        let link_cera = base_tmp.path().join("cera");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target_tmp.path(), &link_cera).expect("unix symlink");
        }
        #[cfg(windows)]
        {
            let status = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(&link_cera)
                .arg(target_tmp.path())
                .output()
                .expect("spawn mklink /J");
            assert!(
                status.status.success(),
                "mklink /J failed: {}",
                String::from_utf8_lossy(&status.stderr)
            );
        }

        let err = resolve_shim_stage_dir_with(|k| {
            (k == "LOCALAPPDATA").then(|| base_tmp.path().as_os_str().to_os_string())
        })
        .expect_err("symlink or junction cera component must fail");
        assert!(
            err.contains("symlink") || err.contains("reparse-point"),
            "unexpected error message: {err}"
        );
        assert!(
            !target_tmp.path().join("xdna").exists(),
            "staging directory resolution must not create `xdna` inside link target"
        );
    }

    #[test]
    fn stage_lock_contends_and_refuses_reparse_lockfile() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let stale = tmp.path().join("cera_xdna_shim_999999_deadbeef.dll");
        std::fs::write(&stale, b"stale").expect("write stale");
        let guard = acquire_stage_lock(tmp.path()).expect("acquire stage lock");
        let contender = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp.path().join("cera_xdna_shim.lock"))
            .expect("open contender handle");
        assert!(
            contender.try_lock().is_err(),
            "second handle must observe active exclusive lock on cera_xdna_shim.lock"
        );
        drop(contender);
        sweep_stale_shims(tmp.path());
        assert!(!stale.exists(), "unlocked stale DLL must be swept");
        let (active_path, active_lock) =
            stage_shim_dll_in(tmp.path(), b"live-dll").expect("stage live");
        drop(guard);
        assert!(active_path.exists());
        drop(active_lock);

        // Replace lockfile with a directory/junction reparse point and verify refusal.
        let lock_tmp = tempfile::tempdir().expect("lock tempdir");
        let lock_target = tempfile::tempdir().expect("lock target");
        let lock_link = lock_tmp.path().join("cera_xdna_shim.lock");
        #[cfg(unix)]
        {
            let target_file = lock_target.path().join("target.lock");
            std::fs::write(&target_file, b"").expect("target file");
            std::os::unix::fs::symlink(&target_file, &lock_link).expect("unix file symlink");
        }
        #[cfg(windows)]
        {
            let status = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(&lock_link)
                .arg(lock_target.path())
                .output()
                .expect("spawn mklink /J");
            assert!(status.status.success());
        }
        assert!(
            acquire_stage_lock(lock_tmp.path()).is_err(),
            "acquire_stage_lock must refuse reparse-point or symlinked lock path"
        );
    }

    #[test]
    fn fake_shim_bo_backing_survives_scope_and_frees_on_drop() {
        let (mut bo, _) = fake::with_shim(fake::Config::default(), || {
            let dev = crate::backend::xdna::device::XdnaDevice::open(0).expect("open");
            let kern = dev
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("load");
            let mut bo = kern.alloc_bo(32, 3).expect("alloc");
            bo.as_mut_slice()[0] = 0x5A;
            bo
        });
        assert_eq!(fake::live_bo_count(), 1);
        assert_eq!(bo.as_slice()[0], 0x5A);
        bo.as_mut_slice()[31] = 0xA5;
        assert_eq!(bo.as_slice()[31], 0xA5);
        drop(bo);
        assert_eq!(fake::live_bo_count(), 0);
    }
}

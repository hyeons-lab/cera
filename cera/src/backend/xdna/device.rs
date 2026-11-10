//! Device and kernel handles over the XRT shim.

use std::ffi::c_void;

use super::sys::{check, new_err_buf, shim};

/// An opened XRT device (the NPU at index 0).
pub struct XdnaDevice {
    handle: *mut c_void,
}

/// A loaded `MLIR_AIE` kernel: hardware context plus uploaded instructions.
pub struct XdnaKernel {
    handle: *mut c_void,
}

impl XdnaDevice {
    pub fn open(index: u32) -> Result<Self, String> {
        let shim = shim()?;
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe { (shim.device_open)(index, &mut handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "open XRT device")?;
        Ok(Self { handle })
    }

    /// The device's BDF-style identifier, for logs.
    pub fn bdf(&self) -> Result<String, String> {
        let shim = shim()?;
        let mut buf = new_err_buf();
        let mut err = new_err_buf();
        let rc = unsafe {
            (shim.device_bdf)(
                self.handle,
                buf.as_mut_ptr(),
                buf.len(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "read device identifier")?;
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Load an in-memory `.xclbin` with its IRON instruction words. This is
    /// also the support gate: an xclbin the driver rejects (wrong silicon,
    /// wrong XRT) fails here with the driver's reason.
    pub fn load_kernel(&self, xclbin: &[u8], instr: &[u32]) -> Result<XdnaKernel, String> {
        let shim = shim()?;
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (shim.kernel_load)(
                self.handle,
                xclbin.as_ptr() as *const c_void,
                xclbin.len(),
                instr.as_ptr(),
                instr.len(),
                &mut handle,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "load xclbin")?;
        Ok(XdnaKernel { handle })
    }
}

/// A host-only buffer bound to one kernel argument.
pub struct XdnaBo {
    handle: *mut c_void,
    ptr: *mut c_void,
    len: usize,
}

/// One kernel invocation with its arguments bound.
pub struct XdnaRun {
    handle: *mut c_void,
}

/// A set of runs executed in submission order on one hardware context.
pub struct XdnaRunList {
    handle: *mut c_void,
}

impl XdnaKernel {
    /// Allocate a host-only buffer for data argument `arg` (3 or above).
    pub fn alloc_bo(&self, bytes: usize, arg: u32) -> Result<XdnaBo, String> {
        let shim = shim()?;
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (shim.bo_alloc)(
                self.handle,
                bytes,
                arg,
                &mut handle,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "allocate buffer")?;
        let ptr = unsafe { (shim.bo_map)(handle) };
        if ptr.is_null() {
            unsafe { (shim.bo_free)(handle) }
            return Err("allocate buffer: mapping failed".to_string());
        }
        Ok(XdnaBo {
            handle,
            ptr,
            len: bytes,
        })
    }

    /// Create a run with `bos[i]` bound to argument `3 + i`.
    pub fn create_run(&self, bos: &[&XdnaBo]) -> Result<XdnaRun, String> {
        let shim = shim()?;
        let raw: Vec<*mut c_void> = bos.iter().map(|b| b.handle).collect();
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (shim.run_create)(
                self.handle,
                raw.as_ptr(),
                raw.len(),
                &mut handle,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "create run")?;
        Ok(XdnaRun { handle })
    }

    /// Create an empty runlist on this kernel's hardware context.
    pub fn create_runlist(&self) -> Result<XdnaRunList, String> {
        let shim = shim()?;
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc =
            unsafe { (shim.runlist_create)(self.handle, &mut handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "create runlist")?;
        Ok(XdnaRunList { handle })
    }
}

impl XdnaBo {
    /// The mapped host memory.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    /// The mapped host memory, read-only.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// Sync the buffer to (`to_device = true`) or from the device.
    pub fn sync(&self, to_device: bool) -> Result<(), String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc =
            unsafe { (shim.bo_sync)(self.handle, to_device as i32, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "sync buffer")?;
        Ok(())
    }

    /// Sync `size` bytes starting at `offset` to (`to_device = true`) or from the device.
    pub fn sync_range(&self, to_device: bool, size: usize, offset: usize) -> Result<(), String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc = unsafe {
            (shim.bo_sync_range)(
                self.handle,
                to_device as i32,
                size,
                offset,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "sync buffer range")?;
        Ok(())
    }
}

impl Drop for XdnaBo {
    fn drop(&mut self) {
        if let Ok(shim) = shim() {
            unsafe { (shim.bo_free)(self.handle) }
        }
    }
}

impl XdnaRun {
    /// Start the run asynchronously.
    pub fn start(&self) -> Result<(), String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc = unsafe { (shim.run_start)(self.handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "start run")?;
        Ok(())
    }

    /// Wait for completion; returns the `ert_cmd_state` value (4 is
    /// `ERT_CMD_STATE_COMPLETED`). `timeout_ms` of 0 waits without a limit.
    pub fn wait(&self, timeout_ms: u32) -> Result<i32, String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc = unsafe { (shim.run_wait)(self.handle, timeout_ms, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "wait for run")
    }
}

impl Drop for XdnaRun {
    fn drop(&mut self) {
        if let Ok(shim) = shim() {
            unsafe { (shim.run_free)(self.handle) }
        }
    }
}

impl XdnaRunList {
    /// Append a run; runs execute in the order added.
    pub fn add(&mut self, run: &XdnaRun) -> Result<(), String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc =
            unsafe { (shim.runlist_add)(self.handle, run.handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "add run to runlist")?;
        Ok(())
    }

    /// Execute the whole list.
    pub fn execute(&self) -> Result<(), String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc = unsafe { (shim.runlist_execute)(self.handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "execute runlist")?;
        Ok(())
    }

    /// Wait for the whole list; `timeout_ms` of 0 waits without a limit.
    pub fn wait(&self, timeout_ms: u32) -> Result<(), String> {
        let shim = shim()?;
        let mut err = new_err_buf();
        let rc =
            unsafe { (shim.runlist_wait)(self.handle, timeout_ms, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "wait for runlist")?;
        Ok(())
    }
}

impl Drop for XdnaRunList {
    fn drop(&mut self) {
        if let Ok(shim) = shim() {
            unsafe { (shim.runlist_free)(self.handle) }
        }
    }
}

impl Drop for XdnaDevice {
    fn drop(&mut self) {
        if let Ok(shim) = shim() {
            unsafe { (shim.device_free)(self.handle) }
        }
    }
}

impl Drop for XdnaKernel {
    fn drop(&mut self) {
        if let Ok(shim) = shim() {
            unsafe { (shim.kernel_free)(self.handle) }
        }
    }
}

/// Open the NPU and load `xclbin`, returning the device on success. Any
/// failure (no driver, no device, rejected xclbin) is the reason string.
pub fn probe(xclbin: &[u8], instr: &[u32]) -> Result<XdnaDevice, String> {
    let device = XdnaDevice::open(0)?;
    let _kernel = device.load_kernel(xclbin, instr)?;
    Ok(device)
}

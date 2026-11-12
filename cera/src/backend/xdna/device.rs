//! Device and kernel handles over the XRT shim.

use std::ffi::c_void;
use std::sync::Mutex;

use super::sys::{Shim, check, new_err_buf, shim};

/// Process-wide count of live [`PowerVote`] holders. The Windows power request
/// (`cera_xdna_set_wakelock`) is process-wide, so the vote is acquired on the
/// 0 -> 1 transition and released on 1 -> 0; dropping a short-lived probe
/// device must not cancel the wakelock another live [`XdnaDevice`] still needs,
/// and a failed [`XdnaDevice::open`] must release its share instead of leaking.
struct VoteCounter(Mutex<usize>);

impl VoteCounter {
    const fn new() -> Self {
        Self(Mutex::new(0))
    }

    fn enter(&self, on_first: impl FnOnce()) {
        let mut n = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if *n == 0 {
            on_first();
        }
        *n += 1;
    }

    fn leave(&self, on_last: impl FnOnce()) {
        let mut n = self.0.lock().unwrap_or_else(|p| p.into_inner());
        *n = n.saturating_sub(1);
        if *n == 0 {
            on_last();
        }
    }
}

static POWER_VOTES: VoteCounter = VoteCounter::new();

struct PowerVote {
    shim: &'static Shim,
    counter: &'static VoteCounter,
}

impl PowerVote {
    fn acquire_in(shim: &'static Shim, counter: &'static VoteCounter) -> Self {
        counter.enter(|| unsafe { (shim.set_wakelock)(1) });
        Self { shim, counter }
    }
}

impl Drop for PowerVote {
    fn drop(&mut self) {
        let shim = self.shim;
        self.counter.leave(|| unsafe { (shim.set_wakelock)(0) });
    }
}

/// An opened XRT device (the NPU at index 0).
///
/// Field declaration order matters: Rust drops `handle` via [`Drop`] before
/// dropping `_power_vote`, so the last device closes its XRT handle before
/// clearing the process wakelock.
pub struct XdnaDevice {
    handle: *mut c_void,
    shim: &'static Shim,
    _power_vote: PowerVote,
}

/// A loaded `MLIR_AIE` kernel: hardware context plus uploaded instructions.
pub struct XdnaKernel {
    handle: *mut c_void,
    shim: &'static Shim,
}

impl XdnaDevice {
    pub fn open(index: u32) -> Result<Self, String> {
        Self::open_with_counter(index, &POWER_VOTES)
    }

    fn open_with_counter(index: u32, counter: &'static VoteCounter) -> Result<Self, String> {
        let shim = shim()?;
        // Acquire before the fallible open call so an early error drops
        // `power_vote` and balances 0 -> 1 -> 0 cleanly.
        let power_vote = PowerVote::acquire_in(shim, counter);
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe { (shim.device_open)(index, &mut handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "open XRT device")?;
        Ok(Self {
            handle,
            shim,
            _power_vote: power_vote,
        })
    }

    /// The device's BDF-style identifier, for logs.
    pub fn bdf(&self) -> Result<String, String> {
        let mut buf = new_err_buf();
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.device_bdf)(
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
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.kernel_load)(
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
        Ok(XdnaKernel {
            handle,
            shim: self.shim,
        })
    }
}

/// A host-only buffer bound to one kernel argument.
pub struct XdnaBo {
    handle: *mut c_void,
    ptr: *mut c_void,
    len: usize,
    shim: &'static Shim,
}

/// One kernel invocation with its arguments bound.
pub struct XdnaRun {
    handle: *mut c_void,
    shim: &'static Shim,
}

/// A set of runs executed in submission order on one hardware context.
pub struct XdnaRunList {
    handle: *mut c_void,
    shim: &'static Shim,
}

impl XdnaKernel {
    /// Allocate a host-only buffer for data argument `arg` (3 or above).
    pub fn alloc_bo(&self, bytes: usize, arg: u32) -> Result<XdnaBo, String> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.bo_alloc)(
                self.handle,
                bytes,
                arg,
                &mut handle,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "allocate buffer")?;
        let ptr = unsafe { (self.shim.bo_map)(handle) };
        if ptr.is_null() {
            unsafe { (self.shim.bo_free)(handle) }
            return Err("allocate buffer: mapping failed".to_string());
        }
        Ok(XdnaBo {
            handle,
            ptr,
            len: bytes,
            shim: self.shim,
        })
    }

    /// Create a run with `bos[i]` bound to argument `3 + i`.
    pub fn create_run(&self, bos: &[&XdnaBo]) -> Result<XdnaRun, String> {
        let raw: Vec<*mut c_void> = bos.iter().map(|b| b.handle).collect();
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.run_create)(
                self.handle,
                raw.as_ptr(),
                raw.len(),
                &mut handle,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        check(rc, &err, "create run")?;
        Ok(XdnaRun {
            handle,
            shim: self.shim,
        })
    }

    /// Create an empty runlist on this kernel's hardware context.
    pub fn create_runlist(&self) -> Result<XdnaRunList, String> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.runlist_create)(self.handle, &mut handle, err.as_mut_ptr(), err.len())
        };
        check(rc, &err, "create runlist")?;
        Ok(XdnaRunList {
            handle,
            shim: self.shim,
        })
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

    /// Sync the entire buffer from host memory to the device.
    pub fn sync_to_device(&self) -> Result<(), String> {
        let mut err = new_err_buf();
        let rc = unsafe { (self.shim.bo_sync)(self.handle, 1, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "sync buffer to device")?;
        Ok(())
    }

    /// Sync the entire buffer from the device into host memory.
    ///
    /// Takes `&mut self` so no slice returned by [`Self::as_slice`] or
    /// [`Self::as_mut_slice`] can alias the DMA write into mapped host memory.
    pub fn sync_from_device(&mut self) -> Result<(), String> {
        let mut err = new_err_buf();
        let rc = unsafe { (self.shim.bo_sync)(self.handle, 0, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "sync buffer from device")?;
        Ok(())
    }

    fn check_range(&self, size: usize, offset: usize) -> Result<(), String> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| format!("sync buffer range overflow: offset={offset} size={size}"))?;
        if end > self.len {
            return Err(format!(
                "sync buffer range out of bounds: offset={offset} size={size} len={}",
                self.len
            ));
        }
        Ok(())
    }

    /// Sync `size` bytes starting at `offset` from host memory to the device.
    pub fn sync_range_to_device(&self, size: usize, offset: usize) -> Result<(), String> {
        self.check_range(size, offset)?;
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.bo_sync_range)(self.handle, 1, size, offset, err.as_mut_ptr(), err.len())
        };
        check(rc, &err, "sync buffer range to device")?;
        Ok(())
    }

    /// Sync `size` bytes starting at `offset` from the device into host memory.
    ///
    /// Takes `&mut self` so no slice returned by [`Self::as_slice`] or
    /// [`Self::as_mut_slice`] can alias the DMA write into mapped host memory.
    pub fn sync_range_from_device(&mut self, size: usize, offset: usize) -> Result<(), String> {
        self.check_range(size, offset)?;
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.bo_sync_range)(self.handle, 0, size, offset, err.as_mut_ptr(), err.len())
        };
        check(rc, &err, "sync buffer range from device")?;
        Ok(())
    }
}

impl Drop for XdnaBo {
    fn drop(&mut self) {
        unsafe { (self.shim.bo_free)(self.handle) }
    }
}

impl XdnaRun {
    /// Start the run asynchronously.
    pub fn start(&self) -> Result<(), String> {
        let mut err = new_err_buf();
        let rc = unsafe { (self.shim.run_start)(self.handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "start run")?;
        Ok(())
    }

    /// Wait for completion; returns the `ert_cmd_state` value (4 is
    /// `ERT_CMD_STATE_COMPLETED`). `timeout_ms` of 0 waits without a limit.
    pub fn wait(&self, timeout_ms: u32) -> Result<i32, String> {
        let mut err = new_err_buf();
        let rc =
            unsafe { (self.shim.run_wait)(self.handle, timeout_ms, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "wait for run")
    }
}

impl Drop for XdnaRun {
    fn drop(&mut self) {
        unsafe { (self.shim.run_free)(self.handle) }
    }
}

impl XdnaRunList {
    /// Append a run; runs execute in the order added.
    pub fn add(&mut self, run: &XdnaRun) -> Result<(), String> {
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.runlist_add)(self.handle, run.handle, err.as_mut_ptr(), err.len())
        };
        check(rc, &err, "add run to runlist")?;
        Ok(())
    }

    /// Execute the whole list.
    pub fn execute(&self) -> Result<(), String> {
        let mut err = new_err_buf();
        let rc = unsafe { (self.shim.runlist_execute)(self.handle, err.as_mut_ptr(), err.len()) };
        check(rc, &err, "execute runlist")?;
        Ok(())
    }

    /// Wait for the whole list; `timeout_ms` of 0 waits without a limit.
    pub fn wait(&self, timeout_ms: u32) -> Result<(), String> {
        let mut err = new_err_buf();
        let rc = unsafe {
            (self.shim.runlist_wait)(self.handle, timeout_ms, err.as_mut_ptr(), err.len())
        };
        check(rc, &err, "wait for runlist")?;
        Ok(())
    }
}

impl Drop for XdnaRunList {
    fn drop(&mut self) {
        unsafe { (self.shim.runlist_free)(self.handle) }
    }
}

impl Drop for XdnaDevice {
    fn drop(&mut self) {
        unsafe { (self.shim.device_free)(self.handle) }
    }
}

impl Drop for XdnaKernel {
    fn drop(&mut self) {
        unsafe { (self.shim.kernel_free)(self.handle) }
    }
}

/// Open the NPU and load `xclbin`, returning the device on success. Any
/// failure (no driver, no device, rejected xclbin) is the reason string.
pub fn probe(xclbin: &[u8], instr: &[u32]) -> Result<XdnaDevice, String> {
    let device = XdnaDevice::open(0)?;
    let _kernel = device.load_kernel(xclbin, instr)?;
    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::xdna::sys::fake::{Config, Event, with_shim};

    #[test]
    fn power_vote_refcounted_across_overlapping_devices() {
        static COUNTER: VoteCounter = VoteCounter::new();
        let ((), events) = with_shim(Config::default(), || {
            let d1 = XdnaDevice::open_with_counter(0, &COUNTER).expect("open d1");
            let d2 = XdnaDevice::open_with_counter(0, &COUNTER).expect("open d2");
            assert_eq!(d1.bdf().as_deref(), Ok("0000:c5:00.1"));
            drop(d1);
            // `d2` is still live, so `Wakelock(false)` must NOT have fired yet.
            drop(d2);
        });
        assert_eq!(
            events,
            vec![
                Event::Wakelock(true),
                Event::DeviceOpen {
                    index: 0,
                    handle: 0x1000
                },
                Event::DeviceOpen {
                    index: 0,
                    handle: 0x1001
                },
                Event::DeviceFree(0x1000),
                Event::DeviceFree(0x1001),
                Event::Wakelock(false),
            ]
        );
    }

    #[test]
    fn failed_device_open_releases_power_vote() {
        static COUNTER: VoteCounter = VoteCounter::new();
        let (res, events) = with_shim(
            Config {
                fail_open: true,
                ..Config::default()
            },
            || XdnaDevice::open_with_counter(0, &COUNTER),
        );
        let err = res.err().expect("open must fail");
        assert!(err.contains("synthetic open refusal"), "{err}");
        assert_eq!(events, vec![Event::Wakelock(true), Event::Wakelock(false)]);
    }

    #[test]
    fn failed_bo_map_frees_allocated_bo() {
        static COUNTER: VoteCounter = VoteCounter::new();
        let (err, events) = with_shim(
            Config {
                fail_bo_map: true,
                ..Config::default()
            },
            || {
                let dev = XdnaDevice::open_with_counter(0, &COUNTER).expect("open");
                let k = dev
                    .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                    .expect("load");
                k.alloc_bo(4096, 3).err().expect("bo_map must fail")
            },
        );
        assert!(err.contains("mapping failed"), "{err}");
        assert!(
            events.contains(&Event::BoAlloc {
                bytes: 4096,
                arg: 3,
                handle: 1,
            }) && events.contains(&Event::BoMap(1))
                && events.contains(&Event::BoFree(1)),
            "allocated BO must be freed when map returns null: {events:?}"
        );
    }

    #[test]
    fn bo_sync_range_rejects_out_of_bounds_and_overflow() {
        static COUNTER: VoteCounter = VoteCounter::new();
        let _ = with_shim(Config::default(), || {
            let dev = XdnaDevice::open_with_counter(0, &COUNTER).expect("open");
            let k = dev
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("load");
            let mut bo = k.alloc_bo(64, 3).expect("alloc");
            assert!(bo.sync_range_to_device(64, 0).is_ok());
            assert!(bo.sync_range_from_device(32, 32).is_ok());
            assert!(bo.sync_range_to_device(65, 0).is_err());
            assert!(bo.sync_range_from_device(1, 64).is_err());
            assert!(bo.sync_range_to_device(1, usize::MAX).is_err());
        });
    }
}

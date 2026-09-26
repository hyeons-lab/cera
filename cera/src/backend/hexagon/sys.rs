//! Dynamic loading and FFI bindings for Qualcomm FastRPC (`libcdsprpc.so`).
//!
//! FastRPC provides userspace communication between the Application Processor (AP)
//! and the Hexagon Compute DSP (CDSP) over shared DMA memory (`rpcmem`).

use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;

use crate::session::CeraError;

pub type RemoteHandle64 = u64;
pub type DspQueueHandle = *mut c_void;
pub type DspQueueCallback = extern "C" fn(context: *mut c_void);

pub const DOMAIN_CDSP: i32 = 3;
pub const FASTRPC_MAP_FD: u32 = 2;
pub const FASTRPC_MAP_FD_DELAYED: u32 = 3;
pub const DSPQUEUE_TIMEOUT_US: u32 = 1_000_000;
pub const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
pub const RPCMEM_DEFAULT_FLAGS: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RemoteBuf {
    pub buf: *mut c_void,
    pub len: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union RemoteArg {
    pub buf: RemoteBuf,
    pub h: u32,
}

/// Pack scalar descriptors for `remote_handle64_invoke`.
pub const fn remote_scalars_make(method: u32, in_bufs: u32, out_bufs: u32) -> u32 {
    ((method & 0xff) << 24) | ((in_bufs & 0xff) << 16) | ((out_bufs & 0xff) << 8)
}

// Function pointer signatures for symbols loaded from libcdsprpc.so
type RpcmemAllocFn = extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
type RpcmemAlloc2Fn = extern "C" fn(heapid: i32, flags: u32, size: usize) -> *mut c_void;
type RpcmemFreeFn = extern "C" fn(po: *mut c_void);
type RpcmemToFdFn = extern "C" fn(po: *mut c_void) -> i32;

type FastrpcMmapFn = extern "C" fn(
    domain: i32,
    fd: i32,
    addr: *mut c_void,
    offset: i32,
    length: usize,
    flags: u32,
) -> i32;
type FastrpcMunmapFn = extern "C" fn(domain: i32, fd: i32, addr: *mut c_void, length: usize) -> i32;

type RemoteHandle64OpenFn = extern "C" fn(name: *const c_char, ph: *mut RemoteHandle64) -> i32;
type RemoteHandle64InvokeFn =
    extern "C" fn(h: RemoteHandle64, dw_scalars: u32, pra: *mut RemoteArg) -> i32;
type RemoteHandle64CloseFn = extern "C" fn(h: RemoteHandle64) -> i32;
type RemoteSessionControlFn = extern "C" fn(req: u32, data: *mut c_void, datalen: u32) -> i32;

type DspqueueCreateFn = extern "C" fn(
    domain: i32,
    flags: u32,
    req_queue_size: u32,
    resp_queue_size: u32,
    packet_cb: Option<DspQueueCallback>,
    error_cb: Option<DspQueueCallback>,
    callback_context: *mut c_void,
    queue: *mut DspQueueHandle,
) -> i32;
type DspqueueCloseFn = extern "C" fn(queue: DspQueueHandle) -> i32;
type DspqueueExportFn = extern "C" fn(queue: DspQueueHandle, queue_id: *mut u64) -> i32;
type DspqueueWriteFn = extern "C" fn(
    queue: DspQueueHandle,
    flags: u32,
    num_buffers: u32,
    buffers: *const super::types::DspQueueBuffer,
    msg_len: u32,
    msg: *const u8,
    timeout_us: u32,
) -> i32;
type DspqueueReadFn = extern "C" fn(
    queue: DspQueueHandle,
    flags: *mut u32,
    max_buffers: u32,
    num_buffers: *mut u32,
    buffers: *mut super::types::DspQueueBuffer,
    max_msg_len: u32,
    msg_len: *mut u32,
    msg: *mut u8,
    timeout_us: u32,
) -> i32;

/// Dynamically loaded FastRPC library handle and dispatch table.
pub struct FastRpcDriver {
    #[cfg(unix)]
    handle: *mut c_void,

    // Memory allocation
    rpcmem_alloc: RpcmemAllocFn,
    rpcmem_alloc2: Option<RpcmemAlloc2Fn>,
    rpcmem_free: RpcmemFreeFn,
    rpcmem_to_fd: RpcmemToFdFn,

    // FastRPC mmap
    fastrpc_mmap: FastrpcMmapFn,
    fastrpc_munmap: FastrpcMunmapFn,

    // Remote handle
    remote_handle64_open: RemoteHandle64OpenFn,
    remote_handle64_invoke: RemoteHandle64InvokeFn,
    remote_handle64_close: RemoteHandle64CloseFn,
    #[allow(dead_code)]
    remote_session_control: Option<RemoteSessionControlFn>,

    // DSP queue
    dspqueue_create: DspqueueCreateFn,
    dspqueue_close: DspqueueCloseFn,
    dspqueue_export: DspqueueExportFn,
    dspqueue_write: DspqueueWriteFn,
    dspqueue_read: DspqueueReadFn,

    /// Non-blocking completion polling (llama's `GGML_HEXAGON_OPPOLL`):
    /// `dspqueue_read` is issued with timeout 0 and retried on
    /// `EWOULDBLOCK` instead of sleeping in the kernel. Cuts wakeup
    /// latency (~35% decode gain at small flush counts) at the cost of a
    /// spinning host thread during DSP batches. Default on (one busy
    /// core during active inference is cheaper than CPU inference);
    /// `CERA_HEXAGON_OPPOLL=0` restores blocking reads.
    oppoll: bool,
}

// FastRPC driver dispatch is thread-safe across function invocations.
unsafe impl Send for FastRpcDriver {}
unsafe impl Sync for FastRpcDriver {}

impl FastRpcDriver {
    /// Attempt to dynamically load `libcdsprpc.so` (or system driver on Linux/Android).
    pub fn load() -> Result<Arc<Self>, CeraError> {
        #[cfg(unix)]
        {
            // Soname first: inside an APK the app linker namespace resolves
            // `libcdsprpc.so` once the manifest declares it (see the AAR
            // manifest's `<uses-native-library>` entry; the lib is on the
            // vendor public list). Absolute vendor paths stay blocked by
            // that same namespace, so they are only a fallback for
            // shell/CLI contexts outside APKs. Every miss is non-fatal:
            // the loop keeps trying until one `dlopen` succeeds.
            let lib_names = [
                "libcdsprpc.so",
                "libadsprpc.so",
                "/vendor/lib64/libcdsprpc.so",
                "/vendor/lib64/libadsprpc.so",
            ];
            let mut lib_handle = std::ptr::null_mut();

            for name in &lib_names {
                let c_name = CString::new(*name).unwrap();
                let h = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
                if !h.is_null() {
                    lib_handle = h;
                    break;
                }
            }

            if lib_handle.is_null() {
                return Err(CeraError::Backend(
                    "Qualcomm FastRPC driver (libcdsprpc.so) not found on this system".into(),
                ));
            }

            unsafe {
                macro_rules! resolve {
                    ($name:expr, $type:ty) => {{
                        let c_sym = CString::new($name).unwrap();
                        let sym = libc::dlsym(lib_handle, c_sym.as_ptr());
                        if sym.is_null() {
                            libc::dlclose(lib_handle);
                            return Err(CeraError::Backend(format!(
                                "failed to resolve required FastRPC symbol: {}",
                                $name
                            )));
                        }
                        std::mem::transmute::<*mut c_void, $type>(sym)
                    }};
                }

                macro_rules! resolve_opt {
                    ($name:expr, $type:ty) => {{
                        let c_sym = CString::new($name).unwrap();
                        let sym = libc::dlsym(lib_handle, c_sym.as_ptr());
                        if sym.is_null() {
                            None
                        } else {
                            Some(std::mem::transmute::<*mut c_void, $type>(sym))
                        }
                    }};
                }

                let driver = Self {
                    handle: lib_handle,
                    rpcmem_alloc: resolve!("rpcmem_alloc", RpcmemAllocFn),
                    rpcmem_alloc2: resolve_opt!("rpcmem_alloc2", RpcmemAlloc2Fn),
                    rpcmem_free: resolve!("rpcmem_free", RpcmemFreeFn),
                    rpcmem_to_fd: resolve!("rpcmem_to_fd", RpcmemToFdFn),

                    fastrpc_mmap: resolve!("fastrpc_mmap", FastrpcMmapFn),
                    fastrpc_munmap: resolve!("fastrpc_munmap", FastrpcMunmapFn),

                    remote_handle64_open: resolve!("remote_handle64_open", RemoteHandle64OpenFn),
                    remote_handle64_invoke: resolve!(
                        "remote_handle64_invoke",
                        RemoteHandle64InvokeFn
                    ),
                    remote_handle64_close: resolve!("remote_handle64_close", RemoteHandle64CloseFn),
                    remote_session_control: resolve_opt!(
                        "remote_session_control",
                        RemoteSessionControlFn
                    ),

                    dspqueue_create: resolve!("dspqueue_create", DspqueueCreateFn),
                    dspqueue_close: resolve!("dspqueue_close", DspqueueCloseFn),
                    dspqueue_export: resolve!("dspqueue_export", DspqueueExportFn),
                    dspqueue_write: resolve!("dspqueue_write", DspqueueWriteFn),
                    dspqueue_read: resolve!("dspqueue_read", DspqueueReadFn),
                    oppoll: std::env::var("CERA_HEXAGON_OPPOLL")
                        .map(|v| v != "0")
                        .unwrap_or(true),
                };

                Ok(Arc::new(driver))
            }
        }

        #[cfg(not(unix))]
        {
            Err(CeraError::Backend(
                "Qualcomm Hexagon backend is only supported on Unix/Android targets".into(),
            ))
        }
    }

    /// Allocate a shared memory buffer via `rpcmem`.
    pub fn rpcmem_alloc(&self, size: usize) -> Result<*mut u8, CeraError> {
        if size == 0 || size > i32::MAX as usize {
            return Err(CeraError::Backend(format!(
                "invalid rpcmem allocation size: {size} bytes (must be between 1 and {} bytes)",
                i32::MAX
            )));
        }

        let ptr = if let Some(alloc2) = self.rpcmem_alloc2 {
            (alloc2)(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size)
        } else {
            (self.rpcmem_alloc)(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size as i32)
        };

        if ptr.is_null() {
            Err(CeraError::OutOfMemory {
                requested_bytes: size as u64,
            })
        } else {
            Ok(ptr as *mut u8)
        }
    }

    /// Free a shared memory buffer allocated via `rpcmem`.
    pub fn rpcmem_free(&self, ptr: *mut u8) {
        if !ptr.is_null() {
            (self.rpcmem_free)(ptr as *mut c_void);
        }
    }

    /// Obtain the underlying DMA file descriptor for an `rpcmem` allocation.
    pub fn rpcmem_to_fd(&self, ptr: *mut u8) -> Result<i32, CeraError> {
        let fd = (self.rpcmem_to_fd)(ptr as *mut c_void);
        if fd < 0 {
            Err(CeraError::Backend(format!(
                "rpcmem_to_fd failed for buffer {:p} (error {})",
                ptr, fd
            )))
        } else {
            Ok(fd)
        }
    }

    /// Map a memory buffer into the CDSP virtual memory address space.
    pub fn fastrpc_mmap(&self, fd: i32, addr: *mut u8, length: usize) -> Result<(), CeraError> {
        let ret = (self.fastrpc_mmap)(
            DOMAIN_CDSP,
            fd,
            addr as *mut c_void,
            0,
            length,
            FASTRPC_MAP_FD,
        );
        if ret != 0 {
            Err(CeraError::Backend(format!(
                "fastrpc_mmap failed for fd {} length {} (error {})",
                fd, length, ret
            )))
        } else {
            Ok(())
        }
    }

    /// Unmap a memory buffer from the CDSP virtual memory address space.
    pub fn fastrpc_munmap(&self, fd: i32, addr: *mut u8, length: usize) -> Result<(), CeraError> {
        let ret = (self.fastrpc_munmap)(DOMAIN_CDSP, fd, addr as *mut c_void, length);
        if ret != 0 {
            Err(CeraError::Backend(format!(
                "fastrpc_munmap failed for fd {} length {} (error {})",
                fd, length, ret
            )))
        } else {
            Ok(())
        }
    }

    /// Enable Qualcomm Unsigned Process Domain (Unsigned PD) for the given domain.
    pub fn enable_unsigned_pd(&self, domain: u32) -> Result<(), CeraError> {
        if let Some(control_fn) = self.remote_session_control {
            #[repr(C)]
            struct UnsignedModuleControl {
                domain: u32,
                enable: u32,
            }
            let mut ctrl = UnsignedModuleControl { domain, enable: 1 };
            let ret = control_fn(
                2, // DSPRPC_CONTROL_UNSIGNED_MODULE
                &mut ctrl as *mut _ as *mut c_void,
                std::mem::size_of::<UnsignedModuleControl>() as u32,
            );
            if ret != 0 {
                return Err(CeraError::Backend(format!(
                    "remote_session_control(unsigned) failed for domain {domain} (error 0x{ret:08x})"
                )));
            }
        }
        Ok(())
    }

    /// Open a FastRPC handle to a skeleton library (e.g. `file:///libggml-htp-v75.so?domain=3`).
    pub fn open_skel_handle(&self, uri: &str) -> Result<RemoteHandle64, CeraError> {
        let c_uri = CString::new(uri).map_err(|e| CeraError::Backend(e.to_string()))?;
        let mut handle: RemoteHandle64 = 0;
        let ret = (self.remote_handle64_open)(c_uri.as_ptr(), &mut handle);
        if ret != 0 {
            Err(CeraError::Backend(format!(
                "remote_handle64_open failed for uri '{}' (error 0x{:08x})",
                uri, ret
            )))
        } else {
            Ok(handle)
        }
    }

    /// Close a FastRPC handle.
    pub fn close_skel_handle(&self, handle: RemoteHandle64) {
        if handle != 0 {
            let _ = (self.remote_handle64_close)(handle);
        }
    }

    /// Invoke an IDL method on the skeleton library.
    pub fn invoke_skel(
        &self,
        handle: RemoteHandle64,
        scalars: u32,
        args: &mut [RemoteArg],
    ) -> Result<(), CeraError> {
        let ret = (self.remote_handle64_invoke)(handle, scalars, args.as_mut_ptr());
        if ret != 0 {
            Err(CeraError::Backend(format!(
                "remote_handle64_invoke failed (error 0x{:08x})",
                ret
            )))
        } else {
            Ok(())
        }
    }

    /// Create an asynchronous DSP command queue.
    pub fn create_dsp_queue(
        &self,
        req_size: u32,
        resp_size: u32,
    ) -> Result<DspQueueHandle, CeraError> {
        let mut queue: DspQueueHandle = std::ptr::null_mut();
        let ret = (self.dspqueue_create)(
            DOMAIN_CDSP,
            0,
            req_size,
            resp_size,
            None,
            None,
            std::ptr::null_mut(),
            &mut queue,
        );
        if ret != 0 || queue.is_null() {
            Err(CeraError::Backend(format!(
                "dspqueue_create failed (error 0x{:08x})",
                ret
            )))
        } else {
            Ok(queue)
        }
    }

    /// Export a DSP queue ID for registration with `htp_iface_start`.
    pub fn export_dsp_queue(&self, queue: DspQueueHandle) -> Result<u64, CeraError> {
        let mut queue_id: u64 = 0;
        let ret = (self.dspqueue_export)(queue, &mut queue_id);
        if ret != 0 {
            Err(CeraError::Backend(format!(
                "dspqueue_export failed (error 0x{:08x})",
                ret
            )))
        } else {
            Ok(queue_id)
        }
    }

    /// Close a DSP command queue.
    pub fn close_dsp_queue(&self, queue: DspQueueHandle) {
        if !queue.is_null() {
            let _ = (self.dspqueue_close)(queue);
        }
    }

    /// Write a batch payload to the DSP command queue.
    pub fn write_dsp_queue(
        &self,
        queue: DspQueueHandle,
        buffers: &[super::types::DspQueueBuffer],
        msg: &[u8],
    ) -> Result<(), CeraError> {
        let ret = (self.dspqueue_write)(
            queue,
            0,
            buffers.len() as u32,
            buffers.as_ptr(),
            msg.len() as u32,
            msg.as_ptr(),
            DSPQUEUE_TIMEOUT_US,
        );
        if ret != 0 {
            Err(CeraError::Backend(format!(
                "dspqueue_write failed (error 0x{:08x})",
                ret
            )))
        } else {
            Ok(())
        }
    }

    /// Read a response payload from the DSP command queue.
    pub fn read_dsp_queue(
        &self,
        queue: DspQueueHandle,
        buffers: &mut [super::types::DspQueueBuffer],
        msg: &mut [u8],
    ) -> Result<u32, CeraError> {
        let mut flags: u32 = 0;
        let mut n_bufs: u32 = 0;
        let mut msg_len: u32 = 0;
        let mut timeouts = 0;
        // Oppoll: timeout 0 turns the read non-blocking; the retry loop
        // below spins on EWOULDBLOCK until the response lands.
        let mut timeout = if self.oppoll { 0 } else { DSPQUEUE_TIMEOUT_US };
        // Oppoll hang guard: the same 30s budget blocking mode gets from
        // its per-read timeouts. Without it a wedged DSP pins this thread
        // at 100% forever, teardown included (`Drop` flushes through this
        // loop). (llama.cpp spins unbounded here; the deadline is where
        // cera deliberately differs.) Checked at the top of every iteration
        // so no retry arm (including AEE_EINTERRUPTED) can bypass it; lazy
        // so blocking mode pays no clock read for a budget it never uses.
        let deadline = self
            .oppoll
            .then(|| std::time::Instant::now() + std::time::Duration::from_secs(30));
        let mut spins: u64 = 0;
        loop {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Err(CeraError::Backend(format!(
                    "dspqueue_read: no DSP response after 30s ({spins} spins)"
                )));
            }
            let ret = (self.dspqueue_read)(
                queue,
                &mut flags,
                buffers.len() as u32,
                &mut n_bufs,
                buffers.as_mut_ptr(),
                msg.len() as u32,
                &mut msg_len,
                msg.as_mut_ptr(),
                timeout,
            );
            if ret == 0 {
                return Ok(msg_len);
            }
            if ret == 0x204 {
                // AEE_EINTERRUPTED: retry immediately
                continue;
            }
            if ret == 0x0c
                || ret == (0x8000_0407u32 as i32)
                || ret == (0x8000_0408u32 as i32)
                || ret == 11
            {
                // ETIMEDOUT, AEE_EEXPIRED, AEE_EWOULDBLOCK: DSP is still
                // processing.
                if self.oppoll {
                    spins += 1;
                    if spins > 10_000 {
                        // Slow batch: stop burning a core and block for the
                        // response like blocking mode does.
                        timeout = DSPQUEUE_TIMEOUT_US;
                    }
                    // Expiry is checked at the top of the next iteration.
                    continue;
                }
                timeouts += 1;
                if timeouts < 30 {
                    continue;
                }
            }
            return Err(CeraError::Backend(format!(
                "dspqueue_read failed (error 0x{:08x})",
                ret
            )));
        }
    }
}

impl Drop for FastRpcDriver {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            unsafe {
                libc::dlclose(self.handle);
            }
        }
    }
}

//! Asynchronous DSP command queue session for Qualcomm Hexagon NPU.
//!
//! Batches operations into `htp_opbatch_req` descriptors and dispatches
//! to the DSP hardware via FastRPC `dspqueue`.

use std::collections::HashMap;
use std::sync::Arc;

use super::rpcmem::RpcmemBuffer;
use super::sys::FastRpcDriver;
use super::types::{
    DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER, DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT, DspQueueBuffer,
    HtpBufDesc, HtpOpBatchReq, HtpOpBatchRsp, HtpOpCode, HtpOpDesc, HtpProfDesc, HtpStatus,
    HtpTensor,
};
use crate::session::CeraError;

/// Staging buffer size for batch queue serialization (4 MB).
const STAGING_BUFFER_SIZE: usize = 4 * 1024 * 1024;

/// Cap on consecutive stale responses drained in one flush. Stale responses
/// are bounded by prior timeouts (one per timed-out batch); anything past
/// this is firmware misbehavior, and an uncapped drain would hang the flush
/// (including from `Drop`) on a stuck seq.
const MAX_CONSECUTIVE_STALE: u32 = 32;

fn step_enabled() -> bool {
    static STEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STEP.get_or_init(|| std::env::var_os("CERA_HEXAGON_STEP").is_some())
}

fn debug_enabled() -> bool {
    static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var_os("CERA_HEXAGON_DEBUG").is_some())
}

fn profile_enabled() -> bool {
    static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROFILE.get_or_init(|| std::env::var_os("CERA_HEXAGON_PROFILE").is_some())
}

/// One stale-drain decision: what the drain loop does with a freshly-read
/// `rsp_seq` when `expected` was awaited and `drained` stale responses
/// have already been consumed this flush. Pure so the seq/cap boundary is
/// unit-testable without a DSP.
#[derive(Debug, PartialEq, Eq)]
enum StaleDrainAction {
    /// `rsp_seq < expected`, under the cap: count and keep draining.
    DrainStale,
    /// `rsp_seq < expected`, over the cap: fail the flush closed.
    CapExceeded,
    /// `rsp_seq == expected`: this batch's response; stop draining.
    Current,
    /// `rsp_seq > expected`: a future batch's response (desync); fail.
    Future,
}

fn stale_drain_action(rsp_seq: u64, expected: u64, drained: u32) -> StaleDrainAction {
    if rsp_seq < expected {
        // `drained >= MAX` is the overflow-free form of
        // `drained + 1 > MAX` (this response would be the 33rd).
        if drained >= MAX_CONSECUTIVE_STALE {
            StaleDrainAction::CapExceeded
        } else {
            StaleDrainAction::DrainStale
        }
    } else if rsp_seq == expected {
        StaleDrainAction::Current
    } else {
        StaleDrainAction::Future
    }
}

/// Inline flat-array buffer map avoiding heap allocations on the hot path.
#[derive(Clone, Debug, Default)]
pub struct BufferIndexMap {
    inline: [(i32, u16); 16],
    count: usize,
    overflow: Option<HashMap<i32, u16>>,
}

impl BufferIndexMap {
    #[inline]
    pub fn new() -> Self {
        Self {
            inline: [(0, 0); 16],
            count: 0,
            overflow: None,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.count = 0;
        if let Some(m) = &mut self.overflow {
            m.clear();
        }
    }

    #[inline]
    pub fn get(&self, fd: i32) -> Option<u16> {
        for i in 0..self.count {
            if self.inline[i].0 == fd {
                return Some(self.inline[i].1);
            }
        }
        if let Some(m) = &self.overflow {
            return m.get(&fd).copied();
        }
        None
    }

    #[inline]
    pub fn insert(&mut self, fd: i32, idx: u16) {
        for i in 0..self.count {
            if self.inline[i].0 == fd {
                self.inline[i].1 = idx;
                return;
            }
        }
        if self.count < 16 {
            self.inline[self.count] = (fd, idx);
            self.count += 1;
        } else {
            self.overflow
                .get_or_insert_with(HashMap::new)
                .insert(fd, idx);
        }
    }

    pub fn to_hash_map(&self) -> HashMap<i32, u16> {
        let mut map =
            HashMap::with_capacity(self.count + self.overflow.as_ref().map_or(0, |m| m.len()));
        for i in 0..self.count {
            map.insert(self.inline[i].0, self.inline[i].1);
        }
        if let Some(overflow) = &self.overflow {
            map.extend(overflow.iter().map(|(&k, &v)| (k, v)));
        }
        map
    }
}

/// Pre-serialized and reusable batch representation in contiguous host memory.
///
/// Holds serialized buffer descriptors, tensor descriptors, op descriptors,
/// and profiling descriptors ready for single-memcpy transfer to `staging_buf`.
#[derive(Clone, Debug)]
pub struct StagedBatch {
    pub raw_bytes: Vec<u8>,
    pub n_bufs: u32,
    pub n_tensors: u32,
    pub n_ops: u32,
    pub bufs_bytes: usize,
    pub tens_bytes: usize,
    pub ops_bytes: usize,
    pub prof_bytes: usize,
    pub total_bytes: usize,
}

impl StagedBatch {
    /// Reject a batch whose byte accounting disagrees with `raw_bytes`, since
    /// the flush paths copy `total_bytes` out of it into DSP-visible memory
    /// and every field is public.
    fn validate(&self) -> Result<(), CeraError> {
        let parts = self
            .bufs_bytes
            .checked_add(self.tens_bytes)
            .and_then(|n| n.checked_add(self.ops_bytes))
            .and_then(|n| n.checked_add(self.prof_bytes));
        // Each section's byte count must match its descriptor count: the DSP
        // walks `n_*` descriptors, so a lying count reads past the section.
        let counted = |n: u32, size: usize| (n as usize).checked_mul(size);
        let counts_match = counted(self.n_bufs, std::mem::size_of::<HtpBufDesc>())
            == Some(self.bufs_bytes)
            && counted(self.n_tensors, std::mem::size_of::<HtpTensor>()) == Some(self.tens_bytes)
            && counted(self.n_ops, std::mem::size_of::<HtpOpDesc>()) == Some(self.ops_bytes)
            && counted(self.n_ops, std::mem::size_of::<HtpProfDesc>()) == Some(self.prof_bytes);
        if !counts_match
            || parts != Some(self.total_bytes)
            || self.raw_bytes.len() < self.total_bytes
        {
            return Err(CeraError::Backend(format!(
                "inconsistent StagedBatch: total {} vs parts {parts:?} vs raw {}",
                self.total_bytes,
                self.raw_bytes.len()
            )));
        }
        Ok(())
    }

    /// Read tensor descriptor `ti` from the staged command buffer.
    pub fn tensor(&self, ti: usize) -> HtpTensor {
        self.read_desc(self.tensor_offset(ti))
    }

    /// Read-modify-write tensor descriptor `ti` in the staged command buffer.
    #[inline]
    pub fn update_tensor(&mut self, ti: usize, f: impl FnOnce(&mut HtpTensor)) {
        let offset = self.tensor_offset(ti);
        self.update_desc(offset, f);
    }

    /// Read operation descriptor `op_idx` from the staged command buffer.
    pub fn op(&self, op_idx: usize) -> HtpOpDesc {
        self.read_desc(self.op_offset(op_idx))
    }

    /// Read-modify-write operation descriptor `op_idx`.
    #[inline]
    pub fn update_op(&mut self, op_idx: usize, f: impl FnOnce(&mut HtpOpDesc)) {
        let offset = self.op_offset(op_idx);
        self.update_desc(offset, f);
    }

    fn tensor_offset(&self, ti: usize) -> usize {
        assert!(
            ti < self.n_tensors as usize,
            "tensor index {ti} out of bounds ({})",
            self.n_tensors
        );
        self.bufs_bytes + ti * std::mem::size_of::<HtpTensor>()
    }

    fn op_offset(&self, op_idx: usize) -> usize {
        assert!(
            op_idx < self.n_ops as usize,
            "op index {op_idx} out of bounds ({})",
            self.n_ops
        );
        self.bufs_bytes + self.tens_bytes + op_idx * std::mem::size_of::<HtpOpDesc>()
    }

    /// Range-check `size_of::<T>()` bytes at `offset`. `raw_bytes` is a
    /// `Vec<u8>` (align 1) and descriptors need align 8, so descriptors are
    /// accessed by value with unaligned reads and writes: nothing depends on
    /// what alignment the allocator happened to return.
    #[inline]
    fn check_desc_range<T>(&self, offset: usize) {
        let end = offset.checked_add(std::mem::size_of::<T>());
        assert!(
            end.is_some_and(|e| e <= self.raw_bytes.len()),
            "descriptor at {offset} overruns staged batch ({} bytes)",
            self.raw_bytes.len()
        );
    }

    fn read_desc<T: Copy>(&self, offset: usize) -> T {
        self.check_desc_range::<T>(offset);
        // SAFETY: in bounds (checked above); `T` is a plain `repr(C)`
        // descriptor valid for any bit pattern.
        unsafe { (self.raw_bytes.as_ptr().add(offset) as *const T).read_unaligned() }
    }

    fn update_desc<T: Copy>(&mut self, offset: usize, f: impl FnOnce(&mut T)) {
        let mut value: T = self.read_desc(offset);
        f(&mut value);
        // SAFETY: in bounds (checked by `read_desc`); the pointer derives
        // from `as_mut_ptr`, so writing through it is permitted, and the
        // write is unaligned for a plain descriptor.
        unsafe { (self.raw_bytes.as_mut_ptr().add(offset) as *mut T).write_unaligned(value) }
    }
}

/// An active DSP command queue session managing batched request and response dispatch.
pub struct HexagonQueueSession {
    driver: Arc<FastRpcDriver>,
    queue: crate::backend::hexagon::sys::DspQueueHandle,
    queue_id: u64,
    staging_buf: RpcmemBuffer,
    bufs: Vec<HtpBufDesc>,
    buf_map: BufferIndexMap,
    tens: Vec<HtpTensor>,
    ops: Vec<HtpOpDesc>,
    /// Auto-flush once this many ops are queued (`None` = unbounded).
    /// Small-M prefill chunks cap this (see `SMALL_M_MAX_OPS_PER_FLUSH`):
    /// large single-flush batches compute nondeterministically there.
    max_ops_per_flush: Option<usize>,
    seq: u64,
    /// Aggregate DSP microseconds per opcode across flushes (profiling only).
    prof: HashMap<u32, (u64, u64)>,
    prof_host_us: u64,
    prof_dsp_us: u64,
    prof_flushes: u64,
    /// DSP worker threads from `htp_iface_hwinfo` (llama's `sess->n_threads`).
    /// kparams thread counts derive from this; the default is llama's
    /// hwinfo-failure fallback until `HexagonDevice` overwrites it.
    dsp_threads: u32,
    resident_staged_id: Option<u64>,
}

// Queue session operations are Send across threads when guarded by model session locks.
unsafe impl Send for HexagonQueueSession {}

impl HexagonQueueSession {
    /// Create a new command queue session.
    pub fn new(driver: Arc<FastRpcDriver>) -> Result<Self, CeraError> {
        let queue = driver.create_dsp_queue(128 * 1024, 64 * 1024)?;
        let queue_id = match driver.export_dsp_queue(queue) {
            Ok(id) => id,
            Err(e) => {
                driver.close_dsp_queue(queue);
                return Err(e);
            }
        };

        let staging_buf = match RpcmemBuffer::alloc(Arc::clone(&driver), STAGING_BUFFER_SIZE, true)
        {
            Ok(buf) => buf,
            Err(e) => {
                driver.close_dsp_queue(queue);
                return Err(e);
            }
        };

        Ok(Self {
            driver,
            queue,
            queue_id,
            staging_buf,
            bufs: Vec::with_capacity(32),
            buf_map: BufferIndexMap::new(),
            tens: Vec::with_capacity(256),
            ops: Vec::with_capacity(128),
            max_ops_per_flush: None,
            seq: 1,
            prof: HashMap::new(),
            prof_host_us: 0,
            prof_dsp_us: 0,
            prof_flushes: 0,
            dsp_threads: 8,
            resident_staged_id: None,
        })
    }

    /// DSP worker threads for kparams (from `htp_iface_hwinfo`).
    pub fn dsp_threads(&self) -> u32 {
        self.dsp_threads.max(1)
    }

    /// Record the `htp_iface_hwinfo` thread count (0 keeps the default).
    pub fn set_dsp_threads(&mut self, n: u32) {
        if n > 0 {
            self.dsp_threads = n;
        }
    }

    /// Underlying exported DSP queue identifier for registration with `htp_iface_start`.
    pub fn queue_id(&self) -> u64 {
        self.queue_id
    }

    /// Cap ops per flush (`None` restores unbounded batching). While set,
    /// `enqueue_op` flushes automatically once the cap is reached.
    pub fn set_max_ops_per_flush(&mut self, max: Option<usize>) {
        self.max_ops_per_flush = max.filter(|&m| m >= 1);
    }

    /// Next free DSP-visible index for a registry. The DSP reads `u16`
    /// indices, and `0xffff` is the wire absent-operand marker (unused
    /// src/dst slots are padded with it), so a batch past 65534 entries is
    /// an error, never a silently aliasing truncation.
    fn batch_index(len: usize, what: &str) -> Result<u16, CeraError> {
        if len >= 0xffff {
            return Err(CeraError::Backend(format!(
                "HTP batch exceeds 65534 {what} ({len})"
            )));
        }
        Ok(len as u16)
    }

    /// Drop the pending batch. Used when registration fails: the batch is
    /// uncompletable (its owning dispatch already failed) and retaining it
    /// would wedge the session, since every later registration fails the
    /// same way. No `seq` advance: nothing was attempted, so the
    /// stale-drain accounting is unaffected.
    pub fn drop_pending_batch(&mut self) {
        self.bufs.clear();
        self.buf_map.clear();
        self.tens.clear();
        self.ops.clear();
        self.resident_staged_id = None;
    }

    /// Number of operations currently enqueued in the pending batch.
    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }

    /// Export the currently pending batch as a reusable template.
    pub fn export_batch(
        &self,
    ) -> (
        Vec<HtpBufDesc>,
        HashMap<i32, u16>,
        Vec<HtpTensor>,
        Vec<HtpOpDesc>,
    ) {
        (
            self.bufs.clone(),
            self.buf_map.to_hash_map(),
            self.tens.clone(),
            self.ops.clone(),
        )
    }

    /// Load a previously prepared batch template into the queue session.
    pub fn load_batch(
        &mut self,
        bufs: &[HtpBufDesc],
        buf_map: &HashMap<i32, u16>,
        tens: &[HtpTensor],
        ops: &[HtpOpDesc],
    ) {
        self.bufs.clear();
        self.bufs.extend_from_slice(bufs);
        self.buf_map.clear();
        for (&fd, &idx) in buf_map {
            self.buf_map.insert(fd, idx);
        }
        self.tens.clear();
        self.tens.extend_from_slice(tens);
        self.ops.clear();
        self.ops.extend_from_slice(ops);
    }

    /// Export the currently pending batch as a pre-serialized staged template.
    pub fn export_staged_batch(&self) -> Result<StagedBatch, CeraError> {
        let bufs_bytes = self.bufs.len() * std::mem::size_of::<HtpBufDesc>();
        let tens_bytes = self.tens.len() * std::mem::size_of::<HtpTensor>();
        let ops_bytes = self.ops.len() * std::mem::size_of::<HtpOpDesc>();
        let prof_bytes = self.ops.len() * std::mem::size_of::<HtpProfDesc>();
        let total_bytes = bufs_bytes + tens_bytes + ops_bytes + prof_bytes;

        if total_bytes > self.staging_buf.size() {
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_buf.size()
            )));
        }

        let mut raw_bytes = vec![0u8; total_bytes];
        unsafe {
            let base = raw_bytes.as_mut_ptr();
            std::ptr::copy_nonoverlapping(self.bufs.as_ptr() as *const u8, base, bufs_bytes);
            std::ptr::copy_nonoverlapping(
                self.tens.as_ptr() as *const u8,
                base.add(bufs_bytes),
                tens_bytes,
            );
            std::ptr::copy_nonoverlapping(
                self.ops.as_ptr() as *const u8,
                base.add(bufs_bytes + tens_bytes),
                ops_bytes,
            );
            // Explicitly zero prof_bytes segment to guarantee cleared HtpProfDesc descriptors,
            // mirroring flush() behavior.
            if prof_bytes > 0 {
                std::ptr::write_bytes(base.add(bufs_bytes + tens_bytes + ops_bytes), 0, prof_bytes);
            }
        }

        Ok(StagedBatch {
            raw_bytes,
            n_bufs: self.bufs.len() as u32,
            n_tensors: self.tens.len() as u32,
            n_ops: self.ops.len() as u32,
            bufs_bytes,
            tens_bytes,
            ops_bytes,
            prof_bytes,
            total_bytes,
        })
    }

    /// Dispatch a pre-staged command batch without rebuilding or re-serializing descriptors.
    pub fn flush_staged(&mut self, staged: &StagedBatch) -> Result<(), CeraError> {
        self.resident_staged_id = None;
        staged.validate()?;
        let total_bytes = staged.total_bytes;
        if total_bytes > self.staging_buf.size() {
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_buf.size()
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                staged.raw_bytes.as_ptr(),
                self.staging_buf.as_mut_ptr(),
                total_bytes,
            );
        }

        let prof_offset = staged.bufs_bytes + staged.tens_bytes + staged.ops_bytes;
        self.dispatch_batch_buffer(
            total_bytes,
            staged.n_bufs,
            staged.n_tensors,
            staged.n_ops,
            prof_offset,
        )
    }

    /// Dispatch a pre-staged command batch, retaining resident descriptor memory in `staging_buf`.
    /// When `resident_id` matches the currently resident batch, only the memory slices in
    /// `patch_ranges` are copied from `staged.raw_bytes` to `staging_buf`, eliminating full-buffer copies.
    pub fn flush_staged_resident(
        &mut self,
        resident_id: u64,
        staged: &StagedBatch,
        patch_ranges: &[std::ops::Range<usize>],
    ) -> Result<(), CeraError> {
        staged.validate()?;
        let total_bytes = staged.total_bytes;
        if total_bytes > self.staging_buf.size() {
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_buf.size()
            )));
        }

        if self.resident_staged_id != Some(resident_id) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staged.raw_bytes.as_ptr(),
                    self.staging_buf.as_mut_ptr(),
                    total_bytes,
                );
            }
            self.resident_staged_id = Some(resident_id);
        } else {
            for range in patch_ranges {
                let start = range.start;
                let end = range.end.min(total_bytes);
                if start < end {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            staged.raw_bytes.as_ptr().add(start),
                            self.staging_buf.as_mut_ptr().add(start),
                            end - start,
                        );
                    }
                }
            }
        }

        let prof_offset = staged.bufs_bytes + staged.tens_bytes + staged.ops_bytes;
        self.dispatch_batch_buffer(
            total_bytes,
            staged.n_bufs,
            staged.n_tensors,
            staged.n_ops,
            prof_offset,
        )
    }

    /// Register a buffer in the batch, returning its index.
    pub fn add_buffer(&mut self, buf: &RpcmemBuffer) -> Result<u16, CeraError> {
        let fd = buf.fd();
        if let Some(idx) = self.buf_map.get(fd) {
            return Ok(idx);
        }
        let idx = Self::batch_index(self.bufs.len(), "buffers")
            .inspect_err(|_| self.drop_pending_batch())?;
        self.bufs.push(HtpBufDesc {
            base: buf.as_ptr() as u64,
            size: buf.size() as u64,
            flags: 0,
            fd: fd as u32,
        });
        self.buf_map.insert(fd, idx);
        Ok(idx)
    }

    /// Register a tensor in the batch, returning its index.
    // One arg per HtpTensor field; a builder would just move the arity.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        buf: &RpcmemBuffer,
        offset: usize,
        size: usize,
        flags: u32,
        dtype: u32,
        ne: [u32; 4],
        nb: [u32; 4],
    ) -> Result<u16, CeraError> {
        let bi = self.add_buffer(buf)?;
        let ti = Self::batch_index(self.tens.len(), "tensors")
            .inspect_err(|_| self.drop_pending_batch())?;
        // The DSP descriptor carries a u32 size; a wrapped size would reach the
        // DSP as a small tensor with no error.
        let size = u32::try_from(size).map_err(|_| {
            self.drop_pending_batch();
            CeraError::Backend(format!(
                "tensor size {size} exceeds the u32 descriptor limit"
            ))
        })?;
        self.tens.push(HtpTensor {
            data: offset as u64,
            size,
            flags,
            dtype,
            bi,
            ti,
            ne,
            nb,
        });
        Ok(ti)
    }

    /// Enqueue an operation into the current batch. Returns the capped
    /// auto-flush (or step-mode flush) error instead of panicking: the cap is
    /// set on production prefill/decode paths, and a panic at the UniFFI
    /// boundary aborts the host process.
    pub fn enqueue_op(
        &mut self,
        opcode: u32,
        src: &[u16],
        dst: &[u16],
        params: [i32; 16],
        kernel_params: [i32; 32],
    ) -> Result<(), CeraError> {
        let mut op = HtpOpDesc {
            opcode,
            flags: 0,
            params,
            kernel_params,
            src: [0xffff; 10],
            dst: [0xffff; 4],
            pad: [0; 2],
        };
        if src.len() > 10 {
            self.drop_pending_batch();
            return Err(CeraError::Backend(format!(
                "enqueue_op opcode {opcode}: too many src operands ({} > 10)",
                src.len()
            )));
        }
        if dst.len() > 4 {
            self.drop_pending_batch();
            return Err(CeraError::Backend(format!(
                "enqueue_op opcode {opcode}: too many dst operands ({} > 4)",
                dst.len()
            )));
        }
        for (i, &s) in src.iter().enumerate() {
            if s != 0xffff && (s as usize) >= self.tens.len() {
                self.drop_pending_batch();
                return Err(CeraError::Backend(format!(
                    "enqueue_op opcode {opcode}: src[{i}] index {s} out of bounds (registered {})",
                    self.tens.len()
                )));
            }
            op.src[i] = s;
        }
        for (i, &d) in dst.iter().enumerate() {
            if d != 0xffff && (d as usize) >= self.tens.len() {
                self.drop_pending_batch();
                return Err(CeraError::Backend(format!(
                    "enqueue_op opcode {opcode}: dst[{i}] index {d} out of bounds (registered {})",
                    self.tens.len()
                )));
            }
            op.dst[i] = d;
        }
        self.ops.push(op);
        if self.max_ops_per_flush.is_some_and(|m| self.ops.len() >= m) {
            self.flush().map_err(|e| {
                CeraError::Backend(format!(
                    "HTP capped flush failed (cap={:?}): {e}",
                    self.max_ops_per_flush
                ))
            })?;
        }
        if step_enabled() {
            eprintln!("[cera-hexagon] step op opcode={opcode}");
            self.flush().map_err(|e| {
                CeraError::Backend(format!("HTP step failed on opcode {opcode}: {e}"))
            })?;
        }
        Ok(())
    }

    /// Flush all queued operations in a single atomic batch execution.
    pub fn flush(&mut self) -> Result<(), CeraError> {
        self.resident_staged_id = None;
        if self.ops.is_empty() {
            return Ok(());
        }

        if debug_enabled() {
            eprintln!(
                "[cera-hexagon] flush: bufs.len={} tens.len={} ops.len={}",
                self.bufs.len(),
                self.tens.len(),
                self.ops.len()
            );
            for (i, b) in self.bufs.iter().enumerate() {
                eprintln!(
                    "  buf[{}]: fd={} size={} (0x{:x}) base=0x{:x}",
                    i, b.fd, b.size, b.size, b.base
                );
            }
            for (i, t) in self.tens.iter().enumerate() {
                eprintln!(
                    "  ten[{}]: bi={} data=0x{:x} size={} dtype={} flags={} ne={:?} nb={:?}",
                    i, t.bi, t.data, t.size, t.dtype, t.flags, t.ne, t.nb
                );
            }
            for (i, o) in self.ops.iter().enumerate() {
                eprintln!(
                    "  op[{}]: opcode={} src={:?} dst={:?} params={:?}",
                    i,
                    o.opcode,
                    o.src
                        .iter()
                        .cloned()
                        .filter(|&s| s != 0xffff)
                        .collect::<Vec<_>>(),
                    o.dst
                        .iter()
                        .cloned()
                        .filter(|&d| d != 0xffff)
                        .collect::<Vec<_>>(),
                    o.params,
                );
                eprintln!("    kparams={:?}", o.kernel_params);
            }
        }

        let bufs_bytes = self.bufs.len() * std::mem::size_of::<HtpBufDesc>();
        let tens_bytes = self.tens.len() * std::mem::size_of::<HtpTensor>();
        let ops_bytes = self.ops.len() * std::mem::size_of::<HtpOpDesc>();
        let prof_bytes = self.ops.len() * std::mem::size_of::<HtpProfDesc>();
        let total_bytes = bufs_bytes + tens_bytes + ops_bytes + prof_bytes;

        if total_bytes > self.staging_buf.size() {
            self.drop_pending_batch();
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_buf.size()
            )));
        }

        unsafe {
            let base = self.staging_buf.as_mut_ptr();
            std::ptr::copy_nonoverlapping(self.bufs.as_ptr() as *const u8, base, bufs_bytes);
            std::ptr::copy_nonoverlapping(
                self.tens.as_ptr() as *const u8,
                base.add(bufs_bytes),
                tens_bytes,
            );
            std::ptr::copy_nonoverlapping(
                self.ops.as_ptr() as *const u8,
                base.add(bufs_bytes + tens_bytes),
                ops_bytes,
            );
            std::ptr::write_bytes(base.add(bufs_bytes + tens_bytes + ops_bytes), 0, prof_bytes);
        }

        let n_bufs = self.bufs.len() as u32;
        let n_tensors = self.tens.len() as u32;
        let n_ops = self.ops.len() as u32;
        let prof_offset = bufs_bytes + tens_bytes + ops_bytes;

        // Attempts are single-shot: drop the pending batch whether dispatch
        // succeeded or failed.
        self.drop_pending_batch();

        self.dispatch_batch_buffer(total_bytes, n_bufs, n_tensors, n_ops, prof_offset)
    }

    /// Internal batch submission and response polling helper shared by `flush` and `flush_staged`.
    fn dispatch_batch_buffer(
        &mut self,
        total_bytes: usize,
        n_bufs: u32,
        n_tensors: u32,
        n_ops: u32,
        prof_offset: usize,
    ) -> Result<(), CeraError> {
        self.staging_buf.flush_cpu_cache(0, total_bytes);

        let dbuf = DspQueueBuffer {
            fd: self.staging_buf.fd() as u32,
            size: total_bytes as u32,
            offset: 0,
            flags: DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT,
            ptr: self.staging_buf.as_mut_ptr() as *mut std::ffi::c_void,
        };

        let req = HtpOpBatchReq {
            seq: self.seq,
            n_bufs,
            n_tensors,
            n_ops,
            n_traces: 0,
        };

        let req_bytes = unsafe {
            std::slice::from_raw_parts(
                &req as *const _ as *const u8,
                std::mem::size_of::<HtpOpBatchReq>(),
            )
        };

        let host_start = std::time::Instant::now();
        let write_res = self.driver.write_dsp_queue(self.queue, &[dbuf], req_bytes);

        let mut rsp = HtpOpBatchRsp::default();
        let mut resp_bufs = [DspQueueBuffer::default(); 1];

        // Drain stale responses from prior timed-out batches, then require
        // the response for THIS batch. In blocking mode a 30s timeout
        // returns Err while the DSP may still complete the batch later; its
        // response then sits in the queue, and without the drain the next
        // flush would consume it as its own, permanently desyncing status
        // attribution by one batch.
        let mut read_res: Result<(), CeraError> = Ok(());
        if write_res.is_ok() {
            let mut stale_drained = 0u32;
            loop {
                let rsp_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut rsp as *mut _ as *mut u8,
                        std::mem::size_of::<HtpOpBatchRsp>(),
                    )
                };
                match self
                    .driver
                    .read_dsp_queue(self.queue, &mut resp_bufs, rsp_bytes)
                {
                    Err(e) => {
                        read_res = Err(e);
                        break;
                    }
                    Ok(n_read) => {
                        if (n_read as usize) < std::mem::size_of::<HtpOpBatchRsp>() {
                            read_res = Err(CeraError::Backend(format!(
                                "DSP queue response truncated: got {n_read} bytes, expected at least {}",
                                std::mem::size_of::<HtpOpBatchRsp>()
                            )));
                            break;
                        }
                        match stale_drain_action(rsp.seq, self.seq, stale_drained) {
                            StaleDrainAction::DrainStale => {
                                stale_drained += 1;
                                tracing::warn!(
                                    stale_seq = rsp.seq,
                                    expected_seq = self.seq,
                                    "drained stale DSP queue response"
                                );
                                // No `tracing` subscriber on the shipping NPU
                                // platforms; without this the drain is silent
                                // exactly where field debugging needs it.
                                eprintln!(
                                    "[cera-hexagon] drained stale DSP queue response \
                                     (stale seq {}, expected {})",
                                    rsp.seq, self.seq
                                );
                                continue;
                            }
                            StaleDrainAction::CapExceeded => {
                                stale_drained += 1;
                                read_res = Err(CeraError::Backend(format!(
                                    "DSP queue returned {stale_drained} consecutive stale \
                                     responses (expected seq {})",
                                    self.seq
                                )));
                                break;
                            }
                            StaleDrainAction::Current => break,
                            StaleDrainAction::Future => {
                                read_res = Err(CeraError::Backend(format!(
                                    "DSP queue sequence mismatch (expected {}, got {})",
                                    self.seq, rsp.seq
                                )));
                                break;
                            }
                        }
                    }
                }
            }
        }

        if profile_enabled() && total_bytes > prof_offset {
            self.staging_buf
                .invalidate_cpu_cache(prof_offset, total_bytes - prof_offset);
        }

        // Attempts are single-shot: advance `seq` whether
        // this attempt succeeded or failed.
        self.seq += 1;

        write_res?;
        read_res?;

        if rsp.status != HtpStatus::Ok as u32 {
            let status_detail = HtpStatus::from_u32(rsp.status)
                .map(|s| format!("{s:?} ({})", rsp.status))
                .unwrap_or_else(|| rsp.status.to_string());
            return Err(CeraError::Backend(format!(
                "HTP batch seq {} failed with status {status_detail}",
                rsp.seq
            )));
        }

        if profile_enabled() {
            self.record_profile(
                rsp.seq,
                n_ops as usize,
                rsp.usecs,
                rsp.cycles_stop.saturating_sub(rsp.cycles_start),
                host_start.elapsed(),
                prof_offset,
            );
        }

        Ok(())
    }

    /// Aggregate DSP-side per-op timing from the profile descriptors the
    /// firmware wrote into staging, and log a one-line batch summary.
    fn record_profile(
        &mut self,
        batch_seq: u64,
        n_ops: usize,
        batch_usecs: u32,
        batch_cycles: u64,
        host_elapsed: std::time::Duration,
        prof_offset: usize,
    ) {
        let prof_size = std::mem::size_of::<HtpProfDesc>();
        let mut dsp_total: u64 = 0;
        static PROFILE_OPS_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let per_op = *PROFILE_OPS_ENABLED
            .get_or_init(|| std::env::var_os("CERA_HEXAGON_PROFILE_OPS").is_some());
        for i in 0..n_ops {
            let desc = unsafe {
                (self.staging_buf.as_ptr().add(prof_offset + i * prof_size) as *const HtpProfDesc)
                    .read_unaligned()
            };
            let e = self.prof.entry(desc.opcode).or_insert((0, 0));
            e.0 += 1;
            e.1 += desc.usecs as u64;
            dsp_total += desc.usecs as u64;
            if per_op {
                eprintln!(
                    "[cera-hexagon] profile-op: seq={} idx={} op={} {} usec={}",
                    batch_seq,
                    i,
                    desc.opcode,
                    htp_opcode_name(desc.opcode),
                    desc.usecs,
                );
            }
        }
        self.prof_flushes += 1;
        self.prof_host_us += host_elapsed.as_micros() as u64;
        self.prof_dsp_us += batch_usecs as u64;
        let mhz = if batch_usecs > 0 {
            batch_cycles as f64 / batch_usecs as f64
        } else {
            0.0
        };
        eprintln!(
            "[cera-hexagon] profile: seq={} ops={} dsp_op_us={} dsp_batch_us={} host_us={} mhz={:.1}",
            batch_seq,
            n_ops,
            dsp_total,
            batch_usecs,
            host_elapsed.as_micros(),
            mhz,
        );
    }
}

impl Drop for HexagonQueueSession {
    fn drop(&mut self) {
        // Do not attempt to flush uncommitted batches on drop: if the device
        // session was already stopped, flushing will hang for 30 seconds
        // awaiting a response that will never arrive. Discard pending work instead.
        if !self.ops.is_empty() {
            eprintln!(
                "[cera-hexagon] dropping queue session with {} uncommitted ops; discarding batch",
                self.ops.len()
            );
            self.drop_pending_batch();
        }
        if let Some(rtt) = self
            .prof_host_us
            .saturating_sub(self.prof_dsp_us)
            .checked_div(self.prof_flushes)
        {
            eprintln!(
                "[cera-hexagon] profile: {} flushes, host_total={}us dsp_total={}us mean_rtt={}us",
                self.prof_flushes, self.prof_host_us, self.prof_dsp_us, rtt
            );
            // Per-op descs are zero unless the DSP profiler was enabled
            // (`HexagonDevice` enables it under `CERA_HEXAGON_PROFILE`); the
            // batch totals above stay valid either way.
            if self.prof.values().any(|&(_, us)| us > 0) {
                let mut rows: Vec<(u32, (u64, u64))> =
                    self.prof.iter().map(|(&k, &v)| (k, v)).collect();
                rows.sort_by_key(|&(_, (_, us))| std::cmp::Reverse(us));
                eprintln!(
                    "[cera-hexagon] profile: {:>5} {:>14} {:>8} {:>10} {:>10}",
                    "op", "name", "count", "total_us", "mean_us"
                );
                for (op, (count, us)) in rows {
                    eprintln!(
                        "[cera-hexagon] profile: {:>5} {:>14} {:>8} {:>10} {:>10}",
                        op,
                        htp_opcode_name(op),
                        count,
                        us,
                        us / count.max(1)
                    );
                }
            }
        }
        self.driver.close_dsp_queue(self.queue);
    }
}

/// Short opcode names for profile tables. Discriminants are firmware-ABI
/// pinned (see `HtpOpCode`); unknown ids print numerically via the `op` column.
fn htp_opcode_name(opcode: u32) -> &'static str {
    HtpOpCode::from_u32(opcode).map_or("unknown", |op| op.name())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch shaped like `export_staged_batch` output: `n_tensors`
    /// tensors and `n_ops` ops, no buffers.
    fn staged(n_tensors: u32, n_ops: u32) -> StagedBatch {
        let tens_bytes = n_tensors as usize * std::mem::size_of::<HtpTensor>();
        let ops_bytes = n_ops as usize * std::mem::size_of::<HtpOpDesc>();
        let prof_bytes = n_ops as usize * std::mem::size_of::<HtpProfDesc>();
        let total_bytes = tens_bytes + ops_bytes + prof_bytes;
        StagedBatch {
            raw_bytes: vec![0u8; total_bytes],
            n_bufs: 0,
            n_tensors,
            n_ops,
            bufs_bytes: 0,
            tens_bytes,
            ops_bytes,
            prof_bytes,
            total_bytes,
        }
    }

    #[test]
    fn staged_batch_validate_accepts_consistent_and_rejects_lies() {
        assert!(staged(3, 2).validate().is_ok());
        let mut b = staged(3, 2);
        b.total_bytes += 1;
        assert!(b.validate().is_err(), "total disagrees with the sections");
        let mut b = staged(3, 2);
        b.raw_bytes.pop();
        assert!(b.validate().is_err(), "raw_bytes shorter than total");
        let mut b = staged(3, 2);
        b.total_bytes -= 1;
        assert!(
            b.validate().is_err(),
            "total below the sections (raw stays longer)"
        );
        let mut b = staged(3, 2);
        b.n_tensors = 1000;
        assert!(b.validate().is_err(), "count larger than its section");
        let mut b = staged(3, 2);
        b.n_ops = 5;
        assert!(
            b.validate().is_err(),
            "op count disagrees with ops and prof bytes"
        );
        let mut b = staged(3, 2);
        b.n_bufs = 1;
        assert!(b.validate().is_err(), "buffer count with no buffer bytes");
        let mut b = staged(3, 2);
        b.tens_bytes = usize::MAX;
        assert!(b.validate().is_err(), "overflowing section size");
    }

    #[test]
    fn staged_descriptor_access_is_bounds_checked() {
        let mut b = staged(2, 1);
        b.update_tensor(1, |t| t.ne[0] = 7);
        assert_eq!(b.tensor(1).ne[0], 7);
        assert_eq!(b.tensor(0).ne[0], 0, "neighbouring descriptor untouched");
        b.update_op(0, |o| o.kernel_params[2] = 9);
        assert_eq!(b.op(0).kernel_params[2], 9);
    }

    /// A section that claims more descriptors than the bytes hold trips the
    /// range check instead of reading past `raw_bytes`.
    #[test]
    #[should_panic(expected = "overruns staged batch")]
    fn staged_descriptor_overrun_panics() {
        let mut short = staged(2, 1);
        short.raw_bytes.truncate(std::mem::size_of::<HtpTensor>());
        short.update_tensor(1, |t| t.ne[0] = 1);
    }

    /// Descriptors sit at a deliberately odd offset (1-byte buffer section),
    /// so a regression to aligned `&mut T` access would be misaligned UB.
    #[test]
    fn staged_descriptor_access_is_unaligned_safe() {
        let mut b = staged(2, 1);
        b.raw_bytes.insert(0, 0);
        b.bufs_bytes = 1;
        b.total_bytes += 1;
        let addr = b.raw_bytes.as_ptr() as usize + b.tensor_offset(1);
        assert_ne!(
            addr % std::mem::align_of::<HtpTensor>(),
            0,
            "fixture must place the descriptor at a misaligned address"
        );
        b.update_tensor(1, |t| t.ne[0] = 11);
        b.update_op(0, |o| o.kernel_params[1] = 13);
        assert_eq!(b.tensor(1).ne[0], 11);
        assert_eq!(b.tensor(0).ne[0], 0);
        assert_eq!(b.op(0).kernel_params[1], 13);
    }

    /// The DSP reads `u16` indices and `0xffff` is the absent-operand
    /// marker, so the top usable index is 65534 and one past it is `Err`,
    /// never a silently aliasing truncation or marker collision.
    #[test]
    fn batch_index_boundary() {
        assert_eq!(HexagonQueueSession::batch_index(0, "buffers").unwrap(), 0);
        assert_eq!(
            HexagonQueueSession::batch_index(65534, "buffers").unwrap(),
            65534
        );
        assert!(HexagonQueueSession::batch_index(65535, "buffers").is_err());
        assert!(HexagonQueueSession::batch_index(usize::MAX, "tensors").is_err());
    }

    /// The drain decision table: stale/current/future seqs plus the exact
    /// 32-cap boundary (the 33rd consecutive stale fails the flush).
    #[test]
    fn stale_drain_action_table() {
        use super::StaleDrainAction::*;
        // Stale seqs drain while under the cap...
        assert_eq!(stale_drain_action(5, 10, 0), DrainStale);
        assert_eq!(stale_drain_action(9, 10, 31), DrainStale);
        // ...and the 33rd consecutive stale (32 already drained) fails.
        assert_eq!(stale_drain_action(5, 10, 32), CapExceeded);
        assert_eq!(stale_drain_action(9, 10, u32::MAX), CapExceeded);
        // The awaited batch's own response stops the drain.
        assert_eq!(stale_drain_action(10, 10, 7), Current);
        assert_eq!(stale_drain_action(0, 0, 0), Current);
        // A future batch's response is a desync, never drained past.
        assert_eq!(stale_drain_action(11, 10, 7), Future);
        assert_eq!(stale_drain_action(u64::MAX, 10, 0), Future);
    }

    #[test]
    fn test_htp_opcode_name() {
        assert_eq!(htp_opcode_name(HtpOpCode::Mul as u32), "Mul");
        assert_eq!(htp_opcode_name(HtpOpCode::Norm as u32), "Norm");
        assert_eq!(htp_opcode_name(HtpOpCode::Clamp as u32), "Clamp");
        assert_eq!(htp_opcode_name(HtpOpCode::Conv1D as u32), "Conv1D");
        assert_eq!(htp_opcode_name(HtpOpCode::UnarySnake as u32), "UnarySnake");
        assert_eq!(htp_opcode_name(HtpOpCode::UnarySin as u32), "UnarySin");
        assert_eq!(htp_opcode_name(HtpOpCode::UnaryCos as u32), "UnaryCos");
        assert_eq!(
            htp_opcode_name(HtpOpCode::ConvTranspose1D as u32),
            "ConvTranspose1D"
        );
        assert_eq!(
            htp_opcode_name(HtpOpCode::UnaryHardSigmoid as u32),
            "UnaryHardSigmoid"
        );
        assert_eq!(
            htp_opcode_name(HtpOpCode::UnaryHardSwish as u32),
            "UnaryHardSwish"
        );
        assert_eq!(htp_opcode_name(HtpOpCode::UnaryElu as u32), "UnaryElu");
        assert_eq!(htp_opcode_name(9999), "unknown");
    }
}

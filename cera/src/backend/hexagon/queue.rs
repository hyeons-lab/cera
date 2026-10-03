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

/// `CERA_HEXAGON_DEBUG` set, read once.
pub(crate) fn debug_enabled() -> bool {
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
    /// The skel handle of the device this queue belongs to, for `htp_iface`
    /// calls that name a buffer (set by `HexagonDevice`).
    skel_handle: Option<crate::backend::hexagon::sys::RemoteHandle64>,
    #[cfg(test)]
    staging_cap_override: Option<usize>,
    queue: crate::backend::hexagon::sys::DspQueueHandle,
    queue_id: u64,
    staging_buf: RpcmemBuffer,
    bufs: Vec<HtpBufDesc>,
    buf_map: BufferIndexMap,
    tens: Vec<HtpTensor>,
    ops: Vec<HtpOpDesc>,
    /// Auto-flush once this many ops are queued (`None` = unbounded).
    /// Small-M prefill chunks cap this (see `MAX_OPS_PER_FLUSH`):
    /// large single-flush batches compute nondeterministically there.
    max_ops_per_flush: Option<usize>,
    /// Flush at an op-group boundary once the batch holds this many tensors
    /// (`None` = unbounded). On the S25 Ultra a decode batch of a few dozen
    /// tensors or more computes nondeterministically (logit swings up to 3, a
    /// different result on every run); see [`Self::set_max_tensors_per_flush`].
    max_tensors_per_flush: Option<usize>,
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
    /// Sleep in the kernel for each batch response instead of polling for it,
    /// whatever the driver's default. Polling wakes sooner but keeps a host
    /// core busy for the whole batch; a queue running long, latency-tolerant
    /// batches (the audio encoder in a background task) would rather free it.
    blocking_wait: bool,
    /// Flush after every op (`CERA_HEXAGON_STEP`). A field, not a per-call
    /// env read, so tests can drive the step-mode error path.
    step_mode: bool,
    /// Batches written to the DSP whose response has not been read back yet.
    /// A read timeout leaves the batch outstanding: the DSP may still be
    /// running it, and writing the buffers it targets (KV / recurrent state)
    /// is unsafe until [`Self::quiesce`] has seen its response.
    outstanding: u64,
    /// Test hook: fail `add_tensor` once this many tensors are registered,
    /// the way a real over-full batch does, before anything reaches the DSP.
    #[cfg(test)]
    tensor_cap: Option<usize>,
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
            skel_handle: None,
            #[cfg(test)]
            staging_cap_override: None,
            queue,
            queue_id,
            staging_buf,
            bufs: Vec::with_capacity(32),
            buf_map: BufferIndexMap::new(),
            tens: Vec::with_capacity(256),
            ops: Vec::with_capacity(128),
            max_ops_per_flush: None,
            max_tensors_per_flush: None,
            seq: 1,
            prof: HashMap::new(),
            prof_host_us: 0,
            prof_dsp_us: 0,
            prof_flushes: 0,
            dsp_threads: 8,
            resident_staged_id: None,
            blocking_wait: false,
            step_mode: step_enabled(),
            outstanding: 0,
            #[cfg(test)]
            tensor_cap: None,
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

    /// Cap the tensors in a batch (`None` restores unbounded batching): once the
    /// batch holds this many, the next op-group boundary ([`Self::end_group`])
    /// flushes it. Unlike [`Self::set_max_ops_per_flush`] this never cuts a
    /// helper's ops apart, which share tensor indices.
    ///
    /// Why it exists: decode on the S25 Ultra (v79) was found to compute
    /// nondeterministically when a whole token went to the DSP as one batch:
    /// every identical run gave a different logit sequence, differing by up to
    /// 3 from decode step 1, while the CPU and the NPU's prefill were bit-exact.
    /// Ending the batch at every 16 op groups or fewer gave the same logits on
    /// every run, and at 24 groups or more did not; counted in tensors, caps of
    /// 40 or fewer were reproducible on every model tried and 48 or more were
    /// not on all of them. Enlarging the DSP's dirty-range table did not change
    /// it, so the cause is elsewhere in the firmware's handling of long
    /// batches, and this is a host-side bound on that. It costs decode speed
    /// (8% to 20% on the models tried; more batches, and no resident template).
    pub fn set_max_tensors_per_flush(&mut self, max: Option<usize>) {
        self.max_tensors_per_flush = max.filter(|&m| m >= 1);
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

    /// Number of batch dispatch attempts so far; advances on success and
    /// failure. Counts only batches that reached the DSP queue (an empty flush
    /// or a batch rejected before dispatch, e.g. over the staging size, does
    /// not advance it), so a caller can tell whether a failed forward pass
    /// could have run any op on the device.
    pub fn dispatch_attempts(&self) -> u64 {
        self.seq
    }

    /// Record the skel handle `htp_iface` calls about this queue's buffers go
    /// through.
    pub(crate) fn set_skel_handle(&mut self, handle: super::sys::RemoteHandle64) {
        self.skel_handle = Some(handle);
    }

    /// [`Self::release_dsp_reference`] for each of `bufs`: what a model's
    /// `Drop` calls, while its device is still open, ahead of the unmaps that
    /// dropping the buffers performs.
    pub(crate) fn release_dsp_references<'a>(
        &self,
        bufs: impl IntoIterator<Item = &'a RpcmemBuffer>,
    ) {
        if self.outstanding > 0 {
            super::hexagon_warn!(
                "not releasing buffers: {} batch(es) unanswered",
                self.outstanding
            );
            return;
        }
        for buf in bufs {
            self.release_dsp_reference(buf);
        }
    }

    /// Tell the DSP to drop its reference to `buf` (`htp_iface_munmap`, IDL
    /// method 5). Once a batch has read a buffer the DSP keeps its own hold on
    /// it, and the host's `fastrpc_munmap` is refused (error 1) until that is
    /// released. Best effort, as in llama.cpp: a buffer no batch has touched
    /// has nothing to release.
    ///
    /// Skipped while a batch is unanswered: after a read timeout the DSP may
    /// still be running it against `buf`, and releasing would pull the
    /// mapping from under it. The host unmap that follows when the buffer
    /// drops is not prevented: if the DSP already holds the buffer it is
    /// refused (the mapping leaks), and if the batch has not taken its hold yet
    /// the memory is freed regardless, as it was before this guard.
    pub(crate) fn release_dsp_reference(&self, buf: &RpcmemBuffer) {
        let Some(handle) = self.skel_handle else {
            return;
        };
        if self.outstanding > 0 {
            return;
        }
        let mut fd = buf.fd() as u32;
        let mut args = [super::sys::RemoteArg {
            buf: super::sys::RemoteBuf {
                buf: &mut fd as *mut u32 as *mut std::ffi::c_void,
                len: std::mem::size_of::<u32>(),
            },
        }];
        // An error is expected for a buffer no batch has read; a real refusal
        // surfaces as the host unmap failing right after, with its own error.
        if let Err(e) =
            self.driver
                .invoke_skel(handle, super::sys::remote_scalars_make(5, 1, 0), &mut args)
        {
            tracing::debug!("cera::hexagon: htp_iface_munmap(fd {}): {e}", buf.fd());
        }
    }

    /// Wait until every batch written to the DSP has answered.
    ///
    /// After a read timeout the DSP may still complete the batch and write
    /// the buffers it targets, so a caller about to zero or overwrite such a
    /// buffer (a state reset) must quiesce first. Errors when a response
    /// still does not arrive (the read's own 30 s hang guard); the caller
    /// must then leave the buffers alone.
    ///
    /// Blocking: each outstanding batch can wait the full 30 s guard and the
    /// wait does not poll cancellation. Callers that quiesce twice on a hung
    /// DSP (a rewind to 0 refused, then a full reset) wait up to twice that
    /// while holding the device lock; no failure memo shortens the second wait.
    pub fn quiesce(&mut self) -> Result<(), CeraError> {
        let mut rsp = HtpOpBatchRsp::default();
        let mut resp_bufs = [DspQueueBuffer::default(); 1];
        while self.outstanding > 0 {
            let rsp_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    &mut rsp as *mut _ as *mut u8,
                    std::mem::size_of::<HtpOpBatchRsp>(),
                )
            };
            self.driver
                .read_dsp_queue_with(
                    self.queue,
                    &mut resp_bufs,
                    rsp_bytes,
                    self.spin_for_responses(),
                )
                .map_err(|e| {
                    CeraError::Backend(format!(
                        "{} DSP batch(es) still outstanding after a timeout: {e}",
                        self.outstanding
                    ))
                })?;
            self.outstanding -= 1;
        }
        Ok(())
    }

    /// Test hook: make `add_tensor` fail once `cap` tensors are registered
    /// (`None` lifts it). Fails before any flush, so nothing is dispatched.
    #[cfg(test)]
    pub(crate) fn set_tensor_cap(&mut self, cap: Option<usize>) {
        self.tensor_cap = cap;
    }

    /// Test hook: the tensors-per-flush cap currently set.
    #[cfg(test)]
    pub(crate) fn tensor_flush_cap(&self) -> Option<usize> {
        self.max_tensors_per_flush
    }

    /// Number of batches written to the DSP with no response read back yet.
    pub(crate) fn outstanding_batches(&self) -> u64 {
        self.outstanding
    }

    /// Bytes the pending batch would occupy in the staging buffer if flushed
    /// now (descriptors only).
    pub(crate) fn pending_bytes(&self) -> usize {
        self.bufs.len() * std::mem::size_of::<HtpBufDesc>()
            + self.tens.len() * std::mem::size_of::<HtpTensor>()
            + self.ops.len()
                * (std::mem::size_of::<HtpOpDesc>() + std::mem::size_of::<HtpProfDesc>())
    }

    /// Size of the staging buffer a batch must fit in.
    pub(crate) fn staging_capacity(&self) -> usize {
        #[cfg(test)]
        if let Some(cap) = self.staging_cap_override {
            return cap.min(self.staging_buf.size());
        }
        self.staging_buf.size()
    }

    /// Test hook: pretend the staging buffer is smaller, so size-driven flushes
    /// can be exercised with a model small enough to run on the host.
    #[cfg(test)]
    pub(crate) fn set_staging_capacity_for_test(&mut self, cap: Option<usize>) {
        self.staging_cap_override = cap;
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

        if total_bytes > self.staging_capacity() {
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_capacity()
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
        if total_bytes > self.staging_capacity() {
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_capacity()
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
        if total_bytes > self.staging_capacity() {
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_capacity()
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
            // The DSP wrote per-op timings into the resident profile region
            // last flush; `flush()` re-zeroes it every batch, and the patch
            // ranges never cover it, so without this `CERA_HEXAGON_PROFILE`
            // would aggregate the previous batch's stale descriptors.
            let prof_start = staged.bufs_bytes + staged.tens_bytes + staged.ops_bytes;
            let prof_end = (prof_start + staged.prof_bytes).min(total_bytes);
            if prof_start < prof_end {
                unsafe {
                    std::ptr::write_bytes(
                        self.staging_buf.as_mut_ptr().add(prof_start),
                        0,
                        prof_end - prof_start,
                    );
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
        #[cfg(test)]
        if self.tensor_cap.is_some_and(|cap| self.tens.len() >= cap) {
            self.drop_pending_batch();
            return Err(CeraError::Backend(
                "HTP batch exceeds the test tensor cap".into(),
            ));
        }
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
        Ok(())
    }

    /// Mark an op-group boundary: a run of ops that may share tensor
    /// indices (one `dispatch::*` helper, one model-local op emitter). Under
    /// `CERA_HEXAGON_STEP` this flushes the group so a DSP fault names the
    /// group that caused it. With a tensor cap ([`Self::set_max_tensors_per_flush`])
    /// it flushes once the batch has reached it; otherwise it is a no-op. A flush clears the
    /// tensor table, so it must only run where no later op reuses an index
    /// registered before the boundary. Flushing per `enqueue_op` instead
    /// (the old behavior) broke every helper that registers a tensor once
    /// and reuses it across ops.
    pub fn end_group(&mut self) -> Result<(), CeraError> {
        if !self.step_mode {
            if let Some(cap) = self.max_tensors_per_flush
                && self.tens.len() >= cap
            {
                let pending = self.tens.len();
                return self.flush().map_err(|e| {
                    CeraError::Backend(format!(
                        "HTP tensor-capped flush failed (cap={cap}, tensors={pending}): {e}"
                    ))
                });
            }
            return Ok(());
        }
        let n_ops = self.ops.len();
        eprintln!("cera-hexagon: step group ({n_ops} ops)");
        self.flush()
            .map_err(|e| CeraError::Backend(format!("HTP step failed after {n_ops} ops: {e}")))
    }

    /// Choose how this queue waits for batch responses: `true` sleeps in the
    /// kernel (no busy core; wakeup costs about a scheduler tick more), `false`
    /// follows the driver's `CERA_HEXAGON_OPPOLL` default. Returns the previous
    /// setting so a caller can restore it.
    pub fn set_blocking_wait(&mut self, blocking: bool) -> bool {
        std::mem::replace(&mut self.blocking_wait, blocking)
    }

    fn spin_for_responses(&self) -> bool {
        self.driver.polls_responses() && !self.blocking_wait
    }

    /// Whether `CERA_HEXAGON_STEP` bisect mode is active. Templated decode
    /// exports a batch that step mode would already have flushed, so callers
    /// must not take the template path while this is set.
    pub fn step_mode(&self) -> bool {
        self.step_mode
    }

    /// Test hook: force `CERA_HEXAGON_STEP` behavior without touching the
    /// process environment.
    #[cfg(test)]
    pub(crate) fn set_step_mode(&mut self, on: bool) {
        self.step_mode = on;
    }

    /// Flush all queued operations in a single atomic batch execution.
    pub fn flush(&mut self) -> Result<(), CeraError> {
        // Test-only op capture at the one choke point every flush passes
        // through (explicit, capped and step-mode alike), so goldens see
        // queue-internal auto-flushes too.
        self.resident_staged_id = None;
        if self.ops.is_empty() {
            return Ok(());
        }
        #[cfg(test)]
        super::op_capture::record(self);

        if debug_enabled() {
            eprintln!(
                "cera-hexagon: flush bufs={} tens={} ops={}",
                self.bufs.len(),
                self.tens.len(),
                self.ops.len()
            );
            for (i, b) in self.bufs.iter().enumerate() {
                eprintln!(
                    "cera-hexagon:   buf[{}]: fd={} size={} (0x{:x}) base=0x{:x}",
                    i, b.fd, b.size, b.size, b.base
                );
            }
            for (i, t) in self.tens.iter().enumerate() {
                eprintln!(
                    "cera-hexagon:   ten[{}]: bi={} data=0x{:x} size={} dtype={} flags={} ne={:?} nb={:?}",
                    i, t.bi, t.data, t.size, t.dtype, t.flags, t.ne, t.nb
                );
            }
            for (i, o) in self.ops.iter().enumerate() {
                eprintln!(
                    "cera-hexagon:   op[{}]: opcode={} src={:?} dst={:?} params={:?} kparams={:?}",
                    i,
                    o.opcode,
                    o.src
                        .iter()
                        .copied()
                        .filter(|&s| s != 0xffff)
                        .collect::<Vec<_>>(),
                    o.dst
                        .iter()
                        .copied()
                        .filter(|&d| d != 0xffff)
                        .collect::<Vec<_>>(),
                    o.params,
                    o.kernel_params,
                );
            }
        }

        let bufs_bytes = self.bufs.len() * std::mem::size_of::<HtpBufDesc>();
        let tens_bytes = self.tens.len() * std::mem::size_of::<HtpTensor>();
        let ops_bytes = self.ops.len() * std::mem::size_of::<HtpOpDesc>();
        let prof_bytes = self.ops.len() * std::mem::size_of::<HtpProfDesc>();
        let total_bytes = bufs_bytes + tens_bytes + ops_bytes + prof_bytes;
        // Row-boundary flushes in the model decide from `pending_bytes()`.
        debug_assert_eq!(total_bytes, self.pending_bytes());

        if total_bytes > self.staging_capacity() {
            self.drop_pending_batch();
            return Err(CeraError::Backend(format!(
                "HTP batch size ({total_bytes} bytes) exceeds staging buffer ({})",
                self.staging_capacity()
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
            self.outstanding += 1;
            let mut stale_drained = 0u32;
            loop {
                let rsp_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut rsp as *mut _ as *mut u8,
                        std::mem::size_of::<HtpOpBatchRsp>(),
                    )
                };
                match self.driver.read_dsp_queue_with(
                    self.queue,
                    &mut resp_bufs,
                    rsp_bytes,
                    self.spin_for_responses(),
                ) {
                    Err(e) => {
                        read_res = Err(e);
                        break;
                    }
                    Ok(n_read) => {
                        // Any message read consumed one outstanding response.
                        self.outstanding = self.outstanding.saturating_sub(1);
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
                                super::hexagon_warn!(
                                    "drained stale DSP queue response \
                                     (stale_seq={}, expected_seq={})",
                                    rsp.seq,
                                    self.seq
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
                    "cera-hexagon: profile-op: seq={} idx={} op={} {} usec={}",
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
            "cera-hexagon: profile: seq={} ops={} dsp_op_us={} dsp_batch_us={} host_us={} mhz={:.1}",
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
            super::hexagon_warn!(
                "dropping queue session with {} uncommitted ops; discarding batch",
                self.ops.len()
            );
            self.drop_pending_batch();
        }
        if self.outstanding > 0 {
            super::hexagon_warn!(
                "dropping queue session with {} unanswered batch(es)",
                self.outstanding
            );
        }
        if let Some(rtt) = self
            .prof_host_us
            .saturating_sub(self.prof_dsp_us)
            .checked_div(self.prof_flushes)
        {
            eprintln!(
                "cera-hexagon: profile: {} flushes, host_total={}us dsp_total={}us mean_rtt={}us",
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
                    "cera-hexagon: profile: {:>5} {:>14} {:>8} {:>10} {:>10}",
                    "op", "name", "count", "total_us", "mean_us"
                );
                for (op, (count, us)) in rows {
                    eprintln!(
                        "cera-hexagon: profile: {:>5} {:>14} {:>8} {:>10} {:>10}",
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
    use super::super::sys::fake;
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

    /// Hermetic: no ops-per-flush cap and no step mode, whatever the process
    /// exports (`CERA_HEXAGON_STEP` is read at session creation). Tests that
    /// want step mode set it explicitly.
    fn test_session() -> HexagonQueueSession {
        let mut q = HexagonQueueSession::new(fake::driver()).expect("fake queue session");
        q.step_mode = false;
        q.set_max_ops_per_flush(None);
        q
    }

    /// Dropping a model releases all of its buffers, each once.
    #[test]
    fn release_dsp_references_releases_every_buffer_given() {
        fake::reset();
        fake::with(|s| s.distinct_fds = true);
        let driver = fake::driver();
        let bufs: Vec<RpcmemBuffer> = (0..3)
            .map(|_| RpcmemBuffer::alloc(Arc::clone(&driver), 4096, false).unwrap())
            .collect();
        let mut q = test_session();
        q.set_skel_handle(7);
        q.release_dsp_references(&bufs);
        let released: Vec<i32> = fake::events()
            .into_iter()
            .filter_map(|e| match e {
                fake::Event::Release(fd) => Some(fd),
                _ => None,
            })
            .collect();
        assert_eq!(
            released,
            bufs.iter().map(RpcmemBuffer::fd).collect::<Vec<_>>()
        );
    }

    /// A flush the tensor cap triggers says so when it fails: the cap is the
    /// first suspect when chasing the nondeterminism it works around.
    #[test]
    fn a_capped_flush_failure_names_the_cap() {
        fake::reset();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        let mut q = test_session();
        q.set_max_tensors_per_flush(Some(1));
        fake::with(|s| s.fail_write = true);
        // The helper ends its group, which is where the cap flushes.
        let err = enqueue_with_tensor(&mut q, &buf).unwrap_err().to_string();
        assert!(err.contains("cap=1") && err.contains("tensors=1"), "{err}");
    }

    /// A batch the DSP has not answered may still be running against the
    /// buffer, so the release is withheld rather than raced.
    #[test]
    fn a_release_is_withheld_while_a_batch_is_unanswered() {
        fake::reset();
        let mut q = test_session();
        q.set_skel_handle(7);
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        let released = || fake::events().contains(&fake::Event::Release(buf.fd()));
        fake::with(|s| s.fail_read = true);
        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap_err();
        assert_eq!(q.outstanding_batches(), 1);
        // Both forms hold back: the buffer-at-a-time one is what the encoder's
        // per-call buffers and the pager use.
        q.release_dsp_reference(&buf);
        assert!(!released(), "single form: no release while unanswered");
        q.release_dsp_references([&buf]);
        assert!(!released(), "batch form: no release while unanswered");
        fake::with(|s| s.fail_read = false);
        q.quiesce().unwrap();
        q.release_dsp_reference(&buf);
        assert!(released(), "released once answered");
    }

    /// Holding back is announced once per call, not once per buffer, so a paged
    /// model's dozens of buffers do not flood the log.
    #[test]
    fn a_withheld_release_warns_once_for_all_the_buffers() {
        use crate::audio_profile::tests::warnings_of;
        fake::reset();
        fake::with(|s| s.distinct_fds = true);
        let mut q = test_session();
        q.set_skel_handle(7);
        let bufs: Vec<_> = (0..3)
            .map(|_| RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap())
            .collect();
        fake::with(|s| s.fail_read = true);
        enqueue_with_tensor(&mut q, &bufs[0]).unwrap();
        q.flush().unwrap_err();
        let warned = warnings_of(|| q.release_dsp_references(&bufs));
        let n = warned
            .iter()
            .filter(|m| m.contains("not releasing buffers"))
            .count();
        assert_eq!(n, 1, "{warned:?}");
    }

    /// The DSP is told to drop its hold on a buffer through the skel handle,
    /// and only when the session has one.
    #[test]
    fn release_dsp_reference_invokes_the_skel_for_that_buffer() {
        fake::reset();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        let mut q = test_session();
        q.release_dsp_reference(&buf);
        assert!(
            !fake::events().contains(&fake::Event::Release(buf.fd())),
            "no skel handle, nothing to call"
        );
        q.set_skel_handle(7);
        q.release_dsp_reference(&buf);
        assert!(fake::events().contains(&fake::Event::Release(buf.fd())));
        // A refusal is not fatal (a buffer no batch read has no hold), and the
        // call still reaches the skel.
        let invokes = || {
            fake::events()
                .iter()
                .filter(|e| **e == fake::Event::Invoke(5))
                .count()
        };
        let before = invokes();
        fake::with(|s| s.fail_invoke_method = Some(5));
        q.release_dsp_reference(&buf);
        assert_eq!(invokes(), before + 1);
    }

    /// Register one tensor (so `tens`/`bufs` are non-empty) then enqueue.
    fn enqueue_with_tensor(
        q: &mut HexagonQueueSession,
        buf: &RpcmemBuffer,
    ) -> Result<(), CeraError> {
        let ti = q.add_tensor(buf, 0, 64, 0, 0, [1; 4], [4; 4])?;
        q.enqueue_op(HtpOpCode::Add as u32, &[ti], &[ti], [0; 16], [0; 32])?;
        // The group boundary is where step mode flushes.
        q.end_group()
    }

    fn assert_pending_empty(q: &HexagonQueueSession, ctx: &str) {
        let (bufs, map, tens, ops) = q.export_batch();
        assert!(
            bufs.is_empty() && map.is_empty() && tens.is_empty() && ops.is_empty(),
            "{ctx}: pending batch must be dropped, not retained"
        );
        assert_eq!(q.ops_len(), 0, "{ctx}");
    }

    /// Every flush error path is single-shot: the pending batch is dropped
    /// (never half-retained), `seq` advances, and the session then accepts
    /// and completes a fresh batch. Covers the capped auto-flush in
    /// `enqueue_op` (write failure and read failure) and the step-mode flush.
    #[test]
    fn flush_errors_drop_pending_batch_and_session_recovers() {
        for (name, fail_write, step, cap) in [
            ("capped write", true, false, Some(1)),
            ("capped read", false, false, Some(1)),
            ("step write", true, true, None),
        ] {
            fake::reset();
            let mut q = test_session();
            let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
            q.set_max_ops_per_flush(cap);
            q.step_mode = step;
            fake::with(|s| {
                s.fail_write = fail_write;
                s.fail_read = !fail_write;
            });
            let err = enqueue_with_tensor(&mut q, &buf).unwrap_err();
            assert!(err.to_string().contains("failed"), "{name}: {err}");
            assert_pending_empty(&q, name);
            assert_eq!(q.seq, 2, "{name}: failed attempt still advances seq");

            fake::with(|s| {
                s.fail_write = false;
                s.fail_read = false;
            });
            q.set_max_ops_per_flush(None);
            q.step_mode = false;
            enqueue_with_tensor(&mut q, &buf).expect(name);
            assert_eq!(q.ops_len(), 1, "{name}");
            q.flush().expect(name);
            assert_pending_empty(&q, name);
            assert_eq!(q.seq, 3, "{name}");
        }
    }

    /// `dispatch_attempts` advances once per batch that reached the DSP
    /// queue, on success and on write or read failure alike, and not for an
    /// empty flush.
    #[test]
    fn dispatch_attempts_advance_on_success_and_failure() {
        fake::reset();
        let mut q = test_session();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        let start = q.dispatch_attempts();
        q.flush().unwrap();
        assert_eq!(
            q.dispatch_attempts(),
            start,
            "empty flush dispatches nothing"
        );

        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap();
        assert_eq!(q.dispatch_attempts(), start + 1, "success");

        for (n, (fail_write, fail_read)) in [(true, false), (false, true)].into_iter().enumerate() {
            fake::with(|s| {
                s.fail_write = fail_write;
                s.fail_read = fail_read;
            });
            enqueue_with_tensor(&mut q, &buf).unwrap();
            q.flush().unwrap_err();
            assert_eq!(q.dispatch_attempts(), start + 2 + n as u64, "failure {n}");
        }
    }

    /// A tensor cap flushes at the first op-group boundary at or past it, never
    /// between the ops of one group, and `None` restores unbounded batching.
    #[test]
    fn a_tensor_cap_flushes_at_group_boundaries_only() {
        fake::reset();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        let mut q = test_session();
        let writes = || {
            fake::events()
                .iter()
                .filter(|e| matches!(e, fake::Event::Write(_)))
                .count()
        };
        // Each helper call registers one tensor (a group of one op).
        q.set_max_tensors_per_flush(Some(3));
        for n in 1..=7 {
            // `enqueue_with_tensor` ends its group, which is where a cap acts.
            enqueue_with_tensor(&mut q, &buf).unwrap();
            assert_eq!(writes(), n / 3, "after group {n}");
        }
        // Unbounded again: nothing more flushes at a group boundary.
        q.set_max_tensors_per_flush(None);
        for _ in 0..5 {
            enqueue_with_tensor(&mut q, &buf).unwrap();
        }
        assert_eq!(writes(), 2);
        // A cap never splits a group: three ops over one registered tensor stay
        // together however small the cap is.
        q.set_max_tensors_per_flush(Some(1));
        let before = writes();
        let ti = q.add_tensor(&buf, 0, 64, 0, 0, [1; 4], [4; 4]).unwrap();
        for _ in 0..3 {
            q.enqueue_op(HtpOpCode::Add as u32, &[ti], &[ti], [0; 16], [0; 32])
                .unwrap();
        }
        assert_eq!(writes(), before, "no flush inside the group");
        q.end_group().unwrap();
        assert_eq!(writes(), before + 1, "flushed at its end");
    }

    /// A session on a polling driver spins for responses until it asks to
    /// sleep, and the choice is per session and restorable.
    #[test]
    fn blocking_wait_overrides_response_polling_per_session() {
        fake::reset();
        let driver = fake::driver_with_polling(true);
        let mut q = HexagonQueueSession::new(Arc::clone(&driver)).expect("fake queue session");
        q.step_mode = false;
        q.set_max_ops_per_flush(None);
        let buf = RpcmemBuffer::alloc(driver, 4096, false).unwrap();
        let last_timeout = || fake::with(|s| *s.read_timeouts.last().expect("a read"));

        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap();
        assert_eq!(last_timeout(), 0, "the driver's default is a polled read");

        assert!(!q.set_blocking_wait(true), "previous setting returned");
        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap();
        assert_eq!(
            last_timeout(),
            crate::backend::hexagon::sys::DSPQUEUE_TIMEOUT_US,
            "a blocking session sleeps in the kernel"
        );

        assert!(q.set_blocking_wait(false));
        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap();
        assert_eq!(last_timeout(), 0, "restored to the driver default");
    }

    /// A read timeout leaves the batch outstanding (the DSP may still write
    /// its buffers); `quiesce` fails while the response is missing and
    /// succeeds, consuming it, once it arrives. A write failure never
    /// reached the DSP, so it leaves nothing outstanding.
    #[test]
    fn quiesce_waits_for_outstanding_batches() {
        fake::reset();
        let mut q = test_session();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();

        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap();
        assert_eq!(q.outstanding_batches(), 0, "answered batch");
        q.quiesce().unwrap();

        fake::with(|s| s.fail_write = true);
        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap_err();
        assert_eq!(q.outstanding_batches(), 0, "write failure");
        fake::with(|s| s.fail_write = false);

        fake::with(|s| s.fail_read = true);
        enqueue_with_tensor(&mut q, &buf).unwrap();
        q.flush().unwrap_err();
        assert_eq!(q.outstanding_batches(), 1, "timed-out batch");
        let err = q.quiesce().unwrap_err();
        assert!(err.to_string().contains("outstanding"), "{err}");
        assert_eq!(q.outstanding_batches(), 1);

        fake::with(|s| s.fail_read = false);
        q.quiesce().unwrap();
        assert_eq!(q.outstanding_batches(), 0);
    }

    /// A DSP-reported batch failure also leaves nothing pending.
    #[test]
    fn flush_status_error_drops_pending_batch() {
        fake::reset();
        let mut q = test_session();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        fake::with(|s| s.rsp_status = Some(HtpStatus::InvalParams as u32));
        enqueue_with_tensor(&mut q, &buf).unwrap();
        assert!(q.flush().is_err());
        assert_pending_empty(&q, "status");
    }

    /// Resident patching must re-zero the profile-descriptor region the DSP
    /// filled last flush (as `flush()` does), or `CERA_HEXAGON_PROFILE`
    /// aggregates stale timings.
    #[test]
    fn resident_flush_rezeroes_profile_region() {
        fake::reset();
        let mut q = test_session();
        let buf = RpcmemBuffer::alloc(fake::driver(), 4096, false).unwrap();
        enqueue_with_tensor(&mut q, &buf).unwrap();
        let staged = q.export_staged_batch().unwrap();
        q.drop_pending_batch();
        assert!(staged.prof_bytes > 0);
        let prof = staged.bufs_bytes + staged.tens_bytes + staged.ops_bytes;

        q.flush_staged_resident(1, &staged, &[]).unwrap();
        // The DSP writes timings into the profile region.
        unsafe {
            std::ptr::write_bytes(
                q.staging_buf.as_mut_ptr().add(prof),
                0xAB,
                staged.prof_bytes,
            )
        };
        q.flush_staged_resident(1, &staged, &[]).unwrap();
        let region = &q.staging_buf.as_slice()[prof..prof + staged.prof_bytes];
        assert!(
            region.iter().all(|&b| b == 0),
            "stale profile descs survived"
        );
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

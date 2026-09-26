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
    HtpBufDesc, HtpOpBatchReq, HtpOpBatchRsp, HtpOpDesc, HtpProfDesc, HtpStatus, HtpTensor,
};
use crate::session::CeraError;

/// Staging buffer size for batch queue serialization (4 MB).
const STAGING_BUFFER_SIZE: usize = 4 * 1024 * 1024;

/// Cap on consecutive stale responses drained in one flush. Stale responses
/// are bounded by prior timeouts (one per timed-out batch); anything past
/// this is firmware misbehavior, and an uncapped drain would hang the flush
/// (including from `Drop`) on a stuck seq.
const MAX_CONSECUTIVE_STALE: u32 = 32;

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

/// An active DSP command queue session managing batched request and response dispatch.
pub struct HexagonQueueSession {
    driver: Arc<FastRpcDriver>,
    queue: crate::backend::hexagon::sys::DspQueueHandle,
    queue_id: u64,
    staging_buf: RpcmemBuffer,
    bufs: Vec<HtpBufDesc>,
    buf_map: HashMap<i32, u16>,
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
            buf_map: HashMap::new(),
            tens: Vec::with_capacity(256),
            ops: Vec::with_capacity(128),
            max_ops_per_flush: None,
            seq: 1,
            prof: HashMap::new(),
            prof_host_us: 0,
            prof_dsp_us: 0,
            prof_flushes: 0,
            dsp_threads: 8,
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
    fn drop_pending_batch(&mut self) {
        self.bufs.clear();
        self.buf_map.clear();
        self.tens.clear();
        self.ops.clear();
    }

    /// Register a buffer in the batch, returning its index.
    pub fn add_buffer(&mut self, buf: &RpcmemBuffer) -> Result<u16, CeraError> {
        let fd = buf.fd();
        if let Some(&idx) = self.buf_map.get(&fd) {
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
        self.tens.push(HtpTensor {
            data: offset as u64,
            size: size as u32,
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
        for (i, &s) in src.iter().enumerate().take(10) {
            op.src[i] = s;
        }
        for (i, &d) in dst.iter().enumerate().take(4) {
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
        if std::env::var_os("CERA_HEXAGON_STEP").is_some() {
            eprintln!("[cera-hexagon] step op opcode={opcode}");
            self.flush().map_err(|e| {
                CeraError::Backend(format!("HTP step failed on opcode {opcode}: {e}"))
            })?;
        }
        Ok(())
    }

    /// Flush all queued operations in a single atomic batch execution.
    pub fn flush(&mut self) -> Result<(), CeraError> {
        if self.ops.is_empty() {
            return Ok(());
        }

        if std::env::var_os("CERA_HEXAGON_DEBUG").is_some() {
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
            n_bufs: self.bufs.len() as u32,
            n_tensors: self.tens.len() as u32,
            n_ops: self.ops.len() as u32,
            n_traces: 0,
        };

        let req_bytes = unsafe {
            std::slice::from_raw_parts(
                &req as *const _ as *const u8,
                std::mem::size_of::<HtpOpBatchReq>(),
            )
        };

        let n_ops = self.ops.len();
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
                    Ok(_) => {
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

        self.staging_buf.invalidate_cpu_cache(0, total_bytes);

        // Attempts are single-shot: drop the batch and advance `seq` whether
        // this attempt succeeded or failed. Nothing ever retries (every
        // flush-error path aborts its forward), while the session outlives
        // the forward — retaining a failed batch would piggyback stale ops
        // onto the next forward's flush. And the drain above is correct only
        // if `seq` advances strictly per attempt: reusing a timed-out
        // batch's `seq` would accept its late response as the new batch's.
        self.bufs.clear();
        self.buf_map.clear();
        self.tens.clear();
        self.ops.clear();
        self.seq += 1;

        write_res?;
        read_res?;

        if rsp.status != HtpStatus::Ok as u32 {
            return Err(CeraError::Backend(format!(
                "HTP batch seq {} failed with status {}",
                rsp.seq, rsp.status
            )));
        }

        if std::env::var_os("CERA_HEXAGON_PROFILE").is_some() {
            self.record_profile(
                n_ops,
                rsp.usecs,
                rsp.cycles_stop.saturating_sub(rsp.cycles_start),
                host_start.elapsed(),
                bufs_bytes + tens_bytes + ops_bytes,
            );
        }

        Ok(())
    }

    /// Aggregate DSP-side per-op timing from the profile descriptors the
    /// firmware wrote into staging, and log a one-line batch summary.
    fn record_profile(
        &mut self,
        n_ops: usize,
        batch_usecs: u32,
        batch_cycles: u64,
        host_elapsed: std::time::Duration,
        prof_offset: usize,
    ) {
        let prof_size = std::mem::size_of::<HtpProfDesc>();
        let mut dsp_total: u64 = 0;
        let per_op = std::env::var_os("CERA_HEXAGON_PROFILE_OPS").is_some();
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
                    self.seq,
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
            self.seq,
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
        // Unpropagatable from `Drop`, but worth one line: `flush` returns
        // `Ok` when idle, so normal shutdown never reaches the `eprintln`
        // anyway — only a genuine mid-batch death prints, ungated.
        if let Err(e) = self.flush() {
            eprintln!("[cera-hexagon] drop flush failed: {e}");
        }
        if std::env::var_os("CERA_HEXAGON_PROFILE").is_some() && self.prof_flushes > 0 {
            let rtt = self.prof_host_us.saturating_sub(self.prof_dsp_us) / self.prof_flushes;
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
    match opcode {
        0 => "Mul",
        1 => "Add",
        4 => "MulMat",
        6 => "MulMatNx",
        8 => "MulMatAdd",
        9 => "RmsNorm",
        10 => "RmsNormMul",
        21 => "GluSwiglu",
        27 => "Rope",
        28 => "FlashAttnExt",
        29 => "SetRows",
        30 => "GetRows",
        31 => "Scale",
        32 => "Cpy",
        39 => "SsmConv",
        50 => "Concat",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

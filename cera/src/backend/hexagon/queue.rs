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

    /// Register a buffer in the batch, returning its index.
    pub fn add_buffer(&mut self, buf: &RpcmemBuffer) -> u16 {
        let fd = buf.fd();
        if let Some(&idx) = self.buf_map.get(&fd) {
            return idx;
        }
        let idx = self.bufs.len() as u16;
        self.bufs.push(HtpBufDesc {
            base: buf.as_ptr() as u64,
            size: buf.size() as u64,
            flags: 0,
            fd: fd as u32,
        });
        self.buf_map.insert(fd, idx);
        idx
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
    ) -> u16 {
        let bi = self.add_buffer(buf);
        let ti = self.tens.len() as u16;
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
        ti
    }

    /// Enqueue an operation into the current batch.
    pub fn enqueue_op(
        &mut self,
        opcode: u32,
        src: &[u16],
        dst: &[u16],
        params: [i32; 16],
        kernel_params: [i32; 32],
    ) {
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
        if self.max_ops_per_flush.is_some_and(|m| self.ops.len() >= m)
            && let Err(e) = self.flush()
        {
            panic!(
                "HTP capped flush failed (cap={:?}): {e}",
                self.max_ops_per_flush
            );
        }
        if std::env::var_os("CERA_HEXAGON_STEP").is_some() {
            eprintln!("[cera-hexagon] step op opcode={opcode}");
            if let Err(e) = self.flush() {
                eprintln!("[cera-hexagon] step op opcode={opcode} failed: {e}");
                panic!("HTP step failed on opcode {opcode}: {e}");
            }
        }
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
        let rsp_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut rsp as *mut _ as *mut u8,
                std::mem::size_of::<HtpOpBatchRsp>(),
            )
        };

        let read_res = if write_res.is_ok() {
            self.driver
                .read_dsp_queue(self.queue, &mut resp_bufs, rsp_bytes)
        } else {
            Ok(0)
        };

        self.staging_buf.invalidate_cpu_cache(0, total_bytes);

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
        let _ = self.flush();
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

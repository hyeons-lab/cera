//! Batch assembly and command queue execution for Hexagon HTP.
//!
//! Operations are batched on the host into a single contiguous shared memory
//! descriptor buffer (`[htp_buf_desc][htp_tensor][htp_op_desc]`) and dispatched
//! asynchronously to the DSP via `dspqueue_write`.

use std::sync::Arc;

use super::rpcmem::RpcmemBuffer;
use super::sys::{DspQueueHandle, FastRpcDriver};
use super::types::*;
use crate::session::CeraError;

/// An execution batch containing mapped buffers, tensor descriptors, and operations.
pub struct HexagonOpBatch {
    bufs: Vec<HtpBufDesc>,
    tensors: Vec<HtpTensor>,
    ops: Vec<HtpOpDesc>,
}

impl Default for HexagonOpBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl HexagonOpBatch {
    pub fn new() -> Self {
        Self {
            bufs: Vec::with_capacity(8),
            tensors: Vec::with_capacity(64),
            ops: Vec::with_capacity(32),
        }
    }

    /// Clear all recorded operations and descriptors for reuse.
    pub fn reset(&mut self) {
        self.bufs.clear();
        self.tensors.clear();
        self.ops.clear();
    }

    /// Register a shared memory buffer with this batch.
    pub fn add_buffer(&mut self, base: u64, size: u64, flags: u32, fd: u32) -> u16 {
        let bi = self.bufs.len() as u16;
        self.bufs.push(HtpBufDesc {
            base,
            size,
            flags,
            fd,
        });
        bi
    }

    /// Register an RpcmemBuffer with this batch.
    pub fn add_rpcmem_buffer(&mut self, buf: &RpcmemBuffer, flags: u32) -> u16 {
        self.add_buffer(buf.base(), buf.size() as u64, flags, buf.fd() as u32)
    }

    /// Register a tensor descriptor with this batch.
    pub fn add_tensor(&mut self, mut tensor: HtpTensor) -> u16 {
        let ti = self.tensors.len() as u16;
        tensor.ti = ti;
        self.tensors.push(tensor);
        ti
    }

    /// Register an operation to be executed on the DSP.
    pub fn add_op(&mut self, op: HtpOpDesc) {
        self.ops.push(op);
    }

    /// Calculate the byte size needed to serialize this batch descriptor buffer.
    pub fn serialized_size(&self) -> usize {
        let b_size = std::mem::size_of::<HtpBufDesc>() * self.bufs.len();
        let t_size = std::mem::size_of::<HtpTensor>() * self.tensors.len();
        let o_size = std::mem::size_of::<HtpOpDesc>() * self.ops.len();
        b_size + t_size + o_size
    }

    /// Pack batch descriptors into a destination byte buffer.
    pub fn serialize_into(
        &self,
        seq: u64,
        req: &mut HtpOpBatchReq,
        dst: &mut [u8],
    ) -> Result<usize, CeraError> {
        let required_size = self.serialized_size();
        if dst.len() < required_size {
            return Err(CeraError::Backend(format!(
                "queue buffer too small (needed {} bytes, available {})",
                required_size,
                dst.len()
            )));
        }

        req.seq = seq;
        req.flags = 0;
        req.n_bufs = self.bufs.len() as u32;
        req.n_tensors = self.tensors.len() as u32;
        req.n_ops = self.ops.len() as u32;

        let mut offset = 0;

        // Copy buffer descriptors
        let b_bytes = self.bufs.len() * std::mem::size_of::<HtpBufDesc>();
        if b_bytes > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.bufs.as_ptr() as *const u8,
                    dst.as_mut_ptr().add(offset),
                    b_bytes,
                );
            }
            offset += b_bytes;
        }

        // Copy tensor descriptors
        let t_bytes = self.tensors.len() * std::mem::size_of::<HtpTensor>();
        if t_bytes > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.tensors.as_ptr() as *const u8,
                    dst.as_mut_ptr().add(offset),
                    t_bytes,
                );
            }
            offset += t_bytes;
        }

        // Copy op descriptors
        let o_bytes = self.ops.len() * std::mem::size_of::<HtpOpDesc>();
        if o_bytes > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ops.as_ptr() as *const u8,
                    dst.as_mut_ptr().add(offset),
                    o_bytes,
                );
            }
            offset += o_bytes;
        }

        Ok(offset)
    }
}

/// An active DSP command queue session managing async request and response dispatch.
pub struct HexagonQueueSession {
    driver: Arc<FastRpcDriver>,
    queue: DspQueueHandle,
    queue_id: u64,
    staging_buf: RpcmemBuffer,
}

// Queue session operations are Send across threads when guarded by model session locks.
unsafe impl Send for HexagonQueueSession {}

impl HexagonQueueSession {
    /// Create a new command queue session with an allocated staging descriptor buffer.
    pub fn new(driver: Arc<FastRpcDriver>, staging_size: usize) -> Result<Self, CeraError> {
        let queue = driver.create_dsp_queue(staging_size as u32, 64 * 1024)?;
        let queue_id = match driver.export_dsp_queue(queue) {
            Ok(id) => id,
            Err(e) => {
                driver.close_dsp_queue(queue);
                return Err(e);
            }
        };

        let staging_buf = match RpcmemBuffer::alloc(Arc::clone(&driver), staging_size, true) {
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
        })
    }

    /// Underlying exported DSP queue identifier for registration with `htp_iface_start`.
    pub fn queue_id(&self) -> u64 {
        self.queue_id
    }

    /// Submit a batch of operations to the DSP for asynchronous execution.
    pub fn submit(&mut self, batch: &HexagonOpBatch, seq: u64) -> Result<(), CeraError> {
        let mut req = HtpOpBatchReq::default();
        let payload_len = batch.serialize_into(seq, &mut req, self.staging_buf.as_mut_slice())?;
        self.staging_buf.flush_cpu_cache(0, payload_len);

        let qbuf = DspQueueBuffer {
            ptr: self.staging_buf.as_mut_ptr(),
            size: payload_len as u32,
            flags: 0,
        };

        let req_bytes = unsafe {
            std::slice::from_raw_parts(
                &req as *const HtpOpBatchReq as *const u8,
                std::mem::size_of::<HtpOpBatchReq>(),
            )
        };

        self.driver.write_dsp_queue(self.queue, &[qbuf], req_bytes)
    }

    /// Poll for batch completion and verify execution status.
    pub fn wait_completion(&mut self, expected_seq: u64) -> Result<HtpOpBatchRsp, CeraError> {
        let mut rsp = HtpOpBatchRsp::default();
        let mut qbufs = [DspQueueBuffer::default()];

        // Drain any stale responses from prior timed-out or canceled batches
        loop {
            let rsp_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    &mut rsp as *mut HtpOpBatchRsp as *mut u8,
                    std::mem::size_of::<HtpOpBatchRsp>(),
                )
            };

            self.driver
                .read_dsp_queue(self.queue, &mut qbufs, rsp_bytes)?;

            if rsp.seq < expected_seq {
                tracing::warn!(
                    stale_seq = rsp.seq,
                    expected_seq,
                    "drained stale DSP queue response"
                );
                continue;
            }

            if rsp.seq != expected_seq {
                return Err(CeraError::Backend(format!(
                    "DSP queue sequence mismatch (expected {}, got {})",
                    expected_seq, rsp.seq
                )));
            }

            break;
        }

        if rsp.status != HtpStatus::Ok as u32 {
            return Err(CeraError::Backend(format!(
                "DSP execution failed with status code {}",
                rsp.status
            )));
        }

        Ok(rsp)
    }
}

impl Drop for HexagonQueueSession {
    fn drop(&mut self) {
        self.driver.close_dsp_queue(self.queue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_serialization() {
        let mut batch = HexagonOpBatch::new();

        let bi0 = batch.add_buffer(0x1000, 4096, 0, 7);
        let bi1 = batch.add_buffer(0x2000, 8192, 0, 8);
        assert_eq!(bi0, 0);
        assert_eq!(bi1, 1);

        let ti0 = batch.add_tensor(HtpTensor {
            data: 0,
            size: 1024,
            flags: 0,
            dtype: HtpDataType::F32 as u32,
            bi: bi0,
            ti: 0,
            ne: [256, 1, 1, 1],
            nb: [4, 1024, 1024, 1024],
        });
        assert_eq!(ti0, 0);

        let mut op = HtpOpDesc::default();
        op.opcode = HtpOpCode::RmsNorm as u32;
        op.src[0] = ti0;
        op.dst[0] = ti0;
        batch.add_op(op);

        let expected_size = 2 * std::mem::size_of::<HtpBufDesc>()
            + std::mem::size_of::<HtpTensor>()
            + std::mem::size_of::<HtpOpDesc>();
        assert_eq!(batch.serialized_size(), expected_size);

        let mut req = HtpOpBatchReq::default();
        let mut buffer = vec![0u8; expected_size];
        let written = batch.serialize_into(42, &mut req, &mut buffer).unwrap();
        assert_eq!(written, expected_size);
        assert_eq!(req.seq, 42);
        assert_eq!(req.n_bufs, 2);
        assert_eq!(req.n_tensors, 1);
        assert_eq!(req.n_ops, 1);

        // Fail when buffer is too small
        let mut short_buf = vec![0u8; expected_size - 1];
        assert!(batch.serialize_into(43, &mut req, &mut short_buf).is_err());
    }
}

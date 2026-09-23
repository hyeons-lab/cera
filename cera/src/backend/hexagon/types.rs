//! Qualcomm Hexagon Tensor Processor (HTP) type definitions and C-ABI layouts.
//!
//! Provides ABI-compatible memory structures matching `libggml-htp`
//! on Snapdragon Compute DSPs.

use std::ffi::c_void;

/// Execution status returned by the HTP runtime.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtpStatus {
    Ok = 1,
    InternalErr = 2,
    NoSupport = 3,
    InvalParams = 4,
    VtcmTooSmall = 5,
}

impl HtpStatus {
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            1 => Some(Self::Ok),
            2 => Some(Self::InternalErr),
            3 => Some(Self::NoSupport),
            4 => Some(Self::InvalParams),
            5 => Some(Self::VtcmTooSmall),
            _ => None,
        }
    }
}

/// Data types understood by the HTP kernels.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtpDataType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    // K-quants share the Q4_1/Q6_K tiled wire layouts (`HTP_TYPE_Q4_K=12`,
    // `HTP_TYPE_Q6_K=14`). There is no Q5_K wire type; Q5_K weights are
    // requanted to Q8_0 on the host at load.
    Q4K = 12,
    Q6K = 14,
    Iq4Nl = 20,
    I32 = 26,
    I64 = 27,
    Mxfp4 = 39,

    Invalid = 0xFFFF_FFFF,
}

/// Operation codes dispatched to the DSP execution queue.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtpOpCode {
    Mul = 0,
    Add = 1,
    Sub = 2,
    Div = 3,
    MulMat = 4,
    MulMatId = 5,
    MulMatNx = 6,
    MulMatIdNx = 7,
    MulMatAdd = 8,
    RmsNorm = 9,
    RmsNormMul = 10,
    UnarySilu = 11,
    UnaryGelu = 12,
    UnarySigmoid = 13,
    UnaryExp = 14,
    UnaryNeg = 15,
    UnarySoftplus = 16,
    UnaryTanh = 17,
    UnaryAbs = 18,
    UnaryLog = 19,
    UnaryRelu = 20,
    GluSwiglu = 21,
    GluSwigluOai = 22,
    GluGeglu = 23,
    GluGegluQuick = 24,
    Softmax = 25,
    AddId = 26,
    Rope = 27,
    FlashAttnExt = 28,
    SetRows = 29,
    GetRows = 30,
    Scale = 31,
    Cpy = 32,
    CpyFence = 33,
    Argsort = 34,
    TopK = 35,
    Sqr = 36,
    Sqrt = 37,
    SumRows = 38,
    SsmConv = 39,
    Repeat = 40,
    Cumsum = 41,
    Fill = 42,
    Diag = 43,
    SolveTri = 44,
    L2Norm = 45,
    GatedDeltaNet = 46,
    Tri = 47,
    Pad = 48,
    Norm = 49,
    Concat = 50,
    Clamp = 51,

    Invalid = 0xFFFF_FFFF,
}

/// Compute (non-weight) tensor: flags 0, so the DSP tracks its dirty ranges
/// and keeps intra-batch producer/consumer edges coherent. Upstream defines
/// no COMPUTE bit; bit 0 is WEIGHT (read-only, skipped by dirty tracking).
pub const HTP_TENSOR_COMPUTE: u32 = 0;
/// Tensor holds read-only model weight data (skipped by dirty tracking).
pub const HTP_TENSOR_WEIGHT: u32 = 1;
/// Tensor is in repacked tiled format.
pub const HTP_TENSOR_REPACK: u32 = 2;
/// Tensor is a synchronization fence (explicitly managed).
pub const HTP_TENSOR_FENCE: u32 = 4;
pub const HTP_TENSOR_DIRTY: u32 = 2;

pub const HTP_MAX_DIMS: usize = 4;
pub const HTP_MAX_OP_PARAMS: usize = 16;
pub const HTP_MAX_PACKET_BUFFERS: usize = 8;

/// Buffer in FastRPC batch descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HtpBufDesc {
    pub base: u64,
    pub size: u64,
    pub flags: u32,
    pub fd: u32,
}

/// Tensor descriptor in FastRPC batch descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HtpTensor {
    pub data: u64,
    pub size: u32,
    pub flags: u32,
    pub dtype: u32,
    pub bi: u16,
    pub ti: u16,
    pub ne: [u32; 4],
    pub nb: [u32; 4],
}

/// Operation descriptor in FastRPC batch descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HtpOpDesc {
    pub opcode: u32,
    pub flags: u32,
    pub params: [i32; 16],
    pub kernel_params: [i32; 32],
    pub src: [u16; 10],
    pub dst: [u16; 4],
    pub pad: [u16; 2],
}

impl Default for HtpOpDesc {
    fn default() -> Self {
        Self {
            opcode: 0,
            flags: 0,
            params: [0; 16],
            kernel_params: [0; 32],
            src: [0xffff; 10],
            dst: [0xffff; 4],
            pad: [0; 2],
        }
    }
}

/// Profile descriptor written by the DSP into the shared batch staging buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpProfDesc {
    pub opcode: u32,
    pub usecs: u32,
    pub cycles_start: u32,
    pub cycles_stop: u32,
    pub pmu: [u32; 8],
}

/// Batch request sent to the DSP over FastRPC queue.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HtpOpBatchReq {
    pub seq: u64,
    pub n_bufs: u32,
    pub n_tensors: u32,
    pub n_ops: u32,
    pub n_traces: u32,
}

/// Batch response returned by the DSP over FastRPC queue.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpOpBatchRsp {
    pub seq: u64,
    pub cycles_start: u64,
    pub cycles_stop: u64,
    pub status: u32,
    pub n_bufs: u32,
    pub n_tensors: u32,
    pub n_ops: u32,
    pub usecs: u32,
    pub n_traces: [u32; 11],
}

/// Buffer flag constants.
pub const DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER: u32 = 16;
pub const DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT: u32 = 128;

pub const DSPQBUF_TYPE_CONSTANT: u32 = 0;
pub const DSPQBUF_TYPE_HOST_WRITE_DSP_READ: u32 =
    DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
pub const DSPQBUF_TYPE_DSP_WRITE_HOST_READ: u32 = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

/// Buffer descriptor passed to `dspqueue_write` and `dspqueue_read`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DspQueueBuffer {
    pub fd: u32,
    pub size: u32,
    pub offset: u32,
    pub flags: u32,
    pub ptr: *mut c_void,
}

impl Default for DspQueueBuffer {
    fn default() -> Self {
        Self {
            fd: 0,
            size: 0,
            offset: 0,
            flags: 0,
            ptr: std::ptr::null_mut(),
        }
    }
}

/// Hardware information returned by DSP probe.
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpHwInfo {
    pub n_threads: u32,
    pub n_hvx: u32,
    pub n_hmx: u32,
    pub vtcm_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_htp_abi_sizes() {
        assert_eq!(std::mem::size_of::<HtpBufDesc>(), 24);
        assert_eq!(std::mem::size_of::<HtpTensor>(), 56);
        assert_eq!(std::mem::size_of::<HtpOpDesc>(), 232);
        assert_eq!(std::mem::size_of::<HtpOpBatchReq>(), 24);
        assert_eq!(std::mem::size_of::<HtpOpBatchRsp>(), 88);
        assert_eq!(std::mem::size_of::<DspQueueBuffer>(), 24);
    }
}

//! C ABI types, opcodes, and descriptors for the Hexagon Tensor Processor (HTP).
//!
//! These structures and constants precisely match upstream `ggml-hexagon`
//! (`htp-ops.h`, `htp-tensor.h`, and `htp/main.c`) to maintain binary compatibility
//! with the compiled DSP skel libraries (`libggml-htp-v*.so`).

/// Status codes returned by the DSP runtime.
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
    Q4_1 = 3,
    Q8_0 = 8,
    Q4K = 12,
    Q6K = 14,
    Iq4Nl = 20,
    I32 = 26,
    I64 = 27,
    Mxfp4 = 39,

    // Internal tiled formats produced by host-side repacking
    Q4_0Tiled = 200,
    Q4_1Tiled = 201,
    Q8_0Tiled = 202,
    Mxfp4Tiled = 203,

    Invalid = 0xFFFF_FFFF,
}

/// Tiling constants for repacked quant formats.
pub const QK_Q4_0_TILED: usize = 256; // 32x32 Q4_0 tiled layout
pub const QK_Q8_0_TILED: usize = 128; // 32x32 Q8_0 tiled layout
pub const QK_MXFP4_TILED: usize = 256; // 32x32 MXFP4 tiled layout

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
    LeakyRelu = 52,
    Im2col = 53,
    Fence = 54,
    Allreduce = 55,
    AllreduceAdd = 56,
    GluSwigluClamp = 57,
    MdevGroup = 58,
    Roll = 59,

    Invalid = 0xFFFF_FFFF,
}

pub const HTP_OP_MAX_DIMS: usize = 4;
pub const HTP_OP_MAX_INPUTS: usize = 10;
pub const HTP_OP_MAX_OUTPUTS: usize = 4;
pub const HTP_OP_MAX_PARAMS: usize = 16;
pub const HTP_OP_MAX_KERN_PARAMS: usize = 32;
pub const HTP_OP_MAX_BUFS: usize = 16;
pub const HTP_OP_MAX_TENSORS: usize = 8192;

/// Flags for tensor descriptors.
pub const HTP_TENSOR_WEIGHT: u32 = 1 << 0; // Tensor buffer holds static model weight data
pub const HTP_TENSOR_REPACK: u32 = 1 << 1; // Tensor is in repacked tiled format
pub const HTP_TENSOR_FENCE: u32 = 1 << 2; // Tensor is synchronization fence

/// Flags for buffer descriptors.
pub const HTP_BUF_EXTENDED: u32 = 1 << 0;

/// Tensor descriptor sent over FastRPC queue.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HtpTensor {
    /// Buffer offset in messages, and data pointer on the NPU.
    pub data: u64,
    /// Data size in bytes.
    pub size: u32,
    /// Tensor flags (`HTP_TENSOR_*`).
    pub flags: u32,
    /// Data type (`HtpDataType`).
    pub dtype: u32,
    /// Buffer index within the batch buffer list.
    pub bi: u16,
    /// Tensor index within the batch tensor list.
    pub ti: u16,
    /// Number of elements per dimension (up to 4D).
    pub ne: [u32; HTP_OP_MAX_DIMS],
    /// Stride in bytes per dimension.
    pub nb: [u32; HTP_OP_MAX_DIMS],
}

impl Default for HtpTensor {
    fn default() -> Self {
        Self {
            data: 0,
            size: 0,
            flags: 0,
            dtype: HtpDataType::Invalid as u32,
            bi: 0,
            ti: 0,
            ne: [0; HTP_OP_MAX_DIMS],
            nb: [0; HTP_OP_MAX_DIMS],
        }
    }
}

/// Buffer descriptor describing mapped memory segments.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpBufDesc {
    /// Base address of the memory mapping.
    pub base: u64,
    /// Total size in bytes.
    pub size: u64,
    /// Buffer flags (`HTP_BUF_*`).
    pub flags: u32,
    /// Shared memory file descriptor.
    pub fd: u32,
}

/// Op flags.
pub const HTP_OPFLAGS_STUB: u32 = 1 << 0;

/// Operation descriptor encoding a single node dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HtpOpDesc {
    /// Opcode (`HtpOpCode`).
    pub opcode: u32,
    /// Op flags (`HTP_OPFLAGS_*`).
    pub flags: u32,
    /// General operation parameters (e.g. epsilon for RMS norm).
    pub params: [i32; HTP_OP_MAX_PARAMS],
    /// Precomputed kernel parameters (e.g. FastDiv factors, head strides).
    pub kernel_params: [i32; HTP_OP_MAX_KERN_PARAMS],
    /// Input tensor indices referencing the batch tensor table.
    pub src: [u16; HTP_OP_MAX_INPUTS],
    /// Output tensor indices referencing the batch tensor table.
    pub dst: [u16; HTP_OP_MAX_OUTPUTS],
    /// Alignment padding to 64 bits.
    pub pad: [u16; 2],
}

impl Default for HtpOpDesc {
    fn default() -> Self {
        Self {
            opcode: HtpOpCode::Invalid as u32,
            flags: 0,
            params: [0; HTP_OP_MAX_PARAMS],
            kernel_params: [0; HTP_OP_MAX_KERN_PARAMS],
            src: [0xFFFF; HTP_OP_MAX_INPUTS],
            dst: [0xFFFF; HTP_OP_MAX_OUTPUTS],
            pad: [0; 2],
        }
    }
}

/// Batch request header sent as message payload to `dspqueue_write`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpOpBatchReq {
    /// Monotonic sequence counter.
    pub seq: u64,
    /// Batch request flags.
    pub flags: u32,
    /// Number of buffer descriptors packed in queue buffer.
    pub n_bufs: u32,
    /// Number of tensor descriptors packed in queue buffer.
    pub n_tensors: u32,
    /// Number of op descriptors packed in queue buffer.
    pub n_ops: u32,
}

/// Batch response header read from `dspqueue_read`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpOpBatchRsp {
    /// Monotonic sequence counter matching request.
    pub seq: u64,
    /// Completion status (`HtpStatus`).
    pub status: u32,
    /// Performance cycles elapsed on DSP.
    pub perf_cycles: u32,
}

/// Buffer handle used with `dspqueue_write` and `dspqueue_read`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DspQueueBuffer {
    pub ptr: *mut u8,
    pub size: u32,
    pub flags: u32,
}

impl Default for DspQueueBuffer {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            size: 0,
            flags: 0,
        }
    }
}

/// Hardware information returned by `htp_iface_hwinfo`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HtpHwInfo {
    pub n_threads: u32,
    pub n_hvx: u32,
    pub n_hmx: u32,
    pub vtcm_size: u64,
}

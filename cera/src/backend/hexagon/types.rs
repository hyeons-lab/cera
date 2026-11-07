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
    Q4_1 = 3,
    Q8_0 = 8,
    Q2K = 10,
    Q3K = 11,
    // K-quants have their own tiled wire layouts (`HTP_TYPE_Q4_K=12`, `HTP_TYPE_Q5_K=13`,
    // `HTP_TYPE_Q6_K=14`). cera still requants Q5_K weights to Q8_0 on the host at load.
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq4Nl = 20,
    I32 = 26,
    I64 = 27,
    Bf16 = 30,
    Mxfp4 = 39,
    Q1_0 = 41,
    Q2_0 = 42,

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
    RmsNormMulAdd = 11,
    UnarySilu = 12,
    UnaryGelu = 13,
    UnarySigmoid = 14,
    UnaryExp = 15,
    UnaryNeg = 16,
    UnarySoftplus = 17,
    UnaryTanh = 18,
    UnaryAbs = 19,
    UnaryLog = 20,
    UnaryRelu = 21,
    UnaryStep = 22,
    GluSwiglu = 23,
    GluSwigluOai = 24,
    GluGeglu = 25,
    GluGegluQuick = 26,
    Softmax = 27,
    AddId = 28,
    Rope = 29,
    FlashAttnExt = 30,
    SetRows = 31,
    GetRows = 32,
    Scale = 33,
    Cpy = 34,
    CpyFence = 35,
    Argsort = 36,
    TopK = 37,
    Sqr = 38,
    Sqrt = 39,
    Sum = 40,
    SumRows = 41,
    SsmConv = 42,
    Repeat = 43,
    Cumsum = 44,
    Fill = 45,
    Diag = 46,
    SolveTri = 47,
    L2Norm = 48,
    GatedDeltaNet = 49,
    Tri = 50,
    Pad = 51,
    Norm = 52,
    Concat = 53,
    Clamp = 54,
    LeakyRelu = 55,
    Im2col = 56,
    Conv2d = 57,
    Conv2dDw = 58,
    Fence = 59,
    Allreduce = 60,
    AllreduceAdd = 61,
    GluSwigluClamp = 62,
    MdevGroup = 63,
    Roll = 64,
    Argmax = 65,
    UnaryGeluErf = 66,
    GluGegluErf = 67,
    Pool2D = 68,
    Pool1D = 69,
    UnaryGeluQuick = 70,
    Im2col3d = 71,
    Hadamard = 72,
    GluReglu = 73,
    GluSigmoid = 74,
    UnaryCeil = 75,
    UnarySgn = 76,
    Upscale = 77,
    Conv3d = 78,
    Dsv4HcPre = 79,
    Dsv4HcPost = 80,
    TopKMask = 81,
    LightningIndexer = 82,
    Conv1D = 83,
    UnarySnake = 84,
    UnarySin = 85,
    UnaryCos = 86,
    ConvTranspose1D = 87,
    UnaryHardSigmoid = 88,
    UnaryHardSwish = 89,
    UnaryElu = 90,

    Invalid = 0xFFFF_FFFF,
}

impl HtpOpCode {
    /// Return the canonical short name for this opcode.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Mul => "Mul",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Div => "Div",
            Self::MulMat => "MulMat",
            Self::MulMatId => "MulMatId",
            Self::MulMatNx => "MulMatNx",
            Self::MulMatIdNx => "MulMatIdNx",
            Self::MulMatAdd => "MulMatAdd",
            Self::RmsNorm => "RmsNorm",
            Self::RmsNormMul => "RmsNormMul",
            Self::RmsNormMulAdd => "RmsNormMulAdd",
            Self::UnarySilu => "UnarySilu",
            Self::UnaryGelu => "UnaryGelu",
            Self::UnarySigmoid => "UnarySigmoid",
            Self::UnaryExp => "UnaryExp",
            Self::UnaryNeg => "UnaryNeg",
            Self::UnarySoftplus => "UnarySoftplus",
            Self::UnaryTanh => "UnaryTanh",
            Self::UnaryAbs => "UnaryAbs",
            Self::UnaryLog => "UnaryLog",
            Self::UnaryRelu => "UnaryRelu",
            Self::UnaryStep => "UnaryStep",
            Self::GluSwiglu => "GluSwiglu",
            Self::GluSwigluOai => "GluSwigluOai",
            Self::GluGeglu => "GluGeglu",
            Self::GluGegluQuick => "GluGegluQuick",
            Self::Softmax => "Softmax",
            Self::AddId => "AddId",
            Self::Rope => "Rope",
            Self::FlashAttnExt => "FlashAttnExt",
            Self::SetRows => "SetRows",
            Self::GetRows => "GetRows",
            Self::Scale => "Scale",
            Self::Cpy => "Cpy",
            Self::CpyFence => "CpyFence",
            Self::Argsort => "Argsort",
            Self::TopK => "TopK",
            Self::Sqr => "Sqr",
            Self::Sqrt => "Sqrt",
            Self::Sum => "Sum",
            Self::SumRows => "SumRows",
            Self::SsmConv => "SsmConv",
            Self::Repeat => "Repeat",
            Self::Cumsum => "Cumsum",
            Self::Fill => "Fill",
            Self::Diag => "Diag",
            Self::SolveTri => "SolveTri",
            Self::L2Norm => "L2Norm",
            Self::GatedDeltaNet => "GatedDeltaNet",
            Self::Tri => "Tri",
            Self::Pad => "Pad",
            Self::Norm => "Norm",
            Self::Concat => "Concat",
            Self::Clamp => "Clamp",
            Self::LeakyRelu => "LeakyRelu",
            Self::Im2col => "Im2col",
            Self::Conv2d => "Conv2d",
            Self::Conv2dDw => "Conv2dDw",
            Self::Fence => "Fence",
            Self::Allreduce => "Allreduce",
            Self::AllreduceAdd => "AllreduceAdd",
            Self::GluSwigluClamp => "GluSwigluClamp",
            Self::MdevGroup => "MdevGroup",
            Self::Roll => "Roll",
            Self::Argmax => "Argmax",
            Self::UnaryGeluErf => "UnaryGeluErf",
            Self::GluGegluErf => "GluGegluErf",
            Self::Pool2D => "Pool2D",
            Self::Pool1D => "Pool1D",
            Self::UnaryGeluQuick => "UnaryGeluQuick",
            Self::Im2col3d => "Im2col3d",
            Self::Hadamard => "Hadamard",
            Self::GluReglu => "GluReglu",
            Self::GluSigmoid => "GluSigmoid",
            Self::UnaryCeil => "UnaryCeil",
            Self::UnarySgn => "UnarySgn",
            Self::Upscale => "Upscale",
            Self::Conv3d => "Conv3d",
            Self::Dsv4HcPre => "Dsv4HcPre",
            Self::Dsv4HcPost => "Dsv4HcPost",
            Self::TopKMask => "TopKMask",
            Self::LightningIndexer => "LightningIndexer",
            Self::Conv1D => "Conv1D",
            Self::UnarySnake => "UnarySnake",
            Self::UnarySin => "UnarySin",
            Self::UnaryCos => "UnaryCos",
            Self::ConvTranspose1D => "ConvTranspose1D",
            Self::UnaryHardSigmoid => "UnaryHardSigmoid",
            Self::UnaryHardSwish => "UnaryHardSwish",
            Self::UnaryElu => "UnaryElu",
            Self::Invalid => "Invalid",
        }
    }

    /// Match an opcode value against pinned firmware-ABI discriminants.
    pub const fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::Mul),
            1 => Some(Self::Add),
            2 => Some(Self::Sub),
            3 => Some(Self::Div),
            4 => Some(Self::MulMat),
            5 => Some(Self::MulMatId),
            6 => Some(Self::MulMatNx),
            7 => Some(Self::MulMatIdNx),
            8 => Some(Self::MulMatAdd),
            9 => Some(Self::RmsNorm),
            10 => Some(Self::RmsNormMul),
            11 => Some(Self::RmsNormMulAdd),
            12 => Some(Self::UnarySilu),
            13 => Some(Self::UnaryGelu),
            14 => Some(Self::UnarySigmoid),
            15 => Some(Self::UnaryExp),
            16 => Some(Self::UnaryNeg),
            17 => Some(Self::UnarySoftplus),
            18 => Some(Self::UnaryTanh),
            19 => Some(Self::UnaryAbs),
            20 => Some(Self::UnaryLog),
            21 => Some(Self::UnaryRelu),
            22 => Some(Self::UnaryStep),
            23 => Some(Self::GluSwiglu),
            24 => Some(Self::GluSwigluOai),
            25 => Some(Self::GluGeglu),
            26 => Some(Self::GluGegluQuick),
            27 => Some(Self::Softmax),
            28 => Some(Self::AddId),
            29 => Some(Self::Rope),
            30 => Some(Self::FlashAttnExt),
            31 => Some(Self::SetRows),
            32 => Some(Self::GetRows),
            33 => Some(Self::Scale),
            34 => Some(Self::Cpy),
            35 => Some(Self::CpyFence),
            36 => Some(Self::Argsort),
            37 => Some(Self::TopK),
            38 => Some(Self::Sqr),
            39 => Some(Self::Sqrt),
            40 => Some(Self::Sum),
            41 => Some(Self::SumRows),
            42 => Some(Self::SsmConv),
            43 => Some(Self::Repeat),
            44 => Some(Self::Cumsum),
            45 => Some(Self::Fill),
            46 => Some(Self::Diag),
            47 => Some(Self::SolveTri),
            48 => Some(Self::L2Norm),
            49 => Some(Self::GatedDeltaNet),
            50 => Some(Self::Tri),
            51 => Some(Self::Pad),
            52 => Some(Self::Norm),
            53 => Some(Self::Concat),
            54 => Some(Self::Clamp),
            55 => Some(Self::LeakyRelu),
            56 => Some(Self::Im2col),
            57 => Some(Self::Conv2d),
            58 => Some(Self::Conv2dDw),
            59 => Some(Self::Fence),
            60 => Some(Self::Allreduce),
            61 => Some(Self::AllreduceAdd),
            62 => Some(Self::GluSwigluClamp),
            63 => Some(Self::MdevGroup),
            64 => Some(Self::Roll),
            65 => Some(Self::Argmax),
            66 => Some(Self::UnaryGeluErf),
            67 => Some(Self::GluGegluErf),
            68 => Some(Self::Pool2D),
            69 => Some(Self::Pool1D),
            70 => Some(Self::UnaryGeluQuick),
            71 => Some(Self::Im2col3d),
            72 => Some(Self::Hadamard),
            73 => Some(Self::GluReglu),
            74 => Some(Self::GluSigmoid),
            75 => Some(Self::UnaryCeil),
            76 => Some(Self::UnarySgn),
            77 => Some(Self::Upscale),
            78 => Some(Self::Conv3d),
            79 => Some(Self::Dsv4HcPre),
            80 => Some(Self::Dsv4HcPost),
            81 => Some(Self::TopKMask),
            82 => Some(Self::LightningIndexer),
            83 => Some(Self::Conv1D),
            84 => Some(Self::UnarySnake),
            85 => Some(Self::UnarySin),
            86 => Some(Self::UnaryCos),
            87 => Some(Self::ConvTranspose1D),
            88 => Some(Self::UnaryHardSigmoid),
            89 => Some(Self::UnaryHardSwish),
            90 => Some(Self::UnaryElu),
            0xFFFF_FFFF => Some(Self::Invalid),
            _ => None,
        }
    }
}

/// Layout format for weight tensors stored in shared rpcmem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexagonWeightFormat {
    RepackedQ8_0,
    RepackedQ4_0,
}

/// Metadata describing a repacked weight tensor stored in shared rpcmem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonWeightDesc {
    pub offset: usize,
    pub size_bytes: usize,
    pub format: HexagonWeightFormat,
    pub rows: usize,
    pub cols: usize,
}

/// Compute (non-weight) tensor: flags 0, so the DSP tracks its dirty ranges
/// and keeps intra-batch producer/consumer edges coherent. Upstream defines
/// no COMPUTE bit; bit 0 is WEIGHT (read-only, skipped by dirty tracking).
pub const HTP_TENSOR_COMPUTE: u32 = 0;
/// Tensor holds read-only model weight data (skipped by dirty tracking).
pub const HTP_TENSOR_WEIGHT: u32 = 1;
/// Tensor is in the DSP's tiled weight layout (upstream's `HTP_TENSOR_TILED`; named `REPACK`
/// before the Nov 2026 refresh, same bit).
pub const HTP_TENSOR_TILED: u32 = 2;
/// Tensor is a synchronization fence (explicitly managed).
pub const HTP_TENSOR_FENCE: u32 = 4;

/// `HTP_POWER_MODE_BURST`: the DCVS/HMX profile that pins every corner to its maximum and keeps
/// the DSP awake (`enum htp_power_mode`; lower values trade speed for power).
pub const HTP_POWER_MODE_BURST: u32 = 7;

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

/// Round `sz` up to the 128-byte HVX vector / DMA alignment every rpcmem
/// offset and DSP row stride must honor. The single definition: layout code
/// that must agree with DSP offsets should never carry its own copy.
pub(crate) const fn align128(sz: usize) -> usize {
    sz.next_multiple_of(128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_htp_abi_sizes() {
        assert_eq!(std::mem::size_of::<HtpBufDesc>(), 24);
        assert_eq!(std::mem::size_of::<HtpTensor>(), 56);
        assert_eq!(std::mem::size_of::<HtpOpDesc>(), 232);
        assert_eq!(std::mem::size_of::<HtpProfDesc>(), 48);
        assert_eq!(std::mem::size_of::<HtpOpBatchReq>(), 24);
        assert_eq!(std::mem::size_of::<HtpOpBatchRsp>(), 88);
        assert_eq!(std::mem::size_of::<DspQueueBuffer>(), 24);
    }

    #[test]
    fn test_htp_opcode_from_u32() {
        assert_eq!(HtpOpCode::from_u32(0), Some(HtpOpCode::Mul));
        assert_eq!(HtpOpCode::from_u32(1), Some(HtpOpCode::Add));
        assert_eq!(HtpOpCode::from_u32(90), Some(HtpOpCode::UnaryElu));
        assert_eq!(HtpOpCode::from_u32(0xFFFF_FFFF), Some(HtpOpCode::Invalid));
        assert_eq!(HtpOpCode::from_u32(91), None);
    }

    /// The discriminants are the wire values of `enum htp_op_code` in the bundled skels, so a
    /// shift is a silent miscompute rather than an error. Pin the ones around every insertion
    /// point upstream made (`RMS_NORM_MUL_ADD` at 11, the conv2d pair after `IM2COL`, and the
    /// ops appended after `UNARY_GELU_QUICK`) and every op our extensions add.
    #[test]
    fn htp_opcode_values_match_the_bundled_skels() {
        for (op, want) in [
            (HtpOpCode::RmsNormMul, 10u32),
            (HtpOpCode::RmsNormMulAdd, 11),
            (HtpOpCode::UnarySilu, 12),
            (HtpOpCode::Im2col, 56),
            (HtpOpCode::Conv2d, 57),
            (HtpOpCode::Conv2dDw, 58),
            (HtpOpCode::Fence, 59),
            (HtpOpCode::Roll, 64),
            (HtpOpCode::Argmax, 65),
            (HtpOpCode::UnaryGeluErf, 66),
            (HtpOpCode::GluGegluErf, 67),
            (HtpOpCode::Pool2D, 68),
            (HtpOpCode::Pool1D, 69),
            (HtpOpCode::UnaryGeluQuick, 70),
            (HtpOpCode::Im2col3d, 71),
            (HtpOpCode::Hadamard, 72),
            (HtpOpCode::Upscale, 77),
            (HtpOpCode::LightningIndexer, 82),
            (HtpOpCode::Conv1D, 83),
            (HtpOpCode::UnarySnake, 84),
            (HtpOpCode::UnarySin, 85),
            (HtpOpCode::UnaryCos, 86),
            (HtpOpCode::ConvTranspose1D, 87),
            (HtpOpCode::UnaryHardSigmoid, 88),
            (HtpOpCode::UnaryHardSwish, 89),
            (HtpOpCode::UnaryElu, 90),
        ] {
            assert_eq!(op as u32, want, "{}", op.name());
            assert_eq!(HtpOpCode::from_u32(want), Some(op), "{}", op.name());
        }
    }

    #[test]
    fn align128_rounds_up_to_vector_width() {
        assert_eq!(align128(0), 0);
        assert_eq!(align128(1), 128);
        assert_eq!(align128(128), 128);
        assert_eq!(align128(129), 256);
    }
}

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
    UnaryStep = 21,
    GluSwiglu = 22,
    GluSwigluOai = 23,
    GluGeglu = 24,
    GluGegluQuick = 25,
    Softmax = 26,
    AddId = 27,
    Rope = 28,
    FlashAttnExt = 29,
    SetRows = 30,
    GetRows = 31,
    Scale = 32,
    Cpy = 33,
    CpyFence = 34,
    Argsort = 35,
    TopK = 36,
    Sqr = 37,
    Sqrt = 38,
    Sum = 39,
    SumRows = 40,
    SsmConv = 41,
    Repeat = 42,
    Cumsum = 43,
    Fill = 44,
    Diag = 45,
    SolveTri = 46,
    L2Norm = 47,
    GatedDeltaNet = 48,
    Tri = 49,
    Pad = 50,
    Norm = 51,
    Concat = 52,
    Clamp = 53,
    LeakyRelu = 54,
    Im2col = 55,
    Fence = 56,
    Allreduce = 57,
    AllreduceAdd = 58,
    GluSwigluClamp = 59,
    MdevGroup = 60,
    Roll = 61,
    Argmax = 62,
    UnaryGeluErf = 63,
    GluGegluErf = 64,
    Pool2D = 65,
    Pool1D = 66,
    UnaryGeluQuick = 67,
    Conv1D = 68,
    UnarySnake = 69,
    UnarySin = 70,
    UnaryCos = 71,
    ConvTranspose1D = 72,
    UnaryHardSigmoid = 73,
    UnaryHardSwish = 74,
    UnaryElu = 75,

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
            11 => Some(Self::UnarySilu),
            12 => Some(Self::UnaryGelu),
            13 => Some(Self::UnarySigmoid),
            14 => Some(Self::UnaryExp),
            15 => Some(Self::UnaryNeg),
            16 => Some(Self::UnarySoftplus),
            17 => Some(Self::UnaryTanh),
            18 => Some(Self::UnaryAbs),
            19 => Some(Self::UnaryLog),
            20 => Some(Self::UnaryRelu),
            21 => Some(Self::UnaryStep),
            22 => Some(Self::GluSwiglu),
            23 => Some(Self::GluSwigluOai),
            24 => Some(Self::GluGeglu),
            25 => Some(Self::GluGegluQuick),
            26 => Some(Self::Softmax),
            27 => Some(Self::AddId),
            28 => Some(Self::Rope),
            29 => Some(Self::FlashAttnExt),
            30 => Some(Self::SetRows),
            31 => Some(Self::GetRows),
            32 => Some(Self::Scale),
            33 => Some(Self::Cpy),
            34 => Some(Self::CpyFence),
            35 => Some(Self::Argsort),
            36 => Some(Self::TopK),
            37 => Some(Self::Sqr),
            38 => Some(Self::Sqrt),
            39 => Some(Self::Sum),
            40 => Some(Self::SumRows),
            41 => Some(Self::SsmConv),
            42 => Some(Self::Repeat),
            43 => Some(Self::Cumsum),
            44 => Some(Self::Fill),
            45 => Some(Self::Diag),
            46 => Some(Self::SolveTri),
            47 => Some(Self::L2Norm),
            48 => Some(Self::GatedDeltaNet),
            49 => Some(Self::Tri),
            50 => Some(Self::Pad),
            51 => Some(Self::Norm),
            52 => Some(Self::Concat),
            53 => Some(Self::Clamp),
            54 => Some(Self::LeakyRelu),
            55 => Some(Self::Im2col),
            56 => Some(Self::Fence),
            57 => Some(Self::Allreduce),
            58 => Some(Self::AllreduceAdd),
            59 => Some(Self::GluSwigluClamp),
            60 => Some(Self::MdevGroup),
            61 => Some(Self::Roll),
            62 => Some(Self::Argmax),
            63 => Some(Self::UnaryGeluErf),
            64 => Some(Self::GluGegluErf),
            65 => Some(Self::Pool2D),
            66 => Some(Self::Pool1D),
            67 => Some(Self::UnaryGeluQuick),
            68 => Some(Self::Conv1D),
            69 => Some(Self::UnarySnake),
            70 => Some(Self::UnarySin),
            71 => Some(Self::UnaryCos),
            72 => Some(Self::ConvTranspose1D),
            73 => Some(Self::UnaryHardSigmoid),
            74 => Some(Self::UnaryHardSwish),
            75 => Some(Self::UnaryElu),
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
/// Tensor is in repacked tiled format.
pub const HTP_TENSOR_REPACK: u32 = 2;
/// Tensor is a synchronization fence (explicitly managed).
pub const HTP_TENSOR_FENCE: u32 = 4;

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

/// Round `sz` up to 256 bytes (the KV/state slab and theta-row alignment).
pub(crate) const fn align256(sz: usize) -> usize {
    sz.next_multiple_of(256)
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
        assert_eq!(HtpOpCode::from_u32(75), Some(HtpOpCode::UnaryElu));
        assert_eq!(HtpOpCode::from_u32(0xFFFF_FFFF), Some(HtpOpCode::Invalid));
        assert_eq!(HtpOpCode::from_u32(76), None);
    }

    /// The discriminants are the wire values of `enum htp_op_code` in the bundled skels, so a
    /// shift is a silent miscompute rather than an error. Pin the ones around the insertion point
    /// (llama.cpp added five upstream ops at 63 to 67) and every op our extensions add.
    #[test]
    fn htp_opcode_values_match_the_bundled_skels() {
        for (op, want) in [
            (HtpOpCode::Argmax, 62u32),
            (HtpOpCode::UnaryGeluErf, 63),
            (HtpOpCode::GluGegluErf, 64),
            (HtpOpCode::Pool2D, 65),
            (HtpOpCode::Pool1D, 66),
            (HtpOpCode::UnaryGeluQuick, 67),
            (HtpOpCode::Conv1D, 68),
            (HtpOpCode::UnarySnake, 69),
            (HtpOpCode::UnarySin, 70),
            (HtpOpCode::UnaryCos, 71),
            (HtpOpCode::ConvTranspose1D, 72),
            (HtpOpCode::UnaryHardSigmoid, 73),
            (HtpOpCode::UnaryHardSwish, 74),
            (HtpOpCode::UnaryElu, 75),
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
        assert_eq!(align256(0), 0);
        assert_eq!(align256(1), 256);
        assert_eq!(align256(256), 256);
        assert_eq!(align256(257), 512);
    }
}

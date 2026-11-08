//! Matmul kernel parameters for the Hexagon DSP: a port of llama.cpp's host solver
//! (`ggml-hexagon/htp-kparams.cpp`, `htp/matmul-ops.h`).
//!
//! The DSP's `op_matmul*` kernels read `struct htp_mm_kernel_params` (kernel variant, chunk
//! sizes, VTCM footprint, fastdiv constants) from the op's `kernel_params` blob and answer
//! `NO_SUPPORT` or compute garbage for anything else, so every field here is what the real host
//! would send for the same tensors. `testdata/matmul*_golden.txt` pin that word for word: the
//! files come from compiling llama.cpp's own solver (`scripts/hexagon-golden`).
//!
//! The solver sees a ggml tensor (type, extents, byte strides, and whether the weight is already
//! in the DSP's tiled layout); [`MmTensor`] is that view. A registered weight descriptor of a tiled
//! tensor carries the tiled strides instead, so [`MmTensor::from`] rebuilds the contiguous ggml
//! tensor the host solver would have seen.
//!
//! What differs from the C++, deliberately:
//! - Tiling comes only from the tensor's flag. The host also tiles any contiguous weight in a
//!   device buffer at load (`opt_rt_repack`); cera uploads its own layout and never relies on it.
//! - `ggml_is_view` is not visible from a descriptor, so a view is never refused for being one.
//! - The HMX path needs [`HMX_MM_MIN_NROWS`] activation rows here (upstream: more than 4). Below
//!   that the HMX worker is racy on device, so a session from [`MmSession::new`] keeps the HMX off
//!   there, for the `build_*` entry points and the `solve_*` solvers alike. The golden tests build
//!   their sessions with upstream's floor (`hmx_min_rows`) to pin the unmodified solver.

use super::params::{FastDivValues, init_fastdiv, wire_byte};
use super::types::{HTP_TENSOR_TILED, HtpDataType, HtpTensor};

/// Minimum activation rows for cera to take the HMX kernel. Upstream's `HTP_MM_HMX_MIN_NROWS` is
/// 4 (HMX for more than that), but the HMX worker is nondeterministic below 8 rows on device
/// (8-row microtile overhang; M <= 7 racy, M >= 8 bitwise stable and at CPU parity), so smaller
/// runs take the HVX kernel.
pub const HMX_MM_MIN_NROWS: usize = 8;

/// `HTP_MM_HMX_MIN_NROWS`: the solver's own floor (HMX needs more rows than this).
const UPSTREAM_HMX_MIN_NROWS: u64 = 4;

// `enum htp_mm_kernel_type`.
pub const MM_KERNEL_UNSUPPORTED: u8 = 0;
pub const MM_KERNEL_HMX_2D: u8 = 1;
pub const MM_KERNEL_HMX_BATCHED: u8 = 2;
pub const MM_KERNEL_HVX_F16_F16_VTCM: u8 = 3;
pub const MM_KERNEL_HVX_F32_F32_VTCM: u8 = 4;
pub const MM_KERNEL_HVX_BF16_F32_VTCM: u8 = 5;
pub const MM_KERNEL_HVX_QUANT_ROW: u8 = 6;
pub const MM_KERNEL_HVX_QUANT_BLOCK: u8 = 7;

const HMX_TILE_N_COLS: u64 = 32;
const HMX_TILE_N_ROWS: u64 = 32;
/// `HTP_MM_HMX_TILE_SIZE` (32 x 32 f16), also the VTCM group alignment.
const HMX_TILE_BYTES: u64 = 2048;
/// Activation block (`HTP_MM_ACT_BLOCK_K`) and the tiled activation tile sizes per 32 elements.
const ACT_BLOCK_K: u64 = 128;
const ACT_TILE_Q8_0: u64 = 1152;
const ACT_TILE_Q8_1: u64 = 1280;
const ACT_TILE_F16: u64 = 2048;
/// Solver cost weights (`HTP_MM_HMX_COST_*`).
const HMX_COST_W_DEQUANT: u64 = 3;
const HMX_COST_A_CONVERT: u64 = 2;
/// `HTP_MM_DMA_ACT_MULTIPLIER` (2 * `HTP_MM_DMA_ACT_ROWS_PER_STEP`).
const DMA_ACT_MULTIPLIER: u64 = 4;

const F32: u32 = HtpDataType::F32 as u32;
const F16: u32 = HtpDataType::F16 as u32;
const BF16: u32 = HtpDataType::Bf16 as u32;
const I32: u32 = HtpDataType::I32 as u32;
const Q4_0: u32 = HtpDataType::Q4_0 as u32;
const Q4_1: u32 = HtpDataType::Q4_1 as u32;
const Q8_0: u32 = HtpDataType::Q8_0 as u32;
const Q2_K: u32 = HtpDataType::Q2K as u32;
const Q3_K: u32 = HtpDataType::Q3K as u32;
const Q4_K: u32 = HtpDataType::Q4K as u32;
const Q5_K: u32 = HtpDataType::Q5K as u32;
const Q6_K: u32 = HtpDataType::Q6K as u32;
const IQ4_NL: u32 = HtpDataType::Iq4Nl as u32;
const MXFP4: u32 = HtpDataType::Mxfp4 as u32;
const Q1_0: u32 = HtpDataType::Q1_0 as u32;
const Q2_0: u32 = HtpDataType::Q2_0 as u32;

// --- ggml / HTP type facts -------------------------------------------------------------------

/// `(ggml_blck_size, ggml_type_size)` for the types a matmul can see. The one table of block sizes
/// and type sizes in this module tree: `params.rs` reads its operands through it too.
pub(super) fn type_info(dtype: u32) -> Option<(u64, u64)> {
    Some(match dtype {
        F32 | I32 => (1, 4),
        F16 | BF16 => (1, 2),
        Q4_0 | IQ4_NL => (32, 18),
        Q4_1 => (32, 20),
        Q8_0 => (32, 34),
        MXFP4 => (32, 17),
        Q1_0 => (128, 18),
        Q2_0 => (64, 18),
        Q2_K => (256, 84),
        Q3_K => (256, 110),
        Q4_K => (256, 144),
        Q5_K => (256, 176),
        Q6_K => (256, 210),
        _ => return None,
    })
}

fn is_quantized(dtype: u32) -> bool {
    !matches!(dtype, F32 | F16 | BF16 | I32)
}

/// `htp_is_tiled_weight_type`: the weight types the DSP can read in its tiled layout.
fn is_tiled_weight_type(dtype: u32) -> bool {
    matches!(
        dtype,
        Q1_0 | Q2_0
            | Q4_0
            | Q4_1
            | MXFP4
            | IQ4_NL
            | Q8_0
            | Q6_K
            | Q2_K
            | Q3_K
            | Q4_K
            | Q5_K
            | F16
            | F32
            | BF16
    )
}

/// `htp_tensor_get_tile_size`: bytes of one 32 x tile_k weight tile, 0 for untiled types.
pub(super) fn tile_size(dtype: u32) -> u64 {
    match dtype {
        Q4_0 | IQ4_NL | Q1_0 | Q2_0 => 576,
        Q4_1 | Q4_K => 640,
        Q8_0 => 1088,
        Q5_K => 768,
        Q6_K => 896,
        Q3_K => 512,
        Q2_K => 336,
        MXFP4 => 544,
        F16 => 2048,
        _ => 0,
    }
}

/// `htp_tensor_get_tile_k`: weight elements along K per tile.
pub(super) fn tile_k(dtype: u32) -> u64 {
    match dtype {
        Q1_0 => 128,
        Q2_0 => 64,
        _ => 32,
    }
}

/// `htp_tensor_get_tile_padded_size`: the tile size rounded up to a DMA line.
fn tile_padded_size(dtype: u32) -> u64 {
    (tile_size(dtype) + 127) & !127
}

/// `htp_tensor_get_row_size`: bytes of one row of `ne00` elements, 0 for an unknown type.
pub(super) fn row_size(dtype: u32, ne00: u64) -> u64 {
    match dtype {
        F32 | I32 => ne00 * 4,
        F16 | BF16 => ne00 * 2,
        Q8_0 => (ne00 / 32) * 34,
        Q4_0 | IQ4_NL => (ne00 / 32) * 18,
        Q4_1 => (ne00 / 32) * 20,
        MXFP4 => (ne00 / 32) * 17,
        Q2_K => (ne00 / 256) * 84,
        Q2_0 => (ne00 / 64) * 18,
        Q3_K => (ne00 / 256) * 110,
        Q4_K => (ne00 / 256) * 144,
        Q5_K => (ne00 / 256) * 176,
        Q6_K => (ne00 / 256) * 210,
        Q1_0 => (ne00 / 128) * 18,
        x if x == HtpDataType::I64 as u32 => ne00 * 8,
        _ => 0,
    }
}

/// `htp_tiled_row_size` (`htp-repack.h`): bytes of one row of the tiled layout, which is the
/// weight's `nb[1]` as the HMX kernel strides it. The Q3_K..Q6_K tiles store scales and mins
/// uncompressed, so they are wider than the ggml blocks.
pub(crate) fn tiled_row_size(dtype: u32, ne0: u64) -> u64 {
    match dtype {
        F16 | F32 | BF16 => ne0 * 2,
        Q6_K | Q4_K | Q5_K | Q3_K => (ne0 / 32) * (tile_size(dtype) / 32),
        Q2_0 => (ne0 / 64) * 18,
        _ => type_info(dtype).map_or(0, |(blck, size)| size * ne0 / blck),
    }
}

/// `htp_mm_weight_has_offset`: weight tiles of the form `x = d * q + m` need the activation block
/// sums, which the q8_1 activation tiles carry.
fn weight_has_offset(dtype: u32) -> bool {
    matches!(dtype, Q4_1 | Q4_K | Q5_K | Q2_K | Q2_0)
}

/// `hex_round_up`: 32-bit arithmetic, as the C helper takes `uint32_t`.
fn hex_round_up(n: u64, m: u64) -> u64 {
    let (n, m) = (n as u32, m as u32);
    u64::from(m.wrapping_mul(n.wrapping_add(m).wrapping_sub(1) / m))
}

/// `hex_align_up` / `htp_mm_round_up`: `size_t` arithmetic.
fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

fn q8_0_tiled_row_size(ne: u64) -> u64 {
    (hex_round_up(ne, ACT_BLOCK_K) / 32) * ACT_TILE_Q8_0
}

fn q8_1_tiled_row_size(ne: u64) -> u64 {
    (hex_round_up(ne, ACT_BLOCK_K) / 32) * ACT_TILE_Q8_1
}

fn f16_tiled_row_size(ne: u64) -> u64 {
    (hex_round_up(ne, ACT_BLOCK_K) / 32) * ACT_TILE_F16
}

/// `htp_mm_tiled_act_row_size`: bytes of one activation row after the DSP tiles and quantizes it
/// for `weight_type`.
fn tiled_act_row_size(weight_type: u32, ne: u64) -> u64 {
    if weight_type == F16 {
        f16_tiled_row_size(ne)
    } else if weight_has_offset(weight_type) {
        q8_1_tiled_row_size(ne)
    } else {
        q8_0_tiled_row_size(ne)
    }
}

/// `htp_mm_act_raw_row_size`: the f32 staging row the activation DMA lands in.
fn act_raw_row_size(ne: u64) -> u64 {
    hex_round_up(ne, ACT_BLOCK_K) * 4
}

/// `init_fastdiv_values(d)`. A zero divisor happens for a routed matmul with fewer tokens than
/// experts (`ne12 / ne02`); the C++ then divides by zero, which on the aarch64 host is a
/// hardware `udiv` yielding 0, so `mp = 0 + 1`. The routed kernels do not read that divider.
fn fastdiv(d: u64) -> FastDivValues {
    if d as u32 == 0 {
        return FastDivValues { mp: 1, l: 0 };
    }
    init_fastdiv(d as u32)
}

// --- tensors -----------------------------------------------------------------------------------

/// A ggml tensor as the host solver sees it: element type, extents, byte strides, and whether a
/// weight is stored in the DSP's tiled layout (the host's `HTP_TENSOR_TILED` flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmTensor {
    /// `HtpDataType` value (equal to the `ggml_type`).
    pub dtype: u32,
    pub ne: [u64; 4],
    pub nb: [u64; 4],
    pub tiled: bool,
}

impl MmTensor {
    pub fn new(dtype: u32, ne: [u64; 4], nb: [u64; 4], tiled: bool) -> Self {
        Self {
            dtype,
            ne,
            nb,
            tiled,
        }
    }

    /// A ggml-contiguous tensor: rows of `ne[0]` elements back to back, then planes.
    pub fn contiguous(dtype: u32, ne: [u64; 4]) -> Self {
        let (blck, size) = type_info(dtype).unwrap_or((1, 0));
        let nb1 = size * ne[0] / blck;
        let nb2 = nb1 * ne[1];
        Self::new(dtype, ne, [size, nb1, nb2, nb2 * ne[2]], false)
    }

    /// The contiguous ggml tensor behind a weight stored tiled, which is what the host solver
    /// receives for it (the tiled strides live in the registered descriptor, not here).
    pub fn tiled_weight(dtype: u32, ne: [u64; 4]) -> Self {
        Self {
            tiled: true,
            ..Self::contiguous(dtype, ne)
        }
    }

    /// An F32 tensor with `rows` rows `nb1` bytes apart (the activations and outputs the kernels
    /// read and write; `nb1 == ne[0] * 4` unless the rows are padded or the view is a slice).
    pub fn f32_rows(ne: [u64; 4], nb1: u64) -> Self {
        let nb2 = nb1 * ne[1];
        Self::new(F32, ne, [4, nb1, nb2, nb2 * ne[2]], false)
    }

    fn blck(&self) -> u64 {
        type_info(self.dtype).map_or(1, |(b, _)| b)
    }

    fn type_size(&self) -> u64 {
        type_info(self.dtype).map_or(0, |(_, s)| s)
    }

    /// `ggml_is_contiguous`.
    pub fn is_contiguous(&self) -> bool {
        let blck = self.blck();
        let mut next_nb = self.type_size();
        if self.ne[0] != blck && self.nb[0] != next_nb {
            return false;
        }
        next_nb *= self.ne[0] / blck;
        for i in 1..4 {
            if self.ne[i] != 1 && self.nb[i] != next_nb {
                return false;
            }
            next_nb *= self.ne[i];
        }
        true
    }

    /// `ggml_is_permuted`.
    pub fn is_permuted(&self) -> bool {
        self.nb[0] > self.nb[1] || self.nb[1] > self.nb[2] || self.nb[2] > self.nb[3]
    }

    /// `ggml_are_same_shape`.
    pub fn same_shape(&self, other: &Self) -> bool {
        self.ne == other.ne
    }

    /// `ggml_nbytes`.
    pub fn nbytes(&self) -> u64 {
        if self.ne.contains(&0) {
            return 0;
        }
        let blck = self.blck();
        let mut n;
        if blck == 1 {
            n = self.type_size();
            for i in 0..4 {
                n += (self.ne[i] - 1) * self.nb[i];
            }
        } else {
            n = self.ne[0] * self.nb[0] / blck;
            for i in 1..4 {
                n += (self.ne[i] - 1) * self.nb[i];
            }
        }
        n
    }

    /// `htp_can_row_partition`: every populated outer stride is a whole DMA line.
    fn can_row_partition(&self) -> bool {
        !((self.ne[1] > 1 && self.nb[1] & 127 != 0)
            || (self.ne[2] > 1 && self.nb[2] & 127 != 0)
            || (self.ne[3] > 1 && self.nb[3] & 127 != 0))
    }
}

impl From<&HtpTensor> for MmTensor {
    fn from(t: &HtpTensor) -> Self {
        let ne = t.ne.map(u64::from);
        if t.flags & HTP_TENSOR_TILED != 0 {
            return Self::tiled_weight(t.dtype, ne);
        }
        Self::new(t.dtype, ne, t.nb.map(u64::from), false)
    }
}

/// The session limits the solver reads (`htp_session_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmSession {
    pub n_threads: u32,
    pub vtcm_bytes: usize,
    /// HMX units on the DSP; 0 keeps every HMX kernel off.
    pub n_hmx: u32,
    /// DSP devices the op is split over (1 for every cera session).
    pub mdev_count: u32,
    /// The fewest activation rows that may take the HMX: cera's [`HMX_MM_MIN_NROWS`] for
    /// [`MmSession::new`], upstream's `UPSTREAM_HMX_MIN_NROWS + 1` for the sessions that pin the
    /// unmodified solver.
    pub hmx_min_rows: u64,
}

impl MmSession {
    pub fn new(n_threads: u32, vtcm_bytes: usize, n_hmx: u32) -> Self {
        Self {
            n_threads,
            vtcm_bytes,
            n_hmx,
            mdev_count: 1,
            hmx_min_rows: HMX_MM_MIN_NROWS as u64,
        }
    }

    /// The thread count the solvers divide by: a session without threads divides by zero in the
    /// C, which never sees one, so it counts as one thread here.
    fn threads(&self) -> u32 {
        self.n_threads.max(1)
    }
}

// --- the params blob ---------------------------------------------------------------------------

/// `struct htp_mm_kernel_params` (`matmul-ops.h`) as it lies in the op's 32-word `kernel_params`
/// blob: eight one-byte fields in the first two words, seven `int32` sizes, then seven `{mp, l}`
/// fastdiv pairs (23 words; the rest stay zero). Build and patch through this struct rather than
/// by word index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MmKernelParams {
    /// `enum htp_mm_kernel_type` (the `MM_KERNEL_*` constants).
    pub kernel_type: u8,
    /// 1 = pipelined (double-buffered) execution.
    pub pipeline: u8,
    /// Reserved.
    pub unused: u8,
    /// 1 = HMX, 0 = HVX.
    pub n_hmx: u8,
    pub n_threads: u8,
    pub n_act_threads: u8,
    pub n_prefetch: u8,
    /// Weight count of a fused `MulMatNx` / `MulMatIdNx`.
    pub n_weights: u8,
    pub m_chunk: i32,
    pub n_chunk: i32,
    pub tile_size: i32,
    pub aligned_tile_size: i32,
    pub act_row_size: i32,
    pub vtcm_size: i32,
    /// Bias scratch (`MulMatAdd` only).
    pub vtcm_bias_size: i32,
    pub div_ne12_ne1: FastDivValues,
    pub div_ne1: FastDivValues,
    pub div_r2: FastDivValues,
    pub div_r3: FastDivValues,
    pub div_ne12: FastDivValues,
    pub div_n_act_threads: FastDivValues,
    pub div_ne00_padded: FastDivValues,
}

impl MmKernelParams {
    /// The `kernel_params` words the DSP reads.
    pub fn to_words(&self) -> [i32; 32] {
        let mut w = [0i32; 32];
        let bytes = |b: [u8; 4]| u32::from_le_bytes(b) as i32;
        w[0] = bytes([self.kernel_type, self.pipeline, self.unused, self.n_hmx]);
        w[1] = bytes([
            self.n_threads,
            self.n_act_threads,
            self.n_prefetch,
            self.n_weights,
        ]);
        w[2] = self.m_chunk;
        w[3] = self.n_chunk;
        w[4] = self.tile_size;
        w[5] = self.aligned_tile_size;
        w[6] = self.act_row_size;
        w[7] = self.vtcm_size;
        w[8] = self.vtcm_bias_size;
        for (i, d) in [
            self.div_ne12_ne1,
            self.div_ne1,
            self.div_r2,
            self.div_r3,
            self.div_ne12,
            self.div_n_act_threads,
            self.div_ne00_padded,
        ]
        .iter()
        .enumerate()
        {
            w[9 + 2 * i] = d.mp as i32;
            w[10 + 2 * i] = d.l as i32;
        }
        w
    }

    /// Read a blob back, e.g. to patch a field of an already-built set of parameters.
    pub fn from_words(w: &[i32; 32]) -> Self {
        let b0 = (w[0] as u32).to_le_bytes();
        let b1 = (w[1] as u32).to_le_bytes();
        let div = |i: usize| FastDivValues {
            mp: w[9 + 2 * i] as u32,
            l: w[10 + 2 * i] as u32,
        };
        Self {
            kernel_type: b0[0],
            pipeline: b0[1],
            unused: b0[2],
            n_hmx: b0[3],
            n_threads: b1[0],
            n_act_threads: b1[1],
            n_prefetch: b1[2],
            n_weights: b1[3],
            m_chunk: w[2],
            n_chunk: w[3],
            tile_size: w[4],
            aligned_tile_size: w[5],
            act_row_size: w[6],
            vtcm_size: w[7],
            vtcm_bias_size: w[8],
            div_ne12_ne1: div(0),
            div_ne1: div(1),
            div_r2: div(2),
            div_r3: div(3),
            div_ne12: div(4),
            div_n_act_threads: div(5),
            div_ne00_padded: div(6),
        }
    }
}

// --- HMX chunk search and VTCM layouts -------------------------------------------------------

/// `htp_mm_hmx_compute_chunks`: the `(m_chunk, n_chunk, total_bytes)` that minimise the reload
/// cost within `vtcm_total`, ties going to the chunk that leaves the fewest columns idle in the
/// last N block, then to the larger one. `None` when nothing fits or a size overflows.
#[allow(clippy::too_many_arguments)]
pub fn mm_hmx_compute_chunks(
    vtcm_total: u64,
    overhead: u64,
    per_n_cost: u64,
    per_m_cost: u64,
    per_mn_cost: u64,
    m: u64,
    n: u64,
    m_block_cost: u64,
    n_block_cost: u64,
) -> Option<(u64, u64, u64)> {
    if m == 0 || n == 0 || vtcm_total <= overhead {
        return None;
    }
    if per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0 {
        return None;
    }
    let usable = vtcm_total - overhead;
    let (mut best_cost, mut best_tail_waste, mut best_mn) = (u64::MAX, u64::MAX, 0u64);
    let (mut best_m, mut best_n) = (0u64, 0u64);
    let max_nc_budget = usable / per_n_cost;
    let n_max = n.min(max_nc_budget) / HMX_TILE_N_COLS * HMX_TILE_N_COLS;
    let mut nc = n_max;
    while nc >= HMX_TILE_N_COLS {
        let candidate = (|| {
            let n_fixed = nc.checked_mul(per_n_cost)?;
            if n_fixed >= usable {
                return None;
            }
            let ncmn = nc.checked_mul(per_mn_cost)?;
            let mc_denom = per_m_cost.checked_add(ncmn)?;
            if mc_denom == 0 {
                return None;
            }
            let mc = ((usable - n_fixed) / mc_denom) / HMX_TILE_N_ROWS * HMX_TILE_N_ROWS;
            let mc = mc.min(m);
            (mc != 0).then_some(mc)
        })();
        if let Some(mc) = candidate {
            let mblocks = m.div_ceil(mc);
            let nblocks = n.div_ceil(nc);
            let cost = mblocks
                .wrapping_mul(m_block_cost)
                .wrapping_add(nblocks.wrapping_mul(n_block_cost));
            let rem = n % nc;
            let tail_waste = if rem == 0 { 0 } else { nc - rem };
            let mn = mc * nc;
            if cost < best_cost
                || (cost == best_cost && tail_waste < best_tail_waste)
                || (cost == best_cost && tail_waste == best_tail_waste && mn > best_mn)
            {
                best_cost = cost;
                best_tail_waste = tail_waste;
                best_mn = mn;
                best_m = mc;
                best_n = nc;
            }
        }
        if nc == HMX_TILE_N_COLS {
            break;
        }
        nc -= HMX_TILE_N_COLS;
    }
    if best_m == 0 || best_n == 0 {
        return None;
    }
    let total = best_n
        .checked_mul(per_n_cost)?
        .checked_add(best_m.checked_mul(per_m_cost)?)?
        .checked_add(best_m.checked_mul(best_n)?.checked_mul(per_mn_cost)?)?
        .checked_add(overhead)?;
    Some((best_m, best_n, total))
}

/// `htp_mm_hmx_pipeline`: double-buffered execution needs more than one M tile.
fn hmx_pipeline(m: u64) -> bool {
    m > 32
}

/// `htp_mm_hmx_get_2d_chunk_costs`: bytes per N column, per M row and per M*N output element.
fn hmx_2d_chunk_costs(
    wtype: u32,
    k: u64,
    pipeline: bool,
    aligned_tile_size: u64,
) -> (u64, u64, u64) {
    let is_quant = !matches!(wtype, F16 | F32 | BF16);
    let row_stride = row_size(wtype, k);
    let vec_dot_size = k * 2;
    let n_weight_tiles = (k / tile_k(wtype)) as u32;
    let qweight_row_stride = if is_quant {
        u64::from(n_weight_tiles.wrapping_mul(aligned_tile_size as u32)) / 32
    } else {
        0
    };
    let p = if pipeline { 2 } else { 1 };
    let per_n = p * if is_quant {
        qweight_row_stride
    } else {
        row_stride
    } + if pipeline {
        2 * vec_dot_size
    } else {
        vec_dot_size
    };
    (per_n, vec_dot_size, p * 2)
}

/// `htp_mm_hmx_get_batched_chunk_costs`.
fn hmx_batched_chunk_costs(k: u64, group_size: u64) -> (u64, u64, u64) {
    let vec_dot_size = k * 2;
    (3 * vec_dot_size, group_size * vec_dot_size, 2)
}

/// `htp_mm_hmx_get_2d_overhead`.
fn hmx_2d_overhead(pipeline: bool, is_matmul_id: bool) -> u64 {
    let regions = if pipeline {
        7
    } else if is_matmul_id {
        4
    } else {
        5
    };
    regions * HMX_TILE_BYTES + 256
}

/// `htp_mm_hmx_get_batched_overhead`.
fn hmx_batched_overhead() -> u64 {
    5 * HMX_TILE_BYTES + 256
}

/// The group-C activation conversion scratch and the layout total, shared by both HMX layouts:
/// group C overlaps group B, so the total is the larger of the two after group A.
fn hmx_layout_total(off_a: u64, group_b: u64, act_threads: u64, k: u64) -> u64 {
    let min_f32 = align_up(act_threads * DMA_ACT_MULTIPLIER * k * 4, 128);
    let max_f32 = act_threads * 64 * k * 4;
    let act_f32 = align_up(max_f32.min(min_f32.max(group_b)), 128);
    off_a + group_b.max(act_f32)
}

/// `htp_mm_hmx_vtcm_layout_build`, `HTP_MM_KERNEL_HMX_2D`: the VTCM bytes of one chunking.
#[allow(clippy::too_many_arguments)]
fn hmx_2d_vtcm_size(
    wtype: u32,
    k: u64,
    mc: u64,
    nc: u64,
    pipeline: bool,
    act_threads: u64,
    aligned_tile_size: u64,
    bias_size: u64,
) -> u64 {
    let is_quant = !matches!(wtype, F16 | F32 | BF16);
    let row_stride = row_size(wtype, k);
    let vec_dot_size = k * 2;
    let n_weight_tiles = (k / tile_k(wtype)) as u32;
    let weight_area = if is_quant {
        align_up(
            (nc / 32) * u64::from(n_weight_tiles) * aligned_tile_size,
            HMX_TILE_BYTES,
        )
    } else {
        align_up(nc * row_stride, HMX_TILE_BYTES)
    };
    let act_area = align_up(mc * vec_dot_size, HMX_TILE_BYTES);
    let output_area = align_up(mc * nc * 2, HMX_TILE_BYTES);
    let scratch0 = align_up(nc * vec_dot_size, HMX_TILE_BYTES);
    // Group A: scales, activation tiles, bias.
    let mut off_a = HMX_TILE_BYTES + act_area;
    if bias_size > 0 {
        off_a += align_up(bias_size, HMX_TILE_BYTES);
    }
    // Group B: weights, output, scratch (each doubled when pipelined).
    let mut group_b = weight_area + output_area + scratch0;
    if pipeline {
        group_b += weight_area + scratch0 + output_area;
    }
    hmx_layout_total(off_a, group_b, act_threads, k)
}

/// `htp_mm_hmx_vtcm_layout_build`, `HTP_MM_KERNEL_HMX_BATCHED`.
fn hmx_batched_vtcm_size(
    k: u64,
    mc: u64,
    nc: u64,
    group_size: u64,
    act_threads: u64,
    bias_size: u64,
) -> u64 {
    let vec_dot_size = k * 2;
    let weight_area = align_up(nc * vec_dot_size, HMX_TILE_BYTES);
    let activation_area = align_up(group_size * mc * k * 2, HMX_TILE_BYTES);
    let output_area = align_up(group_size * mc * nc * 2, HMX_TILE_BYTES);
    let scratch_area = align_up(nc * vec_dot_size, HMX_TILE_BYTES);
    let mut off_a = activation_area + HMX_TILE_BYTES;
    if bias_size > 0 {
        off_a += align_up(bias_size, HMX_TILE_BYTES);
    }
    let group_b = weight_area + output_area + 2 * scratch_area;
    hmx_layout_total(off_a, group_b, act_threads, k)
}

/// One HMX chunking: `(m_chunk, n_chunk, act_threads, vtcm_bytes)`.
type HmxChunking = (u64, u64, u64, u64);

/// `htp_mm_hmx_solve_batched_params`: the grouped (GQA) F16 chunking, trying the activation
/// thread count from `n_threads` halving down to 1 and keeping the fewest M blocks.
#[allow(clippy::too_many_arguments)]
fn hmx_solve_batched(
    k: u64,
    ne01_padded: u64,
    ne11: u64,
    group_size: u64,
    n_threads: i32,
    bias_size: u64,
    vtcm_budget: u64,
) -> Option<HmxChunking> {
    let mut best: Option<(u64, HmxChunking)> = None;
    let mut act_threads = n_threads;
    while act_threads >= 1 {
        let overhead = hmx_batched_overhead()
            + if bias_size > 0 {
                align_up(bias_size, HMX_TILE_BYTES)
            } else {
                0
            };
        let (per_n, per_m, per_mn) = hmx_batched_chunk_costs(k, group_size);
        if let Some((mc, nc, _)) = mm_hmx_compute_chunks(
            vtcm_budget,
            overhead,
            per_n,
            per_m,
            per_mn,
            align_up(ne11, 32),
            ne01_padded,
            ne01_padded * HMX_COST_W_DEQUANT,
            ne11 * HMX_COST_A_CONVERT,
        ) {
            let exact = hmx_batched_vtcm_size(k, mc, nc, group_size, act_threads as u64, bias_size);
            if exact <= vtcm_budget {
                let mblocks = ne11.div_ceil(mc);
                if best.is_none_or(|(bm, (_, _, bat, _))| {
                    mblocks < bm || (mblocks == bm && act_threads as u64 > bat)
                }) {
                    best = Some((mblocks, (mc, nc, act_threads as u64, exact)));
                }
            }
        }
        act_threads = if act_threads == 1 { 0 } else { act_threads / 2 };
    }
    best.map(|(_, c)| c)
}

/// `htp_mm_hmx_solve_2d_params`: the 2-D HMX chunking (also the routed-expert one) over the
/// activation-thread counts `n_threads`, `n_threads / 2`, .. 1: the candidate with the fewest M
/// blocks wins, then the most threads.
#[allow(clippy::too_many_arguments)]
pub fn mm_hmx_solve_2d(
    wtype: u32,
    k: u32,
    m_id_rows: u32,
    ne01_padded: u32,
    ne11_padded: u32,
    m_for_cost: u32,
    n_threads: i32,
    pipeline: bool,
    is_matmul_id: bool,
    aligned_tile_size: u32,
    bias_size: u64,
    vtcm_budget: u64,
) -> Option<(u64, u64, u64, u64)> {
    let m_for_chunks = if is_matmul_id {
        align_up(u64::from(m_id_rows), 32)
    } else {
        u64::from(ne11_padded)
    };
    let mut best: Option<(u64, HmxChunking)> = None;
    let mut act_threads = n_threads;
    while act_threads >= 1 {
        let overhead = hmx_2d_overhead(pipeline, is_matmul_id)
            + if bias_size > 0 {
                align_up(bias_size, HMX_TILE_BYTES)
            } else {
                0
            };
        let (per_n, per_m, per_mn) =
            hmx_2d_chunk_costs(wtype, u64::from(k), pipeline, u64::from(aligned_tile_size));
        if let Some((mc, nc, _)) = mm_hmx_compute_chunks(
            vtcm_budget,
            overhead,
            per_n,
            per_m,
            per_mn,
            m_for_chunks,
            u64::from(ne01_padded),
            u64::from(ne01_padded) * HMX_COST_W_DEQUANT,
            u64::from(m_for_cost) * HMX_COST_A_CONVERT,
        ) {
            let exact = hmx_2d_vtcm_size(
                wtype,
                u64::from(k),
                mc,
                nc,
                pipeline,
                if is_matmul_id { 0 } else { act_threads as u64 },
                u64::from(aligned_tile_size),
                bias_size,
            );
            if exact <= vtcm_budget {
                let mblocks = u64::from(m_for_cost).div_ceil(mc);
                if best.is_none_or(|(bm, (_, _, bat, _))| {
                    mblocks < bm || (mblocks == bm && act_threads as u64 > bat)
                }) {
                    best = Some((mblocks, (mc, nc, act_threads as u64, exact)));
                }
            }
        }
        act_threads = if act_threads == 1 { 0 } else { act_threads / 2 };
    }
    best.map(|(_, c)| c)
}

/// `struct htp_mm_hvx_vtcm_layout`: only the sizes the solver reads.
#[derive(Debug, Clone, Copy, Default)]
struct HvxLayout {
    src0_bytes: u64,
    bias_bytes: u64,
    dst_bytes: u64,
    total_bytes: u64,
}

/// `htp_mm_hvx_vtcm_layout_build`: the VTCM an HVX kernel needs for `act_nrows` activation rows.
/// Group A (activations, bias) persists; groups B (weights, output slice) and C (raw activation
/// staging) share the space after it.
#[allow(clippy::too_many_arguments)]
fn hvx_layout(
    kernel_type: u8,
    wtype: u32,
    ne10: u64,
    act_nrows: u64,
    n_threads: u64,
    dst_row_size: u64,
    src0_row_size: u64,
    bias_row_size: u64,
    n_prefetch: u64,
    is_matmul_id: bool,
    is_fused_nx: bool,
) -> HvxLayout {
    let mut bias_sz = if bias_row_size > 0 {
        align_up(bias_row_size, 128)
    } else {
        0
    };
    let (src0_sz, act_sz, dst_sz, act_raw_sz);
    if is_fused_nx {
        let aligned_tile_size = tile_padded_size(wtype);
        let n_k_tiles = ne10.div_ceil(tile_k(wtype));
        let tile_row_size = u64::from((n_k_tiles as u32).wrapping_mul(aligned_tile_size as u32));
        let weight_sz_per_thread = hex_round_up(
            u64::from((n_prefetch as u32).wrapping_mul(tile_row_size as u32)),
            128,
        );
        let q_act_sz = hex_round_up(tiled_act_row_size(wtype, ne10) * act_nrows, 128);
        src0_sz = weight_sz_per_thread * n_threads;
        act_sz = q_act_sz;
        bias_sz = 0;
        dst_sz = 0;
        act_raw_sz = hex_round_up(act_raw_row_size(ne10) * act_nrows, 128);
    } else if is_matmul_id {
        let aligned_tile_size = tile_padded_size(wtype);
        let n_k_tiles = ne10.div_ceil(tile_k(wtype));
        let tile_row_size = u64::from((n_k_tiles as u32).wrapping_mul(aligned_tile_size as u32));
        let src0_per_thread = align_up(
            u64::from((n_prefetch as u32).wrapping_mul(tile_row_size as u32)),
            256,
        );
        act_sz = align_up(tiled_act_row_size(wtype, ne10) * act_nrows, 256);
        src0_sz = src0_per_thread * n_threads;
        dst_sz = 0;
        bias_sz = 0;
        act_raw_sz = hex_round_up(act_raw_row_size(ne10) * act_nrows, 128);
    } else {
        let src0_row_size_padded = align_up(src0_row_size, 128);
        let dst_nrows = u64::from(act_nrows <= 1);
        match kernel_type {
            MM_KERNEL_HVX_F16_F16_VTCM => {
                let act_row = align_up(ne10 * 2, 128);
                act_sz = align_up(act_row * act_nrows, 256);
                src0_sz = align_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz = if dst_nrows > 0 {
                    align_up(dst_row_size, 128) * n_threads
                } else {
                    0
                };
                act_raw_sz = hex_round_up(hex_round_up(ne10 * 4, 128) * act_nrows, 128);
            }
            MM_KERNEL_HVX_F32_F32_VTCM | MM_KERNEL_HVX_BF16_F32_VTCM => {
                let act_row = align_up(ne10 * 4, 128);
                act_sz = align_up(act_row * act_nrows, 256);
                src0_sz = align_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz = if dst_nrows > 0 {
                    align_up(dst_row_size, 128) * n_threads
                } else {
                    0
                };
                act_raw_sz = 0;
            }
            MM_KERNEL_HVX_QUANT_BLOCK | MM_KERNEL_HVX_QUANT_ROW => {
                let q_act_row_size = tiled_act_row_size(wtype, ne10);
                let aligned_tile_size = tile_padded_size(wtype);
                let n_weight_tiles = ne10.div_ceil(tile_k(wtype));
                let tile_row_size =
                    u64::from((n_weight_tiles as u32).wrapping_mul(aligned_tile_size as u32));
                let tiled_vtcm_size = align_up(
                    u64::from((n_prefetch as u32).wrapping_mul(tile_row_size as u32)),
                    256,
                );
                src0_sz = tiled_vtcm_size * n_threads;
                act_sz = align_up(q_act_row_size * act_nrows, 256);
                let dst_slice_per_thread = if dst_nrows > 0 && act_nrows == 1 {
                    align_up(dst_row_size.div_ceil(n_threads), 128)
                } else {
                    0
                };
                dst_sz = dst_slice_per_thread * n_threads;
                act_raw_sz = hex_round_up(act_raw_row_size(ne10) * act_nrows, 128);
            }
            _ => {
                src0_sz = 0;
                act_sz = 0;
                dst_sz = 0;
                act_raw_sz = 0;
            }
        }
    }
    // Group A: activations, then the bias when there is one.
    let off_a = act_sz + bias_sz;
    // Group B (weights, output slice) and group C (raw activation staging) overlap.
    let group_b = src0_sz + dst_sz;
    HvxLayout {
        src0_bytes: src0_sz,
        bias_bytes: bias_sz,
        dst_bytes: dst_sz,
        total_bytes: off_a + group_b.max(act_raw_sz),
    }
}

/// `htp_mm_hvx_solve_vtcm_params`: the layout for all `act_nrows` rows when it fits, otherwise the
/// largest even row chunk that does (`None` when not even one row fits beside the fixed buffers).
/// Returns the layout and `m_chunk` (all the rows when unchunked).
#[allow(clippy::too_many_arguments)]
fn hvx_solve_vtcm_params(
    kernel_type: u8,
    wtype: u32,
    ne10: u64,
    act_nrows: u64,
    n_threads: u64,
    dst_row_size: u64,
    src0_row_size: u64,
    bias_row_size: u64,
    n_prefetch: u64,
    vtcm_budget: u64,
) -> Option<(HvxLayout, u64)> {
    let layout = |rows: u64| {
        hvx_layout(
            kernel_type,
            wtype,
            ne10,
            rows,
            n_threads,
            dst_row_size,
            src0_row_size,
            bias_row_size,
            n_prefetch,
            false,
            false,
        )
    };
    let l = layout(act_nrows);
    if l.total_bytes <= vtcm_budget {
        return Some((l, act_nrows));
    }
    let fixed_bytes = l.src0_bytes + l.bias_bytes + l.dst_bytes;
    if vtcm_budget <= fixed_bytes {
        return None;
    }
    let avail_act = vtcm_budget - fixed_bytes;
    let row_size = match kernel_type {
        MM_KERNEL_HVX_QUANT_ROW | MM_KERNEL_HVX_QUANT_BLOCK => tiled_act_row_size(wtype, ne10),
        MM_KERNEL_HVX_F16_F16_VTCM => hex_round_up(ne10 * 2, 128),
        _ => hex_round_up(ne10 * 4, 128),
    };
    if row_size == 0 {
        return None;
    }
    let mut eff_row_size = row_size;
    if matches!(
        kernel_type,
        MM_KERNEL_HVX_QUANT_ROW | MM_KERNEL_HVX_QUANT_BLOCK
    ) {
        eff_row_size += act_raw_row_size(ne10);
    }
    let mut m_chunk = (avail_act / eff_row_size) as u32;
    if m_chunk > 1 {
        m_chunk &= !1;
    }
    if u64::from(m_chunk) > act_nrows {
        m_chunk = act_nrows as u32;
    }
    if m_chunk < 1 {
        return None;
    }
    let mut l = layout(u64::from(m_chunk));
    while m_chunk > 2 && l.total_bytes > vtcm_budget {
        m_chunk -= 2;
        l = layout(u64::from(m_chunk));
    }
    (l.total_bytes <= vtcm_budget).then_some((l, u64::from(m_chunk)))
}

// --- the solver --------------------------------------------------------------------------------

/// `htp_matmul_is_hmx_eligible`: whether the tensors may run on the HMX at all (the solver may
/// still find no chunking that fits).
fn hmx_eligible(
    src0: &MmTensor,
    src1: &MmTensor,
    is_matmul_id: bool,
    is_batched: bool,
    min_rows: u64,
) -> bool {
    if src1.dtype != F32 && (src1.dtype != F16 || is_matmul_id) {
        return false;
    }
    let (ne00, ne01, ne11, ne12) = (src0.ne[0], src0.ne[1], src1.ne[1], src1.ne[2]);
    let wtype = src0.dtype;
    if ne01 % HMX_TILE_N_ROWS != 0 && is_matmul_id {
        return false;
    }
    if !is_tiled_weight_type(wtype) || ne00 % HMX_TILE_N_COLS != 0 {
        return false;
    }
    let is_tiled = src0.tiled;
    if is_quantized(wtype) && !is_tiled {
        return false;
    }
    if !is_matmul_id && is_batched && is_tiled {
        return false;
    }
    if src0.nb[0] > src0.nb[1] || src1.nb[0] > src1.nb[1] {
        return false;
    }
    let m = if is_matmul_id { ne12 } else { ne11 };
    m >= min_rows
}

/// `htp_kparams_solve_hmx_mm`. `dst` is `None` for the fused `MulMatNx`, which has no single
/// output to split across devices or to count expert rows from.
#[allow(clippy::too_many_arguments)]
fn solve_hmx_mm(
    sess: &MmSession,
    src0: &MmTensor,
    act: &MmTensor,
    dst: Option<&MmTensor>,
    wtype: u32,
    ne00_padded: u64,
    ne01_padded: u64,
    ne11_padded: u64,
    is_matmul_id: bool,
    is_batched: bool,
    src2_size: u64,
) -> Option<MmKernelParams> {
    let is_tiled = src0.tiled;
    let aligned_tile_size = if is_tiled { tile_padded_size(wtype) } else { 0 };
    let n_threads = sess.threads() as i32;
    let mdev_count = u64::from(sess.mdev_count);
    let (ne02, ne11, ne12, ne10) = (src0.ne[2], act.ne[1], act.ne[2], act.ne[0]);
    let vtcm_budget = sess.vtcm_bytes as u64;

    let mut m_for_solver = ne11;
    let mut m_for_solver_padded = ne11_padded;
    if !is_matmul_id && mdev_count > 1 && ne11 >= mdev_count {
        let dst_row_split = dst.is_none_or(MmTensor::can_row_partition);
        let act_row_split = act.can_row_partition();
        if dst_row_split && act_row_split {
            m_for_solver = ne11.div_ceil(mdev_count);
            m_for_solver_padded = hex_round_up(m_for_solver.max(32), 32);
        }
    }
    let pipeline = !is_matmul_id && hmx_pipeline(m_for_solver);
    let is_batched_val = !is_matmul_id && is_batched;
    let group_size = ne12.checked_div(ne02).unwrap_or(1);

    let mut chunking = None;
    if is_batched_val && wtype == F16 && !is_tiled && group_size > 1 {
        chunking = hmx_solve_batched(
            ne00_padded,
            ne01_padded,
            m_for_solver,
            group_size,
            n_threads,
            src2_size,
            vtcm_budget,
        );
    }
    let (m_chunk, n_chunk, act_threads, vtcm_size) = match chunking {
        Some(c) => c,
        None => {
            let mut m_id_rows = 0u64;
            if let (Some(dst), true) = (dst, is_matmul_id) {
                let n_experts = if ne02 > 0 { ne02 } else { 1 };
                let total_expert_rows = dst.ne[1] * dst.ne[2];
                let mut m_per_expert = total_expert_rows.div_ceil(n_experts);
                if mdev_count > 1 && dst.can_row_partition() {
                    m_per_expert = m_per_expert.div_ceil(mdev_count);
                }
                m_id_rows = hex_round_up(m_per_expert.max(32), 32);
            }
            let cost_m = if is_matmul_id {
                m_id_rows
            } else {
                m_for_solver
            };
            mm_hmx_solve_2d(
                wtype,
                ne00_padded as u32,
                m_id_rows as u32,
                ne01_padded as u32,
                m_for_solver_padded as u32,
                cost_m as u32,
                n_threads,
                pipeline,
                is_matmul_id,
                aligned_tile_size as u32,
                src2_size,
                vtcm_budget,
            )?
        }
    };

    Some(MmKernelParams {
        kernel_type: if is_batched && !is_matmul_id {
            MM_KERNEL_HMX_BATCHED
        } else {
            MM_KERNEL_HMX_2D
        },
        pipeline: u8::from(pipeline),
        n_hmx: 1,
        n_threads: wire_byte(sess.threads() as usize),
        n_act_threads: wire_byte(act_threads as usize),
        m_chunk: m_chunk as i32,
        n_chunk: n_chunk as i32,
        tile_size: if is_tiled { tile_size(wtype) as i32 } else { 0 },
        aligned_tile_size: aligned_tile_size as i32,
        act_row_size: tiled_act_row_size(wtype, ne10) as i32,
        vtcm_size: vtcm_size as i32,
        vtcm_bias_size: src2_size as i32,
        div_n_act_threads: fastdiv(act_threads),
        div_ne00_padded: fastdiv(ne00_padded),
        ..Default::default()
    })
}

/// `htp_kparams_solve_hvx_mm`: fills the HVX fields of `kp`, or leaves `kernel_type` at
/// `MM_KERNEL_UNSUPPORTED` when no variant takes the tensors within the VTCM budget.
#[allow(clippy::too_many_arguments)]
fn solve_hvx_mm(
    sess: &MmSession,
    src0: &MmTensor,
    src1: &MmTensor,
    dst: &MmTensor,
    wtype: u32,
    is_tiled: bool,
    is_matmul_id: bool,
    src2_row_size: u64,
    kp: &mut MmKernelParams,
) {
    let n_threads = u64::from(sess.threads());
    let vtcm_budget = sess.vtcm_bytes as u64;
    kp.n_hmx = 0;
    kp.n_threads = wire_byte(sess.threads() as usize);

    let ne10 = src1.ne[0];
    let src1_nrows = src1.ne[1] * src1.ne[2] * src1.ne[3];
    let unsupported = |kp: &mut MmKernelParams| kp.kernel_type = MM_KERNEL_UNSUPPORTED;

    if is_tiled {
        kp.tile_size = tile_size(wtype) as i32;
        kp.aligned_tile_size = tile_padded_size(wtype) as i32;
        let k_align = ne10.is_multiple_of(tile_k(wtype));
        let row_or_block = if src1_nrows < n_threads {
            MM_KERNEL_HVX_QUANT_BLOCK
        } else {
            MM_KERNEL_HVX_QUANT_ROW
        };
        // The deepest weight prefetch that fits: 16 rows ahead for a handful of activation rows
        // (the weights dominate), 2 once there are enough to hide the DMA.
        let max_prefetch: u64 = if src1_nrows > UPSTREAM_HMX_MIN_NROWS {
            2
        } else {
            16
        };
        if is_matmul_id {
            kp.kernel_type = row_or_block;
            kp.act_row_size = tiled_act_row_size(wtype, ne10) as i32;
            let mut best_n_prefetch = 2;
            let mut l = HvxLayout::default();
            let mut d = max_prefetch;
            while d >= 2 {
                l = hvx_layout(
                    kp.kernel_type,
                    wtype,
                    ne10,
                    src1_nrows,
                    n_threads,
                    0,
                    src0.nb[1],
                    0,
                    d,
                    true,
                    false,
                );
                if l.total_bytes <= vtcm_budget {
                    best_n_prefetch = d;
                    break;
                }
                d /= 2;
            }
            if l.total_bytes > vtcm_budget {
                unsupported(kp);
                return;
            }
            kp.n_prefetch = best_n_prefetch as u8;
            kp.vtcm_size = l.total_bytes as i32;
            kp.vtcm_bias_size = 0;
            return;
        }
        if k_align {
            kp.act_row_size = tiled_act_row_size(wtype, ne10) as i32;
            kp.kernel_type = row_or_block;
            let mut best_n_prefetch = 2;
            let mut d = max_prefetch;
            while d >= 2 {
                let l = hvx_layout(
                    kp.kernel_type,
                    wtype,
                    ne10,
                    src1_nrows,
                    n_threads,
                    dst.nb[1],
                    src0.nb[1],
                    src2_row_size,
                    d,
                    false,
                    false,
                );
                if l.total_bytes <= vtcm_budget {
                    best_n_prefetch = d;
                    break;
                }
                d /= 2;
            }
            if let Some((l, m_chunk)) = hvx_solve_vtcm_params(
                kp.kernel_type,
                wtype,
                ne10,
                src1_nrows,
                n_threads,
                dst.nb[1],
                src0.nb[1],
                src2_row_size,
                best_n_prefetch,
                vtcm_budget,
            ) {
                kp.n_prefetch = best_n_prefetch as u8;
                kp.m_chunk = if m_chunk < src1_nrows {
                    m_chunk as i32
                } else {
                    0
                };
                kp.vtcm_size = l.total_bytes as i32;
                kp.vtcm_bias_size = l.bias_bytes as i32;
                return;
            }
        }
        unsupported(kp);
        return;
    }
    if is_matmul_id {
        unsupported(kp);
        return;
    }
    if matches!(wtype, F16 | F32 | BF16) {
        let (ktype, act_elem) = match wtype {
            F32 => (MM_KERNEL_HVX_F32_F32_VTCM, 4),
            BF16 => (MM_KERNEL_HVX_BF16_F32_VTCM, 4),
            _ => (MM_KERNEL_HVX_F16_F16_VTCM, 2),
        };
        if let Some((l, m_chunk)) = hvx_solve_vtcm_params(
            ktype,
            wtype,
            ne10,
            src1_nrows,
            n_threads,
            dst.nb[1],
            src0.nb[1],
            src2_row_size,
            16,
            vtcm_budget,
        ) {
            kp.kernel_type = ktype;
            kp.m_chunk = if m_chunk < src1_nrows {
                m_chunk as i32
            } else {
                0
            };
            kp.act_row_size = hex_round_up(ne10 * act_elem, 128) as i32;
            kp.vtcm_size = l.total_bytes as i32;
            kp.vtcm_bias_size = l.bias_bytes as i32;
            kp.n_prefetch = 16;
            return;
        }
    }
    unsupported(kp);
}

/// The `finalize` of the solver: the activation-batch dividers the kernels index rows with.
fn finalize(kp: &mut MmKernelParams, src0: &MmTensor, act: &MmTensor) {
    let (ne02, ne03) = (src0.ne[2], src0.ne[3]);
    let (ne11, ne12, ne13) = (act.ne[1], act.ne[2], act.ne[3]);
    kp.div_ne12_ne1 = fastdiv(ne12 * ne11);
    kp.div_ne1 = fastdiv(ne11);
    kp.div_r2 = fastdiv(ne12.checked_div(ne02).unwrap_or(1));
    kp.div_r3 = fastdiv(ne13.checked_div(ne03).unwrap_or(1));
    kp.div_ne12 = fastdiv(ne12);
}

/// `wtype` and padded K as the solver derives them: a tiled F32 / BF16 weight is stored as F16.
fn weight_type(src0: &MmTensor) -> u32 {
    if src0.tiled && matches!(src0.dtype, F32 | BF16) {
        F16
    } else {
        src0.dtype
    }
}

/// `htp_kparams_solve_matmul_impl`: HMX when eligible and a chunking fits, else HVX. `dst.ne`
/// and `dst.nb[1]` feed the output slice and the expert row count; `is_matmul_id` selects the
/// routed-expert variants. `None` when no kernel takes the shapes within the VTCM budget.
fn solve_matmul_impl(
    sess: &MmSession,
    src0: &MmTensor,
    src1: &MmTensor,
    dst: &MmTensor,
    is_matmul_id: bool,
    src2_row_size: u64,
    src2_size: u64,
) -> Option<MmKernelParams> {
    let (ne00, ne01, ne02, ne03) = (src0.ne[0], src0.ne[1], src0.ne[2], src0.ne[3]);
    let (ne11, ne12, ne13) = (src1.ne[1], src1.ne[2], src1.ne[3]);
    let is_tiled = src0.tiled;
    let wtype = weight_type(src0);
    let ne00_padded = if is_tiled {
        hex_round_up(ne00, tile_k(wtype))
    } else {
        ne00
    };
    let ne11_padded = hex_round_up(ne11, 32);
    let ne01_tiled = hex_round_up(ne01, 32);
    let is_batched = ne02 * ne03 > 1 || ne12 * ne13 > 1;
    let vtcm_budget = sess.vtcm_bytes as u64;

    let mut kp = None;
    if sess.n_hmx > 0 && hmx_eligible(src0, src1, is_matmul_id, is_batched, sess.hmx_min_rows) {
        kp = solve_hmx_mm(
            sess,
            src0,
            src1,
            Some(dst),
            wtype,
            ne00_padded,
            ne01_tiled,
            ne11_padded,
            is_matmul_id,
            is_batched,
            src2_size,
        );
    }
    let mut kp = match kp {
        Some(kp) => kp,
        None => {
            let mut kp = MmKernelParams::default();
            solve_hvx_mm(
                sess,
                src0,
                src1,
                dst,
                wtype,
                is_tiled,
                is_matmul_id,
                src2_row_size,
                &mut kp,
            );
            kp
        }
    };
    finalize(&mut kp, src0, src1);
    // `(size_t) vtcm_size > budget` in the C++: a negative (overflowed) size compares huge.
    (kp.kernel_type != MM_KERNEL_UNSUPPORTED && kp.vtcm_size as i64 as u64 <= vtcm_budget)
        .then_some(kp)
}

/// `htp_matmul_check_types`: the operand types and layouts the matmul kernels read. `ids` is the
/// expert-index tensor of a `MulMatId` (`None` for `MulMat`); `prec_f32` is the op's
/// `GGML_PREC_F32` request, which the routed kernels cannot honour.
fn check_types(
    src0: &MmTensor,
    src1: &MmTensor,
    dst: &MmTensor,
    ids: Option<&MmTensor>,
    prec_f32: bool,
) -> bool {
    if dst.dtype != F32 {
        return false;
    }
    let quant = matches!(
        src0.dtype,
        Q1_0 | Q2_0 | Q4_0 | Q4_1 | Q8_0 | IQ4_NL | MXFP4 | Q4_K | Q5_K | Q6_K | Q3_K | Q2_K
    );
    let float = matches!(src0.dtype, F16 | F32 | BF16);
    if let Some(ids) = ids {
        if src1.dtype != F32 || ids.dtype != I32 || prec_f32 {
            return false;
        }
        if quant {
            return src0.is_contiguous()
                && !src0.is_permuted()
                && src0.ne[0].is_multiple_of(src0.blck());
        }
        if float {
            return src0.is_contiguous() && !src0.is_permuted() && src0.ne[0].is_multiple_of(32);
        }
        return false;
    }
    if src1.dtype != F32 && src1.dtype != F16 {
        return false;
    }
    if src1.ne[2] < src0.ne[2]
        || src1.ne[3] < src0.ne[3]
        || !src1.ne[2].is_multiple_of(src0.ne[2])
        || !src1.ne[3].is_multiple_of(src0.ne[3])
    {
        return false;
    }
    if quant {
        return src0.is_contiguous()
            && !src0.is_permuted()
            && src0.ne[0].is_multiple_of(src0.blck());
    }
    match src0.dtype {
        F32 => src1.dtype == F32 && src0.nb[1] >= src0.nb[0],
        F16 | BF16 => src0.nb[1] >= src0.nb[0],
        _ => false,
    }
}

// --- collapsing batched activations ---------------------------------------------------------

/// `htp_rows_stride`: the byte distance between consecutive rows when dims 1..3 are laid out as
/// one run of rows, `None` when they are not.
fn rows_stride(ne: &[u64; 4], nb: &[u64; 4]) -> Option<u64> {
    let (mut s, mut next) = (0u64, 0u64);
    for i in 1..4 {
        if ne[i] == 1 {
            continue;
        }
        if s == 0 {
            s = nb[i];
            next = s * ne[i];
            continue;
        }
        if nb[i] != next {
            return None;
        }
        next *= ne[i];
    }
    Some(if s != 0 { s } else { nb[1] })
}

/// `htp_matmul_can_collapse`: a 2-D weight against activations whose batch dims are plain
/// repeats of rows can run as one tall 2-D matmul.
fn matmul_can_collapse(src0: &MmTensor, src1: &MmTensor, dst: &MmTensor) -> bool {
    src0.ne[2] == 1
        && src0.ne[3] == 1
        && src1.ne[2] * src1.ne[3] > 1
        && src1.nb[0] == src1.type_size()
        && rows_stride(&src1.ne, &src1.nb).is_some()
        && dst.nb[0] == dst.type_size()
        && rows_stride(&dst.ne, &dst.nb).is_some()
}

/// `htp_tensor_collapse_rows`: dims 1..3 folded into one dim of rows.
fn collapse_rows(t: &MmTensor) -> MmTensor {
    let stride = rows_stride(&t.ne, &t.nb).unwrap_or(0);
    let mut c = *t;
    c.ne[1] = t.ne[1] * t.ne[2] * t.ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    c
}

/// The activations and output of a `MulMat` as the DSP should see them: folded into 2-D rows
/// when the weight is 2-D and the batch dims collapse (the host's `htp_opnode_collapse`), else
/// unchanged. Register these tensors, then [`solve_mul_mat`] them: the kernel params describe
/// the tensors they are solved against.
pub fn collapse_mul_mat(src0: &MmTensor, src1: &MmTensor, dst: &MmTensor) -> (MmTensor, MmTensor) {
    if matmul_can_collapse(src0, src1, dst) {
        (collapse_rows(src1), collapse_rows(dst))
    } else {
        (*src1, *dst)
    }
}

// --- entry points --------------------------------------------------------------------------

/// Kernel params of a `MulMat` (`htp_matmul_check_types` + `htp_kparams_solve_matmul_impl`) over
/// the tensors as given, which are the ones the DSP will read (collapse them first with
/// [`collapse_mul_mat`] if wanted). `None` where the host would fall back to the CPU.
pub fn solve_mul_mat(
    sess: &MmSession,
    src0: &MmTensor,
    src1: &MmTensor,
    dst: &MmTensor,
) -> Option<MmKernelParams> {
    if !check_types(src0, src1, dst, None, false) {
        return None;
    }
    solve_matmul_impl(sess, src0, src1, dst, false, 0, 0)
}

/// Kernel params of a `MulMatId` over `src0` `[k, n, n_expert]`, activations `src1`, expert
/// indices `ids` and `dst`. `prec_f32` is the op's `GGML_PREC_F32` request (refused).
pub fn solve_mul_mat_id(
    sess: &MmSession,
    src0: &MmTensor,
    src1: &MmTensor,
    ids: &MmTensor,
    dst: &MmTensor,
    prec_f32: bool,
) -> Option<MmKernelParams> {
    if !check_types(src0, src1, dst, Some(ids), prec_f32) {
        return None;
    }
    solve_matmul_impl(sess, src0, src1, dst, true, 0, 0)
}

/// The activations, output and bias of a `MulMatAdd` as the DSP should see them (the host's
/// `htp_opnode_collapse` of the fused node): folded into 2-D rows when the weight is 2-D, the batch
/// dims collapse and the bias is one row or has the output's shape; unchanged otherwise.
pub fn collapse_mul_mat_add(
    src0: &MmTensor,
    src1: &MmTensor,
    src2: Option<&MmTensor>,
    dst: &MmTensor,
) -> (MmTensor, MmTensor, Option<MmTensor>) {
    let add_can_collapse = matmul_can_collapse(src0, src1, dst)
        && src2.is_none_or(|s2| {
            s2.ne[1] * s2.ne[2] * s2.ne[3] == 1
                || (s2.nb[0] == s2.type_size()
                    && s2.same_shape(dst)
                    && rows_stride(&s2.ne, &s2.nb).is_some())
        });
    if !add_can_collapse {
        return (*src1, *dst, src2.copied());
    }
    let src2c = src2.map(|s| {
        if s.ne[1] * s.ne[2] * s.ne[3] > 1 {
            collapse_rows(s)
        } else {
            *s
        }
    });
    (collapse_rows(src1), collapse_rows(dst), src2c)
}

/// `htp_kparams_solve_matmul_add`: a `MulMat` whose F32 bias `src2` (one row, or `dst`'s shape) the
/// kernel adds to the output. The params describe the tensors [`collapse_mul_mat_add`] returns,
/// which the host registers instead of the originals; the bias row size and its padded byte size
/// enter the VTCM layout. The fusion rules that decide whether to use it (HVX only for one row, no
/// HMX efficiency loss) belong to the caller. Unlike [`solve_mul_mat`] it does not check operand
/// types.
pub fn solve_mul_mat_add(
    sess: &MmSession,
    src0: &MmTensor,
    src1: &MmTensor,
    src2: Option<&MmTensor>,
    dst: &MmTensor,
) -> Option<MmKernelParams> {
    let (src1, dst, src2) = collapse_mul_mat_add(src0, src1, src2, dst);
    let row = src2.map_or(0, |s| s.nb[1]);
    let size = src2.map_or(0, |s| align_up(s.nbytes(), 128));
    solve_matmul_impl(sess, src0, &src1, &dst, false, row, size)
}

/// `htp_kparams_solve_mmnx`: one activation shared by `n_weights` weights of the same K and type
/// (a fused Q/K/V, gate/up). The params are solved for `src0`, the first weight, which should
/// have the largest N. Returns `None` when no fused kernel takes it; whether the DSP has the
/// fused kernel the solver picked is [`mmnx_kernel_supported`].
pub fn solve_mmnx(
    sess: &MmSession,
    src0: &MmTensor,
    act: &MmTensor,
    n_weights: u32,
) -> Option<MmKernelParams> {
    let n_threads = u64::from(sess.threads());
    let vtcm_budget = sess.vtcm_bytes as u64;
    let (ne00, ne01, ne02, ne03) = (src0.ne[0], src0.ne[1], src0.ne[2], src0.ne[3]);
    let (ne10, ne11, ne12, ne13) = (act.ne[0], act.ne[1], act.ne[2], act.ne[3]);
    let is_tiled = src0.tiled;
    let wtype = weight_type(src0);
    let ne00_padded = if is_tiled {
        hex_round_up(ne00, 32)
    } else {
        ne00
    };
    let ne11_padded = hex_round_up(ne11, 32);
    let ne01_tiled = hex_round_up(ne01, 32);
    let is_batched = ne02 * ne03 > 1 || ne12 * ne13 > 1;

    let mut kp = MmKernelParams {
        n_threads: wire_byte(sess.threads() as usize),
        ..Default::default()
    };
    let mut done = false;
    if sess.n_hmx > 0
        && hmx_eligible(src0, act, false, is_batched, sess.hmx_min_rows)
        && let Some(mut hmx) = solve_hmx_mm(
            sess,
            src0,
            act,
            None,
            wtype,
            ne00_padded,
            ne01_tiled,
            ne11_padded,
            false,
            is_batched,
            0,
        )
    {
        hmx.n_weights = wire_byte(n_weights as usize);
        kp = hmx;
        done = true;
    }
    if !done {
        if !is_tiled || act.dtype != F32 || !ne10.is_multiple_of(32) {
            return None;
        }
        let act_nrows = ne11 * ne12 * ne13;
        let act_row_size = tiled_act_row_size(wtype, ne10);
        let src0_row_size = src0.nb[1];
        let layout = |d: u64| {
            hvx_layout(
                MM_KERNEL_HVX_QUANT_ROW,
                wtype,
                ne10,
                act_nrows,
                n_threads,
                0,
                src0_row_size,
                0,
                d,
                false,
                true,
            )
        };
        let max_prefetch: u64 = if act_nrows > UPSTREAM_HMX_MIN_NROWS {
            2
        } else {
            16
        };
        let mut best_n_prefetch = 2;
        let mut d = max_prefetch;
        while d >= 2 {
            if layout(d).total_bytes <= vtcm_budget {
                best_n_prefetch = d;
                break;
            }
            d /= 2;
        }
        let l = layout(best_n_prefetch);
        if l.total_bytes > vtcm_budget {
            return None;
        }
        kp.kernel_type = MM_KERNEL_HVX_QUANT_ROW;
        kp.tile_size = tile_size(wtype) as i32;
        kp.aligned_tile_size = tile_padded_size(wtype) as i32;
        kp.act_row_size = act_row_size as i32;
        kp.vtcm_bias_size = 0;
        kp.vtcm_size = l.total_bytes as i32;
        kp.n_prefetch = best_n_prefetch as u8;
        kp.n_weights = wire_byte(n_weights as usize);
    }
    finalize(&mut kp, src0, act);
    (kp.vtcm_size as i64 as u64 <= vtcm_budget).then_some(kp)
}

/// Whether the DSP has the fused kernel behind `kp` (`is_supported_mul_mat_nx_kernel`): the HMX
/// 2-D kernel, or the row-parallel tiled HVX one.
pub fn mmnx_kernel_supported(src0: &MmTensor, kp: &MmKernelParams) -> bool {
    if kp.n_hmx != 0 {
        return kp.kernel_type == MM_KERNEL_HMX_2D;
    }
    src0.tiled && kp.kernel_type == MM_KERNEL_HVX_QUANT_ROW
}

/// `htp_kparams_solve_mmidnx`: the routed-expert counterpart of [`solve_mmnx`]. `dst` is the
/// first output.
pub fn solve_mmidnx(
    sess: &MmSession,
    src0: &MmTensor,
    act: &MmTensor,
    dst: &MmTensor,
    n_weights: u32,
) -> Option<MmKernelParams> {
    let ok = solve_matmul_impl(sess, src0, act, dst, true, 0, 0);
    // The C++ fills `n_weights` on the result whether or not the solve succeeded; a failed one
    // is discarded either way.
    ok.map(|mut kp| {
        kp.n_weights = wire_byte(n_weights as usize);
        kp
    })
}

/// Whether the DSP has the fused routed kernel behind `kp` (`is_supported_mul_mat_id_nx_kernel`).
pub fn mmidnx_kernel_supported(src0: &MmTensor, kp: &MmKernelParams) -> bool {
    if kp.n_hmx != 0 {
        return kp.kernel_type == MM_KERNEL_HMX_2D;
    }
    src0.tiled
        && matches!(
            kp.kernel_type,
            MM_KERNEL_HVX_QUANT_ROW | MM_KERNEL_HVX_QUANT_BLOCK
        )
}

// --- builders for cera's call sites -----------------------------------------------------------
//
// The call sites know shapes, not tensors, and pick HMX or HVX themselves (the model's HMX knob,
// the per-run row count), so these take scalars, build the tensors the host would have held and
// solve them. A builder with HMX off solves a session without HMX units, which is exactly the
// solver's own HVX fallback.

/// The session a builder solves: `n_threads` DSP threads, `vtcm_budget` bytes, HMX on or off.
fn builder_session(n_threads: u32, vtcm_budget: usize, hmx: bool) -> MmSession {
    MmSession::new(n_threads.max(1), vtcm_budget, u32::from(hmx))
}

fn words(kp: &MmKernelParams) -> [i32; 32] {
    kp.to_words()
}

/// The HVX kernel params of a MulMat with tiled quantized weights (Q4_0, Q8_0, Q4_K, Q6_K..):
/// `ne10` is the activation width (K), `ne11` x `ne12` the activation rows and batches (1 for
/// single-token decode), `dst_row_size` the output row stride in bytes (`4 * N`). Rows the VTCM
/// cannot hold at once come back chunked (`m_chunk`); `None` when no HVX kernel fits the budget.
pub fn build_mul_mat_kernel_params(
    wtype: HtpDataType,
    ne10: usize,
    ne11: u32,
    ne12: u32,
    dst_row_size: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    if ne10 == 0 || ne11 == 0 || ne12 == 0 || dst_row_size < 4 {
        return None;
    }
    let sess = builder_session(n_threads, vtcm_budget, false);
    let n = (dst_row_size / 4) as u64;
    let k_padded = (ne10 as u64).next_multiple_of(32);
    let src0 = MmTensor::tiled_weight(wtype as u32, [k_padded, n.next_multiple_of(32), 1, 1]);
    let src1 = MmTensor::f32_rows(
        [ne10 as u64, u64::from(ne11), u64::from(ne12), 1],
        ne10 as u64 * 4,
    );
    let dst = MmTensor::f32_rows(
        [n, u64::from(ne11), u64::from(ne12), 1],
        dst_row_size as u64,
    );
    solve_mul_mat(&sess, &src0, &src1, &dst).as_ref().map(words)
}

/// The HMX kernel params of a MulMat over `m` activation rows against a `[k, n]` weight
/// (`k` and `n` padded to 32): tiled for the quantized types, plain row-major for F16. `None`
/// when the HMX cannot take it (too few rows, an ineligible shape, or no chunking that fits
/// `vtcm_budget`); the caller then takes the HVX builder.
pub fn build_hmx_mm_kernel_params(
    wtype: HtpDataType,
    k: usize,
    n: usize,
    m: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    let sess = builder_session(n_threads, vtcm_budget, true);
    let ne = [k as u64, n as u64, 1, 1];
    let src0 = if wtype == HtpDataType::F16 {
        MmTensor::contiguous(wtype as u32, ne)
    } else {
        MmTensor::tiled_weight(wtype as u32, ne)
    };
    let src1 = MmTensor::f32_rows([k as u64, m as u64, 1, 1], k as u64 * 4);
    let dst = MmTensor::f32_rows([n as u64, m as u64, 1, 1], n as u64 * 4);
    let kp = solve_mul_mat(&sess, &src0, &src1, &dst)?;
    (kp.n_hmx == 1).then(|| words(&kp))
}

/// Whether `m` rows against a `[k, n]` weight (K and N padded to 32) may take the HMX at all,
/// before any chunking is tried: the solver's eligibility rule at cera's row floor.
pub fn mm_is_hmx_eligible(wtype: HtpDataType, k: usize, n: usize, m: usize) -> bool {
    if k == 0 || n == 0 || m < HMX_MM_MIN_NROWS || !k.is_multiple_of(32) || !n.is_multiple_of(32) {
        return false;
    }
    let ne = [k as u64, n as u64, 1, 1];
    let src0 = if wtype == HtpDataType::F16 {
        MmTensor::contiguous(wtype as u32, ne)
    } else {
        MmTensor::tiled_weight(wtype as u32, ne)
    };
    let src1 = MmTensor::f32_rows([k as u64, m as u64, 1, 1], k as u64 * 4);
    hmx_eligible(&src0, &src1, false, false, HMX_MM_MIN_NROWS as u64)
}

/// The weight's `nb[1]` as the HMX kernel strides it: one tiled row (`htp_tiled_row_size`), or
/// the plain row for F16.
pub fn mm_hmx_nb1(wtype: HtpDataType, k: usize) -> usize {
    tiled_row_size(wtype as u32, k as u64) as usize
}

/// The fused `MulMatNx` kernel params for `n_weights` tiled weights sharing one activation of
/// `ne10` columns by `rows` rows: `ne00` x `n0` is the first weight (the one with the most rows,
/// which the params are sized for). HMX per `hmx` and the row floor, else the fused HVX kernel.
/// `None` when the solver or the DSP's fused kernels do not take it; the caller then runs the
/// weights as separate matmuls.
#[allow(clippy::too_many_arguments)]
pub fn build_mul_mat_nx_kernel_params(
    wtype: HtpDataType,
    ne00: usize,
    ne10: usize,
    n0: usize,
    n_weights: usize,
    rows: usize,
    n_threads: u32,
    vtcm_budget: usize,
    hmx: bool,
) -> Option<[i32; 32]> {
    if [ne00, ne10, n0, n_weights, rows].contains(&0) {
        return None;
    }
    let sess = builder_session(n_threads, vtcm_budget, hmx);
    let src0 = MmTensor::tiled_weight(wtype as u32, [ne00 as u64, n0 as u64, 1, 1]);
    let act = MmTensor::f32_rows([ne10 as u64, rows as u64, 1, 1], ne10 as u64 * 4);
    let kp = solve_mmnx(&sess, &src0, &act, n_weights as u32)?;
    mmnx_kernel_supported(&src0, &kp).then(|| words(&kp))
}

/// The most activation rows the fused HVX `MulMatNx` can take in `vtcm_budget`. That kernel has
/// no row chunking, so a caller with more rows runs the projections separately (each chunks
/// itself). The deepest weight prefetch is the one that fits the most rows, so this is the
/// layout at prefetch 2.
pub(crate) fn mm_hvx_fused_nx_max_rows(
    wtype: HtpDataType,
    ne10: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> usize {
    let n_threads = u64::from(n_threads.max(1));
    let fits = |rows: u64| {
        hvx_layout(
            MM_KERNEL_HVX_QUANT_ROW,
            wtype as u32,
            ne10 as u64,
            rows,
            n_threads,
            0,
            0,
            0,
            2,
            false,
            true,
        )
        .total_bytes
            <= vtcm_budget as u64
    };
    let row_q = tiled_act_row_size(wtype as u32, ne10 as u64).max(1);
    let (mut lo, mut hi) = (0u64, vtcm_budget as u64 / row_q);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo as usize
}

/// Shape of an F32 x F32 matmul, in the strides the host solver reads. `ne0x` are the weight-side
/// (`src0`) dims, `ne1x` the activation-side (`src1`) dims; the contraction length is `ne00`.
/// `src0_nb0` / `src0_nb1` are `src0`'s element and row strides in bytes (a transposed view has
/// `nb0 > nb1`, which no kernel takes) and `dst_nb1` the destination's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulMatF32Shape {
    pub ne00: usize,
    pub ne01: usize,
    pub ne02: usize,
    pub ne03: usize,
    pub src0_nb0: usize,
    pub src0_nb1: usize,
    pub ne11: usize,
    pub ne12: usize,
    pub ne13: usize,
    pub dst_nb1: usize,
}

/// Kernel params for an F32 x F32 `MulMat` on the HVX path: `dst[n, m] = sum_k src0[k, n] *
/// src1[k, m]`, batched over `ne02 x ne03` weight matrices that `ne12 x ne13` activation batches
/// are broadcast over. It never takes the HMX: attention scores keep their f32 accumulation.
/// `None` when even one activation row does not fit `vtcm_budget` (the host falls back to the
/// CPU then), or an extent is zero.
pub fn build_mul_mat_f32_kernel_params(
    shape: MulMatF32Shape,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    let MulMatF32Shape {
        ne00,
        ne01,
        ne02,
        ne03,
        src0_nb0,
        src0_nb1,
        ne11,
        ne12,
        ne13,
        dst_nb1,
    } = shape;
    // The kernel divides by these (row indices and the weights' batch broadcast).
    if [ne00, ne01, ne02, ne03, ne11, ne12, ne13].contains(&0) {
        return None;
    }
    let sess = builder_session(n_threads, vtcm_budget, false);
    let u = |v: usize| v as u64;
    let src0 = MmTensor::new(
        F32,
        [u(ne00), u(ne01), u(ne02), u(ne03)],
        [
            u(src0_nb0),
            u(src0_nb1),
            u(src0_nb1) * u(ne01),
            u(src0_nb1) * u(ne01) * u(ne02),
        ],
        false,
    );
    let src1 = MmTensor::f32_rows([u(ne00), u(ne11), u(ne12), u(ne13)], u(ne00) * 4);
    let dst = MmTensor::f32_rows([u(ne01), u(ne11), u(ne12), u(ne13)], u(dst_nb1));
    solve_mul_mat(&sess, &src0, &src1, &dst).as_ref().map(words)
}

/// The HVX kernel params of a `MulMatId` over the given operands (`src0` `[k, n, n_expert]`
/// tiled quantized, `ids` the I32 expert indices): cera's routed matmuls run one token at a time,
/// below the row floor of the HMX kernels, so this never solves for the HMX. `None` when the
/// routed HVX kernels do not take it within `vtcm_budget`.
pub fn build_mul_mat_id_kernel_params(
    src0: &MmTensor,
    src1: &MmTensor,
    ids: &MmTensor,
    dst: &MmTensor,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    // like the other cera builders: an empty extent is refused, not solved as a zero-sized matmul
    if [src0, src1, ids, dst].iter().any(|t| t.ne.contains(&0)) {
        return None;
    }
    let sess = builder_session(n_threads, vtcm_budget, false);
    solve_mul_mat_id(&sess, src0, src1, ids, dst, false)
        .as_ref()
        .map(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(s: &str) -> Vec<u64> {
        s.split_whitespace().map(|w| w.parse().unwrap()).collect()
    }

    /// "type ne0..3 nb0..3 flags", the oracle's `mm_print_t`.
    fn tensor(s: &str) -> MmTensor {
        let n = nums(s);
        assert_eq!(n.len(), 10, "{s}");
        MmTensor::new(
            n[0] as u32,
            [n[1], n[2], n[3], n[4]],
            [n[5], n[6], n[7], n[8]],
            n[9] & u64::from(HTP_TENSOR_TILED) != 0,
        )
    }

    /// "threads vtcm hmx mdev", the oracle's `mm_print_cfg`.
    fn session(s: &str) -> MmSession {
        let n = nums(s);
        MmSession {
            n_threads: n[0] as u32,
            vtcm_bytes: n[1] as usize,
            n_hmx: n[2] as u32,
            mdev_count: n[3] as u32,
            hmx_min_rows: UPSTREAM_HMX_MIN_NROWS + 1,
        }
    }

    /// The 23 words of `htp_mm_kernel_params` (the rest of the blob is zero), `None` for
    /// "unsupported".
    fn want_words(s: &str) -> Option<[i32; 32]> {
        let s = s.trim();
        if s == "unsupported" {
            return None;
        }
        let mut w = [0i32; 32];
        let n = nums(s);
        assert_eq!(n.len(), 23, "{s}");
        for (dst, v) in w.iter_mut().zip(n) {
            *dst = v as u32 as i32;
        }
        Some(w)
    }

    fn assert_params(line: &str, got: Option<MmKernelParams>, want: Option<[i32; 32]>) {
        assert_eq!(got.as_ref().map(MmKernelParams::to_words), want, "{line}");
        if let Some(kp) = got {
            assert_eq!(MmKernelParams::from_words(&kp.to_words()), kp);
        }
    }

    fn golden(text: &str) -> impl Iterator<Item = &str> {
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
    }

    /// Kernel variants a golden file's solved lines reach: `(kernel_type, n_hmx)` counts.
    #[derive(Default)]
    struct Reach {
        cases: usize,
        unsupported: usize,
        variants: std::collections::BTreeMap<(u8, u8), usize>,
    }

    impl Reach {
        fn note(&mut self, want: &Option<[i32; 32]>) {
            self.cases += 1;
            match want {
                None => self.unsupported += 1,
                Some(w) => {
                    let kp = MmKernelParams::from_words(w);
                    *self.variants.entry((kp.kernel_type, kp.n_hmx)).or_default() += 1;
                }
            }
        }

        fn has(&self, kernel: u8, hmx: u8) -> usize {
            self.variants.get(&(kernel, hmx)).copied().unwrap_or(0)
        }
    }

    /// Check every line of a `MUL_MAT` golden file (`oracle/matmul*.cpp`): the node's collapse,
    /// the params solved against the collapsed tensors, and against the original ones.
    fn check_mul_mat_golden(text: &str) -> (Reach, usize) {
        let mut reach = Reach::default();
        let mut collapsed = 0usize;
        for line in golden(text) {
            let (l, r) = line.split_once("=>").unwrap();
            let mut sides = l.split('|');
            let w = tensor(sides.next().unwrap());
            let x = tensor(sides.next().unwrap());
            let d = tensor(sides.next().unwrap());
            let sess = session(sides.next().unwrap());
            let mut res = r.split('|');
            let node_t = |s: &str, orig: &MmTensor| match s.trim() {
                "=" => *orig,
                t => tensor(t),
            };
            let (xc, dc) = collapse_mul_mat(&w, &x, &d);
            assert_eq!(xc, node_t(res.next().unwrap(), &x), "{line}");
            assert_eq!(dc, node_t(res.next().unwrap(), &d), "{line}");
            let want = want_words(res.next().unwrap());
            assert_params(line, solve_mul_mat(&sess, &w, &xc, &dc), want);
            reach.note(&want);
            if let Some(raw) = res.next().unwrap().trim().strip_prefix("raw") {
                // the same op solved without collapsing: the tensors the caller registered
                assert_params(line, solve_mul_mat(&sess, &w, &x, &d), want_words(raw));
                collapsed += 1;
            }
        }
        (reach, collapsed)
    }

    /// Every case of `testdata/matmul_golden.txt`: llama.cpp's real op-node path (collapse, type
    /// check, solver) for tiled quantized weights, over the shapes, row counts, session limits and
    /// operand layouts cera meets, plus the operands the host sends to the CPU.
    #[test]
    fn mul_mat_params_match_the_llama_cpp_host_code() {
        let (reach, collapsed) = check_mul_mat_golden(include_str!("testdata/matmul_golden.txt"));
        assert!(
            reach.cases > 8000 && reach.unsupported > 1500,
            "golden file looks truncated: {} cases, {} unsupported",
            reach.cases,
            reach.unsupported
        );
        assert!(collapsed > 100, "only {collapsed} collapsed cases");
        // the HMX 2-D kernel, and both HVX quantized kernels
        assert!(reach.has(MM_KERNEL_HMX_2D, 1) > 2000);
        assert!(reach.has(MM_KERNEL_HVX_QUANT_ROW, 0) > 1000);
        assert!(reach.has(MM_KERNEL_HVX_QUANT_BLOCK, 0) > 1000);
    }

    /// Every case of `testdata/matmul_flat_golden.txt`: plain F16 / F32 / BF16 weights (the HMX
    /// flat and batched kernels, the HVX flat kernels) and collapsing batched activations.
    #[test]
    fn mul_mat_flat_params_match_the_llama_cpp_host_code() {
        let (reach, collapsed) =
            check_mul_mat_golden(include_str!("testdata/matmul_flat_golden.txt"));
        assert!(
            reach.cases > 9000 && reach.unsupported > 100,
            "golden file looks truncated: {} cases, {} unsupported",
            reach.cases,
            reach.unsupported
        );
        assert!(collapsed > 100, "only {collapsed} collapsed cases");
        assert!(reach.has(MM_KERNEL_HMX_2D, 1) > 1000);
        assert!(reach.has(MM_KERNEL_HMX_BATCHED, 1) > 100);
        assert!(reach.has(MM_KERNEL_HVX_F16_F16_VTCM, 0) > 500);
        assert!(reach.has(MM_KERNEL_HVX_F32_F32_VTCM, 0) > 500);
        assert!(reach.has(MM_KERNEL_HVX_BF16_F32_VTCM, 0) >= 40);
    }

    /// Every case of `testdata/matmul_id_golden.txt`: llama.cpp's op-node path for `MUL_MAT_ID`
    /// with tiled quantized and plain dense expert weights, decode and prefill token counts,
    /// session limits (several devices included), and the operands the host refuses.
    #[test]
    fn mul_mat_id_params_match_the_llama_cpp_host_code() {
        // `GGML_PREC_F32`
        const PREC_F32: u64 = 10;
        let mut reach = Reach::default();
        for line in golden(include_str!("testdata/matmul_id_golden.txt")) {
            let (l, r) = line.split_once("=>").unwrap();
            let mut sides = l.split('|');
            let w = tensor(sides.next().unwrap());
            let x = tensor(sides.next().unwrap());
            let ids = tensor(sides.next().unwrap());
            let d = tensor(sides.next().unwrap());
            let prec = nums(sides.next().unwrap())[0];
            let sess = session(sides.next().unwrap());
            let want = want_words(r);
            assert_params(
                line,
                solve_mul_mat_id(&sess, &w, &x, &ids, &d, prec == PREC_F32),
                want,
            );
            reach.note(&want);
        }
        assert!(
            reach.cases > 7000 && reach.unsupported > 500,
            "golden file looks truncated: {} cases, {} unsupported",
            reach.cases,
            reach.unsupported
        );
        assert!(reach.has(MM_KERNEL_HMX_2D, 1) > 1000);
        assert!(reach.has(MM_KERNEL_HVX_QUANT_ROW, 0) > 500);
        assert!(reach.has(MM_KERNEL_HVX_QUANT_BLOCK, 0) > 500);
    }

    /// Every case of `testdata/matmul_add_golden.txt`: `htp_kparams_solve_matmul_add` (a matmul
    /// that adds an F32 bias row or a full-shape bias) and the collapse of the fused node.
    #[test]
    fn mul_mat_add_params_match_the_llama_cpp_host_code() {
        let mut reach = Reach::default();
        let mut collapsed = 0usize;
        for line in golden(include_str!("testdata/matmul_add_golden.txt")) {
            let (l, r) = line.split_once("=>").unwrap();
            let mut sides = l.split('|');
            let w = tensor(sides.next().unwrap());
            let x = tensor(sides.next().unwrap());
            let bias = match sides.next().unwrap().trim() {
                "-" => None,
                b => Some(tensor(b)),
            };
            let d = tensor(sides.next().unwrap());
            let sess = session(sides.next().unwrap());
            let mut res = r.split('|');
            let node_t = |s: &str, orig: &MmTensor| match s.trim() {
                "=" => *orig,
                t => tensor(t),
            };
            let (xc, dc, bc) = collapse_mul_mat_add(&w, &x, bias.as_ref(), &d);
            assert_eq!(xc, node_t(res.next().unwrap(), &x), "{line}");
            assert_eq!(dc, node_t(res.next().unwrap(), &d), "{line}");
            let want_bias = res.next().unwrap().trim();
            match (&bc, &bias) {
                (None, None) => assert_eq!(want_bias, "-", "{line}"),
                (Some(bc), Some(b)) => assert_eq!(*bc, node_t(want_bias, b), "{line}"),
                _ => panic!("bias presence differs: {line}"),
            }
            collapsed += usize::from(xc != x);
            let want = want_words(res.next().unwrap());
            assert_params(
                line,
                solve_mul_mat_add(&sess, &w, &x, bias.as_ref(), &d),
                want,
            );
            reach.note(&want);
        }
        assert!(
            reach.cases > 3500 && reach.unsupported > 100 && collapsed > 100,
            "golden file looks truncated: {} cases, {} unsupported, {collapsed} collapsed",
            reach.cases,
            reach.unsupported
        );
        assert!(reach.has(MM_KERNEL_HMX_2D, 1) > 500);
        assert!(reach.has(MM_KERNEL_HVX_QUANT_ROW, 0) > 100);
    }

    /// Split an `mmnx` / `mmidnx` result: "unsupported", or the `supported` flag then the words.
    fn want_nx(s: &str) -> (Option<[i32; 32]>, bool) {
        match s.trim().strip_prefix(['0', '1']) {
            Some(words) if !s.trim().starts_with("unsupported") => {
                (want_words(words), s.trim().starts_with('1'))
            }
            _ => (None, false),
        }
    }

    /// Every case of `testdata/mmnx_golden.txt`: `htp_kparams_solve_mmnx` for fused matmuls that
    /// share an activation, and whether the DSP has the fused kernel the solver picked.
    #[test]
    fn fused_mul_mat_nx_params_match_the_llama_cpp_host_code() {
        let mut reach = Reach::default();
        let (mut supported, mut unsupported_kernel) = (0usize, 0usize);
        for line in golden(include_str!("testdata/mmnx_golden.txt")) {
            let (l, r) = line.split_once("=>").unwrap();
            let mut sides = l.split('|');
            let w = tensor(sides.next().unwrap());
            let x = tensor(sides.next().unwrap());
            let n_weights = nums(sides.next().unwrap())[0] as u32;
            let sess = session(sides.next().unwrap());
            let (want, want_supported) = want_nx(r);
            let got = solve_mmnx(&sess, &w, &x, n_weights);
            assert_params(line, got, want);
            reach.note(&want);
            if let Some(kp) = got {
                assert_eq!(mmnx_kernel_supported(&w, &kp), want_supported, "{line}");
                if want_supported {
                    supported += 1;
                } else {
                    unsupported_kernel += 1;
                }
            }
        }
        assert!(
            reach.cases > 9000 && reach.unsupported > 500,
            "golden file looks truncated: {} cases, {} unsupported",
            reach.cases,
            reach.unsupported
        );
        assert!(supported > 3000 && unsupported_kernel > 10);
        assert!(reach.has(MM_KERNEL_HMX_2D, 1) > 1000);
        assert!(reach.has(MM_KERNEL_HVX_QUANT_ROW, 0) > 1000);
    }

    /// Every case of `testdata/mmidnx_golden.txt`: `htp_kparams_solve_mmidnx` for fused routed
    /// matmuls and whether the DSP has the fused routed kernel the solver picked.
    #[test]
    fn fused_mul_mat_id_nx_params_match_the_llama_cpp_host_code() {
        let mut reach = Reach::default();
        let mut supported = 0usize;
        for line in golden(include_str!("testdata/mmidnx_golden.txt")) {
            let (l, r) = line.split_once("=>").unwrap();
            let mut sides = l.split('|');
            let w = tensor(sides.next().unwrap());
            let x = tensor(sides.next().unwrap());
            let d = tensor(sides.next().unwrap());
            let n_weights = nums(sides.next().unwrap())[0] as u32;
            let sess = session(sides.next().unwrap());
            let (want, want_supported) = want_nx(r);
            let got = solve_mmidnx(&sess, &w, &x, &d, n_weights);
            assert_params(line, got, want);
            reach.note(&want);
            if let Some(kp) = got {
                assert_eq!(mmidnx_kernel_supported(&w, &kp), want_supported, "{line}");
                supported += usize::from(want_supported);
            }
        }
        assert!(
            reach.cases > 3500 && reach.unsupported > 300 && supported > 1500,
            "golden file looks truncated: {} cases, {} unsupported",
            reach.cases,
            reach.unsupported
        );
    }

    /// Every case of `testdata/hmx_chunks_golden.txt`: llama.cpp's `htp_mm_hmx_compute_chunks`
    /// (`matmul-ops.h`) over a grid of VTCM budgets, costs and ragged N, whose tie-break prefers
    /// the candidate that wastes the fewest columns in the last N block.
    #[test]
    fn hmx_chunk_search_matches_the_llama_cpp_header() {
        let (mut cases, mut fits) = (0usize, 0usize);
        for line in golden(include_str!("testdata/hmx_chunks_golden.txt")) {
            let (inputs, want) = line.split_once("=>").unwrap();
            let v = nums(inputs);
            // the C++ return code is -1 for "no chunking", so it does not parse as unsigned
            let w: Vec<i64> = want
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let got = mm_hmx_compute_chunks(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
            if w[0] == 0 {
                assert_eq!(got, Some((w[1] as u64, w[2] as u64, w[3] as u64)), "{line}");
                fits += 1;
            } else {
                assert_eq!(got, None, "{line}");
            }
            cases += 1;
        }
        assert!(
            cases > 20000 && fits > 15000,
            "golden file looks truncated: {cases} cases, {fits} fit"
        );
    }

    /// Every case of `testdata/hmx_solve_golden.txt`: llama.cpp's `htp_mm_hmx_solve_2d_params`
    /// (cost model, exact layout check and activation-thread search; plain, F16/F32, biased and
    /// routed) and `htp_mm_hmx_solve_batched_params` (grouped GQA), at several thread counts and
    /// VTCM budgets down to 64 KiB.
    #[test]
    fn hmx_solvers_match_the_llama_cpp_header() {
        let (mut cases, mut no_fit, mut halved) = (0usize, 0usize, 0usize);
        for line in golden(include_str!("testdata/hmx_solve_golden.txt")) {
            let (inputs, want) = line.split_once("=>").unwrap();
            let (kind, inputs) = inputs.trim().split_once(' ').unwrap();
            let v = nums(inputs);
            let w = nums(want);
            let (got, threads) = match kind {
                "2d" => (
                    mm_hmx_solve_2d(
                        v[0] as u32,
                        v[1] as u32,
                        v[2] as u32,
                        v[3] as u32,
                        v[4] as u32,
                        v[5] as u32,
                        v[6] as i32,
                        v[7] != 0,
                        v[8] != 0,
                        v[9] as u32,
                        v[10],
                        v[11],
                    ),
                    v[6],
                ),
                "batched" => (
                    hmx_solve_batched(v[0], v[1], v[2], v[3], v[4] as i32, v[6], v[7]),
                    v[4],
                ),
                other => panic!("unexpected solver {other}"),
            };
            let want = (w[0] == 1).then_some((w[1], w[2], w[3], w[4]));
            assert_eq!(got, want, "{line}");
            cases += 1;
            match want {
                None => no_fit += 1,
                Some((_, _, act_threads, _)) if act_threads < threads => halved += 1,
                Some(_) => {}
            }
        }
        assert!(cases > 26000, "golden file looks truncated: {cases} cases");
        // The grid has to keep reaching the paths a roomy budget never takes.
        assert!(no_fit > 1000, "only {no_fit} cases do not fit");
        assert!(
            halved > 100,
            "only {halved} cases halve the activation threads"
        );
    }

    /// The eight one-byte fields share the first two words, then seven `int32` sizes, then the
    /// seven `{mp, l}` dividers, as `struct htp_mm_kernel_params` lays them out.
    #[test]
    fn params_pack_into_the_dsp_struct_layout() {
        let fd = |d: u32| init_fastdiv(d);
        let kp = MmKernelParams {
            kernel_type: 2,
            pipeline: 1,
            unused: 0,
            n_hmx: 1,
            n_threads: 8,
            n_act_threads: 4,
            n_prefetch: 16,
            n_weights: 3,
            m_chunk: 32,
            n_chunk: 2304,
            tile_size: 1088,
            aligned_tile_size: 1152,
            act_row_size: 36864,
            vtcm_size: 7587840,
            vtcm_bias_size: 4096,
            div_ne12_ne1: fd(5),
            div_ne1: fd(6),
            div_r2: fd(7),
            div_r3: fd(9),
            div_ne12: fd(10),
            div_n_act_threads: fd(4),
            div_ne00_padded: fd(1024),
        };
        let w = kp.to_words();
        assert_eq!(
            w[0] as u32, 0x0100_0102,
            "kernel_type | pipeline << 8 | n_hmx << 24"
        );
        assert_eq!(
            w[1] as u32, 0x0310_0408,
            "n_threads | n_act_threads << 8 | n_prefetch << 16 | n_weights << 24"
        );
        assert_eq!(&w[2..9], &[32, 2304, 1088, 1152, 36864, 7587840, 4096]);
        assert_eq!((w[9], w[10]), (fd(5).mp as i32, fd(5).l as i32));
        assert_eq!((w[21], w[22]), (fd(1024).mp as i32, fd(1024).l as i32));
        assert_eq!(&w[23..], &[0; 9], "the rest of the blob stays zero");
        assert_eq!(MmKernelParams::from_words(&w), kp);
    }

    /// Hand-checked against `matmul_golden.txt` (Q8_0, K 1024, N 4608, the shipped session): 32
    /// rows take the unpipelined HMX kernel in two 2304-wide chunks, a single row the block-parallel
    /// HVX kernel with the deepest weight prefetch.
    #[test]
    fn builders_reproduce_the_golden_entries() {
        let q8 = HtpDataType::Q8_0;
        let b8 = 8 << 20;
        let hmx = build_hmx_mm_kernel_params(q8, 1024, 4608, 32, 8, b8).unwrap();
        let kp = MmKernelParams::from_words(&hmx);
        assert_eq!(
            (
                kp.kernel_type,
                kp.pipeline,
                kp.n_hmx,
                kp.n_threads,
                kp.n_act_threads
            ),
            (MM_KERNEL_HMX_2D, 0, 1, 8, 8)
        );
        assert_eq!(
            (kp.m_chunk, kp.n_chunk, kp.vtcm_size),
            (32, 2304, 7_587_840)
        );
        assert_eq!(
            (kp.tile_size, kp.aligned_tile_size, kp.act_row_size),
            (1088, 1152, 36864)
        );
        assert_eq!(kp.div_ne00_padded, init_fastdiv(1024));
        // 64 rows pipeline in 1152-wide chunks
        let kp64 = MmKernelParams::from_words(
            &build_hmx_mm_kernel_params(q8, 1024, 4608, 64, 8, b8).unwrap(),
        );
        assert_eq!((kp64.pipeline, kp64.m_chunk, kp64.n_chunk), (1, 64, 1152));

        let hvx = build_mul_mat_kernel_params(q8, 1024, 1, 1, 4608 * 4, 8, b8).unwrap();
        let kp = MmKernelParams::from_words(&hvx);
        assert_eq!(
            kp.kernel_type, MM_KERNEL_HVX_QUANT_BLOCK,
            "one row, eight threads"
        );
        assert_eq!(
            (kp.n_hmx, kp.n_prefetch, kp.m_chunk, kp.vtcm_size),
            (0, 16, 0, 4_773_888)
        );
        assert_eq!(kp.div_ne12_ne1, init_fastdiv(1));
        // enough rows to give every thread its own: the row-parallel kernel, prefetch 2
        let rows = MmKernelParams::from_words(
            &build_mul_mat_kernel_params(q8, 1024, 8, 1, 4608 * 4, 8, b8).unwrap(),
        );
        assert_eq!(
            (rows.kernel_type, rows.n_prefetch),
            (MM_KERNEL_HVX_QUANT_ROW, 2)
        );
    }

    /// Below cera's row floor the builder never takes the HMX (the HMX worker is racy there),
    /// though the host solver itself would from 5 rows.
    #[test]
    fn the_hmx_needs_cera_s_minimum_rows() {
        let q8 = HtpDataType::Q8_0;
        for rows in 1..HMX_MM_MIN_NROWS {
            assert!(!mm_is_hmx_eligible(q8, 1024, 1024, rows), "{rows} rows");
            assert!(build_hmx_mm_kernel_params(q8, 1024, 1024, rows, 8, 8 << 20).is_none());
        }
        assert!(mm_is_hmx_eligible(q8, 1024, 1024, HMX_MM_MIN_NROWS));
        assert!(build_hmx_mm_kernel_params(q8, 1024, 1024, HMX_MM_MIN_NROWS, 8, 8 << 20).is_some());
        // the unmodified solver, given upstream's floor, is happy with 5 ...
        let upstream = MmSession {
            hmx_min_rows: UPSTREAM_HMX_MIN_NROWS + 1,
            ..MmSession::new(8, 8 << 20, 1)
        };
        let w = MmTensor::tiled_weight(Q8_0, [1024, 1024, 1, 1]);
        let (x, d) = (
            MmTensor::f32_rows([1024, 5, 1, 1], 4096),
            MmTensor::f32_rows([1024, 5, 1, 1], 4096),
        );
        assert_eq!(solve_mul_mat(&upstream, &w, &x, &d).unwrap().n_hmx, 1);
        // ... and a session from `new` keeps cera's floor in the solver itself, not only in the builders
        let cera = MmSession::new(8, 8 << 20, 1);
        assert_eq!(solve_mul_mat(&cera, &w, &x, &d).unwrap().n_hmx, 0);
        let (x8, d8) = (
            MmTensor::f32_rows([1024, 8, 1, 1], 4096),
            MmTensor::f32_rows([1024, 8, 1, 1], 4096),
        );
        assert_eq!(solve_mul_mat(&cera, &w, &x8, &d8).unwrap().n_hmx, 1);
        // dims that are not whole tiles, and types the HMX has no path for, stay off it; a tiled
        // type that does have one (Q4_1) takes it once the rows are there
        assert!(!mm_is_hmx_eligible(q8, 1000, 1024, 64));
        assert!(!mm_is_hmx_eligible(q8, 1024, 1000, 64));
        assert!(!mm_is_hmx_eligible(q8, 1024, 0, 64));
        assert!(!mm_is_hmx_eligible(q8, 0, 1024, 64));
        assert!(mm_is_hmx_eligible(HtpDataType::F16, 1024, 1024, 64));
        assert!(mm_is_hmx_eligible(HtpDataType::Q4_1, 1024, 1024, 64));
        assert!(!mm_is_hmx_eligible(HtpDataType::I32, 1024, 1024, 64));
    }

    /// The HMX weight stride is one tiled row: the ggml row for Q4_0 / Q8_0, a wider one for the
    /// K-quants (uncompressed scales), 2 bytes per element for plain F16.
    #[test]
    fn hmx_weight_stride_is_one_tiled_row() {
        assert_eq!(mm_hmx_nb1(HtpDataType::Q8_0, 1024), 1024 / 32 * 34);
        assert_eq!(mm_hmx_nb1(HtpDataType::Q4_0, 1024), 1024 / 32 * 18);
        assert_eq!(mm_hmx_nb1(HtpDataType::Q4K, 1024), 1024 / 32 * 20);
        assert_eq!(mm_hmx_nb1(HtpDataType::Q6K, 1024), 1024 / 32 * 28);
        assert_eq!(mm_hmx_nb1(HtpDataType::F16, 1024), 2048);
    }

    /// The fused HVX kernel cannot chunk its activation rows, so its row limit is the largest
    /// count the solver still takes (with the HMX off), for every shape and budget.
    #[test]
    fn the_fused_nx_row_limit_is_the_largest_that_fits() {
        for wtype in [HtpDataType::Q8_0, HtpDataType::Q4_0, HtpDataType::Q4K] {
            for ne10 in [256usize, 768, 1024, 4096] {
                for threads in [1u32, 4, 8] {
                    for budget in [256 << 10, 1 << 20, 8 << 20] {
                        let max = mm_hvx_fused_nx_max_rows(wtype, ne10, threads, budget);
                        let fused = |rows: usize| {
                            build_mul_mat_nx_kernel_params(
                                wtype, ne10, ne10, 1024, 3, rows, threads, budget, false,
                            )
                        };
                        assert!(
                            max == 0 || fused(max).is_some(),
                            "{wtype:?} {ne10} {threads} {budget}: {max}"
                        );
                        assert!(
                            fused(max + 1).is_none(),
                            "{wtype:?} {ne10} {threads} {budget}: {max}"
                        );
                    }
                }
            }
        }
    }

    /// A fused matmul takes the HMX when its rows allow, else the fused HVX kernel; only the
    /// shapes the DSP's fused kernels cover come back.
    #[test]
    fn fused_builder_picks_hmx_or_fused_hvx() {
        let q8 = HtpDataType::Q8_0;
        let nx = |rows: usize, hmx: bool| {
            build_mul_mat_nx_kernel_params(q8, 1024, 1024, 1024, 3, rows, 8, 8 << 20, hmx)
                .map(|w| MmKernelParams::from_words(&w))
        };
        let fused = nx(64, true).unwrap();
        assert_eq!(
            (fused.kernel_type, fused.n_hmx, fused.n_weights),
            (MM_KERNEL_HMX_2D, 1, 3)
        );
        let hvx = nx(64, false).unwrap();
        assert_eq!(
            (hvx.kernel_type, hvx.n_hmx, hvx.n_weights),
            (MM_KERNEL_HVX_QUANT_ROW, 0, 3)
        );
        assert_eq!(nx(1, true).unwrap().n_hmx, 0, "below the row floor");
        // an activation width the HVX kernel cannot take
        assert!(
            build_mul_mat_nx_kernel_params(q8, 1024, 1000, 1024, 3, 1, 8, 8 << 20, false).is_none()
        );
    }

    /// The routed builder is HVX only, and refuses what the host refuses.
    #[test]
    fn routed_builder_is_hvx_only() {
        let u = |v: usize| v as u64;
        let (k, n, experts, used) = (1024usize, 512usize, 8usize, 2usize);
        let w = MmTensor::tiled_weight(Q8_0, [u(k), u(n), u(experts), 1]);
        let x = MmTensor::f32_rows([u(k), 1, 1, 1], u(k) * 4);
        let ids = MmTensor::contiguous(I32, [u(used), 1, 1, 1]);
        let d = MmTensor::f32_rows([u(n), u(used), 1, 1], u(n) * 4);
        let kp = MmKernelParams::from_words(
            &build_mul_mat_id_kernel_params(&w, &x, &ids, &d, 8, 8 << 20).unwrap(),
        );
        assert_eq!((kp.n_hmx, kp.kernel_type), (0, MM_KERNEL_HVX_QUANT_BLOCK));
        let wrong_ids = MmTensor::contiguous(HtpDataType::I64 as u32, [u(used), 1, 1, 1]);
        assert!(build_mul_mat_id_kernel_params(&w, &x, &wrong_ids, &d, 8, 8 << 20).is_none());
        assert!(build_mul_mat_id_kernel_params(&w, &x, &ids, &d, 8, 64 << 10).is_none());

        // with enough tokens to reach the HMX row floor it is still HVX only
        let x64 = MmTensor::f32_rows([u(k), 1, 64, 1], u(k) * 4);
        let ids64 = MmTensor::contiguous(I32, [u(used), 64, 1, 1]);
        let d64 = MmTensor::f32_rows([u(n), u(used), 64, 1], u(n) * 4);
        let kp64 = MmKernelParams::from_words(
            &build_mul_mat_id_kernel_params(&w, &x64, &ids64, &d64, 8, 8 << 20).unwrap(),
        );
        assert_eq!(kp64.n_hmx, 0);
    }

    /// The fusion pass's support check says no to a weight that is not tiled and to an HVX kernel
    /// the fused ops do not have; the golden files only ever see supported kernels.
    #[test]
    fn fused_kernel_support_checks_refuse_untiled_weights_and_other_kernels() {
        let u = |v: usize| v as u64;
        let (k, n, experts, used) = (1024usize, 512usize, 8usize, 2usize);
        let w = MmTensor::tiled_weight(Q8_0, [u(k), u(n), u(experts), 1]);
        let x = MmTensor::f32_rows([u(k), 1, 1, 1], u(k) * 4);
        let ids = MmTensor::contiguous(I32, [u(used), 1, 1, 1]);
        let d = MmTensor::f32_rows([u(n), u(used), 1, 1], u(n) * 4);
        let kp = MmKernelParams::from_words(
            &build_mul_mat_id_kernel_params(&w, &x, &ids, &d, 8, 8 << 20).unwrap(),
        );
        assert_eq!(kp.kernel_type, MM_KERNEL_HVX_QUANT_BLOCK);
        assert!(mmidnx_kernel_supported(&w, &kp));
        let untiled = MmTensor::contiguous(Q8_0, [u(k), u(n), u(experts), 1]);
        assert!(!mmidnx_kernel_supported(&untiled, &kp));
        let other = MmKernelParams {
            kernel_type: MM_KERNEL_HVX_F32_F32_VTCM,
            ..kp
        };
        assert!(!mmidnx_kernel_supported(&w, &other));
        // the plain fused kernel has the row kernel only, not the block one
        assert!(!mmnx_kernel_supported(&w, &kp));
        let row = MmKernelParams {
            kernel_type: MM_KERNEL_HVX_QUANT_ROW,
            ..kp
        };
        assert!(mmnx_kernel_supported(&w, &row));
        assert!(!mmnx_kernel_supported(&untiled, &row));
    }

    /// A session without threads divides by zero in the C, which never sees one; here it counts
    /// as one thread in every solver, so the result is the one-thread result.
    #[test]
    fn a_zero_thread_session_solves_as_one_thread() {
        let w = MmTensor::tiled_weight(Q8_0, [1024, 1024, 1, 1]);
        let (x, d) = (
            MmTensor::f32_rows([1024, 1, 1, 1], 4096),
            MmTensor::f32_rows([1024, 1, 1, 1], 4096),
        );
        let (none, one) = (MmSession::new(0, 8 << 20, 1), MmSession::new(1, 8 << 20, 1));
        let solved = solve_mul_mat(&none, &w, &x, &d);
        assert!(solved.is_some());
        assert_eq!(solved, solve_mul_mat(&one, &w, &x, &d));
        assert_eq!(solve_mmnx(&none, &w, &x, 3), solve_mmnx(&one, &w, &x, 3));
        assert_eq!(solved.unwrap().n_threads, 1);
    }

    /// The cera builders refuse an empty extent instead of solving a zero-sized matmul.
    #[test]
    fn the_builders_refuse_empty_extents() {
        let q8 = HtpDataType::Q8_0;
        assert!(build_mul_mat_kernel_params(q8, 1024, 1, 1, 4096, 8, 8 << 20).is_some());
        assert!(build_mul_mat_kernel_params(q8, 0, 1, 1, 4096, 8, 8 << 20).is_none());
        assert!(build_mul_mat_kernel_params(q8, 1024, 0, 1, 4096, 8, 8 << 20).is_none());
        assert!(build_mul_mat_kernel_params(q8, 1024, 1, 0, 4096, 8, 8 << 20).is_none());
        assert!(build_mul_mat_kernel_params(q8, 1024, 1, 1, 0, 8, 8 << 20).is_none());
        let nx = |ne00, ne10, n0, weights, rows| {
            build_mul_mat_nx_kernel_params(q8, ne00, ne10, n0, weights, rows, 8, 8 << 20, false)
        };
        assert!(nx(1024, 1024, 1024, 3, 1).is_some());
        assert!(nx(0, 1024, 1024, 3, 1).is_none());
        assert!(nx(1024, 0, 1024, 3, 1).is_none());
        assert!(nx(1024, 1024, 0, 3, 1).is_none());
        assert!(nx(1024, 1024, 1024, 0, 1).is_none());
        assert!(nx(1024, 1024, 1024, 3, 0).is_none());
        // the routed builder too: no tokens, or no experts used per token
        let routed = |tokens: usize, used: usize| {
            let u = |v: usize| v as u64;
            let w = MmTensor::tiled_weight(Q8_0, [u(1024), u(512), u(8), 1]);
            let x = MmTensor::f32_rows([u(1024), 1, u(tokens), 1], u(1024) * 4);
            let ids = MmTensor::contiguous(I32, [u(used), u(tokens), 1, 1]);
            let d = MmTensor::f32_rows([u(512), u(used), u(tokens), 1], u(512) * 4);
            build_mul_mat_id_kernel_params(&w, &x, &ids, &d, 8, 8 << 20)
        };
        assert!(routed(1, 2).is_some());
        assert!(routed(0, 2).is_none());
        assert!(routed(1, 0).is_none());
    }

    /// A per-head attention-score matmul worked out from the host solver: K as `[64, 126, 8]`
    /// (row stride 2048 B) against 8 batches of 126 query rows. 1008 activation rows of 256 B, 16
    /// prefetched 2048 B weight rows per thread, no output slice, all in one chunk.
    #[test]
    fn f32_matmul_params_follow_the_host_solver() {
        let shape = MulMatF32Shape {
            ne00: 64,
            ne01: 126,
            ne02: 8,
            ne03: 1,
            src0_nb0: 4,
            src0_nb1: 2048,
            ne11: 126,
            ne12: 8,
            ne13: 1,
            dst_nb1: 126 * 4,
        };
        let k = build_mul_mat_f32_kernel_params(shape, 8, 8 << 20).unwrap();
        let m = MmKernelParams::from_words(&k);
        assert_eq!(m.kernel_type, MM_KERNEL_HVX_F32_F32_VTCM);
        assert_eq!((m.n_threads, m.n_hmx, m.n_prefetch), (8, 0, 16));
        assert_eq!(m.m_chunk, 0, "everything fits: no chunking");
        assert_eq!(m.act_row_size, 256); // round_up(64 * 4, 128)
        // activations round_up(256 * 1008, 256) = 258048, then the weights' prefetch rows
        // round_up(16 * 2048, 256) * 8 threads = 262144 (no output slice for more than one row)
        assert_eq!(m.vtcm_size, 258048 + 262144);
        // ne12 * ne11, ne11, ne12 / ne02 = 1, ne13 / ne03 = 1, ne12
        let fd = init_fastdiv;
        assert_eq!(m.div_ne12_ne1, fd(8 * 126));
        assert_eq!(m.div_ne1, fd(126));
        assert_eq!(m.div_r2, fd(1));
        assert_eq!(m.div_r3, fd(1));
        assert_eq!(m.div_ne12, fd(8));
        // it never takes the HMX, whatever the rows
        let many = MulMatF32Shape { ne11: 512, ..shape };
        assert_eq!(
            MmKernelParams::from_words(&build_mul_mat_f32_kernel_params(many, 8, 8 << 20).unwrap())
                .n_hmx,
            0
        );
    }

    /// Too little VTCM for all the activation rows: the largest even chunk that fits, and refusal
    /// when not even one row does.
    #[test]
    fn f32_matmul_params_chunk_the_activation_rows_or_refuse() {
        let shape = MulMatF32Shape {
            ne00: 64,
            ne01: 126,
            ne02: 8,
            ne03: 1,
            src0_nb0: 4,
            src0_nb1: 2048,
            ne11: 126,
            ne12: 8,
            ne13: 1,
            dst_nb1: 504,
        };
        // 262144 for the weights leaves 100000 B: 390 rows of 256 B, made even.
        let k = build_mul_mat_f32_kernel_params(shape, 8, 262144 + 100_000).unwrap();
        assert_eq!(MmKernelParams::from_words(&k).m_chunk, 390);
        assert!(MmKernelParams::from_words(&k).vtcm_size as usize <= 262144 + 100_000);
        assert!(build_mul_mat_f32_kernel_params(shape, 8, 262144).is_none());
    }

    /// A zero extent, or weights whose batch count does not divide the activations', has no kernel.
    #[test]
    fn f32_matmul_params_refuse_shapes_with_a_zero_divisor() {
        let ok = MulMatF32Shape {
            ne00: 64,
            ne01: 126,
            ne02: 8,
            ne03: 1,
            src0_nb0: 4,
            src0_nb1: 2048,
            ne11: 126,
            ne12: 8,
            ne13: 1,
            dst_nb1: 504,
        };
        let params = |s: MulMatF32Shape| build_mul_mat_f32_kernel_params(s, 8, 8 << 20);
        assert!(params(ok).is_some());
        // Weights broadcast over a larger activation batch: fine.
        assert!(params(MulMatF32Shape { ne02: 4, ..ok }).is_some());
        assert!(
            params(MulMatF32Shape { ne02: 16, ..ok }).is_none(),
            "more weight batches"
        );
        assert!(
            params(MulMatF32Shape { ne02: 3, ..ok }).is_none(),
            "not a divisor"
        );
        assert!(params(MulMatF32Shape { ne11: 0, ..ok }).is_none());
        assert!(
            params(MulMatF32Shape {
                src0_nb0: 4096,
                ..ok
            })
            .is_none(),
            "a transposed weight view"
        );
        assert!(params(MulMatF32Shape { ne12: 0, ..ok }).is_none());
        assert!(params(MulMatF32Shape { ne02: 0, ..ok }).is_none());
        assert!(params(MulMatF32Shape { ne00: 0, ..ok }).is_none());
        assert!(params(MulMatF32Shape { ne01: 0, ..ok }).is_none());
        assert!(params(MulMatF32Shape { ne03: 0, ..ok }).is_none());
        assert!(params(MulMatF32Shape { ne13: 0, ..ok }).is_none());
    }

    /// A tiled weight's registered descriptor carries the tiled strides; the solver wants the
    /// contiguous ggml tensor behind it.
    #[test]
    fn a_tiled_descriptor_maps_to_its_ggml_tensor() {
        let t = HtpTensor {
            data: 0,
            size: 0,
            flags: crate::backend::hexagon::HTP_TENSOR_WEIGHT | HTP_TENSOR_TILED,
            dtype: Q8_0,
            bi: 0,
            ti: 0,
            ne: [1024, 4608, 1, 1],
            // HVX-style stride, 32 rows of K tiles
            nb: [34, 32 * 1088, 4608 / 32 * 32 * 1088, 4608 / 32 * 32 * 1088],
        };
        let m = MmTensor::from(&t);
        assert_eq!(m, MmTensor::tiled_weight(Q8_0, [1024, 4608, 1, 1]));
        assert!(m.is_contiguous());
        let plain = HtpTensor {
            flags: crate::backend::hexagon::HTP_TENSOR_WEIGHT,
            dtype: F16,
            nb: [2, 2048, 2048 * 4608, 2048 * 4608],
            ..t
        };
        let p = MmTensor::from(&plain);
        assert!(!p.tiled && p.is_contiguous());
        assert_eq!(p.nbytes(), 2 * 1024 * 4608);
    }
}

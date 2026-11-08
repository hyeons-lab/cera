//! Kernel parameters for the DSP's elementwise ops: binary (`Add`, `Mul`, `Sub`, `Div`, `AddId`),
//! the unary family (activations, `Scale`/`Clamp`, the norms, the fused `RmsNormMul`, and our
//! `Snake`/`Sin`/`Cos`/`Elu`/`HardSigmoid`/`HardSwish` extensions) and the GLU family.
//!
//! Ports of llama.cpp's host solvers (`htp_kparams_solve_binary`, `htp_kparams_solve_unary_impl`,
//! `htp_kparams_solve_unary_mul_glu`, `htp_collapse_dims`, `htp_opnode_collapse`) and of the
//! layout helpers in `htp/binary-ops.h`, `htp/unary-ops.h` and `htp/glu-ops.h`. The DSP kernels
//! size their VTCM from the same helpers and read the result of these solvers verbatim, so the
//! words here must equal the C++'s for the same tensors and session limits; the golden tests
//! (`testdata/{binary,unary,glu}_golden.txt`, produced by `scripts/hexagon-golden/gen.py` from the
//! unmodified C++) pin that word for word.
//!
//! The solvers see tensors the way ggml does: dense, up to four dimensions, byte strides, and
//! (here) one of the element types the kernels take, F32 or F16 (plus I32 for the `AddId` index
//! tensor). Arithmetic mirrors the C++ integer widths, including its 32-bit truncations, so an
//! out-of-range size gives the same answer rather than a panic.

use super::params::{CopyTensor, FastDivValues, init_fastdiv};
use super::types::{HtpDataType, HtpOpCode};

/// `enum htp_binary_kernel_type` (`htp/binary-ops.h`).
pub const HTP_BINARY_KERNEL_SAME_SHAPE: u8 = 0;
pub const HTP_BINARY_KERNEL_ROW_BCAST: u8 = 1;
/// `src1` is a table of scalars `[1, ne11, ne12, ne13]` staged in VTCM; row `i` of `src0` is
/// combined with the table entry for its position.
pub const HTP_BINARY_KERNEL_SCALAR: u8 = 2;
pub const HTP_BINARY_KERNEL_ADD_ID: u8 = 3;
pub const HTP_BINARY_KERNEL_COMPLEX: u8 = 4;
pub const HTP_BINARY_KERNEL_REPEAT: u8 = 5;
/// Contiguous `src0`/`src1`/`dst` of one shape, or a single-element `src1`: the kernel streams
/// flat chunks instead of rows.
pub const HTP_BINARY_KERNEL_CHUNKED: u8 = 6;

/// `htp_binary_kernel_params`, packed into the `kernel_params` words the way the DSP reads it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BinaryKernelParams {
    pub kernel_type: u8,
    pub split_at_ne01: u8,
    pub split_at_ne02: u8,
    pub src1_strided: u8,
    pub n_srcs: u8,
    pub is_scalar: u8,
    pub n_threads: u32,
    pub rows_per_buffer: u32,
    pub src0_row_size_aligned: u32,
    pub src1_row_size_aligned: u32,
    pub dst_row_size_aligned: u32,
    pub src1_size: u32,
    pub src2_size: u32,
    pub vtcm_size: u32,
    pub chunk_size: u32,
    pub chunk_bytes: u32,
    pub div_ne01: FastDivValues,
    pub div_ne02: FastDivValues,
    pub div_ne012: FastDivValues,
    pub div_ne11: FastDivValues,
    pub div_ne12: FastDivValues,
    pub div_ne13: FastDivValues,
}

impl BinaryKernelParams {
    /// The `kernel_params` words: four one-byte fields, two more with two bytes of padding, then
    /// nine `u32` and six `{mp, l}` dividers.
    pub fn to_words(&self) -> [i32; 32] {
        let mut w = [0i32; 32];
        let bytes = |b: [u8; 4]| u32::from_le_bytes(b) as i32;
        w[0] = bytes([
            self.kernel_type,
            self.split_at_ne01,
            self.split_at_ne02,
            self.src1_strided,
        ]);
        w[1] = bytes([self.n_srcs, self.is_scalar, 0, 0]);
        w[2] = self.n_threads as i32;
        w[3] = self.rows_per_buffer as i32;
        w[4] = self.src0_row_size_aligned as i32;
        w[5] = self.src1_row_size_aligned as i32;
        w[6] = self.dst_row_size_aligned as i32;
        w[7] = self.src1_size as i32;
        w[8] = self.src2_size as i32;
        w[9] = self.vtcm_size as i32;
        w[10] = self.chunk_size as i32;
        w[11] = self.chunk_bytes as i32;
        let divs = [
            self.div_ne01,
            self.div_ne02,
            self.div_ne012,
            self.div_ne11,
            self.div_ne12,
            self.div_ne13,
        ];
        for (i, d) in divs.iter().enumerate() {
            w[12 + 2 * i] = d.mp as i32;
            w[13 + 2 * i] = d.l as i32;
        }
        w
    }
}

/// `kernel_type` of a binary op's kernel params (the low byte of word 0).
pub fn binary_kernel_type(kernel_params: &[i32; 32]) -> u8 {
    kernel_params[0].to_le_bytes()[0]
}

/// A tensor as the host solvers read it. Wire descriptors carry `u32` extents and strides; ggml
/// holds `int64_t` extents and `size_t` strides, and the solvers multiply them, so the port
/// works in `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gg {
    dtype: u32,
    ne: [u64; 4],
    nb: [u64; 4],
}

impl From<&CopyTensor> for Gg {
    fn from(t: &CopyTensor) -> Self {
        Self {
            dtype: t.dtype,
            ne: t.ne.map(u64::from),
            nb: t.nb.map(u64::from),
        }
    }
}

impl Gg {
    fn to_wire(self) -> CopyTensor {
        CopyTensor {
            dtype: self.dtype,
            ne: self.ne.map(|v| v as u32),
            nb: self.nb.map(|v| v as u32),
        }
    }

    fn is(&self, t: HtpDataType) -> bool {
        self.dtype == t as u32
    }

    /// `ggml_type_size` for the element types an elementwise op can be handed; `None` for any
    /// other (the solvers refuse those on type before they size anything).
    fn type_size(&self) -> Option<u64> {
        use HtpDataType as T;
        match self.dtype {
            x if x == T::F32 as u32 || x == T::I32 as u32 => Some(4),
            x if x == T::F16 as u32 || x == T::Bf16 as u32 => Some(2),
            x if x == T::I64 as u32 => Some(8),
            _ => None,
        }
    }

    fn nelements(&self) -> u64 {
        self.ne.iter().fold(1u64, |a, &n| a.wrapping_mul(n))
    }

    fn is_empty(&self) -> bool {
        self.ne.contains(&0)
    }

    fn same_shape(&self, o: &Gg) -> bool {
        self.ne == o.ne
    }

    /// Planes and batches follow one another with no gap: `nb[2] == nb[1] * ne[1]` and
    /// `nb[3] == nb[2] * ne[2]`. A stride only counts where something sits above it: with a single
    /// plane the plane stride says nothing, unless there are batches, whose distance it sets.
    fn planes_packed(&self) -> bool {
        let above_rows = self.ne[2] > 1 || self.ne[3] > 1;
        (!above_rows || self.nb[2] == self.nb[1].wrapping_mul(self.ne[1]))
            && (self.ne[3] == 1 || self.nb[3] == self.nb[2].wrapping_mul(self.ne[2]))
    }

    /// `ggml_is_permuted`.
    fn is_permuted(&self) -> bool {
        self.nb[0] > self.nb[1] || self.nb[1] > self.nb[2] || self.nb[2] > self.nb[3]
    }

    /// `ggml_is_contiguous_rows`: elements within a row are adjacent.
    fn is_contiguous_rows(&self, ts: u64) -> bool {
        self.ne[0] == 1 || self.nb[0] == ts
    }

    /// `ggml_is_contiguous_m_n(t, m, 4)` for a type with a block size of one: dimensions above
    /// `m` must be packed, dimensions up to `m` may carry any stride.
    fn is_contiguous_from(&self, m: usize, ts: u64) -> bool {
        if self.ne[0] != 1 && self.nb[0] != ts {
            return false;
        }
        let mut next_nb = ts.wrapping_mul(self.ne[0]);
        for i in 1..4 {
            if i > m {
                if self.ne[i] != 1 && self.nb[i] != next_nb {
                    return false;
                }
                next_nb = next_nb.wrapping_mul(self.ne[i]);
            } else {
                next_nb = self.ne[i].wrapping_mul(self.nb[i]);
            }
        }
        true
    }

    fn is_contiguous(&self, ts: u64) -> bool {
        self.is_contiguous_from(0, ts)
    }

    fn is_contiguous_1(&self, ts: u64) -> bool {
        self.is_contiguous_from(1, ts)
    }
}

/// `ggml_can_repeat(t0, t1)`: `t1` is a whole number of repeats of `t0`.
fn can_repeat(t0: &Gg, t1: &Gg) -> bool {
    if t0.is_empty() {
        t1.is_empty()
    } else {
        (0..4).all(|i| t1.ne[i].is_multiple_of(t0.ne[i]))
    }
}

/// `hex_round_up(n, m)`: 32-bit, so a size past 4 GiB wraps like the C. Callers pass the `size_t`
/// the C++ passes, which the C++ narrows to `uint32_t` on the call.
fn round_up(n: u64, m: u32) -> u32 {
    let (n, m) = (n as u32, m);
    m.wrapping_mul(n.wrapping_add(m).wrapping_sub(1) / m)
}

fn align_down(v: u64, a: u64) -> u64 {
    v / a * a
}

fn is_binary_opcode(op: u32) -> bool {
    [
        HtpOpCode::Mul,
        HtpOpCode::Add,
        HtpOpCode::Sub,
        HtpOpCode::Div,
        HtpOpCode::AddId,
    ]
    .iter()
    .any(|&o| o as u32 == op)
}

/// `htp_binary_strides_safe`: rows are packed element runs and no row overlaps the next.
fn binary_strides_safe(s0: &Gg, s1: &Gg, dst: &Gg, es: u64) -> bool {
    for t in [s0, s1, dst] {
        if t.ne[0] != 1 && t.nb[0] != es {
            return false;
        }
    }
    for t in [s0, s1, dst] {
        if t.ne[1] > 1 && t.nb[1] < t.ne[0].wrapping_mul(es) {
            return false;
        }
    }
    true
}

/// What `htp_binary_vtcm_layout_build` yields that the solver reads back.
struct BinaryLayout {
    total_bytes: usize,
    rows_per_buffer: u32,
}

/// `htp_binary_vtcm_layout_build`.
fn binary_vtcm_layout(k: &BinaryKernelParams, vtcm_size: usize) -> BinaryLayout {
    let none = BinaryLayout {
        total_bytes: 0,
        rows_per_buffer: 0,
    };
    let nt = u64::from(k.n_threads);
    if nt == 0 {
        return none;
    }
    let n_srcs = u32::from(k.n_srcs).max(2);
    let vtcm = vtcm_size as u64;

    if k.kernel_type == HTP_BINARY_KERNEL_CHUNKED {
        let chunk_bytes = u64::from(k.chunk_bytes);
        if chunk_bytes == 0 {
            return none;
        }
        let scalar = k.is_scalar != 0;
        let src0_bpt = 2 * chunk_bytes;
        let src1_bpt = if scalar {
            0
        } else {
            u64::from(2 * (n_srcs - 1)) * chunk_bytes
        };
        let dst_bpt = 2 * chunk_bytes;
        let src0_total = nt * src0_bpt;
        let src1_total = if scalar { 128 } else { nt * src1_bpt };
        let dst_total = nt * dst_bpt;
        let off = src0_total + src1_total + dst_total;
        if off > vtcm {
            return BinaryLayout {
                total_bytes: 0,
                rows_per_buffer: 1,
            };
        }
        return BinaryLayout {
            total_bytes: off as usize,
            rows_per_buffer: 1,
        };
    }

    let same_shape = k.kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE;
    // 32-bit sums, as in the C (the fields are `uint32_t`)
    let spad_row_total = if same_shape {
        2u32.wrapping_mul(
            k.src0_row_size_aligned
                .wrapping_add((n_srcs - 1).wrapping_mul(k.src1_row_size_aligned))
                .wrapping_add(k.dst_row_size_aligned),
        )
    } else {
        2u32.wrapping_mul(k.src0_row_size_aligned.wrapping_add(k.dst_row_size_aligned))
    };
    let spad_row_total = u64::from(spad_row_total);
    let static_vtcm = u64::from(k.src1_size.wrapping_add(k.src2_size));
    if spad_row_total == 0 || vtcm < static_vtcm {
        return none;
    }
    let rows_per_buffer = if k.rows_per_buffer > 0 {
        u64::from(k.rows_per_buffer)
    } else {
        (vtcm - static_vtcm) / (nt * spad_row_total)
    };
    if rows_per_buffer == 0 {
        return none;
    }

    let src0_bpt = rows_per_buffer * 2 * u64::from(k.src0_row_size_aligned);
    let dst_bpt = rows_per_buffer * 2 * u64::from(k.dst_row_size_aligned);
    let src1_bpt = if same_shape {
        rows_per_buffer * 2 * u64::from(n_srcs - 1) * u64::from(k.src1_row_size_aligned)
    } else {
        0
    };
    let src0_total = nt * src0_bpt;
    let src1_total = if k.src1_size > 0 {
        u64::from(k.src1_size)
    } else {
        nt * src1_bpt
    };
    let src2_total = u64::from(k.src2_size);
    let dst_total = nt * dst_bpt;
    BinaryLayout {
        total_bytes: (src0_total + src1_total + src2_total + dst_total) as usize,
        rows_per_buffer: rows_per_buffer as u32,
    }
}

/// Port of `htp_kparams_solve_binary`: kernel parameters for `Add`/`Mul`/`Sub`/`Div` (`src2`
/// `None`) or `AddId` (`src2` the I32 expert-index tensor), with `n_threads` the session's DSP
/// worker count and `vtcm_size` its VTCM budget. `None` where the C++ answers unsupported: types
/// other than F32/F16 (F32 for `AddId`), permuted or overlapping-row layouts, a `src1` that does
/// not tile `src0`, or a working set that does not fit.
///
/// The op only matters through `AddId`; which of the four arithmetic ops runs is the opcode.
pub fn solve_binary(
    opcode: u32,
    src0: &CopyTensor,
    src1: &CopyTensor,
    src2: Option<&CopyTensor>,
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_size: usize,
) -> Option<[i32; 32]> {
    if !is_binary_opcode(opcode) {
        return None;
    }
    let (s0, s1, d) = (Gg::from(src0), Gg::from(src1), Gg::from(dst));
    let is_add_id = opcode == HtpOpCode::AddId as u32;
    let s2 = src2.map(Gg::from);
    if is_add_id {
        let s2 = s2.as_ref()?;
        if !s0.is(HtpDataType::F32)
            || !s1.is(HtpDataType::F32)
            || !d.is(HtpDataType::F32)
            || !s2.is(HtpDataType::I32)
        {
            return None;
        }
        if !s0.same_shape(&d) {
            return None;
        }
        if s1.ne[0] != s0.ne[0] || s1.ne[2] != 1 || s1.ne[3] != 1 {
            return None;
        }
        if s2.ne[0] != s0.ne[1] || s2.ne[1] != s0.ne[2] {
            return None;
        }
        if s0.nb[0] != 4 || s1.nb[0] != 4 || d.nb[0] != 4 || s2.nb[0] != 4 {
            return None;
        }
        if !s0.is_contiguous(4) || !s1.is_contiguous(4) || !d.is_contiguous(4) {
            return None;
        }
        if s2.nb[1] < s2.ne[0] * 4 {
            return None;
        }
    } else {
        if (!s0.is(HtpDataType::F32) && !s0.is(HtpDataType::F16))
            || s1.dtype != s0.dtype
            || d.dtype != s0.dtype
        {
            return None;
        }
        if s0.is_permuted() || d.is_permuted() || !s0.same_shape(&d) {
            return None;
        }
        if !can_repeat(&s1, &s0) || s1.is_permuted() {
            return None;
        }
    }

    // a session never reports zero workers; the C++ divides by the count
    let max_threads_sess = n_threads.max(1);
    let es = s0.type_size()?;
    let src0_row_size = s0.ne[0].wrapping_mul(es);
    let src1_row_size = s1.ne[0].wrapping_mul(es);
    let dst_row_size = d.ne[0].wrapping_mul(es);
    let src0_ra = round_up(src0_row_size, 128);
    let src1_ra = round_up(src1_row_size, 128);
    let dst_ra = round_up(dst_row_size, 128);

    if !binary_strides_safe(&s0, &s1, &d, es) {
        return None;
    }

    let is_scalar = !is_add_id && s1.ne[0] == 1;
    let is_row_bcast = !is_add_id
        && !is_scalar
        && s1.ne[0] == s0.ne[0]
        && (s0.ne[1] > 1 || s0.ne[2] > 1 || s0.ne[3] > 1)
        && s1.ne[1] == 1
        && s1.ne[2] == 1
        && s1.ne[3] == 1;
    let is_same_shape = !is_add_id
        && !is_scalar
        && s1.ne[0] == s0.ne[0]
        && s1.ne[1] == s0.ne[1]
        && (s1.ne[2] == s0.ne[2] || s1.ne[2] == 1)
        && (s1.ne[3] == s0.ne[3] || s1.ne[3] == 1);
    let is_complex =
        !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && s1.ne[0] == s0.ne[0];
    let is_contig = s0.is_contiguous(es) && s1.is_contiguous(es) && d.is_contiguous(es);
    let is_scalar_broadcast = !is_add_id && s1.nelements() == 1;

    let mut k = BinaryKernelParams::default();

    if !is_add_id && is_contig && (s0.same_shape(&s1) || is_scalar_broadcast) {
        let total_elems = s0.nelements() as u32;
        let max_chunk_elems = (32768 / es) as u32;
        let min_chunk_elems = 256u32;
        let two_t = max_threads_sess.wrapping_mul(2);
        let target = round_up(
            u64::from(total_elems.wrapping_add(two_t).wrapping_sub(1) / two_t),
            32,
        );
        let chunk_size = max_chunk_elems.min(target.max(min_chunk_elems));
        let chunk_bytes = round_up(u64::from(chunk_size) * es, 128);
        let total_chunks = total_elems.wrapping_add(chunk_size).wrapping_sub(1) / chunk_size;
        let threads = if total_chunks >= 2 {
            max_threads_sess.min(total_chunks)
        } else {
            1
        };

        k.kernel_type = HTP_BINARY_KERNEL_CHUNKED;
        k.n_threads = threads;
        k.n_srcs = 2;
        k.rows_per_buffer = 1;
        k.src0_row_size_aligned = src0_ra;
        k.src1_row_size_aligned = if is_scalar_broadcast { 0 } else { src1_ra };
        k.dst_row_size_aligned = dst_ra;
        k.src1_size = 0;
        k.chunk_size = chunk_size;
        k.chunk_bytes = chunk_bytes;
        k.is_scalar = u8::from(is_scalar_broadcast);

        let l = binary_vtcm_layout(&k, vtcm_size);
        if l.total_bytes == 0 || l.total_bytes > vtcm_size {
            return None;
        }
        k.vtcm_size = l.total_bytes as u32;
        return Some(k.to_words());
    }

    let vtcm = vtcm_size as u64;
    let kernel_type;
    let mut src1_size = 0u32;
    let mut src2_size = 0u32;
    if is_add_id {
        let s2 = s2.as_ref()?;
        kernel_type = HTP_BINARY_KERNEL_ADD_ID;
        src1_size = round_up(s1.ne[1].wrapping_mul(u64::from(src1_ra)), 128);
        src2_size = round_up(s2.nelements().wrapping_mul(4), 128);
        if u64::from(src1_size) + u64::from(src2_size) > vtcm {
            return None;
        }
    } else if is_row_bcast {
        kernel_type = HTP_BINARY_KERNEL_ROW_BCAST;
        src1_size = src1_ra;
    } else if is_scalar {
        src1_size = round_up(s1.nelements().wrapping_mul(es), 128);
        if u64::from(src1_size) > vtcm {
            return None;
        }
        kernel_type = HTP_BINARY_KERNEL_SCALAR;
        k.src1_strided = u8::from(!s1.is_contiguous(es));
    } else if is_same_shape {
        kernel_type = HTP_BINARY_KERNEL_SAME_SHAPE;
    } else if is_complex {
        src1_size = round_up(s1.nelements().wrapping_mul(es).wrapping_add(128), 128);
        if u64::from(src1_size) > vtcm {
            return None;
        }
        kernel_type = HTP_BINARY_KERNEL_COMPLEX;
        k.src1_strided = u8::from(!s1.is_contiguous(es));
    } else {
        src1_size = round_up(s1.nelements().wrapping_mul(es).wrapping_add(128), 128);
        if u64::from(src1_size) > vtcm {
            return None;
        }
        kernel_type = HTP_BINARY_KERNEL_REPEAT;
        k.src1_strided = u8::from(!s1.is_contiguous(es));
    }

    let src0_nrows = s0.ne[1].wrapping_mul(s0.ne[2]).wrapping_mul(s0.ne[3]) as u32;
    // the C++ divides by the row size; an empty row has no rows worth running
    let min_rows_per_thread = (128 + dst_row_size - 1).checked_div(dst_row_size)?.max(1) as u32;
    let max_threads = (src0_nrows / min_rows_per_thread).max(1);

    k.kernel_type = kernel_type;
    k.n_threads = max_threads_sess.min(max_threads);
    k.n_srcs = 2;
    k.src0_row_size_aligned = src0_ra;
    k.src1_row_size_aligned = src1_ra;
    k.dst_row_size_aligned = dst_ra;
    k.src1_size = src1_size;
    k.src2_size = src2_size;

    let src0_contig_dim1 = s0.nb[2] == s0.ne[1].wrapping_mul(s0.nb[1]);
    let src1_contig_dim1 = s1.nb[2] == s1.ne[1].wrapping_mul(s1.nb[1]);
    let dst_contig_dim1 = d.nb[2] == s0.ne[1].wrapping_mul(d.nb[1]);
    let src0_contig_dim2 = s0.nb[3] == s0.ne[2].wrapping_mul(s0.nb[2]);
    let src1_contig_dim2 = s1.nb[3] == s1.ne[2].wrapping_mul(s1.nb[2]);
    let dst_contig_dim2 = d.nb[3] == s0.ne[2].wrapping_mul(d.nb[2]);

    let mut split_at_ne01 = false;
    if s0.ne[2] > 1 {
        split_at_ne01 = match kernel_type {
            HTP_BINARY_KERNEL_SAME_SHAPE => {
                s1.ne[2] != s0.ne[2] || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1
            }
            HTP_BINARY_KERNEL_ROW_BCAST => !src0_contig_dim1 || !dst_contig_dim1,
            _ => true,
        };
    }
    k.split_at_ne01 = u8::from(split_at_ne01);

    let mut split_at_ne02 = false;
    if s0.ne[3] > 1 {
        split_at_ne02 = match kernel_type {
            HTP_BINARY_KERNEL_SAME_SHAPE => {
                s1.ne[3] != s0.ne[3] || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2
            }
            HTP_BINARY_KERNEL_ROW_BCAST => !src0_contig_dim2 || !dst_contig_dim2,
            _ => true,
        };
    }
    k.split_at_ne02 = u8::from(split_at_ne02);

    k.div_ne01 = init_fastdiv(s0.ne[1] as u32);
    k.div_ne02 = init_fastdiv(s0.ne[2] as u32);
    k.div_ne012 = init_fastdiv(s0.ne[1].wrapping_mul(s0.ne[2]) as u32);
    k.div_ne11 = init_fastdiv(s1.ne[1] as u32);
    k.div_ne12 = init_fastdiv(s1.ne[2] as u32);
    k.div_ne13 = init_fastdiv(s1.ne[3] as u32);

    let l = binary_vtcm_layout(&k, vtcm_size);
    if l.rows_per_buffer == 0 || l.total_bytes > vtcm_size {
        // the C++ then retries the flat chunked kernel for a contiguous same-shape op, but such an
        // op was answered above, so the retry is unreachable
        return None;
    }
    k.rows_per_buffer = l.rows_per_buffer;
    k.vtcm_size = l.total_bytes as u32;
    Some(k.to_words())
}

/// `htp_collapse_dims`: merge contiguous dimensions of same-shaped elementwise tensors so rows
/// fill at least one HVX vector, and split a long single row so every thread has work. `elem` is
/// the element size; `n_threads` the session's.
fn collapse_dims(t: &mut [Gg], elem: u64, n_threads: u32) {
    if t.is_empty() {
        return;
    }
    for i in 0..t.len() {
        if t[i].nb[0] != elem || (i > 0 && !t[0].same_shape(&t[i])) {
            return;
        }
    }

    // The steps below rewrite the plane and batch strides from the row stride and the new extents,
    // which describes the same bytes only when the planes and batches follow one another with no
    // gap. Upstream does not check, so a view with a gap (a slice of a larger buffer, a padded
    // batch) comes out addressing other planes than it went in with. Such a tensor is left as it
    // is: the DSP reads strides from the descriptor, and the uncollapsed ones are the true ones.
    if t.iter().any(|x| !x.planes_packed()) {
        return;
    }

    while t[0].ne[1] > 1 || t[0].ne[2] > 1 || t[0].ne[3] > 1 {
        let row_size = t[0].ne[0].wrapping_mul(elem);
        let rows_contig = t.iter().all(|x| x.nb[1] == x.ne[0].wrapping_mul(elem));
        let all_ne1_one = t.iter().all(|x| x.ne[1] == 1);
        let can_merge = all_ne1_one
            || (rows_contig
                && (t[0].ne[0] == 1
                    || (row_size < 128 && row_size.wrapping_mul(t[0].ne[1]) <= 2048)));
        if !can_merge {
            break;
        }
        for x in t.iter_mut() {
            x.ne[0] = x.ne[0].wrapping_mul(x.ne[1]);
            x.ne[1] = x.ne[2];
            x.ne[2] = x.ne[3];
            x.ne[3] = 1;
            x.nb[1] = x.nb[2];
            x.nb[2] = x.nb[3];
            x.nb[3] = x.nb[2].wrapping_mul(x.ne[2]);
        }
    }

    // ne1 too big to merge in full: merge it in chunks of k rows (k divides ne1)
    let row_size = t[0].ne[0].wrapping_mul(elem);
    let rows_contig = t.iter().all(|x| x.nb[1] == row_size);
    if t[0].ne[1] > 1 && row_size < 128 && rows_contig && row_size > 0 {
        let mut k = 2048 / row_size;
        while k > 1 && !t[0].ne[1].is_multiple_of(k) {
            k -= 1;
        }
        if k > 1 {
            for x in t.iter_mut() {
                x.ne[0] = x.ne[0].wrapping_mul(k);
                x.ne[1] /= k;
                x.nb[1] = x.nb[1].wrapping_mul(k);
                x.nb[2] = x.nb[1].wrapping_mul(x.ne[1]);
                x.nb[3] = x.nb[2].wrapping_mul(x.ne[2]);
            }
        }
    }

    // nrows too small to saturate threads: split ne0 into k rows (k divides ne0)
    let row_size_curr = t[0].ne[0].wrapping_mul(elem);
    let nrows = t[0].ne[1].wrapping_mul(t[0].ne[2]).wrapping_mul(t[0].ne[3]) as u32;
    let rows_contig = t.iter().all(|x| x.ne[1] == 1 || x.nb[1] == row_size_curr);
    if nrows > 0 && nrows < n_threads && rows_contig && row_size_curr >= 4096 {
        let vec_elems = 128 / elem;
        let want = (n_threads
            .wrapping_mul(4)
            .wrapping_add(nrows)
            .wrapping_sub(1)
            / nrows) as u64;
        let mut k = (row_size_curr / 16384).max(want);
        k = k.min(row_size_curr / 2048);
        while k > 1 && !t[0].ne[0].is_multiple_of(k * vec_elems) {
            k -= 1;
        }
        if k > 1 {
            for x in t.iter_mut() {
                x.ne[0] /= k;
                x.ne[1] = x.ne[1].wrapping_mul(k);
                x.nb[1] = x.ne[0].wrapping_mul(elem);
                x.nb[2] = x.nb[1].wrapping_mul(x.ne[1]);
                x.nb[3] = x.nb[2].wrapping_mul(x.ne[2]);
            }
        }
    }
}

/// `htp_op_is_unary` minus `Hadamard`, which needs the matmul node it hangs off and is not
/// something the wire format can express.
fn is_unary_opcode(op: u32) -> bool {
    use HtpOpCode as O;
    [
        O::Clamp,
        O::LeakyRelu,
        O::Norm,
        O::RmsNorm,
        O::RmsNormMul,
        O::RmsNormMulAdd,
        O::Scale,
        O::Sqr,
        O::Sqrt,
        O::UnaryNeg,
        O::UnaryExp,
        O::UnarySigmoid,
        O::UnarySilu,
        O::UnaryGelu,
        O::UnaryGeluQuick,
        O::UnaryGeluErf,
        O::UnarySoftplus,
        O::UnaryTanh,
        O::UnaryAbs,
        O::UnaryLog,
        O::UnaryRelu,
        O::UnaryStep,
        O::UnaryCeil,
        O::UnarySgn,
        O::L2Norm,
        O::Tri,
        O::UnarySnake,
        O::UnarySin,
        O::UnaryCos,
        O::UnaryHardSigmoid,
        O::UnaryHardSwish,
        O::UnaryElu,
    ]
    .iter()
    .any(|&o| o as u32 == op)
}

/// `htp_unary_is_elementwise`: ops that act on each element alone (the norms, `L2Norm` and `Tri`
/// look across a row and keep their shape).
fn is_unary_elementwise(op: u32) -> bool {
    use HtpOpCode as O;
    is_unary_opcode(op)
        && ![
            O::Norm,
            O::RmsNorm,
            O::RmsNormMul,
            O::RmsNormMulAdd,
            O::L2Norm,
            O::Tri,
        ]
        .iter()
        .any(|&o| o as u32 == op)
}

fn is_glu_opcode(op: u32) -> bool {
    use HtpOpCode as O;
    [
        O::GluSwiglu,
        O::GluSwigluOai,
        O::GluSwigluClamp,
        O::GluGeglu,
        O::GluGegluQuick,
        O::GluGegluErf,
        O::GluReglu,
        O::GluSigmoid,
    ]
    .iter()
    .any(|&o| o as u32 == op)
}

/// Whether `opcode` takes the kernel parameters this module derives.
pub(crate) fn is_eltwise_opcode(opcode: u32) -> bool {
    is_binary_opcode(opcode) || is_unary_opcode(opcode) || is_glu_opcode(opcode)
}

/// What the host does to an elementwise unary node before solving it (`htp_opnode_collapse`):
/// returns `(src0, dst)` as the real host registers them with the DSP. The tensors are returned
/// unchanged when the op is not elementwise, the shapes differ, or nothing merges or splits.
///
/// The DSP reads extents and strides from the tensor descriptors it is handed and the kernel
/// parameters must be solved for exactly those, so a caller that wants the collapsed shape has to
/// register the collapsed tensors, not only call this.
pub fn collapse_unary(
    opcode: u32,
    src0: &CopyTensor,
    dst: &CopyTensor,
    n_threads: u32,
) -> (CopyTensor, CopyTensor) {
    let (s, d) = (Gg::from(src0), Gg::from(dst));
    if !is_unary_elementwise(opcode) || !s.same_shape(&d) {
        return (*src0, *dst);
    }
    let Some(es) = s.type_size() else {
        return (*src0, *dst);
    };
    let mut t = [s, d];
    collapse_dims(&mut t, es, n_threads);
    if t[0].ne[0] != s.ne[0] || t[0].ne[1] != s.ne[1] {
        (t[0].to_wire(), t[1].to_wire())
    } else {
        (*src0, *dst)
    }
}

/// [`collapse_unary`] for a two-source GLU node `(src0, src1, dst)`; a single-source node is
/// never collapsed.
pub fn collapse_glu(
    opcode: u32,
    src0: &CopyTensor,
    src1: Option<&CopyTensor>,
    dst: &CopyTensor,
    n_threads: u32,
) -> (CopyTensor, Option<CopyTensor>, CopyTensor) {
    let unchanged = (*src0, src1.copied(), *dst);
    let Some(src1) = src1 else {
        return unchanged;
    };
    let (s0, s1, d) = (Gg::from(src0), Gg::from(src1), Gg::from(dst));
    if !is_glu_opcode(opcode) || !s0.same_shape(&d) || !s1.same_shape(&d) {
        return unchanged;
    }
    let Some(es) = s0.type_size() else {
        return unchanged;
    };
    let mut t = [s0, s1, d];
    collapse_dims(&mut t, es, n_threads);
    if t[0].ne[0] != s0.ne[0] || t[0].ne[1] != s0.ne[1] {
        (t[0].to_wire(), Some(t[1].to_wire()), t[2].to_wire())
    } else {
        unchanged
    }
}

/// `htp_unary_kernel_params`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnaryKernelParams {
    pub n_threads: u32,
    /// Columns per tile when a row does not fit VTCM and the kernel walks it in tiles (F32 only),
    /// else 0.
    pub col_tile: u32,
    /// Rows a thread stages per VTCM buffer (0 in tiled mode).
    pub block: u32,
    pub broadcast_weight: u32,
    pub src0_contig: u32,
    pub src1_contig: u32,
    pub vtcm_size: u32,
    pub tiles_per_row: u32,
    pub div_ne01: FastDivValues,
    pub div_ne02: FastDivValues,
    pub div_ne012: FastDivValues,
    pub div_tpr: FastDivValues,
}

impl UnaryKernelParams {
    pub fn to_words(&self) -> [i32; 32] {
        let mut w = [0i32; 32];
        w[0] = self.n_threads as i32;
        w[1] = self.col_tile as i32;
        w[2] = self.block as i32;
        w[3] = self.broadcast_weight as i32;
        w[4] = self.src0_contig as i32;
        w[5] = self.src1_contig as i32;
        w[6] = self.vtcm_size as i32;
        w[7] = self.tiles_per_row as i32;
        let divs = [self.div_ne01, self.div_ne02, self.div_ne012, self.div_tpr];
        for (i, d) in divs.iter().enumerate() {
            w[8 + 2 * i] = d.mp as i32;
            w[9 + 2 * i] = d.l as i32;
        }
        w
    }
}

/// `htp_unary_vtcm_layout_build`: bytes of VTCM an op occupies for `block` staged rows per thread
/// (or a column tile), double buffered. The kernel rebuilds this layout from the same inputs.
#[allow(clippy::too_many_arguments)]
fn unary_vtcm_total(
    op: u32,
    src0_data_row: u64,
    dst_data_row: u64,
    src1_data_row: u64,
    elem: u64,
    col_tile: u32,
    block: u32,
    broadcast_weight: bool,
    n_threads: u32,
) -> u64 {
    let has_weight = op == HtpOpCode::RmsNormMul as u32 || op == HtpOpCode::RmsNormMulAdd as u32;
    let src0_ra = u64::from(round_up(src0_data_row, 128));
    let dst_ra = u64::from(round_up(dst_data_row, 128));
    let src1_ra = if has_weight {
        u64::from(round_up(src1_data_row, 128))
    } else {
        0
    };
    let (src0_bytes, src1_bytes, src2_bytes, dst_bytes);
    if col_tile > 0 {
        let tile_bytes = u64::from(col_tile) * elem;
        src0_bytes = tile_bytes * 2;
        dst_bytes = tile_bytes * 2;
        src1_bytes = 0;
        src2_bytes = 0;
    } else {
        let block = u64::from(block);
        src0_bytes = src0_ra * block * 2;
        src2_bytes = if op == HtpOpCode::RmsNormMulAdd as u32 {
            src0_bytes
        } else {
            0
        };
        dst_bytes = dst_ra * block * 2;
        src1_bytes = if !has_weight {
            0
        } else if broadcast_weight {
            src1_ra
        } else {
            src1_ra * block * 2
        };
    }
    let nt = u64::from(n_threads);
    src0_bytes * nt + src1_bytes * nt + src2_bytes * nt + dst_bytes * nt
}

/// `htp_kparams_solve_unary_impl` for one op over `(src0, src1, dst)`, `src1` being the weight of
/// the fused norms. `None` where the C++ answers unsupported: another element type than F32
/// (F16 for a short list of ops), a destination that is not contiguous, a `src0` with strided
/// elements, shapes that differ, an `RmsNormMulAdd` weight that is not one broadcast row, or no
/// room in the VTCM budget for even one row (F32 falls back to column tiles first).
///
/// The tensors are solved as given: the real host solves the collapsed tensors
/// ([`collapse_unary`]) and registers those. `Hadamard` has no wire form and is refused.
pub fn solve_unary(
    opcode: u32,
    src0: &CopyTensor,
    src1: Option<&CopyTensor>,
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_size: usize,
) -> Option<[i32; 32]> {
    use HtpOpCode as O;
    if !is_unary_opcode(opcode) {
        return None;
    }
    let (s0, d) = (Gg::from(src0), Gg::from(dst));
    if (!s0.is(HtpDataType::F32) && !s0.is(HtpDataType::F16)) || d.dtype != s0.dtype {
        return None;
    }
    if opcode == O::Tri as u32 && (!s0.is(HtpDataType::F32) || !s0.is_contiguous(4)) {
        return None;
    }
    if s0.is(HtpDataType::F16) {
        let f16_ok = [
            O::Norm,
            O::RmsNorm,
            O::L2Norm,
            O::Scale,
            O::Clamp,
            O::Sqr,
            O::Sqrt,
            O::UnaryLog,
            O::UnaryAbs,
            O::UnaryStep,
            O::UnaryCeil,
            O::UnarySgn,
        ]
        .iter()
        .any(|&o| o as u32 == opcode);
        if !f16_ok {
            return None;
        }
    }
    let es = s0.type_size()?;
    if !s0.is_contiguous_rows(es) || !d.is_contiguous(es) || !s0.same_shape(&d) {
        return None;
    }

    let dst_data_row = d.ne[0].wrapping_mul(es);
    let src0_nrows = s0.ne[1].wrapping_mul(s0.ne[2]).wrapping_mul(s0.ne[3]) as u32;
    // the C++ divides by the row size; an empty row has no rows worth running
    let min_rows_per_thread = (128 + dst_data_row - 1).checked_div(dst_data_row)?.max(1) as u32;
    let max_threads = (src0_nrows / min_rows_per_thread).max(1);
    // a session never reports zero workers; the C++ divides by the count
    let n_threads = n_threads.max(1).min(max_threads);
    let vtcm = vtcm_size as u64;

    let mut k = UnaryKernelParams {
        n_threads,
        ..Default::default()
    };

    let has_weight = opcode == O::RmsNormMul as u32 || opcode == O::RmsNormMulAdd as u32;
    let s1 = match (has_weight, src1) {
        (true, Some(s)) => Some(Gg::from(s)),
        (true, None) => return None,
        (false, _) => None,
    };
    let broadcast_weight = s1
        .as_ref()
        .is_some_and(|s| s.ne[1].wrapping_mul(s.ne[2]).wrapping_mul(s.ne[3]) == 1);
    if opcode == O::RmsNormMulAdd as u32 && !broadcast_weight {
        return None;
    }
    k.broadcast_weight = u32::from(broadcast_weight);

    let src0_contig = s0.is_contiguous_1(es);
    let src1_contig = match &s1 {
        Some(s) if !broadcast_weight => s.is_contiguous_1(s.type_size()?),
        _ => true,
    };
    let nrows_per_thread = src0_nrows.wrapping_add(n_threads).wrapping_sub(1) / n_threads;
    let min_block_rows = (2048 / dst_data_row).max(1) as u32;
    let mut max_block = if src0_contig && src1_contig {
        src0_nrows
    } else {
        s0.ne[1] as u32
    };
    if nrows_per_thread >= min_block_rows.wrapping_mul(2) {
        max_block = max_block.min(min_block_rows.max(nrows_per_thread / 4));
    }
    k.src0_contig = u32::from(src0_contig);
    k.src1_contig = u32::from(src1_contig);

    // htp_unary_solve_layout
    let ne00 = s0.ne[0] as u32;
    let ne10 = d.ne[0] as u32;
    let ne11 = s1.as_ref().map_or(0, |s| s.ne[0] as u32);
    let src0_data_row = u64::from(ne00) * es;
    let dst_data_row_l = u64::from(ne10) * es;
    let src1_data_row = if has_weight { u64::from(ne11) * 4 } else { 0 };
    let src0_ra = u64::from(round_up(src0_data_row, 128));
    let dst_ra = u64::from(round_up(dst_data_row_l, 128));
    let src1_ra = if has_weight {
        u64::from(round_up(src1_data_row, 128))
    } else {
        0
    };
    let nt = u64::from(n_threads);

    let mut rpt: u64 = if opcode == O::RmsNormMulAdd as u32 {
        let src1_total = nt * src1_ra;
        let avail = vtcm.saturating_sub(src1_total);
        let per_row = 2 * (2 * src0_ra + dst_ra);
        avail.checked_div(nt * per_row)?
    } else if opcode == O::RmsNormMul as u32 {
        if broadcast_weight {
            let src1_total = nt * src1_ra;
            let avail = vtcm.saturating_sub(src1_total);
            let per_row = 2 * (src0_ra + dst_ra);
            avail.checked_div(nt * per_row)?
        } else {
            let per_row = 2 * (src0_ra + dst_ra + src1_ra);
            vtcm.checked_div(nt * per_row)?
        }
    } else {
        let per_row = 2 * (src0_ra + dst_ra);
        // a row of 2^32 bytes or more wraps its aligned size to 0, which the C divides by
        vtcm.checked_div(nt * per_row)?
    };
    if max_block > 0 && rpt > u64::from(max_block) {
        rpt = u64::from(max_block);
    }

    let is_reduction = [
        O::Norm,
        O::RmsNorm,
        O::RmsNormMul,
        O::RmsNormMulAdd,
        O::L2Norm,
    ]
    .iter()
    .any(|&o| o as u32 == opcode);
    // only F32 has tiled kernels; F16 stays on the row-block path even when no row fits
    let is_f16 = es == 2;
    let mut col_tile = 0u32;
    if rpt == 0 && !is_reduction && !is_f16 {
        let per_thread_budget = vtcm / nt;
        let tile_bytes = align_down(per_thread_budget / 4, 128);
        col_tile = (tile_bytes / es) as u32;
    }
    let block = if col_tile > 0 { 0 } else { rpt as u32 };
    let total = unary_vtcm_total(
        opcode,
        src0_data_row,
        dst_data_row_l,
        src1_data_row,
        es,
        col_tile,
        block,
        broadcast_weight,
        n_threads,
    );

    k.col_tile = col_tile;
    k.vtcm_size = total as u32;
    k.block = block;
    let tiles_per_row = if col_tile > 0 {
        s0.ne[0].div_ceil(u64::from(col_tile)) as u32
    } else {
        1
    };
    k.tiles_per_row = tiles_per_row;
    k.div_ne01 = init_fastdiv(s0.ne[1] as u32);
    k.div_ne02 = init_fastdiv(s0.ne[2] as u32);
    k.div_ne012 = init_fastdiv(s0.ne[1].wrapping_mul(s0.ne[2]) as u32);
    k.div_tpr = init_fastdiv(tiles_per_row);

    let rpt32 = rpt as u32;
    ((col_tile > 0 || rpt32 > 0) && total <= vtcm).then(|| k.to_words())
}

/// `htp_kparams_solve_unary_mul_glu`: kernel parameters for a GLU op over `src0` and `src1`
/// (same shape), or over the two halves of each `src0` row when `src1` is `None` (then `dst` is
/// half as wide as `src0`). Words: `n_threads, vtcm_row_per_thread, vtcm_size`. `None` where the
/// C++ answers unsupported: types, a strided or non-contiguous layout, or a budget that holds
/// fewer than two rows per thread.
///
/// The tensors are solved as given; the real host collapses a two-source node first
/// ([`collapse_glu`]).
pub fn solve_glu(
    opcode: u32,
    src0: &CopyTensor,
    src1: Option<&CopyTensor>,
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_size: usize,
) -> Option<[i32; 32]> {
    use HtpOpCode as O;
    if !is_glu_opcode(opcode) {
        return None;
    }
    let (s0, d) = (Gg::from(src0), Gg::from(dst));
    if (!s0.is(HtpDataType::F32) && !s0.is(HtpDataType::F16)) || d.dtype != s0.dtype {
        return None;
    }
    if (opcode == O::GluGegluQuick as u32 || opcode == O::GluGegluErf as u32)
        && !s0.is(HtpDataType::F32)
    {
        return None;
    }
    let es = s0.type_size()?;
    let valid_src = |t: &Gg| {
        t.is_contiguous_1(es)
            && t.nb[0] == es
            && (t.ne[1] <= 1 || t.nb[1] >= t.ne[0].wrapping_mul(es))
    };
    if !valid_src(&s0) || !d.is_contiguous(es) {
        return None;
    }
    if let Some(s1) = src1 {
        let s1 = Gg::from(s1);
        if s1.dtype != s0.dtype || !s0.same_shape(&s1) || !valid_src(&s1) {
            return None;
        }
    }

    let dst_row = d.ne[0].wrapping_mul(es);
    let src0_nrows = s0.ne[1].wrapping_mul(s0.ne[2]).wrapping_mul(s0.ne[3]) as u32;
    // the C++ divides by the row size; an empty row has no rows worth running
    let min_rows_per_thread = (128 + dst_row - 1).checked_div(dst_row)?.max(1) as u32;
    let max_threads = (src0_nrows / min_rows_per_thread).max(1);
    // a session never reports zero workers; the C++ divides by the count
    let n_threads = n_threads.max(1).min(max_threads);
    let vtcm = vtcm_size as u64;

    let row_aligned = u64::from(round_up(dst_row, 128));
    let nt = u64::from(n_threads);
    let mut rpt = vtcm.checked_div(nt * 3 * row_aligned)? as u32;
    if rpt < 2 {
        return None;
    }
    let nrows_per_thread = src0_nrows.wrapping_add(n_threads).wrapping_sub(1) / n_threads;
    if nrows_per_thread >= 2 && rpt > nrows_per_thread {
        let target = if nrows_per_thread >= 4 {
            (nrows_per_thread / 2) & !1u32
        } else {
            2
        };
        rpt = 2.max(rpt.min(target));
    }

    // htp_glu_vtcm_layout_build: three row buffers (gate, linear, result) per thread
    let total = 3 * row_aligned * u64::from(rpt) * nt;
    let mut w = [0i32; 32];
    w[0] = n_threads as i32;
    w[1] = rpt as i32;
    w[2] = total as u32 as i32;
    Some(w)
}

/// How many source operands an elementwise op takes, as `(min, max)`.
pub(crate) fn eltwise_operand_counts(opcode: u32) -> (usize, usize) {
    if opcode == HtpOpCode::AddId as u32 || opcode == HtpOpCode::RmsNormMulAdd as u32 {
        (3, 3)
    } else if is_binary_opcode(opcode) || opcode == HtpOpCode::RmsNormMul as u32 {
        (2, 2)
    } else if is_glu_opcode(opcode) {
        (1, 2)
    } else {
        (1, 1)
    }
}

/// Kernel parameters for an elementwise op over the tensors registered for it, as the queue
/// derives them when a caller passes none. `src` holds the op's operands in order and `dst` its
/// result, already converted to solver tensors; `None` when the operand count does not fit the
/// op or the solver answers unsupported.
///
/// Solved on the tensors as registered: collapsing would change what the DSP is handed, which is
/// the caller's to register (see [`collapse_unary`]).
pub(crate) fn derive_eltwise(
    opcode: u32,
    src: &[CopyTensor],
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_size: usize,
) -> Option<[i32; 32]> {
    if is_binary_opcode(opcode) {
        let is_add_id = opcode == HtpOpCode::AddId as u32;
        if src.len() != if is_add_id { 3 } else { 2 } {
            return None;
        }
        return solve_binary(
            opcode,
            &src[0],
            &src[1],
            is_add_id.then(|| &src[2]),
            dst,
            n_threads,
            vtcm_size,
        );
    }
    if is_unary_opcode(opcode) {
        let want = match opcode {
            x if x == HtpOpCode::RmsNormMul as u32 => 2,
            x if x == HtpOpCode::RmsNormMulAdd as u32 => 3,
            _ => 1,
        };
        if src.len() != want {
            return None;
        }
        // The fused norms read the weight as one row of `src0`'s width, and the residual with
        // `src0`'s own strides (llama.cpp's fusion pass only fuses such nodes; ggml gives no such
        // guarantee for a node built by hand).
        if want > 1 && src[1].ne[0] != src[0].ne[0] {
            return None;
        }
        if want == 3 && src[2] != src[0] {
            return None;
        }
        return solve_unary(opcode, &src[0], src.get(1), dst, n_threads, vtcm_size);
    }
    if is_glu_opcode(opcode) {
        if src.is_empty() || src.len() > 2 {
            return None;
        }
        // The kernels take their row length from `dst` and their row count from `src0`; ggml fixes
        // how the two relate when the node is built (same shape, or `dst` half as wide for a
        // single source), and the solver does not look.
        let (s, d) = (&src[0], dst);
        let shape_ok = if src.len() == 2 {
            s.ne == d.ne
        } else {
            s.ne[0] == 2 * d.ne[0] && s.ne[1..] == d.ne[1..]
        };
        if !shape_ok {
            return None;
        }
        return solve_glu(opcode, &src[0], src.get(1), dst, n_threads, vtcm_size);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `type ne0..3 nb0..3` from a golden-file side.
    fn tensor(f: &mut std::str::SplitWhitespace<'_>) -> CopyTensor {
        let mut next = || -> u64 { f.next().unwrap().parse().unwrap() };
        let dtype = next() as u32;
        let ne = [next() as u32, next() as u32, next() as u32, next() as u32];
        let nb = [next() as u32, next() as u32, next() as u32, next() as u32];
        CopyTensor { dtype, ne, nb }
    }

    /// `-` or a tensor.
    fn tensor_or_dash(s: &str) -> Option<CopyTensor> {
        let s = s.trim();
        (s != "-").then(|| tensor(&mut s.split_whitespace()))
    }

    fn words(s: &str) -> Option<[i32; 32]> {
        let s = s.trim();
        if s == "unsupported" {
            return None;
        }
        let v: Vec<i32> = s
            .split_whitespace()
            .map(|w| w.parse::<u32>().unwrap() as i32)
            .collect();
        Some(v.try_into().expect("32 words"))
    }

    /// `<op name> <opcode> ...`: the C constant's name must be the Rust opcode's, which proves the
    /// Rust discriminants match the ones the host and the skels were built with.
    fn check_opcode(name: &str, id: u32) -> u32 {
        let op = HtpOpCode::from_u32(id).unwrap_or_else(|| panic!("opcode {id} unknown"));
        let c = name.strip_prefix("HTP_OP_").expect(name);
        let want: String = c.chars().filter(|&ch| ch != '_').collect();
        assert_eq!(op.name().to_lowercase(), want.to_lowercase(), "{name}");
        id
    }

    fn golden_lines(golden: &str) -> impl Iterator<Item = (&str, &str)> {
        golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(|l| l.split_once("=>").unwrap())
    }

    fn f32_tensor(ne: [u32; 4]) -> CopyTensor {
        let nb1 = ne[0] * 4;
        CopyTensor {
            dtype: HtpDataType::F32 as u32,
            ne,
            nb: [4, nb1, nb1 * ne[1], nb1 * ne[1] * ne[2]],
        }
    }

    /// The operand relations the solvers leave to ggml (the graph builder asserts them) are
    /// checked where the queue derives the params, because the DSP loops over them unchecked.
    #[test]
    fn derived_params_refuse_operands_the_kernels_would_misread() {
        let x = f32_tensor([64, 4, 1, 1]);
        let w = f32_tensor([64, 1, 1, 1]);
        let derive = |op: HtpOpCode, src: &[CopyTensor], dst: &CopyTensor| {
            derive_eltwise(op as u32, src, dst, 6, 8 << 20)
        };
        assert!(derive(HtpOpCode::RmsNormMul, &[x, w], &x).is_some());
        assert!(derive(HtpOpCode::RmsNormMul, &[x, f32_tensor([32, 1, 1, 1])], &x).is_none());
        assert!(derive(HtpOpCode::RmsNormMulAdd, &[x, w, x], &x).is_some());
        let padded = CopyTensor {
            nb: [4, 272, 1088, 1088],
            ..x
        };
        assert!(derive(HtpOpCode::RmsNormMulAdd, &[x, w, padded], &x).is_none());
        // a GLU's two sources and result share a shape; a single source is twice as wide as its result
        assert!(derive(HtpOpCode::GluSwiglu, &[x, x], &x).is_some());
        assert!(derive(HtpOpCode::GluSwiglu, &[x, x], &f32_tensor([64, 3, 1, 1])).is_none());
        assert!(derive(HtpOpCode::GluSwiglu, &[f32_tensor([128, 4, 1, 1])], &x).is_some());
        assert!(derive(HtpOpCode::GluSwiglu, &[x], &x).is_none());
        // operand counts
        assert!(derive(HtpOpCode::Add, &[x], &x).is_none());
        assert!(derive(HtpOpCode::UnarySilu, &[x, x], &x).is_none());
        assert!(derive(HtpOpCode::Conv1D, &[x, w], &x).is_none());
    }

    /// The C divides by the session's thread count and by a result row's size, and keeps a row's
    /// bytes in 32 bits. A session without threads counts as one here, an empty row is refused, and
    /// a row of 2^30 floats (2^32 bytes) must not panic.
    #[test]
    fn solvers_handle_zero_threads_empty_rows_and_enormous_rows() {
        let x = f32_tensor([2048, 4, 1, 1]);
        let derive = |op: HtpOpCode, src: &[CopyTensor], dst: &CopyTensor, threads: u32| {
            derive_eltwise(op as u32, src, dst, threads, 8 << 20)
        };
        let one = derive(HtpOpCode::UnarySilu, &[x], &x, 1);
        assert!(one.is_some());
        assert_eq!(derive(HtpOpCode::UnarySilu, &[x], &x, 0), one);
        let glu = derive(HtpOpCode::GluSwiglu, &[x, x], &x, 1);
        assert!(glu.is_some());
        assert_eq!(derive(HtpOpCode::GluSwiglu, &[x, x], &x, 0), glu);

        let empty = f32_tensor([0, 4, 1, 1]);
        assert!(derive(HtpOpCode::UnarySilu, &[empty], &empty, 6).is_none());
        assert!(derive(HtpOpCode::GluSwiglu, &[empty, empty], &empty, 6).is_none());

        let huge = CopyTensor {
            dtype: x.dtype,
            ne: [1 << 30, 1, 1, 1],
            nb: [4, 0, 0, 0],
        };
        let _ = derive(HtpOpCode::UnarySilu, &[huge], &huge, 6);
        let _ = derive(HtpOpCode::GluSwiglu, &[huge, huge], &huge, 6);
        let _ = derive(HtpOpCode::RmsNorm, &[huge], &huge, 6);
        let _ = derive(HtpOpCode::RmsNormMul, &[huge, huge], &huge, 6);
        let scale = CopyTensor {
            ne: [1 << 30, 1, 1, 1],
            ..huge
        };
        let _ = derive(HtpOpCode::RmsNormMulAdd, &[huge, scale, scale], &huge, 6);
    }

    #[test]
    fn binary_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/binary_golden.txt");
        let (mut cases, mut unsupported) = (0usize, 0usize);
        let mut kernels = [0usize; 7];
        for (inputs, want) in golden_lines(golden) {
            let mut sides = inputs.split('|');
            let s0 = tensor(&mut sides.next().unwrap().split_whitespace());
            let s1 = tensor(&mut sides.next().unwrap().split_whitespace());
            let s2 = tensor_or_dash(sides.next().unwrap());
            let d = tensor(&mut sides.next().unwrap().split_whitespace());
            let mut run = sides.next().unwrap().split_whitespace();
            let name = run.next().unwrap();
            let opcode = check_opcode(name, run.next().unwrap().parse().unwrap());
            let threads: u32 = run.next().unwrap().parse().unwrap();
            let vtcm: usize = run.next().unwrap().parse().unwrap();
            let got = solve_binary(opcode, &s0, &s1, s2.as_ref(), &d, threads, vtcm);
            let line = format!("{inputs}=>{want}");
            assert_eq!(got, words(want), "{line}");
            match got {
                None => unsupported += 1,
                Some(w) => kernels[usize::from(binary_kernel_type(&w))] += 1,
            }
            cases += 1;
        }
        assert!(
            cases > 7000 && unsupported > 2000 && kernels.iter().all(|&n| n > 50),
            "golden file looks truncated or misses a kernel: {cases} cases, {unsupported} unsupported, {kernels:?}"
        );
    }

    #[test]
    fn unary_kernel_params_and_collapse_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/unary_golden.txt");
        let (mut cases, mut unsupported, mut collapsed, mut tiled, mut deviations) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for (inputs, want) in golden_lines(golden) {
            let mut sides = inputs.split('|');
            let s0 = tensor(&mut sides.next().unwrap().split_whitespace());
            let s1 = tensor_or_dash(sides.next().unwrap());
            let d = tensor(&mut sides.next().unwrap().split_whitespace());
            let mut run = sides.next().unwrap().split_whitespace();
            let name = run.next().unwrap();
            let opcode = check_opcode(name, run.next().unwrap().parse().unwrap());
            let threads: u32 = run.next().unwrap().parse().unwrap();
            let vtcm: usize = run.next().unwrap().parse().unwrap();

            let mut out = want.split('|');
            let want_s0 = tensor(&mut out.next().unwrap().split_whitespace());
            let want_d = tensor(&mut out.next().unwrap().split_whitespace());
            let want_words = words(out.next().unwrap());

            let line = format!("{inputs}=>{want}");
            let (c0, cd) = collapse_unary(opcode, &s0, &d, threads);
            if !(Gg::from(&s0).planes_packed() && Gg::from(&d).planes_packed())
                && (want_s0, want_d) != (s0, d)
            {
                // upstream rewrote the strides of a view with a gap between planes; see collapse_dims
                assert_eq!((c0, cd), (s0, d), "left alone: {line}");
                deviations += 1;
                continue;
            }
            assert_eq!((c0, cd), (want_s0, want_d), "collapse: {line}");
            let got = solve_unary(opcode, &c0, s1.as_ref(), &cd, threads, vtcm);
            assert_eq!(got, want_words, "{line}");
            collapsed += usize::from(c0 != s0);
            match got {
                None => unsupported += 1,
                Some(w) => tiled += usize::from(w[1] != 0),
            }
            cases += 1;
        }
        assert!(
            cases > 5000 && unsupported > 1000 && collapsed > 1000 && tiled > 50 && deviations > 0,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported, {collapsed} collapsed, {tiled} tiled, {deviations} deviations"
        );
    }

    #[test]
    fn glu_kernel_params_and_collapse_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/glu_golden.txt");
        let (mut cases, mut unsupported, mut collapsed, mut direct, mut deviations) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for (inputs, want) in golden_lines(golden) {
            let mut sides = inputs.split('|');
            let s0 = tensor(&mut sides.next().unwrap().split_whitespace());
            let s1 = tensor_or_dash(sides.next().unwrap());
            let d = tensor(&mut sides.next().unwrap().split_whitespace());
            let mut run = sides.next().unwrap().split_whitespace();
            let name = run.next().unwrap();
            let opcode = check_opcode(name, run.next().unwrap().parse().unwrap());
            let is_direct = run.next().unwrap() == "direct";
            let threads: u32 = run.next().unwrap().parse().unwrap();
            let vtcm: usize = run.next().unwrap().parse().unwrap();

            let mut out = want.split('|');
            let want_s0 = tensor(&mut out.next().unwrap().split_whitespace());
            let want_s1 = tensor_or_dash(out.next().unwrap());
            let want_d = tensor(&mut out.next().unwrap().split_whitespace());
            let want_words = words(out.next().unwrap());

            let line = format!("{inputs}=>{want}");
            let (c0, c1, cd) = if is_direct {
                (s0, s1, d)
            } else {
                collapse_glu(opcode, &s0, s1.as_ref(), &d, threads)
            };
            let packed = Gg::from(&s0).planes_packed()
                && s1.is_none_or(|s| Gg::from(&s).planes_packed())
                && Gg::from(&d).planes_packed();
            if !packed && (want_s0, want_s1, want_d) != (s0, s1, d) {
                // upstream rewrote the strides of a view with a gap between planes; see collapse_dims
                assert_eq!((c0, c1, cd), (s0, s1, d), "left alone: {line}");
                deviations += 1;
                continue;
            }
            assert_eq!((c0, c1, cd), (want_s0, want_s1, want_d), "collapse: {line}");
            let got = solve_glu(opcode, &c0, c1.as_ref(), &cd, threads, vtcm);
            assert_eq!(got, want_words, "{line}");
            collapsed += usize::from(c0 != s0);
            direct += usize::from(is_direct);
            unsupported += usize::from(got.is_none());
            cases += 1;
        }
        assert!(
            cases > 5000
                && unsupported > 1000
                && collapsed > 1000
                && direct > 500
                && deviations > 0,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported, {collapsed} collapsed, {direct} direct, {deviations} deviations"
        );
    }

    /// The byte offset of every element of `t`, in linear element order.
    fn element_offsets(t: &CopyTensor) -> Vec<u64> {
        let mut out = Vec::new();
        for i3 in 0..u64::from(t.ne[3]) {
            for i2 in 0..u64::from(t.ne[2]) {
                for i1 in 0..u64::from(t.ne[1]) {
                    for i0 in 0..u64::from(t.ne[0]) {
                        out.push(
                            i0 * u64::from(t.nb[0])
                                + i1 * u64::from(t.nb[1])
                                + i2 * u64::from(t.nb[2])
                                + i3 * u64::from(t.nb[3]),
                        );
                    }
                }
            }
        }
        out
    }

    /// Collapsing a unary or GLU operand must leave every element at the byte it was at, whether
    /// or not the planes and batches carry a gap. Upstream's merge and split steps rewrote the
    /// plane strides of a padded view and moved elements; this port leaves such views alone.
    #[test]
    fn collapse_keeps_every_element_where_it_was() {
        let (mut moved_by_a_gap, mut collapsed_packed) = (0usize, 0usize);
        for dtype in [HtpDataType::F32, HtpDataType::F16] {
            let ts: u32 = if dtype == HtpDataType::F16 { 2 } else { 4 };
            for ne0 in [3u32, 30, 64, 100] {
                for ne1 in [1u32, 2, 17, 64] {
                    for ne2 in [1u32, 3] {
                        for ne3 in [1u32, 2] {
                            for (gap_plane, gap_batch) in
                                [(0u32, 0u32), (64, 0), (0, 128), (64, 128)]
                            {
                                for threads in [1u32, 6] {
                                    let nb1 = ts * ne0;
                                    let nb2 = nb1 * ne1 + gap_plane;
                                    let nb3 = nb2 * ne2 + gap_batch;
                                    let src = CopyTensor {
                                        dtype: dtype as u32,
                                        ne: [ne0, ne1, ne2, ne3],
                                        nb: [ts, nb1, nb2, nb3],
                                    };
                                    let dst = CopyTensor {
                                        nb: [ts, nb1, nb1 * ne1, nb1 * ne1 * ne2],
                                        ..src
                                    };
                                    let (c0, cd) = collapse_unary(
                                        HtpOpCode::UnarySilu as u32,
                                        &src,
                                        &dst,
                                        threads,
                                    );
                                    assert_eq!(
                                        element_offsets(&c0),
                                        element_offsets(&src),
                                        "{src:?}"
                                    );
                                    assert_eq!(
                                        element_offsets(&cd),
                                        element_offsets(&dst),
                                        "{dst:?}"
                                    );
                                    let (g0, g1, gd) = collapse_glu(
                                        HtpOpCode::GluSwiglu as u32,
                                        &src,
                                        Some(&src),
                                        &dst,
                                        threads,
                                    );
                                    assert_eq!(
                                        element_offsets(&g0),
                                        element_offsets(&src),
                                        "{src:?}"
                                    );
                                    assert_eq!(
                                        element_offsets(&g1.unwrap()),
                                        element_offsets(&src)
                                    );
                                    assert_eq!(element_offsets(&gd), element_offsets(&dst));
                                    // a gap only exists between planes that exist
                                    if !Gg::from(&src).planes_packed() {
                                        moved_by_a_gap += 1;
                                        assert_eq!(c0, src, "a view with a gap is left alone");
                                    } else if c0 != src {
                                        collapsed_packed += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            moved_by_a_gap > 100 && collapsed_packed > 100,
            "the grid should hold both kinds: {moved_by_a_gap} with a gap, {collapsed_packed} collapsed"
        );
    }
}

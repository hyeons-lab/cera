//! Host-side parameter precomputations for Hexagon HTP operations.
//!
//! Hexagon DSP kernels avoid runtime hardware division and transcendental calls
//! by consuming precomputed integer division constants (Granlund and Montgomery FastDiv)
//! and layout descriptors generated on the host.

use super::types::{HtpDataType, HtpOpCode};

/// A value for one of the one-byte fields of the matmul params. Out-of-contract counts
/// saturate at 255 rather than wrapping: a wrapped count aliases to a plausible but wrong
/// kernel configuration (`256` threads would enqueue as `0`).
pub(super) fn wire_byte(v: usize) -> u8 {
    debug_assert!(
        v <= usize::from(u8::MAX),
        "{v} does not fit a one-byte kernel param"
    );
    v.min(usize::from(u8::MAX)) as u8
}

/// Precomputed integer division constants using Granlund and Montgomery's algorithm.
///
/// Permits the DSP to calculate `n / d` without hardware division via:
/// `((mulhi(n, mp) + n) >> l)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FastDivValues {
    pub mp: u32,
    pub l: u32,
}

/// Compute fast integer division constants for divisor `d`.
pub fn init_fastdiv(d: u32) -> FastDivValues {
    if d == 0 {
        return FastDivValues { mp: 0, l: 0 };
    }
    let mut l = 0;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - (d as u64))) / (d as u64) + 1) as u32;
    FastDivValues { mp, l }
}

/// Host-computed parameters for RMS norm.
pub fn build_rms_norm_params(eps: f32) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = eps.to_bits() as i32;
    params
}

/// Host-computed parameters for Layer norm (opcode 51 `Norm`).
pub fn build_layer_norm_params(eps: f32) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = eps.to_bits() as i32;
    params
}

/// Host-computed parameters for 1D convolution (`Conv1D`).
///
/// Encodes stride, padding, dilation, and channel groups into `op.params[0..4]`.
pub fn build_conv1d_params(stride: usize, pad: usize, dilation: usize, groups: usize) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = stride.max(1) as i32;
    params[1] = pad as i32;
    params[2] = dilation.max(1) as i32;
    params[3] = groups.max(1) as i32;
    params
}

/// Host-computed parameters for Snake activation (`UnarySnake`).
///
/// Encodes frequency parameter `alpha` (default 1.0) into `op.params[0]`.
pub fn build_snake_params(alpha: f32) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = alpha.to_bits() as i32;
    params
}

/// Host-computed parameters for 1D transposed convolution (`ConvTranspose1D`).
///
/// Encodes stride, padding, and dilation into `op.params[0..3]`.
pub fn build_conv_transpose1d_params(stride: usize, pad: usize, dilation: usize) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = stride.max(1) as i32;
    params[1] = pad as i32;
    params[2] = dilation.max(1) as i32;
    params
}

/// Host-computed parameters for Exponential Linear Unit (`UnaryElu`).
///
/// Encodes negative slope scale parameter `alpha` into `op.params[0]`.
pub fn build_elu_params(alpha: f32) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = alpha.to_bits() as i32;
    params
}

// The solvers below port llama.cpp's `htp-kparams.cpp` and the layout helpers in `htp/*-ops.h`
// word for word; `scripts/hexagon-golden` runs the real C++ against each of them.

/// `hex_round_up` (`hex-common.h`): 32-bit like the C, so an out-of-range size wraps the same way.
fn hex_round_up(n: u32, m: u32) -> u32 {
    m.wrapping_mul(n.wrapping_add(m - 1) / m)
}

/// `{mp, l}` of `init_fastdiv_values(d)` as two kernel-param words.
fn fastdiv_words(d: u32) -> [i32; 2] {
    let f = init_fastdiv(d);
    [f.mp as i32, f.l as i32]
}

/// `(ggml_blck_size, ggml_type_size)` of a wire dtype. `None` for a type none of the solvers below
/// accepts: every caller refuses such an operand whether or not its layout would pass, so a missing
/// entry never changes an answer.
fn ggml_type_layout(dtype: u32) -> Option<(u64, u64)> {
    use HtpDataType as T;
    let is = |t: T| dtype == t as u32;
    Some(if is(T::F32) || is(T::I32) {
        (1, 4)
    } else if is(T::F16) || is(T::Bf16) {
        (1, 2)
    } else if is(T::I64) {
        (1, 8)
    } else if is(T::Q4_0) || is(T::Iq4Nl) {
        (32, 18)
    } else if is(T::Q4_1) {
        (32, 20)
    } else if is(T::Q8_0) {
        (32, 34)
    } else if is(T::Mxfp4) {
        (32, 17)
    } else if is(T::Q2K) {
        (256, 84)
    } else if is(T::Q3K) {
        (256, 110)
    } else if is(T::Q4K) {
        (256, 144)
    } else if is(T::Q5K) {
        (256, 176)
    } else if is(T::Q6K) {
        (256, 210)
    } else if is(T::Q1_0) {
        (128, 18)
    } else if is(T::Q2_0) {
        (64, 18)
    } else {
        return None;
    })
}

/// `ggml_is_contiguous`: dense in every dimension (a dimension of 1 has no stride to check).
fn ggml_is_contiguous(t: &CopyTensor) -> bool {
    let Some((blck, size)) = ggml_type_layout(t.dtype) else {
        return false;
    };
    if u64::from(t.ne[0]) != blck && u64::from(t.nb[0]) != size {
        return false;
    }
    let mut next_nb = size * (u64::from(t.ne[0]) / blck);
    for i in 1..4 {
        if t.ne[i] != 1 && u64::from(t.nb[i]) != next_nb {
            return false;
        }
        next_nb = next_nb.wrapping_mul(u64::from(t.ne[i]));
    }
    true
}

/// `ggml_is_contiguous_rows`: each row is dense, whatever the strides between rows.
fn ggml_is_contiguous_rows(t: &CopyTensor) -> bool {
    ggml_type_layout(t.dtype)
        .is_some_and(|(blck, size)| u64::from(t.ne[0]) == blck || u64::from(t.nb[0]) == size)
}

/// `ggml_nelements` truncated to 32 bits, the way the solvers hold row and element counts.
fn rows_u32(ne: &[u32]) -> u32 {
    ne.iter().fold(1u64, |a, &n| a.wrapping_mul(u64::from(n))) as u32
}

/// `htp_tensor_get_tile_size`: bytes of one tiled weight tile, 0 for a type without tiles.
fn htp_tile_size(dtype: u32) -> u64 {
    use HtpDataType as T;
    let is = |t: T| dtype == t as u32;
    if is(T::Q4_0) || is(T::Iq4Nl) || is(T::Q1_0) || is(T::Q2_0) {
        576
    } else if is(T::Q4_1) || is(T::Q4K) {
        640
    } else if is(T::Q8_0) {
        1088
    } else if is(T::Q5K) {
        768
    } else if is(T::Q6K) {
        896
    } else if is(T::Q3K) {
        512
    } else if is(T::Q2K) {
        336
    } else if is(T::Mxfp4) {
        544
    } else if is(T::F16) {
        2048
    } else {
        0
    }
}

/// `htp_tensor_get_tile_k`: elements along K in one tile.
fn htp_tile_k(dtype: u32) -> u32 {
    if dtype == HtpDataType::Q1_0 as u32 {
        128
    } else if dtype == HtpDataType::Q2_0 as u32 {
        64
    } else {
        32
    }
}

/// Which `get_rows` kernel runs (`enum htp_get_rows_kernel_type`).
const GET_ROWS_SAMETYPE: u32 = 0;
const GET_ROWS_TILED: u32 = 1;
const GET_ROWS_FLAT: u32 = 2;

/// `htp_get_rows_vtcm_layout_build(...).total_bytes`: the staged index buffer, then (for the
/// kernels that convert) a double-buffered source row and F32 output row per thread.
fn get_rows_vtcm_total(
    kernel: u32,
    dtype: u32,
    ne00: u32,
    total_tasks: u32,
    idx_dtype: u32,
    n_threads: u32,
) -> u64 {
    let align = |x: u64, to: u64| (x + to - 1) & !(to - 1);
    let elem_size = if idx_dtype == HtpDataType::I64 as u32 {
        8
    } else {
        4
    };
    let src1_bytes = align(u64::from(total_tasks) * elem_size, 128) + 128;
    let src1_alloc = align(src1_bytes, 256);
    let (mut src0_bpt, mut dst_bpt) = (0, 0);
    if kernel != GET_ROWS_SAMETYPE {
        let src0_row = if kernel == GET_ROWS_TILED {
            let tile_stride = (htp_tile_size(dtype) + 127) & !127;
            let n_k_tiles = u64::from(ne00 / htp_tile_k(dtype));
            if n_k_tiles > 0 {
                n_k_tiles * tile_stride
            } else {
                tile_stride
            }
        } else {
            u64::from(htp_tensor_row_size(dtype, ne00))
        };
        src0_bpt = 2 * align(src0_row, 256);
        dst_bpt = 2 * align(u64::from(ne00) * 4, 256);
    }
    src1_alloc + (src0_bpt + dst_bpt) * u64::from(n_threads)
}

/// Kernel parameters for a `GetRows` gather of rows of `src0` chosen by `src1` into `dst`: port of
/// `htp_kparams_solve_repeat_get_rows` (`htp-kparams.cpp`) and `htp_get_rows_vtcm_layout_build`.
/// `tiled` is the table's `HTP_TENSOR_TILED` flag. `None` when the DSP has no kernel for the
/// operands, or even one thread's working set does not fit `vtcm_bytes`.
///
/// The kernel is `TILED` for a tiled table, else `SAMETYPE` when table and output share a type, else
/// `FLAT`. One task per gathered row; the thread count starts at `min(n_threads, rows)` (at least 128
/// output bytes per thread) and drops until the working set fits. Every kernel stages the index
/// vector in VTCM, so even `SAMETYPE` reports a non-zero `vtcm_size`.
///
/// A gather with no rows is accepted and reports no VTCM, as in the C++ (the DSP returns early for
/// it); callers still skip enqueuing one, since it may carry zero threads.
///
/// Words (`htp_get_rows_kernel_params`): `kernel_type, n_threads, total_tasks, vtcm_size`, then the
/// dividers `ne10, ne10 * ne11, ne02, ne03` (each `mp, l`; `{0, 0}` for a zero divisor).
pub fn build_get_rows_kernel_params(
    src0: &CopyTensor,
    src1: &CopyTensor,
    dst: &CopyTensor,
    tiled: bool,
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    use HtpDataType as T;
    let is = |d: u32, t: T| d == t as u32;
    let gatherable = [
        T::F32,
        T::F16,
        T::Bf16,
        T::I32,
        T::Q1_0,
        T::Q2_0,
        T::Q4_0,
        T::Q8_0,
        T::Q4K,
        T::Q5K,
        T::Q6K,
    ];
    if !gatherable.iter().any(|&t| is(src0.dtype, t)) {
        return None;
    }
    if !ggml_is_contiguous_rows(src0)
        || !ggml_is_contiguous_rows(src1)
        || !ggml_is_contiguous_rows(dst)
    {
        return None;
    }
    if !is(src1.dtype, T::I32) && !is(src1.dtype, T::I64) {
        return None;
    }
    if src0.dtype == dst.dtype {
        if !is(src0.dtype, T::F32) && !is(src0.dtype, T::I32) && !is(src0.dtype, T::F16) {
            return None;
        }
    } else if is(src0.dtype, T::I32) || !is(dst.dtype, T::F32) {
        return None;
    }

    let (ne00, ne02, ne03) = (src0.ne[0], src0.ne[2], src0.ne[3]);
    let (ne10, ne11, ne12) = (src1.ne[0], src1.ne[1], src1.ne[2]);
    let nr = ne10.wrapping_mul(ne11).wrapping_mul(ne12);

    // a tiled F32 or BF16 table is stored as F16 tiles
    let wtype = if tiled && (is(src0.dtype, T::F32) || is(src0.dtype, T::Bf16)) {
        T::F16 as u32
    } else {
        src0.dtype
    };
    let kernel = if tiled {
        GET_ROWS_TILED
    } else if src0.dtype == dst.dtype {
        GET_ROWS_SAMETYPE
    } else {
        GET_ROWS_FLAT
    };

    let total_tasks = nr;
    let dst_nb1 = u64::from(dst.nb[1]);
    if dst_nb1 == 0 {
        return None; // the C++ divides by it
    }
    let min_tasks_per_thread = 128u64.div_ceil(dst_nb1).max(1) as u32;
    let max_threads = (total_tasks / min_tasks_per_thread).max(1);
    let mut threads = n_threads.min(max_threads);

    let total = |n: u32| get_rows_vtcm_total(kernel, wtype, ne00, total_tasks, src1.dtype, n);
    let mut total_bytes = 0u64;
    while threads > 0 {
        total_bytes = total(threads);
        if total_bytes <= vtcm_bytes as u64 {
            break;
        }
        threads -= 1;
    }
    if threads == 0 && total_tasks > 0 {
        total_bytes = total(1);
    }
    let vtcm_size = if total_tasks == 0 {
        0
    } else {
        total_bytes as u32
    };

    let div = |d: u32| if d > 0 { fastdiv_words(d) } else { [0, 0] };
    let mut k = [0i32; 32];
    k[0] = kernel as i32;
    k[1] = threads as i32;
    k[2] = total_tasks as i32;
    k[3] = vtcm_size as i32;
    for (i, d) in [ne10, ne10.wrapping_mul(ne11), ne02, ne03]
        .into_iter()
        .enumerate()
    {
        k[4 + 2 * i..6 + 2 * i].copy_from_slice(&div(d));
    }
    let fits = threads > 0 && u64::from(vtcm_size) <= vtcm_bytes as u64;
    (total_tasks == 0 || fits).then_some(k)
}

/// The 16 op-param words of a `Rope` (`struct ggml_tensor::op_params`): `n_dims` (1), `mode` (2),
/// the truncate flag (3, 0: the only form the DSP implements), `n_ctx_orig` (4), the floats
/// `freq_base`, `freq_scale`, `ext_factor` (0: no YaRN), `attn_factor`, `beta_fast`, `beta_slow`
/// (5 to 10), and zero multi-axis sections (11 to 14) and row offset (15). The DSP reads them itself;
/// the host solver ([`build_rope_kernel_params`]) reads them to size the kernel.
pub fn build_rope_params(
    n_dims: usize,
    mode: u32,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[1] = n_dims as i32;
    params[2] = mode as i32;
    params[4] = n_ctx_orig as i32;
    params[5] = freq_base.to_bits() as i32;
    params[6] = freq_scale.to_bits() as i32;
    params[7] = 0.0f32.to_bits() as i32; // ext_factor
    params[8] = 1.0f32.to_bits() as i32; // attn_factor
    params[9] = 32.0f32.to_bits() as i32; // beta_fast
    params[10] = 1.0f32.to_bits() as i32; // beta_slow
    params
}

/// Operand views for [`build_rope_kernel_params`], as `htp_kparams_solve_rope` reads them.
#[derive(Debug, Clone, Copy)]
pub struct RopeOperands<'a> {
    /// The rotated activation (`src0`).
    pub src0: &'a CopyTensor,
    /// The I32 position vector (`src1`).
    pub positions: &'a CopyTensor,
    /// The optional F32 frequency factors (`src2`): each rotated pair's angle is divided by its factor.
    pub freq_factors: Option<&'a CopyTensor>,
    pub dst: &'a CopyTensor,
}

/// Kernel parameters for a `Rope`: port of `htp_kparams_solve_rope` (`htp-kparams.cpp`) and
/// `htp_rope_vtcm_layout_build` (`rope-ops.h`). `op_params` is the op's 16 parameter words
/// ([`build_rope_params`]): `n_dims` (1), `mode` (2), the truncate flag (3, must be 0), `freq_base`
/// (5), the multi-axis `sections` (11..15) and `n_offs` (15). `None` where the solver refuses:
/// non-F32 operands, a destination of another shape, a rotated span (`n_offs + n_dims`) wider than the
/// row, `n_offs` not a multiple of 32, unknown modes, too few positions or factors, strided rows, or
/// a working set that does not fit `vtcm_bytes`.
///
/// Each thread holds a theta cache and 32 rows of VTCM, and the frequency factors (when present) get
/// one shared 256-byte-aligned slot after the per-thread slabs.
///
/// Words (`htp_rope_kernel_params`): `n_threads` (the session's, capped at the row count),
/// `vtcm_size`, then the dividers `ne2 * ne1` and `ne1` of the destination (each `mp, l`).
pub fn build_rope_kernel_params(
    ops: &RopeOperands<'_>,
    op_params: &[i32; 16],
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    use HtpDataType as T;
    const ROPE_NORMAL: i32 = 0;
    const ROPE_NEOX: i32 = 2;
    const ROPE_MROPE: i32 = 8;
    const ROPE_VISION: i32 = 24;
    const ROPE_IMROPE: i32 = 40;
    let RopeOperands {
        src0,
        positions,
        freq_factors,
        dst,
    } = *ops;

    // op_params[3]: the DSP implements only the truncating form
    if op_params[3] != 0 {
        return None;
    }
    if src0.ne != dst.ne {
        return None;
    }
    if src0.dtype != T::F32 as u32 || dst.dtype != T::F32 as u32 || positions.dtype != T::I32 as u32
    {
        return None;
    }
    if src0.ne[0] == 0 {
        return None;
    }
    let src0_nrows = rows_u32(&src0.ne[1..]);
    if src0_nrows == 0 {
        return None;
    }

    let n_dims = op_params[1];
    let mode = op_params[2];
    let n_offs = op_params[15];
    if n_dims < 0 || n_dims % 2 != 0 {
        return None;
    }
    if n_offs < 0
        || n_offs % 32 != 0
        || i64::from(n_offs) + i64::from(n_dims) > i64::from(src0.ne[0])
    {
        return None;
    }
    if f32::from_bits(op_params[5] as u32) < 0.0 {
        return None;
    }
    if ![ROPE_NORMAL, ROPE_NEOX, ROPE_MROPE, ROPE_VISION, ROPE_IMROPE].contains(&mode) {
        return None;
    }
    let is_mrope = mode & ROPE_MROPE != 0;
    // vision rotates the whole row, so it takes no offset and exactly half the row as n_dims
    if mode == ROPE_VISION && (i64::from(n_dims) != i64::from(src0.ne[0] / 2) || n_offs != 0) {
        return None;
    }
    if is_mrope && op_params[11..14].iter().all(|&s| s <= 0) {
        return None;
    }
    // multi-axis modes read four position streams
    let min_pos_len = if is_mrope || mode == ROPE_VISION {
        i64::from(src0.ne[2]) * 4
    } else {
        i64::from(src0.ne[2])
    };
    if i64::from(positions.ne[0]) < min_pos_len || !ggml_is_contiguous(positions) {
        return None;
    }
    if let Some(ff) = freq_factors
        && (ff.dtype != T::F32 as u32
            || !ggml_is_contiguous(ff)
            || i64::from(ff.ne[0]) < i64::from(n_dims / 2))
    {
        return None;
    }
    if src0.nb[0] != 4
        || dst.nb[0] != 4
        || u64::from(src0.nb[1]) < u64::from(src0.ne[0]) * 4
        || u64::from(dst.nb[1]) < u64::from(dst.ne[0]) * 4
    {
        return None;
    }

    let threads = n_threads.min(src0_nrows);
    let n_freq_factors = freq_factors.map_or(0, |ff| ff.ne[0]);
    // htp_rope_vtcm_layout_build
    let row = u64::from(src0.ne[0]) * 4;
    let src0_row_aligned = u64::from(hex_round_up(row as u32, 128));
    let theta_aligned = u64::from(hex_round_up(row as u32, 256));
    let ff_aligned = u64::from(hex_round_up((u64::from(n_freq_factors) * 4) as u32, 256));
    // HTP_ROPE_SPAD_NROWS = HTP_ROPE_SPAD_BLOCK (8) * HTP_ROPE_SPAD_NSLOTS (4)
    let bytes_per_thread = theta_aligned + 32 * src0_row_aligned;
    let total =
        bytes_per_thread * u64::from(threads) + if n_freq_factors > 0 { ff_aligned } else { 0 };
    if total > vtcm_bytes as u64 {
        return None;
    }

    let mut k = [0i32; 32];
    k[0] = threads as i32;
    k[1] = total as u32 as i32;
    k[2..4].copy_from_slice(&fastdiv_words(
        (u64::from(dst.ne[2]) * u64::from(dst.ne[1])) as u32,
    ));
    k[4..6].copy_from_slice(&fastdiv_words(dst.ne[1]));
    Some(k)
}

/// Bytes in one row of `ne00` elements of wire type `dtype`: port of `htp_tensor_get_row_size`
/// (`htp-tensor.h`), 0 for a type it does not know. 32-bit arithmetic, like the C.
pub(crate) fn htp_tensor_row_size(dtype: u32, ne00: u32) -> u32 {
    use HtpDataType as T;
    let is = |t: T| dtype == t as u32;
    if is(T::F32) || is(T::I32) {
        ne00.wrapping_mul(4)
    } else if is(T::F16) || is(T::Bf16) {
        ne00.wrapping_mul(2)
    } else if is(T::Q8_0) {
        (ne00 / 32).wrapping_mul(34)
    } else if is(T::Q4_0) || is(T::Iq4Nl) {
        (ne00 / 32).wrapping_mul(18)
    } else if is(T::Q4_1) {
        (ne00 / 32).wrapping_mul(20)
    } else if is(T::Mxfp4) {
        (ne00 / 32).wrapping_mul(17)
    } else if is(T::Q2K) {
        (ne00 / 256).wrapping_mul(84)
    } else if is(T::Q2_0) {
        (ne00 / 64).wrapping_mul(18)
    } else if is(T::Q1_0) {
        (ne00 / 128).wrapping_mul(18)
    } else if is(T::Q3K) {
        (ne00 / 256).wrapping_mul(110)
    } else if is(T::Q4K) {
        (ne00 / 256).wrapping_mul(144)
    } else if is(T::Q5K) {
        (ne00 / 256).wrapping_mul(176)
    } else if is(T::Q6K) {
        (ne00 / 256).wrapping_mul(210)
    } else if is(T::I64) {
        ne00.wrapping_mul(8)
    } else {
        0
    }
}

/// Kernel parameters for a `SetRows` that scatters `src0` (F32 value rows, `ne[1]` of them) into
/// the cache `dst` at the positions in `idx` (I32 or I64): port of
/// `htp_kparams_solve_set_rows` and `htp_set_rows_vtcm_layout_build`. `None` when the kernels do
/// not take the operands or the working set does not fit `vtcm_bytes`.
///
/// Words (`htp_set_rows_kernel_params`): 0 `n_threads` (the session's, capped at the row count),
/// 1 `total_tasks` (the row count), 2 `vtcm_size`. The DSP derives its divisors itself.
pub fn build_set_rows_kernel_params(
    src0: &CopyTensor,
    idx: &CopyTensor,
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    use HtpDataType as T;
    let ne00 = src0.ne[0];
    if dst.dtype == T::Q8_0 as u32 && ne00 < 32 {
        return None;
    }
    if src0.dtype != T::F32 as u32 {
        return None;
    }
    if idx.dtype != T::I32 as u32 && idx.dtype != T::I64 as u32 {
        return None;
    }
    if ![T::F32 as u32, T::F16 as u32, T::Q8_0 as u32].contains(&dst.dtype) {
        return None;
    }
    let nr = src0.ne[1];
    let threads = n_threads.min(nr);
    // htp_set_rows_vtcm_layout_build: double-buffered value and cache rows, 256-aligned, per thread
    let align256 = |n: u64| (n + 255) & !255;
    let src0_row = align256(u64::from(ne00) * 4);
    let dst_row = align256(u64::from(htp_tensor_row_size(dst.dtype, ne00)));
    let total = (src0_row * 2 + dst_row * 2) * u64::from(threads);
    let mut k = [0i32; 32];
    k[0] = threads as i32;
    k[1] = nr as i32;
    k[2] = total as u32 as i32;
    (total <= vtcm_bytes as u64).then_some(k)
}

/// Kernel parameters for an `SsmConv` (depthwise 1-D convolution, `y[c, t] = sum_j x[t + j, c] *
/// w[j, c]`): port of `htp_kparams_solve_ssm_conv` (`htp-kparams.cpp`) with
/// `htp_ssm_conv_vtcm_layout_build` (`ssm-conv.h`). `src0` is the `[d_conv - 1 + n_t, d_inner,
/// n_s]` window (rows of `ne[0]` dense floats), `src1` the `[d_conv, d_inner]` taps, `dst` the
/// `[d_inner, n_t, n_s]` output. `None` when the operands are not a convolution the kernel takes
/// (F32 only, at most 32 taps, dense rows) or its working set does not fit `vtcm_bytes`.
///
/// Words (`htp_ssm_conv_kernel_params`): `n_threads` (the session's, capped at one per 32
/// channels), `d_conv`, `d_inner`, `n_t`, `n_s`, `d_inner_tile` (the channel tile one thread
/// stages), `vtcm_size`, `silu`, then the divider of `d_inner_tile` (`mp, l`). `silu` is set by
/// llama.cpp's op fusion when a SiLU follows the convolution; cera applies its activations as
/// separate ops, so it stays 0. A session without threads divides by zero in the C and never
/// reaches it, so it counts as one here.
pub fn build_ssm_conv_kernel_params(
    src0: &CopyTensor,
    src1: &CopyTensor,
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    let f32t = HtpDataType::F32 as u32;
    if src0.dtype != f32t || src1.dtype != f32t || dst.dtype != f32t {
        return None;
    }
    if src0.ne[3] != 1 || src1.ne[2] != 1 || src1.ne[3] != 1 || dst.ne[3] != 1 {
        return None;
    }
    let (d_conv, d_inner, n_t, n_s) = (src1.ne[0], src0.ne[1], dst.ne[1], dst.ne[2]);
    if d_conv == 0 || d_conv > 32 || d_inner == 0 {
        return None;
    }
    if u64::from(src0.ne[0]) != u64::from(d_conv) - 1 + u64::from(n_t)
        || src0.ne[2] != n_s
        || src1.ne[1] != d_inner
        || dst.ne[0] != d_inner
    {
        return None;
    }
    // the kernel DMAs rows of the window and the taps whole
    if src0.nb[0] != 4 || src1.nb[0] != 4 || dst.nb[0] != 4 {
        return None;
    }
    if u64::from(src0.nb[1]) != u64::from(src0.ne[0]) * 4
        || u64::from(src1.nb[1]) != u64::from(src1.ne[0]) * 4
    {
        return None;
    }

    let ncs = src0.ne[0];
    let n_threads = n_threads.min(d_inner.div_ceil(32)).max(1);
    let raw_rpt = d_inner.div_ceil(n_threads);
    let d_inner_per_thread = hex_round_up(raw_rpt, 32);

    let mut k = [0i32; 32];
    // the weight side of one thread's slab, in both branches: the raw taps plus the transposed
    // copy the HVX kernel multiplies from
    let ssm_src1_per_thread = |dipt: u32| -> u64 {
        let raw = u64::from(hex_round_up(
            (u64::from(dipt) * u64::from(d_conv) * 4) as u32,
            128,
        )) + 128;
        let transposed = u64::from(hex_round_up(
            (u64::from(d_conv) * u64::from(dipt) * 4) as u32,
            128,
        ));
        raw + transposed
    };

    let d_inner_tile = if n_t == 1 {
        d_inner_per_thread
    } else {
        let vtcm_src1_per_thread = ssm_src1_per_thread(d_inner_per_thread) as u32;
        let vtcm_budget = if vtcm_bytes > 0 {
            vtcm_bytes as u64 / u64::from(n_threads)
        } else {
            1024 * 1024
        };
        let src0_block_t = hex_round_up(ncs.wrapping_mul(32 * 4), 128);
        let fixed_bytes = u64::from(vtcm_src1_per_thread.wrapping_add(src0_block_t));
        let avail_for_tiles = if vtcm_budget > fixed_bytes {
            vtcm_budget - fixed_bytes
        } else {
            128 * 1024
        };

        // the tile is a quarter of a thread's channels, between 32 and 128
        let target_max_tile = hex_round_up(d_inner_per_thread.div_ceil(4), 32).clamp(32, 128);
        let raw_stride_bytes = u64::from(ncs) * 4 + 4;
        let per_channel = 2 * (raw_stride_bytes + u64::from(n_t) * 4);
        let tile = ((avail_for_tiles / per_channel) as u32 / 32 * 32).max(32);
        tile.min(target_max_tile).min(d_inner_per_thread)
    };

    // htp_ssm_conv_vtcm_layout_build
    let (src0_per_thread, dst_per_thread);
    if n_t == 1 {
        let src0_tile_raw = u64::from(hex_round_up(
            (u64::from(d_inner_tile) * u64::from(d_conv) * 4) as u32,
            128,
        ));
        let src0_t = u64::from(hex_round_up(
            (u64::from(d_conv) * u64::from(d_inner_tile) * 4) as u32,
            128,
        ));
        src0_per_thread = 2 * src0_tile_raw + src0_t;
        dst_per_thread = 2 * u64::from(hex_round_up((u64::from(d_inner_tile) * 4) as u32, 128));
    } else {
        let raw_stride_bytes = u64::from(ncs) * 4 + 4;
        let src0_block_t = u64::from(hex_round_up((u64::from(ncs) * 32 * 4) as u32, 128));
        let src0_tile_raw = u64::from(hex_round_up(
            (u64::from(d_inner_tile) * raw_stride_bytes) as u32,
            128,
        ));
        src0_per_thread = 2 * src0_tile_raw + src0_block_t;
        dst_per_thread = 2 * u64::from(hex_round_up(
            (u64::from(d_inner_tile) * u64::from(n_t) * 4) as u32,
            128,
        ));
    }
    let total = (src0_per_thread + ssm_src1_per_thread(d_inner_per_thread) + dst_per_thread)
        * u64::from(n_threads);

    let div = init_fastdiv(d_inner_tile);
    for (i, v) in [
        n_threads,
        d_conv,
        d_inner,
        n_t,
        n_s,
        d_inner_tile,
        total as u32,
        0, // silu
        div.mp,
        div.l,
    ]
    .into_iter()
    .enumerate()
    {
        k[i] = v as i32;
    }
    (total <= vtcm_bytes as u64).then_some(k)
}

/// `enum htp_fa_kernel_type`: the HVX kernel walks one query row (or one GQA group) at a time, the
/// HMX kernel runs tiled `Q K^T` and `P V` on the matrix unit.
const FA_KERNEL_HVX: u8 = 1;
const FA_KERNEL_HMX: u8 = 2;
/// `HTP_FA_FLAG_TEAM_SPLIT`: several devices share the query blocks of one KV head.
const FA_FLAG_TEAM_SPLIT: u8 = 0x01;
/// KV rows per block of the HVX kernel (`FA_HVX_BLOCK_SIZE`).
const FA_HVX_BLOCK_SIZE: usize = 64;
/// Query rows (GQA group members) one HVX pass handles (`FA_HVX_G_MAX`).
const FA_HVX_G_MAX: usize = 8;
/// HVX mask rows cached per thread (`HVX_FA_DMA_CACHE_SIZE`).
const FA_HVX_DMA_CACHE: usize = 128;
/// HMX flash-attention VTCM group alignment and tile size (`HTP_FA_HMX_TILE_SIZE`).
const FA_HMX_TILE_BYTES: usize = 2048;
/// HMX FA tile rows and columns (`HMX_FP16_TILE_N_ROWS`, `HMX_FP16_TILE_N_COLS`).
const FA_TILE_ROWS: usize = 32;
const FA_TILE_COLS: usize = 32;
/// Minimum KV blocks for pipelining (`FA_MIN_KV_BLOCKS`).
const FA_MIN_KV_BLOCKS: usize = 3;
/// HMX FA mask DMA cache slots (`HMX_FA_DMA_CACHE_SIZE`).
const FA_HMX_DMA_CACHE: usize = 4;
/// Solver cost weights, calibrated from profiling (`hmx_fa_find_chunk_size`).
const FA_C_Q_FIXED: usize = 800;
const FA_C_ITER_BASE: usize = 200;
const FA_C_SOFTMAX: usize = 600;

/// What `htp_kparams_solve_flash_attn` reads about the session and the host options.
///
/// [`FlashAttnHost::new`] is a single device with llama.cpp's shipped options
/// (`opt_fa_select = 2`, `opt_fa_split = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttnHost {
    /// `htp_session_n_threads`.
    pub n_threads: u32,
    /// `htp_session_vtcm_size`: the VTCM budget a tiling must fit.
    pub vtcm_bytes: usize,
    /// `htp_session_n_hmx`: 0 keeps the HMX kernel off.
    pub n_hmx: u32,
    /// `htp_session_mdev_count`.
    pub mdev_count: u32,
    /// `opt_fa_select`: 0 refuses the op, 1 is HVX only, 2 picks HMX by the heuristics and falls
    /// back to HVX, above 2 takes HMX whenever it is eligible.
    pub select: i32,
    /// `opt_fa_split`: partition by KV heads (and by teams on several devices).
    pub split: bool,
}

impl FlashAttnHost {
    pub fn new(n_threads: u32, vtcm_bytes: usize, n_hmx: u32) -> Self {
        Self {
            n_threads,
            vtcm_bytes,
            n_hmx,
            mdev_count: 1,
            select: 2,
            split: true,
        }
    }

    /// The same session with the HMX kernel off, for A/B runs against the HVX one.
    pub fn hvx_only(self) -> Self {
        Self { n_hmx: 0, ..self }
    }
}

/// The operands of one `FLASH_ATTN_EXT`: `q`, `k`, `v`, the optional mask and attention sinks, the
/// output, and the three op params llama.cpp stores in `op_params[0..3]`. Only extents and types
/// are read; strides play no part in the choice.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnOp<'a> {
    pub q: &'a CopyTensor,
    pub k: &'a CopyTensor,
    pub v: &'a CopyTensor,
    pub mask: Option<&'a CopyTensor>,
    pub sinks: Option<&'a CopyTensor>,
    pub dst: &'a CopyTensor,
    pub scale: f32,
    pub max_bias: f32,
    pub logit_softcap: f32,
}

/// `kernel_params` that ask the queue (`validate_and_derive_op_params`) to solve a `FlashAttnExt`
/// on the HVX kernel only (`CERA_HEXAGON_*_FA=hvx`, `CERA_HEXAGON_HMX=0`). The
/// kernel byte alone is never a solved blob: those always carry a scale and a block count.
pub const FLASH_ATTN_HVX_ONLY: [i32; 32] = {
    let mut k = [0i32; 32];
    k[0] = FA_KERNEL_HVX as i32;
    k
};

/// `op_params` of a `FlashAttnExt`, laid out as ggml's node stores them: scale, ALiBi bias, soft
/// cap. The DSP reads none of them (the solved kernel params carry the folded values); the queue
/// reads them back to solve the kernel params when the caller passes none.
pub fn flash_attn_op_params(scale: f32, logit_softcap: f32) -> [i32; 16] {
    let mut p = [0i32; 16];
    p[0] = scale.to_bits() as i32;
    p[2] = logit_softcap.to_bits() as i32;
    p
}

/// The `kernel_params` argument of a `FlashAttnExt` whose kernel the queue solves from the
/// registered tensors: free to take HMX when the heuristics pick it, or HVX only.
pub fn flash_attn_kernel_request(allow_hmx: bool) -> [i32; 32] {
    if allow_hmx {
        [0; 32]
    } else {
        FLASH_ATTN_HVX_ONLY
    }
}

/// The two flash-attention kernels place different things in the tail of `htp_fa_kernel_params`
/// (a union in the C struct).
#[derive(Debug, Clone, Copy, PartialEq)]
enum FaKernelUnion {
    Hmx {
        g_br: u32,
        mask_broadcast: i32,
        pipeline: i32,
        div_g: FastDivValues,
    },
    Hvx {
        size_q_row_padded: u32,
        size_k_row_padded: u32,
        size_v_row_padded: u32,
        src0_div21: FastDivValues,
        src0_div1: FastDivValues,
    },
}

/// `struct htp_fa_kernel_params` (`flash-attn-ops.h`): the four one-byte fields and four `u16`s
/// share the first three words, then the floats and sizes, six `{mp, l}` dividers, and the union.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FaKernelParams {
    kernel_type: u8,
    head_split: u8,
    flags: u8,
    n_threads: u8,
    br: u16,
    bc: u16,
    n_kv_blocks: u16,
    g: u16,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    vtcm_size: u32,
    m0: f32,
    m1: f32,
    n_head_log2: u32,
    src3_div2: FastDivValues,
    src3_div3: FastDivValues,
    broadcast_rk2: FastDivValues,
    broadcast_rk3: FastDivValues,
    broadcast_rv2: FastDivValues,
    broadcast_rv3: FastDivValues,
    tail: FaKernelUnion,
}

impl FaKernelParams {
    fn to_words(self) -> [i32; 32] {
        let mut w = [0i32; 32];
        w[0] = u32::from_le_bytes([
            self.kernel_type,
            self.head_split,
            self.flags,
            self.n_threads,
        ]) as i32;
        w[1] = (u32::from(self.br) | (u32::from(self.bc) << 16)) as i32;
        w[2] = (u32::from(self.n_kv_blocks) | (u32::from(self.g) << 16)) as i32;
        w[3] = self.scale.to_bits() as i32;
        w[4] = self.max_bias.to_bits() as i32;
        w[5] = self.logit_softcap.to_bits() as i32;
        w[6] = self.vtcm_size as i32;
        w[7] = self.m0.to_bits() as i32;
        w[8] = self.m1.to_bits() as i32;
        w[9] = self.n_head_log2 as i32;
        let divs = [
            self.src3_div2,
            self.src3_div3,
            self.broadcast_rk2,
            self.broadcast_rk3,
            self.broadcast_rv2,
            self.broadcast_rv3,
        ];
        for (i, d) in divs.iter().enumerate() {
            w[10 + 2 * i] = d.mp as i32;
            w[11 + 2 * i] = d.l as i32;
        }
        match self.tail {
            FaKernelUnion::Hmx {
                g_br,
                mask_broadcast,
                pipeline,
                div_g,
            } => {
                w[22] = g_br as i32;
                w[23] = mask_broadcast;
                w[24] = pipeline;
                w[25] = div_g.mp as i32;
                w[26] = div_g.l as i32;
            }
            FaKernelUnion::Hvx {
                size_q_row_padded,
                size_k_row_padded,
                size_v_row_padded,
                src0_div21,
                src0_div1,
            } => {
                w[22] = size_q_row_padded as i32;
                w[23] = size_k_row_padded as i32;
                w[24] = size_v_row_padded as i32;
                w[25] = src0_div21.mp as i32;
                w[26] = src0_div21.l as i32;
                w[27] = src0_div1.mp as i32;
                w[28] = src0_div1.l as i32;
            }
        }
        w
    }
}

/// `hex_gcd_u32`.
fn fa_gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `hex_round_up`: 32-bit like the C, so an oversized operand wraps the same way.
fn fa_round_up(n: u32, m: u32) -> u32 {
    m.wrapping_mul(n.wrapping_add(m - 1) / m)
}

/// Exact HMX FA VTCM footprint (`hmx_fa_vtcm_layout_build`, its `total_bytes`). `dk`/`dv` are the
/// head dims padded to 64.
#[allow(clippy::too_many_arguments)]
fn fa_hmx_layout_total(
    g: usize,
    dk: usize,
    dv: usize,
    br: usize,
    bc: usize,
    n_threads: usize,
    pipeline: bool,
    is_q_fp32: bool,
    has_sinks: bool,
    n_heads: usize,
) -> usize {
    let g_br = (g * br).next_multiple_of(FA_TILE_ROWS);
    let q_tile = (g_br * dk * 2).next_multiple_of(FA_HMX_TILE_BYTES);
    let o_tile = (g_br * dv * 2).next_multiple_of(FA_HMX_TILE_BYTES);
    let k_tile = (bc * dk * 2).next_multiple_of(FA_HMX_TILE_BYTES);
    let v_tile = (bc * dv * 2).next_multiple_of(FA_HMX_TILE_BYTES);
    let s_tile = (g_br * bc * 2).next_multiple_of(FA_HMX_TILE_BYTES);
    // The rescale matrices are diagonal: only the g_br/32 diagonal tiles exist, one tile apart.
    let d_tile = (g_br / FA_TILE_ROWS) * FA_HMX_TILE_BYTES;
    let q_dma = (g_br * dk * if is_q_fp32 { 4 } else { 2 }).next_multiple_of(128);
    let k_dma = (bc * fa_round_up((dk * 2) as u32, 128) as usize).next_multiple_of(128);
    let v_dma = (bc * fa_round_up((dv * 2) as u32, 128) as usize).next_multiple_of(128);
    let col_vec = (g_br * 4).next_multiple_of(256);
    let row_vec = (bc * 2).next_multiple_of(256);
    let m_line = (bc * 2).next_multiple_of(128);
    let m_buf = (br * m_line).next_multiple_of(256) * FA_HMX_DMA_CACHE;
    let slopes = (g_br * 2).next_multiple_of(128);
    let sinks = if has_sinks {
        fa_round_up((n_heads * 4) as u32, 128) as usize
    } else {
        0
    };

    // Group A, part 1: the HMX-tiled buffers (Q, two O, the diagonal rescale tiles, 1/l).
    let mut off = q_tile + o_tile * 2 + d_tile + d_tile;
    if pipeline {
        off += d_tile;
    }
    // Groups B (compute-only buffers) and C (the Q fetch buffer) overlap from a 2 KiB boundary.
    let base = off.next_multiple_of(FA_HMX_TILE_BYTES);
    let copies = if pipeline { 2 } else { 1 };
    let group_b = (k_tile + v_tile + s_tile * 2) * copies + col_vec * 2 + row_vec * 2 * n_threads;
    off = base + group_b.max(q_dma);
    // Group A, part 2: the K/V staging rows, the running max and sum, the scale constants.
    off + k_dma * 2 + v_dma * 2 + col_vec * 2 + 256 + 256 + m_buf + slopes + sinks
}

/// HVX FA VTCM footprint (`hvx_fa_vtcm_layout_build`, its `total_bytes`): per thread one Q block,
/// double-buffered K and V blocks, the mask rows when there is a mask, and the f32 accumulators.
fn fa_hvx_layout_total(
    dk: usize,
    dv: usize,
    is_q_fp32: bool,
    has_mask: bool,
    has_sinks: bool,
    n_heads: usize,
    n_threads: usize,
) -> usize {
    let q_row = fa_round_up((dk * if is_q_fp32 { 4 } else { 2 }) as u32, 128) as usize;
    let k_row = fa_round_up((dk * 2) as u32, 128) as usize;
    let v_row = fa_round_up((dv * 2) as u32, 128) as usize;
    let q_block = q_row * FA_HVX_G_MAX;
    let k_block = k_row * FA_HVX_BLOCK_SIZE;
    let v_block = v_row * FA_HVX_BLOCK_SIZE;
    let m_block = fa_round_up((FA_HVX_BLOCK_SIZE * 2) as u32, 128) as usize;
    let vkq_acc = fa_round_up((dv * 4) as u32, 128) as usize * FA_HVX_G_MAX;
    let sinks = fa_round_up((n_heads * 4) as u32, 128) as usize;
    q_block * n_threads
        + k_block * 2 * n_threads
        + v_block * 2 * n_threads
        + if has_mask {
            m_block * FA_HVX_DMA_CACHE * n_threads
        } else {
            0
        }
        + vkq_acc * n_threads
        + if has_sinks { sinks } else { 0 }
}

/// HMX FA (Br, Bc) solver (`hmx_fa_find_chunk_size`): the cost-minimal tiling whose VTCM layout
/// fits `vtcm_budget`. `dk`/`dv` are padded to 64; `q_split` is the team size (query blocks one
/// device of a team owns).
#[allow(clippy::too_many_arguments)]
fn fa_hmx_find_chunk_size(
    g: usize,
    dk: usize,
    dv: usize,
    qo_len: usize,
    kv_len: usize,
    vtcm_budget: usize,
    n_threads: usize,
    is_q_fp32: bool,
    has_sinks: bool,
    n_heads: usize,
    q_split: usize,
) -> Option<(usize, usize)> {
    let br_unit = FA_TILE_ROWS.div_ceil(g);
    let bc_unit = FA_TILE_COLS * 2;
    let can_pipeline = kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2;
    let br_max = if qo_len >= br_unit {
        qo_len / br_unit * br_unit
    } else {
        br_unit
    };
    // cap Bc so there are enough blocks to pipeline, unless the KV is too short for that
    let mut bc_limit = if can_pipeline {
        kv_len / FA_MIN_KV_BLOCKS / bc_unit * bc_unit
    } else if kv_len >= bc_unit {
        kv_len / bc_unit * bc_unit
    } else {
        bc_unit
    };
    // small-Q sliding-window-sized caches keep Bc at 256 for block-skipping granularity
    if qo_len <= 16 && kv_len <= 2048 && bc_limit > 256 {
        bc_limit = 256;
    }
    let mut best_cost = usize::MAX;
    let mut best_mn = 0;
    let mut best = (0, 0);
    let mut br = br_max;
    loop {
        let mut bc = bc_limit;
        while bc >= bc_unit {
            let need = fa_hmx_layout_total(
                g,
                dk,
                dv,
                br,
                bc,
                n_threads,
                can_pipeline,
                is_q_fp32,
                has_sinks,
                n_heads,
            );
            if need <= vtcm_budget {
                let q_blocks = qo_len.div_ceil(br);
                let kv_blocks = kv_len.div_ceil(bc);
                let actual = if kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2 {
                    n_threads
                } else {
                    1
                };
                let cnt = (br * g).div_ceil(64);
                let n_use = cnt.min(actual);
                let vpt = if n_use > 0 { cnt.div_ceil(n_use) } else { 1 };
                let iter = FA_C_ITER_BASE + FA_C_SOFTMAX * vpt;
                let q_blocks_split = q_blocks.div_ceil(q_split.max(1));
                let cost = q_blocks_split * (FA_C_Q_FIXED + kv_blocks * iter);
                let mn = br * bc;
                if cost < best_cost || (cost == best_cost && mn > best_mn) {
                    best_cost = cost;
                    best_mn = mn;
                    best = (br, bc);
                }
                // Bc descends, so the first fit is the largest for this Br
                break;
            }
            bc -= bc_unit;
        }
        if br == br_unit {
            break;
        }
        br -= br_unit;
    }
    (best != (0, 0)).then_some(best)
}

/// Whether the HMX kernel is a candidate (`htp_flash_attn_is_hmx_eligible`): an HMX unit, f16 or
/// Q8_0 K and V, head dims that tile by 8, then llama.cpp's calibrated crossover from the HVX
/// kernel by query count, KV length, GQA factor and thread count.
fn fa_is_hmx_eligible(host: &FlashAttnHost, op: &FlashAttnOp<'_>) -> bool {
    use HtpDataType as T;
    if host.n_hmx == 0 || host.select < 2 {
        return false;
    }
    let kv_ok = |t: &CopyTensor| t.dtype == T::F16 as u32 || t.dtype == T::Q8_0 as u32;
    if !kv_ok(op.k) || !kv_ok(op.v) {
        return false;
    }
    let dk = op.q.ne[0];
    let dv = op.v.ne[0];
    if !dk.is_multiple_of(8) || !dv.is_multiple_of(8) {
        return false;
    }
    if host.select > 2 {
        return true;
    }
    let m = op.q.ne[1];
    if m > 1 {
        return true;
    }
    let n_threads = host.n_threads;
    let n_head = op.q.ne[2];
    let n_kv_heads = op.k.ne[2];
    let g = n_head.checked_div(n_kv_heads).unwrap_or(1);
    let s = op.k.ne[1];
    let total_threads = n_threads.wrapping_mul(host.mdev_count);
    if dk > 256 || (dk == 256 && (dv > 256 || n_kv_heads < total_threads)) {
        return true;
    }
    let tile_aligned = g > 0 && 32 % g == 0;
    if !tile_aligned {
        if n_threads >= 6 {
            return false;
        }
        return s >= 1024;
    }
    let s_cross = if dk == 256 {
        4096
    } else if dk <= 64 {
        if n_threads >= 8 {
            2048
        } else if n_threads >= 6 {
            768
        } else {
            512
        }
    } else if n_threads >= 8 {
        1024
    } else {
        512
    };
    s >= s_cross
}

/// `{mp, l}` for a divisor the solver takes from tensor extents. llama.cpp divides by it without a
/// check (a zero is undefined behaviour there); here it is an unsupported shape.
fn fa_fastdiv(d: u64) -> Option<FastDivValues> {
    // init_fastdiv_values takes a uint32_t: wider extents truncate
    let d = d as u32;
    (d != 0).then(|| init_fastdiv(d))
}

/// Port of `htp_kparams_solve_flash_attn`: `None` where the solver answers unsupported.
fn solve_flash_attn(host: &FlashAttnHost, op: &FlashAttnOp<'_>) -> Option<FaKernelParams> {
    use HtpDataType as T;
    if host.select < 1 {
        return None;
    }
    let (q, k, v, dst) = (op.q, op.k, op.v, op.dst);
    let is = |t: &CopyTensor, ty: T| t.dtype == ty as u32;
    if !(is(q, T::F16) || is(q, T::F32))
        || !(is(k, T::F16) || is(k, T::Q8_0))
        || !(is(v, T::F16) || is(v, T::Q8_0))
    {
        return None;
    }
    if op.mask.is_some_and(|m| !is(m, T::F16)) {
        return None;
    }
    if op.sinks.is_some_and(|s| !is(s, T::F32)) {
        return None;
    }
    if !(is(dst, T::F32) || is(dst, T::F16)) {
        return None;
    }

    let n_threads = host.n_threads;
    let q_is_f32 = is(q, T::F32);
    let (neq0, neq1, neq2) = (q.ne[0], q.ne[1], q.ne[2]);
    let nek1 = k.ne[1];
    let (dk, dv) = (neq0, v.ne[0]);
    let n_kv_heads = k.ne[2];
    // llama.cpp divides by the KV head count and a zero GQA factor reaches fastdiv(0)
    if n_kv_heads == 0 || neq2 / n_kv_heads == 0 {
        return None;
    }
    let g = neq2 / n_kv_heads;
    let n_head = neq2;

    // the host folds the soft cap into the scale: the DSP computes tanh(scale * s) * softcap
    let mut scale = op.scale;
    if op.logit_softcap != 0.0 {
        scale /= op.logit_softcap;
    }

    let mut kp = FaKernelParams {
        kernel_type: 0,
        head_split: u8::from(host.split),
        flags: 0,
        n_threads: 0,
        br: 0,
        bc: 0,
        n_kv_blocks: 0,
        g: g as u16,
        scale,
        max_bias: op.max_bias,
        logit_softcap: op.logit_softcap,
        vtcm_size: 0,
        m0: 0.0,
        m1: 0.0,
        n_head_log2: 0,
        src3_div2: FastDivValues::default(),
        src3_div3: FastDivValues::default(),
        broadcast_rk2: FastDivValues::default(),
        broadcast_rk3: FastDivValues::default(),
        broadcast_rv2: FastDivValues::default(),
        broadcast_rv3: FastDivValues::default(),
        tail: FaKernelUnion::Hmx {
            g_br: 0,
            mask_broadcast: 0,
            pipeline: 0,
            div_g: FastDivValues::default(),
        },
    };

    // team-split only when a team has more than one member (n_kv_heads < the device count)
    let mut fa_team_size = 1usize;
    if host.split && host.mdev_count > 1 {
        let teams = fa_gcd(n_kv_heads, host.mdev_count);
        fa_team_size = (host.mdev_count / teams) as usize;
        if fa_team_size > 1 {
            kp.flags |= FA_FLAG_TEAM_SPLIT;
        }
    }

    // ALiBi slopes: m0 for the first n_head_log2 heads, m1 for the rest
    kp.n_head_log2 = 1u32 << (31 - n_head.leading_zeros());
    let log2_heads = kp.n_head_log2 as f32;
    kp.m0 = 2.0f32.powf(-op.max_bias / log2_heads);
    kp.m1 = 2.0f32.powf(-(op.max_bias / 2.0) / log2_heads);

    let has_sinks = op.sinks.is_some();
    if fa_is_hmx_eligible(host, op) {
        let dk_pad = fa_round_up(dk, 64);
        let dv_pad = fa_round_up(dv, 64);
        if let Some((br, bc)) = fa_hmx_find_chunk_size(
            g as usize,
            dk_pad as usize,
            dv_pad as usize,
            neq1 as usize,
            nek1 as usize,
            host.vtcm_bytes,
            n_threads as usize,
            q_is_f32,
            has_sinks,
            n_head as usize,
            fa_team_size,
        ) {
            use std::f32::consts::LOG2_E;
            kp.kernel_type = FA_KERNEL_HMX;
            // the HMX kernel works in the log2 domain: log2(e) goes into the scale, or into the
            // soft cap when there is one
            if op.logit_softcap == 0.0 {
                kp.scale = scale * LOG2_E;
                kp.logit_softcap = 0.0;
            } else {
                kp.scale = scale;
                kp.logit_softcap = op.logit_softcap * LOG2_E;
            }
            kp.br = br as u16;
            kp.bc = bc as u16;
            kp.n_kv_blocks = (nek1 as usize).div_ceil(bc) as u16;
            let pipeline = kp.n_kv_blocks >= 3 && n_threads >= 2;
            kp.n_threads = if pipeline { n_threads as u8 } else { 1 };
            kp.vtcm_size = fa_hmx_layout_total(
                g as usize,
                dk_pad as usize,
                dv_pad as usize,
                br,
                bc,
                usize::from(kp.n_threads),
                pipeline,
                q_is_f32,
                has_sinks,
                n_head as usize,
            ) as u32;
            if let Some(m) = op.mask {
                kp.src3_div2 = fa_fastdiv(u64::from(m.ne[2]))?;
                kp.src3_div3 = fa_fastdiv(u64::from(m.ne[3]))?;
            }
            kp.tail = FaKernelUnion::Hmx {
                g_br: (g as usize * br).next_multiple_of(FA_TILE_ROWS) as u32,
                mask_broadcast: i32::from(op.mask.is_some_and(|m| m.ne[2] == 1)),
                pipeline: i32::from(pipeline),
                div_g: fa_fastdiv(u64::from(g))?,
            };
            return Some(kp);
        }
    }

    kp.kernel_type = FA_KERNEL_HVX;
    kp.br = 1;
    kp.bc = FA_HVX_BLOCK_SIZE as u16;
    kp.n_kv_blocks = (u64::from(k.ne[1]).div_ceil(FA_HVX_BLOCK_SIZE as u64)) as u16;
    kp.n_threads = n_threads as u8;

    kp.vtcm_size = fa_hvx_layout_total(
        dk as usize,
        dv as usize,
        q_is_f32,
        op.mask.is_some(),
        has_sinks,
        n_head as usize,
        n_threads as usize,
    ) as u32;
    if kp.vtcm_size as usize > host.vtcm_bytes {
        return None;
    }

    let q_row = u64::from(q.ne[0]) * if q_is_f32 { 4 } else { 2 };
    kp.tail = FaKernelUnion::Hvx {
        size_q_row_padded: fa_round_up(q_row as u32, 128),
        size_k_row_padded: fa_round_up((u64::from(k.ne[0]) * 2) as u32, 128),
        size_v_row_padded: fa_round_up((u64::from(v.ne[0]) * 2) as u32, 128),
        src0_div21: fa_fastdiv(u64::from(q.ne[2]) * u64::from(q.ne[1]))?,
        src0_div1: fa_fastdiv(u64::from(q.ne[1]))?,
    };
    kp.broadcast_rk2 = fa_fastdiv(u64::from(q.ne[2] / k.ne[2]))?;
    kp.broadcast_rk3 = fa_fastdiv(u64::from(q.ne[3].checked_div(k.ne[3])?))?;
    kp.broadcast_rv2 = fa_fastdiv(u64::from(q.ne[2].checked_div(v.ne[2])?))?;
    kp.broadcast_rv3 = fa_fastdiv(u64::from(q.ne[3].checked_div(v.ne[3])?))?;
    if let Some(m) = op.mask {
        kp.src3_div2 = fa_fastdiv(u64::from(m.ne[2]))?;
        kp.src3_div3 = fa_fastdiv(u64::from(m.ne[3]))?;
    }
    Some(kp)
}

/// Kernel parameters for a `FlashAttnExt`: port of `htp_kparams_solve_flash_attn`
/// (`htp-kparams.cpp`) with `hmx_fa_find_chunk_size`, `hmx_fa_vtcm_layout_build` and
/// `hvx_fa_vtcm_layout_build` (`flash-attn-ops.h`). `None` when the kernels do not take the
/// operands or the HVX working set does not fit the VTCM budget; the HMX kernel is tried first
/// when it is eligible and falls back to HVX when no tiling fits.
///
/// Words (`htp_fa_kernel_params`): 0 packs `kernel_type`, `head_split`, `flags`, `n_threads`;
/// 1 `Br | Bc << 16`; 2 `n_kv_blocks | G << 16`; 3..=5 scale, max_bias, logit soft cap (f32 bits);
/// 6 `vtcm_size`; 7, 8, 9 the ALiBi `m0`, `m1`, `n_head_log2`; 10..=21 six `{mp, l}` dividers (mask
/// dims 2 and 3, then the Q:K and Q:V broadcast ratios); 22.. the kernel-specific tail. The
/// query-row split is the DSP's own and no longer a host word.
pub fn build_flash_attn_kernel_params(
    host: &FlashAttnHost,
    op: &FlashAttnOp<'_>,
) -> Option<[i32; 32]> {
    solve_flash_attn(host, op).map(|k| k.to_words())
}

/// Kernel parameters for a `Softmax` over F32 rows: port of `htp_kparams_solve_softmax`
/// (`htp-kparams.cpp`) and `htp_softmax_vtcm_layout_build` (`softmax-ops.h`).
/// `dst = softmax(scale * src0 + slope * mask)` along dim 0, with `scale` and `max_bias` the first
/// two words of `op_params` (as floats). The optional F32 or F16 `mask` has `src0`'s row width, at
/// least its row count, and `ne2`/`ne3` that divide `src0`'s (it broadcasts over heads and batches).
/// A positive `max_bias` adds ALiBi slopes `m0`/`m1` per head.
///
/// `None` where the solver refuses: non-F32 `src0` or `dst`, more than 512 heads, a mask of the wrong
/// type or shape, or non-contiguous `src0`/`dst`/`mask` (attention sinks, the solver's `src2`, are
/// not representable here and are refused by the caller). The thread count is the session's capped
/// at one 128-byte output chunk per thread; there is no fallback to fewer threads when the working
/// set does not fit `vtcm_bytes`.
///
/// Words (`htp_softmax_kernel_params`): `kernel_type` (0 no mask, 1 F32 mask, 2 F16 mask),
/// `n_threads`, `vtcm_size`, `n_head`, `n_head_log2`, `use_src1`, `use_f16`, `scale`, `max_bias`,
/// `m0`, `m1`, then the dividers `ne01`, `ne02` of `src0` and `ne12`, `ne13` of the mask (each `mp, l`).
pub fn build_softmax_kernel_params(
    src0: &CopyTensor,
    mask: Option<&CopyTensor>,
    dst: &CopyTensor,
    op_params: &[i32; 16],
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    softmax_solve(src0, mask, dst, op_params, n_threads, vtcm_bytes, false)
}

/// [`build_softmax_kernel_params`] for a mask that is a strided view: its rows only have to be dense.
///
/// The host solver demands a contiguous mask, but the DSP kernel (`softmax-ops.c`) fetches one mask row
/// per source row by DMA at `i1 * nb11 + i12 * nb12 + i13 * nb13`, so any row strides work. The audio
/// encoder's relative-position shift is such a view (rows one element shorter than the real stride) and
/// feeds it in place rather than copying it. Everything else is the host's answer.
pub fn build_softmax_kernel_params_strided_mask(
    src0: &CopyTensor,
    mask: Option<&CopyTensor>,
    dst: &CopyTensor,
    op_params: &[i32; 16],
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    softmax_solve(src0, mask, dst, op_params, n_threads, vtcm_bytes, true)
}

fn softmax_solve(
    src0: &CopyTensor,
    mask: Option<&CopyTensor>,
    dst: &CopyTensor,
    op_params: &[i32; 16],
    n_threads: u32,
    vtcm_bytes: usize,
    strided_mask: bool,
) -> Option<[i32; 32]> {
    use HtpDataType as T;
    const KERNEL_NOMASK: i32 = 0;
    const KERNEL_MASK_F32: i32 = 1;
    const KERNEL_MASK_F16: i32 = 2;
    if src0.dtype != T::F32 as u32 || dst.dtype != T::F32 as u32 || src0.ne[2] > 512 {
        return None;
    }
    if let Some(m) = mask {
        if m.dtype != T::F32 as u32 && m.dtype != T::F16 as u32 {
            return None;
        }
        // `ne2`/`ne3` of a mask must divide the source's; a zero count divides nothing
        if m.ne[2] == 0 || m.ne[3] == 0 {
            return None;
        }
        let rows_ok = if strided_mask {
            ggml_is_contiguous_rows(m)
        } else {
            ggml_is_contiguous(m)
        };
        if src0.ne[0] != m.ne[0]
            || m.ne[1] < src0.ne[1]
            || !src0.ne[2].is_multiple_of(m.ne[2])
            || !src0.ne[3].is_multiple_of(m.ne[3])
            || !rows_ok
        {
            return None;
        }
    }
    if !ggml_is_contiguous(src0) || !ggml_is_contiguous(dst) {
        return None;
    }

    let src0_nrows = rows_u32(&src0.ne[1..]);
    let dst_row_size = u64::from(dst.ne[0]) * 4;
    if dst_row_size == 0 {
        return None; // the C++ divides by it
    }
    let min_rows_per_thread = 128u64.div_ceil(dst_row_size).max(1) as u32;
    let max_threads = (src0_nrows / min_rows_per_thread).max(1);
    let threads = n_threads.min(max_threads);

    let scale = f32::from_bits(op_params[0] as u32);
    let max_bias = f32::from_bits(op_params[1] as u32);

    let n_head = src0.ne[2];
    // `1u << (uint32_t) floor(log2(n_head))`; an empty head dim reads as 1
    let n_head_log2 = if n_head == 0 {
        1
    } else {
        1u32 << n_head.ilog2()
    };
    let (m0, m1) = if max_bias > 0.0 && n_head_log2 > 0 {
        let log2 = n_head_log2 as f32;
        (
            2.0f32.powf(-max_bias / log2),
            2.0f32.powf(-(max_bias / 2.0) / log2),
        )
    } else {
        (1.0, 1.0)
    };

    let use_src1 = mask.is_some();
    let use_f16 = mask.is_some_and(|m| m.dtype == T::F16 as u32);
    // htp_softmax_vtcm_layout_build: double-buffered rows, 128-aligned, per thread
    let ne10 = mask.map_or(1, |m| m.ne[0]);
    let aligned = |bytes: u64| u64::from(hex_round_up(bytes as u32, 128));
    let src0_row = aligned(u64::from(src0.ne[0]) * 4);
    let dst_row = aligned(u64::from(src0.ne[0]) * 4);
    let src1_row = if use_src1 {
        aligned(u64::from(ne10) * if use_f16 { 2 } else { 4 })
    } else {
        0
    };
    let total = (2 * src0_row + 2 * dst_row + 2 * src1_row) * u64::from(threads);
    if total > vtcm_bytes as u64 {
        return None;
    }

    let kernel = match (use_src1, use_f16) {
        (false, _) => KERNEL_NOMASK,
        (true, true) => KERNEL_MASK_F16,
        (true, false) => KERNEL_MASK_F32,
    };
    let div = |d: u32| if d > 0 { fastdiv_words(d) } else { [0, 0] };
    let mut k = [0i32; 32];
    k[0] = kernel;
    k[1] = threads as i32;
    k[2] = total as u32 as i32;
    k[3] = n_head as i32;
    k[4] = n_head_log2 as i32;
    k[5] = i32::from(use_src1);
    k[6] = i32::from(use_f16);
    k[7] = scale.to_bits() as i32;
    k[8] = max_bias.to_bits() as i32;
    k[9] = m0.to_bits() as i32;
    k[10] = m1.to_bits() as i32;
    let (ne12, ne13) = mask.map_or((1, 1), |m| (m.ne[2], m.ne[3]));
    for (i, d) in [src0.ne[1], src0.ne[2], ne12, ne13].into_iter().enumerate() {
        k[11 + 2 * i..13 + 2 * i].copy_from_slice(&div(d));
    }
    Some(k)
}

/// Kernel parameters for `Sum`, `SumRows` and `Argmax` (`opcode`) over a dense F32 `src0`: port of
/// `htp_kparams_solve_sum_rows`. `dst` is F32, or I32 for `Argmax`, and dense. `None` for any other
/// operand types or strides, for another opcode, and for a zero-thread session where the C++ would
/// divide by zero.
///
/// A `Sum` (and an `Argmax` of a single row) splits its elements across up to 10 threads once there
/// are 1024 of them, in 32-aligned slices; other shapes split rows, at least 32 rows per thread
/// while a row is under 1 KiB.
///
/// Words (`htp_sum_rows_kernel_params`): `n_threads`, `elems_per_thread`.
pub fn build_sum_rows_kernel_params(
    opcode: u32,
    src0: &CopyTensor,
    dst: &CopyTensor,
    n_threads: u32,
) -> Option<[i32; 32]> {
    use HtpDataType as T;
    /// `HTP_MAX_NTHREADS`
    const MAX_NTHREADS: u32 = 10;
    let is_sum = opcode == HtpOpCode::Sum as u32;
    let is_argmax = opcode == HtpOpCode::Argmax as u32;
    if !is_sum && !is_argmax && opcode != HtpOpCode::SumRows as u32 {
        return None;
    }
    if src0.dtype != T::F32 as u32 {
        return None;
    }
    let dst_want = if is_argmax { T::I32 } else { T::F32 };
    if dst.dtype != dst_want as u32 {
        return None;
    }
    if !ggml_is_contiguous(src0) || !ggml_is_contiguous(dst) {
        return None;
    }

    let ne00 = src0.ne[0];
    let src0_nrows = rows_u32(&src0.ne[1..]);
    // split `n` elements into 32-aligned slices over threads
    let slices = |n: u32| -> Option<(u32, u32)> {
        let threads = if n >= 1024 {
            n_threads.min(MAX_NTHREADS)
        } else {
            1
        };
        let chunk = n
            .wrapping_add(threads.wrapping_sub(1))
            .checked_div(threads)?;
        Some((threads, hex_round_up(chunk, 32)))
    };
    let (threads, elems) = if is_sum {
        slices(rows_u32(&src0.ne))?
    } else if is_argmax && src0_nrows <= 1 {
        slices(ne00)?
    } else {
        let min_rows_per_thread = if u64::from(src0.nb[1]) < 1024 { 32 } else { 1 };
        let max_threads = (src0_nrows / min_rows_per_thread).max(1);
        (n_threads.min(max_threads), 0)
    };
    let mut k = [0i32; 32];
    k[0] = threads as i32;
    k[1] = elems as i32;
    Some(k)
}

/// `enum ggml_sort_order`: what `Argsort` reads from its first op param.
const SORT_ORDER_DESC: u32 = 1;

/// `htp_sort_vtcm_layout_build`: the working-set layout of `Argsort`/`TopK`, as the solver
/// searches it.
#[derive(Debug, Clone, Copy, Default)]
struct SortLayout {
    total_bytes: u64,
    phase1_slot_size: u64,
    chunk_elems: u32,
    n_chunks: u32,
    merge_elems: u32,
    n_threads: u32,
    n_slots: u32,
}

impl SortLayout {
    fn build(
        total_rows: u32,
        chunk_elems: u32,
        n_chunks: u32,
        merge_elems: u32,
        n_threads: u32,
        n_slots: u32,
    ) -> Self {
        let align = |x: u64, to: u64| (x + to - 1) & !(to - 1);
        let values_size = align(u64::from(chunk_elems) * 4, 128);
        let indices_size = align(u64::from(chunk_elems) * 4, 128);
        let phase1_slot_size = align(values_size + indices_size, 256);
        let mut total = if total_rows > 1 {
            phase1_slot_size * u64::from(n_slots) * u64::from(n_threads)
        } else {
            phase1_slot_size * u64::from(n_chunks) * u64::from(n_slots)
        };
        if total_rows <= 1 && n_chunks > 1 {
            // the merge buffers: values, then indices
            total += 2 * align(u64::from(merge_elems) * 4, 128);
        }
        Self {
            total_bytes: total,
            phase1_slot_size,
            chunk_elems,
            n_chunks,
            merge_elems,
            n_threads,
            n_slots,
        }
    }

    /// `htp_sort_solve_layout`: `None` when even the smallest layout exceeds `budget`.
    fn solve(
        ne00: u32,
        total_rows: u32,
        k: u32,
        n_threads_max: u32,
        budget: u64,
        is_top_k: bool,
    ) -> Option<Self> {
        let pow2_vecs = |n: u32| -> u32 {
            let n_vec = n.wrapping_add(31) / 32;
            let mut p = 1u32;
            while p < n_vec {
                p <<= 1;
            }
            p
        };
        let align = |x: u64, to: u64| (x + to - 1) & !(to - 1);
        let n_vec_pow2 = pow2_vecs(ne00);

        if total_rows > 1 {
            let mut n_threads = total_rows.min(n_threads_max);
            let ne00_padded = n_vec_pow2.wrapping_mul(32);
            let values_size = align(u64::from(ne00_padded) * 4, 128);
            let indices_size = align(u64::from(ne00_padded) * 4, 128);
            let spad_per_slot = align(values_size + indices_size, 256);
            let n_slots = if spad_per_slot * 2 > budget { 1 } else { 2 };
            let spad_per_thread = spad_per_slot * u64::from(n_slots);
            while n_threads > 1 && spad_per_thread * u64::from(n_threads) > budget {
                n_threads -= 1;
            }
            let layout = Self::build(total_rows, ne00_padded, 1, 0, n_threads, n_slots);
            return (layout.total_bytes <= budget).then_some(layout);
        }

        // one row: split it across threads in power-of-two chunks, merge on one thread
        let mut n_chunks = 1u32;
        if ne00 > 1024 {
            while n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2 {
                n_chunks *= 2;
            }
        }
        let mut chunk_elems = (n_vec_pow2 / n_chunks) * 32;
        let mut merge_elems = 0u32;
        if n_chunks > 1 {
            merge_elems = if is_top_k {
                let local_k = k.min(chunk_elems);
                pow2_vecs(n_chunks.wrapping_mul(local_k)).wrapping_mul(32)
            } else {
                n_vec_pow2.wrapping_mul(32)
            };
        }
        let mut layout = Self::build(1, chunk_elems, n_chunks, merge_elems, n_chunks, 1);
        if layout.total_bytes > budget && n_chunks > 1 {
            n_chunks = 1;
            chunk_elems = n_vec_pow2.wrapping_mul(32);
            merge_elems = 0;
            layout = Self::build(1, chunk_elems, n_chunks, merge_elems, 1, 1);
        }
        if layout.total_bytes > budget {
            return None;
        }
        if n_chunks == 1 && layout.phase1_slot_size * 2 <= budget {
            layout = Self::build(1, chunk_elems, 1, 0, 1, 2);
        }
        Some(layout)
    }
}

/// Kernel parameters for `Argsort` or `TopK` (`opcode`) of dense-or-strided F32 rows into I32 indices:
/// port of `htp_kparams_solve_sort` and `htp_sort_solve_layout` (`argsort-ops.h`). `op_params[0]`
/// is the sort order (an `Argsort` only; a `TopK` is always descending); `k` is `dst.ne[0]`.
/// `None` for other operand types, another opcode, or when even the smallest layout does not fit
/// `vtcm_bytes`. A session of 0 threads or 0 VTCM bytes is read as the host's defaults (4 threads,
/// 8 MiB).
///
/// Many rows: one row per thread (fewer threads when the per-thread scratch will not fit, one slot
/// instead of two when two will not). One long row: up to one power-of-two chunk per thread, sorted
/// separately and merged.
///
/// Words (`htp_sort_kernel_params`): `n_threads`, `order`, `is_top_k`, `chunk_elems`, `n_chunks`,
/// `vtcm_size`, `merge_elems`, `n_slots`, then the dividers `ne01` and `ne01 * ne02` (each `mp, l`).
pub fn build_sort_kernel_params(
    opcode: u32,
    src0: &CopyTensor,
    dst: &CopyTensor,
    op_params: &[i32; 16],
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    use HtpDataType as T;
    let is_top_k = opcode == HtpOpCode::TopK as u32;
    if !is_top_k && opcode != HtpOpCode::Argsort as u32 {
        return None;
    }
    if src0.dtype != T::F32 as u32 || dst.dtype != T::I32 as u32 {
        return None;
    }
    let total_rows = rows_u32(&src0.ne[1..]);
    let n_threads_max = if n_threads > 0 { n_threads } else { 4 };
    let budget = if vtcm_bytes > 0 {
        vtcm_bytes as u64
    } else {
        8 * 1024 * 1024
    };
    let order = if is_top_k {
        SORT_ORDER_DESC
    } else {
        op_params[0] as u32
    };
    let layout = SortLayout::solve(
        src0.ne[0],
        total_rows,
        dst.ne[0],
        n_threads_max,
        budget,
        is_top_k,
    )?;

    let mut k = [0i32; 32];
    k[0] = layout.n_threads as i32;
    k[1] = order as i32;
    k[2] = i32::from(is_top_k);
    k[3] = layout.chunk_elems as i32;
    k[4] = layout.n_chunks as i32;
    k[5] = layout.total_bytes as u32 as i32;
    k[6] = layout.merge_elems as i32;
    k[7] = layout.n_slots as i32;
    k[8..10].copy_from_slice(&fastdiv_words(src0.ne[1]));
    k[10..12].copy_from_slice(&fastdiv_words(
        (u64::from(src0.ne[1]) * u64::from(src0.ne[2])) as u32,
    ));
    Some(k)
}

/// One side of a `Cpy`, as `ggml_hexagon_precompute_cpy_params` sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyTensor {
    /// `HtpDataType` value; only F32, F16 and I32 are copyable.
    pub dtype: u32,
    pub ne: [u32; 4],
    /// Byte strides.
    pub nb: [u32; 4],
}

impl From<&super::types::HtpTensor> for CopyTensor {
    fn from(t: &super::types::HtpTensor) -> Self {
        Self {
            dtype: t.dtype,
            ne: t.ne,
            nb: t.nb,
        }
    }
}

impl CopyTensor {
    fn type_size(&self) -> Option<u64> {
        match self.dtype {
            x if x == HtpDataType::F32 as u32 || x == HtpDataType::I32 as u32 => Some(4),
            x if x == HtpDataType::F16 as u32 => Some(2),
            _ => None,
        }
    }

    fn nelements(&self) -> Option<u64> {
        self.ne
            .iter()
            .try_fold(1u64, |acc, &n| acc.checked_mul(u64::from(n)))
    }

    /// `ggml_is_contiguous` for a type with a block size of 1.
    fn is_contiguous(&self, type_size: u64) -> bool {
        if self.ne[0] != 1 && u64::from(self.nb[0]) != type_size {
            return false;
        }
        let mut next_nb = type_size * u64::from(self.ne[0]);
        for i in 1..4 {
            if self.ne[i] != 1 && u64::from(self.nb[i]) != next_nb {
                return false;
            }
            next_nb *= u64::from(self.ne[i]);
        }
        true
    }
}

/// `htp_gather_rows_tensor_outer_contiguous` (`gather-rows.h`): dims 2 and 3 follow dim 1 with no
/// gap, so the tensor is a run of rows of `nb[1]` bytes.
fn gather_rows_outer_contiguous(t: &CopyTensor) -> bool {
    let mut next_nb = u64::from(t.nb[1]).wrapping_mul(u64::from(t.ne[1]));
    for i in 2..4 {
        if t.ne[i] != 1 && u64::from(t.nb[i]) != next_nb {
            return false;
        }
        next_nb = next_nb.wrapping_mul(u64::from(t.ne[i]));
    }
    true
}

/// `htp_gather_rows_params` (`gather-rows.h`): the plan of the gather-rows kernel, which joins
/// short rows (`ne` elements) into vectors and moves a whole run of them per DMA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GatherRowsPlan {
    na: u32,
    nb: u32,
    ne: u32,
    rg: u32,
    vg: u32,
    nr_max: u32,
    v1: u32,
    span1: u32,
    region_size: u32,
    slice: u32,
    total_rows: u32,
    rows_per_thread: u32,
    b_dense: bool,
}

impl GatherRowsPlan {
    fn write(&self, out: &mut [i32]) {
        let words = [
            self.na,
            self.nb,
            self.ne,
            self.rg,
            self.vg,
            self.nr_max,
            self.v1,
            self.span1,
            self.region_size,
            self.slice,
            self.total_rows,
            self.rows_per_thread,
            u32::from(self.b_dense),
        ];
        for (o, w) in out.iter_mut().zip(words) {
            *o = w as i32;
        }
    }
}

/// Bytes one thread's slice must hold for a gather plan: port of
/// `htp_gather_rows_vtcm_layout_build`, in the header's mixed 32/64-bit arithmetic.
fn gather_rows_slice_bytes(vg: u32, region_size: u32, nr_max: u32, ne: u32, elem_size: u32) -> u64 {
    let vec_bytes = u64::from(vg) * 128;
    let region_aligned = u64::from(hex_round_up(region_size, 128));
    let out_end =
        region_aligned.wrapping_add(u64::from(nr_max) * u64::from(ne) * u64::from(elem_size));
    let out_aligned = u64::from(hex_round_up(out_end as u32, 128)).wrapping_sub(region_aligned);
    // lookup table, increment and cursor vectors, then two (region, output) buffer pairs
    (3 * vec_bytes)
        .wrapping_add(2 * region_aligned)
        .wrapping_add(2u64.wrapping_mul(out_aligned))
}

/// Port of `htp_gather_rows_solve_layout` (`gather-rows.h`), shared by the gather variants of
/// `Cpy` and `Concat`: `a` rows of `na` elements (stride `a_nb1`) optionally followed by `b` rows
/// of `nb` (`b_dense`: rows of `b_nb1` bytes; otherwise `b` is read column by column), written
/// as rows of `na + nb`. `None` when the rows are too wide or no run of them fits `vtcm_size`
/// split over `n_threads`. A zero thread count or device count divides by zero in the C and
/// never reaches it from a session, so it is a refusal here.
#[allow(clippy::too_many_arguments)]
fn gather_rows_solve_layout(
    na: u32,
    a_nb1: u32,
    nb: u32,
    b_nb1: u32,
    b_dense: bool,
    elem_size: u32,
    total_rows: u32,
    n_threads: u32,
    mdev_count: u32,
    vtcm_size: u32,
) -> Option<GatherRowsPlan> {
    if elem_size != 4 && elem_size != 2 {
        return None;
    }
    let ne = na.wrapping_add(nb);
    if ne == 0 || ne > 128 / elem_size || n_threads == 0 || mdev_count == 0 {
        return None;
    }

    let v_lanes = 128 / elem_size;
    let (mut x, mut y) = (ne, v_lanes);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    let rg = v_lanes / x;
    let vg = ne / x;

    let slice = (vtcm_size / n_threads) & !127u32;
    let total_chunks = total_rows / rg;
    let chunks_per_dev = total_chunks.wrapping_add(mdev_count).wrapping_sub(1) / mdev_count;
    let max_rows_per_dev = if mdev_count > 1 && total_chunks >= mdev_count {
        chunks_per_dev
            .wrapping_mul(rg)
            .wrapping_add(total_rows % rg)
    } else {
        total_rows
    };
    let rows_per_thread = hex_round_up(
        max_rows_per_dev.wrapping_add(n_threads).wrapping_sub(1) / n_threads,
        rg,
    );

    let fixed = 3u32.wrapping_mul(vg).wrapping_mul(128);
    // bytes of source rows one output row needs: the `a` row plus the `b` row (or its columns)
    let region_row = a_nb1.wrapping_add(if nb == 0 {
        0
    } else if b_dense {
        b_nb1
    } else {
        nb.wrapping_mul(b_nb1)
    });
    let per_row = region_row.wrapping_add(ne.wrapping_mul(elem_size));
    if slice <= fixed.wrapping_add(2u32.wrapping_mul(per_row).wrapping_mul(rg)) {
        return None;
    }

    let avail_per_buf = slice.wrapping_sub(fixed) / 2;
    let mut nr_max = avail_per_buf.checked_div(per_row)?.min(rows_per_thread);
    if elem_size == 2 && region_row > 0 {
        // f16 gathers address VTCM with 16-bit offsets, so a region stays under 32 KiB
        nr_max = nr_max.min((32768 - 256) / region_row);
    }
    nr_max = (nr_max / rg) * rg;

    let (mut v1, mut span1, mut region_size) = (0u32, 0u32, 0u32);
    while nr_max > 0 {
        v1 = hex_round_up(
            (nr_max - 1)
                .wrapping_mul(a_nb1)
                .wrapping_add(na.wrapping_mul(elem_size)),
            128,
        );
        if nb == 0 {
            span1 = 0;
            region_size = v1;
        } else if b_dense {
            span1 = elem_size;
            region_size = v1
                .wrapping_add((nr_max - 1).wrapping_mul(b_nb1))
                .wrapping_add(nb.wrapping_mul(elem_size));
        } else {
            span1 = (nr_max - 1).wrapping_mul(b_nb1).wrapping_add(elem_size);
            region_size = v1.wrapping_add(nb.wrapping_mul(span1));
        }
        let total = gather_rows_slice_bytes(vg, region_size, nr_max, ne, elem_size);
        if (elem_size != 2 || region_size <= 32768) && total <= u64::from(slice) {
            break;
        }
        nr_max -= rg;
    }
    if nr_max == 0 {
        return None;
    }

    Some(GatherRowsPlan {
        na,
        nb,
        ne,
        rg,
        vg,
        nr_max,
        v1,
        span1,
        region_size,
        slice,
        total_rows,
        rows_per_thread,
        b_dense,
    })
}

/// Words per `htp_copy_transpose_params` in the blob.
const COPY_TRANSPOSE_WORDS: usize = 25;
/// Words before the kernel union of `htp_copy_kernel_params`.
const COPY_HEADER_WORDS: usize = 4;

const COPY_KERNEL_1D_CONTIG: u32 = 1;
const COPY_KERNEL_SAMESHAPE_SAMETYPE: u32 = 2;
const COPY_KERNEL_SAMESHAPE_CONVERT: u32 = 3;
const COPY_KERNEL_RESHAPE: u32 = 4;
const COPY_KERNEL_SCALAR: u32 = 5;
const COPY_KERNEL_GATHER_ROWS: u32 = 6;
const COPY_KERNEL_TRANSPOSE: u32 = 7;

/// Bytes of one VTCM staging buffer of the convert copy (`HTP_COPY_VTCM_BLK_BYTES`).
const COPY_VTCM_BLK_BYTES: u32 = 32 * 1024;
/// Largest staging buffer of the transposing copy (`HTP_COPY_TRANSPOSE_BLK_BYTES`).
const COPY_TRANSPOSE_BLK_BYTES: u32 = 64 * 1024;

/// `htp_copy_transpose_params` (`cpy-ops.h`): a batched 2-D transpose of `elem_size`-byte
/// elements, dim 0 dense in the destination and dim 1 dense in the source, tiled through VTCM.
#[derive(Debug, Clone, Copy, Default)]
struct CopyTranspose {
    ne: [u32; 3],
    src_nb: [u32; 3],
    dst_nb: [u32; 3],
    div_nkb: FastDivValues,
    div_n0b: FastDivValues,
    div_ne2: FastDivValues,
    blk0: u32,
    blkk: u32,
    n0b: u32,
    nkb: u32,
    in_stride: u32,
    out_stride: u32,
    in_buf_size: u32,
    out_buf_size: u32,
    total_tiles: u32,
    elem_size: u8,
    packed: bool,
    log2_m: u8,
    log2_r: u8,
}

impl CopyTranspose {
    fn write(&self, out: &mut [i32]) {
        let words: [u32; COPY_TRANSPOSE_WORDS] = [
            self.ne[0],
            self.ne[1],
            self.ne[2],
            self.src_nb[0],
            self.src_nb[1],
            self.src_nb[2],
            self.dst_nb[0],
            self.dst_nb[1],
            self.dst_nb[2],
            self.div_nkb.mp,
            self.div_nkb.l,
            self.div_n0b.mp,
            self.div_n0b.l,
            self.div_ne2.mp,
            self.div_ne2.l,
            self.blk0,
            self.blkk,
            self.n0b,
            self.nkb,
            self.in_stride,
            self.out_stride,
            self.in_buf_size,
            self.out_buf_size,
            self.total_tiles,
            u32::from_le_bytes([
                self.elem_size,
                u8::from(self.packed),
                self.log2_m,
                self.log2_r,
            ]),
        ];
        for (o, w) in out.iter_mut().zip(words) {
            *o = w as i32;
        }
    }

    /// Port of `htp_copy_transpose_layout_build`: the tile (`blk0` x `blkk` elements) for a
    /// staging buffer of `blk_bytes`, or `false` when not even one vector fits.
    fn layout_build(&mut self, blk_bytes: u32) -> bool {
        let (ne0, nek, es) = (self.ne[0], self.ne[1], u32::from(self.elem_size));
        let epv = 128 / es;
        let (blk0, blkk);
        if self.packed {
            // a vector holds epv / ne0 whole destination rows
            blk0 = ne0;
            let Some(mut k) = blk_bytes.checked_div(ne0.wrapping_mul(es)) else {
                return false;
            };
            if es == 2 {
                let cap = (16384u32.checked_div(ne0).unwrap_or(0)).wrapping_sub(64);
                if k > cap {
                    k = cap;
                }
            }
            if k > nek {
                k = nek;
            }
            if k == 0 {
                return false;
            }
            blkk = k;
        } else if nek <= epv {
            // a short contiguous dim is staged whole and dim 0 is expanded
            blkk = nek;
            let Some(mut b) = blk_bytes.checked_div(nek.wrapping_mul(es)) else {
                return false;
            };
            b -= b % epv;
            if b > hex_round_up(ne0, epv) {
                b = hex_round_up(ne0, epv);
            }
            if b < epv {
                return false;
            }
            blk0 = b;
        } else {
            let mut b0 = hex_round_up(ne0, epv);
            if b0 > 256 {
                b0 = 256;
            }
            let Some(mut k) = blk_bytes.checked_div(b0.wrapping_mul(es)) else {
                return false;
            };
            k -= k % epv;
            // f16 gather offsets are 16 bits wide
            let max_blkk = (if es == 2 { 512 } else { 4096 }) / es;
            if k > max_blkk {
                k = max_blkk;
            }
            if k > hex_round_up(nek, epv) {
                k = hex_round_up(nek, epv);
            }
            if k < epv {
                return false;
            }
            blk0 = b0;
            blkk = k;
        }

        self.blk0 = blk0;
        self.blkk = blkk;
        self.n0b = ne0.wrapping_add(blk0).wrapping_sub(1) / blk0;
        self.nkb = nek.wrapping_add(blkk).wrapping_sub(1) / blkk;
        // one element of row padding keeps the staged rows off one VTCM bank
        let pad = if es == 2 { 2 } else { 4 };
        self.in_stride = blkk.wrapping_mul(es).wrapping_add(pad);
        self.out_stride = blk0.wrapping_mul(es);
        // a packed gather overhangs by one vector
        self.in_buf_size = hex_round_up(
            blk0.wrapping_mul(self.in_stride)
                .wrapping_add(if self.packed { 128 } else { 0 }),
            128,
        );
        self.out_buf_size = hex_round_up(blkk.wrapping_mul(self.out_stride), 128);
        self.log2_m = if es == 2 {
            0
        } else {
            (es.trailing_zeros() - 2) as u8
        };
        self.log2_r = (if self.packed { ne0 } else { epv }).trailing_zeros() as u8;
        true
    }
}

/// A transposing `Cpy`: port of `htp_kparams_solve_cpy_transpose` (a permuted copy whose dim 0
/// is strided on one side, such as channel-first to channel-last). Returns the plan and the
/// VTCM bytes it takes, or `None` when the strides are not one of the transposes the kernel
/// tiles.
fn copy_transpose_plan(
    src0: &CopyTensor,
    dst: &CopyTensor,
    type_size: u64,
    n_threads: u32,
    vtcm_size: u64,
) -> Option<(CopyTranspose, u64)> {
    // a contiguous dst is viewed with the src shape, which is how a reshaping copy is handled
    let dst_cont = dst.is_contiguous(dst.type_size()?);
    if !dst_cont && src0.ne != dst.ne {
        return None;
    }

    // drop size-1 dims and merge dims adjacent in both tensors
    let (mut ne, mut nbs, mut nbd) = ([0u64; 4], [0u64; 4], [0u64; 4]);
    let mut dst_stride = type_size;
    let mut n = 0usize;
    for d in 0..4 {
        let sd = if dst_cont {
            dst_stride
        } else {
            u64::from(dst.nb[d])
        };
        dst_stride = dst_stride.wrapping_mul(u64::from(src0.ne[d]));
        if src0.ne[d] == 1 {
            continue;
        }
        if n > 0
            && u64::from(src0.nb[d]) == nbs[n - 1].wrapping_mul(ne[n - 1])
            && sd == nbd[n - 1].wrapping_mul(ne[n - 1])
        {
            ne[n - 1] = ne[n - 1].wrapping_mul(u64::from(src0.ne[d]));
            continue;
        }
        ne[n] = u64::from(src0.ne[d]);
        nbs[n] = u64::from(src0.nb[d]);
        nbd[n] = sd;
        n += 1;
    }

    // fold the leading dense dim into the element block
    let mut es = type_size;
    for i in 0..n {
        if nbs[i] == type_size && nbd[i] == type_size {
            es = es.wrapping_mul(ne[i]);
            for j in i..n - 1 {
                ne[j] = ne[j + 1];
                nbs[j] = nbs[j + 1];
                nbd[j] = nbd[j + 1];
            }
            n -= 1;
            break;
        }
    }
    if es > 128 || es & es.wrapping_sub(1) != 0 {
        return None;
    }

    // the transpose axes: `a` is dense in dst, `k` is dense in src
    let (mut a, mut k) = (None, None);
    for i in 0..n {
        if a.is_none() && nbd[i] == es {
            a = Some(i);
        }
        if k.is_none() && nbs[i] == es {
            k = Some(i);
        }
    }
    let (a, k) = (a?, k?);
    if a == k {
        return None;
    }

    let mut tp = CopyTranspose {
        elem_size: es as u8,
        ne: [ne[a] as u32, ne[k] as u32, 1],
        ..CopyTranspose::default()
    };
    tp.src_nb[0] = nbs[a] as u32;
    tp.dst_nb[0] = nbd[k] as u32;
    let row_bytes = ne[a].wrapping_mul(es);
    tp.packed = nbd[k] == row_bytes && row_bytes < 128 && row_bytes != 0 && 128 % row_bytes == 0;

    let mut ne3 = 1u64;
    let mut m = 0usize;
    for i in 0..n {
        if i == a || i == k {
            continue;
        }
        if m == 0 {
            tp.ne[2] = ne[i] as u32;
        } else {
            ne3 = ne[i];
        }
        tp.src_nb[1 + m] = nbs[i] as u32;
        tp.dst_nb[1 + m] = nbd[i] as u32;
        m += 1;
    }

    // halve the staging buffers until every thread's double-buffered tile pair fits
    let mut blk_bytes = COPY_TRANSPOSE_BLK_BYTES;
    let vtcm = loop {
        if !tp.layout_build(blk_bytes) {
            return None;
        }
        let vtcm = u64::from(n_threads)
            .wrapping_mul(2)
            .wrapping_mul(u64::from(tp.in_buf_size.wrapping_add(tp.out_buf_size)));
        if vtcm <= vtcm_size {
            break vtcm;
        }
        blk_bytes /= 2;
    };

    let total_tiles = u64::from(tp.n0b)
        .wrapping_mul(u64::from(tp.nkb))
        .wrapping_mul(u64::from(tp.ne[2]))
        .wrapping_mul(ne3);
    if total_tiles > u64::from(u32::MAX) {
        return None;
    }
    tp.total_tiles = total_tiles as u32;
    tp.div_nkb = init_fastdiv(tp.nkb);
    tp.div_n0b = init_fastdiv(tp.n0b);
    tp.div_ne2 = init_fastdiv(tp.ne[2]);
    Some((tp, vtcm))
}

/// Kernel parameters for a `Cpy`: port of `htp_kparams_solve_cpy` (`htp-kparams.cpp`) with
/// `htp_copy_kernel_params` and its layout helpers (`cpy-ops.h`, `gather-rows.h`). The session is
/// one device (`mdev_count == 1`).
///
/// The DSP's `op_cpy` has no fallback: it dispatches on `kernel_type` and answers
/// `HTP_STATUS_NO_SUPPORT` for 0, so a copy sent with empty params fails. `None` means the
/// pair is not a copy the kernels handle (the C++ host runs such a node on the CPU).
/// `n_threads == 0` means "unknown" and takes the host's default of 4 where the kernel allows
/// it; `vtcm_bytes == 0` skips the VTCM fit check of the convert kernel, as the host does.
///
/// Words (`htp_copy_kernel_params`): 0 = `kernel_type | src0_type_size << 8 | dst_type_size << 16
/// | n_threads << 24`, 1 `total_elems`, 2 `total_rows`, 3 `vtcm_size`, then the union from word 4:
/// convert = `blk_rows, div_ne01, div_ne02_ne01`; reshape = `div_ne0, div_ne1_ne0,
/// div_ne2_ne1_ne0, div_ne00, div_ne01_ne00, div_ne02_ne01_ne00` (each divider is `mp, l`);
/// gather = a `GatherRowsPlan`; transpose = a `CopyTranspose` (both in their struct order).
pub fn build_copy_kernel_params(
    src: &CopyTensor,
    dst: &CopyTensor,
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    let src_ts = src.type_size()?;
    let dst_ts = dst.type_size()?;
    let nelem = src.nelements()?;
    if Some(nelem) != dst.nelements() {
        return None;
    }
    let pack = |kernel: u32, threads: u32| -> i32 {
        (kernel
            | ((src_ts as u32) << 8)
            | ((dst_ts as u32) << 16)
            | ((wire_byte(threads as usize) as u32) << 24)) as i32
    };
    let mut k = [0i32; 32];
    k[1] = nelem as u32 as i32; // total_elems

    if nelem == 0 {
        k[0] = pack(COPY_KERNEL_1D_CONTIG, 0);
        return Some(k);
    }
    if nelem == 1 {
        let (f32t, f16t, i32t) = (
            HtpDataType::F32 as u32,
            HtpDataType::F16 as u32,
            HtpDataType::I32 as u32,
        );
        let ok = src.dtype == dst.dtype
            || matches!(
                (src.dtype, dst.dtype),
                (a, b) if (a == f32t && b == i32t)
                    || (a == i32t && b == f32t)
                    || (a == f32t && b == f16t)
                    || (a == f16t && b == f32t)
            );
        if !ok {
            return None;
        }
        k[0] = pack(COPY_KERNEL_SCALAR, 0);
        return Some(k);
    }

    let sametype = src.dtype == dst.dtype;
    let same_extents = src.ne == dst.ne;
    let transposed = src.nb[0] > src.nb[1]
        || dst.nb[0] > dst.nb[1]
        || u64::from(src.nb[0]) != src_ts
        || u64::from(dst.nb[0]) != dst_ts
        || u64::from(src.nb[1]) < u64::from(src.ne[0]) * src_ts
        || u64::from(dst.nb[1]) < u64::from(dst.ne[0]) * dst_ts;
    let sameshape = same_extents && !transposed;
    let src_cont = src.is_contiguous(src_ts);
    let dst_cont = dst.is_contiguous(dst_ts);
    let rows = (u64::from(src.ne[1]) * u64::from(src.ne[2]) * u64::from(src.ne[3])) as u32;
    // the C narrows the byte budget to 32 bits wherever it hands it to a gather plan
    let vtcm_size = vtcm_bytes as u64;

    if sametype {
        if src_cont && dst_cont {
            k[0] = pack(COPY_KERNEL_1D_CONTIG, 0);
            return Some(k);
        }

        let (f32t, f16t) = (HtpDataType::F32 as u32, HtpDataType::F16 as u32);
        if (src.dtype == f32t || src.dtype == f16t)
            && dst_cont
            && !src_cont
            && u64::from(src.nb[0]) == src_ts
            && u64::from(src.ne[0]) <= 128 / src_ts
            && u64::from(src.nb[1]) >= u64::from(src.ne[0]) * src_ts
            && rows > 1
            && gather_rows_outer_contiguous(src)
            && let Some(plan) = gather_rows_solve_layout(
                src.ne[0],
                src.nb[1],
                0,
                0,
                false,
                src_ts as u32,
                rows,
                n_threads,
                1,
                vtcm_bytes as u32,
            )
        {
            k[0] = pack(COPY_KERNEL_GATHER_ROWS, n_threads);
            k[3] = plan.slice.wrapping_mul(n_threads) as i32;
            plan.write(&mut k[COPY_HEADER_WORDS..]);
            return Some(k);
        }

        if let Some((tp, vtcm)) = copy_transpose_plan(src, dst, src_ts, n_threads, vtcm_size) {
            k[0] = pack(COPY_KERNEL_TRANSPOSE, n_threads);
            k[3] = vtcm as u32 as i32;
            tp.write(&mut k[COPY_HEADER_WORDS..]);
            return Some(k);
        }

        if sameshape {
            k[0] = pack(COPY_KERNEL_SAMESHAPE_SAMETYPE, 0);
            k[2] = rows as i32;
            return Some(k);
        }
        k[0] = pack(
            COPY_KERNEL_RESHAPE,
            if n_threads > 0 { n_threads } else { 4 },
        );
        let prod = |t: &CopyTensor, upto: usize| -> u32 {
            t.ne[..upto]
                .iter()
                .fold(1u64, |acc, &v| acc.wrapping_mul(u64::from(v))) as u32
        };
        let dividers = [
            init_fastdiv(prod(dst, 1)),
            init_fastdiv(prod(dst, 2)),
            init_fastdiv(prod(dst, 3)),
            init_fastdiv(prod(src, 1)),
            init_fastdiv(prod(src, 2)),
            init_fastdiv(prod(src, 3)),
        ];
        for (i, v) in dividers.iter().enumerate() {
            k[COPY_HEADER_WORDS + 2 * i] = v.mp as i32;
            k[COPY_HEADER_WORDS + 1 + 2 * i] = v.l as i32;
        }
        return Some(k);
    }

    if !sameshape {
        return None;
    }
    let (f32t, f16t, i32t) = (
        HtpDataType::F32 as u32,
        HtpDataType::F16 as u32,
        HtpDataType::I32 as u32,
    );
    let valid = matches!(
        (src.dtype, dst.dtype),
        (a, b) if (a == f32t && b == f16t)
            || (a == f16t && b == f32t)
            || (a == f32t && b == i32t)
            || (a == i32t && b == f32t)
    );
    if !valid {
        return None;
    }

    // htp_copy_convert_solve_layout
    let nth = if n_threads > 0 { n_threads } else { 4 };
    let src_row_stride = hex_round_up((src.ne[0] as u64 * src_ts) as u32, 256);
    let dst_row_stride = hex_round_up((src.ne[0] as u64 * dst_ts) as u32, 256);
    let mut blk_rows = COPY_VTCM_BLK_BYTES
        .checked_div(src_row_stride.max(dst_row_stride))
        .unwrap_or(0);
    if blk_rows > src.ne[1] {
        blk_rows = src.ne[1];
    }
    if blk_rows == 0 {
        blk_rows = 1;
    }
    // htp_copy_convert_vtcm_layout_build: two buffers per thread on each side
    let spad0 = 2u32.wrapping_mul(blk_rows.wrapping_mul(src_row_stride));
    let spad1 = 2u32.wrapping_mul(blk_rows.wrapping_mul(dst_row_stride));
    let total = u64::from(nth) * u64::from(spad0) + u64::from(nth) * u64::from(spad1);
    if vtcm_size > 0 && total > vtcm_size {
        return None;
    }

    k[0] = pack(COPY_KERNEL_SAMESHAPE_CONVERT, nth);
    k[2] = rows as i32;
    k[3] = total as u32 as i32;
    k[COPY_HEADER_WORDS] = blk_rows as i32;
    let div_ne01 = init_fastdiv(src.ne[1]);
    let div_ne02_ne01 = init_fastdiv((u64::from(src.ne[2]) * u64::from(src.ne[1])) as u32);
    k[5] = div_ne01.mp as i32;
    k[6] = div_ne01.l as i32;
    k[7] = div_ne02_ne01.mp as i32;
    k[8] = div_ne02_ne01.l as i32;
    Some(k)
}

const CONCAT_KERNEL_REGULAR: u32 = 1;
const CONCAT_KERNEL_TRANSPOSED: u32 = 2;
const CONCAT_KERNEL_GATHER_ROWS: u32 = 3;

/// Kernel parameters for a `Concat` along `dim` (`op_params[0]`): port of
/// `htp_kparams_solve_concat` (`htp-kparams.cpp`) with `htp_concat_kernel_params`,
/// `htp_concat_transposed_vtcm_layout_build` (`concat-ops.h`) and the gather-rows plan
/// (`gather-rows.h`). The session is one device (`mdev_count == 1`). `None` when the operands
/// are not a concat the kernels handle.
///
/// Three kernels: the gather-rows one for a dim-0 join into short rows (such as a conv state
/// ahead of the new positions), the regular one when every operand is dense along dim 0 (a DMA
/// per row, one thread), and the transposed one for a transposed `src1` joined along dim 0.
///
/// Words (`htp_concat_kernel_params`): 0 = `kernel_type | dim << 8 | n_threads << 16`,
/// 1 `vtcm_size`, then the union from word 2: transposed = `div_ne2` (`mp, l`); gather = a
/// `GatherRowsPlan`. The transposed kernel takes `n_threads` (0 means unknown, which the host
/// defaults to 8) and a VTCM working set.
pub fn build_concat_kernel_params(
    src0: &CopyTensor,
    src1: &CopyTensor,
    dst: &CopyTensor,
    dim: i32,
    n_threads: u32,
    vtcm_bytes: usize,
) -> Option<[i32; 32]> {
    if !(0..4).contains(&dim) {
        return None;
    }
    let d = dim as usize;
    let f32t = HtpDataType::F32 as u32;
    let f16t = HtpDataType::F16 as u32;
    let i32t = HtpDataType::I32 as u32;
    if ![f32t, f16t, i32t].contains(&dst.dtype)
        || src0.dtype != dst.dtype
        || src1.dtype != dst.dtype
    {
        return None;
    }
    let ts = dst.type_size()?;
    for i in 0..4 {
        let ne_i = if i == d {
            u64::from(src0.ne[i]) + u64::from(src1.ne[i])
        } else {
            u64::from(src0.ne[i])
        };
        if u64::from(dst.ne[i]) != ne_i || (i != d && src1.ne[i] != dst.ne[i]) {
            return None;
        }
    }
    let pack = |kernel: u32, threads: u32| {
        (kernel | ((d as u32) << 8) | ((wire_byte(threads as usize) as u32) << 16)) as i32
    };
    let mut k = [0i32; 32];

    let nb0 = |t: &CopyTensor| u64::from(t.nb[0]);
    let total_rows = (u64::from(dst.ne[1]) * u64::from(dst.ne[2]) * u64::from(dst.ne[3])) as u32;
    let b_dense = nb0(src1) == ts;
    let gather_strides_ok = u64::from(src0.nb[1]) >= u64::from(src0.ne[0]) * ts
        && if b_dense {
            u64::from(src1.nb[1]) >= u64::from(src1.ne[0]) * ts
        } else {
            u64::from(src1.nb[0]) >= u64::from(src1.ne[1]) * ts
        };
    if d == 0
        && dst.dtype != i32t
        && u64::from(dst.ne[0]) <= 128 / ts
        && nb0(dst) == ts
        && u64::from(dst.nb[1]) == u64::from(dst.ne[0]) * ts
        && nb0(src0) == ts
        && (nb0(src1) == ts || u64::from(src1.nb[1]) == ts)
        && gather_strides_ok
        && gather_rows_outer_contiguous(dst)
        && gather_rows_outer_contiguous(src0)
        && gather_rows_outer_contiguous(src1)
        && let Some(plan) = gather_rows_solve_layout(
            src0.ne[0],
            src0.nb[1],
            src1.ne[0],
            src1.nb[1],
            b_dense,
            ts as u32,
            total_rows,
            n_threads,
            1,
            vtcm_bytes as u32,
        )
    {
        k[0] = pack(CONCAT_KERNEL_GATHER_ROWS, n_threads);
        k[1] = plan.slice.wrapping_mul(n_threads) as i32;
        plan.write(&mut k[2..]);
        return Some(k);
    }

    if nb0(src0) == ts && nb0(src1) == ts && nb0(dst) == ts {
        k[0] = pack(CONCAT_KERNEL_REGULAR, 1);
        return Some(k);
    }

    let src1_transposed = src1.nb[0] > src1.nb[1];
    let src0_transposed = src0.nb[0] > src0.nb[1];
    let rows_ok = nb0(src0) == ts && u64::from(src1.nb[1]) == ts && nb0(dst) == ts;
    if d == 0 && src1_transposed && !src0_transposed && rows_ok && dst.dtype != i32t {
        let threads = if n_threads > 0 { n_threads } else { 8 };
        // htp_concat_transposed_vtcm_layout_build: src1 is consumed in blocks of tokens, so the
        // working set does not grow with the batch; both sides are double-buffered
        let ts32 = ts as u32;
        let block_rows = 128u32;
        let block_cols = if ts32 == 4 { 256u32 } else { 512 };
        let spad1_stride = (block_rows * ts32).wrapping_add(ts32);
        let block_cols_padded =
            hex_round_up(src1.ne[0], if ts32 == 4 { 32 } else { 64 }).min(block_cols);
        let spad0_row_bytes = hex_round_up(src0.ne[0].wrapping_mul(ts32), 128)
            .wrapping_add(block_cols_padded.wrapping_mul(ts32));
        let spad0 = 2u32.wrapping_mul(block_rows).wrapping_mul(spad0_row_bytes);
        let spad1 = 2u32
            .wrapping_mul(block_cols_padded)
            .wrapping_mul(spad1_stride);
        let total =
            (u64::from(threads) * u64::from(spad0) + u64::from(threads) * u64::from(spad1)) as u32;
        if vtcm_bytes > 0 && u64::from(total) > vtcm_bytes as u64 {
            return None;
        }
        k[0] = pack(CONCAT_KERNEL_TRANSPOSED, threads);
        k[1] = total as i32;
        let div_ne2 = init_fastdiv(dst.ne[2]);
        k[2] = div_ne2.mp as i32;
        k[3] = div_ne2.l as i32;
        return Some(k);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every case of `testdata/ssm_conv_golden.txt`: llama.cpp's real `htp_kparams_solve` for an
    /// `SSM_CONV` (compiled and run by `scripts/hexagon-golden/gen.py`): the decode and the tiled
    /// prefill kernels, tap counts up to the limit, session thread counts and VTCM budgets from
    /// 64 KiB to none, and the operands it refuses.
    #[test]
    fn ssm_conv_kernel_params_match_the_llama_cpp_host_code() {
        let (mut cases, mut unsupported) = (0usize, 0usize);
        for (line, t, run, want) in golden_cases(include_str!("testdata/ssm_conv_golden.txt"), 3) {
            let got =
                build_ssm_conv_kernel_params(&t[0], &t[1], &t[2], run[0] as u32, run[1] as usize);
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
            unsupported += usize::from(want.is_none());
        }
        assert!(
            cases > 6000 && unsupported > 2000,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported"
        );
    }

    /// Parse `ne0..3 nb0..3` after a type id, from a golden-file side.
    fn golden_tensor(f: &mut std::str::SplitWhitespace<'_>) -> CopyTensor {
        let mut next = || -> u64 { f.next().unwrap().parse().unwrap() };
        let dtype = next() as u32;
        let ne = [next() as u32, next() as u32, next() as u32, next() as u32];
        let nb = [next() as u32, next() as u32, next() as u32, next() as u32];
        CopyTensor { dtype, ne, nb }
    }

    /// The cases of a `scripts/hexagon-golden` file whose lines hold `n_tensors` tensors, then
    /// the run fields (`|`-separated) and the words after `=>`: `(line, tensors, run fields,
    /// words)`. The words are `None` for `unsupported`; the file omits trailing zeros of the
    /// 32-word blob, which are restored here.
    #[allow(clippy::type_complexity)]
    fn golden_cases(
        text: &'static str,
        n_tensors: usize,
    ) -> impl Iterator<Item = (&'static str, Vec<CopyTensor>, Vec<i64>, Option<Vec<i32>>)> {
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(move |line| {
                let (inputs, want) = line.split_once("=>").unwrap();
                let mut sides = inputs.split('|');
                let tensors = (0..n_tensors)
                    .map(|_| golden_tensor(&mut sides.next().unwrap().split_whitespace()))
                    .collect();
                let run = sides
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                let want = want.trim();
                let words = (want != "unsupported").then(|| {
                    let mut words: Vec<i32> = want
                        .split_whitespace()
                        .map(|w| w.parse::<u32>().unwrap() as i32)
                        .collect();
                    words.resize(32, 0);
                    words
                });
                (line, tensors, run, words)
            })
    }

    /// The window, taps and output tensors of a dense `SsmConv`.
    fn ssm_conv_tensors(d_conv: u32, d_inner: u32, n_t: u32, n_s: u32) -> [CopyTensor; 3] {
        let f32t = HtpDataType::F32 as u32;
        let dense = |ne: [u32; 4]| CopyTensor {
            dtype: f32t,
            ne,
            nb: [4, ne[0] * 4, ne[0] * ne[1] * 4, ne[0] * ne[1] * ne[2] * 4],
        };
        [
            dense([d_conv - 1 + n_t, d_inner, n_s, 1]),
            dense([d_conv, d_inner, 1, 1]),
            dense([d_inner, n_t, n_s, 1]),
        ]
    }

    /// Every case of `testdata/concat_golden.txt`: llama.cpp's real `htp_kparams_solve` for a
    /// `CONCAT` (compiled and run by `scripts/hexagon-golden/gen.py`): the regular, transposed
    /// and gather-rows kernels over every dim, dtype and operand layout, session thread counts
    /// and VTCM budgets, and the joins it refuses.
    #[test]
    fn concat_kernel_params_match_the_llama_cpp_host_code() {
        let (mut cases, mut unsupported) = (0usize, 0usize);
        let mut kernels = [0usize; 4];
        for (line, t, run, want) in golden_cases(include_str!("testdata/concat_golden.txt"), 3) {
            let got = build_concat_kernel_params(
                &t[0],
                &t[1],
                &t[2],
                run[0] as i32,
                run[1] as u32,
                run[2] as usize,
            );
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
            unsupported += usize::from(want.is_none());
            if let Some(w) = &want {
                kernels[(w[0] & 0xff) as usize] += 1;
            }
        }
        assert!(
            cases > 10000 && unsupported > 4000,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported"
        );
        // every kernel stays reachable from the table: regular, transposed, gather rows
        assert!(
            kernels[1] > 3000 && kernels[2] > 1000 && kernels[3] > 1500,
            "kernel mix {kernels:?}"
        );
    }

    /// The copies and joins cera's graphs issue pick the kernels the firmware was built for (the
    /// golden tables pin every word; this pins why a graph shape lands on a kernel). Kernel type
    /// is the low byte of word 0.
    #[test]
    fn cera_graph_shapes_pick_the_expected_copy_and_concat_kernels() {
        let (f32t, f16t) = (HtpDataType::F32 as u32, HtpDataType::F16 as u32);
        let view = |dtype: u32, ne0: u32, ne1: u32, nb0: u32, nb1: u32| {
            let span = (ne0 - 1) * nb0 + (ne1 - 1) * nb1 + 4;
            CopyTensor {
                dtype,
                ne: [ne0, ne1, 1, 1],
                nb: [nb0, nb1, span, span],
            }
        };
        let kernel = |k: Option<[i32; 32]>| k.map(|w| w[0] as u32 & 0xff);
        let (hs, threads, vtcm) = (1024u32, 8, 8 << 20);

        // LFM2's conv-state prepend, `[2, hs]` state rows + bx as a transposed `[m, hs]` view: the
        // gather-rows kernel while the joined row is at most 32 floats, the transposed worker past it
        for (m, want) in [(1, 3), (8, 3), (30, 3), (31, 2), (32, 2), (128, 2)] {
            let state = view(f32t, 2, hs, 4, 8);
            let bx = view(f32t, m, hs, hs * 4, 4);
            let dst = view(f32t, 2 + m, hs, 4, (2 + m) * 4);
            let got = build_concat_kernel_params(&state, &bx, &dst, 0, threads, vtcm);
            assert_eq!(kernel(got), Some(want), "m = {m}");
        }
        // the conformer's pads (dense on both sides) and a join the kernels cannot read
        let pad = view(f32t, 4, hs, 4, 16);
        let seq = view(f32t, 126, hs, 4, 126 * 4);
        let dst = view(f32t, 130, hs, 4, 130 * 4);
        assert_eq!(
            kernel(build_concat_kernel_params(
                &pad, &seq, &dst, 0, threads, vtcm
            )),
            Some(1)
        );
        let strided = view(f32t, 126, hs, 8, 126 * 8);
        assert_eq!(
            kernel(build_concat_kernel_params(
                &pad, &strided, &dst, 0, threads, vtcm
            )),
            None
        );

        // conv-state write-back (short strided rows into a dense state): gather rows
        let window = view(f32t, 2, hs, 4, 34 * 4);
        let state = view(f32t, 2, hs, 4, 8);
        assert_eq!(
            kernel(build_copy_kernel_params(&window, &state, threads, vtcm)),
            Some(6)
        );
        // time-major to channel-major (and back): the tiled transpose, f32 and f16
        for dtype in [f32t, f16t] {
            let ts = if dtype == f32t { 4 } else { 2 };
            let time_major = view(dtype, 64, hs, hs * ts, ts);
            let channel_major = view(dtype, 64, hs, ts, 64 * ts);
            assert_eq!(
                kernel(build_copy_kernel_params(
                    &time_major,
                    &channel_major,
                    threads,
                    vtcm
                )),
                Some(7)
            );
        }
        // the interleaved conv state (stride 8) goes through the reshaping copy, in both directions
        let interleaved = view(f32t, hs, 1, 8, 8);
        let packed = view(f32t, hs, 1, 4, hs * 4);
        for (a, b) in [(&interleaved, &packed), (&packed, &interleaved)] {
            assert_eq!(
                kernel(build_copy_kernel_params(a, b, threads, vtcm)),
                Some(4)
            );
        }
        // narrowing an activation: the converting copy (needs the VTCM for its staging rows)
        let act = view(f32t, hs, 40, 4, hs * 4);
        let half = view(f16t, hs, 40, 2, hs * 2);
        assert_eq!(
            kernel(build_copy_kernel_params(&act, &half, threads, vtcm)),
            Some(3)
        );
        assert_eq!(
            kernel(build_copy_kernel_params(&act, &half, threads, 4096)),
            None
        );
    }

    /// Every case of `testdata/set_rows_golden.txt`: llama.cpp's real `htp_kparams_solve` for a
    /// `SET_ROWS` (compiled and run by `scripts/hexagon-golden/gen.py`), over the cache dtypes,
    /// row widths, row counts, session thread counts and VTCM budgets cera meets.
    #[test]
    fn set_rows_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/set_rows_golden.txt");
        let (mut cases, mut unsupported) = (0usize, 0usize);
        for line in golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let (inputs, want) = line.split_once("=>").unwrap();
            let mut sides = inputs.split('|');
            let v = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let i = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let d = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let mut run = sides.next().unwrap().split_whitespace();
            let threads: u32 = run.next().unwrap().parse().unwrap();
            let vtcm: usize = run.next().unwrap().parse().unwrap();
            let got = build_set_rows_kernel_params(&v, &i, &d, threads, vtcm);
            let want = want.trim();
            if want == "unsupported" {
                assert_eq!(got, None, "{line}");
                unsupported += 1;
            } else {
                let words: Vec<i32> = want
                    .split_whitespace()
                    .map(|w| w.parse::<u32>().unwrap() as i32)
                    .collect();
                assert_eq!(got.map(|k| k.to_vec()), Some(words), "{line}");
            }
            cases += 1;
        }
        assert!(
            cases > 5000 && unsupported > 1000,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported"
        );
    }

    /// Every case of `testdata/cpy_golden.txt`: llama.cpp's real `htp_kparams_solve` for a `CPY`
    /// (compiled and run by `scripts/hexagon-golden/gen.py`): all seven kernels over dtype pairs,
    /// shapes and operand layouts (dense, row-padded, transposed, permuted, wider views), session
    /// thread counts and VTCM budgets, reshaping copies, the model's strided 2-d copies, and the
    /// pairs it refuses.
    #[test]
    fn copy_kernel_params_match_the_llama_cpp_host_code() {
        let (mut cases, mut unsupported) = (0usize, 0usize);
        let mut kernels = [0usize; 8];
        for (line, t, run, want) in golden_cases(include_str!("testdata/cpy_golden.txt"), 2) {
            let got = build_copy_kernel_params(&t[0], &t[1], run[0] as u32, run[1] as usize);
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
            unsupported += usize::from(want.is_none());
            if let Some(w) = &want {
                kernels[(w[0] & 0xff) as usize] += 1;
            }
        }
        assert!(
            cases > 12000 && unsupported > 3000,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported"
        );
        // every kernel stays reachable from the table
        for (kernel, n) in kernels.iter().enumerate().skip(1) {
            assert!(
                *n > 100,
                "kernel {kernel} only appears {n} times: {kernels:?}"
            );
        }
    }

    /// In-contract counts pass through untouched; 255 is the largest one-byte param.
    #[test]
    fn wire_byte_passes_through_in_contract_counts() {
        assert_eq!(wire_byte(0), 0);
        assert_eq!(wire_byte(1), 1);
        assert_eq!(wire_byte(8), 8);
        assert_eq!(wire_byte(255), 255);
    }

    /// Out-of-contract counts panic in debug builds and saturate at 255 in release builds.
    #[test]
    fn wire_byte_out_of_contract_panics_in_debug_or_saturates_in_release() {
        if cfg!(debug_assertions) {
            let panicked = std::panic::catch_unwind(|| wire_byte(256)).is_err();
            assert!(panicked, "out-of-contract count must trip debug assertion");
        } else {
            assert_eq!(wire_byte(256), 255);
            assert_eq!(wire_byte(1000), 255);
        }
    }

    /// Printed words of a golden line padded with the zeros the rest of the blob holds, or `None`
    /// for an `unsupported` case.
    fn golden_words(want: &str) -> Option<Vec<i32>> {
        let want = want.trim();
        if want == "unsupported" {
            return None;
        }
        let mut words: Vec<i32> = want
            .split_whitespace()
            .map(|w| w.parse::<u32>().unwrap() as i32)
            .collect();
        words.resize(32, 0);
        Some(words)
    }

    /// A tensor side that may be a `-` (absent operand).
    fn golden_optional_tensor(side: &str) -> Option<CopyTensor> {
        (side.trim() != "-").then(|| golden_tensor(&mut side.split_whitespace()))
    }

    /// `<threads> <vtcm bytes>` of a golden line.
    fn golden_session(side: &str) -> (u32, usize) {
        let mut run = side.split_whitespace();
        (
            run.next().unwrap().parse().unwrap(),
            run.next().unwrap().parse().unwrap(),
        )
    }

    /// The case lines of a golden file as `(line, inputs, want)`.
    fn golden_case_lines(golden: &str) -> impl Iterator<Item = (&str, &str, &str)> {
        golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(|l| {
                let (inputs, want) = l.split_once("=>").unwrap();
                (l, inputs, want)
            })
    }

    /// The 16 op-param words of a golden line.
    fn golden_params(side: &str) -> [i32; 16] {
        let mut params = [0i32; 16];
        let words = side.split_whitespace();
        for (p, w) in params.iter_mut().zip(words) {
            *p = w.parse::<u32>().unwrap() as i32;
        }
        params
    }

    /// Every case of `testdata/rope_golden.txt`: llama.cpp's real `htp_kparams_solve` for a `ROPE`
    /// (`scripts/hexagon-golden/gen.py rope`) over head widths, head and token counts, batches, strides,
    /// every rope mode, partial rotary and offsets, frequency factors, session thread counts, VTCM
    /// budgets and the operands the solver refuses.
    #[test]
    fn rope_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/rope_golden.txt");
        let (mut cases, mut unsupported, mut with_ff) = (0usize, 0usize, 0usize);
        for (line, inputs, want) in golden_case_lines(golden) {
            let mut sides = inputs.split('|');
            let mut tensor = || golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let src0 = tensor();
            let positions = tensor();
            let ff = golden_optional_tensor(sides.next().unwrap());
            let dst = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let params = golden_params(sides.next().unwrap());
            let (threads, vtcm) = golden_session(sides.next().unwrap());
            let got = build_rope_kernel_params(
                &RopeOperands {
                    src0: &src0,
                    positions: &positions,
                    freq_factors: ff.as_ref(),
                    dst: &dst,
                },
                &params,
                threads,
                vtcm,
            );
            let want = golden_words(want);
            unsupported += usize::from(want.is_none());
            with_ff += usize::from(ff.is_some());
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
        }
        assert!(
            cases > 5000 && unsupported > 1500 && with_ff > 400,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported, {with_ff} with factors"
        );
    }

    /// Every case of `testdata/softmax_golden.txt` (`gen.py softmax`): mask types and shapes,
    /// broadcasts, ALiBi head counts, the 512-head and VTCM limits and the operands the solver refuses.
    /// `m0` and `m1` come out of the platform's `powf`, which may differ in the last bit between the
    /// machine that made the file and the one running the test, so those two words may be one ulp off.
    #[test]
    fn softmax_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/softmax_golden.txt");
        let (mut cases, mut unsupported, mut sinks, mut alibi) = (0usize, 0usize, 0usize, 0usize);
        for (line, inputs, want) in golden_case_lines(golden) {
            let mut sides = inputs.split('|');
            let src0 = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let mask = golden_optional_tensor(sides.next().unwrap());
            let dst = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let mut op = sides.next().unwrap().split_whitespace();
            let mut params = [0i32; 16];
            params[0] = op.next().unwrap().parse::<u32>().unwrap() as i32;
            params[1] = op.next().unwrap().parse::<u32>().unwrap() as i32;
            let has_sinks = op.next().unwrap() == "1";
            let (threads, vtcm) = golden_session(sides.next().unwrap());
            let want = golden_words(want);
            cases += 1;
            unsupported += usize::from(want.is_none());
            alibi += usize::from(f32::from_bits(params[1] as u32) > 0.0);
            if has_sinks {
                // attention sinks have no kernel: the queue refuses a third operand
                assert_eq!(want, None, "{line}");
                sinks += 1;
                continue;
            }
            let got =
                build_softmax_kernel_params(&src0, mask.as_ref(), &dst, &params, threads, vtcm)
                    .map(|k| k.to_vec());
            match (&got, &want) {
                (Some(g), Some(w)) => {
                    for (i, (g, w)) in g.iter().zip(w).enumerate() {
                        let ulps = (*g as u32).abs_diff(*w as u32);
                        let slope = i == 9 || i == 10;
                        assert!(
                            if slope { ulps <= 1 } else { ulps == 0 },
                            "word {i}: {g} != {w}: {line}"
                        );
                    }
                }
                _ => assert_eq!(got, want, "{line}"),
            }
        }
        assert!(
            cases > 5000 && unsupported > 2000 && sinks > 0 && alibi > 150,
            "golden file looks truncated: {cases} cases"
        );
    }

    /// The strided-mask variant differs from the host solver only in the mask's row strides: it gives
    /// the host's answer for a contiguous mask, and for any other mask the host's answer for the
    /// contiguous mask of the same shape, as long as its rows are dense.
    #[test]
    fn strided_softmax_mask_only_relaxes_the_row_stride_check() {
        let golden = include_str!("testdata/softmax_golden.txt");
        let (mut relaxed, mut strided_rows_refused) = (0usize, 0usize);
        for (line, inputs, _) in golden_case_lines(golden) {
            let mut sides = inputs.split('|');
            let src0 = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let Some(mask) = golden_optional_tensor(sides.next().unwrap()) else {
                continue;
            };
            let dst = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let mut op = sides.next().unwrap().split_whitespace();
            let mut params = [0i32; 16];
            params[0] = op.next().unwrap().parse::<u32>().unwrap() as i32;
            params[1] = op.next().unwrap().parse::<u32>().unwrap() as i32;
            let (threads, vtcm) = golden_session(sides.next().unwrap());
            let strided = build_softmax_kernel_params_strided_mask(
                &src0,
                Some(&mask),
                &dst,
                &params,
                threads,
                vtcm,
            );
            let size = if mask.dtype == HtpDataType::F16 as u32 {
                2
            } else {
                4
            };
            let dense = CopyTensor {
                nb: [
                    size,
                    mask.ne[0] * size,
                    mask.ne[0] * size * mask.ne[1],
                    mask.ne[0] * size * mask.ne[1] * mask.ne[2],
                ],
                ..mask
            };
            let host =
                build_softmax_kernel_params(&src0, Some(&dense), &dst, &params, threads, vtcm);
            if mask.nb[0] != size && mask.ne[0] != 1 {
                assert_eq!(strided, None, "a mask whose rows are not dense: {line}");
                strided_rows_refused += 1;
            } else {
                assert_eq!(strided, host, "{line}");
                relaxed += usize::from(
                    build_softmax_kernel_params(&src0, Some(&mask), &dst, &params, threads, vtcm)
                        .is_none()
                        && host.is_some(),
                );
            }
        }
        assert!(relaxed >= 5, "no strided mask case: {relaxed}");
        assert!(strided_rows_refused >= 6, "{strided_rows_refused}");
    }

    /// Every case of `testdata/sum_rows_golden.txt` (`gen.py sum_rows`): `Sum`, `SumRows` and `Argmax`
    /// over one long row, many rows, strided and mistyped operands, and session thread counts.
    #[test]
    fn sum_rows_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/sum_rows_golden.txt");
        let (mut cases, mut unsupported) = (0usize, 0usize);
        for (line, inputs, want) in golden_case_lines(golden) {
            let mut sides = inputs.split('|');
            let mut src = sides.next().unwrap().split_whitespace();
            let opcode = match src.next().unwrap() {
                "0" => HtpOpCode::Sum,
                "1" => HtpOpCode::SumRows,
                _ => HtpOpCode::Argmax,
            };
            let src0 = golden_tensor(&mut src);
            let dst = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let threads: u32 = sides.next().unwrap().trim().parse().unwrap();
            let got = build_sum_rows_kernel_params(opcode as u32, &src0, &dst, threads);
            let want = golden_words(want);
            unsupported += usize::from(want.is_none());
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
        }
        assert!(
            cases > 5000 && unsupported > 1000,
            "golden file looks truncated: {cases} cases"
        );
        // another opcode is not one of the three
        let t = CopyTensor {
            dtype: HtpDataType::F32 as u32,
            ne: [4, 1, 1, 1],
            nb: [4, 16, 16, 16],
        };
        assert_eq!(
            build_sum_rows_kernel_params(HtpOpCode::Add as u32, &t, &t, 4),
            None
        );
    }

    /// Every case of `testdata/sort_golden.txt` (`gen.py sort`): `Argsort` and `TopK` over one long row
    /// (the chunk and merge layouts at every width around a power of two) and over many rows, with
    /// session thread counts and VTCM budgets from 4 KiB to the 8 MiB default.
    #[test]
    fn sort_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/sort_golden.txt");
        let (mut cases, mut unsupported, mut chunked) = (0usize, 0usize, 0usize);
        for (line, inputs, want) in golden_case_lines(golden) {
            let mut sides = inputs.split('|');
            let mut src = sides.next().unwrap().split_whitespace();
            let opcode = match src.next().unwrap() {
                "0" => HtpOpCode::Argsort,
                _ => HtpOpCode::TopK,
            };
            let mut params = [0i32; 16];
            params[0] = src.next().unwrap().parse().unwrap();
            let src0 = golden_tensor(&mut src);
            let dst = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let (threads, vtcm) = golden_session(sides.next().unwrap());
            let got = build_sort_kernel_params(opcode as u32, &src0, &dst, &params, threads, vtcm);
            let want = golden_words(want);
            unsupported += usize::from(want.is_none());
            chunked += usize::from(want.as_ref().is_some_and(|w| w[4] > 1));
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
        }
        assert!(
            cases > 5000 && unsupported > 500 && chunked > 500,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported, {chunked} chunked"
        );
        // another opcode is not a sort
        let t = CopyTensor {
            dtype: HtpDataType::F32 as u32,
            ne: [4, 1, 1, 1],
            nb: [4, 16, 16, 16],
        };
        assert_eq!(
            build_sort_kernel_params(HtpOpCode::Add as u32, &t, &t, &[0; 16], 4, 1 << 20),
            None
        );
    }

    /// Every case of `testdata/get_rows_golden.txt` (`gen.py get_rows`): the same-type, tiled and flat
    /// kernels over every table type, index type and shape, thread counts and VTCM budgets that force
    /// the thread count down, and the table, index and output types the solver refuses.
    #[test]
    fn get_rows_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/get_rows_golden.txt");
        let (mut cases, mut unsupported, mut zero_rows, mut zero_vtcm) =
            (0usize, 0usize, 0usize, 0usize);
        let mut kernels = [0usize; 3];
        for (line, inputs, want) in golden_case_lines(golden) {
            let mut sides = inputs.split('|');
            let mut table = sides.next().unwrap().split_whitespace();
            let src0 = golden_tensor(&mut table);
            let tiled = table.next().unwrap() == "1";
            let src1 = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let dst = golden_tensor(&mut sides.next().unwrap().split_whitespace());
            let (threads, vtcm) = golden_session(sides.next().unwrap());
            let got = build_get_rows_kernel_params(&src0, &src1, &dst, tiled, threads, vtcm);
            let want = golden_words(want);
            unsupported += usize::from(want.is_none());
            zero_rows += usize::from(src1.ne[0] == 0 && want.is_some());
            zero_vtcm += usize::from(vtcm == 0);
            if let Some(w) = &want {
                kernels[w[0] as usize] += 1;
            }
            assert_eq!(got.map(|k| k.to_vec()), want, "{line}");
            cases += 1;
        }
        assert!(
            cases > 5000 && unsupported > 1000,
            "golden file looks truncated: {cases} cases"
        );
        assert!(zero_rows > 50, "zero-row cases missing: {zero_rows}");
        assert!(zero_vtcm > 500, "zero-VTCM cases missing: {zero_vtcm}");
        assert!(
            kernels.iter().all(|&n| n > 300),
            "a kernel is missing: {kernels:?}"
        );
    }

    fn dense(dtype: HtpDataType, ne: [u32; 4]) -> CopyTensor {
        let size = match dtype {
            HtpDataType::F16 => 2,
            _ => 4,
        };
        CopyTensor {
            dtype: dtype as u32,
            ne,
            nb: [
                size,
                size * ne[0],
                size * ne[0] * ne[1],
                size * ne[0] * ne[1] * ne[2],
            ],
        }
    }

    /// The routed-FFN gather: 4 expert weights from a 32-entry table of single values. The
    /// same-type kernel still stages the index vector (128 bytes plus one vector, 256-aligned).
    #[test]
    fn get_rows_params_of_the_routed_ffn_gather() {
        let table = dense(HtpDataType::F32, [1, 32, 1, 1]);
        let idx = dense(HtpDataType::I32, [4, 1, 1, 1]);
        let out = dense(HtpDataType::F32, [1, 4, 1, 1]);
        let k = build_get_rows_kernel_params(&table, &idx, &out, false, 8, 8 << 20).unwrap();
        // kernel_type SAMETYPE, n_threads (each takes at least 128 output bytes, a row is 4: one
        // thread), tasks, vtcm
        assert_eq!(k[..4], [0, 1, 4, 256]);
        assert_eq!(k[4..6], fastdiv_words(4)); // ne10
        assert_eq!(k[6..8], fastdiv_words(4)); // ne10 * ne11
        assert_eq!(k[8..10], fastdiv_words(1)); // ne02
        assert_eq!(k[10..12], fastdiv_words(1)); // ne03
        assert!(k[12..].iter().all(|&v| v == 0));
        // no VTCM at all is a refusal, not a clipped layout
        assert_eq!(
            build_get_rows_kernel_params(&table, &idx, &out, false, 8, 0),
            None
        );
    }

    /// Greedy sampling over a vocabulary: 8 threads each take a 32-aligned slice of one row.
    #[test]
    fn argmax_params_split_a_vocabulary_row_across_threads() {
        let logits = dense(HtpDataType::F32, [151936, 1, 1, 1]);
        let idx = dense(HtpDataType::I32, [1, 1, 1, 1]);
        let k = build_sum_rows_kernel_params(HtpOpCode::Argmax as u32, &logits, &idx, 8).unwrap();
        assert_eq!(k[..2], [8, 19008]); // ceil(151936 / 8) = 18992, rounded up to a multiple of 32
        let k = build_sum_rows_kernel_params(HtpOpCode::Argmax as u32, &logits, &idx, 12).unwrap();
        assert_eq!(k[..2], [10, 15200], "capped at HTP_MAX_NTHREADS (10)");
        let small = dense(HtpDataType::F32, [1000, 1, 1, 1]);
        let k = build_sum_rows_kernel_params(HtpOpCode::Argmax as u32, &small, &idx, 8).unwrap();
        assert_eq!(
            k[..2],
            [1, 1024],
            "under 1024 elements one thread takes the row"
        );
    }

    /// MoE routing: one row of expert scores, sorted descending in one 32-element vector set.
    #[test]
    fn argsort_params_of_expert_routing() {
        let scores = dense(HtpDataType::F32, [32, 1, 1, 1]);
        let order = dense(HtpDataType::I32, [32, 1, 1, 1]);
        let mut params = [0i32; 16];
        params[0] = 1; // descending
        let k = build_sort_kernel_params(
            HtpOpCode::Argsort as u32,
            &scores,
            &order,
            &params,
            8,
            8 << 20,
        )
        .unwrap();
        // 1 thread, descending, not top-k, 32 elements in 1 chunk, 2 slots of 256 bytes, no merge
        assert_eq!(k[..8], [1, 1, 0, 32, 1, 512, 0, 2]);
        assert_eq!(k[8..10], fastdiv_words(1));
    }

    /// A decode-step rope over `[head_dim, heads]` and the same with frequency factors: one thread
    /// per head up to the session's, a theta cache and 32 rows of scratch per thread, the factors'
    /// 256-aligned slot after the slabs.
    #[test]
    fn rope_params_of_a_decode_step() {
        let act = dense(HtpDataType::F32, [128, 8, 1, 1]);
        let pos = dense(HtpDataType::I32, [1, 1, 1, 1]);
        let mut params = build_rope_params(128, 2, 4096, 10000.0, 1.0);
        params[3] = 0;
        let ops = |ff| RopeOperands {
            src0: &act,
            positions: &pos,
            freq_factors: ff,
            dst: &act,
        };
        let k = build_rope_kernel_params(&ops(None), &params, 6, 8 << 20).unwrap();
        // 512-byte rows: a 512-byte theta cache and 32 x 512 bytes of rows per thread
        assert_eq!(k[..2], [6, 6 * (512 + 32 * 512)]);
        assert_eq!(k[2..4], fastdiv_words(8)); // ne2 * ne1
        assert_eq!(k[4..6], fastdiv_words(8)); // ne1
        let ff = dense(HtpDataType::F32, [64, 1, 1, 1]);
        let k = build_rope_kernel_params(&ops(Some(&ff)), &params, 6, 8 << 20).unwrap();
        assert_eq!(k[1], 6 * (512 + 32 * 512) + 256);
        // fewer factors than rotated pairs, or a truncating-off op, are refused
        let few = dense(HtpDataType::F32, [63, 1, 1, 1]);
        assert_eq!(
            build_rope_kernel_params(&ops(Some(&few)), &params, 6, 8 << 20),
            None
        );
        let mut off = params;
        off[3] = 1;
        assert_eq!(build_rope_kernel_params(&ops(None), &off, 6, 8 << 20), None);
    }

    /// The audio encoder's softmax: scores `[t, t, heads]` plus a relative-position bias that is a view
    /// of a wider matrix. The host solver refuses the strided view; the variant the queue uses takes it.
    #[test]
    fn softmax_params_of_a_relative_position_bias() {
        let t = 126u32;
        let scores = dense(HtpDataType::F32, [t, t, 8, 1]);
        let lp = 256u32; // 2t - 1 rounded up to 32
        let shifted = CopyTensor {
            dtype: HtpDataType::F32 as u32,
            ne: [t, t, 8, 1],
            nb: [4, (lp - 1) * 4, t * lp * 4, t * lp * 4 * 8],
        };
        let mut params = [0i32; 16];
        params[0] = 0.125f32.to_bits() as i32;
        assert_eq!(
            build_softmax_kernel_params(&scores, Some(&shifted), &scores, &params, 8, 8 << 20),
            None
        );
        let k = build_softmax_kernel_params_strided_mask(
            &scores,
            Some(&shifted),
            &scores,
            &params,
            8,
            8 << 20,
        )
        .unwrap();
        // masked F32 kernel, 8 threads, three 512-byte double buffers per thread, 8 heads
        assert_eq!(k[..7], [1, 8, 8 * 3 * 1024, 8, 8, 1, 0]);
        assert_eq!(f32::from_bits(k[7] as u32), 0.125);
        assert_eq!(f32::from_bits(k[9] as u32), 1.0);
        assert_eq!(k[11..13], fastdiv_words(t)); // ne01
        assert_eq!(k[13..15], fastdiv_words(8)); // ne02
        // without a mask: the no-mask kernel and no mask buffers
        let k = build_softmax_kernel_params(&scores, None, &scores, &params, 8, 8 << 20).unwrap();
        assert_eq!((k[0], k[2], k[5]), (0, 8 * 2 * 2 * 512, 0));
    }

    #[test]
    fn test_fastdiv_basic() {
        let divisors = [1, 2, 3, 5, 7, 10, 16, 32, 64, 128, 256, 1024, 4096];
        for &d in &divisors {
            let fd = init_fastdiv(d);
            for n in [0, 1, 2, d - 1, d, d + 1, d * 5 + 3, 65535, 1_000_000] {
                let hi = (((n as u64) * (fd.mp as u64)) >> 32) as u32;
                let q = (hi + n) >> fd.l;
                assert_eq!(q, n / d, "FastDiv mismatch for n={}, d={}", n, d);
            }
        }
    }

    #[test]
    fn test_fastdiv_zero_divisor() {
        let fd = init_fastdiv(0);
        assert_eq!(fd.mp, 0);
        assert_eq!(fd.l, 0);
    }

    #[test]
    fn test_param_builders() {
        let rms = build_rms_norm_params(1e-5);
        assert_eq!(rms[0], (1e-5f32).to_bits() as i32);

        let rope = build_rope_params(128, 0, 4096, 10000.0, 1.0);
        assert_eq!(rope[1], 128);
        assert_eq!(rope[2], 0);
        assert_eq!(rope[4], 4096);
        assert_eq!(rope[5], (10000.0f32).to_bits() as i32);
        assert_eq!(rope[6], (1.0f32).to_bits() as i32);
    }

    #[test]
    fn test_hmx_fa_solver_goldens() {
        // Oracle: hmx_fa_find_chunk_size + hmx_fa_compute_vtcm_usage from
        // upstream flash-attn-ops.h (8MB budget, 4 threads unless noted).
        let b8 = 8 * 1024 * 1024;
        // (g, dk, dv, qo, kv) -> (Br, Bc, vtcm).
        let cases = [
            ((2, 64, 64, 8, 8), (16, 64, 85632)),
            ((2, 64, 64, 13, 13), (16, 64, 85632)),
            ((2, 64, 64, 32, 32), (32, 64, 118400)),
            ((2, 64, 64, 32, 64), (32, 64, 118400)),
            ((2, 64, 64, 32, 256), (32, 64, 155264)),
            ((2, 64, 64, 32, 602), (32, 192, 386688)),
            ((2, 64, 64, 32, 2048), (32, 640, 1195648)),
            ((1, 64, 64, 32, 128), (32, 128, 167552)),
            ((4, 64, 64, 32, 128), (32, 128, 267008)),
            ((8, 64, 64, 32, 128), (32, 128, 400384)),
            ((2, 128, 128, 32, 128), (32, 128, 323200)),
            ((2, 128, 128, 32, 512), (32, 128, 425600)),
            ((2, 64, 64, 512, 512), (512, 128, 2314752)),
            ((2, 64, 64, 5, 5), (16, 64, 85632)),
        ];
        for ((g, dk, dv, qo, kv), (br, bc, vs)) in cases {
            let got = fa_hmx_find_chunk_size(g, dk, dv, qo, kv, b8, 4, true, false, 16, 1);
            assert_eq!(got, Some((br, bc)), "g={g} qo={qo} kv={kv}");
            let pipelined = kv >= 3 * 64;
            let total = fa_hmx_layout_total(g, dk, dv, br, bc, 4, pipelined, true, false, 16);
            assert_eq!(total, vs, "g={g} qo={qo} kv={kv}");
        }
        // Small budget + single thread.
        assert_eq!(
            fa_hmx_find_chunk_size(2, 64, 64, 32, 128, 1024 * 1024, 4, true, false, 16, 1),
            Some((32, 128))
        );
        assert_eq!(
            fa_hmx_layout_total(2, 64, 64, 32, 128, 4, false, true, false, 16),
            200320
        );
        assert_eq!(
            fa_hmx_find_chunk_size(2, 64, 64, 32, 256, b8, 1, true, false, 16, 1),
            Some((32, 256))
        );
        assert_eq!(
            fa_hmx_layout_total(2, 64, 64, 32, 256, 1, false, true, false, 16),
            363136
        );
    }

    /// A tensor of a flash-attention case, `[type ne0 ne1 ne2 ne3]` as the golden file prints it.
    fn fa_tensor(dtype: HtpDataType, ne: [u32; 4]) -> CopyTensor {
        CopyTensor {
            dtype: dtype as u32,
            ne,
            nb: [0; 4],
        }
    }

    /// LFM2-style attention operands: f32 Q, f16 K/V of `s` rows, `[s, m]` f16 mask.
    fn fa_operands(dk: u32, n_heads: u32, n_kv_heads: u32, m: u32, s: u32) -> [CopyTensor; 5] {
        let f32t = HtpDataType::F32;
        let f16t = HtpDataType::F16;
        [
            fa_tensor(f32t, [dk, m, n_heads, 1]),
            fa_tensor(f16t, [dk, s, n_kv_heads, 1]),
            fa_tensor(f16t, [dk, s, n_kv_heads, 1]),
            fa_tensor(f16t, [s, m, 1, 1]),
            fa_tensor(f32t, [dk, n_heads, m, 1]),
        ]
    }

    fn fa_solve(
        host: &FlashAttnHost,
        t: &[CopyTensor; 5],
        scale: f32,
        softcap: f32,
    ) -> Option<[i32; 32]> {
        build_flash_attn_kernel_params(
            host,
            &FlashAttnOp {
                q: &t[0],
                k: &t[1],
                v: &t[2],
                mask: Some(&t[3]),
                sinks: None,
                dst: &t[4],
                scale,
                max_bias: 0.0,
                logit_softcap: softcap,
            },
        )
    }

    /// Every case of `testdata/flash_attn_golden.txt`: llama.cpp's real `htp_kparams_solve` for a
    /// `FLASH_ATTN_EXT` (compiled and run by `scripts/hexagon-golden/gen.py`), over the model
    /// shapes cera runs, query counts and KV lengths, session thread counts and VTCM budgets,
    /// operand types, mask shapes, soft caps, ALiBi biases and the host's `opt_fa_*` options.
    #[test]
    fn flash_attn_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/flash_attn_golden.txt");
        let tensor = |side: &str| -> Option<CopyTensor> {
            let side = side.trim();
            (side != "-").then(|| {
                let mut f = side.split_whitespace();
                let mut next = || -> u32 { f.next().unwrap().parse().unwrap() };
                CopyTensor {
                    dtype: next(),
                    ne: [next(), next(), next(), next()],
                    nb: [0; 4],
                }
            })
        };
        let (mut cases, mut unsupported, mut hmx, mut hvx, mut teams) = (0usize, 0, 0, 0, 0);
        for line in golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let (inputs, want) = line.split_once("=>").unwrap();
            let mut sides = inputs.split('|');
            let q = tensor(sides.next().unwrap()).unwrap();
            let k = tensor(sides.next().unwrap()).unwrap();
            let v = tensor(sides.next().unwrap()).unwrap();
            let mask = tensor(sides.next().unwrap());
            let sinks = tensor(sides.next().unwrap());
            let dst = tensor(sides.next().unwrap()).unwrap();
            let mut params = sides.next().unwrap().split_whitespace();
            let mut float = || f32::from_bits(params.next().unwrap().parse().unwrap());
            let (scale, max_bias, logit_softcap) = (float(), float(), float());
            let kv_rows: u32 = params.next().unwrap().parse().unwrap();
            let mut run = sides.next().unwrap().split_whitespace();
            let mut next = || -> u64 { run.next().unwrap().parse().unwrap() };
            let host = FlashAttnHost {
                n_threads: next() as u32,
                vtcm_bytes: next() as usize,
                n_hmx: next() as u32,
                mdev_count: next() as u32,
                select: next() as i32,
                split: next() != 0,
            };
            // a K/V row list (src[5]) is refused by the solver, and by the queue before it asks
            let got = (kv_rows == 0)
                .then(|| {
                    build_flash_attn_kernel_params(
                        &host,
                        &FlashAttnOp {
                            q: &q,
                            k: &k,
                            v: &v,
                            mask: mask.as_ref(),
                            sinks: sinks.as_ref(),
                            dst: &dst,
                            scale,
                            max_bias,
                            logit_softcap,
                        },
                    )
                })
                .flatten();
            let want = want.trim();
            if want == "unsupported" {
                assert_eq!(got, None, "{line}");
                unsupported += 1;
            } else {
                let mut words: Vec<i32> = want
                    .split_whitespace()
                    .map(|w| w.parse::<u32>().unwrap() as i32)
                    .collect();
                words.resize(32, 0);
                assert_eq!(got.map(|k| k.to_vec()), Some(words), "{line}");
                let got = got.unwrap();
                match got[0] & 0xff {
                    1 => hvx += 1,
                    2 => hmx += 1,
                    other => panic!("kernel type {other}: {line}"),
                }
                teams += usize::from((got[0] >> 16) & 0xff != 0);
            }
            cases += 1;
        }
        assert!(
            cases > 6000 && unsupported > 800 && hmx > 3000 && hvx > 1000 && teams > 100,
            "golden file looks truncated: {cases} cases, {unsupported} unsupported, {hmx} hmx, {hvx} hvx, {teams} team-split"
        );
    }

    /// The decode and prefill-chunk shapes of an LFM2 attention layer, worked from the C++ by hand:
    /// each field of `htp_fa_kernel_params` where it lands in the words.
    #[test]
    fn flash_attn_params_for_lfm2_shapes_carry_the_solver_fields() {
        let host = FlashAttnHost::new(6, 8 << 20, 1);
        let scale = 0.125f32;

        // a 32-row chunk over 602 KV rows: HMX, Br=32, Bc=192, pipelined across 6 threads
        let t = fa_operands(64, 16, 8, 32, 602);
        let kp = fa_solve(&host, &t, scale, 0.0).expect("chunk fits");
        assert_eq!(kp[0].to_le_bytes(), [2, 1, 0, 6]);
        assert_eq!(kp[1], 32 | (192 << 16));
        assert_eq!(kp[2], 4 | (2 << 16)); // ceil(602 / 192) blocks, G = 2
        assert_eq!(kp[3], (scale * std::f32::consts::LOG2_E).to_bits() as i32);
        assert_eq!((kp[4], kp[5]), (0, 0));
        assert_eq!(kp[6], 388736); // the 4-thread layout's 386688 plus two more row buffers per thread
        assert_eq!(
            (kp[7], kp[8]),
            (1.0f32.to_bits() as i32, 1.0f32.to_bits() as i32)
        );
        assert_eq!(kp[9], 16); // n_head_log2
        let one = init_fastdiv(1);
        assert_eq!((kp[10], kp[11]), (one.mp as i32, one.l as i32)); // mask ne[2] = 1
        assert_eq!((kp[22], kp[23], kp[24]), (64, 1, 1)); // g_br, mask_broadcast, pipeline
        let div_g = init_fastdiv(2);
        assert_eq!((kp[25], kp[26]), (div_g.mp as i32, div_g.l as i32));
        assert_eq!(
            &kp[14..22],
            &[0; 8],
            "HMX leaves the HVX broadcast dividers unset"
        );
        assert_eq!(&kp[27..], &[0; 5]);

        // one decode row over 64 KV rows: HVX, one KV block
        let t = fa_operands(64, 16, 8, 1, 64);
        let kp = fa_solve(&host, &t, scale, 0.0).expect("decode fits");
        assert_eq!(kp[0].to_le_bytes(), [1, 1, 0, 6]);
        assert_eq!(kp[1], 1 | (64 << 16));
        assert_eq!(kp[2], 1 | (2 << 16));
        assert_eq!(kp[3], scale.to_bits() as i32);
        assert_eq!(kp[6], 319488); // 6 threads x (Q, 2 K, 2 V, mask rows, accumulators) = 6 x 53248
        assert_eq!((kp[22], kp[23], kp[24]), (256, 128, 128)); // padded Q (f32), K, V rows
        let div21 = init_fastdiv(16);
        assert_eq!((kp[25], kp[26]), (div21.mp as i32, div21.l as i32)); // q.ne[2] * q.ne[1]
        let div1 = init_fastdiv(1);
        assert_eq!((kp[27], kp[28]), (div1.mp as i32, div1.l as i32));
        assert_eq!((kp[14], kp[15]), (div_g.mp as i32, div_g.l as i32)); // rk2 = 16 / 8
    }

    /// The soft cap is folded into the scale on both kernels, and the HMX kernel's log2(e) goes
    /// into the cap instead of the scale when there is one.
    #[test]
    fn flash_attn_softcap_folds_into_the_scale() {
        let host = FlashAttnHost::new(4, 8 << 20, 1);
        let log2e = std::f32::consts::LOG2_E;
        let hvx = fa_solve(&host, &fa_operands(64, 16, 8, 1, 64), 0.125, 50.0).unwrap();
        assert_eq!(hvx[0] & 0xff, 1);
        assert_eq!(hvx[3], (0.125f32 / 50.0).to_bits() as i32);
        assert_eq!(hvx[5], 50.0f32.to_bits() as i32);
        let hmx = fa_solve(&host, &fa_operands(64, 16, 8, 32, 602), 0.125, 50.0).unwrap();
        assert_eq!(hmx[0] & 0xff, 2);
        assert_eq!(hmx[3], (0.125f32 / 50.0).to_bits() as i32);
        assert_eq!(hmx[5], (50.0f32 * log2e).to_bits() as i32);
        let plain = fa_solve(&host, &fa_operands(64, 16, 8, 32, 602), 0.125, 0.0).unwrap();
        assert_eq!(plain[3], (0.125f32 * log2e).to_bits() as i32);
        assert_eq!(plain[5], 0);
    }

    #[test]
    fn flash_attn_kernel_choice_follows_the_host_options() {
        let t = fa_operands(64, 16, 8, 32, 602);
        let hmx = FlashAttnHost::new(6, 8 << 20, 1);
        assert_eq!(fa_solve(&hmx, &t, 0.125, 0.0).unwrap()[0] & 0xff, 2);
        // no HMX unit, or the caller keeps it off: the HVX kernel
        let no_unit = FlashAttnHost::new(6, 8 << 20, 0);
        assert_eq!(fa_solve(&no_unit, &t, 0.125, 0.0).unwrap()[0] & 0xff, 1);
        assert_eq!(
            fa_solve(&hmx.hvx_only(), &t, 0.125, 0.0).unwrap()[0] & 0xff,
            1
        );
        // opt_fa_select 0 refuses the op
        let off = FlashAttnHost { select: 0, ..hmx };
        assert_eq!(fa_solve(&off, &t, 0.125, 0.0), None);
        // a short chunk at a small head dim stays on HVX unless HMX is forced
        let short = fa_operands(64, 16, 8, 4, 602);
        assert_eq!(fa_solve(&hmx, &short, 0.125, 0.0).unwrap()[0] & 0xff, 2);
        let decode = fa_operands(64, 16, 8, 1, 100);
        assert_eq!(fa_solve(&hmx, &decode, 0.125, 0.0).unwrap()[0] & 0xff, 1);
        let forced = FlashAttnHost { select: 3, ..hmx };
        assert_eq!(fa_solve(&forced, &decode, 0.125, 0.0).unwrap()[0] & 0xff, 2);
        // head dims that do not tile by 8 have no HMX path
        let odd = fa_operands(70, 14, 7, 32, 602);
        assert_eq!(fa_solve(&hmx, &odd, 0.125, 0.0).unwrap()[0] & 0xff, 1);
        // an HVX working set above the budget is unsupported
        let tight = FlashAttnHost::new(6, 64 << 10, 0);
        assert_eq!(fa_solve(&tight, &t, 0.125, 0.0), None);
    }

    /// Shapes the solver cannot divide by (llama.cpp would divide by zero) are unsupported rather
    /// than a panic or garbage params.
    #[test]
    fn flash_attn_degenerate_shapes_are_unsupported() {
        let host = FlashAttnHost::new(6, 8 << 20, 1);
        for (n_heads, n_kv_heads) in [(0, 1), (8, 0), (4, 8)] {
            let t = fa_operands(64, n_heads, n_kv_heads, 1, 64);
            assert_eq!(
                fa_solve(&host, &t, 0.125, 0.0),
                None,
                "{n_heads}/{n_kv_heads}"
            );
        }
    }

    #[test]
    fn test_hmx_fa_eligibility() {
        let host = FlashAttnHost::new(6, 8 << 20, 1);
        let eligible = |dk, m, s| {
            let t = fa_operands(dk, 16, 8, m, s);
            let op = FlashAttnOp {
                q: &t[0],
                k: &t[1],
                v: &t[2],
                mask: Some(&t[3]),
                sinks: None,
                dst: &t[4],
                scale: 0.125,
                max_bias: 0.0,
                logit_softcap: 0.0,
            };
            fa_is_hmx_eligible(&host, &op)
        };
        assert!(eligible(64, 5, 100));
        assert!(eligible(64, 32, 100));
        assert!(eligible(128, 32, 100));
        assert!(!eligible(64, 1, 100)); // decode over a short cache: HVX
        assert!(eligible(64, 1, 768)); // ... and HMX past the crossover at 6 threads
        assert!(!eligible(70, 32, 100)); // DK % 8 != 0
    }

    #[test]
    fn test_prefill_param_builders() {
        // SsmConv: scalar (n_t=1) and chunk (n_t=32) branches, LFM2 K=3/C=1024.
        let ssm = |n_t: u32| {
            let [x, w, y] = ssm_conv_tensors(3, 1024, n_t, 1);
            build_ssm_conv_kernel_params(&x, &w, &y, 4, 8 << 20).unwrap()
        };
        let (ssm1, ssm32) = (ssm(1), ssm(32));
        assert_eq!(ssm1[0], 4); // n_threads
        assert_eq!(ssm1[5], 256); // d_inner_tile (scalar: the full per-thread range)
        assert_eq!(ssm32[5], 64); // a quarter of the 256 channels per thread, within 32..=128
    }

    #[test]
    fn test_layer_norm_params() {
        let ln = build_layer_norm_params(1e-5);
        assert_eq!(ln[0], 1e-5f32.to_bits() as i32);
        for &val in &ln[1..] {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_conv1d_params() {
        let p = build_conv1d_params(2, 3, 1, 4);
        assert_eq!(p[0], 2);
        assert_eq!(p[1], 3);
        assert_eq!(p[2], 1);
        assert_eq!(p[3], 4);
        for &val in &p[4..] {
            assert_eq!(val, 0);
        }

        // Test clamping of zero stride, dilation, groups to 1
        let p_clamped = build_conv1d_params(0, 0, 0, 0);
        assert_eq!(p_clamped[0], 1);
        assert_eq!(p_clamped[1], 0);
        assert_eq!(p_clamped[2], 1);
        assert_eq!(p_clamped[3], 1);
    }

    #[test]
    fn test_snake_params() {
        let p = build_snake_params(1.5);
        assert_eq!(p[0], 1.5f32.to_bits() as i32);
        for &val in &p[1..] {
            assert_eq!(val, 0);
        }

        let p_default = build_snake_params(1.0);
        assert_eq!(p_default[0], 1.0f32.to_bits() as i32);
    }

    #[test]
    fn test_conv_transpose1d_params() {
        let p = build_conv_transpose1d_params(2, 1, 3);
        assert_eq!(p[0], 2);
        assert_eq!(p[1], 1);
        assert_eq!(p[2], 3);
        for &val in &p[3..] {
            assert_eq!(val, 0);
        }

        // Test clamping of zero stride and dilation to 1
        let p_clamped = build_conv_transpose1d_params(0, 0, 0);
        assert_eq!(p_clamped[0], 1);
        assert_eq!(p_clamped[1], 0);
        assert_eq!(p_clamped[2], 1);
    }

    #[test]
    fn test_elu_params() {
        let p = build_elu_params(1.0);
        assert_eq!(p[0], 1.0f32.to_bits() as i32);
        for &val in &p[1..] {
            assert_eq!(val, 0);
        }

        let p_custom = build_elu_params(0.5);
        assert_eq!(p_custom[0], 0.5f32.to_bits() as i32);
    }
}

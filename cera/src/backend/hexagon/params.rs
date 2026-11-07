//! Host-side parameter precomputations for Hexagon HTP operations.
//!
//! Hexagon DSP kernels avoid runtime hardware division and transcendental calls
//! by consuming precomputed integer division constants (Granlund and Montgomery FastDiv)
//! and layout descriptors generated on the host.

use super::types::{HtpDataType, align128, align256};

/// A value for one of the one-byte fields of the matmul params. Out-of-contract counts
/// saturate at 255 rather than wrapping: a wrapped count aliases to a plausible but wrong
/// kernel configuration (`256` threads would enqueue as `0`).
fn wire_byte(v: usize) -> u8 {
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

/// The matmul kernel parameters, as `struct htp_mm_kernel_params` (`matmul-ops.h`) lays them out
/// in the 32-word `kernel_params` blob.
///
/// Eight one-byte fields share the first two words (the DSP reads them as `uint8_t`), then the
/// `int32_t` sizes, then seven `{mp, l}` dividers. Build and patch through this struct rather
/// than by word index: before llama.cpp packed the bytes, every field had a word of its own, and
/// code that still indexes the old positions silently scrambles the parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MmKernelParams {
    /// `enum htp_mm_kernel_type`.
    pub kernel_type: u8,
    /// 1 = pipelined execution.
    pub pipeline: u8,
    /// 1 = outer dimensions collapsed into 2-D (llama.cpp's optimization; cera leaves it 0).
    pub collapse: u8,
    /// 1 = HMX, 0 = HVX.
    pub n_hmx: u8,
    pub n_threads: u8,
    pub n_act_threads: u8,
    pub n_prefetch: u8,
    /// Weight count of a fused `MulMatNx`.
    pub n_weights: u8,
    pub m_chunk: i32,
    pub n_chunk: i32,
    pub tile_size: i32,
    pub aligned_tile_size: i32,
    pub act_row_size: i32,
    pub vtcm_size: i32,
    pub vtcm_src0_size: i32,
    pub vtcm_act_size: i32,
    /// Bias scratch (fused add only).
    pub vtcm_bias_size: i32,
    pub vtcm_dst_size: i32,
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
        w[0] = bytes([self.kernel_type, self.pipeline, self.collapse, self.n_hmx]);
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
        w[8] = self.vtcm_src0_size;
        w[9] = self.vtcm_act_size;
        w[10] = self.vtcm_bias_size;
        w[11] = self.vtcm_dst_size;
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
            w[12 + 2 * i] = d.mp as i32;
            w[13 + 2 * i] = d.l as i32;
        }
        w
    }

    /// Read a blob back, e.g. to patch a field of an already-built set of parameters.
    pub fn from_words(w: &[i32; 32]) -> Self {
        let b0 = (w[0] as u32).to_le_bytes();
        let b1 = (w[1] as u32).to_le_bytes();
        let div = |i: usize| FastDivValues {
            mp: w[12 + 2 * i] as u32,
            l: w[13 + 2 * i] as u32,
        };
        Self {
            kernel_type: b0[0],
            pipeline: b0[1],
            collapse: b0[2],
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
            vtcm_src0_size: w[8],
            vtcm_act_size: w[9],
            vtcm_bias_size: w[10],
            vtcm_dst_size: w[11],
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

/// Host-computed kernel parameters for unary operations (RMS norm, activations).
///
/// Mirrors `ggml_hexagon_precompute_unary_params` + `htp_unary_vtcm_layout_build`.
/// `nrows` is dim-1..3 flattened (rows split across threads); `weight_dim` is
/// the RMS weight width (single broadcast row; ignored without weight).
/// Single-row-per-thread blocking here costs milliseconds on wide row counts
/// (QK norms run `m * n_heads` rows), so size it from the VTCM budget.
pub fn build_unary_kernel_params(
    ne0: usize,
    nrows: usize,
    weight_dim: usize,
    vtcm_size: usize,
    sess_threads: u32,
    has_weight: bool,
) -> [i32; 32] {
    // Debug: single-threaded single-row blocking (pre-port behavior). Never
    // read under `cfg(test)`, so an exported `CERA_HEXAGON_UNARY_T1` cannot
    // change goldens or param tests; tests reach the legacy shape through
    // [`build_unary_kernel_params_with`].
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let legacy = !cfg!(test)
        && *LEGACY.get_or_init(|| {
            std::env::var("CERA_HEXAGON_UNARY_T1")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
    build_unary_kernel_params_with(
        legacy,
        ne0,
        nrows,
        weight_dim,
        vtcm_size,
        sess_threads,
        has_weight,
    )
}

/// [`build_unary_kernel_params`] with the legacy (single-thread, single-row)
/// blocking choice explicit instead of read from the environment.
pub(crate) fn build_unary_kernel_params_with(
    legacy: bool,
    ne0: usize,
    nrows: usize,
    weight_dim: usize,
    vtcm_size: usize,
    sess_threads: u32,
    has_weight: bool,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = if legacy {
        1
    } else {
        (sess_threads.max(1) as usize).min(nrows.max(1))
    };
    let row = (ne0 * 4).next_multiple_of(128);
    let wrow = if has_weight {
        (weight_dim * 4).next_multiple_of(128)
    } else {
        0
    };

    // RMS_NORM_MUL broadcast branch (our weights are always one row);
    // plain reductions share the weightless shape.
    let avail = vtcm_size.saturating_sub(n_threads * wrow);
    let per_row = 2 * (row + row);
    let rpt = if legacy {
        1
    } else {
        (avail / (n_threads.max(1) * per_row.max(1))).max(1)
    };

    kparams[0] = n_threads as i32;
    kparams[1] = 0; // col_tile: reductions never column-tile
    kparams[2] = rpt as i32;
    kparams[3] = rpt as i32; // block = (src0_bytes / 2) / row
    kparams[4] = has_weight as i32; // broadcast_weight

    let src0_bytes = row * rpt * 2;
    let dst_bytes = row * rpt * 2;
    let src1_bytes = wrow;

    kparams[5] = src0_bytes as i32;
    kparams[6] = src1_bytes as i32;
    kparams[7] = dst_bytes as i32;

    kparams[8] = (src0_bytes * n_threads) as i32;
    kparams[9] = (src1_bytes * n_threads) as i32;
    kparams[10] = (dst_bytes * n_threads) as i32;

    kparams[11] = row as i32;
    kparams[12] = wrow as i32;
    kparams[13] = row as i32;

    let total_vtcm = (src0_bytes + src1_bytes + dst_bytes) * n_threads;
    kparams[14] = total_vtcm as i32;

    // Row decomposition divs for ne = [ne0, nrows, 1, 1]; tpr = 1.
    let div_rows = if legacy {
        init_fastdiv(1)
    } else {
        init_fastdiv(nrows.max(1) as u32)
    };
    let div_1 = init_fastdiv(1);
    kparams[15] = div_rows.mp as i32;
    kparams[16] = div_rows.l as i32;
    kparams[17] = div_1.mp as i32;
    kparams[18] = div_1.l as i32;
    kparams[19] = div_rows.mp as i32;
    kparams[20] = div_rows.l as i32;
    kparams[21] = div_1.mp as i32;
    kparams[22] = div_1.l as i32;

    kparams
}

/// Which `get_rows` kernel runs (`enum htp_get_rows_kernel_type`).
const GET_ROWS_SAMETYPE: u32 = 0;
const GET_ROWS_TILED: u32 = 1;

/// `htp_mm_get_weight_tile_size`: bytes of one repacked weight tile, 0 for a type without one.
fn mm_weight_tile_size(dtype: u32) -> u64 {
    match dtype {
        2 | 20 => 576,  // Q4_0, IQ4_NL
        3 | 12 => 640,  // Q4_1, Q4_K
        8 => 1088,      // Q8_0
        13 => 768,      // Q5_K
        14 => 896,      // Q6_K
        11 | 10 => 512, // Q3_K, Q2_K
        39 => 544,      // MXFP4
        _ => 0,
    }
}

/// `htp_get_rows_vtcm_layout_build(...).total_bytes`.
fn get_rows_vtcm_total(kernel: u32, dtype: u32, ne00: u32, n_threads: u32) -> u64 {
    let align256 = |x: u64| (x + 255) & !255;
    let n = u64::from(n_threads);
    if kernel == GET_ROWS_SAMETYPE {
        return 0;
    }
    let dst_half = align256(u64::from(ne00) * 4);
    let src0_half = if kernel == GET_ROWS_TILED {
        let tile_stride = (mm_weight_tile_size(dtype) + 127) & !127;
        let n_k_tiles = u64::from(ne00 / 32);
        align256(if n_k_tiles > 0 {
            n_k_tiles * tile_stride
        } else {
            tile_stride
        })
    } else {
        let row = match dtype {
            x if x == HtpDataType::F16 as u32 => u64::from(ne00) * 2,
            x if x == HtpDataType::Q8_0 as u32 => u64::from(ne00 / 32) * 34,
            _ => 0,
        };
        align256(row)
    };
    2 * src0_half * n + 2 * dst_half * n
}

/// What `ggml_hexagon_precompute_get_rows_params` reads of its three tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GetRowsShape {
    /// `HtpDataType` of the table (`src0`) and of the output.
    pub src0_dtype: u32,
    pub dst_dtype: u32,
    /// The table is stored repacked (a Q4_0 table, or one the host repacks before the op).
    pub tiled: bool,
    pub ne00: u32,
    pub ne02: u32,
    pub ne03: u32,
    /// The I32 indices `[ne10, ne11, ne12]`.
    pub ne10: u32,
    pub ne11: u32,
    pub ne12: u32,
}

/// Kernel parameters for a `GetRows` gather: port of `ggml_hexagon_precompute_get_rows_params`
/// and `htp_get_rows_vtcm_layout_build`.
///
/// The kernel is `SAMETYPE` when table and output share a type, else `TILED` for a repacked
/// table, else `FLAT`. One task per gathered row; the thread count starts at
/// `min(n_threads, rows)` and drops until the VTCM working set fits `vtcm_bytes`.
///
/// A zero-thread result with pending tasks mirrors the C++ precompute byte for byte (the
/// goldens pin it); it is the caller's contract never to enqueue one. Dispatch sites skip
/// the empty gather instead of sending `n_threads = 0` across the FastRPC boundary.
///
/// Words (`htp_get_rows_kernel_params`): `n_threads, kernel_type, chunks_per_row, chunk_size,
/// total_tasks, tasks_per_thread, vtcm_size`, then the dividers `ne10, ne10 * ne11,
/// chunks_per_row, ne02, ne03` (each `mp, l`; `{0, 0}` for a zero divisor).
pub fn build_get_rows_kernel_params(
    shape: &GetRowsShape,
    n_threads: u32,
    vtcm_bytes: usize,
) -> [i32; 32] {
    let nr = shape.ne10.wrapping_mul(shape.ne11).wrapping_mul(shape.ne12);
    let kernel = if shape.src0_dtype == shape.dst_dtype {
        GET_ROWS_SAMETYPE
    } else if shape.tiled || shape.src0_dtype == HtpDataType::Q4_0 as u32 {
        // a Q4_0 table is always stored tiled; any other type only when the host repacked it
        GET_ROWS_TILED
    } else {
        2 // FLAT
    };
    let (chunks_per_row, chunk_size, total_tasks) = (1u32, shape.ne00, nr);

    let mut threads = n_threads.min(total_tasks);
    let mut total_bytes = 0u64;
    while threads > 0 {
        total_bytes = get_rows_vtcm_total(kernel, shape.src0_dtype, shape.ne00, threads);
        if total_bytes <= vtcm_bytes as u64 {
            break;
        }
        threads -= 1;
    }
    if threads == 0 && total_tasks > 0 {
        total_bytes = get_rows_vtcm_total(kernel, shape.src0_dtype, shape.ne00, 1);
    }
    let vtcm_size = if total_tasks == 0 { 0 } else { total_bytes };
    let tasks_per_thread = if threads > 0 {
        total_tasks.div_ceil(threads)
    } else {
        0
    };

    let mut k = [0i32; 32];
    k[0] = threads as i32;
    k[1] = kernel as i32;
    k[2] = chunks_per_row as i32;
    k[3] = chunk_size as i32;
    k[4] = total_tasks as i32;
    k[5] = tasks_per_thread as i32;
    k[6] = vtcm_size as u32 as i32;
    let div = |d: u32| {
        if d > 0 {
            init_fastdiv(d)
        } else {
            FastDivValues { mp: 0, l: 0 }
        }
    };
    let dividers = [
        div(shape.ne10),
        div(shape.ne10.wrapping_mul(shape.ne11)),
        div(chunks_per_row),
        div(shape.ne02),
        div(shape.ne03),
    ];
    for (i, f) in dividers.iter().enumerate() {
        k[7 + 2 * i] = f.mp as i32;
        k[8 + 2 * i] = f.l as i32;
    }
    k
}

/// [`build_get_rows_kernel_params`] for the gather cera issues: an F32 table into an F32
/// output of the same type, so the zero-VTCM `SAMETYPE` kernel. The arguments are the table's
/// `ne00, ne02, ne03` and the indices' `ne10, ne11, ne12`.
pub(crate) fn build_get_rows_f32_kernel_params(
    ne00: usize,
    ne02: usize,
    ne03: usize,
    ne10: usize,
    ne11: usize,
    ne12: usize,
    dsp_threads: u32,
) -> [i32; 32] {
    build_get_rows_kernel_params(
        &GetRowsShape {
            src0_dtype: HtpDataType::F32 as u32,
            dst_dtype: HtpDataType::F32 as u32,
            tiled: false,
            ne00: ne00 as u32,
            ne02: ne02 as u32,
            ne03: ne03 as u32,
            ne10: ne10 as u32,
            ne11: ne11 as u32,
            ne12: ne12 as u32,
        },
        dsp_threads.max(1),
        usize::MAX,
    )
}

/// `HTP_BINARY_KERNEL_CHUNKED`: the kernel the DSP runs for a contiguous
/// op whose `src1` is a single element (`ggml_hexagon_precompute_binary_params`'s
/// scalar-broadcast branch). A scalar `src1` under the same-shape kernel makes
/// the DSP read `src0`-many elements from it and fault.
const HTP_BINARY_KERNEL_CHUNKED: i32 = 7;

/// Host-computed kernel parameters for `dst[i] = src0[i] <op> scalar` over a
/// contiguous `total_elems`-element F32 vector (Mul by a routed-expert weight,
/// Div by the renormalization sum). Mirrors the llama.cpp host precompute:
/// chunk size from the thread count, VTCM for double-buffered `src0` and
/// `dst` chunks per thread plus one 128-byte slot for the scalar. `None` when
/// that does not fit `vtcm_size`.
pub(crate) fn build_binary_scalar_kernel_params(
    total_elems: usize,
    vtcm_size: usize,
    n_threads: u32,
) -> Option<[i32; 32]> {
    const ELEM: usize = 4;
    const MAX_CHUNK_BYTES: usize = 32768;
    const MIN_CHUNK_ELEMS: usize = 256;
    let n_threads = n_threads.max(1) as usize;

    let max_chunk_elems = MAX_CHUNK_BYTES / ELEM;
    let target = (total_elems.div_ceil(2 * n_threads)).next_multiple_of(32);
    let chunk_size = max_chunk_elems.min(target.max(MIN_CHUNK_ELEMS));
    let chunk_bytes = (chunk_size * ELEM).next_multiple_of(128);

    // src0 and dst: two chunks per thread each; the scalar takes one slot.
    let vtcm_total = n_threads * 2 * chunk_bytes + 128 + n_threads * 2 * chunk_bytes;
    if vtcm_total > vtcm_size {
        return None;
    }
    let row_aligned = (total_elems * ELEM).next_multiple_of(128);

    let mut kparams = [0i32; 32];
    kparams[0] = HTP_BINARY_KERNEL_CHUNKED;
    kparams[1] = n_threads as i32;
    kparams[2] = 1; // rows_per_buffer
    kparams[3] = row_aligned as i32; // src0_row_size_aligned
    kparams[4] = 0; // src1_row_size_aligned: the scalar needs no row
    kparams[5] = row_aligned as i32; // dst_row_size_aligned
    kparams[6] = 0; // src1_size
    kparams[7] = vtcm_total as i32;
    kparams[8] = chunk_size as i32;
    kparams[9] = chunk_bytes as i32;
    kparams[10] = 1; // is_scalar
    Some(kparams)
}

/// Host-computed kernel parameters for binary operations (Mul, Add, Sub, Div).
// Arity mirrors llama.cpp's fixed kparam builder signature; bundling into a
// struct would diverge from that C truth for no readability gain.
#[allow(clippy::too_many_arguments)]
pub fn build_binary_kernel_params(
    ne00: usize,
    ne10: usize,
    ne11: usize,
    ne12: usize,
    ne13: usize,
    elem_size: usize,
    vtcm_size: usize,
    n_threads: u32,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = n_threads.max(1);

    let src0_row_size = ne00 * elem_size;
    let src1_row_size = ne10 * elem_size;
    let dst_row_size = ne00 * elem_size;

    let src0_row_size_aligned = align128(src0_row_size);
    let src1_row_size_aligned = align128(src1_row_size);
    let dst_row_size_aligned = align128(dst_row_size);

    let is_row_bcast = ne11 == 1 && ne12 == 1 && ne13 == 1 && ne10 == ne00;
    // Single-row same-shape vectors take the ROW_BCAST path (kernel 1),
    // which preloads the one src1 row; SAME_SHAPE (kernel 0) with a nonzero
    // src1_size would under-allocate VTCM and corrupt results.
    let kernel_type = if is_row_bcast { 1u32 } else { 0u32 };
    let src1_size = if is_row_bcast {
        src1_row_size_aligned
    } else {
        0
    };

    let spad_row_total = 2
        * (src0_row_size_aligned
            + dst_row_size_aligned
            + if kernel_type == 0 {
                src1_row_size_aligned
            } else {
                0
            });
    let avail = vtcm_size.saturating_sub(src1_size);
    let denom = spad_row_total * n_threads as usize;
    let rows_per_buffer = avail.checked_div(denom).unwrap_or(0).max(1);
    let total_vtcm = rows_per_buffer * spad_row_total * n_threads as usize + src1_size;

    kparams[0] = kernel_type as i32;
    kparams[1] = n_threads as i32;
    kparams[2] = rows_per_buffer as i32;
    kparams[3] = src0_row_size_aligned as i32;
    kparams[4] = src1_row_size_aligned as i32;
    kparams[5] = dst_row_size_aligned as i32;
    kparams[6] = src1_size as i32;
    kparams[7] = total_vtcm.min(vtcm_size) as i32;

    kparams
}

/// `htp_binary_kernel_params::kernel_type` of the per-row scalar worker
/// (`HTP_BINARY_KERNEL_SCALAR_DMA`): `src1` is a contiguous `[1, ne11]` table that
/// the DSP stages in VTCM, and row `i` of `src0` is combined with `table[i % ne11]`.
pub(crate) const HTP_BINARY_KERNEL_SCALAR_DMA: i32 = 2;

/// Kernel parameters for a binary op whose second operand is a per-row scalar
/// table (`src1 = [1, ne11]`, contiguous), as `ggml_hexagon_precompute_binary_params`
/// builds them for `HTP_BINARY_KERNEL_SCALAR_DMA`.
///
/// `rows_per_buffer` is capped at 16 so the reported footprint stays small. The DSP
/// recomputes the real block size from its own VTCM size and ignores `kparams[2]` and
/// `kparams[7]` (they are reported values only), so what this checks is that the staged
/// `src1` table plus one double-buffered row per thread fits `vtcm_size`: `None` when it
/// does not, as for [`build_binary_scalar_kernel_params`], rather than a DSP
/// `VTCM_TOO_SMALL` at run time.
pub(crate) fn build_binary_scalar_dma_kernel_params(
    ne00: usize,
    ne11: usize,
    elem_size: usize,
    vtcm_size: usize,
    n_threads: u32,
) -> Option<[i32; 32]> {
    const MAX_ROWS_PER_BUFFER: usize = 16;
    let mut kparams = [0i32; 32];
    let n_threads = n_threads.max(1) as usize;

    let src0_row_size_aligned = align128(ne00 * elem_size);
    let src1_row_size_aligned = align128(elem_size);
    let dst_row_size_aligned = align128(ne00 * elem_size);
    let src1_size = align128(ne11 * elem_size);

    let spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    if src1_size + spad_row_total * n_threads > vtcm_size {
        return None;
    }
    let avail = vtcm_size - src1_size;
    let rows_per_buffer =
        (avail / (spad_row_total * n_threads).max(1)).clamp(1, MAX_ROWS_PER_BUFFER);
    let total_vtcm = rows_per_buffer * spad_row_total * n_threads + src1_size;

    kparams[0] = HTP_BINARY_KERNEL_SCALAR_DMA;
    kparams[1] = n_threads as i32;
    kparams[2] = rows_per_buffer as i32;
    kparams[3] = src0_row_size_aligned as i32;
    kparams[4] = src1_row_size_aligned as i32;
    kparams[5] = dst_row_size_aligned as i32;
    kparams[6] = src1_size as i32;
    kparams[7] = total_vtcm as i32;
    Some(kparams)
}

/// Host-computed parameters for RoPE positional embeddings.
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

/// Build kernel parameters for RoPE dispatch.
///
/// Mirrors `htp_rope_kernel_params` plus `htp_rope_vtcm_layout_build` (plain
/// RoPE, no YaRN scaling, no frequency factors).
pub fn build_rope_kernel_params(
    n_dims: usize,
    nrows: usize,
    ne1: usize,
    ne2: usize,
    n_threads: u32,
) -> [i32; 32] {
    build_rope_kernel_params_with_freq_factors(n_dims, nrows, ne1, ne2, n_threads, 0)
}

/// [`build_rope_kernel_params`] with the optional frequency-factor source (`lfm3`'s
/// partial rotary on its global layers): `n_freq_factors` is its length in `f32`s (`0` =
/// none). Its 256-aligned slot follows the per-thread slabs and the total grows by the
/// same amount.
pub fn build_rope_kernel_params_with_freq_factors(
    n_dims: usize,
    nrows: usize,
    ne1: usize,
    ne2: usize,
    n_threads: u32,
    n_freq_factors: usize,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = n_threads.max(1) as usize;
    let row_size = n_dims * 4;
    let row_aligned = align128(row_size);
    let theta_aligned = align256(row_size);
    // HTP_ROPE_SPAD_NROWS = HTP_ROPE_SPAD_BLOCK(8) * HTP_ROPE_SPAD_NSLOTS(4)
    let bytes_per_thread = theta_aligned + 32 * row_aligned;
    // the optional frequency-factor vector sits after the per-thread slabs
    let freq_factors_size = align256(n_freq_factors * 4);
    let total = bytes_per_thread * n_threads + freq_factors_size;

    kparams[0] = n_threads as i32;
    kparams[1] = nrows as i32;
    kparams[2] = nrows.div_ceil(n_threads) as i32;
    kparams[3] = total as i32;
    kparams[4] = bytes_per_thread as i32;
    kparams[5] = theta_aligned as i32;
    kparams[6] = row_aligned as i32;
    kparams[7] = (bytes_per_thread * n_threads) as i32;
    kparams[8] = freq_factors_size as i32;
    let div_ne2_ne1 = init_fastdiv((ne2 * ne1) as u32);
    kparams[9] = div_ne2_ne1.mp as i32;
    kparams[10] = div_ne2_ne1.l as i32;
    let div_ne1 = init_fastdiv(ne1 as u32);
    kparams[11] = div_ne1.mp as i32;
    kparams[12] = div_ne1.l as i32;
    kparams
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

/// `htp_fa_kernel_params.head_split`: 1 partitions flash attention by KV heads when it runs on
/// several threads, 0 by tokens. llama.cpp defaults it on (`opt_fa_head_split`).
const FA_HEAD_SPLIT: u32 = 1;

/// Build kernel parameters for Flash Attention dispatch (HVX path).
///
/// `n_tokens` query rows (1 for decode, M for prefill); `seq_len` total KV
/// length. Queries are `n_heads * n_tokens` head-rows; the mask stays
/// `[kv_len, n_tokens]` with unit dim-2/3 (causal rows via dim-1 stride).
// Arity mirrors llama.cpp's fixed kparam builder signature; see above.
#[allow(clippy::too_many_arguments)]
pub fn build_flash_attn_kernel_params(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    seq_len: usize,
    scale: f32,
    n_threads: u32,
    has_mask: bool,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = wire_byte(n_threads.max(1) as usize);
    let g = (n_heads / n_kv_heads.max(1)).max(1);
    let n_kv_blocks = seq_len.div_ceil(64).max(1);

    // byte 0: kernel_type = 1 (HTP_FA_KERNEL_HVX)
    // byte 1: head_split (llama.cpp's default: partition by KV heads across threads)
    // byte 2: flags (reserved, 0)
    // byte 3: n_threads
    // (These bytes used to be `is_q_fp32` / `is_dst_fp32`; the DSP now reads the tensor types.)
    let b0 = 1u32 | (FA_HEAD_SPLIT << 8) | ((n_threads as u32) << 24);
    kparams[0] = b0 as i32;

    // offset 4..8: Br: u16 = 1, Bc: u16 = 64
    let b1 = 1u32 | (64u32 << 16);
    kparams[1] = b1 as i32;

    // offset 8..12: n_kv_blocks: u16, G: u16
    let b2 = (n_kv_blocks as u32 & 0xffff) | ((g as u32 & 0xffff) << 16);
    kparams[2] = b2 as i32;

    // offset 12..16: scale: f32
    kparams[3] = scale.to_bits() as i32;

    // offset 16..20: max_bias: f32 = 0.0
    kparams[4] = 0;

    // offset 20..24: logit_softcap: f32 = 0.0
    kparams[5] = 0;

    // offset 24..28: vtcm_size: u32
    let size_q_row_padded = align128(head_dim * 4);
    let size_k_row_padded = align128(head_dim * 2);
    let size_v_row_padded = align128(head_dim * 2);
    let size_q_block = size_q_row_padded;
    let size_k_block = size_k_row_padded * 64;
    let size_v_block = size_v_row_padded * 64;
    let size_m_block = align128(64 * 2);
    let size_vkq_acc = align128(head_dim * 4);
    let size_per_thread = size_q_block
        + size_k_block * 2
        + size_v_block * 2
        + if has_mask { size_m_block * 128 } else { 0 }
        + size_vkq_acc;
    let vtcm_size = size_per_thread * (n_threads as usize);
    kparams[6] = vtcm_size as i32;

    // offset 28..32: qrows: u32 = n_heads * n_tokens (all query head-rows)
    let qrows = n_heads * n_tokens.max(1);
    kparams[7] = qrows as i32;

    // offset 32..36: qrows_per_thread: u32
    kparams[8] = qrows.div_ceil(n_threads as usize) as i32;

    // offset 36..40: qrow_start: u32 = 0 (single-batch decode starts at row 0)
    kparams[9] = 0;

    // offset 40..44: m0: f32 = 1.0
    kparams[10] = 1.0f32.to_bits() as i32;

    // offset 44..48: m1: f32 = 1.0
    kparams[11] = 1.0f32.to_bits() as i32;

    // offset 48..52: n_head_log2: u32
    let n_head_log2 = if n_heads > 0 {
        1u32 << (31 - (n_heads as u32).leading_zeros())
    } else {
        0
    };
    kparams[12] = n_head_log2 as i32;

    // offset 52..60: src3_div2: FastDivValues (mask)
    let div_1 = init_fastdiv(1);
    kparams[13] = div_1.mp as i32;
    kparams[14] = div_1.l as i32;

    // offset 60..68: src3_div3: FastDivValues (mask)
    kparams[15] = div_1.mp as i32;
    kparams[16] = div_1.l as i32;

    // offset 68..76: broadcast_rk2: FastDivValues (n_heads / n_kv_heads)
    let div_g = init_fastdiv(g as u32);
    kparams[17] = div_g.mp as i32;
    kparams[18] = div_g.l as i32;

    // offset 76..84: broadcast_rk3: FastDivValues (1)
    kparams[19] = div_1.mp as i32;
    kparams[20] = div_1.l as i32;

    // offset 84..92: broadcast_rv2: FastDivValues (g)
    kparams[21] = div_g.mp as i32;
    kparams[22] = div_g.l as i32;

    // offset 92..100: broadcast_rv3: FastDivValues (1)
    kparams[23] = div_1.mp as i32;
    kparams[24] = div_1.l as i32;

    // offset 100..104: u.hvx.size_q_row_padded
    kparams[25] = size_q_row_padded as i32;

    // offset 104..108: u.hvx.size_k_row_padded
    kparams[26] = size_k_row_padded as i32;

    // offset 108..112: u.hvx.size_v_row_padded
    kparams[27] = size_v_row_padded as i32;

    // offset 112..120: u.hvx.src0_div21: FastDivValues (n_heads * n_tokens)
    let div_qrows = init_fastdiv(qrows as u32);
    kparams[28] = div_qrows.mp as i32;
    kparams[29] = div_qrows.l as i32;

    // offset 120..128: u.hvx.src0_div1: FastDivValues (n_tokens)
    let div_n_tokens = init_fastdiv(n_tokens.max(1) as u32);
    kparams[30] = div_n_tokens.mp as i32;
    kparams[31] = div_n_tokens.l as i32;

    kparams
}

/// Same as [`build_flash_attn_kernel_params`] with attention logit soft-capping.
#[allow(clippy::too_many_arguments)]
pub fn build_flash_attn_kernel_params_with_softcap(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    seq_len: usize,
    scale: f32,
    n_threads: u32,
    has_mask: bool,
    softcap: f32,
) -> [i32; 32] {
    // The DSP computes `tanh(scale * s) * softcap`, so the host divides the scale by the softcap
    // first (`ggml_hexagon_precompute_flash_attn_params`).
    let scale = if softcap != 0.0 {
        scale / softcap
    } else {
        scale
    };
    let mut kparams = build_flash_attn_kernel_params(
        head_dim, n_heads, n_kv_heads, n_tokens, seq_len, scale, n_threads, has_mask,
    );
    kparams[5] = softcap.to_bits() as i32;
    kparams
}

/// HVX weight tile sizes (see `HTP_MM_WEIGHT_*_TILE_SIZE_*`).
fn mm_tile_sizes(wtype: HtpDataType) -> (u32, u32) {
    use super::repack::{TILE_SIZE_Q4_0, TILE_SIZE_Q4_K, TILE_SIZE_Q6_K, TILE_SIZE_Q8_0};
    match wtype {
        HtpDataType::Q4_0 => (TILE_SIZE_Q4_0 as u32, 640),
        // Q4_K repacks into the Q4_1 wire layout (640/640).
        HtpDataType::Q4K => (TILE_SIZE_Q4_K as u32, 640),
        HtpDataType::Q6K => (TILE_SIZE_Q6_K as u32, 896),
        HtpDataType::Q8_0 => (TILE_SIZE_Q8_0 as u32, 1152),
        // F16 weights are plain row-major rows with no tiling (upstream's
        // `htp_mm_get_weight_tile_size` / `_aligned_tile_size` return 0 for F16).
        HtpDataType::F16 => (0, 0),
        other => {
            // New weight type without a tile entry: fail loudly in debug
            // rather than silently emitting Q8_0-sized tiles.
            debug_assert!(false, "mm_tile_sizes: no tile entry for {other:?}");
            (TILE_SIZE_Q8_0 as u32, 1152)
        }
    }
}

/// Tiled activation row size for Q8_0 activations (see `htp_mm_q8_0_tiled_row_size`).
fn mm_q8_0_tiled_row_size(ne: usize) -> usize {
    let ne_padded = ne.next_multiple_of(128);
    (ne_padded / 32) * 1152
}

/// Tiled activation row size for Q8_1 activations (see `htp_mm_q8_1_tiled_row_size`).
fn mm_q8_1_tiled_row_size(ne: usize) -> usize {
    let ne_padded = ne.next_multiple_of(128);
    (ne_padded / 32) * 1280
}

/// Tiled activation row size for a weight type: Q8_1 for Q4_1/Q4_K weights,
/// Q8_0 otherwise (matches the `(wtype == Q4_1 || wtype == Q4_K)` gate in
/// every upstream matmul layout/precompute).
fn mm_act_tiled_row_size(wtype: HtpDataType, ne: usize) -> usize {
    if wtype == HtpDataType::Q4K {
        mm_q8_1_tiled_row_size(ne)
    } else {
        mm_q8_0_tiled_row_size(ne)
    }
}

/// HMX dim-1 weight stride (`nb[1]`): `ggml_hexagon_tiled_row_size`.
/// GGML row size for Q4_0/Q8_0, `(k/32) * (tile/32)` for K quants; both
/// spellings coincide since 576/32 = 18 and 1088/32 = 34.
pub fn mm_hmx_nb1(wtype: HtpDataType, k: usize) -> usize {
    if wtype == HtpDataType::F16 {
        return k * 2;
    }
    let (tile_size, _) = mm_tile_sizes(wtype);
    (k / 32) * (tile_size as usize / 32)
}

/// VTCM footprint of one HVX quantized matmul configuration.
struct MmHvxVtcmLayout {
    src0_bytes: usize,
    src1_bytes: usize,
    dst_bytes: usize,
    total_bytes: usize,
}

/// Mirror of `htp_mm_hvx_vtcm_layout_build` for the non-fused, non-ID
/// `HVX_QUANT_ROW` / `HVX_QUANT_BLOCK` path with repacked weights.
fn mm_hvx_vtcm_layout(
    wtype: HtpDataType,
    ne10: usize,
    src1_nrows: usize,
    n_threads: usize,
    dst_row_size: usize,
    n_prefetch: usize,
) -> MmHvxVtcmLayout {
    let round_up = |n: usize, m: usize| n.next_multiple_of(m);
    let q_src1_row = mm_act_tiled_row_size(wtype, ne10);
    let src1_bytes = round_up(q_src1_row * src1_nrows, 256);

    let (_, aligned_tile) = mm_tile_sizes(wtype);
    let tile_row = ne10.div_ceil(32) * aligned_tile as usize;
    let src0_bytes = round_up(n_prefetch * tile_row, 256) * n_threads;

    let quant_scratch = round_up(ne10 * 4, 128 * 4);
    let dst_nrows = usize::from(src1_nrows <= 1);
    let dst_slice = if dst_nrows > 0 && src1_nrows == 1 {
        round_up(dst_row_size.div_ceil(n_threads), 128)
    } else {
        0
    };
    let dst_bytes = dst_slice.max(quant_scratch) * n_threads;

    MmHvxVtcmLayout {
        src0_bytes,
        src1_bytes,
        dst_bytes,
        total_bytes: src0_bytes + src1_bytes + dst_bytes,
    }
}

/// VTCM the DSP needs for the non-fused HVX quantized-row kernel over `rows`
/// activation rows, as it rebuilds it in `htp_mm_hvx_vtcm_layout_build`: the
/// quantized activations, then the larger of (weight prefetch rows and, for a
/// single row, the output slice) and the raw f32 activation staging buffer,
/// which share the space after them. The DSP checks this against its VTCM and
/// answers `VtcmTooSmall` when it does not fit, so it is what decides the row
/// chunk.
fn mm_hvx_dsp_vtcm(
    wtype: HtpDataType,
    ne10: usize,
    rows: usize,
    n_threads: usize,
    dst_row_size: usize,
    n_prefetch: usize,
) -> usize {
    let round_up = |n: usize, m: usize| n.next_multiple_of(m);
    let src1 = round_up(mm_act_tiled_row_size(wtype, ne10) * rows, 256);
    let (_, aligned_tile) = mm_tile_sizes(wtype);
    let tile_row = ne10.div_ceil(32) * aligned_tile as usize;
    let src0 = round_up(n_prefetch * tile_row, 256) * n_threads;
    let dst = if rows == 1 {
        round_up(dst_row_size.div_ceil(n_threads), 128) * n_threads
    } else {
        0
    };
    let raw = round_up(round_up(ne10 * 4, 128) * rows, 128);
    src1 + (src0 + dst).max(raw)
}

/// The activation rows per chunk (`kparams.m_chunk`) for a quantized matmul
/// over `rows` rows, or `None` when all of them fit `vtcm_budget` at once:
/// the DSP's `htp_mm_hvx_solve_vtcm_params`. The kernel walks the rows in
/// chunks of this size; without one a projection whose quantized activations
/// (36 bytes per element) pass the VTCM, such as the 3072-wide FFN of a vision
/// transformer over a couple of hundred tokens, is rejected outright.
fn mm_hvx_m_chunk(
    wtype: HtpDataType,
    ne10: usize,
    rows: usize,
    n_threads: usize,
    dst_row_size: usize,
    n_prefetch: usize,
    vtcm_budget: usize,
) -> Option<usize> {
    let total = |m: usize| mm_hvx_dsp_vtcm(wtype, ne10, m, n_threads, dst_row_size, n_prefetch);
    if rows <= 1 || total(rows) <= vtcm_budget {
        return None;
    }
    // Rows that fit beside the fixed (weight) part, counting the raw staging
    // row the solver adds for the quantized kernels.
    let (_, aligned_tile) = mm_tile_sizes(wtype);
    let tile_row = ne10.div_ceil(32) * aligned_tile as usize;
    let fixed = (n_prefetch * tile_row).next_multiple_of(256) * n_threads;
    let eff_row = mm_act_tiled_row_size(wtype, ne10) + (ne10 * 4).next_multiple_of(128);
    let mut m = vtcm_budget.saturating_sub(fixed) / eff_row;
    if m > 1 {
        m &= !1;
    }
    let m = m.clamp(1, rows);
    // `eff_row` already counts the rounding slack of every term, so the chunk
    // fits; the sweep test pins that over the shapes in use.
    debug_assert!(m < 2 || total(m) <= vtcm_budget, "m_chunk {m} overruns");
    Some(m)
}

/// The most activation rows one fused `MulMatNx` (several weights sharing one
/// activation, such as a vision transformer's Q, K and V) can take in
/// `vtcm_budget`. That kernel has no row chunking, so a caller with more rows
/// splits the work (the three projections separately chunk themselves).
pub(crate) fn mm_hvx_fused_nx_max_rows(
    wtype: HtpDataType,
    ne10: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> usize {
    const N_PREFETCH: usize = 2;
    let round_up = |n: usize, m: usize| n.next_multiple_of(m);
    let n_threads = n_threads.max(1) as usize;
    let (_, aligned_tile) = mm_tile_sizes(wtype);
    let tile_row = ne10.div_ceil(32) * aligned_tile as usize;
    // `htp_mm_hvx_vtcm_layout_build`, `is_fused_nx`: no output slice, 128-byte
    // weight rounding.
    let src0 = round_up(N_PREFETCH * tile_row, 128) * n_threads;
    let row_q = mm_act_tiled_row_size(wtype, ne10);
    let raw = round_up(ne10 * 4, 128);
    let total = |m: usize| round_up(row_q * m, 128) + src0.max(round_up(raw * m, 128));
    let (mut lo, mut hi) = (0usize, vtcm_budget / row_q.max(1));
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if total(mid) <= vtcm_budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Build kernel parameters for matrix multiplication dispatch.
///
/// Mirrors `ggml_hexagon_precompute_hvx_mm_params` for the quantized HVX
/// decode path (repacked Q4_0/Q8_0 weights, non-batched). `ne10` is the
/// activation width (K), `ne11`/`ne12` the activation rows/batches (1 for
/// single-token decode), `dst_row_size` the output row stride in bytes.
///
/// The `m_chunk` field is non-zero when the activation rows do not fit
/// the VTCM at once; the kernel then walks them in chunks of that size, and
/// the `vtcm_*_size` fields describe one chunk's layout. The fused `MulMatNx` kernel
/// cannot chunk, so a caller building it must keep the rows within
/// `mm_hvx_fused_nx_max_rows` (which implies `m_chunk == 0`).
///
/// The words are the packed [`MmKernelParams`] layout, not the one the earlier skels read: decode
/// them with [`MmKernelParams::from_words`] instead of indexing.
pub fn build_mul_mat_kernel_params(
    wtype: HtpDataType,
    ne10: usize,
    ne11: u32,
    ne12: u32,
    dst_row_size: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> [i32; 32] {
    let n_threads = n_threads.max(1) as usize;
    let src1_nrows = (ne11 as usize) * (ne12 as usize);
    let (tile_size, aligned_tile_size) = mm_tile_sizes(wtype);

    // Pick the largest prefetch depth that fits the VTCM budget.
    let mut best = 2;
    let mut best_layout = None;
    let max_prefetch = if src1_nrows > 4 { 2 } else { 16 };
    let mut d = max_prefetch;
    while d >= 2 {
        let layout = mm_hvx_vtcm_layout(wtype, ne10, src1_nrows, n_threads, dst_row_size, d);
        if layout.total_bytes <= vtcm_budget {
            best = d;
            best_layout = Some(layout);
            break;
        }
        d /= 2;
    }
    // Rows the VTCM cannot hold at once are processed in chunks (the DSP walks
    // `m_chunk` rows at a time); the layout below is the chunk's.
    let m_chunk = mm_hvx_m_chunk(
        wtype,
        ne10,
        src1_nrows,
        n_threads,
        dst_row_size,
        best,
        vtcm_budget,
    );
    // `kparams[7..=11]` come from `mm_hvx_vtcm_layout`, the older model; the
    // chunk decision above uses `mm_hvx_dsp_vtcm`, a port of the DSP's solver.
    // They differ by the destination term, so for a few unchunked shapes the
    // reported `vtcm_size` is above the budget: the DSP rebuilds its own
    // layout from the other fields and rejects an overrun itself.
    let layout_rows = m_chunk.unwrap_or(src1_nrows);
    // The device rebuilds this layout and rejects loudly (VTCM_TOO_SMALL)
    // when it overruns real VTCM, so falling back to depth 2 here is safe.
    let layout = match (m_chunk, best_layout) {
        (None, Some(layout)) => layout,
        _ => mm_hvx_vtcm_layout(wtype, ne10, layout_rows, n_threads, dst_row_size, best),
    };
    let (div_ne12_ne11, div_ne11, div_r2, div_1) = (
        init_fastdiv(ne11 * ne12),
        init_fastdiv(ne11.max(1)),
        init_fastdiv(ne12.max(1)),
        init_fastdiv(1),
    );
    // Non-batched decode: ne02 = ne03 = ne13 = 1. div_n_act_threads and div_ne00_padded are HMX
    // only and stay zero.
    MmKernelParams {
        // fewer rows than threads takes the block-partitioned quant path
        kernel_type: if src1_nrows < n_threads { 6 } else { 5 },
        n_threads: wire_byte(n_threads),
        n_prefetch: wire_byte(best),
        m_chunk: m_chunk.map_or(0, |m| m as i32),
        tile_size: tile_size as i32,
        aligned_tile_size: aligned_tile_size as i32,
        act_row_size: mm_act_tiled_row_size(wtype, ne10) as i32,
        vtcm_size: layout.total_bytes as i32,
        vtcm_src0_size: layout.src0_bytes as i32,
        vtcm_act_size: layout.src1_bytes as i32,
        vtcm_dst_size: layout.dst_bytes as i32,
        div_ne12_ne1: div_ne12_ne11,
        div_ne1: div_ne11,
        div_r2,
        div_r3: div_1,
        div_ne12: div_r2,
        ..Default::default()
    }
    .to_words()
}

/// Shape of an F32 x F32 matmul, in the strides the host precompute reads.
///
/// `ne0x` are the weight-side (`src0`) dims, `ne1x` the activation-side
/// (`src1`) dims; `ne10 == ne00` is the contraction length. `src0_nb1` is
/// `src0`'s row stride in bytes and `dst_nb1` the destination's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulMatF32Shape {
    pub ne00: usize,
    pub ne02: usize,
    pub ne03: usize,
    pub src0_nb1: usize,
    pub ne11: usize,
    pub ne12: usize,
    pub ne13: usize,
    pub dst_nb1: usize,
}

/// VTCM bytes the HVX F32 x F32 kernel needs for `src1_nrows` activation rows
/// (`htp_mm_hvx_vtcm_layout_build`, `HTP_MM_KERNEL_HVX_F32_F32_VTCM`, no
/// fused or ID path): the activation rows, then the per-thread weight
/// prefetch rows and (single-row only) the output slice.
struct MmF32Layout {
    src0: usize,
    src1: usize,
    dst: usize,
    total: usize,
}

fn mm_f32_vtcm_layout(
    ne10: usize,
    src1_nrows: usize,
    n_threads: usize,
    dst_row_size: usize,
    src0_row_size: usize,
    n_prefetch: usize,
) -> MmF32Layout {
    let round_up = |n: usize, m: usize| n.next_multiple_of(m);
    let src0_row_padded = round_up(src0_row_size, 128);
    let src1 = round_up(round_up(ne10 * 4, 128) * src1_nrows, 256);
    let src0 = round_up(n_prefetch * src0_row_padded, 256) * n_threads;
    let dst = if src1_nrows > 1 {
        0
    } else {
        round_up(dst_row_size, 128) * n_threads
    };
    // Group A (the activations) then group B (weights and output); no raw
    // staging buffer on this path.
    MmF32Layout {
        src0,
        src1,
        dst,
        total: src1 + src0 + dst,
    }
}

/// Kernel parameters for an F32 x F32 `MulMat` on the HVX path
/// (`ggml_hexagon_precompute_hvx_mm_params`, F32 branch, plus the
/// `finalize` divisors): `dst[n, m] = sum_k src0[k, n] * src1[k, m]`, batched
/// over `ne02 x ne03` weight matrices that `ne12 x ne13` activation batches
/// are broadcast over (`ne12 / ne02`). `None` when even one activation row
/// does not fit `vtcm_budget` (the host falls back to the CPU then).
///
/// The words are the packed [`MmKernelParams`] layout, not the one the earlier skels read: decode
/// them with [`MmKernelParams::from_words`] instead of indexing.
pub fn build_mul_mat_f32_kernel_params(
    shape: MulMatF32Shape,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    const N_PREFETCH: usize = 16;
    const KERNEL_HVX_F32_F32_VTCM: i32 = 4;
    let MulMatF32Shape {
        ne00,
        ne02,
        ne03,
        src0_nb1,
        ne11,
        ne12,
        ne13,
        dst_nb1,
    } = shape;
    // The kernel divides by these (row indices and the weights' batch
    // broadcast): a zero extent, or a weight batch count that does not divide
    // the activations', has no valid parameters.
    if [ne00, ne02, ne03, ne11, ne12, ne13].contains(&0) || ne12 % ne02 != 0 || ne13 % ne03 != 0 {
        return None;
    }
    let n_threads = n_threads.max(1) as usize;
    let ne10 = ne00;
    let src1_nrows = ne11 * ne12 * ne13;
    let layout =
        |rows: usize| mm_f32_vtcm_layout(ne10, rows, n_threads, dst_nb1, src0_nb1, N_PREFETCH);

    // `htp_mm_hvx_solve_vtcm_params`: all activation rows at once when they
    // fit, otherwise the largest even chunk that does.
    let mut m_chunk = src1_nrows;
    let mut l = layout(m_chunk);
    if l.total > vtcm_budget {
        let fixed = l.src0 + l.dst;
        let avail = vtcm_budget.checked_sub(fixed).filter(|&a| a > 0)?;
        let row = (ne10 * 4).next_multiple_of(128);
        m_chunk = avail / row;
        if m_chunk > 1 {
            m_chunk &= !1;
        }
        m_chunk = m_chunk.min(src1_nrows);
        if m_chunk < 1 {
            return None;
        }
        l = layout(m_chunk);
        while m_chunk > 2 && l.total > vtcm_budget {
            m_chunk -= 2;
            l = layout(m_chunk);
        }
        if l.total > vtcm_budget {
            return None;
        }
    }

    let div = |d: usize| init_fastdiv(d as u32);
    Some(
        MmKernelParams {
            kernel_type: KERNEL_HVX_F32_F32_VTCM as u8,
            n_threads: wire_byte(n_threads),
            n_prefetch: N_PREFETCH as u8,
            m_chunk: if m_chunk < src1_nrows {
                m_chunk as i32
            } else {
                0
            },
            act_row_size: (ne10 * 4).next_multiple_of(128) as i32,
            vtcm_size: l.total as i32,
            vtcm_src0_size: l.src0 as i32,
            vtcm_act_size: l.src1 as i32,
            vtcm_dst_size: l.dst as i32,
            div_ne12_ne1: div(ne12 * ne11),
            div_ne1: div(ne11),
            div_r2: div(ne12 / ne02),
            div_r3: div(ne13 / ne03),
            div_ne12: div(ne12),
            ..Default::default()
        }
        .to_words(),
    )
}

/// Kernel parameters for `Softmax` over F32 rows
/// (`ggml_hexagon_precompute_softmax_params`):
/// `dst = softmax(scale * src0 + mask)` along dim 0, with an optional F32
/// `mask` of `[ne10, ne11, ne12, ne13]` broadcast over `src0`'s dims 2 and 3.
///
/// `mask_dims` is `[ne10, ne12, ne13]` of the mask when there is one.
pub fn build_softmax_kernel_params(
    src0_ne: [usize; 4],
    mask_dims: Option<[usize; 3]>,
    scale: f32,
    n_threads: u32,
) -> [i32; 32] {
    const KERNEL_NOMASK: i32 = 0;
    const KERNEL_MASK_F32: i32 = 1;
    let round_up = |n: usize, m: usize| n.next_multiple_of(m);
    let [ne00, ne01, ne02, ne03] = src0_ne;
    let src0_nrows = ne01 * ne02 * ne03;
    let n_threads = (n_threads.max(1) as usize).min(src0_nrows.max(1));
    let use_src1 = mask_dims.is_some();
    let ne10 = mask_dims.map_or(1, |m| m[0]);

    let src0_row = round_up(ne00 * 4, 128);
    let dst_row = round_up(ne00 * 4, 128);
    let src1_row = if use_src1 { round_up(ne10 * 4, 128) } else { 0 };
    // Double-buffered half-buffers per thread (`htp_softmax_vtcm_layout_build`).
    let (src0_pt, dst_pt, src1_pt) = (src0_row * 2, dst_row * 2, src1_row * 2);
    let total = (src0_pt + dst_pt + src1_pt) * n_threads;

    let n_head = ne02;
    let n_head_log2 = if n_head > 0 {
        1usize << n_head.ilog2()
    } else {
        0
    };

    let mut k = [0i32; 32];
    k[0] = n_threads as i32;
    k[1] = src0_nrows as i32;
    k[2] = src0_nrows.div_ceil(n_threads) as i32;
    k[3] = total as i32;
    k[4] = src0_pt as i32;
    k[5] = src1_pt as i32;
    k[6] = dst_pt as i32;
    k[7] = src0_row as i32; // src0_row_size_aligned
    k[8] = src1_row as i32;
    k[9] = dst_row as i32;
    k[10] = src0_row as i32; // src0_spad_half_size
    k[11] = src1_row as i32;
    k[12] = dst_row as i32;
    k[13] = n_head as i32;
    k[14] = n_head_log2 as i32;
    k[15] = use_src1 as i32;
    k[16] = 0; // use_f16: the mask is F32 here
    k[17] = if use_src1 {
        KERNEL_MASK_F32
    } else {
        KERNEL_NOMASK
    };
    k[18] = scale.to_bits() as i32;
    k[19] = 0f32.to_bits() as i32; // max_bias: no ALiBi
    k[20] = 1f32.to_bits() as i32; // m0
    k[21] = 1f32.to_bits() as i32; // m1
    let [_, ne12, ne13] = mask_dims.unwrap_or([1, 1, 1]);
    for (slot, d) in [(22, ne01), (24, ne02), (26, ne12), (28, ne13)] {
        if d > 0 {
            let f = init_fastdiv(d as u32);
            k[slot] = f.mp as i32;
            k[slot + 1] = f.l as i32;
        }
    }
    k
}

/// Minimum M rows for HMX matmul. Upstream `HTP_MM_HMX_MIN_NROWS` is 4,
/// but the HMX worker is nondeterministic for M < 8 on-device (8-row
/// microtile overhang; measured: M <= 7 racy, M >= 8 bitwise stable and
/// at CPU parity), so small-M prefills take the HVX kernel.
pub const HMX_MM_MIN_NROWS: usize = 8;
/// HMX tile edge (`HTP_MM_HMX_TILE_N_ROWS/COLS`).
const HMX_TILE: usize = 32;
/// HMX tile bytes, also the VTCM group alignment (`HTP_MM_HMX_TILE_SIZE`).
const HMX_TILE_BYTES: usize = 2048;
/// Solver cost weights (`HTP_MM_HMX_COST_*`).
const HMX_COST_W_DEQUANT: usize = 3;
const HMX_COST_A_CONVERT: usize = 2;
/// Activation DMA multiplier (`HTP_MM_DMA_ACT_MULTIPLIER` = 2 * ROWS_PER_STEP).
const HMX_DMA_ACT_MULT: usize = 4;
/// Tiled K quantum for the weight row stride (`QK_Q4_0_TILED`).
const QK_TILED: usize = 256;

/// HMX weight types we emit (`ggml_hexagon_is_hmx_weight_type`, repack subset plus F16).
pub fn mm_is_hmx_eligible(wtype: HtpDataType, k: usize, n: usize, m: usize) -> bool {
    matches!(
        wtype,
        HtpDataType::Q4_0
            | HtpDataType::Q8_0
            | HtpDataType::Q4K
            | HtpDataType::Q6K
            | HtpDataType::F16
    ) && k > 0
        && n > 0
        && k.is_multiple_of(HMX_TILE)
        && n.is_multiple_of(HMX_TILE)
        && m >= HMX_MM_MIN_NROWS
}

/// Double-buffered HMX execution needs M > 32 (`htp_mm_hmx_pipeline`).
fn mm_hmx_pipeline(m: usize) -> bool {
    m > 32
}

/// Padded weight bytes per N row (`htp_mm_get_tiled_row_stride`).
fn mm_tiled_row_stride(wtype: HtpDataType, k: usize) -> usize {
    match wtype {
        HtpDataType::F16 => k * 2,
        HtpDataType::F32 => k * 4,
        _ => k.div_ceil(QK_TILED) * mm_tile_sizes(wtype).0 as usize,
    }
}

/// VTCM cost model for 2D HMX (`htp_mm_hmx_get_2d_chunk_costs`): bytes per
/// N chunk row, per M chunk row, and per M*N output element.
fn mm_hmx_chunk_costs_2d(
    wtype: HtpDataType,
    k: usize,
    pipeline: bool,
    aligned_tile: usize,
) -> (usize, usize, usize) {
    let is_quant = !matches!(wtype, HtpDataType::F16 | HtpDataType::F32);
    let row_stride = mm_tiled_row_stride(wtype, k);
    let vec_dot = k * 2;
    let qw_row = if is_quant {
        (k / HMX_TILE) * aligned_tile / HMX_TILE
    } else {
        0
    };
    let per_n = (if pipeline { 2 } else { 1 }) * (if is_quant { qw_row } else { row_stride })
        + (if pipeline { 2 * vec_dot } else { vec_dot });
    let per_mn = (if pipeline { 2 } else { 1 }) * 2;
    (per_n, vec_dot, per_mn)
}

/// Fixed VTCM overhead for 2D HMX (`htp_mm_hmx_get_2d_overhead`; non-ID).
fn mm_hmx_overhead_2d(pipeline: bool) -> usize {
    (if pipeline { 7 } else { 5 }) * HMX_TILE_BYTES + 256
}

/// Cost-model (M, N) chunk search (`htp_mm_hmx_compute_chunks`). Returns the
/// `(m_chunk, n_chunk)` minimizing reload cost; `None` on overflow or when
/// nothing fits. The C candidate total is unused by the 2D solver (it
/// re-checks the exact layout), so only the chunks are returned.
#[allow(clippy::too_many_arguments)]
fn mm_hmx_compute_chunks(
    vtcm_total: usize,
    overhead: usize,
    per_n: usize,
    per_m: usize,
    per_mn: usize,
    m: usize,
    n: usize,
    m_block_cost: usize,
    n_block_cost: usize,
) -> Option<(usize, usize)> {
    if m == 0 || n == 0 || vtcm_total <= overhead {
        return None;
    }
    if per_n == 0 || per_m == 0 || per_mn == 0 {
        return None;
    }
    let usable = vtcm_total - overhead;
    let mut best_cost = usize::MAX;
    let mut best_tail_waste = usize::MAX;
    let mut best_mn = 0;
    let mut best = (0, 0);
    let n_max = (n.min(usable / per_n) / HMX_TILE) * HMX_TILE;
    let mut nc = n_max;
    while nc >= HMX_TILE {
        let accept = || -> Option<(usize, usize, usize, usize)> {
            let n_fixed = nc.checked_mul(per_n)?;
            if n_fixed >= usable {
                return None;
            }
            let mc_denom = per_m.checked_add(nc.checked_mul(per_mn)?)?;
            if mc_denom == 0 {
                return None;
            }
            let mc = (((usable - n_fixed) / mc_denom) / HMX_TILE * HMX_TILE).min(m);
            if mc == 0 {
                return None;
            }
            let mblocks = m.div_ceil(mc);
            let nblocks = n.div_ceil(nc);
            let cost = mblocks
                .checked_mul(m_block_cost)?
                .checked_add(nblocks.checked_mul(n_block_cost)?)?;
            Some((mc, nc, cost, mc.checked_mul(nc)?))
        };
        if let Some((mc, nc, cost, mn)) = accept() {
            // Equal cost: prefer the chunk that leaves the fewest columns idle in the last N
            // block, then the larger one (llama.cpp's tie-break).
            let rem = n % nc;
            let tail_waste = if rem == 0 { 0 } else { nc - rem };
            if cost < best_cost
                || (cost == best_cost && tail_waste < best_tail_waste)
                || (cost == best_cost && tail_waste == best_tail_waste && mn > best_mn)
            {
                best_cost = cost;
                best_tail_waste = tail_waste;
                best_mn = mn;
                best = (mc, nc);
            }
        }
        if nc == HMX_TILE {
            break;
        }
        nc -= HMX_TILE;
    }
    if best == (0, 0) { None } else { Some(best) }
}

/// Exact 2D HMX VTCM footprint (`htp_mm_hmx_vtcm_layout_build`, `HMX_2D`
/// branch; unfused, so `src2_size` is 0).
fn mm_hmx_layout_2d_total(
    wtype: HtpDataType,
    k: usize,
    mc: usize,
    nc: usize,
    pipeline: bool,
    act_threads: usize,
    aligned_tile: usize,
) -> usize {
    let is_quant = !matches!(wtype, HtpDataType::F16 | HtpDataType::F32);
    let vec_dot = k * 2;
    let min_f32 = (act_threads * HMX_DMA_ACT_MULT * k * 4).next_multiple_of(128);
    let weight_area = if is_quant {
        ((nc / HMX_TILE) * (k / HMX_TILE) * aligned_tile).next_multiple_of(HMX_TILE_BYTES)
    } else {
        (nc * mm_tiled_row_stride(wtype, k)).next_multiple_of(HMX_TILE_BYTES)
    };
    let act_area = (mc * vec_dot).next_multiple_of(HMX_TILE_BYTES);
    let out_area = (mc * nc * 2).next_multiple_of(HMX_TILE_BYTES);
    let scratch0 = (nc * vec_dot).next_multiple_of(HMX_TILE_BYTES);
    // Group A: scales + activation tiles (never overlaps B/C).
    let off_a = HMX_TILE_BYTES + act_area;
    // Group B: compute buffers.
    let mut b = off_a + weight_area;
    if pipeline {
        b += weight_area;
    }
    b += out_area + scratch0;
    if pipeline {
        b += scratch0 + out_area;
    }
    let group_b = b - off_a;
    // Group C: activation prep scratch, overlaps B.
    let max_f32 = act_threads * 64 * k * 4;
    let group_c = max_f32.min(min_f32.max(group_b)).next_multiple_of(128);
    off_a + group_b.max(group_c)
}

/// 2D HMX chunk solver (`htp_mm_hmx_solve_2d_params`, non-ID, unfused).
/// `n` is N padded to 32, `m_pad` M padded to 32, `m` raw rows.
/// Returns `(m_chunk, n_chunk, act_threads, vtcm_bytes)`.
pub fn mm_hmx_solve_2d(
    wtype: HtpDataType,
    k: usize,
    n: usize,
    m_pad: usize,
    m: usize,
    n_threads: usize,
    vtcm_budget: usize,
) -> Option<(usize, usize, usize, usize)> {
    let pipeline = mm_hmx_pipeline(m);
    let aligned_tile = mm_tile_sizes(wtype).1 as usize;
    let (per_n, per_m, per_mn) = mm_hmx_chunk_costs_2d(wtype, k, pipeline, aligned_tile);
    let overhead = mm_hmx_overhead_2d(pipeline);
    let mut best: Option<(usize, usize, usize, usize, usize)> = None;
    let mut act_threads = n_threads.max(1);
    loop {
        if let Some((mc, nc)) = mm_hmx_compute_chunks(
            vtcm_budget,
            overhead,
            per_n,
            per_m,
            per_mn,
            m_pad,
            n,
            n * HMX_COST_W_DEQUANT,
            m * HMX_COST_A_CONVERT,
        ) {
            let exact =
                mm_hmx_layout_2d_total(wtype, k, mc, nc, pipeline, act_threads, aligned_tile);
            if exact <= vtcm_budget {
                let mblocks = m.div_ceil(mc);
                let better = best.is_none_or(|(bm, bat, _, _, _)| {
                    mblocks < bm || (mblocks == bm && act_threads > bat)
                });
                if better {
                    best = Some((mblocks, act_threads, mc, nc, exact));
                }
            }
        }
        if act_threads == 1 {
            break;
        }
        act_threads /= 2;
    }
    best.map(|(_, at, mc, nc, vs)| (mc, nc, at, vs))
}

/// HMX matmul kernel parameters.
///
/// Mirrors `ggml_hexagon_precompute_hmx_mm_params` (`HMX_2D`, non-ID,
/// unfused) plus the shared `finalize` divs. `k`/`n` are padded to 32,
/// `m_pad` is M padded to 32, `m` raw rows. Returns `None` when no chunking
/// fits `vtcm_budget` (caller falls back to HVX).
///
/// The words are the packed [`MmKernelParams`] layout, not the one the earlier skels read: decode
/// them with [`MmKernelParams::from_words`] instead of indexing.
pub fn build_hmx_mm_kernel_params(
    wtype: HtpDataType,
    k: usize,
    n: usize,
    m_pad: usize,
    m: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    let (mc, nc, act_threads, vtcm) = mm_hmx_solve_2d(
        wtype,
        k,
        n,
        m_pad,
        m,
        n_threads.max(1) as usize,
        vtcm_budget,
    )?;
    let (tile_size, aligned_tile_size) = mm_tile_sizes(wtype);
    // Shared finalize dividers for flat [K, M] activations (ne12 = ne02 = 1).
    let (div_m, div_1) = (init_fastdiv(m as u32), init_fastdiv(1));
    Some(
        MmKernelParams {
            kernel_type: 1, // HTP_MM_KERNEL_HMX_2D
            pipeline: mm_hmx_pipeline(m) as u8,
            n_hmx: 1,
            n_threads: wire_byte(n_threads.max(1) as usize),
            n_act_threads: wire_byte(act_threads),
            m_chunk: mc as i32,
            n_chunk: nc as i32,
            tile_size: tile_size as i32,
            aligned_tile_size: aligned_tile_size as i32,
            act_row_size: mm_act_tiled_row_size(wtype, k) as i32,
            vtcm_size: vtcm as i32,
            // the src0/act/bias/dst scratch sizes stay zero outside the fused NX kernel
            div_ne12_ne1: div_m,
            div_ne1: div_m,
            div_r2: div_1,
            div_r3: div_1,
            div_ne12: div_1,
            div_n_act_threads: init_fastdiv(act_threads as u32),
            div_ne00_padded: init_fastdiv(k as u32),
            ..Default::default()
        }
        .to_words(),
    )
}

/// HMX flash-attention VTCM group alignment (`HTP_FA_HMX_TILE_SIZE`).
const FA_HMX_TILE_BYTES: usize = 2048;
/// HMX FA tile rows (`HMX_FP16_TILE_N_ROWS`).
const FA_TILE_ROWS: usize = 32;
/// Minimum KV blocks for pipelining (`FA_MIN_KV_BLOCKS`).
const FA_MIN_KV_BLOCKS: usize = 3;
/// HMX FA mask DMA cache slots (`HMX_FA_DMA_CACHE_SIZE`; HVX uses 128).
const FA_HMX_DMA_CACHE: usize = 4;
/// Solver cost weights (calibrated from profiling).
const FA_C_Q_FIXED: usize = 800;
const FA_C_ITER_BASE: usize = 200;
const FA_C_SOFTMAX: usize = 600;

/// HMX flash-attention eligibility (`ggml_hexagon_flash_attn_is_hmx_eligible`;
/// f16/Q8_0 KV and HMX-present gating done by the caller).
pub fn fa_is_hmx_eligible(head_dim: usize, n_tokens: usize) -> bool {
    head_dim.is_multiple_of(8) && !(head_dim <= 128 && n_tokens < 5)
}

/// Exact HMX FA VTCM footprint (`hmx_fa_vtcm_layout_build` + `..._usage`).
/// `dk`/`dv` are padded to 64 by the caller.
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
    let d_tile = (g_br / FA_TILE_ROWS) * FA_HMX_TILE_BYTES;
    let q_dma = (g_br * dk * if is_q_fp32 { 4 } else { 2 }).next_multiple_of(128);
    let k_dma = (bc * (dk * 2).next_multiple_of(128)).next_multiple_of(128);
    let v_dma = (bc * (dv * 2).next_multiple_of(128)).next_multiple_of(128);
    let col_vec = (g_br * 4).next_multiple_of(256);
    let row_vec = (bc * 2).next_multiple_of(256);
    let m_line = (bc * 2).next_multiple_of(128);
    let m_buf = ((br * m_line).next_multiple_of(256)) * FA_HMX_DMA_CACHE;
    let slopes = (g_br * 2).next_multiple_of(128);
    let sinks = if has_sinks {
        (n_heads * 4).next_multiple_of(128)
    } else {
        0
    };
    // Group A part 1: HMX-tiled buffers.
    let mut off = q_tile + o_tile * 2 + d_tile + d_tile;
    if pipeline {
        off += d_tile;
    }
    // Groups B and C share a 2KB-aligned base.
    let base = off.next_multiple_of(FA_HMX_TILE_BYTES);
    let mut b = base + k_tile + v_tile + s_tile * 2 + col_vec * 2 + row_vec * 2 * n_threads;
    if pipeline {
        b += k_tile + v_tile + s_tile * 2;
    }
    let group_b = b - base;
    let group_c = q_dma;
    off = base + group_b.max(group_c);
    // Group A part 2: DMA + vector buffers.
    off += k_dma * 2 + v_dma * 2 + col_vec * 2 + 256 + 256 + m_buf + slopes + sinks;
    off
}

/// HMX FA (Br, Bc) solver (`hmx_fa_find_chunk_size`). `dk`/`dv` padded to 64.
#[allow(clippy::too_many_arguments)]
pub fn fa_hmx_find_chunk_size(
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
) -> Option<(usize, usize)> {
    let br_unit = FA_TILE_ROWS.div_ceil(g.max(1));
    let bc_unit = 64;
    let can_pipeline = kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2;
    let br_max = if qo_len >= br_unit {
        qo_len / br_unit * br_unit
    } else {
        br_unit
    };
    let bc_limit = if can_pipeline {
        kv_len / FA_MIN_KV_BLOCKS / bc_unit * bc_unit
    } else if kv_len >= bc_unit {
        kv_len / bc_unit * bc_unit
    } else {
        bc_unit
    };
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
                let cost =
                    q_blocks * (FA_C_Q_FIXED + kv_blocks * (FA_C_ITER_BASE + FA_C_SOFTMAX * vpt));
                let mn = br * bc;
                if cost < best_cost || (cost == best_cost && mn > best_mn) {
                    best_cost = cost;
                    best_mn = mn;
                    best = (br, bc);
                }
                // Bc descends: first fit is the largest for this Br.
                break;
            }
            bc -= bc_unit;
        }
        if br == br_unit {
            break;
        }
        br -= br_unit;
    }
    if best == (0, 0) { None } else { Some(best) }
}

/// HMX flash-attention kernel parameters.
///
/// Mirrors the HMX branch of `ggml_hexagon_precompute_flash_attn_params`
/// (f32 Q/dst, f16 KV + mask, no sinks/slopes beyond defaults). Returns
/// `None` when no (Br, Bc) fits `vtcm_budget` (caller falls back to HVX).
#[allow(clippy::too_many_arguments)]
pub fn build_hmx_fa_kernel_params(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    seq_len: usize,
    scale: f32,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    build_hmx_fa_kernel_params_impl(
        head_dim,
        n_heads,
        n_kv_heads,
        n_tokens,
        seq_len,
        scale,
        0.0,
        n_threads,
        vtcm_budget,
    )
}

/// The HMX kernel works in the log2 domain, so the host folds log2(e) into the softmax scale
/// (no softcap) or into the softcap (when there is one), as llama.cpp's host does: with softcap 0,
/// `scale * log2(e)`; otherwise `scale / softcap` with `softcap * log2(e)`.
fn hmx_fa_scale_words(scale: f32, softcap: f32) -> (i32, i32) {
    use std::f32::consts::LOG2_E;
    if softcap == 0.0 {
        ((scale * LOG2_E).to_bits() as i32, 0)
    } else {
        (
            (scale / softcap).to_bits() as i32,
            (softcap * LOG2_E).to_bits() as i32,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn build_hmx_fa_kernel_params_impl(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    seq_len: usize,
    scale: f32,
    softcap: f32,
    n_threads: u32,
    vtcm_budget: usize,
) -> Option<[i32; 32]> {
    let g = (n_heads / n_kv_heads.max(1)).max(1);
    let dk = head_dim.next_multiple_of(64);
    let dv = dk;
    let nt = n_threads.max(1) as usize;
    let (br, bc) = fa_hmx_find_chunk_size(
        g,
        dk,
        dv,
        n_tokens,
        seq_len,
        vtcm_budget,
        nt,
        true,
        false,
        n_heads,
    )?;
    let n_kv_blocks = seq_len.div_ceil(bc);
    let pipelined = n_kv_blocks >= FA_MIN_KV_BLOCKS && nt >= 2;
    let kt = if pipelined { nt } else { 1 };
    let mut kparams = [0i32; 32];
    kparams[0] = (2u32 | (FA_HEAD_SPLIT << 8) | ((wire_byte(kt) as u32) << 24)) as i32;
    kparams[1] = (br as u32 | ((bc as u32) << 16)) as i32;
    kparams[2] = ((n_kv_blocks as u32 & 0xffff) | ((g as u32 & 0xffff) << 16)) as i32;
    let (scale_word, softcap_word) = hmx_fa_scale_words(scale, softcap);
    kparams[3] = scale_word;
    // [4] max_bias: zero.
    kparams[5] = softcap_word;
    kparams[6] = fa_hmx_layout_total(g, dk, dv, br, bc, kt, pipelined, true, false, n_heads) as i32;
    // [7] qrows, [8] qrows_per_thread, [9] qrow_start: zero for HMX.
    kparams[10] = 1.0f32.to_bits() as i32;
    kparams[11] = 1.0f32.to_bits() as i32;
    kparams[12] = if n_heads > 0 {
        (1u32 << (31 - (n_heads as u32).leading_zeros())) as i32
    } else {
        0
    };
    // Mask is [seq, tokens, 1, 1]: src3 divs over unit dims.
    let div_1 = init_fastdiv(1);
    kparams[13] = div_1.mp as i32;
    kparams[14] = div_1.l as i32;
    kparams[15] = div_1.mp as i32;
    kparams[16] = div_1.l as i32;
    // [17..=24] broadcast divs: unset (zero) on the HMX path.
    // HMX union: g_br, row_buf_stride, mask_buf_row_stride, mask_broadcast,
    // pipeline, div_G.
    kparams[25] = ((g * br).next_multiple_of(FA_TILE_ROWS)) as i32;
    kparams[26] = ((bc * 2).next_multiple_of(256) / 128) as i32;
    kparams[27] = ((bc * 2).next_multiple_of(128) / 2) as i32;
    kparams[28] = 1; // mask_broadcast: mask->ne[2] == 1
    kparams[29] = pipelined as i32;
    let div_g = init_fastdiv(g as u32);
    kparams[30] = div_g.mp as i32;
    kparams[31] = div_g.l as i32;
    Some(kparams)
}

/// Same as [`build_hmx_fa_kernel_params`] with attention logit soft-capping.
#[allow(clippy::too_many_arguments)]
pub fn build_hmx_fa_kernel_params_with_softcap(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    seq_len: usize,
    scale: f32,
    n_threads: u32,
    vtcm_budget: usize,
    softcap: f32,
) -> Option<[i32; 32]> {
    build_hmx_fa_kernel_params_impl(
        head_dim,
        n_heads,
        n_kv_heads,
        n_tokens,
        seq_len,
        scale,
        softcap,
        n_threads,
        vtcm_budget,
    )
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

/// `hex_round_up` (`hex-common.h`): 32-bit, so an operand past 4 GiB wraps like the DSP header.
fn hex_round_up(n: u32, m: u32) -> u32 {
    m.wrapping_mul(n.wrapping_add(m).wrapping_sub(1) / m)
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

    /// Every case of `testdata/hmx_solve_golden.txt`: llama.cpp's own `htp_mm_hmx_solve_2d_params`
    /// (cost model, exact layout check and activation-thread search) for the weight types and
    /// shapes cera sends, at several thread counts and VTCM budgets.
    #[test]
    fn hmx_2d_solver_matches_the_llama_cpp_header() {
        let golden = include_str!("testdata/hmx_solve_golden.txt");
        let (mut cases, mut no_fit, mut halved) = (0usize, 0usize, 0usize);
        for line in golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let (inputs, want) = line.split_once("=>").unwrap();
            let v: Vec<usize> = inputs
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let w: Vec<usize> = want
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let wtype = match v[0] {
                2 => HtpDataType::Q4_0,
                8 => HtpDataType::Q8_0,
                12 => HtpDataType::Q4K,
                14 => HtpDataType::Q6K,
                other => panic!("unexpected weight type {other}"),
            };
            // golden fields: wtype k n m_pad m threads vtcm
            let got = mm_hmx_solve_2d(wtype, v[1], v[2], v[3], v[4], v[5], v[6]);
            let want = (w[0] == 1).then_some((w[1], w[2], w[3], w[4]));
            assert_eq!(got, want, "{line}");
            cases += 1;
            match want {
                None => no_fit += 1,
                Some((_, _, act_threads, _)) if act_threads < v[5] => halved += 1,
                Some(_) => {}
            }
        }
        assert!(cases > 6000, "golden file looks truncated: {cases} cases");
        // The grid has to keep reaching the paths a roomy budget never takes.
        assert!(no_fit > 1000, "only {no_fit} cases do not fit");
        assert!(
            halved > 100,
            "only {halved} cases halve the activation threads"
        );
    }

    /// Every case of `testdata/hmx_chunks_golden.txt`: llama.cpp's own `htp_mm_hmx_compute_chunks`
    /// (`matmul-ops.h`) over a grid of VTCM budgets, costs and ragged N, whose tie-break prefers
    /// the candidate that wastes the fewest columns in the last N block.
    #[test]
    fn hmx_chunk_search_matches_the_llama_cpp_header() {
        let golden = include_str!("testdata/hmx_chunks_golden.txt");
        let (mut cases, mut fits) = (0usize, 0usize);
        for line in golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let (inputs, want) = line.split_once("=>").unwrap();
            let v: Vec<usize> = inputs
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let want: Vec<i64> = want
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let got = mm_hmx_compute_chunks(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
            if want[0] == 0 {
                assert_eq!(got, Some((want[1] as usize, want[2] as usize)), "{line}");
                fits += 1;
            } else {
                assert_eq!(got, None, "{line}");
            }
            cases += 1;
        }
        assert!(
            cases > 17000 && fits > 15000,
            "golden file looks truncated: {cases} cases"
        );
    }

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

    /// A per-head attention-score matmul worked out from the host precompute:
    /// K as `[64, 126, 8]` (row stride 2048 B) against 8 batches of 126 query
    /// rows. 1008 activation rows of 256 B, 16 prefetched 2048 B weight rows
    /// per thread, no output slice, all in one chunk.
    #[test]
    fn f32_matmul_params_follow_the_host_precompute() {
        let k = build_mul_mat_f32_kernel_params(
            MulMatF32Shape {
                ne00: 64,
                ne02: 8,
                ne03: 1,
                src0_nb1: 2048,
                ne11: 126,
                ne12: 8,
                ne13: 1,
                dst_nb1: 126 * 4,
            },
            8,
            8 << 20,
        )
        .unwrap();
        let m = MmKernelParams::from_words(&k);
        assert_eq!(m.kernel_type, 4, "HVX_F32_F32_VTCM");
        assert_eq!((m.n_threads, m.n_hmx, m.n_prefetch), (8, 0, 16));
        assert_eq!(m.m_chunk, 0, "everything fits: no chunking");
        assert_eq!(m.act_row_size, 256); // round_up(64 * 4, 128) is 256
        assert_eq!(m.vtcm_act_size, 258048); // round_up(256 * 1008, 256)
        assert_eq!(m.vtcm_src0_size, 262144); // round_up(16 * 2048, 256) * 8 threads
        assert_eq!(m.vtcm_dst_size, 0, "no output slice for more than one row");
        assert_eq!(m.vtcm_size, 258048 + 262144);
        // Dividers: ne12 * ne11, ne11, ne12 / ne02 = 1, ne13 / ne03 = 1, ne12.
        let fd = |d: u32| init_fastdiv(d);
        assert_eq!(m.div_ne12_ne1, fd(8 * 126));
        assert_eq!(m.div_ne1, fd(126));
        assert_eq!(m.div_r2, fd(1));
        assert_eq!(m.div_r3, fd(1));
        assert_eq!(m.div_ne12, fd(8));
    }

    /// Too little VTCM for all the activation rows: the largest even chunk
    /// that fits, and refusal when not even one row does.
    #[test]
    fn f32_matmul_params_chunk_the_activation_rows_or_refuse() {
        let shape = MulMatF32Shape {
            ne00: 64,
            ne02: 8,
            ne03: 1,
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

    /// A zero extent, or weights whose batch count does not divide the
    /// activations', would put a zero divisor in the kernel's fastdiv slots.
    #[test]
    fn f32_matmul_params_refuse_shapes_with_a_zero_divisor() {
        let ok = MulMatF32Shape {
            ne00: 64,
            ne02: 8,
            ne03: 1,
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
        assert!(params(MulMatF32Shape { ne12: 0, ..ok }).is_none());
        assert!(params(MulMatF32Shape { ne02: 0, ..ok }).is_none());
    }

    #[test]
    fn softmax_params_follow_the_host_precompute() {
        let k = build_softmax_kernel_params([126, 126, 8, 1], Some([126, 8, 1]), 0.125, 8);
        // 126 * 8 rows over 8 threads; every row buffer is 512 B aligned and
        // double-buffered for src0, dst and the mask.
        assert_eq!((k[0], k[1], k[2]), (8, 1008, 126));
        assert_eq!(k[3], (1024 * 3) * 8);
        assert_eq!((k[4], k[5], k[6]), (1024, 1024, 1024));
        assert_eq!((k[7], k[8], k[9]), (512, 512, 512));
        assert_eq!((k[13], k[14], k[15], k[16], k[17]), (8, 8, 1, 0, 1));
        assert_eq!(f32::from_bits(k[18] as u32), 0.125);
        assert_eq!(f32::from_bits(k[19] as u32), 0.0);
        assert_eq!(f32::from_bits(k[20] as u32), 1.0);
        let fd = |d: u32| {
            let f = init_fastdiv(d);
            [f.mp as i32, f.l as i32]
        };
        assert_eq!(k[22..24], fd(126));
        assert_eq!(k[24..26], fd(8));
        assert_eq!(k[26..28], fd(8));
        assert_eq!(k[28..30], fd(1));
        // Without a mask the kernel is NOMASK and has no mask buffers.
        let k = build_softmax_kernel_params([126, 126, 8, 1], None, 1.0, 8);
        assert_eq!((k[15], k[17], k[5], k[8]), (0, 0, 0, 0));
        // Fewer rows than threads: one thread per row.
        let k = build_softmax_kernel_params([16, 3, 1, 1], None, 1.0, 8);
        assert_eq!((k[0], k[2]), (3, 1));
    }

    /// The routed-FFN case: gather 4 unbiased expert weights from a 32-entry table (`src0`
    /// `[1, 32]`, indices `[4]`): the same-type kernel, 4 tasks over 4 threads, no VTCM, no
    /// chunking.
    #[test]
    fn get_rows_kparams_follow_the_host_precompute() {
        let k = build_get_rows_f32_kernel_params(1, 1, 1, 4, 1, 1, 8);
        // n_threads, kernel_type (SAMETYPE), chunks_per_row, chunk_size, total, per_thread, vtcm
        assert_eq!(k[..7], [4, 0, 1, 1, 4, 1, 0]);
        let div = |d: u32| {
            let f = init_fastdiv(d);
            [f.mp as i32, f.l as i32]
        };
        assert_eq!(k[7..9], div(4)); // ne10
        assert_eq!(k[9..11], div(4)); // ne10 * ne11
        assert_eq!(k[11..13], div(1)); // chunks_per_row
        assert_eq!(k[13..15], div(1)); // ne02
        assert_eq!(k[15..17], div(1)); // ne03
        assert!(k[17..].iter().all(|&v| v == 0));
        // A wide row is still one task per gathered row (no DMA split, no chunking any more).
        let k = build_get_rows_f32_kernel_params(2048, 1, 1, 4, 1, 1, 8);
        assert_eq!((k[0], k[1], k[2], k[3], k[4], k[5]), (4, 0, 1, 2048, 4, 1));
        let k = build_get_rows_f32_kernel_params(1500, 1, 1, 2, 1, 1, 8);
        assert_eq!((k[0], k[2], k[3], k[4]), (2, 1, 1500, 2));
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

    fn golden_get_rows(line: &str) -> (GetRowsShape, u32, usize, Vec<i32>) {
        let (inputs, want) = line.split_once("=>").unwrap();
        let (shape, run) = inputs.split_once('|').unwrap();
        let v: Vec<u64> = shape
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let mut run = run.split_whitespace();
        let threads: u32 = run.next().unwrap().parse().unwrap();
        let vtcm: usize = run.next().unwrap().parse().unwrap();
        let words: Vec<i32> = want
            .split_whitespace()
            .map(|w| w.parse::<u32>().unwrap() as i32)
            .collect();
        let shape = GetRowsShape {
            src0_dtype: v[0] as u32,
            dst_dtype: v[1] as u32,
            ne00: v[2] as u32,
            ne02: v[3] as u32,
            ne03: v[4] as u32,
            ne10: v[5] as u32,
            ne11: v[6] as u32,
            ne12: v[7] as u32,
            tiled: v[8] != 0,
        };
        (shape, threads, vtcm, words)
    }

    /// Every case of `testdata/get_rows_golden.txt` (llama.cpp's own
    /// `ggml_hexagon_precompute_get_rows_params`), across the same-type, tiled and flat kernels,
    /// thread counts and VTCM budgets that force the thread count down.
    #[test]
    fn get_rows_kernel_params_match_the_llama_cpp_host_code() {
        let golden = include_str!("testdata/get_rows_golden.txt");
        let (mut cases, mut zero_rows, mut zero_vtcm) = (0usize, 0usize, 0usize);
        for line in golden
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let (shape, threads, vtcm, mut words) = golden_get_rows(line);
            if shape.ne10 == 0 {
                zero_rows += 1;
            }
            if vtcm == 0 {
                zero_vtcm += 1;
            }
            words.resize(32, 0);
            let got = build_get_rows_kernel_params(&shape, threads, vtcm);
            assert_eq!(got.to_vec(), words, "{line}");
            cases += 1;
        }
        assert!(cases > 5000, "golden file looks truncated: {cases} cases");
        assert!(zero_rows > 500, "zero-row cases missing: {zero_rows}");
        assert!(zero_vtcm > 500, "zero-VTCM cases missing: {zero_vtcm}");
    }

    /// The scalar-broadcast params the llama.cpp host precompute produces
    /// (`ggml_hexagon_precompute_binary_params`, contiguous src1 of one
    /// element), worked out by hand from its formulas.
    #[test]
    fn scalar_binary_kparams_follow_the_host_precompute() {
        // 64 elements over 8 threads: chunk floors at 256 elements (1 KiB);
        // VTCM = src0 + dst double buffers per thread + one 128 B scalar slot.
        let k = build_binary_scalar_kernel_params(64, 8 << 20, 8).unwrap();
        assert_eq!(
            k[..11],
            [
                7,
                8,
                1,
                256,
                0,
                256,
                0,
                2 * 8 * 2 * 1024 + 128,
                256,
                1024,
                1
            ]
        );
        assert!(k[11..].iter().all(|&v| v == 0));
        // 2048 elements: the row is 8 KiB, the chunk size unchanged.
        let k = build_binary_scalar_kernel_params(2048, 8 << 20, 8).unwrap();
        assert_eq!((k[3], k[5], k[8], k[9]), (8192, 8192, 256, 1024));
        // A big vector caps the chunk at 32 KiB.
        let k = build_binary_scalar_kernel_params(1_000_000, 8 << 20, 4).unwrap();
        assert_eq!((k[8], k[9], k[7]), (8192, 32768, 2 * 4 * 2 * 32768 + 128));
        // Too little VTCM is a refusal, not a clipped layout.
        assert!(build_binary_scalar_kernel_params(64, 1000, 8).is_none());
    }

    /// The per-row scalar worker's descriptor: kernel type 2, the staged table size and
    /// row sizes aligned to 128, and a refusal when one row per thread cannot fit.
    #[test]
    fn scalar_dma_kernel_params_follow_the_host_precompute() {
        // 128 floats per row (512 B), a 64-entry table (256 B), 4 threads
        let k = build_binary_scalar_dma_kernel_params(128, 64, 4, 8 << 20, 4).unwrap();
        assert_eq!((k[0], k[1], k[2]), (HTP_BINARY_KERNEL_SCALAR_DMA, 4, 16));
        assert_eq!((k[3], k[4], k[5], k[6]), (512, 128, 512, 256));
        assert_eq!(k[7] as usize, 16 * 2 * (512 + 512) * 4 + 256);
        // a wide row shrinks the block to what fits
        let k = build_binary_scalar_dma_kernel_params(65536, 8, 4, 8 << 20, 4).unwrap();
        assert!(k[2] >= 1 && k[2] < 16, "{}", k[2]);
        assert!(k[7] as usize <= 8 << 20);
        // not even one double-buffered row per thread: a refusal, not a clipped layout
        assert!(build_binary_scalar_dma_kernel_params(65536, 8, 4, 1 << 20, 4).is_none());
        // the table alone overflowing is a refusal too
        assert!(build_binary_scalar_dma_kernel_params(128, 1 << 20, 4, 1 << 20, 1).is_none());
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

    /// The legacy knob shape is reachable only through the explicit seam, and
    /// differs from the default blocking (one thread, one row).
    #[test]
    fn unary_legacy_blocking_is_explicit() {
        let legacy = build_unary_kernel_params_with(true, 1024, 32, 1024, 8 << 20, 4, true);
        assert_eq!(legacy[0], 1);
        assert_eq!(legacy[2], 1);
        let new = build_unary_kernel_params_with(false, 1024, 32, 1024, 8 << 20, 4, true);
        assert_eq!(
            new,
            build_unary_kernel_params(1024, 32, 1024, 8 << 20, 4, true)
        );
        assert_ne!(new, legacy);
    }

    #[test]
    fn test_param_builders() {
        let rms = build_rms_norm_params(1e-5);
        assert_eq!(rms[0], (1e-5f32).to_bits() as i32);

        let unary_kp = build_unary_kernel_params(1024, 32, 1024, 8 * 1024 * 1024, 4, true);
        assert_eq!(unary_kp[0], 4); // min(sess_threads, nrows)
        assert_eq!(unary_kp[4], 1);
        assert_eq!(unary_kp[11], 4096);
        // 8MB budget, 4KB rows: (8MB - 4*4KB) / (4 * 2*8KB) = 127 rows/thread.
        assert_eq!(unary_kp[2], 127);
        assert_eq!(unary_kp[3], 127);
        // QK-norm shape: 64-wide rows block 2047/thread.
        let qk = build_unary_kernel_params(64, 8192, 64, 8 * 1024 * 1024, 4, true);
        assert_eq!(qk[0], 4);
        assert_eq!(qk[2], 2047);

        let rope = build_rope_params(128, 0, 4096, 10000.0, 1.0);
        assert_eq!(rope[1], 128);
        assert_eq!(rope[2], 0);
        assert_eq!(rope[4], 4096);
        assert_eq!(rope[5], (10000.0f32).to_bits() as i32);
        assert_eq!(rope[6], (1.0f32).to_bits() as i32);

        // No factors: the slot is empty and sits right after the 4 per-thread slabs.
        let plain = build_rope_kernel_params(128, 16, 16, 1, 4);
        assert_eq!((plain[3], plain[7], plain[8]), (4 * 16896, 4 * 16896, 0));
        // 64 factors (one per pair of a 128-wide head): a 256-byte slot after the
        // slabs, counted in the total VTCM size and nowhere else.
        let ff = build_rope_kernel_params_with_freq_factors(128, 16, 16, 1, 4, 64);
        assert_eq!((ff[3], ff[7], ff[8]), (4 * 16896 + 256, 4 * 16896, 256));
        assert_eq!(ff[4], plain[4]);
        // 65 factors spill into a second aligned block.
        assert_eq!(
            build_rope_kernel_params_with_freq_factors(128, 16, 16, 1, 4, 65)[8],
            512
        );

        let mm = build_mul_mat_kernel_params(HtpDataType::Q8_0, 1024, 1, 1, 4096, 4, 8 << 20);
        let mm = MmKernelParams::from_words(&mm);
        assert_eq!(mm.kernel_type, 6); // single row: QUANT_BLOCK
        assert_eq!(mm.n_threads, 4);
        assert_eq!(mm.tile_size, 1088);
        assert_eq!(mm.aligned_tile_size, 1152);
        assert!(mm.vtcm_size > 0);
        assert_eq!(mm.n_weights, 0);
        assert_eq!(mm.div_ne12_ne1, init_fastdiv(1)); // mp 1, l 0
    }

    #[test]
    fn test_hmx_mm_solver_goldens() {
        // Oracle: htp_mm_hmx_solve_2d_params compiled from llama.cpp's matmul-ops.h
        // (`scripts/hexagon-golden/gen.py hmx_solve`; 8 MB budget, 4 threads unless noted). The
        // chunk sizes divide N evenly where they can (the tail-waste tie-break). The full grid is
        // `hmx_2d_solver_matches_the_llama_cpp_header`.
        let b8 = 8 * 1024 * 1024;
        let q8 = HtpDataType::Q8_0;
        let q4 = HtpDataType::Q4_0;
        // (wtype, k, n, m) -> (m_chunk, n_chunk, act_threads, vtcm).
        type HmxCase = (
            (HtpDataType, usize, usize, usize),
            (usize, usize, usize, usize),
        );
        let cases: [HmxCase; 15] = [
            ((q8, 1024, 4608, 5), (32, 2304, 4, 7587840)),
            ((q8, 1024, 4608, 8), (32, 2304, 4, 7587840)),
            ((q8, 1024, 4608, 13), (32, 2304, 4, 7587840)),
            ((q8, 1024, 4608, 32), (32, 2304, 4, 7587840)),
            ((q8, 4608, 1024, 32), (32, 512, 4, 7702528)),
            ((q8, 4608, 1024, 13), (32, 512, 4, 7702528)),
            ((q8, 1024, 3072, 32), (32, 1536, 4, 5081088)),
            ((q8, 1024, 1024, 32), (32, 1024, 4, 3409920)),
            ((q8, 1024, 256, 32), (32, 256, 4, 903168)),
            ((q8, 1024, 4608, 64), (64, 1152, 4, 7800832)),
            ((q8, 1024, 4608, 512), (512, 768, 4, 7538688)),
            ((q8, 4608, 1024, 512), (256, 192, 4, 8087552)),
            ((q4, 1024, 4608, 32), (32, 2304, 4, 6408192)),
            ((q4, 4608, 1024, 32), (32, 512, 4, 6522880)),
            ((q4, 1024, 1024, 8), (32, 1024, 4, 2885632)),
        ];
        for ((w, k, n, m), want) in cases {
            let got = mm_hmx_solve_2d(w, k, n, m.next_multiple_of(32), m, 4, b8);
            assert_eq!(got, Some(want), "k={k} n={n} m={m}");
        }
        // Small-budget chunking stress.
        assert_eq!(
            mm_hmx_solve_2d(q8, 1024, 4608, 32, 32, 4, 1024 * 1024),
            Some((32, 288, 4, 1007616))
        );
        assert_eq!(
            mm_hmx_solve_2d(q8, 4608, 1024, 32, 32, 4, 2 * 1024 * 1024),
            Some((32, 96, 4, 1685504))
        );
    }

    #[test]
    fn test_hmx_mm_kparams_golden() {
        let kp = build_hmx_mm_kernel_params(HtpDataType::Q8_0, 1024, 4608, 32, 32, 4, 8 << 20)
            .expect("gate m=32 fits");
        let m = MmKernelParams::from_words(&kp);
        assert_eq!(m.kernel_type, 1); // HMX_2D
        assert_eq!(m.pipeline, 0); // pipeline needs M > 32
        assert_eq!(m.m_chunk, 32);
        // 4608 = 2 x 2304: the chunk that divides N evenly wins the tie-break
        assert_eq!(m.n_chunk, 2304);
        assert_eq!(m.n_threads, 4);
        assert_eq!(m.n_act_threads, 4);
        assert_eq!(m.n_hmx, 1);
        assert_eq!(m.n_prefetch, 0); // HVX-only
        assert_eq!((m.collapse, m.n_weights), (0, 0));
        assert_eq!(m.tile_size, 1088);
        assert_eq!(m.aligned_tile_size, 1152);
        assert_eq!(
            m.act_row_size,
            mm_act_tiled_row_size(HtpDataType::Q8_0, 1024) as i32
        );
        assert_eq!(m.vtcm_size, 7587840);
        assert_eq!(
            (
                m.vtcm_src0_size,
                m.vtcm_act_size,
                m.vtcm_bias_size,
                m.vtcm_dst_size
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(m.div_ne12_ne1, init_fastdiv(32));
        assert_eq!(m.div_n_act_threads, init_fastdiv(4));
        assert_eq!(m.div_ne00_padded, init_fastdiv(1024));
        // Pipelined shape.
        let kp64 = build_hmx_mm_kernel_params(HtpDataType::Q8_0, 1024, 4608, 64, 64, 4, 8 << 20)
            .expect("gate m=64 fits");
        let m64 = MmKernelParams::from_words(&kp64);
        assert_eq!(m64.pipeline, 1);
        assert_eq!((m64.m_chunk, m64.n_chunk), (64, 1152));
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
            let got = fa_hmx_find_chunk_size(g, dk, dv, qo, kv, b8, 4, true, false, 16);
            assert_eq!(got, Some((br, bc)), "g={g} qo={qo} kv={kv}");
            let pipelined = kv >= 3 * 64;
            let total = fa_hmx_layout_total(g, dk, dv, br, bc, 4, pipelined, true, false, 16);
            assert_eq!(total, vs, "g={g} qo={qo} kv={kv}");
        }
        // Small budget + single thread.
        assert_eq!(
            fa_hmx_find_chunk_size(2, 64, 64, 32, 128, 1024 * 1024, 4, true, false, 16),
            Some((32, 128))
        );
        assert_eq!(
            fa_hmx_layout_total(2, 64, 64, 32, 128, 4, false, true, false, 16),
            200320
        );
        assert_eq!(
            fa_hmx_find_chunk_size(2, 64, 64, 32, 256, b8, 1, true, false, 16),
            Some((32, 256))
        );
        assert_eq!(
            fa_hmx_layout_total(2, 64, 64, 32, 256, 1, false, true, false, 16),
            363136
        );
    }

    #[test]
    fn test_hmx_fa_kparams_golden() {
        let scale = 1.0f32 / 64.0f32.sqrt();
        let kp = build_hmx_fa_kernel_params(64, 16, 8, 32, 602, scale, 4, 8 << 20)
            .expect("m=32 kv=602 fits");
        assert_eq!(kp[0] & 0xff, 2); // HMX
        // bytes 1 and 2 are head_split (on, as in llama.cpp) and reserved flags; they used to be
        // is_q_fp32 / is_dst_fp32
        assert_eq!((kp[0] >> 8) & 0xff, 1);
        assert_eq!((kp[0] >> 16) & 0xff, 0);
        assert_eq!(kp[1], 32 | (192 << 16)); // Br | Bc << 16
        assert_eq!(kp[2] & 0xffff, 4); // n_kv_blocks = ceil(602/192)
        assert_eq!((kp[2] >> 16) & 0xffff, 2); // G
        // the HMX kernel works in the log2 domain: the host folds log2(e) into the scale
        assert_eq!(kp[3], (scale * std::f32::consts::LOG2_E).to_bits() as i32);
        assert_eq!(kp[5], 0);
        assert_eq!(kp[6], 386688); // vtcm_size
        assert_eq!(kp[7], 0);
        assert_eq!(kp[8], 0);
        assert_eq!(kp[12], 16); // n_head_log2
        assert_eq!(kp[17], 0); // broadcast divs unset on HMX
        assert_eq!(kp[25], 64); // g_br = align_up(2*32, 32)
        assert_eq!(kp[26], 4); // row_buf_stride = align_up(384, 256)/128
        assert_eq!(kp[27], 192); // mask_buf_row_stride = align_up(384, 128)/2
        assert_eq!(kp[28], 1); // mask_broadcast
        assert_eq!(kp[29], 1); // pipelined
        let div_g = init_fastdiv(2);
        assert_eq!((kp[30], kp[31]), (div_g.mp as i32, div_g.l as i32));
    }

    #[test]
    fn test_hmx_fa_eligibility() {
        assert!(fa_is_hmx_eligible(64, 5));
        assert!(fa_is_hmx_eligible(64, 32));
        assert!(fa_is_hmx_eligible(128, 32));
        assert!(!fa_is_hmx_eligible(64, 4)); // small-DK short chunk: HVX
        assert!(!fa_is_hmx_eligible(70, 32)); // DK % 8 != 0
    }

    #[test]
    fn test_hmx_mm_eligibility() {
        assert!(mm_is_hmx_eligible(HtpDataType::Q8_0, 1024, 4608, 8));
        assert!(mm_is_hmx_eligible(HtpDataType::Q4_0, 4608, 1024, 32));
        assert!(mm_is_hmx_eligible(HtpDataType::Q4K, 1024, 4608, 8));
        assert!(mm_is_hmx_eligible(HtpDataType::Q6K, 1024, 1024, 32));
        assert!(mm_is_hmx_eligible(HtpDataType::F16, 4320, 1152, 32));
        assert_eq!(mm_hmx_nb1(HtpDataType::F16, 4320), 4320 * 2);
        assert!(
            build_hmx_mm_kernel_params(HtpDataType::F16, 4320, 1152, 768, 768, 6, 8 << 20)
                .is_some()
        );
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1024, 4608, 7)); // M < 8: HVX
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1024, 4608, 4)); // M < 8: HVX
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1000, 4608, 32)); // K % 32
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1024, 4600, 32)); // N % 32
        assert!(!mm_is_hmx_eligible(HtpDataType::F32, 1024, 4608, 32)); // dense unhandled
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

        // Flash attention: decode (M=1) and prefill chunk (M=32).
        let fa1 = build_flash_attn_kernel_params(64, 16, 8, 1, 64, 0.125, 1, true);
        assert_eq!(fa1[7], 16); // qrows
        assert_eq!(fa1[8], 16); // qrows_per_thread (1 thread)
        let fa32 = build_flash_attn_kernel_params(64, 16, 8, 32, 544, 0.125, 4, true);
        assert_eq!(fa32[7], 512);
        assert_eq!(fa32[8], 128);
        assert_eq!(fa32[28], init_fastdiv(512).mp as i32); // src0_div21
        assert_eq!(fa32[30], init_fastdiv(32).mp as i32); // src0_div1
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

    #[test]
    fn test_build_flash_attn_kernel_params_zero_heads() {
        let p = build_flash_attn_kernel_params(64, 0, 1, 1, 1, 0.1, 1, false);
        // Offset 12 corresponds to n_head_log2; must be 0 when n_heads == 0 without underflow
        assert_eq!(p[12], 0);
    }

    #[test]
    fn test_flash_attn_kernel_params_with_softcap() {
        let p = build_flash_attn_kernel_params_with_softcap(64, 16, 8, 1, 64, 0.125, 1, true, 50.0);
        assert_eq!(p[5], 50.0f32.to_bits() as i32);
        // HVX: head_split on, no type flags
        assert_eq!((p[0] >> 8) & 0xff, 1);
        assert_eq!((p[0] >> 16) & 0xff, 0);

        // The scale is divided by the softcap, as the C++ host does
        assert_eq!(p[3], (0.125f32 / 50.0).to_bits() as i32);
        // HMX with a softcap folds log2(e) into the softcap and leaves the (divided) scale alone;
        // without one it folds it into the scale
        let hmx =
            build_hmx_fa_kernel_params_with_softcap(64, 16, 8, 32, 602, 0.125, 4, 8 << 20, 50.0)
                .expect("should fit");
        assert_eq!(hmx[3], (0.125f32 / 50.0).to_bits() as i32);
        assert_eq!(
            hmx[5],
            (50.0f32 * std::f32::consts::LOG2_E).to_bits() as i32
        );
        let plain = build_hmx_fa_kernel_params(64, 16, 8, 32, 602, 0.125, 4, 8 << 20).unwrap();
        assert_eq!(
            plain[3],
            (0.125f32 * std::f32::consts::LOG2_E).to_bits() as i32
        );
        assert_eq!(plain[5], 0);
    }

    const VTCM: usize = 8 * 1024 * 1024;

    /// The ViT's 3072-wide FFN down-projection over 160 tokens (the shape that
    /// the DSP rejected with `VtcmTooSmall`): its quantized activations alone
    /// are 17 MB, so the rows must go in chunks that fit.
    #[test]
    fn rows_the_vtcm_cannot_hold_are_chunked_to_fit() {
        let k = build_mul_mat_kernel_params(HtpDataType::Q8_0, 3072, 160, 1, 768 * 4, 6, VTCM);
        let mm = MmKernelParams::from_words(&k);
        let chunk = mm.m_chunk as usize;
        assert!(
            (2..160).contains(&chunk) && chunk.is_multiple_of(2),
            "m_chunk {chunk}"
        );
        assert!(mm_hvx_dsp_vtcm(HtpDataType::Q8_0, 3072, chunk, 6, 768 * 4, 2) <= VTCM);
        // The DSP's solver: (VTCM - weight prefetch) / (quantized + raw row),
        // rounded down to an even count: (8 MiB - 1327104) / (110592 + 12288).
        assert_eq!(chunk, 56);
        let reported = mm.vtcm_size as usize;
        assert!(reported <= VTCM, "the reported layout is the chunk's");
        assert!(reported > 0, "the DSP is told how much VTCM the chunk uses");
    }

    /// Over every shape the ViT, Whisper and the LFM2 prefill reach: a chunk
    /// exists exactly when the whole does not fit, and is even, at least two
    /// and itself fits (a plan that overran would be rejected by the DSP).
    #[test]
    fn the_chunk_exists_exactly_when_the_rows_do_not_fit() {
        for wtype in [HtpDataType::Q8_0, HtpDataType::Q4_0] {
            for n_threads in [1, 4, 6] {
                for k_dim in [768, 1024, 2048, 3072] {
                    for rows in 2..=512usize {
                        let k = build_mul_mat_kernel_params(
                            wtype,
                            k_dim,
                            rows as u32,
                            1,
                            4096,
                            n_threads,
                            VTCM,
                        );
                        let mm = MmKernelParams::from_words(&k);
                        let (chunk, prefetch) = (mm.m_chunk as usize, mm.n_prefetch as usize);
                        let total = |m: usize| {
                            mm_hvx_dsp_vtcm(wtype, k_dim, m, n_threads as usize, 4096, prefetch)
                        };
                        let at = format!("{wtype:?} K {k_dim} rows {rows} threads {n_threads}");
                        if total(rows) <= VTCM {
                            assert_eq!(chunk, 0, "{at}: fits whole");
                        } else {
                            assert!(
                                chunk >= 2 && chunk.is_multiple_of(2) && chunk < rows,
                                "{at}: chunk {chunk}"
                            );
                            assert!(total(chunk) <= VTCM, "{at}: chunk {chunk} overruns");
                        }
                    }
                }
            }
        }
    }

    /// Shapes that fit, and decode, are left alone: no chunk.
    #[test]
    fn rows_that_fit_and_single_rows_take_no_chunk() {
        for (k_dim, rows) in [(768, 160), (3072, 40), (2048, 1), (2048, 4), (1024, 64)] {
            let k = build_mul_mat_kernel_params(HtpDataType::Q8_0, k_dim, rows, 1, 4096, 6, VTCM);
            assert_eq!(
                MmKernelParams::from_words(&k).m_chunk,
                0,
                "K {k_dim} rows {rows}"
            );
        }
    }

    /// The fused Q/K/V kernel cannot chunk, so its row limit is the largest
    /// count its layout fits.
    #[test]
    fn the_fused_nx_row_limit_is_the_largest_that_fits() {
        let max = mm_hvx_fused_nx_max_rows(HtpDataType::Q8_0, 768, 6, VTCM);
        // 768 wide: 27648 quantized bytes per row, so a couple of hundred rows.
        assert!((200..300).contains(&max), "{max}");
        let tile_row: usize = 24 * 1152;
        let src0 = (2 * tile_row).next_multiple_of(128) * 6;
        let total = |m: usize| {
            (27648 * m).next_multiple_of(128) + src0.max((3072 * m).next_multiple_of(128))
        };
        assert!(total(max) <= VTCM && total(max + 1) > VTCM);
        // A wider activation takes fewer rows.
        assert!(mm_hvx_fused_nx_max_rows(HtpDataType::Q8_0, 3072, 6, VTCM) < max);
    }
}

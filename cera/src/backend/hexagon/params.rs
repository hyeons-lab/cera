//! Host-side parameter precomputations for Hexagon HTP operations.
//!
//! Hexagon DSP kernels avoid runtime hardware division and transcendental calls
//! by consuming precomputed integer division constants (Granlund and Montgomery FastDiv)
//! and layout descriptors generated on the host.

/// Precomputed integer division constants using Granlund and Montgomery's algorithm.
///
/// Permits the DSP to calculate `n / d` without hardware division via:
/// `((mulhi(n, mp) + n) >> l)`.
use super::types::HtpDataType;

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
    let mut kparams = [0i32; 32];
    // Debug: single-threaded single-row blocking (pre-port behavior).
    let legacy = std::env::var("CERA_HEXAGON_UNARY_T1")
        .map(|v| v == "1")
        .unwrap_or(false);
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
        avail / (n_threads.max(1) * per_row.max(1))
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

    let src0_row_size_aligned = (src0_row_size + 127) & !127;
    let src1_row_size_aligned = (src1_row_size + 127) & !127;
    let dst_row_size_aligned = (dst_row_size + 127) & !127;

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
    let total_vtcm =
        rows_per_buffer * 2 * (src0_row_size_aligned + dst_row_size_aligned) * n_threads as usize
            + src1_size;

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
/// Mirrors `htp_rope_kernel_params` plus `htp_rope_vtcm_layout_build` with no
/// frequency factors (plain RoPE, no YaRN scaling).
pub fn build_rope_kernel_params(
    n_dims: usize,
    nrows: usize,
    ne1: usize,
    ne2: usize,
    n_threads: u32,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = n_threads.max(1) as usize;
    let row_size = n_dims * 4;
    let row_aligned = (row_size + 127) & !127;
    let theta_aligned = (row_size + 255) & !255;
    // HTP_ROPE_SPAD_NROWS = HTP_ROPE_SPAD_BLOCK(8) * HTP_ROPE_SPAD_NSLOTS(4)
    let bytes_per_thread = theta_aligned + 32 * row_aligned;
    let total = bytes_per_thread * n_threads;

    kparams[0] = n_threads as i32;
    kparams[1] = nrows as i32;
    kparams[2] = nrows.div_ceil(n_threads) as i32;
    kparams[3] = total as i32;
    kparams[4] = bytes_per_thread as i32;
    kparams[5] = theta_aligned as i32;
    kparams[6] = row_aligned as i32;
    kparams[7] = (bytes_per_thread * n_threads) as i32;
    kparams[8] = 0;
    let div_ne2_ne1 = init_fastdiv((ne2 * ne1) as u32);
    kparams[9] = div_ne2_ne1.mp as i32;
    kparams[10] = div_ne2_ne1.l as i32;
    let div_ne1 = init_fastdiv(ne1 as u32);
    kparams[11] = div_ne1.mp as i32;
    kparams[12] = div_ne1.l as i32;
    kparams
}

/// Build kernel parameters for SetRows dispatch.
///
/// Mirrors `ggml_hexagon_precompute_set_rows_params` +
/// `htp_set_rows_vtcm_layout_build`: `n_rows` value rows (src0->ne[1]),
/// `idx_ne1`/`idx_ne2` index-vector dims (1 for a flat positions vector),
/// `src0_ne2` value dim-2 (head count for KV), `ne00` row width,
/// `dst_f16` cache dtype.
pub fn build_set_rows_kernel_params(
    n_rows: usize,
    idx_ne1: usize,
    idx_ne2: usize,
    src0_ne2: usize,
    ne00: usize,
    dst_f16: bool,
    sess_threads: u32,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = (sess_threads as usize).min(n_rows.max(1)).max(1);
    let tasks_per_thread = n_rows.div_ceil(n_threads);
    kparams[0] = n_threads as i32;
    kparams[1] = n_rows as i32;
    kparams[2] = tasks_per_thread as i32;
    // VTCM layout: value/cache rows 256-aligned, double-buffered, per thread.
    let src0_aligned = (ne00 * 4).next_multiple_of(256);
    let dst_row = if dst_f16 { ne00 * 2 } else { ne00 * 4 };
    let dst_aligned = dst_row.next_multiple_of(256);
    kparams[3] = ((src0_aligned * 2 + dst_aligned * 2) * n_threads) as i32;
    let div_ne11 = init_fastdiv(idx_ne1.max(1) as u32);
    kparams[4] = div_ne11.mp as i32;
    kparams[5] = div_ne11.l as i32;
    let div_ne12 = init_fastdiv(idx_ne2.max(1) as u32);
    kparams[6] = div_ne12.mp as i32;
    kparams[7] = div_ne12.l as i32;
    let div_tpt = init_fastdiv(tasks_per_thread as u32);
    kparams[8] = div_tpt.mp as i32;
    kparams[9] = div_tpt.l as i32;
    let div_ne02 = init_fastdiv(src0_ne2.max(1) as u32);
    kparams[10] = div_ne02.mp as i32;
    kparams[11] = div_ne02.l as i32;
    kparams
}

/// Build kernel parameters for SsmConv dispatch.
///
/// Mirrors `ggml_hexagon_precompute_ssm_conv_params`: `d_conv` taps
/// (src1->ne[0], 3 for LFM2 short conv), `d_inner` channels (src0->ne[1]),
/// `n_t` new positions (dst->ne[1]), `n_s` sequences (dst->ne[2], 1),
/// `ncs` src0 dim-0 (`d_conv - 1 + n_t`).
pub fn build_ssm_conv_kernel_params(
    d_conv: usize,
    d_inner: usize,
    n_t: usize,
    n_s: usize,
    ncs: usize,
    sess_threads: u32,
    vtcm_budget: usize,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = (sess_threads as usize).min(d_inner.div_ceil(32)).max(1);
    kparams[0] = n_threads as i32;
    kparams[1] = d_conv as i32;
    kparams[2] = d_inner as i32;
    kparams[3] = n_t as i32;
    kparams[4] = n_s as i32;
    let d_inner_per_thread = d_inner.div_ceil(n_threads).next_multiple_of(32);
    kparams[5] = d_inner_per_thread as i32;
    let round128 = |n: usize| n.next_multiple_of(128);
    kparams[7] = round128(ncs * 4) as i32;
    kparams[8] = round128(d_conv * 4) as i32;
    kparams[9] = round128(d_inner * 4) as i32;

    // Weight-side VTCM is identical in both branches: raw rows plus the
    // transposed tile the HVX kernel multiplies from.
    let src1_raw = round128(d_inner_per_thread * d_conv * 4) + 128;
    let src1_t = round128(d_conv * d_inner_per_thread * 4);
    let vtcm_src1_per_thread = src1_raw + src1_t;
    kparams[11] = vtcm_src1_per_thread as i32;

    let (vtcm_src0_per_thread, vtcm_dst_per_thread) = if n_t == 1 {
        // Scalar path: one position, full per-thread channel range.
        kparams[6] = d_inner_per_thread as i32;
        let src0_raw = round128(d_inner_per_thread * d_conv * 4) + 128;
        let src0_t = round128(d_conv * d_inner_per_thread * 4);
        (src0_raw + src0_t, round128(d_inner_per_thread * 4))
    } else {
        // Chunk path: tile channels to fit per-thread VTCM budget.
        let budget_per_thread = if vtcm_budget > 0 {
            vtcm_budget / n_threads
        } else {
            1024 * 1024
        };
        let avail = budget_per_thread
            .saturating_sub(vtcm_src1_per_thread)
            .max(128 * 1024);
        let mut tile = (avail / 2) / (ncs * 4 + n_t * 4 + 1);
        tile = (tile / 32) * 32;
        if tile == 0 {
            tile = 32;
        }
        let tile = tile.min(d_inner_per_thread);
        kparams[6] = tile as i32;
        let src0_raw = round128(tile * ncs * 4) + 128;
        let src0_t = round128(ncs * tile * 4);
        (src0_raw + src0_t, round128(tile * n_t * 4))
    };
    kparams[10] = vtcm_src0_per_thread as i32;
    kparams[12] = vtcm_dst_per_thread as i32;
    kparams[13] = (vtcm_src0_per_thread * n_threads) as i32;
    kparams[14] = (vtcm_src1_per_thread * n_threads) as i32;
    kparams[15] = (vtcm_dst_per_thread * n_threads) as i32;
    kparams[16] = (vtcm_src0_per_thread + vtcm_src1_per_thread + vtcm_dst_per_thread) as i32
        * n_threads as i32;
    let div_nt = init_fastdiv(n_threads as u32);
    kparams[17] = div_nt.mp as i32;
    kparams[18] = div_nt.l as i32;
    kparams
}

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
    let n_threads = n_threads.max(1) as u8;
    let g = (n_heads / n_kv_heads).max(1);
    let n_kv_blocks = seq_len.div_ceil(64).max(1);

    // byte 0: kernel_type = 1 (HTP_FA_KERNEL_HVX)
    // byte 1: is_q_fp32 = 1
    // byte 2: is_dst_fp32 = 1
    // byte 3: n_threads
    let b0 = 1u32 | (1u32 << 8) | (1u32 << 16) | ((n_threads as u32) << 24);
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
    let size_q_row_padded = (head_dim * 4 + 127) & !127;
    let size_k_row_padded = (head_dim * 2 + 127) & !127;
    let size_v_row_padded = (head_dim * 2 + 127) & !127;
    let size_q_block = size_q_row_padded;
    let size_k_block = size_k_row_padded * 64;
    let size_v_block = size_v_row_padded * 64;
    let size_m_block = (64 * 2 + 127) & !127;
    let size_vkq_acc = (head_dim * 4 + 127) & !127;
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
    let n_head_log2 = 1u32 << (31 - (n_heads as u32).leading_zeros());
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

/// HVX weight tile sizes (see `HTP_MM_WEIGHT_*_TILE_SIZE_*`).
fn mm_tile_sizes(wtype: HtpDataType) -> (u32, u32) {
    match wtype {
        HtpDataType::Q4_0 => (576, 640),
        // Q4_K repacks into the Q4_1 wire layout (640/640).
        HtpDataType::Q4K => (640, 640),
        HtpDataType::Q6K => (896, 896),
        HtpDataType::Q8_0 => (1088, 1152),
        other => {
            // New weight type without a tile entry: fail loudly in debug
            // rather than silently emitting Q8_0-sized tiles.
            debug_assert!(false, "mm_tile_sizes: no tile entry for {other:?}");
            (1088, 1152)
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
    let tile_row = (ne10 / 32) * aligned_tile as usize;
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

/// Build kernel parameters for matrix multiplication dispatch.
///
/// Mirrors `ggml_hexagon_precompute_hvx_mm_params` for the quantized HVX
/// decode path (repacked Q4_0/Q8_0 weights, non-batched). `ne10` is the
/// activation width (K), `ne11`/`ne12` the activation rows/batches (1 for
/// single-token decode), `dst_row_size` the output row stride in bytes.
pub fn build_mul_mat_kernel_params(
    wtype: HtpDataType,
    ne10: usize,
    ne11: u32,
    ne12: u32,
    dst_row_size: usize,
    n_threads: u32,
    vtcm_budget: usize,
) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let n_threads = n_threads.max(1) as usize;
    let src1_nrows = (ne11 as usize) * (ne12 as usize);

    // Fewer rows than threads takes the block-partitioned quant path.
    kparams[0] = if src1_nrows < n_threads { 6 } else { 5 };
    kparams[1] = 0; // pipeline
    kparams[2] = 0; // m_chunk (decode rows always fit; no chunking)
    kparams[3] = 0; // n_chunk (HMX only)
    kparams[4] = n_threads as i32;
    kparams[5] = 0; // n_act_threads (HMX only)
    kparams[6] = 0; // n_hmx
    let (tile_size, aligned_tile_size) = mm_tile_sizes(wtype);
    kparams[8] = tile_size as i32;
    kparams[9] = aligned_tile_size as i32;
    kparams[10] = mm_act_tiled_row_size(wtype, ne10) as i32; // src1_row_size

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
    // The device rebuilds this layout and rejects loudly (VTCM_TOO_SMALL)
    // when it overruns real VTCM, so falling back to depth 2 here is safe.
    let layout = best_layout
        .unwrap_or_else(|| mm_hvx_vtcm_layout(wtype, ne10, src1_nrows, n_threads, dst_row_size, 2));
    kparams[7] = best as i32; // n_prefetch
    kparams[11] = layout.total_bytes as i32; // vtcm_size
    kparams[12] = layout.src0_bytes as i32; // vtcm_src0_size
    kparams[13] = layout.src1_bytes as i32; // vtcm_src1_size
    kparams[14] = 0; // vtcm_src2_size (fused only)
    kparams[15] = 0; // vtcm_src3_size (fused only)
    kparams[16] = layout.dst_bytes as i32; // vtcm_dst_size
    kparams[17] = 0; // n_weights (fused NX only)

    // Non-batched decode: ne02 = ne03 = ne13 = 1.
    let div_ne12_ne11 = init_fastdiv(ne11 * ne12);
    let div_ne11 = init_fastdiv(ne11.max(1));
    let div_r2 = init_fastdiv(ne12.max(1));
    let div_1 = init_fastdiv(1);
    kparams[18] = div_ne12_ne11.mp as i32;
    kparams[19] = div_ne12_ne11.l as i32;
    kparams[20] = div_ne11.mp as i32;
    kparams[21] = div_ne11.l as i32;
    kparams[22] = div_r2.mp as i32;
    kparams[23] = div_r2.l as i32;
    kparams[24] = div_1.mp as i32;
    kparams[25] = div_1.l as i32;
    kparams[26] = div_r2.mp as i32;
    kparams[27] = div_r2.l as i32;
    // div_n_act_threads, div_ne00_padded: HMX only, stay zero.

    kparams
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

/// HMX weight types we emit (`ggml_hexagon_is_hmx_weight_type`, repack subset).
pub fn mm_is_hmx_eligible(wtype: HtpDataType, k: usize, n: usize, m: usize) -> bool {
    matches!(
        wtype,
        HtpDataType::Q4_0 | HtpDataType::Q8_0 | HtpDataType::Q4K | HtpDataType::Q6K
    ) && k.is_multiple_of(HMX_TILE)
        && n.next_multiple_of(HMX_TILE).is_multiple_of(HMX_TILE)
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
        if let Some((mc, nc, cost, mn)) = accept()
            && (cost < best_cost || (cost == best_cost && mn > best_mn))
        {
            best_cost = cost;
            best_mn = mn;
            best = (mc, nc);
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
    let mut kparams = [0i32; 32];
    let (tile_size, aligned_tile_size) = mm_tile_sizes(wtype);
    kparams[0] = 1; // HTP_MM_KERNEL_HMX_2D
    kparams[1] = mm_hmx_pipeline(m) as i32;
    kparams[2] = mc as i32;
    kparams[3] = nc as i32;
    kparams[4] = n_threads.max(1) as i32;
    kparams[5] = act_threads as i32;
    kparams[6] = 1; // n_hmx
    // [7] n_prefetch: HVX-only, stays zero.
    kparams[8] = tile_size as i32;
    kparams[9] = aligned_tile_size as i32;
    kparams[10] = mm_act_tiled_row_size(wtype, k) as i32;
    kparams[11] = vtcm as i32;
    // [12..=17] src0/1/2/3/dst sizes + n_weights: zero outside fused NX.
    // Shared finalize divs for flat [K, M] activations (ne12 = ne02 = 1).
    let div_m = init_fastdiv(m as u32);
    let div_1 = init_fastdiv(1);
    kparams[18] = div_m.mp as i32;
    kparams[19] = div_m.l as i32;
    kparams[20] = div_m.mp as i32;
    kparams[21] = div_m.l as i32;
    kparams[22] = div_1.mp as i32;
    kparams[23] = div_1.l as i32;
    kparams[24] = div_1.mp as i32;
    kparams[25] = div_1.l as i32;
    kparams[26] = div_1.mp as i32;
    kparams[27] = div_1.l as i32;
    let div_at = init_fastdiv(act_threads as u32);
    kparams[28] = div_at.mp as i32;
    kparams[29] = div_at.l as i32;
    let div_k = init_fastdiv(k as u32);
    kparams[30] = div_k.mp as i32;
    kparams[31] = div_k.l as i32;
    Some(kparams)
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
    kparams[0] = (2u32 | (1u32 << 8) | (1u32 << 16) | ((kt as u32) << 24)) as i32;
    kparams[1] = (br as u32 | ((bc as u32) << 16)) as i32;
    kparams[2] = ((n_kv_blocks as u32 & 0xffff) | ((g as u32 & 0xffff) << 16)) as i32;
    kparams[3] = scale.to_bits() as i32;
    // [4] max_bias, [5] logit_softcap: zero.
    kparams[6] = fa_hmx_layout_total(g, dk, dv, br, bc, kt, pipelined, true, false, n_heads) as i32;
    // [7] qrows, [8] qrows_per_thread, [9] qrow_start: zero for HMX.
    kparams[10] = 1.0f32.to_bits() as i32;
    kparams[11] = 1.0f32.to_bits() as i32;
    kparams[12] = (1u32 << (31 - (n_heads as u32).leading_zeros())) as i32;
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

#[cfg(test)]
mod tests {
    use super::*;

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

        let mm = build_mul_mat_kernel_params(HtpDataType::Q8_0, 1024, 1, 1, 4096, 4, 8 << 20);
        assert_eq!(mm[0], 6); // single row: QUANT_BLOCK
        assert_eq!(mm[4], 4);
        assert_eq!(mm[8], 1088);
        assert_eq!(mm[9], 1152);
        assert!(mm[11] > 0); // vtcm_size
        assert_eq!(mm[17], 0); // n_weights
        assert_eq!(mm[18], 1); // div(1).mp
        assert_eq!(mm[19], 0); // div(1).l
    }

    #[test]
    fn test_hmx_mm_solver_goldens() {
        // Oracle: htp_mm_hmx_solve_2d_params compiled from upstream
        // matmul-ops.h (8MB budget, 4 threads unless noted).
        let b8 = 8 * 1024 * 1024;
        let q8 = HtpDataType::Q8_0;
        let q4 = HtpDataType::Q4_0;
        // (wtype, k, n, m) -> (m_chunk, n_chunk, act_threads, vtcm).
        type HmxCase = (
            (HtpDataType, usize, usize, usize),
            (usize, usize, usize, usize),
        );
        let cases: [HmxCase; 15] = [
            ((q8, 1024, 4608, 5), (32, 2528, 4, 8318976)),
            ((q8, 1024, 4608, 8), (32, 2528, 4, 8318976)),
            ((q8, 1024, 4608, 13), (32, 2528, 4, 8318976)),
            ((q8, 1024, 4608, 32), (32, 2528, 4, 8318976)),
            ((q8, 4608, 1024, 32), (32, 544, 4, 8165376)),
            ((q8, 4608, 1024, 13), (32, 544, 4, 8165376)),
            ((q8, 1024, 3072, 32), (32, 2528, 4, 8318976)),
            ((q8, 1024, 1024, 32), (32, 1024, 4, 3409920)),
            ((q8, 1024, 256, 32), (32, 256, 4, 903168)),
            ((q8, 1024, 4608, 64), (64, 1216, 4, 8226816)),
            ((q8, 1024, 4608, 512), (512, 864, 4, 8349696)),
            ((q8, 4608, 1024, 512), (256, 192, 4, 8087552)),
            ((q4, 1024, 4608, 32), (32, 3008, 4, 8345600)),
            ((q4, 4608, 1024, 32), (32, 640, 4, 8079360)),
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
        assert_eq!(kp[0], 1); // HMX_2D
        assert_eq!(kp[1], 0); // pipeline needs M > 32
        assert_eq!(kp[2], 32); // m_chunk
        assert_eq!(kp[3], 2528); // n_chunk
        assert_eq!(kp[4], 4); // n_threads
        assert_eq!(kp[5], 4); // act_threads
        assert_eq!(kp[6], 1); // n_hmx
        assert_eq!(kp[7], 0); // n_prefetch: HVX-only
        assert_eq!(kp[8], 1088);
        assert_eq!(kp[9], 1152);
        assert_eq!(kp[11], 8318976); // vtcm_size
        assert_eq!(kp[12], 0);
        assert_eq!(kp[17], 0);
        let div32 = init_fastdiv(32);
        assert_eq!((kp[18], kp[19]), (div32.mp as i32, div32.l as i32));
        assert_eq!((kp[28], kp[29]), (init_fastdiv(4).mp as i32, 2));
        let div1k = init_fastdiv(1024);
        assert_eq!((kp[30], kp[31]), (div1k.mp as i32, div1k.l as i32));
        // Pipelined shape.
        let kp64 = build_hmx_mm_kernel_params(HtpDataType::Q8_0, 1024, 4608, 64, 64, 4, 8 << 20)
            .expect("gate m=64 fits");
        assert_eq!(kp64[1], 1);
        assert_eq!((kp64[2], kp64[3]), (64, 1216));
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
        assert_eq!(kp[1], 32 | (192 << 16)); // Br | Bc << 16
        assert_eq!(kp[2] & 0xffff, 4); // n_kv_blocks = ceil(602/192)
        assert_eq!((kp[2] >> 16) & 0xffff, 2); // G
        assert_eq!(kp[3], scale.to_bits() as i32);
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
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1024, 4608, 7)); // M < 8: HVX
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1024, 4608, 4)); // M < 8: HVX
        assert!(!mm_is_hmx_eligible(HtpDataType::Q8_0, 1000, 4608, 32)); // K % 32
        assert!(!mm_is_hmx_eligible(HtpDataType::F32, 1024, 4608, 32)); // dense unhandled
    }

    #[test]
    fn test_prefill_param_builders() {
        // SetRows: single row (decode) and 32-row chunk (prefill).
        let sr1 = build_set_rows_kernel_params(1, 1, 1, 8, 64, true, 4);
        assert_eq!(sr1[0], 1); // n_threads
        assert_eq!(sr1[1], 1); // total_tasks
        assert_eq!(sr1[2], 1); // tasks_per_thread
        assert_eq!(sr1[3], 1024); // vtcm: (512 + 512) * 1
        let sr32 = build_set_rows_kernel_params(32, 1, 1, 8, 64, true, 4);
        assert_eq!(sr32[0], 4);
        assert_eq!(sr32[1], 32);
        assert_eq!(sr32[2], 8);
        assert_eq!(sr32[3], 4096);
        assert_eq!(sr32[8], init_fastdiv(8).mp as i32); // div_tasks_per_thread

        // SsmConv: scalar (n_t=1) and chunk (n_t=32) branches, LFM2 K=3/C=1024.
        let ssm1 = build_ssm_conv_kernel_params(3, 1024, 1, 1, 3, 4, 8 << 20);
        assert_eq!(ssm1[0], 4); // n_threads
        assert_eq!(ssm1[5], 256); // d_inner_per_thread
        assert_eq!(ssm1[6], 256); // d_inner_tile (scalar: full range)
        assert_eq!(ssm1[16], 54272); // vtcm total
        let ssm32 = build_ssm_conv_kernel_params(3, 1024, 32, 1, 34, 4, 8 << 20);
        assert_eq!(ssm32[6], 256); // tile covers the per-thread range
        assert_eq!(ssm32[16], 435200);

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
}

//! Host repack of GGUF weights into the XDNA kernels' stream layout.
//!
//! The layout is the one the spike validated (`tools/xdna-spike/spike.cpp`
//! `q4_pack_weights`, consumed by `kernels/iron/q4gemv` and `q4layer`):
//! tiles in row-major order (worker `w` owns rows `[w*rows/16/16*16, ...)`
//! contiguously), each 16-row tile sent as 2048-column chunks (split-K),
//! each chunk holding per 32-element block two 128-byte int4 B matrices
//! (element `e = k*16 + n`, low nibble first) followed by the 16 bf16 row
//! scales of every block: 288 bytes per block, 4.5 bits per weight.
//!
//! Weights arrive as GGUF Q4_0 (`pack_q4_0_stage`) or as dequantized f32 rows
//! (`pack_f32_stage`, used for Q6_K or other non-Q4_0 tensors quantized with
//! the spike's Q4_0 rule).

use crate::quant::f16_to_f32;

/// Weights per quantization block.
const QBLK: usize = 32;
/// Rows per kernel tile.
const ROWS: usize = 16;
/// Columns per streamed chunk (split-K unit).
const CHUNK_K: usize = 2048;
/// Bytes per block in the packed stream: 256 of int4 data + 32 of scales.
const BLOCK_BYTES: usize = 288;

/// Convert f32 to bf16 with round-to-nearest-even (the spike's rule),
/// preserving quiet NaNs instead of rounding mantissas into infinity.
pub fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((u >> 16) as u16) | 0x0040;
    }
    let rounded = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (rounded >> 16) as u16
}

/// Packed byte count for one stage of `rows` x `cols` weights.
pub fn stage_bytes(rows: usize, cols: usize) -> Result<usize, String> {
    check_shape(rows, cols)?;
    let chunks = cols.div_ceil(CHUNK_K);
    let blocks_per_chunk = CHUNK_K.min(cols) / QBLK;
    (rows / ROWS)
        .checked_mul(chunks)
        .and_then(|v| v.checked_mul(blocks_per_chunk))
        .and_then(|v| v.checked_mul(BLOCK_BYTES))
        .ok_or_else(|| format!("xdna stage byte size overflow for {rows}x{cols}"))
}

/// Validate a stage shape against the kernel's constraints.
fn check_shape(rows: usize, cols: usize) -> Result<(), String> {
    if rows == 0 || !rows.is_multiple_of(ROWS) {
        return Err(format!(
            "xdna stage rows must be a nonzero multiple of {ROWS}, got {rows}"
        ));
    }
    if cols == 0 || !cols.is_multiple_of(64) {
        return Err(format!(
            "xdna stage cols must be a nonzero multiple of 64, got {cols}"
        ));
    }
    let chunk = CHUNK_K.min(cols);
    if !cols.is_multiple_of(chunk) {
        return Err(format!(
            "xdna stage cols {cols} is not a multiple of its {chunk}-column chunk"
        ));
    }
    Ok(())
}

/// A source of 32-element blocks (`q - 8` in `[-8, 7]` plus `f32` scale),
/// addressed the way the packing loops consume them.
trait WeightSource {
    /// Read one 32-weight block `(signed int4 weights in [-8, 7], f32 scale)`.
    fn quant_block(&self, row: usize, block: usize) -> ([i8; QBLK], f32);
}

/// GGUF Q4_0 bytes: 18-byte blocks of f16 scale + 32 nibbles. Blocks beyond
/// `nb_src` read as zero so stages with padded K dimensions (e.g. LFM3 FFN
/// down `7680 -> 8192`) pack cleanly without intermediate allocation.
struct Q4_0Source<'a> {
    /// Raw tensor bytes.
    src: &'a [u8],
    /// Blocks per source row.
    nb_src: usize,
}

impl WeightSource for Q4_0Source<'_> {
    fn quant_block(&self, row: usize, block: usize) -> ([i8; QBLK], f32) {
        if block >= self.nb_src {
            return ([0i8; QBLK], 0.0);
        }
        let bytes = &self.src[(row * self.nb_src + block) * 18..];
        let d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let qs = &bytes[2..18];
        let mut q = [0i8; QBLK];
        for j in 0..16 {
            let b = qs[j];
            q[j] = (b & 0xF) as i8 - 8;
            q[j + 16] = (b >> 4) as i8 - 8;
        }
        (q, d)
    }
}

/// f32 weights quantized with the spike's Q4_0 rule (`d = mx / -8` over the
/// signed element of largest magnitude in the block).
struct F32Source<'a> {
    /// Row-major f32 weights.
    src: &'a [f32],
    /// Columns per row.
    cols: usize,
}

impl WeightSource for F32Source<'_> {
    fn quant_block(&self, row: usize, block: usize) -> ([i8; QBLK], f32) {
        let slice = &self.src[row * self.cols + block * QBLK..][..QBLK];
        let mut mx = 0.0f32;
        let mut amax = 0.0f32;
        for &v in slice {
            if v.is_nan() {
                return ([0i8; QBLK], f32::NAN);
            }
            let av = v.abs();
            if av > amax {
                amax = av;
                mx = v;
            }
        }
        let d = mx / -8.0;
        let mut q = [0i8; QBLK];
        if d != 0.0 && d.is_finite() {
            for (dst, &v) in q.iter_mut().zip(slice.iter()) {
                // The spike's `int(v * id + 8.5f)`: truncation toward zero, clamped.
                *dst = ((v / d + 8.5) as i32).clamp(0, 15) as i8 - 8;
            }
        }
        (q, d)
    }
}

/// Pack one stage from `source` into `dst` (zeroed here first; the packing
/// ORs nibbles in). Returns the byte count written.
#[allow(dead_code)]
fn pack_stage<S: WeightSource>(
    dst: &mut [u8],
    source: &S,
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    pack_stage_with_chunk_k(dst, source, rows, cols, CHUNK_K)
}

fn pack_stage_with_chunk_k<S: WeightSource>(
    dst: &mut [u8],
    source: &S,
    rows: usize,
    cols: usize,
    max_chunk_k: usize,
) -> Result<usize, String> {
    let need = stage_bytes(rows, cols)?;
    if dst.len() < need {
        return Err(format!(
            "xdna repack buffer too small: need {need}, have {}",
            dst.len()
        ));
    }
    let dst = &mut dst[..need];
    dst.fill(0);
    let chunk_k = max_chunk_k.min(cols);
    let chunks = cols / chunk_k;
    let nb_chunk = chunk_k / QBLK;
    let tile_bytes = nb_chunk * BLOCK_BYTES;
    for tile in 0..rows / ROWS {
        let r0 = tile * ROWS;
        for ch in 0..chunks {
            let tp = (tile * chunks + ch) * tile_bytes;
            for lb in 0..nb_chunk {
                let b = ch * nb_chunk + lb;
                let mut qs = [[0i8; QBLK]; ROWS];
                let mut scales = [0u16; ROWS];
                for n in 0..ROWS {
                    let (q, d) = source.quant_block(r0 + n, b);
                    qs[n] = q;
                    scales[n] = f32_to_bf16(d);
                }
                let bp = tp + lb * 256;
                for h in 0..2 {
                    for kk in 0..16 {
                        for (n, q_row) in qs.iter().enumerate() {
                            let e = kk * 16 + n;
                            let nib = (q_row[h * 16 + kk] & 0xF) as u8;
                            let byte = &mut dst[bp + h * 128 + e / 2];
                            if e % 2 == 0 {
                                *byte |= nib;
                            } else {
                                *byte |= nib << 4;
                            }
                        }
                    }
                }
                let sp = tp + nb_chunk * 256 + lb * ROWS * 2;
                for (n, &bits) in scales.iter().enumerate() {
                    dst[sp + n * 2..sp + n * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
        }
    }
    Ok(need)
}

/// Pack a GGUF Q4_0 tensor (`rows` x `cols`) into the kernel stream layout.
pub fn pack_q4_0_stage(
    dst: &mut [u8],
    src: &[u8],
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    pack_q4_0_stage_padded(dst, src, rows, cols, cols)
}

/// Pack a GGUF Q4_0 tensor (`rows` x `cols`) into 1024-K half-tiles (`9,216 B` each)
/// for `B=64` LFM2 NPU prefill weight reuse + `W_L2L1` double-buffering.
pub fn pack_q4_0_stage_k1024(
    dst: &mut [u8],
    src: &[u8],
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    check_shape(rows, cols)?;
    let nb_src = cols / QBLK;
    let expect = rows
        .checked_mul(nb_src)
        .and_then(|v| v.checked_mul(18))
        .ok_or_else(|| format!("Q4_0 byte size overflow for {rows}x{cols}"))?;
    if src.len() < expect {
        return Err(format!(
            "Q4_0 tensor too small: need {expect} bytes, have {}",
            src.len()
        ));
    }
    pack_stage_with_chunk_k(dst, &Q4_0Source { src, nb_src }, rows, cols, 1024)
}

/// Pack a GGUF Q4_0 tensor (`rows` x `src_cols`) into a zero-padded `padded_cols`
/// kernel stream layout (e.g. `7680` -> `8192` for LFM3 FFN down).
fn pack_q4_0_stage_padded_with_chunk_k(
    dst: &mut [u8],
    src: &[u8],
    rows: usize,
    src_cols: usize,
    padded_cols: usize,
    max_chunk_k: usize,
) -> Result<usize, String> {
    if src_cols == 0 || !src_cols.is_multiple_of(QBLK) || src_cols > padded_cols {
        return Err(format!(
            "xdna padded src_cols {src_cols} must be a nonzero multiple of {QBLK} <= padded_cols {padded_cols}"
        ));
    }
    check_shape(rows, padded_cols)?;
    let nb_src = src_cols / QBLK;
    let expect = rows
        .checked_mul(nb_src)
        .and_then(|v| v.checked_mul(18))
        .ok_or_else(|| format!("Q4_0 byte size overflow for {rows}x{src_cols}"))?;
    if src.len() < expect {
        return Err(format!(
            "Q4_0 tensor too small: need {expect} bytes, have {}",
            src.len()
        ));
    }
    pack_stage_with_chunk_k(
        dst,
        &Q4_0Source { src, nb_src },
        rows,
        padded_cols,
        max_chunk_k,
    )
}

pub fn pack_q4_0_stage_padded(
    dst: &mut [u8],
    src: &[u8],
    rows: usize,
    src_cols: usize,
    padded_cols: usize,
) -> Result<usize, String> {
    pack_q4_0_stage_padded_with_chunk_k(dst, src, rows, src_cols, padded_cols, CHUNK_K)
}

/// Pack a GGUF Q4_0 tensor (`rows` x `src_cols`) into zero-padded `padded_cols`
/// with 1024-K half-tiles (`9,216 B` each) for `B=64` LFM3 NPU prefill.
pub fn pack_q4_0_stage_padded_k1024(
    dst: &mut [u8],
    src: &[u8],
    rows: usize,
    src_cols: usize,
    padded_cols: usize,
) -> Result<usize, String> {
    pack_q4_0_stage_padded_with_chunk_k(dst, src, rows, src_cols, padded_cols, 1024)
}

fn pack_f32_stage_with_chunk_k(
    dst: &mut [u8],
    src: &[f32],
    rows: usize,
    cols: usize,
    max_chunk_k: usize,
) -> Result<usize, String> {
    check_shape(rows, cols)?;
    let expect = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("f32 element count overflow for {rows}x{cols}"))?;
    if src.len() < expect {
        return Err(format!(
            "f32 tensor too small: need {expect} elements, have {}",
            src.len()
        ));
    }
    pack_stage_with_chunk_k(dst, &F32Source { src, cols }, rows, cols, max_chunk_k)
}

/// Pack f32 weights (e.g. a dequantized BF16/Q6_K tensor) into 1024-K half-tiles
/// using Q4_0 quantization.
pub fn pack_f32_stage_k1024(
    dst: &mut [u8],
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    pack_f32_stage_with_chunk_k(dst, src, rows, cols, 1024)
}

/// f32 weights quantized into Q4_0 blocks via fast multi-scale + least-squares MSE search
/// aligned to the exact BF16 scale stored for the XDNA2 AIE kernel.
struct F32MseSource<'a> {
    src: &'a [f32],
    cols: usize,
}

impl WeightSource for F32MseSource<'_> {
    fn quant_block(&self, row: usize, block: usize) -> ([i8; QBLK], f32) {
        let slice = &self.src[row * self.cols + block * QBLK..][..QBLK];
        let mut amax = 0.0f32;
        let mut mx = 0.0f32;
        for &v in slice {
            if v.is_nan() {
                return ([0i8; QBLK], f32::NAN);
            }
            let av = v.abs();
            if av > amax {
                amax = av;
                mx = v;
            }
        }
        if amax == 0.0 {
            return ([0i8; QBLK], 0.0);
        }
        if !amax.is_finite() {
            return ([0i8; QBLK], mx / -8.0);
        }

        let quant_eval = |d_raw: f32| -> ([i8; QBLK], f32, f64) {
            let d_bf = f32::from_bits((f32_to_bf16(d_raw) as u32) << 16);
            if d_bf == 0.0 || !d_bf.is_finite() {
                return ([0i8; QBLK], 0.0, f64::INFINITY);
            }
            let mut q = [0i8; QBLK];
            let mut err = 0.0f64;
            for (dst, &v) in q.iter_mut().zip(slice.iter()) {
                let qi = ((v / d_bf + 8.5).floor() as i32).clamp(0, 15) - 8;
                *dst = qi as i8;
                let diff = (v as f64) - (qi as f64) * (d_bf as f64);
                err += diff * diff;
            }
            (q, d_bf, err)
        };

        let mut best = ([0i8; QBLK], 0.0f32, f64::INFINITY);
        let d_base = mx / -8.0;
        let candidates = [
            d_base,
            d_base * 0.90,
            d_base * 0.80,
            d_base * 0.72,
            -d_base,
            -d_base * 0.85,
        ];
        for cand in candidates {
            let trial = quant_eval(cand);
            if trial.2 < best.2 {
                best = trial;
            }
            // One least-squares step `d = sum(x * q) / sum(q^2)` followed by re-quantization.
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (&v, &qi) in slice.iter().zip(trial.0.iter()) {
                let qf = qi as f64;
                num += (v as f64) * qf;
                den += qf * qf;
            }
            if den > 0.0 {
                let refined = quant_eval((num / den) as f32);
                if refined.2 < best.2 {
                    best = refined;
                }
            }
        }
        (best.0, best.1)
    }
}

/// Pack f32 weights (e.g. dequantized Q4_K/Q6_K/BF16 tensors) into 1024-K half-tiles
/// using MSE-optimal BF16-aligned Q4_0 quantization.
pub fn pack_f32_mse_stage_k1024(
    dst: &mut [u8],
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    check_shape(rows, cols)?;
    let expect = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("f32 element count overflow for {rows}x{cols}"))?;
    if src.len() < expect {
        return Err(format!(
            "f32 tensor too small: need {expect} elements, have {}",
            src.len()
        ));
    }
    pack_stage_with_chunk_k(dst, &F32MseSource { src, cols }, rows, cols, 1024)
}

/// Pack f32 weights (e.g. a dequantized Q6_K head) into the stream layout,
/// quantizing with the spike's Q4_0 rule.
pub fn pack_f32_stage(
    dst: &mut [u8],
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    pack_f32_stage_with_chunk_k(dst, src, rows, cols, CHUNK_K)
}

/// Number of AIE2P workers across which every stage's row tiles are partitioned.
const WORKERS: usize = 16;

/// Interleave already-packed stage buffers per worker so a fused TaskGroup
/// (`q+k+v` or `gate+up`) streams all of worker `i`'s stage slices in a single
/// contiguous Shim DMA buffer descriptor. Every stage must have been packed at
/// width `cols` with at least one 16-row tile per worker (a multiple of 256 rows).
pub fn interleave_worker_stages(
    dst: &mut [u8],
    stages: &[&[u8]],
    cols: usize,
) -> Result<usize, String> {
    if stages.is_empty() {
        return Err("xdna interleave requires at least one stage".to_string());
    }
    let min_stage = stage_bytes(WORKERS * ROWS, cols)?;
    let mut total = 0usize;
    for (idx, s) in stages.iter().enumerate() {
        if s.is_empty() || !s.len().is_multiple_of(min_stage) {
            return Err(format!(
                "packed stage {idx} byte length {} is not a nonzero multiple of {min_stage} ({} rows x {cols} cols)",
                s.len(),
                WORKERS * ROWS
            ));
        }
        total = total
            .checked_add(s.len())
            .ok_or_else(|| "xdna interleave byte count overflow".to_string())?;
    }
    if dst.len() < total {
        return Err(format!(
            "xdna interleave buffer too small: need {total}, have {}",
            dst.len()
        ));
    }
    let mut wp = 0usize;
    for w in 0..WORKERS {
        for s in stages {
            let pw = s.len() / WORKERS;
            dst[wp..wp + pw].copy_from_slice(&s[w * pw..(w + 1) * pw]);
            wp += pw;
        }
    }
    Ok(total)
}

/// Supported `k_chunks` bucket counts (`K_pad = k_chunks * 64`) for `gemm_bf16`.
pub const GEMM_BF16_CHUNK_COUNTS: [usize; 14] =
    [1, 2, 4, 8, 12, 16, 18, 24, 32, 48, 64, 68, 96, 128];

/// Select the smallest `gemm_bf16` bucket `k_chunks` covering `cols` (up to `8192`).
pub fn gemm_bf16_bucket_chunks(cols: usize) -> usize {
    let need = cols.div_ceil(super::kernels::GEMM_BF16_K_CHUNK).max(1);
    GEMM_BF16_CHUNK_COUNTS
        .iter()
        .copied()
        .find(|&c| c >= need)
        .unwrap_or(128)
}

enum Bf16Dst<'a> {
    Aligned(&'a mut [u16]),
    Unaligned(&'a mut [u8]),
}

impl<'a> Bf16Dst<'a> {
    fn with(dst_bytes: &'a mut [u8], f: impl FnOnce(&mut Bf16Dst<'_>)) {
        if let Ok(words) = bytemuck::try_cast_slice_mut::<u8, u16>(dst_bytes) {
            f(&mut Bf16Dst::Aligned(words));
        } else {
            f(&mut Bf16Dst::Unaligned(dst_bytes));
        }
    }

    #[inline(always)]
    fn write_word(&mut self, word_idx: usize, val: u16) {
        match self {
            Self::Aligned(words) => words[word_idx] = val.to_le(),
            Self::Unaligned(bytes) => {
                let off = word_idx * 2;
                let le = val.to_le_bytes();
                bytes[off] = le[0];
                bytes[off + 1] = le[1];
            }
        }
    }
}

#[inline]
fn write_gemm_bf16_row(
    dst: &mut Bf16Dst<'_>,
    row: &[f32],
    c: usize,
    n_local: usize,
    col_start: usize,
    k_end: usize,
    k_chunks: usize,
) {
    use super::kernels::GEMM_BF16_K_CHUNK;
    let n8 = n_local / 8;
    let n_in = n_local % 8;
    for kc in 0..k_chunks {
        let k_base = col_start + kc * GEMM_BF16_K_CHUNK;
        if k_base >= k_end {
            break;
        }
        for k8 in 0..8 {
            let tile_base = ((((c * k_chunks + kc) * 8 + k8) * 4 + n8) * 8) * 8 + n_in;
            for k_in in 0..8 {
                let k = k_base + k8 * 8 + k_in;
                if k >= k_end {
                    break;
                }
                dst.write_word(tile_base + k_in * 8, f32_to_bf16(row[k]));
            }
        }
    }
}

/// Pack one `[<=256, <=k_chunks*64]` sub-matrix of row-major `f32` weights (`[total_rows, total_cols]`)
/// into `gemm_bf16`'s per-column `bfloat16` layout:
/// `[8 cols][k_chunks][k8=8][n8=4][8_k][8_n]` (`u16` words as little-endian bytes).
#[allow(clippy::too_many_arguments)]
pub fn pack_gemm_bf16_f32_slice(
    dst_bytes: &mut [u8],
    src_f32: &[f32],
    total_rows: usize,
    total_cols: usize,
    row_start: usize,
    col_start: usize,
    col_len: usize,
    k_chunks: usize,
) {
    use super::kernels::{GEMM_BF16_K_CHUNK, GEMM_BF16_N_COLS, GEMM_BF16_N_CORE, GEMM_BF16_N_TILE};

    let want_bytes = GEMM_BF16_N_TILE * k_chunks * GEMM_BF16_K_CHUNK * 2;
    assert_eq!(dst_bytes.len(), want_bytes);
    assert!(src_f32.len() >= total_rows * total_cols);
    dst_bytes.fill(0);

    let k_end = col_start.saturating_add(col_len).min(total_cols);
    Bf16Dst::with(dst_bytes, |dst| {
        for c in 0..GEMM_BF16_N_COLS {
            for n_local in 0..GEMM_BF16_N_CORE {
                let n = row_start + c * GEMM_BF16_N_CORE + n_local;
                if n >= total_rows {
                    break;
                }
                let row = &src_f32[n * total_cols..(n + 1) * total_cols];
                write_gemm_bf16_row(dst, row, c, n_local, col_start, k_end, k_chunks);
            }
        }
    });
}

/// Pack one `[<=256, <=k_chunks*64]` sub-matrix of an [`crate::model::weights::MmapWeight`]
/// into `gemm_bf16`'s `bfloat16` tile layout using `MmapWeight::dequantize_row`.
pub fn pack_gemm_bf16_weight_slice(
    dst_bytes: &mut [u8],
    weight: &crate::model::weights::MmapWeight,
    row_start: usize,
    col_start: usize,
    col_len: usize,
    k_chunks: usize,
) {
    use super::kernels::{GEMM_BF16_K_CHUNK, GEMM_BF16_N_COLS, GEMM_BF16_N_CORE, GEMM_BF16_N_TILE};

    let want_bytes = GEMM_BF16_N_TILE * k_chunks * GEMM_BF16_K_CHUNK * 2;
    assert_eq!(dst_bytes.len(), want_bytes);
    dst_bytes.fill(0);

    let k_end = col_start.saturating_add(col_len).min(weight.cols);
    let mut row_buf = vec![0.0f32; weight.cols];
    Bf16Dst::with(dst_bytes, |dst| {
        for c in 0..GEMM_BF16_N_COLS {
            for n_local in 0..GEMM_BF16_N_CORE {
                let n = row_start + c * GEMM_BF16_N_CORE + n_local;
                if n >= weight.rows {
                    break;
                }
                weight.dequantize_row(n, &mut row_buf);
                write_gemm_bf16_row(dst, &row_buf, c, n_local, col_start, k_end, k_chunks);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(-2.5), 0xC020);
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        // 0x3f80_8000 (1 + 2^-8) is midway between bf16 0x3F80 (even) and
        // 0x3F81 (odd): nearest-even keeps 0x3F80, and +1 f32 ULP rounds up.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3F81);
    }

    #[test]
    fn shape_validation_rejects_bad_shapes() {
        assert!(stage_bytes(0, 2048).is_err());
        assert!(stage_bytes(2048, 0).is_err());
        assert!(stage_bytes(0, 0).is_err());
        assert!(stage_bytes(100, 2048).is_err());
        assert!(stage_bytes(2048, 100).is_err());
        assert!(stage_bytes(512, 2048).is_ok());
        assert_eq!(stage_bytes(2048, 2048).unwrap(), 128 * 64 * 288);
        assert_eq!(stage_bytes(2048, 8192).unwrap(), 128 * 4 * 64 * 288);
    }

    /// Build Q4_0 bytes with scale 1.0 and a deterministic nibble pattern,
    /// pack them, then consume the packed block the way the kernel does and
    /// check every weight and scale survives the round trip.
    #[test]
    fn q4_0_pack_round_trips_through_kernel_layout() {
        let rows = 16;
        let cols = 64;
        let nb = cols / QBLK;
        let mut src = vec![0u8; rows * nb * 18];
        for r in 0..rows {
            for b in 0..nb {
                let base = (r * nb + b) * 18;
                src[base..base + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // f16 1.0
                for j in 0..16 {
                    let lo = ((r + b * 3 + j) % 16) as u8;
                    let hi = ((r * 2 + b + j) % 16) as u8;
                    src[base + 2 + j] = lo | (hi << 4);
                }
            }
        }
        let mut packed = vec![0u8; stage_bytes(rows, cols).unwrap()];
        let written = pack_q4_0_stage(&mut packed, &src, rows, cols).unwrap();
        assert_eq!(written, packed.len());

        // Consume: one tile, one chunk (K=64 -> nb_chunk=2), block lb.
        let nb_chunk = cols / QBLK;
        for lb in 0..nb_chunk {
            let bp = lb * 256;
            let sp = nb_chunk * 256 + lb * ROWS * 2;
            for n in 0..ROWS {
                let scale = u16::from_le_bytes([packed[sp + n * 2], packed[sp + n * 2 + 1]]);
                assert_eq!(scale, f32_to_bf16(1.0), "row {n} block {lb} scale");
                for h in 0..2 {
                    for kk in 0..16 {
                        let e = kk * 16 + n;
                        let byte = packed[bp + h * 128 + e / 2];
                        let nib = if e % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        // Packed nibbles are signed int4 (Q4_0's q - 8).
                        let signed = if nib >= 8 { nib as i8 - 16 } else { nib as i8 };
                        let j = h * 16 + kk;
                        let want = if j < 16 {
                            (src[((n * nb + lb) * 18) + 2 + j] & 0xF) as i8 - 8
                        } else {
                            (src[((n * nb + lb) * 18) + 2 + (j - 16)] >> 4) as i8 - 8
                        };
                        assert_eq!(signed, want, "row {n} block {lb} elem {j}");
                    }
                }
            }
        }
    }

    #[test]
    fn f32_pack_quantizes_with_spike_rule() {
        // Row 0's block holds 100.0 as its largest-magnitude element, so the
        // spike rule gives d = 100 / -8 = -12.5 and 100.0 itself packs to
        // the -8 extreme, reconstructing as -8 * -12.5 = 100.
        let rows = 16;
        let cols = 64;
        let mut src = vec![0.0f32; rows * cols];
        src[0] = -8.0;
        src[1] = 7.0;
        src[2] = 100.0;
        let mut packed = vec![0u8; stage_bytes(rows, cols).unwrap()];
        pack_f32_stage(&mut packed, &src, rows, cols).unwrap();
        let sp = 512; // nb_chunk=2: scales after 2 * 256 int4 bytes
        let scale_bits = u16::from_le_bytes([packed[sp], packed[sp + 1]]);
        assert_eq!(scale_bits, f32_to_bf16(-12.5));

        let unpack_n0 = |col: usize| -> i8 {
            let e = col * 16;
            let byte = packed[e / 2];
            let nib = byte & 0xF;
            if nib >= 8 { nib as i8 - 16 } else { nib as i8 }
        };
        assert_eq!(unpack_n0(0), 1);
        assert_eq!(unpack_n0(1), -1);
        assert_eq!(unpack_n0(2), -8);
        assert_eq!(unpack_n0(3), 0);

        // Multi-tile, multi-chunk round-trip (rows = 32, cols = 4096).
        let mt_rows = 32;
        let mt_cols = 4096;
        let mut mt_src = vec![0.0f32; mt_rows * mt_cols];
        for r in 0..mt_rows {
            mt_src[r * mt_cols + 2048] = -((r + 1) as f32) * 8.0;
        }
        let mut mt_packed = vec![0u8; stage_bytes(mt_rows, mt_cols).unwrap()];
        assert_eq!(
            pack_f32_stage(&mut mt_packed, &mt_src, mt_rows, mt_cols).unwrap(),
            mt_packed.len()
        );
        let tile = 1;
        let ch = 1;
        let chunks = mt_cols / CHUNK_K;
        let nb_chunk = CHUNK_K / QBLK;
        let tp = (tile * chunks + ch) * (nb_chunk * BLOCK_BYTES);
        let sp2 = tp + nb_chunk * 256;
        let scale_row16 = u16::from_le_bytes([mt_packed[sp2], mt_packed[sp2 + 1]]);
        assert_eq!(scale_row16, f32_to_bf16(17.0));
        let nib_row16_c2048 = mt_packed[tp] & 0xF;
        assert_eq!(
            if nib_row16_c2048 >= 8 {
                nib_row16_c2048 as i8 - 16
            } else {
                nib_row16_c2048 as i8
            },
            -8
        );

        let mut too_small = vec![0u8; mt_packed.len() - 1];
        assert!(pack_f32_stage(&mut too_small, &mt_src, mt_rows, mt_cols).is_err());
        assert!(pack_q4_0_stage(&mut too_small, &[], rows, cols).is_err());
    }

    #[test]
    fn bf16_preserves_quiet_nans_across_payloads() {
        for bits in [
            0x7F80_0001u32,
            0x7FC0_0000u32,
            0x7FFF_FFFFu32,
            0xFFC0_0001u32,
            0xFFFF_FFFFu32,
        ] {
            let h = f32_to_bf16(f32::from_bits(bits));
            let back = f32::from_bits((h as u32) << 16);
            assert!(
                back.is_nan(),
                "f32 NaN {bits:#010x} rounded to non-NaN bf16 {h:#06x} ({back})"
            );
        }
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xFF80);
        assert!(stage_bytes(16, 2112).is_err());
    }

    #[test]
    fn f32_pack_propagates_nan_scale_instead_of_zeroing_block() {
        let rows = 16;
        let cols = 64;
        let mut src = vec![1.0f32; rows * cols];
        src[5] = f32::NAN;
        let mut packed = vec![0u8; stage_bytes(rows, cols).unwrap()];
        pack_f32_stage(&mut packed, &src, rows, cols).unwrap();
        let sp = 512;
        let scale_bits = u16::from_le_bytes([packed[sp], packed[sp + 1]]);
        assert!(
            f32::from_bits((scale_bits as u32) << 16).is_nan(),
            "NaN weight block must preserve NaN scale instead of zeroing: {scale_bits:#06x}"
        );
    }

    #[test]
    fn f32_mse_pack_reduces_outlier_block_error() {
        let rows = 16;
        let cols = 64;
        let mut src = vec![0.0f32; rows * cols];
        // One outlier (+3.1) alongside 31 values at +-0.47 (not a multiple of 3.1/8)
        // in block 0 of row 0.
        for (i, dst) in src.iter_mut().enumerate().take(32) {
            *dst = if i % 2 == 0 { 0.47 } else { -0.47 };
        }
        src[0] = 3.1;

        let mut naive = vec![0u8; stage_bytes(rows, cols).unwrap()];
        let mut mse = vec![0u8; naive.len()];
        pack_f32_stage_k1024(&mut naive, &src, rows, cols).unwrap();
        pack_f32_mse_stage_k1024(&mut mse, &src, rows, cols).unwrap();

        let block_err = |packed: &[u8]| -> f32 {
            let nb_chunk = cols / QBLK;
            let sp = nb_chunk * 256;
            let scale_bits = u16::from_le_bytes([packed[sp], packed[sp + 1]]);
            let d = f32::from_bits((scale_bits as u32) << 16);
            let mut err = 0.0f32;
            for (j, &orig) in src.iter().enumerate().take(32) {
                let h = j / 16;
                let kk = j % 16;
                let e = kk * 16;
                let b = packed[h * 128 + e / 2];
                let nib = if e % 2 == 0 { b & 0xF } else { b >> 4 };
                let q = if nib >= 8 { nib as i8 - 16 } else { nib as i8 };
                let diff = orig - (q as f32) * d;
                err += diff * diff;
            }
            err
        };
        let e_naive = block_err(&naive);
        let e_mse = block_err(&mse);
        assert!(
            e_mse < e_naive * 0.6,
            "MSE Q4_0 packing should substantially reduce outlier error: e_mse={e_mse} vs e_naive={e_naive}"
        );
    }

    #[test]
    fn interleave_worker_stages_groups_worker_slices_contiguously() {
        let cols = 64;
        let tile = stage_bytes(ROWS, cols).unwrap(); // 576 bytes per 16-row tile at K=64
        let mut s0 = vec![0u8; WORKERS * 2 * tile];
        let mut s1 = vec![0u8; WORKERS * tile];
        for w in 0..WORKERS {
            s0[w * 2 * tile..(w + 1) * 2 * tile].fill((w * 2) as u8);
            s1[w * tile..(w + 1) * tile].fill((100 + w) as u8);
        }
        let mut dst = vec![0u8; s0.len() + s1.len()];
        let written = interleave_worker_stages(&mut dst, &[&s0, &s1], cols).unwrap();
        assert_eq!(written, dst.len());
        for w in 0..WORKERS {
            let base = w * 3 * tile;
            assert!(
                dst[base..base + 2 * tile]
                    .iter()
                    .all(|&b| b == (w * 2) as u8)
            );
            assert!(
                dst[base + 2 * tile..base + 3 * tile]
                    .iter()
                    .all(|&b| b == (100 + w) as u8)
            );
        }
        assert!(interleave_worker_stages(&mut dst, &[], cols).is_err());
        assert!(interleave_worker_stages(&mut dst, &[&[]], cols).is_err());
        // Single 16-row tile (divisible by 16 bytes, but < 256 rows) must be rejected.
        let one_tile = vec![0u8; tile];
        assert!(interleave_worker_stages(&mut dst, &[&one_tile], cols).is_err());
        assert!(stage_bytes(usize::MAX - (usize::MAX % ROWS), 2048).is_err());
    }

    #[test]
    fn gemm_bf16_bucket_chunks_selects_smallest_covering_bucket() {
        assert_eq!(gemm_bf16_bucket_chunks(0), 1);
        assert_eq!(gemm_bf16_bucket_chunks(1), 1);
        assert_eq!(gemm_bf16_bucket_chunks(64), 1);
        assert_eq!(gemm_bf16_bucket_chunks(65), 2);
        assert_eq!(gemm_bf16_bucket_chunks(128), 2);
        assert_eq!(gemm_bf16_bucket_chunks(129), 4);
        assert_eq!(gemm_bf16_bucket_chunks(1024), 16);
        assert_eq!(gemm_bf16_bucket_chunks(1025), 18);
        assert_eq!(gemm_bf16_bucket_chunks(1152), 18);
        assert_eq!(gemm_bf16_bucket_chunks(4304), 68);
        assert_eq!(gemm_bf16_bucket_chunks(8192), 128);
        assert_eq!(gemm_bf16_bucket_chunks(12000), 128);
        for cols in [0usize, 1, 64, 65, 512, 1152, 4304, 8192] {
            let c = gemm_bf16_bucket_chunks(cols);
            assert!(GEMM_BF16_CHUNK_COUNTS.contains(&c));
        }
    }

    #[test]
    fn pack_gemm_bf16_f32_and_weight_slice_round_trip_and_unaligned() {
        use super::super::kernels::{
            GEMM_BF16_K_CHUNK, GEMM_BF16_N_COLS, GEMM_BF16_N_CORE, GEMM_BF16_N_TILE,
        };
        use crate::model::weights::MmapWeight;
        use crate::tensor::DType;

        let total_rows = 300usize;
        let total_cols = 160usize;
        let mut src_f32 = vec![0.0f32; total_rows * total_cols];
        for r in 0..total_rows {
            for c in 0..total_cols {
                src_f32[r * total_cols + c] = (((r * 13 + c * 7) % 101) as f32 - 50.0) * 0.03125;
            }
        }

        // Build Q8_0 MmapWeight for the weight-slice check (160 cols = 5 * 32 blocks).
        let blocks_per_row = total_cols / 32;
        let mut q8_bytes = Vec::with_capacity(total_rows * blocks_per_row * 34);
        for r in 0..total_rows {
            for b in 0..blocks_per_row {
                let blk = &src_f32[r * total_cols + b * 32..r * total_cols + (b + 1) * 32];
                let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                let d_f16 = half::f16::from_f32(d);
                q8_bytes.extend_from_slice(&d_f16.to_le_bytes());
                let d_real = d_f16.to_f32();
                for &v in blk {
                    q8_bytes.push((v / d_real).round().clamp(-127.0, 127.0) as i8 as u8);
                }
            }
        }
        let mmap_w = MmapWeight::from_owned_bytes(q8_bytes, DType::Q8_0, total_rows, total_cols);
        let mut dq_ref = vec![0.0f32; total_rows * total_cols];
        for r in 0..total_rows {
            mmap_w.dequantize_row(r, &mut dq_ref[r * total_cols..(r + 1) * total_cols]);
        }

        for (row_start, col_start, col_len, k_chunks) in [
            (0usize, 0usize, 128usize, 2usize),
            (256usize, 128usize, 32usize, 1usize),
        ] {
            let want_bytes = GEMM_BF16_N_TILE * k_chunks * GEMM_BF16_K_CHUNK * 2;
            let mut aligned_f32 = vec![0xAAu8; want_bytes];
            let mut unaligned_f32_backing = vec![0xBBu8; want_bytes + 1];
            pack_gemm_bf16_f32_slice(
                &mut aligned_f32,
                &src_f32,
                total_rows,
                total_cols,
                row_start,
                col_start,
                col_len,
                k_chunks,
            );
            pack_gemm_bf16_f32_slice(
                &mut unaligned_f32_backing[1..],
                &src_f32,
                total_rows,
                total_cols,
                row_start,
                col_start,
                col_len,
                k_chunks,
            );
            assert_eq!(aligned_f32, unaligned_f32_backing[1..]);

            let mut aligned_w = vec![0xAAu8; want_bytes];
            let mut unaligned_w_backing = vec![0xCCu8; want_bytes + 1];
            pack_gemm_bf16_weight_slice(
                &mut aligned_w,
                &mmap_w,
                row_start,
                col_start,
                col_len,
                k_chunks,
            );
            pack_gemm_bf16_weight_slice(
                &mut unaligned_w_backing[1..],
                &mmap_w,
                row_start,
                col_start,
                col_len,
                k_chunks,
            );
            assert_eq!(aligned_w, unaligned_w_backing[1..]);

            let k_end = (col_start + col_len).min(total_cols);
            for c in 0..GEMM_BF16_N_COLS {
                for n_local in 0..GEMM_BF16_N_CORE {
                    let n = row_start + c * GEMM_BF16_N_CORE + n_local;
                    let n8 = n_local / 8;
                    let n_in = n_local % 8;
                    for kc in 0..k_chunks {
                        let k_base = col_start + kc * GEMM_BF16_K_CHUNK;
                        for k8 in 0..8 {
                            let tile_base =
                                ((((c * k_chunks + kc) * 8 + k8) * 4 + n8) * 8) * 8 + n_in;
                            for k_in in 0..8 {
                                let k = k_base + k8 * 8 + k_in;
                                let off = (tile_base + k_in * 8) * 2;
                                let got_f32 =
                                    u16::from_le_bytes([aligned_f32[off], aligned_f32[off + 1]]);
                                let got_w =
                                    u16::from_le_bytes([aligned_w[off], aligned_w[off + 1]]);
                                if n < total_rows && k < k_end {
                                    assert_eq!(
                                        got_f32,
                                        f32_to_bf16(src_f32[n * total_cols + k]),
                                        "f32 mismatch at n={n} k={k}"
                                    );
                                    assert_eq!(
                                        got_w,
                                        f32_to_bf16(dq_ref[n * total_cols + k]),
                                        "weight mismatch at n={n} k={k}"
                                    );
                                } else {
                                    assert_eq!(got_f32, 0, "f32 pad non-zero at n={n} k={k}");
                                    assert_eq!(got_w, 0, "weight pad non-zero at n={n} k={k}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f32_mse_pack_handles_extreme_finite_and_infinite_blocks() {
        let rows = 16;
        let cols = 64;
        let mut src = vec![0.0f32; rows * cols];
        // Row 0, block 0: finite values >= 1e20 (used to overflow f32 MSE `err` to Inf).
        for (i, dst) in src.iter_mut().enumerate().take(32) {
            *dst = ((i as f32) - 15.5) * 1.0e20;
        }
        // Row 1, block 0: near f32::MAX / 8.0 scale regime.
        for i in 0..32 {
            src[cols + i] = if i % 2 == 0 { 1.0e37 } else { -5.0e36 };
        }
        // Row 2, block 0: +Inf; Row 3, block 0: -Inf.
        src[2 * cols] = f32::INFINITY;
        src[3 * cols] = f32::NEG_INFINITY;

        let mut packed = vec![0u8; stage_bytes(rows, cols).unwrap()];
        pack_f32_mse_stage_k1024(&mut packed, &src, rows, cols).unwrap();

        let nb_chunk = cols / QBLK;
        let sp = nb_chunk * 256;
        let read_scale = |row: usize| -> f32 {
            let bits = u16::from_le_bytes([packed[sp + row * 2], packed[sp + row * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        };
        let read_q = |row: usize, col: usize| -> i8 {
            let e = col * 16 + row;
            let byte = packed[e / 2];
            let nib = if e.is_multiple_of(2) {
                byte & 0xF
            } else {
                byte >> 4
            };
            if nib >= 8 { nib as i8 - 16 } else { nib as i8 }
        };

        let d0 = read_scale(0);
        assert!(
            d0.is_finite() && d0 != 0.0,
            "extreme 1e20 block collapsed: d0={d0}"
        );
        let rec0 = (read_q(0, 31) as f32) * d0;
        assert!(
            (rec0 - src[31]).abs() / src[31].abs() < 0.25,
            "extreme 1e20 reconstruction drifted: got {rec0} vs {}",
            src[31]
        );

        let d1 = read_scale(1);
        assert!(
            d1.is_finite() && d1 != 0.0,
            "extreme 1e37 block collapsed: d1={d1}"
        );
        let rec1 = (read_q(1, 0) as f32) * d1;
        assert!(
            (rec1 - src[cols]).abs() / src[cols].abs() < 0.25,
            "extreme 1e37 reconstruction drifted: got {rec1} vs {}",
            src[cols]
        );

        // Non-finite amax preserves `mx / -8.0` like F32Source.
        assert_eq!(read_scale(2), f32::NEG_INFINITY);
        assert_eq!(read_scale(3), f32::INFINITY);
    }
}

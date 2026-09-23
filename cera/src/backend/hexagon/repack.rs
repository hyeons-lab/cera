//! Host-side weight repacking for Hexagon Tensor Processor (HTP) kernels.
//!
//! Quantized weight matrices (Q4_0, Q8_0, Q4_K, Q6_K) must be repacked from
//! linear GGUF block order into 32x32 tiled layout (one 1088-byte tile per
//! 32 rows x 32 columns for Q8_0, 576 bytes for Q4_0, 640 for Q4_K, 896 for
//! Q6_K) so HVX vector units can stream quants and scales with aligned SIMD
//! loads. Mirrors upstream `repack_q8_0_tiled` / `repack_q4_0_tiled` /
//! `repack_q4_K_tiled` / `repack_q6_K_tiled`. Q5_K has no DSP wire format and
//! is requanted to Q8_0 on the host ([`requant_q5_k_to_q8_0`]) before repack.

use crate::quant::{
    BlockQ4_0, BlockQ4KM, BlockQ5K, BlockQ6K, BlockQ8_0, decode_q4km_scales, dequantize_q5_k_block,
    f16_to_f32, f32_to_f16,
};
use crate::session::CeraError;

/// Packed tile size in bytes for Q8_0 weights (1024 quant bytes + 64 scale bytes).
pub const TILE_SIZE_Q8_0: usize = 1088;
/// Packed tile size in bytes for Q4_0 weights (512 quant bytes + 64 scale bytes).
pub const TILE_SIZE_Q4_0: usize = 576;
/// Packed tile size in bytes for Q4_K weights (512 quant bytes + 128 scale/min bytes).
pub const TILE_SIZE_Q4_K: usize = 640;
/// Packed tile size in bytes for Q6_K weights (512 low + 256 high + 128 scale bytes).
pub const TILE_SIZE_Q6_K: usize = 896;
/// Quant bytes per Q8_0 tile (32 rows x 32 quants).
const TILE_QUANTS_Q8_0: usize = 1024;
/// Quant bytes per Q4_0 tile (32 rows x 16 packed bytes).
const TILE_QUANTS_Q4_0: usize = 512;
/// Quant bytes per Q4_K tile (16 pairs x 32 column-interleaved bytes).
const TILE_QUANTS_Q4_K: usize = 512;
/// Low-nibble bytes per Q6_K tile (4 vectors x 128 bytes).
const TILE_LO_Q6_K: usize = 512;
/// High-bit bytes per Q6_K tile (2 vectors x 128 bytes).
const TILE_HI_Q6_K: usize = 256;

fn ceil32(x: usize) -> usize {
    x.next_multiple_of(32)
}

/// Calculate the total buffer size in bytes for a repacked Q4_0 matrix.
pub fn repacked_matrix_size_q4_0(ne0: usize, ne1: usize) -> usize {
    (ceil32(ne1) / 32) * (ceil32(ne0) / 32) * TILE_SIZE_Q4_0
}

/// Calculate the total buffer size in bytes for a repacked Q8_0 matrix.
pub fn repacked_matrix_size_q8_0(ne0: usize, ne1: usize) -> usize {
    (ceil32(ne1) / 32) * (ceil32(ne0) / 32) * TILE_SIZE_Q8_0
}

/// Calculate the total buffer size in bytes for a repacked Q4_K matrix.
pub fn repacked_matrix_size_q4_k(ne0: usize, ne1: usize) -> usize {
    (ceil32(ne1) / 32) * (ceil32(ne0) / 32) * TILE_SIZE_Q4_K
}

/// Calculate the total buffer size in bytes for a repacked Q6_K matrix.
pub fn repacked_matrix_size_q6_k(ne0: usize, ne1: usize) -> usize {
    (ceil32(ne1) / 32) * (ceil32(ne0) / 32) * TILE_SIZE_Q6_K
}

/// Repack a linear GGUF Q8_0 weight matrix into HTP 32x32 tiled layout.
///
/// The matrix is covered by column tiles (`ct`) of 32 rows and K tiles (`kt`)
/// of 32 columns. Each tile holds 32 rows of interleaved int8 quants
/// (`tile[cp * 64 + 2 * row + {0,1}] = qs[2 * cp + {0,1}]` for quant pairs
/// `cp` in 0..16) followed by 32 f16 scales, one per row.
pub fn repack_q8_0(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    let blocks_per_row = ne0.div_ceil(32);
    let block_size = std::mem::size_of::<BlockQ8_0>();
    let total_src_bytes = ne1 * blocks_per_row * block_size;
    if src_bytes.len() < total_src_bytes {
        return Err(CeraError::Backend(format!(
            "repack_q8_0: source buffer too short (expected {} bytes, got {})",
            total_src_bytes,
            src_bytes.len()
        )));
    }

    let n_col_tiles = ceil32(ne1) / 32;
    let n_k_tiles = ceil32(ne0) / 32;
    let matrix_size = n_col_tiles * n_k_tiles * TILE_SIZE_Q8_0;
    if dst.len() < matrix_size {
        return Err(CeraError::Backend(format!(
            "repack_q8_0: destination buffer too short (expected {} bytes, got {})",
            matrix_size,
            dst.len()
        )));
    }
    // Zero the tail so padded rows/cols read back as zero weights.
    dst[..matrix_size].fill(0);

    for ct in 0..n_col_tiles {
        for kt in 0..n_k_tiles {
            let tile_dst = &mut dst[(ct * n_k_tiles + kt) * TILE_SIZE_Q8_0..][..TILE_SIZE_Q8_0];
            let (tile_quants, tile_scales) = tile_dst.split_at_mut(TILE_QUANTS_Q8_0);
            for cp in 0..16 {
                for row in 0..32 {
                    let r = ct * 32 + row;
                    if r < ne1 && kt < ne0 / 32 {
                        let blk_off = (r * blocks_per_row + kt) * block_size;
                        let quants = &src_bytes[blk_off + 2..blk_off + 34];
                        tile_quants[cp * 64 + 2 * row] = quants[2 * cp];
                        tile_quants[cp * 64 + 2 * row + 1] = quants[2 * cp + 1];
                    }
                }
            }
            for row in 0..32 {
                let r = ct * 32 + row;
                if r < ne1 && kt < ne0 / 32 {
                    let blk_off = (r * blocks_per_row + kt) * block_size;
                    tile_scales[row * 2..row * 2 + 2]
                        .copy_from_slice(&src_bytes[blk_off..blk_off + 2]);
                }
            }
        }
    }

    Ok(())
}

/// Repack a linear GGUF Q4_0 weight matrix into HTP 32x32 tiled layout.
///
/// Same tiling as Q8_0, but each block's 16 packed nibble bytes are first
/// unpacked to 32 nibbles (low nibble = quants 0..16, high nibble = quants
/// 16..32; padded entries read 8, the zero point) and re-packed column-wise:
/// `tile[cp * 32 + row] = (q[2 * cp + 1] << 4) | q[2 * cp]`.
pub fn repack_q4_0(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    let blocks_per_row = ne0.div_ceil(32);
    let block_size = std::mem::size_of::<BlockQ4_0>();
    let total_src_bytes = ne1 * blocks_per_row * block_size;
    if src_bytes.len() < total_src_bytes {
        return Err(CeraError::Backend(format!(
            "repack_q4_0: source buffer too short (expected {} bytes, got {})",
            total_src_bytes,
            src_bytes.len()
        )));
    }

    let n_col_tiles = ceil32(ne1) / 32;
    let n_k_tiles = ceil32(ne0) / 32;
    let matrix_size = n_col_tiles * n_k_tiles * TILE_SIZE_Q4_0;
    if dst.len() < matrix_size {
        return Err(CeraError::Backend(format!(
            "repack_q4_0: destination buffer too short (expected {} bytes, got {})",
            matrix_size,
            dst.len()
        )));
    }
    dst[..matrix_size].fill(0);

    for ct in 0..n_col_tiles {
        for kt in 0..n_k_tiles {
            let tile_dst = &mut dst[(ct * n_k_tiles + kt) * TILE_SIZE_Q4_0..][..TILE_SIZE_Q4_0];
            let (tile_quants, tile_scales) = tile_dst.split_at_mut(TILE_QUANTS_Q4_0);

            let mut quants = [[8u8; 32]; 32];
            for row in 0..32 {
                let r = ct * 32 + row;
                if r < ne1 && kt < ne0 / 32 {
                    let blk_off = (r * blocks_per_row + kt) * block_size;
                    let packed = &src_bytes[blk_off + 2..blk_off + 18];
                    for (i, q) in quants[row].iter_mut().enumerate() {
                        *q = if i < 16 {
                            packed[i] & 0x0f
                        } else {
                            packed[i - 16] >> 4
                        };
                    }
                    tile_scales[row * 2..row * 2 + 2]
                        .copy_from_slice(&src_bytes[blk_off..blk_off + 2]);
                }
            }
            for cp in 0..16 {
                for row in 0..32 {
                    tile_quants[cp * 32 + row] =
                        (quants[row][2 * cp + 1] << 4) | quants[row][2 * cp];
                }
            }
        }
    }

    Ok(())
}

/// Repack a linear GGUF Q4_K weight matrix into HTP 32x32 tiled layout.
///
/// Mirrors `repack_q4_K_tiled` (Q4_1 wire format, 640-byte tiles). Each
/// super-block (256 K values) feeds 8 K tiles; tile `(ct, kt)` takes the
/// `kt_local = kt % 8` sub-block's 32 quants, column-interleaved as
/// `tile[cp * 32 + row] = (q[2*cp+1] << 4) | q[2*cp]`, with per-row f16
/// `D = d * sc` / `M = -dmin * m` scale pairs (`tile[512 + 4*row ..]`).
pub fn repack_q4_k(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    if !ne0.is_multiple_of(256) {
        return Err(CeraError::Backend(format!(
            "repack_q4_k: K dim {ne0} is not a multiple of 256"
        )));
    }
    let sb_per_row = ne0 / 256;
    let block_size = std::mem::size_of::<BlockQ4KM>();
    let total_src_bytes = ne1 * sb_per_row * block_size;
    if src_bytes.len() < total_src_bytes {
        return Err(CeraError::Backend(format!(
            "repack_q4_k: source buffer too short (expected {} bytes, got {})",
            total_src_bytes,
            src_bytes.len()
        )));
    }

    let n_col_tiles = ceil32(ne1) / 32;
    let n_k_tiles = ceil32(ne0) / 32;
    let matrix_size = n_col_tiles * n_k_tiles * TILE_SIZE_Q4_K;
    if dst.len() < matrix_size {
        return Err(CeraError::Backend(format!(
            "repack_q4_k: destination buffer too short (expected {} bytes, got {})",
            matrix_size,
            dst.len()
        )));
    }
    dst[..matrix_size].fill(0);

    for r in 0..ne1 {
        let ct = r / 32;
        let row = r % 32;
        for kt in 0..n_k_tiles {
            let kt_local = kt % 8;
            let blk_off = (r * sb_per_row + kt / 8) * block_size;
            let b = &src_bytes[blk_off..blk_off + block_size];
            let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
            let scales: &[u8; 12] = b[4..16].try_into().unwrap();
            let (sc, mn) = decode_q4km_scales(scales);
            let dd = d * f32::from(sc[kt_local]);
            let mm = -dmin * f32::from(mn[kt_local]);

            let tile_dst = &mut dst[(ct * n_k_tiles + kt) * TILE_SIZE_Q4_K..][..TILE_SIZE_Q4_K];
            let (tile_quants, tile_scales) = tile_dst.split_at_mut(TILE_QUANTS_Q4_K);
            let qs_base = 16 + (kt_local / 2) * 32;
            let shift = (kt_local & 1) * 4;
            for cp in 0..16 {
                let q0 = (b[qs_base + 2 * cp] >> shift) & 0x0f;
                let q1 = (b[qs_base + 2 * cp + 1] >> shift) & 0x0f;
                tile_quants[cp * 32 + row] = (q1 << 4) | q0;
            }
            tile_scales[row * 4..row * 4 + 2].copy_from_slice(&f32_to_f16(dd).to_le_bytes());
            tile_scales[row * 4 + 2..row * 4 + 4].copy_from_slice(&f32_to_f16(mm).to_le_bytes());
        }
    }

    Ok(())
}

/// Unsigned 6-bit value (0..63) of element `e` of a Q6_K block.
///
/// Same bit layout as `dequantize_row_q6_K`: `ql` holds the low nibbles
/// (interleaved across the two 128-element halves), `qh` the high 2 bits.
fn q6_k_get_quant(block: &[u8], e: usize) -> u8 {
    let c = e / 128;
    let w = e % 128;
    let g = w / 32;
    let l = w % 32;
    let ql = |i: usize| block[i];
    let qh = |i: usize| block[128 + i];
    let (lo, hi) = match g {
        0 => (ql(c * 64 + l) & 0x0f, qh(c * 32 + l) & 3),
        1 => (ql(c * 64 + l + 32) & 0x0f, (qh(c * 32 + l) >> 2) & 3),
        2 => (ql(c * 64 + l) >> 4, (qh(c * 32 + l) >> 4) & 3),
        _ => (ql(c * 64 + l + 32) >> 4, (qh(c * 32 + l) >> 6) & 3),
    };
    lo | (hi << 4)
}

/// Repack a linear GGUF Q6_K weight matrix into HTP 32x32 tiled layout.
///
/// Mirrors `repack_q6_K_tiled` (896-byte tiles): low nibbles go to the
/// 512-byte `lo` plane (`lo[(g>>1)*128 + row*4 + (lk&3)]`), high 2-bit
/// pairs to the 256-byte `hi` plane (`hi[(g>>2)*128 + ...]`), and per-row
/// f16 `d * scales[]` pairs (`tile[768 + 64*sub + 2*row ..]`). The OR-ed
/// nibble writes require the zeroed tiles above.
pub fn repack_q6_k(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    if !ne0.is_multiple_of(256) {
        return Err(CeraError::Backend(format!(
            "repack_q6_k: K dim {ne0} is not a multiple of 256"
        )));
    }
    let sb_per_row = ne0 / 256;
    let block_size = std::mem::size_of::<BlockQ6K>();
    let total_src_bytes = ne1 * sb_per_row * block_size;
    if src_bytes.len() < total_src_bytes {
        return Err(CeraError::Backend(format!(
            "repack_q6_k: source buffer too short (expected {} bytes, got {})",
            total_src_bytes,
            src_bytes.len()
        )));
    }

    let n_col_tiles = ceil32(ne1) / 32;
    let n_k_tiles = ceil32(ne0) / 32;
    let matrix_size = n_col_tiles * n_k_tiles * TILE_SIZE_Q6_K;
    if dst.len() < matrix_size {
        return Err(CeraError::Backend(format!(
            "repack_q6_k: destination buffer too short (expected {} bytes, got {})",
            matrix_size,
            dst.len()
        )));
    }
    dst[..matrix_size].fill(0);

    for r in 0..ne1 {
        let ct = r / 32;
        let row = r % 32;
        for kt in 0..n_k_tiles {
            let kt_local = kt % 8;
            let blk_off = (r * sb_per_row + kt / 8) * block_size;
            let b = &src_bytes[blk_off..blk_off + block_size];
            let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));

            let tile = &mut dst[(ct * n_k_tiles + kt) * TILE_SIZE_Q6_K..][..TILE_SIZE_Q6_K];
            let (lo_pl, rest) = tile.split_at_mut(TILE_LO_Q6_K);
            let (hi_pl, sc_pl) = rest.split_at_mut(TILE_HI_Q6_K);
            for lk in 0..32 {
                let q6 = q6_k_get_quant(b, kt_local * 32 + lk);
                let g = lk >> 2;
                let pos = row * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (q6 & 0x0f) << ((g & 1) * 4);
                hi_pl[(g >> 2) * 128 + pos] |= (q6 >> 4) << ((g & 3) * 2);
            }
            for sub in 0..2 {
                let s = d * f32::from(b[192 + kt_local * 2 + sub] as i8);
                let off = sub * 64 + row * 2;
                sc_pl[off..off + 2].copy_from_slice(&f32_to_f16(s).to_le_bytes());
            }
        }
    }

    Ok(())
}

/// Quantize 32 f32 values to one Q8_0 block (absmax/127 scale).
///
/// Mirrors ggml `quantize_row_q8_0_ref`: `d = amax / 127`, quants are
/// `round(x / d)`; an all-zero block yields scale 0 and zero quants.
fn quantize_q8_0_block(vals: &[f32; 32]) -> BlockQ8_0 {
    let mut amax = 0.0f32;
    for &v in vals {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut quants = [0i8; 32];
    for (q, &v) in quants.iter_mut().zip(vals.iter()) {
        *q = (v * inv).round() as i8;
    }
    BlockQ8_0 {
        delta: f32_to_f16(d),
        quants,
    }
}

/// Requant a linear GGUF Q5_K weight matrix to Q8_0 blocks.
///
/// Q5_K has no HTP wire type or DSP kernels (upstream supports Q4_K/Q6_K
/// only), so Q5_K weights are dequantized and re-emitted as Q8_0 (which
/// carries more bits, so the round-trip is near-lossless) and then flow
/// through the standard Q8_0 repack. Output holds `ne1 * (ne0 / 32)`
/// Q8_0 blocks in GGUF row-major block order for the same `ne0 x ne1`.
pub fn requant_q5_k_to_q8_0(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<u8>, CeraError> {
    if !ne0.is_multiple_of(256) {
        return Err(CeraError::Backend(format!(
            "requant_q5_k_to_q8_0: K dim {ne0} is not a multiple of 256"
        )));
    }
    let sb_per_row = ne0 / 256;
    let block_size = std::mem::size_of::<BlockQ5K>();
    let total_src_bytes = ne1 * sb_per_row * block_size;
    if src_bytes.len() < total_src_bytes {
        return Err(CeraError::Backend(format!(
            "requant_q5_k_to_q8_0: source buffer too short (expected {} bytes, got {})",
            total_src_bytes,
            src_bytes.len()
        )));
    }
    let mut out = vec![0u8; ne1 * (ne0 / 32) * std::mem::size_of::<BlockQ8_0>()];
    for r in 0..ne1 {
        for sb in 0..sb_per_row {
            let blk_off = (r * sb_per_row + sb) * block_size;
            let b = &src_bytes[blk_off..blk_off + block_size];
            // Reinterpret through the packed struct (little-endian, packed).
            let block = BlockQ5K {
                d: u16::from_le_bytes([b[0], b[1]]),
                dmin: u16::from_le_bytes([b[2], b[3]]),
                scales: b[4..16].try_into().unwrap(),
                qh: b[16..48].try_into().unwrap(),
                qs: b[48..176].try_into().unwrap(),
            };
            let vals = dequantize_q5_k_block(&block);
            for (i, chunk) in vals.as_chunks::<32>().0.iter().enumerate() {
                let q = quantize_q8_0_block(chunk);
                let dst_off = (r * sb_per_row * 8 + sb * 8 + i) * 34;
                out[dst_off..dst_off + 2].copy_from_slice(&q.delta.to_le_bytes());
                for (j, &v) in q.quants.iter().enumerate() {
                    out[dst_off + 2 + j] = v as u8;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repack_q8_0_sizes() {
        assert_eq!(repacked_matrix_size_q8_0(1024, 32), 32 * 1088);
        assert_eq!(repacked_matrix_size_q8_0(1000, 20), 1 * 32 * 1088);

        let (ne0, ne1) = (64, 32);
        let size = repacked_matrix_size_q8_0(ne0, ne1);
        let src = vec![0u8; ne1 * (ne0 / 32) * 34];
        let mut dst = vec![0xccu8; size];
        assert!(repack_q8_0(&src, ne0, ne1, &mut dst).is_ok());
    }

    #[test]
    fn test_repack_q4_0_sizes() {
        assert_eq!(repacked_matrix_size_q4_0(256, 4), 1 * 8 * 576);

        let (ne0, ne1) = (64, 32);
        let size = repacked_matrix_size_q4_0(ne0, ne1);
        let src = vec![0u8; ne1 * (ne0 / 32) * 18];
        let mut dst = vec![0xccu8; size];
        assert!(repack_q4_0(&src, ne0, ne1, &mut dst).is_ok());
    }

    #[test]
    fn test_repack_q8_0_single_tile_layout() {
        // One 32x32 tile: block (row r, k-tile 0) holds quants 100+r and
        // scale bytes [r, r+1]. Verify the interleaved tile encoding.
        let (ne0, ne1) = (32, 32);
        let mut src = vec![0u8; 32 * 34];
        for r in 0..32 {
            src[r * 34] = r as u8;
            src[r * 34 + 1] = (r + 1) as u8;
            for q in 0..32 {
                src[r * 34 + 2 + q] = (100 + r) as u8;
            }
        }
        let mut dst = vec![0u8; TILE_SIZE_Q8_0];
        repack_q8_0(&src, ne0, ne1, &mut dst).unwrap();
        for cp in 0..16 {
            for row in 0..32 {
                assert_eq!(dst[cp * 64 + 2 * row], (100 + row) as u8);
                assert_eq!(dst[cp * 64 + 2 * row + 1], (100 + row) as u8);
            }
        }
        for row in 0..32 {
            assert_eq!(dst[1024 + row * 2], row as u8);
            assert_eq!(dst[1024 + row * 2 + 1], (row + 1) as u8);
        }
    }

    #[test]
    fn test_repack_q4_k_sizes() {
        assert_eq!(repacked_matrix_size_q4_k(256, 32), 1 * 8 * 640);
        assert_eq!(repacked_matrix_size_q4_k(1024, 4608), 144 * 32 * 640);

        let (ne0, ne1) = (256, 4);
        let size = repacked_matrix_size_q4_k(ne0, ne1);
        let src = vec![0u8; ne1 * (ne0 / 256) * 144];
        let mut dst = vec![0xccu8; size];
        assert!(repack_q4_k(&src, ne0, ne1, &mut dst).is_ok());
        // All-zero source repacks to zero quants; the M scale slot holds
        // f16(-0.0) = 0x8000 (M = -dmin * m, matching upstream).
        assert!(dst[..512].iter().all(|&b| b == 0));
        assert_eq!(u16::from_le_bytes([dst[514], dst[515]]), 0x8000);

        assert!(repack_q4_k(&src[..src.len() - 1], ne0, ne1, &mut dst).is_err());
        assert!(repack_q4_k(&src, ne0, ne1, &mut dst[..size - 1]).is_err());
        assert!(repack_q4_k(&src, 128, ne1, &mut dst).is_err());
    }

    #[test]
    fn test_repack_q6_k_sizes() {
        assert_eq!(repacked_matrix_size_q6_k(256, 32), 1 * 8 * 896);
        assert_eq!(repacked_matrix_size_q6_k(1024, 1024), 32 * 32 * 896);

        let (ne0, ne1) = (256, 4);
        let size = repacked_matrix_size_q6_k(ne0, ne1);
        let src = vec![0u8; ne1 * (ne0 / 256) * 210];
        let mut dst = vec![0xccu8; size];
        assert!(repack_q6_k(&src, ne0, ne1, &mut dst).is_ok());

        assert!(repack_q6_k(&src[..src.len() - 1], ne0, ne1, &mut dst).is_err());
        assert!(repack_q6_k(&src, ne0, ne1, &mut dst[..size - 1]).is_err());
        assert!(repack_q6_k(&src, 512 - 128, ne1, &mut dst).is_err());
    }

    #[test]
    fn test_repack_q4_k_single_tile_layout() {
        // ne0=256 (8 k-tiles), ne1=1: block d=1.0, dmin=0.5, scales[0]=33,
        // scales[4]=17, qs[2cp]=cp, qs[2cp+1]=15-cp. Tile kt=0 takes
        // sub-block 0: tile[cp*32] = ((15-cp) << 4) | cp, D=33.0, M=-8.5.
        let (ne0, ne1) = (256, 1);
        let mut src = vec![0u8; 144];
        src[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        src[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        src[4] = 33;
        src[8] = 17;
        for cp in 0..16 {
            src[16 + 2 * cp] = cp as u8;
            src[16 + 2 * cp + 1] = 15 - cp as u8;
        }
        let mut dst = vec![0u8; 8 * TILE_SIZE_Q4_K];
        repack_q4_k(&src, ne0, ne1, &mut dst).unwrap();
        for cp in 0..16 {
            assert_eq!(dst[cp * 32], ((15 - cp) << 4 | cp) as u8, "cp={cp}");
        }
        assert_eq!(u16::from_le_bytes([dst[512], dst[513]]), f32_to_f16(33.0));
        assert_eq!(u16::from_le_bytes([dst[514], dst[515]]), f32_to_f16(-8.5));
        // Padding rows 1..32 of tile 0 stay zero.
        assert!(dst[1..32].iter().all(|&b| b == 0));
        assert!(dst[512 + 4..512 + 128].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_repack_q6_k_single_tile_layout() {
        // ne0=256, ne1=1: d=2.0, scales[0]=3, scales[1]=-2, ql[0]=0xAB,
        // qh[0]=0xE4. Element 0 decodes to 0x0B: lo[0]=0xB, hi[0]=0;
        // scales are f16(6.0)=0x4600 / f16(-4.0)=0xC400.
        let (ne0, ne1) = (256, 1);
        let mut src = vec![0u8; 210];
        src[0] = 0xab;
        src[128] = 0xe4;
        src[192] = 3;
        src[193] = (-2i8) as u8;
        src[208..210].copy_from_slice(&0x4000u16.to_le_bytes());
        let mut dst = vec![0u8; 8 * TILE_SIZE_Q6_K];
        repack_q6_k(&src, ne0, ne1, &mut dst).unwrap();
        assert_eq!(dst[0], 0x0b);
        assert_eq!(dst[512], 0x00);
        assert_eq!(u16::from_le_bytes([dst[768], dst[769]]), 0x4600);
        assert_eq!(u16::from_le_bytes([dst[832], dst[833]]), 0xc400);
        // Padding rows stay zero (pos = row*4+(lk&3) only hits row 0).
        assert!(dst[4..128].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_requant_q5_k_to_q8_0() {
        // One ramp block: d=1.0, no min, qs[i]=i%32 pattern via qh/qs.
        // Round-trip error must stay within half a Q8_0 step.
        let (ne0, ne1) = (256, 1);
        let mut src = vec![0u8; 176];
        src[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for j in 0..8 {
            src[4 + j] = 32;
        }
        for i in 0..128 {
            src[48 + i] = (i % 16) as u8 | (((i + 1) % 16) as u8) << 4;
        }
        let out = requant_q5_k_to_q8_0(&src, ne0, ne1).unwrap();
        assert_eq!(out.len(), 8 * 34);
        // Each Q8_0 block dequantizes close to the Q5_K values.
        let block = BlockQ5K {
            d: u16::from_le_bytes([src[0], src[1]]),
            dmin: u16::from_le_bytes([src[2], src[3]]),
            scales: src[4..16].try_into().unwrap(),
            qh: src[16..48].try_into().unwrap(),
            qs: src[48..176].try_into().unwrap(),
        };
        let vals = dequantize_q5_k_block(&block);
        let mut max_err = 0.0f32;
        for (i, chunk) in vals.as_chunks::<32>().0.iter().enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([out[i * 34], out[i * 34 + 1]]));
            for (j, &v) in chunk.iter().enumerate() {
                let q = out[i * 34 + 2 + j] as i8 as f32;
                max_err = max_err.max((v - q * d).abs());
            }
        }
        let amax = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(max_err <= amax / 127.0, "max_err={max_err} amax={amax}");

        // All-zero block requants to zero blocks.
        let zero = vec![0u8; 176];
        let out0 = requant_q5_k_to_q8_0(&zero, ne0, ne1).unwrap();
        assert!(out0.iter().all(|&b| b == 0));

        assert!(requant_q5_k_to_q8_0(&src[..175], ne0, ne1).is_err());
        assert!(requant_q5_k_to_q8_0(&src, 128, ne1).is_err());
    }

    #[test]
    fn test_repack_q4_0_single_tile_layout() {
        // One 32x32 tile: block row r holds packed nibbles (r%16) and scale
        // bytes [r, 0]. Nibble j of row r unpacks to (low: qs[j], high:
        // qs[j-16]); tile byte = (q[2cp+1] << 4) | q[2cp].
        let (ne0, ne1) = (32, 32);
        let mut src = vec![0u8; 32 * 18];
        for r in 0..32 {
            src[r * 18] = r as u8;
            for j in 0..16 {
                src[r * 18 + 2 + j] = (((r + 1) & 0x0f) | (((r + 2) & 0x0f) << 4)) as u8;
            }
        }
        let mut dst = vec![0u8; TILE_SIZE_Q4_0];
        repack_q4_0(&src, ne0, ne1, &mut dst).unwrap();
        for cp in 0..16 {
            for row in 0..32 {
                let q = |j: usize| {
                    if j < 16 {
                        (row + 1) & 0x0f
                    } else {
                        (row + 2) & 0x0f
                    }
                };
                let expect = ((q(2 * cp + 1) << 4) | q(2 * cp)) as u8;
                assert_eq!(dst[cp * 32 + row], expect, "cp={cp} row={row}");
            }
        }
        for row in 0..32 {
            assert_eq!(dst[512 + row * 2], row as u8);
            assert_eq!(dst[512 + row * 2 + 1], 0);
        }
    }
}

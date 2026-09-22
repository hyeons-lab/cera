//! Host-side weight repacking for Hexagon Tensor Processor (HTP) kernels.
//!
//! Upstream `ggml-hexagon` requires quantized weight matrices (Q4_0, Q8_0) to be
//! repacked from linear GGUF row-major blocks into 32x32 tiles. In this tiled format,
//! HVX vector registers (128 bytes) and HMX matrix units can access weights with zero
//! stride penalties and optimal memory alignments.

use crate::quant::{BlockQ4_0, BlockQ8_0};
use crate::session::CeraError;

/// Tile dimensions processed by HMX kernels.
pub const HTP_TILE_COLS: usize = 32;
pub const HTP_TILE_ROWS: usize = 32;

/// Repacked tile byte sizes.
pub const HTP_TILE_SIZE_Q4_0: usize = 576; // 512 bytes quants + 64 bytes f16 scales
pub const HTP_TILE_SIZE_Q8_0: usize = 1088; // 1024 bytes quants + 64 bytes f16 scales

#[inline(always)]
pub const fn round_up_32(val: usize) -> usize {
    (val + 31) & !31
}

/// Calculate the total buffer size in bytes for a repacked Q4_0 matrix.
pub fn tiled_matrix_size_q4_0(ne0: usize, ne1: usize) -> usize {
    let n_k_tiles = round_up_32(ne0) / HTP_TILE_COLS;
    let n_col_tiles = round_up_32(ne1) / HTP_TILE_ROWS;
    n_col_tiles * n_k_tiles * HTP_TILE_SIZE_Q4_0
}

/// Calculate the total buffer size in bytes for a repacked Q8_0 matrix.
pub fn tiled_matrix_size_q8_0(ne0: usize, ne1: usize) -> usize {
    let n_k_tiles = round_up_32(ne0) / HTP_TILE_COLS;
    let n_col_tiles = round_up_32(ne1) / HTP_TILE_ROWS;
    n_col_tiles * n_k_tiles * HTP_TILE_SIZE_Q8_0
}

/// Repack a linear GGUF Q4_0 weight matrix into HTP 32x32 tiled layout.
///
/// `ne0` is the inner dimension (columns/k), which must be a multiple of 32.
/// `ne1` is the outer dimension (rows/n).
pub fn repack_q4_0_tiled(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    if ne0 % 32 != 0 {
        return Err(CeraError::Backend(format!(
            "repack_q4_0: ne0 ({}) must be a multiple of 32",
            ne0
        )));
    }

    let blocks_per_row = ne0 / 32;
    let expected_src_len = ne1 * blocks_per_row * std::mem::size_of::<BlockQ4_0>();
    if src_bytes.len() < expected_src_len {
        return Err(CeraError::Backend(format!(
            "repack_q4_0: source buffer too short (expected {} bytes, got {})",
            expected_src_len,
            src_bytes.len()
        )));
    }

    let expected_dst_len = tiled_matrix_size_q4_0(ne0, ne1);
    if dst.len() < expected_dst_len {
        return Err(CeraError::Backend(format!(
            "repack_q4_0: destination buffer too short (expected {} bytes, got {})",
            expected_dst_len,
            dst.len()
        )));
    }

    let n_k_tiles = round_up_32(ne0) / HTP_TILE_COLS;
    let n_col_tiles = round_up_32(ne1) / HTP_TILE_ROWS;

    // Zero out destination to handle padded boundaries cleanly
    dst[..expected_dst_len].fill(0);

    let src_blocks = unsafe {
        std::slice::from_raw_parts(src_bytes.as_ptr() as *const BlockQ4_0, ne1 * blocks_per_row)
    };

    for ct in 0..n_col_tiles {
        for kt in 0..n_k_tiles {
            let tile_offset = (ct * n_k_tiles + kt) * HTP_TILE_SIZE_Q4_0;
            let tile_dst = &mut dst[tile_offset..tile_offset + HTP_TILE_SIZE_Q4_0];

            let mut tile_quants = [[0u8; 32]; 32];

            // Unpack 4-bit quants for up to 32 rows within this tile
            for row in 0..32 {
                let r = ct * 32 + row;
                if r < ne1 && kt < blocks_per_row {
                    let block = &src_blocks[r * blocks_per_row + kt];
                    // Unpack block: low nibbles for cols 0..15, high nibbles for cols 16..31
                    for i in 0..16 {
                        tile_quants[row][i] = block.qs[i] & 0x0F;
                        tile_quants[row][i + 16] = block.qs[i] >> 4;
                    }
                }
            }

            // Pack column pairs into tiled layout: 16 column pairs across 32 rows = 512 bytes
            for cp in 0..16 {
                let col0 = 2 * cp;
                let col1 = col0 + 1;
                for row in 0..32 {
                    tile_dst[cp * 32 + row] =
                        (tile_quants[row][col1] << 4) | tile_quants[row][col0];
                }
            }

            // Pack f16 scales into bytes 512..576
            for row in 0..32 {
                let r = ct * 32 + row;
                let scale = if r < ne1 && kt < blocks_per_row {
                    src_blocks[r * blocks_per_row + kt].d
                } else {
                    0
                };
                let scale_bytes = scale.to_le_bytes();
                tile_dst[512 + 2 * row] = scale_bytes[0];
                tile_dst[512 + 2 * row + 1] = scale_bytes[1];
            }
        }
    }

    Ok(())
}

/// Repack a linear GGUF Q8_0 weight matrix into HTP 32x32 tiled layout.
///
/// `ne0` is the inner dimension (columns/k), which must be a multiple of 32.
/// `ne1` is the outer dimension (rows/n).
pub fn repack_q8_0_tiled(
    src_bytes: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    if ne0 % 32 != 0 {
        return Err(CeraError::Backend(format!(
            "repack_q8_0: ne0 ({}) must be a multiple of 32",
            ne0
        )));
    }

    let blocks_per_row = ne0 / 32;
    let expected_src_len = ne1 * blocks_per_row * std::mem::size_of::<BlockQ8_0>();
    if src_bytes.len() < expected_src_len {
        return Err(CeraError::Backend(format!(
            "repack_q8_0: source buffer too short (expected {} bytes, got {})",
            expected_src_len,
            src_bytes.len()
        )));
    }

    let expected_dst_len = tiled_matrix_size_q8_0(ne0, ne1);
    if dst.len() < expected_dst_len {
        return Err(CeraError::Backend(format!(
            "repack_q8_0: destination buffer too short (expected {} bytes, got {})",
            expected_dst_len,
            dst.len()
        )));
    }

    let n_k_tiles = round_up_32(ne0) / HTP_TILE_COLS;
    let n_col_tiles = round_up_32(ne1) / HTP_TILE_ROWS;

    // Zero out destination to handle padded boundaries cleanly
    dst[..expected_dst_len].fill(0);

    let src_blocks = unsafe {
        std::slice::from_raw_parts(src_bytes.as_ptr() as *const BlockQ8_0, ne1 * blocks_per_row)
    };

    for ct in 0..n_col_tiles {
        for kt in 0..n_k_tiles {
            let tile_offset = (ct * n_k_tiles + kt) * HTP_TILE_SIZE_Q8_0;
            let tile_dst = &mut dst[tile_offset..tile_offset + HTP_TILE_SIZE_Q8_0];

            // 16 column pairs, 32 rows, 2 bytes per row = 1024 bytes
            for cp in 0..16 {
                let col0 = 2 * cp;
                let col1 = col0 + 1;
                for row in 0..32 {
                    let r = ct * 32 + row;
                    let (q0, q1) = if r < ne1 && kt < blocks_per_row {
                        let block = &src_blocks[r * blocks_per_row + kt];
                        (block.quants[col0] as u8, block.quants[col1] as u8)
                    } else {
                        (0, 0)
                    };

                    tile_dst[cp * 64 + 2 * row + 0] = q0;
                    tile_dst[cp * 64 + 2 * row + 1] = q1;
                }
            }

            // Pack f16 scales into bytes 1024..1088
            for row in 0..32 {
                let r = ct * 32 + row;
                let scale = if r < ne1 && kt < blocks_per_row {
                    src_blocks[r * blocks_per_row + kt].delta
                } else {
                    0
                };
                let scale_bytes = scale.to_le_bytes();
                tile_dst[1024 + 2 * row] = scale_bytes[0];
                tile_dst[1024 + 2 * row + 1] = scale_bytes[1];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_matrix_size_calculation() {
        assert_eq!(tiled_matrix_size_q4_0(32, 32), 576);
        assert_eq!(tiled_matrix_size_q4_0(64, 32), 2 * 576);
        assert_eq!(tiled_matrix_size_q4_0(64, 64), 4 * 576);

        assert_eq!(tiled_matrix_size_q8_0(32, 32), 1088);
        assert_eq!(tiled_matrix_size_q8_0(64, 32), 2 * 1088);
        assert_eq!(tiled_matrix_size_q8_0(64, 64), 4 * 1088);
    }

    #[test]
    fn test_repack_q4_0_single_tile() {
        // Construct a single 32x32 block of Q4_0
        let mut blocks = Vec::new();
        for r in 0..32 {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                // Low nibble = (i & 0xF), high nibble = ((i + 1) & 0xF)
                qs[i] = ((i as u8 + 1) << 4) | (i as u8);
            }
            blocks.push(BlockQ4_0 {
                d: (r as u16 + 100),
                qs,
            });
        }

        let src_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ4_0>(),
            )
        };

        let mut dst = vec![0u8; HTP_TILE_SIZE_Q4_0];
        repack_q4_0_tiled(src_bytes, 32, 32, &mut dst).expect("repack should succeed");

        // Verify column pair 0 for row 5
        let cp0_row5 = dst[0 * 32 + 5];
        assert_eq!(cp0_row5 & 0x0F, 0); // col0 is 0
        assert_eq!(cp0_row5 >> 4, 1); // col1 is 1

        // Verify scale for row 5
        let scale_bytes = [dst[512 + 2 * 5], dst[512 + 2 * 5 + 1]];
        assert_eq!(u16::from_le_bytes(scale_bytes), 105);
    }

    #[test]
    fn test_repack_q8_0_single_tile() {
        let mut blocks = Vec::new();
        for r in 0..32 {
            let mut quants = [0i8; 32];
            for i in 0..32 {
                quants[i] = (i as i8) - 16;
            }
            blocks.push(BlockQ8_0 {
                delta: (r as u16 + 200),
                quants,
            });
        }

        let src_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ8_0>(),
            )
        };

        let mut dst = vec![0u8; HTP_TILE_SIZE_Q8_0];
        repack_q8_0_tiled(src_bytes, 32, 32, &mut dst).expect("repack should succeed");

        // Verify column pair 2 (col 4 and col 5) for row 7
        let q0 = dst[2 * 64 + 2 * 7 + 0] as i8;
        let q1 = dst[2 * 64 + 2 * 7 + 1] as i8;
        assert_eq!(q0, 4 - 16);
        assert_eq!(q1, 5 - 16);

        // Verify scale for row 7
        let scale_bytes = [dst[1024 + 2 * 7], dst[1024 + 2 * 7 + 1]];
        assert_eq!(u16::from_le_bytes(scale_bytes), 207);
    }
}

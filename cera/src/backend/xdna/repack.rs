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

/// Convert f32 to bf16 with round-to-nearest-even (the spike's rule).
pub fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    let rounded = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (rounded >> 16) as u16
}

/// Packed byte count for one stage of `rows` x `cols` weights.
pub fn stage_bytes(rows: usize, cols: usize) -> Result<usize, String> {
    check_shape(rows, cols)?;
    let chunks = cols.div_ceil(CHUNK_K);
    Ok(rows / ROWS * chunks * (CHUNK_K.min(cols) / QBLK) * BLOCK_BYTES)
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

/// A source of signed int4 weights (`q - 8` in [-8, 7]) and f32 row scales,
/// addressed the way the packing loops consume them.
trait WeightSource {
    /// Signed int4 weight at (row, col), in [-8, 7].
    fn wq(&self, row: usize, col: usize) -> i8;
    /// f32 scale of (row, block).
    fn scale(&self, row: usize, block: usize) -> f32;
}

/// GGUF Q4_0 bytes: 18-byte blocks of f16 scale + 32 nibbles.
struct Q4_0Source<'a> {
    /// Raw tensor bytes.
    src: &'a [u8],
    /// Blocks per row.
    nb: usize,
}

impl WeightSource for Q4_0Source<'_> {
    fn wq(&self, row: usize, col: usize) -> i8 {
        let block = &self.src[(row * self.nb + col / QBLK) * 18..];
        let qs = &block[2..18];
        let j = col % QBLK;
        let byte = qs[j % 16];
        let nib = if j < 16 { byte & 0xF } else { byte >> 4 };
        nib as i8 - 8
    }

    fn scale(&self, row: usize, block: usize) -> f32 {
        let bytes = &self.src[(row * self.nb + block) * 18..];
        f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
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
    fn wq(&self, row: usize, col: usize) -> i8 {
        let (d, _) = self.block_scale(row, col / QBLK);
        if d == 0.0 {
            return 0;
        }
        let v = self.src[row * self.cols + col] / d;
        // The spike's `int(v * id + 8.5f)`: truncation toward zero, clamped.
        ((v + 8.5) as i32).clamp(0, 15) as i8 - 8
    }

    fn scale(&self, row: usize, block: usize) -> f32 {
        self.block_scale(row, block).1
    }
}

impl F32Source<'_> {
    /// (scale as used to quantize signed values, f32 scale) of one block.
    /// The kernel's `wq` is `q - 8` with values `(q-8) * d`; the spike
    /// quantizes with `d = mx / -8`, so the stored scale is that `d`.
    fn block_scale(&self, row: usize, block: usize) -> (f32, f32) {
        let base = row * self.cols + block * QBLK;
        let mut mx = 0.0f32;
        let mut amax = 0.0f32;
        for j in 0..QBLK {
            let v = self.src[base + j];
            if v.abs() > amax {
                amax = v.abs();
                mx = v;
            }
        }
        let d = mx / -8.0;
        (d, d)
    }
}

/// Pack one stage from `source` into `dst` (zeroed here first; the packing
/// ORs nibbles in). Returns the byte count written.
fn pack_stage<S: WeightSource>(
    dst: &mut [u8],
    source: &S,
    rows: usize,
    cols: usize,
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
    let chunk_k = CHUNK_K.min(cols);
    let chunks = cols / chunk_k;
    let nb_chunk = chunk_k / QBLK;
    let tile_bytes = nb_chunk * BLOCK_BYTES;
    for tile in 0..rows / ROWS {
        let r0 = tile * ROWS;
        for ch in 0..chunks {
            let tp = (tile * chunks + ch) * tile_bytes;
            for lb in 0..nb_chunk {
                let b = ch * nb_chunk + lb;
                let bp = tp + lb * 256;
                for h in 0..2 {
                    for kk in 0..16 {
                        for n in 0..16 {
                            let e = kk * 16 + n;
                            let nib = (source.wq(r0 + n, b * QBLK + h * 16 + kk) & 0xF) as u8;
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
                for n in 0..16 {
                    let bits = f32_to_bf16(source.scale(r0 + n, b));
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
    check_shape(rows, cols)?;
    let nb = cols / QBLK;
    let expect = rows * nb * 18;
    if src.len() < expect {
        return Err(format!(
            "Q4_0 tensor too small: need {expect} bytes, have {}",
            src.len()
        ));
    }
    pack_stage(dst, &Q4_0Source { src, nb }, rows, cols)
}

/// Pack f32 weights (e.g. a dequantized Q6_K head) into the stream layout,
/// quantizing with the spike's Q4_0 rule.
pub fn pack_f32_stage(
    dst: &mut [u8],
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<usize, String> {
    check_shape(rows, cols)?;
    if src.len() < rows * cols {
        return Err(format!(
            "f32 tensor too small: need {} elements, have {}",
            rows * cols,
            src.len()
        ));
    }
    pack_stage(dst, &F32Source { src, cols }, rows, cols)
}

/// Number of AIE2P workers across which every stage's row tiles are partitioned.
const WORKERS: usize = 16;

/// Interleave already-packed stage buffers per worker so a fused TaskGroup
/// (`q+k+v` or `gate+up`) streams all of worker `i`'s stage slices in a single
/// contiguous Shim DMA buffer descriptor.
pub fn interleave_worker_stages(dst: &mut [u8], stages: &[&[u8]]) -> Result<usize, String> {
    if stages.is_empty() {
        return Err("xdna interleave requires at least one stage".to_string());
    }
    let mut total = 0usize;
    for (idx, s) in stages.iter().enumerate() {
        if s.is_empty() || !s.len().is_multiple_of(WORKERS) {
            return Err(format!(
                "packed stage {idx} byte length {} is not a nonzero multiple of {WORKERS}",
                s.len()
            ));
        }
        total += s.len();
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
    }

    #[test]
    fn interleave_worker_stages_groups_worker_slices_contiguously() {
        let s0: Vec<u8> = (0u8..32).collect(); // 2 bytes/worker
        let s1: Vec<u8> = (100u8..116).collect(); // 1 byte/worker
        let mut dst = vec![0u8; 48];
        let written = interleave_worker_stages(&mut dst, &[&s0, &s1]).unwrap();
        assert_eq!(written, 48);
        for w in 0..WORKERS {
            let base = w * 3;
            assert_eq!(dst[base], (w * 2) as u8);
            assert_eq!(dst[base + 1], (w * 2 + 1) as u8);
            assert_eq!(dst[base + 2], (100 + w) as u8);
        }
        assert!(interleave_worker_stages(&mut dst, &[]).is_err());
        assert!(interleave_worker_stages(&mut dst, &[&[]]).is_err());
        assert!(interleave_worker_stages(&mut dst, &[&[0u8; 15]]).is_err());
    }
}

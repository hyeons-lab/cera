// SIMD-optimized kernels for quantized operations.
//
// Platform-specific implementations behind cfg gates.
// The dispatch functions select the best available implementation at compile time.

#[cfg(target_arch = "aarch64")]
use crate::quant::BlockQ6K;
use crate::quant::{BlockQ4_0, BlockQ4KM, BlockQ8_0};
// `half::f16` is consumed by the NEON / AVX2 kernels below and by the
// `#[cfg(test)] mod tests` further down (the tests aren't arch-gated and
// use `f16::from_f32` to seed quantized blocks). Including `test` in the
// gate keeps `cargo test` compilable on architectures that don't have a
// SIMD kernel here (e.g. armv7, riscv64) — without it those archs build
// the tests but lose the import. On non-test wasm32 builds the import
// remains correctly elided.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", test))]
use half::f16;

// ── aarch64 NEON ────────────────────────────────────────────────────────────

/// Send+Sync pointer wrapper for parallel GEMV closures.
/// Stores pointers as usize to satisfy Send+Sync (raw pointers don't implement them).
/// Safety: callers ensure non-overlapping row access and immutable source data.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct GemvPtrs {
    a: usize,
    xq: usize,
    xs: usize,
}
#[cfg(target_arch = "aarch64")]
impl GemvPtrs {
    fn a(&self) -> *const u8 {
        self.a as *const u8
    }
    fn xq(&self) -> *const i8 {
        self.xq as *const i8
    }
    fn xs(&self) -> *const f32 {
        self.xs as *const f32
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop, unused_unsafe)]
pub(crate) mod neon {
    use super::*;
    use crate::backend::cpu_features::{CpuTier, cpu_features};
    use std::arch::aarch64::*;
    use std::mem::size_of;

    // ── Shared GEMM dot-product macros ─────────────────────────────────────
    // Used by both Q4_0 and Q8_0 GEMM kernels to avoid duplicating the
    // Q8_0 input loading + vdotq_s32 + scale accumulation pattern.

    /// Accumulate dot products for a pair of decoded weight blocks against one Q8_0 column.
    /// `$w0_lo/$w0_hi`: decoded weight int8x16 for block bi (lo/hi halves)
    /// `$w1_lo/$w1_hi`: decoded weight int8x16 for block bi+1
    /// `$d0_w/$d1_w`: weight-side f32 scales for blocks bi and bi+1
    macro_rules! gemm_dot_pair {
        ($w0_lo:expr, $w0_hi:expr, $w1_lo:expr, $w1_hi:expr,
         $d0_w:expr, $d1_w:expr,
         $xq:expr, $xs:expr, $bi:expr,
         $sumv0:expr, $sumv1:expr) => {{
            let y0_lo = vld1q_s8($xq.add($bi * 32));
            let y0_hi = vld1q_s8($xq.add($bi * 32 + 16));
            let y1_lo = vld1q_s8($xq.add(($bi + 1) * 32));
            let y1_hi = vld1q_s8($xq.add(($bi + 1) * 32 + 16));
            let z = vdupq_n_s32(0);
            let p_0 = vdotq_s32(vdotq_s32(z, $w0_lo, y0_lo), $w0_hi, y0_hi);
            let p_1 = vdotq_s32(vdotq_s32(z, $w1_lo, y1_lo), $w1_hi, y1_hi);
            $sumv0 = vmlaq_n_f32($sumv0, vcvtq_f32_s32(p_0), $d0_w * *$xs.add($bi));
            $sumv1 = vmlaq_n_f32($sumv1, vcvtq_f32_s32(p_1), $d1_w * *$xs.add($bi + 1));
        }};
    }

    /// Accumulate dot product for a single decoded weight block against one Q8_0 column.
    macro_rules! gemm_dot_single {
        ($w_lo:expr, $w_hi:expr, $d_w:expr,
         $xq:expr, $xs:expr, $bi:expr, $sumv:expr) => {{
            let y_lo = vld1q_s8($xq.add($bi * 32));
            let y_hi = vld1q_s8($xq.add($bi * 32 + 16));
            let z = vdupq_n_s32(0);
            let p = vdotq_s32(vdotq_s32(z, $w_lo, y_lo), $w_hi, y_hi);
            $sumv = vmlaq_n_f32($sumv, vcvtq_f32_s32(p), $d_w * *$xs.add($bi));
        }};
    }

    /// NEON-optimized Q8_0 dot product with f32 vector.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q8_0_f32_neon(block: &BlockQ8_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.delta).to_f32();

            let mut sumv = vdupq_n_f32(0.0);
            let quants_ptr = block.quants.as_ptr();
            let y_ptr = y.as_ptr();

            for i in (0..32).step_by(8) {
                // Load 8 i8 values, sign-extend to i16, then split to i32, convert to f32
                let q_bytes = vld1_s8(quants_ptr.add(i));
                let q_i16 = vmovl_s8(q_bytes);

                let q_lo_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_i16)));
                let q_hi_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_i16)));

                let y_lo = vld1q_f32(y_ptr.add(i));
                let y_hi = vld1q_f32(y_ptr.add(i + 4));

                sumv = vfmaq_f32(sumv, q_lo_f32, y_lo);
                sumv = vfmaq_f32(sumv, q_hi_f32, y_hi);
            }

            d * vaddvq_f32(sumv)
        }
    }

    /// NEON-optimized Q4_0 dot product with f32 vector.
    ///
    /// Q4_0 block: 16 bytes `qs` holding 32 4-bit unsigned values (low nibble first,
    /// then high nibble). Values are offset by -8: value = (nibble - 8) * d.
    ///
    /// Uses vector nibble extraction (vand/vshr on uint8x8) then widens to f32
    /// without scalar code in the inner loop.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_f32_neon(block: &BlockQ4_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.d).to_f32();
            let offset = vdupq_n_f32(8.0);
            let mask_lo = vdup_n_u8(0x0F);

            let mut sumv = vdupq_n_f32(0.0);
            let qs_ptr = block.qs.as_ptr();
            let y_ptr = y.as_ptr();

            // Process 8 bytes at a time → 8 low nibbles + 8 high nibbles = 16 values.
            // Two iterations cover all 16 bytes (32 values).
            for i in (0..16).step_by(8) {
                // Load 8 bytes of quantized data
                let qbytes = vld1_u8(qs_ptr.add(i));

                // Extract low and high nibbles as u8 vectors
                let lo_u8 = vand_u8(qbytes, mask_lo);
                let hi_u8 = vshr_n_u8::<4>(qbytes);

                // Widen low nibbles: u8x8 → u16x8 → split → u32x4 → f32x4
                let lo_u16 = vmovl_u8(lo_u8);
                let lo_f32_0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_u16))), offset);
                let lo_f32_1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_u16))), offset);

                // Widen high nibbles similarly
                let hi_u16 = vmovl_u8(hi_u8);
                let hi_f32_0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_u16))), offset);
                let hi_f32_1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_u16))), offset);

                // FMA with corresponding y values
                // Low nibbles: y[i..i+4], y[i+4..i+8]
                sumv = vfmaq_f32(sumv, lo_f32_0, vld1q_f32(y_ptr.add(i)));
                sumv = vfmaq_f32(sumv, lo_f32_1, vld1q_f32(y_ptr.add(i + 4)));
                // High nibbles: y[i+16..i+20], y[i+20..i+24]
                sumv = vfmaq_f32(sumv, hi_f32_0, vld1q_f32(y_ptr.add(i + 16)));
                sumv = vfmaq_f32(sumv, hi_f32_1, vld1q_f32(y_ptr.add(i + 16 + 4)));
            }

            d * vaddvq_f32(sumv)
        }
    }

    /// NEON-optimized Q4_K_M dot product with f32 vector.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_k_m_f32_neon(block: &BlockQ4KM, y: &[f32]) -> f32 {
        unsafe {
            let d = f16::from_bits(block.d).to_f32();
            let dmin = f16::from_bits(block.dmin).to_f32();

            let scales = &block.scales;
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..4 {
                sc[j] = scales[j] & 63;
                mn[j] = scales[j + 4] & 63;
            }
            for j in 4..8 {
                sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }

            let qs = &block.qs;
            let y_ptr = y.as_ptr();
            let mut sumf = 0.0f32;
            let mut qi = 0usize;
            let mut yi = 0usize;

            for j in 0..4 {
                let sc1 = d * sc[j * 2] as f32;
                let mn1 = dmin * mn[j * 2] as f32;
                let sc2 = d * sc[j * 2 + 1] as f32;
                let mn2 = dmin * mn[j * 2 + 1] as f32;

                let mut sum1v = vdupq_n_f32(0.0);
                let mut sum2v = vdupq_n_f32(0.0);
                let mut sum_mn1v = vdupq_n_f32(0.0);
                let mut sum_mn2v = vdupq_n_f32(0.0);

                for l in (0..32).step_by(4) {
                    let q0 = qs[qi + l] as u32;
                    let q1 = qs[qi + l + 1] as u32;
                    let q2 = qs[qi + l + 2] as u32;
                    let q3 = qs[qi + l + 3] as u32;

                    let lo = [
                        (q0 & 0xF) as f32,
                        (q1 & 0xF) as f32,
                        (q2 & 0xF) as f32,
                        (q3 & 0xF) as f32,
                    ];
                    let lo_v = vld1q_f32(lo.as_ptr());

                    let hi = [
                        (q0 >> 4) as f32,
                        (q1 >> 4) as f32,
                        (q2 >> 4) as f32,
                        (q3 >> 4) as f32,
                    ];
                    let hi_v = vld1q_f32(hi.as_ptr());

                    let y1 = vld1q_f32(y_ptr.add(yi + l));
                    let y2 = vld1q_f32(y_ptr.add(yi + l + 32));

                    sum1v = vfmaq_f32(sum1v, lo_v, y1);
                    sum2v = vfmaq_f32(sum2v, hi_v, y2);
                    sum_mn1v = vaddq_f32(sum_mn1v, y1);
                    sum_mn2v = vaddq_f32(sum_mn2v, y2);
                }

                let sum1 = vaddvq_f32(sum1v);
                let sum2 = vaddvq_f32(sum2v);
                let sum_mn1 = vaddvq_f32(sum_mn1v);
                let sum_mn2 = vaddvq_f32(sum_mn2v);

                sumf += sc1 * sum1 + sc2 * sum2 - mn1 * sum_mn1 - mn2 * sum_mn2;
                qi += 32;
                yi += 64;
            }

            sumf
        }
    }

    /// Quantize f32 vector to Q8_0 format (NEON-vectorized, f16 scale roundtrip).
    /// Stores scales and quants into caller-provided buffers.
    /// Returns the number of blocks written.
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_f32_to_q8_0_neon(
        x: &[f32],
        scales: &mut [f32],
        quants: &mut [i8],
    ) -> usize {
        unsafe {
            let k = x.len();
            debug_assert_eq!(
                k % 32,
                0,
                "quantize_f32_to_q8_0: x.len() must be divisible by 32"
            );
            debug_assert!(scales.len() >= k / 32);
            debug_assert!(quants.len() >= k);
            let n_blocks = k / 32;

            for bi in 0..n_blocks {
                let base = bi * 32;
                let x_ptr = x.as_ptr().add(base);

                let s0 = vld1q_f32(x_ptr);
                let s1 = vld1q_f32(x_ptr.add(4));
                let s2 = vld1q_f32(x_ptr.add(8));
                let s3 = vld1q_f32(x_ptr.add(12));
                let s4 = vld1q_f32(x_ptr.add(16));
                let s5 = vld1q_f32(x_ptr.add(20));
                let s6 = vld1q_f32(x_ptr.add(24));
                let s7 = vld1q_f32(x_ptr.add(28));

                let a0 = vmaxq_f32(vabsq_f32(s0), vabsq_f32(s1));
                let a1 = vmaxq_f32(vabsq_f32(s2), vabsq_f32(s3));
                let a2 = vmaxq_f32(vabsq_f32(s4), vabsq_f32(s5));
                let a3 = vmaxq_f32(vabsq_f32(s6), vabsq_f32(s7));
                let a4 = vmaxq_f32(a0, a1);
                let a5 = vmaxq_f32(a2, a3);
                let a6 = vmaxq_f32(a4, a5);
                let amax = vmaxvq_f32(a6);

                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let d_stored = f16::from_f32(d).to_f32();
                scales[bi] = d_stored;

                // Quantize 32 f32 → 32 i8 using NEON vector narrowing.
                // f32→i32 (vcvtnq), then i32→i16→i8 via vqmovn (saturating narrow).
                // Process 8 values at a time → 4 iterations for 32 values.
                let qp = quants.as_mut_ptr().add(base);
                let vi0 = vcvtnq_s32_f32(vmulq_n_f32(s0, id));
                let vi1 = vcvtnq_s32_f32(vmulq_n_f32(s1, id));
                let vi2 = vcvtnq_s32_f32(vmulq_n_f32(s2, id));
                let vi3 = vcvtnq_s32_f32(vmulq_n_f32(s3, id));
                let vi4 = vcvtnq_s32_f32(vmulq_n_f32(s4, id));
                let vi5 = vcvtnq_s32_f32(vmulq_n_f32(s5, id));
                let vi6 = vcvtnq_s32_f32(vmulq_n_f32(s6, id));
                let vi7 = vcvtnq_s32_f32(vmulq_n_f32(s7, id));

                // Extract i32 lanes to i8, matching ggml's vgetq_lane_s32 approach.
                // This avoids the double-saturating-narrow path (vqmovn_s32 + vqmovn_s16)
                // which may produce different results at boundary values.
                for (j, vi) in [vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7].iter().enumerate() {
                    *qp.add(4 * j) = vgetq_lane_s32::<0>(*vi) as i8;
                    *qp.add(4 * j + 1) = vgetq_lane_s32::<1>(*vi) as i8;
                    *qp.add(4 * j + 2) = vgetq_lane_s32::<2>(*vi) as i8;
                    *qp.add(4 * j + 3) = vgetq_lane_s32::<3>(*vi) as i8;
                }
            }
            n_blocks
        }
    }

    /// NEON integer GEMV using pre-quantized Q8_0 input.
    /// Call `quantize_f32_to_q8_0_neon` first, then call this for each weight matrix.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemv_q4_0_q8_0_neon_dotprod(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        unsafe {
            let blocks_per_row = k / 32;
            let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();

            let ptrs = GemvPtrs {
                a: a_quant.as_ptr() as usize,
                xq: x_quants.as_ptr() as usize,
                xs: x_scales.as_ptr() as usize,
            };

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let mask_lo = vdupq_n_u8(0x0F);
                let offset_8 = vdupq_n_s8(0x8);
                let row_start = i * row_bytes;
                let mut sumv0 = vdupq_n_f32(0.0);
                let mut sumv1 = vdupq_n_f32(0.0);

                let mut bi = 0usize;
                while bi + 1 < blocks_per_row {
                    // Prefetch next weight block pair
                    if bi + 3 < blocks_per_row {
                        _prefetch(
                            ptrs.a().add(row_start + (bi + 2) * size_of::<BlockQ4_0>())
                                as *const i8,
                            _PREFETCH_READ,
                            _PREFETCH_LOCALITY2,
                        );
                    }
                    let b0 = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);
                    let b1 = &*(ptrs.a().add(row_start + (bi + 1) * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);

                    let v0 = vld1q_u8(b0.qs.as_ptr());
                    let v1 = vld1q_u8(b1.qs.as_ptr());
                    let v0_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, mask_lo)), offset_8);
                    let v0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), offset_8);
                    let v1_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, mask_lo)), offset_8);
                    let v1_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), offset_8);

                    let y0_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let y0_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let y1_lo = vld1q_s8(ptrs.xq().add((bi + 1) * 32));
                    let y1_hi = vld1q_s8(ptrs.xq().add((bi + 1) * 32 + 16));

                    let z = vdupq_n_s32(0);
                    let p_0 = vdotq_s32(vdotq_s32(z, v0_lo, y0_lo), v0_hi, y0_hi);
                    let p_1 = vdotq_s32(vdotq_s32(z, v1_lo, y1_lo), v1_hi, y1_hi);

                    let d0 = f16::from_bits(b0.d).to_f32() * *ptrs.xs().add(bi);
                    let d1 = f16::from_bits(b1.d).to_f32() * *ptrs.xs().add(bi + 1);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
                    bi += 2;
                }

                if bi < blocks_per_row {
                    let b = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);
                    let v = vld1q_u8(b.qs.as_ptr());
                    let v_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask_lo)), offset_8);
                    let v_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), offset_8);
                    let y_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let y_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let z = vdupq_n_s32(0);
                    let p = vdotq_s32(vdotq_s32(z, v_lo, y_lo), v_hi, y_hi);
                    let d = f16::from_bits(b.d).to_f32() * *ptrs.xs().add(bi);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d);
                }

                *yi = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
            };

            if y.len() >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows(y, crate::backend::cpu::gemv_min_rows(), compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON Q4_0 GEMV: y[m] = A_q4_0[m,k] @ x_f32[k]. Quantizes x to Q8_0 into
    /// the caller-provided scratch (avoiding a per-call heap alloc) then defers
    /// to the pre-quantized dispatcher, which picks the dotprod or `_base` path.
    pub unsafe fn gemv_q4_0_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        unsafe {
            let n_blocks = k / 32;
            q8_scales.resize(n_blocks, 0.0);
            q8_quants.resize(k, 0);
            quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
            gemv_q4_0_q8_0_neon(a_quant, q8_scales, q8_quants, y, _m, k);
        }
    }

    /// NEON Q8_0 × Q8_0 GEMV with pre-quantized input (no quantization step).
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemv_q8_0_q8_0_neon_dotprod(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        unsafe {
            let n_blocks = k / 32;
            let row_bytes = n_blocks * size_of::<BlockQ8_0>();

            let ptrs = GemvPtrs {
                a: a_quant.as_ptr() as usize,
                xq: x_quants.as_ptr() as usize,
                xs: x_scales.as_ptr() as usize,
            };

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let row_start = i * row_bytes;
                let mut sumv0 = vdupq_n_f32(0.0);
                let mut sumv1 = vdupq_n_f32(0.0);

                let mut bi = 0usize;
                while bi + 1 < n_blocks {
                    if bi + 3 < n_blocks {
                        _prefetch(
                            ptrs.a().add(row_start + (bi + 2) * size_of::<BlockQ8_0>())
                                as *const i8,
                            _PREFETCH_READ,
                            _PREFETCH_LOCALITY2,
                        );
                    }
                    let wb0 = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);
                    let wb1 = &*(ptrs.a().add(row_start + (bi + 1) * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);

                    let w0_lo = vld1q_s8(wb0.quants.as_ptr());
                    let w0_hi = vld1q_s8(wb0.quants.as_ptr().add(16));
                    let w1_lo = vld1q_s8(wb1.quants.as_ptr());
                    let w1_hi = vld1q_s8(wb1.quants.as_ptr().add(16));

                    // Load input quants
                    let x0_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let x0_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let x1_lo = vld1q_s8(ptrs.xq().add((bi + 1) * 32));
                    let x1_hi = vld1q_s8(ptrs.xq().add((bi + 1) * 32 + 16));

                    // Integer dot product: 2 × vdotq_s32 per block
                    let z = vdupq_n_s32(0);
                    let p_0 = vdotq_s32(vdotq_s32(z, w0_lo, x0_lo), w0_hi, x0_hi);
                    let p_1 = vdotq_s32(vdotq_s32(z, w1_lo, x1_lo), w1_hi, x1_hi);

                    // Scale: d_weight × d_input
                    let d0 = f16::from_bits(wb0.delta).to_f32() * *ptrs.xs().add(bi);
                    let d1 = f16::from_bits(wb1.delta).to_f32() * *ptrs.xs().add(bi + 1);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
                    bi += 2;
                }

                if bi < n_blocks {
                    let wb = &*(ptrs.a().add(row_start + bi * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);
                    let w_lo = vld1q_s8(wb.quants.as_ptr());
                    let w_hi = vld1q_s8(wb.quants.as_ptr().add(16));
                    let x_lo = vld1q_s8(ptrs.xq().add(bi * 32));
                    let x_hi = vld1q_s8(ptrs.xq().add(bi * 32 + 16));
                    let z = vdupq_n_s32(0);
                    let p = vdotq_s32(vdotq_s32(z, w_lo, x_lo), w_hi, x_hi);
                    let d = f16::from_bits(wb.delta).to_f32() * *ptrs.xs().add(bi);
                    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), d);
                }

                *yi = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
            };

            if y.len() >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows(y, crate::backend::cpu::gemv_min_rows(), compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON Q8_0 GEMV: y[m] = A_q8_0[m,k] @ x_f32[k]. Quantizes x to Q8_0 into
    /// the caller-provided scratch then defers to the pre-quantized dispatcher,
    /// which picks the dotprod or `_base` path.
    pub unsafe fn gemv_q8_0_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        unsafe {
            let n_blocks = k / 32;
            q8_scales.resize(n_blocks, 0.0);
            q8_quants.resize(k, 0);
            quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
            gemv_q8_0_q8_0_neon(a_quant, q8_scales, q8_quants, y, _m, k);
        }
    }

    /// NEON Q6_K × Q8_0 integer GEMV with pre-quantized input.
    ///
    /// Extracts 6-bit quants as i8, dots with Q8_0 input using vdotq_s32.
    /// 16 sub-blocks of 16 values per Q6_K block, each with its own scale.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemv_q6k_q8_0_neon_dotprod(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        unsafe {
            let blocks_per_row = k / 256;
            let row_bytes = blocks_per_row * size_of::<BlockQ6K>();
            let a_base = a_quant.as_ptr() as usize;
            let xq_base = x_quants.as_ptr() as usize;
            let xs_base = x_scales.as_ptr() as usize;

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let row_start = i * row_bytes;
                let mut sumf = 0.0f32;
                let mask_0f = vdupq_n_u8(0x0F);
                let mask_03 = vdupq_n_u8(0x03);
                let offset_32 = vdupq_n_s8(32);
                let z = vdupq_n_s32(0);

                for bi in 0..blocks_per_row {
                    let blk =
                        &*((a_base + row_start + bi * size_of::<BlockQ6K>()) as *const BlockQ6K);
                    let d = f16::from_bits(blk.d).to_f32();
                    let ql = blk.ql.as_ptr();
                    let qh = blk.qh.as_ptr();
                    let sc = blk.scales.as_ptr();
                    let xq_off = bi * 256;

                    // Fused extraction + dot product: extract 16 6-bit quants, immediately
                    // dot with Q8_0 input. No intermediate buffer — stays in registers.
                    // Scale index tracks which of the 16 sub-block scales to use.
                    let mut sc_idx = 0usize;
                    let mut ql_p = 0usize;
                    let mut qh_p = 0usize;
                    let mut y_p = 0usize;

                    for _pass in 0..2 {
                        for half in 0..2 {
                            let l_off = half * 16;
                            let ql_lo_v = vld1q_u8(ql.add(ql_p + l_off));
                            let ql_hi_v = vld1q_u8(ql.add(ql_p + l_off + 32));
                            let qh_v = vld1q_u8(qh.add(qh_p + l_off));

                            // q1: values at y_p + l_off (16 values, sc_idx)
                            let q1 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vandq_u8(ql_lo_v, mask_0f),
                                    vshlq_n_u8::<4>(vandq_u8(qh_v, mask_03)),
                                )),
                                offset_32,
                            );
                            let xv1 = vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off));
                            let q8_bi1 = (xq_off + y_p + l_off) / 32;
                            let d1 =
                                d * (*sc.add(sc_idx) as f32) * *(xs_base as *const f32).add(q8_bi1);
                            sumf += d1 * vaddvq_s32(vdotq_s32(z, q1, xv1)) as f32;

                            // q2: values at y_p + l_off + 32 (16 values, sc_idx + 2)
                            let q2 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vandq_u8(ql_hi_v, mask_0f),
                                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh_v), mask_03)),
                                )),
                                offset_32,
                            );
                            let xv2 =
                                vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off + 32));
                            let q8_bi2 = (xq_off + y_p + l_off + 32) / 32;
                            let d2 = d
                                * (*sc.add(sc_idx + 2) as f32)
                                * *(xs_base as *const f32).add(q8_bi2);
                            sumf += d2 * vaddvq_s32(vdotq_s32(z, q2, xv2)) as f32;

                            // q3: values at y_p + l_off + 64 (16 values, sc_idx + 4)
                            let q3 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vshrq_n_u8::<4>(ql_lo_v),
                                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh_v), mask_03)),
                                )),
                                offset_32,
                            );
                            let xv3 =
                                vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off + 64));
                            let q8_bi3 = (xq_off + y_p + l_off + 64) / 32;
                            let d3 = d
                                * (*sc.add(sc_idx + 4) as f32)
                                * *(xs_base as *const f32).add(q8_bi3);
                            sumf += d3 * vaddvq_s32(vdotq_s32(z, q3, xv3)) as f32;

                            // q4: values at y_p + l_off + 96 (16 values, sc_idx + 6)
                            let q4 = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(
                                    vshrq_n_u8::<4>(ql_hi_v),
                                    vshlq_n_u8::<4>(vshrq_n_u8::<6>(qh_v)),
                                )),
                                offset_32,
                            );
                            let xv4 =
                                vld1q_s8((xq_base as *const i8).add(xq_off + y_p + l_off + 96));
                            let q8_bi4 = (xq_off + y_p + l_off + 96) / 32;
                            let d4 = d
                                * (*sc.add(sc_idx + 6) as f32)
                                * *(xs_base as *const f32).add(q8_bi4);
                            sumf += d4 * vaddvq_s32(vdotq_s32(z, q4, xv4)) as f32;

                            sc_idx += 1; // advance by 1 per half (is = l/16 = half)
                        }
                        y_p += 128;
                        ql_p += 64;
                        qh_p += 32;
                        sc_idx = 8; // second pass uses scales 8..15
                    }
                }

                *yi = sumf;
            };

            if y.len() >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows(y, crate::backend::cpu::gemv_min_rows(), compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON Q4_K × Q8_0 integer GEMV with pre-quantized input.
    ///
    /// Q4_K superblock = 256 values = 8 sub-blocks of 32, each with a 6-bit
    /// scale `sc` and 6-bit min `mn`. Dequant is `w = d·sc·q − dmin·mn` (q in
    /// 0..15, no zero-point offset). Dotting a weight row with activations x
    /// gives, per sub-block s: `d·sc_s·Σ(q·xq) − dmin·mn_s·Σ(xq)`, then scaled by
    /// the Q8_0 input scale `xs[s]`. Q8_0 input blocks are 32-wide, aligning 1:1
    /// with the sub-blocks; `Σ(xq)` (the min term) is `vdotq_s32` against an
    /// all-ones vector. The nibble/scale layout mirrors `vec_dot_q4_k_m_f32`:
    /// 4 groups of 64 values over 32 qs bytes, low nibble → sub-block 2j, high
    /// nibble → sub-block 2j+1.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemv_q4k_q8_0_neon_dotprod(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 256, 0, "Q4_K GEMV: k must be divisible by 256");
        debug_assert_eq!(y.len(), _m, "Q4_K GEMV: y.len() must equal m");
        unsafe {
            let blocks_per_row = k / 256;
            let row_bytes = blocks_per_row * size_of::<BlockQ4KM>();
            // Guards the unsafe per-row pointer math below (mirrors the scalar
            // `gemv_q4km_f32`): each of the m rows reads `row_bytes` from a_quant.
            debug_assert_eq!(
                a_quant.len(),
                _m * row_bytes,
                "Q4_K GEMV: a_quant size mismatch"
            );
            let a_base = a_quant.as_ptr() as usize;
            let xq_base = x_quants.as_ptr() as usize;
            let xs_base = x_scales.as_ptr() as usize;

            let compute_row = move |(i, yi): (usize, &mut f32)| unsafe {
                let row_start = i * row_bytes;
                let mask_0f = vdupq_n_u8(0x0F);
                let ones = vdupq_n_s8(1);
                let z = vdupq_n_s32(0);
                let mut sumf = 0.0f32;

                for bi in 0..blocks_per_row {
                    let blk =
                        &*((a_base + row_start + bi * size_of::<BlockQ4KM>()) as *const BlockQ4KM);
                    let d = f16::from_bits(blk.d).to_f32();
                    let dmin = f16::from_bits(blk.dmin).to_f32();

                    // Decode the 8 sub-block 6-bit scales and mins (shared with
                    // the scalar/f32 paths, so the packing can't drift).
                    let (sc, mn) = crate::quant::decode_q4km_scales(&blk.scales);

                    let qs = blk.qs.as_ptr();
                    let xq_off = bi * 256;

                    for j in 0..4 {
                        let qb0 = vld1q_u8(qs.add(j * 32));
                        let qb1 = vld1q_u8(qs.add(j * 32 + 16));

                        // Low nibbles → sub-block 2j; high nibbles → sub-block 2j+1.
                        // 4-bit quants are 0..15, so they stay positive as i8.
                        let wlo0 = vreinterpretq_s8_u8(vandq_u8(qb0, mask_0f));
                        let wlo1 = vreinterpretq_s8_u8(vandq_u8(qb1, mask_0f));
                        let whi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qb0));
                        let whi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qb1));

                        let sblo = 2 * j;
                        let sbhi = 2 * j + 1;
                        let xlo0 = vld1q_s8((xq_base as *const i8).add(xq_off + sblo * 32));
                        let xlo1 = vld1q_s8((xq_base as *const i8).add(xq_off + sblo * 32 + 16));
                        let xhi0 = vld1q_s8((xq_base as *const i8).add(xq_off + sbhi * 32));
                        let xhi1 = vld1q_s8((xq_base as *const i8).add(xq_off + sbhi * 32 + 16));

                        // Σ(q·xq) per sub-block (integer dot).
                        let dp_lo = vaddvq_s32(vdotq_s32(vdotq_s32(z, wlo0, xlo0), wlo1, xlo1));
                        let dp_hi = vaddvq_s32(vdotq_s32(vdotq_s32(z, whi0, xhi0), whi1, xhi1));
                        // Σ(xq) per sub-block (min term), via dot with all-ones.
                        let sx_lo = vaddvq_s32(vdotq_s32(vdotq_s32(z, ones, xlo0), ones, xlo1));
                        let sx_hi = vaddvq_s32(vdotq_s32(vdotq_s32(z, ones, xhi0), ones, xhi1));

                        let xs_lo = *(xs_base as *const f32).add((xq_off + sblo * 32) / 32);
                        let xs_hi = *(xs_base as *const f32).add((xq_off + sbhi * 32) / 32);

                        sumf += xs_lo
                            * (d * sc[sblo] as f32 * dp_lo as f32
                                - dmin * mn[sblo] as f32 * sx_lo as f32);
                        sumf += xs_hi
                            * (d * sc[sbhi] as f32 * dp_hi as f32
                                - dmin * mn[sbhi] as f32 * sx_hi as f32);
                    }
                }

                *yi = sumf;
            };

            if y.len() >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows(y, crate::backend::cpu::gemv_min_rows(), compute_row);
            } else {
                y.iter_mut().enumerate().for_each(compute_row);
            }
        }
    }

    /// Columns processed per pass in the K-quant GEMMs.
    ///
    /// This is the whole point of the GEMM over the GEMV: a Q4_K/Q6_K weight block
    /// is expensive to decode (nibble/6-bit extraction plus packed sub-block scales),
    /// and the GEMV pays that cost once per *column*. Decoding once and dotting
    /// against `KQ_COLS` activation columns amortizes it. 8 keeps the activation
    /// working set (`8 · k` bytes) inside L1 while the weight row — a few hundred
    /// bytes — stays hot across all `n / 8` passes.
    const KQ_COLS: usize = 8;

    /// Σ(xq) for every (column, 32-block) of the pre-quantized activations.
    ///
    /// The K-quant min term is `−dmin · mn_s · Σ(xq)`, and `Σ(xq)` depends only on
    /// the *activation* column — not the weight row. Computing it inside the row
    /// loop would redo identical work `m` times; hoisting it costs `n · k/32` and
    /// saves `m · n · k/32` dot-pairs.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn q8_0_col_sums(b_quants: &[i8], n: usize, k: usize) -> Vec<i32> {
        unsafe {
            let nb32 = k / 32;
            let mut sums = vec![0i32; n * nb32];
            let ones = vdupq_n_s8(1);
            let z = vdupq_n_s32(0);
            for j in 0..n {
                let base = b_quants.as_ptr().add(j * k);
                for b in 0..nb32 {
                    let p = base.add(b * 32);
                    let x0 = vld1q_s8(p);
                    let x1 = vld1q_s8(p.add(16));
                    sums[j * nb32 + b] = vaddvq_s32(vdotq_s32(vdotq_s32(z, ones, x0), ones, x1));
                }
            }
            sums
        }
    }

    /// Batched GEMM: C[m, n] = A_q4_k[m, k] @ B_q8_0[k, n].
    ///
    /// Same layout contract as the Q4_0 GEMM (`b_scales[j*nb + b]`,
    /// `b_quants[j*k + ..]`, row-major `out[i*n + j]`), and the same per-sub-block
    /// math as `gemv_q4k_q8_0_neon_dotprod`:
    ///
    /// ```text
    /// out[i][j] += xs_s · ( d·sc_s·Σ(q·xq) − dmin·mn_s·Σ(xq) )
    /// ```
    ///
    /// A Q4_K superblock is 256 values in 8 sub-blocks of 32, which aligns 1:1 with
    /// the Q8_0 input blocks — so sub-block `s` of superblock `bi` is input block
    /// `bi*8 + s`. Nibble layout mirrors `vec_dot_q4_k_m_f32`: 4 groups of 64 values
    /// over 32 `qs` bytes, low nibble → sub-block `2g`, high nibble → `2g+1`. The
    /// 6-bit scales come from the shared `quant::decode_q4km_scales`, so the packing
    /// cannot drift from the scalar path.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemm_q4_k_q8_0_neon_dotprod(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 256, 0, "Q4_K GEMM: k must be divisible by 256");
        let sb = k / 256;
        let nb32 = k / 32;
        let row_bytes = sb * size_of::<BlockQ4KM>();
        debug_assert_eq!(a_quant.len(), m * row_bytes, "Q4_K GEMM: a_quant size");
        debug_assert_eq!(b_quants.len(), n * k, "Q4_K GEMM: b_quants size");
        debug_assert_eq!(b_scales.len(), n * nb32, "Q4_K GEMM: b_scales size");
        debug_assert_eq!(out.len(), m * n, "Q4_K GEMM: out size");

        unsafe {
            let col_sums = q8_0_col_sums(b_quants, n, k);
            let a_ptr = a_quant.as_ptr() as usize;
            let bq_ptr = b_quants.as_ptr() as usize;
            let bs_ptr = b_scales.as_ptr() as usize;
            let cs_ptr = col_sums.as_ptr() as usize;

            let compute_row = move |(i, row_out): (usize, &mut [f32])| unsafe {
                let a = a_ptr as *const u8;
                let bq = bq_ptr as *const i8;
                let bs = bs_ptr as *const f32;
                let cs = cs_ptr as *const i32;
                let mask_0f = vdupq_n_u8(0x0F);
                let z = vdupq_n_s32(0);
                let row_start = i * row_bytes;

                let mut j0 = 0usize;
                while j0 < n {
                    let cols = KQ_COLS.min(n - j0);
                    let mut acc = [0.0f32; KQ_COLS];

                    for bi in 0..sb {
                        let blk = &*((a as usize + row_start + bi * size_of::<BlockQ4KM>())
                            as *const BlockQ4KM);
                        let d = half::f16::from_bits(blk.d).to_f32();
                        let dmin = half::f16::from_bits(blk.dmin).to_f32();
                        let (sc, mn) = crate::quant::decode_q4km_scales(&blk.scales);
                        let qs = blk.qs.as_ptr();

                        for g in 0..4 {
                            let qb0 = vld1q_u8(qs.add(g * 32));
                            let qb1 = vld1q_u8(qs.add(g * 32 + 16));
                            // 4-bit quants are 0..15, so they stay positive as i8.
                            let w = [
                                (
                                    vreinterpretq_s8_u8(vandq_u8(qb0, mask_0f)),
                                    vreinterpretq_s8_u8(vandq_u8(qb1, mask_0f)),
                                    2 * g,
                                ),
                                (
                                    vreinterpretq_s8_u8(vshrq_n_u8::<4>(qb0)),
                                    vreinterpretq_s8_u8(vshrq_n_u8::<4>(qb1)),
                                    2 * g + 1,
                                ),
                            ];

                            for (w0, w1, s) in w {
                                let xb = bi * 8 + s;
                                let dsc = d * sc[s] as f32;
                                let dmn = dmin * mn[s] as f32;
                                // `acc_j`, not `a` — `a` is the weight base pointer in
                                // the enclosing scope, and shadowing it inside an
                                // `unsafe` block is how a future edit reaching for the
                                // weights silently gets an `&mut f32` and reads
                                // arbitrary memory.
                                for (jj, acc_j) in acc.iter_mut().enumerate().take(cols) {
                                    let j = j0 + jj;
                                    let xp = bq.add(j * k + xb * 32);
                                    let x0 = vld1q_s8(xp);
                                    let x1 = vld1q_s8(xp.add(16));
                                    let dp = vaddvq_s32(vdotq_s32(vdotq_s32(z, w0, x0), w1, x1));
                                    let xs = *bs.add(j * nb32 + xb);
                                    let sx = *cs.add(j * nb32 + xb);
                                    *acc_j += xs * (dsc * dp as f32 - dmn * sx as f32);
                                }
                            }
                        }
                    }

                    row_out[j0..j0 + cols].copy_from_slice(&acc[..cols]);
                    j0 += cols;
                }
            };

            if m >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            } else {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            }
        }
    }

    /// Batched GEMM: C[m, n] = A_q6_k[m, k] @ B_q8_0[k, n].
    ///
    /// Same layout contract and 8-column amortization as the Q4_K GEMM, but the
    /// block geometry is meaningfully different and is the one real trap here:
    ///
    /// - **No min term.** Q6_K quants are signed (`−32` offset baked in at decode),
    ///   so there is no `dmin`/`Σ(xq)` correction — just `d · sc_s · Σ(q·xq)`.
    /// - **Sub-blocks are 16 wide, not 32.** A Q8_0 input block (32) therefore spans
    ///   *two* Q6_K scales, unlike Q4_K's clean 1:1. Conveniently a NEON `int8x16`
    ///   register is exactly one 16-element sub-block, so the 32-value group splits
    ///   into two registers, each dotted against its own half of the Q8_0 block and
    ///   scaled independently.
    ///
    /// Index math mirrors `dequantize_q6_k_block`: two halves of 128 values
    /// (`ql_off += 64`, `qh_off += 32`, `sc_off += 8`), each half holding 4 groups of
    /// 32 whose 6-bit quants are assembled as `(ql nibble) | (qh 2-bit pair) << 4`.
    /// Group `g` of half `nh` uses scales `sc[nh*8 + 2g + is]` (`is = l/16`) and lands
    /// on Q8_0 input block `bi*8 + nh*4 + g`.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemm_q6_k_q8_0_neon_dotprod(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 256, 0, "Q6_K GEMM: k must be divisible by 256");
        let sb = k / 256;
        let nb32 = k / 32;
        let row_bytes = sb * size_of::<BlockQ6K>();
        debug_assert_eq!(a_quant.len(), m * row_bytes, "Q6_K GEMM: a_quant size");
        debug_assert_eq!(b_quants.len(), n * k, "Q6_K GEMM: b_quants size");
        debug_assert_eq!(b_scales.len(), n * nb32, "Q6_K GEMM: b_scales size");
        debug_assert_eq!(out.len(), m * n, "Q6_K GEMM: out size");

        unsafe {
            let a_ptr = a_quant.as_ptr() as usize;
            let bq_ptr = b_quants.as_ptr() as usize;
            let bs_ptr = b_scales.as_ptr() as usize;

            let compute_row = move |(i, row_out): (usize, &mut [f32])| unsafe {
                let a = a_ptr as *const u8;
                let bq = bq_ptr as *const i8;
                let bs = bs_ptr as *const f32;
                let mask_0f = vdupq_n_u8(0x0F);
                let mask_03 = vdupq_n_u8(0x03);
                let off_32 = vdupq_n_s8(32);
                let z = vdupq_n_s32(0);
                let row_start = i * row_bytes;

                let mut j0 = 0usize;
                while j0 < n {
                    let cols = KQ_COLS.min(n - j0);
                    let mut acc = [0.0f32; KQ_COLS];

                    for bi in 0..sb {
                        let blk = &*((a as usize + row_start + bi * size_of::<BlockQ6K>())
                            as *const BlockQ6K);
                        let d = half::f16::from_bits(blk.d).to_f32();
                        let sc = blk.scales;
                        let ql = blk.ql.as_ptr();
                        let qh = blk.qh.as_ptr();

                        for nh in 0..2 {
                            let qlp = ql.add(nh * 64);
                            let qhp = qh.add(nh * 32);
                            // `a`/`b` suffix = the two 16-lane sub-blocks (l = 0..15,
                            // 16..31) of each 32-value group.
                            let ql_a0 = vld1q_u8(qlp);
                            let ql_a1 = vld1q_u8(qlp.add(16));
                            let ql_b0 = vld1q_u8(qlp.add(32));
                            let ql_b1 = vld1q_u8(qlp.add(48));
                            let qh0 = vld1q_u8(qhp);
                            let qh1 = vld1q_u8(qhp.add(16));

                            // 6-bit quant = nibble | (2 high bits << 4), then −32.
                            // `$qhs` arrives pre-shifted: `vshrq_n_u8::<0>` is illegal
                            // (N must be 1..=8), so group 0 passes `qh` unshifted.
                            macro_rules! q6 {
                                ($lo:expr, $qhs:expr, $hi_nibble:expr) => {{
                                    let lo = if $hi_nibble {
                                        vshrq_n_u8::<4>($lo)
                                    } else {
                                        vandq_u8($lo, mask_0f)
                                    };
                                    let hi = vandq_u8($qhs, mask_03);
                                    vsubq_s8(
                                        vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8::<4>(hi))),
                                        off_32,
                                    )
                                }};
                            }

                            let groups = [
                                (q6!(ql_a0, qh0, false), q6!(ql_a1, qh1, false)),
                                (
                                    q6!(ql_b0, vshrq_n_u8::<2>(qh0), false),
                                    q6!(ql_b1, vshrq_n_u8::<2>(qh1), false),
                                ),
                                (
                                    q6!(ql_a0, vshrq_n_u8::<4>(qh0), true),
                                    q6!(ql_a1, vshrq_n_u8::<4>(qh1), true),
                                ),
                                (
                                    q6!(ql_b0, vshrq_n_u8::<6>(qh0), true),
                                    q6!(ql_b1, vshrq_n_u8::<6>(qh1), true),
                                ),
                            ];

                            // Accumulate in the *same order and grouping* as
                            // `gemv_q6k_q8_0_neon_dotprod`: half outer, group inner,
                            // with `d · sc · xs` formed before multiplying the dot.
                            //
                            // This is not pedantry. Both orderings are valid math, but
                            // only this one is **bit-identical** to the per-token path,
                            // and Q6_K sums terms that nearly cancel: summing a group's
                            // two halves together first (the obvious structure) drifts
                            // to 3.4e-4 relative at ffn_down's k=4608 — invisible at
                            // small k, but enough to move the model's logits (cosine
                            // 0.9995 vs the 1.000000 the Q4_0 path achieves).
                            for h in 0..2 {
                                for (g, (w_h0, w_h1)) in groups.iter().enumerate() {
                                    let w = if h == 0 { *w_h0 } else { *w_h1 };
                                    let xb = bi * 8 + nh * 4 + g;
                                    let d_sc = d * sc[nh * 8 + 2 * g + h] as f32;
                                    for (jj, acc_j) in acc.iter_mut().enumerate().take(cols) {
                                        let j = j0 + jj;
                                        let xv = vld1q_s8(bq.add(j * k + xb * 32 + h * 16));
                                        let dp = vaddvq_s32(vdotq_s32(z, w, xv));
                                        let scale = d_sc * *bs.add(j * nb32 + xb);
                                        *acc_j += scale * dp as f32;
                                    }
                                }
                            }
                        }
                    }

                    row_out[j0..j0 + cols].copy_from_slice(&acc[..cols]);
                    j0 += cols;
                }
            };

            if m >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            } else {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            }
        }
    }

    /// Q6_K × Q8_0 GEMM dispatcher. Requires `dotprod` and `k % 256 == 0`; see
    /// [`gemm_q4_k_q8_0_neon`] for why the `k` check lives here rather than only in
    /// the caller's gate. Returns `false` without writing `out` when it cannot run.
    #[allow(dead_code)]
    pub unsafe fn gemm_q6_k_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        if !k_quant_gemm_available() || k % 256 != 0 {
            return false;
        }
        unsafe { gemm_q6_k_q8_0_neon_dotprod(a_quant, b_scales, b_quants, out, m, n, k) };
        true
    }

    /// Batched GEMM: C[m, n] = A_q4_0[m, k] @ B_q8_0[k, n].
    ///
    /// `b_scales` layout: n columns × blocks_per_col, i.e. b_scales[j * nb + b]
    /// `b_quants` layout: n columns × k elements, i.e. b_quants[j * k + b*32..]
    /// `out` layout: row-major m × n, i.e. out[i * n + j]
    ///
    /// Reads each weight row once and dots against all n Q8_0 columns.
    /// Uses 4-column grouping to amortize Q4_0 nibble extraction.
    /// Parallelized across output rows with rayon.
    ///
    /// Unused under the `blas` feature (SGEMM via Accelerate replaces it on
    /// the prefill hot path) but kept compiled so the GEMM microbench can
    /// still A/B against it.
    #[allow(dead_code)]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemm_q4_0_q8_0_neon_dotprod(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        debug_assert_eq!(a_quant.len(), m * (k / 32) * size_of::<BlockQ4_0>());
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * (k / 32));
        debug_assert_eq!(out.len(), m * n);
        unsafe {
            let nb = k / 32;
            let row_bytes = nb * size_of::<BlockQ4_0>();
            let a_ptr = a_quant.as_ptr() as usize;
            let bq_ptr = b_quants.as_ptr() as usize;
            let bs_ptr = b_scales.as_ptr() as usize;

            // Decode helpers — Q4_0 nibble extraction is the only difference vs Q8_0 GEMM.
            macro_rules! decode_q4_pair {
                ($a_base:expr, $off0:expr, $off1:expr, $mask_lo:expr, $offset_8:expr) => {{
                    let b0 = &*($a_base.add($off0) as *const BlockQ4_0);
                    let b1 = &*($a_base.add($off1) as *const BlockQ4_0);
                    let v0 = vld1q_u8(b0.qs.as_ptr());
                    let v1 = vld1q_u8(b1.qs.as_ptr());
                    (
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, $mask_lo)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, $mask_lo)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), $offset_8),
                        f16::from_bits(b0.d).to_f32(),
                        f16::from_bits(b1.d).to_f32(),
                    )
                }};
            }
            macro_rules! decode_q4_single {
                ($a_base:expr, $off:expr, $mask_lo:expr, $offset_8:expr) => {{
                    let b = &*($a_base.add($off) as *const BlockQ4_0);
                    let v = vld1q_u8(b.qs.as_ptr());
                    (
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, $mask_lo)), $offset_8),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), $offset_8),
                        f16::from_bits(b.d).to_f32(),
                    )
                }};
            }

            let compute_row = move |(i, row_out): (usize, &mut [f32])| unsafe {
                let mask_lo = vdupq_n_u8(0x0F);
                let offset_8 = vdupq_n_s8(0x8);
                let rs = i * row_bytes;
                let a = a_ptr as *const u8;
                let bq = bq_ptr as *const i8;
                let bs = bs_ptr as *const f32;
                let bsz = size_of::<BlockQ4_0>();

                // Process 8 columns at a time (halves Q4_0 decode count vs 4-col)
                let mut j = 0usize;
                while j + 8 <= n {
                    let mut s0a = vdupq_n_f32(0.0);
                    let mut s0b = vdupq_n_f32(0.0);
                    let mut s1a = vdupq_n_f32(0.0);
                    let mut s1b = vdupq_n_f32(0.0);
                    let mut s2a = vdupq_n_f32(0.0);
                    let mut s2b = vdupq_n_f32(0.0);
                    let mut s3a = vdupq_n_f32(0.0);
                    let mut s3b = vdupq_n_f32(0.0);
                    let mut s4a = vdupq_n_f32(0.0);
                    let mut s4b = vdupq_n_f32(0.0);
                    let mut s5a = vdupq_n_f32(0.0);
                    let mut s5b = vdupq_n_f32(0.0);
                    let mut s6a = vdupq_n_f32(0.0);
                    let mut s6b = vdupq_n_f32(0.0);
                    let mut s7a = vdupq_n_f32(0.0);
                    let mut s7b = vdupq_n_f32(0.0);
                    let xq = [
                        bq.add(j * k),
                        bq.add((j + 1) * k),
                        bq.add((j + 2) * k),
                        bq.add((j + 3) * k),
                        bq.add((j + 4) * k),
                        bq.add((j + 5) * k),
                        bq.add((j + 6) * k),
                        bq.add((j + 7) * k),
                    ];
                    let xs = [
                        bs.add(j * nb),
                        bs.add((j + 1) * nb),
                        bs.add((j + 2) * nb),
                        bs.add((j + 3) * nb),
                        bs.add((j + 4) * nb),
                        bs.add((j + 5) * nb),
                        bs.add((j + 6) * nb),
                        bs.add((j + 7) * nb),
                    ];

                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        if bi + 3 < nb {
                            _prefetch(
                                a.add(rs + (bi + 2) * bsz) as *const i8,
                                _PREFETCH_READ,
                                _PREFETCH_LOCALITY2,
                            );
                        }
                        let (w0l, w0h, w1l, w1h, d0, d1) = decode_q4_pair!(
                            a,
                            rs + bi * bsz,
                            rs + (bi + 1) * bsz,
                            mask_lo,
                            offset_8
                        );
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[0], xs[0], bi, s0a, s0b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[1], xs[1], bi, s1a, s1b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[2], xs[2], bi, s2a, s2b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[3], xs[3], bi, s3a, s3b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[4], xs[4], bi, s4a, s4b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[5], xs[5], bi, s5a, s5b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[6], xs[6], bi, s6a, s6b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq[7], xs[7], bi, s7a, s7b);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q4_single!(a, rs + bi * bsz, mask_lo, offset_8);
                        gemm_dot_single!(wl, wh, d, xq[0], xs[0], bi, s0a);
                        gemm_dot_single!(wl, wh, d, xq[1], xs[1], bi, s1a);
                        gemm_dot_single!(wl, wh, d, xq[2], xs[2], bi, s2a);
                        gemm_dot_single!(wl, wh, d, xq[3], xs[3], bi, s3a);
                        gemm_dot_single!(wl, wh, d, xq[4], xs[4], bi, s4a);
                        gemm_dot_single!(wl, wh, d, xq[5], xs[5], bi, s5a);
                        gemm_dot_single!(wl, wh, d, xq[6], xs[6], bi, s6a);
                        gemm_dot_single!(wl, wh, d, xq[7], xs[7], bi, s7a);
                    }
                    row_out[j] = vaddvq_f32(s0a) + vaddvq_f32(s0b);
                    row_out[j + 1] = vaddvq_f32(s1a) + vaddvq_f32(s1b);
                    row_out[j + 2] = vaddvq_f32(s2a) + vaddvq_f32(s2b);
                    row_out[j + 3] = vaddvq_f32(s3a) + vaddvq_f32(s3b);
                    row_out[j + 4] = vaddvq_f32(s4a) + vaddvq_f32(s4b);
                    row_out[j + 5] = vaddvq_f32(s5a) + vaddvq_f32(s5b);
                    row_out[j + 6] = vaddvq_f32(s6a) + vaddvq_f32(s6b);
                    row_out[j + 7] = vaddvq_f32(s7a) + vaddvq_f32(s7b);
                    j += 8;
                }
                // 4-column remainder
                while j + 4 <= n {
                    let mut s0a = vdupq_n_f32(0.0);
                    let mut s0b = vdupq_n_f32(0.0);
                    let mut s1a = vdupq_n_f32(0.0);
                    let mut s1b = vdupq_n_f32(0.0);
                    let mut s2a = vdupq_n_f32(0.0);
                    let mut s2b = vdupq_n_f32(0.0);
                    let mut s3a = vdupq_n_f32(0.0);
                    let mut s3b = vdupq_n_f32(0.0);
                    let (xq0, xq1, xq2, xq3) = (
                        bq.add(j * k),
                        bq.add((j + 1) * k),
                        bq.add((j + 2) * k),
                        bq.add((j + 3) * k),
                    );
                    let (xs0, xs1, xs2, xs3) = (
                        bs.add(j * nb),
                        bs.add((j + 1) * nb),
                        bs.add((j + 2) * nb),
                        bs.add((j + 3) * nb),
                    );
                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        let (w0l, w0h, w1l, w1h, d0, d1) = decode_q4_pair!(
                            a,
                            rs + bi * bsz,
                            rs + (bi + 1) * bsz,
                            mask_lo,
                            offset_8
                        );
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq0, xs0, bi, s0a, s0b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq1, xs1, bi, s1a, s1b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq2, xs2, bi, s2a, s2b);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq3, xs3, bi, s3a, s3b);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q4_single!(a, rs + bi * bsz, mask_lo, offset_8);
                        gemm_dot_single!(wl, wh, d, xq0, xs0, bi, s0a);
                        gemm_dot_single!(wl, wh, d, xq1, xs1, bi, s1a);
                        gemm_dot_single!(wl, wh, d, xq2, xs2, bi, s2a);
                        gemm_dot_single!(wl, wh, d, xq3, xs3, bi, s3a);
                    }
                    row_out[j] = vaddvq_f32(s0a) + vaddvq_f32(s0b);
                    row_out[j + 1] = vaddvq_f32(s1a) + vaddvq_f32(s1b);
                    row_out[j + 2] = vaddvq_f32(s2a) + vaddvq_f32(s2b);
                    row_out[j + 3] = vaddvq_f32(s3a) + vaddvq_f32(s3b);
                    j += 4;
                }
                // Remaining columns (< 4)
                while j < n {
                    let mut sumv0 = vdupq_n_f32(0.0);
                    let mut sumv1 = vdupq_n_f32(0.0);
                    let xq = bq.add(j * k);
                    let xs = bs.add(j * nb);
                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        let (w0l, w0h, w1l, w1h, d0, d1) = decode_q4_pair!(
                            a,
                            rs + bi * bsz,
                            rs + (bi + 1) * bsz,
                            mask_lo,
                            offset_8
                        );
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq, xs, bi, sumv0, sumv1);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q4_single!(a, rs + bi * bsz, mask_lo, offset_8);
                        gemm_dot_single!(wl, wh, d, xq, xs, bi, sumv0);
                    }
                    row_out[j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
                    j += 1;
                }
            };

            if m >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            } else {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            }
        }
    }

    /// Batched GEMM: C[m, n] = A_q8_0[m, k] @ B_q8_0[k, n].
    ///
    /// Same layout and shared dot-product macros as Q4_0 GEMM, but with
    /// Q8_0 weight blocks (direct i8 load, no nibble extraction).
    ///
    /// Unused under the `blas` feature (SGEMM via Accelerate replaces it on
    /// the prefill hot path) but kept compiled so the GEMM microbench can
    /// still A/B against it.
    #[allow(dead_code)]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemm_q8_0_q8_0_neon_dotprod(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        debug_assert_eq!(a_quant.len(), m * (k / 32) * size_of::<BlockQ8_0>());
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * (k / 32));
        debug_assert_eq!(out.len(), m * n);
        unsafe {
            let nb = k / 32;
            let row_bytes = nb * size_of::<BlockQ8_0>();
            let a_ptr = a_quant.as_ptr() as usize;
            let bq_ptr = b_quants.as_ptr() as usize;
            let bs_ptr = b_scales.as_ptr() as usize;

            // Q8_0 decode: direct i8 load (no nibble extraction needed).
            macro_rules! decode_q8_pair {
                ($a_base:expr, $off0:expr, $off1:expr) => {{
                    let b0 = &*($a_base.add($off0) as *const BlockQ8_0);
                    let b1 = &*($a_base.add($off1) as *const BlockQ8_0);
                    (
                        vld1q_s8(b0.quants.as_ptr()),
                        vld1q_s8(b0.quants.as_ptr().add(16)),
                        vld1q_s8(b1.quants.as_ptr()),
                        vld1q_s8(b1.quants.as_ptr().add(16)),
                        f16::from_bits(b0.delta).to_f32(),
                        f16::from_bits(b1.delta).to_f32(),
                    )
                }};
            }
            macro_rules! decode_q8_single {
                ($a_base:expr, $off:expr) => {{
                    let b = &*($a_base.add($off) as *const BlockQ8_0);
                    (
                        vld1q_s8(b.quants.as_ptr()),
                        vld1q_s8(b.quants.as_ptr().add(16)),
                        f16::from_bits(b.delta).to_f32(),
                    )
                }};
            }

            let compute_row = move |(i, row_out): (usize, &mut [f32])| unsafe {
                let rs = i * row_bytes;
                let a = a_ptr as *const u8;
                let bq = bq_ptr as *const i8;
                let bs = bs_ptr as *const f32;
                let bsz = size_of::<BlockQ8_0>();

                // Single-column loop (Q8_0 has no nibble decode to amortize,
                // so 4-column grouping provides minimal benefit)
                for j in 0..n {
                    let mut sumv0 = vdupq_n_f32(0.0);
                    let mut sumv1 = vdupq_n_f32(0.0);
                    let xq = bq.add(j * k);
                    let xs = bs.add(j * nb);
                    let mut bi = 0usize;
                    while bi + 1 < nb {
                        let (w0l, w0h, w1l, w1h, d0, d1) =
                            decode_q8_pair!(a, rs + bi * bsz, rs + (bi + 1) * bsz);
                        gemm_dot_pair!(w0l, w0h, w1l, w1h, d0, d1, xq, xs, bi, sumv0, sumv1);
                        bi += 2;
                    }
                    if bi < nb {
                        let (wl, wh, d) = decode_q8_single!(a, rs + bi * bsz);
                        gemm_dot_single!(wl, wh, d, xq, xs, bi, sumv0);
                    }
                    row_out[j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
                }
            };

            if m >= super::super::cpu::gemv_par_threshold() {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            } else {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            }
        }
    }

    /// NEON Q6_K GEMV dispatcher: quantizes x to Q8_0 then runs the integer
    /// dotprod path when available, else a plain-NEON f32 fallback.
    pub unsafe fn gemv_q6k_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        if cpu_features().tier >= CpuTier::NeonDotprod {
            unsafe {
                let n_blocks = k / 32;
                q8_scales.resize(n_blocks, 0.0);
                q8_quants.resize(k, 0);
                quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
                gemv_q6k_q8_0_neon_dotprod(a_quant, q8_scales, q8_quants, y, _m, k);
            }
        } else {
            gemv_q6k_fallback(a_quant, x, y, k);
        }
    }

    /// NEON Q4_K GEMV: y[m] = A_q4_k[m,k] @ x_f32[k]. Quantizes x to Q8_0 into
    /// the caller-provided scratch then runs the integer dotprod kernel; on
    /// baseline NEON (no FEAT_DotProd) it defers to the exact-f32
    /// `backend::cpu::gemv_q4km_f32`.
    pub unsafe fn gemv_q4k_f32_neon(
        a_quant: &[u8],
        x: &[f32],
        y: &mut [f32],
        _m: usize,
        k: usize,
        q8_scales: &mut Vec<f32>,
        q8_quants: &mut Vec<i8>,
    ) {
        // Shape checks matching the scalar `gemv_q4km_f32`; in particular
        // `k % 256 == 0`, else `blocks_per_row = k / 256` would silently
        // truncate the row instead of failing.
        debug_assert_eq!(x.len(), k);
        debug_assert_eq!(y.len(), _m);
        debug_assert_eq!(k % 256, 0, "Q4_K GEMV: k must be divisible by 256");
        if cpu_features().tier >= CpuTier::NeonDotprod {
            unsafe {
                let n_blocks = k / 32;
                q8_scales.resize(n_blocks, 0.0);
                q8_quants.resize(k, 0);
                quantize_f32_to_q8_0_neon(x, q8_scales, q8_quants);
                gemv_q4k_q8_0_neon_dotprod(a_quant, q8_scales, q8_quants, y, _m, k);
            }
        } else {
            crate::backend::cpu::gemv_q4km_f32(a_quant, x, y, _m, k);
        }
    }

    // ── Pre-quantized / GEMM dispatchers + NEON-without-dotprod fallbacks ────
    //
    // The kernels above tagged `*_dotprod` require FEAT_DotProd (`vdotq_s32`).
    // These public entry points keep the original signatures so every call site
    // is unchanged; they branch on `cpu_features().tier` (so `CERA_CPU_TIER` can
    // force a lower path). The `*_base`
    // fallbacks run on baseline NEON: they emulate `vdotq_s32` with `vmull_s8` +
    // pairwise-add (bit-identical to the real instruction), staying on the
    // integer path and avoiding per-element int→f32 conversion. Q6_K keeps the
    // simpler f32-reconstruct fallback — its 6-bit unpack isn't worth a bespoke
    // integer kernel for a rare quant. Correctness is verified on dotprod
    // hardware by `fallback_tests`, comparing each fallback to its `*_dotprod`
    // sibling.

    /// Emulate `vdotq_s32(acc, a, b)` on baseline NEON (no FEAT_DotProd).
    /// `vmull_s8` widens the 16 signed int8×int8 products to int16; two pairwise
    /// adds then reduce them into the same four groups-of-four int32 lanes that
    /// `vdotq_s32` produces — so the result is bit-identical.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vdotq_s32_emu(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        unsafe {
            let p_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
            let p_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
            let s_lo = vpaddlq_s16(p_lo);
            let s_hi = vpaddlq_s16(p_hi);
            vaddq_s32(acc, vpaddq_s32(s_lo, s_hi))
        }
    }

    /// Reconstruct f32 input from a Q8_0-quantized vector (`x[i] = q[i] * scale`).
    /// Used only by the Q6_K f32-reconstruct fallback.
    fn reconstruct_q8_0_input(x_scales: &[f32], x_quants: &[i8], k: usize) -> Vec<f32> {
        let nb = k / 32;
        let mut xf = vec![0.0f32; k];
        for bi in 0..nb {
            let s = x_scales[bi];
            for l in 0..32 {
                xf[bi * 32 + l] = x_quants[bi * 32 + l] as f32 * s;
            }
        }
        xf
    }

    /// Baseline-NEON Q4_0 × Q8_0 GEMV using the emulated integer dot.
    #[target_feature(enable = "neon")]
    unsafe fn gemv_q4_0_q8_0_neon_base(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();
        let compute_row = |(i, yi): (usize, &mut f32)| unsafe {
            let mask_lo = vdupq_n_u8(0x0F);
            let offset_8 = vdupq_n_s8(0x8);
            let row_start = i * row_bytes;
            // Accumulate into an f32x4 (scaled per block via vmlaq_n_f32) like the
            // dotprod kernels; the cross-lane reduction happens once at the end.
            let mut sumv = vdupq_n_f32(0.0);
            for bi in 0..blocks_per_row {
                let b = &*(a_quant
                    .as_ptr()
                    .add(row_start + bi * size_of::<BlockQ4_0>())
                    as *const BlockQ4_0);
                let v = vld1q_u8(b.qs.as_ptr());
                let v_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask_lo)), offset_8);
                let v_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), offset_8);
                let y_lo = vld1q_s8(x_quants.as_ptr().add(bi * 32));
                let y_hi = vld1q_s8(x_quants.as_ptr().add(bi * 32 + 16));
                let z = vdupq_n_s32(0);
                let p = vdotq_s32_emu(vdotq_s32_emu(z, v_lo, y_lo), v_hi, y_hi);
                let d = f16::from_bits(b.d).to_f32() * x_scales[bi];
                sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), d);
            }
            *yi = vaddvq_f32(sumv);
        };
        if y.len() >= super::super::cpu::gemv_par_threshold() {
            super::super::cpu::par_rows(y, super::super::cpu::gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }

    /// Baseline-NEON Q8_0 × Q8_0 GEMV using the emulated integer dot.
    #[target_feature(enable = "neon")]
    unsafe fn gemv_q8_0_q8_0_neon_base(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * size_of::<BlockQ8_0>();
        let compute_row = |(i, yi): (usize, &mut f32)| unsafe {
            let row_start = i * row_bytes;
            let mut sumv = vdupq_n_f32(0.0);
            for bi in 0..blocks_per_row {
                let wb = &*(a_quant
                    .as_ptr()
                    .add(row_start + bi * size_of::<BlockQ8_0>())
                    as *const BlockQ8_0);
                let w_lo = vld1q_s8(wb.quants.as_ptr());
                let w_hi = vld1q_s8(wb.quants.as_ptr().add(16));
                let x_lo = vld1q_s8(x_quants.as_ptr().add(bi * 32));
                let x_hi = vld1q_s8(x_quants.as_ptr().add(bi * 32 + 16));
                let z = vdupq_n_s32(0);
                let p = vdotq_s32_emu(vdotq_s32_emu(z, w_lo, x_lo), w_hi, x_hi);
                let d = f16::from_bits(wb.delta).to_f32() * x_scales[bi];
                sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), d);
            }
            *yi = vaddvq_f32(sumv);
        };
        if y.len() >= super::super::cpu::gemv_par_threshold() {
            super::super::cpu::par_rows(y, super::super::cpu::gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }

    /// Baseline-NEON Q4_0 × Q8_0 GEMM using the emulated integer dot.
    // Dispatch target of `gemm_q4_0_q8_0_neon`, whose only non-test consumer is
    // `transformer::gemm_preq` (gated `not(feature = "blas")`); dead under
    // --all-features (blas on), live under the default CI gate.
    #[allow(dead_code)]
    #[target_feature(enable = "neon")]
    unsafe fn gemm_q4_0_q8_0_neon_base(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ4_0>();
        let compute_row = |(i, row): (usize, &mut [f32])| unsafe {
            let mask_lo = vdupq_n_u8(0x0F);
            let offset_8 = vdupq_n_s8(0x8);
            let row_start = i * row_bytes;
            for j in 0..n {
                let mut sumv = vdupq_n_f32(0.0);
                for bi in 0..nb {
                    let b = &*(a_quant
                        .as_ptr()
                        .add(row_start + bi * size_of::<BlockQ4_0>())
                        as *const BlockQ4_0);
                    let v = vld1q_u8(b.qs.as_ptr());
                    let v_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask_lo)), offset_8);
                    let v_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), offset_8);
                    let y_lo = vld1q_s8(b_quants.as_ptr().add(j * k + bi * 32));
                    let y_hi = vld1q_s8(b_quants.as_ptr().add(j * k + bi * 32 + 16));
                    let z = vdupq_n_s32(0);
                    let p = vdotq_s32_emu(vdotq_s32_emu(z, v_lo, y_lo), v_hi, y_hi);
                    let d = f16::from_bits(b.d).to_f32() * b_scales[j * nb + bi];
                    sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), d);
                }
                row[j] = vaddvq_f32(sumv);
            }
        };
        if m >= super::super::cpu::gemv_par_threshold() {
            super::super::cpu::par_rows_n(out, n, 256, compute_row);
        } else {
            out.chunks_mut(n).enumerate().for_each(compute_row);
        }
    }

    /// Baseline-NEON Q8_0 × Q8_0 GEMM using the emulated integer dot.
    // Dispatch target of `gemm_q8_0_q8_0_neon`, whose only non-test consumer is
    // `transformer::gemm_preq` (gated `not(feature = "blas")`); dead under
    // --all-features (blas on), live under the default CI gate.
    #[allow(dead_code)]
    #[target_feature(enable = "neon")]
    unsafe fn gemm_q8_0_q8_0_neon_base(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ8_0>();
        let compute_row = |(i, row): (usize, &mut [f32])| unsafe {
            let row_start = i * row_bytes;
            for j in 0..n {
                let mut sumv = vdupq_n_f32(0.0);
                for bi in 0..nb {
                    let wb = &*(a_quant
                        .as_ptr()
                        .add(row_start + bi * size_of::<BlockQ8_0>())
                        as *const BlockQ8_0);
                    let w_lo = vld1q_s8(wb.quants.as_ptr());
                    let w_hi = vld1q_s8(wb.quants.as_ptr().add(16));
                    let y_lo = vld1q_s8(b_quants.as_ptr().add(j * k + bi * 32));
                    let y_hi = vld1q_s8(b_quants.as_ptr().add(j * k + bi * 32 + 16));
                    let z = vdupq_n_s32(0);
                    let p = vdotq_s32_emu(vdotq_s32_emu(z, w_lo, y_lo), w_hi, y_hi);
                    let d = f16::from_bits(wb.delta).to_f32() * b_scales[j * nb + bi];
                    sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), d);
                }
                row[j] = vaddvq_f32(sumv);
            }
        };
        if m >= super::super::cpu::gemv_par_threshold() {
            super::super::cpu::par_rows_n(out, n, 256, compute_row);
        } else {
            out.chunks_mut(n).enumerate().for_each(compute_row);
        }
    }

    /// Plain-NEON Q6_K GEMV fallback. Mirrors `backend::cpu::gemv_q6k_f32`.
    fn gemv_q6k_fallback(a_quant: &[u8], x: &[f32], y: &mut [f32], k: usize) {
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * size_of::<BlockQ6K>();
        let compute_row = |(i, yi): (usize, &mut f32)| {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let off = row_start + bi * size_of::<BlockQ6K>();
                let block = unsafe { &*(a_quant.as_ptr().add(off) as *const BlockQ6K) };
                sum += crate::quant::vec_dot_q6_k_f32(block, &x[bi * 256..(bi + 1) * 256]);
            }
            *yi = sum;
        };
        if y.len() >= super::super::cpu::gemv_par_threshold() {
            super::super::cpu::par_rows(y, super::super::cpu::gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }

    /// Q4_0 pre-quantized GEMV dispatcher (input already Q8_0).
    pub unsafe fn gemv_q4_0_q8_0_neon(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        // Compare against `tier` (not the raw `dotprod` flag) so `CERA_CPU_TIER`
        // can force the base path, e.g. for parity testing on dotprod hardware.
        if cpu_features().tier >= CpuTier::NeonDotprod {
            unsafe { gemv_q4_0_q8_0_neon_dotprod(a_quant, x_scales, x_quants, y, _m, k) }
        } else {
            unsafe { gemv_q4_0_q8_0_neon_base(a_quant, x_scales, x_quants, y, _m, k) }
        }
    }

    /// Q8_0 pre-quantized GEMV dispatcher (input already Q8_0).
    pub unsafe fn gemv_q8_0_q8_0_neon(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        if cpu_features().tier >= CpuTier::NeonDotprod {
            unsafe { gemv_q8_0_q8_0_neon_dotprod(a_quant, x_scales, x_quants, y, _m, k) }
        } else {
            unsafe { gemv_q8_0_q8_0_neon_base(a_quant, x_scales, x_quants, y, _m, k) }
        }
    }

    /// Q6_K pre-quantized GEMV dispatcher (input already Q8_0).
    pub unsafe fn gemv_q6k_q8_0_neon(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        _m: usize,
        k: usize,
    ) {
        if cpu_features().tier >= CpuTier::NeonDotprod {
            unsafe { gemv_q6k_q8_0_neon_dotprod(a_quant, x_scales, x_quants, y, _m, k) }
        } else {
            let xf = reconstruct_q8_0_input(x_scales, x_quants, k);
            gemv_q6k_fallback(a_quant, &xf, y, k);
        }
    }

    // ── aarch64 i8mm tier ────────────────────────────────────────────────────
    //
    // `vmmlaq_s32` (FEAT_I8MM, ARMv8.6) does a 2×8 · 8×2 → 2×2 int8 matmul in one
    // op, a natural fit for Q8_0 GEMM (2 weight rows × 2 input cols per step).
    // i8mm always implies dotprod, so this is purely a prefill speedup over the
    // dotprod GEMM — never a correctness necessity (the dispatcher still has the
    // dotprod path).
    //
    // NOTE: i8mm is ARMv8.6, which the aarch64 dev host (M1: dotprod, no i8mm)
    // can't execute — it compiles natively (the intrinsic is gated, not run).
    // It IS validated on CI by the `simd-i8mm` job on `ubuntu-24.04-arm` (Azure
    // Cobalt 100 / Neoverse N2), which runs `i8mm_gemm_matches_dotprod` under
    // `CERA_REQUIRE_SIMD=i8mm` so a missing feature fails rather than skips.

    /// Scalar single-output Q8_0 GEMM dot, for odd row/col remainders.
    // Remainder helper for the i8mm/neon Q8_0 kernels, reachable (non-test) only
    // via `transformer::gemm_preq` (gated `not(feature = "blas")`); dead under
    // --all-features (blas on), live under the default CI gate.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn gemm_q8_0_scalar_dot(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        i: usize,
        j: usize,
        nb: usize,
        k: usize,
        row_bytes: usize,
    ) -> f32 {
        let mut acc = 0.0f32;
        for bi in 0..nb {
            let wb = unsafe {
                &*(a_quant
                    .as_ptr()
                    .add(i * row_bytes + bi * size_of::<BlockQ8_0>())
                    as *const BlockQ8_0)
            };
            let dw = f16::from_bits(wb.delta).to_f32();
            let db = b_scales[j * nb + bi];
            let mut s = 0i32;
            for l in 0..32 {
                s += wb.quants[l] as i32 * b_quants[j * k + bi * 32 + l] as i32;
            }
            acc += s as f32 * dw * db;
        }
        acc
    }

    /// i8mm Q8_0 × Q8_0 GEMM. Processes 2×2 output tiles with `vmmlaq_s32`,
    /// parallelized across row-pairs; odd row/col remainders use the scalar dot.
    // i8mm dispatch target of `gemm_q8_0_q8_0_neon`, whose only non-test consumer
    // is `transformer::gemm_preq` (gated `not(feature = "blas")`); dead under
    // --all-features (blas on), live under the default CI gate.
    #[allow(dead_code)]
    #[target_feature(enable = "neon,i8mm")]
    unsafe fn gemm_q8_0_q8_0_neon_i8mm(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ8_0>();
        let m_even = m & !1;
        let n_even = n & !1;

        // Main even×even tiles, parallel over 2-row strips.
        {
            #[cfg_attr(not(feature = "parallel"), allow(unused_imports))]
            use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            out[..m_even * n]
                .par_chunks_mut(2 * n)
                .enumerate()
                .for_each(|(p, strip)| {
                    let i = p * 2;
                    for j in (0..n_even).step_by(2) {
                        let (mut s00, mut s01, mut s10, mut s11) = (0.0f32, 0.0, 0.0, 0.0);
                        for bi in 0..nb {
                            let (wb0, wb1) = unsafe {
                                (
                                    &*(a_quant
                                        .as_ptr()
                                        .add(i * row_bytes + bi * size_of::<BlockQ8_0>())
                                        as *const BlockQ8_0),
                                    &*(a_quant
                                        .as_ptr()
                                        .add((i + 1) * row_bytes + bi * size_of::<BlockQ8_0>())
                                        as *const BlockQ8_0),
                                )
                            };
                            let dw0 = f16::from_bits(wb0.delta).to_f32();
                            let dw1 = f16::from_bits(wb1.delta).to_f32();
                            let db0 = b_scales[j * nb + bi];
                            let db1 = b_scales[(j + 1) * nb + bi];
                            let mut acc = unsafe { vdupq_n_s32(0) };
                            for c in 0..4 {
                                let off = c * 8;
                                unsafe {
                                    // a: row0 = weight i, row1 = weight i+1 (8 deep).
                                    let av = vcombine_s8(
                                        vld1_s8(wb0.quants.as_ptr().add(off)),
                                        vld1_s8(wb1.quants.as_ptr().add(off)),
                                    );
                                    // b: row0 = input col j, row1 = input col j+1.
                                    let bv = vcombine_s8(
                                        vld1_s8(b_quants.as_ptr().add(j * k + bi * 32 + off)),
                                        vld1_s8(b_quants.as_ptr().add((j + 1) * k + bi * 32 + off)),
                                    );
                                    acc = vmmlaq_s32(acc, av, bv);
                                }
                            }
                            // Lanes: [W_i·B_j, W_i·B_{j+1}, W_{i+1}·B_j, W_{i+1}·B_{j+1}].
                            let (d00, d01, d10, d11) = unsafe {
                                (
                                    vgetq_lane_s32::<0>(acc) as f32,
                                    vgetq_lane_s32::<1>(acc) as f32,
                                    vgetq_lane_s32::<2>(acc) as f32,
                                    vgetq_lane_s32::<3>(acc) as f32,
                                )
                            };
                            s00 += d00 * dw0 * db0;
                            s01 += d01 * dw0 * db1;
                            s10 += d10 * dw1 * db0;
                            s11 += d11 * dw1 * db1;
                        }
                        strip[j] = s00;
                        strip[j + 1] = s01;
                        strip[n + j] = s10;
                        strip[n + j + 1] = s11;
                    }
                    // Odd last column within this strip.
                    if n_even < n {
                        let j = n - 1;
                        strip[j] = gemm_q8_0_scalar_dot(
                            a_quant, b_scales, b_quants, i, j, nb, k, row_bytes,
                        );
                        strip[n + j] = gemm_q8_0_scalar_dot(
                            a_quant,
                            b_scales,
                            b_quants,
                            i + 1,
                            j,
                            nb,
                            k,
                            row_bytes,
                        );
                    }
                });
        }

        // Odd last row (covers all columns).
        if m_even < m {
            let i = m - 1;
            for j in 0..n {
                out[i * n + j] =
                    gemm_q8_0_scalar_dot(a_quant, b_scales, b_quants, i, j, nb, k, row_bytes);
            }
        }
    }

    /// Scalar single-output Q4_0 GEMM dot, for odd row/col remainders of the
    /// i8mm kernel. Decodes each block's 32 nibbles in the canonical Q4_0 order
    /// (low nibble of `qs[l]` = element `l`, high nibble = element `16 + l`),
    /// matching `gemm_dot_single!`.
    // Remainder helper for the i8mm Q4_0 kernel, reachable (non-test) only via
    // `transformer::gemm_preq` (gated `not(feature = "blas")`); dead under
    // --all-features (blas on), live under the default CI gate.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn gemm_q4_0_scalar_dot(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        i: usize,
        j: usize,
        nb: usize,
        k: usize,
        row_bytes: usize,
    ) -> f32 {
        let mut acc = 0.0f32;
        for bi in 0..nb {
            let wb = unsafe {
                &*(a_quant
                    .as_ptr()
                    .add(i * row_bytes + bi * size_of::<BlockQ4_0>())
                    as *const BlockQ4_0)
            };
            let dw = f16::from_bits(wb.d).to_f32();
            let db = b_scales[j * nb + bi];
            let mut s = 0i32;
            for l in 0..16 {
                let lo = (wb.qs[l] & 0x0F) as i32 - 8;
                let hi = (wb.qs[l] >> 4) as i32 - 8;
                s += lo * b_quants[j * k + bi * 32 + l] as i32;
                s += hi * b_quants[j * k + bi * 32 + 16 + l] as i32;
            }
            acc += s as f32 * dw * db;
        }
        acc
    }

    /// i8mm Q4_0 × Q8_0 GEMM. Same 2×2-tile `vmmlaq_s32` structure as
    /// [`gemm_q8_0_q8_0_neon_i8mm`], parallelized across 2-row strips; the only
    /// difference is decoding each Q4_0 block's 32 packed nibbles into two
    /// `int8x16` halves (low = elements 0..15, high = elements 16..31, recentered
    /// by −8) before feeding the four 8-lane chunks to the matrix-multiply. The
    /// activation (B) side is already Q8_0 int8, identical to the Q8_0 kernel.
    /// Odd row/col remainders fall back to [`gemm_q4_0_scalar_dot`].
    // i8mm dispatch target of `gemm_q4_0_q8_0_neon`, whose only non-test consumer
    // is `transformer::gemm_preq` (gated `not(feature = "blas")`); dead under
    // --all-features (blas on), live under the default CI gate.
    #[allow(dead_code)]
    #[target_feature(enable = "neon,i8mm")]
    unsafe fn gemm_q4_0_q8_0_neon_i8mm(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        debug_assert_eq!(a_quant.len(), m * (k / 32) * size_of::<BlockQ4_0>());
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * (k / 32));
        debug_assert_eq!(out.len(), m * n);

        let nb = k / 32;
        let bsz = size_of::<BlockQ4_0>();
        let row_bytes = nb * bsz;
        let m_even = m & !1;
        let n_even = n & !1;
        let mask_lo = unsafe { vdupq_n_u8(0x0F) };
        let offset_8 = unsafe { vdupq_n_s8(0x8) };

        // Main even×even tiles, parallel over 2-row strips.
        {
            #[cfg_attr(not(feature = "parallel"), allow(unused_imports))]
            use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            out[..m_even * n]
                .par_chunks_mut(2 * n)
                .enumerate()
                .for_each(|(p, strip)| {
                    let i = p * 2;
                    for j in (0..n_even).step_by(2) {
                        let (mut s00, mut s01, mut s10, mut s11) = (0.0f32, 0.0, 0.0, 0.0);
                        for bi in 0..nb {
                            let (wb0, wb1) = unsafe {
                                (
                                    &*(a_quant.as_ptr().add(i * row_bytes + bi * bsz)
                                        as *const BlockQ4_0),
                                    &*(a_quant.as_ptr().add((i + 1) * row_bytes + bi * bsz)
                                        as *const BlockQ4_0),
                                )
                            };
                            let dw0 = f16::from_bits(wb0.d).to_f32();
                            let dw1 = f16::from_bits(wb1.d).to_f32();
                            let db0 = b_scales[j * nb + bi];
                            let db1 = b_scales[(j + 1) * nb + bi];
                            let acc = unsafe {
                                // Decode both weight rows: low/high nibbles → int8, −8.
                                let v0 = vld1q_u8(wb0.qs.as_ptr());
                                let v1 = vld1q_u8(wb1.qs.as_ptr());
                                let w0l =
                                    vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, mask_lo)), offset_8);
                                let w0h =
                                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), offset_8);
                                let w1l =
                                    vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, mask_lo)), offset_8);
                                let w1h =
                                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), offset_8);
                                // 8-lane chunks in element order: lo.lo, lo.hi, hi.lo, hi.hi
                                // (elements 0..8, 8..16, 16..24, 24..32).
                                let a_row0 = [
                                    vget_low_s8(w0l),
                                    vget_high_s8(w0l),
                                    vget_low_s8(w0h),
                                    vget_high_s8(w0h),
                                ];
                                let a_row1 = [
                                    vget_low_s8(w1l),
                                    vget_high_s8(w1l),
                                    vget_low_s8(w1h),
                                    vget_high_s8(w1h),
                                ];
                                let mut acc = vdupq_n_s32(0);
                                for c in 0..4 {
                                    let off = c * 8;
                                    // a: row0 = weight i, row1 = weight i+1 (8 deep).
                                    let av = vcombine_s8(a_row0[c], a_row1[c]);
                                    // b: row0 = input col j, row1 = input col j+1.
                                    let bv = vcombine_s8(
                                        vld1_s8(b_quants.as_ptr().add(j * k + bi * 32 + off)),
                                        vld1_s8(b_quants.as_ptr().add((j + 1) * k + bi * 32 + off)),
                                    );
                                    acc = vmmlaq_s32(acc, av, bv);
                                }
                                acc
                            };
                            // Lanes: [W_i·B_j, W_i·B_{j+1}, W_{i+1}·B_j, W_{i+1}·B_{j+1}].
                            let (d00, d01, d10, d11) = unsafe {
                                (
                                    vgetq_lane_s32::<0>(acc) as f32,
                                    vgetq_lane_s32::<1>(acc) as f32,
                                    vgetq_lane_s32::<2>(acc) as f32,
                                    vgetq_lane_s32::<3>(acc) as f32,
                                )
                            };
                            s00 += d00 * dw0 * db0;
                            s01 += d01 * dw0 * db1;
                            s10 += d10 * dw1 * db0;
                            s11 += d11 * dw1 * db1;
                        }
                        strip[j] = s00;
                        strip[j + 1] = s01;
                        strip[n + j] = s10;
                        strip[n + j + 1] = s11;
                    }
                    // Odd last column within this strip.
                    if n_even < n {
                        let j = n - 1;
                        strip[j] = gemm_q4_0_scalar_dot(
                            a_quant, b_scales, b_quants, i, j, nb, k, row_bytes,
                        );
                        strip[n + j] = gemm_q4_0_scalar_dot(
                            a_quant,
                            b_scales,
                            b_quants,
                            i + 1,
                            j,
                            nb,
                            k,
                            row_bytes,
                        );
                    }
                });
        }

        // Odd last row (covers all columns).
        if m_even < m {
            let i = m - 1;
            for j in 0..n {
                out[i * n + j] =
                    gemm_q4_0_scalar_dot(a_quant, b_scales, b_quants, i, j, nb, k, row_bytes);
            }
        }
    }

    /// Q4_0 × Q8_0 GEMM dispatcher. Prefers i8mm (`vmmlaq_s32`) when the tier is
    /// resolved to it, else the dotprod GEMM, else the emulated-integer base.
    // Only non-test consumer is `transformer::gemm_preq`, gated
    // `not(feature = "blas")`; dead under --all-features (blas on), live under
    // the default CI gate.
    #[allow(dead_code)]
    pub unsafe fn gemm_q4_0_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        match cpu_features().tier {
            CpuTier::NeonI8mm => unsafe {
                gemm_q4_0_q8_0_neon_i8mm(a_quant, b_scales, b_quants, out, m, n, k)
            },
            CpuTier::NeonDotprod => unsafe {
                gemm_q4_0_q8_0_neon_dotprod(a_quant, b_scales, b_quants, out, m, n, k)
            },
            _ => unsafe { gemm_q4_0_q8_0_neon_base(a_quant, b_scales, b_quants, out, m, n, k) },
        }
    }

    /// Is the K-quant (Q4_K/Q6_K) int8 GEMM usable on this CPU?
    ///
    /// Unlike Q4_0/Q8_0, the K-quant GEMMs have **no baseline-NEON fallback** — they
    /// exist only in `dotprod` form. Callers must consult this *before* gating a
    /// weight onto the batched path: if the gate admits a dtype the kernel then
    /// declines, `gemm_preq` returns `false` and the matmul is **silently skipped**
    /// (wrong output, not merely slow). Every ARMv8.2+ core ships FEAT_DotProd, so
    /// the `false` arm is for genuinely ancient hardware, which simply keeps the
    /// per-token path.
    pub fn k_quant_gemm_available() -> bool {
        cpu_features().tier >= CpuTier::NeonDotprod
    }

    /// Q4_K × Q8_0 GEMM dispatcher. Requires `dotprod`; see
    /// [`k_quant_gemm_available`]. Returns `false` without writing `out` when this
    /// CPU or this `k` cannot run it, so the caller can fall back rather than ship
    /// a wrong answer.
    ///
    /// The `k % 256` check is here, not only in the caller's gate: superblocks are
    /// 256 wide, so a `k` that is not a multiple of 256 would make `sb = k / 256`
    /// silently drop the tail of every dot product — a truncated matmul, in release,
    /// with no assert. A guard that lives only in the gate is a guard the next caller
    /// can walk past.
    #[allow(dead_code)]
    pub unsafe fn gemm_q4_k_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        if !k_quant_gemm_available() || k % 256 != 0 {
            return false;
        }
        unsafe { gemm_q4_k_q8_0_neon_dotprod(a_quant, b_scales, b_quants, out, m, n, k) };
        true
    }

    /// Q8_0 × Q8_0 GEMM dispatcher. Prefers i8mm (`vmmlaq_s32`) when the tier is
    /// resolved to it, else dotprod, else the emulated-integer base.
    // Only non-test consumer is `transformer::gemm_preq`, gated
    // `not(feature = "blas")`; dead under --all-features (blas on), live under
    // the default CI gate.
    #[allow(dead_code)]
    pub unsafe fn gemm_q8_0_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        match cpu_features().tier {
            CpuTier::NeonI8mm => unsafe {
                gemm_q8_0_q8_0_neon_i8mm(a_quant, b_scales, b_quants, out, m, n, k)
            },
            CpuTier::NeonDotprod => unsafe {
                gemm_q8_0_q8_0_neon_dotprod(a_quant, b_scales, b_quants, out, m, n, k)
            },
            _ => unsafe { gemm_q8_0_q8_0_neon_base(a_quant, b_scales, b_quants, out, m, n, k) },
        }
    }

    // Verify each NEON-without-dotprod fallback against its `*_dotprod` sibling.
    // These only run on dotprod-capable hardware (e.g. Apple Silicon) — where
    // both paths are valid to call — and assert they agree to f32 tolerance.
    // Both consume the same q8_0-quantized input; the Q4_0/Q8_0 `_base` kernels
    // use the bit-identical emulated `vdotq_s32`, so they differ from the
    // dotprod path only in f32 scale-accumulation order, while the Q6_K path
    // reconstructs to f32 and reuses `vec_dot_q6_k_f32`.
    #[cfg(test)]
    mod fallback_tests {
        use super::*;

        /// Deterministic LCG → f32 in [-1, 1). Avoids `rand` and is stable.
        fn lcg(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) - 1.0
        }

        fn blocks_to_bytes<T: Copy>(blocks: &[T]) -> Vec<u8> {
            unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    std::mem::size_of_val(blocks),
                )
                .to_vec()
            }
        }

        fn quantize_col(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
            let nb = x.len() / 32;
            let mut s = vec![0.0f32; nb];
            let mut q = vec![0i8; x.len()];
            unsafe { quantize_f32_to_q8_0_neon(x, &mut s, &mut q) };
            (s, q)
        }

        fn assert_close(a: &[f32], b: &[f32]) {
            assert_eq!(a.len(), b.len());
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-2 * (1.0 + x.abs()),
                    "row {i}: dotprod={x} fallback={y}"
                );
            }
        }

        /// Tier-test gate. Returns true if the test should run. Normally skips
        /// (returns false) when the host lacks `feature`; but if the
        /// `CERA_REQUIRE_SIMD` env var lists `feature`, a missing feature is a
        /// hard failure — so a dedicated CI job on known-capable hardware proves
        /// the kernel actually executed rather than silently skipping.
        fn require_simd_or_skip(feature: &str, detected: bool) -> bool {
            if detected {
                return true;
            }
            let required = std::env::var("CERA_REQUIRE_SIMD").unwrap_or_default();
            assert!(
                !required.split(',').any(|f| f.trim() == feature),
                "CERA_REQUIRE_SIMD requires `{feature}` but this host doesn't report it"
            );
            false
        }

        #[test]
        fn q4_0_gemv_fallback_matches_dotprod() {
            if !cpu_features().dotprod {
                return;
            }
            let (m, k, nb) = (6usize, 128usize, 4usize);
            let mut st = 0x1234_5678u64;
            let blocks: Vec<BlockQ4_0> = (0..m * nb)
                .map(|_| {
                    let mut qs = [0u8; 16];
                    for b in qs.iter_mut() {
                        *b = (lcg(&mut st) * 127.0) as i32 as u8;
                    }
                    BlockQ4_0 {
                        d: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                        qs,
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let (xs, xq) = quantize_col(&x);

            let mut y_dot = vec![0.0f32; m];
            unsafe { gemv_q4_0_q8_0_neon_dotprod(&a, &xs, &xq, &mut y_dot, m, k) };
            let mut y_fb = vec![0.0f32; m];
            unsafe { gemv_q4_0_q8_0_neon_base(&a, &xs, &xq, &mut y_fb, m, k) };
            assert_close(&y_dot, &y_fb);
        }

        #[test]
        fn q8_0_gemv_fallback_matches_dotprod() {
            if !cpu_features().dotprod {
                return;
            }
            let (m, k, nb) = (6usize, 128usize, 4usize);
            let mut st = 0x9e37_79b9u64;
            let blocks: Vec<BlockQ8_0> = (0..m * nb)
                .map(|_| {
                    let mut quants = [0i8; 32];
                    for q in quants.iter_mut() {
                        *q = (lcg(&mut st) * 127.0) as i32 as i8;
                    }
                    BlockQ8_0 {
                        delta: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                        quants,
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let (xs, xq) = quantize_col(&x);

            let mut y_dot = vec![0.0f32; m];
            unsafe { gemv_q8_0_q8_0_neon_dotprod(&a, &xs, &xq, &mut y_dot, m, k) };
            let mut y_fb = vec![0.0f32; m];
            unsafe { gemv_q8_0_q8_0_neon_base(&a, &xs, &xq, &mut y_fb, m, k) };
            assert_close(&y_dot, &y_fb);
        }

        #[test]
        fn q6k_gemv_fallback_matches_dotprod() {
            if !cpu_features().dotprod {
                return;
            }
            let (m, k, nb) = (5usize, 256usize, 1usize); // one Q6_K super-block / row
            let mut st = 0xdead_beefu64;
            let blocks: Vec<BlockQ6K> = (0..m * nb)
                .map(|_| {
                    let mut ql = [0u8; 128];
                    let mut qh = [0u8; 64];
                    let mut scales = [0i8; 16];
                    for b in ql.iter_mut() {
                        *b = (lcg(&mut st) * 127.0) as i32 as u8;
                    }
                    for b in qh.iter_mut() {
                        *b = (lcg(&mut st) * 127.0) as i32 as u8;
                    }
                    for s in scales.iter_mut() {
                        *s = (lcg(&mut st) * 16.0) as i32 as i8;
                    }
                    BlockQ6K {
                        ql,
                        qh,
                        scales,
                        d: f16::from_f32(0.02 + lcg(&mut st).abs() * 0.05).to_bits(),
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let (xs, xq) = quantize_col(&x);

            let mut y_dot = vec![0.0f32; m];
            unsafe { gemv_q6k_q8_0_neon_dotprod(&a, &xs, &xq, &mut y_dot, m, k) };
            let xf = reconstruct_q8_0_input(&xs, &xq, k);
            let mut y_fb = vec![0.0f32; m];
            gemv_q6k_fallback(&a, &xf, &mut y_fb, k);
            assert_close(&y_dot, &y_fb);
        }

        fn random_q4km(st: &mut u64) -> BlockQ4KM {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for b in scales.iter_mut() {
                *b = (lcg(st).abs() * 255.0) as i32 as u8;
            }
            for b in qs.iter_mut() {
                *b = (lcg(st).abs() * 255.0) as i32 as u8;
            }
            BlockQ4KM {
                d: f16::from_f32(0.02 + lcg(st).abs() * 0.05).to_bits(),
                dmin: f16::from_f32(0.01 + lcg(st).abs() * 0.03).to_bits(),
                scales,
                qs,
            }
        }

        /// The Q4_K GEMM must agree with the (already-validated) Q4_K GEMV run
        /// column-by-column on the *same* pre-quantized inputs.
        ///
        /// This is the strong oracle: both consume identical Q8_0 activations, so
        /// the only legitimate difference is float summation order — no
        /// quantization error to hide a real bug behind. Comparing against the f32
        /// scalar instead would need a tolerance wide enough to swallow a dropped
        /// sub-block.
        ///
        /// `n = 11` deliberately straddles `KQ_COLS` (8): one full 8-column pass
        /// plus a 3-column tail, so a bug in the tail cannot hide.
        #[test]
        fn q4k_gemm_matches_gemv_per_column() {
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            let (m, n, k) = (7usize, 11usize, 512usize);
            let nb = k / 256;
            let mut st = 0xfeed_beefu64;
            let blocks: Vec<BlockQ4KM> = (0..m * nb).map(|_| random_q4km(&mut st)).collect();
            let a = blocks_to_bytes(&blocks);

            // Column-major activations: column j is b[j*k .. (j+1)*k].
            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut st)).collect();
            let mut b_scales = vec![0.0f32; n * (k / 32)];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let (s, q) = quantize_col(&b[j * k..(j + 1) * k]);
                b_scales[j * (k / 32)..(j + 1) * (k / 32)].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out = vec![0.0f32; m * n];
            unsafe {
                gemm_q4_k_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out, m, n, k);
            }

            // Oracle: the GEMV, once per column, on the identical quantized inputs.
            for j in 0..n {
                let mut y = vec![0.0f32; m];
                unsafe {
                    gemv_q4k_q8_0_neon_dotprod(
                        &a,
                        &b_scales[j * (k / 32)..(j + 1) * (k / 32)],
                        &b_quants[j * k..(j + 1) * k],
                        &mut y,
                        m,
                        k,
                    );
                }
                for (i, &want) in y.iter().enumerate() {
                    let got = out[i * n + j];
                    // Tight: both paths run the *same* int8 arithmetic on the same
                    // Q8_0 activations, so only float summation order may differ. A
                    // loose bound here (1e-3) is wide enough to hide a real kernel
                    // bug — which is exactly what it did on the first cut of this.
                    assert!(
                        (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                        "col {j} row {i}: gemm={got} gemv={want}"
                    );
                }
            }
        }

        /// The Q6_K GEMM must agree with the existing Q6_K GEMV run column-by-column
        /// on the *same* pre-quantized inputs.
        ///
        /// The GEMV (`gemv_q6k_q8_0_neon_dotprod`) is the right oracle precisely
        /// because it is what the per-token path runs: both consume identical Q8_0
        /// activations and the same int8 dot, so agreement must be near-exact and any
        /// gap is a real bug rather than quantization noise. It is *not* independent of
        /// the kernel's index math — a shared misunderstanding of the 6-bit layout would
        /// pass — so the absolute correctness of that layout rests on the GEMV's own
        /// chain down to `vec_dot_q6_k_f32` / `dequantize_q6_k_block`, which is already
        /// covered. What this pins down is the thing that actually broke: that the
        /// batched form reproduces the per-token form exactly.
        ///
        /// (The first cut of this test used a dequantized-f32 reference at 1e-3 on the
        /// mistaken belief that no Q6_K int8 GEMV existed. It does. That tolerance was
        /// wide enough to pass a kernel the model-level parity test then caught.)
        ///
        /// `n = 11` straddles `KQ_COLS` (8) so the column tail is exercised.
        #[test]
        fn q6k_gemm_matches_gemv_per_column() {
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            let (m, n, k) = (5usize, 11usize, 512usize);
            let nb = k / 256;
            let mut st = 0x0bad_c0deu64;

            let blocks: Vec<crate::quant::BlockQ6K> = (0..m * nb)
                .map(|_| {
                    let mut ql = [0u8; 128];
                    let mut qh = [0u8; 64];
                    let mut scales = [0i8; 16];
                    for b in ql.iter_mut() {
                        *b = (lcg(&mut st).abs() * 255.0) as i32 as u8;
                    }
                    for b in qh.iter_mut() {
                        *b = (lcg(&mut st).abs() * 255.0) as i32 as u8;
                    }
                    for s in scales.iter_mut() {
                        *s = (lcg(&mut st) * 60.0) as i32 as i8;
                    }
                    crate::quant::BlockQ6K {
                        ql,
                        qh,
                        scales,
                        d: half::f16::from_f32(0.02 + lcg(&mut st).abs() * 0.05).to_bits(),
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);

            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut st)).collect();
            let mut b_scales = vec![0.0f32; n * (k / 32)];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let (s, q) = quantize_col(&b[j * k..(j + 1) * k]);
                b_scales[j * (k / 32)..(j + 1) * (k / 32)].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out = vec![0.0f32; m * n];
            unsafe {
                gemm_q6_k_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out, m, n, k);
            }

            // Oracle: the existing Q6_K GEMV, once per column, on the identical
            // quantized inputs — the same int8 arithmetic, so agreement must be
            // near-exact. (The first cut of this test used a dequantized-f32
            // reference at 1e-3, which was slack enough to pass a kernel that the
            // model-level parity test then caught. Use the tightest oracle available.)
            for j in 0..n {
                let mut y = vec![0.0f32; m];
                unsafe {
                    gemv_q6k_q8_0_neon_dotprod(
                        &a,
                        &b_scales[j * (k / 32)..(j + 1) * (k / 32)],
                        &b_quants[j * k..(j + 1) * k],
                        &mut y,
                        m,
                        k,
                    );
                }
                for (i, &want) in y.iter().enumerate() {
                    let got = out[i * n + j];
                    assert!(
                        (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                        "col {j} row {i}: gemm={got} gemv={want}"
                    );
                }
            }
        }

        /// The Q4_K GEMM must be **bit-exact** against the GEMV at *real model
        /// shapes*, not merely close.
        ///
        /// The small-shape parity tests above cannot see accumulation-order bugs:
        /// rounding error grows with `k`, and LFM2.5-350M's `ffn_down` has k=4608
        /// (18 superblocks) against the 512 a unit test reaches for. A Q6_K sibling
        /// of this test is what caught a real ordering bug worth 3.4e-4 at k=4608 —
        /// invisible at k=512, but enough to move the model's logits.
        #[test]
        fn q4k_gemm_bit_exact_vs_gemv_at_model_shapes() {
            // `require_simd_or_skip`, not a bare `return`: on a host without dotprod a
            // bare return reports a green test that asserted nothing, and this is the
            // test that catches accumulation-order bugs. `CERA_REQUIRE_SIMD=dotprod`
            // turns that skip into a hard failure on CI hardware that should have it.
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            for &(m, n, k) in &[(4608usize, 64usize, 1024usize), (1024, 64, 4608)] {
                let nb = k / 256;
                let mut st = 0x1234_9999u64;
                let blocks: Vec<BlockQ4KM> = (0..m * nb).map(|_| random_q4km(&mut st)).collect();
                let a = blocks_to_bytes(&blocks);
                let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut st)).collect();
                let mut bs = vec![0.0f32; n * (k / 32)];
                let mut bq = vec![0i8; n * k];
                for j in 0..n {
                    let (s2, q2) = quantize_col(&b[j * k..(j + 1) * k]);
                    bs[j * (k / 32)..(j + 1) * (k / 32)].copy_from_slice(&s2);
                    bq[j * k..(j + 1) * k].copy_from_slice(&q2);
                }
                let mut out = vec![0.0f32; m * n];
                unsafe { gemm_q4_k_q8_0_neon_dotprod(&a, &bs, &bq, &mut out, m, n, k) };
                let mut worst = 0.0f32;
                for j in 0..n {
                    let mut y = vec![0.0f32; m];
                    unsafe {
                        gemv_q4k_q8_0_neon_dotprod(
                            &a,
                            &bs[j * (k / 32)..(j + 1) * (k / 32)],
                            &bq[j * k..(j + 1) * k],
                            &mut y,
                            m,
                            k,
                        );
                    }
                    for (i, &want) in y.iter().enumerate() {
                        let got = out[i * n + j];
                        let rel = (got - want).abs() / (1.0 + want.abs());
                        if rel > worst {
                            worst = rel;
                        }
                    }
                }
                assert_eq!(
                    worst, 0.0,
                    "Q4_K GEMM m={m} n={n} k={k}: not bit-exact vs the GEMV \
                     (max_rel={worst:e}) — the batched and per-token paths must run \
                     the same arithmetic in the same order"
                );
            }
        }

        /// The Q6_K GEMM must be **bit-exact** against the GEMV at real model shapes.
        ///
        /// This is the test that caught the bug: summing a group's two 16-wide halves
        /// together before accumulating (rather than in the GEMV's half-outer order,
        /// with `d·sc·xs` formed before the dot) drifts to 3.4e-4 relative at
        /// `ffn_down`'s k=4608, because Q6_K sums terms that nearly cancel. At k=512
        /// the same bug reads as 5.5e-6 and slips through any reasonable tolerance.
        #[test]
        fn q6k_gemm_bit_exact_vs_gemv_at_model_shapes() {
            // See the Q4_K twin: a bare `return` here would vacuously pass the very
            // test that caught the Q6_K accumulation-order bug.
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            for &(m, n, k) in &[(1024usize, 64usize, 4608usize), (5, 11, 512)] {
                let nb = k / 256;
                let mut st = 0x5150_7777u64;
                let blocks: Vec<crate::quant::BlockQ6K> = (0..m * nb)
                    .map(|_| {
                        let mut ql = [0u8; 128];
                        let mut qh = [0u8; 64];
                        let mut scales = [0i8; 16];
                        for b in ql.iter_mut() {
                            *b = (lcg(&mut st).abs() * 255.0) as i32 as u8;
                        }
                        for b in qh.iter_mut() {
                            *b = (lcg(&mut st).abs() * 255.0) as i32 as u8;
                        }
                        for sx in scales.iter_mut() {
                            *sx = (lcg(&mut st) * 60.0) as i32 as i8;
                        }
                        crate::quant::BlockQ6K {
                            ql,
                            qh,
                            scales,
                            d: half::f16::from_f32(0.02 + lcg(&mut st).abs() * 0.05).to_bits(),
                        }
                    })
                    .collect();
                let a = blocks_to_bytes(&blocks);
                let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut st)).collect();
                let mut bs = vec![0.0f32; n * (k / 32)];
                let mut bq = vec![0i8; n * k];
                for j in 0..n {
                    let (s2, q2) = quantize_col(&b[j * k..(j + 1) * k]);
                    bs[j * (k / 32)..(j + 1) * (k / 32)].copy_from_slice(&s2);
                    bq[j * k..(j + 1) * k].copy_from_slice(&q2);
                }
                let mut out = vec![0.0f32; m * n];
                unsafe { gemm_q6_k_q8_0_neon_dotprod(&a, &bs, &bq, &mut out, m, n, k) };
                let mut worst = 0.0f32;
                let mut worst_pair = (0.0f32, 0.0f32);
                for j in 0..n {
                    let mut y = vec![0.0f32; m];
                    unsafe {
                        gemv_q6k_q8_0_neon_dotprod(
                            &a,
                            &bs[j * (k / 32)..(j + 1) * (k / 32)],
                            &bq[j * k..(j + 1) * k],
                            &mut y,
                            m,
                            k,
                        );
                    }
                    for (i, &want) in y.iter().enumerate() {
                        let got = out[i * n + j];
                        let rel = (got - want).abs() / (1.0 + want.abs());
                        if rel > worst {
                            worst = rel;
                            worst_pair = (got, want);
                        }
                    }
                }
                assert_eq!(
                    worst, 0.0,
                    "Q6_K GEMM m={m} n={n} k={k}: not bit-exact vs the GEMV \
                     (max_rel={worst:e}, gemm={} gemv={}) — accumulation order must \
                     match `gemv_q6k_q8_0_neon_dotprod` exactly",
                    worst_pair.0, worst_pair.1
                );
            }
        }

        #[test]
        fn q4k_gemv_dotprod_matches_scalar() {
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            // 2 super-blocks/row (k=512) × 5 rows: exercises multi-block
            // accumulation and the per-sub-block scale/min indexing.
            let (m, k, nb) = (5usize, 512usize, 2usize);
            let mut st = 0xcafe_f00du64;
            let blocks: Vec<BlockQ4KM> = (0..m * nb).map(|_| random_q4km(&mut st)).collect();
            let a = blocks_to_bytes(&blocks);
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let (xs, xq) = quantize_col(&x);

            let mut y_dot = vec![0.0f32; m];
            unsafe { gemv_q4k_q8_0_neon_dotprod(&a, &xs, &xq, &mut y_dot, m, k) };
            // The scalar path consumes the SAME quantized activations
            // (reconstructed to f32), so the integer-dot and scalar/f32 paths
            // agree up to fp accumulation order.
            let xf = reconstruct_q8_0_input(&xs, &xq, k);
            let mut y_fb = vec![0.0f32; m];
            crate::backend::cpu::gemv_q4km_f32(&a, &xf, &mut y_fb, m, k);
            assert_close(&y_dot, &y_fb);
        }

        #[test]
        fn q4k_gemv_wrapper_matches_exact_f32() {
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            // End-to-end: the wrapper quantizes x to Q8_0 internally, so compare
            // against the exact-f32 reference with a tolerance that admits the
            // ~1% relative error of Q8_0 activation quantization.
            let (m, k, nb) = (4usize, 256usize, 1usize);
            let mut st = 0x00c0_ffeeu64;
            let blocks: Vec<BlockQ4KM> = (0..m * nb).map(|_| random_q4km(&mut st)).collect();
            let a = blocks_to_bytes(&blocks);
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();

            let mut s = Vec::new();
            let mut q = Vec::new();
            let mut y_dot = vec![0.0f32; m];
            unsafe { gemv_q4k_f32_neon(&a, &x, &mut y_dot, m, k, &mut s, &mut q) };

            let mut y_exact = vec![0.0f32; m];
            crate::backend::cpu::gemv_q4km_f32(&a, &x, &mut y_exact, m, k);

            for (i, (&yd, &ye)) in y_dot.iter().zip(&y_exact).enumerate() {
                assert!(
                    (yd - ye).abs() <= 3e-2 * (1.0 + ye.abs()),
                    "row {i}: dotprod={yd} exact={ye}"
                );
            }
        }

        #[test]
        fn q4_0_gemm_fallback_matches_dotprod() {
            if !cpu_features().dotprod {
                return;
            }
            let (m, n, k, nb) = (4usize, 3usize, 96usize, 3usize);
            let mut st = 0x0bad_f00du64;
            let blocks: Vec<BlockQ4_0> = (0..m * nb)
                .map(|_| {
                    let mut qs = [0u8; 16];
                    for b in qs.iter_mut() {
                        *b = (lcg(&mut st) * 127.0) as i32 as u8;
                    }
                    BlockQ4_0 {
                        d: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                        qs,
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);
            // B: n columns of length k, quantized to Q8_0 column-major.
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
                let (s, q) = quantize_col(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out_dot = vec![0.0f32; m * n];
            unsafe { gemm_q4_0_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out_dot, m, n, k) };
            let mut out_fb = vec![0.0f32; m * n];
            unsafe { gemm_q4_0_q8_0_neon_base(&a, &b_scales, &b_quants, &mut out_fb, m, n, k) };
            assert_close(&out_dot, &out_fb);
        }

        #[test]
        fn q8_0_gemm_fallback_matches_dotprod() {
            if !cpu_features().dotprod {
                return;
            }
            let (m, n, k, nb) = (4usize, 3usize, 96usize, 3usize);
            let mut st = 0xcafe_d00du64;
            let blocks: Vec<BlockQ8_0> = (0..m * nb)
                .map(|_| {
                    let mut quants = [0i8; 32];
                    for q in quants.iter_mut() {
                        *q = (lcg(&mut st) * 127.0) as i32 as i8;
                    }
                    BlockQ8_0 {
                        delta: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                        quants,
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
                let (s, q) = quantize_col(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out_dot = vec![0.0f32; m * n];
            unsafe { gemm_q8_0_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out_dot, m, n, k) };
            let mut out_fb = vec![0.0f32; m * n];
            unsafe { gemm_q8_0_q8_0_neon_base(&a, &b_scales, &b_quants, &mut out_fb, m, n, k) };
            assert_close(&out_dot, &out_fb);
        }

        /// i8mm Q8_0 GEMM vs the dotprod kernel. Skips on the dev host (M1 has no
        /// i8mm); runs where `is_aarch64_feature_detected!("i8mm")` (ARMv8.6) —
        /// notably the `simd-i8mm` CI job, which enforces it via
        /// `CERA_REQUIRE_SIMD=i8mm`. Covers odd m and n for the remainder paths.
        #[test]
        fn i8mm_gemm_matches_dotprod() {
            if !require_simd_or_skip("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")) {
                return;
            }
            for &(m, n, k) in &[(4usize, 4usize, 64usize), (5, 3, 96), (2, 7, 64)] {
                let nb = k / 32;
                let mut st = 0x5151_2323u64 ^ ((m * 131 + n * 17 + k) as u64);
                let blocks: Vec<BlockQ8_0> = (0..m * nb)
                    .map(|_| {
                        let mut quants = [0i8; 32];
                        for q in quants.iter_mut() {
                            *q = (lcg(&mut st) * 127.0) as i32 as i8;
                        }
                        BlockQ8_0 {
                            delta: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                            quants,
                        }
                    })
                    .collect();
                let a = blocks_to_bytes(&blocks);
                let mut b_scales = vec![0.0f32; n * nb];
                let mut b_quants = vec![0i8; n * k];
                for j in 0..n {
                    let col: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
                    let (s, q) = quantize_col(&col);
                    b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                    b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
                }

                let mut out_dot = vec![0.0f32; m * n];
                unsafe {
                    gemm_q8_0_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out_dot, m, n, k)
                };
                let mut out_i8mm = vec![0.0f32; m * n];
                unsafe {
                    gemm_q8_0_q8_0_neon_i8mm(&a, &b_scales, &b_quants, &mut out_i8mm, m, n, k)
                };
                assert_close(&out_dot, &out_i8mm);
            }
        }

        /// i8mm Q4_0 GEMM vs the dotprod kernel. Same gating as
        /// [`i8mm_gemm_matches_dotprod`] — skips on the dev host (M1 has no
        /// i8mm), runs (and is enforced) on the `simd-i8mm` CI job. Covers odd m
        /// and n for the scalar-remainder paths.
        #[test]
        fn q4_0_gemm_i8mm_matches_dotprod() {
            if !require_simd_or_skip("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")) {
                return;
            }
            for &(m, n, k) in &[(4usize, 4usize, 64usize), (5, 3, 96), (2, 7, 64)] {
                let nb = k / 32;
                let mut st = 0x9e37_79b9u64 ^ ((m * 131 + n * 17 + k) as u64);
                let blocks: Vec<BlockQ4_0> = (0..m * nb)
                    .map(|_| {
                        let mut qs = [0u8; 16];
                        for b in qs.iter_mut() {
                            *b = (lcg(&mut st) * 127.0) as i32 as u8;
                        }
                        BlockQ4_0 {
                            d: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                            qs,
                        }
                    })
                    .collect();
                let a = blocks_to_bytes(&blocks);
                let mut b_scales = vec![0.0f32; n * nb];
                let mut b_quants = vec![0i8; n * k];
                for j in 0..n {
                    let col: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
                    let (s, q) = quantize_col(&col);
                    b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                    b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
                }

                let mut out_dot = vec![0.0f32; m * n];
                unsafe {
                    gemm_q4_0_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out_dot, m, n, k)
                };
                let mut out_i8mm = vec![0.0f32; m * n];
                unsafe {
                    gemm_q4_0_q8_0_neon_i8mm(&a, &b_scales, &b_quants, &mut out_i8mm, m, n, k)
                };
                assert_close(&out_dot, &out_i8mm);
            }
        }
    }
}

// ── x86_64 AVX2 ─────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2-optimized Q8_0 dot product with f32 vector.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q8_0_f32_avx2(block: &BlockQ8_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.delta).to_f32();

            let mut sum256 = _mm256_setzero_ps();
            let quants_ptr = block.quants.as_ptr();
            let y_ptr = y.as_ptr();

            for i in (0..32).step_by(8) {
                let q = [
                    *quants_ptr.add(i) as i32,
                    *quants_ptr.add(i + 1) as i32,
                    *quants_ptr.add(i + 2) as i32,
                    *quants_ptr.add(i + 3) as i32,
                    *quants_ptr.add(i + 4) as i32,
                    *quants_ptr.add(i + 5) as i32,
                    *quants_ptr.add(i + 6) as i32,
                    *quants_ptr.add(i + 7) as i32,
                ];
                let qi32 = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
                let qf32 = _mm256_cvtepi32_ps(qi32);
                let yv = _mm256_loadu_ps(y_ptr.add(i));
                sum256 = _mm256_fmadd_ps(qf32, yv, sum256);
            }

            d * hsum_avx(sum256)
        }
    }

    /// AVX2-optimized Q4_0 dot product with f32 vector.
    ///
    /// Loads all 16 qs bytes at once, extracts nibbles with vector AND/SHIFT,
    /// then widens to i32 and converts to f32 for FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q4_0_f32_avx2(block: &BlockQ4_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.d).to_f32();
            let offset = _mm256_set1_ps(8.0);
            let mask_lo = _mm_set1_epi8(0x0F);

            let mut sum256 = _mm256_setzero_ps();
            let y_ptr = y.as_ptr();

            // Load all 16 bytes of qs
            let qbytes = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);

            // Extract low nibbles (AND with 0x0F) and high nibbles (shift right 4)
            let lo_bytes = _mm_and_si128(qbytes, mask_lo);
            let hi_bytes = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_lo);

            // Process low nibbles: 16 u8 values → 2 groups of 8 i32 → f32
            // First 8 low nibbles
            let lo_0_i32 = _mm256_cvtepu8_epi32(lo_bytes); // lower 8 bytes → 8 i32
            let lo_0_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo_0_i32), offset);
            sum256 = _mm256_fmadd_ps(lo_0_f32, _mm256_loadu_ps(y_ptr), sum256);

            // Next 8 low nibbles
            let lo_hi_half = _mm_srli_si128(lo_bytes, 8); // shift right 8 bytes
            let lo_1_i32 = _mm256_cvtepu8_epi32(lo_hi_half);
            let lo_1_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo_1_i32), offset);
            sum256 = _mm256_fmadd_ps(lo_1_f32, _mm256_loadu_ps(y_ptr.add(8)), sum256);

            // Process high nibbles: same pattern, y offset by 16
            let hi_0_i32 = _mm256_cvtepu8_epi32(hi_bytes);
            let hi_0_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi_0_i32), offset);
            sum256 = _mm256_fmadd_ps(hi_0_f32, _mm256_loadu_ps(y_ptr.add(16)), sum256);

            let hi_hi_half = _mm_srli_si128(hi_bytes, 8);
            let hi_1_i32 = _mm256_cvtepu8_epi32(hi_hi_half);
            let hi_1_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi_1_i32), offset);
            sum256 = _mm256_fmadd_ps(hi_1_f32, _mm256_loadu_ps(y_ptr.add(24)), sum256);

            d * hsum_avx(sum256)
        }
    }

    /// AVX2-optimized Q4_K_M dot product with f32 vector.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q4_k_m_f32_avx2(block: &BlockQ4KM, y: &[f32]) -> f32 {
        unsafe {
            let d = f16::from_bits(block.d).to_f32();
            let dmin = f16::from_bits(block.dmin).to_f32();

            let scales = &block.scales;
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..4 {
                sc[j] = scales[j] & 63;
                mn[j] = scales[j + 4] & 63;
            }
            for j in 4..8 {
                sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }

            let qs = &block.qs;
            let y_ptr = y.as_ptr();
            let mut sumf = 0.0f32;
            let mut qi = 0usize;
            let mut yi = 0usize;

            for j in 0..4 {
                let sc1 = d * sc[j * 2] as f32;
                let mn1 = dmin * mn[j * 2] as f32;
                let sc2 = d * sc[j * 2 + 1] as f32;
                let mn2 = dmin * mn[j * 2 + 1] as f32;

                let mut sum1_acc = _mm256_setzero_ps();
                let mut sum2_acc = _mm256_setzero_ps();
                let mut mn1_acc = _mm256_setzero_ps();
                let mut mn2_acc = _mm256_setzero_ps();

                for l in (0..32).step_by(8) {
                    let mut lo_arr = [0i32; 8];
                    let mut hi_arr = [0i32; 8];
                    for k in 0..8 {
                        lo_arr[k] = (qs[qi + l + k] & 0xF) as i32;
                        hi_arr[k] = (qs[qi + l + k] >> 4) as i32;
                    }

                    let lo_f32 =
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(lo_arr.as_ptr() as *const __m256i));
                    let hi_f32 =
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(hi_arr.as_ptr() as *const __m256i));

                    let y1 = _mm256_loadu_ps(y_ptr.add(yi + l));
                    let y2 = _mm256_loadu_ps(y_ptr.add(yi + l + 32));

                    sum1_acc = _mm256_fmadd_ps(lo_f32, y1, sum1_acc);
                    sum2_acc = _mm256_fmadd_ps(hi_f32, y2, sum2_acc);
                    mn1_acc = _mm256_add_ps(mn1_acc, y1);
                    mn2_acc = _mm256_add_ps(mn2_acc, y2);
                }

                sumf += sc1 * hsum_avx(sum1_acc) + sc2 * hsum_avx(sum2_acc)
                    - mn1 * hsum_avx(mn1_acc)
                    - mn2 * hsum_avx(mn2_acc);
                qi += 32;
                yi += 64;
            }

            sumf
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// ── x86_64 AVX-512 ──────────────────────────────────────────────────────────
//
// 512-bit-wide f32 vec_dot kernels — the AVX2 algorithm at double the vector
// width (16 f32 lanes per op). The x86 hot path is int8-weight × f32-activation,
// so these stay on the f32-FMA path; a true VNNI int8×int8 GEMV would need a
// quantized-activation path on x86 (like aarch64's pre-quant kernels) and is a
// separate, larger change. Only Q8_0 and Q4_0 have AVX-512 kernels; Q4_K_M
// stays on AVX2 even at the Avx512 tier (the dispatcher routes it there).
//
// NOTE: not executable on the aarch64 dev host. Verified by `avx512_tests`
// below, which run only where `is_x86_feature_detected!("avx512f")` holds
// (e.g. Zen 4/5, Skylake-X), comparing each kernel against the scalar reference.
//
// Behind the default-on `avx512` crate feature: the `_mm512_*` intrinsics need
// Rust 1.89 (past the 1.85 MSRV), so disabling the feature keeps x86 building on
// 1.85–1.88 (the tier then caps at AVX2; `detect()` won't produce `Avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod avx512 {
    // 1.89 `_mm512_*` intrinsics vs the 1.85 MSRV — see the module gate above.
    // The `avx512` feature lets MSRV-sensitive builds opt out; when it's on, the
    // build already requires 1.89, so silence the (correct) lint here.
    #![allow(clippy::incompatible_msrv)]
    use super::*;
    use std::arch::x86_64::*;

    /// AVX-512 Q8_0 dot product with f32 vector. 16 lanes/iter, 2 iters for 32.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn vec_dot_q8_0_f32_avx512(block: &BlockQ8_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.delta).to_f32();
            let quants_ptr = block.quants.as_ptr();
            let y_ptr = y.as_ptr();
            let mut acc = _mm512_setzero_ps();
            for i in (0..32).step_by(16) {
                // 16 int8 → 16 i32 (sign-extend) → 16 f32.
                let q128 = _mm_loadu_si128(quants_ptr.add(i) as *const __m128i);
                let qf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q128));
                let yv = _mm512_loadu_ps(y_ptr.add(i));
                acc = _mm512_fmadd_ps(qf, yv, acc);
            }
            d * _mm512_reduce_add_ps(acc)
        }
    }

    /// AVX-512 Q4_0 dot product with f32 vector. Low 16 nibbles ↔ y[0..16],
    /// high 16 ↔ y[16..32]; each nibble is `(n - 8) * d`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn vec_dot_q4_0_f32_avx512(block: &BlockQ4_0, y: &[f32]) -> f32 {
        unsafe {
            debug_assert_eq!(y.len(), 32);
            let d = f16::from_bits(block.d).to_f32();
            let offset = _mm512_set1_ps(8.0);
            let mask_lo = _mm_set1_epi8(0x0F);
            let y_ptr = y.as_ptr();

            let qbytes = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);
            let lo = _mm_and_si128(qbytes, mask_lo);
            let hi = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_lo);

            let mut acc = _mm512_setzero_ps();
            // 16 low nibbles (zero-extend u8 → i32) → f32, minus 8, FMA y[0..16].
            let lo_f = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo)), offset);
            acc = _mm512_fmadd_ps(lo_f, _mm512_loadu_ps(y_ptr), acc);
            let hi_f = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi)), offset);
            acc = _mm512_fmadd_ps(hi_f, _mm512_loadu_ps(y_ptr.add(16)), acc);

            d * _mm512_reduce_add_ps(acc)
        }
    }

    // ── Row kernels ─────────────────────────────────────────────────────────
    //
    // The `vec_dot_*` pair above is a *per-block* API, so it has to finish with
    // `_mm512_reduce_add_ps` on every 32 elements — a ~5-op, long-latency
    // horizontal collapse in the innermost loop, serialized against the next
    // block's FMAs. These row kernels keep one vector accumulator across the
    // whole row and reduce exactly once, at the cost of scaling each block's
    // contribution by `d` in-vector (two extra `mul_ps` per block) instead of
    // once in scalar afterwards.
    //
    // The per-block entry points stay: `vec_dot_q4_0_f32` / `vec_dot_q8_0_f32`
    // are public API and are used where only a single block is in hand.

    /// Q4_0 row dot: `<dequant(row), y>` accumulated across `nb` blocks.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn row_dot_q4_0_f32_avx512(row: *const u8, y: &[f32], nb: usize) -> f32 {
        unsafe {
            // `row` is a raw pointer, so nothing here is bounds-checked: the
            // caller promises `row` covers `nb` blocks and `y` covers `nb * 32`
            // floats. Cheap to state, and a debug build turns a silent
            // out-of-bounds read into a named failure.
            debug_assert!(
                y.len() >= nb * 32,
                "row_dot: y has {} floats, need {}",
                y.len(),
                nb * 32
            );
            let bsz = size_of::<BlockQ4_0>();
            let offset = _mm512_set1_ps(8.0);
            let mask_lo = _mm_set1_epi8(0x0F);
            let mut acc = _mm512_setzero_ps();
            for b in 0..nb {
                let block = &*(row.add(b * bsz) as *const BlockQ4_0);
                let d = _mm512_set1_ps(f16::from_bits(block.d).to_f32());
                let qbytes = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i);
                let lo = _mm_and_si128(qbytes, mask_lo);
                let hi = _mm_and_si128(_mm_srli_epi16(qbytes, 4), mask_lo);
                let y_ptr = y.as_ptr().add(b * 32);

                let lo_f = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo)), offset);
                acc = _mm512_fmadd_ps(_mm512_mul_ps(d, lo_f), _mm512_loadu_ps(y_ptr), acc);
                let hi_f = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi)), offset);
                acc = _mm512_fmadd_ps(_mm512_mul_ps(d, hi_f), _mm512_loadu_ps(y_ptr.add(16)), acc);
            }
            _mm512_reduce_add_ps(acc)
        }
    }

    /// Q8_0 row dot: `<dequant(row), y>` accumulated across `nb` blocks.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn row_dot_q8_0_f32_avx512(row: *const u8, y: &[f32], nb: usize) -> f32 {
        unsafe {
            // `row` is a raw pointer, so nothing here is bounds-checked: the
            // caller promises `row` covers `nb` blocks and `y` covers `nb * 32`
            // floats. Cheap to state, and a debug build turns a silent
            // out-of-bounds read into a named failure.
            debug_assert!(
                y.len() >= nb * 32,
                "row_dot: y has {} floats, need {}",
                y.len(),
                nb * 32
            );
            let bsz = size_of::<BlockQ8_0>();
            let mut acc = _mm512_setzero_ps();
            for b in 0..nb {
                let block = &*(row.add(b * bsz) as *const BlockQ8_0);
                let d = _mm512_set1_ps(f16::from_bits(block.delta).to_f32());
                let quants_ptr = block.quants.as_ptr();
                let y_ptr = y.as_ptr().add(b * 32);
                for i in (0..32).step_by(16) {
                    let q128 = _mm_loadu_si128(quants_ptr.add(i) as *const __m128i);
                    let qf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q128));
                    acc = _mm512_fmadd_ps(_mm512_mul_ps(d, qf), _mm512_loadu_ps(y_ptr.add(i)), acc);
                }
            }
            _mm512_reduce_add_ps(acc)
        }
    }

    #[cfg(test)]
    mod avx512_tests {
        use super::*;
        use crate::quant::{vec_dot_q4_0_f32_scalar, vec_dot_q8_0_f32_scalar};

        fn lcg(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) - 1.0
        }

        /// Tier-test gate. Skips when the host lacks `feature`, unless
        /// `CERA_REQUIRE_SIMD` lists it — then a missing feature fails the test,
        /// so a CI job on AVX-512 hardware proves the kernel actually ran.
        fn require_simd_or_skip(feature: &str, detected: bool) -> bool {
            if detected {
                return true;
            }
            let required = std::env::var("CERA_REQUIRE_SIMD").unwrap_or_default();
            assert!(
                !required.split(',').any(|f| f.trim() == feature),
                "CERA_REQUIRE_SIMD requires `{feature}` but this host doesn't report it"
            );
            false
        }

        #[test]
        fn q8_0_avx512_matches_scalar() {
            // Only runs on AVX-512 hardware (e.g. the AMD Zen 5 box).
            if !require_simd_or_skip("avx512", is_x86_feature_detected!("avx512f")) {
                return;
            }
            let mut st = 0x1357_9bdfu64;
            let mut quants = [0i8; 32];
            for q in quants.iter_mut() {
                *q = (lcg(&mut st) * 127.0) as i32 as i8;
            }
            let block = BlockQ8_0 {
                delta: f16::from_f32(0.043).to_bits(),
                quants,
            };
            let y: Vec<f32> = (0..32).map(|_| lcg(&mut st)).collect();
            let got = unsafe { vec_dot_q8_0_f32_avx512(&block, &y) };
            let want = vec_dot_q8_0_f32_scalar(&block, &y);
            assert!(
                (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                "{got} vs {want}"
            );
        }

        #[test]
        fn q4_0_avx512_matches_scalar() {
            if !require_simd_or_skip("avx512", is_x86_feature_detected!("avx512f")) {
                return;
            }
            let mut st = 0x2468_ace0u64;
            let mut qs = [0u8; 16];
            for b in qs.iter_mut() {
                *b = (lcg(&mut st) * 127.0) as i32 as u8;
            }
            let block = BlockQ4_0 {
                d: f16::from_f32(0.037).to_bits(),
                qs,
            };
            let y: Vec<f32> = (0..32).map(|_| lcg(&mut st)).collect();
            let got = unsafe { vec_dot_q4_0_f32_avx512(&block, &y) };
            let want = vec_dot_q4_0_f32_scalar(&block, &y);
            assert!(
                (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                "{got} vs {want}"
            );
        }

        /// The row kernels must agree with summing the per-block kernel — that
        /// equivalence is the whole claim of hoisting the reduction out.
        #[test]
        fn row_dot_q4_0_avx512_matches_per_block_sum() {
            if !require_simd_or_skip("avx512", is_x86_feature_detected!("avx512f")) {
                return;
            }
            let mut st = 0x1122_3344u64;
            let nb = 5;
            let mut row = Vec::new();
            for _ in 0..nb {
                row.extend_from_slice(&f16::from_f32(0.03).to_bits().to_le_bytes());
                for _ in 0..16 {
                    row.push(((lcg(&mut st) + 1.0) * 127.0) as u8);
                }
            }
            let y: Vec<f32> = (0..nb * 32).map(|_| lcg(&mut st)).collect();

            let want: f32 = (0..nb)
                .map(|b| {
                    let blk = unsafe {
                        &*(row.as_ptr().add(b * size_of::<BlockQ4_0>()) as *const BlockQ4_0)
                    };
                    vec_dot_q4_0_f32_scalar(blk, &y[b * 32..(b + 1) * 32])
                })
                .sum();
            let got = unsafe { row_dot_q4_0_f32_avx512(row.as_ptr(), &y, nb) };
            assert!(
                (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                "{got} vs {want}"
            );
        }

        #[test]
        fn row_dot_q8_0_avx512_matches_per_block_sum() {
            if !require_simd_or_skip("avx512", is_x86_feature_detected!("avx512f")) {
                return;
            }
            let mut st = 0x5566_7788u64;
            let nb = 5;
            let mut row = Vec::new();
            for _ in 0..nb {
                row.extend_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
                for _ in 0..32 {
                    row.push((lcg(&mut st) * 127.0) as i32 as i8 as u8);
                }
            }
            let y: Vec<f32> = (0..nb * 32).map(|_| lcg(&mut st)).collect();

            let want: f32 = (0..nb)
                .map(|b| {
                    let blk = unsafe {
                        &*(row.as_ptr().add(b * size_of::<BlockQ8_0>()) as *const BlockQ8_0)
                    };
                    vec_dot_q8_0_f32_scalar(blk, &y[b * 32..(b + 1) * 32])
                })
                .sum();
            let got = unsafe { row_dot_q8_0_f32_avx512(row.as_ptr(), &y, nb) };
            assert!(
                (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                "{got} vs {want}"
            );
        }
    }
}

// ── x86_64 AVX-512 VNNI (int8 activations) ──────────────────────────────────
//
// The quantized-activation path the `avx512` module above defers to: instead of
// widening int8 weights to f32 and running FMA, quantize the *activations* to
// Q8_0 once and keep the whole dot product in int8, exactly like the aarch64
// dotprod/i8mm kernels. `_mm256_dpbusd_epi32` retires 32 int8 MACs per op
// against `_mm256_fmadd_ps`'s 8 f32 lanes.
//
// **Signedness.** VNNI's `dpbusd` takes an *unsigned* first operand and a signed
// second, but both our operands are signed. The standard fix (llama.cpp uses the
// same one) is `_mm256_sign_epi8`: `ax = sign(w, w)` is `|w|` (unsigned-safe —
// even `-128` maps to the byte `0x80` = 128, which is a valid u8 multiplicand),
// and `sy = sign(a, w)` folds `w`'s sign onto the activation. Their product is
// `|w| * sign(w) * a == w * a`, and the `w == 0` lanes zero both sides. No
// correction term, unlike the `-8 * sum(act)` a recentre-first formulation needs.
//
// **Accumulation.** Each Q8_0/Q4_0 block carries its own scale, so the int32 dot
// has to be scaled per block — but the *reduction* does not. Per block we convert
// the 8 int32 partials to f32 and FMA them into a vector accumulator, then
// reduce once per row. The `avx512` f32 kernels above instead call
// `_mm512_reduce_add_ps` per 32-element block; that horizontal reduce is a long
// dependency chain in the innermost loop.
//
// NOTE: not executable on the aarch64 dev host, and needs a VNNI-capable x86
// (Zen 4/5, Sapphire Rapids). `avx512_vnni_tests` below run only where
// `is_x86_feature_detected!("avx512vnni")` holds and compare against the scalar
// reference; elsewhere they skip.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod avx512_vnni {
    // 1.89 `_mm512_*`/`_mm256_dpbusd_*` intrinsics vs the 1.85 MSRV — same
    // rationale as the `avx512` module above.
    #![allow(clippy::incompatible_msrv)]
    use super::*;
    use std::arch::x86_64::*;

    /// Horizontal sum of the 8 f32 lanes. Called once per output element, never
    /// inside the block loop — see the module note on accumulation.
    #[target_feature(enable = "avx")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum = _mm_add_ps(hi, lo);
        let shuf = _mm_movehl_ps(sum, sum);
        let sum = _mm_add_ps(sum, shuf);
        let shuf = _mm_shuffle_ps(sum, sum, 0x55);
        _mm_cvtss_f32(_mm_add_ss(sum, shuf))
    }

    /// int8 dot of one 32-element block → 8 int32 partial sums.
    ///
    /// See the module note for why the operands go through `_mm256_sign_epi8`.
    ///
    /// **Precondition:** activation bytes must be in `[-127, 127]`. `-128` is
    /// the one value the sign trick cannot represent — `sign(a, w)` negates `a`
    /// when `w < 0`, and negating `-128` wraps back to `-128`, silently flipping
    /// that lane's sign. Weights may be `-128` (only `|w|` is taken, and `0x80`
    /// is a valid u8 multiplicand); activations may not. Every activation
    /// reaching here comes from `quantize_f32_to_q8_0_*`, which is bounded by
    /// `amax / 127` and guards the non-finite case, so it cannot emit `-128`.
    #[inline]
    #[target_feature(enable = "avx2,avx512vl,avx512vnni")]
    unsafe fn dot32(w: __m256i, a: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(w, w);
        let sy = _mm256_sign_epi8(a, w);
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy)
    }

    /// Unpack one Q4_0 block's 16 nibble-pairs into 32 signed bytes in
    /// `[-8, 7]`. Low nibbles are elements 0..16, high nibbles 16..32 — the
    /// same layout the NEON and scalar Q4_0 kernels assume.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_q4_0(qs: *const u8) -> __m256i {
        unsafe {
            let qb = _mm_loadu_si128(qs as *const __m128i);
            let mask = _mm_set1_epi8(0x0F);
            let lo = _mm_and_si128(qb, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
            _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
        }
    }

    /// Quantize `x` to Q8_0 blocks (scales + int8 quants).
    ///
    /// Mirrors `quantize_f32_to_q8_0_neon`, including the f16 round-trip of the
    /// scale: the aarch64 kernels store `d` as f16 because that is what a Q8_0
    /// block holds on disk, and a GEMV that mixed an f32 `d` here with an f16 `d`
    /// there would drift from the reference by more than rounding.
    #[target_feature(enable = "avx512f,avx512vl,avx2")]
    pub unsafe fn quantize_f32_to_q8_0_avx512(x: &[f32], scales: &mut [f32], quants: &mut [i8]) {
        unsafe {
            let k = x.len();
            debug_assert_eq!(
                k % 32,
                0,
                "quantize_f32_to_q8_0: x.len() must be divisible by 32"
            );
            debug_assert!(scales.len() >= k / 32);
            debug_assert!(quants.len() >= k);

            let abs_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
            // Indexed rather than `scales.iter_mut().take(..).enumerate()`: the
            // iterator form measured ~6% slower on prefill (LFM2.5-350M Q4_0,
            // Zen 5 — 175 vs 190 tok/s median over 3 reps), which is why the
            // lint is silenced here instead of obeyed.
            #[allow(clippy::needless_range_loop)]
            for bi in 0..k / 32 {
                let scale = &mut scales[bi];
                let base = bi * 32;
                let x_ptr = x.as_ptr().add(base);
                let v0 = _mm512_loadu_ps(x_ptr);
                let v1 = _mm512_loadu_ps(x_ptr.add(16));

                let amax = _mm512_reduce_max_ps(_mm512_max_ps(
                    _mm512_and_ps(v0, abs_mask),
                    _mm512_and_ps(v1, abs_mask),
                ));

                let d = amax / 127.0;
                // A near-zero block drives `d` denormal, and then `1.0 / d`
                // overflows to infinity. `_mm512_cvtps_epi32` maps any non-finite
                // operand to INT_MIN, which `cvtsepi32_epi8` saturates to -128 —
                // the one activation value `dot32`'s sign trick cannot represent
                // (`_mm256_sign_epi8` wraps negating it). The scalar quantizer
                // saturates the other way (+127), so without this the two
                // disagree byte-for-byte on the same input. The stored f16 scale
                // flushes to 0 for every such block, so results are unaffected
                // either way — but that is a coincidence, not a contract. Pin
                // both paths to a defined 0.
                let id = match 1.0 / d {
                    r if d != 0.0 && r.is_finite() => r,
                    _ => 0.0,
                };
                // Round-trip through f16 so the scale matches a stored Q8_0 block.
                *scale = f16::from_f32(d).to_f32();

                let idv = _mm512_set1_ps(id);
                let p0 = _mm512_mul_ps(v0, idv);
                let p1 = _mm512_mul_ps(v1, idv);
                // Scrub non-finite lanes to 0 before converting. A NaN
                // activation — or an infinity, whose product with the guarded
                // `id` is NaN — converts to INT_MIN on x86 and saturates to
                // -128, the one value `dot32`'s sign trick cannot represent.
                // Unlike the denormal case above, a single NaN among otherwise
                // normal values leaves the block scale perfectly normal, so that
                // -128 is *live*: it would silently flip that lane's sign
                // against any negative weight, turning a NaN that should have
                // propagated into a plausible finite number. Mapping to 0
                // matches what the scalar path's saturating `as i8` cast
                // already does, keeping the two byte-identical.
                let p0 = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask::<_CMP_ORD_Q>(p0, p0), p0);
                let p1 = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask::<_CMP_ORD_Q>(p1, p1), p1);
                // Default rounding is round-to-nearest-even, matching NEON's `vcvtnq`.
                let q0 = _mm512_cvtps_epi32(p0);
                let q1 = _mm512_cvtps_epi32(p1);
                let out = quants.as_mut_ptr().add(base);
                _mm_storeu_si128(out as *mut __m128i, _mm512_cvtsepi32_epi8(q0));
                _mm_storeu_si128(out.add(16) as *mut __m128i, _mm512_cvtsepi32_epi8(q1));
            }
        }
    }

    /// Dot one Q4_0 weight row against the pre-quantized activation.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    unsafe fn row_dot_q4_0(row: *const u8, x_scales: &[f32], x_quants: &[i8], nb: usize) -> f32 {
        unsafe {
            let bsz = size_of::<BlockQ4_0>();
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let block = &*(row.add(b * bsz) as *const BlockQ4_0);
                let w = unpack_q4_0(block.qs.as_ptr());
                let a = _mm256_loadu_si256(x_quants.as_ptr().add(b * 32) as *const __m256i);
                let scale =
                    _mm256_set1_ps(f16::from_bits(block.d).to_f32() * *x_scales.get_unchecked(b));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc);
            }
            hsum256_ps(acc)
        }
    }

    /// Dot one Q8_0 weight row against the pre-quantized activation.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    unsafe fn row_dot_q8_0(row: *const u8, x_scales: &[f32], x_quants: &[i8], nb: usize) -> f32 {
        unsafe {
            let bsz = size_of::<BlockQ8_0>();
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let block = &*(row.add(b * bsz) as *const BlockQ8_0);
                let w = _mm256_loadu_si256(block.quants.as_ptr() as *const __m256i);
                let a = _mm256_loadu_si256(x_quants.as_ptr().add(b * 32) as *const __m256i);
                let scale = _mm256_set1_ps(
                    f16::from_bits(block.delta).to_f32() * *x_scales.get_unchecked(b),
                );
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc);
            }
            hsum256_ps(acc)
        }
    }

    /// Q4_0 weights × pre-quantized Q8_0 activations: `y[m] = A[m,k] @ x[k]`.
    ///
    /// Row-parallel over the **RowPool** above `gemv_par_threshold()`, matching
    /// the NEON pre-quantized GEMV — decode dispatches these constantly, and
    /// rayon's fork-join barrier per call is the wrong trade there.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    pub unsafe fn gemv_q4_0_q8_0_avx512(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        m: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "Q4_0 GEMV: k must be divisible by 32");
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ4_0>();
        debug_assert_eq!(a_quant.len(), m * row_bytes);
        debug_assert_eq!(y.len(), m);
        debug_assert!(x_scales.len() >= nb && x_quants.len() >= k);

        let base = a_quant.as_ptr() as usize;
        let compute_row = |(i, yi): (usize, &mut f32)| {
            // SAFETY: row `i` spans `a_quant[i*row_bytes ..][..row_bytes]`, in
            // bounds by the assert above; `y` rows are disjoint per worker.
            *yi = unsafe {
                row_dot_q4_0(
                    (base as *const u8).add(i * row_bytes),
                    x_scales,
                    x_quants,
                    nb,
                )
            };
        };

        if m >= crate::backend::cpu::gemv_par_threshold() {
            crate::backend::cpu::par_rows(y, crate::backend::cpu::gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }

    /// Q8_0 weights × pre-quantized Q8_0 activations: `y[m] = A[m,k] @ x[k]`.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    pub unsafe fn gemv_q8_0_q8_0_avx512(
        a_quant: &[u8],
        x_scales: &[f32],
        x_quants: &[i8],
        y: &mut [f32],
        m: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "Q8_0 GEMV: k must be divisible by 32");
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ8_0>();
        debug_assert_eq!(a_quant.len(), m * row_bytes);
        debug_assert_eq!(y.len(), m);
        debug_assert!(x_scales.len() >= nb && x_quants.len() >= k);

        let base = a_quant.as_ptr() as usize;
        let compute_row = |(i, yi): (usize, &mut f32)| {
            // SAFETY: as in the Q4_0 GEMV above.
            *yi = unsafe {
                row_dot_q8_0(
                    (base as *const u8).add(i * row_bytes),
                    x_scales,
                    x_quants,
                    nb,
                )
            };
        };

        if m >= crate::backend::cpu::gemv_par_threshold() {
            crate::backend::cpu::par_rows(y, crate::backend::cpu::gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }

    // ── Batched prefill GEMM ────────────────────────────────────────────────
    //
    // The point of these over a per-token GEMV loop: one weight row is decoded
    // once and reused across `TILE_N` activation columns, so a prefill of `n`
    // tokens streams the weight matrix `n / TILE_N` times instead of `n`. Prefill
    // on x86 is weight-bandwidth bound, so that ratio *is* the speedup.
    //
    // Column-major activations: `quantize_columns` packs column `j` contiguously
    // at `b_quants[j * k ..]` with scales at `b_scales[j * nb ..]`, so each of the
    // `TILE_N` loads below is a straight 32-byte read.

    /// Activation columns processed per weight decode. Four keeps the tile's
    /// accumulators plus the decoded weight inside the 16 available ymm
    /// registers; eight spills on the Q4_0 path (the nibble unpack needs its own
    /// temporaries) and measured slower.
    const TILE_N: usize = 4;

    /// One output row of the Q4_0 GEMM: `out[j] = <A_row, B_col_j>` for all `n`.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    unsafe fn gemm_q4_0_row(
        row: *const u8,
        b_scales: &[f32],
        b_quants: &[i8],
        out_row: &mut [f32],
        n: usize,
        nb: usize,
    ) {
        unsafe {
            let bsz = size_of::<BlockQ4_0>();
            let k = nb * 32;
            let mut j = 0;

            while j + TILE_N <= n {
                let mut acc = [_mm256_setzero_ps(); TILE_N];
                for b in 0..nb {
                    let block = &*(row.add(b * bsz) as *const BlockQ4_0);
                    let dw = f16::from_bits(block.d).to_f32();
                    let w = unpack_q4_0(block.qs.as_ptr());
                    for (t, a_t) in acc.iter_mut().enumerate() {
                        let col = j + t;
                        let a = _mm256_loadu_si256(
                            b_quants.as_ptr().add(col * k + b * 32) as *const __m256i
                        );
                        let scale = _mm256_set1_ps(dw * *b_scales.get_unchecked(col * nb + b));
                        *a_t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, *a_t);
                    }
                }
                for (t, a_t) in acc.iter().enumerate() {
                    *out_row.get_unchecked_mut(j + t) = hsum256_ps(*a_t);
                }
                j += TILE_N;
            }

            // Column remainder (n % TILE_N). Same math, one column at a time.
            while j < n {
                let mut acc = _mm256_setzero_ps();
                for b in 0..nb {
                    let block = &*(row.add(b * bsz) as *const BlockQ4_0);
                    let w = unpack_q4_0(block.qs.as_ptr());
                    let a =
                        _mm256_loadu_si256(b_quants.as_ptr().add(j * k + b * 32) as *const __m256i);
                    let scale = _mm256_set1_ps(
                        f16::from_bits(block.d).to_f32() * *b_scales.get_unchecked(j * nb + b),
                    );
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc);
                }
                *out_row.get_unchecked_mut(j) = hsum256_ps(acc);
                j += 1;
            }
        }
    }

    /// One output row of the Q8_0 GEMM.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    unsafe fn gemm_q8_0_row(
        row: *const u8,
        b_scales: &[f32],
        b_quants: &[i8],
        out_row: &mut [f32],
        n: usize,
        nb: usize,
    ) {
        unsafe {
            let bsz = size_of::<BlockQ8_0>();
            let k = nb * 32;
            let mut j = 0;

            while j + TILE_N <= n {
                let mut acc = [_mm256_setzero_ps(); TILE_N];
                for b in 0..nb {
                    let block = &*(row.add(b * bsz) as *const BlockQ8_0);
                    let dw = f16::from_bits(block.delta).to_f32();
                    let w = _mm256_loadu_si256(block.quants.as_ptr() as *const __m256i);
                    for (t, a_t) in acc.iter_mut().enumerate() {
                        let col = j + t;
                        let a = _mm256_loadu_si256(
                            b_quants.as_ptr().add(col * k + b * 32) as *const __m256i
                        );
                        let scale = _mm256_set1_ps(dw * *b_scales.get_unchecked(col * nb + b));
                        *a_t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, *a_t);
                    }
                }
                for (t, a_t) in acc.iter().enumerate() {
                    *out_row.get_unchecked_mut(j + t) = hsum256_ps(*a_t);
                }
                j += TILE_N;
            }

            while j < n {
                let mut acc = _mm256_setzero_ps();
                for b in 0..nb {
                    let block = &*(row.add(b * bsz) as *const BlockQ8_0);
                    let w = _mm256_loadu_si256(block.quants.as_ptr() as *const __m256i);
                    let a =
                        _mm256_loadu_si256(b_quants.as_ptr().add(j * k + b * 32) as *const __m256i);
                    let scale = _mm256_set1_ps(
                        f16::from_bits(block.delta).to_f32() * *b_scales.get_unchecked(j * nb + b),
                    );
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc);
                }
                *out_row.get_unchecked_mut(j) = hsum256_ps(acc);
                j += 1;
            }
        }
    }

    /// Batched Q4_0 × Q8_0 GEMM: `out[m,n] = A_q4_0[m,k] @ B_q8_0[k,n]`,
    /// parallel over output rows.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    pub unsafe fn gemm_q4_0_q8_0_avx512(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ4_0>();
        debug_assert_eq!(a_quant.len(), m * row_bytes);
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * nb);
        debug_assert_eq!(out.len(), m * n);

        #[cfg(feature = "parallel")]
        {
            use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            let base = a_quant.as_ptr() as usize;
            out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
                // SAFETY: row `i` reads `a_quant[i*row_bytes ..][..row_bytes]`
                // (in bounds by the assert above, read-only and shared), and
                // writes only its own disjoint `out_row` chunk.
                unsafe {
                    gemm_q4_0_row(
                        (base as *const u8).add(i * row_bytes),
                        b_scales,
                        b_quants,
                        out_row,
                        n,
                        nb,
                    );
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        for (i, out_row) in out.chunks_mut(n).enumerate() {
            // SAFETY: as in the parallel branch above.
            unsafe {
                gemm_q4_0_row(
                    a_quant.as_ptr().add(i * row_bytes),
                    b_scales,
                    b_quants,
                    out_row,
                    n,
                    nb,
                );
            }
        }
    }

    /// Batched Q8_0 × Q8_0 GEMM: `out[m,n] = A_q8_0[m,k] @ B_q8_0[k,n]`.
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
    pub unsafe fn gemm_q8_0_q8_0_avx512(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "GEMM: k must be divisible by 32");
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ8_0>();
        debug_assert_eq!(a_quant.len(), m * row_bytes);
        debug_assert_eq!(b_quants.len(), n * k);
        debug_assert_eq!(b_scales.len(), n * nb);
        debug_assert_eq!(out.len(), m * n);

        #[cfg(feature = "parallel")]
        {
            use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            let base = a_quant.as_ptr() as usize;
            out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
                // SAFETY: as in the Q4_0 GEMM above.
                unsafe {
                    gemm_q8_0_row(
                        (base as *const u8).add(i * row_bytes),
                        b_scales,
                        b_quants,
                        out_row,
                        n,
                        nb,
                    );
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        for (i, out_row) in out.chunks_mut(n).enumerate() {
            // SAFETY: as in the parallel branch above.
            unsafe {
                gemm_q8_0_row(
                    a_quant.as_ptr().add(i * row_bytes),
                    b_scales,
                    b_quants,
                    out_row,
                    n,
                    nb,
                );
            }
        }
    }

    #[cfg(test)]
    mod avx512_vnni_tests {
        use super::*;

        /// Uniform `[0, 1)`. Deliberately not the `avx512_tests` `lcg`, which
        /// returns `[-1, 0)` — the byte-pattern builders below cast to `u8`, and
        /// a negative float saturates to 0 there, which would quietly test a
        /// matrix of all-zero nibbles.
        fn lcg01(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 40) as f32 / (1u64 << 24) as f32
        }

        /// Every feature in these kernels' `#[target_feature]` list, not just
        /// `avx512vnni`. Calling them needs the whole set, so gating on VNNI
        /// alone would be UB on a part that reports VNNI without, say, `avx512vl`
        /// — the same conjunction `cpu_features` requires to pick the tier.
        fn vnni_kernels_callable() -> bool {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
        }

        /// Tier-test gate; mirrors the one in `avx512_tests`. With
        /// `CERA_REQUIRE_SIMD=avx512vnni` a missing feature fails instead of
        /// skipping, so CI on VNNI hardware proves these kernels actually ran.
        fn require_simd_or_skip(feature: &str, detected: bool) -> bool {
            if detected {
                return true;
            }
            let required = std::env::var("CERA_REQUIRE_SIMD").unwrap_or_default();
            assert!(
                !required.split(',').any(|f| f.trim() == feature),
                "CERA_REQUIRE_SIMD requires `{feature}` but this host doesn't report it"
            );
            false
        }

        /// Scalar mirror of `quantize_f32_to_q8_0_avx512`, including the f16
        /// round-trip of `d` and round-to-nearest-even.
        fn ref_quantize(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
            let mut scales = Vec::new();
            let mut quants = Vec::new();
            for blk in x.chunks(32) {
                let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let d = amax / 127.0;
                // Mirrors the non-finite guard in the kernel under test.
                let id = match 1.0 / d {
                    r if d != 0.0 && r.is_finite() => r,
                    _ => 0.0,
                };
                scales.push(f16::from_f32(d).to_f32());
                for &v in blk {
                    quants.push((v * id).round_ties_even().clamp(-128.0, 127.0) as i8);
                }
            }
            (scales, quants)
        }

        /// Exact integer reference for a Q4_0 × Q8_0 dot: the kernel's contract
        /// is `sum_b d_w[b] * d_x[b] * <nibbles-8, x_quants>`, so the reference
        /// does that in i32 rather than dequantizing to f32 (which would fold in
        /// a second, different rounding and make a real mismatch unreadable).
        fn ref_gemv_q4_0(a: &[u8], xs: &[f32], xq: &[i8], m: usize, k: usize) -> Vec<f32> {
            let nb = k / 32;
            let bsz = size_of::<BlockQ4_0>();
            let mut y = vec![0.0f32; m];
            for (i, yi) in y.iter_mut().enumerate() {
                for b in 0..nb {
                    let blk =
                        unsafe { &*(a.as_ptr().add(i * nb * bsz + b * bsz) as *const BlockQ4_0) };
                    let mut acc = 0i32;
                    for t in 0..16 {
                        let byte = blk.qs[t];
                        acc += ((byte & 0xF) as i32 - 8) * xq[b * 32 + t] as i32;
                        acc += ((byte >> 4) as i32 - 8) * xq[b * 32 + t + 16] as i32;
                    }
                    *yi += f16::from_bits(blk.d).to_f32() * xs[b] * acc as f32;
                }
            }
            y
        }

        fn ref_gemv_q8_0(a: &[u8], xs: &[f32], xq: &[i8], m: usize, k: usize) -> Vec<f32> {
            let nb = k / 32;
            let bsz = size_of::<BlockQ8_0>();
            let mut y = vec![0.0f32; m];
            for (i, yi) in y.iter_mut().enumerate() {
                for b in 0..nb {
                    let blk =
                        unsafe { &*(a.as_ptr().add(i * nb * bsz + b * bsz) as *const BlockQ8_0) };
                    let mut acc = 0i32;
                    for t in 0..32 {
                        acc += blk.quants[t] as i32 * xq[b * 32 + t] as i32;
                    }
                    *yi += f16::from_bits(blk.delta).to_f32() * xs[b] * acc as f32;
                }
            }
            y
        }

        fn rand_q4_0_rows(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let nb = k / 32;
            let mut v = Vec::with_capacity(m * nb * size_of::<BlockQ4_0>());
            for _ in 0..m * nb {
                v.extend_from_slice(
                    &f16::from_f32(lcg01(st) * 0.05 + 0.01)
                        .to_bits()
                        .to_le_bytes(),
                );
                for _ in 0..16 {
                    v.push((lcg01(st) * 255.0) as u8);
                }
            }
            v
        }

        fn rand_q8_0_rows(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let nb = k / 32;
            let mut v = Vec::with_capacity(m * nb * size_of::<BlockQ8_0>());
            for _ in 0..m * nb {
                v.extend_from_slice(
                    &f16::from_f32(lcg01(st) * 0.05 + 0.01)
                        .to_bits()
                        .to_le_bytes(),
                );
                for _ in 0..32 {
                    v.push(((lcg01(st) * 254.0) as i32 - 127) as i8 as u8);
                }
            }
            v
        }

        fn assert_close(got: &[f32], want: &[f32], what: &str) {
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
                    "{what}[{i}]: {g} vs {w}"
                );
            }
        }

        #[test]
        fn quantize_q8_0_avx512_matches_scalar() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let mut st = 0x1357_9bdfu64;
            let x: Vec<f32> = (0..256).map(|_| lcg01(&mut st) * 4.0 - 2.0).collect();
            let (ws, wq) = ref_quantize(&x);
            let mut gs = vec![0.0f32; 8];
            let mut gq = vec![0i8; 256];
            unsafe { quantize_f32_to_q8_0_avx512(&x, &mut gs, &mut gq) };
            assert_eq!(gs, ws, "scales");
            assert_eq!(gq, wq, "quants");
        }

        /// An all-zero block makes `d == 0`, so the reciprocal is forced to 0
        /// rather than inf — check the kernel takes that branch too.
        #[test]
        fn quantize_q8_0_avx512_handles_zero_block() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let x = vec![0.0f32; 64];
            let mut gs = vec![9.0f32; 2];
            let mut gq = vec![9i8; 64];
            unsafe { quantize_f32_to_q8_0_avx512(&x, &mut gs, &mut gq) };
            assert_eq!(gs, vec![0.0, 0.0]);
            assert!(gq.iter().all(|&q| q == 0));
        }

        /// Non-finite inputs must not reach `dot32` as `-128`.
        ///
        /// Two distinct hazards, both x86-specific: a near-zero block drives `d`
        /// denormal and `1.0 / d` to infinity, and a NaN activation converts
        /// straight to INT_MIN. Either saturates to `-128`, the one activation
        /// value the sign trick cannot represent, while the scalar path
        /// saturates to `+127`/`0` — so the two quantizers disagreed
        /// byte-for-byte on the same input.
        ///
        /// The mixed cases are the dangerous ones: with one NaN among normal
        /// values the block scale stays perfectly normal, so the `-128` is live
        /// and would silently flip that lane's sign against a negative weight,
        /// converting a NaN that should have propagated into a plausible finite
        /// number.
        #[test]
        fn quantize_q8_0_avx512_non_finite_blocks_match_scalar() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let mut cases: Vec<(&str, Vec<f32>)> = Vec::new();
            for (label, amax) in [
                ("small", 1e-30f32),
                ("denormal-d", 1e-38),
                ("denormal-d2", 1e-40),
                ("flush-to-zero", 1e-44),
                ("all-nan", f32::NAN),
            ] {
                let mut x = vec![0.0f32; 32];
                x[0] = amax;
                x[1] = -amax;
                x[2] = amax / 2.0;
                cases.push((label, x));
            }
            // Scale stays normal here, so a stray -128 would be live.
            let mut mixed_nan = vec![0.25f32; 32];
            mixed_nan[0] = f32::NAN;
            mixed_nan[1] = -1.0;
            cases.push(("nan-with-normal", mixed_nan));
            let mut mixed_inf = vec![0.25f32; 32];
            mixed_inf[0] = f32::INFINITY;
            mixed_inf[1] = -1.0;
            cases.push(("inf-with-normal", mixed_inf));

            for (label, x) in cases {
                let (ws, wq) = ref_quantize(&x);
                let mut gs = vec![0.0f32; 1];
                let mut gq = vec![0i8; 32];
                unsafe { quantize_f32_to_q8_0_avx512(&x, &mut gs, &mut gq) };

                assert_eq!(gq, wq, "quants disagree with scalar ({label})");
                assert!(
                    gq.iter().all(|&q| q != -128),
                    "emitted -128, which breaks the dot32 sign trick ({label})"
                );
                assert_eq!(gs[0].is_nan(), ws[0].is_nan(), "scale NaN-ness ({label})");
                if !gs[0].is_nan() {
                    assert_eq!(gs[0], ws[0], "scale disagrees with scalar ({label})");
                }
            }
        }

        #[test]
        fn gemv_q4_0_q8_0_avx512_matches_scalar() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, k) = (37, 128); // odd m exercises the row tail
            let mut st = 0x2468_1357u64;
            let a = rand_q4_0_rows(m, k, &mut st);
            let x: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
            let (xs, xq) = ref_quantize(&x);
            let mut y = vec![0.0f32; m];
            unsafe { gemv_q4_0_q8_0_avx512(&a, &xs, &xq, &mut y, m, k) };
            assert_close(&y, &ref_gemv_q4_0(&a, &xs, &xq, m, k), "gemv_q4_0");
        }

        #[test]
        fn gemv_q8_0_q8_0_avx512_matches_scalar() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, k) = (37, 128);
            let mut st = 0x9876_5432u64;
            let a = rand_q8_0_rows(m, k, &mut st);
            let x: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
            let (xs, xq) = ref_quantize(&x);
            let mut y = vec![0.0f32; m];
            unsafe { gemv_q8_0_q8_0_avx512(&a, &xs, &xq, &mut y, m, k) };
            assert_close(&y, &ref_gemv_q8_0(&a, &xs, &xq, m, k), "gemv_q8_0");
        }

        /// The GEMM must agree with the GEMV column-by-column. `n = 7` is
        /// deliberately not a multiple of `TILE_N`, so this covers both the
        /// 4-wide tile and the scalar column remainder.
        #[test]
        fn gemm_q4_0_avx512_matches_gemv_per_column() {
            if !require_simd_or_skip("avx512vnni", is_x86_feature_detected!("avx512vnni")) {
                return;
            }
            let (m, n, k) = (13, 7, 96);
            let nb = k / 32;
            let mut st = 0x0bad_c0deu64;
            let a = rand_q4_0_rows(m, k, &mut st);

            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out = vec![0.0f32; m * n];
            unsafe { gemm_q4_0_q8_0_avx512(&a, &b_scales, &b_quants, &mut out, m, n, k) };

            for j in 0..n {
                let mut y = vec![0.0f32; m];
                unsafe {
                    gemv_q4_0_q8_0_avx512(
                        &a,
                        &b_scales[j * nb..(j + 1) * nb],
                        &b_quants[j * k..(j + 1) * k],
                        &mut y,
                        m,
                        k,
                    )
                };
                let col: Vec<f32> = (0..m).map(|i| out[i * n + j]).collect();
                assert_close(&col, &y, &format!("gemm_q4_0 col {j}"));
            }
        }

        #[test]
        fn gemm_q8_0_avx512_matches_gemv_per_column() {
            if !require_simd_or_skip("avx512vnni", is_x86_feature_detected!("avx512vnni")) {
                return;
            }
            let (m, n, k) = (13, 7, 96);
            let nb = k / 32;
            let mut st = 0xfeed_face_u64;
            let a = rand_q8_0_rows(m, k, &mut st);

            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out = vec![0.0f32; m * n];
            unsafe { gemm_q8_0_q8_0_avx512(&a, &b_scales, &b_quants, &mut out, m, n, k) };

            for j in 0..n {
                let mut y = vec![0.0f32; m];
                unsafe {
                    gemv_q8_0_q8_0_avx512(
                        &a,
                        &b_scales[j * nb..(j + 1) * nb],
                        &b_quants[j * k..(j + 1) * k],
                        &mut y,
                        m,
                        k,
                    )
                };
                let col: Vec<f32> = (0..m).map(|i| out[i * n + j]).collect();
                assert_close(&col, &y, &format!("gemm_q8_0 col {j}"));
            }
        }
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

/// Best available Q4_0 dot product.
pub fn vec_dot_q4_0_f32(block: &BlockQ4_0, y: &[f32]) -> f32 {
    assert_eq!(y.len(), 32, "Q4_0 vec_dot requires y.len() == 32");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_dot_q4_0_f32_neon(block, y) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use crate::backend::cpu_features::{CpuTier, cpu_features};
        match cpu_features().tier {
            CpuTier::Scalar => crate::quant::vec_dot_q4_0_f32_scalar(block, y),
            // `Avx512` is only produced when the `avx512` feature is on; the
            // gated arm matches the module's gate so the build is consistent
            // either way (with it off, the tier folds into the AVX2 arm).
            #[cfg(feature = "avx512")]
            CpuTier::Avx512 => unsafe { avx512::vec_dot_q4_0_f32_avx512(block, y) },
            _ => unsafe { avx2::vec_dot_q4_0_f32_avx2(block, y) },
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::quant::vec_dot_q4_0_f32_scalar(block, y)
    }
}

/// Best available Q8_0 dot product.
pub fn vec_dot_q8_0_f32(block: &BlockQ8_0, y: &[f32]) -> f32 {
    assert_eq!(y.len(), 32, "Q8_0 vec_dot requires y.len() == 32");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_dot_q8_0_f32_neon(block, y) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use crate::backend::cpu_features::{CpuTier, cpu_features};
        match cpu_features().tier {
            CpuTier::Scalar => crate::quant::vec_dot_q8_0_f32_scalar(block, y),
            #[cfg(feature = "avx512")]
            CpuTier::Avx512 => unsafe { avx512::vec_dot_q8_0_f32_avx512(block, y) },
            _ => unsafe { avx2::vec_dot_q8_0_f32_avx2(block, y) },
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::quant::vec_dot_q8_0_f32_scalar(block, y)
    }
}

/// Best available Q4_K_M dot product.
pub fn vec_dot_q4_k_m_f32(block: &BlockQ4KM, y: &[f32]) -> f32 {
    assert_eq!(y.len(), 256, "Q4_K_M vec_dot requires y.len() == 256");
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_dot_q4_k_m_f32_neon(block, y) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use crate::backend::cpu_features::{CpuTier, cpu_features};
        match cpu_features().tier {
            CpuTier::Scalar => crate::quant::vec_dot_q4_k_m_f32_scalar(block, y),
            _ => unsafe { avx2::vec_dot_q4_k_m_f32_avx2(block, y) },
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::quant::vec_dot_q4_k_m_f32_scalar(block, y)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_q4_0_matches_scalar() {
        let block = BlockQ4_0 {
            d: f16::from_f32(0.5).to_bits(),
            qs: {
                let mut q = [0u8; 16];
                for (i, qi) in q.iter_mut().enumerate() {
                    *qi = ((i % 13) as u8) | (((i % 7) as u8) << 4);
                }
                q
            },
        };
        let y: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let scalar = crate::quant::vec_dot_q4_0_f32_scalar(&block, &y);
        let simd = vec_dot_q4_0_f32(&block, &y);

        assert!(
            (scalar - simd).abs() < 1e-3,
            "SIMD Q4_0 mismatch: scalar={scalar}, simd={simd}"
        );
    }

    #[test]
    fn test_simd_q8_0_matches_scalar() {
        let block = BlockQ8_0 {
            delta: f16::from_f32(0.3).to_bits(),
            quants: {
                let mut q = [0i8; 32];
                for (i, qi) in q.iter_mut().enumerate() {
                    *qi = (i as i8) * 3 - 48;
                }
                q
            },
        };
        let y: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let scalar = crate::quant::vec_dot_q8_0_f32_scalar(&block, &y);
        let simd = vec_dot_q8_0_f32(&block, &y);

        assert!(
            (scalar - simd).abs() < 1e-3,
            "SIMD Q8_0 mismatch: scalar={scalar}, simd={simd}"
        );
    }

    #[test]
    fn test_simd_q4km_matches_scalar() {
        let mut block = BlockQ4KM {
            d: f16::from_f32(0.5).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };

        for i in 0..4 {
            block.scales[i] = 3;
        }
        for i in 4..8 {
            block.scales[i] = 1;
        }
        for i in 8..12 {
            block.scales[i] = 0x12;
        }

        for (i, b) in block.qs.iter_mut().enumerate() {
            *b = ((i % 13) as u8) | (((i % 9) as u8) << 4);
        }

        let y: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let scalar = crate::quant::vec_dot_q4_k_m_f32_scalar(&block, &y);
        let simd = vec_dot_q4_k_m_f32(&block, &y);

        assert!(
            (scalar - simd).abs() < 1e-2,
            "SIMD Q4_K_M mismatch: scalar={scalar}, simd={simd}"
        );
    }

    /// Build a Q4_0-quantized weight matrix (m rows × k cols) from f32 values.
    /// Returns raw bytes suitable for GEMM kernels.
    #[cfg(target_arch = "aarch64")]
    fn build_q4_0_matrix(values: &[f32], m: usize, k: usize) -> Vec<u8> {
        assert_eq!(values.len(), m * k);
        let nb = k / 32;
        let mut bytes = Vec::new();
        for row in 0..m {
            for b in 0..nb {
                let block_start = row * k + b * 32;
                let block = &values[block_start..block_start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 7.0;
                let d_f16 = half::f16::from_f32(d);
                bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut qs = [0u8; 16];
                for i in 0..16 {
                    let lo = ((block[i] * id + 8.5) as u8).min(15);
                    let hi = ((block[16 + i] * id + 8.5) as u8).min(15);
                    qs[i] = lo | (hi << 4);
                }
                bytes.extend_from_slice(&qs);
            }
        }
        bytes
    }

    /// Build a Q8_0-quantized weight matrix (m rows × k cols) from f32 values.
    #[cfg(target_arch = "aarch64")]
    fn build_q8_0_matrix(values: &[f32], m: usize, k: usize) -> Vec<u8> {
        assert_eq!(values.len(), m * k);
        let nb = k / 32;
        let mut bytes = Vec::new();
        for row in 0..m {
            for b in 0..nb {
                let block_start = row * k + b * 32;
                let block = &values[block_start..block_start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 127.0;
                let d_f16 = half::f16::from_f32(d);
                bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut qs = [0i8; 32];
                for i in 0..32 {
                    qs[i] = (block[i] * id).round().clamp(-128.0, 127.0) as i8;
                }
                bytes.extend_from_slice(bytemuck::cast_slice(&qs));
            }
        }
        bytes
    }

    /// Quantize n columns of f32 input to Q8_0 format (scales + quants).
    #[cfg(target_arch = "aarch64")]
    fn quantize_input_columns(inputs: &[Vec<f32>], k: usize) -> (Vec<f32>, Vec<i8>) {
        let n = inputs.len();
        let nb = k / 32;
        let mut scales = vec![0.0f32; n * nb];
        let mut quants = vec![0i8; n * k];
        for (j, col) in inputs.iter().enumerate() {
            unsafe {
                neon::quantize_f32_to_q8_0_neon(
                    col,
                    &mut scales[j * nb..(j + 1) * nb],
                    &mut quants[j * k..(j + 1) * k],
                );
            }
        }
        (scales, quants)
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_matches_sequential_gemv() {
        let m = 8;
        let k = 64; // 2 Q4_0 blocks per row
        let n = 7; // not divisible by 4, tests remainder path

        // Random-ish weight values
        let weights: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let a_bytes = build_q4_0_matrix(&weights, m, k);

        // Random-ish input columns
        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|i| ((i * 13 + j * 7 + 5) % 23) as f32 * 0.2 - 2.3)
                    .collect()
            })
            .collect();

        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        // GEMM
        let mut gemm_out = vec![0.0f32; m * n];
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }

        // Sequential GEMV for each column
        for j in 0..n {
            let col_scales = &b_scales[j * (k / 32)..(j + 1) * (k / 32)];
            let col_quants = &b_quants[j * k..(j + 1) * k];
            let mut gemv_out = vec![0.0f32; m];
            unsafe {
                neon::gemv_q4_0_q8_0_neon(&a_bytes, col_scales, col_quants, &mut gemv_out, m, k);
            }
            for i in 0..m {
                let diff = (gemm_out[i * n + j] - gemv_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "GEMM/GEMV Q4_0 mismatch at [{i},{j}]: gemm={}, gemv={}, diff={diff}",
                    gemm_out[i * n + j],
                    gemv_out[i]
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q8_0_matches_sequential_gemv() {
        let m = 8;
        let k = 64;
        let n = 5;

        let weights: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let a_bytes = build_q8_0_matrix(&weights, m, k);

        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|i| ((i * 13 + j * 7 + 5) % 23) as f32 * 0.2 - 2.3)
                    .collect()
            })
            .collect();

        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        let mut gemm_out = vec![0.0f32; m * n];
        unsafe {
            neon::gemm_q8_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }

        for j in 0..n {
            let col_scales = &b_scales[j * (k / 32)..(j + 1) * (k / 32)];
            let col_quants = &b_quants[j * k..(j + 1) * k];
            let mut gemv_out = vec![0.0f32; m];
            unsafe {
                neon::gemv_q8_0_q8_0_neon(&a_bytes, col_scales, col_quants, &mut gemv_out, m, k);
            }
            for i in 0..m {
                let diff = (gemm_out[i * n + j] - gemv_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "GEMM/GEMV Q8_0 mismatch at [{i},{j}]: gemm={}, gemv={}, diff={diff}",
                    gemm_out[i * n + j],
                    gemv_out[i]
                );
            }
        }
    }

    /// Helper: run GEMM and compare against sequential GEMV for given dimensions.
    #[cfg(target_arch = "aarch64")]
    fn assert_gemm_q4_0_matches_gemv(m: usize, k: usize, n: usize) {
        let weights: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let a_bytes = build_q4_0_matrix(&weights, m, k);
        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|i| ((i * 13 + j * 7 + 5) % 23) as f32 * 0.2 - 2.3)
                    .collect()
            })
            .collect();
        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        let mut gemm_out = vec![0.0f32; m * n];
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }
        for j in 0..n {
            let col_scales = &b_scales[j * (k / 32)..(j + 1) * (k / 32)];
            let col_quants = &b_quants[j * k..(j + 1) * k];
            let mut gemv_out = vec![0.0f32; m];
            unsafe {
                neon::gemv_q4_0_q8_0_neon(&a_bytes, col_scales, col_quants, &mut gemv_out, m, k);
            }
            for i in 0..m {
                let diff = (gemm_out[i * n + j] - gemv_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "GEMM/GEMV Q4_0 mismatch at [{i},{j}] (m={m},k={k},n={n}): gemm={}, gemv={}, diff={diff}",
                    gemm_out[i * n + j],
                    gemv_out[i]
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_8col() {
        // n=8: exact 8-column path, no remainder
        assert_gemm_q4_0_matches_gemv(8, 64, 8);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_8col_plus_remainder() {
        // n=11: 8-column path (1 iter) + 3-column remainder (exercises all code paths)
        assert_gemm_q4_0_matches_gemv(8, 64, 11);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_16col() {
        // n=16: two iterations of 8-column path
        assert_gemm_q4_0_matches_gemv(8, 64, 16);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_q4_0_single_column() {
        // n=1: tests single-column fallback path
        let m = 4;
        let k = 32;
        let n = 1;

        let weights: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 * 0.3 - 1.5).collect();
        let a_bytes = build_q4_0_matrix(&weights, m, k);
        let inputs = vec![(0..k).map(|i| (i % 7) as f32 * 0.5 - 1.75).collect()];
        let (b_scales, b_quants) = quantize_input_columns(&inputs, k);

        let mut gemm_out = vec![0.0f32; m];
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemm_out, m, n, k);
        }

        let mut gemv_out = vec![0.0f32; m];
        unsafe {
            neon::gemv_q4_0_q8_0_neon(&a_bytes, &b_scales, &b_quants, &mut gemv_out, m, k);
        }

        for i in 0..m {
            let diff = (gemm_out[i] - gemv_out[i]).abs();
            assert!(diff < 1e-4, "n=1 mismatch at row {i}: {diff}");
        }
    }
}

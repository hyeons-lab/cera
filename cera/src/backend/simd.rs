// SIMD-optimized kernels for quantized operations.
//
// Platform-specific implementations behind cfg gates.
// The dispatch functions select the best available implementation at compile time.

// `BlockQ4_1` / `BlockQ6K` are used only by the aarch64 NEON kernels below (the
// x86 macro kernels reference `crate::quant::…` fully-qualified), so importing
// them unconditionally trips `-D unused-imports` on x86.
use crate::quant::{BlockQ4_0, BlockQ4KM, BlockQ8_0};
#[cfg(target_arch = "aarch64")]
use crate::quant::{BlockQ4_1, BlockQ6K};
// `half::f16` is consumed by the NEON / AVX2 kernels below and by the
// `#[cfg(test)] mod tests` further down (the tests aren't arch-gated and
// use `f16::from_f32` to seed quantized blocks). Including `test` in the
// gate keeps `cargo test` compilable on architectures that don't have a
// SIMD kernel here (e.g. armv7, riscv64) — without it those archs build
// the tests but lose the import. On non-test wasm32 builds the import
// remains correctly elided.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", test))]
use half::f16;

/// Tier-test gate shared by every SIMD test module in this file.
///
/// Returns true when the test should run. A host without `feature` skips
/// silently by default; if `CERA_REQUIRE_SIMD` lists `feature`, the missing
/// feature is a hard failure instead — so a CI job on known-capable hardware
/// proves the kernel actually executed rather than quietly reporting green.
///
/// One definition per *crate module*, not one per test module: the skip-vs-fail
/// rule is a project-wide convention, and a copy per module is how one ends up
/// with a weaker gate than the CI leg that targets it assumes. There were three
/// before this change, and the AVX2 module would have made four. The
/// integration-test binaries under `cera/tests/` still hand-roll it — they are
/// separate crates and cannot name a `#[cfg(test)]` item in this one.
///
/// Cfg'd to the architectures that have a SIMD test module: at module scope a
/// single `#[cfg(test)]` would be `dead_code` on wasm32/riscv64, which the four
/// per-module copies this replaced were cfg'd out of. Only `cargo check
/// --profile test` on such a target shows it, which CI does not run.
#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
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

    /// Batched GEMM: C[m, n] = A_q4_1[m, k] @ B_q8_0[k, n].
    ///
    /// Q4_1 dequant is `w = d·q + m` with `q ∈ [0, 15]` — no `−8` recentering. Against
    /// a Q8_0-quantized activation column (`x = xs · xq`), the per-32-block contribution
    /// is
    ///
    /// ```text
    /// out[i][j] += xs · ( d·Σ(q·xq) + m·Σ(xq) )
    /// ```
    ///
    /// `Σ(q·xq)` is the int8 dot; `Σ(xq)` is the activation block-sum, hoisted once per
    /// column by [`q8_0_col_sums`] exactly like the Q4_K min term — but **added**, since
    /// Q4_1's `m` raises the value where the K-quant `dmin` subtracts. A Q4_1 block is 32
    /// values, aligning 1:1 with the Q8_0 input blocks, so weight block `bi` dots input
    /// block `bi` with no superblock bookkeeping. Nibble layout mirrors
    /// `dequantize_q4_1_block`: low nibble of `qs[t]` → element index `t`, high nibble →
    /// index `t + 16`, so the low/high halves pair with input halves `x0`/`x1`.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemm_q4_1_q8_0_neon_dotprod(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert_eq!(k % 32, 0, "Q4_1 GEMM: k must be divisible by 32");
        let nb = k / 32;
        let row_bytes = nb * size_of::<BlockQ4_1>();
        debug_assert_eq!(a_quant.len(), m * row_bytes, "Q4_1 GEMM: a_quant size");
        debug_assert_eq!(b_quants.len(), n * k, "Q4_1 GEMM: b_quants size");
        debug_assert_eq!(b_scales.len(), n * nb, "Q4_1 GEMM: b_scales size");
        debug_assert_eq!(out.len(), m * n, "Q4_1 GEMM: out size");

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

                    for bi in 0..nb {
                        let blk = &*((a as usize + row_start + bi * size_of::<BlockQ4_1>())
                            as *const BlockQ4_1);
                        let d = half::f16::from_bits(blk.d).to_f32();
                        let mmin = half::f16::from_bits(blk.m).to_f32();
                        // Low nibbles → element indices 0..16, high nibbles → 16..32;
                        // the nibble *values* are all in `[0, 15]`, so they are
                        // non-negative as `i8`.
                        let qb = vld1q_u8(blk.qs.as_ptr());
                        let w_lo = vreinterpretq_s8_u8(vandq_u8(qb, mask_0f));
                        let w_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qb));

                        for (jj, acc_j) in acc.iter_mut().enumerate().take(cols) {
                            let j = j0 + jj;
                            let xp = bq.add(j * k + bi * 32);
                            let x0 = vld1q_s8(xp);
                            let x1 = vld1q_s8(xp.add(16));
                            let dp = vaddvq_s32(vdotq_s32(vdotq_s32(z, w_lo, x0), w_hi, x1));
                            let xs = *bs.add(j * nb + bi);
                            let sx = *cs.add(j * nb + bi);
                            *acc_j += xs * (d * dp as f32 + mmin * sx as f32);
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

    /// Q4_1 × Q8_0 GEMM dispatcher. Requires `dotprod` — like the K-quants, Q4_1 has
    /// no baseline-NEON fallback (the min term reuses [`q8_0_col_sums`], which is
    /// `dotprod`-only). Returns `false` without writing `out` when this CPU cannot run
    /// it, so the caller falls back to the per-token path rather than shipping a wrong
    /// answer. Unlike the K-quants there is no `k % 256` constraint: Q4_1 blocks are 32
    /// wide, so any `k` divisible by 32 (guaranteed by the `gemm_preq` wrapper) is fine.
    #[allow(dead_code)]
    pub unsafe fn gemm_q4_1_q8_0_neon(
        a_quant: &[u8],
        b_scales: &[f32],
        b_quants: &[i8],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        if !k_quant_gemm_available() {
            return false;
        }
        unsafe { gemm_q4_1_q8_0_neon_dotprod(a_quant, b_scales, b_quants, out, m, n, k) };
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

        use crate::backend::simd::require_simd_or_skip;

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

        /// The Q4_1 GEMM must agree with the scalar `vec_dot_q4_1_f32` reference run
        /// column-by-column on the *same* Q8_0-quantized activations. `vec_dot_q4_1_f32`
        /// owns the canonical `d·Σ(q·y) + m·Σy` layout and is independently tested, so a
        /// disagreement is a real bug in the GEMM's nibble decode or min term, not
        /// quantization noise. The tolerance is `1e-4` (not the `1e-5` the Q4_K test can
        /// afford) because the oracle reconstructs f32 activations and dots in float,
        /// whereas the kernel keeps the dot in exact int8 — so their float summation
        /// order genuinely differs. `n = 11` straddles `KQ_COLS` (8): a full 8-column
        /// pass plus a 3-column tail, so a tail bug cannot hide.
        #[test]
        fn q4_1_gemm_matches_scalar_vec_dot() {
            if !require_simd_or_skip("dotprod", cpu_features().dotprod) {
                return;
            }
            let (m, n, k) = (7usize, 11usize, 128usize);
            let nb = k / 32;
            let mut st = 0x0451_1a1au64;
            let blocks: Vec<BlockQ4_1> = (0..m * nb)
                .map(|_| {
                    let mut qs = [0u8; 16];
                    for b in qs.iter_mut() {
                        *b = (lcg(&mut st).abs() * 255.0) as i32 as u8;
                    }
                    BlockQ4_1 {
                        d: f16::from_f32(0.03 + lcg(&mut st).abs() * 0.1).to_bits(),
                        m: f16::from_f32(lcg(&mut st) * 0.5).to_bits(),
                        qs,
                    }
                })
                .collect();
            let a = blocks_to_bytes(&blocks);

            // Column-major activations: column j is b[j*k .. (j+1)*k].
            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut st)).collect();
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let (s, q) = quantize_col(&b[j * k..(j + 1) * k]);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut out = vec![0.0f32; m * n];
            unsafe { gemm_q4_1_q8_0_neon_dotprod(&a, &b_scales, &b_quants, &mut out, m, n, k) };

            // Oracle: reconstruct each activation column from its Q8_0 (scale·quant) and
            // dot it against the Q4_1 weight blocks with the scalar reference.
            for j in 0..n {
                let mut x = vec![0.0f32; k];
                for bi in 0..nb {
                    let s = b_scales[j * nb + bi];
                    for t in 0..32 {
                        x[bi * 32 + t] = s * b_quants[j * k + bi * 32 + t] as f32;
                    }
                }
                for i in 0..m {
                    let mut want = 0.0f32;
                    for bi in 0..nb {
                        want += crate::quant::vec_dot_q4_1_f32(
                            &blocks[i * nb + bi],
                            &x[bi * 32..(bi + 1) * 32],
                        );
                    }
                    let got = out[i * n + j];
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "col {j} row {i}: gemm={got} scalar={want}"
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
// width (16 f32 lanes per op).
//
// HISTORY: these were the x86 hot path back when it was int8-weight ×
// f32-activation. It no longer is. The int8×int8 GEMV that this comment used to
// describe as "a separate, larger change" landed, first for VNNI and then for
// AVX2 via the shared `int8_gemm_kernels!` macro below, and the dispatcher now
// prefers it at every tier from `Avx2` up. The f32 row kernels here therefore
// have no production caller left; they are kept, and exercised by `avx512_tests`,
// only as a measured fallback should the int8 path ever be reverted — see the
// note in `cpu::gemv_q4_0_f32`. Q4_K/Q6_K likewise moved off f32 and onto the
// int8 GEMV (`avx2_int8::gemv_q4k_f32` / `gemv_q6k_f32`).
//
// NOTE: not executable on the aarch64 dev host. Verified by `avx512_tests`
// below, which run only where `is_x86_feature_detected!("avx512f")` holds
// (e.g. Zen 4/5, Skylake-X), comparing each kernel against the scalar reference.
//
// Behind the default-on `avx512` crate feature: disabling it caps x86 at the
// AVX2 tier (`detect()` won't produce `Avx512`). The `_mm512_*` intrinsics need
// Rust 1.89, which is the crate MSRV, so the gate is about hardware coverage,
// not toolchain support.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod avx512 {
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
    // Retained without a production caller: the int8 arms in `cpu.rs` return for
    // every tier from `Avx2` up, so this f32 row-dot is unreachable on any
    // shipping x86 CPU. It is kept (and still exercised by `avx512_tests`)
    // because narrowing the int8 gate would need it back, and because #277/#283
    // tuned it — deleting it would throw that away for a change that is one line
    // to revert. `allow(dead_code)`, not deletion, states that intent.
    #[allow(dead_code)]
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

                // Sum the block's two halves *before* scaling by `d`, so the
                // loop-carried `acc` chain takes one FMA per block instead of
                // two. `acc` is the bottleneck: at ~4-cycle FMA latency, two
                // dependent FMAs put a floor of ~8 cycles on each block.
                //
                // Algebraically identical (`d·a·y + d·b·y == d·(a·y + b·y)`) but
                // not bit-identical — the halves are summed at a different point,
                // so rounding differs. That is why the per-block-sum test carries
                // a tolerance rather than asserting equality.
                let lo_f = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo)), offset);
                let hi_f = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi)), offset);
                let block_sum = _mm512_mul_ps(lo_f, _mm512_loadu_ps(y_ptr));
                let block_sum = _mm512_fmadd_ps(hi_f, _mm512_loadu_ps(y_ptr.add(16)), block_sum);
                acc = _mm512_fmadd_ps(d, block_sum, acc);
            }
            _mm512_reduce_add_ps(acc)
        }
    }

    /// Q8_0 row dot: `<dequant(row), y>` accumulated across `nb` blocks.
    // Retained without a production caller: the int8 arms in `cpu.rs` return for
    // every tier from `Avx2` up, so this f32 row-dot is unreachable on any
    // shipping x86 CPU. It is kept (and still exercised by `avx512_tests`)
    // because narrowing the int8 gate would need it back, and because #277/#283
    // tuned it — deleting it would throw that away for a change that is one line
    // to revert. `allow(dead_code)`, not deletion, states that intent.
    #[allow(dead_code)]
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

                // Written out rather than looped over the two halves so this
                // mirrors `row_dot_q4_0_f32_avx512` line for line — the two
                // kernels differ only in how a block is unpacked, and that is
                // easier to check when their shapes match. Codegen is the same
                // either way: a 2-iteration constant-bound loop unrolls.
                // See `row_dot_q4_0_f32_avx512`: halves summed before scaling
                // by `d`, so `acc` carries one FMA per block rather than two.
                let qf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                    quants_ptr as *const __m128i,
                )));
                let qf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                    quants_ptr.add(16) as *const __m128i,
                )));
                let block_sum = _mm512_mul_ps(qf_lo, _mm512_loadu_ps(y_ptr));
                let block_sum = _mm512_fmadd_ps(qf_hi, _mm512_loadu_ps(y_ptr.add(16)), block_sum);
                acc = _mm512_fmadd_ps(d, block_sum, acc);
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

        use crate::backend::simd::require_simd_or_skip;

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

// ── x86_64 int8 activation kernels (shared by the VNNI and AVX2 tiers) ─────
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
// NOTE: this header covers the SHARED kernel body below (the `int8_gemm_kernels`
// macro), which both the VNNI and the AVX2 instantiations expand. Statements
// about `dpbusd` and about EVEX registers describe the VNNI instantiation; the
// `avx2_int8` module documents where its tier differs. Neither is executable on
// the aarch64 dev host. Most of `avx512_vnni_tests` runs only where
// `is_x86_feature_detected!("avx512vnni")` holds, comparing against the scalar
// reference; elsewhere it skips. The exception is the quantizer group, gated on
// `quantizer_callable()` — that kernel needs no VNNI, and skipping it on a
// non-VNNI AVX-512 host would skip it on the host class it exists for.
// ── Shared int8 GEMM kernels (VNNI / AVX2 instantiations) ───────────────────
//
// Every kernel below is 256-bit AVX2 code. The only architecture-specific
// pieces are the three dot primitives — `dot32` (signed weights, via the
// `_mm256_sign_epi8` trick), `dot32u` (already-unsigned K-quant weights) and
// `dot16u` (the 128-bit half, used by `q8_0_col_sums16`) — plus the
// `#[target_feature]` set and the register-budget-dependent tile constants.
//
// `#[target_feature]` is a compile-time property and a VNNI-enabled function
// will not inline into an AVX2-only one, so the kernels are instantiated once
// per tier rather than branching at runtime.
//
// The macro exists so the two instantiations cannot drift: this body is the
// single definition of the row-tiled Q4_0/Q8_0 GEMM and the Q4_K/Q6_K GEMM.
// Each invoking module supplies `dot32`, `dot32u` and `dot16u` before
// invoking it; omitting one is a compile error far from this comment.
// Only x86_64 instantiates this; without the cfg it is an `unused macro
// definition` warning on every aarch64 and wasm32 build, which the CI lint
// leg (x86 only) would never catch.
#[cfg(target_arch = "x86_64")]
macro_rules! int8_gemm_kernels {
    ($feat:literal, $tile_n:expr, $tile_m:expr, $strip_n:expr, $kq_cols:expr) => {
        /// Horizontal sum of the 8 f32 lanes. Called once per output element, never
        /// inside the block loop — see the module note on accumulation.
        #[target_feature(enable = $feat)]
        unsafe fn hsum256_ps(v: __m256) -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let sum = _mm_add_ps(hi, lo);
            let shuf = _mm_movehl_ps(sum, sum);
            let sum = _mm_add_ps(sum, shuf);
            let shuf = _mm_shuffle_ps(sum, sum, 0x55);
            _mm_cvtss_f32(_mm_add_ss(sum, shuf))
        }

        /// Unpack one Q4_0 block's 16 nibble-pairs into 32 signed bytes in
        /// `[-8, 7]`. Low nibbles are elements 0..16, high nibbles 16..32 — the
        /// same layout the NEON and scalar Q4_0 kernels assume.
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn unpack_q4_0(qs: *const u8) -> __m256i {
            unsafe {
                let qb = _mm_loadu_si128(qs as *const __m128i);
                let mask = _mm_set1_epi8(0x0F);
                let lo = _mm_and_si128(qb, mask);
                let hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
                _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
            }
        }

        // ── Batched prefill GEMM ────────────────────────────────────────────────
        //
        // The point of these over a per-token GEMV loop: one weight row is decoded
        // once and reused across `TILE_N` activation columns, so a prefill of `n`
        // tokens streams the weight matrix `n / TILE_N` times instead of `n`, and the
        // decode is amortized across the tile.
        //
        // NOTE: these per-row kernels are no longer the production path for the
        // Q4_0/Q8_0 GEMMs — see "Row tiling" below, which processes `TILE_M` rows per
        // task and reaches them only for the `m % TILE_M` tail. Read that block for
        // the current cost model; in particular, "weight-bandwidth bound" is the
        // wrong summary at these shapes (the activation panel is L2-resident and the
        // binding constraint is loads and uops per `dot32`, not DRAM).
        //
        // Column-major activations: `quantize_columns` packs column `j` contiguously
        // at `b_quants[j * k ..]` with scales at `b_scales[j * nb ..]`, so each of the
        // `TILE_N` loads below is a straight 32-byte read.

        /// Dot one Q4_0 weight row against the pre-quantized activation.
        #[target_feature(enable = $feat)]
        unsafe fn row_dot_q4_0(
            row: *const u8,
            x_scales: &[f32],
            x_quants: &[i8],
            nb: usize,
        ) -> f32 {
            unsafe {
                let bsz = size_of::<BlockQ4_0>();
                let mut acc = _mm256_setzero_ps();
                for b in 0..nb {
                    let block = &*(row.add(b * bsz) as *const BlockQ4_0);
                    let w = unpack_q4_0(block.qs.as_ptr());
                    let a = _mm256_loadu_si256(x_quants.as_ptr().add(b * 32) as *const __m256i);
                    let scale = _mm256_set1_ps(
                        f16::from_bits(block.d).to_f32() * *x_scales.get_unchecked(b),
                    );
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc);
                }
                hsum256_ps(acc)
            }
        }

        /// Dot one Q8_0 weight row against the pre-quantized activation.
        #[target_feature(enable = $feat)]
        unsafe fn row_dot_q8_0(
            row: *const u8,
            x_scales: &[f32],
            x_quants: &[i8],
            nb: usize,
        ) -> f32 {
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
        #[target_feature(enable = $feat)]
        pub unsafe fn gemv_q4_0_q8_0(
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
        #[target_feature(enable = $feat)]
        pub unsafe fn gemv_q8_0_q8_0(
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

        /// Activation columns processed per weight decode, amortizing the decode
        /// across the tile.
        ///
        /// NOTE: this doc is shared by both instantiations, and every measured
        /// number and register-count claim below is the **VNNI** one's. The AVX2
        /// instantiation runs under VEX with 16 ymm registers and takes its own,
        /// narrower value — see its invocation site.
        ///
        /// For VNNI, this was 4, on the reasoning that 8 would spill "the 16
        /// available ymm registers". That premise is wrong there: those kernels are
        /// inside `#[target_feature(...)]` enabling AVX-512, so EVEX exposes 32
        /// vector registers, not 16. 8 accumulators plus the decoded weight fit
        /// comfortably. (16 genuinely does spill, so the concern was real, just off
        /// by 2x.) On AVX2 the original 16-register premise does hold, which is why
        /// the two instantiations disagree.
        ///
        /// 8 measures faster than 4 on both dtypes. Interleaved A/B, 8 paired
        /// rounds, 2048x512x2048, rayon pool fixed at 16 threads: Q4_0 626->660
        /// GOP/s (+5.6%, paired t=7.15, 8/8 rounds), Q8_0 563->638 (+13.4%, t=16.81,
        /// 8/8).
        ///
        /// CAVEAT: `microbench_gemm` can no longer reproduce that. It drives the
        /// public GEMMs at m=2048, and since row tiling landed those dispatch every
        /// full `TILE_M`-row strip to `gemm_*_strip` (which tiles by `STRIP_N`), so
        /// `TILE_N` has no effect at that shape — the header it prints names a
        /// constant it is not testing. `TILE_N` now governs only the `m % TILE_M`
        /// tail, and production out-feature counts are multiples of 4, so it is
        /// effectively unexercised in production. To re-tune it, drive
        /// `gemm_*_row` directly or pick an `m` with `m % TILE_M != 0`.
        ///
        /// Getting a trustworthy number here took two tries, and the method is worth
        /// recording. Unpinned, this host's run-to-run spread is ~2.8x (an identical
        /// binary spanned 417-1154 GOP/s), because the GEMM parallelises over rows
        /// and rayon's pool size/placement varies per process; pinning the pool
        /// collapses that to ~9%. A first pass also left one arm at a ~0.2s window,
        /// which on a boosting CPU just samples whatever clock state it landed in.
        /// Both together made noise ~5x the effect and produced a false regression.
        /// Interleaving arms controls for drift *between* arms; it does nothing about
        /// a too-short window or per-process variance in the pool. Interleaving is
        /// necessary, not sufficient — pin the pool and report the paired statistic.
        ///
        /// The tile width is still not where this kernel's time goes: it runs at
        /// roughly 4% of this host's int8 peak. Each `dpbusd` drags along an
        /// int32->float convert, a scalar load-multiply-broadcast of the combined
        /// weight/activation scale, and a float FMA.
        ///
        /// Transposing `b_scales` to `[block][col]` is the obvious idea and does
        /// *not* remove that scalar work: `vpdpbusd` yields 8 int32 lanes that are
        /// partial sums of the *same* dot product, so the scale is a per-column
        /// broadcast (`set1`), not a vector of 8 distinct column scales — the
        /// transpose can only make the tiny scale loads more contiguous, which is
        /// not the bottleneck. The structural fix was row tiling — feed each
        /// activation load to several weight rows, as the GPU kernel does (#267) —
        /// and that has since shipped: see "Row tiling" below, which also lists the
        /// headroom that remains.
        const TILE_N: usize = $tile_n;

        /// One output row of the Q4_0 GEMM: `out[j] = <A_row, B_col_j>` for all `n`.
        #[target_feature(enable = $feat)]
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
                        let a = _mm256_loadu_si256(
                            b_quants.as_ptr().add(j * k + b * 32) as *const __m256i
                        );
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
        #[target_feature(enable = $feat)]
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
                        let a = _mm256_loadu_si256(
                            b_quants.as_ptr().add(j * k + b * 32) as *const __m256i
                        );
                        let scale = _mm256_set1_ps(
                            f16::from_bits(block.delta).to_f32()
                                * *b_scales.get_unchecked(j * nb + b),
                        );
                        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc);
                    }
                    *out_row.get_unchecked_mut(j) = hsum256_ps(acc);
                    j += 1;
                }
            }
        }

        // ── Row tiling ──────────────────────────────────────────────────────────
        //
        // The per-row kernels above parallelise one weight row per task and re-read
        // the entire `n*k` activation panel for every row.
        //
        // What that costs is **load-port pressure, not DRAM bandwidth** — say this
        // precisely, because the obvious bandwidth story is wrong and misleads the
        // next tuning pass. At the benchmark shape (2048x512x2048) the activation
        // panel is n*k + n*nb*4 = 1.18 MB, so it is L2-resident and those re-reads
        // are cache hits; the only thing streaming from memory is ~4.3 MB of weights,
        // read once. The per-row TILE_N=8 kernel issues 1 weight + 1 f16 + 8
        // activation + 8 scale loads per 8 `dot32` (2.25 loads/dot32); a 4x4 strip
        // issues 4+4+4+4 per 16 (1.00), and carries 16 independent accumulator chains
        // instead of 8 to hide `vpdpbusd` latency. That is the mechanism.
        //
        // The arithmetic above uses the VNNI instantiation's 8/4/4; substitute the
        // AVX2 one's 4/2/4 for that tier. The mechanism is the same, the ratios
        // are not.
        //
        // Measured against the per-row driver by `microbench_gemm_rowtile`, 8 paired
        // rounds with alternating arm order, pseudo-random activations, on an idle
        // host — three repetitions: **Q8_0 +19.1/+24.1/+21.8%, Q4_0
        // +19.8/+18.9/+19.0%, 8/8 rounds won in all six** (~250 -> ~305 GOP/s).
        //
        // Two measurement notes, both learned the hard way here. Constant-filled
        // activations inflate *absolute* throughput ~2x (both arms run out of a
        // trivially-predictable working set) and distort the ratio, so the benchmark
        // quantizes real pseudo-random columns. And a contended machine collapses the
        // effect to low single digits while looking like a valid run — take these
        // numbers on an otherwise idle host or not at all.
        //
        // Output is bit-identical to the per-row path. That is enforced per
        // instantiation, not by the ignored benchmark:
        // `gemm_avx512_row_tiled_matches_per_row_bit_exact` at the VNNI tile
        // constants and `avx2_row_tiled_matches_per_row_bit_exact` at the AVX2
        // ones — the latter needs no VNNI host, which matters because the tile
        // constants it pins are the ones most x86 hosts actually run.
        //
        // Known headroom, in the order worth attacking:
        //   1. Weight decode is redone per column tile (the 2x factor below is the
        //      VNNI instantiation's; at the AVX2 constants `TILE_N == STRIP_N`, so
        //      there is no regression to recover there) — `w`/`dw` are hoisted out of
        //      the `t` loop but not out of the `j` loop, so a row's blocks are decoded
        //      n/STRIP_N times against a theoretical `nb`. That is 2x more f16->f32
        //      converts and `unpack_q4_0` calls than the TILE_N=8 per-row kernel did,
        //      and is the likeliest reason Q4_0 (which pays the nibble unpack) gains
        //      less than Q8_0. A per-strip decode buffer is 8 KB, L1-resident.
        //   2. `_mm256_set1_ps(dw[r] * da)` sits in the innermost loop: 16 broadcasts
        //      per (block, tile) where 8 would do, on a port that is already busy.
        //   3. No blocking over `n`. Each strip still walks the whole panel, so at
        //      large `k` (e.g. ffn_down, k=8192) the panel leaves L2 and the real
        //      bandwidth story finally does apply.
        //
        // The trade-off is granularity: this divides the parallel task count by
        // `TILE_M`. `m` is a projection's out-feature count, and the small end is a
        // GQA kv_dim — 128 for a 2-KV-head model, i.e. 32 strips against this host's
        // 32 workers, one per worker with no stealing slack (an MQA kv_dim of 64
        // leaves half the pool idle). The strip still wins at those shapes (measured
        // +184%/+103%/+35% at m=64/128/512, 6/6 rounds) because they are nowhere near
        // thread-bound, but the pool is underfed there and a 2-D split over strips x
        // n-panels is the fix if that ever matters.

        /// Weight rows per strip.
        const TILE_M: usize = $tile_m;

        /// Activation columns per accumulator tile inside a strip. `TILE_M *
        /// STRIP_N` fp32 accumulators must fit the register file with room for the
        /// `TILE_M` decoded weights, the shared activation, and temporaries: 4x4 =
        /// 16 leaves half of EVEX's 32 registers free and measured fastest — for
        /// the VNNI instantiation. The AVX2 one has 16 VEX registers total and
        /// uses 2x4; its invocation site carries that reasoning. The
        /// 24-accumulator shapes (6x4, 4x6) spill and regress; this is distinct from
        /// the per-row `TILE_N = 8`, which tiles one row against 8 columns.
        const STRIP_N: usize = $strip_n;

        /// One strip of `TILE_M` consecutive Q8_0 weight rows against all `n`
        /// columns. `rows` points at the first row; rows are `row_bytes` apart and
        /// `out` is `TILE_M * n` row-major.
        #[target_feature(enable = $feat)]
        unsafe fn gemm_q8_0_strip(
            rows: *const u8,
            row_bytes: usize,
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            n: usize,
            nb: usize,
        ) {
            // The kernel indexes `out` unchecked up to `TILE_M * n - 1` and reads
            // `TILE_M` whole weight rows behind `rows`, so state the output half of
            // that contract here rather than relying on the caller's arithmetic.
            debug_assert_eq!(out.len(), TILE_M * n);
            debug_assert_eq!(b_quants.len(), n * nb * 32);
            debug_assert_eq!(b_scales.len(), n * nb);
            unsafe {
                let bsz = size_of::<BlockQ8_0>();
                let k = nb * 32;
                let mut j = 0;
                // The tile loops carry numeric meaning beyond the index (`col = j +
                // t`, weight offset `r * row_bytes`, output offset `r * n + j + t`),
                // so range loops read more directly than zipped iterators.
                #[allow(clippy::needless_range_loop)]
                while j + STRIP_N <= n {
                    let mut acc = [[_mm256_setzero_ps(); STRIP_N]; TILE_M];
                    for b in 0..nb {
                        // Decode the TILE_M weight blocks once, reused across cols.
                        let mut w = [_mm256_setzero_si256(); TILE_M];
                        let mut dw = [0.0f32; TILE_M];
                        for r in 0..TILE_M {
                            let block = &*(rows.add(r * row_bytes + b * bsz) as *const BlockQ8_0);
                            w[r] = _mm256_loadu_si256(block.quants.as_ptr() as *const __m256i);
                            dw[r] = f16::from_bits(block.delta).to_f32();
                        }
                        for t in 0..STRIP_N {
                            let col = j + t;
                            // One activation load, fed to every row in the strip.
                            let a = _mm256_loadu_si256(
                                b_quants.as_ptr().add(col * k + b * 32) as *const __m256i
                            );
                            let da = *b_scales.get_unchecked(col * nb + b);
                            for r in 0..TILE_M {
                                let prod = _mm256_cvtepi32_ps(dot32(w[r], a));
                                let scale = _mm256_set1_ps(dw[r] * da);
                                acc[r][t] = _mm256_fmadd_ps(prod, scale, acc[r][t]);
                            }
                        }
                    }
                    for r in 0..TILE_M {
                        for t in 0..STRIP_N {
                            *out.get_unchecked_mut(r * n + j + t) = hsum256_ps(acc[r][t]);
                        }
                    }
                    j += STRIP_N;
                }
                // Column remainder (n % STRIP_N). The block loop is outermost so the
                // single activation load is still shared across the strip's rows and
                // the TILE_M accumulator chains stay independent — the same reuse the
                // tile loop gets, just one column wide. Per (row, column) the fmadd
                // order over `b` is unchanged, so this stays bit-identical to the
                // per-row kernel.
                #[allow(clippy::needless_range_loop)]
                while j < n {
                    let mut acc = [_mm256_setzero_ps(); TILE_M];
                    for b in 0..nb {
                        let a = _mm256_loadu_si256(
                            b_quants.as_ptr().add(j * k + b * 32) as *const __m256i
                        );
                        let da = *b_scales.get_unchecked(j * nb + b);
                        for r in 0..TILE_M {
                            let block = &*(rows.add(r * row_bytes + b * bsz) as *const BlockQ8_0);
                            let w = _mm256_loadu_si256(block.quants.as_ptr() as *const __m256i);
                            let scale = _mm256_set1_ps(f16::from_bits(block.delta).to_f32() * da);
                            acc[r] =
                                _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc[r]);
                        }
                    }
                    for (r, acc_r) in acc.iter().enumerate() {
                        *out.get_unchecked_mut(r * n + j) = hsum256_ps(*acc_r);
                    }
                    j += 1;
                }
            }
        }

        /// One strip of `TILE_M` consecutive Q4_0 weight rows. Mirrors
        /// `gemm_q8_0_strip`; the only differences are the block type and the nibble
        /// unpack that produces each weight register.
        #[target_feature(enable = $feat)]
        unsafe fn gemm_q4_0_strip(
            rows: *const u8,
            row_bytes: usize,
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            n: usize,
            nb: usize,
        ) {
            // Same unchecked contract as `gemm_q8_0_strip`.
            debug_assert_eq!(out.len(), TILE_M * n);
            debug_assert_eq!(b_quants.len(), n * nb * 32);
            debug_assert_eq!(b_scales.len(), n * nb);
            unsafe {
                let bsz = size_of::<BlockQ4_0>();
                let k = nb * 32;
                let mut j = 0;
                #[allow(clippy::needless_range_loop)]
                while j + STRIP_N <= n {
                    let mut acc = [[_mm256_setzero_ps(); STRIP_N]; TILE_M];
                    for b in 0..nb {
                        let mut w = [_mm256_setzero_si256(); TILE_M];
                        let mut dw = [0.0f32; TILE_M];
                        for r in 0..TILE_M {
                            let block = &*(rows.add(r * row_bytes + b * bsz) as *const BlockQ4_0);
                            w[r] = unpack_q4_0(block.qs.as_ptr());
                            dw[r] = f16::from_bits(block.d).to_f32();
                        }
                        for t in 0..STRIP_N {
                            let col = j + t;
                            let a = _mm256_loadu_si256(
                                b_quants.as_ptr().add(col * k + b * 32) as *const __m256i
                            );
                            let da = *b_scales.get_unchecked(col * nb + b);
                            for r in 0..TILE_M {
                                let prod = _mm256_cvtepi32_ps(dot32(w[r], a));
                                let scale = _mm256_set1_ps(dw[r] * da);
                                acc[r][t] = _mm256_fmadd_ps(prod, scale, acc[r][t]);
                            }
                        }
                    }
                    for r in 0..TILE_M {
                        for t in 0..STRIP_N {
                            *out.get_unchecked_mut(r * n + j + t) = hsum256_ps(acc[r][t]);
                        }
                    }
                    j += STRIP_N;
                }
                // Column remainder — block loop outermost, as in `gemm_q8_0_strip`.
                // `r` indexes both `acc` and the weight-row offset, so a range loop
                // is the direct spelling here.
                #[allow(clippy::needless_range_loop)]
                while j < n {
                    let mut acc = [_mm256_setzero_ps(); TILE_M];
                    for b in 0..nb {
                        let a = _mm256_loadu_si256(
                            b_quants.as_ptr().add(j * k + b * 32) as *const __m256i
                        );
                        let da = *b_scales.get_unchecked(j * nb + b);
                        for r in 0..TILE_M {
                            let block = &*(rows.add(r * row_bytes + b * bsz) as *const BlockQ4_0);
                            let w = unpack_q4_0(block.qs.as_ptr());
                            let scale = _mm256_set1_ps(f16::from_bits(block.d).to_f32() * da);
                            acc[r] =
                                _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(w, a)), scale, acc[r]);
                        }
                    }
                    for (r, acc_r) in acc.iter().enumerate() {
                        *out.get_unchecked_mut(r * n + j) = hsum256_ps(*acc_r);
                    }
                    j += 1;
                }
            }
        }

        /// Batched Q4_0 × Q8_0 GEMM: `out[m,n] = A_q4_0[m,k] @ B_q8_0[k,n]`,
        /// parallel over strips of `TILE_M` output rows.
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q4_0_q8_0(
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

            // One `TILE_M`-row strip per chunk; the final chunk may be short
            // (`m % TILE_M`) and is finished row-by-row.
            //
            // This must stay a closure inside this `#[target_feature]` fn: closures
            // inherit the enclosing function's feature set, so the strip and row
            // kernels inline here with this instantiation's codegen. Hoisting it
            // to a free `fn` would silently drop the features and un-inline both
            // kernels.
            let handle = |out_chunk: &mut [f32], s: usize| {
                // Strip `s` owns rows `s * TILE_M ..` — `rows_here` of them, which is
                // `TILE_M` for every chunk but a short final one. Compare against the
                // exact byte length rather than a truncating division so a chunk that
                // is not a whole number of rows takes the row path instead of
                // silently entering the strip kernel.
                let rows_here = out_chunk.len() / n;
                // SAFETY: strip `s` reads `a_quant[s * TILE_M * row_bytes ..]` for
                // `rows_here * row_bytes` bytes — up to `TILE_M * row_bytes`, not one
                // row — which is in bounds because `out.len() == m * n` bounds `s` and
                // `a_quant.len() == m * row_bytes`. Reads are shared and read-only;
                // the write goes only to this task's disjoint `out_chunk`.
                unsafe {
                    let rows = a_quant.as_ptr().add(s * TILE_M * row_bytes);
                    if out_chunk.len() == TILE_M * n {
                        gemm_q4_0_strip(rows, row_bytes, b_scales, b_quants, out_chunk, n, nb);
                    } else {
                        debug_assert_eq!(out_chunk.len(), rows_here * n);
                        for (r, out_row) in out_chunk.chunks_mut(n).enumerate() {
                            gemm_q4_0_row(
                                rows.add(r * row_bytes),
                                b_scales,
                                b_quants,
                                out_row,
                                n,
                                nb,
                            );
                        }
                    }
                }
            };

            #[cfg(feature = "parallel")]
            {
                use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
                out.par_chunks_mut(TILE_M * n)
                    .enumerate()
                    .for_each(|(s, out_chunk)| handle(out_chunk, s));
            }
            #[cfg(not(feature = "parallel"))]
            for (s, out_chunk) in out.chunks_mut(TILE_M * n).enumerate() {
                handle(out_chunk, s);
            }
        }

        // ── Repacked Q4_0 prefill GEMM (8-row interleave) ───────────────────
        //
        // `gemm_q4_0_q8_0` above produces, per `dot32`, 8 int32 lanes that are
        // partial sums of ONE (row, col) dot, so it reduces them — deferred into
        // `facc`, one `hsum256_ps` per column. This kernel consumes a weight
        // layout (`repack_q4_0_8x8`, built once at load) that interleaves 8 rows
        // so the 8 lanes of one `dpbusd` are 8 DISTINCT OUTPUT ROWS for a shared
        // column. That removes the hsum entirely: the reduction becomes a plain
        // lane-per-row accumulate, weights load contiguously, and each 4-element
        // activation group is broadcast with one `set1_epi32`.
        //
        // Q4_0's `d·(q−8)` splits the same way the Q4_K mins term does: keep the
        // nibbles unsigned (`q` in 0..15) for `dpbusd`, and carry the `−8` as a
        // per-(column, block) `−8·d_row·d_act·Σa` correction (`sum_a` from
        // `q8_0_col_sums`). So `eff = Σqa − 8·Σa` in float, then one
        // `fmadd(eff, scale8, facc)` per block, where `scale8` is the 8 rows'
        // weight scales times the activation scale.
        //
        // The packed layout stores the nibbles (128 bytes per (super-row,
        // block) = 8× a `BlockQ4_0`'s 16 `qs` bytes, so the *nibble* footprint
        // is unchanged; the scales are stored separately, as f32, by
        // `repack_q4_0_8x8`). Byte `i` of a k-group's 16 packed bytes pairs row
        // `i/4` (low nibble) with row `i/4 + 4` (high nibble) at k-element
        // `i%4`, which is exactly what the standard low/high nibble unpack
        // (`set_m128i(hi, lo)`) reassembles into the 32-byte, lane-per-row
        // weight vector.
        //
        // `m % 8 == 0` is a precondition — `q4_0_repack_supported` gates the
        // repack at load, so a weight with a ragged row count is never repacked
        // and takes the standard kernel instead.
        //
        // `allow(dead_code)` under `blas`: the sole caller
        // `gemm_preq_repacked_q4_0_dispatch` is `cfg(not(blas))`, so a non-test
        // `--features blas` lib build has no caller (the tests still do). Same
        // pattern the other `blas`-dead kernels here carry.
        #[cfg_attr(feature = "blas", allow(dead_code))]
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q4_0_8x8_q8_0(
            packed: &[u8],
            scales: &[f32],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) {
            debug_assert_eq!(m % 8, 0, "repacked Q4_0 GEMM: m must be a multiple of 8");
            debug_assert_eq!(k % 32, 0, "repacked Q4_0 GEMM: k must be divisible by 32");
            let nb = k / 32;
            debug_assert_eq!(out.len(), m * n);
            debug_assert_eq!(packed.len(), (m / 8) * nb * 128);
            debug_assert_eq!(scales.len(), (m / 8) * nb * 8);
            debug_assert_eq!(b_quants.len(), n * k);
            debug_assert_eq!(b_scales.len(), n * nb);

            // Per-(column, block) activation sums for the `−8` offset correction.
            let col_sums = unsafe { q8_0_col_sums(b_quants, n, k) };
            let cs: &[i32] = &col_sums;
            let mask = _mm_set1_epi8(0x0F);

            // Unpack one k-group's 16 packed bytes into the 32-byte, lane-per-row
            // unsigned weight vector (rows 0..3 in the low 128, 4..7 in the high).
            let unpack = |p: *const u8| -> __m256i {
                unsafe {
                    let qb = _mm_loadu_si128(p as *const __m128i);
                    let lo = _mm_and_si128(qb, mask);
                    let hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
                    _mm256_set_m128i(hi, lo)
                }
            };

            // One accumulator strip = the 8 output rows of a super-row, across a
            // tile of `TILE_N` columns (weight decode amortized over the tile).
            let compute = |sr: usize, chunk: &mut [f32]| unsafe {
                // `chunk[r*n + j] == out[(8*sr + r)*n + j]`.
                let mut j = 0;
                while j + TILE_N <= n {
                    let mut facc = [_mm256_setzero_ps(); TILE_N];
                    for b in 0..nb {
                        let mut acc = [_mm256_setzero_si256(); TILE_N];
                        let pbase = (sr * nb + b) * 128;
                        for g in 0..8usize {
                            let w = unpack(packed.as_ptr().add(pbase + g * 16));
                            for (t, acc_t) in acc.iter_mut().enumerate() {
                                let col = j + t;
                                let a4 = (b_quants.as_ptr().add(col * k + b * 32 + g * 4)
                                    as *const i32)
                                    .read_unaligned();
                                *acc_t = _mm256_add_epi32(*acc_t, dot32u(w, _mm256_set1_epi32(a4)));
                            }
                        }
                        let d_row8 = _mm256_loadu_ps(scales.as_ptr().add((sr * nb + b) * 8));
                        for (t, facc_t) in facc.iter_mut().enumerate() {
                            let col = j + t;
                            let d_a = *b_scales.get_unchecked(col * nb + b);
                            let scale8 = _mm256_mul_ps(d_row8, _mm256_set1_ps(d_a));
                            let sum_a = *cs.get_unchecked(col * nb + b);
                            let eff = _mm256_sub_ps(
                                _mm256_cvtepi32_ps(acc[t]),
                                _mm256_set1_ps(8.0 * sum_a as f32),
                            );
                            *facc_t = _mm256_fmadd_ps(eff, scale8, *facc_t);
                        }
                    }
                    for (t, facc_t) in facc.iter().enumerate() {
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), *facc_t);
                        for (r, &v) in tmp.iter().enumerate() {
                            *chunk.get_unchecked_mut(r * n + j + t) = v;
                        }
                    }
                    j += TILE_N;
                }
                // Column remainder (`n % TILE_N`), one column at a time.
                while j < n {
                    let mut facc = _mm256_setzero_ps();
                    for b in 0..nb {
                        let mut acc = _mm256_setzero_si256();
                        let pbase = (sr * nb + b) * 128;
                        for g in 0..8usize {
                            let w = unpack(packed.as_ptr().add(pbase + g * 16));
                            let a4 = (b_quants.as_ptr().add(j * k + b * 32 + g * 4) as *const i32)
                                .read_unaligned();
                            acc = _mm256_add_epi32(acc, dot32u(w, _mm256_set1_epi32(a4)));
                        }
                        let d_row8 = _mm256_loadu_ps(scales.as_ptr().add((sr * nb + b) * 8));
                        let d_a = *b_scales.get_unchecked(j * nb + b);
                        let scale8 = _mm256_mul_ps(d_row8, _mm256_set1_ps(d_a));
                        let sum_a = *cs.get_unchecked(j * nb + b);
                        let eff = _mm256_sub_ps(
                            _mm256_cvtepi32_ps(acc),
                            _mm256_set1_ps(8.0 * sum_a as f32),
                        );
                        facc = _mm256_fmadd_ps(eff, scale8, facc);
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), facc);
                    for (r, &v) in tmp.iter().enumerate() {
                        *chunk.get_unchecked_mut(r * n + j) = v;
                    }
                    j += 1;
                }
            };

            #[cfg(feature = "parallel")]
            {
                use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
                out.par_chunks_mut(8 * n)
                    .enumerate()
                    .for_each(|(sr, chunk)| compute(sr, chunk));
            }
            #[cfg(not(feature = "parallel"))]
            for (sr, chunk) in out.chunks_mut(8 * n).enumerate() {
                compute(sr, chunk);
            }
        }

        /// Batched Q4_K × Q8_0 GEMM on the 8-row-interleaved layout
        /// (`repack_q4_k_8x8`): `out[m,n] = A_q4_k[m,k] @ B_q8_0[k,n]`.
        ///
        /// The Q4_K twin of `gemm_q4_0_8x8_q8_0`. A Q4_K super-block is 8
        /// sub-blocks of 32, so its `k/32` 32-element blocks index the packed
        /// nibbles exactly as Q4_0's do, and the 8 `dpbusd` lanes of a k-group are
        /// 8 output rows for a shared column — no per-column hsum. What differs is
        /// the reduction: each 32-block carries its own per-row scale
        /// `dsc = d·sc_s` and min `dmn = dmin·mn_s` (baked at repack), so the
        /// contribution is `xs·(dsc·Σqa − dmn·Σa)` per row, with `Σa` the column
        /// sum (`q8_0_col_sums`) and `xs` the activation scale. Both terms are now
        /// per-row *vectors* — the mins term is a scalar in the standard kernel,
        /// where a column resolves to one row.
        ///
        /// `allow(dead_code)` under `blas`: the sole caller
        /// `gemm_preq_repacked_q4_k_dispatch` is `cfg(not(blas))`.
        #[cfg_attr(feature = "blas", allow(dead_code))]
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q4_k_8x8_q8_0(
            packed: &[u8],
            dsc: &[f32],
            dmn: &[f32],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) {
            debug_assert_eq!(m % 8, 0, "repacked Q4_K GEMM: m must be a multiple of 8");
            debug_assert_eq!(k % 256, 0, "repacked Q4_K GEMM: k must be divisible by 256");
            let nb32 = k / 32;
            debug_assert_eq!(out.len(), m * n);
            debug_assert_eq!(packed.len(), (m / 8) * nb32 * 128);
            debug_assert_eq!(dsc.len(), (m / 8) * nb32 * 8);
            debug_assert_eq!(dmn.len(), (m / 8) * nb32 * 8);
            debug_assert_eq!(b_quants.len(), n * k);
            debug_assert_eq!(b_scales.len(), n * nb32);

            // Per-(column, 32-block) activation sums for the mins term.
            let col_sums = unsafe { q8_0_col_sums(b_quants, n, k) };
            let cs: &[i32] = &col_sums;
            let mask = _mm_set1_epi8(0x0F);

            // Unpack one k-group's 16 packed bytes into the 32-byte, lane-per-row
            // unsigned weight vector (rows 0..3 in the low 128, 4..7 in the high).
            let unpack = |p: *const u8| -> __m256i {
                unsafe {
                    let qb = _mm_loadu_si128(p as *const __m128i);
                    let lo = _mm_and_si128(qb, mask);
                    let hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
                    _mm256_set_m128i(hi, lo)
                }
            };

            // One accumulator strip = the 8 output rows of a super-row, across a
            // tile of `TILE_N` columns.
            let compute = |sr: usize, chunk: &mut [f32]| unsafe {
                // `chunk[r*n + j] == out[(8*sr + r)*n + j]`.
                let mut j = 0;
                while j + TILE_N <= n {
                    let mut facc = [_mm256_setzero_ps(); TILE_N];
                    for block in 0..nb32 {
                        let mut acc = [_mm256_setzero_si256(); TILE_N];
                        let pbase = (sr * nb32 + block) * 128;
                        for g in 0..8usize {
                            let w = unpack(packed.as_ptr().add(pbase + g * 16));
                            for (t, acc_t) in acc.iter_mut().enumerate() {
                                let col = j + t;
                                let a4 = (b_quants.as_ptr().add(col * k + block * 32 + g * 4)
                                    as *const i32)
                                    .read_unaligned();
                                *acc_t = _mm256_add_epi32(*acc_t, dot32u(w, _mm256_set1_epi32(a4)));
                            }
                        }
                        let sbase = (sr * nb32 + block) * 8;
                        let dsc8 = _mm256_loadu_ps(dsc.as_ptr().add(sbase));
                        let dmn8 = _mm256_loadu_ps(dmn.as_ptr().add(sbase));
                        for (t, facc_t) in facc.iter_mut().enumerate() {
                            let col = j + t;
                            let xs = *b_scales.get_unchecked(col * nb32 + block);
                            let sx = *cs.get_unchecked(col * nb32 + block);
                            // facc += xs·dsc8·acc − xs·dmn8·sx, per row lane.
                            let eff = _mm256_mul_ps(dsc8, _mm256_set1_ps(xs));
                            *facc_t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[t]), eff, *facc_t);
                            *facc_t =
                                _mm256_fnmadd_ps(dmn8, _mm256_set1_ps(xs * sx as f32), *facc_t);
                        }
                    }
                    for (t, facc_t) in facc.iter().enumerate() {
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), *facc_t);
                        for (r, &v) in tmp.iter().enumerate() {
                            *chunk.get_unchecked_mut(r * n + j + t) = v;
                        }
                    }
                    j += TILE_N;
                }
                // Column remainder (`n % TILE_N`), one column at a time.
                while j < n {
                    let mut facc = _mm256_setzero_ps();
                    for block in 0..nb32 {
                        let mut acc = _mm256_setzero_si256();
                        let pbase = (sr * nb32 + block) * 128;
                        for g in 0..8usize {
                            let w = unpack(packed.as_ptr().add(pbase + g * 16));
                            let a4 = (b_quants.as_ptr().add(j * k + block * 32 + g * 4)
                                as *const i32)
                                .read_unaligned();
                            acc = _mm256_add_epi32(acc, dot32u(w, _mm256_set1_epi32(a4)));
                        }
                        let sbase = (sr * nb32 + block) * 8;
                        let dsc8 = _mm256_loadu_ps(dsc.as_ptr().add(sbase));
                        let dmn8 = _mm256_loadu_ps(dmn.as_ptr().add(sbase));
                        let xs = *b_scales.get_unchecked(j * nb32 + block);
                        let sx = *cs.get_unchecked(j * nb32 + block);
                        let eff = _mm256_mul_ps(dsc8, _mm256_set1_ps(xs));
                        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), eff, facc);
                        facc = _mm256_fnmadd_ps(dmn8, _mm256_set1_ps(xs * sx as f32), facc);
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), facc);
                    for (r, &v) in tmp.iter().enumerate() {
                        *chunk.get_unchecked_mut(r * n + j) = v;
                    }
                    j += 1;
                }
            };

            #[cfg(feature = "parallel")]
            {
                use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
                out.par_chunks_mut(8 * n)
                    .enumerate()
                    .for_each(|(sr, chunk)| compute(sr, chunk));
            }
            #[cfg(not(feature = "parallel"))]
            for (sr, chunk) in out.chunks_mut(8 * n).enumerate() {
                compute(sr, chunk);
            }
        }

        /// Batched Q8_0 × Q8_0 GEMM: `out[m,n] = A_q8_0[m,k] @ B_q8_0[k,n]`.
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q8_0_q8_0(
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

            // One `TILE_M`-row strip per chunk; short final chunk row-by-row. Must
            // stay a closure for the target-feature reason given in the Q4_0 GEMM.
            let handle = |out_chunk: &mut [f32], s: usize| {
                let rows_here = out_chunk.len() / n;
                // SAFETY: as in the Q4_0 GEMM above — up to `TILE_M * row_bytes`.
                unsafe {
                    let rows = a_quant.as_ptr().add(s * TILE_M * row_bytes);
                    if out_chunk.len() == TILE_M * n {
                        gemm_q8_0_strip(rows, row_bytes, b_scales, b_quants, out_chunk, n, nb);
                    } else {
                        debug_assert_eq!(out_chunk.len(), rows_here * n);
                        for (r, out_row) in out_chunk.chunks_mut(n).enumerate() {
                            gemm_q8_0_row(
                                rows.add(r * row_bytes),
                                b_scales,
                                b_quants,
                                out_row,
                                n,
                                nb,
                            );
                        }
                    }
                }
            };

            #[cfg(feature = "parallel")]
            {
                use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
                out.par_chunks_mut(TILE_M * n)
                    .enumerate()
                    .for_each(|(s, out_chunk)| handle(out_chunk, s));
            }
            #[cfg(not(feature = "parallel"))]
            for (s, out_chunk) in out.chunks_mut(TILE_M * n).enumerate() {
                handle(out_chunk, s);
            }
        }

        // ── K-quant int8 kernels ────────────────────────────────────────────────
        //
        // Like the Q4_0/Q8_0 GEMMs above, these defer the `vpdpbusd`/`dot32u`
        // reduction: the 8 dot lanes stay in a per-column float accumulator
        // (`facc`), scaled per sub-block, and collapse with a single `hsum256_ps`
        // per column at the end — not an hsum per (sub-block, column). The Q4_K
        // mins term and the Q6_K −32 recentering carry no dot lanes, so they
        // accumulate as scalars (`macc`) alongside.
        //
        // Still NOT row-tiled. The Q4_0/Q8_0 GEMMs process `TILE_M` weight rows per
        // task ("Row tiling"); these stay per-row. A row-tiled Q4_K prototype once
        // measured **-9.3% at 4x8 and -17% at 4x4, 0/8 rounds won** — but that was
        // against the *old* per-block-hsum structure, whose "no vector-accumulator
        // ILP to reuse" premise this deferral removed, so the number no longer
        // describes the kernel; row-tiling on the deferred structure is simply
        // unmeasured. Each sub-block still carries a per-sub-block scale and a mins
        // correction (heavier per-block work than Q4_0/Q8_0), and the K-quant GEMV
        // *is* this GEMM at n=1, so any tiling here would also have to not regress
        // decode.
        //
        // x86 analogue of the NEON K-quant GEMM family. Two structural differences,
        // both forced by `vpdpbusd` taking an *unsigned* × signed operand pair:
        //
        //  - Q4_K nibbles (0..15) and Q6_K quants (0..63) stay unsigned and go in
        //    the u8 operand directly — no sign trick, unlike Q4_0, which recenters
        //    by −8 at decode and needs one.
        //  - Q6_K's −32 recentering cannot be baked into the weights as NEON does
        //    (that would make them signed). It is applied algebraically instead:
        //    Σ((q−32)·a) = Σ(q·a) − 32·Σa, with Σa precomputed per column at
        //    *16-element* granularity because Q6_K scales are 16-wide.
        //
        // `vpdpbusd` (the non-saturating form) is safe here: per-lane sums are
        // bounded (≤ 4·63·127) and each `dot32u` result is converted to float and
        // accumulated there per sub-block — the i32 lanes from one call are never
        // fed back into another, so the i32 accumulator cannot overflow.
        //
        // The AVX2 instantiation's `dot32u` DOES saturate its i16 intermediate,
        // so it needs the tighter bound, and it holds: `maddubs` sums two
        // adjacent products, peaking at 2·63·127 = 16002 against 32767. See the
        // `avx2_int8` module header for the same argument over every caller.

        /// Column tile width. Decoding a K-quant super-block is the expensive part;
        /// applying each decoded group to `KQ_COLS` activation columns amortizes
        /// it — mirrors the NEON kernels' choice of 8. (The AVX2 instantiation
        /// passes 4, for the register budget reason at its invocation site.)
        const KQ_COLS: usize = $kq_cols;

        /// Horizontal sum of 8 i32 lanes.
        #[target_feature(enable = $feat)]
        unsafe fn hsum256_epi32(v: __m256i) -> i32 {
            let s = _mm_add_epi32(_mm256_extracti128_si256(v, 1), _mm256_castsi256_si128(v));
            let s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
            let s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
            _mm_cvtsi128_si32(s)
        }

        /// Horizontal sum of 4 i32 lanes.
        #[target_feature(enable = $feat)]
        unsafe fn hsum128_epi32(v: __m128i) -> i32 {
            let s = _mm_add_epi32(v, _mm_srli_si128(v, 8));
            let s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
            _mm_cvtsi128_si32(s)
        }

        /// Per-column, per-32-element-block sums of the Q8_0 activation quants:
        /// `sums[j * (k/32) + b]`. The Q4_K mins term consumes these.
        #[target_feature(enable = $feat)]
        unsafe fn q8_0_col_sums(b_quants: &[i8], n: usize, k: usize) -> Vec<i32> {
            unsafe {
                let nb32 = k / 32;
                let mut sums = vec![0i32; n * nb32];
                let ones = _mm256_set1_epi8(1);
                for j in 0..n {
                    let base = b_quants.as_ptr().add(j * k);
                    for b in 0..nb32 {
                        let x = _mm256_loadu_si256(base.add(b * 32) as *const __m256i);
                        sums[j * nb32 + b] = hsum256_epi32(dot32u(ones, x));
                    }
                }
                sums
            }
        }

        /// Same, at 16-element granularity: `sums16[j * (k/16) + h]`. Q6_K scales
        /// are 16-wide, so its recentering needs half-block activation sums.
        #[target_feature(enable = $feat)]
        unsafe fn q8_0_col_sums16(b_quants: &[i8], n: usize, k: usize) -> Vec<i32> {
            unsafe {
                let nh = k / 16;
                let mut sums = vec![0i32; n * nh];
                let ones = _mm_set1_epi8(1);
                for j in 0..n {
                    let base = b_quants.as_ptr().add(j * k);
                    for h in 0..nh {
                        let x = _mm_loadu_si128(base.add(h * 16) as *const __m128i);
                        sums[j * nh + h] = hsum128_epi32(dot16u(ones, x));
                    }
                }
                sums
            }
        }

        /// Batched GEMM: `C[m, n] = A_q4_k[m, k] @ B_q8_0[k, n]`.
        ///
        /// Layout contract matches the Q4_0/Q8_0 GEMMs (`b_scales[n][k/32]`,
        /// `b_quants[n][k]`), structure matches the NEON twin: decode each
        /// super-block group once, apply to `KQ_COLS` columns. Per sub-block `s`
        /// the contribution is `xs·(d·sc_s·Σ(q·aq) − dmin·mn_s·Σaq)`.
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q4_k_q8_0(
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
            let row_bytes = sb * size_of::<crate::quant::BlockQ4KM>();
            debug_assert_eq!(a_quant.len(), m * row_bytes, "Q4_K GEMM: a_quant size");
            debug_assert_eq!(b_quants.len(), n * k, "Q4_K GEMM: b_quants size");
            debug_assert_eq!(b_scales.len(), n * nb32, "Q4_K GEMM: b_scales size");
            debug_assert_eq!(out.len(), m * n, "Q4_K GEMM: out size");

            let col_sums = unsafe { q8_0_col_sums(b_quants, n, k) };
            let cs: &[i32] = &col_sums;

            // Slices captured by reference — Send, no pointer→usize laundering.
            let compute_row = |(i, row_out): (usize, &mut [f32])| {
                // SAFETY: row `i` spans `a_quant[i*row_bytes..][..row_bytes]`; the
                // debug_asserts above pin every slice length.
                unsafe {
                    let mask_0f = _mm256_set1_epi8(0x0F);
                    let row_start = i * row_bytes;

                    let mut j0 = 0usize;
                    while j0 < n {
                        // Decode fast path (n == 1): the single output column has
                        // ONE `facc` chain, so the per-sub-block `fmadd` is
                        // latency-bound (each depends on the previous). Split it
                        // across `NACC` independent accumulators, cycling sub-blocks
                        // round-robin, so the chains overlap and hide the fmadd
                        // latency — the same trick the multi-column tile below gets
                        // for free from its independent columns (which is why that
                        // path is NOT multi-accumulated: `KQ_COLS * NACC` vectors
                        // would spill).
                        //
                        // Gated on `n == 1`, NOT `cols == 1`: the float reduction
                        // order differs from the tiled path, so a single-column
                        // *remainder* of a wider GEMM (n % KQ_COLS == 1) would break
                        // the AVX2/VNNI cross-tier bit-identity the prefill path
                        // holds (the two tiers tile at different widths). Decode is
                        // the only n == 1 caller and its parity is a cosine floor
                        // (already relaxed by the #292 repack), not bit-exact, so it
                        // absorbs the reorder; prefill stays bit-identical.
                        if n == 1 {
                            // Fixed at 4: the round-robin mask (`& (NACC - 1)`)
                            // needs a power of two and the tree-reduce below
                            // hardcodes `facc[0..4]` — not a free tuning knob.
                            const NACC: usize = 4;
                            let j = j0;
                            let mut facc = [_mm256_setzero_ps(); NACC];
                            let mut macc = 0.0f32;
                            let mut slot = 0usize;
                            for bi in 0..sb {
                                let blk = &*(a_quant
                                    .as_ptr()
                                    .add(row_start + bi * size_of::<crate::quant::BlockQ4KM>())
                                    as *const crate::quant::BlockQ4KM);
                                let d = half::f16::from_bits(blk.d).to_f32();
                                let dmin = half::f16::from_bits(blk.dmin).to_f32();
                                let (sc, mn) = crate::quant::decode_q4km_scales(&blk.scales);
                                let qs = blk.qs.as_ptr();
                                for g in 0..4 {
                                    let qb = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                                    let w_lo = _mm256_and_si256(qb, mask_0f);
                                    let w_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_0f);
                                    for (w, s) in [(w_lo, 2 * g), (w_hi, 2 * g + 1)] {
                                        let xb = bi * 8 + s;
                                        let dsc = d * sc[s] as f32;
                                        let dmn = dmin * mn[s] as f32;
                                        let x = _mm256_loadu_si256(
                                            b_quants.as_ptr().add(j * k + xb * 32)
                                                as *const __m256i,
                                        );
                                        let lanes = _mm256_cvtepi32_ps(dot32u(w, x));
                                        let xs = *b_scales.get_unchecked(j * nb32 + xb);
                                        let sx = *cs.get_unchecked(j * nb32 + xb);
                                        let scale = _mm256_set1_ps(xs * dsc);
                                        // `slot` is masked to 0..NACC, so this is
                                        // in-bounds — skip the bounds check in the
                                        // innermost decode loop (matches the
                                        // `get_unchecked` idiom used for the loads
                                        // above).
                                        let acc = facc.get_unchecked_mut(slot);
                                        *acc = _mm256_fmadd_ps(lanes, scale, *acc);
                                        macc += xs * dmn * sx as f32;
                                        slot = (slot + 1) & (NACC - 1);
                                    }
                                }
                            }
                            // Tree-reduce the NACC accumulators, then one hsum.
                            let f01 = _mm256_add_ps(facc[0], facc[1]);
                            let f23 = _mm256_add_ps(facc[2], facc[3]);
                            let f = _mm256_add_ps(f01, f23);
                            *row_out.get_unchecked_mut(j) = hsum256_ps(f) - macc;
                            j0 += 1;
                            continue;
                        }
                        let cols = KQ_COLS.min(n - j0);
                        // Defer the reduction: `dp = Σ_lane dot32u(w,x)[lane]` and
                        // the answer is `Σ_s xs·dsc·dp - Σ_s xs·dmn·sx`. Since the
                        // lane-sum is linear, `Σ_s (xs·dsc)·dp = hsum(Σ_s (xs·dsc)·
                        // lanes)`, so the 8 dpbusd lanes stay in a float vector
                        // accumulator and collapse with a single hsum per column at
                        // the end — the same structure the Q4_0/Q8_0 kernels use,
                        // instead of an hsum per (sub-block, column). The mins term
                        // carries no dot lanes, so it accumulates as a scalar.
                        let mut facc = [_mm256_setzero_ps(); KQ_COLS];
                        let mut macc = [0.0f32; KQ_COLS];

                        for bi in 0..sb {
                            let blk = &*(a_quant
                                .as_ptr()
                                .add(row_start + bi * size_of::<crate::quant::BlockQ4KM>())
                                as *const crate::quant::BlockQ4KM);
                            let d = half::f16::from_bits(blk.d).to_f32();
                            let dmin = half::f16::from_bits(blk.dmin).to_f32();
                            let (sc, mn) = crate::quant::decode_q4km_scales(&blk.scales);
                            let qs = blk.qs.as_ptr();

                            for g in 0..4 {
                                // Chunk g: low nibbles = sub-block 2g, high = 2g+1;
                                // byte l is element l of its sub-block, matching the
                                // contiguous 32-quant activation block.
                                let qb = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                                let w_lo = _mm256_and_si256(qb, mask_0f);
                                let w_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_0f);

                                for (w, s) in [(w_lo, 2 * g), (w_hi, 2 * g + 1)] {
                                    let xb = bi * 8 + s;
                                    let dsc = d * sc[s] as f32;
                                    let dmn = dmin * mn[s] as f32;
                                    for jj in 0..cols {
                                        let j = j0 + jj;
                                        let x = _mm256_loadu_si256(
                                            b_quants.as_ptr().add(j * k + xb * 32)
                                                as *const __m256i,
                                        );
                                        let lanes = _mm256_cvtepi32_ps(dot32u(w, x));
                                        let xs = *b_scales.get_unchecked(j * nb32 + xb);
                                        let sx = *cs.get_unchecked(j * nb32 + xb);
                                        let scale = _mm256_set1_ps(xs * dsc);
                                        facc[jj] = _mm256_fmadd_ps(lanes, scale, facc[jj]);
                                        macc[jj] += xs * dmn * sx as f32;
                                    }
                                }
                            }
                        }

                        for jj in 0..cols {
                            *row_out.get_unchecked_mut(j0 + jj) = hsum256_ps(facc[jj]) - macc[jj];
                        }
                        j0 += cols;
                    }
                }
            };

            if m < crate::backend::cpu::gemv_par_threshold() {
                // Too few rows to amortize any fork-join — serial, matching the
                // Q4_0/Q8_0 GEMV tails, which also gate their `par_rows` call on
                // this threshold.
                out.chunks_mut(n).enumerate().for_each(compute_row);
            } else if n == 1 {
                // Decode (n == 1): route the per-token GEMV through the *decode*
                // pool, not the wide prefill pool `par_rows_n` uses. Fanning a
                // 1-column K-quant GEMV across every core is dominated by
                // fork-join overhead — the per-row work is tiny — so it runs no
                // faster than serial while burning every core spinning. Q4_0/Q8_0
                // decode already gates the same `par_rows` call this way; this
                // brings K-quants in line. The math is unchanged (same
                // `compute_row`), so decode stays bit-identical to the batched
                // GEMM at n = 1.
                crate::backend::cpu::par_rows(
                    out,
                    crate::backend::cpu::gemv_min_rows(),
                    |(row, yv)| compute_row((row, core::slice::from_mut(yv))),
                );
            } else {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            }
        }

        /// Batched GEMM: `C[m, n] = A_q4_1[m, k] @ B_q8_0[k, n]`.
        ///
        /// Q4_1 dequant is `w = d·q + m`, `q ∈ [0, 15]` (no `−8` recenter). Per
        /// 32-element block the contribution to column `j` is
        /// `xs·(d·Σ(q·aq) + m·Σaq)` — the same deferred-reduction structure as
        /// `gemm_q4_k_q8_0` (dot lanes accumulate in `facc`, the block-sum term in
        /// `macc`), but the sum term is **added** (Q4_1's `m` is a raising offset,
        /// unlike the K-quant `dmin` which subtracts) and there are no sub-block
        /// scales — a Q4_1 block maps 1:1 onto a Q8_0 activation block. The 16 `qs`
        /// bytes expand to a 32-lane weight vector matching `dequantize_q4_1_block`'s
        /// order: low nibbles → element indices `0..16` (lane-0 half), high nibbles →
        /// `16..32` (lane-1 half). The nibble values themselves are all in `[0, 15]`.
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q4_1_q8_0(
            a_quant: &[u8],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) {
            debug_assert_eq!(k % 32, 0, "Q4_1 GEMM: k must be divisible by 32");
            let nb = k / 32;
            let row_bytes = nb * size_of::<crate::quant::BlockQ4_1>();
            debug_assert_eq!(a_quant.len(), m * row_bytes, "Q4_1 GEMM: a_quant size");
            debug_assert_eq!(b_quants.len(), n * k, "Q4_1 GEMM: b_quants size");
            debug_assert_eq!(b_scales.len(), n * nb, "Q4_1 GEMM: b_scales size");
            debug_assert_eq!(out.len(), m * n, "Q4_1 GEMM: out size");

            let col_sums = unsafe { q8_0_col_sums(b_quants, n, k) };
            let cs: &[i32] = &col_sums;

            let compute_row = |(i, row_out): (usize, &mut [f32])| {
                // SAFETY: row `i` spans `a_quant[i*row_bytes..][..row_bytes]`; the
                // debug_asserts above pin every slice length.
                unsafe {
                    let mask128 = _mm_set1_epi8(0x0F);
                    let row_start = i * row_bytes;

                    let mut j0 = 0usize;
                    while j0 < n {
                        let cols = KQ_COLS.min(n - j0);
                        let mut facc = [_mm256_setzero_ps(); KQ_COLS];
                        let mut macc = [0.0f32; KQ_COLS];

                        for bi in 0..nb {
                            let blk = &*(a_quant
                                .as_ptr()
                                .add(row_start + bi * size_of::<crate::quant::BlockQ4_1>())
                                as *const crate::quant::BlockQ4_1);
                            let d = half::f16::from_bits(blk.d).to_f32();
                            let mmin = half::f16::from_bits(blk.m).to_f32();
                            // 16 packed bytes → 32 weight values. Low nibbles fill the
                            // low 128-bit lane (element indices 0..16), high nibbles the
                            // high lane (16..32), matching the contiguous activation block.
                            let qb = _mm_loadu_si128(blk.qs.as_ptr() as *const __m128i);
                            let lo = _mm_and_si128(qb, mask128);
                            let hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask128);
                            let w = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

                            for jj in 0..cols {
                                let j = j0 + jj;
                                let x = _mm256_loadu_si256(
                                    b_quants.as_ptr().add(j * k + bi * 32) as *const __m256i
                                );
                                let lanes = _mm256_cvtepi32_ps(dot32u(w, x));
                                let xs = *b_scales.get_unchecked(j * nb + bi);
                                let sx = *cs.get_unchecked(j * nb + bi);
                                // Fold the weight (`d`) and activation (`xs`) scales into
                                // one per-(block,column) factor, as the Q4_K kernel does.
                                let scale = _mm256_set1_ps(xs * d);
                                facc[jj] = _mm256_fmadd_ps(lanes, scale, facc[jj]);
                                macc[jj] += xs * mmin * sx as f32;
                            }
                        }

                        for jj in 0..cols {
                            *row_out.get_unchecked_mut(j0 + jj) = hsum256_ps(facc[jj]) + macc[jj];
                        }
                        j0 += cols;
                    }
                }
            };

            if m < crate::backend::cpu::gemv_par_threshold() {
                out.chunks_mut(n).enumerate().for_each(compute_row);
            } else if n == 1 {
                crate::backend::cpu::par_rows(
                    out,
                    crate::backend::cpu::gemv_min_rows(),
                    |(row, yv)| compute_row((row, core::slice::from_mut(yv))),
                );
            } else {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            }
        }

        /// Batched GEMM: `C[m, n] = A_q6_k[m, k] @ B_q8_0[k, n]`.
        ///
        /// Q6_K geometry (mirrors `dequantize_q6_k_block`): two 128-value halves per
        /// super-block (`ql += 64`, `qh += 32`, `sc += 8`); half `nh`, group `g`
        /// covers elements `nh*128 + g*32..+32` with quants
        /// `(ql[(g&1)*32 + l] nibble(g<2 ? lo : hi)) | (((qh[l] >> 2g) & 3) << 4)`
        /// and scales `sc[nh*8 + 2g + is]`, `is = l/16`. Scales are 16-wide, so a
        /// 32-quant activation block spans two of them — the dpbusd lanes split
        /// cleanly (lanes 0..3 = first 16 bytes, 4..7 = second 16).
        #[target_feature(enable = $feat)]
        pub unsafe fn gemm_q6_k_q8_0(
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
            let nh16 = k / 16;
            let row_bytes = sb * size_of::<crate::quant::BlockQ6K>();
            debug_assert_eq!(a_quant.len(), m * row_bytes, "Q6_K GEMM: a_quant size");
            debug_assert_eq!(b_quants.len(), n * k, "Q6_K GEMM: b_quants size");
            debug_assert_eq!(b_scales.len(), n * nb32, "Q6_K GEMM: b_scales size");
            debug_assert_eq!(out.len(), m * n, "Q6_K GEMM: out size");

            let col_sums16 = unsafe { q8_0_col_sums16(b_quants, n, k) };
            let cs16: &[i32] = &col_sums16;

            let compute_row = |(i, row_out): (usize, &mut [f32])| {
                // SAFETY: as in `gemm_q4_k_q8_0`.
                unsafe {
                    let mask_0f = _mm256_set1_epi8(0x0F);
                    let mask_03 = _mm256_set1_epi8(0x03);
                    let row_start = i * row_bytes;

                    let mut j0 = 0usize;
                    while j0 < n {
                        // Decode fast path (n == 1) — mirrors the Q4_K kernel's:
                        // the single column's one `facc` chain is latency-bound, so
                        // split it across `NACC` independent accumulators cycled
                        // round-robin. See the Q4_K `n == 1` block for the full
                        // rationale, the `n == 1` (not `cols == 1`) gate, and why
                        // the tiled path stays single-accumulator.
                        if n == 1 {
                            // Fixed at 4 — see the Q4_K block: the round-robin mask
                            // and the tree-reduce below both hardcode it.
                            const NACC: usize = 4;
                            let j = j0;
                            let mut facc = [_mm256_setzero_ps(); NACC];
                            let mut macc = 0.0f32;
                            let mut slot = 0usize;
                            for bi in 0..sb {
                                let blk = &*(a_quant
                                    .as_ptr()
                                    .add(row_start + bi * size_of::<crate::quant::BlockQ6K>())
                                    as *const crate::quant::BlockQ6K);
                                let d = half::f16::from_bits(blk.d).to_f32();
                                let d32 = d * 32.0;
                                let ql = blk.ql.as_ptr();
                                let qh = blk.qh.as_ptr();
                                for nh in 0..2usize {
                                    let qhb = _mm256_loadu_si256(qh.add(nh * 32) as *const __m256i);
                                    for g in 0..4usize {
                                        let qlb = _mm256_loadu_si256(
                                            ql.add(nh * 64 + (g & 1) * 32) as *const __m256i,
                                        );
                                        let l4 = if g < 2 {
                                            _mm256_and_si256(qlb, mask_0f)
                                        } else {
                                            _mm256_and_si256(_mm256_srli_epi16(qlb, 4), mask_0f)
                                        };
                                        let h2 = _mm256_and_si256(
                                            _mm256_srl_epi16(qhb, _mm_cvtsi32_si128(2 * g as i32)),
                                            mask_03,
                                        );
                                        let w = _mm256_or_si256(l4, _mm256_slli_epi16(h2, 4));
                                        let sc0 = *blk.scales.get_unchecked(nh * 8 + 2 * g) as f32;
                                        let sc1 =
                                            *blk.scales.get_unchecked(nh * 8 + 2 * g + 1) as f32;
                                        let xb = bi * 8 + nh * 4 + g;
                                        let sc_split_d = _mm256_set_m128(
                                            _mm_set1_ps(sc1 * d),
                                            _mm_set1_ps(sc0 * d),
                                        );
                                        let x = _mm256_loadu_si256(
                                            b_quants.as_ptr().add(j * k + xb * 32)
                                                as *const __m256i,
                                        );
                                        let lanes = _mm256_cvtepi32_ps(dot32u(w, x));
                                        let xs = *b_scales.get_unchecked(j * nb32 + xb);
                                        let sx0 = *cs16.get_unchecked(j * nh16 + xb * 2);
                                        let sx1 = *cs16.get_unchecked(j * nh16 + xb * 2 + 1);
                                        let scale = _mm256_mul_ps(sc_split_d, _mm256_set1_ps(xs));
                                        // `slot` is masked to 0..NACC, so this is
                                        // in-bounds — skip the bounds check in the
                                        // innermost decode loop (matches the
                                        // `get_unchecked` idiom used for the loads
                                        // above).
                                        let acc = facc.get_unchecked_mut(slot);
                                        *acc = _mm256_fmadd_ps(lanes, scale, *acc);
                                        macc += xs * d32 * (sc0 * sx0 as f32 + sc1 * sx1 as f32);
                                        slot = (slot + 1) & (NACC - 1);
                                    }
                                }
                            }
                            let f01 = _mm256_add_ps(facc[0], facc[1]);
                            let f23 = _mm256_add_ps(facc[2], facc[3]);
                            let f = _mm256_add_ps(f01, f23);
                            *row_out.get_unchecked_mut(j) = hsum256_ps(f) - macc;
                            j0 += 1;
                            continue;
                        }
                        let cols = KQ_COLS.min(n - j0);
                        // Deferred reduction, as in the Q4_K kernel — but a 32-quant
                        // group here spans two 16-wide Q6_K scales, so the 8 dpbusd
                        // lanes carry two distinct scales: lanes 0..3 (first 16
                        // elements) weight by `sc0`, lanes 4..7 by `sc1`. A per-lane
                        // scale vector `[sc0×4, sc1×4]·d` folds both into one fmadd,
                        // so the two hsum128s per (group, column) become one
                        // hsum256 per column. The −32 recentering has no dot lanes,
                        // so it accumulates as a scalar.
                        let mut facc = [_mm256_setzero_ps(); KQ_COLS];
                        let mut macc = [0.0f32; KQ_COLS];

                        for bi in 0..sb {
                            let blk = &*(a_quant
                                .as_ptr()
                                .add(row_start + bi * size_of::<crate::quant::BlockQ6K>())
                                as *const crate::quant::BlockQ6K);
                            let d = half::f16::from_bits(blk.d).to_f32();
                            let d32 = d * 32.0;
                            let ql = blk.ql.as_ptr();
                            let qh = blk.qh.as_ptr();

                            for nh in 0..2usize {
                                let qhb = _mm256_loadu_si256(qh.add(nh * 32) as *const __m256i);
                                for g in 0..4usize {
                                    let qlb = _mm256_loadu_si256(
                                        ql.add(nh * 64 + (g & 1) * 32) as *const __m256i
                                    );
                                    let l4 = if g < 2 {
                                        _mm256_and_si256(qlb, mask_0f)
                                    } else {
                                        _mm256_and_si256(_mm256_srli_epi16(qlb, 4), mask_0f)
                                    };
                                    // Runtime shift by 2g (0/2/4/6): `srl` with a
                                    // count register — `srli` needs a const.
                                    let h2 = _mm256_and_si256(
                                        _mm256_srl_epi16(qhb, _mm_cvtsi32_si128(2 * g as i32)),
                                        mask_03,
                                    );
                                    // h2 ≤ 3, so `<< 4` ≤ 48: stays inside its own
                                    // byte, no bleed across the epi16 lane boundary.
                                    let w = _mm256_or_si256(l4, _mm256_slli_epi16(h2, 4));

                                    let sc0 = *blk.scales.get_unchecked(nh * 8 + 2 * g) as f32;
                                    let sc1 = *blk.scales.get_unchecked(nh * 8 + 2 * g + 1) as f32;
                                    let xb = bi * 8 + nh * 4 + g;
                                    // [sc0,sc0,sc0,sc0, sc1,sc1,sc1,sc1] · d — the
                                    // column loop only broadcasts and folds in `xs`.
                                    let sc_split_d =
                                        _mm256_set_m128(_mm_set1_ps(sc1 * d), _mm_set1_ps(sc0 * d));

                                    for jj in 0..cols {
                                        let j = j0 + jj;
                                        let x = _mm256_loadu_si256(
                                            b_quants.as_ptr().add(j * k + xb * 32)
                                                as *const __m256i,
                                        );
                                        let lanes = _mm256_cvtepi32_ps(dot32u(w, x));
                                        let xs = *b_scales.get_unchecked(j * nb32 + xb);
                                        let sx0 = *cs16.get_unchecked(j * nh16 + xb * 2);
                                        let sx1 = *cs16.get_unchecked(j * nh16 + xb * 2 + 1);
                                        let scale = _mm256_mul_ps(sc_split_d, _mm256_set1_ps(xs));
                                        facc[jj] = _mm256_fmadd_ps(lanes, scale, facc[jj]);
                                        macc[jj] +=
                                            xs * d32 * (sc0 * sx0 as f32 + sc1 * sx1 as f32);
                                    }
                                }
                            }
                        }

                        for jj in 0..cols {
                            *row_out.get_unchecked_mut(j0 + jj) = hsum256_ps(facc[jj]) - macc[jj];
                        }
                        j0 += cols;
                    }
                }
            };

            if m < crate::backend::cpu::gemv_par_threshold() {
                // Too few rows to amortize any fork-join — serial, matching the
                // Q4_0/Q8_0 GEMV tails, which also gate their `par_rows` call on
                // this threshold.
                out.chunks_mut(n).enumerate().for_each(compute_row);
            } else if n == 1 {
                // Decode (n == 1): route the per-token GEMV through the *decode*
                // pool, not the wide prefill pool `par_rows_n` uses. Fanning a
                // 1-column K-quant GEMV across every core is dominated by
                // fork-join overhead — the per-row work is tiny — so it runs no
                // faster than serial while burning every core spinning. Q4_0/Q8_0
                // decode already gates the same `par_rows` call this way; this
                // brings K-quants in line. The math is unchanged (same
                // `compute_row`), so decode stays bit-identical to the batched
                // GEMM at n = 1.
                crate::backend::cpu::par_rows(
                    out,
                    crate::backend::cpu::gemv_min_rows(),
                    |(row, yv)| compute_row((row, core::slice::from_mut(yv))),
                );
            } else {
                crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
            }
        }

        /// Q4_K GEMV: quantize `x` to Q8_0 and run this tier's GEMM at `n = 1`.
        ///
        /// Emitted per tier so decode and batched prefill share arithmetic **by
        /// construction** — not by two implementations happening to agree.
        /// Shipping the GEMM without it is a live trap: with the AVX2 GEMM wired
        /// up and decode still on the f32 `vec_dot` path, LFM2.5-230M-Q4_K_M
        /// scored cosine 0.999335 against the parity tests' 0.9999 naive bar
        /// (VNNI, where both paths already matched, scores exactly 1.000000).
        /// aarch64 gets the same property from both paths sharing the NEON dot.
        ///
        /// The quantizer is the crate's dispatching one, so it is by definition
        /// the same function prefill's `quantize_columns` reaches on this host —
        /// the other half of the invariant. Which arm it picks (see
        /// `cpu::avx512_quantizer_available`) does not matter here precisely
        /// because both sides of the invariant go through the same call.
        #[target_feature(enable = $feat)]
        pub unsafe fn gemv_q4k_f32(
            a_quant: &[u8],
            x: &[f32],
            y: &mut [f32],
            m: usize,
            k: usize,
            q8_scales: &mut Vec<f32>,
            q8_quants: &mut Vec<i8>,
        ) {
            unsafe {
                q8_scales.resize(k / 32, 0.0);
                q8_quants.resize(k, 0);
                crate::backend::cpu::quantize_f32_to_q8_0_into(x, q8_scales, q8_quants);
                gemm_q4_k_q8_0(a_quant, q8_scales, q8_quants, y, m, 1, k);
            }
        }

        /// Q6_K GEMV — see [`gemv_q4k_f32`] for why this shares the GEMM.
        #[target_feature(enable = $feat)]
        pub unsafe fn gemv_q6k_f32(
            a_quant: &[u8],
            x: &[f32],
            y: &mut [f32],
            m: usize,
            k: usize,
            q8_scales: &mut Vec<f32>,
            q8_quants: &mut Vec<i8>,
        ) {
            unsafe {
                q8_scales.resize(k / 32, 0.0);
                q8_quants.resize(k, 0);
                crate::backend::cpu::quantize_f32_to_q8_0_into(x, q8_scales, q8_quants);
                gemm_q6_k_q8_0(a_quant, q8_scales, q8_quants, y, m, 1, k);
            }
        }
    };
}

// ── x86_64 AVX2 int8 (VNNI-free `dpbusd` emulation) ─────────────────────────
//
// The same int8 GEMM kernels, for hosts without AVX-512 VNNI. That is not a
// niche: every Zen 1-3 and every pre-Ice-Lake Intel part lands here, as does
// Skylake-X (AVX-512 but no VNNI). Before this module they all took the
// per-token GEMV fallback for prefill, which `warn_unbatchable` reports as
// "several times slower than it should be" — accurately.
//
// **The emulation.** `vpdpbusd(0, u, s)` is `_mm256_maddubs_epi16` (u8 x i8 ->
// i16, pairwise-summed) followed by `_mm256_madd_epi16` against ones (i16 pairs
// -> i32). Two instructions instead of one, and `maddubs` saturates its i16
// intermediate — so the emulation is only equal to VNNI while that intermediate
// cannot overflow.
//
// **Why it cannot.** `maddubs` sums two adjacent products into one i16, so the
// bound is `2 * max|w| * max|a|` against 32767:
//
//   - `dot32`   Q8_0 weights: `2 * 128 * 127 = 32512`  (fits, 255 to spare)
//   - `dot32`   Q4_0 weights, `[-8, 7]`: `2 * 8 * 127 = 2032`
//   - `dot32u`  Q4_K weights, `0..15`:   `2 * 15 * 127 = 3810`
//   - `dot32u`  Q4_1 weights, `0..15`:   `2 * 15 * 127 = 3810`
//   - `dot32u`  Q6_K weights, `0..63`:   `2 * 63 * 127 = 16002`
//   - `dot32u`  activation sums (`w = 1`): `2 * 1 * 127 = 254`
//
// The `127` is not an assumption: it is the same precondition `dot32` already
// documents for the sign trick, enforced by the quantizers. The Q8_0 case is the
// tight one and it is tight *because* weights may be `-128` while activations
// may not — had both been unbounded i8 this would overflow and the emulation
// would be wrong.
//
// So this is exact, not approximate: the `avx2_*_matches_vnni_bit_exact` tests
// below assert the two produce identical bits on a VNNI host, over inputs that
// include the `-128` weight extreme.
//
// THE `-128` ACTIVATION CORNER, worked through because "exact" is a strong word
// and the `[-127, 127]` precondition is not quite a theorem. A block whose
// `amax` is subnormal loses so much precision in `d = amax / 127` that
// `v * (1/d)` can overshoot and clamp to `-128` — see `dot32`'s precondition
// doc. Exactness survives it anyway, and the reason is the *sign* of the
// product, not its magnitude: the sign trick forms `|w| * sign(a, w)`, and a
// `+128` second operand is unreachable (it would need `a = +128`, or `a = -128`
// against a positive weight, which `sign` leaves negative). So the extreme pair
// is `2 * 128 * -128 = -32768` — exactly `i16::MIN`, representable, not
// saturated. Positive products stay capped at `2 * 128 * 127 = 32512`. Such a
// block also has `f16::from_f32(d) == 0.0`, so it contributes nothing either
// way; the point is that even its raw dot matches `dpbusd`.
//
// **Registers.** VEX exposes 16 vector registers, not EVEX's 32, so the tile
// constants are narrower than the VNNI instantiation's — a 4x4 strip would be
// 16 accumulators before a single weight or activation register. That reasoning
// picks the right values but predicts a much larger effect than measurement
// shows; see the invocation below for the numbers and their caveat.
//
// Note this module needs no `avx512` crate feature: `maddubs`/`madd` are
// SSSE3/SSE2-era intrinsics, stable well before the crate MSRV.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_int8 {
    use super::*;
    use std::arch::x86_64::*;

    /// `vpdpbusd(0, w, a)` for an already-unsigned `w`, without VNNI.
    ///
    /// **Precondition:** `2 * max(w) * max|a| <= 32767`, or the `maddubs` i16
    /// intermediate saturates and the result silently stops matching VNNI. The
    /// module note above enumerates every caller against that bound.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot32u(w: __m256i, a: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), _mm256_set1_epi16(1))
    }

    /// 128-bit `dot32u`, for Q6_K's 16-element sub-block activation sums. Same
    /// saturation bound as the 256-bit form — the lane width does not change
    /// what `maddubs` accumulates into an i16.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot16u(w: __m128i, a: __m128i) -> __m128i {
        _mm_madd_epi16(_mm_maddubs_epi16(w, a), _mm_set1_epi16(1))
    }

    /// `vpdpbusd(0, |w|, sign(a, w))` for signed weights, without VNNI.
    ///
    /// Identical sign trick to the VNNI `dot32`, and identical precondition:
    /// activations must be in `[-127, 127]`, since `_mm256_sign_epi8` cannot
    /// negate `-128`. Weights may be `-128`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot32(w: __m256i, a: __m256i) -> __m256i {
        unsafe { dot32u(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w)) }
    }

    // 16 VEX registers, so the tiles are narrower than the VNNI instantiation's
    // 8/4/4/8: TILE_N=4 per-row columns, and a 2x4 (8-accumulator) strip, which
    // leaves room for the 2 decoded weights, 4 activations, and scale temps.
    //
    // The register argument predicts this correctly but **overstates it**. Both
    // sets were measured on LFM2.5-350M-Q4_0 pp512, `CERA_CPU_TIER=avx2`, pool
    // pinned at 16, alternating arm order, 3 reps: narrow won 3/3 at 203/204/198
    // vs 200/199/196 tok/s — about 2%, not the collapse an outright spill would
    // cause. 4/4/4/4 and 8/2/4/8 measured in between. So: keep the narrow tiles,
    // but do not read this as "the wide tiles spill catastrophically".
    //
    // CAVEAT on all four numbers: they come from a Zen 5 part *executing* AVX2,
    // which renames far more physical registers than the Zen 1-3 / Skylake-X
    // hosts this module actually targets. The ordering is likely to hold there
    // and the margin is not. Re-tune on a real AVX2-only part before treating
    // these as final.
    int8_gemm_kernels!("avx2,fma", 4, 2, 4, 4);

    #[cfg(test)]
    mod avx2_int8_tests {
        //! The AVX2 emulation is claimed to be *exact*, not approximate, so the
        //! test is equality against VNNI rather than a tolerance — on a host
        //! that has both. A tolerance-based test here would pass even if
        //! `maddubs` saturated, which is the one failure mode worth catching.
        //!
        //! The VNNI comparisons need a host with *both* ISAs, and the
        //! GitHub-hosted x86 pool is not guaranteed to provide one — AVX2 is,
        //! VNNI is not, and AVX2-only is precisely where these kernels run. So
        //! the suite is split by what it *requires*, not by what it asserts:
        //!
        //!   - `#[cfg(feature = "avx512")]` + `both_callable()`: the three
        //!     `*_matches_vnni_bit_exact` tests. The strongest check there is,
        //!     and the only one that can confirm the emulation is exact rather
        //!     than merely close — but it needs a VNNI host, so treat it as
        //!     opportunistic. `CERA_REQUIRE_SIMD=avx512vnni` turns its skip into
        //!     a failure where the hardware is known.
        //!   - Everything else: no VNNI dependency, no `avx512` crate feature.
        //!     These carry the CI coverage, and `CERA_REQUIRE_SIMD=avx2` (set by
        //!     the x86 CI leg) turns their skip into a failure. Note that
        //!     `avx2_row_tiled_matches_per_row_bit_exact` is bit-exact too — it
        //!     compares the strip kernel against the per-row one, which needs
        //!     only AVX2.
        //!
        //! Deliberately not enumerated by name here: a list in a comment is one
        //! renamed test away from lying about its own coverage.
        use super::*;
        use crate::quant::{BlockQ4_0, BlockQ8_0};

        // The comparison target only exists when the `avx512` feature is on —
        // this module deliberately does not depend on it, so under
        // `--no-default-features` the three `*_matches_vnni_bit_exact` tests
        // compile out and every other test in the module still runs.
        #[cfg(feature = "avx512")]
        use super::super::avx512_vnni as vnni;

        use crate::backend::simd::require_simd_or_skip;

        /// Every feature in these kernels' `#[target_feature]` list, not a subset:
        /// calling one without all of them detected is UB. Named rather than
        /// spelled out per test so adding a feature to the kernels is one edit,
        /// not six — a missed copy is a test that runs where the kernel is
        /// illegal. Mirrors `vnni_kernels_callable` in the VNNI module.
        fn avx2_kernels_callable() -> bool {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }

        #[cfg(feature = "avx512")]
        fn both_callable() -> bool {
            // Every feature in the VNNI kernels' `#[target_feature]` set, not a
            // subset: calling one without all of them detected is UB, which is
            // the same reason `vnni_kernels_callable` lists all five.
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vnni")
        }

        fn lcg(st: &mut u64) -> u64 {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *st >> 33
        }

        /// Q8_0 weight rows that deliberately include `-128`.
        ///
        /// `-128` is the tight case for the emulation: `2 * 128 * 127 = 32512`
        /// is the closest any caller comes to the *positive* i16 saturation
        /// point (the module header works through the negative direction, which
        /// reaches `i16::MIN` exactly and is therefore representable), and it
        /// is also the value the sign trick handles only on the weight side.
        /// The existing `rand_q8_0_rows` helper draws from `[-127, 127]` and so
        /// never exercises it.
        fn q8_0_rows_with_extremes(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let nb = k / 32;
            let mut v = Vec::with_capacity(m * nb * size_of::<BlockQ8_0>());
            for blk in 0..m * nb {
                v.extend_from_slice(
                    &half::f16::from_f32(0.01 + 0.003 * (blk % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for t in 0..32 {
                    // Seed the saturation worst case deterministically rather
                    // than hoping for it: `maddubs` sums two *adjacent* products
                    // into one i16, so the tight case is two neighbouring
                    // `-128` weights, which is what lanes 0 and 1 force. Leaving
                    // it to the RNG would hit it about 1 pair in 65536. The
                    // matching activation lanes in `activations` carry a single
                    // sign per pair so the two products add instead of
                    // cancelling — see the note there.
                    let q: i8 = match t {
                        0 | 1 => -128,
                        2 | 3 => 127,
                        4 => -127,
                        _ => (lcg(st) % 256) as i32 as i8,
                    };
                    v.push(q as u8);
                }
            }
            v
        }

        fn q4_0_rows(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let nb = k / 32;
            let mut v = Vec::with_capacity(m * nb * size_of::<BlockQ4_0>());
            for blk in 0..m * nb {
                v.extend_from_slice(
                    &half::f16::from_f32(0.01 + 0.004 * (blk % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for _ in 0..16 {
                    v.push((lcg(st) % 256) as u8);
                }
            }
            v
        }

        /// Random Q4_1 weight rows (20-byte blocks: `d`, `m`, 16 nibble bytes).
        /// `d`/`m` are controlled rather than random f16 bits (which can be
        /// inf/NaN); the nibbles are random. `m` is deliberately negative on some
        /// blocks so the `+m·Σ(x)` term is genuinely exercised in both signs.
        fn q4_1_rows(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let nb = k / 32;
            let mut v = Vec::with_capacity(m * nb * size_of::<crate::quant::BlockQ4_1>());
            for blk in 0..m * nb {
                v.extend_from_slice(
                    &half::f16::from_f32(0.01 + 0.004 * (blk % 7) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                v.extend_from_slice(
                    &half::f16::from_f32(-0.05 + 0.02 * (blk % 5) as f32)
                        .to_bits()
                        .to_le_bytes(),
                );
                for _ in 0..16 {
                    v.push((lcg(st) % 256) as u8);
                }
            }
            v
        }

        /// Random Q4_K and Q6_K weight rows sharing one RNG stream.
        ///
        /// `d`/`dmin` are controlled rather than random bytes: random f16 bits
        /// can be inf/NaN, and a NaN output would compare bit-equal to itself
        /// and prove nothing in the bit-exactness test. Everything else — the
        /// 6-bit scales, the nibbles, the qh bits — is fully random.
        fn k_quant_rows(m: usize, sb: usize, st: &mut u64) -> (Vec<u8>, Vec<u8>) {
            let q4_sz = size_of::<crate::quant::BlockQ4KM>();
            let mut q4k = vec![0u8; m * sb * q4_sz];
            for (bi, chunk) in q4k.chunks_mut(q4_sz).enumerate() {
                let d = half::f16::from_f32(0.01 + 0.005 * (bi % 7) as f32);
                let dmin = half::f16::from_f32(0.02 + 0.003 * (bi % 5) as f32);
                chunk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                chunk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
                for b in chunk[4..].iter_mut() {
                    *b = (lcg(st) % 256) as u8;
                }
            }
            let q6_sz = size_of::<crate::quant::BlockQ6K>();
            let mut q6k = vec![0u8; m * sb * q6_sz];
            for (bi, chunk) in q6k.chunks_mut(q6_sz).enumerate() {
                for b in chunk[..q6_sz - 2].iter_mut() {
                    *b = (lcg(st) % 256) as u8;
                }
                // Derived, not the literal 208 the older fixtures hardcode: a
                // change to `BlockQ6K` should move this, not silently corrupt it.
                let d = half::f16::from_f32(0.01 + 0.004 * (bi % 6) as f32);
                chunk[q6_sz - 2..].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            (q4k, q6k)
        }

        /// Activation columns at the `[-127, 127]` bound the kernels require.
        ///
        /// The extreme lanes are placed to *add* rather than cancel against
        /// `q8_0_rows_with_extremes`. `maddubs` folds two adjacent products
        /// into one i16, and the sign trick rewrites the pair as
        /// `|w| * sign(a, w)`, so a pair only approaches the i16 bound when
        /// both products land on the same sign. Lanes 0|1 face `-128` weights
        /// and lanes 2|3 face `+127` weights; giving each pair a single
        /// activation sign yields `-32512` and `-32258` respectively — the two
        /// closest approaches any caller makes to `32767`. Opposite signs
        /// within a pair would sum to zero and test nothing.
        fn activations(n: usize, k: usize, st: &mut u64) -> (Vec<f32>, Vec<i8>) {
            let nb = k / 32;
            let mut scales = vec![0.0f32; n * nb];
            let mut quants = vec![0i8; n * k];
            for j in 0..n {
                for b in 0..nb {
                    scales[j * nb + b] = 0.02 + 0.001 * ((j + b) % 9) as f32;
                    for t in 0..32 {
                        quants[j * k + b * 32 + t] = match t {
                            0 | 1 => 127,
                            2 | 3 => -127,
                            _ => ((lcg(st) % 255) as i32 - 127) as i8,
                        };
                    }
                }
            }
            (scales, quants)
        }

        #[cfg(feature = "avx512")]
        fn assert_same_bits(avx2: &[f32], vnni: &[f32], tag: &str) {
            for (i, (a, v)) in avx2.iter().zip(vnni).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    v.to_bits(),
                    "{tag}: AVX2 emulation diverged from VNNI at index {i} ({a} vs {v})"
                );
            }
        }

        /// Shapes chosen to hit both tile paths and both remainders: full
        /// strips plus an `m % TILE_M` tail, full tiles plus an `n % STRIP_N`
        /// remainder, and an `n` below the tile width entirely.
        const SHAPES: [(usize, usize); 4] = [(9, 7), (8, 8), (3, 2), (16, 5)];

        /// Exact integer reference: the kernel's contract is
        /// `sum_b d_w[b] * d_x[b] * <weights, activations>`, done in i32 rather
        /// than by dequantizing to f32, so a mismatch is unambiguous instead of
        /// being folded into a second rounding.
        fn ref_gemm(
            weights: &[u8],
            bs: &[f32],
            bq: &[i8],
            m: usize,
            n: usize,
            k: usize,
            q4: bool,
        ) -> Vec<f32> {
            let nb = k / 32;
            let bsz = if q4 {
                size_of::<BlockQ4_0>()
            } else {
                size_of::<BlockQ8_0>()
            };
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for b in 0..nb {
                        let off = i * nb * bsz + b * bsz;
                        let (dw, acc) = if q4 {
                            let blk = unsafe { &*(weights.as_ptr().add(off) as *const BlockQ4_0) };
                            let mut acc = 0i32;
                            for t in 0..16 {
                                let byte = blk.qs[t];
                                acc += ((byte & 0xF) as i32 - 8) * bq[j * k + b * 32 + t] as i32;
                                acc +=
                                    ((byte >> 4) as i32 - 8) * bq[j * k + b * 32 + t + 16] as i32;
                            }
                            (half::f16::from_bits(blk.d).to_f32(), acc)
                        } else {
                            let blk = unsafe { &*(weights.as_ptr().add(off) as *const BlockQ8_0) };
                            let mut acc = 0i32;
                            for t in 0..32 {
                                acc += blk.quants[t] as i32 * bq[j * k + b * 32 + t] as i32;
                            }
                            (half::f16::from_bits(blk.delta).to_f32(), acc)
                        };
                        sum += dw * bs[j * nb + b] * acc as f32;
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        }

        /// Runs on any AVX2 host, including one with no VNNI — where the
        /// VNNI comparisons skip and this is the check that remains.
        ///
        /// The `1e-5` bound is chosen, not inherited, and it is worth being
        /// precise about what it does and does not catch.
        ///
        /// It catches what this test exists for: `maddubs` saturating (which
        /// moves the i32 accumulator by hundreds), a mis-scaled block, a
        /// transposed index — errors that are a large fraction of the output.
        /// It does *not* catch a single unit of the i32 accumulator on the Q8_0
        /// arm: one unit is `d_w * d_x` absolute against an output built from
        /// an accumulator of order 1e5, i.e. ~2e-6 relative, which sits inside
        /// `1e-5` no matter how the fixture is scaled. That is a property of
        /// `k`, not of the bound — a relative check cannot resolve one unit in
        /// 1e5 without dropping below f32 summation noise (~1e-7). The Q4_0 arm,
        /// whose accumulators are ~64x smaller, is the sensitive half.
        ///
        /// Bit-exact single-unit coverage comes from elsewhere and is not this
        /// test's job: `avx2_row_tiled_matches_per_row_bit_exact` (no VNNI
        /// needed) and the three `*_matches_vnni_bit_exact` tests compare bit
        /// patterns, where one unit is a failure by construction.
        ///
        /// Mutation-checked rather than argued: injecting a one-unit per-lane
        /// error into `dot32u` — which accumulates over lanes and blocks into
        /// something this bound can see — fails this test, and removing it
        /// passes. Honesty about that check: the `1e-3` this test used to carry
        /// also failed on that mutation. What `1e-3` did not have was margin.
        /// A saturating `maddubs` clamps an i16 that wanted to be larger, so
        /// the error is the overshoot: marginal cases cost a unit or two, but a
        /// genuinely out-of-range weight/activation pair costs hundreds — order
        /// 1e-3 relative at this fixture, i.e. landing right on the old bound.
        /// `1e-5` puts two orders between the bound and that case.
        #[test]
        fn avx2_gemm_matches_integer_reference() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            let k = 128;
            for (m, n) in SHAPES {
                let mut st = 0x77aa_33ccu64;
                let q4 = q4_0_rows(m, k, &mut st);
                let q8 = q8_0_rows_with_extremes(m, k, &mut st);
                let (bs, bq) = activations(n, k, &mut st);

                let mut got = vec![0.0f32; m * n];
                unsafe { gemm_q4_0_q8_0(&q4, &bs, &bq, &mut got, m, n, k) };
                let want = ref_gemm(&q4, &bs, &bq, m, n, k, true);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "q4_0 {m}x{n} index {i}: {g} vs reference {w}"
                    );
                }

                let mut got = vec![0.0f32; m * n];
                unsafe { gemm_q8_0_q8_0(&q8, &bs, &bq, &mut got, m, n, k) };
                let want = ref_gemm(&q8, &bs, &bq, m, n, k, false);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "q8_0 {m}x{n} index {i}: {g} vs reference {w}"
                    );
                }
            }
        }

        /// GEMV coverage that runs on any AVX2 host.
        ///
        /// The GEMVs are NOT the GEMM at n=1 for Q4_0/Q8_0 — they go through
        /// `row_dot_q4_0`/`row_dot_q8_0`, which the tiled GEMM never touches —
        /// so the GEMM test above does not cover them. They are the production
        /// decode path on every non-VNNI x86 host, where a wrong result is wrong
        /// tokens rather than a crash.
        #[test]
        fn avx2_gemv_matches_integer_reference() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            let k = 128;
            for m in [1usize, 3, 8, 9, 16] {
                let mut st = 0x2244_6688u64;
                let q4 = q4_0_rows(m, k, &mut st);
                let q8 = q8_0_rows_with_extremes(m, k, &mut st);
                // One activation column, in the GEMV (not column-major) layout.
                let (xs, xq) = activations(1, k, &mut st);

                let mut got = vec![0.0f32; m];
                unsafe { gemv_q4_0_q8_0(&q4, &xs, &xq, &mut got, m, k) };
                let want = ref_gemm(&q4, &xs, &xq, m, 1, k, true);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "q4_0 gemv m={m} row {i}: {g} vs reference {w}"
                    );
                }

                let mut got = vec![0.0f32; m];
                unsafe { gemv_q8_0_q8_0(&q8, &xs, &xq, &mut got, m, k) };
                let want = ref_gemm(&q8, &xs, &xq, m, 1, k, false);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "q8_0 gemv m={m} row {i}: {g} vs reference {w}"
                    );
                }
            }
        }

        /// Row tiling must not change a single bit, at the AVX2 constants.
        ///
        /// The VNNI module has this property test already, but only at *its*
        /// tile constants (8/4/4/8) and only on a host with VNNI. The shipping
        /// AVX2 constants are 4/2/4/4, and a CI runner is not guaranteed to have
        /// VNNI, so without this the strip kernel's agreement with the per-row
        /// kernel can go unpinned exactly where the code runs. The integer-reference test
        /// above would not catch a divergence that stays inside its `1e-5`
        /// bound; equality of bit patterns admits nothing.
        #[test]
        fn avx2_row_tiled_matches_per_row_bit_exact() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            // (m, n) against TILE_M = 2, STRIP_N = 4, TILE_N = 4: full strips
            // plus a tail row, full tiles plus a column remainder, and both
            // dimensions below one tile.
            let shapes = [(13, 11), (16, 8), (1, 5), (8, 2), (9, 4), (3, 4)];
            let k = 96;
            let nb = k / 32;

            for (m, n) in shapes {
                let mut st = 0x9e37_79b9u64 ^ ((m * 131 + n) as u64);
                let (b_scales, b_quants) = activations(n, k, &mut st);

                let a4 = q4_0_rows(m, k, &mut st);
                let a8 = q8_0_rows_with_extremes(m, k, &mut st);

                let mut tiled = vec![0.0f32; m * n];
                let mut per_row = vec![0.0f32; m * n];
                unsafe { gemm_q4_0_q8_0(&a4, &b_scales, &b_quants, &mut tiled, m, n, k) };
                let row_bytes = nb * size_of::<BlockQ4_0>();
                for (r, out_row) in per_row.chunks_mut(n).enumerate() {
                    // SAFETY: row `r` is in bounds — `a4` holds `m` rows of
                    // `row_bytes`, and `out_row` is exactly `n` wide.
                    unsafe {
                        gemm_q4_0_row(
                            a4.as_ptr().add(r * row_bytes),
                            &b_scales,
                            &b_quants,
                            out_row,
                            n,
                            nb,
                        )
                    };
                }
                assert_same_bits_untiled(&per_row, &tiled, "q4_0", m, n);

                let mut tiled = vec![0.0f32; m * n];
                let mut per_row = vec![0.0f32; m * n];
                unsafe { gemm_q8_0_q8_0(&a8, &b_scales, &b_quants, &mut tiled, m, n, k) };
                let row_bytes = nb * size_of::<BlockQ8_0>();
                for (r, out_row) in per_row.chunks_mut(n).enumerate() {
                    // SAFETY: as above, for the Q8_0 row stride.
                    unsafe {
                        gemm_q8_0_row(
                            a8.as_ptr().add(r * row_bytes),
                            &b_scales,
                            &b_quants,
                            out_row,
                            n,
                            nb,
                        )
                    };
                }
                assert_same_bits_untiled(&per_row, &tiled, "q8_0", m, n);
            }
        }

        /// Repacked Q4_0 (8-row interleave) must match the integer reference at
        /// the AVX2 tile constants, on any AVX2 host (no VNNI needed) — the
        /// coverage that remains where the VNNI equivalence test skips. `m` is a
        /// whole number of 8-row super-rows (the repack precondition); `n`
        /// spans full tiles, a column remainder, and below one tile.
        #[test]
        fn avx2_gemm_q4_0_8x8_matches_reference() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            let k = 128;
            for (m, n) in [(8usize, 7usize), (16, 4), (8, 1), (24, 5), (8, 3)] {
                let mut st = 0x5b8f_2a1du64 ^ ((m * 131 + n) as u64);
                let q4 = q4_0_rows(m, k, &mut st);
                let (bs, bq) = activations(n, k, &mut st);

                let (packed, scales) = crate::backend::cpu::repack_q4_0_8x8(&q4, m, k);
                let mut got = vec![0.0f32; m * n];
                unsafe { gemm_q4_0_8x8_q8_0(&packed, &scales, &bs, &bq, &mut got, m, n, k) };
                let want = ref_gemm(&q4, &bs, &bq, m, n, k, true);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "q4_0 repack {m}x{n} index {i}: {g} vs reference {w}"
                    );
                }
            }
        }

        /// Independent f64 reference for the repacked Q4_K GEMM: dequantize each
        /// element as `d·sc_s·q − dmin·mn_s` (q unsigned 0..15), dot with the
        /// activations, scale by the per-block activation scale, in f64. Shares
        /// no structure with the kernel, so a mismatch is unambiguous.
        fn ref_gemm_q4_k(
            a: &[u8],
            bs: &[f32],
            bq: &[i8],
            m: usize,
            n: usize,
            k: usize,
        ) -> Vec<f32> {
            let sb = k / 256;
            let nb32 = k / 32;
            let bsz = size_of::<crate::quant::BlockQ4KM>();
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for bi in 0..sb {
                        let blk = unsafe {
                            &*(a.as_ptr().add((i * sb + bi) * bsz)
                                as *const crate::quant::BlockQ4KM)
                        };
                        let d = half::f16::from_bits(blk.d).to_f32() as f64;
                        let dmin = half::f16::from_bits(blk.dmin).to_f32() as f64;
                        let (sc, mn) = crate::quant::decode_q4km_scales(&blk.scales);
                        for s in 0..8 {
                            let block = bi * 8 + s;
                            let (mut dp, mut sa) = (0i64, 0i64);
                            for e in 0..32 {
                                let byte = blk.qs[(s / 2) * 32 + e];
                                let q = if s.is_multiple_of(2) {
                                    byte & 0x0F
                                } else {
                                    byte >> 4
                                } as i64;
                                let av = bq[j * k + block * 32 + e] as i64;
                                dp += q * av;
                                sa += av;
                            }
                            let xs = bs[j * nb32 + block] as f64;
                            sum += xs
                                * (d * sc[s] as f64 * dp as f64 - dmin * mn[s] as f64 * sa as f64);
                        }
                    }
                    out[i * n + j] = sum as f32;
                }
            }
            out
        }

        #[test]
        fn avx2_gemm_q4_k_8x8_matches_reference() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            // k=512 is two super-blocks; shapes hit full column tiles, a
            // remainder, n below the tile width, and multiple super-rows.
            let k = 512;
            for (m, n) in [(8usize, 7usize), (16, 4), (8, 1), (24, 5), (8, 3)] {
                let mut st = 0x71b3_c0deu64 ^ ((m * 131 + n) as u64);
                let (q4k, _q6k) = k_quant_rows(m, k / 256, &mut st);
                let (bs, bq) = activations(n, k, &mut st);

                let (packed, dsc, dmn) = crate::backend::cpu::repack_q4_k_8x8(&q4k, m, k);
                let mut got = vec![0.0f32; m * n];
                unsafe { gemm_q4_k_8x8_q8_0(&packed, &dsc, &dmn, &bs, &bq, &mut got, m, n, k) };
                let want = ref_gemm_q4_k(&q4k, &bs, &bq, m, n, k);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                        "q4_k repack {m}x{n} index {i}: {g} vs reference {w}"
                    );
                }
            }
        }

        /// Exact bit-pattern equality, reporting the first differing element.
        fn assert_same_bits_untiled(per_row: &[f32], tiled: &[f32], tag: &str, m: usize, n: usize) {
            if let Some((i, (x, y))) = per_row
                .iter()
                .zip(tiled)
                .enumerate()
                .find(|(_, (x, y))| x.to_bits() != y.to_bits())
            {
                panic!(
                    "{tag} row-tiled diverged at {m}x{n} index {i} (row {}, col {}): \
                     per-row {x:e} vs tiled {y:e}",
                    i / n,
                    i % n,
                );
            }
        }

        /// The gate must be ON here, not merely off at `Scalar`.
        ///
        /// `int8_gemm_gate.rs` and `unbatchable_warning.rs` assert only the
        /// negative direction (false at `Scalar`). If `avx2_int8_available()`
        /// regressed to require VNNI, every non-VNNI x86 host would silently
        /// drop back to per-token prefill, both of those would still pass, and
        /// the parity tests would print SKIP and report green — the exact
        /// "green forever without running" failure they exist to prevent.
        #[test]
        fn avx2_host_reports_int8_gemm_available() {
            // Two separate questions, and conflating them has now produced the
            // same bug twice in this file's history:
            //
            //   1. Does the *hardware* have avx2+fma? That is what
            //      `CERA_REQUIRE_SIMD=avx2` asks about, so it is the CPUID
            //      check — a host without the ISA skips (or fails where CI says
            //      the ISA is present).
            //   2. Did the *tier* resolve to `Avx2` or above? A deliberate
            //      `CERA_CPU_TIER=scalar` run on capable hardware answers no,
            //      and that is not a failure — it is `int8_gemm_gate.rs`
            //      pulling its lever. Skip rather than assert.
            //
            // Gating (1) on the tier made `CERA_CPU_TIER=scalar
            // CERA_REQUIRE_SIMD=avx2` a hard failure; gating (2) on CPUID made
            // the assert fire under a plain downgrade.
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            if crate::backend::cpu_features::cpu_features().tier
                < crate::backend::cpu_features::CpuTier::Avx2
            {
                return;
            }
            assert!(
                crate::backend::cpu::int8_gemm_available(),
                "avx2+fma detected and the tier is Avx2 or better, but \
                 int8_gemm_available() is false — batched prefill and int8 \
                 decode would silently disable on every AVX2 host"
            );
        }

        /// A VNNI host must actually dispatch to the VNNI kernels.
        ///
        /// The emulation is bit-exact with `dpbusd`, which means no numeric
        /// assertion anywhere can tell the two apart — and every VNNI test calls
        /// `vnni::*` directly rather than through the dispatcher. So an arm that
        /// stopped selecting VNNI would cost the whole native-`dpbusd` speedup
        /// on every Zen 4/5 and Ice Lake host while CI stayed green forever.
        /// Verified: narrowing `vnni_int8_available()` to `> Avx512Vnni` —
        /// which no x86 tier satisfies, so VNNI is never selected — passed the
        /// entire suite before this test existed.
        #[cfg(feature = "avx512")]
        #[test]
        fn vnni_host_reports_vnni_kernels_available() {
            if !require_simd_or_skip("avx512vnni", both_callable()) {
                return;
            }
            // Same split as above: CPUID says the hardware can, the tier says
            // whether we chose to. A downgrade run skips instead of failing.
            if crate::backend::cpu_features::cpu_features().tier
                < crate::backend::cpu_features::CpuTier::Avx512Vnni
            {
                return;
            }
            assert!(
                super::super::super::cpu::vnni_int8_available(),
                "host has VNNI and the tier resolved to Avx512Vnni, but \
                 vnni_int8_available() is false — every VNNI host would run the \
                 AVX2 emulation instead, at no numeric cost and full speed cost"
            );
        }

        /// The K-quant decode path, on any AVX2 host.
        ///
        /// `gemv_q4k_f32`/`gemv_q6k_f32` are the whole Q4_K/Q6_K decode path on
        /// every non-VNNI x86 box. They are thin, but "thin" is where a wrong
        /// scratch resize or a transposed `n`/`k` lives, and the failure mode is
        /// wrong tokens rather than a crash.
        #[test]
        fn avx2_k_quant_gemv_matches_gemm() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            let (m, k) = (5usize, 512usize);
            let sb = k / 256;
            let mut st = 0x4b1d_7e11u64;
            let (q4k, q6k) = k_quant_rows(m, sb, &mut st);

            // A real f32 activation vector — the wrappers quantize it themselves,
            // which is precisely the step under test.
            let x: Vec<f32> = (0..k)
                .map(|_| (lcg(&mut st) % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            // The GEMM reference consumes the same activation, quantized by the
            // same function the wrapper uses, at n = 1.
            let mut xs = vec![0.0f32; k / 32];
            let mut xq = vec![0i8; k];
            crate::backend::cpu::quantize_f32_to_q8_0_into(&x, &mut xs, &mut xq);

            for (tag, w) in [("q4_k", &q4k), ("q6_k", &q6k)] {
                let (mut got, mut want) = (vec![0.0f32; m], vec![0.0f32; m]);
                // Deliberately pre-dirtied and oversized: the wrapper must resize
                // both scratch buffers, and a stale tail must not be read.
                let mut s_scratch = vec![7.0f32; k];
                let mut q_scratch = vec![7i8; k * 2];
                unsafe {
                    if tag == "q4_k" {
                        gemv_q4k_f32(w, &x, &mut got, m, k, &mut s_scratch, &mut q_scratch);
                        gemm_q4_k_q8_0(w, &xs, &xq, &mut want, m, 1, k);
                    } else {
                        gemv_q6k_f32(w, &x, &mut got, m, k, &mut s_scratch, &mut q_scratch);
                        gemm_q6_k_q8_0(w, &xs, &xq, &mut want, m, 1, k);
                    }
                }
                for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        wv.to_bits(),
                        "{tag} gemv row {i}: {g} vs gemm-at-n=1 {wv}"
                    );
                }
            }
        }

        /// The invariant the parity bar rests on: decode and batched prefill must
        /// compute the same thing.
        ///
        /// For Q4_K/Q6_K that holds by construction (the GEMV *is* the GEMM at
        /// n = 1). For Q4_0/Q8_0 it does NOT — the GEMV runs `row_dot_*` while
        /// the GEMM runs the tiled strip kernel — so it needs asserting, at the
        /// AVX2 tile constants specifically. Without this, retuning TILE_M or
        /// STRIP_N could break the 0.9999 parity bar with no unit test failing
        /// on the hosts that run this code.
        #[test]
        fn avx2_gemv_matches_gemm_at_n1() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            let k = 128;
            for m in [1usize, 2, 3, 8, 9] {
                let mut st = 0x33cc_55aau64;
                let q4 = q4_0_rows(m, k, &mut st);
                let q8 = q8_0_rows_with_extremes(m, k, &mut st);
                let (xs, xq) = activations(1, k, &mut st);
                for (tag, w, q4_0) in [("q4_0", &q4, true), ("q8_0", &q8, false)] {
                    let (mut gemv, mut gemm) = (vec![0.0f32; m], vec![0.0f32; m]);
                    unsafe {
                        if q4_0 {
                            gemv_q4_0_q8_0(w, &xs, &xq, &mut gemv, m, k);
                            gemm_q4_0_q8_0(w, &xs, &xq, &mut gemm, m, 1, k);
                        } else {
                            gemv_q8_0_q8_0(w, &xs, &xq, &mut gemv, m, k);
                            gemm_q8_0_q8_0(w, &xs, &xq, &mut gemm, m, 1, k);
                        }
                    }
                    for (i, (a, b)) in gemv.iter().zip(&gemm).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{tag} m={m} row {i}: gemv {a} != gemm-at-n=1 {b} — decode and \
                             prefill have diverged, which breaks the parity bar"
                        );
                    }
                }
            }
        }

        /// K-quant coverage that runs on any AVX2 host.
        ///
        /// Mirrors `gemm_q4_k_avx512_matches_dequant_reference` but without the
        /// VNNI gate. Without this, the Q4_K/Q6_K AVX2 kernels — live for every
        /// Q4_K_M model on every non-VNNI x86 box, and carrying the recentring
        /// and mins arithmetic that `q8_0_col_sums`/`dot16u` feed — are executed
        /// by no test on the hosts that actually run them.
        #[test]
        fn avx2_k_quant_gemm_matches_dequant_reference() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            let (m, k) = (3usize, 512usize);
            let sb = k / 256;
            let mut st = 0x9f1e_2d3cu64;
            let (q4k, q6k) = k_quant_rows(m, sb, &mut st);

            let q4_row = sb * size_of::<crate::quant::BlockQ4KM>();
            let q6_row = sb * size_of::<crate::quant::BlockQ6K>();
            // `n = 5` exercises the KQ_COLS-tiled path; `n = 1` the decode
            // multi-accumulator fast path (the `n == 1` gate), whose float
            // reduction order differs from the tiled path — cover both on this
            // tier since there is no separate AVX2 GEMV test for the n=1 kernel.
            for n in [5usize, 1usize] {
                let (bs, bq) = activations(n, k, &mut st);
                for (tag, weights, row_bytes) in [("q4_k", &q4k, q4_row), ("q6_k", &q6k, q6_row)] {
                    let mut got = vec![0.0f32; m * n];
                    unsafe {
                        if tag == "q4_k" {
                            gemm_q4_k_q8_0(weights, &bs, &bq, &mut got, m, n, k);
                        } else {
                            gemm_q6_k_q8_0(weights, &bs, &bq, &mut got, m, n, k);
                        }
                    }
                    for i in 0..m {
                        let mut w = vec![0.0f32; k];
                        let row = &weights[i * row_bytes..(i + 1) * row_bytes];
                        if tag == "q4_k" {
                            crate::quant::dequantize_q4_k_m_row(row, &mut w);
                        } else {
                            crate::quant::dequantize_q6_k_row(row, &mut w);
                        }
                        for j in 0..n {
                            let mut want = 0.0f64;
                            for e in 0..k {
                                let xa = bs[j * (k / 32) + e / 32] * bq[j * k + e] as f32;
                                want += (w[e] * xa) as f64;
                            }
                            let g = got[i * n + j] as f64;
                            assert!(
                                // 1e-5, not the 1e-3 the older VNNI fixtures use:
                                // instrumented, the real error here is 4e-6..1.7e-4,
                                // and at 1e-3 a +/-1 error in the Q6_K recentring
                                // term passes. This still leaves ~50x headroom.
                                (g - want).abs() <= 1e-5 * (1.0 + want.abs()),
                                "{tag} n={n} [{i},{j}]: got {g} want {want}"
                            );
                        }
                    }
                }
            }
        }

        /// Q4_1 coverage that runs on any AVX2 host.
        ///
        /// The x86 `gemm_q4_1_q8_0` has no aarch64 twin to lean on for these
        /// hosts and carries the `+m·Σ(x)` min term (the piece most likely to
        /// harbor a lane-packing or sign bug), so it needs its own parity bar —
        /// the same reason `avx2_k_quant_gemm_matches_dequant_reference` exists
        /// for the K-quant kernels. Reference: dequantize each Q4_1 row and dot it
        /// against the reconstructed Q8_0 activations in f64.
        #[test]
        fn avx2_gemm_q4_1_matches_dequant_reference() {
            if !require_simd_or_skip("avx2", avx2_kernels_callable()) {
                return;
            }
            // n = 11 straddles KQ_COLS (8): a full 8-column pass plus a 3-column
            // tail, so a bug at the x86 kernel's full-width→tail column boundary
            // cannot hide (the sibling K-quant test uses n=5 and only ever hits
            // the tail; this new kernel warrants the wider case).
            let (m, n, k) = (3usize, 11usize, 512usize);
            let mut st = 0x0451_7b3du64;
            let weights = q4_1_rows(m, k, &mut st);
            let (bs, bq) = activations(n, k, &mut st);
            let row_bytes = (k / 32) * size_of::<crate::quant::BlockQ4_1>();

            let mut got = vec![0.0f32; m * n];
            unsafe { gemm_q4_1_q8_0(&weights, &bs, &bq, &mut got, m, n, k) };

            for i in 0..m {
                let mut w = vec![0.0f32; k];
                crate::quant::dequantize_q4_1_row(
                    &weights[i * row_bytes..(i + 1) * row_bytes],
                    &mut w,
                );
                for j in 0..n {
                    let mut want = 0.0f64;
                    for e in 0..k {
                        let xa = bs[j * (k / 32) + e / 32] * bq[j * k + e] as f32;
                        want += (w[e] * xa) as f64;
                    }
                    let g = got[i * n + j] as f64;
                    assert!(
                        (g - want).abs() <= 1e-5 * (1.0 + want.abs()),
                        "q4_1 [{i},{j}]: got {g} want {want}"
                    );
                }
            }
        }

        #[test]
        #[cfg(feature = "avx512")]
        fn avx2_q8_0_gemm_matches_vnni_bit_exact() {
            if !require_simd_or_skip("avx512vnni", both_callable()) {
                return;
            }
            let k = 128;
            for (m, n) in SHAPES {
                let mut st = 0xabcd_1234u64;
                let w = q8_0_rows_with_extremes(m, k, &mut st);
                let (bs, bq) = activations(n, k, &mut st);
                let mut got = vec![0.0f32; m * n];
                let mut want = vec![0.0f32; m * n];
                unsafe {
                    gemm_q8_0_q8_0(&w, &bs, &bq, &mut got, m, n, k);
                    vnni::gemm_q8_0_q8_0(&w, &bs, &bq, &mut want, m, n, k);
                }
                assert_same_bits(&got, &want, &format!("q8_0 {m}x{n}"));
            }
        }

        #[test]
        #[cfg(feature = "avx512")]
        fn avx2_q4_0_gemm_matches_vnni_bit_exact() {
            if !require_simd_or_skip("avx512vnni", both_callable()) {
                return;
            }
            let k = 128;
            for (m, n) in SHAPES {
                let mut st = 0x5eed_9911u64;
                let w = q4_0_rows(m, k, &mut st);
                let (bs, bq) = activations(n, k, &mut st);
                let mut got = vec![0.0f32; m * n];
                let mut want = vec![0.0f32; m * n];
                unsafe {
                    gemm_q4_0_q8_0(&w, &bs, &bq, &mut got, m, n, k);
                    vnni::gemm_q4_0_q8_0(&w, &bs, &bq, &mut want, m, n, k);
                }
                assert_same_bits(&got, &want, &format!("q4_0 {m}x{n}"));
            }
        }

        #[test]
        #[cfg(feature = "avx512")]
        fn avx2_k_quant_gemm_matches_vnni_bit_exact() {
            if !require_simd_or_skip("avx512vnni", both_callable()) {
                return;
            }
            let (m, n, k) = (3usize, 5usize, 512usize);
            let sb = k / 256;
            let mut st = 0x1357_2468u64;

            let (q4k, q6k) = k_quant_rows(m, sb, &mut st);

            let (bs, bq) = activations(n, k, &mut st);
            for (tag, weights, is_q4k) in [("q4_k", &q4k, true), ("q6_k", &q6k, false)] {
                let mut got = vec![0.0f32; m * n];
                let mut want = vec![0.0f32; m * n];
                unsafe {
                    if is_q4k {
                        gemm_q4_k_q8_0(weights, &bs, &bq, &mut got, m, n, k);
                        vnni::gemm_q4_k_q8_0(weights, &bs, &bq, &mut want, m, n, k);
                    } else {
                        gemm_q6_k_q8_0(weights, &bs, &bq, &mut got, m, n, k);
                        vnni::gemm_q6_k_q8_0(weights, &bs, &bq, &mut want, m, n, k);
                    }
                }
                assert_same_bits(&got, &want, tag);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod avx512_vnni {
    use super::*;
    use std::arch::x86_64::*;

    /// Unsigned-weight int8 dot: the K-quant kernels hand `dpbusd` weights that
    /// are already unsigned (Q4_K 0..15, Q6_K 0..63, or a literal 1 when summing
    /// activations), so they need no sign trick — the instruction's operand
    /// contract is met directly.
    #[inline]
    #[target_feature(enable = "avx2,avx512vl,avx512vnni")]
    unsafe fn dot32u(w: __m256i, a: __m256i) -> __m256i {
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, a)
    }

    /// 128-bit `dot32u`, for Q6_K's 16-element sub-block activation sums.
    #[inline]
    #[target_feature(enable = "avx2,avx512vl,avx512vnni")]
    unsafe fn dot16u(w: __m128i, a: __m128i) -> __m128i {
        _mm_dpbusd_epi32(_mm_setzero_si128(), w, a)
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
    /// reaching here comes from `quantize_f32_to_q8_0_*`, which scales by
    /// `127 / amax` and guards the non-finite case, so the rounded magnitude is
    /// 127 (the worst case observed by exhaustive search over the f32 exponent
    /// range is 127.00003, which rounds to 127). The one input that could reach
    /// the quantizer's `-128.0` clamp is a block whose `amax` is denormal, where
    /// `d = amax / 127` loses enough precision that `v * (1/d)` overshoots; that
    /// block is harmless for a second reason — its `f16` scale is exactly 0.0,
    /// so the lane contributes nothing whatever its sign.
    #[inline]
    #[target_feature(enable = "avx2,avx512vl,avx512vnni")]
    unsafe fn dot32(w: __m256i, a: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(w, w);
        let sy = _mm256_sign_epi8(a, w);
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy)
    }

    // EVEX exposes 32 vector registers here, so the tiles are the wide ones:
    // TILE_N=8 per-row columns, and a 4x4 (16-accumulator) row-tiled strip.
    int8_gemm_kernels!("avx512f,avx512vl,avx512vnni,avx2,fma", 8, 4, 4, 8);

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

        /// The quantizer's own feature set — strictly weaker than the kernels'.
        ///
        /// `quantize_f32_to_q8_0_avx512` declares `avx512f,avx512vl,avx2` and
        /// uses no `dpbusd`, and `cpu::avx512_quantizer_available` dispatches to
        /// it at the plain `Avx512` tier. Gating its tests on
        /// `vnni_kernels_callable()` would skip them on exactly the host class
        /// that widening was for — the one where scalar-vs-AVX-512 bit equality
        /// newly became load-bearing.
        fn quantizer_callable() -> bool {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx2")
                // Not in the kernel's `#[target_feature]` list, but
                // `avx512_quantizer_available` gates on `tier >= Avx512`, which
                // `detect()` only awards with `fma`. Matching the predicate
                // keeps the test from opening where the predicate would not.
                && is_x86_feature_detected!("fma")
        }

        use crate::backend::simd::require_simd_or_skip;

        /// Independently written mirror of the Q8_0 quantizer contract: f16
        /// round-trip of `d`, round-to-nearest-even, the non-finite guard.
        ///
        /// Deliberately NOT a call to `cpu::quantize_f32_to_q8_0_scalar`, even
        /// though that is now reachable from here. This is the *oracle* — the
        /// only thing in the tree that pins either quantizer to a
        /// separately-written statement of the spec. Delegating would turn
        /// `quantize_q8_0_avx512_matches_scalar` into production-vs-production,
        /// which `quantize_q8_0_scalar_matches_avx512` already covers, and would
        /// leave a shared misreading of the spec undetectable.
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

        /// K-quant GEMM vs literal dequantize-then-dot (same quantized
        /// activations, f64 reference accumulation). Odd `n` exercises the
        /// KQ_COLS tail; two super-blocks exercise the `bi` loop.
        #[test]
        fn gemm_q4_k_avx512_matches_dequant_reference() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, n, k) = (3usize, 5usize, 512usize);
            let sb = k / 256;
            let mut st = 0x51ed_c0deu64;

            // Random Q4_K rows: controlled d/dmin (random f16 bits can be
            // inf/NaN), fully random scales and nibbles.
            let row_bytes = sb * size_of::<crate::quant::BlockQ4KM>();
            let mut a = vec![0u8; m * row_bytes];
            for (bi, chunk) in a
                .chunks_mut(size_of::<crate::quant::BlockQ4KM>())
                .enumerate()
            {
                let d = half::f16::from_f32(0.01 + 0.005 * (bi % 7) as f32);
                let dmin = half::f16::from_f32(0.02 + 0.003 * (bi % 5) as f32);
                chunk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                chunk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
                for b in chunk[4..].iter_mut() {
                    *b = (lcg01(&mut st) * 255.0) as u8;
                }
            }

            // Random activations, pre-quantized to Q8_0 in the GEMM layout.
            let mut b_scales = vec![0.0f32; n * (k / 32)];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (cs, cq) = ref_quantize(&col);
                b_scales[j * (k / 32)..(j + 1) * (k / 32)].copy_from_slice(&cs);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&cq);
            }

            let mut got = vec![0.0f32; m * n];
            unsafe { gemm_q4_k_q8_0(&a, &b_scales, &b_quants, &mut got, m, n, k) };

            for i in 0..m {
                let mut w = vec![0.0f32; k];
                crate::quant::dequantize_q4_k_m_row(&a[i * row_bytes..(i + 1) * row_bytes], &mut w);
                for j in 0..n {
                    let mut want = 0.0f64;
                    for e in 0..k {
                        let xa = b_scales[j * (k / 32) + e / 32] * b_quants[j * k + e] as f32;
                        want += (w[e] * xa) as f64;
                    }
                    let g = got[i * n + j] as f64;
                    assert!(
                        (g - want).abs() <= 1e-3 * (1.0 + want.abs()),
                        "[{i},{j}]: got {g} want {want}"
                    );
                }
            }
        }

        #[test]
        fn gemm_q6_k_avx512_matches_dequant_reference() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, n, k) = (3usize, 5usize, 512usize);
            let sb = k / 256;
            let mut st = 0x6b1d_5ca1u64;

            let row_bytes = sb * size_of::<crate::quant::BlockQ6K>();
            let mut a = vec![0u8; m * row_bytes];
            for (bi, chunk) in a
                .chunks_mut(size_of::<crate::quant::BlockQ6K>())
                .enumerate()
            {
                for b in chunk[..208].iter_mut() {
                    *b = (lcg01(&mut st) * 255.0) as u8;
                }
                let d = half::f16::from_f32(0.008 + 0.004 * (bi % 5) as f32);
                chunk[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }

            let mut b_scales = vec![0.0f32; n * (k / 32)];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (cs, cq) = ref_quantize(&col);
                b_scales[j * (k / 32)..(j + 1) * (k / 32)].copy_from_slice(&cs);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&cq);
            }

            let mut got = vec![0.0f32; m * n];
            unsafe { gemm_q6_k_q8_0(&a, &b_scales, &b_quants, &mut got, m, n, k) };

            for i in 0..m {
                let mut w = vec![0.0f32; k];
                crate::quant::dequantize_q6_k_row(&a[i * row_bytes..(i + 1) * row_bytes], &mut w);
                for j in 0..n {
                    let mut want = 0.0f64;
                    for e in 0..k {
                        let xa = b_scales[j * (k / 32) + e / 32] * b_quants[j * k + e] as f32;
                        want += (w[e] * xa) as f64;
                    }
                    let g = got[i * n + j] as f64;
                    assert!(
                        (g - want).abs() <= 1e-3 * (1.0 + want.abs()),
                        "[{i},{j}]: got {g} want {want}"
                    );
                }
            }
        }

        /// The GEMV wrappers against dequantized weights dotted with the
        /// *quantized* activations (`ref_quantize`, which
        /// `quantize_q8_0_avx512_matches_scalar` proves bit-identical to the
        /// kernel quantizer). The GEMM tests above are handed pre-quantized
        /// activations, so they cannot catch a wrapper bug — a stale scratch
        /// resize, a swapped scale/quant argument; this exercises the
        /// quantize-then-dot pipeline against a reference that shares the
        /// quantization, keeping the bar tight. (Comparing against the raw f32
        /// activations instead would fold ±½-step quantization noise into the
        /// tolerance — ~1.4σ misses at 2% on this data — and testing the
        /// quantizer is not this test's job.)
        #[test]
        fn gemv_q4k_avx512_matches_dequant_reference() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, k) = (8usize, 512usize);
            let sb = k / 256;
            let mut st = 0x7a11_ce55u64;
            let row_bytes = sb * size_of::<crate::quant::BlockQ4KM>();
            let mut a = vec![0u8; m * row_bytes];
            for (bi, chunk) in a
                .chunks_mut(size_of::<crate::quant::BlockQ4KM>())
                .enumerate()
            {
                let d = half::f16::from_f32(0.01 + 0.004 * (bi % 5) as f32);
                let dmin = half::f16::from_f32(0.015);
                chunk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                chunk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
                for b in chunk[4..].iter_mut() {
                    *b = (lcg01(&mut st) * 255.0) as u8;
                }
            }
            let x: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();

            let mut y = vec![0.0f32; m];
            let mut scr_s = Vec::new();
            let mut scr_q = Vec::new();
            // Pre-dirty the scratch with wrong-sized garbage: the wrapper must
            // resize and overwrite, not trust what it was handed.
            scr_s.resize(3, 9.9);
            scr_q.resize(7, 99);
            unsafe { gemv_q4k_f32(&a, &x, &mut y, m, k, &mut scr_s, &mut scr_q) };

            let (xs, xq) = ref_quantize(&x);
            for i in 0..m {
                let mut w = vec![0.0f32; k];
                crate::quant::dequantize_q4_k_m_row(&a[i * row_bytes..(i + 1) * row_bytes], &mut w);
                let want: f64 = w
                    .iter()
                    .enumerate()
                    .map(|(e, &we)| (we * xs[e / 32] * xq[e] as f32) as f64)
                    .sum();
                let g = y[i] as f64;
                assert!(
                    (g - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "row {i}: got {g} want {want}"
                );
            }
        }

        #[test]
        fn gemv_q6k_avx512_matches_dequant_reference() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, k) = (8usize, 512usize);
            let sb = k / 256;
            let mut st = 0x2f5e_11d3u64;
            let row_bytes = sb * size_of::<crate::quant::BlockQ6K>();
            let mut a = vec![0u8; m * row_bytes];
            for chunk in a.chunks_mut(size_of::<crate::quant::BlockQ6K>()) {
                for b in chunk[..208].iter_mut() {
                    *b = (lcg01(&mut st) * 255.0) as u8;
                }
                let d = half::f16::from_f32(0.01);
                chunk[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            let x: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();

            let mut y = vec![0.0f32; m];
            let mut scr_s = Vec::new();
            let mut scr_q = Vec::new();
            unsafe { gemv_q6k_f32(&a, &x, &mut y, m, k, &mut scr_s, &mut scr_q) };

            let (xs, xq) = ref_quantize(&x);
            for i in 0..m {
                let mut w = vec![0.0f32; k];
                crate::quant::dequantize_q6_k_row(&a[i * row_bytes..(i + 1) * row_bytes], &mut w);
                let want: f64 = w
                    .iter()
                    .enumerate()
                    .map(|(e, &we)| (we * xs[e / 32] * xq[e] as f32) as f64)
                    .sum();
                let g = y[i] as f64;
                assert!(
                    (g - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "row {i}: got {g} want {want}"
                );
            }
        }

        #[test]
        fn quantize_q8_0_avx512_matches_scalar() {
            if !require_simd_or_skip("avx512", quantizer_callable()) {
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

        /// The *production* scalar quantizer, bit-for-bit against the AVX-512
        /// one.
        ///
        /// `quantize_q8_0_avx512_matches_scalar` above compares against
        /// `ref_quantize`, an independently written oracle — which proves the
        /// kernel matches the spec, not that it matches the code every other
        /// host runs. The interop claim that actually ships is different:
        /// `quantize_f32_to_q8_0_into` picks one of these two by tier, decode and
        /// prefill both go through it, and the AVX2 GEMM consumes whichever it
        /// produced. If the
        /// two ever disagree, a host that quantizes activations one way and
        /// weights another gets silently wrong logits — no panic, no warning.
        ///
        /// Includes the `-127`/`127` bound cases and an all-zero block, where
        /// `d == 0` forces the reciprocal branch that differs most between the
        /// two implementations.
        #[test]
        fn quantize_q8_0_scalar_matches_avx512() {
            if !require_simd_or_skip("avx512", quantizer_callable()) {
                return;
            }
            let mut st = 0x2468_ace0u64;
            let mut x: Vec<f32> = (0..320).map(|_| lcg01(&mut st) * 8.0 - 4.0).collect();
            // Block 1: every lane subnormal, so `1.0 / d` overflows to
            // infinity. This is the *discriminating* block — it is
            // the only input where the two implementations could disagree, and
            // it is what the non-finite guards on both sides exist for. A single
            // tiny lane among normal ones proves nothing: `amax` comes from the
            // whole block, so one `1e-30` next to a `3.9` leaves `d` normal.
            // amax ~1e-40: subnormal, but not so small that `d = amax / 127`
            // (~1e-42 here) underflows to exactly zero — at that point both
            // sides take the `d == 0` branch and agree for the wrong reason.
            // Here `d` is a nonzero subnormal and `1.0 / d` overflows to
            // infinity, which is the case the guards actually handle.
            for (t, v) in x[32..64].iter_mut().enumerate() {
                *v = f32::from_bits(71_000 + (t as u32 % 11) * 37);
            }
            // Block 2: all zero, the `d == 0` branch.
            x[64..96].fill(0.0);
            // And the bound cases in an otherwise ordinary block.
            x[100] = -7.5;
            x[101] = 7.5;

            let nb = x.len() / 32;
            let (mut ss, mut sq) = (vec![0.0f32; nb], vec![0i8; x.len()]);
            let (mut vs, mut vq) = (vec![0.0f32; nb], vec![0i8; x.len()]);
            crate::backend::cpu::quantize_f32_to_q8_0_scalar(&x, &mut ss, &mut sq);
            unsafe { quantize_f32_to_q8_0_avx512(&x, &mut vs, &mut vq) };

            for (b, (a, v)) in ss.iter().zip(&vs).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    v.to_bits(),
                    "block {b} scale: scalar {a:e} vs avx512 {v:e}"
                );
            }
            assert_eq!(sq, vq, "quantized bytes diverged");
        }

        /// An all-zero block makes `d == 0`, so the reciprocal is forced to 0
        /// rather than inf — check the kernel takes that branch too.
        #[test]
        fn quantize_q8_0_avx512_handles_zero_block() {
            if !require_simd_or_skip("avx512", quantizer_callable()) {
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
            if !require_simd_or_skip("avx512", quantizer_callable()) {
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
        fn gemv_q4_0_q8_0_matches_scalar() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, k) = (37, 128); // odd m exercises the row tail
            let mut st = 0x2468_1357u64;
            let a = rand_q4_0_rows(m, k, &mut st);
            let x: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
            let (xs, xq) = ref_quantize(&x);
            let mut y = vec![0.0f32; m];
            unsafe { gemv_q4_0_q8_0(&a, &xs, &xq, &mut y, m, k) };
            assert_close(&y, &ref_gemv_q4_0(&a, &xs, &xq, m, k), "gemv_q4_0");
        }

        #[test]
        fn gemv_q8_0_q8_0_matches_scalar() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, k) = (37, 128);
            let mut st = 0x9876_5432u64;
            let a = rand_q8_0_rows(m, k, &mut st);
            let x: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
            let (xs, xq) = ref_quantize(&x);
            let mut y = vec![0.0f32; m];
            unsafe { gemv_q8_0_q8_0(&a, &xs, &xq, &mut y, m, k) };
            assert_close(&y, &ref_gemv_q8_0(&a, &xs, &xq, m, k), "gemv_q8_0");
        }

        /// The GEMM must agree with the GEMV column-by-column.
        ///
        /// The shape is load-bearing, and now on three constants, not one. Keep
        /// ALL of these true when any of them is retuned — a fixed `(m, n)` that
        /// was fine at one tiling can silently degenerate into partial coverage
        /// at the next, with the suite still green:
        ///   - `n > TILE_N && n % TILE_N != 0` — the per-row kernel's tile loop
        ///     and its column remainder (reached via the `m % TILE_M` tail).
        ///   - `n > STRIP_N && n % STRIP_N != 0` — the strip kernel's tile loop
        ///     and its column remainder.
        ///   - `m > TILE_M && m % TILE_M != 0` — full strips plus the short final
        ///     strip that falls back to the per-row kernel.
        ///
        /// `gemm_avx512_row_tiled_matches_per_row_bit_exact` covers the same
        /// branches across several shapes and is the better guard; this test adds
        /// an independent oracle (the GEMV) rather than another shape.
        #[test]
        fn gemm_q4_0_avx512_matches_gemv_per_column() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, n, k) = (13, 11, 96);
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
            unsafe { gemm_q4_0_q8_0(&a, &b_scales, &b_quants, &mut out, m, n, k) };

            for j in 0..n {
                let mut y = vec![0.0f32; m];
                unsafe {
                    gemv_q4_0_q8_0(
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

        /// The row-tiled drivers must be **bit-identical** to driving the per-row
        /// kernels directly — a column's accumulator chain is the same fmadd
        /// sequence in the same order either way, so any difference is a bug.
        ///
        /// This is the guard for the property the row-tiling comment asserts, and
        /// it runs in CI (unlike `microbench_gemm_rowtile`, which is `#[ignore]`d).
        /// Data is pseudo-random, not a repeated constant: with uniform inputs a
        /// transposed index or a wrong row offset still yields identical output,
        /// so a constant-filled comparison cannot see the bug class row tiling
        /// introduces.
        ///
        /// Shapes are chosen to cover every path, and each is annotated with what
        /// it exercises so the coverage survives a `TILE_M`/`STRIP_N` retune.
        #[test]
        fn gemm_avx512_row_tiled_matches_per_row_bit_exact() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            // (m, n): full strips + tail rows, full tiles + column remainder.
            let shapes = [
                (13, 11), // 3 strips + 1 tail row; 2 tiles + 3 remainder cols
                (16, 8),  // exact strips, exact tiles: no remainder at all
                (3, 5),   // m < TILE_M: every row on the tail path
                (8, 2),   // n < STRIP_N: tiled loop never runs, all remainder
                (9, 4),   // exactly one tile wide, 2 strips + 1 tail row
            ];
            let k = 96;
            let nb = k / 32;

            for (m, n) in shapes {
                let mut st = 0x51de_0000u64 ^ ((m * 131 + n) as u64);
                let mut b_scales = vec![0.0f32; n * nb];
                let mut b_quants = vec![0i8; n * k];
                for j in 0..n {
                    let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                    let (s, q) = ref_quantize(&col);
                    b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                    b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
                }

                // Q4_0
                let a4 = rand_q4_0_rows(m, k, &mut st);
                let mut tiled = vec![0.0f32; m * n];
                let mut per_row = vec![0.0f32; m * n];
                unsafe { gemm_q4_0_q8_0(&a4, &b_scales, &b_quants, &mut tiled, m, n, k) };
                let row_bytes4 = nb * size_of::<BlockQ4_0>();
                for (r, out_row) in per_row.chunks_mut(n).enumerate() {
                    // SAFETY: row `r` is in bounds of `a4` (m rows of row_bytes4).
                    unsafe {
                        gemm_q4_0_row(
                            a4.as_ptr().add(r * row_bytes4),
                            &b_scales,
                            &b_quants,
                            out_row,
                            n,
                            nb,
                        )
                    };
                }
                assert_bits_eq(&per_row, &tiled, "q4_0", m, n);

                // Q8_0
                let a8 = rand_q8_0_rows(m, k, &mut st);
                let mut tiled = vec![0.0f32; m * n];
                let mut per_row = vec![0.0f32; m * n];
                unsafe { gemm_q8_0_q8_0(&a8, &b_scales, &b_quants, &mut tiled, m, n, k) };
                let row_bytes8 = nb * size_of::<BlockQ8_0>();
                for (r, out_row) in per_row.chunks_mut(n).enumerate() {
                    // SAFETY: row `r` is in bounds of `a8` (m rows of row_bytes8).
                    unsafe {
                        gemm_q8_0_row(
                            a8.as_ptr().add(r * row_bytes8),
                            &b_scales,
                            &b_quants,
                            out_row,
                            n,
                            nb,
                        )
                    };
                }
                assert_bits_eq(&per_row, &tiled, "q8_0", m, n);
            }
        }

        /// Exact bit-pattern equality, reporting the first differing element.
        fn assert_bits_eq(per_row: &[f32], tiled: &[f32], tag: &str, m: usize, n: usize) {
            if let Some((i, (x, y))) = per_row
                .iter()
                .zip(tiled)
                .enumerate()
                .find(|(_, (x, y))| x.to_bits() != y.to_bits())
            {
                panic!(
                    "{tag} row-tiled diverged at {}x{} index {i} (row {}, col {}): \
                     per-row {x:e} ({:#010x}) vs tiled {y:e} ({:#010x})",
                    m,
                    n,
                    i / n,
                    i % n,
                    x.to_bits(),
                    y.to_bits()
                );
            }
        }

        /// Prefill GEMM throughput against this machine's int8 peak.
        ///
        /// This GEMM accounts for ~56% of prefill samples (samply, Llama-3.2-1B
        /// Q8_0, pp512), so its efficiency is the prefill number. Shape is one
        /// real Llama-1B projection at pp512.
        ///
        /// Run with:
        /// `cargo test -p cera --release --lib backend::simd::avx512_vnni::avx512_vnni_tests::microbench_gemm -- --ignored --nocapture`
        #[test]
        #[ignore]
        fn microbench_gemm() {
            // `vnni_kernels_callable()` is the full conjunction (F/VL/VNNI/AVX2/
            // FMA); the feature name is just the headline for the skip/require
            // message. Consistent with the correctness tests, and honours
            // `CERA_REQUIRE_SIMD=avx512vnni` (fail instead of skip).
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            use std::time::Instant;
            let (m, n, k) = (2048usize, 512usize, 2048usize);
            let nb = k / 32;
            // 200 iterations, both arms. A ~0.2s window (20 iters) samples
            // whichever clock state the process lands in and isn't long enough
            // to resolve a tile-width effect.
            let iters = 200;
            let ops = 2.0 * m as f64 * n as f64 * k as f64;

            let run = || {
                // m is a multiple of TILE_M, so every chunk is a full strip and
                // this measures the row-tiled path (TILE_M x STRIP_N). It does
                // NOT measure TILE_N — that governs only the `m % TILE_M` tail —
                // so the header names the constants actually under test.
                let report = |tag: &str, secs: f64| {
                    eprintln!(
                        "=== {tag} {m}x{n}x{k} (TILE_M={TILE_M}, STRIP_N={STRIP_N}) ===\n  {:.1} ms/call   {:.0} GOP/s",
                        secs * 1e3,
                        ops / secs / 1e9
                    );
                };
                // Q4_0 first: the tile constants are shared with that kernel,
                // whose nibble unpack needs extra temporaries, so a tile size
                // good for Q8_0 can regress it.
                {
                    let a4 = vec![7u8; m * nb * size_of::<BlockQ4_0>()];
                    let bs = vec![0.01f32; n * nb];
                    let bq = vec![3i8; n * k];
                    let mut c4 = vec![0.0f32; m * n];
                    unsafe { gemm_q4_0_q8_0(&a4, &bs, &bq, &mut c4, m, n, k) };
                    let t = Instant::now();
                    for _ in 0..iters {
                        unsafe { gemm_q4_0_q8_0(&a4, &bs, &bq, &mut c4, m, n, k) };
                    }
                    report("gemm_q4_0", t.elapsed().as_secs_f64() / iters as f64);
                }
                let a = vec![7u8; m * nb * size_of::<BlockQ8_0>()];
                let b_scales = vec![0.01f32; n * nb];
                let b_quants = vec![3i8; n * k];
                let mut c = vec![0.0f32; m * n];

                unsafe { gemm_q8_0_q8_0(&a, &b_scales, &b_quants, &mut c, m, n, k) };
                let t = Instant::now();
                for _ in 0..iters {
                    unsafe { gemm_q8_0_q8_0(&a, &b_scales, &b_quants, &mut c, m, n, k) };
                }
                report("gemm_q8_0", t.elapsed().as_secs_f64() / iters as f64);
            };

            // Pin the pool to the physical (performance) core count so the
            // number is reproducible. The default rayon pool is all logical
            // CPUs; with SMT, threads land on siblings differently each run and
            // this kernel spanned 417-1154 GOP/s on an identical binary — a ~2.8x
            // swing that dwarfs any tile-width effect. Pinning collapses it to
            // ~9%. Done here rather than left to `RAYON_NUM_THREADS` so the
            // benchmark is self-contained.
            #[cfg(feature = "parallel")]
            rayon::ThreadPoolBuilder::new()
                .num_threads(crate::backend::cpu_features::performance_core_count())
                .build()
                .expect("build fixed-size rayon pool")
                .install(run);
            #[cfg(not(feature = "parallel"))]
            run();
        }

        // ── Row-tiling A/B (task #17) ───────────────────────────────────────
        //
        // The production GEMM drivers are now row-tiled (`gemm_*_strip`). These
        // per-row reference drivers reproduce the pre-tiling behaviour — one
        // weight row per task — so `microbench_gemm_rowtile` can measure the
        // shipped kernels against the design they replaced, in one process on
        // the same pinned rayon pool. Read the paired win/loss line, not the
        // means (see `microbench_gemm` for why).

        /// Pre-row-tiling Q8_0 driver: one weight row per task.
        #[cfg(feature = "parallel")]
        #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
        unsafe fn ref_gemm_q8_0_perrow(
            a_quant: &[u8],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            n: usize,
            k: usize,
        ) {
            use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            let nb = k / 32;
            let row_bytes = nb * size_of::<BlockQ8_0>();
            let base = a_quant.as_ptr() as usize;
            out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
                // SAFETY: row `i` reads its own `row_bytes` span; writes `out_row`.
                unsafe {
                    gemm_q8_0_row(
                        (base as *const u8).wrapping_add(i * row_bytes),
                        b_scales,
                        b_quants,
                        out_row,
                        n,
                        nb,
                    );
                }
            });
        }

        /// Pre-row-tiling Q4_0 driver: one weight row per task.
        #[cfg(feature = "parallel")]
        #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
        unsafe fn ref_gemm_q4_0_perrow(
            a_quant: &[u8],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            n: usize,
            k: usize,
        ) {
            use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            let nb = k / 32;
            let row_bytes = nb * size_of::<BlockQ4_0>();
            let base = a_quant.as_ptr() as usize;
            out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
                // SAFETY: as in `ref_gemm_q8_0_perrow`.
                unsafe {
                    gemm_q4_0_row(
                        (base as *const u8).wrapping_add(i * row_bytes),
                        b_scales,
                        b_quants,
                        out_row,
                        n,
                        nb,
                    );
                }
            });
        }

        /// A/B of the production row-tiled GEMM against the per-row reference it
        /// replaced, for both dtypes, in one process.
        ///
        /// ```text
        /// cargo test -p cera --release --lib microbench_gemm_rowtile -- --ignored --nocapture
        /// ```
        ///
        /// `--release` is not optional: in a debug build the intrinsics are
        /// unoptimised and the run takes hours rather than ~45 s.
        ///
        /// Read the paired win/loss line, not the means. Both arms run inside a
        /// rayon pool of **fixed size** (the performance-core count) — the pool is
        /// sized, not affinity-pinned; workers are still placed by the OS. Sizing
        /// alone is what collapses the spread, because the variance comes from
        /// rayon's per-process pool *size*, not from placement (see
        /// `microbench_gemm` for the measured numbers).
        ///
        /// Parallel-only: both arms are thread-pool drivers, so there is nothing
        /// to compare in a serial build.
        #[cfg(feature = "parallel")]
        #[test]
        #[ignore]
        fn microbench_gemm_rowtile() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            use std::time::Instant;
            let (m, n, k) = (2048usize, 512usize, 2048usize);
            let nb = k / 32;
            let ops = 2.0 * m as f64 * n as f64 * k as f64;
            let iters = 200;
            let rounds = 8;
            // Varied activations, not a repeated constant. Correctness aside (the
            // bit-exact guard is a real test now), uniform data lets the branch
            // predictor and the caches behave in ways real activations do not.
            let mut st = 0x7ea1_c0deu64;
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            // Paired rounds. The arm order ALTERNATES on round parity: a fixed
            // order would let the second arm always inherit the first arm's cache
            // and clock state, and that bias has constant sign — which is exactly
            // what a genuine win also looks like in the rounds-won statistic.
            // Interleaving alone does not fix this; alternating does.
            fn report(tag: &str, rounds: usize, bench: &mut dyn FnMut(bool) -> f64) {
                let (mut wins, mut sref, mut snew) = (0usize, 0.0f64, 0.0f64);
                eprintln!("\n=== {tag} row-tile A/B (TILE_M={TILE_M}, STRIP_N={STRIP_N}) ===");
                for r in 0..rounds {
                    let (g_ref, g_new) = if r % 2 == 0 {
                        let a = bench(false);
                        (a, bench(true))
                    } else {
                        let b = bench(true);
                        (bench(false), b)
                    };
                    if g_new > g_ref {
                        wins += 1;
                    }
                    sref += g_ref;
                    snew += g_new;
                    let first = if r % 2 == 0 { "ref" } else { "new" };
                    eprintln!(
                        "  round {r} ({first} first):  per-row {g_ref:.0}   row-tiled {g_new:.0} GOP/s"
                    );
                }
                eprintln!(
                    "  mean:    per-row {:.0}   row-tiled {:.0} GOP/s   {:+.1}%   tiled wins {wins}/{rounds}",
                    sref / rounds as f64,
                    snew / rounds as f64,
                    (snew - sref) / sref * 100.0
                );
            }

            let run = || {
                {
                    let mut wst = 0x1234_abcdu64;
                    let a = rand_q8_0_rows(m, k, &mut wst);
                    let mut c_ref = vec![0.0f32; m * n];
                    let mut c_new = vec![0.0f32; m * n];
                    let mut bench = |tiled: bool| -> f64 {
                        let t = Instant::now();
                        for _ in 0..iters {
                            if tiled {
                                unsafe {
                                    gemm_q8_0_q8_0(&a, &b_scales, &b_quants, &mut c_new, m, n, k)
                                };
                            } else {
                                unsafe {
                                    ref_gemm_q8_0_perrow(&a, &b_scales, &b_quants, &mut c_ref, n, k)
                                };
                            }
                        }
                        let secs = t.elapsed().as_secs_f64();
                        // Observe the outputs so the timed calls cannot be folded
                        // away as dead stores: the inputs are loop-invariant and
                        // nothing downstream reads the results.
                        std::hint::black_box((&c_ref, &c_new));
                        ops / (secs / iters as f64) / 1e9
                    };
                    report("gemm_q8_0", rounds, &mut bench);
                }

                {
                    let mut wst = 0xfeed_5eedu64;
                    let a = rand_q4_0_rows(m, k, &mut wst);
                    let mut c_ref = vec![0.0f32; m * n];
                    let mut c_new = vec![0.0f32; m * n];
                    let mut bench = |tiled: bool| -> f64 {
                        let t = Instant::now();
                        for _ in 0..iters {
                            if tiled {
                                unsafe {
                                    gemm_q4_0_q8_0(&a, &b_scales, &b_quants, &mut c_new, m, n, k)
                                };
                            } else {
                                unsafe {
                                    ref_gemm_q4_0_perrow(&a, &b_scales, &b_quants, &mut c_ref, n, k)
                                };
                            }
                        }
                        let secs = t.elapsed().as_secs_f64();
                        std::hint::black_box((&c_ref, &c_new));
                        ops / (secs / iters as f64) / 1e9
                    };
                    report("gemm_q4_0", rounds, &mut bench);
                }
            };

            rayon::ThreadPoolBuilder::new()
                .num_threads(crate::backend::cpu_features::performance_core_count())
                .build()
                .expect("build fixed-size rayon pool")
                .install(run);
        }

        #[test]
        fn gemm_q8_0_avx512_matches_gemv_per_column() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            // Same `n > TILE_N && n % TILE_N != 0` invariant as the Q4_0 test.
            let (m, n, k) = (13, 11, 96);
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
            unsafe { gemm_q8_0_q8_0(&a, &b_scales, &b_quants, &mut out, m, n, k) };

            for j in 0..n {
                let mut y = vec![0.0f32; m];
                unsafe {
                    gemv_q8_0_q8_0(
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

        // ── K-quant deferred-hsum A/B ───────────────────────────────────────
        //
        // The production K-quant GEMMs above defer the dpbusd reduction into a
        // per-column float accumulator (one hsum per column), the way the
        // Q4_0/Q8_0 kernels do. These references reproduce the *previous*
        // structure — an int hsum per (sub-block, column) — so the two can be
        // A/B'd in one process without a rebuild across machine states.

        #[cfg(feature = "parallel")]
        #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
        unsafe fn ref_gemm_q4_k_hsum(
            a_quant: &[u8],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            n: usize,
            k: usize,
        ) {
            let sb = k / 256;
            let nb32 = k / 32;
            let row_bytes = sb * size_of::<crate::quant::BlockQ4KM>();
            let col_sums = unsafe { q8_0_col_sums(b_quants, n, k) };
            let cs: &[i32] = &col_sums;
            let compute_row = |(i, row_out): (usize, &mut [f32])| unsafe {
                let mask_0f = _mm256_set1_epi8(0x0F);
                let row_start = i * row_bytes;
                let mut j0 = 0usize;
                while j0 < n {
                    let cols = KQ_COLS.min(n - j0);
                    let mut acc = [0.0f32; KQ_COLS];
                    for bi in 0..sb {
                        let blk = &*(a_quant
                            .as_ptr()
                            .add(row_start + bi * size_of::<crate::quant::BlockQ4KM>())
                            as *const crate::quant::BlockQ4KM);
                        let d = half::f16::from_bits(blk.d).to_f32();
                        let dmin = half::f16::from_bits(blk.dmin).to_f32();
                        let (sc, mn) = crate::quant::decode_q4km_scales(&blk.scales);
                        let qs = blk.qs.as_ptr();
                        for g in 0..4 {
                            let qb = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                            let w_lo = _mm256_and_si256(qb, mask_0f);
                            let w_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_0f);
                            for (w, s) in [(w_lo, 2 * g), (w_hi, 2 * g + 1)] {
                                let xb = bi * 8 + s;
                                let dsc = d * sc[s] as f32;
                                let dmn = dmin * mn[s] as f32;
                                for (jj, acc_j) in acc.iter_mut().enumerate().take(cols) {
                                    let j = j0 + jj;
                                    let x =
                                        _mm256_loadu_si256(b_quants.as_ptr().add(j * k + xb * 32)
                                            as *const __m256i);
                                    let dp = hsum256_epi32(dot32u(w, x));
                                    let xs = *b_scales.get_unchecked(j * nb32 + xb);
                                    let sx = *cs.get_unchecked(j * nb32 + xb);
                                    *acc_j += xs * (dsc * dp as f32 - dmn * sx as f32);
                                }
                            }
                        }
                    }
                    row_out[j0..j0 + cols].copy_from_slice(&acc[..cols]);
                    j0 += cols;
                }
            };
            crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
        }

        #[cfg(feature = "parallel")]
        #[target_feature(enable = "avx512f,avx512vl,avx512vnni,avx2,fma")]
        unsafe fn ref_gemm_q6_k_hsum(
            a_quant: &[u8],
            b_scales: &[f32],
            b_quants: &[i8],
            out: &mut [f32],
            n: usize,
            k: usize,
        ) {
            let sb = k / 256;
            let nb32 = k / 32;
            let nh16 = k / 16;
            let row_bytes = sb * size_of::<crate::quant::BlockQ6K>();
            let col_sums16 = unsafe { q8_0_col_sums16(b_quants, n, k) };
            let cs16: &[i32] = &col_sums16;
            let compute_row = |(i, row_out): (usize, &mut [f32])| unsafe {
                let mask_0f = _mm256_set1_epi8(0x0F);
                let mask_03 = _mm256_set1_epi8(0x03);
                let row_start = i * row_bytes;
                let mut j0 = 0usize;
                while j0 < n {
                    let cols = KQ_COLS.min(n - j0);
                    let mut acc = [0.0f32; KQ_COLS];
                    for bi in 0..sb {
                        let blk = &*(a_quant
                            .as_ptr()
                            .add(row_start + bi * size_of::<crate::quant::BlockQ6K>())
                            as *const crate::quant::BlockQ6K);
                        let d = half::f16::from_bits(blk.d).to_f32();
                        let ql = blk.ql.as_ptr();
                        let qh = blk.qh.as_ptr();
                        for nh in 0..2usize {
                            let qhb = _mm256_loadu_si256(qh.add(nh * 32) as *const __m256i);
                            for g in 0..4usize {
                                let qlb = _mm256_loadu_si256(
                                    ql.add(nh * 64 + (g & 1) * 32) as *const __m256i
                                );
                                let l4 = if g < 2 {
                                    _mm256_and_si256(qlb, mask_0f)
                                } else {
                                    _mm256_and_si256(_mm256_srli_epi16(qlb, 4), mask_0f)
                                };
                                let h2 = _mm256_and_si256(
                                    _mm256_srl_epi16(qhb, _mm_cvtsi32_si128(2 * g as i32)),
                                    mask_03,
                                );
                                let w = _mm256_or_si256(l4, _mm256_slli_epi16(h2, 4));
                                let sc0 = *blk.scales.get_unchecked(nh * 8 + 2 * g) as f32;
                                let sc1 = *blk.scales.get_unchecked(nh * 8 + 2 * g + 1) as f32;
                                let xb = bi * 8 + nh * 4 + g;
                                for (jj, acc_j) in acc.iter_mut().enumerate().take(cols) {
                                    let j = j0 + jj;
                                    let x =
                                        _mm256_loadu_si256(b_quants.as_ptr().add(j * k + xb * 32)
                                            as *const __m256i);
                                    let lanes = dot32u(w, x);
                                    let dp0 = hsum128_epi32(_mm256_castsi256_si128(lanes));
                                    let dp1 = hsum128_epi32(_mm256_extracti128_si256(lanes, 1));
                                    let xs = *b_scales.get_unchecked(j * nb32 + xb);
                                    let sx0 = *cs16.get_unchecked(j * nh16 + xb * 2);
                                    let sx1 = *cs16.get_unchecked(j * nh16 + xb * 2 + 1);
                                    *acc_j += xs
                                        * d
                                        * (sc0 * (dp0 - 32 * sx0) as f32
                                            + sc1 * (dp1 - 32 * sx1) as f32);
                                }
                            }
                        }
                    }
                    row_out[j0..j0 + cols].copy_from_slice(&acc[..cols]);
                    j0 += cols;
                }
            };
            crate::backend::cpu::par_rows_n(out, n, 64, compute_row);
        }

        fn rand_q4_k_rows(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let sb = k / 256;
            let bsz = size_of::<crate::quant::BlockQ4KM>();
            let mut a = vec![0u8; m * sb * bsz];
            for (bi, chunk) in a.chunks_mut(bsz).enumerate() {
                let d = half::f16::from_f32(0.01 + 0.005 * (bi % 7) as f32);
                let dmin = half::f16::from_f32(0.02 + 0.003 * (bi % 5) as f32);
                chunk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                chunk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
                for b in chunk[4..].iter_mut() {
                    *b = (lcg01(st) * 255.0) as u8;
                }
            }
            a
        }

        fn rand_q6_k_rows(m: usize, k: usize, st: &mut u64) -> Vec<u8> {
            let sb = k / 256;
            let bsz = size_of::<crate::quant::BlockQ6K>();
            let mut a = vec![0u8; m * sb * bsz];
            for (bi, chunk) in a.chunks_mut(bsz).enumerate() {
                for b in chunk[..208].iter_mut() {
                    *b = (lcg01(st) * 255.0) as u8;
                }
                let d = half::f16::from_f32(0.008 + 0.004 * (bi % 5) as f32);
                chunk[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            a
        }

        /// In-process A/B of the deferred-hsum K-quant GEMMs against the
        /// per-block-hsum references above, for both Q4_K and Q6_K.
        ///
        /// ```text
        /// cargo test -p cera --release --lib microbench_gemm_kquant -- --ignored --nocapture
        /// ```
        ///
        /// Same discipline as `microbench_gemm_rowtile`: fixed-size rayon pool,
        /// paired rounds with the arm order alternating on round parity, real
        /// pseudo-random activations. Read the wins line, not the means.
        #[cfg(feature = "parallel")]
        #[test]
        #[ignore]
        fn microbench_gemm_kquant() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            use std::time::Instant;
            let (m, n, k) = (2048usize, 512usize, 2048usize);
            let nb = k / 32;
            let ops = 2.0 * m as f64 * n as f64 * k as f64;
            let iters = 200;
            let rounds = 8;
            let mut st = 0x4b1d_c0deu64;
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            fn report(tag: &str, rounds: usize, bench: &mut dyn FnMut(bool) -> f64) {
                let (mut wins, mut sref, mut snew) = (0usize, 0.0f64, 0.0f64);
                eprintln!("\n=== {tag} deferred-hsum A/B ===");
                for r in 0..rounds {
                    let (g_ref, g_new) = if r % 2 == 0 {
                        let a = bench(false);
                        (a, bench(true))
                    } else {
                        let b = bench(true);
                        (bench(false), b)
                    };
                    if g_new > g_ref {
                        wins += 1;
                    }
                    sref += g_ref;
                    snew += g_new;
                    let first = if r % 2 == 0 { "ref" } else { "new" };
                    eprintln!(
                        "  round {r} ({first} first):  per-block-hsum {g_ref:.0}   deferred {g_new:.0} GOP/s"
                    );
                }
                eprintln!(
                    "  mean:    per-block-hsum {:.0}   deferred {:.0} GOP/s   {:+.1}%   deferred wins {wins}/{rounds}",
                    sref / rounds as f64,
                    snew / rounds as f64,
                    (snew - sref) / sref * 100.0
                );
            }

            let run = || {
                {
                    let mut wst = 0x2222_abcdu64;
                    let a = rand_q4_k_rows(m, k, &mut wst);
                    let mut c_ref = vec![0.0f32; m * n];
                    let mut c_new = vec![0.0f32; m * n];
                    // Self-check before trusting the timing: the deferred kernel
                    // and the per-block-hsum reference must agree.
                    unsafe { gemm_q4_k_q8_0(&a, &b_scales, &b_quants, &mut c_new, m, n, k) };
                    unsafe { ref_gemm_q4_k_hsum(&a, &b_scales, &b_quants, &mut c_ref, n, k) };
                    assert_close(&c_new, &c_ref, "q4_k deferred vs per-block-hsum");
                    let mut bench = |deferred: bool| -> f64 {
                        let t = Instant::now();
                        for _ in 0..iters {
                            if deferred {
                                unsafe {
                                    gemm_q4_k_q8_0(&a, &b_scales, &b_quants, &mut c_new, m, n, k)
                                };
                            } else {
                                unsafe {
                                    ref_gemm_q4_k_hsum(&a, &b_scales, &b_quants, &mut c_ref, n, k)
                                };
                            }
                        }
                        let secs = t.elapsed().as_secs_f64();
                        std::hint::black_box((&c_ref, &c_new));
                        ops / (secs / iters as f64) / 1e9
                    };
                    report("gemm_q4_k", rounds, &mut bench);
                }

                {
                    let mut wst = 0x3333_5eedu64;
                    let a = rand_q6_k_rows(m, k, &mut wst);
                    let mut c_ref = vec![0.0f32; m * n];
                    let mut c_new = vec![0.0f32; m * n];
                    // Self-check before trusting the timing, as for Q4_K.
                    unsafe { gemm_q6_k_q8_0(&a, &b_scales, &b_quants, &mut c_new, m, n, k) };
                    unsafe { ref_gemm_q6_k_hsum(&a, &b_scales, &b_quants, &mut c_ref, n, k) };
                    assert_close(&c_new, &c_ref, "q6_k deferred vs per-block-hsum");
                    let mut bench = |deferred: bool| -> f64 {
                        let t = Instant::now();
                        for _ in 0..iters {
                            if deferred {
                                unsafe {
                                    gemm_q6_k_q8_0(&a, &b_scales, &b_quants, &mut c_new, m, n, k)
                                };
                            } else {
                                unsafe {
                                    ref_gemm_q6_k_hsum(&a, &b_scales, &b_quants, &mut c_ref, n, k)
                                };
                            }
                        }
                        let secs = t.elapsed().as_secs_f64();
                        std::hint::black_box((&c_ref, &c_new));
                        ops / (secs / iters as f64) / 1e9
                    };
                    report("gemm_q6_k", rounds, &mut bench);
                }
            };

            rayon::ThreadPoolBuilder::new()
                .num_threads(crate::backend::cpu_features::performance_core_count())
                .build()
                .expect("build fixed-size rayon pool")
                .install(run);
        }

        // ── Repacked Q4_0 GEMM: equivalence + A/B ───────────────────────────
        //
        // The production kernel (`gemm_q4_0_8x8_q8_0`, above in the macro) and
        // its load-time layout builder (`cpu::repack_q4_0_8x8`) are exercised
        // here: the equivalence test pins agreement with an independent
        // dequantize-then-dot reference (f64 accumulation), and
        // `microbench_gemm_q4_0_8x8` A/Bs the repacked kernel against the
        // standard-layout one in one process (no rebuild across machine states).

        /// Repacked Q4_0 GEMM vs a literal dequantize-then-dot reference (same
        /// quantized activations, f64 accumulation) — independent of both
        /// production kernels, so a shared misreading of the `−8`/scale contract
        /// cannot pass it vacuously.
        ///
        /// `n = 13` is one full `TILE_N` tile plus a remainder on BOTH tiers
        /// (VNNI `TILE_N=8`: 8+5; AVX2 `TILE_N=4`: 12+1) — the earlier `n=5`
        /// entered only the remainder loop on VNNI, leaving the main tiled path
        /// untested. `m = 16` is two super-rows; `k = 512` is 16 blocks.
        #[test]
        fn gemm_q4_0_8x8_matches_reference() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, n, k) = (16usize, 13usize, 512usize);
            let nb = k / 32;
            let mut st = 0x9a1e_c0deu64;
            let a = rand_q4_0_rows(m, k, &mut st);

            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            // Independent reference: dequantize the Q4_0 weights to f32, then dot
            // against the dequantized activations in f64.
            let mut wdeq = vec![0.0f32; m * k];
            crate::quant::dequantize_q4_0_matrix(&a, m, k, &mut wdeq);
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for e in 0..k {
                        let xa = b_scales[j * nb + e / 32] * b_quants[j * k + e] as f32;
                        acc += (wdeq[i * k + e] * xa) as f64;
                    }
                    want[i * n + j] = acc as f32;
                }
            }

            let (packed, scales) = crate::backend::cpu::repack_q4_0_8x8(&a, m, k);
            let mut got = vec![0.0f32; m * n];
            unsafe {
                gemm_q4_0_8x8_q8_0(&packed, &scales, &b_scales, &b_quants, &mut got, m, n, k)
            };
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "q4_0_8x8 [{},{}]: {g} vs reference {w}",
                    i / n,
                    i % n,
                );
            }
        }

        /// A/B: repacked 8-row-interleave Q4_0 GEMM vs the production
        /// standard-layout kernel. Same discipline as `microbench_gemm_rowtile`.
        ///
        /// ```text
        /// cargo test -p cera --release --lib microbench_gemm_q4_0_8x8 -- --ignored --nocapture
        /// ```
        #[cfg(feature = "parallel")]
        #[test]
        #[ignore]
        fn microbench_gemm_q4_0_8x8() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            use std::time::Instant;
            let (m, n, k) = (2048usize, 512usize, 2048usize);
            let nb = k / 32;
            let ops = 2.0 * m as f64 * n as f64 * k as f64;
            let iters = 200;
            let rounds = 8;
            let mut st = 0x8b1d_c0deu64;
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut wst = 0x1234_abcdu64;
            let a = rand_q4_0_rows(m, k, &mut wst);
            let (packed, scales) = crate::backend::cpu::repack_q4_0_8x8(&a, m, k);
            let mut c_ref = vec![0.0f32; m * n];
            let mut c_new = vec![0.0f32; m * n];
            unsafe { gemm_q4_0_q8_0(&a, &b_scales, &b_quants, &mut c_ref, m, n, k) };
            unsafe {
                gemm_q4_0_8x8_q8_0(&packed, &scales, &b_scales, &b_quants, &mut c_new, m, n, k)
            };
            assert_close(&c_new, &c_ref, "q4_0_8x8 repacked vs standard-layout");

            let report = |bench: &mut dyn FnMut(bool) -> f64| {
                let (mut wins, mut sref, mut snew) = (0usize, 0.0f64, 0.0f64);
                eprintln!("\n=== gemm_q4_0 repack (8x8) A/B ===");
                for r in 0..rounds {
                    let (g_ref, g_new) = if r % 2 == 0 {
                        let a = bench(false);
                        (a, bench(true))
                    } else {
                        let b = bench(true);
                        (bench(false), b)
                    };
                    if g_new > g_ref {
                        wins += 1;
                    }
                    sref += g_ref;
                    snew += g_new;
                    let first = if r % 2 == 0 { "std" } else { "repack" };
                    eprintln!(
                        "  round {r} ({first} first):  standard {g_ref:.0}   repacked {g_new:.0} GOP/s"
                    );
                }
                eprintln!(
                    "  mean:    standard {:.0}   repacked {:.0} GOP/s   {:+.1}%   repacked wins {wins}/{rounds}",
                    sref / rounds as f64,
                    snew / rounds as f64,
                    (snew - sref) / sref * 100.0
                );
            };

            let run = || {
                let mut bench = |repacked: bool| -> f64 {
                    let t = Instant::now();
                    for _ in 0..iters {
                        if repacked {
                            unsafe {
                                gemm_q4_0_8x8_q8_0(
                                    &packed, &scales, &b_scales, &b_quants, &mut c_new, m, n, k,
                                )
                            };
                        } else {
                            unsafe {
                                gemm_q4_0_q8_0(&a, &b_scales, &b_quants, &mut c_ref, m, n, k)
                            };
                        }
                    }
                    let secs = t.elapsed().as_secs_f64();
                    std::hint::black_box((&c_ref, &c_new));
                    ops / (secs / iters as f64) / 1e9
                };
                report(&mut bench);
            };

            rayon::ThreadPoolBuilder::new()
                .num_threads(crate::backend::cpu_features::performance_core_count())
                .build()
                .expect("build fixed-size rayon pool")
                .install(run);
        }

        // ── Repacked Q4_K GEMM: equivalence + A/B ───────────────────────────
        //
        // The Q4_K twin of the block above: `gemm_q4_k_8x8_q8_0` (in the macro)
        // and its layout builder (`cpu::repack_q4_k_8x8`), pinned against an
        // independent dequantize-then-dot reference and A/B'd against the
        // standard-layout `gemm_q4_k_q8_0`.

        /// Repacked Q4_K GEMM vs a literal dequantize-then-dot reference. Reuses
        /// `dequantize_q4_k_m_block` (the shipped, separately-tested dequant), so
        /// it shares no arithmetic with the int8 kernel — a wrong scale/min or a
        /// mispacked nibble cannot pass it vacuously.
        ///
        /// `n = 13` is a full `TILE_N` tile plus a remainder on both tiers; `m =
        /// 16` is two super-rows; `k = 512` is two super-blocks.
        #[test]
        fn gemm_q4_k_8x8_matches_reference() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            let (m, n, k) = (16usize, 13usize, 512usize);
            let nb = k / 32;
            let sb = k / 256;
            let bsz = size_of::<crate::quant::BlockQ4KM>();
            let mut st = 0x4b1d_c0deu64;
            let a = rand_q4_k_rows(m, k, &mut st);

            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            // Independent reference: dequantize the Q4_K weights block-by-block to
            // f32, then dot against the dequantized activations in f64.
            let mut wdeq = vec![0.0f32; m * k];
            for i in 0..m {
                for bi in 0..sb {
                    let blk = unsafe {
                        &*(a.as_ptr().add((i * sb + bi) * bsz) as *const crate::quant::BlockQ4KM)
                    };
                    let vals = crate::quant::dequantize_q4_k_m_block(blk);
                    wdeq[i * k + bi * 256..i * k + bi * 256 + 256].copy_from_slice(&vals);
                }
            }
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for e in 0..k {
                        let xa = b_scales[j * nb + e / 32] * b_quants[j * k + e] as f32;
                        acc += (wdeq[i * k + e] * xa) as f64;
                    }
                    want[i * n + j] = acc as f32;
                }
            }

            let (packed, dsc, dmn) = crate::backend::cpu::repack_q4_k_8x8(&a, m, k);
            let mut got = vec![0.0f32; m * n];
            unsafe {
                gemm_q4_k_8x8_q8_0(&packed, &dsc, &dmn, &b_scales, &b_quants, &mut got, m, n, k)
            };
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "q4_k_8x8 [{},{}]: {g} vs reference {w}",
                    i / n,
                    i % n,
                );
            }
        }

        /// A/B: repacked 8-row-interleave Q4_K GEMM vs the production
        /// standard-layout kernel.
        ///
        /// ```text
        /// cargo test -p cera --release --lib microbench_gemm_q4_k_8x8 -- --ignored --nocapture
        /// ```
        #[cfg(feature = "parallel")]
        #[test]
        #[ignore]
        fn microbench_gemm_q4_k_8x8() {
            if !require_simd_or_skip("avx512vnni", vnni_kernels_callable()) {
                return;
            }
            use std::time::Instant;
            let (m, n, k) = (2048usize, 512usize, 2048usize);
            let nb = k / 32;
            let ops = 2.0 * m as f64 * n as f64 * k as f64;
            let iters = 200;
            let rounds = 8;
            let mut st = 0x2c1d_c0deu64;
            let mut b_scales = vec![0.0f32; n * nb];
            let mut b_quants = vec![0i8; n * k];
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|_| lcg01(&mut st) * 2.0 - 1.0).collect();
                let (s, q) = ref_quantize(&col);
                b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
                b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
            }

            let mut wst = 0x5678_abcdu64;
            let a = rand_q4_k_rows(m, k, &mut wst);
            let (packed, dsc, dmn) = crate::backend::cpu::repack_q4_k_8x8(&a, m, k);
            let mut c_ref = vec![0.0f32; m * n];
            let mut c_new = vec![0.0f32; m * n];
            unsafe { gemm_q4_k_q8_0(&a, &b_scales, &b_quants, &mut c_ref, m, n, k) };
            unsafe {
                gemm_q4_k_8x8_q8_0(
                    &packed, &dsc, &dmn, &b_scales, &b_quants, &mut c_new, m, n, k,
                )
            };
            assert_close(&c_new, &c_ref, "q4_k_8x8 repacked vs standard-layout");

            let report = |bench: &mut dyn FnMut(bool) -> f64| {
                let (mut wins, mut sref, mut snew) = (0usize, 0.0f64, 0.0f64);
                eprintln!("\n=== gemm_q4_k repack (8x8) A/B ===");
                for r in 0..rounds {
                    let (g_ref, g_new) = if r % 2 == 0 {
                        let a = bench(false);
                        (a, bench(true))
                    } else {
                        let b = bench(true);
                        (bench(false), b)
                    };
                    if g_new > g_ref {
                        wins += 1;
                    }
                    sref += g_ref;
                    snew += g_new;
                    let first = if r % 2 == 0 { "std" } else { "repack" };
                    eprintln!(
                        "  round {r} ({first} first):  standard {g_ref:.0}   repacked {g_new:.0} GOP/s"
                    );
                }
                eprintln!(
                    "  mean:    standard {:.0}   repacked {:.0} GOP/s   {:+.1}%   repacked wins {wins}/{rounds}",
                    sref / rounds as f64,
                    snew / rounds as f64,
                    (snew - sref) / sref * 100.0
                );
            };

            let run = || {
                let mut bench = |repacked: bool| -> f64 {
                    let t = Instant::now();
                    for _ in 0..iters {
                        if repacked {
                            unsafe {
                                gemm_q4_k_8x8_q8_0(
                                    &packed, &dsc, &dmn, &b_scales, &b_quants, &mut c_new, m, n, k,
                                )
                            };
                        } else {
                            unsafe {
                                gemm_q4_k_q8_0(&a, &b_scales, &b_quants, &mut c_ref, m, n, k)
                            };
                        }
                    }
                    let secs = t.elapsed().as_secs_f64();
                    std::hint::black_box((&c_ref, &c_new));
                    ops / (secs / iters as f64) / 1e9
                };
                report(&mut bench);
            };

            rayon::ThreadPoolBuilder::new()
                .num_threads(crate::backend::cpu_features::performance_core_count())
                .build()
                .expect("build fixed-size rayon pool")
                .install(run);
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

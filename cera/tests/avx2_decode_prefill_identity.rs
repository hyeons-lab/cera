//! Decode and batched prefill must run the *same* arithmetic at the AVX2 tier.
//!
//! Why this exists: it is the invariant that actually broke. When the AVX2 int8
//! GEMM landed without a matching GEMV, prefill went int8 while decode stayed on
//! the f32 `vec_dot`, and LFM2.5-230M-Q4_K_M scored cosine 0.999335 against the
//! parity tests' 0.9999 bar. The fix was an AVX2 int8 GEMV for every dtype.
//!
//! "Same arithmetic" is literal for the K-quants — `gemv_q4k_f32` calls
//! `gemm_q4_k_q8_0` with `n = 1`, so for those two dtypes this test pins the
//! *dispatcher wiring* rather than the arithmetic. Q4_0 and Q8_0 are the
//! stronger case: their GEMV goes through `row_dot_*`, which the tiled GEMM
//! never touches, so bit-equality here is a real agreement between two distinct
//! kernels (the same property `avx2_row_tiled_matches_per_row_bit_exact` pins
//! between the strip and per-row GEMM kernels).
//!
//! Why it is a dedicated test binary: the property only means something at a
//! forced tier, and `CERA_CPU_TIER` is read once per process into a `OnceLock`.
//! A test that set it alongside others would race whoever touched
//! `cpu_features()` first. One process, one lever, set before the first read.
//!
//! Why not rely on the parity tests: those need a real GGUF, are `#[ignore]`d,
//! and run only on the advisory CI leg. Deleting the AVX2 arms from
//! `gemv_q4_0_f32`/`gemv_q8_0_f32` — the Q4_0/Q8_0 form of the regression above,
//! which was first observed on the K-quant arms — passed the entire default
//! `cargo test --workspace` before this file existed, including under
//! `CERA_CPU_TIER=avx2 CERA_REQUIRE_SIMD=avx2`.
//!
//! The comparison is on **bit patterns**, not a tolerance. "Same arithmetic" is
//! the claim; anything looser would pass while decode quietly used a different
//! kernel, which is precisely the failure this guards.

#![cfg(all(target_arch = "x86_64", not(feature = "blas")))]

use cera::backend::cpu;
use cera::backend::cpu_features::{CpuTier, cpu_features};
use cera::tensor::DType;

/// Deterministic byte source. A fixed stream keeps a failure reproducible;
/// nothing here depends on the distribution beyond "not all the same".
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

/// Random weight rows with *controlled* block scales.
///
/// Random bytes in an f16 scale field decode to inf or NaN, and a NaN output
/// compares bit-equal to itself — the assertion would hold while proving
/// nothing. Everything else (nibbles, 6-bit scales, `qh`) stays fully random.
fn weights(dtype: DType, m: usize, k: usize, st: &mut u64) -> Vec<u8> {
    let bb = dtype.block_bytes();
    let nb = k / dtype.block_size();
    let mut data: Vec<u8> = (0..m * nb * bb).map(|_| (lcg(st) % 256) as u8).collect();
    for (bi, blk) in data.chunks_mut(bb).enumerate() {
        let d = half::f16::from_f32(0.01 + 0.004 * (bi % 7) as f32);
        match dtype {
            DType::Q4_0 | DType::Q8_0 | DType::Q4KM => {
                blk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            DType::Q6K => {
                let n = blk.len();
                blk[n - 2..].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            _ => unreachable!("dtype without an int8 kernel"),
        }
        if dtype == DType::Q4KM {
            let dmin = half::f16::from_f32(0.02 + 0.003 * (bi % 5) as f32);
            blk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        }
    }
    data
}

#[test]
fn avx2_decode_gemv_is_bit_identical_to_prefill_gemm_at_n1() {
    // SAFETY: no other thread in this binary reads the environment before this
    // test body completes — it holds exactly one test, and this is the first
    // statement in it. Must also precede any `cpu_features()` call, which caches
    // into a `OnceLock` on first read.
    unsafe {
        std::env::set_var("CERA_CPU_TIER", "avx2");
    }

    let tier = cpu_features().tier;
    if tier < CpuTier::Avx2 {
        // A host below the AVX2 tier has no int8 kernel to compare, so there is
        // nothing to assert. Honour the project's skip convention: silent by
        // default, a hard failure where CI says the hardware is known-capable.
        let required = std::env::var("CERA_REQUIRE_SIMD").unwrap_or_default();
        assert!(
            !required.split(',').any(|f| f.trim() == "avx2"),
            "CERA_REQUIRE_SIMD requires `avx2` but the resolved tier is {tier:?}"
        );
        eprintln!("[avx2-identity] SKIP: tier {tier:?} is below Avx2");
        return;
    }
    assert_eq!(
        tier,
        CpuTier::Avx2,
        "CERA_CPU_TIER=avx2 did not pin the tier — the comparison below would \
         run at some other tier and prove nothing about the AVX2 kernels"
    );

    // The downgrade lever must take the AVX-512 quantizer offline too, not just
    // the VNNI kernels. Verified as a surviving mutation: widening
    // `avx512_quantizer_available()`'s tier floor to `Avx2` passed the entire
    // suite, which would mean a `CERA_CPU_TIER=avx2` run on a real AVX-512 box
    // still executes `_mm512_*` — defeating the one lever this binary, the
    // batched-prefill gate test and the unbatchable-warning test all rely on.
    #[cfg(feature = "avx512")]
    assert!(
        !cera::backend::cpu::avx512_quantizer_available(),
        "CERA_CPU_TIER=avx2 left the AVX-512 quantizer enabled — the tier \
         override is no longer a working downgrade lever"
    );

    // k = 512 gives two K-quant super-blocks (so a wrong super-block stride is
    // visible) and 16 Q4_0/Q8_0 blocks. m is deliberately not a multiple of the
    // AVX2 TILE_M of 2, so the GEMM's row-tail path is exercised too.
    let (m, k) = (7usize, 512usize);
    let mut st = 0xa5a5_0f0fu64;
    let x: Vec<f32> = (0..k)
        .map(|_| (lcg(&mut st) % 4000) as f32 / 1000.0 - 2.0)
        .collect();

    // The activation column the GEMM consumes. Decode quantizes internally
    // through the same function, which is half of why the two agree.
    let mut b_scales = vec![0.0f32; k / 32];
    let mut b_quants = vec![0i8; k];
    cpu::quantize_f32_to_q8_0_into(&x, &mut b_scales, &mut b_quants);

    for dtype in [DType::Q4_0, DType::Q8_0, DType::Q4KM, DType::Q6K] {
        let data = weights(dtype, m, k, &mut st);

        let mut decode = vec![0.0f32; m];
        cpu::gemv_dispatch(dtype, &data, &x, &mut decode, m, k, None);

        let mut prefill = vec![0.0f32; m];
        let ran =
            cpu::gemm_preq_dispatch(dtype, &data, &b_scales, &b_quants, &mut prefill, m, 1, k);
        assert!(
            ran,
            "{dtype:?}: the batched GEMM declined at the Avx2 tier, so prefill \
             would fall back to per-token while decode did not — the two halves \
             of this invariant are not even running the same path"
        );

        // Both buffers start as zeros, so bit-equality alone would also hold
        // if *neither* side computed anything. Nothing currently makes that
        // possible for the decode side, which is exactly why it is worth one
        // line to keep it that way.
        assert!(
            decode.iter().any(|v| *v != 0.0),
            "{dtype:?}: decode produced all zeros — the comparison below would \
             pass against an equally empty prefill buffer and prove nothing"
        );

        for (i, (d, p)) in decode.iter().zip(&prefill).enumerate() {
            assert_eq!(
                d.to_bits(),
                p.to_bits(),
                "{dtype:?} row {i}: decode {d:e} vs prefill-at-n=1 {p:e} — the \
                 AVX2 decode GEMV is not the AVX2 prefill GEMM at n=1, which is \
                 exactly the divergence that broke the batched-prefill parity \
                 bar (cosine 0.999335 against a 0.9999 floor)"
            );
        }
    }
}

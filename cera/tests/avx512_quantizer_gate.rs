//! The AVX-512 activation quantizer must not be gated on VNNI.
//!
//! `quantize_f32_to_q8_0_avx512` declares `avx512f,avx512vl,avx2` and uses no
//! `dpbusd`. It used to sit behind the same predicate that selects the VNNI
//! *kernels*, which denied it to every AVX-512-without-VNNI host (Skylake-X
//! class) for no reason. `avx512_quantizer_available()` is the split-out,
//! corrected predicate.
//!
//! Why a dedicated binary: on a VNNI host `tier >= Avx512` and
//! `tier >= Avx512Vnni` are both true, so a re-narrowed predicate is
//! indistinguishable in-process — the mutation was verified to pass the entire
//! suite. Forcing `CERA_CPU_TIER=avx512` is what separates them, and the tier is
//! read once per process into a `OnceLock`. One process, one lever, set before
//! the first read.
//!
//! What this does *not* claim: that the vectorized quantizer is faster. A/B'd on
//! LFM2.5-230M-Q4_K_M at this tier, the two arms are indistinguishable (decode
//! p50 75.7 vs 76.1 tok/s, n=20). This is a correctness-of-predicate guard.

#![cfg(all(target_arch = "x86_64", feature = "avx512", not(feature = "blas")))]

use cera::backend::cpu_features::{CpuTier, cpu_features};

#[test]
fn quantizer_gate_is_open_at_the_avx512_tier() {
    // SAFETY: no other thread in this binary reads the environment before this
    // test body completes — it holds exactly one test, and this is the first
    // statement in it. Must also precede any `cpu_features()` call, which caches
    // into a `OnceLock` on first read.
    unsafe {
        std::env::set_var("CERA_CPU_TIER", "avx512");
    }

    // `avx512vl` is checked separately because `detect()` awards the `Avx512`
    // tier on `avx512f && avx2 && fma` alone, while the quantizer additionally
    // needs VL. On an avx512f-without-VL part (KNL-class) the gate is correctly
    // closed, and asserting it open would report a re-narrowed predicate that
    // is not what happened.
    let tier = cpu_features().tier;
    if tier < CpuTier::Avx512 || !cpu_features().avx512vl {
        // Below AVX-512 there is no vectorized quantizer to gate. Project skip
        // convention: silent by default, hard failure where CI asserts the
        // hardware is known-capable.
        // `CERA_REQUIRE_SIMD` escalates a skip only when the skip means "this
        // host cannot", not when it means "this host correctly declines". An
        // `avx512f`-without-VL part is the latter: the gate is *supposed* to be
        // closed there, so escalating would report a re-narrowed predicate that
        // did not happen.
        let required = std::env::var("CERA_REQUIRE_SIMD").unwrap_or_default();
        assert!(
            !cpu_features().avx512vl || !required.split(',').any(|f| f.trim() == "avx512"),
            "CERA_REQUIRE_SIMD requires `avx512` but this host resolved to tier \
             {tier:?} with avx512vl={}",
            cpu_features().avx512vl
        );
        eprintln!(
            "[quantizer-gate] SKIP: tier {tier:?}, avx512vl={}",
            cpu_features().avx512vl
        );
        return;
    }

    // The lever must actually have moved, or everything below is vacuous: at
    // `Avx512Vnni` the two predicates agree and this test proves nothing.
    assert_eq!(
        tier,
        CpuTier::Avx512,
        "CERA_CPU_TIER=avx512 did not pin the tier — at any other tier the \
         VNNI and non-VNNI spellings of this predicate cannot be told apart"
    );

    assert!(
        cera::backend::cpu::avx512_quantizer_available(),
        "the AVX-512 quantizer gate is closed at the Avx512 tier — it has been \
         re-narrowed to VNNI, and every AVX-512-without-VNNI host is back on the \
         scalar quantizer for an instruction set the kernel never uses"
    );
}

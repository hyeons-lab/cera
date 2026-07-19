//! The batched prefill gate must actually decline on a host without the int8
//! GEMM.
//!
//! Why this is a dedicated test binary: `cpu_features()` caches into a
//! `OnceLock`, so `CERA_CPU_TIER` is read exactly once per process. A test that
//! set it alongside others would race whoever touched the features first and
//! pass or fail depending on test order. One process, one test, set before the
//! first read.
//!
//! What it protects: `forward_prefill_batched` admits a layer only if the dtype
//! is allowlisted *and* `batched_gemm_supports` says a kernel can run here. Drop
//! the second condition and a non-VNNI x86 host still enters the batched path,
//! `gemm_preq` computes nothing, and the callers — which reuse one output buffer
//! across layers — silently consume the previous layer's activations. Wrong
//! numbers, no crash, no warning.

#![cfg(all(target_arch = "x86_64", not(feature = "blas")))]

/// Downgrading the tier below `Avx512Vnni` must take the int8 GEMM offline.
/// If this ever reports `true`, the gate feeding the batched-prefill decision
/// is broken and the corruption path above is reachable again.
#[test]
fn tier_downgrade_disables_int8_gemm() {
    // SAFETY: single-threaded, and this is the first thing in the process to
    // touch the environment — no other thread can observe the mutation. Must
    // precede any `cpu_features()` call, which is why this test binary holds
    // exactly one test.
    unsafe {
        std::env::set_var("CERA_CPU_TIER", "avx512");
    }

    let tier = cera::backend::cpu_features::cpu_features().tier;
    assert!(
        tier < cera::backend::cpu_features::CpuTier::Avx512Vnni,
        "CERA_CPU_TIER=avx512 did not downgrade the tier (got {tier:?}) — the \
         override is the only lever this test has, so a silent no-op here would \
         make the assertion below vacuous"
    );

    assert!(
        !cera::backend::cpu::int8_gemm_available(),
        "int8_gemm_available() is true at tier {tier:?}, below Avx512Vnni — the \
         batched-prefill gate would admit a host with no int8 kernel"
    );
}

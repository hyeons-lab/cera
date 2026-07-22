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
//! the second condition and a host with no int8 kernel still enters the batched
//! path, `gemm_preq` computes nothing, and the callers — which reuse one output
//! buffer across layers — silently consume the previous layer's activations.
//! Wrong numbers, no crash, no warning.
//!
//! The lever moved once already. This test used to downgrade to `avx512`,
//! because the x86 int8 GEMM needed VNNI and `Avx512` was therefore below the
//! bar. Since the AVX2 kernels landed (`dpbusd` emulated with `maddubs`), every
//! tier from `Avx2` up has an int8 GEMM, so `avx512` no longer proves anything
//! and `scalar` is the only remaining x86 tier that takes the path offline. If
//! a future change gives `Scalar` an int8 GEMM too, this test needs a different
//! lever — not deletion.

#![cfg(all(target_arch = "x86_64", not(feature = "blas")))]

use cera::backend::cpu_features::{CpuTier, cpu_features};
use cera::tensor::DType;

/// Downgrading the tier below the lowest one with an int8 kernel must take the
/// GEMM offline. If this ever reports `true`, the gate feeding the
/// batched-prefill decision is broken and the corruption path above is
/// reachable again.
#[test]
fn tier_downgrade_disables_int8_gemm() {
    // SAFETY: single-threaded, and this is the first thing in the process to
    // touch the environment — no other thread can observe the mutation. Must
    // precede any `cpu_features()` call, which is why this test binary holds
    // exactly one test.
    unsafe {
        std::env::set_var("CERA_CPU_TIER", "scalar");
    }

    let tier = cpu_features().tier;
    assert!(
        tier < CpuTier::Avx2,
        "CERA_CPU_TIER=scalar did not downgrade the tier (got {tier:?}) — the \
         override is the only lever this test has, so a silent no-op here would \
         make the assertion below vacuous"
    );

    assert!(
        !cera::backend::cpu::int8_gemm_available(),
        "int8_gemm_available() is true at tier {tier:?}, below Avx2 — the \
         batched-prefill gate would admit a host with no int8 kernel"
    );

    // And the consumer, not just the predicate. Asserting only on
    // `int8_gemm_available()` left the gate itself unpinned: replacing every
    // arm of `batched_gemm_supports` with `true` passed the entire suite,
    // because nothing in `cargo test` drives the function whose doc comment
    // calls the host check "the load-bearing one". The corruption it describes
    // — `gemm_preq` declining, callers ignoring the return, one output buffer
    // reused across layers, so the previous layer's activations survive as this
    // layer's result — is reachable exactly when this admits a host that cannot
    // run the kernel.
    for dtype in [DType::Q4_0, DType::Q8_0, DType::Q4KM, DType::Q6K] {
        assert!(
            !cera::model::transformer::batched_gemm_supports(dtype, 256),
            "batched_gemm_supports({dtype:?}, 256) is true at tier {tier:?}, \
             below Avx2 — a host with no int8 kernel would enter the batched \
             path and consume the previous layer's activations"
        );
    }
}

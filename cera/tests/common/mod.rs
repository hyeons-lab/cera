//! Shared helpers for integration tests.
//!
//! Lives under `tests/common/` (not `tests/helpers/`) to follow Cargo's
//! convention for non-test files inside the integration-test directory
//! — Cargo skips the `common/` subdir when picking up test binaries.
//!
//! [`download::ensure_cached`] serves tests that need a real GGUF but don't
//! want to bake multi-hundred-MB fixtures into the repo. The download helper
//! compiles only when the `remote` feature is active; callers are
//! `#[cfg(feature = "remote")]`'d accordingly.
//!
//! [`metal_context`] is the shared skip-vs-fail gate used by every Metal
//! suite that dispatches a kernel.

#![allow(dead_code)]

#[cfg(feature = "remote")]
pub mod download;

/// Acquire a Metal device, or skip the calling test.
///
/// A host without a Metal device skips silently; with `CERA_REQUIRE_METAL=1` the
/// missing device is a hard failure instead, so the CI leg that targets
/// known-capable hardware proves the kernels actually executed rather than
/// reporting green on an empty test set. Same skip-vs-fail convention as
/// `CERA_REQUIRE_SIMD` (`require_simd_or_skip` in `backend/simd.rs`).
///
/// Defined once here rather than per test binary, for the reason that helper
/// states: a copy per module is how one ends up with a weaker gate than the CI
/// leg targeting it assumes. It cannot come from `simd.rs` itself — an
/// integration test is a separate crate and cannot name a `#[cfg(test)]` item in
/// the library — but `tests/common/` is shared across test binaries.
///
/// Callers: `metal_shaders_parity`, `metal_turboquant_oracle`,
/// `metal_kv_shift_oracle`. `metal_params_layout` deliberately does not use it —
/// it compares struct layouts and needs no device.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub fn metal_context() -> Option<cera::backend::metal::MetalContext> {
    match cera::backend::metal::MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_METAL").as_deref() != Ok("1"),
                "CERA_REQUIRE_METAL=1 but no Metal device is available ({e})"
            );
            eprintln!("skipping: no Metal device ({e})");
            None
        }
    }
}

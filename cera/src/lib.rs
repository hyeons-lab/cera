#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_neon_dotprod, stdarch_aarch64_prefetch, stdarch_neon_i8mm)
)]

/// Crate version, sourced from `Cargo.toml` at compile time. Useful
/// for FFI / wrapper crates that want to surface the core lib version
/// to their consumers (e.g. `cera-wasm::ceraVersion()`) without
/// re-reading the manifest themselves.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Short git SHA of this build, embedded by `build.rs`. Best-effort: `"unknown"`
/// when git was unavailable at build time (a packaged source build). Can be
/// pinned via the `CERA_GIT_SHA` build-env override.
pub const GIT_SHA: &str = env!("CERA_GIT_SHA");

/// Build provenance for telemetry — `"<version>+<git-sha>"`, e.g.
/// `"0.2.5+1a2b3c4d5e6f"`. This is the analog of the llama.cpp build commit that
/// a benchmark harness records alongside results to identify exactly which
/// engine build produced them.
pub fn build_info() -> String {
    format!("{VERSION}+{GIT_SHA}")
}

pub mod audio_engine;
pub mod backend;
#[cfg(feature = "remote")]
pub mod bundle;
pub mod engine;
/// Auto-generated FlatBuffers code for KV cache serialization.
/// Regenerate with: `flatc --rust -o src/generated schema/kv_cache.fbs`
#[allow(warnings)]
mod generated {
    include!("generated/kv_cache_generated.rs");
}
pub mod gguf;
pub mod grammar;
pub mod kv_cache;
pub mod lora;
pub mod manifest;
pub mod model;
pub mod par;
pub mod quant;
pub mod sampler;
pub mod session;
pub mod sysmem;
pub mod tensor;
pub mod time;
pub mod tokenizer;
pub mod tools;
pub mod turboquant;

// Canonical public re-exports for the stateful API. Consumers should
// `use cera::{Session, ModalitySink, ...}` rather than reaching into
// `cera::session::*`.
pub use backend::cpu_features::{CpuFeatures, CpuTier, cpu_features, cpu_tier};
pub use engine::{BackendPreference, CeraEngine, EngineConfig, ModelFiles, ModelMetadata};
pub use session::{
    CeraError, FinishReason, GenerateOpts, GenerateSummary, ModalityCapabilities, ModalitySink,
    Session, SessionConfig,
};
pub use sysmem::{available_memory_bytes, fits_in_available_memory};

#[cfg(test)]
mod build_info_tests {
    use super::*;

    #[test]
    fn build_info_is_version_plus_sha() {
        let info = build_info();
        assert_eq!(info, format!("{VERSION}+{GIT_SHA}"));
        // version prefix, single '+' separator, non-empty sha segment.
        let (ver, sha) = info.split_once('+').expect("build_info has a '+'");
        assert_eq!(ver, VERSION);
        assert!(!sha.is_empty());
    }
}

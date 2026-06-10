//! `uniffi-bindgen` — binding generator CLI for `cera-ffi`.
//!
//! Thin wrapper around `uniffi::uniffi_bindgen_main()` so the standard
//! `uniffi-bindgen generate ...` command is available in-repo without
//! installing `uniffi_bindgen` globally. Invoked from `just bindings`
//! and from CI (`.github/workflows/ci.yml`'s `ffi-bindings-drift` job).
//!
//! Typical use (needs `--features bindgen` to turn on `uniffi/cli`;
//! the binary target's `required-features` enforces this):
//!
//! ```text
//! cargo run -p cera-ffi --bin uniffi-bindgen --features bindgen -- \
//!     generate --library target/debug/libcera_ffi.dylib \
//!     --language kotlin --out-dir cera-ffi/bindings/kotlin
//! ```
//!
//! See `cera-ffi/README.md` for the full workflow.

fn main() {
    uniffi::uniffi_bindgen_main()
}

//! Real-FFI-boundary parity check: rust ↔ swift-via-UniFFI.
//!
//! Drives the same prompt through `cera::CeraEngine` (Rust reference)
//! and the vendored Swift runner under `cera-parity/legs/swift/`,
//! then asserts byte-identical greedy-decoded token streams. Sister
//! to `parity_kotlin.rs` — same diffing logic, different binding
//! generator. A bug specific to UniFFI's Swift output (record /
//! object / scalar marshalling, optional / enum lifting) would
//! surface here as token divergence even when the Kotlin leg agrees.
//!
//! Two env vars gate this test (belt-and-suspenders, mirrors the
//! `CERA_TEST_DOWNLOAD` style):
//!   - `CERA_PARITY_RUN=1` — opt into the ~210 MB model download.
//!   - `CERA_PARITY_SWIFT_RUNNER=<path/to/CeraParitySwift>` — points
//!     at the SPM-built executable produced by
//!     `cd cera-parity/legs/swift && swift build -c release \
//!         -Xlinker -L<workspace>/target/debug`.
//!
//! Optional:
//!   - `CERA_PARITY_LIB_DIR=<path/to/dir/containing/libcera_ffi.dylib>` —
//!     where dyld should look for the cdylib (set as
//!     `DYLD_LIBRARY_PATH` for the spawned runner). Defaults to
//!     `<workspace>/target/debug` (the `cargo build -p cera-ffi`
//!     output dir).
//!
//! Manual invocation (run from the workspace root):
//!   ```sh
//!   WS=$(pwd)
//!   cargo build -p cera-ffi
//!   swift build -c release \
//!     --package-path "$WS/cera-parity/legs/swift" \
//!     -Xlinker -L"$WS/target/debug"
//!   CERA_PARITY_RUN=1 \
//!     CERA_PARITY_SWIFT_RUNNER="$WS/cera-parity/legs/swift/.build/release/CeraParitySwift" \
//!     cargo test -p cera-parity --test parity_swift -- --ignored
//!   ```
//! `--package-path` keeps the swift build out-of-process from the
//! workspace shell so no `cd` / `$(pwd)` / subshell indirection is
//! required. `WS=$(pwd)` captures the workspace root once for use
//! in both the build flag and the env-var path.
//!
//! macOS-only: this test depends on `swift build` + `libcera_ffi.dylib`
//! + `DYLD_LIBRARY_PATH`. On Linux, `swift build` could in principle
//! build the same source against `libcera_ffi.so` + `LD_LIBRARY_PATH`,
//! but the toolchain weight isn't worth the marginal coverage —
//! `parity_kotlin` already exercises the FFI surface on Linux.

use std::path::PathBuf;

#[test]
#[ignore = "downloads ~210 MB + needs CERA_PARITY_SWIFT_RUNNER set"]
fn rust_and_swift_uniffi_produce_identical_tokens() {
    if std::env::var("CERA_PARITY_RUN").is_err() {
        eprintln!("skipping: CERA_PARITY_RUN not set");
        return;
    }
    let runner = match std::env::var("CERA_PARITY_SWIFT_RUNNER") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skipping: CERA_PARITY_SWIFT_RUNNER not set");
            return;
        }
    };
    assert!(
        runner.exists(),
        "CERA_PARITY_SWIFT_RUNNER points at {} which does not exist",
        runner.display()
    );

    // Default lib_dir = <workspace>/target/debug. CARGO_MANIFEST_DIR
    // points at this crate (cera-parity); workspace root is one up.
    let lib_dir = std::env::var("CERA_PARITY_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest
                .parent()
                .expect("CARGO_MANIFEST_DIR has no parent")
                .join("target")
                .join("debug")
        });

    let cache = cera_parity::default_cache_dir().expect("cache dir");
    let args = cera_parity::RunArgs {
        bundle: "LFM2-350M-Extract-GGUF",
        quant: "Q4_0",
        prompt: "The capital of France is",
        max_tokens: 16,
        seed: 0,
        cache_dir: &cache,
    };

    let (rust, rust_ms) = cera_parity::run_rust(&args).expect("rust path");
    let (swift, swift_ms) =
        cera_parity::run_swift_uniffi(&args, &runner, &lib_dir).expect("swift-uniffi path");
    eprintln!("perf: rust={rust_ms:?} swift-uniffi={swift_ms:?}");

    if let Some(idx) = cera_parity::first_divergence(&rust, &swift) {
        let start = idx.saturating_sub(2);
        let end = idx.saturating_add(3);
        let rust_window = start..end.min(rust.len());
        let swift_window = start..end.min(swift.len());
        let rust_dump = format!("{:?}", &rust[rust_window.clone()]);
        let swift_dump = format!("{:?}", &swift[swift_window.clone()]);
        panic!(
            "rust ↔ swift-uniffi diverged at index {idx}\n  rust [{rust_window:?}] = {rust_dump}\n  swift[{swift_window:?}] = {swift_dump}\n  rust.len() = {}, swift.len() = {}",
            rust.len(),
            swift.len(),
        );
    }
    // Both legs must honor the max_tokens cap independently. If only
    // the rust leg were checked, a swift-side bug that ignored the
    // cap would surface as a token divergence at index 16 rather than
    // as a clear "swift didn't honor max_tokens" message.
    assert!(
        rust.len() <= 16,
        "rust honored max_tokens cap (got {} tokens)",
        rust.len()
    );
    assert!(
        swift.len() <= 16,
        "swift-uniffi honored max_tokens cap (got {} tokens)",
        swift.len()
    );
    assert!(
        !rust.is_empty(),
        "expected at least one decoded token from a non-empty prompt"
    );
}

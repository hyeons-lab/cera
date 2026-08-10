// swift-tools-version:5.9
//
// cera-parity Swift-via-UniFFI leg.
//
// Builds an executable (`.build/release/CeraParitySwift` — name
// follows the `executableTarget`, not the package) that loads the
// generated UniFFI Swift bindings + the cera-ffi cdylib at run time,
// drives one harness run per stdin JSON request, and writes the
// resulting `RunOutput` to stdout. The Rust harness in
// `cera-parity/src/lib.rs::run_swift_uniffi` spawns this binary as a
// subprocess (with `DYLD_LIBRARY_PATH` pointed at the cdylib's
// directory) and diffs the tokens against the Rust reference leg.
//
// Pure SPM — no Xcode project, no Swift Package Registry publishing
// here. The mobile-consumer story for these bindings lives elsewhere
// (the existing `apple-xcframework` CI job builds the consumable
// XCFramework for iOS/Mac apps).
//
// Two-target layout:
//   - `CeraFFI` — `.systemLibrary` wrapping the generated C FFI
//     header (`CeraFFI.h`) via `module.modulemap`. The module name MUST
//     match `ffi_module_name` in cera-ffi/uniffi.toml, because the
//     generated `cera_ffi.swift` uses `#if canImport(CeraFFI) ; import
//     CeraFFI` to discover the C symbols. A different module name
//     leaves every `uniffi_cera_ffi_fn_*` symbol unresolved at compile
//     time. Symlinks the header from `cera-ffi/bindings/swift/`.
//   - `CeraParitySwift` — `.executableTarget` consuming the bindings
//     and the generated Swift wrapper (`cera_ffi.swift`, also
//     symlinked from the same dir).
//
// The `-L<dir>` link flag is supplied via `swift build -Xlinker -L<dir>`
// at build time rather than baked into `linkerSettings.unsafeFlags(...)`
// — that keeps `Package.swift` portable across debug/release/in-tree/
// out-of-tree builds. (`LDFLAGS` is not honored consistently by
// `swift build` — `-Xlinker` is the SPM-supported mechanism.) Recipe
// lives in `cera-parity/legs/swift/README.md` and the CI job.

import PackageDescription

let package = Package(
    name: "cera-parity-swift",
    platforms: [
        // Apple Silicon Macs. The cera-ffi XCFramework ships
        // arm64-only; the parity test only needs to validate the
        // FFI surface on a host the runner can execute, and the
        // CI baseline runs on macOS-26 (Tahoe) — pin the deployment
        // target to macOS 14 (Sonoma) which is comfortably below
        // the runner's OS while staying current enough to match
        // the rest of the repo's Apple targets. No iOS deployment
        // target — this binary only ever runs on the build host.
        .macOS(.v14),
    ],
    targets: [
        .systemLibrary(
            name: "CeraFFI",
            path: "Sources/CeraFFI"
        ),
        .executableTarget(
            name: "CeraParitySwift",
            dependencies: ["CeraFFI"],
            path: "Sources/CeraParitySwift",
            linkerSettings: [
                // Resolved at link time against `swift build -Xlinker
                // -L<dir>`. The `-l` flag itself is portable.
                .linkedLibrary("cera_ffi"),
            ]
        ),
    ]
)

// swift-tools-version:5.9
//
// Cera — consumable Swift Package for the cera inference engine (CPU-only).
//
// Consume it from any iOS / macOS app:
//
//     .package(url: "https://github.com/hyeons-lab/cera", from: "0.2.3")
//
// then add the `Cera` product as a target dependency. The package pulls a
// prebuilt `CeraFFI.xcframework` (a `.binaryTarget`) from the matching
// `v<version>` GitHub release, so consumers never compile Rust — they only
// download the arm64 XCFramework and the thin Swift wrapper below.
//
// ── CPU-only ────────────────────────────────────────────────────────────────
// The shipped XCFramework is built WITHOUT the `metal` feature (see
// `just apple-xcframework`). Inference runs on the CPU (Accelerate/NEON) on
// device, Simulator, and native macOS. A Metal-accelerated iOS slice is a
// planned follow-up; it is intentionally out of scope here.
//
// ── Slices ──────────────────────────────────────────────────────────────────
// The XCFramework carries three arm64-only slices — ios-arm64 (device),
// ios-arm64-simulator, macos-arm64. No x86_64 (Apple stopped selling Intel
// Macs in 2023), which is why the deployment targets below are arm64-era OSes.
//
// ── Targets ─────────────────────────────────────────────────────────────────
//   - `CeraFFI` (binaryTarget) — the remote XCFramework. Its `Headers/` carry
//     `module.modulemap` declaring the clang module `cera_ffiFFI`. That module
//     name is load-bearing: the generated wrapper does
//     `#if canImport(cera_ffiFFI) ; import cera_ffiFFI`, so the module the
//     binaryTarget vends MUST be named `cera_ffiFFI` exactly.
//   - `Cera` (Swift target) — holds the UniFFI-generated Swift wrapper
//     (`cera_ffi.swift`). It is a COMMITTED COPY of
//     `cera-ffi/bindings/swift/cera_ffi.swift`; re-sync it after regenerating
//     the bindings with `just spm-sync-binding` (the two files must stay
//     byte-identical or the Swift surface drifts from the Rust FFI). Depends on
//     `CeraFFI` so `import cera_ffiFFI` resolves against the binaryTarget's
//     clang module.
//
// ── Release wiring ──────────────────────────────────────────────────────────
// The `url` + `checksum` below carry the literal placeholders `RELEASE_VERSION`
// / `RELEASE_CHECKSUM`. The `release` job in `.github/workflows/publish.yml`
// rewrites them to the real `v<version>` URL and the XCFramework zip's
// `swift package compute-checksum`, commits the result to `main`, and points
// the `v<version>` tag at THAT commit — so `.package(url:, from:)` resolves the
// tag with a valid checksum. Do NOT hand-edit these two literals.
//
// ── Local validation ────────────────────────────────────────────────────────
// The remote `url` can't resolve until a release exists. To validate locally,
// build the framework and temporarily point the binaryTarget at the local path
// (`just spm-xcframework-zip` builds + zips + checksums it):
//
//     just apple-xcframework
//     # then swap the `.binaryTarget(url:checksum:)` below for:
//     #   .binaryTarget(name: "CeraFFI",
//     #                 path: "target/xcframework-build/CeraFFI.xcframework")
//     swift build      # compiles `cera_ffi.swift` against the local slice
//
// Revert to the url/placeholder form before committing.

import PackageDescription

let package = Package(
    name: "Cera",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(name: "Cera", targets: ["Cera"]),
    ],
    targets: [
        .binaryTarget(
            name: "CeraFFI",
            url: "https://github.com/hyeons-lab/cera/releases/download/vRELEASE_VERSION/CeraFFI.xcframework.zip",
            checksum: "RELEASE_CHECKSUM"
        ),
        .target(
            name: "Cera",
            dependencies: ["CeraFFI"],
            path: "cera-ffi/apple/Sources/Cera"
        ),
    ]
)

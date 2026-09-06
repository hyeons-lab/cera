// swift-tools-version:5.9
//
// Cera — consumable Swift Package for the cera inference engine.
//
// Consume it from any iOS / macOS app:
//
//     .package(url: "https://github.com/hyeons-lab/cera", from: "0.4.0")
//
// then add the `Cera` product as a target dependency. The package pulls a
// prebuilt `CeraFFI.xcframework` (a `.binaryTarget`) from the matching
// `v<version>` GitHub release, so consumers never compile Rust — they only
// download the arm64 XCFramework and the thin Swift wrapper below.
//
// ── Metal GPU ───────────────────────────────────────────────────────────────
// The shipped XCFramework is built WITH the `metal` feature (see
// `just apple-xcframework`). Inference prefers the native Metal backend
// (Auto probes Metal → CPU) on device, Simulator, and native macOS, falling
// back to the CPU (Accelerate/NEON) when Metal is unavailable. The slices are
// Metal-enabled *dynamic* frameworks, so they carry their own load commands for
// `Metal.framework` and `Foundation` and dyld resolves them — no
// `linkerSettings` needed here.
//
// They used to be static archives, which cost two workarounds: the consumer had
// to name those system frameworks explicitly, and anything resolving symbols at
// RUNTIME rather than link time (Dart FFI via `DynamicLibrary.process()`, i.e.
// the whole Flutter plugin) got nothing at all, because the linker pulls in no
// archive members when nothing references them.
//
// ── Slices ──────────────────────────────────────────────────────────────────
// The XCFramework carries three arm64-only slices — ios-arm64 (device),
// ios-arm64-simulator, macos-arm64. No x86_64 (Apple stopped selling Intel
// Macs in 2023), which is why the deployment targets below are arm64-era OSes.
//
// ── Targets ─────────────────────────────────────────────────────────────────
//   - `CeraFFI` (binaryTarget) — the remote XCFramework. Its `Headers/` carry
//     `module.modulemap` declaring the clang module `CeraFFI`. That module
//     name is load-bearing: the generated wrapper does
//     `#if canImport(CeraFFI) ; import CeraFFI`, so the module the
//     binaryTarget vends MUST be named `CeraFFI` exactly.
//   - `Cera` (Swift target) — holds the UniFFI-generated Swift wrapper
//     (`cera_ffi.swift`). It is a COMMITTED COPY of
//     `cera-ffi/bindings/swift/cera_ffi.swift`, because a `.package(url:)`
//     consumer never has the Rust tree to generate from. `just bindings` writes
//     both, and `just bindings-check` diffs both in CI, so the two cannot
//     drift; do not hand-edit this copy. Depends on `CeraFFI` so
//     `import CeraFFI` resolves against the binaryTarget's clang module.
//
// ── Release wiring ──────────────────────────────────────────────────────────
// The `url` + `checksum` below carry the literal placeholders `RELEASE_VERSION`
// / `RELEASE_CHECKSUM`. The `release` job in `.github/workflows/publish.yml`
// rewrites them to the real `v<version>` URL and the XCFramework zip's
// `swift package compute-checksum` in a commit it points the `v<version>` tag
// at — WITHOUT pushing to `main` (the branch ruleset forbids direct pushes), so
// `main` keeps these placeholders while `.package(url:, from:)` resolves the
// TAG, which carries the valid checksum. Do NOT hand-edit these two literals.
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
            url: "https://github.com/hyeons-lab/cera/releases/download/v0.5.4/CeraFFI.xcframework.zip",
            checksum: "75c52f75e048858f3001896007188072f17e4ea90cc043d3c8151468031c41b2"
        ),
        .target(
            name: "Cera",
            dependencies: ["CeraFFI"],
            path: "cera-ffi/apple/Sources/Cera"
            // No `linkerSettings`. The XCFramework vends dynamic
            // frameworks, which record their own dependencies
            // (Metal.framework for the device / command queue / MSL
            // pipeline objects, Foundation for Metal's Objective-C
            // runtime) in their load commands, so dyld resolves them
            // without the consumer restating anything. The static-lib
            // era needed `.linkedFramework` here; re-adding it now would
            // just be dead weight.
        ),
    ]
)

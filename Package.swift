// swift-tools-version:5.9
//
// Cera — consumable Swift Package for the cera inference engine.
//
// Consume it from any iOS / macOS app:
//
//     .package(url: "https://github.com/hyeons-lab/cera", from: "0.3.0")
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
// back to the CPU (Accelerate/NEON) when Metal is unavailable. Because the
// slices are Metal-enabled *static* libs, the `Cera` target links
// `Metal.framework` + `Foundation` explicitly below — a `.binaryTarget`
// static lib does not auto-link the system frameworks its symbols reference.
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
            url: "https://github.com/hyeons-lab/cera/releases/download/vRELEASE_VERSION/CeraFFI.xcframework.zip",
            checksum: "RELEASE_CHECKSUM"
        ),
        .target(
            name: "Cera",
            dependencies: ["CeraFFI"],
            path: "cera-ffi/apple/Sources/Cera",
            // The XCFramework is a Metal-enabled *static* library. A
            // `.binaryTarget` static lib does NOT auto-link the system
            // frameworks its symbols reference, so consumers must link
            // them explicitly or they hit undefined-symbol errors at
            // link time. The Metal backend references Metal.framework
            // (device / command queue / MSL pipeline objects) and
            // Foundation (Metal's Objective-C runtime dependency).
            // Accelerate is NOT listed: the Rust `accelerate-src` dep is
            // wired for the native-macOS BLAS path only and the linker
            // resolves it from the staticlib without a framework flag on
            // the slices we ship — adding it here would be dead weight.
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)

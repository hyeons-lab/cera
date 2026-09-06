// swift-tools-version: 5.9
//
// Swift Package Manager side of the `cera_ffi_flutter` macOS plugin.
//
// Flutter uses this when the consuming app has SPM enabled
// (`flutter config --enable-swift-package-manager`) and falls back to
// ../cera_ffi_flutter.podspec otherwise. Both must stay in step; Flutter is
// migrating to SPM and warns on plugins that ship only a podspec.
//
// This mirrors the repo root `Package.swift`: the same
// `CeraFFI.xcframework.zip` release asset, pulled as a `.binaryTarget`. It is
// deliberately NOT a dependency on the root package — that one also vends the
// friendly `Cera` Swift wrapper, which a Dart FFI consumer never uses. Here we
// want the binary and nothing else.
//
// ── Release wiring ────────────────────────────────────────────────────────────
// `RELEASE_VERSION` / `RELEASE_CHECKSUM` are placeholders rewritten to real
// values by the `release` job in .github/workflows/publish.yml on the
// v<version> tag, exactly as it already does for the root Package.swift. `main`
// keeps the placeholders; a published package carries real values. Do NOT
// hand-edit these two literals.
//
// ── Local override ────────────────────────────────────────────────────────────
// A locally-built XCFramework next to this manifest always wins, mirroring the
// podspec's `prepare_command`. Without it there is no way to test an unreleased
// engine build on macOS through SPM, because a `.binaryTarget(url:checksum:)`
// pointing at placeholders cannot resolve. Populate it with:
//
//     just apple-xcframework
//     cp -R target/xcframework-build/CeraFFI.xcframework \
//           cera_ffi_flutter/macos/cera_ffi_flutter/
//
// It is gitignored and excluded from the published package, so it can only ever
// be something the developer put there deliberately.
import Foundation
import PackageDescription

let localXCFramework = "CeraFFI.xcframework"
let hasLocalXCFramework = FileManager.default.fileExists(
    atPath: URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent(localXCFramework)
        .path
)

let ceraBinaryTarget: Target = hasLocalXCFramework
    ? .binaryTarget(name: "CeraFFI", path: localXCFramework)
    : .binaryTarget(
        name: "CeraFFI",
        url: "https://github.com/hyeons-lab/cera/releases/download/v0.5.4/CeraFFI.xcframework.zip",
        checksum: "75c52f75e048858f3001896007188072f17e4ea90cc043d3c8151468031c41b2"
    )

let package = Package(
    name: "cera_ffi_flutter",
    platforms: [
        .macOS(.v12),
    ],
    products: [
        .library(name: "cera-ffi-flutter", targets: ["cera_ffi_flutter"]),
    ],
    dependencies: [
        // Required by Flutter's SPM integration even for an FFI-only plugin:
        // the tool refuses to wire up a plugin package that does not depend on
        // the Flutter framework package it generates alongside it.
        .package(name: "FlutterFramework", path: "../FlutterFramework"),
    ],
    targets: [
        ceraBinaryTarget,
        // A Swift target that exists only to carry the binary dependency into
        // the app. Flutter requires a plugin's SPM package to expose a target
        // named after the plugin; SPM in turn requires that target to have at
        // least one source file, hence the near-empty Sources/ file.
        //
        // No `linkerSettings`: the XCFramework vends dynamic frameworks that
        // record Metal.framework and Foundation in their own load commands.
        .target(
            name: "cera_ffi_flutter",
            dependencies: [
                "CeraFFI",
                .product(name: "FlutterFramework", package: "FlutterFramework"),
            ]
        ),
    ]
)

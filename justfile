# Default recipe
default: build

# Build all crates
build:
    cargo build --workspace

# Build in release mode
release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run clippy lints
clippy:
    cargo clippy --workspace -- -D warnings

# Check formatting
fmt:
    cargo fmt --check

# Format code
fmt-fix:
    cargo fmt

# Run the CLI with arguments
run *ARGS:
    cargo run --bin cera -- {{ARGS}}

# Run benchmarks
bench *ARGS:
    cargo run --release --bin cera -- bench {{ARGS}}

# Run all CI checks locally (mirrors GitHub Actions)
ci: fmt clippy test

# Print the host's resolved SIMD tier, then run the tier-specific kernel tests.
# Each test self-skips unless the host has the feature it covers, so the useful
# output is host-dependent: on aarch64+dotprod the NEON fallback comparisons run;
# on an AVX-512 box the avx512 tests run; on ARMv8.6 (i8mm) the i8mm test runs.
# Nothing here needs a model file.
verify-simd:
    @echo "── detected CPU backend ──────────────────────────────"
    cargo run -q -p cera-cli -- cpu
    @echo "── cpu_features + tier-gated kernel tests ────────────"
    cargo test -p cera --lib -- cpu_features fallback_tests avx512

# Platform-specific shared-library path for the uniffi-bindgen
# `--library` argument. `os()` is a just built-in.
# - macOS: `libcera_ffi.dylib`
# - Linux / other unix: `libcera_ffi.so`
# - Windows: `cera_ffi.dll` (no `lib` prefix — Rust follows the
#   Windows convention on that target).
CERA_FFI_DYLIB := if os() == "macos" {
    "target/debug/libcera_ffi.dylib"
} else if os() == "windows" {
    "target/debug/cera_ffi.dll"
} else {
    "target/debug/libcera_ffi.so"
}

# Regenerate the vendored Kotlin + Swift bindings in cera-ffi/bindings/.
# Runs the `uniffi-bindgen` binary in this repo against the freshly-built
# debug cdylib. Kotlin output is ktlint-formatted automatically (uniffi
# invokes ktlint on PATH); Swift is formatter-free (no standard Swift
# formatter in the pipeline). Commit the resulting diff when Rust-side
# exports change.
#
# `--features bindgen` on the `cargo run` invocations turns on the
# opt-in `cera-ffi/bindgen` crate feature, which pulls in
# `uniffi/cli` (clap + friends) only for the binary build. Mobile
# consumers of the library / cdylib / staticlib never build with
# this feature, so their binaries stay lean.
#
# Requires `ktlint` on PATH — macOS: `brew install ktlint`; Linux:
# download the standalone binary from ktlint releases or use your
# package manager. CI installs it as part of the ffi-bindings-drift
# job.
bindings:
    cargo build -p cera-ffi
    cargo run -q -p cera-ffi --bin uniffi-bindgen --features bindgen -- generate \
        --library {{CERA_FFI_DYLIB}} \
        --language kotlin \
        --out-dir cera-ffi/bindings/kotlin
    cargo run -q -p cera-ffi --bin uniffi-bindgen --features bindgen -- generate \
        --library {{CERA_FFI_DYLIB}} \
        --language swift \
        --out-dir cera-ffi/bindings/swift

# Build the `cera-ffi` cdylib with the `ffi-buffer` feature — required by the
# Dart bindings. `uniffi-bindgen-dart` calls `uniffi_ffibuffer_*` trampolines
# that UniFFI only emits under `scaffolding-ffi-buffer-fns`. Kotlin/Swift use
# the standard symbols and don't need this, so the feature stays off for them.
dart-libs:
    cargo build -p cera-ffi --features ffi-buffer
    @echo "Built {{CERA_FFI_DYLIB}} (with ffi-buffer trampolines)."
    @echo "Point Dart at it via CERA_FFI_LIB or place it on the loader path."

# Generate + patch the Dart/Flutter bindings into the cera-ffi-flutter package.
# Builds + runs the VENDORED uniffi-bindgen-dart (third_party/) — patched for
# Cera with callback-argument lowering + the foreign-trait vtable-init symbol fix
# that makes streaming work; built from source rather than `cargo install`ing the
# upstream 0.1.3 so those fixes are in effect (to be upstreamed). It runs against
# the ffi-buffer cdylib, then `tool/patch_generated_bindings.dart` (deterministic
# + idempotent) fixes symbol names, native-lib resolution, and the EngineConfig
# record encoder. The patched result analyzes clean and round-trips real
# inference, including async + streaming. See V2.17.
#
# `cargo run --manifest-path` is used over a hardcoded target/ binary path so it
# stays portable (handles the Windows `.exe` suffix automatically).
#
# Requires a Dart SDK >= 3.3.
dart-bindings: dart-libs
    cargo run --release --manifest-path third_party/uniffi-bindgen-dart/Cargo.toml -- \
        generate {{CERA_FFI_DYLIB}} \
        --out-dir cera-ffi-flutter/lib/src/generated
    cd cera-ffi-flutter && dart run tool/patch_generated_bindings.dart

# Verify the committed Dart bindings are up to date with the current FFI
# surface (regenerate + patch in place, fail on diff) and analyze the package.
dart-bindings-check: dart-bindings
    @if [ -n "$(git status --porcelain cera-ffi-flutter/lib/src/generated)" ]; then \
        echo "ERROR: Dart bindings are stale. Run \`just dart-bindings\` and commit the diff."; \
        git --no-pager diff cera-ffi-flutter/lib/src/generated; \
        exit 1; \
    fi
    cd cera-ffi-flutter && dart pub get && dart analyze

# Verify the committed Kotlin + Swift bindings are up to date with the
# current Rust FFI surface. Regenerates in-place and fails if `git diff`
# shows changes — signals that someone touched a `#[uniffi::*]` export
# without running `just bindings`. CI runs this too; see ci.yml.
bindings-check: bindings
    @if [ -n "$(git status --porcelain cera-ffi/bindings)" ]; then \
        echo "ERROR: vendored bindings are stale. Run \`just bindings\` and commit the diff."; \
        git --no-pager diff cera-ffi/bindings; \
        exit 1; \
    fi

# Cross-compile `cera-ffi` as a `.so` for every Android ABI supported
# by the Android NDK: arm64-v8a (modern devices), armeabi-v7a (older),
# x86_64 (emulator on Intel hosts), x86 (emulator on legacy Intel hosts).
# Produces `target/<triple>/release/libcera_ffi.so` per ABI.
#
# Requires `cargo-ndk` v4.x (`cargo install cargo-ndk --version '^4'
# --locked` — pin the major because 4.0 changed the flag shape to
# `--target <abi>`; earlier releases used `--arch` / `--platform`
# and would fail against the recipes below) and the Rust targets:
# `rustup target add aarch64-linux-android armv7-linux-androideabi
# x86_64-linux-android i686-linux-android`. The NDK itself comes from
# Android Studio (ndk/<version>/) or `sdkmanager --install ndk`.
# `ANDROID_NDK_HOME` must point at the NDK root; CI sets it via the
# `nttld/setup-ndk` action.
#
# Release profile for the size drop — debug builds are ~75 MB per .so
# due to embedded debuginfo, release is ~2.5 MB with LTO + strip.
android-all:
    cargo ndk \
        --target arm64-v8a \
        --target armeabi-v7a \
        --target x86_64 \
        --target x86 \
        build -p cera-ffi --release

# Single-ABI variant — useful when iterating on one device architecture
# and you don't need to rebuild all four every cycle. Picks arm64-v8a
# as the default since it's what real Android phones ship with today.
android-arm64:
    cargo ndk --target arm64-v8a build -p cera-ffi --release

# Stage the cera-ffi cdylib for the HOST desktop platform into the
# `cera-ffi-jvm` module's JNA resource layout, for local
# `./gradlew :cera-ffi-jvm:publishToMavenLocal` testing. CI stages all three
# desktop targets (macOS .dylib, Linux .so, Windows .dll) per-runner; see
# the `jvm` leg of `.github/workflows/publish.yml`. JNA resolves `libcera_ffi` from the
# classpath via its platform resource prefix (darwin-aarch64 / linux-x86-64 /
# win32-x86-64).
jvm-libs-host:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -p cera-ffi --release
    case "$(uname -s)-$(uname -m)" in
      Darwin-arm64)  prefix=darwin-aarch64; lib=libcera_ffi.dylib ;;
      Darwin-x86_64) prefix=darwin-x86-64;  lib=libcera_ffi.dylib ;;
      Linux-x86_64)  prefix=linux-x86-64;   lib=libcera_ffi.so ;;
      *) echo "unsupported host $(uname -s)-$(uname -m) for jvm-libs-host" >&2; exit 1 ;;
    esac
    dest="cera-ffi-kotlin/cera-ffi-jvm/src/main/resources/$prefix"
    mkdir -p "$dest"
    cp "target/release/$lib" "$dest/$lib"
    echo "staged $dest/$lib"

# Cross-compile cera-ffi for all four Android ABIs and stage them directly into
# the `cera-ffi-android` module's jniLibs (cargo-ndk's `-o` writes the
# `<abi>/libcera_ffi.so` layout). Requires the same cargo-ndk + NDK setup as
# `android-all`.
android-libs:
    cargo ndk -o cera-ffi-kotlin/cera-ffi-android/src/main/jniLibs \
        --target arm64-v8a --target armeabi-v7a --target x86_64 --target x86 \
        build -p cera-ffi --release

# Cross-compile `cera-ffi` to all three arm64-only Apple-platform
# targets and assemble a `CeraFFI.xcframework` ready for Swift
# Package Manager / Xcode consumption. Three single-arch slices:
# real iPhones (`ios-arm64`), Apple Silicon Mac iOS Simulator
# (`ios-arm64-simulator`), and native Apple Silicon Macs
# (`macos-arm64`). x86_64 is deliberately omitted — Apple stopped
# selling Intel Macs in 2023 and modern consumer apps drop support.
#
# Requires Xcode (for `xcodebuild`) + the rustup targets:
# `rustup target add aarch64-apple-ios aarch64-apple-ios-sim
# aarch64-apple-darwin`. `RUSTFLAGS=""` overrides the workspace's
# `target-cpu=native` for the apple-darwin slice so the shipped
# staticlib is portable across Apple Silicon Macs (otherwise the
# build host's specific microarch leaks into the binary).
#
# The vendored Swift bindings under `cera-ffi/bindings/swift/`
# provide the headers + module map; CI regenerates them via the
# `ffi-bindings-drift` job so they stay locked to the current Rust
# surface.
#
# Output: `target/xcframework-build/CeraFFI.xcframework` (~125 MB,
# 42 MB per slice). CI uploads the same path as a per-run artifact.
apple-xcframework:
    #!/usr/bin/env bash
    set -euo pipefail
    # Metal-enabled slices. The Metal backend is iOS-portable (Shared
    # storage + system_default device); `--features metal` makes it the
    # Auto-preferred GPU backend on all three arm64 slices, with CPU
    # fallback. The `Cera` SwiftPM target links Metal.framework +
    # Foundation (see Package.swift `linkerSettings`) — a static lib
    # doesn't auto-link the system frameworks its symbols reference.
    #
    # Deployment targets pin the slices to Package.swift's
    # `.macOS(.v12)` / `.iOS(.v15)` so a newer host SDK doesn't stamp a
    # higher `minos` into the staticlib (which otherwise warns
    # "built for newer macOS version than being linked" at consumer
    # link time). MACOSX_/IPHONEOS_ are each read only by the matching
    # target, so exporting both is safe.
    export MACOSX_DEPLOYMENT_TARGET=12.0
    export IPHONEOS_DEPLOYMENT_TARGET=15.0
    RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-ios --release --features metal
    RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-ios-sim --release --features metal
    RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-darwin --release --features metal
    OUT=target/xcframework-build
    rm -rf "$OUT"
    mkdir -p "$OUT/headers"
    # Stage the headers + module map next to where xcodebuild will
    # look. UniFFI-generated `cera_ffiFFI.modulemap` is renamed to
    # `module.modulemap` on the way in — Xcode's framework conventions
    # require that exact filename inside a `Headers/` directory.
    cp cera-ffi/bindings/swift/cera_ffiFFI.h "$OUT/headers/"
    cp cera-ffi/bindings/swift/cera_ffiFFI.modulemap "$OUT/headers/module.modulemap"
    xcodebuild -create-xcframework \
        -library target/aarch64-apple-ios/release/libcera_ffi.a -headers "$OUT/headers" \
        -library target/aarch64-apple-ios-sim/release/libcera_ffi.a -headers "$OUT/headers" \
        -library target/aarch64-apple-darwin/release/libcera_ffi.a -headers "$OUT/headers" \
        -output "$OUT/CeraFFI.xcframework"
    echo "Built $OUT/CeraFFI.xcframework"

# Re-sync the root SwiftPM package's copy of the UniFFI Swift wrapper.
#
# The consumable `Cera` SwiftPM target (`cera-ffi/apple/Sources/Cera/`)
# holds a COMMITTED COPY of the generated `cera_ffi.swift` so the
# package resolves standalone (a `.package(url:)` consumer never has
# the Rust tree). The canonical source is the vendored binding under
# `cera-ffi/bindings/swift/`, regenerated by the `ffi-bindings-drift`
# CI job. Run this after that binding changes so the two stay
# byte-identical — otherwise the package's Swift surface drifts from
# the Rust FFI. CI could enforce this with a `diff` check (follow-up).
spm-sync-binding:
    cp cera-ffi/bindings/swift/cera_ffi.swift cera-ffi/apple/Sources/Cera/cera_ffi.swift
    echo "Synced cera-ffi/apple/Sources/Cera/cera_ffi.swift"

# Build + zip + checksum the CeraFFI XCFramework for a SwiftPM binary
# target release. Produces `target/xcframework-build/CeraFFI.xcframework.zip`
# and prints its `swift package compute-checksum` — the two values the
# `release` job in `.github/workflows/publish.yml` bakes into
# `Package.swift`'s remote `.binaryTarget(url:checksum:)`.
#
# Manual/local counterpart to the workflow's `build-spm` job. To
# validate the package end-to-end locally, temporarily point the
# root `Package.swift` binaryTarget at
# `path: "target/xcframework-build/CeraFFI.xcframework"` and run
# `swift build` (see the header comment in `Package.swift`).
#
# Requires the same toolchain as `apple-xcframework` (Xcode +
# aarch64-apple-{ios,ios-sim,darwin} rustup targets).
spm-xcframework-zip: apple-xcframework
    #!/usr/bin/env bash
    set -euo pipefail
    cd target/xcframework-build
    rm -f CeraFFI.xcframework.zip
    zip -r CeraFFI.xcframework.zip CeraFFI.xcframework >/dev/null
    cd - >/dev/null
    CS=$(swift package compute-checksum target/xcframework-build/CeraFFI.xcframework.zip)
    echo "zip:      target/xcframework-build/CeraFFI.xcframework.zip"
    echo "checksum: $CS"

# Single-target iOS smoke test — verifies the device cross-compile
# works without paying for the full apple-xcframework pipeline (3
# cross-compiles + xcodebuild → ~90s+; this single build → ~30s).
# Output `.a` isn't directly usable in an iOS app (consumers need
# the XCFramework or a custom SPM `linkedLibrary` wiring); this
# recipe is mostly a "did the cross-compile break?" fast probe.
# Assumes `aarch64-apple-ios` is rustup-installed.
#
# `RUSTFLAGS=""` mirrors the `apple-xcframework` + `swift-smoke`
# recipes for consistency. Strictly a no-op for iOS targets
# (`.cargo/config.toml` only sets `target-cpu=native` on
# apple-darwin), but the override forestalls an externally-set
# RUSTFLAGS environment variable from contaminating this smoke build.
ios-arm64:
    IPHONEOS_DEPLOYMENT_TARGET=15.0 RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-ios --release --features metal

# End-to-end Swift integration test against the macOS slice. Compiles
# `cera-ffi/tests/swift/main.swift` together with the vendored Swift
# binding, links against the freshly-built `aarch64-apple-darwin`
# staticlib, runs the resulting binary. Exercises function calls,
# enum + record marshaling, and FfiError round-trip end-to-end.
#
# Why macOS-only smoke: the Rust FFI is identical across iOS device,
# iOS Simulator, and native macOS — same Swift binding, same C ABI,
# same staticlib. Validating macOS proves the integration; iOS
# device + Simulator share the same code path so the test covers
# them by proxy.
#
# Requires Xcode (`swiftc`) + `aarch64-apple-darwin` rustup target.
# Builds the staticlib first if it isn't already cached.
swift-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    MACOSX_DEPLOYMENT_TARGET=12.0 RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-darwin --release --features metal
    # Metal-enabled staticlib references Metal.framework symbols the
    # linker must resolve explicitly (`-framework Metal`); Foundation
    # auto-links on Apple platforms but is listed for parity with the
    # SwiftPM `Cera` target's linkerSettings.
    swiftc \
        cera-ffi/tests/swift/main.swift \
        cera-ffi/bindings/swift/cera_ffi.swift \
        -import-objc-header cera-ffi/bindings/swift/cera_ffiFFI.h \
        -L target/aarch64-apple-darwin/release \
        -lcera_ffi \
        -framework Metal \
        -o target/cera-swift-smoke
    target/cera-swift-smoke

# Build the `cera-wasm` npm-shaped package via `wasm-pack`
# (bundler target — see `wasm-web` / `wasm-node` for siblings).
#
# Wraps `cargo build --target wasm32-unknown-unknown` + `wasm-bindgen-cli`
# + `wasm-opt -O3` and writes the output to `cera-wasm/pkg-bundler/`
# (gitignored — the matrix layout uses `pkg-<target>` to keep the
# three target outputs from colliding). The result includes
# `package.json`, `cera_wasm.js`, `cera_wasm.d.ts`,
# `cera_wasm_bg.wasm`, and the README — drop-in for
# `npm install ./cera-wasm/pkg-bundler`.
#
# Target is `bundler` (webpack / Vite / Rollup-friendly ESM). Use
# `just wasm-web` for direct browser ESM (`<script type="module">`)
# or `just wasm-node` for CommonJS Node consumers.
#
# `--scope hyeons-lab` makes the generated `package.json.name`
# `@hyeons-lab/cera-wasm` so a published artifact lands under the
# right npm scope. The publish workflow itself is a follow-up PR;
# this just locks the name.
#
# Requires:
#   - `wasm-pack`            (`cargo install wasm-pack`)
#   - `wasm-opt` on PATH     (macOS: `brew install binaryen`,
#                             linux: `apt-get install -y binaryen`)
#   - `wasm32-unknown-unknown` rustup target
#     (`rustup target add wasm32-unknown-unknown`)
#
# wasm-opt flags are pinned in `cera-wasm/Cargo.toml` under
# `[package.metadata.wasm-pack.profile.release]` so this recipe and the
# CI `cera-wasm-pack` job produce byte-identical output.
wasm:
    wasm-pack build cera-wasm --target bundler --release --scope hyeons-lab --out-dir pkg-bundler
    @echo "--- cera-wasm/pkg-bundler/ ---"
    @ls -lh cera-wasm/pkg-bundler/

# Build the `--target web` variant — direct browser ESM, no bundler
# required. Consumers `import init, { ... } from './cera_wasm.js'`
# and `await init()` once before calling exports. Right shape for
# `<script type="module">` and bundler-less workflows.
wasm-web:
    wasm-pack build cera-wasm --target web --release --scope hyeons-lab --out-dir pkg-web
    @echo "--- cera-wasm/pkg-web/ ---"
    @ls -lh cera-wasm/pkg-web/

# Build the `--target nodejs` variant — CommonJS module that Node
# consumers `require('@hyeons-lab/cera-wasm')` directly without the
# experimental-wasm-modules dance. Right shape for Node CLI tools
# / scripts that prefer CommonJS or are stuck on older Node.
wasm-node:
    wasm-pack build cera-wasm --target nodejs --release --scope hyeons-lab --out-dir pkg-nodejs
    @echo "--- cera-wasm/pkg-nodejs/ ---"
    @ls -lh cera-wasm/pkg-nodejs/

# ── Multi-threaded wasm builds ──────────────────────────────────────────
#
# Threaded variants light up `cera`'s rayon paths (batched prefill
# GEMM, parallel GEMV row sweeps, dequant_rows_to_f32) on the wasm
# target via `wasm-bindgen-rayon`. The generated package surfaces a
# `initThreadPool(numThreads)` JS export that callers `await` once
# before driving inference.
#
# Three things turn this on together — none of them are useful
# without the others:
#   1. `--features parallel` on `cera-wasm` enables `cera/parallel`
#      (rayon) and links `wasm-bindgen-rayon` (the JS thread-pool
#      shim).
#   2. `RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals"`
#      makes rustc emit atomic ops + thread-local storage
#      instructions. bulk-memory and mutable-globals are already
#      enabled by wasm-opt; the rustflags entry forces them on at
#      compile time too because atomics requires both.
#   3. `-Z build-std=panic_abort,std` rebuilds std with atomics on.
#      The precompiled std rustup ships isn't built with atomics,
#      so anything that touches a sync primitive (rayon definitely
#      does) fails to link without this. Requires the `rust-src`
#      rustup component (`rustup component add rust-src --toolchain
#      $(cat rust-toolchain.toml | grep channel | cut -d'"' -f2)`)
#      and a nightly toolchain — both already in
#      `rust-toolchain.toml`.
#
# Browsers also need cross-origin isolation (COOP `same-origin` +
# COEP `require-corp` headers on the host page) for
# `SharedArrayBuffer`. Node has no equivalent gate.
#
# `--target bundler` is intentionally not provided — `wasm-bindgen-rayon`
# doesn't have canonical bundler-side worker glue, so we ship `web` +
# `nodejs` only.
#
# Link-arg breakdown (all required, none optional):
#   --shared-memory          memory definition gets the SHARED flag.
#                            Without it the linker emits non-shared memory
#                            even with `+atomics`, and Web Workers can't
#                            see the same heap.
#   --import-memory          memory comes from JS (`env.memory`) instead
#                            of being defined inside the wasm. Required
#                            because each Web Worker creates its own
#                            wasm instance and they all need to share
#                            the same `WebAssembly.Memory` — the only
#                            way to do that is to import it.
#   --max-memory=<bytes>     shared memory must declare a max. 4 GB
#                            (`4294967296`) is the wasm32 ceiling and
#                            matches what `wasm-bindgen-rayon`'s docs
#                            recommend.
#   --export=__wasm_init_tls + __tls_size + __tls_align + __tls_base
#                            wasm-bindgen-cli's threading transform
#                            looks these up by name in the export
#                            table. LLD generates them when shared
#                            memory is on but doesn't auto-export them
#                            — without these four flags wasm-bindgen
#                            fails with `failed to find __wasm_init_tls`.
WASM_MT_RUSTFLAGS := "-C target-feature=+atomics,+bulk-memory,+mutable-globals" + \
    " -C link-arg=--shared-memory" + \
    " -C link-arg=--import-memory" + \
    " -C link-arg=--max-memory=4294967296" + \
    " -C link-arg=--export=__wasm_init_tls" + \
    " -C link-arg=--export=__tls_size" + \
    " -C link-arg=--export=__tls_align" + \
    " -C link-arg=--export=__tls_base"

# Build the `--target web` threaded variant — `pkg-web-mt/`.
# Browser consumers `await initThreadPool(navigator.hardwareConcurrency)`
# once after `await init()` resolves; subsequent `Session.generate`
# calls run rayon work on the worker pool.
wasm-web-mt:
    RUSTFLAGS="{{WASM_MT_RUSTFLAGS}}" \
    wasm-pack build cera-wasm \
        --target web --release \
        --scope hyeons-lab --out-dir pkg-web-mt \
        -- --features parallel \
        -Z build-std=panic_abort,std
    @echo "--- cera-wasm/pkg-web-mt/ ---"
    @ls -lh cera-wasm/pkg-web-mt/

# Build the `--target nodejs` threaded variant — `pkg-nodejs-mt/`.
# Node consumers `await initThreadPool(os.cpus().length)` once before
# driving inference; the pool is backed by `worker_threads`.
wasm-node-mt:
    RUSTFLAGS="{{WASM_MT_RUSTFLAGS}}" \
    wasm-pack build cera-wasm \
        --target nodejs --release \
        --scope hyeons-lab --out-dir pkg-nodejs-mt \
        -- --features parallel \
        -Z build-std=panic_abort,std
    @echo "--- cera-wasm/pkg-nodejs-mt/ ---"
    @ls -lh cera-wasm/pkg-nodejs-mt/

# ── WebGPU (single-threaded GPU) wasm build + demo ──────────────────────
#
# The `wgpu` feature turns on `cera/gpu` so inference runs on the GPU via
# WebGPU in the browser. Single-threaded only — `wgpu` and `parallel` are
# mutually exclusive (wgpu's Send+Sync impls vanish under the `atomics`
# target-feature; see `cera-wasm/Cargo.toml`). The browser GPU surface is
# `WebGpuSession` (async `create` + `generate`).

# Build the `--target web` WebGPU package straight into the demo page's
# `pkg/` dir, so `cera-wasm/examples/webgpu/index.html` resolves
# `./pkg/cera_wasm.js`. Serve it with `just wasm-demo-wgpu`.
wasm-web-wgpu:
    wasm-pack build cera-wasm --target web --release \
        --out-dir examples/webgpu/pkg -- --features wgpu
    @echo "--- cera-wasm/examples/webgpu/pkg/ ---"
    @ls -lh cera-wasm/examples/webgpu/pkg/

# Build + serve the in-browser WebGPU LFM2 demo on http://localhost:8000
# (WebGPU is allowed on localhost without HTTPS). Open the page, pick a
# real LFM2 GGUF, and watch it generate on the GPU. Ctrl-C to stop.
wasm-demo-wgpu: wasm-web-wgpu
    @echo "Serving WebGPU demo at http://localhost:8000  (Ctrl-C to stop)"
    cd cera-wasm/examples/webgpu && python3 -m http.server 8000

# Run the headless-Chrome WebGPU smoke test (async device init + readback
# round-trip on real browser WebGPU). Requires a WebGPU-capable Chrome and
# a chromedriver whose MAJOR version matches it on PATH — wasm-pack cannot
# auto-fetch chromedriver on Apple Silicon. Chrome flags that enable
# headless WebGPU live in `cera-wasm/webdriver.json`.
wasm-test-wgpu:
    cd cera-wasm && wasm-pack test --headless --chrome --features wgpu

# Clean build artifacts
clean:
    cargo clean

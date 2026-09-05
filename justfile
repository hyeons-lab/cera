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

# Check formatting. The second command exists because rustfmt does not descend
# into `include!`d files, so the build-script helpers under cera/build_support/
# are invisible to `cargo fmt`. CI mirrors both.
fmt:
    cargo fmt --check
    rustfmt --edition 2024 --check cera/build_support/*.rs

# Format code
fmt-fix:
    cargo fmt
    rustfmt --edition 2024 cera/build_support/*.rs

# Recompile the committed Slang SPIR-V fallbacks from their .slang sources.
# build.rs compiles with slangc directly; the checked-in .spv is only the
# fallback for build hosts without slangc (e.g. CI). Run this after editing a
# .slang so the fallback stays in sync. Needs slangc on PATH or ~/.local/slang/bin.
# Use slangc 2026.14.1: the CI drift check byte-compares against that version's
# output, and SPIR-V is only byte-reproducible within one slangc version.
slang:
    #!/usr/bin/env bash
    set -euo pipefail
    SLANGC="${SLANGC:-$(command -v slangc || echo "$HOME/.local/slang/bin/slangc")}"
    "$SLANGC" -v || true  # informational; some slangc builds exit nonzero on -v
    dir=cera/src/backend/shaders/spirv
    for f in "$dir"/*.slang; do
        name=$(basename "$f" .slang)
        echo "==> slangc $name"
        "$SLANGC" "$f" -target spirv -O3 -entry main -stage compute -o "$dir/$name.spv"
    done
    # Multi-target kernels: one source, WGSL *and* MSL. Unlike the SPIR-V
    # kernels above, the entry points are the kernel's own function names rather
    # than `main`, because both backends look each kernel up by name. slangc has
    # no auto-discovery, so pass one -entry per entry point: default to the
    # basename, or read a `// slang-entries: a b c` header for kernels whose
    # entry name differs (gelu) or that expose several (elementwise). build.rs
    # and the CI drift check parse the same header.
    dir=cera/src/backend/shaders/slang
    for f in "$dir"/*.slang; do
        name=$(basename "$f" .slang)
        entries=$(sed -n 's|^[[:space:]]*//[[:space:]]*slang-entries:[[:space:]]*||p' "$f")
        [ -z "$entries" ] && entries="$name"
        entry_args=()
        for e in $entries; do entry_args+=(-entry "$e"); done
        for target in wgsl metal; do
            echo "==> slangc $name -> $target ($entries)"
            "$SLANGC" "$f" -target "$target" -O3 "${entry_args[@]}" -stage compute -o "$dir/$name.$target"
        done
    done

# Run the CLI with arguments
run *ARGS:
    cargo run --bin cera -- {{ARGS}}

# Run benchmarks
bench *ARGS:
    cargo run --release --bin cera -- bench {{ARGS}}

# Profile host CPU prefill/decode hotspots (perf or samply).
# Builds unstripped with frame pointers — the release profile strips, which
# would leave the profile as bare addresses.
profile-cpu MODEL *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    # Env RUSTFLAGS *replaces* `.cargo/config.toml`'s `[target.*] rustflags`; it
    # does not merge with them. Setting it for the frame pointers therefore
    # silently dropped this host's `target-cpu`, so the profiled binary was
    # tuned differently from the one `just release` produces — and a profile of
    # code that never ships is worse than no profile. Re-state it here.
    # Keep in sync with `.cargo/config.toml`.
    case "$(uname -s)-$(uname -m)" in
      Darwin-*)     TARGET_CPU=native ;;
      Linux-x86_64) TARGET_CPU=x86-64-v3 ;;
      *)            TARGET_CPU= ;;   # aarch64-linux: generic baseline, as configured
    esac
    FLAGS='-C force-frame-pointers=yes'
    if [[ -n "$TARGET_CPU" ]]; then
      FLAGS="$FLAGS -C target-cpu=$TARGET_CPU"
    fi
    echo "==> building with RUSTFLAGS=$FLAGS"
    CARGO_PROFILE_RELEASE_STRIP=false RUSTFLAGS="$FLAGS" \
        cargo build --release -p cera-cli
    ./scripts/profile_cpu.sh --model "{{MODEL}}" {{ARGS}}

# Run python unit tests
python-test:
    python3 tests/test_format_review_comment.py

# Run all CI checks locally (mirrors GitHub Actions)
ci: fmt clippy test python-test

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
    cargo run -q -p cera-ffi --bin uniffi-bindgen --features bindgen -- generate \
        --library {{CERA_FFI_DYLIB}} \
        --language python \
        --out-dir cera-ffi/bindings/python
    # The root SwiftPM package's `Cera` target needs the wrapper *inside* the
    # target directory (a `.package(url:)` consumer never has the Rust tree), so
    # it holds a committed copy. Syncing it here rather than in a recipe someone
    # has to remember is the whole point: the copy drifted 800 lines behind the
    # Rust surface precisely because it was manual, and `canImport` fails
    # silently, so the symptom is undeclared symbols pointing nowhere near the
    # cause. `bindings-check` diffs it too.
    cp cera-ffi/bindings/swift/cera_ffi.swift cera-ffi/apple/Sources/Cera/cera_ffi.swift

# Build the `cera-ffi` cdylib with the `ffi-buffer` feature — required by the
# Dart bindings. `uniffi-bindgen-dart` calls `uniffi_ffibuffer_*` trampolines
# that UniFFI only emits under `scaffolding-ffi-buffer-fns`. Kotlin/Swift use
# the standard symbols and don't need this, so the feature stays off for them.
dart-libs:
    cargo build -p cera-ffi --features ffi-buffer
    @echo "Built {{CERA_FFI_DYLIB}} (with ffi-buffer trampolines)."
    @echo "Point Dart at it via CERA_FFI_LIB or place it on the loader path."

# Materialize the patched Dart bindings generator from its upstream pin plus
# `third_party/uniffi-bindgen-dart/patches/`. The fork is not committed; see
# that directory's README for how to add a patch. The upstream clone is cached,
# so this only touches the network on the first run or after a pin change.
vendor-generator:
    ./scripts/vendor-uniffi-bindgen-dart.sh

# Generate + patch the Dart bindings into the cera_ffi package (NOT
# cera_ffi_flutter, which is the thin Flutter plugin wrapping it).
# Builds + runs the vendored uniffi-bindgen-dart, which is upstream 0.1.3 plus
# the patch series in `third_party/uniffi-bindgen-dart/patches/`: eight fixes
# upstream does not have, covering the callback ABI (argument lowering, the
# vtable-init symbol, slot order, RustBuffer by value, cross-thread delivery)
# and web-stub generation, which is what makes this recipe also emit
# `cera_ffi/lib/src/generated/cera_ffi_web.dart`. See that directory's README.
# It runs against the ffi-buffer cdylib, then
# `tool/patch_generated_bindings.dart` (deterministic + idempotent) fixes symbol
# names, native-lib resolution, and the EngineConfig record encoder. The patched
# result analyzes clean and round-trips real inference, including async +
# streaming. See V2.17.
#
# `cargo run --manifest-path` is used over a hardcoded target/ binary path so it
# stays portable (handles the Windows `.exe` suffix automatically).
#
# Requires a Dart SDK >= 3.3.
dart-bindings: vendor-generator dart-libs
    cargo run --release --manifest-path third_party/uniffi-bindgen-dart/build/Cargo.toml -- \
        generate {{CERA_FFI_DYLIB}} \
        --out-dir cera_ffi/lib/src/generated
    cd cera_ffi && dart run tool/patch_generated_bindings.dart

# Verify the committed Dart bindings are up to date with the current FFI
# surface (regenerate + patch in place, fail on diff) and analyze both packages.
dart-bindings-check: dart-bindings
    @if [ -n "$(git status --porcelain cera_ffi/lib/src/generated)" ]; then \
        echo "ERROR: Dart bindings are stale. Run \`just dart-bindings\` and commit the diff."; \
        git --no-pager diff cera_ffi/lib/src/generated; \
        exit 1; \
    fi
    # `dart`, not `flutter`, and that is the point of the split: `cera_ffi`
    # declares no Flutter SDK constraint, so plain Dart can resolve it. If this
    # line ever has to become `flutter pub get`, something has reintroduced a
    # Flutter dependency and plain-Dart consumers are broken.
    cd cera_ffi && dart pub get && dart analyze
    # The plugin does need Flutter: pub refuses to publish a package declaring
    # `flutter.plugin.platforms` without an SDK constraint, and that constraint
    # is exactly what `dart pub get` rejects. See cera_ffi_flutter/pubspec.yaml.
    cd cera_ffi_flutter && flutter pub get && flutter analyze

# Format every hand-written Dart file, the way CI checks it.
#
# Not a bare `dart format .`, for two reasons CI's "Format" step documents at
# length:
#
#   - `lib/src/generated/` is excluded. UniFFI writes it and `dart-bindings`
#     patches it, so formatting it would fight the generator on every regen.
#   - each package is formatted from its own root, because `dart format` takes
#     its style from the package's language version and the nested example
#     declares a different one. Sweeping it up from the parent applies the
#     wrong style to it.
#
# Run `just dart-pub-get` first (or any recipe that resolves the packages) if
# this is a fresh checkout: with no `.dart_tool/` the formatter ignores the
# declared language version entirely and silently uses the newest style.
dart-fmt:
    cd cera_ffi && git ls-files '*.dart' | grep -v '^lib/src/generated/' | xargs dart format
    cd cera_ffi_flutter && git ls-files '*.dart' | grep -v '^example/' | xargs dart format
    cd cera_ffi_flutter/example && git ls-files '*.dart' | xargs dart format

# Resolve all three Dart packages so `dart format` reads their declared language
# version instead of defaulting to the newest style on a fresh checkout. `dart`
# for `cera_ffi` and `flutter` for the two Flutter packages, for the same reason
# the split matters in `dart-bindings-check`.
dart-pub-get:
    cd cera_ffi && dart pub get
    cd cera_ffi_flutter && flutter pub get
    cd cera_ffi_flutter/example && flutter pub get

# Check Dart formatting without rewriting anything. Mirrors CI's gate.
dart-fmt-check:
    cd cera_ffi && git ls-files '*.dart' | grep -v '^lib/src/generated/' | xargs dart format --output=none --set-exit-if-changed
    cd cera_ffi_flutter && git ls-files '*.dart' | grep -v '^example/' | xargs dart format --output=none --set-exit-if-changed
    cd cera_ffi_flutter/example && git ls-files '*.dart' | xargs dart format --output=none --set-exit-if-changed

# Verify the committed Kotlin + Swift bindings are up to date with the
# current Rust FFI surface. Regenerates in-place and fails if `git diff`
# shows changes — signals that someone touched a `#[uniffi::*]` export
# without running `just bindings`. CI runs this too; see ci.yml.
bindings-check: bindings
    @if [ -n "$(git status --porcelain cera-ffi/bindings cera-ffi/apple/Sources/Cera)" ]; then \
        echo "::error::Vendored bindings (cera-ffi/bindings/, cera-ffi/apple/) are out of date."; \
        echo "Run \`just bindings\` locally and commit the resulting diff."; \
        git --no-pager diff cera-ffi/bindings cera-ffi/apple/Sources/Cera; \
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
    cargo ndk --target arm64-v8a build -p cera-ffi --release --features ffi-buffer

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
    # `ffi-buffer` for the same reason as android-libs: one cdylib serves the
    # JVM bindings and the Flutter plugin's desktop targets.
    cargo build -p cera-ffi --release --features ffi-buffer
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
#
# `ffi-buffer` is not optional here even though Kotlin never uses it: the same
# AAR backs the Flutter plugin, whose Dart bindings call `uniffi_ffibuffer_*`.
# See scripts/assert-ffibuffer.sh.
android-libs:
    cargo ndk -o cera-ffi-kotlin/cera-ffi-android/src/main/jniLibs \
        --target arm64-v8a --target armeabi-v7a --target x86_64 --target x86 \
        build -p cera-ffi --release --features ffi-buffer
    scripts/assert-ffibuffer.sh \
        cera-ffi-kotlin/cera-ffi-android/src/main/jniLibs/*/libcera_ffi.so

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
    # fallback.
    #
    # ── Dynamic, not static ────────────────────────────────────────────
    # These slices ship as DYNAMIC frameworks. They used to be static
    # archives (`-library libcera_ffi.a`), which forced two workarounds:
    # consumers had to name Metal.framework + Foundation explicitly
    # (a static lib does not auto-link the frameworks its symbols
    # reference), and any consumer resolving symbols at RUNTIME rather
    # than link time — Dart FFI via `DynamicLibrary.process()`, i.e. the
    # whole Flutter plugin — got nothing at all, because the linker pulls
    # in no archive members when nothing references them. That needed a
    # brittle `-force_load` pointing at a hardcoded slice path.
    #
    # A dynamic framework is loaded whole by dyld, so every symbol is
    # present at runtime and the system frameworks link themselves. Size
    # is a wash: `-force_load` was already pulling in the entire archive.
    #
    # ── The framework must be named `CeraFFI` ──────────────────────
    # The UniFFI-generated Swift wrapper does
    # `#if canImport(CeraFFI) ; import CeraFFI`, and a framework's clang
    # module takes the framework's name. Naming it anything else
    # silently breaks every Swift consumer. The name comes from
    # `ffi_module_name` in cera-ffi/uniffi.toml — change it there, not
    # here, and regenerate (`just bindings`) so the two stay in step.
    #
    # Deployment targets pin the slices to Package.swift's
    # `.macOS(.v12)` / `.iOS(.v15)` so a newer host SDK doesn't stamp a
    # higher `minos` into the binary (which otherwise warns "built for
    # newer macOS version than being linked" at consumer link time).
    # MACOSX_/IPHONEOS_ are each read only by the matching target, so
    # exporting both is safe.
    export MACOSX_DEPLOYMENT_TARGET=12.0
    export IPHONEOS_DEPLOYMENT_TARGET=15.0
    # ── Why `ffi-buffer` is in here ────────────────────────────────────
    # One XCFramework serves two consumers: the Swift package and the
    # Flutter/Dart plugin. Swift calls the standard `uniffi_cera_ffi_*`
    # scaffolding, but the Dart bindings call `uniffi_ffibuffer_*`
    # trampolines, which UniFFI only emits under the `ffi-buffer`
    # feature (`uniffi/scaffolding-ffi-buffer-fns`).
    #
    # Without it the framework links, embeds, signs, and loads
    # perfectly, and then every Dart call dies at `dlsym` with "symbol
    # not found" — nothing catches it before runtime, because the
    # symbols the Swift side needs are all present. The trampolines are
    # thin wrappers over the same scaffolding, so carrying them costs
    # almost nothing and Swift consumers never look at them.
    FEATURES=metal,ffi-buffer
    RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-ios --release --features "$FEATURES"
    RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-ios-sim --release --features "$FEATURES"
    RUSTFLAGS="" cargo build -p cera-ffi --target aarch64-apple-darwin --release --features "$FEATURES"
    OUT=target/xcframework-build
    rm -rf "$OUT"
    mkdir -p "$OUT"

    # Framework name == the UniFFI Swift module name, which the generated
    # wrapper imports (`import CeraFFI`). Set by `ffi_module_name` in
    # cera-ffi/uniffi.toml; a framework's clang module takes the framework's
    # name, so these two MUST agree or every Swift consumer breaks.
    FW=CeraFFI
    HDR=cera-ffi/bindings/swift/CeraFFI.h
    # Version stamped into each slice's Info.plist. Single source of truth is
    # the workspace VERSION file (kept in lockstep by scripts/bump-version.sh).
    FW_VERSION="$(tr -d '[:space:]' < VERSION)"

    # Assemble one .framework per slice from the Rust cdylib.
    #
    # $1 = rust target dir, $2 = staging subdir, $3 = platform
    # ("macos" uses the versioned bundle layout Apple requires there;
    # iOS uses the flat layout, and a versioned bundle is REJECTED on
    # iOS).
    make_framework() {
      local rust_target="$1" stage="$2" platform="$3"
      local src="target/${rust_target}/release/libcera_ffi.dylib"
      local root="${OUT}/${stage}/${FW}.framework"

      if [ ! -f "$src" ]; then
        echo "missing cdylib: $src" >&2
        exit 1
      fi

      local bin_dir hdr_dir mod_dir res_dir
      if [ "$platform" = "macos" ]; then
        bin_dir="$root/Versions/A"
        hdr_dir="$root/Versions/A/Headers"
        mod_dir="$root/Versions/A/Modules"
        res_dir="$root/Versions/A/Resources"
      else
        bin_dir="$root"
        hdr_dir="$root/Headers"
        mod_dir="$root/Modules"
        res_dir="$root"
      fi
      mkdir -p "$bin_dir" "$hdr_dir" "$mod_dir" "$res_dir"

      cp "$src" "$bin_dir/${FW}"
      chmod +w "$bin_dir/${FW}"
      # dyld resolves the framework relative to whatever embeds it; the
      # Rust cdylib's own install_name (an absolute build path) would
      # fail to load anywhere else.
      install_name_tool -id "@rpath/${FW}.framework/${FW}" "$bin_dir/${FW}"

      cp "$HDR" "$hdr_dir/"
      # Inside a framework the module must be declared `framework module`
      # and live at Modules/module.modulemap. The plain `module` form the
      # generator emits is only valid for a headers directory.
      cat > "$mod_dir/module.modulemap" <<MODMAP_EOF
    framework module ${FW} {
        umbrella header "CeraFFI.h"
        export *
    }
    MODMAP_EOF

      local min_key min_ver plist_dir
      if [ "$platform" = "macos" ]; then
        min_key=LSMinimumSystemVersion; min_ver=12.0; plist_dir="$res_dir"
      else
        min_key=MinimumOSVersion; min_ver=15.0; plist_dir="$res_dir"
      fi
      cat > "$plist_dir/Info.plist" <<PLIST_EOF
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      <key>CFBundleDevelopmentRegion</key><string>en</string>
      <key>CFBundleExecutable</key><string>${FW}</string>
      <key>CFBundleIdentifier</key><string>com.hyeons-lab.cera-ffi</string>
      <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
      <key>CFBundleName</key><string>${FW}</string>
      <key>CFBundlePackageType</key><string>FMWK</string>
      <key>CFBundleShortVersionString</key><string>${FW_VERSION}</string>
      <key>CFBundleVersion</key><string>${FW_VERSION}</string>
      <key>${min_key}</key><string>${min_ver}</string>
    </dict>
    </plist>
    PLIST_EOF

      if [ "$platform" = "macos" ]; then
        ln -s A "$root/Versions/Current"
        ln -s Versions/Current/"${FW}" "$root/${FW}"
        ln -s Versions/Current/Headers "$root/Headers"
        ln -s Versions/Current/Modules "$root/Modules"
        ln -s Versions/Current/Resources "$root/Resources"
      fi
    }

    make_framework aarch64-apple-ios     ios-arm64           ios
    make_framework aarch64-apple-ios-sim ios-arm64-simulator ios
    make_framework aarch64-apple-darwin  macos-arm64         macos

    xcodebuild -create-xcframework \
        -framework "$OUT/ios-arm64/${FW}.framework" \
        -framework "$OUT/ios-arm64-simulator/${FW}.framework" \
        -framework "$OUT/macos-arm64/${FW}.framework" \
        -output "$OUT/CeraFFI.xcframework"
    # Guard the `ffi-buffer` feature. Dropping it produces an XCFramework
    # that links, embeds, signs, and loads perfectly and then fails every
    # Dart call at `dlsym`, because only the Dart bindings use the
    # `uniffi_ffibuffer_*` trampolines. Nothing else catches it before
    # runtime on a device, so assert here.
    for slice in ios-arm64 ios-arm64-simulator macos-arm64; do
        count=$(nm -gU "$OUT/$slice/${FW}.framework/${FW}" | grep -c uniffi_ffibuffer || true)
        if [ "$count" -eq 0 ]; then
            echo "ERROR: $slice exports no uniffi_ffibuffer_* trampolines." >&2
            echo "  The Dart/Flutter plugin cannot call into this build." >&2
            echo "  Build cera-ffi with --features ffi-buffer." >&2
            exit 1
        fi
        echo "  $slice: $count ffibuffer trampolines"
    done
    echo "Built $OUT/CeraFFI.xcframework (dynamic ${FW}.framework slices)"

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
        -import-objc-header cera-ffi/bindings/swift/CeraFFI.h \
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
    scripts/assert-wasm-simd.sh cera-wasm/pkg-bundler/cera_wasm_bg.wasm

# Build the `--target web` variant — direct browser ESM, no bundler
# required. Consumers `import init, { ... } from './cera_wasm.js'`
# and `await init()` once before calling exports. Right shape for
# `<script type="module">` and bundler-less workflows.
wasm-web:
    wasm-pack build cera-wasm --target web --release --scope hyeons-lab --out-dir pkg-web
    @echo "--- cera-wasm/pkg-web/ ---"
    @ls -lh cera-wasm/pkg-web/
    scripts/assert-wasm-simd.sh cera-wasm/pkg-web/cera_wasm_bg.wasm

# Build the `--target nodejs` variant — CommonJS module that Node
# consumers `require('@hyeons-lab/cera-wasm')` directly without the
# experimental-wasm-modules dance. Right shape for Node CLI tools
# / scripts that prefer CommonJS or are stuck on older Node.
wasm-node:
    wasm-pack build cera-wasm --target nodejs --release --scope hyeons-lab --out-dir pkg-nodejs
    @echo "--- cera-wasm/pkg-nodejs/ ---"
    @ls -lh cera-wasm/pkg-nodejs/
    scripts/assert-wasm-simd.sh cera-wasm/pkg-nodejs/cera_wasm_bg.wasm

# Run the `simd128` kernel oracle tests under Node.
#
# These cannot run under a plain `cargo test`: the host cannot execute a
# wasm SIMD instruction, so a host-only suite would report green while
# never touching `backend::simd::wasm_simd`. Each test pins a kernel
# against the scalar reference in `crate::quant` — the same code wasm32
# fell through to before these kernels existed, which is what makes
# "matches scalar" the property worth asserting.
#
# `--lib` matters: without it the runner walks every integration test
# binary in `cera/tests/`, none of which have wasm tests, and prints
# "no tests to run!" 40 times.
#
# Requires: `wasm-pack`, node, and the `wasm32-unknown-unknown` target.
wasm-simd-test:
    #!/usr/bin/env bash
    set -euo pipefail
    # Captured rather than run straight through, because `wasm-pack test`
    # exits 0 when its filter matches nothing. The suite carries the only
    # check that the `simd128` dispatch is enabled at all
    # (`simd128_is_actually_enabled`), so a rename that stopped `wasm_simd`
    # from matching would retire that check silently, with a green run.
    out=$(wasm-pack test --node cera --lib -- wasm_simd 2>&1) || {
        printf '%s\n' "$out"
        exit 1
    }
    printf '%s\n' "$out"
    # "no tests to run!" is the one that fires today: wasm-bindgen-cli applies
    # the filter before its own is-empty check, so a filter that matches
    # nothing lands in the same branch as a binary with no wasm tests at all.
    # Checked against a deliberately wrong filter rather than assumed, because
    # the first version of this guard grepped only for "running 0 tests" and
    # let the empty run through. "running 0 tests" is kept as the second
    # spelling in case a future runner reports it that way instead.
    if printf '%s\n' "$out" | grep -qE 'no tests to run|running 0 tests'; then
        echo "wasm-simd-test: the wasm_simd filter matched no tests" >&2
        exit 1
    fi

# Run `cera-wasm`'s own unit tests under Node.
#
# Same reason as `wasm-simd-test`: the crate is
# `cfg(target_arch = "wasm32")`, so `cargo test` builds an empty lib and
# reports green without executing anything in it. These are
# `#[wasm_bindgen_test]`, which needs a real wasm runtime.
#
# `--node`, not `--headless --chrome`: the tests here are the bundle
# store's pure URL/addressing logic, which has no browser dependency.
# The OPFS paths themselves need a browser and live in `tests/`.
wasm-test:
    wasm-pack test --node cera-wasm --lib

# Run the OPFS bundle-store test in headless Chrome.
#
# Separate from `wasm-test` because it cannot run under Node: there is no
# `navigator.storage.getDirectory` there, which is exactly the layer
# under test. The URL it downloads is the harness page itself, served by
# wasm-bindgen-test-runner, so this needs no network and no fixture.
#
# Requires a Chrome and a chromedriver whose MAJOR version matches it on
# PATH (same constraint as `wasm-test-wgpu`).
wasm-test-opfs:
    cd cera-wasm && wasm-pack test --headless --chrome --test opfs_bundle

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
# Every recipe that produces a package ends by running
# `scripts/assert-wasm-simd.sh` on it. (`wasm-demo-wgpu` only serves what
# `wasm-web-wgpu` already asserted, and the two test recipes build nothing
# that ships.) The threaded recipes need it most: a build that loses
# `+simd128` from the list below is silent, since it succeeds and the
# artifact gets smaller.
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
#                          + __heap_base
#                            wasm-bindgen-cli's threading transform
#                            looks these up by name in the export
#                            table. LLD generates them when shared
#                            memory is on but doesn't auto-export them
#                            — without these flags wasm-bindgen fails
#                            with `failed to find __wasm_init_tls`
#                            (and, since the LLD in nightly-2026-07-10
#                            stopped auto-exporting it, `failed to find
#                            __heap_base for injecting thread id`).
# `+simd128` is repeated here on purpose. `.cargo/config.toml` sets it for
# wasm32-unknown-unknown, but `RUSTFLAGS` in the environment REPLACES the
# config-file `rustflags` rather than appending to it — so omitting it here
# would build the threaded variant against the scalar kernels and quietly lose
# the SIMD speedup while everything still worked.
WASM_MT_RUSTFLAGS := "-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128" + \
    " -C link-arg=--shared-memory" + \
    " -C link-arg=--import-memory" + \
    " -C link-arg=--max-memory=4294967296" + \
    " -C link-arg=--export=__wasm_init_tls" + \
    " -C link-arg=--export=__tls_size" + \
    " -C link-arg=--export=__tls_align" + \
    " -C link-arg=--export=__tls_base" + \
    " -C link-arg=--export=__heap_base"

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
    scripts/assert-wasm-simd.sh cera-wasm/pkg-web-mt/cera_wasm_bg.wasm

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
    scripts/assert-wasm-simd.sh cera-wasm/pkg-nodejs-mt/cera_wasm_bg.wasm

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
    scripts/assert-wasm-simd.sh cera-wasm/examples/webgpu/pkg/cera_wasm_bg.wasm

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
# headless WebGPU live in `cera-wasm/webdriver.json`, which the test runner
# picks up from the working directory. Those flags are macOS-specific
# (`--use-angle=metal`); CI runs the same test on Linux and overrides the
# file via `WASM_BINDGEN_TEST_WEBDRIVER_JSON=cera-wasm/webdriver-linux.json`.
# Keep the two in step when changing either.
wasm-test-wgpu:
    cd cera-wasm && wasm-pack test --headless --chrome --features wgpu

# Clean build artifacts
clean:
    cargo clean

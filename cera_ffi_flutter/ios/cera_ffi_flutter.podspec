#
# iOS side of the `cera_ffi_flutter` FFI plugin (CocoaPods).
#
# Builds no Rust. `CeraFFI.xcframework.zip` is already published as a GitHub
# release asset for the Swift Package (see the repo root `Package.swift`), so
# this downloads that exact artifact, verifies its SHA-256, and vendors it.
#
# The slices are *dynamic* `CeraFFI.framework`s built with the `metal` feature.
# Dynamic matters here: Dart resolves symbols at runtime through
# `DynamicLibrary.process()`, and dyld has already loaded the embedded
# framework by then. With the static archive this used to ship, the linker
# pulled in no members at all (nothing references them at link time) and every
# lookup failed.
#
# Flutter prefers Swift Package Manager when the app has it enabled; see
# ios/cera_ffi_flutter/Package.swift. This podspec stays for apps that have not
# migrated.
#
# ── Release wiring ────────────────────────────────────────────────────────────
# `RELEASE_VERSION` / `RELEASE_CHECKSUM` are placeholders, rewritten to the real
# values by the `release` job in .github/workflows/publish.yml on the v<version>
# tag — the same substitution it already does for Package.swift, and the same
# SHA-256 (`swift package compute-checksum` is a plain sha256 of the zip). So
# `main` keeps placeholders while a tagged/published package carries real
# values. Do NOT hand-edit these two literals.
#
Pod::Spec.new do |s|
  s.name             = 'cera_ffi_flutter'
  s.version          = '0.5.2'
  s.summary          = 'On-device LLM inference for Flutter, powered by the Cera engine.'
  s.description      = <<-DESC
Flutter FFI bindings for the Cera inference engine. Runs GGUF models on-device
with a Metal-accelerated backend.
                       DESC
  s.homepage         = 'https://github.com/hyeons-lab/cera'
  s.license          = { :type => 'Apache-2.0 OR MIT', :file => '../LICENSE' }
  s.author           = { 'Hyeons Lab' => 'https://github.com/hyeons-lab' }
  s.source           = { :path => '.' }
  s.dependency 'Flutter'
  s.platform = :ios, '15.0'

  # arm64 only: Apple stopped selling Intel Macs in 2023, so there is no
  # x86_64 simulator slice to match.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
  }
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
  }

  # A dynamic framework records Metal.framework and Foundation in its own load
  # commands, so there is nothing to restate here and no -force_load needed.
  s.vendored_frameworks = 'CeraFFI.xcframework'

  # Fetched at `pod install` rather than committed: the zip is ~91 MB, far past
  # what belongs in a pub.dev archive (100 MB total, paid for by every
  # platform).
  s.prepare_command = <<-CMD
    set -euo pipefail
    VERSION="RELEASE_VERSION"
    EXPECTED_SHA="RELEASE_CHECKSUM"

    # A locally-built XCFramework always wins. `just apple-xcframework` writes
    # one to target/xcframework-build/; copy it next to this podspec to test an
    # unreleased engine build. Mirrors the local-validation workflow documented
    # for the root Package.swift.
    if [ -d "CeraFFI.xcframework" ] && [ ! -f ".cera-xcframework-sha256" ]; then
      echo "cera-ffi: using locally-supplied CeraFFI.xcframework (not verified)"
      exit 0
    fi

    case "${VERSION}" in
      RELEASE_VERSION)
        echo "cera-ffi: this podspec still has release placeholders." >&2
        echo "  Consume a published version of cera_ffi_flutter (pub.dev or a" >&2
        echo "  v<version> git tag), or drop a locally-built" >&2
        echo "  CeraFFI.xcframework next to this podspec (just apple-xcframework)." >&2
        exit 1
        ;;
    esac

    URL="https://github.com/hyeons-lab/cera/releases/download/v${VERSION}/CeraFFI.xcframework.zip"

    if [ -d "CeraFFI.xcframework" ] && [ -f ".cera-xcframework-sha256" ] && \
       [ "$(cat .cera-xcframework-sha256)" = "${EXPECTED_SHA}" ]; then
      echo "cera-ffi: CeraFFI.xcframework already present and verified"
      exit 0
    fi

    echo "cera-ffi: downloading CeraFFI.xcframework ${VERSION}"
    curl -fsSL --retry 3 -o CeraFFI.xcframework.zip "${URL}"

    ACTUAL_SHA="$(shasum -a 256 CeraFFI.xcframework.zip | cut -d' ' -f1)"
    if [ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]; then
      echo "cera-ffi: checksum mismatch for CeraFFI.xcframework.zip" >&2
      echo "  expected ${EXPECTED_SHA}" >&2
      echo "  actual   ${ACTUAL_SHA}" >&2
      rm -f CeraFFI.xcframework.zip
      exit 1
    fi

    rm -rf CeraFFI.xcframework
    unzip -q -o CeraFFI.xcframework.zip
    rm -f CeraFFI.xcframework.zip
    echo "${EXPECTED_SHA}" > .cera-xcframework-sha256
  CMD
end

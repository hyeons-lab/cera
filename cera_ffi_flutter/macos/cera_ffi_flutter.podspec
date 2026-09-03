#
# macOS side of the `cera_ffi_flutter` FFI plugin (CocoaPods).
#
# Same artifact as the iOS side: the published `CeraFFI.xcframework` carries a
# `macos-arm64` slice alongside the two iOS ones, so Flutter desktop on Apple
# Silicon reuses the release asset the Swift Package already consumes.
#
# The slices are *dynamic* `CeraFFI.framework`s. See the iOS podspec for why
# that matters (Dart resolves symbols at runtime; a static archive contributed
# nothing).
#
# arm64 only. The XCFramework has no macos-x86_64 slice, so an Intel Mac has no
# library here. The JVM/JNA path does ship a `darwin-x86-64` cdylib; if Flutter
# on Intel Macs is ever needed, the XCFramework build has to grow that slice.
#
# Flutter prefers Swift Package Manager when the app has it enabled; see
# macos/cera_ffi_flutter/Package.swift. This podspec stays for apps that have
# not migrated.
#
# `RELEASE_VERSION` / `RELEASE_CHECKSUM` are rewritten at release time; see the
# iOS podspec header for the full explanation.
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
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '12.0'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS' => 'x86_64',
  }
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS' => 'x86_64',
  }

  # A dynamic framework records Metal.framework and Foundation in its own load
  # commands, so there is nothing to restate here and no -force_load needed.
  s.vendored_frameworks = 'CeraFFI.xcframework'

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

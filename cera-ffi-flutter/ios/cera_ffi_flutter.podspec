#
# iOS side of the `cera_ffi_flutter` FFI plugin.
#
# Builds no Rust. `CeraFFI.xcframework.zip` is already published as a GitHub
# release asset for the Swift Package (see the repo root `Package.swift`), so
# this downloads that exact artifact, verifies its SHA-256, and vendors it.
#
# The XCFramework carries arm64 slices for device and simulator, built WITH the
# `metal` feature, so inference prefers the native Metal backend and falls back
# to CPU (Accelerate/NEON).
#
Pod::Spec.new do |s|
  s.name             = 'cera_ffi_flutter'
  s.version          = '0.4.0'
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

  # The slices are arm64-only: Apple stopped selling Intel Macs in 2023, so
  # there is no x86_64 simulator slice to match.
  # The XCFramework vends `libcera_ffi.a` (static). Dart resolves symbols at
  # runtime through `DynamicLibrary.process()`, so nothing references them at
  # link time and the linker would otherwise pull in no archive members at all.
  # `-force_load` keeps every symbol in the app binary.
  #
  # The path must be `$(PODS_XCFRAMEWORKS_BUILD_DIR)`, the directory CocoaPods
  # stages the *selected* slice into. Pointing at a slice inside the
  # XCFramework directly would hardcode `ios-arm64` and break simulator builds,
  # and the `.symlinks` layout differs between iOS and macOS anyway.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
  }
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
  }

  # Metal-enabled static lib: a vendored static library does NOT auto-link the
  # system frameworks its symbols reference, so name them explicitly. Mirrors
  # the `linkerSettings` on the `Cera` target in the root Package.swift.
  s.frameworks = 'Metal', 'Foundation'
  s.static_framework = true

  s.vendored_frameworks = 'CeraFFI.xcframework'

  # Fetched at `pod install` time rather than committed: the zip is ~91 MB, far
  # past what belongs in a pub.dev archive (100 MB total, and it would be paid
  # for by every platform).
  s.prepare_command = <<-CMD
    set -euo pipefail
    VERSION="0.4.0"
    EXPECTED_SHA="fe084d10bbe9f4e4996d077cd2427c17c458af637d614010f4d5e1eb112b5ac0"
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

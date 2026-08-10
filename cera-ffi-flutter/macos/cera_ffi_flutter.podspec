#
# macOS side of the `cera_ffi_flutter` FFI plugin.
#
# Same artifact as the iOS side: the published `CeraFFI.xcframework` carries a
# `macos-arm64` slice alongside the two iOS ones, so Flutter desktop on Apple
# Silicon reuses the release asset the Swift Package already consumes.
#
# arm64 only. The XCFramework has no macos-x86_64 slice, so an Intel Mac has no
# library here. (The JVM/JNA path does ship a `darwin-x86-64` cdylib; if Flutter
# on Intel Macs is ever needed, the XCFramework build has to grow that slice.)
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
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '12.0'

  # Static archive; Dart looks symbols up at runtime via
  # `DynamicLibrary.process()`, so nothing references them at link time and
  # without -force_load the linker pulls in no archive members at all.
  #
  # `$(PODS_XCFRAMEWORKS_BUILD_DIR)` is where CocoaPods stages the *selected*
  # slice. Pointing into the XCFramework directly would hardcode the slice name
  # and the `.symlinks` layout, which differs between iOS (`ios/.symlinks`) and
  # macOS (`Flutter/ephemeral/.symlinks`).
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS' => 'x86_64',
  }
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS' => 'x86_64',
    'OTHER_LDFLAGS' => '-force_load "$(PODS_ROOT)/../Flutter/ephemeral/.symlinks/plugins/cera_ffi_flutter/macos/CeraFFI.xcframework/macos-arm64/libcera_ffi.a"',
  }

  s.frameworks = 'Metal', 'Foundation'
  s.static_framework = true

  s.vendored_frameworks = 'CeraFFI.xcframework'

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

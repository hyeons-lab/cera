# Changelog

## 0.4.0

First release to pub.dev. The package existed in-tree before this, but was
marked `publish_to: none` and was not a Flutter plugin.

### Added

- **Flutter FFI plugin support for Android, iOS, macOS, Linux, and Windows.**
  Each platform resolves the prebuilt native library from an already-published
  release artifact, so app developers need neither a Rust toolchain nor the
  Android NDK: the `cera-ffi-android` AAR via Gradle, `CeraFFI.xcframework` via
  a CocoaPods podspec *and* a Swift Package Manager manifest, and the desktop
  cdylibs via CMake with a SHA-256 check.
- Flutter example app (`example/`) running inference on a background isolate,
  plus plain-Dart CLI examples under `example/dart/`.
- Standalone Linux and Windows cdylib release assets.

### Fixed

- **Eleven engine methods threw `UnsupportedError` at runtime.** The Dart
  generator's object-method renderer decoded only records, enums, and maps, so
  every other RustBuffer return became a stub. That took out `decodeTokens`,
  `encodeText`, `encodeTextSpecial`, `applyChatTemplate`,
  `applyChatTemplateWithTools`, `transcribe`, `toolFormat`, and the
  `hiddenStates*` family. With `decodeTokens` among them there was no way to
  turn a `generate` result into text from Dart at all.
- **`fromBundleIdAsync` threw `UnsupportedError`.** The generator drove the
  rust-future poll/complete lifecycle for object methods but not for
  constructors, so async constructors fell through to a stub even though their
  ffi-buffer trampoline was exported.
- **The committed bindings had drifted behind the Rust FFI surface**, which
  makes UniFFI's per-method checksum verification fail at construction: the
  package threw `StateError` before any call. `just dart-bindings-check` now
  runs in CI, along with an assertion that no generated method is a stub.
- Apple frameworks are now **dynamic** rather than static. A static archive
  contributes nothing to a Dart FFI consumer, which resolves symbols at runtime:
  the linker pulled in no members and every lookup failed.

### Notes

- Requires iOS 15.0+ / macOS 12.0+ / Android API 24+. An app left at Flutter's
  default deployment target fails with an SPM error that does not name the fix;
  see the README.
- Flutter Web is not supported: the bindings are `dart:ffi`-based. Importing the
  package in a multi-platform app is safe (the web stub throws a clear
  `UnsupportedError`). For browsers, use `cera-wasm`.

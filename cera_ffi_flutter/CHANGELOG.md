# Changelog

## 0.5.4

### Changed

- **Version Alignment**: Bumped dependency on `cera_ffi` to `0.5.4` and aligned platform manifests across Android, iOS, macOS, Linux, and Windows.

## 0.5.3

### Added

- **Streaming Text Chunks**: Re-exported `generateStreamChunks` for batched token streaming in Flutter UI pipelines.
- **Multimodal Envelopes**: Re-exported `MultimodalEnvelope`, `ImageInput`, and `AudioInput` for multimodal inference.
- **Android Model Download Service**: Android background service integration for downloading model assets.

### Changed

- **Version Alignment**: Bumped dependency on `cera_ffi` to `0.5.3` and aligned platform manifests across Android, iOS, macOS, Linux, and Windows.

## 0.5.2

### Added

- **Web Platform Support**: Registered the `web` platform in `pubspec.yaml` via a lightweight plugin registrant (`CeraFfiFlutterWeb`) so pub.dev and Flutter tooling officially recognise Web platform support for the plugin.

## 0.5.1

### Added

- **WebGPU Dense Transformer Support**: Native WebGPU acceleration for dense transformer architectures (`llama`, `qwen2`, `qwen3`, `granite`) alongside LFM2 models.
- **License Formatting**: Cleaned dual Apache-2.0 / MIT license formatting for full OSI license scoring on pub.dev.

## 0.5.0

### Added

- **Silero VAD v5 integration**: Re-exports native Silero VAD v5 classes (`FfiSileroVad`, `FfiVadSampleRate`, `FfiVadIterator`, `FfiVadConfig`, `sileroVadDefaultConfig`) for audio activity detection in Flutter apps.
- **Modern MVI UI Architecture**: Complete architectural rewrite of the Flutter example application using Model-View-Intent pattern for predictable state transitions and lifecycle management.
- **Voice Interaction Modes**: 4 dedicated interaction modes (`SpeechToText`, `VoiceChat`, `TextToSpeech`, and `TextOnly`) with capability-based mode switching, dynamic audio waveforms, and exclusive audio streaming playback via `AudioPlayerService`.
- **Hugging Face Model Repositories**: Support for downloading and loading models directly from arbitrary Hugging Face repositories and URLs in addition to LeapBundles.
- **WebGPU Depthformer Acceleration**: High-performance compute shaders for Depthformer audio decoder running seamlessly on WebGPU in web builds.

## 0.4.0

First release to pub.dev. The package existed in-tree before this, but was
marked `publish_to: none` and was not a Flutter plugin.

### Changed

- **The Dart API moved to the new `cera_ffi` package**, which this one now
  depends on and re-exports. `import 'package:cera_ffi_flutter/cera_ffi_flutter.dart'`
  is unchanged and still gives you everything. The split exists because pub
  refuses to publish a package declaring `flutter.plugin.platforms` without a
  Flutter SDK constraint, and declaring that constraint makes `dart pub get`
  refuse the package: one package could not be both a plugin and usable from
  plain Dart.

### Added

- **Flutter FFI plugin support for Android, iOS, macOS, Linux, and Windows.**
  Each platform resolves the prebuilt native library from an already-published
  release artifact, so app developers need neither a Rust toolchain nor the
  Android NDK: the `cera-ffi-android` AAR via Gradle, `CeraFFI.xcframework` via
  a CocoaPods podspec *and* a Swift Package Manager manifest, and the desktop
  cdylibs via CMake with a SHA-256 check.
- **Web support**, through the `Cera` API re-exported from `cera_ffi`: WebGPU
  where the browser has it, a wasm CPU build where it does not. Install the
  runtime once with `dart run cera_ffi_flutter:install_web`.
- Flutter example app (`example/`), one code path for every platform including
  the web, plus plain-Dart CLI examples, which ship with `cera_ffi`.
- Standalone Linux and Windows cdylib release assets.

### Fixed

- **Four engine methods segfaulted the process.** `bosToken`, `eosToken`,
  `specialTokenId` and `toolCallStartToken` return `Option<u32>`, which UniFFI
  passes back as a RustBuffer; the generator special-cased optional primitives
  as a JSON-encoded C string and dereferenced the buffer's capacity word as a
  pointer. SIGSEGV, no exception, nothing an analyzer could see. No caller had
  reached one until `Cera` needed BOS framing.

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
- **The prebuilt native libraries carried no `uniffi_ffibuffer_*` trampolines**,
  so every Dart call failed at `dlsym` on all five platforms. They were built
  for the Kotlin and Swift bindings, which call the standard scaffolding; only
  Dart uses the trampolines, and UniFFI emits them solely under the `ffi-buffer`
  feature. Every build now enables it and asserts the result.
- **Vec<u8> was passed and returned without its length prefix.** UniFFI encodes
  it as an `i32` length then the bytes. Passing one panicked; returning one
  silently prepended four junk bytes, so `hiddenStatesFor*` reported 1025 floats
  for a 1024-wide model. `appendImage` was affected too.
- The Android module declared `minSdk 24` and `compileSdk 35` against an AAR
  published at 28 and consumers compiling against 36, so every app using the
  plugin failed to build at the manifest merger.

### Notes

- Requires iOS 15.0+ / macOS 12.0+ / Android API 28+. An app left at Flutter's
  default deployment target fails with an SPM error that does not name the fix;
  see the README.
- Web inference goes through `Cera` and needs its runtime installed once with
  `dart run cera_ffi_flutter:install_web`. The GPU path supports LFM2 and dense
  transformers (`llama`, `qwen2`, `qwen3`, `granite`); other architectures fall
  back to the wasm CPU build. `openPath` is unavailable there (no filesystem),
  `reset` is unavailable on the GPU backend, and `cancel` is best-effort. The
  *generated bindings* remain native-only: their web branch is a stub whose
  every engine call throws `UnsupportedError`.

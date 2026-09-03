# Changelog

## 0.5.1

### Added

- **WebGPU Dense Transformer Support**: Direct GPU execution support for dense transformer architectures (`llama`, `qwen2`, `qwen3`, `granite`) in addition to `lfm2`/`lfm2.5`/`lfm2moe`.
- **License Formatting**: Cleaned dual Apache-2.0 / MIT license formatting for automated OSI recognition by `pana` on pub.dev.

## 0.5.0

### Added

- **Native Silero VAD v5 engine and bindings**: Added `FfiSileroVad`, `FfiVadSampleRate`, `FfiVadIterator`, `FfiVadConfig`, and `sileroVadDefaultConfig` for real-time speech activity detection, streaming audio chunk processing, and speech segment timestamping.
- **Hugging Face model repository support**: Direct loading from Hugging Face model repository specs and URLs with optional on-the-fly streaming quantization.
- **Multimodal audio & vision enhancements**: WebGPU Depthformer acceleration, 4 voice modes (SpeechToText, VoiceChat, TextToSpeech, TextOnly), microphone audio input, silence trimming, and vision encoder ViT optimizations.

## 0.4.0

First release. The Dart bindings previously lived inside the
`cera_ffi_flutter` package; they were split out here so that they can be used
without Flutter at all.

### Added

- **`Cera`, a portable asynchronous API that runs on every target including the
  web.** One surface over two transports: the Rust async runtime natively, a Web
  Worker running `cera-wasm` in a browser. Loading, chat templating, tokenizing
  and streaming generation, with generations serialized against one KV cache.
  The generated bindings stay synchronous and native-only; `Cera` exists because
  a browser cannot offer a synchronous `generate` at any price.
- **Web inference**, on WebGPU where the browser has it and on a wasm CPU build
  where it does not (58 tok/s against 1.4 tok/s on the same machine and model).
  `dart run cera_ffi:install_web` puts the runtime in an app's `web/`; no
  COOP/COEP headers are required.
- `CeraEngine.fromPathAsync` and `fromBytesAsync`, so loading a model no longer
  blocks the calling isolate.
- `dart:ffi` bindings for the Cera inference engine, generated from the compiled
  `cera-ffi` cdylib by a vendored `uniffi-bindgen-dart` and committed under
  `lib/src/generated/`.
- `CeraLibrary`, a platform-aware loader: an explicit path, then `CERA_FFI_LIB`,
  then the platform's normal search path.
- Plain-Dart examples under `example/`: chat, streaming, async, and download
  progress.

### Notes

- **This package resolves under plain `dart pub get`, which is the reason it
  exists.** pub will not publish a package declaring `flutter.plugin.platforms`
  without a Flutter SDK constraint, and declaring that constraint makes
  `dart pub get` refuse the package outright. A Flutter plugin therefore cannot
  also be a plain-Dart package, so the bindings live here and
  `cera_ffi_flutter` depends on them.
- No native library ships with this package. Flutter apps get one from
  `cera_ffi_flutter`; everyone else points `CERA_FFI_LIB` at a `cera-ffi` build.
  That build needs the `ffi-buffer` cargo feature, without which every call
  fails at `dlsym`.
- The bindings are still exported conditionally, and the web branch is still a
  generated stub with the same API and no `dart:ffi`: data types are real,
  engine entry points throw `UnsupportedError`. That is what lets a
  multi-platform app build at all, since an unconditional `dart:ffi` import
  fails the whole build rather than one branch.

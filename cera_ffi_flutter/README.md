# cera_ffi_flutter

Flutter and Dart bindings for the [Cera](https://github.com/hyeons-lab/cera)
inference engine: on-device LLM inference with no network round trip.

This package wraps the **`cera-ffi` UniFFI surface**, the same C ABI that backs
the Kotlin (`cera-ffi-kotlin`) and Swift bindings, and adds a platform-aware
native-library loader. The Dart bindings are generated from the compiled
`cera-ffi` cdylib by a vendored `uniffi-bindgen-dart`, then run through a small
deterministic patch tool.

It is a Flutter **FFI plugin**: the native library is fetched and linked by each
platform's own build system, with no method channels and no Dart-side setup. It
also works as a plain Dart package (server, CLI) by pointing `CERA_FFI_LIB` at a
locally built cdylib.

## Supported platforms

| Platform | Native library ships as | Notes |
|----------|------------------------|-------|
| Android  | `cera-ffi-android` AAR (Maven Central) | arm64-v8a, armeabi-v7a, x86_64 |
| iOS      | `CeraFFI.xcframework` | Metal enabled; device + simulator |
| macOS    | `CeraFFI.xcframework` | Metal enabled; arm64 |
| Linux    | `libcera_ffi.so` | downloaded + checksummed by CMake |
| Windows  | `cera_ffi.dll` | downloaded + checksummed by CMake |
| Web      | not supported | needs `dart:ffi`; see `cera-wasm` |

Apple targets are wired for both **Swift Package Manager** and CocoaPods;
Flutter picks SPM when the project has it enabled and falls back to the podspec
otherwise.

## Quick start

```dart
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

final engine = CeraEngine.fromPath(modelPath, const EngineConfig(
  contextSize: 2048,
  backend: BackendPreference.auto,
  bundleRepo: null,
));

final prompt = engine.applyChatTemplate(
  [ChatMessage(role: 'user', content: 'Why is the sky blue?')], true);

final session = engine.newSession(const SessionConfig(
  maxSeqLen: null,
  kvCompression: KvCompressionNone(),
  nKeep: 0,
  seed: null,
  ubatchSize: 512,
));
session.appendTokens(engine.encodeTextSpecial(prompt, true));

final out = session.generate(const GenerateOpts(
  maxTokens: 256, temperature: 0.7, topP: 0.95, topK: 40, minP: 0.0,
  repetitionPenalty: 1.1, stopTokens: <int>[], ignoreEos: false,
  grammar: null, grammarTriggerTokens: <int>[],
  flushEveryTokens: 0, flushEveryMs: 0,
));

print(engine.decodeTokens(out.tokens));
```

`generate` returns token IDs plus a timing summary; `decodeTokens` turns them
into text. For token-by-token output use `generateStreamingAsync` with a
`ModalitySink` and decode incrementally.

**Run inference off the UI isolate.** `generate` blocks its isolate for the
whole decode. In a Flutter app, drive it from `Isolate.run` (as
`example/lib/main.dart` does) or use `generateStreamingAsync`.

## Examples

- `example/` — a Flutter chat app (the pub.dev example), inference on a
  background isolate.
- `example/dart/` — plain-Dart CLI scripts covering each surface:
  `cera_chat.dart` (template → generate → decode), `cera_generate.dart`,
  `cera_async.dart`, `cera_stream.dart`, `cera_progress.dart`.

```sh
just dart-bindings
cd cera_ffi_flutter
CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
  dart run example/dart/cera_chat.dart /path/to/model.gguf "Why is the sky blue?"
```

Supported architectures match the engine: `lfm2`/`lfm2.5` (incl. vision and
audio), `llama` (incl. classic Mistral), `qwen2`/`qwen3`, `granite`.

> The callback vtable's static `NativeCallable`s keep the isolate alive, so a
> CLI script must `exit()` explicitly (the examples do); a Flutter app stays
> running regardless.

## Native library resolution

For Flutter apps the platform build embeds the library and nothing is needed at
runtime. `CeraLibrary.open()` resolves, in order:

1. an explicit `path` argument;
2. `$CERA_FFI_LIB`;
3. the already-loaded process image (iOS, and macOS when the framework is
   embedded);
4. the platform filename: `libcera_ffi.dylib` / `libcera_ffi.so` /
   `cera_ffi.dll`.

## Layout

```
cera_ffi_flutter/
├── pubspec.yaml              # ffi dep, SDK ^3.3.0, plugin platform block
├── android/                  # Gradle module -> cera-ffi-android AAR
├── ios/, macos/              # podspec + SPM manifest -> CeraFFI.xcframework
├── linux/, windows/          # CMake: download + checksum the cdylib
├── tool/
│   └── patch_generated_bindings.dart  # post-gen fixups (idempotent)
├── example/                  # Flutter app + plain-Dart scripts
└── lib/
    ├── cera_ffi_flutter.dart # public barrel (loader + generated bindings)
    └── src/
        ├── library_loader.dart       # conditional export (io / web stub)
        ├── library_loader_io.dart    # CeraLibrary.open(): dylib resolution
        └── generated/cera_ffi.dart   # generated + patched bindings (committed)
```

## Generating the bindings

The generator is vendored at `third_party/uniffi-bindgen-dart/` (it builds
against `uniffi_bindgen 0.31`, matching this workspace). From the **repo root**:

```sh
just dart-bindings        # builds the cdylib (--features ffi-buffer), generates, patches
```

`just dart-bindings-check` regenerates + patches in place and fails on a diff.
CI runs it, so committed bindings cannot drift from the Rust FFI surface. That
drift is not cosmetic: UniFFI checksums every method at construction, so stale
bindings fail with a `StateError` before any call.

The `cera-ffi` crate must be built with the **`ffi-buffer`** feature
(`just dart-libs` does this): the Dart generator calls `uniffi_ffibuffer_*`
trampolines that UniFFI only emits under `scaffolding-ffi-buffer-fns`.

### Why a patch step?

`tool/patch_generated_bindings.dart` applies deterministic, idempotent fixes the
generator still gets wrong:

- corrects the `rustbuffer_*` / `rust_future_*` symbol names (spurious `uniffi_`
  infix) and the `.ref.pointer` → `.ref.ptr` union field;
- rewrites native-library resolution to delegate to `CeraLibrary.open`;
- synthesizes the `EngineConfig` record encoder (the generator stubs records
  that contain an interface-handle field);
- marks the public `fromBundleIdAsync` wrapper `async`.

## Generator fixes carried here

The vendored generator adds fixes on top of upstream 0.1.3, to be upstreamed:

- **callback support** (six fixes): callback-arg lowering, vtable-init symbol,
  vtable slot order, RustBuffer callback-arg ABI, the per-interface
  `listener`/`isolateLocal` choice, and freeing RustBuffer callback arguments
  after decode plus a null `errorBuf` on the error path (both leak fixes).
- **RustBuffer return decoding for object methods.** The object-method renderer
  handled only records, enums, and maps; every other RustBuffer return became a
  `throw UnsupportedError`. That silently killed `decodeTokens`, `encodeText`,
  `encodeTextSpecial`, `applyChatTemplate`, `transcribe`, `toolFormat`, and the
  `hiddenStates*` family. The top-level-function renderer had it right all
  along; both now share one helper.
- **async constructors.** The rust-future poll/complete path existed for methods
  only, so every async constructor fell through to a stub even though its
  `uniffi_ffibuffer_*` trampoline was exported. `fromBundleIdAsync` works.

## Status

The full engine API works end to end on macOS and in plain Dart: model load,
sessions, sync and async `generate`, sync and async streaming, `transcribe`,
tokenizer access, chat templates, `BundleRepo` with download progress, and
`fromBundleIdAsync`. No method throws `UnsupportedError`; CI asserts that.

Not yet verified on real devices: **Android, iOS, Linux, and Windows builds.**
The Apple manifests resolve `CeraFFI.xcframework` from a tagged release, so
Apple targets need a published release (or a locally built xcframework) before
they resolve.

Web is out of scope: this package needs `dart:ffi`. Importing it in a
multi-platform app is safe (the web stub throws a clear `UnsupportedError`);
for browsers use `cera-wasm`.

## License

Apache-2.0 OR MIT, matching the rest of the workspace.

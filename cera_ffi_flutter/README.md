# cera_ffi_flutter

Flutter and Dart bindings for the [Cera](https://github.com/hyeons-lab/cera)
inference engine: on-device LLM inference with no network round trip.

This is the package Flutter apps depend on. It is an **FFI plugin**: the native
library is fetched and linked by each platform's own build system, with no
method channels and no Dart-side setup.

The Dart API is not here. It lives in
[`cera_ffi`](https://pub.dev/packages/cera_ffi), which this package depends on
and re-exports, so `import 'package:cera_ffi_flutter/cera_ffi_flutter.dart'`
gives you the whole API and you need only this one dependency. The two are split
because pub will not publish a package declaring `flutter.plugin.platforms`
without a Flutter SDK constraint, and declaring that constraint is exactly what
makes `dart pub get` refuse a package. One package cannot be both a plugin and
plain-Dart-resolvable, so the bindings live in the half that has no Flutter
constraint.

Both wrap the **`cera-ffi` UniFFI surface**, the same C ABI that backs the
Kotlin (`cera-ffi-kotlin`) and Swift bindings.

## Supported platforms

| Platform | Minimum | Native library ships as | Notes |
|----------|---------|------------------------|-------|
| Android  | API 28 | `cera-ffi-android` AAR (Maven Central) | arm64-v8a, armeabi-v7a, x86_64 |
| iOS      | 15.0 | `CeraFFI.xcframework` | Metal enabled; device + simulator |
| macOS    | 12.0 | `CeraFFI.xcframework` | Metal enabled; arm64 |
| Linux    | — | `libcera_ffi.so` | downloaded + checksummed by CMake |
| Windows  | — | `cera_ffi.dll` | downloaded + checksummed by CMake |
| Web      | not supported | — | needs `dart:ffi`; see `cera-wasm` |

Apple targets are wired for both **Swift Package Manager** and CocoaPods;
Flutter picks SPM when the project has it enabled and falls back to the podspec
otherwise.

> **Set your app's deployment target to iOS 15.0 / macOS 12.0.** Flutter's own
> default is lower (iOS 13.0, macOS 10.15), and an app left at the default fails
> with an SPM error that does not say what to do:
>
> ```
> error: The package product 'cera-ffi-flutter' requires minimum platform
> version 15.0 for the iOS platform, but this target supports 13.0
> ```
>
> Raise `IPHONEOS_DEPLOYMENT_TARGET` / `MACOSX_DEPLOYMENT_TARGET` in Xcode (or
> the `platform :ios, '15.0'` line in your Podfile) and the error goes away.
> Flutter propagates the app's deployment target into the plugin package it
> generates, so this is the only knob you need. The floor is not arbitrary: the
> prebuilt slices are compiled with `minos` 15.0 / 12.0.

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
- `../cera_ffi/example/` — plain-Dart CLI scripts covering each surface:
  `cera_chat.dart` (template → generate → decode), `cera_generate.dart`,
  `cera_async.dart`, `cera_stream.dart`, `cera_progress.dart`. They live with
  the API package because they need no Flutter.

```sh
just dart-libs
cd cera_ffi
CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
  dart run example/cera_chat.dart /path/to/model.gguf "Why is the sky blue?"
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

Two packages, side by side in the repo. This one holds only the native build
wiring and a one-line re-export; everything a caller writes against lives in
`cera_ffi/`.

```
cera_ffi_flutter/             # the plugin: native build wiring
├── pubspec.yaml              # cera_ffi dep, flutter constraint, plugin platform block
├── pubspec_overrides.yaml    # resolve cera_ffi from ../ during development
├── android/                  # Gradle module -> cera-ffi-android AAR
├── ios/, macos/              # podspec + SPM manifest -> CeraFFI.xcframework
├── linux/, windows/          # CMake: download + checksum the cdylib
├── example/                  # Flutter app
└── lib/
    └── cera_ffi_flutter.dart # re-export of package:cera_ffi

cera_ffi/                     # the Dart API: no Flutter constraint
├── pubspec.yaml
├── tool/
│   └── patch_generated_bindings.dart  # post-gen fixups (idempotent)
├── example/                  # plain-Dart scripts
├── test/
└── lib/
    ├── cera_ffi.dart         # public barrel (loader + generated bindings)
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

Web is not supported, and the failure is at **compile time**, not run time. The
generated bindings import `dart:ffi` unconditionally, so an app that targets web
and depends on this package fails to build:

```
Error: Dart library 'dart:ffi' is not available on this platform.
Info: The unavailable library 'dart:ffi' is imported through these packages:
    main.dart => package:cera_ffi => dart:ffi
```

Only the *loader* is conditionally exported (`library_loader.dart`), so the
`UnsupportedError` stub it provides is never reached: compilation stops first.
Use `cera-wasm` in browsers.

Making the package merely importable on web needs a stub mirroring the whole
generated API surface, which is why it is not a one-line conditional export.
Making it *work* on web needs a platform-interface package that both the FFI
and wasm implementations satisfy; see "Web" in the repo root README.

## License

Apache-2.0 OR MIT, matching the rest of the workspace.

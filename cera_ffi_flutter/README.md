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
| Linux    | - | `libcera_ffi.so` | downloaded + checksummed by CMake |
| Windows  | - | `cera_ffi.dll` | downloaded + checksummed by CMake |
| Web      | WebGPU, or wasm | `cera_wasm_bg.wasm` in your `web/` | one setup command; see [Web](#web) |

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

`Cera` is the portable API. It works on every platform in the table above, web
included, and it is what an app should reach for first:

```dart
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

// `openPath` where there is a filesystem, `openBytes` where there is not.
// `supportsPaths` is false only on the web.
final cera = Cera.supportsPaths
    ? await Cera.openPath(modelPath)
    : await Cera.openBytes(modelBytes);

debugPrint(cera.backend);   // "native (auto)", "webgpu: … (BrowserWebGpu)", …

final prompt = await cera.applyChatTemplate(
  [const CeraMessage.user('Why is the sky blue?')],
);

var answer = '';
await for (final piece in cera.generate(prompt, maxTokens: 256)) {
  answer += piece;   // fragments, not tokens: just append them
}

await cera.close();
```

Nothing there needs an isolate. Loading and decoding, the two long operations,
already happen off the Dart thread on both transports: natively the engine runs
on a Rust async runtime and completes the future from there, and on the web it
runs in a Web Worker. One step is not off-thread natively, prefill, because
`appendTokens` has no async twin yet; a long prompt blocks the calling isolate
for its prefill, and decode is unaffected.

`Cera.openBundle` downloads a model published on `LiquidAI/LeapBundles` by
`<name, quant>` and opens it, caching it for next time; `Cera.listBundles`
returns the catalog for a picker to offer. It is the constructor to reach for
first on every platform, and especially in a browser, where it keeps the weights
out of JavaScript entirely: one copy of the model instead of two, and clear of
the roughly 2 GiB ceiling on a single contiguous JS allocation. Pass `storeDir`
from `path_provider` on Android and iOS, where it is required.

For a model the user supplies instead, `Cera.openBytes` is the constructor that
also works in a browser, which has no filesystem to point `openPath` at.
`Cera.supportsPaths` says which one to reach
for, and it is worth branching on before the file picker rather than after: a
picker has to be asked for the file's *bytes* up front on the web, and asking
for them on native reads a multi-gigabyte model into the heap that the engine
would otherwise have memory-mapped. `example/lib/main.dart` does exactly this.

### The generated bindings

`Cera` covers loading, chat templating, tokenizing and streaming generation.
Everything else the engine can do (LoRA adapters, vision and audio input,
embeddings, GBNF grammars, tool calling, TurboQuant KV compression) is on the
generated bindings, which are `dart:ffi`-based and therefore **native-only**.
Most of that surface is synchronous:

```dart
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

**This `generate` blocks its isolate for the whole decode**, unlike
`Cera.generate`. Drive it from `Isolate.run`, or use `generateStreamingAsync`,
or use `Cera` and skip the question.

### Voice Modes & Speech Processing

The Flutter package supports multimodal audio pipelines and 4 interactive voice modes:
- **`SpeechToText`**: Microphone input streaming with automatic silence trimming and audio transcription.
- **`VoiceChat`**: Full-duplex interleaved text and audio conversations.
- **`TextToSpeech`**: Synthesizes speech outputs on device with streaming audio playback via `AudioPlayerService`.
- **`TextOnly`**: Standard LLM text turn generation.

```dart
// Voice Activity Detection with Silero VAD v5 (re-exported from cera_ffi)
final vad = FfiSileroVad.fromFile('/path/to/silero_vad.gguf');
final iterator = FfiVadIterator(
  rate: FfiVadSampleRate.rate16kHz,
  config: sileroVadDefaultConfig(),
);
final event = iterator.processChunk(vad, audioFrame512);
```

## Web

Inference in a browser runs through WebGPU when the browser has it, and falls
back to a wasm CPU build when it does not. The difference is not marginal:
**~58 tok/s against ~1.4 tok/s** measured on the same machine and model, so
treat the fallback as a fallback.

The wasm runtime is a build artifact and is not in the pub archive, for the same
reason no other platform's native library is. Install it into your app's `web/`
directory once:

```sh
dart run cera_ffi_flutter:install_web
```

That writes `cera_worker.js`, `cera_wasm.js` and `cera_wasm_bg.wasm` (~3 MB)
into `web/cera/`, which is where the defaults look for them. Re-run it with
`--force` after upgrading the package, since it skips files already present and
the artifacts are versioned with the engine: a stale runtime beside new Dart is
exactly what the flag exists to prevent. Use `--out` to install elsewhere and
pass matching `CeraWebAssets` paths, and `--from DIR` to take the wasm from a
local `just wasm-web-wgpu` build instead of the release.

Then `Cera.openBytes` works as it does anywhere else. **No COOP/COEP headers are
required**: the GPU path does not use threads, and neither does the CPU
fallback, so nothing here needs `SharedArrayBuffer` or a cross-origin-isolated
page.

What is narrower on the web than on native:

- **`openPath` throws.** There is no filesystem; use `openBundle` or `openBytes`.
- **The generated bindings are stubs**, as they have always been on the web:
  `dart:ffi` does not exist there. Everything in "The generated bindings" above
  is native-only.
- **`reset` throws on the GPU path.** Its KV cache lives on the GPU with no way
  to clear it in place. Close the engine and open it again.
- **`cancel` is best-effort.** Cancelling the `generate` stream's
  subscription stops delivery to your app immediately, which is what
  a Stop button needs.

## Examples

- `example/`: a Flutter chat app (the pub.dev example). One code path for every
  platform, web included, because it is written against `Cera`. To run it in a
  browser, build the runtime from this checkout first (the released assets only
  exist from the version that added them onward):
  `just wasm-web-wgpu`, then from `example/`
  `dart run cera_ffi_flutter:install_web --from ../../cera-wasm/examples/webgpu/pkg`
  and `flutter run -d chrome`.
- `../cera_ffi/example/`: plain-Dart CLI scripts covering each surface:
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
audio), `lfm2moe` (routed mixture-of-experts), `llama` (incl. classic Mistral),
`qwen2`/`qwen3`, `granite`.

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

Web **runs**, through `Cera`; see the [Web](#web) section above for setup and
for what is narrower there. What follows is about the *generated bindings*,
which remain native-only.

`dart:ffi` does not exist on the web, and importing it anywhere on the graph
fails the whole build rather than one branch of it. So `cera_ffi` exports its
bindings conditionally, and the web branch is a *generated* stub with the same
API and no FFI. It comes out of the same `just dart-bindings` run as the real
bindings, from the same interface, and CI compiles a throwaway web app against
it, so it cannot quietly fall behind.

Data types are real there: `EngineConfig`, `GenerateOpts`, the error hierarchy
and the enums all construct and compare normally, so shared code that builds a
request stays platform-agnostic. Every engine entry point throws
`UnsupportedError` naming itself:

```
CeraEngine.fromPath is not available on this platform: package `cera_ffi`
needs dart:ffi, which the web does not provide.
```

That is the stub, not the package. Inference in a browser goes through `Cera`,
which is async precisely because the synchronous calls above cannot exist there.
`cera-wasm` remains the option for a non-Dart browser app.

## License

Apache-2.0 OR MIT, matching the rest of the workspace.

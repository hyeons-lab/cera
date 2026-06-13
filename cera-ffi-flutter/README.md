# cera_ffi_flutter

Flutter/Dart bindings for the [Cera](https://github.com/hyeons-lab/cera)
inference engine.

This package wraps the **`cera-ffi` UniFFI surface** — the same C ABI that backs
the Kotlin (`cera-ffi-kotlin`) and Swift bindings — and adds a platform-aware
native-library loader. The Dart bindings are generated from the compiled
`cera-ffi` cdylib by `uniffi-bindgen-dart` and then run through a small,
deterministic patch tool that fixes the generator's known bugs.

> **Status: working (V2.17).** The synchronous engine API — model load,
> sessions, `generate`, `transcribe`, tokenizer access — round-trips real
> inference (verified loading a Qwen2 GGUF and generating tokens). The
> async + streaming-callback surface (`generateStreaming*`, progress sinks) is
> stubbed to throw pending upstream codegen support — see [Limitations](#limitations).

## Layout

```
cera-ffi-flutter/
├── pubspec.yaml              # ffi dep, SDK ^3.3.0
├── analysis_options.yaml     # excludes generated/ from lints
├── tool/
│   └── patch_generated_bindings.dart  # post-gen fixups (idempotent)
├── example/
│   └── cera_generate.dart    # load a GGUF + generate
└── lib/
    ├── cera_ffi_flutter.dart # public barrel (loader + generated bindings)
    └── src/
        ├── library_loader.dart        # CeraLibrary.open() — platform dylib resolution
        └── generated/cera_ffi.dart    # generated + patched UniFFI bindings (committed)
```

## Generating the bindings

Prerequisites: `cargo install uniffi-bindgen-dart` (0.1.3 builds against
`uniffi_bindgen 0.31`, matching this workspace) and a Dart SDK ≥ 3.3.

From the **repo root**:

```sh
just dart-bindings        # builds the cdylib (--features ffi-buffer), generates, patches
```

`just dart-bindings-check` regenerates + patches in place and fails on a diff —
the drift guard for the committed bindings.

### Why a patch step?

`uniffi-bindgen-dart` 0.1.3 has several codegen bugs against Cera's FFI surface.
`tool/patch_generated_bindings.dart` applies deterministic, idempotent fixes:

- corrects the `rustbuffer_*` / `rust_future_*` symbol names (spurious `uniffi_`
  infix) and the `.ref.pointer` → `.ref.ptr` union field;
- rewrites native-library resolution to honor `CERA_FFI_LIB` + a platform name;
- synthesizes the `EngineConfig` record encoder (the generator stubs records
  that contain an interface-handle field);
- fixes the async-constructor return type;
- stubs the unsupported callback-sink methods to throw a clear error.

The `cera-ffi` crate must be built with the **`ffi-buffer`** feature
(`just dart-libs` does this): the Dart generator calls `uniffi_ffibuffer_*`
trampolines that UniFFI only emits under `scaffolding-ffi-buffer-fns`.

## Running the example

```sh
just dart-bindings
cd cera-ffi-flutter
CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
  dart run example/cera_generate.dart /path/to/model.gguf "Once upon a time"
```

The FFI surface returns token IDs (no detokenizer is exposed yet), so the
example prints the token count + decode timing. Supported architectures match
the engine (`lfm2`, `qwen2`, `qwen3` at time of writing).

## Native library

`CeraLibrary.open()` resolves the cdylib per platform:

| Platform | Resolves to |
|----------|-------------|
| macOS    | `libcera_ffi.dylib` |
| Linux / Android | `libcera_ffi.so` |
| Windows  | `cera_ffi.dll` |
| iOS      | `DynamicLibrary.process()` (statically linked) |

The generated default loader (patched) honors `CERA_FFI_LIB` for an explicit
path and otherwise opens the platform filename. Packaging the prebuilt libs per
target (Android jniLibs, iOS xcframework, desktop bundles) is follow-up work.

## Limitations

Tracked in `docs/IMPLEMENTATION_PLAN.md` → **V2.17**:

**Works (verified):**
- **Sync `generate`** — `example/cera_generate.dart`.
- **Token streaming** — `Session.generateStreaming(opts, sink)` → Dart
  `ModalitySink` (`onTextTokens`/`onAudioFrames`/`onDone`),
  `example/cera_stream.dart`. The vendored generator
  (`third_party/uniffi-bindgen-dart/`) carries the four fixes that enable it
  (callback-arg lowering, vtable-init symbol, vtable slot order, RustBuffer
  callback-arg ABI — to be upstreamed).
- **`generateAsync`** — real `Future` via the rust-future poll loop; the event
  loop stays responsive during decode (`example/cera_async.dart`).
- **`BundleRepo.withProgress`** — `DownloadProgressSink.onProgress` fires with
  correctly decoded args (`example/cera_progress.dart`).

**Not yet supported (throw `UnsupportedError`):**
- **`generateStreamingAsync`** — cera runs it on a tokio worker thread, so the
  sink fires off-isolate; `NativeCallable.isolateLocal` can't service that. Needs
  `NativeCallable.listener` vtables. Use sync `generateStreaming` (optionally in
  a Dart `Isolate`) instead.
- **`fromBundleIdAsync`** — async constructor returning an object handle; needs
  the object/pointer rust-future variant.
- **No detokenizer** over FFI — `generate` returns token IDs.

> The callback vtable's static `NativeCallable.isolateLocal`s keep the isolate
> alive, so a CLI script must `exit()` explicitly (the example does); a Flutter
> app stays running regardless.

# cera_ffi_flutter

Flutter/Dart bindings for the [Cera](https://github.com/hyeons-lab/cera)
inference engine.

This package wraps the **`cera-ffi` UniFFI surface** — the same C ABI that backs
the Kotlin (`cera-ffi-kotlin`) and Swift bindings — and adds a platform-aware
native-library loader. Only the Dart bindings under `lib/src/generated/` are
generated (produced from the compiled `cera-ffi` cdylib by
`uniffi-bindgen-dart`, never edited by hand); the loader, barrel, and package
scaffold are maintained in-tree.

> **Status: scaffold (V2.17, ⬜ in progress).** The package skeleton, loader,
> and generation tooling are in place. The generated bindings are **not yet
> committed** because `uniffi-bindgen-dart` 0.1.3 emits invalid Dart for Cera's
> async + streaming-callback surface — see [Known gaps](#known-gaps).

## Layout

```
cera-ffi-flutter/
├── pubspec.yaml              # ffi dep, SDK ^3.3.0
├── analysis_options.yaml     # excludes generated/ from lints
├── lib/
│   ├── cera_ffi_flutter.dart # public barrel (loader + generated re-export)
│   └── src/
│       ├── library_loader.dart      # conditional export (io vs web)
│       ├── library_loader_io.dart   # CeraLibrary.open() — dart:ffi dylib resolution
│       ├── library_loader_web.dart  # web stub — open() throws UnsupportedError
│       └── generated/
│           └── cera_ffi.dart     # generated bindings (gitignored; run codegen)
```

## Generating the bindings

Prerequisites: `cargo install uniffi-bindgen-dart` (0.1.3 builds against
`uniffi_bindgen 0.31`, matching this workspace) and a Dart SDK ≥ 3.3.

From the **repo root**:

```sh
just dart-bindings
```

This builds the `cera-ffi` debug cdylib and runs the generator into
`cera-ffi-flutter/lib/src/generated/cera_ffi.dart`. After it exists, uncomment
the generated export in `lib/cera_ffi_flutter.dart`.

## Native library

`CeraLibrary.open()` resolves the cdylib per platform:

| Platform | Resolves to |
|----------|-------------|
| macOS    | `libcera_ffi.dylib` |
| Linux / Android | `libcera_ffi.so` |
| Windows  | `cera_ffi.dll` |
| iOS      | `DynamicLibrary.process()` (statically linked) |

Pass the result into the generated API explicitly:

```dart
final lib = CeraLibrary.open();           // or CeraLibrary.open(path: '…')
final ffi = CeraFfiFfi(dynamicLibrary: lib);
```

We inject the library rather than relying on the generated name-based lookup:
the generator defaults to `uniffi_cera_ffi`, but the actual cdylib base name is
`cera_ffi`. Packaging the prebuilt libs per target (Android jniLibs, iOS
xcframework, desktop bundles) is follow-up work.

## Platform support

**Runs on native platforms only** (Android, iOS, macOS, Linux, Windows) — it's
a `dart:ffi` package and Flutter Web has no FFI. The loader uses a conditional
export (`library_loader.dart` → `library_loader_io.dart` when `dart:io` is
available, else `library_loader_web.dart`), so a multi-platform app that also
targets web can still **import** the package: on web `CeraLibrary.open()` throws
a clear `UnsupportedError` instead of breaking compilation. Web *support* (not
just import-safety) would require a non-FFI transport (e.g. WASM via
`cera-wasm`) and is out of scope for these bindings.

> Note: once the generated UniFFI bindings are committed and exported from the
> barrel, they'll need the same conditional-export guard — they're `dart:ffi`
> throughout. The export is commented out for now (see [Generating the
> bindings](#generating-the-bindings)), so the committed scaffold is currently
> web-import-safe end-to-end.

## Known gaps

`dart analyze` on freshly generated bindings reports **8 errors**, all in the
advanced FFI surface (the sync core — structs, enums, `CeraEngine.transcribe` —
is clean):

- callback / foreign-trait sinks `DownloadProgressSink`, `ModalitySink`
  (download progress + audio-modality streaming) → invalid casts;
- async constructor `fromBundleIdAsync` returns `CeraEngine` instead of
  `Future<CeraEngine>`;
- `_UniFfiFfiBufferElement.pointer` undefined getter in sequence handling.

Tracked in `docs/IMPLEMENTATION_PLAN.md` → **V2.17**. Paths forward: narrow the
Dart-exposed surface + hand-written shims, patch upstream, or use
`flutter_rust_bridge` for the streaming pieces.

/// Flutter/Dart bindings for the Cera inference engine.
///
/// This is the package's public entry point. It re-exports:
/// - [CeraLibrary] — the platform-aware native-library loader.
/// - the UniFFI bindings from `src/generated/cera_ffi.dart` (the engine API:
///   `CeraEngine`, `EngineConfig`, …).
///
/// ## Regenerating the bindings
///
/// The generated UniFFI bindings under `lib/src/generated/` are committed and
/// exported (see V2.17 in `docs/IMPLEMENTATION_PLAN.md`). They're regenerated +
/// patched from the repo root with:
///
/// ```sh
/// just dart-bindings        # rebuilds the cdylib, regenerates, patches
/// just dart-bindings-check  # fails if the committed bindings drift
/// ```
///
/// Edit the Rust `#[uniffi::*]` surface or `tool/patch_generated_bindings.dart`,
/// never `lib/src/generated/cera_ffi.dart` by hand.
///
/// ## Usage sketch
///
/// ```dart
/// import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
///
/// final lib = CeraLibrary.open();
/// final ffi = CeraFfiFfi(dynamicLibrary: lib); // from generated bindings
/// // … construct an engine, run generate(), etc.
/// ```
library cera_ffi_flutter;

export 'src/library_loader.dart';
export 'src/generated/cera_ffi.dart';

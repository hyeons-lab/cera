/// Flutter/Dart bindings for the Cera inference engine.
///
/// This is the package's public entry point. It re-exports:
/// - [CeraLibrary] — the platform-aware native-library loader.
/// - the UniFFI engine bindings from `src/generated/cera_ffi.dart`
///   (`CeraEngine`, `EngineConfig`, `Session`, `ModalitySink`, …).
///
/// ## Regenerating the bindings
///
/// The generated UniFFI bindings are committed under
/// `src/generated/cera_ffi.dart`. Regenerate them from the repo root if the
/// Rust FFI surface changes (see V2.17 in `docs/IMPLEMENTATION_PLAN.md`):
///
/// ```sh
/// just dart-bindings
/// ```
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

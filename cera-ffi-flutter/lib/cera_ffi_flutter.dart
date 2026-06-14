/// Flutter/Dart bindings for the Cera inference engine.
///
/// This is the package's public entry point. It re-exports:
/// - [CeraLibrary] — the platform-aware native-library loader.
/// - (after generation) the UniFFI bindings from `src/generated/cera_ffi.dart`.
///
/// ## Generating the bindings
///
/// The generated UniFFI bindings are not committed (see V2.17 in
/// `docs/IMPLEMENTATION_PLAN.md`). Produce them from the repo root with:
///
/// ```sh
/// just dart-bindings
/// ```
///
/// which writes `lib/src/generated/cera_ffi.dart`. Once present, uncomment the
/// export below to surface the engine API (`CeraEngine`, `EngineConfig`, …).
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

// Uncomment once `just dart-bindings` has generated the file. Kept commented
// so the committed scaffold analyzes cleanly without the generated artifact.
// export 'src/generated/cera_ffi.dart';

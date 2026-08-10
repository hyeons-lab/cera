/// Dart bindings for the Cera inference engine.
///
/// This is the package's public entry point. It re-exports:
/// - [CeraLibrary] — the platform-aware native-library loader.
/// - the UniFFI engine bindings from `src/generated/cera_ffi.dart`
///   (`CeraEngine`, `EngineConfig`, `Session`, `ModalitySink`, …).
///
/// Pure `dart:ffi`: nothing here imports `package:flutter`, so this works from
/// a plain Dart CLI, server, or test. Flutter apps normally depend on
/// `package:cera_ffi_flutter` instead, which re-exports this library *and*
/// wires each platform's build to ship the native library. Depending on this
/// package alone means supplying that library yourself, via `CERA_FFI_LIB` or
/// the loader path.
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
/// import 'package:cera_ffi/cera_ffi.dart';
///
/// final engine = CeraEngine(
///   modelPath: '/path/to/model.gguf',
///   config: const EngineConfig(),
/// );
/// final out = engine.generate(prompt: 'Why is the sky blue?');
/// print(engine.decodeTokens(tokens: out.tokens));
/// ```
library;

export 'src/library_loader.dart';
export 'src/generated/cera_ffi.dart';

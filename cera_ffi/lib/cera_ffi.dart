/// Dart bindings for the Cera inference engine.
///
/// This is the package's public entry point. It re-exports:
/// - [Cera], the portable asynchronous API. Runs everywhere, web included.
/// - [CeraLibrary], the platform-aware native-library loader.
/// - the UniFFI engine bindings from `src/generated/cera_ffi.dart`
///   (`CeraEngine`, `EngineConfig`, `Session`, `ModalitySink`, …), which are
///   `dart:ffi`-based and therefore native-only.
///
/// Nothing here imports `package:flutter`, so this works from a plain Dart
/// CLI, server, or test. Flutter apps normally depend on
/// `package:cera_ffi_flutter` instead, which re-exports this library *and*
/// wires each platform's build to ship the native library. Depending on this
/// package alone means supplying that library yourself, via `CERA_FFI_LIB` or
/// the loader path.
///
/// ## Web
///
/// Inference runs in a browser through [Cera], on WebGPU where the browser has
/// it and on a wasm CPU build where it does not. It needs the runtime installed
/// once with `dart run cera_ffi:install_web`; see the README.
///
/// The *generated bindings* remain native-only, and the web branch of them is a
/// generated stub whose every entry point throws [UnsupportedError]. That stub
/// is what lets a multi-platform app compile at all: `dart:ffi` is unavailable
/// on web, and an unconditional import of it is a compile error for the whole
/// build, not a runtime failure on one branch. Data types (records, enums,
/// errors) are real there, so code that only builds a config or inspects a
/// result still runs.
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
/// final cera = await Cera.openPath('/path/to/model.gguf');
/// await for (final piece in cera.generate('Why is the sky blue?')) {
///   // fragments, not tokens: just append them
/// }
/// await cera.close();
/// ```
library;

export 'src/async/cera.dart';
export 'src/library_loader.dart';

// The stub is the DEFAULT and the real bindings are the conditional branch,
// which is the way round a conditional export has to be written: the first URI
// is used when no condition holds. `dart.library.ffi` rather than
// `dart.library.io`, because FFI availability is exactly what differs.
//
// Both files are generated from the same interface by `just dart-bindings`, and
// the generator has tests asserting their public surfaces match member for
// member; a name in one and not the other is a compile error on whichever
// platform is built second.
export 'src/generated/cera_ffi_web.dart'
    if (dart.library.ffi) 'src/generated/cera_ffi.dart';

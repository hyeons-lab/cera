import 'dart:ffi' as ffi;
import 'dart:io' show Platform;

/// Resolves and opens the `cera-ffi` native library.
///
/// The Rust crate `cera-ffi` builds a cdylib whose base name is `cera_ffi`
/// (`libcera_ffi.{so,dylib}` / `cera_ffi.dll`). Note this differs from the
/// UniFFI-generated default of `uniffi_cera_ffi` — so callers should pass the
/// [ffi.DynamicLibrary] returned here into the generated API constructor
/// (`CeraFfiFfi(dynamicLibrary: CeraLibrary.open())`) rather than relying on
/// the generated code's name-based lookup.
///
/// On iOS the symbols are linked statically into the host process, so we open
/// the process image instead of a standalone file.
class CeraLibrary {
  CeraLibrary._();

  /// Base name of the cdylib, without platform prefix/suffix.
  static const String baseName = 'cera_ffi';

  /// Open the native library using platform conventions.
  ///
  /// Pass [path] to override the lookup with an explicit file (useful for
  /// tests or when shipping the lib in a non-standard location).
  static ffi.DynamicLibrary open({String? path}) {
    if (path != null) {
      return ffi.DynamicLibrary.open(path);
    }
    if (Platform.isIOS) {
      // Static archive linked into the app binary.
      return ffi.DynamicLibrary.process();
    }
    return ffi.DynamicLibrary.open(_platformFileName());
  }

  static String _platformFileName() {
    if (Platform.isMacOS) return 'lib$baseName.dylib';
    if (Platform.isWindows) return '$baseName.dll';
    // Android, Linux, and other unix-likes.
    return 'lib$baseName.so';
  }
}

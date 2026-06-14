import 'dart:ffi' as ffi;
import 'dart:io' show Platform;

/// Resolves and opens the `cera-ffi` native library (native platforms).
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
///
/// This is the `dart:ffi` implementation, selected via the conditional export
/// in `library_loader.dart` on every target that has `dart:io`. Flutter Web
/// gets [library_loader_web.dart] instead (a stub that throws).
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
    if (Platform.isAndroid || Platform.isLinux) return 'lib$baseName.so';
    // iOS is handled by `open()` (static process image); anything else has no
    // bundled library, so fail loudly instead of guessing a `.so` name.
    throw UnsupportedError(
      'cera-ffi has no bundled native library for ${Platform.operatingSystem}; '
      'pass an explicit path to CeraLibrary.open(path: ...).',
    );
  }
}

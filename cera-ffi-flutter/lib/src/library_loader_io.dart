import 'dart:ffi' as ffi;
import 'dart:io' show Platform;

/// Resolves and opens the `cera-ffi` native library (native platforms).
///
/// This is the single source of truth for library resolution: the generated
/// bindings' own default loader delegates here (see
/// `tool/patch_generated_bindings.dart`), so a Flutter app and a plain
/// `dart run` script resolve the library the same way.
///
/// This is the `dart:ffi` implementation, selected via the conditional export
/// in `library_loader.dart` on every target that has `dart:io`. Flutter Web
/// gets `library_loader_web.dart` instead (a stub that throws).
class CeraLibrary {
  CeraLibrary._();

  /// Base name of the library, without platform prefix/suffix.
  static const String baseName = 'cera_ffi';

  /// A symbol that is always exported by the cera-ffi scaffolding. Used to
  /// tell "statically linked into this process" from "not here at all" when
  /// probing the process image.
  static const String _probeSymbol = 'ffi_cera_ffi_uniffi_contract_version';

  /// Environment variable holding an explicit path to the library. Primarily
  /// for local development and the plain-Dart examples, where the cdylib sits
  /// in `target/debug/` rather than anywhere a platform loader would look.
  static const String pathEnvVar = 'CERA_FFI_LIB';

  /// Open the native library using platform conventions.
  ///
  /// Resolution order:
  ///  1. [path], when given.
  ///  2. `$CERA_FFI_LIB`, when set and non-empty.
  ///  3. The process image, on platforms where the library is linked
  ///     statically into the host binary (see below).
  ///  4. The platform-conventional shared-library filename.
  ///
  /// **Apple platforms link statically.** The published
  /// `CeraFFI.xcframework` vends `libcera_ffi.a`, so on iOS and on macOS under
  /// Flutter the symbols end up in the app binary and there is no dylib to
  /// open. iOS is always that case. macOS can be either: statically linked
  /// under Flutter, or a loose `libcera_ffi.dylib` for a plain Dart script, so
  /// it probes the process image first and falls back to the dylib.
  static ffi.DynamicLibrary open({String? path}) {
    if (path != null && path.isNotEmpty) {
      return ffi.DynamicLibrary.open(path);
    }

    final envPath = Platform.environment[pathEnvVar];
    if (envPath != null && envPath.isNotEmpty) {
      return ffi.DynamicLibrary.open(envPath);
    }

    if (Platform.isIOS) {
      // Always a static archive linked into the app binary.
      return ffi.DynamicLibrary.process();
    }

    if (Platform.isMacOS) {
      final process = ffi.DynamicLibrary.process();
      if (process.providesSymbol(_probeSymbol)) {
        return process;
      }
      return ffi.DynamicLibrary.open('lib$baseName.dylib');
    }

    return ffi.DynamicLibrary.open(_platformFileName());
  }

  static String _platformFileName() {
    if (Platform.isWindows) return '$baseName.dll';
    // Android gets the .so from the cera-ffi-android AAR's jniLibs; Linux gets
    // it from the plugin's CMake install step. Both resolve by bare name.
    if (Platform.isAndroid || Platform.isLinux) return 'lib$baseName.so';
    // iOS and macOS are handled in `open()`; anything else has no bundled
    // library, so fail loudly instead of guessing a `.so` name.
    throw UnsupportedError(
      'cera-ffi has no bundled native library for ${Platform.operatingSystem}; '
      'pass an explicit path to CeraLibrary.open(path: ...) or set '
      '\$$pathEnvVar.',
    );
  }
}

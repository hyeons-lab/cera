/// Flutter Web stub for [CeraLibrary].
///
/// This package is `dart:ffi`-based and cannot run on the web (no FFI). This
/// stub exists only so the package stays *importable* in a multi-platform app
/// that also targets web — calling [open] throws a clear [UnsupportedError]
/// instead of failing to compile. It is selected by the conditional export in
/// `library_loader.dart` when `dart:io` is unavailable (i.e. web).
///
/// Web support would require a non-FFI transport (e.g. WASM via `cera-wasm`),
/// which is out of scope for these UniFFI bindings.
class CeraLibrary {
  CeraLibrary._();

  /// Base name of the native library, without platform prefix/suffix.
  static const String baseName = 'cera_ffi';

  /// Environment variable holding an explicit path to the library.
  ///
  /// Meaningless on web (there is no filesystem and no FFI), but declared so
  /// this stub keeps the same public surface as the `dart:io` implementation.
  /// The conditional export in `library_loader.dart` means static analysis
  /// resolves against THIS class by default, so anything missing here is a
  /// compile error for callers even though the VM runs the other variant.
  static const String pathEnvVar = 'CERA_FFI_LIB';

  /// Always throws on web — `cera-ffi` requires `dart:ffi`.
  static Never open({String? path}) => throw UnsupportedError(
    'cera_ffi requires dart:ffi and does not support the web; '
    'use it only from native targets (Android, iOS, macOS, Linux, Windows). '
    'For browsers, use cera-wasm.',
  );
}

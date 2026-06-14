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

  /// Base name of the cdylib, without platform prefix/suffix.
  static const String baseName = 'cera_ffi';

  /// Always throws on web — `cera-ffi` requires `dart:ffi`.
  static Never open({String? path}) => throw UnsupportedError(
    'cera_ffi_flutter requires dart:ffi and does not support Flutter Web; '
    'use it only from native targets (Android, iOS, macOS, Linux, Windows).',
  );
}

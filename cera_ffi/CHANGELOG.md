# Changelog

## 0.4.0

First release. The Dart bindings previously lived inside the
`cera_ffi_flutter` package; they were split out here so that they can be used
without Flutter at all.

### Added

- `dart:ffi` bindings for the Cera inference engine, generated from the compiled
  `cera-ffi` cdylib by a vendored `uniffi-bindgen-dart` and committed under
  `lib/src/generated/`.
- `CeraLibrary`, a platform-aware loader: an explicit path, then `CERA_FFI_LIB`,
  then the platform's normal search path.
- Plain-Dart examples under `example/`: chat, streaming, async, and download
  progress.

### Notes

- **This package resolves under plain `dart pub get`, which is the reason it
  exists.** pub will not publish a package declaring `flutter.plugin.platforms`
  without a Flutter SDK constraint, and declaring that constraint makes
  `dart pub get` refuse the package outright. A Flutter plugin therefore cannot
  also be a plain-Dart package, so the bindings live here and
  `cera_ffi_flutter` depends on them.
- No native library ships with this package. Flutter apps get one from
  `cera_ffi_flutter`; everyone else points `CERA_FFI_LIB` at a `cera-ffi` build.
  That build needs the `ffi-buffer` cargo feature, without which every call
  fails at `dlsym`.
- The web compiles but runs nothing. The bindings are exported conditionally and
  the web branch is a generated stub with the same API and no `dart:ffi`: data
  types are real, engine entry points throw `UnsupportedError`. This exists so a
  multi-platform app builds at all, since an unconditional `dart:ffi` import
  fails the whole build rather than one branch. Use `cera-wasm` in browsers.

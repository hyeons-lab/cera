// Placeholder source for the `cera_ffi_flutter` SPM target.
//
// `cera_ffi_flutter` is an FFI-only plugin: there is no platform channel and no
// Swift API surface. Dart talks to the native library directly through
// `dart:ffi`. This target exists purely so SPM has something to build and to
// carry the `CeraFFI` binary dependency into the app; SPM rejects a target with
// no sources.
//
// Deliberately empty apart from this marker. Anything added here would be dead
// code from Dart's point of view.

enum CeraFFIPlugin {
    /// Version of the plugin this package was generated for. Not read by
    /// Flutter or by Dart; present so the file declares something and so a
    /// human reading the built app can tell which plugin version linked in.
    static let version = "0.5.3"
}

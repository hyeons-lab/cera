/// Platform-aware entry point for the `cera-ffi` native-library loader.
///
/// Re-exports the right [CeraLibrary] for the target: the `dart:ffi`
/// implementation ([library_loader_io.dart]) wherever `dart:io` exists, and a
/// throwing stub ([library_loader_web.dart]) on Flutter Web (no FFI). The
/// conditional keeps the package importable in multi-platform apps that also
/// target web — see the README "Platform support" section.
library;

export 'library_loader_web.dart'
    if (dart.library.io) 'library_loader_io.dart';

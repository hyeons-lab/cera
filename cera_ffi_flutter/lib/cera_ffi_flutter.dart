/// Flutter bindings for the Cera inference engine.
///
/// This library is a re-export of `package:cera_ffi`, and deliberately holds no
/// code of its own. The Dart API lives in that package because it has no
/// Flutter SDK constraint and so resolves under plain `dart pub get`; a plugin
/// cannot avoid declaring one, and declaring one is what makes `dart pub get`
/// refuse a package. Everything this package adds is under `android/`, `ios/`,
/// `macos/`, `linux/` and `windows/`: the per-platform build wiring that ships
/// the native library with your app.
///
/// So a Flutter app depends on this package and imports this one library:
///
/// ```dart
/// import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
///
/// final engine = CeraEngine(
///   modelPath: path,
///   config: const EngineConfig(),
/// );
/// ```
///
/// Importing `package:cera_ffi/cera_ffi.dart` directly works too and is
/// identical; the two names exist so that non-Flutter Dart programs have
/// something to depend on.
library;

export 'package:cera_ffi/cera_ffi.dart';

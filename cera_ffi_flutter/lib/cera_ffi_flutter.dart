/// Flutter bindings for the Cera inference engine.
///
/// This library is a re-export of `package:cera_ffi`, and deliberately holds no
/// code of its own. The Dart API lives in that package because it has no
/// Flutter SDK constraint and so resolves under plain `dart pub get`; a plugin
/// cannot avoid declaring one, and declaring one is what makes `dart pub get`
/// refuse a package. Everything this package adds is under `android/`, `ios/`,
/// `macos/`, `linux/` and `windows/`: the per-platform build wiring that ships
/// the native library with your app, plus `bin/install_web.dart`, the one-time
/// setup command for the browser runtime.
///
/// So a Flutter app depends on this package and imports this one library:
///
/// ```dart
/// import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
///
/// final cera = await Cera.openPath(path);
/// var answer = '';
/// await for (final piece in cera.generate('Why is the sky blue?')) {
///   answer += piece;
/// }
/// ```
///
/// [Cera] is the portable API and runs on every platform this plugin supports,
/// the web included. The generated `CeraEngine` bindings underneath cover the
/// whole engine but are synchronous and `dart:ffi`-based, so they are
/// native-only. See the README.
///
/// Importing `package:cera_ffi/cera_ffi.dart` directly works too and is
/// identical; the two names exist so that non-Flutter Dart programs have
/// something to depend on.
library;

export 'package:cera_ffi/cera_ffi.dart';

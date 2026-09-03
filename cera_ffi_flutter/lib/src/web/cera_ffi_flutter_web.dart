import 'package:flutter_web_plugins/flutter_web_plugins.dart';

/// Web registrant for the Cera Flutter plugin.
///
/// Cera browser inference executes inside a Web Worker via cera-wasm and
/// dart:js_interop, so no platform channel registration is needed.
class CeraFfiFlutterWeb {
  /// Registers this plugin with the Flutter Web engine.
  static void registerWith(Registrar registrar) {
    // No-op: Cera communicates directly with cera.worker.js
  }
}

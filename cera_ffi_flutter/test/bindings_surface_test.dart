@TestOn('vm')
library;

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:test/test.dart';

/// Compile-time guard on the generated binding surface.
///
/// These are not behavior tests — none of them call into the native library.
/// They exist because two whole classes of regression are invisible to
/// `dart analyze`:
///
///  1. A method silently regressing into a `throw UnsupportedError` body. The
///     generator did exactly this for every RustBuffer-returning method, which
///     killed `decodeTokens`, `applyChatTemplate`, and the embedding methods at
///     runtime while the package still analyzed clean.
///  2. The committed bindings drifting behind the Rust FFI surface. That
///     happened for ~3 weeks: `GenerateOpts` lost three fields and
///     `encodeTextSpecial` was missing entirely, which makes the library fail
///     its UniFFI checksum at construction.
///
/// Referencing the symbols means the package stops compiling if they vanish.
void main() {
  test('GenerateOpts carries every field the Rust record declares', () {
    // 12 fields as of cera-ffi/src/lib.rs. `grammarTriggerTokens`,
    // `flushEveryTokens`, and `flushEveryMs` are the three that went missing.
    const opts = GenerateOpts(
      maxTokens: 1,
      temperature: 0.0,
      topP: 1.0,
      topK: 0,
      minP: 0.0,
      repetitionPenalty: 1.0,
      stopTokens: <int>[],
      ignoreEos: false,
      grammar: null,
      grammarTriggerTokens: <int>[],
      flushEveryTokens: 0,
      flushEveryMs: 0,
    );

    expect(opts.maxTokens, 1);
    expect(opts.grammarTriggerTokens, isEmpty);
    expect(opts.flushEveryTokens, 0);
    expect(opts.flushEveryMs, 0);
  });

  test('the RustBuffer-returning methods are declared on CeraEngine', () {
    // The guard is `_surfaceGuard` below, which is resolved at compile time.
    // If any of those methods is dropped or renamed, this file stops
    // compiling and the test fails to load. Nothing to assert at runtime.
    expect(_surfaceGuard, isA<Function>());
  });
}

/// Compile-time reference to every method that used to be a runtime stub.
///
/// Deliberately never called: passing a real `CeraEngine` would need a model on
/// disk. Only static resolution matters, and that happens whether or not the
/// body runs.
void Function(CeraEngine) get _surfaceGuard => (CeraEngine engine) {
      // String returns.
      engine.decodeTokens(const <int>[]);
      engine.applyChatTemplate(const <ChatMessage>[], true);
      engine.applyChatTemplateWithTools(
          const <ChatMessage>[], const <ToolDef>[], true);
      engine.transcribe(const <double>[], 16000);
      // Sequence returns.
      engine.encodeText('');
      engine.encodeTextSpecial('', false);
      // Optional-enum return.
      engine.toolFormat();
    };

@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:cera_ffi/cera_ffi.dart';
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

  test('the RustBuffer-returning methods are declared on Session', () {
    expect(_sessionSurfaceGuard, isA<Function>());
  });

  test('the RustBuffer-returning methods are declared on BundleRepo', () {
    expect(_bundleRepoSurfaceGuard, isA<Function>());
  });

  test('fromPartsAsync returns a Future, not a bare engine', () {
    // Same async-constructor hazard as `fromBundleIdAsync` below, and the same
    // signal. The multimodal constructor matters on its own: it is the only way
    // to load a model plus its mmproj without a filesystem, so if the generator
    // drops it, `Cera.openBytes(mmproj: ...)` silently loses its native half
    // and VL-from-memory becomes web-only again.
    expect(
      CeraEngine.fromPartsAsync,
      isA<
          Future<CeraEngine> Function(
              Uint8List, Uint8List?, String?, EngineConfig)>(),
    );
  });

  test('fromBundleIdAsync returns a Future, not a bare engine', () {
    // The generator had no rust-future path for async *constructors* (only for
    // methods), so this used to be a synchronous `throw UnsupportedError`. The
    // static type is the regression signal: a stub body that threw would still
    // satisfy `Future<CeraEngine>` if the wrapper were marked `async`, but the
    // tear-off's type would not survive the generator dropping the ctor.
    expect(CeraEngine.fromBundleIdAsync,
        isA<Future<CeraEngine> Function(String, String, EngineConfig)>());
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

/// The same guard for `Session`, which the RustBuffer regression hit just as
/// hard as `CeraEngine`. The `hiddenStates*` trio is the reason this matters
/// beyond "it throws": those three went from throwing to returning *corrupt*
/// data when `Vec<u8>`'s wire format was wrong in both directions, which a
/// stub-only check would have called fixed.
void Function(Session) get _sessionSurfaceGuard => (Session session) {
      // Record returns.
      session.capabilities();
      session.generate(const GenerateOpts());
      // `Vec<u8>` / `Vec<f32>` returns.
      session.hiddenStatesForText('');
      session.hiddenStatesForTokens(const <int>[]);
      session.hiddenStatesMeanPooled(const <int>[]);
      // `Vec<u8>` argument alongside an optional-primitive argument.
      session.appendImage(Uint8List(0), null);
      session.setImageMaxLongSize(null);
    };

/// And for `BundleRepo`, whose `storeDir` is a plain string return and so was
/// stubbed by the same renderer gap.
void Function(BundleRepo) get _bundleRepoSurfaceGuard => (BundleRepo repo) {
      repo.storeDir();
      repo.cacheSize();
    };

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
    // 13 fields as of cera-ffi/src/lib.rs (including spec). `grammarTriggerTokens`,
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
      spec: SpecDecodeConfig(ngram: 3, k: 5),
    );

    expect(opts.maxTokens, 1);
    expect(opts.grammarTriggerTokens, isEmpty);
    expect(opts.flushEveryTokens, 0);
    expect(opts.flushEveryMs, 0);
    expect(opts.spec?.ngram, 3);
    expect(opts.spec?.k, 5);
  });

  test(
    'CeraOptions carries kvCompression, ubatchSize, and effectiveKvCompression',
    () {
      const optsDefault = CeraOptions();
      expect(optsDefault.ubatchSize, 512);
      expect(optsDefault.turboQuant, isFalse);
      expect(optsDefault.kvCompression, isNull);
      expect(optsDefault.effectiveKvCompression, CeraKvCompression.none);

      const optsTurboQuant = CeraOptions(turboQuant: true);
      expect(
        optsTurboQuant.effectiveKvCompression,
        CeraKvCompression.turboQuant,
      );

      const optsF16 = CeraOptions(kvCompression: CeraKvCompression.f16);
      expect(optsF16.effectiveKvCompression, CeraKvCompression.f16);

      const optsF16Override = CeraOptions(
        turboQuant: true,
        kvCompression: CeraKvCompression.f16,
        ubatchSize: 256,
      );
      expect(optsF16Override.ubatchSize, 256);
      expect(optsF16Override.effectiveKvCompression, CeraKvCompression.f16);

      const optsKvTurboQuant = CeraOptions(
        kvCompression: CeraKvCompression.turboQuant,
      );
      expect(
        optsKvTurboQuant.effectiveKvCompression,
        CeraKvCompression.turboQuant,
      );
      expect(optsKvTurboQuant.turboQuant, isFalse);

      final copied = optsDefault.copyWith(
        ubatchSize: 256,
        kvCompression: CeraKvCompression.f16,
      );
      expect(copied.ubatchSize, 256);
      expect(copied.effectiveKvCompression, CeraKvCompression.f16);
      expect(copied.contextSize, optsDefault.contextSize);

      const spec = CeraSpecDecode(ngram: 4, k: 10);
      expect(spec.ngram, 4);
      expect(spec.k, 10);
      expect(spec, equals(const CeraSpecDecode(ngram: 4, k: 10)));
      expect(
        spec.hashCode,
        equals(const CeraSpecDecode(ngram: 4, k: 10).hashCode),
      );
      expect(spec, isNot(equals(const CeraSpecDecode(ngram: 2, k: 10))));
      expect(spec, isNot(equals(const CeraSpecDecode(ngram: 4, k: 6))));
      expect(spec.copyWith(k: 8), equals(const CeraSpecDecode(ngram: 4, k: 8)));
      expect(
        spec.copyWith(ngram: 3),
        equals(const CeraSpecDecode(ngram: 3, k: 10)),
      );
      expect(() => CeraSpecDecode(ngram: 0), throwsA(isA<AssertionError>()));
      expect(() => CeraSpecDecode(k: 0), throwsA(isA<AssertionError>()));
      expect(() => CeraSpecDecode(ngram: -1), throwsA(isA<AssertionError>()));
    },
  );

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
          Uint8List,
          Uint8List?,
          String?,
          EngineConfig,
        )
      >(),
    );
  });

  test('fromBundleIdAsync returns a Future, not a bare engine', () {
    // The generator had no rust-future path for async *constructors* (only for
    // methods), so this used to be a synchronous `throw UnsupportedError`. The
    // static type is the regression signal: a stub body that threw would still
    // satisfy `Future<CeraEngine>` if the wrapper were marked `async`, but the
    // tear-off's type would not survive the generator dropping the ctor.
    expect(
      CeraEngine.fromBundleIdAsync,
      isA<Future<CeraEngine> Function(String, String, EngineConfig)>(),
    );
  });

  test('FfiSileroVad and VAD types are exposed on the Dart surface', () {
    expect(_vadSurfaceGuard, isA<Function>());
    expect(_vadIteratorSurfaceGuard, isA<Function>());
    const cfg = FfiVadConfig(
      threshold: 0.5,
      negThreshold: 0.35,
      minSpeechDurationMs: 64,
      minSilenceDurationMs: 100,
      speechPadMs: 30,
    );
    expect(cfg.threshold, 0.5);
    expect(cfg.minSpeechDurationMs, 64);
    expect(FfiVadSampleRate.values, contains(FfiVadSampleRate.rate16kHz));
    expect(FfiVadSampleRate.values, contains(FfiVadSampleRate.rate8kHz));
  });

  test('GenerateSummary exposes throughput calculations', () {
    const summary = GenerateSummary(
      tokensGenerated: 100,
      promptEvalTokens: 50,
      promptEvalMs: 500,
      decodeMs: 2000,
      totalDurationMs: 2500,
      decodeTokPerSec: 50.0,
      promptEvalTokPerSec: 100.0,
      finishReason: FinishReasonStop(),
    );
    expect(summary.totalDurationMs, 2500);
    expect(summary.decodeTokPerSec, 50.0);
    expect(summary.promptEvalTokPerSec, 100.0);
  });

  test('UserMessage and AudioInput types are exposed on the Dart surface', () {
    const audio = AudioInput(pcm: [0.0, 0.5], sampleRate: 16000);
    expect(audio.sampleRate, 16000);
    expect(audio.pcm.length, 2);

    const msg = UserMessage(text: 'hello', images: <Uint8List>[], audio: audio);
    expect(msg.text, 'hello');
    expect(msg.images, isEmpty);
    expect(msg.audio, isNotNull);
  });
}

void Function(FfiSileroVad) get _vadSurfaceGuard => (FfiSileroVad vad) {
  vad.processChunk(const <double>[], FfiVadSampleRate.rate16kHz);
  vad.getSpeechTimestamps(const <double>[], FfiVadSampleRate.rate16kHz, null);
  vad.reset();
};

void Function(FfiVadIterator, FfiSileroVad) get _vadIteratorSurfaceGuard => (
  FfiVadIterator it,
  FfiSileroVad vad,
) {
  it.processChunk(vad, const <double>[]);
  it.flush();
  it.reset();
};

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
    const <ChatMessage>[],
    const <ToolDef>[],
    true,
  );
  engine.transcribe(const <double>[], 16000);
  // Sequence returns.
  engine.encodeText('');
  engine.encodeTextSpecial('', false);
  // Optional-enum return.
  engine.toolFormat();
};

final class _TestModalitySink implements ModalitySink {
  @override
  void onThoughtChunk(String text) {}
  @override
  void onTextChunk(String text) {}
  @override
  void onAudioFrames(List<double> pcm, int sampleRate) {}
  @override
  void onDone(FinishReason reason) {}
}

/// The same guard for `Session`, which the RustBuffer regression hit just as
/// hard as `CeraEngine`. The `hiddenStates*` trio is the reason this matters
/// beyond "it throws": those three went from throwing to returning *corrupt*
/// data when `Vec<u8>`'s wire format was wrong in both directions, which a
/// stub-only check would have called fixed.
void Function(Session) get _sessionSurfaceGuard => (Session session) {
  // Record returns.
  session.capabilities();
  session.generate(const GenerateOpts());
  final sink = _TestModalitySink();
  session.generateStreaming(const GenerateOpts(), sink);
  session.generateStreamingAsync(const GenerateOpts(), sink);
  session.sendMessage(const UserMessage(text: '', images: [], audio: null));
  session.sendMessageAndGenerate(
    const UserMessage(text: '', images: [], audio: null),
    const GenerateOpts(),
  );
  session.sendMessageStreaming(
    const UserMessage(text: '', images: [], audio: null),
    const GenerateOpts(),
    sink,
  );
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
  repo.downloadBundle('', '');
};

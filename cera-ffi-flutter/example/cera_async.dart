// Async example: `generateAsync` returns a `Future` driven by UniFFI's
// rust-future poll/complete loop, so the Dart event loop stays responsive while
// Rust decodes on a worker.
//
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
//     dart run example/cera_async.dart /path/to/model.gguf "Once upon a time"
import 'dart:io' show exit;

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    print('usage: dart run example/cera_async.dart <model.gguf> [prompt]');
    return;
  }
  final modelPath = args[0];
  final prompt = args.length > 1 ? args[1] : 'Hello';

  print('cera ${ceraFfiVersion()} · ${cpuBackendReport()}');
  final engine = CeraEngine.fromPath(
    modelPath,
    const EngineConfig(
      contextSize: 2048,
      backend: BackendPreference.cpu,
      bundleRepo: null,
    ),
  );
  final session = engine.newSession(const SessionConfig(
    maxSeqLen: null,
    kvCompression: KvCompressionNone(),
    nKeep: 0,
    seed: null,
    ubatchSize: 512,
  ));
  session.appendText(prompt);

  // Prove the event loop stays live while Rust decodes: tick a timer.
  var ticks = 0;
  final ticker = Stream.periodic(const Duration(milliseconds: 200), (i) => i)
      .listen((_) => ticks++);

  print('awaiting generateAsync …');
  final out = await session.generateAsync(const GenerateOpts(
    maxTokens: 24,
    temperature: 0.0,
    topP: 1.0,
    topK: 0,
    repetitionPenalty: 1.0,
    stopTokens: <int>[],
    flushEveryTokens: 0,
    flushEveryMs: 0,
  ));
  await ticker.cancel();

  print('generateAsync done: ${out.tokens.length} tokens, '
      '${out.summary.decodeMs}ms, event-loop ticks during decode=$ticks');

  // generateStreamingAsync mixes async + a callback sink. cera runs the sink on
  // a tokio worker thread; the ModalitySink vtable uses NativeCallable.listener,
  // which delivers those cross-thread callbacks onto this isolate's event loop —
  // so the sink fires live, per token, while the Future is awaited.
  final sink = _AsyncSink();
  session.appendText(prompt);
  await session.generateStreamingAsync(
    const GenerateOpts(
      maxTokens: 16,
      temperature: 0.0,
      topP: 1.0,
      topK: 0,
      repetitionPenalty: 1.0,
      stopTokens: <int>[],
      flushEveryTokens: 1,
      flushEveryMs: 0,
    ),
    sink,
  );
  print('generateStreamingAsync done: ${sink.count} tokens, finish=${sink.finish}');
  exit(0);
}

class _AsyncSink implements ModalitySink {
  int count = 0;
  FinishReason? finish;
  @override
  void onTextTokens(List<int> t) => count += t.length;
  @override
  void onAudioFrames(List<double> pcm, int sampleRate) {}
  @override
  void onDone(FinishReason reason) => finish = reason;
}

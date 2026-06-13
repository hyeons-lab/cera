// Streaming example: implement a Dart [ModalitySink] and pass it to
// [Session.generateStreaming]. Rust calls back into Dart with each batch of
// generated tokens (and audio frames for audio models), then `onDone` once.
//
//   just dart-bindings
//   cd cera-ffi-flutter
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
//     dart run example/cera_stream.dart /path/to/model.gguf "Once upon a time"
import 'dart:io' show exit;

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

/// Collects streamed tokens as Rust pushes them across the FFI boundary.
class _CollectingSink implements ModalitySink {
  final List<int> tokens = <int>[];
  int batches = 0;
  FinishReason? finish;

  @override
  void onTextTokens(List<int> t) {
    batches++;
    tokens.addAll(t);
    print('  ← onTextTokens batch #$batches: ${t.length} token(s)');
  }

  @override
  void onAudioFrames(List<double> pcm, int sampleRate) {
    print('  ← onAudioFrames: ${pcm.length} samples @ ${sampleRate}Hz');
  }

  @override
  void onDone(FinishReason reason) {
    finish = reason;
    print('  ← onDone: $reason');
  }
}

void main(List<String> args) {
  if (args.isEmpty) {
    print('usage: dart run example/cera_stream.dart <model.gguf> [prompt]');
    return;
  }
  final modelPath = args[0];
  final prompt = args.length > 1 ? args[1] : 'Hello';

  print('cera ${ceraFfiVersion()} · ${cpuBackendReport()}');
  print('loading $modelPath …');

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

  final sink = _CollectingSink();
  print('streaming (generate_streaming → Dart ModalitySink):');
  final summary = session.generateStreaming(
    const GenerateOpts(
      maxTokens: 24,
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

  print('done: ${sink.tokens.length} tokens over ${sink.batches} callback batches, '
      'finish=${sink.finish}, decode=${summary.decodeMs}ms');

  // The callback vtable's static `NativeCallable.isolateLocal`s keep the isolate
  // alive, so a CLI script won't exit on its own. A real Flutter app stays
  // running anyway; here we exit explicitly.
  exit(0);
}

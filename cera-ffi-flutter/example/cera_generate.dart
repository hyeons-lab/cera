// End-to-end example: load a GGUF model through the Dart bindings and run a
// synchronous generation.
//
// Prerequisites:
//   1. Build the native lib:   just dart-libs
//      (equivalently `cargo build -p cera-ffi --features ffi-buffer` — the
//      `ffi-buffer` feature is required; a plain `cargo build -p cera-ffi`
//      omits the `uniffi_ffibuffer_*` trampolines and fails at runtime.)
//   2. Generate the bindings:  just dart-bindings
//   3. Point at the native lib via CERA_FFI_LIB (or place it on the loader path):
//
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
//     dart run example/cera_generate.dart /path/to/model.gguf "Once upon a time"
//
// The FFI surface returns token IDs (no detokenizer is exposed yet — see
// V2.17), so this prints the token count and decode timing rather than text.
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    print('usage: dart run example/cera_generate.dart <model.gguf> [prompt]');
    print('host CPU backend: ${cpuBackendReport()}');
    print('cera version:     ${ceraFfiVersion()}');
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
  final out = session.generate(const GenerateOpts(
    maxTokens: 32,
    temperature: 0.0,
    topP: 1.0,
    topK: 0,
    repetitionPenalty: 1.0,
    stopTokens: <int>[],
    flushEveryTokens: 0,
    flushEveryMs: 0,
  ));

  final s = out.summary;
  print('generated ${out.tokens.length} tokens in ${s.decodeMs} ms');
  print('token ids: ${out.tokens}');
}

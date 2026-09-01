// End-to-end chat example: render a chat template, generate, and decode the
// result back to text.
//
// This exercises the surface that used to throw `UnsupportedError` because the
// generator had no decoder for RustBuffer returns carrying a String, a
// sequence, or an optional: `applyChatTemplate`, `encodeText`,
// `encodeTextSpecial`, `decodeTokens`, and `toolFormat`.
//
// Prerequisites:
//   1. Build the native lib:   just dart-libs
//   2. Generate the bindings:  just dart-bindings
//
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
//     dart run example/cera_chat.dart /path/to/model.gguf "Why is the sky blue?"
import 'dart:io';

import 'package:cera_ffi/cera_ffi.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    print('usage: dart run example/cera_chat.dart <model.gguf> [prompt]');
    exit(64);
  }

  final modelPath = args[0];
  final prompt = args.length > 1 ? args[1] : 'Why is the sky blue?';

  print('cera ${ceraFfiVersion()} · ${cpuBackendReport()}');

  final engine = CeraEngine.fromPath(
    modelPath,
    const EngineConfig(
      contextSize: 2048,
      backend: BackendPreference.cpu,
      bundleRepo: null,
      draftModel: null,
    ),
  );

  // ── String return ────────────────────────────────────────────────────────
  final messages = [ChatMessage(role: 'user', content: prompt)];
  final String rendered =
      engine.hasChatTemplate()
          ? engine.applyChatTemplate(messages, true)
          : prompt;
  print(
    'chat template: ${engine.hasChatTemplate() ? "yes" : "none, raw prompt"}',
  );

  // ── Sequence return ──────────────────────────────────────────────────────
  final List<int> promptTokens = engine.encodeTextSpecial(rendered, true);
  print('prompt tokens: ${promptTokens.length}');

  // ── Optional-enum return ─────────────────────────────────────────────────
  print('tool format:   ${engine.toolFormat() ?? "none"}');

  final session = engine.newSession(
    const SessionConfig(
      maxSeqLen: null,
      kvCompression: KvCompressionNone(),
      nKeep: 0,
      seed: null,
      ubatchSize: 512,
    ),
  );

  session.appendTokens(promptTokens);
  final out = session.generate(
    const GenerateOpts(
      maxTokens: 64,
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
    ),
  );

  // ── String return, the one that made text output impossible ──────────────
  final String text = engine.decodeTokens(out.tokens);

  print('');
  print(text.trim());
  print('');
  print('${out.tokens.length} tokens in ${out.summary.decodeMs} ms');
}

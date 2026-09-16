import 'dart:io';

import 'package:cera_ffi/cera_ffi.dart';

// Multi-turn conversational chat with live KV cache retention in Dart.
//
// Demonstrates:
// 1. Loading a generative model and creating an execution session.
// 2. Converting the session into a transactional ChatSession.
// 3. Ingesting user turns and completing responses.
// 4. Retaining live KV context across consecutive turns without recomputation.
// 5. Reclaiming the underlying Session upon completion.
// Run (from cera_ffi package directory):
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib dart run example/chat.dart model.gguf
// Or (from repository root):
//   CERA_FFI_LIB=target/debug/libcera_ffi.dylib dart run cera_ffi/example/chat.dart model.gguf
void main(List<String> args) {
  if (args.isEmpty) {
    throw ArgumentError('usage: chat.dart <model.gguf>');
  }
  final modelPath = args[0];
  stdout.writeln('Loading model from: $modelPath');

  final loader = ModelLoader.create(
    ModelSourcePath(path: modelPath),
    const EngineConfig(backend: BackendPreference.cpu),
  );
  try {
    final model = loader.buildGenerative();
    try {
      final session = model.createSession(const SessionConfig(seed: 42));
      try {
        final chat = session.intoChat();
        try {
          stdout.writeln('Initial chat phase: ${chat.phase()}');
          stdout.writeln('Initial position: ${chat.position()}');

          const opts = GenerateOpts(maxTokens: 64, temperature: 0.7);

          // --- Turn 1: System and User messages ---
          stdout.writeln('\n--- Turn 1 ---');
          final turn1Messages = [
            chatMessageSystem(
              'You are a helpful and concise systems engineering assistant.',
            ),
            chatMessageUser(
              'What is a KV cache in LLM inference? Answer in one sentence.',
            ),
          ];

          final summary1 = chat.ingestMessages(turn1Messages);
          stdout.writeln(
            'Ingested ${summary1.inputTokens} tokens (position: ${summary1.positionBefore} -> ${summary1.positionAfter})',
          );
          stdout.writeln('Phase after ingest: ${chat.phase()}');

          final turn1 = chat.complete(opts);
          stdout.writeln('Assistant: ${turn1.text.trim()}');
          stdout.writeln(
            'Generated ${turn1.summary.tokensGenerated} tokens (final position: ${chat.position()})',
          );
          stdout.writeln('Phase after completion: ${chat.phase()}');
          assert(chat.phase() == SessionPhase.turnComplete);

          // --- Turn 2: Warm Continuation ---
          // The previous context remains in the KV cache; only new user input is ingested.
          stdout.writeln('\n--- Turn 2 (Warm Continuation) ---');
          final turn2User = chatMessageUser('When should it be discarded?');
          final summary2 = chat.ingest(turn2User);
          stdout.writeln(
            'Ingested ${summary2.inputTokens} new tokens (position: ${summary2.positionBefore} -> ${summary2.positionAfter})',
          );

          final turn2 = chat.complete(opts);
          stdout.writeln('Assistant: ${turn2.text.trim()}');
          stdout.writeln(
            'Generated ${turn2.summary.tokensGenerated} tokens (final position: ${chat.position()})',
          );
          stdout.writeln('Phase after completion: ${chat.phase()}');
          assert(chat.phase() == SessionPhase.turnComplete);

          // --- Reclaim raw Session ---
          final reclaimedSession = chat.intoSession();
          try {
            stdout.writeln(
              '\nReclaimed raw session at position ${reclaimedSession.position()}',
            );
          } finally {
            reclaimedSession.close();
          }
        } finally {
          chat.close();
        }
      } finally {
        session.close();
      }
    } finally {
      model.close();
    }
  } finally {
    loader.close();
  }
  // The generated bindings retain process-wide callback listeners.
  exit(0);
}

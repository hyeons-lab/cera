import 'dart:io';

import 'package:cera_ffi/cera_ffi.dart';

// Run with a generative GGUF path and a raw completion prompt.
void main(List<String> args) {
  if (args.length != 2) {
    throw ArgumentError('usage: explicit_loading.dart model.gguf prompt');
  }
  final loader = ModelLoader.create(
    ModelSourcePath(path: args[0]),
    const EngineConfig(backend: BackendPreference.cpu),
  );
  try {
    final model = loader.buildGenerative();
    try {
      final engine = model.engine();
      try {
        final session = model.createSession(const SessionConfig(seed: 42));
        try {
          session.appendTokens(engine.encodeText(args[1]));
          final output = session.generate(
            const GenerateOpts(maxTokens: 32, temperature: 0.7),
          );
          stdout.writeln(engine.decodeTokens(output.tokens));
        } finally {
          session.close();
        }
      } finally {
        engine.close();
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

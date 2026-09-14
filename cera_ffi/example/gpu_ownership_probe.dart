/// Runs against the tiny fixtures exported by the GPU ownership tests.
library;

import 'dart:convert';
import 'dart:io';

import 'package:cera_ffi/cera_ffi.dart';

void require(bool condition, String message) {
  if (!condition) throw StateError(message);
}

Future<void> main(List<String> args) async {
  try {
    await runProbe(args);
  } on Object catch (error, stack) {
    stderr.writeln(error);
    stderr.writeln(stack);
    exit(1);
  }
  // Generated callback vtables retain process-global Dart listeners. All model
  // handles are closed by runProbe before this standalone CLI exits.
  exit(0);
}

Future<void> runProbe(List<String> args) async {
  if (args.length != 1 || !(Platform.isMacOS || Platform.isIOS)) {
    throw ArgumentError('Pass the exported fixture directory on a Metal host.');
  }
  final primary = await File('${args.single}/primary.gguf').readAsBytes();
  final encoder = await File('${args.single}/encoder.gguf').readAsBytes();
  const options = CeraOptions(backend: CeraBackend.gpu, contextSize: 512);
  final opened = <Cera>[];
  Future<Cera> open() async {
    final model = await Cera.openBytes(
      primary,
      mmproj: encoder,
      inferenceType: 'llama.cpp/lfm2-audio-v1',
      options: options,
    );
    opened.add(model);
    return model;
  }

  Future<String> generate(Cera model, String prompt, {int? seed}) =>
      model.generate(prompt, maxTokens: 3, temperature: 0, seed: seed).join();

  try {
    final live = await open();
    final control = await open();
    final first = await generate(live, 'aba', seed: 42);
    require(first.isNotEmpty, 'seeded generation returned no text');
    require(
      first == await generate(control, 'aba', seed: 42),
      'seeded generation differs from an independent control',
    );
    final pcm = List<double>.generate(3200, (i) => (i % 32 - 16) / 80.0);
    final emptyConversation = await open();
    final expected = await emptyConversation.transcribe(pcm, sampleRate: 16000);
    require(expected.isNotEmpty, 'transcription returned no text');
    require(
      await live.transcribe(pcm, sampleRate: 16000) == expected,
      'transcription changed with conversation history',
    );
    var rejected = false;
    try {
      await live.transcribe([], sampleRate: 16000);
    } on Object {
      rejected = true;
    }
    require(rejected, 'empty PCM unexpectedly succeeded');
    require(
      await live.transcribe(pcm, sampleRate: 16000) == expected,
      'transcription did not recover after invalid PCM',
    );
    require(
      await generate(live, 'bab') == await generate(control, 'bab'),
      'conversation continuation changed after transcription',
    );
    await live.reset();
    require(
      await generate(live, 'aba', seed: 42) == first,
      'reseed after reset differs from the initial seeded generation',
    );
    print(
      jsonEncode({
        'passed': true,
        'cases': [
          'seeded generation',
          'empty-conversation transcription',
          'retained-conversation transcription',
          'transcription failure recovery',
          'conversation continuation',
          'reseed after reset',
        ],
      }),
    );
  } finally {
    for (final model in opened.reversed) {
      await model.close();
      await model.close();
    }
  }
}

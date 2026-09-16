import 'dart:async';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;
import 'package:cera_ffi_flutter_example/chat_controller.dart';
import 'package:cera_ffi_flutter_example/chat_intent.dart';
import 'package:cera_ffi_flutter_example/model_source.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(const MethodChannel('cera/audio_player'), (
          call,
        ) async {
          return;
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('cera/audio_player'),
          null,
        );
  });

  group('ChatController transactional chat coordinator', () {
    test('initializes in idle phase', () {
      final controller = ChatController();
      expect(controller.sessionPhase, SessionPhase.idle);
      controller.dispose();
    });

    test(
      'coordinates Turn 1 initial, Turn 2 continuation, interruption and reset',
      () async {
        final fakeCera = FakeCera();
        final modelSource = TestModelSource('test-model', fakeCera);
        final controller = ChatController();

        // Load model
        await controller.dispatch(LoadLocalModelIntent(modelSource));
        expect(controller.sessionPhase, SessionPhase.idle);
        expect(controller.value.loadedModel?.name, 'test-model');

        // Turn 1: Initial message
        await controller.dispatch(const SendMessageIntent('Hello'));
        expect(controller.sessionPhase, SessionPhase.turnComplete);
        expect(fakeCera.templateCalls.length, 1);
        final turn1Messages = fakeCera.templateCalls[0];
        expect(turn1Messages.any((m) => m.role == 'system'), isTrue);
        expect(
          turn1Messages.any((m) => m.role == 'user' && m.content == 'Hello'),
          isTrue,
        );
        expect(fakeCera.generatedPrompts.length, 1);
        expect(fakeCera.generatedPrompts[0], contains('<|im_start|>system'));
        expect(
          fakeCera.generatedPrompts[0],
          contains('<|im_start|>user\nHello<|im_end|>'),
        );

        // Turn 2: Continuation message
        await controller.dispatch(const SendMessageIntent('How are you?'));
        expect(controller.sessionPhase, SessionPhase.turnComplete);
        expect(fakeCera.templateCalls.length, 2);
        final turn2Messages = fakeCera.templateCalls[1];
        // System prompt must NOT be re-injected on continuation turns
        expect(turn2Messages.any((m) => m.role == 'system'), isFalse);
        expect(turn2Messages.length, 1);
        expect(turn2Messages[0].role, 'user');
        expect(turn2Messages[0].content, 'How are you?');
        expect(fakeCera.generatedPrompts.length, 2);
        // Continuation prompt must prepend <|im_end|>\n to seal the prior assistant turn
        expect(
          fakeCera.generatedPrompts[1].startsWith(
            '<|im_end|>\n<|im_start|>user\nHow are you?',
          ),
          isTrue,
        );

        // Turn 3: Interrupted turn
        fakeCera.pauseGeneration = true;
        final sendFuture = controller.dispatch(
          const SendMessageIntent('Tell me a story'),
        );
        // Wait for generation to start
        await Future<void>.delayed(const Duration(milliseconds: 20));
        expect(controller.value.isGenerating, isTrue);

        // Stop generation
        await controller.dispatch(const StopGenerationIntent());
        expect(controller.sessionPhase, SessionPhase.interrupted);
        fakeCera.resume();
        await sendFuture;

        // Turn 4: Continuation after interruption
        fakeCera.pauseGeneration = false;
        await controller.dispatch(const SendMessageIntent('Continue'));
        expect(controller.sessionPhase, SessionPhase.turnComplete);
        expect(fakeCera.generatedPrompts.length, 4);
        // Interrupted continuation must prepend <|im_end|>\n to close the uncommitted assistant turn
        expect(
          fakeCera.generatedPrompts[3].startsWith(
            '<|im_end|>\n<|im_start|>user\nContinue',
          ),
          isTrue,
        );

        // Clear transcript: in-place reset back to idle
        await controller.dispatch(const ClearTranscriptIntent());
        expect(controller.sessionPhase, SessionPhase.idle);
        expect(fakeCera.resetCount, 1);
        expect(controller.value.turns, isEmpty);

        // Turn 5: Post-clear initial message
        await controller.dispatch(const SendMessageIntent('Fresh start'));
        expect(controller.sessionPhase, SessionPhase.turnComplete);
        expect(fakeCera.templateCalls.length, 5);
        final turn5Messages = fakeCera.templateCalls[4];
        // System prompt must be re-injected on fresh turn 1
        expect(turn5Messages.any((m) => m.role == 'system'), isTrue);
        expect(
          turn5Messages.any(
            (m) => m.role == 'user' && m.content == 'Fresh start',
          ),
          isTrue,
        );

        controller.dispose();
      },
    );

    test(
      'marks phase unusable on reset failure, preserves turns, and refuses further dispatches',
      () async {
        final fakeCera = FakeCera()..failReset = true;
        final modelSource = TestModelSource('test-model', fakeCera);
        final controller = ChatController();

        await controller.dispatch(LoadLocalModelIntent(modelSource));
        expect(controller.sessionPhase, SessionPhase.idle);

        // Send a message so turns is non-empty
        await controller.dispatch(const SendMessageIntent('Preserve me'));
        expect(controller.value.turns, isNotEmpty);

        // Attempt clear transcript with failing reset
        await controller.dispatch(const ClearTranscriptIntent());
        expect(controller.sessionPhase, SessionPhase.unusable);
        expect(controller.value.status, contains('Engine reset failed'));
        // Turns must NOT be cleared if reset failed
        expect(controller.value.turns, isNotEmpty);

        // Dispatch while unusable should be refused
        await controller.dispatch(const SendMessageIntent('Blocked prompt'));
        expect(fakeCera.generatedPrompts.length, 1);

        controller.dispose();
      },
    );

    test(
      'trims leading newline after stripping BOS marker in continuation turns',
      () async {
        final fakeCera = FakeCera()..prefixWithNewline = true;
        final modelSource = TestModelSource('test-model', fakeCera);
        final controller = ChatController();

        await controller.dispatch(LoadLocalModelIntent(modelSource));
        await controller.dispatch(const SendMessageIntent('Turn 1'));
        await controller.dispatch(const SendMessageIntent('Turn 2'));

        // Continuation prompt should be '<|im_end|>\n<|im_start|>user...' not '<|im_end|>\n\n<|im_start|>user...'
        expect(
          fakeCera.generatedPrompts[1].startsWith(
            '<|im_end|>\n<|im_start|>user',
          ),
          isTrue,
        );
        expect(
          fakeCera.generatedPrompts[1].startsWith(
            '<|im_end|>\n\n<|im_start|>user',
          ),
          isFalse,
        );

        controller.dispose();
      },
    );
  });
}

class TestModelSource extends LoadedModel {
  TestModelSource(this.name, this.engine);

  @override
  final String name;
  final Cera engine;

  @override
  Future<Cera> open({CeraOptions options = const CeraOptions()}) async =>
      engine;
}

class FakeCera implements Cera {
  final List<List<CeraMessage>> templateCalls = [];
  final List<String> generatedPrompts = [];
  int resetCount = 0;
  bool failReset = false;
  bool pauseGeneration = false;
  bool prefixWithNewline = false;
  Completer<void>? _pauseCompleter;

  void resume() {
    _pauseCompleter?.complete();
    _pauseCompleter = null;
  }

  @override
  CeraCapabilities get capabilities => const CeraCapabilities(
    textIn: true,
    textOut: true,
    imageIn: false,
    audioIn: true,
    audioOut: true,
  );

  @override
  String get backend => 'gpu';

  @override
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  }) async {
    templateCalls.add(messages);
    final buf = StringBuffer(
      prefixWithNewline ? '<|startoftext|>\n' : '<|startoftext|>',
    );
    for (final m in messages) {
      buf.write('<|im_start|>${m.role}\n${m.content}<|im_end|>\n');
    }
    if (addGenerationPrompt) {
      buf.write('<|im_start|>assistant\n');
    }
    return buf.toString();
  }

  @override
  Stream<String> generate(
    String prompt, {
    int? maxTokens,
    double? temperature,
    double? topP,
    int? topK,
    int? seed,
    CeraSpecDecode? spec,
    void Function(String thought)? onThought,
    void Function(List<double> pcm, int sampleRate)? onAudio,
  }) async* {
    generatedPrompts.add(prompt);
    yield 'Hello ';
    if (pauseGeneration) {
      _pauseCompleter = Completer<void>();
      await _pauseCompleter!.future;
    }
    yield 'world!';
  }

  @override
  Future<void> reset() async {
    if (failReset) {
      throw StateError('Simulated reset backend failure');
    }
    resetCount++;
  }

  @override
  Future<void> cancel() async {
    resume();
  }

  @override
  Future<List<int>> encode(String text, {bool addSpecial = true}) async {
    return List.filled(text.split(' ').length, 42);
  }

  @override
  Future<String> decode(List<int> tokens) async => 'decoded';

  @override
  Future<void> appendImage(Uint8List bytes, {int? maxLongSize}) async {}

  @override
  Future<void> appendAudio(
    List<double> pcm, {
    int sampleRate = 16000,
    String? prompt,
    String? systemPrompt,
  }) async {}

  @override
  Future<String> transcribe(
    List<double> pcm, {
    required int sampleRate,
  }) async => '';

  @override
  Future<void> close() async {}

  @override
  Future<void> terminate() async {}
}

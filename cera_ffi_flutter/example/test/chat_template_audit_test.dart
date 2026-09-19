import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;
import 'package:cera_ffi_flutter_example/chat_controller.dart';
import 'package:cera_ffi_flutter_example/chat_intent.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'chat_coordinator_test.dart' show FakeCera, TestModelSource;

class LlamaCera extends FakeCera {
  @override
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  }) async {
    final out = StringBuffer('<|begin_of_text|>');
    for (final message in messages) {
      out.write(
        '<|start_header_id|>${message.role}<|end_header_id|>\n\n${message.content}<|eot_id|>',
      );
    }
    if (addGenerationPrompt) {
      out.write('<|start_header_id|>assistant<|end_header_id|>\n\n');
    }
    return out.toString();
  }
}

class GemmaCera extends FakeCera {
  @override
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  }) async {
    final out = StringBuffer('<bos>');
    for (final message in messages) {
      out.write(
        '<start_of_turn>${message.role}\n${message.content}<end_of_turn>\n',
      );
    }
    if (addGenerationPrompt) out.write('<start_of_turn>model\n');
    return out.toString();
  }
}

class UnknownTemplateCera extends FakeCera {
  @override
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  }) async => messages.map((message) => message.content).join('\n');
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Llama continuation uses the model turn delimiter', () async {
    SharedPreferences.setMockInitialValues({});
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('cera/audio_player'),
          (_) async => null,
        );
    final engine = LlamaCera();
    final controller = ChatController();
    await controller.dispatch(
      LoadLocalModelIntent(TestModelSource('llama', engine)),
    );
    await controller.dispatch(const SendMessageIntent('Hello'));
    await controller.dispatch(const SendMessageIntent('Continue'));
    final actual = engine.generatedPrompts[1];
    controller.dispose();
    expect(actual, startsWith('<|eot_id|>'));
    expect(actual, isNot(contains('<|im_end|>')));
  });
  test('Gemma continuation seals the turn and omits BOS', () async {
    SharedPreferences.setMockInitialValues({});
    final engine = GemmaCera();
    final controller = ChatController();
    await controller.dispatch(
      LoadLocalModelIntent(TestModelSource('gemma', engine)),
    );
    await controller.dispatch(const SendMessageIntent('Hello'));
    await controller.dispatch(const SendMessageIntent('Continue'));
    expect(
      engine.generatedPrompts[1],
      startsWith('<end_of_turn>\n<start_of_turn>user\n'),
    );
    expect(engine.generatedPrompts[1], isNot(contains('<bos>')));
    expect(engine.generatedPrompts[1], isNot(contains('<|im_end|>')));
    controller.dispose();
  });
  test('unknown continuation leaves native state untouched', () async {
    SharedPreferences.setMockInitialValues({});
    final engine = UnknownTemplateCera();
    final controller = ChatController();
    await controller.dispatch(
      LoadLocalModelIntent(TestModelSource('unknown', engine)),
    );
    await controller.dispatch(const SendMessageIntent('Hello'));
    await controller.dispatch(const SendMessageIntent('Continue'));
    expect(engine.generatedPrompts, hasLength(1));
    expect(engine.resetCount, 0);
    expect(controller.value.isGenerating, isFalse);
    expect(controller.value.turns.last.text, contains('Start a new chat'));
    controller.dispose();
  });
}

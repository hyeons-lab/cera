// Smoke test for the example app.
//
// Deliberately does not load a model: that needs a real .gguf on disk and takes
// seconds. This only asserts the app builds and reaches its empty state, which
// is enough to catch a broken widget tree in CI.

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart'
    show CeraAudioProfile, CeraTtsVoice;
import 'package:cera_ffi_flutter_example/chat_controller.dart';
import 'package:cera_ffi_flutter_example/chat_state.dart';
import 'package:cera_ffi_flutter_example/main.dart';
import 'package:cera_ffi_flutter_example/model_source.dart';
import 'package:cera_ffi_flutter_example/widgets/audio_waveform.dart';
import 'package:cera_ffi_flutter_example/widgets/bundle_picker_dialog.dart';
import 'package:cera_ffi_flutter_example/widgets/message_list.dart';
import 'package:cera_ffi_flutter_example/widgets/tts_studio_view.dart';
import 'package:cera_ffi_flutter_example/widgets/voice_persona_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  testWidgets('renders the empty state before a model is picked', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const CeraExampleApp());

    expect(find.text('Cera'), findsOneWidget);
    expect(
      find.text('Download a published model, or open a .gguf, to start.'),
      findsOneWidget,
    );
  });

  testWidgets('vision and audio buttons are hidden before a model is loaded', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const CeraExampleApp());

    expect(find.byIcon(Icons.add_photo_alternate_outlined), findsNothing);
    expect(find.byIcon(Icons.mic_none_rounded), findsNothing);
  });

  testWidgets(
    'message list displays the specific model name badge for each assistant response',
    (WidgetTester tester) async {
      final turns = [
        Turn(role: 'user', text: 'Hello model 1'),
        Turn(
          role: 'assistant',
          text: 'Response from model 1',
          modelName: 'LFM2-700M · Q4_0',
          stats: const TurnStats(
            tokens: 15,
            totalMs: 300,
            ttftMs: 50,
            tps: 50.0,
          ),
        ),
        Turn(role: 'user', text: 'Hello model 2'),
        Turn(
          role: 'assistant',
          text: 'Response from model 2',
          modelName: 'Gemma-2-2B · Q4_K_M',
          stats: const TurnStats(
            tokens: 20,
            totalMs: 400,
            ttftMs: 40,
            tps: 55.0,
          ),
        ),
      ];

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: MessageList(
              turns: turns,
              scrollController: ScrollController(),
            ),
          ),
        ),
      );

      expect(find.text('Response from model 1'), findsOneWidget);
      expect(find.text('LFM2-700M · Q4_0'), findsOneWidget);

      expect(find.text('Response from model 2'), findsOneWidget);
      expect(find.text('Gemma-2-2B · Q4_K_M'), findsOneWidget);
    },
  );

  testWidgets('message list displays voice note badge for audio prompt turns', (
    WidgetTester tester,
  ) async {
    final turns = [
      Turn(
        role: 'user',
        text: 'What is the weather?',
        audioDurationSeconds: 3.5,
      ),
      Turn(
        role: 'assistant',
        text: 'It is sunny today.',
        modelName: 'LFM2.5-Audio-1.5B · Q4_0',
        stats: const TurnStats(tokens: 10, totalMs: 200, ttftMs: 30, tps: 50.0),
      ),
    ];

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: MessageList(turns: turns, scrollController: ScrollController()),
        ),
      ),
    );

    expect(find.byType(AudioWaveformBubble), findsOneWidget);
    expect(find.text('3.5s'), findsOneWidget);
    expect(find.text('What is the weather?'), findsOneWidget);
    expect(find.text('It is sunny today.'), findsOneWidget);
    expect(find.text('LFM2.5-Audio-1.5B · Q4_0'), findsOneWidget);
  });

  testWidgets(
    'bundle picker dialog renders catalog with DSpark quant choices',
    (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: BundlePickerDialog(
              currentBundleName: 'LFM2.5-1.2B-Instruct-GGUF',
              currentQuant: 'Q4_K_M + DSpark',
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Select Model'), findsOneWidget);
      expect(find.text('Catalog & Download'), findsOneWidget);

      // Switch to Catalog & Download tab
      await tester.tap(find.text('Catalog & Download'));
      await tester.pumpAndSettle();

      // Verify bundle list renders with DSpark sidecar indicators
      expect(find.text('LFM2.5-1.2B-Instruct'), findsWidgets);
      expect(find.textContaining('DSpark'), findsWidgets);
    },
  );

  testWidgets(
    'tts studio view initializes with clean default text without quant suffix',
    (WidgetTester tester) async {
      const bundle = BundleModelSource(
        name: 'LFM2.5-Audio-1.5B · Q4_0',
        bundleName: 'LFM2.5-Audio-1.5B-GGUF',
        quant: 'Q4_0',
        displayName: 'LFM2.5-Audio-1.5B',
      );
      final state = const ChatState().copyWith(loadedModel: () => bundle);
      final controller = ChatController();

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: TtsStudioView(
              state: state,
              controller: controller,
              onOpenCatalog: () {},
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(
        find.text(
          'Hello, this voice was synthesized entirely on-device with the LFM2.5-Audio-1.5B model powered by Cera.',
        ),
        findsOneWidget,
      );
      expect(find.textContaining('· Q4_0'), findsNothing);
    },
  );

  const englishProfile = CeraAudioProfile(
    ttsSystemPrompt: 'Perform TTS. Use the US female voice.',
    interleavedSystemPrompt:
        'Respond with interleaved text and audio. Use the US female voice.',
    voices: [
      CeraTtsVoice(
        label: 'Narrator',
        prompt: 'Use the US female voice.',
        ttsSystemPrompt: 'Perform TTS. Use the US female voice.',
        interleavedSystemPrompt:
            'Respond with interleaved text and audio. Use the US female voice.',
      ),
      CeraTtsVoice(
        label: 'Studio Warm',
        prompt: 'Use the UK male voice.',
        ttsSystemPrompt: 'Perform TTS. Use the UK male voice.',
        interleavedSystemPrompt:
            'Respond with interleaved text and audio. Use the UK male voice.',
      ),
    ],
  );
  const japaneseProfile = CeraAudioProfile(
    ttsSystemPrompt: 'Perform TTS in japanese.',
    interleavedSystemPrompt: 'Respond with interleaved text and audio.',
    sampleTexts: ['こんにちは、このデバイス上で{model}モデルを使って音声を合成しています。'],
  );
  const plainProfile = CeraAudioProfile(
    ttsSystemPrompt: 'Perform TTS.',
    interleavedSystemPrompt: 'Respond with interleaved text and audio.',
  );

  Future<void> pumpStudio(
    WidgetTester tester,
    BundleModelSource bundle,
    CeraAudioProfile profile, {
    ChatController? controller,
    ChatSettings settings = const ChatSettings(),
  }) async {
    final state = const ChatState().copyWith(
      loadedModel: () => bundle,
      audioProfile: () => profile,
      settings: settings,
    );
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: TtsStudioView(
            state: state,
            controller: controller ?? ChatController(),
            onOpenCatalog: () {},
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
  }

  testWidgets('tts studio lists the voices the model profile offers', (
    WidgetTester tester,
  ) async {
    await pumpStudio(
      tester,
      const BundleModelSource(
        name: 'LFM2.5-Audio-1.5B · Q4_0',
        bundleName: 'LFM2.5-Audio-1.5B-GGUF',
        quant: 'Q4_0',
        displayName: 'LFM2.5-Audio-1.5B',
      ),
      englishProfile,
    );
    expect(find.text('VOICE PERSONA'), findsOneWidget);
    expect(find.text('Narrator'), findsOneWidget);
    expect(find.text('Studio Warm'), findsOneWidget);
  });

  testWidgets(
    'tapping a studio voice saves the voice\'s prompt, not its label',
    (WidgetTester tester) async {
      final controller = ChatController();
      await pumpStudio(
        tester,
        const BundleModelSource(
          name: 'LFM2.5-Audio-1.5B · Q4_0',
          bundleName: 'LFM2.5-Audio-1.5B-GGUF',
          quant: 'Q4_0',
          displayName: 'LFM2.5-Audio-1.5B',
        ),
        englishProfile,
        controller: controller,
      );
      await tester.tap(find.text('Studio Warm'));
      await tester.pumpAndSettle();
      expect(
        controller.value.settings.ttsStudioVoice,
        'Use the UK male voice.',
      );
    },
  );

  bool chipSelected(WidgetTester tester, String label) => tester
      .widget<ChoiceChip>(find.widgetWithText(ChoiceChip, label))
      .selected;

  testWidgets(
    'the studio highlights the saved voice, else the model\'s first',
    (WidgetTester tester) async {
      const bundle = BundleModelSource(
        name: 'LFM2.5-Audio-1.5B · Q4_0',
        bundleName: 'LFM2.5-Audio-1.5B-GGUF',
        quant: 'Q4_0',
        displayName: 'LFM2.5-Audio-1.5B',
      );
      // Nothing saved, and a voice saved under another model: the model's
      // first voice is the one that will be sent, so it reads as selected.
      for (final saved in ['', 'Use the Martian voice.']) {
        await pumpStudio(
          tester,
          bundle,
          englishProfile,
          settings: ChatSettings(ttsStudioVoice: saved),
        );
        expect(chipSelected(tester, 'Narrator'), isTrue, reason: saved);
        expect(chipSelected(tester, 'Studio Warm'), isFalse, reason: saved);
      }
      await pumpStudio(
        tester,
        bundle,
        englishProfile,
        settings: const ChatSettings(ttsStudioVoice: 'Use the UK male voice.'),
      );
      expect(chipSelected(tester, 'Narrator'), isFalse);
      expect(chipSelected(tester, 'Studio Warm'), isTrue);
    },
  );

  testWidgets(
    'the studio text follows a model switch unless the user edited it',
    (WidgetTester tester) async {
      const english = BundleModelSource(
        name: 'LFM2.5-Audio-1.5B · Q4_0',
        bundleName: 'LFM2.5-Audio-1.5B-GGUF',
        quant: 'Q4_0',
        displayName: 'LFM2.5-Audio-1.5B',
      );
      const japanese = BundleModelSource(
        name: 'LFM2.5-Audio-1.5B-JP · Q4_0',
        bundleName: 'LFM2.5-Audio-1.5B-JP-GGUF',
        quant: 'Q4_0',
        displayName: 'LFM2.5-Audio-1.5B-JP',
      );
      const jpSample = 'こんにちは、このデバイス上でLFM2.5-Audio-1.5B-JPモデルを使って音声を合成しています。';
      final controller = ChatController();
      Future<void> show(BundleModelSource b, CeraAudioProfile p) async {
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: TtsStudioView(
                state: const ChatState().copyWith(
                  loadedModel: () => b,
                  audioProfile: () => p,
                ),
                controller: controller,
                onOpenCatalog: () {},
              ),
            ),
          ),
        );
        await tester.pumpAndSettle();
      }

      String text() => tester
          .widget<TextField>(find.byType(TextField).first)
          .controller!
          .text;

      await show(english, englishProfile);
      expect(text(), startsWith('Hello, this voice was synthesized'));
      // An untouched default follows the model: speaking English text with the
      // Japanese model would not be what its profile is for.
      await show(japanese, japaneseProfile);
      expect(text(), jpSample);

      // Typed text is the user's, and survives another switch.
      await tester.enterText(find.byType(TextField).first, 'my own words');
      await show(english, englishProfile);
      expect(text(), 'my own words');
    },
  );

  Future<void> pumpPicker(
    WidgetTester tester, {
    required CeraAudioProfile? profile,
    String saved = '',
    ValueChanged<String?>? onChanged,
  }) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: VoicePersonaPicker(
            profile: profile,
            saved: saved,
            onChanged: onChanged ?? (_) {},
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
  }

  testWidgets('the voice persona picker is absent for a model without voices', (
    WidgetTester tester,
  ) async {
    await pumpPicker(tester, profile: japaneseProfile);
    expect(find.text('Voice Persona'), findsNothing);
    await pumpPicker(tester, profile: null);
    expect(find.text('Voice Persona'), findsNothing);
  });

  testWidgets(
    'a voice saved under another model reads as the default voice, without throwing',
    (WidgetTester tester) async {
      await pumpPicker(
        tester,
        profile: englishProfile,
        saved: 'Use the Martian voice.',
      );
      expect(find.text('Voice Persona'), findsOneWidget);
      expect(find.text('Default Voice'), findsOneWidget);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets(
    'picking a persona reports its prompt, and Default reports null',
    (WidgetTester tester) async {
      final picked = <String?>[];
      await pumpPicker(
        tester,
        profile: englishProfile,
        saved: 'Use the US female voice.',
        onChanged: picked.add,
      );
      expect(
        find.text('Narrator'),
        findsOneWidget,
        reason: 'the saved voice shows',
      );

      await tester.tap(find.byType(DropdownButton<String?>));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Studio Warm').last);
      await tester.pumpAndSettle();
      expect(picked, ['Use the UK male voice.']);

      await tester.tap(find.byType(DropdownButton<String?>));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Default Voice').last);
      await tester.pumpAndSettle();
      expect(picked, ['Use the UK male voice.', null]);
    },
  );

  testWidgets(
    'tts studio for a model whose profile has no voices hides the picker and uses its sample text',
    (WidgetTester tester) async {
      await pumpStudio(
        tester,
        const BundleModelSource(
          name: 'LFM2.5-Audio-1.5B-JP · Q4_0',
          bundleName: 'LFM2.5-Audio-1.5B-JP-GGUF',
          quant: 'Q4_0',
          displayName: 'LFM2.5-Audio-1.5B-JP',
        ),
        japaneseProfile,
      );
      // A voice the model was not trained on makes it answer in text and never
      // speak, so a model without voices offers no picker.
      expect(find.text('VOICE PERSONA'), findsNothing);
      expect(find.text('Narrator'), findsNothing);
      expect(find.text('👩 US Female'), findsNothing);
      expect(
        find.text('こんにちは、このデバイス上でLFM2.5-Audio-1.5B-JPモデルを使って音声を合成しています。'),
        findsOneWidget,
      );
    },
  );

  testWidgets(
    'tts studio falls back to the generic English text when the profile has none',
    (WidgetTester tester) async {
      await pumpStudio(
        tester,
        const BundleModelSource(
          name: 'SomeNewAudioModel-2B · Q4_0',
          bundleName: 'SomeNewAudioModel-2B-GGUF',
          quant: 'Q4_0',
          displayName: 'SomeNewAudioModel-2B',
        ),
        plainProfile,
      );
      expect(find.text('VOICE PERSONA'), findsNothing);
      expect(
        find.text(
          'Hello, this voice was synthesized entirely on-device with the SomeNewAudioModel-2B model powered by Cera.',
        ),
        findsOneWidget,
      );
    },
  );
}

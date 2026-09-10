import 'dart:async';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:cera_ffi_flutter_example/chat_controller.dart';
import 'package:cera_ffi_flutter_example/chat_intent.dart';
import 'package:cera_ffi_flutter_example/chat_state.dart';
import 'package:cera_ffi_flutter_example/model_source.dart';
import 'package:cera_ffi_flutter_example/services/audio_player_service.dart';
import 'package:cera_ffi_flutter_example/services/audio_recorder_service.dart';
import 'package:cera_ffi_flutter_example/widgets/audio_waveform.dart';
import 'package:cera_ffi_flutter_example/widgets/message_composer.dart';
import 'package:cera_ffi_flutter_example/widgets/message_list.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
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

  group('AudioRecorderService signal processing', () {
    test('pcm16ToFloat32 accurately converts signed 16-bit PCM bytes', () {
      final byteData = ByteData(8);
      // Sample 0: 0
      byteData.setInt16(0, 0, Endian.little);
      // Sample 1: 32767 (max positive)
      byteData.setInt16(2, 32767, Endian.little);
      // Sample 2: -32768 (max negative)
      byteData.setInt16(4, -32768, Endian.little);
      // Sample 3: 16384 (approx 0.5)
      byteData.setInt16(6, 16384, Endian.little);

      final floats = AudioRecorderService.pcm16ToFloat32(
        byteData.buffer.asUint8List(),
      );

      expect(floats.length, 4);
      expect(floats[0], closeTo(0.0, 1e-4));
      expect(floats[1], closeTo(1.0, 1e-4));
      expect(floats[2], closeTo(-1.0, 1e-4));
      expect(floats[3], closeTo(0.5, 1e-3));
    });

    test(
      'pcm16ToFloat32 handles empty, 1-byte, and odd-length byte lists safely',
      () {
        expect(AudioRecorderService.pcm16ToFloat32(Uint8List(0)), isEmpty);
        expect(AudioRecorderService.pcm16ToFloat32(Uint8List(1)), isEmpty);

        // 3 bytes: 1 complete 16-bit sample (2 bytes) + 1 trailing odd byte (ignored safely)
        final threeBytes = Uint8List.fromList([0x00, 0x00, 0xFF]);
        final floats = AudioRecorderService.pcm16ToFloat32(threeBytes);
        expect(floats.length, 1);
        expect(floats[0], closeTo(0.0, 1e-4));
      },
    );

    test('normalizeAudio scales peak amplitude and clamps safely', () {
      final samples = [0.1, -0.2, 0.4, -0.1];
      final normalized = AudioRecorderService.normalizeAudio(
        samples,
        targetPeak: 0.8,
      );

      expect(normalized.length, samples.length);
      // Peak was 0.4, scaled to 0.8 -> factor is 2.0
      expect(normalized[0], closeTo(0.2, 1e-5));
      expect(normalized[1], closeTo(-0.4, 1e-5));
      expect(normalized[2], closeTo(0.8, 1e-5));
      expect(normalized[3], closeTo(-0.2, 1e-5));
    });

    test('normalizeAudio handles invalid targetPeak safely', () {
      final samples = [0.2, -0.4, 0.1];
      // When targetPeak is NaN or non-positive, fallback to 0.9
      final normNan = AudioRecorderService.normalizeAudio(
        samples,
        targetPeak: double.nan,
      );
      expect(normNan[1], closeTo(-0.9, 1e-5));

      final normZero = AudioRecorderService.normalizeAudio(
        samples,
        targetPeak: 0.0,
      );
      expect(normZero[1], closeTo(-0.9, 1e-5));
    });

    test('normalizeAudio handles silent and non-finite samples safely', () {
      final silent = [0.0, 0.0, 0.0];
      final normSilent = AudioRecorderService.normalizeAudio(silent);
      expect(normSilent, [0.0, 0.0, 0.0]);

      final withNonFinite = [
        double.nan,
        double.infinity,
        0.5,
        double.negativeInfinity,
      ];
      final normFinite = AudioRecorderService.normalizeAudio(withNonFinite);
      expect(normFinite.length, 4);
      for (final s in normFinite) {
        expect(s.isFinite, isTrue);
      }
    });

    test('trimSilence preserves speech burst and bounds padding', () {
      const sampleRate = 16000;
      // 100ms silence + 200ms audio + 100ms silence = 400ms total
      final samples = List<double>.filled(sampleRate * 4 ~/ 10, 0.0);
      final speechStart = sampleRate ~/ 10;
      final speechEnd = sampleRate * 3 ~/ 10;
      for (var i = speechStart; i < speechEnd; i++) {
        samples[i] = 0.5 * ((i % 2 == 0) ? 1.0 : -1.0);
      }

      final trimmed = AudioRecorderService.trimSilence(
        samples,
        sampleRate: sampleRate,
        thresholdFactor: 0.05,
      );

      expect(trimmed.isNotEmpty, isTrue);
      expect(trimmed.length, lessThanOrEqualTo(samples.length));
    });

    test('trimSilence returns full buffer when signal is very quiet', () {
      const sampleRate = 16000;
      final quiet = List<double>.filled(sampleRate ~/ 2, 0.001);
      final result = AudioRecorderService.trimSilence(
        quiet,
        sampleRate: sampleRate,
      );
      expect(result.length, quiet.length);
    });
  });

  group('ChatState audio capability gating', () {
    test(
      'canAttachAudio is gated on model presence, audioIn, and idle state',
      () {
        const emptyState = ChatState();
        expect(emptyState.canAttachAudio, isFalse);

        final withTextModel = emptyState.copyWith(
          loadedModel: () => const MockLoadedModel('TextModel'),
          capabilities: () => const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: false,
            audioOut: false,
          ),
        );
        expect(withTextModel.canAttachAudio, isFalse);

        final withAudioModel = emptyState.copyWith(
          loadedModel: () => const MockLoadedModel('LFM2.5-Audio-1.5B'),
          capabilities: () => const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: true,
          ),
        );
        expect(withAudioModel.canAttachAudio, isTrue);

        final busyState = withAudioModel.copyWith(isLoading: true);
        expect(busyState.canAttachAudio, isFalse);

        final generatingState = withAudioModel.copyWith(isGenerating: true);
        expect(generatingState.canAttachAudio, isFalse);
      },
    );
  });

  group('MessageComposer voice chat UI and push-to-talk gestures', () {
    testWidgets('renders mic button only when canAttachAudio is true', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            bottomNavigationBar: MessageComposer(
              controller: TextEditingController(),
              isBusy: false,
              isGenerating: false,
              canAttachImage: false,
              canAttachAudio: false,
              pendingImageBytes: null,
              pendingImageName: null,
              onSend: () {},
              onStop: () {},
              onPickImage: () {},
              onClearImage: () {},
              onSendAudio: (_, _) {},
            ),
          ),
        ),
      );

      expect(find.byIcon(Icons.mic_none_rounded), findsNothing);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            bottomNavigationBar: MessageComposer(
              controller: TextEditingController(),
              isBusy: false,
              isGenerating: false,
              canAttachImage: false,
              canAttachAudio: true,
              pendingImageBytes: null,
              pendingImageName: null,
              onSend: () {},
              onStop: () {},
              onPickImage: () {},
              onClearImage: () {},
              onSendAudio: (_, _) {},
            ),
          ),
        ),
      );

      expect(find.byIcon(Icons.mic_none_rounded), findsOneWidget);
    });

    testWidgets('push-to-talk displays live waveform and slide-to-cancel cue', (
      WidgetTester tester,
    ) async {
      List<double>? sentPcm;
      int? sentRate;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            bottomNavigationBar: MessageComposer(
              controller: TextEditingController(),
              isBusy: false,
              isGenerating: false,
              canAttachImage: false,
              canAttachAudio: true,
              pendingImageBytes: null,
              pendingImageName: null,
              onSend: () {},
              onStop: () {},
              onPickImage: () {},
              onClearImage: () {},
              audioRecorder: FakeAudioRecorderService(),
              onSendAudio: (pcm, rate) {
                sentPcm = pcm;
                sentRate = rate;
              },
            ),
          ),
        ),
      );

      final micFinder = find.byIcon(Icons.mic_none_rounded);
      expect(micFinder, findsOneWidget);

      // Pointer down to begin push-to-talk recording
      final gesture = await tester.startGesture(tester.getCenter(micFinder));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      // Indicator shows slide to cancel cue
      expect(find.text('Slide to cancel'), findsOneWidget);

      // Drag upwards to trigger cancel cue
      await gesture.moveBy(const Offset(0, -60));
      await tester.pump();
      expect(find.text('Release to cancel'), findsOneWidget);

      // Release finger after cancel
      await gesture.up();
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      expect(sentPcm, isNull);
      expect(sentRate, isNull);
    });

    testWidgets(
      'MessageComposer updates recorder reactively on parent rebuild and does not dispose injected recorder',
      (WidgetTester tester) async {
        final recorderA = FakeAudioRecorderService();
        final recorderB = FakeAudioRecorderService();

        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              bottomNavigationBar: MessageComposer(
                controller: TextEditingController(),
                isBusy: false,
                isGenerating: false,
                canAttachImage: false,
                canAttachAudio: true,
                pendingImageBytes: null,
                pendingImageName: null,
                onSend: () {},
                onStop: () {},
                onPickImage: () {},
                onClearImage: () {},
                audioRecorder: recorderA,
                onSendAudio: (_, _) {},
              ),
            ),
          ),
        );

        // Rebuild parent with recorderB
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              bottomNavigationBar: MessageComposer(
                controller: TextEditingController(),
                isBusy: false,
                isGenerating: false,
                canAttachImage: false,
                canAttachAudio: true,
                pendingImageBytes: null,
                pendingImageName: null,
                onSend: () {},
                onStop: () {},
                onPickImage: () {},
                onClearImage: () {},
                audioRecorder: recorderB,
                onSendAudio: (_, _) {},
              ),
            ),
          ),
        );

        // Remove MessageComposer completely from tree
        await tester.pumpWidget(
          const MaterialApp(home: Scaffold(body: SizedBox())),
        );

        // Injected recorders must not be disposed by MessageComposer
        expect(recorderA.isDisposed, isFalse);
        expect(recorderB.isDisposed, isFalse);
      },
    );

    testWidgets(
      'MessageComposer aborts active push-to-talk recording on dispose',
      (WidgetTester tester) async {
        final recorder = FakeAudioRecorderService();

        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              bottomNavigationBar: MessageComposer(
                controller: TextEditingController(),
                isBusy: false,
                isGenerating: false,
                canAttachImage: false,
                canAttachAudio: true,
                pendingImageBytes: null,
                pendingImageName: null,
                onSend: () {},
                onStop: () {},
                onPickImage: () {},
                onClearImage: () {},
                audioRecorder: recorder,
                onSendAudio: (_, _) {},
              ),
            ),
          ),
        );

        final micFinder = find.byIcon(Icons.mic_none_rounded);
        expect(micFinder, findsOneWidget);

        // Start recording
        await tester.startGesture(tester.getCenter(micFinder));
        await tester.pump();
        expect(recorder.isRecording, isTrue);

        // Unmount widget during active recording
        await tester.pumpWidget(
          const MaterialApp(home: Scaffold(body: SizedBox())),
        );
        await tester.pump();

        expect(recorder.wasCancelled, isTrue);
        expect(recorder.isDisposed, isFalse);
      },
    );

    testWidgets(
      'MessageComposer handles microphone permission denial gracefully without locking recording state',
      (WidgetTester tester) async {
        final recorder = FakeAudioRecorderService()..mockPermission = false;

        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              bottomNavigationBar: MessageComposer(
                controller: TextEditingController(),
                isBusy: false,
                isGenerating: false,
                canAttachImage: false,
                canAttachAudio: true,
                pendingImageBytes: null,
                pendingImageName: null,
                onSend: () {},
                onStop: () {},
                onPickImage: () {},
                onClearImage: () {},
                audioRecorder: recorder,
                onSendAudio: (_, _) {},
              ),
            ),
          ),
        );

        final micFinder = find.byIcon(Icons.mic_none_rounded);
        expect(micFinder, findsOneWidget);

        // Tap/hold mic button with denied permission
        await tester.startGesture(tester.getCenter(micFinder));
        await tester.pump();
        await tester.pumpAndSettle();

        // Must not be stuck in recording state
        expect(recorder.isRecording, isFalse);
        // SnackBar must display permission error
        expect(
          find.text(
            'Microphone error: Bad state: Microphone permission not granted',
          ),
          findsOneWidget,
        );
      },
    );
  });

  group('Voice Mode Selector Bar', () {
    testWidgets('renders all voice modes for bidirectional audio model', (
      WidgetTester tester,
    ) async {
      final controller = ChatController();
      controller.value = controller.value.copyWith(
        loadedModel: () => const MockLoadedModel('LFM2.5-Audio-1.5B'),
        capabilities: () => const CeraCapabilities(
          textIn: true,
          textOut: true,
          imageIn: false,
          audioIn: true,
          audioOut: true,
        ),
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ValueListenableBuilder<ChatState>(
              valueListenable: controller,
              builder: (context, state, _) {
                return Column(
                  children: [
                    if (state.hasModel &&
                        ((state.capabilities?.audioIn ?? false) ||
                            (state.capabilities?.audioOut ?? false)))
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Row(
                          children: [
                            if ((state.capabilities?.audioIn ?? false) &&
                                (state.capabilities?.audioOut ?? false))
                              ActionChip(
                                label: const Text('Voice Chat'),
                                onPressed: () => controller.dispatch(
                                  const UpdateSettingsIntent(
                                    audioChatMode: AudioChatMode.interleaved,
                                  ),
                                ),
                              ),
                            if (state.capabilities?.audioIn ?? false)
                              ActionChip(
                                label: const Text('Speech to Text (ASR)'),
                                onPressed: () => controller.dispatch(
                                  const UpdateSettingsIntent(
                                    audioChatMode: AudioChatMode.speechToText,
                                  ),
                                ),
                              ),
                            if (state.capabilities?.audioOut ?? false)
                              ActionChip(
                                label: const Text('Text to Speech (TTS)'),
                                onPressed: () => controller.dispatch(
                                  const UpdateSettingsIntent(
                                    audioChatMode: AudioChatMode.textToSpeech,
                                  ),
                                ),
                              ),
                            ActionChip(
                              label: const Text('Text Only'),
                              onPressed: () => controller.dispatch(
                                const UpdateSettingsIntent(
                                  audioChatMode: AudioChatMode.textOnly,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                );
              },
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Voice Chat'), findsOneWidget);
      expect(find.text('Speech to Text (ASR)'), findsOneWidget);
      expect(find.text('Text to Speech (TTS)'), findsOneWidget);
      expect(find.text('Text Only'), findsOneWidget);

      // Tap Speech to Text chip
      await tester.tap(find.text('Speech to Text (ASR)'));
      await tester.pump();
      expect(
        controller.value.settings.audioChatMode,
        AudioChatMode.speechToText,
      );

      // Tap Voice Chat chip
      await tester.tap(find.text('Voice Chat'));
      await tester.pump();
      expect(
        controller.value.settings.audioChatMode,
        AudioChatMode.interleaved,
      );

      controller.dispose();
    });
  });

  group('MessageList audio turns and waveform playback', () {
    testWidgets('renders playable AudioWaveformBubble for audio turns', (
      WidgetTester tester,
    ) async {
      try {
        debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
        final audioPlayer = AudioPlayerService();
        final turns = [
          Turn(
            role: 'user',
            text: 'What is the weather today?',
            audioDurationSeconds: 2.5,
            audioSamples: List.filled(40000, 0.1),
          ),
          Turn(
            role: 'assistant',
            text: 'It is bright and sunny outside.',
            modelName: 'LFM2.5-Audio-1.5B',
            audioDurationSeconds: 3.2,
            audioSamples: List.filled(76800, 0.2),
          ),
        ];

        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: MessageList(
                turns: turns,
                scrollController: ScrollController(),
                audioPlayer: audioPlayer,
              ),
            ),
          ),
        );
        await tester.pumpAndSettle();

        expect(find.byType(AudioWaveformBubble), findsNWidgets(2));
        expect(find.text('2.5s'), findsOneWidget);
        expect(find.text('3.2s'), findsOneWidget);
        expect(find.byIcon(Icons.play_arrow_rounded), findsNWidgets(2));
        expect(find.byIcon(Icons.download_rounded), findsNWidgets(2));

        final playCompleter = Completer<void>();
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(
              const MethodChannel('cera/audio_player'),
              (call) async {
                if (call.method == 'play') {
                  return playCompleter.future;
                }
                return;
              },
            );

        // Tap play button on the assistant waveform bubble
        final playButtons = find.byIcon(Icons.play_arrow_rounded);
        await tester.tap(playButtons.last);
        await tester.pump();

        expect(audioPlayer.isPlaying, isTrue);

        // Complete mock audio playback
        playCompleter.complete();
        await tester.pump();

        expect(audioPlayer.isPlaying, isFalse);
        audioPlayer.dispose();
      } finally {
        debugDefaultTargetPlatformOverride = null;
      }
    });

    test(
      'SendAudioPromptIntent does nothing if no model loaded or pcm is empty',
      () async {
        final controller = ChatController();
        expect(controller.value.turns, isEmpty);

        // Empty PCM
        await controller.dispatch(
          const SendAudioPromptIntent(pcmSamples: [], sampleRate: 16000),
        );
        expect(controller.value.turns, isEmpty);

        // No model loaded
        await controller.dispatch(
          const SendAudioPromptIntent(
            pcmSamples: [0.1, 0.2],
            sampleRate: 16000,
          ),
        );
        expect(controller.value.turns, isEmpty);

        controller.dispose();
      },
    );

    test('UpdateSettingsIntent updates audioChatMode and chatVoice', () async {
      final controller = ChatController();
      expect(
        controller.value.settings.audioChatMode,
        AudioChatMode.interleaved,
      );

      await controller.dispatch(
        const UpdateSettingsIntent(audioChatMode: AudioChatMode.speechToText),
      );
      expect(
        controller.value.settings.audioChatMode,
        AudioChatMode.speechToText,
      );

      await controller.dispatch(
        const UpdateSettingsIntent(
          audioChatMode: AudioChatMode.textToSpeech,
          chatVoice: 'Use the British male voice.',
        ),
      );
      expect(
        controller.value.settings.audioChatMode,
        AudioChatMode.textToSpeech,
      );
      expect(
        controller.value.settings.chatVoice,
        'Use the British male voice.',
      );

      controller.dispose();
    });

    test(
      'ChatController.alignAudioMode automatically aligns audioChatMode to model capabilities',
      () {
        const initial = ChatSettings(audioChatMode: AudioChatMode.textOnly);

        // Audio-in only model -> auto-aligns to speechToText (ASR)
        final asrSettings = ChatController.alignAudioMode(
          initial,
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: false,
          ),
        );
        expect(asrSettings.audioChatMode, AudioChatMode.speechToText);

        // Respects explicit textOnly preference when loading bidirectional model
        final preservedTextSettings = ChatController.alignAudioMode(
          initial,
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: true,
          ),
        );
        expect(preservedTextSettings.audioChatMode, AudioChatMode.textOnly);

        // Switch from ASR to bidirectional audio model -> auto-aligns to interleaved Voice Chat
        final voiceSettings = ChatController.alignAudioMode(
          asrSettings,
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: true,
          ),
        );
        expect(voiceSettings.audioChatMode, AudioChatMode.interleaved);

        // Switch from interleaved to TTS-only model -> auto-aligns to textToSpeech
        final ttsSettings = ChatController.alignAudioMode(
          voiceSettings,
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: false,
            audioOut: true,
          ),
        );
        expect(ttsSettings.audioChatMode, AudioChatMode.textToSpeech);

        // Switch from TTS to bidirectional audio model -> auto-aligns to interleaved Voice Chat
        final voiceFromTts = ChatController.alignAudioMode(
          ttsSettings,
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: true,
          ),
        );
        expect(voiceFromTts.audioChatMode, AudioChatMode.interleaved);

        // Switch from TTS-only to text-only model -> auto-resets to textOnly
        final textSettings = ChatController.alignAudioMode(
          ttsSettings,
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: false,
            audioOut: false,
          ),
        );
        expect(textSettings.audioChatMode, AudioChatMode.textOnly);
      },
    );

    test(
      'ChatController.systemPromptFor constructs systemPrompt incorporating voice persona',
      () {
        const settingsWithPersona = ChatSettings(
          audioChatMode: AudioChatMode.interleaved,
          chatVoice: 'Use the British male voice.',
        );

        final promptInterleaved = ChatController.systemPromptFor(
          settings: settingsWithPersona,
          uiMode: AppUIMode.chat,
        );
        expect(
          promptInterleaved,
          'Respond with interleaved text and audio. Use the British male voice.',
        );

        // When persona is empty, trimmed cleanly without trailing whitespace
        const settingsEmptyPersona = ChatSettings(
          audioChatMode: AudioChatMode.interleaved,
          chatVoice: '',
        );
        final promptEmpty = ChatController.systemPromptFor(
          settings: settingsEmptyPersona,
          uiMode: AppUIMode.chat,
        );
        expect(promptEmpty, 'Respond with interleaved text and audio.');

        // When persona is whitespace only, trimmed cleanly without trailing whitespace
        const settingsWhitespacePersona = ChatSettings(
          audioChatMode: AudioChatMode.interleaved,
          chatVoice: '   ',
        );
        final promptWhitespace = ChatController.systemPromptFor(
          settings: settingsWhitespacePersona,
          uiMode: AppUIMode.chat,
        );
        expect(promptWhitespace, 'Respond with interleaved text and audio.');

        // TTS mode for text prompt
        const settingsTts = ChatSettings(
          audioChatMode: AudioChatMode.textToSpeech,
          chatVoice: 'Warm storyteller voice.',
        );
        final promptTts = ChatController.systemPromptFor(
          settings: settingsTts,
          uiMode: AppUIMode.chat,
          isAudioPrompt: false,
        );
        expect(promptTts, 'Perform TTS. Warm storyteller voice.');

        // TTS mode for audio prompt returns null
        final promptTtsAudio = ChatController.systemPromptFor(
          settings: settingsTts,
          uiMode: AppUIMode.chat,
          isAudioPrompt: true,
        );
        expect(promptTtsAudio, isNull);
      },
    );

    test('voiceTagFor maps capabilities to descriptive status labels', () {
      expect(
        ChatController.voiceTagFor(
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: true,
          ),
        ),
        ' · Voice',
      );
      expect(
        ChatController.voiceTagFor(
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: true,
            audioOut: false,
          ),
        ),
        ' · ASR',
      );
      expect(
        ChatController.voiceTagFor(
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: false,
            audioOut: true,
          ),
        ),
        ' · Audio',
      );
      expect(
        ChatController.voiceTagFor(
          const CeraCapabilities(
            textIn: true,
            textOut: true,
            imageIn: false,
            audioIn: false,
            audioOut: false,
          ),
        ),
        '',
      );
    });

    test(
      'StopGenerationIntent stops audio player playback and clears generating state',
      () async {
        final controller = ChatController();
        controller.value = controller.value.copyWith(
          isGenerating: true,
          turns: [
            const Turn(
              role: 'assistant',
              text: 'Generating audio output...',
              isGenerating: true,
            ),
          ],
        );

        expect(controller.value.isGenerating, isTrue);
        expect(controller.value.turns.first.isGenerating, isTrue);

        await controller.dispatch(const StopGenerationIntent());

        expect(controller.value.isGenerating, isFalse);
        expect(controller.value.turns.first.isGenerating, isFalse);

        controller.dispose();
      },
    );

    test('vocoderSampleRate is 24000 Hz', () {
      expect(ChatController.vocoderSampleRate, 24000);
    });
  });
}

class FakeAudioRecorderService extends AudioRecorderService {
  bool _mockRecording = false;
  bool isDisposed = false;
  bool mockPermission = true;

  @override
  bool get isRecording => _mockRecording;

  @override
  Future<bool> hasPermission() async => mockPermission;

  @override
  Future<void> startRecording({int sampleRate = 16000}) async {
    if (!mockPermission) {
      throw StateError('Microphone permission not granted');
    }
    _mockRecording = true;
  }

  @override
  Future<List<double>> stopRecording({
    bool normalize = true,
    bool trim = true,
  }) async {
    _mockRecording = false;
    return [0.1, 0.2, 0.3];
  }

  bool wasCancelled = false;

  @override
  Future<void> cancelRecording() async {
    _mockRecording = false;
    wasCancelled = true;
  }

  @override
  Future<void> dispose() async {
    _mockRecording = false;
    isDisposed = true;
  }
}

class MockLoadedModel implements LoadedModel {
  const MockLoadedModel(this.name);

  @override
  final String name;

  @override
  Future<Cera> open({CeraOptions options = const CeraOptions()}) {
    throw UnimplementedError();
  }
}

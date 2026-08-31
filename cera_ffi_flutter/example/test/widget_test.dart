// Smoke test for the example app.
//
// Deliberately does not load a model: that needs a real .gguf on disk and takes
// seconds. This only asserts the app builds and reaches its empty state, which
// is enough to catch a broken widget tree in CI.

import 'package:cera_ffi_flutter_example/chat_state.dart';
import 'package:cera_ffi_flutter_example/main.dart';
import 'package:cera_ffi_flutter_example/widgets/audio_waveform.dart';
import 'package:cera_ffi_flutter_example/widgets/bundle_picker_dialog.dart';
import 'package:cera_ffi_flutter_example/widgets/message_list.dart';
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
}

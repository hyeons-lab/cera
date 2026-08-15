// Smoke test for the example app.
//
// Deliberately does not load a model: that needs a real .gguf on disk and takes
// seconds. This only asserts the app builds and reaches its empty state, which
// is enough to catch a broken widget tree in CI.

import 'package:cera_ffi_flutter_example/benchmark.dart';
import 'package:cera_ffi_flutter_example/chat_state.dart';
import 'package:cera_ffi_flutter_example/main.dart';
import 'package:cera_ffi_flutter_example/model_source.dart';
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

  testWidgets(
    'the speed icon is disabled and vision attach is hidden before a model is loaded',
    (WidgetTester tester) async {
      await tester.pumpWidget(const CeraExampleApp());

      final button = tester.widget<IconButton>(
        find.widgetWithIcon(IconButton, Icons.speed),
      );
      expect(button.onPressed, isNull);
      expect(find.byIcon(Icons.add_photo_alternate_outlined), findsNothing);
    },
  );

  testWidgets('benchmark page displays the loaded model and enables run', (
    WidgetTester tester,
  ) async {
    final source = ModelSource.forTesting(
      name: 'test-model.gguf',
      path: '/tmp/test-model.gguf',
    );
    await tester.pumpWidget(MaterialApp(home: BenchmarkPage(model: source)));

    expect(find.text('CPU vs GPU'), findsOneWidget);
    expect(find.text('Model: test-model.gguf'), findsOneWidget);
    final run = tester.widget<FilledButton>(
      find.widgetWithText(FilledButton, 'Run benchmark'),
    );
    expect(run.onPressed, isNotNull);
  });

  testWidgets('benchmark page works with a bundle model source', (
    WidgetTester tester,
  ) async {
    const bundle = BundleModelSource(
      name: 'LFM2-700M · Q4_0',
      bundleName: 'LFM2-700M',
      quant: 'Q4_0',
      storeDir: '/tmp/cache',
    );
    await tester.pumpWidget(
      const MaterialApp(home: BenchmarkPage(model: bundle)),
    );

    expect(find.text('CPU vs GPU'), findsOneWidget);
    expect(find.text('Model: LFM2-700M · Q4_0'), findsOneWidget);
    final run = tester.widget<FilledButton>(
      find.widgetWithText(FilledButton, 'Run benchmark'),
    );
    expect(run.onPressed, isNotNull);
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
}

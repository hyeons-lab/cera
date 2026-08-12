// Smoke test for the example app.
//
// Deliberately does not load a model: that needs a real .gguf on disk and takes
// seconds. This only asserts the app builds and reaches its empty state, which
// is enough to catch a broken widget tree in CI.

import 'package:cera_ffi_flutter_example/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('renders the empty state before a model is picked', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const CeraExampleApp());

    expect(find.text('Cera'), findsOneWidget);
    expect(find.text('Open a .gguf model to start.'), findsOneWidget);
  });

  // The benchmark page is reachable only through the app-bar icon, so a broken
  // route or a rename of that icon would strand it with nothing failing. Like
  // the test above this loads no model: it asserts the page builds and reaches
  // its own empty state, which is where a widget-tree regression would show.
  testWidgets('the speed icon opens the benchmark page', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const CeraExampleApp());

    await tester.tap(find.byIcon(Icons.speed));
    await tester.pumpAndSettle();

    expect(find.text('CPU vs GPU'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, 'Run'), findsOneWidget);
    // Nothing is loaded, so Run must be inert.
    final run = tester.widget<FilledButton>(
      find.widgetWithText(FilledButton, 'Run'),
    );
    expect(run.onPressed, isNull);
  });
}

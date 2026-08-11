// Smoke test for the example app.
//
// Deliberately does not load a model: that needs a real .gguf on disk and takes
// seconds. This only asserts the app builds and reaches its empty state, which
// is enough to catch a broken widget tree in CI.

import 'package:cera_ffi_flutter_example/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('renders the empty state before a model is picked',
      (WidgetTester tester) async {
    await tester.pumpWidget(const CeraExampleApp());

    expect(find.text('Cera'), findsOneWidget);
    expect(find.text('Open a .gguf model to start.'), findsOneWidget);
  });
}

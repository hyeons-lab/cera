library;

import 'dart:io';

import 'package:test/test.dart';

/// Asserts the record writer [writerName] in the generated bindings clones
/// its object handle before writing it, and that the clone (not the raw
/// lowered handle) is what reaches the wire.
///
/// Shared by the `EngineConfig` / `LoraAdapterEntry` ownership guard tests:
/// both writers are synthesized by `tool/patch_generated_bindings.dart`
/// (the generator stubs them), and both guards fail the same way, so the
/// defect explanations live here rather than drifting between two files.
/// The per-record use-after-free background stays on each test's own doc
/// comment.
void expectRecordWriterClonesHandle({
  required String writerName,
  required String cloneFn,
  required String rawLower,
  required String handleType,
}) {
  final source = File('lib/src/generated/cera_ffi.dart').readAsStringSync();
  final start = source.indexOf('void $writerName(');
  expect(
    start,
    isNonNegative,
    reason:
        '$writerName is missing from the generated bindings; '
        'tool/patch_generated_bindings.dart synthesizes it, so it should '
        'always be there',
  );
  final body = source.substring(start, source.indexOf('\n}\n', start));

  expect(
    body,
    contains(cloneFn),
    reason:
        'the handle must be cloned before it is written: Rust lifts this '
        "field with into_arc and drops the Arc, so lowering the caller's own "
        'handle leaves the Dart $handleType dangling (use-after-free on '
        'close() or on finalization)',
  );

  // The clone has to be what reaches the wire. Calling the clone function
  // and then writing the raw lowered handle anyway would pass the check
  // above while still transferring the caller's reference, and would
  // additionally leak the clone.
  expect(
    body,
    contains('writer.writeU64(clonedHandle)'),
    reason: 'the CLONED handle must be the one written, not the original',
  );
  expect(
    body,
    isNot(contains('writer.writeU64($rawLower')),
    reason: 'writing the raw lowered handle is the bug this test exists for',
  );
}

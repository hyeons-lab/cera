@TestOn('vm')
library;

import 'dart:io';

import 'package:test/test.dart';

/// Guard on who owns the `BundleRepo` handle written into an `EngineConfig`.
///
/// A source-level assertion, matching `optional_primitive_return_test.dart`:
/// the defect is visible in the generated text, and reaching it at runtime
/// means building the cdylib and making a real call.
///
/// The defect it guards is a use-after-free. UniFFI lifts an object field of a
/// record with `into_arc`, which TAKES OWNERSHIP: `EngineConfig::try_from`
/// clones the inner repo and then drops that `Arc`, dropping the refcount to
/// zero. So a record writer that lowers the raw handle hands Rust the caller's
/// only strong reference, and every Dart `BundleRepo` used this way is dangling
/// the moment the call returns. `close()`, or the finalizer if nobody calls it,
/// then frees a slot Rust has already freed and something else may have taken.
///
/// It is not theoretical. Before the fix, a probe that lowered a repo rooted at
/// `/tmp/cera-uaf-probe` into an `EngineConfig`, made one call, then allocated
/// eight more repos, read its own `storeDir()` back as `/tmp/churn1`: a
/// different, later-allocated `BundleRepo` sitting in the freed slot.
///
/// Nothing else catches it. `dart analyze` cannot (the generated directory is
/// excluded and the code is well-typed either way), and the generator's own
/// tests cannot, because this writer is synthesized by
/// `tool/patch_generated_bindings.dart` rather than emitted by the generator.
/// The bug also stayed invisible for as long as it did because nothing passed a
/// repo through `EngineConfig`: `Cera.openBundle` was the first caller.
void main() {
  test('the EngineConfig record writer clones the BundleRepo handle', () {
    final source = File('lib/src/generated/cera_ffi.dart').readAsStringSync();
    final start = source.indexOf('void _uniffiWriteEngineConfig(');
    expect(
      start,
      isNonNegative,
      reason:
          '_uniffiWriteEngineConfig is missing from the generated bindings; '
          'tool/patch_generated_bindings.dart synthesizes it, so it should '
          'always be there',
    );
    final body = source.substring(start, source.indexOf('\n}\n', start));

    expect(
      body,
      contains('_bundleRepoClone'),
      reason:
          'the handle must be cloned before it is written: Rust lifts this '
          "field with into_arc and drops the Arc, so lowering the caller's own "
          'handle leaves the Dart BundleRepo dangling (use-after-free on '
          'close() or on finalization)',
    );

    // The clone has to be what reaches the wire. Calling `_bundleRepoClone`
    // and then writing `BundleRepoFfiCodec.lower(...)` anyway would pass the
    // check above while still transferring the caller's reference, and would
    // additionally leak the clone.
    expect(
      body,
      contains('writer.writeU64(clonedHandle)'),
      reason: 'the CLONED handle must be the one written, not the original',
    );
    expect(
      body,
      isNot(contains('writer.writeU64(BundleRepoFfiCodec.lower(')),
      reason: 'writing the raw lowered handle is the bug this test exists for',
    );
  });
}

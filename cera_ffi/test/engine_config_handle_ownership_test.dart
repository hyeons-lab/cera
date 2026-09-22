@TestOn('vm')
library;

import 'package:test/test.dart';

import 'helpers/record_writer_ownership.dart';

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
    expectRecordWriterClonesHandle(
      writerName: '_uniffiWriteEngineConfig',
      cloneFn: '_bundleRepoClone',
      rawLower: 'BundleRepoFfiCodec.lower(',
      handleType: 'BundleRepo',
    );
  });
}

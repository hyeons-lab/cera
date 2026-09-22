@TestOn('vm')
library;

import 'package:test/test.dart';

import 'helpers/record_writer_ownership.dart';

/// Guard on who owns the `LoraAdapters` handle written into a
/// `LoraAdapterEntry`.
///
/// A source-level assertion: the defect is visible in the generated text,
/// and reaching it at runtime means building the cdylib and making a real
/// call.
///
/// The defect it guards is a use-after-free with the same shape as the
/// `EngineConfig` one. UniFFI lifts an object field of a record with
/// `into_arc`, which TAKES OWNERSHIP: `Session::set_lora_adapters` clones
/// the inner weights out of the entry and then drops that `Arc`, dropping
/// the refcount to zero. So a record writer that lowers the raw handle hands
/// Rust the caller's only strong reference, and every Dart `LoraAdapters`
/// used this way is dangling the moment the call returns.
///
/// Nothing else catches it. `dart analyze` cannot (the generated directory is
/// excluded and the code is well-typed either way), and the generator's own
/// tests cannot, because this writer is synthesized by
/// `tool/patch_generated_bindings.dart` rather than emitted by the generator.
void main() {
  test('the LoraAdapterEntry record writer clones the LoraAdapters handle', () {
    expectRecordWriterClonesHandle(
      writerName: '_uniffiWriteLoraAdapterEntry',
      cloneFn: '_loraAdaptersClone',
      rawLower: 'LoraAdaptersFfiCodec.lower(',
      handleType: 'LoraAdapters',
    );
  });
}

@TestOn('vm')
library;

import 'dart:io';

import 'package:test/test.dart';

/// Guard on how the generated bindings decode an `Option<primitive>` return.
///
/// This is a source-level assertion rather than a call, because calling is the
/// expensive option: the four methods below are all on `CeraEngine`, so
/// reaching one means building the cdylib and loading a GGUF. The defect is
/// visible in the generated text, so read the text.
///
/// The defect it guards: UniFFI returns `Option<u32>` as a RustBuffer, like
/// every other non-scalar. The generator special-cased it as a JSON-encoded
/// `Pointer<Utf8>` and read the buffer's CAPACITY word as that pointer, so the
/// first call dereferenced a small integer and segfaulted the process: no
/// exception, no stack, just SIGSEGV at address 0x8.
///
/// Nothing caught it for a long time. `dart analyze` cannot: the generated
/// directory is excluded, and the code is well-typed regardless. The
/// generator's own unit tests could not either, because reaching this renderer
/// needs a real cdylib with the `uniffi_ffibuffer_*` trampolines, which a unit
/// test has no way to produce. What found it was the first caller: the
/// portable `Cera` API needs `bosToken()` to decide BOS framing.
void main() {
  // Every SYNCHRONOUS `Option<primitive>`-returning method on the FFI surface.
  // Adding one to the Rust side without adding it here leaves it unguarded,
  // which is exactly how these four went unnoticed.
  //
  // The async path is deliberately out of scope. It routes through a separate
  // renderer that is coordinated with `async_support.rs`, which maps optional
  // primitives (and optional records, optional enums, string-keyed maps) onto
  // a `Pointer<Utf8>` JSON convention on purpose. Whether that convention is
  // correct is a real question with no evidence attached to it yet: no
  // `cera-ffi` async method returns an optional primitive, so nothing exercises
  // it. The first one that does should arrive with a test here.
  const methods = [
    'ceraEngineInvokeBosToken',
    'ceraEngineInvokeEosToken',
    'ceraEngineInvokeSpecialTokenId',
    'ceraEngineInvokeToolCallStartToken',
  ];

  late final String source;

  setUpAll(() {
    final file = File('lib/src/generated/cera_ffi.dart');
    expect(
      file.existsSync(),
      isTrue,
      reason: 'run from the cera_ffi package root, where the bindings live',
    );
    source = file.readAsStringSync();
  });

  for (final method in methods) {
    test('$method decodes from the returned RustBuffer', () {
      final start = source.indexOf('  int? $method(');
      expect(start, isNot(-1), reason: '$method is missing from the bindings');
      // The wrapper ends at the first dedented closing brace.
      final end = source.indexOf('\n  }\n', start);
      expect(end, isNot(-1), reason: 'could not find the end of $method');
      final body = source.substring(start, end);

      expect(
        body,
        contains('_UniFfiBinaryReader'),
        reason: 'must read the optional tag and payload out of the RustBuffer',
      );
      expect(
        body,
        isNot(contains('toDartString()')),
        reason:
            'reading the return as a C string is the segfault: the pointer it '
            'casts is the RustBuffer capacity word, not an address',
      );
    });
  }
}

@TestOn('vm')
library;

import 'package:cera_ffi/cera_ffi.dart';
import 'package:cera_ffi/src/async/cera_io.dart' show sessionConfigOf;
import 'package:test/test.dart';

/// An omitted `kvCompression` now means the core default (f16 where the model honors it), so the
/// wrapper's documented full-precision default has to ask for it by name. These tests pin that
/// mapping; none of them call into the native library.
void main() {
  test('the default mode asks for the full-precision cache explicitly', () {
    final config = sessionConfigOf(const CeraOptions());
    expect(config.kvCompression, isA<KvCompressionNone>());
  });

  test('f16 and TurboQuant map to their own modes', () {
    expect(
      sessionConfigOf(
        const CeraOptions(kvCompression: CeraKvCompression.f16),
      ).kvCompression,
      isA<KvCompressionF16>(),
    );
    expect(
      sessionConfigOf(const CeraOptions(turboQuant: true)).kvCompression,
      isA<KvCompressionTurboQuant>(),
    );
  });
}

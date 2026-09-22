@TestOn('vm')
library;

import 'package:cera_ffi/cera_ffi.dart';
import 'package:test/test.dart';

/// Bounds on the per-request `generate` seed, shared by both backends.
///
/// The native and web `generate` implementations both funnel through
/// [checkGenerateSeedRange] with their platform max, and neither engine is
/// constructible without a model, so the helper itself is the test target:
///
/// * native: `0 <= seed < 2^63` (Dart's `int` range; negatives throw),
/// * web: `0 <= seed < 2^53` (the seed crosses `postMessage` as a JS
///   number, exact only below 2^53).
void main() {
  group('checkGenerateSeedRange', () {
    test('omitted seed is always allowed', () {
      checkGenerateSeedRange(null, 0x7FFFFFFFFFFFFFFF);
      checkGenerateSeedRange(null, 9007199254740991);
    });

    test('native bound is 0..=2^63-1', () {
      const max = 0x7FFFFFFFFFFFFFFF;
      checkGenerateSeedRange(0, max);
      checkGenerateSeedRange(max, max);
      expect(
        () => checkGenerateSeedRange(-1, max),
        throwsA(isA<RangeError>().having((e) => e.name, 'name', 'seed')),
      );
    });

    test('web bound is 0..=2^53-1', () {
      const max = 9007199254740991;
      checkGenerateSeedRange(0, max);
      checkGenerateSeedRange(max, max);
      expect(
        () => checkGenerateSeedRange(-1, max),
        throwsA(isA<RangeError>().having((e) => e.name, 'name', 'seed')),
      );
      expect(
        () => checkGenerateSeedRange(max + 1, max),
        throwsA(isA<RangeError>().having((e) => e.name, 'name', 'seed')),
      );
    });

    test('web rejects what native accepts above 2^53', () {
      // The guards differ: a seed exact in a Dart int but not in a JS number
      // passes the native bound and fails the web one.
      const webMax = 9007199254740991;
      checkGenerateSeedRange(webMax + 1, 0x7FFFFFFFFFFFFFFF);
      expect(
        () => checkGenerateSeedRange(webMax + 1, webMax),
        throwsRangeError,
      );
    });
  });
}

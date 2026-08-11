// Behaviour of the generated web stub.
//
// This imports `src/generated/cera_ffi_web.dart` DIRECTLY rather than through
// `package:cera_ffi/cera_ffi.dart`. Going through the entry point would resolve
// the conditional export to the native bindings here, since the test runs on the
// VM where `dart.library.ffi` is available, and would test nothing.
//
// Compiling the stub for a browser is a different check and needs a Flutter web
// build; CI does that. What this covers is the part a compile cannot: that the
// data types are usable and every engine entry point refuses loudly.
@TestOn('vm')
library;

import 'package:cera_ffi/src/generated/cera_ffi_web.dart';
import 'package:test/test.dart';

void main() {
  group('web stub data types', () {
    test('records are real, not throwing shells', () {
      const config = EngineConfig(
        contextSize: 2048,
        backend: BackendPreference.cpu,
        bundleRepo: null,
      );
      expect(config.contextSize, 2048);
      expect(config.backend, BackendPreference.cpu);
      // copyWith and equality come from the shared renderer, so they work too.
      expect(config.copyWith(contextSize: 4096).contextSize, 4096);
      expect(
        config,
        equals(
          const EngineConfig(
            contextSize: 2048,
            backend: BackendPreference.cpu,
            bundleRepo: null,
          ),
        ),
      );
    });

    test('enums keep their variants', () {
      expect(BackendPreference.values, contains(BackendPreference.metal));
    });

    test('sealed error types can be constructed and matched', () {
      const FfiError err = FfiErrorBackend(detail: 'nope');
      expect(err, isA<FfiErrorBackend>());
      expect((err as FfiErrorBackend).detail, 'nope');
    });
  });

  group('web stub entry points', () {
    test('top-level functions throw UnsupportedError naming themselves', () {
      expect(
        () => ceraFfiVersion(),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            allOf(contains('ceraFfiVersion'), contains('dart:ffi')),
          ),
        ),
      );
      expect(cpuBackendReport, throwsUnsupportedError);
    });

    test('object constructors throw', () {
      expect(
        () => CeraEngine.fromPath(
          'model.gguf',
          const EngineConfig(
            contextSize: 0,
            backend: BackendPreference.auto,
            bundleRepo: null,
          ),
        ),
        throwsUnsupportedError,
      );
      expect(() => BundleRepo.create('/tmp/cache'), throwsUnsupportedError);
    });

    test('configureDefaultBindings throws but reset stays a no-op', () {
      expect(configureDefaultBindings, throwsUnsupportedError);
      // Teardown must not throw: it is the kind of call that ends up in a
      // `finally`, where throwing would replace the real error.
      expect(resetDefaultBindings, returnsNormally);
    });
  });
}

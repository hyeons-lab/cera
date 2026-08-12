@TestOn('vm')
library;

import 'package:cera_ffi/cera_ffi.dart';
import 'package:test/test.dart';

void main() {
  group('CeraLibrary', () {
    test('base name matches the Rust lib name', () {
      // Derived from the crate name `cera-ffi` (no `[lib] name` override), so
      // the platform files are libcera_ffi.{so,dylib} / cera_ffi.dll. If this
      // ever changes, every platform integration has to change with it.
      expect(CeraLibrary.baseName, 'cera_ffi');
    });

    test('exposes the path override env var', () {
      expect(CeraLibrary.pathEnvVar, 'CERA_FFI_LIB');
    });

    test('an explicit path that does not exist fails loudly', () {
      expect(
        () => CeraLibrary.open(path: '/nonexistent/libcera_ffi.dylib'),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('an empty explicit path is ignored rather than opened', () {
      // `open(path: '')` must not call DynamicLibrary.open(''); it falls
      // through to the normal resolution order. Whatever happens next, the
      // failure must not be about an empty filename.
      //
      // Return-value assertions are avoided here on purpose: the conditional
      // export in library_loader.dart resolves to the web stub for static
      // analysis, whose `open` returns `Never`, so anything after the call
      // analyzes as dead code even though the VM runs the dart:io variant.
      try {
        CeraLibrary.open(path: '');
      } on ArgumentError catch (err) {
        expect(err.toString(), isNot(contains("''")));
      } on UnsupportedError {
        // Platform with no bundled library and no override: also acceptable.
      }
    });
  });
}

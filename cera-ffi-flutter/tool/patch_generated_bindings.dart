// Post-generation patch for the UniFFI-generated Dart bindings.
//
// `uniffi-bindgen-dart` 0.1.3 emits a handful of bugs against Cera's FFI
// surface (see V2.17 in docs/IMPLEMENTATION_PLAN.md). This tool applies the
// fixes that are unambiguously correct, run automatically by `just dart-bindings`
// right after generation. It is idempotent — re-running on an already-patched
// file is a no-op.
//
// What it fixes:
//   1. `.ref.pointer` -> `.ref.ptr`  (3 sites)
//      The `_UniFfiFfiBufferElement` union field is named `ptr`; the generator
//      reads a non-existent `pointer` getter when unpacking returned pointers.
//   2. async constructor return type (1 site)
//      `fromBundleIdAsync` is declared `Future<CeraEngine>` but its body returns
//      the (generator-stubbed, synchronous) inner call. Marking the wrapper
//      `async` auto-wraps the return into a Future and turns the stub's throw
//      into a rejected Future — type-correct, behaviour unchanged.
//
//   3. callback-interface lowering (4 sites)
//      The generator can't lower `DownloadProgressSink` / `ModalitySink`
//      arguments to their handle ints, so `*WithProgress` and `*Streaming*`
//      methods don't type-check. There is no correct codegen we can recover
//      here, so we neutralize them: the sink-lowering assignments become a
//      `throw UnsupportedError(...)` (a `throw` is a bottom-typed expression,
//      so it satisfies the `int` field without dead code), and the unused
//      `onProgress` bridge call is made type-correct. Net effect: every
//      *synchronous* engine API (`fromPath`, `generate`, `transcribe`, …)
//      compiles and works; the progress/streaming variants throw at call time.
//      Tracked in V2.17.

import 'dart:io';

void main(List<String> args) {
  final path = args.isNotEmpty
      ? args.first
      : 'lib/src/generated/cera_ffi.dart';
  final file = File(path);
  if (!file.existsSync()) {
    stderr.writeln('patch_generated_bindings: $path not found — run `just dart-bindings` first.');
    exit(1);
  }

  var src = file.readAsStringSync();
  var applied = 0;

  // Fix 1: union field name.
  const badGetter = '.ref.pointer';
  const goodGetter = '.ref.ptr';
  final getterHits = badGetter.allMatches(src).length;
  if (getterHits > 0) {
    src = src.replaceAll(badGetter, goodGetter);
    applied += getterHits;
    stdout.writeln('  fixed .ref.pointer -> .ref.ptr ($getterHits sites)');
  }

  // Fix 2: async constructor return type.
  const asyncSig =
      'fromBundleIdAsync(String bundleId, String quant, EngineConfig config) {';
  const asyncFixed =
      'fromBundleIdAsync(String bundleId, String quant, EngineConfig config) async {';
  if (src.contains(asyncSig)) {
    src = src.replaceAll(asyncSig, asyncFixed);
    applied += 1;
    stdout.writeln('  fixed fromBundleIdAsync return type (1 site)');
  }

  // Fix 3: stub the entry points that still aren't usable. Working now (NOT
  // stubbed): sync `generate` / `generate_streaming`, and `generateAsync`
  // (ffibuffer rust-future poll/complete — verified async, event loop stays
  // responsive).
  //
  // `generateStreamingAsync` is async AND takes a callback sink. cera runs it on
  // a tokio blocking-pool thread, so the sink fires from a NON-isolate thread —
  // and `NativeCallable.isolateLocal` can only be invoked from its owning
  // isolate ("Cannot invoke native callback outside an isolate"). Supporting it
  // needs `NativeCallable.listener` (cross-thread, queued) callback vtables, a
  // larger generator change. Stub until then. (`generate_streaming` sync is
  // fine: callbacks fire on the calling thread.)
  //
  // `from_bundle_id_async` (async constructor returning an object handle) stays
  // on the generator's own stub — it needs the object/pointer rust-future
  // variant, which the generator doesn't emit yet.
  // generate_streaming_async is now usable: the trait-callback vtable's void
  // methods use NativeCallable.listener (cross-thread delivery), so sink
  // callbacks fired from cera's tokio worker thread reach the Dart isolate.
  const methodStubs = <String, String>{};
  methodStubs.forEach((bad, good) {
    if (src.contains(bad)) {
      src = src.replaceAll(bad, good);
      applied += 1;
      stdout.writeln('  stubbed unsupported entry point');
    }
  });

  // Fix 4: native-library resolution. The generator emits a single
  // `libraryName = 'uniffi_cera_ffi'` and `DynamicLibrary.open(libraryName)`,
  // which is both the wrong base name (the cdylib is `cera_ffi`) and missing
  // the platform prefix/suffix. Rewrite the default `open()` to honor a
  // `CERA_FFI_LIB` path override and otherwise resolve the platform-correct
  // filename. (An explicit `dynamicLibrary`/`libraryPath` still wins.)
  const importAnchor = "import 'dart:typed_data';";
  const importWithIo = "import 'dart:typed_data';\nimport 'dart:io' as io;";
  if (src.contains(importAnchor) && !src.contains("import 'dart:io' as io;")) {
    src = src.replaceFirst(importAnchor, importWithIo);
  }

  const openBad =
      'return ffi.DynamicLibrary.open(_libraryPath ?? libraryName);';
  const openGood = '''
final envPath = io.Platform.environment['CERA_FFI_LIB'];
    if (provided == null && _libraryPath == null && envPath != null && envPath.isNotEmpty) {
      return ffi.DynamicLibrary.open(envPath);
    }
    return ffi.DynamicLibrary.open(_libraryPath ?? _ceraDefaultLibraryFile());''';
  if (src.contains(openBad)) {
    src = src.replaceAll(openBad, openGood);
    if (!src.contains('String _ceraDefaultLibraryFile()')) {
      src += '''

// Added by tool/patch_generated_bindings.dart — platform-correct default name
// for the cera-ffi cdylib (`cera_ffi`). iOS links statically; manual users can
// pass a DynamicLibrary or set CERA_FFI_LIB.
String _ceraDefaultLibraryFile() {
  if (io.Platform.isMacOS) return 'libcera_ffi.dylib';
  if (io.Platform.isWindows) return 'cera_ffi.dll';
  return 'libcera_ffi.so';
}
''';
    }
    applied += 1;
    stdout.writeln('  fixed native-library resolution (CERA_FFI_LIB + platform name)');
  }

  // Fix 5: RustBuffer / rust_future symbol names. The generator emits the
  // `rustbuffer_*` and `rust_future_*_rust_buffer` symbol families with a
  // spurious `uniffi_` infix (`ffi_uniffi_cera_ffi_*`); UniFFI exports them as
  // `ffi_cera_ffi_*`. (The `uniffi_ffibuffer_*`, `uniffi_cera_ffi_checksum_*`,
  // and `ffi_cera_ffi_uniffi_contract_version` symbols are already correct and
  // don't contain this substring, so the replacement is safe.)
  const symBad = 'ffi_uniffi_cera_ffi_';
  const symGood = 'ffi_cera_ffi_';
  final symHits = symBad.allMatches(src).length;
  if (symHits > 0) {
    src = src.replaceAll(symBad, symGood);
    applied += symHits;
    stdout.writeln('  fixed rustbuffer/rust_future symbol names ($symHits sites)');
  }

  // Fix 6: EngineConfig record encoding. The generator stubs the writer for
  // any record containing an interface-handle field, and EngineConfig has
  // `bundleRepo: BundleRepo?`. We synthesize it from the Rust record shape
  // (context_size: u64, backend: BackendPreference enum, bundle_repo:
  // Option<Arc<BundleRepo>>), mirroring the binary format the other (working)
  // record writers use — primitives, an enum tag via _uniffiWriteBackendPreference,
  // and the Option flag byte (writeI8 0/1) seen in _uniffiWriteSessionConfig.
  const writeStub =
      "void _uniffiWriteEngineConfig(EngineConfig value, _UniFfiBinaryWriter writer) {\n"
      "  throw UnsupportedError('UniFFI binary encode not fully supported for EngineConfig');\n"
      "}";
  const writeImpl =
      "void _uniffiWriteEngineConfig(EngineConfig value, _UniFfiBinaryWriter writer) {\n"
      "  writer.writeU64(value.contextSize);\n"
      "  _uniffiWriteBackendPreference(value.backend, writer);\n"
      "  if (value.bundleRepo == null) {\n"
      "    writer.writeI8(0);\n"
      "  } else {\n"
      "    writer.writeI8(1);\n"
      "    writer.writeU64(BundleRepoFfiCodec.lower(value.bundleRepo!));\n"
      "  }\n"
      "}";
  if (src.contains(writeStub)) {
    src = src.replaceAll(writeStub, writeImpl);
    applied += 1;
    stdout.writeln('  implemented _uniffiWriteEngineConfig (record with handle field)');
  }
  const encodeStub =
      "Uint8List _uniffiEncodeEngineConfig(EngineConfig value) {\n"
      "  throw UnsupportedError('UniFFI binary encode not fully supported for EngineConfig');\n"
      "}";
  const encodeImpl =
      "Uint8List _uniffiEncodeEngineConfig(EngineConfig value) {\n"
      "  final writer = _UniFfiBinaryWriter();\n"
      "  _uniffiWriteEngineConfig(value, writer);\n"
      "  return writer.toBytes();\n"
      "}";
  if (src.contains(encodeStub)) {
    src = src.replaceAll(encodeStub, encodeImpl);
    applied += 1;
    stdout.writeln('  implemented _uniffiEncodeEngineConfig');
  }

  if (applied == 0) {
    stdout.writeln('  no patches needed (already patched or upstream fixed).');
  }
  file.writeAsStringSync(src);
}

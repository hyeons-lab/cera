// Post-generation patch for the UniFFI-generated Dart bindings.
//
// `uniffi-bindgen-dart` 0.1.3 emits a handful of bugs against Cera's FFI
// surface (see V2.17 in docs/IMPLEMENTATION_PLAN.md). This tool applies the
// fixes that are unambiguously correct, run automatically by `just dart-bindings`
// right after generation. It is idempotent — re-running on an already-patched
// file is a no-op.
//
// What it fixes:
//   1. `.ref.pointer` -> `.ref.ptr`  (0 sites today; kept as a backstop)
//      The `_UniFfiFfiBufferElement` union field is named `ptr`; the generator
//      read a non-existent `pointer` getter when unpacking returned pointers.
//      Every site was an `Option<primitive>` return decoded as a JSON C string,
//      and the generator no longer emits those, so this now matches nothing.
//      Left in place because it is a pure no-op when there is nothing to fix
//      and the upstream shape could come back.
//   2. async constructor wrappers (3 sites)
//      `CeraEngine.fromBundleIdAsync`, `fromPathAsync` and `fromBytesAsync` are
//      declared `Future<CeraEngine>` but not marked `async`, so each returns the
//      inner binding call's Future directly. The types already agree (the inner
//      call returns `Future<CeraEngine>` too), so this is about behaviour rather
//      than typing: an `async` body turns a synchronous throw from `_bindings()`
//      (a missing or unloadable native library) into a failed future, which is
//      what a `Future`-returning constructor should do. All three are patched
//      because all three have the same shape; patching one made them differ on
//      that failure.
//
//   (Callback-interface lowering is NO LONGER patched here. The vendored
//    generator under `third_party/uniffi-bindgen-dart` now lowers
//    `DownloadProgressSink` / `ModalitySink` arguments and emits working
//    callback vtables, so the `*WithProgress` / `*Streaming*` entry points —
//    including `generateStreamingAsync` via NativeCallable.listener — type-check
//    and run without a patch. The async-constructor body is no longer patched
//    either: the generator drives the real rust-future lifecycle for
//    `fromBundleIdAsync` rather than emitting a throwing stub.)
//
// Plus native-lib resolution, RustBuffer/rust_future symbol names, and the
// EngineConfig record encoder (Fixes 4–6 below).

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

  // Fix 2: mark the public async-constructor wrapper `async`.
  const asyncSigs = [
    'fromBundleIdAsync(String bundleId, String quant, EngineConfig config) {',
    'fromPathAsync(String path, EngineConfig config) {',
    'fromBytesAsync(Uint8List bytes, EngineConfig config) {',
  ];
  for (final sig in asyncSigs) {
    final fixed = '${sig.substring(0, sig.length - 1)}async {';
    if (src.contains(sig)) {
      src = src.replaceAll(sig, fixed);
      applied += 1;
      final name = sig.substring(0, sig.indexOf('('));
      stdout.writeln('  marked $name wrapper async (1 site)');
    }
  }

  // (No callback-stubbing fix: the vendored generator now lowers the sink
  // arguments and emits working callback vtables, so `*WithProgress` and
  // `*Streaming*` — including `generateStreamingAsync` via NativeCallable.listener
  // — type-check and run without a patch. `fromBundleIdAsync` needs no body
  // patch either; the generator emits its rust-future poll/complete lifecycle.)

  // Fix 4: native-library resolution. The generator emits a single
  // `libraryName = 'uniffi_cera_ffi'` and `DynamicLibrary.open(libraryName)`,
  // which is both the wrong base name (the library is `cera_ffi`) and missing
  // the platform prefix/suffix. It also cannot express the Apple case, where
  // the published XCFramework is a *static* `libcera_ffi.a` linked into the
  // app binary — there is no file to open, the symbols are in the process.
  //
  // Rather than reimplement that here, delegate to `CeraLibrary.open()` so
  // Flutter apps and plain `dart run` scripts resolve the library through one
  // code path. (An explicit `dynamicLibrary` still wins; an explicit
  // `libraryPath` is forwarded.)
  const importAnchor = "import 'dart:typed_data';";
  const importWithLoader =
      "import 'dart:typed_data';\nimport '../library_loader.dart';";
  if (src.contains(importAnchor) &&
      !src.contains("import '../library_loader.dart';")) {
    src = src.replaceFirst(importAnchor, importWithLoader);
  }

  const openBad =
      'return ffi.DynamicLibrary.open(_libraryPath ?? libraryName);';
  const openGood = 'return CeraLibrary.open(path: _libraryPath);';
  if (src.contains(openBad)) {
    src = src.replaceAll(openBad, openGood);
    applied += 1;
    stdout.writeln('  fixed native-library resolution (delegates to CeraLibrary.open)');
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

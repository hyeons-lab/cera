// Verifies BundleRepo.withProgress: a DownloadProgressSink whose onProgress
// fires (from the calling thread, since fromBundleId is synchronous) while a
// bundle downloads. To avoid pulling hundreds of MB, the sink prints the
// first callback then throws — enough to prove the RustBuffer-decoded args
// (url: String, bytesDownloaded: u64, totalBytes: Option<u64>) round-trip.
//
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
//     dart run example/cera_progress.dart <bundleId> <quant> [storeDir]
//
// e.g. bundleId=LFM2-350M-GGUF quant=Q4_0
import 'dart:io' show Directory, exit;

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

class _AbortAfterFirst implements DownloadProgressSink {
  @override
  void onProgress(String url, int bytesDownloaded, int? totalBytes) {
    // Verdict printed from inside the callback — proof the RustBuffer ABI
    // round-trips a String, a u64, and an Option<u64> across the boundary.
    // Then throw to abort before pulling the full model; the throw unwinds the
    // synchronous download (nothing after fromBundleId reliably runs).
    // ignore: avoid_print
    print('✓ onProgress fired — url=$url bytesDownloaded=$bytesDownloaded '
        'totalBytes=$totalBytes');
    throw _StopDownload();
  }
}

class _StopDownload implements Exception {}

void main(List<String> args) {
  final bundleId = args.isNotEmpty ? args[0] : 'LFM2-350M-GGUF';
  final quant = args.length > 1 ? args[1] : 'Q4_0';
  final storeDir = args.length > 2
      ? args[2]
      : Directory.systemTemp.createTempSync('cera_prog_').path;

  print('cera ${ceraFfiVersion()} — verifying BundleRepo.withProgress');
  print('bundle=$bundleId quant=$quant store=$storeDir');

  final repo = BundleRepo.withProgress(storeDir, _AbortAfterFirst());
  print('BundleRepo.withProgress constructed OK; downloading (will abort)…');

  try {
    CeraEngine.fromBundleId(
      bundleId,
      quant,
      EngineConfig(
        contextSize: 2048,
        backend: BackendPreference.cpu,
        bundleRepo: repo,
      ),
    );
  } catch (e) {
    // Expected when the sink throws to abort; anything else (e.g. a network /
    // HTTP error before any progress) is surfaced here for diagnosis.
    print('fromBundleId ended: ${e.runtimeType}: $e');
  }
  exit(0);
}

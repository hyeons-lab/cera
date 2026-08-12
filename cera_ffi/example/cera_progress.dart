// Verifies BundleRepo.withProgress: a DownloadProgressSink whose onProgress
// fires while a bundle downloads, proving the RustBuffer-decoded args
// (url: String, bytesDownloaded: u64, totalBytes: Option<u64>) round-trip.
//
// Uses the ASYNC loader, and has to. `onProgress` is a void method on a
// callback interface handed to an object constructor, so the generator emits it
// as a `NativeCallable.listener`: Rust invokes it from a tokio blocking worker,
// and the call is queued on this isolate's event loop rather than run on the
// calling thread. The synchronous `fromBundleId` would block that event loop
// for the whole download, so every callback would sit in the queue until it
// returned, and `exit(0)` would then run before any of them got a turn.
// `fromBundleIdAsync` leaves the loop free, so they arrive as they happen.
//
// For the same reason the sink cannot abort the download by throwing: a
// listener has no channel back to Rust, so the generated bridge swallows the
// exception. This pulls the whole bundle (a few hundred MB for the default),
// cached in `storeDir`, so a second run against the same directory is fast and
// emits no progress at all.
//
//   CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
//     dart run example/cera_progress.dart <bundleId> <quant> [storeDir]
//
// e.g. bundleId=LFM2-350M-GGUF quant=Q4_0
import 'dart:io' show Directory, exit;

import 'package:cera_ffi/cera_ffi.dart';

class _PrintProgress implements DownloadProgressSink {
  /// Total callbacks received, reported at the end as the proof of life.
  int calls = 0;

  /// The file the last event was about, so only the first event per file is
  /// printed.
  String? _url;

  @override
  void onProgress(String url, int bytesDownloaded, int? totalBytes) {
    calls += 1;
    // Printed from inside the callback: proof the RustBuffer ABI round-trips a
    // String, a u64 and an Option<u64> across the boundary.
    //
    // First event per file only. A bundle emits one roughly every 256 KB, so
    // printing them all would bury the result under thousands of lines.
    if (url == _url) return;
    _url = url;
    // ignore: avoid_print
    print(
      'onProgress fired: url=$url bytesDownloaded=$bytesDownloaded '
      'totalBytes=$totalBytes',
    );
  }
}

Future<void> main(List<String> args) async {
  final bundleId = args.isNotEmpty ? args[0] : 'LFM2-350M-GGUF';
  final quant = args.length > 1 ? args[1] : 'Q4_0';
  final storeDir =
      args.length > 2
          ? args[2]
          : Directory.systemTemp.createTempSync('cera_prog_').path;

  print('cera ${ceraFfiVersion()}: verifying BundleRepo.withProgress');
  print('bundle=$bundleId quant=$quant store=$storeDir');

  final sink = _PrintProgress();
  final repo = BundleRepo.withProgress(storeDir, sink);
  print('BundleRepo.withProgress constructed OK; downloading…');

  try {
    final engine = await CeraEngine.fromBundleIdAsync(
      bundleId,
      quant,
      EngineConfig(
        contextSize: 2048,
        backend: BackendPreference.cpu,
        bundleRepo: repo,
      ),
    );
    print('fromBundleIdAsync loaded the model; ${sink.calls} progress events');
    engine.close();
  } catch (e) {
    // A network or HTTP failure, or a bundle id / quant that does not exist.
    // The sink can no longer abort, so this is never the expected path.
    print('fromBundleIdAsync failed: ${e.runtimeType}: $e');
  }
  exit(0);
}

import 'bundle_size_probe_stub.dart'
    if (dart.library.io) 'bundle_size_probe_io.dart'
    if (dart.library.js_interop) 'bundle_size_probe_web.dart'
    as impl;

/// Probes a bundle manifest and returns a mapping from URL/filename to file size in bytes.
Future<Map<String, int>> probeBundleFileSizes(
  String bundleName,
  String quant,
) => impl.probeBundleFileSizes(bundleName, quant);

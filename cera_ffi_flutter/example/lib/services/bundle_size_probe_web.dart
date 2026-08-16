import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';

@JS('ceraProbeBundleFiles')
external JSPromise<JSString>? _ceraProbeBundleFiles(
  JSString bundleName,
  JSString quant,
);

/// Web implementation using the index.html helper function.
Future<Map<String, int>> probeBundleFileSizes(
  String bundleName,
  String quant,
) async {
  try {
    final promise = _ceraProbeBundleFiles(bundleName.toJS, quant.toJS);
    if (promise == null) return {};
    final jsStr = await promise.toDart.timeout(const Duration(seconds: 4));
    final dartStr = jsStr.toDart;
    final decoded = jsonDecode(dartStr) as Map<String, dynamic>;
    return {
      for (final entry in decoded.entries)
        if (entry.value is num) entry.key: (entry.value as num).toInt(),
    };
  } catch (_) {
    return {};
  }
}

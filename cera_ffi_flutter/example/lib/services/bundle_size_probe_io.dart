import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// Native (dart:io) implementation to probe manifest and query file sizes via HTTP HEAD.
Future<Map<String, int>> probeBundleFileSizes(
  String bundleName,
  String quant,
) async {
  final client = HttpClient();
  client.connectionTimeout = const Duration(seconds: 3);
  try {
    final manifestUrl =
        'https://huggingface.co/LiquidAI/LeapBundles/raw/main/$bundleName/$quant.json';
    final manifestUri = Uri.parse(manifestUrl);
    final req = await client
        .getUrl(manifestUri)
        .timeout(const Duration(seconds: 3));
    final res = await req.close().timeout(const Duration(seconds: 3));
    if (res.statusCode != 200) return {};
    final jsonStr = await utf8.decoder
        .bind(res)
        .join()
        .timeout(const Duration(seconds: 3));
    final data = jsonDecode(jsonStr) as Map<String, dynamic>;
    final loadTime = data['load_time_parameters'] as Map<String, dynamic>?;
    if (loadTime == null) return {};

    final fileUrls = <String>[];
    for (final key in [
      'model',
      'multimodal_projector',
      'audio_decoder',
      'audio_tokenizer',
    ]) {
      final val = loadTime[key];
      if (val is String && val.trim().isNotEmpty) {
        final trimmed = val.trim();
        final resolved = trimmed.startsWith('http')
            ? trimmed
            : manifestUri.resolve(trimmed).toString();
        fileUrls.add(resolved);
      }
    }

    final sizes = <String, int>{};
    await Future.wait(
      fileUrls.map((url) async {
        try {
          final uri = Uri.parse(url);
          final headReq = await client
              .headUrl(uri)
              .timeout(const Duration(seconds: 3));
          final headRes = await headReq.close().timeout(
            const Duration(seconds: 3),
          );
          if (headRes.contentLength > 0) {
            sizes[url] = headRes.contentLength;
            final fileName = url.split('/').last;
            sizes[fileName] = headRes.contentLength;
          }
        } catch (_) {}
      }),
    );
    return sizes;
  } catch (_) {
    return {};
  } finally {
    client.close(force: true);
  }
}

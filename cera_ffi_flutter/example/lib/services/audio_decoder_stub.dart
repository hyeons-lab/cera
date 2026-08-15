import 'dart:typed_data';

/// Stub decoder for non-web platforms.
Future<Float32List?> decodeAudioBlob(String blobUrl, int sampleRate) async {
  return null;
}

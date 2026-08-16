import 'dart:typed_data';

import 'file_download_stub.dart'
    if (dart.library.js_interop) 'file_download_web.dart'
    if (dart.library.io) 'file_download_io.dart'
    as platform;
import 'wav_encoder.dart';

/// Downloads or saves a byte buffer as a file on the user's device.
Future<void> downloadFile(
  Uint8List bytes, {
  required String filename,
  String mimeType = 'application/octet-stream',
}) async {
  await platform.downloadFileBytes(
    bytes,
    filename: filename,
    mimeType: mimeType,
  );
}

/// Encodes Float32 audio samples into a 16-bit WAV file and triggers download.
Future<void> downloadAudioWav(
  List<double> samples, {
  required String filename,
  int sampleRate = 24000,
}) async {
  if (samples.isEmpty) return;
  final wavBytes = encodeWav(samples, sampleRate: sampleRate);
  await downloadFile(wavBytes, filename: filename, mimeType: 'audio/wav');
}

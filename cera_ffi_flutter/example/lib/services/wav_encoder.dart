import 'dart:typed_data';

/// Encodes mono Float32 PCM samples into a standard 16-bit PCM WAV byte buffer.
Uint8List encodeWav(List<double> samples, {int sampleRate = 24000}) {
  final numSamples = samples.length;
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample ~/ 8;
  final byteRate = sampleRate * numChannels * bytesPerSample;
  const blockAlign = numChannels * bytesPerSample;
  final dataSize = numSamples * bytesPerSample;
  final fileSize = 36 + dataSize;

  final buffer = ByteData(44 + dataSize);

  // RIFF header
  buffer.setUint8(0, 0x52); // 'R'
  buffer.setUint8(1, 0x49); // 'I'
  buffer.setUint8(2, 0x46); // 'F'
  buffer.setUint8(3, 0x46); // 'F'
  buffer.setUint32(4, fileSize, Endian.little);
  buffer.setUint8(8, 0x57); // 'W'
  buffer.setUint8(9, 0x41); // 'A'
  buffer.setUint8(10, 0x56); // 'V'
  buffer.setUint8(11, 0x45); // 'E'

  // 'fmt ' subchunk
  buffer.setUint8(12, 0x66); // 'f'
  buffer.setUint8(13, 0x6D); // 'm'
  buffer.setUint8(14, 0x74); // 't'
  buffer.setUint8(15, 0x20); // ' '
  buffer.setUint32(16, 16, Endian.little); // Subchunk1Size for PCM
  buffer.setUint16(20, 1, Endian.little); // AudioFormat 1 = PCM
  buffer.setUint16(22, numChannels, Endian.little);
  buffer.setUint32(24, sampleRate, Endian.little);
  buffer.setUint32(28, byteRate, Endian.little);
  buffer.setUint16(32, blockAlign, Endian.little);
  buffer.setUint16(34, bitsPerSample, Endian.little);

  // 'data' subchunk
  buffer.setUint8(36, 0x64); // 'd'
  buffer.setUint8(37, 0x61); // 'a'
  buffer.setUint8(38, 0x74); // 't'
  buffer.setUint8(39, 0x61); // 'a'
  buffer.setUint32(40, dataSize, Endian.little);

  // Convert Float32 samples to Int16 PCM
  var offset = 44;
  for (var i = 0; i < numSamples; i++) {
    final s = samples[i];
    final sample = s.isFinite ? s.clamp(-1.0, 1.0) : 0.0;
    final int16Val = (sample * 32767.0).round().clamp(-32768, 32767);
    buffer.setInt16(offset, int16Val, Endian.little);
    offset += 2;
  }

  return buffer.buffer.asUint8List(buffer.offsetInBytes, buffer.lengthInBytes);
}

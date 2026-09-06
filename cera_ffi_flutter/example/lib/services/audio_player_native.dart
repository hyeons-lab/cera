import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';

Process? _activeProcess;
File? _currentTempFile;

final List<double> _streamBuffer = [];
int _streamSampleRate = 24000;

/// Whether audio playback is supported natively on this platform.
bool get isAudioPlaybackSupported => Platform.isMacOS;

/// Converts Float32 PCM samples to 16-bit mono RIFF/WAVE bytes.
Uint8List _pcmToWav(
  Float32List samples,
  int sampleRate, {
  int numChannels = 1,
}) {
  final byteData = ByteData(44 + samples.length * 2);

  // RIFF chunk descriptor
  byteData.setUint8(0, 0x52); // 'R'
  byteData.setUint8(1, 0x49); // 'I'
  byteData.setUint8(2, 0x46); // 'F'
  byteData.setUint8(3, 0x46); // 'F'
  byteData.setUint32(4, 36 + samples.length * 2, Endian.little);
  byteData.setUint8(8, 0x57); // 'W'
  byteData.setUint8(9, 0x41); // 'A'
  byteData.setUint8(10, 0x56); // 'V'
  byteData.setUint8(11, 0x45); // 'E'

  // fmt subchunk
  byteData.setUint8(12, 0x66); // 'f'
  byteData.setUint8(13, 0x6d); // 'm'
  byteData.setUint8(14, 0x74); // 't'
  byteData.setUint8(15, 0x20); // ' '
  byteData.setUint32(16, 16, Endian.little); // Subchunk1Size (16 for PCM)
  byteData.setUint16(20, 1, Endian.little); // AudioFormat (1 for PCM)
  byteData.setUint16(22, numChannels, Endian.little);
  byteData.setUint32(24, sampleRate, Endian.little);
  byteData.setUint32(
    28,
    sampleRate * numChannels * 2,
    Endian.little,
  ); // ByteRate
  byteData.setUint16(32, numChannels * 2, Endian.little); // BlockAlign
  byteData.setUint16(34, 16, Endian.little); // BitsPerSample

  // data subchunk
  byteData.setUint8(36, 0x64); // 'd'
  byteData.setUint8(37, 0x61); // 'a'
  byteData.setUint8(38, 0x74); // 't'
  byteData.setUint8(39, 0x61); // 'a'
  byteData.setUint32(40, samples.length * 2, Endian.little);

  int offset = 44;
  for (int i = 0; i < samples.length; i++) {
    final sample = samples[i];
    final valid = sample.isFinite ? sample : 0.0;
    final clamped = valid.clamp(-1.0, 1.0);
    final s16 = (clamped < 0 ? clamped * 32768.0 : clamped * 32767.0)
        .round()
        .clamp(-32768, 32767);
    byteData.setInt16(offset, s16, Endian.little);
    offset += 2;
  }

  return byteData.buffer.asUint8List();
}

void _cleanTempFile() {
  final file = _currentTempFile;
  _currentTempFile = null;
  if (file != null) {
    try {
      if (file.existsSync()) {
        file.deleteSync();
      }
    } catch (_) {}
  }
}

/// Plays a completed mono Float32 PCM waveform natively via macOS afplay.
Future<void> playAudioPcm(Float32List samples, int sampleRate) async {
  if (samples.isEmpty) return;
  stopAudioPlayback();

  if (!Platform.isMacOS) {
    debugPrint(
      '[cera:audio_player_native] Native playback is currently supported on macOS',
    );
    return;
  }

  try {
    final wavBytes = _pcmToWav(samples, sampleRate);
    final tempDir = Directory.systemTemp;
    final tempFile = File(
      '${tempDir.path}/cera_audio_${DateTime.now().microsecondsSinceEpoch}.wav',
    );
    await tempFile.writeAsBytes(wavBytes, flush: true);
    _currentTempFile = tempFile;

    final process = await Process.start('/usr/bin/afplay', [tempFile.path]);
    _activeProcess = process;
    await process.exitCode;
  } catch (err) {
    debugPrint('[cera:audio_player_native] playAudioPcm failed: $err');
  } finally {
    _cleanTempFile();
    _activeProcess = null;
  }
}

/// Stops any active audio playback.
void stopAudioPlayback() {
  final proc = _activeProcess;
  _activeProcess = null;
  if (proc != null) {
    try {
      proc.kill(ProcessSignal.sigterm);
    } catch (_) {}
  }
  _cleanTempFile();
}

/// Begins an audio streaming session.
void startAudioStream(int sampleRate) {
  debugPrint('[cera:audio_player_native] startAudioStream ($sampleRate Hz)');
  stopAudioStream();
  _streamSampleRate = sampleRate;
  _streamBuffer.clear();
}

/// Appends an audio PCM chunk to the live stream buffer.
void appendAudioStreamChunk(Float32List chunk) {
  if (chunk.isNotEmpty) {
    _streamBuffer.addAll(chunk);
  }
}

/// Flushes remaining buffered chunks when stream generation completes.
void finishAudioStream() {
  debugPrint(
    '[cera:audio_player_native] finishAudioStream (${_streamBuffer.length} samples)',
  );
  if (_streamBuffer.isNotEmpty) {
    final allSamples = Float32List.fromList(_streamBuffer);
    _streamBuffer.clear();
    unawaited(playAudioPcm(allSamples, _streamSampleRate));
  }
}

/// Stops streaming and ends audio playback.
void stopAudioStream() {
  _streamBuffer.clear();
  stopAudioPlayback();
}

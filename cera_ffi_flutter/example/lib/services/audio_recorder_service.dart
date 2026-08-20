import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

import 'audio_decoder_stub.dart'
    if (dart.library.js_interop) 'audio_decoder_web.dart'
    as web_audio;

/// Service managing microphone capture for Cera multimodal audio models.
class AudioRecorderService {
  final AudioRecorder _recorder = AudioRecorder();
  final List<double> _accumulatedPcm = [];
  StreamSubscription<Uint8List>? _streamSub;
  bool _isRecording = false;
  int _currentSampleRate = 16000;

  bool get isRecording => _isRecording;

  /// Checks and requests microphone recording permissions.
  Future<bool> hasPermission() async {
    try {
      return await _recorder.hasPermission();
    } catch (_) {
      return false;
    }
  }

  /// Begins streaming 16-bit PCM audio from the microphone (or recording on web).
  Future<void> startRecording({int sampleRate = 16000}) async {
    if (_isRecording) return;
    _accumulatedPcm.clear();
    _currentSampleRate = sampleRate;

    final hasPerm = await hasPermission();
    if (!hasPerm) {
      throw StateError('Microphone permission not granted');
    }

    if (kIsWeb) {
      await _recorder.start(
        RecordConfig(sampleRate: sampleRate, numChannels: 1),
        path: '',
      );
      _isRecording = true;
    } else {
      final stream = await _recorder.startStream(
        RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: sampleRate,
          numChannels: 1,
        ),
      );

      _isRecording = true;
      _streamSub = stream.listen(
        (chunk) {
          final samples = pcm16ToFloat32(chunk);
          _accumulatedPcm.addAll(samples);
        },
        onError: (err) {
          debugPrint('[cera:audio_recorder] stream error: $err');
          _isRecording = false;
        },
        cancelOnError: true,
      );
    }
  }

  /// Stops recording and returns all accumulated normalized Float32 PCM samples.
  Future<List<double>> stopRecording({
    bool normalize = true,
    bool trim = true,
  }) async {
    if (!_isRecording) return [];
    _isRecording = false;
    await _streamSub?.cancel();
    _streamSub = null;

    List<double> pcm;
    if (kIsWeb) {
      final blobUrl = await _recorder.stop();
      if (blobUrl != null && blobUrl.isNotEmpty) {
        final decoded = await web_audio.decodeAudioBlob(
          blobUrl,
          _currentSampleRate,
        );
        pcm = decoded != null ? decoded.toList() : [];
      } else {
        pcm = [];
      }
    } else {
      try {
        await _recorder.stop();
      } catch (_) {}
      pcm = List<double>.from(_accumulatedPcm);
    }

    if (pcm.isEmpty) return [];

    if (trim) {
      pcm = trimSilence(pcm, sampleRate: _currentSampleRate);
    }
    if (normalize) {
      pcm = normalizeAudio(pcm);
    }
    return pcm;
  }

  /// Cancels recording and discards audio.
  Future<void> cancelRecording() async {
    _isRecording = false;
    await _streamSub?.cancel();
    _streamSub = null;
    _accumulatedPcm.clear();
    try {
      await _recorder.stop();
    } catch (_) {}
  }

  Future<void> dispose() async {
    await cancelRecording();
    await _recorder.dispose();
  }

  /// Converts 16-bit signed integer little-endian PCM bytes into normalized [-1.0, 1.0] floats.
  static List<double> pcm16ToFloat32(Uint8List bytes) {
    final byteData = ByteData.sublistView(bytes);
    final count = bytes.lengthInBytes ~/ 2;
    final floats = Float32List(count);
    for (var i = 0; i < count; i++) {
      final sample = byteData.getInt16(i * 2, Endian.little);
      floats[i] = (sample / 32768.0).clamp(-1.0, 1.0);
    }
    return floats;
  }

  /// Trims leading and trailing silence from PCM samples using frame RMS energy.
  static List<double> trimSilence(
    List<double> samples, {
    int sampleRate = 16000,
    double thresholdFactor = 0.08,
    double minThreshold = 0.015,
    int paddingMs = 120,
  }) {
    if (samples.length < sampleRate ~/ 10) return samples; // < 100ms: skip

    final frameSize = (sampleRate * 0.02).round(); // 20ms frame
    if (frameSize <= 0) return samples;

    final numFrames = samples.length ~/ frameSize;
    if (numFrames == 0) return samples;

    final frameRms = Float64List(numFrames);
    double maxRms = 0.0;

    for (var f = 0; f < numFrames; f++) {
      double sumSquares = 0.0;
      final start = f * frameSize;
      for (var i = 0; i < frameSize; i++) {
        final val = samples[start + i];
        sumSquares += val * val;
      }
      final rms = math.sqrt(sumSquares / frameSize);
      frameRms[f] = rms;
      if (rms > maxRms) {
        maxRms = rms;
      }
    }

    if (maxRms < minThreshold) {
      // Entire signal is very quiet: keep original to avoid false empty
      return samples;
    }

    final dynamicThreshold = math.max(minThreshold, maxRms * thresholdFactor);

    var startFrame = 0;
    while (startFrame < numFrames && frameRms[startFrame] < dynamicThreshold) {
      startFrame++;
    }

    var endFrame = numFrames - 1;
    while (endFrame > startFrame && frameRms[endFrame] < dynamicThreshold) {
      endFrame--;
    }

    final padSamples = (sampleRate * (paddingMs / 1000.0)).round();
    final startSample = math.max(0, (startFrame * frameSize) - padSamples);
    final endSample = math.min(
      samples.length,
      ((endFrame + 1) * frameSize) + padSamples,
    );

    if (startSample >= endSample) return samples;
    return samples.sublist(startSample, endSample);
  }

  /// Normalizes PCM samples so peak absolute amplitude is scaled to [targetPeak].
  static List<double> normalizeAudio(
    List<double> samples, {
    double targetPeak = 0.9,
  }) {
    if (samples.isEmpty) return samples;

    double maxAmp = 0.0;
    for (final s in samples) {
      final absVal = s.abs();
      if (absVal > maxAmp) maxAmp = absVal;
    }

    if (maxAmp < 1e-4) return samples; // Silent or nearly silent

    final scale = targetPeak / maxAmp;
    final normalized = Float32List(samples.length);
    for (var i = 0; i < samples.length; i++) {
      normalized[i] = (samples[i] * scale).clamp(-1.0, 1.0);
    }
    return normalized;
  }
}

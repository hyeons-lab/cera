import 'dart:async';
import 'package:flutter/foundation.dart';

import 'audio_player_stub.dart'
    if (dart.library.js_interop) 'audio_player_web.dart'
    as web_player;

/// Cross-platform audio player service for streaming and playing back model audio output.
class AudioPlayerService {
  bool _isPlaying = false;

  bool get isPlaying => _isPlaying;

  /// Plays a completed mono Float32 PCM waveform.
  Future<void> playPcm(List<double> samples, {int sampleRate = 24000}) async {
    if (samples.isEmpty) return;
    _isPlaying = true;
    try {
      final floatList = Float32List.fromList(samples);
      if (kIsWeb) {
        await web_player.playAudioPcm(floatList, sampleRate);
      }
    } finally {
      _isPlaying = false;
    }
  }

  /// Begins an audio streaming session.
  void startStream({int sampleRate = 24000}) {
    _isPlaying = true;
    if (kIsWeb) {
      web_player.startAudioStream(sampleRate);
    }
  }

  /// Appends an audio PCM chunk to the live stream.
  void appendChunk(List<double> chunk) {
    if (chunk.isEmpty || !_isPlaying) return;
    final floatList = Float32List.fromList(chunk);
    if (kIsWeb) {
      web_player.appendAudioStreamChunk(floatList);
    }
  }

  /// Stops streaming and ends audio playback.
  void stop() {
    _isPlaying = false;
    if (kIsWeb) {
      web_player.stopAudioStream();
      web_player.stopAudioPlayback();
    }
  }

  void dispose() {
    stop();
  }
}

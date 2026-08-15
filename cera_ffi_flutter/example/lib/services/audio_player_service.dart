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
    debugPrint(
      '[cera:audio_player_service] playPcm called with ${samples.length} samples at $sampleRate Hz (kIsWeb=$kIsWeb)',
    );
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
    debugPrint(
      '[cera:audio_player_service] startStream called (sampleRate=$sampleRate, kIsWeb=$kIsWeb)',
    );
    _isPlaying = true;
    if (kIsWeb) {
      web_player.startAudioStream(sampleRate);
    }
  }

  /// Appends an audio PCM chunk to the live stream.
  void appendChunk(List<double> chunk) {
    debugPrint(
      '[cera:audio_player_service] appendChunk called with ${chunk.length} samples (isPlaying=$_isPlaying, kIsWeb=$kIsWeb)',
    );
    if (chunk.isEmpty || !_isPlaying) return;
    final floatList = Float32List.fromList(chunk);
    if (kIsWeb) {
      web_player.appendAudioStreamChunk(floatList);
    }
  }

  /// Stops streaming and ends audio playback.
  void stop() {
    debugPrint('[cera:audio_player_service] stop called (kIsWeb=$kIsWeb)');
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

import 'dart:async';
import 'package:flutter/foundation.dart';

import 'audio_player_stub.dart'
    if (dart.library.js_interop) 'audio_player_web.dart'
    if (dart.library.io) 'audio_player_native.dart'
    as audio_backend;

/// Cross-platform audio player service for streaming and playing back model audio output.
class AudioPlayerService extends ChangeNotifier {
  bool _isPlaying = false;
  Object? _activeAudioSource;
  int _currentPlayId = 0;

  bool get isPlaying => _isPlaying;
  Object? get activeAudioSource => _activeAudioSource;

  /// Whether real-time audio playback is supported on the active platform.
  bool get isSupported => audio_backend.isAudioPlaybackSupported;

  /// Whether a specific audio source (e.g. sample buffer) is currently playing.
  bool isSourcePlaying(Object? source) {
    if (!_isPlaying || _activeAudioSource == null || source == null) {
      return false;
    }
    return identical(_activeAudioSource, source) ||
        _activeAudioSource == source;
  }

  /// Plays a completed mono Float32 PCM waveform.
  Future<void> playPcm(
    List<double> samples, {
    int sampleRate = 24000,
    Object? source,
  }) async {
    if (samples.isEmpty) return;
    debugPrint(
      '[cera:audio_player_service] playPcm called with ${samples.length} samples at $sampleRate Hz',
    );
    stop();
    final playId = ++_currentPlayId;
    _isPlaying = true;
    _activeAudioSource = source ?? samples;
    _notifySafely();

    try {
      final floatList = Float32List.fromList(samples);
      await audio_backend.playAudioPcm(floatList, sampleRate);
    } finally {
      if (_currentPlayId == playId) {
        _isPlaying = false;
        _activeAudioSource = null;
        _notifySafely();
      }
    }
  }

  /// Begins an audio streaming session.
  void startStream({int sampleRate = 24000}) {
    debugPrint(
      '[cera:audio_player_service] startStream called (sampleRate=$sampleRate)',
    );
    stop();
    _currentPlayId++;
    _isPlaying = true;
    _activeAudioSource = this;
    _notifySafely();
    audio_backend.startAudioStream(sampleRate);
  }

  /// Appends an audio PCM chunk to the live stream.
  void appendChunk(List<double> chunk) {
    debugPrint(
      '[cera:audio_player_service] appendChunk called with ${chunk.length} samples (isPlaying=$_isPlaying)',
    );
    if (chunk.isEmpty || !_isPlaying || !identical(_activeAudioSource, this)) {
      return;
    }
    final floatList = Float32List.fromList(chunk);
    audio_backend.appendAudioStreamChunk(floatList);
  }

  bool _disposed = false;

  void _notifySafely() {
    if (!_disposed) {
      notifyListeners();
    }
  }

  /// Flushes remaining buffered chunks when stream generation completes.
  void finishStream() {
    debugPrint('[cera:audio_player_service] finishStream called');
    if (!identical(_activeAudioSource, this)) return;
    _isPlaying = false;
    _activeAudioSource = null;
    _notifySafely();
    audio_backend.finishAudioStream();
  }

  /// Stops streaming and ends audio playback.
  void stop() {
    debugPrint('[cera:audio_player_service] stop called');
    _currentPlayId++;
    final wasPlaying = _isPlaying || _activeAudioSource != null;
    _isPlaying = false;
    _activeAudioSource = null;
    if (wasPlaying) {
      _notifySafely();
    }
    audio_backend.stopAudioStream();
    audio_backend.stopAudioPlayback();
  }

  @override
  void dispose() {
    _disposed = true;
    stop();
    super.dispose();
  }
}

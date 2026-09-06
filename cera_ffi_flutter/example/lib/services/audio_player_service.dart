import 'dart:async';
import 'package:flutter/foundation.dart';

import 'audio_player_stub.dart'
    if (dart.library.js_interop) 'audio_player_web.dart'
    if (dart.library.io) 'audio_player_native.dart'
    as audio_backend;

/// Cross-platform audio player service for streaming and playing back model audio output.
class AudioPlayerService extends ChangeNotifier {
  final audio_backend.AudioPlayerBackend _backend;
  bool _isPlaying = false;
  Object? _activeAudioSource;
  int _currentPlayId = 0;
  bool _disposed = false;

  AudioPlayerService({audio_backend.AudioPlayerBackend? backend})
    : _backend = backend ?? audio_backend.AudioPlayerBackend();

  bool get isPlaying => _isPlaying;
  Object? get activeAudioSource => _activeAudioSource;

  /// Whether real-time audio playback is supported on the active platform.
  bool get isSupported => _backend.isSupported;

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
    if (samples.isEmpty || _disposed) return;
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
      await _backend.playPcm(floatList, sampleRate);
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
    if (_disposed) return;
    debugPrint(
      '[cera:audio_player_service] startStream called (sampleRate=$sampleRate)',
    );
    stop();
    _currentPlayId++;
    _isPlaying = true;
    _activeAudioSource = this;
    _notifySafely();
    _backend.startStream(sampleRate);
  }

  /// Appends an audio PCM chunk to the live stream.
  void appendChunk(List<double> chunk) {
    if (chunk.isEmpty ||
        _disposed ||
        !_isPlaying ||
        !identical(_activeAudioSource, this)) {
      return;
    }
    final floatList = Float32List.fromList(chunk);
    _backend.appendStreamChunk(floatList);
  }

  void _notifySafely() {
    if (!_disposed) {
      notifyListeners();
    }
  }

  /// Flushes remaining buffered chunks when stream generation completes.
  void finishStream() {
    debugPrint('[cera:audio_player_service] finishStream called');
    if (_disposed || !identical(_activeAudioSource, this)) return;
    _isPlaying = false;
    _activeAudioSource = null;
    _notifySafely();
    _backend.finishStream();
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
    _backend.stopStream();
    _backend.stopPlayback();
  }

  @override
  void dispose() {
    _disposed = true;
    stop();
    _backend.dispose();
    super.dispose();
  }
}

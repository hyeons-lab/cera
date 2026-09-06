import 'dart:async';
import 'dart:js_interop';
import 'package:flutter/foundation.dart';

@JS('ceraPlayAudio')
external JSPromise<JSAny?>? _ceraPlayAudio(
  JSFloat32Array float32Array,
  JSNumber sampleRate,
);

@JS('ceraStopAudio')
external void _ceraStopAudio();

@JS('ceraStartAudioStream')
external void _ceraStartAudioStream(JSNumber sampleRate);

@JS('ceraAppendAudioStreamChunk')
external void _ceraAppendAudioStreamChunk(JSFloat32Array chunk);

@JS('ceraFinishAudioStream')
external void _ceraFinishAudioStream();

@JS('ceraStopAudioStream')
external void _ceraStopAudioStream();

/// Whether audio playback is supported in the browser via Web Audio API.
bool get isAudioPlaybackSupported => true;

/// Web Audio API backend implementation for browser environments.
class AudioPlayerBackend {
  /// Whether audio playback is supported in the browser via Web Audio API.
  bool get isSupported => isAudioPlaybackSupported;

  /// Plays a single audio PCM buffer using the browser Web Audio API.
  Future<void> playPcm(Float32List samples, int sampleRate) async {
    if (samples.isEmpty) return;
    debugPrint(
      '[cera:audio_player_web] playPcm calling ceraPlayAudio with ${samples.length} samples at $sampleRate Hz',
    );
    try {
      final promise = _ceraPlayAudio(samples.toJS, sampleRate.toJS);
      if (promise != null) {
        await promise.toDart;
      }
    } catch (err) {
      debugPrint('[cera:audio_player_web] playPcm failed: $err');
    }
  }

  /// Stops any active audio playback.
  void stopPlayback() {
    debugPrint('[cera:audio_player_web] stopPlayback calling ceraStopAudio');
    try {
      _ceraStopAudio();
    } catch (err) {
      debugPrint('[cera:audio_player_web] stopPlayback failed: $err');
    }
  }

  /// Starts an audio streaming playback session.
  void startStream(int sampleRate) {
    debugPrint(
      '[cera:audio_player_web] startStream calling ceraStartAudioStream ($sampleRate Hz)',
    );
    try {
      _ceraStartAudioStream(sampleRate.toJS);
    } catch (err) {
      debugPrint('[cera:audio_player_web] startStream failed: $err');
    }
  }

  /// Appends a PCM chunk to the active audio stream.
  void appendStreamChunk(Float32List chunk) {
    debugPrint(
      '[cera:audio_player_web] appendStreamChunk calling ceraAppendAudioStreamChunk with ${chunk.length} samples',
    );
    try {
      _ceraAppendAudioStreamChunk(chunk.toJS);
    } catch (err) {
      debugPrint('[cera:audio_player_web] appendStreamChunk failed: $err');
    }
  }

  /// Flushes and finishes active stream playback.
  void finishStream() {
    debugPrint(
      '[cera:audio_player_web] finishStream calling ceraFinishAudioStream',
    );
    try {
      _ceraFinishAudioStream();
    } catch (err) {
      debugPrint('[cera:audio_player_web] finishStream failed: $err');
    }
  }

  /// Stops and closes the audio streaming session.
  void stopStream() {
    debugPrint(
      '[cera:audio_player_web] stopStream calling ceraStopAudioStream',
    );
    try {
      _ceraStopAudioStream();
    } catch (err) {
      debugPrint('[cera:audio_player_web] stopStream failed: $err');
    }
  }

  /// Disposes backend resources.
  void dispose() {
    stopStream();
    stopPlayback();
  }
}

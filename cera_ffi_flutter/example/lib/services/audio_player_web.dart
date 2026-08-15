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

@JS('ceraStopAudioStream')
external void _ceraStopAudioStream();

/// Plays a single audio PCM buffer using the browser Web Audio API.
Future<void> playAudioPcm(Float32List samples, int sampleRate) async {
  debugPrint(
    '[cera:audio_player_web] playAudioPcm calling ceraPlayAudio with ${samples.length} samples at $sampleRate Hz',
  );
  try {
    final promise = _ceraPlayAudio(samples.toJS, sampleRate.toJS);
    if (promise != null) {
      await promise.toDart;
    }
  } catch (err) {
    debugPrint('[cera:audio_player_web] playAudioPcm failed: $err');
  }
}

/// Stops any active audio playback.
void stopAudioPlayback() {
  debugPrint('[cera:audio_player_web] stopAudioPlayback calling ceraStopAudio');
  try {
    _ceraStopAudio();
  } catch (err) {
    debugPrint('[cera:audio_player_web] stopAudioPlayback failed: $err');
  }
}

/// Starts an audio streaming playback session.
void startAudioStream(int sampleRate) {
  debugPrint(
    '[cera:audio_player_web] startAudioStream calling ceraStartAudioStream ($sampleRate Hz)',
  );
  try {
    _ceraStartAudioStream(sampleRate.toJS);
  } catch (err) {
    debugPrint('[cera:audio_player_web] startAudioStream failed: $err');
  }
}

/// Appends a PCM chunk to the active audio stream.
void appendAudioStreamChunk(Float32List chunk) {
  debugPrint(
    '[cera:audio_player_web] appendAudioStreamChunk calling ceraAppendAudioStreamChunk with ${chunk.length} samples',
  );
  try {
    _ceraAppendAudioStreamChunk(chunk.toJS);
  } catch (err) {
    debugPrint('[cera:audio_player_web] appendAudioStreamChunk failed: $err');
  }
}

/// Stops and closes the audio streaming session.
void stopAudioStream() {
  debugPrint(
    '[cera:audio_player_web] stopAudioStream calling ceraStopAudioStream',
  );
  try {
    _ceraStopAudioStream();
  } catch (err) {
    debugPrint('[cera:audio_player_web] stopAudioStream failed: $err');
  }
}

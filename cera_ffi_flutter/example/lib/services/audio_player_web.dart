import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

@JS('window.ceraPlayAudio')
external JSPromise<JSAny?>? _ceraPlayAudio(
  JSFloat32Array float32Array,
  JSNumber sampleRate,
);

@JS('window.ceraStopAudio')
external void _ceraStopAudio();

@JS('window.ceraStartAudioStream')
external void _ceraStartAudioStream(JSNumber sampleRate);

@JS('window.ceraAppendAudioStreamChunk')
external void _ceraAppendAudioStreamChunk(JSFloat32Array chunk);

@JS('window.ceraStopAudioStream')
external void _ceraStopAudioStream();

/// Plays a single audio PCM buffer using the browser Web Audio API.
Future<void> playAudioPcm(Float32List samples, int sampleRate) async {
  try {
    final promise = _ceraPlayAudio(samples.toJS, sampleRate.toJS);
    if (promise != null) {
      await promise.toDart;
    }
  } catch (_) {}
}

/// Stops any active audio playback.
void stopAudioPlayback() {
  try {
    _ceraStopAudio();
  } catch (_) {}
}

/// Starts an audio streaming playback session.
void startAudioStream(int sampleRate) {
  try {
    _ceraStartAudioStream(sampleRate.toJS);
  } catch (_) {}
}

/// Appends a PCM chunk to the active audio stream.
void appendAudioStreamChunk(Float32List chunk) {
  try {
    _ceraAppendAudioStreamChunk(chunk.toJS);
  } catch (_) {}
}

/// Stops and closes the audio streaming session.
void stopAudioStream() {
  try {
    _ceraStopAudioStream();
  } catch (_) {}
}

@JS('window.ceraSpeakText')
external void _ceraSpeakText(JSString text);

@JS('window.ceraStopSpeech')
external void _ceraStopSpeech();

/// Synthesizes and speaks text using browser speech synthesis.
void speakText(String text) {
  try {
    _ceraSpeakText(text.toJS);
  } catch (_) {}
}

/// Stops any ongoing speech synthesis.
void stopSpeech() {
  try {
    _ceraStopSpeech();
  } catch (_) {}
}

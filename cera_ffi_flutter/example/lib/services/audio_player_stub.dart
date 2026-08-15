import 'dart:typed_data';

/// Stub audio player for non-web platforms.
Future<void> playAudioPcm(Float32List samples, int sampleRate) async {}

void stopAudioPlayback() {}

void startAudioStream(int sampleRate) {}

void appendAudioStreamChunk(Float32List chunk) {}

void stopAudioStream() {}

void speakText(String text) {}

void stopSpeech() {}

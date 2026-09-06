import 'dart:typed_data';

/// Stub audio player backend for unsupported platforms.
class AudioPlayerBackend {
  bool get isSupported => false;

  Future<void> playPcm(Float32List samples, int sampleRate) async {}

  void stopPlayback() {}

  void startStream(int sampleRate) {}

  void appendStreamChunk(Float32List chunk) {}

  void finishStream() {}

  void stopStream() {}

  void dispose() {}
}

bool get isAudioPlaybackSupported => false;

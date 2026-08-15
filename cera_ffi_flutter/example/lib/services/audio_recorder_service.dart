import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';

/// Service managing microphone capture for Cera multimodal audio models.
class AudioRecorderService {
  final AudioRecorder _recorder = AudioRecorder();
  final List<double> _accumulatedPcm = [];
  StreamSubscription<Uint8List>? _streamSub;
  bool _isRecording = false;

  bool get isRecording => _isRecording;

  /// Checks and requests microphone recording permissions.
  Future<bool> hasPermission() async {
    try {
      return await _recorder.hasPermission();
    } catch (_) {
      return false;
    }
  }

  /// Begins streaming 16-bit PCM audio from the microphone.
  Future<void> startRecording({int sampleRate = 16000}) async {
    if (_isRecording) return;
    _accumulatedPcm.clear();

    final hasPerm = await hasPermission();
    if (!hasPerm) {
      throw StateError('Microphone permission not granted');
    }

    final stream = await _recorder.startStream(
      RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: sampleRate,
        numChannels: 1,
      ),
    );

    _isRecording = true;
    _streamSub = stream.listen((chunk) {
      final samples = pcm16ToFloat32(chunk);
      _accumulatedPcm.addAll(samples);
    });
  }

  /// Stops recording and returns all accumulated normalized Float32 PCM samples.
  Future<List<double>> stopRecording() async {
    if (!_isRecording) return [];
    _isRecording = false;
    await _streamSub?.cancel();
    _streamSub = null;
    try {
      await _recorder.stop();
    } catch (_) {}
    return List<double>.from(_accumulatedPcm);
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
}

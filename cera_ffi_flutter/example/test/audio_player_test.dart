import 'dart:io';
import 'package:cera_ffi_flutter_example/services/audio_player_service.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('cera/audio_player'),
          (call) async {
            return null;
          },
        );
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('cera/audio_player'),
          null,
        );
  });

  group('AudioPlayerService', () {

    test('isSupported reflects active platform capabilities', () {
      final player = AudioPlayerService();
      // On macOS native and Web, audio playback is supported.
      if (Platform.isMacOS) {
        expect(player.isSupported, isTrue);
      }
      player.dispose();
    });

    test('isSourcePlaying returns false when idle', () {
      final player = AudioPlayerService();
      expect(player.isPlaying, isFalse);
      expect(player.activeAudioSource, isNull);
      expect(player.isSourcePlaying([0.1, 0.2]), isFalse);
      player.dispose();
    });

    test('playPcm handles empty samples gracefully without setting state', () async {
      final player = AudioPlayerService();
      bool listenerCalled = false;
      player.addListener(() {
        listenerCalled = true;
      });

      await player.playPcm([], sampleRate: 24000);
      expect(player.isPlaying, isFalse);
      expect(listenerCalled, isFalse);
      player.dispose();
    });

    test('startStream, appendChunk, and stop lifecycle', () {
      final player = AudioPlayerService();
      expect(player.isPlaying, isFalse);

      player.startStream(sampleRate: 24000);
      expect(player.isPlaying, isTrue);
      expect(player.activeAudioSource, same(player));

      // Appending valid chunks
      player.appendChunk([0.0, 0.5, -0.5]);
      expect(player.isPlaying, isTrue);

      player.stop();
      expect(player.isPlaying, isFalse);
      expect(player.activeAudioSource, isNull);

      player.dispose();
    });

    test('finishStream resets active stream state', () {
      final player = AudioPlayerService();
      player.startStream(sampleRate: 24000);
      expect(player.isPlaying, isTrue);

      player.finishStream();
      expect(player.isPlaying, isFalse);
      expect(player.activeAudioSource, isNull);

      player.dispose();
    });
  });
}

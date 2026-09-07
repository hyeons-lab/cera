import 'package:cera_ffi_flutter_example/services/audio_player_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(const MethodChannel('cera/audio_player'), (
          call,
        ) async {
          return null;
        });
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
      if (kIsWeb) {
        expect(player.isSupported, isTrue);
      } else {
        try {
          debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
          expect(player.isSupported, isTrue);
          debugDefaultTargetPlatformOverride = TargetPlatform.iOS;
          expect(player.isSupported, isTrue);
          debugDefaultTargetPlatformOverride = TargetPlatform.linux;
          expect(player.isSupported, isTrue);
          debugDefaultTargetPlatformOverride = TargetPlatform.android;
          expect(player.isSupported, isFalse);
          debugDefaultTargetPlatformOverride = TargetPlatform.windows;
          expect(player.isSupported, isFalse);
        } finally {
          debugDefaultTargetPlatformOverride = null;
        }
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

    test(
      'playPcm handles empty samples gracefully without setting state',
      () async {
        final player = AudioPlayerService();
        bool listenerCalled = false;
        player.addListener(() {
          listenerCalled = true;
        });

        await player.playPcm([], sampleRate: 24000);
        expect(player.isPlaying, isFalse);
        expect(listenerCalled, isFalse);
        player.dispose();
      },
    );

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

    test('playPcm followed immediately by stop resets state cleanly', () async {
      final player = AudioPlayerService();
      final future = player.playPcm([0.1, -0.1, 0.2], sampleRate: 24000);
      player.stop();
      await future;
      expect(player.isPlaying, isFalse);
      expect(player.activeAudioSource, isNull);

      player.dispose();
    });

    test('abrupt dispose during active stream terminates cleanly', () {
      final player = AudioPlayerService();
      player.startStream(sampleRate: 24000);
      player.appendChunk([0.1, 0.2, 0.3]);
      expect(player.isPlaying, isTrue);

      // Disposal while active should terminate playback and stream without throwing
      expect(() => player.dispose(), returnsNormally);
      expect(player.isPlaying, isFalse);
      expect(player.activeAudioSource, isNull);
    });

    test('abrupt dispose during active playPcm terminates cleanly', () async {
      final player = AudioPlayerService();
      final future = player.playPcm([0.1, 0.2, -0.1], sampleRate: 24000);
      expect(() => player.dispose(), returnsNormally);
      await future;
      expect(player.isPlaying, isFalse);
      expect(player.activeAudioSource, isNull);
    });

    test(
      'multiple AudioPlayerService instances have isolated backend state',
      () {
        final player1 = AudioPlayerService();
        final player2 = AudioPlayerService();

        player1.startStream(sampleRate: 24000);
        player1.appendChunk([0.1, 0.2]);
        expect(player1.isPlaying, isTrue);
        expect(player2.isPlaying, isFalse);

        player2.startStream(sampleRate: 16000);
        expect(player1.isPlaying, isTrue);
        expect(player2.isPlaying, isTrue);

        player1.stop();
        expect(player1.isPlaying, isFalse);
        expect(player2.isPlaying, isTrue);

        player2.stop();
        expect(player2.isPlaying, isFalse);

        player1.dispose();
        player2.dispose();
      },
    );

    test('native streaming platform channel calls on macOS and iOS', () async {
      final List<MethodCall> calls = [];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(const MethodChannel('cera/audio_player'), (
            call,
          ) async {
            calls.add(call);
            return null;
          });

      for (final platform in [TargetPlatform.macOS, TargetPlatform.iOS]) {
        calls.clear();
        try {
          debugDefaultTargetPlatformOverride = platform;
          final player = AudioPlayerService();

          player.startStream(sampleRate: 24000);
          expect(player.isPlaying, isTrue);
          await pumpEventQueue();

          player.appendChunk([0.1, -0.2, 0.3]);
          await pumpEventQueue();

          player.finishStream();
          expect(player.isPlaying, isFalse);
          await pumpEventQueue();

          player.stop();
          await pumpEventQueue();

          final methodNames = calls.map((c) => c.method).toList();
          expect(methodNames, contains('startStream'));
          expect(methodNames, contains('appendStreamChunk'));
          expect(methodNames, contains('finishStream'));
          expect(methodNames, contains('stopStream'));

          final startCall = calls.firstWhere((c) => c.method == 'startStream');
          expect(startCall.arguments, equals({'sampleRate': 24000}));

          final appendCall = calls.firstWhere(
            (c) => c.method == 'appendStreamChunk',
          );
          expect(
            appendCall.arguments['data'],
            equals(Float32List.fromList([0.1, -0.2, 0.3])),
          );

          player.dispose();
        } finally {
          debugDefaultTargetPlatformOverride = null;
        }
      }
    });

    test(
      'native streaming handles platform channel errors gracefully',
      () async {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(
              const MethodChannel('cera/audio_player'),
              (call) async {
                throw PlatformException(
                  code: 'AUDIO_ERROR',
                  message: 'Simulated failure',
                );
              },
            );

        try {
          debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
          final player = AudioPlayerService();

          // None of these should throw or crash the service
          expect(() => player.startStream(sampleRate: 24000), returnsNormally);
          await pumpEventQueue();

          expect(() => player.appendChunk([0.1, 0.2]), returnsNormally);
          await pumpEventQueue();

          expect(() => player.finishStream(), returnsNormally);
          await pumpEventQueue();

          expect(() => player.stop(), returnsNormally);
          await pumpEventQueue();

          player.dispose();
        } finally {
          debugDefaultTargetPlatformOverride = null;
        }
      },
    );

    test(
      'native streaming handles short streams below prebuffer threshold',
      () async {
        final List<MethodCall> calls = [];
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(
              const MethodChannel('cera/audio_player'),
              (call) async {
                calls.add(call);
                return null;
              },
            );

        try {
          debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
          final player = AudioPlayerService();

          player.startStream(sampleRate: 24000);
          await pumpEventQueue();

          // Single chunk below prebuffer threshold
          player.appendChunk([0.05, -0.05]);
          await pumpEventQueue();

          player.finishStream();
          expect(player.isPlaying, isFalse);
          await pumpEventQueue();

          final methodNames = calls.map((c) => c.method).toList();
          expect(
            methodNames,
            containsAllInOrder([
              'startStream',
              'appendStreamChunk',
              'finishStream',
            ]),
          );

          player.dispose();
        } finally {
          debugDefaultTargetPlatformOverride = null;
        }
      },
    );
  });
}

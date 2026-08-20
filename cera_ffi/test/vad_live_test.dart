@TestOn('vm')
library;

import 'dart:io';

import 'package:cera_ffi/cera_ffi.dart';
import 'package:test/test.dart';

String _platformDylibName() {
  if (Platform.isMacOS) return 'libcera_ffi.dylib';
  if (Platform.isWindows) return 'cera_ffi.dll';
  return 'libcera_ffi.so';
}

void main() {
  final dylibName = _platformDylibName();
  final dylibPath =
      File('../target/debug/$dylibName').existsSync()
          ? '../target/debug/$dylibName'
          : (File('../../target/debug/$dylibName').existsSync()
              ? '../../target/debug/$dylibName'
              : (File('../target/release/$dylibName').existsSync()
                  ? '../target/release/$dylibName'
                  : (File('../../target/release/$dylibName').existsSync()
                      ? '../../target/release/$dylibName'
                      : null)));

  final modelPath =
      File('../models/silero_vad.gguf').existsSync()
          ? '../models/silero_vad.gguf'
          : (File('../../models/silero_vad.gguf').existsSync()
              ? '../../models/silero_vad.gguf'
              : (File('models/silero_vad.gguf').existsSync()
                  ? 'models/silero_vad.gguf'
                  : null));

  group(
    'FfiSileroVad Live Dart FFI Tests',
    () {
      setUpAll(() {
        if (dylibPath != null) {
          configureDefaultBindings(libraryPath: dylibPath);
        }
      });

      test('Loads Silero VAD model from file and processes audio chunks', () {
        final vad = FfiSileroVad.fromFile(modelPath!);
        expect(vad.isClosed, isFalse);

        // Process 512 zeros (silence) at 16kHz
        final silenceChunk = List<double>.filled(512, 0.0);
        final prob1 = vad.processChunk(
          silenceChunk,
          FfiVadSampleRate.rate16kHz,
        );
        expect(prob1, isA<double>());
        expect(prob1, lessThan(0.1));

        // Reset state
        vad.reset();

        // Process 256 zeros (silence) at 8kHz
        final silence8k = List<double>.filled(256, 0.0);
        final prob2 = vad.processChunk(silence8k, FfiVadSampleRate.rate8kHz);
        expect(prob2, isA<double>());
        expect(prob2, lessThan(0.1));

        // Batch timestamps on silence
        final silenceLong = List<double>.filled(16000, 0.0);
        final timestamps = vad.getSpeechTimestamps(
          silenceLong,
          FfiVadSampleRate.rate16kHz,
          const FfiVadConfig(
            threshold: 0.5,
            negThreshold: 0.35,
            minSpeechDurationMs: 64,
            minSilenceDurationMs: 100,
            speechPadMs: 30,
          ),
        );
        expect(timestamps, isEmpty);

        vad.close();
        expect(vad.isClosed, isTrue);
      });

      test('Loads Silero VAD model from bytes', () {
        final bytes = File(modelPath!).readAsBytesSync();
        final vad = FfiSileroVad.fromBytes(bytes);
        expect(vad.isClosed, isFalse);

        final chunk = List<double>.filled(512, 0.0);
        final prob = vad.processChunk(chunk, FfiVadSampleRate.rate16kHz);
        expect(prob, lessThan(0.1));

        vad.close();
      });

      test('FfiVadIterator streams audio and emits speech boundary events', () {
        final vad = FfiSileroVad.fromFile(modelPath!);
        final iterator = FfiVadIterator.create(
          FfiVadSampleRate.rate16kHz,
          sileroVadDefaultConfig(),
        );

        // Process 10 silence chunks
        final silence = List<double>.filled(512, 0.0);
        for (int i = 0; i < 10; i++) {
          final ev = iterator.processChunk(vad, silence);
          expect(ev, isNull);
        }

        final flushed = iterator.flush();
        expect(flushed, isNull);

        iterator.reset();
        iterator.close();

        // Also verify custom FfiVadConfig instantiation
        final customIterator = FfiVadIterator.create(
          FfiVadSampleRate.rate16kHz,
          const FfiVadConfig(
            threshold: 0.6,
            negThreshold: 0.4,
            minSpeechDurationMs: 64,
            minSilenceDurationMs: 100,
            speechPadMs: 30,
          ),
        );
        for (int i = 0; i < 5; i++) {
          final ev = customIterator.processChunk(vad, silence);
          expect(ev, isNull);
        }
        customIterator.close();
        vad.close();
      });
    },
    skip:
        (dylibPath == null || modelPath == null)
            ? 'dylib or model fixture missing'
            : null,
  );
}

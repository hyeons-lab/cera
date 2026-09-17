import 'dart:async';

import '../cera_ffi.dart';

/// Extension on [ChatSession] providing an idiomatic asynchronous Dart [Stream].
extension ChatSessionStreaming on ChatSession {
  /// Streams generated text tokens as an asynchronous Dart [Stream].
  ///
  /// Cancelling the subscription signals wait-free cancellation to the underlying [ChatSession].
  Stream<String> stream(GenerateOpts opts) {
    late StreamController<String> controller;
    controller = StreamController<String>(
      onCancel: () {
        cancel();
      },
      onListen: () {
        final sink = _ChatSessionStreamSink(
          onText: (text) {
            if (!controller.isClosed) controller.add(text);
          },
          onDoneCallback: (reason) {
            if (!controller.isClosed) {
              if (reason is FinishReasonError) {
                controller.addError(StateError(reason.message));
              }
              controller.close();
            }
          },
        );
        generateStreamingAsync(opts, sink).then(
          (_) {
            if (!controller.isClosed) controller.close();
          },
          onError: (Object err, StackTrace stack) {
            if (!controller.isClosed) {
              controller.addError(err, stack);
              controller.close();
            }
          },
        );
      },
    );
    return controller.stream;
  }

  /// Streams generated text tokens conforming to a JSON Schema as an asynchronous Dart [Stream].
  Stream<String> streamJson(GenerateOpts opts, String schemaJson) {
    final grammar = jsonSchemaToGrammar(schemaJson);
    final constrainedOpts = GenerateOpts(
      maxTokens: opts.maxTokens,
      temperature: opts.temperature,
      topP: opts.topP,
      topK: opts.topK,
      minP: opts.minP,
      repetitionPenalty: opts.repetitionPenalty,
      stopTokens: opts.stopTokens,
      ignoreEos: opts.ignoreEos,
      grammar: grammar,
      grammarTriggerTokens: opts.grammarTriggerTokens,
      flushEveryTokens: opts.flushEveryTokens,
      flushEveryMs: opts.flushEveryMs,
      spec: opts.spec,
    );
    return stream(constrainedOpts);
  }
}

final class _ChatSessionStreamSink implements ModalitySink {
  _ChatSessionStreamSink({required this.onText, required this.onDoneCallback});

  final void Function(String) onText;
  final void Function(FinishReason) onDoneCallback;

  @override
  void onThoughtChunk(String text) {}

  @override
  void onTextChunk(String text) => onText(text);

  @override
  void onAudioFrames(List<double> pcm, int sampleRate) {}

  @override
  void onDone(FinishReason reason) => onDoneCallback(reason);
}

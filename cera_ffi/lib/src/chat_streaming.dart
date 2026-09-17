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
        generateStreamingAsync(opts, sink).then((_) {
          if (!controller.isClosed) controller.close();
        }, onError: (Object err, StackTrace stack) {
          if (!controller.isClosed) {
            controller.addError(err, stack);
            controller.close();
          }
        });
      },
    );
    return controller.stream;
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

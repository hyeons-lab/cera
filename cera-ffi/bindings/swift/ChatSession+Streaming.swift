import Foundation
#if canImport(CeraFFI)
import CeraFFI
#endif

extension ChatSession {
    /// Streams generated text tokens as an AsyncThrowingStream.
    ///
    /// Cancelling consumption of the stream signals wait-free cancellation to the underlying session.
    public func stream(opts: GenerateOpts) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            final class StreamSink: ModalitySink, @unchecked Sendable {
                let continuation: AsyncThrowingStream<String, Error>.Continuation
                private let lock = NSLock()
                private var _hasFinished = false

                var hasFinished: Bool {
                    lock.lock()
                    defer { lock.unlock() }
                    return _hasFinished
                }

                init(_ continuation: AsyncThrowingStream<String, Error>.Continuation) {
                    self.continuation = continuation
                }

                func onThoughtChunk(text: String) {}

                func onTextChunk(text: String) {
                    continuation.yield(text)
                }

                func onAudioFrames(pcm: [Float], sampleRate: UInt32) {}

                func onDone(reason: FinishReason) {
                    lock.lock()
                    _hasFinished = true
                    lock.unlock()
                    switch reason {
                    case .stop, .maxTokens, .contextFull, .cancelled, .grammarDeadEnd:
                        continuation.finish()
                    case .error(let msg):
                        continuation.finish(throwing: NSError(
                            domain: "CeraFFI",
                            code: 1,
                            userInfo: [NSLocalizedDescriptionKey: msg]
                        ))
                    }
                }
            }

            let sink = StreamSink(continuation)
            continuation.onTermination = { @Sendable [weak self] _ in
                self?.cancel()
            }

            Task { [weak self] in
                guard let self = self else {
                    continuation.finish()
                    return
                }
                do {
                    _ = try await self.generateStreamingAsync(opts: opts, sink: sink)
                } catch {
                    if !sink.hasFinished {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    /// Streams generated text tokens conforming to a JSON Schema as an AsyncThrowingStream.
    public func streamJson(opts: GenerateOpts, schemaJson: String) -> AsyncThrowingStream<String, Error> {
        do {
            let grammar = try jsonSchemaToGrammar(schemaJson: schemaJson)
            var constrainedOpts = opts
            constrainedOpts.grammar = grammar
            return stream(opts: constrainedOpts)
        } catch {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: error)
            }
        }
    }
}

extension GenerateOpts {
    /// Returns a copy of GenerateOpts with JSON Schema constrained grammar.
    public func withJsonSchema(_ schemaJson: String) throws -> GenerateOpts {
        var opts = self
        opts.grammar = try jsonSchemaToGrammar(schemaJson: schemaJson)
        return opts
    }
}


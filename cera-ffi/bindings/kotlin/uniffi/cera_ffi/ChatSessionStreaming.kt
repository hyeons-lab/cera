@file:Suppress("PackageName")

package uniffi.cera_ffi

import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

/**
 * Streams generation output text tokens as a cold [Flow].
 *
 * Cancelling flow collection triggers wait-free cancellation of the in-flight decode
 * on the underlying [ChatSession].
 */
fun ChatSession.stream(opts: GenerateOpts): Flow<String> =
    callbackFlow {
        val finished = java.util.concurrent.atomic.AtomicBoolean(false)
        val terminalError = java.util.concurrent.atomic.AtomicReference<RuntimeException?>(null)
        val sink =
            object : ModalitySink {
                override fun onThoughtChunk(text: String) {}

                override fun onTextChunk(text: String) {
                    trySendBlocking(text)
                }

                override fun onAudioFrames(
                    pcm: List<Float>,
                    sampleRate: UInt,
                ) {}

                override fun onDone(reason: FinishReason) {
                    if (reason is FinishReason.Error) {
                        terminalError.set(RuntimeException(reason.message))
                    }
                }
            }

        val job =
            launch {
                try {
                    generateStreamingAsync(opts, sink)
                    // The callback precedes completion of the native future. Closing
                    // earlier can cancel that future and leave its cancellation flag set.
                    finished.set(true)
                    close(terminalError.get())
                } catch (e: Exception) {
                    finished.set(true)
                    close(e)
                }
            }

        awaitClose {
            if (!finished.get()) {
                this@stream.cancel()
                job.cancel()
            }
        }
    }

/**
 * Constrain generation options with a JSON Schema definition string.
 */
fun GenerateOpts.withJsonSchema(schemaJson: String): GenerateOpts = copy(grammar = jsonSchemaToGrammar(schemaJson))

/**
 * Streams generation output text tokens conforming to a JSON Schema as a cold [Flow].
 */
fun ChatSession.streamJson(
    opts: GenerateOpts,
    schemaJson: String,
): Flow<String> = stream(opts.withJsonSchema(schemaJson))

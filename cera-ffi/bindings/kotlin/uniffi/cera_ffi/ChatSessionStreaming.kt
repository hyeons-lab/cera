@file:Suppress("PackageName")

package uniffi.cera_ffi

import kotlinx.coroutines.channels.awaitClose
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
        val sink =
            object : ModalitySink {
                override fun onThoughtChunk(text: String) {}

                override fun onTextChunk(text: String) {
                    trySend(text)
                }

                override fun onAudioFrames(
                    pcm: List<Float>,
                    sampleRate: UInt,
                ) {}

                override fun onDone(reason: FinishReason) {
                    when (reason) {
                        is FinishReason.Error -> close(RuntimeException(reason.message))
                        else -> close()
                    }
                }
            }

        val job =
            launch {
                try {
                    generateStreamingAsync(opts, sink)
                } catch (e: Exception) {
                    close(e)
                }
            }

        awaitClose {
            cancel()
            job.cancel()
        }
    }

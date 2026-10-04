package com.hyeonslab.cera.probe

import uniffi.cera_ffi.FfiAudioPipeline
import uniffi.cera_ffi.FfiAudioPipelineEvent
import kotlin.math.abs

/** What the runner needs from the audio pipeline; [FfiPipeline] is the real one. */
interface AudioPipelinePort : AutoCloseable {
    /** Feed `count` samples of 16 kHz mono PCM16. */
    fun process(pcm: ShortArray, count: Int): List<FfiAudioPipelineEvent>

    /** End of audio: finish the utterance in flight and label everything still waiting. */
    fun flush(): List<FfiAudioPipelineEvent>
}

class FfiPipeline(
    private val pipeline: FfiAudioPipeline,
    chunkSamples: Int = CHUNK_SAMPLES,
) : AudioPipelinePort {
    private var bytes = ByteArray(chunkSamples * 2)

    override fun process(pcm: ShortArray, count: Int): List<FfiAudioPipelineEvent> {
        // The runner sets the chunk length; grow rather than assume it matches this hint.
        if (bytes.size < count * 2) bytes = ByteArray(count * 2)
        pcm16ToLeBytes(pcm, count, bytes)
        // The exact length: the binding copies the whole array.
        return pipeline.processChunkPcm16(if (count * 2 == bytes.size) bytes else bytes.copyOf(count * 2))
    }

    override fun flush(): List<FfiAudioPipelineEvent> = pipeline.flush()

    override fun close() = pipeline.close()
}

/**
 * Reads chunks of [chunkSamples] samples from [source], feeds them to [pipeline] and hands every
 * event to [onEvent].
 * Owns neither: the caller closes them. Pure Kotlin, so it runs in a JVM unit test with a fake
 * source and pipeline.
 *
 * [onProgress] gets the audio seconds processed so far and the largest absolute sample since the
 * previous report, every [progressEverySeconds] seconds of audio. The service reports CPU time per
 * audio second with it, and the peak shows whether the microphone is live: a recorder Android
 * silences in the background delivers all zeros, which is easy to mistake for a quiet room.
 */
class PipelineRunner(
    private val source: AudioSource,
    private val pipeline: AudioPipelinePort,
    private val onEvent: (FfiAudioPipelineEvent) -> Unit,
    private val onProgress: (audioSeconds: Double, peak: Int) -> Unit = { _, _ -> },
    private val progressEverySeconds: Int = 60,
    private val chunkSamples: Int = CHUNK_SAMPLES,
) {
    /** Samples handed to the pipeline so far. */
    var samples: Long = 0
        private set

    /**
     * Run until [shouldStop] returns true, then flush. [shouldStop] is checked between chunks, so
     * a stop takes effect within one read. Throws if the source fails while running; the pipeline
     * is not flushed then, because a capture error says nothing about the audio already
     * processed. A failure that lands after a stop was requested (the service closes the source
     * to unblock a stuck read) instead finishes like end-of-stream: it is the stop, not an error.
     */
    fun run(shouldStop: () -> Boolean) {
        val pcm = ShortArray(chunkSamples)
        var nextProgress = progressEverySeconds.toLong() * SAMPLE_RATE
        var peak = 0
        while (!shouldStop()) {
            var filled = 0
            while (filled < chunkSamples) {
                val n = source.read(pcm, filled, chunkSamples - filled)
                if (n == AudioSource.END_OF_STREAM || (n < 0 && shouldStop())) {
                    // The source ran out, or a stop closed it mid-read: process the partial
                    // chunk, then flush.
                    if (filled > 0) {
                        for (i in 0 until filled) peak = maxOf(peak, abs(pcm[i].toInt()))
                        pipeline.process(pcm, filled).forEach(onEvent)
                        samples += filled
                    }
                    pipeline.flush().forEach(onEvent)
                    return
                }
                if (n < 0) throw IllegalStateException("audio capture failed with code $n")
                filled += n
                if (shouldStop() && filled == 0) break
            }
            if (filled < chunkSamples) break
            for (i in 0 until filled) peak = maxOf(peak, abs(pcm[i].toInt()))
            pipeline.process(pcm, filled).forEach(onEvent)
            samples += filled
            if (samples >= nextProgress) {
                onProgress(samples.toDouble() / SAMPLE_RATE, peak)
                peak = 0
                nextProgress += progressEverySeconds.toLong() * SAMPLE_RATE
            }
        }
        pipeline.flush().forEach(onEvent)
    }
}

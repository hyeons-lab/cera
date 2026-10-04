package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import uniffi.cera_ffi.FfiAudioPipelineEvent

/**
 * Hands out `chunks` full chunks, `piece` samples per read, then reports `endCode`. [onDrained]
 * runs when the last sample is handed out, which is where a test raises its stop flag (the service's
 * stop flag is a plain idempotent flag; a call-counting lambda would be consumed by the runner's
 * own checks).
 */
private class FakeSource(
    chunks: Int,
    private val piece: Int = CHUNK_SAMPLES,
    private val endCode: Int = -1,
    private val onDrained: () -> Unit = {},
) : AudioSource {
    private var left = chunks * CHUNK_SAMPLES

    override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
        if (left == 0) return endCode
        val n = minOf(length, piece, left)
        for (i in 0 until n) buffer[offset + i] = 16384
        left -= n
        if (left == 0) onDrained()
        return n
    }

    override fun close() {}
}

private class FakePipeline : AudioPipelinePort {
    val chunks = mutableListOf<ShortArray>()
    var flushed = 0

    override fun process(pcm: ShortArray, count: Int): List<FfiAudioPipelineEvent> {
        chunks += pcm.copyOf(count)
        return if (chunks.size % 2 == 0) {
            listOf(FfiAudioPipelineEvent.SpeechStart(chunks.size.toULong(), 0f))
        } else {
            emptyList()
        }
    }

    override fun flush(): List<FfiAudioPipelineEvent> {
        flushed++
        return listOf(FfiAudioPipelineEvent.SpeechEnd(0uL, 1uL, 0f, 0f))
    }

    override fun close() {}
}

class PipelineRunnerTest {
    @Test
    fun partial_reads_are_assembled_into_whole_chunks() {
        val pipeline = FakePipeline()
        var stop = false
        val source = FakeSource(chunks = 3, piece = 400, onDrained = { stop = true })
        val runner = PipelineRunner(source, pipeline, onEvent = {})
        runner.run { stop }
        assertEquals(3, pipeline.chunks.size)
        assertTrue(pipeline.chunks.all { it.size == CHUNK_SAMPLES && it[0] == 16384.toShort() })
        assertEquals(3L * CHUNK_SAMPLES, runner.samples)
    }

    @Test
    fun events_are_delivered_and_the_stream_is_flushed_on_a_clean_stop() {
        val pipeline = FakePipeline()
        val events = mutableListOf<FfiAudioPipelineEvent>()
        var stop = false
        PipelineRunner(
            FakeSource(chunks = 4, onDrained = { stop = true }),
            pipeline,
            onEvent = { events += it },
        ).run { stop }
        assertEquals(1, pipeline.flushed)
        // Two SpeechStart events from the chunks, then the flush's SpeechEnd.
        assertEquals(3, events.size)
        assertTrue(events.last() is FfiAudioPipelineEvent.SpeechEnd)
    }

    @Test
    fun a_capture_error_throws_and_does_not_flush() {
        val pipeline = FakePipeline()
        val runner = PipelineRunner(FakeSource(chunks = 1, endCode = -3), pipeline, onEvent = {})
        val e = assertThrows(IllegalStateException::class.java) { runner.run { false } }
        assertTrue(e.message!!.contains("-3"))
        assertEquals(0, pipeline.flushed)
        assertEquals(1, pipeline.chunks.size)
    }

    @Test
    fun a_capture_error_after_a_stop_finishes_cleanly_like_end_of_stream() {
        val pipeline = FakePipeline()
        var stop = false
        // The service asked to stop, then closed the source mid-read: the read fails with the
        // stop already requested, which is the stop, not an error.
        val source = object : AudioSource {
            var reads = 0
            override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
                reads++
                if (reads > 1) return -3
                for (i in 0 until 100) buffer[offset + i] = 5
                stop = true
                return 100
            }

            override fun close() {}
        }
        val runner = PipelineRunner(source, pipeline, onEvent = {})
        runner.run { stop }
        assertEquals(1, pipeline.chunks.size)
        assertEquals(100, pipeline.chunks[0].size)
        assertEquals(1, pipeline.flushed)
        assertEquals(100L, runner.samples)
    }

    @Test
    fun a_stop_before_any_audio_flushes_without_processing() {
        val pipeline = FakePipeline()
        PipelineRunner(FakeSource(chunks = 5), pipeline, onEvent = {}).run { true }
        assertEquals(0, pipeline.chunks.size)
        assertEquals(1, pipeline.flushed)
    }

    @Test
    fun progress_is_reported_every_interval_of_audio() {
        val progress = mutableListOf<Double>()
        val peaks = mutableListOf<Int>()
        var stop = false
        // Report every second of audio, over 2.5 s worth of chunks.
        val perSecond = SAMPLE_RATE / CHUNK_SAMPLES
        PipelineRunner(
            FakeSource(chunks = perSecond * 5 / 2, onDrained = { stop = true }),
            FakePipeline(),
            onEvent = {},
            onProgress = { audio, peak ->
                progress += audio
                peaks += peak
            },
            progressEverySeconds = 1,
        ).run { stop }
        assertEquals(listOf(1.0, 2.0), progress)
        // The fake plays a constant 16384; the peak is per interval, not a running maximum.
        assertEquals(listOf(16384, 16384), peaks)
    }

    @Test
    fun the_peak_is_reset_after_each_report() {
        val peaks = mutableListOf<Int>()
        var stop = false
        val perSecond = SAMPLE_RATE / CHUNK_SAMPLES
        var delivered = 0
        // Loud (-30000) in the first second, silent after: the second report must say 0.
        val source = object : AudioSource {
            override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
                val loud = delivered < perSecond * CHUNK_SAMPLES
                for (i in 0 until length) buffer[offset + i] = if (loud) -30000 else 0
                delivered += length
                if (delivered >= 2 * perSecond * CHUNK_SAMPLES) stop = true
                return length
            }

            override fun close() {}
        }
        PipelineRunner(
            source,
            FakePipeline(),
            onEvent = {},
            onProgress = { _, peak -> peaks += peak },
            progressEverySeconds = 1,
        ).run { stop }
        assertEquals(listOf(30000, 0), peaks)
    }

    @Test
    fun the_end_of_a_finite_source_processes_the_partial_chunk_and_flushes_cleanly() {
        val pipeline = FakePipeline()
        val partial = object : AudioSource {
            var given = false
            override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
                if (given) return AudioSource.END_OF_STREAM
                given = true
                for (i in 0 until 100) buffer[offset + i] = 5
                return 100
            }

            override fun close() {}
        }
        val runner = PipelineRunner(partial, pipeline, onEvent = {})
        runner.run { false }
        assertEquals(1, pipeline.chunks.size)
        assertEquals(100, pipeline.chunks[0].size)
        assertEquals(1, pipeline.flushed)
        assertEquals(100L, runner.samples)
    }
}

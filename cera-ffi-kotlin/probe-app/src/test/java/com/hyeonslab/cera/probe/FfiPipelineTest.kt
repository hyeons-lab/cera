package com.hyeonslab.cera.probe

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test
import uniffi.cera_ffi.FfiAudioPipeline
import uniffi.cera_ffi.FfiAudioPipelineEvent
import uniffi.cera_ffi.NoHandle

private class CapturingPipeline : FfiAudioPipeline(NoHandle) {
    val sent = mutableListOf<ByteArray>()

    override fun processChunkPcm16(pcm: ByteArray): List<FfiAudioPipelineEvent> {
        sent += pcm
        return emptyList()
    }
}

class FfiPipelineTest {
    @Test
    fun a_full_chunk_reuses_the_scratch_buffer_without_copying() {
        val backend = CapturingPipeline()
        val port = FfiPipeline(backend, chunkSamples = 8)
        val pcm = ShortArray(8) { (it + 1).toShort() }
        port.process(pcm, 8)
        port.process(pcm, 8)
        assertEquals(2, backend.sent.size)
        assertSame(backend.sent[0], backend.sent[1])
        assertEquals(16, backend.sent[0].size)
        assertEquals(1, backend.sent[0][0].toInt())
        assertEquals(0, backend.sent[0][1].toInt())
    }

    @Test
    fun a_partial_chunk_is_trimmed_to_its_exact_length() {
        val backend = CapturingPipeline()
        val port = FfiPipeline(backend, chunkSamples = 8)
        port.process(shortArrayOf(1, -2, 300), 3)
        assertEquals(1, backend.sent.size)
        assertArrayEquals(
            byteArrayOf(0x01, 0x00, 0xFE.toByte(), 0xFF.toByte(), 0x2C, 0x01),
            backend.sent[0],
        )
    }

    @Test
    fun a_chunk_past_the_hint_grows_the_buffer() {
        val backend = CapturingPipeline()
        val port = FfiPipeline(backend, chunkSamples = 8)
        port.process(ShortArray(16) { 7 }, 16)
        assertEquals(1, backend.sent.size)
        assertEquals(32, backend.sent[0].size)
    }
}

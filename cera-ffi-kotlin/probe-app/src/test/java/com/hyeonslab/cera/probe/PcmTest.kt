package com.hyeonslab.cera.probe

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class PcmTest {
    @Test
    fun samples_are_written_little_endian() {
        val out = ByteArray(8)
        pcm16ToLeBytes(shortArrayOf(Short.MIN_VALUE, 0, Short.MAX_VALUE, 0x4001), 4, out)
        assertArrayEquals(
            byteArrayOf(0x00, 0x80.toByte(), 0x00, 0x00, 0xFF.toByte(), 0x7F, 0x01, 0x40),
            out,
        )
    }

    @Test
    fun negative_samples_keep_their_sign_bytes() {
        val out = ByteArray(2)
        pcm16ToLeBytes(shortArrayOf(-2), 1, out)
        assertArrayEquals(byteArrayOf(0xFE.toByte(), 0xFF.toByte()), out)
    }

    @Test
    fun only_count_samples_are_written() {
        val out = ByteArray(6) { 7 }
        pcm16ToLeBytes(shortArrayOf(1, 2, 3), 1, out)
        assertEquals(1, out[0].toInt())
        assertEquals(7, out[2].toInt())
    }

    @Test
    fun a_count_past_the_buffers_is_refused() {
        assertThrows(IllegalArgumentException::class.java) {
            pcm16ToLeBytes(ShortArray(2), 3, ByteArray(6))
        }
        assertThrows(IllegalArgumentException::class.java) {
            pcm16ToLeBytes(ShortArray(3), 3, ByteArray(5))
        }
    }

    @Test
    fun the_chunk_length_is_held_to_its_bounds() {
        assertEquals(1_600, chunkSamplesForMs(100))
        assertEquals(8_000, chunkSamplesForMs(500))
        assertEquals("below the minimum", 1_600, chunkSamplesForMs(1))
        assertEquals("above the maximum", 32_000, chunkSamplesForMs(60_000))
    }
}

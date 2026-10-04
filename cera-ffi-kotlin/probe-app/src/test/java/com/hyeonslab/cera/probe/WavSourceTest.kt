package com.hyeonslab.cera.probe

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.FileOutputStream

private fun le16(v: Int) = byteArrayOf(v.toByte(), (v shr 8).toByte())

private fun le32(v: Long) = le16((v and 0xFFFF).toInt()) + le16((v shr 16).toInt())

private fun chunk(id: String, body: ByteArray): ByteArray =
    id.toByteArray() + le32(body.size.toLong()) + body + if (body.size % 2 == 1) byteArrayOf(0) else byteArrayOf()

private fun fmt(rate: Int = 16_000, channels: Int = 1, bits: Int = 16, format: Int = 1) =
    chunk("fmt ", le16(format) + le16(channels) + le32(rate.toLong()) + le32(0) + le16(0) + le16(bits))

private fun wav(vararg chunks: ByteArray): ByteArray {
    val body = ByteArrayOutputStream()
    body.write("WAVE".toByteArray())
    chunks.forEach { body.write(it) }
    return "RIFF".toByteArray() + le32(body.size().toLong()) + body.toByteArray()
}

private fun samplesBytes(vararg s: Int) = s.fold(byteArrayOf()) { acc, v -> acc + le16(v) }

class WavSourceTest {
    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun a_plain_wav_yields_its_samples() {
        val file = wav(fmt(), chunk("data", samplesBytes(1, -2, 300)))
        assertArrayEquals(shortArrayOf(1, -2, 300), WavSource.parse(file))
    }

    @Test
    fun chunks_before_data_are_skipped_including_odd_sized_ones() {
        val file = wav(fmt(), chunk("LIST", ByteArray(5)), chunk("data", samplesBytes(7, 8)))
        assertArrayEquals(shortArrayOf(7, 8), WavSource.parse(file))
    }

    @Test
    fun a_streamed_size_of_zero_takes_whatever_follows() {
        val data = "data".toByteArray() + le32(0) + samplesBytes(5, 6, 7)
        val file = "RIFF".toByteArray() + le32(0) + "WAVE".toByteArray() + fmt() + data
        assertArrayEquals(shortArrayOf(5, 6, 7), WavSource.parse(file))
    }

    @Test
    fun the_wrong_format_is_refused_with_what_was_found() {
        val stereo = wav(fmt(channels = 2), chunk("data", samplesBytes(1, 2)))
        assertEquals(
            "need 16 kHz mono 16-bit PCM, got format=1 channels=2 rate=16000 bits=16",
            assertThrows(IllegalArgumentException::class.java) { WavSource.parse(stereo) }.message,
        )
        val rate = wav(fmt(rate = 44_100), chunk("data", samplesBytes(1)))
        assertThrows(IllegalArgumentException::class.java) { WavSource.parse(rate) }
        val float = wav(fmt(format = 3, bits = 32), chunk("data", samplesBytes(1, 2)))
        assertThrows(IllegalArgumentException::class.java) { WavSource.parse(float) }
    }

    @Test
    fun garbage_and_missing_chunks_are_refused() {
        assertThrows(IllegalArgumentException::class.java) { WavSource.parse(ByteArray(40)) }
        assertThrows(IllegalArgumentException::class.java) { WavSource.parse(wav(fmt())) }
        assertThrows(IllegalArgumentException::class.java) {
            WavSource.parse(wav(chunk("data", samplesBytes(1))))
        }
    }

    @Test
    fun a_chunk_declaring_more_than_remains_is_refused() {
        // 0x7FFFFFF0 narrowed to a negative step and threw an index crash; 0xFFFFFFF8 never
        // advanced the walker at all and hung. Both must refuse loudly now.
        for (declared in listOf(0x7FFFFFF0L, 0xFFFFFFF8L)) {
            val junk = "JUNK".toByteArray() + le32(declared) + ByteArray(4)
            val e = assertThrows(IllegalArgumentException::class.java) {
                WavSource.parse(wav(fmt(), junk))
            }
            assertTrue("$declared", e.message!!.contains("declares $declared bytes"))
        }
    }

    @Test
    fun a_file_past_the_replay_cap_is_refused() {
        val big = tmp.newFile("big.wav")
        // Just over the 64 MB cap, in 1 MB writes. Zeros would fail RIFF parsing anyway, so
        // the cap message is what proves the refusal came from the cap.
        FileOutputStream(big).use { out ->
            val mb = ByteArray(1024 * 1024)
            repeat(64) { out.write(mb) }
            out.write(0)
        }
        val e = assertThrows(IllegalArgumentException::class.java) { WavSource.open(big) }
        assertTrue(e.message!!, e.message!!.contains("replay cap"))
    }

    @Test
    fun a_negative_or_non_finite_speed_is_refused() {
        assertThrows(IllegalArgumentException::class.java) {
            WavSource(ShortArray(10), speed = -1.0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            WavSource(ShortArray(10), speed = Double.NaN)
        }
        assertThrows(IllegalArgumentException::class.java) {
            WavSource(ShortArray(10), speed = Double.POSITIVE_INFINITY)
        }
    }

    @Test
    fun reads_are_paced_to_real_time_and_end_with_end_of_stream() {
        var now = 0L
        val sleeps = mutableListOf<Long>()
        val src = WavSource(
            ShortArray(SAMPLE_RATE) { 1 }, // one second
            speed = 1.0,
            nowNanos = { now },
            sleep = { ms -> sleeps += ms; now += ms * 1_000_000 },
        )
        val buf = ShortArray(SAMPLE_RATE / 2)
        assertEquals(buf.size, src.read(buf, 0, buf.size))
        assertEquals(buf.size, src.read(buf, 0, buf.size))
        // The first half second is due at 500 ms, the second at 1000 ms.
        assertEquals(listOf(500L, 500L), sleeps)
        assertEquals(AudioSource.END_OF_STREAM, src.read(buf, 0, buf.size))
    }

    @Test
    fun speed_zero_does_not_wait() {
        val sleeps = mutableListOf<Long>()
        val src = WavSource(ShortArray(100), speed = 0.0, sleep = { sleeps += it })
        src.read(ShortArray(100), 0, 100)
        assertEquals(emptyList<Long>(), sleeps)
    }
}

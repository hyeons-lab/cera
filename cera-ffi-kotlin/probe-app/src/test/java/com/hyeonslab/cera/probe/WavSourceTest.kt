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
        // One leg falsifying exactly each conjunct of the combined guard: the float leg
        // above falsifies format and bits together, and 0xFFFE never reaches this guard.
        val formatOnly = wav(fmt(format = 3, bits = 16), chunk("data", samplesBytes(1, 2)))
        assertEquals(
            "need 16 kHz mono 16-bit PCM, got format=3 channels=1 rate=16000 bits=16",
            assertThrows(IllegalArgumentException::class.java) { WavSource.parse(formatOnly) }.message,
        )
        val bitsOnly = wav(fmt(bits = 8), chunk("data", samplesBytes(1, 2)))
        assertEquals(
            "need 16 kHz mono 16-bit PCM, got format=1 channels=1 rate=16000 bits=8",
            assertThrows(IllegalArgumentException::class.java) { WavSource.parse(bitsOnly) }.message,
        )
        // The extensible container names its true cause instead of misreporting the audio,
        // which may genuinely be 16 kHz mono 16-bit.
        val extensible = wav(fmt(format = 0xFFFE), chunk("data", samplesBytes(1, 2)))
        assertEquals(
            "WAVEFORMATEXTENSIBLE is not supported; convert to plain 16 kHz mono 16-bit PCM " +
                "(channels=1 rate=16000 bits=16)",
            assertThrows(IllegalArgumentException::class.java) { WavSource.parse(extensible) }.message,
        )
    }

    @Test
    fun garbage_and_missing_chunks_are_refused() {
        // Each leg names its guard: a throw-only assert passes whichever guard fires, so a
        // deleted header guard survives behind the later "no data chunk" refusal.
        assertEquals(
            "not a RIFF/WAVE file",
            assertThrows(IllegalArgumentException::class.java) { WavSource.parse(ByteArray(40)) }.message,
        )
        assertEquals(
            "no data chunk",
            assertThrows(IllegalArgumentException::class.java) { WavSource.parse(wav(fmt())) }.message,
        )
        assertEquals(
            "data chunk before the fmt chunk",
            assertThrows(IllegalArgumentException::class.java) {
                WavSource.parse(wav(chunk("data", samplesBytes(1))))
            }.message,
        )
    }

    @Test
    fun a_truncated_fmt_chunk_is_refused() {
        val short = "fmt ".toByteArray() + le32(8) + ByteArray(8)
        assertEquals(
            "truncated fmt chunk",
            assertThrows(IllegalArgumentException::class.java) {
                WavSource.parse(wav(short, chunk("data", samplesBytes(1, 2))))
            }.message,
        )
    }

    @Test
    fun a_chunk_declaring_more_than_remains_is_refused() {
        // 0x7FFFFFF0 narrowed to a negative step and threw an index crash; 0xFFFFFFF8 never
        // advanced the walker at all and hung. Both must refuse loudly now, as must the
        // maximum u32.
        for (declared in listOf(0x7FFFFFF0L, 0xFFFFFFF8L, 0xFFFFFFFFL)) {
            val junk = "JUNK".toByteArray() + le32(declared) + ByteArray(4)
            val e = assertThrows(IllegalArgumentException::class.java) {
                WavSource.parse(wav(fmt(), junk))
            }
            assertTrue("$declared", e.message!!.contains("declares $declared bytes"))
        }
    }

    @Test
    fun a_streamed_size_of_max_u32_takes_whatever_follows() {
        // Only the data chunk may declare more than remains: recorders that stream the file
        // write the maximum size up front.
        val body = samplesBytes(-5, 6)
        val file = wav(fmt(), "data".toByteArray() + le32(0xFFFFFFFFL) + body)
        assertArrayEquals(shortArrayOf(-5, 6), WavSource.parse(file))
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
    fun open_reads_the_file_through_the_parser() {
        val wavFile = tmp.newFile("speech.wav")
        FileOutputStream(wavFile).use { it.write(wav(fmt(), chunk("data", samplesBytes(1, -2, 300)))) }
        val src = WavSource.open(wavFile, speed = 0.0)
        val buf = ShortArray(8)
        assertEquals(3, src.read(buf, 0, buf.size))
        assertArrayEquals(shortArrayOf(1, -2, 300), buf.copyOf(3))
        assertEquals(AudioSource.END_OF_STREAM, src.read(buf, 0, buf.size))
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
        // Below the 0.1 floor (other than 0, as fast as possible) sleeps minutes per read.
        assertThrows(IllegalArgumentException::class.java) {
            WavSource(ShortArray(10), speed = 0.05)
        }
        WavSource(ShortArray(10), speed = 0.1)
        WavSource(ShortArray(10), speed = 0.0)
    }

    @Test
    fun a_close_during_a_paced_read_returns_minus_one_at_once() {
        val sleeps = mutableListOf<Long>()
        var src: WavSource? = null
        src = WavSource(ShortArray(SAMPLE_RATE), speed = 0.1, sleep = { ms -> sleeps += ms; src?.close() })
        // The first 100 ms slice closes the source, so the 5 s wait ends after one slice.
        assertEquals(-1, src.read(ShortArray(8000), 0, 8000))
        assertEquals(listOf(100L), sleeps)
        // And a closed source stays closed.
        assertEquals(-1, src.read(ShortArray(8000), 0, 8000))
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
        // The first half second is due at 500 ms, the second at 1000 ms, each slept in
        // 100 ms slices so a close cuts the wait short.
        assertEquals(List(10) { 100L }, sleeps)
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

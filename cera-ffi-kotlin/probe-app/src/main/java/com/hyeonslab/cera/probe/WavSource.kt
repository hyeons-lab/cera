package com.hyeonslab.cera.probe

import java.io.File

/**
 * Plays a 16 kHz mono 16-bit WAV file through the service as if it were the microphone, which is
 * how the speech path (VAD, Whisper, diarizer) is exercised on a device without making noise in
 * the room. The whole file is read into memory up front.
 *
 * [speed] is the playback rate relative to real time: 1.0 paces reads like a live recording, so
 * CPU-per-audio-second numbers mean what they do for the microphone; 0 reads as fast as the
 * pipeline consumes. [sleep] and [nowNanos] are parameters so pacing is testable.
 */
class WavSource(
    private val pcm: ShortArray,
    private val speed: Double = 1.0,
    private val nowNanos: () -> Long = System::nanoTime,
    private val sleep: (millis: Long) -> Unit = Thread::sleep,
) : AudioSource {
    private var position = 0
    private var startNanos = -1L

    override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
        if (position >= pcm.size) return AudioSource.END_OF_STREAM
        val n = minOf(length, pcm.size - position)
        if (speed > 0) {
            if (startNanos < 0) startNanos = nowNanos()
            // Deliver sample `position + n` no earlier than it would have been recorded.
            val dueNanos = ((position + n) * 1e9 / SAMPLE_RATE / speed).toLong()
            val waitMs = (dueNanos - (nowNanos() - startNanos)) / 1_000_000
            if (waitMs > 0) sleep(waitMs)
        }
        System.arraycopy(pcm, position, buffer, offset, n)
        position += n
        return n
    }

    override fun close() {}

    companion object {
        fun open(file: File, speed: Double = 1.0): WavSource = WavSource(parse(file.readBytes()), speed)

        /**
         * The samples of a 16 kHz mono 16-bit PCM WAV. Walks the RIFF chunks rather than assuming
         * a 44-byte header, since tools add LIST and other chunks before `data`.
         */
        fun parse(bytes: ByteArray): ShortArray {
            fun u16(at: Int) = (bytes[at].toInt() and 0xFF) or ((bytes[at + 1].toInt() and 0xFF) shl 8)
            fun u32(at: Int) = u16(at).toLong() or (u16(at + 2).toLong() shl 16)
            require(bytes.size >= 12 && String(bytes, 0, 4) == "RIFF" && String(bytes, 8, 4) == "WAVE") {
                "not a RIFF/WAVE file"
            }
            var at = 12
            var format: IntArray? = null // audio format, channels, rate, bits
            while (at + 8 <= bytes.size) {
                val id = String(bytes, at, 4)
                val size = u32(at + 4)
                val body = at + 8
                when (id) {
                    "fmt " -> {
                        require(size >= 16 && body + 16 <= bytes.size) { "truncated fmt chunk" }
                        format = intArrayOf(u16(body), u16(body + 2), u32(body + 4).toInt(), u16(body + 14))
                    }
                    "data" -> {
                        val f = requireNotNull(format) { "data chunk before the fmt chunk" }
                        require(f[0] == 1 && f[1] == 1 && f[2] == SAMPLE_RATE && f[3] == 16) {
                            "need 16 kHz mono 16-bit PCM, got format=${f[0]} channels=${f[1]} " +
                                "rate=${f[2]} bits=${f[3]}"
                        }
                        // A streamed WAV may declare a size of 0 or 0xFFFFFFFF: take what is there.
                        val available = (bytes.size - body).toLong()
                        val length = if (size == 0L || size > available) available else size
                        val samples = (length / 2).toInt()
                        return ShortArray(samples) { i -> u16(body + 2 * i).toShort() }
                    }
                }
                at = body + size.toInt() + (size.toInt() and 1) // chunks are word aligned
            }
            throw IllegalArgumentException("no data chunk")
        }
    }
}

package com.hyeonslab.cera.probe

import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Plays a 16 kHz mono 16-bit WAV file through the service as if it were the microphone, which is
 * how the speech path (VAD, Whisper, diarizer) is exercised on a device without making noise in
 * the room. The whole file is read into memory up front.
 *
 * [speed] is the playback rate relative to real time: 1.0 paces reads like a live recording, so
 * CPU-per-audio-second numbers mean what they do for the microphone; 0 reads as fast as the
 * pipeline consumes. Speeds below 0.1 (other than 0) are refused: they would sleep minutes per
 * read with no timely stop. [sleep] and [nowNanos] are parameters so pacing is testable.
 */
class WavSource(
    private val pcm: ShortArray,
    private val speed: Double = 1.0,
    private val nowNanos: () -> Long = System::nanoTime,
    private val sleep: (millis: Long) -> Unit = Thread::sleep,
) : AudioSource {
    private var position = 0
    private var startNanos = -1L
    private val closed = AtomicBoolean(false)

    init {
        requireValidSpeed(speed)
    }

    override fun read(buffer: ShortArray, offset: Int, length: Int): Int {
        if (closed.get()) return -1
        if (position >= pcm.size) return AudioSource.END_OF_STREAM
        val n = minOf(length, pcm.size - position)
        if (speed > 0) {
            if (startNanos < 0) startNanos = nowNanos()
            // Deliver sample `position + n` no earlier than it would have been recorded.
            val dueNanos = ((position + n) * 1e9 / SAMPLE_RATE / speed).toLong()
            // Sleep in slices so a close (a stop) cuts the wait short instead of riding it out.
            var remaining = (dueNanos - (nowNanos() - startNanos)) / 1_000_000
            while (remaining > 0 && !closed.get()) {
                val slice = minOf(remaining, 100)
                sleep(slice)
                remaining -= slice
            }
            if (closed.get()) return -1
        }
        System.arraycopy(pcm, position, buffer, offset, n)
        position += n
        return n
    }

    override fun close() {
        closed.set(true)
    }

    companion object {
        /** Replay files are capped: the whole file is buffered, and the path is intent input. */
        private const val MAX_WAV_BYTES = 64 * 1024 * 1024 // ~35 min at 16 kHz mono 16-bit

        /** Slowest paced replay besides 0 (as fast as possible): bounds the worst sleep to 10x. */
        private const val MIN_SPEED = 0.1

        internal fun requireValidSpeed(speed: Double) {
            require(speed == 0.0 || (speed >= MIN_SPEED && speed.isFinite())) {
                "speed must be 0 (as fast as possible) or at least $MIN_SPEED, got $speed"
            }
        }

        fun open(file: File, speed: Double = 1.0): WavSource {
            requireValidSpeed(speed)
            // Streamed with a running cap rather than readBytes(): the length of a special file
            // cannot be trusted, and readBytes would hold the whole thing before any check runs.
            // (A manual loop because readNBytes needs API 33 and minSdk is 28.)
            val out = ByteArrayOutputStream()
            val buf = ByteArray(8192)
            file.inputStream().buffered().use { input ->
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    require(out.size() + n <= MAX_WAV_BYTES) {
                        "wav file exceeds the $MAX_WAV_BYTES-byte replay cap"
                    }
                    out.write(buf, 0, n)
                }
            }
            return WavSource(parse(out.toByteArray()), speed)
        }

        /**
         * The samples of a 16 kHz mono 16-bit PCM WAV. Walks the RIFF chunks rather than assuming
         * a 44-byte header, since tools add LIST and other chunks before `data`.
         */
        fun parse(bytes: ByteArray): ShortArray {
            fun u16(at: Int) = (bytes[at].toInt() and 0xFF) or ((bytes[at + 1].toInt() and 0xFF) shl 8)
            fun u32(at: Int) = u16(at).toLong() or (u16(at + 2).toLong() shl 16)
            require(
                bytes.size >= 12 &&
                    String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
                    String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE",
            ) {
                "not a RIFF/WAVE file"
            }
            var at = 12
            var format: IntArray? = null // audio format, channels, rate, bits
            while (at + 8 <= bytes.size) {
                val id = String(bytes, at, 4, Charsets.US_ASCII)
                val size = u32(at + 4)
                val body = at + 8
                when (id) {
                    "fmt " -> {
                        require(size >= 16 && body + 16 <= bytes.size) { "truncated fmt chunk" }
                        format = intArrayOf(u16(body), u16(body + 2), u32(body + 4).toInt(), u16(body + 14))
                    }
                    "data" -> {
                        val f = requireNotNull(format) { "data chunk before the fmt chunk" }
                        require(f[0] != 0xFFFE) {
                            "WAVEFORMATEXTENSIBLE is not supported; convert to plain 16 kHz mono " +
                                "16-bit PCM (channels=${f[1]} rate=${f[2]} bits=${f[3]})"
                        }
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
                // A corrupt size must refuse loudly, not narrow to a negative step (which wedged
                // the walk or threw an index crash instead of the documented refusal).
                require(size <= bytes.size - body) {
                    "chunk $id declares $size bytes but only ${bytes.size - body} remain"
                }
                // Chunks are word aligned. Long math so the step itself can never wrap, clamped
                // to the size so the final narrowing cannot wrap either (any step past the end
                // exits the loop identically).
                at = minOf(body.toLong() + size + (size and 1L), bytes.size.toLong()).toInt()
            }
            throw IllegalArgumentException("no data chunk")
        }
    }
}

package com.hyeonslab.cera.probe

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicBoolean

/** A blocking source of 16 kHz mono PCM16. */
interface AudioSource : AutoCloseable {
    /**
     * Fill `buffer[offset until offset + length]`, blocking until at least one sample arrives.
     * Returns the number of samples read, [END_OF_STREAM] when a finite source has run out, or
     * another negative value on a capture error.
     */
    fun read(buffer: ShortArray, offset: Int, length: Int): Int

    companion object {
        /** Returned by [read] once a finite source (a file) is exhausted: stop cleanly. */
        const val END_OF_STREAM = Int.MIN_VALUE
    }
}

/**
 * The device microphone. Needs `RECORD_AUDIO` granted, and on Android 14 and later a foreground
 * service of type `microphone` that was started while the app was visible.
 */
class MicSource private constructor(private val record: AudioRecord) : AudioSource {
    // onDestroy closes the source to unblock a stuck read, and the worker's finally closes it
    // again: exactly one of them must release the recorder.
    private val closed = AtomicBoolean(false)

    override fun read(buffer: ShortArray, offset: Int, length: Int): Int =
        record.read(buffer, offset, length)

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        runCatching { record.stop() }
        record.release()
    }

    companion object {
        /** Seconds of audio the capture buffer holds, so a slow pipeline call drops nothing. */
        private const val BUFFER_SECONDS = 2

        @SuppressLint("MissingPermission") // the caller checks RECORD_AUDIO before starting
        fun open(): MicSource {
            val minBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            check(minBytes > 0) { "AudioRecord rejected 16 kHz mono PCM16 (min buffer $minBytes)" }
            val bytes = maxOf(minBytes, SAMPLE_RATE * 2 * BUFFER_SECONDS)
            val record = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bytes,
            )
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                record.release()
                error("AudioRecord failed to initialize (is RECORD_AUDIO granted?)")
            }
            record.startRecording()
            check(record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                record.release()
                "AudioRecord did not start recording (another app holds the microphone?)"
            }
            return MicSource(record)
        }
    }
}

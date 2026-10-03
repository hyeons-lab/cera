package com.hyeonslab.cera.probe

/** The pipeline's input format: 16 kHz mono. */
const val SAMPLE_RATE = 16_000

/**
 * Samples per pipeline call by default: 500 ms. Every FFI call pays a fixed cost for JNA to build
 * its call-status and buffer structures, about five JNA calls and a few milliseconds of CPU on a
 * phone. At 100 ms chunks that was roughly 60% of the whole service's CPU; the pipeline tracks
 * sample positions itself, so a larger chunk only delays events by up to its own length.
 */
const val CHUNK_SAMPLES = 8_000

/** Bounds for the `chunk_ms` setting: below 100 ms the call overhead dominates, above 2 s events lag. */
const val MIN_CHUNK_MS = 100
const val MAX_CHUNK_MS = 2_000

/** Samples in a chunk of [ms] milliseconds, with [ms] held to [MIN_CHUNK_MS]..[MAX_CHUNK_MS]. */
fun chunkSamplesForMs(ms: Int): Int = ms.coerceIn(MIN_CHUNK_MS, MAX_CHUNK_MS) * SAMPLE_RATE / 1000

/**
 * Writes `count` samples of [src] into [dst] as 16-bit little-endian bytes, the format
 * `FfiAudioPipeline.processChunkPcm16` takes. A byte array crosses the FFI in one copy; the
 * `List<Float>` that `processChunk` takes is lowered element by element, which cost about 0.05
 * CPU-seconds per audio second on a phone.
 */
fun pcm16ToLeBytes(src: ShortArray, count: Int, dst: ByteArray) {
    require(count <= src.size && count * 2 <= dst.size) { "count $count exceeds the buffers" }
    for (i in 0 until count) {
        val v = src[i].toInt()
        dst[2 * i] = v.toByte()
        dst[2 * i + 1] = (v shr 8).toByte()
    }
}

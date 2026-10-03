package com.hyeonslab.cera.probe

/**
 * CPU time this process spent per second of audio, which is the number an always-on service lives
 * or dies by: Android demotes background CPU work, so every CPU-second counts twice. [cpuMillis]
 * reads the process CPU clock (user plus system, all threads), [Process.getElapsedCpuTime] on a
 * device; it is a parameter so the arithmetic is testable.
 */
class CpuMeter(private val cpuMillis: () -> Long) {
    private val startCpu = cpuMillis()
    private var lastCpu = startCpu
    private var lastAudio = 0.0

    /** CPU milliseconds per audio second since the previous call (or the start). */
    data class Interval(val cpuMs: Long, val audioSeconds: Double) {
        val cpuSecondsPerAudioSecond: Double
            get() = if (audioSeconds > 0) cpuMs / 1000.0 / audioSeconds else 0.0
    }

    /** Close the interval ending at [audioSeconds] of audio processed in total. */
    fun lap(audioSeconds: Double): Interval {
        val now = cpuMillis()
        val interval = Interval(now - lastCpu, audioSeconds - lastAudio)
        lastCpu = now
        lastAudio = audioSeconds
        return interval
    }

    /** CPU milliseconds since the meter was created. */
    fun totalCpuMs(): Long = cpuMillis() - startCpu
}

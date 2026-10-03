package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Test

class CpuMeterTest {
    @Test
    fun laps_report_the_cpu_spent_in_each_interval() {
        var cpu = 1_000L
        val meter = CpuMeter { cpu }
        cpu += 300
        val first = meter.lap(60.0)
        assertEquals(300, first.cpuMs)
        assertEquals(60.0, first.audioSeconds, 0.0)
        assertEquals(0.005, first.cpuSecondsPerAudioSecond, 1e-9)
        cpu += 120
        val second = meter.lap(120.0)
        assertEquals(120, second.cpuMs)
        assertEquals(60.0, second.audioSeconds, 0.0)
        assertEquals(0.002, second.cpuSecondsPerAudioSecond, 1e-9)
        assertEquals(420, meter.totalCpuMs())
    }

    @Test
    fun an_interval_without_audio_is_zero_not_a_division_by_zero() {
        val meter = CpuMeter { 5L }
        assertEquals(0.0, meter.lap(0.0).cpuSecondsPerAudioSecond, 0.0)
    }
}

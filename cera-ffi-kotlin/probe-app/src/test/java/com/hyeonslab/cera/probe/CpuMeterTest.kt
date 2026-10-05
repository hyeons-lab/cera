package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Locale

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
        // A backwards clock reports zero too, instead of a negative rate.
        var cpu = 0L
        val skewed = CpuMeter { cpu }
        skewed.lap(10.0)
        cpu = 100 // CPU advanced while the audio clock went backwards
        assertEquals(0.0, skewed.lap(5.0).cpuSecondsPerAudioSecond, 0.0)
    }

    @Test
    fun the_stats_line_uses_us_decimals_regardless_of_device_locale() {
        // %.4f discriminates under comma-decimal (decimal separator), %.0f under
        // digit-shaping (shaped digits); between the two every Locale.US is load-bearing.
        for (locale in listOf(Locale.GERMANY, Locale.forLanguageTag("ar-EG"))) {
            withLocale(locale) {
                val line = statsLine(60.0, CpuMeter.Interval(300, 60.0), 420, 16384, false)
                assertEquals(
                    "stats audio=60s interval_cpu=300ms/60s cpu_per_audio_s=0.0050 " +
                        "total_cpu=420ms peak=16384 silenced=false",
                    line,
                )
            }
        }
    }
}

package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import uniffi.cera_ffi.FfiAudioPipelineEvent

class EventLogTest {
    private fun labeled(speaker: UInt?, overlapping: UInt? = null, text: String = " hello ") =
        FfiAudioPipelineEvent.UtteranceLabeled(text, 65_000f, 66_000f, speaker, 0.9f, overlapping)

    @Test
    fun speakers_are_numbered_from_one() {
        assertEquals("1:05 S1: hello", eventLine(labeled(0u)))
        assertEquals("1:05 S3 (+S2): hello", eventLine(labeled(2u, 1u)))
        assertEquals("1:05 S?: hello", eventLine(labeled(null)))
    }

    @Test
    fun speech_boundaries_are_not_shown_or_kept() {
        val start = FfiAudioPipelineEvent.SpeechStart(16_000uL, 1_000f)
        val end = FfiAudioPipelineEvent.SpeechEnd(0uL, 16_000uL, 0f, 1_000f)
        assertNull(eventLine(start))
        assertNull(eventLine(end))
        assertNull(eventJson(start))
        assertNull(eventJson(end))
    }

    @Test
    fun a_wake_word_line_names_the_keyword() {
        val ev = FfiAudioPipelineEvent.WakeWordDetected("Hey Liquid", 0.93f, 2_000f, 32_000uL)
        assertEquals("0:02 wake word \"Hey Liquid\" (0.93)", eventLine(ev))
    }

    @Test
    fun json_escapes_quotes_backslashes_and_control_characters() {
        assertEquals("\"a\\\"b\\\\c\\nd\\u0001\"", jsonString("a\"b\\c\nd\u0001"))
    }

    @Test
    fun a_labeled_utterance_is_one_json_line() {
        assertEquals(
            "{\"type\":\"utterance\",\"start_ms\":65000.0,\"end_ms\":66000.0,\"speaker\":1," +
                "\"text\":\"say \\\"hi\\\"\"}",
            eventJson(labeled(1u, text = "say \"hi\"")),
        )
        assertEquals(
            "{\"type\":\"utterance\",\"start_ms\":65000.0,\"end_ms\":66000.0,\"speaker\":null," +
                "\"text\":\"x\"}",
            eventJson(labeled(null, text = "x")),
        )
    }
}

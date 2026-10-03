package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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

    @Test
    fun whispers_non_speech_tags_are_recognised() {
        for (tag in listOf("[BLANK_AUDIO]", " [MUSIC PLAYING] ", "(applause)", "[Music]")) {
            assertTrue(tag, isNonSpeechTag(tag))
        }
        for (words in listOf("hello", "I said [BLANK_AUDIO] twice", "[]", "", "(a) (b)")) {
            assertFalse(words, isNonSpeechTag(words))
        }
    }

    @Test
    fun a_non_speech_tag_is_neither_shown_nor_saved() {
        val tag = FfiAudioPipelineEvent.UtteranceTranscribed("[BLANK_AUDIO]", 0f, 1000f, 16_000uL)
        assertNull(eventLine(tag))
        assertNull(eventJson(tag))
        assertNull(eventLine(labeled(0u, text = "[MUSIC PLAYING]")))
        assertNull(eventJson(labeled(0u, text = "[MUSIC PLAYING]")))
        // Real speech that merely mentions brackets is kept.
        assertEquals("1:05 S1: see [1] for details", eventLine(labeled(0u, text = "see [1] for details")))
    }
}

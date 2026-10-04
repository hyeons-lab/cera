package com.hyeonslab.cera.probe

import uniffi.cera_ffi.FfiAudioPipelineEvent
import java.util.Locale

/**
 * Whether [text] is only one of Whisper's bracketed non-speech tags, such as `[BLANK_AUDIO]`,
 * `[MUSIC PLAYING]` or `(applause)`. Whisper emits them for noise and silence the VAD let through;
 * they are not something anyone said, so they are neither shown nor saved.
 */
fun isNonSpeechTag(text: String): Boolean = NON_SPEECH_TAG.matches(text.trim())

private val NON_SPEECH_TAG = Regex("""\[[^\[\]]{1,40}\]|\([^()]{1,40}\)""")

/** Whether [event] carries a non-speech tag that is neither shown nor saved. */
private fun isDroppedTag(event: FfiAudioPipelineEvent): Boolean = when (event) {
    is FfiAudioPipelineEvent.UtteranceTranscribed -> isNonSpeechTag(event.text)
    is FfiAudioPipelineEvent.UtteranceLabeled -> isNonSpeechTag(event.text)
    is FfiAudioPipelineEvent.WakeWordDetected -> false
    is FfiAudioPipelineEvent.SpeechStart -> false
    is FfiAudioPipelineEvent.SpeechEnd -> false
}

/**
 * One line per event that is worth showing or keeping: transcripts and speaker labels, wake words.
 * Speech boundaries and Whisper's non-speech tags are bookkeeping and return null.
 */
fun eventLine(event: FfiAudioPipelineEvent): String? =
    if (isDroppedTag(event)) null else speechLine(event)

private fun speechLine(event: FfiAudioPipelineEvent): String? = when (event) {
    is FfiAudioPipelineEvent.UtteranceTranscribed ->
        "${stamp(event.startMs)} ${event.text.trim()}"
    is FfiAudioPipelineEvent.UtteranceLabeled -> {
        val who = event.speaker?.let { "S${it + 1u}" } ?: "S?"
        val also = event.overlapping?.let { " (+S${it + 1u})" } ?: ""
        "${stamp(event.startMs)} $who$also: ${event.text.trim()}"
    }
    is FfiAudioPipelineEvent.WakeWordDetected ->
        "${stamp(event.timestampMs)} wake word \"${event.keyword}\" " +
        "(${String.format(Locale.US, "%.2f", event.confidence)})"
    is FfiAudioPipelineEvent.SpeechStart -> null
    is FfiAudioPipelineEvent.SpeechEnd -> null
}

/** `m:ss` of a pipeline timestamp in milliseconds; negative and non-finite input renders `0:00`. */
fun stamp(ms: Float): String {
    val total = if (ms.isFinite()) (ms / 1000f).toLong().coerceAtLeast(0) else 0L
    return String.format(Locale.US, "%d:%02d", total / 60, total % 60)
}

/** [text] as a JSON string literal, quotes included. */
fun jsonString(text: String): String = buildString {
    append('"')
    for (c in text) {
        when {
            c == '"' -> append("\\\"")
            c == '\\' -> append("\\\\")
            c == '\n' -> append("\\n")
            c == '\r' -> append("\\r")
            c == '\t' -> append("\\t")
            c < ' ' -> append("\\u%04x".format(c.code))
            else -> append(c)
        }
    }
    append('"')
}

/**
 * A transcript record for `transcript.jsonl`, or null for an event that is not kept.
 * `speaker` and `overlapping` keep the diarizer's zero-based ids; [eventLine] shows them
 * one-based (`S1`, …). The text is kept raw here while [eventLine] trims it for display.
 */
fun eventJson(event: FfiAudioPipelineEvent): String? =
    if (isDroppedTag(event)) null else speechJson(event)

private fun speechJson(event: FfiAudioPipelineEvent): String? = when (event) {
    is FfiAudioPipelineEvent.UtteranceLabeled ->
        "{\"type\":\"utterance\",\"start_ms\":${jsonFloat(event.startMs)},\"end_ms\":${jsonFloat(event.endMs)}," +
            "\"speaker\":${event.speaker ?: "null"}," +
            "\"overlapping\":${event.overlapping ?: "null"}," +
            "\"confidence\":${event.confidence?.let(::jsonFloat) ?: "null"}," +
            "\"text\":${jsonString(event.text)}}"
    is FfiAudioPipelineEvent.UtteranceTranscribed ->
        "{\"type\":\"transcript\",\"start_ms\":${jsonFloat(event.startMs)},\"end_ms\":${jsonFloat(event.endMs)}," +
            "\"text\":${jsonString(event.text)}}"
    is FfiAudioPipelineEvent.WakeWordDetected ->
        "{\"type\":\"wake_word\",\"start_ms\":${jsonFloat(event.timestampMs)}," +
            "\"keyword\":${jsonString(event.keyword)},\"confidence\":${jsonFloat(event.confidence)}}"
    is FfiAudioPipelineEvent.SpeechStart -> null
    is FfiAudioPipelineEvent.SpeechEnd -> null
}

/** A float as JSON: non-finite values are not numbers in JSON, so they become null. */
private fun jsonFloat(value: Float): String = if (value.isFinite()) value.toString() else "null"

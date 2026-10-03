package com.hyeonslab.cera.probe

import uniffi.cera_ffi.FfiAudioPipelineEvent

/**
 * Whether [text] is only one of Whisper's bracketed non-speech tags, such as `[BLANK_AUDIO]`,
 * `[MUSIC PLAYING]` or `(applause)`. Whisper emits them for noise and silence the VAD let through;
 * they are not something anyone said, so they are neither shown nor saved.
 */
fun isNonSpeechTag(text: String): Boolean = NON_SPEECH_TAG.matches(text.trim())

private val NON_SPEECH_TAG = Regex("""[\[(][^\])]{1,40}[\])]""")

/**
 * One line per event that is worth showing or keeping: transcripts and speaker labels, wake words.
 * Speech boundaries and Whisper's non-speech tags are bookkeeping and return null.
 */
fun eventLine(event: FfiAudioPipelineEvent): String? = when {
    event is FfiAudioPipelineEvent.UtteranceTranscribed && isNonSpeechTag(event.text) -> null
    event is FfiAudioPipelineEvent.UtteranceLabeled && isNonSpeechTag(event.text) -> null
    else -> speechLine(event)
}

private fun speechLine(event: FfiAudioPipelineEvent): String? = when (event) {
    is FfiAudioPipelineEvent.UtteranceTranscribed ->
        "${stamp(event.startMs)} ${event.text.trim()}"
    is FfiAudioPipelineEvent.UtteranceLabeled -> {
        val who = event.speaker?.let { "S${it + 1u}" } ?: "S?"
        val also = event.overlapping?.let { " (+S${it + 1u})" } ?: ""
        "${stamp(event.startMs)} $who$also: ${event.text.trim()}"
    }
    is FfiAudioPipelineEvent.WakeWordDetected ->
        "${stamp(event.timestampMs)} wake word \"${event.keyword}\" (${"%.2f".format(event.confidence)})"
    else -> null
}

/** `m:ss` of a pipeline timestamp in milliseconds. */
fun stamp(ms: Float): String {
    val total = (ms / 1000f).toLong().coerceAtLeast(0)
    return "%d:%02d".format(total / 60, total % 60)
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

/** A transcript record for `transcript.jsonl`, or null for an event that is not kept. */
fun eventJson(event: FfiAudioPipelineEvent): String? = when {
    event is FfiAudioPipelineEvent.UtteranceTranscribed && isNonSpeechTag(event.text) -> null
    event is FfiAudioPipelineEvent.UtteranceLabeled && isNonSpeechTag(event.text) -> null
    else -> speechJson(event)
}

private fun speechJson(event: FfiAudioPipelineEvent): String? = when (event) {
    is FfiAudioPipelineEvent.UtteranceLabeled ->
        "{\"type\":\"utterance\",\"start_ms\":${event.startMs},\"end_ms\":${event.endMs}," +
            "\"speaker\":${event.speaker ?: "null"},\"text\":${jsonString(event.text)}}"
    is FfiAudioPipelineEvent.UtteranceTranscribed ->
        "{\"type\":\"transcript\",\"start_ms\":${event.startMs},\"end_ms\":${event.endMs}," +
            "\"text\":${jsonString(event.text)}}"
    is FfiAudioPipelineEvent.WakeWordDetected ->
        "{\"type\":\"wake_word\",\"ms\":${event.timestampMs},\"keyword\":${jsonString(event.keyword)}}"
    else -> null
}

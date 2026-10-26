package com.hyeonslab.cera.decision

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.DecisionEngine

/**
 * Thin coroutine-friendly owner of a [DecisionEngine]: a public d1 decision model that answers
 * typed `choice`, `score` and `noul` questions about a state in one forward pass each, with
 * nothing generated.
 *
 * The request and the response are the JSON bodies of llama.cpp's `POST /v1/systemone`:
 * `{"state": ..., "images": [...], "questions": {id: {"type", "instructions", "criteria"}}}` in,
 * `{"answers": {id: {...}}, "usage": {...}}` out. An `audio/` data URL in `images` is a speech
 * clip (`d1-omni` only).
 *
 * One request runs at a time per client; a second [decide] waits for the first. Close it when
 * done: it holds the model.
 */
class DecisionClient private constructor(
    private val engine: DecisionEngine,
) : AutoCloseable {
    /** The model type from the file: `lfm2-d1-omni`, `lfm2-d1` or `converted`. */
    val decisionType: String get() = engine.decisionType()

    /** Whether image requests can be answered (the model's vision tower is loaded). */
    val supportsImages: Boolean get() = engine.supportsImages()

    /** Whether speech requests can be answered (the model's speech tower is loaded). */
    val supportsAudio: Boolean get() = engine.supportsAudio()

    /** The backend the model runs on. Always `cpu` in this version. */
    val backendName: String get() = engine.backendName()

    /**
     * Answer a `/v1/systemone` request and return the response JSON, off the calling thread.
     *
     * Throws `uniffi.cera_ffi.FfiException` for a request the model cannot accept (the message
     * says what is wrong) and for backend errors.
     *
     * Cancelling the calling coroutine stops the request before its next question; the forward
     * pass in flight finishes first, so the worker stays busy for up to one question (a few
     * hundred milliseconds for text, a few seconds for a large image) and a following [decide]
     * waits for it.
     */
    suspend fun decide(requestJson: String): String = engine.decideJsonAsync(requestJson)

    /** Release the model. The client must not be used afterwards. */
    override fun close() = engine.close()

    companion object {
        /**
         * Load a decision model on [Dispatchers.IO]. [mmprojPath] is the `mmproj` GGUF of a
         * `d1-omni` model, which carries its vision and speech towers; without it the model
         * answers text only. [backend] must be [BackendPreference.CPU] or
         * [BackendPreference.AUTO] (which runs on the CPU): the d1 head and towers have no
         * accelerated path yet, and an accelerator is refused with an `FfiException`.
         */
        suspend fun open(
            modelPath: String,
            mmprojPath: String? = null,
            backend: BackendPreference = BackendPreference.CPU,
        ): DecisionClient =
            withContext(Dispatchers.IO) {
                DecisionClient(DecisionEngine.openAsync(modelPath, mmprojPath, backend))
            }
    }
}

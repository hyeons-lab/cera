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

    /** The backend the trunk runs on: `cpu`, `metal`, `wgpu` or `hexagon`. */
    val backendName: String get() = engine.backendName()

    /**
     * Where each part of the model runs (trunk, head, vision and speech towers), and why an
     * [BackendPreference.AUTO] open ended on the CPU if it did. For logs; not a stable format.
     */
    val backendReport: String get() = engine.backendReport()

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
         * Load a decision model on [Dispatchers.IO]. [mmprojPath] is the `mmproj` GGUF of the
         * model, which carries its vision tower (and, for `d1-omni`, its speech tower); without
         * it the model answers text only.
         *
         * [backend] is a request. [BackendPreference.CPU], [BackendPreference.GPU] (wgpu),
         * [BackendPreference.METAL] and [BackendPreference.HEXAGON] / [BackendPreference.NPU]
         * open on that backend or throw `FfiException` with the reason. [BackendPreference.AUTO]
         * takes the best accelerator that loads the model and answers a warm-up request, else
         * the CPU; see [backendReport] for what it chose and why.
         */
        suspend fun open(
            modelPath: String,
            mmprojPath: String? = null,
            backend: BackendPreference = BackendPreference.AUTO,
        ): DecisionClient =
            withContext(Dispatchers.IO) {
                DecisionClient(DecisionEngine.openAsync(modelPath, mmprojPath, backend))
            }
    }
}

package com.hyeonslab.cera.probe

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * What the service is doing, for the UI to show. The service and the activity share a process, so
 * a plain in-memory flow is enough: the transcript itself is persisted to `transcript.jsonl`, and
 * this holds only the most recent lines.
 */
object AudioServiceState {
    enum class Status { STOPPED, STARTING, RUNNING, FAILED }

    data class Snapshot(
        val status: Status = Status.STOPPED,
        /** One line about the pipeline (stages, where each runs) or the failure. */
        val detail: String = "",
        val lines: List<String> = emptyList(),
    )

    private const val MAX_LINES = 200

    private val mutable = MutableStateFlow(Snapshot())
    val state: StateFlow<Snapshot> = mutable

    fun status(status: Status, detail: String) =
        mutable.update { it.copy(status = status, detail = detail) }

    fun line(text: String) =
        mutable.update { it.copy(lines = (it.lines + text).takeLast(MAX_LINES)) }
}

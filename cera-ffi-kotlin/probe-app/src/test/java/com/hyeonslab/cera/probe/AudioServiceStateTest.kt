package com.hyeonslab.cera.probe

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Test

class AudioServiceStateTest {
    @After
    fun restoreInitialState() {
        AudioServiceState.reset()
    }

    @Test
    fun lines_keep_only_the_latest_two_hundred() {
        // 250 fresh lines push out anything earlier tests appended, so the window is exact.
        repeat(250) { AudioServiceState.line("l$it") }
        val lines = AudioServiceState.state.value.lines
        assertEquals(200, lines.size)
        assertEquals("l50", lines.first())
        assertEquals("l249", lines.last())
    }

    @Test
    fun reset_restores_the_initial_snapshot() {
        AudioServiceState.status(AudioServiceState.Status.RUNNING, "detail")
        AudioServiceState.line("x")
        AudioServiceState.reset()
        val state = AudioServiceState.state.value
        assertEquals(AudioServiceState.Status.STOPPED, state.status)
        assertEquals("", state.detail)
        assertEquals(emptyList<String>(), state.lines)
    }

    @Test
    fun the_status_is_the_latest_set_and_a_stop_owns_failures() {
        AudioServiceState.status(AudioServiceState.Status.RUNNING, "detail")
        val state = AudioServiceState.state.value
        assertEquals(AudioServiceState.Status.RUNNING, state.status)
        assertEquals("detail", state.detail)
        assertEquals(AudioServiceState.Status.STOPPED, failureStatus(stopWasRequested = true))
        assertEquals(AudioServiceState.Status.FAILED, failureStatus(stopWasRequested = false))
    }

    @Test
    fun destroy_outcome_prefers_a_recorded_failure_over_the_swallow_computation() {
        // A published FAILED stays untouched.
        assertEquals(null, destroyOutcome(AudioServiceState.Status.FAILED, "boom", 7, true, 7))
        // A recorded failure beats both the swallowed hint and the blank stop: a replacement
        // start landing between a worker failure and onDestroy must not bury the detail.
        assertEquals(
            AudioServiceState.Status.FAILED to "boom",
            destroyOutcome(AudioServiceState.Status.STARTING, "boom", 7, true, 8),
        )
        assertEquals(
            AudioServiceState.Status.FAILED to "boom",
            destroyOutcome(AudioServiceState.Status.STOPPED, "boom", -1, false, -1),
        )
        // With no failure the swallow computation stands.
        assertEquals(
            AudioServiceState.Status.STOPPED to "stopped while starting; tap Start again",
            destroyOutcome(AudioServiceState.Status.STARTING, null, 7, true, -1),
        )
        // The owner-mismatch route reaches the same hint without the flag.
        assertEquals(
            AudioServiceState.Status.STOPPED to "stopped while starting; tap Start again",
            destroyOutcome(AudioServiceState.Status.STARTING, null, 7, false, 8),
        )
        assertEquals(
            AudioServiceState.Status.STOPPED to "",
            destroyOutcome(AudioServiceState.Status.RUNNING, null, -1, false, -1),
        )
    }
}

package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.FileNotFoundException

class TranscriptRotationTest {
    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun a_log_past_the_cap_rotates_aside_and_a_small_log_is_left_alone() {
        val name = AudioPipelineService.TRANSCRIPT_FILE
        val smallDir = tmp.newFolder()
        val small = File(smallDir, name).apply { writeText("x\n") }
        assertTrue(AudioPipelineService.rotateTranscriptIfNeeded(small))
        assertTrue(small.exists())
        val bigDir = tmp.newFolder()
        val big = File(bigDir, name).apply { writeBytes(ByteArray(10_000_001)) }
        assertTrue(AudioPipelineService.rotateTranscriptIfNeeded(big))
        assertFalse(big.exists())
        assertEquals(10_000_001, File(bigDir, "$name.1").length())
        // Exactly at the cap stays: the rotation trips only past it.
        val exactDir = tmp.newFolder()
        val exact = File(exactDir, name).apply { writeBytes(ByteArray(10_000_000)) }
        assertTrue(AudioPipelineService.rotateTranscriptIfNeeded(exact))
        assertTrue(exact.exists())
    }

    @Test
    fun a_failed_rotation_reports_false_and_keeps_the_original() {
        val name = AudioPipelineService.TRANSCRIPT_FILE
        val dir = tmp.newFolder()
        val big = File(dir, name).apply { writeBytes(ByteArray(10_000_001)) }
        // A non-empty directory at the rotation target fails the rename on every platform.
        val squat = File(dir, "$name.1").apply { mkdir() }
        File(squat, "child").writeText("x")
        assertFalse(AudioPipelineService.rotateTranscriptIfNeeded(big))
        assertTrue(big.exists())
    }

    @Test
    fun storing_a_record_past_the_cap_rotates_then_appends_through_the_seam() {
        val name = AudioPipelineService.TRANSCRIPT_FILE
        val dir = tmp.newFolder()
        val big = File(dir, name).apply { writeBytes(ByteArray(10_000_001)) }
        val (rotated, fault) = AudioPipelineService.storeTranscriptRecord(big, "{}")
        assertTrue(rotated)
        assertNull(fault)
        assertEquals(10_000_001, File(dir, "$name.1").length())
        assertEquals("{}\n", big.readText())
    }

    @Test
    fun storing_a_record_with_rotation_blocked_reports_false_and_keeps_the_original() {
        val name = AudioPipelineService.TRANSCRIPT_FILE
        val dir = tmp.newFolder()
        val big = File(dir, name).apply { writeBytes(ByteArray(10_000_001)) }
        // A non-empty directory at the rotation target fails the rename on every platform.
        val squat = File(dir, "$name.1").apply { mkdir() }
        File(squat, "child").writeText("x")
        val (rotated, fault) = AudioPipelineService.storeTranscriptRecord(big, "{}")
        assertFalse(rotated)
        assertNull(fault)
        assertTrue(big.exists())
    }

    @Test
    fun a_failed_append_is_returned_not_thrown() {
        val ok = File(tmp.newFolder(), AudioPipelineService.TRANSCRIPT_FILE)
        assertNull(AudioPipelineService.appendTranscriptRecord(ok, "{}"))
        assertEquals("{}\n", ok.readText())
        // A directory is not appendable on any platform: the fault comes back as a value.
        assertTrue(AudioPipelineService.appendTranscriptRecord(tmp.newFolder(), "{}") is FileNotFoundException)
    }
}

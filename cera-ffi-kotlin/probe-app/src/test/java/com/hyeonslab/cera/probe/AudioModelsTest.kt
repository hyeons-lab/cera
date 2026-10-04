package com.hyeonslab.cera.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class AudioModelsTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private fun touch(dir: File, name: String, bytes: Int = 4) {
        File(dir, name).writeBytes(ByteArray(bytes))
    }

    @Test
    fun a_directory_without_a_vad_has_no_models() {
        val dir = tmp.newFolder()
        touch(dir, AudioModels.WHISPER)
        assertNull(AudioModels.find(listOf(dir)))
    }

    @Test
    fun only_the_models_that_exist_become_stages() {
        val dir = tmp.newFolder()
        touch(dir, AudioModels.VAD)
        touch(dir, AudioModels.WHISPER)
        assertEquals(listOf("vad", "whisper"), AudioModels.find(listOf(dir))!!.stages)
        touch(dir, AudioModels.DIARIZER)
        touch(dir, AudioModels.HOTWORD)
        assertEquals(
            listOf("vad", "hotword", "whisper", "diarizer"),
            AudioModels.find(listOf(dir))!!.stages,
        )
    }

    @Test
    fun an_empty_optional_file_is_a_failed_copy_not_a_model() {
        val dir = tmp.newFolder()
        touch(dir, AudioModels.VAD)
        touch(dir, AudioModels.WHISPER, bytes = 0)
        assertNull(AudioModels.find(listOf(dir))!!.whisper)
    }

    @Test
    fun the_first_directory_with_a_vad_wins() {
        val private = tmp.newFolder()
        val external = tmp.newFolder()
        touch(private, AudioModels.VAD)
        touch(external, AudioModels.VAD)
        touch(external, AudioModels.WHISPER)
        // The private copy has no Whisper; the external one is not mixed in.
        assertEquals(listOf("vad"), AudioModels.find(listOf(private, external))!!.stages)
        assertEquals(private, AudioModels.find(listOf(private, external))!!.vad.parentFile)
    }

    @Test
    fun the_summary_names_each_stage_with_its_size_in_megabytes() {
        val dir = tmp.newFolder()
        touch(dir, AudioModels.VAD, bytes = 2_183_520)
        touch(dir, AudioModels.WHISPER, bytes = 82_172_352)
        touch(dir, AudioModels.DIARIZER, bytes = 126_828_608)
        assertEquals(
            "vad + whisper (82 MB) + diarizer (127 MB)",
            AudioModels.find(listOf(dir))!!.summary,
        )
        File(dir, AudioModels.WHISPER).delete()
        assertEquals("vad + diarizer (127 MB)", AudioModels.find(listOf(dir))!!.summary)
    }
}

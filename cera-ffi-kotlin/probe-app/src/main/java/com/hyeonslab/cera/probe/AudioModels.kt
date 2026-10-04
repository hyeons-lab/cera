package com.hyeonslab.cera.probe

import java.io.File

/**
 * The model files an always-on audio service runs, found by fixed names in a directory. Only the
 * VAD is required: a service with just a VAD reports speech boundaries, and each model added turns
 * on the matching stage (Whisper transcribes, the diarizer labels speakers, the hotword model gates
 * transcription behind a wake word).
 *
 * Whisper must be Q8_0 or Q4_0 and the diarizer must be a `--tail-outtype q8_0` GGUF to run on the
 * NPU; anything else works but falls back to the CPU, which Android demotes in the background.
 *
 * The recommended Whisper is `base` at Q8_0 (82 MB): clearly more accurate than `tiny` and still
 * about twice as fast as real time on the NPU. `small` is the most accurate but takes longer than
 * real time on continuous speech. Make the file with
 * `cera transcribe --model base --quant q8_0 --download-model`.
 */
data class AudioModels(
    val vad: File,
    val hotword: File?,
    val whisper: File?,
    val diarizer: File?,
) {
    /**
     * The stages with the size of each model file, for the startup log: it is the only record of
     * which Whisper was loaded, and the models are swapped by copying a file over `whisper.gguf`.
     */
    val summary: String
        get() = buildList {
            add("vad")
            if (hotword != null) add("hotword ${megabytes(hotword)}")
            if (whisper != null) add("whisper ${megabytes(whisper)}")
            if (diarizer != null) add("diarizer ${megabytes(diarizer)}")
        }.joinToString(" + ")

    /** Names of the stages that will run, for the status line. */
    val stages: List<String>
        get() = buildList {
            add("vad")
            if (hotword != null) add("hotword")
            if (whisper != null) add("whisper")
            if (diarizer != null) add("diarizer")
        }

    private fun megabytes(file: File) = "(${(file.length() + 500_000) / 1_000_000} MB)"

    companion object {
        const val DIR_NAME = "audio-models"
        const val VAD = "vad.gguf"
        const val HOTWORD = "hotword.gguf"
        const val WHISPER = "whisper.gguf"
        const val DIARIZER = "diarizer.gguf"

        /**
         * The models in the first of [dirs] that holds a VAD, or null when none does. Directories
         * are searched in order so the app's private storage wins over a copy pushed to the
         * external files directory.
         */
        /**
         * The directories to search, in order: the app's private storage first so it wins over a
         * copy pushed to the external files directory.
         */
        fun dirs(filesDir: File, externalFilesDir: File?): List<File> = listOfNotNull(
            File(filesDir, DIR_NAME),
            externalFilesDir?.let { File(it, DIR_NAME) },
        )

        fun find(dirs: List<File>): AudioModels? {
            for (dir in dirs) {
                val vad = File(dir, VAD)
                // A 0-byte VAD is a failed copy like an empty optional, not a model.
                if (!vad.isFile || vad.length() == 0L) continue
                fun optional(name: String) = File(dir, name).takeIf { it.isFile && it.length() > 0 }
                return AudioModels(vad, optional(HOTWORD), optional(WHISPER), optional(DIARIZER))
            }
            return null
        }
    }
}

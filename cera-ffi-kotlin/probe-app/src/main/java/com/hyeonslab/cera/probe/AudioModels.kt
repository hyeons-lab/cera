package com.hyeonslab.cera.probe

import java.io.File

/** Which diarizer GGUF [AudioModels.diarizer] is: 4-speaker Sortformer or 8-speaker Nemotron-3. */
enum class DiarizerKind {
    SORTFORMER,
    NEMOTRON3,
}

/**
 * The model files an always-on audio service runs, found by fixed names in a directory. Only the
 * VAD is required: a service with just a VAD reports speech boundaries, and each model added turns
 * on the matching stage (Whisper transcribes, the diarizer labels speakers, the hotword model gates
 * transcription behind a wake word).
 *
 * Whisper runs on the NPU from Q8_0, Q4_0, F16, or F32 weights (the engine repacks F16/F32
 * to Q8_0 at load) and the diarizer must be a `--tail-outtype q8_0` GGUF to run on the NPU;
 * anything else works but falls back to the CPU, which Android demotes in the background.
 *
 * The recommended Whisper is `base` at Q8_0 (82 MB): clearly more accurate than `tiny` and still
 * about twice as fast as real time on the NPU. `small` is the most accurate but takes longer than
 * real time on continuous speech. Make the file with
 * `cera transcribe --model base --quant q8_0 --download-model`.
 *
 * The diarizer is either Sortformer (`diarizer.gguf`) or Nemotron-3-Diarization
 * (`diarizer-nemotron3.gguf`, 8 speakers). When both files are present the Nemotron-3 one
 * wins, so a comparison run is a file copy away.
 */
data class AudioModels(
    val vad: File,
    val hotword: File?,
    val whisper: File?,
    val diarizer: File?,
    val diarizerKind: DiarizerKind = DiarizerKind.SORTFORMER,
) {
    /**
     * The stages with the size of each model file but the VAD, for the startup log: it is the
     * only record of which Whisper was loaded, and the models are swapped by copying a file
     * over `whisper.gguf`.
     */
    val summary: String
        get() = presentStages.joinToString(" + ") { (name, file) ->
            if (file == null) name else "$name ${megabytes(file)}"
        }

    /** Names of the stages that will run, for the status line. */
    val stages: List<String>
        get() = presentStages.map { (name, _) -> name }

    /** The stages with models, in order: the VAD is always present, the rest optional. */
    private val presentStages: List<Pair<String, File?>>
        get() = buildList {
            add("vad" to null)
            if (hotword != null) add("hotword" to hotword)
            if (whisper != null) add("whisper" to whisper)
            if (diarizer != null) {
                val name = when (diarizerKind) {
                    DiarizerKind.NEMOTRON3 -> "diarizer-nemotron3"
                    DiarizerKind.SORTFORMER -> "diarizer"
                }
                add(name to diarizer)
            }
        }

    private fun megabytes(file: File) = "(${(file.length() + 500_000) / 1_000_000} MB)"

    companion object {
        const val DIR_NAME = "audio-models"
        const val VAD = "vad.gguf"
        const val HOTWORD = "hotword.gguf"
        const val WHISPER = "whisper.gguf"
        const val DIARIZER = "diarizer.gguf"
        const val DIARIZER_NEMOTRON3 = "diarizer-nemotron3.gguf"

        /**
         * The directories to search, in order: the app's private storage first so it wins over a
         * copy pushed to the external files directory.
         */
        fun dirs(filesDir: File, externalFilesDir: File?): List<File> = listOfNotNull(
            File(filesDir, DIR_NAME),
            externalFilesDir?.let { File(it, DIR_NAME) },
        )

        /**
         * The models in the first of [dirs] that holds a VAD, or null when none does. Directories
         * are searched in order so the app's private storage wins over a copy pushed to the
         * external files directory.
         */
        fun find(dirs: List<File>): AudioModels? {
            for (dir in dirs) {
                val vad = File(dir, VAD)
                // A 0-byte VAD is a failed copy like an empty optional, not a model.
                if (!vad.isFile || vad.length() == 0L) continue
                fun optional(name: String) = File(dir, name).takeIf { it.isFile && it.length() > 0 }
                val nemotron3 = optional(DIARIZER_NEMOTRON3)
                val diarizer = nemotron3 ?: optional(DIARIZER)
                val kind = if (nemotron3 != null) DiarizerKind.NEMOTRON3 else DiarizerKind.SORTFORMER
                return AudioModels(vad, optional(HOTWORD), optional(WHISPER), diarizer, kind)
            }
            return null
        }
    }
}

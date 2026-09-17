package whisperprobe

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import uniffi.cera_ffi.FfiException
import uniffi.cera_ffi.FfiWhisperModel
import uniffi.cera_ffi.whisperDefaultTranscribeOpts
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path

// This helper is also the application-facing example in the Whisper guide.
suspend fun transcribeRecording(
    modelPath: String,
    pcm16kMono: List<Float>,
): String {
    return withContext(Dispatchers.IO) {
        FfiWhisperModel.fromFile(modelPath).use { model ->
            val opts = whisperDefaultTranscribeOpts().copy(language = "en")
            model.transcribeAsync(pcm16kMono, opts)
        }
    }
}

fun main(args: Array<String>) =
    runBlocking {
        val directory = Path.of(args.single())
        val raw = Files.readAllBytes(directory.resolve("audio.f32"))
        check(raw.isNotEmpty() && raw.size % 4 == 0)
        val buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        val pcm = List(raw.size / 4) { buffer.float }
        val bytes = Files.readAllBytes(directory.resolve("a.gguf"))
        FfiWhisperModel.fromBytes(bytes).use { a ->
            bytes.fill(0)
            val copy = directory.resolve("kotlin-owned.gguf")
            Files.copy(directory.resolve("b.gguf"), copy)
            FfiWhisperModel.fromFile(copy.toString()).use { b ->
                Files.delete(copy)
                check(a.isMultilingual() && !b.isMultilingual())
                check(a.languages().size == 100 && a.languages().take(2) == listOf("en", "zh"))
                val defaults = whisperDefaultTranscribeOpts()
                check(defaults.language == null && !defaults.translate && !defaults.timestamps)
                check(defaults.maxTokens == 448u && defaults.temperature == 0f)
                val opts = defaults.copy(language = "en", maxTokens = 3u)
                check(a.transcribe(pcm, opts) == "aaa")
                check(a.transcribe(emptyList(), null).isEmpty())
                val short = opts.copy(maxTokens = 2u)
                val first = async { a.transcribeAsync(pcm, opts) }
                val second = async { a.transcribeAsync(pcm, short) }
                val third = async { b.transcribeAsync(pcm, short) }
                check(listOf(first.await(), second.await(), third.await()) == listOf("aaa", "aa", "bb"))
                check(a.transcribe(pcm, opts.copy(maxTokens = null, temperature = null)) == "a".repeat(13))
                check(transcribeRecording(directory.resolve("a.gguf").toString(), pcm) == "a".repeat(13))
            }
        }
        try {
            FfiWhisperModel.fromBytes(byteArrayOf(1, 2)).use { error("Malformed GGUF accepted") }
        } catch (_: FfiException.Backend) {
        }
        val missing = directory.resolve("missing.gguf")
        check(!Files.exists(missing))
        try {
            FfiWhisperModel.fromFile(missing.toString()).use { error("Missing GGUF accepted") }
        } catch (_: FfiException.Backend) {
        }
        println(
            """{"cases":["bytes-owned","file-owned","languages","defaults","sync","async-shared","async-distinct","empty","malformed","missing","recording-example"],"text":["aaa","aa","bb"]}""",
        )
    }

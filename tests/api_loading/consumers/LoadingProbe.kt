package loadingprobe

import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.loading_native.LoadConfig
import uniffi.loading_native.Source
import uniffi.loading_native.futureHandleForProbe
import uniffi.loading_native.modelLoaderWithEngineConfig
import java.io.File
import uniffi.loading_native.ProbeGenerationDefaults as GenerationDefaults
import uniffi.loading_native.ProbeGenerativeModel as GenerativeModel
import uniffi.loading_native.ProbeLoadException as LoadException
import uniffi.loading_native.ProbeModelFiles as ModelFiles
import uniffi.loading_native.ProbeModelLoader as ModelLoader
import uniffi.loading_native.ProbeModelParts as ModelParts
import uniffi.loading_native.ProbeSamplingDefaults as SamplingDefaults
import uniffi.loading_native.ProbeSession as Session

private fun config(
    backend: String = "cpu",
    parts: Boolean = false,
) = LoadConfig(24u, backend, if (parts) "missing-probe-draft.gguf" else null, parts)

fun consumed(loader: ModelLoader) {
    for (typed in listOf(false, true)) {
        try {
            if (typed) loader.buildGenerative() else loader.build()
            error("reused loader succeeded")
        } catch (_: LoadException.Consumed) {
            // Both entry points share the consumed state.
        }
    }
}

fun checkInfo(
    model: GenerativeModel,
    parts: Boolean = false,
) {
    val info = model.info()
    check(info.requestedContext == 24uL && info.capacity == 24u)
    check(info.backend == "Cpu" && info.gpuDepthformer == parts)
    check(info.draftModel == if (parts) "missing-probe-draft.gguf" else null)
    if (parts) {
        check(info.chatTemplate == "probe-template")
        check(kotlin.math.abs(info.temperature - 0.37f) < 0.00001f)
        check(kotlin.math.abs(info.topP - 0.71f) < 0.00001f && info.topK == 7u)
        check(kotlin.math.abs(info.minP - 0.13f) < 0.00001f)
        check(kotlin.math.abs(info.repetitionPenalty - 1.23f) < 0.00001f)
    }
}

fun runSession(session: Session): String =
    session.use {
        it.append(listOf(0u, 1u))
        check(it.position() == 2u)
        val tokens = it.generate()
        check(tokens.size == 3 && tokens.all { token -> token < 2u })
        check(it.position() == 5u)
        "{\"tokens\":[${tokens.joinToString(",")}],\"position\":${it.position()}}"
    }

private fun runDefaults(bytes: ByteArray): List<String> {
    val empty = SamplingDefaults(null, null, null, null, null)
    val sampling = SamplingDefaults(0.37f, 0.71f, 7u, 0.13f, 1.23f)
    val profiles =
        mutableListOf<Triple<String, GenerationDefaults?, GenerationDefaults>>(
            Triple("absent", null, GenerationDefaults.Text(empty)),
            Triple("text-empty", GenerationDefaults.Text(empty), GenerationDefaults.Text(empty)),
        )
    for ((name, value) in listOf(
        "audio" to GenerationDefaults.Audio(sampling, 3u, 0.625f, 11u),
        "audio" to GenerationDefaults.Audio(empty, 0u, 0f, 0u),
        "audio" to GenerationDefaults.Audio(empty, UInt.MAX_VALUE, 1f, UInt.MAX_VALUE),
        "audio-empty" to GenerationDefaults.Audio(empty, null, null, null),
    )) {
        profiles += Triple(name, value, value)
    }
    for ((raw, canonical) in listOf(
        " { \"nested\" : [true, null, {\"x\":7}], \"label\": \"line\\ntext\" } " to
            "{\"label\":\"line\\ntext\",\"nested\":[true,null,{\"x\":7}]}",
        "[1, 2, null]" to "[1,2,null]",
        "42" to "42",
        "true" to "true",
        "\"text\"" to "\"text\"",
        "null" to "null",
    )) {
        profiles += Triple("other", GenerationDefaults.Other(raw), GenerationDefaults.Other(canonical))
    }
    for ((_, defaults, expected) in profiles) {
        for (typed in listOf(false, true)) {
            val parts = ModelParts(bytes, null, null, null, null, null, null, defaults)
            val loader = ModelLoader(Source.Parts(parts), config())
            val handle = if (typed) null else loader.build()
            val model = if (typed) loader.buildGenerative() else checkNotNull(handle?.asGenerative())
            consumed(loader)
            val observed = model.generationDefaultsForProbe()
            val session = model.createSession()
            loader.close()
            handle?.close()
            model.close()
            check(observed == expected)
            check(runSession(session) == "{\"tokens\":[0,1,0],\"position\":5}")
        }
    }
    for (raw in listOf("", "{", "null trailing", "{\"x\":NaN}")) {
        for (typed in listOf(false, true)) {
            val parts = ModelParts(bytes, null, null, null, null, null, null, GenerationDefaults.Other(raw))
            ModelLoader(Source.Parts(parts), config()).use { loader ->
                try {
                    if (typed) loader.buildGenerative() else loader.build()
                    error("malformed defaults succeeded")
                } catch (error: LoadException.InvalidConfig) {
                    check(error.field == "generation_defaults.raw_json" && error.value == raw)
                    check(error.reason == "invalid_json" && error.detail.isNotEmpty())
                }
                consumed(loader)
            }
        }
    }
    return profiles.map { "parts-defaults-${it.first}" }.distinct().sorted() + "parts-defaults-invalid-json"
}

private fun runConfigs(bytes: ByteArray): List<String> {
    data class Profile(
        val request: ULong?,
        val observed: ULong,
        val capacity: UInt,
    )
    val profiles =
        listOf(
            Profile(null, 4096uL, 64u),
            Profile(0uL, 64uL, 64u),
            Profile(64uL, 64uL, 64u),
            Profile(65uL, 65uL, 64u),
            Profile(UInt.MAX_VALUE.toULong() + 25uL, UInt.MAX_VALUE.toULong() + 25uL, 64u),
            Profile(ULong.MAX_VALUE, 64uL, 64u),
            Profile(1uL, 1uL, 1u),
        )
    for ((request, observed, capacity) in profiles) {
        val options = if (request == null) LoadConfig(backend = "cpu") else LoadConfig(contextSize = request, backend = "cpu")
        check(options.contextSize == (request ?: 4096uL))
        check(options.draftModel == null && !options.gpuDepthformer)
        for (typed in listOf(false, true)) {
            val loader =
                if (typed) {
                    modelLoaderWithEngineConfig(
                        Source.Bytes(bytes),
                        EngineConfig(options.contextSize, BackendPreference.CPU, null, null, false),
                    )
                } else {
                    ModelLoader(Source.Bytes(bytes), options)
                }
            val handle = if (typed) null else loader.build()
            val model = if (typed) loader.buildGenerative() else checkNotNull(handle!!.asGenerative())
            consumed(loader)
            val info = model.info()
            check(info.requestedContext == observed && info.capacity == capacity)
            check(info.backend == "Cpu" && info.draftModel == null && !info.gpuDepthformer)
            val session = model.createSession()
            loader.close()
            handle?.close()
            model.close()
            if (capacity == 1u) {
                session.use {
                    it.append(listOf(0u))
                    try {
                        it.append(listOf(1u))
                        error("append exceeded the configured context")
                    } catch (error: LoadException.Engine) {
                        check(error.detail.isNotEmpty())
                    }
                    check(it.position() == 1u)
                }
            } else {
                check(runSession(session) == "{\"tokens\":[0,1,0],\"position\":5}")
            }
        }
    }
    return listOf("native-config-default", "native-config-zero", "native-config-cap", "native-config-wide", "native-config-small")
}

private fun runFiles(root: File): List<String> {
    fun path(name: String) = File(root, name).absolutePath

    fun files(
        primary: String,
        inference: String?,
    ) = ModelFiles(
        path(primary),
        "missing-projector.gguf",
        path("missing-decoder.gguf"),
        "missing-tokenizer.gguf",
        "missing-draft.gguf",
        mapOf("future_file" to "future.bin", "absolute_file" to path("absolute.bin")),
        inference,
        "file-template",
    )
    for (typed in listOf(false, true)) {
        for (inference in listOf(null, "llama.cpp/text-to-text")) {
            val loader = ModelLoader(Source.Files(files("model.gguf", inference)), config())
            val handle = if (typed) null else loader.build()
            if (handle != null) check(handle.kind() == "Generative")
            val model = if (typed) loader.buildGenerative() else checkNotNull(handle!!.asGenerative())
            consumed(loader)
            checkInfo(model)
            val resolved = model.files()
            check(resolved.model == path("model.gguf"))
            check(resolved.multimodalProjector == path("missing-projector.gguf"))
            check(resolved.audioDecoder == path("missing-decoder.gguf"))
            check(resolved.audioTokenizer == path("missing-tokenizer.gguf"))
            check(resolved.draftModel == path("missing-draft.gguf"))
            check(resolved.extras == mapOf("future_file" to path("future.bin"), "absolute_file" to path("absolute.bin")))
            check(resolved.inferenceType == "llama.cpp/text-to-text")
            check(resolved.chatTemplate == "file-template" && model.info().chatTemplate == "file-template")
            val session = model.createSession()
            loader.close()
            handle?.close()
            model.close()
            check(runSession(session) == "{\"tokens\":[0,1,0],\"position\":5}")
        }
    }
    for (name in listOf("kind", "missing", "inference")) {
        for (typed in listOf(false, true)) {
            val source =
                files(
                    if (name ==
                        "kind"
                    ) {
                        "kws.gguf"
                    } else {
                        "missing-primary.gguf"
                    },
                    if (name == "inference") "future/unsupported" else null,
                )
            ModelLoader(Source.Files(source), config("metal")).use { loader ->
                try {
                    if (typed) loader.buildGenerative() else loader.build()
                    error("invalid file load succeeded")
                } catch (error: LoadException.KindMismatch) {
                    check(name == "kind" && error.expected == "Generative" && error.actual == "Hotword" && error.architecture == "kws")
                } catch (error: LoadException.Source) {
                    check(name == "missing" && error.sourceKind == "files" && error.detail.isNotEmpty())
                } catch (error: LoadException.UnsupportedInferenceType) {
                    check(name == "inference" && error.inferenceType == "future/unsupported")
                }
                consumed(loader)
            }
        }
    }
    return listOf("native-files", "native-files-kind", "native-files-missing", "native-files-inference")
}

fun main(args: Array<String>) {
    require(args.size == 2) { "Expected model and remote fixture directories" }
    val root = File(args[0])
    val bytes = File(root, "model.gguf").readBytes()
    val cases = mutableListOf<String>()
    val input = bytes.copyOf()
    val loader = ModelLoader(Source.Bytes(input), config())
    input.fill(0)
    val handle = loader.build()
    consumed(loader)
    check(handle.kind() == "Generative")
    val first = checkNotNull(handle.asGenerative())
    val second = checkNotNull(handle.asGenerative())
    checkInfo(second)
    loader.close()
    handle.close()
    first.close()
    val session = second.createSession()
    second.close()
    val result = runSession(session)
    cases += "bytes-lifetime"

    val defaults = SamplingDefaults(0.37f, 0.71f, 7u, 0.13f, 1.23f)
    val parts =
        ModelParts(
            bytes,
            byteArrayOf(1),
            byteArrayOf(2),
            byteArrayOf(3),
            byteArrayOf(4),
            "llama.cpp/text-to-text",
            "probe-template",
            GenerationDefaults.Text(defaults),
        )
    ModelLoader(Source.Parts(parts), config(parts = true)).use { partsLoader ->
        partsLoader.buildGenerative().use { model ->
            consumed(partsLoader)
            checkInfo(model, parts = true)
            runSession(model.createSession())
        }
    }
    cases += "parts-defaults"
    cases += runDefaults(bytes)
    ModelLoader(Source.Path(File(root, "model.gguf").absolutePath), config()).use { pathLoader ->
        pathLoader.buildGenerative().use { model ->
            consumed(pathLoader)
            checkInfo(model)
            runSession(model.createSession())
        }
    }
    cases += "native-path"
    cases += runFiles(root)
    cases += runConfigs(bytes)
    cases += runProduction(root, bytes)
    cases += runRemote(File(args[1]))
    cases += runProductionRemote(File(args[1]))

    for ((arch, kind) in listOf(
        "bert" to "Encoder",
        "modernbert" to "Encoder",
        "whisper" to "Whisper",
        "silero_vad" to "Vad",
        "kws" to "Hotword",
    )) {
        for (typed in listOf(false, true)) {
            ModelLoader(Source.Bytes(File(root, "$arch.gguf").readBytes()), config("metal")).use { bad ->
                try {
                    if (typed) bad.buildGenerative() else bad.build()
                    error("wrong kind succeeded")
                } catch (error: LoadException.KindMismatch) {
                    check(error.expected == "Generative" && error.actual == kind && error.architecture == arch) {
                        "$arch kind mismatch payload (typed=$typed)"
                    }
                }
                consumed(bad)
            }
        }
        cases += "kind-$arch"
    }
    for (name in listOf("unknown", "malformed", "backend", "invalid-backend", "assembly", "inference")) {
        for (typed in listOf(false, true)) {
            val data =
                when (name) {
                    "unknown" -> File(root, "unknown.gguf").readBytes()
                    "assembly" -> File(root, "llama.gguf").readBytes()
                    "malformed" -> byteArrayOf(0, 1)
                    else -> bytes
                }
            val backend =
                when (name) {
                    "backend" -> "metal"
                    "invalid-backend" -> "invalid-probe"
                    else -> "cpu"
                }
            val source = if (name == "inference") Source.Parts(parts.copy(inferenceType = "future/unsupported")) else Source.Bytes(data)
            ModelLoader(source, config(backend)).use { bad ->
                try {
                    if (typed) bad.buildGenerative() else bad.build()
                    error("invalid load succeeded")
                } catch (error: LoadException.UnsupportedArchitecture) {
                    check(name == "unknown" && error.architecture == "future_probe")
                } catch (error: LoadException.Source) {
                    check(name == "malformed" && error.sourceKind == "bytes" && error.detail.isNotEmpty())
                } catch (error: LoadException.Assembly) {
                    check(name == "backend" || name == "assembly")
                    check(error.backend == (if (name == "backend") "Metal" else "Cpu") && error.detail.isNotEmpty())
                } catch (error: LoadException.InvalidConfig) {
                    check(name == "invalid-backend" && error.field == "backend")
                    check(error.value == "invalid-probe" && error.reason == "unknown_backend" && error.detail.isNotEmpty())
                } catch (error: LoadException.UnsupportedInferenceType) {
                    check(name == "inference" && error.inferenceType == "future/unsupported")
                }
                consumed(bad)
            }
        }
        cases += name
    }
    futureHandleForProbe().use { future ->
        check(future.kind() == "future-probe" && future.asGenerative() == null)
    }
    cases += "future-kind"
    println("{\"cases\":[${cases.sorted().joinToString(",") { "\"$it\"" }}],\"generation\":$result}")
}

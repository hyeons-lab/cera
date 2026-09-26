package loadingprobe

import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.CeraEngine
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.FfiConverterTypeBackendPreference
import uniffi.cera_ffi.FfiException
import uniffi.cera_ffi.FinishReason
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.GenerationDefaults
import uniffi.cera_ffi.KvCompression
import uniffi.cera_ffi.LoadException
import uniffi.cera_ffi.ModalitySink
import uniffi.cera_ffi.ModelFiles
import uniffi.cera_ffi.ModelLoader
import uniffi.cera_ffi.ModelParts
import uniffi.cera_ffi.SamplingDefaults
import uniffi.cera_ffi.Session
import uniffi.cera_ffi.SessionConfig
import uniffi.loading_native.enginesShareForProbe
import uniffi.loading_native.productionDefaultsForProbe
import uniffi.loading_native.sessionConfigForProbe
import java.io.File
import java.nio.ByteBuffer
import uniffi.cera_ffi.ModelSource as Source

private class ProductionSink(
    val session: Session,
) : ModalitySink {
    val done = mutableListOf<FinishReason>()
    var text = ""

    override fun onThoughtChunk(text: String) {
        error("unexpected thoughts")
    }

    override fun onAudioFrames(
        pcm: List<Float>,
        sampleRate: UInt,
    ) {
        error("unexpected audio")
    }

    override fun onTextChunk(text: String) {
        this.text += text
        session.cancel()
    }

    override fun onDone(reason: FinishReason) {
        done += reason
    }
}

fun productionConsumed(loader: ModelLoader) {
    for (direct in listOf(false, true)) {
        try {
            if (direct) loader.buildGenerative() else loader.build()
            error("production loader reused")
        } catch (_: uniffi.cera_ffi.LoadException.Consumed) {
        }
    }
}

fun runProduction(
    root: File,
    bytes: ByteArray,
): List<String> {
    for ((index, backend) in BackendPreference.entries.withIndex()) {
        val buffer = ByteBuffer.allocate(4)
        FfiConverterTypeBackendPreference.write(backend, buffer)
        check(buffer.array().contentEquals(byteArrayOf(0, 0, 0, (index + 1).toByte())))
        buffer.flip()
        check(FfiConverterTypeBackendPreference.read(buffer) == backend && !buffer.hasRemaining())
    }
    for (tag in listOf(0, 6, -1, Int.MAX_VALUE)) {
        val buffer = ByteBuffer.allocate(4).putInt(tag).flip()
        try {
            FfiConverterTypeBackendPreference.read(buffer)
            error("malformed production backend enum accepted")
        } catch (error: RuntimeException) {
            check(error.cause is IndexOutOfBoundsException && !buffer.hasRemaining())
        }
    }
    for (backend in listOf(BackendPreference.GPU, BackendPreference.METAL)) {
        for (direct in listOf(false, true)) {
            ModelLoader(Source.Bytes(bytes), EngineConfig(backend = backend)).use { loader ->
                try {
                    if (direct) loader.buildGenerative() else loader.build()
                    error("unavailable production backend accepted")
                } catch (error: uniffi.cera_ffi.LoadException.Assembly) {
                    check(error.backend == (if (backend == BackendPreference.GPU) "Gpu" else "Metal") && error.detail.isNotEmpty())
                }
                productionConsumed(loader)
            }
        }
    }
    val defaults = EngineConfig(backend = BackendPreference.CPU)
    check(defaults.contextSize == 4096uL && defaults.bundleRepo == null)
    check(defaults.draftModel == null && !defaults.gpuDepthformer)
    val sessionDefaults = SessionConfig()
    check(sessionDefaults.maxSeqLen == null && sessionDefaults.kvCompression == null)
    check(sessionDefaults.nKeep == 0u && sessionDefaults.seed == null)
    check(sessionDefaults.ubatchSize == 512u && !sessionDefaults.gpuDepthformer)
    val path = File(root, "model.gguf").absolutePath
    val sources =
        listOf(
            Source.Bytes(bytes),
            Source.Path(path),
            Source.Files(ModelFiles(path, null, null, null, null, emptyMap(), null, null)),
            Source.Parts(ModelParts(bytes, null, null, null, null, null, null, null)),
        )
    for (source in sources) {
        for (direct in listOf(false, true)) {
            val loader = ModelLoader(source, defaults)
            val handle = if (direct) null else loader.build()
            val model = if (direct) loader.buildGenerative() else checkNotNull(handle!!.asGenerative())
            productionConsumed(loader)
            val engine: CeraEngine = model.engine()
            model.engine().use { other -> check(enginesShareForProbe(engine, other)) }
            check(engine.contextSize() == 4096uL && engine.metadata().maxSeqLen == 64u)
            CeraEngine.fromBytes(bytes, defaults).use { independent ->
                check(!enginesShareForProbe(engine, independent))
            }
            val config =
                SessionConfig(
                    maxSeqLen = 8u,
                    kvCompression = KvCompression.F16,
                    seed = ULong.MAX_VALUE,
                    ubatchSize = 1u,
                    gpuDepthformer = true,
                )
            val session: Session = model.createSession(config)
            check(sessionConfigForProbe(session) == config)
            val sibling = engine.newSession(SessionConfig())
            loader.close()
            handle?.close()
            model.close()
            check(engine.contextSize() == 4096uL)
            check(engine.encodeText("ab") == listOf(0u, 1u))
            check(engine.decodeTokens(listOf(0u, 1u)) == "ab")
            sibling.appendTokens(listOf(1u))
            session.appendTokens(listOf(0u, 1u))
            engine.clearPrefixCache()
            check(session.position() == 2u && sibling.position() == 1u)
            engine.close()
            sibling.use {
                session.use {
                    val one = GenerateOpts(maxTokens = 1u, temperature = 0.7f, ignoreEos = true)
                    val first = session.generate(one)
                    check(first.tokens.size == 1 && session.position() == 3u)
                    val next = session.generate(GenerateOpts(maxTokens = 2u, temperature = 0.7f, ignoreEos = true))
                    check(next.tokens.size == 2 && session.position() == 5u)
                    check(sibling.position() == 1u)
                    session.reset()
                    check(session.position() == 0u)
                    try {
                        session.appendTokens(emptyList())
                        error("empty input accepted")
                    } catch (_: FfiException.EmptyInput) {
                    }
                    session.appendTokens(listOf(0u, 1u))
                    val sink = ProductionSink(session)
                    val streamed =
                        session.generateStreaming(
                            GenerateOpts(maxTokens = 3u, temperature = 0.7f, ignoreEos = true, flushEveryTokens = 1u),
                            sink,
                        )
                    check(sink.text.isNotEmpty() && sink.done.size == 1)
                    check(streamed.finishReason == FinishReason.Cancelled && streamed.tokensGenerated == 1u)
                    val position = session.position()
                    session.clearCancel()
                    session.generate(one)
                    check(session.position() == position + 1u)
                }
            }
        }
    }
    ModelLoader(Source.Bytes(bytes), defaults).use { loader ->
        loader.buildGenerative().use { model ->
            val modes =
                listOf(
                    null,
                    KvCompression.None,
                    KvCompression.F16,
                    KvCompression.TurboQuant(0uL, true, true),
                    KvCompression.TurboQuant(ULong.MAX_VALUE, true, false),
                    KvCompression.TurboQuant(42uL, false, true),
                )
            for (mode in modes) {
                val config = SessionConfig(8u, mode, 1u, 0uL, 0u, true)
                model.createSession(config).use { session ->
                    check(sessionConfigForProbe(session) == config.copy(kvCompression = mode ?: KvCompression.None))
                    session.appendTokens(listOf(0u, 1u))
                    check(session.generate(GenerateOpts(maxTokens = 1u, temperature = 0f, ignoreEos = true)).tokens.size == 1)
                }
            }
            model.createSession(SessionConfig(maxSeqLen = 1u)).use { capped ->
                capped.appendTokens(listOf(0u))
                try {
                    capped.appendTokens(listOf(1u))
                    error("session cap ignored")
                } catch (error: FfiException.ContextOverflow) {
                    check(error.maxSeqLen == 1u && error.by == 1u && capped.position() == 1u)
                }
            }
        }
    }
    return runProductionDefaults(bytes) + runProductionErrors(root, bytes) +
        listOf(
            "production-backend-transport",
            "production-defaults",
            "production-sources",
            "production-session",
            "production-shared-engine",
            "production-kv-config",
            "production-stream-cancel",
        )
}

private fun runProductionDefaults(bytes: ByteArray): List<String> {
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
            val loader = ModelLoader(Source.Parts(parts), EngineConfig(backend = BackendPreference.CPU))
            val handle = if (typed) null else loader.build()
            val model = if (typed) loader.buildGenerative() else checkNotNull(handle?.asGenerative())
            productionConsumed(loader)
            val observed = model.engine().use { productionDefaultsForProbe(it) }
            val session = model.createSession(SessionConfig(seed = 42uL))
            loader.close()
            handle?.close()
            model.close()
            check(observed == expected)
            session.use {
                it.appendTokens(listOf(0u, 1u))
                val output = it.generate(GenerateOpts(maxTokens = 3u, temperature = 0f, ignoreEos = true))
                check(output.tokens == listOf(0u, 1u, 0u) && it.position() == 5u)
            }
        }
    }
    for (raw in listOf("", "{", "null trailing", "{\"x\":NaN}")) {
        for (typed in listOf(false, true)) {
            val parts = ModelParts(bytes, null, null, null, null, null, null, GenerationDefaults.Other(raw))
            ModelLoader(Source.Parts(parts), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
                try {
                    if (typed) loader.buildGenerative() else loader.build()
                    error("malformed defaults succeeded")
                } catch (error: LoadException.InvalidConfig) {
                    check(error.field == "generation_defaults.raw_json" && error.value == raw)
                    check(error.reason == "invalid_json" && error.detail.isNotEmpty())
                }
                productionConsumed(loader)
            }
        }
    }
    return profiles.map { "production-parts-defaults-${it.first}" }.distinct().sorted() + "production-parts-defaults-invalid-json"
}

private fun runProductionErrors(
    root: File,
    bytes: ByteArray,
): List<String> {
    val cases = mutableListOf<String>()
    val parts = ModelParts(bytes, null, null, null, null, null, null, null)
    for ((arch, kind) in listOf(
        "bert" to "Encoder",
        "modernbert" to "Encoder",
        "whisper" to "Whisper",
        "silero_vad" to "Vad",
        "kws" to "Hotword",
    )) {
        for (typed in listOf(false, true)) {
            ModelLoader(Source.Bytes(File(root, "$arch.gguf").readBytes()), EngineConfig(backend = BackendPreference.METAL)).use { bad ->
                try {
                    if (typed) bad.buildGenerative() else bad.build()
                    error("wrong kind succeeded")
                } catch (error: LoadException.KindMismatch) {
                    check(error.expected == "Generative" && error.actual == kind && error.architecture == arch) {
                        "$arch kind mismatch payload (typed=$typed)"
                    }
                }
                productionConsumed(bad)
            }
        }
        cases += "production-kind-$arch"
    }
    for (name in listOf("unknown", "malformed", "backend", "assembly", "inference")) {
        for (typed in listOf(false, true)) {
            val data =
                when (name) {
                    "unknown" -> File(root, "unknown.gguf").readBytes()
                    "assembly" -> File(root, "llama.gguf").readBytes()
                    "malformed" -> byteArrayOf(0, 1)
                    else -> bytes
                }
            val backend = if (name == "backend") BackendPreference.METAL else BackendPreference.CPU
            val source = if (name == "inference") Source.Parts(parts.copy(inferenceType = "future/unsupported")) else Source.Bytes(data)
            ModelLoader(source, EngineConfig(backend = backend)).use { bad ->
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
                } catch (error: LoadException.UnsupportedInferenceType) {
                    check(name == "inference" && error.inferenceType == "future/unsupported")
                }
                productionConsumed(bad)
            }
        }
        cases += "production-$name"
    }
    return cases
}

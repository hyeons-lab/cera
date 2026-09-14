package sessionprobe

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import uniffi.cera_ffi.*

// Each call owns one complete conversation; use closes its Session on return.
fun generateConversation(engine: CeraEngine, tokens: List<UInt>): List<UInt> =
    engine.newSession(SessionConfig(seed = 42uL)).use { session ->
        session.appendTokens(tokens)
        session.generate(options()).tokens
    }

fun options() = GenerateOpts(maxTokens = 3u, temperature = 0f, ignoreEos = true, flushEveryTokens = 1u)

fun busy(engine: CeraEngine, config: SessionConfig) {
    try {
        engine.newSession(config).use { error("Expected Busy while the GPU session is retained") }
    } catch (_: FfiException.Busy) { }
}

suspend fun gpuCases(bytes: ByteArray, backend: BackendPreference, compression: KvCompression): List<String> {
    val opened = mutableListOf<AutoCloseable>()
    fun <T : AutoCloseable> own(value: T): T { opened.add(value); return value }
    try {
        val load = EngineConfig(contextSize = 64uL, backend = backend)
        val config = SessionConfig(kvCompression = compression, seed = 42uL)
        val engine = own(CeraEngine.fromBytesAsync(bytes, load))
        val controlEngine = own(CeraEngine.fromBytesAsync(bytes, load))
        val active = own(engine.newSession(config))
        val control = own(controlEngine.newSession(config))
        active.appendTokens(listOf(0u, 1u, 0u))
        control.appendTokens(listOf(0u, 1u, 0u))
        busy(engine, config)
        busy(engine, SessionConfig(kvCompression = KvCompression.TurboQuant(99uL, true, true)))
        val position = active.position()
        check(active.hiddenStatesForTokens(listOf(1u, 0u)).size == 2 * 32 * 4)
        check(active.position() == position)
        busy(engine, config)
        active.cancel()
        busy(engine, config)
        active.clearCancel()
        try {
            active.generate(options().copy(grammar = "root ::= ("))
            error("Malformed grammar accepted")
        } catch (_: FfiException.GrammarParse) { }
        busy(engine, config)
        val actual = active.generate(options())
        val expected = control.generate(options())
        check(actual.tokens.size == 3 && actual.tokens == expected.tokens)
        check(active.position() == control.position())
        active.reset()
        check(active.position() == 0u)
        busy(engine, config)
        active.close()
        try {
            engine.newSession(SessionConfig(kvCompression = KvCompression.TurboQuant(99uL, true, true))).use {
                error("Compression conflict accepted")
            }
        } catch (_: FfiException.KvCompressionConflict) { }
        val successor = own(engine.newSession(config))
        successor.appendTokens(listOf(1u, 0u, 1u))
        val freshEngine = own(CeraEngine.fromBytesAsync(bytes, load))
        val fresh = own(freshEngine.newSession(config))
        fresh.appendTokens(listOf(1u, 0u, 1u))
        check(successor.generate(options()).tokens == fresh.generate(options()).tokens)
        val parent = own(CeraEngine.fromBytesAsync(bytes, load))
        val retained = own(parent.newSession(config))
        parent.close()
        retained.appendTokens(listOf(0u, 1u))
        check(retained.generate(options()).tokens.size == 3)
        return listOf("busy", "busy-before-config", "extraction-retains", "cancel-retains",
            "generation-error-retains", "continuation", "reset-retains", "release",
            "constructor-failure-release", "successor", "parent-release")
    } finally {
        opened.asReversed().forEach { it.close() }
    }
}

class PausedSink : ModalitySink {
    val entered = CountDownLatch(1)
    val release = CountDownLatch(1)
    private val paused = AtomicBoolean(false)
    private val done = mutableListOf<FinishReason>()
    override fun onThoughtChunk(text: String) { }
    override fun onAudioFrames(pcm: List<Float>, sampleRate: UInt) { }
    override fun onTextChunk(text: String) {
        if (paused.compareAndSet(false, true)) {
            entered.countDown()
            check(release.await(20, TimeUnit.SECONDS)) { "Callback release timed out" }
        }
    }
    override fun onDone(reason: FinishReason) { synchronized(done) { done.add(reason) } }
    fun completedReasons(): List<FinishReason> = synchronized(done) { done.toList() }
}

suspend fun asyncCases(bytes: ByteArray, backend: BackendPreference): List<String> = coroutineScope {
    CeraEngine.fromBytesAsync(bytes, EngineConfig(contextSize = 64uL, backend = backend)).use { engine ->
        val active = engine.newSession(SessionConfig(seed = 42uL))
        val sink = PausedSink()
        try {
            active.appendTokens(listOf(0u, 1u))
            val work = async(Dispatchers.Default) { active.generateStreamingAsync(options(), sink) }
            try {
                check(sink.entered.await(20, TimeUnit.SECONDS)) { "No streaming callback" }
                active.cancel()
                active.close()
                busy(engine, SessionConfig())
            } finally {
                sink.release.countDown()
            }
            val summary = work.await()
            check(summary.finishReason == FinishReason.Cancelled)
            check(sink.completedReasons() == listOf(FinishReason.Cancelled))
            engine.newSession(SessionConfig(seed = 42uL)).use { successor ->
                successor.appendTokens(listOf(1u, 0u))
                check(successor.generate(options()).tokens.size == 3)
            }
        } finally {
            sink.release.countDown()
            active.close()
        }
    }
    listOf("async-retains", "async-cancel-release")
}

fun main(args: Array<String>) = runBlocking {
    val bytes = Files.readAllBytes(Path.of(args.single()))
    val cases = mutableListOf<String>()
    for ((name, backend) in listOf("metal" to BackendPreference.METAL, "wgpu" to BackendPreference.GPU)) {
        for ((mode, compression) in listOf("none" to KvCompression.None,
            "turboquant" to KvCompression.TurboQuant(42uL, true, true))) {
            cases += gpuCases(bytes, backend, compression).map { "$name/$mode/$it" }
        }
        cases += asyncCases(bytes, backend).map { "$name/$it" }
        CeraEngine.fromBytesAsync(bytes, EngineConfig(contextSize = 64uL, backend = backend)).use { engine ->
            val first = generateConversation(engine, listOf(0u, 1u))
            check(first.size == 3 && first == generateConversation(engine, listOf(0u, 1u)))
        }
        cases += "$name/scoped-example"
    }
    CeraEngine.fromBytesAsync(bytes, EngineConfig(contextSize = 64uL, backend = BackendPreference.CPU)).use { cpu ->
        cpu.newSession(SessionConfig(seed = 42uL)).use { first ->
            cpu.newSession(SessionConfig(seed = 42uL)).use { second ->
                first.appendTokens(listOf(0u, 1u))
                second.appendTokens(listOf(0u, 1u))
                check(first.generate(options()).tokens == second.generate(options()).tokens)
            }
        }
    }
    cases += "cpu/sharing"
    println(cases.sorted().joinToString(prefix = "{\"cases\":[", postfix = "]}") { "\"$it\"" })
}

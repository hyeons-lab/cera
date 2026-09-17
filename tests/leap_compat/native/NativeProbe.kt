package nativeprobe

import ai.liquid.leap.HiddenStates
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.CeraEngine
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.FfiException
import uniffi.cera_ffi.Session
import uniffi.cera_ffi.SessionConfig
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path

// Raw CPU boundary experiment. This is not a ModelRunner implementation.
class CeraNativeProbe(
    val session: Session,
) {
    fun hiddenStates(tokens: List<UInt>): HiddenStates {
        val bytes = session.hiddenStatesForTokens(tokens)
        val dimensions = session.hiddenSize().toInt()
        check(bytes.size.toLong() == tokens.size.toLong() * dimensions * 4)
        val data = FloatArray(bytes.size / 4)
        ByteBuffer
            .wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
            .get(data)
        return HiddenStates(data, tokens.size, dimensions)
    }
}

fun main(args: Array<String>) {
    val config = EngineConfig(contextSize = 32uL, backend = BackendPreference.CPU)
    try {
        CeraEngine.fromBytes(byteArrayOf(0, 1, 2), config).close()
        error("Invalid model bytes were accepted")
    } catch (_: FfiException) {
    }
    val engine = CeraEngine.fromBytes(Files.readAllBytes(Path.of(args[0])), config)
    val tokens = engine.encodeText("ab")
    check(tokens == listOf(0u, 1u))
    val configForSession = SessionConfig(maxSeqLen = 32u, seed = 7uL)
    val probe = CeraNativeProbe(engine.newSession(configForSession))
    val control = engine.newSession(configForSession)
    val options = engine.defaultGenerateOpts().copy(maxTokens = 2u, temperature = 0f)
    engine.close()
    try {
        probe.session.appendTokens(tokens)
        control.appendTokens(tokens)
        val position = probe.session.position()
        // Different content and length detect a destructive reset-and-replay of live KV.
        val queryTokens = listOf(1u, 0u, 1u)
        val states = probe.hiddenStates(queryTokens)
        check(states.tokenCount == 3 && states.embeddingDim == 32)
        check(probe.session.position() == position)
        try {
            probe.hiddenStates(listOf(9999u))
            error("Invalid token was accepted")
        } catch (_: FfiException.InvalidToken) {
        }
        check(probe.session.position() == position)
        val copied = states[0]
        val original = states.data[0]
        copied[0] = original + 1f
        check(states.data[0] == original)
        val actual = probe.session.generate(options).tokens
        val expected = control.generate(options).tokens
        check(actual == expected && actual.size == 2)
        check(states.data.all { it.isFinite() })
        val bits = states.data.map { it.toRawBits().toUInt() }
        println("{\"tokens\":$tokens,\"query_tokens\":$queryTokens,\"hidden_bits\":$bits,\"generated\":$actual,\"position\":$position}")
    } finally {
        probe.session.close()
        control.close()
    }
}

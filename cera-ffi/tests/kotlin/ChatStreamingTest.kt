import uniffi.cera_ffi.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList

// Run against a local generative GGUF and the freshly built native library.
fun main(args: Array<String>) = runBlocking {
    val engine = CeraEngine.fromPath(args.single(), EngineConfig(contextSize = 256uL, backend = BackendPreference.CPU))
    val chat = engine.newChatSession(SessionConfig(seed = 42uL))
    try {
        chat.ingest(chatMessageUser("Hi"))
        val empty = GenerateOpts(maxTokens = 0u, temperature = 0.0f)
        repeat(300) {
            chat.stream(empty).toList()
            check(chat.complete(empty).summary.finishReason != FinishReason.Cancelled) {
                "Normal stream completion cancelled the next operation at iteration $it"
            }
        }
        println("PASS: 300 normal streams preserve the next operation")

        // first() abandons collection while native generation can still be active.
        withTimeout(30_000) {
            chat.stream(GenerateOpts(maxTokens = 128u, temperature = 0.0f, ignoreEos = true)).first()
        }
        withTimeout(30_000) {
            while (true) {
                try {
                    chat.phase()
                    break
                } catch (e: FfiException.Busy) {
                    delay(10)
                }
            }
        }
        check(chat.phase() == SessionPhase.INTERRUPTED)
        check(chat.position() < 128uL) { "Abandoned stream decoded its entire budget" }
        chat.reset()
        chat.ingest(chatMessageUser("Hi"))
        check(chat.complete(empty).summary.finishReason != FinishReason.Cancelled)
        println("PASS: abandoned collection interrupts generation and reset recovers")
    } finally {
        chat.destroy()
        engine.destroy()
    }
}

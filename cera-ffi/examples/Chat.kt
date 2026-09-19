import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.ModelLoader
import uniffi.cera_ffi.ModelSource
import uniffi.cera_ffi.SessionConfig
import uniffi.cera_ffi.SessionPhase
import uniffi.cera_ffi.chatMessageSystem
import uniffi.cera_ffi.chatMessageUser

// Multi-turn conversational chat with live KV cache retention in Kotlin.
//
// Demonstrates:
// 1. Loading a generative model and creating an execution session.
// 2. Converting the session into a transactional ChatSession.
// 3. Ingesting user turns and completing responses.
// 4. Retaining live KV context across consecutive turns without recomputation.
// 5. Reclaiming the underlying Session upon completion.
// Run:
//   kotlinc -jvm-target 21 -classpath "jna.jar:kotlinx-coroutines-core.jar" \
//     cera-ffi/bindings/kotlin/uniffi/cera_ffi/cera_ffi.kt cera-ffi/examples/Chat.kt -include-runtime -d Chat.jar
//   java -Djna.library.path=target/debug -cp "Chat.jar:jna.jar:kotlinx-coroutines-core.jar" ChatKt model.gguf
fun main(args: Array<String>) {
    require(args.size >= 1) { "usage: ChatKt <model.gguf>" }
    val modelPath = args[0]
    println("Loading model from: $modelPath")

    ModelLoader(ModelSource.Path(modelPath), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
        loader.buildGenerative().use { model ->
            model.createSession(SessionConfig(seed = 42uL)).use { session ->
                // Convert Session into transactional ChatSession.
                session.intoChat().use { chat ->
                    println("Initial chat phase: ${chat.phase()}")
                    println("Initial position: ${chat.position()}")

                    val opts = GenerateOpts(maxTokens = 64u, temperature = 0.7f)

                    // --- Turn 1: Initialization with System and User messages ---
                    println("\n--- Turn 1 ---")
                    val turn1Messages = listOf(
                        chatMessageSystem("You are a helpful and concise systems engineering assistant."),
                        chatMessageUser("What is a KV cache in LLM inference? Answer in one sentence.")
                    )

                    val summary1 = chat.ingestMessages(turn1Messages)
                    println("Ingested ${summary1.inputTokens} tokens (position: ${summary1.positionBefore} -> ${summary1.positionAfter})")
                    println("Phase after ingest: ${chat.phase()}")

                    val turn1 = chat.complete(opts)
                    println("Assistant: ${turn1.text.trim()}")
                    println("Generated ${turn1.summary.tokensGenerated} tokens (final position: ${chat.position()})")
                    println("Phase after completion: ${chat.phase()}")
                    if (chat.phase() != SessionPhase.TURN_COMPLETE) {
                        println("Turn stopped before its terminal marker; reset or replace messages before a new user turn.")
                        chat.intoSession().use { reclaimedSession ->
                            println("Reclaimed raw session at position ${reclaimedSession.position()}")
                        }
                        return
                    }

                    // --- Turn 2: Warm Continuation ---
                    // The previous context remains in the KV cache; only new user input is ingested.
                    println("\n--- Turn 2 (Warm Continuation) ---")
                    val turn2User = chatMessageUser("When should it be discarded?")
                    val summary2 = chat.ingest(turn2User)
                    println("Ingested ${summary2.inputTokens} new tokens (position: ${summary2.positionBefore} -> ${summary2.positionAfter})")

                    val turn2 = chat.complete(opts)
                    println("Assistant: ${turn2.text.trim()}")
                    println("Generated ${turn2.summary.tokensGenerated} tokens (final position: ${chat.position()})")
                    println("Phase after completion: ${chat.phase()}")
                    // --- Reclaim raw Session ---
                    chat.intoSession().use { reclaimedSession ->
                        println("\nReclaimed raw session at position ${reclaimedSession.position()}")
                    }
                }
            }
        }
    }
}

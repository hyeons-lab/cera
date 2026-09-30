import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.FfiException
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.KvCompression
import uniffi.cera_ffi.Message
import uniffi.cera_ffi.ModelLoader
import uniffi.cera_ffi.ModelSource
import uniffi.cera_ffi.RecoveryOutcome
import uniffi.cera_ffi.Role
import uniffi.cera_ffi.SessionConfig
import uniffi.cera_ffi.SessionPhase

// Run with a GGUF path, turn 1 prompt and turn 2 prompt; optionally --compressed.
fun main(args: Array<String>) {
    require(args.size == 3 || (args.size == 4 && args[3] == "--compressed")) {
        "usage: IngestionRecoveryKt model.gguf prompt1 prompt2 [--compressed]"
    }
    ModelLoader(ModelSource.Path(args[0]), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
        loader.buildGenerative().use { model ->
            val compression =
                if (args.size == 4) KvCompression.TurboQuant(7uL, true, true) else KvCompression.None
            val config = SessionConfig(seed = 42uL, ubatchSize = 1u, kvCompression = compression)
            model.createSession(config).use { session ->
                session.intoChat().use { chat ->
                    // Turn 1 establishes prior conversational context in the KV cache:
                    val turn1 = Message(role = Role.USER, content = args[1])
                    chat.ingest(turn1)
                    val opts = GenerateOpts(maxTokens = 64u)
                    val turn1Result = chat.complete(opts)
                    check(chat.phase() == SessionPhase.TURN_COMPLETE) {
                        "turn 1 did not complete with terminal marker; cannot continue"
                    }

                    // Turn 2: arm cancellation to interrupt prefill after at least one microbatch:
                    val turn2 = Message(role = Role.USER, content = args[2])
                    chat.cancel()
                    try {
                        chat.ingest(turn2)
                        error("use a message longer than one token to demonstrate cancellation")
                    } catch (_: FfiException.Cancelled) {
                        // The original operation error is preserved independently of recovery.
                    }

                    val status = chat.recoveryStatus()
                    check(status.usable) { "recovery failed; reset successfully or recreate the session" }
                    val recovery = checkNotNull(status.lastIngestRecovery)
                    println("recovery: ${recovery.outcome}; position: ${status.position}")

                    chat.clearCancel()
                    when (recovery.outcome) {
                        RecoveryOutcome.UNCHANGED, RecoveryOutcome.RESTORED -> {
                            // Context is preserved in the KV cache; retry ingesting turn 2 directly:
                            chat.ingest(turn2)
                        }
                        RecoveryOutcome.RESET -> {
                            // KV cache was reset; re-supply full conversational history:
                            val assistantMsg = Message(role = Role.ASSISTANT, content = turn1Result.text)
                            chat.replaceMessages(listOf(turn1, assistantMsg, turn2))
                        }
                        RecoveryOutcome.UNUSABLE, RecoveryOutcome.UNKNOWN -> {
                            error("recreate the session before retrying")
                        }
                    }
                    println("retry succeeded; position: ${chat.position()}")
                }
            }
        }
    }
}

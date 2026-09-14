import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.FfiException
import uniffi.cera_ffi.KvCompression
import uniffi.cera_ffi.ModelLoader
import uniffi.cera_ffi.ModelSource
import uniffi.cera_ffi.RecoveryOutcome
import uniffi.cera_ffi.SessionConfig
import uniffi.cera_ffi.UserMessage

// Run with a GGUF path, prefix and multi-token message; optionally --compressed.
fun main(args: Array<String>) {
    require(args.size == 3 || (args.size == 4 && args[3] == "--compressed")) {
        "usage: IngestionRecoveryKt model.gguf prefix message [--compressed]"
    }
    ModelLoader(ModelSource.Path(args[0]), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
        loader.buildGenerative().use { model ->
            model.engine().use { engine ->
                val compression =
                    if (args.size == 4) KvCompression.TurboQuant(7uL, true, true) else KvCompression.None
                model.createSession(SessionConfig(seed = 42uL, ubatchSize = 1u, kvCompression = compression)).use { session ->
                    val prefix = engine.encodeText(args[1])
                    session.appendTokens(prefix)
                    val message = UserMessage(text = args[2])
                    session.cancel()
                    try {
                        session.sendMessage(message)
                        error("use a message longer than one token to demonstrate cancellation")
                    } catch (_: FfiException.Cancelled) {
                        // The original operation error is preserved independently of recovery.
                    }
                    val status = session.recoveryStatus()
                    check(status.usable) { "recovery failed; reset successfully or recreate the session" }
                    val recovery = checkNotNull(status.lastIngestRecovery)
                    println("recovery: ${recovery.outcome}; position: ${status.position}")
                    val replay =
                        when (recovery.outcome) {
                            RecoveryOutcome.UNCHANGED, RecoveryOutcome.RESTORED -> false
                            RecoveryOutcome.RESET -> true
                            RecoveryOutcome.UNUSABLE, RecoveryOutcome.UNKNOWN -> error("recreate the session before retrying")
                        }
                    session.clearCancel()
                    if (replay) session.appendTokens(prefix)
                    session.sendMessage(message)
                    println("retry succeeded; position: ${session.position()}")
                }
            }
        }
    }
}

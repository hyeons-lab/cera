import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.FfiConverterTypeSessionRecoveryStatus
import uniffi.cera_ffi.FfiException
import uniffi.cera_ffi.FinishReason
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.IngestRecovery
import uniffi.cera_ffi.KvCompression
import uniffi.cera_ffi.KvRewindFailure
import uniffi.cera_ffi.ModalitySink
import uniffi.cera_ffi.ModelLoader
import uniffi.cera_ffi.ModelSource
import uniffi.cera_ffi.RecoveryOutcome
import uniffi.cera_ffi.Session
import uniffi.cera_ffi.SessionConfig
import uniffi.cera_ffi.SessionRecoveryStatus
import uniffi.cera_ffi.UserMessage

private class ProbeSink(
    val session: Session,
) : ModalitySink {
    var busy = 0
    var unexpected = 0
    val done = mutableListOf<FinishReason>()

    override fun onThoughtChunk(text: String) {}

    override fun onAudioFrames(
        pcm: List<Float>,
        sampleRate: UInt,
    ) {}

    override fun onTextChunk(text: String) {
        try {
            session.recoveryStatus()
            unexpected++
        } catch (_: FfiException.Busy) {
            busy++
        } catch (_: Exception) {
            unexpected++
        }
    }

    override fun onDone(reason: FinishReason) {
        done.add(reason)
    }
}

fun main(args: Array<String>) {
    require(args.size == 2)
    val backend =
        when (args[1]) {
            "cpu" -> BackendPreference.CPU
            "metal" -> BackendPreference.METAL
            "wgpu" -> BackendPreference.GPU
            else -> error("unknown backend")
        }
    val wide = (1uL shl 40) + 7uL
    val reasons =
        listOf(
            KvRewindFailure.OutOfBounds(wide, 3uL),
            KvRewindFailure.Compressed,
            KvRewindFailure.NonCausal,
            KvRewindFailure.MissingConvolutionCheckpoint(2uL, wide),
            KvRewindFailure.InvalidCacheLayout(5uL, "unequal rows"),
            KvRewindFailure.BackendUnsupported,
            KvRewindFailure.Unknown("future rewind cause"),
        )
    for (reason in reasons) {
        val original =
            SessionRecoveryStatus(
                false,
                3u,
                IngestRecovery(
                    RecoveryOutcome.UNUSABLE,
                    reason,
                    FfiException.OutOfMemory(wide),
                ),
            )
        val decoded = FfiConverterTypeSessionRecoveryStatus.lift(FfiConverterTypeSessionRecoveryStatus.lower(original))
        check(!decoded.usable && decoded.position == 3u)
        check(decoded.lastIngestRecovery?.outcome == RecoveryOutcome.UNUSABLE)
        check(decoded.lastIngestRecovery?.rewindError == reason)
        check((decoded.lastIngestRecovery?.resetError as FfiException.OutOfMemory).requestedBytes == wide)
    }
    var cases = 0
    val compressed = KvCompression.TurboQuant(7uL, true, true)
    for (compression in listOf(KvCompression.None, KvCompression.F16, compressed)) {
        for (operation in 0..2) {
            ModelLoader(ModelSource.Path(args[0]), EngineConfig(backend = backend)).use { loader ->
                loader.buildGenerative().use { model ->
                    ModelLoader(ModelSource.Path(args[0]), EngineConfig(backend = backend)).use { referenceLoader ->
                        referenceLoader.buildGenerative().use { referenceModel ->
                            val config = SessionConfig(maxSeqLen = 32u, seed = 1361uL, ubatchSize = 1u, kvCompression = compression)
                            model.createSession(config).use { actual ->
                                referenceModel.createSession(config).use { reference ->
                                    check(actual.recoveryStatus().lastIngestRecovery == null)
                                    actual.appendTokens(listOf(0u, 1u))
                                    try {
                                        actual.sendMessage(UserMessage(text = "a".repeat(64)))
                                        error("expected overflow")
                                    } catch (
                                        e: FfiException.ContextOverflow,
                                    ) {
                                        check(e.maxSeqLen == 32u && e.by == 34u)
                                    }
                                    val unchanged = actual.recoveryStatus()
                                    check(
                                        unchanged.usable && unchanged.position == 2u &&
                                            unchanged.lastIngestRecovery?.outcome == RecoveryOutcome.UNCHANGED,
                                    )
                                    actual.appendTokens(listOf(1u))
                                    check(actual.recoveryStatus().lastIngestRecovery?.outcome == RecoveryOutcome.UNCHANGED)
                                    actual.reset()
                                    check(actual.recoveryStatus().lastIngestRecovery == null)
                                    actual.appendTokens(listOf(0u, 1u))
                                    actual.cancel()
                                    val message = UserMessage(text = "baba")
                                    val failedSink = ProbeSink(actual)
                                    try {
                                        when (operation) {
                                            0 -> actual.sendMessage(message)
                                            1 -> actual.sendMessageAndGenerate(message, GenerateOpts(maxTokens = 4u))
                                            else -> actual.sendMessageStreaming(message, GenerateOpts(maxTokens = 4u), failedSink)
                                        }
                                        error("expected cancellation")
                                    } catch (_: FfiException.Cancelled) {
                                    }
                                    val status = actual.recoveryStatus()
                                    val expected =
                                        if (args[1] == "cpu" &&
                                            compression != compressed
                                        ) {
                                            RecoveryOutcome.RESTORED
                                        } else {
                                            RecoveryOutcome.RESET
                                        }
                                    check(status.usable && status.lastIngestRecovery?.outcome == expected)
                                    check(status.position == if (expected == RecoveryOutcome.RESTORED) 2u else 0u)
                                    check(status.lastIngestRecovery?.resetError == null)
                                    if (expected == RecoveryOutcome.RESET) {
                                        check(
                                            status.lastIngestRecovery?.rewindError ==
                                                if (args[1] == "cpu") KvRewindFailure.Compressed else KvRewindFailure.BackendUnsupported,
                                        )
                                    }
                                    if (operation == 2) check(failedSink.done == listOf(FinishReason.Cancelled))
                                    try {
                                        actual.sendMessage(message)
                                        error("cancel latch cleared")
                                    } catch (_: FfiException.Cancelled) {
                                    }
                                    actual.clearCancel()
                                    if (expected == RecoveryOutcome.RESET) actual.appendTokens(listOf(0u, 1u))
                                    reference.appendTokens(listOf(0u, 1u))
                                    for (session in listOf(actual, reference)) session.sendMessage(message)
                                    check(actual.recoveryStatus().lastIngestRecovery == null)
                                    check(status.lastIngestRecovery?.outcome == expected)
                                    val options = GenerateOpts(maxTokens = 4u, temperature = 1.0f)
                                    val actualTokens = actual.generate(options).tokens
                                    val referenceTokens = reference.generate(options).tokens
                                    check(actualTokens == referenceTokens && actualTokens.size == 4)
                                    // A same-length wrong prefix must be observable, or token parity is vacuous.
                                    reference.reset()
                                    reference.appendTokens(listOf(1u, 0u))
                                    reference.sendMessage(message)
                                    val wrongPrefixTokens = reference.generate(options).tokens
                                    check(actualTokens != wrongPrefixTokens) { "continuation oracle cannot detect a wrong prefix" }
                                    val sink = ProbeSink(actual)
                                    actual.generateStreaming(GenerateOpts(maxTokens = 2u), sink)
                                    check(sink.busy > 0 && sink.unexpected == 0 && sink.done.size == 1)
                                    cases++
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println("passed $cases recovery cases: ${args[1]}")
}

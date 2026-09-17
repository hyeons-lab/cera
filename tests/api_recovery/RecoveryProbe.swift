import Cera
import Foundation

private func require(_ condition: Bool, _ detail: String) throws {
  if !condition {
    throw NSError(domain: "RecoveryProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: detail])
  }
}

private final class ProbeSink: ModalitySink, @unchecked Sendable {
  let session: Session
  var busy = 0
  var unexpected = 0
  var done: [FinishReason] = []
  init(_ session: Session) { self.session = session }
  func onThoughtChunk(text: String) {}
  func onAudioFrames(pcm: [Float], sampleRate: UInt32) {}
  func onTextChunk(text: String) {
    do {
      _ = try session.recoveryStatus()
      unexpected += 1
    } catch FfiError.Busy { busy += 1 } catch { unexpected += 1 }
  }
  func onDone(reason: FinishReason) { done.append(reason) }
}

@main
struct RecoveryProbe {
  static func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    try require(args.count == 2, "pass fixture and cpu/metal/wgpu")
    let backend: BackendPreference
    switch args[1] {
    case "cpu": backend = .cpu
    case "metal": backend = .metal
    case "wgpu": backend = .gpu
    default: throw NSError(domain: "backend", code: 1)
    }
    let wide: UInt64 = (1 << 40) + 7
    let reasons: [KvRewindFailure] = [
      .outOfBounds(requested: wide, current: 3), .compressed, .nonCausal,
      .missingConvolutionCheckpoint(layer: 2, position: wide),
      .invalidCacheLayout(layer: 5, detail: "unequal rows"), .backendUnsupported,
      .unknown(detail: "future rewind cause"),
    ]
    for reason in reasons {
      let original = SessionRecoveryStatus(
        usable: false, position: 3,
        lastIngestRecovery: IngestRecovery(
          outcome: .unusable, rewindError: reason,
          resetError: .OutOfMemory(requestedBytes: wide)))
      let decoded = try FfiConverterTypeSessionRecoveryStatus_lift(
        FfiConverterTypeSessionRecoveryStatus_lower(original))
      try require(decoded == original, "nested diagnostic wire payload")
    }
    var cases = 0
    for compression: KvCompression in [.none, .f16, .turboQuant(seed: 7, keys: true, values: true)]
    {
      for operation in 0..<3 {
        let model = try ModelLoader(
          source: .path(path: args[0]), config: EngineConfig(backend: backend)
        ).buildGenerative()
        let referenceModel = try ModelLoader(
          source: .path(path: args[0]), config: EngineConfig(backend: backend)
        ).buildGenerative()
        let config = SessionConfig(
          maxSeqLen: 32, kvCompression: compression, seed: 1361, ubatchSize: 1)
        let actual = try model.createSession(config: config)
        let reference = try referenceModel.createSession(config: config)
        try require(try actual.recoveryStatus().lastIngestRecovery == nil, "initial diagnostic")
        try actual.appendTokens(tokens: [0, 1])
        do {
          try actual.sendMessage(message: UserMessage(text: String(repeating: "a", count: 64)))
          throw NSError(domain: "expected overflow", code: 1)
        } catch FfiError.ContextOverflow(let max, let by) {
          try require(max == 32 && by == 34, "overflow payload")
        }
        let unchanged = try actual.recoveryStatus()
        try require(
          unchanged.usable && unchanged.position == 2
            && unchanged.lastIngestRecovery?.outcome == .unchanged, "unchanged")
        try actual.appendTokens(tokens: [1])
        try require(
          try actual.recoveryStatus().lastIngestRecovery?.outcome == .unchanged,
          "raw report lifetime")
        try actual.reset()
        try require(
          try actual.recoveryStatus().lastIngestRecovery == nil, "explicit reset clears report")
        try actual.appendTokens(tokens: [0, 1])
        actual.cancel()
        let message = UserMessage(text: "baba")
        let failedSink = ProbeSink(actual)
        do {
          switch operation {
          case 0: try actual.sendMessage(message: message)
          case 1:
            _ = try actual.sendMessageAndGenerate(
              message: message, opts: GenerateOpts(maxTokens: 4))
          default:
            _ = try actual.sendMessageStreaming(
              message: message, opts: GenerateOpts(maxTokens: 4), sink: failedSink)
          }
          throw NSError(domain: "expected cancellation", code: 1)
        } catch FfiError.Cancelled {}
        let status = try actual.recoveryStatus()
        let expected: RecoveryOutcome =
          args[1] == "cpu" && compression != .turboQuant(seed: 7, keys: true, values: true)
          ? .restored : .reset
        try require(
          status.usable && status.lastIngestRecovery?.outcome == expected, "recovery outcome")
        try require(status.position == (expected == .restored ? 2 : 0), "recovery position")
        try require(status.lastIngestRecovery?.resetError == nil, "unexpected reset failure")
        if expected == .reset {
          try require(
            status.lastIngestRecovery?.rewindError
              == (args[1] == "cpu" ? .compressed : .backendUnsupported), "rewind cause")
        }
        if operation == 2 {
          try require(failedSink.done == [.cancelled], "streaming primary error")
        }
        // Observation must not clear cancellation; another multi-chunk message still fails.
        do {
          try actual.sendMessage(message: message)
          throw NSError(domain: "cancel latch cleared", code: 1)
        } catch FfiError.Cancelled {}
        actual.clearCancel()
        if expected == .reset { try actual.appendTokens(tokens: [0, 1]) }
        try reference.appendTokens(tokens: [0, 1])
        for session in [actual, reference] { try session.sendMessage(message: message) }
        try require(try actual.recoveryStatus().lastIngestRecovery == nil, "success clears report")
        try require(status.lastIngestRecovery?.outcome == expected, "owned snapshot changed")
        let options = GenerateOpts(maxTokens: 4, temperature: 1.0)
        let actualTokens = try actual.generate(opts: options).tokens
        let referenceTokens = try reference.generate(opts: options).tokens
        try require(
          actualTokens == referenceTokens && actualTokens.count == 4, "fresh continuation")
        // A same-length wrong prefix must be observable, or token parity is vacuous.
        try reference.reset()
        try reference.appendTokens(tokens: [1, 0])
        try reference.sendMessage(message: message)
        let wrongPrefixTokens = try reference.generate(opts: options).tokens
        try require(
          actualTokens != wrongPrefixTokens, "continuation oracle cannot detect a wrong prefix")
        let sink = ProbeSink(actual)
        _ = try actual.generateStreaming(opts: GenerateOpts(maxTokens: 2), sink: sink)
        try require(
          sink.busy > 0 && sink.unexpected == 0 && sink.done.count == 1, "callback reentrancy")
        cases += 1
      }
    }
    print("passed \(cases) recovery cases: \(args[1])")
  }
}

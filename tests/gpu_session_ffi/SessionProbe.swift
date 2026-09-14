import Cera
import Foundation

func require(_ value: Bool, _ message: String = "Assertion failed") {
    precondition(value, message)
}

// Each call owns one complete conversation. ARC releases its Session on return.
func generateConversation(engine: CeraEngine, tokens: [UInt32]) throws -> [UInt32] {
    let session = try engine.newSession(config: SessionConfig(seed: 42))
    try session.appendTokens(tokens: tokens)
    return try session.generate(opts: options()).tokens
}

func options() -> GenerateOpts {
    GenerateOpts(maxTokens: 3, temperature: 0, ignoreEos: true, flushEveryTokens: 1)
}

func busy(_ engine: CeraEngine, _ config: SessionConfig) throws {
    do {
        _ = try engine.newSession(config: config)
        preconditionFailure("Expected Busy while the GPU session is retained")
    } catch FfiError.Busy { }
}

func gpuCases(bytes: Data, backend: BackendPreference, compression: KvCompression) async throws -> [String] {
    let load = EngineConfig(contextSize: 64, backend: backend)
    let config = SessionConfig(kvCompression: compression, seed: 42)
    let engine = try await CeraEngine.fromBytesAsync(bytes: bytes, config: load)
    let controlEngine = try await CeraEngine.fromBytesAsync(bytes: bytes, config: load)
    var active: Session? = try engine.newSession(config: config)
    weak let witness = active
    let control = try controlEngine.newSession(config: config)
    try active!.appendTokens(tokens: [0, 1, 0])
    try control.appendTokens(tokens: [0, 1, 0])
    try busy(engine, config)
    try busy(engine, SessionConfig(kvCompression: .turboQuant(seed: 99, keys: true, values: true)))

    let position = active!.position()
    let extracted = try active!.hiddenStatesForTokens(tokens: [1, 0])
    require(extracted.count == 2 * 32 * 4 && active!.position() == position)
    try busy(engine, config)
    active!.cancel()
    try busy(engine, config)
    active!.clearCancel()
    var invalid = options()
    invalid.grammar = "root ::= ("
    do {
        _ = try active!.generate(opts: invalid)
        preconditionFailure("Malformed grammar accepted")
    } catch FfiError.GrammarParse { }
    try busy(engine, config)
    let actual = try active!.generate(opts: options())
    let expected = try control.generate(opts: options())
    require(actual.tokens.count == 3 && actual.tokens == expected.tokens)
    require(active!.position() == control.position())

    try active!.reset()
    require(active!.position() == 0)
    try busy(engine, config)
    active = nil
    require(witness == nil, "Session wrapper did not release")
    do {
        _ = try engine.newSession(config: SessionConfig(
            kvCompression: .turboQuant(seed: 99, keys: true, values: true)))
        preconditionFailure("Compression conflict accepted")
    } catch FfiError.KvCompressionConflict { }
    let successor = try engine.newSession(config: config)
    try successor.appendTokens(tokens: [1, 0, 1])
    let freshEngine = try await CeraEngine.fromBytesAsync(bytes: bytes, config: load)
    let fresh = try freshEngine.newSession(config: config)
    try fresh.appendTokens(tokens: [1, 0, 1])
    require(try successor.generate(opts: options()).tokens == fresh.generate(opts: options()).tokens)

    var parent: CeraEngine? = try await CeraEngine.fromBytesAsync(bytes: bytes, config: load)
    weak let parentWitness = parent
    let retained = try parent!.newSession(config: config)
    parent = nil
    require(parentWitness == nil)
    try retained.appendTokens(tokens: [0, 1])
    require(try retained.generate(opts: options()).tokens.count == 3)
    return ["busy", "busy-before-config", "extraction-retains", "cancel-retains",
            "generation-error-retains", "continuation", "reset-retains", "release",
            "constructor-failure-release", "successor", "parent-release"]
}

final class PausedSink: ModalitySink, @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var paused = false
    private var done: [FinishReason] = []
    func onThoughtChunk(text: String) { }
    func onAudioFrames(pcm: [Float], sampleRate: UInt32) { }
    func onTextChunk(text: String) {
        lock.lock()
        let first = !paused
        paused = true
        lock.unlock()
        if first {
            entered.signal()
            require(release.wait(timeout: .now() + 20) == .success, "Callback release timed out")
        }
    }
    func onDone(reason: FinishReason) {
        lock.lock()
        done.append(reason)
        lock.unlock()
    }
    func waitForCallback() async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: self.entered.wait(timeout: .now() + 20) == .success)
            }
        }
    }
    func completedReasons() -> [FinishReason] {
        lock.lock()
        defer { lock.unlock() }
        return done
    }
}

func launch(_ session: Session, _ sink: PausedSink) -> Task<GenerateSummary, Error> {
    Task.detached { try await session.generateStreamingAsync(opts: options(), sink: sink) }
}

func asyncCases(bytes: Data, backend: BackendPreference) async throws -> [String] {
    let engine = try await CeraEngine.fromBytesAsync(bytes: bytes, config: EngineConfig(contextSize: 64, backend: backend))
    var active: Session? = try engine.newSession(config: SessionConfig(seed: 42))
    weak let witness = active
    try active!.appendTokens(tokens: [0, 1])
    let sink = PausedSink()
    var work: Task<GenerateSummary, Error>? = launch(active!, sink)
    defer { sink.release.signal() }
    require(await sink.waitForCallback(), "No streaming callback")
    active!.cancel()
    active = nil
    require(witness != nil, "Pending call should retain the Swift Session")
    try busy(engine, SessionConfig())
    sink.release.signal()
    let summary = try await work!.value
    work = nil
    require(summary.finishReason == .cancelled)
    require(sink.completedReasons() == [.cancelled])
    require(witness == nil, "Completed call still retains the Session")
    let successor = try engine.newSession(config: SessionConfig(seed: 42))
    try successor.appendTokens(tokens: [1, 0])
    require(try successor.generate(opts: options()).tokens.count == 3)
    return ["async-retains", "async-cancel-release"]
}

@main
struct SessionProbe {
    static func main() async throws {
        let bytes = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        var cases: [String] = []
        for (name, backend) in [("metal", BackendPreference.metal), ("wgpu", .gpu)] {
            for (mode, compression) in [("none", KvCompression.none),
                                        ("turboquant", .turboQuant(seed: 42, keys: true, values: true))] {
                cases += try await gpuCases(bytes: bytes, backend: backend, compression: compression).map { "\(name)/\(mode)/\($0)" }
            }
            cases += try await asyncCases(bytes: bytes, backend: backend).map { "\(name)/\($0)" }
            let engine = try await CeraEngine.fromBytesAsync(bytes: bytes, config: EngineConfig(contextSize: 64, backend: backend))
            let first = try generateConversation(engine: engine, tokens: [0, 1])
            let second = try generateConversation(engine: engine, tokens: [0, 1])
            require(first.count == 3 && first == second)
            cases.append("\(name)/scoped-example")
        }
        let cpu = try await CeraEngine.fromBytesAsync(bytes: bytes, config: EngineConfig(contextSize: 64, backend: .cpu))
        let first = try cpu.newSession(config: SessionConfig(seed: 42))
        let second = try cpu.newSession(config: SessionConfig(seed: 42))
        try first.appendTokens(tokens: [0, 1])
        try second.appendTokens(tokens: [0, 1])
        require(try first.generate(opts: options()).tokens == second.generate(opts: options()).tokens)
        cases.append("cpu/sharing")
        let json = try JSONSerialization.data(withJSONObject: ["cases": cases.sorted()])
        print(String(decoding: json, as: UTF8.self))
    }
}

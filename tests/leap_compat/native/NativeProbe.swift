import Cera
import Foundation
import LeapSDK

// Raw CPU boundary experiment. This is not a ModelRunner implementation.
struct CeraNativeProbe {
  let session: Cera.Session

  func hiddenStates(tokens: [UInt32]) throws -> LeapSDK.HiddenStates {
    let bytes = try session.hiddenStatesForTokens(tokens: tokens)
    let dimensions = Int(session.hiddenSize())
    precondition(bytes.count == tokens.count * dimensions * 4)
    let data = KotlinFloatArray(size: Int32(bytes.count / 4))
    bytes.withUnsafeBytes { raw in
      for index in 0..<(bytes.count / 4) {
        let word = UInt32(
          littleEndian: raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self))
        data.set(index: Int32(index), value: Float(bitPattern: word))
      }
    }
    return LeapSDK.HiddenStates(
      data: data, tokenCount: Int32(tokens.count), embeddingDim: Int32(dimensions))
  }
}

@main
struct NativeProbeMain {
  static func main() throws {
    let config = Cera.EngineConfig(contextSize: 32, backend: .cpu)
    do {
      _ = try Cera.CeraEngine.fromBytes(bytes: Data([0, 1, 2]), config: config)
      fatalError("Invalid model bytes were accepted")
    } catch is Cera.FfiError {}

    let bytes = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    var engine: Cera.CeraEngine? = try Cera.CeraEngine.fromBytes(bytes: bytes, config: config)
    weak let weakEngine = engine
    let tokens = engine!.encodeText(text: "ab")
    precondition(tokens == [0, 1])
    let configForSession = Cera.SessionConfig(maxSeqLen: 32, seed: 7)
    let probe = CeraNativeProbe(session: try engine!.newSession(config: configForSession))
    let control = try engine!.newSession(config: configForSession)
    var options = engine!.defaultGenerateOpts()
    options.maxTokens = 2
    options.temperature = 0
    engine = nil
    precondition(weakEngine == nil, "Swift model wrapper is still retained")

    try probe.session.appendTokens(tokens: tokens)
    try control.appendTokens(tokens: tokens)
    let position = probe.session.position()
    // Different content and length detect a destructive reset-and-replay of live KV.
    let queryTokens: [UInt32] = [1, 0, 1]
    let states = try probe.hiddenStates(tokens: queryTokens)
    precondition(states.tokenCount == 3 && states.embeddingDim == 32)
    precondition(probe.session.position() == position)
    do {
      _ = try probe.hiddenStates(tokens: [9999])
      fatalError("Invalid token was accepted")
    } catch Cera.FfiError.InvalidToken {}
    precondition(probe.session.position() == position)
    let copied = states.get(token: 0)
    let original = states.data.get(index: 0)
    copied.set(index: 0, value: original + 1)
    precondition(states.data.get(index: 0) == original)
    let actual = try probe.session.generate(opts: options).tokens
    let expected = try control.generate(opts: options).tokens
    precondition(actual == expected && actual.count == 2)
    let bits = states.data.toFloatArray().map { $0.bitPattern }
    precondition(states.data.toFloatArray().allSatisfy { $0.isFinite })
    let result: [String: Any] = [
      "tokens": tokens, "query_tokens": queryTokens, "hidden_bits": bits,
      "generated": actual, "position": position,
    ]
    print(
      String(
        data: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]),
        encoding: .utf8)!)
  }
}

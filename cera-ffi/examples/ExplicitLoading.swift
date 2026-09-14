import Cera
import Foundation

// Run with a generative GGUF path and a raw completion prompt.
@main
struct ExplicitLoading {
  static func main() throws {
    guard CommandLine.arguments.count == 3 else {
      throw NSError(
        domain: "ExplicitLoading", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "usage: explicit-loading model.gguf prompt"])
    }
    let loader = ModelLoader(
      source: .path(path: CommandLine.arguments[1]), config: EngineConfig(backend: .cpu))
    let model = try loader.buildGenerative()
    let engine = model.engine()
    let session = try model.createSession(config: SessionConfig(seed: 42))
    let prompt = engine.encodeText(text: CommandLine.arguments[2])
    try session.appendTokens(tokens: prompt)
    let output = try session.generate(opts: GenerateOpts(maxTokens: 32, temperature: 0.7))
    print(engine.decodeTokens(tokens: output.tokens))
  }
}

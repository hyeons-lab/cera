import Cera
import Foundation

// Run with a GGUF path, prefix and multi-token message; optionally --compressed.
@main
struct IngestionRecoveryExample {
  static func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 3 || (args.count == 4 && args[3] == "--compressed") else {
      throw failure("usage: ingestion-recovery model.gguf prefix message [--compressed]")
    }
    let model = try ModelLoader(
      source: .path(path: args[0]), config: EngineConfig(backend: .cpu)
    ).buildGenerative()
    let engine = model.engine()
    let session = try model.createSession(
      config: SessionConfig(
        kvCompression: args.count == 4
          ? .turboQuant(seed: 7, keys: true, values: true) : KvCompression.none,
        seed: 42, ubatchSize: 1))
    let prefix = engine.encodeText(text: args[1])
    try session.appendTokens(tokens: prefix)
    let message = UserMessage(text: args[2])
    session.cancel()
    do {
      try session.sendMessage(message: message)
      throw failure("use a message longer than one token to demonstrate cancellation")
    } catch FfiError.Cancelled {
      // The original operation error is preserved independently of recovery.
    }
    let status = try session.recoveryStatus()
    guard status.usable, let recovery = status.lastIngestRecovery else {
      throw failure("recovery failed; reset successfully or recreate the session")
    }
    print("recovery: \(recovery.outcome); position: \(status.position)")
    let replay: Bool
    switch recovery.outcome {
    case .unchanged, .restored: replay = false
    case .reset: replay = true
    case .unusable, .unknown: throw failure("recreate the session before retrying")
    }
    session.clearCancel()
    if replay { try session.appendTokens(tokens: prefix) }
    try session.sendMessage(message: message)
    print("retry succeeded; position: \(session.position())")
  }

  private static func failure(_ message: String) -> NSError {
    NSError(domain: "IngestionRecovery", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
}

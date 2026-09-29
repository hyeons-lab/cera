import Cera
import Foundation

// Run with a GGUF path, turn 1 prompt and turn 2 prompt; optionally --compressed.
@main
struct IngestionRecoveryExample {
  static func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 3 || (args.count == 4 && args[3] == "--compressed") else {
      throw failure("usage: ingestion-recovery model.gguf prompt1 prompt2 [--compressed]")
    }
    let model = try ModelLoader(
      source: .path(path: args[0]), config: EngineConfig(backend: .cpu)
    ).buildGenerative()
    let session = try model.createSession(
      config: SessionConfig(
        kvCompression: args.count == 4
          ? .turboQuant(seed: 7, keys: true, values: true) : KvCompression.none,
        seed: 42, ubatchSize: 1))
    var chat = try session.intoChat()

    // Turn 1 establishes prior conversational context in the KV cache:
    let turn1 = Message(role: .user, content: args[1])
    _ = try chat.ingest(message: turn1)
    let opts = GenerateOpts(maxTokens: 64)
    let turn1Result = try chat.complete(opts: opts)
    guard try chat.phase() == .turnComplete else {
      throw failure("turn 1 did not complete with terminal marker; cannot continue")
    }

    // Turn 2: arm cancellation to interrupt prefill after at least one microbatch:
    let turn2 = Message(role: .user, content: args[2])
    chat.cancel()
    do {
      _ = try chat.ingest(message: turn2)
      throw failure("use a message longer than one token to demonstrate cancellation")
    } catch FfiError.Cancelled {
      // The original operation error is preserved independently of recovery.
    }

    let status = try chat.recoveryStatus()
    guard status.usable, let recovery = status.lastIngestRecovery else {
      throw failure("recovery failed; reset successfully or recreate the session")
    }
    print("recovery: \(recovery.outcome); position: \(status.position)")

    try chat.clearCancel()
    switch recovery.outcome {
    case .unchanged, .restored:
      // Context is preserved in the KV cache; retry ingesting turn 2 directly:
      _ = try chat.ingest(message: turn2)
    case .reset:
      // KV cache was reset; re-supply full conversational history:
      let assistantMsg = Message(role: .assistant, content: turn1Result.text)
      _ = try chat.replaceMessages(messages: [turn1, assistantMsg, turn2])
    case .unusable, .unknown:
      // Release existing chat coordinator and session before acquiring a replacement:
      let freshSession = try model.createSession(
        config: SessionConfig(
          kvCompression: args.count == 4
            ? .turboQuant(seed: 7, keys: true, values: true) : KvCompression.none,
          seed: 42, ubatchSize: 1))
      chat = try freshSession.intoChat()
      let assistantMsg = Message(role: .assistant, content: turn1Result.text)
      _ = try chat.replaceMessages(messages: [turn1, assistantMsg, turn2])
    }
    print("retry succeeded; position: \(try chat.position())")
  }

  private static func failure(_ message: String) -> NSError {
    NSError(domain: "IngestionRecovery", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
}

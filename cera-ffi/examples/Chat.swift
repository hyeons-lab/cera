import Cera
import Foundation

/// Multi-turn conversational chat with live KV cache retention in Swift.
///
/// Demonstrates:
/// 1. Loading a generative model and creating an execution session.
/// 2. Converting the session into a transactional ChatSession.
/// 3. Ingesting user turns and completing responses.
/// 4. Retaining live KV context across consecutive turns without recomputation.
/// 5. Reclaiming the underlying Session upon completion.
/// Run in a SwiftPM consumer package importing Cera:
///   swift run Chat model.gguf
/// Or direct compilation against pre-built Cera module:
///   swiftc -parse-as-library -I target/debug -L target/debug -lCera \
///     -Xlinker -rpath -Xlinker target/debug cera-ffi/examples/Chat.swift -o Chat
///   ./Chat model.gguf
@main
struct ChatExample {
  static func main() throws {
    guard CommandLine.arguments.count >= 2 else {
      throw NSError(
        domain: "ChatExample", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "usage: Chat <model.gguf>"])
    }
    let modelPath = CommandLine.arguments[1]
    print("Loading model from: \(modelPath)")

    let loader = ModelLoader(
      source: .path(path: modelPath),
      config: EngineConfig(backend: .cpu)
    )
    let model = try loader.buildGenerative()
    let session = try model.createSession(config: SessionConfig(seed: 42))

    // Convert Session into transactional ChatSession.
    let chat = try session.intoChat()
    print("Initial chat phase: \(try chat.phase())")
    print("Initial position: \(try chat.position())")

    let opts = GenerateOpts(maxTokens: 64, temperature: 0.7)

    // --- Turn 1: Initialization with System and User messages ---
    print("\n--- Turn 1 ---")
    let turn1Messages = [
      chatMessageSystem(content: "You are a helpful and concise systems engineering assistant."),
      chatMessageUser(content: "What is a KV cache in LLM inference? Answer in one sentence."),
    ]

    let summary1 = try chat.ingestMessages(messages: turn1Messages)
    print(
      "Ingested \(summary1.inputTokens) tokens (position: \(summary1.positionBefore) -> \(summary1.positionAfter))"
    )
    print("Phase after ingest: \(try chat.phase())")

    let turn1 = try chat.complete(opts: opts)
    print("Assistant: \(turn1.text.trimmingCharacters(in: .whitespacesAndNewlines))")
    print(
      "Generated \(turn1.summary.tokensGenerated) tokens (final position: \(try chat.position()))"
    )
    print("Phase after completion: \(try chat.phase())")
    let firstPhase = try chat.phase()
    guard firstPhase == .turnComplete else {
      print("Turn stopped before its terminal marker; reset or replace messages before a new user turn.")
      let reclaimedSession = try chat.intoSession()
      print("Reclaimed raw session at position \(reclaimedSession.position())")
      return
    }

    // --- Turn 2: Warm Continuation ---
    // The previous context remains in the KV cache; only new user input is ingested.
    print("\n--- Turn 2 (Warm Continuation) ---")
    let turn2User = chatMessageUser(content: "When should it be discarded?")
    let summary2 = try chat.ingest(message: turn2User)
    print(
      "Ingested \(summary2.inputTokens) new tokens (position: \(summary2.positionBefore) -> \(summary2.positionAfter))"
    )

    let turn2 = try chat.complete(opts: opts)
    print("Assistant: \(turn2.text.trimmingCharacters(in: .whitespacesAndNewlines))")
    print(
      "Generated \(turn2.summary.tokensGenerated) tokens (final position: \(try chat.position()))"
    )
    print("Phase after completion: \(try chat.phase())")
    // --- Reclaim raw Session ---
    let reclaimedSession = try chat.intoSession()
    print("\nReclaimed raw session at position \(reclaimedSession.position())")
  }
}

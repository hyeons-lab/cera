import Foundation
import LeapSDK

// Imported async requirements are another valid custom-conformer shape.
final class AsyncRunnerConsumer: NSObject, ModelRunner {
  let delegate: any ModelRunner

  init(delegate: any ModelRunner) { self.delegate = delegate }

  var modelId: String { delegate.modelId }

  func createConversation(systemPrompt: String?) -> any Conversation {
    delegate.createConversation(systemPrompt: systemPrompt)
  }

  func createConversationFromHistory(history: [ChatMessage]) -> any Conversation {
    delegate.createConversationFromHistory(history: history)
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __unload() async throws {
    try await delegate.__unload()
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __getPromptTokensSize(messages: [ChatMessage], addBosToken: Bool) async throws -> KotlinInt {
    try await delegate.__getPromptTokensSize(messages: messages, addBosToken: addBosToken)
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __generateFromConversation(
    conversation: any Conversation, callback: any ModelRunnerGenerationCallback,
    generationOptions: GenerationOptions?
  ) async throws -> any ModelRunnerGenerationHandler {
    try await delegate.__generateFromConversation(
      conversation: conversation, callback: callback, generationOptions: generationOptions)
  }
}

import Foundation
import LeapSDK

// Forwarding implementation checks the protocol requirements without fake inference.
final class RunnerConsumer: NSObject, ModelRunner {
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
  func __unload(completionHandler: @escaping @Sendable (Error?) -> Void) {
    delegate.__unload(completionHandler: completionHandler)
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __getPromptTokensSize(
    messages: [ChatMessage], addBosToken: Bool,
    completionHandler: @escaping @Sendable (KotlinInt?, Error?) -> Void
  ) {
    delegate.__getPromptTokensSize(
      messages: messages, addBosToken: addBosToken, completionHandler: completionHandler)
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __generateFromConversation(
    conversation: any Conversation, callback: any ModelRunnerGenerationCallback,
    generationOptions: GenerationOptions?,
    completionHandler: @escaping @Sendable ((any ModelRunnerGenerationHandler)?, Error?) -> Void
  ) {
    delegate.__generateFromConversation(
      conversation: conversation, callback: callback,
      generationOptions: generationOptions, completionHandler: completionHandler)
  }
}

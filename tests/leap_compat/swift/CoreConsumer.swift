import Foundation
import LeapSDK

// Compile-only consumer: no model files or inference runtime are executed.
func stream(_ conversation: any Conversation) -> SkieSwiftFlow<any MessageResponse> {
  let options = GenerationOptions().with(temperature: 0.2).with(rngSeed: 42).with(maxTokens: 16)
  return conversation.generateResponse(userTextMessage: "hello", generationOptions: options)
}

func consume(_ conversation: any Conversation) async -> [String] {
  let responses: SkieSwiftFlow<any MessageResponse> = stream(conversation)
  var text: [String] = []
  for await response in responses {
    switch onEnum(of: response) {
    case .chunk(let chunk): text.append(chunk.text)
    case .reasoningChunk(let chunk): text.append(chunk.reasoning)
    case .error(let error): text.append(error.message)
    case .complete, .audioSample, .functionCalls: break
    }
  }
  return text
}

func history(_ runner: any ModelRunner, messages: [ChatMessage]) async throws -> KotlinInt {
  let conversation: any Conversation = runner.createConversationFromHistory(history: messages)
  conversation.removeLastMessage()
  let _: [ChatMessage] = conversation.history
  return try await runner.getPromptTokensSize(messages: messages, addBosToken: true)
}

final class CallbackConsumer: NSObject, ModelRunnerGenerationCallback {
  func onResponse(response: any MessageResponse) {}
  func onError(error: KotlinThrowable) {}
}

final class HandlerConsumer: NSObject, ModelRunnerGenerationHandler {
  func stop() {}
}

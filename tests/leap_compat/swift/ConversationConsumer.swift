import Foundation
import LeapSDK

final class ConversationConsumer: NSObject, Conversation {
  let delegate: any Conversation
  init(delegate: any Conversation) { self.delegate = delegate }
  var modelRunner: any ModelRunner { delegate.modelRunner }
  var generatingLock: any Kotlinx_coroutines_coreMutex { delegate.generatingLock }
  var history: [ChatMessage] { delegate.history }
  var functions: [LeapFunction] { delegate.functions }
  var isGenerating: Bool { delegate.isGenerating }

  func appendToHistory(message: ChatMessage) { delegate.appendToHistory(message: message) }
  func removeLastMessage() { delegate.removeLastMessage() }
  func registerFunction(function: LeapFunction) { delegate.registerFunction(function: function) }
  func registerFunctions(functions: [LeapFunction]) {
    delegate.registerFunctions(functions: functions)
  }

  func generateResponse(
    userTextMessage: String, generationOptions: GenerationOptions?
  ) -> SkieSwiftFlow<any MessageResponse> {
    delegate.generateResponse(
      userTextMessage: userTextMessage, generationOptions: generationOptions)
  }
  func generateResponse(
    message: ChatMessage, generationOptions: GenerationOptions?
  ) -> SkieSwiftFlow<any MessageResponse> {
    delegate.generateResponse(message: message, generationOptions: generationOptions)
  }
}

final class ParserConsumer: LeapFunctionCallParser {
  override func parse() -> [LeapFunctionCall] { [] }
  override func dump(functionCalls: [LeapFunctionCall]) -> String { "[]" }
}

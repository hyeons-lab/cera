import LeapSDK

func payloads(samples: KotlinFloatArray, error: KotlinThrowable) -> [any MessageResponse] {
  let call = LeapFunctionCall(name: "lookup", arguments: ["key": "value"])
  let message = ChatMessage(
    role: .assistant, content: [ChatMessageContent.Text(text: "answer")],
    reasoningContent: "reason", functionCalls: [call])
  let stats = GenerationStats(
    promptTokens: 2, completionTokens: 1, totalTokens: 3,
    tokenPerSecond: 1, cachedPromptTokens: 0)
  let complete = MessageResponseComplete(fullMessage: message, finishReason: .stop, stats: stats)
  let _: [ChatMessageContent] = complete.fullMessage.content
  let _: Int64? = complete.stats?.cachedPromptTokens
  let _: String = call.name
  let _: [String: Any?] = call.arguments
  return [
    MessageResponseChunk(text: "answer"), MessageResponseReasoningChunk(reasoning: "reason"),
    MessageResponseFunctionCalls(functionCalls: [call]),
    MessageResponseAudioSample(samples: samples, sampleRate: 16000),
    MessageResponseError(throwable: error, message: "failed"), complete,
  ]
}

import LeapSDK

func requireNonthrowing<S: AsyncSequence>(_ sequence: S) where S.Failure == Never {}

func typedStream(_ conversation: any Conversation) -> SkieSwiftFlow<any MessageResponse> {
  let responses: SkieSwiftFlow<any MessageResponse> = conversation.generateResponse(
    userTextMessage: "hello", generationOptions: nil)
  requireNonthrowing(responses)
  return responses
}

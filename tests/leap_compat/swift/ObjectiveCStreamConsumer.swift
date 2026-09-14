import LeapSDK

func bridgedStream(_ conversation: any Conversation) {
  let responses: SkieSwiftFlow<any MessageResponse> = conversation.generateResponse(
    userTextMessage: "hello", generationOptions: nil)
  _ = responses._bridgeToObjectiveC()
}

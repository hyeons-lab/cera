import LeapSDK

func isTerminal(_ response: any MessageResponse) -> Bool {
  switch onEnum(of: response) {
  case .complete, .error: return true
  case .chunk, .reasoningChunk, .audioSample, .functionCalls: return false
  }
}

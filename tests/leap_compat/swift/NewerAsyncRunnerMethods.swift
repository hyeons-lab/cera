import LeapSDK

extension AsyncRunnerConsumer {
  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __setLoraAdapters(adapters: [LoraAdapterConfig]) async throws {
    try await delegate.__setLoraAdapters(adapters: adapters)
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __hiddenStates(text: String, adapters: [LoraAdapterConfig]) async throws -> HiddenStates {
    try await delegate.__hiddenStates(text: text, adapters: adapters)
  }
}

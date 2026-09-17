import Foundation
import LeapSDK

extension RunnerConsumer {
  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __setLoraAdapters(
    adapters: [LoraAdapterConfig], completionHandler: @escaping @Sendable (Error?) -> Void
  ) {
    delegate.__setLoraAdapters(adapters: adapters, completionHandler: completionHandler)
  }

  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __hiddenStates(
    text: String, adapters: [LoraAdapterConfig],
    completionHandler: @escaping @Sendable (HiddenStates?, Error?) -> Void
  ) {
    delegate.__hiddenStates(text: text, adapters: adapters, completionHandler: completionHandler)
  }
}

import Foundation
import LeapSDK

// Negative control: Swift extension defaults cannot supply Objective-C witnesses.
extension ModelRunner {
  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __setLoraAdapters(adapters: [LoraAdapterConfig]) async throws {
    throw NSError(domain: "probe.unsupported", code: 1)
  }
  // swift-format-ignore: AlwaysUseLowerCamelCase
  func __hiddenStates(text: String, adapters: [LoraAdapterConfig]) async throws -> HiddenStates {
    throw NSError(domain: "probe.unsupported", code: 1)
  }
}

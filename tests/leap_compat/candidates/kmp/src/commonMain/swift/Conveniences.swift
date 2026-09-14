import Foundation

extension GenerationOptions {
  public convenience init() {
    self.init(
      temperature: nil, topP: nil, minP: nil, repetitionPenalty: nil, topK: nil,
      rngSeed: nil, jsonSchemaConstraint: nil, functionCallParser: nil,
      injectSchemaIntoPrompt: true, maxTokens: nil, inlineThinkingTags: false,
      enableThinking: false, extras: nil)
  }

  @discardableResult public func with(temperature: Float) -> GenerationOptions {
    self.temperature = KotlinFloat(value: temperature)
    return self
  }
  @discardableResult public func with(rngSeed: Int64) -> GenerationOptions {
    self.rngSeed = KotlinLong(value: rngSeed)
    return self
  }
  @discardableResult public func with(maxTokens: Int32) -> GenerationOptions {
    self.maxTokens = KotlinInt(value: maxTokens)
    return self
  }
}

extension LiquidInferenceEngineOptions {
  public convenience init(bundlePath: String) {
    self.init(bundlePath: bundlePath, loraAdapters: nil)
  }
  @discardableResult public func with(loraAdapters: [LoraAdapterConfig]?)
    -> LiquidInferenceEngineOptions
  {
    self.loraAdapters = loraAdapters
    return self
  }
}

extension KotlinFloatArray {
  public func toFloatArray() -> [Float] {
    (0..<size).map { get(index: $0) }
  }
}

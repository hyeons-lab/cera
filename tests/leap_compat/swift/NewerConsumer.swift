import LeapSDK

func adaptersAndEmbeddings(_ runner: any ModelRunner) async throws -> [[Float]] {
  let adapters = [LoraAdapterConfig(path: "adapter.gguf", scale: 0.5)]
  let _: LiquidInferenceEngineOptions = LiquidInferenceEngineOptions(bundlePath: "model.bundle")
    .with(loraAdapters: adapters)
  try await runner.setLoraAdapters(adapters: adapters)
  let states: HiddenStates = try await runner.hiddenStates(text: "hello", adapters: [])
  let _: Int32 = states.tokenCount
  let _: Int32 = states.embeddingDim
  let _: [Float] = states.data.toFloatArray()
  let _: KotlinArray<KotlinFloatArray> = states.toMatrix()
  return (0..<states.tokenCount).map { states.get(token: $0).toFloatArray() }
}

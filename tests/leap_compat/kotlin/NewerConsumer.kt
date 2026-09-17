package consumer

import ai.liquid.leap.HiddenStates
import ai.liquid.leap.LoraAdapterConfig
import ai.liquid.leap.ModelLoadingOptions
import ai.liquid.leap.ModelRunner

suspend fun adaptersAndEmbeddings(runner: ModelRunner): Array<FloatArray> {
    val adapters = listOf(LoraAdapterConfig(path = "adapter.gguf", scale = 0.5f))
    val options = ModelLoadingOptions(loraAdapters = adapters)
    runner.setLoraAdapters(options.loraAdapters.orEmpty())
    val states: HiddenStates = runner.hiddenStates(text = "hello", adapters = emptyList())
    val flat: FloatArray = states.data
    val tokens: Int = states.tokenCount
    val dimensions: Int = states.embeddingDim
    check(flat.size.toLong() == tokens.toLong() * dimensions)
    if (tokens > 0) check(states[0].size == dimensions)
    return states.toMatrix()
}

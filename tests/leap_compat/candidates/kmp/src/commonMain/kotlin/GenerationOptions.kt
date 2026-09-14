package ai.liquid.leap

import ai.liquid.leap.function.LeapFunctionCallParser

// Configuration surface only; no translation into Cera sampling is performed here.
data class GenerationOptions(
    var temperature: Float? = null,
    var topP: Float? = null,
    var minP: Float? = null,
    var repetitionPenalty: Float? = null,
    var topK: Int? = null,
    var rngSeed: Long? = null,
    var jsonSchemaConstraint: String? = null,
    var functionCallParser: LeapFunctionCallParser? = null,
    var injectSchemaIntoPrompt: Boolean = true,
    var maxTokens: Int? = null,
    var inlineThinkingTags: Boolean = false,
    var enableThinking: Boolean = false,
    var extras: String? = null,
) {
    companion object {
        fun build(buildAction: GenerationOptions.() -> Unit): GenerationOptions = GenerationOptions().apply(buildAction)
    }
}

// Only these loading members are covered by this increment's consumers.
data class ModelLoadingOptions(
    var contextSize: Int? = 8192,
    var useMmap: Boolean? = null,
    var loraAdapters: List<LoraAdapterConfig>? = null,
)

class LiquidInferenceEngineOptions(
    val bundlePath: String,
    var loraAdapters: List<LoraAdapterConfig>? = null,
)

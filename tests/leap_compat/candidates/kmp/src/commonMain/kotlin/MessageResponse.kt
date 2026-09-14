package ai.liquid.leap.message

import ai.liquid.leap.function.LeapFunctionCall

enum class GenerationFinishReason { STOP, EXCEED_CONTEXT, INTERRUPTED, CONSTRAINT, ERROR }

data class GenerationStats(
    val promptTokens: Long,
    val completionTokens: Long,
    val totalTokens: Long,
    val tokenPerSecond: Float,
    val cachedPromptTokens: Long = 0,
)

sealed interface MessageResponse {
    data class Chunk(
        val text: String,
    ) : MessageResponse

    data class ReasoningChunk(
        val reasoning: String,
    ) : MessageResponse

    data class FunctionCalls(
        val functionCalls: List<LeapFunctionCall>,
    ) : MessageResponse

    data class AudioSample(
        val samples: FloatArray,
        val sampleRate: Int,
    ) : MessageResponse

    data class Complete(
        val fullMessage: ChatMessage,
        val finishReason: GenerationFinishReason,
        val stats: GenerationStats?,
    ) : MessageResponse

    data class Error(
        val throwable: Throwable,
        val message: String = throwable.message ?: throwable.toString(),
    ) : MessageResponse
}

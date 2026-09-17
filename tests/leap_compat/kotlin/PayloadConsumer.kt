package consumer

import ai.liquid.leap.function.LeapFunctionCall
import ai.liquid.leap.message.ChatMessage
import ai.liquid.leap.message.ChatMessageContent
import ai.liquid.leap.message.GenerationFinishReason
import ai.liquid.leap.message.GenerationStats
import ai.liquid.leap.message.MessageResponse

fun payloads(
    samples: FloatArray,
    error: Throwable,
): List<MessageResponse> {
    val call = LeapFunctionCall(name = "lookup", arguments = mapOf("key" to "value"))
    val message =
        ChatMessage(
            role = ChatMessage.Role.ASSISTANT,
            content = listOf(ChatMessageContent.Text("answer")),
            reasoningContent = "reason",
            functionCalls = listOf(call),
        )
    val stats = GenerationStats(promptTokens = 2, completionTokens = 1, totalTokens = 3, tokenPerSecond = 1f, cachedPromptTokens = 0)
    return listOf(
        MessageResponse.Chunk("answer"),
        MessageResponse.ReasoningChunk("reason"),
        MessageResponse.FunctionCalls(listOf(call)),
        MessageResponse.AudioSample(samples, 16000),
        MessageResponse.Error(error, "failed"),
        MessageResponse.Complete(message, GenerationFinishReason.STOP, stats),
    )
}

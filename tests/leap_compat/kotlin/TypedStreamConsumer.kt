package consumer

import ai.liquid.leap.Conversation
import ai.liquid.leap.message.MessageResponse
import kotlinx.coroutines.flow.Flow

fun typedStream(conversation: Conversation): Flow<MessageResponse> =
    conversation.generateResponse(userTextMessage = "hello", generationOptions = null)

fun isTerminal(response: MessageResponse): Boolean =
    when (response) {
        is MessageResponse.Complete, is MessageResponse.Error -> true

        is MessageResponse.Chunk, is MessageResponse.ReasoningChunk,
        is MessageResponse.AudioSample, is MessageResponse.FunctionCalls,
        -> false
    }

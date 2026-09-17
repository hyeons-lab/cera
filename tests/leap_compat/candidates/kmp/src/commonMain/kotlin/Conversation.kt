package ai.liquid.leap

import ai.liquid.leap.function.LeapFunction
import ai.liquid.leap.message.ChatMessage
import ai.liquid.leap.message.MessageResponse
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.sync.Mutex

// Declarations only: implementations must supply history, locking and generation.
interface Conversation {
    val modelRunner: ModelRunner
    val generatingLock: Mutex
    val history: List<ChatMessage>
    val functions: List<LeapFunction>
    val isGenerating: Boolean

    fun appendToHistory(message: ChatMessage)

    fun removeLastMessage()

    fun registerFunction(function: LeapFunction)

    fun registerFunctions(functions: List<LeapFunction>)

    fun generateResponse(
        userTextMessage: String,
        generationOptions: GenerationOptions? = null,
    ): Flow<MessageResponse>

    fun generateResponse(
        message: ChatMessage,
        generationOptions: GenerationOptions? = null,
    ): Flow<MessageResponse>
}

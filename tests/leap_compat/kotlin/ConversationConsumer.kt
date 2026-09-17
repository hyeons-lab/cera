package consumer

import ai.liquid.leap.Conversation
import ai.liquid.leap.GenerationOptions
import ai.liquid.leap.ModelRunner
import ai.liquid.leap.function.LeapFunction
import ai.liquid.leap.function.LeapFunctionCall
import ai.liquid.leap.function.LeapFunctionCallParser
import ai.liquid.leap.message.ChatMessage
import ai.liquid.leap.message.MessageResponse
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.sync.Mutex

class ConversationConsumer(
    private val delegate: Conversation,
) : Conversation {
    override val modelRunner: ModelRunner get() = delegate.modelRunner
    override val generatingLock: Mutex get() = delegate.generatingLock
    override val history: List<ChatMessage> get() = delegate.history
    override val functions: List<LeapFunction> get() = delegate.functions
    override val isGenerating: Boolean get() = delegate.isGenerating

    override fun appendToHistory(message: ChatMessage) = delegate.appendToHistory(message)

    override fun removeLastMessage() = delegate.removeLastMessage()

    override fun registerFunction(function: LeapFunction) = delegate.registerFunction(function)

    override fun registerFunctions(functions: List<LeapFunction>) = delegate.registerFunctions(functions)

    override fun generateResponse(
        userTextMessage: String,
        generationOptions: GenerationOptions?,
    ): Flow<MessageResponse> = delegate.generateResponse(userTextMessage, generationOptions)

    override fun generateResponse(
        message: ChatMessage,
        generationOptions: GenerationOptions?,
    ): Flow<MessageResponse> = delegate.generateResponse(message, generationOptions)
}

class ParserConsumer : LeapFunctionCallParser("<tool>", "</tool>") {
    override fun parse(): List<LeapFunctionCall> = emptyList()

    override fun dump(functionCalls: List<LeapFunctionCall>): String = "[]"
}

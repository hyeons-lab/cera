package ai.liquid.leap

import ai.liquid.leap.message.ChatMessage
import ai.liquid.leap.message.MessageResponse

interface ModelRunner {
    val modelId: String

    fun createConversation(systemPrompt: String? = null): Conversation

    fun createConversationFromHistory(history: List<ChatMessage>): Conversation

    suspend fun unload()

    suspend fun getPromptTokensSize(
        messages: List<ChatMessage>,
        addBosToken: Boolean = true,
    ): Int

    suspend fun generateFromConversation(
        conversation: Conversation,
        callback: GenerationCallback,
        generationOptions: GenerationOptions? = null,
    ): GenerationHandler

    interface GenerationCallback {
        fun onResponse(response: MessageResponse)

        fun onError(error: Throwable)
    }

    interface GenerationHandler {
        fun stop()
    }
}

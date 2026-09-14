package consumer

import ai.liquid.leap.Conversation
import ai.liquid.leap.GenerationOptions
import ai.liquid.leap.ModelLoadingOptions
import ai.liquid.leap.ModelRunner
import ai.liquid.leap.message.ChatMessage
import ai.liquid.leap.message.MessageResponse
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.toList

fun stream(conversation: Conversation): Flow<MessageResponse> =
    conversation.generateResponse(
        userTextMessage = "hello",
        generationOptions =
            GenerationOptions.build {
                temperature = 0.2f
                rngSeed = 42L
                maxTokens = 16
            },
    )

suspend fun consume(conversation: Conversation): List<String> =
    stream(conversation).toList().map { response ->
        when (response) {
            is MessageResponse.Chunk -> response.text
            is MessageResponse.ReasoningChunk -> response.reasoning
            is MessageResponse.Error -> response.message
            is MessageResponse.Complete, is MessageResponse.AudioSample, is MessageResponse.FunctionCalls -> ""
        }
    }

fun loadingOptions(): ModelLoadingOptions = ModelLoadingOptions(contextSize = 2048, useMmap = false)

suspend fun history(
    runner: ModelRunner,
    messages: List<ChatMessage>,
): Int {
    val conversation: Conversation = runner.createConversationFromHistory(messages)
    conversation.removeLastMessage()
    val history: List<ChatMessage> = conversation.history
    return runner.getPromptTokensSize(history)
}

class RunnerConsumer(
    private val delegate: ModelRunner,
) : ModelRunner {
    override val modelId: String get() = delegate.modelId

    override fun createConversation(systemPrompt: String?): Conversation = delegate.createConversation(systemPrompt)

    override fun createConversationFromHistory(history: List<ChatMessage>): Conversation = delegate.createConversationFromHistory(history)

    override suspend fun unload() = delegate.unload()

    override suspend fun getPromptTokensSize(
        messages: List<ChatMessage>,
        addBosToken: Boolean,
    ): Int = delegate.getPromptTokensSize(messages, addBosToken)

    override suspend fun generateFromConversation(
        conversation: Conversation,
        callback: ModelRunner.GenerationCallback,
        generationOptions: GenerationOptions?,
    ): ModelRunner.GenerationHandler = delegate.generateFromConversation(conversation, callback, generationOptions)
}

class CallbackConsumer : ModelRunner.GenerationCallback {
    override fun onResponse(response: MessageResponse) {}

    override fun onError(error: Throwable) {}
}

class HandlerConsumer : ModelRunner.GenerationHandler {
    override fun stop() {}
}

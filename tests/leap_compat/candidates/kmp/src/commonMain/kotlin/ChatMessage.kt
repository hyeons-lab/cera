package ai.liquid.leap.message

import ai.liquid.leap.function.LeapFunctionCall

// Text payloads only; media, serialization and schema conversion remain separate gates.
sealed class ChatMessageContent {
    data class Text(
        val text: String,
    ) : ChatMessageContent()
}

data class ChatMessage(
    val role: Role,
    val content: List<ChatMessageContent>,
    val reasoningContent: String? = null,
    val functionCalls: List<LeapFunctionCall>? = null,
) {
    enum class Role(
        val type: String,
    ) {
        SYSTEM("system"),
        USER("user"),
        ASSISTANT("assistant"),
        TOOL("tool"),
    }

    constructor(role: Role, content: ChatMessageContent) : this(role, listOf(content))
    constructor(role: Role, textContent: String) : this(role, ChatMessageContent.Text(textContent))
}

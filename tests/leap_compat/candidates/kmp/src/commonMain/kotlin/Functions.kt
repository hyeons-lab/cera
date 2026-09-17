package ai.liquid.leap.function

// Nominal types for protocol signatures; schema/parameter serialization is not implemented.
class LeapFunction

data class LeapFunctionCall(
    val name: String,
    val arguments: Map<String, Any?>,
)

open class LeapFunctionCallParser(
    val toolCallStartToken: String,
    val toolCallEndToken: String,
) {
    protected val buffer = StringBuilder()

    fun append(chunk: String) {
        buffer.append(chunk)
    }

    fun clear() {
        buffer.clear()
    }

    open fun parse(): List<LeapFunctionCall> = throw UnsupportedOperationException("Parser implementation is outside the export probe")

    open fun dump(functionCalls: List<LeapFunctionCall>): String =
        throw UnsupportedOperationException("Parser implementation is outside the export probe")
}

package ai.liquid.leap

data class LoraAdapterConfig(
    val path: String,
    val scale: Float = 1.0f,
) {
    init {
        require(scale.isFinite()) { "Adapter scale must be finite" }
    }
}

class HiddenStates(
    val data: FloatArray,
    val tokenCount: Int,
    val embeddingDim: Int,
) {
    init {
        require(tokenCount >= 0 && embeddingDim >= 0) { "Dimensions must be nonnegative" }
        require(tokenCount.toLong() * embeddingDim == data.size.toLong()) { "Hidden-state shape mismatch" }
    }

    operator fun get(token: Int): FloatArray {
        if (token !in 0 until tokenCount) throw IndexOutOfBoundsException("Token index out of bounds")
        return data.copyOfRange(token * embeddingDim, (token + 1) * embeddingDim)
    }

    fun toMatrix(): Array<FloatArray> = Array(tokenCount) { get(it) }
}

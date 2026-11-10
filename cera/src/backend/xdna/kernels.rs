//! Committed kernel artifacts (see `kernels/README.md` for provenance,
//! regeneration, and the drift check).

/// Fused Llama-3.2-1B layer: all seven GEMV stages in one run.
pub const Q4LAYER_XCLBIN: &[u8] = include_bytes!("kernels/q4layer.xclbin");
/// Instruction words for [`Q4LAYER_XCLBIN`].
pub const Q4LAYER_INSTS: &[u8] = include_bytes!("kernels/insts_q4layer.bin");
/// LM-head GEMV at the padded 131072x2048 shape.
pub const Q4HEAD_XCLBIN: &[u8] = include_bytes!("kernels/q4head.xclbin");
/// Instruction words for [`Q4HEAD_XCLBIN`].
pub const Q4HEAD_INSTS: &[u8] = include_bytes!("kernels/insts_q4head.bin");

/// Instruction bytes as little-endian words, as the shim uploads them.
pub fn inst_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

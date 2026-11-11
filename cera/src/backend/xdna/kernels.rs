//! Committed kernel artifacts (see `kernels/README.md` for provenance,
//! regeneration, and the drift check).

/// Fused Llama-3.2-1B layer: all seven GEMV stages + GQA attention in one run.
pub const Q4LAYER_XCLBIN: &[u8] = include_bytes!("kernels/q4layer.xclbin");
/// Instruction words for [`Q4LAYER_XCLBIN`] (full 9-chunk / 648-token KV stream).
pub const Q4LAYER_INSTS: &[u8] = include_bytes!("kernels/insts_q4layer.bin");
/// Exact per-KV-chunk instruction streams (`c1..c8`, `72` cached tokens per chunk)
/// sharing [`Q4LAYER_XCLBIN`].
pub const Q4LAYER_INSTS_C1: &[u8] = include_bytes!("kernels/insts_q4layer_c1.bin");
pub const Q4LAYER_INSTS_C2: &[u8] = include_bytes!("kernels/insts_q4layer_c2.bin");
pub const Q4LAYER_INSTS_C3: &[u8] = include_bytes!("kernels/insts_q4layer_c3.bin");
pub const Q4LAYER_INSTS_C4: &[u8] = include_bytes!("kernels/insts_q4layer_c4.bin");
pub const Q4LAYER_INSTS_C5: &[u8] = include_bytes!("kernels/insts_q4layer_c5.bin");
pub const Q4LAYER_INSTS_C6: &[u8] = include_bytes!("kernels/insts_q4layer_c6.bin");
pub const Q4LAYER_INSTS_C7: &[u8] = include_bytes!("kernels/insts_q4layer_c7.bin");
pub const Q4LAYER_INSTS_C8: &[u8] = include_bytes!("kernels/insts_q4layer_c8.bin");

/// `(kv_chunks, inst_bytes)` buckets in ascending order (`72` tokens per chunk).
pub const Q4LAYER_KV_BUCKETS: [(usize, &[u8]); 9] = [
    (1, Q4LAYER_INSTS_C1),
    (2, Q4LAYER_INSTS_C2),
    (3, Q4LAYER_INSTS_C3),
    (4, Q4LAYER_INSTS_C4),
    (5, Q4LAYER_INSTS_C5),
    (6, Q4LAYER_INSTS_C6),
    (7, Q4LAYER_INSTS_C7),
    (8, Q4LAYER_INSTS_C8),
    (9, Q4LAYER_INSTS),
];

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

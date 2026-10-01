//! Fixtures shared by the hexagon model test modules.

use super::*;
use crate::gguf::GgufBuilder;

/// Minimal GGUF v3 holding one `token_embd.weight` of the given GGML type id.
pub(super) fn embd_gguf(ggml_type: u32, k: usize, rows: usize, data: &[u8]) -> GgufFile {
    GgufBuilder::new()
        .tensor("token_embd.weight", &[k, rows], ggml_type, data.to_vec())
        .unpadded_tail()
        .build()
}

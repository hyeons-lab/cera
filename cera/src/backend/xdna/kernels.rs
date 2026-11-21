//! Committed kernel artifacts (see `kernels/README.md` for provenance,
//! regeneration, and the drift check).

/// Fused Llama-3.2-1B layer: all seven GEMV stages + GQA attention in one run.
pub const Q4LAYER_XCLBIN: &[u8] = include_bytes!("kernels/q4layer.xclbin");
/// Instruction words for [`Q4LAYER_XCLBIN`] (full 64-chunk / 4096-token KV stream).
pub const Q4LAYER_INSTS: &[u8] = include_bytes!("kernels/insts_q4layer.bin");
/// Reduced-KV instruction streams (`64` cached tokens per chunk) sharing
/// [`Q4LAYER_XCLBIN`].
pub const Q4LAYER_INSTS_C1: &[u8] = include_bytes!("kernels/insts_q4layer_c1.bin");
pub const Q4LAYER_INSTS_C2: &[u8] = include_bytes!("kernels/insts_q4layer_c2.bin");
pub const Q4LAYER_INSTS_C4: &[u8] = include_bytes!("kernels/insts_q4layer_c4.bin");
pub const Q4LAYER_INSTS_C8: &[u8] = include_bytes!("kernels/insts_q4layer_c8.bin");
pub const Q4LAYER_INSTS_C12: &[u8] = include_bytes!("kernels/insts_q4layer_c12.bin");
pub const Q4LAYER_INSTS_C16: &[u8] = include_bytes!("kernels/insts_q4layer_c16.bin");
pub const Q4LAYER_INSTS_C24: &[u8] = include_bytes!("kernels/insts_q4layer_c24.bin");
pub const Q4LAYER_INSTS_C32: &[u8] = include_bytes!("kernels/insts_q4layer_c32.bin");
pub const Q4LAYER_INSTS_C48: &[u8] = include_bytes!("kernels/insts_q4layer_c48.bin");

/// `(kv_chunks, inst_bytes)` buckets in ascending order (`64` tokens per chunk).
pub const Q4LAYER_KV_BUCKETS: [(usize, &[u8]); 10] = [
    (1, Q4LAYER_INSTS_C1),
    (2, Q4LAYER_INSTS_C2),
    (4, Q4LAYER_INSTS_C4),
    (8, Q4LAYER_INSTS_C8),
    (12, Q4LAYER_INSTS_C12),
    (16, Q4LAYER_INSTS_C16),
    (24, Q4LAYER_INSTS_C24),
    (32, Q4LAYER_INSTS_C32),
    (48, Q4LAYER_INSTS_C48),
    (64, Q4LAYER_INSTS),
];

/// LM-head GEMV at the padded 131072x2048 shape.
pub const Q4HEAD_XCLBIN: &[u8] = include_bytes!("kernels/q4head.xclbin");
/// Instruction words for [`Q4HEAD_XCLBIN`].
pub const Q4HEAD_INSTS: &[u8] = include_bytes!("kernels/insts_q4head.bin");

/// Fused B=32 Q4 prefill transformer layer: all 7 GEMMs + FlashAttention in one run.
pub const Q4PREFILL_LAYER_XCLBIN: &[u8] = include_bytes!("kernels/q4prefill_layer.xclbin");
/// Bucketed `q4prefill_layer` instruction streams sharing [`Q4PREFILL_LAYER_XCLBIN`].
pub const Q4PREFILL_INSTS_C0: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c0.bin");
pub const Q4PREFILL_INSTS_C1: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c1.bin");
pub const Q4PREFILL_INSTS_C2: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c2.bin");
pub const Q4PREFILL_INSTS_C4: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c4.bin");
pub const Q4PREFILL_INSTS_C8: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c8.bin");
pub const Q4PREFILL_INSTS_C12: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c12.bin");
pub const Q4PREFILL_INSTS_C16: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c16.bin");
pub const Q4PREFILL_INSTS_C24: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c24.bin");
pub const Q4PREFILL_INSTS_C32: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c32.bin");
pub const Q4PREFILL_INSTS_C48: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c48.bin");
pub const Q4PREFILL_INSTS_C63: &[u8] = include_bytes!("kernels/insts_q4prefill_layer_c63.bin");

/// `(kv_chunks, inst_bytes)` buckets in ascending order (`64` past tokens per chunk).
pub const Q4PREFILL_LAYER_KV_BUCKETS: [(usize, &[u8]); 11] = [
    (0, Q4PREFILL_INSTS_C0),
    (1, Q4PREFILL_INSTS_C1),
    (2, Q4PREFILL_INSTS_C2),
    (4, Q4PREFILL_INSTS_C4),
    (8, Q4PREFILL_INSTS_C8),
    (12, Q4PREFILL_INSTS_C12),
    (16, Q4PREFILL_INSTS_C16),
    (24, Q4PREFILL_INSTS_C24),
    (32, Q4PREFILL_INSTS_C32),
    (48, Q4PREFILL_INSTS_C48),
    (63, Q4PREFILL_INSTS_C63),
];

/// Unified fused B=64 Q4 LFM3 prefill layer (Conv + Causal/SWA + Bidir Phase A/B + >8K Paged FA).
pub const Q4PREFILL_LFM3_XCLBIN: &[u8] = include_bytes!("kernels/q4prefill_lfm3.xclbin");
/// Mode-0 (ShortConv + FFN) prefill instruction stream for [`Q4PREFILL_LFM3_XCLBIN`].
pub const Q4PREFILL_LFM3_CONV_INSTS: &[u8] = include_bytes!("kernels/insts_lfm3_pre_conv.bin");

/// Mode-1 (causal/SWA) instruction stream with `0` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C0: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c0.bin");
/// Mode-1 (causal/SWA) instruction stream with `1` past KV chunk.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c1.bin");
/// Mode-1 (causal/SWA) instruction stream with `2` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c2.bin");
/// Mode-1 (causal/SWA) instruction stream with `4` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c4.bin");
/// Mode-1 (causal/SWA) instruction stream with `8` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c8.bin");
/// Mode-1 (causal/SWA) instruction stream with `12` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c12.bin");
/// Mode-1 (causal/SWA) instruction stream with `16` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c16.bin");
/// Mode-1 (causal/SWA) instruction stream with `24` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c24.bin");
/// Mode-1 (causal/SWA) instruction stream with `32` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c32.bin");
/// Mode-1 (causal/SWA) instruction stream with `48` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c48.bin");
/// Mode-1 (causal/SWA) instruction stream with `64` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c64.bin");
/// Mode-1 (causal/SWA) instruction stream with `96` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c96.bin");
/// Mode-1 (causal/SWA) instruction stream with `128` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c128.bin");
/// Mode-1 (causal/SWA) instruction stream with `192` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c192.bin");
/// Mode-1 (causal/SWA) instruction stream with `255` past KV chunks.
pub const Q4PREFILL_LFM3_ATTN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_attn_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM3 causal/SWA prefill.
pub const Q4PREFILL_LFM3_ATTN_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4PREFILL_LFM3_ATTN_INSTS_C0),
    (1, Q4PREFILL_LFM3_ATTN_INSTS_C1),
    (2, Q4PREFILL_LFM3_ATTN_INSTS_C2),
    (4, Q4PREFILL_LFM3_ATTN_INSTS_C4),
    (8, Q4PREFILL_LFM3_ATTN_INSTS_C8),
    (12, Q4PREFILL_LFM3_ATTN_INSTS_C12),
    (16, Q4PREFILL_LFM3_ATTN_INSTS_C16),
    (24, Q4PREFILL_LFM3_ATTN_INSTS_C24),
    (32, Q4PREFILL_LFM3_ATTN_INSTS_C32),
    (48, Q4PREFILL_LFM3_ATTN_INSTS_C48),
    (64, Q4PREFILL_LFM3_ATTN_INSTS_C64),
    (96, Q4PREFILL_LFM3_ATTN_INSTS_C96),
    (128, Q4PREFILL_LFM3_ATTN_INSTS_C128),
    (192, Q4PREFILL_LFM3_ATTN_INSTS_C192),
    (255, Q4PREFILL_LFM3_ATTN_INSTS_C255),
];

/// Mode-2 (Phase A) instruction stream shared by bidirectional and `>8K` prefill.
pub const Q4PREFILL_LFM3_BIDIR_A_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_phase_a.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `1` KV chunk.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c1.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `2` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c2.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `4` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c4.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `8` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c8.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `12` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c12.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `16` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c16.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `24` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c24.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `32` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c32.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `48` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c48.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `64` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c64.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `96` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c96.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `128` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c128.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `192` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c192.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `255` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c255.bin");
/// Mode-3 (bidirectional Phase B) instruction stream with `256` KV chunks.
pub const Q4PREFILL_LFM3_BIDIR_B_INSTS_C256: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_bidir_b_c256.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM3 bidirectional Phase B prefill.
pub const Q4PREFILL_LFM3_BIDIR_B_BUCKETS: [(usize, &[u8]); 15] = [
    (1, Q4PREFILL_LFM3_BIDIR_B_INSTS_C1),
    (2, Q4PREFILL_LFM3_BIDIR_B_INSTS_C2),
    (4, Q4PREFILL_LFM3_BIDIR_B_INSTS_C4),
    (8, Q4PREFILL_LFM3_BIDIR_B_INSTS_C8),
    (12, Q4PREFILL_LFM3_BIDIR_B_INSTS_C12),
    (16, Q4PREFILL_LFM3_BIDIR_B_INSTS_C16),
    (24, Q4PREFILL_LFM3_BIDIR_B_INSTS_C24),
    (32, Q4PREFILL_LFM3_BIDIR_B_INSTS_C32),
    (48, Q4PREFILL_LFM3_BIDIR_B_INSTS_C48),
    (64, Q4PREFILL_LFM3_BIDIR_B_INSTS_C64),
    (96, Q4PREFILL_LFM3_BIDIR_B_INSTS_C96),
    (128, Q4PREFILL_LFM3_BIDIR_B_INSTS_C128),
    (192, Q4PREFILL_LFM3_BIDIR_B_INSTS_C192),
    (255, Q4PREFILL_LFM3_BIDIR_B_INSTS_C255),
    (256, Q4PREFILL_LFM3_BIDIR_B_INSTS_C256),
];

/// Mode-4 (FA partial, first block `blk == 0`, reading `SLOT_NORM_1`) prefill instruction stream.
pub const Q4PREFILL_LFM3_FA_PART0_C256_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_part0_c256.bin");
/// Mode-4 (FA partial, subsequent blocks `blk >= 1`, reading `SLOT_NORM_2`) prefill instruction stream over one full `256`-chunk KV block.
pub const Q4PREFILL_LFM3_FA_PART_C256_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_part_c256.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `0` past KV chunks in final block.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C0: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c0.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `1` past KV chunk.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c1.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `2` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c2.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `4` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c4.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `8` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c8.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `12` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c12.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `16` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c16.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `24` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c24.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `32` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c32.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `48` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c48.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `64` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c64.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `96` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c96.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `128` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c128.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `192` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c192.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `255` past KV chunks.
pub const Q4PREFILL_LFM3_FA_FIN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm3_pre_fa_fin_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM3 `>8K` Mode-5 final-block prefill.
pub const Q4PREFILL_LFM3_FA_FIN_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4PREFILL_LFM3_FA_FIN_INSTS_C0),
    (1, Q4PREFILL_LFM3_FA_FIN_INSTS_C1),
    (2, Q4PREFILL_LFM3_FA_FIN_INSTS_C2),
    (4, Q4PREFILL_LFM3_FA_FIN_INSTS_C4),
    (8, Q4PREFILL_LFM3_FA_FIN_INSTS_C8),
    (12, Q4PREFILL_LFM3_FA_FIN_INSTS_C12),
    (16, Q4PREFILL_LFM3_FA_FIN_INSTS_C16),
    (24, Q4PREFILL_LFM3_FA_FIN_INSTS_C24),
    (32, Q4PREFILL_LFM3_FA_FIN_INSTS_C32),
    (48, Q4PREFILL_LFM3_FA_FIN_INSTS_C48),
    (64, Q4PREFILL_LFM3_FA_FIN_INSTS_C64),
    (96, Q4PREFILL_LFM3_FA_FIN_INSTS_C96),
    (128, Q4PREFILL_LFM3_FA_FIN_INSTS_C128),
    (192, Q4PREFILL_LFM3_FA_FIN_INSTS_C192),
    (255, Q4PREFILL_LFM3_FA_FIN_INSTS_C255),
];

/// Fused B=1 Q4 LFM3 decode layer (Conv + GQA/SWA Attention + FFN).
pub const Q4LAYER_LFM3_XCLBIN: &[u8] = include_bytes!("kernels/q4layer_lfm3.xclbin");
/// Mode-0 (ShortConv + FFN) decode instruction stream for [`Q4LAYER_LFM3_XCLBIN`].
pub const Q4LAYER_LFM3_CONV_INSTS: &[u8] = include_bytes!("kernels/insts_lfm3_dec_conv.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `0` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C0: &[u8] = include_bytes!("kernels/insts_lfm3_dec_attn_c0.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `1` past KV chunk.
pub const Q4LAYER_LFM3_ATTN_INSTS_C1: &[u8] = include_bytes!("kernels/insts_lfm3_dec_attn_c1.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `2` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C2: &[u8] = include_bytes!("kernels/insts_lfm3_dec_attn_c2.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `4` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C4: &[u8] = include_bytes!("kernels/insts_lfm3_dec_attn_c4.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `8` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C8: &[u8] = include_bytes!("kernels/insts_lfm3_dec_attn_c8.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `12` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c12.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `16` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c16.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `24` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c24.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `32` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c32.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `48` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c48.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `64` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c64.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `96` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c96.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `128` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c128.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `192` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c192.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `255` past KV chunks.
pub const Q4LAYER_LFM3_ATTN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_attn_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM3 single-token attention decode.
pub const Q4LAYER_LFM3_KV_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4LAYER_LFM3_ATTN_INSTS_C0),
    (1, Q4LAYER_LFM3_ATTN_INSTS_C1),
    (2, Q4LAYER_LFM3_ATTN_INSTS_C2),
    (4, Q4LAYER_LFM3_ATTN_INSTS_C4),
    (8, Q4LAYER_LFM3_ATTN_INSTS_C8),
    (12, Q4LAYER_LFM3_ATTN_INSTS_C12),
    (16, Q4LAYER_LFM3_ATTN_INSTS_C16),
    (24, Q4LAYER_LFM3_ATTN_INSTS_C24),
    (32, Q4LAYER_LFM3_ATTN_INSTS_C32),
    (48, Q4LAYER_LFM3_ATTN_INSTS_C48),
    (64, Q4LAYER_LFM3_ATTN_INSTS_C64),
    (96, Q4LAYER_LFM3_ATTN_INSTS_C96),
    (128, Q4LAYER_LFM3_ATTN_INSTS_C128),
    (192, Q4LAYER_LFM3_ATTN_INSTS_C192),
    (255, Q4LAYER_LFM3_ATTN_INSTS_C255),
];

/// Mode-2 (Attention Phase A) decode instruction stream for [`Q4LAYER_LFM3_XCLBIN`].
pub const Q4LAYER_LFM3_PHASE_A_INSTS: &[u8] = include_bytes!("kernels/insts_lfm3_dec_phase_a.bin");
/// Mode-3 (FA partial) decode instruction stream over one full `256`-chunk KV block.
pub const Q4LAYER_LFM3_FA_PART_C256_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_part_c256.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `0` past KV chunks in final block.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C0: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c0.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `1` past KV chunk.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c1.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `2` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c2.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `4` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c4.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `8` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c8.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `12` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c12.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `16` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c16.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `24` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c24.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `32` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c32.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `48` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c48.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `64` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c64.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `96` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c96.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `128` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c128.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `192` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c192.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `255` past KV chunks.
pub const Q4LAYER_LFM3_FA_FIN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm3_dec_fa_fin_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM3 `>8K` Mode-4 final-block decode.
pub const Q4LAYER_LFM3_FA_FIN_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4LAYER_LFM3_FA_FIN_INSTS_C0),
    (1, Q4LAYER_LFM3_FA_FIN_INSTS_C1),
    (2, Q4LAYER_LFM3_FA_FIN_INSTS_C2),
    (4, Q4LAYER_LFM3_FA_FIN_INSTS_C4),
    (8, Q4LAYER_LFM3_FA_FIN_INSTS_C8),
    (12, Q4LAYER_LFM3_FA_FIN_INSTS_C12),
    (16, Q4LAYER_LFM3_FA_FIN_INSTS_C16),
    (24, Q4LAYER_LFM3_FA_FIN_INSTS_C24),
    (32, Q4LAYER_LFM3_FA_FIN_INSTS_C32),
    (48, Q4LAYER_LFM3_FA_FIN_INSTS_C48),
    (64, Q4LAYER_LFM3_FA_FIN_INSTS_C64),
    (96, Q4LAYER_LFM3_FA_FIN_INSTS_C96),
    (128, Q4LAYER_LFM3_FA_FIN_INSTS_C128),
    (192, Q4LAYER_LFM3_FA_FIN_INSTS_C192),
    (255, Q4LAYER_LFM3_FA_FIN_INSTS_C255),
];

/// Unified fused B=32 Q4 LFM2 / LFM2.5 prefill layer (`ShortConv` mode 0 + `GQA` mode 1).
pub const Q4PREFILL_LFM2_XCLBIN: &[u8] = include_bytes!("kernels/q4prefill_lfm2.xclbin");
/// Mode-0 (ShortConv + FFN) prefill instruction stream for [`Q4PREFILL_LFM2_XCLBIN`].
pub const Q4PREFILL_LFM2_CONV_INSTS: &[u8] = include_bytes!("kernels/insts_lfm2_pre_conv.bin");

/// Mode-1 (causal Attention + FFN) prefill instruction stream with `0` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C0: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c0.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `1` past KV chunk.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c1.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `2` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c2.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `4` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c4.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `8` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c8.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `12` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c12.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `16` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c16.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `24` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c24.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `32` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c32.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `48` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c48.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `64` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c64.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `96` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c96.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `128` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c128.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `192` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c192.bin");
/// Mode-1 (causal Attention + FFN) prefill instruction stream with `255` past KV chunks.
pub const Q4PREFILL_LFM2_ATTN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_attn_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM2 / LFM2.5 causal prefill.
pub const Q4PREFILL_LFM2_ATTN_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4PREFILL_LFM2_ATTN_INSTS_C0),
    (1, Q4PREFILL_LFM2_ATTN_INSTS_C1),
    (2, Q4PREFILL_LFM2_ATTN_INSTS_C2),
    (4, Q4PREFILL_LFM2_ATTN_INSTS_C4),
    (8, Q4PREFILL_LFM2_ATTN_INSTS_C8),
    (12, Q4PREFILL_LFM2_ATTN_INSTS_C12),
    (16, Q4PREFILL_LFM2_ATTN_INSTS_C16),
    (24, Q4PREFILL_LFM2_ATTN_INSTS_C24),
    (32, Q4PREFILL_LFM2_ATTN_INSTS_C32),
    (48, Q4PREFILL_LFM2_ATTN_INSTS_C48),
    (64, Q4PREFILL_LFM2_ATTN_INSTS_C64),
    (96, Q4PREFILL_LFM2_ATTN_INSTS_C96),
    (128, Q4PREFILL_LFM2_ATTN_INSTS_C128),
    (192, Q4PREFILL_LFM2_ATTN_INSTS_C192),
    (255, Q4PREFILL_LFM2_ATTN_INSTS_C255),
];

/// Mode-2 (Phase A for `>8K` prefill) instruction stream for [`Q4PREFILL_LFM2_XCLBIN`].
pub const Q4PREFILL_LFM2_PHASE_A_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_phase_a.bin");
/// Mode-4 (FA partial over one full `256`-chunk KV block) instruction stream for [`Q4PREFILL_LFM2_XCLBIN`].
pub const Q4PREFILL_LFM2_FA_PART_C256_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_part_c256.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `0` past KV chunks in final block.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C0: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c0.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `1` past KV chunk.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c1.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `2` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c2.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `4` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c4.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `8` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c8.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `12` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c12.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `16` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c16.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `24` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c24.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `32` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c32.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `48` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c48.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `64` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c64.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `96` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c96.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `128` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c128.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `192` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c192.bin");
/// Mode-5 (FA final + O + FFN) prefill instruction stream with `255` past KV chunks.
pub const Q4PREFILL_LFM2_FA_FIN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm2_pre_fa_fin_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM2 / LFM2.5 `>8K` Mode-5 final-block prefill.
pub const Q4PREFILL_LFM2_FA_FIN_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4PREFILL_LFM2_FA_FIN_INSTS_C0),
    (1, Q4PREFILL_LFM2_FA_FIN_INSTS_C1),
    (2, Q4PREFILL_LFM2_FA_FIN_INSTS_C2),
    (4, Q4PREFILL_LFM2_FA_FIN_INSTS_C4),
    (8, Q4PREFILL_LFM2_FA_FIN_INSTS_C8),
    (12, Q4PREFILL_LFM2_FA_FIN_INSTS_C12),
    (16, Q4PREFILL_LFM2_FA_FIN_INSTS_C16),
    (24, Q4PREFILL_LFM2_FA_FIN_INSTS_C24),
    (32, Q4PREFILL_LFM2_FA_FIN_INSTS_C32),
    (48, Q4PREFILL_LFM2_FA_FIN_INSTS_C48),
    (64, Q4PREFILL_LFM2_FA_FIN_INSTS_C64),
    (96, Q4PREFILL_LFM2_FA_FIN_INSTS_C96),
    (128, Q4PREFILL_LFM2_FA_FIN_INSTS_C128),
    (192, Q4PREFILL_LFM2_FA_FIN_INSTS_C192),
    (255, Q4PREFILL_LFM2_FA_FIN_INSTS_C255),
];

/// Fused B=1 Q4 LFM2 / LFM2.5 decode layer (Conv + GQA Attention + FFN).
pub const Q4LAYER_LFM2_XCLBIN: &[u8] = include_bytes!("kernels/q4layer_lfm2.xclbin");
/// Mode-0 (ShortConv + FFN) decode instruction stream for [`Q4LAYER_LFM2_XCLBIN`].
pub const Q4LAYER_LFM2_CONV_INSTS: &[u8] = include_bytes!("kernels/insts_lfm2_dec_conv.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `0` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C0: &[u8] = include_bytes!("kernels/insts_lfm2_dec_attn_c0.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `1` past KV chunk.
pub const Q4LAYER_LFM2_ATTN_INSTS_C1: &[u8] = include_bytes!("kernels/insts_lfm2_dec_attn_c1.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `2` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C2: &[u8] = include_bytes!("kernels/insts_lfm2_dec_attn_c2.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `4` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C4: &[u8] = include_bytes!("kernels/insts_lfm2_dec_attn_c4.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `8` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C8: &[u8] = include_bytes!("kernels/insts_lfm2_dec_attn_c8.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `12` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c12.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `16` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c16.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `24` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c24.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `32` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c32.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `48` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c48.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `64` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c64.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `96` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c96.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `128` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c128.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `192` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c192.bin");
/// Mode-1 (Attention + FFN) decode instruction stream with `255` past KV chunks.
pub const Q4LAYER_LFM2_ATTN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_attn_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM2 / LFM2.5 single-token attention decode.
pub const Q4LAYER_LFM2_KV_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4LAYER_LFM2_ATTN_INSTS_C0),
    (1, Q4LAYER_LFM2_ATTN_INSTS_C1),
    (2, Q4LAYER_LFM2_ATTN_INSTS_C2),
    (4, Q4LAYER_LFM2_ATTN_INSTS_C4),
    (8, Q4LAYER_LFM2_ATTN_INSTS_C8),
    (12, Q4LAYER_LFM2_ATTN_INSTS_C12),
    (16, Q4LAYER_LFM2_ATTN_INSTS_C16),
    (24, Q4LAYER_LFM2_ATTN_INSTS_C24),
    (32, Q4LAYER_LFM2_ATTN_INSTS_C32),
    (48, Q4LAYER_LFM2_ATTN_INSTS_C48),
    (64, Q4LAYER_LFM2_ATTN_INSTS_C64),
    (96, Q4LAYER_LFM2_ATTN_INSTS_C96),
    (128, Q4LAYER_LFM2_ATTN_INSTS_C128),
    (192, Q4LAYER_LFM2_ATTN_INSTS_C192),
    (255, Q4LAYER_LFM2_ATTN_INSTS_C255),
];

/// Mode-2 (Phase A for `>8K` decode) instruction stream for [`Q4LAYER_LFM2_XCLBIN`].
pub const Q4LAYER_LFM2_PHASE_A_INSTS: &[u8] = include_bytes!("kernels/insts_lfm2_dec_phase_a.bin");
/// Mode-3 (FA partial over one full `256`-chunk KV block) instruction stream for [`Q4LAYER_LFM2_XCLBIN`].
pub const Q4LAYER_LFM2_FA_PART_C256_INSTS: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_part_c256.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `0` past KV chunks in final block.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C0: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c0.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `1` past KV chunk.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C1: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c1.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `2` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C2: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c2.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `4` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C4: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c4.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `8` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C8: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c8.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `12` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C12: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c12.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `16` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C16: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c16.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `24` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C24: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c24.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `32` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C32: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c32.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `48` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C48: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c48.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `64` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C64: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c64.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `96` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C96: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c96.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `128` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C128: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c128.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `192` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C192: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c192.bin");
/// Mode-4 (FA final + O + FFN) decode instruction stream with `255` past KV chunks.
pub const Q4LAYER_LFM2_FA_FIN_INSTS_C255: &[u8] =
    include_bytes!("kernels/insts_lfm2_dec_fa_fin_c255.bin");

/// Ascending `(kv_chunks, inst_bytes)` buckets for LFM2 / LFM2.5 `>8K` Mode-4 final-block decode.
pub const Q4LAYER_LFM2_FA_FIN_BUCKETS: [(usize, &[u8]); 15] = [
    (0, Q4LAYER_LFM2_FA_FIN_INSTS_C0),
    (1, Q4LAYER_LFM2_FA_FIN_INSTS_C1),
    (2, Q4LAYER_LFM2_FA_FIN_INSTS_C2),
    (4, Q4LAYER_LFM2_FA_FIN_INSTS_C4),
    (8, Q4LAYER_LFM2_FA_FIN_INSTS_C8),
    (12, Q4LAYER_LFM2_FA_FIN_INSTS_C12),
    (16, Q4LAYER_LFM2_FA_FIN_INSTS_C16),
    (24, Q4LAYER_LFM2_FA_FIN_INSTS_C24),
    (32, Q4LAYER_LFM2_FA_FIN_INSTS_C32),
    (48, Q4LAYER_LFM2_FA_FIN_INSTS_C48),
    (64, Q4LAYER_LFM2_FA_FIN_INSTS_C64),
    (96, Q4LAYER_LFM2_FA_FIN_INSTS_C96),
    (128, Q4LAYER_LFM2_FA_FIN_INSTS_C128),
    (192, Q4LAYER_LFM2_FA_FIN_INSTS_C192),
    (255, Q4LAYER_LFM2_FA_FIN_INSTS_C255),
];

/// Dense `bfloat16 x bfloat16 -> float32` GEMM (`128 x 256` output tile per run).
pub const GEMM_BF16_XCLBIN: &[u8] = include_bytes!("kernels/gemm_bf16.xclbin");
/// `gemm_bf16` instruction stream with `1` `K=64` chunk (`K=64`).
pub const GEMM_BF16_INSTS_K1: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k1.bin");
/// `gemm_bf16` instruction stream with `2` `K=64` chunks (`K=128`).
pub const GEMM_BF16_INSTS_K2: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k2.bin");
/// `gemm_bf16` instruction stream with `4` `K=64` chunks (`K=256`).
pub const GEMM_BF16_INSTS_K4: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k4.bin");
/// `gemm_bf16` instruction stream with `8` `K=64` chunks (`K=512`).
pub const GEMM_BF16_INSTS_K8: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k8.bin");
/// `gemm_bf16` instruction stream with `12` `K=64` chunks (`K=768`).
pub const GEMM_BF16_INSTS_K12: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k12.bin");
/// `gemm_bf16` instruction stream with `16` `K=64` chunks (`K=1024`).
pub const GEMM_BF16_INSTS_K16: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k16.bin");
/// `gemm_bf16` instruction stream with `18` `K=64` chunks (`K=1152`).
pub const GEMM_BF16_INSTS_K18: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k18.bin");
/// `gemm_bf16` instruction stream with `24` `K=64` chunks (`K=1536`).
pub const GEMM_BF16_INSTS_K24: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k24.bin");
/// `gemm_bf16` instruction stream with `32` `K=64` chunks (`K=2048`).
pub const GEMM_BF16_INSTS_K32: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k32.bin");
/// `gemm_bf16` instruction stream with `48` `K=64` chunks (`K=3072`).
pub const GEMM_BF16_INSTS_K48: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k48.bin");
/// `gemm_bf16` instruction stream with `64` `K=64` chunks (`K=4096`).
pub const GEMM_BF16_INSTS_K64: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k64.bin");
/// `gemm_bf16` instruction stream with `68` `K=64` chunks (`K=4352`).
pub const GEMM_BF16_INSTS_K68: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k68.bin");
/// `gemm_bf16` instruction stream with `96` `K=64` chunks (`K=6144`).
pub const GEMM_BF16_INSTS_K96: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k96.bin");
/// `gemm_bf16` instruction stream with `128` `K=64` chunks (`K=8192`).
pub const GEMM_BF16_INSTS_K128: &[u8] = include_bytes!("kernels/insts_gemm_bf16_k128.bin");

/// Ascending `(k_chunks, inst_bytes)` buckets for [`GEMM_BF16_XCLBIN`] (`K = k_chunks * 64`).
pub const GEMM_BF16_BUCKETS: [(usize, &[u8]); 14] = [
    (1, GEMM_BF16_INSTS_K1),
    (2, GEMM_BF16_INSTS_K2),
    (4, GEMM_BF16_INSTS_K4),
    (8, GEMM_BF16_INSTS_K8),
    (12, GEMM_BF16_INSTS_K12),
    (16, GEMM_BF16_INSTS_K16),
    (18, GEMM_BF16_INSTS_K18),
    (24, GEMM_BF16_INSTS_K24),
    (32, GEMM_BF16_INSTS_K32),
    (48, GEMM_BF16_INSTS_K48),
    (64, GEMM_BF16_INSTS_K64),
    (68, GEMM_BF16_INSTS_K68),
    (96, GEMM_BF16_INSTS_K96),
    (128, GEMM_BF16_INSTS_K128),
];

/// Rows per `gemm_bf16` tile (`4` AIE rows × `32` rows per core).
pub const GEMM_BF16_M_TILE: usize = 128;
/// Output columns per `gemm_bf16` tile (`8` AIE columns × `32` columns per core).
pub const GEMM_BF16_N_TILE: usize = 256;
/// Inner-dimension elements per `gemm_bf16` chunk.
pub const GEMM_BF16_K_CHUNK: usize = 64;
/// Number of AIE columns in `gemm_bf16`.
pub const GEMM_BF16_N_COLS: usize = 8;
/// Output columns per AIE column in `gemm_bf16`.
pub const GEMM_BF16_N_CORE: usize = 32;

/// Activation slot stride in `f32` elements inside `q4layer`'s `bo_y`.
pub const Q4LAYER_ACT_SLOT: usize = 8192;
/// Number of activation slots in `q4layer`'s `bo_y`.
pub const Q4LAYER_N_SLOTS: usize = 8;
/// Float offset of attention RMSNorm weights inside slot 0 of `bo_y`.
pub const Q4LAYER_ATTN_NORM_W: usize = 2048;
/// Float offset of packed RoPE cos/sin table inside slot 0 of `bo_y`.
pub const Q4LAYER_ROPE_CS: usize = 4096;
/// Float offset of FFN RMSNorm weights inside slot 4 of `bo_y`.
pub const Q4LAYER_FFN_NORM_W: usize = 2048;
/// Stream tile width in `f32` words (`18,432` bytes).
pub const Q4LAYER_TILE_F32: usize = 4608;
/// Half-tile offset in `f32` words (`9,216` bytes) separating K and V halves.
pub const Q4LAYER_SILU_HALF: usize = 2304;
/// Cached tokens per attention KV chunk.
pub const Q4LAYER_ATTN_CHUNK_T: usize = 64;
/// Total attention KV chunks in `q4layer`'s standalone `bo_kv` layout.
pub const Q4LAYER_ATTN_CHUNKS: usize = 64;
/// Number of GQA KV heads in `q4layer`.
pub const Q4LAYER_N_KV_HEADS: usize = 8;
/// Attention head dimension in `q4layer`.
pub const Q4LAYER_HEAD_DIM: usize = 64;
/// Maximum number of prior cached tokens supported by `q4layer`'s 64x64-token
/// bf16 KV stream (`pos < 4096`, up to 4096 total tokens including current).
pub const MAX_KV_CACHE_TOKENS: usize = Q4LAYER_ATTN_CHUNKS * Q4LAYER_ATTN_CHUNK_T;
/// Per-KV-head stride in `f32` words inside `bo_kv`.
pub const Q4LAYER_KV_HEAD_STRIDE: usize = Q4LAYER_ATTN_CHUNKS * Q4LAYER_TILE_F32;
/// Total `f32` words reserved for the bf16 KV cache inside `bo_kv`.
pub const Q4LAYER_KV_SLOT: usize = Q4LAYER_N_KV_HEADS * Q4LAYER_KV_HEAD_STRIDE;
/// Total byte length of `q4layer`'s standalone `bo_kv` buffer.
pub const Q4LAYER_KV_BYTES: usize = Q4LAYER_KV_SLOT * 4;
/// Total `f32` words in `q4layer`'s `bo_y` activation buffer.
pub const Q4LAYER_ACT_FLOATS: usize = Q4LAYER_N_SLOTS * Q4LAYER_ACT_SLOT;
/// Total byte length of `q4layer`'s `bo_y` activation buffer.
pub const Q4LAYER_ACT_BYTES: usize = Q4LAYER_ACT_FLOATS * 4;
/// Word offset of `n_kv_cache` (`pos`) inside slot 0 of `bo_y`.
pub const Q4LAYER_ATTN_N_OFF: usize = Q4LAYER_ROPE_CS + 128;
/// Word offset of `skip_kv` inside slot 0 of `bo_y`.
pub const Q4LAYER_ATTN_SKIP_KV_OFF: usize = Q4LAYER_ROPE_CS + 131;

/// Batch size per `q4prefill_layer` invocation.
pub const Q4PREFILL_B: usize = 32;
/// Maximum `pos0` supported by `q4prefill_layer` (`63 * 64 = 4032`, covering up to token `4063`).
pub const Q4PREFILL_MAX_POS0: usize = 63 * Q4LAYER_ATTN_CHUNK_T;
/// Total `f32` words per `q4prefill_layer` packed weight buffer (`bo_w`).
pub const Q4PREFILL_W_FLOATS: usize = 1856 * Q4LAYER_TILE_F32;
/// Byte length of `q4prefill_layer`'s `bo_w`.
pub const Q4PREFILL_W_BYTES: usize = Q4PREFILL_W_FLOATS * 4;
/// Float offset of `SLOT_HIDDEN_IN` inside `q4prefill_layer`'s `bo_y`.
pub const Q4PREFILL_SLOT_HIDDEN_IN: usize = 0;
/// Float offset of `SLOT_KV_CUR` (`8 * 4608` f32 words storing bf16 K/V for 32 tokens).
pub const Q4PREFILL_SLOT_KV_CUR: usize = 258_048;
/// Float offset of `SLOT_META` (`norm_w[4096]`, `params[16]`, `rope_cs[4608..6656]`).
pub const Q4PREFILL_SLOT_META: usize = 1_081_344;
/// Total `f32` words in `q4prefill_layer`'s `bo_y` buffer.
pub const Q4PREFILL_ACT_FLOATS: usize = Q4PREFILL_SLOT_META + 12_288;
/// Total byte length of `q4prefill_layer`'s `bo_y` buffer.
pub const Q4PREFILL_ACT_BYTES: usize = Q4PREFILL_ACT_FLOATS * 4;

/// Choose the smallest prefill KV bucket (`>= n_past_tokens.div_ceil(64)`).
pub fn select_q4prefill_kv_bucket(bucket_chunks: &[usize], n_past_tokens: usize) -> (usize, usize) {
    assert!(
        !bucket_chunks.is_empty(),
        "q4prefill bucket_chunks must be non-empty"
    );
    assert!(
        n_past_tokens <= Q4PREFILL_MAX_POS0,
        "n_past_tokens ({n_past_tokens}) exceeds Q4PREFILL_MAX_POS0 ({Q4PREFILL_MAX_POS0})"
    );
    let needed = n_past_tokens.div_ceil(Q4LAYER_ATTN_CHUNK_T);
    let idx = bucket_chunks
        .iter()
        .position(|&c| c >= needed)
        .expect("q4prefill bucket_chunks must reach Q4PREFILL_MAX_POS0 / 64");
    (idx, bucket_chunks[idx])
}

/// Populate static attention + FFN RMSNorm weights inside `q4prefill_layer`'s `bo_y`.
pub fn init_q4prefill_bo_y_norms(buf: &mut [u8], attn_norm: &[f32], ffn_norm: &[f32]) {
    assert_eq!(buf.len(), Q4PREFILL_ACT_BYTES);
    assert_eq!(attn_norm.len(), 2048);
    assert_eq!(ffn_norm.len(), 2048);
    buf.fill(0);
    write_le_f32_slice(buf, Q4PREFILL_SLOT_META, attn_norm);
    write_le_f32_slice(buf, Q4PREFILL_SLOT_META + 2048, ffn_norm);
}

/// Write per-chunk metadata (`params[0..8]` and `32 * 64` RoPE cos/sin floats) into `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn write_q4prefill_step_params(
    buf: &mut [u8],
    pos0: usize,
    n_valid: usize,
    bucket_chunks: usize,
    n_past_tokens: usize,
    is_causal: bool,
    rope_mode: i32,
    eps: f32,
    rope_cs_32x64: &[f32],
) {
    assert_eq!(buf.len(), Q4PREFILL_ACT_BYTES);
    assert!((1..=Q4PREFILL_B).contains(&n_valid));
    assert_eq!(rope_cs_32x64.len(), Q4PREFILL_B * Q4LAYER_HEAD_DIM);
    let params_off = Q4PREFILL_SLOT_META + 4096;
    write_le_i32(buf, params_off, pos0 as i32);
    write_le_i32(buf, params_off + 1, n_valid as i32);
    write_le_i32(buf, params_off + 2, bucket_chunks as i32);
    write_le_i32(buf, params_off + 3, n_past_tokens as i32);
    write_le_i32(buf, params_off + 4, if is_causal { 1 } else { 0 });
    write_le_i32(buf, params_off + 5, rope_mode);
    write_le_i32(buf, params_off + 6, 0);
    write_le_i32(buf, params_off + 7, eps.to_bits() as i32);
    write_le_f32_slice(buf, Q4PREFILL_SLOT_META + 4608, rope_cs_32x64);
}

/// Hidden dimension fixed in all LFM3 XDNA kernels.
pub const LFM3_H: usize = 2048;
/// Total query projection width in LFM3 XDNA kernels (`16 * 128`).
pub const LFM3_Q_DIM: usize = 2048;
/// Total K/V projection width in LFM3 XDNA kernels (`4 * 128`).
pub const LFM3_KV_DIM: usize = 512;
/// Number of query attention heads in LFM3 XDNA kernels.
pub const LFM3_N_HEADS: usize = 16;
/// Number of GQA key/value heads in LFM3 XDNA kernels.
pub const LFM3_N_KV_HEADS: usize = 4;
/// Attention head dimension in LFM3 XDNA kernels.
pub const LFM3_HEAD_DIM: usize = 128;
/// Unpadded FFN intermediate width in LFM3 GGUFs.
pub const LFM3_FFN_SRC: usize = 7680;
/// Padded FFN intermediate width in LFM3 XDNA weight layouts.
pub const LFM3_FFN_PAD: usize = 8192;
/// ShortConv kernel tap count (`l_cache = 7`).
pub const LFM3_CONV_TAP: usize = 7;
/// Batch size per LFM3 prefill kernel run.
pub const LFM3_PREFILL_B: usize = 64;
/// Past KV chunk token stride in LFM3 XDNA kernels.
pub const LFM3_KV_CHUNK_T: usize = 32;
/// Total KV chunks allocated per head in one LFM3 `bo_kv` block.
pub const LFM3_MAX_KV_CHUNKS: usize = 256;
/// Tokens stored in one `256`-chunk LFM3 `bo_kv` block (`8192`).
pub const LFM3_BLOCK_TOKENS: usize = LFM3_MAX_KV_CHUNKS * LFM3_KV_CHUNK_T;
/// Maximum sequence length supported across paged LFM3 KV blocks (`i32::MAX` kernel ABI bound,
/// practically bounded by `cfg.max_seq_len` and host RAM for on-demand 8K KV blocks).
pub const LFM3_MAX_TOKENS: usize = i32::MAX as usize;
/// Byte size of one LFM3 layer weight `bo_w` (`1994` tiles of `4608` f32 words, including
/// `4` trailing zero half-tiles for FFN `p == 15` padding).
pub const LFM3_W_BYTES: usize = 1994 * Q4LAYER_TILE_F32 * 4;
/// Byte size of one LFM3 layer `bo_kv` (`256 * 4 * 4608` f32 words).
pub const LFM3_KV_BYTES: usize = LFM3_MAX_KV_CHUNKS * LFM3_N_KV_HEADS * Q4LAYER_TILE_F32 * 4;
/// Byte size of one LFM3 `bo_kvo` output scratch (`8 * 4608` f32 words).
pub const LFM3_KVO_BYTES: usize = 8 * Q4LAYER_TILE_F32 * 4;

/// Float offset of `SLOT_FA_Q` inside unified `B=64` LFM3 prefill `bo_y` (`73_728` floats).
pub const LFM3_PREFILL_SLOT_Q_BF16: usize = 737_280;
/// Float offset of `SLOT_ATTN_GATE` inside unified `B=64` LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_ATTN_GATE: usize = 1_019_904;
/// Float offset of `SLOT_NORM_0` (`attn_norm[2048]` + `g_params[16]`) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_NORM_0: usize = 1_022_208;
/// Float offset of `SLOT_NORM_1` (`post_attn_norm[2048]`) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_NORM_1: usize = 1_024_512;
/// Float offset of `SLOT_NORM_2` (`ffn_norm[2048]`) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_NORM_2: usize = 1_026_816;
/// Float offset of `SLOT_NORM_3` (`post_ffn_norm[2048]`) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_NORM_3: usize = 1_029_120;
/// Float offset of `SLOT_ROPE_PKT_0A` (tok `0..16` cos/sin + `q_norm` + `k_norm`) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_ROPE_PKT_0A: usize = 1_031_424;
/// Float offset of `SLOT_ROPE_PKT_0B` (tok `16..32` cos/sin) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_ROPE_PKT_0B: usize = 1_033_728;
/// Float offset of `SLOT_ROPE_PKT_1A` (tok `32..48` cos/sin + `q_norm` + `k_norm`) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_ROPE_PKT_1A: usize = 1_036_032;
/// Float offset of `SLOT_ROPE_PKT_1B` (tok `48..64` cos/sin) inside LFM3 prefill `bo_y`.
pub const LFM3_PREFILL_SLOT_ROPE_PKT_1B: usize = 1_038_336;
/// Byte length of unified `B=64` LFM3 prefill `bo_y` (`1_040_640 * 4`).
pub const LFM3_PREFILL_ACT_BYTES: usize = (LFM3_PREFILL_SLOT_ROPE_PKT_1B + 2304) * 4;
/// Byte length of LFM3 prefill `bo_yn` (`64 * 2048` floats).
pub const LFM3_PREFILL_YN_BYTES: usize = LFM3_PREFILL_B * LFM3_H * 4;

/// Float offset of `SLOT_NORM_0` inside LFM3 decode `bo_y`.
pub const LFM3_DEC_SLOT_NORM_0: usize = 41_472;
/// Float offset of `SLOT_NORM_1` inside LFM3 decode `bo_y`.
pub const LFM3_DEC_SLOT_NORM_1: usize = 46_080;
/// Float offset of `SLOT_ROPE_PKT` inside LFM3 decode `bo_y`.
pub const LFM3_DEC_SLOT_ROPE_PKT: usize = 50_688;
/// Float offset of `SLOT_DEC_FIN_NORM` (`post_attn_norm[2048..4095]` + `params_fin[4096..4111]`) inside LFM3 decode `bo_y`.
pub const LFM3_DEC_SLOT_FIN_NORM: usize = 55_296;
/// Byte length of LFM3 decode `bo_y`.
pub const LFM3_DEC_ACT_BYTES: usize = (LFM3_DEC_SLOT_FIN_NORM + 4608) * 4;
/// Byte length of LFM3 decode `bo_yn` (`2048` floats).
pub const LFM3_DEC_YN_BYTES: usize = LFM3_H * 4;

/// Choose the smallest `(bucket_idx, chunk_cap)` whose capacity reaches `needed_chunks`.
pub fn select_lfm3_kv_bucket(bucket_chunks: &[usize], needed_chunks: usize) -> (usize, usize) {
    assert!(
        !bucket_chunks.is_empty(),
        "lfm3 bucket_chunks must be non-empty"
    );
    let idx = bucket_chunks
        .iter()
        .position(|&c| c >= needed_chunks)
        .unwrap_or_else(|| {
            panic!(
                "lfm3 needed_chunks ({needed_chunks}) exceeds maximum bucket capacity ({})",
                bucket_chunks.last().copied().unwrap_or(0)
            )
        });
    (idx, bucket_chunks[idx])
}

/// Write static pre/post RMSNorm weights and optional Q/K norm weights into LFM3 `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn init_lfm3_bo_y_norms(
    buf: &mut [u8],
    slot_norm_0: usize,
    slot_norm_1: usize,
    slot_rope_pkt: Option<usize>,
    attn_norm: &[f32],
    post_attn_norm: &[f32],
    ffn_norm: &[f32],
    post_ffn_norm: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
) {
    assert_eq!(attn_norm.len(), LFM3_H);
    assert_eq!(post_attn_norm.len(), LFM3_H);
    assert_eq!(ffn_norm.len(), LFM3_H);
    assert_eq!(post_ffn_norm.len(), LFM3_H);
    write_le_f32_slice(buf, slot_norm_0, attn_norm);
    write_le_f32_slice(buf, slot_norm_0 + LFM3_H, post_attn_norm);
    write_le_f32_slice(buf, slot_norm_1, ffn_norm);
    write_le_f32_slice(buf, slot_norm_1 + LFM3_H, post_ffn_norm);
    if buf.len() >= LFM3_DEC_ACT_BYTES {
        write_le_f32_slice(buf, LFM3_DEC_SLOT_FIN_NORM + LFM3_H, post_attn_norm);
    }
    if let (Some(_rope_slot), Some(qn), Some(kn)) = (slot_rope_pkt, q_norm, k_norm) {
        assert_eq!(qn.len(), LFM3_HEAD_DIM);
        assert_eq!(kn.len(), LFM3_HEAD_DIM);
        write_le_f32_slice(buf, slot_norm_0 + 4240, qn);
        write_le_f32_slice(buf, slot_norm_0 + 4368, kn);
    }
}

/// Write static pre/post RMSNorm weights and optional Q/K norm weights into unified `B=64` LFM3 prefill `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn init_lfm3_prefill_bo_y_norms(
    buf: &mut [u8],
    attn_norm: &[f32],
    post_attn_norm: &[f32],
    ffn_norm: &[f32],
    post_ffn_norm: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
) {
    assert_eq!(attn_norm.len(), LFM3_H);
    assert_eq!(post_attn_norm.len(), LFM3_H);
    assert_eq!(ffn_norm.len(), LFM3_H);
    assert_eq!(post_ffn_norm.len(), LFM3_H);
    write_le_f32_slice(buf, LFM3_PREFILL_SLOT_NORM_0, attn_norm);
    write_le_f32_slice(buf, LFM3_PREFILL_SLOT_NORM_1, post_attn_norm);
    write_le_f32_slice(buf, LFM3_PREFILL_SLOT_NORM_2, ffn_norm);
    write_le_f32_slice(buf, LFM3_PREFILL_SLOT_NORM_3, post_ffn_norm);
    if let (Some(qn), Some(kn)) = (q_norm, k_norm) {
        assert_eq!(qn.len(), LFM3_HEAD_DIM);
        assert_eq!(kn.len(), LFM3_HEAD_DIM);
        write_le_f32_slice(buf, LFM3_PREFILL_SLOT_ROPE_PKT_0A + 2048, qn);
        write_le_f32_slice(buf, LFM3_PREFILL_SLOT_ROPE_PKT_0A + 2176, kn);
        write_le_f32_slice(buf, LFM3_PREFILL_SLOT_ROPE_PKT_1A + 2048, qn);
        write_le_f32_slice(buf, LFM3_PREFILL_SLOT_ROPE_PKT_1A + 2176, kn);
    }
}

/// LFM2 / LFM2.5 model hidden dimension (`H = 2048`).
pub const LFM2_H: usize = 2048;
/// LFM2 / LFM2.5 feed-forward intermediate dimension (`FFN = 8192`).
pub const LFM2_FFN: usize = 8192;
/// LFM2 / LFM2.5 query head count (`32`).
pub const LFM2_N_HEADS: usize = 32;
/// LFM2 / LFM2.5 KV head count (`8`).
pub const LFM2_N_KV_HEADS: usize = 8;
/// LFM2 / LFM2.5 head dimension (`64`).
pub const LFM2_HEAD_DIM: usize = 64;
/// LFM2 / LFM2.5 query projection output width (`32 * 64 = 2048`).
pub const LFM2_Q_DIM: usize = LFM2_N_HEADS * LFM2_HEAD_DIM;
/// LFM2 / LFM2.5 KV projection output width (`8 * 64 = 512`).
pub const LFM2_KV_DIM: usize = LFM2_N_KV_HEADS * LFM2_HEAD_DIM;
/// LFM2 / LFM2.5 ShortConv tap count (`l_cache = 3`).
pub const LFM2_CONV_TAPS: usize = 3;
/// LFM2 / LFM2.5 rolling ShortConv state frames (`l_cache - 1 = 2`).
pub const LFM2_D_CONV: usize = LFM2_CONV_TAPS - 1;
/// Tokens per LFM2 / LFM2.5 KV chunk (`32`).
pub const LFM2_KV_CHUNK_T: usize = 32;
/// Tokens per LFM2 / LFM2.5 NPU prefill batch (`64`).
pub const LFM2_PREFILL_CHUNK_T: usize = 64;
/// Maximum KV chunks per LFM2 / LFM2.5 attention layer (`256`, covering `8192` tokens).
pub const LFM2_MAX_KV_CHUNKS: usize = 256;
/// Tokens stored in one `256`-chunk LFM2 `bo_kv` block (`8192`).
pub const LFM2_BLOCK_TOKENS: usize = LFM2_MAX_KV_CHUNKS * LFM2_KV_CHUNK_T;
/// Maximum sequence tokens supported across paged LFM2 KV blocks (`i32::MAX` kernel ABI bound,
/// practically bounded by `cfg.max_seq_len` and host RAM for on-demand 8K KV blocks).
pub const LFM2_MAX_TOKENS: usize = i32::MAX as usize;
/// Byte size of one LFM2 layer weight `bo_w` (`2056` tiles of `4608` f32 words).
pub const LFM2_W_BYTES: usize = 2056 * Q4LAYER_TILE_F32 * 4;
/// Byte size of one LFM2 attention layer `bo_kv` (`256 * 8 * 4608` f32 words).
pub const LFM2_KV_BYTES: usize = LFM2_MAX_KV_CHUNKS * LFM2_N_KV_HEADS * Q4LAYER_TILE_F32 * 4;
/// Byte size of one LFM2 `bo_kvo` output scratch (`8 * 4608` f32 words).
pub const LFM2_KVO_BYTES: usize = 8 * Q4LAYER_TILE_F32 * 4;

/// Float offset of `SLOT_NORM_0` inside LFM2 prefill `bo_y`.
pub const LFM2_PREFILL_SLOT_NORM_0: usize = 1_146_880;
/// Float offset of `SLOT_NORM_1` inside LFM2 prefill `bo_y`.
pub const LFM2_PREFILL_SLOT_NORM_1: usize = 1_149_184;
/// Float offset of `SLOT_ROPE_PKT_0` (`sub=0` cos/sin + q/k norm) inside LFM2 prefill `bo_y`.
pub const LFM2_PREFILL_SLOT_ROPE_PKT_0: usize = 1_151_488;
/// Float offset of `SLOT_ROPE_PKT_1` (`sub=1` cos/sin) inside LFM2 prefill `bo_y`.
pub const LFM2_PREFILL_SLOT_ROPE_PKT_1: usize = 1_153_792;
/// Byte length of LFM2 prefill `bo_y`.
pub const LFM2_PREFILL_ACT_BYTES: usize = (LFM2_PREFILL_SLOT_ROPE_PKT_1 + 2304) * 4;
/// Byte length of LFM2 prefill `bo_yn` (`64 * 2048` floats).
pub const LFM2_PREFILL_YN_BYTES: usize = LFM2_PREFILL_CHUNK_T * LFM2_H * 4;

/// Float offset of `SLOT_NORM_0` inside LFM2 decode `bo_y`.
pub const LFM2_DEC_SLOT_NORM_0: usize = 41_472;
/// Float offset of `SLOT_NORM_1` inside LFM2 decode `bo_y`.
pub const LFM2_DEC_SLOT_NORM_1: usize = 46_080;
/// Float offset of `SLOT_ROPE_PKT` inside LFM2 decode `bo_y`.
pub const LFM2_DEC_SLOT_ROPE_PKT: usize = 50_688;
/// Byte length of LFM2 decode `bo_y`.
pub const LFM2_DEC_ACT_BYTES: usize = (LFM2_DEC_SLOT_ROPE_PKT + 4608) * 4;
/// Byte length of LFM2 decode `bo_yn` (`2048` floats).
pub const LFM2_DEC_YN_BYTES: usize = LFM2_H * 4;

/// Write static pre-RMSNorm and optional Q/K norm weights into LFM2 prefill or decode `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn init_lfm2_bo_y_norms(
    buf: &mut [u8],
    slot_norm_0: usize,
    slot_norm_1: usize,
    slot_rope_pkt: Option<usize>,
    attn_norm: &[f32],
    ffn_norm: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
) {
    assert_eq!(attn_norm.len(), LFM2_H);
    assert_eq!(ffn_norm.len(), LFM2_H);
    write_le_f32_slice(buf, slot_norm_0, attn_norm);
    write_le_f32_slice(buf, slot_norm_1, ffn_norm);
    if let (Some(_rope_slot), Some(qn), Some(kn)) = (slot_rope_pkt, q_norm, k_norm) {
        assert_eq!(qn.len(), LFM2_HEAD_DIM);
        assert_eq!(kn.len(), LFM2_HEAD_DIM);
        write_le_f32_slice(buf, slot_norm_0 + 4240, qn);
        write_le_f32_slice(buf, slot_norm_0 + 4368, kn);
    }
}

/// Write static pre-RMSNorm and optional Q/K norm weights into LFM2 `B=64` prefill `bo_y`.
pub fn init_lfm2_prefill_bo_y_norms(
    buf: &mut [u8],
    attn_norm: &[f32],
    ffn_norm: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
) {
    assert_eq!(attn_norm.len(), LFM2_H);
    assert_eq!(ffn_norm.len(), LFM2_H);
    write_le_f32_slice(buf, LFM2_PREFILL_SLOT_NORM_0, attn_norm);
    write_le_f32_slice(buf, LFM2_PREFILL_SLOT_NORM_1, ffn_norm);
    if let (Some(qn), Some(kn)) = (q_norm, k_norm) {
        assert_eq!(qn.len(), LFM2_HEAD_DIM);
        assert_eq!(kn.len(), LFM2_HEAD_DIM);
        write_le_f32_slice(buf, LFM2_PREFILL_SLOT_ROPE_PKT_0 + 2048, qn);
        write_le_f32_slice(buf, LFM2_PREFILL_SLOT_ROPE_PKT_0 + 2112, kn);
        write_le_f32_slice(buf, LFM2_PREFILL_SLOT_ROPE_PKT_1 + 2048, qn);
        write_le_f32_slice(buf, LFM2_PREFILL_SLOT_ROPE_PKT_1 + 2112, kn);
    }
}

/// Write `g_params[0..8]` and packed `B=64` RoPE cos/sin into an LFM2 prefill `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn write_lfm2_prefill_step_params(
    buf: &mut [u8],
    pos0: usize,
    n_valid: usize,
    kv_chunks: usize,
    n_past: usize,
    is_causal: bool,
    swa_window: usize,
    layer_mode: i32,
    eps: f32,
    rope_cs: Option<&[f32]>,
) {
    let p_off = LFM2_PREFILL_SLOT_NORM_0 + 2048;
    write_le_i32(buf, p_off, pos0 as i32);
    write_le_i32(buf, p_off + 1, n_valid as i32);
    write_le_i32(buf, p_off + 2, kv_chunks as i32);
    write_le_i32(buf, p_off + 3, n_past as i32);
    write_le_i32(buf, p_off + 4, i32::from(is_causal));
    write_le_i32(buf, p_off + 5, swa_window as i32);
    write_le_i32(buf, p_off + 6, layer_mode);
    write_le_i32(buf, p_off + 7, eps.to_bits() as i32);
    write_le_i32(buf, p_off + 8, 0);
    write_le_i32(buf, p_off + 9, 0);
    write_le_i32(buf, p_off + 10, 0);
    if let Some(cs) = rope_cs {
        let sub0_f32 = LFM2_KV_CHUNK_T * LFM2_HEAD_DIM;
        let n0 = cs.len().min(sub0_f32);
        write_le_f32_slice(buf, LFM2_PREFILL_SLOT_ROPE_PKT_0, &cs[..n0]);
        if cs.len() > sub0_f32 {
            write_le_f32_slice(buf, LFM2_PREFILL_SLOT_ROPE_PKT_1, &cs[sub0_f32..]);
        }
    }
}

/// Write `g_params[8..11]` (`kv_start_chunk`, `has_fa_state`, `skip_cur_kv`) for paged `>8K` LFM2 prefill.
#[inline]
pub fn write_lfm2_prefill_ext_params(
    buf: &mut [u8],
    kv_start_chunk: usize,
    has_fa_state: bool,
    skip_cur_kv: bool,
) {
    let p_off = LFM2_PREFILL_SLOT_NORM_0 + 2048;
    write_le_i32(buf, p_off + 8, kv_start_chunk as i32);
    write_le_i32(buf, p_off + 9, i32::from(has_fa_state));
    write_le_i32(buf, p_off + 10, i32::from(skip_cur_kv));
}

/// Write all 11 LFM3 step+ext parameter words starting at float offset `p_off`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn write_lfm3_params_at(
    buf: &mut [u8],
    p_off: usize,
    pos0: usize,
    n_valid: usize,
    kv_chunks: usize,
    n_past: usize,
    is_causal: bool,
    swa_window: usize,
    layer_mode: i32,
    eps: f32,
    kv_start_chunk: usize,
    has_fa_state: bool,
    skip_cur_kv: bool,
) {
    write_le_i32(buf, p_off, pos0 as i32);
    write_le_i32(buf, p_off + 1, n_valid as i32);
    write_le_i32(buf, p_off + 2, kv_chunks as i32);
    write_le_i32(buf, p_off + 3, n_past as i32);
    write_le_i32(buf, p_off + 4, i32::from(is_causal));
    write_le_i32(buf, p_off + 5, swa_window as i32);
    write_le_i32(buf, p_off + 6, layer_mode);
    write_le_i32(buf, p_off + 7, eps.to_bits() as i32);
    write_le_i32(buf, p_off + 8, kv_start_chunk as i32);
    write_le_i32(buf, p_off + 9, i32::from(has_fa_state));
    write_le_i32(buf, p_off + 10, i32::from(skip_cur_kv));
}

/// Write `g_params[0..8]` and packed RoPE cos/sin into an LFM3 prefill or decode `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn write_lfm3_step_params(
    buf: &mut [u8],
    slot_norm_0: usize,
    slot_rope_pkt: Option<usize>,
    pos0: usize,
    n_valid: usize,
    kv_chunks: usize,
    n_past: usize,
    is_causal: bool,
    swa_window: usize,
    layer_mode: i32,
    eps: f32,
    rope_cs: Option<&[f32]>,
) {
    write_lfm3_params_at(
        buf,
        slot_norm_0 + 4096,
        pos0,
        n_valid,
        kv_chunks,
        n_past,
        is_causal,
        swa_window,
        layer_mode,
        eps,
        0,
        false,
        false,
    );
    if let (Some(rope_slot), Some(cs)) = (slot_rope_pkt, rope_cs) {
        write_le_f32_slice(buf, rope_slot, cs);
    }
}

/// Write `g_params[8..11]` (`kv_start_chunk`, `has_fa_state`, `skip_cur_kv`) for paged `>8K` LFM3 attention.
#[inline]
pub fn write_lfm3_ext_params(
    buf: &mut [u8],
    slot_norm_0: usize,
    kv_start_chunk: usize,
    has_fa_state: bool,
    skip_cur_kv: bool,
) {
    let p_off = slot_norm_0 + 4096;
    write_le_i32(buf, p_off + 8, kv_start_chunk as i32);
    write_le_i32(buf, p_off + 9, i32::from(has_fa_state));
    write_le_i32(buf, p_off + 10, i32::from(skip_cur_kv));
}

/// Write `g_params[0..8]` and packed `B=64` RoPE cos/sin into a unified LFM3 prefill `bo_y`.
#[allow(clippy::too_many_arguments)]
pub fn write_lfm3_prefill_step_params(
    buf: &mut [u8],
    pos0: usize,
    n_valid: usize,
    kv_chunks: usize,
    n_past: usize,
    is_causal: bool,
    swa_window: usize,
    layer_mode: i32,
    eps: f32,
    rope_cs: Option<&[f32]>,
) {
    write_lfm3_params_at(
        buf,
        LFM3_PREFILL_SLOT_NORM_0 + 2048,
        pos0,
        n_valid,
        kv_chunks,
        n_past,
        is_causal,
        swa_window,
        layer_mode,
        eps,
        0,
        false,
        false,
    );
    if let Some(cs) = rope_cs {
        const SLOTS: [usize; 4] = [
            LFM3_PREFILL_SLOT_ROPE_PKT_0A,
            LFM3_PREFILL_SLOT_ROPE_PKT_0B,
            LFM3_PREFILL_SLOT_ROPE_PKT_1A,
            LFM3_PREFILL_SLOT_ROPE_PKT_1B,
        ];
        const CHUNK_F32: usize = 16 * LFM3_HEAD_DIM;
        for (i, &slot) in SLOTS.iter().enumerate() {
            let start = i * CHUNK_F32;
            if start >= cs.len() {
                break;
            }
            let end = (start + CHUNK_F32).min(cs.len());
            write_le_f32_slice(buf, slot, &cs[start..end]);
        }
    }
}

/// Write `g_params[8..11]` (`kv_start_chunk`, `has_fa_state`, `skip_cur_kv`) into unified LFM3 prefill `bo_y`.
#[inline]
pub fn write_lfm3_prefill_ext_params(
    buf: &mut [u8],
    kv_start_chunk: usize,
    has_fa_state: bool,
    skip_cur_kv: bool,
) {
    let p_off = LFM3_PREFILL_SLOT_NORM_0 + 2048;
    write_le_i32(buf, p_off + 8, kv_start_chunk as i32);
    write_le_i32(buf, p_off + 9, i32::from(has_fa_state));
    write_le_i32(buf, p_off + 10, i32::from(skip_cur_kv));
}

/// Certified minimum mean logit cosine similarity across teacher-forced decode
/// positions (`docs/XDNA_PHASE0_RESULTS.md` Section 4: `mean_cos >= 0.99`).
pub const PARITY_MIN_MEAN_COS: f64 = 0.99;
/// Diagnostic per-step minimum logit cosine similarity floor (`min_cos >= 0.97`,
/// certified Llama-3.2-1B-Instruct Q4_0 baseline `min_cos = 0.979014`).
pub const PARITY_MIN_STEP_COS: f64 = 0.97;
/// Certified maximum CPU top-2 logit margin on any top-1 disagreement (`<= 2.0`).
pub const PARITY_MAX_CPU_MARGIN: f32 = 2.0;

/// Select `(bucket_idx, skip_kv)` for `pos` prior cached tokens across
/// strictly ascending `bucket_chunks` capacities in `1..=Q4LAYER_ATTN_CHUNKS`.
pub fn select_q4layer_kv_bucket(bucket_chunks: &[usize], pos: usize) -> (usize, i32) {
    assert!(!bucket_chunks.is_empty(), "bucket_chunks must be non-empty");
    assert!(
        pos < MAX_KV_CACHE_TOKENS,
        "pos ({pos}) must be < MAX_KV_CACHE_TOKENS ({MAX_KV_CACHE_TOKENS})"
    );
    assert!(
        bucket_chunks
            .iter()
            .all(|&c| (1..=Q4LAYER_ATTN_CHUNKS).contains(&c)),
        "bucket_chunks entries must be in 1..={Q4LAYER_ATTN_CHUNKS}"
    );
    assert!(
        bucket_chunks.windows(2).all(|w| w[0] < w[1]),
        "bucket_chunks must be strictly ascending"
    );
    let needed_chunks = pos.max(1).div_ceil(Q4LAYER_ATTN_CHUNK_T);
    let bucket_idx = bucket_chunks
        .iter()
        .position(|&c| c >= needed_chunks)
        .expect("pos exceeds maximum bucket capacity in bucket_chunks");
    let skip_kv = i32::try_from(Q4LAYER_ATTN_CHUNKS - bucket_chunks[bucket_idx])
        .expect("skip_kv fits in i32");
    (bucket_idx, skip_kv)
}

/// Pack 32-pair `(cos, sin)` tables into the two-half 128-float layout expected
/// by `q4layer.cc` at [`Q4LAYER_ROPE_CS`].
pub fn pack_q4layer_rope_cs(
    rope_cos: &[f32; Q4LAYER_HEAD_DIM / 2],
    rope_sin: &[f32; Q4LAYER_HEAD_DIM / 2],
) -> [f32; 128] {
    let mut rope_cs = [0.0f32; 128];
    for h in 0..2 {
        let base = h * 64;
        rope_cs[base..base + 16].copy_from_slice(&rope_cos[..16]);
        rope_cs[base + 16..base + 32].copy_from_slice(&rope_sin[..16]);
        rope_cs[base + 32..base + 48].copy_from_slice(&rope_cos[16..32]);
        rope_cs[base + 48..base + 64].copy_from_slice(&rope_sin[16..32]);
    }
    rope_cs
}

/// Write little-endian `f32` values into `buf` starting at `float_idx`.
#[inline]
pub fn write_le_f32_slice(buf: &mut [u8], float_idx: usize, vals: &[f32]) {
    let off = float_idx
        .checked_mul(4)
        .expect("write_le_f32_slice float_idx byte offset overflow");
    let byte_len = vals
        .len()
        .checked_mul(4)
        .expect("write_le_f32_slice slice byte length overflow");
    let end = off
        .checked_add(byte_len)
        .expect("write_le_f32_slice end byte offset overflow");
    let dst = &mut buf[off..end];
    for (chunk, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(vals) {
        *chunk = v.to_le_bytes();
    }
}

/// Read little-endian `f32` values from `buf` starting at `float_idx`.
#[inline]
pub fn read_le_f32_slice(buf: &[u8], float_idx: usize, dst: &mut [f32]) {
    let off = float_idx
        .checked_mul(4)
        .expect("read_le_f32_slice float_idx byte offset overflow");
    let byte_len = dst
        .len()
        .checked_mul(4)
        .expect("read_le_f32_slice slice byte length overflow");
    let end = off
        .checked_add(byte_len)
        .expect("read_le_f32_slice end byte offset overflow");
    let src = &buf[off..end];
    for (out, chunk) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
        *out = f32::from_le_bytes(*chunk);
    }
}

/// Write a little-endian `i32` into `buf` at `word_idx`.
#[inline]
pub fn write_le_i32(buf: &mut [u8], word_idx: usize, val: i32) {
    let off = word_idx
        .checked_mul(4)
        .expect("write_le_i32 word_idx byte offset overflow");
    let end = off
        .checked_add(4)
        .expect("write_le_i32 end byte offset overflow");
    buf[off..end].copy_from_slice(&val.to_le_bytes());
}

/// Zero `bo_y` and stage static attention and FFN RMSNorm weight vectors at
/// [`Q4LAYER_ATTN_NORM_W`] and `4 * Q4LAYER_ACT_SLOT + Q4LAYER_FFN_NORM_W`.
pub fn init_q4layer_bo_y_norms(buf: &mut [u8], attn_norm: &[f32], ffn_norm: &[f32]) {
    assert_eq!(buf.len(), Q4LAYER_ACT_BYTES);
    assert_eq!(attn_norm.len(), Q4LAYER_ATTN_NORM_W);
    assert_eq!(ffn_norm.len(), Q4LAYER_FFN_NORM_W);
    buf.fill(0);
    write_le_f32_slice(buf, Q4LAYER_ATTN_NORM_W, attn_norm);
    write_le_f32_slice(buf, 4 * Q4LAYER_ACT_SLOT + Q4LAYER_FFN_NORM_W, ffn_norm);
}

/// Stage per-token RoPE table, cached token count (`pos`), and `skip_kv` into
/// slot 0 of `bo_y`.
#[inline]
pub fn write_q4layer_step_params(buf: &mut [u8], rope_cs: &[f32; 128], pos: usize, skip_kv: i32) {
    assert_eq!(buf.len(), Q4LAYER_ACT_BYTES);
    assert!(pos < MAX_KV_CACHE_TOKENS);
    assert!(
        (0..Q4LAYER_ATTN_CHUNKS as i32).contains(&skip_kv),
        "skip_kv ({skip_kv}) out of range 0..{Q4LAYER_ATTN_CHUNKS}"
    );
    let active_chunks = Q4LAYER_ATTN_CHUNKS - skip_kv as usize;
    assert!(
        active_chunks * Q4LAYER_ATTN_CHUNK_T >= pos.max(1),
        "skip_kv ({skip_kv}) leaves fewer KV tokens than pos ({pos})"
    );
    write_le_f32_slice(buf, Q4LAYER_ROPE_CS, rope_cs);
    write_le_i32(buf, Q4LAYER_ATTN_N_OFF, pos as i32);
    write_le_i32(buf, Q4LAYER_ATTN_SKIP_KV_OFF, skip_kv);
}

/// Convert one KV head's `(k_head, v_head)` at `pos` to bf16 and write into
/// `q4layer`'s chunked KV region in `bo_y`, returning the byte offset of the
/// written K row (V sits `Q4LAYER_SILU_HALF * 4` bytes past it).
#[inline]
pub fn write_q4layer_kv_bf16(
    buf: &mut [u8],
    kvh: usize,
    pos: usize,
    k_head: &[f32],
    v_head: &[f32],
) -> usize {
    assert_eq!(buf.len(), Q4LAYER_KV_BYTES);
    assert!(kvh < Q4LAYER_N_KV_HEADS);
    assert!(pos < MAX_KV_CACHE_TOKENS);
    assert_eq!(k_head.len(), Q4LAYER_HEAD_DIM);
    assert_eq!(v_head.len(), Q4LAYER_HEAD_DIM);
    let c = pos / Q4LAYER_ATTN_CHUNK_T;
    let rem = pos % Q4LAYER_ATTN_CHUNK_T;
    let chunk_byte_off = (kvh * Q4LAYER_ATTN_CHUNKS + c) * Q4LAYER_TILE_F32 * 4;
    let k_byte_off = chunk_byte_off + rem * Q4LAYER_HEAD_DIM * 2;
    let v_byte_off = k_byte_off + Q4LAYER_SILU_HALF * 4;
    for d in 0..Q4LAYER_HEAD_DIM {
        let kb = super::repack::f32_to_bf16(k_head[d]).to_le_bytes();
        let vb = super::repack::f32_to_bf16(v_head[d]).to_le_bytes();
        buf[k_byte_off + d * 2..k_byte_off + d * 2 + 2].copy_from_slice(&kb);
        buf[v_byte_off + d * 2..v_byte_off + d * 2 + 2].copy_from_slice(&vb);
    }
    k_byte_off
}

/// Parse instruction bytes as little-endian 32-bit words, rejecting empty or
/// non-4-byte-aligned streams instead of silently dropping trailing bytes.
pub fn try_inst_words(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "XDNA instruction stream must be non-empty and a multiple of 4 bytes, got {} bytes",
            bytes.len()
        ));
    }
    let (chunks, rem) = bytes.as_chunks::<4>();
    debug_assert!(rem.is_empty());
    Ok(chunks.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Instruction bytes as little-endian words, as the shim uploads them.
///
/// Panics if `bytes` is empty or not a whole number of 32-bit words; use
/// [`try_inst_words`] for untrusted inputs.
pub fn inst_words(bytes: &[u8]) -> Vec<u32> {
    try_inst_words(bytes).expect("XDNA instruction stream must be non-empty and 4-byte aligned")
}

/// Return the top `k` `(token_idx, logit)` pairs sorted descending by logit,
/// canonicalizing `-0.0` to `+0.0` and breaking equal-logit ties by ascending
/// token index so greedy top-1 matches `cera::sampler::argmax` across finite
/// logits and all-`NaN` inputs. Non-`NaN` values rank ahead of `NaN`, and `NaN`
/// entries break ties by ascending token index.
pub fn top_k_logits(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &(usize, f32), b: &(usize, f32)| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)),
    };
    let canonical = |v: f32| if v == 0.0 { 0.0 } else { v };
    if k <= 64 && k < logits.len() {
        let mut best: Vec<(usize, f32)> = Vec::with_capacity(k);
        for (idx, &raw) in logits.iter().enumerate() {
            let candidate = (idx, canonical(raw));
            if best.len() == k && cmp(&candidate, &best[k - 1]) != std::cmp::Ordering::Less {
                continue;
            }
            let pos =
                best.partition_point(|entry| cmp(entry, &candidate) == std::cmp::Ordering::Less);
            if best.len() == k {
                best.pop();
            }
            best.insert(pos, candidate);
        }
        return best;
    }
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().map(canonical).enumerate().collect();
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, cmp);
        pairs.truncate(k);
        pairs.shrink_to_fit();
    }
    pairs.sort_by(cmp);
    pairs
}

/// Cosine similarity in `f64` between two non-empty, finite, non-zero logit slices.
pub fn logit_cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 || !na.is_finite() || !nb.is_finite() {
        return None;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    cos.is_finite().then(|| cos.clamp(-1.0, 1.0))
}

/// Verify teacher-forced parity summary metrics against Section 4 of
/// `docs/XDNA_PHASE0_RESULTS.md`.
pub fn check_parity_thresholds(
    target_k: usize,
    mean_overlap: f64,
    mean_cos: f64,
    min_cos: f64,
    max_cpu_margin: f32,
) -> Result<(), String> {
    let min_expected_overlap = (target_k as f64 - 1.0).max(1.0);
    if !mean_overlap.is_finite()
        || !mean_cos.is_finite()
        || !min_cos.is_finite()
        || !max_cpu_margin.is_finite()
        || mean_overlap < min_expected_overlap
        || mean_overlap > target_k as f64
        || mean_cos < PARITY_MIN_MEAN_COS
        || mean_cos > 1.0
        || min_cos < PARITY_MIN_STEP_COS
        || min_cos > 1.0
        || max_cpu_margin < 0.0
        || max_cpu_margin > PARITY_MAX_CPU_MARGIN
    {
        return Err(format!(
            "parity below threshold (mean_overlap={mean_overlap:.2} [min {min_expected_overlap:.1}], mean_cos={mean_cos:.6} [min {PARITY_MIN_MEAN_COS:.2}], min_cos={min_cos:.6} [min {PARITY_MIN_STEP_COS:.2}], max_cpu_margin={max_cpu_margin:.4} [max {PARITY_MAX_CPU_MARGIN:.1}])"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_inst_words_rejects_empty_and_unaligned_and_parses_embedded() {
        assert!(try_inst_words(&[]).is_err());
        assert!(try_inst_words(&[1, 2, 3]).is_err());
        assert!(try_inst_words(&[1, 2, 3, 4, 5]).is_err());
        assert_eq!(
            try_inst_words(&[0x01, 0x02, 0x03, 0x04]).unwrap(),
            vec![0x0403_0201]
        );
        for &(_, inst_bytes) in &Q4LAYER_KV_BUCKETS {
            assert!(!inst_words(inst_bytes).is_empty());
        }
        assert!(!inst_words(Q4HEAD_INSTS).is_empty());
    }

    #[test]
    fn q4layer_bo_y_framing_places_norms_step_params_and_chunked_kv() {
        let mut bo_y = vec![0xFFu8; Q4LAYER_ACT_BYTES];
        let attn_norm = vec![1.25f32; Q4LAYER_ATTN_NORM_W];
        let ffn_norm = vec![2.5f32; Q4LAYER_FFN_NORM_W];
        init_q4layer_bo_y_norms(&mut bo_y, &attn_norm, &ffn_norm);

        let mut got_attn = vec![0.0f32; Q4LAYER_ATTN_NORM_W];
        let mut got_ffn = vec![0.0f32; Q4LAYER_FFN_NORM_W];
        read_le_f32_slice(&bo_y, Q4LAYER_ATTN_NORM_W, &mut got_attn);
        read_le_f32_slice(
            &bo_y,
            4 * Q4LAYER_ACT_SLOT + Q4LAYER_FFN_NORM_W,
            &mut got_ffn,
        );
        assert_eq!(got_attn, attn_norm);
        assert_eq!(got_ffn, ffn_norm);

        let mut cos = [0.0f32; Q4LAYER_HEAD_DIM / 2];
        let mut sin = [0.0f32; Q4LAYER_HEAD_DIM / 2];
        for i in 0..Q4LAYER_HEAD_DIM / 2 {
            cos[i] = i as f32;
            sin[i] = -(i as f32);
        }
        let rope_cs = pack_q4layer_rope_cs(&cos, &sin);
        assert_eq!(&rope_cs[0..16], &cos[0..16]);
        assert_eq!(&rope_cs[16..32], &sin[0..16]);
        assert_eq!(&rope_cs[32..48], &cos[16..32]);
        assert_eq!(&rope_cs[48..64], &sin[16..32]);
        assert_eq!(&rope_cs[64..128], &rope_cs[0..64]);

        let buckets: Vec<usize> = Q4LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        let (bucket_idx, skip_kv) = select_q4layer_kv_bucket(&buckets, 540);
        assert_eq!((bucket_idx, skip_kv), (4, 52));
        write_q4layer_step_params(&mut bo_y, &rope_cs, 540, skip_kv);
        let n_off = Q4LAYER_ATTN_N_OFF * 4;
        let skip_off = Q4LAYER_ATTN_SKIP_KV_OFF * 4;
        assert_eq!(
            i32::from_le_bytes(bo_y[n_off..n_off + 4].try_into().unwrap()),
            540
        );
        assert_eq!(
            i32::from_le_bytes(bo_y[skip_off..skip_off + 4].try_into().unwrap()),
            52
        );

        let mut bo_kv = vec![0xFFu8; Q4LAYER_KV_BYTES];
        let k_head = [1.0f32; Q4LAYER_HEAD_DIM];
        let v_head = [-2.0f32; Q4LAYER_HEAD_DIM];
        let k_off = write_q4layer_kv_bf16(&mut bo_kv, 7, 4095, &k_head, &v_head);
        let v_off = k_off + Q4LAYER_SILU_HALF * 4;
        assert!(v_off + Q4LAYER_HEAD_DIM * 2 <= Q4LAYER_KV_BYTES);
        assert_eq!(u16::from_le_bytes([bo_kv[k_off], bo_kv[k_off + 1]]), 0x3F80);
        assert_eq!(u16::from_le_bytes([bo_kv[v_off], bo_kv[v_off + 1]]), 0xC000);
    }

    #[test]
    #[should_panic(expected = "byte offset overflow")]
    fn write_le_f32_slice_rejects_wrapping_word_index() {
        let mut buf = [0u8; 16];
        write_le_f32_slice(&mut buf, (usize::MAX / 4) + 1, &[1.0]);
    }

    #[test]
    #[should_panic(expected = "byte offset overflow")]
    fn read_le_f32_slice_rejects_wrapping_word_index() {
        let buf = [0u8; 16];
        let mut dst = [0.0f32; 1];
        read_le_f32_slice(&buf, (usize::MAX / 4) + 1, &mut dst);
    }

    #[test]
    #[should_panic(expected = "byte offset overflow")]
    fn write_le_i32_rejects_wrapping_word_index() {
        let mut buf = [0u8; 16];
        write_le_i32(&mut buf, (usize::MAX / 4) + 1, 7);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum bucket capacity")]
    fn select_q4layer_kv_bucket_panics_when_pos_exceeds_bucket_table() {
        let _ = select_q4layer_kv_bucket(&[1, 2], 200);
    }

    #[test]
    #[should_panic(expected = "leaves fewer KV tokens")]
    fn write_q4layer_step_params_rejects_undersized_active_chunks() {
        let mut bo_y = vec![0u8; Q4LAYER_ACT_BYTES];
        let rope_cs = [0.0f32; 128];
        write_q4layer_step_params(&mut bo_y, &rope_cs, 100, (Q4LAYER_ATTN_CHUNKS - 1) as i32);
    }

    #[test]
    fn top_k_logits_ranks_non_nan_above_nan_and_bounds_small_k_alloc() {
        let nan_pos = f32::NAN;
        let nan_neg = -f32::NAN;
        let nan_payload = f32::from_bits(0x7fc0_1234);

        let mixed = [nan_pos, -1.0f32, nan_neg, 2.5f32, 2.5f32, nan_payload];
        let ranked = top_k_logits(&mixed, 6);
        assert_eq!(ranked[0], (3, 2.5));
        assert_eq!(ranked[1], (4, 2.5));
        assert_eq!(ranked[2], (1, -1.0));
        assert_eq!(ranked[3].0, 0);
        assert!(ranked[3].1.is_nan());
        assert_eq!(ranked[4].0, 2);
        assert!(ranked[4].1.is_nan());
        assert_eq!(ranked[5].0, 5);
        assert!(ranked[5].1.is_nan());
        assert_eq!(
            top_k_logits(&mixed, 1)[0].0,
            crate::sampler::argmax(&mixed) as usize
        );

        let all_nan = [nan_neg, nan_payload, nan_pos];
        let all_nan_top = top_k_logits(&all_nan, 2);
        assert_eq!(all_nan_top.len(), 2);
        assert_eq!(all_nan_top[0].0, 0);
        assert_eq!(all_nan_top[1].0, 1);
        assert_eq!(all_nan_top[0].0, crate::sampler::argmax(&all_nan) as usize);

        let large = vec![0.5f32; 256];
        let small_top = top_k_logits(&large, 4);
        assert_eq!(small_top.len(), 4);
        assert_eq!(small_top.capacity(), 4);

        let medium_top = top_k_logits(&large, 80);
        assert_eq!(medium_top.len(), 80);
        assert_eq!(medium_top.capacity(), 80);

        assert!(top_k_logits(&mixed, 0).is_empty());
        assert!(top_k_logits(&[], 4).is_empty());
    }

    #[test]
    fn parity_helpers_canonicalize_signed_zero_ties_and_match_certified_baseline() {
        let ranked = top_k_logits(&[-0.0, 0.0, 1.5, 1.5, -2.0], 4);
        assert_eq!(ranked, vec![(2, 1.5), (3, 1.5), (0, 0.0), (1, 0.0)]);
        assert_eq!(ranked[2].1.to_bits(), 0.0f32.to_bits());

        assert_eq!(logit_cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None);
        assert_eq!(logit_cosine_similarity(&[f32::NAN, 1.0], &[1.0, 1.0]), None);
        let cos = logit_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).expect("unit cos");
        assert!((cos - 1.0).abs() < 1e-12);
        let cos_over = logit_cosine_similarity(&[1e30, 1e30], &[1e30, 1e30]).expect("clamped");
        assert!((-1.0..=1.0).contains(&cos_over));

        // Certified Llama-3.2-1B-Instruct-Q4_0 512-step hardware metrics
        // (`docs/XDNA_PHASE0_RESULTS.md` Section 4).
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.979_014, 1.0658).is_ok());
        assert!(check_parity_thresholds(10, 8.95, 0.995_958, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 10.5, 0.995_958, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.989_000, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 1.01, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.969_000, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 1.01, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.979_014, -0.1).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.979_014, 2.05).is_err());

        for w in Q4LAYER_KV_BUCKETS.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "KV bucket chunk counts must strictly increase"
            );
        }
        assert_eq!(Q4LAYER_KV_BUCKETS.last().unwrap().0, Q4LAYER_ATTN_CHUNKS);
        for &(_, inst_bytes) in &Q4PREFILL_LAYER_KV_BUCKETS {
            assert!(!inst_words(inst_bytes).is_empty());
        }
        for w in Q4PREFILL_LAYER_KV_BUCKETS.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        assert_eq!(Q4PREFILL_LAYER_KV_BUCKETS.first().unwrap().0, 0);
        assert_eq!(
            Q4PREFILL_LAYER_KV_BUCKETS.last().unwrap().0 * Q4LAYER_ATTN_CHUNK_T,
            Q4PREFILL_MAX_POS0
        );
    }

    #[test]
    fn q4prefill_bucket_selection_and_bo_y_framing_match_kernel_layout() {
        let buckets: Vec<usize> = Q4PREFILL_LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        assert_eq!(select_q4prefill_kv_bucket(&buckets, 0), (0, 0));
        assert_eq!(select_q4prefill_kv_bucket(&buckets, 1), (1, 1));
        assert_eq!(select_q4prefill_kv_bucket(&buckets, 64), (1, 1));
        assert_eq!(select_q4prefill_kv_bucket(&buckets, 65), (2, 2));
        assert_eq!(
            select_q4prefill_kv_bucket(&buckets, Q4PREFILL_MAX_POS0),
            (buckets.len() - 1, 63)
        );

        let mut bo_y = vec![0xAAu8; Q4PREFILL_ACT_BYTES];
        let attn_norm = vec![1.5f32; 2048];
        let ffn_norm = vec![-0.75f32; 2048];
        init_q4prefill_bo_y_norms(&mut bo_y, &attn_norm, &ffn_norm);
        let mut got_attn = vec![0.0f32; 2048];
        let mut got_ffn = vec![0.0f32; 2048];
        read_le_f32_slice(&bo_y, Q4PREFILL_SLOT_META, &mut got_attn);
        read_le_f32_slice(&bo_y, Q4PREFILL_SLOT_META + 2048, &mut got_ffn);
        assert_eq!(got_attn, attn_norm);
        assert_eq!(got_ffn, ffn_norm);

        let mut rope_cs = [0.0f32; Q4PREFILL_B * Q4LAYER_HEAD_DIM];
        for (i, dst) in rope_cs.iter_mut().enumerate() {
            *dst = (i as f32) * 0.25;
        }
        write_q4prefill_step_params(&mut bo_y, 128, 17, 2, 128, true, 0, 1e-5, &rope_cs);
        let p_off = (Q4PREFILL_SLOT_META + 4096) * 4;
        let read_i32 = |idx: usize| {
            let off = p_off + idx * 4;
            i32::from_le_bytes(bo_y[off..off + 4].try_into().unwrap())
        };
        assert_eq!(read_i32(0), 128);
        assert_eq!(read_i32(1), 17);
        assert_eq!(read_i32(2), 2);
        assert_eq!(read_i32(3), 128);
        assert_eq!(read_i32(4), 1);
        assert_eq!(read_i32(5), 0);
        assert_eq!(read_i32(6), 0);
        assert_eq!(read_i32(7) as u32, 1e-5f32.to_bits());
        let mut got_rope = vec![0.0f32; rope_cs.len()];
        read_le_f32_slice(&bo_y, Q4PREFILL_SLOT_META + 4608, &mut got_rope);
        assert_eq!(got_rope, rope_cs);
    }

    #[test]
    #[should_panic(expected = "exceeds Q4PREFILL_MAX_POS0")]
    fn select_q4prefill_kv_bucket_rejects_past_max_pos0() {
        let _ = select_q4prefill_kv_bucket(&[0, 63], Q4PREFILL_MAX_POS0 + 1);
    }

    #[test]
    #[should_panic(expected = "n_valid")]
    fn write_q4prefill_step_params_rejects_invalid_n_valid() {
        let mut bo_y = vec![0u8; Q4PREFILL_ACT_BYTES];
        let rope_cs = [0.0f32; Q4PREFILL_B * Q4LAYER_HEAD_DIM];
        write_q4prefill_step_params(&mut bo_y, 0, 33, 0, 0, true, 0, 1e-5, &rope_cs);
    }

    #[test]
    fn lfm3_buckets_and_bo_y_helpers_match_kernel_layout() {
        for table in [
            &Q4PREFILL_LFM3_ATTN_BUCKETS[..],
            &Q4PREFILL_LFM3_BIDIR_B_BUCKETS[..],
            &Q4LAYER_LFM3_KV_BUCKETS[..],
        ] {
            for &(_, inst_bytes) in table {
                assert!(!inst_words(inst_bytes).is_empty());
            }
            for w in table.windows(2) {
                assert!(w[0].0 < w[1].0);
            }
        }
        assert!(!inst_words(Q4PREFILL_LFM3_CONV_INSTS).is_empty());
        assert!(!inst_words(Q4PREFILL_LFM3_BIDIR_A_INSTS).is_empty());
        assert!(!inst_words(Q4LAYER_LFM3_CONV_INSTS).is_empty());

        let attn_buckets: Vec<usize> = Q4PREFILL_LFM3_ATTN_BUCKETS
            .iter()
            .map(|&(c, _)| c)
            .collect();
        assert_eq!(select_lfm3_kv_bucket(&attn_buckets, 0), (0, 0));
        assert_eq!(select_lfm3_kv_bucket(&attn_buckets, 3), (3, 4));
        assert_eq!(
            select_lfm3_kv_bucket(&attn_buckets, 255),
            (attn_buckets.len() - 1, 255)
        );

        let bidir_buckets: Vec<usize> = Q4PREFILL_LFM3_BIDIR_B_BUCKETS
            .iter()
            .map(|&(c, _)| c)
            .collect();
        assert_eq!(select_lfm3_kv_bucket(&bidir_buckets, 1), (0, 1));
        assert_eq!(
            select_lfm3_kv_bucket(&bidir_buckets, 256),
            (bidir_buckets.len() - 1, 256)
        );

        let mut bo_y = vec![0u8; LFM3_PREFILL_ACT_BYTES];
        let attn_norm = vec![1.25f32; LFM3_H];
        let post_attn_norm = vec![-0.5f32; LFM3_H];
        let ffn_norm = vec![0.75f32; LFM3_H];
        let post_ffn_norm = vec![-1.5f32; LFM3_H];
        let q_norm = vec![2.0f32; LFM3_HEAD_DIM];
        let k_norm = vec![-2.0f32; LFM3_HEAD_DIM];
        init_lfm3_prefill_bo_y_norms(
            &mut bo_y,
            &attn_norm,
            &post_attn_norm,
            &ffn_norm,
            &post_ffn_norm,
            Some(&q_norm),
            Some(&k_norm),
        );
        let mut got = vec![0.0f32; LFM3_H];
        read_le_f32_slice(&bo_y, LFM3_PREFILL_SLOT_NORM_0, &mut got);
        assert_eq!(got, attn_norm);
        read_le_f32_slice(&bo_y, LFM3_PREFILL_SLOT_NORM_1, &mut got);
        assert_eq!(got, post_attn_norm);
        read_le_f32_slice(&bo_y, LFM3_PREFILL_SLOT_NORM_2, &mut got);
        assert_eq!(got, ffn_norm);
        read_le_f32_slice(&bo_y, LFM3_PREFILL_SLOT_NORM_3, &mut got);
        assert_eq!(got, post_ffn_norm);
        let mut got_qk = vec![0.0f32; LFM3_HEAD_DIM];
        for slot in [LFM3_PREFILL_SLOT_ROPE_PKT_0A, LFM3_PREFILL_SLOT_ROPE_PKT_1A] {
            read_le_f32_slice(&bo_y, slot + 2048, &mut got_qk);
            assert_eq!(got_qk, q_norm);
            read_le_f32_slice(&bo_y, slot + 2176, &mut got_qk);
            assert_eq!(got_qk, k_norm);
        }

        let rope_cs: Vec<f32> = (0..(LFM3_PREFILL_B * LFM3_HEAD_DIM))
            .map(|i| (i as f32) * 0.125)
            .collect();
        write_lfm3_prefill_step_params(
            &mut bo_y,
            64,
            45,
            4,
            96,
            false,
            512,
            3,
            1e-5,
            Some(&rope_cs),
        );
        let p_off = (LFM3_PREFILL_SLOT_NORM_0 + 2048) * 4;
        let read_i32 = |idx: usize| {
            let off = p_off + idx * 4;
            i32::from_le_bytes(bo_y[off..off + 4].try_into().unwrap())
        };
        assert_eq!(read_i32(0), 64);
        assert_eq!(read_i32(1), 45);
        assert_eq!(read_i32(2), 4);
        assert_eq!(read_i32(3), 96);
        assert_eq!(read_i32(4), 0);
        assert_eq!(read_i32(5), 512);
        assert_eq!(read_i32(6), 3);
        assert_eq!(read_i32(7) as u32, 1e-5f32.to_bits());
        let mut got_rope = Vec::with_capacity(rope_cs.len());
        for slot in [
            LFM3_PREFILL_SLOT_ROPE_PKT_0A,
            LFM3_PREFILL_SLOT_ROPE_PKT_0B,
            LFM3_PREFILL_SLOT_ROPE_PKT_1A,
            LFM3_PREFILL_SLOT_ROPE_PKT_1B,
        ] {
            let mut pkt = vec![0.0f32; 16 * LFM3_HEAD_DIM];
            read_le_f32_slice(&bo_y, slot, &mut pkt);
            got_rope.extend_from_slice(&pkt);
        }
        assert_eq!(got_rope, rope_cs);
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and XRT driver"]
    fn hw_verify_q4prefill_lfm3_conv() {
        run_hw_verify_lfm3(true, false);
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and XRT driver"]
    fn hw_verify_q4prefill_lfm3_attn() {
        run_hw_verify_lfm3(false, true);
    }

    fn run_hw_verify_lfm3(run_conv: bool, run_attn: bool) {
        use crate::backend::xdna::device::XdnaDevice;
        use crate::backend::xdna::repack::{f32_to_bf16, pack_f32_stage_k1024};
        use half::bf16;

        const B: usize = LFM3_PREFILL_B;
        const H: usize = 2048;
        const FFN_SRC: usize = 7680;
        const FFN_PAD: usize = 8192;
        const Q_DIM: usize = 2048;
        const KV_DIM: usize = 512;
        const N_H: usize = 16;
        const N_KV: usize = 4;
        const HD: usize = 128;
        const W_ELEMS: usize = 4608;
        const W_TOTAL: usize = 1994 * W_ELEMS;
        const ACT_TOTAL: usize = LFM3_PREFILL_ACT_BYTES / 4;
        const YN_TOTAL: usize = B * H;
        const KV_TOTAL: usize = 256 * 4 * W_ELEMS;
        const KVO_TOTAL: usize = 8 * W_ELEMS;
        const SLOT_Q_BF16: usize = LFM3_PREFILL_SLOT_Q_BF16;
        const SLOT_ATTN_GATE: usize = LFM3_PREFILL_SLOT_ATTN_GATE;
        const SLOT_NORM_0: usize = LFM3_PREFILL_SLOT_NORM_0;

        fn det_val(seed: usize, i: usize, scale: f32) -> f32 {
            let mut z = (seed as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (((z & 0xFFFF) as f32) / 32768.0 - 1.0) * scale
        }

        fn make_f32_weights(
            seed: usize,
            rows: usize,
            cols: usize,
            pad_cols: usize,
            scale: f32,
        ) -> Vec<f32> {
            let mut w = vec![0.0f32; rows * pad_cols];
            for r in 0..rows {
                for c in 0..cols {
                    w[r * pad_cols + c] = det_val(seed + r, c, scale);
                }
            }
            w
        }

        fn pack_w_f32_to_bo(
            bo: &mut [u8],
            tile_off: usize,
            w: &[f32],
            rows: usize,
            cols: usize,
        ) -> Vec<f32> {
            let need = crate::backend::xdna::repack::stage_bytes(rows, cols).unwrap();
            let dst = &mut bo[tile_off * W_ELEMS * 4..tile_off * W_ELEMS * 4 + need];
            pack_f32_stage_k1024(dst, w, rows, cols).unwrap();
            // Decode back the exact Q4_0 + bf16-scale weights that live in bo_w so CPU f64 ref
            // isolates kernel/NPU accuracy rather than synthetic f32->Q4 quantization noise.
            let mut decoded = vec![0.0f32; rows * cols];
            let chunk_k = 1024usize.min(cols);
            let chunks = cols / chunk_k;
            let nb_chunk = chunk_k / 32;
            let tile_bytes = nb_chunk * 288;
            for tile in 0..rows / 16 {
                for ch in 0..chunks {
                    let tp = (tile * chunks + ch) * tile_bytes;
                    for lb in 0..nb_chunk {
                        let b = ch * nb_chunk + lb;
                        for n in 0..16 {
                            let sp = tp + nb_chunk * 256 + lb * 32 + n * 2;
                            let d = bf16::from_le_bytes([dst[sp], dst[sp + 1]]).to_f32();
                            for h in 0..2 {
                                for kk in 0..16 {
                                    let e = kk * 16 + n;
                                    let byte = dst[tp + lb * 256 + h * 128 + e / 2];
                                    let nib = if e % 2 == 0 { byte & 0xF } else { byte >> 4 } as i8;
                                    let q = if nib >= 8 { nib - 16 } else { nib };
                                    decoded[(tile * 16 + n) * cols + b * 32 + h * 16 + kk] =
                                        (q as f32) * d;
                                }
                            }
                        }
                    }
                }
            }
            decoded
        }

        fn rmsnorm_cpu(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
            let n = x.len();
            let ss: f64 = x.iter().map(|&v| v * v).sum();
            let inv = 1.0 / (ss / (n as f64) + eps).sqrt();
            x.iter()
                .zip(w.iter())
                .map(|(&v, &wi)| v * inv * (wi as f64))
                .collect()
        }

        fn matmul_cpu(a: &[f64], w: &[f32], rows: usize, cols: usize) -> Vec<f64> {
            (0..rows)
                .map(|r| {
                    let wr = &w[r * cols..(r + 1) * cols];
                    a.iter()
                        .zip(wr.iter())
                        .map(|(&ai, &wi)| ai * (wi as f64))
                        .sum()
                })
                .collect()
        }

        fn neox_rope_cpu(x: &mut [f64], cs: &[f32]) {
            for i in 0..64 {
                let c = cs[i] as f64;
                let s = cs[64 + i] as f64;
                let x0 = x[i];
                let x1 = x[64 + i];
                x[i] = x0 * c - x1 * s;
                x[64 + i] = x0 * s + x1 * c;
            }
        }

        fn silu_cpu(x: f64) -> f64 {
            x / (1.0 + (-x).exp())
        }

        fn rel_rms(got: &[f32], ref_f64: &[f64], label: &str) {
            assert_eq!(got.len(), ref_f64.len());
            let mut err2 = 0.0f64;
            let mut ref2 = 0.0f64;
            let mut nans = 0usize;
            for (&g, &r) in got.iter().zip(ref_f64.iter()) {
                if !g.is_finite() {
                    nans += 1;
                } else {
                    let d = (g as f64) - r;
                    err2 += d * d;
                    ref2 += r * r;
                }
            }
            let rel = (err2 / ref2.max(1e-12)).sqrt();
            println!("{label}: nans={nans} rel_rms={rel:.5}");
            assert_eq!(nans, 0, "{label} produced NaN/Inf");
            assert!(rel < 0.03, "{label} rel_rms too high: {rel:.5}");
        }

        let dev = XdnaDevice::open(0).expect("open NPU");

        // --- Test 1: q4prefill_lfm3_conv (Mode 0: ShortConv l_cache=7) ---
        if run_conv {
            let k_conv = dev
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_CONV_INSTS),
                )
                .expect("load conv");
            let mut bo_w = k_conv.alloc_bo(W_TOTAL * 4, 3).unwrap();
            let mut bo_y = k_conv.alloc_bo(ACT_TOTAL * 4, 4).unwrap();
            let mut bo_yn = k_conv.alloc_bo(YN_TOTAL * 4, 5).unwrap();
            let mut bo_kv = k_conv.alloc_bo(KV_TOTAL * 4, 6).unwrap();
            let mut bo_kvo = k_conv.alloc_bo(KVO_TOTAL * 4, 7).unwrap();

            let w_b = make_f32_weights(10, H, H, H, 0.04);
            let w_c = make_f32_weights(2100, H, H, H, 0.04);
            let w_x = make_f32_weights(4200, H, H, H, 0.04);
            let w_o = make_f32_weights(6300, H, H, H, 0.04);
            let w_fg = make_f32_weights(9000, FFN_SRC, H, H, 0.03);
            let w_fu = make_f32_weights(17000, FFN_SRC, H, H, 0.03);
            let w_fd = make_f32_weights(25000, H, FFN_SRC, FFN_PAD, 0.02);
            let conv_wt = make_f32_weights(8500, 7, H, H, 0.15);
            let conv_state = make_f32_weights(8600, 6, H, H, 0.25);

            let (w_b, w_c, w_x, w_o, w_fg, w_fu, w_fd) = {
                let wb = bo_w.as_mut_slice();
                wb.fill(0);
                let w_b = pack_w_f32_to_bo(wb, 0, &w_b, H, H);
                let w_c = pack_w_f32_to_bo(wb, 128, &w_c, H, H);
                let w_x = pack_w_f32_to_bo(wb, 256, &w_x, H, H);
                let w_o = pack_w_f32_to_bo(wb, 384, &w_o, H, H);
                let w_fg = pack_w_f32_to_bo(wb, 520, &w_fg, FFN_SRC, H);
                let w_fu = pack_w_f32_to_bo(wb, 1000, &w_fu, FFN_SRC, H);
                let w_fd = pack_w_f32_to_bo(wb, 1480, &w_fd, H, FFN_PAD);
                for col in 0..8 {
                    for k in 0..7 {
                        write_le_f32_slice(
                            wb,
                            (512 + col) * W_ELEMS + k * 256,
                            &conv_wt[k * H + col * 256..k * H + (col + 1) * 256],
                        );
                    }
                }
                (w_b, w_c, w_x, w_o, w_fg, w_fu, w_fd)
            };
            {
                let kb = bo_kv.as_mut_slice();
                kb.fill(0);
                for col in 0..8 {
                    for k in 0..6 {
                        write_le_f32_slice(
                            kb,
                            col * W_ELEMS + k * 256,
                            &conv_state[k * H + col * 256..k * H + (col + 1) * 256],
                        );
                    }
                }
            }

            let hidden_in: Vec<f32> = (0..B * H).map(|i| det_val(1, i, 0.35)).collect();
            let n0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(2, i, 0.2)).collect();
            let p0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(3, i, 0.2)).collect();
            let n1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(4, i, 0.2)).collect();
            let p1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(5, i, 0.2)).collect();
            let eps = 1e-5f32;
            let n_valid = 45usize;
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                write_le_f32_slice(yb, 0, &hidden_in);
                init_lfm3_prefill_bo_y_norms(yb, &n0, &p0, &n1, &p1, None, None);
                write_lfm3_prefill_step_params(yb, 0, n_valid, 0, 0, true, 0, 0, eps, None);
            }

            bo_w.sync_to_device().unwrap();
            bo_y.sync_to_device().unwrap();
            bo_kv.sync_to_device().unwrap();
            let run = k_conv
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            run.start().unwrap();
            run.wait(15_000).unwrap();
            bo_yn.sync_from_device().unwrap();
            bo_kvo.sync_from_device().unwrap();

            let mut out_yn = vec![0.0f32; B * H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut out_yn);
            let mut out_kvo = vec![0.0f32; KVO_TOTAL];
            read_le_f32_slice(bo_kvo.as_slice(), 0, &mut out_kvo);

            // CPU f64 reference
            let mut bx_seq = vec![vec![0.0f64; H]; B];
            let mut c_seq = vec![vec![0.0f64; H]; B];
            let mut h_in64 = vec![vec![0.0f64; H]; B];
            for t in 0..B {
                h_in64[t] = hidden_in[t * H..(t + 1) * H]
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                let xn = rmsnorm_cpu(&h_in64[t], &n0, eps as f64);
                let bv = matmul_cpu(&xn, &w_b, H, H);
                c_seq[t] = matmul_cpu(&xn, &w_c, H, H);
                let xv = matmul_cpu(&xn, &w_x, H, H);
                for d in 0..H {
                    bx_seq[t][d] = bv[d] * xv[d];
                }
            }
            let mut cpu_out = vec![0.0f64; B * H];
            for t in 0..B {
                let mut conv_y = vec![0.0f64; H];
                for d in 0..H {
                    let mut acc = 0.0f64;
                    for k in 0..7 {
                        let sidx = t as isize + (k as isize) - 6;
                        let s = if sidx < 0 {
                            conv_state[(6 + sidx) as usize * H + d] as f64
                        } else {
                            bx_seq[sidx as usize][d]
                        };
                        acc += s * (conv_wt[k * H + d] as f64);
                    }
                    conv_y[d] = acc * c_seq[t][d];
                }
                let oproj = matmul_cpu(&conv_y, &w_o, H, H);
                let post_o = rmsnorm_cpu(&oproj, &p0, eps as f64);
                let h1: Vec<f64> = h_in64[t]
                    .iter()
                    .zip(post_o.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                let fn_x = rmsnorm_cpu(&h1, &n1, eps as f64);
                let fg = matmul_cpu(&fn_x, &w_fg, FFN_SRC, H);
                let fu = matmul_cpu(&fn_x, &w_fu, FFN_SRC, H);
                let mut sw = vec![0.0f64; FFN_PAD];
                for i in 0..FFN_SRC {
                    sw[i] = silu_cpu(fg[i]) * fu[i];
                }
                let fd = matmul_cpu(&sw, &w_fd, H, FFN_PAD);
                let post_f = rmsnorm_cpu(&fd, &p1, eps as f64);
                for d in 0..H {
                    cpu_out[t * H + d] = h1[d] + post_f[d];
                }
            }
            rel_rms(&out_yn[..n_valid * H], &cpu_out[..n_valid * H], "conv_yn");

            let mut cpu_state_next = vec![0.0f64; KVO_TOTAL];
            for col in 0..8 {
                for k in 0..6 {
                    let idx = (n_valid as isize) - 6 + (k as isize);
                    for d in 0..256 {
                        cpu_state_next[col * W_ELEMS + k * 256 + d] = if idx < 0 {
                            conv_state[(6 + idx) as usize * H + col * 256 + d] as f64
                        } else {
                            bx_seq[idx as usize][col * 256 + d]
                        };
                    }
                }
                rel_rms(
                    &out_kvo[col * W_ELEMS..col * W_ELEMS + 6 * 256],
                    &cpu_state_next[col * W_ELEMS..col * W_ELEMS + 6 * 256],
                    &format!("conv_state_c{col}"),
                );
            }
        }

        // --- Test 2: q4prefill_lfm3_attn Mode 1 (c0 & c1 SWA) + Mode 2/3 (bidir 2 chunks = 64 tokens) ---
        if run_attn {
            let k_m1_c0 = dev
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_ATTN_INSTS_C0),
                )
                .expect("load attn m1 c0");
            let k_m1_c1 = dev
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_ATTN_INSTS_C1),
                )
                .expect("load attn m1 c1");
            let k_m2 = dev
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_BIDIR_A_INSTS),
                )
                .expect("load attn m2");
            let k_m3_c1 = dev
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_BIDIR_B_INSTS_C1),
                )
                .expect("load attn m3 c1");

            let mut bo_w = k_m1_c0.alloc_bo(W_TOTAL * 4, 3).unwrap();
            let mut bo_y = k_m1_c0.alloc_bo(ACT_TOTAL * 4, 4).unwrap();
            let mut bo_yn = k_m1_c0.alloc_bo(YN_TOTAL * 4, 5).unwrap();
            let mut bo_kv = k_m1_c0.alloc_bo(KV_TOTAL * 4, 6).unwrap();
            let mut bo_kvo = k_m1_c0.alloc_bo(KVO_TOTAL * 4, 7).unwrap();

            let w_q = make_f32_weights(100, Q_DIM, H, H, 0.04);
            let w_k = make_f32_weights(2200, KV_DIM, H, H, 0.04);
            let w_v = make_f32_weights(2800, KV_DIM, H, H, 0.04);
            let w_g = make_f32_weights(3400, N_H, H, H, 0.04);
            let w_o = make_f32_weights(3500, H, Q_DIM, Q_DIM, 0.04);
            let w_fg = make_f32_weights(5600, FFN_SRC, H, H, 0.03);
            let w_fu = make_f32_weights(13600, FFN_SRC, H, H, 0.03);
            let w_fd = make_f32_weights(21600, H, FFN_SRC, FFN_PAD, 0.02);

            let (w_q, w_k, w_v, w_g, w_o, w_fg, w_fu, w_fd) = {
                let wb = bo_w.as_mut_slice();
                wb.fill(0);
                (
                    pack_w_f32_to_bo(wb, 0, &w_q, Q_DIM, H),
                    pack_w_f32_to_bo(wb, 128, &w_k, KV_DIM, H),
                    pack_w_f32_to_bo(wb, 160, &w_v, KV_DIM, H),
                    pack_w_f32_to_bo(wb, 192, &w_g, N_H, H),
                    pack_w_f32_to_bo(wb, 193, &w_o, H, Q_DIM),
                    pack_w_f32_to_bo(wb, 321, &w_fg, FFN_SRC, H),
                    pack_w_f32_to_bo(wb, 801, &w_fu, FFN_SRC, H),
                    pack_w_f32_to_bo(wb, 1281, &w_fd, H, FFN_PAD),
                )
            };

            let n0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(6, i, 0.2)).collect();
            let p0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(7, i, 0.2)).collect();
            let n1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(8, i, 0.2)).collect();
            let p1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(9, i, 0.2)).collect();
            let qnw: Vec<f32> = (0..HD).map(|i| 0.8 + det_val(10, i, 0.2)).collect();
            let knw: Vec<f32> = (0..HD).map(|i| 0.8 + det_val(11, i, 0.2)).collect();
            let eps = 1e-5f32;

            let make_rope = |pos0: usize| -> Vec<f32> {
                let mut cs = vec![0.0f32; B * HD];
                for t in 0..B {
                    let pos = (pos0 + t) as f32;
                    for i in 0..64 {
                        let ang = pos / 1000000.0f32.powf((2 * i) as f32 / (HD as f32));
                        cs[t * HD + i] = ang.cos();
                        cs[t * HD + 64 + i] = ang.sin();
                    }
                }
                cs
            };

            let stage_attn_meta = |yb: &mut [u8],
                                   hidden: &[f32],
                                   pos0: usize,
                                   n_valid: usize,
                                   kv_ch: usize,
                                   n_past: usize,
                                   is_causal: bool,
                                   swa: usize,
                                   mode: i32| {
                write_le_f32_slice(yb, 0, hidden);
                init_lfm3_prefill_bo_y_norms(yb, &n0, &p0, &n1, &p1, Some(&qnw), Some(&knw));
                let rope = make_rope(pos0);
                write_lfm3_prefill_step_params(
                    yb,
                    pos0,
                    n_valid,
                    kv_ch,
                    n_past,
                    is_causal,
                    swa,
                    mode,
                    eps,
                    Some(&rope),
                );
            };

            // CPU helper for causal/SWA and bidir attention over chunk sequences
            type QkvgTuple = (
                Vec<Vec<f64>>,
                Vec<Vec<f64>>,
                Vec<Vec<f64>>,
                Vec<Vec<f64>>,
                Vec<Vec<f64>>,
            );
            let compute_qkvg = |hidden: &[f32], rope: &[f32]| -> QkvgTuple {
                let n_tok = hidden.len() / H;
                let mut h64 = vec![vec![0.0f64; H]; n_tok];
                let mut qs = vec![vec![0.0f64; Q_DIM]; n_tok];
                let mut ks = vec![vec![0.0f64; KV_DIM]; n_tok];
                let mut vs = vec![vec![0.0f64; KV_DIM]; n_tok];
                let mut gs = vec![vec![0.0f64; N_H]; n_tok];
                for t in 0..n_tok {
                    h64[t] = hidden[t * H..(t + 1) * H]
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                    let xn = rmsnorm_cpu(&h64[t], &n0, eps as f64);
                    let mut q = matmul_cpu(&xn, &w_q, Q_DIM, H);
                    let mut k = matmul_cpu(&xn, &w_k, KV_DIM, H);
                    vs[t] = matmul_cpu(&xn, &w_v, KV_DIM, H);
                    let gl = matmul_cpu(&xn, &w_g, N_H, H);
                    for h in 0..N_H {
                        let mut qh = rmsnorm_cpu(&q[h * HD..(h + 1) * HD], &qnw, eps as f64);
                        neox_rope_cpu(&mut qh, &rope[t * HD..(t + 1) * HD]);
                        q[h * HD..(h + 1) * HD].copy_from_slice(&qh);
                        gs[t][h] = 1.0 / (1.0 + (-gl[h]).exp());
                    }
                    for kh in 0..N_KV {
                        let mut kkh = rmsnorm_cpu(&k[kh * HD..(kh + 1) * HD], &knw, eps as f64);
                        neox_rope_cpu(&mut kkh, &rope[t * HD..(t + 1) * HD]);
                        k[kh * HD..(kh + 1) * HD].copy_from_slice(&kkh);
                    }
                    qs[t] = q;
                    ks[t] = k;
                }
                (h64, qs, ks, vs, gs)
            };

            let run_attn_tail_cpu = |h_in: &[f64],
                                     qs: &[f64],
                                     gs: &[f64],
                                     all_k: &[Vec<f64>],
                                     all_v: &[Vec<f64>],
                                     lo: usize,
                                     hi: usize|
             -> Vec<f64> {
                let scale = 1.0 / (HD as f64).sqrt();
                let mut a_out = vec![0.0f64; Q_DIM];
                for h in 0..N_H {
                    let kvh = h / 4;
                    let qh = &qs[h * HD..(h + 1) * HD];
                    let mut scores = Vec::with_capacity(hi - lo);
                    let mut max_s = f64::NEG_INFINITY;
                    for k_row in all_k.iter().take(hi).skip(lo) {
                        let kh = &k_row[kvh * HD..(kvh + 1) * HD];
                        let s: f64 = qh.iter().zip(kh.iter()).map(|(&a, &b)| a * b * scale).sum();
                        if s > max_s {
                            max_s = s;
                        }
                        scores.push(s);
                    }
                    let mut l = 0.0f64;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        l += *s;
                    }
                    let gate = gs[h] / l;
                    for (idx, kt) in (lo..hi).enumerate() {
                        let p = scores[idx] * gate;
                        let vh = &all_v[kt][kvh * HD..(kvh + 1) * HD];
                        for d in 0..HD {
                            a_out[h * HD + d] += p * vh[d];
                        }
                    }
                }
                let oproj = matmul_cpu(&a_out, &w_o, H, Q_DIM);
                let post_o = rmsnorm_cpu(&oproj, &p0, eps as f64);
                let h1: Vec<f64> = h_in.iter().zip(post_o.iter()).map(|(a, b)| a + b).collect();
                let fn_x = rmsnorm_cpu(&h1, &n1, eps as f64);
                let fg = matmul_cpu(&fn_x, &w_fg, FFN_SRC, H);
                let fu = matmul_cpu(&fn_x, &w_fu, FFN_SRC, H);
                let mut sw = vec![0.0f64; FFN_PAD];
                for i in 0..FFN_SRC {
                    sw[i] = silu_cpu(fg[i]) * fu[i];
                }
                let fd = matmul_cpu(&sw, &w_fd, H, FFN_PAD);
                let post_f = rmsnorm_cpu(&fd, &p1, eps as f64);
                h1.iter().zip(post_f.iter()).map(|(a, b)| a + b).collect()
            };

            bo_w.sync_to_device().unwrap();

            // 2a: Mode 1 causal SWA (c0, pos0=0, n_valid=32, swa=16)
            bo_y.as_mut_slice().fill(0);
            bo_kv.as_mut_slice().fill(0);
            let h_c0: Vec<f32> = (0..B * H).map(|i| det_val(40, i, 0.35)).collect();
            stage_attn_meta(bo_y.as_mut_slice(), &h_c0, 0, B, 0, 0, true, 16, 1);
            bo_y.sync_to_device().unwrap();
            bo_kv.sync_to_device().unwrap();
            let r_2a = k_m1_c0
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            r_2a.start().unwrap();
            r_2a.wait(15_000).unwrap();
            bo_y.sync_from_device().unwrap();
            bo_yn.sync_from_device().unwrap();
            bo_kvo.sync_from_device().unwrap();
            let mut yn0 = vec![0.0f32; B * H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut yn0);
            let rope0 = make_rope(0);
            let (h0_64, q0_64, k0_64, v0_64, g0_64) = compute_qkvg(&h_c0, &rope0);
            let mut cpu0 = Vec::with_capacity(B * H);
            for t in 0..B {
                let lo = (t + 1).saturating_sub(16);
                cpu0.extend_from_slice(&run_attn_tail_cpu(
                    &h0_64[t],
                    &q0_64[t],
                    &g0_64[t],
                    &k0_64,
                    &v0_64,
                    lo,
                    t + 1,
                ));
            }
            rel_rms(&yn0, &cpu0, "attn_m1_c0_swa16");

            // 2b: Mode 1 c1 (pos0=27, n_past=27 past tokens in chunk 0 of bo_kv)
            // Copy kvo chunk from chunk 0 (only 27 valid past tokens needed) by creating 27 past tokens from k0/v0
            let n_past = 27usize;
            {
                let kb = bo_kv.as_mut_slice();
                for kvh in 0..N_KV {
                    let base = kvh * 256 * W_ELEMS * 4;
                    for t in 0..n_past {
                        for d in 0..HD {
                            let kb_bits = f32_to_bf16(k0_64[t][kvh * HD + d] as f32).to_le_bytes();
                            let vb_bits = f32_to_bf16(v0_64[t][kvh * HD + d] as f32).to_le_bytes();
                            let idx = (t * HD + d) * 2;
                            kb[base + idx..base + idx + 2].copy_from_slice(&kb_bits);
                            kb[base + 2304 * 4 + idx..base + 2304 * 4 + idx + 2]
                                .copy_from_slice(&vb_bits);
                        }
                    }
                }
            }
            let h_c1: Vec<f32> = (0..B * H).map(|i| det_val(77, i, 0.35)).collect();
            let n_v1 = 21usize;
            stage_attn_meta(
                bo_y.as_mut_slice(),
                &h_c1,
                n_past,
                n_v1,
                1,
                n_past,
                true,
                32,
                1,
            );
            bo_y.sync_to_device().unwrap();
            bo_kv.sync_to_device().unwrap();
            let r_2b = k_m1_c1
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            r_2b.start().unwrap();
            r_2b.wait(15_000).unwrap();
            bo_yn.sync_from_device().unwrap();
            let mut yn1 = vec![0.0f32; n_v1 * H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut yn1);
            let rope1 = make_rope(n_past);
            let (h1_64, q1_64, k1_64, v1_64, g1_64) = compute_qkvg(&h_c1, &rope1);
            let mut all_k = k0_64[..n_past].to_vec();
            all_k.extend_from_slice(&k1_64[..n_v1]);
            let mut all_v = v0_64[..n_past].to_vec();
            all_v.extend_from_slice(&v1_64[..n_v1]);
            let mut cpu1 = Vec::with_capacity(n_v1 * H);
            for t in 0..n_v1 {
                let gpos = n_past + t;
                let lo = (gpos + 1).saturating_sub(32);
                cpu1.extend(run_attn_tail_cpu(
                    &h1_64[t],
                    &q1_64[t],
                    &g1_64[t],
                    &all_k,
                    &all_v,
                    lo,
                    gpos + 1,
                ));
            }
            rel_rms(&yn1, &cpu1, "attn_m1_c1_swa32");

            // 2c: Bidir Phase A (Mode 2) + Phase B (Mode 3, 2 chunks = 50 tokens -> kv_ch=1 per chunk? Wait chunk1 needs kv_ch=2! Test 32 tokens cleanly with c1 or test cross-chunk with 2 chunks? Note with bidir_b_c1 we can test 32 tokens attending globally to all 32 tokens!)
            //Wait: Mode 3 with kv_ch=1 reads chunk 0 (32 tokens) from bo_kv and ignores cur_kv! Let's run Phase A on 29 tokens -> copy kvo -> bo_kv[chunk 0] -> save q_bf16 & gate -> Phase B (kv_ch=1, n_past=29, is_causal=false) -> verify bidirectional attention on all 29 tokens!
            let n_bidir = 29usize;
            bo_y.as_mut_slice().fill(0);
            bo_kv.as_mut_slice().fill(0);
            stage_attn_meta(bo_y.as_mut_slice(), &h_c0, 0, n_bidir, 0, 0, false, 0, 2);
            bo_y.sync_to_device().unwrap();
            bo_kv.sync_to_device().unwrap();
            let r_2c = k_m2
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            r_2c.start().unwrap();
            r_2c.wait(15_000).unwrap();
            bo_y.sync_from_device().unwrap();
            bo_kvo.sync_from_device().unwrap();
            // Copy kvo -> bo_kv chunk 0
            {
                let ko = bo_kvo.as_slice();
                let kb = bo_kv.as_mut_slice();
                for kvh in 0..N_KV {
                    let src = &ko[kvh * W_ELEMS * 4..(kvh + 1) * W_ELEMS * 4];
                    let dst = &mut kb[(kvh * 256) * W_ELEMS * 4..(kvh * 256 + 1) * W_ELEMS * 4];
                    dst.copy_from_slice(src);
                }
            }
            // Update meta to Mode 3, kv_ch=1, n_past=29, is_causal=false (SLOT_Q_BF16, SLOT_ATTN_GATE, SLOT_HIDDEN_IN already intact in bo_y!)
            {
                let yb = bo_y.as_mut_slice();
                write_le_i32(yb, SLOT_NORM_0 + 2050, 1);
                write_le_i32(yb, SLOT_NORM_0 + 2051, n_bidir as i32);
                write_le_i32(yb, SLOT_NORM_0 + 2052, 0);
                write_le_i32(yb, SLOT_NORM_0 + 2053, 0);
                write_le_i32(yb, SLOT_NORM_0 + 2054, 3);
            }
            bo_y.sync_to_device().unwrap();
            bo_kv.sync_to_device().unwrap();
            let r_2d = k_m3_c1
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            r_2d.start().unwrap();
            r_2d.wait(15_000).unwrap();
            bo_yn.sync_from_device().unwrap();
            let mut yn_b = vec![0.0f32; n_bidir * H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut yn_b);
            let mut cpu_b = Vec::with_capacity(n_bidir * H);
            for t in 0..n_bidir {
                // Full bidirectional global attention [0..n_bidir) (future tokens attend too!)
                cpu_b.extend(run_attn_tail_cpu(
                    &h0_64[t],
                    &q0_64[t],
                    &g0_64[t],
                    &k0_64[..n_bidir],
                    &v0_64[..n_bidir],
                    0,
                    n_bidir,
                ));
            }
            let _ = (SLOT_Q_BF16, SLOT_ATTN_GATE, bf16::ZERO);
            rel_rms(&yn_b, &cpu_b, "attn_bidir_global");
        }
    }

    /// Verify `q4layer_lfm3` (`B=1` fused ShortConv + GQA/SWA Attention decode) on NPU2 hardware.
    #[test]
    #[ignore = "requires AMD XDNA NPU hardware (/dev/accel or xrt_coreutil.dll)"]
    fn hw_verify_q4layer_lfm3() {
        use crate::backend::xdna::device::XdnaDevice;
        use crate::backend::xdna::repack::pack_f32_stage_k1024;
        use half::bf16;

        const H: usize = 2048;
        const Q_DIM: usize = 2048;
        const N_H: usize = 16;
        const N_KV: usize = 4;
        const HD: usize = 128;
        const KV_DIM: usize = N_KV * HD;
        const FFN_SRC: usize = 7168;
        const FFN_PAD: usize = 8192;
        const W_ELEMS: usize = 4608;
        const W_TOTAL: usize = 1992 * W_ELEMS;
        const SLOT_NORM_0: usize = 41472;
        const SLOT_NORM_1: usize = 46080;
        const ACT_TOTAL: usize = LFM3_DEC_ACT_BYTES / 4;
        const KV_TOTAL: usize = 256 * N_KV * W_ELEMS;
        const KVO_TOTAL: usize = 8 * W_ELEMS;

        let det_val = |seed: usize, idx: usize, amp: f32| -> f32 {
            let mut z = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 40) as f32) * (1.0 / ((1u64 << 24) as f32));
            (u * 2.0 - 1.0) * amp
        };

        let make_f32_weights =
            |seed: usize, rows: usize, cols_src: usize, cols_pad: usize, amp: f32| -> Vec<f32> {
                let mut w = vec![0.0f32; rows * cols_pad];
                for r in 0..rows {
                    for c in 0..cols_src {
                        w[r * cols_pad + c] = det_val(seed + r, c, amp);
                    }
                }
                w
            };

        let pack_w_f32_to_bo = |bo_bytes: &mut [u8],
                                tile_off: usize,
                                w: &[f32],
                                rows: usize,
                                cols: usize|
         -> Vec<f32> {
            let need = crate::backend::xdna::repack::stage_bytes(rows, cols).unwrap();
            let dst = &mut bo_bytes[tile_off * W_ELEMS * 4..tile_off * W_ELEMS * 4 + need];
            pack_f32_stage_k1024(dst, w, rows, cols).unwrap();
            let mut decoded = vec![0.0f32; rows * cols];
            let chunk_k = 1024usize.min(cols);
            let chunks = cols / chunk_k;
            let nb_chunk = chunk_k / 32;
            let tile_bytes = nb_chunk * 288;
            for tile in 0..rows / 16 {
                for ch in 0..chunks {
                    let tp = (tile * chunks + ch) * tile_bytes;
                    for lb in 0..nb_chunk {
                        let b = ch * nb_chunk + lb;
                        for n in 0..16 {
                            let sp = tp + nb_chunk * 256 + lb * 32 + n * 2;
                            let d = bf16::from_le_bytes([dst[sp], dst[sp + 1]]).to_f32();
                            for h in 0..2 {
                                for kk in 0..16 {
                                    let e = kk * 16 + n;
                                    let byte = dst[tp + lb * 256 + h * 128 + e / 2];
                                    let nib = if e % 2 == 0 { byte & 0xF } else { byte >> 4 } as i8;
                                    let q = if nib >= 8 { nib - 16 } else { nib };
                                    decoded[(tile * 16 + n) * cols + b * 32 + h * 16 + kk] =
                                        (q as f32) * d;
                                }
                            }
                        }
                    }
                }
            }
            decoded
        };

        let rmsnorm_cpu = |x: &[f64], w: &[f32], eps: f64| -> Vec<f64> {
            let n = x.len();
            let ss: f64 = x.iter().map(|v| v * v).sum::<f64>() / (n as f64);
            let inv = 1.0 / (ss + eps).sqrt();
            (0..n).map(|i| x[i] * inv * (w[i] as f64)).collect()
        };

        let matmul_cpu = |x: &[f64], w: &[f32], rows: usize, cols: usize| -> Vec<f64> {
            (0..rows)
                .map(|r| {
                    let row = &w[r * cols..(r + 1) * cols];
                    row.iter()
                        .zip(x.iter())
                        .map(|(&a, &b)| (a as f64) * b)
                        .sum()
                })
                .collect()
        };

        let silu_cpu = |x: f64| -> f64 { x / (1.0 + (-x).exp()) };

        let neox_rope_cpu = |v: &mut [f64], cs: &[f32]| {
            for i in 0..64 {
                let x0 = v[i];
                let x1 = v[64 + i];
                let c = cs[i] as f64;
                let s = cs[64 + i] as f64;
                v[i] = x0 * c - x1 * s;
                v[64 + i] = x0 * s + x1 * c;
            }
        };

        let rel_rms = |got: &[f32], ref_f64: &[f64], label: &str| {
            let nans = got.iter().filter(|v| !v.is_finite()).count();
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (&g, &r) in got.iter().zip(ref_f64.iter()) {
                let d = (g as f64) - r;
                num += d * d;
                den += r * r;
            }
            let rr = (num / den.max(1e-30)).sqrt();
            eprintln!("{label}: nans={nans} rel_rms={rr:.5}");
            assert_eq!(nans, 0, "{label} produced NaNs");
            assert!(rr < 0.03, "{label} rel_rms {rr:.5} >= 0.03");
        };

        let dev = XdnaDevice::open(0).expect("open XDNA device");

        // --- Test 1: ShortConv + FFN decode (layer_mode = 0) ---
        {
            let k_conv = dev
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(Q4LAYER_LFM3_CONV_INSTS))
                .expect("load q4layer_lfm3 conv");
            let mut bo_w = k_conv.alloc_bo(W_TOTAL * 4, 3).unwrap();
            let mut bo_y = k_conv.alloc_bo(ACT_TOTAL * 4, 4).unwrap();
            let mut bo_yn = k_conv.alloc_bo(H * 4, 5).unwrap();
            let mut bo_kv = k_conv.alloc_bo(KV_TOTAL * 4, 6).unwrap();
            let mut bo_kvo = k_conv.alloc_bo(KVO_TOTAL * 4, 7).unwrap();

            let w_b = make_f32_weights(10, H, H, H, 0.04);
            let w_c = make_f32_weights(2100, H, H, H, 0.04);
            let w_x = make_f32_weights(4200, H, H, H, 0.04);
            let w_out = make_f32_weights(6300, H, H, H, 0.04);
            let w_fg = make_f32_weights(8400, FFN_SRC, H, H, 0.03);
            let w_fu = make_f32_weights(16400, FFN_SRC, H, H, 0.03);
            let w_fd = make_f32_weights(24400, H, FFN_SRC, FFN_PAD, 0.02);

            let (w_b, w_c, w_x, w_out, w_fg, w_fu, w_fd) = {
                let wb = bo_w.as_mut_slice();
                wb.fill(0);
                let b = pack_w_f32_to_bo(wb, 0, &w_b, H, H);
                let c = pack_w_f32_to_bo(wb, 128, &w_c, H, H);
                let x = pack_w_f32_to_bo(wb, 256, &w_x, H, H);
                let o = pack_w_f32_to_bo(wb, 384, &w_out, H, H);
                let fg = pack_w_f32_to_bo(wb, 520, &w_fg, FFN_SRC, H);
                let fu = pack_w_f32_to_bo(wb, 1000, &w_fu, FFN_SRC, H);
                let fd = pack_w_f32_to_bo(wb, 1480, &w_fd, H, FFN_PAD);
                (b, c, x, o, fg, fu, fd)
            };

            let conv_wt: Vec<f32> = (0..7 * H).map(|i| det_val(90, i, 0.35)).collect();
            for col in 0..8usize {
                for k in 0..7usize {
                    let src = &conv_wt[k * H + col * 256..k * H + (col + 1) * 256];
                    write_le_f32_slice(
                        bo_w.as_mut_slice(),
                        512 * W_ELEMS + col * W_ELEMS + k * 256,
                        src,
                    );
                }
            }
            bo_w.sync_to_device().unwrap();

            let conv_state: Vec<f32> = (0..6 * H).map(|i| det_val(55, i, 0.5)).collect();
            bo_kv.as_mut_slice().fill(0);
            for col in 0..8usize {
                for k in 0..6usize {
                    let src = &conv_state[k * H + col * 256..k * H + (col + 1) * 256];
                    write_le_f32_slice(bo_kv.as_mut_slice(), col * W_ELEMS + k * 256, src);
                }
            }
            bo_kv.sync_to_device().unwrap();

            let h_in: Vec<f32> = (0..H).map(|i| det_val(1, i, 0.7)).collect();
            let n0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(2, i, 0.2)).collect();
            let p0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(3, i, 0.2)).collect();
            let n1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(4, i, 0.2)).collect();
            let p1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(5, i, 0.2)).collect();
            let eps = 1e-5f32;

            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                write_le_f32_slice(yb, 0, &h_in);
                write_le_f32_slice(yb, SLOT_NORM_0, &n0);
                write_le_f32_slice(yb, SLOT_NORM_0 + H, &p0);
                write_le_i32(yb, SLOT_NORM_0 + 4096, 17);
                write_le_i32(yb, SLOT_NORM_0 + 4098, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4099, 17);
                write_le_i32(yb, SLOT_NORM_0 + 4100, 1);
                write_le_i32(yb, SLOT_NORM_0 + 4101, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4102, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4103, eps.to_bits() as i32);
                write_le_f32_slice(yb, SLOT_NORM_1, &n1);
                write_le_f32_slice(yb, SLOT_NORM_1 + H, &p1);
            }
            bo_y.sync_to_device().unwrap();
            bo_yn.as_mut_slice().fill(0);
            bo_yn.sync_to_device().unwrap();
            bo_kvo.as_mut_slice().fill(0);
            bo_kvo.sync_to_device().unwrap();

            let run = k_conv
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            run.start().unwrap();
            run.wait(15_000).unwrap();
            bo_yn.sync_from_device().unwrap();
            bo_kvo.sync_from_device().unwrap();

            let mut out_yn = vec![0.0f32; H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut out_yn);
            let mut out_kvo = vec![0.0f32; KVO_TOTAL];
            read_le_f32_slice(bo_kvo.as_slice(), 0, &mut out_kvo);

            let h64: Vec<f64> = h_in.iter().map(|&v| v as f64).collect();
            let xn = rmsnorm_cpu(&h64, &n0, eps as f64);
            let bv = matmul_cpu(&xn, &w_b, H, H);
            let cv = matmul_cpu(&xn, &w_c, H, H);
            let xv = matmul_cpu(&xn, &w_x, H, H);
            let mut bx = vec![0.0f64; H];
            let mut conv_y = vec![0.0f64; H];
            for d in 0..H {
                bx[d] = bv[d] * xv[d];
                let mut acc = bx[d] * (conv_wt[6 * H + d] as f64);
                for k in 0..6 {
                    acc += (conv_state[k * H + d] as f64) * (conv_wt[k * H + d] as f64);
                }
                conv_y[d] = acc * cv[d];
            }
            let op = matmul_cpu(&conv_y, &w_out, H, H);
            let post_op = rmsnorm_cpu(&op, &p0, eps as f64);
            let h1: Vec<f64> = h64.iter().zip(post_op.iter()).map(|(a, b)| a + b).collect();
            let fn_x = rmsnorm_cpu(&h1, &n1, eps as f64);
            let fg = matmul_cpu(&fn_x, &w_fg, FFN_SRC, H);
            let fu = matmul_cpu(&fn_x, &w_fu, FFN_SRC, H);
            let mut sw = vec![0.0f64; FFN_PAD];
            for i in 0..FFN_SRC {
                sw[i] = silu_cpu(fg[i]) * fu[i];
            }
            let fd = matmul_cpu(&sw, &w_fd, H, FFN_PAD);
            let post_f = rmsnorm_cpu(&fd, &p1, eps as f64);
            let cpu_out: Vec<f64> = h1.iter().zip(post_f.iter()).map(|(a, b)| a + b).collect();
            rel_rms(&out_yn, &cpu_out, "dec_conv_yn");

            let mut cpu_st_next = vec![0.0f64; KVO_TOTAL];
            for col in 0..8 {
                for k in 0..5 {
                    for d in 0..256 {
                        cpu_st_next[col * W_ELEMS + k * 256 + d] =
                            conv_state[(k + 1) * H + col * 256 + d] as f64;
                    }
                }
                for d in 0..256 {
                    cpu_st_next[col * W_ELEMS + 5 * 256 + d] = bx[col * 256 + d];
                }
                rel_rms(
                    &out_kvo[col * W_ELEMS..col * W_ELEMS + 6 * 256],
                    &cpu_st_next[col * W_ELEMS..col * W_ELEMS + 6 * 256],
                    &format!("dec_conv_state_c{col}"),
                );
            }
        }

        // --- Test 2: Attention + FFN decode (c0 at pos=0, and c1 SWA16 at pos=27) ---
        {
            let k_c0 = dev
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(Q4LAYER_LFM3_ATTN_INSTS_C0))
                .expect("load q4layer_lfm3 attn c0");
            let k_c1 = dev
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(Q4LAYER_LFM3_ATTN_INSTS_C1))
                .expect("load q4layer_lfm3 attn c1");

            let mut bo_w = k_c0.alloc_bo(W_TOTAL * 4, 3).unwrap();
            let mut bo_y = k_c0.alloc_bo(ACT_TOTAL * 4, 4).unwrap();
            let mut bo_yn = k_c0.alloc_bo(H * 4, 5).unwrap();
            let mut bo_kv = k_c0.alloc_bo(KV_TOTAL * 4, 6).unwrap();
            let mut bo_kvo = k_c0.alloc_bo(KVO_TOTAL * 4, 7).unwrap();

            let w_q = make_f32_weights(100, Q_DIM, H, H, 0.04);
            let w_k = make_f32_weights(2200, KV_DIM, H, H, 0.04);
            let w_v = make_f32_weights(2800, KV_DIM, H, H, 0.04);
            let w_g = make_f32_weights(3400, N_H, H, H, 0.04);
            let w_o = make_f32_weights(3500, H, Q_DIM, Q_DIM, 0.04);
            let w_fg = make_f32_weights(5600, FFN_SRC, H, H, 0.03);
            let w_fu = make_f32_weights(13600, FFN_SRC, H, H, 0.03);
            let w_fd = make_f32_weights(21600, H, FFN_SRC, FFN_PAD, 0.02);

            let (w_q, w_k, w_v, w_g, w_o, w_fg, w_fu, w_fd) = {
                let wb = bo_w.as_mut_slice();
                wb.fill(0);
                (
                    pack_w_f32_to_bo(wb, 0, &w_q, Q_DIM, H),
                    pack_w_f32_to_bo(wb, 128, &w_k, KV_DIM, H),
                    pack_w_f32_to_bo(wb, 160, &w_v, KV_DIM, H),
                    pack_w_f32_to_bo(wb, 192, &w_g, N_H, H),
                    pack_w_f32_to_bo(wb, 193, &w_o, H, Q_DIM),
                    pack_w_f32_to_bo(wb, 321, &w_fg, FFN_SRC, H),
                    pack_w_f32_to_bo(wb, 801, &w_fu, FFN_SRC, H),
                    pack_w_f32_to_bo(wb, 1281, &w_fd, H, FFN_PAD),
                )
            };
            bo_w.sync_to_device().unwrap();

            let n0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(6, i, 0.2)).collect();
            let p0: Vec<f32> = (0..H).map(|i| 0.8 + det_val(7, i, 0.2)).collect();
            let n1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(8, i, 0.2)).collect();
            let p1: Vec<f32> = (0..H).map(|i| 0.8 + det_val(9, i, 0.2)).collect();
            let qnw: Vec<f32> = (0..HD).map(|i| 0.8 + det_val(10, i, 0.2)).collect();
            let knw: Vec<f32> = (0..HD).map(|i| 0.8 + det_val(11, i, 0.2)).collect();
            let eps = 1e-5f32;

            let make_rope = |pos: usize| -> Vec<f32> {
                let mut cs = vec![0.0f32; HD];
                for i in 0..64 {
                    let ang = (pos as f32) / 1000000.0f32.powf((2 * i) as f32 / (HD as f32));
                    cs[i] = ang.cos();
                    cs[64 + i] = ang.sin();
                }
                cs
            };

            let cpu_step = |h_in: &[f32],
                            rope: &[f32],
                            past_k: &[Vec<f64>],
                            past_v: &[Vec<f64>],
                            lo: usize|
             -> (Vec<f64>, Vec<f64>, Vec<f64>) {
                let h64: Vec<f64> = h_in.iter().map(|&v| v as f64).collect();
                let xn = rmsnorm_cpu(&h64, &n0, eps as f64);
                let mut q = matmul_cpu(&xn, &w_q, Q_DIM, H);
                let mut k = matmul_cpu(&xn, &w_k, KV_DIM, H);
                let v = matmul_cpu(&xn, &w_v, KV_DIM, H);
                let gl = matmul_cpu(&xn, &w_g, N_H, H);
                let mut gs = [0.0f64; N_H];
                for h in 0..N_H {
                    let mut qh = rmsnorm_cpu(&q[h * HD..(h + 1) * HD], &qnw, eps as f64);
                    neox_rope_cpu(&mut qh, rope);
                    q[h * HD..(h + 1) * HD].copy_from_slice(&qh);
                    gs[h] = 1.0 / (1.0 + (-gl[h]).exp());
                }
                for kh in 0..N_KV {
                    let mut kh_n = rmsnorm_cpu(&k[kh * HD..(kh + 1) * HD], &knw, eps as f64);
                    neox_rope_cpu(&mut kh_n, rope);
                    k[kh * HD..(kh + 1) * HD].copy_from_slice(&kh_n);
                }

                let mut all_k: Vec<Vec<f64>> = past_k.to_vec();
                all_k.push(k.clone());
                let mut all_v: Vec<Vec<f64>> = past_v.to_vec();
                all_v.push(v.clone());

                let scale = 1.0 / (HD as f64).sqrt();
                let mut a_out = vec![0.0f64; Q_DIM];
                let hi = all_k.len();
                for h in 0..N_H {
                    let kvh = h / 4;
                    let qh = &q[h * HD..(h + 1) * HD];
                    let mut scores = Vec::with_capacity(hi - lo);
                    let mut max_s = f64::NEG_INFINITY;
                    for k_row in all_k.iter().take(hi).skip(lo) {
                        let kh = &k_row[kvh * HD..(kvh + 1) * HD];
                        let s: f64 = qh.iter().zip(kh.iter()).map(|(&a, &b)| a * b * scale).sum();
                        if s > max_s {
                            max_s = s;
                        }
                        scores.push(s);
                    }
                    let mut l = 0.0f64;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        l += *s;
                    }
                    let gsc = gs[h] / l;
                    for (idx, kt) in (lo..hi).enumerate() {
                        let p = scores[idx] * gsc;
                        let vh = &all_v[kt][kvh * HD..(kvh + 1) * HD];
                        for d in 0..HD {
                            a_out[h * HD + d] += p * vh[d];
                        }
                    }
                }
                let oproj = matmul_cpu(&a_out, &w_o, H, Q_DIM);
                let post_o = rmsnorm_cpu(&oproj, &p0, eps as f64);
                let h1: Vec<f64> = h64.iter().zip(post_o.iter()).map(|(a, b)| a + b).collect();
                let fn_x = rmsnorm_cpu(&h1, &n1, eps as f64);
                let fg = matmul_cpu(&fn_x, &w_fg, FFN_SRC, H);
                let fu = matmul_cpu(&fn_x, &w_fu, FFN_SRC, H);
                let mut sw = vec![0.0f64; FFN_PAD];
                for i in 0..FFN_SRC {
                    sw[i] = silu_cpu(fg[i]) * fu[i];
                }
                let fd = matmul_cpu(&sw, &w_fd, H, FFN_PAD);
                let post_f = rmsnorm_cpu(&fd, &p1, eps as f64);
                let yn: Vec<f64> = h1.iter().zip(post_f.iter()).map(|(a, b)| a + b).collect();
                (yn, k, v)
            };

            // 2a: kv_chunks = 0, pos = 0
            let h0: Vec<f32> = (0..H).map(|i| det_val(21, i, 0.7)).collect();
            let r0 = make_rope(0);
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                write_le_f32_slice(yb, 0, &h0);
                write_le_f32_slice(yb, SLOT_NORM_0, &n0);
                write_le_f32_slice(yb, SLOT_NORM_0 + H, &p0);
                write_le_i32(yb, SLOT_NORM_0 + 4096, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4098, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4099, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4100, 1);
                write_le_i32(yb, SLOT_NORM_0 + 4101, 0);
                write_le_i32(yb, SLOT_NORM_0 + 4102, 1);
                write_le_i32(yb, SLOT_NORM_0 + 4103, eps.to_bits() as i32);
                write_le_f32_slice(yb, SLOT_NORM_1, &n1);
                write_le_f32_slice(yb, SLOT_NORM_1 + H, &p1);
                write_le_f32_slice(yb, SLOT_NORM_0 + 4112, &r0);
                write_le_f32_slice(yb, SLOT_NORM_0 + 4240, &qnw);
                write_le_f32_slice(yb, SLOT_NORM_0 + 4368, &knw);
            }
            bo_y.sync_to_device().unwrap();
            bo_kv.as_mut_slice().fill(0);
            bo_kv.sync_to_device().unwrap();
            let run0 = k_c0
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            run0.start().unwrap();
            run0.wait(0).unwrap();
            bo_yn.sync_from_device().unwrap();
            bo_kvo.sync_from_device().unwrap();
            let mut yn0 = vec![0.0f32; H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut yn0);
            let (cpu0, k_c0_ref, v_c0_ref) = cpu_step(&h0, &r0, &[], &[], 0);

            // Check cur KV output in kvo_buf before asserting yn0
            let kvo_bytes = bo_kvo.as_slice();
            let mut k_got = vec![0.0f32; KV_DIM];
            let mut v_got = vec![0.0f32; KV_DIM];
            for kvh in 0..N_KV {
                let k_off = kvh * 512 * 4 + 128 * 4;
                let v_off = kvh * 512 * 4 + 192 * 4;
                for d in 0..HD {
                    k_got[kvh * HD + d] = bf16::from_le_bytes([
                        kvo_bytes[k_off + d * 2],
                        kvo_bytes[k_off + d * 2 + 1],
                    ])
                    .to_f32();
                    v_got[kvh * HD + d] = bf16::from_le_bytes([
                        kvo_bytes[v_off + d * 2],
                        kvo_bytes[v_off + d * 2 + 1],
                    ])
                    .to_f32();
                }
            }
            rel_rms(&k_got, &k_c0_ref, "dec_attn_k_cur");
            rel_rms(&v_got, &v_c0_ref, "dec_attn_v_cur");
            rel_rms(&yn0, &cpu0, "dec_attn_c0");

            // 2b: kv_chunks = 1, n_past = 27, pos = 27, SWA = 16 -> attends to [12..28)
            let n_past = 27usize;
            let mut past_k = Vec::with_capacity(n_past);
            let mut past_v = Vec::with_capacity(n_past);
            {
                let kb = bo_kv.as_mut_slice();
                kb.fill(0);
                for kt in 0..n_past {
                    let mut kt_v = Vec::with_capacity(KV_DIM);
                    let mut vt_v = Vec::with_capacity(KV_DIM);
                    for kvh in 0..N_KV {
                        let k_base = (kvh * 256) * W_ELEMS * 4 + kt * HD * 2;
                        let v_base = k_base + 2304 * 4;
                        for d in 0..HD {
                            let kv = bf16::from_f32(det_val(70 + kt, kvh * HD + d, 0.6));
                            let vv = bf16::from_f32(det_val(140 + kt, kvh * HD + d, 0.6));
                            kb[k_base + d * 2..k_base + d * 2 + 2]
                                .copy_from_slice(&kv.to_le_bytes());
                            kb[v_base + d * 2..v_base + d * 2 + 2]
                                .copy_from_slice(&vv.to_le_bytes());
                            kt_v.push(kv.to_f32() as f64);
                            vt_v.push(vv.to_f32() as f64);
                        }
                    }
                    past_k.push(kt_v);
                    past_v.push(vt_v);
                }
            }
            bo_kv.sync_to_device().unwrap();
            let h1: Vec<f32> = (0..H).map(|i| det_val(31, i, 0.7)).collect();
            let r1 = make_rope(n_past);
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                write_le_f32_slice(yb, 0, &h1);
                write_le_f32_slice(yb, SLOT_NORM_0, &n0);
                write_le_f32_slice(yb, SLOT_NORM_0 + H, &p0);
                write_le_i32(yb, SLOT_NORM_0 + 4096, n_past as i32);
                write_le_i32(yb, SLOT_NORM_0 + 4098, 1);
                write_le_i32(yb, SLOT_NORM_0 + 4099, n_past as i32);
                write_le_i32(yb, SLOT_NORM_0 + 4100, 1);
                write_le_i32(yb, SLOT_NORM_0 + 4101, 16);
                write_le_i32(yb, SLOT_NORM_0 + 4102, 1);
                write_le_i32(yb, SLOT_NORM_0 + 4103, eps.to_bits() as i32);
                write_le_f32_slice(yb, SLOT_NORM_1, &n1);
                write_le_f32_slice(yb, SLOT_NORM_1 + H, &p1);
                write_le_f32_slice(yb, SLOT_NORM_0 + 4112, &r1);
                write_le_f32_slice(yb, SLOT_NORM_0 + 4240, &qnw);
                write_le_f32_slice(yb, SLOT_NORM_0 + 4368, &knw);
            }
            bo_y.sync_to_device().unwrap();
            bo_yn.as_mut_slice().fill(0);
            bo_yn.sync_to_device().unwrap();
            bo_kvo.as_mut_slice().fill(0);
            bo_kvo.sync_to_device().unwrap();
            let run1 = k_c1
                .create_run(&[&bo_w, &bo_y, &bo_yn, &bo_kv, &bo_kvo])
                .unwrap();
            run1.start().unwrap();
            run1.wait(0).unwrap();
            bo_yn.sync_from_device().unwrap();
            let mut yn1 = vec![0.0f32; H];
            read_le_f32_slice(bo_yn.as_slice(), 0, &mut yn1);
            let lo = (n_past + 1).saturating_sub(16);
            let (cpu1, _, _) = cpu_step(&h1, &r1, &past_k, &past_v, lo);
            rel_rms(&yn1, &cpu1, "dec_attn_c1_swa16");
        }
    }
}

//! Native Qualcomm Hexagon NPU Whisper ASR model.
//!
//! Accelerates the complete OpenAI Whisper speech recognition pipeline:
//! including 1D convolution subsampling stem, audio encoder, cross-attention,
//! and autoregressive decoder: on Qualcomm Hexagon Tensor Processors (HTP)
//! using FastRPC shared memory and asynchronous command queues.
//!
//! Keeps all weights and activations inside three unified DMA buffers (rpcmem)
//! to stay within Qualcomm FastRPC HTP_MAX_MMAPS = 16 mapping bounds, executing
//! convolutions, LayerNorm, linear GEMM, unmasked FlashAttention, and token
//! sampling on the NPU for background execution on Android.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::hexagon::dispatch::{self, LayerNormArgs, TokenShape, TokenTile};
use crate::backend::hexagon::{
    FastRpcDriver, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonContext,
    HexagonDevice, HexagonQueueSession, HexagonWeightFormat, HtpDataType, HtpOpCode, RpcmemBuffer,
    align128, build_binary_kernel_params, build_flash_attn_kernel_params,
    build_mul_mat_kernel_params, hexagon_warn, quantize_f32_to_q8_0, repack_q4_0, repack_q8_0,
    repacked_matrix_size_q4_0, repacked_matrix_size_q8_0,
};
use crate::backend::hexagon::{
    LockOrRecover, lock_or_discard, lock_reporting_poison, report_poison,
};
use crate::model::weights::MmapWeight;
use crate::model::whisper::{
    Conv1dWeights, WhisperConfig, WhisperSpecialTokens, WhisperTranscribeOpts, WhisperWeights,
};
use crate::session::CeraError;
use crate::tensor::DType;

/// Token count per tile for the Whisper encoder (1,500 frames exceed VTCM in
/// one op). The ViT and detokenizer run whole (`VIT_TILE`, `DETOK_TILE`).
const WHISPER_TILE: TokenTile = TokenTile::Tiles(64);

/// Tensors per batch for the encoder, from `CERA_WHISPER_FLUSH_TENSORS`; unset means one
/// batch per layer.
fn encoder_flush_cap() -> Option<usize> {
    static CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CERA_WHISPER_FLUSH_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

fn plan_vec_f32(cur_off: &mut usize, len: usize) -> usize {
    let off = *cur_off;
    *cur_off += align128(len * 4);
    off
}

fn plan_vec_f16(cur_off: &mut usize, len: usize) -> usize {
    let off = *cur_off;
    *cur_off += align128(len * 2);
    off
}

fn plan_linear(cur_off: &mut usize, w: &MmapWeight) -> Result<HexagonWhisperWeightDesc, CeraError> {
    let off = *cur_off;
    let (fmt, sz) = match w.dtype {
        DType::Q8_0 => {
            let sz = repacked_matrix_size_q8_0(w.cols, w.rows)?;
            (HexagonWeightFormat::RepackedQ8_0, sz)
        }
        DType::Q4_0 => {
            let sz = repacked_matrix_size_q4_0(w.cols, w.rows)?;
            (HexagonWeightFormat::RepackedQ4_0, sz)
        }
        DType::F16 | DType::F32 => {
            let sz = repacked_matrix_size_q8_0(w.cols, w.rows)?;
            (HexagonWeightFormat::RepackedQ8_0, sz)
        }
        other => {
            return Err(CeraError::Backend(format!(
                "unsupported whisper weight dtype {other:?} for NPU execution; expected Q8_0, Q4_0, F16, or F32"
            )));
        }
    };
    *cur_off += align128(sz);
    Ok(HexagonWhisperWeightDesc {
        offset: off,
        size_bytes: sz,
        format: fmt,
        rows: w.rows,
        cols: w.cols,
    })
}

pub use crate::backend::hexagon::types::HexagonWeightDesc as HexagonWhisperWeightDesc;

/// 3-slice decomposition weights for a 1D convolution layer with kernel size 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonWhisperConvWeights {
    pub w0: HexagonWhisperWeightDesc,
    pub w1: HexagonWhisperWeightDesc,
    pub w2: HexagonWhisperWeightDesc,
    pub bias_off: usize,
}

/// Weight offsets for one Whisper audio encoder Transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonWhisperEncoderBlockOffsets {
    pub attn_ln_w_off: usize,
    pub attn_ln_b_off: usize,
    pub q_w: HexagonWhisperWeightDesc,
    pub q_b_off: Option<usize>,
    pub k_w: HexagonWhisperWeightDesc,
    pub k_b_off: Option<usize>,
    pub v_w: HexagonWhisperWeightDesc,
    pub v_b_off: Option<usize>,
    pub o_w: HexagonWhisperWeightDesc,
    pub o_b_off: Option<usize>,

    pub mlp_ln_w_off: usize,
    pub mlp_ln_b_off: usize,
    pub mlp_0_w: HexagonWhisperWeightDesc,
    pub mlp_0_b_off: Option<usize>,
    pub mlp_2_w: HexagonWhisperWeightDesc,
    pub mlp_2_b_off: Option<usize>,
}

/// Weight offsets for one Whisper text decoder Transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonWhisperDecoderBlockOffsets {
    // 1. Masked Causal Self-Attention
    pub attn_ln_w_off: usize,
    pub attn_ln_b_off: usize,
    pub attn_q_w: HexagonWhisperWeightDesc,
    pub attn_q_b_off: Option<usize>,
    pub attn_k_w: HexagonWhisperWeightDesc,
    pub attn_k_b_off: Option<usize>,
    pub attn_v_w: HexagonWhisperWeightDesc,
    pub attn_v_b_off: Option<usize>,
    pub attn_out_w: HexagonWhisperWeightDesc,
    pub attn_out_b_off: Option<usize>,

    // 2. Cross-Attention over static encoder hidden states
    pub cross_attn_ln_w_off: usize,
    pub cross_attn_ln_b_off: usize,
    pub cross_attn_q_w: HexagonWhisperWeightDesc,
    pub cross_attn_q_b_off: Option<usize>,
    pub cross_attn_k_w: HexagonWhisperWeightDesc,
    pub cross_attn_k_b_off: Option<usize>,
    pub cross_attn_v_w: HexagonWhisperWeightDesc,
    pub cross_attn_v_b_off: Option<usize>,
    pub cross_attn_out_w: HexagonWhisperWeightDesc,
    pub cross_attn_out_b_off: Option<usize>,

    // 3. MLP with GELU
    pub mlp_ln_w_off: usize,
    pub mlp_ln_b_off: usize,
    pub mlp_0_w: HexagonWhisperWeightDesc,
    pub mlp_0_b_off: Option<usize>,
    pub mlp_2_w: HexagonWhisperWeightDesc,
    pub mlp_2_b_off: Option<usize>,
}

/// Complete weights buffer layout offsets across all Whisper layers in `weights_buf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonWhisperWeightOffsets {
    pub conv1: HexagonWhisperConvWeights,
    pub conv2: HexagonWhisperConvWeights,
    pub encoder_pos_embed_off: usize,
    pub encoder_blocks: Vec<HexagonWhisperEncoderBlockOffsets>,
    pub encoder_ln_post_w_off: usize,
    pub encoder_ln_post_b_off: usize,

    pub decoder_blocks: Vec<HexagonWhisperDecoderBlockOffsets>,
    pub decoder_ln_post_w_off: usize,
    pub decoder_ln_post_b_off: usize,
    pub proj_w: HexagonWhisperWeightDesc,

    pub total_bytes: usize,
}

impl HexagonWhisperWeightOffsets {
    /// Compute memory layout and byte offsets for all Whisper weights in shared rpcmem.
    pub fn plan(weights: &WhisperWeights) -> Result<Self, CeraError> {
        let mut cur_off = 0;
        let enc = &weights.encoder;
        let dec = &weights.decoder;

        // Helper for Conv1D 3-tap weights
        let plan_conv = |cur_off: &mut usize,
                         out_ch: usize,
                         in_ch: usize,
                         bias_len: usize|
         -> Result<HexagonWhisperConvWeights, CeraError> {
            let padded_in_ch = in_ch.next_multiple_of(32);
            let sz = repacked_matrix_size_q8_0(padded_in_ch, out_ch)?;
            let w0_off = *cur_off;
            *cur_off += align128(sz);
            let w1_off = *cur_off;
            *cur_off += align128(sz);
            let w2_off = *cur_off;
            *cur_off += align128(sz);
            let bias_off = plan_vec_f32(cur_off, bias_len);

            Ok(HexagonWhisperConvWeights {
                w0: HexagonWhisperWeightDesc {
                    offset: w0_off,
                    size_bytes: sz,
                    format: HexagonWeightFormat::RepackedQ8_0,
                    rows: out_ch,
                    cols: padded_in_ch,
                },
                w1: HexagonWhisperWeightDesc {
                    offset: w1_off,
                    size_bytes: sz,
                    format: HexagonWeightFormat::RepackedQ8_0,
                    rows: out_ch,
                    cols: padded_in_ch,
                },
                w2: HexagonWhisperWeightDesc {
                    offset: w2_off,
                    size_bytes: sz,
                    format: HexagonWeightFormat::RepackedQ8_0,
                    rows: out_ch,
                    cols: padded_in_ch,
                },
                bias_off,
            })
        };

        let conv1 = plan_conv(
            &mut cur_off,
            enc.conv1.out_channels,
            enc.conv1.in_channels,
            enc.conv1.bias.len(),
        )?;
        let conv2 = plan_conv(
            &mut cur_off,
            enc.conv2.out_channels,
            enc.conv2.in_channels,
            enc.conv2.bias.len(),
        )?;
        let encoder_pos_embed_off = plan_vec_f32(&mut cur_off, enc.positional_embedding.len());

        let mut encoder_blocks = Vec::with_capacity(enc.blocks.len());
        for eb in &enc.blocks {
            let attn_ln_w_off = plan_vec_f32(&mut cur_off, eb.attn_ln_w.len());
            let attn_ln_b_off = plan_vec_f32(&mut cur_off, eb.attn_ln_b.len());
            let q_w = plan_linear(&mut cur_off, &eb.attn_q_w)?;
            let q_b_off = eb
                .attn_q_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let k_w = plan_linear(&mut cur_off, &eb.attn_k_w)?;
            let k_b_off = eb
                .attn_k_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let v_w = plan_linear(&mut cur_off, &eb.attn_v_w)?;
            let v_b_off = eb
                .attn_v_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let o_w = plan_linear(&mut cur_off, &eb.attn_out_w)?;
            let o_b_off = eb
                .attn_out_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));

            let mlp_ln_w_off = plan_vec_f32(&mut cur_off, eb.mlp_ln_w.len());
            let mlp_ln_b_off = plan_vec_f32(&mut cur_off, eb.mlp_ln_b.len());
            let mlp_0_w = plan_linear(&mut cur_off, &eb.mlp_0_w)?;
            let mlp_0_b_off = eb
                .mlp_0_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let mlp_2_w = plan_linear(&mut cur_off, &eb.mlp_2_w)?;
            let mlp_2_b_off = eb
                .mlp_2_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));

            encoder_blocks.push(HexagonWhisperEncoderBlockOffsets {
                attn_ln_w_off,
                attn_ln_b_off,
                q_w,
                q_b_off,
                k_w,
                k_b_off,
                v_w,
                v_b_off,
                o_w,
                o_b_off,
                mlp_ln_w_off,
                mlp_ln_b_off,
                mlp_0_w,
                mlp_0_b_off,
                mlp_2_w,
                mlp_2_b_off,
            });
        }

        let encoder_ln_post_w_off = plan_vec_f32(&mut cur_off, enc.ln_post_w.len());
        let encoder_ln_post_b_off = plan_vec_f32(&mut cur_off, enc.ln_post_b.len());

        let mut decoder_blocks = Vec::with_capacity(dec.blocks.len());
        for db in &dec.blocks {
            let attn_ln_w_off = plan_vec_f32(&mut cur_off, db.attn_ln_w.len());
            let attn_ln_b_off = plan_vec_f32(&mut cur_off, db.attn_ln_b.len());
            let attn_q_w = plan_linear(&mut cur_off, &db.attn_q_w)?;
            let attn_q_b_off = db
                .attn_q_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let attn_k_w = plan_linear(&mut cur_off, &db.attn_k_w)?;
            let attn_k_b_off = db
                .attn_k_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let attn_v_w = plan_linear(&mut cur_off, &db.attn_v_w)?;
            let attn_v_b_off = db
                .attn_v_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let attn_out_w = plan_linear(&mut cur_off, &db.attn_out_w)?;
            let attn_out_b_off = db
                .attn_out_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));

            let cross_attn_ln_w_off = plan_vec_f32(&mut cur_off, db.cross_attn_ln_w.len());
            let cross_attn_ln_b_off = plan_vec_f32(&mut cur_off, db.cross_attn_ln_b.len());
            let cross_attn_q_w = plan_linear(&mut cur_off, &db.cross_attn_q_w)?;
            let cross_attn_q_b_off = db
                .cross_attn_q_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let cross_attn_k_w = plan_linear(&mut cur_off, &db.cross_attn_k_w)?;
            let cross_attn_k_b_off = db
                .cross_attn_k_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let cross_attn_v_w = plan_linear(&mut cur_off, &db.cross_attn_v_w)?;
            let cross_attn_v_b_off = db
                .cross_attn_v_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let cross_attn_out_w = plan_linear(&mut cur_off, &db.cross_attn_out_w)?;
            let cross_attn_out_b_off = db
                .cross_attn_out_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));

            let mlp_ln_w_off = plan_vec_f32(&mut cur_off, db.mlp_ln_w.len());
            let mlp_ln_b_off = plan_vec_f32(&mut cur_off, db.mlp_ln_b.len());
            let mlp_0_w = plan_linear(&mut cur_off, &db.mlp_0_w)?;
            let mlp_0_b_off = db
                .mlp_0_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));
            let mlp_2_w = plan_linear(&mut cur_off, &db.mlp_2_w)?;
            let mlp_2_b_off = db
                .mlp_2_b
                .as_ref()
                .map(|b| plan_vec_f32(&mut cur_off, b.len()));

            decoder_blocks.push(HexagonWhisperDecoderBlockOffsets {
                attn_ln_w_off,
                attn_ln_b_off,
                attn_q_w,
                attn_q_b_off,
                attn_k_w,
                attn_k_b_off,
                attn_v_w,
                attn_v_b_off,
                attn_out_w,
                attn_out_b_off,
                cross_attn_ln_w_off,
                cross_attn_ln_b_off,
                cross_attn_q_w,
                cross_attn_q_b_off,
                cross_attn_k_w,
                cross_attn_k_b_off,
                cross_attn_v_w,
                cross_attn_v_b_off,
                cross_attn_out_w,
                cross_attn_out_b_off,
                mlp_ln_w_off,
                mlp_ln_b_off,
                mlp_0_w,
                mlp_0_b_off,
                mlp_2_w,
                mlp_2_b_off,
            });
        }

        let decoder_ln_post_w_off = plan_vec_f32(&mut cur_off, dec.ln_post_w.len());
        let decoder_ln_post_b_off = plan_vec_f32(&mut cur_off, dec.ln_post_b.len());
        let proj_w = plan_linear(&mut cur_off, &dec.proj_w)?;

        Ok(Self {
            conv1,
            conv2,
            encoder_pos_embed_off,
            encoder_blocks,
            encoder_ln_post_w_off,
            encoder_ln_post_b_off,
            decoder_blocks,
            decoder_ln_post_w_off,
            decoder_ln_post_b_off,
            proj_w,
            total_bytes: cur_off,
        })
    }
}

/// Stage Whisper model weights into shared rpcmem DMA buffer.
pub fn stage_whisper_weights(
    weights: &WhisperWeights,
    offsets: &HexagonWhisperWeightOffsets,
    dst: &mut [u8],
) -> Result<(), CeraError> {
    if dst.len() < offsets.total_bytes {
        return Err(CeraError::Backend(format!(
            "destination buffer length {} smaller than required offsets total_bytes {}",
            dst.len(),
            offsets.total_bytes
        )));
    }

    let copy_vec_f32 = |dst: &mut [u8], off: usize, src: &[f32]| {
        let bytes = bytemuck::cast_slice(src);
        dst[off..off + bytes.len()].copy_from_slice(bytes);
    };

    let copy_linear =
        |dst: &mut [u8], desc: HexagonWhisperWeightDesc, w: &MmapWeight| -> Result<(), CeraError> {
            let dst_slice = &mut dst[desc.offset..desc.offset + desc.size_bytes];
            match desc.format {
                HexagonWeightFormat::RepackedQ8_0 => match w.dtype {
                    DType::Q8_0 => {
                        repack_q8_0(w.data(), w.cols, w.rows, dst_slice).map_err(|e| {
                            CeraError::Backend(format!("whisper repack Q8_0 failed: {e}"))
                        })?;
                    }
                    DType::F16 | DType::F32 => {
                        let mut f32_vals = vec![0.0f32; w.rows * w.cols];
                        for r in 0..w.rows {
                            w.dequantize_row(r, &mut f32_vals[r * w.cols..(r + 1) * w.cols]);
                        }
                        let q8_bytes = quantize_f32_to_q8_0(&f32_vals, w.cols, w.rows)?;
                        repack_q8_0(&q8_bytes, w.cols, w.rows, dst_slice).map_err(|e| {
                            CeraError::Backend(format!("whisper repack quantized Q8_0 failed: {e}"))
                        })?;
                    }
                    other => {
                        return Err(CeraError::Backend(format!(
                            "unexpected linear weight dtype {other:?}"
                        )));
                    }
                },
                HexagonWeightFormat::RepackedQ4_0 => {
                    repack_q4_0(w.data(), w.cols, w.rows, dst_slice).map_err(|e| {
                        CeraError::Backend(format!("whisper repack Q4_0 failed: {e}"))
                    })?;
                }
            }
            Ok(())
        };

    // Stage Conv1 & Conv2
    stage_conv_taps(dst, &weights.encoder.conv1, &offsets.conv1)?;
    stage_conv_taps(dst, &weights.encoder.conv2, &offsets.conv2)?;

    // Positional embeddings
    copy_vec_f32(
        dst,
        offsets.encoder_pos_embed_off,
        &weights.encoder.positional_embedding,
    );

    // Encoder blocks
    for (eb, blk_offs) in weights.encoder.blocks.iter().zip(&offsets.encoder_blocks) {
        copy_vec_f32(dst, blk_offs.attn_ln_w_off, &eb.attn_ln_w);
        copy_vec_f32(dst, blk_offs.attn_ln_b_off, &eb.attn_ln_b);
        copy_linear(dst, blk_offs.q_w, &eb.attn_q_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.q_b_off, &eb.attn_q_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.k_w, &eb.attn_k_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.k_b_off, &eb.attn_k_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.v_w, &eb.attn_v_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.v_b_off, &eb.attn_v_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.o_w, &eb.attn_out_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.o_b_off, &eb.attn_out_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_vec_f32(dst, blk_offs.mlp_ln_w_off, &eb.mlp_ln_w);
        copy_vec_f32(dst, blk_offs.mlp_ln_b_off, &eb.mlp_ln_b);
        copy_linear(dst, blk_offs.mlp_0_w, &eb.mlp_0_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.mlp_0_b_off, &eb.mlp_0_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.mlp_2_w, &eb.mlp_2_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.mlp_2_b_off, &eb.mlp_2_b) {
            copy_vec_f32(dst, b_off, b);
        }
    }

    copy_vec_f32(
        dst,
        offsets.encoder_ln_post_w_off,
        &weights.encoder.ln_post_w,
    );
    copy_vec_f32(
        dst,
        offsets.encoder_ln_post_b_off,
        &weights.encoder.ln_post_b,
    );

    // Decoder blocks
    for (db, blk_offs) in weights.decoder.blocks.iter().zip(&offsets.decoder_blocks) {
        copy_vec_f32(dst, blk_offs.attn_ln_w_off, &db.attn_ln_w);
        copy_vec_f32(dst, blk_offs.attn_ln_b_off, &db.attn_ln_b);
        copy_linear(dst, blk_offs.attn_q_w, &db.attn_q_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.attn_q_b_off, &db.attn_q_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.attn_k_w, &db.attn_k_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.attn_k_b_off, &db.attn_k_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.attn_v_w, &db.attn_v_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.attn_v_b_off, &db.attn_v_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.attn_out_w, &db.attn_out_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.attn_out_b_off, &db.attn_out_b) {
            copy_vec_f32(dst, b_off, b);
        }

        copy_vec_f32(dst, blk_offs.cross_attn_ln_w_off, &db.cross_attn_ln_w);
        copy_vec_f32(dst, blk_offs.cross_attn_ln_b_off, &db.cross_attn_ln_b);
        copy_linear(dst, blk_offs.cross_attn_q_w, &db.cross_attn_q_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.cross_attn_q_b_off, &db.cross_attn_q_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.cross_attn_k_w, &db.cross_attn_k_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.cross_attn_k_b_off, &db.cross_attn_k_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.cross_attn_v_w, &db.cross_attn_v_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.cross_attn_v_b_off, &db.cross_attn_v_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.cross_attn_out_w, &db.cross_attn_out_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.cross_attn_out_b_off, &db.cross_attn_out_b) {
            copy_vec_f32(dst, b_off, b);
        }

        copy_vec_f32(dst, blk_offs.mlp_ln_w_off, &db.mlp_ln_w);
        copy_vec_f32(dst, blk_offs.mlp_ln_b_off, &db.mlp_ln_b);
        copy_linear(dst, blk_offs.mlp_0_w, &db.mlp_0_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.mlp_0_b_off, &db.mlp_0_b) {
            copy_vec_f32(dst, b_off, b);
        }
        copy_linear(dst, blk_offs.mlp_2_w, &db.mlp_2_w)?;
        if let (Some(b_off), Some(b)) = (blk_offs.mlp_2_b_off, &db.mlp_2_b) {
            copy_vec_f32(dst, b_off, b);
        }
    }

    copy_vec_f32(
        dst,
        offsets.decoder_ln_post_w_off,
        &weights.decoder.ln_post_w,
    );
    copy_vec_f32(
        dst,
        offsets.decoder_ln_post_b_off,
        &weights.decoder.ln_post_b,
    );

    copy_linear(dst, offsets.proj_w, &weights.decoder.proj_w)?;

    Ok(())
}

/// State buffer layout offsets for encoder hidden output and KV caches in `state_buf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonWhisperStateOffsets {
    pub encoder_hidden_off: usize,
    pub cross_kv: Vec<(usize, usize)>,
    pub self_kv: Vec<(usize, usize)>,
    pub total_bytes: usize,
}

impl HexagonWhisperStateOffsets {
    pub fn plan(cfg: &WhisperConfig) -> Self {
        let mut cur_off = 0;

        // Encoder hidden state: [n_audio_ctx (1500) * d_model] in F32
        let encoder_hidden_off = plan_vec_f32(&mut cur_off, cfg.n_audio_ctx * cfg.n_audio_embd);

        // Precomputed static Cross-Attention KV caches (F16): [1500 * d_model] each
        let mut cross_kv = Vec::with_capacity(cfg.n_text_layer);
        for _ in 0..cfg.n_text_layer {
            let k_off = plan_vec_f16(&mut cur_off, cfg.n_audio_ctx * cfg.n_text_embd);
            let v_off = plan_vec_f16(&mut cur_off, cfg.n_audio_ctx * cfg.n_text_embd);
            cross_kv.push((k_off, v_off));
        }

        // Rolling Decoder Self-Attention KV caches (F16): [n_text_ctx (448) * d_model] each
        let mut self_kv = Vec::with_capacity(cfg.n_text_layer);
        for _ in 0..cfg.n_text_layer {
            let k_off = plan_vec_f16(&mut cur_off, cfg.n_text_ctx * cfg.n_text_embd);
            let v_off = plan_vec_f16(&mut cur_off, cfg.n_text_ctx * cfg.n_text_embd);
            self_kv.push((k_off, v_off));
        }

        Self {
            encoder_hidden_off,
            cross_kv,
            self_kv,
            total_bytes: cur_off,
        }
    }
}

/// Scratch activation buffer offsets in `scratch_buf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonWhisperScratchOffsets {
    pub mel_in_off: usize,
    pub conv1_out_off: usize,
    pub conv_tmp_off: usize,
    pub enc_x_off: usize,
    pub enc_norm_off: usize,
    pub enc_q_off: usize,
    pub enc_k_f16_off: usize,
    pub enc_v_f16_off: usize,
    pub enc_attn_out_off: usize,
    pub enc_mlp_mid_off: usize,
    pub enc_mlp_out_off: usize,
    pub dec_x_off: usize,
    pub dec_norm_off: usize,
    pub dec_q_off: usize,
    pub dec_k_off: usize,
    pub dec_v_off: usize,
    pub dec_pos_off: usize,
    pub dec_attn_out_off: usize,
    pub dec_cross_q_off: usize,
    pub dec_cross_out_off: usize,
    pub dec_mlp_mid_off: usize,
    pub dec_mlp_out_off: usize,
    pub logits_off: usize,
    /// `[n_vocab]` F32 added to the logits before the on-DSP argmax: `-1e30` for the tokens a
    /// greedy step must not pick (see `decode_step_greedy`), 0 elsewhere.
    pub greedy_mask_off: usize,
    /// The on-DSP argmax result, one I32.
    pub argmax_off: usize,
    /// Scratch the tanh GELU works through, `gelu_tmp_rows` rows of the widest
    /// MLP activation.
    pub gelu_tmp_off: usize,
    pub gelu_tmp_rows: usize,
    pub total_bytes: usize,
}

impl HexagonWhisperScratchOffsets {
    pub fn plan(cfg: &WhisperConfig) -> Self {
        let mut cur_off = 0;
        let padded_mels = cfg.n_audio_mel_bins.next_multiple_of(32);
        let d_model = cfg.n_audio_embd;
        let d_text = cfg.n_text_embd;
        let max_mlp_mid = 4 * d_model.max(d_text);

        // Padded log-mel input: 3002 rows of size padded_mels (row 0 and 3001 are 0.0)
        let mel_in_off = plan_vec_f32(&mut cur_off, 3002 * padded_mels);

        // Conv1 output: 3002 rows of size d_model (row 0 and 3001 are 0.0 for Conv2 padding)
        let conv1_out_off = plan_vec_f32(&mut cur_off, 3002 * d_model);

        // Temporary scratch for Conv1 / Conv2 tap summation: 3000 rows of size d_model
        let conv_tmp_off = plan_vec_f32(&mut cur_off, 3000 * d_model);

        // Encoder residual stream x: 1500 rows of size d_model
        let enc_x_off = plan_vec_f32(&mut cur_off, 1500 * d_model);

        // Encoder LayerNorm normalized output: 1500 rows of size d_model
        let enc_norm_off = plan_vec_f32(&mut cur_off, 1500 * d_model);

        // Encoder Q: 1500 rows of size d_model
        let enc_q_off = plan_vec_f32(&mut cur_off, 1500 * d_model);

        // Encoder K & V in F16 format for FlashAttnExt: 1500 rows of size d_model
        let enc_k_f16_off = plan_vec_f16(&mut cur_off, 1500 * d_model);
        let enc_v_f16_off = plan_vec_f16(&mut cur_off, 1500 * d_model);

        // Encoder attention out: 1500 rows of size d_model
        let enc_attn_out_off = plan_vec_f32(&mut cur_off, 1500 * d_model);

        // Encoder MLP intermediate activations: 1500 rows of size max_mlp_mid
        let enc_mlp_mid_off = plan_vec_f32(&mut cur_off, 1500 * max_mlp_mid);

        // Encoder MLP output: 1500 rows of size d_model
        let enc_mlp_out_off = plan_vec_f32(&mut cur_off, 1500 * d_model);

        // Decoder activations (1 token decode)
        let dec_x_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_norm_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_q_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_k_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_v_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_pos_off = cur_off;
        cur_off += align128(4); // I32 position
        let dec_attn_out_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_cross_q_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_cross_out_off = plan_vec_f32(&mut cur_off, d_text);
        let dec_mlp_mid_off = plan_vec_f32(&mut cur_off, max_mlp_mid);
        let dec_mlp_out_off = plan_vec_f32(&mut cur_off, d_text);

        // Decoder logits [n_vocab]
        let logits_off = plan_vec_f32(&mut cur_off, cfg.n_vocab);
        let greedy_mask_off = plan_vec_f32(&mut cur_off, cfg.n_vocab);
        let argmax_off = plan_vec_f32(&mut cur_off, 1);

        // The tanh GELU's intermediate: a few rows of the widest activation.
        let gelu_tmp_rows = dispatch::GELU_TMP_ROWS;
        let gelu_tmp_off = plan_vec_f32(&mut cur_off, gelu_tmp_rows * max_mlp_mid);

        Self {
            mel_in_off,
            conv1_out_off,
            conv_tmp_off,
            enc_x_off,
            enc_norm_off,
            enc_q_off,
            enc_k_f16_off,
            enc_v_f16_off,
            enc_attn_out_off,
            enc_mlp_mid_off,
            enc_mlp_out_off,
            dec_x_off,
            dec_norm_off,
            dec_q_off,
            dec_k_off,
            dec_v_off,
            dec_pos_off,
            dec_attn_out_off,
            dec_cross_q_off,
            dec_cross_out_off,
            dec_mlp_mid_off,
            dec_mlp_out_off,
            logits_off,
            greedy_mask_off,
            argmax_off,
            gelu_tmp_off,
            gelu_tmp_rows,
            total_bytes: cur_off,
        }
    }
}

/// The logit added to a token a greedy step must not pick (finite, so the DSP never sees an
/// infinity; far below any real logit).
const GREEDY_SUPPRESSED: f32 = -1e30;

/// The little-endian `i32` at `at` in a DSP readback. Fallible indexing: on a short
/// scratch (driver fault, planning bug) this returns a recoverable error instead of
/// panicking the phone.
fn readback_i32_le(scratch: &[u8], at: usize) -> Result<i32, CeraError> {
    let Some(&bytes) = scratch.get(at..).and_then(|tail| tail.first_chunk::<4>()) else {
        return Err(CeraError::Backend(format!(
            "the DSP argmax read back a short row (offset {at} + 4 over scratch len {})",
            scratch.len()
        )));
    };
    Ok(i32::from_le_bytes(bytes))
}

/// The additive logit mask of a greedy step: `GREEDY_SUPPRESSED` for the tokens the host's
/// `suppress_whisper_special_tokens` sets to `-inf`, 0 for the rest. Derived from the host
/// function itself, so the two cannot drift apart.
fn greedy_mask(t: &WhisperSpecialTokens, n_vocab: usize, timestamps: bool) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_vocab];
    crate::model::whisper::suppress_whisper_special_tokens(&mut mask, t, timestamps);
    for v in &mut mask {
        if *v == f32::NEG_INFINITY {
            *v = GREEDY_SUPPRESSED;
        }
    }
    mask
}

/// What a decoder step hands back.
enum StepOut<'a> {
    /// The full logits row, copied to the caller's buffer.
    Logits(&'a mut [f32]),
    /// The next token, picked on the DSP.
    Greedy { timestamps: bool },
}

/// Native Qualcomm Hexagon NPU Whisper speech recognition model.
pub struct HexagonWhisperModel {
    device: Arc<Mutex<HexagonDevice>>,
    config: WhisperConfig,
    weights_offsets: HexagonWhisperWeightOffsets,
    weights_buf: RpcmemBuffer,
    state_offsets: HexagonWhisperStateOffsets,
    state_buf: Mutex<RpcmemBuffer>,
    scratch_offsets: HexagonWhisperScratchOffsets,
    scratch_buf: Mutex<RpcmemBuffer>,
    special_tokens: WhisperSpecialTokens,
    token_embeddings: MmapWeight,
    positional_embedding: Vec<f32>,
    session_lock: Mutex<()>,
    /// The queue session's wait mode before [`Self::new`] put it to sleep, restored by [`Drop`].
    blocking_wait_prev: bool,
    /// The `timestamps` setting the greedy suppression mask in scratch was staged for.
    greedy_mask_for: Mutex<Option<bool>>,
    /// The log-mel front end on the DSP; `None` runs it on the host.
    mel: Option<crate::model::whisper_mel_hexagon::WhisperMelDsp>,
    /// How often [`Self::log_mel`] fell back to the host after a DSP failure.
    mel_fallbacks: AtomicU64,
    /// True once `encode_audio` has fully rewritten `state_buf`.
    encoded_ok: AtomicBool,
}

impl Drop for HexagonWhisperModel {
    /// Let the DSP let go of every buffer before the host unmaps them.
    fn drop(&mut self) {
        let mut device = self.device.lock_or_recover();
        let state = self.state_buf.lock_or_recover();
        let scratch = self.scratch_buf.lock_or_recover();
        let mel = self.mel.as_ref().map(|m| m.buffers());
        let mut held = vec![&self.weights_buf, &*state, &*scratch];
        if let Some((weights, mel_scratch)) = &mel {
            held.push(weights);
            held.push(mel_scratch);
        }
        device.queue_session_mut().release_dsp_references(held);
        // The release above is the model's last wait: leave the session as it was found.
        device
            .queue_session_mut()
            .set_blocking_wait(self.blocking_wait_prev);
    }
}

unsafe impl Send for HexagonWhisperModel {}
unsafe impl Sync for HexagonWhisperModel {}

impl std::fmt::Debug for HexagonWhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HexagonWhisperModel")
            .field("config", &self.config)
            .finish()
    }
}

impl HexagonWhisperModel {
    /// Initialize a Hexagon Whisper model by allocating and staging shared DMA memory buffers.
    ///
    /// Puts `device`'s queue-session waits to sleep in the kernel from here on (restored on
    /// drop), so pass a device this model does not share (as [`init_hexagon_whisper`] does).
    /// The driver's default wait spins
    /// a core for as long as the DSP runs: an encoder window is about 3,800 ops in five batches
    /// and 340 ms of DSP time, and with the default wait the NPU path cost the CPU as much time
    /// as the DSP took (336 ms per utterance on an S25 Ultra, 2 ms asleep), which is what moving
    /// Whisper to the NPU for a background service is meant to avoid. The queue also flushes by
    /// itself whenever a batch fills, so the setting has to hold for the whole call, not just the
    /// final flush. A sleeping wake-up costs about a scheduler tick per batch.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        weights: &WhisperWeights,
        tokenizer: &crate::tokenizer::BpeTokenizer,
    ) -> Result<Self, CeraError> {
        let weights_offsets = HexagonWhisperWeightOffsets::plan(weights)?;
        let mut weights_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), weights_offsets.total_bytes, true)?;
        stage_whisper_weights(weights, &weights_offsets, weights_buf.as_mut_slice())?;
        weights_buf.flush_cpu_cache(0, weights_offsets.total_bytes);

        let state_offsets = HexagonWhisperStateOffsets::plan(&weights.config);
        let state_buf = RpcmemBuffer::alloc(Arc::clone(&driver), state_offsets.total_bytes, true)?;

        let scratch_offsets = HexagonWhisperScratchOffsets::plan(&weights.config);
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true)?;

        let special_tokens = WhisperSpecialTokens::from_tokenizer(tokenizer);
        // The 80- and 128-bin windows are the ones the host front end supports too.
        let mel = if matches!(weights.config.n_audio_mel_bins, 80 | 128) {
            crate::model::whisper_mel_hexagon::WhisperMelDsp::new(
                Arc::clone(&driver),
                weights.config.n_audio_mel_bins,
            )
            .inspect_err(|e| hexagon_warn!("whisper: log-mel stays on the host ({e})"))
            .ok()
        } else {
            None
        };
        let blocking_wait_prev = device
            .lock_or_recover()
            .queue_session_mut()
            .set_blocking_wait(true);

        Ok(Self {
            device,
            config: weights.config.clone(),
            weights_offsets,
            weights_buf,
            state_offsets,
            state_buf: Mutex::new(state_buf),
            scratch_offsets,
            scratch_buf: Mutex::new(scratch_buf),
            special_tokens,
            token_embeddings: weights.decoder.token_embeddings.clone(),
            positional_embedding: weights.decoder.positional_embedding.clone(),
            session_lock: Mutex::new(()),
            blocking_wait_prev,
            greedy_mask_for: Mutex::new(None),
            mel,
            mel_fallbacks: AtomicU64::new(0),
            encoded_ok: AtomicBool::new(false),
        })
    }

    /// Dispatch 1D convolution layer with 3-tap decomposition: `Y = W0·X0 + W1·X1 + W2·X2 + bias`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_conv1d_3tap(
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        w_conv: &HexagonWhisperConvWeights,
        src: &RpcmemBuffer,
        src_base_offset: usize,
        padded_in_channels: usize,
        stride: usize,
        n_tokens: usize,
        out_channels: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        tmp: &RpcmemBuffer,
        tmp_offset: usize,
    ) -> Result<(), CeraError> {
        let (w_dtype, block_bytes, tile_size) = match w_conv.w0.format {
            HexagonWeightFormat::RepackedQ8_0 => (HtpDataType::Q8_0, 34usize, 32 * 34usize),
            HexagonWeightFormat::RepackedQ4_0 => (HtpDataType::Q4_0, 18usize, 32 * 18usize),
        };

        let ne0 = w_conv.w0.cols;
        let ne1 = w_conv.w0.rows;
        let tiled_row_bytes = ne0.div_ceil(32) * tile_size;
        let w_tot = ne1.div_ceil(32) * tiled_row_bytes;

        let w0_ti = session.add_tensor(
            weights,
            w_conv.w0.offset,
            w_conv.w0.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                block_bytes as u32,
                tiled_row_bytes as u32,
                w_tot as u32,
                w_tot as u32,
            ],
        )?;
        let w1_ti = session.add_tensor(
            weights,
            w_conv.w1.offset,
            w_conv.w1.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                block_bytes as u32,
                tiled_row_bytes as u32,
                w_tot as u32,
                w_tot as u32,
            ],
        )?;
        let w2_ti = session.add_tensor(
            weights,
            w_conv.w2.offset,
            w_conv.w2.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                block_bytes as u32,
                tiled_row_bytes as u32,
                w_tot as u32,
                w_tot as u32,
            ],
        )?;

        let b_bytes = out_channels * 4;
        let b_ti = session.add_tensor(
            weights,
            w_conv.bias_off,
            b_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [out_channels as u32, 1, 1, 1],
            [4, b_bytes as u32, b_bytes as u32, b_bytes as u32],
        )?;

        let row_bytes = padded_in_channels * 4;
        let stride_bytes = (stride * row_bytes) as u32;
        let chunk_size = 64;

        let mut t_start = 0;
        while t_start < n_tokens {
            let chunk = (n_tokens - t_start).min(chunk_size);
            let cur_x_bytes = chunk * stride * row_bytes;
            let cur_dst_off = dst_offset + t_start * out_channels * 4;
            let cur_dst_bytes = chunk * out_channels * 4;

            let cur_dst_ti = session.add_tensor(
                dst,
                cur_dst_off,
                cur_dst_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [out_channels as u32, chunk as u32, 1, 1],
                [
                    4,
                    (out_channels * 4) as u32,
                    cur_dst_bytes as u32,
                    cur_dst_bytes as u32,
                ],
            )?;

            let tmp_ti = session.add_tensor(
                tmp,
                tmp_offset,
                cur_dst_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [out_channels as u32, chunk as u32, 1, 1],
                [
                    4,
                    (out_channels * 4) as u32,
                    cur_dst_bytes as u32,
                    cur_dst_bytes as u32,
                ],
            )?;

            let kparams_mm = build_mul_mat_kernel_params(
                w_dtype,
                padded_in_channels,
                chunk as u32,
                1,
                out_channels * 4,
                session.dsp_threads(),
                dispatch::VTCM_BUDGET,
            );

            // Tap 0 -> cur_dst
            let x0_off = conv_tap_input_offset(src_base_offset, row_bytes, 0, t_start, stride);
            let x0_ti = session.add_tensor(
                src,
                x0_off,
                cur_x_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [padded_in_channels as u32, chunk as u32, 1, 1],
                [4, stride_bytes, cur_x_bytes as u32, cur_x_bytes as u32],
            )?;
            session.enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w0_ti, x0_ti],
                &[cur_dst_ti],
                [0i32; 16],
                kparams_mm,
            )?;

            // Tap 1 -> tmp; cur_dst += tmp
            let x1_off = conv_tap_input_offset(src_base_offset, row_bytes, 1, t_start, stride);
            let x1_ti = session.add_tensor(
                src,
                x1_off,
                cur_x_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [padded_in_channels as u32, chunk as u32, 1, 1],
                [4, stride_bytes, cur_x_bytes as u32, cur_x_bytes as u32],
            )?;
            session.enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w1_ti, x1_ti],
                &[tmp_ti],
                [0i32; 16],
                kparams_mm,
            )?;

            let add_kparams = build_binary_kernel_params(
                out_channels,
                out_channels,
                chunk,
                1,
                1,
                4,
                dispatch::VTCM_BUDGET,
                session.dsp_threads(),
            );
            session.enqueue_op(
                HtpOpCode::Add as u32,
                &[cur_dst_ti, tmp_ti],
                &[cur_dst_ti],
                [0i32; 16],
                add_kparams,
            )?;

            // Tap 2 -> tmp; cur_dst += tmp
            let x2_off = conv_tap_input_offset(src_base_offset, row_bytes, 2, t_start, stride);
            let x2_ti = session.add_tensor(
                src,
                x2_off,
                cur_x_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [padded_in_channels as u32, chunk as u32, 1, 1],
                [4, stride_bytes, cur_x_bytes as u32, cur_x_bytes as u32],
            )?;
            session.enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w2_ti, x2_ti],
                &[tmp_ti],
                [0i32; 16],
                kparams_mm,
            )?;
            session.enqueue_op(
                HtpOpCode::Add as u32,
                &[cur_dst_ti, tmp_ti],
                &[cur_dst_ti],
                [0i32; 16],
                add_kparams,
            )?;

            // Add bias: cur_dst += bias
            let bias_add_kparams = build_binary_kernel_params(
                out_channels,
                out_channels,
                1,
                1,
                1,
                4,
                dispatch::VTCM_BUDGET,
                session.dsp_threads(),
            );
            session.enqueue_op(
                HtpOpCode::Add as u32,
                &[cur_dst_ti, b_ti],
                &[cur_dst_ti],
                [0i32; 16],
                bias_add_kparams,
            )?;

            t_start += chunk;
        }
        session.end_group()?;

        // GELU over the whole output, as a pass of its own (the taps' tensors
        // are done with, and a helper that ends its own group must not run
        // between ops that share tensor indices). `tmp` is free again and holds
        // `chunk_size` rows.
        dispatch::gelu_tanh(
            session,
            dst,
            dst_offset,
            tmp,
            tmp_offset,
            chunk_size,
            TokenShape {
                dim: out_channels,
                n_tokens,
            },
        )
    }

    /// Dispatch unmasked multi-head self-attention on DSP via FlashAttnExt.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_self_attention_bidirectional(
        session: &mut HexagonQueueSession,
        q: &RpcmemBuffer,
        q_offset: usize,
        k: &RpcmemBuffer,
        k_offset: usize,
        v: &RpcmemBuffer,
        v_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let q_bytes = q_dim * n_tokens * 4;
        let kv_bytes = q_dim * n_tokens * 2;

        let q_ti = session.add_tensor(
            q,
            q_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_tokens as u32, n_heads as u32, 1],
            [4, (q_dim * 4) as u32, (head_dim * 4) as u32, q_bytes as u32],
        )?;
        let k_ti = session.add_tensor(
            k,
            k_offset,
            kv_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, n_tokens as u32, n_heads as u32, 1],
            [
                2,
                (q_dim * 2) as u32,
                (head_dim * 2) as u32,
                kv_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v,
            v_offset,
            kv_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, n_tokens as u32, n_heads as u32, 1],
            [
                2,
                (q_dim * 2) as u32,
                (head_dim * 2) as u32,
                kv_bytes as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, n_tokens as u32, 1],
            [4, (head_dim * 4) as u32, (q_dim * 4) as u32, q_bytes as u32],
        )?;

        let mut params = [0i32; 16];
        params[0] = scale.to_bits() as i32;

        let kparams = build_flash_attn_kernel_params(
            head_dim,
            n_heads,
            n_heads,
            n_tokens,
            n_tokens,
            scale,
            session.dsp_threads(),
            false,
        );

        session.enqueue_op(
            HtpOpCode::FlashAttnExt as u32,
            &[q_ti, k_ti, v_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        session.end_group()
    }

    /// Dispatch SetRows KV cache insertion: appends 1 row of K/V into F16 cache at `pos`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_set_rows(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        cache: &RpcmemBuffer,
        cache_offset: usize,
        kv_dim: usize,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        let src_bytes = kv_dim * 4;
        let cache_bytes = kv_dim * max_seq_len * 2;
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [kv_dim as u32, 1, 1, 1],
            [4, (kv_dim * 4) as u32, src_bytes as u32, src_bytes as u32],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [1, 1, 1, 1],
            [4, 4, 4, 4],
        )?;
        let cache_ti = session.add_tensor(
            cache,
            cache_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                2,
                (kv_dim * 2) as u32,
                cache_bytes as u32,
                cache_bytes as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = crate::backend::hexagon::params::build_set_rows_kernel_params(
            1,
            1,
            1,
            1,
            kv_dim,
            true,
            session.dsp_threads(),
        );
        session.enqueue_op(
            HtpOpCode::SetRows as u32,
            &[src_ti, pos_ti],
            &[cache_ti],
            params,
            kparams,
        )?;
        session.end_group()
    }

    /// Dispatch single-token decode FlashAttnExt over rolling or static KV cache.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_flash_attn_decode(
        session: &mut HexagonQueueSession,
        q: &RpcmemBuffer,
        q_offset: usize,
        k: &RpcmemBuffer,
        k_offset: usize,
        v: &RpcmemBuffer,
        v_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        head_dim: usize,
        n_heads: usize,
        seq_len: usize,
        max_seq_len: usize,
        scale: f32,
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let q_bytes = q_dim * 4;
        let kv_bytes = q_dim * max_seq_len * 2;

        let q_ti = session.add_tensor(
            q,
            q_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, n_heads as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                q_bytes as u32,
            ],
        )?;
        let k_ti = session.add_tensor(
            k,
            k_offset,
            kv_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_heads as u32, 1],
            [
                2,
                (q_dim * 2) as u32,
                (head_dim * 2) as u32,
                kv_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v,
            v_offset,
            kv_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_heads as u32, 1],
            [
                2,
                (q_dim * 2) as u32,
                (head_dim * 2) as u32,
                kv_bytes as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, n_heads as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                q_bytes as u32,
            ],
        )?;

        let mut params = [0i32; 16];
        params[0] = scale.to_bits() as i32;

        let kparams = build_flash_attn_kernel_params(
            head_dim,
            n_heads,
            n_heads,
            1,
            seq_len,
            scale,
            session.dsp_threads(),
            false,
        );

        session.enqueue_op(
            HtpOpCode::FlashAttnExt as u32,
            &[q_ti, k_ti, v_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        session.end_group()
    }

    /// Execute audio encoder and precompute static cross-attention on Qualcomm Hexagon NPU.
    ///
    /// A failed or panicked encode leaves the model needing a fresh one:
    /// `decode_step` fails closed until this succeeds.
    pub fn encode_audio(&self, mel: &[f32]) -> Result<(), CeraError> {
        let n_mels = self.config.n_audio_mel_bins;
        if mel.len() != n_mels * 3000 {
            return Err(CeraError::Backend(format!(
                "mel spectrogram size mismatch: expected {}, got {}",
                n_mels * 3000,
                mel.len()
            )));
        }

        let padded_mels = n_mels.next_multiple_of(32);
        let d_model = self.config.n_audio_embd;
        let head_dim = self.config.audio_head_dim();
        let n_heads = self.config.n_audio_head;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scratch = self.scratch_buf.lock_or_recover();
        // The whole buffer is rewritten below, so a poisoned lock is recovered
        // (once: the flag is cleared under the lock, so concurrent callers
        // cannot miss it).
        let (state, poisoned) = lock_reporting_poison(&self.state_buf);
        if poisoned {
            report_poison("whisper state", "rewriting it with a fresh encode");
        }
        // `state_buf` is torn from here until the encode completes: an error
        // or panic part-way leaves `encoded_ok` false, so `decode_step` fails
        // closed instead of decoding over half-written K/V. Both stores happen
        // under the state lock, so overlapping encodes cannot leave it true
        // over a half-written buffer.
        self.encoded_ok.store(false, Ordering::SeqCst);

        // 1. Stage mel spectrogram into scratch_buf.mel_in_off (padded from 80 to 96)
        let scratch_slice = scratch.as_mut_slice();
        let mel_base = self.scratch_offsets.mel_in_off;

        // Zero out row 0 and row 3001
        scratch_slice[mel_base..mel_base + padded_mels * 4].fill(0);
        let last_row_off = mel_base + 3001 * padded_mels * 4;
        scratch_slice[last_row_off..last_row_off + padded_mels * 4].fill(0);

        // Copy frames 0..3000 into rows 1..=3000
        for t in 0..3000 {
            let row_off = mel_base + (t + 1) * padded_mels * 4;
            let row_bytes = &mut scratch_slice[row_off..row_off + padded_mels * 4];
            let row_floats: &mut [f32] = bytemuck::cast_slice_mut(row_bytes);
            for m in 0..n_mels {
                row_floats[m] = mel[m * 3000 + t];
            }
            row_floats[n_mels..padded_mels].fill(0.0);
        }

        // Zero out boundary rows for Conv1 output (padding for Conv2)
        let conv1_base = self.scratch_offsets.conv1_out_off;
        scratch_slice[conv1_base..conv1_base + d_model * 4].fill(0);
        let conv1_last_off = conv1_base + 3001 * d_model * 4;
        scratch_slice[conv1_last_off..conv1_last_off + d_model * 4].fill(0);

        scratch.flush_cpu_cache(mel_base, 3002 * padded_mels * 4);
        scratch.flush_cpu_cache(conv1_base, d_model * 4);
        scratch.flush_cpu_cache(conv1_last_off, d_model * 4);

        // 2. Build DSP command queue
        let mut dev_guard = self.device.lock_or_recover();
        let session = dev_guard.queue_session_mut();
        session.drop_pending_batch();
        // A cap on tensors per batch ends a batch at an op-group boundary, so
        // another model's batch can run between two of this encoder's.
        session.set_max_tensors_per_flush(encoder_flush_cap());

        // Conv1: mel_in -> conv1_out (rows 1..=3000)
        let conv1_dst_off = conv1_base + d_model * 4;
        Self::dispatch_conv1d_3tap(
            session,
            &self.weights_buf,
            &self.weights_offsets.conv1,
            &scratch,
            mel_base,
            padded_mels,
            1,
            3000,
            d_model,
            &scratch,
            conv1_dst_off,
            &scratch,
            self.scratch_offsets.conv_tmp_off,
        )?;

        // Conv2: conv1_out -> enc_x (rows 0..1500)
        let enc_x_off = self.scratch_offsets.enc_x_off;
        Self::dispatch_conv1d_3tap(
            session,
            &self.weights_buf,
            &self.weights_offsets.conv2,
            &scratch,
            conv1_base,
            d_model,
            2,
            1500,
            d_model,
            &scratch,
            enc_x_off,
            &scratch,
            self.scratch_offsets.conv_tmp_off,
        )?;

        // Residual initialization: enc_x += positional_embedding
        let pos_off = self.weights_offsets.encoder_pos_embed_off;

        // Add positional embedding
        dispatch::add_residual(
            session,
            &scratch,
            enc_x_off,
            &self.weights_buf,
            pos_off,
            TokenShape {
                dim: d_model,
                n_tokens: 1500,
            },
            WHISPER_TILE,
        )?;

        // 3. Encoder Transformer Blocks
        let norm_off = self.scratch_offsets.enc_norm_off;
        let q_off = self.scratch_offsets.enc_q_off;
        let k_f16_off = self.scratch_offsets.enc_k_f16_off;
        let v_f16_off = self.scratch_offsets.enc_v_f16_off;
        let attn_out_off = self.scratch_offsets.enc_attn_out_off;
        let mlp_mid_off = self.scratch_offsets.enc_mlp_mid_off;
        let mlp_out_off = self.scratch_offsets.enc_mlp_out_off;
        let tmp_off = self.scratch_offsets.conv_tmp_off;

        for blk in &self.weights_offsets.encoder_blocks {
            // LayerNorm 1
            dispatch::layer_norm(
                session,
                LayerNormArgs {
                    src: &scratch,
                    src_offset: enc_x_off,
                    dst: &scratch,
                    dst_offset: norm_off,
                    weights: &self.weights_buf,
                    w_offset: blk.attn_ln_w_off,
                    b_offset: blk.attn_ln_b_off,
                    shape: TokenShape {
                        dim: d_model,
                        n_tokens: 1500,
                    },
                    eps: 1e-5,
                    tile: WHISPER_TILE,
                },
            )?;

            // Q, K, V projections
            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.q_w,
                blk.q_b_off,
                &scratch,
                q_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.k_w,
                blk.k_b_off,
                &scratch,
                tmp_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::cpy_f32_to_f16(
                session,
                &scratch,
                tmp_off,
                &scratch,
                k_f16_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
            )?;

            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.v_w,
                blk.v_b_off,
                &scratch,
                tmp_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::cpy_f32_to_f16(
                session,
                &scratch,
                tmp_off,
                &scratch,
                v_f16_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
            )?;

            // Unmasked FlashAttention
            Self::dispatch_self_attention_bidirectional(
                session,
                &scratch,
                q_off,
                &scratch,
                k_f16_off,
                &scratch,
                v_f16_off,
                &scratch,
                attn_out_off,
                head_dim,
                n_heads,
                1500,
                scale,
            )?;

            // Attention out projection + residual add
            dispatch::linear_m(
                session,
                &scratch,
                attn_out_off,
                &self.weights_buf,
                blk.o_w,
                blk.o_b_off,
                &scratch,
                attn_out_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::add_residual(
                session,
                &scratch,
                enc_x_off,
                &scratch,
                attn_out_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
                WHISPER_TILE,
            )?;

            // LayerNorm 2
            dispatch::layer_norm(
                session,
                LayerNormArgs {
                    src: &scratch,
                    src_offset: enc_x_off,
                    dst: &scratch,
                    dst_offset: norm_off,
                    weights: &self.weights_buf,
                    w_offset: blk.mlp_ln_w_off,
                    b_offset: blk.mlp_ln_b_off,
                    shape: TokenShape {
                        dim: d_model,
                        n_tokens: 1500,
                    },
                    eps: 1e-5,
                    tile: WHISPER_TILE,
                },
            )?;

            // MLP: MLP0 -> GELU -> MLP2 + residual add
            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.mlp_0_w,
                blk.mlp_0_b_off,
                &scratch,
                mlp_mid_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::gelu_tanh(
                session,
                &scratch,
                mlp_mid_off,
                &scratch,
                self.scratch_offsets.gelu_tmp_off,
                self.scratch_offsets.gelu_tmp_rows,
                TokenShape {
                    dim: blk.mlp_0_w.rows,
                    n_tokens: 1500,
                },
            )?;
            dispatch::linear_m(
                session,
                &scratch,
                mlp_mid_off,
                &self.weights_buf,
                blk.mlp_2_w,
                blk.mlp_2_b_off,
                &scratch,
                mlp_out_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::add_residual(
                session,
                &scratch,
                enc_x_off,
                &scratch,
                mlp_out_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
                WHISPER_TILE,
            )?;
            // One batch per layer: the tanh GELU is five ops a tile, and a whole
            // deep encoder in one batch would not fit the staging buffer.
            session.flush()?;
        }

        // 4. Post-LayerNorm -> state_buf.encoder_hidden_off
        let enc_hidden_off = self.state_offsets.encoder_hidden_off;
        dispatch::layer_norm(
            session,
            LayerNormArgs {
                src: &scratch,
                src_offset: enc_x_off,
                dst: &state,
                dst_offset: enc_hidden_off,
                weights: &self.weights_buf,
                w_offset: self.weights_offsets.encoder_ln_post_w_off,
                b_offset: self.weights_offsets.encoder_ln_post_b_off,
                shape: TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
                eps: 1e-5,
                tile: WHISPER_TILE,
            },
        )?;

        // 5. Precompute Cross-Attention K & V into state_buf.cross_kv
        for (l, dec_blk) in self.weights_offsets.decoder_blocks.iter().enumerate() {
            let (cross_k_off, cross_v_off) = self.state_offsets.cross_kv[l];

            // Cross-K projection -> F16 Cpy
            dispatch::linear_m(
                session,
                &state,
                enc_hidden_off,
                &self.weights_buf,
                dec_blk.cross_attn_k_w,
                dec_blk.cross_attn_k_b_off,
                &scratch,
                tmp_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::cpy_f32_to_f16(
                session,
                &scratch,
                tmp_off,
                &state,
                cross_k_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
            )?;

            // Cross-V projection -> F16 Cpy
            dispatch::linear_m(
                session,
                &state,
                enc_hidden_off,
                &self.weights_buf,
                dec_blk.cross_attn_v_w,
                dec_blk.cross_attn_v_b_off,
                &scratch,
                tmp_off,
                1500,
                WHISPER_TILE,
            )?;
            dispatch::cpy_f32_to_f16(
                session,
                &scratch,
                tmp_off,
                &state,
                cross_v_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1500,
                },
            )?;
        }

        // Submit DSP batch queue
        session.set_max_tensors_per_flush(None);
        session.flush()?;
        self.encoded_ok.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Execute one autoregressive decoding step on Qualcomm Hexagon NPU.
    ///
    /// # Errors
    ///
    /// `Backend` if `encode_audio` has not completed successfully since the
    /// last failure or panic (the state it fills would be torn).
    pub fn decode_step(
        &self,
        token_id: u32,
        pos: usize,
        logits_out: &mut [f32],
    ) -> Result<(), CeraError> {
        self.decode_step_impl(token_id, pos, StepOut::Logits(logits_out))
            .map(|_| ())
    }

    /// [`Self::decode_step`] that picks the next token on the DSP: the control tokens (and the
    /// timestamp tokens unless `timestamps`) are masked out of the logits and an `Argmax` runs
    /// over them, so the host reads 4 bytes instead of the whole 207 KB row and never scans it.
    /// The same token as suppressing and taking the argmax on the host (greedy decoding,
    /// temperature 0), assuming the DSP `Argmax` breaks exact ties toward the lowest index
    /// like the host's `argmax` and the logits contain no NaN (the on-device probe asserts
    /// the greedy comparison).
    pub fn decode_step_greedy(
        &self,
        token_id: u32,
        pos: usize,
        timestamps: bool,
    ) -> Result<u32, CeraError> {
        self.decode_step_impl(token_id, pos, StepOut::Greedy { timestamps })?
            .ok_or_else(|| CeraError::Backend("greedy step returned no token".into()))
    }

    fn decode_step_impl(
        &self,
        token_id: u32,
        pos: usize,
        mut out: StepOut<'_>,
    ) -> Result<Option<u32>, CeraError> {
        if token_id as usize >= self.config.n_vocab {
            return Err(CeraError::Backend(format!(
                "token_id {token_id} out of bounds for vocab size {}",
                self.config.n_vocab
            )));
        }
        if pos >= self.config.n_text_ctx {
            return Err(CeraError::Backend(format!(
                "pos {pos} exceeds max text context length {}",
                self.config.n_text_ctx
            )));
        }
        if let StepOut::Logits(logits_out) = &out
            && logits_out.len() < self.config.n_vocab
        {
            return Err(CeraError::Backend(format!(
                "logits_out buffer length {} smaller than vocab size {}",
                logits_out.len(),
                self.config.n_vocab
            )));
        }

        let d_model = self.config.n_text_embd;
        let head_dim = self.config.text_head_dim();
        let n_heads = self.config.n_text_head;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scratch = self.scratch_buf.lock_or_recover();
        // `state_buf` carries the encoder output and cross-attention K/V. Refuse
        // to decode over a torn one: a panic here (poison) or an encode that
        // failed part-way (`encoded_ok` false) both need a fresh `encode_audio`.
        let state = self
            .state_buf
            .lock()
            .ok()
            .filter(|_| self.encoded_ok.load(Ordering::SeqCst));
        let Some(state) = state else {
            return Err(CeraError::Backend(
                "Hexagon whisper state is not valid (poisoned or encode incomplete); re-run encode_audio"
                    .into(),
            ));
        };

        if let StepOut::Greedy { timestamps } = &out {
            self.stage_greedy_mask(&mut scratch, *timestamps);
        }
        // 1. Stage input embedding + positional embedding into scratch.dec_x_off
        let scratch_slice = scratch.as_mut_slice();
        let dec_x_off = self.scratch_offsets.dec_x_off;
        let dec_x_slice: &mut [f32] =
            bytemuck::cast_slice_mut(&mut scratch_slice[dec_x_off..dec_x_off + d_model * 4]);

        self.token_embeddings
            .dequantize_row(token_id as usize, dec_x_slice);

        let pos_emb = &self.positional_embedding[pos * d_model..(pos + 1) * d_model];
        for (x, &p) in dec_x_slice.iter_mut().zip(pos_emb.iter()) {
            *x += p;
        }

        // Stage position integer for SetRows
        let pos_off = self.scratch_offsets.dec_pos_off;
        let pos_bytes = (pos as i32).to_le_bytes();
        scratch_slice[pos_off..pos_off + 4].copy_from_slice(&pos_bytes);

        scratch.flush_cpu_cache(dec_x_off, d_model * 4);
        scratch.flush_cpu_cache(pos_off, 4);

        // 2. Build DSP command queue
        let mut dev_guard = self.device.lock_or_recover();
        let session = dev_guard.queue_session_mut();
        session.set_max_tensors_per_flush(None);
        session.drop_pending_batch();

        let norm_off = self.scratch_offsets.dec_norm_off;
        let q_off = self.scratch_offsets.dec_q_off;
        let k_off = self.scratch_offsets.dec_k_off;
        let v_off = self.scratch_offsets.dec_v_off;
        let attn_out_off = self.scratch_offsets.dec_attn_out_off;
        let cross_q_off = self.scratch_offsets.dec_cross_q_off;
        let cross_out_off = self.scratch_offsets.dec_cross_out_off;
        let mlp_mid_off = self.scratch_offsets.dec_mlp_mid_off;
        let mlp_out_off = self.scratch_offsets.dec_mlp_out_off;

        for (l, blk) in self.weights_offsets.decoder_blocks.iter().enumerate() {
            // A. Causal Self-Attention
            dispatch::layer_norm(
                session,
                LayerNormArgs {
                    src: &scratch,
                    src_offset: dec_x_off,
                    dst: &scratch,
                    dst_offset: norm_off,
                    weights: &self.weights_buf,
                    w_offset: blk.attn_ln_w_off,
                    b_offset: blk.attn_ln_b_off,
                    shape: TokenShape {
                        dim: d_model,
                        n_tokens: 1,
                    },
                    eps: 1e-5,
                    tile: WHISPER_TILE,
                },
            )?;

            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_q_w,
                blk.attn_q_b_off,
                &scratch,
                q_off,
                1,
                WHISPER_TILE,
            )?;
            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_k_w,
                blk.attn_k_b_off,
                &scratch,
                k_off,
                1,
                WHISPER_TILE,
            )?;
            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_v_w,
                blk.attn_v_b_off,
                &scratch,
                v_off,
                1,
                WHISPER_TILE,
            )?;

            // SetRows K & V into self_kv cache
            let (self_k_cache_off, self_v_cache_off) = self.state_offsets.self_kv[l];
            Self::dispatch_set_rows(
                session,
                &scratch,
                k_off,
                &scratch,
                pos_off,
                &state,
                self_k_cache_off,
                d_model,
                self.config.n_text_ctx,
            )?;
            Self::dispatch_set_rows(
                session,
                &scratch,
                v_off,
                &scratch,
                pos_off,
                &state,
                self_v_cache_off,
                d_model,
                self.config.n_text_ctx,
            )?;

            // Self-Attention FlashAttnExt over rolling KV cache
            Self::dispatch_flash_attn_decode(
                session,
                &scratch,
                q_off,
                &state,
                self_k_cache_off,
                &state,
                self_v_cache_off,
                &scratch,
                attn_out_off,
                head_dim,
                n_heads,
                pos + 1,
                self.config.n_text_ctx,
                scale,
            )?;

            dispatch::linear_m(
                session,
                &scratch,
                attn_out_off,
                &self.weights_buf,
                blk.attn_out_w,
                blk.attn_out_b_off,
                &scratch,
                attn_out_off,
                1,
                WHISPER_TILE,
            )?;
            dispatch::add_residual(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                attn_out_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1,
                },
                WHISPER_TILE,
            )?;

            // B. Cross-Attention over static encoder hidden states
            dispatch::layer_norm(
                session,
                LayerNormArgs {
                    src: &scratch,
                    src_offset: dec_x_off,
                    dst: &scratch,
                    dst_offset: norm_off,
                    weights: &self.weights_buf,
                    w_offset: blk.cross_attn_ln_w_off,
                    b_offset: blk.cross_attn_ln_b_off,
                    shape: TokenShape {
                        dim: d_model,
                        n_tokens: 1,
                    },
                    eps: 1e-5,
                    tile: WHISPER_TILE,
                },
            )?;

            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.cross_attn_q_w,
                blk.cross_attn_q_b_off,
                &scratch,
                cross_q_off,
                1,
                WHISPER_TILE,
            )?;

            let (cross_k_off, cross_v_off) = self.state_offsets.cross_kv[l];
            Self::dispatch_flash_attn_decode(
                session,
                &scratch,
                cross_q_off,
                &state,
                cross_k_off,
                &state,
                cross_v_off,
                &scratch,
                cross_out_off,
                head_dim,
                n_heads,
                1500,
                1500,
                scale,
            )?;

            dispatch::linear_m(
                session,
                &scratch,
                cross_out_off,
                &self.weights_buf,
                blk.cross_attn_out_w,
                blk.cross_attn_out_b_off,
                &scratch,
                cross_out_off,
                1,
                WHISPER_TILE,
            )?;
            dispatch::add_residual(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                cross_out_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1,
                },
                WHISPER_TILE,
            )?;

            // C. MLP
            dispatch::layer_norm(
                session,
                LayerNormArgs {
                    src: &scratch,
                    src_offset: dec_x_off,
                    dst: &scratch,
                    dst_offset: norm_off,
                    weights: &self.weights_buf,
                    w_offset: blk.mlp_ln_w_off,
                    b_offset: blk.mlp_ln_b_off,
                    shape: TokenShape {
                        dim: d_model,
                        n_tokens: 1,
                    },
                    eps: 1e-5,
                    tile: WHISPER_TILE,
                },
            )?;

            dispatch::linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.mlp_0_w,
                blk.mlp_0_b_off,
                &scratch,
                mlp_mid_off,
                1,
                WHISPER_TILE,
            )?;
            dispatch::gelu_tanh(
                session,
                &scratch,
                mlp_mid_off,
                &scratch,
                self.scratch_offsets.gelu_tmp_off,
                self.scratch_offsets.gelu_tmp_rows,
                TokenShape {
                    dim: blk.mlp_0_w.rows,
                    n_tokens: 1,
                },
            )?;
            dispatch::linear_m(
                session,
                &scratch,
                mlp_mid_off,
                &self.weights_buf,
                blk.mlp_2_w,
                blk.mlp_2_b_off,
                &scratch,
                mlp_out_off,
                1,
                WHISPER_TILE,
            )?;
            dispatch::add_residual(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                mlp_out_off,
                TokenShape {
                    dim: d_model,
                    n_tokens: 1,
                },
                WHISPER_TILE,
            )?;
        }

        // 3. Post-LayerNorm
        dispatch::layer_norm(
            session,
            LayerNormArgs {
                src: &scratch,
                src_offset: dec_x_off,
                dst: &scratch,
                dst_offset: dec_x_off,
                weights: &self.weights_buf,
                w_offset: self.weights_offsets.decoder_ln_post_w_off,
                b_offset: self.weights_offsets.decoder_ln_post_b_off,
                shape: TokenShape {
                    dim: d_model,
                    n_tokens: 1,
                },
                eps: 1e-5,
                tile: WHISPER_TILE,
            },
        )?;

        // 4. LM Head projection: logits = dec_x · proj_wᵀ
        let logits_off = self.scratch_offsets.logits_off;
        dispatch::linear_m(
            session,
            &scratch,
            dec_x_off,
            &self.weights_buf,
            self.weights_offsets.proj_w,
            None,
            &scratch,
            logits_off,
            1,
            WHISPER_TILE,
        )?;

        if let StepOut::Greedy { .. } = &out {
            // Mask the tokens a greedy step must not pick, then argmax the row on the DSP.
            dispatch::add_residual(
                session,
                &scratch,
                logits_off,
                &scratch,
                self.scratch_offsets.greedy_mask_off,
                TokenShape {
                    dim: self.config.n_vocab,
                    n_tokens: 1,
                },
                WHISPER_TILE,
            )?;
            dispatch::argmax_row(
                session,
                &scratch,
                logits_off,
                self.scratch_offsets.argmax_off,
                self.config.n_vocab,
            )?;
        }

        // Submit DSP queue and wait
        session.flush()?;

        match &mut out {
            StepOut::Logits(logits_out) => {
                // 6. Invalidate CPU cache and copy logits out
                let vocab_bytes = self.config.n_vocab * 4;
                scratch.invalidate_cpu_cache(logits_off, vocab_bytes);

                let scratch_slice = scratch.as_slice();
                let logits_slice: &[f32] =
                    bytemuck::cast_slice(&scratch_slice[logits_off..logits_off + vocab_bytes]);
                logits_out[..self.config.n_vocab].copy_from_slice(logits_slice);
                Ok(None)
            }
            StepOut::Greedy { .. } => {
                let at = self.scratch_offsets.argmax_off;
                scratch.invalidate_cpu_cache(at, 4);
                let token = readback_i32_le(scratch.as_slice(), at)?;
                u32::try_from(token)
                    .ok()
                    .filter(|&t| (t as usize) < self.config.n_vocab)
                    .map(Some)
                    .ok_or_else(|| {
                        CeraError::Backend(format!("the DSP argmax returned token {token}"))
                    })
            }
        }
    }

    /// Stage the greedy suppression mask for `timestamps` unless it already is: `-1e30` for the
    /// control tokens between `sot` and `no_timestamps` (except `eot`) and, without timestamps,
    /// for every token from `timestamp_begin` on; the host's `suppress_whisper_special_tokens`.
    fn stage_greedy_mask(&self, scratch: &mut RpcmemBuffer, timestamps: bool) {
        // The staged mask persists across calls: on poison the flag is discarded, so the mask
        // is restaged rather than trusted torn.
        let mut staged = lock_or_discard(&self.greedy_mask_for);
        if *staged == Some(timestamps) {
            return;
        }
        let n = self.config.n_vocab;
        let mask = greedy_mask(&self.special_tokens, n, timestamps);
        let at = self.scratch_offsets.greedy_mask_off;
        scratch.as_mut_slice()[at..at + n * 4].copy_from_slice(bytemuck::cast_slice(&mask));
        scratch.flush_cpu_cache(at, n * 4);
        *staged = Some(timestamps);
    }

    /// Whether the log-mel front end is staged on the DSP. This pins staging only: a staged
    /// call can still fall back per call (see [`Self::mel_fallbacks`]), so a caller timing the
    /// NPU path proves the timed calls ran on the DSP with the counter, not this predicate.
    #[doc(hidden)]
    pub fn mel_on_dsp(&self) -> bool {
        self.mel.is_some()
    }

    /// How often [`Self::log_mel`] fell back to the host after a DSP failure. A caller timing
    /// the NPU path captures this before its calls and asserts it is unchanged after.
    #[doc(hidden)]
    pub fn mel_fallbacks(&self) -> u64 {
        self.mel_fallbacks.load(Ordering::Relaxed)
    }

    /// Whisper's log-mel for `pcm`: the DFT and the filterbank on the DSP when they are staged,
    /// the host's FFT otherwise (and if the DSP call fails, with a warning).
    #[doc(hidden)]
    pub fn log_mel(&self, pcm: &[f32]) -> Vec<f32> {
        use crate::model::whisper_preprocessor::{
            active_frames, extract_whisper_mel, finish_whisper_mel, padded_whisper_audio_upto,
            samples_for_frames,
        };
        let n_mels = self.config.n_audio_mel_bins;
        if let Some(dsp) = &self.mel {
            let n_active = active_frames(pcm.len());
            let padded = padded_whisper_audio_upto(pcm, samples_for_frames(n_active));
            let energies = {
                let mut device = self.device.lock_or_recover();
                dsp.energies(device.queue_session_mut(), &padded, n_active)
            };
            match energies {
                Ok(e) => return finish_whisper_mel(&e, n_mels, n_active),
                Err(e) => {
                    self.mel_fallbacks.fetch_add(1, Ordering::Relaxed);
                    hexagon_warn!("whisper: log-mel failed on the DSP ({e}); using the host")
                }
            }
        }
        extract_whisper_mel(pcm, n_mels)
    }

    /// Transcribe PCM audio samples completely on Qualcomm Hexagon NPU.
    pub fn transcribe(
        &self,
        tokenizer: &crate::tokenizer::BpeTokenizer,
        pcm: &[f32],
        opts: &WhisperTranscribeOpts,
    ) -> Result<String, CeraError> {
        let _session_guard = self.session_lock.lock_or_recover();

        if opts
            .cancel
            .as_ref()
            .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(CeraError::Cancelled);
        }

        if pcm.is_empty() {
            return Ok(String::new());
        }

        // 1. Audio preprocessor: PCM -> log-mel spectrogram [n_mels x 3000]
        let mel = self.log_mel(pcm);

        if opts
            .cancel
            .as_ref()
            .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(CeraError::Cancelled);
        }

        // 2. Encode audio and precompute cross-attention on Hexagon DSP
        self.encode_audio(&mel)?;

        // 3. Assemble prompt tokens with dynamic language detection when language is unset or "auto"
        let is_multilingual = tokenizer.token_to_id("<|transcribe|>").is_some();
        let is_auto = opts
            .language
            .as_deref()
            .is_none_or(|l| l.is_empty() || l.eq_ignore_ascii_case("auto"));

        let mut logits = vec![0.0f32; self.config.n_vocab];
        let p = if is_multilingual && is_auto {
            self.decode_step(self.special_tokens.sot, 0, &mut logits)?;
            let max_valid_tok = (self.config.n_vocab.saturating_sub(1)) as u32;
            let lang_start = self.special_tokens.sot.saturating_add(1).min(max_valid_tok);
            let lang_end = self
                .special_tokens
                .translate
                .min(self.config.n_vocab as u32);
            let detected_lang_tok = (lang_start..lang_end)
                .filter(|&tok| logits.get(tok as usize).is_some_and(|l| l.is_finite()))
                .max_by(|&a, &b| {
                    let la = logits[a as usize];
                    let lb = logits[b as usize];
                    la.total_cmp(&lb)
                })
                .unwrap_or(lang_start);

            let mut prefix = vec![self.special_tokens.sot, detected_lang_tok];
            if opts.translate {
                prefix.push(self.special_tokens.translate);
            } else {
                prefix.push(self.special_tokens.transcribe);
            }
            if !opts.timestamps {
                prefix.push(self.special_tokens.no_timestamps);
            }

            if prefix.len() >= self.config.n_text_ctx {
                return Err(CeraError::Backend(format!(
                    "prompt length {} exceeds text context length {}",
                    prefix.len(),
                    self.config.n_text_ctx
                )));
            }

            // pos 0 was already decoded into kv_cache; prefill remaining prefix tokens (1..len-1)
            for (pos, &prompt_tok) in prefix
                .iter()
                .enumerate()
                .take(prefix.len().saturating_sub(1))
                .skip(1)
            {
                if opts
                    .cancel
                    .as_ref()
                    .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                {
                    return Err(CeraError::Cancelled);
                }
                self.decode_step(prompt_tok, pos, &mut logits)?;
            }
            prefix
        } else {
            let p = crate::model::whisper::assemble_whisper_prompt(
                &self.special_tokens,
                Some(tokenizer),
                opts.language.as_deref(),
                opts.translate,
                opts.timestamps,
            );

            if p.len() >= self.config.n_text_ctx {
                return Err(CeraError::Backend(format!(
                    "prompt length {} exceeds text context length {}",
                    p.len(),
                    self.config.n_text_ctx
                )));
            }

            // Prefill all prompt tokens except the last one
            for (pos, &prompt_tok) in p.iter().enumerate().take(p.len().saturating_sub(1)) {
                if opts
                    .cancel
                    .as_ref()
                    .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                {
                    return Err(CeraError::Cancelled);
                }
                self.decode_step(prompt_tok, pos, &mut logits)?;
            }
            p
        };

        // 4. Autoregressive decoding loop
        let mut current_token = *p
            .last()
            .ok_or_else(|| CeraError::Backend("empty prompt".into()))?;
        let mut generated_tokens = Vec::new();

        let start_pos = p.len().saturating_sub(1);
        let max_pos = self
            .config
            .n_text_ctx
            .min(start_pos.saturating_add(opts.max_tokens));

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplerConfig {
            temperature: opts.temperature,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        });

        // Temperature 0 is greedy decoding: the argmax is taken on the DSP.
        let greedy = !opts.temperature.is_finite() || opts.temperature <= 0.0;

        for pos in start_pos..max_pos {
            if opts
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            {
                return Err(CeraError::Cancelled);
            }

            let next_token = if greedy {
                self.decode_step_greedy(current_token, pos, opts.timestamps)?
            } else {
                self.decode_step(current_token, pos, &mut logits)?;

                // Suppress special control tokens during autoregressive generation
                crate::model::whisper::suppress_whisper_special_tokens(
                    &mut logits,
                    &self.special_tokens,
                    opts.timestamps,
                );

                sampler.sample(&mut logits)
            };
            if next_token == self.special_tokens.eot {
                break;
            }

            generated_tokens.push(next_token);
            current_token = next_token;
        }

        Ok(tokenizer.decode(&generated_tokens))
    }
}

/// Byte offset of the activation window feeding tap `k` for the token run
/// starting at `t_start`. The source is time-major with one zero pad row in
/// front, so tap `k` of output `t` reads row `t * stride + k`.
fn conv_tap_input_offset(
    src_base_offset: usize,
    row_bytes: usize,
    k: usize,
    t_start: usize,
    stride: usize,
) -> usize {
    src_base_offset + (k + t_start * stride) * row_bytes
}

/// Initialize Hexagon Whisper model, returning a detailed CeraError on failure.
pub fn init_hexagon_whisper(
    weights: &WhisperWeights,
    tokenizer: &crate::tokenizer::BpeTokenizer,
) -> Result<Arc<HexagonWhisperModel>, CeraError> {
    // Only a context failure means "driver failed to load"; classify it here
    // (kill switch / absent driver stay at info, anything else warns to
    // logcat) rather than in the caller, which also sees probe and model
    // construction errors that have different causes.
    let context = HexagonContext::new().map_err(|e| {
        crate::backend::hexagon::log_context_unavailable("HexagonWhisperModel", &e);
        CeraError::Backend(format!("Hexagon context creation failed: {e}"))
    })?;

    let arch_override = crate::backend::hexagon::arch_override();

    let dev = crate::backend::hexagon::probe_device(context.driver(), arch_override)
        .map_err(|e| CeraError::Backend(format!("Hexagon DSP device probe failed: {e}")))?;
    let device = Arc::new(Mutex::new(dev));

    let model = HexagonWhisperModel::new(Arc::clone(context.driver()), device, weights, tokenizer)?;
    tracing::info!("whisper: using native Qualcomm Hexagon NPU pipeline");
    Ok(Arc::new(model))
}

/// Why `weights` cannot run on the NPU, or `None` when their formats are all supported. The NPU
/// reads Q8_0, Q4_0, F16 and F32 matrices; the K-quants (`cera transcribe` converts to Q4_K_M
/// on CPU paths unless told otherwise) are not among them.
fn npu_weight_problem(weights: &WhisperWeights) -> Option<String> {
    HexagonWhisperWeightOffsets::plan(weights).err().map(|e| {
        format!(
            "these weights cannot run on the Hexagon NPU ({e}); reconvert the model to Q8_0 \
             (for a catalog model: `cera transcribe --model <alias> --quant q8_0 \
             --download-model`; a local GGUF must be reconverted from its source; Q8_0 is \
             the more accurate of the two quantized formats it reads (Q8_0 and Q4_0))"
        )
    })
}

/// Probe for Qualcomm Hexagon DSP and instantiate `HexagonWhisperModel` if available.
pub fn try_hexagon_whisper(
    weights: &WhisperWeights,
    tokenizer: &crate::tokenizer::BpeTokenizer,
) -> Option<Arc<HexagonWhisperModel>> {
    // A weight format the NPU cannot read is worth a warning, not a debug line: the model then
    // quietly runs on the CPU at about ten times the CPU time, and nothing says why.
    if let Some(problem) = npu_weight_problem(weights) {
        hexagon_warn!("whisper: {problem}; running on the CPU instead");
        return None;
    }
    match init_hexagon_whisper(weights, tokenizer) {
        Ok(model) => Some(model),
        Err(e) => {
            // Context failures were already classified inside
            // `init_hexagon_whisper`; probe and model-construction errors
            // are ordinary fallbacks, as in the vision loader.
            tracing::debug!("whisper: Hexagon NPU unavailable ({e}), falling back");
            None
        }
    }
}

/// Stage one 3-tap Conv1d into `dst`: tap `k` goes to the descriptor `w{k}`
/// (input offsets -1, 0, +1), each as a repacked Q8_0 matrix, then the bias.
fn stage_conv_taps(
    dst: &mut [u8],
    conv: &Conv1dWeights,
    desc: &HexagonWhisperConvWeights,
) -> Result<(), CeraError> {
    let in_ch = conv.in_channels;
    let padded_in_ch = in_ch.next_multiple_of(32);
    let out_ch = conv.out_channels;

    if conv.kernel_size != 3 {
        return Err(CeraError::Backend(format!(
            "conv kernel_size {} is not 3 for 3-tap decomposition",
            conv.kernel_size
        )));
    }
    let expected_weight_len = out_ch * in_ch * 3;
    if conv.weight.len() != expected_weight_len {
        return Err(CeraError::Backend(format!(
            "conv weight length {} does not match expected {}",
            conv.weight.len(),
            expected_weight_len
        )));
    }

    for (k, w_desc) in [desc.w0, desc.w1, desc.w2].iter().enumerate() {
        let tap_vals = conv_tap_matrix(conv, k, padded_in_ch);
        let q8_bytes = quantize_f32_to_q8_0(&tap_vals, padded_in_ch, out_ch)?;
        let dst_slice = &mut dst[w_desc.offset..w_desc.offset + w_desc.size_bytes];
        repack_q8_0(&q8_bytes, padded_in_ch, out_ch, dst_slice)
            .map_err(|e| CeraError::Backend(format!("conv tap {k} repack Q8_0 failed: {e}")))?;
    }
    let bias_bytes: &[u8] = bytemuck::cast_slice(&conv.bias);
    dst[desc.bias_off..desc.bias_off + bias_bytes.len()].copy_from_slice(bias_bytes);
    Ok(())
}

/// Tap `k` (0, 1, 2 = input offsets -1, 0, +1) of a 3-tap Conv1d as a row-major
/// `[out_channels, padded_in_ch]` matrix, zero-padded past `in_channels`. The
/// weights are `[out, in, 3]`, so tap `k` is every third element from `k`.
/// Shared by staging and the tests; the decomposition test pins the tap
/// order against a direct convolution.
fn conv_tap_matrix(conv: &Conv1dWeights, k: usize, padded_in_ch: usize) -> Vec<f32> {
    let (in_ch, out_ch) = (conv.in_channels, conv.out_channels);
    let mut tap = vec![0.0f32; out_ch * padded_in_ch];
    for r in 0..out_ch {
        for c in 0..in_ch {
            tap[r * padded_in_ch + c] = conv.weight[r * in_ch * 3 + c * 3 + k];
        }
    }
    tap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::whisper::*;

    /// Tile policy pin: the encoder's 1,500 frames exceed VTCM in one op, so
    /// Whisper tiles at 64 tokens (the ViT and detokenizer run whole). Pinned
    /// by behavior: 130 tokens are 3 tiles of a Norm/Mul/Add triple.
    #[test]
    fn tile_policy_is_64() {
        assert_eq!(dispatch::testing::layer_norm_op_count(WHISPER_TILE), 3 * 3);
    }

    #[test]
    fn test_state_offsets_plan_tiny() {
        let cfg = WhisperConfig {
            n_audio_layer: 4,
            n_audio_embd: 384,
            n_audio_head: 6,
            n_audio_mel_bins: 80,
            n_audio_ctx: 1500,
            n_text_layer: 4,
            n_text_embd: 384,
            n_text_head: 6,
            n_text_ctx: 448,
            n_vocab: 51865,
        };

        let state = HexagonWhisperStateOffsets::plan(&cfg);
        assert!(state.total_bytes > 0);
        assert_eq!(state.total_bytes % 128, 0);
        assert_eq!(state.cross_kv.len(), 4);
        assert_eq!(state.self_kv.len(), 4);
        for &(k_off, v_off) in &state.cross_kv {
            assert_eq!(k_off % 128, 0);
            assert_eq!(v_off % 128, 0);
        }
        for &(k_off, v_off) in &state.self_kv {
            assert_eq!(k_off % 128, 0);
            assert_eq!(v_off % 128, 0);
        }
    }

    #[test]
    fn test_scratch_offsets_plan_tiny() {
        let cfg = WhisperConfig {
            n_audio_layer: 4,
            n_audio_embd: 384,
            n_audio_head: 6,
            n_audio_mel_bins: 80,
            n_audio_ctx: 1500,
            n_text_layer: 4,
            n_text_embd: 384,
            n_text_head: 6,
            n_text_ctx: 448,
            n_vocab: 51865,
        };

        let scratch = HexagonWhisperScratchOffsets::plan(&cfg);
        assert!(scratch.total_bytes > 0);
        assert!(scratch.logits_off > scratch.enc_mlp_mid_off);
        assert!(scratch.total_bytes > scratch.logits_off);
        assert_eq!(scratch.mel_in_off % 128, 0);
        assert_eq!(scratch.conv1_out_off % 128, 0);
        assert_eq!(scratch.enc_x_off % 128, 0);
        assert_eq!(scratch.logits_off % 128, 0);
        assert_eq!(scratch.total_bytes % 128, 0);
    }

    #[test]
    fn test_conv1d_3tap_decomposition_mathematical_equivalence() {
        // Golden: replay the production tap layout (`conv_tap_matrix` for the
        // staged weights, `conv_tap_input_offset` for the activation windows)
        // against the CPU reference `Conv1dWeights::forward`. The activation is
        // laid out exactly as `encode_audio` stages it: time-major rows of
        // `padded_in` floats with one zero pad row on each side.
        let in_channels: usize = 80;
        let padded_in = in_channels.next_multiple_of(32);
        let out_channels = 24;
        let t_in = 300;

        let weight: Vec<f32> = (0..out_channels * in_channels * 3)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let bias: Vec<f32> = (0..out_channels).map(|i| (i as f32) * 0.01).collect();
        let conv = Conv1dWeights {
            weight,
            bias: bias.clone(),
            out_channels,
            in_channels,
            kernel_size: 3,
        };
        let in_data: Vec<f32> = (0..in_channels * t_in)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.1)
            .collect();

        let row_bytes = padded_in * 4;
        let mut padded = vec![0.0f32; (t_in + 2) * padded_in];
        for t in 0..t_in {
            for c in 0..in_channels {
                padded[(t + 1) * padded_in + c] = in_data[c * t_in + t];
            }
        }
        let taps: Vec<Vec<f32>> = (0..3)
            .map(|k| conv_tap_matrix(&conv, k, padded_in))
            .collect();

        for stride in [1usize, 2] {
            let t_out = t_in / stride;
            let mut ref_out = vec![0.0f32; out_channels * t_out];
            conv.forward(&in_data, t_in, stride, 1, &mut ref_out)
                .unwrap();

            let mut max_err = 0.0f32;
            // Two runs (split mid-way) so `t_start` offsets are exercised too.
            for (t_start, run) in [(0, t_out / 3), (t_out / 3, t_out - t_out / 3)] {
                for t in 0..run {
                    for r in 0..out_channels {
                        let mut sum = bias[r];
                        for (k, tap) in taps.iter().enumerate() {
                            let off = conv_tap_input_offset(0, row_bytes, k, t_start, stride);
                            let row = off / row_bytes + t * stride;
                            for c in 0..padded_in {
                                sum += padded[row * padded_in + c] * tap[r * padded_in + c];
                            }
                        }
                        let ref_val = ref_out[r * t_out + t_start + t];
                        max_err = max_err.max((ref_val - sum).abs());
                    }
                }
            }
            assert!(
                max_err < 1e-3,
                "stride {stride}: max error {max_err} exceeds threshold"
            );
        }
    }

    /// A one-layer, 64-wide Whisper with Q8_0 weights: small enough to stage on the fake driver.
    fn tiny_whisper_weights() -> WhisperWeights {
        let d_model = 64;
        let n_vocab = 128;
        let config = WhisperConfig {
            n_audio_layer: 1,
            n_audio_embd: d_model,
            n_audio_head: 2,
            n_audio_mel_bins: 80,
            n_audio_ctx: 1500,
            n_text_layer: 1,
            n_text_embd: d_model,
            n_text_head: 2,
            n_text_ctx: 448,
            n_vocab,
        };

        let make_q8 = |r, c| {
            let f32s = vec![0.05f32; r * c];
            let q8_bytes = quantize_f32_to_q8_0(&f32s, c, r).unwrap();
            MmapWeight::from_owned_bytes(q8_bytes, DType::Q8_0, r, c)
        };

        let conv1 = Conv1dWeights {
            weight: vec![0.02; d_model * 80 * 3],
            bias: vec![0.0; d_model],
            out_channels: d_model,
            in_channels: 80,
            kernel_size: 3,
        };
        let conv2 = Conv1dWeights {
            weight: vec![0.02; d_model * d_model * 3],
            bias: vec![0.0; d_model],
            out_channels: d_model,
            in_channels: d_model,
            kernel_size: 3,
        };

        use crate::model::whisper::{
            WhisperDecoderBlockWeights, WhisperEncoderBlockWeights, WhisperEncoderWeights,
        };

        let enc_block = WhisperEncoderBlockWeights {
            attn_ln_w: vec![1.0; d_model],
            attn_ln_b: vec![0.0; d_model],
            attn_q_w: make_q8(d_model, d_model),
            attn_q_b: Some(vec![0.0; d_model]),
            attn_k_w: make_q8(d_model, d_model),
            attn_k_b: Some(vec![0.0; d_model]),
            attn_v_w: make_q8(d_model, d_model),
            attn_v_b: Some(vec![0.0; d_model]),
            attn_out_w: make_q8(d_model, d_model),
            attn_out_b: Some(vec![0.0; d_model]),
            mlp_ln_w: vec![1.0; d_model],
            mlp_ln_b: vec![0.0; d_model],
            mlp_0_w: make_q8(d_model * 4, d_model),
            mlp_0_b: Some(vec![0.0; d_model * 4]),
            mlp_2_w: make_q8(d_model, d_model * 4),
            mlp_2_b: Some(vec![0.0; d_model]),
        };

        let encoder = WhisperEncoderWeights {
            conv1,
            conv2,
            positional_embedding: vec![0.01; 1500 * d_model],
            blocks: vec![enc_block],
            ln_post_w: vec![1.0; d_model],
            ln_post_b: vec![0.0; d_model],
        };

        let dec_block = WhisperDecoderBlockWeights {
            attn_ln_w: vec![1.0; d_model],
            attn_ln_b: vec![0.0; d_model],
            attn_q_w: make_q8(d_model, d_model),
            attn_q_b: Some(vec![0.0; d_model]),
            attn_k_w: make_q8(d_model, d_model),
            attn_k_b: Some(vec![0.0; d_model]),
            attn_v_w: make_q8(d_model, d_model),
            attn_v_b: Some(vec![0.0; d_model]),
            attn_out_w: make_q8(d_model, d_model),
            attn_out_b: Some(vec![0.0; d_model]),
            cross_attn_ln_w: vec![1.0; d_model],
            cross_attn_ln_b: vec![0.0; d_model],
            cross_attn_q_w: make_q8(d_model, d_model),
            cross_attn_q_b: Some(vec![0.0; d_model]),
            cross_attn_k_w: make_q8(d_model, d_model),
            cross_attn_k_b: Some(vec![0.0; d_model]),
            cross_attn_v_w: make_q8(d_model, d_model),
            cross_attn_v_b: Some(vec![0.0; d_model]),
            cross_attn_out_w: make_q8(d_model, d_model),
            cross_attn_out_b: Some(vec![0.0; d_model]),
            mlp_ln_w: vec![1.0; d_model],
            mlp_ln_b: vec![0.0; d_model],
            mlp_0_w: make_q8(d_model * 4, d_model),
            mlp_0_b: Some(vec![0.0; d_model * 4]),
            mlp_2_w: make_q8(d_model, d_model * 4),
            mlp_2_b: Some(vec![0.0; d_model]),
        };

        let decoder = WhisperDecoderWeights {
            token_embeddings: make_q8(n_vocab, d_model),
            positional_embedding: vec![0.01; 448 * d_model],
            blocks: vec![dec_block],
            ln_post_w: vec![1.0; d_model],
            ln_post_b: vec![0.0; d_model],
            proj_w: make_q8(n_vocab, d_model),
        };

        WhisperWeights {
            config,
            encoder,
            decoder,
        }
    }

    #[test]
    fn test_whisper_weight_planning_and_staging() {
        let weights = tiny_whisper_weights();
        let offsets = HexagonWhisperWeightOffsets::plan(&weights).unwrap();
        assert!(offsets.total_bytes > 0);

        let mut staged_bytes = vec![0u8; offsets.total_bytes];
        stage_whisper_weights(&weights, &offsets, &mut staged_bytes).unwrap();
        assert!(staged_bytes.iter().any(|&b| b != 0));
    }

    /// The greedy mask suppresses exactly the tokens the host's `suppress_whisper_special_tokens`
    /// does, with and without timestamps, and leaves `eot` alone. `eot` sits inside the control
    /// range, so the carve-out is exercised rather than passing trivially.
    #[test]
    fn the_greedy_mask_matches_the_host_suppression() {
        let t = WhisperSpecialTokens {
            sot: 10,
            eot: 12,
            transcribe: 12,
            translate: 11,
            no_timestamps: 14,
            sot_prev: 13,
            sot_lm: 15,
            no_speech: 16,
            timestamp_begin: 17,
        };
        for timestamps in [false, true] {
            let mut host = vec![0.0f32; 40];
            crate::model::whisper::suppress_whisper_special_tokens(&mut host, &t, timestamps);
            let mask = greedy_mask(&t, 40, timestamps);
            for (i, (&h, &m)) in host.iter().zip(&mask).enumerate() {
                assert_eq!(
                    h == f32::NEG_INFINITY,
                    m == GREEDY_SUPPRESSED,
                    "token {i}, timestamps {timestamps}"
                );
                assert!(m == 0.0 || m == GREEDY_SUPPRESSED);
            }
            assert_eq!(mask[t.eot as usize], 0.0, "eot stays selectable");
        }
    }

    /// The DSP readback returns the row's `i32` in bounds and a recoverable error (never a
    /// panic) on a short scratch, including `at` past the end or at `usize::MAX`.
    #[test]
    fn readback_i32_le_reads_in_bounds_and_errors_out_of_them() {
        let scratch: Vec<u8> = (0..32).collect();
        assert_eq!(readback_i32_le(&scratch, 0).unwrap(), 0x0302_0100);
        assert_eq!(readback_i32_le(&scratch, 28).unwrap(), 0x1f1e_1d1c);
        for at in [29, 30, 31, 32, 33, 1000, usize::MAX] {
            assert!(
                readback_i32_le(&scratch, at).is_err(),
                "at={at} should err, not panic"
            );
        }
        // The error associates each value with its label (presence alone would pass a
        // swapped message), so the failure is debuggable from the log as printed.
        let msg = format!("{}", readback_i32_le(&scratch, 29).unwrap_err());
        assert!(
            msg.contains("offset 29") && msg.contains("len 32"),
            "short-row error must label offset and len: {msg}"
        );
        let empty: &[u8] = &[];
        assert!(readback_i32_le(empty, 0).is_err());
    }

    /// The log-mel fallback counter starts at 0, stays there while the DSP answers, and
    /// counts host fallbacks (which stay bit-identical to the host front end). The on-device
    /// probe asserts both before trusting NPU mel timings.
    #[test]
    fn log_mel_counts_host_fallbacks() {
        use crate::backend::hexagon::sys::fake;
        fake::reset();
        let (driver, device) = crate::backend::hexagon::op_capture::fresh_device();
        let device = Arc::new(Mutex::new(device));
        let tokenizer = crate::tokenizer::BpeTokenizer::empty_for_test();
        let model = HexagonWhisperModel::new(
            driver,
            Arc::clone(&device),
            &tiny_whisper_weights(),
            &tokenizer,
        )
        .expect("stage the tiny model on the fake driver");
        assert!(model.mel_on_dsp());
        assert_eq!(model.mel_fallbacks(), 0);
        let pcm = vec![0.1f32; 16_000];
        model.log_mel(&pcm);
        assert_eq!(
            model.mel_fallbacks(),
            0,
            "the DSP answered, nothing fell back"
        );
        fake::with(|s| s.fail_write = true);
        let fell_back = model.log_mel(&pcm);
        assert_eq!(
            fell_back,
            crate::model::whisper_preprocessor::extract_whisper_mel(&pcm, 80)
        );
        assert_eq!(model.mel_fallbacks(), 1);
        fake::reset();

        // An unstaged front end (unsupported bin count) runs on the host without counting:
        // the probe asserts the staging predicate first for exactly this reason.
        let mut unstaged = tiny_whisper_weights();
        unstaged.config.n_audio_mel_bins = 64;
        let (driver, device) = crate::backend::hexagon::op_capture::fresh_device();
        let model =
            HexagonWhisperModel::new(driver, Arc::new(Mutex::new(device)), &unstaged, &tokenizer)
                .expect("stage the tiny model on the fake driver");
        assert!(!model.mel_on_dsp());
        model.log_mel(&pcm);
        assert_eq!(model.mel_fallbacks(), 0);
    }

    /// A K-quant weight (what `cera transcribe` converts to by default) is reported with the way
    /// to convert the model; Q8_0 weights are fine.
    #[test]
    fn a_weight_format_the_npu_cannot_read_is_explained() {
        let mut weights = tiny_whisper_weights();
        assert_eq!(npu_weight_problem(&weights), None);
        let (rows, cols) = (64, 256);
        weights.encoder.blocks[0].attn_q_w =
            MmapWeight::from_owned_bytes(vec![0u8; rows * 144], DType::Q4KM, rows, cols);
        let problem = npu_weight_problem(&weights).expect("a Q4_K weight is not supported");
        assert!(
            problem.contains("Q4KM")
                && problem.contains("--quant q8_0")
                && problem.contains("--model <alias>")
                && problem.contains("--download-model"),
            "{problem}"
        );
    }

    /// The model puts its own session's waits to sleep (see [`HexagonWhisperModel::new`]):
    /// `set_blocking_wait` returns the previous setting, which must already be `true`. Dropping
    /// the model restores the previous setting.
    #[test]
    fn a_whisper_model_makes_its_session_waits_sleep() {
        let (driver, device) = crate::backend::hexagon::op_capture::fresh_device();
        let device = Arc::new(Mutex::new(device));
        assert!(
            !device
                .lock_or_recover()
                .queue_session_mut()
                .set_blocking_wait(false),
            "a fresh session spins by default"
        );
        let tokenizer = crate::tokenizer::BpeTokenizer::empty_for_test();
        let model = HexagonWhisperModel::new(
            driver,
            Arc::clone(&device),
            &tiny_whisper_weights(),
            &tokenizer,
        )
        .expect("stage the tiny model on the fake driver");
        assert!(
            device
                .lock_or_recover()
                .queue_session_mut()
                .set_blocking_wait(true),
            "the model left its session spinning"
        );
        drop(model);
        assert!(
            !device
                .lock_or_recover()
                .queue_session_mut()
                .set_blocking_wait(false),
            "dropping the model did not restore the wait"
        );
    }

    #[test]
    fn test_stage_whisper_weights_destination_too_small() {
        let config = WhisperConfig {
            n_audio_layer: 1,
            n_audio_embd: 32,
            n_audio_head: 1,
            n_audio_mel_bins: 80,
            n_audio_ctx: 1500,
            n_text_layer: 1,
            n_text_embd: 32,
            n_text_head: 1,
            n_text_ctx: 448,
            n_vocab: 64,
        };
        let make_q8 = |r, c| {
            let f32s = vec![0.05f32; r * c];
            let q8_bytes = quantize_f32_to_q8_0(&f32s, c, r).unwrap();
            MmapWeight::from_owned_bytes(q8_bytes, DType::Q8_0, r, c)
        };
        let d_model = 32;
        let n_vocab = 64;
        let encoder = WhisperEncoderWeights {
            conv1: Conv1dWeights {
                weight: vec![0.01; d_model * 80 * 3],
                bias: vec![0.0; d_model],
                out_channels: d_model,
                in_channels: 80,
                kernel_size: 3,
            },
            conv2: Conv1dWeights {
                weight: vec![0.01; d_model * d_model * 3],
                bias: vec![0.0; d_model],
                out_channels: d_model,
                in_channels: d_model,
                kernel_size: 3,
            },
            positional_embedding: vec![0.01; 1500 * d_model],
            blocks: vec![WhisperEncoderBlockWeights {
                attn_ln_w: vec![1.0; d_model],
                attn_ln_b: vec![0.0; d_model],
                attn_q_w: make_q8(d_model, d_model),
                attn_q_b: Some(vec![0.0; d_model]),
                attn_k_w: make_q8(d_model, d_model),
                attn_k_b: Some(vec![0.0; d_model]),
                attn_v_w: make_q8(d_model, d_model),
                attn_v_b: Some(vec![0.0; d_model]),
                attn_out_w: make_q8(d_model, d_model),
                attn_out_b: Some(vec![0.0; d_model]),
                mlp_ln_w: vec![1.0; d_model],
                mlp_ln_b: vec![0.0; d_model],
                mlp_0_w: make_q8(d_model * 4, d_model),
                mlp_0_b: Some(vec![0.0; d_model * 4]),
                mlp_2_w: make_q8(d_model, d_model * 4),
                mlp_2_b: Some(vec![0.0; d_model]),
            }],
            ln_post_w: vec![1.0; d_model],
            ln_post_b: vec![0.0; d_model],
        };
        let decoder = WhisperDecoderWeights {
            token_embeddings: make_q8(n_vocab, d_model),
            positional_embedding: vec![0.01; 448 * d_model],
            blocks: vec![WhisperDecoderBlockWeights {
                attn_ln_w: vec![1.0; d_model],
                attn_ln_b: vec![0.0; d_model],
                attn_q_w: make_q8(d_model, d_model),
                attn_q_b: Some(vec![0.0; d_model]),
                attn_k_w: make_q8(d_model, d_model),
                attn_k_b: Some(vec![0.0; d_model]),
                attn_v_w: make_q8(d_model, d_model),
                attn_v_b: Some(vec![0.0; d_model]),
                attn_out_w: make_q8(d_model, d_model),
                attn_out_b: Some(vec![0.0; d_model]),
                cross_attn_ln_w: vec![1.0; d_model],
                cross_attn_ln_b: vec![0.0; d_model],
                cross_attn_q_w: make_q8(d_model, d_model),
                cross_attn_q_b: Some(vec![0.0; d_model]),
                cross_attn_k_w: make_q8(d_model, d_model),
                cross_attn_k_b: Some(vec![0.0; d_model]),
                cross_attn_v_w: make_q8(d_model, d_model),
                cross_attn_v_b: Some(vec![0.0; d_model]),
                cross_attn_out_w: make_q8(d_model, d_model),
                cross_attn_out_b: Some(vec![0.0; d_model]),
                mlp_ln_w: vec![1.0; d_model],
                mlp_ln_b: vec![0.0; d_model],
                mlp_0_w: make_q8(d_model * 4, d_model),
                mlp_0_b: Some(vec![0.0; d_model * 4]),
                mlp_2_w: make_q8(d_model, d_model * 4),
                mlp_2_b: Some(vec![0.0; d_model]),
            }],
            ln_post_w: vec![1.0; d_model],
            ln_post_b: vec![0.0; d_model],
            proj_w: make_q8(n_vocab, d_model),
        };
        let weights = WhisperWeights {
            config,
            encoder,
            decoder,
        };
        let offsets = HexagonWhisperWeightOffsets::plan(&weights).unwrap();
        let mut short_buf = vec![0u8; offsets.total_bytes - 1];
        let res = stage_whisper_weights(&weights, &offsets, &mut short_buf);
        let err = res.unwrap_err().to_string();
        assert!(err.contains("smaller than required offsets total_bytes"));
    }

    /// Tap `k` must land in descriptor `w{k}`: stage a small conv, then compare
    /// each descriptor's bytes with the repack of the matching tap matrix.
    #[test]
    fn stage_conv_taps_maps_each_tap_to_its_descriptor() {
        let (in_ch, out_ch) = (32usize, 32usize);
        let padded = in_ch.next_multiple_of(32);
        let weight: Vec<f32> = (0..out_ch * in_ch * 3)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.03)
            .collect();
        let conv = Conv1dWeights {
            weight,
            bias: vec![0.5; out_ch],
            out_channels: out_ch,
            in_channels: in_ch,
            kernel_size: 3,
        };
        let tap_size = repacked_matrix_size_q8_0(padded, out_ch).unwrap();
        let desc_at = |i: usize| HexagonWhisperWeightDesc {
            offset: i * align128(tap_size),
            size_bytes: tap_size,
            format: HexagonWeightFormat::RepackedQ8_0,
            rows: out_ch,
            cols: padded,
        };
        let desc = HexagonWhisperConvWeights {
            w0: desc_at(0),
            w1: desc_at(1),
            w2: desc_at(2),
            bias_off: 3 * align128(tap_size),
        };
        let mut dst = vec![0u8; desc.bias_off + align128(out_ch * 4)];
        stage_conv_taps(&mut dst, &conv, &desc).unwrap();
        for (k, d) in [desc.w0, desc.w1, desc.w2].iter().enumerate() {
            let tap = conv_tap_matrix(&conv, k, padded);
            let q8 = quantize_f32_to_q8_0(&tap, padded, out_ch).unwrap();
            let mut want = vec![0u8; tap_size];
            repack_q8_0(&q8, padded, out_ch, &mut want).unwrap();
            assert_eq!(
                &dst[d.offset..d.offset + d.size_bytes],
                &want[..],
                "tap {k}"
            );
        }
        assert_eq!(
            &dst[desc.bias_off..desc.bias_off + 4],
            &0.5f32.to_ne_bytes()
        );
    }
}

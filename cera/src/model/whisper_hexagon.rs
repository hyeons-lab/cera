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

use std::sync::{Arc, Mutex};

use crate::backend::hexagon::{
    FastRpcDriver, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch,
    HexagonContext, HexagonDevice, HexagonQueueSession, HexagonWeightFormat, HtpDataType,
    HtpOpCode, RpcmemBuffer, build_binary_kernel_params, build_flash_attn_kernel_params,
    build_layer_norm_params, build_mul_mat_kernel_params, build_unary_kernel_params,
    quantize_f32_to_q8_0, repack_q4_0, repack_q8_0, repacked_matrix_size_q4_0,
    repacked_matrix_size_q8_0,
};
use crate::model::weights::MmapWeight;
use crate::model::whisper::{
    Conv1dWeights, WhisperConfig, WhisperSpecialTokens, WhisperTranscribeOpts, WhisperWeights,
};
use crate::session::CeraError;
use crate::tensor::DType;

/// Alignment for DMA buffer offsets in rpcmem (128-byte HVX vector alignment).
const ALIGN_128: usize = 128;

fn align128(sz: usize) -> usize {
    (sz + ALIGN_128 - 1) & !(ALIGN_128 - 1)
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

    let stage_conv = |dst: &mut [u8],
                      conv: &Conv1dWeights,
                      desc: &HexagonWhisperConvWeights|
     -> Result<(), CeraError> {
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
            let mut tap_vals = vec![0.0f32; out_ch * padded_in_ch];
            for r in 0..out_ch {
                for c in 0..in_ch {
                    tap_vals[r * padded_in_ch + c] = conv.weight[r * in_ch * 3 + c * 3 + k];
                }
            }
            let q8_bytes = quantize_f32_to_q8_0(&tap_vals, padded_in_ch, out_ch)?;
            let dst_slice = &mut dst[w_desc.offset..w_desc.offset + w_desc.size_bytes];
            repack_q8_0(&q8_bytes, padded_in_ch, out_ch, dst_slice)
                .map_err(|e| CeraError::Backend(format!("conv tap {k} repack Q8_0 failed: {e}")))?;
        }
        copy_vec_f32(dst, desc.bias_off, &conv.bias);
        Ok(())
    };

    // Stage Conv1 & Conv2
    stage_conv(dst, &weights.encoder.conv1, &offsets.conv1)?;
    stage_conv(dst, &weights.encoder.conv2, &offsets.conv2)?;

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
            total_bytes: cur_off,
        }
    }
}

/// Native Qualcomm Hexagon NPU Whisper speech recognition model.
pub struct HexagonWhisperModel {
    #[allow(dead_code)]
    driver: Arc<FastRpcDriver>,
    device: Arc<Mutex<HexagonDevice>>,
    config: WhisperConfig,
    #[allow(dead_code)]
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

        Ok(Self {
            driver,
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
                8 * 1024 * 1024,
            );

            // Tap 0 -> cur_dst
            let x0_off = src_base_offset + t_start * stride * row_bytes;
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
            let x1_off = src_base_offset + row_bytes + t_start * stride * row_bytes;
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
                8 * 1024 * 1024,
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
            let x2_off = src_base_offset + 2 * row_bytes + t_start * stride * row_bytes;
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
                8 * 1024 * 1024,
                session.dsp_threads(),
            );
            session.enqueue_op(
                HtpOpCode::Add as u32,
                &[cur_dst_ti, b_ti],
                &[cur_dst_ti],
                [0i32; 16],
                bias_add_kparams,
            )?;

            // In-place GELU
            let gelu_kparams = build_unary_kernel_params(
                out_channels,
                chunk,
                0,
                8 * 1024 * 1024,
                session.dsp_threads(),
                false,
            );
            session.enqueue_op(
                HtpOpCode::UnaryGelu as u32,
                &[cur_dst_ti],
                &[cur_dst_ti],
                [0i32; 16],
                gelu_kparams,
            )?;

            t_start += chunk;
        }

        Ok(())
    }

    /// Dispatch GELU activation in-place: `buf[i] = gelu(buf[i])`.
    fn dispatch_gelu(
        session: &mut HexagonQueueSession,
        buf: &RpcmemBuffer,
        offset: usize,
        dim: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let chunk_size = 64;
        let mut t_start = 0;
        while t_start < n_tokens {
            let chunk = (n_tokens - t_start).min(chunk_size);
            let cur_off = offset + t_start * dim * 4;
            let bytes = dim * chunk * 4;
            let ti = session.add_tensor(
                buf,
                cur_off,
                bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [dim as u32, chunk as u32, 1, 1],
                [4, (dim * 4) as u32, bytes as u32, bytes as u32],
            )?;

            let params = [0i32; 16];
            let kparams = build_unary_kernel_params(
                dim,
                chunk,
                0,
                8 * 1024 * 1024,
                session.dsp_threads(),
                false,
            );
            session.enqueue_op(HtpOpCode::UnaryGelu as u32, &[ti], &[ti], params, kparams)?;
            t_start += chunk;
        }
        Ok(())
    }

    /// Dispatch LayerNorm: `Norm` -> `Mul` (weight) -> `Add` (bias).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_layer_norm(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        weights: &RpcmemBuffer,
        w_offset: usize,
        b_offset: usize,
        dim: usize,
        n_tokens: usize,
        eps: f32,
    ) -> Result<(), CeraError> {
        let w_bytes = dim * 4;
        let w_ti = session.add_tensor(
            weights,
            w_offset,
            w_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [dim as u32, 1, 1, 1],
            [4, w_bytes as u32, w_bytes as u32, w_bytes as u32],
        )?;

        let b_ti = session.add_tensor(
            weights,
            b_offset,
            w_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [dim as u32, 1, 1, 1],
            [4, w_bytes as u32, w_bytes as u32, w_bytes as u32],
        )?;

        let chunk_size = 64;
        let mut t_start = 0;
        while t_start < n_tokens {
            let chunk = (n_tokens - t_start).min(chunk_size);
            let bytes = dim * chunk * 4;
            let cur_src_off = src_offset + t_start * dim * 4;
            let cur_dst_off = dst_offset + t_start * dim * 4;

            let src_ti = session.add_tensor(
                src,
                cur_src_off,
                bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [dim as u32, chunk as u32, 1, 1],
                [4, (dim * 4) as u32, bytes as u32, bytes as u32],
            )?;
            let dst_ti = session.add_tensor(
                dst,
                cur_dst_off,
                bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [dim as u32, chunk as u32, 1, 1],
                [4, (dim * 4) as u32, bytes as u32, bytes as u32],
            )?;

            let params = build_layer_norm_params(eps);
            let kparams = build_unary_kernel_params(
                dim,
                chunk,
                0,
                8 * 1024 * 1024,
                session.dsp_threads(),
                false,
            );
            session.enqueue_op(
                HtpOpCode::Norm as u32,
                &[src_ti],
                &[dst_ti],
                params,
                kparams,
            )?;

            let mul_params = [0i32; 16];
            let mul_kparams = build_binary_kernel_params(
                dim,
                dim,
                1,
                1,
                1,
                4,
                8 * 1024 * 1024,
                session.dsp_threads(),
            );
            session.enqueue_op(
                HtpOpCode::Mul as u32,
                &[dst_ti, w_ti],
                &[dst_ti],
                mul_params,
                mul_kparams,
            )?;

            let add_params = [0i32; 16];
            let add_kparams = build_binary_kernel_params(
                dim,
                dim,
                1,
                1,
                1,
                4,
                8 * 1024 * 1024,
                session.dsp_threads(),
            );
            session.enqueue_op(
                HtpOpCode::Add as u32,
                &[dst_ti, b_ti],
                &[dst_ti],
                add_params,
                add_kparams,
            )?;

            t_start += chunk;
        }

        Ok(())
    }

    /// Dispatch linear matrix multiplication + optional bias add: `dst = x · Wᵀ + bias`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_m(
        session: &mut HexagonQueueSession,
        x: &RpcmemBuffer,
        x_offset: usize,
        weights: &RpcmemBuffer,
        w_desc: HexagonWhisperWeightDesc,
        b_offset: Option<usize>,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let (w_dtype, block_bytes, tile_size) = match w_desc.format {
            HexagonWeightFormat::RepackedQ8_0 => (HtpDataType::Q8_0, 34usize, 32 * 34usize),
            HexagonWeightFormat::RepackedQ4_0 => (HtpDataType::Q4_0, 18usize, 32 * 18usize),
        };

        let ne0 = w_desc.cols;
        let ne1 = w_desc.rows;
        let tiled_row_bytes = ne0.div_ceil(32) * tile_size;
        let w_tot = ne1.div_ceil(32) * tiled_row_bytes;
        let w_ti = session.add_tensor(
            weights,
            w_desc.offset,
            w_desc.size_bytes,
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

        let b_ti = if let Some(b_off) = b_offset {
            let b_bytes = w_desc.rows * 4;
            let ti = session.add_tensor(
                weights,
                b_off,
                b_bytes,
                HTP_TENSOR_WEIGHT,
                HtpDataType::F32 as u32,
                [w_desc.rows as u32, 1, 1, 1],
                [4, b_bytes as u32, b_bytes as u32, b_bytes as u32],
            )?;
            Some(ti)
        } else {
            None
        };

        let chunk_size = 64;
        let mut t_start = 0;
        while t_start < n_tokens {
            let chunk = (n_tokens - t_start).min(chunk_size);
            let cur_x_off = x_offset + t_start * w_desc.cols * 4;
            let cur_x_bytes = chunk * w_desc.cols * 4;
            let x_ti = session.add_tensor(
                x,
                cur_x_off,
                cur_x_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [w_desc.cols as u32, chunk as u32, 1, 1],
                [
                    4,
                    (w_desc.cols * 4) as u32,
                    cur_x_bytes as u32,
                    cur_x_bytes as u32,
                ],
            )?;

            let cur_dst_off = dst_offset + t_start * w_desc.rows * 4;
            let cur_dst_bytes = chunk * w_desc.rows * 4;
            let dst_ti = session.add_tensor(
                dst,
                cur_dst_off,
                cur_dst_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [w_desc.rows as u32, chunk as u32, 1, 1],
                [
                    4,
                    (w_desc.rows * 4) as u32,
                    cur_dst_bytes as u32,
                    cur_dst_bytes as u32,
                ],
            )?;

            let params = [0i32; 16];
            let kparams = build_mul_mat_kernel_params(
                w_dtype,
                w_desc.cols,
                chunk as u32,
                1,
                w_desc.rows * 4,
                session.dsp_threads(),
                8 * 1024 * 1024,
            );

            session.enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w_ti, x_ti],
                &[dst_ti],
                params,
                kparams,
            )?;

            if let Some(bti) = b_ti {
                let add_params = [0i32; 16];
                let add_kparams = build_binary_kernel_params(
                    w_desc.rows,
                    w_desc.rows,
                    1,
                    1,
                    1,
                    4,
                    8 * 1024 * 1024,
                    session.dsp_threads(),
                );
                session.enqueue_op(
                    HtpOpCode::Add as u32,
                    &[dst_ti, bti],
                    &[dst_ti],
                    add_params,
                    add_kparams,
                )?;
            }

            t_start += chunk;
        }

        Ok(())
    }

    /// Dispatch F32 to F16 data type conversion in rpcmem.
    fn dispatch_cpy_f32_to_f16(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let chunk_size = 64;
        let mut t_start = 0;
        while t_start < n_tokens {
            let chunk = (n_tokens - t_start).min(chunk_size);
            let src_bytes = dim * chunk * 4;
            let dst_bytes = dim * chunk * 2;
            let cur_src_off = src_offset + t_start * dim * 4;
            let cur_dst_off = dst_offset + t_start * dim * 2;

            let src_ti = session.add_tensor(
                src,
                cur_src_off,
                src_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [dim as u32, chunk as u32, 1, 1],
                [4, (dim * 4) as u32, src_bytes as u32, src_bytes as u32],
            )?;
            let dst_ti = session.add_tensor(
                dst,
                cur_dst_off,
                dst_bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F16 as u32,
                [dim as u32, chunk as u32, 1, 1],
                [2, (dim * 2) as u32, dst_bytes as u32, dst_bytes as u32],
            )?;
            let params = [0i32; 16];
            let kparams = [0i32; 32];
            session.enqueue_op(HtpOpCode::Cpy as u32, &[src_ti], &[dst_ti], params, kparams)?;
            t_start += chunk;
        }
        Ok(())
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
        Ok(())
    }

    /// Dispatch in-place residual addition: `dst += src`.
    fn dispatch_add_residual(
        session: &mut HexagonQueueSession,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        src: &RpcmemBuffer,
        src_offset: usize,
        dim: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let chunk_size = 64;
        let mut t_start = 0;
        while t_start < n_tokens {
            let chunk = (n_tokens - t_start).min(chunk_size);
            let cur_dst_off = dst_offset + t_start * dim * 4;
            let cur_src_off = src_offset + t_start * dim * 4;
            let bytes = dim * chunk * 4;
            let ne = [dim as u32, chunk as u32, 1, 1];
            let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];

            let dst_ti = session.add_tensor(
                dst,
                cur_dst_off,
                bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                ne,
                nb,
            )?;
            let src_ti = session.add_tensor(
                src,
                cur_src_off,
                bytes,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                ne,
                nb,
            )?;

            let params = [0i32; 16];
            let kparams = build_binary_kernel_params(
                dim,
                dim,
                chunk,
                1,
                1,
                4,
                8 * 1024 * 1024,
                session.dsp_threads(),
            );

            session.enqueue_op(
                HtpOpCode::Add as u32,
                &[dst_ti, src_ti],
                &[dst_ti],
                params,
                kparams,
            )?;

            t_start += chunk;
        }
        Ok(())
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
        Ok(())
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
        Ok(())
    }

    /// Execute audio encoder and precompute static cross-attention on Qualcomm Hexagon NPU.
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

        let mut scratch = self.scratch_buf.lock().map_err(|e| {
            CeraError::Backend(format!("scratch_buf lock poisoned in encode_audio: {e}"))
        })?;
        let state = self.state_buf.lock().map_err(|e| {
            CeraError::Backend(format!("state_buf lock poisoned in encode_audio: {e}"))
        })?;

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
        let mut dev_guard = self.device.lock().map_err(|e| {
            CeraError::Backend(format!("device lock poisoned in encode_audio: {e}"))
        })?;
        let session = dev_guard.queue_session_mut();
        session.drop_pending_batch();

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
        Self::dispatch_add_residual(
            session,
            &scratch,
            enc_x_off,
            &self.weights_buf,
            pos_off,
            d_model,
            1500,
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
            Self::dispatch_layer_norm(
                session,
                &scratch,
                enc_x_off,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_ln_w_off,
                blk.attn_ln_b_off,
                d_model,
                1500,
                1e-5,
            )?;

            // Q, K, V projections
            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.q_w,
                blk.q_b_off,
                &scratch,
                q_off,
                1500,
            )?;
            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.k_w,
                blk.k_b_off,
                &scratch,
                tmp_off,
                1500,
            )?;
            Self::dispatch_cpy_f32_to_f16(
                session, &scratch, tmp_off, &scratch, k_f16_off, d_model, 1500,
            )?;

            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.v_w,
                blk.v_b_off,
                &scratch,
                tmp_off,
                1500,
            )?;
            Self::dispatch_cpy_f32_to_f16(
                session, &scratch, tmp_off, &scratch, v_f16_off, d_model, 1500,
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
            Self::dispatch_linear_m(
                session,
                &scratch,
                attn_out_off,
                &self.weights_buf,
                blk.o_w,
                blk.o_b_off,
                &scratch,
                attn_out_off,
                1500,
            )?;
            Self::dispatch_add_residual(
                session,
                &scratch,
                enc_x_off,
                &scratch,
                attn_out_off,
                d_model,
                1500,
            )?;

            // LayerNorm 2
            Self::dispatch_layer_norm(
                session,
                &scratch,
                enc_x_off,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.mlp_ln_w_off,
                blk.mlp_ln_b_off,
                d_model,
                1500,
                1e-5,
            )?;

            // MLP: MLP0 -> GELU -> MLP2 + residual add
            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.mlp_0_w,
                blk.mlp_0_b_off,
                &scratch,
                mlp_mid_off,
                1500,
            )?;
            Self::dispatch_gelu(session, &scratch, mlp_mid_off, blk.mlp_0_w.rows, 1500)?;
            Self::dispatch_linear_m(
                session,
                &scratch,
                mlp_mid_off,
                &self.weights_buf,
                blk.mlp_2_w,
                blk.mlp_2_b_off,
                &scratch,
                mlp_out_off,
                1500,
            )?;
            Self::dispatch_add_residual(
                session,
                &scratch,
                enc_x_off,
                &scratch,
                mlp_out_off,
                d_model,
                1500,
            )?;
        }

        // 4. Post-LayerNorm -> state_buf.encoder_hidden_off
        let enc_hidden_off = self.state_offsets.encoder_hidden_off;
        Self::dispatch_layer_norm(
            session,
            &scratch,
            enc_x_off,
            &state,
            enc_hidden_off,
            &self.weights_buf,
            self.weights_offsets.encoder_ln_post_w_off,
            self.weights_offsets.encoder_ln_post_b_off,
            d_model,
            1500,
            1e-5,
        )?;

        // 5. Precompute Cross-Attention K & V into state_buf.cross_kv
        for (l, dec_blk) in self.weights_offsets.decoder_blocks.iter().enumerate() {
            let (cross_k_off, cross_v_off) = self.state_offsets.cross_kv[l];

            // Cross-K projection -> F16 Cpy
            Self::dispatch_linear_m(
                session,
                &state,
                enc_hidden_off,
                &self.weights_buf,
                dec_blk.cross_attn_k_w,
                dec_blk.cross_attn_k_b_off,
                &scratch,
                tmp_off,
                1500,
            )?;
            Self::dispatch_cpy_f32_to_f16(
                session,
                &scratch,
                tmp_off,
                &state,
                cross_k_off,
                d_model,
                1500,
            )?;

            // Cross-V projection -> F16 Cpy
            Self::dispatch_linear_m(
                session,
                &state,
                enc_hidden_off,
                &self.weights_buf,
                dec_blk.cross_attn_v_w,
                dec_blk.cross_attn_v_b_off,
                &scratch,
                tmp_off,
                1500,
            )?;
            Self::dispatch_cpy_f32_to_f16(
                session,
                &scratch,
                tmp_off,
                &state,
                cross_v_off,
                d_model,
                1500,
            )?;
        }

        // Submit DSP batch queue
        session.flush()?;
        Ok(())
    }

    /// Execute one autoregressive decoding step on Qualcomm Hexagon NPU.
    pub fn decode_step(
        &self,
        token_id: u32,
        pos: usize,
        logits_out: &mut [f32],
    ) -> Result<(), CeraError> {
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
        if logits_out.len() < self.config.n_vocab {
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

        let mut scratch = self.scratch_buf.lock().map_err(|e| {
            CeraError::Backend(format!("scratch_buf lock poisoned in decode_step: {e}"))
        })?;
        let state = self.state_buf.lock().map_err(|e| {
            CeraError::Backend(format!("state_buf lock poisoned in decode_step: {e}"))
        })?;

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
        let mut dev_guard = self
            .device
            .lock()
            .map_err(|e| CeraError::Backend(format!("device lock poisoned in decode_step: {e}")))?;
        let session = dev_guard.queue_session_mut();
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
            Self::dispatch_layer_norm(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_ln_w_off,
                blk.attn_ln_b_off,
                d_model,
                1,
                1e-5,
            )?;

            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_q_w,
                blk.attn_q_b_off,
                &scratch,
                q_off,
                1,
            )?;
            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_k_w,
                blk.attn_k_b_off,
                &scratch,
                k_off,
                1,
            )?;
            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.attn_v_w,
                blk.attn_v_b_off,
                &scratch,
                v_off,
                1,
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

            Self::dispatch_linear_m(
                session,
                &scratch,
                attn_out_off,
                &self.weights_buf,
                blk.attn_out_w,
                blk.attn_out_b_off,
                &scratch,
                attn_out_off,
                1,
            )?;
            Self::dispatch_add_residual(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                attn_out_off,
                d_model,
                1,
            )?;

            // B. Cross-Attention over static encoder hidden states
            Self::dispatch_layer_norm(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.cross_attn_ln_w_off,
                blk.cross_attn_ln_b_off,
                d_model,
                1,
                1e-5,
            )?;

            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.cross_attn_q_w,
                blk.cross_attn_q_b_off,
                &scratch,
                cross_q_off,
                1,
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

            Self::dispatch_linear_m(
                session,
                &scratch,
                cross_out_off,
                &self.weights_buf,
                blk.cross_attn_out_w,
                blk.cross_attn_out_b_off,
                &scratch,
                cross_out_off,
                1,
            )?;
            Self::dispatch_add_residual(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                cross_out_off,
                d_model,
                1,
            )?;

            // C. MLP
            Self::dispatch_layer_norm(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.mlp_ln_w_off,
                blk.mlp_ln_b_off,
                d_model,
                1,
                1e-5,
            )?;

            Self::dispatch_linear_m(
                session,
                &scratch,
                norm_off,
                &self.weights_buf,
                blk.mlp_0_w,
                blk.mlp_0_b_off,
                &scratch,
                mlp_mid_off,
                1,
            )?;
            Self::dispatch_gelu(session, &scratch, mlp_mid_off, blk.mlp_0_w.rows, 1)?;
            Self::dispatch_linear_m(
                session,
                &scratch,
                mlp_mid_off,
                &self.weights_buf,
                blk.mlp_2_w,
                blk.mlp_2_b_off,
                &scratch,
                mlp_out_off,
                1,
            )?;
            Self::dispatch_add_residual(
                session,
                &scratch,
                dec_x_off,
                &scratch,
                mlp_out_off,
                d_model,
                1,
            )?;
        }

        // 3. Post-LayerNorm
        Self::dispatch_layer_norm(
            session,
            &scratch,
            dec_x_off,
            &scratch,
            dec_x_off,
            &self.weights_buf,
            self.weights_offsets.decoder_ln_post_w_off,
            self.weights_offsets.decoder_ln_post_b_off,
            d_model,
            1,
            1e-5,
        )?;

        // 4. LM Head projection: logits = dec_x · proj_wᵀ
        let logits_off = self.scratch_offsets.logits_off;
        Self::dispatch_linear_m(
            session,
            &scratch,
            dec_x_off,
            &self.weights_buf,
            self.weights_offsets.proj_w,
            None,
            &scratch,
            logits_off,
            1,
        )?;

        // Submit DSP queue and wait
        session.flush()?;

        // 6. Invalidate CPU cache and copy logits out
        let vocab_bytes = self.config.n_vocab * 4;
        scratch.invalidate_cpu_cache(logits_off, vocab_bytes);

        let scratch_slice = scratch.as_slice();
        let logits_slice: &[f32] =
            bytemuck::cast_slice(&scratch_slice[logits_off..logits_off + vocab_bytes]);
        logits_out[..self.config.n_vocab].copy_from_slice(logits_slice);

        Ok(())
    }

    /// Transcribe PCM audio samples completely on Qualcomm Hexagon NPU.
    pub fn transcribe(
        &self,
        tokenizer: &crate::tokenizer::BpeTokenizer,
        pcm: &[f32],
        opts: &WhisperTranscribeOpts,
    ) -> Result<String, CeraError> {
        let _session_guard = self
            .session_lock
            .lock()
            .map_err(|e| CeraError::Backend(format!("session lock poison: {e}")))?;

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
        let mel = crate::model::whisper_preprocessor::extract_whisper_mel(
            pcm,
            self.config.n_audio_mel_bins,
        );

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

        for pos in start_pos..max_pos {
            if opts
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            {
                return Err(CeraError::Cancelled);
            }

            self.decode_step(current_token, pos, &mut logits)?;

            // Suppress special control tokens during autoregressive generation
            crate::model::whisper::suppress_whisper_special_tokens(
                &mut logits,
                &self.special_tokens,
                opts.timestamps,
            );

            let next_token = sampler.sample(&mut logits);
            if next_token == self.special_tokens.eot {
                break;
            }

            generated_tokens.push(next_token);
            current_token = next_token;
        }

        Ok(tokenizer.decode(&generated_tokens))
    }
}

/// Initialize Hexagon Whisper model, returning a detailed CeraError on failure.
pub fn init_hexagon_whisper(
    weights: &WhisperWeights,
    tokenizer: &crate::tokenizer::BpeTokenizer,
) -> Result<Arc<HexagonWhisperModel>, CeraError> {
    if std::env::var("CERA_DISABLE_HEXAGON").is_ok() || std::env::var("CERA_NO_HEXAGON").is_ok() {
        return Err(CeraError::Backend(
            "Hexagon backend disabled via environment variable".into(),
        ));
    }

    let context = HexagonContext::new()
        .map_err(|e| CeraError::Backend(format!("Hexagon context creation failed: {e}")))?;

    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(HexagonArch::from_u32);

    let dev = crate::backend::hexagon::probe_device(context.driver(), arch_override)
        .map_err(|e| CeraError::Backend(format!("Hexagon DSP device probe failed: {e}")))?;
    let device = Arc::new(Mutex::new(dev));

    let model = HexagonWhisperModel::new(Arc::clone(context.driver()), device, weights, tokenizer)?;
    tracing::info!("whisper: using native Qualcomm Hexagon NPU pipeline");
    Ok(Arc::new(model))
}

/// Probe for Qualcomm Hexagon DSP and instantiate `HexagonWhisperModel` if available.
pub fn try_hexagon_whisper(
    weights: &WhisperWeights,
    tokenizer: &crate::tokenizer::BpeTokenizer,
) -> Option<Arc<HexagonWhisperModel>> {
    match init_hexagon_whisper(weights, tokenizer) {
        Ok(model) => Some(model),
        Err(e) => {
            tracing::info!("HexagonWhisperModel unavailable ({e}), falling back");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::whisper::*;

    #[test]
    fn test_align128() {
        assert_eq!(align128(0), 0);
        assert_eq!(align128(1), 128);
        assert_eq!(align128(128), 128);
        assert_eq!(align128(129), 256);
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
        // Verify that 3-tap decomposition mathematically matches Conv1dWeights::forward
        let in_channels = 80;
        let out_channels = 384;
        let kernel_size = 3;
        let t_in = 3000;

        let mut weight = vec![0.0f32; out_channels * in_channels * kernel_size];
        for (i, w) in weight.iter_mut().enumerate() {
            *w = ((i % 17) as f32 - 8.0) * 0.05;
        }
        let bias: Vec<f32> = (0..out_channels).map(|i| (i as f32) * 0.01).collect();

        let conv = Conv1dWeights {
            weight: weight.clone(),
            bias: bias.clone(),
            out_channels,
            in_channels,
            kernel_size,
        };

        let mut in_data = vec![0.0f32; in_channels * t_in];
        for (i, v) in in_data.iter_mut().enumerate() {
            *v = ((i % 23) as f32 - 11.0) * 0.1;
        }

        // 3-tap matrix decomposition:
        // Extract taps W0, W1, W2 of shape [out_channels, in_channels]
        let mut w0 = vec![0.0f32; out_channels * in_channels];
        let mut w1 = vec![0.0f32; out_channels * in_channels];
        let mut w2 = vec![0.0f32; out_channels * in_channels];

        for r in 0..out_channels {
            for c in 0..in_channels {
                w0[r * in_channels + c] = weight[r * in_channels * 3 + c * 3];
                w1[r * in_channels + c] = weight[r * in_channels * 3 + c * 3 + 1];
                w2[r * in_channels + c] = weight[r * in_channels * 3 + c * 3 + 2];
            }
        }

        for stride in [1, 2] {
            let t_out = t_in / stride;
            let mut ref_out = vec![0.0f32; out_channels * t_out];
            conv.forward(&in_data, t_in, stride, 1, &mut ref_out)
                .unwrap();

            // Decomposed forward: time-major output [t_out, out_channels]
            let mut tap_out = vec![0.0f32; t_out * out_channels];
            for t in 0..t_out {
                let in_t = t * stride;
                for r in 0..out_channels {
                    let mut sum = bias[r];
                    // Tap 0 (in_t - 1)
                    if in_t > 0 {
                        for c in 0..in_channels {
                            sum += in_data[c * t_in + (in_t - 1)] * w0[r * in_channels + c];
                        }
                    }
                    // Tap 1 (in_t)
                    for c in 0..in_channels {
                        sum += in_data[c * t_in + in_t] * w1[r * in_channels + c];
                    }
                    // Tap 2 (in_t + 1)
                    if in_t + 1 < t_in {
                        for c in 0..in_channels {
                            sum += in_data[c * t_in + (in_t + 1)] * w2[r * in_channels + c];
                        }
                    }
                    tap_out[t * out_channels + r] = sum;
                }
            }

            // Verify that tap_out[t, r] == ref_out[r, t]
            let mut max_err = 0.0f32;
            for t in 0..t_out {
                for r in 0..out_channels {
                    let ref_val = ref_out[r * t_out + t];
                    let tap_val = tap_out[t * out_channels + r];
                    let diff = (ref_val - tap_val).abs();
                    max_err = max_err.max(diff);
                }
            }
            assert!(
                max_err < 1e-4,
                "stride {stride}: max error {max_err} exceeds threshold"
            );
        }
    }

    #[test]
    fn test_whisper_weight_planning_and_staging() {
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

        let weights = WhisperWeights {
            config,
            encoder,
            decoder,
        };

        let offsets = HexagonWhisperWeightOffsets::plan(&weights).unwrap();
        assert!(offsets.total_bytes > 0);

        let mut staged_bytes = vec![0u8; offsets.total_bytes];
        stage_whisper_weights(&weights, &offsets, &mut staged_bytes).unwrap();
        assert!(staged_bytes.iter().any(|&b| b != 0));
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
        assert!(res.is_err());
    }
}

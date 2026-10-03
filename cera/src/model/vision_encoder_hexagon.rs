//! Native Qualcomm Hexagon NPU Vision Transformer (ViT) encoder.
//!
//! Accelerates the LFM2-VL ViT vision encoder on Snapdragon Hexagon Tensor
//! Processors (HTP) using FastRPC shared memory and asynchronous command queues.
//!
//! Keeps all weights and activations inside unified DMA buffers (rpcmem) to
//! stay within hardware and kernel memory mapping bounds, executing LayerNorm,
//! linear GEMM, and activations on the NPU.

use anyhow::{Result, anyhow, ensure};
use std::sync::{Arc, Mutex};

use crate::backend::hexagon::dispatch::{self, LayerNormArgs, TokenShape, TokenTile};
use crate::backend::hexagon::{
    FastRpcDriver, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch,
    HexagonContext, HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, LockOrRecover,
    RpcmemBuffer, align128, build_binary_kernel_params, repack_q4_0, repack_q8_0,
    repacked_matrix_size_q4_0, repacked_matrix_size_q8_0,
};
use crate::model::vision_encoder::{
    PatchEmbedWeights, ProjectorWeights, VisionEncoderConfig, VisionEncoderWeights, VitStageDump,
    interpolate_pos_embed_2d, patch_embed_compute, pixel_shuffle,
};
use crate::model::vision_encoder_gpu::{MAX_VIT_TOKENS, VisionGpuEncode};
use crate::model::weights::MmapWeight;
use crate::session::CeraError;
use crate::tensor::DType;

pub use crate::backend::hexagon::HexagonWeightFormat;
/// The ViT emits one op over all tokens (see `dispatch::TokenTile`); only the
/// Whisper encoder tiles. Every `dispatch::*` call here passes this constant.
const VIT_TILE: TokenTile = TokenTile::Whole;

pub use crate::backend::hexagon::types::HexagonWeightDesc as HexagonVitWeightDesc;

/// Whether `n_tokens` rows fit the fused Q/K/V kernel's VTCM layout. That
/// kernel cannot chunk its activation rows, so past this the three projections
/// run on their own, each chunking itself.
fn qkv_fits_fused(dtype: HtpDataType, cols: usize, n_tokens: usize, threads: u32) -> bool {
    n_tokens
        <= crate::backend::hexagon::mm_hvx_fused_nx_max_rows(
            dtype,
            cols,
            threads,
            dispatch::VTCM_BUDGET,
        )
}

/// Weight offsets for one ViT block in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonVitBlockOffsets {
    pub ln1_w_off: usize,
    pub ln1_b_off: usize,
    pub q_w: HexagonVitWeightDesc,
    pub q_b_off: usize,
    pub k_w: HexagonVitWeightDesc,
    pub k_b_off: usize,
    pub v_w: HexagonVitWeightDesc,
    pub v_b_off: usize,
    pub o_w: HexagonVitWeightDesc,
    pub o_b_off: usize,
    pub ln2_w_off: usize,
    pub ln2_b_off: usize,
    pub ffn_up_w: HexagonVitWeightDesc,
    pub ffn_up_b_off: usize,
    pub ffn_down_w: HexagonVitWeightDesc,
    pub ffn_down_b_off: usize,
}

/// Offsets for the optional projector layers in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonVitProjectorOffsets {
    pub mm1_w: HexagonVitWeightDesc,
    pub mm1_b_off: usize,
    pub mm2_w: HexagonVitWeightDesc,
    pub mm2_b_off: usize,
}

/// Weight buffer layout offsets across all ViT blocks, post-norm, and projector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonVitWeightOffsets {
    pub blocks: Vec<HexagonVitBlockOffsets>,
    pub post_ln_w_off: usize,
    pub post_ln_b_off: usize,
    pub projector: Option<HexagonVitProjectorOffsets>,
    pub total_bytes: usize,
}

impl HexagonVitWeightOffsets {
    /// Compute memory layout and byte offsets for all ViT weights in shared rpcmem.
    pub fn plan(weights: &VisionEncoderWeights) -> Result<Self, CeraError> {
        let mut cur_off = 0;

        let plan_vec_f32 = |cur_off: &mut usize, len: usize| -> usize {
            let off = *cur_off;
            *cur_off += align128(len * 4);
            off
        };

        let plan_linear =
            |cur_off: &mut usize, w: &MmapWeight| -> Result<HexagonVitWeightDesc, CeraError> {
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
                    other => {
                        return Err(CeraError::Backend(format!(
                            "Hexagon ViT linear weight requires Q8_0 or Q4_0, got {other:?}"
                        )));
                    }
                };
                *cur_off += align128(sz);
                Ok(HexagonVitWeightDesc {
                    offset: off,
                    size_bytes: sz,
                    format: fmt,
                    rows: w.rows,
                    cols: w.cols,
                })
            };

        let mut block_offsets = Vec::with_capacity(weights.blocks.len());
        for blk in &weights.blocks {
            let ln1_w_off = plan_vec_f32(&mut cur_off, blk.ln1_w.len());
            let ln1_b_off = plan_vec_f32(&mut cur_off, blk.ln1_b.len());
            let q_w = plan_linear(&mut cur_off, &blk.q_w)?;
            let q_b_off = plan_vec_f32(&mut cur_off, blk.q_b.len());
            let k_w = plan_linear(&mut cur_off, &blk.k_w)?;
            let k_b_off = plan_vec_f32(&mut cur_off, blk.k_b.len());
            let v_w = plan_linear(&mut cur_off, &blk.v_w)?;
            let v_b_off = plan_vec_f32(&mut cur_off, blk.v_b.len());
            let o_w = plan_linear(&mut cur_off, &blk.o_w)?;
            let o_b_off = plan_vec_f32(&mut cur_off, blk.o_b.len());
            let ln2_w_off = plan_vec_f32(&mut cur_off, blk.ln2_w.len());
            let ln2_b_off = plan_vec_f32(&mut cur_off, blk.ln2_b.len());
            let ffn_up_w = plan_linear(&mut cur_off, &blk.ffn_up_w)?;
            let ffn_up_b_off = plan_vec_f32(&mut cur_off, blk.ffn_up_b.len());
            let ffn_down_w = plan_linear(&mut cur_off, &blk.ffn_down_w)?;
            let ffn_down_b_off = plan_vec_f32(&mut cur_off, blk.ffn_down_b.len());

            block_offsets.push(HexagonVitBlockOffsets {
                ln1_w_off,
                ln1_b_off,
                q_w,
                q_b_off,
                k_w,
                k_b_off,
                v_w,
                v_b_off,
                o_w,
                o_b_off,
                ln2_w_off,
                ln2_b_off,
                ffn_up_w,
                ffn_up_b_off,
                ffn_down_w,
                ffn_down_b_off,
            });
        }

        let post_ln_w_off = plan_vec_f32(&mut cur_off, weights.post_ln_w.len());
        let post_ln_b_off = plan_vec_f32(&mut cur_off, weights.post_ln_b.len());

        let projector = if (weights.projector.mm1_w.dtype == DType::Q8_0
            || weights.projector.mm1_w.dtype == DType::Q4_0)
            && (weights.projector.mm2_w.dtype == DType::Q8_0
                || weights.projector.mm2_w.dtype == DType::Q4_0)
        {
            let mm1_w = plan_linear(&mut cur_off, &weights.projector.mm1_w)?;
            let mm1_b_off = plan_vec_f32(&mut cur_off, weights.projector.mm1_b.len());
            let mm2_w = plan_linear(&mut cur_off, &weights.projector.mm2_w)?;
            let mm2_b_off = plan_vec_f32(&mut cur_off, weights.projector.mm2_b.len());
            Some(HexagonVitProjectorOffsets {
                mm1_w,
                mm1_b_off,
                mm2_w,
                mm2_b_off,
            })
        } else {
            None
        };

        Ok(HexagonVitWeightOffsets {
            blocks: block_offsets,
            post_ln_w_off,
            post_ln_b_off,
            projector,
            total_bytes: cur_off,
        })
    }
}

/// Layout offsets for runtime intermediate activations inside `scratch_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonVitScratchOffsets {
    pub tokens_off: usize,
    pub pre_norm_off: usize,
    pub q_off: usize,
    pub k_off: usize,
    pub v_off: usize,
    pub k_f16_off: usize,
    pub v_f16_off: usize,
    pub attn_out_off: usize,
    pub proj_out_off: usize,
    pub ffn_mid_off: usize,
    pub ffn_out_off: usize,
    pub proj_mid_off: usize,
    pub proj_final_off: usize,
    /// Scratch for the tanh GELU's intermediate: a few dozen rows wide enough
    /// for the feed-forward activation (`gelu_tmp_bytes` of them at `gelu_tmp_off`).
    pub gelu_tmp_off: usize,
    pub gelu_tmp_bytes: usize,
    pub total_bytes: usize,
}

impl HexagonVitScratchOffsets {
    /// Compute scratch allocation sized for up to `MAX_VIT_TOKENS` patches.
    pub fn new(n_embd: usize, n_ff: usize) -> Self {
        let token_bytes = align128(MAX_VIT_TOKENS * n_embd * 4);
        let token_f16_bytes = align128(MAX_VIT_TOKENS * n_embd * 2);
        let ffn_mid_bytes = align128(MAX_VIT_TOKENS * n_ff * 4);

        let mut off = 0;
        let tokens_off = off;
        off += token_bytes;

        let pre_norm_off = off;
        off += token_bytes;

        let q_off = off;
        off += token_bytes;

        let k_off = off;
        off += token_bytes;

        let v_off = off;
        off += token_bytes;

        let k_f16_off = off;
        off += token_f16_bytes;

        let v_f16_off = off;
        off += token_f16_bytes;

        let attn_out_off = off;
        off += token_bytes;

        let proj_out_off = off;
        off += token_bytes;

        let ffn_mid_off = off;
        off += ffn_mid_bytes;

        let ffn_out_off = off;
        off += token_bytes;

        let gelu_tmp_off = off;
        let gelu_tmp_bytes = align128(dispatch::GELU_TMP_ROWS * n_ff * 4);
        off += gelu_tmp_bytes;

        // Projector runs after ViT blocks complete. Reusing ffn_mid_off and
        // attn_out_off avoids inflating rpcmem scratch footprint.
        let proj_mid_off = ffn_mid_off;
        let proj_final_off = attn_out_off;

        Self {
            tokens_off,
            pre_norm_off,
            q_off,
            k_off,
            v_off,
            k_f16_off,
            v_f16_off,
            attn_out_off,
            proj_out_off,
            ffn_mid_off,
            ffn_out_off,
            proj_mid_off,
            proj_final_off,
            gelu_tmp_off,
            gelu_tmp_bytes,
            total_bytes: off,
        }
    }
}

/// Hexagon NPU Vision Transformer encoder.
pub struct HexagonVisionEncoder {
    device: Arc<Mutex<HexagonDevice>>,
    config: VisionEncoderConfig,
    weights_buf: RpcmemBuffer,
    weights_offsets: HexagonVitWeightOffsets,
    scratch_buf: Mutex<RpcmemBuffer>,
    scratch_offsets: HexagonVitScratchOffsets,
    patch_embed: PatchEmbedWeights,
    position_embed: Arc<[f32]>,
    projector: ProjectorWeights,
}

// SAFETY: Synchronization across threads is enforced by the device and scratch Mutex locks.
unsafe impl Send for HexagonVisionEncoder {}
unsafe impl Sync for HexagonVisionEncoder {}

impl HexagonVisionEncoder {
    /// Initialize a Hexagon ViT encoder from loaded model weights.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        weights: &VisionEncoderWeights,
    ) -> Result<Self, CeraError> {
        let cfg = &weights.config;
        let scratch_offsets = HexagonVitScratchOffsets::new(cfg.n_embd, cfg.n_ff);
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true)?;

        // Plan weights buffer layout
        let weights_offsets = HexagonVitWeightOffsets::plan(weights)?;
        let mut weights_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), weights_offsets.total_bytes, true)?;

        // Populate weights buffer
        let copy_vec_f32 = |dst: &mut [u8], off: usize, src: &[f32]| {
            let bytes = bytemuck::cast_slice(src);
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        };

        let copy_linear = |dst: &mut [u8],
                           desc: HexagonVitWeightDesc,
                           w: &MmapWeight|
         -> Result<(), CeraError> {
            let dst_slice = &mut dst[desc.offset..desc.offset + desc.size_bytes];
            match desc.format {
                HexagonWeightFormat::RepackedQ8_0 => {
                    repack_q8_0(w.data(), w.cols, w.rows, dst_slice)
                        .map_err(|e| CeraError::Backend(format!("ViT repack Q8_0 failed: {e}")))?;
                }
                HexagonWeightFormat::RepackedQ4_0 => {
                    repack_q4_0(w.data(), w.cols, w.rows, dst_slice)
                        .map_err(|e| CeraError::Backend(format!("ViT repack Q4_0 failed: {e}")))?;
                }
            }
            Ok(())
        };

        let buf_slice = weights_buf.as_mut_slice();
        for (blk_weights, blk_offs) in weights.blocks.iter().zip(weights_offsets.blocks.iter()) {
            copy_vec_f32(buf_slice, blk_offs.ln1_w_off, &blk_weights.ln1_w);
            copy_vec_f32(buf_slice, blk_offs.ln1_b_off, &blk_weights.ln1_b);
            copy_linear(buf_slice, blk_offs.q_w, &blk_weights.q_w)?;
            copy_vec_f32(buf_slice, blk_offs.q_b_off, &blk_weights.q_b);
            copy_linear(buf_slice, blk_offs.k_w, &blk_weights.k_w)?;
            copy_vec_f32(buf_slice, blk_offs.k_b_off, &blk_weights.k_b);
            copy_linear(buf_slice, blk_offs.v_w, &blk_weights.v_w)?;
            copy_vec_f32(buf_slice, blk_offs.v_b_off, &blk_weights.v_b);
            copy_linear(buf_slice, blk_offs.o_w, &blk_weights.o_w)?;
            copy_vec_f32(buf_slice, blk_offs.o_b_off, &blk_weights.o_b);
            copy_vec_f32(buf_slice, blk_offs.ln2_w_off, &blk_weights.ln2_w);
            copy_vec_f32(buf_slice, blk_offs.ln2_b_off, &blk_weights.ln2_b);
            copy_linear(buf_slice, blk_offs.ffn_up_w, &blk_weights.ffn_up_w)?;
            copy_vec_f32(buf_slice, blk_offs.ffn_up_b_off, &blk_weights.ffn_up_b);
            copy_linear(buf_slice, blk_offs.ffn_down_w, &blk_weights.ffn_down_w)?;
            copy_vec_f32(buf_slice, blk_offs.ffn_down_b_off, &blk_weights.ffn_down_b);
        }

        copy_vec_f32(buf_slice, weights_offsets.post_ln_w_off, &weights.post_ln_w);
        copy_vec_f32(buf_slice, weights_offsets.post_ln_b_off, &weights.post_ln_b);

        if let Some(proj_offs) = weights_offsets.projector {
            copy_linear(buf_slice, proj_offs.mm1_w, &weights.projector.mm1_w)?;
            copy_vec_f32(buf_slice, proj_offs.mm1_b_off, &weights.projector.mm1_b);
            copy_linear(buf_slice, proj_offs.mm2_w, &weights.projector.mm2_w)?;
            copy_vec_f32(buf_slice, proj_offs.mm2_b_off, &weights.projector.mm2_b);
        }

        weights_buf.flush_cpu_cache(0, weights_offsets.total_bytes);

        Ok(Self {
            device,
            config: weights.config.clone(),
            weights_buf,
            weights_offsets,
            scratch_buf: Mutex::new(scratch_buf),
            scratch_offsets,
            patch_embed: PatchEmbedWeights {
                conv_w: weights.patch_embed.conv_w.clone(),
                conv_b: weights.patch_embed.conv_b.clone(),
            },
            position_embed: Arc::from(weights.position_embed.as_slice()),
            projector: ProjectorWeights {
                mm1_w: weights.projector.mm1_w.clone(),
                mm1_b: weights.projector.mm1_b.clone(),
                mm2_w: weights.projector.mm2_w.clone(),
                mm2_b: weights.projector.mm2_b.clone(),
            },
        })
    }

    /// Dispatch broadcast bias add: `dst[i] += bias[i]`.
    fn dispatch_bias_add(
        session: &mut HexagonQueueSession,
        dst_ti: u16,
        weights: &RpcmemBuffer,
        b_offset: usize,
        rows: usize,
    ) -> Result<(), CeraError> {
        let b_bytes = rows * 4;
        let b_ne = [rows as u32, 1, 1, 1];
        let b_nb = [4, b_bytes as u32, b_bytes as u32, b_bytes as u32];
        let b_ti = session.add_tensor(
            weights,
            b_offset,
            b_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            b_ne,
            b_nb,
        )?;

        let add_params = [0i32; 16];
        let add_kparams = build_binary_kernel_params(
            rows,
            rows,
            1,
            1,
            1,
            4,
            dispatch::VTCM_BUDGET,
            session.dsp_threads(),
        );
        session
            .enqueue_op(
                HtpOpCode::Add as u32,
                &[dst_ti, b_ti],
                &[dst_ti],
                add_params,
                add_kparams,
            )
            .map_err(|e| CeraError::Backend(format!("linear_bias_add: {e}")))?;

        Ok(())
    }

    /// Dispatch fused linear QKV matmuls via MulMatNx followed by bias additions.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_qkv_m(
        session: &mut HexagonQueueSession,
        x: &RpcmemBuffer,
        x_offset: usize,
        weights: &RpcmemBuffer,
        q_w: HexagonVitWeightDesc,
        q_b_off: usize,
        k_w: HexagonVitWeightDesc,
        k_b_off: usize,
        v_w: HexagonVitWeightDesc,
        v_b_off: usize,
        dst: &RpcmemBuffer,
        q_dst_off: usize,
        k_dst_off: usize,
        v_dst_off: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let fused_dtype = match q_w.format {
            HexagonWeightFormat::RepackedQ8_0 => HtpDataType::Q8_0,
            HexagonWeightFormat::RepackedQ4_0 => HtpDataType::Q4_0,
        };
        // The fused kernel cannot chunk its activation rows, so past what its
        // VTCM layout holds the three projections run on their own (each
        // chunks itself): a few hundred tokens of a large image.
        let fits_fused = qkv_fits_fused(fused_dtype, q_w.cols, n_tokens, session.dsp_threads());
        if !(fits_fused
            && q_w.cols == k_w.cols
            && k_w.cols == v_w.cols
            && q_w.rows == k_w.rows
            && k_w.rows == v_w.rows
            && q_w.format == k_w.format
            && k_w.format == v_w.format)
        {
            dispatch::linear_m(
                session, x, x_offset, weights, q_w, q_b_off, dst, q_dst_off, n_tokens, VIT_TILE,
            )?;
            dispatch::linear_m(
                session, x, x_offset, weights, k_w, k_b_off, dst, k_dst_off, n_tokens, VIT_TILE,
            )?;
            dispatch::linear_m(
                session, x, x_offset, weights, v_w, v_b_off, dst, v_dst_off, n_tokens, VIT_TILE,
            )?;
            return Ok(());
        }

        let (w_dtype, block_bytes, tile_size) = match q_w.format {
            HexagonWeightFormat::RepackedQ8_0 => (HtpDataType::Q8_0, 34usize, 32 * 34usize),
            HexagonWeightFormat::RepackedQ4_0 => (HtpDataType::Q4_0, 18usize, 32 * 18usize),
        };

        let ne0 = q_w.cols;
        let ne1 = q_w.rows;
        let tiled_row_bytes = ne0.div_ceil(32) * tile_size;
        let w_tot = ne1.div_ceil(32) * tiled_row_bytes;
        let w_nb = [
            block_bytes as u32,
            tiled_row_bytes as u32,
            w_tot as u32,
            w_tot as u32,
        ];
        let w_ne = [ne0 as u32, ne1 as u32, 1, 1];

        let q_w_ti = session.add_tensor(
            weights,
            q_w.offset,
            q_w.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            w_ne,
            w_nb,
        )?;
        let k_w_ti = session.add_tensor(
            weights,
            k_w.offset,
            k_w.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            w_ne,
            w_nb,
        )?;
        let v_w_ti = session.add_tensor(
            weights,
            v_w.offset,
            v_w.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            w_ne,
            w_nb,
        )?;

        let x_bytes = n_tokens * ne0 * 4;
        let x_ne = [ne0 as u32, n_tokens as u32, 1, 1];
        let x_nb = [4, (ne0 * 4) as u32, x_bytes as u32, x_bytes as u32];
        let x_ti = session.add_tensor(
            x,
            x_offset,
            x_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            x_ne,
            x_nb,
        )?;

        let dst_bytes = n_tokens * ne1 * 4;
        let dst_ne = [ne1 as u32, n_tokens as u32, 1, 1];
        let dst_nb = [4, (ne1 * 4) as u32, dst_bytes as u32, dst_bytes as u32];

        let q_dst_ti = session.add_tensor(
            dst,
            q_dst_off,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            dst_ne,
            dst_nb,
        )?;
        let k_dst_ti = session.add_tensor(
            dst,
            k_dst_off,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            dst_ne,
            dst_nb,
        )?;
        let v_dst_ti = session.add_tensor(
            dst,
            v_dst_off,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            dst_ne,
            dst_nb,
        )?;

        let mut kparams = crate::backend::hexagon::build_mul_mat_kernel_params(
            w_dtype,
            ne0,
            n_tokens as u32,
            1,
            ne1 * 4,
            session.dsp_threads(),
            dispatch::VTCM_BUDGET,
        );
        // `fits_fused` above keeps the rows inside what the fused layout holds,
        // which implies the plain layout needs no chunk; the fused kernel could
        // not honour one.
        debug_assert_eq!(kparams[2], 0, "fused QKV must not need a row chunk");
        kparams[0] = 5; // HTP_MM_KERNEL_HVX_QUANT_ROW
        kparams[17] = 3; // n_weights

        let params = [0i32; 16];
        session
            .enqueue_op(
                HtpOpCode::MulMatNx as u32,
                &[q_w_ti, k_w_ti, v_w_ti, x_ti],
                &[q_dst_ti, k_dst_ti, v_dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("dispatch_linear_qkv_m: {e}")))?;

        // Broadcast bias adds
        Self::dispatch_bias_add(session, q_dst_ti, weights, q_b_off, ne1)?;
        Self::dispatch_bias_add(session, k_dst_ti, weights, k_b_off, ne1)?;
        Self::dispatch_bias_add(session, v_dst_ti, weights, v_b_off, ne1)?;

        session.end_group()
    }

    /// Dispatch unmasked multi-head self-attention on DSP via FlashAttnExt.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_self_attention(
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

        let kparams = crate::backend::hexagon::build_flash_attn_kernel_params(
            head_dim,
            n_heads,
            n_heads,
            n_tokens,
            n_tokens,
            scale,
            session.dsp_threads(),
            false,
        );

        session
            .enqueue_op(
                HtpOpCode::FlashAttnExt as u32,
                &[q_ti, k_ti, v_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("dispatch_self_attention: {e}")))?;
        session.end_group()
    }
}

impl VisionGpuEncode for HexagonVisionEncoder {
    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        self.encode_with(pixels, grid_w, grid_h, None, None)
    }
}

impl HexagonVisionEncoder {
    /// The tokens after the first `n_blocks` ViT blocks (before the final
    /// LayerNorm, pooling and projector), for the probe to compare against the
    /// CPU encoder block by block.
    #[doc(hidden)]
    pub fn debug_blocks(
        &self,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
        n_blocks: usize,
    ) -> Result<Vec<f32>> {
        self.encode_with(pixels, grid_w, grid_h, Some(n_blocks), None)
    }

    /// The first block's intermediates as the device left them, in the layouts
    /// of [`VisionEncoderWeights::debug_first_block`].
    #[doc(hidden)]
    pub fn debug_first_block(
        &self,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
    ) -> Result<VitStageDump> {
        let mut dump = VitStageDump::default();
        self.encode_with(pixels, grid_w, grid_h, Some(1), Some(&mut dump))?;
        Ok(dump)
    }

    /// The whole encoder, or with `stop_after_blocks` just the blocks and the
    /// raw `[patches, n_embd]` tokens they leave.
    fn encode_with(
        &self,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
        stop_after_blocks: Option<usize>,
        dump: Option<&mut VitStageDump>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        ensure!(grid_w > 0 && grid_h > 0, "grid dims must be > 0");
        ensure!(
            cfg.scale_factor > 0,
            "vision encoder config has scale_factor=0"
        );
        ensure!(
            grid_w.is_multiple_of(cfg.scale_factor) && grid_h.is_multiple_of(cfg.scale_factor),
            "grid {grid_w}x{grid_h} not divisible by scale_factor ({})",
            cfg.scale_factor,
        );

        let p = cfg.patch_size;
        let n_patches = grid_w * grid_h;
        let expected_pixels = grid_w
            .checked_mul(p)
            .and_then(|w| w.checked_mul(grid_h))
            .and_then(|wh| wh.checked_mul(p))
            .and_then(|whp| whp.checked_mul(3))
            .ok_or_else(|| anyhow!("encode_image: integer overflow in image dimensions"))?;
        ensure!(
            pixels.len() == expected_pixels,
            "encode_image: pixels length {} != expected {}",
            pixels.len(),
            expected_pixels
        );
        ensure!(
            n_patches <= MAX_VIT_TOKENS,
            "encode_image: {n_patches} patches exceeds MAX_VIT_TOKENS ({MAX_VIT_TOKENS})"
        );

        // 1. Patch embedding on host CPU
        let mut tokens = patch_embed_compute(pixels, &self.patch_embed, cfg, grid_w, grid_h);

        // 2. Add interpolated position embeddings
        let trained_side = (cfg.n_trained_patches as f64).sqrt().round() as usize;
        ensure!(
            trained_side * trained_side == cfg.n_trained_patches,
            "non-square trained pos-embed grid is not supported"
        );
        let pos: std::borrow::Cow<'_, [f32]> = if grid_w == trained_side && grid_h == trained_side {
            std::borrow::Cow::Borrowed(self.position_embed.as_ref())
        } else {
            std::borrow::Cow::Owned(interpolate_pos_embed_2d(
                &self.position_embed,
                trained_side,
                trained_side,
                grid_h,
                grid_w,
                cfg.n_embd,
            ))
        };
        ensure!(
            tokens.len() == pos.len(),
            "encode_image: tokens length {} != pos embedding length {}",
            tokens.len(),
            pos.len()
        );
        for (t, p) in tokens.iter_mut().zip(pos.iter()) {
            *t += *p;
        }

        let mut dump = dump;
        if let Some(d) = dump.as_deref_mut() {
            d.x0 = tokens.clone();
        }

        // 3. Hexagon NPU ViT blocks execution
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let head_dim = n_embd / n_head;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut dev_guard = self.device.lock_or_recover();
        let mut scratch_guard = self.scratch_buf.lock_or_recover();

        let so = self.scratch_offsets;
        let tokens_bytes = n_patches * n_embd * 4;

        // Copy input tokens to rpcmem scratch
        {
            let scratch_slice =
                &mut scratch_guard.as_mut_slice()[so.tokens_off..so.tokens_off + tokens_bytes];
            scratch_slice.copy_from_slice(bytemuck::cast_slice(&tokens));
        }
        scratch_guard.flush_cpu_cache(so.tokens_off, tokens_bytes);

        let session = dev_guard.queue_session_mut();
        session.drop_pending_batch();

        let run_vit_blocks = |session: &mut HexagonQueueSession,
                              scratch_guard: &mut RpcmemBuffer|
         -> Result<(), CeraError> {
            for blk in self
                .weights_offsets
                .blocks
                .iter()
                .take(stop_after_blocks.unwrap_or(usize::MAX))
            {
                // Pre-attention LayerNorm: tokens -> pre_norm
                dispatch::layer_norm(
                    session,
                    LayerNormArgs {
                        src: scratch_guard,
                        src_offset: so.tokens_off,
                        dst: scratch_guard,
                        dst_offset: so.pre_norm_off,
                        weights: &self.weights_buf,
                        w_offset: blk.ln1_w_off,
                        b_offset: blk.ln1_b_off,
                        shape: TokenShape {
                            dim: n_embd,
                            n_tokens: n_patches,
                        },
                        eps: cfg.eps,
                        tile: VIT_TILE,
                    },
                )?;

                // Fused Q, K, V linear projections via MulMatNx
                Self::dispatch_linear_qkv_m(
                    session,
                    scratch_guard,
                    so.pre_norm_off,
                    &self.weights_buf,
                    blk.q_w,
                    blk.q_b_off,
                    blk.k_w,
                    blk.k_b_off,
                    blk.v_w,
                    blk.v_b_off,
                    scratch_guard,
                    so.q_off,
                    so.k_off,
                    so.v_off,
                    n_patches,
                )?;

                // Convert K and V from F32 to F16 in rpcmem for FlashAttnExt
                dispatch::cpy_f32_to_f16(
                    session,
                    scratch_guard,
                    so.k_off,
                    scratch_guard,
                    so.k_f16_off,
                    TokenShape {
                        dim: n_embd,
                        n_tokens: n_patches,
                    },
                )?;
                dispatch::cpy_f32_to_f16(
                    session,
                    scratch_guard,
                    so.v_off,
                    scratch_guard,
                    so.v_f16_off,
                    TokenShape {
                        dim: n_embd,
                        n_tokens: n_patches,
                    },
                )?;

                // Full on-NPU Flash Attention
                Self::dispatch_self_attention(
                    session,
                    scratch_guard,
                    so.q_off,
                    scratch_guard,
                    so.k_f16_off,
                    scratch_guard,
                    so.v_f16_off,
                    scratch_guard,
                    so.attn_out_off,
                    head_dim,
                    n_head,
                    n_patches,
                    scale,
                )?;

                // Out projection + bias: attn_out -> proj_out
                dispatch::linear_m(
                    session,
                    scratch_guard,
                    so.attn_out_off,
                    &self.weights_buf,
                    blk.o_w,
                    blk.o_b_off,
                    scratch_guard,
                    so.proj_out_off,
                    n_patches,
                    VIT_TILE,
                )?;

                // Residual add: tokens += proj_out
                dispatch::add_residual(
                    session,
                    scratch_guard,
                    so.tokens_off,
                    scratch_guard,
                    so.proj_out_off,
                    TokenShape {
                        dim: n_embd,
                        n_tokens: n_patches,
                    },
                    VIT_TILE,
                )?;

                // Pre-MLP LayerNorm: tokens -> pre_norm
                dispatch::layer_norm(
                    session,
                    LayerNormArgs {
                        src: scratch_guard,
                        src_offset: so.tokens_off,
                        dst: scratch_guard,
                        dst_offset: so.pre_norm_off,
                        weights: &self.weights_buf,
                        w_offset: blk.ln2_w_off,
                        b_offset: blk.ln2_b_off,
                        shape: TokenShape {
                            dim: n_embd,
                            n_tokens: n_patches,
                        },
                        eps: cfg.eps,
                        tile: VIT_TILE,
                    },
                )?;

                // FFN up projection: pre_norm -> ffn_mid
                dispatch::linear_m(
                    session,
                    scratch_guard,
                    so.pre_norm_off,
                    &self.weights_buf,
                    blk.ffn_up_w,
                    blk.ffn_up_b_off,
                    scratch_guard,
                    so.ffn_mid_off,
                    n_patches,
                    VIT_TILE,
                )?;

                // GELU activation on ffn_mid
                dispatch::gelu_tanh(
                    session,
                    scratch_guard,
                    so.ffn_mid_off,
                    scratch_guard,
                    so.gelu_tmp_off,
                    dispatch::gelu_tmp_rows(so.gelu_tmp_bytes, cfg.n_ff)?,
                    TokenShape {
                        dim: cfg.n_ff,
                        n_tokens: n_patches,
                    },
                )?;

                // FFN down projection: ffn_mid -> ffn_out
                dispatch::linear_m(
                    session,
                    scratch_guard,
                    so.ffn_mid_off,
                    &self.weights_buf,
                    blk.ffn_down_w,
                    blk.ffn_down_b_off,
                    scratch_guard,
                    so.ffn_out_off,
                    n_patches,
                    VIT_TILE,
                )?;

                // Residual add: tokens += ffn_out
                dispatch::add_residual(
                    session,
                    scratch_guard,
                    so.tokens_off,
                    scratch_guard,
                    so.ffn_out_off,
                    TokenShape {
                        dim: n_embd,
                        n_tokens: n_patches,
                    },
                    VIT_TILE,
                )?;
            }

            // Post-LN on tokens (the debug entry stops before it).
            if stop_after_blocks.is_none() {
                dispatch::layer_norm(
                    session,
                    LayerNormArgs {
                        src: scratch_guard,
                        src_offset: so.tokens_off,
                        dst: scratch_guard,
                        dst_offset: so.tokens_off,
                        weights: &self.weights_buf,
                        w_offset: self.weights_offsets.post_ln_w_off,
                        b_offset: self.weights_offsets.post_ln_b_off,
                        shape: TokenShape {
                            dim: n_embd,
                            n_tokens: n_patches,
                        },
                        eps: cfg.eps,
                        tile: VIT_TILE,
                    },
                )?;
            }

            // Execute all queued operations on DSP
            session.flush()?;
            Ok(())
        };

        let vit_res = run_vit_blocks(session, &mut scratch_guard);
        if vit_res.is_err() {
            session.drop_pending_batch();
        }
        vit_res.map_err(|e| anyhow!("Hexagon ViT forward pass failed: {e}"))?;

        // Read normalized tokens back to host
        scratch_guard.invalidate_cpu_cache(so.tokens_off, tokens_bytes);
        let normed_slice: &[f32] = bytemuck::cast_slice(
            &scratch_guard.as_slice()[so.tokens_off..so.tokens_off + tokens_bytes],
        );

        if stop_after_blocks.is_some() {
            if let Some(d) = dump {
                let all: &[f32] = bytemuck::cast_slice(scratch_guard.as_slice());
                let region = |off: usize, floats: usize| all[off / 4..off / 4 + floats].to_vec();
                let (n, ff) = (n_patches * n_embd, n_patches * cfg.n_ff);
                d.q = region(so.q_off, n);
                d.k = region(so.k_off, n);
                d.v = region(so.v_off, n);
                d.attn_out = region(so.attn_out_off, n);
                d.attn_proj = region(so.proj_out_off, n);
                d.ln2 = region(so.pre_norm_off, n);
                d.ffn_mid = region(so.ffn_mid_off, ff);
                d.ffn_out = region(so.ffn_out_off, n);
                d.tokens = normed_slice.to_vec();
            }
            return Ok(normed_slice.to_vec());
        }

        // 4. Pixel-shuffle pool over dynamic grid
        let pooled = pixel_shuffle(normed_slice, cfg, grid_w, grid_h);

        // 5. 2-layer MLP Projector to LLM embedding dimension
        if let Some(proj_offs) = self.weights_offsets.projector {
            let in_dim = proj_offs.mm1_w.cols;
            let mid_dim = proj_offs.mm1_w.rows;
            let out_dim = cfg.projection_dim;
            if in_dim > 0 && !pooled.is_empty() && pooled.len().is_multiple_of(in_dim) {
                let n_tokens = pooled.len() / in_dim;
                let in_bytes = pooled.len() * 4;
                let out_bytes = n_tokens * out_dim * 4;

                let session = dev_guard.queue_session_mut();
                session.drop_pending_batch();

                scratch_guard.as_mut_slice()[so.tokens_off..so.tokens_off + in_bytes]
                    .copy_from_slice(bytemuck::cast_slice(&pooled));
                scratch_guard.flush_cpu_cache(so.tokens_off, in_bytes);

                let proj_res = (|| -> Result<(), CeraError> {
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.tokens_off,
                        &self.weights_buf,
                        proj_offs.mm1_w,
                        proj_offs.mm1_b_off,
                        &scratch_guard,
                        so.proj_mid_off,
                        n_tokens,
                        VIT_TILE,
                    )?;
                    dispatch::gelu_tanh(
                        session,
                        &scratch_guard,
                        so.proj_mid_off,
                        &scratch_guard,
                        so.gelu_tmp_off,
                        dispatch::gelu_tmp_rows(so.gelu_tmp_bytes, mid_dim)?,
                        TokenShape {
                            dim: mid_dim,
                            n_tokens,
                        },
                    )?;
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.proj_mid_off,
                        &self.weights_buf,
                        proj_offs.mm2_w,
                        proj_offs.mm2_b_off,
                        &scratch_guard,
                        so.proj_final_off,
                        n_tokens,
                        VIT_TILE,
                    )?;
                    session.flush()?;
                    Ok(())
                })();

                match proj_res {
                    Ok(()) => {
                        scratch_guard.invalidate_cpu_cache(so.proj_final_off, out_bytes);
                        let out_slice: &[f32] = bytemuck::cast_slice(
                            &scratch_guard.as_slice()
                                [so.proj_final_off..so.proj_final_off + out_bytes],
                        );
                        return Ok(out_slice.to_vec());
                    }
                    Err(e) => {
                        crate::backend::hexagon::hexagon_warn!(
                            "HexagonVisionEncoder: DSP projector failed ({e}), falling back to CPU"
                        );
                        session.drop_pending_batch();
                    }
                }
            }
        }

        drop(scratch_guard);
        drop(dev_guard);

        Ok(self.projector.forward(&pooled, cfg.projection_dim))
    }
}

/// Attempt to create a hardware-accelerated Hexagon NPU vision encoder.
///
/// Probes for Qualcomm DSP hardware, loading Unsigned PD skeleton libraries.
/// Returns `None` if Hexagon hardware or libraries are unavailable, falling
/// back cleanly to GPU or CPU execution.
pub fn try_hexagon_vision_encoder(
    weights: &VisionEncoderWeights,
) -> Option<Arc<dyn VisionGpuEncode>> {
    try_new_hexagon_vision_encoder(weights).map(|e| Arc::new(e) as Arc<dyn VisionGpuEncode>)
}

/// [`try_hexagon_vision_encoder`] with the concrete type, for the probe that
/// compares it with the CPU encoder block by block.
#[doc(hidden)]
pub fn try_new_hexagon_vision_encoder(
    weights: &VisionEncoderWeights,
) -> Option<HexagonVisionEncoder> {
    let context = HexagonContext::new()
        .inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonVisionEncoder", e);
        })
        .ok()?;

    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(HexagonArch::from_u32);

    let dev = match crate::backend::hexagon::probe_device(context.driver(), arch_override) {
        Ok(d) => d,
        Err(e) => {
            tracing::info!("HexagonVisionEncoder: DSP device unavailable ({e}), falling back");
            return None;
        }
    };
    let device = Arc::new(Mutex::new(dev));

    match HexagonVisionEncoder::new(Arc::clone(context.driver()), device, weights) {
        Ok(encoder) => Some(encoder),
        Err(e) => {
            crate::backend::hexagon::hexagon_error!("failed to create HexagonVisionEncoder: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The fused Q/K/V kernel is used up to its row limit and not one row past.
    #[test]
    fn the_fused_qkv_kernel_stops_at_its_row_limit() {
        for (dtype, cols) in [(HtpDataType::Q8_0, 768), (HtpDataType::Q4_0, 1152)] {
            let max = crate::backend::hexagon::mm_hvx_fused_nx_max_rows(
                dtype,
                cols,
                6,
                dispatch::VTCM_BUDGET,
            );
            assert!(max > 1, "{dtype:?} {cols}");
            assert!(qkv_fits_fused(dtype, cols, max, 6), "at the limit");
            assert!(!qkv_fits_fused(dtype, cols, max + 1, 6), "one past it");
        }
    }

    use crate::model::vision_encoder::VitBlockWeights;

    /// Tile policy pin: a tiled ViT would multiply the op count and change
    /// the emitted graph, so a flip must be deliberate. Pinned by behavior:
    /// 130 tokens are one Norm/Mul/Add triple.
    #[test]
    fn tile_policy_is_whole() {
        assert_eq!(dispatch::testing::layer_norm_op_count(VIT_TILE), 3);
    }

    #[test]
    fn test_scratch_offsets_alignment_and_bounds() {
        let n_embd = 768;
        let n_ff = 3072;
        let so = HexagonVitScratchOffsets::new(n_embd, n_ff);

        assert_eq!(so.tokens_off, 0);
        assert!(so.pre_norm_off >= MAX_VIT_TOKENS * n_embd * 4);
        assert_eq!(so.tokens_off % 128, 0);
        assert_eq!(so.pre_norm_off % 128, 0);
        assert_eq!(so.q_off % 128, 0);
        assert_eq!(so.k_off % 128, 0);
        assert_eq!(so.v_off % 128, 0);
        assert_eq!(so.k_f16_off % 128, 0);
        assert_eq!(so.v_f16_off % 128, 0);
        assert_eq!(so.attn_out_off % 128, 0);
        assert_eq!(so.proj_out_off % 128, 0);
        assert_eq!(so.ffn_mid_off % 128, 0);
        assert_eq!(so.ffn_out_off % 128, 0);
        assert_eq!(so.proj_mid_off % 128, 0);
        assert_eq!(so.proj_final_off % 128, 0);
        assert!(so.total_bytes > so.proj_final_off);
        assert!(so.total_bytes < 64 * 1024 * 1024);
    }

    #[test]
    fn test_scratch_offsets_smaller_dimensions() {
        let n_embd = 512;
        let n_ff = 2048;
        let so = HexagonVitScratchOffsets::new(n_embd, n_ff);

        assert_eq!(so.tokens_off, 0);
        assert!(so.pre_norm_off >= MAX_VIT_TOKENS * n_embd * 4);
        assert_eq!(so.tokens_off % 128, 0);
        assert_eq!(so.pre_norm_off % 128, 0);
        assert_eq!(so.q_off % 128, 0);
        assert_eq!(so.k_off % 128, 0);
        assert_eq!(so.v_off % 128, 0);
        assert_eq!(so.k_f16_off % 128, 0);
        assert_eq!(so.v_f16_off % 128, 0);
        assert_eq!(so.attn_out_off % 128, 0);
        assert_eq!(so.proj_out_off % 128, 0);
        assert_eq!(so.ffn_mid_off % 128, 0);
        assert_eq!(so.ffn_out_off % 128, 0);
        assert_eq!(so.proj_mid_off % 128, 0);
        assert_eq!(so.proj_final_off % 128, 0);
        assert!(so.total_bytes > so.proj_final_off);
        assert!(so.total_bytes < 32 * 1024 * 1024);
    }

    #[test]
    fn test_try_hexagon_vision_encoder_returns_none_for_unloadable_weights() {
        let cfg = VisionEncoderConfig {
            n_layer: 1,
            n_embd: 64,
            n_ff: 128,
            n_head: 2,
            eps: 1e-5,
            image_size: 32,
            patch_size: 16,
            n_trained_patches: 4,
            projection_dim: 32,
            scale_factor: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 32 * 32,
            image_max_pixels: 32 * 32,
        };
        let weights = VisionEncoderWeights {
            config: cfg,
            patch_embed: PatchEmbedWeights {
                conv_w: vec![0.0f32; 3 * 16 * 16 * 64],
                conv_b: vec![0.0f32; 64],
            },
            position_embed: vec![0.0f32; 4 * 64],
            blocks: vec![VitBlockWeights {
                ln1_w: vec![1.0f32; 64],
                ln1_b: vec![0.0f32; 64],
                q_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                q_b: vec![0.0f32; 64],
                k_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                k_b: vec![0.0f32; 64],
                v_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                v_b: vec![0.0f32; 64],
                o_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                o_b: vec![0.0f32; 64],
                ln2_w: vec![1.0f32; 64],
                ln2_b: vec![0.0f32; 64],
                ffn_up_w: MmapWeight::from_owned_f32(vec![0.0f32; 128 * 64], 128, 64),
                ffn_up_b: vec![0.0f32; 128],
                ffn_down_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 128], 64, 128),
                ffn_down_b: vec![0.0f32; 64],
            }],
            post_ln_w: vec![1.0f32; 64],
            post_ln_b: vec![0.0f32; 64],
            projector: ProjectorWeights {
                mm1_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 256], 64, 256),
                mm1_b: vec![0.0f32; 64],
                mm2_w: MmapWeight::from_owned_f32(vec![0.0f32; 32 * 64], 32, 64),
                mm2_b: vec![0.0f32; 32],
            },
        };

        let result = try_hexagon_vision_encoder(&weights);
        assert!(result.is_none());
    }

    #[test]
    fn test_hexagon_vit_weight_offsets_plan_rejects_dense_f32() {
        let cfg = VisionEncoderConfig {
            n_layer: 1,
            n_embd: 64,
            n_ff: 128,
            n_head: 2,
            eps: 1e-5,
            image_size: 32,
            patch_size: 16,
            n_trained_patches: 4,
            projection_dim: 32,
            scale_factor: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 32 * 32,
            image_max_pixels: 32 * 32,
        };
        let weights = VisionEncoderWeights {
            config: cfg,
            patch_embed: PatchEmbedWeights {
                conv_w: vec![0.0f32; 3 * 16 * 16 * 64],
                conv_b: vec![0.0f32; 64],
            },
            position_embed: vec![0.0f32; 4 * 64],
            blocks: vec![VitBlockWeights {
                ln1_w: vec![1.0f32; 64],
                ln1_b: vec![0.0f32; 64],
                q_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                q_b: vec![0.0f32; 64],
                k_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                k_b: vec![0.0f32; 64],
                v_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                v_b: vec![0.0f32; 64],
                o_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64),
                o_b: vec![0.0f32; 64],
                ln2_w: vec![1.0f32; 64],
                ln2_b: vec![0.0f32; 64],
                ffn_up_w: MmapWeight::from_owned_f32(vec![0.0f32; 128 * 64], 128, 64),
                ffn_up_b: vec![0.0f32; 128],
                ffn_down_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 128], 64, 128),
                ffn_down_b: vec![0.0f32; 64],
            }],
            post_ln_w: vec![1.0f32; 64],
            post_ln_b: vec![0.0f32; 64],
            projector: ProjectorWeights {
                mm1_w: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 256], 64, 256),
                mm1_b: vec![0.0f32; 64],
                mm2_w: MmapWeight::from_owned_f32(vec![0.0f32; 32 * 64], 32, 64),
                mm2_b: vec![0.0f32; 32],
            },
        };

        let plan_res = HexagonVitWeightOffsets::plan(&weights);
        let err = plan_res.unwrap_err().to_string();
        assert!(err.contains("requires Q8_0 or Q4_0"));
    }

    #[test]
    fn test_hexagon_vit_weight_offsets_plan_q8_0() {
        let make_q8 = |rows: usize, cols: usize| -> MmapWeight {
            let n_blocks = (rows * cols) / 32;
            let bytes = vec![0u8; n_blocks * 34];
            MmapWeight::from_owned_bytes(bytes, DType::Q8_0, rows, cols)
        };

        let cfg = VisionEncoderConfig {
            n_layer: 1,
            n_embd: 64,
            n_ff: 128,
            n_head: 2,
            eps: 1e-5,
            image_size: 32,
            patch_size: 16,
            n_trained_patches: 4,
            projection_dim: 32,
            scale_factor: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 32 * 32,
            image_max_pixels: 32 * 32,
        };
        let weights = VisionEncoderWeights {
            config: cfg,
            patch_embed: PatchEmbedWeights {
                conv_w: vec![0.0f32; 3 * 16 * 16 * 64],
                conv_b: vec![0.0f32; 64],
            },
            position_embed: vec![0.0f32; 4 * 64],
            blocks: vec![VitBlockWeights {
                ln1_w: vec![1.0f32; 64],
                ln1_b: vec![0.0f32; 64],
                q_w: make_q8(64, 64),
                q_b: vec![0.0f32; 64],
                k_w: make_q8(64, 64),
                k_b: vec![0.0f32; 64],
                v_w: make_q8(64, 64),
                v_b: vec![0.0f32; 64],
                o_w: make_q8(64, 64),
                o_b: vec![0.0f32; 64],
                ln2_w: vec![1.0f32; 64],
                ln2_b: vec![0.0f32; 64],
                ffn_up_w: make_q8(128, 64),
                ffn_up_b: vec![0.0f32; 128],
                ffn_down_w: make_q8(64, 128),
                ffn_down_b: vec![0.0f32; 64],
            }],
            post_ln_w: vec![1.0f32; 64],
            post_ln_b: vec![0.0f32; 64],
            projector: ProjectorWeights {
                mm1_w: make_q8(64, 256),
                mm1_b: vec![0.0f32; 64],
                mm2_w: make_q8(32, 64),
                mm2_b: vec![0.0f32; 32],
            },
        };

        let plan = HexagonVitWeightOffsets::plan(&weights).expect("Q8_0 weights should plan");
        assert_eq!(plan.blocks.len(), 1);
        assert!(plan.projector.is_some());
        let proj = plan.projector.unwrap();
        assert_eq!(proj.mm1_w.format, HexagonWeightFormat::RepackedQ8_0);
        assert_eq!(proj.mm2_w.format, HexagonWeightFormat::RepackedQ8_0);
        assert!(plan.total_bytes > proj.mm2_b_off);
    }
}

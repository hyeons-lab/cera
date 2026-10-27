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
    FastRpcDriver, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonContext,
    HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, LockOrRecover, RpcmemBuffer,
    align128, build_binary_kernel_params, repack_q4_0, repack_q8_0, repacked_matrix_size_q4_0,
    repacked_matrix_size_q8_0,
};
use crate::model::vision_encoder::{
    PatchEmbedWeights, ProjectorWeights, VisionEncoderConfig, VisionEncoderWeights,
    VitBlockWeights, VitStageDump, interpolate_pos_embed_2d, patch_embed_compute, pixel_shuffle,
};
use crate::model::vision_encoder_gpu::{
    MAX_VIT_TOKENS, VisionGpuEncode, VitStack, VitStackActivation, VitStackSpec,
};
use crate::model::weights::MmapWeight;
use crate::session::CeraError;
use crate::tensor::DType;

use crate::backend::hexagon::HexagonWeightFormat;
/// The ViT emits one op over all tokens (see `dispatch::TokenTile`); only the
/// Whisper encoder tiles. Every `dispatch::*` call here passes this constant.
const VIT_TILE: TokenTile = TokenTile::Whole;

/// Weight layout in the ViT's rpcmem weights buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexagonVitWeightFormat {
    RepackedQ8_0,
    RepackedQ4_0,
    /// Plain row-major F16, no repack. Only the HMX matmul reads it.
    F16,
}

/// Placement of one ViT linear weight in the weights buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonVitWeightDesc {
    pub offset: usize,
    pub size_bytes: usize,
    pub format: HexagonVitWeightFormat,
    pub rows: usize,
    pub cols: usize,
}

/// The HMX matmul option for a ViT projection: on when the DSP has HMX, off with
/// `CERA_HEXAGON_VIT_MM=hvx` to compare against the HVX matmul.
fn vit_hmx(session: &HexagonQueueSession) -> Option<dispatch::HmxMm> {
    let hvx_only = std::env::var("CERA_HEXAGON_VIT_MM").is_ok_and(|v| v == "hvx");
    (!hvx_only && session.dsp_hmx() > 0).then(|| dispatch::HmxMm {
        vtcm_budget: session.dsp_vtcm_bytes(),
    })
}

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

/// FFN width rounded up to the 32-wide tile the DSP matmul works in.
fn ffn_padded_width(n_ff: usize) -> usize {
    n_ff.next_multiple_of(32)
}

/// Row layout for zero-padding a 2D byte buffer along the row axis.
#[derive(Debug, Clone, Copy)]
struct RowPadSpec {
    rows: usize,
    row_bytes: usize,
    rows_pad: usize,
}

/// Copy `spec.rows` rows of `spec.row_bytes` bytes and append zero rows up to `spec.rows_pad`.
///
/// A zero Q8_0/Q4_0 block (zero scale, zero quants) decodes to exactly 0.
fn pad_rows_raw(src: &[u8], spec: RowPadSpec) -> Result<Vec<u8>, CeraError> {
    let RowPadSpec {
        rows,
        row_bytes,
        rows_pad,
    } = spec;
    let src_bytes = rows.checked_mul(row_bytes).ok_or_else(|| {
        CeraError::Backend(format!(
            "ViT weight byte count overflow for {rows} rows of {row_bytes} bytes"
        ))
    })?;
    let pad_bytes = rows_pad.checked_mul(row_bytes).ok_or_else(|| {
        CeraError::Backend(format!(
            "ViT padded weight byte count overflow for {rows_pad} rows of {row_bytes} bytes"
        ))
    })?;
    if src.len() < src_bytes || rows_pad < rows {
        return Err(CeraError::Backend(format!(
            "ViT weight ({} bytes) too short for {rows} rows of {row_bytes} bytes (padded to {rows_pad})",
            src.len()
        )));
    }
    let mut out = vec![0u8; pad_bytes];
    out[..src_bytes].copy_from_slice(&src[..src_bytes]);
    Ok(out)
}

/// Copy an F16 `[rows, cols]` matrix, zero-padding each row to `cols_pad` columns.
fn pad_f16_cols(w: &MmapWeight, cols_pad: usize) -> Result<Vec<u8>, CeraError> {
    if w.dtype != DType::F16 {
        return Err(CeraError::Backend(format!(
            "pad_f16_cols requires F16 weight, got {:?}",
            w.dtype
        )));
    }
    let (rows, cols) = (w.rows, w.cols);
    let bytes = w.data();
    let row_bytes = cols.checked_mul(2).ok_or_else(|| {
        CeraError::Backend(format!("f16 ViT weight column byte overflow ({cols})"))
    })?;
    let src_bytes = rows.checked_mul(row_bytes).ok_or_else(|| {
        CeraError::Backend(format!("f16 ViT weight byte overflow ({rows}x{cols})"))
    })?;
    let pad_row_bytes = cols_pad.checked_mul(2).ok_or_else(|| {
        CeraError::Backend(format!("f16 ViT padded column byte overflow ({cols_pad})"))
    })?;
    let pad_bytes = rows.checked_mul(pad_row_bytes).ok_or_else(|| {
        CeraError::Backend(format!(
            "f16 ViT padded weight byte overflow ({rows}x{cols_pad})"
        ))
    })?;
    if bytes.len() < src_bytes || cols_pad < cols {
        return Err(CeraError::Backend(format!(
            "f16 ViT weight {rows}x{cols} ({} bytes) cannot pad to {cols_pad} columns",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; pad_bytes];
    for r in 0..rows {
        out[r * pad_row_bytes..r * pad_row_bytes + row_bytes]
            .copy_from_slice(&bytes[r * row_bytes..(r + 1) * row_bytes]);
    }
    Ok(out)
}

/// Dequantize an F16 `[rows, cols]` matrix, zero-pad each row to `cols_pad` and
/// quantize it to linear GGUF Q8_0 blocks.
fn f16_matrix_to_padded_q8_0(w: &MmapWeight, cols_pad: usize) -> Result<Vec<u8>, CeraError> {
    if w.dtype != DType::F16 {
        return Err(CeraError::Backend(format!(
            "f16_matrix_to_padded_q8_0 requires F16 weight, got {:?}",
            w.dtype
        )));
    }
    let (rows, cols) = (w.rows, w.cols);
    let bytes = w.data();
    let row_bytes = cols.checked_mul(2).ok_or_else(|| {
        CeraError::Backend(format!("f16 ViT weight column byte overflow ({cols})"))
    })?;
    let src_bytes = rows.checked_mul(row_bytes).ok_or_else(|| {
        CeraError::Backend(format!("f16 ViT weight byte overflow ({rows}x{cols})"))
    })?;
    let pad_elems = rows.checked_mul(cols_pad).ok_or_else(|| {
        CeraError::Backend(format!(
            "f16 ViT padded element overflow ({rows}x{cols_pad})"
        ))
    })?;
    if bytes.len() < src_bytes || cols_pad < cols {
        return Err(CeraError::Backend(format!(
            "f16 ViT weight too short: {} bytes for {rows}x{cols} (padded to {cols_pad})",
            bytes.len()
        )));
    }
    let mut vals = vec![0f32; pad_elems];
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let dst = &mut vals[r * cols_pad..r * cols_pad + cols];
        for (d, &c) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
            *d = crate::quant::f16_to_f32(u16::from_le_bytes(c));
        }
    }
    crate::backend::hexagon::quantize_f32_to_q8_0(&vals, cols_pad, rows)
}

/// How F16 `ffn_down` weights are stored in the ViT rpcmem weights buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VitF16Policy {
    /// Keep row-major F16 (zero-padded to 32-wide tiles) for the HMX matmul unit.
    NativeF16,
    /// Requantize F16 weights to Q8_0 (zero-padded to 32-wide tiles) when HMX is unavailable.
    RequantQ8_0,
}

#[derive(Debug, Clone, Copy)]
struct QuantLinearDims {
    dtype: DType,
    rows: usize,
    cols: usize,
}

impl HexagonVitWeightOffsets {
    /// Compute memory layout and byte offsets for all ViT weights in shared rpcmem,
    /// keeping F16 `ffn_down` weights in native F16 for the HMX matmul path.
    pub fn plan(weights: &VisionEncoderWeights) -> Result<Self, CeraError> {
        Self::plan_with(weights, VitF16Policy::NativeF16)
    }

    /// Like [`Self::plan`], with `f16_policy` selecting whether an F16 `ffn_down`
    /// stays F16 (zero-padded to the 32-wide tile) for the HMX matmul or is
    /// requantized to Q8_0 when HMX is unavailable.
    pub fn plan_with(
        weights: &VisionEncoderWeights,
        f16_policy: VitF16Policy,
    ) -> Result<Self, CeraError> {
        let mut cur_off = 0;

        let plan_vec_f32 = |cur_off: &mut usize, len: usize| -> usize {
            let off = *cur_off;
            *cur_off += align128(len * 4);
            off
        };

        let plan_linear_dims = |cur_off: &mut usize,
                                dims: QuantLinearDims|
         -> Result<HexagonVitWeightDesc, CeraError> {
            let QuantLinearDims { dtype, rows, cols } = dims;
            let off = *cur_off;
            let (fmt, sz) = match dtype {
                DType::Q8_0 => {
                    let sz = repacked_matrix_size_q8_0(cols, rows)?;
                    (HexagonVitWeightFormat::RepackedQ8_0, sz)
                }
                DType::Q4_0 => {
                    let sz = repacked_matrix_size_q4_0(cols, rows)?;
                    (HexagonVitWeightFormat::RepackedQ4_0, sz)
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
                rows,
                cols,
            })
        };
        let plan_linear =
            |cur_off: &mut usize, w: &MmapWeight| -> Result<HexagonVitWeightDesc, CeraError> {
                plan_linear_dims(
                    cur_off,
                    QuantLinearDims {
                        dtype: w.dtype,
                        rows: w.rows,
                        cols: w.cols,
                    },
                )
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
            // The FFN width is padded to the DSP's 32-wide tiles (zero rows in
            // ffn_up and zero columns in ffn_down are exact no-ops), and an F16
            // ffn_down is F16 in the GGUF whenever the width is not a multiple of 32 (Q8_0
            // blocks are 32 wide, e.g. 4304). It stays F16 for the HMX matmul, or is
            // requantized to Q8_0 when native F16 is unavailable.
            let ffn_pad = ffn_padded_width(blk.ffn_up_w.rows);
            let ffn_up_w = plan_linear_dims(
                &mut cur_off,
                QuantLinearDims {
                    dtype: blk.ffn_up_w.dtype,
                    rows: ffn_pad,
                    cols: blk.ffn_up_w.cols,
                },
            )?;
            let ffn_up_b_off = plan_vec_f32(&mut cur_off, ffn_pad);
            let ffn_down_w = match (blk.ffn_down_w.dtype, f16_policy) {
                (DType::F16, VitF16Policy::NativeF16)
                    if blk.ffn_down_w.rows > 0 && blk.ffn_down_w.rows.is_multiple_of(32) =>
                {
                    let rows = blk.ffn_down_w.rows;
                    let size_bytes = rows * ffn_pad * 2;
                    let off = cur_off;
                    cur_off += align128(size_bytes);
                    HexagonVitWeightDesc {
                        offset: off,
                        size_bytes,
                        format: HexagonVitWeightFormat::F16,
                        rows,
                        cols: ffn_pad,
                    }
                }
                (DType::F16, _) => plan_linear_dims(
                    &mut cur_off,
                    QuantLinearDims {
                        dtype: DType::Q8_0,
                        rows: blk.ffn_down_w.rows,
                        cols: ffn_pad,
                    },
                )?,
                _ => plan_linear(&mut cur_off, &blk.ffn_down_w)?,
            };
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
            // The projector's hidden activation reuses the feed-forward's
            // scratch and its GELU works through scratch sized by `n_ff`, so a
            // wider one cannot run.
            let n_ff_pad = ffn_padded_width(weights.config.n_ff);
            if weights.projector.mm1_w.rows > n_ff_pad {
                return Err(CeraError::Backend(format!(
                    "ViT projector hidden width {} exceeds the feed-forward width {}",
                    weights.projector.mm1_w.rows, weights.config.n_ff
                )));
            }
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
        Self::with_rows(n_embd, n_ff, MAX_VIT_TOKENS)
    }

    /// Compute scratch allocation sized for up to `rows` tokens.
    pub fn with_rows(n_embd: usize, n_ff: usize, rows: usize) -> Self {
        let token_bytes = align128(rows * n_embd * 4);
        let token_f16_bytes = align128(rows * n_embd * 2);
        let ffn_mid_bytes = align128(rows * n_ff * 4);

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
    /// FFN width after padding to the 32-wide DSP tile (equals `n_ff` when aligned).
    n_ff_pad: usize,
    /// Most tokens one pass takes: the scratch is sized for it.
    max_rows: usize,
    /// What sits between a block's two feed-forward projections.
    activation: VitStackActivation,
    /// Whether the blocks end in the final LayerNorm.
    post_ln: bool,
}

impl Drop for HexagonVisionEncoder {
    /// Let the DSP let go of every buffer before the host unmaps them.
    fn drop(&mut self) {
        let mut device = self.device.lock_or_recover();
        device
            .queue_session_mut()
            .release_dsp_references([&self.weights_buf, &*self.scratch_buf.lock_or_recover()]);
    }
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
        Self::with_rows(driver, device, weights, MAX_VIT_TOKENS)
    }

    /// [`Self::new`] with scratch for `max_rows` tokens per pass.
    pub fn with_rows(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        weights: &VisionEncoderWeights,
        max_rows: usize,
    ) -> Result<Self, CeraError> {
        let cfg = &weights.config;
        // F16 weights are only read by the HMX matmul; without HMX (or when rows are not
        // 32-aligned for HMX tiles) they are requantized.
        let f16_policy = {
            let mut dev = device.lock_or_recover();
            let ffn_down_hmx_aligned = weights
                .blocks
                .iter()
                .all(|b| b.ffn_down_w.rows > 0 && b.ffn_down_w.rows.is_multiple_of(32));
            if vit_hmx(dev.queue_session_mut()).is_some() && ffn_down_hmx_aligned {
                VitF16Policy::NativeF16
            } else {
                VitF16Policy::RequantQ8_0
            }
        };
        let n_ff_pad = ffn_padded_width(cfg.n_ff);
        let scratch_offsets = HexagonVitScratchOffsets::with_rows(cfg.n_embd, n_ff_pad, max_rows);
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true)?;

        // Plan weights buffer layout
        let weights_offsets = HexagonVitWeightOffsets::plan_with(weights, f16_policy)?;
        let mut weights_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), weights_offsets.total_bytes, true)?;

        // Populate weights buffer
        let copy_vec_f32 = |dst: &mut [u8], off: usize, src: &[f32]| {
            let bytes = bytemuck::cast_slice(src);
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        };

        let repack_raw = |dst: &mut [u8],
                          desc: HexagonVitWeightDesc,
                          data: &[u8]|
         -> Result<(), CeraError> {
            let dst_slice = &mut dst[desc.offset..desc.offset + desc.size_bytes];
            match desc.format {
                HexagonVitWeightFormat::RepackedQ8_0 => {
                    repack_q8_0(data, desc.cols, desc.rows, dst_slice)
                        .map_err(|e| CeraError::Backend(format!("ViT repack Q8_0 failed: {e}")))?;
                }
                HexagonVitWeightFormat::RepackedQ4_0 => {
                    repack_q4_0(data, desc.cols, desc.rows, dst_slice)
                        .map_err(|e| CeraError::Backend(format!("ViT repack Q4_0 failed: {e}")))?;
                }
                HexagonVitWeightFormat::F16 => {
                    if data.len() != desc.size_bytes {
                        return Err(CeraError::Backend(format!(
                            "ViT F16 weight is {} bytes, expected {}",
                            data.len(),
                            desc.size_bytes
                        )));
                    }
                    dst_slice.copy_from_slice(data);
                }
            }
            Ok(())
        };
        let copy_linear = |dst: &mut [u8],
                           desc: HexagonVitWeightDesc,
                           w: &MmapWeight|
         -> Result<(), CeraError> { repack_raw(dst, desc, w.data()) };

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
            let ffn_pad = blk_offs.ffn_up_w.rows;
            if ffn_pad == blk_weights.ffn_up_w.rows {
                copy_linear(buf_slice, blk_offs.ffn_up_w, &blk_weights.ffn_up_w)?;
                copy_vec_f32(buf_slice, blk_offs.ffn_up_b_off, &blk_weights.ffn_up_b);
            } else {
                let w = &blk_weights.ffn_up_w;
                let block_bytes = match w.dtype {
                    DType::Q8_0 => 34,
                    DType::Q4_0 => 18,
                    other => {
                        return Err(CeraError::Backend(format!(
                            "ViT ffn_up padding: unsupported dtype {other:?}"
                        )));
                    }
                };
                let padded = pad_rows_raw(
                    w.data(),
                    RowPadSpec {
                        rows: w.rows,
                        row_bytes: w.cols / 32 * block_bytes,
                        rows_pad: ffn_pad,
                    },
                )?;
                repack_raw(buf_slice, blk_offs.ffn_up_w, &padded)?;
                let mut bias = blk_weights.ffn_up_b.clone();
                bias.resize(ffn_pad, 0.0);
                copy_vec_f32(buf_slice, blk_offs.ffn_up_b_off, &bias);
            }
            match (blk_weights.ffn_down_w.dtype, blk_offs.ffn_down_w.format) {
                (DType::F16, HexagonVitWeightFormat::F16) => {
                    let f16 = pad_f16_cols(&blk_weights.ffn_down_w, blk_offs.ffn_down_w.cols)?;
                    repack_raw(buf_slice, blk_offs.ffn_down_w, &f16)?;
                }
                (DType::F16, _) => {
                    let q8 = f16_matrix_to_padded_q8_0(&blk_weights.ffn_down_w, ffn_pad)?;
                    repack_raw(buf_slice, blk_offs.ffn_down_w, &q8)?;
                }
                _ => {
                    copy_linear(buf_slice, blk_offs.ffn_down_w, &blk_weights.ffn_down_w)?;
                }
            }
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
            n_ff_pad,
            max_rows,
            activation: VitStackActivation::GeluTanh,
            post_ln: true,
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

    /// Dispatch linear layer `dst[tokens, rows] = x[tokens, cols] . Wt + bias`.
    ///
    /// Quantized weights go through [`dispatch::linear_m_with`]; plain F16 weights
    /// (`ffn_down` when the feed-forward width is not a multiple of 32) have no
    /// HVX repack path and run on the HMX matmul unit directly.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_m(
        session: &mut HexagonQueueSession,
        x: &RpcmemBuffer,
        x_offset: usize,
        weights: &RpcmemBuffer,
        w_desc: HexagonVitWeightDesc,
        b_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let format = match w_desc.format {
            HexagonVitWeightFormat::RepackedQ8_0 => Some(HexagonWeightFormat::RepackedQ8_0),
            HexagonVitWeightFormat::RepackedQ4_0 => Some(HexagonWeightFormat::RepackedQ4_0),
            HexagonVitWeightFormat::F16 => None,
        };
        if let Some(format) = format {
            return dispatch::linear_m_with(
                session,
                x,
                x_offset,
                weights,
                crate::backend::hexagon::HexagonWeightDesc {
                    offset: w_desc.offset,
                    size_bytes: w_desc.size_bytes,
                    format,
                    rows: w_desc.rows,
                    cols: w_desc.cols,
                },
                b_offset,
                dst,
                dst_offset,
                n_tokens,
                VIT_TILE,
                vit_hmx(session),
            );
        }

        let (ne0, ne1) = (w_desc.cols, w_desc.rows);
        let hmx_kparams = vit_hmx(session)
            .filter(|_| {
                crate::backend::hexagon::mm_is_hmx_eligible(HtpDataType::F16, ne0, ne1, n_tokens)
            })
            .and_then(|h| {
                crate::backend::hexagon::build_hmx_mm_kernel_params(
                    HtpDataType::F16,
                    ne0,
                    ne1,
                    n_tokens.next_multiple_of(32),
                    n_tokens,
                    session.dsp_threads(),
                    h.vtcm_budget,
                )
            })
            .ok_or_else(|| {
                CeraError::Backend(format!(
                    "F16 ViT matmul {ne0}x{ne1} (M={n_tokens}) needs the HMX path, which did not fit"
                ))
            })?;

        let row_bytes = crate::backend::hexagon::mm_hmx_nb1(HtpDataType::F16, ne0);
        let w_tot = ne1 * row_bytes;
        let w_ti = session.add_tensor(
            weights,
            w_desc.offset,
            w_desc.size_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F16 as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [2, row_bytes as u32, w_tot as u32, w_tot as u32],
        )?;

        let x_bytes = n_tokens * ne0 * 4;
        let x_ti = session.add_tensor(
            x,
            x_offset,
            x_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [ne0 as u32, n_tokens as u32, 1, 1],
            [4, (ne0 * 4) as u32, x_bytes as u32, x_bytes as u32],
        )?;

        let dst_bytes = n_tokens * ne1 * 4;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [ne1 as u32, n_tokens as u32, 1, 1],
            [4, (ne1 * 4) as u32, dst_bytes as u32, dst_bytes as u32],
        )?;

        session
            .enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w_ti, x_ti],
                &[dst_ti],
                [0i32; 16],
                hmx_kparams,
            )
            .map_err(|e| CeraError::Backend(format!("dispatch_linear_m (F16): {e}")))?;

        Self::dispatch_bias_add(session, dst_ti, weights, b_offset, ne1)?;
        session.end_group()
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
        let fused_quant = match q_w.format {
            HexagonVitWeightFormat::RepackedQ8_0 => {
                Some((HtpDataType::Q8_0, 34usize, 32 * 34usize))
            }
            HexagonVitWeightFormat::RepackedQ4_0 => {
                Some((HtpDataType::Q4_0, 18usize, 32 * 18usize))
            }
            HexagonVitWeightFormat::F16 => None,
        };
        // The fused kernel cannot chunk its activation rows, so past what its
        // VTCM layout holds the three projections run on their own (each
        // chunks itself): a few hundred tokens of a large image.
        let fits_fused = fused_quant.is_some_and(|(fused_dtype, _, _)| {
            qkv_fits_fused(fused_dtype, q_w.cols, n_tokens, session.dsp_threads())
        });
        // The HMX matmul beats the fused HVX kernel once there are rows to feed it.
        let hmx_matmul = vit_hmx(session).is_some()
            && fused_quant.is_some_and(|(fused_dtype, _, _)| {
                crate::backend::hexagon::mm_is_hmx_eligible(
                    fused_dtype,
                    q_w.cols,
                    q_w.rows,
                    n_tokens,
                )
            });
        let Some((w_dtype, block_bytes, tile_size)) = fused_quant.filter(|_| {
            fits_fused
                && !hmx_matmul
                && q_w.cols == k_w.cols
                && k_w.cols == v_w.cols
                && q_w.rows == k_w.rows
                && k_w.rows == v_w.rows
                && q_w.format == k_w.format
                && k_w.format == v_w.format
        }) else {
            Self::dispatch_linear_m(
                session, x, x_offset, weights, q_w, q_b_off, dst, q_dst_off, n_tokens,
            )?;
            Self::dispatch_linear_m(
                session, x, x_offset, weights, k_w, k_b_off, dst, k_dst_off, n_tokens,
            )?;
            Self::dispatch_linear_m(
                session, x, x_offset, weights, v_w, v_b_off, dst, v_dst_off, n_tokens,
            )?;
            return Ok(());
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

        let mut mm = crate::backend::hexagon::MmKernelParams::from_words(
            &crate::backend::hexagon::build_mul_mat_kernel_params(
                w_dtype,
                ne0,
                n_tokens as u32,
                1,
                ne1 * 4,
                session.dsp_threads(),
                dispatch::VTCM_BUDGET,
            ),
        );
        // `fits_fused` above keeps the rows inside what the fused layout holds,
        // which implies the plain layout needs no chunk; the fused kernel could
        // not honour one.
        debug_assert_eq!(mm.m_chunk, 0, "fused QKV must not need a row chunk");
        mm.kernel_type = 5; // HTP_MM_KERNEL_HVX_QUANT_ROW
        mm.n_weights = 3;
        let kparams = mm.to_words();

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

        // Many query rows against the same K/V: run the HMX kernel when the DSP has one
        // and a tiling fits VTCM, as the LFM2 prefill does; `CERA_HEXAGON_VIT_FA=hvx`
        // keeps the row-at-a-time HVX kernel for comparison.
        let hvx_only = std::env::var("CERA_HEXAGON_VIT_FA").is_ok_and(|v| v == "hvx");
        let hmx_kparams = if !hvx_only
            && session.dsp_hmx() > 0
            && crate::backend::hexagon::fa_is_hmx_eligible(head_dim, n_tokens)
        {
            crate::backend::hexagon::build_hmx_fa_kernel_params(
                head_dim,
                n_heads,
                n_heads,
                n_tokens,
                n_tokens,
                scale,
                session.dsp_threads(),
                session.dsp_vtcm_bytes(),
            )
        } else {
            None
        };
        let kparams = hmx_kparams.unwrap_or_else(|| {
            crate::backend::hexagon::build_flash_attn_kernel_params(
                head_dim,
                n_heads,
                n_heads,
                n_tokens,
                n_tokens,
                scale,
                session.dsp_threads(),
                false,
            )
        });

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
    fn backend_label(&self) -> &'static str {
        "hexagon"
    }

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

    /// The blocks (and the final LayerNorm, when the encoder has one) over `n_patches` rows of
    /// `tokens` (`[n_patches, n_embd]`, already embedded), read back as the rows they leave; with
    /// `stop_after_blocks` just that many blocks and no final norm.
    fn run_blocks(
        &self,
        tokens: &[f32],
        n_patches: usize,
        stop_after_blocks: Option<usize>,
        dump: Option<&mut VitStageDump>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        ensure!(n_patches > 0, "a ViT pass needs at least one token");
        ensure!(
            n_patches <= self.max_rows,
            "{n_patches} tokens exceed the encoder's {} rows",
            self.max_rows
        );
        ensure!(
            tokens.len() == n_patches * cfg.n_embd,
            "{} values are not {n_patches} rows of {}",
            tokens.len(),
            cfg.n_embd
        );
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
            scratch_slice.copy_from_slice(bytemuck::cast_slice(tokens));
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
                Self::dispatch_linear_m(
                    session,
                    scratch_guard,
                    so.attn_out_off,
                    &self.weights_buf,
                    blk.o_w,
                    blk.o_b_off,
                    scratch_guard,
                    so.proj_out_off,
                    n_patches,
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
                Self::dispatch_linear_m(
                    session,
                    scratch_guard,
                    so.pre_norm_off,
                    &self.weights_buf,
                    blk.ffn_up_w,
                    blk.ffn_up_b_off,
                    scratch_guard,
                    so.ffn_mid_off,
                    n_patches,
                )?;

                // The feed-forward activation on ffn_mid
                match self.activation {
                    VitStackActivation::GeluTanh => dispatch::gelu_tanh(
                        session,
                        scratch_guard,
                        so.ffn_mid_off,
                        scratch_guard,
                        so.gelu_tmp_off,
                        dispatch::gelu_tmp_rows(so.gelu_tmp_bytes, self.n_ff_pad)?,
                        TokenShape {
                            dim: self.n_ff_pad,
                            n_tokens: n_patches,
                        },
                    )?,
                    VitStackActivation::Relu => dispatch::relu(
                        session,
                        scratch_guard,
                        so.ffn_mid_off,
                        TokenShape {
                            dim: self.n_ff_pad,
                            n_tokens: n_patches,
                        },
                        VIT_TILE,
                    )?,
                }

                // FFN down projection: ffn_mid -> ffn_out
                Self::dispatch_linear_m(
                    session,
                    scratch_guard,
                    so.ffn_mid_off,
                    &self.weights_buf,
                    blk.ffn_down_w,
                    blk.ffn_down_b_off,
                    scratch_guard,
                    so.ffn_out_off,
                    n_patches,
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
            if stop_after_blocks.is_none() && self.post_ln {
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
                scratch_guard.invalidate_cpu_cache(0, so.total_bytes);
                let all: &[f32] = bytemuck::cast_slice(scratch_guard.as_slice());
                let region = |off: usize, floats: usize| all[off / 4..off / 4 + floats].to_vec();
                let n = n_patches * n_embd;
                d.q = region(so.q_off, n);
                d.k = region(so.k_off, n);
                d.v = region(so.v_off, n);
                d.attn_out = region(so.attn_out_off, n);
                d.attn_proj = region(so.proj_out_off, n);
                d.ln2 = region(so.pre_norm_off, n);
                d.ffn_mid = (0..n_patches)
                    .flat_map(|r| {
                        let base = so.ffn_mid_off / 4 + r * self.n_ff_pad;
                        all[base..base + cfg.n_ff].iter().copied()
                    })
                    .collect();
                d.ffn_out = region(so.ffn_out_off, n);
                d.tokens = normed_slice.to_vec();
            }
            return Ok(normed_slice.to_vec());
        }

        Ok(normed_slice.to_vec())
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
            n_patches <= self.max_rows,
            "encode_image: {n_patches} patches exceeds the encoder's {} rows",
            self.max_rows
        );

        let vit_timing = std::env::var_os("CERA_HEXAGON_VIT_TIMING").is_some();
        let host_start = std::time::Instant::now();
        // 1. Patch embedding on host CPU
        let mut tokens = patch_embed_compute(pixels, &self.patch_embed, cfg, grid_w, grid_h);
        let patch_embed_ms = host_start.elapsed().as_secs_f64() * 1000.0;

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

        let host_pre_ms = host_start.elapsed().as_secs_f64() * 1000.0;
        let mut dump = dump;
        if let Some(d) = dump.as_deref_mut() {
            d.x0 = tokens.clone();
        }

        let blocks_start = std::time::Instant::now();
        let normed = self.run_blocks(&tokens, n_patches, stop_after_blocks, dump)?;
        let blocks_ms = blocks_start.elapsed().as_secs_f64() * 1000.0;
        let post_start = std::time::Instant::now();
        if stop_after_blocks.is_some() {
            return Ok(normed);
        }
        // 4. Pixel-shuffle pool over dynamic grid
        let pooled = pixel_shuffle(&normed, cfg, grid_w, grid_h);

        // 5. 2-layer MLP Projector to LLM embedding dimension
        match self.device_projector(&pooled) {
            Ok(Some(out)) => {
                if vit_timing {
                    eprintln!(
                        "ViT timing: patch_embed {patch_embed_ms:.1} ms | host_pre {host_pre_ms:.1} ms | dsp_blocks {blocks_ms:.1} ms | post (readback, pixel shuffle, projector) {:.1} ms",
                        post_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                return Ok(out);
            }
            Ok(None) => {}
            Err(e) => crate::backend::hexagon::hexagon_warn!(
                "HexagonVisionEncoder: DSP projector failed ({e}), falling back to CPU"
            ),
        }
        Ok(self.projector.forward(&pooled, cfg.projection_dim))
    }

    /// The projector (two matmuls and a GELU) on the DSP over `pooled` rows (`[n, mm1 cols]`), or
    /// `None` when this encoder has no projector on the device or `pooled` does not fit it.
    fn device_projector(&self, pooled: &[f32]) -> Result<Option<Vec<f32>>, CeraError> {
        let Some(proj_offs) = self.weights_offsets.projector else {
            return Ok(None);
        };
        let in_dim = proj_offs.mm1_w.cols;
        let mid_dim = proj_offs.mm1_w.rows;
        let out_dim = self.config.projection_dim;
        if in_dim == 0 || pooled.is_empty() || !pooled.len().is_multiple_of(in_dim) {
            return Ok(None);
        }
        let n_tokens = pooled.len() / in_dim;
        if n_tokens * in_dim > self.max_rows * self.config.n_embd {
            return Ok(None);
        }
        let in_bytes = pooled.len() * 4;
        let out_bytes = n_tokens * out_dim * 4;
        let so = self.scratch_offsets;
        let mut dev_guard = self.device.lock_or_recover();
        let mut scratch_guard = self.scratch_buf.lock_or_recover();
        let session = dev_guard.queue_session_mut();
        session.drop_pending_batch();

        scratch_guard.as_mut_slice()[so.tokens_off..so.tokens_off + in_bytes]
            .copy_from_slice(bytemuck::cast_slice(pooled));
        scratch_guard.flush_cpu_cache(so.tokens_off, in_bytes);

        let run = (|| -> Result<(), CeraError> {
            Self::dispatch_linear_m(
                session,
                &scratch_guard,
                so.tokens_off,
                &self.weights_buf,
                proj_offs.mm1_w,
                proj_offs.mm1_b_off,
                &scratch_guard,
                so.proj_mid_off,
                n_tokens,
            )?;
            // the projector's GELU is the exact (erf) form, as the host projector's is: the
            // skels' native op, one pass and no scratch
            dispatch::gelu_erf(
                session,
                &scratch_guard,
                so.proj_mid_off,
                TokenShape {
                    dim: mid_dim,
                    n_tokens,
                },
                TokenTile::Whole,
            )?;
            Self::dispatch_linear_m(
                session,
                &scratch_guard,
                so.proj_mid_off,
                &self.weights_buf,
                proj_offs.mm2_w,
                proj_offs.mm2_b_off,
                &scratch_guard,
                so.proj_final_off,
                n_tokens,
            )?;
            session.flush()?;
            Ok(())
        })();
        if let Err(e) = run {
            session.drop_pending_batch();
            return Err(e);
        }
        scratch_guard.invalidate_cpu_cache(so.proj_final_off, out_bytes);
        let out: &[f32] = bytemuck::cast_slice(
            &scratch_guard.as_slice()[so.proj_final_off..so.proj_final_off + out_bytes],
        );
        Ok(Some(out.to_vec()))
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

    let arch_override = crate::backend::hexagon::arch_override();

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

/// A [`VitStackSpec`] on the Hexagon NPU: the blocks (and the final LayerNorm, when the spec has
/// one) of a transformer whose tokens the caller embeds, as the d1 vision tower and the d1
/// decision head need. It is the ViT encoder above without the patch embedding, the position
/// table and the projector, with ReLU available between a block's feed-forward projections.
/// The matrices run as the file stores them when they are Q8_0 or Q4_0; an F16 or F32 file's are
/// rounded once to Q8_0 when they are staged.
pub struct HexagonVitStack {
    encoder: HexagonVisionEncoder,
}

impl HexagonVitStack {
    /// Stage `spec` on `device`, with scratch for `rows` tokens per pass.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        spec: &VitStackSpec,
        rows: usize,
    ) -> Result<Self, CeraError> {
        let weights =
            stack_weights(spec).map_err(|e| CeraError::Backend(format!("ViT stack: {e:#}")))?;
        let mut encoder = HexagonVisionEncoder::with_rows(driver, device, &weights, rows)?;
        encoder.activation = spec.activation;
        encoder.post_ln = spec.post.is_some();
        Ok(Self { encoder })
    }
}

impl VitStack for HexagonVitStack {
    fn run(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        self.encoder.run_blocks(x, tokens, None, None)
    }

    fn run_projector(&self, pooled: &[f32]) -> Result<Option<Vec<f32>>> {
        Ok(self.encoder.device_projector(pooled)?)
    }
}

/// A weight the DSP takes as it is (Q8_0 or Q4_0), or the weight rounded once to Q8_0.
pub(crate) fn npu_weight(w: &MmapWeight) -> Result<MmapWeight, CeraError> {
    if matches!(w.dtype, DType::Q8_0 | DType::Q4_0) {
        return Ok(w.clone());
    }
    let bytes = crate::backend::hexagon::quantize_f32_to_q8_0(&w.to_dense_f32(), w.cols, w.rows)?;
    Ok(MmapWeight::from_owned_bytes(
        bytes,
        DType::Q8_0,
        w.rows,
        w.cols,
    ))
}

/// The weights a [`VitStackSpec`] describes, in the shape the encoder plans from. The patch
/// embedding, positions and projector are never read by a stack pass, so they are empty.
fn stack_weights(spec: &VitStackSpec) -> Result<VisionEncoderWeights> {
    ensure!(
        !spec.blocks.is_empty(),
        "a ViT stack needs at least one block"
    );
    // the same check every GPU builder runs: without it `run_blocks` divides by zero on a
    // zero head count and the DSP reads ragged weights and short vectors as if they fit
    crate::model::vision_encoder_gpu::check_spec(spec)?;
    let config = VisionEncoderConfig {
        n_layer: spec.blocks.len(),
        n_embd: spec.width,
        n_ff: spec.ffn,
        n_head: spec.heads,
        eps: spec.eps,
        image_size: 0,
        patch_size: 1,
        n_trained_patches: 1,
        projection_dim: spec.projector.as_ref().map_or(spec.width, |p| p.mm2.rows),
        scale_factor: 1,
        image_mean: [0.0; 3],
        image_std: [1.0; 3],
        image_min_pixels: 0,
        image_max_pixels: 0,
    };
    let blocks = spec
        .blocks
        .iter()
        .map(|b| {
            Ok(VitBlockWeights {
                ln1_w: b.ln1_w.clone(),
                ln1_b: b.ln1_b.clone(),
                q_w: npu_weight(&b.q)?,
                q_b: b.q_b.clone(),
                k_w: npu_weight(&b.k)?,
                k_b: b.k_b.clone(),
                v_w: npu_weight(&b.v)?,
                v_b: b.v_b.clone(),
                o_w: npu_weight(&b.o)?,
                o_b: b.o_b.clone(),
                ln2_w: b.ln2_w.clone(),
                ln2_b: b.ln2_b.clone(),
                ffn_up_w: npu_weight(&b.up)?,
                ffn_up_b: b.up_b.clone(),
                ffn_down_w: npu_weight(&b.down)?,
                ffn_down_b: b.down_b.clone(),
            })
        })
        .collect::<Result<Vec<_>, CeraError>>()?;
    let (post_ln_w, post_ln_b) = spec.post.clone().unwrap_or_default();
    let unused = || MmapWeight::from_owned_f32(Vec::new(), 0, 0);
    Ok(VisionEncoderWeights {
        config,
        patch_embed: PatchEmbedWeights {
            conv_w: Vec::new(),
            conv_b: Vec::new(),
        },
        position_embed: Vec::new(),
        blocks,
        post_ln_w,
        post_ln_b,
        projector: match &spec.projector {
            Some(p) => ProjectorWeights {
                mm1_w: npu_weight(&p.mm1)?,
                mm1_b: p.mm1_b.clone(),
                mm2_w: npu_weight(&p.mm2)?,
                mm2_b: p.mm2_b.clone(),
            },
            None => ProjectorWeights {
                mm1_w: unused(),
                mm1_b: Vec::new(),
                mm2_w: unused(),
                mm2_b: Vec::new(),
            },
        },
    })
}

/// Stage `spec` on the Hexagon NPU, or say why not. `rows` is the most tokens one pass takes.
pub(crate) fn try_hexagon_vit_stack(spec: &VitStackSpec, rows: usize) -> Option<Arc<dyn VitStack>> {
    let context = HexagonContext::new()
        .inspect_err(|e| crate::backend::hexagon::log_context_unavailable("HexagonVitStack", e))
        .ok()?;
    let dev = crate::backend::hexagon::probe_device(
        context.driver(),
        crate::backend::hexagon::arch_override(),
    )
    .inspect_err(|e| tracing::info!("HexagonVitStack: DSP device unavailable ({e})"))
    .ok()?;
    let device = Arc::new(Mutex::new(dev));
    match HexagonVitStack::new(Arc::clone(context.driver()), device, spec, rows) {
        Ok(stack) => Some(Arc::new(stack)),
        Err(e) => {
            crate::backend::hexagon::hexagon_error!("failed to create HexagonVitStack: {e}");
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
        assert_eq!(proj.mm1_w.format, HexagonVitWeightFormat::RepackedQ8_0);
        assert_eq!(proj.mm2_w.format, HexagonVitWeightFormat::RepackedQ8_0);
        assert!(plan.total_bytes > proj.mm2_b_off);

        // A projector wider than the feed-forward has no scratch to run in.
        let mut wide = weights;
        wide.projector.mm1_w = make_q8(256, 256);
        let err = HexagonVitWeightOffsets::plan(&wide)
            .unwrap_err()
            .to_string();
        assert!(err.contains("projector hidden width 256"), "{err}");
    }

    #[test]
    fn test_ffn_padded_width_rounds_to_dsp_tile() {
        assert_eq!(ffn_padded_width(4304), 4320);
        assert_eq!(ffn_padded_width(4320), 4320);
        assert_eq!(ffn_padded_width(3072), 3072);
    }

    #[test]
    fn test_pad_rows_raw_appends_zero_rows() {
        let src: Vec<u8> = (1..=12).collect(); // 3 rows of 4 bytes
        let spec = RowPadSpec {
            rows: 3,
            row_bytes: 4,
            rows_pad: 5,
        };
        let out = pad_rows_raw(&src, spec).expect("pad rows");
        assert_eq!(out.len(), 20);
        assert_eq!(&out[..12], src.as_slice());
        assert!(out[12..].iter().all(|&b| b == 0));
        assert!(pad_rows_raw(&src[..8], spec).is_err());
    }

    #[test]
    fn test_f16_matrix_to_padded_q8_0_zero_pads_and_keeps_values() {
        // 2 rows x 48 cols of F16 (cols not a multiple of 32), padded to 64.
        let rows = 2usize;
        let cols = 48usize;
        let cols_pad = 64usize;
        // F16 bit patterns for 1.0, 2.0, 0.5 and -1.0.
        let pattern = [0x3C00u16, 0x4000, 0x3800, 0xBC00];
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for i in 0..rows * cols {
            bytes.extend_from_slice(&pattern[i % pattern.len()].to_le_bytes());
        }
        let w = MmapWeight::from_owned_bytes(bytes, DType::F16, rows, cols);
        let q8 = f16_matrix_to_padded_q8_0(&w, cols_pad).expect("quantize");
        let blocks_per_row = cols_pad / 32;
        assert_eq!(q8.len(), rows * blocks_per_row * 34);

        for r in 0..rows {
            for b in 0..blocks_per_row {
                let off = (r * blocks_per_row + b) * 34;
                let delta = crate::quant::f16_to_f32(u16::from_le_bytes([q8[off], q8[off + 1]]));
                for j in 0..32 {
                    let col = b * 32 + j;
                    let got = delta * (q8[off + 2 + j] as i8) as f32;
                    if col >= cols {
                        assert_eq!(got, 0.0, "padded column {col} must be exactly zero");
                    } else {
                        let want = crate::quant::f16_to_f32(pattern[(r * cols + col) % 4]);
                        assert!(
                            (got - want).abs() < 0.02,
                            "row {r} col {col}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_pad_f16_cols_keeps_values_and_zero_pads() {
        // 2 rows x 3 cols of F16, padded to 4 columns.
        let bytes: Vec<u8> = (1u8..=12).collect();
        let w = MmapWeight::from_owned_bytes(bytes, DType::F16, 2, 3);
        let out = pad_f16_cols(&w, 4).expect("pad");
        assert_eq!(out.len(), 2 * 4 * 2);
        assert_eq!(&out[0..6], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&out[6..8], &[0, 0]);
        assert_eq!(&out[8..14], &[7, 8, 9, 10, 11, 12]);
        assert_eq!(&out[14..16], &[0, 0]);

        let q8 = MmapWeight::from_owned_bytes(vec![0u8; 34], DType::Q8_0, 1, 32);
        assert!(pad_f16_cols(&q8, 32).is_err());
        assert!(f16_matrix_to_padded_q8_0(&q8, 32).is_err());
    }

    #[test]
    fn test_plan_native_f16_and_requant_fallback_for_unaligned_ffn() {
        let make_q8 = |rows: usize, cols: usize| -> MmapWeight {
            let n_blocks = (rows * cols) / 32;
            MmapWeight::from_owned_bytes(vec![0u8; n_blocks * 34], DType::Q8_0, rows, cols)
        };
        // n_ff = 112 is not a multiple of 32, so ffn_down is F16 in the GGUF.
        let cfg = VisionEncoderConfig {
            n_layer: 1,
            n_embd: 64,
            n_ff: 112,
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
        let mut weights = VisionEncoderWeights {
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
                ffn_up_w: make_q8(112, 64),
                ffn_up_b: vec![0.0f32; 112],
                ffn_down_w: MmapWeight::from_owned_bytes(
                    vec![0u8; 64 * 112 * 2],
                    DType::F16,
                    64,
                    112,
                ),
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

        // Default `plan` keeps F16 `ffn_down` in native F16, padded to 128 cols.
        let native = HexagonVitWeightOffsets::plan(&weights).expect("native f16 plan");
        let blk = &native.blocks[0];
        assert_eq!(
            blk.ffn_up_w.rows, 128,
            "ffn_up rows padded to the 32-wide tile"
        );
        assert_eq!(blk.ffn_down_w.format, HexagonVitWeightFormat::F16);
        assert_eq!(blk.ffn_down_w.cols, 128, "K padded to the 32-wide tile");
        assert_eq!(blk.ffn_down_w.rows, 64);
        assert_eq!(blk.ffn_down_w.size_bytes, 64 * 128 * 2);

        // `plan_with(..., VitF16Policy::RequantQ8_0)` requantizes F16 `ffn_down` to padded Q8_0.
        let requant = HexagonVitWeightOffsets::plan_with(&weights, VitF16Policy::RequantQ8_0)
            .expect("requant plan");
        let rblk = &requant.blocks[0];
        assert_eq!(rblk.ffn_up_w.rows, 128);
        assert_eq!(rblk.ffn_down_w.cols, 128);
        assert_eq!(rblk.ffn_down_w.rows, 64);
        assert_eq!(rblk.ffn_down_w.format, HexagonVitWeightFormat::RepackedQ8_0);

        // If `ffn_down_w.rows` is not a multiple of 32 (e.g. 48), NativeF16 falls back to RepackedQ8_0.
        weights.blocks[0].ffn_down_w =
            MmapWeight::from_owned_bytes(vec![0u8; 48 * 112 * 2], DType::F16, 48, 112);
        let unaligned_rows = HexagonVitWeightOffsets::plan_with(&weights, VitF16Policy::NativeF16)
            .expect("unaligned rows fallback plan");
        assert_eq!(
            unaligned_rows.blocks[0].ffn_down_w.format,
            HexagonVitWeightFormat::RepackedQ8_0
        );
    }

    #[test]
    fn a_dense_matrix_is_rounded_once_to_q8_0_and_a_quantized_one_kept() {
        let values: Vec<f32> = (0..64 * 32)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 24.0)
            .collect();
        let dense = MmapWeight::from_owned_f32(values.clone(), 64, 32);
        let rounded = npu_weight(&dense).unwrap();
        assert_eq!(
            (rounded.dtype, rounded.rows, rounded.cols),
            (DType::Q8_0, 64, 32)
        );
        // dequantizing the rounded matrix stays within a Q8_0 step of the original
        let back = rounded.to_dense_f32();
        let worst = values
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 2.0 / 127.0 + 1e-6, "worst error {worst}");
        // a matrix that is already Q8_0 is the same bytes
        let again = npu_weight(&rounded).unwrap();
        assert_eq!(again.data(), rounded.data());
    }

    #[test]
    fn a_stack_spec_becomes_weights_with_its_activation_widths_and_projector() {
        use crate::model::vision_encoder_gpu::{VitStackBlock, VitStackProjector};
        let (d, ff) = (64usize, 128usize);
        let w = |rows: usize, cols: usize| {
            MmapWeight::from_owned_f32(vec![0.25; rows * cols], rows, cols)
        };
        let vec = |n: usize| vec![1.0f32; n];
        let block = || VitStackBlock {
            ln1_w: vec(d),
            ln1_b: vec(d),
            q: w(d, d),
            q_b: vec(d),
            k: w(d, d),
            k_b: vec(d),
            v: w(d, d),
            v_b: vec(d),
            o: w(d, d),
            o_b: vec(d),
            ln2_w: vec(d),
            ln2_b: vec(d),
            up: w(ff, d),
            up_b: vec(ff),
            down: w(d, ff),
            down_b: vec(d),
        };
        let mut spec = VitStackSpec {
            width: d,
            heads: 1,
            ffn: ff,
            eps: 1e-5,
            activation: VitStackActivation::Relu,
            blocks: vec![block(), block()],
            post: None,
            projector: Some(VitStackProjector {
                mm1: w(96, d * 4),
                mm1_b: vec(96),
                mm2: w(32, 96),
                mm2_b: vec(32),
            }),
        };
        let weights = stack_weights(&spec).unwrap();
        assert_eq!(weights.blocks.len(), 2);
        assert_eq!((weights.config.n_embd, weights.config.n_ff), (d, ff));
        assert_eq!(weights.config.projection_dim, 32);
        assert!(weights.blocks.iter().all(|b| b.q_w.dtype == DType::Q8_0));
        assert_eq!(weights.projector.mm1_w.dtype, DType::Q8_0);
        // the planner takes the projector and the dense blocks as Q8_0
        let offsets = HexagonVitWeightOffsets::plan(&weights).unwrap();
        assert!(offsets.projector.is_some());
        // without a projector the planner sees none
        spec.projector = None;
        let plain = stack_weights(&spec).unwrap();
        assert!(
            HexagonVitWeightOffsets::plan(&plain)
                .unwrap()
                .projector
                .is_none()
        );
        // a stack needs a block, and heads that divide its width
        let mut headless = VitStackSpec {
            width: d,
            heads: 0,
            ffn: ff,
            eps: 1e-5,
            activation: VitStackActivation::Relu,
            blocks: vec![block()],
            post: None,
            projector: None,
        };
        let err = stack_weights(&headless).err().unwrap().to_string();
        assert!(err.contains("heads"), "{err}");
        headless.heads = 3; // 64 is not a multiple of 3
        assert!(stack_weights(&headless).is_err());
        headless.heads = 4;
        assert!(stack_weights(&headless).is_ok());
        spec.blocks.clear();
        assert!(stack_weights(&spec).is_err());
    }
}

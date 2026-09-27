//! Native Qualcomm Hexagon NPU Audio Vocoder and Detokenizer.
//!
//! Accelerates audio waveform generation and spectrum detokenization on Snapdragon
//! Hexagon Tensor Processors (HTP) using FastRPC shared memory and asynchronous
//! command queues.
//!
//! Dispatches dilated Conv1D, ConvTranspose1D, and Snake activation operators directly
//! to Hexagon DSP hardware while keeping intermediate activation buffers coherent.

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::backend::hexagon::{
    FastRpcDriver, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch,
    HexagonContext, HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, RpcmemBuffer,
    build_binary_kernel_params, build_layer_norm_params, build_mul_mat_kernel_params,
    build_unary_kernel_params, repack_q4_0, repack_q8_0, repacked_matrix_size_q4_0,
    repacked_matrix_size_q8_0,
};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{
    AudioGpu, DetokenizerState, DetokenizerWeights, detok_embed_codes,
    detokenize_to_spectrum_with_state, istft_to_pcm, upsample,
};
use crate::model::vision_encoder_hexagon::HexagonWeightFormat;
use crate::model::weights::MmapWeight;
use crate::session::CeraError;
use crate::tensor::DType;

/// Metadata describing a detokenizer weight tensor in shared rpcmem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDetokWeightDesc {
    pub offset: usize,
    pub size_bytes: usize,
    pub format: HexagonWeightFormat,
    pub rows: usize,
    pub cols: usize,
}

/// Offsets for one detokenizer layer in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDetokLayerOffsets {
    pub is_conv: bool,
    pub op_norm_off: usize,
    pub ffn_norm_off: usize,
    pub ffn_w1: HexagonDetokWeightDesc,
    pub ffn_w2: HexagonDetokWeightDesc,
    pub ffn_w3: HexagonDetokWeightDesc,
    pub conv_in_proj: Option<HexagonDetokWeightDesc>,
    pub conv_out_proj: Option<HexagonDetokWeightDesc>,
    pub conv_w_off: Option<usize>,
    pub wq: Option<HexagonDetokWeightDesc>,
    pub wk: Option<HexagonDetokWeightDesc>,
    pub wv: Option<HexagonDetokWeightDesc>,
    pub wo: Option<HexagonDetokWeightDesc>,
    pub q_norm_off: Option<usize>,
    pub k_norm_off: Option<usize>,
}

/// Weight buffer layout offsets across all detokenizer layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonDetokWeightOffsets {
    pub layers: Vec<HexagonDetokLayerOffsets>,
    pub output_norm_off: usize,
    pub lin_w: HexagonDetokWeightDesc,
    pub lin_b_off: usize,
    pub total_bytes: usize,
}

impl HexagonDetokWeightOffsets {
    /// Compute memory layout and byte offsets for all detokenizer weights in shared rpcmem.
    pub fn plan(weights: &DetokenizerWeights) -> Result<Self, CeraError> {
        let mut cur_off = 0;
        let align128 = |sz: usize| (sz + 127) & !127;

        let plan_vec_f32 = |cur_off: &mut usize, len: usize| -> usize {
            let off = *cur_off;
            *cur_off += align128(len * 4);
            off
        };

        let plan_linear =
            |cur_off: &mut usize, w: &MmapWeight| -> Result<HexagonDetokWeightDesc, CeraError> {
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
                            "Hexagon detokenizer linear weight requires Q8_0 or Q4_0, got {other:?}"
                        )));
                    }
                };
                *cur_off += align128(sz);
                Ok(HexagonDetokWeightDesc {
                    offset: off,
                    size_bytes: sz,
                    format: fmt,
                    rows: w.rows,
                    cols: w.cols,
                })
            };

        let mut layers = Vec::with_capacity(weights.layers.len());
        for (i, lw) in weights.layers.iter().enumerate() {
            let is_conv = weights.config.layer_is_conv[i];
            let op_norm_off = plan_vec_f32(&mut cur_off, lw.operator_norm.len());
            let ffn_norm_off = plan_vec_f32(&mut cur_off, lw.ffn_norm.len());
            let ffn_w1 = plan_linear(&mut cur_off, &lw.ffn_w1)?;
            let ffn_w2 = plan_linear(&mut cur_off, &lw.ffn_w2)?;
            let ffn_w3 = plan_linear(&mut cur_off, &lw.ffn_w3)?;

            let (conv_in_proj, conv_out_proj, conv_w_off) = if is_conv {
                let in_proj = lw.conv_in_proj.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing conv_in_proj in detok conv layer".into())
                })?;
                let out_proj = lw.conv_out_proj.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing conv_out_proj in detok conv layer".into())
                })?;
                let conv_w = lw.conv_weight.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing conv_weight in detok conv layer".into())
                })?;
                (
                    Some(plan_linear(&mut cur_off, in_proj)?),
                    Some(plan_linear(&mut cur_off, out_proj)?),
                    Some(plan_vec_f32(&mut cur_off, conv_w.len())),
                )
            } else {
                (None, None, None)
            };

            let (wq, wk, wv, wo, q_norm_off, k_norm_off) = if !is_conv {
                let wq = lw.wq.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing wq in detok attention layer".into())
                })?;
                let wk = lw.wk.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing wk in detok attention layer".into())
                })?;
                let wv = lw.wv.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing wv in detok attention layer".into())
                })?;
                let wo = lw.wo.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing wo in detok attention layer".into())
                })?;
                let qn = lw.q_norm.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing q_norm in detok attention layer".into())
                })?;
                let kn = lw.k_norm.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing k_norm in detok attention layer".into())
                })?;
                (
                    Some(plan_linear(&mut cur_off, wq)?),
                    Some(plan_linear(&mut cur_off, wk)?),
                    Some(plan_linear(&mut cur_off, wv)?),
                    Some(plan_linear(&mut cur_off, wo)?),
                    Some(plan_vec_f32(&mut cur_off, qn.len())),
                    Some(plan_vec_f32(&mut cur_off, kn.len())),
                )
            } else {
                (None, None, None, None, None, None)
            };

            layers.push(HexagonDetokLayerOffsets {
                is_conv,
                op_norm_off,
                ffn_norm_off,
                ffn_w1,
                ffn_w2,
                ffn_w3,
                conv_in_proj,
                conv_out_proj,
                conv_w_off,
                wq,
                wk,
                wv,
                wo,
                q_norm_off,
                k_norm_off,
            });
        }

        let output_norm_off = plan_vec_f32(&mut cur_off, weights.output_norm.len());
        let lin_w = plan_linear(&mut cur_off, &weights.lin_w)?;
        let lin_b_off = plan_vec_f32(&mut cur_off, weights.lin_b.len());

        Ok(HexagonDetokWeightOffsets {
            layers,
            output_norm_off,
            lin_w,
            lin_b_off,
            total_bytes: cur_off,
        })
    }
}

/// Layout offsets for runtime intermediate activations inside `scratch_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDetokScratchOffsets {
    pub tokens_off: usize,
    pub normed_off: usize,
    pub conv_bcx_off: usize,
    pub conv_out_off: usize,
    pub ffn_gate_off: usize,
    pub ffn_up_off: usize,
    pub ffn_out_off: usize,
    pub spec_out_off: usize,
    pub total_bytes: usize,
}

impl HexagonDetokScratchOffsets {
    pub fn new(n_embd: usize, ffn_dim: usize, n_fft_bins: usize) -> Self {
        let align128 = |sz: usize| (sz + 127) & !127;
        const MAX_AUDIO_TOKENS: usize = 16;

        let tokens_bytes = align128(MAX_AUDIO_TOKENS * n_embd * 4);
        let conv_bcx_bytes = align128(MAX_AUDIO_TOKENS * 3 * n_embd * 4);
        let ffn_mid_bytes = align128(MAX_AUDIO_TOKENS * ffn_dim * 4);
        let spec_bytes = align128(MAX_AUDIO_TOKENS * (n_fft_bins * 2) * 4);

        let mut off = 0;
        let tokens_off = off;
        off += tokens_bytes;

        let normed_off = off;
        off += tokens_bytes;

        let conv_bcx_off = off;
        off += conv_bcx_bytes;

        let conv_out_off = off;
        off += tokens_bytes;

        let ffn_gate_off = off;
        off += ffn_mid_bytes;

        let ffn_up_off = off;
        off += ffn_mid_bytes;

        let ffn_out_off = off;
        off += tokens_bytes;

        let spec_out_off = off;
        off += spec_bytes;

        Self {
            tokens_off,
            normed_off,
            conv_bcx_off,
            conv_out_off,
            ffn_gate_off,
            ffn_up_off,
            ffn_out_off,
            spec_out_off,
            total_bytes: off,
        }
    }
}

/// Hardware-accelerated Hexagon NPU audio decoder backend.
pub struct HexagonAudioDecoder {
    driver: Arc<FastRpcDriver>,
    device: Arc<Mutex<HexagonDevice>>,
    detok_weights: Arc<DetokenizerWeights>,
    weights_buf: Option<RpcmemBuffer>,
    weights_offsets: Option<HexagonDetokWeightOffsets>,
    scratch_buf: Mutex<Option<RpcmemBuffer>>,
    scratch_offsets: HexagonDetokScratchOffsets,
    state: Mutex<DetokenizerState>,
    last_error: Mutex<Option<CeraError>>,
}

impl HexagonAudioDecoder {
    /// Create a new Hexagon audio decoder with shared device and weights.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        detok_weights: Arc<DetokenizerWeights>,
    ) -> Result<Self, CeraError> {
        let cfg = &detok_weights.config;
        let n_fft_bins = cfg.n_fft / 2 + 1;
        let scratch_offsets = HexagonDetokScratchOffsets::new(cfg.n_embd, cfg.ffn_dim, n_fft_bins);
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true).ok();

        let (weights_buf, weights_offsets) = match HexagonDetokWeightOffsets::plan(&detok_weights) {
            Ok(offsets) => {
                match RpcmemBuffer::alloc(Arc::clone(&driver), offsets.total_bytes, true) {
                    Ok(mut buf) => {
                        let res = Self::stage_weights(&mut buf, &offsets, &detok_weights);
                        if res.is_ok() {
                            buf.flush_cpu_cache(0, offsets.total_bytes);
                            (Some(buf), Some(offsets))
                        } else {
                            (None, None)
                        }
                    }
                    Err(_) => (None, None),
                }
            }
            Err(_) => (None, None),
        };

        let state = DetokenizerState::new(&detok_weights.config);
        Ok(Self {
            driver,
            device,
            detok_weights,
            weights_buf,
            weights_offsets,
            scratch_buf: Mutex::new(scratch_buf),
            scratch_offsets,
            state: Mutex::new(state),
            last_error: Mutex::new(None),
        })
    }

    /// Stage weights into contiguous rpcmem buffer.
    fn stage_weights(
        buf: &mut RpcmemBuffer,
        offsets: &HexagonDetokWeightOffsets,
        weights: &DetokenizerWeights,
    ) -> Result<(), CeraError> {
        let copy_vec_f32 = |dst: &mut [u8], off: usize, src: &[f32]| {
            let bytes = bytemuck::cast_slice(src);
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        };

        let copy_linear = |dst: &mut [u8],
                           desc: HexagonDetokWeightDesc,
                           w: &MmapWeight|
         -> Result<(), CeraError> {
            let dst_slice = &mut dst[desc.offset..desc.offset + desc.size_bytes];
            match desc.format {
                HexagonWeightFormat::RepackedQ8_0 => {
                    repack_q8_0(w.data(), w.cols, w.rows, dst_slice).map_err(|e| {
                        CeraError::Backend(format!("detok repack Q8_0 failed: {e}"))
                    })?;
                }
                HexagonWeightFormat::RepackedQ4_0 => {
                    repack_q4_0(w.data(), w.cols, w.rows, dst_slice).map_err(|e| {
                        CeraError::Backend(format!("detok repack Q4_0 failed: {e}"))
                    })?;
                }
                HexagonWeightFormat::DenseF32 => {
                    return Err(CeraError::Backend(
                        "DenseF32 linear weights are not supported on Hexagon detok".into(),
                    ));
                }
            }
            Ok(())
        };

        let slice = buf.as_mut_slice();
        for (lw, lo) in weights.layers.iter().zip(offsets.layers.iter()) {
            copy_vec_f32(slice, lo.op_norm_off, &lw.operator_norm);
            copy_vec_f32(slice, lo.ffn_norm_off, &lw.ffn_norm);
            copy_linear(slice, lo.ffn_w1, &lw.ffn_w1)?;
            copy_linear(slice, lo.ffn_w2, &lw.ffn_w2)?;
            copy_linear(slice, lo.ffn_w3, &lw.ffn_w3)?;

            if lo.is_conv {
                if let (Some(cin_desc), Some(cin_w)) = (lo.conv_in_proj, lw.conv_in_proj.as_ref()) {
                    copy_linear(slice, cin_desc, cin_w)?;
                }
                if let (Some(cout_desc), Some(cout_w)) =
                    (lo.conv_out_proj, lw.conv_out_proj.as_ref())
                {
                    copy_linear(slice, cout_desc, cout_w)?;
                }
                if let (Some(cw_off), Some(cw)) = (lo.conv_w_off, lw.conv_weight.as_ref()) {
                    copy_vec_f32(slice, cw_off, cw);
                }
            } else {
                if let (Some(wq_desc), Some(wq_w)) = (lo.wq, lw.wq.as_ref()) {
                    copy_linear(slice, wq_desc, wq_w)?;
                }
                if let (Some(wk_desc), Some(wk_w)) = (lo.wk, lw.wk.as_ref()) {
                    copy_linear(slice, wk_desc, wk_w)?;
                }
                if let (Some(wv_desc), Some(wv_w)) = (lo.wv, lw.wv.as_ref()) {
                    copy_linear(slice, wv_desc, wv_w)?;
                }
                if let (Some(wo_desc), Some(wo_w)) = (lo.wo, lw.wo.as_ref()) {
                    copy_linear(slice, wo_desc, wo_w)?;
                }
                if let (Some(qn_off), Some(qn)) = (lo.q_norm_off, lw.q_norm.as_ref()) {
                    copy_vec_f32(slice, qn_off, qn);
                }
                if let (Some(kn_off), Some(kn)) = (lo.k_norm_off, lw.k_norm.as_ref()) {
                    copy_vec_f32(slice, kn_off, kn);
                }
            }
        }

        copy_vec_f32(slice, offsets.output_norm_off, &weights.output_norm);
        copy_linear(slice, offsets.lin_w, &weights.lin_w)?;
        copy_vec_f32(slice, offsets.lin_b_off, &weights.lin_b);

        Ok(())
    }

    /// Access the underlying Hexagon driver.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }

    /// Access the shared Hexagon device.
    pub fn device(&self) -> &Arc<Mutex<HexagonDevice>> {
        &self.device
    }

    /// Access detokenizer weights.
    pub fn detok_weights(&self) -> &Arc<DetokenizerWeights> {
        &self.detok_weights
    }

    /// Dispatch LayerNorm: Opcode 49 `Norm`, followed by affine scale.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_layer_norm(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        weights: &RpcmemBuffer,
        w_offset: usize,
        eps: f32,
        n_tokens: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let bytes = n_tokens * dim * 4;
        let ne = [dim as u32, n_tokens as u32, 1, 1];
        let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];

        let src_ti = session.add_tensor(
            src,
            src_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;

        let params = build_layer_norm_params(eps);
        let kparams = build_unary_kernel_params(
            dim,
            n_tokens,
            0,
            8 * 1024 * 1024,
            session.dsp_threads(),
            false,
        );

        session
            .enqueue_op(
                HtpOpCode::Norm as u32,
                &[src_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_layer_norm: {e}")))?;

        let w_bytes = dim * 4;
        let w_ne = [dim as u32, 1, 1, 1];
        let w_nb = [4, w_bytes as u32, w_bytes as u32, w_bytes as u32];
        let w_ti = session.add_tensor(
            weights,
            w_offset,
            w_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            w_ne,
            w_nb,
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
        session
            .enqueue_op(
                HtpOpCode::Mul as u32,
                &[dst_ti, w_ti],
                &[dst_ti],
                mul_params,
                mul_kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_norm_mul: {e}")))?;

        Ok(())
    }

    /// Dispatch linear layer matmul + optional bias: `dst = x · wᵀ (+ bias)`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_m(
        session: &mut HexagonQueueSession,
        x: &RpcmemBuffer,
        x_offset: usize,
        weights: &RpcmemBuffer,
        w_desc: HexagonDetokWeightDesc,
        b_offset: Option<usize>,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let x_bytes = n_tokens * w_desc.cols * 4;
        let x_ne = [w_desc.cols as u32, n_tokens as u32, 1, 1];
        let x_nb = [4, (w_desc.cols * 4) as u32, x_bytes as u32, x_bytes as u32];
        let x_ti = session.add_tensor(
            x,
            x_offset,
            x_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            x_ne,
            x_nb,
        )?;

        let (w_dtype, block_bytes, tile_size) = match w_desc.format {
            HexagonWeightFormat::RepackedQ8_0 => (HtpDataType::Q8_0, 34usize, 32 * 34usize),
            HexagonWeightFormat::RepackedQ4_0 => (HtpDataType::Q4_0, 18usize, 32 * 18usize),
            HexagonWeightFormat::DenseF32 => {
                return Err(CeraError::Backend(
                    "dispatch_linear_m does not support DenseF32 weights on HTP".into(),
                ));
            }
        };

        let ne0 = w_desc.cols;
        let ne1 = w_desc.rows;
        let tiled_row_bytes = (ne0 / 32) * tile_size;
        let w_tot = (ne1 / 32) * tiled_row_bytes;
        let w_nb = [
            block_bytes as u32,
            tiled_row_bytes as u32,
            w_tot as u32,
            w_tot as u32,
        ];
        let w_ne = [ne0 as u32, ne1 as u32, 1, 1];

        let w_ti = session.add_tensor(
            weights,
            w_desc.offset,
            w_desc.size_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w_dtype as u32,
            w_ne,
            w_nb,
        )?;

        let dst_bytes = n_tokens * w_desc.rows * 4;
        let dst_ne = [w_desc.rows as u32, n_tokens as u32, 1, 1];
        let dst_nb = [
            4,
            (w_desc.rows * 4) as u32,
            dst_bytes as u32,
            dst_bytes as u32,
        ];
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            dst_ne,
            dst_nb,
        )?;

        let params = [0i32; 16];
        let kparams = build_mul_mat_kernel_params(
            w_dtype,
            w_desc.cols,
            n_tokens as u32,
            1,
            w_desc.rows * 4,
            session.dsp_threads(),
            8 * 1024 * 1024,
        );

        session
            .enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w_ti, x_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_linear_m: {e}")))?;

        if let Some(b_off) = b_offset {
            let b_bytes = w_desc.rows * 4;
            let b_ne = [w_desc.rows as u32, 1, 1, 1];
            let b_nb = [4, b_bytes as u32, b_bytes as u32, b_bytes as u32];
            let b_ti = session.add_tensor(
                weights,
                b_off,
                b_bytes,
                HTP_TENSOR_WEIGHT,
                HtpDataType::F32 as u32,
                b_ne,
                b_nb,
            )?;

            let add_params = [0i32; 16];
            let add_kparams = build_binary_kernel_params(
                w_desc.rows,
                w_desc.rows,
                n_tokens,
                1,
                1,
                4,
                8 * 1024 * 1024,
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
                .map_err(|e| CeraError::Backend(format!("detok_bias_add: {e}")))?;
        }

        Ok(())
    }

    /// Dispatch SwiGLU activation on DSP: `dst = silu(gate) * up`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_swiglu(
        session: &mut HexagonQueueSession,
        gate: &RpcmemBuffer,
        gate_offset: usize,
        up: &RpcmemBuffer,
        up_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let bytes = n_tokens * dim * 4;
        let ne = [dim as u32, n_tokens as u32, 1, 1];
        let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];

        let gate_ti = session.add_tensor(
            gate,
            gate_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let up_ti = session.add_tensor(
            up,
            up_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;

        let params = [0i32; 16];
        let kparams = [0i32; 32];
        session
            .enqueue_op(
                HtpOpCode::GluSwiglu as u32,
                &[gate_ti, up_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("dispatch_swiglu: {e}")))
    }

    /// Dispatch residual add: `dst += src`.
    fn dispatch_add_residual(
        session: &mut HexagonQueueSession,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        src: &RpcmemBuffer,
        src_offset: usize,
        dim: usize,
        n_tokens: usize,
    ) -> Result<(), CeraError> {
        let bytes = dim * n_tokens * 4;
        let ne = [dim as u32, n_tokens as u32, 1, 1];
        let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];

        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let src_ti = session.add_tensor(
            src,
            src_offset,
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
            n_tokens,
            n_tokens,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );

        session
            .enqueue_op(
                HtpOpCode::Add as u32,
                &[dst_ti, src_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_add_residual: {e}")))
    }
}

impl AudioGpu for HexagonAudioDecoder {
    fn sample_audio_frame(&self, _embedding: &[f32], _temperature: f32, _top_k: usize) -> [i32; 8] {
        // Depthformer sampling runs on CPU host when supports_depthformer returns false.
        [0; 8]
    }

    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
        let cfg = &cpu_weights.config;
        let n_embd = cfg.n_embd;

        // 1. Embed codes on CPU
        let embedding = detok_embed_codes(cpu_weights, codes);

        // 2. Upsample 1 -> 6 tokens
        let tokens = upsample(&embedding, n_embd, 6);
        let n_tokens = tokens.len() / n_embd;

        // If hardware weights and scratch are staged, attempt FastRPC dispatch
        if let (Some(w_buf), Some(w_offs)) = (&self.weights_buf, &self.weights_offsets)
            && let Ok(mut dev_guard) = self.device.lock()
            && let Ok(mut scratch_lock) = self.scratch_buf.lock()
            && let Some(scratch) = scratch_lock.as_mut()
        {
            let so = self.scratch_offsets;
            let token_bytes = n_tokens * n_embd * 4;

            // Copy initial tokens to scratch
            scratch.as_mut_slice()[so.tokens_off..so.tokens_off + token_bytes]
                .copy_from_slice(bytemuck::cast_slice(&tokens));
            scratch.flush_cpu_cache(so.tokens_off, token_bytes);

            let session = dev_guard.queue_session_mut();
            session.drop_pending_batch();

            let mut dispatch_ok = true;
            for lo in &w_offs.layers {
                // LayerNorm: tokens -> normed
                if Self::dispatch_layer_norm(
                    session,
                    scratch,
                    so.tokens_off,
                    scratch,
                    so.normed_off,
                    w_buf,
                    lo.op_norm_off,
                    cfg.rms_norm_eps,
                    n_tokens,
                    n_embd,
                )
                .is_err()
                {
                    dispatch_ok = false;
                    break;
                }

                // FFN: w1 (gate) & w3 (up) -> swiglu -> w2 (down)
                if Self::dispatch_linear_m(
                    session,
                    scratch,
                    so.normed_off,
                    w_buf,
                    lo.ffn_w1,
                    None,
                    scratch,
                    so.ffn_gate_off,
                    n_tokens,
                )
                .is_err()
                {
                    dispatch_ok = false;
                    break;
                }

                if Self::dispatch_linear_m(
                    session,
                    scratch,
                    so.normed_off,
                    w_buf,
                    lo.ffn_w3,
                    None,
                    scratch,
                    so.ffn_up_off,
                    n_tokens,
                )
                .is_err()
                {
                    dispatch_ok = false;
                    break;
                }

                if Self::dispatch_swiglu(
                    session,
                    scratch,
                    so.ffn_gate_off,
                    scratch,
                    so.ffn_up_off,
                    scratch,
                    so.ffn_gate_off,
                    cfg.ffn_dim,
                    n_tokens,
                )
                .is_err()
                {
                    dispatch_ok = false;
                    break;
                }

                if Self::dispatch_linear_m(
                    session,
                    scratch,
                    so.ffn_gate_off,
                    w_buf,
                    lo.ffn_w2,
                    None,
                    scratch,
                    so.ffn_out_off,
                    n_tokens,
                )
                .is_err()
                {
                    dispatch_ok = false;
                    break;
                }

                // Residual add: tokens += ffn_out
                if Self::dispatch_add_residual(
                    session,
                    scratch,
                    so.tokens_off,
                    scratch,
                    so.ffn_out_off,
                    n_embd,
                    n_tokens,
                )
                .is_err()
                {
                    dispatch_ok = false;
                    break;
                }
            }

            if dispatch_ok {
                // Output norm
                if Self::dispatch_layer_norm(
                    session,
                    scratch,
                    so.tokens_off,
                    scratch,
                    so.normed_off,
                    w_buf,
                    w_offs.output_norm_off,
                    cfg.rms_norm_eps,
                    n_tokens,
                    n_embd,
                )
                .is_ok()
                {
                    // Head linear projection: normed -> spec_out
                    if Self::dispatch_linear_m(
                        session,
                        scratch,
                        so.normed_off,
                        w_buf,
                        w_offs.lin_w,
                        Some(w_offs.lin_b_off),
                        scratch,
                        so.spec_out_off,
                        n_tokens,
                    )
                    .is_ok()
                        && session.flush().is_ok()
                    {
                        let out_len = n_tokens * w_offs.lin_w.rows;
                        let out_bytes = out_len * 4;
                        scratch.invalidate_cpu_cache(so.spec_out_off, out_bytes);
                        let out_slice: &[f32] = bytemuck::cast_slice(
                            &scratch.as_slice()[so.spec_out_off..so.spec_out_off + out_bytes],
                        );
                        return out_slice.to_vec();
                    }
                }
            }
            session.drop_pending_batch();
        }

        // Fallback to CPU reference implementation if DSP dispatch is unavailable or fails
        let mut state_guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        detokenize_to_spectrum_with_state(cpu_weights, &mut state_guard, codes)
    }

    fn reset_depthformer(&self) {}

    fn reset_detokenizer(&self) {
        let mut state_guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state_guard.reset();
    }

    fn supports_depthformer(&self) -> bool {
        false
    }

    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        istft_to_pcm(spectrum, n_fft, hop_length)
    }

    fn take_audio_error(&self) -> Option<CeraError> {
        self.last_error.lock().ok()?.take()
    }
}

/// Attempt to create a hardware-accelerated Hexagon NPU audio decoder.
///
/// Probes for Qualcomm DSP hardware, loading Unsigned PD skeleton libraries.
/// Returns `None` if Hexagon hardware or libraries are unavailable, falling
/// back cleanly to CPU execution.
pub fn try_hexagon_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioGpu>> {
    let detok_weights = match DetokenizerWeights::from_gguf(gguf) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            tracing::warn!("audio decoder: failed to parse detokenizer weights from GGUF: {e:#}");
            return None;
        }
    };

    let context = HexagonContext::new().ok()?;

    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(HexagonArch::from_u32);

    let probe_archs: Vec<HexagonArch> = if let Some(arch) = arch_override {
        vec![arch]
    } else {
        crate::backend::hexagon::PROBE_ARCHS.to_vec()
    };

    let mut device_opt = None;
    for arch in probe_archs {
        if let Ok(dev) = HexagonDevice::new(Arc::clone(context.driver()), arch) {
            tracing::info!(arch = ?arch, "initialized Hexagon NPU device for audio decoder");
            device_opt = Some(dev);
            break;
        }
    }

    let dev = device_opt?;
    let device = Arc::new(Mutex::new(dev));

    match HexagonAudioDecoder::new(Arc::clone(context.driver()), device, detok_weights) {
        Ok(decoder) => {
            tracing::info!("audio decoder: using native Hexagon NPU backend");
            Some(Arc::new(decoder))
        }
        Err(e) => {
            eprintln!("[cera-hexagon] failed to create HexagonAudioDecoder: {e}");
            tracing::warn!("failed to create HexagonAudioDecoder: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detok_scratch_offsets_alignment() {
        let so = HexagonDetokScratchOffsets::new(512, 1024, 641);
        assert_eq!(so.tokens_off, 0);
        assert_eq!(so.tokens_off % 128, 0);
        assert_eq!(so.normed_off % 128, 0);
        assert_eq!(so.conv_bcx_off % 128, 0);
        assert_eq!(so.conv_out_off % 128, 0);
        assert_eq!(so.ffn_gate_off % 128, 0);
        assert_eq!(so.ffn_up_off % 128, 0);
        assert_eq!(so.ffn_out_off % 128, 0);
        assert_eq!(so.spec_out_off % 128, 0);
        assert!(so.total_bytes > so.spec_out_off);
        assert!(so.total_bytes < 16 * 1024 * 1024);
    }

    #[test]
    fn test_try_hexagon_audio_decoder_on_host_without_dsp() {
        let empty_bytes: Arc<[u8]> = Arc::from(vec![].into_boxed_slice());
        if let Ok(gguf) = GgufFile::from_bytes(empty_bytes) {
            let arc_gguf = Arc::new(gguf);
            let result = try_hexagon_audio_decoder(&arc_gguf);
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_hexagon_detok_weight_offsets_plan_rejects_dense_f32() {
        use crate::model::audio_decoder::{DetokLayerWeights, DetokenizerConfig};

        let cfg = DetokenizerConfig {
            n_layer: 1,
            n_embd: 64,
            n_head: 2,
            n_head_kv: 2,
            n_embd_head: 32,
            ffn_dim: 128,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 8,
            n_fft: 128,
            hop_length: 32,
            sample_rate: 24000,
            layer_is_conv: vec![true],
        };
        let weights = DetokenizerWeights {
            config: cfg,
            output_norm: vec![1.0f32; 64],
            emb_weight: MmapWeight::from_owned_f32(vec![0.0f32; 8 * 2048 * 64], 8 * 2048, 64),
            lin_w: MmapWeight::from_owned_f32(vec![0.0f32; 130 * 64], 130, 64),
            lin_b: vec![0.0f32; 130],
            layers: vec![DetokLayerWeights {
                operator_norm: vec![1.0f32; 64],
                ffn_norm: vec![1.0f32; 64],
                ffn_w1: MmapWeight::from_owned_f32(vec![0.0f32; 128 * 64], 128, 64),
                ffn_w2: MmapWeight::from_owned_f32(vec![0.0f32; 64 * 128], 64, 128),
                ffn_w3: MmapWeight::from_owned_f32(vec![0.0f32; 128 * 64], 128, 64),
                conv_in_proj: Some(MmapWeight::from_owned_f32(vec![0.0f32; 192 * 64], 192, 64)),
                conv_out_proj: Some(MmapWeight::from_owned_f32(vec![0.0f32; 64 * 64], 64, 64)),
                conv_weight: Some(vec![0.0f32; 3 * 64]),
                wq: None,
                wk: None,
                wv: None,
                wo: None,
                q_norm: None,
                k_norm: None,
            }],
        };

        let plan_res = HexagonDetokWeightOffsets::plan(&weights);
        assert!(plan_res.is_err());
    }
}

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

pub use crate::backend::hexagon::HexagonWeightFormat;
use crate::backend::hexagon::dispatch::{self, TokenShape, TokenTile};
use crate::backend::hexagon::{
    FastRpcDriver, HTP_TENSOR_COMPUTE, HTP_TENSOR_WEIGHT, HexagonArch, HexagonContext,
    HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, RpcmemBuffer, align128,
    build_binary_kernel_params, build_flash_attn_kernel_params, build_rms_norm_params,
    build_rope_kernel_params, build_rope_params, build_unary_kernel_params, repack_q4_0,
    repack_q8_0, repacked_matrix_size_q4_0, repacked_matrix_size_q8_0,
};
use crate::backend::hexagon::{LockOrRecover, lock_reporting_poison, report_poison};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{
    AudioAccelerator, AudioDecoderWeights, DepthformerConfig, DetokenizerConfig, DetokenizerState,
    DetokenizerWeights, detok_embed_codes, istft_to_pcm, upsample,
};
use crate::model::weights::MmapWeight;
use crate::session::CeraError;
use crate::tensor::DType;

/// Metadata describing a detokenizer weight tensor in shared rpcmem.
pub use crate::backend::hexagon::types::HexagonWeightDesc as HexagonDetokWeightDesc;

/// The detokenizer emits one op over all tokens (see `dispatch::TokenTile`);
/// only the Whisper encoder tiles. Every `dispatch::*` call here passes this.
const DETOK_TILE: TokenTile = TokenTile::Whole;

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
    pub conv_w0_off: Option<usize>,
    pub conv_w1_off: Option<usize>,
    pub conv_w2_off: Option<usize>,
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

            let (conv_in_proj, conv_out_proj, conv_w0_off, conv_w1_off, conv_w2_off) = if is_conv {
                let in_proj = lw.conv_in_proj.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing conv_in_proj in detok conv layer".into())
                })?;
                let out_proj = lw.conv_out_proj.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing conv_out_proj in detok conv layer".into())
                })?;
                let _conv_w = lw.conv_weight.as_ref().ok_or_else(|| {
                    CeraError::Backend("missing conv_weight in detok conv layer".into())
                })?;
                let n_embd = in_proj.cols;
                (
                    Some(plan_linear(&mut cur_off, in_proj)?),
                    Some(plan_linear(&mut cur_off, out_proj)?),
                    Some(plan_vec_f32(&mut cur_off, n_embd)),
                    Some(plan_vec_f32(&mut cur_off, n_embd)),
                    Some(plan_vec_f32(&mut cur_off, n_embd)),
                )
            } else {
                (None, None, None, None, None)
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
                conv_w0_off,
                conv_w1_off,
                conv_w2_off,
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

/// Offsets for one depthformer layer in Depthformer weights buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDepthformerLayerOffsets {
    pub op_norm_off: usize,
    pub wqkv: HexagonDetokWeightDesc,
    pub q_norm_off: usize,
    pub k_norm_off: usize,
    pub wo: HexagonDetokWeightDesc,
    pub ffn_norm_off: usize,
    pub w1: HexagonDetokWeightDesc,
    pub w2: HexagonDetokWeightDesc,
    pub w3: HexagonDetokWeightDesc,
}

/// Weight buffer layout offsets across all depthformer components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonDepthformerWeightOffsets {
    pub depth_linear_slices: Vec<HexagonDetokWeightDesc>,
    pub depth_linear_biases: Vec<usize>,
    pub layers: Vec<HexagonDepthformerLayerOffsets>,
    pub cb_norms: Vec<usize>,
    pub cb_to_logits: Vec<HexagonDetokWeightDesc>,
    pub total_bytes: usize,
}

impl HexagonDepthformerWeightOffsets {
    /// Compute memory layout and byte offsets for all depthformer weights in shared rpcmem.
    pub fn plan(weights: &AudioDecoderWeights) -> Result<Self, CeraError> {
        let mut cur_off = 0;

        let plan_vec_f32 = |cur_off: &mut usize, len: usize| -> usize {
            let off = *cur_off;
            *cur_off += align128(len * 4);
            off
        };

        let plan_matrix = |cur_off: &mut usize,
                           rows: usize,
                           cols: usize,
                           dtype: DType|
         -> Result<HexagonDetokWeightDesc, CeraError> {
            let off = *cur_off;
            let (fmt, sz) = match dtype {
                DType::Q8_0 | DType::F32 => {
                    let sz = repacked_matrix_size_q8_0(cols, rows)?;
                    (HexagonWeightFormat::RepackedQ8_0, sz)
                }
                DType::Q4_0 => {
                    let sz = repacked_matrix_size_q4_0(cols, rows)?;
                    (HexagonWeightFormat::RepackedQ4_0, sz)
                }
                other => {
                    return Err(CeraError::Backend(format!(
                        "Hexagon depthformer weight requires Q8_0, Q4_0, or F32, got {other:?}"
                    )));
                }
            };
            *cur_off += align128(sz);
            Ok(HexagonDetokWeightDesc {
                offset: off,
                size_bytes: sz,
                format: fmt,
                rows,
                cols,
            })
        };

        let n_cb = weights.decoder_config.n_codebook;
        let n_embd_d = weights.depth_linear_w.rows / n_cb;
        let dl_cols = weights.depth_linear_w.cols;

        let mut depth_linear_slices = Vec::with_capacity(n_cb);
        let mut depth_linear_biases = Vec::with_capacity(n_cb);
        for _ in 0..n_cb {
            depth_linear_slices.push(plan_matrix(
                &mut cur_off,
                n_embd_d,
                dl_cols,
                weights.depth_linear_w.dtype,
            )?);
            depth_linear_biases.push(plan_vec_f32(&mut cur_off, n_embd_d));
        }

        let mut layers = Vec::with_capacity(weights.depthformer_layers.len());
        for lw in &weights.depthformer_layers {
            let op_norm_off = plan_vec_f32(&mut cur_off, lw.operator_norm.len());
            let wqkv = plan_matrix(&mut cur_off, lw.wqkv.rows, lw.wqkv.cols, lw.wqkv.dtype)?;
            let q_norm_off = plan_vec_f32(&mut cur_off, lw.q_norm.len());
            let k_norm_off = plan_vec_f32(&mut cur_off, lw.k_norm.len());
            let wo = plan_matrix(&mut cur_off, lw.wo.rows, lw.wo.cols, lw.wo.dtype)?;
            let ffn_norm_off = plan_vec_f32(&mut cur_off, lw.ffn_norm.len());
            let w1 = plan_matrix(&mut cur_off, lw.w1.rows, lw.w1.cols, lw.w1.dtype)?;
            let w2 = plan_matrix(&mut cur_off, lw.w2.rows, lw.w2.cols, lw.w2.dtype)?;
            let w3 = plan_matrix(&mut cur_off, lw.w3.rows, lw.w3.cols, lw.w3.dtype)?;
            layers.push(HexagonDepthformerLayerOffsets {
                op_norm_off,
                wqkv,
                q_norm_off,
                k_norm_off,
                wo,
                ffn_norm_off,
                w1,
                w2,
                w3,
            });
        }

        let mut cb_norms = Vec::with_capacity(n_cb);
        let mut cb_to_logits = Vec::with_capacity(n_cb);
        for cb in &weights.depth_embeddings {
            cb_norms.push(plan_vec_f32(&mut cur_off, cb.norm.len()));
            cb_to_logits.push(plan_matrix(
                &mut cur_off,
                cb.to_logits.rows,
                cb.to_logits.cols,
                cb.to_logits.dtype,
            )?);
        }

        Ok(Self {
            depth_linear_slices,
            depth_linear_biases,
            layers,
            cb_norms,
            cb_to_logits,
            total_bytes: cur_off,
        })
    }
}

/// Offsets for recurrent rolling conv states in `state_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDetokConvStateOffset {
    pub s0_off: usize,
    pub s1_off: usize,
}

/// Offsets for sliding-window attention KV cache in `state_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDetokAttnStateOffset {
    pub k_cache_off: usize,
    pub v_cache_off: usize,
}

/// State buffer layout offsets across all detokenizer layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonDetokStateOffsets {
    pub conv_states: Vec<Option<HexagonDetokConvStateOffset>>,
    pub attn_states: Vec<Option<HexagonDetokAttnStateOffset>>,
    pub total_bytes: usize,
}

impl HexagonDetokStateOffsets {
    /// Plan state memory allocation for Conv rolling states and attention KV caches.
    pub fn plan(cfg: &DetokenizerConfig) -> Self {
        let mut cur_off = 0;
        let n_embd = cfg.n_embd;
        let kv_dim = cfg.n_head_kv * cfg.n_embd_head;
        let swa = cfg.swa_window_size;

        let mut conv_states = Vec::with_capacity(cfg.n_layer);
        let mut attn_states = Vec::with_capacity(cfg.n_layer);

        for &is_conv in &cfg.layer_is_conv {
            if is_conv {
                let s0 = cur_off;
                cur_off += align128(n_embd * 4);
                let s1 = cur_off;
                cur_off += align128(n_embd * 4);
                conv_states.push(Some(HexagonDetokConvStateOffset {
                    s0_off: s0,
                    s1_off: s1,
                }));
                attn_states.push(None);
            } else {
                let k_off = cur_off;
                cur_off += align128(swa * kv_dim * 4);
                let v_off = cur_off;
                cur_off += align128(swa * kv_dim * 4);
                conv_states.push(None);
                attn_states.push(Some(HexagonDetokAttnStateOffset {
                    k_cache_off: k_off,
                    v_cache_off: v_off,
                }));
            }
        }

        Self {
            conv_states,
            attn_states,
            total_bytes: cur_off,
        }
    }
}

/// Layout offsets for runtime intermediate activations inside `scratch_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDetokScratchOffsets {
    pub tokens_off: usize,
    pub normed_off: usize,
    pub conv_bcx_off: usize,
    pub conv_bx_off: usize,
    pub conv_t0_off: usize,
    pub conv_t1_off: usize,
    pub conv_y_off: usize,
    pub conv_out_off: usize,
    pub q_off: usize,
    pub k_off: usize,
    pub v_off: usize,
    pub k_f16_off: usize,
    pub v_f16_off: usize,
    pub attn_out_off: usize,
    pub attn_proj_off: usize,
    pub pos_off: usize,
    pub ffn_gate_off: usize,
    pub ffn_up_off: usize,
    pub ffn_mid_off: usize,
    pub ffn_out_off: usize,
    pub spec_out_off: usize,
    pub total_bytes: usize,
}

impl HexagonDetokScratchOffsets {
    pub fn new(n_embd: usize, ffn_dim: usize, n_fft_bins: usize, max_seq_len: usize) -> Self {
        const MAX_AUDIO_TOKENS: usize = 16;
        let n_head = 8;
        let n_kv = 2;
        let hd = (n_embd / n_head).max(1);

        let tokens_bytes = align128(MAX_AUDIO_TOKENS * n_embd * 4);
        let conv_bcx_bytes = align128(MAX_AUDIO_TOKENS * 3 * n_embd * 4);
        let q_bytes = align128(MAX_AUDIO_TOKENS * (n_head * hd) * 4);
        let kv_f32_bytes = align128(MAX_AUDIO_TOKENS * (n_kv * hd) * 4);
        let kv_f16_bytes = align128(max_seq_len.max(64) * (n_kv * hd) * 2);
        let pos_bytes = align128(MAX_AUDIO_TOKENS * 4);
        let ffn_mid_bytes = align128(MAX_AUDIO_TOKENS * ffn_dim * 4);
        let spec_bytes = align128(MAX_AUDIO_TOKENS * (n_fft_bins * 2) * 4);

        let mut off = 0;
        let tokens_off = off;
        off += tokens_bytes;

        let normed_off = off;
        off += tokens_bytes;

        let conv_bcx_off = off;
        off += conv_bcx_bytes;

        let conv_bx_off = off;
        off += tokens_bytes;

        let conv_t0_off = off;
        off += tokens_bytes;

        let conv_t1_off = off;
        off += tokens_bytes;

        let conv_y_off = off;
        off += tokens_bytes;

        let conv_out_off = off;
        off += tokens_bytes;

        let q_off = off;
        off += q_bytes;

        let k_off = off;
        off += kv_f32_bytes;

        let v_off = off;
        off += kv_f32_bytes;

        let k_f16_off = off;
        off += kv_f16_bytes;

        let v_f16_off = off;
        off += kv_f16_bytes;

        let attn_out_off = off;
        off += q_bytes;

        let attn_proj_off = off;
        off += tokens_bytes;

        let pos_off = off;
        off += pos_bytes;

        let ffn_gate_off = off;
        off += ffn_mid_bytes;

        let ffn_up_off = off;
        off += ffn_mid_bytes;

        let ffn_mid_off = off;
        off += ffn_mid_bytes;

        let ffn_out_off = off;
        off += tokens_bytes;

        let spec_out_off = off;
        off += spec_bytes;

        Self {
            tokens_off,
            normed_off,
            conv_bcx_off,
            conv_bx_off,
            conv_t0_off,
            conv_t1_off,
            conv_y_off,
            conv_out_off,
            q_off,
            k_off,
            v_off,
            k_f16_off,
            v_f16_off,
            attn_out_off,
            attn_proj_off,
            pos_off,
            ffn_gate_off,
            ffn_up_off,
            ffn_mid_off,
            ffn_out_off,
            spec_out_off,
            total_bytes: off,
        }
    }
}

/// Layout offsets for runtime intermediate activations inside Depthformer `scratch_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDepthformerScratchOffsets {
    pub llm_emb_off: usize,
    pub hidden_off: usize,
    pub token_emb_off: usize,
    pub pos_off: usize,
    pub normed_off: usize,
    pub qkv_off: usize,
    pub attn_out_off: usize,
    pub attn_proj_off: usize,
    pub ffn_gate_off: usize,
    pub ffn_up_off: usize,
    pub ffn_mid_off: usize,
    pub ffn_out_off: usize,
    pub logits_off: usize,
    pub total_bytes: usize,
}

impl HexagonDepthformerScratchOffsets {
    pub fn new(
        n_embd_llm: usize,
        n_embd: usize,
        ffn_dim: usize,
        n_vocab: usize,
        n_head: usize,
        n_head_kv: usize,
        hd: usize,
    ) -> Self {
        let mut off = 0;

        let llm_emb_off = off;
        off += align128(n_embd_llm * 4);

        let hidden_off = off;
        off += align128(n_embd * 4);

        let token_emb_off = off;
        off += align128(n_embd * 4);

        let pos_off = off;
        off += align128(4);

        let normed_off = off;
        off += align128(n_embd * 4);

        let qkv_dim = (n_head + 2 * n_head_kv) * hd;
        let qkv_off = off;
        off += align128(qkv_dim * 4);

        let attn_out_off = off;
        off += align128(n_head * hd * 4);

        let attn_proj_off = off;
        off += align128(n_embd * 4);

        let ffn_gate_off = off;
        off += align128(ffn_dim * 4);

        let ffn_up_off = off;
        off += align128(ffn_dim * 4);

        let ffn_mid_off = off;
        off += align128(ffn_dim * 4);

        let ffn_out_off = off;
        off += align128(n_embd * 4);

        let logits_off = off;
        off += align128(n_vocab.next_multiple_of(32) * 4);

        Self {
            llm_emb_off,
            hidden_off,
            token_emb_off,
            pos_off,
            normed_off,
            qkv_off,
            attn_out_off,
            attn_proj_off,
            ffn_gate_off,
            ffn_up_off,
            ffn_mid_off,
            ffn_out_off,
            logits_off,
            total_bytes: off,
        }
    }
}

/// Offsets for one depthformer layer's KV cache in `state_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexagonDepthformerLayerStateOffsets {
    pub k_cache_off: usize,
    pub v_cache_off: usize,
}

/// State buffer layout offsets across all depthformer layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HexagonDepthformerStateOffsets {
    pub layers: Vec<HexagonDepthformerLayerStateOffsets>,
    pub total_bytes: usize,
}

impl HexagonDepthformerStateOffsets {
    pub fn plan(cfg: &DepthformerConfig) -> Self {
        let mut cur_off = 0;
        let kv_dim = cfg.n_head_kv * cfg.n_embd_head;
        let max_seq = cfg.max_seq_len.max(8);
        let cache_bytes = align128(max_seq * kv_dim * 2);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let k_cache_off = cur_off;
            cur_off += cache_bytes;
            let v_cache_off = cur_off;
            cur_off += cache_bytes;
            layers.push(HexagonDepthformerLayerStateOffsets {
                k_cache_off,
                v_cache_off,
            });
        }

        Self {
            layers,
            total_bytes: cur_off,
        }
    }
}

/// Hardware-accelerated Hexagon NPU Depthformer backend.
pub struct HexagonDepthformer {
    driver: Arc<FastRpcDriver>,
    weights_buf: RpcmemBuffer,
    weights_offsets: HexagonDepthformerWeightOffsets,
    scratch_buf: Mutex<RpcmemBuffer>,
    scratch_offsets: HexagonDepthformerScratchOffsets,
    state_buf: Mutex<RpcmemBuffer>,
    state_offsets: HexagonDepthformerStateOffsets,
    weights: Arc<AudioDecoderWeights>,
    n_past: std::sync::atomic::AtomicUsize,
    logits_scratch: Mutex<Vec<f32>>,
    indices_scratch: Mutex<Vec<usize>>,
    emb_scratch: Mutex<Vec<f32>>,
}

impl HexagonDepthformer {
    /// Tell the DSP to let go of this depthformer's buffers, ahead of the
    /// unmaps that dropping it performs (the decoder's `Drop` calls this with
    /// its device open; the depthformer has no device of its own).
    fn release_dsp_references(&self, session: &HexagonQueueSession) {
        session.release_dsp_references([
            &self.weights_buf,
            &*self.scratch_buf.lock_or_recover(),
            &*self.state_buf.lock_or_recover(),
        ]);
    }

    pub fn new(
        driver: Arc<FastRpcDriver>,
        weights: Arc<AudioDecoderWeights>,
    ) -> Result<Self, CeraError> {
        let n_vocab = weights.decoder_config.n_vocab;
        let n_embd = weights.depthformer_config.n_embd;
        let scratch_offsets = HexagonDepthformerScratchOffsets::new(
            weights.decoder_config.n_embd,
            n_embd,
            weights.depthformer_config.ffn_dim,
            n_vocab,
            weights.depthformer_config.n_head,
            weights.depthformer_config.n_head_kv,
            weights.depthformer_config.n_embd_head,
        );
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true).map_err(
                |e| {
                    CeraError::Backend(format!(
                        "failed to allocate depthformer scratch buffer: {e}"
                    ))
                },
            )?;

        let state_offsets = HexagonDepthformerStateOffsets::plan(&weights.depthformer_config);
        let state_buf = RpcmemBuffer::alloc(Arc::clone(&driver), state_offsets.total_bytes, true)
            .map_err(|e| {
            CeraError::Backend(format!("failed to allocate depthformer state buffer: {e}"))
        })?;
        unsafe {
            std::ptr::write_bytes(state_buf.as_mut_ptr(), 0, state_buf.size());
        }
        state_buf.flush_cpu_cache(0, state_buf.size());

        let offsets = HexagonDepthformerWeightOffsets::plan(&weights)?;
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), offsets.total_bytes, true)
            .map_err(|e| {
            CeraError::Backend(format!(
                "failed to allocate depthformer weights buffer: {e}"
            ))
        })?;
        Self::stage_weights(&mut weights_buf, &offsets, &weights)?;
        weights_buf.flush_cpu_cache(0, offsets.total_bytes);

        Ok(Self {
            driver,
            weights_buf,
            weights_offsets: offsets,
            scratch_buf: Mutex::new(scratch_buf),
            scratch_offsets,
            state_buf: Mutex::new(state_buf),
            state_offsets,
            weights,
            n_past: std::sync::atomic::AtomicUsize::new(0),
            logits_scratch: Mutex::new(Vec::with_capacity(n_vocab)),
            indices_scratch: Mutex::new(Vec::with_capacity(n_vocab)),
            emb_scratch: Mutex::new(Vec::with_capacity(n_embd)),
        })
    }

    pub fn reset(&self) {
        // The buffer is fully overwritten with zeros, so a poisoned lock is
        // recoverable; skipping it would reset the counters over stale state.
        {
            let sb_guard = self.state_buf.lock_or_recover();
            unsafe {
                std::ptr::write_bytes(sb_guard.as_mut_ptr(), 0, sb_guard.size());
            }
            sb_guard.flush_cpu_cache(0, sb_guard.size());
        }
        self.n_past.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Access the underlying FastRPC driver.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }

    pub fn stage_weights(
        buf: &mut RpcmemBuffer,
        offsets: &HexagonDepthformerWeightOffsets,
        weights: &AudioDecoderWeights,
    ) -> Result<(), CeraError> {
        let copy_vec_f32 = |dst: &mut [u8], off: usize, src: &[f32]| {
            let bytes = bytemuck::cast_slice::<f32, u8>(src);
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        };

        let copy_linear = |dst: &mut [u8],
                           desc: HexagonDetokWeightDesc,
                           w: &MmapWeight|
         -> Result<(), CeraError> {
            let dst_slice = &mut dst[desc.offset..desc.offset + desc.size_bytes];
            match desc.format {
                HexagonWeightFormat::RepackedQ8_0 => {
                    if w.dtype == DType::F32 {
                        let f32_vec;
                        let f32_slice = match w.try_as_f32() {
                            Some(slice) => slice,
                            None => {
                                f32_vec = w.to_dense_f32();
                                &f32_vec[..]
                            }
                        };
                        let q8_bytes = crate::backend::hexagon::quantize_f32_to_q8_0(
                            f32_slice, w.cols, w.rows,
                        )?;
                        repack_q8_0(&q8_bytes, w.cols, w.rows, dst_slice).map_err(|e| {
                            CeraError::Backend(format!("depthformer repack Q8_0 failed: {e}"))
                        })?;
                    } else {
                        repack_q8_0(w.data(), w.cols, w.rows, dst_slice).map_err(|e| {
                            CeraError::Backend(format!("depthformer repack Q8_0 failed: {e}"))
                        })?;
                    }
                }
                HexagonWeightFormat::RepackedQ4_0 => {
                    repack_q4_0(w.data(), w.cols, w.rows, dst_slice).map_err(|e| {
                        CeraError::Backend(format!("depthformer repack Q4_0 failed: {e}"))
                    })?;
                }
            }
            Ok(())
        };

        let n_cb = weights.decoder_config.n_codebook;
        let n_embd_d = weights.depth_linear_w.rows / n_cb;
        let dl_cols = weights.depth_linear_w.cols;
        let dl_w = &weights.depth_linear_w;

        let row_bytes = match dl_w.dtype {
            DType::Q4_0 => (dl_cols / 32) * 18,
            DType::Q8_0 => (dl_cols / 32) * 34,
            DType::F32 => dl_cols * 4,
            other => {
                return Err(CeraError::Backend(format!(
                    "unsupported depth_linear_w dtype {other:?}"
                )));
            }
        };
        let slice_byte_len = n_embd_d * row_bytes;

        let slice = buf.as_mut_slice();

        for j in 0..n_cb {
            let desc = offsets.depth_linear_slices[j];
            let dst_slice = &mut slice[desc.offset..desc.offset + desc.size_bytes];
            let src_start = j * slice_byte_len;
            let src_slice = &dl_w.data()[src_start..src_start + slice_byte_len];
            match desc.format {
                HexagonWeightFormat::RepackedQ4_0 => {
                    repack_q4_0(src_slice, dl_cols, n_embd_d, dst_slice).map_err(|e| {
                        CeraError::Backend(format!(
                            "depth_linear slice {j} repack Q4_0 failed: {e}"
                        ))
                    })?;
                }
                HexagonWeightFormat::RepackedQ8_0 => {
                    if dl_w.dtype == DType::F32 {
                        let f32_vec;
                        let f32_slice = match bytemuck::try_cast_slice::<u8, f32>(src_slice) {
                            Ok(slice) => slice,
                            Err(_) => {
                                f32_vec = src_slice
                                    .as_chunks::<4>()
                                    .0
                                    .iter()
                                    .map(|chunk| f32::from_ne_bytes(*chunk))
                                    .collect::<Vec<f32>>();
                                &f32_vec[..]
                            }
                        };
                        let q8_bytes = crate::backend::hexagon::quantize_f32_to_q8_0(
                            f32_slice, dl_cols, n_embd_d,
                        )?;
                        repack_q8_0(&q8_bytes, dl_cols, n_embd_d, dst_slice).map_err(|e| {
                            CeraError::Backend(format!(
                                "depth_linear slice {j} repack Q8_0 failed: {e}"
                            ))
                        })?;
                    } else {
                        repack_q8_0(src_slice, dl_cols, n_embd_d, dst_slice).map_err(|e| {
                            CeraError::Backend(format!(
                                "depth_linear slice {j} repack Q8_0 failed: {e}"
                            ))
                        })?;
                    }
                }
            }

            let bias_slice = &weights.depth_linear_b[j * n_embd_d..(j + 1) * n_embd_d];
            copy_vec_f32(slice, offsets.depth_linear_biases[j], bias_slice);
        }

        for (lw, lo) in weights.depthformer_layers.iter().zip(offsets.layers.iter()) {
            copy_vec_f32(slice, lo.op_norm_off, &lw.operator_norm);
            copy_linear(slice, lo.wqkv, &lw.wqkv)?;
            copy_vec_f32(slice, lo.q_norm_off, &lw.q_norm);
            copy_vec_f32(slice, lo.k_norm_off, &lw.k_norm);
            copy_linear(slice, lo.wo, &lw.wo)?;
            copy_vec_f32(slice, lo.ffn_norm_off, &lw.ffn_norm);
            copy_linear(slice, lo.w1, &lw.w1)?;
            copy_linear(slice, lo.w2, &lw.w2)?;
            copy_linear(slice, lo.w3, &lw.w3)?;
        }

        for (j, cb) in weights.depth_embeddings.iter().enumerate() {
            copy_vec_f32(slice, offsets.cb_norms[j], &cb.norm);
            copy_linear(slice, offsets.cb_to_logits[j], &cb.to_logits)?;
        }

        Ok(())
    }

    pub fn sample_frame(
        &self,
        device: &Mutex<HexagonDevice>,
        embedding: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<[i32; 8], CeraError> {
        let dec = &self.weights.decoder_config;
        let cfg = &self.weights.depthformer_config;
        let n_embd = cfg.n_embd;
        let hd = cfg.n_embd_head;
        let n_head = cfg.n_head;
        let n_kv = cfg.n_head_kv;
        let kv_dim = n_kv * hd;
        let q_bytes = n_head * hd * 4;
        let k_bytes = n_kv * hd * 4;
        let so = self.scratch_offsets;
        let n_vocab = dec.n_vocab;
        let logits_bytes = n_vocab * 4;

        self.reset();

        let mut codes = [0i32; 8];
        let mut prev_token: i32 = -1;

        let mut dev_guard = device.lock_or_recover();
        let mut scratch_guard = self.scratch_buf.lock_or_recover();
        let state_buf_guard = self.state_buf.lock_or_recover();
        // `reset()` above rewrote the state; a panic mid-dispatch in an
        // earlier frame can still have left a half-built batch queued, which
        // the next flush would run together with this frame's ops.
        dev_guard.queue_session_mut().drop_pending_batch();

        // 1. Stage LLM embedding into scratch buffer at llm_emb_off
        {
            let copy_len = embedding.len().min(dec.n_embd);
            let dst_slice = bytemuck::cast_slice_mut::<u8, f32>(
                &mut scratch_guard.as_mut_slice()[so.llm_emb_off..so.llm_emb_off + dec.n_embd * 4],
            );
            if copy_len > 0 {
                dst_slice[..copy_len].copy_from_slice(&embedding[..copy_len]);
            }
            if copy_len < dec.n_embd {
                dst_slice[copy_len..].fill(0.0);
            }
            scratch_guard.flush_cpu_cache(so.llm_emb_off, dec.n_embd * 4);
        }

        let mut emb_scratch = self.emb_scratch.lock_or_recover();
        let mut logits_scratch = self.logits_scratch.lock_or_recover();
        let mut indices_scratch = self.indices_scratch.lock_or_recover();

        for (j, code_slot) in codes.iter_mut().enumerate().take(dec.n_codebook) {
            let pos = self.n_past.load(std::sync::atomic::Ordering::Relaxed);

            // 2. Stage previous codebook embedding if j > 0
            if j > 0 && prev_token >= 0 {
                let prev_cb = &self.weights.depth_embeddings[j - 1];
                let tok = prev_token as usize;
                if tok < prev_cb.embedding.rows {
                    emb_scratch.resize(n_embd, 0.0);
                    prev_cb.embedding.dequantize_row(tok, &mut emb_scratch);
                    let dst_bytes = bytemuck::cast_slice::<f32, u8>(&emb_scratch[..n_embd]);
                    scratch_guard.as_mut_slice()
                        [so.token_emb_off..so.token_emb_off + dst_bytes.len()]
                        .copy_from_slice(dst_bytes);
                    scratch_guard.flush_cpu_cache(so.token_emb_off, dst_bytes.len());
                }
            }

            // 3. Stage position into pos_off for RoPE
            {
                let pos_bytes = (pos as i32).to_le_bytes();
                scratch_guard.as_mut_slice()[so.pos_off..so.pos_off + 4]
                    .copy_from_slice(&pos_bytes);
                scratch_guard.flush_cpu_cache(so.pos_off, 4);
            }

            let session = dev_guard.queue_session_mut();

            // 4. Depth linear projection: LLM embedding -> hidden_off
            let res = (|| -> Result<(), CeraError> {
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.llm_emb_off,
                    &self.weights_buf,
                    self.weights_offsets.depth_linear_slices[j],
                    Some(self.weights_offsets.depth_linear_biases[j]),
                    &scratch_guard,
                    so.hidden_off,
                    1,
                    DETOK_TILE,
                )?;

                // 5. Add previous codebook token embedding
                if j > 0 && prev_token >= 0 {
                    let prev_cb = &self.weights.depth_embeddings[j - 1];
                    let tok = prev_token as usize;
                    if tok < prev_cb.embedding.rows {
                        dispatch::add_residual(
                            session,
                            &scratch_guard,
                            so.hidden_off,
                            &scratch_guard,
                            so.token_emb_off,
                            TokenShape {
                                dim: n_embd,
                                n_tokens: 1,
                            },
                            DETOK_TILE,
                        )?;
                    }
                }

                // 6. Depthformer 6 transformer layers
                let scale = 1.0f32 / (hd as f32).sqrt();
                for (il, lw_offsets) in self.weights_offsets.layers.iter().enumerate() {
                    // Operator RMSNorm -> normed_off
                    HexagonAudioDecoder::dispatch_rms_norm_mul(
                        session,
                        &scratch_guard,
                        so.hidden_off,
                        &self.weights_buf,
                        lw_offsets.op_norm_off,
                        &scratch_guard,
                        so.normed_off,
                        cfg.rms_norm_eps,
                        n_embd,
                        1,
                    )?;

                    // QKV projection -> qkv_off
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.normed_off,
                        &self.weights_buf,
                        lw_offsets.wqkv,
                        None,
                        &scratch_guard,
                        so.qkv_off,
                        1,
                        DETOK_TILE,
                    )?;

                    // Q per-head RMSNorm
                    HexagonAudioDecoder::dispatch_rms_norm_mul(
                        session,
                        &scratch_guard,
                        so.qkv_off,
                        &self.weights_buf,
                        lw_offsets.q_norm_off,
                        &scratch_guard,
                        so.qkv_off,
                        cfg.rms_norm_eps,
                        hd,
                        n_head,
                    )?;

                    // K per-head RMSNorm
                    HexagonAudioDecoder::dispatch_rms_norm_mul(
                        session,
                        &scratch_guard,
                        so.qkv_off + q_bytes,
                        &self.weights_buf,
                        lw_offsets.k_norm_off,
                        &scratch_guard,
                        so.qkv_off + q_bytes,
                        cfg.rms_norm_eps,
                        hd,
                        n_kv,
                    )?;

                    // RoPE on Q (interleaved, mode = 0)
                    HexagonAudioDecoder::dispatch_rope_m(
                        session,
                        &scratch_guard,
                        so.qkv_off,
                        &scratch_guard,
                        so.pos_off,
                        hd,
                        n_head,
                        1,
                        cfg.max_seq_len,
                        cfg.rope_freq_base,
                        0,
                    )?;

                    // RoPE on K (interleaved, mode = 0)
                    HexagonAudioDecoder::dispatch_rope_m(
                        session,
                        &scratch_guard,
                        so.qkv_off + q_bytes,
                        &scratch_guard,
                        so.pos_off,
                        hd,
                        n_kv,
                        1,
                        cfg.max_seq_len,
                        cfg.rope_freq_base,
                        0,
                    )?;

                    // Write K and V into KV cache as F16
                    let k_dst = self.state_offsets.layers[il].k_cache_off + pos * (kv_dim * 2);
                    let v_dst = self.state_offsets.layers[il].v_cache_off + pos * (kv_dim * 2);
                    dispatch::cpy_f32_to_f16(
                        session,
                        &scratch_guard,
                        so.qkv_off + q_bytes,
                        &state_buf_guard,
                        k_dst,
                        TokenShape {
                            dim: kv_dim,
                            n_tokens: 1,
                        },
                    )?;
                    dispatch::cpy_f32_to_f16(
                        session,
                        &scratch_guard,
                        so.qkv_off + q_bytes + k_bytes,
                        &state_buf_guard,
                        v_dst,
                        TokenShape {
                            dim: kv_dim,
                            n_tokens: 1,
                        },
                    )?;

                    // Multi-head self-attention
                    let seq_len = pos + 1;
                    HexagonAudioDecoder::dispatch_self_attention(
                        session,
                        &scratch_guard,
                        so.qkv_off,
                        &state_buf_guard,
                        self.state_offsets.layers[il].k_cache_off,
                        &state_buf_guard,
                        self.state_offsets.layers[il].v_cache_off,
                        &scratch_guard,
                        so.attn_out_off,
                        hd,
                        n_head,
                        n_kv,
                        1,
                        seq_len,
                        scale,
                    )?;

                    // Out projection -> attn_proj_off
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.attn_out_off,
                        &self.weights_buf,
                        lw_offsets.wo,
                        None,
                        &scratch_guard,
                        so.attn_proj_off,
                        1,
                        DETOK_TILE,
                    )?;

                    // Residual add: hidden += attn_proj
                    dispatch::add_residual(
                        session,
                        &scratch_guard,
                        so.hidden_off,
                        &scratch_guard,
                        so.attn_proj_off,
                        TokenShape {
                            dim: n_embd,
                            n_tokens: 1,
                        },
                        DETOK_TILE,
                    )?;

                    // FFN RMSNorm -> normed_off
                    HexagonAudioDecoder::dispatch_rms_norm_mul(
                        session,
                        &scratch_guard,
                        so.hidden_off,
                        &self.weights_buf,
                        lw_offsets.ffn_norm_off,
                        &scratch_guard,
                        so.normed_off,
                        cfg.rms_norm_eps,
                        n_embd,
                        1,
                    )?;

                    // Gate + Up GEMVs
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.normed_off,
                        &self.weights_buf,
                        lw_offsets.w1,
                        None,
                        &scratch_guard,
                        so.ffn_gate_off,
                        1,
                        DETOK_TILE,
                    )?;
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.normed_off,
                        &self.weights_buf,
                        lw_offsets.w3,
                        None,
                        &scratch_guard,
                        so.ffn_up_off,
                        1,
                        DETOK_TILE,
                    )?;

                    // SwiGLU: silu(gate) * up -> ffn_mid_off
                    HexagonAudioDecoder::dispatch_swiglu(
                        session,
                        &scratch_guard,
                        so.ffn_gate_off,
                        &scratch_guard,
                        so.ffn_up_off,
                        &scratch_guard,
                        so.ffn_mid_off,
                        cfg.ffn_dim,
                        1,
                    )?;

                    // Down projection -> ffn_out_off
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.ffn_mid_off,
                        &self.weights_buf,
                        lw_offsets.w2,
                        None,
                        &scratch_guard,
                        so.ffn_out_off,
                        1,
                        DETOK_TILE,
                    )?;

                    // Residual add: hidden += ffn_out
                    dispatch::add_residual(
                        session,
                        &scratch_guard,
                        so.hidden_off,
                        &scratch_guard,
                        so.ffn_out_off,
                        TokenShape {
                            dim: n_embd,
                            n_tokens: 1,
                        },
                        DETOK_TILE,
                    )?;
                }

                // 7. Codebook norm + to_logits
                HexagonAudioDecoder::dispatch_rms_norm_mul(
                    session,
                    &scratch_guard,
                    so.hidden_off,
                    &self.weights_buf,
                    self.weights_offsets.cb_norms[j],
                    &scratch_guard,
                    so.normed_off,
                    dec.rms_norm_eps,
                    n_embd,
                    1,
                )?;

                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.normed_off,
                    &self.weights_buf,
                    self.weights_offsets.cb_to_logits[j],
                    None,
                    &scratch_guard,
                    so.logits_off,
                    1,
                    DETOK_TILE,
                )?;

                session.flush()?;
                Ok(())
            })();

            if let Err(e) = res {
                crate::backend::hexagon::hexagon_warn!(
                    "HexagonDepthformer NPU execution failed at codebook {j}: {e}"
                );
                session.drop_pending_batch();
                return Err(e);
            }

            // Invalidate cache for logits
            scratch_guard.invalidate_cpu_cache(so.logits_off, logits_bytes);
            let logits_slice = bytemuck::cast_slice::<u8, f32>(
                &scratch_guard.as_slice()[so.logits_off..so.logits_off + logits_bytes],
            );

            // Sample token
            let sampled = if logits_slice.is_empty() {
                0
            } else if !temperature.is_finite() || temperature <= 0.0 || top_k <= 1 {
                crate::sampler::argmax(logits_slice) as i32
            } else {
                logits_scratch.clear();
                logits_scratch.extend_from_slice(logits_slice);
                let inv_temp = 1.0 / temperature;
                for l in logits_scratch.iter_mut() {
                    *l *= inv_temp;
                }
                crate::backend::cpu::softmax_inplace(&mut logits_scratch);
                let n_logits = logits_scratch.len();
                let k = top_k.min(n_logits).max(1);
                indices_scratch.clear();
                indices_scratch.extend(0..n_logits);
                if k < n_logits {
                    indices_scratch.select_nth_unstable_by(k - 1, |&a, &b| {
                        logits_scratch[b].total_cmp(&logits_scratch[a])
                    });
                }
                let top_indices = &indices_scratch[..k];
                let sum: f32 = top_indices.iter().map(|&i| logits_scratch[i]).sum();
                let mut r = rand::random::<f32>() * sum;
                let mut picked = top_indices[0];
                for &i in top_indices {
                    r -= logits_scratch[i];
                    if r <= 0.0 {
                        picked = i;
                        break;
                    }
                }
                picked as i32
            };

            *code_slot = sampled;
            prev_token = sampled;
            self.n_past
                .store(pos + 1, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(codes)
    }
}

/// Maximum audio tokens supported in scratch buffer staging.
pub const MAX_AUDIO_TOKENS: usize = 16;

/// Hardware-accelerated Hexagon NPU audio decoder backend.
pub struct HexagonAudioDecoder {
    driver: Arc<FastRpcDriver>,
    device: Arc<Mutex<HexagonDevice>>,
    detok_weights: Arc<DetokenizerWeights>,
    weights_buf: RpcmemBuffer,
    weights_offsets: HexagonDetokWeightOffsets,
    scratch_buf: Mutex<RpcmemBuffer>,
    scratch_offsets: HexagonDetokScratchOffsets,
    state_buf: Mutex<RpcmemBuffer>,
    state_offsets: HexagonDetokStateOffsets,
    state: Mutex<DetokenizerState>,
    depthformer: Option<HexagonDepthformer>,
    last_error: Mutex<Option<CeraError>>,
    session_active: std::sync::atomic::AtomicBool,
}

impl Drop for HexagonAudioDecoder {
    /// The DSP holds a reference to every buffer a batch read; letting go of
    /// them while the device is open keeps the host unmaps that follow from
    /// failing and leaking their address space.
    fn drop(&mut self) {
        let mut device = self.device.lock_or_recover();
        let session = device.queue_session_mut();
        session.release_dsp_references([
            &self.weights_buf,
            &*self.scratch_buf.lock_or_recover(),
            &*self.state_buf.lock_or_recover(),
        ]);
        if let Some(depthformer) = &self.depthformer {
            depthformer.release_dsp_references(session);
        }
    }
}

impl HexagonAudioDecoder {
    /// Create a new Hexagon audio decoder with shared device and weights.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        detok_weights: Arc<DetokenizerWeights>,
        audio_dec_weights: Option<Arc<AudioDecoderWeights>>,
    ) -> Result<Self, CeraError> {
        let cfg = &detok_weights.config;
        let n_fft_bins = cfg.n_fft / 2 + 1;
        let scratch_offsets = HexagonDetokScratchOffsets::new(
            cfg.n_embd,
            cfg.ffn_dim,
            n_fft_bins,
            cfg.swa_window_size,
        );
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true).map_err(
                |e| CeraError::Backend(format!("failed to allocate detok scratch buffer: {e}")),
            )?;

        let state_offsets = HexagonDetokStateOffsets::plan(cfg);
        let state_buf = RpcmemBuffer::alloc(Arc::clone(&driver), state_offsets.total_bytes, true)
            .map_err(|e| {
            CeraError::Backend(format!("failed to allocate detok state buffer: {e}"))
        })?;
        unsafe {
            std::ptr::write_bytes(state_buf.as_mut_ptr(), 0, state_buf.size());
        }
        state_buf.flush_cpu_cache(0, state_buf.size());

        let offsets = HexagonDetokWeightOffsets::plan(&detok_weights)?;
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), offsets.total_bytes, true)
            .map_err(|e| {
            CeraError::Backend(format!("failed to allocate detok weights buffer: {e}"))
        })?;
        Self::stage_weights(&mut weights_buf, &offsets, &detok_weights)?;
        weights_buf.flush_cpu_cache(0, offsets.total_bytes);

        let depthformer = if let Some(ad_weights) = audio_dec_weights {
            match HexagonDepthformer::new(Arc::clone(&driver), ad_weights) {
                Ok(df) => {
                    tracing::info!("audio decoder: Hexagon NPU Depthformer initialized");
                    Some(df)
                }
                Err(e) => {
                    crate::backend::hexagon::hexagon_warn!(
                        "audio decoder: Hexagon NPU Depthformer unavailable: {e:#}"
                    );
                    None
                }
            }
        } else {
            None
        };

        let state = DetokenizerState::new(&detok_weights.config);
        Ok(Self {
            driver,
            device,
            detok_weights,
            weights_buf,
            weights_offsets: offsets,
            scratch_buf: Mutex::new(scratch_buf),
            scratch_offsets,
            state_buf: Mutex::new(state_buf),
            state_offsets,
            state: Mutex::new(state),
            depthformer,
            last_error: Mutex::new(None),
            session_active: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Record the first fault into the sticky error slot.
    fn record_error(&self, err: CeraError) {
        crate::model::record_first_fault(&self.last_error, err);
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
                if let (Some(w0_off), Some(cw)) = (lo.conv_w0_off, lw.conv_weight.as_ref()) {
                    let hs = lw.operator_norm.len();
                    if cw.len() == 3 * hs {
                        let mut w0 = vec![0.0f32; hs];
                        let mut w1 = vec![0.0f32; hs];
                        let mut w2 = vec![0.0f32; hs];
                        for ch in 0..hs {
                            w0[ch] = cw[ch * 3];
                            w1[ch] = cw[ch * 3 + 1];
                            w2[ch] = cw[ch * 3 + 2];
                        }
                        copy_vec_f32(slice, w0_off, &w0);
                        if let Some(w1_off) = lo.conv_w1_off {
                            copy_vec_f32(slice, w1_off, &w1);
                        }
                        if let Some(w2_off) = lo.conv_w2_off {
                            copy_vec_f32(slice, w2_off, &w2);
                        }
                    }
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

    /// Dispatch RMS norm + mul: `dst = rmsnorm(src) * weight`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_rms_norm_mul(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        weight: &RpcmemBuffer,
        weight_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        eps: f32,
        dim: usize,
        n_tokens: usize,
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

        let w_bytes = dim * 4;
        let w_ne = [dim as u32, 1, 1, 1];
        let w_nb = [4, w_bytes as u32, w_bytes as u32, w_bytes as u32];
        let weight_ti = session.add_tensor(
            weight,
            weight_offset,
            w_bytes,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            w_ne,
            w_nb,
        )?;

        let params = build_rms_norm_params(eps);
        let kparams = build_unary_kernel_params(
            dim,
            n_tokens,
            dim,
            8 * 1024 * 1024,
            session.dsp_threads(),
            true,
        );

        session
            .enqueue_op(
                HtpOpCode::RmsNormMul as u32,
                &[src_ti, weight_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_rms_norm_mul: {e}")))?;
        session.end_group()
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
            .map_err(|e| CeraError::Backend(format!("dispatch_swiglu: {e}")))?;
        session.end_group()
    }

    /// Dispatch elementwise multiplication on DSP: `dst = src0 * src1`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_mul(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let bytes = dim * 4;
        let ne = [dim as u32, 1, 1, 1];
        let nb = [4, bytes as u32, bytes as u32, bytes as u32];

        let src0_ti = session.add_tensor(
            src0,
            src0_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let src1_ti = session.add_tensor(
            src1,
            src1_offset,
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
        let kparams = build_binary_kernel_params(
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
                &[src0_ti, src1_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_mul: {e}")))?;
        session.end_group()
    }

    /// Dispatch elementwise addition on DSP: `dst = src0 + src1`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_add(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let bytes = dim * 4;
        let ne = [dim as u32, 1, 1, 1];
        let nb = [4, bytes as u32, bytes as u32, bytes as u32];

        let src0_ti = session.add_tensor(
            src0,
            src0_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let src1_ti = session.add_tensor(
            src1,
            src1_offset,
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
        let kparams = build_binary_kernel_params(
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
                HtpOpCode::Add as u32,
                &[src0_ti, src1_ti],
                &[dst_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_add: {e}")))?;
        session.end_group()
    }

    /// Dispatch flat buffer copy on DSP: `dst = src`.
    fn dispatch_cpy(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        bytes: usize,
    ) -> Result<(), CeraError> {
        let ne = [(bytes / 4).max(1) as u32, 1, 1, 1];
        let nb = [4, bytes as u32, bytes as u32, bytes as u32];

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

        let params = [0i32; 16];
        let kparams = [0i32; 32];

        session
            .enqueue_op(HtpOpCode::Cpy as u32, &[src_ti], &[dst_ti], params, kparams)
            .map_err(|e| CeraError::Backend(format!("detok_cpy: {e}")))?;
        session.end_group()
    }

    /// Dispatch multi-head RoPE NeoX rotation on DSP.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_rope_m(
        session: &mut HexagonQueueSession,
        act: &RpcmemBuffer,
        act_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_tokens: usize,
        max_seq_len: usize,
        rope_theta: f32,
        mode: u32,
    ) -> Result<(), CeraError> {
        let act_bytes = head_dim * n_heads * n_tokens * 4;
        let act_ti = session.add_tensor(
            act,
            act_offset,
            act_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, n_tokens as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * n_heads * 4) as u32,
                act_bytes as u32,
            ],
        )?;
        let pos_bytes = n_tokens * 4;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            pos_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_tokens as u32, 1, 1, 1],
            [
                4,
                (n_tokens * 4) as u32,
                (n_tokens * 4) as u32,
                (n_tokens * 4) as u32,
            ],
        )?;
        let params = build_rope_params(head_dim, mode, max_seq_len as u32, rope_theta, 1.0);
        let nrows = n_heads * n_tokens;
        let n_threads = session.dsp_threads().min(nrows as u32).max(1);
        let kparams = build_rope_kernel_params(head_dim, nrows, n_heads, n_tokens, n_threads);
        session
            .enqueue_op(
                HtpOpCode::Rope as u32,
                &[act_ti, pos_ti],
                &[act_ti],
                params,
                kparams,
            )
            .map_err(|e| CeraError::Backend(format!("detok_rope_m: {e}")))?;
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
        n_kv_heads: usize,
        n_tokens: usize,
        seq_len: usize,
        scale: f32,
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let q_bytes = q_dim * n_tokens * 4;
        let kv_dim = head_dim * n_kv_heads;
        let k_bytes = kv_dim * seq_len * 2;
        let v_bytes = kv_dim * seq_len * 2;

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
            k_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                2,
                (kv_dim * 2) as u32,
                (head_dim * 2) as u32,
                k_bytes as u32,
            ],
        )?;

        let v_ti = session.add_tensor(
            v,
            v_offset,
            v_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                2,
                (kv_dim * 2) as u32,
                (head_dim * 2) as u32,
                v_bytes as u32,
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
            n_kv_heads,
            n_tokens,
            seq_len,
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
            .map_err(|e| CeraError::Backend(format!("detok_self_attention: {e}")))?;
        session.end_group()
    }

    /// Dispatch native audio detokenizer forward pass entirely on Qualcomm Hexagon NPU.
    fn detokenize_to_spectrum_npu(
        &self,
        tokens: &[f32],
        n_tokens: usize,
    ) -> Result<Vec<f32>, CeraError> {
        let hs = self.detok_weights.config.n_embd;
        let eps = self.detok_weights.config.rms_norm_eps;
        let ffn_dim = self.detok_weights.config.ffn_dim;
        let n_head = self.detok_weights.config.n_head;
        let n_kv = self.detok_weights.config.n_head_kv;
        let hd = self.detok_weights.config.n_embd_head;
        let swa_window_size = self.detok_weights.config.swa_window_size;
        let rope_freq_base = self.detok_weights.config.rope_freq_base;
        let n_fft_bins = self.detok_weights.config.n_fft / 2 + 1;
        let spec_per_frame = n_fft_bins * 2;
        let so = self.scratch_offsets;

        if n_tokens == 0 || n_tokens > 16 {
            return Err(CeraError::Backend(format!(
                "detokenize_to_spectrum_npu requires 1 <= n_tokens <= 16, got {n_tokens}"
            )));
        }
        if tokens.len() != n_tokens * hs {
            return Err(CeraError::Backend(format!(
                "detokenize_to_spectrum_npu token slice length mismatch: expected {}, got {}",
                n_tokens * hs,
                tokens.len()
            )));
        }

        let mut dev_guard = self.device.lock_or_recover();
        let mut scratch_guard = self.scratch_buf.lock_or_recover();
        // Unlike the scratch buffers, the recurrent state persists across
        // calls and `n_past` only advances after the dispatches, so after a
        // poisoning panic the pair may be torn. Recover the locks, then start
        // the stream over rather than continue over half-advanced state.
        let (state_buf_guard, sb_poisoned) = lock_reporting_poison(&self.state_buf);
        let (mut state_guard, st_poisoned) = lock_reporting_poison(&self.state);
        if sb_poisoned || st_poisoned {
            report_poison("audio detokenizer state", "resetting the stream");
            unsafe {
                std::ptr::write_bytes(state_buf_guard.as_mut_ptr(), 0, state_buf_guard.size());
            }
            state_buf_guard.flush_cpu_cache(0, state_buf_guard.size());
            state_guard.reset();
        }

        // 1. Stage tokens into scratch buffer
        let tokens_bytes = tokens.len() * 4;
        scratch_guard.as_mut_slice()[so.tokens_off..so.tokens_off + tokens_bytes]
            .copy_from_slice(bytemuck::cast_slice(tokens));

        // 2. Stage token positions into pos buffer for RoPE
        let n_past = state_guard.n_past;
        let mut pos_vals = [0i32; 16];
        for (t, slot) in pos_vals.iter_mut().take(n_tokens).enumerate() {
            *slot = (n_past + t) as i32;
        }
        let pos_bytes = n_tokens * 4;
        scratch_guard.as_mut_slice()[so.pos_off..so.pos_off + pos_bytes]
            .copy_from_slice(bytemuck::cast_slice(&pos_vals[..n_tokens]));

        scratch_guard.flush_cpu_cache(so.tokens_off, tokens_bytes);
        scratch_guard.flush_cpu_cache(so.pos_off, pos_bytes);

        // 3. Obtain queue session and reset pending batch
        let session = dev_guard.queue_session_mut();
        session.drop_pending_batch();

        // 4. Dispatch all detokenizer layers
        for (il, lo) in self.weights_offsets.layers.iter().enumerate() {
            if lo.is_conv {
                let conv_state = self.state_offsets.conv_states[il].as_ref().ok_or_else(|| {
                    CeraError::Backend(format!("missing conv state offset for layer {il}"))
                })?;
                let w0_off = lo.conv_w0_off.ok_or_else(|| {
                    CeraError::Backend(format!("missing conv_w0_off for layer {il}"))
                })?;
                let w1_off = lo.conv_w1_off.ok_or_else(|| {
                    CeraError::Backend(format!("missing conv_w1_off for layer {il}"))
                })?;
                let w2_off = lo.conv_w2_off.ok_or_else(|| {
                    CeraError::Backend(format!("missing conv_w2_off for layer {il}"))
                })?;
                let in_proj = lo.conv_in_proj.ok_or_else(|| {
                    CeraError::Backend(format!("missing conv_in_proj for layer {il}"))
                })?;
                let out_proj = lo.conv_out_proj.ok_or_else(|| {
                    CeraError::Backend(format!("missing conv_out_proj for layer {il}"))
                })?;

                for t in 0..n_tokens {
                    let tok_off = so.tokens_off + t * hs * 4;

                    // RMSNorm on current token: cur -> normed
                    Self::dispatch_rms_norm_mul(
                        session,
                        &scratch_guard,
                        tok_off,
                        &self.weights_buf,
                        lo.op_norm_off,
                        &scratch_guard,
                        so.normed_off,
                        eps,
                        hs,
                        1,
                    )?;

                    // in_proj: normed -> conv_bcx [3 * hs]
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.normed_off,
                        &self.weights_buf,
                        in_proj,
                        None,
                        &scratch_guard,
                        so.conv_bcx_off,
                        1,
                        DETOK_TILE,
                    )?;

                    // bx = b * x (b at offset 0, x at offset 2 * hs * 4)
                    Self::dispatch_mul(
                        session,
                        &scratch_guard,
                        so.conv_bcx_off,
                        &scratch_guard,
                        so.conv_bcx_off + 2 * hs * 4,
                        &scratch_guard,
                        so.conv_bx_off,
                        hs,
                    )?;

                    // Rolling conv: conv_out = s0 * w0 + s1 * w1 + bx * w2
                    Self::dispatch_mul(
                        session,
                        &state_buf_guard,
                        conv_state.s0_off,
                        &self.weights_buf,
                        w0_off,
                        &scratch_guard,
                        so.conv_t0_off,
                        hs,
                    )?;
                    Self::dispatch_mul(
                        session,
                        &state_buf_guard,
                        conv_state.s1_off,
                        &self.weights_buf,
                        w1_off,
                        &scratch_guard,
                        so.conv_t1_off,
                        hs,
                    )?;
                    Self::dispatch_mul(
                        session,
                        &scratch_guard,
                        so.conv_bx_off,
                        &self.weights_buf,
                        w2_off,
                        &scratch_guard,
                        so.conv_out_off,
                        hs,
                    )?;
                    Self::dispatch_add(
                        session,
                        &scratch_guard,
                        so.conv_t0_off,
                        &scratch_guard,
                        so.conv_t1_off,
                        &scratch_guard,
                        so.conv_t0_off,
                        hs,
                    )?;
                    Self::dispatch_add(
                        session,
                        &scratch_guard,
                        so.conv_t0_off,
                        &scratch_guard,
                        so.conv_out_off,
                        &scratch_guard,
                        so.conv_out_off,
                        hs,
                    )?;

                    // Update rolling states: s0 = s1; s1 = bx
                    Self::dispatch_cpy(
                        session,
                        &state_buf_guard,
                        conv_state.s1_off,
                        &state_buf_guard,
                        conv_state.s0_off,
                        hs * 4,
                    )?;
                    Self::dispatch_cpy(
                        session,
                        &scratch_guard,
                        so.conv_bx_off,
                        &state_buf_guard,
                        conv_state.s1_off,
                        hs * 4,
                    )?;

                    // Gating: y = c * conv_out (c at offset hs * 4)
                    Self::dispatch_mul(
                        session,
                        &scratch_guard,
                        so.conv_bcx_off + hs * 4,
                        &scratch_guard,
                        so.conv_out_off,
                        &scratch_guard,
                        so.conv_y_off,
                        hs,
                    )?;

                    // out_proj: gated y -> conv_out
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.conv_y_off,
                        &self.weights_buf,
                        out_proj,
                        None,
                        &scratch_guard,
                        so.conv_out_off,
                        1,
                        DETOK_TILE,
                    )?;

                    // Residual add: tok[t] += conv_out
                    Self::dispatch_add(
                        session,
                        &scratch_guard,
                        tok_off,
                        &scratch_guard,
                        so.conv_out_off,
                        &scratch_guard,
                        tok_off,
                        hs,
                    )?;

                    // FFN: rmsnorm -> gate, up -> swiglu -> down -> residual
                    Self::dispatch_rms_norm_mul(
                        session,
                        &scratch_guard,
                        tok_off,
                        &self.weights_buf,
                        lo.ffn_norm_off,
                        &scratch_guard,
                        so.normed_off,
                        eps,
                        hs,
                        1,
                    )?;
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.normed_off,
                        &self.weights_buf,
                        lo.ffn_w1,
                        None,
                        &scratch_guard,
                        so.ffn_gate_off,
                        1,
                        DETOK_TILE,
                    )?;
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.normed_off,
                        &self.weights_buf,
                        lo.ffn_w3,
                        None,
                        &scratch_guard,
                        so.ffn_up_off,
                        1,
                        DETOK_TILE,
                    )?;
                    Self::dispatch_swiglu(
                        session,
                        &scratch_guard,
                        so.ffn_gate_off,
                        &scratch_guard,
                        so.ffn_up_off,
                        &scratch_guard,
                        so.ffn_mid_off,
                        ffn_dim,
                        1,
                    )?;
                    dispatch::linear_m(
                        session,
                        &scratch_guard,
                        so.ffn_mid_off,
                        &self.weights_buf,
                        lo.ffn_w2,
                        None,
                        &scratch_guard,
                        so.ffn_out_off,
                        1,
                        DETOK_TILE,
                    )?;
                    Self::dispatch_add(
                        session,
                        &scratch_guard,
                        tok_off,
                        &scratch_guard,
                        so.ffn_out_off,
                        &scratch_guard,
                        tok_off,
                        hs,
                    )?;
                }
            } else {
                let attn_state = self.state_offsets.attn_states[il].as_ref().ok_or_else(|| {
                    CeraError::Backend(format!("missing attn state offset for layer {il}"))
                })?;
                let wq = lo
                    .wq
                    .ok_or_else(|| CeraError::Backend(format!("missing wq for layer {il}")))?;
                let wk = lo
                    .wk
                    .ok_or_else(|| CeraError::Backend(format!("missing wk for layer {il}")))?;
                let wv = lo
                    .wv
                    .ok_or_else(|| CeraError::Backend(format!("missing wv for layer {il}")))?;
                let wo = lo
                    .wo
                    .ok_or_else(|| CeraError::Backend(format!("missing wo for layer {il}")))?;
                let q_norm_off = lo
                    .q_norm_off
                    .ok_or_else(|| CeraError::Backend(format!("missing q_norm for layer {il}")))?;
                let k_norm_off = lo
                    .k_norm_off
                    .ok_or_else(|| CeraError::Backend(format!("missing k_norm for layer {il}")))?;

                // 1. RMSNorm on all tokens
                Self::dispatch_rms_norm_mul(
                    session,
                    &scratch_guard,
                    so.tokens_off,
                    &self.weights_buf,
                    lo.op_norm_off,
                    &scratch_guard,
                    so.normed_off,
                    eps,
                    hs,
                    n_tokens,
                )?;

                // 2. Q, K, V projections
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.normed_off,
                    &self.weights_buf,
                    wq,
                    None,
                    &scratch_guard,
                    so.q_off,
                    n_tokens,
                    DETOK_TILE,
                )?;
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.normed_off,
                    &self.weights_buf,
                    wk,
                    None,
                    &scratch_guard,
                    so.k_off,
                    n_tokens,
                    DETOK_TILE,
                )?;
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.normed_off,
                    &self.weights_buf,
                    wv,
                    None,
                    &scratch_guard,
                    so.v_off,
                    n_tokens,
                    DETOK_TILE,
                )?;

                // 3. Per-head RMSNorm on Q and K
                Self::dispatch_rms_norm_mul(
                    session,
                    &scratch_guard,
                    so.q_off,
                    &self.weights_buf,
                    q_norm_off,
                    &scratch_guard,
                    so.q_off,
                    eps,
                    hd,
                    n_head * n_tokens,
                )?;
                Self::dispatch_rms_norm_mul(
                    session,
                    &scratch_guard,
                    so.k_off,
                    &self.weights_buf,
                    k_norm_off,
                    &scratch_guard,
                    so.k_off,
                    eps,
                    hd,
                    n_kv * n_tokens,
                )?;

                // 4. RoPE on Q and K
                Self::dispatch_rope_m(
                    session,
                    &scratch_guard,
                    so.q_off,
                    &scratch_guard,
                    so.pos_off,
                    hd,
                    n_head,
                    n_tokens,
                    2048,
                    rope_freq_base,
                    2,
                )?;
                Self::dispatch_rope_m(
                    session,
                    &scratch_guard,
                    so.k_off,
                    &scratch_guard,
                    so.pos_off,
                    hd,
                    n_kv,
                    n_tokens,
                    2048,
                    rope_freq_base,
                    2,
                )?;

                // 5. Write K and V into ring buffer
                let kv_bytes = (n_kv * hd) * 4;
                for t in 0..n_tokens {
                    let pos = n_past + t;
                    let cache_pos = pos % swa_window_size;
                    let k_src = so.k_off + t * kv_bytes;
                    let v_src = so.v_off + t * kv_bytes;
                    let k_dst = attn_state.k_cache_off + cache_pos * kv_bytes;
                    let v_dst = attn_state.v_cache_off + cache_pos * kv_bytes;
                    Self::dispatch_cpy(
                        session,
                        &scratch_guard,
                        k_src,
                        &state_buf_guard,
                        k_dst,
                        kv_bytes,
                    )?;
                    Self::dispatch_cpy(
                        session,
                        &scratch_guard,
                        v_src,
                        &state_buf_guard,
                        v_dst,
                        kv_bytes,
                    )?;
                }

                // 6. Sliding window length and F16 conversion
                let seq_len = (n_past + n_tokens).min(swa_window_size);
                dispatch::cpy_f32_to_f16(
                    session,
                    &state_buf_guard,
                    attn_state.k_cache_off,
                    &scratch_guard,
                    so.k_f16_off,
                    TokenShape {
                        dim: n_kv * hd,
                        n_tokens: seq_len,
                    },
                )?;
                dispatch::cpy_f32_to_f16(
                    session,
                    &state_buf_guard,
                    attn_state.v_cache_off,
                    &scratch_guard,
                    so.v_f16_off,
                    TokenShape {
                        dim: n_kv * hd,
                        n_tokens: seq_len,
                    },
                )?;

                // 7. FlashAttnExt
                let scale = 1.0f32 / (hd as f32).sqrt();
                Self::dispatch_self_attention(
                    session,
                    &scratch_guard,
                    so.q_off,
                    &scratch_guard,
                    so.k_f16_off,
                    &scratch_guard,
                    so.v_f16_off,
                    &scratch_guard,
                    so.attn_out_off,
                    hd,
                    n_head,
                    n_kv,
                    n_tokens,
                    seq_len,
                    scale,
                )?;

                // 8. Out projection
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.attn_out_off,
                    &self.weights_buf,
                    wo,
                    None,
                    &scratch_guard,
                    so.attn_proj_off,
                    n_tokens,
                    DETOK_TILE,
                )?;

                // 9. Residual add
                dispatch::add_residual(
                    session,
                    &scratch_guard,
                    so.tokens_off,
                    &scratch_guard,
                    so.attn_proj_off,
                    TokenShape { dim: hs, n_tokens },
                    DETOK_TILE,
                )?;

                // 10. FFN
                Self::dispatch_rms_norm_mul(
                    session,
                    &scratch_guard,
                    so.tokens_off,
                    &self.weights_buf,
                    lo.ffn_norm_off,
                    &scratch_guard,
                    so.normed_off,
                    eps,
                    hs,
                    n_tokens,
                )?;
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.normed_off,
                    &self.weights_buf,
                    lo.ffn_w1,
                    None,
                    &scratch_guard,
                    so.ffn_gate_off,
                    n_tokens,
                    DETOK_TILE,
                )?;
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.normed_off,
                    &self.weights_buf,
                    lo.ffn_w3,
                    None,
                    &scratch_guard,
                    so.ffn_up_off,
                    n_tokens,
                    DETOK_TILE,
                )?;
                Self::dispatch_swiglu(
                    session,
                    &scratch_guard,
                    so.ffn_gate_off,
                    &scratch_guard,
                    so.ffn_up_off,
                    &scratch_guard,
                    so.ffn_mid_off,
                    ffn_dim,
                    n_tokens,
                )?;
                dispatch::linear_m(
                    session,
                    &scratch_guard,
                    so.ffn_mid_off,
                    &self.weights_buf,
                    lo.ffn_w2,
                    None,
                    &scratch_guard,
                    so.ffn_out_off,
                    n_tokens,
                    DETOK_TILE,
                )?;
                dispatch::add_residual(
                    session,
                    &scratch_guard,
                    so.tokens_off,
                    &scratch_guard,
                    so.ffn_out_off,
                    TokenShape { dim: hs, n_tokens },
                    DETOK_TILE,
                )?;
            }
        }

        // 5. Output norm
        Self::dispatch_rms_norm_mul(
            session,
            &scratch_guard,
            so.tokens_off,
            &self.weights_buf,
            self.weights_offsets.output_norm_off,
            &scratch_guard,
            so.normed_off,
            eps,
            hs,
            n_tokens,
        )?;

        // 6. Linear projection head
        dispatch::linear_m(
            session,
            &scratch_guard,
            so.normed_off,
            &self.weights_buf,
            self.weights_offsets.lin_w,
            Some(self.weights_offsets.lin_b_off),
            &scratch_guard,
            so.spec_out_off,
            n_tokens,
            DETOK_TILE,
        )?;

        // 7. Submit session and await completion
        session.flush()?;

        // 8. Read back spectrogram frames
        let spec_total_floats = n_tokens * spec_per_frame;
        scratch_guard.invalidate_cpu_cache(so.spec_out_off, spec_total_floats * 4);
        let spec_bytes =
            &scratch_guard.as_slice()[so.spec_out_off..so.spec_out_off + spec_total_floats * 4];
        let spec_slice = bytemuck::cast_slice::<u8, f32>(spec_bytes);
        let result = spec_slice.to_vec();

        // 9. Advance past token count
        state_guard.n_past += n_tokens;

        Ok(result)
    }
}

impl AudioAccelerator for HexagonAudioDecoder {
    fn sample_audio_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8] {
        if let Some(ref df) = self.depthformer {
            match df.sample_frame(&self.device, embedding, temperature, top_k) {
                Ok(codes) => codes,
                Err(e) => {
                    self.record_error(e);
                    [0; 8]
                }
            }
        } else {
            [0; 8]
        }
    }

    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
        let n_frames = 6;
        let hs = cpu_weights.config.n_embd;
        let n_fft_bins = cpu_weights.config.n_fft / 2 + 1;
        let spec_per_frame = n_fft_bins * 2;
        let expected_spec_len = n_frames * spec_per_frame;

        // 1. Embed codes and upsample to 6 tokens on host CPU
        let tokens = {
            let emb = detok_embed_codes(cpu_weights, codes);
            upsample(&emb, hs, n_frames)
        };

        // 2. Dispatch forward pass to Hexagon NPU
        match self.detokenize_to_spectrum_npu(&tokens, n_frames) {
            Ok(spec) if spec.len() == expected_spec_len => spec,
            Err(e) => {
                crate::backend::hexagon::hexagon_warn!(
                    "Hexagon audio detokenizer NPU execution failed: {e}"
                );
                {
                    let mut dev = self.device.lock_or_recover();
                    dev.queue_session_mut().drop_pending_batch();
                }
                // The failed pass may have advanced the on-device conv/KV
                // state partway (`n_past` only moves on success). The engine
                // recomputes this frame on CPU; restart the NPU detokenizer
                // from a clean state rather than continue over a torn one.
                self.reset_detokenizer();
                self.record_error(e);
                Vec::new()
            }
            Ok(spec) => {
                let err = CeraError::Backend(format!(
                    "unexpected spectrum size: got {}, expected {}",
                    spec.len(),
                    expected_spec_len
                ));
                crate::backend::hexagon::hexagon_warn!(
                    "Hexagon audio detokenizer NPU output length mismatch: {err}"
                );
                self.reset_detokenizer();
                self.record_error(err);
                Vec::new()
            }
        }
    }

    fn reset_depthformer(&self) {
        if let Some(ref df) = self.depthformer {
            df.reset();
        }
    }

    fn reset_detokenizer(&self) {
        // The buffer is fully overwritten with zeros, so a poisoned lock is
        // recoverable; skipping it would reset the counters over stale state.
        {
            let sb_guard = self.state_buf.lock_or_recover();
            unsafe {
                std::ptr::write_bytes(sb_guard.as_mut_ptr(), 0, sb_guard.size());
            }
            sb_guard.flush_cpu_cache(0, sb_guard.size());
        }
        let mut state_guard = self.state.lock_or_recover();
        state_guard.reset();
    }

    fn supports_depthformer(&self) -> bool {
        self.depthformer.is_some()
    }

    fn depthformer_default_on(&self) -> bool {
        true
    }

    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        istft_to_pcm(spectrum, n_fft, hop_length)
    }

    fn take_audio_error(&self) -> Option<CeraError> {
        crate::model::take_fault(&self.last_error)
    }

    fn try_acquire_session(&self) -> bool {
        self.session_active
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    }

    fn release_session(&self) {
        self.session_active
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Attempt to create a hardware-accelerated Hexagon NPU audio decoder.
///
/// Probes for Qualcomm DSP hardware, loading Unsigned PD skeleton libraries.
/// Returns `None` if Hexagon hardware or libraries are unavailable, falling
/// back cleanly to CPU execution.
pub fn try_hexagon_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    let detok_weights = match DetokenizerWeights::from_gguf(gguf) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            crate::backend::hexagon::hexagon_warn!(
                "audio decoder: failed to parse detokenizer weights from GGUF: {e:#}"
            );
            return None;
        }
    };
    let audio_dec_weights = AudioDecoderWeights::from_gguf(gguf).ok().map(Arc::new);

    let context = HexagonContext::new()
        .inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonAudioDecoder", e);
        })
        .ok()?;

    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(HexagonArch::from_u32);

    let dev = match crate::backend::hexagon::probe_device(context.driver(), arch_override) {
        Ok(d) => d,
        Err(e) => {
            tracing::info!("HexagonAudioDecoder: DSP device unavailable ({e}), falling back");
            return None;
        }
    };
    let device = Arc::new(Mutex::new(dev));

    match HexagonAudioDecoder::new(
        Arc::clone(context.driver()),
        device,
        detok_weights,
        audio_dec_weights,
    ) {
        Ok(decoder) => {
            tracing::info!("audio decoder: using native Hexagon NPU backend");
            Some(Arc::new(decoder))
        }
        Err(e) => {
            crate::backend::hexagon::hexagon_error!("failed to create HexagonAudioDecoder: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tile policy pin: see `VIT_TILE`; a flip must be deliberate. Pinned by
    /// behavior: 130 tokens are one Norm/Mul/Add triple.
    #[test]
    fn tile_policy_is_whole() {
        assert_eq!(
            crate::backend::hexagon::dispatch::testing::layer_norm_op_count(DETOK_TILE),
            3
        );
    }

    #[test]
    fn test_detok_scratch_offsets_alignment() {
        let so = HexagonDetokScratchOffsets::new(512, 1024, 641, 64);
        assert_eq!(so.tokens_off, 0);
        assert_eq!(so.tokens_off % 128, 0);
        assert_eq!(so.normed_off % 128, 0);
        assert_eq!(so.conv_bcx_off % 128, 0);
        assert_eq!(so.conv_bx_off % 128, 0);
        assert_eq!(so.conv_t0_off % 128, 0);
        assert_eq!(so.conv_t1_off % 128, 0);
        assert_eq!(so.conv_y_off % 128, 0);
        assert_eq!(so.conv_out_off % 128, 0);
        assert_eq!(so.q_off % 128, 0);
        assert_eq!(so.k_off % 128, 0);
        assert_eq!(so.v_off % 128, 0);
        assert_eq!(so.k_f16_off % 128, 0);
        assert_eq!(so.v_f16_off % 128, 0);
        assert_eq!(so.attn_out_off % 128, 0);
        assert_eq!(so.attn_proj_off % 128, 0);
        assert_eq!(so.pos_off % 128, 0);
        assert_eq!(so.ffn_gate_off % 128, 0);
        assert_eq!(so.ffn_up_off % 128, 0);
        assert_eq!(so.ffn_mid_off % 128, 0);
        assert_eq!(so.ffn_out_off % 128, 0);
        assert_eq!(so.spec_out_off % 128, 0);
        assert!(so.total_bytes > so.spec_out_off);
        assert!(so.total_bytes < 16 * 1024 * 1024);
    }

    #[test]
    fn test_detok_state_offsets_alignment() {
        let cfg = DetokenizerConfig {
            n_layer: 8,
            n_embd: 512,
            n_head: 8,
            n_head_kv: 2,
            n_embd_head: 64,
            ffn_dim: 1024,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 8,
            n_fft: 1280,
            hop_length: 320,
            sample_rate: 24000,
            layer_is_conv: vec![true, true, false, true, false, true, false, true],
        };

        let so = HexagonDetokStateOffsets::plan(&cfg);
        assert_eq!(so.conv_states.len(), 8);
        assert_eq!(so.attn_states.len(), 8);
        assert!(so.conv_states[0].is_some());
        assert!(so.attn_states[0].is_none());
        assert!(so.conv_states[2].is_none());
        assert!(so.attn_states[2].is_some());
        assert_eq!(so.total_bytes % 128, 0);
        assert!(so.total_bytes > 0);
        assert!(so.total_bytes < 1024 * 1024);
    }

    #[test]
    fn test_try_hexagon_audio_decoder_on_host_without_dsp() {
        let gguf = Arc::new(crate::gguf::GgufBuilder::new().build());
        let result = try_hexagon_audio_decoder(&gguf);
        assert!(result.is_none());
    }

    #[test]
    fn test_hexagon_detok_weight_offsets_plan_rejects_f32() {
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
        use crate::model::audio_decoder::DetokLayerWeights;
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
        let err = plan_res.unwrap_err().to_string();
        assert!(err.contains("requires Q8_0 or Q4_0"));
    }

    #[test]
    fn test_depthformer_scratch_offsets_alignment() {
        let so = HexagonDepthformerScratchOffsets::new(2048, 1024, 4096, 2049, 32, 8, 32);
        assert_eq!(so.llm_emb_off, 0);
        assert_eq!(so.llm_emb_off % 128, 0);
        assert_eq!(so.hidden_off % 128, 0);
        assert_eq!(so.token_emb_off % 128, 0);
        assert_eq!(so.pos_off % 128, 0);
        assert_eq!(so.normed_off % 128, 0);
        assert_eq!(so.qkv_off % 128, 0);
        assert_eq!(so.attn_out_off % 128, 0);
        assert_eq!(so.attn_proj_off % 128, 0);
        assert_eq!(so.ffn_gate_off % 128, 0);
        assert_eq!(so.ffn_up_off % 128, 0);
        assert_eq!(so.ffn_mid_off % 128, 0);
        assert_eq!(so.ffn_out_off % 128, 0);
        assert_eq!(so.logits_off % 128, 0);
        assert!(so.total_bytes > so.logits_off);
        assert!(so.total_bytes < 1024 * 1024);
    }

    #[test]
    fn test_depthformer_state_offsets_alignment() {
        let cfg = DepthformerConfig {
            n_layer: 6,
            n_embd: 1024,
            n_head: 32,
            n_head_kv: 8,
            n_embd_head: 32,
            ffn_dim: 4096,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            max_seq_len: 8,
        };
        let so = HexagonDepthformerStateOffsets::plan(&cfg);
        assert_eq!(so.layers.len(), 6);
        for l in &so.layers {
            assert_eq!(l.k_cache_off % 128, 0);
            assert_eq!(l.v_cache_off % 128, 0);
        }
        assert_eq!(so.total_bytes % 128, 0);
        assert!(so.total_bytes > 0);
        assert!(so.total_bytes < 1024 * 1024);
    }

    #[test]
    fn test_depthformer_weight_offsets_plan() {
        use crate::model::audio_decoder::{
            CodebookWeights, DecoderConfig, DepthformerLayerWeights,
        };
        let df_cfg = DepthformerConfig {
            n_layer: 1,
            n_embd: 64,
            n_head: 2,
            n_head_kv: 1,
            n_embd_head: 32,
            ffn_dim: 128,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            max_seq_len: 8,
        };
        let dec_cfg = DecoderConfig {
            n_codebook: 2,
            n_vocab: 64,
            n_embd: 64,
            rms_norm_eps: 1e-5,
        };
        let weights = AudioDecoderWeights {
            depthformer_config: df_cfg,
            decoder_config: dec_cfg,
            depthformer_layers: vec![DepthformerLayerWeights {
                operator_norm: vec![1.0f32; 64],
                wqkv: MmapWeight::from_owned_bytes(vec![0u8; 128 * 2 * 18], DType::Q4_0, 128, 64),
                q_norm: vec![1.0f32; 32],
                k_norm: vec![1.0f32; 32],
                wo: MmapWeight::from_owned_bytes(vec![0u8; 64 * 2 * 18], DType::Q4_0, 64, 64),
                ffn_norm: vec![1.0f32; 64],
                w1: MmapWeight::from_owned_bytes(vec![0u8; 128 * 2 * 18], DType::Q4_0, 128, 64),
                w2: MmapWeight::from_owned_bytes(vec![0u8; 64 * 4 * 18], DType::Q4_0, 64, 128),
                w3: MmapWeight::from_owned_bytes(vec![0u8; 128 * 2 * 18], DType::Q4_0, 128, 64),
            }],
            depth_linear_w: MmapWeight::from_owned_bytes(
                vec![0u8; 128 * 2 * 18],
                DType::Q4_0,
                128,
                64,
            ),
            depth_linear_b: vec![0.0f32; 128],
            depth_embeddings: vec![
                CodebookWeights {
                    embedding: MmapWeight::from_owned_bytes(
                        vec![0u8; 64 * 2 * 18],
                        DType::Q4_0,
                        64,
                        64,
                    ),
                    norm: vec![1.0f32; 64],
                    to_logits: MmapWeight::from_owned_bytes(
                        vec![0u8; 64 * 2 * 18],
                        DType::Q4_0,
                        64,
                        64,
                    ),
                },
                CodebookWeights {
                    embedding: MmapWeight::from_owned_bytes(
                        vec![0u8; 64 * 2 * 18],
                        DType::Q4_0,
                        64,
                        64,
                    ),
                    norm: vec![1.0f32; 64],
                    to_logits: MmapWeight::from_owned_bytes(
                        vec![0u8; 64 * 2 * 18],
                        DType::Q4_0,
                        64,
                        64,
                    ),
                },
            ],
            audio_embedding: CodebookWeights {
                embedding: MmapWeight::from_owned_bytes(
                    vec![0u8; 64 * 2 * 18],
                    DType::Q4_0,
                    64,
                    64,
                ),
                norm: vec![1.0f32; 64],
                to_logits: MmapWeight::from_owned_bytes(
                    vec![0u8; 64 * 2 * 18],
                    DType::Q4_0,
                    64,
                    64,
                ),
            },
            interleave: crate::audio_engine::InterleaveCadence::default(),
        };

        let plan =
            HexagonDepthformerWeightOffsets::plan(&weights).expect("plan depthformer weights");
        assert_eq!(plan.depth_linear_slices.len(), 2);
        assert_eq!(plan.depth_linear_biases.len(), 2);
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.cb_norms.len(), 2);
        assert_eq!(plan.cb_to_logits.len(), 2);
        assert_eq!(plan.total_bytes % 128, 0);
        assert!(plan.total_bytes > 0);
    }

    #[test]
    fn test_hexagon_audio_decoder_error_recording() {
        let last_error = Mutex::new(None);

        // Record first error
        crate::model::record_first_fault(&last_error, CeraError::Backend("dsp failure 1".into()));
        // Record second error (must be discarded by sticky contract)
        crate::model::record_first_fault(&last_error, CeraError::Backend("dsp failure 2".into()));

        let drained = crate::model::take_fault(&last_error);
        assert!(drained.is_some());
        assert_eq!(drained.unwrap().to_string(), "backend: dsp failure 1");

        // Subsequent drain is empty
        let drained_again = crate::model::take_fault(&last_error);
        assert!(drained_again.is_none());
    }
}

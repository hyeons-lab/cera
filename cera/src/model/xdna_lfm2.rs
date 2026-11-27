//! AMD XDNA2 NPU backend for causal LFM2 / LFM2.5 models (`arch = "lfm2"`).
//!
//! Uses dedicated AIE2P IRON kernels for `LFM2.5-1.2B` dimensions:
//! - `H = 2048`, `FFN = 8192`, `N_HEADS = 32`, `N_KV_HEADS = 8`, `HEAD_DIM = 64`
//! - `ShortConv` (`l_cache = 3`, `d_conv = 2`), standard pre-norm residuals, QK-norm + NeoX RoPE.
//! - Fused `B=32` prefill (`q4prefill_lfm2`) with device-side layer
//!   chaining (`layer_bo_y[l] -> layer_bo_y[l+1]`), plus fused `B=1` decode (`q4layer_lfm2`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, bail, ensure};

use crate::CeraError;
use crate::backend::xdna::device::{XdnaBo, XdnaDevice, XdnaKernel, XdnaRun, XdnaRunList};
use crate::backend::xdna::kernels::{
    LFM2_BLOCK_TOKENS, LFM2_CONV_TAPS, LFM2_D_CONV, LFM2_DEC_ACT_BYTES, LFM2_DEC_MAX_KV_PKTS,
    LFM2_DEC_PKT_TOKENS, LFM2_DEC_SLOT_NORM_0, LFM2_DEC_SLOT_NORM_1, LFM2_DEC_SLOT_ROPE_PKT,
    LFM2_DEC_YN_BYTES, LFM2_FFN, LFM2_H, LFM2_HEAD_DIM, LFM2_KV_BYTES, LFM2_KV_CHUNK_BYTES,
    LFM2_KV_CHUNK_T, LFM2_KV_DIM, LFM2_KVO_BYTES, LFM2_MAX_KV_CHUNKS, LFM2_MAX_TOKENS,
    LFM2_N_HEADS, LFM2_N_KV_HEADS, LFM2_PREFILL_ACT_BYTES, LFM2_PREFILL_CHUNK_T,
    LFM2_PREFILL_SLOT_EXT_PARAMS_0, LFM2_PREFILL_SLOT_EXT_PARAMS_1, LFM2_PREFILL_SLOT_EXT_PARAMS_2,
    LFM2_PREFILL_SLOT_NORM_0, LFM2_PREFILL_SLOT_ROPE_PKT_0, LFM2_PREFILL_SLOT_ROPE_PKT_1,
    LFM2_PREFILL_YN_BYTES, LFM2_Q_DIM, LFM2_W_BYTES, Q4LAYER_LFM2_ATTN_NO_FFN_BUCKETS,
    Q4LAYER_LFM2_CONV_INSTS, Q4LAYER_LFM2_CONV_NO_FFN_INSTS, Q4LAYER_LFM2_FA_FIN_BUCKETS,
    Q4LAYER_LFM2_FA_FIN_NO_FFN_BUCKETS, Q4LAYER_LFM2_FA_PART_C512_INSTS, Q4LAYER_LFM2_KV_BUCKETS,
    Q4LAYER_LFM2_PHASE_A_INSTS, Q4LAYER_LFM2_XCLBIN, Q4LAYER_TILE_F32, Q4PREFILL_LFM2_ATTN_BUCKETS,
    Q4PREFILL_LFM2_ATTN_NO_FFN_BUCKETS, Q4PREFILL_LFM2_CONV_INSTS,
    Q4PREFILL_LFM2_CONV_NO_FFN_INSTS, Q4PREFILL_LFM2_EXT_CONV_INSTS,
    Q4PREFILL_LFM2_EXT_CONV_NO_FFN_INSTS, Q4PREFILL_LFM2_EXT_XCLBIN, Q4PREFILL_LFM2_FA_FIN_BUCKETS,
    Q4PREFILL_LFM2_FA_FIN_NO_FFN_BUCKETS, Q4PREFILL_LFM2_FA_PART_C1024_INSTS,
    Q4PREFILL_LFM2_FA_PART0_C1024_INSTS, Q4PREFILL_LFM2_PHASE_A_INSTS, Q4PREFILL_LFM2_XCLBIN,
    init_lfm2_bo_y_norms, init_lfm2_prefill_bo_y_norms, inst_words, read_le_f32_slice,
    try_select_lfm3_kv_bucket, write_le_f32_slice, write_lfm2_prefill_step_params,
    write_lfm3_params_at, write_lfm3_step_params,
};
use crate::backend::xdna::repack::{f32_to_bf16, pack_f32_mse_stage_k1024, pack_q4_0_stage_k1024};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError, LayerState};
use crate::model::lfm2::{LfmModel, XdnaMoePrefillScratch};
use crate::model::xdna_llama::{
    HybridPendingSlot, QUIESCE_TIMEOUT_MS, check_hybrid_kv_rewind_state,
    validate_xdna_kv_compression,
};
use crate::model::{BlockType, Model, ModelConfig, ModelSessionGate, ModelSessionLease, WeightRef};
use crate::tensor::DType;

const TILE_BYTES: usize = Q4LAYER_TILE_F32 * 4;
const CONV_SYNC_BYTES: usize = 7 * TILE_BYTES + LFM2_D_CONV * 256 * 4;

#[derive(Debug)]
enum SyncErr {
    Input(String),
    Device(String),
}

fn exec_owned_run(
    stalled_runs: &mut Vec<XdnaRun>,
    run: XdnaRun,
    label: &str,
) -> Result<(), String> {
    run.start().map_err(|e| format!("{label}: {e}"))?;
    if let Err(e) = run.wait(QUIESCE_TIMEOUT_MS) {
        // Keep the run handle alive while hardware may still reference BOs.
        stalled_runs.push(run);
        return Err(format!("{label}: {e}"));
    }
    Ok(())
}

fn exec_cached_run_slot(
    slot: &mut Option<XdnaRun>,
    stalled_runs: &mut Vec<XdnaRun>,
    label: &str,
) -> Result<(), String> {
    let run = slot.as_ref().expect("populated cached run");
    run.start().map_err(|e| format!("{label}: {e}"))?;
    if let Err(e) = run.wait(QUIESCE_TIMEOUT_MS) {
        if let Some(stalled) = slot.take() {
            // Detach timed-out cached runs into `stalled_runs` so teardown never drops them while live.
            stalled_runs.push(stalled);
        }
        return Err(format!("{label}: {e}"));
    }
    Ok(())
}

fn take_fault(slot: &Mutex<Option<String>>) -> Option<CeraError> {
    slot.lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .map(CeraError::Backend)
}

fn validate_xdna_lfm2_cpu(cpu: &LfmModel) -> Result<()> {
    let cfg = cpu.config();
    ensure!(
        matches!(cfg.architecture.as_str(), "lfm2" | "lfm2moe"),
        "expected architecture `lfm2` or `lfm2moe`, got `{}`",
        cfg.architecture
    );
    ensure!(cfg.is_causal, "XDNA LFM2 backend requires causal attention");
    let is_moe = cfg.moe.is_some();
    if let Some(moe_cfg) = &cfg.moe {
        ensure!(
            cfg.architecture == "lfm2moe",
            "unexpected MoE config on architecture `{}`",
            cfg.architecture
        );
        ensure!(
            moe_cfg.n_expert > 0
                && moe_cfg.n_expert_used > 0
                && moe_cfg.n_expert_used <= moe_cfg.n_expert
                && moe_cfg.expert_ff_len > 0,
            "invalid MoE config for XDNA LFM2-MoE: n_expert={}, n_expert_used={}, expert_ff_len={}",
            moe_cfg.n_expert,
            moe_cfg.n_expert_used,
            moe_cfg.expert_ff_len
        );
    } else {
        ensure!(
            cfg.intermediate_size > 0,
            "XDNA LFM2 requires positive intermediate_size, got {}",
            cfg.intermediate_size
        );
    }
    let host_ffn = is_moe || cfg.intermediate_size != LFM2_FFN;
    ensure!(
        cfg.hidden_size == LFM2_H,
        "XDNA LFM2 requires hidden_size={LFM2_H}, got {}",
        cfg.hidden_size
    );
    ensure!(
        cfg.n_heads == LFM2_N_HEADS,
        "XDNA LFM2 requires n_heads={LFM2_N_HEADS}, got {}",
        cfg.n_heads
    );
    ensure!(
        cfg.head_dim == LFM2_HEAD_DIM,
        "XDNA LFM2 requires head_dim={LFM2_HEAD_DIM}, got {}",
        cfg.head_dim
    );
    ensure!(
        cfg.n_kv_heads == LFM2_N_KV_HEADS,
        "XDNA LFM2 requires n_kv_heads={LFM2_N_KV_HEADS}, got {}",
        cfg.n_kv_heads
    );
    ensure!(
        cfg.max_seq_len > 0 && cfg.max_seq_len <= LFM2_MAX_TOKENS,
        "unsupported context length {}: XDNA LFM2 supports 1..={LFM2_MAX_TOKENS}",
        cfg.max_seq_len
    );
    ensure!(cfg.n_layers > 0, "model has 0 layers");
    let (embd_ref, logits_ref) = cpu.xdna_embed_logits_refs();
    ensure!(
        crate::model::transformer::supports_row_dequant(embd_ref.dtype),
        "token_embd dtype {:?} unsupported by CPU row dequantizer",
        embd_ref.dtype
    );
    ensure!(
        crate::model::transformer::supports_row_dequant(logits_ref.dtype),
        "logits weight dtype {:?} unsupported by CPU row dequantizer",
        logits_ref.dtype
    );
    if cfg.block_types.contains(&BlockType::GatedConv) {
        ensure!(
            cfg.conv_kernel_size == Some(LFM2_CONV_TAPS),
            "XDNA LFM2 requires conv_kernel_size={LFM2_CONV_TAPS}, got {:?}",
            cfg.conv_kernel_size
        );
    }

    let check_weight =
        |label: &str, layer: usize, w: &WeightRef, m: usize, k: usize| -> Result<()> {
            ensure!(
                crate::model::transformer::supports_row_dequant(w.dtype),
                "layer {layer} {label}: XDNA LFM2 does not support weight dtype {:?}",
                w.dtype
            );
            ensure!(
                w.m == m && w.k == k,
                "layer {layer} {label}: shape {}x{} != expected {m}x{k}",
                w.m,
                w.k
            );
            Ok(())
        };
    let check_shape =
        |label: &str, layer: usize, w: &WeightRef, m: usize, k: usize| -> Result<()> {
            ensure!(
                w.m == m && w.k == k,
                "layer {layer} {label}: shape {}x{} != expected {m}x{k}",
                w.m,
                w.k
            );
            Ok(())
        };

    for l in 0..cfg.n_layers {
        ensure!(
            cpu.xdna_lfm3_post_attn_norm(l).is_none()
                && cpu.xdna_lfm3_post_ffn_norm(l).is_none()
                && cpu.xdna_lfm3_attn_gate_ref(l).is_none(),
            "layer {l}: XDNA LFM2 does not support post-norms or attn_gate"
        );
        if host_ffn {
            if let Some(moe) = cpu.xdna_moe_refs(l) {
                check_shape("ffn_gate_inp", l, &moe.router, moe.n_expert, LFM2_H)?;
                ensure!(
                    moe.gate.len() == moe.n_expert
                        && moe.up.len() == moe.n_expert
                        && moe.down.len() == moe.n_expert,
                    "layer {l}: MoE expert tensor count mismatch"
                );
                for e in 0..moe.n_expert {
                    check_shape("ffn_gate_exps", l, &moe.gate[e], moe.expert_ff_len, LFM2_H)?;
                    check_shape("ffn_up_exps", l, &moe.up[e], moe.expert_ff_len, LFM2_H)?;
                    check_shape("ffn_down_exps", l, &moe.down[e], LFM2_H, moe.expert_ff_len)?;
                }
            } else {
                let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
                let is = cfg.intermediate_size;
                check_weight("ffn_gate", l, fg, is, LFM2_H)?;
                check_weight("ffn_up", l, fu, is, LFM2_H)?;
                check_weight("ffn_down", l, fd, LFM2_H, is)?;
            }
        } else {
            let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
            check_weight("ffn_gate", l, fg, LFM2_FFN, LFM2_H)?;
            check_weight("ffn_up", l, fu, LFM2_FFN, LFM2_H)?;
            check_weight("ffn_down", l, fd, LFM2_H, LFM2_FFN)?;
        }

        match cfg.block_types[l] {
            BlockType::GatedConv => {
                let (in_proj, out_proj) = cpu
                    .xdna_layer_conv_refs(l)
                    .with_context(|| format!("layer {l} missing shortconv weights"))?;
                check_weight("shortconv.in_proj", l, in_proj, 3 * LFM2_H, LFM2_H)?;
                check_weight("shortconv.out_proj", l, out_proj, LFM2_H, LFM2_H)?;
                let taps = cpu
                    .xdna_conv_tap_first(l)
                    .with_context(|| format!("layer {l} missing shortconv.conv weight"))?;
                ensure!(
                    taps.len() == LFM2_CONV_TAPS * LFM2_H,
                    "layer {l} conv weight length {} != {}",
                    taps.len(),
                    LFM2_CONV_TAPS * LFM2_H
                );
            }
            BlockType::Attention => {
                let (wq, wk, wv, wo) = cpu
                    .xdna_layer_attn_refs(l)
                    .with_context(|| format!("layer {l} missing attention weights"))?;
                check_weight("attn_q", l, wq, LFM2_Q_DIM, LFM2_H)?;
                check_weight("attn_k", l, wk, LFM2_KV_DIM, LFM2_H)?;
                check_weight("attn_v", l, wv, LFM2_KV_DIM, LFM2_H)?;
                check_weight("attn_output", l, wo, LFM2_H, LFM2_Q_DIM)?;
                ensure!(
                    cfg.kv_heads_per_layer[l] == LFM2_N_KV_HEADS,
                    "layer {l}: kv_heads {} != {LFM2_N_KV_HEADS}",
                    cfg.kv_heads_per_layer[l]
                );
                let qn = cpu
                    .attn_q_norm_weight(l)
                    .with_context(|| format!("layer {l} missing attn_q_norm"))?;
                let kn = cpu
                    .attn_k_norm_weight(l)
                    .with_context(|| format!("layer {l} missing attn_k_norm"))?;
                ensure!(
                    qn.len() == LFM2_HEAD_DIM && kn.len() == LFM2_HEAD_DIM,
                    "layer {l}: q/k norm length mismatch"
                );
            }
            other => bail!("layer {l}: unsupported block type {other:?} for XDNA LFM2"),
        }
    }
    Ok(())
}

struct Lfm2LayerStatic {
    is_conv: bool,
    rope_theta: f32,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

struct Lfm2Kernels {
    device: XdnaDevice,
    host_ffn: bool,
    prefill_conv: XdnaKernel,
    prefill_ext_conv: Option<XdnaKernel>,
    prefill_attn: Vec<Option<XdnaKernel>>,
    prefill_phase_a: Option<XdnaKernel>,
    prefill_fa_part0: Option<XdnaKernel>,
    prefill_fa_part: Option<XdnaKernel>,
    prefill_fa_fin: Vec<Option<XdnaKernel>>,
    dec_conv: XdnaKernel,
    dec_attn: Vec<Option<XdnaKernel>>,
    dec_phase_a: Option<XdnaKernel>,
    dec_fa_part: Option<XdnaKernel>,
    dec_fa_fin: Vec<Option<XdnaKernel>>,
}

impl Lfm2Kernels {
    fn open(host_ffn: bool) -> Result<Self> {
        let device = XdnaDevice::open(0).map_err(|e| anyhow::anyhow!("XDNA open failed: {e}"))?;
        let pre_conv_insts = if host_ffn {
            Q4PREFILL_LFM2_CONV_NO_FFN_INSTS
        } else {
            Q4PREFILL_LFM2_CONV_INSTS
        };
        let prefill_conv = device
            .load_kernel(Q4PREFILL_LFM2_XCLBIN, &inst_words(pre_conv_insts))
            .map_err(|e| anyhow::anyhow!("XDNA load q4prefill_lfm2 conv failed: {e}"))?;

        // Preload every prefill attention bucket consecutively right after `prefill_conv`
        // so all prefill kernels share one XRT hardware context and can be batched
        // together in a single `XdnaRunList`.
        let pre_attn_buckets = if host_ffn {
            &Q4PREFILL_LFM2_ATTN_NO_FFN_BUCKETS
        } else {
            &Q4PREFILL_LFM2_ATTN_BUCKETS
        };
        let mut prefill_attn: Vec<Option<XdnaKernel>> =
            (0..pre_attn_buckets.len()).map(|_| None).collect();
        for (bi, &(chunks, inst_bytes)) in pre_attn_buckets.iter().enumerate() {
            prefill_attn[bi] = Some(
                device
                    .load_kernel(Q4PREFILL_LFM2_XCLBIN, &inst_words(inst_bytes))
                    .map_err(|e| {
                        anyhow::anyhow!("XDNA load q4prefill_lfm2 attn (c={chunks}) failed: {e}")
                    })?,
            );
        }

        let dec_conv_insts = if host_ffn {
            Q4LAYER_LFM2_CONV_NO_FFN_INSTS
        } else {
            Q4LAYER_LFM2_CONV_INSTS
        };
        let dec_conv = device
            .load_kernel(Q4LAYER_LFM2_XCLBIN, &inst_words(dec_conv_insts))
            .map_err(|e| anyhow::anyhow!("XDNA load q4layer_lfm2 conv failed: {e}"))?;

        // Preload every decode attention bucket consecutively right after `dec_conv`
        // so all decode kernels share one XRT hardware context and can be batched
        // together in a single `XdnaRunList`.
        let dec_attn_buckets = if host_ffn {
            &Q4LAYER_LFM2_ATTN_NO_FFN_BUCKETS
        } else {
            &Q4LAYER_LFM2_KV_BUCKETS
        };
        let mut dec_attn: Vec<Option<XdnaKernel>> =
            (0..dec_attn_buckets.len()).map(|_| None).collect();
        for (bi, &(chunks, inst_bytes)) in dec_attn_buckets.iter().enumerate() {
            dec_attn[bi] = Some(
                device
                    .load_kernel(Q4LAYER_LFM2_XCLBIN, &inst_words(inst_bytes))
                    .map_err(|e| {
                        anyhow::anyhow!("XDNA load q4layer_lfm2 attn (c={chunks}) failed: {e}")
                    })?,
            );
        }

        Ok(Self {
            device,
            host_ffn,
            prefill_conv,
            prefill_ext_conv: None,
            prefill_attn,
            prefill_phase_a: None,
            prefill_fa_part0: None,
            prefill_fa_part: None,
            prefill_fa_fin: (0..Q4PREFILL_LFM2_FA_FIN_BUCKETS.len())
                .map(|_| None)
                .collect(),
            dec_conv,
            dec_attn,
            dec_phase_a: None,
            dec_fa_part: None,
            dec_fa_fin: (0..Q4LAYER_LFM2_FA_FIN_BUCKETS.len())
                .map(|_| None)
                .collect(),
        })
    }

    fn prefill_attn(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.prefill_attn[bidx].is_none() {
            let buckets = if self.host_ffn {
                &Q4PREFILL_LFM2_ATTN_NO_FFN_BUCKETS
            } else {
                &Q4PREFILL_LFM2_ATTN_BUCKETS
            };
            let (chunks, inst) = buckets[bidx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM2_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load prefill attn bucket c={chunks}: {e}"))?;
            self.prefill_attn[bidx] = Some(k);
        }
        Ok(self.prefill_attn[bidx].as_ref().expect("populated bucket"))
    }

    fn prefill_ext_conv(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_ext_conv.is_none() {
            let inst = if self.host_ffn {
                Q4PREFILL_LFM2_EXT_CONV_NO_FFN_INSTS
            } else {
                Q4PREFILL_LFM2_EXT_CONV_INSTS
            };
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM2_EXT_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load prefill ext conv: {e}"))?;
            self.prefill_ext_conv = Some(k);
        }
        Ok(self.prefill_ext_conv.as_ref().expect("populated ext conv"))
    }

    fn prefill_phase_a(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_phase_a.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM2_EXT_XCLBIN,
                    &inst_words(Q4PREFILL_LFM2_PHASE_A_INSTS),
                )
                .map_err(|e| format!("load prefill phase_a: {e}"))?;
            self.prefill_phase_a = Some(k);
        }
        Ok(self.prefill_phase_a.as_ref().expect("populated phase_a"))
    }

    fn prefill_fa_part0(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_part0.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM2_EXT_XCLBIN,
                    &inst_words(Q4PREFILL_LFM2_FA_PART0_C1024_INSTS),
                )
                .map_err(|e| format!("load prefill fa_part0: {e}"))?;
            self.prefill_fa_part0 = Some(k);
        }
        Ok(self.prefill_fa_part0.as_ref().expect("populated fa_part0"))
    }

    fn prefill_fa_part(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_part.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM2_EXT_XCLBIN,
                    &inst_words(Q4PREFILL_LFM2_FA_PART_C1024_INSTS),
                )
                .map_err(|e| format!("load prefill fa_part: {e}"))?;
            self.prefill_fa_part = Some(k);
        }
        Ok(self.prefill_fa_part.as_ref().expect("populated fa_part"))
    }

    fn prefill_fa_fin(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_fin[bidx].is_none() {
            let buckets = if self.host_ffn {
                &Q4PREFILL_LFM2_FA_FIN_NO_FFN_BUCKETS
            } else {
                &Q4PREFILL_LFM2_FA_FIN_BUCKETS
            };
            let (chunks, inst) = buckets[bidx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM2_EXT_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load prefill fa_fin bucket c={chunks}: {e}"))?;
            self.prefill_fa_fin[bidx] = Some(k);
        }
        Ok(self.prefill_fa_fin[bidx]
            .as_ref()
            .expect("populated fa_fin"))
    }

    fn dec_attn(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.dec_attn[bidx].is_none() {
            let buckets = if self.host_ffn {
                &Q4LAYER_LFM2_ATTN_NO_FFN_BUCKETS
            } else {
                &Q4LAYER_LFM2_KV_BUCKETS
            };
            let (chunks, inst) = buckets[bidx];
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM2_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load decode attn bucket c={chunks}: {e}"))?;
            self.dec_attn[bidx] = Some(k);
        }
        Ok(self.dec_attn[bidx].as_ref().expect("populated bucket"))
    }

    fn dec_phase_a(&mut self) -> Result<&XdnaKernel, String> {
        if self.dec_phase_a.is_none() {
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM2_XCLBIN, &inst_words(Q4LAYER_LFM2_PHASE_A_INSTS))
                .map_err(|e| format!("load decode phase_a: {e}"))?;
            self.dec_phase_a = Some(k);
        }
        Ok(self.dec_phase_a.as_ref().expect("populated dec_phase_a"))
    }

    fn dec_fa_part(&mut self) -> Result<&XdnaKernel, String> {
        if self.dec_fa_part.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4LAYER_LFM2_XCLBIN,
                    &inst_words(Q4LAYER_LFM2_FA_PART_C512_INSTS),
                )
                .map_err(|e| format!("load decode fa_part: {e}"))?;
            self.dec_fa_part = Some(k);
        }
        Ok(self.dec_fa_part.as_ref().expect("populated dec_fa_part"))
    }

    fn dec_fa_fin(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.dec_fa_fin[bidx].is_none() {
            let buckets = if self.host_ffn {
                &Q4LAYER_LFM2_FA_FIN_NO_FFN_BUCKETS
            } else {
                &Q4LAYER_LFM2_FA_FIN_BUCKETS
            };
            let (chunks, inst) = buckets[bidx];
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM2_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load decode fa_fin bucket c={chunks}: {e}"))?;
            self.dec_fa_fin[bidx] = Some(k);
        }
        Ok(self.dec_fa_fin[bidx]
            .as_ref()
            .expect("populated dec_fa_fin"))
    }
}

struct Lfm2ExtRunCache {
    chains: [Vec<(usize, usize, Lfm2PrefillChain)>; 2],
    conv_groups: [Vec<Option<Lfm2PrefillChain>>; 2],
    attn_chains: Vec<Vec<(usize, usize, Lfm2PrefillChain)>>,
}

impl Lfm2ExtRunCache {
    fn new(n_layers: usize) -> Self {
        Self {
            chains: [Vec::new(), Vec::new()],
            conv_groups: [
                (0..n_layers).map(|_| None).collect(),
                (0..n_layers).map(|_| None).collect(),
            ],
            attn_chains: (0..n_layers).map(|_| Vec::new()).collect(),
        }
    }
}

struct Lfm2PrefillAttnMirrorEntry {
    layer: usize,
    pos0: usize,
    blk0: usize,
    loc0: usize,
    n0: usize,
    blk1: usize,
    loc1: usize,
    n1: usize,
}

struct Lfm2PrefillChain {
    // Drop the runlist before its child run handles so XRT tears down the list cleanly first.
    runlist: XdnaRunList,
    _runs: Vec<XdnaRun>,
}

struct Lfm2PrefillBuffers {
    // Drop compiled chains/runs before the BOs and kernels they reference.
    bucket_plans: [Vec<Option<Lfm2PrefillChain>>; 2],
    ext_runs: Option<Lfm2ExtRunCache>,
    moe_conv_runs: [Vec<Option<XdnaRun>>; 2],
    moe_attn_runs: Vec<Vec<Option<XdnaRun>>>,
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    bo_yn: XdnaBo,
    conv_kv_in_kvo: bool,
    last_conv_n_valid: Option<usize>,
}

struct Lfm2DecodeBucketPlan {
    // Drop the runlist before its child run handles so XRT tears down the list cleanly first.
    runlist: XdnaRunList,
    _runs: Vec<XdnaRun>,
}

/// Per-parity bounded LRU cache of extended (`>8K`) decode runlists so long contexts
/// cannot retain unbounded live WDDM command allocations across block transitions.
struct Lfm2ExtPlanLru {
    cap: usize,
    entries: [VecDeque<((usize, usize), Lfm2DecodeBucketPlan)>; 2],
}

impl Lfm2ExtPlanLru {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            entries: [VecDeque::new(), VecDeque::new()],
        }
    }

    fn get_or_insert_with<F>(
        &mut self,
        parity_idx: usize,
        key: (usize, usize),
        build: F,
    ) -> Result<&Lfm2DecodeBucketPlan, String>
    where
        F: FnOnce() -> Result<Lfm2DecodeBucketPlan, String>,
    {
        let q = &mut self.entries[parity_idx];
        if let Some(pos) = q.iter().position(|(k, _)| *k == key) {
            if pos + 1 != q.len()
                && let Some(item) = q.remove(pos)
            {
                q.push_back(item);
            }
            return Ok(&q.back().expect("present").1);
        }
        while q.len() >= self.cap {
            q.pop_front();
        }
        let plan = build()?;
        q.push_back((key, plan));
        Ok(&q.back().expect("inserted").1)
    }

    fn peek(&self, parity_idx: usize, key: (usize, usize)) -> Option<&Lfm2DecodeBucketPlan> {
        self.entries[parity_idx]
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
    }

    fn clear(&mut self) {
        self.entries[0].clear();
        self.entries[1].clear();
    }
}

struct Lfm2DecodeBuffers {
    // Drop compiled runlists before the BOs and kernels they reference.
    bucket_plans: [Vec<Option<Lfm2DecodeBucketPlan>>; 2],
    ext_plans: Lfm2ExtPlanLru,
    moe_conv_runs: [Vec<Option<XdnaRun>>; 2],
    moe_attn_runs: Vec<Vec<Option<XdnaRun>>>,
    moe_ext_plans: [Vec<Vec<(usize, usize, Lfm2DecodeBucketPlan)>>; 2],
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    bo_yn: XdnaBo,
    conv_kv_in_kvo: bool,
}

#[derive(Clone, Copy)]
enum InFlightLfm2Run {
    DecFast {
        parity_idx: usize,
        bidx: usize,
    },
    DecExt {
        parity_idx: usize,
        key: (usize, usize),
    },
    DecMoeExt {
        parity_idx: usize,
        l: usize,
        blk: usize,
        fin_bidx: usize,
    },
    PreFast {
        parity_idx: usize,
        bidx: usize,
    },
    PreExt {
        parity_idx: usize,
        key: (usize, usize),
    },
    PreExtConv {
        parity_idx: usize,
        l: usize,
    },
    PreExtAttn {
        l: usize,
        blk: usize,
        fin_bidx: usize,
    },
}

const LFM2_ATTN_DEC_KVO_BYTES: usize = 8 * 1024;

struct XdnaLfm2State {
    // Drop active/stalled runs and compiled chains before weights/KV buffers and `kernels` last.
    in_flight: Option<InFlightLfm2Run>,
    stalled_runs: Vec<XdnaRun>,
    prefill: Option<Lfm2PrefillBuffers>,
    decode: Option<Lfm2DecodeBuffers>,
    layer_bo_w: Vec<XdnaBo>,
    layer_kv_blocks: Vec<Vec<XdnaBo>>,
    pending_kv: HybridPendingSlot,
    synced_len: usize,
    force_conv_sync: bool,
    head_q8_scratch: (Vec<f32>, Vec<i8>),
    prefill_embs_scratch: (Vec<f32>, Vec<f32>),
    moe_ffn_input: Vec<f32>,
    moe_hidden_scratch: Vec<f32>,
    moe_prefill_scratch: XdnaMoePrefillScratch,
    kernels: Lfm2Kernels,
}

impl XdnaLfm2State {
    fn quiesce_in_flight(&mut self) -> Result<(), String> {
        while let Some(run) = self.stalled_runs.pop() {
            if let Err(e) = run.wait(QUIESCE_TIMEOUT_MS) {
                self.stalled_runs.push(run);
                return Err(format!("lfm2 stalled run wait failed: {e}"));
            }
        }
        let Some(run) = self.in_flight.as_ref() else {
            return Ok(());
        };
        let list = match run {
            InFlightLfm2Run::DecFast { parity_idx, bidx } => self
                .decode
                .as_ref()
                .and_then(|d| d.bucket_plans[*parity_idx].get(*bidx))
                .and_then(|o| o.as_ref())
                .map(|p| &p.runlist),
            InFlightLfm2Run::DecExt { parity_idx, key } => self
                .decode
                .as_ref()
                .and_then(|d| d.ext_plans.peek(*parity_idx, *key))
                .map(|p| &p.runlist),
            InFlightLfm2Run::DecMoeExt {
                parity_idx,
                l,
                blk,
                fin_bidx,
            } => self
                .decode
                .as_ref()
                .and_then(|d| d.moe_ext_plans[*parity_idx].get(*l))
                .and_then(|v| v.iter().find(|(b, f, _)| *b == *blk && *f == *fin_bidx))
                .map(|(_, _, p)| &p.runlist),
            InFlightLfm2Run::PreFast { parity_idx, bidx } => self
                .prefill
                .as_ref()
                .and_then(|p| p.bucket_plans[*parity_idx].get(*bidx))
                .and_then(|o| o.as_ref())
                .map(|c| &c.runlist),
            InFlightLfm2Run::PreExt { parity_idx, key } => self
                .prefill
                .as_ref()
                .and_then(|p| p.ext_runs.as_ref())
                .and_then(|e| {
                    e.chains[*parity_idx]
                        .iter()
                        .find(|(b, f, _)| (*b, *f) == *key)
                })
                .map(|(_, _, c)| &c.runlist),
            InFlightLfm2Run::PreExtConv { parity_idx, l } => self
                .prefill
                .as_ref()
                .and_then(|p| p.ext_runs.as_ref())
                .and_then(|e| e.conv_groups[*parity_idx].get(*l))
                .and_then(|o| o.as_ref())
                .map(|c| &c.runlist),
            InFlightLfm2Run::PreExtAttn { l, blk, fin_bidx } => self
                .prefill
                .as_ref()
                .and_then(|p| p.ext_runs.as_ref())
                .and_then(|e| e.attn_chains.get(*l))
                .and_then(|v| v.iter().find(|(b, f, _)| *b == *blk && *f == *fin_bidx))
                .map(|(_, _, c)| &c.runlist),
        };
        if let Some(runlist) = list {
            runlist
                .wait(QUIESCE_TIMEOUT_MS)
                .map_err(|e| format!("lfm2 quiesce wait failed: {e}"))?;
        }
        self.in_flight = None;
        Ok(())
    }
}

unsafe impl Send for XdnaLfm2State {}

fn dequant_matrix(
    cpu: &LfmModel,
    w: &WeightRef,
    row_start: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; rows * cols];
    let gguf = cpu.gguf();
    for r in 0..rows {
        crate::model::transformer::dequantize_row_into(
            gguf,
            w,
            row_start + r,
            &mut buf[r * cols..(r + 1) * cols],
        );
    }
    buf
}

fn pack_weight_stage(
    cpu: &LfmModel,
    wb: &mut [u8],
    tile_off: usize,
    w: &WeightRef,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if w.dtype == DType::Q4_0 {
        let raw = cpu.weight_bytes(w);
        pack_q4_0_stage_k1024(&mut wb[tile_off * TILE_BYTES..], raw, rows, cols)
            .map_err(|e| anyhow::anyhow!("pack weight stage at tile {tile_off}: {e}"))?;
    } else {
        let f32_buf = dequant_matrix(cpu, w, 0, rows, cols);
        pack_f32_mse_stage_k1024(&mut wb[tile_off * TILE_BYTES..], &f32_buf, rows, cols)
            .map_err(|e| anyhow::anyhow!("pack f32 weight stage at tile {tile_off}: {e}"))?;
    }
    Ok(())
}

fn pack_conv_in_proj(cpu: &LfmModel, wb: &mut [u8], in_proj: &WeightRef) -> Result<()> {
    if in_proj.dtype == DType::Q4_0 {
        let raw = cpu.weight_bytes(in_proj);
        let part_bytes = raw.len() / 3;
        for (part, tile_off) in [(0usize, 0usize), (1, 128), (2, 256)] {
            let slice = &raw[part * part_bytes..(part + 1) * part_bytes];
            pack_q4_0_stage_k1024(&mut wb[tile_off * TILE_BYTES..], slice, LFM2_H, LFM2_H)
                .map_err(|e| anyhow::anyhow!("pack conv in_proj part {part}: {e}"))?;
        }
    } else {
        for (part, tile_off) in [(0usize, 0usize), (1, 128), (2, 256)] {
            let f32_buf = dequant_matrix(cpu, in_proj, part * LFM2_H, LFM2_H, LFM2_H);
            pack_f32_mse_stage_k1024(&mut wb[tile_off * TILE_BYTES..], &f32_buf, LFM2_H, LFM2_H)
                .map_err(|e| anyhow::anyhow!("pack f32 conv in_proj part {part}: {e}"))?;
        }
    }
    Ok(())
}

fn pack_conv_wt(wb: &mut [u8], tile_off: usize, conv_wt: &[f32]) -> Result<()> {
    ensure!(
        conv_wt.len() == LFM2_CONV_TAPS * LFM2_H,
        "conv_wt length {} != {}",
        conv_wt.len(),
        LFM2_CONV_TAPS * LFM2_H
    );
    for col in 0..8 {
        let dst_off_f32 = (tile_off + col) * Q4LAYER_TILE_F32;
        for row in 0..4 {
            let chan = col * 256 + row * 64;
            for k in 0..LFM2_CONV_TAPS {
                write_le_f32_slice(
                    wb,
                    dst_off_f32 + k * 256 + row * 64,
                    &conv_wt[k * LFM2_H + chan..k * LFM2_H + chan + 64],
                );
            }
        }
    }
    Ok(())
}

fn compute_lfm2_rope_cs(pos: usize, rope_theta: f32) -> [f32; LFM2_HEAD_DIM] {
    let half = LFM2_HEAD_DIM / 2;
    let theta_scale = rope_theta.powf(-2.0 / LFM2_HEAD_DIM as f32);
    let mut theta = pos as f32;
    let mut cs = [0.0f32; LFM2_HEAD_DIM];
    for i in 0..half {
        let (sin_t, cos_t) = theta.sin_cos();
        cs[i] = cos_t;
        cs[half + i] = sin_t;
        theta *= theta_scale;
    }
    cs
}

fn fill_prefill_rope_cs(pos0: usize, n_valid: usize, rope_theta: f32) -> Vec<f32> {
    let n_slots = if n_valid > LFM2_KV_CHUNK_T {
        LFM2_PREFILL_CHUNK_T
    } else {
        LFM2_KV_CHUNK_T
    };
    let mut out = vec![0.0f32; n_slots * LFM2_HEAD_DIM];
    for t in 0..n_valid {
        let cs = compute_lfm2_rope_cs(pos0 + t, rope_theta);
        out[t * LFM2_HEAD_DIM..(t + 1) * LFM2_HEAD_DIM].copy_from_slice(&cs);
    }
    out
}

enum PrefillSource<'a> {
    Tokens(&'a [u32]),
    Embeddings(&'a [f32], usize),
}

pub struct XdnaLfm2Model {
    cpu: LfmModel,
    host_ffn: bool,
    layers: Vec<Lfm2LayerStatic>,
    npu: Mutex<XdnaLfm2State>,
    decode_error: Mutex<Option<String>>,
    state_torn: AtomicBool,
    session_gate: ModelSessionGate,
    profile_decode: bool,
    _prefill_warm_guard: crate::backend::PrefillWarmGuard,
}

impl Drop for XdnaLfm2Model {
    fn drop(&mut self) {
        let state = match self.npu.get_mut() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        if let Err(e) = state.quiesce_in_flight() {
            tracing::warn!("XdnaLfm2Model::drop: quiesce failed: {e}");
            eprintln!("cera: XdnaLfm2Model::drop: quiesce failed: {e}");
        }
    }
}

impl XdnaLfm2Model {
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let arch = gguf.get_str("general.architecture");
        let is_moe_arch = arch.is_some_and(|a| a.eq_ignore_ascii_case("lfm2moe"));
        let ffn_len = arch.and_then(|a| gguf.get_u32(&format!("{a}.feed_forward_length")));
        let host_ffn_hint = is_moe_arch || ffn_len.is_some_and(|n| n as usize != LFM2_FFN);
        let cpu = if host_ffn_hint {
            LfmModel::from_gguf_with_id_ffn_repack(gguf, context_size, model_id)?
        } else {
            LfmModel::from_gguf_with_id_no_repack(gguf, context_size, model_id)?
        };
        validate_xdna_lfm2_cpu(&cpu)?;
        let cfg = cpu.config();
        let host_ffn = cfg.moe.is_some() || cfg.intermediate_size != LFM2_FFN;
        let n_layers = cfg.n_layers;
        let kernels = Lfm2Kernels::open(host_ffn)?;

        let mut layers = Vec::with_capacity(n_layers);
        let mut layer_bo_w = Vec::with_capacity(n_layers);
        let mut layer_kv_blocks = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let is_conv = matches!(cfg.block_types[l], BlockType::GatedConv);
            let mut bo_w = kernels
                .prefill_conv
                .alloc_bo(LFM2_W_BYTES, 3)
                .map_err(|e| anyhow::anyhow!("layer {l} bo_w alloc failed: {e}"))?;
            {
                let wb = bo_w.as_mut_slice();
                wb.fill(0);
                if is_conv {
                    let (in_proj, out_proj) = cpu
                        .xdna_layer_conv_refs(l)
                        .expect("validated shortconv refs");
                    let conv_wt = cpu.xdna_conv_tap_first(l).expect("validated conv weights");
                    pack_conv_in_proj(&cpu, wb, in_proj)?;
                    pack_weight_stage(&cpu, wb, 384, out_proj, LFM2_H, LFM2_H)?;
                    pack_conv_wt(wb, 512, conv_wt)?;
                    if !host_ffn {
                        let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
                        pack_weight_stage(&cpu, wb, 520, fg, LFM2_FFN, LFM2_H)?;
                        pack_weight_stage(&cpu, wb, 1032, fu, LFM2_FFN, LFM2_H)?;
                        pack_weight_stage(&cpu, wb, 1544, fd, LFM2_H, LFM2_FFN)?;
                    }
                } else {
                    let (wq, wk, wv, wo) =
                        cpu.xdna_layer_attn_refs(l).expect("validated attn refs");
                    pack_weight_stage(&cpu, wb, 0, wq, LFM2_Q_DIM, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 128, wk, LFM2_KV_DIM, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 160, wv, LFM2_KV_DIM, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 192, wo, LFM2_H, LFM2_Q_DIM)?;
                    if !host_ffn {
                        let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
                        pack_weight_stage(&cpu, wb, 320, fg, LFM2_FFN, LFM2_H)?;
                        pack_weight_stage(&cpu, wb, 832, fu, LFM2_FFN, LFM2_H)?;
                        pack_weight_stage(&cpu, wb, 1344, fd, LFM2_H, LFM2_FFN)?;
                    }
                }
            }
            bo_w.sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} bo_w sync failed: {e}"))?;
            layer_bo_w.push(bo_w);

            let kv_bytes = if is_conv {
                LFM2_KVO_BYTES
            } else {
                LFM2_KV_BYTES
            };
            let mut bo_kv = kernels
                .prefill_conv
                .alloc_bo(kv_bytes, 6)
                .map_err(|e| anyhow::anyhow!("layer {l} bo_kv alloc failed: {e}"))?;
            bo_kv.as_mut_slice().fill(0);
            bo_kv
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} bo_kv sync failed: {e}"))?;
            layer_kv_blocks.push(vec![bo_kv]);

            let (rope_theta, _) = cpu.xdna_lfm3_rope_params(l);
            layers.push(Lfm2LayerStatic {
                is_conv,
                rope_theta,
                attn_norm: cpu.attn_norm_weight(l).to_vec(),
                ffn_norm: cpu.ffn_norm_weight(l).to_vec(),
                q_norm: cpu.attn_q_norm_weight(l).map(|s| s.to_vec()),
                k_norm: cpu.attn_k_norm_weight(l).map(|s| s.to_vec()),
            });
        }

        let npu = XdnaLfm2State {
            in_flight: None,
            stalled_runs: Vec::new(),
            kernels,
            layer_bo_w,
            layer_kv_blocks,
            prefill: None,
            decode: None,
            pending_kv: HybridPendingSlot::default(),
            synced_len: 0,
            force_conv_sync: false,
            head_q8_scratch: (Vec::new(), Vec::new()),
            prefill_embs_scratch: (Vec::new(), Vec::new()),
            moe_ffn_input: Vec::new(),
            moe_hidden_scratch: Vec::new(),
            moe_prefill_scratch: XdnaMoePrefillScratch::default(),
        };

        let warm_guard = crate::backend::acquire_prefill_warm_spin();
        crate::backend::warm_prefill_pool(LfmModel::xdna_head_threads());

        Ok(Self {
            cpu,
            host_ffn,
            layers,
            npu: Mutex::new(npu),
            decode_error: Mutex::new(None),
            state_torn: AtomicBool::new(false),
            session_gate: ModelSessionGate::default(),
            profile_decode: crate::backend::cpu_features::profile_decode_enabled(),
            _prefill_warm_guard: warm_guard,
        })
    }

    fn lock_npu(&self) -> MutexGuard<'_, XdnaLfm2State> {
        match self.npu.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                tracing::warn!(
                    "cera::xdna: LFM2 NPU mutex poisoned after prior panic; discarding pending KV and tearing state"
                );
                self.state_torn.store(true, Ordering::SeqCst);
                let mut err_guard = self.decode_error.lock().unwrap_or_else(|e| e.into_inner());
                if err_guard.is_none() {
                    *err_guard =
                        Some("XDNA NPU lock poisoned after a fault; call try_reset_kv".to_string());
                }
                let mut guard = poisoned.into_inner();
                guard.pending_kv.discard();
                self.npu.clear_poison();
                guard
            }
        }
    }

    fn reset_npu_session_state(&self, npu: &mut XdnaLfm2State) -> Result<(), CeraError> {
        if let Err(e) = npu.quiesce_in_flight() {
            self.state_torn.store(true, Ordering::SeqCst);
            return Err(CeraError::Backend(e));
        }
        npu.pending_kv.discard();
        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv {
                npu.layer_kv_blocks[l][0].as_mut_slice()[..LFM2_KVO_BYTES].fill(0);
                npu.layer_kv_blocks[l][0]
                    .sync_range_to_device(LFM2_KVO_BYTES, 0)
                    .map_err(|e| {
                        self.state_torn.store(true, Ordering::SeqCst);
                        CeraError::Backend(format!("layer {l} reset conv bo_kv sync failed: {e}"))
                    })?;
            }
        }
        npu.synced_len = 0;
        npu.force_conv_sync = false;
        if let Some(dec) = npu.decode.as_mut() {
            dec.conv_kv_in_kvo = false;
            dec.ext_plans.clear();
            for p in &mut dec.moe_ext_plans {
                for v in p {
                    v.clear();
                }
            }
        }
        if let Some(pre) = npu.prefill.as_mut() {
            pre.conv_kv_in_kvo = false;
            pre.last_conv_n_valid = None;
            pre.ext_runs = None;
        }
        let _ = take_fault(&self.decode_error);
        self.state_torn.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn record_device_fault(&self, detail: String, vocab_size: usize) -> Vec<f32> {
        self.state_torn.store(true, Ordering::SeqCst);
        let mut guard = self.decode_error.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(detail);
        }
        vec![0.0f32; vocab_size]
    }

    fn reject_input(&self, detail: String, vocab_size: usize) -> Vec<f32> {
        let mut guard = self.decode_error.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(detail);
        }
        vec![0.0f32; vocab_size]
    }

    fn reject_input_err(&self, err: CeraError, vocab_size: usize) -> Vec<f32> {
        self.reject_input(err.to_string(), vocab_size)
    }

    fn validate_state(&self, state: &InferenceState) -> Result<(), String> {
        let cfg = self.cpu.config();
        if state.layers.len() != cfg.n_layers {
            return Err(format!(
                "InferenceState layer count {} != model n_layers {}",
                state.layers.len(),
                cfg.n_layers
            ));
        }
        if state.lora.is_some() {
            return Err("LoRA adapters are not supported by XDNA LFM2".to_string());
        }
        if state.is_compressed() {
            return Err("TurboQuant KV compression is not supported by XDNA LFM2".to_string());
        }
        Ok(())
    }

    fn alloc_prefill_buffers(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
    ) -> Result<Lfm2PrefillBuffers> {
        let mut layer_bo_y = Vec::with_capacity(layers.len());
        let mut layer_bo_kvo = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let mut bo_y = kernels
                .prefill_conv
                .alloc_bo(LFM2_PREFILL_ACT_BYTES, 4)
                .map_err(|e| anyhow::anyhow!("layer {l} prefill bo_y alloc: {e}"))?;
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                init_lfm2_prefill_bo_y_norms(
                    yb,
                    &layer.attn_norm,
                    &layer.ffn_norm,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                );
            }
            bo_y.sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} prefill bo_y sync: {e}"))?;
            layer_bo_y.push(bo_y);

            let mut bo_kvo = kernels
                .prefill_conv
                .alloc_bo(LFM2_KVO_BYTES, 7)
                .map_err(|e| anyhow::anyhow!("layer {l} prefill bo_kvo alloc: {e}"))?;
            bo_kvo.as_mut_slice().fill(0);
            bo_kvo
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} prefill bo_kvo sync: {e}"))?;
            layer_bo_kvo.push(bo_kvo);
        }
        let mut bo_yn = kernels
            .prefill_conv
            .alloc_bo(LFM2_PREFILL_YN_BYTES, 5)
            .map_err(|e| anyhow::anyhow!("prefill bo_yn alloc: {e}"))?;
        bo_yn.as_mut_slice().fill(0);
        bo_yn
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("prefill bo_yn sync: {e}"))?;
        let n_pre_buckets = Q4PREFILL_LFM2_ATTN_BUCKETS.len();
        let make_pre_empty = || (0..n_pre_buckets).map(|_| None).collect::<Vec<_>>();
        let n_layers = layers.len();
        let make_pre_moe_empty = || (0..n_layers).map(|_| None).collect::<Vec<_>>();
        Ok(Lfm2PrefillBuffers {
            layer_bo_y,
            layer_bo_kvo,
            bo_yn,
            conv_kv_in_kvo: false,
            last_conv_n_valid: None,
            bucket_plans: [make_pre_empty(), make_pre_empty()],
            ext_runs: None,
            moe_conv_runs: [make_pre_moe_empty(), make_pre_moe_empty()],
            moe_attn_runs: (0..n_pre_buckets).map(|_| make_pre_moe_empty()).collect(),
        })
    }

    fn alloc_decode_buffers(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
    ) -> Result<Lfm2DecodeBuffers> {
        let mut layer_bo_y = Vec::with_capacity(layers.len());
        let mut layer_bo_kvo = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let mut bo_y = kernels
                .dec_conv
                .alloc_bo(LFM2_DEC_ACT_BYTES, 4)
                .map_err(|e| anyhow::anyhow!("layer {l} decode bo_y alloc: {e}"))?;
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                init_lfm2_bo_y_norms(
                    yb,
                    LFM2_DEC_SLOT_NORM_0,
                    LFM2_DEC_SLOT_NORM_1,
                    &layer.attn_norm,
                    &layer.ffn_norm,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                );
            }
            bo_y.sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} decode bo_y sync: {e}"))?;
            layer_bo_y.push(bo_y);

            let mut bo_kvo = kernels
                .dec_conv
                .alloc_bo(LFM2_KVO_BYTES, 7)
                .map_err(|e| anyhow::anyhow!("layer {l} decode bo_kvo alloc: {e}"))?;
            bo_kvo.as_mut_slice().fill(0);
            bo_kvo
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} decode bo_kvo sync: {e}"))?;
            layer_bo_kvo.push(bo_kvo);
        }
        let mut bo_yn = kernels
            .dec_conv
            .alloc_bo(LFM2_DEC_YN_BYTES, 5)
            .map_err(|e| anyhow::anyhow!("decode bo_yn alloc: {e}"))?;
        bo_yn.as_mut_slice().fill(0);
        bo_yn
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("decode bo_yn sync: {e}"))?;
        let n_dec_buckets = Q4LAYER_LFM2_KV_BUCKETS.len();
        let make_dec_empty = || (0..n_dec_buckets).map(|_| None).collect::<Vec<_>>();
        let n_layers = layers.len();
        let make_dec_moe_empty = || (0..n_layers).map(|_| None).collect::<Vec<_>>();
        Ok(Lfm2DecodeBuffers {
            layer_bo_y,
            layer_bo_kvo,
            bo_yn,
            conv_kv_in_kvo: false,
            bucket_plans: [make_dec_empty(), make_dec_empty()],
            ext_plans: Lfm2ExtPlanLru::new(2),
            moe_conv_runs: [make_dec_moe_empty(), make_dec_moe_empty()],
            moe_attn_runs: (0..n_dec_buckets).map(|_| make_dec_moe_empty()).collect(),
            moe_ext_plans: [
                (0..n_layers).map(|_| Vec::new()).collect(),
                (0..n_layers).map(|_| Vec::new()).collect(),
            ],
        })
    }

    fn ensure_paged_block(
        kernels: &Lfm2Kernels,
        blocks: &mut Vec<XdnaBo>,
        blk_idx: usize,
        layer: usize,
    ) -> Result<(), String> {
        while blocks.len() <= blk_idx {
            let mut bo = kernels
                .prefill_conv
                .alloc_bo(LFM2_KV_BYTES, 6)
                .map_err(|e| format!("layer {layer} alloc KV block {}: {e}", blocks.len()))?;
            bo.as_mut_slice().fill(0);
            bo.sync_to_device()
                .map_err(|e| format!("layer {layer} zero-sync KV block {}: {e}", blocks.len()))?;
            blocks.push(bo);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_decode_bucket_plan(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        dec_layer_bo_y: &[XdnaBo],
        dec_layer_bo_kvo: &[XdnaBo],
        dec_bo_yn: &XdnaBo,
        attn_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm2DecodeBucketPlan, String> {
        kernels.dec_attn(attn_bidx)?;
        let n_layers = layers.len();
        let mut runs = Vec::with_capacity(n_layers);
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &dec_layer_bo_y[l + 1]
            } else {
                dec_bo_yn
            };
            let run = if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&dec_layer_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &dec_layer_bo_kvo[l])
                };
                kernels.dec_conv.create_run(&[
                    &layer_bo_w[l],
                    &dec_layer_bo_y[l],
                    y_next,
                    kv_in,
                    kv_out,
                ])
            } else {
                kernels.dec_attn(attn_bidx)?.create_run(&[
                    &layer_bo_w[l],
                    &dec_layer_bo_y[l],
                    y_next,
                    &layer_kv_blocks[l][0],
                    &dec_layer_bo_kvo[l],
                ])
            }
            .map_err(|e| format!("layer {l} dec create_run: {e}"))?;
            runs.push(run);
        }
        let mut runlist = kernels
            .dec_conv
            .create_runlist()
            .map_err(|e| format!("dec create_runlist: {e}"))?;
        for (l, run) in runs.iter().enumerate() {
            runlist
                .add(run)
                .map_err(|e| format!("dec runlist add layer {l}: {e}"))?;
        }
        Ok(Lfm2DecodeBucketPlan {
            runlist,
            _runs: runs,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_decode_ext_plan(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        dec_layer_bo_y: &[XdnaBo],
        dec_layer_bo_kvo: &[XdnaBo],
        dec_bo_yn: &XdnaBo,
        num_full: usize,
        fin_blk: usize,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm2DecodeBucketPlan, String> {
        kernels.dec_phase_a()?;
        kernels.dec_fa_part()?;
        kernels.dec_fa_fin(fin_bidx)?;
        let n_layers = layers.len();
        let mut runs = Vec::new();
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &dec_layer_bo_y[l + 1]
            } else {
                dec_bo_yn
            };
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&dec_layer_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &dec_layer_bo_kvo[l])
                };
                runs.push(
                    kernels
                        .dec_conv
                        .create_run(&[&layer_bo_w[l], &dec_layer_bo_y[l], y_next, kv_in, kv_out])
                        .map_err(|e| format!("layer {l} ext dec conv create_run: {e}"))?,
                );
            } else {
                runs.push(
                    kernels
                        .dec_phase_a()?
                        .create_run(&[
                            &layer_bo_w[l],
                            &dec_layer_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &dec_layer_bo_kvo[l],
                        ])
                        .map_err(|e| format!("layer {l} ext dec phase A create_run: {e}"))?,
                );
                let k_part = kernels.dec_fa_part()?;
                for (blk, layer_kv) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                    runs.push(
                        k_part
                            .create_run(&[
                                &layer_bo_w[l],
                                &dec_layer_bo_y[l],
                                y_next,
                                layer_kv,
                                &dec_layer_bo_kvo[l],
                            ])
                            .map_err(|e| {
                                format!("layer {l} blk {blk} ext dec fa_part create_run: {e}")
                            })?,
                    );
                }
                runs.push(
                    kernels
                        .dec_fa_fin(fin_bidx)?
                        .create_run(&[
                            &layer_bo_w[l],
                            &dec_layer_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][fin_blk],
                            &dec_layer_bo_kvo[l],
                        ])
                        .map_err(|e| format!("layer {l} ext dec fa_fin create_run: {e}"))?,
                );
            }
        }
        let mut runlist = kernels
            .dec_conv
            .create_runlist()
            .map_err(|e| format!("ext dec create_runlist: {e}"))?;
        for (idx, run) in runs.iter().enumerate() {
            runlist
                .add(run)
                .map_err(|e| format!("ext dec runlist add step {idx}: {e}"))?;
        }
        Ok(Lfm2DecodeBucketPlan {
            runlist,
            _runs: runs,
        })
    }

    fn assemble_prefill_runlist(
        root: &XdnaKernel,
        runs: Vec<XdnaRun>,
        tag: &str,
    ) -> Result<Lfm2PrefillChain, String> {
        let mut runlist = root
            .create_runlist()
            .map_err(|e| format!("{tag} create_runlist: {e}"))?;
        for (i, run) in runs.iter().enumerate() {
            runlist
                .add(run)
                .map_err(|e| format!("{tag} runlist add op {i}: {e}"))?;
        }
        Ok(Lfm2PrefillChain {
            _runs: runs,
            runlist,
        })
    }

    fn build_prefill_chain(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        pre: &Lfm2PrefillBuffers,
        attn_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm2PrefillChain, String> {
        let n_layers = layers.len();
        let mut runs = Vec::with_capacity(n_layers);
        kernels.prefill_attn(attn_bidx)?;

        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &pre.layer_bo_y[l + 1]
            } else {
                &pre.bo_yn
            };
            let run = if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&pre.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &pre.layer_bo_kvo[l])
                };
                kernels.prefill_conv.create_run(&[
                    &layer_bo_w[l],
                    &pre.layer_bo_y[l],
                    y_next,
                    kv_in,
                    kv_out,
                ])
            } else {
                kernels.prefill_attn(attn_bidx)?.create_run(&[
                    &layer_bo_w[l],
                    &pre.layer_bo_y[l],
                    y_next,
                    &layer_kv_blocks[l][0],
                    &pre.layer_bo_kvo[l],
                ])
            }
            .map_err(|e| format!("layer {l} prefill create_run: {e}"))?;
            runs.push(run);
        }

        Self::assemble_prefill_runlist(&kernels.prefill_conv, runs, "prefill")
    }

    #[allow(clippy::too_many_arguments)]
    fn append_prefill_ext_attn_runs(
        kernels: &mut Lfm2Kernels,
        l: usize,
        layer_w: &XdnaBo,
        layer_y: &XdnaBo,
        y_next: &XdnaBo,
        kv_blocks: &[XdnaBo],
        layer_kvo: &XdnaBo,
        cur_blk: usize,
        fin_bidx: usize,
        runs: &mut Vec<XdnaRun>,
    ) -> Result<(), String> {
        runs.push(
            kernels
                .prefill_phase_a()?
                .create_run(&[layer_w, layer_y, y_next, &kv_blocks[0], layer_kvo])
                .map_err(|e| format!("layer {l} ext phase_a create_run: {e}"))?,
        );
        for (b, kv_blk) in kv_blocks.iter().enumerate().take(cur_blk) {
            let k_part = if b == 0 {
                kernels.prefill_fa_part0()?
            } else {
                kernels.prefill_fa_part()?
            };
            runs.push(
                k_part
                    .create_run(&[layer_w, layer_y, y_next, kv_blk, layer_kvo])
                    .map_err(|e| format!("layer {l} ext fa_part blk={b} create_run: {e}"))?,
            );
        }
        runs.push(
            kernels
                .prefill_fa_fin(fin_bidx)?
                .create_run(&[layer_w, layer_y, y_next, &kv_blocks[cur_blk], layer_kvo])
                .map_err(|e| format!("layer {l} ext fa_fin create_run: {e}"))?,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_prefill_ext_chain(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        pre: &Lfm2PrefillBuffers,
        cur_blk: usize,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm2PrefillChain, String> {
        let n_layers = layers.len();
        kernels.prefill_ext_conv()?;
        kernels.prefill_phase_a()?;
        kernels.prefill_fa_part0()?;
        if cur_blk > 1 {
            kernels.prefill_fa_part()?;
        }
        kernels.prefill_fa_fin(fin_bidx)?;

        let mut runs = Vec::with_capacity(n_layers * (2 + cur_blk));
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &pre.layer_bo_y[l + 1]
            } else {
                &pre.bo_yn
            };
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&pre.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &pre.layer_bo_kvo[l])
                };
                runs.push(
                    kernels
                        .prefill_ext_conv()?
                        .create_run(&[&layer_bo_w[l], &pre.layer_bo_y[l], y_next, kv_in, kv_out])
                        .map_err(|e| format!("layer {l} ext conv create_run: {e}"))?,
                );
            } else {
                Self::append_prefill_ext_attn_runs(
                    kernels,
                    l,
                    &layer_bo_w[l],
                    &pre.layer_bo_y[l],
                    y_next,
                    &layer_kv_blocks[l],
                    &pre.layer_bo_kvo[l],
                    cur_blk,
                    fin_bidx,
                    &mut runs,
                )?;
            }
        }

        let k_conv = kernels.prefill_ext_conv.as_ref().expect("loaded ext conv");
        Self::assemble_prefill_runlist(k_conv, runs, "prefill_ext")
    }

    #[allow(clippy::too_many_arguments)]
    fn build_prefill_ext_attn_layer_chain(
        kernels: &mut Lfm2Kernels,
        l: usize,
        layer_w: &XdnaBo,
        layer_y: &XdnaBo,
        y_next: &XdnaBo,
        kv_blocks: &[XdnaBo],
        layer_kvo: &XdnaBo,
        cur_blk: usize,
        fin_bidx: usize,
    ) -> Result<Lfm2PrefillChain, String> {
        kernels.prefill_phase_a()?;
        kernels.prefill_fa_part0()?;
        if cur_blk > 1 {
            kernels.prefill_fa_part()?;
        }
        kernels.prefill_fa_fin(fin_bidx)?;

        let mut runs = Vec::with_capacity(2 + cur_blk);
        Self::append_prefill_ext_attn_runs(
            kernels, l, layer_w, layer_y, y_next, kv_blocks, layer_kvo, cur_blk, fin_bidx,
            &mut runs,
        )?;
        let k_a = kernels.prefill_phase_a.as_ref().expect("loaded phase_a");
        Self::assemble_prefill_runlist(k_a, runs, "prefill_ext_attn")
    }

    fn resolve_prefill_rope<'a>(
        cached_rope: &'a mut Option<(f32, Vec<f32>)>,
        prefetched_rope: &mut Option<(usize, usize, f32, Vec<f32>)>,
        pos0: usize,
        n_valid: usize,
        rope_theta: f32,
    ) -> &'a [f32] {
        let cache_hit = cached_rope
            .as_ref()
            .is_some_and(|(theta, _)| *theta == rope_theta);
        if !cache_hit {
            let cs = match prefetched_rope.take() {
                Some((p_pos0, p_nv, p_theta, p_cs))
                    if p_pos0 == pos0 && p_nv == n_valid && p_theta == rope_theta =>
                {
                    p_cs
                }
                _ => fill_prefill_rope_cs(pos0, n_valid, rope_theta),
            };
            *cached_rope = Some((rope_theta, cs));
        }
        &cached_rope.as_ref().expect("cached rope").1
    }

    #[allow(clippy::too_many_arguments)]
    fn sync_ext_attn_layer_params(
        layer: &Lfm2LayerStatic,
        l: usize,
        c: usize,
        bo_y: &mut XdnaBo,
        cached_rope: &mut Option<(f32, Vec<f32>)>,
        prefetched_rope: &mut Option<(usize, usize, f32, Vec<f32>)>,
        pos0: usize,
        n_valid: usize,
        eps: f32,
        cur_blk: usize,
        fin_cap: usize,
        fin_lmode: i32,
    ) -> Result<(), String> {
        {
            let rope_cs = Self::resolve_prefill_rope(
                cached_rope,
                prefetched_rope,
                pos0,
                n_valid,
                layer.rope_theta,
            );
            let yb = bo_y.as_mut_slice();
            write_lfm2_prefill_step_params(
                yb,
                pos0,
                n_valid,
                0,
                pos0,
                true,
                0,
                2,
                eps,
                Some(rope_cs),
            );
            write_lfm3_params_at(
                yb,
                LFM2_PREFILL_SLOT_EXT_PARAMS_0 + 2048,
                pos0,
                n_valid,
                LFM2_MAX_KV_CHUNKS,
                pos0,
                true,
                0,
                3,
                eps,
                0,
                false,
                false,
            );
            if cur_blk == 2 {
                write_lfm3_params_at(
                    yb,
                    LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048,
                    pos0,
                    n_valid,
                    LFM2_MAX_KV_CHUNKS,
                    pos0,
                    true,
                    0,
                    4,
                    eps,
                    LFM2_MAX_KV_CHUNKS,
                    true,
                    false,
                );
            }
            write_lfm3_params_at(
                yb,
                LFM2_PREFILL_SLOT_EXT_PARAMS_2 + 2048,
                pos0,
                n_valid,
                fin_cap,
                pos0,
                true,
                0,
                fin_lmode,
                eps,
                cur_blk * LFM2_MAX_KV_CHUNKS,
                true,
                false,
            );
        }
        let sync_start_f32 = LFM2_PREFILL_SLOT_NORM_0 + 2048;
        let rope_end_f32 = if n_valid > LFM2_KV_CHUNK_T {
            LFM2_PREFILL_SLOT_ROPE_PKT_1 + (n_valid - LFM2_KV_CHUNK_T) * LFM2_HEAD_DIM
        } else {
            LFM2_PREFILL_SLOT_ROPE_PKT_0 + n_valid * LFM2_HEAD_DIM
        };
        bo_y.sync_range_to_device((rope_end_f32 - sync_start_f32) * 4, sync_start_f32 * 4)
            .map_err(|e| format!("prefill >32K c={c} layer {l} phase_a params+rope sync: {e}"))?;
        bo_y.sync_range_to_device(64, (LFM2_PREFILL_SLOT_EXT_PARAMS_0 + 2048) * 4)
            .map_err(|e| format!("prefill >32K c={c} layer {l} fa_part0 params sync: {e}"))?;
        if cur_blk == 2 {
            bo_y.sync_range_to_device(64, (LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048) * 4)
                .map_err(|e| format!("prefill >32K c={c} layer {l} fa_part params sync: {e}"))?;
        }
        bo_y.sync_range_to_device(64, (LFM2_PREFILL_SLOT_EXT_PARAMS_2 + 2048) * 4)
            .map_err(|e| format!("prefill >32K c={c} layer {l} fa_fin params sync: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_prefill_chain_with_overlap(
        &self,
        chain: &Lfm2PrefillChain,
        in_flight: &mut Option<InFlightLfm2Run>,
        active_run: InFlightLfm2Run,
        next_rope_spec: Option<(usize, usize, f32)>,
        src: &PrefillSource<'_>,
        c: usize,
        n_chunks: usize,
        n_tokens: usize,
        pending_attn_mirrors: &mut Vec<Lfm2PrefillAttnMirrorEntry>,
        layer_kv_blocks: &mut [Vec<XdnaBo>],
        layers_state: &mut [LayerState],
        use_f16: bool,
        prefetched_rope: &mut Option<(usize, usize, f32, Vec<f32>)>,
        prefill_embs_scratch: &mut (Vec<f32>, Vec<f32>),
    ) -> Result<(bool, bool), String> {
        let mut next_tok_ok = true;
        let mut swapped_next_embs = false;
        *in_flight = Some(active_run);
        let exec_res = match chain.runlist.execute() {
            Ok(()) => {
                Self::flush_pending_attn_mirrors(
                    pending_attn_mirrors,
                    layer_kv_blocks,
                    layers_state,
                    use_f16,
                );
                if let Some((np0, nnv, th)) = next_rope_spec {
                    *prefetched_rope = Some((np0, nnv, th, fill_prefill_rope_cs(np0, nnv, th)));
                }
                if let PrefillSource::Tokens(toks) = src
                    && c + 1 < n_chunks
                {
                    let next_off = (c + 1) * LFM2_PREFILL_CHUNK_T;
                    let next_nv = (n_tokens - next_off).min(LFM2_PREFILL_CHUNK_T);
                    next_tok_ok = self.cpu.embed_token_rows_into(
                        &toks[next_off..next_off + next_nv],
                        &mut prefill_embs_scratch.1[..next_nv * LFM2_H],
                    );
                    swapped_next_embs = next_tok_ok;
                }
                let w = chain.runlist.wait(QUIESCE_TIMEOUT_MS);
                if w.is_ok() {
                    *in_flight = None;
                }
                w
            }
            Err(e) => {
                *in_flight = None;
                Err(e)
            }
        };
        if let Err(e) = exec_res {
            Self::flush_pending_attn_mirrors(
                pending_attn_mirrors,
                layer_kv_blocks,
                layers_state,
                use_f16,
            );
            return Err(format!("prefill c={c} runlist exec: {e}"));
        }
        Ok((next_tok_ok, swapped_next_embs))
    }

    #[allow(clippy::too_many_arguments)]
    fn sync_and_save_prefill_kvo_outputs(
        kernels: &Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
        pre_bo_kvo: &mut [XdnaBo],
        layer_kv_blocks: &mut [Vec<XdnaBo>],
        layers_state: &mut [LayerState],
        pending_attn_mirrors: &mut Vec<Lfm2PrefillAttnMirrorEntry>,
        conv_kv_in_kvo: bool,
        is_last_chunk: bool,
        pos0: usize,
        n_valid: usize,
        attn_kvo_sync_bytes: usize,
        use_f16: bool,
        c: usize,
    ) -> Result<(), String> {
        for (l, layer) in layers.iter().enumerate() {
            if layer.is_conv {
                if is_last_chunk {
                    let kvo_written = if conv_kv_in_kvo {
                        &mut layer_kv_blocks[l][0]
                    } else {
                        &mut pre_bo_kvo[l]
                    };
                    if let Err(e) = kvo_written.sync_range_from_device(LFM2_KVO_BYTES, 0) {
                        Self::flush_pending_attn_mirrors(
                            pending_attn_mirrors,
                            layer_kv_blocks,
                            layers_state,
                            use_f16,
                        );
                        return Err(format!("prefill c={c} layer {l} conv kvo readback: {e}"));
                    }
                    Self::save_conv_kvo_to_state(
                        kvo_written,
                        &mut layers_state[l],
                        pos0 + n_valid,
                        true,
                    );
                }
            } else {
                if let Err(e) = pre_bo_kvo[l].sync_range_from_device(attn_kvo_sync_bytes, 0) {
                    Self::flush_pending_attn_mirrors(
                        pending_attn_mirrors,
                        layer_kv_blocks,
                        layers_state,
                        use_f16,
                    );
                    return Err(format!("prefill c={c} layer {l} attn kvo readback: {e}"));
                }
                match Self::save_attn_prefill_kvo_to_bo_kv(
                    kernels,
                    &pre_bo_kvo[l],
                    &mut layer_kv_blocks[l],
                    l,
                    pos0,
                    n_valid,
                ) {
                    Ok(entry) => {
                        if is_last_chunk {
                            Self::mirror_attn_prefill_bo_kv_to_state(
                                &layer_kv_blocks[l],
                                &mut layers_state[l],
                                &entry,
                                use_f16,
                            );
                        } else {
                            pending_attn_mirrors.push(entry);
                        }
                    }
                    Err(e) => {
                        Self::flush_pending_attn_mirrors(
                            pending_attn_mirrors,
                            layer_kv_blocks,
                            layers_state,
                            use_f16,
                        );
                        return Err(format!("prefill c={c} layer {l} attn save: {e}"));
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_decode_ext_layer_plan(
        kernels: &mut Lfm2Kernels,
        l: usize,
        layer_bo_w: &XdnaBo,
        layer_bo_y: &XdnaBo,
        y_next: &XdnaBo,
        layer_bo_kvo: &XdnaBo,
        layer_kv_blocks: &[XdnaBo],
        cur_blk: usize,
        fin_bidx: usize,
    ) -> Result<Lfm2DecodeBucketPlan, String> {
        let mut runs = Vec::with_capacity(2 + cur_blk);
        runs.push(
            kernels
                .dec_phase_a()?
                .create_run(&[
                    layer_bo_w,
                    layer_bo_y,
                    y_next,
                    &layer_kv_blocks[0],
                    layer_bo_kvo,
                ])
                .map_err(|e| format!("layer {l} ext dec phase_a create_run: {e}"))?,
        );
        for (blk, layer_kv) in layer_kv_blocks.iter().enumerate().take(cur_blk) {
            runs.push(
                kernels
                    .dec_fa_part()?
                    .create_run(&[layer_bo_w, layer_bo_y, y_next, layer_kv, layer_bo_kvo])
                    .map_err(|e| format!("layer {l} blk {blk} ext dec fa_part create_run: {e}"))?,
            );
        }
        runs.push(
            kernels
                .dec_fa_fin(fin_bidx)?
                .create_run(&[
                    layer_bo_w,
                    layer_bo_y,
                    y_next,
                    &layer_kv_blocks[cur_blk],
                    layer_bo_kvo,
                ])
                .map_err(|e| format!("layer {l} ext dec fa_fin create_run: {e}"))?,
        );
        let mut runlist = kernels
            .dec_conv
            .create_runlist()
            .map_err(|e| format!("layer {l} ext dec create_runlist: {e}"))?;
        for (idx, run) in runs.iter().enumerate() {
            runlist
                .add(run)
                .map_err(|e| format!("layer {l} ext dec runlist add step {idx}: {e}"))?;
        }
        Ok(Lfm2DecodeBucketPlan {
            _runs: runs,
            runlist,
        })
    }

    /// Execute `>=64K` (`cur_blk >= 2`) extended decode for one attention layer with
    /// distinct per-block `kv_start_chunk` and `has_fa_state` parameters.
    #[allow(clippy::too_many_arguments)]
    fn run_decode_ext_attn_multi_block_layer(
        kernels: &mut Lfm2Kernels,
        l: usize,
        layer_w: &XdnaBo,
        layer_y: &mut XdnaBo,
        y_next: &XdnaBo,
        layer_kvo: &XdnaBo,
        kv_blocks: &[XdnaBo],
        stalled_runs: &mut Vec<XdnaRun>,
        pos: usize,
        eps: f32,
        cur_blk: usize,
        fin_bidx: usize,
    ) -> Result<(), String> {
        let run_a = kernels
            .dec_phase_a()?
            .create_run(&[layer_w, layer_y, y_next, &kv_blocks[0], layer_kvo])
            .map_err(|e| format!("layer {l} ext dec phase_a create_run: {e}"))?;
        exec_owned_run(
            stalled_runs,
            run_a,
            &format!("layer {l} ext dec phase_a exec"),
        )?;

        let k_part = kernels.dec_fa_part()?;
        for (blk, kv_blk) in kv_blocks.iter().enumerate().take(cur_blk) {
            write_lfm3_params_at(
                layer_y.as_mut_slice(),
                LFM2_DEC_SLOT_NORM_1 + 4096,
                pos,
                1,
                LFM2_DEC_MAX_KV_PKTS,
                pos,
                true,
                0,
                3,
                eps,
                blk * LFM2_DEC_MAX_KV_PKTS,
                blk > 0,
                true,
            );
            layer_y
                .sync_range_to_device(64, (LFM2_DEC_SLOT_NORM_1 + 4096) * 4)
                .map_err(|e| format!("layer {l} blk {blk} ext dec fa_part params sync: {e}"))?;
            let run_p = k_part
                .create_run(&[layer_w, layer_y, y_next, kv_blk, layer_kvo])
                .map_err(|e| format!("layer {l} blk {blk} ext dec fa_part create_run: {e}"))?;
            exec_owned_run(
                stalled_runs,
                run_p,
                &format!("layer {l} blk {blk} ext dec fa_part exec"),
            )?;
        }

        let run_f = kernels
            .dec_fa_fin(fin_bidx)?
            .create_run(&[layer_w, layer_y, y_next, &kv_blocks[cur_blk], layer_kvo])
            .map_err(|e| format!("layer {l} ext dec fa_fin create_run: {e}"))?;
        exec_owned_run(
            stalled_runs,
            run_f,
            &format!("layer {l} ext dec fa_fin exec"),
        )
    }

    /// Execute `>=96K` (`cur_blk >= 3`) extended prefill for one attention layer with
    /// distinct per-block `kv_start_chunk` parameters across `b in 1..cur_blk`.
    #[allow(clippy::too_many_arguments)]
    fn run_prefill_ext_attn_multi_block_layer(
        kernels: &mut Lfm2Kernels,
        l: usize,
        c: usize,
        layer_w: &XdnaBo,
        layer_y: &mut XdnaBo,
        y_next: &XdnaBo,
        kv_blocks: &[XdnaBo],
        layer_kvo: &XdnaBo,
        stalled_runs: &mut Vec<XdnaRun>,
        pos0: usize,
        n_valid: usize,
        eps: f32,
        cur_blk: usize,
        fin_bidx: usize,
    ) -> Result<(), String> {
        let run_a = kernels
            .prefill_phase_a()?
            .create_run(&[layer_w, layer_y, y_next, &kv_blocks[0], layer_kvo])
            .map_err(|e| format!("prefill >96K c={c} layer {l} phase_a create_run: {e}"))?;
        exec_owned_run(
            stalled_runs,
            run_a,
            &format!("prefill >96K c={c} layer {l} phase_a exec"),
        )?;

        let run_p0 = kernels
            .prefill_fa_part0()?
            .create_run(&[layer_w, layer_y, y_next, &kv_blocks[0], layer_kvo])
            .map_err(|e| format!("prefill >96K c={c} layer {l} blk=0 fa_part0 create_run: {e}"))?;
        exec_owned_run(
            stalled_runs,
            run_p0,
            &format!("prefill >96K c={c} layer {l} blk=0 fa_part0 exec"),
        )?;

        let k_part = kernels.prefill_fa_part()?;
        for (b, kv_blk) in kv_blocks.iter().enumerate().take(cur_blk).skip(1) {
            write_lfm3_params_at(
                layer_y.as_mut_slice(),
                LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048,
                pos0,
                n_valid,
                LFM2_MAX_KV_CHUNKS,
                pos0,
                true,
                0,
                4,
                eps,
                b * LFM2_MAX_KV_CHUNKS,
                true,
                false,
            );
            layer_y
                .sync_range_to_device(64, (LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048) * 4)
                .map_err(|e| {
                    format!("prefill >96K c={c} layer {l} blk={b} fa_part params sync: {e}")
                })?;
            let run_p = k_part
                .create_run(&[layer_w, layer_y, y_next, kv_blk, layer_kvo])
                .map_err(|e| {
                    format!("prefill >96K c={c} layer {l} blk={b} fa_part create_run: {e}")
                })?;
            exec_owned_run(
                stalled_runs,
                run_p,
                &format!("prefill >96K c={c} layer {l} blk={b} fa_part exec"),
            )?;
        }

        let run_f = kernels
            .prefill_fa_fin(fin_bidx)?
            .create_run(&[layer_w, layer_y, y_next, &kv_blocks[cur_blk], layer_kvo])
            .map_err(|e| format!("prefill >96K c={c} layer {l} fa_fin create_run: {e}"))?;
        exec_owned_run(
            stalled_runs,
            run_f,
            &format!("prefill >96K c={c} layer {l} fa_fin exec"),
        )
    }

    fn validate_cpu_state_for_sync(
        &self,
        state: &InferenceState,
        pending: Option<&crate::model::xdna_llama::PendingHybridCpuKv>,
        target_pos: usize,
        use_f16: bool,
    ) -> Result<(), String> {
        self.validate_state(state)?;
        if target_pos != state.seq_len {
            return Err(format!(
                "target_pos={target_pos} does not match state.seq_len={}",
                state.seq_len
            ));
        }
        let effective_cached_pos = match (target_pos, pending) {
            (t, Some(p)) if t > 0 && p.pending_pos() + 1 == t => p.pending_pos(),
            (t, _) => t,
        };
        let needed_kv = effective_cached_pos
            .checked_mul(LFM2_KV_DIM)
            .ok_or_else(|| "target_pos overflows LFM2 KV cache length".to_string())?;
        let expected_conv_len = LFM2_D_CONV * LFM2_H;
        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv {
                match &state.layers[l] {
                    LayerState::Conv { buffer, .. } => {
                        if target_pos > 0 && buffer.len() < expected_conv_len {
                            return Err(format!(
                                "layer {l} conv state len {} < {expected_conv_len}",
                                buffer.len()
                            ));
                        }
                    }
                    _ => return Err(format!("layer {l} expected Conv state")),
                }
            } else {
                match &state.layers[l] {
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } => {
                        let is_f16 = key_cache.is_empty() && !key_cache_f16.is_empty();
                        let (k_len, v_len) = if is_f16 || (use_f16 && key_cache.is_empty()) {
                            (key_cache_f16.len(), value_cache_f16.len())
                        } else {
                            (key_cache.len(), value_cache.len())
                        };
                        if k_len < needed_kv || v_len < needed_kv {
                            return Err(format!(
                                "layer {l} KV cache too small for target_pos={target_pos}: {k_len}/{v_len} < {needed_kv}"
                            ));
                        }
                    }
                    _ => return Err(format!("layer {l} expected Attention state")),
                }
            }
        }
        Ok(())
    }

    fn sync_and_write_prefill_hidden(
        bo_y: &mut XdnaBo,
        hidden: &[f32],
        n_valid: usize,
        label: &str,
    ) -> Result<(), String> {
        write_le_f32_slice(bo_y.as_mut_slice(), 0, &hidden[..n_valid * LFM2_H]);
        if n_valid < LFM2_PREFILL_CHUNK_T {
            let start_b = n_valid * LFM2_H * 4;
            let end_b = LFM2_PREFILL_CHUNK_T * LFM2_H * 4;
            bo_y.as_mut_slice()[start_b..end_b].fill(0);
        }
        bo_y.sync_range_to_device(LFM2_PREFILL_YN_BYTES, 0)
            .map_err(|e| format!("{label}: {e}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_moe_prefill_layer(
        &self,
        bo_yn: &mut XdnaBo,
        layer_bo_y_next: Option<&mut XdnaBo>,
        moe_hidden_scratch: &mut Vec<f32>,
        moe_prefill_scratch: &mut XdnaMoePrefillScratch,
        l: usize,
        n_valid: usize,
        state: &mut InferenceState,
    ) -> Result<(), String> {
        let need_elems = n_valid * LFM2_H;
        bo_yn
            .sync_range_from_device(need_elems * 4, 0)
            .map_err(|e| format!("prefill layer {l} bo_yn readback: {e}"))?;
        moe_hidden_scratch.resize(need_elems, 0.0);
        read_le_f32_slice(bo_yn.as_slice(), 0, &mut moe_hidden_scratch[..need_elems]);
        self.cpu.xdna_prefill_ffn_chunk(
            l,
            &mut moe_hidden_scratch[..need_elems],
            n_valid,
            moe_prefill_scratch,
            state,
        );
        if let Some(next_bo_y) = layer_bo_y_next {
            Self::sync_and_write_prefill_hidden(
                next_bo_y,
                moe_hidden_scratch,
                n_valid,
                &format!("prefill layer {} hidden sync", l + 1),
            )?;
        }
        Ok(())
    }

    fn sync_cpu_state_to_npu(
        &self,
        npu: &mut XdnaLfm2State,
        state: &mut InferenceState,
        target_pos: usize,
        use_f16: bool,
    ) -> Result<(), SyncErr> {
        self.validate_cpu_state_for_sync(state, npu.pending_kv.as_ref(), target_pos, use_f16)
            .map_err(SyncErr::Input)?;
        // Preserve pending CPU KV across no-op sync checks so callers can still overlap its
        // flush with async NPU execution when no host-to-device resync is required.
        if !npu.force_conv_sync
            && target_pos == npu.synced_len
            && (target_pos == 0
                || npu
                    .pending_kv
                    .as_ref()
                    .is_none_or(|p| p.matches_state(state)))
        {
            if target_pos == 0 {
                npu.pending_kv.discard();
            }
            return Ok(());
        }
        npu.quiesce_in_flight().map_err(SyncErr::Device)?;
        if target_pos == 0 {
            npu.pending_kv.discard();
        } else if !npu.pending_kv.flush_into(state) {
            npu.synced_len = 0;
            npu.force_conv_sync = true;
            // Re-verify post-discard state because KV still held `pending_pos < target_pos` rows.
            self.validate_cpu_state_for_sync(state, None, target_pos, use_f16)
                .map_err(SyncErr::Input)?;
        }
        let sync_conv = npu.force_conv_sync || npu.synced_len != target_pos;
        let sync_attn = npu.synced_len != target_pos;
        if !sync_conv && !sync_attn {
            return Ok(());
        }
        // Invalidate NPU sync markers before mutating device buffers so a mid-sync
        // alloc/DMA failure never leaves `synced_len` matching `target_pos`.
        npu.synced_len = 0;
        npu.force_conv_sync = true;
        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv && sync_conv {
                match &state.layers[l] {
                    LayerState::Conv { buffer, .. } => {
                        let kv_buf = npu.layer_kv_blocks[l][0].as_mut_slice();
                        kv_buf[..LFM2_KVO_BYTES].fill(0);
                        for col in 0..8 {
                            let col_off = col * TILE_BYTES;
                            for k in 0..LFM2_D_CONV {
                                let src_off = k * LFM2_H + col * 256;
                                if src_off + 256 <= buffer.len() {
                                    write_le_f32_slice(
                                        kv_buf,
                                        (col_off / 4) + k * 256,
                                        &buffer[src_off..src_off + 256],
                                    );
                                }
                            }
                        }
                        npu.layer_kv_blocks[l][0]
                            .sync_range_to_device(LFM2_KVO_BYTES, 0)
                            .map_err(|e| {
                                SyncErr::Device(format!("layer {l} conv bo_kv sync failed: {e}"))
                            })?;
                    }
                    _ => return Err(SyncErr::Input(format!("layer {l} expected Conv state"))),
                }
            } else if !layer.is_conv && sync_attn {
                match &state.layers[l] {
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } => {
                        let n_blocks = if target_pos == 0 {
                            1
                        } else {
                            ((target_pos - 1) / LFM2_BLOCK_TOKENS) + 1
                        };
                        Self::ensure_paged_block(
                            &npu.kernels,
                            &mut npu.layer_kv_blocks[l],
                            n_blocks - 1,
                            l,
                        )
                        .map_err(SyncErr::Device)?;
                        for blk in 0..n_blocks {
                            npu.layer_kv_blocks[l][blk].as_mut_slice().fill(0);
                        }
                        let is_f16 = key_cache.is_empty() && !key_cache_f16.is_empty();
                        for t in 0..target_pos {
                            let g_chk = t / LFM2_KV_CHUNK_T;
                            let blk = g_chk / LFM2_MAX_KV_CHUNKS;
                            let chk = g_chk % LFM2_MAX_KV_CHUNKS;
                            let slot = t % LFM2_KV_CHUNK_T;
                            let kv_buf = npu.layer_kv_blocks[l][blk].as_mut_slice();
                            for kvh in 0..LFM2_N_KV_HEADS {
                                let src = t * LFM2_KV_DIM + kvh * LFM2_HEAD_DIM;
                                let chk_off =
                                    (kvh * LFM2_MAX_KV_CHUNKS + chk) * LFM2_KV_CHUNK_BYTES;
                                let k_off = chk_off + slot * LFM2_HEAD_DIM * 2;
                                // Both LFM2 prefill and decode kernels read V at +1024*4 inside each 2304-f32 chunk.
                                let v_off = chk_off + 1024 * 4 + slot * LFM2_HEAD_DIM * 2;
                                for d in 0..LFM2_HEAD_DIM {
                                    let (kf, vf) = if is_f16 {
                                        (
                                            crate::quant::f16_to_f32(key_cache_f16[src + d]),
                                            crate::quant::f16_to_f32(value_cache_f16[src + d]),
                                        )
                                    } else {
                                        (key_cache[src + d], value_cache[src + d])
                                    };
                                    let vb = f32_to_bf16(vf).to_le_bytes();
                                    kv_buf[k_off + d * 2..k_off + d * 2 + 2]
                                        .copy_from_slice(&f32_to_bf16(kf).to_le_bytes());
                                    kv_buf[v_off + d * 2..v_off + d * 2 + 2].copy_from_slice(&vb);
                                }
                            }
                        }
                        let total_chunks = target_pos.div_ceil(LFM2_KV_CHUNK_T);
                        for blk in 0..n_blocks {
                            let blk_chunks = if blk + 1 < n_blocks {
                                LFM2_MAX_KV_CHUNKS
                            } else {
                                total_chunks.saturating_sub(blk * LFM2_MAX_KV_CHUNKS)
                            };
                            if blk_chunks == 0 {
                                continue;
                            }
                            for kvh in 0..LFM2_N_KV_HEADS {
                                npu.layer_kv_blocks[l][blk]
                                    .sync_range_to_device(
                                        blk_chunks * LFM2_KV_CHUNK_BYTES,
                                        kvh * LFM2_MAX_KV_CHUNKS * LFM2_KV_CHUNK_BYTES,
                                    )
                                    .map_err(|e| {
                                        SyncErr::Device(format!(
                                            "layer {l} blk {blk} kvh {kvh} bo_kv sync failed: {e}"
                                        ))
                                    })?;
                            }
                        }
                    }
                    _ => {
                        return Err(SyncErr::Input(format!(
                            "layer {l} expected Attention state"
                        )));
                    }
                }
            }
        }
        npu.synced_len = target_pos;
        npu.force_conv_sync = false;
        if let Some(dec) = npu.decode.as_mut() {
            dec.conv_kv_in_kvo = false;
        }
        if let Some(pre) = npu.prefill.as_mut() {
            pre.conv_kv_in_kvo = false;
            pre.last_conv_n_valid = None;
        }
        Ok(())
    }

    fn save_conv_kvo_to_state(
        kvo: &XdnaBo,
        layer_state: &mut LayerState,
        next_pos: usize,
        record_history: bool,
    ) {
        let src = &kvo.as_slice()[..LFM2_KVO_BYTES];
        if let LayerState::Conv { buffer, history } = layer_state {
            if buffer.len() < LFM2_D_CONV * LFM2_H {
                buffer.resize(LFM2_D_CONV * LFM2_H, 0.0);
            }
            for col in 0..8 {
                let col_off = col * TILE_BYTES;
                for k in 0..LFM2_D_CONV {
                    let dst_off = k * LFM2_H + col * 256;
                    read_le_f32_slice(
                        src,
                        (col_off / 4) + k * 256,
                        &mut buffer[dst_off..dst_off + 256],
                    );
                }
            }
            if record_history {
                history.push(next_pos, buffer);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn save_attn_prefill_kvo_to_bo_kv(
        kernels: &Lfm2Kernels,
        kvo: &XdnaBo,
        kv_blocks: &mut Vec<XdnaBo>,
        layer: usize,
        pos0: usize,
        n_valid: usize,
    ) -> Result<Lfm2PrefillAttnMirrorEntry, String> {
        let save_sub =
            |kvo_bytes: &[u8], kv_bytes: &mut [u8], sub: usize, loc_chunk: usize, n_sub: usize| {
                let sync_bytes = n_sub * LFM2_HEAD_DIM * 2;
                for kvh in 0..LFM2_N_KV_HEADS {
                    let src_base = (sub * LFM2_N_KV_HEADS + kvh) * (2304 * 4);
                    let dst_base = (kvh * LFM2_MAX_KV_CHUNKS + loc_chunk) * LFM2_KV_CHUNK_BYTES;
                    kv_bytes[dst_base..dst_base + sync_bytes]
                        .copy_from_slice(&kvo_bytes[src_base..src_base + sync_bytes]);
                    kv_bytes[dst_base + 1024 * 4..dst_base + 1024 * 4 + sync_bytes]
                        .copy_from_slice(
                            &kvo_bytes[src_base + 1024 * 4..src_base + 1024 * 4 + sync_bytes],
                        );
                }
            };
        let chunk0 = pos0 / LFM2_KV_CHUNK_T;
        let blk0 = chunk0 / LFM2_MAX_KV_CHUNKS;
        let loc0 = chunk0 % LFM2_MAX_KV_CHUNKS;
        let n0 = n_valid.min(LFM2_KV_CHUNK_T);
        let n1 = n_valid.saturating_sub(LFM2_KV_CHUNK_T);
        let (blk1, loc1) = if n1 > 0 {
            let chunk1 = chunk0 + 1;
            (chunk1 / LFM2_MAX_KV_CHUNKS, chunk1 % LFM2_MAX_KV_CHUNKS)
        } else {
            (blk0, loc0)
        };
        Self::ensure_paged_block(kernels, kv_blocks, blk1, layer)?;
        {
            let kvo_bytes = kvo.as_slice();
            save_sub(kvo_bytes, kv_blocks[blk0].as_mut_slice(), 0, loc0, n0);
            if n1 > 0 {
                save_sub(kvo_bytes, kv_blocks[blk1].as_mut_slice(), 1, loc1, n1);
            }
        }
        let sync_bytes0 = n0 * LFM2_HEAD_DIM * 2;
        if n1 > 0 && blk0 == blk1 && loc1 == loc0 + 1 {
            let sync_bytes1 = n1 * LFM2_HEAD_DIM * 2;
            let coalesced_bytes = LFM2_KV_CHUNK_BYTES + 1024 * 4 + sync_bytes1;
            for kvh in 0..LFM2_N_KV_HEADS {
                let dst0 = (kvh * LFM2_MAX_KV_CHUNKS + loc0) * LFM2_KV_CHUNK_BYTES;
                kv_blocks[blk0]
                    .sync_range_to_device(coalesced_bytes, dst0)
                    .map_err(|e| {
                        format!("kvh {kvh} prefill KV bo_kv coalesced sync failed: {e}")
                    })?;
            }
        } else {
            for kvh in 0..LFM2_N_KV_HEADS {
                let dst0 = (kvh * LFM2_MAX_KV_CHUNKS + loc0) * LFM2_KV_CHUNK_BYTES;
                kv_blocks[blk0]
                    .sync_range_to_device(1024 * 4 + sync_bytes0, dst0)
                    .map_err(|e| format!("kvh {kvh} prefill sub0 KV bo_kv sync failed: {e}"))?;
                if n1 > 0 {
                    let sync_bytes1 = n1 * LFM2_HEAD_DIM * 2;
                    let dst1 = (kvh * LFM2_MAX_KV_CHUNKS + loc1) * LFM2_KV_CHUNK_BYTES;
                    kv_blocks[blk1]
                        .sync_range_to_device(1024 * 4 + sync_bytes1, dst1)
                        .map_err(|e| format!("kvh {kvh} prefill sub1 KV bo_kv sync failed: {e}"))?;
                }
            }
        }
        Ok(Lfm2PrefillAttnMirrorEntry {
            layer,
            pos0,
            blk0,
            loc0,
            n0,
            blk1,
            loc1,
            n1,
        })
    }

    fn mirror_prefill_sub_bo_kv_to_state(
        bo_bytes: &[u8],
        loc_chunk: usize,
        layer_state: &mut LayerState,
        pos0: usize,
        n_valid: usize,
        use_f16: bool,
    ) {
        let LayerState::Attention {
            key_cache,
            value_cache,
            key_cache_f16,
            value_cache_f16,
            ..
        } = layer_state
        else {
            return;
        };
        let needed = (pos0 + n_valid) * LFM2_KV_DIM;
        let row_bytes = LFM2_HEAD_DIM * 2;
        if use_f16 {
            if key_cache_f16.len() < needed {
                key_cache_f16.resize(needed, 0);
            }
            if value_cache_f16.len() < needed {
                value_cache_f16.resize(needed, 0);
            }
            for t in 0..n_valid {
                let dst_row = (pos0 + t) * LFM2_KV_DIM;
                for kvh in 0..LFM2_N_KV_HEADS {
                    let k_src = (kvh * LFM2_MAX_KV_CHUNKS + loc_chunk) * LFM2_KV_CHUNK_BYTES
                        + t * row_bytes;
                    let v_src = k_src + 1024 * 4;
                    let dst = dst_row + kvh * LFM2_HEAD_DIM;
                    crate::backend::cpu::bf16_le_bytes_to_f16_into(
                        &bo_bytes[k_src..k_src + row_bytes],
                        &mut key_cache_f16[dst..dst + LFM2_HEAD_DIM],
                    );
                    crate::backend::cpu::bf16_le_bytes_to_f16_into(
                        &bo_bytes[v_src..v_src + row_bytes],
                        &mut value_cache_f16[dst..dst + LFM2_HEAD_DIM],
                    );
                }
            }
        } else {
            if key_cache.len() < needed {
                key_cache.resize(needed, 0.0);
            }
            if value_cache.len() < needed {
                value_cache.resize(needed, 0.0);
            }
            for t in 0..n_valid {
                let dst_row = (pos0 + t) * LFM2_KV_DIM;
                for kvh in 0..LFM2_N_KV_HEADS {
                    let k_src = (kvh * LFM2_MAX_KV_CHUNKS + loc_chunk) * LFM2_KV_CHUNK_BYTES
                        + t * row_bytes;
                    let v_src = k_src + 1024 * 4;
                    let dst = dst_row + kvh * LFM2_HEAD_DIM;
                    crate::backend::cpu::bf16_le_bytes_to_f32_into(
                        &bo_bytes[k_src..k_src + row_bytes],
                        &mut key_cache[dst..dst + LFM2_HEAD_DIM],
                    );
                    crate::backend::cpu::bf16_le_bytes_to_f32_into(
                        &bo_bytes[v_src..v_src + row_bytes],
                        &mut value_cache[dst..dst + LFM2_HEAD_DIM],
                    );
                }
            }
        }
    }

    fn mirror_attn_prefill_bo_kv_to_state(
        blocks: &[XdnaBo],
        layer_state: &mut LayerState,
        entry: &Lfm2PrefillAttnMirrorEntry,
        use_f16: bool,
    ) {
        Self::mirror_prefill_sub_bo_kv_to_state(
            blocks[entry.blk0].as_slice(),
            entry.loc0,
            layer_state,
            entry.pos0,
            entry.n0,
            use_f16,
        );
        if entry.n1 > 0 {
            Self::mirror_prefill_sub_bo_kv_to_state(
                blocks[entry.blk1].as_slice(),
                entry.loc1,
                layer_state,
                entry.pos0 + LFM2_KV_CHUNK_T,
                entry.n1,
                use_f16,
            );
        }
    }

    fn flush_pending_attn_mirrors(
        pending: &mut Vec<Lfm2PrefillAttnMirrorEntry>,
        layer_kv_blocks: &[Vec<XdnaBo>],
        layers_state: &mut [LayerState],
        use_f16: bool,
    ) {
        for entry in pending.drain(..) {
            Self::mirror_attn_prefill_bo_kv_to_state(
                &layer_kv_blocks[entry.layer],
                &mut layers_state[entry.layer],
                &entry,
                use_f16,
            );
        }
    }

    fn save_attn_dec_kvo_to_npu(
        kernels: &Lfm2Kernels,
        kvo: &XdnaBo,
        kv_blocks: &mut Vec<XdnaBo>,
        pos: usize,
        layer: usize,
    ) -> Result<(), String> {
        let g_chk = pos / LFM2_KV_CHUNK_T;
        let blk = g_chk / LFM2_MAX_KV_CHUNKS;
        let chk = g_chk % LFM2_MAX_KV_CHUNKS;
        let slot = pos % LFM2_KV_CHUNK_T;
        let row_bytes = LFM2_HEAD_DIM * 2;
        Self::ensure_paged_block(kernels, kv_blocks, blk, layer)?;
        let bo_kv = &mut kv_blocks[blk];
        {
            let kvo_bytes = kvo.as_slice();
            let kv_bytes = bo_kv.as_mut_slice();

            // Decode `kvo_buf` drains col `c = 0..7` at `c * 256` f32 (`c * 1024` bytes),
            // with `K` at `+128` f32 (`+512` B) and `V` at `+192` f32 (`+768` B).
            for kvh in 0..LFM2_N_KV_HEADS {
                let src_base = kvh * 1024;
                let k_src = &kvo_bytes[src_base + 512..src_base + 512 + row_bytes];
                let v_src = &kvo_bytes[src_base + 768..src_base + 768 + row_bytes];
                let dst_chk = (kvh * LFM2_MAX_KV_CHUNKS + chk) * LFM2_KV_CHUNK_BYTES;
                let k_off = dst_chk + slot * row_bytes;
                let v_off = dst_chk + 1024 * 4 + slot * row_bytes;
                kv_bytes[k_off..k_off + row_bytes].copy_from_slice(k_src);
                kv_bytes[v_off..v_off + row_bytes].copy_from_slice(v_src);
            }
        }
        for kvh in 0..LFM2_N_KV_HEADS {
            let dst_chk = (kvh * LFM2_MAX_KV_CHUNKS + chk) * LFM2_KV_CHUNK_BYTES;
            let k_off = dst_chk + slot * row_bytes;
            bo_kv
                .sync_range_to_device(1024 * 4 + row_bytes, k_off)
                .map_err(|e| format!("kvh {kvh} dec KV bo_kv sync: {e}"))?;
        }
        Ok(())
    }

    fn flush_dec_conv_pingpong(&self, npu: &mut XdnaLfm2State) -> Result<(), String> {
        let Some(dec) = npu.decode.as_mut() else {
            return Ok(());
        };
        if dec.conv_kv_in_kvo {
            for (l, layer) in self.layers.iter().enumerate() {
                if layer.is_conv {
                    let src = &dec.layer_bo_kvo[l].as_slice()[..LFM2_KVO_BYTES];
                    npu.layer_kv_blocks[l][0].as_mut_slice()[..LFM2_KVO_BYTES].copy_from_slice(src);
                    npu.layer_kv_blocks[l][0]
                        .sync_range_to_device(LFM2_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} flush dec conv ping-pong: {e}"))?;
                }
            }
            dec.conv_kv_in_kvo = false;
        }
        Ok(())
    }

    fn flush_prefill_conv_pingpong(&self, npu: &mut XdnaLfm2State) -> Result<(), String> {
        let Some(pre) = npu.prefill.as_mut() else {
            return Ok(());
        };
        if pre.conv_kv_in_kvo {
            for (l, layer) in self.layers.iter().enumerate() {
                if layer.is_conv {
                    let src = &pre.layer_bo_kvo[l].as_slice()[..LFM2_KVO_BYTES];
                    npu.layer_kv_blocks[l][0].as_mut_slice()[..LFM2_KVO_BYTES].copy_from_slice(src);
                    npu.layer_kv_blocks[l][0]
                        .sync_range_to_device(LFM2_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} flush prefill conv ping-pong: {e}"))?;
                }
            }
            pre.conv_kv_in_kvo = false;
        }
        Ok(())
    }

    fn run_npu_decode_embedding_step(
        &self,
        npu: &mut XdnaLfm2State,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, String> {
        let vocab_size = self.cpu.config().vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        self.run_npu_decode_embedding_step_into(npu, embedding, pos, state, &mut logits)?;
        Ok(logits)
    }

    fn run_npu_decode_embedding_step_into(
        &self,
        npu: &mut XdnaLfm2State,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
        logits_out: &mut [f32],
    ) -> Result<(), String> {
        self.run_npu_decode_embedding_step_inner(npu, embedding, pos, state, true, logits_out)
    }

    fn run_npu_decode_embedding_step_inner(
        &self,
        npu: &mut XdnaLfm2State,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
        logits_out: &mut [f32],
    ) -> Result<(), String> {
        let cfg = self.cpu.config();
        let eps = cfg.rms_norm_eps;
        let prof = self.profile_decode;
        let t_prep0 = prof.then(crate::time::Instant::now);

        if npu.decode.is_none() {
            npu.decode = Some(
                Self::alloc_decode_buffers(&mut npu.kernels, &self.layers)
                    .map_err(|e| format!("{e:#}"))?,
            );
        }

        let cur_blk = pos / LFM2_BLOCK_TOKENS;
        let is_ext = cur_blk > 0;
        let conv_kv_in_kvo = npu.decode.as_ref().expect("dec").conv_kv_in_kvo;
        let parity_idx = usize::from(conv_kv_in_kvo);
        let n_layers = self.layers.len();

        let XdnaLfm2State {
            ref mut kernels,
            ref layer_bo_w,
            ref mut layer_kv_blocks,
            ref mut decode,
            ref mut pending_kv,
            ref mut head_q8_scratch,
            ref mut force_conv_sync,
            ref mut in_flight,
            ref mut stalled_runs,
            ref mut moe_ffn_input,
            ..
        } = *npu;
        let dec = decode.as_mut().expect("initialized decode buffers");

        let (attn_kv_cap, ext_key, attn_bidx) = if !is_ext {
            let attn_bucket_caps: Vec<usize> =
                Q4LAYER_LFM2_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
            let (bidx, cap) = if pos == 0 {
                (0, 0)
            } else {
                try_select_lfm3_kv_bucket(&attn_bucket_caps, pos.div_ceil(LFM2_DEC_PKT_TOKENS))?
            };
            if !self.host_ffn && dec.bucket_plans[parity_idx][bidx].is_none() {
                let plan = Self::build_decode_bucket_plan(
                    kernels,
                    &self.layers,
                    layer_bo_w,
                    layer_kv_blocks,
                    &dec.layer_bo_y,
                    &dec.layer_bo_kvo,
                    &dec.bo_yn,
                    bidx,
                    conv_kv_in_kvo,
                )?;
                dec.bucket_plans[parity_idx][bidx] = Some(plan);
            }
            (cap, None, bidx)
        } else {
            let fin_caps: Vec<usize> = Q4LAYER_LFM2_FA_FIN_BUCKETS
                .iter()
                .map(|&(c, _)| c)
                .collect();
            let past_pkts_total = pos.div_ceil(LFM2_DEC_PKT_TOKENS);
            let fin_past_pkts = past_pkts_total.saturating_sub(cur_blk * LFM2_DEC_MAX_KV_PKTS);
            let (fin_bidx, fin_cap) = try_select_lfm3_kv_bucket(&fin_caps, fin_past_pkts)?;
            for (l, layer) in self.layers.iter().enumerate() {
                if !layer.is_conv {
                    Self::ensure_paged_block(kernels, &mut layer_kv_blocks[l], cur_blk, l)?;
                }
            }
            if !self.host_ffn && cur_blk <= 1 {
                let key = (cur_blk, fin_bidx);
                let Lfm2DecodeBuffers {
                    layer_bo_y: dec_bo_y,
                    layer_bo_kvo: dec_bo_kvo,
                    bo_yn: dec_bo_yn,
                    ext_plans: dec_ext_plans,
                    ..
                } = dec;
                let _ = dec_ext_plans.get_or_insert_with(parity_idx, key, || {
                    Self::build_decode_ext_plan(
                        kernels,
                        &self.layers,
                        layer_bo_w,
                        layer_kv_blocks,
                        dec_bo_y,
                        dec_bo_kvo,
                        dec_bo_yn,
                        cur_blk,
                        cur_blk,
                        fin_bidx,
                        conv_kv_in_kvo,
                    )
                })?;
            }
            (
                cur_blk * LFM2_DEC_MAX_KV_PKTS + fin_cap,
                Some((cur_blk, fin_bidx)),
                0,
            )
        };

        write_le_f32_slice(dec.layer_bo_y[0].as_mut_slice(), 0, embedding);
        dec.layer_bo_y[0]
            .sync_range_to_device(LFM2_DEC_YN_BYTES, 0)
            .map_err(|e| format!("decode bo_y[0] hidden sync: {e}"))?;

        let conv_lmode = if self.host_ffn { 5 } else { 0 };
        let attn_lmode = if self.host_ffn { 6 } else { 1 };
        let ext_fin_lmode = if self.host_ffn { 7 } else { 4 };

        for (l, layer) in self.layers.iter().enumerate() {
            let yb = dec.layer_bo_y[l].as_mut_slice();
            if layer.is_conv {
                write_lfm3_step_params(
                    yb,
                    LFM2_DEC_SLOT_NORM_0,
                    None,
                    pos,
                    1,
                    0,
                    0,
                    true,
                    0,
                    conv_lmode,
                    eps,
                    None,
                );
                dec.layer_bo_y[l]
                    .sync_range_to_device(64, (LFM2_DEC_SLOT_NORM_0 + 4096) * 4)
                    .map_err(|e| format!("layer {l} dec conv params sync: {e}"))?;
            } else if !is_ext {
                let cs = compute_lfm2_rope_cs(pos, layer.rope_theta);
                write_lfm3_step_params(
                    yb,
                    LFM2_DEC_SLOT_NORM_0,
                    Some(LFM2_DEC_SLOT_NORM_0 + 4112),
                    pos,
                    1,
                    attn_kv_cap,
                    pos,
                    true,
                    0,
                    attn_lmode,
                    eps,
                    Some(&cs),
                );
                dec.layer_bo_y[l]
                    .sync_range_to_device(
                        (16 + LFM2_HEAD_DIM) * 4,
                        (LFM2_DEC_SLOT_NORM_0 + 4096) * 4,
                    )
                    .map_err(|e| format!("layer {l} dec attn params+rope sync: {e}"))?;
            } else {
                let (cur_blk, fin_bidx) = ext_key.expect("ext key");
                let fin_cap = Q4LAYER_LFM2_FA_FIN_BUCKETS[fin_bidx].0;
                let cs = compute_lfm2_rope_cs(pos, layer.rope_theta);
                // Mode 2 (Phase A) reads SLOT_NORM_0; Mode 3 (fa_part) reads SLOT_NORM_1;
                // Mode 4/7 (fa_fin) reads SLOT_ROPE_PKT.
                write_lfm3_step_params(
                    yb,
                    LFM2_DEC_SLOT_NORM_0,
                    Some(LFM2_DEC_SLOT_NORM_0 + 4112),
                    pos,
                    1,
                    0,
                    pos,
                    true,
                    0,
                    2,
                    eps,
                    Some(&cs),
                );
                if cur_blk <= 1 {
                    write_lfm3_params_at(
                        yb,
                        LFM2_DEC_SLOT_NORM_1 + 4096,
                        pos,
                        1,
                        LFM2_DEC_MAX_KV_PKTS,
                        pos,
                        true,
                        0,
                        3,
                        eps,
                        0,
                        false,
                        true,
                    );
                }
                write_lfm3_params_at(
                    yb,
                    LFM2_DEC_SLOT_ROPE_PKT + 4096,
                    pos,
                    1,
                    fin_cap,
                    pos,
                    true,
                    0,
                    ext_fin_lmode,
                    eps,
                    cur_blk * LFM2_DEC_MAX_KV_PKTS,
                    true,
                    false,
                );
                dec.layer_bo_y[l]
                    .sync_range_to_device(
                        (16 + LFM2_HEAD_DIM) * 4,
                        (LFM2_DEC_SLOT_NORM_0 + 4096) * 4,
                    )
                    .map_err(|e| format!("layer {l} ext dec phase A params+rope sync: {e}"))?;
                if cur_blk <= 1 {
                    dec.layer_bo_y[l]
                        .sync_range_to_device(64, (LFM2_DEC_SLOT_NORM_1 + 4096) * 4)
                        .map_err(|e| format!("layer {l} ext dec fa_part params sync: {e}"))?;
                }
                dec.layer_bo_y[l]
                    .sync_range_to_device(64, (LFM2_DEC_SLOT_ROPE_PKT + 4096) * 4)
                    .map_err(|e| format!("layer {l} ext dec fa_fin params sync: {e}"))?;
            }
        }
        let t_prep_us = t_prep0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let mut hidden_out = [0.0f32; LFM2_H];
        let (t_npu_us, t_kv_us) = if self.host_ffn {
            let t0 = prof.then(crate::time::Instant::now);
            let mut npu_exec_us = 0.0f64;
            let mut sync_us = 0.0f64;
            let mut ffn_us = 0.0f64;
            let mut pending_next = pending_kv.prepare(pos, LFM2_D_CONV, LFM2_H, LFM2_KV_DIM);
            if !pending_kv.flush_into(state) {
                *force_conv_sync = true;
            }
            for (l, layer) in self.layers.iter().enumerate() {
                let tn0 = prof.then(crate::time::Instant::now);
                if layer.is_conv {
                    if dec.moe_conv_runs[parity_idx][l].is_none() {
                        let (kv_in, kv_out) = if conv_kv_in_kvo {
                            (&dec.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                        } else {
                            (&layer_kv_blocks[l][0], &dec.layer_bo_kvo[l])
                        };
                        dec.moe_conv_runs[parity_idx][l] = Some(
                            kernels
                                .dec_conv
                                .create_run(&[
                                    &layer_bo_w[l],
                                    &dec.layer_bo_y[l],
                                    &dec.bo_yn,
                                    kv_in,
                                    kv_out,
                                ])
                                .map_err(|e| format!("layer {l} moe dec conv create_run: {e}"))?,
                        );
                    }
                    exec_cached_run_slot(
                        &mut dec.moe_conv_runs[parity_idx][l],
                        stalled_runs,
                        &format!("layer {l} moe dec conv exec"),
                    )?;
                } else if let Some((blk, fin_bidx)) = ext_key {
                    if blk <= 1 {
                        if !dec.moe_ext_plans[parity_idx][l]
                            .iter()
                            .any(|(b, f, _)| *b == blk && *f == fin_bidx)
                        {
                            let plan = Self::build_decode_ext_layer_plan(
                                kernels,
                                l,
                                &layer_bo_w[l],
                                &dec.layer_bo_y[l],
                                &dec.bo_yn,
                                &dec.layer_bo_kvo[l],
                                &layer_kv_blocks[l],
                                blk,
                                fin_bidx,
                            )?;
                            dec.moe_ext_plans[parity_idx][l].push((blk, fin_bidx, plan));
                        }
                        let plan = dec.moe_ext_plans[parity_idx][l]
                            .iter()
                            .find(|(b, f, _)| *b == blk && *f == fin_bidx)
                            .map(|(_, _, p)| p)
                            .expect("initialized moe dec ext plan");
                        *in_flight = Some(InFlightLfm2Run::DecMoeExt {
                            parity_idx,
                            l,
                            blk,
                            fin_bidx,
                        });
                        if let Err(e) = plan.runlist.execute() {
                            *in_flight = None;
                            return Err(format!("layer {l} moe dec ext exec: {e}"));
                        }
                        plan.runlist
                            .wait(QUIESCE_TIMEOUT_MS)
                            .map_err(|e| format!("layer {l} moe dec ext wait: {e}"))?;
                        *in_flight = None;
                    } else {
                        Self::run_decode_ext_attn_multi_block_layer(
                            kernels,
                            l,
                            &layer_bo_w[l],
                            &mut dec.layer_bo_y[l],
                            &dec.bo_yn,
                            &dec.layer_bo_kvo[l],
                            &layer_kv_blocks[l],
                            stalled_runs,
                            pos,
                            eps,
                            blk,
                            fin_bidx,
                        )?;
                    }
                } else {
                    if dec.moe_attn_runs[attn_bidx][l].is_none() {
                        dec.moe_attn_runs[attn_bidx][l] = Some(
                            kernels
                                .dec_attn(attn_bidx)?
                                .create_run(&[
                                    &layer_bo_w[l],
                                    &dec.layer_bo_y[l],
                                    &dec.bo_yn,
                                    &layer_kv_blocks[l][0],
                                    &dec.layer_bo_kvo[l],
                                ])
                                .map_err(|e| format!("layer {l} moe dec attn create_run: {e}"))?,
                        );
                    }
                    exec_cached_run_slot(
                        &mut dec.moe_attn_runs[attn_bidx][l],
                        stalled_runs,
                        &format!("layer {l} moe dec attn exec"),
                    )?;
                }
                if let Some(t) = tn0 {
                    npu_exec_us += t.elapsed().as_secs_f64() * 1e6;
                }

                let ts0 = prof.then(crate::time::Instant::now);
                dec.bo_yn
                    .sync_range_from_device(LFM2_DEC_YN_BYTES, 0)
                    .map_err(|e| format!("layer {l} moe dec bo_yn readback: {e}"))?;
                read_le_f32_slice(dec.bo_yn.as_slice(), 0, &mut hidden_out);

                if layer.is_conv {
                    let kvo_written = if conv_kv_in_kvo {
                        &mut layer_kv_blocks[l][0]
                    } else {
                        &mut dec.layer_bo_kvo[l]
                    };
                    kvo_written
                        .sync_range_from_device(CONV_SYNC_BYTES, 0)
                        .map_err(|e| format!("layer {l} dec conv kvo readback: {e}"))?;
                    pending_next.push_conv_from_kvo(l, &kvo_written.as_slice()[..CONV_SYNC_BYTES]);
                } else {
                    dec.layer_bo_kvo[l]
                        .sync_range_from_device(LFM2_ATTN_DEC_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} dec attn kvo readback: {e}"))?;
                    Self::save_attn_dec_kvo_to_npu(
                        kernels,
                        &dec.layer_bo_kvo[l],
                        &mut layer_kv_blocks[l],
                        pos,
                        l,
                    )?;
                    pending_next.push_attn_bf16(
                        l,
                        dec.layer_bo_kvo[l].as_slice(),
                        LFM2_N_KV_HEADS,
                        LFM2_HEAD_DIM,
                        1024,
                    );
                }
                if let Some(t) = ts0 {
                    sync_us += t.elapsed().as_secs_f64() * 1e6;
                }

                let tf0 = prof.then(crate::time::Instant::now);
                self.cpu
                    .xdna_forward_ffn_step(l, &mut hidden_out, moe_ffn_input, state);
                if let Some(t) = tf0 {
                    ffn_us += t.elapsed().as_secs_f64() * 1e6;
                }
                let ts1 = prof.then(crate::time::Instant::now);
                if l + 1 < n_layers {
                    write_le_f32_slice(dec.layer_bo_y[l + 1].as_mut_slice(), 0, &hidden_out);
                    dec.layer_bo_y[l + 1]
                        .sync_range_to_device(LFM2_DEC_YN_BYTES, 0)
                        .map_err(|e| format!("decode bo_y[{}] hidden sync: {e}", l + 1))?;
                }
                if let Some(t) = ts1 {
                    sync_us += t.elapsed().as_secs_f64() * 1e6;
                }
            }
            dec.conv_kv_in_kvo = !conv_kv_in_kvo;
            pending_kv.commit(pending_next);
            state.seq_len = pos + 1;
            npu.synced_len = pos + 1;
            if pos + 1 >= cfg.max_seq_len.min(LFM2_MAX_TOKENS) && !npu.pending_kv.flush_into(state)
            {
                npu.synced_len = 0;
                npu.force_conv_sync = true;
            }
            if prof {
                eprintln!(
                    "lfm2moe decode pos={pos} bucket={attn_kv_cap}: prep={t_prep_us:.0}us npu_run={npu_exec_us:.0}us dma_sync={sync_us:.0}us host_ffn={ffn_us:.0}us"
                );
            }
            (t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6), sync_us)
        } else {
            let t_npu0 = prof.then(crate::time::Instant::now);
            let mut pending_next = if matches!(ext_key, Some((blk, _)) if blk >= 2) {
                let (blk, fin_bidx) = ext_key.expect("ext key");
                let pending_next = pending_kv.prepare(pos, LFM2_D_CONV, LFM2_H, LFM2_KV_DIM);
                if !pending_kv.flush_into(state) {
                    *force_conv_sync = true;
                }
                for (l, layer) in self.layers.iter().enumerate() {
                    let (cur_bo_y, y_next): (&mut XdnaBo, &XdnaBo) = if l + 1 < n_layers {
                        let (head, tail) = dec.layer_bo_y.split_at_mut(l + 1);
                        (&mut head[l], &tail[0])
                    } else {
                        (&mut dec.layer_bo_y[l], &dec.bo_yn)
                    };
                    if layer.is_conv {
                        let (kv_in, kv_out) = if conv_kv_in_kvo {
                            (&dec.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                        } else {
                            (&layer_kv_blocks[l][0], &dec.layer_bo_kvo[l])
                        };
                        let run = kernels
                            .dec_conv
                            .create_run(&[&layer_bo_w[l], cur_bo_y, y_next, kv_in, kv_out])
                            .map_err(|e| format!("layer {l} ext dec conv create_run: {e}"))?;
                        exec_owned_run(stalled_runs, run, &format!("layer {l} ext dec conv exec"))?;
                    } else {
                        Self::run_decode_ext_attn_multi_block_layer(
                            kernels,
                            l,
                            &layer_bo_w[l],
                            cur_bo_y,
                            y_next,
                            &dec.layer_bo_kvo[l],
                            &layer_kv_blocks[l],
                            stalled_runs,
                            pos,
                            eps,
                            blk,
                            fin_bidx,
                        )?;
                    }
                }
                pending_next
            } else {
                let (plan, active_run) = if let Some(key) = ext_key {
                    let p = dec
                        .ext_plans
                        .peek(parity_idx, key)
                        .expect("initialized decode ext plan");
                    (p, InFlightLfm2Run::DecExt { parity_idx, key })
                } else {
                    let p = dec.bucket_plans[parity_idx][attn_bidx]
                        .as_ref()
                        .expect("initialized decode bucket plan");
                    (
                        p,
                        InFlightLfm2Run::DecFast {
                            parity_idx,
                            bidx: attn_bidx,
                        },
                    )
                };
                *in_flight = Some(active_run);
                if let Err(e) = plan.runlist.execute() {
                    *in_flight = None;
                    return Err(format!("dec runlist execute: {e}"));
                }
                let pending_next = pending_kv.prepare(pos, LFM2_D_CONV, LFM2_H, LFM2_KV_DIM);
                if !pending_kv.flush_into(state) {
                    *force_conv_sync = true;
                }
                plan.runlist
                    .wait(QUIESCE_TIMEOUT_MS)
                    .map_err(|e| format!("dec runlist wait: {e}"))?;
                *in_flight = None;
                pending_next
            };
            let t_npu_us = t_npu0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

            let t_kv0 = prof.then(crate::time::Instant::now);
            if need_logits {
                dec.bo_yn
                    .sync_range_from_device(LFM2_DEC_YN_BYTES, 0)
                    .map_err(|e| format!("decode bo_yn readback: {e}"))?;
                read_le_f32_slice(dec.bo_yn.as_slice(), 0, &mut hidden_out);
            }

            for (l, layer) in self.layers.iter().enumerate() {
                if layer.is_conv {
                    let kvo_written = if conv_kv_in_kvo {
                        &mut layer_kv_blocks[l][0]
                    } else {
                        &mut dec.layer_bo_kvo[l]
                    };
                    kvo_written
                        .sync_range_from_device(CONV_SYNC_BYTES, 0)
                        .map_err(|e| format!("layer {l} dec conv kvo readback: {e}"))?;
                    pending_next.push_conv_from_kvo(l, &kvo_written.as_slice()[..CONV_SYNC_BYTES]);
                } else {
                    dec.layer_bo_kvo[l]
                        .sync_range_from_device(LFM2_ATTN_DEC_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} dec attn kvo readback: {e}"))?;
                    Self::save_attn_dec_kvo_to_npu(
                        kernels,
                        &dec.layer_bo_kvo[l],
                        &mut layer_kv_blocks[l],
                        pos,
                        l,
                    )?;
                    pending_next.push_attn_bf16(
                        l,
                        dec.layer_bo_kvo[l].as_slice(),
                        LFM2_N_KV_HEADS,
                        LFM2_HEAD_DIM,
                        1024,
                    );
                }
            }
            dec.conv_kv_in_kvo = !conv_kv_in_kvo;

            pending_kv.commit(pending_next);
            state.seq_len = pos + 1;
            npu.synced_len = pos + 1;
            if pos + 1 >= cfg.max_seq_len.min(LFM2_MAX_TOKENS) && !npu.pending_kv.flush_into(state)
            {
                npu.synced_len = 0;
                npu.force_conv_sync = true;
            }
            (
                t_npu_us,
                t_kv0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6),
            )
        };

        let t_head0 = prof.then(crate::time::Instant::now);
        if need_logits {
            self.cpu.xdna_project_logits_into_with_scratch(
                &hidden_out,
                logits_out,
                Some((&mut head_q8_scratch.0, &mut head_q8_scratch.1)),
            );
        }
        let t_head_us = t_head0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        if prof {
            let tag = if is_ext { " >8K" } else { "" };
            eprintln!(
                "lfm2 decode{tag} pos={pos} bucket={attn_kv_cap}: prep={t_prep_us:.0}us npu={t_npu_us:.0}us kv={t_kv_us:.0}us head={t_head_us:.0}us"
            );
        }
        Ok(())
    }

    fn run_npu_prefill_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
    ) -> Vec<f32> {
        self.run_npu_prefill(
            PrefillSource::Embeddings(embeddings, n_tokens),
            start_pos,
            state,
            need_logits,
        )
    }

    fn run_npu_prefill(
        &self,
        src: PrefillSource<'_>,
        start_pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
    ) -> Vec<f32> {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        let out_len = if need_logits { vocab_size } else { 0 };
        if self.state_torn.load(Ordering::SeqCst) {
            return self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                out_len,
            );
        }
        let n_tokens = match src {
            PrefillSource::Tokens(toks) => {
                if toks.is_empty() {
                    return self
                        .reject_input("forward_prefill requires non-empty tokens".into(), out_len);
                }
                if toks.iter().any(|&t| (t as usize) >= vocab_size) {
                    return self.reject_input(
                        format!("token id out of range for vocab {vocab_size}"),
                        out_len,
                    );
                }
                toks.len()
            }
            PrefillSource::Embeddings(embeddings, n_tokens) => {
                if n_tokens == 0 {
                    return self
                        .reject_input("prefill requires non-empty input".to_string(), out_len);
                }
                let Some(expected_emb_len) = n_tokens.checked_mul(LFM2_H) else {
                    return self.reject_input(
                        format!("prefill token count {n_tokens} overflows hidden projection size"),
                        out_len,
                    );
                };
                if embeddings.len() != expected_emb_len {
                    return self.reject_input(
                        format!(
                            "prefill embeddings length {} != {n_tokens} * {LFM2_H}",
                            embeddings.len()
                        ),
                        out_len,
                    );
                }
                n_tokens
            }
        };
        let Some(total_tokens) = start_pos.checked_add(n_tokens) else {
            return self.reject_input(
                format!("prefill range {start_pos} + {n_tokens} overflows position indexing"),
                out_len,
            );
        };
        if total_tokens > cfg.max_seq_len || total_tokens > LFM2_MAX_TOKENS {
            return self.reject_input(
                format!(
                    "prefill tokens {total_tokens} exceed context limit {}",
                    cfg.max_seq_len.min(LFM2_MAX_TOKENS)
                ),
                out_len,
            );
        }
        if let Err(e) = self.validate_state(state) {
            return self.reject_input(e, out_len);
        }

        let eps = cfg.rms_norm_eps;
        let use_f16 = state.kv_f16;
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            return self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                out_len,
            );
        }
        if let Err(e) =
            self.validate_cpu_state_for_sync(state, npu.pending_kv.as_ref(), start_pos, use_f16)
        {
            return self.reject_input(e, out_len);
        }

        // Sync any CPU-written / truncated state before executing.
        if let Err(e) = self.flush_dec_conv_pingpong(&mut npu) {
            return self.record_device_fault(e, out_len);
        }
        match self.sync_cpu_state_to_npu(&mut npu, state, start_pos, use_f16) {
            Ok(()) => {}
            Err(SyncErr::Input(e)) => return self.reject_input(e, out_len),
            Err(SyncErr::Device(e)) => return self.record_device_fault(e, out_len),
        }

        // Unaligned causal continuations (start_pos % 32 != 0) step single-token B=1 decode up to
        // the next 32-token boundary, bypassing B=64 prefill when <= 3 tokens remain.
        let mut cur_start = 0usize;
        if !start_pos.is_multiple_of(LFM2_KV_CHUNK_T) || n_tokens <= 3 {
            if let Err(e) = self.flush_prefill_conv_pingpong(&mut npu) {
                return self.record_device_fault(e, out_len);
            }
            let align_rem = if start_pos.is_multiple_of(LFM2_KV_CHUNK_T) {
                0
            } else {
                LFM2_KV_CHUNK_T - (start_pos % LFM2_KV_CHUNK_T)
            };
            let step_count = if n_tokens <= 3 || n_tokens.saturating_sub(align_rem) <= 3 {
                n_tokens
            } else {
                align_rem.min(n_tokens)
            };
            let mut last_logits = if need_logits && step_count == n_tokens {
                vec![0.0f32; vocab_size]
            } else {
                Vec::new()
            };
            let mut single_emb = match src {
                PrefillSource::Tokens(_) => vec![0.0f32; LFM2_H],
                PrefillSource::Embeddings(_, _) => Vec::new(),
            };
            for i in 0..step_count {
                let emb: &[f32] = match src {
                    PrefillSource::Tokens(toks) => {
                        if !self
                            .cpu
                            .embed_token_rows_into(&toks[i..i + 1], &mut single_emb)
                        {
                            return self.reject_input(
                                format!("token id out of range for vocab {vocab_size}"),
                                out_len,
                            );
                        }
                        &single_emb
                    }
                    PrefillSource::Embeddings(embeddings, _) => {
                        &embeddings[i * LFM2_H..(i + 1) * LFM2_H]
                    }
                };
                let step_need_logits = need_logits && (i + 1 == n_tokens);
                if let Err(e) = self.run_npu_decode_embedding_step_inner(
                    &mut npu,
                    emb,
                    start_pos + i,
                    state,
                    step_need_logits,
                    &mut last_logits,
                ) {
                    return self.record_device_fault(e, out_len);
                }
            }
            cur_start = step_count;
            if cur_start == n_tokens {
                return last_logits;
            }
            if let Err(e) = self.flush_dec_conv_pingpong(&mut npu) {
                return self.record_device_fault(e, out_len);
            }
        }

        let pos_aligned = start_pos + cur_start;
        let rem_tokens = n_tokens - cur_start;
        if npu.prefill.is_none() {
            npu.prefill = match Self::alloc_prefill_buffers(&mut npu.kernels, &self.layers) {
                Ok(p) => Some(p),
                Err(e) => return self.record_device_fault(format!("{e:#}"), out_len),
            };
        }
        if !npu.pending_kv.flush_into(state) {
            npu.synced_len = 0;
            npu.force_conv_sync = true;
        }

        let rem_src = match src {
            PrefillSource::Tokens(toks) => PrefillSource::Tokens(&toks[cur_start..]),
            PrefillSource::Embeddings(embeddings, _) => {
                PrefillSource::Embeddings(&embeddings[cur_start * LFM2_H..], rem_tokens)
            }
        };
        let n_chunks = rem_tokens.div_ceil(LFM2_PREFILL_CHUNK_T);
        if let PrefillSource::Tokens(toks) = rem_src {
            let need = LFM2_PREFILL_CHUNK_T * LFM2_H;
            if npu.prefill_embs_scratch.0.len() < need {
                npu.prefill_embs_scratch.0.resize(need, 0.0);
            }
            if npu.prefill_embs_scratch.1.len() < need {
                npu.prefill_embs_scratch.1.resize(need, 0.0);
            }
            let n0 = rem_tokens.min(LFM2_PREFILL_CHUNK_T);
            if !self
                .cpu
                .embed_token_rows_into(&toks[..n0], &mut npu.prefill_embs_scratch.0[..n0 * LFM2_H])
            {
                return self.reject_input(
                    format!("token id out of range for vocab {vocab_size}"),
                    out_len,
                );
            }
        }
        let attn_caps: Vec<usize> = Q4PREFILL_LFM2_ATTN_BUCKETS
            .iter()
            .map(|&(c, _)| c)
            .collect();
        let fin_caps: Vec<usize> = Q4PREFILL_LFM2_FA_FIN_BUCKETS
            .iter()
            .map(|&(c, _)| c)
            .collect();
        let n_layers = self.layers.len();

        let mut last_hidden = vec![0.0f32; LFM2_H];
        let mut pending_attn_mirrors: Vec<Lfm2PrefillAttnMirrorEntry> =
            Vec::with_capacity(n_layers);
        let default_attn_theta = self
            .layers
            .iter()
            .find(|l| !l.is_conv)
            .map(|l| l.rope_theta);
        let mut prefetched_rope: Option<(usize, usize, f32, Vec<f32>)> = None;

        for c in 0..n_chunks {
            let local_off = c * LFM2_PREFILL_CHUNK_T;
            let n_valid = (rem_tokens - local_off).min(LFM2_PREFILL_CHUNK_T);
            let pos0 = pos_aligned + local_off;
            let past_chunks_c = pos0.div_ceil(LFM2_KV_CHUNK_T);
            let is_ext_c = past_chunks_c >= LFM2_MAX_KV_CHUNKS;
            let is_last_chunk = c + 1 == n_chunks;
            let attn_kvo_sync_bytes = if n_valid > LFM2_KV_CHUNK_T {
                15 * 2304 * 4 + 1024 * 4 + (n_valid - LFM2_KV_CHUNK_T) * LFM2_HEAD_DIM * 2
            } else {
                7 * 2304 * 4 + 1024 * 4 + n_valid * LFM2_HEAD_DIM * 2
            };
            let mut cached_rope: Option<(f32, Vec<f32>)> = None;

            if !is_ext_c {
                let (attn_bidx, attn_kv_cap) =
                    match try_select_lfm3_kv_bucket(&attn_caps, past_chunks_c) {
                        Ok(v) => v,
                        Err(e) => {
                            Self::flush_pending_attn_mirrors(
                                &mut pending_attn_mirrors,
                                &npu.layer_kv_blocks,
                                &mut state.layers,
                                use_f16,
                            );
                            return self.record_device_fault(e, out_len);
                        }
                    };
                let mut swapped_next_embs = false;
                {
                    let XdnaLfm2State {
                        ref mut kernels,
                        ref layer_bo_w,
                        ref mut layer_kv_blocks,
                        ref mut prefill,
                        ref mut in_flight,
                        ref mut stalled_runs,
                        ref mut prefill_embs_scratch,
                        ref mut moe_hidden_scratch,
                        ref mut moe_prefill_scratch,
                        ..
                    } = *npu;
                    let pre = prefill.as_mut().expect("initialized prefill buffers");
                    let chunk_embs: &[f32] = match rem_src {
                        PrefillSource::Tokens(_) => &prefill_embs_scratch.0[..n_valid * LFM2_H],
                        PrefillSource::Embeddings(embeddings, _) => {
                            &embeddings[local_off * LFM2_H..(local_off + n_valid) * LFM2_H]
                        }
                    };
                    let conv_kv_in_kvo = pre.conv_kv_in_kvo;
                    let parity_idx = usize::from(conv_kv_in_kvo);
                    if !self.host_ffn && pre.bucket_plans[parity_idx][attn_bidx].is_none() {
                        let built = Self::build_prefill_chain(
                            kernels,
                            &self.layers,
                            layer_bo_w,
                            layer_kv_blocks,
                            pre,
                            attn_bidx,
                            conv_kv_in_kvo,
                        );
                        pre.bucket_plans[parity_idx][attn_bidx] = Some(match built {
                            Ok(ch) => ch,
                            Err(e) => {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(e, out_len);
                            }
                        });
                    }

                    if let Err(e) = Self::sync_and_write_prefill_hidden(
                        &mut pre.layer_bo_y[0],
                        chunk_embs,
                        n_valid,
                        &format!("prefill c={c} layer 0 hidden sync"),
                    ) {
                        Self::flush_pending_attn_mirrors(
                            &mut pending_attn_mirrors,
                            layer_kv_blocks,
                            &mut state.layers,
                            use_f16,
                        );
                        return self.record_device_fault(e, out_len);
                    }

                    let need_conv_params = pre.last_conv_n_valid != Some(n_valid);
                    let conv_lmode = if self.host_ffn { 5 } else { 0 };
                    let attn_lmode = if self.host_ffn { 6 } else { 1 };
                    for (l, layer) in self.layers.iter().enumerate() {
                        let yb = pre.layer_bo_y[l].as_mut_slice();
                        if layer.is_conv {
                            if need_conv_params {
                                write_lfm2_prefill_step_params(
                                    yb, pos0, n_valid, 0, 0, true, 0, conv_lmode, eps, None,
                                );
                                if let Err(e) = pre.layer_bo_y[l]
                                    .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                                {
                                    Self::flush_pending_attn_mirrors(
                                        &mut pending_attn_mirrors,
                                        layer_kv_blocks,
                                        &mut state.layers,
                                        use_f16,
                                    );
                                    return self.record_device_fault(
                                        format!("prefill c={c} layer {l} conv params sync: {e}"),
                                        out_len,
                                    );
                                }
                            }
                        } else {
                            let rope_cs = Self::resolve_prefill_rope(
                                &mut cached_rope,
                                &mut prefetched_rope,
                                pos0,
                                n_valid,
                                layer.rope_theta,
                            );
                            write_lfm2_prefill_step_params(
                                yb,
                                pos0,
                                n_valid,
                                attn_kv_cap,
                                pos0,
                                true,
                                0,
                                attn_lmode,
                                eps,
                                Some(rope_cs),
                            );
                            let sync_start_f32 = LFM2_PREFILL_SLOT_NORM_0 + 2048;
                            let rope_end_f32 = if n_valid > LFM2_KV_CHUNK_T {
                                LFM2_PREFILL_SLOT_ROPE_PKT_1
                                    + (n_valid - LFM2_KV_CHUNK_T) * LFM2_HEAD_DIM
                            } else {
                                LFM2_PREFILL_SLOT_ROPE_PKT_0 + n_valid * LFM2_HEAD_DIM
                            };
                            if let Err(e) = pre.layer_bo_y[l].sync_range_to_device(
                                (rope_end_f32 - sync_start_f32) * 4,
                                sync_start_f32 * 4,
                            ) {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(
                                    format!("prefill c={c} layer {l} attn params+rope sync: {e}"),
                                    out_len,
                                );
                            }
                        }
                    }
                    pre.last_conv_n_valid = Some(n_valid);

                    let pre_npu0 = self.profile_decode.then(crate::time::Instant::now);
                    let next_rope_spec = if c + 1 < n_chunks {
                        let next_off = (c + 1) * LFM2_PREFILL_CHUNK_T;
                        let next_pos0 = pos_aligned + next_off;
                        default_attn_theta.map(|th| {
                            (
                                next_pos0,
                                (rem_tokens - next_off).min(LFM2_PREFILL_CHUNK_T),
                                th,
                            )
                        })
                    } else {
                        None
                    };

                    if self.host_ffn {
                        for (l, layer) in self.layers.iter().enumerate() {
                            let slot = if layer.is_conv {
                                if pre.moe_conv_runs[parity_idx][l].is_none() {
                                    let (kv_in, kv_out) = if conv_kv_in_kvo {
                                        (&pre.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                                    } else {
                                        (&layer_kv_blocks[l][0], &pre.layer_bo_kvo[l])
                                    };
                                    match kernels.prefill_conv.create_run(&[
                                        &layer_bo_w[l],
                                        &pre.layer_bo_y[l],
                                        &pre.bo_yn,
                                        kv_in,
                                        kv_out,
                                    ]) {
                                        Ok(r) => pre.moe_conv_runs[parity_idx][l] = Some(r),
                                        Err(e) => {
                                            Self::flush_pending_attn_mirrors(
                                                &mut pending_attn_mirrors,
                                                layer_kv_blocks,
                                                &mut state.layers,
                                                use_f16,
                                            );
                                            return self.record_device_fault(
                                                format!("prefill c={c} layer {l} moe conv create_run: {e}"),
                                                out_len,
                                            );
                                        }
                                    }
                                }
                                &mut pre.moe_conv_runs[parity_idx][l]
                            } else {
                                if pre.moe_attn_runs[attn_bidx][l].is_none() {
                                    let built = kernels.prefill_attn(attn_bidx).and_then(|k| {
                                        k.create_run(&[
                                            &layer_bo_w[l],
                                            &pre.layer_bo_y[l],
                                            &pre.bo_yn,
                                            &layer_kv_blocks[l][0],
                                            &pre.layer_bo_kvo[l],
                                        ])
                                        .map_err(|e| format!("moe attn create_run: {e}"))
                                    });
                                    match built {
                                        Ok(r) => pre.moe_attn_runs[attn_bidx][l] = Some(r),
                                        Err(e) => {
                                            Self::flush_pending_attn_mirrors(
                                                &mut pending_attn_mirrors,
                                                layer_kv_blocks,
                                                &mut state.layers,
                                                use_f16,
                                            );
                                            return self.record_device_fault(
                                                format!("prefill c={c} layer {l} {e}"),
                                                out_len,
                                            );
                                        }
                                    }
                                }
                                &mut pre.moe_attn_runs[attn_bidx][l]
                            };

                            let mut next_tok_ok = true;
                            let run_ref = slot.as_ref().expect("populated moe run");
                            let exec_res = match run_ref.start() {
                                Ok(()) => {
                                    if l == 0 {
                                        Self::flush_pending_attn_mirrors(
                                            &mut pending_attn_mirrors,
                                            layer_kv_blocks,
                                            &mut state.layers,
                                            use_f16,
                                        );
                                        if let Some((np0, nnv, th)) = next_rope_spec {
                                            prefetched_rope = Some((
                                                np0,
                                                nnv,
                                                th,
                                                fill_prefill_rope_cs(np0, nnv, th),
                                            ));
                                        }
                                        if let PrefillSource::Tokens(toks) = rem_src
                                            && c + 1 < n_chunks
                                        {
                                            let next_off = (c + 1) * LFM2_PREFILL_CHUNK_T;
                                            let next_nv =
                                                (rem_tokens - next_off).min(LFM2_PREFILL_CHUNK_T);
                                            next_tok_ok = self.cpu.embed_token_rows_into(
                                                &toks[next_off..next_off + next_nv],
                                                &mut prefill_embs_scratch.1[..next_nv * LFM2_H],
                                            );
                                            swapped_next_embs = next_tok_ok;
                                        }
                                    }
                                    match run_ref.wait(QUIESCE_TIMEOUT_MS) {
                                        Ok(_) => Ok(()),
                                        Err(e) => {
                                            if let Some(stalled) = slot.take() {
                                                stalled_runs.push(stalled);
                                            }
                                            Err(e)
                                        }
                                    }
                                }
                                Err(e) => Err(e),
                            };
                            if let Err(e) = exec_res {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(
                                    format!("prefill c={c} layer {l} moe run exec: {e}"),
                                    out_len,
                                );
                            }
                            if !next_tok_ok {
                                return self.reject_input(
                                    format!("token id out of range for vocab {vocab_size}"),
                                    out_len,
                                );
                            }

                            if layer.is_conv {
                                if is_last_chunk {
                                    let kvo_written = if conv_kv_in_kvo {
                                        &mut layer_kv_blocks[l][0]
                                    } else {
                                        &mut pre.layer_bo_kvo[l]
                                    };
                                    if let Err(e) =
                                        kvo_written.sync_range_from_device(LFM2_KVO_BYTES, 0)
                                    {
                                        Self::flush_pending_attn_mirrors(
                                            &mut pending_attn_mirrors,
                                            layer_kv_blocks,
                                            &mut state.layers,
                                            use_f16,
                                        );
                                        return self.record_device_fault(
                                            format!(
                                                "prefill c={c} layer {l} conv kvo readback: {e}"
                                            ),
                                            out_len,
                                        );
                                    }
                                    Self::save_conv_kvo_to_state(
                                        kvo_written,
                                        &mut state.layers[l],
                                        pos0 + n_valid,
                                        true,
                                    );
                                }
                            } else {
                                if let Err(e) = pre.layer_bo_kvo[l]
                                    .sync_range_from_device(attn_kvo_sync_bytes, 0)
                                {
                                    Self::flush_pending_attn_mirrors(
                                        &mut pending_attn_mirrors,
                                        layer_kv_blocks,
                                        &mut state.layers,
                                        use_f16,
                                    );
                                    return self.record_device_fault(
                                        format!("prefill c={c} layer {l} attn kvo readback: {e}"),
                                        out_len,
                                    );
                                }
                                match Self::save_attn_prefill_kvo_to_bo_kv(
                                    kernels,
                                    &pre.layer_bo_kvo[l],
                                    &mut layer_kv_blocks[l],
                                    l,
                                    pos0,
                                    n_valid,
                                ) {
                                    Ok(entry) => {
                                        if is_last_chunk {
                                            Self::mirror_attn_prefill_bo_kv_to_state(
                                                &layer_kv_blocks[l],
                                                &mut state.layers[l],
                                                &entry,
                                                use_f16,
                                            );
                                        } else {
                                            pending_attn_mirrors.push(entry);
                                        }
                                    }
                                    Err(e) => {
                                        Self::flush_pending_attn_mirrors(
                                            &mut pending_attn_mirrors,
                                            layer_kv_blocks,
                                            &mut state.layers,
                                            use_f16,
                                        );
                                        return self.record_device_fault(
                                            format!("prefill c={c} layer {l} attn save: {e}"),
                                            out_len,
                                        );
                                    }
                                }
                            }

                            let next_bo_y = pre.layer_bo_y.get_mut(l + 1);
                            if let Err(e) = self.finish_moe_prefill_layer(
                                &mut pre.bo_yn,
                                next_bo_y,
                                moe_hidden_scratch,
                                moe_prefill_scratch,
                                l,
                                n_valid,
                                state,
                            ) {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(e, out_len);
                            }
                        }
                        if is_last_chunk && need_logits {
                            last_hidden.copy_from_slice(
                                &moe_hidden_scratch[(n_valid - 1) * LFM2_H..n_valid * LFM2_H],
                            );
                        }
                    } else {
                        let chain = pre.bucket_plans[parity_idx][attn_bidx]
                            .as_ref()
                            .expect("built prefill chain");
                        match self.execute_prefill_chain_with_overlap(
                            chain,
                            in_flight,
                            InFlightLfm2Run::PreFast {
                                parity_idx,
                                bidx: attn_bidx,
                            },
                            next_rope_spec,
                            &rem_src,
                            c,
                            n_chunks,
                            rem_tokens,
                            &mut pending_attn_mirrors,
                            layer_kv_blocks,
                            &mut state.layers,
                            use_f16,
                            &mut prefetched_rope,
                            prefill_embs_scratch,
                        ) {
                            Ok((false, _)) => {
                                return self.reject_input(
                                    format!("token id out of range for vocab {vocab_size}"),
                                    out_len,
                                );
                            }
                            Ok((true, swapped)) => swapped_next_embs = swapped,
                            Err(e) => return self.record_device_fault(e, out_len),
                        }
                    }
                    if let Some(t) = pre_npu0 {
                        eprintln!(
                            "lfm2 prefill c={c} n={n_valid} bucket={attn_kv_cap}: segs={:.0}us",
                            t.elapsed().as_secs_f64() * 1e6
                        );
                    }
                }

                let t_kv0 = self.profile_decode.then(crate::time::Instant::now);
                let npu_ref = &mut *npu;
                let pre = npu_ref.prefill.as_mut().expect("prefill buffers");
                let layer_kv_blocks = &mut npu_ref.layer_kv_blocks;
                let conv_kv_in_kvo = pre.conv_kv_in_kvo;
                if !self.host_ffn
                    && let Err(e) = Self::sync_and_save_prefill_kvo_outputs(
                        &npu_ref.kernels,
                        &self.layers,
                        &mut pre.layer_bo_kvo,
                        layer_kv_blocks,
                        &mut state.layers,
                        &mut pending_attn_mirrors,
                        conv_kv_in_kvo,
                        is_last_chunk,
                        pos0,
                        n_valid,
                        attn_kvo_sync_bytes,
                        use_f16,
                        c,
                    )
                {
                    return self.record_device_fault(e, out_len);
                }
                pre.conv_kv_in_kvo = !conv_kv_in_kvo;
                if let Some(t) = t_kv0 {
                    eprintln!(
                        "lfm2 prefill c={c} kv_save={:.0}us",
                        t.elapsed().as_secs_f64() * 1e6
                    );
                }
                if swapped_next_embs {
                    let s = &mut npu.prefill_embs_scratch;
                    std::mem::swap(&mut s.0, &mut s.1);
                }
            } else {
                // >32K paged prefill chunk on q4prefill_lfm2_ext.xclbin (all layers in one runlist for !host_ffn).
                let cur_blk = pos0 / LFM2_BLOCK_TOKENS;
                let max_kv_blk = (pos0 + n_valid - 1) / LFM2_BLOCK_TOKENS;
                let fin_past_chunks = past_chunks_c.saturating_sub(cur_blk * LFM2_MAX_KV_CHUNKS);
                let (fin_bidx, fin_cap) =
                    match try_select_lfm3_kv_bucket(&fin_caps, fin_past_chunks) {
                        Ok(v) => v,
                        Err(e) => {
                            Self::flush_pending_attn_mirrors(
                                &mut pending_attn_mirrors,
                                &npu.layer_kv_blocks,
                                &mut state.layers,
                                use_f16,
                            );
                            return self.record_device_fault(e, out_len);
                        }
                    };
                if !self.host_ffn {
                    let swapped_next_embs;
                    let pre_npu0 = self.profile_decode.then(crate::time::Instant::now);
                    {
                        let XdnaLfm2State {
                            ref mut kernels,
                            ref layer_bo_w,
                            ref mut layer_kv_blocks,
                            ref mut prefill,
                            ref mut in_flight,
                            ref mut stalled_runs,
                            ref mut prefill_embs_scratch,
                            ..
                        } = *npu;
                        let pre = prefill.as_mut().expect("initialized prefill buffers");
                        let chunk_embs: &[f32] = match rem_src {
                            PrefillSource::Tokens(_) => &prefill_embs_scratch.0[..n_valid * LFM2_H],
                            PrefillSource::Embeddings(embeddings, _) => {
                                &embeddings[local_off * LFM2_H..(local_off + n_valid) * LFM2_H]
                            }
                        };
                        if pre.ext_runs.is_none() {
                            pre.ext_runs = Some(Lfm2ExtRunCache::new(n_layers));
                        }
                        for (l, layer) in self.layers.iter().enumerate() {
                            if !layer.is_conv
                                && let Err(e) = Self::ensure_paged_block(
                                    kernels,
                                    &mut layer_kv_blocks[l],
                                    max_kv_blk,
                                    l,
                                )
                            {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(e, out_len);
                            }
                        }
                        let conv_kv_in_kvo = pre.conv_kv_in_kvo;
                        let parity_idx = usize::from(conv_kv_in_kvo);
                        let chain_missing = cur_blk <= 2
                            && pre.ext_runs.as_ref().expect("ext cache").chains[parity_idx]
                                .iter()
                                .all(|(b, f, _)| *b != cur_blk || *f != fin_bidx);
                        if chain_missing {
                            match Self::build_prefill_ext_chain(
                                kernels,
                                &self.layers,
                                layer_bo_w,
                                layer_kv_blocks,
                                pre,
                                cur_blk,
                                fin_bidx,
                                conv_kv_in_kvo,
                            ) {
                                Ok(ch) => {
                                    pre.ext_runs.as_mut().expect("ext cache").chains[parity_idx]
                                        .push((cur_blk, fin_bidx, ch));
                                }
                                Err(e) => {
                                    Self::flush_pending_attn_mirrors(
                                        &mut pending_attn_mirrors,
                                        layer_kv_blocks,
                                        &mut state.layers,
                                        use_f16,
                                    );
                                    return self.record_device_fault(e, out_len);
                                }
                            }
                        }
                        if let Err(e) = Self::sync_and_write_prefill_hidden(
                            &mut pre.layer_bo_y[0],
                            chunk_embs,
                            n_valid,
                            &format!("prefill >32K c={c} layer 0 hidden sync"),
                        ) {
                            Self::flush_pending_attn_mirrors(
                                &mut pending_attn_mirrors,
                                layer_kv_blocks,
                                &mut state.layers,
                                use_f16,
                            );
                            return self.record_device_fault(e, out_len);
                        }
                        let need_conv_params = pre.last_conv_n_valid != Some(n_valid);
                        for (l, layer) in self.layers.iter().enumerate() {
                            if layer.is_conv {
                                if need_conv_params {
                                    write_lfm2_prefill_step_params(
                                        pre.layer_bo_y[l].as_mut_slice(),
                                        pos0,
                                        n_valid,
                                        0,
                                        0,
                                        true,
                                        0,
                                        0,
                                        eps,
                                        None,
                                    );
                                    if let Err(e) = pre.layer_bo_y[l].sync_range_to_device(
                                        64,
                                        (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4,
                                    ) {
                                        Self::flush_pending_attn_mirrors(
                                            &mut pending_attn_mirrors,
                                            layer_kv_blocks,
                                            &mut state.layers,
                                            use_f16,
                                        );
                                        return self.record_device_fault(
                                            format!(
                                                "prefill >32K c={c} layer {l} conv params sync: {e}"
                                            ),
                                            out_len,
                                        );
                                    }
                                }
                            } else if let Err(e) = Self::sync_ext_attn_layer_params(
                                layer,
                                l,
                                c,
                                &mut pre.layer_bo_y[l],
                                &mut cached_rope,
                                &mut prefetched_rope,
                                pos0,
                                n_valid,
                                eps,
                                cur_blk,
                                fin_cap,
                                5,
                            ) {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(e, out_len);
                            }
                        }
                        pre.last_conv_n_valid = Some(n_valid);

                        let next_rope_spec = if c + 1 < n_chunks {
                            let next_off = (c + 1) * LFM2_PREFILL_CHUNK_T;
                            let next_pos0 = pos_aligned + next_off;
                            default_attn_theta.map(|th| {
                                (
                                    next_pos0,
                                    (rem_tokens - next_off).min(LFM2_PREFILL_CHUNK_T),
                                    th,
                                )
                            })
                        } else {
                            None
                        };
                        if cur_blk >= 3 {
                            Self::flush_pending_attn_mirrors(
                                &mut pending_attn_mirrors,
                                layer_kv_blocks,
                                &mut state.layers,
                                use_f16,
                            );
                            for (l, layer) in self.layers.iter().enumerate() {
                                let (cur_bo_y, y_next): (&mut XdnaBo, &XdnaBo) = if l + 1 < n_layers
                                {
                                    let (head, tail) = pre.layer_bo_y.split_at_mut(l + 1);
                                    (&mut head[l], &tail[0])
                                } else {
                                    (&mut pre.layer_bo_y[l], &pre.bo_yn)
                                };
                                let layer_res = if layer.is_conv {
                                    let (kv_in, kv_out) = if conv_kv_in_kvo {
                                        (&pre.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                                    } else {
                                        (&layer_kv_blocks[l][0], &pre.layer_bo_kvo[l])
                                    };
                                    kernels
                                        .prefill_ext_conv()
                                        .and_then(|k| {
                                            k.create_run(&[
                                                &layer_bo_w[l],
                                                cur_bo_y,
                                                y_next,
                                                kv_in,
                                                kv_out,
                                            ])
                                            .map_err(|e| {
                                                format!(
                                                    "prefill >96K c={c} layer {l} conv create_run: {e}"
                                                )
                                            })
                                        })
                                        .and_then(|run| {
                                            exec_owned_run(
                                                stalled_runs,
                                                run,
                                                &format!("prefill >96K c={c} layer {l} conv exec"),
                                            )
                                        })
                                } else {
                                    Self::run_prefill_ext_attn_multi_block_layer(
                                        kernels,
                                        l,
                                        c,
                                        &layer_bo_w[l],
                                        cur_bo_y,
                                        y_next,
                                        &layer_kv_blocks[l],
                                        &pre.layer_bo_kvo[l],
                                        stalled_runs,
                                        pos0,
                                        n_valid,
                                        eps,
                                        cur_blk,
                                        fin_bidx,
                                    )
                                };
                                if let Err(e) = layer_res {
                                    return self.record_device_fault(e, out_len);
                                }
                            }
                            if let Some((np0, nnv, th)) = next_rope_spec {
                                prefetched_rope =
                                    Some((np0, nnv, th, fill_prefill_rope_cs(np0, nnv, th)));
                            }
                            if let PrefillSource::Tokens(toks) = rem_src
                                && c + 1 < n_chunks
                            {
                                let next_off = (c + 1) * LFM2_PREFILL_CHUNK_T;
                                let next_nv = (rem_tokens - next_off).min(LFM2_PREFILL_CHUNK_T);
                                if !self.cpu.embed_token_rows_into(
                                    &toks[next_off..next_off + next_nv],
                                    &mut prefill_embs_scratch.0[..next_nv * LFM2_H],
                                ) {
                                    return self.reject_input(
                                        format!("token id out of range for vocab {vocab_size}"),
                                        out_len,
                                    );
                                }
                            }
                            swapped_next_embs = false;
                        } else {
                            let chain = &pre.ext_runs.as_ref().expect("ext cache").chains
                                [parity_idx]
                                .iter()
                                .find(|(b, f, _)| *b == cur_blk && *f == fin_bidx)
                                .expect("built ext chain")
                                .2;
                            match self.execute_prefill_chain_with_overlap(
                                chain,
                                in_flight,
                                InFlightLfm2Run::PreExt {
                                    parity_idx,
                                    key: (cur_blk, fin_bidx),
                                },
                                next_rope_spec,
                                &rem_src,
                                c,
                                n_chunks,
                                rem_tokens,
                                &mut pending_attn_mirrors,
                                layer_kv_blocks,
                                &mut state.layers,
                                use_f16,
                                &mut prefetched_rope,
                                prefill_embs_scratch,
                            ) {
                                Ok((false, _)) => {
                                    return self.reject_input(
                                        format!("token id out of range for vocab {vocab_size}"),
                                        out_len,
                                    );
                                }
                                Ok((true, swapped)) => swapped_next_embs = swapped,
                                Err(e) => return self.record_device_fault(e, out_len),
                            }
                        }
                    }
                    if let Some(t) = pre_npu0 {
                        eprintln!(
                            "lfm2 prefill >32K c={c} n={n_valid} bucket={}: segs={:.0}us",
                            cur_blk * LFM2_MAX_KV_CHUNKS + fin_cap,
                            t.elapsed().as_secs_f64() * 1e6
                        );
                    }
                    let t_kv0 = self.profile_decode.then(crate::time::Instant::now);
                    let npu_ref = &mut *npu;
                    let pre = npu_ref.prefill.as_mut().expect("prefill buffers");
                    let layer_kv_blocks = &mut npu_ref.layer_kv_blocks;
                    let conv_kv_in_kvo = pre.conv_kv_in_kvo;
                    if let Err(e) = Self::sync_and_save_prefill_kvo_outputs(
                        &npu_ref.kernels,
                        &self.layers,
                        &mut pre.layer_bo_kvo,
                        layer_kv_blocks,
                        &mut state.layers,
                        &mut pending_attn_mirrors,
                        conv_kv_in_kvo,
                        is_last_chunk,
                        pos0,
                        n_valid,
                        attn_kvo_sync_bytes,
                        use_f16,
                        c,
                    ) {
                        return self.record_device_fault(e, out_len);
                    }
                    pre.conv_kv_in_kvo = !conv_kv_in_kvo;
                    if let Some(t) = t_kv0 {
                        eprintln!(
                            "lfm2 prefill >32K c={c} kv_save={:.0}us",
                            t.elapsed().as_secs_f64() * 1e6
                        );
                    }
                    if swapped_next_embs {
                        let s = &mut npu.prefill_embs_scratch;
                        std::mem::swap(&mut s.0, &mut s.1);
                    }
                } else {
                    let npu_ref = &mut *npu;
                    let XdnaLfm2State {
                        ref mut kernels,
                        ref layer_bo_w,
                        ref mut layer_kv_blocks,
                        ref mut prefill,
                        ref mut in_flight,
                        ref mut stalled_runs,
                        ref mut prefill_embs_scratch,
                        ref mut moe_hidden_scratch,
                        ref mut moe_prefill_scratch,
                        ..
                    } = *npu_ref;
                    let pre = prefill.as_mut().expect("initialized prefill buffers");
                    pre.last_conv_n_valid = None;
                    let chunk_embs: &[f32] = match rem_src {
                        PrefillSource::Tokens(_) => &prefill_embs_scratch.0[..n_valid * LFM2_H],
                        PrefillSource::Embeddings(embeddings, _) => {
                            &embeddings[local_off * LFM2_H..(local_off + n_valid) * LFM2_H]
                        }
                    };
                    if pre.ext_runs.is_none() {
                        pre.ext_runs = Some(Lfm2ExtRunCache::new(n_layers));
                    }
                    for (l, layer) in self.layers.iter().enumerate() {
                        if !layer.is_conv
                            && let Err(e) = Self::ensure_paged_block(
                                kernels,
                                &mut layer_kv_blocks[l],
                                max_kv_blk,
                                l,
                            )
                        {
                            return self.record_device_fault(e, out_len);
                        }
                    }
                    let conv_kv_in_kvo = pre.conv_kv_in_kvo;
                    let parity_idx = usize::from(conv_kv_in_kvo);
                    if let Err(e) = Self::sync_and_write_prefill_hidden(
                        &mut pre.layer_bo_y[0],
                        chunk_embs,
                        n_valid,
                        &format!("prefill >32K c={c} layer 0 hidden sync"),
                    ) {
                        return self.record_device_fault(e, out_len);
                    }
                    if let PrefillSource::Tokens(toks) = rem_src
                        && c + 1 < n_chunks
                    {
                        let next_off = (c + 1) * LFM2_PREFILL_CHUNK_T;
                        let next_nv = (rem_tokens - next_off).min(LFM2_PREFILL_CHUNK_T);
                        if !self.cpu.embed_token_rows_into(
                            &toks[next_off..next_off + next_nv],
                            &mut prefill_embs_scratch.0[..next_nv * LFM2_H],
                        ) {
                            return self.reject_input(
                                format!("token id out of range for vocab {vocab_size}"),
                                out_len,
                            );
                        }
                    }

                    let t_ext0 = self.profile_decode.then(crate::time::Instant::now);
                    for l in 0..n_layers {
                        if self.layers[l].is_conv {
                            write_lfm2_prefill_step_params(
                                pre.layer_bo_y[l].as_mut_slice(),
                                pos0,
                                n_valid,
                                0,
                                0,
                                true,
                                0,
                                7,
                                eps,
                                None,
                            );
                            if let Err(e) = pre.layer_bo_y[l]
                                .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                            {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(
                                    format!("prefill >32K c={c} layer {l} conv params sync: {e}"),
                                    out_len,
                                );
                            }
                            let ext = pre.ext_runs.as_mut().expect("ext cache");
                            if ext.conv_groups[parity_idx][l].is_none() {
                                let (kv_in, kv_out) = if conv_kv_in_kvo {
                                    (&pre.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                                } else {
                                    (&layer_kv_blocks[l][0], &pre.layer_bo_kvo[l])
                                };
                                let built = kernels.prefill_ext_conv().and_then(|k| {
                                    let run = k
                                        .create_run(&[
                                            &layer_bo_w[l],
                                            &pre.layer_bo_y[l],
                                            &pre.bo_yn,
                                            kv_in,
                                            kv_out,
                                        ])
                                        .map_err(|e| {
                                            format!("prefill >32K layer {l} conv create_run: {e}")
                                        })?;
                                    Self::assemble_prefill_runlist(k, vec![run], "prefill_ext_conv")
                                });
                                match built {
                                    Ok(grp) => ext.conv_groups[parity_idx][l] = Some(grp),
                                    Err(e) => {
                                        Self::flush_pending_attn_mirrors(
                                            &mut pending_attn_mirrors,
                                            layer_kv_blocks,
                                            &mut state.layers,
                                            use_f16,
                                        );
                                        return self.record_device_fault(e, out_len);
                                    }
                                }
                            }
                            let grp = ext.conv_groups[parity_idx][l]
                                .as_ref()
                                .expect("populated conv group");
                            *in_flight = Some(InFlightLfm2Run::PreExtConv { parity_idx, l });
                            let exec_res = match grp.runlist.execute() {
                                Ok(()) => {
                                    Self::flush_pending_attn_mirrors(
                                        &mut pending_attn_mirrors,
                                        layer_kv_blocks,
                                        &mut state.layers,
                                        use_f16,
                                    );
                                    let w = grp.runlist.wait(QUIESCE_TIMEOUT_MS);
                                    if w.is_ok() {
                                        *in_flight = None;
                                    }
                                    w
                                }
                                Err(e) => {
                                    *in_flight = None;
                                    Err(e)
                                }
                            };
                            if let Err(e) = exec_res {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(
                                    format!("prefill >32K c={c} layer {l} conv exec: {e}"),
                                    out_len,
                                );
                            }
                            if is_last_chunk {
                                let kvo_written = if conv_kv_in_kvo {
                                    &mut layer_kv_blocks[l][0]
                                } else {
                                    &mut pre.layer_bo_kvo[l]
                                };
                                if let Err(e) =
                                    kvo_written.sync_range_from_device(LFM2_KVO_BYTES, 0)
                                {
                                    Self::flush_pending_attn_mirrors(
                                        &mut pending_attn_mirrors,
                                        layer_kv_blocks,
                                        &mut state.layers,
                                        use_f16,
                                    );
                                    return self.record_device_fault(
                                        format!(
                                            "prefill >32K c={c} layer {l} conv kvo readback: {e}"
                                        ),
                                        out_len,
                                    );
                                }
                                Self::save_conv_kvo_to_state(
                                    kvo_written,
                                    &mut state.layers[l],
                                    pos0 + n_valid,
                                    true,
                                );
                            }
                        } else {
                            if let Err(e) = Self::sync_ext_attn_layer_params(
                                &self.layers[l],
                                l,
                                c,
                                &mut pre.layer_bo_y[l],
                                &mut cached_rope,
                                &mut prefetched_rope,
                                pos0,
                                n_valid,
                                eps,
                                cur_blk,
                                fin_cap,
                                6,
                            ) {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                return self.record_device_fault(e, out_len);
                            }
                            if cur_blk >= 3 {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                if let Err(e) = Self::run_prefill_ext_attn_multi_block_layer(
                                    kernels,
                                    l,
                                    c,
                                    &layer_bo_w[l],
                                    &mut pre.layer_bo_y[l],
                                    &pre.bo_yn,
                                    &layer_kv_blocks[l],
                                    &pre.layer_bo_kvo[l],
                                    stalled_runs,
                                    pos0,
                                    n_valid,
                                    eps,
                                    cur_blk,
                                    fin_bidx,
                                ) {
                                    return self.record_device_fault(e, out_len);
                                }
                            } else {
                                let ext = pre.ext_runs.as_mut().expect("ext cache");
                                let chain_missing = ext.attn_chains[l]
                                    .iter()
                                    .all(|(b, f, _)| *b != cur_blk || *f != fin_bidx);
                                if chain_missing {
                                    match Self::build_prefill_ext_attn_layer_chain(
                                        kernels,
                                        l,
                                        &layer_bo_w[l],
                                        &pre.layer_bo_y[l],
                                        &pre.bo_yn,
                                        &layer_kv_blocks[l],
                                        &pre.layer_bo_kvo[l],
                                        cur_blk,
                                        fin_bidx,
                                    ) {
                                        Ok(ch) => ext.attn_chains[l].push((cur_blk, fin_bidx, ch)),
                                        Err(e) => {
                                            Self::flush_pending_attn_mirrors(
                                                &mut pending_attn_mirrors,
                                                layer_kv_blocks,
                                                &mut state.layers,
                                                use_f16,
                                            );
                                            return self.record_device_fault(e, out_len);
                                        }
                                    }
                                }
                                let chain = &ext.attn_chains[l]
                                    .iter()
                                    .find(|(b, f, _)| *b == cur_blk && *f == fin_bidx)
                                    .expect("built attn layer ext chain")
                                    .2;
                                *in_flight = Some(InFlightLfm2Run::PreExtAttn {
                                    l,
                                    blk: cur_blk,
                                    fin_bidx,
                                });
                                let exec_res = match chain.runlist.execute() {
                                    Ok(()) => {
                                        Self::flush_pending_attn_mirrors(
                                            &mut pending_attn_mirrors,
                                            layer_kv_blocks,
                                            &mut state.layers,
                                            use_f16,
                                        );
                                        let w = chain.runlist.wait(QUIESCE_TIMEOUT_MS);
                                        if w.is_ok() {
                                            *in_flight = None;
                                        }
                                        w
                                    }
                                    Err(e) => {
                                        *in_flight = None;
                                        Err(e)
                                    }
                                };
                                if let Err(e) = exec_res {
                                    Self::flush_pending_attn_mirrors(
                                        &mut pending_attn_mirrors,
                                        layer_kv_blocks,
                                        &mut state.layers,
                                        use_f16,
                                    );
                                    return self.record_device_fault(
                                        format!("prefill >32K c={c} layer {l} attn exec: {e}"),
                                        out_len,
                                    );
                                }
                            }
                            if let Err(e) =
                                pre.layer_bo_kvo[l].sync_range_from_device(attn_kvo_sync_bytes, 0)
                            {
                                return self.record_device_fault(
                                    format!("prefill >32K c={c} layer {l} attn kvo readback: {e}"),
                                    out_len,
                                );
                            }
                            match Self::save_attn_prefill_kvo_to_bo_kv(
                                kernels,
                                &pre.layer_bo_kvo[l],
                                &mut layer_kv_blocks[l],
                                l,
                                pos0,
                                n_valid,
                            ) {
                                Ok(entry) => {
                                    if is_last_chunk {
                                        Self::mirror_attn_prefill_bo_kv_to_state(
                                            &layer_kv_blocks[l],
                                            &mut state.layers[l],
                                            &entry,
                                            use_f16,
                                        );
                                    } else {
                                        pending_attn_mirrors.push(entry);
                                    }
                                }
                                Err(e) => {
                                    return self.record_device_fault(
                                        format!("prefill >32K c={c} layer {l} attn save: {e}"),
                                        out_len,
                                    );
                                }
                            }
                        }
                        let next_bo_y = pre.layer_bo_y.get_mut(l + 1);
                        if let Err(e) = self.finish_moe_prefill_layer(
                            &mut pre.bo_yn,
                            next_bo_y,
                            moe_hidden_scratch,
                            moe_prefill_scratch,
                            l,
                            n_valid,
                            state,
                        ) {
                            return self.record_device_fault(e, out_len);
                        }
                    }
                    pre.conv_kv_in_kvo = !conv_kv_in_kvo;
                    if is_last_chunk && need_logits {
                        last_hidden.copy_from_slice(
                            &moe_hidden_scratch[(n_valid - 1) * LFM2_H..n_valid * LFM2_H],
                        );
                    }
                    if let Some(t) = t_ext0 {
                        eprintln!(
                            "lfm2 prefill >32K c={c} n={n_valid} bucket={}: total={:.0}us",
                            cur_blk * LFM2_MAX_KV_CHUNKS + fin_cap,
                            t.elapsed().as_secs_f64() * 1e6
                        );
                    }
                }
            }

            let pre = npu.prefill.as_mut().expect("prefill buffers");
            if !self.host_ffn && is_last_chunk && need_logits {
                let last_bytes = LFM2_H * 4;
                let last_off = (n_valid - 1) * last_bytes;
                if let Err(e) = pre.bo_yn.sync_range_from_device(last_bytes, last_off) {
                    return self.record_device_fault(
                        format!("prefill last_hidden sync_from_device: {e}"),
                        out_len,
                    );
                }
                read_le_f32_slice(
                    pre.bo_yn.as_slice(),
                    (n_valid - 1) * LFM2_H,
                    &mut last_hidden,
                );
            }
        }

        if let Err(e) = self.flush_prefill_conv_pingpong(&mut npu) {
            return self.record_device_fault(e, out_len);
        }

        state.seq_len = total_tokens;
        npu.synced_len = total_tokens;
        if !need_logits {
            return Vec::new();
        }
        let head_scratch = &mut npu.head_q8_scratch;
        self.cpu.xdna_project_logits_from_hidden_with_scratch(
            &last_hidden,
            Some((&mut head_scratch.0, &mut head_scratch.1)),
        )
    }

    fn run_single_embedding_forward(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        if self.state_torn.load(Ordering::SeqCst) {
            return self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                vocab_size,
            );
        }
        if embedding.len() != LFM2_H {
            return self.reject_input(
                format!("embedding length {} != {LFM2_H}", embedding.len()),
                vocab_size,
            );
        }
        if pos >= cfg.max_seq_len || pos >= LFM2_MAX_TOKENS {
            return self.reject_input(
                format!(
                    "pos {pos} exceeds context length {}",
                    cfg.max_seq_len.min(LFM2_MAX_TOKENS)
                ),
                vocab_size,
            );
        }
        if let Err(e) = self.validate_state(state) {
            return self.reject_input(e, vocab_size);
        }
        let use_f16 = state.kv_f16;
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            return self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                vocab_size,
            );
        }
        if let Err(e) =
            self.validate_cpu_state_for_sync(state, npu.pending_kv.as_ref(), pos, use_f16)
        {
            return self.reject_input(e, vocab_size);
        }
        if let Err(e) = self.flush_prefill_conv_pingpong(&mut npu) {
            return self.record_device_fault(e, vocab_size);
        }
        match self.sync_cpu_state_to_npu(&mut npu, state, pos, use_f16) {
            Ok(()) => {}
            Err(SyncErr::Input(e)) => return self.reject_input(e, vocab_size),
            Err(SyncErr::Device(e)) => return self.record_device_fault(e, vocab_size),
        }
        match self.run_npu_decode_embedding_step(&mut npu, embedding, pos, state) {
            Ok(logits) => logits,
            Err(e) => self.record_device_fault(e, vocab_size),
        }
    }
}

impl Model for XdnaLfm2Model {
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        let lease = self.session_gate.try_acquire()?;
        let mut npu = self.lock_npu();
        self.reset_npu_session_state(&mut npu)?;
        Ok(Some(lease))
    }

    fn take_decode_error(&self) -> Option<CeraError> {
        take_fault(&self.decode_error)
    }

    fn backend_label(&self) -> &'static str {
        "xdna"
    }

    fn f16_kv_supported(&self) -> bool {
        true
    }

    fn configure_kv_compression(&self, compression: &KvCompression) -> Result<(), CeraError> {
        validate_xdna_kv_compression(self.cpu.config(), compression)?;
        self.cpu.configure_kv_compression(compression)
    }

    fn supports_hidden_states(&self) -> bool {
        self.cpu.supports_hidden_states()
    }

    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        let cfg = self.cpu.config();
        if tokens.is_empty() {
            return self.reject_input_err(CeraError::EmptyInput, 0);
        }
        let safe_out_len = tokens.len().min(cfg.max_seq_len) * cfg.hidden_size;
        if tokens.len() > cfg.max_seq_len {
            return self.reject_input_err(
                CeraError::ContextOverflow {
                    max_seq_len: cfg.max_seq_len as u32,
                    by: (tokens.len() - cfg.max_seq_len) as u32,
                },
                safe_out_len,
            );
        }
        if state.is_compressed() {
            return self.reject_input(
                "TurboQuant KV compression is not supported by the XDNA backend".to_string(),
                safe_out_len,
            );
        }
        if let Some(&bad_tok) = tokens.iter().find(|&&t| (t as usize) >= cfg.vocab_size) {
            return self.reject_input_err(
                CeraError::InvalidToken {
                    id: bad_tok,
                    vocab_size: cfg.vocab_size as u32,
                },
                safe_out_len,
            );
        }
        {
            let mut npu = self.lock_npu();
            if self.state_torn.load(Ordering::SeqCst) {
                return self.record_device_fault(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                    safe_out_len,
                );
            }
            if let Err(e) = npu.quiesce_in_flight() {
                return self.record_device_fault(
                    format!("hidden_states failed to quiesce NPU: {e}"),
                    safe_out_len,
                );
            }
        }
        let mut isolated = match InferenceState::from_config(cfg) {
            Ok(s) => s,
            Err(e) => {
                return self.reject_input_err(
                    CeraError::Backend(format!(
                        "failed to allocate isolated CPU state for hidden_states: {e}"
                    )),
                    safe_out_len,
                );
            }
        };
        isolated.lora = state.lora.clone();
        self.cpu.hidden_states(tokens, &mut isolated)
    }

    fn supports_embedding_input(&self) -> bool {
        true
    }

    fn embed_token_rows(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.cpu.embed_token_rows(tokens)
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.run_single_embedding_forward(embedding, pos, state)
    }

    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.run_npu_prefill_embeddings(embeddings, n_tokens, start_pos, state, true)
    }

    fn forward_prefill_from_embeddings_chunk(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
    ) -> Vec<f32> {
        self.run_npu_prefill_embeddings(embeddings, n_tokens, start_pos, state, need_logits)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let vocab_size = self.cpu.config().vocab_size;
        if tokens.len() != 1 {
            return self.reject_input(
                format!("XDNA LFM2 decode requires 1 token, got {}", tokens.len()),
                vocab_size,
            );
        }
        let Some(emb) = self.cpu.embed_token_rows(tokens) else {
            return self.reject_input(
                format!("token id {} out of range for vocab {vocab_size}", tokens[0]),
                vocab_size,
            );
        };
        self.run_single_embedding_forward(&emb, pos, state)
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        if tokens.len() != 1
            || self.state_torn.load(Ordering::SeqCst)
            || pos >= cfg.max_seq_len
            || pos >= LFM2_MAX_TOKENS
            || self.validate_state(state).is_err()
        {
            let logits = self.forward(tokens, pos, state);
            return crate::sampler::argmax(&logits);
        }
        let Some(emb) = self.cpu.embed_token_rows(tokens) else {
            let _ = self.reject_input(
                format!("token id {} out of range for vocab {vocab_size}", tokens[0]),
                vocab_size,
            );
            return 0;
        };
        let use_f16 = state.kv_f16;
        if state.scratch.logits.len() < vocab_size {
            state.scratch.logits.resize(vocab_size, 0.0);
        }
        let mut logits_buf = std::mem::take(&mut state.scratch.logits);
        let tok = {
            let mut npu = self.lock_npu();
            if self.state_torn.load(Ordering::SeqCst) {
                let _ = self.reject_input_err(
                    CeraError::Backend(
                        "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                    ),
                    vocab_size,
                );
                0
            } else if let Err(e) =
                self.validate_cpu_state_for_sync(state, npu.pending_kv.as_ref(), pos, use_f16)
            {
                let _ = self.reject_input(e, vocab_size);
                0
            } else if let Err(e) = self.flush_prefill_conv_pingpong(&mut npu) {
                let _ = self.record_device_fault(e, vocab_size);
                0
            } else if let Err(err) = self.sync_cpu_state_to_npu(&mut npu, state, pos, use_f16) {
                let _ = match err {
                    SyncErr::Input(e) => self.reject_input(e, vocab_size),
                    SyncErr::Device(e) => self.record_device_fault(e, vocab_size),
                };
                0
            } else {
                match self.run_npu_decode_embedding_step_into(
                    &mut npu,
                    &emb,
                    pos,
                    state,
                    &mut logits_buf[..vocab_size],
                ) {
                    Ok(()) => crate::sampler::argmax(&logits_buf[..vocab_size]),
                    Err(e) => {
                        let _ = self.record_device_fault(e, vocab_size);
                        0
                    }
                }
            }
        };
        state.scratch.logits = logits_buf;
        tok
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.run_npu_prefill(PrefillSource::Tokens(tokens), start_pos, state, true)
    }

    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        ubatch: usize,
        cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        crate::model::run_chunked_prefill(tokens, start_pos, ubatch, cancel, |chunk, pos| {
            let need_logits = (pos - start_pos) + chunk.len() == tokens.len();
            let logits =
                self.run_npu_prefill(PrefillSource::Tokens(chunk), pos, state, need_logits);
            if self
                .decode_error
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .is_some()
            {
                (0, Some(logits))
            } else {
                (chunk.len(), Some(logits))
            }
        })
    }

    fn supports_kv_shift(&self) -> bool {
        self.cpu.supports_kv_shift()
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            let _ = self.record_device_fault("shift_kv called with torn NPU state".to_string(), 0);
            return;
        }
        if let Err(e) = npu.quiesce_in_flight() {
            let _ = self.record_device_fault(e, 0);
            return;
        }
        if !npu.pending_kv.flush_into(state) {
            npu.synced_len = 0;
        }
        self.cpu.shift_kv(state, n_keep, shift);
        npu.synced_len = npu.synced_len.min(n_keep).min(state.seq_len);
        npu.force_conv_sync = true;
    }

    fn config(&self) -> &ModelConfig {
        self.cpu.config()
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        assert!(
            len <= state.seq_len,
            "truncate_kv({len}) exceeds seq_len {}",
            state.seq_len
        );
        if len == state.seq_len {
            return;
        }
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            let _ =
                self.record_device_fault("truncate_kv called with torn NPU state".to_string(), 0);
            return;
        }
        if let Err(e) = npu.quiesce_in_flight() {
            let _ = self.record_device_fault(e, 0);
            return;
        }
        if !npu.pending_kv.discard_for_truncate(state, len) {
            npu.synced_len = 0;
        }
        if len < state.seq_len {
            self.cpu.truncate_kv(state, len);
        }
        npu.synced_len = npu.synced_len.min(len);
        npu.force_conv_sync = true;
    }

    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        let npu = self.lock_npu();
        check_hybrid_kv_rewind_state(
            self.cpu.config().is_causal,
            self.state_torn.load(Ordering::SeqCst),
            npu.pending_kv.as_ref(),
            state,
            len,
        )
    }

    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        let mut npu = self.lock_npu();
        check_hybrid_kv_rewind_state(
            self.cpu.config().is_causal,
            self.state_torn.load(Ordering::SeqCst),
            npu.pending_kv.as_ref(),
            state,
            len,
        )?;
        if len == state.seq_len {
            return Ok(());
        }
        if let Err(e) = npu.quiesce_in_flight() {
            let _ = self.record_device_fault(e, 0);
            return Err(KvRewindError::BackendUnsupported);
        }
        if !npu.pending_kv.discard_for_truncate(state, len) {
            npu.synced_len = 0;
        }
        self.cpu.try_truncate_kv(state, len)?;
        npu.synced_len = npu.synced_len.min(len);
        npu.force_conv_sync = true;
        Ok(())
    }

    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        validate_xdna_kv_compression(self.cpu.config(), compression)?;
        let mut npu = self.lock_npu();
        let mut fresh =
            InferenceState::from_config_capped(self.cpu.config(), compression, max_seq_len)?;
        fresh.lora = state.lora.clone();
        self.reset_npu_session_state(&mut npu)?;
        *state = fresh;
        Ok(())
    }
}

impl std::fmt::Debug for XdnaLfm2Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XdnaLfm2Model")
            .field("config", self.cpu.config())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::xdna::sys::fake::{Config as FakeConfig, Event, with_shim};
    use crate::gguf::GgufBuilder;

    fn make_q4_0_bytes(rows: usize, cols: usize) -> Vec<u8> {
        vec![0u8; rows * (cols / 32) * 18]
    }

    fn tiny_lfm2_gguf() -> GgufFile {
        tiny_lfm2_gguf_with_ctx(128)
    }

    fn tiny_lfm2_gguf_with_ctx(ctx_len: u32) -> GgufFile {
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "lfm2")
            .kv_u32("lfm2.block_count", 2)
            .kv_u32("lfm2.embedding_length", LFM2_H as u32)
            .kv_u32("lfm2.feed_forward_length", LFM2_FFN as u32)
            .kv_u32("lfm2.attention.head_count", LFM2_N_HEADS as u32)
            .kv_i32_array(
                "lfm2.attention.head_count_kv",
                vec![0, LFM2_N_KV_HEADS as i32],
            )
            .kv_u32("lfm2.vocab_size", 64)
            .kv_u32("lfm2.context_length", ctx_len)
            .kv_f32("lfm2.rope.freq_base", 1_000_000.0)
            .kv_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("lfm2.shortconv.l_cache", LFM2_CONV_TAPS as u32);

        b = b.tensor(
            "token_embd.weight",
            &[LFM2_H, 64],
            2,
            make_q4_0_bytes(64, LFM2_H),
        );
        b = b.tensor_f32("token_embd_norm.weight", &[LFM2_H], &vec![1.0f32; LFM2_H]);

        for l in 0..2 {
            b = b
                .tensor_f32(
                    format!("blk.{l}.attn_norm.weight"),
                    &[LFM2_H],
                    &vec![1.0; LFM2_H],
                )
                .tensor_f32(
                    format!("blk.{l}.ffn_norm.weight"),
                    &[LFM2_H],
                    &vec![1.0; LFM2_H],
                )
                .tensor(
                    format!("blk.{l}.ffn_gate.weight"),
                    &[LFM2_H, LFM2_FFN],
                    2,
                    make_q4_0_bytes(LFM2_FFN, LFM2_H),
                )
                .tensor(
                    format!("blk.{l}.ffn_up.weight"),
                    &[LFM2_H, LFM2_FFN],
                    2,
                    make_q4_0_bytes(LFM2_FFN, LFM2_H),
                )
                .tensor(
                    format!("blk.{l}.ffn_down.weight"),
                    &[LFM2_FFN, LFM2_H],
                    2,
                    make_q4_0_bytes(LFM2_H, LFM2_FFN),
                );
            if l == 0 {
                b = b
                    .tensor(
                        "blk.0.shortconv.in_proj.weight",
                        &[LFM2_H, 3 * LFM2_H],
                        2,
                        make_q4_0_bytes(3 * LFM2_H, LFM2_H),
                    )
                    .tensor(
                        "blk.0.shortconv.out_proj.weight",
                        &[LFM2_H, LFM2_H],
                        2,
                        make_q4_0_bytes(LFM2_H, LFM2_H),
                    )
                    .tensor_f32(
                        "blk.0.shortconv.conv.weight",
                        &[LFM2_CONV_TAPS, LFM2_H],
                        &vec![0.1f32; LFM2_CONV_TAPS * LFM2_H],
                    );
            } else {
                b = b
                    .tensor(
                        "blk.1.attn_q.weight",
                        &[LFM2_H, LFM2_Q_DIM],
                        2,
                        make_q4_0_bytes(LFM2_Q_DIM, LFM2_H),
                    )
                    .tensor(
                        "blk.1.attn_k.weight",
                        &[LFM2_H, LFM2_KV_DIM],
                        2,
                        make_q4_0_bytes(LFM2_KV_DIM, LFM2_H),
                    )
                    .tensor(
                        "blk.1.attn_v.weight",
                        &[LFM2_H, LFM2_KV_DIM],
                        2,
                        make_q4_0_bytes(LFM2_KV_DIM, LFM2_H),
                    )
                    .tensor(
                        "blk.1.attn_output.weight",
                        &[LFM2_Q_DIM, LFM2_H],
                        2,
                        make_q4_0_bytes(LFM2_H, LFM2_Q_DIM),
                    )
                    .tensor_f32(
                        "blk.1.attn_q_norm.weight",
                        &[LFM2_HEAD_DIM],
                        &vec![1.0; LFM2_HEAD_DIM],
                    )
                    .tensor_f32(
                        "blk.1.attn_k_norm.weight",
                        &[LFM2_HEAD_DIM],
                        &vec![1.0; LFM2_HEAD_DIM],
                    );
            }
        }
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (xf, yf) = (x as f64, y as f64);
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        let d = na.sqrt() * nb.sqrt();
        if d == 0.0 { 0.0 } else { (dot / d) as f32 }
    }

    fn argmax_idx(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn causal_lfm2_load_prefill_decode_and_rewind_on_fake_shim() {
        let _ = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm2_gguf();
            let model = crate::model::load_model_xdna(gguf, None, 128)
                .expect("load_model_xdna should load LFM2 on fake shim");
            assert_eq!(model.backend_label(), "xdna");
            assert!(model.config().is_causal);

            let mut state = InferenceState::from_config(model.config()).unwrap();
            let logits = model.forward_prefill(&[1, 2, 3, 4], 0, &mut state);
            assert_eq!(logits.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 4);
            assert!(model.take_decode_error().is_none());

            // Run 3 decode steps across both conv KV ping-pong parities.
            for (step, tok) in [5u32, 6, 7].into_iter().enumerate() {
                let next = model.forward(&[tok], 4 + step, &mut state);
                assert_eq!(next.len(), model.config().vocab_size);
                assert_eq!(state.seq_len, 5 + step);
                assert!(model.take_decode_error().is_none());
            }

            // Rewind to a decode checkpoint (pos 5) and continue decoding.
            model.try_truncate_kv(&mut state, 5).unwrap();
            assert_eq!(state.seq_len, 5);
            let cont = model.forward(&[8], 5, &mut state);
            assert_eq!(cont.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 6);
            assert!(model.take_decode_error().is_none());

            // Rewind back to prefill boundary (pos 4) and continue with prefill + decode.
            model.try_truncate_kv(&mut state, 4).unwrap();
            assert_eq!(state.seq_len, 4);
            let cont_pre = model.forward_prefill(&[9, 10], 4, &mut state);
            assert_eq!(cont_pre.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 6);
            assert!(model.take_decode_error().is_none());
        });
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2.5-1.2B-Instruct-Q4_0.gguf"]
    fn hw_gate_lfm2_causal_parity_and_throughput() {
        let path = std::path::PathBuf::from(
            r"C:\Users\iamst\.cache\triage\models\huggingface.co\LiquidAI\LFM2.5-1.2B-Instruct-GGUF\resolve\main\LFM2.5-1.2B-Instruct-Q4_0.gguf",
        );
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open LFM2.5-1.2B-Instruct-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu =
            crate::model::load_model_xdna(gguf, Some(&path), 2048).expect("load LFM2.5 on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), 2048)
            .expect("load LFM2.5 on CPU");
        assert_eq!(npu.backend_label(), "xdna");

        let prompt = "The AMD Ryzen AI NPU uses a spatial dataflow array of AIE2P tiles \
                      for low-power matrix multiplication, convolution, and attention across \
                      multi-layer hybrid transformer and short-convolution models.";
        let mut tokens = tok.encode(prompt);
        while tokens.len() < 45 {
            tokens.push(tokens[tokens.len() % tokens.len().max(1)]);
        }
        tokens.truncate(45);

        let mut npu_state = InferenceState::from_config(npu.config()).unwrap();
        let mut cpu_state = InferenceState::from_config(cpu.config()).unwrap();

        let _ = npu.forward_prefill(&tokens[..32], 0, &mut npu_state);
        npu.try_reset_kv(&mut npu_state, &KvCompression::None, 2048)
            .unwrap();

        let t_prefill = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        if let Some(err) = npu.take_decode_error() {
            panic!("prefill error: {err}");
        }

        let prefill_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let npu_top = argmax_idx(&npu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "LFM2.5 NPU prefill (n=45): cos={prefill_cos:.6} top1_npu={npu_top} top1_cpu={cpu_top} time={:.2}ms ({:.1} tok/s)",
            prefill_s * 1e3,
            tokens.len() as f64 / prefill_s
        );
        assert!(
            prefill_cos >= 0.985,
            "prefill cosine too low: {prefill_cos}"
        );

        let mut decode_coss = Vec::new();
        let mut top1_matches = 0usize;
        let mut next_tok = cpu_top as u32;
        let mut decode_secs = 0.0f64;
        for step in 0..16usize {
            let pos = tokens.len() + step;
            let t0 = std::time::Instant::now();
            let nl = npu.forward(&[next_tok], pos, &mut npu_state);
            decode_secs += t0.elapsed().as_secs_f64();
            let cl = cpu.forward(&[next_tok], pos, &mut cpu_state);
            if let Some(err) = npu.take_decode_error() {
                panic!("decode error: {err:?}");
            }

            let c = cosine_similarity(&nl, &cl);
            let nt = argmax_idx(&nl);
            let ct = argmax_idx(&cl);
            if nt == ct {
                top1_matches += 1;
            }
            decode_coss.push(c);
            eprintln!("  decode step {step:2} pos={pos:2}: cos={c:.6} npu_top={nt} cpu_top={ct}");
            next_tok = ct as u32;
        }
        let mean_cos = decode_coss.iter().sum::<f32>() / decode_coss.len() as f32;
        let min_cos = decode_coss.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "LFM2.5 NPU decode (16 steps): mean_cos={mean_cos:.6} min_cos={min_cos:.6} top1={top1_matches}/16 ms/tok={:.2} ({:.1} tok/s)",
            (decode_secs * 1e3) / 16.0,
            16.0 / decode_secs
        );
        assert!(mean_cos >= 0.99, "mean decode cosine too low: {mean_cos}");
        assert!(min_cos >= 0.97, "min decode cosine too low: {min_cos}");
    }

    fn run_hw_lfm2_long_context_parity(prompt_len: usize, ctx_len: usize, tag: &str) {
        let path = std::path::PathBuf::from(
            r"C:\Users\iamst\.cache\triage\models\huggingface.co\LiquidAI\LFM2.5-1.2B-Instruct-GGUF\resolve\main\LFM2.5-1.2B-Instruct-Q4_0.gguf",
        );
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open LFM2.5-1.2B-Instruct-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu =
            crate::model::load_model_xdna(gguf, Some(&path), ctx_len).expect("load LFM2.5 on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), ctx_len)
            .expect("load LFM2.5 on CPU");

        let seed = tok
            .encode("The AMD Ryzen AI NPU uses a spatial dataflow array for low-power inference.");
        let mut tokens = Vec::with_capacity(prompt_len);
        for i in 0..prompt_len {
            tokens.push(seed[i % seed.len()]);
        }

        let mut npu_state = InferenceState::from_config(npu.config()).unwrap();
        let mut cpu_state = InferenceState::from_config(cpu.config()).unwrap();

        let t_pre = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let pre_s = t_pre.elapsed().as_secs_f64();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        if let Some(err) = npu.take_decode_error() {
            panic!("prefill {tag} error: {err}");
        }
        let pre_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "LFM2.5 NPU {tag} prefill (n={prompt_len}): cos={pre_cos:.6} npu_top={} cpu_top={cpu_top} time={:.2}s ({:.1} tok/s)",
            argmax_idx(&npu_logits),
            pre_s,
            prompt_len as f64 / pre_s
        );
        assert!(
            pre_cos >= 0.985,
            "long-context prefill cosine too low: {pre_cos}"
        );

        let mut decode_coss = Vec::new();
        let mut next_tok = cpu_top as u32;
        for step in 0..4usize {
            let pos = prompt_len + step;
            let nl = npu.forward(&[next_tok], pos, &mut npu_state);
            let cl = cpu.forward(&[next_tok], pos, &mut cpu_state);
            if let Some(err) = npu.take_decode_error() {
                panic!("decode {tag} error: {err}");
            }
            let c = cosine_similarity(&nl, &cl);
            let nt = argmax_idx(&nl);
            let ct = argmax_idx(&cl);
            eprintln!("  {tag} decode pos={pos}: cos={c:.6} npu_top={nt} cpu_top={ct}");
            decode_coss.push(c);
            next_tok = ct as u32;
        }
        let mean_cos = decode_coss.iter().sum::<f32>() / decode_coss.len() as f32;
        let min_cos = decode_coss.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            mean_cos >= 0.99,
            "long-context mean decode cosine too low: {mean_cos}"
        );
        assert!(
            min_cos >= 0.97,
            "long-context min decode cosine too low: {min_cos}"
        );
    }

    #[test]
    fn causal_lfm2_ext_bucket_boundaries_single_runlist_and_prefill_routes_on_fake_shim() {
        let (_, events) = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm2_gguf_with_ctx(70000);
            let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 66000, "test".to_string())
                .expect("load_model_xdna LFM2 at >32K context");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                66000,
            )
            .unwrap();
            let vocab = model.config().vocab_size;
            let emb = model.embed_token_rows(&[1]).unwrap();

            let seed_state_to_pos = |npu_m: &XdnaLfm2Model, st: &mut InferenceState, pos: usize| {
                if let LayerState::Attention {
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } = &mut st.layers[1]
                {
                    key_cache_f16.resize(pos * LFM2_KV_DIM, 0);
                    value_cache_f16.resize(pos * LFM2_KV_DIM, 0);
                }
                st.seq_len = pos;
                let mut guard = npu_m.lock_npu();
                guard.pending_kv.discard();
                guard.synced_len = pos;
                let max_blk = pos / LFM2_BLOCK_TOKENS;
                let state_mut = &mut *guard;
                XdnaLfm2Model::ensure_paged_block(
                    &state_mut.kernels,
                    &mut state_mut.layer_kv_blocks[1],
                    max_blk,
                    1,
                )
                .unwrap();
            };

            // Fast bucket covers up to 32767 (512 packets); single-runlist ext covers 32768..=65535
            // (`cur_blk = 1`, `fin_past_pkts = 0..=512`).
            for pos in [32767usize, 32768, 65535] {
                seed_state_to_pos(&npu_model, &mut state, pos);
                let out = model.forward_from_embedding(&emb, pos, &mut state);
                assert_eq!(out.len(), vocab, "pos={pos}");
                assert_eq!(state.seq_len, pos + 1, "pos={pos}");
                assert!(model.take_decode_error().is_none(), "pos={pos}");
            }

            // Aligned n_tokens=3 uses decode bypass; n_tokens=4 & need_logits=false uses prefill;
            // unaligned start_pos=30 with n_tokens=6 steps 2 decodes + 4-token prefill.
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 66000)
                .unwrap();
            let _ = model.forward_prefill(&[1, 2, 3], 0, &mut state);
            assert_eq!(state.seq_len, 3);
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 66000)
                .unwrap();
            let e4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
            let empty = model.forward_prefill_from_embeddings_chunk(&e4, 4, 0, &mut state, false);
            assert!(empty.is_empty());
            assert_eq!(state.seq_len, 4);

            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 66000)
                .unwrap();
            seed_state_to_pos(&npu_model, &mut state, 30);
            let cont6 = model.forward_prefill(&[1, 2, 3, 4, 5, 6], 30, &mut state);
            assert_eq!(cont6.len(), vocab);
            assert_eq!(state.seq_len, 36);
            assert!(model.take_decode_error().is_none());
        });

        let rl_execs = events
            .iter()
            .filter(|e| matches!(e, Event::RunListExecute(_)))
            .count();
        assert_eq!(
            rl_execs, 10,
            "expected exact 10 RunListExecute events (incl. 2 DecExt + fast/prefill steps), got {rl_execs}"
        );
        // Single-runlist >32K decode uses 3 distinct step_params byte offsets in bo_y.
        let has_ext_param_syncs = events.iter().any(|e| {
            matches!(
                e,
                Event::BoSyncRange {
                    to_device: true,
                    offset,
                    ..
                } if *offset == (LFM2_DEC_SLOT_NORM_1 + 4096) * 4
            )
        }) && events.iter().any(|e| {
            matches!(
                e,
                Event::BoSyncRange {
                    to_device: true,
                    offset,
                    ..
                } if *offset == (LFM2_DEC_SLOT_ROPE_PKT + 4096) * 4
            )
        });
        assert!(
            has_ext_param_syncs,
            "expected ext decode step_params syncs at NORM_1 and ROPE_PKT"
        );
    }

    #[test]
    fn causal_lfm2_multi_block_64k_and_96k_decode_prefill_kv_start_on_fake_shim() {
        let (_, events) = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm2_gguf_with_ctx(131072);
            let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 131072, "test".to_string())
                .expect("load_model_xdna LFM2 at 128K context");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                131072,
            )
            .unwrap();
            let vocab = model.config().vocab_size;

            let seed_state_to_pos = |st: &mut InferenceState, pos: usize| {
                if let LayerState::Attention {
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } = &mut st.layers[1]
                {
                    key_cache_f16.resize(pos * LFM2_KV_DIM, 0);
                    value_cache_f16.resize(pos * LFM2_KV_DIM, 0);
                }
                if let LayerState::Conv { buffer, .. } = &mut st.layers[0] {
                    buffer.resize(LFM2_D_CONV * LFM2_H, 0.25);
                }
                st.seq_len = pos;
                let mut guard = npu_model.lock_npu();
                guard.pending_kv.discard();
                guard.synced_len = pos;
                let max_blk = pos / LFM2_BLOCK_TOKENS;
                let state_mut = &mut *guard;
                XdnaLfm2Model::ensure_paged_block(
                    &state_mut.kernels,
                    &mut state_mut.layer_kv_blocks[1],
                    max_blk,
                    1,
                )
                .unwrap();
            };

            let read_i32 = |slice: &[u8], f32_idx: usize| -> i32 {
                let off = f32_idx * 4;
                i32::from_le_bytes(slice[off..off + 4].try_into().unwrap())
            };

            // 1. Multi-block decode at pos=65600 (cur_blk=2): per-block fa_part syncs blk=0 then blk=1 (kv_start=512, has_fa_state=1).
            let emb = model.embed_token_rows(&[1]).unwrap();
            seed_state_to_pos(&mut state, 65600);
            let dec_out = model.forward_from_embedding(&emb, 65600, &mut state);
            assert_eq!(dec_out.len(), vocab);
            assert_eq!(state.seq_len, 65601);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                let yb = guard.decode.as_ref().unwrap().layer_bo_y[1].as_slice();
                assert_eq!(
                    read_i32(yb, LFM2_DEC_SLOT_NORM_1 + 4096 + 8),
                    LFM2_DEC_MAX_KV_PKTS as i32
                );
                assert_eq!(read_i32(yb, LFM2_DEC_SLOT_NORM_1 + 4096 + 9), 1);
                assert_eq!(
                    read_i32(yb, LFM2_DEC_SLOT_ROPE_PKT + 4096 + 8),
                    (2 * LFM2_DEC_MAX_KV_PKTS) as i32
                );
            }

            // 2. Single-runlist >64K prefill at pos0=65536 (cur_blk=2): EXT_PARAMS_1 must write kv_start_chunk=1024 (LFM2_MAX_KV_CHUNKS).
            let embs4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
            seed_state_to_pos(&mut state, 65536);
            let pre_64k =
                model.forward_prefill_from_embeddings_chunk(&embs4, 4, 65536, &mut state, true);
            assert_eq!(pre_64k.len(), vocab);
            assert_eq!(state.seq_len, 65540);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                let yb = guard.prefill.as_ref().unwrap().layer_bo_y[1].as_slice();
                assert_eq!(
                    read_i32(yb, LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048 + 8),
                    LFM2_MAX_KV_CHUNKS as i32
                );
                assert_eq!(
                    read_i32(yb, LFM2_PREFILL_SLOT_EXT_PARAMS_2 + 2048 + 8),
                    (2 * LFM2_MAX_KV_CHUNKS) as i32
                );
            }

            // 3. Multi-block >96K prefill at pos0=98304 (cur_blk=3): per-block loop syncs b=1 (1024) then b=2 (2048).
            seed_state_to_pos(&mut state, 98304);
            let pre_96k =
                model.forward_prefill_from_embeddings_chunk(&embs4, 4, 98304, &mut state, true);
            assert_eq!(pre_96k.len(), vocab);
            assert_eq!(state.seq_len, 98308);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                let yb = guard.prefill.as_ref().unwrap().layer_bo_y[1].as_slice();
                assert_eq!(
                    read_i32(yb, LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048 + 8),
                    (2 * LFM2_MAX_KV_CHUNKS) as i32
                );
                assert_eq!(
                    read_i32(yb, LFM2_PREFILL_SLOT_EXT_PARAMS_2 + 2048 + 8),
                    (3 * LFM2_MAX_KV_CHUNKS) as i32
                );
            }
        });

        let dec_part_syncs = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::BoSyncRange {
                        to_device: true,
                        offset,
                        ..
                    } if *offset == (LFM2_DEC_SLOT_NORM_1 + 4096) * 4
                )
            })
            .count();
        assert_eq!(
            dec_part_syncs, 2,
            "expected 2 dec fa_part syncs (blk 0 and blk 1 at pos=65600), got {dec_part_syncs}"
        );
        let pre_part1_syncs = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::BoSyncRange {
                        to_device: true,
                        offset,
                        ..
                    } if *offset == (LFM2_PREFILL_SLOT_EXT_PARAMS_1 + 2048) * 4
                )
            })
            .count();
        assert_eq!(
            pre_part1_syncs, 3,
            "expected 3 prefill EXT_PARAMS_1 syncs (1 for cur_blk=2 + 2 for cur_blk=3), got {pre_part1_syncs}"
        );
    }

    #[test]
    fn causal_lfm2_greedy_poison_and_session_reset_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm2_gguf();
            let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 128, "test".to_string())
                .expect("load_model_xdna LFM2 on fake shim");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                128,
            )
            .unwrap();

            state.scratch.logits.reserve(model.config().vocab_size);
            let scratch_ptr = state.scratch.logits.as_ptr();
            let t0 = model.forward_greedy(&[1], 0, &mut state);
            assert!((t0 as usize) < model.config().vocab_size);
            assert_eq!(state.seq_len, 1);
            assert_eq!(state.scratch.logits.as_ptr(), scratch_ptr);

            // Caller KV mismatch in `forward_greedy` rejects cleanly without tearing NPU state.
            let bad = model.forward_greedy(&[1], 5, &mut state);
            assert_eq!(bad, 0);
            assert!(!npu_model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());

            // Poisoning `npu` tears state on next `lock_npu` (`forward_greedy` refuses fail-open),
            // and `acquire_session` recovers cleanly.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = npu_model.npu.lock().unwrap();
                panic!("simulated panic in XdnaLfm2Model");
            }));
            let torn = model.forward_greedy(&[1], 0, &mut state);
            assert_eq!(torn, 0);
            assert!(npu_model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());

            let _lease = model
                .acquire_session()
                .expect("acquire_session clears torn state");
            assert!(!npu_model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_none());
        });
    }

    #[test]
    fn causal_lfm2_sync_cpu_state_to_npu_writes_prefill_v_and_ext_prefill_on_fake_shim() {
        let (_, events) = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm2_gguf_with_ctx(70000);
            let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 66000, "test".to_string())
                .expect("load_model_xdna LFM2 at >32K context");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                66000,
            )
            .unwrap();

            // Seed CPU KV rows 0..32 with distinct non-uniform K/V values across `(t, kvh, d)`
            // and call `sync_cpu_state_to_npu`, then verify bf16 V lands at `+1024*4`.
            let target_pos = 32usize;
            if let LayerState::Attention {
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[1]
            {
                key_cache_f16.resize(target_pos * LFM2_KV_DIM, 0);
                value_cache_f16.resize(target_pos * LFM2_KV_DIM, 0);
                for t in 0..target_pos {
                    for kvh in 0..LFM2_N_KV_HEADS {
                        for d in 0..LFM2_HEAD_DIM {
                            let idx = t * LFM2_KV_DIM + kvh * LFM2_HEAD_DIM + d;
                            let kv_seed = ((t * 13 + kvh * 7 + d) % 97) as f32 * 0.0625;
                            key_cache_f16[idx] = crate::quant::f32_to_f16(1.0 + kv_seed);
                            value_cache_f16[idx] = crate::quant::f32_to_f16(-2.0 - kv_seed);
                        }
                    }
                }
            }
            if let LayerState::Conv { buffer, .. } = &mut state.layers[0] {
                buffer.resize(2 * LFM2_H, 0.25);
            }
            state.seq_len = target_pos;
            {
                let mut guard = npu_model.lock_npu();
                npu_model
                    .sync_cpu_state_to_npu(&mut guard, &mut state, target_pos, true)
                    .expect("sync_cpu_state_to_npu succeeds");
                let kv_buf = guard.layer_kv_blocks[1][0].as_slice();
                for (t, kvh, d) in [(0usize, 0usize, 0usize), (17, 5, 43), (31, 7, 63)] {
                    let chk_off = (kvh * LFM2_MAX_KV_CHUNKS) * LFM2_KV_CHUNK_BYTES;
                    let k_off = chk_off + t * LFM2_HEAD_DIM * 2 + d * 2;
                    let v_off = chk_off + 1024 * 4 + t * LFM2_HEAD_DIM * 2 + d * 2;
                    let seed = ((t * 13 + kvh * 7 + d) % 97) as f32 * 0.0625;
                    let expected_k = crate::backend::xdna::repack::f32_to_bf16(
                        crate::quant::f16_to_f32(crate::quant::f32_to_f16(1.0 + seed)),
                    );
                    let expected_v = crate::backend::xdna::repack::f32_to_bf16(
                        crate::quant::f16_to_f32(crate::quant::f32_to_f16(-2.0 - seed)),
                    );
                    assert_eq!(
                        u16::from_le_bytes(kv_buf[k_off..k_off + 2].try_into().unwrap()),
                        expected_k
                    );
                    assert_eq!(
                        u16::from_le_bytes(kv_buf[v_off..v_off + 2].try_into().unwrap()),
                        expected_v,
                        "KV V slot (+1024*4) missing V data"
                    );
                }
            }

            // Exercise `>32K` (`cur_blk=1, fin_past_chunks=0`), `pos0=32800` (`cur_blk=1, fin_past_chunks=1`),
            // and `>64K` (`cur_blk=2`) aligned continuation prefill chunks via `sync_cpu_state_to_npu`.
            let e4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
            for pos0 in [32768usize, 32800usize, 65536usize] {
                if let LayerState::Attention {
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } = &mut state.layers[1]
                {
                    key_cache_f16.resize(pos0 * LFM2_KV_DIM, 0);
                    value_cache_f16.resize(pos0 * LFM2_KV_DIM, 0);
                    for (t, tag) in [(pos0 - 32, 3.25f32), (pos0 - 1, -4.5f32)] {
                        for kvh in [0usize, 7] {
                            let idx = t * LFM2_KV_DIM + kvh * LFM2_HEAD_DIM + 11;
                            key_cache_f16[idx] = crate::quant::f32_to_f16(tag);
                            value_cache_f16[idx] = crate::quant::f32_to_f16(-tag);
                        }
                    }
                }
                state.seq_len = pos0;
                let out =
                    model.forward_prefill_from_embeddings_chunk(&e4, 4, pos0, &mut state, true);
                assert_eq!(out.len(), model.config().vocab_size, "pos0={pos0}");
                assert_eq!(state.seq_len, pos0 + 4, "pos0={pos0}");
                assert!(model.take_decode_error().is_none(), "pos0={pos0}");
                let guard = npu_model.lock_npu();
                for (check_t, tag) in [(pos0 - 32, 3.25f32), (pos0 - 1, -4.5f32)] {
                    let g_chk = check_t / LFM2_KV_CHUNK_T;
                    let blk = g_chk / LFM2_MAX_KV_CHUNKS;
                    let chk = g_chk % LFM2_MAX_KV_CHUNKS;
                    for kvh in [0usize, 7] {
                        let chk_off = (kvh * LFM2_MAX_KV_CHUNKS + chk) * LFM2_KV_CHUNK_BYTES;
                        let rem = check_t % LFM2_KV_CHUNK_T;
                        let k_off = chk_off + rem * LFM2_HEAD_DIM * 2 + 11 * 2;
                        let v_off = chk_off + 1024 * 4 + rem * LFM2_HEAD_DIM * 2 + 11 * 2;
                        let slice = guard.layer_kv_blocks[1][blk].as_slice();
                        let k_val = u16::from_le_bytes(slice[k_off..k_off + 2].try_into().unwrap());
                        let v_val = u16::from_le_bytes(slice[v_off..v_off + 2].try_into().unwrap());
                        assert_eq!(
                            k_val,
                            crate::backend::xdna::repack::f32_to_bf16(tag),
                            "block {blk} K mismatch at pos0={pos0}, check_t={check_t}, kvh={kvh}"
                        );
                        assert_eq!(
                            v_val,
                            crate::backend::xdna::repack::f32_to_bf16(-tag),
                            "block {blk} V mismatch at pos0={pos0}, check_t={check_t}, kvh={kvh}"
                        );
                    }
                }
            }
        });
        assert!(
            events.iter().any(|e| matches!(e, Event::RunListExecute(_))),
            "expected RunListExecute events during >32K LFM2 continuation prefill"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                Event::BoSyncRange {
                    to_device: true,
                    offset,
                    ..
                } if *offset == (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4
            )),
            "expected >32K LFM2 prefill step_params sync at NORM_0"
        );
        let kv_bo_handles: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                Event::BoAlloc { bytes, handle, .. } if *bytes == LFM2_KV_BYTES => Some(*handle),
                _ => None,
            })
            .collect();
        assert!(!kv_bo_handles.is_empty());
        assert!(
            events.iter().any(|e| matches!(
                e,
                Event::BoSyncRange {
                    handle,
                    to_device: true,
                    size,
                    ..
                } if kv_bo_handles.contains(handle) && *size >= LFM2_KV_CHUNK_BYTES
            )),
            "expected >32K LFM2 KV host-to-device block sync"
        );
    }

    #[test]
    fn causal_lfm2_ext_prefill_stalled_run_and_validation_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let over_ctx = tiny_lfm2_gguf_with_ctx((LFM2_MAX_TOKENS + 1) as u32);
            let Err(err) =
                XdnaLfm2Model::from_gguf_with_id(over_ctx, LFM2_MAX_TOKENS + 1, "bad".to_string())
            else {
                panic!("oversize max_seq_len should be rejected");
            };
            assert!(
                err.to_string().contains("unsupported context length"),
                "unexpected error: {err}"
            );
        });

        with_shim(
            FakeConfig {
                fail_runlist_wait: true,
                ..Default::default()
            },
            || {
                let gguf = tiny_lfm2_gguf_with_ctx(35000);
                let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 33024, "test".to_string())
                    .expect("load LFM2 for timeout test");
                let model: &dyn Model = &npu_model;
                let mut state = InferenceState::from_config_capped(
                    model.config(),
                    &crate::kv_cache::KvCompression::F16,
                    33024,
                )
                .unwrap();
                if let LayerState::Attention {
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } = &mut state.layers[1]
                {
                    key_cache_f16.resize(32800 * LFM2_KV_DIM, 0);
                    value_cache_f16.resize(32800 * LFM2_KV_DIM, 0);
                }
                if let LayerState::Conv { buffer, .. } = &mut state.layers[0] {
                    buffer.resize(2 * LFM2_H, 0.25);
                }
                state.seq_len = 32800;
                let e4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
                let out =
                    model.forward_prefill_from_embeddings_chunk(&e4, 4, 32800, &mut state, true);
                assert_eq!(out, vec![0.0; model.config().vocab_size]);
                assert!(model.take_decode_error().is_some());
                assert!(npu_model.state_torn.load(Ordering::SeqCst));
                {
                    let guard = npu_model.lock_npu();
                    assert!(
                        guard.in_flight.is_some(),
                        "timed-out >32K runlist must remain tracked in in_flight"
                    );
                }
                let reset_err = model
                    .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 33024)
                    .expect_err(
                        "try_reset_kv fails to clear while in-flight runlist still times out",
                    );
                assert!(
                    reset_err.to_string().contains("lfm2 quiesce wait failed"),
                    "unexpected reset err: {reset_err}"
                );
                assert!(npu_model.state_torn.load(Ordering::SeqCst));
                assert!(npu_model.lock_npu().in_flight.is_some());
            },
        );
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2.5-1.2B-Instruct-Q4_0.gguf"]
    fn hw_gate_lfm2_long_context_parity() {
        run_hw_lfm2_long_context_parity(8224, 8256, ">8K");
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2.5-1.2B-Instruct-Q4_0.gguf"]
    fn hw_gate_lfm2_16k_ext_parity() {
        run_hw_lfm2_long_context_parity(16448, 16512, ">16K");
    }

    fn tiny_lfm2_moe_gguf() -> GgufFile {
        const DENSE_FFN: usize = 128;
        const EXPERT_FFN: usize = 64;
        const N_EXPERT: usize = 4;
        const N_EXPERT_USED: usize = 2;

        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "lfm2moe")
            .kv_u32("lfm2moe.block_count", 3)
            .kv_u32("lfm2moe.embedding_length", LFM2_H as u32)
            .kv_u32("lfm2moe.feed_forward_length", DENSE_FFN as u32)
            .kv_u32("lfm2moe.leading_dense_block_count", 1)
            .kv_u32("lfm2moe.expert_count", N_EXPERT as u32)
            .kv_u32("lfm2moe.expert_used_count", N_EXPERT_USED as u32)
            .kv_u32("lfm2moe.expert_feed_forward_length", EXPERT_FFN as u32)
            .kv_u32("lfm2moe.expert_gating_func", 2)
            .kv_u32("lfm2moe.attention.head_count", LFM2_N_HEADS as u32)
            .kv_i32_array(
                "lfm2moe.attention.head_count_kv",
                vec![0, LFM2_N_KV_HEADS as i32, 0],
            )
            .kv_u32("lfm2moe.vocab_size", 64)
            .kv_u32("lfm2moe.context_length", 128)
            .kv_f32("lfm2moe.rope.freq_base", 10_000_000.0)
            .kv_f32("lfm2moe.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("lfm2moe.shortconv.l_cache", LFM2_CONV_TAPS as u32);

        b = b.tensor(
            "token_embd.weight",
            &[LFM2_H, 64],
            2,
            make_q4_0_bytes(64, LFM2_H),
        );
        b = b.tensor_f32("token_embd_norm.weight", &[LFM2_H], &vec![1.0f32; LFM2_H]);

        for l in 0..3 {
            b = b
                .tensor_f32(
                    format!("blk.{l}.attn_norm.weight"),
                    &[LFM2_H],
                    &vec![1.0; LFM2_H],
                )
                .tensor_f32(
                    format!("blk.{l}.ffn_norm.weight"),
                    &[LFM2_H],
                    &vec![1.0; LFM2_H],
                );
            if l == 0 {
                b = b
                    .tensor(
                        "blk.0.ffn_gate.weight",
                        &[LFM2_H, DENSE_FFN],
                        2,
                        make_q4_0_bytes(DENSE_FFN, LFM2_H),
                    )
                    .tensor(
                        "blk.0.ffn_up.weight",
                        &[LFM2_H, DENSE_FFN],
                        2,
                        make_q4_0_bytes(DENSE_FFN, LFM2_H),
                    )
                    .tensor(
                        "blk.0.ffn_down.weight",
                        &[DENSE_FFN, LFM2_H],
                        2,
                        make_q4_0_bytes(LFM2_H, DENSE_FFN),
                    );
            } else {
                b = b
                    .tensor_f32(
                        format!("blk.{l}.ffn_gate_inp.weight"),
                        &[LFM2_H, N_EXPERT],
                        &vec![0.05f32; N_EXPERT * LFM2_H],
                    )
                    .tensor_f32(
                        format!("blk.{l}.exp_probs_b.bias"),
                        &[N_EXPERT],
                        &[0.1, 0.2, 0.0, -0.1],
                    )
                    .tensor(
                        format!("blk.{l}.ffn_gate_exps.weight"),
                        &[LFM2_H, EXPERT_FFN, N_EXPERT],
                        2,
                        make_q4_0_bytes(N_EXPERT * EXPERT_FFN, LFM2_H),
                    )
                    .tensor(
                        format!("blk.{l}.ffn_up_exps.weight"),
                        &[LFM2_H, EXPERT_FFN, N_EXPERT],
                        2,
                        make_q4_0_bytes(N_EXPERT * EXPERT_FFN, LFM2_H),
                    )
                    .tensor(
                        format!("blk.{l}.ffn_down_exps.weight"),
                        &[EXPERT_FFN, LFM2_H, N_EXPERT],
                        2,
                        make_q4_0_bytes(N_EXPERT * LFM2_H, EXPERT_FFN),
                    );
            }

            if l != 1 {
                b = b
                    .tensor(
                        format!("blk.{l}.shortconv.in_proj.weight"),
                        &[LFM2_H, 3 * LFM2_H],
                        2,
                        make_q4_0_bytes(3 * LFM2_H, LFM2_H),
                    )
                    .tensor(
                        format!("blk.{l}.shortconv.out_proj.weight"),
                        &[LFM2_H, LFM2_H],
                        2,
                        make_q4_0_bytes(LFM2_H, LFM2_H),
                    )
                    .tensor_f32(
                        format!("blk.{l}.shortconv.conv.weight"),
                        &[LFM2_CONV_TAPS, LFM2_H],
                        &vec![0.1f32; LFM2_CONV_TAPS * LFM2_H],
                    );
            } else {
                b = b
                    .tensor(
                        "blk.1.attn_q.weight",
                        &[LFM2_H, LFM2_Q_DIM],
                        2,
                        make_q4_0_bytes(LFM2_Q_DIM, LFM2_H),
                    )
                    .tensor(
                        "blk.1.attn_k.weight",
                        &[LFM2_H, LFM2_KV_DIM],
                        2,
                        make_q4_0_bytes(LFM2_KV_DIM, LFM2_H),
                    )
                    .tensor(
                        "blk.1.attn_v.weight",
                        &[LFM2_H, LFM2_KV_DIM],
                        2,
                        make_q4_0_bytes(LFM2_KV_DIM, LFM2_H),
                    )
                    .tensor(
                        "blk.1.attn_output.weight",
                        &[LFM2_Q_DIM, LFM2_H],
                        2,
                        make_q4_0_bytes(LFM2_H, LFM2_Q_DIM),
                    )
                    .tensor_f32(
                        "blk.1.attn_q_norm.weight",
                        &[LFM2_HEAD_DIM],
                        &vec![1.0; LFM2_HEAD_DIM],
                    )
                    .tensor_f32(
                        "blk.1.attn_k_norm.weight",
                        &[LFM2_HEAD_DIM],
                        &vec![1.0; LFM2_HEAD_DIM],
                    );
            }
        }
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    #[test]
    fn causal_lfm2moe_load_prefill_decode_and_rewind_on_fake_shim() {
        let _ = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm2_moe_gguf();
            let model = crate::model::load_model_xdna(gguf, None, 128)
                .expect("load_model_xdna should load LFM2-MoE on fake shim");
            assert_eq!(model.backend_label(), "xdna");
            assert!(model.config().is_causal);
            assert!(model.config().moe.is_some());

            let mut state = InferenceState::from_config(model.config()).unwrap();
            let logits = model.forward_prefill(&[1, 2, 3, 4], 0, &mut state);
            assert_eq!(logits.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 4);
            assert!(model.take_decode_error().is_none());

            for (step, tok) in [5u32, 6, 7].into_iter().enumerate() {
                let next = model.forward(&[tok], 4 + step, &mut state);
                assert_eq!(next.len(), model.config().vocab_size);
                assert_eq!(state.seq_len, 5 + step);
                assert!(model.take_decode_error().is_none());
            }

            model.try_truncate_kv(&mut state, 5).unwrap();
            assert_eq!(state.seq_len, 5);
            let cont = model.forward(&[8], 5, &mut state);
            assert_eq!(cont.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 6);
            assert!(model.take_decode_error().is_none());
        });
    }

    #[test]
    fn causal_lfm2_custom_ffn_q8_load_prefill_decode_and_rewind_on_fake_shim() {
        fn make_q8_0_bytes(rows: usize, cols: usize) -> Vec<u8> {
            vec![0u8; rows * (cols / 32) * 34]
        }

        const CUSTOM_FFN: usize = 10752;
        let _ = with_shim(FakeConfig::default(), || {
            let mut b = GgufBuilder::new()
                .kv_str("general.architecture", "lfm2")
                .kv_u32("lfm2.block_count", 2)
                .kv_u32("lfm2.embedding_length", LFM2_H as u32)
                .kv_u32("lfm2.feed_forward_length", CUSTOM_FFN as u32)
                .kv_u32("lfm2.attention.head_count", LFM2_N_HEADS as u32)
                .kv_i32_array(
                    "lfm2.attention.head_count_kv",
                    vec![0, LFM2_N_KV_HEADS as i32],
                )
                .kv_u32("lfm2.vocab_size", 64)
                .kv_u32("lfm2.context_length", 128)
                .kv_f32("lfm2.rope.freq_base", 1_000_000.0)
                .kv_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5)
                .kv_u32("lfm2.shortconv.l_cache", LFM2_CONV_TAPS as u32)
                .tensor(
                    "token_embd.weight",
                    &[LFM2_H, 64],
                    8,
                    make_q8_0_bytes(64, LFM2_H),
                )
                .tensor_f32("token_embd_norm.weight", &[LFM2_H], &vec![1.0f32; LFM2_H]);

            for l in 0..2usize {
                b = b
                    .tensor_f32(
                        format!("blk.{l}.attn_norm.weight"),
                        &[LFM2_H],
                        &vec![1.0; LFM2_H],
                    )
                    .tensor_f32(
                        format!("blk.{l}.ffn_norm.weight"),
                        &[LFM2_H],
                        &vec![1.0; LFM2_H],
                    )
                    .tensor(
                        format!("blk.{l}.ffn_gate.weight"),
                        &[LFM2_H, CUSTOM_FFN],
                        8,
                        make_q8_0_bytes(CUSTOM_FFN, LFM2_H),
                    )
                    .tensor(
                        format!("blk.{l}.ffn_up.weight"),
                        &[LFM2_H, CUSTOM_FFN],
                        8,
                        make_q8_0_bytes(CUSTOM_FFN, LFM2_H),
                    )
                    .tensor(
                        format!("blk.{l}.ffn_down.weight"),
                        &[CUSTOM_FFN, LFM2_H],
                        8,
                        make_q8_0_bytes(LFM2_H, CUSTOM_FFN),
                    );
                if l == 0 {
                    b = b
                        .tensor(
                            "blk.0.shortconv.in_proj.weight",
                            &[LFM2_H, 3 * LFM2_H],
                            8,
                            make_q8_0_bytes(3 * LFM2_H, LFM2_H),
                        )
                        .tensor(
                            "blk.0.shortconv.out_proj.weight",
                            &[LFM2_H, LFM2_H],
                            8,
                            make_q8_0_bytes(LFM2_H, LFM2_H),
                        )
                        .tensor_f32(
                            "blk.0.shortconv.conv.weight",
                            &[LFM2_CONV_TAPS, LFM2_H],
                            &vec![0.1f32; LFM2_CONV_TAPS * LFM2_H],
                        );
                } else {
                    b = b
                        .tensor(
                            "blk.1.attn_q.weight",
                            &[LFM2_H, LFM2_Q_DIM],
                            8,
                            make_q8_0_bytes(LFM2_Q_DIM, LFM2_H),
                        )
                        .tensor(
                            "blk.1.attn_k.weight",
                            &[LFM2_H, LFM2_KV_DIM],
                            8,
                            make_q8_0_bytes(LFM2_KV_DIM, LFM2_H),
                        )
                        .tensor(
                            "blk.1.attn_v.weight",
                            &[LFM2_H, LFM2_KV_DIM],
                            8,
                            make_q8_0_bytes(LFM2_KV_DIM, LFM2_H),
                        )
                        .tensor(
                            "blk.1.attn_output.weight",
                            &[LFM2_Q_DIM, LFM2_H],
                            8,
                            make_q8_0_bytes(LFM2_H, LFM2_Q_DIM),
                        )
                        .tensor_f32(
                            "blk.1.attn_q_norm.weight",
                            &[LFM2_HEAD_DIM],
                            &vec![1.0; LFM2_HEAD_DIM],
                        )
                        .tensor_f32(
                            "blk.1.attn_k_norm.weight",
                            &[LFM2_HEAD_DIM],
                            &vec![1.0; LFM2_HEAD_DIM],
                        );
                }
            }
            let gguf =
                GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice()))
                    .unwrap();
            let model = crate::model::load_model_xdna(gguf, None, 128)
                .expect("load_model_xdna should load Q8_0 FFN=10752 LFM2 on fake shim");
            assert_eq!(model.backend_label(), "xdna");
            assert_eq!(model.config().intermediate_size, CUSTOM_FFN);

            let mut state = InferenceState::from_config(model.config()).unwrap();
            let logits = model.forward_prefill(&[1, 2, 3, 4], 0, &mut state);
            assert_eq!(logits.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 4);
            assert!(model.take_decode_error().is_none());

            for (step, tok) in [5u32, 6, 7].into_iter().enumerate() {
                let next = model.forward(&[tok], 4 + step, &mut state);
                assert_eq!(next.len(), model.config().vocab_size);
                assert_eq!(state.seq_len, 5 + step);
                assert!(model.take_decode_error().is_none());
            }

            model.try_truncate_kv(&mut state, 5).unwrap();
            assert_eq!(state.seq_len, 5);
            let cont = model.forward(&[8], 5, &mut state);
            assert_eq!(cont.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 6);
            assert!(model.take_decode_error().is_none());
        });
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2.5-1.2B-Instruct-Q4_0.gguf"]
    fn hw_gate_lfm2_32k_plus_ext_parity() {
        run_hw_lfm2_long_context_parity(32900, 33024, "32k_plus_ext");
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2.5-24B-A2B-QAD-Q4_0.gguf"]
    fn hw_gate_lfm2moe_causal_parity_and_throughput() {
        let path = std::path::PathBuf::from(
            r"C:\Users\iamst\.cache\cera\huggingface.co\LiquidAI\LFM2.5-24B-A2B-GGUF\resolve\d790f111166905bfb36a81e12a403701ac23a1ca\LFM2.5-24B-A2B-QAD-Q4_0.gguf",
        );
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open LFM2.5-24B-A2B-QAD-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu = crate::model::load_model_xdna(gguf, Some(&path), 2048)
            .expect("load LFM2.5-24B-A2B on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), 2048)
            .expect("load LFM2.5-24B-A2B on CPU");
        assert_eq!(npu.backend_label(), "xdna");

        let prompt = "The AMD Ryzen AI NPU uses a spatial dataflow array of AIE2P tiles \
                      for low-power matrix multiplication, convolution, and attention across \
                      multi-layer hybrid transformer and short-convolution models.";
        let mut tokens = tok.encode(prompt);
        while tokens.len() < 64 {
            tokens.push(tokens[tokens.len() % tokens.len().max(1)]);
        }
        tokens.truncate(64);

        let mut npu_state = InferenceState::from_config(npu.config()).unwrap();
        let mut cpu_state = InferenceState::from_config(cpu.config()).unwrap();

        let _ = npu.forward_prefill(&tokens[..32], 0, &mut npu_state);
        npu.try_reset_kv(&mut npu_state, &KvCompression::None, 2048)
            .unwrap();

        let t_prefill = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let t_cpu_pre = std::time::Instant::now();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        let cpu_prefill_s = t_cpu_pre.elapsed().as_secs_f64();
        if let Some(err) = npu.take_decode_error() {
            panic!("prefill error: {err:?}");
        }

        let prefill_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let npu_top = argmax_idx(&npu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "LFM2.5-24B-A2B prefill (n=64): cos={prefill_cos:.6} top1_npu={npu_top} top1_cpu={cpu_top} npu={:.2}ms ({:.1} tok/s) cpu={:.2}ms ({:.1} tok/s)",
            prefill_s * 1e3,
            tokens.len() as f64 / prefill_s,
            cpu_prefill_s * 1e3,
            tokens.len() as f64 / cpu_prefill_s,
        );
        assert!(
            prefill_cos >= 0.985,
            "prefill cosine too low: {prefill_cos}"
        );

        let mut decode_coss = Vec::new();
        let mut top1_matches = 0usize;
        let mut next_tok = cpu_top as u32;
        let mut decode_secs = 0.0f64;
        let mut cpu_decode_secs = 0.0f64;
        for step in 0..8usize {
            let pos = tokens.len() + step;
            let t0 = std::time::Instant::now();
            let nl = npu.forward(&[next_tok], pos, &mut npu_state);
            decode_secs += t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let cl = cpu.forward(&[next_tok], pos, &mut cpu_state);
            cpu_decode_secs += t1.elapsed().as_secs_f64();
            if let Some(err) = npu.take_decode_error() {
                panic!("decode error: {err:?}");
            }

            let c = cosine_similarity(&nl, &cl);
            let nt = argmax_idx(&nl);
            let ct = argmax_idx(&cl);
            if nt == ct {
                top1_matches += 1;
            }
            decode_coss.push(c);
            eprintln!("  decode step {step:2} pos={pos:2}: cos={c:.6} npu_top={nt} cpu_top={ct}");
            next_tok = ct as u32;
        }
        let mean_cos = decode_coss.iter().sum::<f32>() / decode_coss.len() as f32;
        let min_cos = decode_coss.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "LFM2.5-24B-A2B decode (8 steps): mean_cos={mean_cos:.6} min_cos={min_cos:.6} top1={top1_matches}/8 npu_ms={:.2} ({:.1} tok/s) cpu_ms={:.2} ({:.1} tok/s)",
            (decode_secs * 1e3) / 8.0,
            8.0 / decode_secs,
            (cpu_decode_secs * 1e3) / 8.0,
            8.0 / cpu_decode_secs,
        );
        assert!(mean_cos >= 0.985, "mean decode cosine too low: {mean_cos}");
        assert!(min_cos >= 0.97, "min decode cosine too low: {min_cos}");
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2-2.6B-Q4_K_M.gguf"]
    fn hw_gate_lfm2_2_6b_q4_k_m_causal_parity_and_throughput() {
        let path = std::path::PathBuf::from(
            r"C:\Users\iamst\.cache\triage\models\huggingface.co\LiquidAI\LFM2-2.6B-GGUF\resolve\main\LFM2-2.6B-Q4_K_M.gguf",
        );
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open LFM2-2.6B-Q4_K_M.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu = crate::model::load_model_xdna(gguf, Some(&path), 2048)
            .expect("load LFM2-2.6B-Q4_K_M on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), 2048)
            .expect("load LFM2-2.6B-Q4_K_M on CPU");
        assert_eq!(npu.backend_label(), "xdna");

        let prompt = "The AMD Ryzen AI NPU uses a spatial dataflow array of AIE2P tiles \
                      for low-power matrix multiplication, convolution, and attention across \
                      multi-layer hybrid transformer and short-convolution models.";
        let mut tokens = tok.encode(prompt);
        while tokens.len() < 64 {
            tokens.push(tokens[tokens.len() % tokens.len().max(1)]);
        }
        tokens.truncate(64);

        let mut npu_state = InferenceState::from_config(npu.config()).unwrap();
        let mut cpu_state = InferenceState::from_config(cpu.config()).unwrap();

        let _ = npu.forward_prefill(&tokens[..32], 0, &mut npu_state);
        npu.try_reset_kv(&mut npu_state, &KvCompression::None, 2048)
            .unwrap();

        let t_prefill = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let t_cpu_pre = std::time::Instant::now();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        let cpu_prefill_s = t_cpu_pre.elapsed().as_secs_f64();
        if let Some(err) = npu.take_decode_error() {
            panic!("prefill error: {err:?}");
        }

        let prefill_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let npu_top = argmax_idx(&npu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "LFM2-2.6B-Q4_K_M prefill (n=64): cos={prefill_cos:.6} top1_npu={npu_top} top1_cpu={cpu_top} npu={:.2}ms ({:.1} tok/s) cpu={:.2}ms ({:.1} tok/s)",
            prefill_s * 1e3,
            tokens.len() as f64 / prefill_s,
            cpu_prefill_s * 1e3,
            tokens.len() as f64 / cpu_prefill_s,
        );
        assert!(
            prefill_cos >= 0.985,
            "prefill cosine too low: {prefill_cos}"
        );

        let mut decode_coss = Vec::new();
        let mut top1_matches = 0usize;
        let mut next_tok = cpu_top as u32;
        let mut decode_secs = 0.0f64;
        let mut cpu_decode_secs = 0.0f64;
        for step in 0..8usize {
            let pos = tokens.len() + step;
            let t0 = std::time::Instant::now();
            let nl = npu.forward(&[next_tok], pos, &mut npu_state);
            decode_secs += t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let cl = cpu.forward(&[next_tok], pos, &mut cpu_state);
            cpu_decode_secs += t1.elapsed().as_secs_f64();
            if let Some(err) = npu.take_decode_error() {
                panic!("decode error: {err:?}");
            }

            let c = cosine_similarity(&nl, &cl);
            let nt = argmax_idx(&nl);
            let ct = argmax_idx(&cl);
            if nt == ct {
                top1_matches += 1;
            }
            decode_coss.push(c);
            eprintln!("  decode step {step:2} pos={pos:2}: cos={c:.6} npu_top={nt} cpu_top={ct}");
            next_tok = ct as u32;
        }
        let mean_cos = decode_coss.iter().sum::<f32>() / decode_coss.len() as f32;
        let min_cos = decode_coss.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "LFM2-2.6B-Q4_K_M decode (8 steps): mean_cos={mean_cos:.6} min_cos={min_cos:.6} top1={top1_matches}/8 npu_ms={:.2} ({:.1} tok/s) cpu_ms={:.2} ({:.1} tok/s)",
            (decode_secs * 1e3) / 8.0,
            8.0 / decode_secs,
            (cpu_decode_secs * 1e3) / 8.0,
            8.0 / cpu_decode_secs,
        );
        assert!(mean_cos >= 0.985, "mean decode cosine too low: {mean_cos}");
        assert!(min_cos >= 0.97, "min decode cosine too low: {min_cos}");
    }
}

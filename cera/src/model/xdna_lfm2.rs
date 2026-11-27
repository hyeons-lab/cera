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
    LFM2_BLOCK_TOKENS, LFM2_CONV_TAPS, LFM2_D_CONV, LFM2_DEC_ACT_BYTES, LFM2_DEC_SLOT_NORM_0,
    LFM2_DEC_SLOT_NORM_1, LFM2_DEC_SLOT_ROPE_PKT, LFM2_DEC_YN_BYTES, LFM2_FFN, LFM2_H,
    LFM2_HEAD_DIM, LFM2_KV_BYTES, LFM2_KV_CHUNK_T, LFM2_KV_DIM, LFM2_KVO_BYTES, LFM2_MAX_KV_CHUNKS,
    LFM2_MAX_TOKENS, LFM2_N_HEADS, LFM2_N_KV_HEADS, LFM2_PREFILL_ACT_BYTES, LFM2_PREFILL_CHUNK_T,
    LFM2_PREFILL_SLOT_NORM_0, LFM2_PREFILL_SLOT_ROPE_PKT_0, LFM2_PREFILL_SLOT_ROPE_PKT_1,
    LFM2_PREFILL_YN_BYTES, LFM2_Q_DIM, LFM2_W_BYTES, Q4LAYER_LFM2_CONV_INSTS,
    Q4LAYER_LFM2_FA_FIN_BUCKETS, Q4LAYER_LFM2_FA_PART_C256_INSTS, Q4LAYER_LFM2_KV_BUCKETS,
    Q4LAYER_LFM2_PHASE_A_INSTS, Q4LAYER_LFM2_XCLBIN, Q4LAYER_TILE_F32, Q4PREFILL_LFM2_ATTN_BUCKETS,
    Q4PREFILL_LFM2_CONV_INSTS, Q4PREFILL_LFM2_FA_FIN_BUCKETS, Q4PREFILL_LFM2_FA_PART_C256_INSTS,
    Q4PREFILL_LFM2_PHASE_A_INSTS, Q4PREFILL_LFM2_XCLBIN, init_lfm2_bo_y_norms,
    init_lfm2_prefill_bo_y_norms, inst_words, read_le_f32_slice, try_select_lfm3_kv_bucket,
    write_le_f32_slice, write_lfm2_prefill_ext_params, write_lfm2_prefill_step_params,
    write_lfm3_params_at, write_lfm3_step_params,
};
use crate::backend::xdna::repack::{f32_to_bf16, pack_q4_0_stage_k1024};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError, LayerState};
use crate::model::lfm2::LfmModel;
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
        cfg.architecture == "lfm2",
        "expected architecture `lfm2`, got `{}`",
        cfg.architecture
    );
    ensure!(cfg.is_causal, "XDNA LFM2 backend requires causal attention");
    ensure!(
        cfg.moe.is_none(),
        "XDNA LFM2 backend does not support MoE layers"
    );
    ensure!(
        cfg.hidden_size == LFM2_H,
        "XDNA LFM2 requires hidden_size={LFM2_H}, got {}",
        cfg.hidden_size
    );
    ensure!(
        cfg.intermediate_size == LFM2_FFN,
        "XDNA LFM2 requires intermediate_size={LFM2_FFN}, got {}",
        cfg.intermediate_size
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

    let check_q4 = |label: &str, layer: usize, w: &WeightRef, m: usize, k: usize| -> Result<()> {
        ensure!(
            w.dtype == DType::Q4_0,
            "layer {layer} {label}: XDNA LFM2 requires Q4_0 weights, got {:?}",
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

    for l in 0..cfg.n_layers {
        ensure!(
            cpu.xdna_lfm3_post_attn_norm(l).is_none()
                && cpu.xdna_lfm3_post_ffn_norm(l).is_none()
                && cpu.xdna_lfm3_attn_gate_ref(l).is_none(),
            "layer {l}: XDNA LFM2 does not support post-norms or attn_gate"
        );
        let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
        check_q4("ffn_gate", l, fg, LFM2_FFN, LFM2_H)?;
        check_q4("ffn_up", l, fu, LFM2_FFN, LFM2_H)?;
        check_q4("ffn_down", l, fd, LFM2_H, LFM2_FFN)?;

        match cfg.block_types[l] {
            BlockType::GatedConv => {
                let (in_proj, out_proj) = cpu
                    .xdna_layer_conv_refs(l)
                    .with_context(|| format!("layer {l} missing shortconv weights"))?;
                check_q4("shortconv.in_proj", l, in_proj, 3 * LFM2_H, LFM2_H)?;
                check_q4("shortconv.out_proj", l, out_proj, LFM2_H, LFM2_H)?;
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
                check_q4("attn_q", l, wq, LFM2_Q_DIM, LFM2_H)?;
                check_q4("attn_k", l, wk, LFM2_KV_DIM, LFM2_H)?;
                check_q4("attn_v", l, wv, LFM2_KV_DIM, LFM2_H)?;
                check_q4("attn_output", l, wo, LFM2_H, LFM2_Q_DIM)?;
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
    prefill_conv: XdnaKernel,
    prefill_attn: Vec<Option<XdnaKernel>>,
    prefill_phase_a: Option<XdnaKernel>,
    prefill_fa_part: Option<XdnaKernel>,
    prefill_fa_fin: Vec<Option<XdnaKernel>>,
    dec_conv: XdnaKernel,
    dec_attn: Vec<Option<XdnaKernel>>,
    dec_phase_a: Option<XdnaKernel>,
    dec_fa_part: Option<XdnaKernel>,
    dec_fa_fin: Vec<Option<XdnaKernel>>,
}

impl Lfm2Kernels {
    fn open() -> Result<Self> {
        let device = XdnaDevice::open(0).map_err(|e| anyhow::anyhow!("XDNA open failed: {e}"))?;
        let prefill_conv = device
            .load_kernel(
                Q4PREFILL_LFM2_XCLBIN,
                &inst_words(Q4PREFILL_LFM2_CONV_INSTS),
            )
            .map_err(|e| anyhow::anyhow!("XDNA load q4prefill_lfm2 conv failed: {e}"))?;

        // Preload every prefill attention bucket consecutively right after `prefill_conv`
        // so all prefill kernels share one XRT hardware context and can be batched
        // together in a single `XdnaRunList`.
        let mut prefill_attn: Vec<Option<XdnaKernel>> = (0..Q4PREFILL_LFM2_ATTN_BUCKETS.len())
            .map(|_| None)
            .collect();
        for (bi, &(chunks, inst_bytes)) in Q4PREFILL_LFM2_ATTN_BUCKETS.iter().enumerate() {
            prefill_attn[bi] = Some(
                device
                    .load_kernel(Q4PREFILL_LFM2_XCLBIN, &inst_words(inst_bytes))
                    .map_err(|e| {
                        anyhow::anyhow!("XDNA load q4prefill_lfm2 attn (c={chunks}) failed: {e}")
                    })?,
            );
        }

        let dec_conv = device
            .load_kernel(Q4LAYER_LFM2_XCLBIN, &inst_words(Q4LAYER_LFM2_CONV_INSTS))
            .map_err(|e| anyhow::anyhow!("XDNA load q4layer_lfm2 conv failed: {e}"))?;

        // Preload every decode attention bucket consecutively right after `dec_conv`
        // so all decode kernels share one XRT hardware context and can be batched
        // together in a single `XdnaRunList`.
        let mut dec_attn: Vec<Option<XdnaKernel>> =
            (0..Q4LAYER_LFM2_KV_BUCKETS.len()).map(|_| None).collect();
        for (bi, &(chunks, inst_bytes)) in Q4LAYER_LFM2_KV_BUCKETS.iter().enumerate() {
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
            prefill_conv,
            prefill_attn,
            prefill_phase_a: None,
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
            let (chunks, inst) = Q4PREFILL_LFM2_ATTN_BUCKETS[bidx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM2_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load prefill attn bucket c={chunks}: {e}"))?;
            self.prefill_attn[bidx] = Some(k);
        }
        Ok(self.prefill_attn[bidx].as_ref().expect("populated bucket"))
    }

    fn prefill_phase_a(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_phase_a.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM2_XCLBIN,
                    &inst_words(Q4PREFILL_LFM2_PHASE_A_INSTS),
                )
                .map_err(|e| format!("load prefill phase_a: {e}"))?;
            self.prefill_phase_a = Some(k);
        }
        Ok(self.prefill_phase_a.as_ref().expect("populated phase_a"))
    }

    fn prefill_fa_part(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_part.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM2_XCLBIN,
                    &inst_words(Q4PREFILL_LFM2_FA_PART_C256_INSTS),
                )
                .map_err(|e| format!("load prefill fa_part: {e}"))?;
            self.prefill_fa_part = Some(k);
        }
        Ok(self.prefill_fa_part.as_ref().expect("populated fa_part"))
    }

    fn prefill_fa_fin(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_fin[bidx].is_none() {
            let (chunks, inst) = Q4PREFILL_LFM2_FA_FIN_BUCKETS[bidx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM2_XCLBIN, &inst_words(inst))
                .map_err(|e| format!("load prefill fa_fin bucket c={chunks}: {e}"))?;
            self.prefill_fa_fin[bidx] = Some(k);
        }
        Ok(self.prefill_fa_fin[bidx]
            .as_ref()
            .expect("populated fa_fin"))
    }

    fn dec_attn(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.dec_attn[bidx].is_none() {
            let (chunks, inst) = Q4LAYER_LFM2_KV_BUCKETS[bidx];
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
                    &inst_words(Q4LAYER_LFM2_FA_PART_C256_INSTS),
                )
                .map_err(|e| format!("load decode fa_part: {e}"))?;
            self.dec_fa_part = Some(k);
        }
        Ok(self.dec_fa_part.as_ref().expect("populated dec_fa_part"))
    }

    fn dec_fa_fin(&mut self, bidx: usize) -> Result<&XdnaKernel, String> {
        if self.dec_fa_fin[bidx].is_none() {
            let (chunks, inst) = Q4LAYER_LFM2_FA_FIN_BUCKETS[bidx];
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
    conv_groups: [Vec<Option<Lfm2PrefillChain>>; 2],
    phase_a_runs: Vec<Option<XdnaRun>>,
    fa_part_runs: Vec<Vec<Option<XdnaRun>>>,
    fa_fin_runs: Vec<Vec<Vec<Option<XdnaRun>>>>,
    n_fin_buckets: usize,
}

impl Lfm2ExtRunCache {
    fn new(n_layers: usize, n_fin_buckets: usize) -> Self {
        Self {
            conv_groups: [
                (0..n_layers).map(|_| None).collect(),
                (0..n_layers).map(|_| None).collect(),
            ],
            phase_a_runs: (0..n_layers).map(|_| None).collect(),
            fa_part_runs: (0..n_layers).map(|_| Vec::new()).collect(),
            fa_fin_runs: (0..n_layers).map(|_| Vec::new()).collect(),
            n_fin_buckets,
        }
    }

    fn ensure_fa_part(
        &mut self,
        l: usize,
        blk: usize,
        make: impl FnOnce() -> Result<XdnaRun, String>,
    ) -> Result<&mut Option<XdnaRun>, String> {
        let row = &mut self.fa_part_runs[l];
        if row.len() <= blk {
            row.resize_with(blk + 1, || None);
        }
        if row[blk].is_none() {
            row[blk] = Some(make()?);
        }
        Ok(&mut row[blk])
    }

    fn ensure_fa_fin(
        &mut self,
        l: usize,
        blk: usize,
        fin_bidx: usize,
        make: impl FnOnce() -> Result<XdnaRun, String>,
    ) -> Result<&mut Option<XdnaRun>, String> {
        let row = &mut self.fa_fin_runs[l];
        let n_b = self.n_fin_buckets;
        while row.len() <= blk {
            row.push((0..n_b).map(|_| None).collect());
        }
        if row[blk][fin_bidx].is_none() {
            row[blk][fin_bidx] = Some(make()?);
        }
        Ok(&mut row[blk][fin_bidx])
    }
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
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    bo_yn: XdnaBo,
    conv_kv_in_kvo: bool,
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
    PreFast {
        parity_idx: usize,
        bidx: usize,
    },
    PreExtConv {
        parity_idx: usize,
        start_l: usize,
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
            InFlightLfm2Run::PreFast { parity_idx, bidx } => self
                .prefill
                .as_ref()
                .and_then(|p| p.bucket_plans[*parity_idx].get(*bidx))
                .and_then(|o| o.as_ref())
                .map(|c| &c.runlist),
            InFlightLfm2Run::PreExtConv {
                parity_idx,
                start_l,
            } => self
                .prefill
                .as_ref()
                .and_then(|p| p.ext_runs.as_ref())
                .and_then(|e| e.conv_groups[*parity_idx].get(*start_l))
                .and_then(|o| o.as_ref())
                .map(|c| &c.runlist),
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

fn pack_weight_stage(
    cpu: &LfmModel,
    wb: &mut [u8],
    tile_off: usize,
    w: &WeightRef,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let raw = cpu.weight_bytes(w);
    pack_q4_0_stage_k1024(&mut wb[tile_off * TILE_BYTES..], raw, rows, cols)
        .map_err(|e| anyhow::anyhow!("pack weight stage at tile {tile_off}: {e}"))?;
    Ok(())
}

fn pack_conv_in_proj(cpu: &LfmModel, wb: &mut [u8], in_proj: &WeightRef) -> Result<()> {
    let raw = cpu.weight_bytes(in_proj);
    let part_bytes = raw.len() / 3;
    for (part, tile_off) in [(0usize, 0usize), (1, 128), (2, 256)] {
        let slice = &raw[part * part_bytes..(part + 1) * part_bytes];
        pack_q4_0_stage_k1024(&mut wb[tile_off * TILE_BYTES..], slice, LFM2_H, LFM2_H)
            .map_err(|e| anyhow::anyhow!("pack conv in_proj part {part}: {e}"))?;
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

pub struct XdnaLfm2Model {
    cpu: LfmModel,
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
        let cpu = LfmModel::from_gguf_with_id_no_repack(gguf, context_size, model_id)?;
        validate_xdna_lfm2_cpu(&cpu)?;
        let cfg = cpu.config();
        let n_layers = cfg.n_layers;
        let kernels = Lfm2Kernels::open()?;

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
                let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
                if is_conv {
                    let (in_proj, out_proj) = cpu
                        .xdna_layer_conv_refs(l)
                        .expect("validated shortconv refs");
                    let conv_wt = cpu.xdna_conv_tap_first(l).expect("validated conv weights");
                    pack_conv_in_proj(&cpu, wb, in_proj)?;
                    pack_weight_stage(&cpu, wb, 384, out_proj, LFM2_H, LFM2_H)?;
                    pack_conv_wt(wb, 512, conv_wt)?;
                    pack_weight_stage(&cpu, wb, 520, fg, LFM2_FFN, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 1032, fu, LFM2_FFN, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 1544, fd, LFM2_H, LFM2_FFN)?;
                } else {
                    let (wq, wk, wv, wo) =
                        cpu.xdna_layer_attn_refs(l).expect("validated attn refs");
                    pack_weight_stage(&cpu, wb, 0, wq, LFM2_Q_DIM, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 128, wk, LFM2_KV_DIM, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 160, wv, LFM2_KV_DIM, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 192, wo, LFM2_H, LFM2_Q_DIM)?;
                    pack_weight_stage(&cpu, wb, 320, fg, LFM2_FFN, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 832, fu, LFM2_FFN, LFM2_H)?;
                    pack_weight_stage(&cpu, wb, 1344, fd, LFM2_H, LFM2_FFN)?;
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
        };

        let warm_guard = crate::backend::acquire_prefill_warm_spin();
        crate::backend::warm_prefill_pool(LfmModel::xdna_head_threads());

        Ok(Self {
            cpu,
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
        }
        if let Some(pre) = npu.prefill.as_mut() {
            pre.conv_kv_in_kvo = false;
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
        Ok(Lfm2PrefillBuffers {
            layer_bo_y,
            layer_bo_kvo,
            bo_yn,
            conv_kv_in_kvo: false,
            bucket_plans: [make_pre_empty(), make_pre_empty()],
            ext_runs: None,
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
        Ok(Lfm2DecodeBuffers {
            layer_bo_y,
            layer_bo_kvo,
            bo_yn,
            conv_kv_in_kvo: false,
            bucket_plans: [make_dec_empty(), make_dec_empty()],
            ext_plans: Lfm2ExtPlanLru::new(2),
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

    #[allow(clippy::too_many_arguments)]
    fn run_decode_ext_multi_block_step(
        kernels: &mut Lfm2Kernels,
        layers: &[Lfm2LayerStatic],
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        dec: &mut Lfm2DecodeBuffers,
        stalled_runs: &mut Vec<XdnaRun>,
        pos: usize,
        eps: f32,
        num_full: usize,
        fin_blk: usize,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<(), String> {
        kernels.dec_phase_a()?;
        kernels.dec_fa_part()?;
        kernels.dec_fa_fin(fin_bidx)?;
        for (l, layer) in layers.iter().enumerate() {
            let (bo_y_head, bo_y_tail) = dec.layer_bo_y.split_at_mut(l + 1);
            let y_curr = &mut bo_y_head[l];
            let y_next = bo_y_tail.first().unwrap_or(&dec.bo_yn);
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&dec.layer_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &dec.layer_bo_kvo[l])
                };
                let run = kernels
                    .dec_conv
                    .create_run(&[&layer_bo_w[l], y_curr, y_next, kv_in, kv_out])
                    .map_err(|e| format!("layer {l} ext dec conv create_run: {e}"))?;
                exec_owned_run(stalled_runs, run, "ext dec conv exec")
                    .map_err(|e| format!("layer {l} {e}"))?;
            } else {
                {
                    let run_a = kernels
                        .dec_phase_a()?
                        .create_run(&[
                            &layer_bo_w[l],
                            y_curr,
                            y_next,
                            &layer_kv_blocks[l][0],
                            &dec.layer_bo_kvo[l],
                        ])
                        .map_err(|e| format!("layer {l} ext dec phase A create_run: {e}"))?;
                    exec_owned_run(stalled_runs, run_a, "ext dec phase A exec")
                        .map_err(|e| format!("layer {l} {e}"))?;
                }

                let k_part = kernels.dec_fa_part()?;
                for (blk, layer_kv) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                    write_lfm3_params_at(
                        y_curr.as_mut_slice(),
                        LFM2_DEC_SLOT_NORM_1 + 4096,
                        pos,
                        1,
                        LFM2_MAX_KV_CHUNKS,
                        pos,
                        true,
                        0,
                        3,
                        eps,
                        blk * LFM2_MAX_KV_CHUNKS,
                        true,
                        true,
                    );
                    y_curr
                        .sync_range_to_device(64, (LFM2_DEC_SLOT_NORM_1 + 4096) * 4)
                        .map_err(|e| format!("layer {l} blk {blk} fa_part params sync: {e}"))?;
                    let run_p = k_part
                        .create_run(&[
                            &layer_bo_w[l],
                            y_curr,
                            y_next,
                            layer_kv,
                            &dec.layer_bo_kvo[l],
                        ])
                        .map_err(|e| {
                            format!("layer {l} blk {blk} ext dec fa_part create_run: {e}")
                        })?;
                    exec_owned_run(stalled_runs, run_p, "ext dec fa_part exec")
                        .map_err(|e| format!("layer {l} blk {blk} {e}"))?;
                }

                let run_fin = kernels
                    .dec_fa_fin(fin_bidx)?
                    .create_run(&[
                        &layer_bo_w[l],
                        y_curr,
                        y_next,
                        &layer_kv_blocks[l][fin_blk],
                        &dec.layer_bo_kvo[l],
                    ])
                    .map_err(|e| format!("layer {l} ext dec fa_fin create_run: {e}"))?;
                exec_owned_run(stalled_runs, run_fin, "ext dec fa_fin exec")
                    .map_err(|e| format!("layer {l} {e}"))?;
            }
        }
        Ok(())
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

        let mut runlist = kernels
            .prefill_conv
            .create_runlist()
            .map_err(|e| format!("prefill create_runlist: {e}"))?;
        for (l, run) in runs.iter().enumerate() {
            runlist
                .add(run)
                .map_err(|e| format!("prefill runlist add layer {l}: {e}"))?;
        }
        Ok(Lfm2PrefillChain {
            _runs: runs,
            runlist,
        })
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
                                let chk_off = (kvh * LFM2_MAX_KV_CHUNKS + chk) * TILE_BYTES;
                                let k_off = chk_off + slot * LFM2_HEAD_DIM * 2;
                                // Prefill kernels read V at +1024*4; decode kernels read V at +2304*4.
                                let v_pre_off = chk_off + 1024 * 4 + slot * LFM2_HEAD_DIM * 2;
                                let v_dec_off = chk_off + 2304 * 4 + slot * LFM2_HEAD_DIM * 2;
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
                                    kv_buf[v_pre_off + d * 2..v_pre_off + d * 2 + 2]
                                        .copy_from_slice(&vb);
                                    kv_buf[v_dec_off + d * 2..v_dec_off + d * 2 + 2]
                                        .copy_from_slice(&vb);
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
                                        blk_chunks * TILE_BYTES,
                                        kvh * LFM2_MAX_KV_CHUNKS * TILE_BYTES,
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
    fn save_attn_prefill_kvo_to_kv_and_state(
        kernels: &Lfm2Kernels,
        kvo: &XdnaBo,
        kv_blocks: &mut Vec<XdnaBo>,
        state: Option<&mut LayerState>,
        pos0: usize,
        n_valid: usize,
        layer: usize,
        use_f16: bool,
    ) -> Result<(), String> {
        let save_sub =
            |kvo_bytes: &[u8], kv_bytes: &mut [u8], sub: usize, loc_chunk: usize, n_sub: usize| {
                let sync_bytes = n_sub * LFM2_HEAD_DIM * 2;
                for kvh in 0..LFM2_N_KV_HEADS {
                    let src_base = (sub * LFM2_N_KV_HEADS + kvh) * (2304 * 4);
                    let dst_base = (kvh * LFM2_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES;
                    kv_bytes[dst_base..dst_base + sync_bytes]
                        .copy_from_slice(&kvo_bytes[src_base..src_base + sync_bytes]);
                    kv_bytes[dst_base + 1024 * 4..dst_base + 1024 * 4 + sync_bytes]
                        .copy_from_slice(
                            &kvo_bytes[src_base + 1024 * 4..src_base + 1024 * 4 + sync_bytes],
                        );
                    kv_bytes[dst_base + 2304 * 4..dst_base + 2304 * 4 + sync_bytes]
                        .copy_from_slice(
                            &kvo_bytes[src_base + 1024 * 4..src_base + 1024 * 4 + sync_bytes],
                        );
                }
            };
        let chunk0 = pos0 / LFM2_KV_CHUNK_T;
        let blk0 = chunk0 / LFM2_MAX_KV_CHUNKS;
        let loc0 = chunk0 % LFM2_MAX_KV_CHUNKS;
        let n_sub0 = n_valid.min(LFM2_KV_CHUNK_T);
        let has_sub1 = n_valid > LFM2_KV_CHUNK_T;
        let max_blk = if has_sub1 {
            (chunk0 + 1) / LFM2_MAX_KV_CHUNKS
        } else {
            blk0
        };
        Self::ensure_paged_block(kernels, kv_blocks, max_blk, layer)?;
        {
            let kvo_bytes = kvo.as_slice();
            save_sub(kvo_bytes, kv_blocks[blk0].as_mut_slice(), 0, loc0, n_sub0);
            if has_sub1 {
                let chunk1 = chunk0 + 1;
                let blk1 = chunk1 / LFM2_MAX_KV_CHUNKS;
                let loc1 = chunk1 % LFM2_MAX_KV_CHUNKS;
                save_sub(
                    kvo_bytes,
                    kv_blocks[blk1].as_mut_slice(),
                    1,
                    loc1,
                    n_valid - LFM2_KV_CHUNK_T,
                );
            }
        }
        let sync_bytes0 = n_sub0 * LFM2_HEAD_DIM * 2;
        for kvh in 0..LFM2_N_KV_HEADS {
            let dst0 = (kvh * LFM2_MAX_KV_CHUNKS + loc0) * TILE_BYTES;
            kv_blocks[blk0]
                .sync_range_to_device(2304 * 4 + sync_bytes0, dst0)
                .map_err(|e| format!("kvh {kvh} prefill sub0 KV bo_kv sync failed: {e}"))?;
            if has_sub1 {
                let chunk1 = chunk0 + 1;
                let blk1 = chunk1 / LFM2_MAX_KV_CHUNKS;
                let loc1 = chunk1 % LFM2_MAX_KV_CHUNKS;
                let sync_bytes1 = (n_valid - LFM2_KV_CHUNK_T) * LFM2_HEAD_DIM * 2;
                let dst1 = (kvh * LFM2_MAX_KV_CHUNKS + loc1) * TILE_BYTES;
                kv_blocks[blk1]
                    .sync_range_to_device(2304 * 4 + sync_bytes1, dst1)
                    .map_err(|e| format!("kvh {kvh} prefill sub1 KV bo_kv sync failed: {e}"))?;
            }
        }
        let kvo_bytes = kvo.as_slice();
        if let Some(LayerState::Attention {
            key_cache,
            value_cache,
            key_cache_f16,
            value_cache_f16,
            ..
        }) = state
        {
            let total_elems = n_valid * LFM2_N_KV_HEADS * LFM2_HEAD_DIM;
            let row_bytes = LFM2_HEAD_DIM * 2;
            if use_f16 {
                key_cache_f16.reserve(total_elems);
                value_cache_f16.reserve(total_elems);
            } else {
                key_cache.reserve(total_elems);
                value_cache.reserve(total_elems);
            }
            for t in 0..n_valid {
                let sub = usize::from(t >= LFM2_KV_CHUNK_T);
                let lt = t % LFM2_KV_CHUNK_T;
                for kvh in 0..LFM2_N_KV_HEADS {
                    let src_base = (sub * LFM2_N_KV_HEADS + kvh) * (2304 * 4);
                    let k_off = src_base + lt * row_bytes;
                    let v_off = src_base + 1024 * 4 + lt * row_bytes;
                    Self::append_bf16_kv_row(
                        &kvo_bytes[k_off..k_off + row_bytes],
                        &kvo_bytes[v_off..v_off + row_bytes],
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        use_f16,
                    );
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn append_bf16_kv_row(
        k_src: &[u8],
        v_src: &[u8],
        key_cache: &mut Vec<f32>,
        value_cache: &mut Vec<f32>,
        key_cache_f16: &mut Vec<u16>,
        value_cache_f16: &mut Vec<u16>,
        use_f16: bool,
    ) {
        if use_f16 {
            key_cache_f16.extend(
                k_src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| crate::quant::bf16_to_f16(u16::from_le_bytes(*p))),
            );
            value_cache_f16.extend(
                v_src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| crate::quant::bf16_to_f16(u16::from_le_bytes(*p))),
            );
        } else {
            key_cache.extend(
                k_src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| f32::from_bits((u16::from_le_bytes(*p) as u32) << 16)),
            );
            value_cache.extend(
                v_src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| f32::from_bits((u16::from_le_bytes(*p) as u32) << 16)),
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
                let dst_chk = (kvh * LFM2_MAX_KV_CHUNKS + chk) * TILE_BYTES;
                let k_off = dst_chk + slot * row_bytes;
                let v_pre_off = dst_chk + 1024 * 4 + slot * row_bytes;
                let v_dec_off = dst_chk + 2304 * 4 + slot * row_bytes;
                kv_bytes[k_off..k_off + row_bytes].copy_from_slice(k_src);
                kv_bytes[v_pre_off..v_pre_off + row_bytes].copy_from_slice(v_src);
                kv_bytes[v_dec_off..v_dec_off + row_bytes].copy_from_slice(v_src);
            }
        }
        for kvh in 0..LFM2_N_KV_HEADS {
            let dst_chk = (kvh * LFM2_MAX_KV_CHUNKS + chk) * TILE_BYTES;
            let k_off = dst_chk + slot * row_bytes;
            bo_kv
                .sync_range_to_device(2304 * 4 + row_bytes, k_off)
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

        let past_chunks_total = pos.div_ceil(LFM2_KV_CHUNK_T);
        let is_ext = past_chunks_total >= LFM2_MAX_KV_CHUNKS;
        let conv_kv_in_kvo = npu.decode.as_ref().expect("dec").conv_kv_in_kvo;
        let parity_idx = usize::from(conv_kv_in_kvo);
        let num_full = past_chunks_total / LFM2_MAX_KV_CHUNKS;
        let fin_past_chunks = past_chunks_total % LFM2_MAX_KV_CHUNKS;
        let fin_blk = if fin_past_chunks > 0 { num_full } else { 0 };
        let cur_tok_blk = pos / LFM2_BLOCK_TOKENS;

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
            ..
        } = *npu;
        let dec = decode.as_mut().expect("initialized decode buffers");
        let Lfm2DecodeBuffers {
            layer_bo_y: dec_bo_y,
            layer_bo_kvo: dec_bo_kvo,
            bo_yn: dec_bo_yn,
            bucket_plans: dec_bucket_plans,
            ext_plans: dec_ext_plans,
            ..
        } = dec;

        let (attn_kv_cap, ext_info, attn_bidx) = if !is_ext {
            let attn_bucket_caps: Vec<usize> =
                Q4LAYER_LFM2_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
            let (bidx, cap) = if pos == 0 {
                (0, 0)
            } else {
                try_select_lfm3_kv_bucket(&attn_bucket_caps, past_chunks_total)?
            };
            if dec_bucket_plans[parity_idx][bidx].is_none() {
                let plan = Self::build_decode_bucket_plan(
                    kernels,
                    &self.layers,
                    layer_bo_w,
                    layer_kv_blocks,
                    dec_bo_y,
                    dec_bo_kvo,
                    dec_bo_yn,
                    bidx,
                    conv_kv_in_kvo,
                )?;
                dec_bucket_plans[parity_idx][bidx] = Some(plan);
            }
            (cap, None, bidx)
        } else {
            let fin_caps: Vec<usize> = Q4LAYER_LFM2_FA_FIN_BUCKETS
                .iter()
                .map(|&(c, _)| c)
                .collect();
            let (fin_bidx, fin_cap) = try_select_lfm3_kv_bucket(&fin_caps, fin_past_chunks)?;
            let max_blk_needed = num_full.saturating_sub(1).max(fin_blk).max(cur_tok_blk);
            for (l, layer) in self.layers.iter().enumerate() {
                if !layer.is_conv {
                    Self::ensure_paged_block(kernels, &mut layer_kv_blocks[l], max_blk_needed, l)?;
                }
            }
            if num_full == 1 {
                let key = (num_full, fin_bidx);
                let _ = dec_ext_plans.get_or_insert_with(parity_idx, key, || {
                    Self::build_decode_ext_plan(
                        kernels,
                        &self.layers,
                        layer_bo_w,
                        layer_kv_blocks,
                        dec_bo_y,
                        dec_bo_kvo,
                        dec_bo_yn,
                        num_full,
                        fin_blk,
                        fin_bidx,
                        conv_kv_in_kvo,
                    )
                })?;
            }
            (
                num_full * LFM2_MAX_KV_CHUNKS + fin_cap,
                Some((num_full, fin_bidx)),
                0,
            )
        };

        write_le_f32_slice(dec.layer_bo_y[0].as_mut_slice(), 0, embedding);
        dec.layer_bo_y[0]
            .sync_range_to_device(LFM2_DEC_YN_BYTES, 0)
            .map_err(|e| format!("decode bo_y[0] hidden sync: {e}"))?;

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
                    0,
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
                    1,
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
                let (num_full, fin_bidx) = ext_info.expect("ext info");
                let fin_cap = Q4LAYER_LFM2_FA_FIN_BUCKETS[fin_bidx].0;
                let cs = compute_lfm2_rope_cs(pos, layer.rope_theta);
                // Mode 2 (Phase A) reads SLOT_NORM_0; Mode 3 (fa_part) reads SLOT_NORM_1;
                // Mode 4 (fa_fin) reads SLOT_ROPE_PKT.
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
                if num_full == 1 {
                    write_lfm3_params_at(
                        yb,
                        LFM2_DEC_SLOT_NORM_1 + 4096,
                        pos,
                        1,
                        LFM2_MAX_KV_CHUNKS,
                        pos,
                        true,
                        0,
                        3,
                        eps,
                        0,
                        true,
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
                    4,
                    eps,
                    num_full * LFM2_MAX_KV_CHUNKS,
                    true,
                    false,
                );
                dec.layer_bo_y[l]
                    .sync_range_to_device(
                        (16 + LFM2_HEAD_DIM) * 4,
                        (LFM2_DEC_SLOT_NORM_0 + 4096) * 4,
                    )
                    .map_err(|e| format!("layer {l} ext dec phase A params+rope sync: {e}"))?;
                if num_full == 1 {
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

        let t_npu0 = prof.then(crate::time::Instant::now);
        let mut pending_next = match ext_info {
            Some((n_full, fin_bidx)) if n_full >= 2 => {
                let pending_next = pending_kv.prepare(pos, LFM2_D_CONV, LFM2_H, LFM2_KV_DIM);
                if !pending_kv.flush_into(state) {
                    *force_conv_sync = true;
                }
                Self::run_decode_ext_multi_block_step(
                    kernels,
                    &self.layers,
                    layer_bo_w,
                    layer_kv_blocks,
                    dec,
                    stalled_runs,
                    pos,
                    eps,
                    n_full,
                    fin_blk,
                    fin_bidx,
                    conv_kv_in_kvo,
                )?;
                pending_next
            }
            _ => {
                let (plan, active_run) = if let Some((n_full, fin_bidx)) = ext_info {
                    let key = (n_full, fin_bidx);
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
            }
        };
        let t_npu_us = t_npu0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let t_kv0 = prof.then(crate::time::Instant::now);
        let mut hidden_out = [0.0f32; LFM2_H];
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
        if pos + 1 >= cfg.max_seq_len.min(LFM2_MAX_TOKENS) && !npu.pending_kv.flush_into(state) {
            npu.synced_len = 0;
            npu.force_conv_sync = true;
        }
        let t_kv_us = t_kv0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

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
        if n_tokens == 0 {
            return self.reject_input("prefill requires non-empty input".to_string(), out_len);
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
            for i in 0..step_count {
                let emb = &embeddings[i * LFM2_H..(i + 1) * LFM2_H];
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

        let n_chunks = rem_tokens.div_ceil(LFM2_PREFILL_CHUNK_T);
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

        for c in 0..n_chunks {
            let local_off = c * LFM2_PREFILL_CHUNK_T;
            let n_valid = (rem_tokens - local_off).min(LFM2_PREFILL_CHUNK_T);
            let pos0 = pos_aligned + local_off;
            let past_chunks_c = pos0.div_ceil(LFM2_KV_CHUNK_T);
            let is_ext_c = past_chunks_c >= LFM2_MAX_KV_CHUNKS;
            let is_last_chunk = c + 1 == n_chunks;
            let emb_off = (cur_start + local_off) * LFM2_H;
            let chunk_embs = &embeddings[emb_off..emb_off + n_valid * LFM2_H];
            let attn_kvo_sync_bytes = if n_valid > LFM2_KV_CHUNK_T {
                15 * 2304 * 4 + 1024 * 4 + (n_valid - LFM2_KV_CHUNK_T) * LFM2_HEAD_DIM * 2
            } else {
                7 * 2304 * 4 + 1024 * 4 + n_valid * LFM2_HEAD_DIM * 2
            };

            if !is_ext_c {
                let (attn_bidx, attn_kv_cap) =
                    match try_select_lfm3_kv_bucket(&attn_caps, past_chunks_c) {
                        Ok(v) => v,
                        Err(e) => return self.record_device_fault(e, out_len),
                    };
                {
                    let npu_ref = &mut *npu;
                    let pre = npu_ref
                        .prefill
                        .as_mut()
                        .expect("initialized prefill buffers");
                    let conv_kv_in_kvo = pre.conv_kv_in_kvo;
                    let parity_idx = usize::from(conv_kv_in_kvo);
                    if pre.bucket_plans[parity_idx][attn_bidx].is_none() {
                        let built = Self::build_prefill_chain(
                            &mut npu_ref.kernels,
                            &self.layers,
                            &npu_ref.layer_bo_w,
                            &npu_ref.layer_kv_blocks,
                            pre,
                            attn_bidx,
                            conv_kv_in_kvo,
                        );
                        pre.bucket_plans[parity_idx][attn_bidx] = Some(match built {
                            Ok(ch) => ch,
                            Err(e) => return self.record_device_fault(e, out_len),
                        });
                    }

                    write_le_f32_slice(pre.layer_bo_y[0].as_mut_slice(), 0, chunk_embs);
                    if n_valid < LFM2_PREFILL_CHUNK_T {
                        let start_b = n_valid * LFM2_H * 4;
                        let end_b = LFM2_PREFILL_CHUNK_T * LFM2_H * 4;
                        pre.layer_bo_y[0].as_mut_slice()[start_b..end_b].fill(0);
                    }
                    if let Err(e) = pre.layer_bo_y[0].sync_range_to_device(LFM2_PREFILL_YN_BYTES, 0)
                    {
                        return self.record_device_fault(
                            format!("prefill c={c} layer 0 hidden sync: {e}"),
                            out_len,
                        );
                    }

                    for (l, layer) in self.layers.iter().enumerate() {
                        let yb = pre.layer_bo_y[l].as_mut_slice();
                        if layer.is_conv {
                            write_lfm2_prefill_step_params(
                                yb, pos0, n_valid, 0, 0, true, 0, 0, eps, None,
                            );
                            if let Err(e) = pre.layer_bo_y[l]
                                .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                            {
                                return self.record_device_fault(
                                    format!("prefill c={c} layer {l} conv params sync: {e}"),
                                    out_len,
                                );
                            }
                        } else {
                            let rope_cs = fill_prefill_rope_cs(pos0, n_valid, layer.rope_theta);
                            write_lfm2_prefill_step_params(
                                yb,
                                pos0,
                                n_valid,
                                attn_kv_cap,
                                pos0,
                                true,
                                0,
                                1,
                                eps,
                                Some(&rope_cs),
                            );
                            if let Err(e) = pre.layer_bo_y[l]
                                .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                            {
                                return self.record_device_fault(
                                    format!("prefill c={c} layer {l} attn params sync: {e}"),
                                    out_len,
                                );
                            }
                            let rope_bytes = if n_valid > LFM2_KV_CHUNK_T {
                                (2304 + (n_valid - LFM2_KV_CHUNK_T) * LFM2_HEAD_DIM) * 4
                            } else {
                                n_valid * LFM2_HEAD_DIM * 4
                            };
                            if let Err(e) = pre.layer_bo_y[l]
                                .sync_range_to_device(rope_bytes, LFM2_PREFILL_SLOT_ROPE_PKT_0 * 4)
                            {
                                return self.record_device_fault(
                                    format!("prefill c={c} layer {l} rope_cs sync: {e}"),
                                    out_len,
                                );
                            }
                        }
                    }

                    let pre_npu0 = self.profile_decode.then(crate::time::Instant::now);
                    let chain = pre.bucket_plans[parity_idx][attn_bidx]
                        .as_ref()
                        .expect("built prefill chain");
                    npu_ref.in_flight = Some(InFlightLfm2Run::PreFast {
                        parity_idx,
                        bidx: attn_bidx,
                    });
                    let exec_res = chain.runlist.execute();
                    let run_res = match exec_res {
                        Ok(()) => {
                            let w = chain.runlist.wait(QUIESCE_TIMEOUT_MS);
                            if w.is_ok() {
                                npu_ref.in_flight = None;
                            }
                            w
                        }
                        Err(e) => {
                            npu_ref.in_flight = None;
                            Err(e)
                        }
                    };
                    if let Err(e) = run_res {
                        return self.record_device_fault(
                            format!("prefill c={c} runlist exec: {e}"),
                            out_len,
                        );
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
                for (l, layer) in self.layers.iter().enumerate() {
                    if layer.is_conv {
                        if is_last_chunk {
                            let kvo_written = if conv_kv_in_kvo {
                                &mut layer_kv_blocks[l][0]
                            } else {
                                &mut pre.layer_bo_kvo[l]
                            };
                            if let Err(e) = kvo_written.sync_range_from_device(LFM2_KVO_BYTES, 0) {
                                return self.record_device_fault(
                                    format!("prefill c={c} layer {l} conv kvo readback: {e}"),
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
                        if let Err(e) =
                            pre.layer_bo_kvo[l].sync_range_from_device(attn_kvo_sync_bytes, 0)
                        {
                            return self.record_device_fault(
                                format!("prefill c={c} layer {l} attn kvo readback: {e}"),
                                out_len,
                            );
                        }
                        if let Err(e) = Self::save_attn_prefill_kvo_to_kv_and_state(
                            &npu_ref.kernels,
                            &pre.layer_bo_kvo[l],
                            &mut layer_kv_blocks[l],
                            Some(&mut state.layers[l]),
                            pos0,
                            n_valid,
                            l,
                            use_f16,
                        ) {
                            return self.record_device_fault(
                                format!("prefill c={c} layer {l} attn save: {e}"),
                                out_len,
                            );
                        }
                    }
                }
                pre.conv_kv_in_kvo = !conv_kv_in_kvo;
                if let Some(t) = t_kv0 {
                    eprintln!(
                        "lfm2 prefill c={c} kv_save={:.0}us",
                        t.elapsed().as_secs_f64() * 1e6
                    );
                }
            } else {
                // >8K paged prefill chunk (Mode 2 Phase A + Mode 4 c=256 + Mode 5 final + batched Conv)
                let num_full = past_chunks_c / LFM2_MAX_KV_CHUNKS;
                let fin_past_chunks = past_chunks_c % LFM2_MAX_KV_CHUNKS;
                let fin_blk = if fin_past_chunks > 0 { num_full } else { 0 };
                let max_kv_blk = ((pos0 + n_valid - 1) / LFM2_BLOCK_TOKENS).max(fin_blk);
                let (fin_bidx, fin_cap) =
                    match try_select_lfm3_kv_bucket(&fin_caps, fin_past_chunks) {
                        Ok(v) => v,
                        Err(e) => return self.record_device_fault(e, out_len),
                    };
                let npu_ref = &mut *npu;
                let XdnaLfm2State {
                    ref mut kernels,
                    ref layer_bo_w,
                    ref mut layer_kv_blocks,
                    ref mut prefill,
                    ref mut in_flight,
                    ref mut stalled_runs,
                    ..
                } = *npu_ref;
                let pre = prefill.as_mut().expect("initialized prefill buffers");
                if pre.ext_runs.is_none() {
                    pre.ext_runs = Some(Lfm2ExtRunCache::new(n_layers, fin_caps.len()));
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
                write_le_f32_slice(pre.layer_bo_y[0].as_mut_slice(), 0, chunk_embs);
                if n_valid < LFM2_PREFILL_CHUNK_T {
                    let start_b = n_valid * LFM2_H * 4;
                    let end_b = LFM2_PREFILL_CHUNK_T * LFM2_H * 4;
                    pre.layer_bo_y[0].as_mut_slice()[start_b..end_b].fill(0);
                }
                if let Err(e) = pre.layer_bo_y[0].sync_range_to_device(LFM2_PREFILL_YN_BYTES, 0) {
                    return self.record_device_fault(
                        format!("prefill >8K c={c} layer 0 hidden sync: {e}"),
                        out_len,
                    );
                }

                let t_ext0 = self.profile_decode.then(crate::time::Instant::now);
                let mut l = 0usize;
                while l < n_layers {
                    if self.layers[l].is_conv {
                        let start_l = l;
                        let mut end_l = l + 1;
                        while end_l < n_layers && self.layers[end_l].is_conv {
                            end_l += 1;
                        }
                        for cl in start_l..end_l {
                            write_lfm2_prefill_step_params(
                                pre.layer_bo_y[cl].as_mut_slice(),
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
                            if let Err(e) = pre.layer_bo_y[cl]
                                .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                            {
                                return self.record_device_fault(
                                    format!("prefill >8K c={c} layer {cl} conv params sync: {e}"),
                                    out_len,
                                );
                            }
                        }
                        let ext = pre.ext_runs.as_mut().expect("ext cache");
                        if ext.conv_groups[parity_idx][start_l].is_none() {
                            let mut runs = Vec::with_capacity(end_l - start_l);
                            for cl in start_l..end_l {
                                let y_next = if cl + 1 < n_layers {
                                    &pre.layer_bo_y[cl + 1]
                                } else {
                                    &pre.bo_yn
                                };
                                let (kv_in, kv_out) = if conv_kv_in_kvo {
                                    (&pre.layer_bo_kvo[cl], &layer_kv_blocks[cl][0])
                                } else {
                                    (&layer_kv_blocks[cl][0], &pre.layer_bo_kvo[cl])
                                };
                                let run = match kernels.prefill_conv.create_run(&[
                                    &layer_bo_w[cl],
                                    &pre.layer_bo_y[cl],
                                    y_next,
                                    kv_in,
                                    kv_out,
                                ]) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        return self.record_device_fault(
                                            format!("prefill >8K layer {cl} conv create_run: {e}"),
                                            out_len,
                                        );
                                    }
                                };
                                runs.push(run);
                            }
                            let mut rl = match kernels.prefill_conv.create_runlist() {
                                Ok(rl) => rl,
                                Err(e) => {
                                    return self.record_device_fault(
                                        format!("prefill >8K conv create_runlist: {e}"),
                                        out_len,
                                    );
                                }
                            };
                            for run in &runs {
                                if let Err(e) = rl.add(run) {
                                    return self.record_device_fault(
                                        format!("prefill >8K conv runlist add: {e}"),
                                        out_len,
                                    );
                                }
                            }
                            ext.conv_groups[parity_idx][start_l] = Some(Lfm2PrefillChain {
                                _runs: runs,
                                runlist: rl,
                            });
                        }
                        let grp = ext.conv_groups[parity_idx][start_l]
                            .as_ref()
                            .expect("populated conv group");
                        *in_flight = Some(InFlightLfm2Run::PreExtConv {
                            parity_idx,
                            start_l,
                        });
                        let exec_res = grp.runlist.execute();
                        let run_res = match exec_res {
                            Ok(()) => {
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
                        if let Err(e) = run_res {
                            return self.record_device_fault(
                                format!(
                                    "prefill >8K c={c} layer {start_l}..{end_l} conv exec: {e}"
                                ),
                                out_len,
                            );
                        }
                        if is_last_chunk {
                            for (cl, kv_blks) in layer_kv_blocks
                                .iter_mut()
                                .enumerate()
                                .take(end_l)
                                .skip(start_l)
                            {
                                let kvo_written = if conv_kv_in_kvo {
                                    &mut kv_blks[0]
                                } else {
                                    &mut pre.layer_bo_kvo[cl]
                                };
                                if let Err(e) =
                                    kvo_written.sync_range_from_device(LFM2_KVO_BYTES, 0)
                                {
                                    return self.record_device_fault(
                                        format!(
                                            "prefill >8K c={c} layer {cl} conv kvo readback: {e}"
                                        ),
                                        out_len,
                                    );
                                }
                                Self::save_conv_kvo_to_state(
                                    kvo_written,
                                    &mut state.layers[cl],
                                    pos0 + n_valid,
                                    true,
                                );
                            }
                        }
                        l = end_l;
                        continue;
                    }

                    let rope_cs = fill_prefill_rope_cs(pos0, n_valid, self.layers[l].rope_theta);
                    write_lfm2_prefill_step_params(
                        pre.layer_bo_y[l].as_mut_slice(),
                        pos0,
                        n_valid,
                        0,
                        pos0,
                        true,
                        0,
                        2,
                        eps,
                        Some(&rope_cs),
                    );
                    if let Err(e) = pre.layer_bo_y[l]
                        .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                    {
                        return self.record_device_fault(
                            format!("prefill >8K c={c} layer {l} phase A params sync: {e}"),
                            out_len,
                        );
                    }
                    let n0 = n_valid.min(LFM2_KV_CHUNK_T);
                    if let Err(e) = pre.layer_bo_y[l].sync_range_to_device(
                        n0 * LFM2_HEAD_DIM * 4,
                        LFM2_PREFILL_SLOT_ROPE_PKT_0 * 4,
                    ) {
                        return self.record_device_fault(
                            format!("prefill >8K c={c} layer {l} rope sub0 sync: {e}"),
                            out_len,
                        );
                    }
                    if n_valid > LFM2_KV_CHUNK_T {
                        let n1 = n_valid - LFM2_KV_CHUNK_T;
                        if let Err(e) = pre.layer_bo_y[l].sync_range_to_device(
                            n1 * LFM2_HEAD_DIM * 4,
                            LFM2_PREFILL_SLOT_ROPE_PKT_1 * 4,
                        ) {
                            return self.record_device_fault(
                                format!("prefill >8K c={c} layer {l} rope sub1 sync: {e}"),
                                out_len,
                            );
                        }
                    }

                    {
                        let ext = pre.ext_runs.as_mut().expect("ext cache");
                        if ext.phase_a_runs[l].is_none() {
                            let y_next = if l + 1 < n_layers {
                                &pre.layer_bo_y[l + 1]
                            } else {
                                &pre.bo_yn
                            };
                            let run = match kernels.prefill_phase_a().and_then(|k| {
                                k.create_run(&[
                                    &layer_bo_w[l],
                                    &pre.layer_bo_y[l],
                                    y_next,
                                    &layer_kv_blocks[l][0],
                                    &pre.layer_bo_kvo[l],
                                ])
                                .map_err(|e| format!("create_run: {e}"))
                            }) {
                                Ok(r) => r,
                                Err(e) => {
                                    return self.record_device_fault(
                                        format!("prefill >8K layer {l} phase A: {e}"),
                                        out_len,
                                    );
                                }
                            };
                            ext.phase_a_runs[l] = Some(run);
                        }
                        if let Err(e) = exec_cached_run_slot(
                            &mut ext.phase_a_runs[l],
                            stalled_runs,
                            "phase A exec",
                        ) {
                            return self.record_device_fault(
                                format!("prefill >8K c={c} layer {l} {e}"),
                                out_len,
                            );
                        }
                    }

                    if let Err(e) =
                        pre.layer_bo_kvo[l].sync_range_from_device(attn_kvo_sync_bytes, 0)
                    {
                        return self.record_device_fault(
                            format!("prefill >8K c={c} layer {l} attn kvo readback: {e}"),
                            out_len,
                        );
                    }
                    if let Err(e) = Self::save_attn_prefill_kvo_to_kv_and_state(
                        kernels,
                        &pre.layer_bo_kvo[l],
                        &mut layer_kv_blocks[l],
                        Some(&mut state.layers[l]),
                        pos0,
                        n_valid,
                        l,
                        use_f16,
                    ) {
                        return self.record_device_fault(
                            format!("prefill >8K c={c} layer {l} attn save: {e}"),
                            out_len,
                        );
                    }

                    for (b, layer_kv) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                        let yb = pre.layer_bo_y[l].as_mut_slice();
                        write_lfm2_prefill_step_params(
                            yb,
                            pos0,
                            n_valid,
                            LFM2_MAX_KV_CHUNKS,
                            pos0,
                            true,
                            0,
                            4,
                            eps,
                            None,
                        );
                        write_lfm2_prefill_ext_params(yb, b * LFM2_MAX_KV_CHUNKS, b > 0, false);
                        if let Err(e) = pre.layer_bo_y[l]
                            .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                        {
                            return self.record_device_fault(
                                format!(
                                    "prefill >8K c={c} layer {l} blk {b} fa_part params sync: {e}"
                                ),
                                out_len,
                            );
                        }
                        let ext = pre.ext_runs.as_mut().expect("ext cache");
                        let layer_w = &layer_bo_w[l];
                        let layer_y = &pre.layer_bo_y[l];
                        let y_next = if l + 1 < n_layers {
                            &pre.layer_bo_y[l + 1]
                        } else {
                            &pre.bo_yn
                        };
                        let layer_kvo = &pre.layer_bo_kvo[l];
                        let run_slot = match ext.ensure_fa_part(l, b, || {
                            let k = kernels.prefill_fa_part()?;
                            k.create_run(&[layer_w, layer_y, y_next, layer_kv, layer_kvo])
                                .map_err(|e| format!("fa_part create_run: {e}"))
                        }) {
                            Ok(r) => r,
                            Err(e) => {
                                return self.record_device_fault(
                                    format!("prefill >8K layer {l} blk {b}: {e}"),
                                    out_len,
                                );
                            }
                        };
                        if let Err(e) = exec_cached_run_slot(run_slot, stalled_runs, "fa_part exec")
                        {
                            return self.record_device_fault(
                                format!("prefill >8K c={c} layer {l} blk {b} {e}"),
                                out_len,
                            );
                        }
                    }

                    {
                        let yb = pre.layer_bo_y[l].as_mut_slice();
                        write_lfm2_prefill_step_params(
                            yb, pos0, n_valid, fin_cap, pos0, true, 0, 5, eps, None,
                        );
                        write_lfm2_prefill_ext_params(
                            yb,
                            num_full * LFM2_MAX_KV_CHUNKS,
                            true,
                            false,
                        );
                        if let Err(e) = pre.layer_bo_y[l]
                            .sync_range_to_device(64, (LFM2_PREFILL_SLOT_NORM_0 + 2048) * 4)
                        {
                            return self.record_device_fault(
                                format!("prefill >8K c={c} layer {l} fa_fin params sync: {e}"),
                                out_len,
                            );
                        }
                        let ext = pre.ext_runs.as_mut().expect("ext cache");
                        let layer_w = &layer_bo_w[l];
                        let layer_y = &pre.layer_bo_y[l];
                        let y_next = if l + 1 < n_layers {
                            &pre.layer_bo_y[l + 1]
                        } else {
                            &pre.bo_yn
                        };
                        let layer_kv = &layer_kv_blocks[l][fin_blk];
                        let layer_kvo = &pre.layer_bo_kvo[l];
                        let run_slot = match ext.ensure_fa_fin(l, fin_blk, fin_bidx, || {
                            let k = kernels.prefill_fa_fin(fin_bidx)?;
                            k.create_run(&[layer_w, layer_y, y_next, layer_kv, layer_kvo])
                                .map_err(|e| format!("fa_fin create_run: {e}"))
                        }) {
                            Ok(r) => r,
                            Err(e) => {
                                return self.record_device_fault(
                                    format!("prefill >8K layer {l}: {e}"),
                                    out_len,
                                );
                            }
                        };
                        if let Err(e) = exec_cached_run_slot(run_slot, stalled_runs, "fa_fin exec")
                        {
                            return self.record_device_fault(
                                format!("prefill >8K c={c} layer {l} {e}"),
                                out_len,
                            );
                        }
                    }
                    l += 1;
                }
                pre.conv_kv_in_kvo = !conv_kv_in_kvo;
                if let Some(t) = t_ext0 {
                    eprintln!(
                        "lfm2 prefill >8K c={c} n={n_valid} bucket={}: total={:.0}us",
                        num_full * LFM2_MAX_KV_CHUNKS + fin_cap,
                        t.elapsed().as_secs_f64() * 1e6
                    );
                }
            }

            let pre = npu.prefill.as_mut().expect("prefill buffers");
            if is_last_chunk && need_logits {
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
        let vocab_size = self.cpu.config().vocab_size;
        if tokens.is_empty() {
            return self.reject_input(
                "forward_prefill requires non-empty tokens".into(),
                vocab_size,
            );
        }
        let Some(embs) = self.cpu.embed_token_rows(tokens) else {
            return self.reject_input(
                format!("token id out of range for vocab {vocab_size}"),
                vocab_size,
            );
        };
        self.run_npu_prefill_embeddings(&embs, tokens.len(), start_pos, state, true)
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
            let logits = self.forward_prefill(chunk, pos, state);
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

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM2.5-1.2B-Instruct-Q4_0.gguf"]
    fn hw_gate_lfm2_long_context_parity() {
        let path = std::path::PathBuf::from(
            r"C:\Users\iamst\.cache\triage\models\huggingface.co\LiquidAI\LFM2.5-1.2B-Instruct-GGUF\resolve\main\LFM2.5-1.2B-Instruct-Q4_0.gguf",
        );
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let ctx_len = 8256usize;
        let gguf = GgufFile::open(&path).expect("open LFM2.5-1.2B-Instruct-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu =
            crate::model::load_model_xdna(gguf, Some(&path), ctx_len).expect("load LFM2.5 on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), ctx_len)
            .expect("load LFM2.5 on CPU");

        let seed = tok
            .encode("The AMD Ryzen AI NPU uses a spatial dataflow array for low-power inference.");
        let prompt_len = 8224usize;
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
            panic!("prefill >8K error: {err}");
        }
        let pre_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "LFM2.5 NPU >8K prefill (n={prompt_len}): cos={pre_cos:.6} npu_top={} cpu_top={cpu_top} time={:.2}s ({:.1} tok/s)",
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
                panic!("decode >8K error: {err}");
            }
            let c = cosine_similarity(&nl, &cl);
            let nt = argmax_idx(&nl);
            let ct = argmax_idx(&cl);
            eprintln!("  >8K decode pos={pos}: cos={c:.6} npu_top={nt} cpu_top={ct}");
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
            let gguf = tiny_lfm2_gguf_with_ctx(32768);
            let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 16512, "test".to_string())
                .expect("load_model_xdna LFM2 at >16K context");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                16512,
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

            // Boundaries around 8K (chunks 255 -> 256) and 16K (chunks 511 -> 512):
            // fast bucket <= 8160, single-runlist ext across 8161..=16352, multi-block at 16353.
            for pos in [8160usize, 8161, 8191, 8192, 16352, 16353] {
                seed_state_to_pos(&npu_model, &mut state, pos);
                let out = model.forward_from_embedding(&emb, pos, &mut state);
                assert_eq!(out.len(), vocab, "pos={pos}");
                assert_eq!(state.seq_len, pos + 1, "pos={pos}");
                assert!(model.take_decode_error().is_none(), "pos={pos}");
            }

            // Aligned n_tokens=3 uses decode bypass; n_tokens=4 & need_logits=false uses prefill;
            // unaligned start_pos=30 with n_tokens=6 steps 2 decodes + 4-token prefill.
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 16512)
                .unwrap();
            let _ = model.forward_prefill(&[1, 2, 3], 0, &mut state);
            assert_eq!(state.seq_len, 3);
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 16512)
                .unwrap();
            let e4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
            let empty = model.forward_prefill_from_embeddings_chunk(&e4, 4, 0, &mut state, false);
            assert!(empty.is_empty());
            assert_eq!(state.seq_len, 4);

            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 16512)
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
            rl_execs, 12,
            "expected exact 12 RunListExecute events (incl. 4 DecExt single-runlist steps), got {rl_execs}"
        );
        // Single-runlist >8K decode uses 3 distinct step_params byte offsets in bo_y.
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
            let gguf = tiny_lfm2_gguf_with_ctx(32768);
            let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 16512, "test".to_string())
                .expect("load_model_xdna LFM2 at >16K context");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                16512,
            )
            .unwrap();

            // Seed CPU KV rows 0..32 with distinct non-uniform K/V values across `(t, kvh, d)`
            // and call `sync_cpu_state_to_npu`, then verify bf16 V lands at both `+1024*4`
            // (prefill V) and `+2304*4` (decode V).
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
                    let chk_off = (kvh * LFM2_MAX_KV_CHUNKS) * TILE_BYTES;
                    let k_off = chk_off + t * LFM2_HEAD_DIM * 2 + d * 2;
                    let v_pre_off = chk_off + 1024 * 4 + t * LFM2_HEAD_DIM * 2 + d * 2;
                    let v_dec_off = chk_off + 2304 * 4 + t * LFM2_HEAD_DIM * 2 + d * 2;
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
                        u16::from_le_bytes(kv_buf[v_pre_off..v_pre_off + 2].try_into().unwrap()),
                        expected_v,
                        "prefill V slot (+1024*4) missing V data"
                    );
                    assert_eq!(
                        u16::from_le_bytes(kv_buf[v_dec_off..v_dec_off + 2].try_into().unwrap()),
                        expected_v,
                        "decode V slot (+2304*4) missing V data"
                    );
                }
            }

            // Exercise `>8K` (`num_full=1, fin_past_chunks=0`), `pos0=8224` (`num_full=1, fin_past_chunks=1`, `fin_blk=1`),
            // and `>16K` (`num_full=2`) aligned continuation prefill chunks via `sync_cpu_state_to_npu`.
            let e4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
            for pos0 in [8192usize, 8224usize, 16384usize] {
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
                        let chk_off = (kvh * LFM2_MAX_KV_CHUNKS + chk) * TILE_BYTES;
                        let rem = check_t % LFM2_KV_CHUNK_T;
                        let k_off = chk_off + rem * LFM2_HEAD_DIM * 2 + 11 * 2;
                        let v_pre_off = chk_off + 1024 * 4 + rem * LFM2_HEAD_DIM * 2 + 11 * 2;
                        let v_dec_off = chk_off + 2304 * 4 + rem * LFM2_HEAD_DIM * 2 + 11 * 2;
                        let slice = guard.layer_kv_blocks[1][blk].as_slice();
                        let k_val = u16::from_le_bytes(slice[k_off..k_off + 2].try_into().unwrap());
                        let v_pre =
                            u16::from_le_bytes(slice[v_pre_off..v_pre_off + 2].try_into().unwrap());
                        let v_dec =
                            u16::from_le_bytes(slice[v_dec_off..v_dec_off + 2].try_into().unwrap());
                        assert_eq!(
                            k_val,
                            crate::backend::xdna::repack::f32_to_bf16(tag),
                            "block {blk} K mismatch at pos0={pos0}, check_t={check_t}, kvh={kvh}"
                        );
                        assert_eq!(
                            v_pre,
                            crate::backend::xdna::repack::f32_to_bf16(-tag),
                            "block {blk} prefill V mismatch at pos0={pos0}, check_t={check_t}, kvh={kvh}"
                        );
                        assert_eq!(
                            v_dec,
                            crate::backend::xdna::repack::f32_to_bf16(-tag),
                            "block {blk} decode V mismatch at pos0={pos0}, check_t={check_t}, kvh={kvh}"
                        );
                    }
                }
            }
        });
        assert!(
            events.iter().any(|e| matches!(e, Event::RunListExecute(_))),
            "expected RunListExecute events during >8K LFM2 continuation prefill"
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
            "expected >8K LFM2 prefill step_params sync at NORM_0"
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
                } if kv_bo_handles.contains(handle) && *size >= TILE_BYTES
            )),
            "expected >8K LFM2 KV host-to-device block sync"
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
                run_wait_state: Some(0),
                ..Default::default()
            },
            || {
                let gguf = tiny_lfm2_gguf_with_ctx(32768);
                let npu_model = XdnaLfm2Model::from_gguf_with_id(gguf, 16512, "test".to_string())
                    .expect("load LFM2 for timeout test");
                let model: &dyn Model = &npu_model;
                let mut state = InferenceState::from_config_capped(
                    model.config(),
                    &crate::kv_cache::KvCompression::F16,
                    16512,
                )
                .unwrap();
                if let LayerState::Attention {
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } = &mut state.layers[1]
                {
                    key_cache_f16.resize(8224 * LFM2_KV_DIM, 0);
                    value_cache_f16.resize(8224 * LFM2_KV_DIM, 0);
                }
                if let LayerState::Conv { buffer, .. } = &mut state.layers[0] {
                    buffer.resize(2 * LFM2_H, 0.25);
                }
                state.seq_len = 8224;
                let e4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
                let out =
                    model.forward_prefill_from_embeddings_chunk(&e4, 4, 8224, &mut state, true);
                assert_eq!(out, vec![0.0; model.config().vocab_size]);
                assert!(model.take_decode_error().is_some());
                assert!(npu_model.state_torn.load(Ordering::SeqCst));
                {
                    let guard = npu_model.lock_npu();
                    assert!(
                        !guard.stalled_runs.is_empty(),
                        "timed-out >8K cached run must move into stalled_runs"
                    );
                }
                let reset_err = model
                    .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 16512)
                    .expect_err("try_reset_kv fails to clear while stalled run still times out");
                assert!(
                    reset_err.to_string().contains("stalled run wait failed"),
                    "unexpected reset err: {reset_err}"
                );
                assert!(npu_model.state_torn.load(Ordering::SeqCst));
                assert!(!npu_model.lock_npu().stalled_runs.is_empty());
            },
        );
    }
}

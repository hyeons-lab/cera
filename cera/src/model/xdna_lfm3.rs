//! AMD XDNA2 NPU-accelerated inference for dense LFM3 models (causal `LFM3` and
//! bidirectional `lfm3-d1` / `d1-2B`).
//!
//! All dense `H=2048, n_heads=16, n_kv_heads=4, head_dim=128, ff=7680,
//! conv_k=7` LFM3 layers execute on the AIE2P array across two compiled kernels:
//! - [`Q4PREFILL_LFM3_XCLBIN`]: unified `B=64` prefill ShortConv, causal/SWA Attention,
//!   bidirectional Phase A/B, and `>8K` paged FA partial/final
//! - [`Q4LAYER_LFM3_XCLBIN`]: `B=1` decode ShortConv, Attention, and `>8K` paged FA
//!
//! Embeddings, final RMSNorm + LM-head projection, and non-XDNA fallback run on CPU.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, bail, ensure};

use crate::backend::xdna::device::{XdnaBo, XdnaDevice, XdnaKernel, XdnaRun, XdnaRunList};
use crate::backend::xdna::kernels::{
    LFM3_CONV_TAP, LFM3_DEC_ACT_BYTES, LFM3_DEC_SLOT_FIN_NORM, LFM3_DEC_SLOT_NORM_0,
    LFM3_DEC_SLOT_NORM_1, LFM3_DEC_SLOT_ROPE_PKT, LFM3_DEC_YN_BYTES, LFM3_FFN_PAD, LFM3_FFN_SRC,
    LFM3_H, LFM3_HEAD_DIM, LFM3_KV_BYTES, LFM3_KV_CHUNK_T, LFM3_KV_DIM, LFM3_KVO_BYTES,
    LFM3_MAX_KV_CHUNKS, LFM3_MAX_TOKENS, LFM3_N_HEADS, LFM3_N_KV_HEADS, LFM3_PREFILL_ACT_BYTES,
    LFM3_PREFILL_B, LFM3_PREFILL_SLOT_ATTN_GATE, LFM3_PREFILL_SLOT_NORM_0,
    LFM3_PREFILL_SLOT_NORM_1, LFM3_PREFILL_SLOT_NORM_2, LFM3_PREFILL_SLOT_NORM_3,
    LFM3_PREFILL_SLOT_Q_BF16, LFM3_PREFILL_SLOT_ROPE_PKT_0A, LFM3_PREFILL_SLOT_ROPE_PKT_0B,
    LFM3_PREFILL_SLOT_ROPE_PKT_1A, LFM3_PREFILL_SLOT_ROPE_PKT_1B, LFM3_PREFILL_YN_BYTES,
    LFM3_Q_DIM, LFM3_W_BYTES, Q4LAYER_LFM3_CONV_INSTS, Q4LAYER_LFM3_FA_FIN_BUCKETS,
    Q4LAYER_LFM3_FA_PART_C256_INSTS, Q4LAYER_LFM3_KV_BUCKETS, Q4LAYER_LFM3_PHASE_A_INSTS,
    Q4LAYER_LFM3_XCLBIN, Q4LAYER_TILE_F32, Q4PREFILL_LFM3_ATTN_BUCKETS,
    Q4PREFILL_LFM3_BIDIR_A_INSTS, Q4PREFILL_LFM3_BIDIR_B_BUCKETS, Q4PREFILL_LFM3_CONV_INSTS,
    Q4PREFILL_LFM3_FA_FIN_BUCKETS, Q4PREFILL_LFM3_FA_PART_C256_INSTS,
    Q4PREFILL_LFM3_FA_PART0_C256_INSTS, Q4PREFILL_LFM3_XCLBIN, init_lfm3_bo_y_norms,
    init_lfm3_prefill_bo_y_norms, inst_words, read_le_f32_slice, select_lfm3_kv_bucket,
    write_le_f32_slice, write_lfm3_ext_params, write_lfm3_params_at, write_lfm3_prefill_ext_params,
    write_lfm3_prefill_step_params, write_lfm3_step_params,
};
use crate::backend::xdna::repack::{
    f32_to_bf16, pack_f32_stage_k1024, pack_q4_0_stage_k1024, pack_q4_0_stage_padded_k1024,
};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError, LayerState};
use crate::model::lfm2::LfmModel;
use crate::model::transformer::{self, WeightRef};
use crate::model::xdna_llama::{
    HybridPendingSlot, PendingHybridCpuKv, check_hybrid_kv_rewind_state,
    validate_xdna_kv_compression,
};
use crate::model::{
    BlockType, Model, ModelConfig, ModelSessionGate, ModelSessionLease, record_first_fault,
    take_fault,
};
use crate::session::CeraError;
use crate::tensor::DType;

const QUIESCE_TIMEOUT_MS: u32 = 30_000;
const TILE_BYTES: usize = Q4LAYER_TILE_F32 * 4;
const CONV_TAPS_STORED: usize = LFM3_CONV_TAP - 1;
const CONV_SYNC_BYTES: usize = 7 * TILE_BYTES + CONV_TAPS_STORED * 256 * 4;
const ATTN_PREFILL_SUB_KVO_BYTES: usize = LFM3_N_KV_HEADS * TILE_BYTES;
const ATTN_PREFILL_KVO_BYTES: usize = 2 * ATTN_PREFILL_SUB_KVO_BYTES;
const ATTN_DEC_KVO_BYTES: usize = LFM3_N_KV_HEADS * 512 * 4;
const ATTN_KV_CHUNK_V_OFFSET: usize = 2304 * 4;
const PHASE_A_Q_BYTES: usize = 73_728 * 4;
const PHASE_A_GATE_BYTES: usize = 2_304 * 4;
const MAX_DEC_KV_CHUNKS: usize = 255;
const PREFILL_STEP_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_0 + 2048) * 4;
const PREFILL_NORM1_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_1 + 2048) * 4;
const PREFILL_NORM2_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_2 + 2048) * 4;
const PREFILL_NORM3_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_3 + 2048) * 4;
const PREFILL_STEP_PARAMS_BYTES: usize = 12 * 4;
const DEC_NORM1_PARAMS_BYTE_OFF: usize = (LFM3_DEC_SLOT_NORM_1 + 4096) * 4;
const DEC_FIN_PARAMS_BYTE_OFF: usize = (LFM3_DEC_SLOT_FIN_NORM + 4096) * 4;
const PREFILL_ROPE_BYTE_OFF: usize = LFM3_PREFILL_SLOT_ROPE_PKT_0A * 4;
const PREFILL_ROPE_SUB0_SYNC_BYTES: usize = (2304 + 2048) * 4;
const PREFILL_ROPE_SYNC_BYTES: usize = LFM3_PREFILL_ACT_BYTES - PREFILL_ROPE_BYTE_OFF;

#[inline]
fn lfm3_swa_keep_chunks(swa_window: usize) -> usize {
    swa_window
        .div_ceil(LFM3_KV_CHUNK_T)
        .min(LFM3_MAX_KV_CHUNKS / 2)
}

#[inline]
fn lfm3_swa_limit_chunks(swa_window: usize) -> usize {
    (lfm3_swa_keep_chunks(swa_window) + 8).min(LFM3_MAX_KV_CHUNKS)
}

#[inline]
fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16)
}

#[inline]
fn bf16_bytes_to_f16(bytes: [u8; 2]) -> u16 {
    crate::quant::f32_to_f16(bf16_bytes_to_f32(bytes))
}

/// Pack `compute_lfm3_rope_cs(pos)` as `[cos[0..64], sin[0..64]]` matching
/// `cpu::rope_neox_freq_factors`.
fn compute_lfm3_rope_cs(
    pos: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) -> [f32; LFM3_HEAD_DIM] {
    let half_dim = LFM3_HEAD_DIM / 2;
    let theta_scale = freq_base.powf(-2.0 / LFM3_HEAD_DIM as f32);
    let mut cs = [0.0f32; LFM3_HEAD_DIM];
    let mut theta = pos as f32;
    for i in 0..half_dim {
        let factor = freq_factors.map_or(1.0, |f| f[i]);
        let (s, c) = (theta / factor).sin_cos();
        cs[i] = c;
        cs[half_dim + i] = s;
        theta *= theta_scale;
    }
    cs
}

fn fill_lfm3_prefill_rope_cs_into(
    pos0: usize,
    n_valid: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
    out_bytes: &mut [u8],
) {
    const SLOTS: [usize; 4] = [
        LFM3_PREFILL_SLOT_ROPE_PKT_0A,
        LFM3_PREFILL_SLOT_ROPE_PKT_0B,
        LFM3_PREFILL_SLOT_ROPE_PKT_1A,
        LFM3_PREFILL_SLOT_ROPE_PKT_1B,
    ];
    for t in 0..LFM3_PREFILL_B {
        let slot = SLOTS[t >> 4];
        let off = slot + (t & 15) * LFM3_HEAD_DIM;
        if t < n_valid {
            let cs = compute_lfm3_rope_cs(pos0 + t, freq_base, freq_factors);
            write_le_f32_slice(out_bytes, off, &cs);
        } else {
            out_bytes[off * 4..(off + LFM3_HEAD_DIM) * 4].fill(0);
        }
    }
}

fn fill_prefill_rope_cs(
    pos0: usize,
    n_valid: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; LFM3_PREFILL_B * LFM3_HEAD_DIM];
    for t in 0..n_valid {
        let cs = compute_lfm3_rope_cs(pos0 + t, freq_base, freq_factors);
        out[t * LFM3_HEAD_DIM..(t + 1) * LFM3_HEAD_DIM].copy_from_slice(&cs);
    }
    out
}

fn dequant_matrix_padded(
    cpu: &LfmModel,
    wref: &WeightRef,
    row_start: usize,
    rows: usize,
    src_cols: usize,
    padded_cols: usize,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; rows * padded_cols];
    let gguf = cpu.gguf();
    for r in 0..rows {
        let dst = &mut buf[r * padded_cols..r * padded_cols + src_cols];
        transformer::dequantize_row_into(gguf, wref, row_start + r, dst);
    }
    buf
}

fn pack_weight_stage(
    cpu: &LfmModel,
    dst_bo: &mut [u8],
    tile_off: usize,
    wref: &WeightRef,
    rows: usize,
    src_cols: usize,
    padded_cols: usize,
) -> Result<()> {
    ensure!(
        wref.m == rows && wref.k == src_cols,
        "weight shape mismatch: got {}x{}, expected {rows}x{src_cols}",
        wref.m,
        wref.k
    );
    let dst = &mut dst_bo[tile_off * TILE_BYTES..];
    if wref.dtype == DType::Q4_0 {
        let raw = cpu.weight_bytes(wref);
        if src_cols == padded_cols {
            pack_q4_0_stage_k1024(dst, raw, rows, padded_cols)
        } else {
            pack_q4_0_stage_padded_k1024(dst, raw, rows, src_cols, padded_cols)
        }
        .map_err(|e| anyhow::anyhow!(e))?;
    } else {
        let f32_buf = dequant_matrix_padded(cpu, wref, 0, rows, src_cols, padded_cols);
        pack_f32_stage_k1024(dst, &f32_buf, rows, padded_cols).map_err(|e| anyhow::anyhow!(e))?;
    }
    Ok(())
}

fn pack_conv_in_proj(cpu: &LfmModel, dst_bo: &mut [u8], wref: &WeightRef) -> Result<()> {
    ensure!(
        wref.m == 3 * LFM3_H && wref.k == LFM3_H,
        "shortconv.in_proj shape mismatch: got {}x{}, expected {}x{}",
        wref.m,
        wref.k,
        3 * LFM3_H,
        LFM3_H
    );
    const TILE_OFFS: [usize; 3] = [0, 128, 256];
    if wref.dtype == DType::Q4_0 {
        let row_bytes = (LFM3_H / 32) * 18;
        let split_bytes = LFM3_H * row_bytes;
        let raw = cpu.weight_bytes(wref);
        ensure!(
            raw.len() >= 3 * split_bytes,
            "shortconv.in_proj buffer too small: {} < {}",
            raw.len(),
            3 * split_bytes
        );
        for (i, &tile_off) in TILE_OFFS.iter().enumerate() {
            let src = &raw[i * split_bytes..(i + 1) * split_bytes];
            pack_q4_0_stage_k1024(&mut dst_bo[tile_off * TILE_BYTES..], src, LFM3_H, LFM3_H)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
    } else {
        for (i, &tile_off) in TILE_OFFS.iter().enumerate() {
            let f32_buf = dequant_matrix_padded(cpu, wref, i * LFM3_H, LFM3_H, LFM3_H, LFM3_H);
            pack_f32_stage_k1024(
                &mut dst_bo[tile_off * TILE_BYTES..],
                &f32_buf,
                LFM3_H,
                LFM3_H,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        }
    }
    Ok(())
}

fn pack_conv_wt(dst_bo: &mut [u8], tile_off: usize, tap_first: &[f32]) -> Result<()> {
    ensure!(
        tap_first.len() == LFM3_CONV_TAP * LFM3_H,
        "shortconv.conv weight len {} != {} * {}",
        tap_first.len(),
        LFM3_CONV_TAP,
        LFM3_H
    );
    for col in 0..8 {
        for k in 0..LFM3_CONV_TAP {
            write_le_f32_slice(
                dst_bo,
                (tile_off + col) * Q4LAYER_TILE_F32 + k * 256,
                &tap_first[k * LFM3_H + col * 256..][..256],
            );
        }
    }
    Ok(())
}

fn validate_xdna_lfm3_cpu(cpu: &LfmModel) -> Result<()> {
    let cfg = cpu.config();
    ensure!(
        cfg.architecture == "lfm3",
        "expected architecture `lfm3`, got `{}`",
        cfg.architecture
    );
    ensure!(
        cfg.moe.is_none(),
        "AMD XDNA LFM3 kernels only support dense LFM3 models (MoE is unsupported)"
    );
    ensure!(
        cfg.hidden_size == LFM3_H,
        "unsupported hidden_size {}: XDNA LFM3 requires {LFM3_H}",
        cfg.hidden_size
    );
    ensure!(
        cfg.n_heads == LFM3_N_HEADS,
        "unsupported n_heads {}: XDNA LFM3 requires {LFM3_N_HEADS}",
        cfg.n_heads
    );
    ensure!(
        cfg.n_kv_heads == LFM3_N_KV_HEADS,
        "unsupported n_kv_heads {}: XDNA LFM3 requires {LFM3_N_KV_HEADS}",
        cfg.n_kv_heads
    );
    ensure!(
        cfg.head_dim == LFM3_HEAD_DIM,
        "unsupported head_dim {}: XDNA LFM3 requires {LFM3_HEAD_DIM}",
        cfg.head_dim
    );
    ensure!(
        cfg.intermediate_size == LFM3_FFN_SRC,
        "unsupported intermediate_size {}: XDNA LFM3 requires {LFM3_FFN_SRC}",
        cfg.intermediate_size
    );
    ensure!(
        cfg.conv_kernel_size == Some(LFM3_CONV_TAP),
        "unsupported conv_kernel_size {:?}: XDNA LFM3 requires {LFM3_CONV_TAP}",
        cfg.conv_kernel_size
    );
    ensure!(
        cfg.max_seq_len > 0 && cfg.max_seq_len <= LFM3_MAX_TOKENS,
        "unsupported context length {}: XDNA LFM3 supports 1..={LFM3_MAX_TOKENS}",
        cfg.max_seq_len
    );
    ensure!(cfg.n_layers > 0, "model has 0 layers");
    ensure!(
        cpu.output_norm_weight().len() == LFM3_H,
        "output_norm length {} != {LFM3_H}",
        cpu.output_norm_weight().len()
    );

    for l in 0..cfg.n_layers {
        ensure!(
            cpu.attn_norm_weight(l).len() == LFM3_H && cpu.ffn_norm_weight(l).len() == LFM3_H,
            "layer {l}: pre-norm length mismatch"
        );
        let p0 = cpu
            .xdna_lfm3_post_attn_norm(l)
            .with_context(|| format!("layer {l}: XDNA LFM3 requires post_attention_norm.weight"))?;
        let p1 = cpu
            .xdna_lfm3_post_ffn_norm(l)
            .with_context(|| format!("layer {l}: XDNA LFM3 requires post_ffw_norm.weight"))?;
        ensure!(
            p0.len() == LFM3_H && p1.len() == LFM3_H,
            "layer {l}: post-norm length mismatch"
        );

        let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
        ensure!(
            fg.m == LFM3_FFN_SRC
                && fg.k == LFM3_H
                && fu.m == LFM3_FFN_SRC
                && fu.k == LFM3_H
                && fd.m == LFM3_H
                && fd.k == LFM3_FFN_SRC,
            "layer {l}: FFN tensor shape mismatch"
        );

        match cfg.block_types[l] {
            BlockType::GatedConv => {
                let (in_proj, out_proj) = cpu
                    .xdna_layer_conv_refs(l)
                    .with_context(|| format!("layer {l}: missing shortconv refs"))?;
                ensure!(
                    in_proj.m == 3 * LFM3_H && in_proj.k == LFM3_H,
                    "layer {l}: shortconv.in_proj shape mismatch"
                );
                ensure!(
                    out_proj.m == LFM3_H && out_proj.k == LFM3_H,
                    "layer {l}: shortconv.out_proj shape mismatch"
                );
                let conv = cpu
                    .xdna_conv_tap_first(l)
                    .with_context(|| format!("layer {l}: missing shortconv.conv weight"))?;
                ensure!(
                    conv.len() == LFM3_CONV_TAP * LFM3_H,
                    "layer {l}: shortconv.conv length mismatch"
                );
            }
            BlockType::Attention => {
                let (wq, wk, wv, wo) = cpu
                    .xdna_layer_attn_refs(l)
                    .with_context(|| format!("layer {l}: missing attention refs"))?;
                let wg = cpu
                    .xdna_lfm3_attn_gate_ref(l)
                    .with_context(|| format!("layer {l}: missing LFM3 attention gate"))?;
                ensure!(
                    wq.m == LFM3_Q_DIM
                        && wq.k == LFM3_H
                        && wk.m == LFM3_KV_DIM
                        && wk.k == LFM3_H
                        && wv.m == LFM3_KV_DIM
                        && wv.k == LFM3_H
                        && wo.m == LFM3_H
                        && wo.k == LFM3_Q_DIM
                        && wg.m == LFM3_N_HEADS
                        && wg.k == LFM3_H,
                    "layer {l}: attention projection shape mismatch"
                );
                let qn = cpu
                    .attn_q_norm_weight(l)
                    .with_context(|| format!("layer {l}: XDNA LFM3 requires attn_q_norm"))?;
                let kn = cpu
                    .attn_k_norm_weight(l)
                    .with_context(|| format!("layer {l}: XDNA LFM3 requires attn_k_norm"))?;
                ensure!(
                    qn.len() == LFM3_HEAD_DIM && kn.len() == LFM3_HEAD_DIM,
                    "layer {l}: q/k norm length mismatch"
                );
            }
            other => bail!("layer {l}: unsupported block type {other:?} for XDNA LFM3"),
        }
    }
    Ok(())
}

struct Lfm3LayerStatic {
    is_conv: bool,
    is_non_causal: bool,
    swa_window: usize,
    rope_theta: f32,
    rope_freq_factors: Option<Vec<f32>>,
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

struct Lfm3Kernels {
    device: XdnaDevice,
    prefill_conv: XdnaKernel,
    prefill_attn: Vec<Option<XdnaKernel>>,
    prefill_bidir_a: Option<XdnaKernel>,
    prefill_bidir_b: Vec<Option<XdnaKernel>>,
    prefill_fa_part0: Option<XdnaKernel>,
    prefill_fa_part: Option<XdnaKernel>,
    prefill_fa_fin: Vec<Option<XdnaKernel>>,
    dec_conv: Option<XdnaKernel>,
    dec_attn: Vec<Option<XdnaKernel>>,
    dec_phase_a: Option<XdnaKernel>,
    dec_fa_part: Option<XdnaKernel>,
    dec_fa_fin: Vec<Option<XdnaKernel>>,
}

impl Lfm3Kernels {
    fn open(is_causal: bool) -> Result<Self> {
        let device = XdnaDevice::open(0).map_err(|e| anyhow::anyhow!("XDNA open failed: {e}"))?;
        let prefill_conv = device
            .load_kernel(
                Q4PREFILL_LFM3_XCLBIN,
                &inst_words(Q4PREFILL_LFM3_CONV_INSTS),
            )
            .map_err(|e| anyhow::anyhow!("XDNA load q4prefill_lfm3 (conv) failed: {e}"))?;

        let mut prefill_attn: Vec<Option<XdnaKernel>> = (0..Q4PREFILL_LFM3_ATTN_BUCKETS.len())
            .map(|_| None)
            .collect();
        prefill_attn[0] = Some(
            device
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_ATTN_BUCKETS[0].1),
                )
                .map_err(|e| anyhow::anyhow!("XDNA load q4prefill_lfm3 (attn c=0) failed: {e}"))?,
        );

        let mut prefill_bidir_a = None;
        let mut prefill_bidir_b: Vec<Option<XdnaKernel>> = (0..Q4PREFILL_LFM3_BIDIR_B_BUCKETS
            .len())
            .map(|_| None)
            .collect();
        let prefill_fa_fin: Vec<Option<XdnaKernel>> = (0..Q4PREFILL_LFM3_FA_FIN_BUCKETS.len())
            .map(|_| None)
            .collect();
        let mut dec_conv = None;
        let mut dec_attn: Vec<Option<XdnaKernel>> =
            (0..Q4LAYER_LFM3_KV_BUCKETS.len()).map(|_| None).collect();
        let dec_fa_fin: Vec<Option<XdnaKernel>> = (0..Q4LAYER_LFM3_FA_FIN_BUCKETS.len())
            .map(|_| None)
            .collect();

        if is_causal {
            dec_conv = Some(
                device
                    .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(Q4LAYER_LFM3_CONV_INSTS))
                    .map_err(|e| anyhow::anyhow!("XDNA load q4layer_lfm3 (conv) failed: {e}"))?,
            );
            // Preload every decode attention bucket consecutively right after `dec_conv`
            // so all decode kernels share one XRT hardware context and can be batched
            // together in a single `XdnaRunList`.
            for (bi, &(chunks, inst_bytes)) in Q4LAYER_LFM3_KV_BUCKETS.iter().enumerate() {
                dec_attn[bi] = Some(
                    device
                        .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(inst_bytes))
                        .map_err(|e| {
                            anyhow::anyhow!("XDNA load q4layer_lfm3 (c={chunks}) failed: {e}")
                        })?,
                );
            }
        } else {
            prefill_bidir_a = Some(
                device
                    .load_kernel(
                        Q4PREFILL_LFM3_XCLBIN,
                        &inst_words(Q4PREFILL_LFM3_BIDIR_A_INSTS),
                    )
                    .map_err(|e| anyhow::anyhow!("XDNA load q4prefill_lfm3 bidir_a failed: {e}"))?,
            );
            prefill_bidir_b[0] = Some(
                device
                    .load_kernel(
                        Q4PREFILL_LFM3_XCLBIN,
                        &inst_words(Q4PREFILL_LFM3_BIDIR_B_BUCKETS[0].1),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("XDNA load q4prefill_lfm3 bidir_b (c=1) failed: {e}")
                    })?,
            );
        }

        Ok(Self {
            device,
            prefill_conv,
            prefill_attn,
            prefill_bidir_a,
            prefill_bidir_b,
            prefill_fa_part0: None,
            prefill_fa_part: None,
            prefill_fa_fin,
            dec_conv,
            dec_attn,
            dec_phase_a: None,
            dec_fa_part: None,
            dec_fa_fin,
        })
    }

    fn prefill_attn(&mut self, bucket_idx: usize) -> Result<&XdnaKernel, String> {
        if self.prefill_attn[bucket_idx].is_none() {
            let (chunks, inst_bytes) = Q4PREFILL_LFM3_ATTN_BUCKETS[bucket_idx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM3_XCLBIN, &inst_words(inst_bytes))
                .map_err(|e| format!("XDNA load q4prefill_lfm3 (attn c={chunks}) failed: {e}"))?;
            self.prefill_attn[bucket_idx] = Some(k);
        }
        Ok(self.prefill_attn[bucket_idx].as_ref().expect("just loaded"))
    }

    fn prefill_bidir_a(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_bidir_a.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_BIDIR_A_INSTS),
                )
                .map_err(|e| format!("XDNA load q4prefill_lfm3 bidir_a failed: {e}"))?;
            self.prefill_bidir_a = Some(k);
        }
        Ok(self.prefill_bidir_a.as_ref().expect("just loaded"))
    }

    fn prefill_bidir_b(&mut self, bucket_idx: usize) -> Result<&XdnaKernel, String> {
        if self.prefill_bidir_b[bucket_idx].is_none() {
            let (chunks, inst_bytes) = Q4PREFILL_LFM3_BIDIR_B_BUCKETS[bucket_idx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM3_XCLBIN, &inst_words(inst_bytes))
                .map_err(|e| {
                    format!("XDNA load q4prefill_lfm3 bidir_b (c={chunks}) failed: {e}")
                })?;
            self.prefill_bidir_b[bucket_idx] = Some(k);
        }
        Ok(self.prefill_bidir_b[bucket_idx]
            .as_ref()
            .expect("just loaded"))
    }

    fn dec_conv(&mut self) -> Result<&XdnaKernel, String> {
        if self.dec_conv.is_none() {
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(Q4LAYER_LFM3_CONV_INSTS))
                .map_err(|e| format!("XDNA load q4layer_lfm3 (conv) failed: {e}"))?;
            self.dec_conv = Some(k);
        }
        Ok(self.dec_conv.as_ref().expect("just loaded"))
    }

    fn dec_attn(&mut self, bucket_idx: usize) -> Result<&XdnaKernel, String> {
        if self.dec_attn[bucket_idx].is_none() {
            let (chunks, inst_bytes) = Q4LAYER_LFM3_KV_BUCKETS[bucket_idx];
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(inst_bytes))
                .map_err(|e| format!("XDNA load q4layer_lfm3 (c={chunks}) failed: {e}"))?;
            self.dec_attn[bucket_idx] = Some(k);
        }
        Ok(self.dec_attn[bucket_idx].as_ref().expect("just loaded"))
    }

    fn prefill_fa_part0(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_part0.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_FA_PART0_C256_INSTS),
                )
                .map_err(|e| format!("XDNA load q4prefill_lfm3 fa_part0 failed: {e}"))?;
            self.prefill_fa_part0 = Some(k);
        }
        Ok(self.prefill_fa_part0.as_ref().expect("just loaded"))
    }

    fn prefill_fa_part(&mut self) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_part.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4PREFILL_LFM3_XCLBIN,
                    &inst_words(Q4PREFILL_LFM3_FA_PART_C256_INSTS),
                )
                .map_err(|e| format!("XDNA load q4prefill_lfm3 fa_part failed: {e}"))?;
            self.prefill_fa_part = Some(k);
        }
        Ok(self.prefill_fa_part.as_ref().expect("just loaded"))
    }

    fn prefill_fa_fin(&mut self, bucket_idx: usize) -> Result<&XdnaKernel, String> {
        if self.prefill_fa_fin[bucket_idx].is_none() {
            let (chunks, inst_bytes) = Q4PREFILL_LFM3_FA_FIN_BUCKETS[bucket_idx];
            let k = self
                .device
                .load_kernel(Q4PREFILL_LFM3_XCLBIN, &inst_words(inst_bytes))
                .map_err(|e| format!("XDNA load q4prefill_lfm3 fa_fin (c={chunks}) failed: {e}"))?;
            self.prefill_fa_fin[bucket_idx] = Some(k);
        }
        Ok(self.prefill_fa_fin[bucket_idx]
            .as_ref()
            .expect("just loaded"))
    }

    fn dec_phase_a(&mut self) -> Result<&XdnaKernel, String> {
        if self.dec_phase_a.is_none() {
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(Q4LAYER_LFM3_PHASE_A_INSTS))
                .map_err(|e| format!("XDNA load q4layer_lfm3 phase_a failed: {e}"))?;
            self.dec_phase_a = Some(k);
        }
        Ok(self.dec_phase_a.as_ref().expect("just loaded"))
    }

    fn dec_fa_part(&mut self) -> Result<&XdnaKernel, String> {
        if self.dec_fa_part.is_none() {
            let k = self
                .device
                .load_kernel(
                    Q4LAYER_LFM3_XCLBIN,
                    &inst_words(Q4LAYER_LFM3_FA_PART_C256_INSTS),
                )
                .map_err(|e| format!("XDNA load q4layer_lfm3 fa_part failed: {e}"))?;
            self.dec_fa_part = Some(k);
        }
        Ok(self.dec_fa_part.as_ref().expect("just loaded"))
    }

    fn dec_fa_fin(&mut self, bucket_idx: usize) -> Result<&XdnaKernel, String> {
        if self.dec_fa_fin[bucket_idx].is_none() {
            let (chunks, inst_bytes) = Q4LAYER_LFM3_FA_FIN_BUCKETS[bucket_idx];
            let k = self
                .device
                .load_kernel(Q4LAYER_LFM3_XCLBIN, &inst_words(inst_bytes))
                .map_err(|e| format!("XDNA load q4layer_lfm3 fa_fin (c={chunks}) failed: {e}"))?;
            self.dec_fa_fin[bucket_idx] = Some(k);
        }
        Ok(self.dec_fa_fin[bucket_idx].as_ref().expect("just loaded"))
    }
}

struct Lfm3DecodeBucketPlan {
    // Drop the runlist before its child run handles so XRT tears down the list cleanly first.
    runlist: XdnaRunList,
    _runs: Vec<XdnaRun>,
}

/// Per-parity bounded LRU cache of compiled NPU runlists so long contexts
/// (`>8K`) cannot accumulate thousands of live WDDM command allocations.
struct Lfm3ParityPlanCache<K: Eq, V> {
    cap: usize,
    entries: [VecDeque<(K, V)>; 2],
}

impl<K: Eq, V> Lfm3ParityPlanCache<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            entries: [VecDeque::new(), VecDeque::new()],
        }
    }

    fn get_or_insert_with<F>(&mut self, parity: bool, key: K, build: F) -> Result<&V, String>
    where
        F: FnOnce() -> Result<V, String>,
    {
        let q = &mut self.entries[usize::from(parity)];
        if let Some(pos) = q.iter().position(|(k, _)| *k == key) {
            if pos + 1 != q.len()
                && let Some(item) = q.remove(pos)
            {
                q.push_back(item);
            }
            return Ok(&q.back().expect("present").1);
        }
        // Evict oldest before creating the next runlist so XRT/WDDM releases old command allocations first.
        while q.len() >= self.cap {
            q.pop_front();
        }
        let val = build()?;
        q.push_back((key, val));
        Ok(&q.back().expect("inserted").1)
    }
}

type Lfm3FastPlanMap = Lfm3ParityPlanCache<(usize, usize), Lfm3DecodeBucketPlan>;
type Lfm3ExtPlanMap = Lfm3ParityPlanCache<(usize, usize, bool, usize), Lfm3DecodeBucketPlan>;

struct XdnaLfm3DecodeBuffers {
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    bo_yn: XdnaBo,
    bucket_plans: Lfm3FastPlanMap,
    ext_plans: Lfm3ExtPlanMap,
    conv_kv_in_kvo: bool,
}

struct XdnaLfm3PrefillBuffers {
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    bucket_plans: Lfm3FastPlanMap,
    ext_plans: Lfm3ExtPlanMap,
    conv_kv_in_kvo: bool,
}

struct XdnaLfm3State {
    kernels: Lfm3Kernels,
    layer_bo_w: Vec<XdnaBo>,
    layer_kv_blocks: Vec<Vec<XdnaBo>>,
    swa_base_chunk: Vec<usize>,
    pre_bo_y: XdnaBo,
    pre_bo_yn: XdnaBo,
    pre_bo_kvo: XdnaBo,
    decode: Option<XdnaLfm3DecodeBuffers>,
    prefill: Option<XdnaLfm3PrefillBuffers>,
    pending_kv: HybridPendingSlot,
    synced_len: usize,
    force_conv_sync: bool,
}

// SAFETY: Every field in `XdnaLfm3State` (including `XdnaBo`/`XdnaKernel` raw XRT handles)
// is only accessed while holding the enclosing `Mutex<XdnaLfm3State>`.
unsafe impl Send for XdnaLfm3State {}

/// AMD XDNA2 NPU model executing dense LFM3 layers (`LFM3` causal and
/// bidirectional `lfm3-d1` / `d1-2B`) on AIE2P and embeddings/head on CPU.
pub struct XdnaLfm3Model {
    cpu: LfmModel,
    layers: Vec<Lfm3LayerStatic>,
    npu: Mutex<XdnaLfm3State>,
    state_torn: AtomicBool,
    decode_error: Mutex<Option<CeraError>>,
    session_gate: ModelSessionGate,
    profile_decode: bool,
    _prefill_warm_guard: Option<crate::backend::PrefillWarmGuard>,
}

impl XdnaLfm3Model {
    /// Load and pack an `lfm3` GGUF onto the AMD XDNA2 NPU.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let arch = gguf
            .get_str("general.architecture")
            .context("missing general.architecture")?
            .to_lowercase();
        ensure!(arch == "lfm3", "expected architecture `lfm3`, got `{arch}`");
        let cpu = LfmModel::from_gguf_with_id(gguf, context_size, model_id)
            .context("failed to load CPU LFM3 weights for XDNA")?;
        validate_xdna_lfm3_cpu(&cpu).context("unsupported LFM3 model for XDNA")?;

        let cfg = cpu.config();
        let n_layers = cfg.n_layers;
        let is_causal = cfg.is_causal;
        let mut kernels = Lfm3Kernels::open(is_causal)?;

        let mut layers = Vec::with_capacity(n_layers);
        let mut layer_bo_w = Vec::with_capacity(n_layers);
        let mut layer_kv_blocks = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let is_conv = matches!(cfg.block_types[l], BlockType::GatedConv);
            let mut bo_w = kernels
                .prefill_conv
                .alloc_bo(LFM3_W_BYTES, 3)
                .map_err(|e| anyhow::anyhow!("layer {l} bo_w alloc failed: {e}"))?;
            {
                let wb = bo_w.as_mut_slice();
                wb.fill(0);
                let (fg, fu, fd) = cpu.xdna_layer_ffn_refs(l)?;
                if is_conv {
                    let (in_proj, out_proj) = cpu
                        .xdna_layer_conv_refs(l)
                        .expect("validated shortconv refs");
                    let conv_wt = cpu
                        .xdna_conv_tap_first(l)
                        .expect("validated shortconv conv weight");
                    pack_conv_in_proj(&cpu, wb, in_proj)?;
                    pack_weight_stage(&cpu, wb, 384, out_proj, LFM3_H, LFM3_H, LFM3_H)?;
                    pack_conv_wt(&mut *wb, 512, conv_wt)?;
                    pack_weight_stage(&cpu, wb, 520, fg, LFM3_FFN_SRC, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 1000, fu, LFM3_FFN_SRC, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 1480, fd, LFM3_H, LFM3_FFN_SRC, LFM3_FFN_PAD)?;
                } else {
                    let (wq, wk, wv, wo) = cpu
                        .xdna_layer_attn_refs(l)
                        .expect("validated attention refs");
                    let wg = cpu
                        .xdna_lfm3_attn_gate_ref(l)
                        .expect("validated attention gate");
                    pack_weight_stage(&cpu, wb, 0, wq, LFM3_Q_DIM, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 128, wk, LFM3_KV_DIM, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 160, wv, LFM3_KV_DIM, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 192, wg, LFM3_N_HEADS, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 193, wo, LFM3_H, LFM3_Q_DIM, LFM3_Q_DIM)?;
                    pack_weight_stage(&cpu, wb, 321, fg, LFM3_FFN_SRC, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 801, fu, LFM3_FFN_SRC, LFM3_H, LFM3_H)?;
                    pack_weight_stage(&cpu, wb, 1281, fd, LFM3_H, LFM3_FFN_SRC, LFM3_FFN_PAD)?;
                }
            }
            bo_w.sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} bo_w sync failed: {e}"))?;
            layer_bo_w.push(bo_w);

            let mut bo_kv = kernels
                .prefill_conv
                .alloc_bo(LFM3_KV_BYTES, 6)
                .map_err(|e| anyhow::anyhow!("layer {l} bo_kv alloc failed: {e}"))?;
            bo_kv.as_mut_slice().fill(0);
            bo_kv
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} bo_kv sync failed: {e}"))?;
            layer_kv_blocks.push(vec![bo_kv]);

            let (rope_theta, rope_freq_factors) = cpu.xdna_lfm3_rope_params(l);
            layers.push(Lfm3LayerStatic {
                is_conv,
                is_non_causal: cpu.layer_is_non_causal(l),
                swa_window: cpu.xdna_lfm3_attn_window(l).unwrap_or(0),
                rope_theta,
                rope_freq_factors: rope_freq_factors.map(|f| f.to_vec()),
                attn_norm: cpu.attn_norm_weight(l).to_vec(),
                post_attn_norm: cpu
                    .xdna_lfm3_post_attn_norm(l)
                    .expect("validated post_attn_norm")
                    .to_vec(),
                ffn_norm: cpu.ffn_norm_weight(l).to_vec(),
                post_ffn_norm: cpu
                    .xdna_lfm3_post_ffn_norm(l)
                    .expect("validated post_ffn_norm")
                    .to_vec(),
                q_norm: cpu.attn_q_norm_weight(l).map(|s| s.to_vec()),
                k_norm: cpu.attn_k_norm_weight(l).map(|s| s.to_vec()),
            });
        }

        let mut pre_bo_y = kernels
            .prefill_conv
            .alloc_bo(LFM3_PREFILL_ACT_BYTES, 4)
            .map_err(|e| anyhow::anyhow!("pre_bo_y alloc failed: {e}"))?;
        pre_bo_y.as_mut_slice().fill(0);
        pre_bo_y
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("pre_bo_y sync failed: {e}"))?;
        let mut pre_bo_yn = kernels
            .prefill_conv
            .alloc_bo(LFM3_PREFILL_YN_BYTES, 5)
            .map_err(|e| anyhow::anyhow!("pre_bo_yn alloc failed: {e}"))?;
        pre_bo_yn.as_mut_slice().fill(0);
        pre_bo_yn
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("pre_bo_yn sync failed: {e}"))?;
        let mut pre_bo_kvo = kernels
            .prefill_conv
            .alloc_bo(LFM3_KVO_BYTES, 7)
            .map_err(|e| anyhow::anyhow!("pre_bo_kvo alloc failed: {e}"))?;
        pre_bo_kvo.as_mut_slice().fill(0);
        pre_bo_kvo
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("pre_bo_kvo sync failed: {e}"))?;

        let decode = if is_causal {
            Some(Self::alloc_decode_buffers(&mut kernels, &layers)?)
        } else {
            None
        };

        Ok(Self {
            cpu,
            layers,
            npu: Mutex::new(XdnaLfm3State {
                kernels,
                layer_bo_w,
                layer_kv_blocks,
                swa_base_chunk: vec![0; n_layers],
                pre_bo_y,
                pre_bo_yn,
                pre_bo_kvo,
                decode,
                prefill: None,
                pending_kv: HybridPendingSlot::default(),
                synced_len: 0,
                force_conv_sync: false,
            }),
            state_torn: AtomicBool::new(false),
            decode_error: Mutex::new(None),
            session_gate: ModelSessionGate::default(),
            profile_decode: crate::backend::cpu_features::profile_decode_enabled(),
            _prefill_warm_guard: is_causal.then(crate::backend::acquire_prefill_warm_spin),
        })
    }

    fn alloc_decode_buffers(
        kernels: &mut Lfm3Kernels,
        layers: &[Lfm3LayerStatic],
    ) -> Result<XdnaLfm3DecodeBuffers> {
        let dec_conv = kernels
            .dec_conv()
            .map_err(|e| anyhow::anyhow!("XDNA load dec_conv failed: {e}"))?;
        let mut layer_bo_y = Vec::with_capacity(layers.len());
        let mut layer_bo_kvo = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let mut bo_y = dec_conv
                .alloc_bo(LFM3_DEC_ACT_BYTES, 4)
                .map_err(|e| anyhow::anyhow!("layer {l} dec_bo_y alloc failed: {e}"))?;
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                init_lfm3_bo_y_norms(
                    yb,
                    LFM3_DEC_SLOT_NORM_0,
                    LFM3_DEC_SLOT_NORM_1,
                    Some(LFM3_DEC_SLOT_ROPE_PKT),
                    &layer.attn_norm,
                    &layer.post_attn_norm,
                    &layer.ffn_norm,
                    &layer.post_ffn_norm,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                );
            }
            bo_y.sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} dec_bo_y sync failed: {e}"))?;
            layer_bo_y.push(bo_y);

            let mut bo_kvo = dec_conv
                .alloc_bo(LFM3_KVO_BYTES, 7)
                .map_err(|e| anyhow::anyhow!("layer {l} dec_bo_kvo alloc failed: {e}"))?;
            bo_kvo.as_mut_slice().fill(0);
            bo_kvo
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("layer {l} dec_bo_kvo sync failed: {e}"))?;
            layer_bo_kvo.push(bo_kvo);
        }
        let mut bo_yn = dec_conv
            .alloc_bo(LFM3_DEC_YN_BYTES, 5)
            .map_err(|e| anyhow::anyhow!("dec_bo_yn alloc failed: {e}"))?;
        bo_yn.as_mut_slice().fill(0);
        bo_yn
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("dec_bo_yn sync failed: {e}"))?;

        Ok(XdnaLfm3DecodeBuffers {
            layer_bo_y,
            layer_bo_kvo,
            bo_yn,
            bucket_plans: Lfm3ParityPlanCache::new(16),
            ext_plans: Lfm3ParityPlanCache::new(2),
            conv_kv_in_kvo: false,
        })
    }

    fn alloc_prefill_buffers(
        kernels: &Lfm3Kernels,
        layers: &[Lfm3LayerStatic],
    ) -> Result<XdnaLfm3PrefillBuffers, String> {
        let mut layer_bo_y = Vec::with_capacity(layers.len());
        let mut layer_bo_kvo = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let mut bo_y = kernels
                .prefill_conv
                .alloc_bo(LFM3_PREFILL_ACT_BYTES, 4)
                .map_err(|e| format!("layer {l} pre_bo_y alloc failed: {e}"))?;
            {
                let yb = bo_y.as_mut_slice();
                yb.fill(0);
                init_lfm3_prefill_bo_y_norms(
                    yb,
                    &layer.attn_norm,
                    &layer.post_attn_norm,
                    &layer.ffn_norm,
                    &layer.post_ffn_norm,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                );
            }
            bo_y.sync_to_device()
                .map_err(|e| format!("layer {l} pre_bo_y sync failed: {e}"))?;
            layer_bo_y.push(bo_y);

            let mut bo_kvo = kernels
                .prefill_conv
                .alloc_bo(LFM3_KVO_BYTES, 7)
                .map_err(|e| format!("layer {l} pre_bo_kvo alloc failed: {e}"))?;
            bo_kvo.as_mut_slice().fill(0);
            bo_kvo
                .sync_to_device()
                .map_err(|e| format!("layer {l} pre_bo_kvo sync failed: {e}"))?;
            layer_bo_kvo.push(bo_kvo);
        }
        Ok(XdnaLfm3PrefillBuffers {
            layer_bo_y,
            layer_bo_kvo,
            bucket_plans: Lfm3ParityPlanCache::new(16),
            ext_plans: Lfm3ParityPlanCache::new(1),
            conv_kv_in_kvo: false,
        })
    }

    fn ensure_paged_block(
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        block_idx: usize,
    ) -> Result<(), String> {
        while blocks.len() <= block_idx {
            let mut bo = allocator
                .alloc_bo(LFM3_KV_BYTES, 6)
                .map_err(|e| format!("paged bo_kv alloc blk={}: {e}", blocks.len()))?;
            bo.as_mut_slice().fill(0);
            bo.sync_to_device()
                .map_err(|e| format!("paged bo_kv sync blk={}: {e}", blocks.len()))?;
            blocks.push(bo);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefill_single_layer_paged_fa(
        kernels: &mut Lfm3Kernels,
        bo_w: &XdnaBo,
        pre_bo_y: &mut XdnaBo,
        pre_bo_yn: &mut XdnaBo,
        kv_blocks: &[XdnaBo],
        pre_bo_kvo: &XdnaBo,
        layer_idx: usize,
        chunk_idx: usize,
        pos0: usize,
        n_valid: usize,
        n_past: usize,
        is_causal: bool,
        skip_cur_kv: bool,
        num_full: usize,
        fin_bidx: usize,
        fin_cap: usize,
        fin_blk: usize,
        eps: f32,
        tag: &str,
    ) -> Result<(), String> {
        for (blk, kv_blk) in kv_blocks.iter().enumerate().take(num_full) {
            let (param_off, sync_off, has_fa_state, use_part0) = if blk == 0 {
                (
                    LFM3_PREFILL_SLOT_NORM_1 + 2048,
                    PREFILL_NORM1_PARAMS_BYTE_OFF,
                    false,
                    true,
                )
            } else {
                (
                    LFM3_PREFILL_SLOT_NORM_2 + 2048,
                    PREFILL_NORM2_PARAMS_BYTE_OFF,
                    true,
                    false,
                )
            };
            {
                let yb = pre_bo_y.as_mut_slice();
                write_lfm3_params_at(
                    yb,
                    param_off,
                    pos0,
                    n_valid,
                    LFM3_MAX_KV_CHUNKS,
                    n_past,
                    is_causal,
                    0,
                    4,
                    eps,
                    blk * LFM3_MAX_KV_CHUNKS,
                    has_fa_state,
                    skip_cur_kv,
                );
            }
            pre_bo_y
                .sync_range_to_device(PREFILL_STEP_PARAMS_BYTES, sync_off)
                .and_then(|()| {
                    let k = if use_part0 {
                        kernels.prefill_fa_part0()?
                    } else {
                        kernels.prefill_fa_part()?
                    };
                    k.create_run(&[bo_w, pre_bo_y, pre_bo_yn, kv_blk, pre_bo_kvo])
                })
                .and_then(|r| {
                    r.start()
                        .and_then(|()| r.wait(QUIESCE_TIMEOUT_MS).map(|_| ()))
                })
                .map_err(|e| {
                    format!("layer {layer_idx} {tag} fa_part c={chunk_idx} blk={blk}: {e}")
                })?;
        }
        {
            let yb = pre_bo_y.as_mut_slice();
            write_lfm3_params_at(
                yb,
                LFM3_PREFILL_SLOT_NORM_3 + 2048,
                pos0,
                n_valid,
                fin_cap,
                n_past,
                is_causal,
                0,
                5,
                eps,
                num_full * LFM3_MAX_KV_CHUNKS,
                num_full > 0,
                skip_cur_kv,
            );
        }
        pre_bo_y
            .sync_range_to_device(PREFILL_STEP_PARAMS_BYTES, PREFILL_NORM3_PARAMS_BYTE_OFF)
            .and_then(|()| {
                kernels.prefill_fa_fin(fin_bidx)?.create_run(&[
                    bo_w,
                    pre_bo_y,
                    pre_bo_yn,
                    &kv_blocks[fin_blk],
                    pre_bo_kvo,
                ])
            })
            .and_then(|r| {
                r.start()
                    .and_then(|()| r.wait(QUIESCE_TIMEOUT_MS).map(|_| ()))
            })
            .and_then(|()| pre_bo_yn.sync_from_device())
            .map_err(|e| format!("layer {layer_idx} {tag} fa_fin c={chunk_idx}: {e}"))?;
        Ok(())
    }

    fn assemble_runlist(
        anchor: &XdnaKernel,
        runs: Vec<XdnaRun>,
        label: &str,
    ) -> Result<Lfm3DecodeBucketPlan, String> {
        let mut runlist = anchor
            .create_runlist()
            .map_err(|e| format!("{label} create_runlist: {e}"))?;
        for (idx, run) in runs.iter().enumerate() {
            runlist
                .add(run)
                .map_err(|e| format!("{label} runlist add #{idx}: {e}"))?;
        }
        Ok(Lfm3DecodeBucketPlan {
            _runs: runs,
            runlist,
        })
    }

    fn ensure_swa_capacity_for_chunk(
        bo_kv: &mut XdnaBo,
        swa_base: &mut usize,
        swa_window: usize,
        global_chunk: usize,
    ) -> Result<(), String> {
        if swa_window == 0 {
            return Ok(());
        }
        let keep_chunks = lfm3_swa_keep_chunks(swa_window);
        let swa_limit = lfm3_swa_limit_chunks(swa_window);
        let rel = global_chunk.saturating_sub(*swa_base);
        if rel < swa_limit {
            return Ok(());
        }
        let shift = rel - keep_chunks;
        let copy_start = shift.min(LFM3_MAX_KV_CHUNKS);
        let end_chunk = rel.min(LFM3_MAX_KV_CHUNKS);
        let n_copy = end_chunk.saturating_sub(copy_start);
        if n_copy > 0 {
            let kb = bo_kv.as_mut_slice();
            for kvh in 0..LFM3_N_KV_HEADS {
                let head_base = kvh * LFM3_MAX_KV_CHUNKS * TILE_BYTES;
                let src0 = head_base + copy_start * TILE_BYTES;
                let src1 = head_base + end_chunk * TILE_BYTES;
                kb.copy_within(src0..src1, head_base);
            }
            for kvh in 0..LFM3_N_KV_HEADS {
                bo_kv
                    .sync_range_to_device(
                        n_copy * TILE_BYTES,
                        kvh * LFM3_MAX_KV_CHUNKS * TILE_BYTES,
                    )
                    .map_err(|e| format!("swa compact kvh {kvh} sync: {e}"))?;
            }
        }
        *swa_base += shift;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_prefill_bucket_plan_slices(
        layers: &[Lfm3LayerStatic],
        kernels: &mut Lfm3Kernels,
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        pre_bo_y_layers: &[XdnaBo],
        pre_bo_kvo_layers: &[XdnaBo],
        pre_bo_yn: &XdnaBo,
        attn_bidx: usize,
        swa_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm3DecodeBucketPlan, String> {
        kernels.prefill_attn(attn_bidx)?;
        kernels.prefill_attn(swa_bidx)?;
        let n_layers = layers.len();
        let mut runs = Vec::with_capacity(n_layers);
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &pre_bo_y_layers[l + 1]
            } else {
                pre_bo_yn
            };
            let run = if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&pre_bo_kvo_layers[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &pre_bo_kvo_layers[l])
                };
                kernels.prefill_conv.create_run(&[
                    &layer_bo_w[l],
                    &pre_bo_y_layers[l],
                    y_next,
                    kv_in,
                    kv_out,
                ])
            } else {
                let bidx = if layer.swa_window > 0 {
                    swa_bidx
                } else {
                    attn_bidx
                };
                kernels.prefill_attn(bidx)?.create_run(&[
                    &layer_bo_w[l],
                    &pre_bo_y_layers[l],
                    y_next,
                    &layer_kv_blocks[l][0],
                    &pre_bo_kvo_layers[l],
                ])
            }
            .map_err(|e| format!("layer {l} prefill create_run: {e}"))?;
            runs.push(run);
        }
        Self::assemble_runlist(&kernels.prefill_conv, runs, "prefill")
    }

    #[allow(clippy::too_many_arguments)]
    fn build_prefill_ext_plan(
        layers: &[Lfm3LayerStatic],
        kernels: &mut Lfm3Kernels,
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        pre_bo_y_layers: &[XdnaBo],
        pre_bo_kvo_layers: &[XdnaBo],
        pre_bo_yn: &XdnaBo,
        swa_bidx: usize,
        num_full: usize,
        has_rem: bool,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm3DecodeBucketPlan, String> {
        kernels.prefill_attn(swa_bidx)?;
        kernels.prefill_bidir_a()?;
        if num_full > 0 {
            kernels.prefill_fa_part0()?;
            if num_full > 1 {
                kernels.prefill_fa_part()?;
            }
        }
        kernels.prefill_fa_fin(fin_bidx)?;
        let n_layers = layers.len();
        let fin_blk = if has_rem { num_full } else { 0 };
        let mut runs = Vec::with_capacity(n_layers * (2 + num_full));
        let k_a = kernels.prefill_bidir_a.as_ref().expect("loaded");
        let k_fin = kernels.prefill_fa_fin[fin_bidx].as_ref().expect("loaded");
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &pre_bo_y_layers[l + 1]
            } else {
                pre_bo_yn
            };
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&pre_bo_kvo_layers[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &pre_bo_kvo_layers[l])
                };
                runs.push(
                    kernels
                        .prefill_conv
                        .create_run(&[&layer_bo_w[l], &pre_bo_y_layers[l], y_next, kv_in, kv_out])
                        .map_err(|e| format!("layer {l} prefill ext conv create_run: {e}"))?,
                );
            } else if layer.swa_window > 0 {
                let k_swa = kernels.prefill_attn[swa_bidx].as_ref().expect("loaded");
                runs.push(
                    k_swa
                        .create_run(&[
                            &layer_bo_w[l],
                            &pre_bo_y_layers[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &pre_bo_kvo_layers[l],
                        ])
                        .map_err(|e| format!("layer {l} prefill ext swa create_run: {e}"))?,
                );
            } else {
                runs.push(
                    k_a.create_run(&[
                        &layer_bo_w[l],
                        &pre_bo_y_layers[l],
                        y_next,
                        &layer_kv_blocks[l][0],
                        &pre_bo_kvo_layers[l],
                    ])
                    .map_err(|e| format!("layer {l} prefill ext phase_a create_run: {e}"))?,
                );
                for (blk, kv_blk) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                    let k_part = if blk == 0 {
                        kernels.prefill_fa_part0.as_ref().expect("loaded")
                    } else {
                        kernels.prefill_fa_part.as_ref().expect("loaded")
                    };
                    runs.push(
                        k_part
                            .create_run(&[
                                &layer_bo_w[l],
                                &pre_bo_y_layers[l],
                                y_next,
                                kv_blk,
                                &pre_bo_kvo_layers[l],
                            ])
                            .map_err(|e| {
                                format!("layer {l} prefill ext fa_part blk={blk} create_run: {e}")
                            })?,
                    );
                }
                runs.push(
                    k_fin
                        .create_run(&[
                            &layer_bo_w[l],
                            &pre_bo_y_layers[l],
                            y_next,
                            &layer_kv_blocks[l][fin_blk],
                            &pre_bo_kvo_layers[l],
                        ])
                        .map_err(|e| format!("layer {l} prefill ext fa_fin create_run: {e}"))?,
                );
            }
        }
        Self::assemble_runlist(&kernels.prefill_conv, runs, "prefill_ext")
    }

    fn lock_npu(&self) -> MutexGuard<'_, XdnaLfm3State> {
        self.npu.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn reject_input_err(&self, err: CeraError, out_len: usize) -> Vec<f32> {
        tracing::error!("cera::xdna: {err}");
        eprintln!("cera::xdna: {err}");
        record_first_fault(&self.decode_error, err);
        vec![0.0f32; out_len]
    }

    fn reject_input(&self, msg: String, out_len: usize) -> Vec<f32> {
        self.reject_input_err(CeraError::Backend(msg), out_len)
    }

    fn record_device_fault(&self, msg: String, out_len: usize) -> Vec<f32> {
        tracing::error!("cera::xdna: {msg}");
        eprintln!("cera::xdna: {msg}");
        self.state_torn.store(true, Ordering::SeqCst);
        record_first_fault(&self.decode_error, CeraError::Backend(msg));
        vec![0.0f32; out_len]
    }

    fn validate_state(&self, state: &InferenceState) -> Result<(), String> {
        if state.is_compressed() {
            return Err(
                "TurboQuant KV compression is not supported by the XDNA backend".to_string(),
            );
        }
        if state.layers.len() < self.layers.len() {
            return Err(format!(
                "InferenceState has {} layers, expected at least {}",
                state.layers.len(),
                self.layers.len()
            ));
        }
        Ok(())
    }

    fn sync_cpu_state_to_npu(
        &self,
        npu: &mut XdnaLfm3State,
        state: &mut InferenceState,
        target_pos: usize,
    ) -> Result<(), String> {
        if target_pos == 0 {
            npu.pending_kv.discard();
        }
        let sync_conv = npu.force_conv_sync || target_pos != npu.synced_len;
        let sync_attn = (target_pos != npu.synced_len)
            || (target_pos == 0 && npu.swa_base_chunk.iter().any(|&b| b != 0));
        if !sync_conv && !sync_attn {
            return Ok(());
        }
        npu.pending_kv.flush_into(state);
        let rewind = target_pos < npu.synced_len;
        let start = if rewind { 0 } else { npu.synced_len };
        let use_f16 = state.kv_f16;
        let needed_kv = target_pos
            .checked_mul(LFM3_KV_DIM)
            .ok_or_else(|| "target_pos overflows KV cache length".to_string())?;

        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv {
                let kv_buf = npu.layer_kv_blocks[l][0].as_mut_slice();
                if sync_conv {
                    if target_pos == 0 {
                        kv_buf[..LFM3_KVO_BYTES].fill(0);
                    } else if let LayerState::Conv { buffer, .. } = &state.layers[l] {
                        let expect = CONV_TAPS_STORED * LFM3_H;
                        if buffer.len() < expect {
                            return Err(format!(
                                "layer {l} conv state len {} < {expect}",
                                buffer.len()
                            ));
                        }
                        kv_buf[..LFM3_KVO_BYTES].fill(0);
                        for col in 0..8 {
                            for k in 0..CONV_TAPS_STORED {
                                write_le_f32_slice(
                                    kv_buf,
                                    col * Q4LAYER_TILE_F32 + k * 256,
                                    &buffer[k * LFM3_H + col * 256..][..256],
                                );
                            }
                        }
                    } else {
                        return Err(format!("layer {l} expected Conv state"));
                    }
                    npu.layer_kv_blocks[l][0]
                        .sync_range_to_device(LFM3_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} conv bo_kv sync failed: {e}"))?;
                }
            } else if sync_attn && (target_pos == 0 || target_pos > start || rewind) {
                if target_pos == 0 {
                    npu.swa_base_chunk[l] = 0;
                    continue;
                }
                match &state.layers[l] {
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } => {
                        if use_f16 {
                            if key_cache_f16.len() < needed_kv || value_cache_f16.len() < needed_kv
                            {
                                return Err(format!(
                                    "layer {l} f16 KV cache too small: {}/{} < {needed_kv}",
                                    key_cache_f16.len(),
                                    value_cache_f16.len()
                                ));
                            }
                        } else if key_cache.len() < needed_kv || value_cache.len() < needed_kv {
                            return Err(format!(
                                "layer {l} f32 KV cache too small: {}/{} < {needed_kv}",
                                key_cache.len(),
                                value_cache.len()
                            ));
                        }
                        let mut eff_start = start;
                        if layer.swa_window > 0 {
                            let swa_limit = lfm3_swa_limit_chunks(layer.swa_window);
                            let last_chunk = (target_pos - 1) / LFM3_KV_CHUNK_T;
                            if rewind {
                                npu.swa_base_chunk[l] = if last_chunk < swa_limit {
                                    0
                                } else {
                                    target_pos.saturating_sub(layer.swa_window) / LFM3_KV_CHUNK_T
                                };
                                eff_start = npu.swa_base_chunk[l] * LFM3_KV_CHUNK_T;
                            } else {
                                Self::ensure_swa_capacity_for_chunk(
                                    &mut npu.layer_kv_blocks[l][0],
                                    &mut npu.swa_base_chunk[l],
                                    layer.swa_window,
                                    last_chunk,
                                )?;
                                eff_start = eff_start.max(npu.swa_base_chunk[l] * LFM3_KV_CHUNK_T);
                            }
                        } else {
                            let max_blk = ((target_pos - 1) / LFM3_KV_CHUNK_T) / LFM3_MAX_KV_CHUNKS;
                            Self::ensure_paged_block(
                                &mut npu.layer_kv_blocks[l],
                                &npu.kernels.prefill_conv,
                                max_blk,
                            )?;
                        }
                        if target_pos <= eff_start {
                            continue;
                        }
                        let base_chunk = if layer.swa_window > 0 {
                            npu.swa_base_chunk[l]
                        } else {
                            0
                        };
                        for t in eff_start..target_pos {
                            let c = t / LFM3_KV_CHUNK_T;
                            let rem = t % LFM3_KV_CHUNK_T;
                            let (blk, loc_c) = if layer.swa_window > 0 {
                                (0, c - base_chunk)
                            } else {
                                (c / LFM3_MAX_KV_CHUNKS, c % LFM3_MAX_KV_CHUNKS)
                            };
                            let kv_buf = npu.layer_kv_blocks[l][blk].as_mut_slice();
                            for kvh in 0..LFM3_N_KV_HEADS {
                                let k_off = (kvh * LFM3_MAX_KV_CHUNKS + loc_c) * TILE_BYTES
                                    + rem * LFM3_HEAD_DIM * 2;
                                let v_off = k_off + ATTN_KV_CHUNK_V_OFFSET;
                                let src = t * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM;
                                for d in 0..LFM3_HEAD_DIM {
                                    let (kf, vf) = if use_f16 {
                                        (
                                            crate::quant::f16_to_f32(key_cache_f16[src + d]),
                                            crate::quant::f16_to_f32(value_cache_f16[src + d]),
                                        )
                                    } else {
                                        (key_cache[src + d], value_cache[src + d])
                                    };
                                    kv_buf[k_off + d * 2..k_off + d * 2 + 2]
                                        .copy_from_slice(&f32_to_bf16(kf).to_le_bytes());
                                    kv_buf[v_off + d * 2..v_off + d * 2 + 2]
                                        .copy_from_slice(&f32_to_bf16(vf).to_le_bytes());
                                }
                            }
                        }
                        if layer.swa_window > 0 {
                            let chunks_to_sync =
                                target_pos.div_ceil(LFM3_KV_CHUNK_T) - npu.swa_base_chunk[l];
                            for kvh in 0..LFM3_N_KV_HEADS {
                                npu.layer_kv_blocks[l][0]
                                    .sync_range_to_device(
                                        chunks_to_sync * TILE_BYTES,
                                        kvh * LFM3_MAX_KV_CHUNKS * TILE_BYTES,
                                    )
                                    .map_err(|e| {
                                        format!("layer {l} kvh {kvh} swa bo_kv sync failed: {e}")
                                    })?;
                            }
                        } else {
                            let c0 = eff_start / LFM3_KV_CHUNK_T;
                            let c1 = target_pos.div_ceil(LFM3_KV_CHUNK_T);
                            let b0 = c0 / LFM3_MAX_KV_CHUNKS;
                            let b1 = (c1 - 1) / LFM3_MAX_KV_CHUNKS;
                            for blk in b0..=b1 {
                                let lo_c = if blk == b0 {
                                    c0 % LFM3_MAX_KV_CHUNKS
                                } else {
                                    0
                                };
                                let hi_c = if blk == b1 {
                                    ((c1 - 1) % LFM3_MAX_KV_CHUNKS) + 1
                                } else {
                                    LFM3_MAX_KV_CHUNKS
                                };
                                for kvh in 0..LFM3_N_KV_HEADS {
                                    npu.layer_kv_blocks[l][blk]
                                        .sync_range_to_device(
                                            (hi_c - lo_c) * TILE_BYTES,
                                            (kvh * LFM3_MAX_KV_CHUNKS + lo_c) * TILE_BYTES,
                                        )
                                        .map_err(|e| {
                                            format!(
                                                "layer {l} blk {blk} kvh {kvh} bo_kv sync failed: {e}"
                                            )
                                        })?;
                                }
                            }
                        }
                    }
                    _ => return Err(format!("layer {l} expected Attention state")),
                }
            }
        }
        npu.synced_len = target_pos;
        npu.force_conv_sync = false;
        if sync_conv {
            if let Some(dec) = npu.decode.as_mut() {
                dec.conv_kv_in_kvo = false;
            }
            if let Some(pre) = npu.prefill.as_mut() {
                pre.conv_kv_in_kvo = false;
            }
        }
        Ok(())
    }

    fn save_conv_kvo_to_state(
        kvo: &XdnaBo,
        layer_state: &mut LayerState,
        next_pos: usize,
        record_history: bool,
    ) {
        let src = &kvo.as_slice()[..LFM3_KVO_BYTES];
        if let LayerState::Conv { buffer, history } = layer_state {
            if buffer.len() < CONV_TAPS_STORED * LFM3_H {
                buffer.resize(CONV_TAPS_STORED * LFM3_H, 0.0);
            }
            for col in 0..8 {
                for k in 0..CONV_TAPS_STORED {
                    read_le_f32_slice(
                        src,
                        col * Q4LAYER_TILE_F32 + k * 256,
                        &mut buffer[k * LFM3_H + col * 256..][..256],
                    );
                }
            }
            if record_history {
                history.push(next_pos, buffer);
            }
        }
    }

    fn save_conv_kvo_to_kv_and_state(
        kvo: &XdnaBo,
        bo_kv: &mut XdnaBo,
        layer_state: Option<&mut LayerState>,
        next_pos: usize,
        record_history: bool,
    ) -> Result<(), String> {
        let src = &kvo.as_slice()[..LFM3_KVO_BYTES];
        bo_kv.as_mut_slice()[..LFM3_KVO_BYTES].copy_from_slice(src);
        bo_kv
            .sync_range_to_device(LFM3_KVO_BYTES, 0)
            .map_err(|e| format!("conv bo_kv sync failed: {e}"))?;

        if let Some(st) = layer_state {
            Self::save_conv_kvo_to_state(kvo, st, next_pos, record_history);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn save_attn_prefill_sub_kvo_to_kv_and_state(
        kvo_bytes: &[u8],
        kvo_head_off: usize,
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        swa_base: &mut usize,
        swa_window: usize,
        layer_state: Option<&mut LayerState>,
        pos0: usize,
        n_valid: usize,
        use_f16: bool,
    ) -> Result<(), String> {
        let global_chunk = pos0 / LFM3_KV_CHUNK_T;
        let (blk, loc_chunk) = if swa_window > 0 {
            Self::ensure_swa_capacity_for_chunk(
                &mut blocks[0],
                swa_base,
                swa_window,
                global_chunk,
            )?;
            let loc_chunk = global_chunk.checked_sub(*swa_base).ok_or_else(|| {
                format!(
                    "prefill swa global_chunk {global_chunk} < swa_base {}",
                    *swa_base
                )
            })?;
            (0, loc_chunk)
        } else {
            let blk = global_chunk / LFM3_MAX_KV_CHUNKS;
            Self::ensure_paged_block(blocks, allocator, blk)?;
            (blk, global_chunk % LFM3_MAX_KV_CHUNKS)
        };
        let bo_kv = &mut blocks[blk];
        for kvh in 0..LFM3_N_KV_HEADS {
            let kvo_off = (kvo_head_off + kvh) * TILE_BYTES;
            let src = &kvo_bytes[kvo_off..kvo_off + TILE_BYTES];
            let dst_off = (kvh * LFM3_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES;
            bo_kv.as_mut_slice()[dst_off..dst_off + TILE_BYTES].copy_from_slice(src);
            bo_kv
                .sync_range_to_device(TILE_BYTES, dst_off)
                .map_err(|e| {
                    format!("attn chunk {global_chunk} kvh {kvh} bo_kv sync failed: {e}")
                })?;
        }
        if let Some(LayerState::Attention {
            key_cache,
            value_cache,
            key_cache_f16,
            value_cache_f16,
            ..
        }) = layer_state
        {
            let needed = (pos0 + n_valid) * LFM3_KV_DIM;
            if use_f16 {
                if key_cache_f16.len() < needed {
                    key_cache_f16.resize(needed, 0);
                }
                if value_cache_f16.len() < needed {
                    value_cache_f16.resize(needed, 0);
                }
            } else {
                if key_cache.len() < needed {
                    key_cache.resize(needed, 0.0);
                }
                if value_cache.len() < needed {
                    value_cache.resize(needed, 0.0);
                }
            }
            for t in 0..n_valid {
                let dst_row = (pos0 + t) * LFM3_KV_DIM;
                for kvh in 0..LFM3_N_KV_HEADS {
                    let k_src = (kvo_head_off + kvh) * TILE_BYTES + t * LFM3_HEAD_DIM * 2;
                    let v_src = k_src + ATTN_KV_CHUNK_V_OFFSET;
                    let dst = dst_row + kvh * LFM3_HEAD_DIM;
                    for d in 0..LFM3_HEAD_DIM {
                        let kb = [kvo_bytes[k_src + d * 2], kvo_bytes[k_src + d * 2 + 1]];
                        let vb = [kvo_bytes[v_src + d * 2], kvo_bytes[v_src + d * 2 + 1]];
                        if use_f16 {
                            key_cache_f16[dst + d] = bf16_bytes_to_f16(kb);
                            value_cache_f16[dst + d] = bf16_bytes_to_f16(vb);
                        } else {
                            key_cache[dst + d] = bf16_bytes_to_f32(kb);
                            value_cache[dst + d] = bf16_bytes_to_f32(vb);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn save_attn_prefill_kvo_to_kv_and_state(
        kvo: &XdnaBo,
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        swa_base: &mut usize,
        swa_window: usize,
        mut layer_state: Option<&mut LayerState>,
        pos0: usize,
        n_valid: usize,
        use_f16: bool,
    ) -> Result<(), String> {
        let kvo_bytes = kvo.as_slice();
        let n0 = n_valid.min(LFM3_KV_CHUNK_T);
        Self::save_attn_prefill_sub_kvo_to_kv_and_state(
            kvo_bytes,
            0,
            blocks,
            allocator,
            swa_base,
            swa_window,
            layer_state.as_deref_mut(),
            pos0,
            n0,
            use_f16,
        )?;
        if n_valid > LFM3_KV_CHUNK_T {
            Self::save_attn_prefill_sub_kvo_to_kv_and_state(
                kvo_bytes,
                LFM3_N_KV_HEADS,
                blocks,
                allocator,
                swa_base,
                swa_window,
                layer_state,
                pos0 + LFM3_KV_CHUNK_T,
                n_valid - LFM3_KV_CHUNK_T,
                use_f16,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn save_attn_dec_kvo_to_kv_and_state(
        kvo: &XdnaBo,
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        swa_base: &mut usize,
        swa_window: usize,
        layer_idx: usize,
        pending_cpu_kv: Option<&mut PendingHybridCpuKv>,
        pos: usize,
    ) -> Result<(), String> {
        let global_chunk = pos / LFM3_KV_CHUNK_T;
        let rem = pos % LFM3_KV_CHUNK_T;
        let (blk, loc_chunk) = if swa_window > 0 {
            Self::ensure_swa_capacity_for_chunk(
                &mut blocks[0],
                swa_base,
                swa_window,
                global_chunk,
            )?;
            let loc_chunk = global_chunk.checked_sub(*swa_base).ok_or_else(|| {
                format!(
                    "decode swa global_chunk {global_chunk} < swa_base {}",
                    *swa_base
                )
            })?;
            (0, loc_chunk)
        } else {
            let blk = global_chunk / LFM3_MAX_KV_CHUNKS;
            Self::ensure_paged_block(blocks, allocator, blk)?;
            (blk, global_chunk % LFM3_MAX_KV_CHUNKS)
        };
        let bo_kv = &mut blocks[blk];
        let kvo_bytes = kvo.as_slice();
        for kvh in 0..LFM3_N_KV_HEADS {
            let k_src = kvh * 512 * 4 + 128 * 4;
            let v_src = kvh * 512 * 4 + 192 * 4;
            let k_dst =
                (kvh * LFM3_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES + rem * LFM3_HEAD_DIM * 2;
            let v_dst = k_dst + ATTN_KV_CHUNK_V_OFFSET;
            {
                let kv_mut = bo_kv.as_mut_slice();
                kv_mut[k_dst..k_dst + LFM3_HEAD_DIM * 2]
                    .copy_from_slice(&kvo_bytes[k_src..k_src + LFM3_HEAD_DIM * 2]);
                kv_mut[v_dst..v_dst + LFM3_HEAD_DIM * 2]
                    .copy_from_slice(&kvo_bytes[v_src..v_src + LFM3_HEAD_DIM * 2]);
            }
            bo_kv
                .sync_range_to_device(ATTN_KV_CHUNK_V_OFFSET + LFM3_HEAD_DIM * 2, k_dst)
                .map_err(|e| format!("decode kvh {kvh} kv bo_kv sync failed: {e}"))?;
        }
        if let Some(pending) = pending_cpu_kv {
            pending.push_attn_bf16(layer_idx, kvo_bytes, LFM3_N_KV_HEADS, LFM3_HEAD_DIM);
        }
        Ok(())
    }

    fn run_prefill_single_chunk(
        kernel: &XdnaKernel,
        bo_w: &XdnaBo,
        bo_y: &mut XdnaBo,
        bo_yn: &mut XdnaBo,
        bo_kv: &XdnaBo,
        bo_kvo: &mut XdnaBo,
        kvo_sync_bytes: usize,
    ) -> Result<(), String> {
        bo_y.sync_to_device()
            .map_err(|e| format!("pre_bo_y sync: {e}"))?;
        bo_yn.as_mut_slice().fill(0);
        bo_yn
            .sync_to_device()
            .map_err(|e| format!("pre_bo_yn sync: {e}"))?;
        bo_kvo.as_mut_slice().fill(0);
        bo_kvo
            .sync_to_device()
            .map_err(|e| format!("pre_bo_kvo sync: {e}"))?;
        let run = kernel
            .create_run(&[bo_w, bo_y, bo_yn, bo_kv, bo_kvo])
            .map_err(|e| format!("prefill create_run: {e}"))?;
        run.start().map_err(|e| format!("prefill start: {e}"))?;
        run.wait(QUIESCE_TIMEOUT_MS)
            .map_err(|e| format!("prefill wait: {e}"))?;
        bo_yn
            .sync_from_device()
            .map_err(|e| format!("pre_bo_yn sync_from_device: {e}"))?;
        if kvo_sync_bytes > 0 {
            bo_kvo
                .sync_range_from_device(kvo_sync_bytes, 0)
                .map_err(|e| format!("pre_bo_kvo sync_from_device: {e}"))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_decode_bucket_plan(
        layers: &[Lfm3LayerStatic],
        kernels: &mut Lfm3Kernels,
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        dec_bo_y: &[XdnaBo],
        dec_bo_kvo: &[XdnaBo],
        dec_bo_yn: &XdnaBo,
        attn_bidx: usize,
        swa_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm3DecodeBucketPlan, String> {
        // Ensure both `dec_conv` and attention buckets are open before binding runs.
        kernels.dec_conv()?;
        kernels.dec_attn(attn_bidx)?;
        kernels.dec_attn(swa_bidx)?;
        let n_layers = layers.len();
        let mut runs = Vec::with_capacity(n_layers);
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &dec_bo_y[l + 1]
            } else {
                dec_bo_yn
            };
            let run = if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&dec_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &dec_bo_kvo[l])
                };
                kernels.dec_conv()?.create_run(&[
                    &layer_bo_w[l],
                    &dec_bo_y[l],
                    y_next,
                    kv_in,
                    kv_out,
                ])
            } else {
                let bidx = if layer.swa_window > 0 {
                    swa_bidx
                } else {
                    attn_bidx
                };
                kernels.dec_attn(bidx)?.create_run(&[
                    &layer_bo_w[l],
                    &dec_bo_y[l],
                    y_next,
                    &layer_kv_blocks[l][0],
                    &dec_bo_kvo[l],
                ])
            }
            .map_err(|e| format!("layer {l} dec create_run: {e}"))?;
            runs.push(run);
        }
        let anchor = kernels.dec_conv()?;
        Self::assemble_runlist(anchor, runs, "dec")
    }

    #[allow(clippy::too_many_arguments)]
    fn build_decode_ext_plan(
        layers: &[Lfm3LayerStatic],
        kernels: &mut Lfm3Kernels,
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        dec_bo_y: &[XdnaBo],
        dec_bo_kvo: &[XdnaBo],
        dec_bo_yn: &XdnaBo,
        swa_bidx: usize,
        num_full: usize,
        has_rem: bool,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<Lfm3DecodeBucketPlan, String> {
        kernels.dec_conv()?;
        kernels.dec_attn(swa_bidx)?;
        kernels.dec_phase_a()?;
        if num_full > 0 {
            kernels.dec_fa_part()?;
        }
        kernels.dec_fa_fin(fin_bidx)?;
        let n_layers = layers.len();
        let fin_blk = if has_rem { num_full } else { 0 };
        let mut runs = Vec::with_capacity(n_layers * (2 + num_full));
        let k_conv = kernels.dec_conv.as_ref().expect("loaded");
        let k_swa = kernels.dec_attn[swa_bidx].as_ref().expect("loaded");
        let k_a = kernels.dec_phase_a.as_ref().expect("loaded");
        let k_part = kernels.dec_fa_part.as_ref();
        let k_fin = kernels.dec_fa_fin[fin_bidx].as_ref().expect("loaded");
        for (l, layer) in layers.iter().enumerate() {
            let y_next = if l + 1 < n_layers {
                &dec_bo_y[l + 1]
            } else {
                dec_bo_yn
            };
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&dec_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &dec_bo_kvo[l])
                };
                runs.push(
                    k_conv
                        .create_run(&[&layer_bo_w[l], &dec_bo_y[l], y_next, kv_in, kv_out])
                        .map_err(|e| format!("layer {l} dec ext conv create_run: {e}"))?,
                );
            } else if layer.swa_window > 0 {
                runs.push(
                    k_swa
                        .create_run(&[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &dec_bo_kvo[l],
                        ])
                        .map_err(|e| format!("layer {l} dec ext swa create_run: {e}"))?,
                );
            } else {
                runs.push(
                    k_a.create_run(&[
                        &layer_bo_w[l],
                        &dec_bo_y[l],
                        y_next,
                        &layer_kv_blocks[l][0],
                        &dec_bo_kvo[l],
                    ])
                    .map_err(|e| format!("layer {l} dec ext phase_a create_run: {e}"))?,
                );
                for (blk, kv_blk) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                    runs.push(
                        k_part
                            .expect("loaded")
                            .create_run(&[
                                &layer_bo_w[l],
                                &dec_bo_y[l],
                                y_next,
                                kv_blk,
                                &dec_bo_kvo[l],
                            ])
                            .map_err(|e| {
                                format!("layer {l} dec ext fa_part blk={blk} create_run: {e}")
                            })?,
                    );
                }
                runs.push(
                    k_fin
                        .create_run(&[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][fin_blk],
                            &dec_bo_kvo[l],
                        ])
                        .map_err(|e| format!("layer {l} dec ext fa_fin create_run: {e}"))?,
                );
            }
        }
        Self::assemble_runlist(k_conv, runs, "dec_ext")
    }

    fn flush_conv_ping_pong_to_kv(&self, npu: &mut XdnaLfm3State) -> Result<(), String> {
        if let Some(dec) = npu.decode.as_mut()
            && dec.conv_kv_in_kvo
        {
            for (l, layer) in self.layers.iter().enumerate() {
                if layer.is_conv {
                    let src = &dec.layer_bo_kvo[l].as_slice()[..LFM3_KVO_BYTES];
                    npu.layer_kv_blocks[l][0].as_mut_slice()[..LFM3_KVO_BYTES].copy_from_slice(src);
                    npu.layer_kv_blocks[l][0]
                        .sync_range_to_device(LFM3_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} flush conv ping-pong: {e}"))?;
                }
            }
            dec.conv_kv_in_kvo = false;
        }
        if let Some(pre) = npu.prefill.as_mut()
            && pre.conv_kv_in_kvo
        {
            for (l, layer) in self.layers.iter().enumerate() {
                if layer.is_conv {
                    let src = &pre.layer_bo_kvo[l].as_slice()[..LFM3_KVO_BYTES];
                    npu.layer_kv_blocks[l][0].as_mut_slice()[..LFM3_KVO_BYTES].copy_from_slice(src);
                    npu.layer_kv_blocks[l][0]
                        .sync_range_to_device(LFM3_KVO_BYTES, 0)
                        .map_err(|e| format!("layer {l} flush prefill conv ping-pong: {e}"))?;
                }
            }
            pre.conv_kv_in_kvo = false;
        }
        Ok(())
    }

    fn run_npu_decode_embedding_step(
        &self,
        npu: &mut XdnaLfm3State,
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
        npu: &mut XdnaLfm3State,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
        logits_out: &mut [f32],
    ) -> Result<(), String> {
        let cfg = self.cpu.config();
        let eps = cfg.rms_norm_eps;
        let is_causal = cfg.is_causal;
        let prof = self.profile_decode;
        let attn_bucket_caps: Vec<usize> =
            Q4LAYER_LFM3_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        let total_past_chunks = pos.div_ceil(LFM3_KV_CHUNK_T);
        let use_fast_plan = total_past_chunks <= MAX_DEC_KV_CHUNKS;
        let (attn_bidx, attn_kv_cap) = if pos == 0 {
            (0, 0)
        } else if use_fast_plan {
            select_lfm3_kv_bucket(&attn_bucket_caps, total_past_chunks)
        } else {
            (0, 0)
        };
        let (num_full, rem) = (
            total_past_chunks / LFM3_MAX_KV_CHUNKS,
            total_past_chunks % LFM3_MAX_KV_CHUNKS,
        );
        let (fin_bidx, fin_cap) = if !use_fast_plan {
            select_lfm3_kv_bucket(&attn_bucket_caps, rem)
        } else {
            (0, 0)
        };
        let has_rem = rem > 0;

        if npu.decode.is_none() {
            npu.decode = Some(
                Self::alloc_decode_buffers(&mut npu.kernels, &self.layers)
                    .map_err(|e| format!("{e:#}"))?,
            );
        }

        let XdnaLfm3State {
            ref mut kernels,
            ref layer_bo_w,
            ref mut layer_kv_blocks,
            ref mut swa_base_chunk,
            ref mut decode,
            ref mut pending_kv,
            ..
        } = *npu;
        let dec = decode.as_mut().expect("initialized decode buffers");
        let parity = dec.conv_kv_in_kvo;
        let XdnaLfm3DecodeBuffers {
            layer_bo_y: dec_bo_y,
            layer_bo_kvo: dec_bo_kvo,
            bo_yn: dec_bo_yn,
            bucket_plans: dec_bucket_plans,
            ext_plans: dec_ext_plans,
            ..
        } = dec;

        let t_prep0 = prof.then(crate::time::Instant::now);
        let step_params_byte_off = (LFM3_DEC_SLOT_NORM_0 + 4096) * 4;
        let step_params_bytes = 12 * 4;

        let mut swa_bidx = attn_bidx;
        for (l, layer) in self.layers.iter().enumerate() {
            let mut sync_ext_part = false;
            let mut sync_ext_fin = false;
            let rope_cs = (!layer.is_conv).then(|| {
                compute_lfm3_rope_cs(pos, layer.rope_theta, layer.rope_freq_factors.as_deref())
            });
            let rope_slice = rope_cs.as_ref().map(|r| &r[..]);
            if !layer.is_conv && layer.swa_window > 0 {
                Self::ensure_swa_capacity_for_chunk(
                    &mut layer_kv_blocks[l][0],
                    &mut swa_base_chunk[l],
                    layer.swa_window,
                    pos / LFM3_KV_CHUNK_T,
                )?;
            }
            {
                let yb = dec_bo_y[l].as_mut_slice();
                if l == 0 {
                    write_le_f32_slice(yb, 0, embedding);
                }
                if layer.is_conv {
                    write_lfm3_step_params(
                        yb,
                        LFM3_DEC_SLOT_NORM_0,
                        None,
                        pos,
                        1,
                        0,
                        pos,
                        true,
                        0,
                        0,
                        eps,
                        None,
                    );
                } else if layer.swa_window > 0 {
                    let local_past = total_past_chunks.saturating_sub(swa_base_chunk[l]);
                    let (sb, swa_cap) = if pos == 0 {
                        (0, 0)
                    } else {
                        select_lfm3_kv_bucket(&attn_bucket_caps, local_past)
                    };
                    swa_bidx = sb;
                    write_lfm3_step_params(
                        yb,
                        LFM3_DEC_SLOT_NORM_0,
                        Some(LFM3_DEC_SLOT_NORM_0 + 4112),
                        pos,
                        1,
                        swa_cap,
                        pos,
                        is_causal,
                        layer.swa_window,
                        1,
                        eps,
                        rope_slice,
                    );
                    write_lfm3_ext_params(
                        yb,
                        LFM3_DEC_SLOT_NORM_0,
                        swa_base_chunk[l],
                        false,
                        false,
                    );
                } else if use_fast_plan {
                    write_lfm3_step_params(
                        yb,
                        LFM3_DEC_SLOT_NORM_0,
                        Some(LFM3_DEC_SLOT_NORM_0 + 4112),
                        pos,
                        1,
                        attn_kv_cap,
                        pos,
                        is_causal,
                        0,
                        1,
                        eps,
                        rope_slice,
                    );
                } else {
                    write_lfm3_step_params(
                        yb,
                        LFM3_DEC_SLOT_NORM_0,
                        Some(LFM3_DEC_SLOT_NORM_0 + 4112),
                        pos,
                        1,
                        0,
                        pos,
                        is_causal,
                        0,
                        2,
                        eps,
                        rope_slice,
                    );
                    if num_full > 0 {
                        write_lfm3_params_at(
                            yb,
                            LFM3_DEC_SLOT_NORM_1 + 4096,
                            pos,
                            1,
                            LFM3_MAX_KV_CHUNKS,
                            pos,
                            is_causal,
                            0,
                            3,
                            eps,
                            0,
                            true,
                            false,
                        );
                        sync_ext_part = true;
                    }
                    write_lfm3_params_at(
                        yb,
                        LFM3_DEC_SLOT_FIN_NORM + 4096,
                        pos,
                        1,
                        fin_cap,
                        pos,
                        is_causal,
                        0,
                        4,
                        eps,
                        num_full * LFM3_MAX_KV_CHUNKS,
                        true,
                        false,
                    );
                    sync_ext_fin = true;
                }
            }
            if l == 0 {
                dec_bo_y[0]
                    .sync_range_to_device(LFM3_H * 4, 0)
                    .map_err(|e| format!("layer 0 hidden sync: {e}"))?;
            }
            let sync_bytes = if layer.is_conv {
                step_params_bytes
            } else {
                (16 + LFM3_HEAD_DIM) * 4
            };
            dec_bo_y[l]
                .sync_range_to_device(sync_bytes, step_params_byte_off)
                .map_err(|e| format!("layer {l} step_params sync: {e}"))?;
            if sync_ext_part {
                dec_bo_y[l]
                    .sync_range_to_device(step_params_bytes, DEC_NORM1_PARAMS_BYTE_OFF)
                    .map_err(|e| format!("layer {l} fa_part step_params sync: {e}"))?;
            }
            if sync_ext_fin {
                dec_bo_y[l]
                    .sync_range_to_device(step_params_bytes, DEC_FIN_PARAMS_BYTE_OFF)
                    .map_err(|e| format!("layer {l} fa_fin step_params sync: {e}"))?;
            }
        }

        let plan = if use_fast_plan {
            let key = (attn_bidx, swa_bidx);
            dec_bucket_plans.get_or_insert_with(parity, key, || {
                Self::build_decode_bucket_plan(
                    &self.layers,
                    kernels,
                    layer_bo_w,
                    layer_kv_blocks,
                    dec_bo_y,
                    dec_bo_kvo,
                    dec_bo_yn,
                    attn_bidx,
                    swa_bidx,
                    parity,
                )
            })?
        } else {
            let key = (swa_bidx, num_full, has_rem, fin_bidx);
            dec_ext_plans.get_or_insert_with(parity, key, || {
                Self::build_decode_ext_plan(
                    &self.layers,
                    kernels,
                    layer_bo_w,
                    layer_kv_blocks,
                    dec_bo_y,
                    dec_bo_kvo,
                    dec_bo_yn,
                    swa_bidx,
                    num_full,
                    has_rem,
                    fin_bidx,
                    parity,
                )
            })?
        };
        let t_prep_us = t_prep0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let t_npu0 = prof.then(crate::time::Instant::now);
        plan.runlist
            .execute()
            .map_err(|e| format!("dec runlist execute: {e}"))?;
        let mut pending_next =
            is_causal.then(|| pending_kv.prepare(pos, CONV_TAPS_STORED, LFM3_H, LFM3_KV_DIM));
        pending_kv.flush_into(state);
        plan.runlist
            .wait(QUIESCE_TIMEOUT_MS)
            .map_err(|e| format!("dec runlist wait: {e}"))?;
        let t_npu_us = t_npu0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        let mut hidden = [0.0f32; LFM3_H];
        dec.conv_kv_in_kvo = !parity;

        let t_kv0 = prof.then(crate::time::Instant::now);
        dec.bo_yn
            .sync_range_from_device(LFM3_DEC_YN_BYTES, 0)
            .map_err(|e| format!("dec_bo_yn sync_from_device: {e}"))?;
        read_le_f32_slice(dec.bo_yn.as_slice(), 0, &mut hidden);

        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv {
                if is_causal {
                    let conv_out_bo = if parity {
                        &mut layer_kv_blocks[l][0]
                    } else {
                        &mut dec.layer_bo_kvo[l]
                    };
                    conv_out_bo
                        .sync_range_from_device(CONV_SYNC_BYTES, 0)
                        .map_err(|e| format!("layer {l} conv kvo sync_from_device: {e}"))?;
                    if let Some(pending) = pending_next.as_mut() {
                        pending.push_conv_from_kvo(l, &conv_out_bo.as_slice()[..CONV_SYNC_BYTES]);
                    }
                }
            } else {
                dec.layer_bo_kvo[l]
                    .sync_range_from_device(ATTN_DEC_KVO_BYTES, 0)
                    .map_err(|e| format!("layer {l} dec_bo_kvo sync_from_device: {e}"))?;
                Self::save_attn_dec_kvo_to_kv_and_state(
                    &dec.layer_bo_kvo[l],
                    &mut layer_kv_blocks[l],
                    &kernels.prefill_conv,
                    &mut swa_base_chunk[l],
                    layer.swa_window,
                    l,
                    pending_next.as_mut(),
                    pos,
                )?;
            }
        }
        let t_kv_us = t_kv0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        if let Some(next) = pending_next {
            pending_kv.commit(next);
        } else {
            pending_kv.discard();
        }
        state.seq_len = pos + 1;
        npu.synced_len = pos + 1;
        if pos + 1 >= cfg.max_seq_len.min(LFM3_MAX_TOKENS) {
            npu.pending_kv.flush_into(state);
        }

        let t_head0 = prof.then(crate::time::Instant::now);
        self.cpu.xdna_project_logits_into(&hidden, logits_out);
        let t_head_us = t_head0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        if prof {
            tracing::info!(
                "lfm3 decode pos={pos} bucket={attn_kv_cap}: prep={t_prep_us:.0}us npu={t_npu_us:.0}us kv={t_kv_us:.0}us head={t_head_us:.0}us"
            );
            eprintln!(
                "lfm3 decode pos={pos} bucket={attn_kv_cap}: prep={t_prep_us:.0}us npu={t_npu_us:.0}us kv={t_kv_us:.0}us head={t_head_us:.0}us"
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
            return self.reject_input("forward_prefill requires non-empty input".into(), out_len);
        }
        if embeddings.len() != n_tokens * LFM3_H {
            return self.reject_input(
                format!(
                    "embeddings length {} != {n_tokens} * {LFM3_H}",
                    embeddings.len()
                ),
                out_len,
            );
        }
        if start_pos + n_tokens > cfg.max_seq_len || start_pos + n_tokens > LFM3_MAX_TOKENS {
            return self.reject_input(
                format!(
                    "prefill range {}..{} exceeds context length {}",
                    start_pos,
                    start_pos + n_tokens,
                    cfg.max_seq_len.min(LFM3_MAX_TOKENS)
                ),
                out_len,
            );
        }
        if !cfg.is_causal && start_pos != 0 {
            return self.reject_input(
                format!("bidirectional XDNA LFM3 prefill requires start_pos == 0, got {start_pos}"),
                out_len,
            );
        }
        if let Err(e) = self.validate_state(state) {
            return self.reject_input(e, out_len);
        }

        let mut npu = self.lock_npu();
        if let Err(e) = self.sync_cpu_state_to_npu(&mut npu, state, start_pos) {
            return self.reject_input(e, out_len);
        }

        // For causal continuation at an unaligned start_pos, step single-token decode until pos % 32 == 0.
        let mut cur_start = 0usize;
        if cfg.is_causal && !start_pos.is_multiple_of(LFM3_KV_CHUNK_T) {
            let align_rem = LFM3_KV_CHUNK_T - (start_pos % LFM3_KV_CHUNK_T);
            let step_count = align_rem.min(n_tokens);
            let mut last_logits = Vec::new();
            for i in 0..step_count {
                let gpos = start_pos + i;
                if let Err(e) = self.sync_cpu_state_to_npu(&mut npu, state, gpos) {
                    return self.reject_input(e, out_len);
                }
                let slice = &embeddings[i * LFM3_H..(i + 1) * LFM3_H];
                match self.run_npu_decode_embedding_step(&mut npu, slice, gpos, state) {
                    Ok(l) => last_logits = l,
                    Err(e) => return self.record_device_fault(e, out_len),
                }
            }
            cur_start = step_count;
            if cur_start == n_tokens {
                return if need_logits { last_logits } else { Vec::new() };
            }
        }

        let pos_aligned = start_pos + cur_start;
        let rem_tokens = n_tokens - cur_start;
        if let Err(e) = self.flush_conv_ping_pong_to_kv(&mut npu) {
            return self.record_device_fault(e, out_len);
        }
        npu.pending_kv.flush_into(state);
        let n_chunks = rem_tokens.div_ceil(LFM3_PREFILL_B);
        let total_tokens = pos_aligned + rem_tokens;
        let eps = cfg.rms_norm_eps;
        let is_causal = cfg.is_causal;
        let use_f16 = state.kv_f16;
        let attn_caps: Vec<usize> = Q4PREFILL_LFM3_ATTN_BUCKETS
            .iter()
            .map(|&(c, _)| c)
            .collect();

        if is_causal {
            if npu.prefill.is_none() {
                match Self::alloc_prefill_buffers(&npu.kernels, &self.layers) {
                    Ok(pb) => npu.prefill = Some(pb),
                    Err(e) => return self.record_device_fault(e, out_len),
                }
            }
            let mut last_hidden = if need_logits {
                vec![0.0f32; LFM3_H]
            } else {
                Vec::new()
            };
            let XdnaLfm3State {
                ref mut kernels,
                ref layer_bo_w,
                ref mut layer_kv_blocks,
                ref mut swa_base_chunk,
                ref mut pre_bo_yn,
                ref mut prefill,
                ..
            } = *npu;
            let pre = prefill.as_mut().expect("initialized prefill buffers");
            let XdnaLfm3PrefillBuffers {
                layer_bo_y: pre_bo_y_layers,
                layer_bo_kvo: pre_bo_kvo_layers,
                bucket_plans: pre_bucket_plans,
                ext_plans: pre_ext_plans,
                conv_kv_in_kvo: pre_conv_kv_in_kvo,
            } = pre;

            let prof = self.profile_decode;
            let t_prefill0 = prof.then(crate::time::Instant::now);
            let mut t_prep_us = 0.0f64;
            let t_conv_npu_us = 0.0f64;
            let mut t_attn_npu_us = 0.0f64;
            let mut t_conv_kv_us = 0.0f64;
            let mut t_attn_kv_us = 0.0f64;

            for c in 0..n_chunks {
                let tc0 = prof.then(crate::time::Instant::now);
                let local_off = c * LFM3_PREFILL_B;
                let n_valid = (rem_tokens - local_off).min(LFM3_PREFILL_B);
                let pos0 = pos_aligned + local_off;
                let needed_past = pos0 / LFM3_KV_CHUNK_T;
                let use_fast_pre = needed_past <= 255;
                let parity = *pre_conv_kv_in_kvo;
                let rope_sync_bytes = if n_valid <= LFM3_KV_CHUNK_T {
                    PREFILL_ROPE_SUB0_SYNC_BYTES
                } else {
                    PREFILL_ROPE_SYNC_BYTES
                };
                let attn_kvo_bytes = if n_valid <= LFM3_KV_CHUNK_T {
                    ATTN_PREFILL_SUB_KVO_BYTES
                } else {
                    ATTN_PREFILL_KVO_BYTES
                };

                {
                    let yb0 = pre_bo_y_layers[0].as_mut_slice();
                    let src_start = (cur_start + local_off) * LFM3_H;
                    write_le_f32_slice(
                        yb0,
                        0,
                        &embeddings[src_start..src_start + n_valid * LFM3_H],
                    );
                    if n_valid < LFM3_PREFILL_B {
                        yb0[n_valid * LFM3_H * 4..LFM3_PREFILL_YN_BYTES].fill(0);
                    }
                }
                let hidden_sync_bytes = if n_valid < LFM3_PREFILL_B {
                    LFM3_PREFILL_YN_BYTES
                } else {
                    n_valid * LFM3_H * 4
                };
                if let Err(e) = pre_bo_y_layers[0].sync_range_to_device(hidden_sync_bytes, 0) {
                    return self.record_device_fault(
                        format!("prefill c={c} layer 0 hidden sync: {e}"),
                        out_len,
                    );
                }

                let (attn_bidx, attn_kv_cap) = if use_fast_pre {
                    select_lfm3_kv_bucket(&attn_caps, needed_past)
                } else {
                    (0, 0)
                };
                let (num_full, rem) = (
                    needed_past / LFM3_MAX_KV_CHUNKS,
                    needed_past % LFM3_MAX_KV_CHUNKS,
                );
                let (fin_bidx, fin_cap) = if !use_fast_pre {
                    select_lfm3_kv_bucket(&attn_caps, rem)
                } else {
                    (0, 0)
                };
                let has_rem = rem > 0;
                let mut swa_bidx = attn_bidx;

                for (l, layer) in self.layers.iter().enumerate() {
                    let mut sync_part0 = false;
                    let mut sync_partn = false;
                    let mut sync_fin = false;
                    if !layer.is_conv
                        && layer.swa_window > 0
                        && let Err(e) = Self::ensure_swa_capacity_for_chunk(
                            &mut layer_kv_blocks[l][0],
                            &mut swa_base_chunk[l],
                            layer.swa_window,
                            pos0 / LFM3_KV_CHUNK_T,
                        )
                    {
                        return self.record_device_fault(
                            format!("prefill c={c} layer {l} swa compact: {e}"),
                            out_len,
                        );
                    }
                    {
                        let yb = pre_bo_y_layers[l].as_mut_slice();
                        if layer.is_conv {
                            write_lfm3_prefill_step_params(
                                yb, pos0, n_valid, 0, 0, true, 0, 0, eps, None,
                            );
                        } else {
                            fill_lfm3_prefill_rope_cs_into(
                                pos0,
                                n_valid,
                                layer.rope_theta,
                                layer.rope_freq_factors.as_deref(),
                                yb,
                            );
                            if layer.swa_window > 0 {
                                let local_past = needed_past.saturating_sub(swa_base_chunk[l]);
                                let (sb, swa_cap) = select_lfm3_kv_bucket(&attn_caps, local_past);
                                swa_bidx = sb;
                                write_lfm3_prefill_step_params(
                                    yb,
                                    pos0,
                                    n_valid,
                                    swa_cap,
                                    pos0,
                                    true,
                                    layer.swa_window,
                                    1,
                                    eps,
                                    None,
                                );
                                write_lfm3_prefill_ext_params(yb, swa_base_chunk[l], false, false);
                            } else if use_fast_pre {
                                write_lfm3_prefill_step_params(
                                    yb,
                                    pos0,
                                    n_valid,
                                    attn_kv_cap,
                                    pos0,
                                    true,
                                    0,
                                    1,
                                    eps,
                                    None,
                                );
                            } else {
                                write_lfm3_prefill_step_params(
                                    yb, pos0, n_valid, 0, pos0, true, 0, 2, eps, None,
                                );
                                if num_full > 0 {
                                    write_lfm3_params_at(
                                        yb,
                                        LFM3_PREFILL_SLOT_NORM_1 + 2048,
                                        pos0,
                                        n_valid,
                                        LFM3_MAX_KV_CHUNKS,
                                        pos0,
                                        true,
                                        0,
                                        4,
                                        eps,
                                        0,
                                        false,
                                        false,
                                    );
                                    sync_part0 = true;
                                    if num_full > 1 {
                                        write_lfm3_params_at(
                                            yb,
                                            LFM3_PREFILL_SLOT_NORM_2 + 2048,
                                            pos0,
                                            n_valid,
                                            LFM3_MAX_KV_CHUNKS,
                                            pos0,
                                            true,
                                            0,
                                            4,
                                            eps,
                                            0,
                                            true,
                                            false,
                                        );
                                        sync_partn = true;
                                    }
                                }
                                write_lfm3_params_at(
                                    yb,
                                    LFM3_PREFILL_SLOT_NORM_3 + 2048,
                                    pos0,
                                    n_valid,
                                    fin_cap,
                                    pos0,
                                    true,
                                    0,
                                    5,
                                    eps,
                                    num_full * LFM3_MAX_KV_CHUNKS,
                                    num_full > 0,
                                    false,
                                );
                                sync_fin = true;
                            }
                        }
                    }
                    if let Err(e) = pre_bo_y_layers[l].sync_range_to_device(
                        PREFILL_STEP_PARAMS_BYTES,
                        PREFILL_STEP_PARAMS_BYTE_OFF,
                    ) {
                        return self.record_device_fault(
                            format!("prefill c={c} layer {l} step_params sync: {e}"),
                            out_len,
                        );
                    }
                    if sync_part0
                        && let Err(e) = pre_bo_y_layers[l].sync_range_to_device(
                            PREFILL_STEP_PARAMS_BYTES,
                            PREFILL_NORM1_PARAMS_BYTE_OFF,
                        )
                    {
                        return self.record_device_fault(
                            format!("prefill c={c} layer {l} fa_part0 sync: {e}"),
                            out_len,
                        );
                    }
                    if sync_partn
                        && let Err(e) = pre_bo_y_layers[l].sync_range_to_device(
                            PREFILL_STEP_PARAMS_BYTES,
                            PREFILL_NORM2_PARAMS_BYTE_OFF,
                        )
                    {
                        return self.record_device_fault(
                            format!("prefill c={c} layer {l} fa_partn sync: {e}"),
                            out_len,
                        );
                    }
                    if sync_fin
                        && let Err(e) = pre_bo_y_layers[l].sync_range_to_device(
                            PREFILL_STEP_PARAMS_BYTES,
                            PREFILL_NORM3_PARAMS_BYTE_OFF,
                        )
                    {
                        return self.record_device_fault(
                            format!("prefill c={c} layer {l} fa_fin sync: {e}"),
                            out_len,
                        );
                    }
                    if !layer.is_conv
                        && let Err(e) = pre_bo_y_layers[l]
                            .sync_range_to_device(rope_sync_bytes, PREFILL_ROPE_BYTE_OFF)
                    {
                        return self.record_device_fault(
                            format!("prefill c={c} layer {l} rope sync: {e}"),
                            out_len,
                        );
                    }
                }

                let plan = if use_fast_pre {
                    let key = (attn_bidx, swa_bidx);
                    match pre_bucket_plans.get_or_insert_with(parity, key, || {
                        Self::build_prefill_bucket_plan_slices(
                            &self.layers,
                            kernels,
                            layer_bo_w,
                            layer_kv_blocks,
                            pre_bo_y_layers,
                            pre_bo_kvo_layers,
                            &*pre_bo_yn,
                            attn_bidx,
                            swa_bidx,
                            parity,
                        )
                    }) {
                        Ok(p) => p,
                        Err(e) => return self.record_device_fault(e, out_len),
                    }
                } else {
                    let key = (swa_bidx, num_full, has_rem, fin_bidx);
                    match pre_ext_plans.get_or_insert_with(parity, key, || {
                        Self::build_prefill_ext_plan(
                            &self.layers,
                            kernels,
                            layer_bo_w,
                            layer_kv_blocks,
                            pre_bo_y_layers,
                            pre_bo_kvo_layers,
                            &*pre_bo_yn,
                            swa_bidx,
                            num_full,
                            has_rem,
                            fin_bidx,
                            parity,
                        )
                    }) {
                        Ok(p) => p,
                        Err(e) => return self.record_device_fault(e, out_len),
                    }
                };

                if let Some(t) = tc0 {
                    t_prep_us += t.elapsed().as_secs_f64() * 1e6;
                }
                let ts0 = prof.then(crate::time::Instant::now);
                let res = plan
                    .runlist
                    .execute()
                    .and_then(|()| plan.runlist.wait(QUIESCE_TIMEOUT_MS));
                if let Some(t) = ts0 {
                    t_attn_npu_us += t.elapsed().as_secs_f64() * 1e6;
                }
                if let Err(e) = res {
                    return self.record_device_fault(
                        format!("prefill c={c} runlist execute/wait: {e}"),
                        out_len,
                    );
                }
                *pre_conv_kv_in_kvo = !parity;

                let is_last_chunk = c + 1 == n_chunks;
                let tkv_start = prof.then(crate::time::Instant::now);
                for (l, layer) in self.layers.iter().enumerate() {
                    if layer.is_conv {
                        let tck0 = prof.then(crate::time::Instant::now);
                        let conv_out_bo = if parity {
                            &mut layer_kv_blocks[l][0]
                        } else {
                            &mut pre_bo_kvo_layers[l]
                        };
                        if let Err(e) = conv_out_bo.sync_range_from_device(LFM3_KVO_BYTES, 0) {
                            return self.record_device_fault(
                                format!("prefill c={c} layer {l} conv kvo sync: {e}"),
                                out_len,
                            );
                        }
                        Self::save_conv_kvo_to_state(
                            conv_out_bo,
                            &mut state.layers[l],
                            pos0 + n_valid,
                            true,
                        );
                        if is_last_chunk && *pre_conv_kv_in_kvo {
                            let src = &pre_bo_kvo_layers[l].as_slice()[..LFM3_KVO_BYTES];
                            layer_kv_blocks[l][0].as_mut_slice()[..LFM3_KVO_BYTES]
                                .copy_from_slice(src);
                            if let Err(e) =
                                layer_kv_blocks[l][0].sync_range_to_device(LFM3_KVO_BYTES, 0)
                            {
                                return self.record_device_fault(
                                    format!("prefill layer {l} final conv bo_kv sync: {e}"),
                                    out_len,
                                );
                            }
                        }
                        if let Some(t) = tck0 {
                            t_conv_kv_us += t.elapsed().as_secs_f64() * 1e6;
                        }
                    } else {
                        let tak0 = prof.then(crate::time::Instant::now);
                        if let Err(e) =
                            pre_bo_kvo_layers[l].sync_range_from_device(attn_kvo_bytes, 0)
                        {
                            return self.record_device_fault(
                                format!("prefill c={c} layer {l} attn kvo sync: {e}"),
                                out_len,
                            );
                        }
                        if let Err(e) = Self::save_attn_prefill_kvo_to_kv_and_state(
                            &pre_bo_kvo_layers[l],
                            &mut layer_kv_blocks[l],
                            &kernels.prefill_conv,
                            &mut swa_base_chunk[l],
                            layer.swa_window,
                            Some(&mut state.layers[l]),
                            pos0,
                            n_valid,
                            use_f16,
                        ) {
                            return self.record_device_fault(
                                format!("prefill c={c} layer {l} attn kvo save: {e}"),
                                out_len,
                            );
                        }
                        if let Some(t) = tak0 {
                            t_attn_kv_us += t.elapsed().as_secs_f64() * 1e6;
                        }
                    }
                }
                let _ = tkv_start;
                if is_last_chunk {
                    *pre_conv_kv_in_kvo = false;
                    if need_logits {
                        let last_row_bytes = LFM3_H * 4;
                        let last_row_off = (n_valid - 1) * last_row_bytes;
                        if let Err(e) =
                            pre_bo_yn.sync_range_from_device(last_row_bytes, last_row_off)
                        {
                            return self.record_device_fault(
                                format!("prefill last_hidden sync_from_device: {e}"),
                                out_len,
                            );
                        }
                        read_le_f32_slice(
                            pre_bo_yn.as_slice(),
                            (n_valid - 1) * LFM3_H,
                            &mut last_hidden,
                        );
                    }
                }
            }

            state.seq_len = total_tokens;
            npu.synced_len = total_tokens;
            let t_head0 = prof.then(crate::time::Instant::now);
            let out_logits = if need_logits {
                self.cpu.xdna_project_logits_from_hidden(&last_hidden)
            } else {
                Vec::new()
            };
            if prof {
                let total_ms = t_prefill0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
                let head_ms = t_head0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
                let other_ms = (total_ms
                    - (t_prep_us + t_conv_npu_us + t_attn_npu_us + t_conv_kv_us + t_attn_kv_us)
                        * 1e-3
                    - head_ms)
                    .max(0.0);
                eprintln!(
                    "lfm3 prefill n={rem_tokens} chunks={n_chunks} total={total_ms:.1}ms: prep={:.1}ms conv_npu={:.1}ms attn_npu={:.1}ms conv_kv={:.1}ms attn_kv={:.1}ms head={head_ms:.1}ms other/paged={other_ms:.1}ms",
                    t_prep_us * 1e-3,
                    t_conv_npu_us * 1e-3,
                    t_attn_npu_us * 1e-3,
                    t_conv_kv_us * 1e-3,
                    t_attn_kv_us * 1e-3,
                );
            }
            return out_logits;
        }

        let mut hidden_all = embeddings[cur_start * LFM3_H..].to_vec();
        let bidir_caps: Vec<usize> = Q4PREFILL_LFM3_BIDIR_B_BUCKETS
            .iter()
            .map(|&(c, _)| c)
            .collect();
        let n_kv_chunks = total_tokens.div_ceil(LFM3_KV_CHUNK_T);

        for (l, layer) in self.layers.iter().enumerate() {
            if !is_causal && layer.is_non_causal {
                // Global bidirectional Attention: Phase A across all B=64 chunks, then Phase B across all B=64 chunks.
                let mut phase_a_saved: Vec<Vec<u8>> = Vec::with_capacity(n_chunks);
                for c in 0..n_chunks {
                    let pos0 = c * LFM3_PREFILL_B;
                    let n_valid = (total_tokens - pos0).min(LFM3_PREFILL_B);
                    let rope_cs = fill_prefill_rope_cs(
                        pos0,
                        n_valid,
                        layer.rope_theta,
                        layer.rope_freq_factors.as_deref(),
                    );
                    {
                        let yb = npu.pre_bo_y.as_mut_slice();
                        yb.fill(0);
                        write_le_f32_slice(
                            yb,
                            0,
                            &hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                        );
                        init_lfm3_prefill_bo_y_norms(
                            yb,
                            &layer.attn_norm,
                            &layer.post_attn_norm,
                            &layer.ffn_norm,
                            &layer.post_ffn_norm,
                            layer.q_norm.as_deref(),
                            layer.k_norm.as_deref(),
                        );
                        write_lfm3_prefill_step_params(
                            yb,
                            pos0,
                            n_valid,
                            0,
                            0,
                            false,
                            0,
                            2,
                            eps,
                            Some(&rope_cs),
                        );
                    }
                    let XdnaLfm3State {
                        ref mut kernels,
                        ref layer_bo_w,
                        ref mut layer_kv_blocks,
                        ref mut pre_bo_y,
                        ref mut pre_bo_yn,
                        ref mut pre_bo_kvo,
                        ..
                    } = *npu;
                    let last_kv_chunk = (c * 2) + usize::from(n_valid > LFM3_KV_CHUNK_T);
                    let max_blk = last_kv_chunk / LFM3_MAX_KV_CHUNKS;
                    if let Err(e) = Self::ensure_paged_block(
                        &mut layer_kv_blocks[l],
                        &kernels.prefill_conv,
                        max_blk,
                    ) {
                        return self.record_device_fault(e, out_len);
                    }
                    let kvo_sync = if n_valid <= LFM3_KV_CHUNK_T {
                        ATTN_PREFILL_SUB_KVO_BYTES
                    } else {
                        ATTN_PREFILL_KVO_BYTES
                    };
                    let res = kernels.prefill_bidir_a().and_then(|k| {
                        Self::run_prefill_single_chunk(
                            k,
                            &layer_bo_w[l],
                            pre_bo_y,
                            pre_bo_yn,
                            &layer_kv_blocks[l][0],
                            pre_bo_kvo,
                            kvo_sync,
                        )
                    });
                    if let Err(e) = res {
                        return self
                            .record_device_fault(format!("layer {l} bidir_a c={c}: {e}"), out_len);
                    }
                    if let Err(e) = npu
                        .pre_bo_y
                        .sync_range_from_device(PHASE_A_Q_BYTES, LFM3_PREFILL_SLOT_Q_BF16 * 4)
                        .and_then(|()| {
                            npu.pre_bo_y.sync_range_from_device(
                                PHASE_A_GATE_BYTES,
                                LFM3_PREFILL_SLOT_ATTN_GATE * 4,
                            )
                        })
                    {
                        return self.record_device_fault(
                            format!("layer {l} bidir_a bo_y readback: {e}"),
                            out_len,
                        );
                    }
                    let mut saved = vec![0u8; PHASE_A_Q_BYTES + PHASE_A_GATE_BYTES];
                    {
                        let yb = npu.pre_bo_y.as_slice();
                        saved[..PHASE_A_Q_BYTES].copy_from_slice(
                            &yb[LFM3_PREFILL_SLOT_Q_BF16 * 4..][..PHASE_A_Q_BYTES],
                        );
                        saved[PHASE_A_Q_BYTES..].copy_from_slice(
                            &yb[LFM3_PREFILL_SLOT_ATTN_GATE * 4..][..PHASE_A_GATE_BYTES],
                        );
                    }
                    phase_a_saved.push(saved);

                    let XdnaLfm3State {
                        ref pre_bo_kvo,
                        ref mut layer_kv_blocks,
                        ..
                    } = *npu;
                    let kvo_bytes = pre_bo_kvo.as_slice();
                    let n_subs = 1 + usize::from(n_valid > LFM3_KV_CHUNK_T);
                    for sub in 0..n_subs {
                        let global_kv_c = c * 2 + sub;
                        let blk = global_kv_c / LFM3_MAX_KV_CHUNKS;
                        let loc_c = global_kv_c % LFM3_MAX_KV_CHUNKS;
                        let kv_mut = layer_kv_blocks[l][blk].as_mut_slice();
                        let head_off = sub * LFM3_N_KV_HEADS;
                        for kvh in 0..LFM3_N_KV_HEADS {
                            let src_off = (head_off + kvh) * TILE_BYTES;
                            let dst_off = (kvh * LFM3_MAX_KV_CHUNKS + loc_c) * TILE_BYTES;
                            kv_mut[dst_off..dst_off + TILE_BYTES]
                                .copy_from_slice(&kvo_bytes[src_off..src_off + TILE_BYTES]);
                        }
                    }
                }
                for (blk, bo_kv) in npu.layer_kv_blocks[l].iter_mut().enumerate() {
                    let blk_chunks = n_kv_chunks
                        .saturating_sub(blk * LFM3_MAX_KV_CHUNKS)
                        .min(LFM3_MAX_KV_CHUNKS);
                    if blk_chunks == 0 {
                        continue;
                    }
                    for kvh in 0..LFM3_N_KV_HEADS {
                        if let Err(e) = bo_kv.sync_range_to_device(
                            blk_chunks * TILE_BYTES,
                            kvh * LFM3_MAX_KV_CHUNKS * TILE_BYTES,
                        ) {
                            return self.record_device_fault(
                                format!("layer {l} bidir blk {blk} bo_kv sync kvh={kvh}: {e}"),
                                out_len,
                            );
                        }
                    }
                }

                if n_kv_chunks <= LFM3_MAX_KV_CHUNKS {
                    let (bidx, bidir_cap) = select_lfm3_kv_bucket(&bidir_caps, n_kv_chunks);
                    for (c, saved) in phase_a_saved.iter().enumerate() {
                        let pos0 = c * LFM3_PREFILL_B;
                        let n_valid = (total_tokens - pos0).min(LFM3_PREFILL_B);
                        {
                            let yb = npu.pre_bo_y.as_mut_slice();
                            yb.fill(0);
                            write_le_f32_slice(
                                yb,
                                0,
                                &hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                            );
                            yb[LFM3_PREFILL_SLOT_Q_BF16 * 4..][..PHASE_A_Q_BYTES]
                                .copy_from_slice(&saved[..PHASE_A_Q_BYTES]);
                            yb[LFM3_PREFILL_SLOT_ATTN_GATE * 4..][..PHASE_A_GATE_BYTES]
                                .copy_from_slice(&saved[PHASE_A_Q_BYTES..]);
                            init_lfm3_prefill_bo_y_norms(
                                yb,
                                &layer.attn_norm,
                                &layer.post_attn_norm,
                                &layer.ffn_norm,
                                &layer.post_ffn_norm,
                                None,
                                None,
                            );
                            write_lfm3_prefill_step_params(
                                yb,
                                pos0,
                                n_valid,
                                bidir_cap,
                                total_tokens,
                                false,
                                0,
                                3,
                                eps,
                                None,
                            );
                        }
                        let XdnaLfm3State {
                            ref mut kernels,
                            ref layer_bo_w,
                            ref layer_kv_blocks,
                            ref mut pre_bo_y,
                            ref mut pre_bo_yn,
                            ref mut pre_bo_kvo,
                            ..
                        } = *npu;
                        let res = kernels.prefill_bidir_b(bidx).and_then(|k| {
                            Self::run_prefill_single_chunk(
                                k,
                                &layer_bo_w[l],
                                pre_bo_y,
                                pre_bo_yn,
                                &layer_kv_blocks[l][0],
                                pre_bo_kvo,
                                0,
                            )
                        });
                        if let Err(e) = res {
                            return self.record_device_fault(
                                format!("layer {l} bidir_b c={c}: {e}"),
                                out_len,
                            );
                        }
                        read_le_f32_slice(
                            npu.pre_bo_yn.as_slice(),
                            0,
                            &mut hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                        );
                    }
                } else {
                    let num_full = n_kv_chunks / LFM3_MAX_KV_CHUNKS;
                    let rem = n_kv_chunks % LFM3_MAX_KV_CHUNKS;
                    let (fin_bidx, fin_cap) = select_lfm3_kv_bucket(&attn_caps, rem);
                    let fin_blk = if rem > 0 { num_full } else { 0 };
                    for (c, saved) in phase_a_saved.iter().enumerate() {
                        let pos0 = c * LFM3_PREFILL_B;
                        let n_valid = (total_tokens - pos0).min(LFM3_PREFILL_B);
                        {
                            let yb = npu.pre_bo_y.as_mut_slice();
                            yb.fill(0);
                            write_le_f32_slice(
                                yb,
                                0,
                                &hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                            );
                            yb[LFM3_PREFILL_SLOT_Q_BF16 * 4..][..PHASE_A_Q_BYTES]
                                .copy_from_slice(&saved[..PHASE_A_Q_BYTES]);
                            yb[LFM3_PREFILL_SLOT_ATTN_GATE * 4..][..PHASE_A_GATE_BYTES]
                                .copy_from_slice(&saved[PHASE_A_Q_BYTES..]);
                            init_lfm3_prefill_bo_y_norms(
                                yb,
                                &layer.attn_norm,
                                &layer.post_attn_norm,
                                &layer.ffn_norm,
                                &layer.post_ffn_norm,
                                None,
                                None,
                            );
                        }
                        if let Err(e) = npu.pre_bo_y.sync_to_device() {
                            return self.record_device_fault(
                                format!("layer {l} bidir >8K pre_bo_y sync c={c}: {e}"),
                                out_len,
                            );
                        }
                        let XdnaLfm3State {
                            ref mut kernels,
                            ref layer_bo_w,
                            ref layer_kv_blocks,
                            ref mut pre_bo_y,
                            ref mut pre_bo_yn,
                            ref mut pre_bo_kvo,
                            ..
                        } = *npu;
                        if let Err(e) = Self::run_prefill_single_layer_paged_fa(
                            kernels,
                            &layer_bo_w[l],
                            pre_bo_y,
                            pre_bo_yn,
                            &layer_kv_blocks[l],
                            pre_bo_kvo,
                            l,
                            c,
                            pos0,
                            n_valid,
                            total_tokens,
                            false,
                            true,
                            num_full,
                            fin_bidx,
                            fin_cap,
                            fin_blk,
                            eps,
                            "bidir >8K",
                        ) {
                            return self.record_device_fault(e, out_len);
                        }
                        read_le_f32_slice(
                            npu.pre_bo_yn.as_slice(),
                            0,
                            &mut hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                        );
                    }
                }
            } else {
                // Sequential causal chunks (`ShortConv` or Causal/SWA Attention Mode 1).
                for c in 0..n_chunks {
                    let local_off = c * LFM3_PREFILL_B;
                    let n_valid = (rem_tokens - local_off).min(LFM3_PREFILL_B);
                    let pos0 = pos_aligned + local_off;
                    let chunk_slice =
                        &mut hidden_all[local_off * LFM3_H..(local_off + n_valid) * LFM3_H];
                    if layer.is_conv {
                        {
                            let yb = npu.pre_bo_y.as_mut_slice();
                            yb.fill(0);
                            write_le_f32_slice(yb, 0, chunk_slice);
                            init_lfm3_prefill_bo_y_norms(
                                yb,
                                &layer.attn_norm,
                                &layer.post_attn_norm,
                                &layer.ffn_norm,
                                &layer.post_ffn_norm,
                                None,
                                None,
                            );
                            write_lfm3_prefill_step_params(
                                yb, pos0, n_valid, 0, 0, true, 0, 0, eps, None,
                            );
                        }
                        let XdnaLfm3State {
                            ref kernels,
                            ref layer_bo_w,
                            ref mut layer_kv_blocks,
                            ref mut pre_bo_y,
                            ref mut pre_bo_yn,
                            ref mut pre_bo_kvo,
                            ..
                        } = *npu;
                        if let Err(e) = Self::run_prefill_single_chunk(
                            &kernels.prefill_conv,
                            &layer_bo_w[l],
                            pre_bo_y,
                            pre_bo_yn,
                            &layer_kv_blocks[l][0],
                            pre_bo_kvo,
                            LFM3_KVO_BYTES,
                        ) {
                            return self.record_device_fault(
                                format!("layer {l} conv c={c}: {e}"),
                                out_len,
                            );
                        }
                        read_le_f32_slice(npu.pre_bo_yn.as_slice(), 0, chunk_slice);
                        let XdnaLfm3State {
                            ref pre_bo_kvo,
                            ref mut layer_kv_blocks,
                            ..
                        } = *npu;
                        if let Err(e) = Self::save_conv_kvo_to_kv_and_state(
                            pre_bo_kvo,
                            &mut layer_kv_blocks[l][0],
                            is_causal.then_some(&mut state.layers[l]),
                            pos0 + n_valid,
                            is_causal,
                        ) {
                            return self.record_device_fault(
                                format!("layer {l} conv kvo save: {e}"),
                                out_len,
                            );
                        }
                    } else {
                        let needed_past = pos0 / LFM3_KV_CHUNK_T;
                        let rope_cs = fill_prefill_rope_cs(
                            pos0,
                            n_valid,
                            layer.rope_theta,
                            layer.rope_freq_factors.as_deref(),
                        );
                        let attn_kvo_bytes = if n_valid <= LFM3_KV_CHUNK_T {
                            ATTN_PREFILL_SUB_KVO_BYTES
                        } else {
                            ATTN_PREFILL_KVO_BYTES
                        };
                        if layer.swa_window > 0 || needed_past <= 255 {
                            if layer.swa_window > 0 {
                                let npu_s = &mut *npu;
                                if let Err(e) = Self::ensure_swa_capacity_for_chunk(
                                    &mut npu_s.layer_kv_blocks[l][0],
                                    &mut npu_s.swa_base_chunk[l],
                                    layer.swa_window,
                                    pos0 / LFM3_KV_CHUNK_T,
                                ) {
                                    return self.record_device_fault(
                                        format!("layer {l} swa compact c={c}: {e}"),
                                        out_len,
                                    );
                                }
                            }
                            let kv_base = if layer.swa_window > 0 {
                                npu.swa_base_chunk[l]
                            } else {
                                0
                            };
                            let (bidx, kv_cap) = select_lfm3_kv_bucket(
                                &attn_caps,
                                needed_past.saturating_sub(kv_base),
                            );
                            {
                                let yb = npu.pre_bo_y.as_mut_slice();
                                yb.fill(0);
                                write_le_f32_slice(yb, 0, chunk_slice);
                                init_lfm3_prefill_bo_y_norms(
                                    yb,
                                    &layer.attn_norm,
                                    &layer.post_attn_norm,
                                    &layer.ffn_norm,
                                    &layer.post_ffn_norm,
                                    layer.q_norm.as_deref(),
                                    layer.k_norm.as_deref(),
                                );
                                write_lfm3_prefill_step_params(
                                    yb,
                                    pos0,
                                    n_valid,
                                    kv_cap,
                                    pos0,
                                    true,
                                    layer.swa_window,
                                    1,
                                    eps,
                                    Some(&rope_cs),
                                );
                                write_lfm3_prefill_ext_params(yb, kv_base, false, false);
                            }
                            let XdnaLfm3State {
                                ref mut kernels,
                                ref layer_bo_w,
                                ref layer_kv_blocks,
                                ref mut pre_bo_y,
                                ref mut pre_bo_yn,
                                ref mut pre_bo_kvo,
                                ..
                            } = *npu;
                            let res = kernels.prefill_attn(bidx).and_then(|k| {
                                Self::run_prefill_single_chunk(
                                    k,
                                    &layer_bo_w[l],
                                    pre_bo_y,
                                    pre_bo_yn,
                                    &layer_kv_blocks[l][0],
                                    pre_bo_kvo,
                                    attn_kvo_bytes,
                                )
                            });
                            if let Err(e) = res {
                                return self.record_device_fault(
                                    format!("layer {l} attn c={c}: {e}"),
                                    out_len,
                                );
                            }
                        } else {
                            // Non-causal model fallback / >8K full-attn sequential chunk: Mode 2 -> b * Mode 4 -> Mode 5
                            {
                                let yb = npu.pre_bo_y.as_mut_slice();
                                yb.fill(0);
                                write_le_f32_slice(yb, 0, chunk_slice);
                                init_lfm3_prefill_bo_y_norms(
                                    yb,
                                    &layer.attn_norm,
                                    &layer.post_attn_norm,
                                    &layer.ffn_norm,
                                    &layer.post_ffn_norm,
                                    layer.q_norm.as_deref(),
                                    layer.k_norm.as_deref(),
                                );
                                write_lfm3_prefill_step_params(
                                    yb,
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
                            }
                            let res_a = {
                                let XdnaLfm3State {
                                    ref mut kernels,
                                    ref layer_bo_w,
                                    ref layer_kv_blocks,
                                    ref mut pre_bo_y,
                                    ref mut pre_bo_yn,
                                    ref mut pre_bo_kvo,
                                    ..
                                } = *npu;
                                kernels.prefill_bidir_a().and_then(|k| {
                                    Self::run_prefill_single_chunk(
                                        k,
                                        &layer_bo_w[l],
                                        pre_bo_y,
                                        pre_bo_yn,
                                        &layer_kv_blocks[l][0],
                                        pre_bo_kvo,
                                        attn_kvo_bytes,
                                    )
                                })
                            };
                            if let Err(e) = res_a {
                                return self.record_device_fault(
                                    format!("layer {l} phase_a c={c}: {e}"),
                                    out_len,
                                );
                            }
                            let num_full = needed_past / LFM3_MAX_KV_CHUNKS;
                            let rem = needed_past % LFM3_MAX_KV_CHUNKS;
                            let XdnaLfm3State {
                                ref mut kernels,
                                ref layer_bo_w,
                                ref layer_kv_blocks,
                                ref mut pre_bo_y,
                                ref mut pre_bo_yn,
                                ref mut pre_bo_kvo,
                                ..
                            } = *npu;
                            let (fin_bidx, fin_cap) = select_lfm3_kv_bucket(&attn_caps, rem);
                            let fin_blk = if rem > 0 { num_full } else { 0 };
                            if let Err(e) = Self::run_prefill_single_layer_paged_fa(
                                kernels,
                                &layer_bo_w[l],
                                pre_bo_y,
                                pre_bo_yn,
                                &layer_kv_blocks[l],
                                pre_bo_kvo,
                                l,
                                c,
                                pos0,
                                n_valid,
                                pos0,
                                true,
                                false,
                                num_full,
                                fin_bidx,
                                fin_cap,
                                fin_blk,
                                eps,
                                "paged",
                            ) {
                                return self.record_device_fault(e, out_len);
                            }
                        }
                        read_le_f32_slice(npu.pre_bo_yn.as_slice(), 0, chunk_slice);
                        let XdnaLfm3State {
                            ref kernels,
                            ref pre_bo_kvo,
                            ref mut layer_kv_blocks,
                            ref mut swa_base_chunk,
                            ..
                        } = *npu;
                        if let Err(e) = Self::save_attn_prefill_kvo_to_kv_and_state(
                            pre_bo_kvo,
                            &mut layer_kv_blocks[l],
                            &kernels.prefill_conv,
                            &mut swa_base_chunk[l],
                            layer.swa_window,
                            is_causal.then_some(&mut state.layers[l]),
                            pos0,
                            n_valid,
                            use_f16,
                        ) {
                            return self.record_device_fault(
                                format!("layer {l} attn kvo save: {e}"),
                                out_len,
                            );
                        }
                    }
                }
            }
        }

        state.seq_len = total_tokens;
        npu.synced_len = total_tokens;
        if !need_logits {
            return Vec::new();
        }
        let last_hidden = &hidden_all[(rem_tokens - 1) * LFM3_H..rem_tokens * LFM3_H];
        self.cpu.xdna_project_logits_from_hidden(last_hidden)
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
        if embedding.len() != LFM3_H {
            return self.reject_input(
                format!("embedding length {} != {LFM3_H}", embedding.len()),
                vocab_size,
            );
        }
        if pos >= cfg.max_seq_len || pos >= LFM3_MAX_TOKENS {
            return self.reject_input(
                format!(
                    "pos {pos} exceeds context length {}",
                    cfg.max_seq_len.min(LFM3_MAX_TOKENS)
                ),
                vocab_size,
            );
        }
        if let Err(e) = self.validate_state(state) {
            return self.reject_input(e, vocab_size);
        }

        let mut npu = self.lock_npu();
        if let Err(e) = self.sync_cpu_state_to_npu(&mut npu, state, pos) {
            return self.reject_input(e, vocab_size);
        }
        match self.run_npu_decode_embedding_step(&mut npu, embedding, pos, state) {
            Ok(logits) => logits,
            Err(e) => self.record_device_fault(e, vocab_size),
        }
    }
}

impl Model for XdnaLfm3Model {
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.session_gate.try_acquire().map(Some)
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
        let _npu_guard = self.lock_npu();
        let mut isolated = match InferenceState::from_config(self.cpu.config()) {
            Ok(s) => s,
            Err(e) => {
                return self.reject_input_err(
                    CeraError::Backend(format!(
                        "failed to allocate isolated CPU state for hidden_states: {e}"
                    )),
                    0,
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
                format!("XDNA LFM3 decode requires 1 token, got {}", tokens.len()),
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
            || pos >= LFM3_MAX_TOKENS
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
        if state.scratch.logits.len() < vocab_size {
            state.scratch.logits.resize(vocab_size, 0.0);
        }
        let mut logits_buf = std::mem::take(&mut state.scratch.logits);
        let res = {
            let mut npu = self.lock_npu();
            self.sync_cpu_state_to_npu(&mut npu, state, pos)
                .and_then(|()| {
                    self.run_npu_decode_embedding_step_into(
                        &mut npu,
                        &emb,
                        pos,
                        state,
                        &mut logits_buf[..vocab_size],
                    )
                })
        };
        let tok = match res {
            Ok(()) => crate::sampler::argmax(&logits_buf[..vocab_size]),
            Err(e) => {
                let _ = self.record_device_fault(e, vocab_size);
                0
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
        // Bidirectional models must process the entire sequence in one pass.
        let eff_ubatch = if !self.cpu.config().is_causal {
            0
        } else {
            ubatch
        };
        crate::model::run_chunked_prefill(tokens, start_pos, eff_ubatch, cancel, |chunk, pos| {
            let logits = self.forward_prefill(chunk, pos, state);
            if self.decode_error.lock().unwrap().is_some() {
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
        npu.pending_kv.flush_into(state);
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
        npu.pending_kv.flush_into(state);
        self.cpu.truncate_kv(state, len);
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
        npu.pending_kv.flush_into(state);
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
        npu.pending_kv.discard();
        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv {
                npu.layer_kv_blocks[l][0].as_mut_slice()[..LFM3_KVO_BYTES].fill(0);
                npu.layer_kv_blocks[l][0]
                    .sync_range_to_device(LFM3_KVO_BYTES, 0)
                    .map_err(|e| {
                        self.state_torn.store(true, Ordering::SeqCst);
                        CeraError::Backend(format!("layer {l} reset conv bo_kv sync failed: {e}"))
                    })?;
            }
            npu.swa_base_chunk[l] = 0;
        }
        npu.synced_len = 0;
        npu.force_conv_sync = false;
        if let Some(dec) = npu.decode.as_mut() {
            dec.conv_kv_in_kvo = false;
        }
        let _ = take_fault(&self.decode_error);
        self.state_torn.store(false, Ordering::SeqCst);
        *state = fresh;
        Ok(())
    }
}

impl std::fmt::Debug for XdnaLfm3Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XdnaLfm3Model")
            .field("config", self.cpu.config())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::xdna::sys::fake::{Config as FakeConfig, with_shim};
    use crate::gguf::GgufBuilder;

    fn make_q4_0_bytes(rows: usize, cols: usize) -> Vec<u8> {
        vec![0u8; rows * (cols / 32) * 18]
    }

    fn bool_array_raw(flags: &[bool]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 8 + flags.len());
        out.extend_from_slice(&7u32.to_le_bytes()); // GGUF_TYPE_BOOL
        out.extend_from_slice(&(flags.len() as u64).to_le_bytes());
        for &f in flags {
            out.push(u8::from(f));
        }
        out
    }

    fn tiny_lfm3_gguf(is_causal: bool) -> GgufFile {
        let vocab = 32usize;
        let ones_h = vec![1.0f32; LFM3_H];
        let ones_hd = vec![1.0f32; LFM3_HEAD_DIM];
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "lfm3")
            .kv_u32("lfm3.block_count", 2)
            .kv_u32("lfm3.embedding_length", LFM3_H as u32)
            .kv_u32("lfm3.feed_forward_length", LFM3_FFN_SRC as u32)
            .kv_u32("lfm3.attention.head_count", LFM3_N_HEADS as u32)
            .kv_i32_array(
                "lfm3.attention.head_count_kv",
                vec![0, LFM3_N_KV_HEADS as i32],
            )
            .kv_u32("lfm3.attention.key_length", LFM3_HEAD_DIM as u32)
            .kv_u32("lfm3.attention.value_length", LFM3_HEAD_DIM as u32)
            .kv_u32("lfm3.context_length", 128)
            .kv_f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("lfm3.shortconv.l_cache", LFM3_CONV_TAP as u32)
            .kv_bool("lfm3.attention.causal", is_causal);
        if !is_causal {
            b = b.kv_str("lfm3.decision.type", "lfm3-d1");
        }
        let b = b
            .kv_f32("lfm3.rope.freq_base", 1_000_000.0)
            .kv_raw(
                "lfm3.attention.sliding_window_pattern",
                9,
                bool_array_raw(&[false, false]),
            )
            .tensor_f32(
                "token_embd.weight",
                &[LFM3_H, vocab],
                &vec![0.01f32; vocab * LFM3_H],
            )
            .tensor_f32("output_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32(
                "rope_freqs.weight",
                &[LFM3_HEAD_DIM / 2],
                &vec![1.0f32; LFM3_HEAD_DIM / 2],
            )
            // Layer 0: ShortConv
            .tensor_f32("blk.0.attn_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32("blk.0.post_attention_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32("blk.0.ffn_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32("blk.0.post_ffw_norm.weight", &[LFM3_H], &ones_h)
            .tensor(
                "blk.0.shortconv.in_proj.weight",
                &[LFM3_H, 3 * LFM3_H],
                2,
                make_q4_0_bytes(3 * LFM3_H, LFM3_H),
            )
            .tensor_f32(
                "blk.0.shortconv.conv.weight",
                &[LFM3_CONV_TAP, LFM3_H],
                &vec![0.1f32; LFM3_CONV_TAP * LFM3_H],
            )
            .tensor(
                "blk.0.shortconv.out_proj.weight",
                &[LFM3_H, LFM3_H],
                2,
                make_q4_0_bytes(LFM3_H, LFM3_H),
            )
            .tensor(
                "blk.0.ffn_gate.weight",
                &[LFM3_H, LFM3_FFN_SRC],
                2,
                make_q4_0_bytes(LFM3_FFN_SRC, LFM3_H),
            )
            .tensor(
                "blk.0.ffn_up.weight",
                &[LFM3_H, LFM3_FFN_SRC],
                2,
                make_q4_0_bytes(LFM3_FFN_SRC, LFM3_H),
            )
            .tensor(
                "blk.0.ffn_down.weight",
                &[LFM3_FFN_SRC, LFM3_H],
                2,
                make_q4_0_bytes(LFM3_H, LFM3_FFN_SRC),
            )
            // Layer 1: Global Attention
            .tensor_f32("blk.1.attn_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32("blk.1.post_attention_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32("blk.1.ffn_norm.weight", &[LFM3_H], &ones_h)
            .tensor_f32("blk.1.post_ffw_norm.weight", &[LFM3_H], &ones_h)
            .tensor(
                "blk.1.attn_q.weight",
                &[LFM3_H, LFM3_Q_DIM],
                2,
                make_q4_0_bytes(LFM3_Q_DIM, LFM3_H),
            )
            .tensor(
                "blk.1.attn_k.weight",
                &[LFM3_H, LFM3_KV_DIM],
                2,
                make_q4_0_bytes(LFM3_KV_DIM, LFM3_H),
            )
            .tensor(
                "blk.1.attn_v.weight",
                &[LFM3_H, LFM3_KV_DIM],
                2,
                make_q4_0_bytes(LFM3_KV_DIM, LFM3_H),
            )
            .tensor(
                "blk.1.attn_gate.weight",
                &[LFM3_H, LFM3_N_HEADS],
                2,
                make_q4_0_bytes(LFM3_N_HEADS, LFM3_H),
            )
            .tensor(
                "blk.1.attn_output.weight",
                &[LFM3_Q_DIM, LFM3_H],
                2,
                make_q4_0_bytes(LFM3_H, LFM3_Q_DIM),
            )
            .tensor_f32("blk.1.attn_q_norm.weight", &[LFM3_HEAD_DIM], &ones_hd)
            .tensor_f32("blk.1.attn_k_norm.weight", &[LFM3_HEAD_DIM], &ones_hd)
            .tensor(
                "blk.1.ffn_gate.weight",
                &[LFM3_H, LFM3_FFN_SRC],
                2,
                make_q4_0_bytes(LFM3_FFN_SRC, LFM3_H),
            )
            .tensor(
                "blk.1.ffn_up.weight",
                &[LFM3_H, LFM3_FFN_SRC],
                2,
                make_q4_0_bytes(LFM3_FFN_SRC, LFM3_H),
            )
            .tensor(
                "blk.1.ffn_down.weight",
                &[LFM3_FFN_SRC, LFM3_H],
                2,
                make_q4_0_bytes(LFM3_H, LFM3_FFN_SRC),
            );
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    #[test]
    fn compute_lfm3_rope_cs_respects_freq_factors() {
        let mut factors = vec![1.0f32; LFM3_HEAD_DIM / 2];
        factors[3] = 1e30;
        let cs = compute_lfm3_rope_cs(7, 1_000_000.0, Some(&factors));
        assert!((cs[3] - 1.0).abs() < 1e-6);
        assert!(cs[LFM3_HEAD_DIM / 2 + 3].abs() < 1e-6);
        let plain = compute_lfm3_rope_cs(7, 1_000_000.0, None);
        assert!((cs[0] - plain[0]).abs() < 1e-6);
        assert!((cs[LFM3_HEAD_DIM / 2] - plain[LFM3_HEAD_DIM / 2]).abs() < 1e-6);
    }

    #[test]
    fn causal_and_bidirectional_lfm3_load_and_prefill_on_fake_shim() {
        for is_causal in [true, false] {
            with_shim(FakeConfig::default(), || {
                let gguf = tiny_lfm3_gguf(is_causal);
                let model = crate::model::load_model_xdna(gguf, None, 128)
                    .expect("load_model_xdna should load LFM3 on fake shim");
                assert_eq!(model.backend_label(), "xdna");
                assert!(model.supports_embedding_input());

                let mut state = InferenceState::from_config(model.config()).unwrap();
                let logits = model.forward_prefill(&[1, 2, 3], 0, &mut state);
                assert_eq!(logits.len(), model.config().vocab_size);
                assert_eq!(state.seq_len, 3);
                assert!(model.take_decode_error().is_none());

                if is_causal {
                    let next = model.forward(&[4], 3, &mut state);
                    assert_eq!(next.len(), model.config().vocab_size);
                    assert_eq!(state.seq_len, 4);
                    assert!(model.take_decode_error().is_none());
                } else {
                    assert!(matches!(
                        model.check_kv_rewind(&state, 1),
                        Err(KvRewindError::NonCausal)
                    ));
                }
            });
        }
    }

    #[test]
    fn causal_decode_conv_ping_pong_rewind_and_continuation_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_gguf(true);
            let model = crate::model::load_model_xdna(gguf, None, 128)
                .expect("load_model_xdna should load causal LFM3 on fake shim");
            let mut state = InferenceState::from_config(model.config()).unwrap();

            let _ = model.forward_prefill(&[1, 2, 3], 0, &mut state);
            assert_eq!(state.seq_len, 3);

            // Run 5 decode steps across even and odd conv KV ping-pong parities.
            for (step, tok) in [4u32, 5, 6, 7, 8].into_iter().enumerate() {
                let logits = model.forward(&[tok], 3 + step, &mut state);
                assert_eq!(logits.len(), model.config().vocab_size);
                assert_eq!(state.seq_len, 4 + step);
                assert!(model.take_decode_error().is_none());
            }

            // Rewind mid-sequence after an odd decode parity step and continue with prefill + decode.
            model
                .try_truncate_kv(&mut state, 5)
                .expect("try_truncate_kv should rewind causal NPU state");
            assert_eq!(state.seq_len, 5);
            assert!(model.take_decode_error().is_none());

            let logits = model.forward_prefill(&[9, 10], 5, &mut state);
            assert_eq!(logits.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 7);
            assert!(model.take_decode_error().is_none());

            let next = model.forward(&[11], 7, &mut state);
            assert_eq!(next.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 8);
            assert!(model.take_decode_error().is_none());
        });
    }

    fn oracle_model_path(file: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target")
            .join("oracle")
            .join("models")
            .join(file)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let xf = x as f64;
            let yf = y as f64;
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom) as f32
        }
    }

    fn argmax_idx(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM3-1.7B-QAD-Q4_0.gguf"]
    fn hw_gate3a_lfm3_causal_parity_and_throughput() {
        let path = oracle_model_path("LFM3-1.7B-QAD-Q4_0.gguf");
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open LFM3-1.7B-QAD-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu =
            crate::model::load_model_xdna(gguf, Some(&path), 2048).expect("load LFM3 on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), 2048)
            .expect("load LFM3 on CPU");
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

        // Warmup 1 aligned 32-token prefill + reset so lazy buckets and XRT contexts are hot.
        let _ = npu.forward_prefill(&tokens[..32], 0, &mut npu_state);
        npu.try_reset_kv(&mut npu_state, &crate::kv_cache::KvCompression::None, 2048)
            .unwrap();

        // Test 1: 45-token prefill (32 aligned + 13 partial chunk)
        let t_prefill = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        assert!(npu.take_decode_error().is_none());

        let prefill_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let npu_top = argmax_idx(&npu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "Gate3a prefill (n=45): cos={prefill_cos:.6} top1_npu={npu_top} top1_cpu={cpu_top} time={:.2}ms ({:.1} tok/s)",
            prefill_s * 1e3,
            tokens.len() as f64 / prefill_s
        );
        assert!(
            prefill_cos >= 0.985,
            "prefill cosine too low: {prefill_cos}"
        );

        // Test 2: 16 B=1 decode steps (crosses bucket/chunk boundaries too), teacher-forced from CPU
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
            assert!(npu.take_decode_error().is_none());

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
            "Gate3a decode (16 steps): mean_cos={mean_cos:.6} min_cos={min_cos:.6} top1={top1_matches}/16 ms/tok={:.2} ({:.1} tok/s)",
            (decode_secs * 1e3) / 16.0,
            16.0 / decode_secs
        );
        assert!(mean_cos >= 0.99, "mean decode cosine too low: {mean_cos}");
        assert!(min_cos >= 0.97, "min decode cosine too low: {min_cos}");

        // Test 3: Multi-turn unaligned continuation from current seq_len (=61 -> +19 = 80)
        let cont = &tokens[..19];
        let start_pos = npu_state.seq_len;
        let npu_cont = npu.forward_prefill(cont, start_pos, &mut npu_state);
        let cpu_cont = cpu.forward_prefill(cont, start_pos, &mut cpu_state);
        assert!(npu.take_decode_error().is_none());
        let cont_cos = cosine_similarity(&npu_cont, &cpu_cont);
        eprintln!(
            "Gate3a continuation (start={start_pos}, n=19): cos={cont_cos:.6} top1_npu={} top1_cpu={}",
            argmax_idx(&npu_cont),
            argmax_idx(&cpu_cont)
        );
        assert!(cont_cos >= 0.985, "continuation cosine too low: {cont_cos}");
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and d1-2B-Q4_0.gguf"]
    fn hw_gate3b_d1_2b_bidirectional_parity_and_decide() {
        let path = oracle_model_path("d1-2B-Q4_0.gguf");
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open d1-2B-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu =
            crate::model::load_model_xdna(gguf, Some(&path), 2048).expect("load d1-2B on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), 2048)
            .expect("load d1-2B on CPU");
        assert_eq!(npu.backend_label(), "xdna");

        let prompt = "Determine whether the package arrived on time and whether all safety \
                      and calibration thresholds passed without anomaly across multiple context chunks \
                      requiring full bidirectional global attention.";
        let mut tokens = tok.encode(prompt);
        while tokens.len() < 48 {
            tokens.push(tokens[tokens.len() % tokens.len().max(1)]);
        }
        tokens.truncate(48);

        let mut npu_state = InferenceState::from_config(npu.config()).unwrap();
        let mut cpu_state = InferenceState::from_config(cpu.config()).unwrap();

        // Warmup + parity check for 48 tokens (2 chunks: 32 + 16, exercising Mode 2 + Mode 3)
        let _ = npu.forward_prefill(&tokens, 0, &mut npu_state);
        npu.try_reset_kv(&mut npu_state, &crate::kv_cache::KvCompression::None, 2048)
            .unwrap();
        let t0 = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let elapsed_s = t0.elapsed().as_secs_f64();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        assert!(npu.take_decode_error().is_none());

        let cos = cosine_similarity(&npu_logits, &cpu_logits);
        let npu_top = argmax_idx(&npu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "Gate3b d1-2B trunk prefill (n=48): cos={cos:.6} top1_npu={npu_top} top1_cpu={cpu_top} time={:.2}ms ({:.1} tok/s)",
            elapsed_s * 1e3,
            tokens.len() as f64 / elapsed_s
        );
        assert!(cos >= 0.99, "d1-2B trunk cosine too low: {cos}");
        drop(npu);
        drop(cpu);

        // End-to-end D1Engine::open + decide_json comparison (NPU vs CPU)
        let npu_d1 = crate::d1::D1Engine::open(&path, None, crate::engine::BackendPreference::Npu)
            .expect("D1Engine::open on NPU");
        let cpu_d1 = crate::d1::D1Engine::open(&path, None, crate::engine::BackendPreference::Cpu)
            .expect("D1Engine::open on CPU");
        eprintln!("Gate3b D1Engine report (NPU): {}", npu_d1.report());
        assert_eq!(npu_d1.report().trunk, "xdna");

        let req = r#"{
            "state": "The parcel arrived on Monday in good condition with seal intact. Temperature stayed at 4C throughout transit.",
            "questions": {
                "arrived": {"type": "noul", "instructions": "Did the parcel arrive on Monday?"},
                "quality": {
                    "type": "choice",
                    "instructions": "What condition was the parcel in?",
                    "criteria": {
                        "damaged": "The parcel was damaged or opened",
                        "intact": "The parcel arrived in good condition with seal intact",
                        "missing": "The parcel did not arrive"
                    }
                },
                "risk": {
                    "type": "score",
                    "instructions": "Rate transit risk.",
                    "criteria": ["Safe", "Minor issue", "Severe issue"]
                }
            }
        }"#;
        let npu_resp = npu_d1.decide_json(req).expect("NPU decide_json");
        let cpu_resp = cpu_d1.decide_json(req).expect("CPU decide_json");
        eprintln!("Gate3b D1Engine NPU response: {npu_resp}");
        eprintln!("Gate3b D1Engine CPU response: {cpu_resp}");

        let npu_json: serde_json::Value = serde_json::from_str(&npu_resp).unwrap();
        let cpu_json: serde_json::Value = serde_json::from_str(&cpu_resp).unwrap();
        let npu_noul = npu_json["answers"]["arrived"]["noul"]
            .as_f64()
            .expect("noul answer float");
        let cpu_noul = cpu_json["answers"]["arrived"]["noul"]
            .as_f64()
            .expect("cpu noul answer float");
        assert!(
            (npu_noul - cpu_noul).abs() <= 0.05,
            "noul probability mismatch: npu={npu_noul}, cpu={cpu_noul}"
        );
        assert_eq!(
            npu_json["answers"]["quality"]["choice"], cpu_json["answers"]["quality"]["choice"],
            "choice answer mismatch"
        );
        let npu_score = npu_json["answers"]["risk"]["score"]
            .as_f64()
            .expect("score answer float");
        let cpu_score = cpu_json["answers"]["risk"]["score"]
            .as_f64()
            .expect("cpu score answer float");
        assert!(
            (npu_score - cpu_score).abs() <= 0.1,
            "score mismatch: npu={npu_score}, cpu={cpu_score}"
        );
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware and LFM3-1.7B-QAD-Q4_0.gguf"]
    fn hw_gate3c_lfm3_long_context_parity() {
        let path = oracle_model_path("LFM3-1.7B-QAD-Q4_0.gguf");
        assert!(path.exists(), "missing GGUF at {}", path.display());

        let gguf = GgufFile::open(&path).expect("open LFM3-1.7B-QAD-Q4_0.gguf");
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let npu = crate::model::load_model_xdna(gguf, Some(&path), 16_384)
            .expect("load LFM3 (16K) on XDNA");
        let cpu = LfmModel::from_gguf(GgufFile::open(&path).expect("open CPU gguf"), 16_384)
            .expect("load LFM3 (16K) on CPU");
        assert_eq!(npu.backend_label(), "xdna");

        let seed_text = "The AMD Ryzen AI NPU executes paged online-softmax attention across \
                         arbitrary context lengths beyond 8192 tokens without host truncation. ";
        let seed = tok.encode(seed_text);
        let mut tokens = Vec::with_capacity(8224);
        while tokens.len() < 8224 {
            let rem = 8224 - tokens.len();
            tokens.extend_from_slice(&seed[..seed.len().min(rem)]);
        }

        let mut npu_state = InferenceState::from_config(npu.config()).unwrap();
        let mut cpu_state = InferenceState::from_config(cpu.config()).unwrap();

        let t0 = std::time::Instant::now();
        let npu_logits = npu.forward_prefill(&tokens, 0, &mut npu_state);
        let pre_s = t0.elapsed().as_secs_f64();
        let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
        assert!(npu.take_decode_error().is_none());

        let pre_cos = cosine_similarity(&npu_logits, &cpu_logits);
        let npu_top = argmax_idx(&npu_logits);
        let cpu_top = argmax_idx(&cpu_logits);
        eprintln!(
            "Gate3c >8K prefill (n=8224): cos={pre_cos:.6} top1_npu={npu_top} top1_cpu={cpu_top} time={pre_s:.2}s ({:.1} tok/s)",
            tokens.len() as f64 / pre_s
        );
        assert!(pre_cos >= 0.98, ">8K prefill cosine too low: {pre_cos}");

        let mut next_tok = cpu_top as u32;
        let mut dec_coss = Vec::new();
        for step in 0..4usize {
            let pos = tokens.len() + step;
            let nl = npu.forward(&[next_tok], pos, &mut npu_state);
            let cl = cpu.forward(&[next_tok], pos, &mut cpu_state);
            assert!(npu.take_decode_error().is_none());
            let c = cosine_similarity(&nl, &cl);
            let nt = argmax_idx(&nl);
            let ct = argmax_idx(&cl);
            dec_coss.push(c);
            eprintln!("  >8K decode step {step} pos={pos}: cos={c:.6} npu_top={nt} cpu_top={ct}");
            next_tok = ct as u32;
        }
        let mean_cos = dec_coss.iter().sum::<f32>() / dec_coss.len() as f32;
        let min_cos = dec_coss.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            mean_cos >= 0.985,
            ">8K mean decode cosine too low: {mean_cos}"
        );
        assert!(min_cos >= 0.97, ">8K min decode cosine too low: {min_cos}");
    }

    #[test]
    fn parity_plan_cache_evicts_before_building_replacement() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct DropProbe {
            id: usize,
            live: Rc<Cell<usize>>,
        }
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
            }
        }

        let live = Rc::new(Cell::new(0usize));
        let mut cache: Lfm3ParityPlanCache<usize, DropProbe> = Lfm3ParityPlanCache::new(2);
        let mk = |id: usize, max_live_before: usize| {
            let live = Rc::clone(&live);
            move || -> Result<DropProbe, String> {
                assert!(
                    live.get() <= max_live_before,
                    "eviction must drop old entry before building next (live={}, max={max_live_before})",
                    live.get()
                );
                live.set(live.get() + 1);
                Ok(DropProbe { id, live })
            }
        };

        assert_eq!(cache.get_or_insert_with(false, 10, mk(1, 0)).unwrap().id, 1);
        assert_eq!(cache.get_or_insert_with(false, 20, mk(2, 1)).unwrap().id, 2);
        assert_eq!(live.get(), 2);
        // Hit key 10 to refresh LRU so key 20 becomes oldest.
        assert_eq!(
            cache.get_or_insert_with(false, 10, mk(99, 0)).unwrap().id,
            1
        );
        // Inserting key 30 must drop key 20 first so live drops from 2 -> 1 before build runs.
        assert_eq!(cache.get_or_insert_with(false, 30, mk(3, 1)).unwrap().id, 3);
        assert_eq!(live.get(), 2);
        assert_eq!(cache.entries[0].len(), 2);
        assert_eq!(cache.entries[0][0].0, 10);
        assert_eq!(cache.entries[0][1].0, 30);
        // Parity `true` is independent.
        assert_eq!(cache.get_or_insert_with(true, 10, mk(4, 2)).unwrap().id, 4);
        assert_eq!(live.get(), 3);
    }
}

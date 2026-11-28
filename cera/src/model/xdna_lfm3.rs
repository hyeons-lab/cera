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
    LFM3_BLOCK_TOKENS, LFM3_CONV_TAP, LFM3_DEC_ACT_BYTES, LFM3_DEC_SLOT_FIN_NORM,
    LFM3_DEC_SLOT_NORM_0, LFM3_DEC_SLOT_NORM_1, LFM3_DEC_YN_BYTES, LFM3_FFN_PAD, LFM3_FFN_SRC,
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
    init_lfm3_prefill_bo_y_norms, inst_words, read_le_f32_slice, try_select_lfm3_kv_bucket,
    write_le_f32_slice, write_lfm3_ext_params, write_lfm3_params_at, write_lfm3_prefill_ext_params,
    write_lfm3_prefill_step_params, write_lfm3_step_params,
};
use crate::backend::xdna::repack::{
    f32_to_bf16, pack_f32_mse_stage_k1024, pack_q4_0_stage_k1024, pack_q4_0_stage_padded_k1024,
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
const MAX_DEC_KV_CHUNKS: usize = 511;
const PREFILL_STEP_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_0 + 2048) * 4;
const PREFILL_NORM1_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_1 + 2048) * 4;
const PREFILL_NORM2_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_2 + 2048) * 4;
const PREFILL_NORM3_PARAMS_BYTE_OFF: usize = (LFM3_PREFILL_SLOT_NORM_3 + 2048) * 4;
const STEP_PARAMS_BYTES: usize = 12 * 4;
const PREFILL_STEP_PARAMS_BYTES: usize = STEP_PARAMS_BYTES;
const DEC_NORM1_PARAMS_BYTE_OFF: usize = (LFM3_DEC_SLOT_NORM_1 + 4096) * 4;
const DEC_FIN_PARAMS_BYTE_OFF: usize = (LFM3_DEC_SLOT_FIN_NORM + 4096) * 4;
const PREFILL_ROPE_BYTE_OFF: usize = LFM3_PREFILL_SLOT_ROPE_PKT_0A * 4;
const PREFILL_ROPE_SUB0_SYNC_BYTES: usize = (2304 + 2048) * 4;
const PREFILL_ROPE_SYNC_BYTES: usize = LFM3_PREFILL_ACT_BYTES - PREFILL_ROPE_BYTE_OFF;
const LFM3_SWA_MAX_WINDOW: usize = (LFM3_MAX_KV_CHUNKS / 2) * LFM3_KV_CHUNK_T;
const LFM3_PREFILL_BO_ARGS: usize = 5;
const LFM3_DEC_BO_ARGS: usize = 6;
const LFM3_PREFILL_SLOT_FA_STATE: usize = 811_008;
const PREFILL_FA_STATE1_FLOATS: usize = 73_728;
const PREFILL_FA_STATE0_FLOATS: usize = 8 * 4 * 2304;
const PREFILL_FA_STATE_BYTES: usize = (PREFILL_FA_STATE1_FLOATS + PREFILL_FA_STATE0_FLOATS) * 4;

#[cfg(test)]
std::thread_local! {
    static RECORDED_RUN_ARGS: std::cell::RefCell<Vec<(&'static str, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn create_lfm3_run(
    kernel: &XdnaKernel,
    bos: &[&XdnaBo],
    expected_bos: usize,
    kind: &'static str,
) -> Result<XdnaRun, String> {
    debug_assert_eq!(bos.len(), expected_bos, "{kind} BO arg count mismatch");
    if bos.len() != expected_bos {
        return Err(format!(
            "{kind}: expected {expected_bos} BO args, got {}",
            bos.len()
        ));
    }
    #[cfg(test)]
    RECORDED_RUN_ARGS.with(|r| r.borrow_mut().push((kind, bos.len())));
    kernel.create_run(bos)
}

fn verify_uniform_swa_step(
    seen: &mut Option<(usize, usize, usize)>,
    window: usize,
    base: usize,
    bidx: usize,
    layer: usize,
) -> Result<(), String> {
    if let Some(&(prev_win, prev_base, prev_bidx)) = seen.as_ref() {
        if (window, base, bidx) != (prev_win, prev_base, prev_bidx) {
            return Err(format!(
                "layer {layer}: non-uniform SWA state (window={window}, base={base}, bidx={bidx}) != ({prev_win}, {prev_base}, {prev_bidx})"
            ));
        }
    } else {
        *seen = Some((window, base, bidx));
    }
    Ok(())
}

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
        stalled_runs.push(run);
        return Err(format!("{label}: {e}"));
    }
    Ok(())
}

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

#[derive(Clone, Copy)]
struct PrefillAttnMirrorEntry {
    layer: usize,
    pos0: usize,
    blk0: usize,
    loc0: usize,
    n0: usize,
    blk1: usize,
    loc1: usize,
    n1: usize,
}

#[inline]
fn write_f32_le_fast(buf: &mut [u8], float_idx: usize, vals: &[f32]) {
    #[cfg(target_endian = "little")]
    {
        let off = float_idx
            .checked_mul(4)
            .expect("write_f32_le_fast float_idx byte offset overflow");
        let byte_len = vals
            .len()
            .checked_mul(4)
            .expect("write_f32_le_fast slice byte length overflow");
        let end = off
            .checked_add(byte_len)
            .expect("write_f32_le_fast end byte offset overflow");
        buf[off..end].copy_from_slice(bytemuck::cast_slice(vals));
    }
    #[cfg(not(target_endian = "little"))]
    write_le_f32_slice(buf, float_idx, vals);
}

#[inline]
fn read_f32_le_fast(buf: &[u8], float_idx: usize, dst: &mut [f32]) {
    #[cfg(target_endian = "little")]
    {
        let off = float_idx
            .checked_mul(4)
            .expect("read_f32_le_fast float_idx byte offset overflow");
        let byte_len = dst
            .len()
            .checked_mul(4)
            .expect("read_f32_le_fast slice byte length overflow");
        let end = off
            .checked_add(byte_len)
            .expect("read_f32_le_fast end byte offset overflow");
        let src = &buf[off..end];
        if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(src) {
            dst.copy_from_slice(floats);
            return;
        }
        read_le_f32_slice(buf, float_idx, dst);
    }
    #[cfg(not(target_endian = "little"))]
    read_le_f32_slice(buf, float_idx, dst);
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
        pack_f32_mse_stage_k1024(dst, &f32_buf, rows, padded_cols)
            .map_err(|e| anyhow::anyhow!(e))?;
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
            pack_f32_mse_stage_k1024(
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
    let (embd_ref, logits_ref) = cpu.xdna_embed_logits_refs();
    let check_dequant = |label: &str, w: &WeightRef| -> Result<()> {
        ensure!(
            crate::model::transformer::supports_row_dequant(w.dtype),
            "{label}: dtype {:?} unsupported by CPU row dequantizer",
            w.dtype
        );
        Ok(())
    };
    check_dequant("token_embd", embd_ref)?;
    check_dequant("logits", logits_ref)?;

    let mut swa_window_expected: Option<usize> = None;
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
        check_dequant(&format!("layer {l} ffn_gate"), fg)?;
        check_dequant(&format!("layer {l} ffn_up"), fu)?;
        check_dequant(&format!("layer {l} ffn_down"), fd)?;
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
                check_dequant(&format!("layer {l} shortconv.in_proj"), in_proj)?;
                check_dequant(&format!("layer {l} shortconv.out_proj"), out_proj)?;
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
                if let Some(win) = cpu.xdna_lfm3_attn_window(l) {
                    ensure!(
                        win <= LFM3_SWA_MAX_WINDOW,
                        "layer {l}: SWA window {win} exceeds XDNA limit {LFM3_SWA_MAX_WINDOW}"
                    );
                    if let Some(prev) = swa_window_expected {
                        ensure!(
                            win == prev,
                            "layer {l}: non-uniform SWA window {win} != {prev} unsupported by XDNA plan key"
                        );
                    } else {
                        swa_window_expected = Some(win);
                    }
                }
                let (wq, wk, wv, wo) = cpu
                    .xdna_layer_attn_refs(l)
                    .with_context(|| format!("layer {l}: missing attention refs"))?;
                let wg = cpu
                    .xdna_lfm3_attn_gate_ref(l)
                    .with_context(|| format!("layer {l}: missing LFM3 attention gate"))?;
                check_dequant(&format!("layer {l} attn_q"), wq)?;
                check_dequant(&format!("layer {l} attn_k"), wk)?;
                check_dequant(&format!("layer {l} attn_v"), wv)?;
                check_dequant(&format!("layer {l} attn_output"), wo)?;
                check_dequant(&format!("layer {l} attn_gate"), wg)?;
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

    fn peek(&self, parity: bool, key: &K) -> Option<&V> {
        self.entries[usize::from(parity)]
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    fn clear(&mut self) {
        self.entries[0].clear();
        self.entries[1].clear();
    }
}

type Lfm3FastPlanMap = Lfm3ParityPlanCache<(usize, usize), Lfm3DecodeBucketPlan>;
type Lfm3ExtPlanMap = Lfm3ParityPlanCache<(usize, usize, bool, usize), Lfm3DecodeBucketPlan>;

#[derive(Clone, Copy)]
enum InFlightLfm3Run {
    DecFast {
        parity: bool,
        key: (usize, usize),
    },
    DecExt {
        parity: bool,
        key: (usize, usize, bool, usize),
    },
    PreFast {
        parity: bool,
        key: (usize, usize),
    },
    PreExt {
        parity: bool,
        key: (usize, usize, bool, usize),
    },
}

struct XdnaLfm3DecodeBuffers {
    // Drop compiled runlists before the BOs and kernels they reference.
    bucket_plans: Lfm3FastPlanMap,
    ext_plans: Lfm3ExtPlanMap,
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    bo_yn: XdnaBo,
    conv_kv_in_kvo: bool,
}

struct XdnaLfm3PrefillBuffers {
    // Drop compiled runlists before the BOs and kernels they reference.
    bucket_plans: Lfm3FastPlanMap,
    ext_plans: Lfm3ExtPlanMap,
    layer_bo_y: Vec<XdnaBo>,
    layer_bo_kvo: Vec<XdnaBo>,
    conv_kv_in_kvo: bool,
}

struct XdnaLfm3State {
    // Field order guarantees active/stalled runs and decode/prefill plans drop before BOs and `kernels` drops last.
    in_flight: Option<InFlightLfm3Run>,
    stalled_runs: Vec<XdnaRun>,
    decode: Option<XdnaLfm3DecodeBuffers>,
    prefill: Option<XdnaLfm3PrefillBuffers>,
    layer_bo_w: Vec<XdnaBo>,
    layer_kv_blocks: Vec<Vec<XdnaBo>>,
    swa_base_chunk: Vec<usize>,
    pre_bo_y: XdnaBo,
    pre_bo_yn: XdnaBo,
    pre_bo_kvo: XdnaBo,
    pending_kv: HybridPendingSlot,
    synced_len: usize,
    force_conv_sync: bool,
    head_q8_scratch: (Vec<f32>, Vec<i8>),
    kernels: Lfm3Kernels,
}

impl XdnaLfm3State {
    fn quiesce_in_flight(&mut self) -> Result<(), String> {
        while let Some(run) = self.stalled_runs.pop() {
            if let Err(e) = run.wait(QUIESCE_TIMEOUT_MS) {
                self.stalled_runs.push(run);
                return Err(format!("lfm3 stalled run wait failed: {e}"));
            }
        }
        let Some(run) = self.in_flight.as_ref() else {
            return Ok(());
        };
        let list = match run {
            InFlightLfm3Run::DecFast { parity, key } => self
                .decode
                .as_ref()
                .and_then(|d| d.bucket_plans.peek(*parity, key))
                .map(|p| &p.runlist),
            InFlightLfm3Run::DecExt { parity, key } => self
                .decode
                .as_ref()
                .and_then(|d| d.ext_plans.peek(*parity, key))
                .map(|p| &p.runlist),
            InFlightLfm3Run::PreFast { parity, key } => self
                .prefill
                .as_ref()
                .and_then(|p| p.bucket_plans.peek(*parity, key))
                .map(|p| &p.runlist),
            InFlightLfm3Run::PreExt { parity, key } => self
                .prefill
                .as_ref()
                .and_then(|p| p.ext_plans.peek(*parity, key))
                .map(|p| &p.runlist),
        };
        if let Some(runlist) = list {
            runlist
                .wait(QUIESCE_TIMEOUT_MS)
                .map_err(|e| format!("lfm3 quiesce wait failed: {e}"))?;
        }
        self.in_flight = None;
        Ok(())
    }
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
    #[cfg(test)]
    bidir_host_ffn_chunk_limit: Option<usize>,
}

impl Drop for XdnaLfm3Model {
    fn drop(&mut self) {
        let state = match self.npu.get_mut() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        if let Err(e) = state.quiesce_in_flight() {
            tracing::warn!("XdnaLfm3Model::drop: quiesce failed: {e}");
            eprintln!("cera: XdnaLfm3Model::drop: quiesce failed: {e}");
        }
    }
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

            let swa_window = cpu.xdna_lfm3_attn_window(l).unwrap_or(0);
            let prealloc_blocks = if is_conv || swa_window > 0 {
                1
            } else {
                cfg.max_seq_len.div_ceil(LFM3_BLOCK_TOKENS).clamp(1, 2)
            };
            let mut blocks = Vec::with_capacity(prealloc_blocks);
            for _ in 0..prealloc_blocks {
                let mut bo_kv = kernels
                    .prefill_conv
                    .alloc_bo(LFM3_KV_BYTES, 6)
                    .map_err(|e| anyhow::anyhow!("layer {l} bo_kv alloc failed: {e}"))?;
                bo_kv.as_mut_slice().fill(0);
                bo_kv
                    .sync_to_device()
                    .map_err(|e| anyhow::anyhow!("layer {l} bo_kv sync failed: {e}"))?;
                blocks.push(bo_kv);
            }
            layer_kv_blocks.push(blocks);

            let (rope_theta, rope_freq_factors) = cpu.xdna_lfm3_rope_params(l);
            layers.push(Lfm3LayerStatic {
                is_conv,
                is_non_causal: cpu.layer_is_non_causal(l),
                swa_window,
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
                in_flight: None,
                stalled_runs: Vec::new(),
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
                head_q8_scratch: (Vec::new(), Vec::new()),
            }),
            state_torn: AtomicBool::new(false),
            decode_error: Mutex::new(None),
            session_gate: ModelSessionGate::default(),
            profile_decode: crate::backend::cpu_features::profile_decode_enabled(),
            _prefill_warm_guard: {
                let g = is_causal.then(crate::backend::acquire_prefill_warm_spin);
                if is_causal {
                    crate::backend::warm_prefill_pool(LfmModel::xdna_head_threads());
                }
                g
            },
            #[cfg(test)]
            bidir_host_ffn_chunk_limit: None,
        })
    }

    #[cfg(test)]
    fn should_run_bidir_host_ffn(&self, chunk: usize) -> bool {
        self.bidir_host_ffn_chunk_limit
            .is_none_or(|limit| chunk < limit)
    }

    #[cfg(not(test))]
    #[inline]
    fn should_run_bidir_host_ffn(&self, _chunk: usize) -> bool {
        true
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
                    Some(LFM3_DEC_SLOT_FIN_NORM),
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
            ext_plans: Lfm3ParityPlanCache::new(8),
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
    fn run_prefill_paged_fa_blocks(
        kernels: &mut Lfm3Kernels,
        bo_w: &XdnaBo,
        pre_bo_y: &mut XdnaBo,
        pre_bo_yn: &mut XdnaBo,
        kv_blocks: &[XdnaBo],
        pre_bo_kvo: &XdnaBo,
        stalled_runs: &mut Vec<XdnaRun>,
        layer_idx: usize,
        chunk_idx: usize,
        pos0: usize,
        n_valid: usize,
        n_past: usize,
        is_causal: bool,
        skip_cur_kv: bool,
        num_blocks: usize,
        eps: f32,
        tag: &str,
    ) -> Result<(), String> {
        for (blk, kv_blk) in kv_blocks.iter().enumerate().take(num_blocks) {
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
                    if use_part0 { 6 } else { 4 },
                    eps,
                    blk * LFM3_MAX_KV_CHUNKS,
                    has_fa_state,
                    skip_cur_kv,
                );
            }
            let run_p = pre_bo_y
                .sync_range_to_device(STEP_PARAMS_BYTES, sync_off)
                .and_then(|()| {
                    let k = if use_part0 {
                        kernels.prefill_fa_part0()?
                    } else {
                        kernels.prefill_fa_part()?
                    };
                    create_lfm3_run(
                        k,
                        &[bo_w, pre_bo_y, pre_bo_yn, kv_blk, pre_bo_kvo],
                        LFM3_PREFILL_BO_ARGS,
                        if use_part0 {
                            "prefill_fa_part0"
                        } else {
                            "prefill_fa_part"
                        },
                    )
                })
                .map_err(|e| {
                    format!("layer {layer_idx} {tag} fa_part c={chunk_idx} blk={blk}: {e}")
                })?;
            exec_owned_run(
                stalled_runs,
                run_p,
                &format!("layer {layer_idx} {tag} fa_part c={chunk_idx} blk={blk}"),
            )?;
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
        stalled_runs: &mut Vec<XdnaRun>,
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
        Self::run_prefill_paged_fa_blocks(
            kernels,
            bo_w,
            pre_bo_y,
            pre_bo_yn,
            kv_blocks,
            pre_bo_kvo,
            stalled_runs,
            layer_idx,
            chunk_idx,
            pos0,
            n_valid,
            n_past,
            is_causal,
            skip_cur_kv,
            num_full,
            eps,
            tag,
        )?;
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
        let run_f = pre_bo_y
            .sync_range_to_device(STEP_PARAMS_BYTES, PREFILL_NORM3_PARAMS_BYTE_OFF)
            .and_then(|()| {
                create_lfm3_run(
                    kernels.prefill_fa_fin(fin_bidx)?,
                    &[bo_w, pre_bo_y, pre_bo_yn, &kv_blocks[fin_blk], pre_bo_kvo],
                    LFM3_PREFILL_BO_ARGS,
                    "prefill_fa_fin",
                )
            })
            .map_err(|e| format!("layer {layer_idx} {tag} fa_fin c={chunk_idx}: {e}"))?;
        exec_owned_run(
            stalled_runs,
            run_f,
            &format!("layer {layer_idx} {tag} fa_fin c={chunk_idx}"),
        )?;
        pre_bo_yn
            .sync_from_device()
            .map_err(|e| format!("layer {layer_idx} {tag} fa_fin c={chunk_idx}: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefill_bidir_paged_fa(
        kernels: &mut Lfm3Kernels,
        bo_w: &XdnaBo,
        pre_bo_y: &mut XdnaBo,
        pre_bo_yn: &mut XdnaBo,
        kv_blocks: &[XdnaBo],
        pre_bo_kvo: &XdnaBo,
        stalled_runs: &mut Vec<XdnaRun>,
        layer_idx: usize,
        chunk_idx: usize,
        pos0: usize,
        n_valid: usize,
        total_tokens: usize,
        num_blocks: usize,
        eps: f32,
    ) -> Result<(), String> {
        // Shipped `q4prefill_lfm3.xclbin` Mode 5 ignores `g_params[10]` (`skip_cur_kv`),
        // so `>8K` bidirectional prefill runs Mode 6 (`fa_part0`) + Mode 4 (`fa_part`)
        // across all KV blocks and finalizes `SLOT_FA_STATE` + `O_PROJ` + `FFN` on host.
        Self::run_prefill_paged_fa_blocks(
            kernels,
            bo_w,
            pre_bo_y,
            pre_bo_yn,
            kv_blocks,
            pre_bo_kvo,
            stalled_runs,
            layer_idx,
            chunk_idx,
            pos0,
            n_valid,
            total_tokens,
            false,
            false,
            num_blocks,
            eps,
            "bidir >8K",
        )?;
        pre_bo_y
            .sync_range_from_device(PREFILL_FA_STATE_BYTES, LFM3_PREFILL_SLOT_FA_STATE * 4)
            .map_err(|e| {
                format!("layer {layer_idx} bidir >8K fa_state readback c={chunk_idx}: {e}")
            })?;
        Ok(())
    }

    fn unpack_bidir_fa_state_into_attn_out(
        bo_y_bytes: &[u8],
        saved_gate_bytes: &[u8],
        n_valid: usize,
        attn_out: &mut [f32],
    ) {
        for t in 0..n_valid {
            let sub = t >> 5;
            let r = (t >> 3) & 3;
            let lt = t & 7;
            for h in 0..LFM3_N_HEADS {
                let c = h >> 1;
                let h_local = h & 1;
                let g_off = (t * LFM3_N_HEADS + h) * 4;
                let g_logit =
                    f32::from_le_bytes(saved_gate_bytes[g_off..g_off + 4].try_into().unwrap());
                let gate_sig = 1.0f32 / (1.0f32 + (-g_logit).exp());
                let state0_core = LFM3_PREFILL_SLOT_FA_STATE
                    + PREFILL_FA_STATE1_FLOATS
                    + c * (4 * 2304)
                    + r * 2304;
                let row_l_f32 = state0_core + 2048 + h_local * 16 + sub * 8 + lt;
                let row_l = f32::from_le_bytes(
                    bo_y_bytes[row_l_f32 * 4..row_l_f32 * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                let scale = gate_sig / row_l.max(1e-35);
                let oa_core = if sub == 0 {
                    state0_core
                } else {
                    LFM3_PREFILL_SLOT_FA_STATE + (c >> 1) * (4 * 4608) + (c & 1) * 2048 + r * 4608
                };
                let oa_head = oa_core + h_local * 1024;
                let dst_head = &mut attn_out[t * LFM3_H + h * LFM3_HEAD_DIM..][..LFM3_HEAD_DIM];
                for d8 in 0..(LFM3_HEAD_DIM / 8) {
                    let src_f32 = oa_head + d8 * 64 + lt * 8;
                    for lane in 0..8 {
                        let b = (src_f32 + lane) * 4;
                        let v = f32::from_le_bytes(bo_y_bytes[b..b + 4].try_into().unwrap());
                        dst_head[d8 * 8 + lane] = v * scale;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_rowmajor_f32(
        gguf: &GgufFile,
        wref: &WeightRef,
        input: &[f32],
        output: &mut [f32],
        n: usize,
        m: usize,
        k: usize,
        bq_scales: &mut Vec<f32>,
        bq_quants: &mut Vec<i8>,
        #[cfg(has_blas)] dequant_scratch: &mut Vec<f32>,
    ) {
        #[cfg(has_blas)]
        if transformer::batched_gemm_supports(wref.dtype, k)
            && n >= 16
            && transformer::try_blas_prefill_gemm_rowmajor(
                gguf,
                wref,
                input,
                output,
                n,
                m,
                k,
                dequant_scratch,
            )
        {
            return;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(has_blas)))]
        if transformer::batched_gemm_supports(wref.dtype, k) && n >= 16 {
            if transformer::try_repacked_f32_gemm_rowmajor(wref, input, output, n, m, k) {
                return;
            }
            let nb = k / 32;
            if bq_scales.len() < n * nb {
                bq_scales.resize(n * nb, 0.0);
            }
            if bq_quants.len() < n * k {
                bq_quants.resize(n * k, 0);
            }
            let b_scales = &mut bq_scales[..n * nb];
            let b_quants = &mut bq_quants[..n * k];
            transformer::quantize_rows(input, k, n, b_scales, b_quants);
            if transformer::gemm_preq_rowmajor(gguf, wref, b_scales, b_quants, output, n, m, k) {
                return;
            }
        }
        #[cfg(not(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(has_blas))))]
        let _ = (bq_scales, bq_quants);
        for t in 0..n {
            transformer::gemv(
                gguf,
                wref,
                &input[t * k..(t + 1) * k],
                &mut output[t * m..(t + 1) * m],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_bidir_fa_chunk_and_ffn(
        &self,
        layer_idx: usize,
        bo_y_bytes: &[u8],
        saved_gate_bytes: &[u8],
        n_valid: usize,
        eps: f32,
        hidden_chunk: &mut [f32],
        attn_out: &mut [f32],
        block_out: &mut [f32],
        bq_scales: &mut Vec<f32>,
        bq_quants: &mut Vec<i8>,
        #[cfg(has_blas)] dequant_scratch: &mut Vec<f32>,
        moe_scratch: &mut crate::model::lfm2::XdnaMoePrefillScratch,
        state: &mut InferenceState,
    ) -> Result<(), String> {
        Self::unpack_bidir_fa_state_into_attn_out(bo_y_bytes, saved_gate_bytes, n_valid, attn_out);
        let (_, _, _, wo) = self
            .cpu
            .xdna_layer_attn_refs(layer_idx)
            .ok_or_else(|| format!("layer {layer_idx}: missing attention output weight ref"))?;
        Self::project_rowmajor_f32(
            self.cpu.gguf(),
            wo,
            &attn_out[..n_valid * LFM3_H],
            &mut block_out[..n_valid * LFM3_H],
            n_valid,
            LFM3_H,
            LFM3_Q_DIM,
            bq_scales,
            bq_quants,
            #[cfg(has_blas)]
            dequant_scratch,
        );
        let post_attn_norm = &self.layers[layer_idx].post_attn_norm;
        for t in 0..n_valid {
            crate::backend::cpu::rmsnorm(
                &mut block_out[t * LFM3_H..(t + 1) * LFM3_H],
                post_attn_norm,
                eps,
            );
        }
        if n_valid >= 8 {
            crate::backend::cpu::par_add_inplace(
                &mut hidden_chunk[..n_valid * LFM3_H],
                &block_out[..n_valid * LFM3_H],
            );
        } else {
            crate::backend::cpu::add_inplace(
                &mut hidden_chunk[..n_valid * LFM3_H],
                &block_out[..n_valid * LFM3_H],
            );
        }
        self.cpu
            .xdna_prefill_ffn_chunk(layer_idx, hidden_chunk, n_valid, moe_scratch, state);
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
                create_lfm3_run(
                    &kernels.prefill_conv,
                    &[&layer_bo_w[l], &pre_bo_y_layers[l], y_next, kv_in, kv_out],
                    LFM3_PREFILL_BO_ARGS,
                    "prefill_conv",
                )
            } else {
                let bidx = if layer.swa_window > 0 {
                    swa_bidx
                } else {
                    attn_bidx
                };
                create_lfm3_run(
                    kernels.prefill_attn(bidx)?,
                    &[
                        &layer_bo_w[l],
                        &pre_bo_y_layers[l],
                        y_next,
                        &layer_kv_blocks[l][0],
                        &pre_bo_kvo_layers[l],
                    ],
                    LFM3_PREFILL_BO_ARGS,
                    "prefill_attn",
                )
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
        if num_full != 1 {
            return Err(format!(
                "prefill ext cached plan requires num_full == 1, got {num_full}"
            ));
        }
        kernels.prefill_attn(swa_bidx)?;
        kernels.prefill_bidir_a()?;
        kernels.prefill_fa_part0()?;
        kernels.prefill_fa_fin(fin_bidx)?;
        let n_layers = layers.len();
        let fin_blk = usize::from(has_rem);
        let mut runs = Vec::with_capacity(n_layers * 3);
        let k_a = kernels.prefill_bidir_a.as_ref().expect("loaded");
        let k_part0 = kernels.prefill_fa_part0.as_ref().expect("loaded");
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
                    create_lfm3_run(
                        &kernels.prefill_conv,
                        &[&layer_bo_w[l], &pre_bo_y_layers[l], y_next, kv_in, kv_out],
                        LFM3_PREFILL_BO_ARGS,
                        "prefill_ext_conv",
                    )
                    .map_err(|e| format!("layer {l} prefill ext conv create_run: {e}"))?,
                );
            } else if layer.swa_window > 0 {
                let k_swa = kernels.prefill_attn[swa_bidx].as_ref().expect("loaded");
                runs.push(
                    create_lfm3_run(
                        k_swa,
                        &[
                            &layer_bo_w[l],
                            &pre_bo_y_layers[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &pre_bo_kvo_layers[l],
                        ],
                        LFM3_PREFILL_BO_ARGS,
                        "prefill_ext_swa",
                    )
                    .map_err(|e| format!("layer {l} prefill ext swa create_run: {e}"))?,
                );
            } else {
                runs.push(
                    create_lfm3_run(
                        k_a,
                        &[
                            &layer_bo_w[l],
                            &pre_bo_y_layers[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &pre_bo_kvo_layers[l],
                        ],
                        LFM3_PREFILL_BO_ARGS,
                        "prefill_ext_phase_a",
                    )
                    .map_err(|e| format!("layer {l} prefill ext phase_a create_run: {e}"))?,
                );
                runs.push(
                    create_lfm3_run(
                        k_part0,
                        &[
                            &layer_bo_w[l],
                            &pre_bo_y_layers[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &pre_bo_kvo_layers[l],
                        ],
                        LFM3_PREFILL_BO_ARGS,
                        "prefill_ext_fa_part0",
                    )
                    .map_err(|e| format!("layer {l} prefill ext fa_part blk=0 create_run: {e}"))?,
                );
                runs.push(
                    create_lfm3_run(
                        k_fin,
                        &[
                            &layer_bo_w[l],
                            &pre_bo_y_layers[l],
                            y_next,
                            &layer_kv_blocks[l][fin_blk],
                            &pre_bo_kvo_layers[l],
                        ],
                        LFM3_PREFILL_BO_ARGS,
                        "prefill_ext_fa_fin",
                    )
                    .map_err(|e| format!("layer {l} prefill ext fa_fin create_run: {e}"))?,
                );
            }
        }
        Self::assemble_runlist(&kernels.prefill_conv, runs, "prefill_ext")
    }

    fn lock_npu(&self) -> MutexGuard<'_, XdnaLfm3State> {
        match self.npu.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                tracing::warn!(
                    "cera::xdna: LFM3 NPU mutex poisoned after prior panic; discarding pending KV and tearing state"
                );
                self.state_torn.store(true, Ordering::SeqCst);
                record_first_fault(
                    &self.decode_error,
                    CeraError::Backend(
                        "XDNA NPU lock poisoned after a fault; call try_reset_kv".to_string(),
                    ),
                );
                let mut guard = poisoned.into_inner();
                guard.pending_kv.discard();
                self.npu.clear_poison();
                guard
            }
        }
    }

    fn reset_npu_session_state(&self, npu: &mut XdnaLfm3State) -> Result<(), CeraError> {
        if let Err(e) = npu.quiesce_in_flight() {
            self.state_torn.store(true, Ordering::SeqCst);
            return Err(CeraError::Backend(e));
        }
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
            dec.ext_plans.clear();
        }
        if let Some(pre) = npu.prefill.as_mut() {
            pre.conv_kv_in_kvo = false;
            pre.ext_plans.clear();
        }
        let _ = take_fault(&self.decode_error);
        self.state_torn.store(false, Ordering::SeqCst);
        Ok(())
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

    fn validate_cpu_state_for_sync(
        &self,
        state: &InferenceState,
        pending: Option<&PendingHybridCpuKv>,
        target_pos: usize,
    ) -> Result<(), String> {
        self.validate_state(state)?;
        if self.cpu.config().is_causal && target_pos != state.seq_len {
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
            .checked_mul(LFM3_KV_DIM)
            .ok_or_else(|| "target_pos overflows KV cache length".to_string())?;
        let use_f16 = state.kv_f16;
        for (l, layer) in self.layers.iter().enumerate() {
            if layer.is_conv && target_pos > 0 {
                match &state.layers[l] {
                    LayerState::Conv { buffer, .. } => {
                        let expect = CONV_TAPS_STORED * LFM3_H;
                        if buffer.len() < expect {
                            return Err(format!(
                                "layer {l} conv state len {} < {expect}",
                                buffer.len()
                            ));
                        }
                    }
                    _ => return Err(format!("layer {l} expected Conv state")),
                }
            } else if !layer.is_conv && target_pos > 0 {
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
                    }
                    _ => return Err(format!("layer {l} expected Attention state")),
                }
            }
        }
        Ok(())
    }

    fn apply_truncate_rewind_sync(
        &self,
        npu: &mut XdnaLfm3State,
        state: &mut InferenceState,
        len: usize,
        truncate_cpu: impl FnOnce(&mut InferenceState) -> Result<(), KvRewindError>,
    ) -> Result<(), KvRewindError> {
        let prev_synced = npu.synced_len;
        if !npu.pending_kv.discard_for_truncate(state, len) {
            npu.synced_len = 0;
        }
        if len < state.seq_len {
            truncate_cpu(state)?;
        }
        if (npu.synced_len == 0 || state.seq_len < prev_synced)
            && npu.swa_base_chunk.iter().any(|&b| b > 0)
        {
            npu.synced_len = 0;
            npu.swa_base_chunk.fill(0);
        } else {
            npu.synced_len = npu.synced_len.min(state.seq_len);
        }
        npu.force_conv_sync = true;
        Ok(())
    }

    fn sync_cpu_state_to_npu(
        &self,
        npu: &mut XdnaLfm3State,
        state: &mut InferenceState,
        target_pos: usize,
    ) -> Result<(), SyncErr> {
        let pending_opt = self
            .cpu
            .config()
            .is_causal
            .then_some(npu.pending_kv.as_ref())
            .flatten();
        self.validate_cpu_state_for_sync(state, pending_opt, target_pos)
            .map_err(SyncErr::Input)?;
        let swa_dirty_at_zero = target_pos == 0 && npu.swa_base_chunk.iter().any(|&b| b != 0);
        // Preserve pending CPU KV across no-op sync checks so callers can still overlap its
        // flush with async NPU execution when no host-to-device resync is required.
        if !npu.force_conv_sync
            && target_pos == npu.synced_len
            && !swa_dirty_at_zero
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
            npu.swa_base_chunk.fill(0);
            // Re-verify post-discard state because KV still held `pending_pos < target_pos` rows.
            self.validate_cpu_state_for_sync(state, None, target_pos)
                .map_err(SyncErr::Input)?;
        }
        let sync_conv = npu.force_conv_sync || target_pos != npu.synced_len;
        let sync_attn = (target_pos != npu.synced_len)
            || (target_pos == 0 && npu.swa_base_chunk.iter().any(|&b| b != 0));
        if !sync_conv && !sync_attn {
            return Ok(());
        }
        let rewind = target_pos < npu.synced_len;
        let start = if rewind { 0 } else { npu.synced_len };
        let use_f16 = state.kv_f16;

        // Invalidate NPU sync markers before mutating device buffers so a mid-sync
        // alloc/DMA failure never leaves `synced_len` matching `target_pos`.
        npu.synced_len = 0;
        npu.force_conv_sync = true;
        if start == 0 {
            npu.swa_base_chunk.fill(0);
        }

        let res = (|| -> Result<(), SyncErr> {
            for (l, layer) in self.layers.iter().enumerate() {
                if layer.is_conv {
                    let kv_buf = npu.layer_kv_blocks[l][0].as_mut_slice();
                    if sync_conv {
                        if target_pos == 0 {
                            kv_buf[..LFM3_KVO_BYTES].fill(0);
                        } else if let LayerState::Conv { buffer, .. } = &state.layers[l] {
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
                            return Err(SyncErr::Input(format!("layer {l} expected Conv state")));
                        }
                        npu.layer_kv_blocks[l][0]
                            .sync_range_to_device(LFM3_KVO_BYTES, 0)
                            .map_err(|e| {
                                SyncErr::Device(format!("layer {l} conv bo_kv sync failed: {e}"))
                            })?;
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
                            let mut eff_start = start;
                            if layer.swa_window > 0 {
                                let swa_limit = lfm3_swa_limit_chunks(layer.swa_window);
                                let last_chunk = (target_pos - 1) / LFM3_KV_CHUNK_T;
                                if rewind || npu.swa_base_chunk[l] == 0 && start == 0 {
                                    npu.swa_base_chunk[l] = if last_chunk < swa_limit {
                                        0
                                    } else {
                                        target_pos.saturating_sub(layer.swa_window)
                                            / LFM3_KV_CHUNK_T
                                    };
                                    eff_start = npu.swa_base_chunk[l] * LFM3_KV_CHUNK_T;
                                } else {
                                    Self::ensure_swa_capacity_for_chunk(
                                        &mut npu.layer_kv_blocks[l][0],
                                        &mut npu.swa_base_chunk[l],
                                        layer.swa_window,
                                        last_chunk,
                                    )
                                    .map_err(SyncErr::Device)?;
                                    eff_start =
                                        eff_start.max(npu.swa_base_chunk[l] * LFM3_KV_CHUNK_T);
                                }
                            } else {
                                let max_blk =
                                    ((target_pos - 1) / LFM3_KV_CHUNK_T) / LFM3_MAX_KV_CHUNKS;
                                Self::ensure_paged_block(
                                    &mut npu.layer_kv_blocks[l],
                                    &npu.kernels.prefill_conv,
                                    max_blk,
                                )
                                .map_err(SyncErr::Device)?;
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
                                            SyncErr::Device(format!(
                                                "layer {l} kvh {kvh} swa bo_kv sync failed: {e}"
                                            ))
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
                                                SyncErr::Device(format!(
                                                    "layer {l} blk {blk} kvh {kvh} bo_kv sync failed: {e}"
                                                ))
                                            })?;
                                    }
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
            Ok(())
        })();
        if let Err(e) = res {
            npu.swa_base_chunk.fill(0);
            return Err(e);
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
    fn save_attn_prefill_sub_kvo_to_bo_kv(
        kvo_bytes: &[u8],
        kvo_head_off: usize,
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        swa_base: &mut usize,
        swa_window: usize,
        pos0: usize,
    ) -> Result<(usize, usize, usize), String> {
        let global_chunk = pos0 / LFM3_KV_CHUNK_T;
        let swa_before = *swa_base;
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
        let shift = swa_base.saturating_sub(swa_before);
        let bo_kv = &mut blocks[blk];
        let kb = bo_kv.as_mut_slice();
        for kvh in 0..LFM3_N_KV_HEADS {
            let kvo_off = (kvo_head_off + kvh) * TILE_BYTES;
            let src = &kvo_bytes[kvo_off..kvo_off + TILE_BYTES];
            let dst_off = (kvh * LFM3_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES;
            kb[dst_off..dst_off + TILE_BYTES].copy_from_slice(src);
        }
        Ok((blk, loc_chunk, shift))
    }

    fn sync_bo_kv_sub_chunk(
        bo_kv: &XdnaBo,
        loc_chunk: usize,
        n_chunks: usize,
        global_chunk: usize,
    ) -> Result<(), String> {
        let sync_bytes = n_chunks * TILE_BYTES;
        for kvh in 0..LFM3_N_KV_HEADS {
            let dst_off = (kvh * LFM3_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES;
            bo_kv
                .sync_range_to_device(sync_bytes, dst_off)
                .map_err(|e| {
                    format!("attn chunk {global_chunk} kvh {kvh} bo_kv sync failed: {e}")
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn save_attn_prefill_kvo_to_bo_kv(
        kvo: &XdnaBo,
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        swa_base: &mut usize,
        swa_window: usize,
        layer: usize,
        pos0: usize,
        n_valid: usize,
    ) -> Result<PrefillAttnMirrorEntry, String> {
        let kvo_bytes = kvo.as_slice();
        let n0 = n_valid.min(LFM3_KV_CHUNK_T);
        let (blk0, mut loc0, _) = Self::save_attn_prefill_sub_kvo_to_bo_kv(
            kvo_bytes, 0, blocks, allocator, swa_base, swa_window, pos0,
        )?;
        let (blk1, loc1, n1) = if n_valid > LFM3_KV_CHUNK_T {
            let n1 = n_valid - LFM3_KV_CHUNK_T;
            let (b1, l1, shift1) = Self::save_attn_prefill_sub_kvo_to_bo_kv(
                kvo_bytes,
                LFM3_N_KV_HEADS,
                blocks,
                allocator,
                swa_base,
                swa_window,
                pos0 + LFM3_KV_CHUNK_T,
            )?;
            loc0 = loc0.saturating_sub(shift1);
            (b1, l1, n1)
        } else {
            (blk0, loc0, 0)
        };
        let gc0 = pos0 / LFM3_KV_CHUNK_T;
        if n1 > 0 && blk0 == blk1 && loc1 == loc0 + 1 {
            Self::sync_bo_kv_sub_chunk(&blocks[blk0], loc0, 2, gc0)?;
        } else {
            Self::sync_bo_kv_sub_chunk(&blocks[blk0], loc0, 1, gc0)?;
            if n1 > 0 {
                Self::sync_bo_kv_sub_chunk(&blocks[blk1], loc1, 1, gc0 + 1)?;
            }
        }
        Ok(PrefillAttnMirrorEntry {
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
        let needed = (pos0 + n_valid) * LFM3_KV_DIM;
        let head_bytes = LFM3_HEAD_DIM * 2;
        if use_f16 {
            if key_cache_f16.len() < needed {
                key_cache_f16.resize(needed, 0);
            }
            if value_cache_f16.len() < needed {
                value_cache_f16.resize(needed, 0);
            }
            for t in 0..n_valid {
                let dst_row = (pos0 + t) * LFM3_KV_DIM;
                for kvh in 0..LFM3_N_KV_HEADS {
                    let k_src =
                        (kvh * LFM3_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES + t * head_bytes;
                    let v_src = k_src + ATTN_KV_CHUNK_V_OFFSET;
                    let dst = dst_row + kvh * LFM3_HEAD_DIM;
                    crate::backend::cpu::bf16_le_bytes_to_f16_into(
                        &bo_bytes[k_src..k_src + head_bytes],
                        &mut key_cache_f16[dst..dst + LFM3_HEAD_DIM],
                    );
                    crate::backend::cpu::bf16_le_bytes_to_f16_into(
                        &bo_bytes[v_src..v_src + head_bytes],
                        &mut value_cache_f16[dst..dst + LFM3_HEAD_DIM],
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
                let dst_row = (pos0 + t) * LFM3_KV_DIM;
                for kvh in 0..LFM3_N_KV_HEADS {
                    let k_src =
                        (kvh * LFM3_MAX_KV_CHUNKS + loc_chunk) * TILE_BYTES + t * head_bytes;
                    let v_src = k_src + ATTN_KV_CHUNK_V_OFFSET;
                    let dst = dst_row + kvh * LFM3_HEAD_DIM;
                    crate::backend::cpu::bf16_le_bytes_to_f32_into(
                        &bo_bytes[k_src..k_src + head_bytes],
                        &mut key_cache[dst..dst + LFM3_HEAD_DIM],
                    );
                    crate::backend::cpu::bf16_le_bytes_to_f32_into(
                        &bo_bytes[v_src..v_src + head_bytes],
                        &mut value_cache[dst..dst + LFM3_HEAD_DIM],
                    );
                }
            }
        }
    }

    fn mirror_attn_prefill_bo_kv_to_state(
        blocks: &[XdnaBo],
        layer_state: &mut LayerState,
        entry: &PrefillAttnMirrorEntry,
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
                entry.pos0 + LFM3_KV_CHUNK_T,
                entry.n1,
                use_f16,
            );
        }
    }

    fn flush_pending_attn_mirrors(
        pending: &mut Vec<PrefillAttnMirrorEntry>,
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

    #[allow(clippy::too_many_arguments)]
    fn save_attn_prefill_kvo_to_kv_and_state(
        kvo: &XdnaBo,
        blocks: &mut Vec<XdnaBo>,
        allocator: &XdnaKernel,
        swa_base: &mut usize,
        swa_window: usize,
        layer_state: Option<&mut LayerState>,
        pos0: usize,
        n_valid: usize,
        use_f16: bool,
    ) -> Result<(), String> {
        let entry = Self::save_attn_prefill_kvo_to_bo_kv(
            kvo, blocks, allocator, swa_base, swa_window, 0, pos0, n_valid,
        )?;
        if let Some(ls) = layer_state {
            Self::mirror_attn_prefill_bo_kv_to_state(blocks, ls, &entry, use_f16);
        }
        Ok(())
    }

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
            pending.push_attn_bf16(
                layer_idx,
                kvo_bytes,
                LFM3_N_KV_HEADS,
                LFM3_HEAD_DIM,
                512 * 4,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefill_single_chunk(
        kernel: &XdnaKernel,
        stalled_runs: &mut Vec<XdnaRun>,
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
        let run = create_lfm3_run(
            kernel,
            &[bo_w, bo_y, bo_yn, bo_kv, bo_kvo],
            LFM3_PREFILL_BO_ARGS,
            "prefill_single",
        )
        .map_err(|e| format!("prefill create_run: {e}"))?;
        exec_owned_run(stalled_runs, run, "prefill wait")?;
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
                create_lfm3_run(
                    kernels.dec_conv()?,
                    &[
                        &layer_bo_w[l],
                        &dec_bo_y[l],
                        y_next,
                        kv_in,
                        kv_out,
                        &layer_kv_blocks[l][0],
                    ],
                    LFM3_DEC_BO_ARGS,
                    "dec_conv",
                )
            } else {
                let bidx = if layer.swa_window > 0 {
                    swa_bidx
                } else {
                    attn_bidx
                };
                let kv_hi = layer_kv_blocks[l].get(1).unwrap_or(&layer_kv_blocks[l][0]);
                create_lfm3_run(
                    kernels.dec_attn(bidx)?,
                    &[
                        &layer_bo_w[l],
                        &dec_bo_y[l],
                        y_next,
                        &layer_kv_blocks[l][0],
                        &dec_bo_kvo[l],
                        kv_hi,
                    ],
                    LFM3_DEC_BO_ARGS,
                    "dec_attn",
                )
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
        if num_full != 1 {
            return Err(format!(
                "decode ext cached plan requires num_full == 1, got {num_full}"
            ));
        }
        kernels.dec_conv()?;
        kernels.dec_attn(swa_bidx)?;
        kernels.dec_phase_a()?;
        kernels.dec_fa_part()?;
        kernels.dec_fa_fin(fin_bidx)?;
        let n_layers = layers.len();
        let fin_blk = usize::from(has_rem);
        let mut runs = Vec::with_capacity(n_layers * 3);
        let k_conv = kernels.dec_conv.as_ref().expect("loaded");
        let k_swa = kernels.dec_attn[swa_bidx].as_ref().expect("loaded");
        let k_a = kernels.dec_phase_a.as_ref().expect("loaded");
        let k_part = kernels.dec_fa_part.as_ref().expect("loaded");
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
                    create_lfm3_run(
                        k_conv,
                        &[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            kv_in,
                            kv_out,
                            &layer_kv_blocks[l][0],
                        ],
                        LFM3_DEC_BO_ARGS,
                        "dec_ext_conv",
                    )
                    .map_err(|e| format!("layer {l} dec ext conv create_run: {e}"))?,
                );
            } else if layer.swa_window > 0 {
                runs.push(
                    create_lfm3_run(
                        k_swa,
                        &[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &dec_bo_kvo[l],
                            &layer_kv_blocks[l][0],
                        ],
                        LFM3_DEC_BO_ARGS,
                        "dec_ext_swa",
                    )
                    .map_err(|e| format!("layer {l} dec ext swa create_run: {e}"))?,
                );
            } else {
                runs.push(
                    create_lfm3_run(
                        k_a,
                        &[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &dec_bo_kvo[l],
                            &layer_kv_blocks[l][0],
                        ],
                        LFM3_DEC_BO_ARGS,
                        "dec_ext_phase_a",
                    )
                    .map_err(|e| format!("layer {l} dec ext phase_a create_run: {e}"))?,
                );
                runs.push(
                    create_lfm3_run(
                        k_part,
                        &[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][0],
                            &dec_bo_kvo[l],
                            &layer_kv_blocks[l][0],
                        ],
                        LFM3_DEC_BO_ARGS,
                        "dec_ext_fa_part",
                    )
                    .map_err(|e| format!("layer {l} dec ext fa_part blk=0 create_run: {e}"))?,
                );
                runs.push(
                    create_lfm3_run(
                        k_fin,
                        &[
                            &layer_bo_w[l],
                            &dec_bo_y[l],
                            y_next,
                            &layer_kv_blocks[l][fin_blk],
                            &dec_bo_kvo[l],
                            &layer_kv_blocks[l][0],
                        ],
                        LFM3_DEC_BO_ARGS,
                        "dec_ext_fa_fin",
                    )
                    .map_err(|e| format!("layer {l} dec ext fa_fin create_run: {e}"))?,
                );
            }
        }
        Self::assemble_runlist(k_conv, runs, "dec_ext")
    }

    #[allow(clippy::too_many_arguments)]
    fn run_decode_ext_multi_block_step(
        layers: &[Lfm3LayerStatic],
        kernels: &mut Lfm3Kernels,
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        dec_bo_y: &mut [XdnaBo],
        dec_bo_kvo: &[XdnaBo],
        dec_bo_yn: &XdnaBo,
        stalled_runs: &mut Vec<XdnaRun>,
        pos: usize,
        is_causal: bool,
        eps: f32,
        swa_bidx: usize,
        num_full: usize,
        fin_blk: usize,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<(), String> {
        for (l, layer) in layers.iter().enumerate() {
            let (cur_bo_y, y_next) = if l + 1 < dec_bo_y.len() {
                let (head, tail) = dec_bo_y.split_at_mut(l + 1);
                (&mut head[l], &tail[0])
            } else {
                (&mut dec_bo_y[l], dec_bo_yn)
            };
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&dec_bo_kvo[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &dec_bo_kvo[l])
                };
                let run = create_lfm3_run(
                    kernels.dec_conv()?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        kv_in,
                        kv_out,
                        &layer_kv_blocks[l][0],
                    ],
                    LFM3_DEC_BO_ARGS,
                    "ext_dec_conv",
                )
                .map_err(|e| format!("layer {l} ext dec conv create_run: {e}"))?;
                exec_owned_run(stalled_runs, run, &format!("layer {l} ext dec conv exec"))?;
            } else if layer.swa_window > 0 {
                let run = create_lfm3_run(
                    kernels.dec_attn(swa_bidx)?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        &layer_kv_blocks[l][0],
                        &dec_bo_kvo[l],
                        &layer_kv_blocks[l][0],
                    ],
                    LFM3_DEC_BO_ARGS,
                    "ext_dec_swa",
                )
                .map_err(|e| format!("layer {l} ext dec swa create_run: {e}"))?;
                exec_owned_run(stalled_runs, run, &format!("layer {l} ext dec swa exec"))?;
            } else {
                let run_a = create_lfm3_run(
                    kernels.dec_phase_a()?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        &layer_kv_blocks[l][0],
                        &dec_bo_kvo[l],
                        &layer_kv_blocks[l][0],
                    ],
                    LFM3_DEC_BO_ARGS,
                    "ext_dec_phase_a",
                )
                .map_err(|e| format!("layer {l} ext dec phase_a create_run: {e}"))?;
                exec_owned_run(
                    stalled_runs,
                    run_a,
                    &format!("layer {l} ext dec phase_a exec"),
                )?;

                let k_part = kernels.dec_fa_part()?;
                for (blk, kv_blk) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                    write_lfm3_params_at(
                        cur_bo_y.as_mut_slice(),
                        LFM3_DEC_SLOT_NORM_1 + 4096,
                        pos,
                        1,
                        LFM3_MAX_KV_CHUNKS,
                        pos,
                        is_causal,
                        0,
                        3,
                        eps,
                        blk * LFM3_MAX_KV_CHUNKS,
                        true,
                        false,
                    );
                    cur_bo_y
                        .sync_range_to_device(STEP_PARAMS_BYTES, DEC_NORM1_PARAMS_BYTE_OFF)
                        .map_err(|e| format!("layer {l} blk {blk} dec fa_part params sync: {e}"))?;
                    let run_p = create_lfm3_run(
                        k_part,
                        &[
                            &layer_bo_w[l],
                            &*cur_bo_y,
                            y_next,
                            kv_blk,
                            &dec_bo_kvo[l],
                            &layer_kv_blocks[l][0],
                        ],
                        LFM3_DEC_BO_ARGS,
                        "ext_dec_fa_part",
                    )
                    .map_err(|e| format!("layer {l} blk {blk} ext dec fa_part create_run: {e}"))?;
                    exec_owned_run(
                        stalled_runs,
                        run_p,
                        &format!("layer {l} blk {blk} ext dec fa_part exec"),
                    )?;
                }

                let run_f = create_lfm3_run(
                    kernels.dec_fa_fin(fin_bidx)?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        &layer_kv_blocks[l][fin_blk],
                        &dec_bo_kvo[l],
                        &layer_kv_blocks[l][0],
                    ],
                    LFM3_DEC_BO_ARGS,
                    "ext_dec_fa_fin",
                )
                .map_err(|e| format!("layer {l} ext dec fa_fin create_run: {e}"))?;
                exec_owned_run(
                    stalled_runs,
                    run_f,
                    &format!("layer {l} ext dec fa_fin exec"),
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefill_ext_multi_block_step(
        layers: &[Lfm3LayerStatic],
        kernels: &mut Lfm3Kernels,
        layer_bo_w: &[XdnaBo],
        layer_kv_blocks: &[Vec<XdnaBo>],
        pre_bo_y_layers: &mut [XdnaBo],
        pre_bo_kvo_layers: &[XdnaBo],
        pre_bo_yn: &XdnaBo,
        stalled_runs: &mut Vec<XdnaRun>,
        pos0: usize,
        n_valid: usize,
        eps: f32,
        swa_bidx: usize,
        num_full: usize,
        fin_blk: usize,
        fin_bidx: usize,
        conv_kv_in_kvo: bool,
    ) -> Result<(), String> {
        for (l, layer) in layers.iter().enumerate() {
            let (cur_bo_y, y_next) = if l + 1 < pre_bo_y_layers.len() {
                let (head, tail) = pre_bo_y_layers.split_at_mut(l + 1);
                (&mut head[l], &tail[0])
            } else {
                (&mut pre_bo_y_layers[l], pre_bo_yn)
            };
            if layer.is_conv {
                let (kv_in, kv_out) = if conv_kv_in_kvo {
                    (&pre_bo_kvo_layers[l], &layer_kv_blocks[l][0])
                } else {
                    (&layer_kv_blocks[l][0], &pre_bo_kvo_layers[l])
                };
                let run = create_lfm3_run(
                    &kernels.prefill_conv,
                    &[&layer_bo_w[l], &*cur_bo_y, y_next, kv_in, kv_out],
                    LFM3_PREFILL_BO_ARGS,
                    "ext_prefill_conv",
                )
                .map_err(|e| format!("layer {l} ext prefill conv create_run: {e}"))?;
                exec_owned_run(
                    stalled_runs,
                    run,
                    &format!("layer {l} ext prefill conv exec"),
                )?;
            } else if layer.swa_window > 0 {
                let run = create_lfm3_run(
                    kernels.prefill_attn(swa_bidx)?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        &layer_kv_blocks[l][0],
                        &pre_bo_kvo_layers[l],
                    ],
                    LFM3_PREFILL_BO_ARGS,
                    "ext_prefill_swa",
                )
                .map_err(|e| format!("layer {l} ext prefill swa create_run: {e}"))?;
                exec_owned_run(
                    stalled_runs,
                    run,
                    &format!("layer {l} ext prefill swa exec"),
                )?;
            } else {
                let run_a = create_lfm3_run(
                    kernels.prefill_bidir_a()?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        &layer_kv_blocks[l][0],
                        &pre_bo_kvo_layers[l],
                    ],
                    LFM3_PREFILL_BO_ARGS,
                    "ext_prefill_phase_a",
                )
                .map_err(|e| format!("layer {l} ext prefill phase_a create_run: {e}"))?;
                exec_owned_run(
                    stalled_runs,
                    run_a,
                    &format!("layer {l} ext prefill phase_a exec"),
                )?;

                for (blk, kv_blk) in layer_kv_blocks[l].iter().enumerate().take(num_full) {
                    if blk >= 1 {
                        write_lfm3_params_at(
                            cur_bo_y.as_mut_slice(),
                            LFM3_PREFILL_SLOT_NORM_2 + 2048,
                            pos0,
                            n_valid,
                            LFM3_MAX_KV_CHUNKS,
                            pos0,
                            true,
                            0,
                            4,
                            eps,
                            blk * LFM3_MAX_KV_CHUNKS,
                            true,
                            false,
                        );
                        cur_bo_y
                            .sync_range_to_device(STEP_PARAMS_BYTES, PREFILL_NORM2_PARAMS_BYTE_OFF)
                            .map_err(|e| {
                                format!("layer {l} blk {blk} prefill fa_part params sync: {e}")
                            })?;
                    }
                    let (k_part, part_kind) = if blk == 0 {
                        (kernels.prefill_fa_part0()?, "ext_prefill_fa_part0")
                    } else {
                        (kernels.prefill_fa_part()?, "ext_prefill_fa_part")
                    };
                    let run_p = create_lfm3_run(
                        k_part,
                        &[
                            &layer_bo_w[l],
                            &*cur_bo_y,
                            y_next,
                            kv_blk,
                            &pre_bo_kvo_layers[l],
                        ],
                        LFM3_PREFILL_BO_ARGS,
                        part_kind,
                    )
                    .map_err(|e| {
                        format!("layer {l} blk {blk} ext prefill fa_part create_run: {e}")
                    })?;
                    exec_owned_run(
                        stalled_runs,
                        run_p,
                        &format!("layer {l} blk {blk} ext prefill fa_part exec"),
                    )?;
                }

                let run_f = create_lfm3_run(
                    kernels.prefill_fa_fin(fin_bidx)?,
                    &[
                        &layer_bo_w[l],
                        &*cur_bo_y,
                        y_next,
                        &layer_kv_blocks[l][fin_blk],
                        &pre_bo_kvo_layers[l],
                    ],
                    LFM3_PREFILL_BO_ARGS,
                    "ext_prefill_fa_fin",
                )
                .map_err(|e| format!("layer {l} ext prefill fa_fin create_run: {e}"))?;
                exec_owned_run(
                    stalled_runs,
                    run_f,
                    &format!("layer {l} ext prefill fa_fin exec"),
                )?;
            }
        }
        Ok(())
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
        self.run_npu_decode_embedding_step_inner(npu, embedding, pos, state, true, logits_out)
    }

    fn run_npu_decode_embedding_step_inner(
        &self,
        npu: &mut XdnaLfm3State,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
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
            try_select_lfm3_kv_bucket(&attn_bucket_caps, total_past_chunks)?
        } else {
            (0, 0)
        };
        let (num_full, rem) = (
            total_past_chunks / LFM3_MAX_KV_CHUNKS,
            total_past_chunks % LFM3_MAX_KV_CHUNKS,
        );
        let (fin_bidx, fin_cap) = if !use_fast_plan {
            let fin_bucket_caps: Vec<usize> = Q4LAYER_LFM3_FA_FIN_BUCKETS
                .iter()
                .map(|&(c, _)| c)
                .collect();
            try_select_lfm3_kv_bucket(&fin_bucket_caps, rem)?
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
            ref mut head_q8_scratch,
            ref mut force_conv_sync,
            ref mut in_flight,
            ref mut stalled_runs,
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
        // Set up per-layer step parameters in `dec_bo_y` above, then execute either
        // a cached RunList (<=16K tokens) or synchronous per-block runs (>=16K tokens).
        let step_params_byte_off = (LFM3_DEC_SLOT_NORM_0 + 4096) * 4;

        let mut swa_bidx = attn_bidx;
        let mut seen_swa: Option<(usize, usize, usize)> = None;
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
                        try_select_lfm3_kv_bucket(&attn_bucket_caps, local_past)?
                    };
                    verify_uniform_swa_step(
                        &mut seen_swa,
                        layer.swa_window,
                        swa_base_chunk[l],
                        sb,
                        l,
                    )?;
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
                    if num_full == 1 {
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
                STEP_PARAMS_BYTES
            } else {
                (16 + LFM3_HEAD_DIM) * 4
            };
            dec_bo_y[l]
                .sync_range_to_device(sync_bytes, step_params_byte_off)
                .map_err(|e| format!("layer {l} step_params sync: {e}"))?;
            if sync_ext_part {
                dec_bo_y[l]
                    .sync_range_to_device(STEP_PARAMS_BYTES, DEC_NORM1_PARAMS_BYTE_OFF)
                    .map_err(|e| format!("layer {l} fa_part step_params sync: {e}"))?;
            }
            if sync_ext_fin {
                dec_bo_y[l]
                    .sync_range_to_device(STEP_PARAMS_BYTES, DEC_FIN_PARAMS_BYTE_OFF)
                    .map_err(|e| format!("layer {l} fa_fin step_params sync: {e}"))?;
            }
        }

        let t_prep_us = t_prep0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        let t_npu0 = prof.then(crate::time::Instant::now);
        let mut pending_next = if use_fast_plan || num_full <= 1 {
            let (plan, active_run) = if use_fast_plan {
                let key = (attn_bidx, swa_bidx);
                let p = dec_bucket_plans.get_or_insert_with(parity, key, || {
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
                })?;
                (p, InFlightLfm3Run::DecFast { parity, key })
            } else {
                let key = (swa_bidx, num_full, has_rem, fin_bidx);
                let p = dec_ext_plans.get_or_insert_with(parity, key, || {
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
                })?;
                (p, InFlightLfm3Run::DecExt { parity, key })
            };

            *in_flight = Some(active_run);
            if let Err(e) = plan.runlist.execute() {
                *in_flight = None;
                return Err(format!("dec runlist execute: {e}"));
            }
            let next =
                is_causal.then(|| pending_kv.prepare(pos, CONV_TAPS_STORED, LFM3_H, LFM3_KV_DIM));
            if !pending_kv.flush_into(state) {
                *force_conv_sync = true;
                swa_base_chunk.fill(0);
            }
            plan.runlist
                .wait(QUIESCE_TIMEOUT_MS)
                .map_err(|e| format!("dec runlist wait: {e}"))?;
            *in_flight = None;
            next
        } else {
            let next =
                is_causal.then(|| pending_kv.prepare(pos, CONV_TAPS_STORED, LFM3_H, LFM3_KV_DIM));
            if !pending_kv.flush_into(state) {
                *force_conv_sync = true;
                swa_base_chunk.fill(0);
            }
            Self::run_decode_ext_multi_block_step(
                &self.layers,
                kernels,
                layer_bo_w,
                layer_kv_blocks,
                dec_bo_y,
                dec_bo_kvo,
                dec_bo_yn,
                stalled_runs,
                pos,
                is_causal,
                eps,
                swa_bidx,
                num_full,
                if has_rem { num_full } else { 0 },
                fin_bidx,
                parity,
            )?;
            next
        };
        let t_npu_us = t_npu0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        let mut hidden = [0.0f32; LFM3_H];
        dec.conv_kv_in_kvo = !parity;

        let t_kv0 = prof.then(crate::time::Instant::now);
        if need_logits {
            dec.bo_yn
                .sync_range_from_device(LFM3_DEC_YN_BYTES, 0)
                .map_err(|e| format!("dec_bo_yn sync_from_device: {e}"))?;
            read_le_f32_slice(dec.bo_yn.as_slice(), 0, &mut hidden);
        }

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
        if pos + 1 >= cfg.max_seq_len.min(LFM3_MAX_TOKENS) && !npu.pending_kv.flush_into(state) {
            npu.synced_len = 0;
            npu.force_conv_sync = true;
            npu.swa_base_chunk.fill(0);
        }

        let t_head0 = prof.then(crate::time::Instant::now);
        if need_logits {
            self.cpu.xdna_project_logits_into_with_scratch(
                &hidden,
                logits_out,
                Some((&mut head_q8_scratch.0, &mut head_q8_scratch.1)),
            );
        }
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

    /// Embed token IDs on CPU and execute one NPU prefill pass, computing final
    /// logits only when `need_logits` is true.
    fn run_npu_prefill_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        need_logits: bool,
    ) -> Vec<f32> {
        let vocab_size = self.cpu.config().vocab_size;
        let out_len = if need_logits { vocab_size } else { 0 };
        if tokens.is_empty() {
            return self.reject_input("forward_prefill requires non-empty tokens".into(), out_len);
        }
        let Some(embs) = self.cpu.embed_token_rows(tokens) else {
            return self.reject_input(
                format!("token id out of range for vocab {vocab_size}"),
                out_len,
            );
        };
        self.run_npu_prefill_embeddings(&embs, tokens.len(), start_pos, state, need_logits)
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
        let Some(expected_len) = n_tokens.checked_mul(LFM3_H) else {
            return self.reject_input(
                format!("prefill token count {n_tokens} overflows hidden projection size"),
                out_len,
            );
        };
        if embeddings.len() != expected_len {
            return self.reject_input(
                format!(
                    "embeddings length {} != {n_tokens} * {LFM3_H}",
                    embeddings.len()
                ),
                out_len,
            );
        }
        let Some(end_pos) = start_pos.checked_add(n_tokens) else {
            return self.reject_input(
                format!("prefill range {start_pos} + {n_tokens} overflows position indexing"),
                out_len,
            );
        };
        if end_pos > cfg.max_seq_len || end_pos > LFM3_MAX_TOKENS {
            return self.reject_input(
                format!(
                    "prefill range {start_pos}..{end_pos} exceeds context length {}",
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
        if self.state_torn.load(Ordering::SeqCst) {
            return self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                out_len,
            );
        }
        match self.sync_cpu_state_to_npu(&mut npu, state, start_pos) {
            Ok(()) => {}
            Err(SyncErr::Input(e)) => return self.reject_input(e, out_len),
            Err(SyncErr::Device(e)) => return self.record_device_fault(e, out_len),
        }

        // For causal continuation at an unaligned start_pos, step single-token decode until pos % 32 == 0,
        // and bypass B=64 prefill altogether when <= 3 tokens remain (cheaper on latency, zero prefill padding).
        let mut cur_start = 0usize;
        if cfg.is_causal && (!start_pos.is_multiple_of(LFM3_KV_CHUNK_T) || n_tokens <= 3) {
            let align_rem = if start_pos.is_multiple_of(LFM3_KV_CHUNK_T) {
                0
            } else {
                LFM3_KV_CHUNK_T - (start_pos % LFM3_KV_CHUNK_T)
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
                let gpos = start_pos + i;
                let slice = &embeddings[i * LFM3_H..(i + 1) * LFM3_H];
                let step_need_logits = need_logits && (i + 1 == n_tokens);
                if let Err(e) = self.run_npu_decode_embedding_step_inner(
                    &mut npu,
                    slice,
                    gpos,
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
        }

        let pos_aligned = start_pos + cur_start;
        let rem_tokens = n_tokens - cur_start;
        if !npu.pending_kv.flush_into(state) {
            npu.synced_len = 0;
            npu.force_conv_sync = true;
            npu.swa_base_chunk.fill(0);
        }
        if let Err(e) = self.flush_conv_ping_pong_to_kv(&mut npu) {
            return self.record_device_fault(e, out_len);
        }
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
                ref mut in_flight,
                ref mut stalled_runs,
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
            let mut pending_attn_mirrors: Vec<PrefillAttnMirrorEntry> =
                Vec::with_capacity(self.layers.len());

            let chunk_loop_res = (|| -> Result<(), String> {
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
                    pre_bo_y_layers[0]
                        .sync_range_to_device(hidden_sync_bytes, 0)
                        .map_err(|e| format!("prefill c={c} layer 0 hidden sync: {e}"))?;

                    let (attn_bidx, attn_kv_cap) = if use_fast_pre {
                        try_select_lfm3_kv_bucket(&attn_caps, needed_past)?
                    } else {
                        (0, 0)
                    };
                    let (num_full, rem) = (
                        needed_past / LFM3_MAX_KV_CHUNKS,
                        needed_past % LFM3_MAX_KV_CHUNKS,
                    );
                    let (fin_bidx, fin_cap) = if !use_fast_pre {
                        try_select_lfm3_kv_bucket(&attn_caps, rem)?
                    } else {
                        (0, 0)
                    };
                    let has_rem = rem > 0;
                    let mut swa_bidx = attn_bidx;
                    let mut seen_swa: Option<(usize, usize, usize)> = None;

                    for (l, layer) in self.layers.iter().enumerate() {
                        let mut sync_part0 = false;
                        let mut sync_fin = false;
                        if !layer.is_conv && layer.swa_window > 0 {
                            let global_chunk = pos0 / LFM3_KV_CHUNK_T;
                            if global_chunk.saturating_sub(swa_base_chunk[l])
                                >= lfm3_swa_limit_chunks(layer.swa_window)
                                && let Some(pos) =
                                    pending_attn_mirrors.iter().position(|e| e.layer == l)
                            {
                                let entry = pending_attn_mirrors.swap_remove(pos);
                                Self::mirror_attn_prefill_bo_kv_to_state(
                                    &layer_kv_blocks[l],
                                    &mut state.layers[l],
                                    &entry,
                                    use_f16,
                                );
                            }
                            Self::ensure_swa_capacity_for_chunk(
                                &mut layer_kv_blocks[l][0],
                                &mut swa_base_chunk[l],
                                layer.swa_window,
                                global_chunk,
                            )
                            .map_err(|e| format!("prefill c={c} layer {l} swa compact: {e}"))?;
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
                                    let (sb, swa_cap) =
                                        try_select_lfm3_kv_bucket(&attn_caps, local_past)?;
                                    verify_uniform_swa_step(
                                        &mut seen_swa,
                                        layer.swa_window,
                                        swa_base_chunk[l],
                                        sb,
                                        l,
                                    )?;
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
                                    write_lfm3_prefill_ext_params(
                                        yb,
                                        swa_base_chunk[l],
                                        false,
                                        false,
                                    );
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
                                            6,
                                            eps,
                                            0,
                                            false,
                                            false,
                                        );
                                        sync_part0 = true;
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
                        pre_bo_y_layers[l]
                            .sync_range_to_device(
                                PREFILL_STEP_PARAMS_BYTES,
                                PREFILL_STEP_PARAMS_BYTE_OFF,
                            )
                            .map_err(|e| {
                                format!("prefill c={c} layer {l} step_params sync: {e}")
                            })?;
                        if sync_part0 {
                            pre_bo_y_layers[l]
                                .sync_range_to_device(
                                    PREFILL_STEP_PARAMS_BYTES,
                                    PREFILL_NORM1_PARAMS_BYTE_OFF,
                                )
                                .map_err(|e| {
                                    format!("prefill c={c} layer {l} fa_part0 sync: {e}")
                                })?;
                        }
                        if sync_fin {
                            pre_bo_y_layers[l]
                                .sync_range_to_device(
                                    PREFILL_STEP_PARAMS_BYTES,
                                    PREFILL_NORM3_PARAMS_BYTE_OFF,
                                )
                                .map_err(|e| format!("prefill c={c} layer {l} fa_fin sync: {e}"))?;
                        }
                        if !layer.is_conv {
                            pre_bo_y_layers[l]
                                .sync_range_to_device(rope_sync_bytes, PREFILL_ROPE_BYTE_OFF)
                                .map_err(|e| format!("prefill c={c} layer {l} rope sync: {e}"))?;
                        }
                    }

                    if let Some(t) = tc0 {
                        t_prep_us += t.elapsed().as_secs_f64() * 1e6;
                    }
                    let ts0 = prof.then(crate::time::Instant::now);
                    let res = if use_fast_pre || num_full <= 1 {
                        let (plan, active_run) = if use_fast_pre {
                            let key = (attn_bidx, swa_bidx);
                            let p = pre_bucket_plans.get_or_insert_with(parity, key, || {
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
                            })?;
                            (p, InFlightLfm3Run::PreFast { parity, key })
                        } else {
                            let key = (swa_bidx, num_full, has_rem, fin_bidx);
                            let p = pre_ext_plans.get_or_insert_with(parity, key, || {
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
                            })?;
                            (p, InFlightLfm3Run::PreExt { parity, key })
                        };

                        *in_flight = Some(active_run);
                        match plan.runlist.execute() {
                            Ok(()) => {
                                Self::flush_pending_attn_mirrors(
                                    &mut pending_attn_mirrors,
                                    layer_kv_blocks,
                                    &mut state.layers,
                                    use_f16,
                                );
                                let w = plan.runlist.wait(QUIESCE_TIMEOUT_MS);
                                if w.is_ok() {
                                    *in_flight = None;
                                }
                                w
                            }
                            Err(e) => {
                                *in_flight = None;
                                Err(e)
                            }
                        }
                    } else {
                        Self::flush_pending_attn_mirrors(
                            &mut pending_attn_mirrors,
                            layer_kv_blocks,
                            &mut state.layers,
                            use_f16,
                        );
                        Self::run_prefill_ext_multi_block_step(
                            &self.layers,
                            kernels,
                            layer_bo_w,
                            layer_kv_blocks,
                            pre_bo_y_layers,
                            pre_bo_kvo_layers,
                            &*pre_bo_yn,
                            stalled_runs,
                            pos0,
                            n_valid,
                            eps,
                            swa_bidx,
                            num_full,
                            if has_rem { num_full } else { 0 },
                            fin_bidx,
                            parity,
                        )
                    };
                    if let Some(t) = ts0 {
                        t_attn_npu_us += t.elapsed().as_secs_f64() * 1e6;
                    }
                    res.map_err(|e| format!("prefill c={c} runlist execute/wait: {e}"))?;
                    *pre_conv_kv_in_kvo = !parity;

                    let is_last_chunk = c + 1 == n_chunks;
                    let tkv_start = prof.then(crate::time::Instant::now);
                    for (l, layer) in self.layers.iter().enumerate() {
                        if layer.is_conv {
                            if is_last_chunk {
                                let tck0 = prof.then(crate::time::Instant::now);
                                let conv_out_bo = if parity {
                                    &mut layer_kv_blocks[l][0]
                                } else {
                                    &mut pre_bo_kvo_layers[l]
                                };
                                conv_out_bo
                                    .sync_range_from_device(LFM3_KVO_BYTES, 0)
                                    .map_err(|e| {
                                        format!("prefill c={c} layer {l} conv kvo sync: {e}")
                                    })?;
                                Self::save_conv_kvo_to_state(
                                    conv_out_bo,
                                    &mut state.layers[l],
                                    pos0 + n_valid,
                                    true,
                                );
                                if *pre_conv_kv_in_kvo {
                                    let src = &pre_bo_kvo_layers[l].as_slice()[..LFM3_KVO_BYTES];
                                    layer_kv_blocks[l][0].as_mut_slice()[..LFM3_KVO_BYTES]
                                        .copy_from_slice(src);
                                    layer_kv_blocks[l][0]
                                        .sync_range_to_device(LFM3_KVO_BYTES, 0)
                                        .map_err(|e| {
                                            format!("prefill layer {l} final conv bo_kv sync: {e}")
                                        })?;
                                }
                                if let Some(t) = tck0 {
                                    t_conv_kv_us += t.elapsed().as_secs_f64() * 1e6;
                                }
                            }
                        } else {
                            let tak0 = prof.then(crate::time::Instant::now);
                            pre_bo_kvo_layers[l]
                                .sync_range_from_device(attn_kvo_bytes, 0)
                                .map_err(|e| {
                                    format!("prefill c={c} layer {l} attn kvo sync: {e}")
                                })?;
                            let entry = Self::save_attn_prefill_kvo_to_bo_kv(
                                &pre_bo_kvo_layers[l],
                                &mut layer_kv_blocks[l],
                                &kernels.prefill_conv,
                                &mut swa_base_chunk[l],
                                layer.swa_window,
                                l,
                                pos0,
                                n_valid,
                            )
                            .map_err(|e| format!("prefill c={c} layer {l} attn kvo save: {e}"))?;
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
                            pre_bo_yn
                                .sync_range_from_device(last_row_bytes, last_row_off)
                                .map_err(|e| {
                                    format!("prefill last_hidden sync_from_device: {e}")
                                })?;
                            read_le_f32_slice(
                                pre_bo_yn.as_slice(),
                                (n_valid - 1) * LFM3_H,
                                &mut last_hidden,
                            );
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = chunk_loop_res {
                Self::flush_pending_attn_mirrors(
                    &mut pending_attn_mirrors,
                    layer_kv_blocks,
                    &mut state.layers,
                    use_f16,
                );
                return self.record_device_fault(e, out_len);
            }

            state.seq_len = total_tokens;
            npu.synced_len = total_tokens;
            let t_head0 = prof.then(crate::time::Instant::now);
            let out_logits = if need_logits {
                let head_scratch = &mut npu.head_q8_scratch;
                self.cpu.xdna_project_logits_from_hidden_with_scratch(
                    &last_hidden,
                    Some((&mut head_scratch.0, &mut head_scratch.1)),
                )
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
                        write_f32_le_fast(
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
                        ref mut stalled_runs,
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
                            stalled_runs,
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
                    let (bidx, bidir_cap) =
                        match try_select_lfm3_kv_bucket(&bidir_caps, n_kv_chunks) {
                            Ok(v) => v,
                            Err(e) => return self.record_device_fault(e, out_len),
                        };
                    for (c, saved) in phase_a_saved.iter().enumerate() {
                        let pos0 = c * LFM3_PREFILL_B;
                        let n_valid = (total_tokens - pos0).min(LFM3_PREFILL_B);
                        {
                            let yb = npu.pre_bo_y.as_mut_slice();
                            yb.fill(0);
                            write_f32_le_fast(
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
                            ref mut stalled_runs,
                            ..
                        } = *npu;
                        let res = kernels.prefill_bidir_b(bidx).and_then(|k| {
                            Self::run_prefill_single_chunk(
                                k,
                                stalled_runs,
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
                        read_f32_le_fast(
                            npu.pre_bo_yn.as_slice(),
                            0,
                            &mut hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                        );
                    }
                } else {
                    let num_blocks = n_kv_chunks.div_ceil(LFM3_MAX_KV_CHUNKS);
                    let mut attn_out = vec![0.0f32; LFM3_PREFILL_B * LFM3_H];
                    let mut block_out = vec![0.0f32; LFM3_PREFILL_B * LFM3_H];
                    let mut bq_scales = Vec::new();
                    let mut bq_quants = Vec::new();
                    #[cfg(has_blas)]
                    let mut dequant_scratch = Vec::new();
                    let mut moe_scratch = crate::model::lfm2::XdnaMoePrefillScratch::default();
                    for (c, saved) in phase_a_saved.iter().enumerate() {
                        let pos0 = c * LFM3_PREFILL_B;
                        let n_valid = (total_tokens - pos0).min(LFM3_PREFILL_B);
                        {
                            let yb = npu.pre_bo_y.as_mut_slice();
                            yb.fill(0);
                            yb[LFM3_PREFILL_SLOT_Q_BF16 * 4..][..PHASE_A_Q_BYTES]
                                .copy_from_slice(&saved[..PHASE_A_Q_BYTES]);
                        }
                        if let Err(e) = npu
                            .pre_bo_y
                            .sync_range_to_device(PHASE_A_Q_BYTES, LFM3_PREFILL_SLOT_Q_BF16 * 4)
                        {
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
                            ref mut stalled_runs,
                            ..
                        } = *npu;
                        if let Err(e) = Self::run_prefill_bidir_paged_fa(
                            kernels,
                            &layer_bo_w[l],
                            pre_bo_y,
                            pre_bo_yn,
                            &layer_kv_blocks[l],
                            pre_bo_kvo,
                            stalled_runs,
                            l,
                            c,
                            pos0,
                            n_valid,
                            total_tokens,
                            num_blocks,
                            eps,
                        ) {
                            return self.record_device_fault(e, out_len);
                        }
                        if self.should_run_bidir_host_ffn(c)
                            && let Err(e) = self.finalize_bidir_fa_chunk_and_ffn(
                                l,
                                npu.pre_bo_y.as_slice(),
                                &saved[PHASE_A_Q_BYTES..],
                                n_valid,
                                eps,
                                &mut hidden_all[pos0 * LFM3_H..(pos0 + n_valid) * LFM3_H],
                                &mut attn_out,
                                &mut block_out,
                                &mut bq_scales,
                                &mut bq_quants,
                                #[cfg(has_blas)]
                                &mut dequant_scratch,
                                &mut moe_scratch,
                                state,
                            )
                        {
                            return self.record_device_fault(e, out_len);
                        }
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
                            write_f32_le_fast(yb, 0, chunk_slice);
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
                            ref mut stalled_runs,
                            ..
                        } = *npu;
                        if let Err(e) = Self::run_prefill_single_chunk(
                            &kernels.prefill_conv,
                            stalled_runs,
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
                        read_f32_le_fast(npu.pre_bo_yn.as_slice(), 0, chunk_slice);
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
                            let (bidx, kv_cap) = match try_select_lfm3_kv_bucket(
                                &attn_caps,
                                needed_past.saturating_sub(kv_base),
                            ) {
                                Ok(v) => v,
                                Err(e) => return self.record_device_fault(e, out_len),
                            };
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
                                ref mut stalled_runs,
                                ..
                            } = *npu;
                            let res = kernels.prefill_attn(bidx).and_then(|k| {
                                Self::run_prefill_single_chunk(
                                    k,
                                    stalled_runs,
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
                                    ref mut stalled_runs,
                                    ..
                                } = *npu;
                                kernels.prefill_bidir_a().and_then(|k| {
                                    Self::run_prefill_single_chunk(
                                        k,
                                        stalled_runs,
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
                                ref mut stalled_runs,
                                ..
                            } = *npu;
                            let (fin_bidx, fin_cap) =
                                match try_select_lfm3_kv_bucket(&attn_caps, rem) {
                                    Ok(v) => v,
                                    Err(e) => return self.record_device_fault(e, out_len),
                                };
                            let fin_blk = if rem > 0 { num_full } else { 0 };
                            if let Err(e) = Self::run_prefill_single_layer_paged_fa(
                                kernels,
                                &layer_bo_w[l],
                                pre_bo_y,
                                pre_bo_yn,
                                &layer_kv_blocks[l],
                                pre_bo_kvo,
                                stalled_runs,
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
        let head_scratch = &mut npu.head_q8_scratch;
        self.cpu.xdna_project_logits_from_hidden_with_scratch(
            last_hidden,
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
        if self.state_torn.load(Ordering::SeqCst) {
            return self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                vocab_size,
            );
        }
        match self.sync_cpu_state_to_npu(&mut npu, state, pos) {
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

impl Model for XdnaLfm3Model {
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

    fn supports_bidirectional_hidden_states(&self) -> bool {
        self.cpu.supports_bidirectional_hidden_states()
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
        let mut npu = self.lock_npu();
        let tok = if self.state_torn.load(Ordering::SeqCst) {
            let _ = self.reject_input_err(
                CeraError::Backend(
                    "XDNA NPU state torn after a fault; call try_reset_kv".to_string(),
                ),
                vocab_size,
            );
            0
        } else if let Err(err) = self.sync_cpu_state_to_npu(&mut npu, state, pos) {
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
        self.run_npu_prefill_tokens(tokens, start_pos, state, true)
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
            let need_logits = pos + chunk.len() == start_pos + tokens.len();
            let logits = self.run_npu_prefill_tokens(chunk, pos, state, need_logits);
            if self
                .decode_error
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .is_some()
            {
                (0, need_logits.then_some(logits))
            } else {
                (chunk.len(), need_logits.then_some(logits))
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
        // SWA ring base offsets cannot stay valid once tokens are shifted out of prefix order.
        npu.synced_len = 0;
        npu.swa_base_chunk.fill(0);
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
        let _ = self.apply_truncate_rewind_sync(&mut npu, state, len, |st| {
            self.cpu.truncate_kv(st, len);
            Ok(())
        });
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
        self.apply_truncate_rewind_sync(&mut npu, state, len, |st| {
            self.cpu.try_truncate_kv(st, len)
        })
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
    use crate::backend::xdna::sys::fake::{Config as FakeConfig, Event, with_shim};
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
        tiny_lfm3_gguf_with_ctx(is_causal, 128)
    }

    fn tiny_lfm3_gguf_with_ctx(is_causal: bool, ctx_len: u32) -> GgufFile {
        tiny_lfm3_gguf_with_swa(is_causal, ctx_len, None)
    }

    fn tiny_lfm3_gguf_with_swa(is_causal: bool, ctx_len: u32, swa_window: Option<u32>) -> GgufFile {
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
            .kv_u32("lfm3.context_length", ctx_len)
            .kv_f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("lfm3.shortconv.l_cache", LFM3_CONV_TAP as u32)
            .kv_bool("lfm3.attention.causal", is_causal);
        if !is_causal {
            b = b.kv_str("lfm3.decision.type", "lfm3-d1");
        }
        if let Some(win) = swa_window {
            b = b.kv_u32("lfm3.attention.sliding_window", win);
        }
        let b = b
            .kv_f32("lfm3.rope.freq_base", 1_000_000.0)
            .kv_raw(
                "lfm3.attention.sliding_window_pattern",
                9,
                bool_array_raw(&[false, swa_window.is_some()]),
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

    #[cfg(feature = "mmap")]
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

    #[cfg(feature = "mmap")]
    fn argmax_idx(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[cfg(feature = "mmap")]
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

    #[cfg(feature = "mmap")]
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

    #[cfg(feature = "mmap")]
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

    #[test]
    fn causal_lfm3_prefill_multi_chunk_unaligned_and_no_logits_on_fake_shim() {
        let (_, events) = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_gguf(true);
            let model = crate::model::load_model_xdna(gguf, None, 256)
                .expect("load_model_xdna causal LFM3");
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                256,
            )
            .unwrap();

            // `n_tokens = 8` (> 3) takes the B=64 causal prefill runlist path.
            let logits8 = model.forward_prefill(&[1, 2, 3, 4, 5, 6, 7, 8], 0, &mut state);
            assert_eq!(logits8.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 8);
            assert!(model.take_decode_error().is_none());

            // Unaligned continuation (`start_pos = 30`, `n_tokens = 6`): align 2 decode steps to 32, then 4-token prefill.
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 256)
                .unwrap();
            let prompt30: Vec<u32> = (0..30).map(|i| (i % 16) as u32).collect();
            let _ = model.forward_prefill(&prompt30, 0, &mut state);
            assert_eq!(state.seq_len, 30);
            let cont6 = model.forward_prefill(&[1, 2, 3, 4, 5, 6], 30, &mut state);
            assert_eq!(cont6.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 36);
            assert!(model.take_decode_error().is_none());

            // Multi-chunk causal prefill (`n_tokens = 70 > 64`) with `need_logits = false`.
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 256)
                .unwrap();
            let prompt70: Vec<u32> = (0..70).map(|i| (i % 16) as u32).collect();
            let embs70 = model.embed_token_rows(&prompt70).unwrap();
            let empty =
                model.forward_prefill_from_embeddings_chunk(&embs70, 70, 0, &mut state, false);
            assert!(empty.is_empty());
            assert_eq!(state.seq_len, 70);
            assert!(model.take_decode_error().is_none());
        });
        assert!(
            events.iter().any(|e| matches!(e, Event::RunListExecute(_))),
            "expected RunListExecute events for causal B=64 prefill"
        );
    }

    #[test]
    fn causal_lfm3_forward_greedy_fault_isolation_poison_and_session_reset_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_gguf(true);
            let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 128, "test".to_string())
                .expect("load_model_xdna causal LFM3");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                128,
            )
            .unwrap();

            // Normal `forward_greedy` keeps caller scratch allocation and updates `seq_len`.
            state.scratch.logits.reserve(model.config().vocab_size);
            let scratch_ptr = state.scratch.logits.as_ptr();
            let t0 = model.forward_greedy(&[1], 0, &mut state);
            assert!((t0 as usize) < model.config().vocab_size);
            assert_eq!(state.seq_len, 1);
            assert_eq!(state.scratch.logits.as_ptr(), scratch_ptr);
            assert!(model.take_decode_error().is_none());

            // Pre-execution validation failure inside `forward_greedy` (`pos` mismatch vs `state`) rejects without tearing.
            let bad = model.forward_greedy(&[1], 5, &mut state);
            assert_eq!(bad, 0);
            assert!(!npu_model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());

            // `check_kv_rewind` handles pending deferred KV row (`len=0` valid, `len=2` OOB).
            assert_eq!(model.check_kv_rewind(&state, 0), Ok(()));
            assert!(matches!(
                model.check_kv_rewind(&state, 2),
                Err(KvRewindError::OutOfBounds {
                    requested: 2,
                    current: 1,
                })
            ));

            // Reset, then run 4-token aligned prefill (`>= 4`) so `npu.prefill` allocates, flip `pre.conv_kv_in_kvo`,
            // and verify `try_reset_kv` clears both decode and prefill parity flags.
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 128)
                .unwrap();
            let _ = model.forward_prefill(&[1, 2, 3, 4], 0, &mut state);
            {
                let mut guard = npu_model.lock_npu();
                if let Some(pre) = guard.prefill.as_mut() {
                    pre.conv_kv_in_kvo = true;
                }
            }
            model
                .try_reset_kv(&mut state, &crate::kv_cache::KvCompression::F16, 128)
                .unwrap();
            {
                let guard = npu_model.lock_npu();
                assert!(!guard.prefill.as_ref().unwrap().conv_kv_in_kvo);
                if let Some(dec) = guard.decode.as_ref() {
                    assert!(!dec.conv_kv_in_kvo);
                }
            }

            // Poisoning `npu` mutex tears state on next `lock_npu` (`forward_greedy` refuses fail-open).
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = npu_model.npu.lock().unwrap();
                panic!("simulated panic in XdnaLfm3Model");
            }));
            let tok_torn = model.forward_greedy(&[1], 0, &mut state);
            assert_eq!(tok_torn, 0);
            assert!(npu_model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());

            // `acquire_session` quiesces, resets torn/stale state cleanly.
            let _lease = model
                .acquire_session()
                .expect("acquire_session clears torn state for new session");
            assert!(!npu_model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_none());
        });
    }

    #[test]
    fn causal_lfm3_ext_multi_block_decode_and_prefill_on_fake_shim() {
        let (_, events) = with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_gguf_with_ctx(true, 32768);
            let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 16512, "test".to_string())
                .expect("load_model_xdna causal LFM3 at >16K context");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                16512,
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
                    key_cache_f16.resize(pos * LFM3_KV_DIM, 0);
                    value_cache_f16.resize(pos * LFM3_KV_DIM, 0);
                    // Fill selected boundaries and interior offsets with non-uniform K/V values.
                    for base in [
                        0usize,
                        32,
                        pos.saturating_sub(64),
                        pos.saturating_sub(32),
                        pos.saturating_sub(1),
                    ] {
                        if base < pos {
                            for kvh in 0..LFM3_N_KV_HEADS {
                                for d in [0usize, 17, LFM3_HEAD_DIM - 1] {
                                    let idx = base * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM + d;
                                    let seed = ((base + kvh * 13 + d * 3) % 89) as f32 * 0.0625;
                                    key_cache_f16[idx] = crate::quant::f32_to_f16(1.0 + seed);
                                    value_cache_f16[idx] = crate::quant::f32_to_f16(-1.5 - seed);
                                }
                            }
                        }
                    }
                }
                if let LayerState::Conv { buffer, .. } = &mut st.layers[0] {
                    buffer.resize(CONV_TAPS_STORED * LFM3_H, 0.25);
                }
                st.seq_len = pos;
            };

            // Fast-bucket decode past 8K (`pos=8192`, `pos=8224` use `Q4LAYER_LFM3_KV_BUCKETS` up to `c511`).
            let emb = model.embed_token_rows(&[1]).unwrap();
            seed_state_to_pos(&mut state, 8192);
            let dec_1a = model.forward_from_embedding(&emb, 8192, &mut state);
            assert_eq!(dec_1a.len(), vocab);
            assert_eq!(state.seq_len, 8193);
            assert!(model.take_decode_error().is_none());

            seed_state_to_pos(&mut state, 8224);
            let dec_1b = model.forward_from_embedding(&emb, 8224, &mut state);
            assert_eq!(dec_1b.len(), vocab);
            assert_eq!(state.seq_len, 8225);
            assert!(model.take_decode_error().is_none());

            // Multi-block decode at `pos=16417` (`num_full=2, has_rem=true`): block 1 must write `kv_start_chunk=256`.
            seed_state_to_pos(&mut state, 16417);
            let dec_out = model.forward_from_embedding(&emb, 16417, &mut state);
            assert_eq!(dec_out.len(), vocab);
            assert_eq!(state.seq_len, 16418);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                assert!(
                    guard.layer_kv_blocks[1].len() >= 3,
                    "expected 3 KV blocks for 16417 tokens, got {}",
                    guard.layer_kv_blocks[1].len()
                );
                let yb = guard.decode.as_ref().unwrap().layer_bo_y[1].as_slice();
                let off = DEC_NORM1_PARAMS_BYTE_OFF + 8 * 4;
                let kv_start = i32::from_le_bytes(yb[off..off + 4].try_into().unwrap());
                assert_eq!(kv_start, LFM3_MAX_KV_CHUNKS as i32);
            }

            // Multi-block aligned causal prefill at `pos0=16448` (`num_full=2, has_rem=true`) via real sync.
            let embs4 = model.embed_token_rows(&[1, 2, 3, 4]).unwrap();
            seed_state_to_pos(&mut state, 16448);
            let pre_out =
                model.forward_prefill_from_embeddings_chunk(&embs4, 4, 16448, &mut state, true);
            assert_eq!(pre_out.len(), vocab);
            assert_eq!(state.seq_len, 16452);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                let yb = guard.prefill.as_ref().unwrap().layer_bo_y[1].as_slice();
                let off = (LFM3_PREFILL_SLOT_NORM_2 + 2048 + 8) * 4;
                let kv_start = i32::from_le_bytes(yb[off..off + 4].try_into().unwrap());
                assert_eq!(kv_start, LFM3_MAX_KV_CHUNKS as i32);
                let blk1_off = 2 * LFM3_MAX_KV_CHUNKS * TILE_BYTES + 17 * 2;
                let k_read = u16::from_le_bytes(
                    guard.layer_kv_blocks[1][1].as_slice()[blk1_off..blk1_off + 2]
                        .try_into()
                        .unwrap(),
                );
                let seed = ((8192 + 2 * 13 + 17 * 3) % 89) as f32 * 0.0625;
                let expected_k = crate::backend::xdna::repack::f32_to_bf16(
                    crate::quant::f16_to_f32(crate::quant::f32_to_f16(1.0 + seed)),
                );
                assert_eq!(
                    k_read, expected_k,
                    "multi-block sync must populate block 1 KV"
                );
                // Block 2 stores positions >= 16384 (`pos=16447` is chunk 513 -> blk 2, loc_c 1, rem 31).
                let chk2_loc1 =
                    (2 * LFM3_MAX_KV_CHUNKS + 1) * TILE_BYTES + 31 * LFM3_HEAD_DIM * 2 + 17 * 2;
                let v_read_b2 = u16::from_le_bytes(
                    guard.layer_kv_blocks[1][2].as_slice()[chk2_loc1 + ATTN_KV_CHUNK_V_OFFSET
                        ..chk2_loc1 + ATTN_KV_CHUNK_V_OFFSET + 2]
                        .try_into()
                        .unwrap(),
                );
                let seed2 = ((16447 + 2 * 13 + 17 * 3) % 89) as f32 * 0.0625;
                let expected_v2 = crate::backend::xdna::repack::f32_to_bf16(
                    crate::quant::f16_to_f32(crate::quant::f32_to_f16(-1.5 - seed2)),
                );
                assert_eq!(
                    v_read_b2, expected_v2,
                    "multi-block sync must populate block 2 V"
                );
            }

            // Aligned >8K causal prefill (`pos0=8192`, `num_full=1`) exercises `build_prefill_ext_plan`.
            seed_state_to_pos(&mut state, 8192);
            let pre_1 =
                model.forward_prefill_from_embeddings_chunk(&embs4, 4, 8192, &mut state, true);
            assert_eq!(pre_1.len(), vocab);
            assert_eq!(state.seq_len, 8196);
            assert!(model.take_decode_error().is_none());
        });
        let dec_norm1_syncs = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::BoSyncRange {
                        to_device: true,
                        offset,
                        ..
                    } if *offset == DEC_NORM1_PARAMS_BYTE_OFF
                )
            })
            .count();
        assert_eq!(
            dec_norm1_syncs, 2,
            "expected exact 2 DEC_NORM1 step_params syncs (0 for fast 8192/8224 + 2 for 16417)"
        );
        assert!(events.iter().any(|e| matches!(
            e,
            Event::BoSyncRange {
                to_device: true,
                offset,
                ..
            } if *offset == PREFILL_NORM2_PARAMS_BYTE_OFF
        )));
    }

    #[test]
    fn swa_rewind_clears_stale_swa_base_chunk_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_gguf_with_swa(true, 4096, Some(64));
            let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 4096, "test".to_string())
                .expect("load_model_xdna SWA LFM3");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                4096,
            )
            .unwrap();
            let pos = 512usize;
            if let LayerState::Attention {
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[1]
            {
                key_cache_f16.resize(pos * LFM3_KV_DIM, crate::quant::f32_to_f16(1.0));
                value_cache_f16.resize(pos * LFM3_KV_DIM, crate::quant::f32_to_f16(-1.0));
            }
            if let LayerState::Conv { buffer, history } = &mut state.layers[0] {
                buffer.resize(CONV_TAPS_STORED * LFM3_H, 0.25);
                *history = crate::kv_cache::ConvHistory::new(CONV_TAPS_STORED * LFM3_H);
                history.push(128, buffer);
            }
            state.seq_len = pos;
            let emb = model.embed_token_rows(&[1]).unwrap();
            let _ = model.forward_from_embedding(&emb, pos, &mut state);
            assert!(model.take_decode_error().is_none());
            let base_after_first = npu_model.lock_npu().swa_base_chunk[1];
            assert!(
                base_after_first > 0,
                "SWA sync past limit_chunks must advance swa_base_chunk organically, got {base_after_first}"
            );

            // Flush pending CPU KV before growing state so the next sync enters `start = 513 > 0`,
            // and verify preserved BF16 K at `pos = 480` survives intact in the SWA ring.
            assert!(
                npu_model.lock_npu().pending_kv.flush_into(&mut state),
                "pending_kv flush before incremental SWA sync must match state"
            );
            assert_eq!(npu_model.lock_npu().synced_len, 513);
            if let LayerState::Attention {
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[1]
            {
                key_cache_f16.resize(544 * LFM3_KV_DIM, crate::quant::f32_to_f16(2.0));
                value_cache_f16.resize(544 * LFM3_KV_DIM, crate::quant::f32_to_f16(-2.0));
            }
            state.seq_len = 544;
            let _ = model.forward_from_embedding(&emb, 544, &mut state);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                assert!(
                    guard.swa_base_chunk[1] >= base_after_first,
                    "incremental forward SWA sync must not wipe swa_base_chunk[1]"
                );
                let chk480 = 480usize / LFM3_KV_CHUNK_T;
                let loc480 = chk480 - guard.swa_base_chunk[1];
                let k480_off = loc480 * TILE_BYTES;
                let k480 = u16::from_le_bytes(
                    guard.layer_kv_blocks[1][0].as_slice()[k480_off..k480_off + 2]
                        .try_into()
                        .unwrap(),
                );
                assert_eq!(
                    k480,
                    crate::backend::xdna::repack::f32_to_bf16(1.0),
                    "preserved SWA ring entry at pos=480 must not be clobbered on start>0 incremental sync"
                );
            }

            // Safe rewind resets `swa_base_chunk` cleanly and allows continued decoding.
            model.try_truncate_kv(&mut state, 128).expect("SWA rewind");
            {
                let guard = npu_model.lock_npu();
                assert_eq!(guard.swa_base_chunk[1], 0);
                assert_eq!(guard.synced_len, 0);
            }
            let post_rewind = model.forward_from_embedding(&emb, 128, &mut state);
            assert_eq!(post_rewind.len(), model.config().vocab_size);
            assert!(model.take_decode_error().is_none());

            // Infallible `truncate_kv` after advancing SWA must also clear `swa_base_chunk` cleanly.
            {
                let mut guard = npu_model.lock_npu();
                guard.swa_base_chunk[1] = 10;
            }
            model.truncate_kv(&mut state, 0);
            assert_eq!(npu_model.lock_npu().swa_base_chunk[1], 0);
            assert_eq!(state.seq_len, 0);
        });
    }

    #[test]
    fn causal_lfm3_prefill_stalled_run_and_swa_validation_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let max_swa = tiny_lfm3_gguf_with_swa(true, 8192, Some(LFM3_SWA_MAX_WINDOW as u32));
            XdnaLfm3Model::from_gguf_with_id(max_swa, 8192, "max-swa".to_string())
                .expect("inclusive LFM3_SWA_MAX_WINDOW should load cleanly");
            let over_swa =
                tiny_lfm3_gguf_with_swa(true, 8192, Some((LFM3_SWA_MAX_WINDOW + 1) as u32));
            let Err(err) = XdnaLfm3Model::from_gguf_with_id(over_swa, 8192, "bad-swa".to_string())
            else {
                panic!("oversize swa_window should be rejected");
            };
            let full_err = format!("{err:#}");
            assert!(
                full_err.contains("SWA window"),
                "unexpected SWA error: {full_err}"
            );
        });

        with_shim(
            FakeConfig {
                run_wait_state: Some(0),
                ..Default::default()
            },
            || {
                let gguf = tiny_lfm3_gguf(false);
                let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 128, "test".to_string())
                    .expect("load non-causal LFM3 for timeout test");
                let model: &dyn Model = &npu_model;
                let mut state = InferenceState::from_config_capped(
                    model.config(),
                    &crate::kv_cache::KvCompression::F16,
                    128,
                )
                .unwrap();
                let out = model.forward_prefill(&[1, 2, 3, 4], 0, &mut state);
                assert_eq!(out, vec![0.0; model.config().vocab_size]);
                assert!(model.take_decode_error().is_some());
                assert!(npu_model.state_torn.load(Ordering::SeqCst));
                assert!(
                    !npu_model.lock_npu().stalled_runs.is_empty(),
                    "timed-out prefill single-chunk run must move into stalled_runs"
                );
            },
        );
    }

    fn clear_recorded_run_args() {
        RECORDED_RUN_ARGS.with(|r| r.borrow_mut().clear());
    }

    fn take_recorded_run_args() -> Vec<(&'static str, usize)> {
        RECORDED_RUN_ARGS.with(|r| std::mem::take(&mut *r.borrow_mut()))
    }

    fn xclbin_bo_arg_count(xclbin: &[u8]) -> usize {
        const PAT: &[u8] = b"<arg name=\"bo";
        xclbin.windows(PAT.len()).filter(|w| *w == PAT).count()
    }

    fn make_nonzero_q4_0_bytes(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let blocks_per_row = cols / 32;
        let mut out = vec![0u8; rows * blocks_per_row * 18];
        let scale = crate::quant::f32_to_f16(0.0625).to_le_bytes();
        for (bi, blk) in out.as_chunks_mut::<18>().0.iter_mut().enumerate() {
            blk[0..2].copy_from_slice(&scale);
            for (k, byte) in blk[2..18].iter_mut().enumerate() {
                let lo = ((bi * 7 + k * 3 + usize::from(seed)) % 13) + 1;
                let hi = ((bi * 11 + k * 5 + usize::from(seed)) % 13) + 1;
                *byte = ((hi << 4) | lo) as u8;
            }
        }
        out
    }

    fn tiny_lfm3_3layer_swa_gguf(is_causal: bool, ctx_len: u32, swa_window: u32) -> GgufFile {
        let vocab = 32usize;
        let ones_h = vec![1.0f32; LFM3_H];
        let ones_hd = vec![1.0f32; LFM3_HEAD_DIM];
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "lfm3")
            .kv_u32("lfm3.block_count", 3)
            .kv_u32("lfm3.embedding_length", LFM3_H as u32)
            .kv_u32("lfm3.feed_forward_length", LFM3_FFN_SRC as u32)
            .kv_u32("lfm3.attention.head_count", LFM3_N_HEADS as u32)
            .kv_i32_array(
                "lfm3.attention.head_count_kv",
                vec![0, LFM3_N_KV_HEADS as i32, LFM3_N_KV_HEADS as i32],
            )
            .kv_u32("lfm3.attention.key_length", LFM3_HEAD_DIM as u32)
            .kv_u32("lfm3.attention.value_length", LFM3_HEAD_DIM as u32)
            .kv_u32("lfm3.context_length", ctx_len)
            .kv_f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("lfm3.shortconv.l_cache", LFM3_CONV_TAP as u32)
            .kv_bool("lfm3.attention.causal", is_causal)
            .kv_u32("lfm3.attention.sliding_window", swa_window)
            .kv_f32("lfm3.rope.freq_base", 1_000_000.0)
            .kv_raw(
                "lfm3.attention.sliding_window_pattern",
                9,
                bool_array_raw(&[false, true, false]),
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
            );
        for l in 1..=2 {
            b = b
                .tensor_f32(format!("blk.{l}.attn_norm.weight"), &[LFM3_H], &ones_h)
                .tensor_f32(
                    format!("blk.{l}.post_attention_norm.weight"),
                    &[LFM3_H],
                    &ones_h,
                )
                .tensor_f32(format!("blk.{l}.ffn_norm.weight"), &[LFM3_H], &ones_h)
                .tensor_f32(format!("blk.{l}.post_ffw_norm.weight"), &[LFM3_H], &ones_h)
                .tensor(
                    format!("blk.{l}.attn_q.weight"),
                    &[LFM3_H, LFM3_Q_DIM],
                    2,
                    make_q4_0_bytes(LFM3_Q_DIM, LFM3_H),
                )
                .tensor(
                    format!("blk.{l}.attn_k.weight"),
                    &[LFM3_H, LFM3_KV_DIM],
                    2,
                    make_q4_0_bytes(LFM3_KV_DIM, LFM3_H),
                )
                .tensor(
                    format!("blk.{l}.attn_v.weight"),
                    &[LFM3_H, LFM3_KV_DIM],
                    2,
                    make_q4_0_bytes(LFM3_KV_DIM, LFM3_H),
                )
                .tensor(
                    format!("blk.{l}.attn_gate.weight"),
                    &[LFM3_H, LFM3_N_HEADS],
                    2,
                    make_q4_0_bytes(LFM3_N_HEADS, LFM3_H),
                )
                .tensor(
                    format!("blk.{l}.attn_output.weight"),
                    &[LFM3_Q_DIM, LFM3_H],
                    2,
                    make_q4_0_bytes(LFM3_H, LFM3_Q_DIM),
                )
                .tensor_f32(
                    format!("blk.{l}.attn_q_norm.weight"),
                    &[LFM3_HEAD_DIM],
                    &ones_hd,
                )
                .tensor_f32(
                    format!("blk.{l}.attn_k_norm.weight"),
                    &[LFM3_HEAD_DIM],
                    &ones_hd,
                )
                .tensor(
                    format!("blk.{l}.ffn_gate.weight"),
                    &[LFM3_H, LFM3_FFN_SRC],
                    2,
                    make_q4_0_bytes(LFM3_FFN_SRC, LFM3_H),
                )
                .tensor(
                    format!("blk.{l}.ffn_up.weight"),
                    &[LFM3_H, LFM3_FFN_SRC],
                    2,
                    make_q4_0_bytes(LFM3_FFN_SRC, LFM3_H),
                )
                .tensor(
                    format!("blk.{l}.ffn_down.weight"),
                    &[LFM3_FFN_SRC, LFM3_H],
                    2,
                    make_q4_0_bytes(LFM3_H, LFM3_FFN_SRC),
                );
        }
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    fn tiny_lfm3_nonzero_bidir_gguf(ctx_len: u32) -> GgufFile {
        let vocab = 32usize;
        let attn_norm: Vec<f32> = (0..LFM3_H).map(|i| 0.8 + ((i % 7) as f32) * 0.05).collect();
        let post_attn_norm: Vec<f32> = (0..LFM3_H).map(|i| 0.9 + ((i % 5) as f32) * 0.04).collect();
        let ffn_norm: Vec<f32> = (0..LFM3_H)
            .map(|i| 0.85 + ((i % 9) as f32) * 0.03)
            .collect();
        let post_ffn_norm: Vec<f32> = (0..LFM3_H)
            .map(|i| 0.95 + ((i % 4) as f32) * 0.05)
            .collect();
        let ones_h = vec![1.0f32; LFM3_H];
        let ones_hd = vec![1.0f32; LFM3_HEAD_DIM];
        let b = GgufBuilder::new()
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
            .kv_u32("lfm3.context_length", ctx_len)
            .kv_f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("lfm3.shortconv.l_cache", LFM3_CONV_TAP as u32)
            .kv_bool("lfm3.attention.causal", false)
            .kv_str("lfm3.decision.type", "lfm3-d1")
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
            .tensor_f32("blk.1.attn_norm.weight", &[LFM3_H], &attn_norm)
            .tensor_f32(
                "blk.1.post_attention_norm.weight",
                &[LFM3_H],
                &post_attn_norm,
            )
            .tensor_f32("blk.1.ffn_norm.weight", &[LFM3_H], &ffn_norm)
            .tensor_f32("blk.1.post_ffw_norm.weight", &[LFM3_H], &post_ffn_norm)
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
                make_nonzero_q4_0_bytes(LFM3_H, LFM3_Q_DIM, 3),
            )
            .tensor_f32("blk.1.attn_q_norm.weight", &[LFM3_HEAD_DIM], &ones_hd)
            .tensor_f32("blk.1.attn_k_norm.weight", &[LFM3_HEAD_DIM], &ones_hd)
            .tensor(
                "blk.1.ffn_gate.weight",
                &[LFM3_H, LFM3_FFN_SRC],
                2,
                make_nonzero_q4_0_bytes(LFM3_FFN_SRC, LFM3_H, 5),
            )
            .tensor(
                "blk.1.ffn_up.weight",
                &[LFM3_H, LFM3_FFN_SRC],
                2,
                make_nonzero_q4_0_bytes(LFM3_FFN_SRC, LFM3_H, 7),
            )
            .tensor(
                "blk.1.ffn_down.weight",
                &[LFM3_FFN_SRC, LFM3_H],
                2,
                make_nonzero_q4_0_bytes(LFM3_H, LFM3_FFN_SRC, 11),
            );
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    fn seed_lfm3_attention_and_conv(st: &mut InferenceState, pos: usize, k_val: f32, v_val: f32) {
        let k_f16 = crate::quant::f32_to_f16(k_val);
        let v_f16 = crate::quant::f32_to_f16(v_val);
        for layer in &mut st.layers {
            match layer {
                LayerState::Attention {
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    key_cache_f16.resize(pos * LFM3_KV_DIM, k_f16);
                    value_cache_f16.resize(pos * LFM3_KV_DIM, v_f16);
                }
                LayerState::Conv { buffer, .. } => {
                    buffer.resize(CONV_TAPS_STORED * LFM3_H, 0.25);
                }
                _ => {}
            }
        }
        st.seq_len = pos;
    }

    #[test]
    fn lfm3_create_run_bo_counts_match_xclbin_metadata_including_ext_and_greedy_16k() {
        let prefill_bos = xclbin_bo_arg_count(Q4PREFILL_LFM3_XCLBIN);
        let dec_bos = xclbin_bo_arg_count(Q4LAYER_LFM3_XCLBIN);
        assert_eq!(prefill_bos, 5);
        assert_eq!(dec_bos, 6);

        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_3layer_swa_gguf(true, 32768, 64);
            let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 16512, "bo-check".to_string())
                .expect("load 3-layer causal SWA+Global LFM3");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                16512,
            )
            .unwrap();

            clear_recorded_run_args();
            let pre_fast = model.forward_prefill(&vec![1; 64], 0, &mut state);
            assert_eq!(pre_fast.len(), model.config().vocab_size);
            let pre_fast_calls = take_recorded_run_args();
            assert!(!pre_fast_calls.is_empty());
            for (kind, count) in &pre_fast_calls {
                assert_eq!(*count, prefill_bos, "{kind} BO count mismatch");
            }

            // Fast decode (`pos = 64`) and ext multi-block greedy decode (`pos = 16353`).
            clear_recorded_run_args();
            let _ = model.forward(&[1], 64, &mut state);
            let fast_dec_calls = take_recorded_run_args();
            assert!(fast_dec_calls.iter().any(|(k, _)| *k == "dec_conv"));
            assert!(fast_dec_calls.iter().any(|(k, _)| *k == "dec_attn"));
            for (kind, count) in &fast_dec_calls {
                assert_eq!(*count, dec_bos, "{kind} BO count mismatch");
            }

            seed_lfm3_attention_and_conv(&mut state, 16353, 1.0, -1.0);
            clear_recorded_run_args();
            let tok = model.forward_greedy(&[1], 16353, &mut state);
            assert!((tok as usize) < model.config().vocab_size);
            assert_eq!(state.seq_len, 16354);
            assert!(model.take_decode_error().is_none());
            let ext_dec_calls = take_recorded_run_args();
            for expected in [
                "ext_dec_conv",
                "ext_dec_swa",
                "ext_dec_phase_a",
                "ext_dec_fa_part",
                "ext_dec_fa_fin",
            ] {
                assert!(
                    ext_dec_calls.iter().any(|(k, _)| *k == expected),
                    "pos=16353 forward_greedy must invoke {expected}, got {ext_dec_calls:?}"
                );
            }
            for (kind, count) in &ext_dec_calls {
                assert_eq!(*count, dec_bos, "{kind} BO count mismatch");
            }

            // Directly exercise `build_decode_ext_plan` (`k_part` included) and check all 6-BO counts.
            clear_recorded_run_args();
            {
                let mut guard = npu_model.lock_npu();
                let XdnaLfm3State {
                    ref mut kernels,
                    ref layer_bo_w,
                    ref layer_kv_blocks,
                    ref mut decode,
                    ..
                } = *guard;
                let dec = decode.as_mut().expect("decode buffers initialized");
                let _plan = XdnaLfm3Model::build_decode_ext_plan(
                    &npu_model.layers,
                    kernels,
                    layer_bo_w,
                    layer_kv_blocks,
                    &dec.layer_bo_y,
                    &dec.layer_bo_kvo,
                    &dec.bo_yn,
                    0,
                    1,
                    true,
                    0,
                    false,
                )
                .expect("build_decode_ext_plan");
            }
            let plan_calls = take_recorded_run_args();
            for expected in [
                "dec_ext_conv",
                "dec_ext_swa",
                "dec_ext_phase_a",
                "dec_ext_fa_part",
                "dec_ext_fa_fin",
            ] {
                assert!(
                    plan_calls.iter().any(|(k, _)| *k == expected),
                    "build_decode_ext_plan must record {expected}, got {plan_calls:?}"
                );
            }
            for (kind, count) in &plan_calls {
                assert_eq!(*count, dec_bos, "{kind} BO count mismatch");
            }

            // Aligned >8K prefill (`pos0 = 8192`) exercises `build_prefill_ext_plan`.
            seed_lfm3_attention_and_conv(&mut state, 8192, 0.5, -0.5);
            clear_recorded_run_args();
            let _ = model.forward_prefill(&[1, 2, 3, 4], 8192, &mut state);
            assert!(model.take_decode_error().is_none());
            let ext_pre_calls = take_recorded_run_args();
            assert!(!ext_pre_calls.is_empty());
            for (kind, count) in &ext_pre_calls {
                assert_eq!(*count, prefill_bos, "{kind} BO count mismatch");
            }
        });
    }

    #[test]
    fn bidirectional_over_8k_uses_paged_part_without_fin_and_matches_cpu_reference() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_nonzero_bidir_gguf(16384);
            let mut npu_model =
                XdnaLfm3Model::from_gguf_with_id(gguf, 16384, "bidir-16k".to_string())
                    .expect("load bidir LFM3 at 16K");

            // Part 1: Fake-shim dispatch for both `rem > 0` (`8224`) and `n_kv_chunks % 256 == 0` (`16384`).
            npu_model.bidir_host_ffn_chunk_limit = Some(1);
            let mut part1_state = InferenceState::from_config_capped(
                npu_model.config(),
                &crate::kv_cache::KvCompression::F16,
                16384,
            )
            .unwrap();
            for (run_idx, n_toks) in [8224usize, 16384usize].into_iter().enumerate() {
                if run_idx > 0 {
                    npu_model
                        .try_reset_kv(
                            &mut part1_state,
                            &crate::kv_cache::KvCompression::F16,
                            16384,
                        )
                        .unwrap();
                }
                let model: &dyn Model = &npu_model;
                let tokens = vec![1u32; n_toks];
                clear_recorded_run_args();
                let logits = model.forward_prefill(&tokens, 0, &mut part1_state);
                assert_eq!(logits.len(), model.config().vocab_size);
                assert!(model.take_decode_error().is_none());
                assert!(!npu_model.state_torn.load(Ordering::SeqCst));
                let calls = take_recorded_run_args();
                let n_chunks = n_toks.div_ceil(LFM3_PREFILL_B);
                assert_eq!(
                    calls
                        .iter()
                        .filter(|(k, _)| *k == "prefill_fa_part0")
                        .count(),
                    n_chunks,
                    "expected 1 Mode 6 block0 run per chunk for total_tokens={n_toks}"
                );
                assert_eq!(
                    calls
                        .iter()
                        .filter(|(k, _)| *k == "prefill_fa_part")
                        .count(),
                    n_chunks,
                    "expected 1 Mode 4 block1 run per chunk for total_tokens={n_toks}"
                );
                assert_eq!(
                    calls
                        .iter()
                        .filter(|(k, _)| *k == "prefill_fa_fin" || *k == "ext_prefill_fa_fin")
                        .count(),
                    0,
                    "bidir >8K must never invoke broken Mode 5 skip_cur_kv"
                );
                for (kind, count) in &calls {
                    assert_eq!(*count, LFM3_PREFILL_BO_ARGS, "{kind} BO count");
                }
            }

            // Part 2: Exact CPU parity of multi-block `SLOT_FA_STATE` + `finalize_bidir_fa_chunk_and_ffn`
            // at `total_tokens = 8224` (`blk=0` 8192 tokens = 128x64 periodic keys + `blk=1` 32 tokens)
            // for chunk 0 (`n_valid=64`, exercising both `State0` and `State1`) and tail chunk 128 (`n_valid=32`).
            npu_model.bidir_host_ffn_chunk_limit = None;
            npu_model
                .try_reset_kv(&mut part1_state, &crate::kv_cache::KvCompression::F16, 8256)
                .unwrap();
            let mut npu_state = part1_state;

            const TOTAL_TOKENS: usize = 8224;
            const BLK0_UNIQUE: usize = 64;
            const BLK0_MULT: f32 = 128.0;
            const BLK1_LEN: usize = TOTAL_TOKENS - 8192;
            const UNIQUE_KEYS: usize = BLK0_UNIQUE + BLK1_LEN;

            let eps = npu_model.cpu.config().rms_norm_eps;
            let layer_idx = 1usize;
            let layer = &npu_model.layers[layer_idx];
            let (_, _, _, wo) = npu_model.cpu.xdna_layer_attn_refs(layer_idx).unwrap();
            let gguf_ref = npu_model.cpu.gguf();
            let (fg, fu, fd) = npu_model.cpu.xdna_layer_ffn_refs(layer_idx).unwrap();

            let k_blk0: Vec<f32> = (0..BLK0_UNIQUE * LFM3_KV_DIM)
                .map(|i| (((i * 17) % 29) as f32 - 14.0) * 0.0625)
                .collect();
            let v_blk0: Vec<f32> = (0..BLK0_UNIQUE * LFM3_KV_DIM)
                .map(|i| (((i * 23) % 31) as f32 - 15.0) * 0.0625)
                .collect();
            let mut k_blk1: Vec<f32> = (0..BLK1_LEN * LFM3_KV_DIM)
                .map(|i| (((i * 19 + 5) % 37) as f32 - 18.0) * 0.0625)
                .collect();
            let v_blk1: Vec<f32> = (0..BLK1_LEN * LFM3_KV_DIM)
                .map(|i| (((i * 13 + 7) % 29) as f32 - 14.0) * 0.0625)
                .collect();
            // Ensure block 1 exceeds block 0's running max on aligned queries while remaining below on others.
            for kvh in 0..LFM3_N_KV_HEADS {
                let dst_row =
                    &mut k_blk1[17 * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..][..LFM3_HEAD_DIM];
                for (d, dst) in dst_row.iter_mut().enumerate() {
                    *dst = (((d * 7) % 23) as f32 - 11.0) * 0.125;
                }
            }

            let scale = 1.0 / (LFM3_HEAD_DIM as f32).sqrt();
            let log2e = std::f32::consts::LOG2_E;
            let mut moe_scratch = crate::model::lfm2::XdnaMoePrefillScratch::default();
            let mut attn_out_scratch = vec![0.0f32; LFM3_PREFILL_B * LFM3_H];
            let mut block_out_scratch = vec![0.0f32; LFM3_PREFILL_B * LFM3_H];
            let mut bq_scales = Vec::new();
            let mut bq_quants = Vec::new();
            #[cfg(has_blas)]
            let mut dequant_scratch = Vec::new();
            let mut saw_rescale = false;
            let mut saw_no_rescale = false;

            for (chunk_tag, n_valid) in [(0usize, 64usize), (128usize, 32usize)] {
                let mut bo_y_bytes = vec![0u8; LFM3_PREFILL_ACT_BYTES];
                let mut saved_gate = vec![0u8; PHASE_A_GATE_BYTES];
                let mut hidden_npu: Vec<f32> = (0..n_valid * LFM3_H)
                    .map(|i| (((i * 13 + chunk_tag * 7) % 19) as f32 - 9.0) * 0.0625)
                    .collect();
                let mut hidden_cpu = hidden_npu.clone();
                let mut cpu_attn_out = vec![0.0f32; n_valid * LFM3_H];

                for t in 0..n_valid {
                    let sub = t >> 5;
                    let r = (t >> 3) & 3;
                    let lt = t & 7;
                    for h in 0..LFM3_N_HEADS {
                        let kvh = h / (LFM3_N_HEADS / LFM3_N_KV_HEADS);
                        let mut q_vec = [0.0f32; LFM3_HEAD_DIM];
                        for (d, qd) in q_vec.iter_mut().enumerate() {
                            *qd = (((t * 31 + h * 19 + d * 7 + chunk_tag) % 23) as f32 - 11.0)
                                * 0.0625;
                        }
                        let g_logit = ((((t * 11 + h * 5 + chunk_tag) % 17) as f32) - 8.0) * 0.125;
                        saved_gate[(t * LFM3_N_HEADS + h) * 4..(t * LFM3_N_HEADS + h) * 4 + 4]
                            .copy_from_slice(&g_logit.to_le_bytes());
                        let g_sig = 1.0f32 / (1.0f32 + (-g_logit).exp());

                        let mut dots = [0.0f32; UNIQUE_KEYS];
                        for (u, dot_dst) in dots.iter_mut().enumerate() {
                            let k_sl = if u < BLK0_UNIQUE {
                                &k_blk0[u * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..][..LFM3_HEAD_DIM]
                            } else {
                                &k_blk1[(u - BLK0_UNIQUE) * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                    [..LFM3_HEAD_DIM]
                            };
                            *dot_dst = crate::backend::cpu::dot_f32(&q_vec, k_sl);
                        }

                        // Exact equivalent of `softmax_inplace` over all 8224 keys (blk0 has 128x multiplicity).
                        let mut max_s = f32::NEG_INFINITY;
                        let mut scores = [0.0f32; UNIQUE_KEYS];
                        for (dst, &dot) in scores.iter_mut().zip(&dots) {
                            let s = dot * scale;
                            *dst = s;
                            max_s = max_s.max(s);
                        }
                        let mut sum = 0.0f64;
                        for (u, w) in scores.iter_mut().enumerate() {
                            let e = crate::backend::cpu::ggml_expf(*w - max_s);
                            *w = e;
                            let mult = if u < BLK0_UNIQUE {
                                f64::from(BLK0_MULT)
                            } else {
                                1.0
                            };
                            sum += f64::from(e) * mult;
                        }
                        let inv_sum = (1.0 / sum) as f32;
                        let cpu_dst =
                            &mut cpu_attn_out[t * LFM3_H + h * LFM3_HEAD_DIM..][..LFM3_HEAD_DIM];
                        for (u, &e) in scores.iter().enumerate() {
                            let (v_sl, mult) = if u < BLK0_UNIQUE {
                                (
                                    &v_blk0[u * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                        [..LFM3_HEAD_DIM],
                                    BLK0_MULT,
                                )
                            } else {
                                (
                                    &v_blk1
                                        [(u - BLK0_UNIQUE) * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                        [..LFM3_HEAD_DIM],
                                    1.0,
                                )
                            };
                            let ws = e * inv_sum * g_sig * mult;
                            for (dst, &v_val) in cpu_dst.iter_mut().zip(v_sl) {
                                *dst += ws * v_val;
                            }
                        }

                        // Multi-block base-2 online softmax matching `q4prefill_lfm3_core.cc`.
                        let mut row_m = -1e30f32;
                        let mut row_l = 0.0f32;
                        let mut o_acc = [0.0f32; LFM3_HEAD_DIM];
                        for (blk_idx, (u0, u1, mult, v_bank)) in [
                            (0usize, BLK0_UNIQUE, BLK0_MULT, v_blk0.as_slice()),
                            (BLK0_UNIQUE, UNIQUE_KEYS, 1.0f32, v_blk1.as_slice()),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let mut blk_m = -1e30f32;
                            for &d in &dots[u0..u1] {
                                blk_m = blk_m.max(d * scale * log2e);
                            }
                            let new_m = row_m.max(blk_m);
                            if row_l > 0.0 && new_m > row_m {
                                saw_rescale = true;
                                let rs = (row_m - new_m).exp2();
                                row_l *= rs;
                                for v in &mut o_acc {
                                    *v *= rs;
                                }
                            } else if blk_idx == 1 {
                                saw_no_rescale = true;
                            }
                            row_m = new_m;
                            for (o, &dot) in dots[u0..u1].iter().enumerate() {
                                let p = (dot * scale * log2e - new_m).exp2() * mult;
                                row_l += p;
                                let v_sl = &v_bank[o * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                    [..LFM3_HEAD_DIM];
                                for (acc_dst, &v_val) in o_acc.iter_mut().zip(v_sl) {
                                    *acc_dst += p * v_val;
                                }
                            }
                        }

                        // Spot-check the folded calculation against an explicit flat 8224-key scan
                        // on both `sub=0` (`t=0, h=0`) and `sub=1` (`t=33, h=5`).
                        if chunk_tag == 0 && ((t == 0 && h == 0) || (t == 33 && h == 5)) {
                            let mut flat_scores = vec![0.0f32; TOTAL_TOKENS];
                            for (j, score_dst) in flat_scores.iter_mut().enumerate() {
                                let k_sl = if j < 8192 {
                                    &k_blk0[(j % BLK0_UNIQUE) * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                        [..LFM3_HEAD_DIM]
                                } else {
                                    &k_blk1[(j - 8192) * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                        [..LFM3_HEAD_DIM]
                                };
                                *score_dst = crate::backend::cpu::dot_f32(&q_vec, k_sl) * scale;
                            }
                            crate::backend::cpu::softmax_inplace(&mut flat_scores);
                            let mut flat_attn = [0.0f32; LFM3_HEAD_DIM];
                            for (j, &w) in flat_scores.iter().enumerate() {
                                let v_sl = if j < 8192 {
                                    &v_blk0[(j % BLK0_UNIQUE) * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                        [..LFM3_HEAD_DIM]
                                } else {
                                    &v_blk1[(j - 8192) * LFM3_KV_DIM + kvh * LFM3_HEAD_DIM..]
                                        [..LFM3_HEAD_DIM]
                                };
                                let ws = w * g_sig;
                                for (dst, &v_val) in flat_attn.iter_mut().zip(v_sl) {
                                    *dst += ws * v_val;
                                }
                            }
                            let flat_diff = flat_attn
                                .iter()
                                .zip(cpu_dst.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            assert!(
                                flat_diff < 1e-5,
                                "flat 8224-key cross-check diverged at (t={t}, h={h}): {flat_diff}"
                            );
                        }

                        let c = h >> 1;
                        let h_local = h & 1;
                        let state0_core = LFM3_PREFILL_SLOT_FA_STATE
                            + PREFILL_FA_STATE1_FLOATS
                            + c * (4 * 2304)
                            + r * 2304;
                        let row_l_off = (state0_core + 2048 + h_local * 16 + sub * 8 + lt) * 4;
                        bo_y_bytes[row_l_off..row_l_off + 4].copy_from_slice(&row_l.to_le_bytes());
                        let oa_core = if sub == 0 {
                            state0_core
                        } else {
                            LFM3_PREFILL_SLOT_FA_STATE
                                + (c >> 1) * (4 * 4608)
                                + (c & 1) * 2048
                                + r * 4608
                        };
                        let oa_head = oa_core + h_local * 1024;
                        for d8 in 0..(LFM3_HEAD_DIM / 8) {
                            let dst_f32 = oa_head + d8 * 64 + lt * 8;
                            for lane in 0..8 {
                                let b = (dst_f32 + lane) * 4;
                                bo_y_bytes[b..b + 4]
                                    .copy_from_slice(&o_acc[d8 * 8 + lane].to_le_bytes());
                            }
                        }
                    }
                }

                // First pin exact `SLOT_FA_STATE` unpacking against the CPU softmax reference.
                XdnaLfm3Model::unpack_bidir_fa_state_into_attn_out(
                    &bo_y_bytes,
                    &saved_gate,
                    n_valid,
                    &mut attn_out_scratch,
                );
                let attn_max_err = attn_out_scratch[..n_valid * LFM3_H]
                    .iter()
                    .zip(&cpu_attn_out)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    attn_max_err < 1e-5,
                    "bidir >8K chunk {chunk_tag} SLOT_FA_STATE unpack error too high: {attn_max_err}"
                );

                // CPU O_PROJ + post_attn_norm + residual + FFN + post_ffn_norm + residual.
                let mut norm_tmp = vec![0.0f32; LFM3_H];
                let mut ffn_g = vec![0.0f32; LFM3_FFN_SRC];
                let mut ffn_u = vec![0.0f32; LFM3_FFN_SRC];
                let mut ffn_d = vec![0.0f32; LFM3_H];
                let mut o_proj = vec![0.0f32; LFM3_H];
                for t in 0..n_valid {
                    let hs_row = &mut hidden_cpu[t * LFM3_H..(t + 1) * LFM3_H];
                    transformer::gemv(
                        gguf_ref,
                        wo,
                        &cpu_attn_out[t * LFM3_H..(t + 1) * LFM3_H],
                        &mut o_proj,
                    );
                    crate::backend::cpu::rmsnorm(&mut o_proj, &layer.post_attn_norm, eps);
                    for (dst, &val) in hs_row.iter_mut().zip(&o_proj) {
                        *dst += val;
                    }
                    norm_tmp.copy_from_slice(hs_row);
                    crate::backend::cpu::rmsnorm(&mut norm_tmp, &layer.ffn_norm, eps);
                    transformer::gemv(gguf_ref, fg, &norm_tmp, &mut ffn_g);
                    transformer::gemv(gguf_ref, fu, &norm_tmp, &mut ffn_u);
                    for (g_ref, &u_val) in ffn_g.iter_mut().zip(&ffn_u) {
                        let g = *g_ref;
                        *g_ref = (g / (1.0 + (-g).exp())) * u_val;
                    }
                    transformer::gemv(gguf_ref, fd, &ffn_g, &mut ffn_d);
                    crate::backend::cpu::rmsnorm(&mut ffn_d, &layer.post_ffn_norm, eps);
                    for (dst, &val) in hs_row.iter_mut().zip(&ffn_d) {
                        *dst += val;
                    }
                }

                npu_model
                    .finalize_bidir_fa_chunk_and_ffn(
                        layer_idx,
                        &bo_y_bytes,
                        &saved_gate,
                        n_valid,
                        eps,
                        &mut hidden_npu,
                        &mut attn_out_scratch,
                        &mut block_out_scratch,
                        &mut bq_scales,
                        &mut bq_quants,
                        #[cfg(has_blas)]
                        &mut dequant_scratch,
                        &mut moe_scratch,
                        &mut npu_state,
                    )
                    .expect("finalize_bidir_fa_chunk_and_ffn");

                let cos = cosine_similarity(&hidden_npu, &hidden_cpu);
                let max_err = hidden_npu
                    .iter()
                    .zip(&hidden_cpu)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                // Batched `project_rowmajor_f32` / `xdna_prefill_ffn_chunk` (`lin16` + `fast_silu`
                // or BLAS SGEMM) vs per-token Q8_0 `gemv` flips occasional Q8_0 activation bins across
                // `O_PROJ` + `7680`-wide FFN (`cos ~ 0.999999`, `max_err ~ 0.014-0.02`, matching `lfm2.rs`).
                assert!(
                    cos >= 0.9999 && max_err < 0.05,
                    "bidir >8K chunk {chunk_tag} parity failed: cos={cos} max_err={max_err}"
                );
            }
            assert!(
                saw_rescale && saw_no_rescale,
                "expected both block-1 rescaling and non-rescaling online softmax paths"
            );
        });
    }

    #[test]
    fn lfm3_truncate_kv_missing_conv_checkpoint_clears_synced_len_and_swa_base() {
        with_shim(FakeConfig::default(), || {
            // Case A: non-SWA (`swa_base == 0`), prefill pushes checkpoint only at pos=4;
            // `truncate_kv(&mut state, 2)` falls back to `seq_len = 0` and must reset `synced_len = 0`.
            let gguf = tiny_lfm3_gguf(true);
            let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 128, "trunc-fallback".into())
                .expect("load causal LFM3");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                128,
            )
            .unwrap();
            let _ = model.forward_prefill(&[1, 2, 3, 4], 0, &mut state);
            assert_eq!(state.seq_len, 4);
            assert_eq!(npu_model.lock_npu().synced_len, 4);

            model.truncate_kv(&mut state, 2);
            assert_eq!(state.seq_len, 0);
            assert_eq!(
                npu_model.lock_npu().synced_len,
                0,
                "synced_len must reset to state.seq_len (0), not stale len (2)"
            );

            // Re-populating CPU KV externally to len=2 and decoding at pos=2 must resync NPU KV.
            seed_lfm3_attention_and_conv(&mut state, 2, 3.0, -3.0);
            let out = model.forward(&[5], 2, &mut state);
            assert_eq!(out.len(), model.config().vocab_size);
            assert!(model.take_decode_error().is_none());
            assert_eq!(npu_model.lock_npu().synced_len, 3);
            let k0 = u16::from_le_bytes(
                npu_model.lock_npu().layer_kv_blocks[1][0].as_slice()[0..2]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                k0,
                crate::backend::xdna::repack::f32_to_bf16(3.0),
                "post-fallback forward must sync CPU KV instead of trusting stale NPU KV"
            );

            // Case B: SWA (`swa_base > 0`) with `prev_synced <= len < state.seq_len` after CPU growth.
            let swa_gguf = tiny_lfm3_gguf_with_swa(true, 1024, Some(64));
            let swa_model = XdnaLfm3Model::from_gguf_with_id(swa_gguf, 1024, "trunc-swa".into())
                .expect("load SWA LFM3");
            let mut swa_state = InferenceState::from_config_capped(
                swa_model.config(),
                &crate::kv_cache::KvCompression::F16,
                1024,
            )
            .unwrap();
            let _ = swa_model.forward_prefill(&[1, 2], 0, &mut swa_state);
            assert_eq!(swa_model.lock_npu().synced_len, 2);
            swa_model.lock_npu().swa_base_chunk[1] = 7;
            seed_lfm3_attention_and_conv(&mut swa_state, 5, 1.0, -1.0);
            swa_model.truncate_kv(&mut swa_state, 3);
            assert_eq!(swa_state.seq_len, 0);
            let guard = swa_model.lock_npu();
            assert_eq!(guard.synced_len, 0);
            assert_eq!(
                guard.swa_base_chunk[1], 0,
                "fallback truncate must clear swa_base_chunk even when len >= prev_synced"
            );
        });
    }

    #[test]
    fn causal_swa_prefill_compaction_with_n_valid_over_32_on_fake_shim() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_gguf_with_swa(true, 1024, Some(64));
            let npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 1024, "swa-pre48".into())
                .expect("load SWA LFM3");
            let model: &dyn Model = &npu_model;
            let mut state = InferenceState::from_config_capped(
                model.config(),
                &crate::kv_cache::KvCompression::F16,
                1024,
            )
            .unwrap();

            // For `swa_window=64`, `keep_chunks=2`, `limit_chunks=10` (`320` tokens).
            // Starting at `pos0 = 288` (`chunk=9`) with `n_valid = 48` crosses `limit=10` between
            // subchunk 0 (`chunk 9`) and subchunk 1 (`chunk 10`), shifting `loc0` by `shift1 = 8`.
            seed_lfm3_attention_and_conv(&mut state, 288, 2.0, -2.0);
            let out = model.forward_prefill(&[1; 48], 288, &mut state);
            assert_eq!(out.len(), model.config().vocab_size);
            assert_eq!(state.seq_len, 336);
            assert!(model.take_decode_error().is_none());
            {
                let guard = npu_model.lock_npu();
                assert_eq!(guard.swa_base_chunk[1], 8);
                let preserved = u16::from_le_bytes(
                    guard.layer_kv_blocks[1][0].as_slice()[0..2]
                        .try_into()
                        .unwrap(),
                );
                assert_eq!(
                    preserved,
                    crate::backend::xdna::repack::f32_to_bf16(2.0),
                    "chunk 8 K must shift intact to loc=0 during n_valid=48 mid-chunk compaction"
                );
            }

            // Directly exercise `save_attn_prefill_kvo_to_bo_kv` + `mirror_attn_prefill_bo_kv_to_state`
            // with distinct sub0 (`4.0`) and sub1 (`5.0`) BF16 payloads to verify shifted `loc0`/`loc1`.
            let mut guard = npu_model.lock_npu();
            guard.swa_base_chunk[1] = 0;
            let XdnaLfm3State {
                ref kernels,
                ref mut layer_kv_blocks,
                ref mut swa_base_chunk,
                ref mut prefill,
                ..
            } = *guard;
            let kvo = &mut prefill.as_mut().unwrap().layer_bo_kvo[1];
            let sub0_k = crate::backend::xdna::repack::f32_to_bf16(4.0).to_le_bytes();
            let sub1_k = crate::backend::xdna::repack::f32_to_bf16(5.0).to_le_bytes();
            {
                let kb = kvo.as_mut_slice();
                kb.fill(0);
                for kvh in 0..LFM3_N_KV_HEADS {
                    let base0 = kvh * TILE_BYTES;
                    let base1 = (LFM3_N_KV_HEADS + kvh) * TILE_BYTES;
                    for t in 0..LFM3_KV_CHUNK_T {
                        for d in 0..LFM3_HEAD_DIM {
                            let off = t * LFM3_HEAD_DIM * 2 + d * 2;
                            kb[base0 + off..base0 + off + 2].copy_from_slice(&sub0_k);
                            if t < 16 {
                                kb[base1 + off..base1 + off + 2].copy_from_slice(&sub1_k);
                            }
                        }
                    }
                }
            }
            let entry = XdnaLfm3Model::save_attn_prefill_kvo_to_bo_kv(
                kvo,
                &mut layer_kv_blocks[1],
                &kernels.prefill_conv,
                &mut swa_base_chunk[1],
                64,
                1,
                288,
                48,
            )
            .expect("save_attn_prefill_kvo_to_bo_kv n_valid=48");
            assert_eq!(entry.loc0, 1);
            assert_eq!(entry.loc1, 2);
            assert_eq!(entry.n0, 32);
            assert_eq!(entry.n1, 16);
            assert_eq!(swa_base_chunk[1], 8);
            XdnaLfm3Model::mirror_attn_prefill_bo_kv_to_state(
                &layer_kv_blocks[1],
                &mut state.layers[1],
                &entry,
                true,
            );
            let (k_f16, _) = state.kv_cache_f16(1);
            assert_eq!(crate::quant::f16_to_f32(k_f16[288 * LFM3_KV_DIM]), 4.0);
            assert_eq!(crate::quant::f16_to_f32(k_f16[320 * LFM3_KV_DIM]), 5.0);
        });
    }

    #[test]
    fn causal_prefill_fault_flushes_pending_attn_mirrors_and_rejects_non_uniform_swa() {
        // Run chunk 0 at `pos0 = 16320` (`num_full=1`, RunList succeeds and buffers `pending_attn_mirrors`)
        // followed by chunk 1 at `pos0 = 16384` (`num_full=2`, `XdnaRun::wait` fails under `run_wait_state=Some(0)`),
        // verifying that chunk 0's pending attention mirror (`16320..16384`) is still flushed to CPU `state`.
        with_shim(
            FakeConfig {
                run_wait_state: Some(0),
                fail_runlist_wait: false,
                ..Default::default()
            },
            || {
                let gguf = tiny_lfm3_gguf_with_ctx(true, 32768);
                let npu_model =
                    XdnaLfm3Model::from_gguf_with_id(gguf, 16512, "mirror-flush".into())
                        .expect("load causal LFM3");
                let model: &dyn Model = &npu_model;
                let mut state = InferenceState::from_config_capped(
                    model.config(),
                    &crate::kv_cache::KvCompression::F16,
                    16512,
                )
                .unwrap();
                seed_lfm3_attention_and_conv(&mut state, 16320, 1.0, -1.0);
                assert_eq!(state.kv_cache_f16(1).0.len(), 16320 * LFM3_KV_DIM);

                let out = model.forward_prefill(&vec![1; 128], 16320, &mut state);
                assert_eq!(out, vec![0.0f32; model.config().vocab_size]);
                assert!(npu_model.state_torn.load(Ordering::SeqCst));
                assert!(model.take_decode_error().is_some());
                assert_eq!(
                    state.kv_cache_f16(1).0.len(),
                    16384 * LFM3_KV_DIM,
                    "chunk 0 pending attention mirror must flush on chunk 1 prefill fault"
                );
            },
        );

        // Verify `verify_uniform_swa_step` catches non-uniform SWA windows/bases across layers.
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_3layer_swa_gguf(true, 1024, 64);
            let mut npu_model = XdnaLfm3Model::from_gguf_with_id(gguf, 1024, "uniform-swa".into())
                .expect("load 3-layer LFM3");
            npu_model.layers[2].swa_window = 128;
            let mut state = InferenceState::from_config_capped(
                npu_model.cpu.config(),
                &crate::kv_cache::KvCompression::F16,
                1024,
            )
            .unwrap();
            let emb = npu_model.cpu.embed_token_rows(&[1]).unwrap();
            {
                let mut guard = npu_model.lock_npu();
                let err = npu_model
                    .run_npu_decode_embedding_step_inner(
                        &mut guard,
                        &emb,
                        0,
                        &mut state,
                        false,
                        &mut [],
                    )
                    .expect_err("non-uniform swa_window across layers must error");
                assert!(
                    err.contains("non-uniform SWA state"),
                    "unexpected error: {err}"
                );
            }
            npu_model.layers[2].swa_window = 64;
            {
                let mut guard = npu_model.lock_npu();
                guard.swa_base_chunk[2] = 1;
                let err = npu_model
                    .run_npu_decode_embedding_step_inner(
                        &mut guard,
                        &emb,
                        0,
                        &mut state,
                        false,
                        &mut [],
                    )
                    .expect_err("non-uniform swa_base_chunk across layers must error");
                assert!(
                    err.contains("non-uniform SWA state"),
                    "unexpected error: {err}"
                );
            }
        });
    }

    #[test]
    fn project_rowmajor_f32_reused_oversized_scratch_matches_gemv() {
        with_shim(FakeConfig::default(), || {
            let gguf = tiny_lfm3_nonzero_bidir_gguf(128);
            let npu_model =
                XdnaLfm3Model::from_gguf_with_id(gguf, 128, "rowmajor-slice".to_string())
                    .expect("load LFM3");
            let (_, _, _, wo) = npu_model.cpu.xdna_layer_attn_refs(1).unwrap();
            let wo_unrepacked = WeightRef::new(wo.start, wo.size, wo.dtype, wo.m, wo.k);
            let gguf_ref = npu_model.cpu.gguf();

            let check_matches_gemv =
                |wref: &WeightRef, input: &[f32], got: &[f32], n: usize, tag: &str| {
                    let mut expected = vec![0.0f32; n * LFM3_H];
                    for t in 0..n {
                        transformer::gemv(
                            gguf_ref,
                            wref,
                            &input[t * LFM3_Q_DIM..(t + 1) * LFM3_Q_DIM],
                            &mut expected[t * LFM3_H..(t + 1) * LFM3_H],
                        );
                    }
                    let cos = cosine_similarity(got, &expected);
                    let max_err = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        cos >= 0.9999 && max_err < 0.05,
                        "{tag} mismatch: cos={cos} max_err={max_err}"
                    );
                };

            for (label, wref) in [("repacked", wo), ("unrepacked", &wo_unrepacked)] {
                let mut bq_scales = Vec::new();
                let mut bq_quants = Vec::new();
                #[cfg(has_blas)]
                let mut dequant_scratch = Vec::new();

                // Grow persistent scratch at n = 64 (when preq path runs), then reuse at n = 32
                // so `bq_scales` (`4096`) and `bq_quants` (`131072`) stay oversized for tail `n = 32`.
                for (step, n) in [64usize, 32usize].into_iter().enumerate() {
                    let input: Vec<f32> = (0..n * LFM3_Q_DIM)
                        .map(|i| (((i * 19 + step * 7) % 31) as f32 - 15.0) * 0.0625)
                        .collect();
                    let mut got = vec![0.0f32; n * LFM3_H];
                    XdnaLfm3Model::project_rowmajor_f32(
                        gguf_ref,
                        wref,
                        &input,
                        &mut got,
                        n,
                        LFM3_H,
                        LFM3_Q_DIM,
                        &mut bq_scales,
                        &mut bq_quants,
                        #[cfg(has_blas)]
                        &mut dequant_scratch,
                    );
                    check_matches_gemv(
                        wref,
                        &input,
                        &got,
                        n,
                        &format!("{label} project_rowmajor_f32 n={n}"),
                    );
                }

                #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(has_blas)))]
                {
                    let nb = LFM3_Q_DIM / 32;
                    if label == "unrepacked" {
                        assert_eq!(bq_scales.len(), 64 * nb);
                        assert_eq!(bq_quants.len(), 64 * LFM3_Q_DIM);
                    } else {
                        // On AVX-512 VNNI, `lin16` takes the f32 path above and leaves `bq_*` empty;
                        // populate oversized scratch explicitly so `gemm_preq_rowmajor` is tested on
                        // repacked weights on every architecture as well.
                        if bq_scales.len() < 64 * nb {
                            bq_scales.resize(64 * nb, 0.0);
                        }
                        if bq_quants.len() < 64 * LFM3_Q_DIM {
                            bq_quants.resize(64 * LFM3_Q_DIM, 0);
                        }
                    }
                    let n = 32usize;
                    let input: Vec<f32> = (0..n * LFM3_Q_DIM)
                        .map(|i| (((i * 23 + 3) % 29) as f32 - 14.0) * 0.0625)
                        .collect();
                    transformer::quantize_rows(
                        &input,
                        LFM3_Q_DIM,
                        n,
                        &mut bq_scales[..n * nb],
                        &mut bq_quants[..n * LFM3_Q_DIM],
                    );
                    let mut got_oversized = vec![0.0f32; n * LFM3_H];
                    assert!(
                        transformer::gemm_preq_rowmajor(
                            gguf_ref,
                            wref,
                            &bq_scales,
                            &bq_quants,
                            &mut got_oversized,
                            n,
                            LFM3_H,
                            LFM3_Q_DIM,
                        ),
                        "{label} gemm_preq_rowmajor with oversized scratch should run"
                    );
                    check_matches_gemv(
                        wref,
                        &input,
                        &got_oversized,
                        n,
                        &format!("{label} gemm_preq_rowmajor oversized"),
                    );
                }
            }
        });
    }
}

//! AMD XDNA NPU backend for Llama-3.2-1B (`q4layer` + `q4head`).
//!
//! Prefill runs on the CPU via [`LlamaModel`] (per Section 4 of
//! `docs/XDNA_PHASE0_RESULTS.md`), and single-token decode runs all 16
//! transformer layers on the NPU through [`Q4LAYER_XCLBIN`] followed by the
//! final RMSNorm and 128,256-row LM head on the NPU through [`Q4HEAD_XCLBIN`]
//! in a single 17-run `xrt::runlist`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::backend::xdna::device::{XdnaBo, XdnaDevice, XdnaKernel, XdnaRun, XdnaRunList};
pub use crate::backend::xdna::kernels::MAX_KV_CACHE_TOKENS;
use crate::backend::xdna::kernels::{
    Q4HEAD_INSTS, Q4HEAD_XCLBIN, Q4LAYER_ACT_BYTES, Q4LAYER_ACT_SLOT, Q4LAYER_ATTN_CHUNK_T,
    Q4LAYER_ATTN_CHUNKS, Q4LAYER_ATTN_N_OFF, Q4LAYER_ATTN_NORM_W, Q4LAYER_KV_BUCKETS,
    Q4LAYER_KV_BYTES, Q4LAYER_N_KV_HEADS, Q4LAYER_ROPE_CS, Q4LAYER_SILU_HALF, Q4LAYER_TILE_F32,
    Q4LAYER_XCLBIN, Q4PREFILL_ACT_BYTES, Q4PREFILL_B, Q4PREFILL_LAYER_KV_BUCKETS,
    Q4PREFILL_LAYER_XCLBIN, Q4PREFILL_MAX_POS0, Q4PREFILL_SLOT_KV_CUR, Q4PREFILL_SLOT_META,
    Q4PREFILL_W_BYTES, init_q4layer_bo_y_norms, init_q4prefill_bo_y_norms, inst_words,
    pack_q4layer_rope_cs, read_le_f32_slice, select_q4layer_kv_bucket, select_q4prefill_kv_bucket,
    write_le_f32_slice, write_le_i32, write_q4layer_kv_bf16, write_q4layer_step_params,
    write_q4prefill_step_params,
};
use crate::backend::xdna::repack::{
    interleave_worker_stages, pack_f32_stage, pack_q4_0_stage, stage_bytes,
};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError, LayerState};
use crate::model::llama::LlamaModel;
use crate::model::transformer;
use crate::model::{
    Model, ModelConfig, ModelSessionGate, ModelSessionLease, record_first_fault, take_fault,
};
use crate::quant::{bf16_to_f32, f32_to_f16};
use crate::session::CeraError;
use crate::tensor::DType;

const N_LAYERS: usize = 16;
const HIDDEN_SIZE: usize = 2048;
const INTERMEDIATE_SIZE: usize = 8192;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const HEAD_PADDED_VOCAB: usize = 131_072;
const HEAD_REPACK_CHUNK_ROWS: usize = 256;
pub(crate) const QUIESCE_TIMEOUT_MS: u32 = 30_000;

const HEAD_Y_BYTES: usize = (Q4LAYER_TILE_F32 + HEAD_PADDED_VOCAB) * 4;

const LAYER_STAGE_SHAPES: [(usize, usize); 7] = [
    (2048, 2048), // Stage 0: Q
    (512, 2048),  // Stage 1: K
    (512, 2048),  // Stage 2: V
    (2048, 2048), // Stage 3: O
    (8192, 2048), // Stage 4: Gate
    (8192, 2048), // Stage 5: Up
    (2048, 8192), // Stage 6: Down
];

type LayerKvPair = (Vec<[f32; KV_DIM]>, Vec<[f32; KV_DIM]>);

struct XdnaPrefillState {
    bucket_chunks: Vec<usize>,
    bucket_runlists: Vec<XdnaRunList>,
    head_run: XdnaRun,
    _layer_runs: Vec<Vec<XdnaRun>>,
    bo_out: XdnaBo,
    bo_y: Vec<XdnaBo>,
    _bo_w: Vec<XdnaBo>,
    _kernels: Vec<XdnaKernel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InFlightXdnaOp {
    DecodeBucket(usize),
    PrefillBucket(usize),
    PrefillHead,
}

#[derive(Debug)]
enum SyncErr {
    Input(String),
    Device(String),
}

struct XdnaDecodeState {
    bucket_chunks: Vec<usize>,
    bucket_runlists: Vec<XdnaRunList>,
    _head_runs: Vec<XdnaRun>,
    _layer_runs: Vec<Vec<XdnaRun>>,
    head_bo_y: XdnaBo,
    _head_bo_w: XdnaBo,
    layer_bo_kv: Vec<XdnaBo>,
    layer_bo_y: Vec<XdnaBo>,
    _layer_bo_w: Vec<XdnaBo>,
    _head_kernel: XdnaKernel,
    _layer_kernels: Vec<XdnaKernel>,
    prefill: Option<Box<XdnaPrefillState>>,
    _device: XdnaDevice,
    in_flight: Option<InFlightXdnaOp>,
    synced_kv_len: usize,
    cached_rope: Option<(usize, [f32; 128])>,
    pending_cpu_kv: Option<(usize, LayerKvPair)>,
    spare_kv: Option<LayerKvPair>,
    q8_scales: Vec<f32>,
    q8_quants: Vec<i8>,
}

impl Drop for XdnaDecodeState {
    fn drop(&mut self) {
        if let Err(e) = quiesce_in_flight(self) {
            tracing::error!("cera::xdna: Drop failed to quiesce in-flight runlist: {e}");
            eprintln!("cera::xdna: Drop failed to quiesce in-flight runlist: {e}");
        }
    }
}

// Safety: all access to XRT handles inside `XdnaDecodeState` is serialized
// through `XdnaLlamaModel::npu: Mutex<XdnaDecodeState>`.
unsafe impl Send for XdnaDecodeState {}

/// Llama-3.2-1B model accelerated by the AMD XDNA NPU for decode.
pub struct XdnaLlamaModel {
    cpu: LlamaModel,
    npu: Mutex<XdnaDecodeState>,
    session_gate: ModelSessionGate,
    decode_error: Mutex<Option<CeraError>>,
    state_torn: AtomicBool,
    profile_decode: bool,
}

fn validate_xdna_llama_cpu(cpu: &LlamaModel) -> Result<()> {
    cpu.ensure_xdna_supported()?;

    let cfg = cpu.config();
    ensure!(
        cfg.is_causal,
        "XDNA q4layer kernel requires causal attention"
    );
    ensure!(
        cfg.n_layers == N_LAYERS
            && cfg.hidden_size == HIDDEN_SIZE
            && cfg.intermediate_size == INTERMEDIATE_SIZE
            && cfg.n_heads == N_HEADS
            && cfg.n_kv_heads == N_KV_HEADS
            && cfg.head_dim == HEAD_DIM,
        "XDNA q4layer kernel is compiled for Llama-3.2-1B shape \
         (layers={N_LAYERS}, hidden={HIDDEN_SIZE}, inter={INTERMEDIATE_SIZE}, \
         heads={N_HEADS}/{N_KV_HEADS}, head_dim={HEAD_DIM}); got \
         layers={}, hidden={}, inter={}, heads={}/{}, head_dim={}",
        cfg.n_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    );
    ensure!(
        cfg.vocab_size > 0 && cfg.vocab_size <= HEAD_PADDED_VOCAB,
        "XDNA q4head kernel supports vocab_size <= {HEAD_PADDED_VOCAB}, got {}",
        cfg.vocab_size
    );
    ensure!(
        cfg.scalars.residual == 1.0 && cfg.scalars.attn.is_none_or(|s| (s - 0.125).abs() <= 1e-6),
        "XDNA q4layer kernel requires unit residual scale and standard 1/sqrt(64) attention scale"
    );
    ensure!(
        (cfg.rms_norm_eps - 1e-5).abs() <= 1e-6,
        "XDNA q4layer kernel is compiled with rms_norm_eps=1e-5, got {}",
        cfg.rms_norm_eps
    );

    let embd_ref = cpu.embd_weight_ref();
    ensure!(
        transformer::supports_row_dequant(embd_ref.dtype),
        "XDNA token embedding tensor has unsupported dtype {:?}",
        embd_ref.dtype
    );

    for il in 0..N_LAYERS {
        let stage_refs = cpu.layer_stage_refs(il);
        for (si, (wref, &(rows, cols))) in
            stage_refs.iter().zip(LAYER_STAGE_SHAPES.iter()).enumerate()
        {
            ensure!(
                wref.m == rows && wref.k == cols,
                "XDNA layer {il} stage {si} shape mismatch: expected {rows}x{cols}, got {}x{}",
                wref.m,
                wref.k
            );
            ensure!(
                transformer::supports_row_dequant(wref.dtype),
                "XDNA layer {il} stage {si} weight has unsupported dtype {:?}",
                wref.dtype
            );
        }
    }

    let head_ref = cpu.lm_head_weight_ref();
    ensure!(
        head_ref.k == HIDDEN_SIZE && head_ref.m >= cfg.vocab_size,
        "XDNA LM head shape mismatch: got {}x{}",
        head_ref.m,
        head_ref.k
    );
    ensure!(
        transformer::supports_row_dequant(head_ref.dtype),
        "XDNA LM head weight has unsupported dtype {:?}",
        head_ref.dtype
    );
    Ok(())
}

fn quiesce_in_flight(npu: &mut XdnaDecodeState) -> Result<(), String> {
    if let Some(op) = npu.in_flight {
        match op {
            InFlightXdnaOp::DecodeBucket(bucket_idx) => npu.bucket_runlists[bucket_idx]
                .wait(QUIESCE_TIMEOUT_MS)
                .map_err(|e| {
                    format!("failed to quiesce in-flight XDNA runlist (bucket {bucket_idx}): {e}")
                })?,
            InFlightXdnaOp::PrefillBucket(pf_bucket_idx) => {
                if let Some(pf) = npu.prefill.as_ref() {
                    pf.bucket_runlists[pf_bucket_idx]
                        .wait(QUIESCE_TIMEOUT_MS)
                        .map_err(|e| {
                            format!(
                                "failed to quiesce in-flight XDNA prefill runlist (bucket {pf_bucket_idx}): {e}"
                            )
                        })?;
                }
            }
            InFlightXdnaOp::PrefillHead => {
                if let Some(pf) = npu.prefill.as_ref() {
                    pf.head_run
                        .wait(QUIESCE_TIMEOUT_MS)
                        .map(|_| ())
                        .map_err(|e| {
                            format!("failed to quiesce in-flight XDNA prefill head run: {e}")
                        })?;
                }
            }
        }
        npu.in_flight = None;
    }
    Ok(())
}

pub(crate) fn validate_xdna_kv_compression(
    config: &ModelConfig,
    compression: &KvCompression,
) -> Result<(), CeraError> {
    if matches!(
        compression.resolved_for(config),
        KvCompression::TurboQuant { .. }
    ) {
        return Err(CeraError::Backend(
            "TurboQuant KV compression is not supported by the XDNA backend".into(),
        ));
    }
    Ok(())
}

/// Deferred host-side CPU `InferenceState` mirror work for hybrid XDNA
/// decode models (`XdnaLfm2Model`, `XdnaLfm3Model`).
///
/// Captures the small per-token NPU readbacks (`conv` taps + BF16 K/V rows)
/// using preallocated scratch buffers so `LayerState::Conv` history pushes and
/// BF16-to-F16/F32 cache writes overlap with the next token's `runlist.execute()`.
pub(crate) struct PendingHybridCpuKv {
    pending_pos: usize,
    conv_taps: usize,
    hidden_size: usize,
    kv_dim: usize,
    conv_len: usize,
    attn_len: usize,
    conv: Vec<(usize, Vec<f32>)>,
    attn: Vec<(usize, Vec<u8>, Vec<u8>)>,
}

impl PendingHybridCpuKv {
    pub(crate) fn prepare(
        spare: &mut Option<Self>,
        pending_pos: usize,
        conv_taps: usize,
        hidden_size: usize,
        kv_dim: usize,
    ) -> Self {
        match spare.take() {
            Some(mut p) => {
                p.pending_pos = pending_pos;
                p.conv_taps = conv_taps;
                p.hidden_size = hidden_size;
                p.kv_dim = kv_dim;
                p.conv_len = 0;
                p.attn_len = 0;
                p
            }
            None => Self {
                pending_pos,
                conv_taps,
                hidden_size,
                kv_dim,
                conv_len: 0,
                attn_len: 0,
                conv: Vec::new(),
                attn: Vec::new(),
            },
        }
    }

    pub(crate) fn pending_pos(&self) -> usize {
        self.pending_pos
    }

    pub(crate) fn push_conv_from_kvo(&mut self, layer: usize, kvo_bytes: &[u8]) {
        let idx = self.conv_len;
        let taps = self.conv_taps;
        let hidden = self.hidden_size;
        let needed = taps * hidden;
        if idx >= self.conv.len() {
            self.conv.push((layer, vec![0.0f32; needed]));
        }
        let entry = &mut self.conv[idx];
        entry.0 = layer;
        if entry.1.len() != needed {
            entry.1.resize(needed, 0.0);
        }
        for col in 0..8 {
            for k in 0..taps {
                read_le_f32_slice(
                    kvo_bytes,
                    col * Q4LAYER_TILE_F32 + k * 256,
                    &mut entry.1[k * hidden + col * 256..][..256],
                );
            }
        }
        self.conv_len = idx + 1;
    }

    pub(crate) fn push_attn_bf16(
        &mut self,
        layer: usize,
        kvo_bytes: &[u8],
        n_kv_heads: usize,
        head_dim: usize,
        kvo_stride_bytes: usize,
    ) {
        let idx = self.attn_len;
        let row_bytes = head_dim * 2;
        assert!(
            768 + row_bytes <= kvo_stride_bytes,
            "kvo_stride_bytes ({kvo_stride_bytes}) too small for K@512 and V@768 + {row_bytes} B row"
        );
        assert!(
            n_kv_heads * kvo_stride_bytes <= kvo_bytes.len(),
            "kvo_bytes ({} B) smaller than {n_kv_heads} x {kvo_stride_bytes} B",
            kvo_bytes.len()
        );
        let total_bytes = n_kv_heads * row_bytes;
        if idx >= self.attn.len() {
            self.attn
                .push((layer, vec![0u8; total_bytes], vec![0u8; total_bytes]));
        }
        let entry = &mut self.attn[idx];
        entry.0 = layer;
        if entry.1.len() != total_bytes {
            entry.1.resize(total_bytes, 0);
        }
        if entry.2.len() != total_bytes {
            entry.2.resize(total_bytes, 0);
        }
        for kvh in 0..n_kv_heads {
            let src_base = kvh * kvo_stride_bytes;
            let dst_off = kvh * row_bytes;
            entry.1[dst_off..dst_off + row_bytes]
                .copy_from_slice(&kvo_bytes[src_base + 512..src_base + 512 + row_bytes]);
            entry.2[dst_off..dst_off + row_bytes]
                .copy_from_slice(&kvo_bytes[src_base + 768..src_base + 768 + row_bytes]);
        }
        self.attn_len = idx + 1;
    }

    pub(crate) fn matches_state(&self, state: &InferenceState) -> bool {
        if state.is_compressed() {
            return false;
        }
        let Some(exp) = self.pending_pos.checked_mul(self.kv_dim) else {
            return false;
        };
        self.pending_pos.checked_add(1) == Some(state.seq_len)
            && self.conv[..self.conv_len].iter().all(|(l, _)| {
                *l < state.layers.len() && matches!(&state.layers[*l], LayerState::Conv { .. })
            })
            && self.attn[..self.attn_len].iter().all(|(l, _, _)| {
                *l < state.layers.len()
                    && matches!(
                        &state.layers[*l],
                        LayerState::Attention {
                            key_cache,
                            value_cache,
                            key_cache_f16,
                            value_cache_f16,
                            ..
                        } if if state.kv_f16 {
                            key_cache.is_empty()
                                && value_cache.is_empty()
                                && key_cache_f16.len() == exp
                                && value_cache_f16.len() == exp
                        } else {
                            key_cache_f16.is_empty()
                                && value_cache_f16.is_empty()
                                && key_cache.len() == exp
                                && value_cache.len() == exp
                        }
                    )
            })
    }

    fn recycle(mut p: Self, spare: &mut Option<Self>) {
        p.conv_len = 0;
        p.attn_len = 0;
        *spare = Some(p);
    }

    pub(crate) fn discard(pending: &mut Option<Self>, spare: &mut Option<Self>) {
        if let Some(p) = pending.take() {
            Self::recycle(p, spare);
        }
    }

    #[must_use]
    pub(crate) fn discard_for_truncate(
        pending: &mut Option<Self>,
        spare: &mut Option<Self>,
        state: &mut InferenceState,
        len: usize,
    ) -> bool {
        if len >= state.seq_len {
            return true;
        }
        let Some(p) = pending.take() else {
            return true;
        };
        let matched = p.matches_state(state);
        let pending_pos = p.pending_pos;
        Self::recycle(p, spare);
        if !matched {
            return false;
        }
        // The deferred row sits at `pending_pos == state.seq_len - 1 >= len`, so dropping it
        // rather than flushing it lands directly at the materialized `pending_pos` prefix.
        state.seq_len = pending_pos;
        if state.seq_len == 0 {
            for layer in &mut state.layers {
                if let LayerState::Conv { buffer, history } = layer {
                    buffer.fill(0.0);
                    history.clear();
                }
            }
        } else if state.seq_len == len
            && state.layers.iter().any(
                |layer| matches!(layer, LayerState::Conv { history, .. } if !history.has_pos(len)),
            )
        {
            // Match `InferenceState::truncate_to`'s missing-checkpoint fallback: clear both
            // `Conv` and `Attention` down to 0 so `Attention` KV is never left with zeroed `Conv`.
            state.truncate_to(0);
            return false;
        }
        true
    }

    #[must_use]
    pub(crate) fn flush_into(
        pending: &mut Option<Self>,
        spare: &mut Option<Self>,
        state: &mut InferenceState,
    ) -> bool {
        let Some(p) = pending.take() else {
            return true;
        };
        if !p.matches_state(state) {
            Self::recycle(p, spare);
            return false;
        }
        let next_pos = p.pending_pos + 1;
        for (l, src_buf) in &p.conv[..p.conv_len] {
            if let LayerState::Conv { buffer, history } = &mut state.layers[*l] {
                if buffer.len() != src_buf.len() {
                    buffer.resize(src_buf.len(), 0.0);
                }
                buffer.copy_from_slice(src_buf);
                history.push(next_pos, buffer);
            }
        }
        let use_f16 = state.kv_f16;
        for (l, k_raw, v_raw) in &p.attn[..p.attn_len] {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[*l]
            {
                if use_f16 {
                    key_cache_f16.reserve(p.kv_dim);
                    value_cache_f16.reserve(p.kv_dim);
                    for (&kc, &vc) in k_raw
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(v_raw.as_chunks::<2>().0.iter())
                    {
                        let kf = bf16_to_f32(u16::from_le_bytes(kc));
                        let vf = bf16_to_f32(u16::from_le_bytes(vc));
                        key_cache_f16.push(f32_to_f16(kf));
                        value_cache_f16.push(f32_to_f16(vf));
                    }
                } else {
                    key_cache.reserve(p.kv_dim);
                    value_cache.reserve(p.kv_dim);
                    for (&kc, &vc) in k_raw
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(v_raw.as_chunks::<2>().0.iter())
                    {
                        key_cache.push(bf16_to_f32(u16::from_le_bytes(kc)));
                        value_cache.push(bf16_to_f32(u16::from_le_bytes(vc)));
                    }
                }
            }
        }
        Self::recycle(p, spare);
        true
    }
}

#[derive(Default)]
pub(crate) struct HybridPendingSlot {
    pending: Option<PendingHybridCpuKv>,
    spare: Option<PendingHybridCpuKv>,
}

impl HybridPendingSlot {
    pub(crate) fn prepare(
        &mut self,
        pending_pos: usize,
        conv_taps: usize,
        hidden_size: usize,
        kv_dim: usize,
    ) -> PendingHybridCpuKv {
        PendingHybridCpuKv::prepare(&mut self.spare, pending_pos, conv_taps, hidden_size, kv_dim)
    }

    pub(crate) fn commit(&mut self, next: PendingHybridCpuKv) {
        self.pending = Some(next);
    }

    pub(crate) fn discard(&mut self) {
        PendingHybridCpuKv::discard(&mut self.pending, &mut self.spare);
    }

    #[must_use]
    pub(crate) fn discard_for_truncate(&mut self, state: &mut InferenceState, len: usize) -> bool {
        PendingHybridCpuKv::discard_for_truncate(&mut self.pending, &mut self.spare, state, len)
    }

    #[must_use]
    pub(crate) fn flush_into(&mut self, state: &mut InferenceState) -> bool {
        PendingHybridCpuKv::flush_into(&mut self.pending, &mut self.spare, state)
    }

    pub(crate) fn as_ref(&self) -> Option<&PendingHybridCpuKv> {
        self.pending.as_ref()
    }
}

pub(crate) fn check_hybrid_kv_rewind_state(
    is_causal: bool,
    state_torn: bool,
    pending: Option<&PendingHybridCpuKv>,
    state: &InferenceState,
    len: usize,
) -> Result<(), KvRewindError> {
    check_xdna_kv_rewind_preconditions(is_causal, state_torn, state, len)?;
    if len == state.seq_len {
        return Ok(());
    }
    if let Some(p) = pending {
        if p.pending_pos.checked_add(1) != Some(state.seq_len) {
            return Err(KvRewindError::InvalidCacheLayout {
                layer: 0,
                detail: "deferred NPU KV row does not match the current position",
            });
        }
        if !p.conv[..p.conv_len]
            .iter()
            .all(|&(l, _)| matches!(state.layers.get(l), Some(LayerState::Conv { .. })))
            || !p.attn[..p.attn_len]
                .iter()
                .all(|&(l, _, _)| matches!(state.layers.get(l), Some(LayerState::Attention { .. })))
        {
            return Err(KvRewindError::InvalidCacheLayout {
                layer: 0,
                detail: "deferred NPU KV row references invalid or mismatched layer",
            });
        }
        let expected =
            p.pending_pos
                .checked_mul(p.kv_dim)
                .ok_or(KvRewindError::InvalidCacheLayout {
                    layer: 0,
                    detail: "deferred NPU KV row count overflows cache indexing",
                })?;
        for (layer, layer_state) in state.layers.iter().enumerate() {
            let invalid = |detail| KvRewindError::InvalidCacheLayout { layer, detail };
            match layer_state {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    let (keys, values, inactive_keys, inactive_values) = if state.kv_f16 {
                        (
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                            key_cache.len(),
                            value_cache.len(),
                        )
                    } else {
                        (
                            key_cache.len(),
                            value_cache.len(),
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                        )
                    };
                    if inactive_keys != 0 || inactive_values != 0 {
                        return Err(invalid("inactive precision contains live rows"));
                    }
                    if keys != values || keys != expected {
                        return Err(invalid("key/value rows do not match the current position"));
                    }
                }
                LayerState::Conv { buffer, history } => {
                    if buffer.len() != history.buf_len() {
                        return Err(invalid("convolution buffer and history widths differ"));
                    }
                    if !history.has_pos(len) {
                        return Err(KvRewindError::MissingConvolutionCheckpoint {
                            layer,
                            position: len,
                        });
                    }
                }
                _ => return Err(invalid("unsupported layer state for XDNA hybrid rewind")),
            }
        }
        return Ok(());
    }
    state.check_truncate_to(len)
}

pub(crate) fn check_xdna_kv_rewind_preconditions(
    is_causal: bool,
    state_torn: bool,
    state: &InferenceState,
    len: usize,
) -> Result<(), KvRewindError> {
    if state_torn {
        return Err(KvRewindError::BackendUnsupported);
    }
    if !is_causal || state.lora.as_ref().is_some_and(|l| l.is_classifier()) {
        return Err(KvRewindError::NonCausal);
    }
    if len > state.seq_len {
        return Err(KvRewindError::OutOfBounds {
            requested: len,
            current: state.seq_len,
        });
    }
    if state.is_compressed() {
        return Err(KvRewindError::Compressed);
    }
    Ok(())
}

fn check_kv_rewind_state(
    is_causal: bool,
    state_torn: bool,
    pending_cpu_kv: Option<&(usize, LayerKvPair)>,
    state: &InferenceState,
    len: usize,
) -> Result<(), KvRewindError> {
    check_xdna_kv_rewind_preconditions(is_causal, state_torn, state, len)?;
    if len == state.seq_len {
        return Ok(());
    }
    if let Some((pending_pos, (pk, pv))) = pending_cpu_kv {
        if pending_pos.checked_add(1) != Some(state.seq_len)
            || pk.len() != N_LAYERS
            || pv.len() != N_LAYERS
            || state.layers.len() != N_LAYERS
        {
            return Err(KvRewindError::InvalidCacheLayout {
                layer: 0,
                detail: "deferred NPU KV row does not match the current position",
            });
        }
        let expected =
            pending_pos
                .checked_mul(KV_DIM)
                .ok_or(KvRewindError::InvalidCacheLayout {
                    layer: 0,
                    detail: "deferred NPU KV row count overflows cache indexing",
                })?;
        for (layer, layer_state) in state.layers.iter().enumerate() {
            let invalid = |detail| KvRewindError::InvalidCacheLayout { layer, detail };
            match layer_state {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    let (keys, values, inactive_keys, inactive_values) = if state.kv_f16 {
                        (
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                            key_cache.len(),
                            value_cache.len(),
                        )
                    } else {
                        (
                            key_cache.len(),
                            value_cache.len(),
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                        )
                    };
                    if inactive_keys != 0 || inactive_values != 0 {
                        return Err(invalid("inactive precision contains live rows"));
                    }
                    if keys != values || keys != expected {
                        return Err(invalid("key/value rows do not match the current position"));
                    }
                }
                _ => return Err(invalid("XDNA Llama layer must use attention KV state")),
            }
        }
        return Ok(());
    }
    state.check_truncate_to(len)
}

fn lock_npu_state<'a>(
    npu: &'a Mutex<XdnaDecodeState>,
    state_torn: &AtomicBool,
) -> MutexGuard<'a, XdnaDecodeState> {
    match npu.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            if let Some((_, kv)) = guard.pending_cpu_kv.take() {
                guard.spare_kv = Some(kv);
            }
            state_torn.store(true, Ordering::SeqCst);
            npu.clear_poison();
            tracing::warn!("cera::xdna: NPU mutex poisoned by a prior panic; marking state torn");
            eprintln!("cera::xdna: NPU mutex poisoned by a prior panic; marking state torn");
            guard
        }
    }
}

impl XdnaLlamaModel {
    /// Load a Llama-3.2-1B GGUF onto the AMD XDNA NPU.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let arch = gguf
            .get_str("general.architecture")
            .context("missing general.architecture")?
            .to_lowercase();
        if arch != "llama" {
            bail!(
                "unsupported architecture for AMD XDNA NPU: {arch} \
                 (XDNA backend currently supports Llama-3.2-1B)"
            );
        }
        let cpu = LlamaModel::from_gguf_with_id(gguf, context_size, model_id)
            .context("failed to load CPU Llama weights for XDNA")?;
        validate_xdna_llama_cpu(&cpu).context("unsupported Llama model for XDNA")?;
        let cfg = cpu.config();
        let gguf_ref = cpu.gguf_file();
        let profile_decode = crate::backend::cpu_features::profile_decode_enabled();

        let device = XdnaDevice::open(0).map_err(|e| anyhow::anyhow!("XDNA open failed: {e}"))?;
        let mut bucket_chunks = Vec::with_capacity(Q4LAYER_KV_BUCKETS.len());
        let mut layer_kernels = Vec::with_capacity(Q4LAYER_KV_BUCKETS.len());
        for &(chunks, inst_bytes) in &Q4LAYER_KV_BUCKETS {
            let insts = inst_words(inst_bytes);
            let k = device
                .load_kernel(Q4LAYER_XCLBIN, &insts)
                .map_err(|e| anyhow::anyhow!("XDNA load q4layer (c={chunks}) failed: {e}"))?;
            bucket_chunks.push(chunks);
            layer_kernels.push(k);
        }
        let head_insts = inst_words(Q4HEAD_INSTS);
        let head_kernel = device
            .load_kernel(Q4HEAD_XCLBIN, &head_insts)
            .map_err(|e| anyhow::anyhow!("XDNA load q4head.xclbin failed: {e}"))?;

        let mut prefill_bucket_chunks = Vec::with_capacity(Q4PREFILL_LAYER_KV_BUCKETS.len());
        let mut prefill_kernels = Vec::with_capacity(Q4PREFILL_LAYER_KV_BUCKETS.len());
        for &(chunks, inst_bytes) in &Q4PREFILL_LAYER_KV_BUCKETS {
            let insts = inst_words(inst_bytes);
            let k = device
                .load_kernel(Q4PREFILL_LAYER_XCLBIN, &insts)
                .map_err(|e| {
                    anyhow::anyhow!("XDNA load q4prefill_layer (c={chunks}) failed: {e}")
                })?;
            prefill_bucket_chunks.push(chunks);
            prefill_kernels.push(k);
        }
        let prefill_alloc_kernel = prefill_kernels
            .last()
            .expect("Q4PREFILL_LAYER_KV_BUCKETS is non-empty");

        let mut stage_sizes = [0usize; 7];
        let mut layer_w_bytes = 0usize;
        for (si, &(rows, cols)) in LAYER_STAGE_SHAPES.iter().enumerate() {
            let sz = stage_bytes(rows, cols).map_err(|e| anyhow::anyhow!(e))?;
            stage_sizes[si] = sz;
            layer_w_bytes += sz;
        }
        let qkv_bytes = stage_sizes[0] + stage_sizes[1] + stage_sizes[2];
        let o_off = qkv_bytes;
        let gu_off = o_off + stage_sizes[3];
        let down_off = gu_off + stage_sizes[4] + stage_sizes[5];

        let alloc_layer_kernel = layer_kernels
            .last()
            .expect("Q4LAYER_KV_BUCKETS is non-empty");
        let mut layer_bo_w = Vec::with_capacity(N_LAYERS);
        let mut layer_bo_y = Vec::with_capacity(N_LAYERS);
        let mut layer_bo_kv = Vec::with_capacity(N_LAYERS);
        let mut prefill_bo_w = Vec::with_capacity(N_LAYERS);
        let mut prefill_bo_y = Vec::with_capacity(N_LAYERS);

        let mut q_tmp = vec![0u8; stage_sizes[0]];
        let mut k_tmp = vec![0u8; stage_sizes[1]];
        let mut v_tmp = vec![0u8; stage_sizes[2]];
        let mut gate_tmp = vec![0u8; stage_sizes[4]];
        let mut up_tmp = vec![0u8; stage_sizes[5]];

        let pack_one_stage = |dst: &mut [u8],
                              wref: &transformer::WeightRef,
                              rows: usize,
                              cols: usize,
                              il: usize,
                              si: usize|
         -> Result<()> {
            if wref.dtype == DType::Q4_0 {
                let src = transformer::weight_data(gguf_ref, wref);
                pack_q4_0_stage(dst, src, rows, cols)
                    .map_err(|e| anyhow::anyhow!("XDNA layer {il} stage {si} repack: {e}"))?;
            } else {
                let chunk_rows = HEAD_REPACK_CHUNK_ROWS.min(rows);
                let chunk_packed = stage_bytes(chunk_rows, cols)
                    .map_err(|e| anyhow::anyhow!("XDNA layer {il} stage {si}: {e}"))?;
                let mut f32_chunk = vec![0.0f32; chunk_rows * cols];
                for r0 in (0..rows).step_by(chunk_rows) {
                    for r in 0..chunk_rows {
                        transformer::dequantize_row_into(
                            gguf_ref,
                            wref,
                            r0 + r,
                            &mut f32_chunk[r * cols..(r + 1) * cols],
                        );
                    }
                    let dst_off = (r0 / chunk_rows) * chunk_packed;
                    pack_f32_stage(
                        &mut dst[dst_off..dst_off + chunk_packed],
                        &f32_chunk,
                        chunk_rows,
                        cols,
                    )
                    .map_err(|e| anyhow::anyhow!("XDNA layer {il} stage {si} f32 repack: {e}"))?;
                }
            }
            Ok(())
        };

        for il in 0..N_LAYERS {
            let mut bo_w = alloc_layer_kernel
                .alloc_bo(layer_w_bytes, 3)
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_w alloc: {e}"))?;
            let mut bo_y = alloc_layer_kernel
                .alloc_bo(Q4LAYER_ACT_BYTES, 4)
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_y alloc: {e}"))?;
            let mut bo_kv = alloc_layer_kernel
                .alloc_bo(Q4LAYER_KV_BYTES, 6)
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_kv alloc: {e}"))?;

            let stage_refs = cpu.layer_stage_refs(il);
            let w_slice = bo_w.as_mut_slice();
            pack_one_stage(
                &mut q_tmp,
                stage_refs[0],
                LAYER_STAGE_SHAPES[0].0,
                LAYER_STAGE_SHAPES[0].1,
                il,
                0,
            )?;
            pack_one_stage(
                &mut k_tmp,
                stage_refs[1],
                LAYER_STAGE_SHAPES[1].0,
                LAYER_STAGE_SHAPES[1].1,
                il,
                1,
            )?;
            pack_one_stage(
                &mut v_tmp,
                stage_refs[2],
                LAYER_STAGE_SHAPES[2].0,
                LAYER_STAGE_SHAPES[2].1,
                il,
                2,
            )?;
            interleave_worker_stages(
                &mut w_slice[..qkv_bytes],
                &[&q_tmp, &k_tmp, &v_tmp],
                HIDDEN_SIZE,
            )
            .map_err(|e| anyhow::anyhow!("XDNA layer {il} qkv interleave: {e}"))?;

            pack_one_stage(
                &mut w_slice[o_off..gu_off],
                stage_refs[3],
                LAYER_STAGE_SHAPES[3].0,
                LAYER_STAGE_SHAPES[3].1,
                il,
                3,
            )?;

            pack_one_stage(
                &mut gate_tmp,
                stage_refs[4],
                LAYER_STAGE_SHAPES[4].0,
                LAYER_STAGE_SHAPES[4].1,
                il,
                4,
            )?;
            pack_one_stage(
                &mut up_tmp,
                stage_refs[5],
                LAYER_STAGE_SHAPES[5].0,
                LAYER_STAGE_SHAPES[5].1,
                il,
                5,
            )?;
            interleave_worker_stages(
                &mut w_slice[gu_off..down_off],
                &[&gate_tmp, &up_tmp],
                HIDDEN_SIZE,
            )
            .map_err(|e| anyhow::anyhow!("XDNA layer {il} gate+up interleave: {e}"))?;

            pack_one_stage(
                &mut w_slice[down_off..],
                stage_refs[6],
                LAYER_STAGE_SHAPES[6].0,
                LAYER_STAGE_SHAPES[6].1,
                il,
                6,
            )?;

            let mut pf_bo_w = prefill_alloc_kernel
                .alloc_bo(Q4PREFILL_W_BYTES, 3)
                .map_err(|e| anyhow::anyhow!("XDNA prefill layer {il} bo_w alloc: {e}"))?;
            {
                let pf_w = pf_bo_w.as_mut_slice();
                let mut off = 0usize;
                for src in [
                    q_tmp.as_slice(),
                    k_tmp.as_slice(),
                    v_tmp.as_slice(),
                    &w_slice[o_off..gu_off],
                    gate_tmp.as_slice(),
                    up_tmp.as_slice(),
                    &w_slice[down_off..],
                ] {
                    pf_w[off..off + src.len()].copy_from_slice(src);
                    off += src.len();
                }
                debug_assert_eq!(off, Q4PREFILL_W_BYTES);
            }

            bo_w.sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_w sync: {e}"))?;

            init_q4layer_bo_y_norms(
                bo_y.as_mut_slice(),
                cpu.attn_norm_slice(il),
                cpu.ffn_norm_slice(il),
            );
            bo_y.sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_y sync: {e}"))?;
            bo_kv.as_mut_slice().fill(0);
            bo_kv
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_kv sync: {e}"))?;

            pf_bo_w
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA prefill layer {il} bo_w sync: {e}"))?;

            let mut pf_bo_y = prefill_alloc_kernel
                .alloc_bo(Q4PREFILL_ACT_BYTES, 4)
                .map_err(|e| anyhow::anyhow!("XDNA prefill layer {il} bo_y alloc: {e}"))?;
            init_q4prefill_bo_y_norms(
                pf_bo_y.as_mut_slice(),
                cpu.attn_norm_slice(il),
                cpu.ffn_norm_slice(il),
            );
            pf_bo_y
                .sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA prefill layer {il} bo_y sync: {e}"))?;

            layer_bo_w.push(bo_w);
            layer_bo_y.push(bo_y);
            layer_bo_kv.push(bo_kv);
            prefill_bo_w.push(pf_bo_w);
            prefill_bo_y.push(pf_bo_y);
        }

        // Allocate and repack the 131,072 x 2,048 LM head in 256-row chunks.
        let head_w_bytes =
            stage_bytes(HEAD_PADDED_VOCAB, HIDDEN_SIZE).map_err(|e| anyhow::anyhow!(e))?;
        let mut head_bo_w = head_kernel
            .alloc_bo(head_w_bytes, 3)
            .map_err(|e| anyhow::anyhow!("XDNA head bo_w alloc: {e}"))?;
        let mut head_bo_y = head_kernel
            .alloc_bo(HEAD_Y_BYTES, 4)
            .map_err(|e| anyhow::anyhow!("XDNA head bo_y alloc: {e}"))?;

        head_bo_y.as_mut_slice().fill(0);
        write_le_f32_slice(
            head_bo_y.as_mut_slice(),
            Q4LAYER_ATTN_NORM_W,
            cpu.output_norm_slice(),
        );
        write_le_i32(head_bo_y.as_mut_slice(), Q4LAYER_ATTN_N_OFF, -1);
        write_le_f32_slice(
            head_bo_y.as_mut_slice(),
            Q4LAYER_ATTN_N_OFF + 1,
            &[1.0, 0.0],
        );
        head_bo_y
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("XDNA head bo_y sync: {e}"))?;

        let head_ref = cpu.lm_head_weight_ref();

        let chunk_packed_bytes =
            stage_bytes(HEAD_REPACK_CHUNK_ROWS, HIDDEN_SIZE).map_err(|e| anyhow::anyhow!(e))?;
        let head_w_slice = head_bo_w.as_mut_slice();
        if head_ref.dtype == DType::Q4_0 && head_ref.m == cfg.vocab_size {
            let src = transformer::weight_data(gguf_ref, head_ref);
            let full_rows = (cfg.vocab_size / HEAD_REPACK_CHUNK_ROWS) * HEAD_REPACK_CHUNK_ROWS;
            if full_rows > 0 {
                pack_q4_0_stage(&mut head_w_slice[..], src, full_rows, HIDDEN_SIZE)
                    .map_err(|e| anyhow::anyhow!("XDNA head Q4_0 repack: {e}"))?;
            }
            let mut f32_chunk = vec![0.0f32; HEAD_REPACK_CHUNK_ROWS * HIDDEN_SIZE];
            for r0 in (full_rows..HEAD_PADDED_VOCAB).step_by(HEAD_REPACK_CHUNK_ROWS) {
                f32_chunk.fill(0.0);
                for r in 0..HEAD_REPACK_CHUNK_ROWS {
                    let row = r0 + r;
                    if row < cfg.vocab_size {
                        transformer::dequantize_row_into(
                            gguf_ref,
                            head_ref,
                            row,
                            &mut f32_chunk[r * HIDDEN_SIZE..(r + 1) * HIDDEN_SIZE],
                        );
                    }
                }
                let dst_off = (r0 / HEAD_REPACK_CHUNK_ROWS) * chunk_packed_bytes;
                pack_f32_stage(
                    &mut head_w_slice[dst_off..dst_off + chunk_packed_bytes],
                    &f32_chunk,
                    HEAD_REPACK_CHUNK_ROWS,
                    HIDDEN_SIZE,
                )
                .map_err(|e| anyhow::anyhow!("XDNA head tail repack: {e}"))?;
            }
        } else {
            let mut f32_chunk = vec![0.0f32; HEAD_REPACK_CHUNK_ROWS * HIDDEN_SIZE];
            for r0 in (0..HEAD_PADDED_VOCAB).step_by(HEAD_REPACK_CHUNK_ROWS) {
                f32_chunk.fill(0.0);
                for r in 0..HEAD_REPACK_CHUNK_ROWS {
                    let row = r0 + r;
                    if row < cfg.vocab_size {
                        transformer::dequantize_row_into(
                            gguf_ref,
                            head_ref,
                            row,
                            &mut f32_chunk[r * HIDDEN_SIZE..(r + 1) * HIDDEN_SIZE],
                        );
                    }
                }
                let dst_off = (r0 / HEAD_REPACK_CHUNK_ROWS) * chunk_packed_bytes;
                pack_f32_stage(
                    &mut head_w_slice[dst_off..dst_off + chunk_packed_bytes],
                    &f32_chunk,
                    HEAD_REPACK_CHUNK_ROWS,
                    HIDDEN_SIZE,
                )
                .map_err(|e| anyhow::anyhow!("XDNA head chunk repack: {e}"))?;
            }
        }
        head_bo_w
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("XDNA head bo_w sync: {e}"))?;

        let mut layer_runs = Vec::with_capacity(layer_kernels.len());
        let mut head_runs = Vec::with_capacity(layer_kernels.len());
        let mut bucket_runlists = Vec::with_capacity(layer_kernels.len());

        for (bi, l_kernel) in layer_kernels.iter().enumerate() {
            let chunks = bucket_chunks[bi];
            let mut b_runs = Vec::with_capacity(N_LAYERS);
            for il in 0..N_LAYERS {
                let y_next = if il + 1 < N_LAYERS {
                    &layer_bo_y[il + 1]
                } else {
                    &head_bo_y
                };
                let run = l_kernel
                    .create_run(&[&layer_bo_w[il], &layer_bo_y[il], y_next, &layer_bo_kv[il]])
                    .map_err(|e| anyhow::anyhow!("XDNA layer {il} (c={chunks}) create_run: {e}"))?;
                b_runs.push(run);
            }
            let h_run = head_kernel
                .create_run(&[&head_bo_w, &head_bo_y, &head_bo_y, &layer_bo_kv[0]])
                .map_err(|e| anyhow::anyhow!("XDNA head (c={chunks}) create_run: {e}"))?;

            let mut runlist = l_kernel
                .create_runlist()
                .map_err(|e| anyhow::anyhow!("XDNA create_runlist (c={chunks}): {e}"))?;
            for (il, run) in b_runs.iter().enumerate() {
                runlist.add(run).map_err(|e| {
                    anyhow::anyhow!("XDNA runlist (c={chunks}) add layer {il}: {e}")
                })?;
            }
            runlist
                .add(&h_run)
                .map_err(|e| anyhow::anyhow!("XDNA runlist (c={chunks}) add head: {e}"))?;

            layer_runs.push(b_runs);
            head_runs.push(h_run);
            bucket_runlists.push(runlist);
        }

        let mut pf_bo_out = prefill_alloc_kernel
            .alloc_bo(Q4PREFILL_ACT_BYTES, 5)
            .map_err(|e| anyhow::anyhow!("XDNA prefill bo_out alloc: {e}"))?;
        pf_bo_out.as_mut_slice().fill(0);
        pf_bo_out
            .sync_to_device()
            .map_err(|e| anyhow::anyhow!("XDNA prefill bo_out sync: {e}"))?;

        let pf_head_run = head_kernel
            .create_run(&[&head_bo_w, &head_bo_y, &head_bo_y, &layer_bo_kv[0]])
            .map_err(|e| anyhow::anyhow!("XDNA prefill head create_run: {e}"))?;

        let mut prefill_layer_runs = Vec::with_capacity(prefill_kernels.len());
        let mut prefill_bucket_runlists = Vec::with_capacity(prefill_kernels.len());
        for (bi, pf_kernel) in prefill_kernels.iter().enumerate() {
            let chunks = prefill_bucket_chunks[bi];
            let mut b_runs = Vec::with_capacity(N_LAYERS);
            for il in 0..N_LAYERS {
                let y_next = if il + 1 < N_LAYERS {
                    &prefill_bo_y[il + 1]
                } else {
                    &pf_bo_out
                };
                let run = pf_kernel
                    .create_run(&[
                        &prefill_bo_w[il],
                        &prefill_bo_y[il],
                        y_next,
                        &layer_bo_kv[il],
                    ])
                    .map_err(|e| {
                        anyhow::anyhow!("XDNA prefill layer {il} (c={chunks}) create_run: {e}")
                    })?;
                b_runs.push(run);
            }
            let mut runlist = pf_kernel
                .create_runlist()
                .map_err(|e| anyhow::anyhow!("XDNA prefill create_runlist (c={chunks}): {e}"))?;
            for (il, run) in b_runs.iter().enumerate() {
                runlist.add(run).map_err(|e| {
                    anyhow::anyhow!("XDNA prefill runlist (c={chunks}) add layer {il}: {e}")
                })?;
            }
            prefill_layer_runs.push(b_runs);
            prefill_bucket_runlists.push(runlist);
        }

        let prefill = Some(Box::new(XdnaPrefillState {
            bucket_chunks: prefill_bucket_chunks,
            bucket_runlists: prefill_bucket_runlists,
            head_run: pf_head_run,
            _layer_runs: prefill_layer_runs,
            bo_out: pf_bo_out,
            bo_y: prefill_bo_y,
            _bo_w: prefill_bo_w,
            _kernels: prefill_kernels,
        }));

        Ok(Self {
            cpu,
            npu: Mutex::new(XdnaDecodeState {
                bucket_chunks,
                bucket_runlists,
                _head_runs: head_runs,
                _layer_runs: layer_runs,
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_kv,
                layer_bo_y,
                _layer_bo_w: layer_bo_w,
                _head_kernel: head_kernel,
                _layer_kernels: layer_kernels,
                prefill,
                _device: device,
                in_flight: None,
                synced_kv_len: 0,
                cached_rope: None,
                pending_cpu_kv: None,
                spare_kv: Some((
                    vec![[0.0f32; KV_DIM]; N_LAYERS],
                    vec![[0.0f32; KV_DIM]; N_LAYERS],
                )),
                q8_scales: vec![0.0f32; HIDDEN_SIZE / 32],
                q8_quants: vec![0i8; HIDDEN_SIZE],
            }),
            session_gate: ModelSessionGate::default(),
            decode_error: Mutex::new(None),
            state_torn: AtomicBool::new(false),
            profile_decode,
        })
    }

    #[cfg(test)]
    fn from_parts_for_test(cpu: LlamaModel, npu: XdnaDecodeState) -> Self {
        Self {
            cpu,
            npu: Mutex::new(npu),
            session_gate: ModelSessionGate::default(),
            decode_error: Mutex::new(None),
            state_torn: AtomicBool::new(false),
            profile_decode: false,
        }
    }

    fn lock_npu(&self) -> MutexGuard<'_, XdnaDecodeState> {
        lock_npu_state(&self.npu, &self.state_torn)
    }

    fn discard_pending_cpu_kv(npu: &mut XdnaDecodeState) {
        if let Some((_, kv)) = npu.pending_cpu_kv.take() {
            npu.spare_kv = Some(kv);
        }
        npu.synced_kv_len = 0;
    }

    fn discard_or_flush_pending_kv(
        npu: &mut XdnaDecodeState,
        state: &mut InferenceState,
        pos: usize,
    ) {
        if pos == 0 {
            Self::discard_pending_cpu_kv(npu);
        } else {
            Self::flush_pending_cpu_kv(npu, state);
            npu.synced_kv_len = npu.synced_kv_len.min(pos);
        }
    }

    fn flush_pending_cpu_kv(npu: &mut XdnaDecodeState, state: &mut InferenceState) {
        let Some((pending_pos, (pending_k, pending_v))) = npu.pending_cpu_kv.take() else {
            return;
        };
        let expected = pending_pos.checked_mul(KV_DIM);
        let state_matches = expected.is_some_and(|exp| {
            pending_pos.checked_add(1) == Some(state.seq_len)
                && pending_k.len() >= N_LAYERS
                && pending_v.len() >= N_LAYERS
                && state.layers.len() >= N_LAYERS
                && (0..N_LAYERS).all(|l| {
                    pending_k[l].len() == KV_DIM
                        && pending_v[l].len() == KV_DIM
                        && matches!(
                            &state.layers[l],
                            LayerState::Attention {
                                key_cache,
                                value_cache,
                                key_cache_f16,
                                value_cache_f16,
                                ..
                            } if if state.kv_f16 {
                                key_cache_f16.len() == exp && value_cache_f16.len() == exp
                            } else {
                                key_cache.len() == exp && value_cache.len() == exp
                            }
                        )
                })
        });
        if !state_matches {
            npu.synced_kv_len = 0;
            npu.spare_kv = Some((pending_k, pending_v));
            return;
        }
        let use_f16 = state.kv_f16;
        for l in 0..N_LAYERS {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut state.layers[l]
            {
                if use_f16 {
                    key_cache_f16.extend(pending_k[l].iter().copied().map(f32_to_f16));
                    value_cache_f16.extend(pending_v[l].iter().copied().map(f32_to_f16));
                } else {
                    key_cache.extend_from_slice(&pending_k[l]);
                    value_cache.extend_from_slice(&pending_v[l]);
                }
            }
        }
        npu.spare_kv = Some((pending_k, pending_v));
    }

    fn compute_rope_pair(&self, pos: usize) -> ([f32; HEAD_DIM / 2], [f32; HEAD_DIM / 2]) {
        let cfg = self.cpu.config();
        let mut unit = [0.0f32; HEAD_DIM];
        for i in 0..HEAD_DIM / 2 {
            unit[2 * i] = 1.0;
        }
        cpu::rope_norm(
            &mut unit,
            &mut [],
            pos,
            1,
            0,
            HEAD_DIM,
            cfg.rope_theta,
            self.cpu.rope_freqs_slice(),
        );
        let mut rope_cos = [0.0f32; HEAD_DIM / 2];
        let mut rope_sin = [0.0f32; HEAD_DIM / 2];
        for i in 0..HEAD_DIM / 2 {
            rope_cos[i] = unit[2 * i];
            rope_sin[i] = unit[2 * i + 1];
        }
        (rope_cos, rope_sin)
    }

    fn compute_rope_cs(&self, pos: usize) -> [f32; 128] {
        let (rope_cos, rope_sin) = self.compute_rope_pair(pos);
        pack_q4layer_rope_cs(&rope_cos, &rope_sin)
    }

    fn sync_cpu_kv_to_npu(
        &self,
        npu: &mut XdnaDecodeState,
        state: &mut InferenceState,
        target_pos: usize,
    ) -> Result<(), SyncErr> {
        if state.is_compressed() {
            return Err(SyncErr::Input(
                "TurboQuant KV compression is not supported by the XDNA backend".to_string(),
            ));
        }
        if state.layers.len() < N_LAYERS {
            return Err(SyncErr::Input(format!(
                "InferenceState has {} layers, expected at least {N_LAYERS}",
                state.layers.len()
            )));
        }
        Self::flush_pending_cpu_kv(npu, state);
        if target_pos < npu.synced_kv_len {
            npu.synced_kv_len = target_pos;
        }
        if npu.synced_kv_len >= target_pos {
            return Ok(());
        }
        let needed = target_pos
            .checked_mul(KV_DIM)
            .ok_or_else(|| SyncErr::Input("target_pos overflows KV cache length".to_string()))?;
        let start = npu.synced_kv_len;
        let use_f16 = state.kv_f16;
        // Validate every layer's KV cache shape before writing BOs or syncing so
        // malformed inputs reject cleanly without tearing NPU state.
        for l in 0..N_LAYERS {
            match &state.layers[l] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    if use_f16 {
                        if key_cache_f16.len() < needed || value_cache_f16.len() < needed {
                            return Err(SyncErr::Input(format!(
                                "layer {l} f16 KV cache has {}/{} elements, expected at least {needed}",
                                key_cache_f16.len(),
                                value_cache_f16.len()
                            )));
                        }
                    } else if key_cache.len() < needed || value_cache.len() < needed {
                        return Err(SyncErr::Input(format!(
                            "layer {l} f32 KV cache has {}/{} elements, expected at least {needed}",
                            key_cache.len(),
                            value_cache.len()
                        )));
                    }
                }
                _ => {
                    return Err(SyncErr::Input(format!(
                        "layer {l} does not have an Attention KV state for XDNA"
                    )));
                }
            }
        }
        for l in 0..N_LAYERS {
            let kv_bytes = npu.layer_bo_kv[l].as_mut_slice();
            let LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } = &state.layers[l]
            else {
                unreachable!("validated Attention layers above");
            };
            for t in start..target_pos {
                for kvh in 0..N_KV_HEADS {
                    let src_fl = t * KV_DIM + kvh * HEAD_DIM;
                    if use_f16 {
                        let mut kh = [0.0f32; HEAD_DIM];
                        let mut vh = [0.0f32; HEAD_DIM];
                        for d in 0..HEAD_DIM {
                            kh[d] = crate::quant::f16_to_f32(key_cache_f16[src_fl + d]);
                            vh[d] = crate::quant::f16_to_f32(value_cache_f16[src_fl + d]);
                        }
                        write_q4layer_kv_bf16(kv_bytes, kvh, t, &kh, &vh);
                    } else {
                        write_q4layer_kv_bf16(
                            kv_bytes,
                            kvh,
                            t,
                            &key_cache[src_fl..src_fl + HEAD_DIM],
                            &value_cache[src_fl..src_fl + HEAD_DIM],
                        );
                    }
                }
            }
            npu.layer_bo_kv[l]
                .sync_range_to_device(Q4LAYER_KV_BYTES, 0)
                .map_err(|e| SyncErr::Device(format!("layer {l} KV cache sync to device: {e}")))?;
        }
        npu.synced_kv_len = target_pos;
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

    fn record_device_fault_err(&self, err: CeraError, out_len: usize) -> Vec<f32> {
        self.state_torn.store(true, Ordering::SeqCst);
        tracing::error!("cera::xdna: {err}");
        eprintln!("cera::xdna: {err}");
        record_first_fault(&self.decode_error, err);
        vec![0.0f32; out_len]
    }

    fn record_device_fault(&self, msg: String, out_len: usize) -> Vec<f32> {
        self.record_device_fault_err(CeraError::Backend(msg), out_len)
    }

    fn forward_npu_decode_into(
        &self,
        token: u32,
        pos: usize,
        state: &mut InferenceState,
        logits_out: &mut [f32],
    ) -> Result<(), Vec<f32>> {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            return Err(self.record_device_fault(
                "XDNA NPU state torn after a fault; reset the session".to_string(),
                vocab_size,
            ));
        }
        if let Err(e) = quiesce_in_flight(&mut npu) {
            return Err(
                self.record_device_fault(format!("forward failed to quiesce NPU: {e}"), vocab_size)
            );
        }
        if pos != state.seq_len {
            return Err(self.reject_input(
                format!(
                    "forward pos ({pos}) does not match state.seq_len ({})",
                    state.seq_len
                ),
                vocab_size,
            ));
        }
        if pos >= cfg.max_seq_len {
            return Err(self.reject_input_err(
                CeraError::ContextOverflow {
                    max_seq_len: cfg.max_seq_len as u32,
                    by: (pos + 1 - cfg.max_seq_len) as u32,
                },
                vocab_size,
            ));
        }
        if pos == 0 {
            Self::discard_pending_cpu_kv(&mut npu);
        }
        if npu.synced_kv_len != pos
            && let Err(e) = self.sync_cpu_kv_to_npu(&mut npu, state, pos)
        {
            return Err(match e {
                SyncErr::Input(msg) => self.reject_input(msg, vocab_size),
                SyncErr::Device(msg) => self.record_device_fault(msg, vocab_size),
            });
        }

        let hidden = self.cpu.embed_token(token, "forward");

        let rope_cs = match npu.cached_rope {
            Some((cached_pos, cs)) if cached_pos == pos => cs,
            _ => self.compute_rope_cs(pos),
        };

        let (bucket_idx, skip_kv) = select_q4layer_kv_bucket(&npu.bucket_chunks, pos);

        let prof = self.profile_decode;
        let tc0 = prof.then(crate::time::Instant::now);
        for l in 0..N_LAYERS {
            let yl = npu.layer_bo_y[l].as_mut_slice();
            if l == 0 {
                write_le_f32_slice(yl, 0, &hidden);
            }
            write_q4layer_step_params(yl, &rope_cs, pos, skip_kv);
            if l == 0
                && let Err(e) = npu.layer_bo_y[0].sync_range_to_device(HIDDEN_SIZE * 4, 0)
            {
                return Err(self.record_device_fault(
                    format!("layer 0 hidden sync to device: {e}"),
                    vocab_size,
                ));
            }
            if let Err(e) = npu.layer_bo_y[l].sync_range_to_device(132 * 4, Q4LAYER_ROPE_CS * 4) {
                return Err(self.record_device_fault(
                    format!("layer {l} bo_y sync to device: {e}"),
                    vocab_size,
                ));
            }
        }
        let t_crit_sync = tc0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let t0 = prof.then(crate::time::Instant::now);
        npu.in_flight = Some(InFlightXdnaOp::DecodeBucket(bucket_idx));
        if let Err(e) = npu.bucket_runlists[bucket_idx].execute() {
            npu.in_flight = None;
            return Err(self.record_device_fault(format!("token runlist execute: {e}"), vocab_size));
        }
        // Overlap previous token's f16 CPU KV cache extension and next token's
        // RoPE cos/sin table while the NPU runs.
        Self::flush_pending_cpu_kv(&mut npu, state);
        npu.cached_rope = Some((pos + 1, self.compute_rope_cs(pos + 1)));
        let (mut pending_k, mut pending_v) = npu.spare_kv.take().unwrap_or_else(|| {
            (
                vec![[0.0f32; KV_DIM]; N_LAYERS],
                vec![[0.0f32; KV_DIM]; N_LAYERS],
            )
        });
        if let Err(e) = npu.bucket_runlists[bucket_idx].wait(QUIESCE_TIMEOUT_MS) {
            npu.spare_kv = Some((pending_k, pending_v));
            return Err(self.record_device_fault(format!("token runlist wait: {e}"), vocab_size));
        }
        npu.in_flight = None;
        let t_npu_wait = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let tkv0 = prof.then(crate::time::Instant::now);
        let mut hidden_dump =
            transformer::oracle_dump::is_active().then(|| vec![0.0f32; HIDDEN_SIZE]);
        for l in 0..N_LAYERS {
            if let Some(hidden) = hidden_dump.as_mut() {
                {
                    let next_bo = if l + 1 < N_LAYERS {
                        &mut npu.layer_bo_y[l + 1]
                    } else {
                        &mut npu.head_bo_y
                    };
                    if let Err(e) = next_bo.sync_range_from_device(HIDDEN_SIZE * 4, 0) {
                        npu.spare_kv = Some((pending_k, pending_v));
                        return Err(self.record_device_fault(
                            format!("layer {l} out sync from device: {e}"),
                            vocab_size,
                        ));
                    }
                    read_le_f32_slice(next_bo.as_slice(), 0, hidden);
                }
                transformer::oracle_dump::record(&format!("l_out-{l}"), hidden);
                if l + 1 == N_LAYERS {
                    let mut normed = vec![0.0f32; HIDDEN_SIZE];
                    let XdnaDecodeState {
                        ref mut q8_scales,
                        ref mut q8_quants,
                        ..
                    } = *npu;
                    cpu::rmsnorm_and_quantize_q8_0(
                        hidden,
                        self.cpu.output_norm_slice(),
                        cfg.rms_norm_eps,
                        q8_scales,
                        q8_quants,
                        Some(&mut normed),
                    );
                    transformer::oracle_dump::record("result_norm", &normed);
                }
            }

            if let Err(e) =
                npu.layer_bo_y[l].sync_range_from_device(KV_DIM * 4, 2 * Q4LAYER_ACT_SLOT * 4)
            {
                npu.spare_kv = Some((pending_k, pending_v));
                return Err(self.record_device_fault(
                    format!("layer {l} slot 2 sync from device: {e}"),
                    vocab_size,
                ));
            }
            if let Err(e) =
                npu.layer_bo_y[l].sync_range_from_device(KV_DIM * 4, 3 * Q4LAYER_ACT_SLOT * 4)
            {
                npu.spare_kv = Some((pending_k, pending_v));
                return Err(self.record_device_fault(
                    format!("layer {l} slot 3 sync from device: {e}"),
                    vocab_size,
                ));
            }
            let k_cur = &mut pending_k[l];
            let v_cur = &mut pending_v[l];
            {
                let yl = npu.layer_bo_y[l].as_slice();
                read_le_f32_slice(yl, 2 * Q4LAYER_ACT_SLOT, k_cur);
                read_le_f32_slice(yl, 3 * Q4LAYER_ACT_SLOT, v_cur);
            }
            let mut k_offs = [0usize; N_KV_HEADS];
            {
                let kv_mut = npu.layer_bo_kv[l].as_mut_slice();
                for (kvh, k_off) in k_offs.iter_mut().enumerate() {
                    let src_fl = kvh * HEAD_DIM;
                    *k_off = write_q4layer_kv_bf16(
                        kv_mut,
                        kvh,
                        pos,
                        &k_cur[src_fl..src_fl + HEAD_DIM],
                        &v_cur[src_fl..src_fl + HEAD_DIM],
                    );
                }
            }
            for &k_byte_off in &k_offs {
                if let Err(e) = npu.layer_bo_kv[l]
                    .sync_range_to_device(Q4LAYER_SILU_HALF * 4 + HEAD_DIM * 2, k_byte_off)
                {
                    npu.spare_kv = Some((pending_k, pending_v));
                    return Err(self.record_device_fault(
                        format!("layer {l} kv sync to device: {e}"),
                        vocab_size,
                    ));
                }
            }
        }
        let t_kv_sync = tkv0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let th0 = prof.then(crate::time::Instant::now);
        if let Err(e) = npu
            .head_bo_y
            .sync_range_from_device(vocab_size * 4, Q4LAYER_TILE_F32 * 4)
        {
            npu.spare_kv = Some((pending_k, pending_v));
            return Err(
                self.record_device_fault(format!("head bo_y sync from device: {e}"), vocab_size)
            );
        }
        let t_head_sync = th0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        if prof && !(20..520).contains(&pos) {
            eprintln!(
                "xdna prof pos={pos}: npu_wait={t_npu_wait:.0}us crit_sync={t_crit_sync:.0}us kv_sync={t_kv_sync:.0}us head_sync={t_head_sync:.0}us"
            );
        }

        read_le_f32_slice(
            npu.head_bo_y.as_slice(),
            Q4LAYER_TILE_F32,
            &mut logits_out[..vocab_size],
        );
        if cfg.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits_out[..vocab_size], 1.0 / cfg.scalars.logit);
        }

        state.seq_len = pos + 1;
        npu.pending_cpu_kv = Some((pos, (pending_k, pending_v)));
        if pos + 1 >= MAX_KV_CACHE_TOKENS {
            Self::flush_pending_cpu_kv(&mut npu, state);
        }
        npu.synced_kv_len = pos + 1;
        transformer::oracle_dump::record("result_output", &logits_out[..vocab_size]);
        Ok(())
    }
}

impl Model for XdnaLlamaModel {
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        let lease = self.session_gate.try_acquire()?;
        let mut npu = self.lock_npu();
        if let Err(e) = quiesce_in_flight(&mut npu) {
            self.state_torn.store(true, Ordering::SeqCst);
            return Err(CeraError::Backend(e));
        }
        if let Some((_, kv)) = npu.pending_cpu_kv.take() {
            npu.spare_kv = Some(kv);
        }
        npu.synced_kv_len = 0;
        npu.cached_rope = None;
        let _ = take_fault(&self.decode_error);
        self.state_torn.store(false, Ordering::SeqCst);
        Ok(Some(lease))
    }

    fn take_decode_error(&self) -> Option<CeraError> {
        take_fault(&self.decode_error)
    }

    fn backend_label(&self) -> &'static str {
        "xdna"
    }

    fn supports_lora(&self) -> bool {
        // The fused `q4layer` / `q4head` NPU kernels have no LoRA delta hooks.
        false
    }

    fn f16_kv_supported(&self) -> bool {
        self.cpu.f16_kv_supported()
    }

    fn configure_kv_compression(&self, compression: &KvCompression) -> Result<(), CeraError> {
        validate_xdna_kv_compression(self.cpu.config(), compression)
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
                    "XDNA NPU state torn after a fault; reset the session".to_string(),
                    safe_out_len,
                );
            }
            if let Err(e) = quiesce_in_flight(&mut npu) {
                return self.record_device_fault(
                    format!("hidden_states failed to quiesce NPU: {e}"),
                    safe_out_len,
                );
            }
        }
        if state.seq_len != 0 {
            return self.reject_input(
                format!(
                    "hidden_states requires a cleared scratch state (seq_len == 0), got {}",
                    state.seq_len
                ),
                safe_out_len,
            );
        }
        // `state` is caller-owned extraction scratch starting at position 0
        // (`Session` passes `hs_scratch`), never the generation KV cache; do not
        // flush `npu.pending_cpu_kv` into `state`.
        self.cpu.hidden_states(tokens, state)
    }

    fn supports_kv_shift(&self) -> bool {
        self.cpu.supports_kv_shift()
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            let msg = "XDNA NPU state torn after a fault; reset the session".to_string();
            tracing::error!("cera::xdna: shift_kv on torn NPU state: {msg}");
            eprintln!("cera::xdna: shift_kv on torn NPU state: {msg}");
            record_first_fault(&self.decode_error, CeraError::Backend(msg));
        } else if let Err(e) = quiesce_in_flight(&mut npu) {
            self.state_torn.store(true, Ordering::SeqCst);
            let msg = format!("shift_kv failed to quiesce NPU: {e}");
            tracing::error!("cera::xdna: {msg}");
            eprintln!("cera::xdna: {msg}");
            record_first_fault(&self.decode_error, CeraError::Backend(msg));
        }
        // Always flush pending host KV and shift `state` even if the NPU faulted,
        // because `Session` unconditionally decrements `current_pos` by `shift`
        // after `shift_kv` returns; keeping `state.seq_len` aligned prevents an
        // assertion panic before the upcoming forward drains `decode_error`.
        Self::flush_pending_cpu_kv(&mut npu, state);
        self.cpu.shift_kv(state, n_keep, shift);
        npu.synced_kv_len = npu.synced_kv_len.min(n_keep).min(state.seq_len);
        npu.cached_rope = None;
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
        if let Err(e) = quiesce_in_flight(&mut npu) {
            self.state_torn.store(true, Ordering::SeqCst);
            tracing::error!("cera::xdna: truncate_kv({len}) failed to quiesce NPU: {e}");
            eprintln!("cera::xdna: truncate_kv({len}) failed to quiesce NPU: {e}");
        }
        Self::flush_pending_cpu_kv(&mut npu, state);
        state.truncate_to(len);
        npu.synced_kv_len = npu.synced_kv_len.min(len);
    }

    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        let npu = self.lock_npu();
        check_kv_rewind_state(
            self.cpu.config().is_causal,
            self.state_torn.load(Ordering::SeqCst),
            npu.pending_cpu_kv.as_ref(),
            state,
            len,
        )
    }

    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        let mut npu = self.lock_npu();
        check_kv_rewind_state(
            self.cpu.config().is_causal,
            self.state_torn.load(Ordering::SeqCst),
            npu.pending_cpu_kv.as_ref(),
            state,
            len,
        )?;
        if len == state.seq_len {
            return Ok(());
        }
        if let Err(e) = quiesce_in_flight(&mut npu) {
            self.state_torn.store(true, Ordering::SeqCst);
            tracing::error!("cera::xdna: KV rewind to {len} failed, NPU not quiesced: {e}");
            eprintln!("cera::xdna: KV rewind to {len} failed, NPU not quiesced: {e}");
            return Err(KvRewindError::BackendUnsupported);
        }
        Self::flush_pending_cpu_kv(&mut npu, state);
        self.cpu.try_truncate_kv(state, len)?;
        npu.synced_kv_len = npu.synced_kv_len.min(len);
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
        if let Err(e) = quiesce_in_flight(&mut npu) {
            self.state_torn.store(true, Ordering::SeqCst);
            return Err(CeraError::Backend(e));
        }
        if let Some((_, kv)) = npu.pending_cpu_kv.take() {
            npu.spare_kv = Some(kv);
        }
        npu.synced_kv_len = 0;
        npu.cached_rope = None;
        let _ = take_fault(&self.decode_error);
        self.state_torn.store(false, Ordering::SeqCst);
        *state = fresh;
        Ok(())
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        if tokens.is_empty() {
            return self.reject_input_err(CeraError::EmptyInput, vocab_size);
        }
        if state.is_compressed() {
            return self.reject_input(
                "TurboQuant KV compression is not supported by the XDNA backend".to_string(),
                vocab_size,
            );
        }
        if let Some(&bad_tok) = tokens.iter().find(|&&t| (t as usize) >= vocab_size) {
            return self.reject_input_err(
                CeraError::InvalidToken {
                    id: bad_tok,
                    vocab_size: vocab_size as u32,
                },
                vocab_size,
            );
        }
        let mut tail_start = None;
        {
            let mut npu = self.lock_npu();
            if self.state_torn.load(Ordering::SeqCst) {
                return self.record_device_fault(
                    "XDNA NPU state torn after a fault; reset the session".to_string(),
                    vocab_size,
                );
            }
            if let Err(e) = quiesce_in_flight(&mut npu) {
                return self.record_device_fault(
                    format!("forward_prefill failed to quiesce NPU: {e}"),
                    vocab_size,
                );
            }
            if start_pos != state.seq_len {
                return self.reject_input(
                    format!(
                        "forward_prefill start_pos ({start_pos}) does not match state.seq_len ({})",
                        state.seq_len
                    ),
                    vocab_size,
                );
            }
            let Some(end_pos) = start_pos.checked_add(tokens.len()) else {
                return self
                    .reject_input("forward_prefill position overflow".to_string(), vocab_size);
            };
            if end_pos > cfg.max_seq_len {
                return self.reject_input_err(
                    CeraError::ContextOverflow {
                        max_seq_len: cfg.max_seq_len as u32,
                        by: (end_pos - cfg.max_seq_len) as u32,
                    },
                    vocab_size,
                );
            }
            Self::discard_or_flush_pending_kv(&mut npu, state, start_pos);

            if tokens.len() < 16
                || start_pos > Q4PREFILL_MAX_POS0
                || state.lora.is_some()
                || npu.prefill.is_none()
            {
                drop(npu);
                return self.cpu.forward_prefill(tokens, start_pos, state);
            }

            // Compute BOS / pos=0 on CPU (~10ms once per session) so `blk.0`/`blk.1`
            // `Q4_1` `ffn_down` weights preserve exact BOS sink KV activations across
            // all layers before chunked `Q4_0` NPU prefill processes `tokens[1..]`.
            let (npu_start_pos, npu_tokens) = if start_pos == 0 {
                let _ = self.cpu.forward(&tokens[..1], 0, state);
                if let Some(err) = self.cpu.take_decode_error() {
                    return self.record_device_fault_err(err, vocab_size);
                }
                (1usize, &tokens[1..])
            } else {
                (start_pos, tokens)
            };

            if npu.synced_kv_len != npu_start_pos
                && let Err(e) = self.sync_cpu_kv_to_npu(&mut npu, state, npu_start_pos)
            {
                return match e {
                    SyncErr::Input(msg) => self.reject_input(msg, vocab_size),
                    SyncErr::Device(msg) => self.record_device_fault(msg, vocab_size),
                };
            }

            let npu_ref = &mut *npu;
            let pf = npu_ref
                .prefill
                .as_deref_mut()
                .expect("checked prefill.is_some()");
            let in_flight = &mut npu_ref.in_flight;
            let layer_bo_kv = &mut npu_ref.layer_bo_kv;
            let head_bo_y = &mut npu_ref.head_bo_y;
            let synced_kv_len = &mut npu_ref.synced_kv_len;
            npu_ref.cached_rope = None;

            let kv_cur_bytes = Q4LAYER_N_KV_HEADS * Q4LAYER_TILE_F32 * 4;
            let kv_row_bytes = HEAD_DIM * 2;
            let meta_sync_off = (Q4PREFILL_SLOT_META + 4096) * 4;
            let meta_sync_bytes = (4608 + Q4PREFILL_B * HEAD_DIM - 4096) * 4;
            let mut k_mirror = vec![0.0f32; Q4PREFILL_B * KV_DIM];
            let mut v_mirror = vec![0.0f32; Q4PREFILL_B * KV_DIM];
            let mut rope_cs = [0.0f32; Q4PREFILL_B * HEAD_DIM];
            let mut last_n_valid = 0usize;

            for chunk_off in (0..npu_tokens.len()).step_by(Q4PREFILL_B) {
                let pos0 = npu_start_pos + chunk_off;
                if pos0 > Q4PREFILL_MAX_POS0 {
                    tail_start = Some(pos0 - start_pos);
                    break;
                }
                let n_valid = (npu_tokens.len() - chunk_off).min(Q4PREFILL_B);
                let chunk_tokens = &npu_tokens[chunk_off..chunk_off + n_valid];
                let (pf_bucket_idx, bucket_chunks) =
                    select_q4prefill_kv_bucket(&pf.bucket_chunks, pos0);

                {
                    let y0 = pf.bo_y[0].as_mut_slice();
                    for (ti, &tok) in chunk_tokens.iter().enumerate() {
                        let emb = self.cpu.embed_token(tok, "forward_prefill");
                        write_le_f32_slice(y0, ti * HIDDEN_SIZE, &emb);
                    }
                    if n_valid < Q4PREFILL_B {
                        y0[n_valid * HIDDEN_SIZE * 4..Q4PREFILL_B * HIDDEN_SIZE * 4].fill(0);
                    }
                }
                if let Err(e) = pf.bo_y[0].sync_range_to_device(Q4PREFILL_B * HIDDEN_SIZE * 4, 0) {
                    return self.record_device_fault(
                        format!("prefill bo_y[0] hidden sync: {e}"),
                        vocab_size,
                    );
                }

                for ti in 0..Q4PREFILL_B {
                    let dst = &mut rope_cs[ti * HEAD_DIM..(ti + 1) * HEAD_DIM];
                    if ti < n_valid {
                        let (c, s) = self.compute_rope_pair(pos0 + ti);
                        dst[..HEAD_DIM / 2].copy_from_slice(&c);
                        dst[HEAD_DIM / 2..].copy_from_slice(&s);
                    } else {
                        dst[..HEAD_DIM / 2].fill(1.0);
                        dst[HEAD_DIM / 2..].fill(0.0);
                    }
                }
                for il in 0..N_LAYERS {
                    write_q4prefill_step_params(
                        pf.bo_y[il].as_mut_slice(),
                        pos0,
                        n_valid,
                        bucket_chunks,
                        pos0,
                        true,
                        0,
                        cfg.rms_norm_eps,
                        &rope_cs,
                    );
                    if let Err(e) = pf.bo_y[il].sync_range_to_device(meta_sync_bytes, meta_sync_off)
                    {
                        return self.record_device_fault(
                            format!("prefill layer {il} meta sync: {e}"),
                            vocab_size,
                        );
                    }
                }

                *in_flight = Some(InFlightXdnaOp::PrefillBucket(pf_bucket_idx));
                if let Err(e) = pf.bucket_runlists[pf_bucket_idx].execute() {
                    *in_flight = None;
                    return self
                        .record_device_fault(format!("prefill runlist failed: {e}"), vocab_size);
                }
                if let Err(e) = pf.bucket_runlists[pf_bucket_idx].wait(QUIESCE_TIMEOUT_MS) {
                    return self
                        .record_device_fault(format!("prefill runlist failed: {e}"), vocab_size);
                }
                *in_flight = None;

                let m_len = n_valid * KV_DIM;
                for (l, kv_bo) in layer_bo_kv.iter_mut().enumerate().take(N_LAYERS) {
                    if let Err(e) =
                        pf.bo_y[l].sync_range_from_device(kv_cur_bytes, Q4PREFILL_SLOT_KV_CUR * 4)
                    {
                        return self.record_device_fault(
                            format!("prefill layer {l} kv sync: {e}"),
                            vocab_size,
                        );
                    }
                    {
                        let src_bytes = pf.bo_y[l].as_slice();
                        let dst_bytes = kv_bo.as_mut_slice();
                        for kvh in 0..N_KV_HEADS {
                            let k_src_base = (Q4PREFILL_SLOT_KV_CUR + kvh * Q4LAYER_TILE_F32) * 4;
                            let v_src_base = k_src_base + Q4LAYER_SILU_HALF * 4;
                            for ti in 0..n_valid {
                                let gpos = pos0 + ti;
                                let c = gpos / Q4LAYER_ATTN_CHUNK_T;
                                let rem = gpos % Q4LAYER_ATTN_CHUNK_T;
                                let k_dst = ((kvh * Q4LAYER_ATTN_CHUNKS + c) * Q4LAYER_TILE_F32)
                                    * 4
                                    + rem * kv_row_bytes;
                                let v_dst = k_dst + Q4LAYER_SILU_HALF * 4;
                                let k_src = k_src_base + ti * kv_row_bytes;
                                let v_src = v_src_base + ti * kv_row_bytes;
                                dst_bytes[k_dst..k_dst + kv_row_bytes]
                                    .copy_from_slice(&src_bytes[k_src..k_src + kv_row_bytes]);
                                dst_bytes[v_dst..v_dst + kv_row_bytes]
                                    .copy_from_slice(&src_bytes[v_src..v_src + kv_row_bytes]);
                                let m_off = ti * KV_DIM + kvh * HEAD_DIM;
                                for d in 0..HEAD_DIM {
                                    k_mirror[m_off + d] = bf16_to_f32(u16::from_le_bytes([
                                        src_bytes[k_src + d * 2],
                                        src_bytes[k_src + d * 2 + 1],
                                    ]));
                                    v_mirror[m_off + d] = bf16_to_f32(u16::from_le_bytes([
                                        src_bytes[v_src + d * 2],
                                        src_bytes[v_src + d * 2 + 1],
                                    ]));
                                }
                            }
                        }
                    }
                    let c0 = pos0 / Q4LAYER_ATTN_CHUNK_T;
                    let r0 = pos0 % Q4LAYER_ATTN_CHUNK_T;
                    let n0 = (Q4LAYER_ATTN_CHUNK_T - r0).min(n_valid);
                    let n1 = n_valid - n0;
                    for kvh in 0..N_KV_HEADS {
                        let k_off0 = ((kvh * Q4LAYER_ATTN_CHUNKS + c0) * Q4LAYER_TILE_F32) * 4
                            + r0 * kv_row_bytes;
                        for byte_off in [k_off0, k_off0 + Q4LAYER_SILU_HALF * 4] {
                            if let Err(e) = kv_bo.sync_range_to_device(n0 * kv_row_bytes, byte_off)
                            {
                                return self.record_device_fault(
                                    format!("prefill layer {l} bo_kv sync: {e}"),
                                    vocab_size,
                                );
                            }
                        }
                        if n1 > 0 {
                            let k_off1 =
                                ((kvh * Q4LAYER_ATTN_CHUNKS + c0 + 1) * Q4LAYER_TILE_F32) * 4;
                            for byte_off in [k_off1, k_off1 + Q4LAYER_SILU_HALF * 4] {
                                if let Err(e) =
                                    kv_bo.sync_range_to_device(n1 * kv_row_bytes, byte_off)
                                {
                                    return self.record_device_fault(
                                        format!("prefill layer {l} bo_kv tail sync: {e}"),
                                        vocab_size,
                                    );
                                }
                            }
                        }
                    }
                    if let LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } = &mut state.layers[l]
                    {
                        if state.kv_f16 {
                            key_cache_f16.extend(k_mirror[..m_len].iter().copied().map(f32_to_f16));
                            value_cache_f16
                                .extend(v_mirror[..m_len].iter().copied().map(f32_to_f16));
                        } else {
                            key_cache.extend_from_slice(&k_mirror[..m_len]);
                            value_cache.extend_from_slice(&v_mirror[..m_len]);
                        }
                    }
                }
                state.seq_len = pos0 + n_valid;
                *synced_kv_len = pos0 + n_valid;
                last_n_valid = n_valid;
            }

            if tail_start.is_none() {
                let last_off = (last_n_valid - 1) * HIDDEN_SIZE * 4;
                if let Err(e) = pf.bo_out.sync_range_from_device(HIDDEN_SIZE * 4, last_off) {
                    return self
                        .record_device_fault(format!("prefill bo_out sync: {e}"), vocab_size);
                }
                head_bo_y.as_mut_slice()[..HIDDEN_SIZE * 4]
                    .copy_from_slice(&pf.bo_out.as_slice()[last_off..last_off + HIDDEN_SIZE * 4]);
                if let Err(e) = head_bo_y.sync_range_to_device(HIDDEN_SIZE * 4, 0) {
                    return self
                        .record_device_fault(format!("prefill head_bo_y sync: {e}"), vocab_size);
                }
                *in_flight = Some(InFlightXdnaOp::PrefillHead);
                if let Err(e) = pf.head_run.start() {
                    *in_flight = None;
                    return self
                        .record_device_fault(format!("prefill head run failed: {e}"), vocab_size);
                }
                if let Err(e) = pf.head_run.wait(QUIESCE_TIMEOUT_MS) {
                    return self
                        .record_device_fault(format!("prefill head run failed: {e}"), vocab_size);
                }
                *in_flight = None;
                if let Err(e) =
                    head_bo_y.sync_range_from_device(vocab_size * 4, Q4LAYER_TILE_F32 * 4)
                {
                    return self
                        .record_device_fault(format!("prefill head logits sync: {e}"), vocab_size);
                }
                let mut logits = vec![0.0f32; vocab_size];
                read_le_f32_slice(head_bo_y.as_slice(), Q4LAYER_TILE_F32, &mut logits);
                if cfg.scalars.logit != 1.0 {
                    cpu::scale_inplace(&mut logits, 1.0 / cfg.scalars.logit);
                }
                return logits;
            }
        }

        let ts = tail_start.expect("only reached with tail tokens");
        let mut logits = vec![0.0f32; vocab_size];
        for (i, &tok) in tokens[ts..].iter().enumerate() {
            logits = self.forward(&[tok], start_pos + ts + i, state);
        }
        logits
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        if tokens.is_empty() {
            return self.reject_input_err(CeraError::EmptyInput, vocab_size);
        }
        if tokens.len() != 1 {
            return self.reject_input(
                format!(
                    "XdnaLlamaModel::forward expects 1 token, got {}",
                    tokens.len()
                ),
                vocab_size,
            );
        }
        if (tokens[0] as usize) >= vocab_size {
            return self.reject_input_err(
                CeraError::InvalidToken {
                    id: tokens[0],
                    vocab_size: vocab_size as u32,
                },
                vocab_size,
            );
        }
        if state.is_compressed() {
            return self.reject_input(
                "TurboQuant KV compression is not supported by the XDNA backend".to_string(),
                vocab_size,
            );
        }

        // Position 0 (BOS attention sink, where `blk.0`/`blk.1` `Q4_1`
        // `ffn_down` matters most) and positions beyond `MAX_KV_CACHE_TOKENS`
        // (or callers passing `state.lora`) fall back to CPU (`state.layers`
        // holds the full KV history because every NPU step mirrors K/V).
        if pos == 0 || pos >= MAX_KV_CACHE_TOKENS || state.lora.is_some() {
            {
                let mut npu = self.lock_npu();
                if self.state_torn.load(Ordering::SeqCst) {
                    return self.record_device_fault(
                        "XDNA NPU state torn after a fault; reset the session".to_string(),
                        vocab_size,
                    );
                }
                if let Err(e) = quiesce_in_flight(&mut npu) {
                    return self.record_device_fault(
                        format!("forward failed to quiesce NPU: {e}"),
                        vocab_size,
                    );
                }
                if pos != state.seq_len {
                    return self.reject_input(
                        format!(
                            "forward pos ({pos}) does not match state.seq_len ({})",
                            state.seq_len
                        ),
                        vocab_size,
                    );
                }
                if pos >= cfg.max_seq_len {
                    return self.reject_input_err(
                        CeraError::ContextOverflow {
                            max_seq_len: cfg.max_seq_len as u32,
                            by: (pos + 1 - cfg.max_seq_len) as u32,
                        },
                        vocab_size,
                    );
                }
                Self::discard_or_flush_pending_kv(&mut npu, state, pos);
            }
            return self.cpu.forward(tokens, pos, state);
        }

        let mut logits = vec![0.0f32; vocab_size];
        match self.forward_npu_decode_into(tokens[0], pos, state, &mut logits) {
            Ok(()) => logits,
            Err(err_logits) => err_logits,
        }
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let cfg = self.cpu.config();
        let vocab_size = cfg.vocab_size;
        if tokens.len() != 1
            || (tokens[0] as usize) >= vocab_size
            || state.is_compressed()
            || self.state_torn.load(Ordering::SeqCst)
            || pos != state.seq_len
            || pos >= cfg.max_seq_len
        {
            let logits = self.forward(tokens, pos, state);
            return crate::sampler::argmax(&logits);
        }

        if pos == 0 || pos >= MAX_KV_CACHE_TOKENS || state.lora.is_some() {
            {
                let mut npu = self.lock_npu();
                if self.state_torn.load(Ordering::SeqCst) {
                    let _ = self.record_device_fault(
                        "XDNA NPU state torn after a fault; reset the session".to_string(),
                        vocab_size,
                    );
                    return 0;
                }
                if let Err(e) = quiesce_in_flight(&mut npu) {
                    let _ = self.record_device_fault(
                        format!("forward_greedy failed to quiesce NPU: {e}"),
                        vocab_size,
                    );
                    return 0;
                }
                if pos != state.seq_len {
                    let _ = self.reject_input(
                        format!(
                            "forward pos ({pos}) does not match state.seq_len ({})",
                            state.seq_len
                        ),
                        vocab_size,
                    );
                    return 0;
                }
                Self::discard_or_flush_pending_kv(&mut npu, state, pos);
            }
            return self.cpu.forward_greedy(tokens, pos, state);
        }

        if state.scratch.logits.len() < vocab_size {
            state.scratch.logits.resize(vocab_size, 0.0);
        }
        let mut logits_buf = std::mem::take(&mut state.scratch.logits);
        let tok = match self.forward_npu_decode_into(
            tokens[0],
            pos,
            state,
            &mut logits_buf[..vocab_size],
        ) {
            Ok(()) => crate::sampler::argmax(&logits_buf[..vocab_size]),
            Err(_) => 0,
        };
        state.scratch.logits = logits_buf;
        tok
    }
}

impl std::fmt::Debug for XdnaLlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XdnaLlamaModel")
            .field("config", self.cpu.config())
            .finish_non_exhaustive()
    }
}

/// Load a model on the AMD XDNA NPU.
pub fn load_model_xdna(
    gguf: GgufFile,
    path: Option<&std::path::Path>,
    context_size: usize,
) -> Result<Box<dyn Model>> {
    let model_id = path
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let arch = gguf
        .get_str("general.architecture")
        .context("missing general.architecture")?
        .to_lowercase();
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    let decode_shape = {
        let head_bytes = gguf
            .tensors
            .get("output.weight")
            .or_else(|| gguf.tensors.get("token_embd.weight"))
            .map(|t| t.size_bytes as u64)
            .unwrap_or(0);
        (head_bytes > 0).then_some(crate::backend::calibrate::DecodeShape {
            weight_bytes: head_bytes,
            dispatches_per_token: 1,
        })
    };
    let model: Box<dyn Model> = match arch.as_str() {
        "llama" => Box::new(XdnaLlamaModel::from_gguf_with_id(
            gguf,
            context_size,
            model_id,
        )?),
        "lfm2" | "lfm2moe" => Box::new(super::xdna_lfm2::XdnaLfm2Model::from_gguf_with_id(
            gguf,
            context_size,
            model_id,
        )?),
        "lfm3" => Box::new(super::xdna_lfm3::XdnaLfm3Model::from_gguf_with_id(
            gguf,
            context_size,
            model_id,
        )?),
        _ => bail!(
            "unsupported architecture for AMD XDNA NPU: {arch} \
             (XDNA backend supports Llama-3.2-1B, LFM2/LFM2.5/LFM2-MoE, and LFM3)"
        ),
    };
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if let Some(shape) = decode_shape {
        crate::backend::calibrate::set_decode_shape(shape);
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::xdna::sys::fake::{Config as FakeConfig, Event as FakeEvent, with_shim};
    use crate::model::ScalarMultipliers;

    fn test_llama_1b_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            n_layers: N_LAYERS,
            hidden_size: HIDDEN_SIZE,
            intermediate_size: INTERMEDIATE_SIZE,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: 128_256,
            max_seq_len: 1024,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-5,
            block_types: vec![crate::model::BlockType::Attention; N_LAYERS],
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer: vec![N_KV_HEADS; N_LAYERS],
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        }
    }

    #[test]
    fn kv_bucket_selection_covers_all_positions() {
        let buckets: Vec<usize> = Q4LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        assert_eq!(select_q4layer_kv_bucket(&buckets, 0), (0, 63));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 1), (0, 63));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 64), (0, 63));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 65), (1, 62));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 128), (1, 62));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 129), (2, 60));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 3072), (8, 16));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 3073), (9, 0));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 4095), (9, 0));
    }

    #[test]
    fn validate_xdna_kv_compression_rejects_turboquant_and_accepts_none_and_f16() {
        let cfg = test_llama_1b_config();
        assert!(validate_xdna_kv_compression(&cfg, &KvCompression::None).is_ok());
        assert!(validate_xdna_kv_compression(&cfg, &KvCompression::F16).is_ok());
        let err = validate_xdna_kv_compression(&cfg, &KvCompression::turboquant(42))
            .expect_err("TurboQuant must be rejected on XDNA");
        assert!(err.to_string().contains("TurboQuant"));
    }

    #[test]
    fn deferred_pending_kv_rewind_allows_pos0_and_post_prefill_and_preserves_state_on_oob() {
        let cfg = test_llama_1b_config();

        // Case 1: single NPU decode at pos=0 leaves key_cache empty and seq_len=1.
        let mut state0 =
            InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64).expect("state0");
        state0.seq_len = 1;
        let pending0 = (
            0usize,
            (
                vec![[1.0f32; KV_DIM]; N_LAYERS],
                vec![[2.0f32; KV_DIM]; N_LAYERS],
            ),
        );
        assert_eq!(
            check_kv_rewind_state(true, false, Some(&pending0), &state0, 0),
            Ok(())
        );
        assert_eq!(
            check_kv_rewind_state(true, false, Some(&pending0), &state0, 2),
            Err(KvRewindError::OutOfBounds {
                requested: 2,
                current: 1
            })
        );
        // OutOfBounds did not touch state0.
        assert_eq!(state0.seq_len, 1);
        assert_eq!(state0.kv_cache_f16(0).0.len(), 0);

        // Case 2: 16-token prefill + 1 NPU decode at pos=16 leaves 16*KV_DIM rows and seq_len=17.
        let mut state17 =
            InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64).expect("state17");
        for l in 0..N_LAYERS {
            state17.append_kv_f16(l, &vec![0.5f32; 16 * KV_DIM], &vec![-0.5f32; 16 * KV_DIM]);
        }
        state17.seq_len = 17;
        let pending16 = (
            16usize,
            (
                vec![[3.0f32; KV_DIM]; N_LAYERS],
                vec![[4.0f32; KV_DIM]; N_LAYERS],
            ),
        );
        assert_eq!(
            check_kv_rewind_state(true, false, Some(&pending16), &state17, 10),
            Ok(())
        );
        assert_eq!(
            check_kv_rewind_state(true, false, Some(&pending16), &state17, 16),
            Ok(())
        );
    }

    #[test]
    fn poisoned_npu_lock_clears_poison_recycles_pending_kv_and_marks_torn() {
        let (_, events) = with_shim(FakeConfig::default(), || {
            let device = XdnaDevice::open(0).expect("fake open");
            let kernel = device
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("fake load");
            let head_bo_w = kernel.alloc_bo(64, 3).expect("bo");
            let head_bo_y = kernel.alloc_bo(64, 4).expect("bo");
            let state = XdnaDecodeState {
                bucket_chunks: vec![9],
                bucket_runlists: Vec::new(),
                _head_runs: Vec::new(),
                _layer_runs: Vec::new(),
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_kv: Vec::new(),
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                prefill: None,
                _device: device,
                in_flight: None,
                synced_kv_len: 16,
                cached_rope: None,
                pending_cpu_kv: Some((
                    15,
                    (
                        vec![[1.0f32; KV_DIM]; N_LAYERS],
                        vec![[2.0f32; KV_DIM]; N_LAYERS],
                    ),
                )),
                spare_kv: None,
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            };
            let mu = Mutex::new(state);
            let torn = AtomicBool::new(false);
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = mu.lock().unwrap();
                panic!("synthetic panic inside forward");
            }));
            assert!(mu.is_poisoned());

            {
                let g = lock_npu_state(&mu, &torn);
                assert!(torn.load(Ordering::SeqCst));
                assert!(g.pending_cpu_kv.is_none());
                assert!(g.spare_kv.is_some());
            }
            // `clear_poison` ran, so a subsequent lock does not re-trigger poison recovery.
            assert!(!mu.is_poisoned());
            torn.store(false, Ordering::SeqCst);
            let _g2 = lock_npu_state(&mu, &torn);
            assert!(
                !torn.load(Ordering::SeqCst),
                "second lock must not re-mark torn after clear_poison"
            );
        });
        assert!(!events.is_empty());
    }

    #[test]
    fn quiesce_in_flight_waits_on_drop_and_retains_bucket_when_wait_fails() {
        let (_, events_ok) = with_shim(FakeConfig::default(), || {
            let device = XdnaDevice::open(0).expect("fake open");
            let kernel = device
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("fake load");
            let head_bo_w = kernel.alloc_bo(64, 3).expect("bo");
            let head_bo_y = kernel.alloc_bo(64, 4).expect("bo");
            let run = kernel.create_run(&[&head_bo_w, &head_bo_y]).expect("run");
            let mut rl = kernel.create_runlist().expect("runlist");
            rl.add(&run).expect("add");
            rl.execute().expect("exec");
            let state = XdnaDecodeState {
                bucket_chunks: vec![9],
                bucket_runlists: vec![rl],
                _head_runs: vec![run],
                _layer_runs: Vec::new(),
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_kv: Vec::new(),
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                prefill: None,
                _device: device,
                in_flight: Some(InFlightXdnaOp::DecodeBucket(0)),
                synced_kv_len: 0,
                cached_rope: None,
                pending_cpu_kv: None,
                spare_kv: None,
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            };
            drop(state);
        });
        let wait_idx = events_ok
            .iter()
            .position(|e| {
                matches!(
                    e,
                    FakeEvent::RunListWait {
                        timeout_ms: 30_000,
                        ..
                    }
                )
            })
            .expect("Drop must wait for in-flight runlist");
        let first_bo_free = events_ok
            .iter()
            .position(|e| matches!(e, FakeEvent::BoFree(_)))
            .expect("BOs freed");
        assert!(
            wait_idx < first_bo_free,
            "in-flight runlist must quiesce before BOs are freed: {events_ok:?}"
        );

        let hung_cfg = FakeConfig {
            fail_runlist_wait: true,
            ..FakeConfig::default()
        };
        let (_, _) = with_shim(hung_cfg, || {
            let device = XdnaDevice::open(0).expect("fake open");
            let kernel = device
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("fake load");
            let head_bo_w = kernel.alloc_bo(64, 3).expect("bo");
            let head_bo_y = kernel.alloc_bo(64, 4).expect("bo");
            let run = kernel.create_run(&[&head_bo_w, &head_bo_y]).expect("run");
            let mut rl = kernel.create_runlist().expect("runlist");
            rl.add(&run).expect("add");
            let mut state = XdnaDecodeState {
                bucket_chunks: vec![9],
                bucket_runlists: vec![rl],
                _head_runs: vec![run],
                _layer_runs: Vec::new(),
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_kv: Vec::new(),
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                prefill: None,
                _device: device,
                in_flight: Some(InFlightXdnaOp::DecodeBucket(0)),
                synced_kv_len: 0,
                cached_rope: None,
                pending_cpu_kv: None,
                spare_kv: None,
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            };
            assert!(quiesce_in_flight(&mut state).is_err());
            assert_eq!(state.in_flight, Some(InFlightXdnaOp::DecodeBucket(0)));
        });
    }

    #[test]
    fn check_kv_rewind_rejects_non_causal_classifier_and_torn_while_allowing_healthy_state() {
        let cfg = test_llama_1b_config();
        let mut state =
            InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64).expect("state");
        for l in 0..N_LAYERS {
            state.append_kv_f16(l, &vec![0.25f32; 4 * KV_DIM], &vec![-0.25f32; 4 * KV_DIM]);
        }
        state.seq_len = 4;

        // Non-causal config is rejected with NonCausal.
        assert_eq!(
            check_kv_rewind_state(false, false, None, &state, 2),
            Err(KvRewindError::NonCausal)
        );
        // Classifier LoRA switches effective execution to bidirectional pooling, so rewind is NonCausal.
        state.lora = Some(crate::lora::LoraAdapterWeights::new_classifier_for_testing(
            vec![0.1f32; 2 * HIDDEN_SIZE],
            None,
            vec!["neg".to_string(), "pos".to_string()],
        ));
        assert_eq!(
            check_kv_rewind_state(true, false, None, &state, 2),
            Err(KvRewindError::NonCausal)
        );
        state.lora = None;

        // Torn hardware state is rejected with BackendUnsupported even for len == 0.
        assert_eq!(
            check_kv_rewind_state(true, true, None, &state, 0),
            Err(KvRewindError::BackendUnsupported)
        );
        assert_eq!(
            check_kv_rewind_state(true, true, None, &state, 4),
            Err(KvRewindError::BackendUnsupported)
        );

        // Non-torn state (including after a caller-input rejection that leaves state_torn false) rewinds cleanly.
        assert_eq!(check_kv_rewind_state(true, false, None, &state, 2), Ok(()));
        assert_eq!(check_kv_rewind_state(true, false, None, &state, 4), Ok(()));
        assert_eq!(state.seq_len, 4);
    }

    #[test]
    fn flush_pending_cpu_kv_guards_against_mismatched_state_and_clamps_cpu_fallback_sync() {
        with_shim(FakeConfig::default(), || {
            let device = XdnaDevice::open(0).expect("fake open");
            let kernel = device
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("fake load");
            let head_bo_w = kernel.alloc_bo(64, 3).expect("bo");
            let head_bo_y = kernel.alloc_bo(64, 4).expect("bo");
            let mut npu = XdnaDecodeState {
                bucket_chunks: vec![9],
                bucket_runlists: Vec::new(),
                _head_runs: Vec::new(),
                _layer_runs: Vec::new(),
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_kv: Vec::new(),
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                prefill: None,
                _device: device,
                in_flight: None,
                synced_kv_len: 5,
                cached_rope: None,
                pending_cpu_kv: Some((
                    4,
                    (
                        vec![[9.0f32; KV_DIM]; N_LAYERS],
                        vec![[8.0f32; KV_DIM]; N_LAYERS],
                    ),
                )),
                spare_kv: None,
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            };

            let cfg = test_llama_1b_config();
            // Mismatched empty state (`seq_len == 0`) must not receive stale pos=4 KV,
            // and `synced_kv_len` must reset to 0 instead of staying at 5.
            let mut fresh_state =
                InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64).expect("fresh");
            XdnaLlamaModel::flush_pending_cpu_kv(&mut npu, &mut fresh_state);
            assert!(npu.pending_cpu_kv.is_none());
            assert!(npu.spare_kv.is_some());
            assert_eq!(npu.synced_kv_len, 0);
            assert_eq!(fresh_state.kv_cache_f16(0).0.len(), 0);

            // `discard_or_flush_pending_kv(..., pos = 0)` on CPU fallback clears stale KV and sync len.
            npu.synced_kv_len = 5;
            npu.pending_cpu_kv = Some((
                4,
                (
                    vec![[7.0f32; KV_DIM]; N_LAYERS],
                    vec![[6.0f32; KV_DIM]; N_LAYERS],
                ),
            ));
            XdnaLlamaModel::discard_or_flush_pending_kv(&mut npu, &mut fresh_state, 0);
            assert!(npu.pending_cpu_kv.is_none());
            assert_eq!(npu.synced_kv_len, 0);
            assert_eq!(fresh_state.kv_cache_f16(0).0.len(), 0);

            // Matching end-of-table state (`pos == MAX_KV_CACHE_TOKENS - 1`, `seq_len == pos + 1`)
            // flushes cleanly and preserves `synced_kv_len` up to `pos`.
            let cap_pos = MAX_KV_CACHE_TOKENS - 1;
            let mut full_state = InferenceState::from_config_capped(
                &cfg,
                &KvCompression::F16,
                MAX_KV_CACHE_TOKENS + 8,
            )
            .expect("full");
            for l in 0..N_LAYERS {
                full_state.append_kv_f16(
                    l,
                    &vec![0.25f32; cap_pos * KV_DIM],
                    &vec![-0.25f32; cap_pos * KV_DIM],
                );
            }
            full_state.seq_len = MAX_KV_CACHE_TOKENS;
            npu.synced_kv_len = MAX_KV_CACHE_TOKENS;
            npu.pending_cpu_kv = Some((
                cap_pos,
                (
                    vec![[1.5f32; KV_DIM]; N_LAYERS],
                    vec![[-1.5f32; KV_DIM]; N_LAYERS],
                ),
            ));
            XdnaLlamaModel::flush_pending_cpu_kv(&mut npu, &mut full_state);
            assert!(npu.pending_cpu_kv.is_none());
            assert_eq!(
                full_state.kv_cache_f16(0).0.len(),
                MAX_KV_CACHE_TOKENS * KV_DIM
            );

            // CPU fallback at a lower position clamps `synced_kv_len` to `pos`.
            let mut fallback_state =
                InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64)
                    .expect("fallback");
            for l in 0..N_LAYERS {
                fallback_state.append_kv_f16(
                    l,
                    &vec![0.5f32; 2 * KV_DIM],
                    &vec![-0.5f32; 2 * KV_DIM],
                );
            }
            fallback_state.seq_len = 3;
            npu.synced_kv_len = 6;
            npu.pending_cpu_kv = Some((
                2,
                (
                    vec![[2.0f32; KV_DIM]; N_LAYERS],
                    vec![[-2.0f32; KV_DIM]; N_LAYERS],
                ),
            ));
            XdnaLlamaModel::discard_or_flush_pending_kv(&mut npu, &mut fallback_state, 3);
            assert!(npu.pending_cpu_kv.is_none());
            assert_eq!(npu.synced_kv_len, 3);
            assert_eq!(fallback_state.kv_cache_f16(0).0.len(), 3 * KV_DIM);
        });
    }

    fn tiny_llama_gguf(arch: &str) -> GgufFile {
        use crate::gguf::GgufBuilder;
        let hs = 32usize;
        let nh = 2usize;
        let nkv = 1usize;
        let hd = 16usize;
        let inter = 32usize;
        let vocab = 16usize;
        let b = GgufBuilder::new()
            .kv_str("general.architecture", arch)
            .kv_u32(format!("{arch}.block_count"), 1)
            .kv_u32(format!("{arch}.embedding_length"), hs as u32)
            .kv_u32(format!("{arch}.feed_forward_length"), inter as u32)
            .kv_u32(format!("{arch}.attention.head_count"), nh as u32)
            .kv_u32(format!("{arch}.attention.head_count_kv"), nkv as u32)
            .kv_f32(format!("{arch}.attention.layer_norm_epsilon"), 1e-5)
            .kv_f32(format!("{arch}.rope.freq_base"), 500_000.0)
            .kv_u32(format!("{arch}.context_length"), 64)
            .kv_u32(format!("{arch}.vocab_size"), vocab as u32)
            .tensor_f32("token_embd.weight", &[hs, vocab], &vec![0.05; hs * vocab])
            .tensor_f32("output_norm.weight", &[hs], &vec![1.0; hs])
            .tensor_f32("blk.0.attn_norm.weight", &[hs], &vec![1.0; hs])
            .tensor_f32("blk.0.ffn_norm.weight", &[hs], &vec![1.0; hs])
            .tensor_f32(
                "blk.0.attn_q.weight",
                &[hs, nh * hd],
                &vec![0.05; hs * nh * hd],
            )
            .tensor_f32(
                "blk.0.attn_k.weight",
                &[hs, nkv * hd],
                &vec![0.05; hs * nkv * hd],
            )
            .tensor_f32(
                "blk.0.attn_v.weight",
                &[hs, nkv * hd],
                &vec![0.05; hs * nkv * hd],
            )
            .tensor_f32(
                "blk.0.attn_output.weight",
                &[nh * hd, hs],
                &vec![0.05; nh * hd * hs],
            )
            .tensor_f32(
                "blk.0.ffn_gate.weight",
                &[hs, inter],
                &vec![0.05; hs * inter],
            )
            .tensor_f32("blk.0.ffn_up.weight", &[hs, inter], &vec![0.05; hs * inter])
            .tensor_f32(
                "blk.0.ffn_down.weight",
                &[inter, hs],
                &vec![0.05; inter * hs],
            );
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    #[test]
    fn from_gguf_with_id_rejects_non_llama_arch_upfront() {
        let gguf = tiny_llama_gguf("qwen2");
        let err = XdnaLlamaModel::from_gguf_with_id(gguf, 64, "test-non-llama".into())
            .expect_err("non-llama architecture must be rejected before CPU loading");
        assert!(
            err.to_string()
                .contains("unsupported architecture for AMD XDNA NPU: qwen2"),
            "{err}"
        );
    }

    #[test]
    fn model_trait_isolates_hidden_states_and_keeps_state_untorn_on_input_errors() {
        with_shim(FakeConfig::default(), || {
            let cpu = LlamaModel::from_gguf_with_id(
                tiny_llama_gguf("llama"),
                64,
                "test-xdna-trait".into(),
            )
            .expect("tiny cpu llama");
            let cfg = cpu.config().clone();
            let device = XdnaDevice::open(0).expect("fake open");
            let kernel = device
                .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
                .expect("fake load");
            let head_bo_w = kernel.alloc_bo(64, 3).expect("bo");
            let head_bo_y = kernel.alloc_bo(64, 4).expect("bo");
            let npu = XdnaDecodeState {
                bucket_chunks: vec![9],
                bucket_runlists: Vec::new(),
                _head_runs: Vec::new(),
                _layer_runs: Vec::new(),
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_kv: Vec::new(),
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                prefill: None,
                _device: device,
                in_flight: None,
                synced_kv_len: 1,
                cached_rope: None,
                pending_cpu_kv: Some((
                    0,
                    (
                        vec![[1.0f32; KV_DIM]; N_LAYERS],
                        vec![[-1.0f32; KV_DIM]; N_LAYERS],
                    ),
                )),
                spare_kv: None,
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            };
            let model = XdnaLlamaModel::from_parts_for_test(cpu, npu);

            // `hidden_states` uses its own extraction scratch and leaves `pending_cpu_kv` intact.
            let mut hs_scratch =
                InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64).unwrap();
            let emb = model.hidden_states(&[0], &mut hs_scratch);
            assert_eq!(emb.len(), cfg.hidden_size);
            assert!(model.take_decode_error().is_none());
            assert!(model.lock_npu().pending_cpu_kv.is_some());

            // Caller input errors record `decode_error` without tearing NPU state.
            let mut state =
                InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64).unwrap();
            let empty_out = model.forward(&[], 0, &mut state);
            assert!(empty_out.iter().all(|&v| v == 0.0));
            assert!(matches!(
                model.take_decode_error(),
                Some(CeraError::EmptyInput)
            ));
            let bad_tok_out = model.forward(&[u32::MAX], 0, &mut state);
            assert_eq!(bad_tok_out.len(), cfg.vocab_size);
            assert!(matches!(
                model.take_decode_error(),
                Some(CeraError::InvalidToken { .. })
            ));
            let multi_tok_out = model.forward(&[0, 1], 0, &mut state);
            assert!(multi_tok_out.iter().all(|&v| v == 0.0));
            assert!(model.take_decode_error().is_some());
            assert!(!model.state_torn.load(Ordering::SeqCst));
            model
                .check_kv_rewind(&state, 0)
                .expect("input errors must not tear KV rewind health");

            // Poisoning the NPU lock tears state on the next lock; `try_reset_kv` clears it.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = model.npu.lock().unwrap();
                panic!("simulated worker panic");
            }));
            let torn_out = model.forward(&[0], 0, &mut state);
            assert_eq!(torn_out.len(), cfg.vocab_size);
            assert!(model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());
            assert_eq!(
                model.check_kv_rewind(&state, 0),
                Err(KvRewindError::BackendUnsupported)
            );
            model
                .try_reset_kv(&mut state, &KvCompression::F16, 64)
                .expect("try_reset_kv clears torn state");
            assert!(!model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_none());
            model
                .check_kv_rewind(&state, 0)
                .expect("reset state is healthy");

            // `forward_greedy` preserves scratch and updates `seq_len` at `pos=0`,
            // `try_truncate_kv(&mut state, 0)` rewinds cleanly, error paths do not tear state,
            // and torn faults hold until reset.
            state.scratch.logits.reserve(cfg.vocab_size);
            let scratch_ptr = state.scratch.logits.as_ptr();
            let greedy_p0 = model.forward_greedy(&[0], 0, &mut state);
            assert!((greedy_p0 as usize) < cfg.vocab_size);
            assert_eq!(state.seq_len, 1);
            assert!(model.take_decode_error().is_none());
            assert!(!model.state_torn.load(Ordering::SeqCst));
            assert_eq!(state.scratch.logits.as_ptr(), scratch_ptr);

            model.try_truncate_kv(&mut state, 0).unwrap();
            assert_eq!(state.seq_len, 0);

            let tok_oob = model.forward_greedy(&[0], 64, &mut state);
            assert_eq!(tok_oob, 0);
            assert!(model.take_decode_error().is_some());
            assert!(!model.state_torn.load(Ordering::SeqCst));

            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = model.npu.lock().unwrap();
                panic!("simulated worker panic before forward_greedy");
            }));
            let tok_torn = model.forward_greedy(&[0], state.seq_len, &mut state);
            assert_eq!(tok_torn, 0);
            assert!(model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());
            model
                .try_reset_kv(&mut state, &KvCompression::F16, 64)
                .expect("try_reset_kv clears torn state after forward_greedy");
            assert!(!model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_none());
        });
    }

    fn test_hybrid_config(conv_taps: usize) -> ModelConfig {
        let n_layers = 2;
        ModelConfig {
            architecture: "lfm2".to_string(),
            n_layers,
            hidden_size: 2048,
            intermediate_size: 4096,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 128,
            vocab_size: 1024,
            max_seq_len: 64,
            rope_theta: 100_000.0,
            rms_norm_eps: 1e-5,
            block_types: vec![
                crate::model::BlockType::GatedConv,
                crate::model::BlockType::Attention,
            ],
            conv_kernel_size: Some(conv_taps),
            ssm: None,
            kv_heads_per_layer: vec![4; n_layers],
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        }
    }

    #[test]
    fn hybrid_pending_slot_rewind_and_flush_cover_conv_and_attn() {
        let cfg = test_hybrid_config(4); // conv_taps = 3 (d_conv) when matching explicit buffer
        let kv_dim = 4 * 128;
        let conv_taps = 3;
        let mut state = InferenceState::from_config_capped(&cfg, &KvCompression::F16, 64)
            .expect("hybrid state");
        // Seed pos=0 checkpoint, leave pos=1 uncheckpointed, and stage pending at pos=1.
        if let LayerState::Conv { buffer, history } = &mut state.layers[0] {
            buffer.resize(conv_taps * 2048, 0.0);
            *history = crate::kv_cache::ConvHistory::new(conv_taps * 2048);
            history.push(0, buffer);
        }
        if let LayerState::Attention {
            key_cache_f16,
            value_cache_f16,
            ..
        } = &mut state.layers[1]
        {
            key_cache_f16.resize(kv_dim, crate::quant::f32_to_f16(0.5));
            value_cache_f16.resize(kv_dim, crate::quant::f32_to_f16(-0.5));
        }
        state.seq_len = 2;

        let mut slot = HybridPendingSlot::default();
        let mut p = slot.prepare(1, conv_taps, 2048, kv_dim);
        let mut kvo_conv = vec![0u8; 8 * Q4LAYER_TILE_F32 * 4];
        write_le_f32_slice(&mut kvo_conv, 0, &[1.25f32; 256]);
        p.push_conv_from_kvo(0, &kvo_conv);

        let mut kvo_attn = vec![0u8; 4 * 512 * 4];
        let bf16_one = 0x3F80u16.to_le_bytes();
        let bf16_two = 0x4000u16.to_le_bytes();
        for d in 0..128 {
            kvo_attn[512 + d * 2..512 + d * 2 + 2].copy_from_slice(&bf16_one);
            kvo_attn[768 + d * 2..768 + d * 2 + 2].copy_from_slice(&bf16_two);
        }
        p.push_attn_bf16(1, &kvo_attn, 4, 128, 512 * 4);
        slot.commit(p);

        // Before flush, `check_hybrid_kv_rewind_state` accepts `len=0` and `len=2` (`len == seq_len`),
        // rejects missing `len=1` with `MissingConvolutionCheckpoint`, and rejects `len=3` (`OutOfBounds`).
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot.as_ref(), &state, 0),
            Ok(())
        );
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot.as_ref(), &state, 2),
            Ok(())
        );
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot.as_ref(), &state, 1),
            Err(KvRewindError::MissingConvolutionCheckpoint {
                layer: 0,
                position: 1,
            })
        );
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot.as_ref(), &state, 3),
            Err(KvRewindError::OutOfBounds {
                requested: 3,
                current: 2,
            })
        );
        assert_eq!(
            check_hybrid_kv_rewind_state(false, false, slot.as_ref(), &state, 0),
            Err(KvRewindError::NonCausal)
        );
        assert_eq!(
            check_hybrid_kv_rewind_state(true, true, slot.as_ref(), &state, 0),
            Err(KvRewindError::BackendUnsupported)
        );

        assert!(slot.flush_into(&mut state));
        assert!(slot.as_ref().is_none());
        assert_eq!(state.kv_cache_f16(1).0.len(), 2 * kv_dim);
        assert_eq!(
            crate::quant::f16_to_f32(state.kv_cache_f16(1).0[kv_dim]),
            1.0
        );
        assert_eq!(
            crate::quant::f16_to_f32(state.kv_cache_f16(1).1[kv_dim]),
            2.0
        );
        if let LayerState::Conv { buffer, history } = &state.layers[0] {
            assert_eq!(buffer[0], 1.25);
            assert!(history.has_pos(2));
        } else {
            panic!("expected conv layer state");
        }
        // `discard_for_truncate` drops pending tail without flushing into state.
        let p2 = slot.prepare(2, conv_taps, 2048, kv_dim);
        slot.commit(p2);
        state.seq_len = 3;
        assert!(slot.discard_for_truncate(&mut state, 2));
        assert!(slot.as_ref().is_none());
        assert_eq!(state.seq_len, 2);
        assert_eq!(state.kv_cache_f16(1).0.len(), 2 * kv_dim);

        // State mismatch returns `false` and clears pending without corrupting `state`.
        let p3 = slot.prepare(99, conv_taps, 2048, kv_dim);
        slot.commit(p3);
        assert!(!slot.flush_into(&mut state));
        assert!(slot.as_ref().is_none());
    }

    #[test]
    fn hybrid_pending_slot_supports_lfm2_1024_byte_kvo_stride() {
        let cfg = test_hybrid_config(2);
        let kv_dim = 8 * 64;
        let mut lfm2_cfg = cfg;
        lfm2_cfg.n_kv_heads = 8;
        lfm2_cfg.head_dim = 64;
        lfm2_cfg.kv_heads_per_layer = vec![8; lfm2_cfg.n_layers];
        let mut state = InferenceState::from_config_capped(&lfm2_cfg, &KvCompression::F16, 64)
            .expect("lfm2 hybrid state");
        state.seq_len = 1;

        let mut slot = HybridPendingSlot::default();
        let mut p = slot.prepare(0, 2, 2048, kv_dim);
        let mut kvo_attn = vec![0u8; 8 * 1024];
        let bf16_one = 0x3F80u16.to_le_bytes();
        let bf16_two = 0x4000u16.to_le_bytes();
        for kvh in 0..8 {
            let base = kvh * 1024;
            for d in 0..64 {
                kvo_attn[base + 512 + d * 2..base + 512 + d * 2 + 2].copy_from_slice(&bf16_one);
                kvo_attn[base + 768 + d * 2..base + 768 + d * 2 + 2].copy_from_slice(&bf16_two);
            }
        }
        p.push_attn_bf16(1, &kvo_attn, 8, 64, 1024);
        slot.commit(p);
        assert!(slot.flush_into(&mut state));
        assert_eq!(state.kv_cache_f16(1).0.len(), kv_dim);
        assert_eq!(
            crate::quant::f16_to_f32(state.kv_cache_f16(1).0[kv_dim - 1]),
            1.0
        );
        assert_eq!(
            crate::quant::f16_to_f32(state.kv_cache_f16(1).1[kv_dim - 1]),
            2.0
        );
    }

    #[test]
    fn xdna_hybrid_conv_history_ring_boundary_handles_64_positions() {
        let mut cfg = test_hybrid_config(2);
        cfg.max_seq_len = 128;
        let kv_dim = 4 * 128;
        let conv_taps = 2;
        let mut state = InferenceState::from_config_capped(&cfg, &KvCompression::F16, 128)
            .expect("hybrid state");
        if let LayerState::Conv { buffer, history } = &mut state.layers[0] {
            buffer.resize(conv_taps * 2048, 0.0);
            *history = crate::kv_cache::ConvHistory::new(conv_taps * 2048);
            for pos in 1..=65usize {
                buffer.fill(pos as f32);
                history.push(pos, buffer);
            }
        }
        if let LayerState::Attention {
            key_cache_f16,
            value_cache_f16,
            ..
        } = &mut state.layers[1]
        {
            key_cache_f16.resize(65 * kv_dim, 0);
            value_cache_f16.resize(65 * kv_dim, 0);
        }
        state.seq_len = 66;

        let mut slot = HybridPendingSlot::default();
        let p = slot.prepare(65, conv_taps, 2048, kv_dim);
        slot.commit(p);

        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot.as_ref(), &state, 1),
            Err(KvRewindError::MissingConvolutionCheckpoint {
                layer: 0,
                position: 1,
            })
        );
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot.as_ref(), &state, 2),
            Ok(())
        );
        assert!(slot.discard_for_truncate(&mut state, 2));
        assert_eq!(state.seq_len, 65);

        // Live pending row at `pending_pos == len > 0` without a Conv checkpoint at `len`
        // must clear both `Attention` and `Conv` down to `seq_len == 0` and return `false`.
        state.seq_len = 2;
        if let LayerState::Attention {
            key_cache_f16,
            value_cache_f16,
            ..
        } = &mut state.layers[1]
        {
            key_cache_f16.resize(kv_dim, 1);
            value_cache_f16.resize(kv_dim, 2);
        }
        let p_missing_ckpt = slot.prepare(1, conv_taps, 2048, kv_dim);
        slot.commit(p_missing_ckpt);
        assert!(!slot.discard_for_truncate(&mut state, 1));
        assert_eq!(state.seq_len, 0);
        if let LayerState::Attention {
            key_cache_f16,
            value_cache_f16,
            ..
        } = &state.layers[1]
        {
            assert!(key_cache_f16.is_empty());
            assert!(value_cache_f16.is_empty());
        }
        if let LayerState::Conv { buffer, .. } = &state.layers[0] {
            assert!(buffer.iter().all(|&x| x == 0.0));
        }

        // Live pending row at `pending_pos == 0` zeroes `Conv` buffer and lands at `seq_len == 0`.
        if let LayerState::Conv { buffer, .. } = &mut state.layers[0] {
            buffer.fill(5.0);
        }
        state.seq_len = 1;
        let p0 = slot.prepare(0, conv_taps, 2048, kv_dim);
        slot.commit(p0);
        assert!(slot.discard_for_truncate(&mut state, 0));
        assert_eq!(state.seq_len, 0);
        if let LayerState::Conv { buffer, .. } = &state.layers[0] {
            assert!(buffer.iter().all(|&x| x == 0.0));
        }

        // `check_hybrid_kv_rewind_state` validates bounds and layer kind of `p.conv` / `p.attn`.
        state.seq_len = 66;
        let kvo_conv = vec![0u8; 8 * Q4LAYER_TILE_F32 * 4];
        let mut slot2 = HybridPendingSlot::default();
        let mut bad_conv = slot2.prepare(65, conv_taps, 2048, kv_dim);
        bad_conv.push_conv_from_kvo(1, &kvo_conv);
        slot2.commit(bad_conv);
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot2.as_ref(), &state, 2),
            Err(KvRewindError::InvalidCacheLayout {
                layer: 0,
                detail: "deferred NPU KV row references invalid or mismatched layer",
            })
        );
        let kvo_attn = vec![0u8; 4 * 1024];
        let mut bad_attn = slot2.prepare(65, conv_taps, 2048, kv_dim);
        bad_attn.push_attn_bf16(0, &kvo_attn, 4, 128, 1024);
        slot2.commit(bad_attn);
        assert_eq!(
            check_hybrid_kv_rewind_state(true, false, slot2.as_ref(), &state, 2),
            Err(KvRewindError::InvalidCacheLayout {
                layer: 0,
                detail: "deferred NPU KV row references invalid or mismatched layer",
            })
        );
    }

    fn make_llama_xdna_gguf_for_test() -> GgufFile {
        use crate::gguf::GgufBuilder;
        let hs = 32usize;
        let nh = 8usize;
        let nkv = N_KV_HEADS;
        let hd = HEAD_DIM;
        let inter = 32usize;
        let vocab = 32usize;
        let mut b = GgufBuilder::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", N_LAYERS as u32)
            .kv_u32("llama.embedding_length", hs as u32)
            .kv_u32("llama.feed_forward_length", inter as u32)
            .kv_u32("llama.attention.head_count", nh as u32)
            .kv_u32("llama.attention.head_count_kv", nkv as u32)
            .kv_u32("llama.attention.key_length", hd as u32)
            .kv_f32("llama.attention.layer_norm_epsilon", 1e-5)
            .kv_f32("llama.rope.freq_base", 500_000.0)
            .kv_u32("llama.context_length", MAX_KV_CACHE_TOKENS as u32)
            .kv_u32("llama.vocab_size", vocab as u32)
            .tensor_f32("token_embd.weight", &[hs, vocab], &vec![0.05; hs * vocab])
            .tensor_f32("output_norm.weight", &[hs], &vec![1.0; hs]);
        for l in 0..N_LAYERS {
            b = b
                .tensor_f32(format!("blk.{l}.attn_norm.weight"), &[hs], &vec![1.0; hs])
                .tensor_f32(format!("blk.{l}.ffn_norm.weight"), &[hs], &vec![1.0; hs])
                .tensor_f32(
                    format!("blk.{l}.attn_q.weight"),
                    &[hs, nh * hd],
                    &vec![0.0; hs * nh * hd],
                )
                .tensor_f32(
                    format!("blk.{l}.attn_k.weight"),
                    &[hs, nkv * hd],
                    &vec![0.0; hs * nkv * hd],
                )
                .tensor_f32(
                    format!("blk.{l}.attn_v.weight"),
                    &[hs, nkv * hd],
                    &vec![0.0; hs * nkv * hd],
                )
                .tensor_f32(
                    format!("blk.{l}.attn_output.weight"),
                    &[nh * hd, hs],
                    &vec![0.0; nh * hd * hs],
                )
                .tensor_f32(
                    format!("blk.{l}.ffn_gate.weight"),
                    &[hs, inter],
                    &vec![0.0; hs * inter],
                )
                .tensor_f32(
                    format!("blk.{l}.ffn_up.weight"),
                    &[hs, inter],
                    &vec![0.0; hs * inter],
                )
                .tensor_f32(
                    format!("blk.{l}.ffn_down.weight"),
                    &[inter, hs],
                    &vec![0.0; inter * hs],
                );
        }
        GgufFile::from_bytes(std::sync::Arc::from(b.build_bytes().into_boxed_slice())).unwrap()
    }

    fn make_llama_xdna_model_for_test(with_prefill: bool) -> (XdnaLlamaModel, ModelConfig) {
        let cpu = LlamaModel::from_gguf_with_id(
            make_llama_xdna_gguf_for_test(),
            MAX_KV_CACHE_TOKENS,
            "test-xdna-llama".into(),
        )
        .expect("load 16-layer cpu llama");
        let cfg = cpu.config().clone();
        let device = XdnaDevice::open(0).expect("fake open");
        let kernel = device
            .load_kernel(&[1, 2, 3, 4], &[0xAA55_0001])
            .expect("fake load");
        let head_bo_w = kernel.alloc_bo(64, 3).expect("head_bo_w");
        let head_bo_y = kernel.alloc_bo(HEAD_Y_BYTES, 4).expect("head_bo_y");
        let mut layer_bo_kv = Vec::with_capacity(N_LAYERS);
        let mut layer_bo_y = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            layer_bo_kv.push(kernel.alloc_bo(Q4LAYER_KV_BYTES, 6).expect("kv"));
            layer_bo_y.push(kernel.alloc_bo(Q4LAYER_ACT_BYTES, 4).expect("y"));
        }
        let bucket_chunks: Vec<usize> = Q4LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        let mut bucket_runlists = Vec::with_capacity(bucket_chunks.len());
        let mut head_runs = Vec::with_capacity(bucket_chunks.len());
        let mut layer_runs = Vec::with_capacity(bucket_chunks.len());
        for _ in &bucket_chunks {
            let run = kernel
                .create_run(&[&head_bo_w, &head_bo_y])
                .expect("dec run");
            let mut rl = kernel.create_runlist().expect("dec rl");
            rl.add(&run).expect("add");
            bucket_runlists.push(rl);
            head_runs.push(
                kernel
                    .create_run(&[&head_bo_w, &head_bo_y])
                    .expect("head run"),
            );
            layer_runs.push(vec![run]);
        }

        let prefill = if with_prefill {
            let pf_chunks: Vec<usize> =
                Q4PREFILL_LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
            let pf_bo_out = kernel.alloc_bo(Q4PREFILL_ACT_BYTES, 5).expect("pf_bo_out");
            let mut pf_bo_y = Vec::with_capacity(N_LAYERS);
            for _ in 0..N_LAYERS {
                pf_bo_y.push(kernel.alloc_bo(Q4PREFILL_ACT_BYTES, 4).expect("pf_bo_y"));
            }
            let pf_head_run = kernel
                .create_run(&[&head_bo_w, &head_bo_y])
                .expect("pf head");
            let mut pf_runlists = Vec::with_capacity(pf_chunks.len());
            let mut pf_layer_runs = Vec::with_capacity(pf_chunks.len());
            for _ in &pf_chunks {
                let run = kernel
                    .create_run(&[&head_bo_w, &pf_bo_out])
                    .expect("pf run");
                let mut rl = kernel.create_runlist().expect("pf rl");
                rl.add(&run).expect("pf add");
                pf_runlists.push(rl);
                pf_layer_runs.push(vec![run]);
            }
            Some(Box::new(XdnaPrefillState {
                bucket_chunks: pf_chunks,
                bucket_runlists: pf_runlists,
                head_run: pf_head_run,
                _layer_runs: pf_layer_runs,
                bo_out: pf_bo_out,
                bo_y: pf_bo_y,
                _bo_w: Vec::new(),
                _kernels: Vec::new(),
            }))
        } else {
            None
        };

        let npu = XdnaDecodeState {
            bucket_chunks,
            bucket_runlists,
            _head_runs: head_runs,
            _layer_runs: layer_runs,
            head_bo_y,
            _head_bo_w: head_bo_w,
            layer_bo_kv,
            layer_bo_y,
            _layer_bo_w: Vec::new(),
            _head_kernel: kernel,
            _layer_kernels: Vec::new(),
            prefill,
            _device: device,
            in_flight: None,
            synced_kv_len: 0,
            cached_rope: None,
            pending_cpu_kv: None,
            spare_kv: Some((
                vec![[0.0f32; KV_DIM]; N_LAYERS],
                vec![[0.0f32; KV_DIM]; N_LAYERS],
            )),
            q8_scales: vec![0.0f32; HIDDEN_SIZE / 32],
            q8_quants: vec![0i8; HIDDEN_SIZE],
        };
        (XdnaLlamaModel::from_parts_for_test(cpu, npu), cfg)
    }

    fn seed_llama_f16_state(cfg: &ModelConfig, pos: usize) -> InferenceState {
        let mut st =
            InferenceState::from_config_capped(cfg, &KvCompression::F16, MAX_KV_CACHE_TOKENS)
                .expect("state");
        let kv_row = vec![0.25f32; KV_DIM];
        for l in 0..N_LAYERS {
            for _ in 0..pos {
                st.append_kv_f16(l, &kv_row, &kv_row);
            }
        }
        st.seq_len = pos;
        st
    }

    #[test]
    fn coalesced_decode_layer_bo_kv_sync_range_across_chunk_boundaries() {
        let expected_size = Q4LAYER_SILU_HALF * 4 + HEAD_DIM * 2;
        for pos in [1usize, 63, 64, 4095] {
            let (_, events) = with_shim(FakeConfig::default(), || {
                let (model, cfg) = make_llama_xdna_model_for_test(false);
                let mut state = seed_llama_f16_state(&cfg, pos);
                {
                    let mut guard = model.lock_npu();
                    guard.synced_kv_len = pos;
                    guard.pending_cpu_kv = None;
                }
                let out = model.forward(&[1], pos, &mut state);
                assert_eq!(out.len(), cfg.vocab_size);
                assert_eq!(state.seq_len, pos + 1);
                assert!(model.take_decode_error().is_none());
            });

            let kv_handles: Vec<usize> = events
                .iter()
                .filter_map(|e| match e {
                    FakeEvent::BoAlloc { bytes, handle, .. } if *bytes == Q4LAYER_KV_BYTES => {
                        Some(*handle)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(kv_handles.len(), N_LAYERS);

            let kv_syncs: Vec<(usize, usize, usize)> = events
                .iter()
                .filter_map(|e| match e {
                    FakeEvent::BoSyncRange {
                        handle,
                        to_device: true,
                        size,
                        offset,
                    } if kv_handles.contains(handle) => Some((*handle, *size, *offset)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                kv_syncs.len(),
                N_LAYERS * N_KV_HEADS,
                "pos={pos}: expected 1 coalesced KV sync per head per layer"
            );

            let c = pos / Q4LAYER_ATTN_CHUNK_T;
            let rem = pos % Q4LAYER_ATTN_CHUNK_T;
            for (l, &handle) in kv_handles.iter().enumerate() {
                let mut layer_offs: Vec<(usize, usize)> = kv_syncs
                    .iter()
                    .filter(|(h, _, _)| *h == handle)
                    .map(|&(_, s, o)| (s, o))
                    .collect();
                assert_eq!(layer_offs.len(), N_KV_HEADS, "layer {l} pos={pos}");
                layer_offs.sort_by_key(|&(_, o)| o);
                for (kvh, (sz, off)) in layer_offs.into_iter().enumerate() {
                    let tile_start = (kvh * Q4LAYER_ATTN_CHUNKS + c) * Q4LAYER_TILE_F32 * 4;
                    let expected_off = tile_start + rem * HEAD_DIM * 2;
                    let tile_end = tile_start + Q4LAYER_TILE_F32 * 4;
                    assert_eq!(sz, expected_size, "pos={pos} layer={l} kvh={kvh}");
                    assert_eq!(off, expected_off, "pos={pos} layer={l} kvh={kvh}");
                    assert!(
                        off + sz <= tile_end,
                        "pos={pos} layer={l} kvh={kvh}: coalesced sync overflowed tile"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_prefill_runlist_and_head_run_timeouts_quiesce_in_flight() {
        // 1. Prefill bucket runlist timeout at `pos0 = 1` sets `InFlightXdnaOp::PrefillBucket(1)`
        // (`needed = 1.div_ceil(64) = 1`) and retains it on wait failure.
        with_shim(
            FakeConfig {
                fail_runlist_wait: true,
                ..FakeConfig::default()
            },
            || {
                let (model, cfg) = make_llama_xdna_model_for_test(true);
                let mut state = seed_llama_f16_state(&cfg, 1);
                model.lock_npu().synced_kv_len = 1;

                let out = model.forward_prefill(&[1; 16], 1, &mut state);
                assert!(out.iter().all(|&v| v == 0.0));
                assert!(model.state_torn.load(Ordering::SeqCst));
                assert!(model.take_decode_error().is_some());
                assert_eq!(
                    model.lock_npu().in_flight,
                    Some(InFlightXdnaOp::PrefillBucket(1))
                );
                assert!(quiesce_in_flight(&mut model.lock_npu()).is_err());
                assert_eq!(
                    model.lock_npu().in_flight,
                    Some(InFlightXdnaOp::PrefillBucket(1))
                );
                assert!(
                    model
                        .try_reset_kv(&mut state, &KvCompression::F16, 64)
                        .is_err()
                );
                assert!(model.state_torn.load(Ordering::SeqCst));
                // Clear handle before Drop since fake config rejects waits.
                model.lock_npu().in_flight = None;
            },
        );

        // 2. Prefill `head_run` timeout after bucket 1 completes sets `InFlightXdnaOp::PrefillHead`
        // and retains it on wait failure.
        let (_, head_timeout_events) = with_shim(
            FakeConfig {
                fail_runlist_wait: false,
                run_wait_state: Some(0),
                ..FakeConfig::default()
            },
            || {
                let (model, cfg) = make_llama_xdna_model_for_test(true);
                let mut state = seed_llama_f16_state(&cfg, 1);
                model.lock_npu().synced_kv_len = 1;

                let out = model.forward_prefill(&[1; 16], 1, &mut state);
                assert!(out.iter().all(|&v| v == 0.0));
                assert!(model.state_torn.load(Ordering::SeqCst));
                assert!(model.take_decode_error().is_some());
                assert_eq!(
                    model.lock_npu().in_flight,
                    Some(InFlightXdnaOp::PrefillHead)
                );
                assert!(quiesce_in_flight(&mut model.lock_npu()).is_err());
                assert_eq!(
                    model.lock_npu().in_flight,
                    Some(InFlightXdnaOp::PrefillHead)
                );
                assert!(
                    model
                        .try_reset_kv(&mut state, &KvCompression::F16, 64)
                        .is_err()
                );
                assert!(model.state_torn.load(Ordering::SeqCst));
                model.lock_npu().in_flight = None;
            },
        );
        assert_eq!(
            head_timeout_events
                .iter()
                .filter(|e| matches!(e, FakeEvent::RunListExecute(_)))
                .count(),
            1,
            "expected prefill bucket runlist execution before PrefillHead timeout"
        );
        assert_eq!(
            head_timeout_events
                .iter()
                .filter(|e| matches!(
                    e,
                    FakeEvent::RunListWait {
                        timeout_ms: QUIESCE_TIMEOUT_MS,
                        ..
                    }
                ))
                .count(),
            1,
            "expected prefill bucket runlist wait before PrefillHead timeout"
        );

        // 3. Healthy quiesce recovery drains pending `PrefillBucket(1)` before `BoFree` on Drop and clears `try_reset_kv`.
        let (_, events) = with_shim(FakeConfig::default(), || {
            let (model, cfg) = make_llama_xdna_model_for_test(true);
            let mut state = seed_llama_f16_state(&cfg, 1);
            {
                let mut guard = model.lock_npu();
                guard.prefill.as_ref().unwrap().bucket_runlists[1]
                    .execute()
                    .unwrap();
                guard.in_flight = Some(InFlightXdnaOp::PrefillBucket(1));
            }
            model.state_torn.store(true, Ordering::SeqCst);
            model
                .try_reset_kv(&mut state, &KvCompression::F16, 64)
                .expect("quiesced prefill bucket allows try_reset_kv recovery");
            assert!(!model.state_torn.load(Ordering::SeqCst));
            assert_eq!(model.lock_npu().in_flight, None);

            // Leave `PrefillHead` in-flight at Drop and assert Drop quiesces without error.
            model.lock_npu().in_flight = Some(InFlightXdnaOp::PrefillHead);
        });
        let wait_idx = events
            .iter()
            .position(|e| {
                matches!(
                    e,
                    FakeEvent::RunListWait {
                        timeout_ms: QUIESCE_TIMEOUT_MS,
                        ..
                    }
                )
            })
            .expect("try_reset_kv must wait for in-flight prefill bucket runlist");
        let first_bo_free = events
            .iter()
            .position(|e| matches!(e, FakeEvent::BoFree(_)))
            .expect("BOs freed on Drop");
        assert!(
            wait_idx < first_bo_free,
            "in-flight prefill bucket must quiesce before BOs are freed: {events:?}"
        );
    }

    #[test]
    fn sync_cpu_kv_to_npu_input_validation_errors_preserve_untorn_state() {
        with_shim(FakeConfig::default(), || {
            let (model, cfg) = make_llama_xdna_model_for_test(true);

            // Short final layer (layer 15 has only 1 row instead of 2) fails pre-validation cleanly
            // without touching `layer_bo_kv[0]` and without tearing `state_torn`.
            let mut short_state = seed_llama_f16_state(&cfg, 2);
            if let LayerState::Attention {
                key_cache_f16,
                value_cache_f16,
                ..
            } = &mut short_state.layers[N_LAYERS - 1]
            {
                key_cache_f16.truncate(KV_DIM);
                value_cache_f16.truncate(KV_DIM);
            }
            model.lock_npu().layer_bo_kv[0].as_mut_slice()[0] = 0x5A;

            let dec_err_out = model.forward(&[1], 2, &mut short_state);
            assert!(dec_err_out.iter().all(|&v| v == 0.0));
            assert!(
                !model.state_torn.load(Ordering::SeqCst),
                "short KV input error in decode must not set state_torn"
            );
            assert!(model.take_decode_error().is_some());
            assert_eq!(
                model.lock_npu().layer_bo_kv[0].as_slice()[0],
                0x5A,
                "pre-validation must reject before mutating earlier layer BOs"
            );

            // Same short-layer input error in `forward_prefill` keeps `state_torn` false.
            let pre_err_out = model.forward_prefill(&[1; 16], 2, &mut short_state);
            assert!(pre_err_out.iter().all(|&v| v == 0.0));
            assert!(
                !model.state_torn.load(Ordering::SeqCst),
                "short KV input error in prefill must not set state_torn"
            );
            assert!(model.take_decode_error().is_some());

            // Wrong layer kind (`LayerState::Conv` instead of `Attention`) returns `SyncErr::Input`.
            let mut conv_state = seed_llama_f16_state(&cfg, 2);
            conv_state.layers[0] = LayerState::Conv {
                buffer: Vec::new(),
                history: crate::kv_cache::ConvHistory::new(0),
            };
            let wrong_out = model.forward(&[1], 2, &mut conv_state);
            assert!(wrong_out.iter().all(|&v| v == 0.0));
            assert!(!model.state_torn.load(Ordering::SeqCst));
            assert!(model.take_decode_error().is_some());

            // Subsequent healthy `forward` still succeeds because NPU was never torn.
            let mut good_state = seed_llama_f16_state(&cfg, 2);
            let ok_out = model.forward(&[1], 2, &mut good_state);
            assert_eq!(ok_out.len(), cfg.vocab_size);
            assert!(model.take_decode_error().is_none());
            assert_eq!(good_state.seq_len, 3);
        });
    }
}

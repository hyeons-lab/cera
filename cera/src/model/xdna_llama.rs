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
    Q4HEAD_INSTS, Q4HEAD_XCLBIN, Q4LAYER_ACT_BYTES, Q4LAYER_ACT_SLOT, Q4LAYER_ATTN_N_OFF,
    Q4LAYER_ATTN_NORM_W, Q4LAYER_KV_BUCKETS, Q4LAYER_KV_CACHE_BASE, Q4LAYER_KV_SLOT,
    Q4LAYER_ROPE_CS, Q4LAYER_SILU_HALF, Q4LAYER_TILE_F32, Q4LAYER_XCLBIN, init_q4layer_bo_y_norms,
    inst_words, pack_q4layer_rope_cs, read_le_f32_slice, select_q4layer_kv_bucket,
    write_le_f32_slice, write_le_i32, write_q4layer_kv_bf16, write_q4layer_step_params,
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
use crate::quant::f32_to_f16;
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
const QUIESCE_TIMEOUT_MS: u32 = 30_000;

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

struct XdnaDecodeState {
    bucket_chunks: Vec<usize>,
    bucket_runlists: Vec<XdnaRunList>,
    _head_runs: Vec<XdnaRun>,
    _layer_runs: Vec<Vec<XdnaRun>>,
    head_bo_y: XdnaBo,
    _head_bo_w: XdnaBo,
    layer_bo_y: Vec<XdnaBo>,
    _layer_bo_w: Vec<XdnaBo>,
    _head_kernel: XdnaKernel,
    _layer_kernels: Vec<XdnaKernel>,
    _device: XdnaDevice,
    in_flight_bucket: Option<usize>,
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
    if let Some(bucket_idx) = npu.in_flight_bucket {
        npu.bucket_runlists[bucket_idx]
            .wait(QUIESCE_TIMEOUT_MS)
            .map_err(|e| {
                format!("failed to quiesce in-flight XDNA runlist (bucket {bucket_idx}): {e}")
            })?;
        npu.in_flight_bucket = None;
    }
    Ok(())
}

fn validate_xdna_kv_compression(
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

fn check_kv_rewind_state(
    is_causal: bool,
    state_torn: bool,
    pending_cpu_kv: Option<&(usize, LayerKvPair)>,
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

            bo_w.sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_w sync: {e}"))?;

            init_q4layer_bo_y_norms(
                bo_y.as_mut_slice(),
                cpu.attn_norm_slice(il),
                cpu.ffn_norm_slice(il),
            );
            bo_y.sync_to_device()
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_y sync: {e}"))?;

            layer_bo_w.push(bo_w);
            layer_bo_y.push(bo_y);
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
                    .create_run(&[&layer_bo_w[il], &layer_bo_y[il], y_next])
                    .map_err(|e| anyhow::anyhow!("XDNA layer {il} (c={chunks}) create_run: {e}"))?;
                b_runs.push(run);
            }
            let h_run = head_kernel
                .create_run(&[&head_bo_w, &head_bo_y, &head_bo_y])
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

        Ok(Self {
            cpu,
            npu: Mutex::new(XdnaDecodeState {
                bucket_chunks,
                bucket_runlists,
                _head_runs: head_runs,
                _layer_runs: layer_runs,
                head_bo_y,
                _head_bo_w: head_bo_w,
                layer_bo_y,
                _layer_bo_w: layer_bo_w,
                _head_kernel: head_kernel,
                _layer_kernels: layer_kernels,
                _device: device,
                in_flight_bucket: None,
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

    fn compute_rope_cs(&self, pos: usize) -> [f32; 128] {
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
        pack_q4layer_rope_cs(&rope_cos, &rope_sin)
    }

    fn sync_cpu_kv_to_npu(
        &self,
        npu: &mut XdnaDecodeState,
        state: &mut InferenceState,
        target_pos: usize,
    ) -> Result<(), String> {
        if state.is_compressed() {
            return Err(
                "TurboQuant KV compression is not supported by the XDNA backend".to_string(),
            );
        }
        if state.layers.len() < N_LAYERS {
            return Err(format!(
                "InferenceState has {} layers, expected at least {N_LAYERS}",
                state.layers.len()
            ));
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
            .ok_or_else(|| "target_pos overflows KV cache length".to_string())?;
        let start = npu.synced_kv_len;
        let use_f16 = state.kv_f16;
        for l in 0..N_LAYERS {
            let y_bytes = npu.layer_bo_y[l].as_mut_slice();
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
                            return Err(format!(
                                "layer {l} f16 KV cache has {}/{} elements, expected at least {needed}",
                                key_cache_f16.len(),
                                value_cache_f16.len()
                            ));
                        }
                    } else if key_cache.len() < needed || value_cache.len() < needed {
                        return Err(format!(
                            "layer {l} f32 KV cache has {}/{} elements, expected at least {needed}",
                            key_cache.len(),
                            value_cache.len()
                        ));
                    }
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
                                write_q4layer_kv_bf16(y_bytes, kvh, t, &kh, &vh);
                            } else {
                                write_q4layer_kv_bf16(
                                    y_bytes,
                                    kvh,
                                    t,
                                    &key_cache[src_fl..src_fl + HEAD_DIM],
                                    &value_cache[src_fl..src_fl + HEAD_DIM],
                                );
                            }
                        }
                    }
                }
                _ => {
                    return Err(format!(
                        "layer {l} does not have an Attention KV state for XDNA"
                    ));
                }
            }
            npu.layer_bo_y[l]
                .sync_range_to_device(Q4LAYER_KV_SLOT * 4, Q4LAYER_KV_CACHE_BASE * 4)
                .map_err(|e| format!("layer {l} KV cache sync to device: {e}"))?;
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
        }
        self.cpu.forward_prefill(tokens, start_pos, state)
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

        // Positions beyond the 648-token NPU KV stream layout (or a raw caller
        // passing `state.lora`) fall back to CPU decode seamlessly (`state.layers`
        // holds the full KV history because every NPU decode step mirrors K/V).
        if pos >= MAX_KV_CACHE_TOKENS || state.lora.is_some() {
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

        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            return self.record_device_fault(
                "XDNA NPU state torn after a fault; reset the session".to_string(),
                vocab_size,
            );
        }
        if let Err(e) = quiesce_in_flight(&mut npu) {
            return self
                .record_device_fault(format!("forward failed to quiesce NPU: {e}"), vocab_size);
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
        if pos == 0 {
            Self::discard_pending_cpu_kv(&mut npu);
        }
        if npu.synced_kv_len != pos
            && let Err(e) = self.sync_cpu_kv_to_npu(&mut npu, state, pos)
        {
            return self.record_device_fault(e, vocab_size);
        }

        let mut hidden = self.cpu.embed_token(tokens[0], "forward");

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
                return self.record_device_fault(
                    format!("layer 0 hidden sync to device: {e}"),
                    vocab_size,
                );
            }
            if let Err(e) = npu.layer_bo_y[l].sync_range_to_device(132 * 4, Q4LAYER_ROPE_CS * 4) {
                return self.record_device_fault(
                    format!("layer {l} bo_y sync to device: {e}"),
                    vocab_size,
                );
            }
        }
        let t_crit_sync = tc0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let t0 = prof.then(crate::time::Instant::now);
        npu.in_flight_bucket = Some(bucket_idx);
        if let Err(e) = npu.bucket_runlists[bucket_idx].execute() {
            npu.in_flight_bucket = None;
            return self.record_device_fault(format!("token runlist execute: {e}"), vocab_size);
        }
        // Overlap host-side allocations, previous token's f16 CPU KV cache
        // extension, and next token's RoPE cos/sin table while the NPU runs.
        let mut logits = vec![0.0f32; vocab_size];
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
            return self.record_device_fault(format!("token runlist wait: {e}"), vocab_size);
        }
        npu.in_flight_bucket = None;
        let t_npu_wait = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let tkv0 = prof.then(crate::time::Instant::now);
        for l in 0..N_LAYERS {
            if transformer::oracle_dump::is_active() {
                {
                    let next_bo = if l + 1 < N_LAYERS {
                        &mut npu.layer_bo_y[l + 1]
                    } else {
                        &mut npu.head_bo_y
                    };
                    if let Err(e) = next_bo.sync_range_from_device(HIDDEN_SIZE * 4, 0) {
                        npu.spare_kv = Some((pending_k, pending_v));
                        return self.record_device_fault(
                            format!("layer {l} out sync from device: {e}"),
                            vocab_size,
                        );
                    }
                    read_le_f32_slice(next_bo.as_slice(), 0, &mut hidden);
                }
                transformer::oracle_dump::record(&format!("l_out-{l}"), &hidden);
                if l + 1 == N_LAYERS {
                    let mut normed = vec![0.0f32; HIDDEN_SIZE];
                    let XdnaDecodeState {
                        ref mut q8_scales,
                        ref mut q8_quants,
                        ..
                    } = *npu;
                    cpu::rmsnorm_and_quantize_q8_0(
                        &hidden,
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
                return self.record_device_fault(
                    format!("layer {l} slot 2 sync from device: {e}"),
                    vocab_size,
                );
            }
            if let Err(e) =
                npu.layer_bo_y[l].sync_range_from_device(KV_DIM * 4, 3 * Q4LAYER_ACT_SLOT * 4)
            {
                npu.spare_kv = Some((pending_k, pending_v));
                return self.record_device_fault(
                    format!("layer {l} slot 3 sync from device: {e}"),
                    vocab_size,
                );
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
                let yl_mut = npu.layer_bo_y[l].as_mut_slice();
                for (kvh, k_off) in k_offs.iter_mut().enumerate() {
                    let src_fl = kvh * HEAD_DIM;
                    *k_off = write_q4layer_kv_bf16(
                        yl_mut,
                        kvh,
                        pos,
                        &k_cur[src_fl..src_fl + HEAD_DIM],
                        &v_cur[src_fl..src_fl + HEAD_DIM],
                    );
                }
            }
            for &k_byte_off in &k_offs {
                if let Err(e) = npu.layer_bo_y[l].sync_range_to_device(HEAD_DIM * 2, k_byte_off) {
                    npu.spare_kv = Some((pending_k, pending_v));
                    return self.record_device_fault(
                        format!("layer {l} k sync to device: {e}"),
                        vocab_size,
                    );
                }
                if let Err(e) = npu.layer_bo_y[l]
                    .sync_range_to_device(HEAD_DIM * 2, k_byte_off + Q4LAYER_SILU_HALF * 4)
                {
                    npu.spare_kv = Some((pending_k, pending_v));
                    return self.record_device_fault(
                        format!("layer {l} v sync to device: {e}"),
                        vocab_size,
                    );
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
            return self
                .record_device_fault(format!("head bo_y sync from device: {e}"), vocab_size);
        }
        let t_head_sync = th0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        if prof && !(20..520).contains(&pos) {
            eprintln!(
                "xdna prof pos={pos}: npu_wait={t_npu_wait:.0}us crit_sync={t_crit_sync:.0}us kv_sync={t_kv_sync:.0}us head_sync={t_head_sync:.0}us"
            );
        }

        read_le_f32_slice(npu.head_bo_y.as_slice(), Q4LAYER_TILE_F32, &mut logits);
        if cfg.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits, 1.0 / cfg.scalars.logit);
        }

        state.seq_len = pos + 1;
        npu.pending_cpu_kv = Some((pos, (pending_k, pending_v)));
        if pos + 1 >= MAX_KV_CACHE_TOKENS {
            Self::flush_pending_cpu_kv(&mut npu, state);
        }
        npu.synced_kv_len = pos + 1;
        transformer::oracle_dump::record("result_output", &logits);
        logits
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
    Ok(Box::new(XdnaLlamaModel::from_gguf_with_id(
        gguf,
        context_size,
        model_id,
    )?))
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
        assert_eq!(select_q4layer_kv_bucket(&buckets, 0), (0, 8));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 1), (0, 8));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 72), (0, 8));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 73), (1, 7));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 144), (1, 7));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 145), (2, 6));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 576), (7, 1));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 577), (8, 0));
        assert_eq!(select_q4layer_kv_bucket(&buckets, 647), (8, 0));
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
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                _device: device,
                in_flight_bucket: None,
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
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                _device: device,
                in_flight_bucket: Some(0),
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
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                _device: device,
                in_flight_bucket: Some(0),
                synced_kv_len: 0,
                cached_rope: None,
                pending_cpu_kv: None,
                spare_kv: None,
                q8_scales: Vec::new(),
                q8_quants: Vec::new(),
            };
            assert!(quiesce_in_flight(&mut state).is_err());
            assert_eq!(state.in_flight_bucket, Some(0));
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
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                _device: device,
                in_flight_bucket: None,
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
                layer_bo_y: Vec::new(),
                _layer_bo_w: Vec::new(),
                _head_kernel: kernel,
                _layer_kernels: Vec::new(),
                _device: device,
                in_flight_bucket: None,
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
        });
    }
}

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
use crate::backend::xdna::kernels::{
    Q4HEAD_INSTS, Q4HEAD_XCLBIN, Q4LAYER_KV_BUCKETS, Q4LAYER_XCLBIN, inst_words,
};
use crate::backend::xdna::repack::{
    f32_to_bf16, interleave_worker_stages, pack_f32_stage, pack_q4_0_stage, stage_bytes,
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

const ACT_SLOT: usize = 8192;
const N_SLOTS: usize = 8;
const ATTN_NORM_W: usize = 2048;
const ROPE_CS: usize = 4096;
const FFN_NORM_W: usize = 2048;

const TILE_F32: usize = 4608;
const SILU_HALF: usize = 2304;
const ATTN_CHUNK_T: usize = 72;
const ATTN_CHUNKS: usize = 9;
/// Maximum number of prior cached tokens supported by `q4layer`'s 9x72-token
/// bf16 KV stream (`pos < 648`, i.e. up to 649 total tokens including current).
pub const MAX_KV_CACHE_TOKENS: usize = ATTN_CHUNKS * ATTN_CHUNK_T;
const KV_HEAD_STRIDE: usize = ATTN_CHUNKS * TILE_F32;
const KV_SLOT: usize = N_KV_HEADS * KV_HEAD_STRIDE;
const KV_CACHE_BASE: usize = N_SLOTS * ACT_SLOT;
const ACT_FLOATS: usize = KV_CACHE_BASE + KV_SLOT;
const ACT_BYTES: usize = ACT_FLOATS * 4;
const ATTN_N_OFF: usize = ROPE_CS + 128;
const ATTN_SKIP_KV_OFF: usize = ROPE_CS + 131;

const HEAD_Y_BYTES: usize = (TILE_F32 + HEAD_PADDED_VOCAB) * 4;

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
    synced_kv_len: usize,
    cached_rope: Option<(usize, [f32; 128])>,
    pending_cpu_kv: Option<(usize, LayerKvPair)>,
    spare_kv: Option<LayerKvPair>,
    q8_scales: Vec<f32>,
    q8_quants: Vec<i8>,
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
        }
    }

    let head_ref = cpu.lm_head_weight_ref();
    ensure!(
        head_ref.k == HIDDEN_SIZE && head_ref.m >= cfg.vocab_size,
        "XDNA LM head shape mismatch: got {}x{}",
        head_ref.m,
        head_ref.k
    );
    Ok(())
}

fn select_kv_bucket(bucket_chunks: &[usize], pos: usize) -> (usize, i32) {
    let needed_chunks = pos.max(1).div_ceil(ATTN_CHUNK_T);
    let bucket_idx = bucket_chunks
        .iter()
        .position(|&c| c >= needed_chunks)
        .unwrap_or(bucket_chunks.len() - 1);
    let skip_kv = (ATTN_CHUNKS - bucket_chunks[bucket_idx]) as i32;
    (bucket_idx, skip_kv)
}

fn parse_profile_decode(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .is_some_and(|v| !v.is_empty() && v != "0")
}

fn profile_decode_enabled() -> bool {
    #[cfg(test)]
    {
        false
    }
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            parse_profile_decode(std::env::var("CERA_PROFILE_DECODE").ok().as_deref())
        })
    }
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
        let cpu = LlamaModel::from_gguf_with_id(gguf, context_size, model_id)?;
        validate_xdna_llama_cpu(&cpu)?;
        let cfg = cpu.config();
        let gguf_ref = cpu.gguf_file();

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
                .alloc_bo(ACT_BYTES, 4)
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
            interleave_worker_stages(&mut w_slice[..qkv_bytes], &[&q_tmp, &k_tmp, &v_tmp])
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
            interleave_worker_stages(&mut w_slice[gu_off..down_off], &[&gate_tmp, &up_tmp])
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} gate+up interleave: {e}"))?;

            pack_one_stage(
                &mut w_slice[down_off..],
                stage_refs[6],
                LAYER_STAGE_SHAPES[6].0,
                LAYER_STAGE_SHAPES[6].1,
                il,
                6,
            )?;

            bo_w.sync(true)
                .map_err(|e| anyhow::anyhow!("XDNA layer {il} bo_w sync: {e}"))?;

            let y_slice = bo_y.as_mut_slice();
            y_slice.fill(0);
            write_f32_slice(y_slice, ATTN_NORM_W, cpu.attn_norm_slice(il));
            write_f32_slice(y_slice, 4 * ACT_SLOT + FFN_NORM_W, cpu.ffn_norm_slice(il));
            bo_y.sync(true)
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
        write_f32_slice(
            head_bo_y.as_mut_slice(),
            ATTN_NORM_W,
            cpu.output_norm_slice(),
        );
        write_i32(head_bo_y.as_mut_slice(), ATTN_N_OFF, -1);
        write_f32_slice(head_bo_y.as_mut_slice(), ATTN_N_OFF + 1, &[1.0, 0.0]);
        head_bo_y
            .sync(true)
            .map_err(|e| anyhow::anyhow!("XDNA head bo_y sync: {e}"))?;

        let head_ref = cpu.lm_head_weight_ref();

        let chunk_packed_bytes =
            stage_bytes(HEAD_REPACK_CHUNK_ROWS, HIDDEN_SIZE).map_err(|e| anyhow::anyhow!(e))?;
        let head_w_slice = head_bo_w.as_mut_slice();
        if head_ref.dtype == DType::Q4_0 && head_ref.m == cfg.vocab_size {
            let src = transformer::weight_data(gguf_ref, head_ref);
            let q4_row_bytes = (HIDDEN_SIZE / 32) * 18;
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
                let _ = q4_row_bytes;
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
            .sync(true)
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
        })
    }

    fn lock_npu(&self) -> MutexGuard<'_, XdnaDecodeState> {
        lock_npu_state(&self.npu, &self.state_torn)
    }

    fn flush_pending_cpu_kv(npu: &mut XdnaDecodeState, state: &mut InferenceState) {
        let Some((pending_pos, (pending_k, pending_v))) = npu.pending_cpu_kv.take() else {
            return;
        };
        let use_f16 = state.kv_f16;
        let expected = pending_pos * KV_DIM;
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
                    if key_cache_f16.len() > expected {
                        key_cache_f16.truncate(expected);
                        value_cache_f16.truncate(expected);
                    }
                    key_cache_f16.extend(pending_k[l].iter().copied().map(f32_to_f16));
                    value_cache_f16.extend(pending_v[l].iter().copied().map(f32_to_f16));
                } else {
                    if key_cache.len() > expected {
                        key_cache.truncate(expected);
                        value_cache.truncate(expected);
                    }
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
        let mut rope_cs = [0.0f32; 128];
        for h in 0..2 {
            let base = h * 64;
            rope_cs[base..base + 16].copy_from_slice(&rope_cos[..16]);
            rope_cs[base + 16..base + 32].copy_from_slice(&rope_sin[..16]);
            rope_cs[base + 32..base + 48].copy_from_slice(&rope_cos[16..32]);
            rope_cs[base + 48..base + 64].copy_from_slice(&rope_sin[16..32]);
        }
        rope_cs
    }

    fn sync_cpu_kv_to_npu(
        &self,
        npu: &mut XdnaDecodeState,
        state: &mut InferenceState,
        target_pos: usize,
    ) -> Result<(), String> {
        Self::flush_pending_cpu_kv(npu, state);
        if target_pos < npu.synced_kv_len {
            npu.synced_kv_len = target_pos;
        }
        if npu.synced_kv_len >= target_pos {
            return Ok(());
        }
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
                                write_kv_bf16(y_bytes, kvh, t, &kh, &vh);
                            } else {
                                write_kv_bf16(
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
                _ => unreachable!("LlamaModel layer {l} must have LayerState::Attention"),
            }
            npu.layer_bo_y[l]
                .sync_range(true, KV_SLOT * 4, KV_CACHE_BASE * 4)
                .map_err(|e| format!("layer {l} KV cache sync to device: {e}"))?;
        }
        npu.synced_kv_len = target_pos;
        Ok(())
    }

    fn record_fault(&self, msg: String) -> Vec<f32> {
        self.state_torn.store(true, Ordering::SeqCst);
        tracing::error!("cera::xdna: {msg}");
        eprintln!("cera::xdna: {msg}");
        record_first_fault(&self.decode_error, CeraError::Backend(msg));
        vec![0.0f32; self.cpu.config().vocab_size]
    }
}

#[inline]
fn write_kv_bf16(buf: &mut [u8], kvh: usize, pos: usize, k_head: &[f32], v_head: &[f32]) -> usize {
    let c = pos / ATTN_CHUNK_T;
    let rem = pos % ATTN_CHUNK_T;
    let chunk_byte_off = (KV_CACHE_BASE + (kvh * ATTN_CHUNKS + c) * TILE_F32) * 4;
    let k_byte_off = chunk_byte_off + rem * HEAD_DIM * 2;
    let v_byte_off = k_byte_off + SILU_HALF * 4;
    for d in 0..HEAD_DIM {
        let kb = f32_to_bf16(k_head[d]).to_le_bytes();
        let vb = f32_to_bf16(v_head[d]).to_le_bytes();
        buf[k_byte_off + d * 2..k_byte_off + d * 2 + 2].copy_from_slice(&kb);
        buf[v_byte_off + d * 2..v_byte_off + d * 2 + 2].copy_from_slice(&vb);
    }
    k_byte_off
}

#[inline]
fn write_i32(buf: &mut [u8], word_idx: usize, val: i32) {
    let off = word_idx * 4;
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_f32_slice(buf: &mut [u8], float_idx: usize, vals: &[f32]) {
    let off = float_idx * 4;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

#[inline]
fn read_f32_slice(buf: &[u8], float_idx: usize, dst: &mut [f32]) {
    let off = float_idx * 4;
    let bytes = &buf[off..off + dst.len() * 4];
    let dst_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 4) };
    dst_bytes.copy_from_slice(bytes);
}

impl Model for XdnaLlamaModel {
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.session_gate.try_acquire().map(Some)
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

    fn config(&self) -> &ModelConfig {
        self.cpu.config()
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        let mut npu = self.lock_npu();
        Self::flush_pending_cpu_kv(&mut npu, state);
        state.truncate_to(len);
        npu.synced_kv_len = npu.synced_kv_len.min(len);
    }

    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        let _npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            return Err(KvRewindError::BackendUnsupported);
        }
        self.cpu.check_kv_rewind(state, len)
    }

    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
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
        let mut npu = self.lock_npu();
        if let Some((_, kv)) = npu.pending_cpu_kv.take() {
            npu.spare_kv = Some(kv);
        }
        self.cpu.try_reset_kv(state, compression, max_seq_len)?;
        npu.synced_kv_len = 0;
        self.state_torn.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        {
            let mut npu = self.lock_npu();
            if self.state_torn.load(Ordering::SeqCst) {
                return self.record_fault(
                    "XDNA NPU state torn after a fault; reset the session".to_string(),
                );
            }
            if start_pos == 0 {
                if let Some((_, kv)) = npu.pending_cpu_kv.take() {
                    npu.spare_kv = Some(kv);
                }
                npu.synced_kv_len = 0;
            } else {
                Self::flush_pending_cpu_kv(&mut npu, state);
            }
        }
        self.cpu.forward_prefill(tokens, start_pos, state)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "XdnaLlamaModel::forward expects 1 token");

        // Positions beyond the 648-token NPU KV stream layout (or a raw caller
        // passing `state.lora`) fall back to CPU decode seamlessly (`state.layers`
        // holds the full KV history because every NPU decode step mirrors K/V).
        if pos >= MAX_KV_CACHE_TOKENS || state.lora.is_some() {
            {
                let mut npu = self.lock_npu();
                if self.state_torn.load(Ordering::SeqCst) {
                    return self.record_fault(
                        "XDNA NPU state torn after a fault; reset the session".to_string(),
                    );
                }
                Self::flush_pending_cpu_kv(&mut npu, state);
            }
            return self.cpu.forward(tokens, pos, state);
        }

        let mut npu = self.lock_npu();
        if self.state_torn.load(Ordering::SeqCst) {
            return self
                .record_fault("XDNA NPU state torn after a fault; reset the session".to_string());
        }
        if pos == 0 {
            if let Some((_, kv)) = npu.pending_cpu_kv.take() {
                npu.spare_kv = Some(kv);
            }
            npu.synced_kv_len = 0;
        }
        if npu.synced_kv_len != pos
            && let Err(e) = self.sync_cpu_kv_to_npu(&mut npu, state, pos)
        {
            return self.record_fault(e);
        }

        let cfg = self.cpu.config();
        let mut hidden = self.cpu.embed_token(tokens[0], "forward");

        let rope_cs = match npu.cached_rope {
            Some((cached_pos, cs)) if cached_pos == pos => cs,
            _ => self.compute_rope_cs(pos),
        };

        let (bucket_idx, skip_kv) = select_kv_bucket(&npu.bucket_chunks, pos);

        let prof = profile_decode_enabled();
        let tc0 = prof.then(std::time::Instant::now);
        for l in 0..N_LAYERS {
            let yl = npu.layer_bo_y[l].as_mut_slice();
            if l == 0 {
                write_f32_slice(yl, 0, &hidden);
            }
            write_f32_slice(yl, ROPE_CS, &rope_cs);
            write_i32(yl, ATTN_N_OFF, pos as i32);
            write_i32(yl, ATTN_SKIP_KV_OFF, skip_kv);
            if l == 0
                && let Err(e) = npu.layer_bo_y[0].sync_range(true, HIDDEN_SIZE * 4, 0)
            {
                return self.record_fault(format!("layer 0 hidden sync to device: {e}"));
            }
            if let Err(e) = npu.layer_bo_y[l].sync_range(true, 132 * 4, ROPE_CS * 4) {
                return self.record_fault(format!("layer {l} bo_y sync to device: {e}"));
            }
        }
        let t_crit_sync = tc0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let t0 = prof.then(std::time::Instant::now);
        if let Err(e) = npu.bucket_runlists[bucket_idx].execute() {
            return self.record_fault(format!("token runlist execute: {e}"));
        }
        // Overlap host-side allocations, previous token's f16 CPU KV cache
        // extension, and next token's RoPE cos/sin table while the NPU runs.
        let mut logits = vec![0.0f32; cfg.vocab_size];
        Self::flush_pending_cpu_kv(&mut npu, state);
        npu.cached_rope = Some((pos + 1, self.compute_rope_cs(pos + 1)));
        let (mut pending_k, mut pending_v) = npu.spare_kv.take().unwrap_or_else(|| {
            (
                vec![[0.0f32; KV_DIM]; N_LAYERS],
                vec![[0.0f32; KV_DIM]; N_LAYERS],
            )
        });
        if let Err(e) = npu.bucket_runlists[bucket_idx].wait(0) {
            return self.record_fault(format!("token runlist wait: {e}"));
        }
        let t_npu_wait = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let tkv0 = prof.then(std::time::Instant::now);
        for l in 0..N_LAYERS {
            if transformer::oracle_dump::is_active() {
                let next_bo = if l + 1 < N_LAYERS {
                    &npu.layer_bo_y[l + 1]
                } else {
                    &npu.head_bo_y
                };
                if let Err(e) = next_bo.sync_range(false, HIDDEN_SIZE * 4, 0) {
                    return self.record_fault(format!("layer {l} out sync from device: {e}"));
                }
                read_f32_slice(next_bo.as_slice(), 0, &mut hidden);
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

            if let Err(e) = npu.layer_bo_y[l].sync_range(false, KV_DIM * 4, 2 * ACT_SLOT * 4) {
                return self.record_fault(format!("layer {l} slot 2 sync from device: {e}"));
            }
            if let Err(e) = npu.layer_bo_y[l].sync_range(false, KV_DIM * 4, 3 * ACT_SLOT * 4) {
                return self.record_fault(format!("layer {l} slot 3 sync from device: {e}"));
            }
            let k_cur = &mut pending_k[l];
            let v_cur = &mut pending_v[l];
            {
                let yl = npu.layer_bo_y[l].as_slice();
                read_f32_slice(yl, 2 * ACT_SLOT, k_cur);
                read_f32_slice(yl, 3 * ACT_SLOT, v_cur);
            }
            let mut k_offs = [0usize; N_KV_HEADS];
            {
                let yl_mut = npu.layer_bo_y[l].as_mut_slice();
                for (kvh, k_off) in k_offs.iter_mut().enumerate() {
                    let src_fl = kvh * HEAD_DIM;
                    *k_off = write_kv_bf16(
                        yl_mut,
                        kvh,
                        pos,
                        &k_cur[src_fl..src_fl + HEAD_DIM],
                        &v_cur[src_fl..src_fl + HEAD_DIM],
                    );
                }
            }
            for &k_byte_off in &k_offs {
                if let Err(e) = npu.layer_bo_y[l].sync_range(true, HEAD_DIM * 2, k_byte_off) {
                    return self.record_fault(format!("layer {l} k sync to device: {e}"));
                }
                if let Err(e) =
                    npu.layer_bo_y[l].sync_range(true, HEAD_DIM * 2, k_byte_off + SILU_HALF * 4)
                {
                    return self.record_fault(format!("layer {l} v sync to device: {e}"));
                }
            }
        }
        npu.pending_cpu_kv = Some((pos, (pending_k, pending_v)));
        if pos + 1 >= MAX_KV_CACHE_TOKENS {
            Self::flush_pending_cpu_kv(&mut npu, state);
        }
        let t_kv_sync = tkv0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);

        let th0 = prof.then(std::time::Instant::now);
        if let Err(e) = npu
            .head_bo_y
            .sync_range(false, cfg.vocab_size * 4, TILE_F32 * 4)
        {
            return self.record_fault(format!("head bo_y sync from device: {e}"));
        }
        let t_head_sync = th0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
        if prof && !(20..520).contains(&pos) {
            eprintln!(
                "xdna prof pos={pos}: npu_wait={t_npu_wait:.0}us crit_sync={t_crit_sync:.0}us kv_sync={t_kv_sync:.0}us head_sync={t_head_sync:.0}us"
            );
        }

        read_f32_slice(npu.head_bo_y.as_slice(), TILE_F32, &mut logits);
        if cfg.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits, 1.0 / cfg.scalars.logit);
        }

        state.seq_len = pos + 1;
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
    use crate::backend::xdna::sys::fake::{Config as FakeConfig, with_shim};

    #[test]
    fn parse_profile_decode_accepts_truthy_and_ignores_disabled() {
        assert!(!parse_profile_decode(None));
        assert!(!parse_profile_decode(Some("")));
        assert!(!parse_profile_decode(Some("   ")));
        assert!(!parse_profile_decode(Some("0")));
        assert!(!parse_profile_decode(Some(" 0 ")));
        assert!(parse_profile_decode(Some("1")));
        assert!(parse_profile_decode(Some("true")));
    }

    #[test]
    fn kv_bucket_selection_covers_all_positions() {
        let buckets: Vec<usize> = Q4LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        assert_eq!(select_kv_bucket(&buckets, 0), (0, 8));
        assert_eq!(select_kv_bucket(&buckets, 1), (0, 8));
        assert_eq!(select_kv_bucket(&buckets, 72), (0, 8));
        assert_eq!(select_kv_bucket(&buckets, 73), (1, 7));
        assert_eq!(select_kv_bucket(&buckets, 144), (1, 7));
        assert_eq!(select_kv_bucket(&buckets, 145), (2, 6));
        assert_eq!(select_kv_bucket(&buckets, 576), (7, 1));
        assert_eq!(select_kv_bucket(&buckets, 577), (8, 0));
        assert_eq!(select_kv_bucket(&buckets, 647), (8, 0));
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
}

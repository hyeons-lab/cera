//! Native Qualcomm Hexagon NPU model implementation for LFM2 hybrid architectures.
//!
//! Provides HTP-accelerated forward execution on Snapdragon mobile and edge platforms
//! using FastRPC shared memory and per-architecture DSP skeleton libraries.

// The dispatch_* builders below mirror llama.cpp's fixed C signatures
// (session + per-tensor buffer/offset/flags + dims); bundling into structs
// would diverge from that truth at every call site for no gain.
#![allow(clippy::too_many_arguments)]

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::cpu::RopeType;
use crate::backend::hexagon::{
    AdpfSession, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch,
    HexagonContext, HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, RpcmemBuffer,
    TILE_SIZE_Q4_0, TILE_SIZE_Q4_K, TILE_SIZE_Q6_K, TILE_SIZE_Q8_0, build_binary_kernel_params,
    build_flash_attn_kernel_params, build_hmx_fa_kernel_params, build_hmx_mm_kernel_params,
    build_mul_mat_kernel_params, build_rms_norm_params, build_rope_kernel_params,
    build_rope_params, build_set_rows_kernel_params, build_ssm_conv_kernel_params,
    build_unary_kernel_params, fa_is_hmx_eligible, mm_hmx_nb1, mm_is_hmx_eligible, repack_q4_0,
    repack_q4_k, repack_q6_k, repack_q8_0, repacked_matrix_size_q4_0, repacked_matrix_size_q4_k,
    repacked_matrix_size_q6_k, repacked_matrix_size_q8_0, requant_q5_k_to_q8_0,
};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError};
use crate::model::session_gate::{ModelSessionGate, ModelSessionLease};
use crate::model::{BlockType, Model, ModelConfig};
use crate::session::CeraError;

#[derive(Clone, Copy, Debug)]
struct HexagonWeight {
    offset: usize,
    in_dim: usize,
    out_dim: usize,
    wire_dtype: HtpDataType,
    block_bytes: usize,
    tile_size: usize,
}

#[derive(Clone, Copy, Debug)]
struct HexagonAttentionLayer {
    attn_norm_offset: usize,
    attn_q: HexagonWeight,
    attn_k: HexagonWeight,
    attn_v: HexagonWeight,
    attn_output: HexagonWeight,
    attn_q_norm_offset: Option<usize>,
    attn_k_norm_offset: Option<usize>,
    ffn_norm_offset: usize,
    ffn_gate: HexagonWeight,
    ffn_up: HexagonWeight,
    ffn_down: HexagonWeight,
    k_offset: usize,
    v_offset: usize,
    q_dim: usize,
    kv_dim: usize,
}

#[derive(Clone, Copy, Debug)]
struct HexagonConvLayer {
    attn_norm_offset: usize,
    in_proj: HexagonWeight,
    out_proj: HexagonWeight,
    conv_w0_offset: usize,
    conv_w1_offset: usize,
    conv_w2_offset: usize,
    /// Interleaved `[3, C]` short-conv taps (oldest-first) for SsmConv dispatch.
    conv_ssm_offset: usize,
    /// Recurrent short-conv state, channel-interleaved `[C, 2]` (slot t
    /// at `state + c*8 + t*4`, oldest-first) in one allocation. The DSP's
    /// transposed-CONCAT worker fetches its s0 side as 2-float pairs at
    /// 8-byte stride, so this layout is what the fast prepend path reads
    /// natively; dense row-major state cannot feed that worker.
    state_offset: usize,
    ffn_norm_offset: usize,
    ffn_gate: HexagonWeight,
    ffn_up: HexagonWeight,
    ffn_down: HexagonWeight,
}

enum HexagonLayer {
    Attention(HexagonAttentionLayer),
    Conv(HexagonConvLayer),
}

/// Reserve `size` bytes in a 256-aligned running total, returning the offset.
fn plan_offset(total: &mut usize, size: usize) -> usize {
    let offset = (*total + 255) & !255;
    *total = offset + size;
    offset
}

/// Prefill chunk rows: activation scratch is sized for this many tokens.
/// HMX double-buffers past 32 rows; VTCM fit is enforced per-op by the
/// HMX chunk solvers (HVX fallback covers M <= 4).
const PREFILL_MAX_ROWS: usize = 512;

/// Chunks below this many rows cap ops per flush (`MAX_OPS_PER_FLUSH`).
/// Large single-flush batches compute nondeterministically: run-to-run logit
/// swings up to ±3.6 at m <= 14 (m >= 15 bit-clean across 1..128) and
/// greedy-decode flips. The corruption needs a large co-batched window;
/// 24 ops/flush is verified bit-clean across prefill shapes and prompts
/// (single and chunked), so small chunks take the cap while large chunks
/// keep single-flush speed. Decode uses the stricter `DECODE_OPS_DEFAULT`
/// (sequential tokens amplify the race: cap 24 flips long greedy runs).
const SMALL_M_FLUSH_CAP_ROWS: usize = 32;
/// Ops-per-flush cap for small-M prefill chunks (see above).
const MAX_OPS_PER_FLUSH: usize = 24;

/// Prefill chunks below this many rows flush after every layer. The
/// cap-24 window still races at tiny M (m = 2,4,6,7 nondeterministic
/// run-to-run; m >= 8 bit-clean): per-layer flushes pin the window
/// phase and restore determinism. Independent of the HMX M gate above
/// (different mechanism: the race persists with HMX fully disabled).
const SMALL_M_BARRIER_ROWS: usize = 8;

/// Default decode ops-per-flush cap. The cap is phase-sensitive, not a
/// safety threshold: cap 20 was clean, then adding conv state-copy ops
/// re-phased its windows back into the race (6/6 64-token greedy md5s
/// diverged). 12 is verified 29/30 over 64-token greedy runs (2 prompts
/// x 2 quants); the single miss is a one-token near-tie flip between two
/// sane attractors (hex consensus == CPU text exactly), cap-independent
/// (cap 8 shows the same attractor pair), i.e. residual LSB DSP noise,
/// not window corruption. Any op-count change per layer must re-verify
/// this cap (see `decode_ops_cap`).
const DECODE_OPS_DEFAULT: usize = 12;

/// Decode ops-per-flush cap; override via `CERA_HEXAGON_DECODE_OPS` (0
/// or unset keeps `DECODE_OPS_DEFAULT`). Any raise must re-verify
/// determinism over 64+ token greedy runs (identical md5 across 6+
/// runs, 2+ prompts) — the failure mode is silent corruption.
fn decode_ops_cap() -> usize {
    std::env::var("CERA_HEXAGON_DECODE_OPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DECODE_OPS_DEFAULT)
}

#[derive(Clone, Copy, Debug)]
struct ScratchOffsets {
    activation: usize,
    normed: usize,
    q: usize,
    k: usize,
    v: usize,
    attn_out: usize,
    conv_in: usize,
    conv_bx: usize,
    conv_t0: usize,
    conv_t1: usize,
    conv_y: usize,
    conv_x: usize,
    conv_ssm_y: usize,
    ffn_gate: usize,
    ffn_up: usize,
    ffn_out: usize,
    logits: usize,
    pos: usize,
    mask: usize,
    total_size: usize,
}

impl ScratchOffsets {
    fn new(
        hidden_size: usize,
        q_dim: usize,
        kv_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Self {
        let align = |x: usize| (x + 4095) & !4095;
        let m = PREFILL_MAX_ROWS;
        let mut cur = 0;
        // Activation regions hold M prefill rows; decode uses the 1-row prefix.
        let activation = cur;
        cur = align(cur + m * hidden_size * 4);
        let normed = cur;
        cur = align(cur + m * hidden_size * 4);
        let q = cur;
        cur = align(cur + m * q_dim * 4);
        let k = cur;
        cur = align(cur + m * kv_dim * 4);
        let v = cur;
        cur = align(cur + m * kv_dim * 4);
        let attn_out = cur;
        cur = align(cur + m * q_dim * 4);
        let conv_in = cur;
        cur = align(cur + m * 3 * hidden_size * 4);
        let conv_bx = cur;
        cur = align(cur + m * hidden_size * 4);
        let conv_t0 = cur;
        cur = align(cur + m * hidden_size * 4);
        let conv_t1 = cur;
        cur = align(cur + m * hidden_size * 4);
        let conv_y = cur;
        cur = align(cur + m * hidden_size * 4);
        let conv_x = cur;
        cur = align(cur + (m + 2) * hidden_size * 4);
        let conv_ssm_y = cur;
        cur = align(cur + m * hidden_size * 4);
        let ffn_gate = cur;
        cur = align(cur + m * intermediate_size * 4);
        let ffn_up = cur;
        cur = align(cur + m * intermediate_size * 4);
        let ffn_out = cur;
        cur = align(cur + m * intermediate_size * 4);
        let logits = cur;
        cur = align(cur + vocab_size * 4);
        let pos = cur;
        cur = align(cur + m * 4);
        let mask = cur;
        cur = align(cur + m * max_seq_len * 2);
        let total_size = cur;

        Self {
            activation,
            normed,
            q,
            k,
            v,
            attn_out,
            conv_in,
            conv_bx,
            conv_t0,
            conv_t1,
            conv_y,
            conv_x,
            conv_ssm_y,
            ffn_gate,
            ffn_up,
            ffn_out,
            logits,
            pos,
            mask,
            total_size,
        }
    }
}

/// Hexagon NPU accelerated model instance for LFM2 dense hybrid transformers.
pub struct HexagonLfm2Model {
    #[allow(dead_code)]
    context: Arc<HexagonContext>,
    device: Mutex<HexagonDevice>,
    config: ModelConfig,
    session_gate: ModelSessionGate,

    // Token embedding on CPU
    token_embd: Vec<f32>,

    // Unified weights buffer in rpcmem (all static weights across all layers)
    weights_buf: RpcmemBuffer,
    layers: Vec<HexagonLayer>,
    output_norm_offset: usize,
    lm_head: HexagonWeight,

    // Unified KV cache & conv state buffer in rpcmem
    kv_state_buf: RpcmemBuffer,

    // Unified scratch buffer in rpcmem (all activations share 1 buffer to remain under HTP_MAX_MMAPS)
    scratch_buf: RpcmemBuffer,
    scratch_offsets: ScratchOffsets,

    // All-zeros f16 attention mask (one per KV slot). Decode attends the full
    // valid prefix so the mask is an identity, but the DSP kernel requires
    // src[3] to be a valid tensor; a null mask crashes the worker.
    mask_buf: RpcmemBuffer,

    _rope_type: RopeType,
    cpu_rope: bool,
    /// Debug barriers: flush the DSP queue after every op group (the
    /// bring-up behavior). Default off: the whole token submits as one
    /// batch. Set `CERA_HEXAGON_BARRIERS=1` to restore per-group flushes
    /// when localizing a DSP-side failure.
    debug_barriers: bool,
    /// ADPF CPU hint session holding host clocks across DSP-bound waits.
    /// `None` when unavailable (non-Android, API < 33) or disabled.
    adpf: Mutex<Option<AdpfSession>>,
    /// Route the short-conv recurrence through the fused SsmConv op.
    /// Default on; `CERA_HEXAGON_SSM_CONV=0` restores the manual
    /// Mul/Add/Cpy chain (decode only; prefill always uses SsmConv).
    use_ssm_conv: bool,
    /// Prefer HMX (matrix unit) matmul/attention kernels for prefill chunks
    /// with M >= 5, falling back to HVX when ineligible or when the solver
    /// finds no chunking. Default on; `CERA_HEXAGON_HMX=0` forces HVX.
    use_hmx: bool,
    vtcm_budget: usize,
    current_seq_len: AtomicUsize,
}

unsafe impl Send for HexagonLfm2Model {}
unsafe impl Sync for HexagonLfm2Model {}

impl HexagonLfm2Model {
    /// Load an LFM2 model onto the Hexagon NPU from GGUF.
    pub fn from_gguf(
        gguf: GgufFile,
        _path: Option<&Path>,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        let config = crate::model::lfm2::Lfm2Model::parse_config(&gguf, context_size)
            .map_err(|e| CeraError::Backend(e.to_string()))?;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let n_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let rope_type = RopeType::Neox;

        // Initialize FastRPC userspace driver
        let context = HexagonContext::new()?;

        // Decode runs fully on the NPU by default (Android demotes background
        // process CPUs, so NPU-only decode performs better overall). Set
        // CERA_HEXAGON_CPU_ROPE=1 to apply RoPE on the host CPU instead.
        let cpu_rope = std::env::var("CERA_HEXAGON_CPU_ROPE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let debug_barriers = std::env::var("CERA_HEXAGON_BARRIERS")
            .map(|v| v == "1")
            .unwrap_or(false);
        let adpf_target_nanos: i64 = std::env::var("CERA_HEXAGON_ADPF_TARGET_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|ms| (ms.saturating_mul(1_000_000)).min(i64::MAX as u64) as i64)
            .unwrap_or(10_000_000);
        let adpf = Mutex::new(AdpfSession::try_open(adpf_target_nanos));
        let use_ssm_conv = std::env::var("CERA_HEXAGON_SSM_CONV")
            .map(|v| v != "0")
            .unwrap_or(true);
        let use_hmx = std::env::var("CERA_HEXAGON_HMX")
            .map(|v| v != "0")
            .unwrap_or(true);

        let arch_override = std::env::var("CERA_HEXAGON_ARCH")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .and_then(HexagonArch::from_u32);

        // Single source of truth with `probe()`: one order decides which DSP
        // wins on multi-arch devices.
        let probe_archs: Vec<HexagonArch> = if let Some(arch) = arch_override {
            vec![arch]
        } else {
            crate::backend::hexagon::PROBE_ARCHS.to_vec()
        };

        let mut device_opt = None;
        let mut probed_errors = Vec::new();
        for arch in probe_archs {
            match HexagonDevice::new(Arc::clone(context.driver()), arch) {
                Ok(dev) => {
                    tracing::info!(arch = ?arch, "initialized Hexagon NPU device");
                    device_opt = Some(dev);
                    break;
                }
                Err(e) => {
                    probed_errors.push(format!("{arch:?}: {e}"));
                }
            }
        }
        let device = match device_opt {
            Some(d) => d,
            None => {
                return Err(CeraError::Backend(format!(
                    "no compatible Hexagon skeleton library found. Errors: {}",
                    probed_errors.join("; ")
                )));
            }
        };

        let token_embd = gguf
            .get_tensor("token_embd.weight")
            .map_err(|e| CeraError::Backend(format!("missing token_embd.weight: {e}")))?
            .to_f32_vec();

        let driver = context.driver();

        let q_dim = n_heads * head_dim;
        let max_kv_dim = config
            .kv_heads_per_layer
            .iter()
            .map(|&h| h * head_dim)
            .max()
            .unwrap_or(head_dim);

        // Allocate unified shared scratch buffer
        let scratch_offsets = ScratchOffsets::new(
            hidden_size,
            q_dim,
            max_kv_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
        );
        let scratch_buf =
            RpcmemBuffer::alloc(Arc::clone(driver), scratch_offsets.total_size, true)?;

        // Flash attention mask: all zeros (decode identity mask), sized for
        // the full KV cache. Shared by every layer.
        let mask_size = (max_seq_len * 2).max(128);
        let mask_buf = RpcmemBuffer::alloc(Arc::clone(driver), mask_size, true)?;
        unsafe {
            std::ptr::write_bytes(mask_buf.as_mut_ptr(), 0, mask_size);
        }
        mask_buf.flush_cpu_cache(0, mask_size);

        // Determine layer norms
        let mut has_attn_q_norm = Vec::with_capacity(n_layers);
        let mut has_attn_k_norm = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let is_attn = config.block_types[i] == BlockType::Attention;
            if is_attn {
                has_attn_q_norm.push(
                    gguf.tensors
                        .contains_key(&format!("blk.{i}.attn_q_norm.weight")),
                );
                has_attn_k_norm.push(
                    gguf.tensors
                        .contains_key(&format!("blk.{i}.attn_k_norm.weight")),
                );
            } else {
                has_attn_q_norm.push(false);
                has_attn_k_norm.push(false);
            }
        }

        let align256 = |x: usize| (x + 255) & !255;

        // Pass 1: Plan offsets for all weights in weights_buf
        let mut weights_total = 0;

        // Returns (repacked byte size, wire dtype, block bytes, tile size).
        let tensor_weight_plan =
            |name: &str| -> Result<(usize, HtpDataType, usize, usize), CeraError> {
                let t = gguf
                    .tensors
                    .get(name)
                    .ok_or_else(|| CeraError::Backend(format!("missing tensor {name}")))?;
                let ne0 = t.shape[0];
                let ne1 = if t.shape.len() > 1 { t.shape[1] } else { 1 };
                match t.dtype {
                    crate::tensor::DType::Q8_0 => Ok((
                        repacked_matrix_size_q8_0(ne0, ne1)?,
                        HtpDataType::Q8_0,
                        34,
                        TILE_SIZE_Q8_0,
                    )),
                    crate::tensor::DType::Q4_0 => Ok((
                        repacked_matrix_size_q4_0(ne0, ne1)?,
                        HtpDataType::Q4_0,
                        18,
                        TILE_SIZE_Q4_0,
                    )),
                    crate::tensor::DType::Q4KM => Ok((
                        repacked_matrix_size_q4_k(ne0, ne1)?,
                        HtpDataType::Q4K,
                        144,
                        TILE_SIZE_Q4_K,
                    )),
                    crate::tensor::DType::Q6K => Ok((
                        repacked_matrix_size_q6_k(ne0, ne1)?,
                        HtpDataType::Q6K,
                        210,
                        TILE_SIZE_Q6_K,
                    )),
                    // No Q5_K wire format: plan Q8_0 bytes; `copy_weight`
                    // requants before repack.
                    crate::tensor::DType::Q5KM => Ok((
                        repacked_matrix_size_q8_0(ne0, ne1)?,
                        HtpDataType::Q8_0,
                        34,
                        TILE_SIZE_Q8_0,
                    )),
                    other => Err(CeraError::Backend(format!(
                        "unsupported quant format {other:?} for Hexagon weight {name}"
                    ))),
                }
            };
        // Plans one weight matrix into a HexagonWeight (offset, dims, tiled
        // wire format). Takes the running total explicitly so each weight
        // needs a single call site.
        let plan_hex_weight = |name: &str,
                               total: &mut usize,
                               in_dim: usize,
                               out_dim: usize|
         -> Result<HexagonWeight, CeraError> {
            let (size, wire_dtype, block_bytes, tile_size) = tensor_weight_plan(name)?;
            let offset = plan_offset(total, size);
            Ok(HexagonWeight {
                offset,
                in_dim,
                out_dim,
                wire_dtype,
                block_bytes,
                tile_size,
            })
        };

        struct PlannedAttention {
            attn_norm_offset: usize,
            attn_q: HexagonWeight,
            attn_k: HexagonWeight,
            attn_v: HexagonWeight,
            attn_output: HexagonWeight,
            attn_q_norm_offset: Option<usize>,
            attn_k_norm_offset: Option<usize>,
            ffn_norm_offset: usize,
            ffn_gate: HexagonWeight,
            ffn_up: HexagonWeight,
            ffn_down: HexagonWeight,
            kv_dim: usize,
        }

        struct PlannedConv {
            attn_norm_offset: usize,
            in_proj: HexagonWeight,
            out_proj: HexagonWeight,
            conv_w0_offset: usize,
            conv_w1_offset: usize,
            conv_w2_offset: usize,
            conv_ssm_offset: usize,
            ffn_norm_offset: usize,
            ffn_gate: HexagonWeight,
            ffn_up: HexagonWeight,
            ffn_down: HexagonWeight,
        }

        enum PlannedLayer {
            Attention(PlannedAttention),
            Conv(PlannedConv),
        }

        let mut planned_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn_norm_offset = plan_offset(&mut weights_total, hidden_size * 4);
            if config.block_types[i] == BlockType::Attention {
                let n_kv = config.kv_heads_per_layer[i];
                let kv_dim = n_kv * head_dim;
                let attn_q = plan_hex_weight(
                    &format!("blk.{i}.attn_q.weight"),
                    &mut weights_total,
                    hidden_size,
                    q_dim,
                )?;
                let attn_k = plan_hex_weight(
                    &format!("blk.{i}.attn_k.weight"),
                    &mut weights_total,
                    hidden_size,
                    kv_dim,
                )?;
                let attn_v = plan_hex_weight(
                    &format!("blk.{i}.attn_v.weight"),
                    &mut weights_total,
                    hidden_size,
                    kv_dim,
                )?;
                let attn_output = plan_hex_weight(
                    &format!("blk.{i}.attn_output.weight"),
                    &mut weights_total,
                    q_dim,
                    hidden_size,
                )?;
                let attn_q_norm_offset = if has_attn_q_norm[i] {
                    Some(plan_offset(&mut weights_total, q_dim * 4))
                } else {
                    None
                };
                let attn_k_norm_offset = if has_attn_k_norm[i] {
                    Some(plan_offset(&mut weights_total, kv_dim * 4))
                } else {
                    None
                };
                let ffn_norm_offset = plan_offset(&mut weights_total, hidden_size * 4);
                let ffn_gate = plan_hex_weight(
                    &format!("blk.{i}.ffn_gate.weight"),
                    &mut weights_total,
                    hidden_size,
                    intermediate_size,
                )?;
                let ffn_up = plan_hex_weight(
                    &format!("blk.{i}.ffn_up.weight"),
                    &mut weights_total,
                    hidden_size,
                    intermediate_size,
                )?;
                let ffn_down = plan_hex_weight(
                    &format!("blk.{i}.ffn_down.weight"),
                    &mut weights_total,
                    intermediate_size,
                    hidden_size,
                )?;

                planned_layers.push(PlannedLayer::Attention(PlannedAttention {
                    attn_norm_offset,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm_offset,
                    attn_k_norm_offset,
                    ffn_norm_offset,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    kv_dim,
                }));
            } else {
                let in_proj = plan_hex_weight(
                    &format!("blk.{i}.shortconv.in_proj.weight"),
                    &mut weights_total,
                    hidden_size,
                    3 * hidden_size,
                )?;
                let out_proj = plan_hex_weight(
                    &format!("blk.{i}.shortconv.out_proj.weight"),
                    &mut weights_total,
                    hidden_size,
                    hidden_size,
                )?;
                let conv_w0_offset = plan_offset(&mut weights_total, hidden_size * 4);
                let conv_w1_offset = plan_offset(&mut weights_total, hidden_size * 4);
                let conv_w2_offset = plan_offset(&mut weights_total, hidden_size * 4);
                let conv_ssm_offset = plan_offset(&mut weights_total, 3 * hidden_size * 4);
                let ffn_norm_offset = plan_offset(&mut weights_total, hidden_size * 4);
                let ffn_gate = plan_hex_weight(
                    &format!("blk.{i}.ffn_gate.weight"),
                    &mut weights_total,
                    hidden_size,
                    intermediate_size,
                )?;
                let ffn_up = plan_hex_weight(
                    &format!("blk.{i}.ffn_up.weight"),
                    &mut weights_total,
                    hidden_size,
                    intermediate_size,
                )?;
                let ffn_down = plan_hex_weight(
                    &format!("blk.{i}.ffn_down.weight"),
                    &mut weights_total,
                    intermediate_size,
                    hidden_size,
                )?;

                planned_layers.push(PlannedLayer::Conv(PlannedConv {
                    attn_norm_offset,
                    in_proj,
                    out_proj,
                    conv_w0_offset,
                    conv_w1_offset,
                    conv_w2_offset,
                    conv_ssm_offset,
                    ffn_norm_offset,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                }));
            }
        }

        let output_norm_offset = plan_offset(&mut weights_total, hidden_size * 4);
        let lm_head_name = if gguf.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head = plan_hex_weight(lm_head_name, &mut weights_total, hidden_size, vocab_size)?;

        // Pass 2: Plan offsets for KV caches and conv states in kv_state_buf
        let state_size = align256(hidden_size * 4);
        let mut kv_state_total = 0;
        let mut layers = Vec::with_capacity(n_layers);

        for (i, planned) in planned_layers.into_iter().enumerate() {
            match planned {
                PlannedLayer::Attention(pa) => {
                    let n_kv = config.kv_heads_per_layer[i];
                    let kv_slab_size = align256(max_seq_len * n_kv * head_dim * 2);
                    let k_offset = align256(kv_state_total);
                    let v_offset = align256(k_offset + kv_slab_size);
                    kv_state_total = v_offset + kv_slab_size;

                    layers.push(HexagonLayer::Attention(HexagonAttentionLayer {
                        attn_norm_offset: pa.attn_norm_offset,
                        attn_q: pa.attn_q,
                        attn_k: pa.attn_k,
                        attn_v: pa.attn_v,
                        attn_output: pa.attn_output,
                        attn_q_norm_offset: pa.attn_q_norm_offset,
                        attn_k_norm_offset: pa.attn_k_norm_offset,
                        ffn_norm_offset: pa.ffn_norm_offset,
                        ffn_gate: pa.ffn_gate,
                        ffn_up: pa.ffn_up,
                        ffn_down: pa.ffn_down,
                        k_offset,
                        v_offset,
                        q_dim,
                        kv_dim: pa.kv_dim,
                    }));
                }
                PlannedLayer::Conv(pc) => {
                    // One channel-interleaved `[C, 2]` slab (see
                    // `HexagonConvLayer::state_offset`).
                    let state_offset = align256(kv_state_total);
                    kv_state_total = align256(state_offset + 2 * state_size);

                    layers.push(HexagonLayer::Conv(HexagonConvLayer {
                        attn_norm_offset: pc.attn_norm_offset,
                        in_proj: pc.in_proj,
                        out_proj: pc.out_proj,
                        conv_w0_offset: pc.conv_w0_offset,
                        conv_w1_offset: pc.conv_w1_offset,
                        conv_w2_offset: pc.conv_w2_offset,
                        conv_ssm_offset: pc.conv_ssm_offset,
                        state_offset,
                        ffn_norm_offset: pc.ffn_norm_offset,
                        ffn_gate: pc.ffn_gate,
                        ffn_up: pc.ffn_up,
                        ffn_down: pc.ffn_down,
                    }));
                }
            }
        }

        // Allocate unified weights and KV state buffers
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(driver), weights_total, true)?;
        let kv_state_buf = RpcmemBuffer::alloc(Arc::clone(driver), kv_state_total, true)?;
        unsafe {
            std::ptr::write_bytes(kv_state_buf.as_mut_ptr(), 0, kv_state_total);
        }
        kv_state_buf.flush_cpu_cache(0, kv_state_total);

        // Helper to copy F32 norm weights directly into weights_buf
        let copy_norm =
            |name: &str, offset: usize, buf: &mut RpcmemBuffer| -> Result<(), CeraError> {
                let t = gguf
                    .get_tensor(name)
                    .map_err(|e| CeraError::Backend(format!("missing tensor {name}: {e}")))?;
                let f32_vals = t.to_f32_vec();
                let byte_size = f32_vals.len() * std::mem::size_of::<f32>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        f32_vals.as_ptr() as *const u8,
                        buf.as_mut_ptr().add(offset),
                        byte_size,
                    );
                }
                Ok(())
            };

        // Helper to repack 2D weight matrix directly into weights_buf
        let copy_weight =
            |name: &str, offset: usize, buf: &mut RpcmemBuffer| -> Result<(), CeraError> {
                let t = gguf
                    .tensors
                    .get(name)
                    .ok_or_else(|| CeraError::Backend(format!("missing tensor {name}")))?;
                let ne0 = t.shape[0];
                let ne1 = if t.shape.len() > 1 { t.shape[1] } else { 1 };
                let raw_data = gguf
                    .tensor_data(name)
                    .map_err(|e| CeraError::Backend(e.to_string()))?;
                let dst_slice = &mut buf.as_mut_slice()[offset..];
                match t.dtype {
                    crate::tensor::DType::Q8_0 => {
                        repack_q8_0(raw_data, ne0, ne1, dst_slice)
                            .map_err(|e| CeraError::Backend(format!("{name}: {e}")))?;
                    }
                    crate::tensor::DType::Q4_0 => {
                        repack_q4_0(raw_data, ne0, ne1, dst_slice)
                            .map_err(|e| CeraError::Backend(format!("{name}: {e}")))?;
                    }
                    crate::tensor::DType::Q4KM => {
                        repack_q4_k(raw_data, ne0, ne1, dst_slice)
                            .map_err(|e| CeraError::Backend(format!("{name}: {e}")))?;
                    }
                    crate::tensor::DType::Q6K => {
                        repack_q6_k(raw_data, ne0, ne1, dst_slice)
                            .map_err(|e| CeraError::Backend(format!("{name}: {e}")))?;
                    }
                    crate::tensor::DType::Q5KM => {
                        let q8 = requant_q5_k_to_q8_0(raw_data, ne0, ne1)
                            .map_err(|e| CeraError::Backend(format!("{name}: {e}")))?;
                        repack_q8_0(&q8, ne0, ne1, dst_slice)
                            .map_err(|e| CeraError::Backend(format!("{name}: {e}")))?;
                    }
                    other => {
                        return Err(CeraError::Backend(format!(
                            "unsupported quant format {other:?} for Hexagon weight {name}"
                        )));
                    }
                }
                Ok(())
            };

        for (i, layer) in layers.iter().enumerate() {
            match layer {
                HexagonLayer::Attention(attn) => {
                    copy_norm(
                        &format!("blk.{i}.attn_norm.weight"),
                        attn.attn_norm_offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.attn_q.weight"),
                        attn.attn_q.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.attn_k.weight"),
                        attn.attn_k.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.attn_v.weight"),
                        attn.attn_v.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.attn_output.weight"),
                        attn.attn_output.offset,
                        &mut weights_buf,
                    )?;
                    if let Some(qn_offset) = attn.attn_q_norm_offset {
                        copy_norm(
                            &format!("blk.{i}.attn_q_norm.weight"),
                            qn_offset,
                            &mut weights_buf,
                        )?;
                    }
                    if let Some(kn_offset) = attn.attn_k_norm_offset {
                        copy_norm(
                            &format!("blk.{i}.attn_k_norm.weight"),
                            kn_offset,
                            &mut weights_buf,
                        )?;
                    }
                    copy_norm(
                        &format!("blk.{i}.ffn_norm.weight"),
                        attn.ffn_norm_offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.ffn_gate.weight"),
                        attn.ffn_gate.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.ffn_up.weight"),
                        attn.ffn_up.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.ffn_down.weight"),
                        attn.ffn_down.offset,
                        &mut weights_buf,
                    )?;
                }
                HexagonLayer::Conv(conv) => {
                    copy_norm(
                        &format!("blk.{i}.attn_norm.weight"),
                        conv.attn_norm_offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.shortconv.in_proj.weight"),
                        conv.in_proj.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.shortconv.out_proj.weight"),
                        conv.out_proj.offset,
                        &mut weights_buf,
                    )?;

                    let conv_tensor = gguf
                        .get_tensor(&format!("blk.{i}.shortconv.conv.weight"))
                        .map_err(|e| {
                            CeraError::Backend(format!("missing shortconv.conv.weight: {e}"))
                        })?;
                    let conv_f32 = conv_tensor.to_f32_vec();
                    let mut w0_vec = vec![0.0f32; hidden_size];
                    let mut w1_vec = vec![0.0f32; hidden_size];
                    let mut w2_vec = vec![0.0f32; hidden_size];
                    for c in 0..hidden_size {
                        w0_vec[c] = conv_f32[c * 3];
                        w1_vec[c] = conv_f32[c * 3 + 1];
                        w2_vec[c] = conv_f32[c * 3 + 2];
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            w0_vec.as_ptr() as *const u8,
                            weights_buf.as_mut_ptr().add(conv.conv_w0_offset),
                            hidden_size * 4,
                        );
                        std::ptr::copy_nonoverlapping(
                            w1_vec.as_ptr() as *const u8,
                            weights_buf.as_mut_ptr().add(conv.conv_w1_offset),
                            hidden_size * 4,
                        );
                        std::ptr::copy_nonoverlapping(
                            w2_vec.as_ptr() as *const u8,
                            weights_buf.as_mut_ptr().add(conv.conv_w2_offset),
                            hidden_size * 4,
                        );
                        // SsmConv taps: the GGUF [3, C] layout is already
                        // oldest-first per channel, usable verbatim.
                        std::ptr::copy_nonoverlapping(
                            conv_f32.as_ptr() as *const u8,
                            weights_buf.as_mut_ptr().add(conv.conv_ssm_offset),
                            conv_f32.len() * 4,
                        );
                    }

                    copy_norm(
                        &format!("blk.{i}.ffn_norm.weight"),
                        conv.ffn_norm_offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.ffn_gate.weight"),
                        conv.ffn_gate.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.ffn_up.weight"),
                        conv.ffn_up.offset,
                        &mut weights_buf,
                    )?;
                    copy_weight(
                        &format!("blk.{i}.ffn_down.weight"),
                        conv.ffn_down.offset,
                        &mut weights_buf,
                    )?;
                }
            }
        }

        let output_norm_name = if gguf.tensors.contains_key("output_norm.weight") {
            "output_norm.weight"
        } else if gguf.tensors.contains_key("token_embd_norm.weight") {
            "token_embd_norm.weight"
        } else {
            return Err(CeraError::Backend("missing output_norm tensor".into()));
        };
        copy_norm(output_norm_name, output_norm_offset, &mut weights_buf)?;

        let lm_head_name = if gguf.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        copy_weight(lm_head_name, lm_head.offset, &mut weights_buf)?;

        weights_buf.flush_cpu_cache(0, weights_total);

        let vtcm_budget = device.hw_info().vtcm_size as usize;

        Ok(Self {
            context,
            device: Mutex::new(device),
            config,
            session_gate: ModelSessionGate::default(),
            token_embd,
            weights_buf,
            layers,
            output_norm_offset,
            lm_head,
            kv_state_buf,
            scratch_buf,
            scratch_offsets,
            mask_buf,
            _rope_type: rope_type,
            cpu_rope,
            debug_barriers,
            adpf,
            use_ssm_conv,
            use_hmx,
            vtcm_budget,
            current_seq_len: AtomicUsize::new(0),
        })
    }
}

impl HexagonLfm2Model {
    /// Contiguous f32 vector descriptor (`[dim,1,1,1]`): the one spelling of
    /// the vec shape+strides all elementwise dispatches share.
    fn add_f32_vec(
        session: &mut HexagonQueueSession,
        buf: &RpcmemBuffer,
        offset: usize,
        dim: usize,
        flags: u32,
    ) -> Result<u16, CeraError> {
        session.add_tensor(
            buf,
            offset,
            dim * 4,
            flags,
            HtpDataType::F32 as u32,
            [dim as u32, 1, 1, 1],
            [4, (dim * 4) as u32, (dim * 4) as u32, (dim * 4) as u32],
        )
    }

    /// Enqueue with op-name context: the one spelling of the
    /// `enqueue_op` + `dispatch_*:` label every dispatch shares.
    fn enqueue_labeled(
        session: &mut HexagonQueueSession,
        label: &str,
        opcode: u32,
        src: &[u16],
        dst: &[u16],
        params: [i32; 16],
        kernel_params: [i32; 32],
    ) -> Result<(), CeraError> {
        session
            .enqueue_op(opcode, src, dst, params, kernel_params)
            .map_err(|e| CeraError::Backend(format!("{label}: {e}")))
    }

    fn dispatch_mul(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src0_flags: u32,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        src1_flags: u32,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, src0_flags)?;
        let src1_ti = Self::add_f32_vec(session, src1, src1_offset, dim, src1_flags)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
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
        Self::enqueue_labeled(
            session,
            "dispatch_mul",
            HtpOpCode::Mul as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Row-wise multiply with strided inputs (llama's strided-view MUL):
    /// `dst[r, c] = a[r, c] * b[r, c]` over `[dim, n_rows]`, reading A/B
    /// with byte row strides `a_row_stride`/`b_row_stride`. The DSP reads the
    /// strides from the tensor descriptors. Used for the conv `b * x` and
    /// gate products straight out of the strided `in_proj` thirds.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_mul_m_strided(
        session: &mut HexagonQueueSession,
        a_buf: &RpcmemBuffer,
        a_offset: usize,
        a_flags: u32,
        b_buf: &RpcmemBuffer,
        b_offset: usize,
        b_flags: u32,
        dst_buf: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        n_rows: usize,
        a_row_stride: usize,
        b_row_stride: usize,
    ) -> Result<(), CeraError> {
        let row = dim * 4;
        let span = |stride: usize| n_rows.saturating_sub(1) * stride + row;
        let a_span = span(a_row_stride);
        let b_span = span(b_row_stride);
        let dst_bytes = row * n_rows;
        let ne = [dim as u32, n_rows as u32, 1, 1];
        let a_ti = session.add_tensor(
            a_buf,
            a_offset,
            a_span,
            a_flags,
            HtpDataType::F32 as u32,
            ne,
            [4, a_row_stride as u32, a_span as u32, a_span as u32],
        )?;
        let b_ti = session.add_tensor(
            b_buf,
            b_offset,
            b_span,
            b_flags,
            HtpDataType::F32 as u32,
            ne,
            [4, b_row_stride as u32, b_span as u32, b_span as u32],
        )?;
        let dst_ti = session.add_tensor(
            dst_buf,
            dst_offset,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            [4, row as u32, dst_bytes as u32, dst_bytes as u32],
        )?;
        let params = [0i32; 16];
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            n_rows,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_mul_m_strided",
            HtpOpCode::Mul as u32,
            &[a_ti, b_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Dim-0 CONCAT of two 2D f32 tensors:
    /// `[s0_rows, dim] + [s1_rows, dim] -> [s0_rows + s1_rows, dim]`.
    /// `params[0]` is the concat dim; kparams are zero (the DSP sizes VTCM
    /// itself). The second source may be a transposed view (`s1_nb0 >
    /// s1_nb1`), which takes the DSP's specialized 2D-transposed worker —
    /// the conv state prepend (`[s0; s1] + bx-as-[m, hs]`).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_concat_2d(
        session: &mut HexagonQueueSession,
        s0_buf: &RpcmemBuffer,
        s0_offset: usize,
        s0_rows: usize,
        s1_buf: &RpcmemBuffer,
        s1_offset: usize,
        s1_rows: usize,
        s1_nb0: usize,
        s1_nb1: usize,
        dst_buf: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let s0_span = s0_rows * dim * 4;
        let s1_span = s1_rows.saturating_sub(1) * s1_nb1 + dim.saturating_sub(1) * s1_nb0 + 4;
        let dst_rows = s0_rows + s1_rows;
        let dst_span = dst_rows * dim * 4;
        let s0_ti = session.add_tensor(
            s0_buf,
            s0_offset,
            s0_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [s0_rows as u32, dim as u32, 1, 1],
            [4, (s0_rows * 4) as u32, s0_span as u32, s0_span as u32],
        )?;
        let s1_ti = session.add_tensor(
            s1_buf,
            s1_offset,
            s1_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [s1_rows as u32, dim as u32, 1, 1],
            [s1_nb0 as u32, s1_nb1 as u32, s1_span as u32, s1_span as u32],
        )?;
        let dst_ti = session.add_tensor(
            dst_buf,
            dst_offset,
            dst_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [dst_rows as u32, dim as u32, 1, 1],
            [4, (dst_rows * 4) as u32, dst_span as u32, dst_span as u32],
        )?;
        let mut params = [0i32; 16];
        params[0] = 0; // concat dim
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_concat_2d",
            HtpOpCode::Concat as u32,
            &[s0_ti, s1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

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
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, HTP_TENSOR_COMPUTE)?;
        let src1_ti = Self::add_f32_vec(session, src1, src1_offset, dim, HTP_TENSOR_COMPUTE)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
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
        Self::enqueue_labeled(
            session,
            "dispatch_add",
            HtpOpCode::Add as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-row (prefill) Add: `dst[m, :] = a[m, :] + b[m, :]` over `n_rows`
    /// contiguous rows of `dim` f32s.
    fn dispatch_add_m(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let bytes = dim * n_rows * 4;
        let ne = [dim as u32, n_rows as u32, 1, 1];
        let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];
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
            n_rows,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_add_m",
            HtpOpCode::Add as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_mul_mat(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        w: &HexagonWeight,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
    ) -> Result<(), CeraError> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        // Tiled wire format: dims padded to 32, row stride = K tiles wide.
        let ne0 = in_dim.div_ceil(32) * 32;
        let ne1 = out_dim.div_ceil(32) * 32;
        let tiled_row_bytes = (ne0 / 32) * w.tile_size;
        let w_ti = session.add_tensor(
            weights,
            w.offset,
            (ne1 / 32) * tiled_row_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w.wire_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                w.block_bytes as u32,
                tiled_row_bytes as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
            ],
        )?;
        let in_ti = session.add_tensor(
            in_act,
            in_offset,
            in_dim * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [in_dim as u32, 1, 1, 1],
            [
                4,
                (in_dim * 4) as u32,
                (in_dim * 4) as u32,
                (in_dim * 4) as u32,
            ],
        )?;
        let out_ti = session.add_tensor(
            out_act,
            out_offset,
            out_dim * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [out_dim as u32, 1, 1, 1],
            [
                4,
                (out_dim * 4) as u32,
                (out_dim * 4) as u32,
                (out_dim * 4) as u32,
            ],
        )?;
        let wtype = w.wire_dtype;
        let params = [0i32; 16];
        let kparams = build_mul_mat_kernel_params(
            wtype,
            in_dim,
            1,
            1,
            out_dim * 4,
            session.dsp_threads(),
            self.vtcm_budget,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat",
            HtpOpCode::MulMat as u32,
            &[w_ti, in_ti],
            &[out_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-row (prefill) GEMM: `out[m, :] = in[m, :] @ W` over `n_rows`
    /// contiguous activation rows (HVX path; weights stream from DDR once
    /// per op, so chunk rows to fit VTCM).
    fn dispatch_mul_mat_m(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        w: &HexagonWeight,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        // Tiled wire format: dims padded to 32, row stride = K tiles wide.
        let ne0 = in_dim.div_ceil(32) * 32;
        let ne1 = out_dim.div_ceil(32) * 32;
        let wtype = w.wire_dtype;
        // HMX first (M >= 5 prefill), HVX fallback — mirrors ggml's
        // HMX-then-HVX selection. The same repacked weights feed both, but
        // the dim-1 stride differs: HVX walks N rows of K tiles while HMX
        // addresses N-tile starts as `nc * nb[1]`, so HMX needs the
        // tiled row size (`ggml_hexagon_tiled_row_size`).
        let hmx_kparams = if self.use_hmx && mm_is_hmx_eligible(wtype, ne0, ne1, n_rows) {
            build_hmx_mm_kernel_params(
                wtype,
                ne0,
                ne1,
                n_rows.next_multiple_of(32),
                n_rows,
                session.dsp_threads(),
                self.vtcm_budget,
            )
        } else {
            None
        };
        let tiled_row_bytes = (ne0 / 32) * w.tile_size;
        let w_nb1 = if hmx_kparams.is_some() {
            mm_hmx_nb1(wtype, ne0)
        } else {
            tiled_row_bytes
        };
        let w_ti = session.add_tensor(
            weights,
            w.offset,
            (ne1 / 32) * tiled_row_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w.wire_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                w.block_bytes as u32,
                w_nb1 as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
            ],
        )?;
        let in_ti = session.add_tensor(
            in_act,
            in_offset,
            in_dim * n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [in_dim as u32, n_rows as u32, 1, 1],
            [
                4,
                (in_dim * 4) as u32,
                (in_dim * n_rows * 4) as u32,
                (in_dim * n_rows * 4) as u32,
            ],
        )?;
        let out_ti = session.add_tensor(
            out_act,
            out_offset,
            out_dim * n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [out_dim as u32, n_rows as u32, 1, 1],
            [
                4,
                (out_dim * 4) as u32,
                (out_dim * n_rows * 4) as u32,
                (out_dim * n_rows * 4) as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = hmx_kparams.unwrap_or_else(|| {
            build_mul_mat_kernel_params(
                wtype,
                in_dim,
                n_rows as u32,
                1,
                out_dim * 4,
                session.dsp_threads(),
                self.vtcm_budget,
            )
        });
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat_m",
            HtpOpCode::MulMat as u32,
            &[w_ti, in_ti],
            &[out_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Fused multi-projection GEMM (MUL_MAT_NX): `dst[i][m, :] = in[m, :] @
    /// W[i]` for N weights sharing one activation. The DSP quantizes the
    /// shared activation once instead of N times (llama's QKV and gate/up
    /// fusion). Sources are weights-first, activation last (`[w0..wN, x]`);
    /// N weight inputs plus one activation fit the 10-source / 4-dst op
    /// descriptor for N <= 4.
    ///
    /// Falls back to N single dispatches when fusion is unsupported: N
    /// outside 2..=4, mixed K (`in_dim`) or wire dtype, HMX-eligibility
    /// mismatch across the set, Q6_K on the HVX path (no fused HVX
    /// kernel), or HMX chunking overflow (which retries HVX first, like
    /// the single path). Kernel parameters are W0's single-matmul
    /// parameters with `n_weights` set, so NX fits VTCM exactly when the
    /// W0 single would; W0 must carry the largest N (`out_dim`) so the
    /// m=1 dst scratch covers every output.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_mul_mat_nx(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        ws: &[&HexagonWeight],
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offsets: &[usize],
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let unfused = |this: &Self, session: &mut HexagonQueueSession| -> Result<(), CeraError> {
            for (w, &off) in ws.iter().zip(out_offsets.iter()) {
                this.dispatch_mul_mat_m(
                    session, weights, w, in_act, in_offset, out_act, off, n_rows,
                )?;
            }
            Ok(())
        };
        let n = ws.len();
        let Some(w0) = ws.first() else {
            return Ok(());
        };
        let pad32 = |d: usize| d.div_ceil(32) * 32;
        let wtype = w0.wire_dtype;
        let k = w0.in_dim;
        let hmx0 = self.use_hmx && mm_is_hmx_eligible(wtype, pad32(k), pad32(w0.out_dim), n_rows);
        let fusable = (2..=4).contains(&n)
            && out_offsets.len() == n
            && ws.iter().all(|w| w.in_dim == k && w.wire_dtype == wtype)
            && ws.iter().all(|w| {
                (self.use_hmx && mm_is_hmx_eligible(wtype, pad32(k), pad32(w.out_dim), n_rows))
                    == hmx0
            })
            && (hmx0 || wtype != HtpDataType::Q6K);
        if !fusable {
            if std::env::var_os("CERA_HEXAGON_DEBUG").is_some() {
                eprintln!("[cera-hexagon] NX fallback: {n} unfused matmuls");
            }
            unfused(self, session)?;
            return Ok(());
        }
        debug_assert!(ws.iter().all(|w| w.out_dim <= w0.out_dim));
        // HMX first, HVX fallback — the same selection as singles, with
        // `n_weights` set. HVX forces QUANT_ROW: NX has no block kernel.
        let hmx_built = if hmx0 {
            build_hmx_mm_kernel_params(
                wtype,
                pad32(k),
                pad32(w0.out_dim),
                n_rows.next_multiple_of(32),
                n_rows,
                session.dsp_threads(),
                self.vtcm_budget,
            )
        } else {
            None
        };
        let kparams = if let Some(mut kp) = hmx_built {
            kp[17] = n as i32; // n_weights
            kp
        } else {
            // HVX path (ineligible or HMX chunking overflow): Q6_K has no
            // fused HVX kernel either way.
            if wtype == HtpDataType::Q6K {
                unfused(self, session)?;
                return Ok(());
            }
            let mut kp = build_mul_mat_kernel_params(
                wtype,
                k,
                n_rows as u32,
                1,
                w0.out_dim * 4,
                session.dsp_threads(),
                self.vtcm_budget,
            );
            kp[0] = 5; // HTP_MM_KERNEL_HVX_QUANT_ROW
            kp[17] = n as i32; // n_weights
            kp
        };
        let hmx_path = kparams[6] == 1; // n_hmx
        let tiled_row_bytes = (pad32(k) / 32) * w0.tile_size;
        let w_nb1 = if hmx_path {
            mm_hmx_nb1(wtype, pad32(k))
        } else {
            tiled_row_bytes
        };
        let mut srcs = Vec::with_capacity(n + 1);
        let mut dsts = Vec::with_capacity(n);
        for w in ws {
            let ne1 = pad32(w.out_dim);
            srcs.push(session.add_tensor(
                weights,
                w.offset,
                (ne1 / 32) * tiled_row_bytes,
                HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                w.wire_dtype as u32,
                [pad32(k) as u32, ne1 as u32, 1, 1],
                [
                    w.block_bytes as u32,
                    w_nb1 as u32,
                    ((ne1 / 32) * tiled_row_bytes) as u32,
                    ((ne1 / 32) * tiled_row_bytes) as u32,
                ],
            )?);
        }
        srcs.push(session.add_tensor(
            in_act,
            in_offset,
            k * n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [k as u32, n_rows as u32, 1, 1],
            [
                4,
                (k * 4) as u32,
                (k * n_rows * 4) as u32,
                (k * n_rows * 4) as u32,
            ],
        )?);
        for (w, &off) in ws.iter().zip(out_offsets.iter()) {
            dsts.push(session.add_tensor(
                out_act,
                off,
                w.out_dim * n_rows * 4,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [w.out_dim as u32, n_rows as u32, 1, 1],
                [
                    4,
                    (w.out_dim * 4) as u32,
                    (w.out_dim * n_rows * 4) as u32,
                    (w.out_dim * n_rows * 4) as u32,
                ],
            )?);
        }
        let params = [0i32; 16];
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat_nx",
            HtpOpCode::MulMatNx as u32,
            &srcs,
            &dsts,
            params,
            kparams,
        )?;
        Ok(())
    }

    /// SwiGLU over `n_rows` contiguous rows of `row_dim` f32s. Rows must
    /// stay split (never flatten to `[row_dim * n_rows, 1]`): the firmware
    /// sizes per-thread VTCM by dim-0, and one giant row overflows the
    /// reservation (the op then fails silent, leaving dst zeros).
    fn dispatch_swiglu(
        session: &mut HexagonQueueSession,
        gate: &RpcmemBuffer,
        gate_offset: usize,
        up: &RpcmemBuffer,
        up_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        row_dim: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let bytes = row_dim * n_rows * 4;
        let ne = [row_dim as u32, n_rows as u32, 1, 1];
        let nb = [4, (row_dim * 4) as u32, bytes as u32, bytes as u32];
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
        // No host precompute: the DSP sizes threads/VTCM itself (llama
        // passes zero kparams).
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_swiglu",
            HtpOpCode::GluSwiglu as u32,
            &[gate_ti, up_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_rope(
        session: &mut HexagonQueueSession,
        act: &RpcmemBuffer,
        act_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        head_dim: usize,
        n_heads: usize,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> Result<(), CeraError> {
        let total_bytes = head_dim * n_heads * 4;
        let act_ti = session.add_tensor(
            act,
            act_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                total_bytes as u32,
                total_bytes as u32,
            ],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [1, 1, 1, 1],
            [4, 4, 4, 4],
        )?;
        let params = build_rope_params(head_dim, 2, max_seq_len as u32, rope_theta, 1.0);
        // Dims are [head_dim, n_heads, 1]: heads are dim 1, tokens dim 2.
        let nrows = n_heads;
        let n_threads = session.dsp_threads().min(nrows as u32).max(1);
        let kparams = build_rope_kernel_params(head_dim, nrows, n_heads, 1, n_threads);
        Self::enqueue_labeled(
            session,
            "dispatch_rope",
            HtpOpCode::Rope as u32,
            &[act_ti, pos_ti],
            &[act_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-token (prefill) RoPE over `[head_dim, n_heads, n_tokens]`
    /// (heads dim 1, tokens dim 2), rotated in place by the `n_tokens`
    /// positions in `pos_buf`.
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
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let total_bytes = q_dim * n_tokens * 4;
        let act_ti = session.add_tensor(
            act,
            act_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, n_tokens as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (q_dim * 4) as u32,
                total_bytes as u32,
            ],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            n_tokens * 4,
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
        let params = build_rope_params(head_dim, 2, max_seq_len as u32, rope_theta, 1.0);
        let nrows = n_heads * n_tokens;
        let n_threads = session.dsp_threads().min(nrows as u32).max(1);
        let kparams = build_rope_kernel_params(head_dim, nrows, n_heads, n_tokens, n_threads);
        Self::enqueue_labeled(
            session,
            "dispatch_rope_m",
            HtpOpCode::Rope as u32,
            &[act_ti, pos_ti],
            &[act_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Single-row (decode) SetRows: appends one K (or V) row into the f16
    /// cache at the absolute slot in the positions vector. Values and
    /// cache are flat 2D (`[kv_dim, rows]`): the DSP worker iterates
    /// `ne02 * rows` DMA steps, so the old 3D per-head view cost a
    /// per-head round-trip (6x at 8 KV heads).
    fn dispatch_set_rows(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        cache: &RpcmemBuffer,
        cache_offset: usize,
        head_dim: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        let kv_dim = head_dim * n_kv_heads;
        let src_bytes = kv_dim * 4;
        let cache_bytes = kv_dim * max_seq_len * 2;
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [kv_dim as u32, 1, 1, 1],
            [4, (kv_dim * 4) as u32, src_bytes as u32, src_bytes as u32],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [1, 1, 1, 1],
            [4, 4, 4, 4],
        )?;
        let cache_ti = session.add_tensor(
            cache,
            cache_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                2,
                (kv_dim * 2) as u32,
                cache_bytes as u32,
                cache_bytes as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = build_set_rows_kernel_params(1, 1, 1, 1, kv_dim, true, session.dsp_threads());
        Self::enqueue_labeled(
            session,
            "dispatch_set_rows",
            HtpOpCode::SetRows as u32,
            &[src_ti, pos_ti],
            &[cache_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-row (prefill) SetRows: appends `n_rows` K (or V) rows from the
    /// flat `[kv_dim, n_rows]` values view into the interleaved f16 cache
    /// (`[kv_dim, max_seq]`, all heads contiguous per position) at the
    /// `n_rows` absolute slots in the positions vector.
    fn dispatch_set_rows_m(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        cache: &RpcmemBuffer,
        cache_offset: usize,
        head_dim: usize,
        n_kv_heads: usize,
        n_rows: usize,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        let kv_dim = head_dim * n_kv_heads;
        let src_bytes = kv_dim * n_rows * 4;
        let cache_bytes = kv_dim * max_seq_len * 2;
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [kv_dim as u32, n_rows as u32, 1, 1],
            [4, (kv_dim * 4) as u32, src_bytes as u32, src_bytes as u32],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_rows as u32, 1, 1, 1],
            [
                4,
                (n_rows * 4) as u32,
                (n_rows * 4) as u32,
                (n_rows * 4) as u32,
            ],
        )?;
        let cache_ti = session.add_tensor(
            cache,
            cache_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                2,
                (kv_dim * 2) as u32,
                cache_bytes as u32,
                cache_bytes as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams =
            build_set_rows_kernel_params(n_rows, 1, 1, 1, kv_dim, true, session.dsp_threads());
        Self::enqueue_labeled(
            session,
            "dispatch_set_rows_m",
            HtpOpCode::SetRows as u32,
            &[src_ti, pos_ti],
            &[cache_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_rms_norm_mul(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        weight: &RpcmemBuffer,
        weight_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        eps: f32,
        head_dim: usize,
        n_heads: usize,
    ) -> Result<(), CeraError> {
        let total_bytes = head_dim * n_heads * 4;
        let src_ti = session.add_tensor(
            src,
            src_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                total_bytes as u32,
                total_bytes as u32,
            ],
        )?;
        let weight_ti = session.add_tensor(
            weight,
            weight_offset,
            head_dim * 4,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                total_bytes as u32,
                total_bytes as u32,
            ],
        )?;
        let params = build_rms_norm_params(eps);
        let kparams = build_unary_kernel_params(
            head_dim,
            n_heads,
            head_dim,
            8 * 1024 * 1024,
            session.dsp_threads(),
            true,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_rms_norm_mul",
            HtpOpCode::RmsNormMul as u32,
            &[src_ti, weight_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_flash_attn_ext(
        session: &mut HexagonQueueSession,
        q: &RpcmemBuffer,
        q_offset: usize,
        k_cache: &RpcmemBuffer,
        k_offset: usize,
        v_cache: &RpcmemBuffer,
        v_offset: usize,
        mask: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        max_seq_len: usize,
        scale: f32,
    ) -> Result<(), CeraError> {
        let q_bytes = head_dim * n_heads * 4;
        let kv_dim = head_dim * n_kv_heads;
        let cache_bytes = kv_dim * max_seq_len * 2;
        let q_ti = session.add_tensor(
            q,
            q_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, n_heads as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                q_bytes as u32,
            ],
        )?;
        // Permuted views of the interleaved `[kv_dim, max_seq]` cache (see
        // `dispatch_flash_attn_m`).
        let k_ti = session.add_tensor(
            k_cache,
            k_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                2,
                (kv_dim * 2) as u32,
                (head_dim * 2) as u32,
                cache_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v_cache,
            v_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                2,
                (kv_dim * 2) as u32,
                (head_dim * 2) as u32,
                cache_bytes as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, n_heads as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                q_bytes as u32,
            ],
        )?;
        let mask_ti = session.add_tensor(
            mask,
            0,
            seq_len * 2,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [seq_len as u32, 1, 1, 1],
            [
                2,
                (seq_len * 2) as u32,
                (seq_len * 2) as u32,
                (seq_len * 2) as u32,
            ],
        )?;
        let mut params = [0i32; 16];
        params[0] = scale.to_bits() as i32;
        let kparams = build_flash_attn_kernel_params(
            head_dim,
            n_heads,
            n_kv_heads,
            1,
            seq_len,
            scale,
            session.dsp_threads(),
            true,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_flash_attn_ext",
            HtpOpCode::FlashAttnExt as u32,
            &[q_ti, k_ti, v_ti, mask_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Debug helper: flush pending ops, then log RMS/max_abs of a scratch
    /// region. Active only with CERA_DUMP_ACT set. Mirrors the CPU backend's
    /// `[cera.hidden]` log points for cross-backend diffing.
    fn dump_hidden(
        session: &mut HexagonQueueSession,
        scratch: &RpcmemBuffer,
        layer_idx: usize,
        tag: &str,
        offset: usize,
        len: usize,
    ) {
        if std::env::var("CERA_DUMP_ACT").is_err() {
            return;
        }
        if let Err(e) = session.flush() {
            tracing::error!("dump_hidden flush failed: {e}");
            return;
        }
        scratch.invalidate_cpu_cache(offset, len * 4);
        let act =
            unsafe { std::slice::from_raw_parts(scratch.as_ptr().add(offset) as *const f32, len) };
        let sum: f64 = act.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        let rms = (sum / len as f64).sqrt();
        let absmax = act.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("[hex] layer {layer_idx} {tag}: rms={rms:e} max_abs={absmax:e}");
    }

    /// M-token (prefill) FlashAttention: `n_tokens` queries in `[head_dim,
    /// n_tokens, n_heads]` over the `[head_dim, seq_len, n_kv_heads]` valid
    /// KV prefix, biased by the `[seq_len, n_tokens]` causal mask (query
    /// rows via dim-1 stride). The output follows the ggml permute(0, 2, 1,
    /// 3) convention: `[head_dim, n_heads, n_tokens]` (the firmware indexes
    /// head via dim-1 and token via dim-2, ignoring `dst->ne`).
    fn dispatch_flash_attn_m(
        &self,
        session: &mut HexagonQueueSession,
        q: &RpcmemBuffer,
        q_offset: usize,
        k_cache: &RpcmemBuffer,
        k_offset: usize,
        v_cache: &RpcmemBuffer,
        v_offset: usize,
        mask: &RpcmemBuffer,
        mask_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        n_tokens: usize,
        seq_len: usize,
        max_seq_len: usize,
        scale: f32,
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let kv_dim = head_dim * n_kv_heads;
        let q_bytes = q_dim * n_tokens * 4;
        let cache_bytes = kv_dim * max_seq_len * 2;
        let q_ti = session.add_tensor(
            q,
            q_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_tokens as u32, n_heads as u32, 1],
            [4, (q_dim * 4) as u32, (head_dim * 4) as u32, q_bytes as u32],
        )?;
        // Permuted views of the interleaved `[kv_dim, max_seq]` cache: head
        // h, position p starts at `p * kv_dim + h * head_dim`.
        let k_ti = session.add_tensor(
            k_cache,
            k_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                2,
                (kv_dim * 2) as u32,
                (head_dim * 2) as u32,
                cache_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v_cache,
            v_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                2,
                (kv_dim * 2) as u32,
                (head_dim * 2) as u32,
                cache_bytes as u32,
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
        let mask_ti = session.add_tensor(
            mask,
            mask_offset,
            seq_len * n_tokens * 2,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [seq_len as u32, n_tokens as u32, 1, 1],
            [
                2,
                (seq_len * 2) as u32,
                (seq_len * n_tokens * 2) as u32,
                (seq_len * n_tokens * 2) as u32,
            ],
        )?;
        let mut params = [0i32; 16];
        params[0] = scale.to_bits() as i32;
        // HMX first (DK % 8, M >= 5 at small head_dim), HVX fallback.
        let kparams = if self.use_hmx
            && fa_is_hmx_eligible(head_dim, n_tokens)
            && let Some(hmx) = build_hmx_fa_kernel_params(
                head_dim,
                n_heads,
                n_kv_heads,
                n_tokens,
                seq_len,
                scale,
                session.dsp_threads(),
                self.vtcm_budget,
            ) {
            hmx
        } else {
            build_flash_attn_kernel_params(
                head_dim,
                n_heads,
                n_kv_heads,
                n_tokens,
                seq_len,
                scale,
                session.dsp_threads(),
                true,
            )
        };
        Self::enqueue_labeled(
            session,
            "dispatch_flash_attn_m",
            HtpOpCode::FlashAttnExt as u32,
            &[q_ti, k_ti, v_ti, mask_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_cpy(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src_ti = Self::add_f32_vec(session, src, src_offset, dim, HTP_TENSOR_COMPUTE)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_cpy",
            HtpOpCode::Cpy as u32,
            &[src_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Strided 2D copy (assembly/transpose/scatter primitive): copies the
    /// `[ne0, ne1]` f32 tile between two strided descriptors (same index
    /// space, independent strides). A transpose carries the transposed
    /// shape on the source side; a scatter (interleaved destination)
    /// strides the destination side. The firmware resolves strides
    /// device-side (no kparams). NOTE: strided sides take the firmware's
    /// scalar per-element path — fine for state-sized (hs-scale) tiles,
    /// prohibitive for m*hs transposes (use CONCAT's transposed worker).
    fn dispatch_cpy_2d(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        src_ne0: usize,
        src_ne1: usize,
        src_nb0: usize,
        src_nb1: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dst_nb0: usize,
        dst_nb1: usize,
    ) -> Result<(), CeraError> {
        let span = |nb0: usize, nb1: usize| {
            src_ne0.saturating_sub(1) * nb0 + src_ne1.saturating_sub(1) * nb1 + 4
        };
        let src_span = span(src_nb0, src_nb1);
        let dst_span = span(dst_nb0, dst_nb1);
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [src_ne0 as u32, src_ne1 as u32, 1, 1],
            [
                src_nb0 as u32,
                src_nb1 as u32,
                src_span as u32,
                src_span as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            dst_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [src_ne0 as u32, src_ne1 as u32, 1, 1],
            [
                dst_nb0 as u32,
                dst_nb1 as u32,
                dst_span as u32,
                dst_span as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_cpy_2d",
            HtpOpCode::Cpy as u32,
            &[src_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// SsmConv dispatch: `y[c, m] = sum_t x[m + t, c] * w[t, c]` over the
    /// `[ncs, C]` input window (`ncs = d_conv - 1 + n_t`: prior states plus
    /// new inputs, time-major), `[d_conv, C]` oldest-first taps, producing
    /// channel-major `[C, n_t]` (already `[M, C]` row-major in memory: dst
    /// dim-1 stride is the token stride, so no transpose is needed).
    fn dispatch_ssm_conv(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        weights_offset: usize,
        conv_x: &RpcmemBuffer,
        conv_x_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        d_conv: usize,
        d_inner: usize,
        n_t: usize,
    ) -> Result<(), CeraError> {
        let ncs = d_conv - 1 + n_t;
        let x_ti = session.add_tensor(
            conv_x,
            conv_x_offset,
            ncs * d_inner * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [ncs as u32, d_inner as u32, 1, 1],
            [
                4,
                (ncs * 4) as u32,
                (ncs * d_inner * 4) as u32,
                (ncs * d_inner * 4) as u32,
            ],
        )?;
        let w_ti = session.add_tensor(
            weights,
            weights_offset,
            d_conv * d_inner * 4,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [d_conv as u32, d_inner as u32, 1, 1],
            [
                4,
                (d_conv * 4) as u32,
                (d_conv * d_inner * 4) as u32,
                (d_conv * d_inner * 4) as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            d_inner * n_t * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [d_inner as u32, n_t as u32, 1, 1],
            [
                4,
                (d_inner * 4) as u32,
                (d_inner * n_t * 4) as u32,
                (d_inner * n_t * 4) as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = build_ssm_conv_kernel_params(
            d_conv,
            d_inner,
            n_t,
            1,
            ncs,
            session.dsp_threads(),
            self.vtcm_budget,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_ssm_conv",
            HtpOpCode::SsmConv as u32,
            &[x_ti, w_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Batched prefill for one chunk of `tokens.len() <= PREFILL_MAX_ROWS`
    /// tokens at `start_pos`, returning last-position logits. All linears
    /// run as M-row HVX GEMMs (weights stream once per chunk); attention
    /// uses multi-query flash attention over the valid prefix with a
    /// host-built causal mask; conv layers use one SsmConv per chunk.
    fn try_forward_prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        let m = tokens.len();
        assert!(!tokens.is_empty() && m <= PREFILL_MAX_ROWS);
        let fwd_start = std::time::Instant::now();

        let vocab_size = self.config.vocab_size;
        if let Some(&bad) = tokens.iter().find(|&&t| (t as usize) >= vocab_size) {
            return Err(CeraError::Backend(format!(
                "token ID {bad} exceeds model vocab size {vocab_size}"
            )));
        }

        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());

        let hs = self.config.hidden_size;
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;

        // Gather token embeddings, positions, and the causal mask.
        let kv_len = start_pos + m;
        unsafe {
            for (i, &t) in tokens.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    self.token_embd.as_ptr().add(t as usize * hs),
                    scratch.as_mut_ptr().add(so.activation + i * hs * 4) as *mut f32,
                    hs,
                );
            }
            let pos_slice =
                std::slice::from_raw_parts_mut(scratch.as_mut_ptr().add(so.pos) as *mut i32, m);
            for (i, slot) in pos_slice.iter_mut().enumerate() {
                *slot = (start_pos + i) as i32;
            }
            // Causal mask [kv_len, M]: query i attends slots <= start_pos + i.
            let mask = std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.mask) as *mut u16,
                kv_len * m,
            );
            for (mm, row) in mask.chunks_mut(kv_len).enumerate() {
                let allowed = start_pos + mm + 1;
                for (kv, slot) in row.iter_mut().enumerate() {
                    *slot = if kv < allowed { 0x0000 } else { 0xFC00 };
                }
            }
        }
        scratch.flush_cpu_cache(so.activation, m * hs * 4);
        scratch.flush_cpu_cache(so.pos, m * 4);
        scratch.flush_cpu_cache(so.mask, kv_len * m * 2);

        let eps = self.config.rms_norm_eps;
        let intermediate_size = self.config.intermediate_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let max_seq_len = self.config.max_seq_len;
        let rope_theta = self.config.rope_theta;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

        let session = device.queue_session_mut();

        // Small-M determinism: cap ops per flush (reset after the final
        // flush below). Unconditional: the session outlives the forward and
        // a prior failed forward `?`-returned with its cap still set (the
        // None resets only run on tail paths), so entry state must not
        // depend on the previous forward's exit path. See
        // `SMALL_M_FLUSH_CAP_ROWS`.
        session.set_max_ops_per_flush(if m < SMALL_M_FLUSH_CAP_ROWS {
            Some(MAX_OPS_PER_FLUSH)
        } else {
            None
        });

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                HexagonLayer::Attention(attn) => {
                    let n_kv_heads = attn.kv_dim / head_dim;
                    let q_dim = attn.q_dim;
                    let kv_dim = attn.kv_dim;
                    // Block norm over M rows.
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        attn.attn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        m,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill attn_norm flush failed: {e}"
                        )));
                    }
                    // QKV projections (fused NX: one shared-activation op).
                    self.dispatch_mul_mat_nx(
                        session,
                        &self.weights_buf,
                        &[&attn.attn_q, &attn.attn_k, &attn.attn_v],
                        scratch,
                        so.normed,
                        scratch,
                        &[so.q, so.k, so.v],
                        m,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!("prefill QKV flush failed: {e}")));
                    }
                    // Per-head QK norms over M*n_heads head-rows.
                    if let Some(qn_offset) = attn.attn_q_norm_offset {
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            so.q,
                            &self.weights_buf,
                            qn_offset,
                            scratch,
                            so.q,
                            eps,
                            head_dim,
                            m * n_heads,
                        )?;
                    }
                    if let Some(kn_offset) = attn.attn_k_norm_offset {
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            so.k,
                            &self.weights_buf,
                            kn_offset,
                            scratch,
                            so.k,
                            eps,
                            head_dim,
                            m * n_kv_heads,
                        )?;
                    }
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill QK norm flush failed: {e}"
                        )));
                    }
                    // RoPE (DSP, or host loop under the debug fallback).
                    if self.cpu_rope {
                        if let Err(e) = session.flush() {
                            return Err(CeraError::Backend(format!(
                                "prefill pre-RoPE flush failed: {e}"
                            )));
                        }
                        let q_bytes = q_dim * m * 4;
                        let k_bytes = kv_dim * m * 4;
                        scratch.invalidate_cpu_cache(so.q, q_bytes);
                        scratch.invalidate_cpu_cache(so.k, k_bytes);
                        unsafe {
                            for mm in 0..m {
                                let q = std::slice::from_raw_parts_mut(
                                    scratch.as_mut_ptr().add(so.q + mm * q_dim * 4) as *mut f32,
                                    q_dim,
                                );
                                let k = std::slice::from_raw_parts_mut(
                                    scratch.as_mut_ptr().add(so.k + mm * kv_dim * 4) as *mut f32,
                                    kv_dim,
                                );
                                crate::backend::cpu::rope(
                                    q,
                                    k,
                                    start_pos + mm,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    rope_theta,
                                );
                            }
                        }
                        scratch.flush_cpu_cache(so.q, q_bytes);
                        scratch.flush_cpu_cache(so.k, k_bytes);
                    } else {
                        Self::dispatch_rope_m(
                            session,
                            scratch,
                            so.q,
                            scratch,
                            so.pos,
                            head_dim,
                            n_heads,
                            m,
                            max_seq_len,
                            rope_theta,
                        )?;
                        Self::dispatch_rope_m(
                            session,
                            scratch,
                            so.k,
                            scratch,
                            so.pos,
                            head_dim,
                            n_kv_heads,
                            m,
                            max_seq_len,
                            rope_theta,
                        )?;
                        if self.debug_barriers
                            && let Err(e) = session.flush()
                        {
                            return Err(CeraError::Backend(format!(
                                "prefill RoPE flush failed: {e}"
                            )));
                        }
                    }
                    // Append M K/V rows (slots == positions: reuse pos vector).
                    Self::dispatch_set_rows_m(
                        session,
                        scratch,
                        so.k,
                        scratch,
                        so.pos,
                        &self.kv_state_buf,
                        attn.k_offset,
                        head_dim,
                        n_kv_heads,
                        m,
                        max_seq_len,
                    )?;
                    Self::dispatch_set_rows_m(
                        session,
                        scratch,
                        so.v,
                        scratch,
                        so.pos,
                        &self.kv_state_buf,
                        attn.v_offset,
                        head_dim,
                        n_kv_heads,
                        m,
                        max_seq_len,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill SetRows flush failed: {e}"
                        )));
                    }
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) prefill v-proj",
                        so.v,
                        m * kv_dim,
                    );
                    // Multi-query attention over the valid prefix.
                    self.dispatch_flash_attn_m(
                        session,
                        scratch,
                        so.q,
                        &self.kv_state_buf,
                        attn.k_offset,
                        &self.kv_state_buf,
                        attn.v_offset,
                        scratch,
                        so.mask,
                        scratch,
                        so.attn_out,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        m,
                        kv_len,
                        max_seq_len,
                        attn_scale,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill FlashAttn flush failed: {e}"
                        )));
                    }
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) prefill fa-out",
                        so.attn_out,
                        m * q_dim,
                    );
                    // o_proj + residual.
                    self.dispatch_mul_mat_m(
                        session,
                        &self.weights_buf,
                        &attn.attn_output,
                        scratch,
                        so.attn_out,
                        scratch,
                        so.normed,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) prefill attn-out",
                        so.normed + (m - 1) * hs * 4,
                        hs,
                    );
                    Self::dispatch_add_m(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) prefill block-out",
                        so.activation + (m - 1) * hs * 4,
                        hs,
                    );
                    // FFN.
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        attn.ffn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        m,
                    )?;
                    self.dispatch_mul_mat_nx(
                        session,
                        &self.weights_buf,
                        &[&attn.ffn_gate, &attn.ffn_up],
                        scratch,
                        so.normed,
                        scratch,
                        &[so.ffn_gate, so.ffn_up],
                        m,
                    )?;
                    Self::dispatch_swiglu(
                        session,
                        scratch,
                        so.ffn_gate,
                        scratch,
                        so.ffn_up,
                        scratch,
                        so.ffn_out,
                        intermediate_size,
                        m,
                    )?;
                    self.dispatch_mul_mat_m(
                        session,
                        &self.weights_buf,
                        &attn.ffn_down,
                        scratch,
                        so.ffn_out,
                        scratch,
                        so.normed,
                        m,
                    )?;
                    Self::dispatch_add_m(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) prefill post-ffn",
                        so.activation + (m - 1) * hs * 4,
                        hs,
                    );
                }
                HexagonLayer::Conv(conv) => {
                    // Block norm + in_proj over M rows.
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        conv.attn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        m,
                    )?;
                    self.dispatch_mul_mat_m(
                        session,
                        &self.weights_buf,
                        &conv.in_proj,
                        scratch,
                        so.normed,
                        scratch,
                        so.conv_in,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) prefill conv_in r0",
                        so.conv_in,
                        3 * hs,
                    );
                    if m > 1 {
                        Self::dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) prefill conv_in r1",
                            so.conv_in + 3 * hs * 4,
                            3 * hs,
                        );
                    }
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill conv/in-proj flush failed: {e}"
                        )));
                    }
                    // b * x straight out of the strided in_proj thirds (no
                    // materializing copies); the DSP reads the row strides.
                    Self::dispatch_mul_m_strided(
                        session,
                        scratch,
                        so.conv_in,
                        HTP_TENSOR_COMPUTE,
                        scratch,
                        so.conv_in + 2 * hs * 4,
                        HTP_TENSOR_COMPUTE,
                        scratch,
                        so.conv_bx,
                        hs,
                        m,
                        3 * hs * 4,
                        3 * hs * 4,
                    )?;
                    // State prepend: CONCAT(state-as-[2, hs] + bx-as-[m,
                    // hs]) into conv_x `[ncs, hs]`, time-inner — one op
                    // replacing the s0/s1 scatter plus the bx transpose.
                    Self::dispatch_concat_2d(
                        session,
                        &self.kv_state_buf,
                        conv.state_offset,
                        2,
                        scratch,
                        so.conv_bx,
                        m,
                        hs * 4,
                        4,
                        scratch,
                        so.conv_x,
                        hs,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill conv/scatter flush failed: {e}"
                        )));
                    }
                    self.dispatch_ssm_conv(
                        session,
                        &self.weights_buf,
                        conv.conv_ssm_offset,
                        scratch,
                        so.conv_x,
                        scratch,
                        so.conv_ssm_y,
                        3,
                        hs,
                        m,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill conv/ssm-only flush failed: {e}"
                        )));
                    }
                    // No transpose: the SsmConv worker writes token t's C
                    // values at `t * C` (dst dim-1 stride is the token
                    // stride), so `conv_ssm_y` already holds [M, C]
                    // row-major. The gate and out_proj below read it
                    // directly; the old strided copy was an identity that
                    // took the firmware's scalar reshape path (10x wall
                    // past m=128).
                    // State writeback into the interleaved `[C, 2]` slots
                    // (slot t at `state + c*8 + t*4`): last two bx rows
                    // when m>=2, else shift + insert.
                    if m >= 2 {
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_bx + (m - 2) * hs * 4,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset,
                            8,
                            8,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_bx + (m - 1) * hs * 4,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            8,
                            8,
                        )?;
                    } else {
                        // Shift via scratch temp: odd->even overlaps in
                        // the state slab, and CPY has memcpy (not
                        // memmove) semantics.
                        Self::dispatch_cpy_2d(
                            session,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            hs,
                            1,
                            8,
                            8,
                            scratch,
                            so.conv_t0,
                            4,
                            hs * 4,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_t0,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset,
                            8,
                            8,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_bx,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            8,
                            8,
                        )?;
                    }
                    // Gate with the strided c third in place (no materialize).
                    Self::dispatch_mul_m_strided(
                        session,
                        scratch,
                        so.conv_ssm_y,
                        HTP_TENSOR_COMPUTE,
                        scratch,
                        so.conv_in + hs * 4,
                        HTP_TENSOR_COMPUTE,
                        scratch,
                        so.conv_ssm_y,
                        hs,
                        m,
                        hs * 4,
                        3 * hs * 4,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill conv/ssm flush failed: {e}"
                        )));
                    }
                    // out_proj + residual.
                    self.dispatch_mul_mat_m(
                        session,
                        &self.weights_buf,
                        &conv.out_proj,
                        scratch,
                        so.conv_ssm_y,
                        scratch,
                        so.normed,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) prefill conv_out r0",
                        so.normed,
                        hs,
                    );
                    if m > 1 {
                        Self::dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) prefill conv_out r1",
                            so.normed + hs * 4,
                            hs,
                        );
                    }
                    Self::dispatch_add_m(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) prefill block-out",
                        so.activation + (m - 1) * hs * 4,
                        hs,
                    );
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "prefill conv/out-proj flush failed: {e}"
                        )));
                    }
                    // FFN (same as attention blocks).
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        conv.ffn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        m,
                    )?;
                    self.dispatch_mul_mat_nx(
                        session,
                        &self.weights_buf,
                        &[&conv.ffn_gate, &conv.ffn_up],
                        scratch,
                        so.normed,
                        scratch,
                        &[so.ffn_gate, so.ffn_up],
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) prefill ffn_gate",
                        so.ffn_gate,
                        m * intermediate_size,
                    );
                    Self::dispatch_swiglu(
                        session,
                        scratch,
                        so.ffn_gate,
                        scratch,
                        so.ffn_up,
                        scratch,
                        so.ffn_out,
                        intermediate_size,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) prefill ffn_swiglu",
                        so.ffn_out,
                        m * intermediate_size,
                    );
                    self.dispatch_mul_mat_m(
                        session,
                        &self.weights_buf,
                        &conv.ffn_down,
                        scratch,
                        so.ffn_out,
                        scratch,
                        so.normed,
                        m,
                    )?;
                    Self::dispatch_add_m(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                        m,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "prefill post-ffn",
                        so.activation + (m - 1) * hs * 4,
                        hs,
                    );
                }
            }
            if (self.debug_barriers || m < SMALL_M_BARRIER_ROWS)
                && let Err(e) = session.flush()
            {
                return Err(CeraError::Backend(format!(
                    "prefill layer {layer_idx} flush failed: {e}"
                )));
            }
        }

        // Final norm + LM head on the last row only.
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            so.activation,
            &self.weights_buf,
            self.output_norm_offset,
            scratch,
            so.normed,
            eps,
            hs,
            m,
        )?;
        self.dispatch_mul_mat(
            session,
            &self.weights_buf,
            &self.lm_head,
            scratch,
            so.normed + (m - 1) * hs * 4,
            scratch,
            so.logits,
        )?;

        if let Err(e) = session.flush() {
            session.set_max_ops_per_flush(None);
            return Err(CeraError::Backend(format!(
                "Hexagon NPU prefill execution failed: {e}"
            )));
        }
        session.set_max_ops_per_flush(None);

        self.current_seq_len.store(start_pos + m, Ordering::SeqCst);
        state.seq_len = start_pos + m;

        scratch.invalidate_cpu_cache(so.logits, vocab_size * 4);
        let logits_slice = unsafe {
            std::slice::from_raw_parts(scratch.as_ptr().add(so.logits) as *const f32, vocab_size)
        };
        if let Ok(mut adpf) = self.adpf.lock()
            && let Some(session) = adpf.as_mut()
        {
            session.report(fwd_start.elapsed().as_nanos().min(i64::MAX as u128) as i64);
        }
        Ok(logits_slice.to_vec())
    }

    fn forward_prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.try_forward_prefill_chunk(tokens, start_pos, state)
            .unwrap_or_else(|e| {
                tracing::error!("Hexagon NPU prefill chunk failed: {e}");
                vec![0.0f32; self.config.vocab_size]
            })
    }

    fn try_forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let fwd_start = std::time::Instant::now();

        let token = tokens[0] as usize;
        if token >= self.config.vocab_size {
            return Err(CeraError::Backend(format!(
                "token ID {token} exceeds model vocab size {}",
                self.config.vocab_size
            )));
        }

        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());

        let hs = self.config.hidden_size;
        let embd_start = token * hs;
        let embd_slice = &self.token_embd[embd_start..embd_start + hs];

        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;

        // Copy token embedding to activation buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                embd_slice.as_ptr() as *const u8,
                scratch.as_mut_ptr().add(so.activation),
                hs * 4,
            );
            let pos_slice =
                std::slice::from_raw_parts_mut(scratch.as_mut_ptr().add(so.pos) as *mut i32, 2);
            pos_slice[0] = pos as i32;
            pos_slice[1] = 0;
        }
        scratch.flush_cpu_cache(so.activation, hs * 4);
        scratch.flush_cpu_cache(so.pos, 64);

        let eps = self.config.rms_norm_eps;
        let intermediate_size = self.config.intermediate_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let max_seq_len = self.config.max_seq_len;
        let rope_theta = self.config.rope_theta;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

        let session = device.queue_session_mut();

        // Decode determinism: cap ops per flush (reset after the final flush
        // below). See `MAX_OPS_PER_FLUSH`.
        session.set_max_ops_per_flush(Some(decode_ops_cap()));

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                HexagonLayer::Attention(attn) => {
                    // Attention RMS norm (fused): normed = rmsnorm(act) * attn_norm
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        attn.attn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        1,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "Attention attn_norm flush failed: {e}"
                        )));
                    }

                    // Projections: Q, K, V (fused NX).
                    self.dispatch_mul_mat_nx(
                        session,
                        &self.weights_buf,
                        &[&attn.attn_q, &attn.attn_k, &attn.attn_v],
                        scratch,
                        so.normed,
                        scratch,
                        &[so.q, so.k, so.v],
                        1,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "Attention QKV proj flush failed: {e}"
                        )));
                    }

                    let n_kv_heads = attn.kv_dim / head_dim;

                    // Optional Q/K norm
                    if let Some(qn_offset) = attn.attn_q_norm_offset {
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            so.q,
                            &self.weights_buf,
                            qn_offset,
                            scratch,
                            so.q,
                            eps,
                            head_dim,
                            n_heads,
                        )?;
                    }
                    if let Some(kn_offset) = attn.attn_k_norm_offset {
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            so.k,
                            &self.weights_buf,
                            kn_offset,
                            scratch,
                            so.k,
                            eps,
                            head_dim,
                            n_kv_heads,
                        )?;
                    }
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "Attention QK norm flush failed: {e}"
                        )));
                    }

                    // RoPE on Q and K: DSP kernel by default, with an optional
                    // host-CPU fallback (CERA_HEXAGON_CPU_ROPE=1) using the same
                    // cpu::rope the CPU backend uses.
                    if self.cpu_rope {
                        // Host-CPU RoPE reads DSP-produced Q/K in place, so
                        // this barrier is mandatory even in fused mode.
                        if let Err(e) = session.flush() {
                            return Err(CeraError::Backend(format!(
                                "Attention pre-RoPE flush failed: {e}"
                            )));
                        }
                        let q_bytes = attn.q_dim * 4;
                        let k_bytes = attn.kv_dim * 4;
                        scratch.invalidate_cpu_cache(so.q, q_bytes);
                        scratch.invalidate_cpu_cache(so.k, k_bytes);
                        unsafe {
                            let q = std::slice::from_raw_parts_mut(
                                scratch.as_mut_ptr().add(so.q) as *mut f32,
                                attn.q_dim,
                            );
                            let k = std::slice::from_raw_parts_mut(
                                scratch.as_mut_ptr().add(so.k) as *mut f32,
                                attn.kv_dim,
                            );
                            crate::backend::cpu::rope(
                                q, k, pos, n_heads, n_kv_heads, head_dim, rope_theta,
                            );
                        }
                        scratch.flush_cpu_cache(so.q, q_bytes);
                        scratch.flush_cpu_cache(so.k, k_bytes);
                    } else {
                        Self::dispatch_rope(
                            session,
                            scratch,
                            so.q,
                            scratch,
                            so.pos,
                            head_dim,
                            n_heads,
                            max_seq_len,
                            rope_theta,
                        )?;
                        Self::dispatch_rope(
                            session,
                            scratch,
                            so.k,
                            scratch,
                            so.pos,
                            head_dim,
                            n_kv_heads,
                            max_seq_len,
                            rope_theta,
                        )?;
                        if self.debug_barriers
                            && let Err(e) = session.flush()
                        {
                            return Err(CeraError::Backend(format!(
                                "Attention RoPE flush failed: {e}"
                            )));
                        }
                    }

                    // SetRows K and V into KV cache
                    Self::dispatch_set_rows(
                        session,
                        scratch,
                        so.k,
                        scratch,
                        so.pos,
                        &self.kv_state_buf,
                        attn.k_offset,
                        head_dim,
                        n_kv_heads,
                        max_seq_len,
                    )?;
                    Self::dispatch_set_rows(
                        session,
                        scratch,
                        so.v,
                        scratch,
                        so.pos,
                        &self.kv_state_buf,
                        attn.v_offset,
                        head_dim,
                        n_kv_heads,
                        max_seq_len,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "Attention SetRows flush failed: {e}"
                        )));
                    }

                    // Flash Attention
                    Self::dispatch_flash_attn_ext(
                        session,
                        scratch,
                        so.q,
                        &self.kv_state_buf,
                        attn.k_offset,
                        &self.kv_state_buf,
                        attn.v_offset,
                        &self.mask_buf,
                        scratch,
                        so.attn_out,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        pos + 1,
                        max_seq_len,
                        attn_scale,
                    )?;
                    if self.debug_barriers
                        && let Err(e) = session.flush()
                    {
                        return Err(CeraError::Backend(format!(
                            "Attention layer {layer_idx} FlashAttnExt flush failed: {e}"
                        )));
                    }

                    // Attention output projection
                    self.dispatch_mul_mat(
                        session,
                        &self.weights_buf,
                        &attn.attn_output,
                        scratch,
                        so.attn_out,
                        scratch,
                        so.normed,
                    )?;

                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) block-out",
                        so.normed,
                        hs,
                    );

                    // Residual add: act = act + normed
                    Self::dispatch_add(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(attn) post-block",
                        so.activation,
                        hs,
                    );

                    // FFN
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        attn.ffn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        1,
                    )?;
                    self.dispatch_mul_mat_nx(
                        session,
                        &self.weights_buf,
                        &[&attn.ffn_gate, &attn.ffn_up],
                        scratch,
                        so.normed,
                        scratch,
                        &[so.ffn_gate, so.ffn_up],
                        1,
                    )?;
                    Self::dispatch_swiglu(
                        session,
                        scratch,
                        so.ffn_gate,
                        scratch,
                        so.ffn_up,
                        scratch,
                        so.ffn_out,
                        intermediate_size,
                        1,
                    )?;
                    self.dispatch_mul_mat(
                        session,
                        &self.weights_buf,
                        &attn.ffn_down,
                        scratch,
                        so.ffn_out,
                        scratch,
                        so.normed,
                    )?;
                    Self::dump_hidden(session, scratch, layer_idx, "ffn-out", so.normed, hs);
                    Self::dispatch_add(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                    )?;
                }
                HexagonLayer::Conv(conv) => {
                    // Conv RMS norm
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        conv.attn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        1,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) normed-act",
                        so.normed,
                        hs,
                    );

                    // in_proj: hs -> 3 * hs (b, c, x)
                    self.dispatch_mul_mat(
                        session,
                        &self.weights_buf,
                        &conv.in_proj,
                        scratch,
                        so.normed,
                        scratch,
                        so.conv_in,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) conv-in",
                        so.conv_in,
                        3 * hs,
                    );

                    if self.use_ssm_conv {
                        // bx = b * x (M=1 rows are contiguous, as in the manual path)
                        Self::dispatch_mul(
                            session,
                            scratch,
                            so.conv_in,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_in + 2 * hs * 4,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_bx,
                            hs,
                        )?;
                        // State prepend: CONCAT([s0; s1] + bx-as-[1, hs])
                        // into conv_x `[3, hs]` (same op as prefill; the
                        // s0/s1 slab is adjacent by construction).
                        Self::dispatch_concat_2d(
                            session,
                            &self.kv_state_buf,
                            conv.state_offset,
                            2,
                            scratch,
                            so.conv_bx,
                            1,
                            hs * 4,
                            4,
                            scratch,
                            so.conv_x,
                            hs,
                        )?;
                        // y = shortconv(conv_x), channel-major [C, 1]
                        self.dispatch_ssm_conv(
                            session,
                            &self.weights_buf,
                            conv.conv_ssm_offset,
                            scratch,
                            so.conv_x,
                            scratch,
                            so.conv_ssm_y,
                            3,
                            hs,
                            1,
                        )?;
                        // [C, 1] -> [1, C] is a flat copy (same bytes)
                        Self::dispatch_cpy(
                            session,
                            scratch,
                            so.conv_ssm_y,
                            scratch,
                            so.conv_y,
                            hs,
                        )?;
                        Self::dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) ssm-y",
                            so.conv_y,
                            hs,
                        );
                        // Update states: s0 = s1; s1 = bx. The odd->even
                        // shift overlaps in the interleaved slab, so it
                        // stages through conv_t0 (unused on this path).
                        Self::dispatch_cpy_2d(
                            session,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            hs,
                            1,
                            8,
                            8,
                            scratch,
                            so.conv_t0,
                            4,
                            hs * 4,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_t0,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset,
                            8,
                            8,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_bx,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            8,
                            8,
                        )?;
                    } else {
                        // bx = b * x
                        Self::dispatch_mul(
                            session,
                            scratch,
                            so.conv_in,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_in + 2 * hs * 4,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_bx,
                            hs,
                        )?;
                        Self::dump_hidden(session, scratch, layer_idx, "(conv) bx", so.conv_bx, hs);

                        // De-interleave [s0; s1] into conv_x rows 0-1
                        // (conv_x is unused on the manual path); the MUL
                        // worker only reads dense rows.
                        Self::dispatch_cpy_2d(
                            session,
                            &self.kv_state_buf,
                            conv.state_offset,
                            hs,
                            1,
                            8,
                            8,
                            scratch,
                            so.conv_x,
                            4,
                            hs * 4,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            hs,
                            1,
                            8,
                            8,
                            scratch,
                            so.conv_x + hs * 4,
                            4,
                            hs * 4,
                        )?;
                        // Rolling conv: y = s0 * w0 + s1 * w1 + bx * w2
                        Self::dispatch_mul(
                            session,
                            &self.weights_buf,
                            conv.conv_w0_offset,
                            HTP_TENSOR_WEIGHT,
                            scratch,
                            so.conv_x,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_t0,
                            hs,
                        )?;
                        Self::dispatch_mul(
                            session,
                            &self.weights_buf,
                            conv.conv_w1_offset,
                            HTP_TENSOR_WEIGHT,
                            scratch,
                            so.conv_x + hs * 4,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_t1,
                            hs,
                        )?;
                        Self::dispatch_mul(
                            session,
                            &self.weights_buf,
                            conv.conv_w2_offset,
                            HTP_TENSOR_WEIGHT,
                            scratch,
                            so.conv_bx,
                            HTP_TENSOR_COMPUTE,
                            scratch,
                            so.conv_y,
                            hs,
                        )?;
                        Self::dispatch_add(
                            session, scratch, so.conv_t0, scratch, so.conv_t1, scratch, so.conv_t0,
                            hs,
                        )?;
                        Self::dispatch_add(
                            session, scratch, so.conv_y, scratch, so.conv_t0, scratch, so.conv_y,
                            hs,
                        )?;

                        // Update states: s0 = s1; s1 = bx. The odd->even
                        // shift stages through conv_ssm_y (unused on the
                        // manual path).
                        Self::dispatch_cpy_2d(
                            session,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            hs,
                            1,
                            8,
                            8,
                            scratch,
                            so.conv_ssm_y,
                            4,
                            hs * 4,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_ssm_y,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset,
                            8,
                            8,
                        )?;
                        Self::dispatch_cpy_2d(
                            session,
                            scratch,
                            so.conv_bx,
                            hs,
                            1,
                            4,
                            hs * 4,
                            &self.kv_state_buf,
                            conv.state_offset + 4,
                            8,
                            8,
                        )?;
                    }

                    // Gate: y = y * c
                    Self::dispatch_mul(
                        session,
                        scratch,
                        so.conv_in + hs * 4,
                        HTP_TENSOR_COMPUTE,
                        scratch,
                        so.conv_y,
                        HTP_TENSOR_COMPUTE,
                        scratch,
                        so.conv_y,
                        hs,
                    )?;
                    Self::dump_hidden(session, scratch, layer_idx, "(conv) gated-y", so.conv_y, hs);

                    // out_proj: hs -> hs
                    self.dispatch_mul_mat(
                        session,
                        &self.weights_buf,
                        &conv.out_proj,
                        scratch,
                        so.conv_y,
                        scratch,
                        so.normed,
                    )?;

                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) block-out",
                        so.normed,
                        hs,
                    );

                    // Residual add: act = act + normed
                    Self::dispatch_add(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) post-block",
                        so.activation,
                        hs,
                    );

                    // FFN
                    Self::dispatch_rms_norm_mul(
                        session,
                        scratch,
                        so.activation,
                        &self.weights_buf,
                        conv.ffn_norm_offset,
                        scratch,
                        so.normed,
                        eps,
                        hs,
                        1,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) normed-ffn",
                        so.normed,
                        hs,
                    );
                    self.dispatch_mul_mat_nx(
                        session,
                        &self.weights_buf,
                        &[&conv.ffn_gate, &conv.ffn_up],
                        scratch,
                        so.normed,
                        scratch,
                        &[so.ffn_gate, so.ffn_up],
                        1,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) ffn-gate",
                        so.ffn_gate,
                        intermediate_size,
                    );
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) ffn-up",
                        so.ffn_up,
                        intermediate_size,
                    );
                    Self::dispatch_swiglu(
                        session,
                        scratch,
                        so.ffn_gate,
                        scratch,
                        so.ffn_up,
                        scratch,
                        so.ffn_out,
                        intermediate_size,
                        1,
                    )?;
                    Self::dump_hidden(
                        session,
                        scratch,
                        layer_idx,
                        "(conv) ffn-out-act",
                        so.ffn_out,
                        intermediate_size,
                    );
                    self.dispatch_mul_mat(
                        session,
                        &self.weights_buf,
                        &conv.ffn_down,
                        scratch,
                        so.ffn_out,
                        scratch,
                        so.normed,
                    )?;
                    Self::dump_hidden(session, scratch, layer_idx, "ffn-out", so.normed, hs);
                    Self::dispatch_add(
                        session,
                        scratch,
                        so.activation,
                        scratch,
                        so.normed,
                        scratch,
                        so.activation,
                        hs,
                    )?;
                }
            }

            if self.debug_barriers
                && let Err(e) = session.flush()
            {
                return Err(CeraError::Backend(format!(
                    "Hexagon NPU layer {layer_idx} execution failed: {e}"
                )));
            }

            Self::dump_hidden(session, scratch, layer_idx, "post-ffn", so.activation, hs);
        }

        // Final output norm (fused): normed = rmsnorm(act) * output_norm
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            so.activation,
            &self.weights_buf,
            self.output_norm_offset,
            scratch,
            so.normed,
            eps,
            hs,
            1,
        )?;

        // Final LM head: logits = mul_mat(lm_head, normed)
        let vocab_size = self.config.vocab_size;
        self.dispatch_mul_mat(
            session,
            &self.weights_buf,
            &self.lm_head,
            scratch,
            so.normed,
            scratch,
            so.logits,
        )?;

        // Flush remaining queued operations to DSP and await execution
        // completion. (Decode flushes every `MAX_OPS_PER_FLUSH` ops
        // for determinism; on failure, re-run with
        // CERA_HEXAGON_BARRIERS=1 (per-group flushes) or
        // CERA_HEXAGON_STEP=1 (per-op flush) to localize the bad op.)
        if let Err(e) = session.flush() {
            session.set_max_ops_per_flush(None);
            return Err(CeraError::Backend(format!(
                "Hexagon NPU execution failed: {e}"
            )));
        }
        session.set_max_ops_per_flush(None);

        self.current_seq_len.store(pos + 1, Ordering::SeqCst);
        state.seq_len = pos + 1;

        // Invalidate CPU cache for logits output buffer before reading
        scratch.invalidate_cpu_cache(so.logits, vocab_size * 4);
        let logits_slice = unsafe {
            std::slice::from_raw_parts(scratch.as_ptr().add(so.logits) as *const f32, vocab_size)
        };
        if let Ok(mut adpf) = self.adpf.lock()
            && let Some(session) = adpf.as_mut()
        {
            session.report(fwd_start.elapsed().as_nanos().min(i64::MAX as u128) as i64);
        }
        Ok(logits_slice.to_vec())
    }
}

impl Model for HexagonLfm2Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.session_gate.try_acquire().map(Some)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.try_forward(tokens, pos, state).unwrap_or_else(|e| {
            tracing::error!("Hexagon NPU decode failed: {e}");
            vec![0.0f32; self.config.vocab_size]
        })
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        if tokens.is_empty() {
            return vec![0.0f32; self.config.vocab_size];
        }
        // Chunk to scratch capacity; sequential chunks append KV in order and
        // the final chunk's last-position logits are the prompt's logits.
        let mut logits = Vec::new();
        for (chunk_idx, chunk) in tokens.chunks(PREFILL_MAX_ROWS).enumerate() {
            logits =
                self.forward_prefill_chunk(chunk, start_pos + chunk_idx * PREFILL_MAX_ROWS, state);
        }
        logits
    }

    fn supports_all_logits(&self) -> bool {
        false
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        state.truncate_to(len);
        self.current_seq_len.store(len, Ordering::SeqCst);
    }

    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        let current = self.current_seq_len.load(Ordering::SeqCst);
        if len > current {
            return Err(KvRewindError::OutOfBounds {
                requested: len,
                current,
            });
        }
        state.check_truncate_to(len)
    }

    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        self.check_kv_rewind(state, len)?;
        state.try_truncate_to(len)?;
        self.current_seq_len.store(len, Ordering::SeqCst);
        Ok(())
    }

    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        if !matches!(compression, KvCompression::None) {
            return Err(CeraError::Backend(
                "TurboQuant KV compression is not supported by the Hexagon backend".into(),
            ));
        }
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        let mut fresh = InferenceState::from_config_capped(&self.config, compression, max_seq_len)?;
        fresh.lora = state.lora.clone();

        // Clear unified KV cache and convolution state buffer
        unsafe {
            std::ptr::write_bytes(self.kv_state_buf.as_mut_ptr(), 0, self.kv_state_buf.size());
        }
        self.kv_state_buf
            .flush_cpu_cache(0, self.kv_state_buf.size());

        self.current_seq_len.store(0, Ordering::SeqCst);
        *state = fresh;
        Ok(())
    }
}

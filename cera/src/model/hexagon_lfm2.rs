//! Native Qualcomm Hexagon NPU model implementation for LFM2 hybrid architectures.
//!
//! Provides HTP-accelerated forward execution on Snapdragon mobile and edge platforms
//! using FastRPC shared memory and per-architecture DSP skeleton libraries.

// The dispatch_* builders below mirror llama.cpp's fixed C signatures
// (session + per-tensor buffer/offset/flags + dims); bundling into structs
// would diverge from that truth at every call site for no gain.
#![allow(clippy::too_many_arguments)]

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::cpu::RopeType;
use crate::backend::hexagon::{
    AdpfSession, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch,
    HexagonContext, HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, RpcmemBuffer,
    StagedBatch, TILE_SIZE_Q4_0, TILE_SIZE_Q4_K, TILE_SIZE_Q6_K, TILE_SIZE_Q8_0,
    build_binary_kernel_params, build_flash_attn_kernel_params, build_hmx_fa_kernel_params,
    build_hmx_mm_kernel_params, build_mul_mat_kernel_params, build_rms_norm_params,
    build_rope_kernel_params, build_rope_params, build_set_rows_kernel_params,
    build_ssm_conv_kernel_params, build_unary_kernel_params, fa_is_hmx_eligible, mm_hmx_nb1,
    mm_is_hmx_eligible, repack_q4_0, repack_q4_k, repack_q6_k, repack_q8_0,
    repacked_matrix_size_q4_0, repacked_matrix_size_q4_k, repacked_matrix_size_q6_k,
    repacked_matrix_size_q8_0, requant_q5_k_to_q8_0,
};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::session_gate::{ModelSessionGate, ModelSessionLease};
use crate::model::{BlockType, Model, ModelConfig, record_first_fault, take_fault};
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

// Dual activation and normed ping-pong buffers isolate adjacent layers,
// preventing read-after-write collisions across layer boundaries.

/// Default decode ops-per-flush cap. The cap is phase-sensitive, not a
/// safety threshold: cap 20 was clean, then adding conv state-copy ops
/// re-phased its windows back into the race (6/6 64-token greedy md5s
/// diverged). 12 is verified 29/30 over 64-token greedy runs (2 prompts
/// x 2 quants); the single miss is a one-token near-tie flip between two
/// sane attractors (hex consensus == CPU text exactly), cap-independent
/// (cap 8 shows the same attractor pair), i.e. residual LSB DSP noise,
/// not window corruption. Any op-count change per layer must re-verify
/// Decode ops-per-flush cap; override via `CERA_HEXAGON_DECODE_OPS` (0
/// or unset enables single-flush decode with ping-pong buffering).
fn decode_ops_cap() -> Option<usize> {
    std::env::var("CERA_HEXAGON_DECODE_OPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

pub const MAX_ALL_LOGITS_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug)]
struct ScratchOffsets {
    activation: usize,
    activation_b: usize,
    normed: usize,
    normed_b: usize,
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
    argmax: usize,
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
        // Dual activation and normed buffers enable ping-pong scratch buffering across layers.
        let activation = cur;
        cur = align(cur + m * hidden_size * 4);
        let activation_b = cur;
        cur = align(cur + m * hidden_size * 4);
        let normed = cur;
        cur = align(cur + m * hidden_size * 4);
        let normed_b = cur;
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
        cur = align(cur + MAX_ALL_LOGITS_TOKENS * vocab_size * 4);
        let argmax = cur;
        cur = align(cur + MAX_ALL_LOGITS_TOKENS * 4);
        let pos = cur;
        cur = align(cur + m * 4);
        let mask = cur;
        cur = align(cur + m * max_seq_len * 2);
        let total_size = cur;

        Self {
            activation,
            activation_b,
            normed,
            normed_b,
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
            argmax,
            pos,
            mask,
            total_size,
        }
    }
}

/// Static pre-serialized command queue template for zero-allocation decode dispatch.
struct DecodeTemplate {
    resident_id: u64,
    staged: StagedBatch,
    flash_attn_patches: Vec<FlashAttnPatch>,
    patch_ranges: Vec<std::ops::Range<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct FlashAttnPatch {
    op_idx: usize,
    k_ti: usize,
    v_ti: usize,
    mask_ti: usize,
    g: usize,
}

/// Hexagon NPU accelerated model instance for LFM2 dense hybrid transformers.
pub struct HexagonLfm2Model {
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
    dump_act: bool,
    current_seq_len: AtomicUsize,
    /// Last decode failure, recorded by `forward` for
    /// [`Model::take_decode_error`]: the trait's decode surface has no
    /// error channel, so without this the session would sample the
    /// zero-logits fallback as token 0 and keep generating.
    decode_error: Mutex<Option<CeraError>>,
    /// KV cache wire data type (F16 or Q8_0).
    kv_dtype: HtpDataType,
    /// Cached static command queue template for zero-allocation decode forward passes.
    decode_template: Mutex<Option<DecodeTemplate>>,
    /// Cached static command queue template for zero-allocation greedy (on-DSP argmax) decode forward passes.
    greedy_decode_template: Mutex<Option<DecodeTemplate>>,
}

unsafe impl Send for HexagonLfm2Model {}
unsafe impl Sync for HexagonLfm2Model {}

enum PrefillInput<'a> {
    Tokens(&'a [u32]),
    Embeddings(&'a [f32]),
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum DecodeInput<'a> {
    Token(u32),
    Embedding(&'a [f32]),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DecodeOutput {
    Logits,
    Hidden,
    Greedy,
}

enum DecodeResult {
    Logits(Vec<f32>),
    Hidden(Vec<f32>),
    Greedy(u32),
}

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
        let dump_act = std::env::var_os("CERA_DUMP_ACT").is_some();
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

        let device = crate::backend::hexagon::probe_device(context.driver(), arch_override)?;

        let token_embd_tensor = gguf
            .get_tensor("token_embd.weight")
            .map_err(|e| CeraError::Backend(format!("missing token_embd.weight: {e}")))?;
        let expected_elements = vocab_size
            .checked_mul(hidden_size)
            .ok_or_else(|| CeraError::Backend("vocab_size * hidden_size overflows usize".into()))?;
        let token_embd = token_embd_tensor.to_f32_vec();
        if token_embd.len() < expected_elements {
            return Err(CeraError::Backend(format!(
                "token_embd.weight has {} elements, expected at least {expected_elements}",
                token_embd.len()
            )));
        }

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
        let kv_dtype = if std::env::var("CERA_HEXAGON_KV_Q8")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        {
            HtpDataType::Q8_0
        } else {
            HtpDataType::F16
        };
        let state_size = align256(hidden_size * 4);
        let mut kv_state_total = 0;
        let mut layers = Vec::with_capacity(n_layers);

        for (i, planned) in planned_layers.into_iter().enumerate() {
            match planned {
                PlannedLayer::Attention(pa) => {
                    let n_kv = config.kv_heads_per_layer[i];
                    let kv_dim = n_kv * head_dim;
                    let kv_slab_size = align256(if kv_dtype == HtpDataType::Q8_0 {
                        max_seq_len * kv_dim.div_ceil(32) * 34
                    } else {
                        max_seq_len * kv_dim * 2
                    });
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
        let copy_norm = |name: &str,
                         offset: usize,
                         buf: &mut RpcmemBuffer|
         -> Result<(), CeraError> {
            let t = gguf
                .get_tensor(name)
                .map_err(|e| CeraError::Backend(format!("missing tensor {name}: {e}")))?;
            let f32_vals = t.to_f32_vec();
            let byte_size = f32_vals.len() * std::mem::size_of::<f32>();
            if offset.saturating_add(byte_size) > buf.size() {
                return Err(CeraError::Backend(format!(
                    "norm tensor {name} byte size ({byte_size}) exceeds weights buffer capacity at offset {offset}"
                )));
            }
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
                    let expected_conv_len = hidden_size.checked_mul(3).ok_or_else(|| {
                        CeraError::Backend(
                            "hidden_size overflow calculating conv weight size".into(),
                        )
                    })?;
                    if conv_f32.len() != expected_conv_len {
                        return Err(CeraError::Backend(format!(
                            "blk.{i}.shortconv.conv.weight size {} != hidden_size * 3 ({} * 3)",
                            conv_f32.len(),
                            hidden_size
                        )));
                    }
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
            dump_act,
            current_seq_len: AtomicUsize::new(0),
            decode_error: Mutex::new(None),
            kv_dtype,
            decode_template: Mutex::new(None),
            greedy_decode_template: Mutex::new(None),
        })
    }

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

    fn dispatch_argmax(
        session: &mut HexagonQueueSession,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
        vocab_size: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let in_ti = session.add_tensor(
            in_act,
            in_offset,
            n_rows * vocab_size * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [vocab_size as u32, n_rows as u32, 1, 1],
            [
                4,
                (vocab_size * 4) as u32,
                (n_rows * vocab_size * 4) as u32,
                (n_rows * vocab_size * 4) as u32,
            ],
        )?;
        let out_ti = session.add_tensor(
            out_act,
            out_offset,
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
        Self::enqueue_labeled(
            session,
            "dispatch_argmax",
            HtpOpCode::Argmax as u32,
            &[in_ti],
            &[out_ti],
            [0i32; 16],
            [0i32; 32],
        )?;
        Ok(())
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
    /// s1_nb1`), which takes the DSP's specialized 2D-transposed worker:
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
        let s1_span = s1_rows.saturating_sub(1) * s1_nb0 + dim.saturating_sub(1) * s1_nb1 + 4;
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
        // HMX first (M >= 5 prefill), HVX fallback: mirrors ggml's
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
            && ws.iter().all(|w| w.out_dim <= w0.out_dim)
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
        // HMX first, HVX fallback: the same selection as singles, with
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
    fn dispatch_set_rows_typed(
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
        kv_dtype: HtpDataType,
    ) -> Result<(), CeraError> {
        let kv_dim = head_dim * n_kv_heads;
        let src_bytes = kv_dim * 4;
        let cache_bytes = if kv_dtype == HtpDataType::Q8_0 {
            max_seq_len * kv_dim.div_ceil(32) * 34
        } else {
            kv_dim * max_seq_len * 2
        };
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
            kv_dtype as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                if kv_dtype == HtpDataType::Q8_0 { 1 } else { 2 },
                if kv_dtype == HtpDataType::Q8_0 {
                    (kv_dim.div_ceil(32) * 34) as u32
                } else {
                    (kv_dim * 2) as u32
                },
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
    /// flat `[kv_dim, n_rows]` values view into the interleaved cache
    /// (`[kv_dim, max_seq]`, all heads contiguous per position) at the
    /// `n_rows` absolute slots in the positions vector.
    fn dispatch_set_rows_m_typed(
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
        kv_dtype: HtpDataType,
    ) -> Result<(), CeraError> {
        let kv_dim = head_dim * n_kv_heads;
        let src_bytes = kv_dim * n_rows * 4;
        let cache_bytes = if kv_dtype == HtpDataType::Q8_0 {
            max_seq_len * kv_dim.div_ceil(32) * 34
        } else {
            kv_dim * max_seq_len * 2
        };
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
            kv_dtype as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                if kv_dtype == HtpDataType::Q8_0 { 1 } else { 2 },
                if kv_dtype == HtpDataType::Q8_0 {
                    (kv_dim.div_ceil(32) * 34) as u32
                } else {
                    (kv_dim * 2) as u32
                },
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

    fn dispatch_flash_attn_ext_typed(
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
        kv_dtype: HtpDataType,
    ) -> Result<(usize, usize, usize), CeraError> {
        let q_bytes = head_dim * n_heads * 4;
        let kv_dim = head_dim * n_kv_heads;
        let cache_bytes = if kv_dtype == HtpDataType::Q8_0 {
            max_seq_len * kv_dim.div_ceil(32) * 34
        } else {
            kv_dim * max_seq_len * 2
        };
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
        let k_ti = session.add_tensor(
            k_cache,
            k_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                if kv_dtype == HtpDataType::Q8_0 { 1 } else { 2 },
                if kv_dtype == HtpDataType::Q8_0 {
                    (kv_dim.div_ceil(32) * 34) as u32
                } else {
                    (kv_dim * 2) as u32
                },
                if kv_dtype == HtpDataType::Q8_0 {
                    (head_dim.div_ceil(32) * 34) as u32
                } else {
                    (head_dim * 2) as u32
                },
                cache_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v_cache,
            v_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                if kv_dtype == HtpDataType::Q8_0 { 1 } else { 2 },
                if kv_dtype == HtpDataType::Q8_0 {
                    (kv_dim.div_ceil(32) * 34) as u32
                } else {
                    (kv_dim * 2) as u32
                },
                if kv_dtype == HtpDataType::Q8_0 {
                    (head_dim.div_ceil(32) * 34) as u32
                } else {
                    (head_dim * 2) as u32
                },
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
        let mask_bytes = seq_len * 2;
        let mask_ti = session.add_tensor(
            mask,
            0,
            mask_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [seq_len as u32, 1, 1, 1],
            [2, mask_bytes as u32, mask_bytes as u32, mask_bytes as u32],
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
        Ok((k_ti as usize, v_ti as usize, mask_ti as usize))
    }

    /// Flush pending ops if debug_barriers is set.
    #[inline]
    fn debug_barrier(
        &self,
        session: &mut HexagonQueueSession,
        label: &str,
    ) -> Result<(), CeraError> {
        if self.debug_barriers {
            session
                .flush()
                .map_err(|e| CeraError::Backend(format!("{label} flush failed: {e}")))?;
        }
        Ok(())
    }

    /// Debug helper: flush pending ops, then log RMS/max_abs of a scratch
    /// region. Active only with CERA_DUMP_ACT set. Mirrors the CPU backend's
    /// `[cera.hidden]` log points for cross-backend diffing.
    fn dump_hidden(
        &self,
        session: &mut HexagonQueueSession,
        scratch: &RpcmemBuffer,
        layer_idx: usize,
        tag: &str,
        offset: usize,
        len: usize,
    ) {
        if !self.dump_act {
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
        let cache_bytes = if self.kv_dtype == HtpDataType::Q8_0 {
            max_seq_len * kv_dim.div_ceil(32) * 34
        } else {
            kv_dim * max_seq_len * 2
        };
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
            self.kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                if self.kv_dtype == HtpDataType::Q8_0 {
                    1
                } else {
                    2
                },
                if self.kv_dtype == HtpDataType::Q8_0 {
                    (kv_dim.div_ceil(32) * 34) as u32
                } else {
                    (kv_dim * 2) as u32
                },
                if self.kv_dtype == HtpDataType::Q8_0 {
                    (head_dim.div_ceil(32) * 34) as u32
                } else {
                    (head_dim * 2) as u32
                },
                cache_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v_cache,
            v_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            self.kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                if self.kv_dtype == HtpDataType::Q8_0 {
                    1
                } else {
                    2
                },
                if self.kv_dtype == HtpDataType::Q8_0 {
                    (kv_dim.div_ceil(32) * 34) as u32
                } else {
                    (kv_dim * 2) as u32
                },
                if self.kv_dtype == HtpDataType::Q8_0 {
                    (head_dim.div_ceil(32) * 34) as u32
                } else {
                    (head_dim * 2) as u32
                },
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
    /// scalar per-element path: fine for state-sized (hs-scale) tiles,
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

    /// Batched prefill for one chunk of up to `PREFILL_MAX_ROWS` rows
    /// at `start_pos`, returning last-position logits. All linears
    /// run as M-row HVX GEMMs (weights stream once per chunk); attention
    /// uses multi-query flash attention over the valid prefix with a
    /// host-built causal mask; conv layers use one SsmConv per chunk.
    fn try_forward_prefill_chunk_input(
        &self,
        input: PrefillInput<'_>,
        start_pos: usize,
        state: &mut InferenceState,
        all_logits: bool,
    ) -> Result<Vec<f32>, CeraError> {
        let hs = self.config.hidden_size;
        let m = match input {
            PrefillInput::Tokens(tokens) => {
                if tokens.is_empty() {
                    return Err(CeraError::EmptyInput);
                }
                tokens.len()
            }
            PrefillInput::Embeddings(embeddings) => {
                if embeddings.is_empty() {
                    return Err(CeraError::EmptyInput);
                }
                if hs == 0 {
                    return Err(CeraError::Backend("hidden_size is zero".to_string()));
                }
                if embeddings.len() % hs != 0 {
                    return Err(CeraError::Backend(format!(
                        "embeddings length ({}) is not a multiple of hidden_size ({hs})",
                        embeddings.len()
                    )));
                }
                embeddings.len() / hs
            }
        };

        if m > PREFILL_MAX_ROWS {
            return Err(CeraError::Backend(format!(
                "prefill chunk size ({m}) exceeds maximum ({PREFILL_MAX_ROWS})"
            )));
        }
        if all_logits && m > MAX_ALL_LOGITS_TOKENS {
            return Err(CeraError::Backend(format!(
                "all_logits prefill chunk size ({m}) exceeds MAX_ALL_LOGITS_TOKENS ({MAX_ALL_LOGITS_TOKENS})"
            )));
        }
        let fwd_start = std::time::Instant::now();

        let vocab_size = self.config.vocab_size;
        if let PrefillInput::Tokens(tokens) = input {
            let bad_token = tokens.iter().copied().find(|&t| (t as usize) >= vocab_size);
            if let Some(bad) = bad_token {
                return Err(CeraError::Backend(format!(
                    "token ID {bad} exceeds model vocab size {vocab_size}"
                )));
            }
        }

        let max_seq_len = self.config.max_seq_len;
        let kv_len = start_pos + m;
        if kv_len > max_seq_len {
            return Err(CeraError::ContextOverflow {
                max_seq_len: max_seq_len as u32,
                by: (kv_len - max_seq_len) as u32,
            });
        }

        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());

        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;

        // Ingest token embeddings or raw float embeddings, positions, and causal mask.
        unsafe {
            match input {
                PrefillInput::Tokens(tokens) => {
                    for (i, &t) in tokens.iter().enumerate() {
                        std::ptr::copy_nonoverlapping(
                            self.token_embd.as_ptr().add(t as usize * hs),
                            scratch.as_mut_ptr().add(so.activation + i * hs * 4) as *mut f32,
                            hs,
                        );
                    }
                }
                PrefillInput::Embeddings(embeddings) => {
                    std::ptr::copy_nonoverlapping(
                        embeddings.as_ptr() as *const u8,
                        scratch.as_mut_ptr().add(so.activation),
                        m * hs * 4,
                    );
                }
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
                let allowed = (start_pos + mm + 1).min(kv_len);
                row[..allowed].fill(0x0000);
                if allowed < kv_len {
                    row[allowed..].fill(0xFC00);
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
        let rope_theta = self.config.rope_theta;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

        let session = device.queue_session_mut();
        session.drop_pending_batch();

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

        let run_res = (|| -> Result<(), CeraError> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let cur_act = if layer_idx % 2 == 0 {
                    so.activation
                } else {
                    so.activation_b
                };
                let next_act = if layer_idx % 2 == 0 {
                    so.activation_b
                } else {
                    so.activation
                };
                let cur_normed = if layer_idx % 2 == 0 {
                    so.normed
                } else {
                    so.normed_b
                };

                match layer {
                    HexagonLayer::Attention(attn) => {
                        let n_kv_heads = attn.kv_dim / head_dim;
                        let q_dim = attn.q_dim;
                        let kv_dim = attn.kv_dim;
                        // Block norm over M rows.
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            cur_act,
                            &self.weights_buf,
                            attn.attn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            m,
                        )?;
                        self.debug_barrier(session, "prefill attn_norm")?;
                        // QKV projections (fused NX: one shared-activation op).
                        self.dispatch_mul_mat_nx(
                            session,
                            &self.weights_buf,
                            &[&attn.attn_q, &attn.attn_k, &attn.attn_v],
                            scratch,
                            cur_normed,
                            scratch,
                            &[so.q, so.k, so.v],
                            m,
                        )?;
                        self.debug_barrier(session, "prefill QKV")?;
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
                        self.debug_barrier(session, "prefill QK norm")?;
                        // RoPE (DSP, or host loop under the debug fallback).
                        if self.cpu_rope {
                            session.flush().map_err(|e| {
                                CeraError::Backend(format!("prefill pre-RoPE flush failed: {e}"))
                            })?;
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
                                        scratch.as_mut_ptr().add(so.k + mm * kv_dim * 4)
                                            as *mut f32,
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
                            self.debug_barrier(session, "prefill RoPE")?;
                        }
                        // Append M K/V rows (slots == positions: reuse pos vector).
                        Self::dispatch_set_rows_m_typed(
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
                            self.kv_dtype,
                        )?;
                        Self::dispatch_set_rows_m_typed(
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
                            self.kv_dtype,
                        )?;
                        self.debug_barrier(session, "prefill SetRows")?;
                        self.dump_hidden(
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
                        self.debug_barrier(session, "prefill FlashAttn")?;
                        self.dump_hidden(
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
                            cur_normed,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(attn) prefill attn-out",
                            cur_normed + (m - 1) * hs * 4,
                            hs,
                        );
                        Self::dispatch_add_m(
                            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(attn) prefill block-out",
                            next_act + (m - 1) * hs * 4,
                            hs,
                        );
                        // FFN.
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            next_act,
                            &self.weights_buf,
                            attn.ffn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            m,
                        )?;
                        self.dispatch_mul_mat_nx(
                            session,
                            &self.weights_buf,
                            &[&attn.ffn_gate, &attn.ffn_up],
                            scratch,
                            cur_normed,
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
                            cur_normed,
                            m,
                        )?;
                        Self::dispatch_add_m(
                            session, scratch, next_act, scratch, cur_normed, scratch, next_act, hs,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(attn) prefill post-ffn",
                            next_act + (m - 1) * hs * 4,
                            hs,
                        );
                    }
                    HexagonLayer::Conv(conv) => {
                        // Block norm + in_proj over M rows.
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            cur_act,
                            &self.weights_buf,
                            conv.attn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            m,
                        )?;
                        self.dispatch_mul_mat_m(
                            session,
                            &self.weights_buf,
                            &conv.in_proj,
                            scratch,
                            cur_normed,
                            scratch,
                            so.conv_in,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) prefill conv_in r0",
                            so.conv_in,
                            3 * hs,
                        );
                        if m > 1 {
                            self.dump_hidden(
                                session,
                                scratch,
                                layer_idx,
                                "(conv) prefill conv_in r1",
                                so.conv_in + 3 * hs * 4,
                                3 * hs,
                            );
                        }
                        self.debug_barrier(session, "prefill conv/in-proj")?;
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
                        // hs]) into conv_x `[ncs, hs]`, time-inner: one op
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
                        self.debug_barrier(session, "prefill conv/scatter")?;
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
                        self.debug_barrier(session, "prefill conv/ssm-only")?;
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
                        self.debug_barrier(session, "prefill conv/ssm")?;
                        // out_proj + residual.
                        self.dispatch_mul_mat_m(
                            session,
                            &self.weights_buf,
                            &conv.out_proj,
                            scratch,
                            so.conv_ssm_y,
                            scratch,
                            cur_normed,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) prefill conv_out r0",
                            cur_normed,
                            hs,
                        );
                        if m > 1 {
                            self.dump_hidden(
                                session,
                                scratch,
                                layer_idx,
                                "(conv) prefill conv_out r1",
                                cur_normed + hs * 4,
                                hs,
                            );
                        }
                        Self::dispatch_add_m(
                            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) prefill block-out",
                            next_act + (m - 1) * hs * 4,
                            hs,
                        );
                        self.debug_barrier(session, "prefill conv/out-proj")?;
                        // FFN (same as attention blocks).
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            next_act,
                            &self.weights_buf,
                            conv.ffn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            m,
                        )?;
                        self.dispatch_mul_mat_nx(
                            session,
                            &self.weights_buf,
                            &[&conv.ffn_gate, &conv.ffn_up],
                            scratch,
                            cur_normed,
                            scratch,
                            &[so.ffn_gate, so.ffn_up],
                            m,
                        )?;
                        self.dump_hidden(
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
                        self.dump_hidden(
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
                            cur_normed,
                            m,
                        )?;
                        Self::dispatch_add_m(
                            session, scratch, next_act, scratch, cur_normed, scratch, next_act, hs,
                            m,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "prefill post-ffn",
                            next_act + (m - 1) * hs * 4,
                            hs,
                        );
                    }
                }
                if self.debug_barriers
                    && let Err(e) = session.flush()
                {
                    return Err(CeraError::Backend(format!(
                        "prefill layer {layer_idx} flush failed: {e}"
                    )));
                }
            }

            let final_act = if self.layers.len().is_multiple_of(2) {
                so.activation
            } else {
                so.activation_b
            };
            let final_normed = if self.layers.len().is_multiple_of(2) {
                so.normed
            } else {
                so.normed_b
            };

            // Final norm + LM head
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                final_act,
                &self.weights_buf,
                self.output_norm_offset,
                scratch,
                final_normed,
                eps,
                hs,
                m,
            )?;
            if all_logits && m > 1 {
                self.dispatch_mul_mat_m(
                    session,
                    &self.weights_buf,
                    &self.lm_head,
                    scratch,
                    final_normed,
                    scratch,
                    so.logits,
                    m,
                )?;
            } else if all_logits {
                self.dispatch_mul_mat(
                    session,
                    &self.weights_buf,
                    &self.lm_head,
                    scratch,
                    final_normed,
                    scratch,
                    so.logits,
                )?;
            } else {
                self.dispatch_mul_mat(
                    session,
                    &self.weights_buf,
                    &self.lm_head,
                    scratch,
                    final_normed + (m - 1) * hs * 4,
                    scratch,
                    so.logits,
                )?;
            }

            session.flush().map_err(|e| {
                CeraError::Backend(format!("Hexagon NPU prefill execution failed: {e}"))
            })?;
            Ok(())
        })();

        session.set_max_ops_per_flush(None);
        if let Err(e) = run_res {
            session.drop_pending_batch();
            return Err(e);
        }

        self.current_seq_len.store(start_pos + m, Ordering::SeqCst);
        state.seq_len = start_pos + m;

        let n_out_tokens = if all_logits { m } else { 1 };
        scratch.invalidate_cpu_cache(so.logits, n_out_tokens * vocab_size * 4);
        let logits_slice = unsafe {
            std::slice::from_raw_parts(
                scratch.as_ptr().add(so.logits) as *const f32,
                n_out_tokens * vocab_size,
            )
        };
        if let Ok(mut adpf) = self.adpf.lock()
            && let Some(session) = adpf.as_mut()
        {
            let per_token_ns =
                (fwd_start.elapsed().as_nanos() / m.max(1) as u128).min(i64::MAX as u128) as i64;
            session.report(per_token_ns);
        }
        Ok(logits_slice.to_vec())
    }

    fn try_forward_prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        self.try_forward_prefill_chunk_input(PrefillInput::Tokens(tokens), start_pos, state, false)
    }

    fn try_forward_prefill_chunk_from_embeddings(
        &self,
        embeddings: &[f32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        self.try_forward_prefill_chunk_input(
            PrefillInput::Embeddings(embeddings),
            start_pos,
            state,
            false,
        )
    }

    fn try_forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        self.try_forward_prefill_chunk_input(PrefillInput::Tokens(tokens), start_pos, state, true)
    }

    /// Run one prefill chunk, or `None` when the chunk failed. A failed
    /// chunk aborts the whole prefill (see `forward_prefill`): continuing
    /// would write later chunks' KV slots over the failed chunk's hole,
    /// leaving a KV-timeline gap behind plausible-looking logits. The
    /// `eprintln!` pairs the structured log because no `tracing`
    /// subscriber exists on the shipping NPU platforms (Android/iOS).
    fn forward_prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Option<Vec<f32>> {
        match self.try_forward_prefill_chunk(tokens, start_pos, state) {
            Ok(logits) => Some(logits),
            Err(e) => {
                tracing::error!("Hexagon NPU prefill chunk failed: {e}");
                eprintln!("[cera-hexagon] prefill chunk failed, aborting prefill: {e}");
                // Record first-wins for `take_decode_error` (same slot as
                // `forward`): the session drain then surfaces the DSP root
                // cause instead of its generic short-prefill message.
                record_first_fault(&self.decode_error, e);
                None
            }
        }
    }

    fn forward_prefill_chunk_from_embeddings(
        &self,
        embeddings: &[f32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Option<Vec<f32>> {
        match self.try_forward_prefill_chunk_from_embeddings(embeddings, start_pos, state) {
            Ok(logits) => Some(logits),
            Err(e) => {
                tracing::error!("Hexagon NPU prefill chunk from embeddings failed: {e}");
                eprintln!(
                    "[cera-hexagon] prefill chunk from embeddings failed, aborting prefill: {e}"
                );
                record_first_fault(&self.decode_error, e);
                None
            }
        }
    }

    fn try_forward_input(
        &self,
        input: DecodeInput<'_>,
        output: DecodeOutput,
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<DecodeResult, CeraError> {
        let fwd_start = std::time::Instant::now();
        let vocab_size = self.config.vocab_size;
        let hs = self.config.hidden_size;

        let max_seq = self.config.max_seq_len;
        if pos >= max_seq {
            return Err(CeraError::ContextOverflow {
                max_seq_len: max_seq as u32,
                by: (pos + 1 - max_seq) as u32,
            });
        }

        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());

        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;

        match input {
            DecodeInput::Token(token_id) => {
                let token = token_id as usize;
                if token >= vocab_size {
                    return Err(CeraError::Backend(format!(
                        "token ID {token} exceeds model vocab size {vocab_size}"
                    )));
                }
                let embd_start = token * hs;
                let embd_slice = &self.token_embd[embd_start..embd_start + hs];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        embd_slice.as_ptr() as *const u8,
                        scratch.as_mut_ptr().add(so.activation),
                        hs * 4,
                    );
                }
            }
            DecodeInput::Embedding(embedding) => {
                if embedding.len() != hs {
                    return Err(CeraError::Backend(format!(
                        "embedding length ({}) != hidden_size ({hs})",
                        embedding.len()
                    )));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        embedding.as_ptr() as *const u8,
                        scratch.as_mut_ptr().add(so.activation),
                        hs * 4,
                    );
                }
            }
        }

        unsafe {
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
        let vocab_size = self.config.vocab_size;

        let session = device.queue_session_mut();
        session.drop_pending_batch();

        let can_use_template = !self.cpu_rope
            && !self.debug_barriers
            && !self.dump_act
            && decode_ops_cap().is_none()
            && (output == DecodeOutput::Logits || output == DecodeOutput::Greedy);

        let final_normed = if self.layers.len().is_multiple_of(2) {
            so.normed
        } else {
            so.normed_b
        };

        if can_use_template {
            let template_slot = if output == DecodeOutput::Greedy {
                &self.greedy_decode_template
            } else {
                &self.decode_template
            };
            let mut guard = template_slot.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(tpl) = guard.as_mut() {
                let seq_len = pos + 1;
                let n_kv_blocks = seq_len.div_ceil(64).max(1) as u32;
                let mask_bytes = (seq_len * 2) as u32;
                for patch in &tpl.flash_attn_patches {
                    let k_ten = tpl.staged.tensor_mut(patch.k_ti);
                    k_ten.ne[1] = seq_len as u32;

                    let v_ten = tpl.staged.tensor_mut(patch.v_ti);
                    v_ten.ne[1] = seq_len as u32;

                    let mask_ten = tpl.staged.tensor_mut(patch.mask_ti);
                    mask_ten.size = mask_bytes;
                    mask_ten.ne[0] = seq_len as u32;
                    mask_ten.nb[1] = mask_bytes;
                    mask_ten.nb[2] = mask_bytes;
                    mask_ten.nb[3] = mask_bytes;

                    let b2 = (n_kv_blocks & 0xffff) | ((patch.g as u32 & 0xffff) << 16);
                    tpl.staged.op_mut(patch.op_idx).kernel_params[2] = b2 as i32;
                }
                session
                    .flush_staged_resident(tpl.resident_id, &tpl.staged, &tpl.patch_ranges)
                    .map_err(|e| {
                        CeraError::Backend(format!("Hexagon NPU execution failed: {e}"))
                    })?;

                self.current_seq_len.store(pos + 1, Ordering::SeqCst);
                state.seq_len = pos + 1;

                if let Ok(mut adpf) = self.adpf.lock()
                    && let Some(session) = adpf.as_mut()
                {
                    session.report(fwd_start.elapsed().as_nanos().min(i64::MAX as u128) as i64);
                }

                return match output {
                    DecodeOutput::Greedy => {
                        scratch.invalidate_cpu_cache(so.argmax, 4);
                        let token = unsafe { *(scratch.as_ptr().add(so.argmax) as *const u32) };
                        Ok(DecodeResult::Greedy(token))
                    }
                    DecodeOutput::Logits => {
                        scratch.invalidate_cpu_cache(so.logits, vocab_size * 4);
                        let logits_slice = unsafe {
                            std::slice::from_raw_parts(
                                scratch.as_ptr().add(so.logits) as *const f32,
                                vocab_size,
                            )
                        };
                        Ok(DecodeResult::Logits(logits_slice.to_vec()))
                    }
                    DecodeOutput::Hidden => unreachable!(),
                };
            }
        }

        // Decode determinism: cap ops per flush if configured.
        session.set_max_ops_per_flush(decode_ops_cap());

        let mut flash_attn_patches = Vec::new();

        let run_res = (|| -> Result<(), CeraError> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let cur_act = if layer_idx % 2 == 0 {
                    so.activation
                } else {
                    so.activation_b
                };
                let next_act = if layer_idx % 2 == 0 {
                    so.activation_b
                } else {
                    so.activation
                };
                let cur_normed = if layer_idx % 2 == 0 {
                    so.normed
                } else {
                    so.normed_b
                };

                match layer {
                    HexagonLayer::Attention(attn) => {
                        // Attention RMS norm (fused): normed = rmsnorm(act) * attn_norm
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            cur_act,
                            &self.weights_buf,
                            attn.attn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            1,
                        )?;
                        self.debug_barrier(session, "Attention attn_norm")?;

                        // Projections: Q, K, V (fused NX).
                        self.dispatch_mul_mat_nx(
                            session,
                            &self.weights_buf,
                            &[&attn.attn_q, &attn.attn_k, &attn.attn_v],
                            scratch,
                            cur_normed,
                            scratch,
                            &[so.q, so.k, so.v],
                            1,
                        )?;
                        self.debug_barrier(session, "Attention QKV proj")?;

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
                        self.debug_barrier(session, "Attention QK norm")?;

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
                            self.debug_barrier(session, "Attention RoPE")?;
                        }

                        // SetRows K and V into KV cache
                        Self::dispatch_set_rows_typed(
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
                            self.kv_dtype,
                        )?;
                        Self::dispatch_set_rows_typed(
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
                            self.kv_dtype,
                        )?;
                        self.debug_barrier(session, "Attention SetRows")?;

                        // Flash Attention
                        let op_idx = session.ops_len();
                        let (k_ti, v_ti, mask_ti) = Self::dispatch_flash_attn_ext_typed(
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
                            self.kv_dtype,
                        )?;
                        if can_use_template {
                            flash_attn_patches.push(FlashAttnPatch {
                                op_idx,
                                k_ti,
                                v_ti,
                                mask_ti,
                                g: (n_heads / n_kv_heads.max(1)).max(1),
                            });
                        }
                        self.debug_barrier(
                            session,
                            &format!("Attention layer {layer_idx} FlashAttnExt"),
                        )?;

                        // Attention output projection
                        self.dispatch_mul_mat(
                            session,
                            &self.weights_buf,
                            &attn.attn_output,
                            scratch,
                            so.attn_out,
                            scratch,
                            cur_normed,
                        )?;

                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(attn) block-out",
                            cur_normed,
                            hs,
                        );

                        // Residual add: next_act = cur_act + cur_normed
                        Self::dispatch_add(
                            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(attn) post-block",
                            next_act,
                            hs,
                        );

                        // FFN
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            next_act,
                            &self.weights_buf,
                            attn.ffn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            1,
                        )?;
                        self.dispatch_mul_mat_nx(
                            session,
                            &self.weights_buf,
                            &[&attn.ffn_gate, &attn.ffn_up],
                            scratch,
                            cur_normed,
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
                            cur_normed,
                        )?;
                        self.dump_hidden(session, scratch, layer_idx, "ffn-out", cur_normed, hs);
                        Self::dispatch_add(
                            session, scratch, next_act, scratch, cur_normed, scratch, next_act, hs,
                        )?;
                    }
                    HexagonLayer::Conv(conv) => {
                        // Conv RMS norm
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            cur_act,
                            &self.weights_buf,
                            conv.attn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            1,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) normed-act",
                            cur_normed,
                            hs,
                        );

                        // in_proj: hs -> 3 * hs (b, c, x)
                        self.dispatch_mul_mat(
                            session,
                            &self.weights_buf,
                            &conv.in_proj,
                            scratch,
                            cur_normed,
                            scratch,
                            so.conv_in,
                        )?;
                        self.dump_hidden(
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
                            self.dump_hidden(
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
                            self.dump_hidden(
                                session,
                                scratch,
                                layer_idx,
                                "(conv) bx",
                                so.conv_bx,
                                hs,
                            );

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
                                session, scratch, so.conv_t0, scratch, so.conv_t1, scratch,
                                so.conv_t0, hs,
                            )?;
                            Self::dispatch_add(
                                session, scratch, so.conv_y, scratch, so.conv_t0, scratch,
                                so.conv_y, hs,
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
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) gated-y",
                            so.conv_y,
                            hs,
                        );

                        // out_proj: hs -> hs
                        self.dispatch_mul_mat(
                            session,
                            &self.weights_buf,
                            &conv.out_proj,
                            scratch,
                            so.conv_y,
                            scratch,
                            cur_normed,
                        )?;

                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) block-out",
                            cur_normed,
                            hs,
                        );

                        // Residual add: next_act = cur_act + cur_normed
                        Self::dispatch_add(
                            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) post-block",
                            next_act,
                            hs,
                        );

                        // FFN
                        Self::dispatch_rms_norm_mul(
                            session,
                            scratch,
                            next_act,
                            &self.weights_buf,
                            conv.ffn_norm_offset,
                            scratch,
                            cur_normed,
                            eps,
                            hs,
                            1,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) normed-ffn",
                            cur_normed,
                            hs,
                        );
                        self.dispatch_mul_mat_nx(
                            session,
                            &self.weights_buf,
                            &[&conv.ffn_gate, &conv.ffn_up],
                            scratch,
                            cur_normed,
                            scratch,
                            &[so.ffn_gate, so.ffn_up],
                            1,
                        )?;
                        self.dump_hidden(
                            session,
                            scratch,
                            layer_idx,
                            "(conv) ffn-gate",
                            so.ffn_gate,
                            intermediate_size,
                        );
                        self.dump_hidden(
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
                        self.dump_hidden(
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
                            cur_normed,
                        )?;
                        self.dump_hidden(session, scratch, layer_idx, "ffn-out", cur_normed, hs);
                        Self::dispatch_add(
                            session, scratch, next_act, scratch, cur_normed, scratch, next_act, hs,
                        )?;
                    }
                }

                self.debug_barrier(session, &format!("Hexagon NPU layer {layer_idx}"))?;

                self.dump_hidden(session, scratch, layer_idx, "post-ffn", next_act, hs);
            }

            let final_act = if self.layers.len().is_multiple_of(2) {
                so.activation
            } else {
                so.activation_b
            };

            // Final output norm (fused): final_normed = rmsnorm(final_act) * output_norm
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                final_act,
                &self.weights_buf,
                self.output_norm_offset,
                scratch,
                final_normed,
                eps,
                hs,
                1,
            )?;

            if output == DecodeOutput::Logits || output == DecodeOutput::Greedy {
                // Final LM head: logits = mul_mat(lm_head, final_normed)
                self.dispatch_mul_mat(
                    session,
                    &self.weights_buf,
                    &self.lm_head,
                    scratch,
                    final_normed,
                    scratch,
                    so.logits,
                )?;
                if output == DecodeOutput::Greedy {
                    Self::dispatch_argmax(
                        session, scratch, so.logits, scratch, so.argmax, vocab_size, 1,
                    )?;
                }
            }

            if can_use_template {
                let staged = session.export_staged_batch()?;
                let (template_slot, resident_id) = if output == DecodeOutput::Greedy {
                    (&self.greedy_decode_template, 2u64)
                } else {
                    (&self.decode_template, 1u64)
                };
                let tensor_sz = std::mem::size_of::<crate::backend::hexagon::HtpTensor>();
                let op_sz = std::mem::size_of::<crate::backend::hexagon::HtpOpDesc>();
                let mut patch_ranges = Vec::with_capacity(flash_attn_patches.len() * 4);
                for patch in &flash_attn_patches {
                    let k_off = staged.bufs_bytes + patch.k_ti * tensor_sz;
                    patch_ranges.push(k_off..k_off + tensor_sz);
                    let v_off = staged.bufs_bytes + patch.v_ti * tensor_sz;
                    patch_ranges.push(v_off..v_off + tensor_sz);
                    let m_off = staged.bufs_bytes + patch.mask_ti * tensor_sz;
                    patch_ranges.push(m_off..m_off + tensor_sz);
                    let op_off = staged.bufs_bytes + staged.tens_bytes + patch.op_idx * op_sz;
                    patch_ranges.push(op_off..op_off + op_sz);
                }
                let mut guard = template_slot.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(DecodeTemplate {
                    resident_id,
                    staged,
                    flash_attn_patches,
                    patch_ranges,
                });
            }

            // Flush remaining queued operations to DSP and await execution completion.
            session
                .flush()
                .map_err(|e| CeraError::Backend(format!("Hexagon NPU execution failed: {e}")))?;
            Ok(())
        })();

        session.set_max_ops_per_flush(None);
        if let Err(e) = run_res {
            session.drop_pending_batch();
            return Err(e);
        }

        self.current_seq_len.store(pos + 1, Ordering::SeqCst);
        state.seq_len = pos + 1;

        let result = match output {
            DecodeOutput::Greedy => {
                scratch.invalidate_cpu_cache(so.argmax, 4);
                let token = unsafe { *(scratch.as_ptr().add(so.argmax) as *const u32) };
                DecodeResult::Greedy(token)
            }
            DecodeOutput::Logits => {
                scratch.invalidate_cpu_cache(so.logits, vocab_size * 4);
                let logits_slice = unsafe {
                    std::slice::from_raw_parts(
                        scratch.as_ptr().add(so.logits) as *const f32,
                        vocab_size,
                    )
                };
                DecodeResult::Logits(logits_slice.to_vec())
            }
            DecodeOutput::Hidden => {
                scratch.invalidate_cpu_cache(final_normed, hs * 4);
                let hidden_slice = unsafe {
                    std::slice::from_raw_parts(scratch.as_ptr().add(final_normed) as *const f32, hs)
                };
                DecodeResult::Hidden(hidden_slice.to_vec())
            }
        };

        if let Ok(mut adpf) = self.adpf.lock()
            && let Some(session) = adpf.as_mut()
        {
            session.report(fwd_start.elapsed().as_nanos().min(i64::MAX as u128) as i64);
        }
        Ok(result)
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
        if tokens.len() > 1 {
            self.forward_prefill(&tokens[..tokens.len() - 1], pos, state);
            return self.try_forward(&tokens[tokens.len() - 1..], pos + tokens.len() - 1, state);
        }
        match self.try_forward_input(
            DecodeInput::Token(tokens[0]),
            DecodeOutput::Logits,
            pos,
            state,
        )? {
            DecodeResult::Logits(logits) => Ok(logits),
            _ => unreachable!(),
        }
    }

    fn try_forward_greedy(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<u32, CeraError> {
        if tokens.is_empty() {
            return Ok(0);
        }
        if tokens.len() > 1 {
            self.forward_prefill(&tokens[..tokens.len() - 1], pos, state);
            return self.try_forward_greedy(
                &tokens[tokens.len() - 1..],
                pos + tokens.len() - 1,
                state,
            );
        }
        match self.try_forward_input(
            DecodeInput::Token(tokens[0]),
            DecodeOutput::Greedy,
            pos,
            state,
        )? {
            DecodeResult::Greedy(token) => Ok(token),
            _ => unreachable!(),
        }
    }

    fn try_forward_from_embedding(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        match self.try_forward_input(
            DecodeInput::Embedding(embedding),
            DecodeOutput::Logits,
            pos,
            state,
        )? {
            DecodeResult::Logits(logits) => Ok(logits),
            _ => unreachable!(),
        }
    }

    fn try_forward_embedding(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        if tokens.is_empty() {
            return Err(CeraError::EmptyInput);
        }
        match self.try_forward_input(
            DecodeInput::Token(tokens[0]),
            DecodeOutput::Hidden,
            pos,
            state,
        )? {
            DecodeResult::Hidden(hidden) => Ok(hidden),
            _ => unreachable!(),
        }
    }

    fn try_forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        match self.try_forward_input(
            DecodeInput::Embedding(embedding),
            DecodeOutput::Hidden,
            pos,
            state,
        )? {
            DecodeResult::Hidden(hidden) => Ok(hidden),
            _ => unreachable!(),
        }
    }
}

/// Run prefill in scratch-capacity chunks, aborting at the first failed
/// chunk. Returns `(consumed, last_logits)`: `consumed` is the prefix
/// length actually written to KV (the whole input on success), and
/// `last_logits` is `Some` iff at least one chunk succeeded. Later chunks
/// must not run past a failure: they would append their KV over the
/// failed chunk's missing rows. The caller decides the failure signal:
/// [`Model::forward_prefill`] maps a short run to zeros (its signature has
/// no error channel), while the [`Model::forward_prefill_chunked`]
/// override below reports the short `consumed` so the session advances
/// `current_pos` exactly over the KV that exists instead of over the
/// failed suffix. Split out of `Model::forward_prefill` so the abort
/// policy is host-testable without a DSP session.
fn run_scratch_chunks(
    tokens: &[u32],
    start_pos: usize,
    state: &mut InferenceState,
    mut runner: impl FnMut(&[u32], usize, &mut InferenceState) -> Option<Vec<f32>>,
) -> (usize, Option<Vec<f32>>) {
    if tokens.is_empty() {
        return (0, None);
    }
    // Chunk to scratch capacity.
    let mut consumed = 0usize;
    let mut logits = None;
    for (chunk_idx, chunk) in tokens.chunks(PREFILL_MAX_ROWS).enumerate() {
        match runner(chunk, start_pos + chunk_idx * PREFILL_MAX_ROWS, state) {
            Some(logits_out) => {
                consumed += chunk.len();
                logits = Some(logits_out);
            }
            None => break,
        }
    }
    (consumed, logits)
}

/// Map a [`run_scratch_chunks`] result to the bare logits
/// `Model::forward_prefill` returns. No error channel here (the trait
/// returns bare logits), so a short run maps to zeros: the abort site
/// already logged the cause on stderr, which is the loud half of the
/// signal; the zeros keep a direct caller from sampling stale last-good
/// logits over a KV hole. A full run returns the last chunk's logits
/// (zeros only when no chunk ran, i.e. empty input). Pure so the caller
/// side of the abort policy is unit-testable.
fn prefill_tail_logits(
    consumed: usize,
    total: usize,
    logits: Option<Vec<f32>>,
    vocab_size: usize,
) -> Vec<f32> {
    if consumed == total {
        logits.unwrap_or_else(|| vec![0.0f32; vocab_size])
    } else {
        vec![0.0f32; vocab_size]
    }
}

fn run_scratch_chunks_embeddings(
    embeddings: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    start_pos: usize,
    state: &mut InferenceState,
    mut runner: impl FnMut(&[f32], usize, &mut InferenceState) -> Option<Vec<f32>>,
) -> (usize, Option<Vec<f32>>) {
    if n_tokens == 0 || embeddings.is_empty() || hidden_size == 0 {
        return (0, None);
    }
    let chunk_floats = PREFILL_MAX_ROWS * hidden_size;
    let mut consumed = 0usize;
    let mut logits = None;
    for (chunk_idx, chunk) in embeddings.chunks(chunk_floats).enumerate() {
        let chunk_tokens = chunk.len() / hidden_size;
        match runner(chunk, start_pos + chunk_idx * PREFILL_MAX_ROWS, state) {
            Some(logits_out) => {
                consumed += chunk_tokens;
                logits = Some(logits_out);
            }
            None => break,
        }
    }
    (consumed, logits)
}

impl Model for HexagonLfm2Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.session_gate.try_acquire().map(Some)
    }

    fn supports_embedding_input(&self) -> bool {
        true
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        match self.try_forward_from_embedding(embedding, pos, state) {
            Ok(logits) => logits,
            Err(e) => {
                tracing::error!("Hexagon NPU decode from embedding failed: {e}");
                eprintln!(
                    "[cera-hexagon] decode from embedding failed, returning zero logits: {e}"
                );
                record_first_fault(&self.decode_error, e);
                vec![0.0f32; self.config.vocab_size]
            }
        }
    }

    fn forward_embedding(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        match self.try_forward_embedding(tokens, pos, state) {
            Ok(hidden) => hidden,
            Err(e) => {
                tracing::error!("Hexagon NPU forward embedding failed: {e}");
                eprintln!("[cera-hexagon] forward embedding failed, returning zero hidden: {e}");
                record_first_fault(&self.decode_error, e);
                vec![0.0f32; self.config.hidden_size]
            }
        }
    }

    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        match self.try_forward_hidden_from_embedding(embedding, pos, state) {
            Ok(hidden) => hidden,
            Err(e) => {
                tracing::error!("Hexagon NPU forward hidden from embedding failed: {e}");
                eprintln!(
                    "[cera-hexagon] forward hidden from embedding failed, returning zero hidden: {e}"
                );
                record_first_fault(&self.decode_error, e);
                vec![0.0f32; self.config.hidden_size]
            }
        }
    }

    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let hs = self.config.hidden_size;
        assert!(
            n_tokens > 0,
            "forward_prefill_from_embeddings requires at least one frame"
        );
        assert_eq!(
            embeddings.len(),
            n_tokens * hs,
            "embeddings.len() ({}) != n_tokens ({n_tokens}) * hidden_size ({hs})",
            embeddings.len(),
        );
        let (consumed, logits) = run_scratch_chunks_embeddings(
            embeddings,
            n_tokens,
            hs,
            start_pos,
            state,
            |chunk, pos, state| self.forward_prefill_chunk_from_embeddings(chunk, pos, state),
        );
        prefill_tail_logits(consumed, n_tokens, logits, self.config.vocab_size)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        match self.try_forward(tokens, pos, state) {
            Ok(logits) => logits,
            Err(e) => {
                tracing::error!("Hexagon NPU decode failed: {e}");
                // No `tracing` subscriber on the shipping NPU platforms; without
                // this the failure is zero logits with zero record.
                eprintln!("[cera-hexagon] decode failed, returning zero logits: {e}");
                // Record for `take_decode_error`: the session fails the
                // generation on this instead of sampling the zeros below
                // as token 0. Sticky until taken (see the trait docs).
                // First fault wins: a multi-chunk prefill can fail more
                // than once per take, and the surfaced error should name
                // the root cause (the `eprintln` above keeps full order).
                record_first_fault(&self.decode_error, e);
                vec![0.0f32; self.config.vocab_size]
            }
        }
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        match self.try_forward_greedy(tokens, pos, state) {
            Ok(token) => token,
            Err(e) => {
                tracing::error!("Hexagon NPU greedy decode failed: {e}");
                eprintln!("[cera-hexagon] greedy decode failed, returning token 0: {e}");
                record_first_fault(&self.decode_error, e);
                0
            }
        }
    }

    fn take_decode_error(&self) -> Option<CeraError> {
        take_fault(&self.decode_error)
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        // The session path below recovers exact accounting through
        // `forward_prefill_chunked`.
        let (consumed, logits) =
            run_scratch_chunks(tokens, start_pos, state, |chunk, pos, state| {
                self.forward_prefill_chunk(chunk, pos, state)
            });
        prefill_tail_logits(consumed, tokens.len(), logits, self.config.vocab_size)
    }

    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        ubatch: usize,
        cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        // The shared loop (`model::run_chunked_prefill`): the fallible
        // per-chunk forward is a scratch-capacity `run_scratch_chunks` call.
        super::run_chunked_prefill(tokens, start_pos, ubatch, cancel, |chunk, pos| {
            run_scratch_chunks(chunk, pos, state, |c, p, s| {
                self.forward_prefill_chunk(c, p, s)
            })
        })
    }

    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        if tokens.is_empty() {
            return Vec::new();
        }
        assert!(
            tokens.len() <= MAX_ALL_LOGITS_TOKENS,
            "forward_prefill_logits_all token count ({}) exceeds MAX_ALL_LOGITS_TOKENS ({MAX_ALL_LOGITS_TOKENS})",
            tokens.len()
        );
        match self.try_forward_prefill_logits_all(tokens, start_pos, state) {
            Ok(logits) => logits,
            Err(e) => {
                tracing::error!("Hexagon NPU forward_prefill_logits_all failed: {e}");
                eprintln!("[cera-hexagon] forward_prefill_logits_all failed: {e}");
                record_first_fault(&self.decode_error, e);
                vec![0.0f32; tokens.len() * self.config.vocab_size]
            }
        }
    }

    fn supports_all_logits(&self) -> bool {
        true
    }

    fn check_kv_rewind(
        &self,
        state: &InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        if len > state.seq_len {
            return Err(crate::kv_cache::KvRewindError::OutOfBounds {
                requested: len,
                current: state.seq_len,
            });
        }
        Ok(())
    }

    fn try_truncate_kv(
        &self,
        state: &mut InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        self.check_kv_rewind(state, len)?;
        self.truncate_kv(state, len);
        Ok(())
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        assert!(
            len <= state.seq_len,
            "truncate_kv({len}) exceeds seq_len {}",
            state.seq_len
        );
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        *self
            .decode_template
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *self
            .greedy_decode_template
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        self.current_seq_len.store(len, Ordering::SeqCst);
        state.seq_len = len;
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
        *self
            .decode_template
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *self
            .greedy_decode_template
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
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

#[cfg(test)]
mod prefill_chunk_tests {
    use super::*;
    use crate::backend::hexagon::{HtpOpDesc, HtpTensor};
    use crate::model::{ModelConfig, ScalarMultipliers};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: "lfm2".into(),
            n_layers: 1,
            hidden_size: 8,
            intermediate_size: 16,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            vocab_size: 32,
            max_seq_len: 2048,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            block_types: vec![BlockType::Attention],
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer: vec![2],
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        }
    }

    #[test]
    fn failed_chunk_aborts_and_skips_later_chunks() {
        let config = tiny_config();
        let mut state = InferenceState::from_config(&config).unwrap();
        // Three chunks; the scripted runner fails chunk 2.
        let tokens: Vec<u32> = (0..PREFILL_MAX_ROWS * 2 + 7)
            .map(|i| (i % 31 + 1) as u32)
            .collect();
        let mut ran: Vec<(usize, usize)> = Vec::new();
        let (consumed, logits) =
            run_scratch_chunks(&tokens, 0, &mut state, |chunk, pos, _state| {
                ran.push((chunk.len(), pos));
                if ran.len() == 2 {
                    return None;
                }
                Some(vec![1.0f32; config.vocab_size])
            });
        // `consumed` stops at the failed chunk (the session advances over
        // exactly this prefix); the last good logits survive for direct
        // callers that can use them.
        assert_eq!(consumed, PREFILL_MAX_ROWS);
        assert_eq!(logits, Some(vec![1.0f32; config.vocab_size]));
        assert_eq!(ran.len(), 2, "chunk 3 ran after chunk 2 failed: {ran:?}");
        assert_eq!(ran[0], (PREFILL_MAX_ROWS, 0));
        assert_eq!(ran[1], (PREFILL_MAX_ROWS, PREFILL_MAX_ROWS));
    }

    #[test]
    fn all_chunks_ok_returns_last_logits() {
        let config = tiny_config();
        let mut state = InferenceState::from_config(&config).unwrap();
        let tokens: Vec<u32> = (0..PREFILL_MAX_ROWS + 3)
            .map(|i| (i % 31 + 1) as u32)
            .collect();
        let (consumed, logits) =
            run_scratch_chunks(&tokens, 0, &mut state, |_chunk, pos, _state| {
                Some(vec![pos as f32; config.vocab_size])
            });
        // Final chunk's logits win; positions advance per chunk.
        assert_eq!(consumed, tokens.len());
        assert_eq!(
            logits,
            Some(vec![PREFILL_MAX_ROWS as f32; config.vocab_size])
        );
        // Empty prompt: `(0, None)` without invoking the runner.
        let (consumed, logits) = run_scratch_chunks(&[], 0, &mut state, |_, _, _| {
            panic!("runner invoked for empty tokens")
        });
        assert_eq!((consumed, logits), (0, None));
    }

    #[test]
    fn prefill_tail_logits_maps_short_run_to_zeros() {
        // Short run → zeros even when a last-good chunk exists: returning
        // the stale logits would sample over a KV hole. Deleting the `else`
        // leg must fail this test.
        assert_eq!(
            prefill_tail_logits(5, 30, Some(vec![1.0f32; 4]), 4),
            vec![0.0f32; 4]
        );
        // Full run → the last chunk's logits untouched.
        assert_eq!(
            prefill_tail_logits(30, 30, Some(vec![2.0f32; 4]), 4),
            vec![2.0f32; 4]
        );
        // Empty input → zeros (no chunk ran, so `None`).
        assert_eq!(prefill_tail_logits(0, 0, None, 4), vec![0.0f32; 4]);
    }

    #[test]
    fn run_scratch_chunks_embeddings_all_ok() {
        let config = tiny_config();
        let mut state = InferenceState::from_config(&config).unwrap();
        let n_tokens = PREFILL_MAX_ROWS * 2 + 5;
        let hs = config.hidden_size;
        let embeddings: Vec<f32> = (0..n_tokens * hs).map(|i| i as f32).collect();
        let (consumed, logits) = run_scratch_chunks_embeddings(
            &embeddings,
            n_tokens,
            hs,
            0,
            &mut state,
            |_chunk, pos, _state| Some(vec![pos as f32; config.vocab_size]),
        );
        assert_eq!(consumed, n_tokens);
        assert_eq!(
            logits,
            Some(vec![(PREFILL_MAX_ROWS * 2) as f32; config.vocab_size])
        );
    }

    #[test]
    fn run_scratch_chunks_embeddings_aborts_on_failure() {
        let config = tiny_config();
        let mut state = InferenceState::from_config(&config).unwrap();
        let n_tokens = PREFILL_MAX_ROWS * 3;
        let hs = config.hidden_size;
        let embeddings: Vec<f32> = vec![0.5f32; n_tokens * hs];
        let mut ran = 0;
        let (consumed, logits) = run_scratch_chunks_embeddings(
            &embeddings,
            n_tokens,
            hs,
            0,
            &mut state,
            |_chunk, _pos, _state| {
                ran += 1;
                if ran == 2 {
                    return None;
                }
                Some(vec![1.0f32; config.vocab_size])
            },
        );
        assert_eq!(consumed, PREFILL_MAX_ROWS);
        assert_eq!(ran, 2);
        assert_eq!(logits, Some(vec![1.0f32; config.vocab_size]));
    }

    #[test]
    fn run_scratch_chunks_embeddings_empty() {
        let config = tiny_config();
        let mut state = InferenceState::from_config(&config).unwrap();
        let (consumed, logits) =
            run_scratch_chunks_embeddings(&[], 0, config.hidden_size, 0, &mut state, |_, _, _| {
                panic!("should not run")
            });
        assert_eq!(consumed, 0);
        assert_eq!(logits, None);
    }

    #[test]
    fn test_scratch_offsets_alignment_and_non_overlapping() {
        let offsets = ScratchOffsets::new(1024, 1024, 256, 4096, 32000, 2048);
        let list = [
            ("activation", offsets.activation),
            ("activation_b", offsets.activation_b),
            ("normed", offsets.normed),
            ("normed_b", offsets.normed_b),
            ("q", offsets.q),
            ("k", offsets.k),
            ("v", offsets.v),
            ("attn_out", offsets.attn_out),
            ("conv_in", offsets.conv_in),
            ("conv_bx", offsets.conv_bx),
            ("conv_t0", offsets.conv_t0),
            ("conv_t1", offsets.conv_t1),
            ("conv_y", offsets.conv_y),
            ("conv_x", offsets.conv_x),
            ("conv_ssm_y", offsets.conv_ssm_y),
            ("ffn_gate", offsets.ffn_gate),
            ("ffn_up", offsets.ffn_up),
            ("ffn_out", offsets.ffn_out),
            ("logits", offsets.logits),
            ("argmax", offsets.argmax),
            ("pos", offsets.pos),
            ("mask", offsets.mask),
            ("total_size", offsets.total_size),
        ];

        // All offsets must be 4096-byte aligned.
        for (name, offset) in &list {
            assert_eq!(
                offset % 4096,
                0,
                "offset for {name} ({offset}) must be 4096-byte aligned"
            );
        }

        // Each section strictly proceeds the previous one without overlapping.
        for i in 0..list.len() - 1 {
            assert!(
                list[i].1 < list[i + 1].1,
                "offset for {} ({}) must be strictly less than next offset {} ({})",
                list[i].0,
                list[i].1,
                list[i + 1].0,
                list[i + 1].1
            );
        }
    }

    #[test]
    fn test_kv_q8_0_sizing_and_strides() {
        let max_seq_len: usize = 2048;
        let kv_dim: usize = 256;
        let head_dim: usize = 64;

        // F16 sizing: 2 bytes per element
        let f16_slab_size = max_seq_len * kv_dim * 2;
        // Q8_0 sizing: 34 bytes per 32-element block
        let q8_slab_size = (max_seq_len * kv_dim.div_ceil(32) * 34 + 255) & !255;

        // Q8_0 slab size is roughly ~53% of F16 slab size (34/64 = 53.125%)
        assert!(q8_slab_size < f16_slab_size);
        assert_eq!(q8_slab_size, 557056);
        assert_eq!(f16_slab_size, 1048576);

        // Stride calculations for Q8_0:
        let elem_bytes = 1;
        let row_stride = (kv_dim.div_ceil(32) * 34) as u32;
        let head_stride = (head_dim.div_ceil(32) * 34) as u32;
        assert_eq!(elem_bytes, 1);
        assert_eq!(row_stride, 272);
        assert_eq!(head_stride, 68);
    }

    #[test]
    fn test_decode_template_flash_attn_patching() {
        let mut tens = [
            HtpTensor {
                data: 0,
                size: 0,
                flags: 0,
                dtype: HtpDataType::F16 as u32,
                bi: 0,
                ti: 0,
                ne: [64, 1, 4, 1],
                nb: [2, 128, 512, 512],
            },
            HtpTensor {
                data: 0,
                size: 0,
                flags: 0,
                dtype: HtpDataType::F16 as u32,
                bi: 0,
                ti: 1,
                ne: [64, 1, 4, 1],
                nb: [2, 128, 512, 512],
            },
            HtpTensor {
                data: 0,
                size: 2,
                flags: 0,
                dtype: HtpDataType::F16 as u32,
                bi: 0,
                ti: 2,
                ne: [1, 1, 1, 1],
                nb: [2, 2, 2, 2],
            },
        ];
        let mut ops = [HtpOpDesc {
            opcode: HtpOpCode::FlashAttnExt as u32,
            flags: 0,
            params: [0; 16],
            kernel_params: [0; 32],
            src: [0; 10],
            dst: [0; 4],
            pad: [0; 2],
        }];
        ops[0].kernel_params = build_flash_attn_kernel_params(64, 16, 4, 1, 1, 0.125, 4, true);
        let patch = FlashAttnPatch {
            op_idx: 0,
            k_ti: 0,
            v_ti: 1,
            mask_ti: 2,
            g: 16 / 4,
        };

        // Simulate token step at pos = 15 (seq_len = 16)
        let pos: usize = 15;
        let seq_len = pos + 1;
        tens[patch.k_ti].ne[1] = seq_len as u32;
        tens[patch.v_ti].ne[1] = seq_len as u32;
        let mask_bytes = (seq_len * 2) as u32;
        tens[patch.mask_ti].size = mask_bytes;
        tens[patch.mask_ti].ne[0] = seq_len as u32;
        tens[patch.mask_ti].nb[1] = mask_bytes;
        tens[patch.mask_ti].nb[2] = mask_bytes;
        tens[patch.mask_ti].nb[3] = mask_bytes;

        let n_kv_blocks = seq_len.div_ceil(64).max(1) as u32;
        let b2 = (n_kv_blocks & 0xffff) | ((patch.g as u32 & 0xffff) << 16);
        ops[patch.op_idx].kernel_params[2] = b2 as i32;

        let expected_kparams =
            build_flash_attn_kernel_params(64, 16, 4, 1, seq_len, 0.125, 4, true);

        assert_eq!(tens[0].ne[1], 16);
        assert_eq!(tens[1].ne[1], 16);
        assert_eq!(tens[2].ne[0], 16);
        assert_eq!(tens[2].size, 32);
        assert_eq!(tens[2].nb[1], 32);
        assert_eq!(ops[patch.op_idx].kernel_params, expected_kparams);
    }
}

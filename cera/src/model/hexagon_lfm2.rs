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
use std::sync::{Arc, Mutex, MutexGuard};

use crate::backend::cpu::RopeType;
use crate::backend::hexagon::{
    AdpfSession, HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch,
    HexagonContext, HexagonDevice, HexagonQueueSession, HtpDataType, HtpOpCode, LockOrRecover,
    RpcmemBuffer, StagedBatch, build_binary_kernel_params, build_binary_scalar_kernel_params,
    build_flash_attn_kernel_params_with_softcap, build_hmx_fa_kernel_params_with_softcap,
    build_hmx_mm_kernel_params, build_mul_mat_kernel_params, build_rms_norm_params,
    build_rope_kernel_params, build_rope_params, build_set_rows_kernel_params,
    build_ssm_conv_kernel_params, build_unary_kernel_params, fa_is_hmx_eligible, lock_or_discard,
    mm_hmx_nb1, mm_is_hmx_eligible,
};
use crate::backend::hexagon::{hexagon_error, hexagon_warn};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::gpu_weight_source::GpuWeightSource;
use crate::model::session_gate::{ModelSessionGate, ModelSessionLease};
use crate::model::transformer::{FfnActivation, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, record_first_fault, take_fault};
use crate::session::CeraError;

/// Type alias exposing the generalized Hexagon NPU model engine.
pub type HexagonModel = HexagonLfmModel;

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
struct HexagonStackedWeight {
    /// Paged experts only: which per-layer expert buffer holds the weight
    /// (`offset` is then local to it). `None` = the resident weights buffer.
    group: Option<usize>,
    offset: usize,
    size: usize,
    in_dim: usize,
    out_dim: usize,
    expert_stride: usize,
    n_expert: usize,
    wire_dtype: HtpDataType,
    tile_size: usize,
    block_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct HexagonMoeFfn {
    router: HexagonWeight,
    exp_probs_b_offset: usize,
    gate: HexagonStackedWeight,
    up: HexagonStackedWeight,
    down: HexagonStackedWeight,
    n_expert: usize,
    n_expert_used: usize,
    expert_ff_len: usize,
}

#[derive(Clone, Copy, Debug)]
struct HexagonDenseFfn {
    gate: HexagonWeight,
    up: HexagonWeight,
    down: HexagonWeight,
    /// Offsets of optional F32 projection biases in the weights buffer
    /// (`ffn_gate.bias` / `ffn_up.bias` / `ffn_down.bias`), added after the
    /// matching projection.
    gate_bias: Option<usize>,
    up_bias: Option<usize>,
    down_bias: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
enum HexagonFfn {
    Dense(HexagonDenseFfn),
    Moe(HexagonMoeFfn),
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
    attn_post_norm_offset: Option<usize>,
    ffn_norm_offset: usize,
    ffn: HexagonFfn,
    ffn_post_norm_offset: Option<usize>,
    k_offset: usize,
    v_offset: usize,
    q_dim: usize,
    kv_dim: usize,
    has_q_gate: bool,
    /// Optional F32 Q/K/V projection biases (all three or none, as on CPU).
    qkv_bias: Option<[usize; 3]>,
    /// Optional F32 attention output projection bias.
    out_bias: Option<usize>,
    /// Q/K norm weights span the whole Q/K vector (Olmo 2/3) instead of one head.
    qk_norm_full: bool,
    /// Sliding-window attention layer (window in `DenseSemantics::swa_window`).
    swa: bool,
    /// YaRN RoPE parameters for this layer (`None` = plain RoPE).
    yarn: Option<crate::backend::cpu::YarnParams>,
}

impl HexagonAttentionLayer {
    /// A plain attention layer: no Q/K norms, biases, gate, sliding window or
    /// YaRN. Constructors set the optional fields with struct-update syntax.
    /// (No `Default`: a zeroed weight would silently alias offset 0.)
    fn plain(
        attn_norm_offset: usize,
        [attn_q, attn_k, attn_v, attn_output]: [HexagonWeight; 4],
        ffn_norm_offset: usize,
        ffn: HexagonFfn,
        (k_offset, v_offset): (usize, usize),
        q_dim: usize,
        kv_dim: usize,
    ) -> Self {
        Self {
            attn_norm_offset,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm_offset: None,
            attn_k_norm_offset: None,
            attn_post_norm_offset: None,
            ffn_norm_offset,
            ffn,
            ffn_post_norm_offset: None,
            k_offset,
            v_offset,
            q_dim,
            kv_dim,
            has_q_gate: false,
            qkv_bias: None,
            out_bias: None,
            qk_norm_full: false,
            swa: false,
            yarn: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct HexagonDeltaNetLayer {
    attn_norm_offset: usize,
    wqkv: HexagonWeight,
    wqkv_gate: HexagonWeight,
    ssm_beta: HexagonWeight,
    ssm_alpha: HexagonWeight,
    ssm_conv1d_offset: usize,
    ssm_conv1d_bias_offset: Option<usize>,
    ssm_dt_offset: usize,
    ssm_a_offset: usize,
    ssm_norm_offset: usize,
    ssm_out: HexagonWeight,
    attn_post_norm_offset: Option<usize>,
    ffn_norm_offset: usize,
    ffn: HexagonFfn,
    ffn_post_norm_offset: Option<usize>,
    conv_state_offset: usize,
    ssm_state_offset: usize,
    conv_dim: usize,
    d_conv: usize,
    d_state: usize,
    dt_rank: usize,
    n_group: usize,
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
    ffn: HexagonFfn,
}

enum HexagonLayer {
    Attention(HexagonAttentionLayer),
    Conv(HexagonConvLayer),
    DeltaNet(HexagonDeltaNetLayer),
}

/// Dense-transformer semantics beyond the plain pre-norm RMSNorm block, all
/// identity for LFM2 and Qwen 3.5 (`Default`). Each field mirrors what the CPU
/// reference (`llama.rs` + `transformer::forward_attn_block`) does, so a model
/// that needs it runs the same math on the NPU instead of silently dropping it.
#[derive(Default)]
struct DenseSemantics {
    /// Softmax scale replacing `1/sqrt(head_dim)` (Granite `attention_multiplier`).
    attn_scale: Option<f32>,
    /// Rotated dims per head when RoPE covers only a prefix of the head
    /// (Qwen 3.5 `rope.dimension_count`); host RoPE route, like YaRN.
    rope_dim: Option<usize>,
    /// Offset of an `[hidden_size]` F32 vector filled with the Granite/MiniCPM
    /// residual multiplier; block outputs are multiplied by it (row broadcast)
    /// before each residual add.
    residual_vec_offset: Option<usize>,
    /// `1 / logits_scaling`, applied on the host to the returned logits (positive,
    /// so the on-DSP argmax is unaffected).
    logit_scale: Option<f32>,
    /// Llama-3 `rope_freqs.weight` factors (`head_dim / 2`), used by the host
    /// RoPE route: the DSP rope kernel params carry no frequency factors.
    rope_freqs: Option<Vec<f32>>,
    /// Olmo 2/3 ordering: no block pre-norm; the block output is normed instead.
    post_norm: bool,
    /// Looped architectures (Nanbeige): after every `n` layers the residual
    /// stream is re-normed with the output norm.
    loop_norm_interval: Option<usize>,
    /// Mistral 3 / Ministral 3 attention temperature `(scale, floor_scale)`,
    /// kept only when the context can reach `floor_scale`.
    attn_temp: Option<(f32, usize)>,
    /// Sliding-window size for SWA layers (kept only when smaller than the context).
    swa_window: Option<usize>,
    /// Decode-time `[max_seq]` F16 mask for SWA layers, rewritten per token.
    mask_swa: Option<RpcmemBuffer>,
}

impl DenseSemantics {
    /// True when a step must run on the host CPU between DSP flushes, which
    /// rules out the recorded decode template.
    fn needs_host_step(&self) -> bool {
        self.attn_temp.is_some()
    }
}

/// Token-embedding lookup that dequantizes one row per token straight from the
/// mapped GGUF, instead of keeping a `[vocab, hidden]` f32 table resident
/// (about 512 MiB for a 65k x 2048 vocabulary). `gguf` shares the mapping and
/// is only read through `mmap_data()`, so it never needs the metadata and
/// tensor tables.
struct EmbeddingTable {
    gguf: Arc<GgufFile>,
    wref: WeightRef,
    /// Embedding multiplier applied to every row (`1.0` = none).
    scale: f32,
}

impl EmbeddingTable {
    /// Table over a borrowed GGUF. The mapping is kept alive through
    /// [`GgufFile::mapping_only`], a handle on the same backing bytes with
    /// empty metadata and tensor tables (no map is cloned). Prefer
    /// [`Self::shared`] when the caller owns the file.
    fn new(
        gguf: &GgufFile,
        vocab_size: usize,
        hidden_size: usize,
        scale: f32,
    ) -> Result<Self, CeraError> {
        let wref = Self::resolve(gguf, vocab_size, hidden_size)?;
        Ok(Self {
            gguf: Arc::new(gguf.mapping_only()),
            wref,
            scale,
        })
    }

    /// Table over a GGUF the caller already shares: no copy at all.
    fn shared(
        gguf: &Arc<GgufFile>,
        vocab_size: usize,
        hidden_size: usize,
        scale: f32,
    ) -> Result<Self, CeraError> {
        Ok(Self {
            wref: Self::resolve(gguf, vocab_size, hidden_size)?,
            gguf: Arc::clone(gguf),
            scale,
        })
    }

    fn resolve(
        gguf: &GgufFile,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<WeightRef, CeraError> {
        let wref = crate::model::transformer::resolve_weight(gguf, "token_embd.weight")
            .map_err(|e| CeraError::Backend(format!("missing token_embd.weight: {e}")))?;
        // `dequantize_row_slice` panics on a dtype it has no arm for, so a
        // table in one is refused here, at load, rather than on the first token.
        if !crate::model::transformer::supports_row_dequant(wref.dtype) {
            return Err(CeraError::Backend(format!(
                "token_embd.weight dtype {:?} has no row dequantizer",
                wref.dtype
            )));
        }
        let block = wref.dtype.block_size();
        if wref.k != hidden_size || wref.m < vocab_size || wref.k % block != 0 {
            return Err(CeraError::Backend(format!(
                "token_embd.weight is [{}, {}] ({:?}), expected at least [{hidden_size}, {vocab_size}]",
                wref.k, wref.m, wref.dtype
            )));
        }
        let row_bytes = wref.k / block * wref.dtype.block_bytes();
        if wref.m.checked_mul(row_bytes).is_none_or(|n| n > wref.size) {
            return Err(CeraError::Backend(format!(
                "token_embd.weight data ({} bytes) is shorter than its {} rows",
                wref.size, wref.m
            )));
        }
        Ok(wref)
    }

    /// Dequantize `token`'s row (times the embedding multiplier) into `out`.
    fn row_into(&self, token: usize, out: &mut [f32]) {
        crate::model::transformer::dequantize_row_into(&self.gguf, &self.wref, token, out);
        if self.scale != 1.0 {
            for x in out.iter_mut() {
                *x *= self.scale;
            }
        }
    }
}

/// Environment knobs read at model load, parsed once with one rule set:
/// opt-in knobs are on only for `1` / `true`, default-on knobs are off only for
/// `0` / `false` (case-insensitive, surrounding whitespace ignored). Anything
/// else keeps the default, so `CERA_HEXAGON_HMX=off` does not silently disable.
/// Knobs are captured per model at load, so changing the environment later
/// does not affect a loaded model.
#[derive(Clone, Debug, PartialEq)]
struct HexagonKnobs {
    /// `CERA_HEXAGON_CPU_ROPE` (opt-in): RoPE on the host CPU. Decode runs fully
    /// on the NPU by default (Android demotes background-process CPUs).
    cpu_rope: bool,
    /// `CERA_HEXAGON_BARRIERS` (opt-in): flush the DSP queue after every op
    /// group, the bring-up behavior.
    debug_barriers: bool,
    /// `CERA_DUMP_ACT` (opt-in): log activation RMS per layer.
    dump_act: bool,
    /// `CERA_HEXAGON_ADPF_TARGET_MS`: ADPF target duration (default 10 ms).
    adpf_target_nanos: i64,
    /// `CERA_HEXAGON_SSM_CONV` (default on): fused SsmConv op for short conv.
    use_ssm_conv: bool,
    /// `CERA_HEXAGON_HMX` (default on): HMX kernels for prefill.
    use_hmx: bool,
    /// `CERA_HEXAGON_ARCH`: skeleton architecture override (numeric id).
    arch_override: Option<HexagonArch>,
    /// `CERA_HEXAGON_KV_Q8` (opt-in): Q8_0 KV cache instead of F16.
    kv_q8: bool,
    /// `CERA_HEXAGON_DECODE_OPS`: decode ops-per-flush cap. Unset, `0` or
    /// unparsable means no cap (single-flush decode); a positive N flushes the
    /// queue every N ops. A bring-up and bisection aid, not a safety
    /// threshold: the nondeterminism it once worked around is phase-sensitive
    /// (cap 20 was clean, then adding conv state-copy ops re-phased its
    /// windows back into the race), so any op-count change per layer must
    /// re-verify a chosen cap over long greedy runs.
    decode_ops: Option<usize>,
}

impl HexagonKnobs {
    fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let opt_in = |key: &str| {
            get(key).is_some_and(|v| {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true")
            })
        };
        let default_on = |key: &str| {
            !get(key).is_some_and(|v| {
                let v = v.trim();
                v == "0" || v.eq_ignore_ascii_case("false")
            })
        };
        Self {
            cpu_rope: opt_in("CERA_HEXAGON_CPU_ROPE"),
            debug_barriers: opt_in("CERA_HEXAGON_BARRIERS"),
            dump_act: opt_in("CERA_DUMP_ACT"),
            adpf_target_nanos: get("CERA_HEXAGON_ADPF_TARGET_MS")
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|ms| (ms.saturating_mul(1_000_000)).min(i64::MAX as u64) as i64)
                .unwrap_or(10_000_000),
            use_ssm_conv: default_on("CERA_HEXAGON_SSM_CONV"),
            use_hmx: default_on("CERA_HEXAGON_HMX"),
            arch_override: get("CERA_HEXAGON_ARCH")
                .and_then(|s| s.trim().parse::<u32>().ok())
                .and_then(HexagonArch::from_u32),
            kv_q8: opt_in("CERA_HEXAGON_KV_Q8"),
            decode_ops: get("CERA_HEXAGON_DECODE_OPS")
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&v| v > 0),
        }
    }

    /// KV cache wire type.
    fn kv_dtype(&self) -> HtpDataType {
        if self.kv_q8 {
            HtpDataType::Q8_0
        } else {
            HtpDataType::F16
        }
    }
}

/// The device side every constructor needs, opened once: FastRPC driver, the
/// probed DSP session, the parsed knobs and the ADPF hint session.
struct Backend {
    driver: Arc<crate::backend::hexagon::FastRpcDriver>,
    device: HexagonDevice,
    knobs: HexagonKnobs,
    adpf: Mutex<Option<AdpfSession>>,
}

impl Backend {
    /// Load the FastRPC driver and probe the DSP (honoring the kill switches).
    fn open() -> Result<Self, CeraError> {
        let context = HexagonContext::new().inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonLfmModel", e);
        })?;
        let knobs = HexagonKnobs::from_env();
        let device = crate::backend::hexagon::probe_device(context.driver(), knobs.arch_override)?;
        Ok(Self::with_device(
            Arc::clone(context.driver()),
            device,
            knobs,
        ))
    }

    fn with_device(
        driver: Arc<crate::backend::hexagon::FastRpcDriver>,
        device: HexagonDevice,
        knobs: HexagonKnobs,
    ) -> Self {
        let adpf = Mutex::new(AdpfSession::try_open(knobs.adpf_target_nanos));
        Self {
            driver,
            device,
            knobs,
            adpf,
        }
    }
}

/// The shared device buffers every constructor allocates after sizing its
/// scratch layout.
struct ModelBuffers {
    scratch_buf: RpcmemBuffer,
    /// Flash attention mask: all zeros (decode identity mask), sized for the
    /// full KV cache and shared by every layer.
    mask_buf: RpcmemBuffer,
    /// Sliding-window layers' decode mask, rewritten each token.
    mask_swa: Option<RpcmemBuffer>,
}

impl ModelBuffers {
    fn alloc(
        driver: &Arc<crate::backend::hexagon::FastRpcDriver>,
        scratch_offsets: &ScratchOffsets,
        mask_size: usize,
        swa: bool,
    ) -> Result<Self, CeraError> {
        let scratch_buf = alloc_scratch(driver, scratch_offsets)?;
        let mask_buf = alloc_zeroed_state(driver, mask_size)?;
        let mask_swa = if swa {
            Some(alloc_zeroed_state(driver, mask_size)?)
        } else {
            None
        };
        Ok(Self {
            scratch_buf,
            mask_buf,
            mask_swa,
        })
    }
}

/// The per-model pieces a constructor produces; [`HexagonLfmModel::from_parts`]
/// adds everything the constructors share (session gate, knob-derived flags,
/// counters, decode templates).
struct ModelParts {
    config: ModelConfig,
    token_embd: EmbeddingTable,
    weights_buf: RpcmemBuffer,
    /// Routed-expert buffers paged through the DSP mapping (`None` = the
    /// experts, if any, live in `weights_buf`).
    pager: Option<ExpertPager>,
    layers: Vec<HexagonLayer>,
    output_norm_offset: usize,
    lm_head: HexagonWeight,
    kv_state_buf: RpcmemBuffer,
    scratch_buf: RpcmemBuffer,
    scratch_offsets: ScratchOffsets,
    mask_buf: RpcmemBuffer,
    rope_type: RopeType,
    cpu_rope: bool,
    dense: DenseSemantics,
    activation: FfnActivation,
    attn_logit_softcapping: Option<f32>,
    final_logit_softcapping: Option<f32>,
    has_deltanet: bool,
    kv_dtype: HtpDataType,
}

/// Which pass an attention layer is emitted for.
enum AttnPass<'a> {
    /// One token at `pos`. `patches` collects the flash-attention ops the
    /// recorded decode template re-patches per token (`None` when no template
    /// is being recorded).
    Decode {
        pos: usize,
        patches: Option<&'a mut Vec<FlashAttnPatch>>,
    },
    /// `m` prefill rows starting at `start_pos`.
    Prefill { start_pos: usize, m: usize },
}

/// Per-layer dense-model data the [`GpuWeightSource`] accessors do not carry:
/// owned copies of the CPU model's optional biases and per-layer attention
/// variants, indexed by logical layer.
struct DenseExtras {
    attn_output_bias: Vec<Option<Vec<f32>>>,
    ffn_gate_bias: Vec<Option<Vec<f32>>>,
    ffn_up_bias: Vec<Option<Vec<f32>>>,
    ffn_down_bias: Vec<Option<Vec<f32>>>,
    /// Sliding window size (`None` = full attention on every layer).
    swa_window: Option<usize>,
    layer_swa: Vec<bool>,
    layer_yarn: Vec<Option<crate::backend::cpu::YarnParams>>,
    attn_temp_scale: Option<(f32, usize)>,
}

impl DenseExtras {
    fn from_llama(cpu: &crate::model::llama::LlamaModel) -> Self {
        let n = GpuWeightSource::config(cpu).n_layers;
        let owned =
            |bias: for<'a> fn(&'a crate::model::llama::LlamaModel, usize) -> Option<&'a [f32]>| {
                (0..n)
                    .map(|i| bias(cpu, i).map(<[f32]>::to_vec))
                    .collect::<Vec<_>>()
            };
        Self {
            attn_output_bias: owned(crate::model::llama::LlamaModel::attn_output_bias),
            ffn_gate_bias: owned(crate::model::llama::LlamaModel::ffn_gate_bias),
            ffn_up_bias: owned(crate::model::llama::LlamaModel::ffn_up_bias),
            ffn_down_bias: owned(crate::model::llama::LlamaModel::ffn_down_bias),
            swa_window: cpu.sliding_window(),
            layer_swa: (0..n)
                .map(|i| cpu.layer_sliding_window(i).is_some())
                .collect(),
            layer_yarn: (0..n).map(|i| cpu.layer_yarn(i)).collect(),
            attn_temp_scale: cpu.attn_temp_scale(),
        }
    }
}

/// The unified activation scratch buffer, with the routed-FFN renormalization
/// slots seeded.
fn alloc_scratch(
    driver: &Arc<crate::backend::hexagon::FastRpcDriver>,
    so: &ScratchOffsets,
) -> Result<RpcmemBuffer, CeraError> {
    let buf = RpcmemBuffer::alloc(Arc::clone(driver), so.total_size, true)?;
    HexagonLfmModel::init_moe_renorm_scratch(&buf, so);
    Ok(buf)
}

/// Whether any layer's FFN is a routed mixture of experts.
fn any_moe(layers: &[HexagonLayer]) -> bool {
    layers.iter().any(|l| match l {
        HexagonLayer::Attention(a) => matches!(a.ffn, HexagonFfn::Moe(_)),
        HexagonLayer::Conv(c) => matches!(c.ffn, HexagonFfn::Moe(_)),
        HexagonLayer::DeltaNet(d) => matches!(d.ffn, HexagonFfn::Moe(_)),
    })
}

/// Smallest divisor the routed-expert weight renormalization allows: f16's
/// smallest positive normal (2^-14), as llama.cpp `build_moe_ffn` and the CPU
/// `select_experts` clamp it.
const MOE_DENOM_FLOOR: f32 = 1.0 / 16384.0;

/// Bytes per token of the renormalization scratch slot: `[sum, floor,
/// floor - sum, relu, denom]` as f32, padded to 64.
const MOE_RENORM_SLOT_BYTES: usize = 64;

/// The renormalization the DSP op chain performs, in the same f32 operation
/// order: `sum = w0 + w1 + ...`, `denom = sum + relu(floor - sum)` (which is
/// `max(sum, floor)` without a Max op), then `w / denom`. Pure so a host test
/// can pin it against the CPU `select_experts`.
#[cfg(test)]
fn moe_renorm_via_op_chain(weights: &[f32]) -> Vec<f32> {
    let mut sum = weights[0];
    for &w in &weights[1..] {
        sum += w;
    }
    let below = (MOE_DENOM_FLOOR - sum).max(0.0);
    let denom = sum + below;
    weights.iter().map(|&w| w / denom).collect()
}

/// F16 `-inf`: the additive attention-mask value for a masked slot.
const MASK_NEG_INF: u16 = 0xFC00;

/// Fill the `[kv_len, m]` prefill attention mask (one `kv_len` row per query).
/// Query `mm` sits at absolute position `start_pos + mm` and attends KV slots
/// `<= start_pos + mm`; with a sliding `window` it attends only the last
/// `window` of them (`slot >= pos + 1 - window`, the CPU `decode_attention` rule).
fn fill_prefill_mask(
    mask: &mut [u16],
    start_pos: usize,
    m: usize,
    kv_len: usize,
    window: Option<usize>,
) {
    debug_assert!(mask.len() >= kv_len * m);
    for (mm, row) in mask[..kv_len * m].chunks_mut(kv_len).enumerate() {
        let allowed = (start_pos + mm + 1).min(kv_len);
        let lo = match window {
            Some(w) if w > 0 => allowed.saturating_sub(w),
            _ => 0,
        };
        row[..lo].fill(MASK_NEG_INF);
        row[lo..allowed].fill(0x0000);
        row[allowed..].fill(MASK_NEG_INF);
    }
}

/// Fill the single-row decode mask for a sliding-window layer: the query at
/// `seq_len - 1` attends the last `window` slots.
fn fill_decode_swa_mask(mask: &mut [u16], seq_len: usize, window: usize) {
    fill_prefill_mask(mask, seq_len - 1, 1, seq_len, Some(window));
}

/// Attention-temperature factor applied to Q after RoPE (Mistral 3 / Llama 4),
/// mirroring `transformer::forward_attn_block`. `None` means no scaling.
fn attn_temp_q_scale(pos: usize, temp: Option<(f32, usize)>) -> Option<f32> {
    let (scale, floor_scale) = temp?;
    if scale > 0.0 && floor_scale > 0 && pos >= floor_scale {
        Some(((pos as f32 / floor_scale as f32).floor() + 1.0).ln() * scale + 1.0)
    } else {
        None
    }
}

/// Host RoPE over one token's Q and K, with the exact routing of the CPU
/// reference (`transformer::forward_attn_block`): YaRN when the layer has it,
/// else plain NEOX, or NORM with the optional Llama-3 frequency factors.
/// Partial rotary (`n_rot < head_dim`) is `llama::rope_partial` itself, which
/// honours neither YaRN nor frequency factors: [`ensure_partial_rope_plain`]
/// rejects such models at load. `scratch` holds its gather buffers.
fn host_rope(
    scratch: &mut crate::model::llama::RopeGather,
    rope_type: RopeType,
    yarn: Option<&crate::backend::cpu::YarnParams>,
    freqs: Option<&[f32]>,
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
) {
    use crate::backend::cpu;
    if n_rot != head_dim {
        // Partial rotary (Qwen 3.5): only the first `n_rot` dims of each head.
        debug_assert!(yarn.is_none() && freqs.is_none());
        crate::model::llama::rope_partial(
            q, k, pos, n_heads, n_kv_heads, head_dim, n_rot, theta, rope_type, scratch,
        );
        return;
    }
    match (rope_type, yarn) {
        (RopeType::Neox, Some(y)) => {
            cpu::rope_neox_yarn(q, k, pos, n_heads, n_kv_heads, head_dim, theta, y)
        }
        (RopeType::Neox, None) => cpu::rope(q, k, pos, n_heads, n_kv_heads, head_dim, theta),
        (RopeType::Norm, Some(y)) => {
            cpu::rope_norm_yarn(q, k, pos, n_heads, n_kv_heads, head_dim, theta, y)
        }
        (RopeType::Norm, None) => {
            cpu::rope_norm(q, k, pos, n_heads, n_kv_heads, head_dim, theta, freqs)
        }
    }
}

/// Partial rotary (`n_rot < head_dim`) runs through `llama::rope_partial`,
/// which ignores YaRN and Llama-3 frequency factors, so a model asking for both
/// would silently rotate wrong. Refuse it at load.
fn ensure_partial_rope_plain(
    rope_dim: Option<usize>,
    head_dim: usize,
    has_yarn: bool,
    has_freqs: bool,
) -> Result<(), CeraError> {
    if rope_dim.is_some_and(|n| n != head_dim) && (has_yarn || has_freqs) {
        return Err(CeraError::Backend(
            "Hexagon: partial rotary embedding combined with YaRN or rope frequency factors \
             is not supported"
                .to_string(),
        ));
    }
    Ok(())
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
/// keep single-flush speed. Decode is uncapped by default, see
/// `HexagonKnobs::decode_ops` for the opt-in decode cap.
const SMALL_M_FLUSH_CAP_ROWS: usize = 32;
/// Ops-per-flush cap for small-M prefill chunks (see above).
const MAX_OPS_PER_FLUSH: usize = 24;

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
    moe_router_logits: usize,
    moe_probs: usize,
    moe_biased_probs: usize,
    moe_selected_ids: usize,
    moe_selected_weights: usize,
    moe_gate: usize,
    moe_up: usize,
    moe_swiglu: usize,
    moe_down: usize,
    moe_temp_weighted: usize,
    /// Per-token top-k weight renormalization slots (`MOE_RENORM_SLOT_BYTES`
    /// each); 0 when the model has no routed FFN.
    moe_renorm: usize,
    logits: usize,
    argmax: usize,
    pos: usize,
    mask: usize,
    /// Prefill mask for sliding-window layers (0 when the model has none).
    mask_swa: usize,
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
        moe_cfg: Option<&crate::model::MoeConfig>,
        deltanet_conv_dim: Option<usize>,
    ) -> Self {
        let align = |x: usize| x.next_multiple_of(4096);
        let m = PREFILL_MAX_ROWS;
        let mut cur = 0;
        let dnet_dim = deltanet_conv_dim.unwrap_or(0);
        let conv_width = (3 * hidden_size).max(dnet_dim);
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
        cur = align(cur + m * conv_width * 4);
        let conv_bx = cur;
        cur = align(cur + m * hidden_size.max(q_dim) * 4);
        let conv_t0 = cur;
        cur = align(cur + m * hidden_size * 4);
        let conv_t1 = cur;
        cur = align(cur + m * hidden_size * 4);
        let conv_y = cur;
        cur = align(cur + m * hidden_size.max(dnet_dim) * 4);
        let conv_x = cur;
        cur = align(cur + (m + 2) * conv_width * 4);
        let conv_ssm_y = cur;
        cur = align(cur + m * hidden_size.max(dnet_dim) * 4);
        let ffn_gate = cur;
        cur = align(cur + m * intermediate_size * 4);
        let ffn_up = cur;
        cur = align(cur + m * intermediate_size * 4);
        let ffn_out = cur;
        cur = align(cur + m * intermediate_size * 4);

        let (
            moe_router_logits,
            moe_probs,
            moe_biased_probs,
            moe_selected_ids,
            moe_selected_weights,
            moe_gate,
            moe_up,
            moe_swiglu,
            moe_down,
            moe_temp_weighted,
        ) = if let Some(mcfg) = moe_cfg {
            let n_exp = mcfg.n_expert;
            let n_used = mcfg.n_expert_used;
            let ff = mcfg.expert_ff_len;

            let router_logits = cur;
            cur = align(cur + m * n_exp * 4);
            let probs = cur;
            cur = align(cur + m * n_exp * 4);
            let biased_probs = cur;
            cur = align(cur + m * n_exp * 4);
            let selected_ids = cur;
            cur = align(cur + m * n_exp * 4);
            let selected_weights = cur;
            cur = align(cur + m * n_used * 4);
            let gate = cur;
            cur = align(cur + m * n_used * ff * 4);
            let up = cur;
            cur = align(cur + m * n_used * ff * 4);
            let swiglu = cur;
            cur = align(cur + m * n_used * ff * 4);
            let down = cur;
            cur = align(cur + m * n_used * hidden_size * 4);
            let temp_weighted = cur;
            cur = align(cur + m * hidden_size * 4);

            (
                router_logits,
                probs,
                biased_probs,
                selected_ids,
                selected_weights,
                gate,
                up,
                swiglu,
                down,
                temp_weighted,
            )
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        };

        let moe_renorm = if moe_cfg.is_some() {
            let offset = cur;
            cur = align(cur + m * MOE_RENORM_SLOT_BYTES);
            offset
        } else {
            0
        };

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
            moe_router_logits,
            moe_probs,
            moe_biased_probs,
            moe_selected_ids,
            moe_selected_weights,
            moe_gate,
            moe_up,
            moe_swiglu,
            moe_down,
            moe_temp_weighted,
            moe_renorm,
            logits,
            argmax,
            pos,
            mask,
            mask_swa: 0,
            total_size,
        }
    }

    /// Reserve the extra `[kv, M]` prefill mask used by sliding-window layers.
    fn with_swa_mask(mut self, max_seq_len: usize) -> Self {
        let align = |x: usize| (x + 4095) & !4095;
        self.mask_swa = align(self.total_size);
        self.total_size = align(self.mask_swa + PREFILL_MAX_ROWS * max_seq_len * 2);
        self
    }
}

/// Static pre-serialized command queue template for zero-allocation decode dispatch.
struct DecodeTemplate {
    resident_id: u64,
    staged: StagedBatch,
    flash_attn_patches: Vec<FlashAttnPatch>,
    patch_ranges: Vec<std::ops::Range<usize>>,
}

/// Byte size of one K or V cache slab holding `max_seq_len` rows of `dim`
/// elements. The single source of the KV layout: every slab allocation and
/// every DSP tensor stride below derives from these three helpers, so a
/// layout change cannot land in one dispatch only.
fn kv_cache_bytes(kv_dtype: HtpDataType, dim: usize, max_seq_len: usize) -> usize {
    max_seq_len * kv_row_stride(kv_dtype, dim)
}

/// Bytes between consecutive rows of `dim` elements (`nb[1]` of a KV tensor).
fn kv_row_stride(kv_dtype: HtpDataType, dim: usize) -> usize {
    if kv_dtype == HtpDataType::Q8_0 {
        dim.div_ceil(crate::tensor::DType::Q8_0.block_size())
            * crate::tensor::DType::Q8_0.block_bytes()
    } else {
        dim * 2
    }
}

/// Bytes per element step (`nb[0]`): 1 for block-quantized Q8_0, else f16.
fn kv_elem_nb0(kv_dtype: HtpDataType) -> u32 {
    if kv_dtype == HtpDataType::Q8_0 { 1 } else { 2 }
}

/// Re-point every flash-attention op of a resident decode template at the
/// current `seq_len`: K/V tensor row counts, the mask tensor size and strides,
/// and the packed KV-block count in `kernel_params[2]`. Shared by decode and
/// its test so the DSP-visible patch cannot drift from what is pinned.
fn apply_flash_attn_patches(staged: &mut StagedBatch, patches: &[FlashAttnPatch], seq_len: usize) {
    let n_kv_blocks = seq_len.div_ceil(64).max(1) as u32;
    let mask_bytes = (seq_len * 2) as u32;
    for patch in patches {
        staged.update_tensor(patch.k_ti, |t| t.ne[1] = seq_len as u32);
        staged.update_tensor(patch.v_ti, |t| t.ne[1] = seq_len as u32);
        staged.update_tensor(patch.mask_ti, |t| {
            t.size = mask_bytes;
            t.ne[0] = seq_len as u32;
            t.nb[1] = mask_bytes;
            t.nb[2] = mask_bytes;
            t.nb[3] = mask_bytes;
        });

        let b2 = (n_kv_blocks & 0xffff) | ((patch.g as u32 & 0xffff) << 16);
        staged.update_op(patch.op_idx, |o| o.kernel_params[2] = b2 as i32);
    }
}

/// Split an all-logits verification batch at `max` tokens per chunk. Causal
/// attention makes chunked evaluation identical to one batch: each chunk
/// continues at `start_pos + tokens already run`, and the rows concatenate in
/// order. The first chunk error aborts and propagates.
fn chunked_all_logits(
    tokens: &[u32],
    start_pos: usize,
    max: usize,
    vocab_size: usize,
    mut run: impl FnMut(&[u32], usize) -> Result<Vec<f32>, CeraError>,
) -> Result<Vec<f32>, CeraError> {
    let mut all = Vec::with_capacity(tokens.len() * vocab_size);
    for (i, chunk) in tokens.chunks(max).enumerate() {
        all.extend(run(chunk, start_pos + i * max)?);
    }
    Ok(all)
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
pub struct HexagonLfmModel {
    device: Mutex<HexagonDevice>,
    config: ModelConfig,
    session_gate: ModelSessionGate,

    // Token embedding on CPU
    token_embd: EmbeddingTable,

    // Unified weights buffer in rpcmem (all static weights across all layers)
    /// Per-layer routed-expert buffers when the experts are paged through the
    /// DSP mapping; stacked weights with a `group` live here, not in `weights_buf`.
    pager: Option<ExpertPager>,
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

    rope_type: RopeType,
    /// Host-CPU RoPE instead of the DSP kernel: the `CERA_HEXAGON_CPU_ROPE`
    /// debug override, or a model that needs YaRN / frequency factors.
    cpu_rope: bool,
    dense: DenseSemantics,
    activation: FfnActivation,
    attn_logit_softcapping: Option<f32>,
    final_logit_softcapping: Option<f32>,
    has_moe: bool,
    has_deltanet: bool,
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
    /// Decode ops-per-flush cap (`CERA_HEXAGON_DECODE_OPS`, `None` = single flush).
    decode_ops_cap: Option<usize>,
    current_seq_len: AtomicUsize,
    /// Set when a forward failed mid-flight on a model with recurrent (conv or
    /// DeltaNet) layers: those states advance in `kv_state_buf` op by op while
    /// `seq_len` only moves on success, so a retry would recompute over
    /// advanced state. Every forward and partial rewind refuses until
    /// `truncate_kv(0)` / `try_reset_kv` zeroes the state. Attention-only
    /// models recompute their KV slots idempotently and never set it.
    state_torn: AtomicBool,
    /// Gather buffers of the partial-rotary host RoPE (see [`crate::model::llama::RopeGather`]).
    rope_scratch: Mutex<crate::model::llama::RopeGather>,
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

unsafe impl Send for HexagonLfmModel {}
unsafe impl Sync for HexagonLfmModel {}

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

mod blocks;
mod host;
mod ops;
mod pager;
mod weights;
use pager::{ExpertPager, Paging};
use weights::*;

impl HexagonLfmModel {
    /// True when any layer keeps recurrent (conv or DeltaNet) state in `kv_state_buf`.
    fn has_recurrent_layers(&self) -> bool {
        self.layers
            .iter()
            .any(|l| !matches!(l, HexagonLayer::Attention(_)))
    }

    /// Record that a forward failed after it may have advanced recurrent
    /// state (see `state_torn`).
    fn mark_state_torn(&self) {
        if self.has_recurrent_layers() {
            self.state_torn.store(true, Ordering::SeqCst);
        }
    }

    /// [`Self::mark_state_torn`] for a failed forward, but only if the queue
    /// attempted at least one dispatch since `dispatches_before`: a failure
    /// before any batch reached the DSP (emit-time validation, registration)
    /// left the recurrent state untouched, so the session stays usable.
    ///
    /// Invariant this relies on: every host-side recurrent-state mutation
    /// (`step_deltanet_recurrence_row`) is preceded by a flush of a non-empty
    /// batch, and device-side mutation only happens inside a dispatched batch.
    /// The DeltaNet call sites `debug_assert` a weak form of the first half:
    /// after their explicit flush nothing is pending and the dispatch-attempt
    /// counter differs from its value before the block's first op (an earlier
    /// step-mode or op-cap group flush, or that explicit flush, advanced it).
    fn mark_state_torn_if_dispatched(&self, session: &HexagonQueueSession, dispatches_before: u64) {
        if session.dispatch_attempts() != dispatches_before {
            self.mark_state_torn();
        }
    }

    /// Lock the device, the mutex that serializes forwards and resets.
    ///
    /// A panic unwinding through a forward poisons the mutex while recurrent
    /// state may already have advanced, and `mark_state_torn` never ran on
    /// the unwind. On poison this marks the state torn, clears the poison and
    /// returns the guard, so the next forward reports the torn error and a
    /// full reset recovers. Every device lock in this file goes through here.
    /// (The audio, vision and Whisper paths fail closed on poison instead:
    /// nothing there can repair the poison, even though the detokenizer and
    /// Whisper decode do resume state across calls.)
    fn lock_device(&self) -> MutexGuard<'_, HexagonDevice> {
        match self.device.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                self.mark_state_torn();
                self.device.clear_poison();
                guard
            }
        }
    }

    /// Refuse to run over recurrent state torn by an earlier failed forward.
    ///
    /// The `_device` witness makes the lock ordering a compile-time property:
    /// the device lock serializes forwards, so the flag cannot change between
    /// this check and the forward only if the check runs under it. A check
    /// made before locking could pass, wait while another forward fails and
    /// tears the state, and then run over it.
    fn ensure_state_intact(
        &self,
        _device: &MutexGuard<'_, HexagonDevice>,
    ) -> Result<(), CeraError> {
        if self.state_torn.load(Ordering::SeqCst) {
            return Err(CeraError::Backend(
                "NPU recurrent state torn after a device fault; reset the session".to_string(),
            ));
        }
        Ok(())
    }

    /// Assemble the model from a constructor's parts plus the shared setup.
    fn from_parts(
        device: HexagonDevice,
        knobs: &HexagonKnobs,
        adpf: Mutex<Option<AdpfSession>>,
        p: ModelParts,
    ) -> Self {
        Self {
            vtcm_budget: device.hw_info().vtcm_size as usize,
            device: Mutex::new(device),
            has_moe: any_moe(&p.layers),
            config: p.config,
            session_gate: ModelSessionGate::default(),
            token_embd: p.token_embd,
            weights_buf: p.weights_buf,
            pager: p.pager,
            layers: p.layers,
            output_norm_offset: p.output_norm_offset,
            lm_head: p.lm_head,
            kv_state_buf: p.kv_state_buf,
            scratch_buf: p.scratch_buf,
            scratch_offsets: p.scratch_offsets,
            mask_buf: p.mask_buf,
            rope_type: p.rope_type,
            cpu_rope: p.cpu_rope,
            dense: p.dense,
            activation: p.activation,
            attn_logit_softcapping: p.attn_logit_softcapping,
            final_logit_softcapping: p.final_logit_softcapping,
            has_deltanet: p.has_deltanet,
            debug_barriers: knobs.debug_barriers,
            adpf,
            use_ssm_conv: knobs.use_ssm_conv,
            use_hmx: knobs.use_hmx,
            dump_act: knobs.dump_act,
            decode_ops_cap: knobs.decode_ops,
            current_seq_len: AtomicUsize::new(0),
            state_torn: AtomicBool::new(false),
            rope_scratch: Mutex::default(),
            decode_error: Mutex::new(None),
            kv_dtype: p.kv_dtype,
            decode_template: Mutex::new(None),
            greedy_decode_template: Mutex::new(None),
        }
    }

    /// Load an LFM2 model onto the Hexagon NPU from GGUF.
    pub fn from_gguf(
        gguf: GgufFile,
        _path: Option<&Path>,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        Self::from_gguf_on(Backend::open()?, gguf, context_size)
    }

    /// [`Self::from_gguf`] on an already opened device (host tests pass a fake).
    fn from_gguf_on(
        backend: Backend,
        gguf: GgufFile,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        Self::from_gguf_on_with(backend, gguf, context_size, Paging::from_env())
    }

    /// [`Self::from_gguf_on`] with the paging policy given (host tests force it).
    fn from_gguf_on_with(
        backend: Backend,
        gguf: GgufFile,
        context_size: usize,
        paging: Paging,
    ) -> Result<Self, CeraError> {
        let gguf = Arc::new(gguf);
        let config = crate::model::lfm2::LfmModel::parse_config(&gguf, context_size)
            .map_err(|e| CeraError::Backend(e.to_string()))?;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let n_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let rope_type = RopeType::Neox;

        let Backend {
            driver,
            device,
            knobs,
            adpf,
        } = backend;
        let kv_dtype = knobs.kv_dtype();
        let cpu_rope = knobs.cpu_rope;
        let driver = &driver;

        let token_embd = EmbeddingTable::shared(&gguf, vocab_size, hidden_size, 1.0)?;

        let q_dim = n_heads * head_dim;
        let max_kv_dim = max_kv_dim(&config, head_dim);

        // Allocate unified shared scratch buffer
        let scratch_offsets = ScratchOffsets::new(
            hidden_size,
            q_dim,
            max_kv_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            config.moe.as_ref(),
            None,
        );
        let ModelBuffers {
            scratch_buf,
            mask_buf,
            ..
        } = ModelBuffers::alloc(driver, &scratch_offsets, (max_seq_len * 2).max(128), false)?;

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

        let src = GgufSource { gguf: &gguf };
        // Routed experts that cannot all stay mapped are planned into one
        // buffer per layer and paged (see `pager`). Scratch is already mapped
        // here, so `mapped_bytes` counts it.
        let map_budget = paging.budget;
        let gguf_bytes: u64 = gguf.tensors.values().map(|t| t.size_bytes as u64).sum();
        let paged = config.moe.is_some() && paging.should_page(gguf_bytes, driver.mapped_bytes());
        let mut plan = WeightPlanner::new(&src).with_paged_experts(paged);
        let mut kv = KvPlanner::new(kv_dtype, max_seq_len);
        let state_size = align256(hidden_size * 4);
        let moe_dims = |i: usize| {
            config
                .moe
                .as_ref()
                .filter(|m| m.is_moe_layer.get(i).copied().unwrap_or(false))
                .map(|m| MoeDims {
                    n_expert: m.n_expert,
                    n_expert_used: m.n_expert_used,
                    expert_ff_len: m.expert_ff_len,
                })
        };

        // Weights are planned in layer order (norms, projections, FFN); KV and
        // conv state offsets follow the same order in their own buffer.
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn_norm_offset = plan.vector(&format!("blk.{i}.attn_norm.weight"), hidden_size);
            if config.block_types[i] == BlockType::Attention {
                let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                let attn_q = plan.weight(&format!("blk.{i}.attn_q.weight"), hidden_size, q_dim)?;
                let attn_k = plan.weight(&format!("blk.{i}.attn_k.weight"), hidden_size, kv_dim)?;
                let attn_v = plan.weight(&format!("blk.{i}.attn_v.weight"), hidden_size, kv_dim)?;
                let attn_output =
                    plan.weight(&format!("blk.{i}.attn_output.weight"), q_dim, hidden_size)?;
                let attn_q_norm_offset = has_attn_q_norm[i]
                    .then(|| plan.vector(&format!("blk.{i}.attn_q_norm.weight"), q_dim));
                let attn_k_norm_offset = has_attn_k_norm[i]
                    .then(|| plan.vector(&format!("blk.{i}.attn_k_norm.weight"), kv_dim));
                let ffn_norm_offset = plan.vector(&format!("blk.{i}.ffn_norm.weight"), hidden_size);
                let ffn = plan.ffn(i, hidden_size, intermediate_size, moe_dims(i), [None; 3])?;
                let (k_offset, v_offset) = kv.attention(kv_dim);
                layers.push(HexagonLayer::Attention(HexagonAttentionLayer {
                    attn_q_norm_offset,
                    attn_k_norm_offset,
                    ..HexagonAttentionLayer::plain(
                        attn_norm_offset,
                        [attn_q, attn_k, attn_v, attn_output],
                        ffn_norm_offset,
                        ffn,
                        (k_offset, v_offset),
                        q_dim,
                        kv_dim,
                    )
                }));
            } else {
                let in_proj = plan.weight(
                    &format!("blk.{i}.shortconv.in_proj.weight"),
                    hidden_size,
                    3 * hidden_size,
                )?;
                let out_proj = plan.weight(
                    &format!("blk.{i}.shortconv.out_proj.weight"),
                    hidden_size,
                    hidden_size,
                )?;
                let ([conv_w0_offset, conv_w1_offset, conv_w2_offset], conv_ssm_offset) =
                    plan.conv_taps(&format!("blk.{i}.shortconv.conv.weight"), hidden_size);
                let ffn_norm_offset = plan.vector(&format!("blk.{i}.ffn_norm.weight"), hidden_size);
                let ffn = plan.ffn(i, hidden_size, intermediate_size, moe_dims(i), [None; 3])?;
                let state_offset = kv.conv(state_size);
                layers.push(HexagonLayer::Conv(HexagonConvLayer {
                    attn_norm_offset,
                    in_proj,
                    out_proj,
                    conv_w0_offset,
                    conv_w1_offset,
                    conv_w2_offset,
                    conv_ssm_offset,
                    state_offset,
                    ffn_norm_offset,
                    ffn,
                }));
            }
        }

        let lm_head_name = if gguf.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output_norm_name = if gguf.tensors.contains_key("output_norm.weight") {
            "output_norm.weight"
        } else if gguf.tensors.contains_key("token_embd_norm.weight") {
            "token_embd_norm.weight"
        } else {
            return Err(CeraError::Backend("missing output_norm tensor".into()));
        };
        let output_norm_offset = plan.vector(output_norm_name, hidden_size);
        let lm_head = plan.weight(lm_head_name, hidden_size, vocab_size)?;

        let (weights_buf, kv_state_buf, expert_bufs) = WeightCopy {
            src: &src,
            weights_total: plan.total,
            kv_total: kv.total,
            copies: &plan.copies,
            groups: &plan.groups,
        }
        .run(driver)?;
        let pager = if expert_bufs.is_empty() {
            None
        } else {
            let n_groups = expert_bufs.len();
            let pager = ExpertPager::new(Arc::clone(driver), expert_bufs, map_budget)?;
            tracing::info!(
                "cera::hexagon: paging routed experts through the DSP mapping: {} of {n_groups} \
                 layers pinned, the rest rotate (budget {} KiB, {} KiB mapped, {} KiB per layer)",
                pager.pinned(),
                map_budget >> 10,
                driver.mapped_bytes() >> 10,
                pager.buf(0).size() >> 10
            );
            Some(pager)
        };

        Ok(Self::from_parts(
            device,
            &knobs,
            adpf,
            ModelParts {
                config,
                token_embd,
                weights_buf,
                pager,
                layers,
                output_norm_offset,
                lm_head,
                kv_state_buf,
                scratch_buf,
                scratch_offsets,
                mask_buf,
                rope_type,
                cpu_rope,
                dense: DenseSemantics::default(),
                activation: FfnActivation::Swiglu,
                attn_logit_softcapping: None,
                final_logit_softcapping: None,
                has_deltanet: false,
                kv_dtype,
            },
        ))
    }

    /// Load a dense transformer model (LLaMA, Qwen2, Qwen3, Granite, Mistral, Phi, etc.)
    /// onto the Hexagon NPU from GGUF.
    pub fn from_llama(
        gguf: GgufFile,
        path: Option<&Path>,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        Self::from_llama_on(Backend::open()?, gguf, path, context_size)
    }

    /// [`Self::from_llama`] on an already opened device.
    fn from_llama_on(
        backend: Backend,
        gguf: GgufFile,
        path: Option<&Path>,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        let model_id = path
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let cpu = crate::model::llama::LlamaModel::from_gguf_with_id_no_repack(
            gguf,
            context_size,
            model_id,
        )
        .map_err(|e| CeraError::Backend(e.to_string()))?;
        let extras = DenseExtras::from_llama(&cpu);
        Self::from_dense_weight_source_on(backend, &cpu, &extras, context_size)
    }

    /// Load a Qwen 3.5 / Ornith 1.0 hybrid linear/full attention model onto the Hexagon NPU.
    pub fn from_qwen35(
        gguf: GgufFile,
        path: Option<&Path>,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        let model_id = path
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let cpu =
            crate::model::qwen35::Qwen35Model::from_gguf_with_id(gguf, context_size, model_id)
                .map_err(|e| CeraError::Backend(e.to_string()))?;
        Self::from_qwen35_model(&cpu, context_size)
    }

    /// Build a Hexagon hybrid linear/full attention model from a loaded Qwen35Model.
    pub fn from_qwen35_model(
        cpu: &crate::model::qwen35::Qwen35Model,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        Self::from_qwen35_model_on(Backend::open()?, cpu, context_size)
    }

    /// [`Self::from_qwen35_model`] on an already opened device.
    fn from_qwen35_model_on(
        backend: Backend,
        cpu: &crate::model::qwen35::Qwen35Model,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        let mut config = cpu.config.clone();
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let n_heads = config.n_heads;
        let head_dim = cpu.head_dim;
        let vocab_size = config.vocab_size;
        let n_layers = config.n_layers;
        // Qwen 3.5's CPU reference (`cpu::rope`) rotates split-halves (NEOX).
        let rope_type = RopeType::Neox;
        // `rope.dimension_count` below head_dim rotates only a prefix of each head
        // (llama.cpp `n_rot`). The DSP rope params here carry `head_dim` as
        // `n_dims`, so the partial case takes the host RoPE route.
        let partial_rope = cpu.rope_dim < head_dim;
        if partial_rope {
            hexagon_warn!(
                "Qwen 3.5 rope.dimension_count ({}) < head_dim ({head_dim}); RoPE runs on the host CPU",
                cpu.rope_dim
            );
        }

        let Backend {
            driver,
            device,
            knobs,
            adpf,
        } = backend;
        let kv_dtype = knobs.kv_dtype();
        let cpu_rope = knobs.cpu_rope;
        let driver = &driver;

        let token_embd = EmbeddingTable::new(&cpu.gguf, vocab_size, hidden_size, 1.0)?;

        let ssm_cfg = config.ssm.as_ref().ok_or_else(|| {
            CeraError::Backend("missing SSM configuration for Qwen 3.5 model".into())
        })?;
        let ssm_conv_dim =
            2 * (ssm_cfg.n_group * ssm_cfg.d_state) + (ssm_cfg.dt_rank * ssm_cfg.d_state);
        let ssm_value_dim = ssm_cfg.dt_rank * ssm_cfg.d_state;
        let q_dim = n_heads * head_dim;

        let mut max_q_out_dim = q_dim;
        for layer in &cpu.layers {
            if let crate::model::qwen35::LayerKindRefs::Attention(ref a) = layer.kind
                && a.attn_q.m > max_q_out_dim
            {
                max_q_out_dim = a.attn_q.m;
            }
        }

        let max_kv_dim = max_kv_dim(&config, head_dim);

        let scratch_offsets = ScratchOffsets::new(
            hidden_size,
            max_q_out_dim,
            max_kv_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            None,
            Some(ssm_conv_dim),
        );
        let ModelBuffers {
            scratch_buf,
            mask_buf,
            ..
        } = ModelBuffers::alloc(driver, &scratch_offsets, max_seq_len * 2, false)?;

        let src = Qwen35Source { cpu };
        let mut plan = WeightPlanner::new(&src);
        let mut kv = KvPlanner::new(kv_dtype, max_seq_len);

        let mut layers = Vec::with_capacity(n_layers);
        for (i, layer) in cpu.layers.iter().enumerate() {
            let attn_norm_offset = plan.vector(&format!("blk.{i}.attn_norm.weight"), hidden_size);
            let ffn_norm_offset = plan.vector(&format!("blk.{i}.ffn_norm.weight"), hidden_size);
            let ffn = plan.ffn(i, hidden_size, intermediate_size, None, [None; 3])?;

            match &layer.kind {
                crate::model::qwen35::LayerKindRefs::Attention(attn) => {
                    let has_q_gate = attn.attn_q.m == 2 * n_heads * head_dim;
                    let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                    let attn_q = plan.weight(
                        &format!("blk.{i}.attn_q.weight"),
                        hidden_size,
                        attn.attn_q.m,
                    )?;
                    let attn_k =
                        plan.weight(&format!("blk.{i}.attn_k.weight"), hidden_size, kv_dim)?;
                    let attn_v =
                        plan.weight(&format!("blk.{i}.attn_v.weight"), hidden_size, kv_dim)?;
                    let attn_output =
                        plan.weight(&format!("blk.{i}.attn_output.weight"), q_dim, hidden_size)?;
                    let attn_q_norm_offset = Some(plan.vector(
                        &format!("blk.{i}.attn_q_norm.weight"),
                        attn.attn_q_norm.len().max(head_dim),
                    ));
                    let attn_k_norm_offset = Some(plan.vector(
                        &format!("blk.{i}.attn_k_norm.weight"),
                        attn.attn_k_norm.len().max(head_dim),
                    ));
                    let (k_offset, v_offset) = kv.attention(kv_dim);

                    layers.push(HexagonLayer::Attention(HexagonAttentionLayer {
                        attn_q_norm_offset,
                        attn_k_norm_offset,
                        has_q_gate,
                        ..HexagonAttentionLayer::plain(
                            attn_norm_offset,
                            [attn_q, attn_k, attn_v, attn_output],
                            ffn_norm_offset,
                            ffn,
                            (k_offset, v_offset),
                            q_dim,
                            kv_dim,
                        )
                    }));
                }
                crate::model::qwen35::LayerKindRefs::DeltaNet(dnet) => {
                    let wqkv = plan.weight(
                        &format!("blk.{i}.attn_qkv.weight"),
                        hidden_size,
                        ssm_conv_dim,
                    )?;
                    let wqkv_gate = plan.weight(
                        &format!("blk.{i}.attn_gate.weight"),
                        hidden_size,
                        ssm_value_dim,
                    )?;
                    let ssm_beta = plan.weight(
                        &format!("blk.{i}.ssm_beta.weight"),
                        hidden_size,
                        ssm_cfg.dt_rank,
                    )?;
                    let ssm_alpha = plan.weight(
                        &format!("blk.{i}.ssm_alpha.weight"),
                        hidden_size,
                        ssm_cfg.dt_rank,
                    )?;
                    let ssm_conv1d_offset =
                        plan.vector(&format!("blk.{i}.ssm_conv1d.weight"), dnet.ssm_conv1d.len());
                    let ssm_conv1d_bias_offset = dnet
                        .ssm_conv1d_bias
                        .as_ref()
                        .map(|b| plan.vector(&format!("blk.{i}.ssm_conv1d.bias"), b.len()));
                    let ssm_dt_offset =
                        plan.vector(&format!("blk.{i}.ssm_dt.bias"), dnet.ssm_dt.len());
                    let ssm_a_offset = plan.vector(&format!("blk.{i}.ssm_a"), dnet.ssm_a.len());
                    let ssm_norm_offset =
                        plan.vector(&format!("blk.{i}.ssm_norm.weight"), dnet.ssm_norm.len());
                    let ssm_out = plan.weight(
                        &format!("blk.{i}.ssm_out.weight"),
                        ssm_value_dim,
                        hidden_size,
                    )?;
                    let (conv_state_offset, ssm_state_offset) = kv.deltanet(
                        ssm_conv_dim,
                        ssm_cfg.d_conv,
                        ssm_cfg.dt_rank,
                        ssm_cfg.d_state,
                    );

                    layers.push(HexagonLayer::DeltaNet(HexagonDeltaNetLayer {
                        attn_norm_offset,
                        wqkv,
                        wqkv_gate,
                        ssm_beta,
                        ssm_alpha,
                        ssm_conv1d_offset,
                        ssm_conv1d_bias_offset,
                        ssm_dt_offset,
                        ssm_a_offset,
                        ssm_norm_offset,
                        ssm_out,
                        attn_post_norm_offset: None,
                        ffn_norm_offset,
                        ffn,
                        ffn_post_norm_offset: None,
                        conv_state_offset,
                        ssm_state_offset,
                        conv_dim: ssm_conv_dim,
                        d_conv: ssm_cfg.d_conv,
                        d_state: ssm_cfg.d_state,
                        dt_rank: ssm_cfg.dt_rank,
                        n_group: ssm_cfg.n_group,
                    }));
                }
            }
        }

        let output_norm_offset = plan.vector("output_norm.weight", hidden_size);
        let head_name = if cpu.output_ref.is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head = plan.weight(head_name, hidden_size, vocab_size)?;

        let (weights_buf, kv_state_buf, expert_bufs) = WeightCopy {
            src: &src,
            weights_total: plan.total,
            kv_total: kv.total,
            copies: &plan.copies,
            groups: &[],
        }
        .run(driver)?;
        debug_assert!(expert_bufs.is_empty());

        let dense = DenseSemantics {
            rope_dim: partial_rope.then_some(cpu.rope_dim),
            ..Default::default()
        };
        let has_yarn = layers
            .iter()
            .any(|l| matches!(l, HexagonLayer::Attention(a) if a.yarn.is_some()));
        ensure_partial_rope_plain(
            dense.rope_dim,
            head_dim,
            has_yarn,
            dense.rope_freqs.is_some(),
        )?;

        Ok(Self::from_parts(
            device,
            &knobs,
            adpf,
            ModelParts {
                config,
                token_embd,
                weights_buf,
                pager: None,
                layers,
                output_norm_offset,
                lm_head,
                kv_state_buf,
                scratch_buf,
                scratch_offsets,
                mask_buf,
                rope_type,
                cpu_rope: cpu_rope || partial_rope,
                dense,
                activation: FfnActivation::Swiglu,
                attn_logit_softcapping: None,
                final_logit_softcapping: None,
                has_deltanet: true,
                kv_dtype,
            },
        ))
    }

    /// Generalized Hexagon loader over a dense transformer [`GpuWeightSource`].
    /// Uploads and repacks weights into contiguous shared memory, allocates
    /// KV caches, scratch buffers, and wires per-architecture parameters.
    fn from_dense_weight_source_on(
        backend: Backend,
        src: &dyn GpuWeightSource,
        extras: &DenseExtras,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        let mut config = src.config().clone();
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim;
        let vocab_size = config.vocab_size;
        let n_layers = config.n_layers;
        let rope_type = src.rope_type();
        let scalars = config.scalars;

        // Semantics beyond the plain pre-norm block (see `DenseSemantics`).
        // A sliding window at least as long as the context never masks anything.
        let swa_window = extras.swa_window.filter(|&w| w > 0 && w < max_seq_len);
        let layer_swa: Vec<bool> = (0..n_layers)
            .map(|i| swa_window.is_some() && extras.layer_swa.get(i).copied().unwrap_or(false))
            .collect();
        let has_swa = layer_swa.iter().any(|&b| b);
        let attn_temp = extras
            .attn_temp_scale
            .filter(|&(scale, floor)| scale > 0.0 && floor > 0 && max_seq_len > floor);
        let rope_freqs = src.rope_freqs().map(<[f32]>::to_vec);
        // The DSP rope kernel carries neither YaRN nor frequency-factor inputs
        // here. NORM RoPE uses the factors only without YaRN (CPU reference).
        let needs_host_rope = extras.layer_yarn.iter().any(Option::is_some)
            || (rope_type == RopeType::Norm && rope_freqs.is_some());
        if needs_host_rope {
            hexagon_warn!(
                "dense path RoPE runs on the host CPU (YaRN or Llama-3 frequency factors); \
                 decode flushes the DSP queue once per attention layer"
            );
        }
        if attn_temp.is_some() {
            hexagon_warn!(
                "dense path attention temperature scaling runs on the host CPU past its floor position"
            );
        }
        let post_norm = n_layers > 0 && src.attn_norm_weight(0).is_empty();

        let Backend {
            driver,
            device,
            knobs,
            adpf,
        } = backend;
        let kv_dtype = knobs.kv_dtype();
        let cpu_rope = knobs.cpu_rope;
        let driver = &driver;

        // Embedding multiplier (Granite, Gemma) is folded into each looked-up row.
        let token_embd = EmbeddingTable::new(
            src.gguf(),
            vocab_size,
            hidden_size,
            src.config().scalars.embedding,
        )?;

        let q_dim = n_heads * head_dim;
        let max_kv_dim = max_kv_dim(&config, head_dim);

        // Allocate unified shared scratch buffer
        let mut scratch_offsets = ScratchOffsets::new(
            hidden_size,
            q_dim,
            max_kv_dim,
            intermediate_size,
            vocab_size,
            max_seq_len,
            config.moe.as_ref(),
            None,
        );
        if has_swa {
            scratch_offsets = scratch_offsets.with_swa_mask(max_seq_len);
        }
        let ModelBuffers {
            scratch_buf,
            mask_buf,
            mask_swa,
        } = ModelBuffers::alloc(
            driver,
            &scratch_offsets,
            (max_seq_len * 2).max(128),
            has_swa,
        )?;

        let src_tensors = DenseSource { src, extras };
        let mut plan = WeightPlanner::new(&src_tensors);
        let mut kv = KvPlanner::new(kv_dtype, max_seq_len);
        let moe_dims = |i: usize| {
            src.moe_refs(i).map(|m| MoeDims {
                n_expert: m.n_expert,
                n_expert_used: m.n_expert_used,
                expert_ff_len: m.expert_ff_len,
            })
        };

        let mut layers = Vec::with_capacity(n_layers);
        // Indexes several per-layer vectors (config, extras, sources) at once.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_layers {
            let kv_dim = config.kv_heads_per_layer[i] * head_dim;
            let attn_norm_offset = plan.vector(&format!("blk.{i}.attn_norm.weight"), hidden_size);

            let attn_q = plan.weight(&format!("blk.{i}.attn_q.weight"), hidden_size, q_dim)?;
            let attn_k = plan.weight(&format!("blk.{i}.attn_k.weight"), hidden_size, kv_dim)?;
            let attn_v = plan.weight(&format!("blk.{i}.attn_v.weight"), hidden_size, kv_dim)?;
            let attn_output =
                plan.weight(&format!("blk.{i}.attn_output.weight"), q_dim, hidden_size)?;

            let attn_q_norm_offset = src.attn_q_norm_weight(i).map(|w| {
                plan.vector(
                    &format!("blk.{i}.attn_q_norm.weight"),
                    w.len().max(head_dim),
                )
            });
            let attn_k_norm_offset = src.attn_k_norm_weight(i).map(|w| {
                plan.vector(
                    &format!("blk.{i}.attn_k_norm.weight"),
                    w.len().max(head_dim),
                )
            });
            let attn_post_norm_offset = src.attn_post_norm_weight(i).map(|w| {
                plan.vector(
                    &format!("blk.{i}.attn_post_norm.weight"),
                    w.len().max(hidden_size),
                )
            });

            // Q/K/V biases apply only as a triple (CPU reference).
            let qkv_bias = match (src.attn_q_bias(i), src.attn_k_bias(i), src.attn_v_bias(i)) {
                (Some(qb), Some(kb), Some(vb)) => Some([
                    plan.bias_required(&format!("blk.{i}.attn_q.bias"), qb, q_dim)?,
                    plan.bias_required(&format!("blk.{i}.attn_k.bias"), kb, kv_dim)?,
                    plan.bias_required(&format!("blk.{i}.attn_v.bias"), vb, kv_dim)?,
                ]),
                _ => None,
            };
            let out_bias = plan.bias(
                &format!("blk.{i}.attn_output.bias"),
                extras.attn_output_bias[i].as_deref(),
                hidden_size,
            )?;
            let qk_norm_full = src
                .attn_q_norm_weight(i)
                .is_some_and(|w| w.len() != head_dim);

            let ffn_norm_offset = plan.vector(&format!("blk.{i}.ffn_norm.weight"), hidden_size);
            let ffn = plan.ffn(
                i,
                hidden_size,
                intermediate_size,
                moe_dims(i),
                [
                    extras.ffn_gate_bias[i].as_deref(),
                    extras.ffn_up_bias[i].as_deref(),
                    extras.ffn_down_bias[i].as_deref(),
                ],
            )?;
            let ffn_post_norm_offset = src.ffn_post_norm_weight(i).map(|w| {
                plan.vector(
                    &format!("blk.{i}.ffn_post_norm.weight"),
                    w.len().max(hidden_size),
                )
            });
            let (k_offset, v_offset) = kv.attention(kv_dim);

            layers.push(HexagonLayer::Attention(HexagonAttentionLayer {
                attn_q_norm_offset,
                attn_k_norm_offset,
                attn_post_norm_offset,
                ffn_post_norm_offset,
                qkv_bias,
                out_bias,
                qk_norm_full,
                swa: layer_swa[i],
                yarn: extras.layer_yarn[i],
                ..HexagonAttentionLayer::plain(
                    attn_norm_offset,
                    [attn_q, attn_k, attn_v, attn_output],
                    ffn_norm_offset,
                    ffn,
                    (k_offset, v_offset),
                    q_dim,
                    kv_dim,
                )
            }));
        }

        // Granite / MiniCPM residual multiplier: a constant `[hidden_size]`
        // vector the DSP multiplies block outputs by (row broadcast).
        let residual_vec_offset = (scalars.residual != 1.0)
            .then(|| plan.constant("residual multiplier", vec![scalars.residual; hidden_size]));

        let output_norm_offset = plan.vector("output_norm.weight", hidden_size);
        let head_name = if src.output_ref().is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head = plan.weight(head_name, hidden_size, vocab_size)?;

        let (weights_buf, kv_state_buf, expert_bufs) = WeightCopy {
            src: &src_tensors,
            weights_total: plan.total,
            kv_total: kv.total,
            copies: &plan.copies,
            groups: &[],
        }
        .run(driver)?;
        debug_assert!(expert_bufs.is_empty());

        let dense = DenseSemantics {
            attn_scale: scalars.attn,
            rope_dim: None,
            residual_vec_offset,
            logit_scale: (scalars.logit != 1.0).then(|| 1.0 / scalars.logit),
            rope_freqs,
            post_norm,
            loop_norm_interval: src.loop_norm_interval(),
            attn_temp,
            swa_window,
            mask_swa,
        };

        Ok(Self::from_parts(
            device,
            &knobs,
            adpf,
            ModelParts {
                config,
                token_embd,
                weights_buf,
                pager: None,
                layers,
                output_norm_offset,
                lm_head,
                kv_state_buf,
                scratch_buf,
                scratch_offsets,
                mask_buf,
                rope_type,
                cpu_rope: cpu_rope || needs_host_rope,
                dense,
                activation: src.activation(),
                attn_logit_softcapping: src.attn_logit_softcapping(),
                final_logit_softcapping: src.final_logit_softcapping(),
                has_deltanet: false,
                kv_dtype,
            },
        ))
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

        let mut device = self.lock_device();
        self.ensure_state_intact(&device)?;

        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;

        // Ingest token embeddings or raw float embeddings, positions, and causal mask.
        unsafe {
            match input {
                PrefillInput::Tokens(tokens) => {
                    for (i, &t) in tokens.iter().enumerate() {
                        let row = std::slice::from_raw_parts_mut(
                            scratch.as_mut_ptr().add(so.activation + i * hs * 4) as *mut f32,
                            hs,
                        );
                        self.token_embd.row_into(t as usize, row);
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
            fill_prefill_mask(mask, start_pos, m, kv_len, None);
            // Sliding-window layers get their own windowed copy.
            if let Some(window) = self.dense.swa_window
                && so.mask_swa != 0
            {
                let swa_mask = std::slice::from_raw_parts_mut(
                    scratch.as_mut_ptr().add(so.mask_swa) as *mut u16,
                    kv_len * m,
                );
                fill_prefill_mask(swa_mask, start_pos, m, kv_len, Some(window));
            }
        }
        scratch.flush_cpu_cache(so.activation, m * hs * 4);
        scratch.flush_cpu_cache(so.pos, m * 4);
        scratch.flush_cpu_cache(so.mask, kv_len * m * 2);
        if so.mask_swa != 0 {
            scratch.flush_cpu_cache(so.mask_swa, kv_len * m * 2);
        }

        let eps = self.config.rms_norm_eps;

        let session = device.queue_session_mut();
        session.drop_pending_batch();
        let dispatches_before = session.dispatch_attempts();

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
                let (cur_act, next_act, cur_normed) = self.layer_buffers(layer_idx);

                match layer {
                    HexagonLayer::Attention(attn) => {
                        self.emit_attention_block(
                            session,
                            attn,
                            layer_idx,
                            cur_act,
                            next_act,
                            cur_normed,
                            AttnPass::Prefill { start_pos, m },
                        )?;
                    }
                    HexagonLayer::Conv(conv) => {
                        self.emit_conv_prefill(
                            session, conv, layer_idx, cur_act, next_act, cur_normed, m,
                        )?;
                    }
                    HexagonLayer::DeltaNet(dnet) => {
                        self.emit_deltanet_prefill(
                            session, dnet, layer_idx, cur_act, next_act, cur_normed, m,
                        )?;
                    }
                }
                self.emit_loop_norm(session, layer_idx, next_act, m)?;
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
            self.mark_state_torn_if_dispatched(session, dispatches_before);
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
        // Advisory power-hint state: safe to keep after a poison.
        let mut adpf = self.adpf.lock_or_recover();
        if let Some(session) = adpf.as_mut() {
            let per_token_ns =
                (fwd_start.elapsed().as_nanos() / m.max(1) as u128).min(i64::MAX as u128) as i64;
            session.report(per_token_ns);
        }
        let mut logits = logits_slice.to_vec();
        if let Some(scale) = self.dense.logit_scale {
            crate::backend::cpu::scale_inplace(&mut logits, scale);
        }
        if let Some(cap) = self.final_logit_softcapping {
            crate::backend::cpu::softcap_inplace(&mut logits, cap);
        }
        Ok(logits)
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
    /// `hexagon_error!` pairs the structured log with stderr because no
    /// `tracing` subscriber exists on the shipping NPU platforms (Android/iOS).
    fn forward_prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Option<Vec<f32>> {
        match self.try_forward_prefill_chunk(tokens, start_pos, state) {
            Ok(logits) => Some(logits),
            Err(e) => {
                hexagon_error!("prefill chunk failed, aborting prefill: {e}");
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
                hexagon_error!("prefill chunk from embeddings failed, aborting prefill: {e}");
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

        let mut device = self.lock_device();
        self.ensure_state_intact(&device)?;

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
                let row = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratch.as_mut_ptr().add(so.activation) as *mut f32,
                        hs,
                    )
                };
                self.token_embd.row_into(token, row);
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
        // Sliding-window layers read a per-token mask (the plain decode mask
        // is an all-zero identity), rewritten here before any op is queued.
        if let (Some(swa_mask), Some(window)) = (&self.dense.mask_swa, self.dense.swa_window) {
            let seq_len = pos + 1;
            let mask = unsafe {
                std::slice::from_raw_parts_mut(swa_mask.as_mut_ptr() as *mut u16, seq_len)
            };
            fill_decode_swa_mask(mask, seq_len, window);
            swa_mask.flush_cpu_cache(0, seq_len * 2);
        }

        let eps = self.config.rms_norm_eps;
        let vocab_size = self.config.vocab_size;

        let session = device.queue_session_mut();
        session.drop_pending_batch();
        let dispatches_before = session.dispatch_attempts();

        let can_use_template = !self.has_moe
            && !self.has_deltanet
            && !self.cpu_rope
            && !self.dense.needs_host_step()
            && !self.debug_barriers
            && !self.dump_act
            && self.decode_ops_cap.is_none()
            && !session.step_mode()
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
            // The patched template is mutated in place, so a panic mid-patch
            // may leave it half-updated: drop it and take the rebuild path.
            let mut guard = lock_or_discard(template_slot);
            if let Some(tpl) = guard.as_mut() {
                apply_flash_attn_patches(&mut tpl.staged, &tpl.flash_attn_patches, pos + 1);
                if let Err(e) =
                    session.flush_staged_resident(tpl.resident_id, &tpl.staged, &tpl.patch_ranges)
                {
                    // The only pre-dispatch exit of a replay is the staging
                    // size check, unreachable with fixture-sized batches, so
                    // this site is covered by the seam test
                    // `torn_only_when_a_dispatch_was_attempted`.
                    self.mark_state_torn_if_dispatched(session, dispatches_before);
                    return Err(CeraError::Backend(format!(
                        "Hexagon NPU execution failed: {e}"
                    )));
                }

                self.current_seq_len.store(pos + 1, Ordering::SeqCst);
                state.seq_len = pos + 1;

                let mut adpf = self.adpf.lock_or_recover();
                if let Some(session) = adpf.as_mut() {
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
                        let mut logits = logits_slice.to_vec();
                        if let Some(scale) = self.dense.logit_scale {
                            crate::backend::cpu::scale_inplace(&mut logits, scale);
                        }
                        if let Some(cap) = self.final_logit_softcapping {
                            crate::backend::cpu::softcap_inplace(&mut logits, cap);
                        }
                        Ok(DecodeResult::Logits(logits))
                    }
                    DecodeOutput::Hidden => unreachable!(),
                };
            }
        }

        // Decode determinism: cap ops per flush if configured.
        session.set_max_ops_per_flush(self.decode_ops_cap);

        let mut flash_attn_patches = Vec::new();

        let run_res = (|| -> Result<(), CeraError> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (cur_act, next_act, cur_normed) = self.layer_buffers(layer_idx);

                match layer {
                    HexagonLayer::Attention(attn) => {
                        self.emit_attention_block(
                            session,
                            attn,
                            layer_idx,
                            cur_act,
                            next_act,
                            cur_normed,
                            AttnPass::Decode {
                                pos,
                                patches: can_use_template.then_some(&mut flash_attn_patches),
                            },
                        )?;
                    }
                    HexagonLayer::Conv(conv) => {
                        self.emit_conv_decode(
                            session, conv, layer_idx, cur_act, next_act, cur_normed,
                        )?;
                    }
                    HexagonLayer::DeltaNet(dnet) => {
                        self.emit_deltanet_decode(
                            session, dnet, layer_idx, cur_act, next_act, cur_normed,
                        )?;
                    }
                }

                self.emit_loop_norm(session, layer_idx, next_act, 1)?;

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
                let mut guard = template_slot.lock_or_recover();
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
            self.mark_state_torn_if_dispatched(session, dispatches_before);
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
                let mut logits = logits_slice.to_vec();
                if let Some(scale) = self.dense.logit_scale {
                    crate::backend::cpu::scale_inplace(&mut logits, scale);
                }
                if let Some(cap) = self.final_logit_softcapping {
                    crate::backend::cpu::softcap_inplace(&mut logits, cap);
                }
                DecodeResult::Logits(logits)
            }
            DecodeOutput::Hidden => {
                scratch.invalidate_cpu_cache(final_normed, hs * 4);
                let hidden_slice = unsafe {
                    std::slice::from_raw_parts(scratch.as_ptr().add(final_normed) as *const f32, hs)
                };
                DecodeResult::Hidden(hidden_slice.to_vec())
            }
        };

        let mut adpf = self.adpf.lock_or_recover();
        if let Some(session) = adpf.as_mut() {
            session.report(fwd_start.elapsed().as_nanos().min(i64::MAX as u128) as i64);
        }
        Ok(result)
    }

    /// Prefill every token but the last. A failed chunk aborts with an error
    /// instead of decoding the last token over the KV hole it left behind.
    fn prefill_head(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<(), CeraError> {
        let head = &tokens[..tokens.len() - 1];
        let (consumed, _) = run_scratch_chunks(head, pos, state, |c, p, s| {
            self.forward_prefill_chunk(c, p, s)
        });
        if consumed != head.len() {
            return Err(CeraError::Backend(format!(
                "Hexagon prefill aborted after {consumed} of {} tokens",
                head.len()
            )));
        }
        Ok(())
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
            self.prefill_head(tokens, pos, state)?;
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
            self.prefill_head(tokens, pos, state)?;
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

/// Embedding-input twin of [`run_scratch_chunks`]: same abort-at-first-failure
/// policy, chunking `embeddings` (`n_tokens * hidden_size` floats) by scratch
/// capacity. `n_tokens` only gates the empty case; the per-chunk token count
/// comes from the slice length.
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

impl Model for HexagonLfmModel {
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
                hexagon_error!("decode from embedding failed, returning zero logits: {e}");
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
                hexagon_error!("forward embedding failed, returning zero hidden: {e}");
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
                hexagon_error!("forward hidden from embedding failed, returning zero hidden: {e}");
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
        // A bad shape is a caller bug, but a panic here would cross the FFI
        // boundary and abort the host: record a fault and return zeros, like
        // the other prefill failures.
        if n_tokens == 0 || embeddings.len() != n_tokens * hs {
            let e = CeraError::Backend(format!(
                "forward_prefill_from_embeddings needs n_tokens > 0 and \
                 embeddings.len() ({}) == n_tokens ({n_tokens}) * hidden_size ({hs})",
                embeddings.len(),
            ));
            eprintln!("[cera-hexagon] {e}");
            record_first_fault(&self.decode_error, e);
            return vec![0.0f32; self.config.vocab_size];
        }
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
                // `hexagon_error!` also writes stderr: no `tracing` subscriber
                // on the shipping NPU platforms, so without it the failure is
                // zero logits with zero record.
                hexagon_error!("decode failed, returning zero logits: {e}");
                // Record for `take_decode_error`: the session fails the
                // generation on this instead of sampling the zeros below
                // as token 0. Sticky until taken (see the trait docs).
                // First fault wins: a multi-chunk prefill can fail more
                // than once per take, and the surfaced error should name
                // the root cause (the log line above keeps full order).
                record_first_fault(&self.decode_error, e);
                vec![0.0f32; self.config.vocab_size]
            }
        }
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        match self.try_forward_greedy(tokens, pos, state) {
            Ok(token) => token,
            Err(e) => {
                hexagon_error!("greedy decode failed, returning token 0: {e}");
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
        // Speculative verification batches `1 + k` tokens and `k` can be 64, so
        // split at the scratch capacity instead of panicking.
        let result = chunked_all_logits(
            tokens,
            start_pos,
            MAX_ALL_LOGITS_TOKENS,
            self.config.vocab_size,
            |chunk, chunk_start| self.try_forward_prefill_logits_all(chunk, chunk_start, state),
        );
        match result {
            Ok(logits) => logits,
            Err(e) => {
                hexagon_error!("forward_prefill_logits_all failed: {e}");
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
        // Read the torn flag under the device lock, like every other reader
        // (see `ensure_state_intact`). No caller holds the device guard here:
        // the session gate serializes callers and `try_truncate_kv` takes its
        // own guard and calls `check_kv_rewind_locked` instead.
        let device = self.lock_device();
        self.check_kv_rewind_locked(&device, state, len)
    }

    fn try_truncate_kv(
        &self,
        state: &mut InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        let mut device = self.lock_device();
        self.check_kv_rewind_locked(&device, state, len)?;
        // A failed quiesce leaves recurrent state torn and `state` untouched,
        // so recovery falls through to a full reset instead of reporting a
        // restored session. The cause is logged here because the error type
        // can only say "unsupported".
        self.truncate_kv_locked(&mut device, state, len)
            .map_err(|e| {
                hexagon_error!("KV rewind to {len} failed, DSP not quiesced: {e}");
                crate::kv_cache::KvRewindError::BackendUnsupported
            })
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        assert!(
            len <= state.seq_len,
            "truncate_kv({len}) exceeds seq_len {}",
            state.seq_len
        );
        let mut device = self.lock_device();
        if let Err(e) = self.truncate_kv_locked(&mut device, state, len) {
            hexagon_error!("state reset skipped, DSP not quiesced: {e}");
            // The infallible trait method still moves the position; the state
            // stays torn, so every later forward refuses until a reset lands.
            self.current_seq_len.store(len, Ordering::SeqCst);
            state.seq_len = len;
        }
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
        let mut device = self.lock_device();
        let mut fresh = InferenceState::from_config_capped(&self.config, compression, max_seq_len)?;
        fresh.lora = state.lora.clone();
        // Never zero state a timed-out batch may still write (see
        // `reset_recurrent_state`); on failure nothing has been mutated.
        self.reset_recurrent_state(&mut device)?;
        self.clear_decode_templates();
        self.current_seq_len.store(0, Ordering::SeqCst);
        *state = fresh;
        Ok(())
    }
}

impl RopeType {
    /// The HTP rope kernel's `mode` parameter (2 = NeoX, 0 = normal).
    pub(super) fn htp_mode(self) -> u32 {
        match self {
            RopeType::Neox => 2,
            RopeType::Norm => 0,
        }
    }
}

impl HexagonLfmModel {
    fn check_kv_rewind_locked(
        &self,
        _device: &HexagonDevice,
        state: &InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        if len > state.seq_len {
            return Err(crate::kv_cache::KvRewindError::OutOfBounds {
                requested: len,
                current: state.seq_len,
            });
        }
        if self.has_deltanet {
            state.check_truncate_to(len)?;
        }
        // Short-conv state lives on the device and advances in place with no
        // history ring, so a partial rewind would keep the discarded tail.
        // Full rewind (0) is served by `truncate_kv` zeroing the state.
        if len != state.seq_len && len != 0 && self.has_recurrent_layers() {
            return Err(crate::kv_cache::KvRewindError::BackendUnsupported);
        }
        // Torn recurrent state (see `state_torn`) is only recoverable by a full
        // reset, so every non-zero target is refused.
        if len != 0 && self.state_torn.load(Ordering::SeqCst) {
            return Err(crate::kv_cache::KvRewindError::BackendUnsupported);
        }
        Ok(())
    }

    fn clear_decode_templates(&self) {
        *self
            .decode_template
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *self
            .greedy_decode_template
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Zero the unified KV / recurrent-state buffer and clear the torn flag.
    ///
    /// A timed-out batch may still be writing the buffer, so this zeroes it
    /// only once the DSP has answered. Otherwise it marks the state torn,
    /// leaves the buffer alone and returns the quiesce error (a retry
    /// re-attempts the quiesce). Shared by `truncate_kv(0)` and `try_reset_kv`.
    fn reset_recurrent_state(&self, device: &mut HexagonDevice) -> Result<(), CeraError> {
        // A failed quiesce sets `state_torn` even for attention-only models
        // (unlike `mark_state_torn`); `try_reset_kv` is what heals it.
        if let Err(e) = device.queue_session_mut().quiesce() {
            self.state_torn.store(true, Ordering::SeqCst);
            return Err(e);
        }
        unsafe {
            std::ptr::write_bytes(self.kv_state_buf.as_mut_ptr(), 0, self.kv_state_buf.size());
        }
        self.kv_state_buf
            .flush_cpu_cache(0, self.kv_state_buf.size());
        self.state_torn.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Move the position to `len`, zeroing recurrent state on a full rewind.
    /// On error nothing about `state` or the position has changed.
    fn truncate_kv_locked(
        &self,
        device: &mut HexagonDevice,
        state: &mut InferenceState,
        len: usize,
    ) -> Result<(), CeraError> {
        if len == 0 && self.has_recurrent_layers() {
            self.reset_recurrent_state(device)?;
        }
        self.clear_decode_templates();
        self.current_seq_len.store(len, Ordering::SeqCst);
        state.seq_len = len;
        Ok(())
    }
}

#[cfg(test)]
mod constructor_golden_tests;
#[cfg(test)]
mod forward_golden_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod unit_tests;

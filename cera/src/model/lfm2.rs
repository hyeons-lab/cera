// LFM2 / LFM2.5 hybrid conv+attention model.

use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvPrefixCache, LayerState};
use crate::model::transformer::{self, FfnWeights};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
// DType's only remaining uses here are the aarch64 per-token GEMV dispatch
// (`forward_conv_block`'s Q4_0/Q8_0 checks). The prefill gates used to name dtypes
// directly too, but they now ask `transformer::batched_gemm_supports`, which takes
// the dtype as a parameter — so on x86_64 + `blas` this import became unused. Gate
// it to where it is actually referenced.
#[cfg(target_arch = "aarch64")]
use crate::tensor::DType;
use crate::turboquant;

// ── Pre-resolved weight reference ───────────────────────────────────────────

// The pre-resolved mmap weight reference is the arch-agnostic one from
// `transformer.rs`; LFM2 shares it (and the weight-plumbing helpers below) so the
// type and kernels have a single definition. `gpu_lfm2.rs` / `metal_lfm2.rs` keep
// referring to `lfm2::WeightRef` via this re-export.
pub(crate) use transformer::WeightRef;

/// The three SwiGLU projections of a dense feed-forward block.
#[derive(Debug, Clone)]
pub(crate) struct DenseFfnRefs {
    pub gate: WeightRef,
    pub up: WeightRef,
    pub down: WeightRef,
    #[allow(dead_code)]
    pub gate_up_f32: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
}

/// Weight references for one mixture-of-experts feed-forward block.
///
/// The GGUF stores each layer's experts stacked into rank-3 tensors; these are
/// pre-split into one ordinary 2D [`WeightRef`] per expert at load time (see
/// [`crate::gguf::GgufFile::tensor_meta_expert`]) so the existing GEMV kernels
/// run against an expert unchanged. `gate`, `up` and `down` are each
/// `n_expert` long and index-aligned.
#[derive(Debug, Clone)]
pub(crate) struct MoeFfnRefs {
    /// Routing parameters for this layer, copied out of
    /// [`crate::model::MoeConfig`] at load.
    ///
    /// Carried here rather than read from `ModelConfig::moe` at the call site so
    /// that having expert weights and having the numbers to drive them is one
    /// fact instead of two. The alternative left the hot path with an
    /// `if let Some(cfg)` whose `else` would silently skip the whole FFN and add
    /// the *previous* block's stale scratch into the residual.
    ///
    /// All three are flattened rather than held as a `MoeConfig`: that struct
    /// also carries the per-model `is_moe_layer` map, which is both meaningless
    /// inside a per-layer struct (this layer is routed by virtue of being an
    /// `FfnRefs::Moe`) and a heap allocation per routed layer to clone.
    pub n_expert: usize,
    /// Experts activated per token. See [`Self::n_expert`].
    pub n_expert_used: usize,
    /// Per-expert feed-forward width. See [`Self::n_expert`].
    pub expert_ff_len: usize,
    /// Router projection (`ffn_gate_inp.weight`, F32, `[hidden, n_expert]`)
    /// producing one logit per expert.
    pub router: WeightRef,
    /// Per-expert selection bias (`exp_probs_b.bias`, F32, `n_expert` long).
    ///
    /// Added to the gate probabilities **only** to pick the top-k experts; the
    /// weights those experts are then combined with are the *unbiased*
    /// probabilities. This is the DeepSeek-V3 convention llama.cpp implements
    /// ("leave probs unbiased as it's later used to get expert weights"), and
    /// conflating the two still yields fluent output, so it is pinned by the
    /// routing unit tests rather than left to inspection.
    pub exp_probs_b: Vec<f32>,
    pub gate: Vec<WeightRef>,
    pub up: Vec<WeightRef>,
    pub down: Vec<WeightRef>,
    #[allow(dead_code)]
    pub gate_up_f32: Vec<std::sync::Arc<std::sync::OnceLock<Vec<f32>>>>,
}

/// A layer's feed-forward block: one dense SwiGLU, or a routed expert set.
///
/// `lfm2moe` mixes both (its leading blocks are dense, the rest MoE),
/// so this is per-layer rather than per-model. Modelled as a sum type instead
/// of optional dense fields plus optional expert fields so that "exactly one of
/// the two is populated" is a type-level fact rather than a load-time
/// invariant the hot paths would have to re-check.
#[derive(Debug, Clone)]
pub(crate) enum FfnRefs {
    Dense(DenseFfnRefs),
    /// Boxed so the enum's size is set by the dense variant rather than by the
    /// larger MoE one, which every dense layer of every model would otherwise
    /// carry as padding.
    Moe(Box<MoeFfnRefs>),
}

impl FfnRefs {
    /// The dense projections, or an error naming the layer when it is MoE.
    ///
    /// Every backend now implements experts, so this is no longer a
    /// not-yet-supported path: it is the accessor a caller reaches for once it
    /// has established the layer is dense, and the error is what keeps a caller
    /// that has *not* from silently falling back onto the wrong weights. Both
    /// GPU loaders ask `GpuWeightSource::moe_refs` first for exactly that
    /// reason.
    pub fn dense(&self) -> Result<&DenseFfnRefs> {
        match self {
            Self::Dense(d) => Ok(d),
            Self::Moe(_) => bail!("expected a dense FFN block, found a mixture-of-experts block"),
        }
    }
}

/// Per-layer weight references for quantized tensors.
#[derive(Debug, Clone)]
pub(crate) struct LayerWeightRefs {
    pub ffn: FfnRefs,
    pub shortconv_in_proj: Option<WeightRef>,
    pub shortconv_out_proj: Option<WeightRef>,
    pub attn_q: Option<WeightRef>,
    pub attn_k: Option<WeightRef>,
    pub attn_v: Option<WeightRef>,
    pub attn_output: Option<WeightRef>,
    #[allow(dead_code)]
    pub qkv_f32: std::sync::Arc<std::sync::OnceLock<Vec<f32>>>,
    #[allow(dead_code, clippy::type_complexity)]
    pub qkv_repacked: std::sync::Arc<std::sync::OnceLock<Option<(Vec<u8>, Vec<f32>)>>>,
}

// ── LFM2 Model ─────────────────────────────────────────────────────────────

pub struct Lfm2Model {
    gguf: GgufFile,
    config: ModelConfig,
    // Pre-dequantized small F32 weights
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    conv_weights: Vec<Option<Vec<f32>>>,
    conv_weights_transposed: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs
    embd_ref: WeightRef,
    layer_refs: Vec<LayerWeightRefs>,
    /// Identifier passed into `KvPrefixCache::new`. CPU prefixes the
    /// caller-supplied id with `"cpu:"` so disk-cache files don't
    /// collide with Metal's f16-byte snapshots of the same model file
    /// (model_fingerprint doesn't include element width). Empty string
    /// when constructed via `from_gguf` (path-less case): warm cache
    /// still works, but disk-cache files for different path-less
    /// from_bytes loads of distinct models would namespace-collide —
    /// acceptable since `from_bytes` is documented as "testing".
    model_id: String,
    /// Two-tier prefix cache (warm in-memory + cold on-disk via
    /// FlatBuffers). Replaced wholesale by `Model::configure_cache`.
    /// Defaults to `KvCacheConfig::default()` (warm-only) at
    /// construction time so warm hits work without explicit config.
    prefix_cache: Mutex<KvPrefixCache>,
    /// Prefix-cache namespace tag for the KV mode this model's sessions use
    /// (`KvCompression::cache_tag`). `None` until `configure_kv_compression` runs;
    /// the empty string is the tag for the f32 default, which is why this is an
    /// `Option` rather than just `""`.
    ///
    /// The model owns this, but the mode is a *per-session* knob — so the two only
    /// stay consistent because configuration is first-call-wins. See
    /// `configure_kv_compression`.
    kv_cache_tag: Mutex<Option<String>>,
}

/// Choose the top-`n_used` experts and their combining weights.
///
/// `probs` are the sigmoid gate probabilities and `biases` the per-expert
/// selection bias (`exp_probs_b`). Experts are *ranked* by `probs + biases` but
/// *weighted* by `probs` alone, then renormalized to sum to 1. This is the DeepSeek-V3
/// convention llama.cpp implements in `build_moe_ffn`.
///
/// Split out of [`Lfm2Model::route_experts`] as a pure function purely so this
/// rule is directly testable: the bias affects rank only, which means the
/// returned weights are *not* necessarily descending, and a version that
/// weights by the biased score produces plausible text while being wrong.
fn select_experts(probs: &[f32], biases: &[f32], n_used: usize, selected: &mut Vec<(usize, f32)>) {
    selected.clear();
    let n_expert = probs.len().min(biases.len());
    let mut stack_biased = [0.0f32; 256];
    let heap_biased;
    let biased: &[f32] = if n_expert <= 256 {
        for i in 0..n_expert {
            stack_biased[i] = probs[i] + biases[i];
        }
        &stack_biased[..n_expert]
    } else {
        heap_biased = probs
            .iter()
            .zip(biases)
            .map(|(&p, &b)| p + b)
            .collect::<Vec<_>>();
        &heap_biased[..]
    };
    (0..n_used.min(n_expert)).for_each(|_| {
        let best = (0..n_expert)
            .filter(|e| !selected.iter().any(|(taken, _)| taken == e))
            .max_by(|&a, &b| {
                // `total_cmp`, not `partial_cmp`: a NaN logit would otherwise
                // make `max_by` silently return an arbitrary expert.
                biased[a]
                    .total_cmp(&biased[b])
                    // Ties go to the lower index, matching the stable order of
                    // `ggml_argsort_top_k`. `max_by` keeps the *last* maximum,
                    // so the reversed index comparison is what makes it the
                    // first.
                    .then(b.cmp(&a))
            });
        if let Some(e) = best {
            selected.push((e, probs[e]));
        }
    });

    // Renormalize the unbiased probabilities over the chosen experts, clamping
    // the divisor to f16's smallest positive normal (2^-14) exactly as
    // llama.cpp does so an all-but-zero gate cannot divide by zero. Spelled as
    // an exponent rather than the decimal expansion because 6.103515625e-5
    // carries more digits than an f32 literal keeps, which clippy rejects.
    const MIN_POSITIVE_F16: f32 = 1.0 / 16384.0;
    let denom = selected
        .iter()
        .map(|&(_, w)| w)
        .sum::<f32>()
        .max(MIN_POSITIVE_F16);
    selected.iter_mut().for_each(|(_, w)| *w /= denom);
}

/// Render a cache tag for the `KvCompressionConflict` message — the empty tag is
/// the f32 default, which reads better spelled out.
fn describe_cache_tag(tag: &str) -> String {
    if tag.is_empty() {
        "f32".to_string()
    } else {
        tag.trim_end_matches(':').to_string()
    }
}

/// Range-check `lfm2.shortconv.l_cache` at the point it enters from GGUF
/// metadata, so every downstream consumer can trust it.
///
/// The GPU short-conv kernels stage a channel's weights and rolling state in
/// fixed-size registers sized for `kernel_size <= 4`
/// (`conv1d_fused_batch.{wgsl,metal,slang}`), and `d_conv = kernel_size - 1`
/// underflows for 0. Both used to be handled per-kernel: the batched shader
/// silently returned on out-of-range params, which turned a malformed model into
/// wrong prefill logits rather than an error, and the CPU cache asserted the
/// lower bound only. Checking the range once here means a bad value is a load
/// failure with a clear message, and the kernels can index on the invariant.
///
/// Every shipped LFM2 GGUF sets `l_cache = 3`.
fn validate_conv_kernel_size(v: Option<usize>) -> anyhow::Result<Option<usize>> {
    if let Some(k) = v {
        anyhow::ensure!(
            (2..=4).contains(&k),
            "lfm2.shortconv.l_cache must be in 2..=4, got {k}"
        );
    }
    Ok(v)
}

impl Lfm2Model {
    /// Prefix-cache namespace: the `"cpu:"` backend prefix, the KV-mode tag, and
    /// the model id. The backend prefix keeps CPU entries away from wgpu's and
    /// Metal's (their state shapes differ even where the byte format matches); the
    /// mode tag keeps f32, f16, and each TurboQuant configuration apart.
    fn cache_namespace(&self) -> String {
        let tag = self.kv_cache_tag.lock().expect("kv_cache_tag poisoned");
        Self::namespace_for(tag.as_deref().unwrap_or(""), &self.model_id)
    }

    /// The namespace string itself, taking the tag by value so a caller already
    /// holding the tag lock doesn't have to re-acquire it (see
    /// `configure_kv_compression`, which updates the tag and the cache atomically).
    fn namespace_for(tag: &str, model_id: &str) -> String {
        format!("cpu:{tag}{model_id}")
    }

    /// Construct without a model identifier. Equivalent to
    /// `from_gguf_with_id(gguf, context_size, "")`. Warm prefix cache
    /// works after `Model::configure_cache`; disk cache (when
    /// configured) would namespace-collide between path-less loads of
    /// different models, which is acceptable for the `from_bytes`
    /// testing use case the doc calls out.
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF
    /// file path) used to namespace prefix-cache entries. The id is
    /// prefixed with `"cpu:"` before being fed to `model_fingerprint`
    /// so CPU and Metal can share a `--cache-dir` without their
    /// disk-cache files (different element widths: CPU=f32, Metal=f16)
    /// colliding.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        ensure!(context_size > 0, "context_size must be > 0");
        // `lfm2moe` is the same graph as `lfm2` with experts in the FFN slot, so
        // it shares this loader, but its metadata keys are namespaced under its
        // own arch name (`lfm2moe.block_count`, not `lfm2.block_count`), so the
        // prefix has to follow the file rather than be hardcoded.
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("lfm2")
            .to_string();
        ensure!(
            arch == "lfm2" || arch == "lfm2moe",
            "Lfm2Model: unsupported architecture {arch}"
        );
        let prefix = arch.as_str();

        let n_layers =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .with_context(|| format!("missing {prefix}.block_count"))? as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .with_context(|| format!("missing {prefix}.embedding_length"))?
            as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .with_context(|| format!("missing {prefix}.feed_forward_length"))?
            as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .with_context(|| format!("missing {prefix}.attention.head_count"))?
            as usize;
        let vocab_size =
            gguf.get_u32(&format!("{prefix}.vocab_size"))
                .with_context(|| format!("missing {prefix}.vocab_size"))? as usize;
        // Cap the model's max_seq_len by the user's requested context_size so
        // KV cache pre-allocation in `InferenceState::from_config_with_compression`
        // matches the actual budget. Mirrors the pattern used by metal_lfm2 and
        // gpu_lfm2.
        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let conv_kernel_size = validate_conv_kernel_size(
            gguf.get_u32(&format!("{prefix}.shortconv.l_cache"))
                .map(|v| v as usize),
        )?;

        // Per-layer KV head counts
        let kv_heads_array = gguf
            .get_i32_array(&format!("{prefix}.attention.head_count_kv"))
            .with_context(|| format!("missing {prefix}.attention.head_count_kv"))?;

        // Validate kv_heads_array length matches n_layers
        anyhow::ensure!(
            kv_heads_array.len() >= n_layers,
            "head_count_kv array length ({}) < block_count ({n_layers})",
            kv_heads_array.len()
        );

        // Detect block types from tensor presence
        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for (i, &kv_heads) in kv_heads_array.iter().enumerate().take(n_layers) {
            let is_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
            if is_attn {
                let n_kv = kv_heads as usize;
                anyhow::ensure!(
                    n_kv > 0 && n_heads.is_multiple_of(n_kv),
                    "layer {i}: n_kv_heads ({n_kv}) must be > 0 and divide n_heads ({n_heads})"
                );
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(n_kv);
            } else {
                block_types.push(BlockType::GatedConv);
                kv_heads_per_layer.push(0);
            }
        }

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0);

        // Which layers route through experts. Read from tensor presence, the
        // same way block types are, rather than from `leading_dense_block_count`.
        // A metadata/tensor disagreement then fails at weight resolution with
        // a missing-tensor name instead of silently running the wrong FFN.
        let moe_layers: Vec<bool> = (0..n_layers)
            .map(|i| {
                gguf.tensors
                    .contains_key(&format!("blk.{i}.ffn_gate_exps.weight"))
            })
            .collect();

        let moe = if moe_layers.iter().any(|&is_moe| is_moe) {
            // Sigmoid (2) is the only gating function implemented. Softmax (1)
            // would load and generate fluent-looking text off wrong weights, so
            // reject it rather than default to it.
            let gating_func = gguf
                .get_u32(&format!("{prefix}.expert_gating_func"))
                .with_context(|| format!("missing {prefix}.expert_gating_func"))?;
            ensure!(
                gating_func == 2,
                "{prefix}: expert_gating_func {gating_func} is not supported (only 2 = sigmoid)"
            );
            let n_expert = gguf
                .get_u32(&format!("{prefix}.expert_count"))
                .with_context(|| format!("missing {prefix}.expert_count"))?
                as usize;
            let n_expert_used = gguf
                .get_u32(&format!("{prefix}.expert_used_count"))
                .with_context(|| format!("missing {prefix}.expert_used_count"))?
                as usize;
            let expert_ff_len = gguf
                .get_u32(&format!("{prefix}.expert_feed_forward_length"))
                .with_context(|| format!("missing {prefix}.expert_feed_forward_length"))?
                as usize;
            ensure!(
                n_expert_used > 0 && n_expert_used <= n_expert,
                "{prefix}: expert_used_count ({n_expert_used}) must be in 1..={n_expert}"
            );
            // The expert GEMVs borrow the dense FFN's `gate`/`up` scratch, which
            // `InferenceState` allocates at `intermediate_size` and never grows.
            // A file whose expert width exceeded the dense width would index
            // past it mid-forward, so reject it at load with a name instead.
            ensure!(
                expert_ff_len <= intermediate_size,
                "{prefix}: expert_feed_forward_length ({expert_ff_len}) exceeds \
                 feed_forward_length ({intermediate_size}), which sizes the FFN scratch"
            );
            // The Q8_0 requantize of the SwiGLU product computes its block count
            // as `expert_ff_len / 32`; a non-multiple would truncate and leave
            // the tail block holding stale scratch rather than this expert's
            // activations.
            ensure!(
                expert_ff_len > 0 && expert_ff_len.is_multiple_of(32),
                "{prefix}: expert_feed_forward_length ({expert_ff_len}) must be a positive \
                 multiple of 32"
            );
            Some(crate::model::MoeConfig {
                n_expert,
                n_expert_used,
                expert_ff_len,
                is_moe_layer: moe_layers.clone(),
            })
        } else {
            None
        };

        let config = ModelConfig {
            // The file's own arch, not a flattened "lfm2": `model_fingerprint`
            // hashes this into the prefix-cache namespace, and an lfm2moe cache
            // entry must not be reachable by an lfm2 model of the same shape.
            architecture: arch.clone(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim: hidden_size / n_heads,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types: block_types.clone(),
            conv_kernel_size,
            kv_heads_per_layer: kv_heads_per_layer.clone(),
            scalars: ScalarMultipliers::default(),
            moe: moe.clone(),
        };

        // Pre-extract small F32 weights
        let output_norm_weight = gguf.get_tensor("token_embd_norm.weight")?.to_f32_vec();

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut conv_weights = Vec::with_capacity(n_layers);
        let mut conv_weights_transposed = Vec::with_capacity(n_layers);

        for (i, bt) in block_types.iter().enumerate() {
            attn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.attn_norm.weight"))?
                    .to_f32_vec(),
            );
            ffn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight"))?
                    .to_f32_vec(),
            );

            if *bt == BlockType::Attention {
                attn_q_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_q_norm.weight"))?
                        .to_f32_vec(),
                ));
                attn_k_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_k_norm.weight"))?
                        .to_f32_vec(),
                ));
                conv_weights.push(None);
                conv_weights_transposed.push(None);
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                let w = gguf
                    .get_tensor(&format!("blk.{i}.shortconv.conv.weight"))?
                    .to_f32_vec();
                let kernel_size = config.conv_kernel_size.unwrap_or(3);
                let hs = config.hidden_size;
                let mut transposed = vec![0.0f32; kernel_size * hs];
                for ch in 0..hs {
                    for k in 0..kernel_size {
                        transposed[k * hs + ch] = w[ch * kernel_size + k];
                    }
                }
                conv_weights.push(Some(w));
                conv_weights_transposed.push(Some(transposed));
            }
        }

        // Pre-resolve quantized weight references
        let embd_ref = Self::resolve_weight(&gguf, "token_embd.weight")?;

        let mut layer_refs = Vec::with_capacity(n_layers);
        for (i, bt) in block_types.iter().enumerate() {
            // `.with_repack` on every projection weight (all hit the batched
            // prefill GEMM at `n > 1`); token_embd is excluded above.
            let ffn = if moe_layers[i] {
                let moe_cfg = moe
                    .as_ref()
                    .context("layer has expert tensors but no MoE metadata")?;
                let n_expert = moe_cfg.n_expert;

                // Experts are deliberately *not* `.with_repack`ed. The repack
                // exists to feed the batched prefill GEMM, and it allocates a
                // full second copy of the weight (x86_64 only). Experts are the
                // bulk of the parameters here, and repacking all 32 per tensor
                // would roughly double resident memory for the whole model,
                // while each expert sees only the tokens routed to it, so its
                // GEMM is a fraction of the width the repack was tuned for.
                // Whether it pays back is a measurement, not a default.
                let expert_refs = |suffix: &str| -> Result<Vec<WeightRef>> {
                    let name = format!("blk.{i}.{suffix}");
                    (0..n_expert)
                        .map(|e| Self::resolve_expert_weight(&gguf, &name, e))
                        .collect()
                };

                let exp_probs_b = gguf
                    .get_tensor(&format!("blk.{i}.exp_probs_b.bias"))?
                    .to_f32_vec();
                ensure!(
                    exp_probs_b.len() == n_expert,
                    "layer {i}: exp_probs_b has {} entries, expected {n_expert}",
                    exp_probs_b.len()
                );

                FfnRefs::Moe(Box::new(MoeFfnRefs {
                    n_expert,
                    n_expert_used: moe_cfg.n_expert_used,
                    expert_ff_len: moe_cfg.expert_ff_len,
                    router: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate_inp.weight"))?,
                    exp_probs_b,
                    gate: expert_refs("ffn_gate_exps.weight")?,
                    up: expert_refs("ffn_up_exps.weight")?,
                    down: expert_refs("ffn_down_exps.weight")?,
                    gate_up_f32: (0..n_expert)
                        .map(|_| std::sync::Arc::new(std::sync::OnceLock::new()))
                        .collect(),
                }))
            } else {
                FfnRefs::Dense(DenseFfnRefs {
                    gate: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?
                        .with_repack(&gguf),
                    up: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?
                        .with_repack(&gguf),
                    down: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?
                        .with_repack(&gguf),
                    gate_up_f32: std::sync::Arc::new(std::sync::OnceLock::new()),
                })
            };

            let (shortconv_in_proj, shortconv_out_proj, attn_q, attn_k, attn_v, attn_output) =
                if *bt == BlockType::GatedConv {
                    (
                        Some(
                            Self::resolve_weight(
                                &gguf,
                                &format!("blk.{i}.shortconv.in_proj.weight"),
                            )?
                            .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(
                                &gguf,
                                &format!("blk.{i}.shortconv.out_proj.weight"),
                            )?
                            .with_repack(&gguf),
                        ),
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        None,
                        None,
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?
                                .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?
                                .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))?
                                .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_output.weight"))?
                                .with_repack(&gguf),
                        ),
                    )
                };

            layer_refs.push(LayerWeightRefs {
                ffn,
                shortconv_in_proj,
                shortconv_out_proj,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                qkv_f32: std::sync::Arc::new(std::sync::OnceLock::new()),
                qkv_repacked: std::sync::Arc::new(std::sync::OnceLock::new()),
            });
        }

        let prefix_cache = Mutex::new(KvPrefixCache::new(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &format!("cpu:{model_id}"),
        ));

        Ok(Self {
            gguf,
            config,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights,
            conv_weights_transposed,
            embd_ref,
            layer_refs,
            model_id,
            prefix_cache,
            kv_cache_tag: Mutex::new(None),
        })
    }

    /// Resolve a tensor name to a pre-computed byte range in the mmap.
    /// Thin wrapper over the shared `transformer::resolve_weight`.
    fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<WeightRef> {
        transformer::resolve_weight(gguf, name)
    }

    /// Resolve expert `expert` of a stacked MoE tensor to a 2D weight ref.
    fn resolve_expert_weight(gguf: &GgufFile, name: &str, expert: usize) -> Result<WeightRef> {
        transformer::resolve_expert_weight(gguf, name, expert)
    }

    /// Pick the experts for one token and their combining weights, leaving both
    /// in `state.scratch.moe_selected`.
    ///
    /// Mirrors llama.cpp's `build_moe_ffn` for `expert_gating_func = 2`:
    ///
    /// 1. `probs = sigmoid(router · x)`
    /// 2. select the top `n_expert_used` by `probs + exp_probs_b`
    /// 3. weight them by the **unbiased** `probs`, renormalized to sum to 1
    ///
    /// Step 3 is the part that is easy to get wrong: `exp_probs_b` is a
    /// DeepSeek-V3 style load-balancing bias that steers *selection* only. Using
    /// the biased score as the combining weight changes every output subtly and
    /// still reads as fluent text, so the `moe_routing_tests` module pins this
    /// against llama.cpp's own values rather than trusting inspection.
    fn route_experts(
        &self,
        layer: usize,
        moe: &MoeFfnRefs,
        lora: Option<&crate::lora::LoraAdapterWeights>,
        ffn_input: &[f32],
        state: &mut InferenceState,
    ) {
        let n_expert = moe.n_expert;
        let n_used = moe.n_expert_used;

        state.scratch.moe_probs.resize(n_expert, 0.0);
        transformer::gemv(
            &self.gguf,
            &moe.router,
            ffn_input,
            &mut state.scratch.moe_probs[..n_expert],
        );
        // The router is a LoRA target in its own right (llama.cpp builds it with
        // `build_lora_mm`), and it is adapted on the *logits*, before the
        // sigmoid. An adapter that moved a token across a selection boundary
        // would otherwise be silently ignored while its expert deltas applied.
        if let Some(lora) = lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGateInp)
        {
            crate::lora::apply_decode(
                t,
                ffn_input,
                &mut state.scratch.moe_probs[..n_expert],
                &mut state.scratch.lora_tmp,
            );
        }
        state.scratch.moe_probs[..n_expert]
            .iter_mut()
            .for_each(|p| *p = 1.0 / (1.0 + (-*p).exp()));

        select_experts(
            &state.scratch.moe_probs[..n_expert],
            &moe.exp_probs_b,
            n_used,
            &mut state.scratch.moe_selected,
        );
    }

    /// Routed feed-forward for one MoE layer, writing the combined expert
    /// output into `state.scratch.out[..hidden_size]`.
    ///
    /// On aarch64 the caller must have quantized `ffn_input` into
    /// `state.scratch.q8_*` already, matching the dense
    /// [`transformer::forward_ffn_block`] contract; each expert's
    /// down-projection re-quantizes its own SwiGLU product.
    ///
    /// The body deliberately duplicates the ~60 lines of
    /// [`transformer::forward_ffn_block`] (the same gate/up GEMV, SwiGLU,
    /// requantize, down GEMV, LoRA sequence) rather than calling it. Sharing
    /// would mean parameterizing the dense helper over per-expert weights, a
    /// per-expert LoRA lookup and an output buffer that is accumulated rather
    /// than written, which puts three new branches in the FFN hot path of every
    /// dense model cera runs so that one architecture can reuse it. Nothing
    /// pins the two copies to each other, so a change to the dense block's
    /// arithmetic has to be mirrored here by hand; if a third routed
    /// architecture lands, that trade stops paying and this should be extracted.
    fn forward_moe_ffn(
        &self,
        layer: usize,
        moe: &MoeFfnRefs,
        hidden_size: usize,
        ffn_input: &[f32],
        state: &mut InferenceState,
    ) {
        let ff = moe.expert_ff_len;
        // Cloned out of `state` up front: the `Arc` bumps a refcount, and the
        // alternative is holding a borrow of `state` across the GEMVs that need
        // it mutably. Same shape as `transformer::forward_ffn_block`.
        let lora = state.lora.clone();
        self.route_experts(layer, moe, lora.as_deref(), ffn_input, state);

        state.scratch.moe_expert_out.resize(hidden_size, 0.0);
        state.scratch.out[..hidden_size].fill(0.0);

        // `moe_selected` is scratch that the expert GEMVs below also borrow
        // from `state`, so take the routing decision out first. It is
        // `n_expert_used` pairs (4 here), not per-expert data.
        let selected = std::mem::take(&mut state.scratch.moe_selected);

        for &(expert, weight) in &selected {
            #[cfg(target_arch = "aarch64")]
            {
                let gate_data = self.weight_data(&moe.gate[expert]);
                let up_data = self.weight_data(&moe.up[expert]);
                cpu::gemv_q4_0_fused2_with_q8(
                    gate_data,
                    up_data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.gate[..ff],
                    &mut state.scratch.up[..ff],
                    ff,
                    hidden_size,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                transformer::gemv(
                    &self.gguf,
                    &moe.gate[expert],
                    ffn_input,
                    &mut state.scratch.gate[..ff],
                );
                transformer::gemv(
                    &self.gguf,
                    &moe.up[expert],
                    ffn_input,
                    &mut state.scratch.up[..ff],
                );
            }

            // Per-expert LoRA on gate/up, before the SwiGLU mul reads both. The
            // adapter's factors are indexed by the *same* expert id as the base
            // weight, matching llama.cpp's `build_lora_mm_id`, which feeds one
            // `ids` tensor to the base and both factors alike.
            if let Some(lora) = &lora {
                if let Some(t) =
                    lora.get_expert(layer, crate::lora::LoraTarget::FfnGateExps, expert)
                {
                    crate::lora::apply_decode(
                        t,
                        ffn_input,
                        &mut state.scratch.gate[..ff],
                        &mut state.scratch.lora_tmp,
                    );
                }
                if let Some(t) = lora.get_expert(layer, crate::lora::LoraTarget::FfnUpExps, expert)
                {
                    crate::lora::apply_decode(
                        t,
                        ffn_input,
                        &mut state.scratch.up[..ff],
                        &mut state.scratch.lora_tmp,
                    );
                }
            }

            cpu::silu_mul_inplace(&mut state.scratch.gate[..ff], &state.scratch.up[..ff]);

            #[cfg(target_arch = "aarch64")]
            {
                // The down projection consumes the SwiGLU product, not the
                // layer input, so it needs its own Q8_0 quantization into down buffers.
                let nb = ff / 32;
                state.scratch.q8_scales_down.resize(nb, 0.0);
                state.scratch.q8_quants_down.resize(ff, 0);
                unsafe {
                    crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                        &state.scratch.gate[..ff],
                        &mut state.scratch.q8_scales_down,
                        &mut state.scratch.q8_quants_down,
                    );
                }
                let down_data = self.weight_data(&moe.down[expert]);
                cpu::gemv_q4_0_with_q8(
                    down_data,
                    &state.scratch.q8_scales_down,
                    &state.scratch.q8_quants_down,
                    &mut state.scratch.moe_expert_out[..hidden_size],
                    hidden_size,
                    ff,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            transformer::gemv(
                &self.gguf,
                &moe.down[expert],
                &state.scratch.gate[..ff],
                &mut state.scratch.moe_expert_out[..hidden_size],
            );

            // LoRA on this expert's down projection. Its input is the SwiGLU
            // product in `gate`, not the layer input, so it must read `gate`
            // *before* the accumulate below consumes `moe_expert_out`.
            if let Some(lora) = &lora
                && let Some(t) =
                    lora.get_expert(layer, crate::lora::LoraTarget::FfnDownExps, expert)
            {
                crate::lora::apply_decode(
                    t,
                    &state.scratch.gate[..ff],
                    &mut state.scratch.moe_expert_out[..hidden_size],
                    &mut state.scratch.lora_tmp,
                );
            }

            let (out, expert_out) = (
                &mut state.scratch.out[..hidden_size],
                &state.scratch.moe_expert_out[..hidden_size],
            );
            out.iter_mut()
                .zip(expert_out)
                .for_each(|(acc, &v)| *acc += weight * v);
        }

        state.scratch.moe_selected = selected;
    }

    /// Prefill feed-forward for a MoE layer: route and run every token
    /// individually, reading `ffn_input` and writing `ffn_out` in the caller's
    /// column-major (`hs × n`) layout.
    ///
    /// Deliberately unbatched. The dense path batches because all `n` tokens
    /// share one weight matrix; here each token picks its own 4-of-32 experts,
    /// so a batched GEMM would first have to group tokens by expert and scatter
    /// the results back. That grouping is the next optimization (plan
    /// 000312-01, phase 1 step 6) and is kept separate from making the
    /// arithmetic right, so the batched version has a reference to match.
    #[allow(clippy::too_many_arguments)]
    fn prefill_moe_ffn(
        &self,
        layer: usize,
        moe: &MoeFfnRefs,
        hs: usize,
        n: usize,
        ffn_input: &[f32],
        ffn_out: &mut [f32],
        col: &mut [f32],
        state: &mut InferenceState,
    ) {
        if n <= 1 || state.lora.is_some() {
            for j in 0..n {
                (0..hs).for_each(|i| col[i] = ffn_input[i * n + j]);

                #[cfg(target_arch = "aarch64")]
                Self::quantize_to_scratch(col, state);

                self.forward_moe_ffn(layer, moe, hs, col, state);

                (0..hs).for_each(|i| ffn_out[i * n + j] = state.scratch.out[i]);
            }
            return;
        }

        let n_expert = moe.n_expert;
        let n_used = moe.n_expert_used;
        let ff = moe.expert_ff_len;

        let mut all_router_logits = vec![0.0f32; n * n_expert];
        #[cfg(feature = "blas")]
        {
            transformer::try_blas_prefill_gemm_rowmajor(
                &self.gguf,
                &moe.router,
                ffn_input,
                &mut all_router_logits,
                n,
                n_expert,
                hs,
            );
        }
        #[cfg(not(feature = "blas"))]
        {
            for j in 0..n {
                let x = &ffn_input[j * hs..(j + 1) * hs];
                let out_slice = &mut all_router_logits[j * n_expert..(j + 1) * n_expert];
                self.gemv(&moe.router, x, out_slice);
            }
        }

        let mut selected = Vec::with_capacity(n_used);
        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];

        for j in 0..n {
            let router_probs = &mut all_router_logits[j * n_expert..(j + 1) * n_expert];
            router_probs
                .iter_mut()
                .for_each(|p| *p = 1.0 / (1.0 + (-*p).exp()));

            select_experts(router_probs, &moe.exp_probs_b, n_used, &mut selected);

            for &(e, weight) in &selected {
                if weight > 0.0 {
                    expert_assignments[e].push((j, weight));
                }
            }
        }

        ffn_out[..hs * n].fill(0.0);

        #[cfg(feature = "blas")]
        {
            use crate::par::{IntoParallelRefIterator, ParallelIterator};

            struct ExpertTask {
                expert: usize,
                assigned: Vec<(usize, f32)>,
            }

            let active_tasks: Vec<ExpertTask> = (0..n_expert)
                .filter(|&e| !expert_assignments[e].is_empty())
                .map(|e| ExpertTask {
                    expert: e,
                    assigned: expert_assignments[e].clone(),
                })
                .collect();

            let expert_results: Vec<(usize, Vec<f32>)> = active_tasks
                .par_iter()
                .map(|task| {
                    let e = task.expert;
                    let assigned = &task.assigned;
                    let k_e = assigned.len();

                    let mut exp_in = vec![0.0f32; k_e * hs];
                    for (slot, &(token_j, _)) in assigned.iter().enumerate() {
                        exp_in[slot * hs..(slot + 1) * hs]
                            .copy_from_slice(&ffn_input[token_j * hs..(token_j + 1) * hs]);
                    }

                    let gate_up = moe.gate_up_f32[e].get_or_init(|| {
                        let g_f32 = transformer::get_dequantized_f32(&self.gguf, &moe.gate[e]);
                        let u_f32 = transformer::get_dequantized_f32(&self.gguf, &moe.up[e]);
                        let mut fused = vec![0.0f32; 2 * ff * hs];
                        fused[..ff * hs].copy_from_slice(g_f32);
                        fused[ff * hs..].copy_from_slice(u_f32);
                        fused
                    });

                    let mut exp_gate_up = vec![0.0f32; k_e * 2 * ff];
                    let mut exp_gate = vec![0.0f32; k_e * ff];
                    let mut exp_down = vec![0.0f32; k_e * hs];

                    crate::backend::blas::sgemm_rowmajor_nt(
                        k_e,
                        2 * ff,
                        hs,
                        &exp_in,
                        gate_up,
                        &mut exp_gate_up,
                    );

                    for t in 0..k_e {
                        let gate_slice = &mut exp_gate[t * ff..(t + 1) * ff];
                        let g_src = &exp_gate_up[t * 2 * ff..t * 2 * ff + ff];
                        let u_src = &exp_gate_up[t * 2 * ff + ff..(t + 1) * 2 * ff];
                        gate_slice.copy_from_slice(g_src);
                        cpu::silu_mul_inplace(gate_slice, u_src);
                    }

                    transformer::try_blas_prefill_gemm_rowmajor(
                        &self.gguf,
                        &moe.down[e],
                        &exp_gate,
                        &mut exp_down,
                        k_e,
                        hs,
                        ff,
                    );

                    (e, exp_down)
                })
                .collect();

            for (e, exp_down) in expert_results {
                let assigned = &expert_assignments[e];
                for (slot, &(token_j, weight)) in assigned.iter().enumerate() {
                    let ed = &exp_down[slot * hs..(slot + 1) * hs];
                    let out_tok = &mut ffn_out[token_j * hs..(token_j + 1) * hs];

                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use core::arch::aarch64::*;
                        let n_chunks = hs / 4;
                        let ed_ptr = ed.as_ptr();
                        let out_ptr = out_tok.as_mut_ptr();
                        let w_vec = vdupq_n_f32(weight);
                        for i in 0..n_chunks {
                            let ed_v = vld1q_f32(ed_ptr.add(i * 4));
                            let out_v = vld1q_f32(out_ptr.add(i * 4));
                            let res_v = vfmaq_f32(out_v, ed_v, w_vec);
                            vst1q_f32(out_ptr.add(i * 4), res_v);
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..hs {
                        out_tok[i] += weight * ed[i];
                    }
                }
            }
        }

        #[cfg(not(feature = "blas"))]
        {
            let mut exp_gate = vec![0.0f32; ff];
            let mut exp_up = vec![0.0f32; ff];
            let mut exp_down = vec![0.0f32; hs];

            #[cfg(target_arch = "aarch64")]
            let mut q8_scales_down = vec![0.0f32; ff / 32];
            #[cfg(target_arch = "aarch64")]
            let mut q8_quants_down = vec![0i8; ff];

            for (e, assigned) in expert_assignments.iter().enumerate().take(n_expert) {
                if assigned.is_empty() {
                    continue;
                }

                for &(token_j, weight) in assigned {
                    let tok_in = &ffn_input[token_j * hs..(token_j + 1) * hs];

                    #[cfg(target_arch = "aarch64")]
                    {
                        Self::quantize_to_scratch(tok_in, state);
                        let gate_data = self.weight_data(&moe.gate[e]);
                        let up_data = self.weight_data(&moe.up[e]);
                        cpu::gemv_q4_0_fused2_with_q8(
                            gate_data,
                            up_data,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut exp_gate[..ff],
                            &mut exp_up[..ff],
                            ff,
                            hs,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.gemv(&moe.gate[e], tok_in, &mut exp_gate[..ff]);
                        self.gemv(&moe.up[e], tok_in, &mut exp_up[..ff]);
                    }

                    cpu::silu_mul_inplace(&mut exp_gate[..ff], &exp_up[..ff]);

                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                                &exp_gate[..ff],
                                &mut q8_scales_down,
                                &mut q8_quants_down,
                            );
                        }
                        let down_data = self.weight_data(&moe.down[e]);
                        cpu::gemv_q4_0_with_q8(
                            down_data,
                            &q8_scales_down,
                            &q8_quants_down,
                            &mut exp_down[..hs],
                            hs,
                            ff,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    self.gemv(&moe.down[e], &exp_gate[..ff], &mut exp_down[..hs]);

                    let out_tok = &mut ffn_out[token_j * hs..(token_j + 1) * hs];
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use core::arch::aarch64::*;
                        let n_chunks = hs / 4;
                        let ed_ptr = exp_down.as_ptr();
                        let out_ptr = out_tok.as_mut_ptr();
                        let w_vec = vdupq_n_f32(weight);
                        for i in 0..n_chunks {
                            let ed_v = vld1q_f32(ed_ptr.add(i * 4));
                            let out_v = vld1q_f32(out_ptr.add(i * 4));
                            let res_v = vfmaq_f32(out_v, ed_v, w_vec);
                            vst1q_f32(out_ptr.add(i * 4), res_v);
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..hs {
                        out_tok[i] += weight * exp_down[i];
                    }
                }
            }
        }
    }

    // ── Public accessors for GPU model construction ───────────────────────

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    pub fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }

    pub fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }

    pub fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }

    pub fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_norm_weights[layer].as_deref()
    }

    pub fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights[layer].as_deref()
    }

    pub fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.conv_weights[layer].as_deref()
    }

    /// Dequantize a token embedding row to f32.
    pub fn dequantize_embedding(&self, token_id: usize) -> Vec<f32> {
        self.dequantize_row(&self.embd_ref, token_id)
    }

    /// Conv in_proj GEMV for a layer.
    pub fn conv_in_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_in_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// Conv out_proj GEMV for a layer.
    pub fn conv_out_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_out_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// FFN gate GEMV for a layer. Errors on a MoE layer, whose gate weight is
    /// per-expert and only meaningful after routing.
    pub fn ffn_gate_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.gemv(&self.layer_refs[layer].ffn.dense()?.gate, x, y);
        Ok(())
    }

    /// FFN up GEMV for a layer. Errors on a MoE layer (see
    /// [`Self::ffn_gate_gemv`]).
    pub fn ffn_up_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.gemv(&self.layer_refs[layer].ffn.dense()?.up, x, y);
        Ok(())
    }

    /// FFN down GEMV for a layer. Errors on a MoE layer (see
    /// [`Self::ffn_gate_gemv`]).
    pub fn ffn_down_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.gemv(&self.layer_refs[layer].ffn.dense()?.down, x, y);
        Ok(())
    }

    /// Get raw weight bytes for a WeightRef (for GPU quantized upload).
    #[allow(dead_code)] // used by metal_lfm2/gpu_lfm2 behind feature gates
    pub(crate) fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        self.weight_data(wref)
    }

    // Full-matrix dequant lives in `transformer::dequantize_weight`; the LFM2
    // `GpuWeightSource` impl delegates to it. (The old inherent duplicate +
    // `dequantize_row_into_slice` helper were removed — single implementation.)

    /// Access the per-layer weight refs (for GPU model construction).
    #[allow(dead_code)]
    pub(crate) fn layer_refs(&self) -> &[LayerWeightRefs] {
        &self.layer_refs
    }

    /// Access the embedding weight ref.
    #[allow(dead_code)]
    pub(crate) fn embd_ref(&self) -> &WeightRef {
        &self.embd_ref
    }

    // ── Internal methods ────────────────────────────────────────────────

    /// Get the raw bytes for a pre-resolved weight.
    #[inline]
    fn weight_data(&self, wref: &WeightRef) -> &[u8] {
        transformer::weight_data(&self.gguf, wref)
    }

    /// GEMV dispatch without scratch buffers (shared `transformer::gemv`).
    fn gemv(&self, wref: &WeightRef, x: &[f32], y: &mut [f32]) {
        transformer::gemv(&self.gguf, wref, x, y);
    }

    /// GEMV with pre-quantized Q8_0 input (shared `transformer::gemv_preq`).
    #[cfg(target_arch = "aarch64")]
    fn gemv_preq(&self, wref: &WeightRef, x_f32: &[f32], q8s: &[f32], q8q: &[i8], y: &mut [f32]) {
        transformer::gemv_preq(&self.gguf, wref, x_f32, q8s, q8q, y);
    }

    // The batched-prefill GEMM helpers (`try_blas_prefill_gemm`, `gemm_preq`,
    // `quantize_columns`) are shared with the dense-transformer path and now
    // live as free functions in `transformer` — call sites use
    // `transformer::{try_blas_prefill_gemm, gemm_preq, quantize_columns}`.

    /// Quantize x to Q8_0 into scratch buffers (shared
    /// `transformer::quantize_to_scratch`).
    #[cfg(target_arch = "aarch64")]
    fn quantize_to_scratch(x: &[f32], state: &mut InferenceState) {
        transformer::quantize_to_scratch(x, state);
    }

    /// Dequantize a single row from a quantized matrix (for embedding lookup).
    fn dequantize_row(&self, wref: &WeightRef, row_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; wref.k];
        self.dequantize_row_into(wref, row_idx, &mut out);
        out
    }

    /// Dequantize a single row into `out` (shared
    /// `transformer::dequantize_row_into`).
    fn dequantize_row_into(&self, wref: &WeightRef, row_idx: usize, out: &mut [f32]) {
        transformer::dequantize_row_into(&self.gguf, wref, row_idx, out);
    }

    /// Process a single conv (recurrent) block using pre-allocated scratch buffers.
    fn forward_conv_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let hidden_size = self.config.hidden_size;
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
        let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
        let conv_weight = self.conv_weights[layer].as_ref().unwrap();

        // Cloned once (cheap Arc bump) so the adapter can be read while the base
        // scratch buffers stay mutably borrowed — same pattern as
        // `forward_attn_block`.
        let lora = state.lora.clone();

        // in_proj: hidden → 3*hidden (uses pre-quantized Q8_0 data when available)
        let proj = &mut state.scratch.conv_proj[..3 * hidden_size];
        #[cfg(target_arch = "aarch64")]
        if in_proj.dtype == DType::Q4_0 || in_proj.dtype == DType::Q8_0 {
            let data = self.weight_data(in_proj);
            if in_proj.dtype == DType::Q4_0 {
                cpu::gemv_q4_0_with_q8(
                    data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    proj,
                    in_proj.m,
                    in_proj.k,
                );
            } else {
                unsafe {
                    crate::backend::simd::neon::gemv_q8_0_q8_0_neon(
                        data,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        proj,
                        in_proj.m,
                        in_proj.k,
                    );
                }
            }
        } else {
            self.gemv(in_proj, hidden, proj);
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(in_proj, hidden, proj);

        // LoRA on the conv in_proj — `proj += scale·B·(A·hidden)`, applied to the
        // full 3·hidden output before it is split into the B/C/x gates. Matches
        // llama.cpp applying the adapter to `shortconv.in_proj` on conv layers.
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvInProj)
        {
            crate::lora::apply_decode(t, hidden, proj, &mut state.scratch.lora_tmp);
        }

        // Split: b, c, x
        let (b, rest) = proj.split_at(hidden_size);
        let (c, x) = rest.split_at(hidden_size);

        // bx = b ⊙ x (element-wise gate before conv)
        let conv_scratch = &mut state.scratch.conv_scratch[..hidden_size];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let mut i = 0usize;
            while i + 3 < hidden_size {
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vx = vld1q_f32(x.as_ptr().add(i));
                vst1q_f32(conv_scratch.as_mut_ptr().add(i), vmulq_f32(vb, vx));
                i += 4;
            }
            while i < hidden_size {
                conv_scratch[i] = b[i] * x[i];
                i += 1;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (out, (bi, xi)) in conv_scratch.iter_mut().zip(b.iter().zip(x.iter())) {
            *out = bi * xi;
        }
        // conv_scratch now holds bx

        // Depthwise conv1d with valid convolution using rolling buffer
        let LayerState::Conv { buffer, history } = &mut state.layers[layer] else {
            panic!("expected Conv state for layer {layer}");
        };

        {
            let out_buf = &mut state.scratch.out[..hidden_size];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let mut ch = 0usize;
                while ch + 3 < hidden_size {
                    let mut vsum = vdupq_n_f32(0.0);
                    for k in 0..d_conv {
                        let mut vw = vdupq_n_f32(0.0);
                        vw = vsetq_lane_f32::<0>(
                            *conv_weight.get_unchecked(ch * kernel_size + k),
                            vw,
                        );
                        vw = vsetq_lane_f32::<1>(
                            *conv_weight.get_unchecked((ch + 1) * kernel_size + k),
                            vw,
                        );
                        vw = vsetq_lane_f32::<2>(
                            *conv_weight.get_unchecked((ch + 2) * kernel_size + k),
                            vw,
                        );
                        vw = vsetq_lane_f32::<3>(
                            *conv_weight.get_unchecked((ch + 3) * kernel_size + k),
                            vw,
                        );
                        let vbuf = vld1q_f32(buffer.as_ptr().add(k * hidden_size + ch));
                        vsum = vmlaq_f32(vsum, vbuf, vw);
                    }
                    let mut vw_last = vdupq_n_f32(0.0);
                    vw_last = vsetq_lane_f32::<0>(
                        *conv_weight.get_unchecked(ch * kernel_size + d_conv),
                        vw_last,
                    );
                    vw_last = vsetq_lane_f32::<1>(
                        *conv_weight.get_unchecked((ch + 1) * kernel_size + d_conv),
                        vw_last,
                    );
                    vw_last = vsetq_lane_f32::<2>(
                        *conv_weight.get_unchecked((ch + 2) * kernel_size + d_conv),
                        vw_last,
                    );
                    vw_last = vsetq_lane_f32::<3>(
                        *conv_weight.get_unchecked((ch + 3) * kernel_size + d_conv),
                        vw_last,
                    );
                    let vcur = vld1q_f32(conv_scratch.as_ptr().add(ch));
                    vsum = vmlaq_f32(vsum, vcur, vw_last);
                    vst1q_f32(out_buf.as_mut_ptr().add(ch), vsum);
                    ch += 4;
                }
                while ch < hidden_size {
                    let mut sum = 0.0f32;
                    for k in 0..d_conv {
                        sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
                    }
                    sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                    out_buf[ch] = sum;
                    ch += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for ch in 0..hidden_size {
                let mut sum = 0.0f32;
                for k in 0..d_conv {
                    sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
                }
                sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                out_buf[ch] = sum;
            }

            // Update rolling buffer: shift left by one slot, append bx
            if d_conv > 0 {
                if d_conv > 1 {
                    buffer.copy_within(hidden_size.., 0);
                }
                let last_slot = (d_conv - 1) * hidden_size;
                buffer[last_slot..last_slot + hidden_size].copy_from_slice(conv_scratch);
                history.push(pos + 1, buffer);
            }

            // o = c ⊙ conv_out (second gate), reuse conv_scratch
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let mut i = 0usize;
                while i + 3 < hidden_size {
                    let vc = vld1q_f32(c.as_ptr().add(i));
                    let vout = vld1q_f32(out_buf.as_ptr().add(i));
                    vst1q_f32(conv_scratch.as_mut_ptr().add(i), vmulq_f32(vc, vout));
                    i += 4;
                }
                while i < hidden_size {
                    conv_scratch[i] = c[i] * out_buf[i];
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (o, (ci, co)) in conv_scratch.iter_mut().zip(c.iter().zip(out_buf.iter())) {
                *o = ci * co;
            }
        }

        // out_proj: hidden → hidden, write result into out_buf
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch_bufs(
                conv_scratch,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
            self.gemv_preq(
                out_proj,
                conv_scratch,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut state.scratch.out[..hidden_size],
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(
            out_proj,
            conv_scratch,
            &mut state.scratch.out[..hidden_size],
        );
        // LoRA on the conv out_proj (input = the gated conv output, before residual).
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
        {
            let out = &mut state.scratch.out[..hidden_size];
            crate::lora::apply_decode(t, conv_scratch, out, &mut state.scratch.lora_tmp);
        }
        // Result is now in state.scratch.out[..hidden_size]
    }

    /// Process a single attention block using pre-allocated scratch buffers.
    fn forward_attn_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;

        // Cloned once (cheap Arc bump) so the base scratch buffers can stay
        // mutably borrowed while the adapter (a disjoint field) is read.
        let lora = state.lora.clone();

        // Q, K, V projections using pre-quantized hidden state
        let q = &mut state.scratch.q[..cfg.hidden_size];
        let k = &mut state.scratch.k[..kv_dim];
        let v = &mut state.scratch.v[..kv_dim];

        // hidden was pre-quantized at layer level — use integer path
        #[cfg(target_arch = "aarch64")]
        {
            let q_ref = refs.attn_q.as_ref().unwrap();
            let k_ref = refs.attn_k.as_ref().unwrap();
            let v_ref = refs.attn_v.as_ref().unwrap();
            if q_ref.dtype == DType::Q4_0
                && k_ref.dtype == DType::Q4_0
                && v_ref.dtype == DType::Q4_0
            {
                let q_data = self.weight_data(q_ref);
                let k_data = self.weight_data(k_ref);
                let v_data = self.weight_data(v_ref);
                cpu::gemv_q4_0_concat3_with_q8(
                    q_data,
                    k_data,
                    v_data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    q,
                    k,
                    v,
                    cfg.hidden_size,
                    kv_dim,
                    kv_dim,
                    cfg.hidden_size,
                );
            } else {
                self.gemv_preq(
                    q_ref,
                    hidden,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    q,
                );
                self.gemv_preq(
                    k_ref,
                    hidden,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    k,
                );
                self.gemv_preq(
                    v_ref,
                    hidden,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    v,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.gemv(refs.attn_q.as_ref().unwrap(), hidden, q);
            self.gemv(refs.attn_k.as_ref().unwrap(), hidden, k);
            self.gemv(refs.attn_v.as_ref().unwrap(), hidden, v);
        }

        // LoRA on Q/K/V — input is the normed hidden, applied before QK-norm/RoPE.
        if let Some(lora) = &lora {
            crate::lora::apply_attn_qkv(lora, layer, hidden, q, k, v, &mut state.scratch.lora_tmp);
        }

        // Per-head QK norm (RMSnorm each head slice with shared weights)
        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
        for h in 0..cfg.n_heads {
            cpu::rmsnorm(
                &mut q[h * head_dim..(h + 1) * head_dim],
                q_norm,
                cfg.rms_norm_eps,
            );
        }
        for h in 0..n_kv_heads {
            cpu::rmsnorm(
                &mut k[h * head_dim..(h + 1) * head_dim],
                k_norm,
                cfg.rms_norm_eps,
            );
        }

        // RoPE
        cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta);

        // Grab per-model TurboQuant state once (None when disabled)
        // TurboQuant rotation state lives on InferenceState now (since PR #12
        // refactor). A single KvCompression::TurboQuant { seed, ... } config
        // on the state is enough — no separate model-side enable needed.
        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
        let tq_config = state.tq_config.as_ref();

        // f16 KV: append converts to half + the naive read path widens on read.
        // Mutually exclusive with TurboQuant (a distinct KvCompression variant),
        // so `use_f16` is never true alongside the TQ branches below.
        let use_f16 = state.kv_f16;

        // Append K, V to cache. Keys and values are compressed independently —
        // whichever side has a CompressedKvCache present gets the TurboQuant
        // path; the other side falls through to the f32 (or f16) cache.
        if let LayerState::Attention {
            key_cache,
            value_cache,
            key_cache_f16,
            value_cache_f16,
            compressed_keys,
            compressed_values,
        } = &mut state.layers[layer]
        {
            let tq_ok =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_encode_scratch.is_some();
            match (tq_ok, compressed_keys.as_mut()) {
                (true, Some(k_cache_tq)) => {
                    turboquant::compress_and_append_keys(
                        &state.scratch.k[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        k_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ if use_f16 => {
                    key_cache_f16.extend(
                        state.scratch.k[..kv_dim]
                            .iter()
                            .map(|&x| crate::quant::f32_to_f16(x)),
                    );
                }
                _ => {
                    key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                }
            }
            match (tq_ok, compressed_values.as_mut()) {
                (true, Some(v_cache_tq)) => {
                    turboquant::compress_and_append_values(
                        &state.scratch.v[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        v_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ if use_f16 => {
                    value_cache_f16.extend(
                        state.scratch.v[..kv_dim]
                            .iter()
                            .map(|&x| crate::quant::f32_to_f16(x)),
                    );
                }
                _ => {
                    value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                }
            }
        }

        // GQA: grouped query attention
        let group_size = cfg.n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        {
            // Access layers and scratch as disjoint fields to avoid whole-state borrow
            let (ck, cv, k_cache, v_cache, k_cache_f16, v_cache_f16) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    compressed_keys,
                    compressed_values,
                } => (
                    compressed_keys.as_ref(),
                    compressed_values.as_ref(),
                    key_cache.as_slice(),
                    value_cache.as_slice(),
                    key_cache_f16.as_slice(),
                    value_cache_f16.as_slice(),
                ),
                _ => panic!("expected Attention state for layer {layer}"),
            };

            // Keys and values are compressed independently — determine which
            // side of the attention read path uses TurboQuant.
            let tq_prereq =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_query_scratch.is_some();
            let use_tq_keys = tq_prereq && ck.is_some();
            let use_tq_values = tq_prereq && cv.is_some();

            // seq_len comes from whichever cache is populated. All four
            // combinations agree on seq_len per layer because encode appends
            // to one cache per side per token.
            let seq_len = if use_tq_keys {
                ck.unwrap().seq_len()
            } else if use_tq_values {
                cv.unwrap().seq_len()
            } else if use_f16 {
                k_cache_f16.len() / kv_dim
            } else {
                k_cache.len() / kv_dim
            };
            let attn_out = &mut state.scratch.attn_out[..cfg.hidden_size];
            let q = &state.scratch.q[..cfg.hidden_size];

            if use_tq_keys || use_tq_values {
                // GQA batched path — one score buffer per group, shared
                // between the key score and value weighted-sum stages.
                let scores = &mut state.scratch.scores;
                let rotation = tq_rotation.unwrap();
                let cfg_tq = tq_config.unwrap();
                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                if use_tq_keys {
                    turboquant::rotate_queries(q, cfg.n_heads, head_dim, rotation, qr_scratch);
                }
                scores.resize(seq_len * group_size, 0.0);
                for kv_h in 0..n_kv_heads {
                    let group_start = kv_h * group_size;
                    let kv_h_offset = kv_h * head_dim;

                    // Scores: TurboQuant or f32.
                    if use_tq_keys {
                        turboquant::attn_scores_turboquant_gqa(
                            ck.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            head_dim,
                            scale,
                            seq_len,
                            cfg_tq,
                            qr_scratch,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                            let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_scores(
                                q_head,
                                k_cache,
                                head_scores,
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                scale,
                                seq_len,
                            );
                        }
                    }

                    // Softmax each head's scores in place.
                    for g in 0..group_size {
                        let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                        cpu::softmax_inplace(head_scores);
                    }

                    // Values: TurboQuant or f32.
                    if use_tq_values {
                        turboquant::attn_values_turboquant_gqa(
                            cv.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            attn_out,
                            head_dim,
                            seq_len,
                            rotation,
                            cfg_tq,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let head_scores = &scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_values(
                                head_scores,
                                v_cache,
                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                seq_len,
                            );
                        }
                    }
                }
            } else {
                // Non-TQ path: f16 (widen-on-read) or f32. Only the active
                // representation is non-empty. Shared with the dense
                // transformers, which run the identical head loop.
                let kv = if use_f16 {
                    transformer::KvView::F16 {
                        k: k_cache_f16,
                        v: v_cache_f16,
                    }
                } else {
                    transformer::KvView::F32 {
                        k: k_cache,
                        v: v_cache,
                    }
                };
                transformer::decode_attention(
                    q,
                    &kv,
                    &transformer::DecodeAttnDims {
                        n_heads: cfg.n_heads,
                        n_kv_heads,
                        head_dim,
                        scale,
                        seq_len,
                    },
                    attn_out,
                    &mut state.scratch.scores,
                );
            }
        }

        // Output projection
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch_bufs(
                &state.scratch.attn_out[..cfg.hidden_size],
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
            self.gemv_preq(
                refs.attn_output.as_ref().unwrap(),
                &state.scratch.attn_out[..cfg.hidden_size],
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut state.scratch.out[..cfg.hidden_size],
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let out = &mut state.scratch.out[..cfg.hidden_size];
            self.gemv(
                refs.attn_output.as_ref().unwrap(),
                &state.scratch.attn_out[..cfg.hidden_size],
                out,
            );
        }
        // LoRA on the output projection (input = the attention output).
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
        {
            let out = &mut state.scratch.out[..cfg.hidden_size];
            crate::lora::apply_decode(
                t,
                &state.scratch.attn_out[..cfg.hidden_size],
                out,
                &mut state.scratch.lora_tmp,
            );
        }
    }
    /// Run all layers + output norm on a hidden state vector. Shared by
    /// forward(), forward_embedding(), and forward_hidden_from_embedding().
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        // Reuse pre-allocated scratch from InferenceState instead of allocating
        // fresh Vecs on every call. Take them out of `state.scratch` to avoid
        // borrow-checker conflicts with the mutable `state` passed to
        // forward_attn_block / forward_conv_block below; put them back at the end.
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        normed.resize(hs, 0.0);
        ffn_input.resize(hs, 0.0);

        let nb = hs / 32;
        state.scratch.q8_scales.resize(nb, 0.0);
        state.scratch.q8_quants.resize(hs, 0);

        for i in 0..cfg.n_layers {
            cpu::rmsnorm_and_quantize_q8_0(
                hidden,
                &self.attn_norm_weights[i],
                cfg.rms_norm_eps,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
                Some(&mut normed),
            );

            if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &normed, pos, state);
            } else {
                self.forward_attn_block(i, &normed, pos, state);
            }

            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            cpu::rmsnorm_and_quantize_q8_0(
                hidden,
                &self.ffn_norm_weights[i],
                cfg.rms_norm_eps,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
                Some(&mut ffn_input),
            );

            match &self.layer_refs[i].ffn {
                FfnRefs::Dense(dense) => {
                    let ffn_weights = FfnWeights {
                        ffn_gate: &dense.gate,
                        ffn_up: &dense.up,
                        ffn_down: &dense.down,
                    };
                    transformer::forward_ffn_block(
                        &self.gguf,
                        i,
                        &ffn_weights,
                        hs,
                        cfg.intermediate_size,
                        &ffn_input,
                        state,
                    );
                }
                FfnRefs::Moe(moe) => self.forward_moe_ffn(i, moe, hs, &ffn_input, state),
            }

            cpu::add_inplace(hidden, &state.scratch.out[..cfg.hidden_size]);
        }

        cpu::rmsnorm_and_quantize_q8_0(
            hidden,
            &self.output_norm_weight,
            cfg.rms_norm_eps,
            &mut state.scratch.q8_scales,
            &mut state.scratch.q8_quants,
            None,
        );
        state.seq_len += 1;

        // Return the scratch buffers for the next call.
        state.scratch.normed = normed;
        state.scratch.ffn_input = ffn_input;
    }

    /// Layer loop + output norm + tied logit projection for batched
    /// prefill. Takes a column-major hidden buffer (`hs × n`, channel
    /// `i` of token `j` at index `i * n + j`) already populated by the
    /// caller, runs the per-layer attn / conv + FFN passes, advances
    /// `state.seq_len` to `start_pos + n`, and projects the last
    /// frame's hidden state to logits over the vocabulary.
    ///
    /// Shared between [`Self::forward_prefill`] (token-id input,
    /// embedding-table lookup) and [`Self::forward_prefill_from_embeddings`]
    /// (raw embedding input, copied + transposed into the column-major
    /// `hidden` layout). The two entry points differ only in how they
    /// fill `hidden`; everything from the first RMSnorm onward is
    /// identical and lives here.
    /// Shared layer loop for prefill passes. Runs all layers and updates `hidden` in place.
    fn prefill_layers_loop(
        &self,
        hidden: &mut [f32],
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        // Cloned once (cheap Arc bump) so the adapter can be read after each
        // projection GEMM while the base-weight scratch buffers stay borrowed.
        let lora = state.lora.clone();

        // Per-layer hidden-state diagnostic. Set `CERA_DEBUG_HIDDEN=1`
        // to print the last token's RMS at each layer entry, after
        // attn/conv (post 1st residual), and after FFN (post 2nd
        // residual). Used to find the layer where cera's hidden
        // states diverge from llama.cpp's reference. Off in
        // production: a missing-env-var check is one syscall per
        // call, gated above every loop body to keep the hot path
        // cold. Any non-`"1"` value (including unset, empty, or
        // `"0"`) leaves diagnostics off — this matches the
        // documented `=1` setter and avoids a stray
        // `CERA_DEBUG_HIDDEN=0` accidentally enabling logging.
        let debug_hidden = std::env::var("CERA_DEBUG_HIDDEN").as_deref() == Ok("1");
        let log_rms = |label: &str, hidden: &[f32]| {
            if !debug_hidden {
                return;
            }
            // Last token's hidden vector lives at `hidden[i * n + (n-1)]`
            // for i in 0..hs (column-major). RMS of those `hs` values
            // is what feeds the next layer / output norm.
            let mut sum_sq = 0.0f64;
            let mut max_abs = 0.0f64;
            for i in 0..hs {
                let v = hidden[i * n + (n - 1)] as f64;
                sum_sq += v * v;
                let abs_v = v.abs();
                if abs_v > max_abs {
                    max_abs = abs_v;
                }
            }
            let rms = (sum_sq / hs as f64).sqrt();
            eprintln!("[cera.hidden] {label}: rms={rms:.6e} max_abs={max_abs:.6e}");
        };
        log_rms("input (pre-layer-0)", hidden);

        // Per-layer loop — pre-allocate all large buffers outside the loop
        let mut normed = vec![0.0f32; hs * n];
        let mut block_out = vec![0.0f32; hs * n];
        let mut ffn_input = vec![0.0f32; hs * n];
        let mut ffn_out = vec![0.0f32; hs * n];
        let mut col = vec![0.0f32; hs];
        let mut gate_col = vec![0.0f32; cfg.intermediate_size];
        let mut up_col = vec![0.0f32; cfg.intermediate_size];
        let mut out_col = vec![0.0f32; hs];
        // Batched projection buffers for conv/attn input projections.
        // Used by the no-`blas` int8 `gemm_preq` path (aarch64 NEON and
        // x86_64 int8, VNNI or AVX2) and the any-arch BLAS path
        // (`try_blas_prefill_gemm`).
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let max_kv_dim =
            cfg.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * (hs / cfg.n_heads);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let proj_rows = (3 * hs).max(hs + 2 * max_kv_dim);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut proj_mat = vec![0.0f32; proj_rows * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut out_proj_input = vec![0.0f32; hs * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut q_mat = vec![0.0f32; hs * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut k_mat = vec![0.0f32; max_kv_dim * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut v_mat = vec![0.0f32; max_kv_dim * n];
        // Pre-allocated GEMM buffers (reused across layers)
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let is = cfg.intermediate_size;
        // bq_*/dq_*/inter_col are scratch for the no-`blas` `gemm_preq` path
        // (aarch64 NEON and x86_64 int8, VNNI or AVX2)
        // (they hold the pre-quantized Q8_0 input matrix). With BLAS on, the
        // SGEMM path consumes f32 directly and these buffers are not needed.
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let nb_hs = hs / 32;
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let nb_is = is / 32;
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut bq_scales = vec![0.0f32; n * nb_hs];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut bq_quants = vec![0i8; n * hs];
        let mut gate_mat = vec![0.0f32; is * n];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut up_mat = vec![0.0f32; is * n];
        #[cfg(feature = "blas")]
        let mut gate_up_mat = vec![0.0f32; 2 * is * n];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut dq_scales = vec![0.0f32; n * nb_is];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut dq_quants = vec![0i8; n * is];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut inter_col = vec![0.0f32; is];
        // Flash attention scratch: contiguous output buffer reused across
        // layers. Sized for the largest possible attention layer (max
        // n_kv_heads * group_size * n * head_dim = hs * n).
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut flash_out = vec![0.0f32; hs * n];
        // f16 mode only: reused across layers to widen the half KV cache to f32
        // for the (f32-only) flash/naive attention kernels. Hoisted out of the
        // layer loop so each widen reuses one allocation instead of a fresh Vec
        // per layer. Stay empty (no alloc) on the f32/TurboQuant paths.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut kv_widen_k: Vec<f32> = Vec::new();
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut kv_widen_v: Vec<f32> = Vec::new();

        let profile_prefill = std::env::var_os("CERA_PROFILE_PREFILL").is_some();
        let mut t_norm = std::time::Duration::ZERO;
        let mut t_conv_in = std::time::Duration::ZERO;
        let mut t_conv_core = std::time::Duration::ZERO;
        let mut t_conv_out = std::time::Duration::ZERO;
        let mut t_attn_qkv = std::time::Duration::ZERO;
        let mut t_attn_out = std::time::Duration::ZERO;
        let mut t_ffn_norm = std::time::Duration::ZERO;
        let mut t_ffn_gate_up = std::time::Duration::ZERO;
        let mut t_ffn_down = std::time::Duration::ZERO;
        let mut t_residuals = std::time::Duration::ZERO;

        for layer in 0..cfg.n_layers {
            let t0 = std::time::Instant::now();
            // RMSnorm each token row
            let w_attn = &self.attn_norm_weights[layer];
            let eps = cfg.rms_norm_eps;
            if n >= 512 {
                let hidden_ptr = hidden.as_ptr() as usize;
                cpu::par_rows_n(&mut normed[..n * hs], hs, 64, move |(j, row)| unsafe {
                    let tok_h =
                        core::slice::from_raw_parts((hidden_ptr as *const f32).add(j * hs), hs);
                    cpu::rmsnorm_into(tok_h, row, w_attn, eps);
                });
            } else {
                for j in 0..n {
                    let tok_h = &hidden[j * hs..(j + 1) * hs];
                    let tok_norm = &mut normed[j * hs..(j + 1) * hs];
                    cpu::rmsnorm_into(tok_h, tok_norm, w_attn, eps);
                }
            }
            if profile_prefill {
                t_norm += t0.elapsed();
            }

            // Operator: conv or attention — batch projections via GEMM, sequential core
            let is_conv = cfg.block_types[layer] == BlockType::GatedConv;

            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
            let used_block_gemm = {
                let refs = &self.layer_refs[layer];
                if is_conv {
                    // --- Conv: batch in_proj + out_proj via GEMM ---
                    let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
                    let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
                    let blas_ok = [
                        ("shortconv.in_proj", in_proj),
                        ("shortconv.out_proj", out_proj),
                    ]
                    .into_iter()
                    .fold(true, |ok, (name, w)| {
                        if transformer::batched_gemm_supports(w.dtype, w.k) {
                            ok
                        } else {
                            transformer::warn_unbatchable(name, w.dtype);
                            false
                        }
                    });
                    if blas_ok {
                        #[cfg(not(feature = "blas"))]
                        {
                            let t = std::time::Instant::now();
                            transformer::quantize_columns(
                                &normed,
                                hs,
                                n,
                                &mut col,
                                &mut bq_scales,
                                &mut bq_quants,
                            );
                            transformer::try_repacked_gemm_rowmajor(
                                in_proj,
                                &bq_scales,
                                &bq_quants,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                3 * hs,
                                hs,
                            );
                            if profile_prefill {
                                t_conv_in += t.elapsed();
                            }
                        }
                        #[cfg(feature = "blas")]
                        {
                            let t = std::time::Instant::now();
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                in_proj,
                                &normed,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                3 * hs,
                                hs,
                            );
                            if profile_prefill {
                                t_conv_in += t.elapsed();
                            }
                        }

                        // LoRA on the conv in_proj
                        if let Some(lora) = &lora
                            && let Some(t) =
                                lora.get(layer, crate::lora::LoraTarget::ShortconvInProj)
                        {
                            crate::lora::apply_prefill(
                                t,
                                &normed,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }

                        // Phase 2: Per-token sequential conv using pre-computed projections
                        let t_c = std::time::Instant::now();
                        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
                        let d_conv = kernel_size - 1;
                        let conv_wt = self.conv_weights_transposed[layer].as_ref().unwrap();
                        let (conv_w0, rest_w) = conv_wt.split_at(hs);
                        let (conv_w1, conv_w2) = rest_w.split_at(hs);

                        for j in 0..n {
                            let proj = &proj_mat[j * 3 * hs..(j + 1) * 3 * hs];
                            let b_slice = &proj[..hs];
                            let c_slice = &proj[hs..2 * hs];
                            let x_slice = &proj[2 * hs..3 * hs];

                            let LayerState::Conv { buffer, history } = &mut state.layers[layer]
                            else {
                                panic!("expected Conv state for layer {layer}");
                            };

                            let out_in = &mut out_proj_input[j * hs..(j + 1) * hs];
                            let conv_scratch = &mut state.scratch.conv_scratch[..hs];

                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                use core::arch::aarch64::*;
                                let n_chunks = hs / 4;
                                let b_ptr = b_slice.as_ptr();
                                let x_ptr = x_slice.as_ptr();
                                let c_ptr = c_slice.as_ptr();
                                let b0_ptr = buffer.as_ptr();
                                let b1_ptr = buffer.as_ptr().add(hs);
                                let w0_ptr = conv_w0.as_ptr();
                                let w1_ptr = conv_w1.as_ptr();
                                let w2_ptr = conv_w2.as_ptr();
                                let out_in_ptr = out_in.as_mut_ptr();
                                let cs_ptr = conv_scratch.as_mut_ptr();

                                for i in 0..n_chunks {
                                    let b_v = vld1q_f32(b_ptr.add(i * 4));
                                    let x_v = vld1q_f32(x_ptr.add(i * 4));
                                    let bx_v = vmulq_f32(b_v, x_v);
                                    vst1q_f32(cs_ptr.add(i * 4), bx_v);

                                    let b0_v = vld1q_f32(b0_ptr.add(i * 4));
                                    let b1_v = vld1q_f32(b1_ptr.add(i * 4));
                                    let w0_v = vld1q_f32(w0_ptr.add(i * 4));
                                    let w1_v = vld1q_f32(w1_ptr.add(i * 4));
                                    let w2_v = vld1q_f32(w2_ptr.add(i * 4));

                                    let out_v = vfmaq_f32(
                                        vfmaq_f32(vmulq_f32(b0_v, w0_v), b1_v, w1_v),
                                        bx_v,
                                        w2_v,
                                    );
                                    let c_v = vld1q_f32(c_ptr.add(i * 4));
                                    let final_v = vmulq_f32(c_v, out_v);
                                    vst1q_f32(out_in_ptr.add(i * 4), final_v);
                                }
                            }

                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                for i in 0..hs {
                                    conv_scratch[i] = b_slice[i] * x_slice[i];
                                    let conv_out = buffer[i] * conv_w0[i]
                                        + buffer[hs + i] * conv_w1[i]
                                        + conv_scratch[i] * conv_w2[i];
                                    out_in[i] = c_slice[i] * conv_out;
                                }
                            }

                            if d_conv > 0 {
                                if d_conv > 1 {
                                    buffer.copy_within(hs.., 0);
                                }
                                let last_slot = (d_conv - 1) * hs;
                                buffer[last_slot..last_slot + hs].copy_from_slice(conv_scratch);
                                if j + 1 == n {
                                    history.push(start_pos + j + 1, buffer);
                                }
                            }
                        }
                        if profile_prefill {
                            t_conv_core += t_c.elapsed();
                        }

                        // Phase 3: Batch out_proj GEMM
                        #[cfg(not(feature = "blas"))]
                        {
                            let t_o = std::time::Instant::now();
                            transformer::quantize_columns(
                                &out_proj_input,
                                hs,
                                n,
                                &mut col,
                                &mut bq_scales,
                                &mut bq_quants,
                            );
                            transformer::try_repacked_gemm_rowmajor(
                                out_proj,
                                &bq_scales,
                                &bq_quants,
                                &mut block_out[..hs * n],
                                n,
                                hs,
                                hs,
                            );
                            if profile_prefill {
                                t_conv_out += t_o.elapsed();
                            }
                        }
                        #[cfg(feature = "blas")]
                        {
                            let t_o = std::time::Instant::now();
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                out_proj,
                                &out_proj_input,
                                &mut block_out,
                                n,
                                hs,
                                hs,
                            );
                            if profile_prefill {
                                t_conv_out += t_o.elapsed();
                            }
                        }

                        // LoRA on the conv out_proj
                        if let Some(lora) = &lora
                            && let Some(t) =
                                lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
                        {
                            crate::lora::apply_prefill(
                                t,
                                &out_proj_input,
                                &mut block_out,
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    // --- Attention: batch Q/K/V + output projections via GEMM ---
                    let attn_q_ref = refs.attn_q.as_ref().unwrap();
                    let attn_k_ref = refs.attn_k.as_ref().unwrap();
                    let attn_v_ref = refs.attn_v.as_ref().unwrap();
                    let attn_output_ref = refs.attn_output.as_ref().unwrap();
                    // Require ALL four projections to be batchable — a mixed-dtype
                    // attention block would leave later matrices silently uncomputed
                    // in the batched path and produce wrong outputs.
                    let blas_ok = [
                        ("attn_q", attn_q_ref),
                        ("attn_k", attn_k_ref),
                        ("attn_v", attn_v_ref),
                        ("attn_output", attn_output_ref),
                    ]
                    .into_iter()
                    .fold(true, |ok, (name, w)| {
                        if transformer::batched_gemm_supports(w.dtype, w.k) {
                            ok
                        } else {
                            transformer::warn_unbatchable(name, w.dtype);
                            false
                        }
                    });
                    if blas_ok {
                        let head_dim = hs / cfg.n_heads;
                        let n_kv_heads = cfg.kv_heads_per_layer[layer];
                        let kv_dim = n_kv_heads * head_dim;

                        // Phase 1: Batch Q/K/V GEMM (Single fused AMX dispatch)
                        #[cfg(feature = "blas")]
                        {
                            let t_qkv = std::time::Instant::now();
                            let qkv_f32 = refs.qkv_f32.get_or_init(|| {
                                let q_f32 =
                                    transformer::get_dequantized_f32(&self.gguf, attn_q_ref);
                                let k_f32 =
                                    transformer::get_dequantized_f32(&self.gguf, attn_k_ref);
                                let v_f32 =
                                    transformer::get_dequantized_f32(&self.gguf, attn_v_ref);
                                let qkv_dim = hs + 2 * kv_dim;
                                let mut fused = vec![0.0f32; hs * qkv_dim];
                                for c in 0..hs {
                                    let dst_row = &mut fused[c * qkv_dim..(c + 1) * qkv_dim];
                                    dst_row[..hs].copy_from_slice(&q_f32[c * hs..(c + 1) * hs]);
                                    dst_row[hs..hs + kv_dim]
                                        .copy_from_slice(&k_f32[c * kv_dim..(c + 1) * kv_dim]);
                                    dst_row[hs + kv_dim..]
                                        .copy_from_slice(&v_f32[c * kv_dim..(c + 1) * kv_dim]);
                                }
                                fused
                            });

                            let qkv_dim = hs + 2 * kv_dim;
                            crate::backend::blas::sgemm_rowmajor_nn_parallel(
                                n,
                                qkv_dim,
                                hs,
                                &normed,
                                qkv_f32,
                                &mut proj_mat[..qkv_dim * n],
                            );

                            for j in 0..n {
                                let row = &proj_mat[j * qkv_dim..(j + 1) * qkv_dim];
                                q_mat[j * hs..(j + 1) * hs].copy_from_slice(&row[..hs]);
                                k_mat[j * kv_dim..(j + 1) * kv_dim]
                                    .copy_from_slice(&row[hs..hs + kv_dim]);
                                v_mat[j * kv_dim..(j + 1) * kv_dim]
                                    .copy_from_slice(&row[hs + kv_dim..qkv_dim]);
                            }
                            if profile_prefill {
                                t_attn_qkv += t_qkv.elapsed();
                            }
                        }
                        #[cfg(not(feature = "blas"))]
                        {
                            let t_qkv = std::time::Instant::now();
                            transformer::quantize_columns(
                                &normed,
                                hs,
                                n,
                                &mut col,
                                &mut bq_scales,
                                &mut bq_quants,
                            );

                            let qkv_dim = hs + 2 * kv_dim;
                            let qkv_repacked = refs.qkv_repacked.get_or_init(|| {
                                #[allow(clippy::collapsible_if)]
                                if let (Some(q_rp), Some(k_rp), Some(v_rp)) = (
                                    &attn_q_ref.repacked,
                                    &attn_k_ref.repacked,
                                    &attn_v_ref.repacked,
                                ) {
                                    if let (
                                        transformer::Repacked::Q40 {
                                            packed: q_p,
                                            scales: q_s,
                                        },
                                        transformer::Repacked::Q40 {
                                            packed: k_p,
                                            scales: k_s,
                                        },
                                        transformer::Repacked::Q40 {
                                            packed: v_p,
                                            scales: v_s,
                                        },
                                    ) = (&q_rp.kind, &k_rp.kind, &v_rp.kind)
                                    {
                                        let mut p =
                                            Vec::with_capacity(q_p.len() + k_p.len() + v_p.len());
                                        p.extend_from_slice(q_p);
                                        p.extend_from_slice(k_p);
                                        p.extend_from_slice(v_p);

                                        let mut s =
                                            Vec::with_capacity(q_s.len() + k_s.len() + v_s.len());
                                        s.extend_from_slice(q_s);
                                        s.extend_from_slice(k_s);
                                        s.extend_from_slice(v_s);
                                        return Some((p, s));
                                    }
                                }
                                None
                            });

                            let mut ran_fused = false;
                            if let Some((p, s)) = qkv_repacked.as_ref() {
                                ran_fused =
                                    crate::backend::cpu::gemm_preq_repacked_q4_0_rowmajor_dispatch(
                                        p,
                                        s,
                                        &bq_scales,
                                        &bq_quants,
                                        &mut proj_mat[..qkv_dim * n],
                                        n,
                                        qkv_dim,
                                        hs,
                                    );
                                if ran_fused {
                                    for j in 0..n {
                                        let row = &proj_mat[j * qkv_dim..(j + 1) * qkv_dim];
                                        q_mat[j * hs..(j + 1) * hs].copy_from_slice(&row[..hs]);
                                        k_mat[j * kv_dim..(j + 1) * kv_dim]
                                            .copy_from_slice(&row[hs..hs + kv_dim]);
                                        v_mat[j * kv_dim..(j + 1) * kv_dim]
                                            .copy_from_slice(&row[hs + kv_dim..qkv_dim]);
                                    }
                                }
                            }

                            if !ran_fused {
                                transformer::try_repacked_gemm_rowmajor(
                                    attn_q_ref,
                                    &bq_scales,
                                    &bq_quants,
                                    &mut q_mat[..hs * n],
                                    n,
                                    hs,
                                    hs,
                                );
                                transformer::try_repacked_gemm_rowmajor(
                                    attn_k_ref,
                                    &bq_scales,
                                    &bq_quants,
                                    &mut k_mat[..kv_dim * n],
                                    n,
                                    kv_dim,
                                    hs,
                                );
                                transformer::try_repacked_gemm_rowmajor(
                                    attn_v_ref,
                                    &bq_scales,
                                    &bq_quants,
                                    &mut v_mat[..kv_dim * n],
                                    n,
                                    kv_dim,
                                    hs,
                                );
                            }
                            if profile_prefill {
                                t_attn_qkv += t_qkv.elapsed();
                            }
                        }

                        // LoRA on Q/K/V — added to the projection outputs before
                        // QK-norm/RoPE, input is the normed hidden `[hs×n]`.
                        if let Some(lora) = &lora {
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnQ) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut q_mat[..hs * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnK) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut k_mat[..kv_dim * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnV) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut v_mat[..kv_dim * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                        }

                        // Phase 2: Per-token attention (QK norm, RoPE, KV cache, scores)
                        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
                        let tq_config = state.tq_config.as_ref();
                        let will_compress_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_encode_scratch.is_some();
                        let will_read_compressed_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_query_scratch.is_some();
                        let use_f16 = state.kv_f16;

                        if let LayerState::Attention {
                            key_cache,
                            value_cache,
                            key_cache_f16,
                            value_cache_f16,
                            compressed_keys,
                            compressed_values,
                        } = &mut state.layers[layer]
                        {
                            match (will_compress_kv, compressed_keys.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.jl_data.iter_mut() {
                                        v.reserve(n * head_dim / 8);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ if use_f16 => {
                                    key_cache_f16.reserve(n * kv_dim);
                                }
                                _ => {
                                    key_cache.reserve(n * kv_dim);
                                }
                            }
                            match (will_compress_kv, compressed_values.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ if use_f16 => {
                                    value_cache_f16.reserve(n * kv_dim);
                                }
                                _ => {
                                    value_cache.reserve(n * kv_dim);
                                }
                            }
                        }
                        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
                        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
                        let group_size = cfg.n_heads / n_kv_heads;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        // ── Pass A: QK-norm + RoPE + KV cache append ──────────
                        if n >= 4 {
                            let q_ptr = q_mat.as_mut_ptr() as usize;
                            let k_ptr = k_mat.as_mut_ptr() as usize;
                            let n_heads = cfg.n_heads;
                            let rope_theta = cfg.rope_theta;
                            let eps = cfg.rms_norm_eps;
                            cpu::par_rows_n(&mut q_mat[..n * hs], hs, 4, move |(j, _)| unsafe {
                                let pos = start_pos + j;
                                let q = core::slice::from_raw_parts_mut(
                                    (q_ptr as *mut f32).add(j * hs),
                                    hs,
                                );
                                let k = core::slice::from_raw_parts_mut(
                                    (k_ptr as *mut f32).add(j * kv_dim),
                                    kv_dim,
                                );
                                for h in 0..n_heads {
                                    cpu::rmsnorm(
                                        &mut q[h * head_dim..(h + 1) * head_dim],
                                        q_norm,
                                        eps,
                                    );
                                }
                                for h in 0..n_kv_heads {
                                    cpu::rmsnorm(
                                        &mut k[h * head_dim..(h + 1) * head_dim],
                                        k_norm,
                                        eps,
                                    );
                                }
                                cpu::rope(q, k, pos, n_heads, n_kv_heads, head_dim, rope_theta);
                            });
                        } else {
                            for j in 0..n {
                                let pos = start_pos + j;
                                let q = &mut q_mat[j * hs..(j + 1) * hs];
                                let k = &mut k_mat[j * kv_dim..(j + 1) * kv_dim];
                                for h in 0..cfg.n_heads {
                                    cpu::rmsnorm(
                                        &mut q[h * head_dim..(h + 1) * head_dim],
                                        q_norm,
                                        cfg.rms_norm_eps,
                                    );
                                }
                                for h in 0..n_kv_heads {
                                    cpu::rmsnorm(
                                        &mut k[h * head_dim..(h + 1) * head_dim],
                                        k_norm,
                                        cfg.rms_norm_eps,
                                    );
                                }
                                cpu::rope(
                                    q,
                                    k,
                                    pos,
                                    cfg.n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    cfg.rope_theta,
                                );
                            }
                        }

                        for j in 0..n {
                            let k = &k_mat[j * kv_dim..(j + 1) * kv_dim];
                            let v = &v_mat[j * kv_dim..(j + 1) * kv_dim];

                            // Append K, V to cache (f32 or TurboQuant-compressed).
                            if let LayerState::Attention {
                                key_cache,
                                value_cache,
                                key_cache_f16,
                                value_cache_f16,
                                compressed_keys,
                                compressed_values,
                            } = &mut state.layers[layer]
                            {
                                match (will_compress_kv, compressed_keys.as_mut()) {
                                    (true, Some(k_cache_tq)) => {
                                        turboquant::compress_and_append_keys(
                                            k,
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            k_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ if use_f16 => {
                                        key_cache_f16
                                            .extend(k.iter().map(|&x| crate::quant::f32_to_f16(x)));
                                    }
                                    _ => {
                                        key_cache.extend_from_slice(k);
                                    }
                                }
                                match (will_compress_kv, compressed_values.as_mut()) {
                                    (true, Some(v_cache_tq)) => {
                                        turboquant::compress_and_append_values(
                                            v,
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            v_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ if use_f16 => {
                                        value_cache_f16
                                            .extend(v.iter().map(|&x| crate::quant::f32_to_f16(x)));
                                    }
                                    _ => {
                                        value_cache.extend_from_slice(v);
                                    }
                                }
                            }
                        }

                        // ── Pass B: attention ────────────────────────────────
                        let use_tq = will_read_compressed_kv
                            && match &state.layers[layer] {
                                LayerState::Attention {
                                    compressed_keys,
                                    compressed_values,
                                    ..
                                } => compressed_keys.is_some() || compressed_values.is_some(),
                                _ => false,
                            };

                        const FLASH_ATTN_THRESHOLD: usize = 16;
                        let use_flash = !use_tq && n >= FLASH_ATTN_THRESHOLD;

                        if use_f16
                            && let LayerState::Attention {
                                key_cache_f16,
                                value_cache_f16,
                                ..
                            } = &state.layers[layer]
                        {
                            kv_widen_k.clear();
                            kv_widen_k
                                .extend(key_cache_f16.iter().map(|&b| crate::quant::f16_to_f32(b)));
                            kv_widen_v.clear();
                            kv_widen_v.extend(
                                value_cache_f16.iter().map(|&b| crate::quant::f16_to_f32(b)),
                            );
                        }

                        if use_flash {
                            let (k_cache, v_cache) = if use_f16 {
                                (kv_widen_k.as_slice(), kv_widen_v.as_slice())
                            } else {
                                match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        ..
                                    } => (key_cache.as_slice(), value_cache.as_slice()),
                                    _ => unreachable!(),
                                }
                            };
                            let n_heads = cfg.n_heads;
                            let head_chunk = n * head_dim;
                            let flash_buf = &mut flash_out[..n_heads * head_chunk];
                            let q_ref = &q_mat[..];

                            cpu::par_rows_n_chunked(flash_buf, head_chunk, 1, 1, |(h, chunk)| {
                                let kv_h = h / group_size;
                                cpu::flash_attention_gqa_cpu(
                                    q_ref,
                                    k_cache,
                                    v_cache,
                                    chunk,
                                    h,
                                    1,
                                    n,
                                    hs,
                                    kv_dim,
                                    kv_h * head_dim,
                                    head_dim,
                                    scale,
                                    start_pos,
                                );
                            });

                            for h in 0..n_heads {
                                let src_base = h * n * head_dim;
                                for j in 0..n {
                                    let dst_base = j * hs + h * head_dim;
                                    let src = &flash_buf
                                        [src_base + j * head_dim..src_base + (j + 1) * head_dim];
                                    out_proj_input[dst_base..dst_base + head_dim]
                                        .copy_from_slice(src);
                                }
                            }
                        } else if use_tq {
                            state.scratch.scores.clear();
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let q = &q_mat[j * hs..(j + 1) * hs];
                                let (ck, cv, k_cache, v_cache) = match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        compressed_keys,
                                        compressed_values,
                                        ..
                                    } => (
                                        compressed_keys.as_ref(),
                                        compressed_values.as_ref(),
                                        key_cache.as_slice(),
                                        value_cache.as_slice(),
                                    ),
                                    _ => unreachable!(),
                                };

                                let use_tq_keys = will_read_compressed_kv && ck.is_some();
                                let use_tq_values = will_read_compressed_kv && cv.is_some();

                                let seq_len = if use_tq_keys {
                                    ck.unwrap().seq_len()
                                } else if use_tq_values {
                                    cv.unwrap().seq_len()
                                } else {
                                    k_cache.len() / kv_dim
                                };
                                let attn_out = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let scores = &mut state.scratch.scores;

                                let rotation = tq_rotation.unwrap();
                                let cfg_tq = tq_config.unwrap();
                                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                                if use_tq_keys {
                                    turboquant::rotate_queries(
                                        q,
                                        cfg.n_heads,
                                        head_dim,
                                        rotation,
                                        qr_scratch,
                                    );
                                }
                                scores.resize(seq_len * group_size, 0.0);
                                for kv_h in 0..n_kv_heads {
                                    let group_start = kv_h * group_size;
                                    let kv_h_offset = kv_h * head_dim;

                                    if use_tq_keys {
                                        turboquant::attn_scores_turboquant_gqa(
                                            ck.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            head_dim,
                                            scale,
                                            seq_len,
                                            cfg_tq,
                                            qr_scratch,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                            let head_scores =
                                                &mut scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_scores(
                                                q_head,
                                                k_cache,
                                                head_scores,
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                scale,
                                                seq_len,
                                            );
                                        }
                                    }

                                    for g in 0..group_size {
                                        let head_scores =
                                            &mut scores[g * seq_len..(g + 1) * seq_len];
                                        cpu::softmax_inplace(head_scores);
                                    }

                                    if use_tq_values {
                                        turboquant::attn_values_turboquant_gqa(
                                            cv.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            attn_out,
                                            head_dim,
                                            seq_len,
                                            rotation,
                                            cfg_tq,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let head_scores =
                                                &scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_values(
                                                head_scores,
                                                v_cache,
                                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                seq_len,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            let (k_cache, v_cache) = if use_f16 {
                                (kv_widen_k.as_slice(), kv_widen_v.as_slice())
                            } else {
                                match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        ..
                                    } => (key_cache.as_slice(), value_cache.as_slice()),
                                    _ => unreachable!(),
                                }
                            };
                            state.scratch.scores.clear();
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let seq_len = (start_pos + j + 1).min(k_cache.len() / kv_dim);
                                let q = &q_mat[j * hs..(j + 1) * hs];
                                let attn_out = &mut out_proj_input[j * hs..(j + 1) * hs];
                                let scores = &mut state.scratch.scores;
                                scores.resize(seq_len, 0.0);
                                for h in 0..cfg.n_heads {
                                    let kv_h = h / group_size;
                                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                    let kv_h_offset = kv_h * head_dim;
                                    cpu::attn_scores(
                                        q_head,
                                        k_cache,
                                        scores,
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        scale,
                                        seq_len,
                                    );
                                    cpu::softmax_inplace(scores);
                                    cpu::attn_values(
                                        scores,
                                        v_cache,
                                        &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        seq_len,
                                    );
                                }
                            }
                        }

                        // Phase 3: Batch output projection GEMM
                        #[cfg(not(feature = "blas"))]
                        {
                            let t_ao = std::time::Instant::now();
                            transformer::quantize_columns(
                                &out_proj_input,
                                hs,
                                n,
                                &mut col,
                                &mut bq_scales,
                                &mut bq_quants,
                            );
                            transformer::try_repacked_gemm_rowmajor(
                                attn_output_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut block_out[..hs * n],
                                n,
                                hs,
                                hs,
                            );
                            if profile_prefill {
                                t_attn_out += t_ao.elapsed();
                            }
                        }
                        #[cfg(feature = "blas")]
                        {
                            let t_ao = std::time::Instant::now();
                            transformer::try_blas_prefill_gemm_rowmajor(
                                &self.gguf,
                                attn_output_ref,
                                &out_proj_input,
                                &mut block_out,
                                n,
                                hs,
                                hs,
                            );
                            if profile_prefill {
                                t_attn_out += t_ao.elapsed();
                            }
                        }

                        // LoRA on the output projection — applied to `block_out`
                        // BEFORE the residual add; input is the attention output
                        // `[hs×n]`.
                        if let Some(lora) = &lora
                            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
                        {
                            crate::lora::apply_prefill(
                                t,
                                &out_proj_input,
                                &mut block_out,
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        true
                    } else {
                        false
                    }
                }
            };

            // Fallback: per-token sequential path. Used on x86_64-no-blas
            // (no batched path compiled), and on any target where the
            // batched path saw mixed dtypes and bailed (`used_block_gemm
            // = false`).
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
            let need_block_fallback = !used_block_gemm;
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas")))]
            let need_block_fallback = true;
            if need_block_fallback {
                block_out.fill(0.0);
                for j in 0..n {
                    for i in 0..hs {
                        col[i] = normed[i * n + j];
                    }
                    #[cfg(target_arch = "aarch64")]
                    Self::quantize_to_scratch(&col, state);

                    if is_conv {
                        self.forward_conv_block(layer, &col, start_pos + j, state);
                    } else {
                        self.forward_attn_block(layer, &col, start_pos + j, state);
                    }

                    for i in 0..hs {
                        block_out[i * n + j] = state.scratch.out[i];
                    }
                }
            }

            // Log the BLOCK OUTPUT magnitude (pre-residual) so we
            // can see how much each layer contributes vs the prior
            // residual. Together with the post-block log this lets
            // us identify whether the magnitude growth is from a
            // misbehaving block, an oversize residual, or both.
            // The `if debug_hidden` guard keeps the per-layer
            // `format!` allocations off the hot path when the env
            // var isn't set.
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(
                    &format!("layer {layer} ({block_kind}) block-out"),
                    &block_out,
                );
            }

            let t_res1 = std::time::Instant::now();
            // Residual: hidden += block_out
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let total = hs * n;
                let n_chunks = total / 4;
                let h_ptr = hidden.as_mut_ptr();
                let bo_ptr = block_out.as_ptr();
                for i in 0..n_chunks {
                    let h_v = vld1q_f32(h_ptr.add(i * 4));
                    let bo_v = vld1q_f32(bo_ptr.add(i * 4));
                    vst1q_f32(h_ptr.add(i * 4), vaddq_f32(h_v, bo_v));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for i in 0..hs * n {
                hidden[i] += block_out[i];
            }
            if profile_prefill {
                t_residuals += t_res1.elapsed();
            }
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(&format!("layer {layer} ({block_kind}) post-block"), hidden);
            }

            let t_fn = std::time::Instant::now();
            // FFN pre-norm each row
            let w_ffn = &self.ffn_norm_weights[layer];
            let eps = cfg.rms_norm_eps;
            if n >= 512 {
                let hidden_ptr = hidden.as_ptr() as usize;
                cpu::par_rows_n(&mut ffn_input[..n * hs], hs, 64, move |(j, row)| unsafe {
                    let tok_h =
                        core::slice::from_raw_parts((hidden_ptr as *const f32).add(j * hs), hs);
                    cpu::rmsnorm_into(tok_h, row, w_ffn, eps);
                });
            } else {
                for j in 0..n {
                    let tok_h = &hidden[j * hs..(j + 1) * hs];
                    let tok_in = &mut ffn_input[j * hs..(j + 1) * hs];
                    cpu::rmsnorm_into(tok_h, tok_in, w_ffn, eps);
                }
            }
            if profile_prefill {
                t_ffn_norm += t_fn.elapsed();
            }

            // FFN: batched GEMM (reads weights once for all n tokens) for the
            // dtypes `batched_gemm_supports` admits. Available on aarch64 (NEON
            // `gemm_preq`), on x86_64 with runtime avx2+fma (int8
            // `gemm_preq`), and on any target with `feature = "blas"` (BLAS
            // SGEMM via `try_blas_prefill_gemm`). Require all three projections
            // (gate/up/down) to be batchable — a mixed-dtype FFN block would
            // leave later matrices silently uncomputed in the batched path and
            // produce wrong outputs.
            let refs = &self.layer_refs[layer];
            // A MoE layer has no single FFN weight to batch: each token picks
            // its own experts, so the dense path below does not apply. Routed
            // per token: correct, but not yet grouped by expert (plan
            // 000312-01, phase 1 step 6).
            'dense_ffn: {
                let dense = match &refs.ffn {
                    FfnRefs::Dense(d) => d,
                    FfnRefs::Moe(moe) => {
                        self.prefill_moe_ffn(
                            layer,
                            moe,
                            hs,
                            n,
                            &ffn_input,
                            &mut ffn_out,
                            &mut col,
                            state,
                        );
                        break 'dense_ffn;
                    }
                };
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
                let used_gemm = if [
                    ("ffn_gate", &dense.gate),
                    ("ffn_up", &dense.up),
                    ("ffn_down", &dense.down),
                ]
                .into_iter()
                .fold(true, |ok, (name, w)| {
                    if transformer::batched_gemm_supports(w.dtype, w.k) {
                        ok
                    } else {
                        transformer::warn_unbatchable(name, w.dtype);
                        false
                    }
                }) {
                    // Pre-quantize all n columns to Q8_0 — only needed for the NEON fallback.
                    #[cfg(not(feature = "blas"))]
                    transformer::quantize_columns(
                        &ffn_input,
                        hs,
                        n,
                        &mut col,
                        &mut bq_scales,
                        &mut bq_quants,
                    );

                    // Gate + Up via Single-Dispatch Fused AMX GEMM (AMX Gate + Up)
                    #[cfg(feature = "blas")]
                    {
                        let t_gu = std::time::Instant::now();
                        let gu_f32 = dense.gate_up_f32.get_or_init(|| {
                            let g_f32 = transformer::get_dequantized_f32(&self.gguf, &dense.gate);
                            let u_f32 = transformer::get_dequantized_f32(&self.gguf, &dense.up);
                            let mut fused = vec![0.0f32; hs * 2 * is];
                            for c in 0..hs {
                                let dst_row = &mut fused[c * 2 * is..(c + 1) * 2 * is];
                                dst_row[..is].copy_from_slice(&g_f32[c * is..(c + 1) * is]);
                                dst_row[is..].copy_from_slice(&u_f32[c * is..(c + 1) * is]);
                            }
                            fused
                        });

                        crate::backend::blas::sgemm_rowmajor_nn_parallel(
                            n,
                            2 * is,
                            hs,
                            &ffn_input,
                            gu_f32,
                            &mut gate_up_mat[..2 * is * n],
                        );

                        for j in 0..n {
                            let (g_slice, u_slice) =
                                gate_up_mat[j * 2 * is..(j + 1) * 2 * is].split_at_mut(is);
                            cpu::silu_mul_inplace(g_slice, u_slice);
                        }

                        if profile_prefill {
                            t_ffn_gate_up += t_gu.elapsed();
                        }
                    }
                    #[cfg(not(feature = "blas"))]
                    {
                        let t_gu = std::time::Instant::now();
                        let fused_ok = lora.is_none()
                            && transformer::try_repacked_gate_up_silu_rowmajor(
                                &dense.gate,
                                &dense.up,
                                &bq_scales,
                                &bq_quants,
                                &mut gate_mat[..is * n],
                                n,
                                is,
                                hs,
                            );
                        if !fused_ok {
                            transformer::try_repacked_gemm_rowmajor(
                                &dense.gate,
                                &bq_scales,
                                &bq_quants,
                                &mut gate_mat[..is * n],
                                n,
                                is,
                                hs,
                            );
                            transformer::try_repacked_gemm_rowmajor(
                                &dense.up,
                                &bq_scales,
                                &bq_quants,
                                &mut up_mat[..is * n],
                                n,
                                is,
                                hs,
                            );
                            if let Some(lora) = &lora {
                                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                                    crate::lora::apply_prefill(
                                        t,
                                        &ffn_input,
                                        &mut gate_mat[..is * n],
                                        n,
                                        &mut state.scratch.lora_tmp,
                                    );
                                }
                                if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                                    crate::lora::apply_prefill(
                                        t,
                                        &ffn_input,
                                        &mut up_mat[..is * n],
                                        n,
                                        &mut state.scratch.lora_tmp,
                                    );
                                }
                            }
                            cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);
                        }
                        if profile_prefill {
                            t_ffn_gate_up += t_gu.elapsed();
                        }
                    }

                    // Re-quantize gate_mat columns for down projection — only needed for NEON fallback.
                    #[cfg(not(feature = "blas"))]
                    transformer::quantize_columns(
                        &gate_mat,
                        is,
                        n,
                        &mut inter_col,
                        &mut dq_scales,
                        &mut dq_quants,
                    );

                    // Down via batched GEMM
                    #[cfg(feature = "blas")]
                    {
                        let t_d = std::time::Instant::now();
                        let down_f32 = transformer::get_dequantized_f32(&self.gguf, &dense.down);
                        crate::backend::blas::sgemm_rowmajor_nn_ld_parallel(
                            n,
                            hs,
                            is,
                            gate_up_mat.as_ptr(),
                            2 * is,
                            down_f32.as_ptr(),
                            hs,
                            ffn_out.as_mut_ptr(),
                            hs,
                        );
                        if profile_prefill {
                            t_ffn_down += t_d.elapsed();
                        }
                    }
                    #[cfg(not(feature = "blas"))]
                    {
                        let t_d = std::time::Instant::now();
                        transformer::try_repacked_gemm_rowmajor(
                            &dense.down,
                            &dq_scales,
                            &dq_quants,
                            &mut ffn_out[..hs * n],
                            n,
                            hs,
                            is,
                        );
                        if profile_prefill {
                            t_ffn_down += t_d.elapsed();
                        }
                    }

                    // LoRA on the down projection — applied to `ffn_out` BEFORE the
                    // residual add; input is the SiLU⊙up product in `gate_mat` `[is×n]`.
                    if let Some(lora) = &lora
                        && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
                    {
                        crate::lora::apply_prefill(
                            t,
                            &gate_mat[..is * n],
                            &mut ffn_out,
                            n,
                            &mut state.scratch.lora_tmp,
                        );
                    }
                    true
                } else {
                    false
                };

                // Fallback: per-token GEMV. Used on x86_64-no-blas (no batched
                // path compiled), and on any target where the FFN weights
                // weren't all batchable (`used_gemm = false`).
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
                let need_fallback = !used_gemm;
                #[cfg(not(any(
                    target_arch = "aarch64",
                    target_arch = "x86_64",
                    feature = "blas"
                )))]
                let need_fallback = true;
                if need_fallback {
                    ffn_out.fill(0.0);
                    for j in 0..n {
                        for i in 0..hs {
                            col[i] = ffn_input[i * n + j];
                        }

                        #[cfg(target_arch = "aarch64")]
                        {
                            Self::quantize_to_scratch(&col, state);
                            self.gemv_preq(
                                &dense.gate,
                                &col,
                                &state.scratch.q8_scales,
                                &state.scratch.q8_quants,
                                &mut gate_col,
                            );
                            self.gemv_preq(
                                &dense.up,
                                &col,
                                &state.scratch.q8_scales,
                                &state.scratch.q8_quants,
                                &mut up_col,
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            self.gemv(&dense.gate, &col, &mut gate_col);
                            self.gemv(&dense.up, &col, &mut up_col);
                        }

                        // LoRA on gate/up (per-token decode hook) — this fallback loop
                        // doesn't route through `forward_ffn_block`, so apply it here.
                        if let Some(lora) = &lora {
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                                crate::lora::apply_decode(
                                    t,
                                    &col,
                                    &mut gate_col,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                                crate::lora::apply_decode(
                                    t,
                                    &col,
                                    &mut up_col,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                        }

                        cpu::silu_mul_inplace(&mut gate_col, &up_col);

                        #[cfg(target_arch = "aarch64")]
                        {
                            Self::quantize_to_scratch(&gate_col, state);
                            self.gemv_preq(
                                &dense.down,
                                &gate_col,
                                &state.scratch.q8_scales,
                                &state.scratch.q8_quants,
                                &mut out_col,
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        self.gemv(&dense.down, &gate_col, &mut out_col);

                        // LoRA on the down projection (per-token decode hook) — input is
                        // the SiLU⊙up product in `gate_col`.
                        if let Some(lora) = &lora
                            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
                        {
                            crate::lora::apply_decode(
                                t,
                                &gate_col,
                                &mut out_col,
                                &mut state.scratch.lora_tmp,
                            );
                        }

                        for i in 0..hs {
                            ffn_out[i * n + j] = out_col[i];
                        }
                    }
                }
            }

            if debug_hidden {
                log_rms(&format!("layer {layer} ffn-out"), &ffn_out);
            }
            let t_res2 = std::time::Instant::now();
            // Second residual: hidden += ffn_out
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let total = hs * n;
                let n_chunks = total / 4;
                let h_ptr = hidden.as_mut_ptr();
                let fo_ptr = ffn_out.as_ptr();
                for i in 0..n_chunks {
                    let h_v = vld1q_f32(h_ptr.add(i * 4));
                    let fo_v = vld1q_f32(fo_ptr.add(i * 4));
                    vst1q_f32(h_ptr.add(i * 4), vaddq_f32(h_v, fo_v));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for i in 0..hs * n {
                hidden[i] += ffn_out[i];
            }
            if profile_prefill {
                t_residuals += t_res2.elapsed();
            }
            if debug_hidden {
                log_rms(&format!("layer {layer} post-ffn"), hidden);
            }
        }

        if profile_prefill {
            eprintln!(
                "[PROFILE PREFILL] n={n} | norm: {:.2}ms | conv_in: {:.2}ms | conv_core: {:.2}ms | conv_out: {:.2}ms | attn_qkv: {:.2}ms | attn_out: {:.2}ms | ffn_norm: {:.2}ms | ffn_gate_up: {:.2}ms | ffn_down: {:.2}ms | residuals: {:.2}ms",
                t_norm.as_secs_f64() * 1000.0,
                t_conv_in.as_secs_f64() * 1000.0,
                t_conv_core.as_secs_f64() * 1000.0,
                t_conv_out.as_secs_f64() * 1000.0,
                t_attn_qkv.as_secs_f64() * 1000.0,
                t_attn_out.as_secs_f64() * 1000.0,
                t_ffn_norm.as_secs_f64() * 1000.0,
                t_ffn_gate_up.as_secs_f64() * 1000.0,
                t_ffn_down.as_secs_f64() * 1000.0,
                t_residuals.as_secs_f64() * 1000.0,
            );
        }

        // seq_len tracks total tokens processed. The conv/attn blocks handle
        // per-token KV cache growth internally. We need seq_len = start_pos + n
        // at the end for the decode phase to continue from the right position.
        // Note: seq_len was NOT incremented inside the block functions — only
        // the single-token forward() does that. So set it here:
        state.seq_len = start_pos + n;
    }

    /// Run the prefill layer loop and project logits for the final (n - 1) token.
    fn prefill_layers_and_logits(
        &self,
        mut hidden: Vec<f32>,
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        self.prefill_layers_loop(&mut hidden, n, start_pos, state);

        // Extract last token, apply output norm + projection
        let mut last_hidden = vec![0.0f32; hs];
        last_hidden.copy_from_slice(&hidden[(n - 1) * hs..n * hs]);
        cpu::rmsnorm(&mut last_hidden, &self.output_norm_weight, cfg.rms_norm_eps);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&last_hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &last_hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &last_hidden, &mut logits);

        logits
    }

    /// Batched projection of post-RMSnorm row-major activations `[n × hs]`
    /// against `token_embd.weight`, yielding row-major `[n × vocab_size]` logits.
    /// Returns `None` if the embedding weight dtype isn't supported by batched
    /// GEMM. Available on aarch64 (NEON `gemm_preq`) and on x86_64 with
    /// runtime avx2+fma (int8 `gemm_preq`).
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(feature = "blas")
    ))]
    fn project_logits_batched(&self, normed_hidden: &[f32], n: usize) -> Option<Vec<f32>> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        if !transformer::batched_gemm_supports(self.embd_ref.dtype, hs) {
            return None;
        }

        let mut b_scales = vec![0.0f32; n * (hs / 32)];
        let mut b_quants = vec![0i8; n * hs];
        let mut row_buf = vec![0.0f32; hs];
        transformer::quantize_columns(
            normed_hidden,
            hs,
            n,
            &mut row_buf,
            &mut b_scales,
            &mut b_quants,
        );

        let rows = self.embd_ref.m;
        let mut out = vec![0.0f32; rows * n];
        transformer::gemm_preq(
            &self.gguf,
            &self.embd_ref,
            &b_scales,
            &b_quants,
            &mut out,
            rows,
            n,
            hs,
        );

        let mut logits = vec![0.0f32; n * vocab];
        transformer::gemm_out_to_rows(&out, rows, n, vocab, &mut logits);
        Some(logits)
    }

    /// Run the prefill layer loop and project logits for ALL n tokens.
    fn prefill_layers_and_all_logits(
        &self,
        mut hidden: Vec<f32>,
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        self.prefill_layers_loop(&mut hidden, n, start_pos, state);

        let mut all_hidden = hidden;
        for j in 0..n {
            cpu::rmsnorm(
                &mut all_hidden[j * hs..(j + 1) * hs],
                &self.output_norm_weight,
                cfg.rms_norm_eps,
            );
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        if let Some(logits) = self.project_logits_batched(&all_hidden, n) {
            return logits;
        }

        let mut all_logits = vec![0.0f32; n * cfg.vocab_size];
        for j in 0..n {
            let tok_h = &all_hidden[j * hs..(j + 1) * hs];
            let tok_l = &mut all_logits[j * cfg.vocab_size..(j + 1) * cfg.vocab_size];
            #[cfg(target_arch = "aarch64")]
            {
                Self::quantize_to_scratch(tok_h, state);
                self.gemv_preq(
                    &self.embd_ref,
                    tok_h,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    tok_l,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            self.gemv(&self.embd_ref, tok_h, tok_l);
        }
        all_logits
    }

    /// Lock-free body of `Model::forward_prefill` — does the actual
    /// embed + layer loop without consulting the prefix cache.
    /// `forward_prefill` wraps this with cache lookup/insert; cache
    /// hits bypass embedding the prefix tokens entirely and re-enter
    /// here with `start_pos = prefix_len` to prefill only the suffix.
    pub(crate) fn forward_prefill_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        assert!(
            !tokens.is_empty(),
            "forward_prefill_inner requires at least one token"
        );

        let mut hidden = vec![0.0f32; hs * n];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden[j * hs..(j + 1) * hs]);
        }

        self.prefill_layers_and_logits(hidden, n, start_pos, state)
    }

    /// Multi-token prefill producing all `[n × vocab_size]` logits (used for speculative verification).
    pub(crate) fn forward_prefill_logits_all_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        assert!(
            !tokens.is_empty(),
            "forward_prefill_logits_all_inner requires at least one token"
        );
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill_logits_all: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );

        let mut hidden = vec![0.0f32; hs * n];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden[j * hs..(j + 1) * hs]);
        }

        self.prefill_layers_and_all_logits(hidden, n, start_pos, state)
    }
}

impl Model for Lfm2Model {
    fn supports_all_logits(&self) -> bool {
        true
    }

    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.forward_prefill_logits_all_inner(tokens, start_pos, state)
    }

    fn supports_hidden_states(&self) -> bool {
        true
    }

    /// The CPU path is the one backend with routed-FFN LoRA hooks: the router
    /// delta feeds `select_experts` and the per-expert factors are applied to
    /// the selected expert's projections, both pinned by `moe_lora_parity`.
    fn supports_moe_lora(&self) -> bool {
        true
    }

    fn f16_kv_supported(&self) -> bool {
        true
    }

    /// Per-token post-final-norm hidden states, row-major `[n * hidden_size]`.
    /// Mirrors [`Self::forward`]'s embedding path (dequantize → `run_layers`,
    /// which applies the output norm) minus the logit projection — so the result
    /// is the same post-`output_norm` vector, matching llama.cpp `--pooling none`.
    /// Per-token (not batched): LFM2's batched prefill is entangled with the
    /// prefix cache, and this stateless one-shot path must not touch it; batched
    /// capture is a possible perf follow-up. `state` must start cleared at pos 0.
    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "hidden_states requires at least one token"
        );
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let mut out = Vec::with_capacity(tokens.len() * hs);
        // Reuse one embedding buffer across tokens instead of a per-token Vec.
        let mut hidden = vec![0.0f32; hs];
        for &token in tokens {
            let token_id = token as usize;
            assert!(
                token_id < cfg.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden);
            // `run_layers` ropes at `pos == seq_len` and appends one cell,
            // bumping seq_len; a cleared state walks positions 0..n.
            let pos = state.seq_len;
            self.run_layers(&mut hidden, pos, state);
            out.extend_from_slice(&hidden);
        }
        out
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LFM2 forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        // 1. Embedding lookup → layers → output norm
        let mut hidden_stack = [0.0f32; 4096];
        let mut hidden_heap;
        let hidden = if cfg.hidden_size <= 4096 {
            &mut hidden_stack[..cfg.hidden_size]
        } else {
            hidden_heap = vec![0.0f32; cfg.hidden_size];
            &mut hidden_heap[..]
        };
        self.dequantize_row_into(&self.embd_ref, token_id, hidden);
        self.run_layers(hidden, pos, state);

        // 2. Output projection (tied embeddings)
        if state.scratch.logits.len() < cfg.vocab_size {
            state.scratch.logits.resize(cfg.vocab_size, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        self.gemv_preq(
            &self.embd_ref,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.logits[..cfg.vocab_size],
        );
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(
            &self.embd_ref,
            hidden,
            &mut state.scratch.logits[..cfg.vocab_size],
        );

        state.scratch.logits[..cfg.vocab_size].to_vec()
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        assert_eq!(tokens.len(), 1, "LFM2 forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        let mut hidden_stack = [0.0f32; 4096];
        let mut hidden_heap;
        let hidden = if cfg.hidden_size <= 4096 {
            &mut hidden_stack[..cfg.hidden_size]
        } else {
            hidden_heap = vec![0.0f32; cfg.hidden_size];
            &mut hidden_heap[..]
        };
        self.dequantize_row_into(&self.embd_ref, token_id, hidden);
        self.run_layers(hidden, pos, state);

        if state.scratch.logits.len() < cfg.vocab_size {
            state.scratch.logits.resize(cfg.vocab_size, 0.0);
        }
        #[cfg(target_arch = "aarch64")]
        self.gemv_preq(
            &self.embd_ref,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.logits[..cfg.vocab_size],
        );
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(
            &self.embd_ref,
            hidden,
            &mut state.scratch.logits[..cfg.vocab_size],
        );

        crate::sampler::argmax(&state.scratch.logits[..cfg.vocab_size])
    }

    fn supports_embedding_input(&self) -> bool {
        true
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);

        // Output projection (tied embeddings)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &hidden, &mut logits);

        logits
    }

    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert_eq!(tokens.len(), 1);
        let token_id = tokens[0] as usize;
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
    }

    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "forward_prefill requires at least one token"
        );

        // Cache participation gate: only on a fresh prefill
        // (`start_pos == 0`). Continuation prefills (chunked /
        // mid-sequence) carry KV state from the prior chunk so a
        // cache restore would clobber it. TurboQuant-compressed
        // states are now supported via `LayerSnapshot::AttentionCompressed`
        // (the `!is_compressed()` exclusion was lifted in this PR).
        //
        // An active LoRA also disables the prefix cache: the batched
        // `forward_prefill_inner` still runs and applies the adapter in-batch (via
        // `apply_prefill`), but the resulting KV is adapter-specific, and the cache
        // key doesn't include the adapter — caching it would let a later base-model
        // (or different-adapter) prefill restore adapted KV. Skip both the lookup
        // and the insert while LoRA is attached.
        let cache_eligible = start_pos == 0 && state.lora.is_none();

        if cache_eligible {
            let hit = self
                .prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .find_longest_prefix(tokens);
            if let Some((snapshot, prefix_len)) = hit {
                // Compatibility gate: snapshot's KV representation
                // must match the live state's. Cross-mode restores
                // would panic or silently corrupt in
                // `InferenceState::restore` (compressed snapshot into
                // `None` slots, an f16 snapshot's u16 bytes decoded as
                // f32, or an uncompressed snapshot into a TurboQuant-
                // configured state with mismatched scratch / rotation
                // shape). Four live-state modes:
                //
                // - fully uncompressed f32 → match `Attention` snapshots.
                // - fully compressed       → match `AttentionCompressed`.
                // - f16                    → match `AttentionF16`.
                // - mixed-mode (one side compressed, the other not):
                //   `snapshot()` returns `None` so the cache never
                //   holds an entry that matches; every branch rejects.
                //
                // `state.is_compressed()` (any-side-compressed) is
                // too loose for the compressed branch: a mixed-mode
                // state would erroneously match a fully-compressed
                // snapshot and panic in `restore`. Use
                // `is_fully_compressed` for the compressed branch,
                // `kv_f16` for the f16 branch, and `!is_compressed &&
                // !kv_f16` for the uncompressed f32 branch — the last
                // guards against an f32 snapshot being restored into an
                // f16 state (whose byte widths differ). Today
                // `model_fingerprint` doesn't include the compression
                // flags, so a `--cache-dir` shared between TurboQuant,
                // f16, and uncompressed runs of the same model file
                // relies on this gate; v2 could fold the KV mode into
                // the fingerprint.
                let compatible = if snapshot.is_compressed() {
                    state.is_fully_compressed() && state.is_compressed()
                } else if snapshot.is_f16() {
                    state.kv_f16
                } else {
                    !state.is_compressed() && !state.kv_f16
                };
                if !compatible {
                    // skip; fall through to cold prefill.
                } else if prefix_len < tokens.len() && prefix_len > 0 {
                    // Strict-prefix-only: a `prefix_len == tokens.len()`
                    // hit would force `use_len = tokens.len() - 1`, but
                    // the restored conv rolling buffer reflects "after
                    // all tokens" — re-running the last token would
                    // advance the conv buffer one position past where
                    // it should be (conv layers don't gate on
                    // seq_len). Skip full hits + fall through to cold
                    // prefill. Same fix wgpu got in PR #120; tracked
                    // as 8f on the punch list.
                    let use_len = prefix_len;
                    state.restore(&snapshot);
                    let logits = self.forward_prefill_inner(&tokens[use_len..], use_len, state);
                    if let Some(snap) = state.snapshot() {
                        self.prefix_cache
                            .lock()
                            .expect("prefix_cache mutex poisoned")
                            .insert(tokens, snap);
                    }
                    return logits;
                }
            }
        }

        let logits = self.forward_prefill_inner(tokens, start_pos, state);
        if cache_eligible && let Some(snap) = state.snapshot() {
            self.prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .insert(tokens, snap);
        }
        logits
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        let id = self.cache_namespace();
        *self
            .prefix_cache
            .lock()
            .expect("prefix_cache mutex poisoned") = KvPrefixCache::new(config, &self.config, &id);
    }

    fn clear_cache(&self) {
        self.prefix_cache
            .lock()
            .expect("prefix_cache mutex poisoned")
            .clear();
    }

    /// The CPU backend allocates its KV from `InferenceState`, so there is nothing
    /// to build here — but the prefix cache's namespace still has to reflect the
    /// mode, or snapshots from different KV modes collide on disk. See
    /// [`KvCompression::cache_tag`] for what that collision costs.
    ///
    /// **First call wins**, matching the trait contract and the GPU backends: a
    /// later call with a *different* mode returns
    /// [`crate::CeraError::KvCompressionConflict`].
    ///
    /// That restriction is not about allocation here — nothing is sized by the mode
    /// on CPU. It is because the tag lives on the *model* while `KvCompression` is a
    /// *per-session* knob, and one model owns one prefix cache. Letting a second
    /// session re-tag would leave the first still writing snapshots, now into the
    /// second's namespace. For two TurboQuant seeds that is silent corruption rather
    /// than a miss: the compatibility gate only checks that both sides are
    /// compressed, and `InferenceState::restore` never validates the seed, so the
    /// first session's KV would be decoded in the wrong Hadamard basis. Two sessions
    /// in different KV modes need two model instances.
    fn configure_kv_compression(
        &self,
        compression: &KvCompression,
    ) -> Result<(), crate::CeraError> {
        let resolved = compression.resolved_for(&self.config).cache_tag();

        // Cache lock outside, tag lock inside, so the tag and the cache's namespace
        // move as one unit. Dropping the tag guard before rebuilding would let two
        // concurrent calls land tag=Y with the cache fingerprinted X — the
        // cross-mode collision this exists to prevent. `configure_cache` takes the
        // tag lock and releases it before taking the cache lock, so no cycle.
        let mut cache = self
            .prefix_cache
            .lock()
            .expect("prefix_cache mutex poisoned");
        let mut tag = self.kv_cache_tag.lock().expect("kv_cache_tag poisoned");
        match tag.as_deref() {
            Some(existing) if existing == resolved => return Ok(()),
            Some(existing) => {
                return Err(crate::CeraError::KvCompressionConflict {
                    configured: describe_cache_tag(existing),
                    requested: describe_cache_tag(&resolved),
                });
            }
            None => {}
        }
        let id = Self::namespace_for(&resolved, &self.model_id);
        *tag = Some(resolved);
        let cache_config = cache.config.clone();
        *cache = KvPrefixCache::new(cache_config, &self.config, &id);
        Ok(())
    }

    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = n_tokens;
        assert!(
            n > 0,
            "forward_prefill_from_embeddings requires at least one frame"
        );
        assert_eq!(
            embeddings.len(),
            n * hs,
            "embeddings.len() ({}) != n_tokens ({}) * hidden_size ({})",
            embeddings.len(),
            n,
            hs
        );

        // An active LoRA is applied in-batch by `prefill_layers_and_logits` (via
        // `apply_prefill` after each projection GEMM), so embedding-input
        // (multimodal) spans get the adapter too — no per-frame fallback needed.

        let hidden = embeddings.to_vec();
        self.prefill_layers_and_logits(hidden, n, start_pos, state)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn turboquant_supported(&self) -> bool {
        let head_dim = self.config.hidden_size / self.config.n_heads;
        head_dim.is_power_of_two()
    }

    fn supports_kv_shift(&self) -> bool {
        // CPU LFM2 implements shift with RoPE re-rotation. The wgpu and
        // Metal overrides mirror this with a GPU-side shift shader, and
        // report `false` only while a TurboQuant cache is active.
        true
    }

    fn shift_kv(&self, state: &mut crate::kv_cache::InferenceState, n_keep: usize, shift: usize) {
        let head_dim = self.config.hidden_size / self.config.n_heads;
        state.shift_kv_with_rope(
            n_keep,
            shift,
            self.config.rope_theta,
            head_dim,
            &self.config.kv_heads_per_layer,
            crate::backend::cpu::RopeType::Neox,
            None,
        );
    }
}

// ── GPU weight source ───────────────────────────────────────────────────────
//
// Drives the wgpu loader (`gpu_lfm2.rs`) for LFM2. Conv layers expose
// `conv_*` refs; attention layers expose `attn_*` refs + QK-norm. LFM2 has no
// QKV bias / untied output / Llama-3 freq-factors, uses NEOX RoPE, identity
// scalars, and supports the batched-prefill GPU path.
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl crate::model::gpu_weight_source::GpuWeightSource for Lfm2Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }
    fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }
    fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }
    fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        Lfm2Model::attn_q_norm_weight(self, layer)
    }
    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        Lfm2Model::attn_k_norm_weight(self, layer)
    }
    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        Lfm2Model::conv_weight(self, layer)
    }
    fn attn_q_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn attn_k_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn attn_v_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn rope_freqs(&self) -> Option<&[f32]> {
        None
    }

    fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        transformer::weight_data(&self.gguf, wref)
    }
    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32> {
        transformer::dequantize_weight(&self.gguf, wref)
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        None
    }
    fn ffn_gate_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.gate)
    }
    fn ffn_up_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.up)
    }
    fn ffn_down_ref(&self, layer: usize) -> Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.down)
    }
    fn moe_refs(&self, layer: usize) -> Option<&MoeFfnRefs> {
        match &self.layer_refs[layer].ffn {
            FfnRefs::Dense(_) => None,
            FfnRefs::Moe(m) => Some(m),
        }
    }
    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].shortconv_in_proj.as_ref()
    }
    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].shortconv_out_proj.as_ref()
    }
    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_q.as_ref()
    }
    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_k.as_ref()
    }
    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_v.as_ref()
    }
    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_output.as_ref()
    }

    fn rope_type(&self) -> crate::backend::cpu::RopeType {
        crate::backend::cpu::RopeType::Neox
    }
    fn supports_batched_prefill(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod conv_kernel_size_tests {
    use super::validate_conv_kernel_size;

    #[test]
    fn accepts_the_shipped_value_and_absence() {
        assert_eq!(validate_conv_kernel_size(Some(3)).unwrap(), Some(3));
        // Absent metadata falls back to 3 downstream, so it stays None here.
        assert_eq!(validate_conv_kernel_size(None).unwrap(), None);
    }

    #[test]
    fn accepts_the_register_array_bounds() {
        for k in [2usize, 4] {
            assert_eq!(validate_conv_kernel_size(Some(k)).unwrap(), Some(k));
        }
    }

    #[test]
    fn rejects_values_the_kernels_cannot_honor() {
        // 0 and 1 underflow `d_conv = kernel_size - 1`; anything above 4 overruns
        // the batched kernel's fixed-size `w_local` / `rb` registers, which used
        // to make it silently skip the dispatch and emit wrong prefill logits.
        for k in [0usize, 1, 5, 64, usize::MAX] {
            let err = validate_conv_kernel_size(Some(k)).unwrap_err().to_string();
            assert!(
                err.contains("l_cache must be in 2..=4"),
                "unexpected error for k={k}: {err}"
            );
        }
    }
}

#[cfg(test)]
mod moe_routing_tests {
    use super::select_experts;

    /// The bias steers *selection*; the weights come from the unbiased
    /// probabilities. Both halves are asserted, because getting only the first
    /// half right is the failure mode that still produces fluent text.
    #[test]
    fn bias_ranks_experts_but_does_not_weight_them() {
        // Expert 1 has the lower probability but a large enough bias to be
        // ranked first; expert 0 wins on raw probability.
        let probs = [0.40, 0.30, 0.05, 0.01];
        let biases = [0.00, 0.50, 0.00, 0.00];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 2, &mut selected);

        let picked: Vec<usize> = selected.iter().map(|&(e, _)| e).collect();
        assert_eq!(picked, vec![1, 0], "bias must decide the ranking");

        // Weights are the unbiased probs renormalized: 0.30/0.70, 0.40/0.70.
        // Note they *ascend*; a biased-weight implementation would emit
        // descending values and pass a "weights are sorted" check.
        let weights: Vec<f32> = selected.iter().map(|&(_, w)| w).collect();
        assert!((weights[0] - 0.30 / 0.70).abs() < 1e-6, "got {weights:?}");
        assert!((weights[1] - 0.40 / 0.70).abs() < 1e-6, "got {weights:?}");
        assert!(
            weights[0] < weights[1],
            "unbiased weights must not be forced into descending order: {weights:?}"
        );
    }

    /// Regression against real llama.cpp output: `ffn_moe_topk-2` /
    /// `ffn_moe_weights_norm-2` for token 3 of "The capital of France is"
    /// (BOS-prefixed) on LFM2.5-8B-A1B-Q4_0. This token is the useful one
    /// precisely because its normalized weights are non-monotonic
    /// (0.3474 < 0.3556), which only happens when ranking and weighting use
    /// different scores.
    #[test]
    fn matches_llama_cpp_non_monotonic_token() {
        // Reconstructed from the reference dump: expert 21 ranks first on
        // biased score despite expert 17 having the higher raw probability.
        let mut probs = [0.0f32; 32];
        let mut biases = [0.0f32; 32];
        probs[21] = 0.3474;
        probs[17] = 0.3556;
        probs[6] = 0.1866;
        probs[12] = 0.1103;
        biases[21] = 0.10;

        let mut selected = Vec::new();
        select_experts(&probs, &biases, 4, &mut selected);

        assert_eq!(
            selected.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
            vec![21, 17, 6, 12]
        );
        let weights: Vec<f32> = selected.iter().map(|&(_, w)| w).collect();
        // The reference weights already sum to 1, so renormalizing is a no-op
        // and they should come back unchanged.
        for (got, want) in weights.iter().zip([0.3474, 0.3556, 0.1866, 0.1103]) {
            assert!((got - want).abs() < 1e-3, "got {weights:?}");
        }
    }

    /// Ties resolve to the lower expert index, matching `ggml_argsort_top_k`.
    /// Without the reversed index tiebreak `max_by` keeps the last maximum and
    /// silently picks the higher index.
    #[test]
    fn ties_break_toward_the_lower_index() {
        let probs = [0.25, 0.25, 0.25, 0.25];
        let biases = [0.0; 4];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 2, &mut selected);
        assert_eq!(
            selected.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    /// An all-zero gate must not divide by zero. llama.cpp clamps the divisor
    /// to f16's smallest positive normal; the weights stay finite.
    #[test]
    fn zero_gate_does_not_divide_by_zero() {
        let probs = [0.0f32; 4];
        let biases = [0.0f32; 4];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 2, &mut selected);
        assert!(selected.iter().all(|&(_, w)| w.is_finite()), "{selected:?}");
    }

    /// Asking for more experts than exist must stop at the expert count rather
    /// than spin or emit duplicates.
    #[test]
    fn n_used_saturates_at_the_expert_count() {
        let probs = [0.5, 0.3];
        let biases = [0.0, 0.0];
        let mut selected = Vec::new();
        select_experts(&probs, &biases, 8, &mut selected);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 0);
        assert_eq!(selected[1].0, 1);
    }
}

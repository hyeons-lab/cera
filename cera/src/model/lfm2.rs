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
        heap_biased = probs.iter().zip(biases).map(|(&p, &b)| p + b).collect::<Vec<_>>();
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
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                conv_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.shortconv.conv.weight"))?
                        .to_f32_vec(),
                ));
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
                }))
            } else {
                FfnRefs::Dense(DenseFfnRefs {
                    gate: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?
                        .with_repack(&gguf),
                    up: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?
                        .with_repack(&gguf),
                    down: Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?
                        .with_repack(&gguf),
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

        for (i, &(expert, weight)) in selected.iter().enumerate() {
            // Restore the Q8_0 quantization of `ffn_input`. The previous
            // iteration's down projection re-quantized this scratch to hold its
            // SwiGLU product, so the gate/up GEMVs below would otherwise read
            // that instead of the layer input. Done at the top of the body, not
            // the bottom, so it does not also run after the final expert where
            // nothing consumes it. The caller guarantees it is already correct
            // on entry, so the first iteration re-does work that is already
            // valid; that is one quantization per layer, against one per expert
            // for the bottom-of-loop form.
            #[cfg(target_arch = "aarch64")]
            if i > 0 {
                Self::quantize_to_scratch(ffn_input, state);
            }

            #[cfg(target_arch = "aarch64")]
            {
                transformer::gemv_preq(
                    &self.gguf,
                    &moe.gate[expert],
                    ffn_input,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.gate[..ff],
                );
                transformer::gemv_preq(
                    &self.gguf,
                    &moe.up[expert],
                    ffn_input,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.up[..ff],
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
                // layer input, so it needs its own Q8_0 quantization, and it
                // overwrites the q8 scratch the gate/up GEMVs above read, which
                // is why this runs after both of them for every expert.
                let nb = ff / 32;
                state.scratch.q8_scales.resize(nb, 0.0);
                state.scratch.q8_quants.resize(ff, 0);
                unsafe {
                    crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                        &state.scratch.gate[..ff],
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                }
                transformer::gemv_preq(
                    &self.gguf,
                    &moe.down[expert],
                    &state.scratch.gate[..ff],
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.moe_expert_out[..hidden_size],
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
        for j in 0..n {
            (0..hs).for_each(|i| col[i] = ffn_input[i * n + j]);

            // `forward_moe_ffn` reads the pre-quantized column on aarch64,
            // exactly as the dense per-token fallback does.
            #[cfg(target_arch = "aarch64")]
            Self::quantize_to_scratch(col, state);

            self.forward_moe_ffn(layer, moe, hs, col, state);

            (0..hs).for_each(|i| ffn_out[i * n + j] = state.scratch.out[i]);
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
    fn forward_conv_block(&self, layer: usize, hidden: &[f32], state: &mut InferenceState) {
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
        for (out, (bi, xi)) in conv_scratch.iter_mut().zip(b.iter().zip(x.iter())) {
            *out = bi * xi;
        }
        // conv_scratch now holds bx

        // Depthwise conv1d with valid convolution using rolling buffer
        let LayerState::Conv { buffer } = &mut state.layers[layer] else {
            panic!("expected Conv state for layer {layer}");
        };

        let out_buf = &mut state.scratch.out[..hidden_size];
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
        }

        // o = c ⊙ conv_out (second gate), reuse conv_scratch
        for (o, (ci, co)) in conv_scratch.iter_mut().zip(c.iter().zip(out_buf.iter())) {
            *o = ci * co;
        }

        // out_proj: hidden → hidden, write result into out_buf
        self.gemv(out_proj, conv_scratch, out_buf);
        // LoRA on the conv out_proj — `out_buf += scale·B·(A·conv_scratch)`, where
        // conv_scratch is the gated conv output that feeds out_proj.
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
        {
            crate::lora::apply_decode(t, conv_scratch, out_buf, &mut state.scratch.lora_tmp);
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
            self.gemv_preq(
                refs.attn_q.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                q,
            );
            self.gemv_preq(
                refs.attn_k.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                k,
            );
            self.gemv_preq(
                refs.attn_v.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                v,
            );
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
        let out = &mut state.scratch.out[..cfg.hidden_size];
        self.gemv(
            refs.attn_output.as_ref().unwrap(),
            &state.scratch.attn_out[..cfg.hidden_size],
            out,
        );
        // LoRA on the output projection (input = the attention output).
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
        {
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

        for i in 0..cfg.n_layers {
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            #[cfg(target_arch = "aarch64")]
            Self::quantize_to_scratch(&normed, state);

            if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &normed, state);
            } else {
                self.forward_attn_block(i, &normed, pos, state);
            }

            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            ffn_input.copy_from_slice(hidden);
            cpu::rmsnorm(&mut ffn_input, &self.ffn_norm_weights[i], cfg.rms_norm_eps);

            // SwiGLU FFN via the shared helper. On aarch64 it consumes the
            // pre-quantized ffn_input, so quantize first (same contract as the
            // llama/qwen per-token path).
            #[cfg(target_arch = "aarch64")]
            Self::quantize_to_scratch(&ffn_input, state);

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

        cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
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
    fn prefill_layers_and_logits(
        &self,
        mut hidden: Vec<f32>,
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
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
        log_rms("input (pre-layer-0)", &hidden);

        // Per-layer loop — pre-allocate all large buffers outside the loop
        let mut normed = vec![0.0f32; hs * n];
        let mut block_out = vec![0.0f32; hs * n];
        let mut ffn_input = vec![0.0f32; hs * n];
        let mut ffn_out = vec![0.0f32; hs * n];
        let mut norm_col = vec![0.0f32; hs];
        let mut ffn_col = vec![0.0f32; hs];
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
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut gate_mat = vec![0.0f32; is * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut up_mat = vec![0.0f32; is * n];
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

        for layer in 0..cfg.n_layers {
            // RMSnorm each column independently
            for j in 0..n {
                for i in 0..hs {
                    norm_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut norm_col,
                    &self.attn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    normed[i * n + j] = norm_col[i];
                }
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
                    // Require BOTH projections to be batchable: a mixed-dtype conv
                    // block would leave the second matrix silently uncomputed. Any
                    // other combo falls through to the per-token fallback — loudly,
                    // because a quiet fallback here costs ~4x prefill.
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
                        // Phase 1: Batch in_proj GEMM: normed[hs×n] → proj_mat[3*hs × n]
                        // quantize_columns is only needed for the NEON fallback. With BLAS
                        // on, the SGEMM path consumes f32 directly so this work is skipped.
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &normed,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                in_proj,
                                &normed,
                                &mut proj_mat,
                                3 * hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        // Sliced, like the LoRA call below: `proj_mat` is sized
                        // `max(3*hs, hs + 2*kv_dim) * n` because it is shared
                        // with the attention projection, so it can be longer
                        // than this GEMM's `3*hs*n` output. `gemm_preq` slices
                        // defensively too, but the invariant belongs where the
                        // over-long buffer is created.
                        #[cfg(not(feature = "blas"))]
                        transformer::gemm_preq(
                            &self.gguf,
                            in_proj,
                            &bq_scales,
                            &bq_quants,
                            &mut proj_mat[..3 * hs * n],
                            3 * hs,
                            n,
                            hs,
                        );

                        // LoRA on the conv in_proj — `proj_mat[3hs×n] += scale·B·(A·normed)`,
                        // applied to the full projection before the B/C/x split. Mirrors
                        // the per-token `forward_conv_block` path for the batched prefill.
                        // `proj_mat` is sized `proj_rows = max(3·hs, hs+2·max_kv_dim)`, which
                        // can exceed `3·hs`; slice to the conv's `3·hs` rows so the length
                        // matches `apply_prefill`'s `t.d × n` contract exactly.
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
                        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
                        let d_conv = kernel_size - 1;
                        let conv_weight = self.conv_weights[layer].as_ref().unwrap();
                        for j in 0..n {
                            let proj = &mut state.scratch.conv_proj[..3 * hs];
                            for i in 0..hs {
                                proj[i] = proj_mat[i * n + j];
                                proj[hs + i] = proj_mat[(hs + i) * n + j];
                                proj[2 * hs + i] = proj_mat[(2 * hs + i) * n + j];
                            }
                            let (b, rest) = proj.split_at(hs);
                            let (c_slice, x_slice) = rest.split_at(hs);

                            let conv_scratch = &mut state.scratch.conv_scratch[..hs];
                            for i in 0..hs {
                                conv_scratch[i] = b[i] * x_slice[i];
                            }

                            let LayerState::Conv { buffer } = &mut state.layers[layer] else {
                                panic!("expected Conv state for layer {layer}");
                            };
                            let out_buf = &mut state.scratch.out[..hs];
                            for ch in 0..hs {
                                let mut sum = 0.0f32;
                                for k in 0..d_conv {
                                    sum += buffer[k * hs + ch] * conv_weight[ch * kernel_size + k];
                                }
                                sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                                out_buf[ch] = sum;
                            }
                            if d_conv > 0 {
                                if d_conv > 1 {
                                    buffer.copy_within(hs.., 0);
                                }
                                let last_slot = (d_conv - 1) * hs;
                                buffer[last_slot..last_slot + hs].copy_from_slice(conv_scratch);
                            }

                            for i in 0..hs {
                                out_proj_input[i * n + j] = c_slice[i] * out_buf[i];
                            }
                        }

                        // Phase 3: Batch out_proj GEMM
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &out_proj_input,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                out_proj,
                                &out_proj_input,
                                &mut block_out,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        transformer::gemm_preq(
                            &self.gguf,
                            out_proj,
                            &bq_scales,
                            &bq_quants,
                            &mut block_out,
                            hs,
                            n,
                            hs,
                        );

                        // LoRA on the conv out_proj — `block_out[hs×n] += scale·B·(A·in)`,
                        // where `in` is the gated conv output (`out_proj_input`), before
                        // the residual add.
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

                        // Phase 1: Batch Q/K/V GEMM
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &normed,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_q_ref,
                                &normed,
                                &mut q_mat,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_k_ref,
                                &normed,
                                &mut k_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_v_ref,
                                &normed,
                                &mut v_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        {
                            transformer::gemm_preq(
                                &self.gguf, attn_q_ref, &bq_scales, &bq_quants, &mut q_mat, hs, n,
                                hs,
                            );
                            transformer::gemm_preq(
                                &self.gguf,
                                attn_k_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut k_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                            );
                            transformer::gemm_preq(
                                &self.gguf,
                                attn_v_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut v_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                            );
                        }

                        // LoRA on Q/K/V — added to the projection outputs before
                        // QK-norm/RoPE, input is the normed hidden `[hs×n]`.
                        if let Some(lora) = &lora {
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnQ) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut q_mat,
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
                        // Hoist tq state capture so the reserve block can match the
                        // exact same condition as the actual append path below, and
                        // so the per-token loop can key off pre-computed bools.
                        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
                        let tq_config = state.tq_config.as_ref();
                        // Needed to encode keys + values (append path).
                        let will_compress_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_encode_scratch.is_some();
                        // Needed to read compressed keys/values (attention path).
                        let will_read_compressed_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_query_scratch.is_some();
                        // f16 KV: append converts to half; Pass B widens back to
                        // an f32 scratch so the flash/naive kernels stay f32-only.
                        // Mutually exclusive with TurboQuant (a distinct
                        // KvCompression variant), so `use_f16` is never true
                        // alongside the `will_compress_kv`/`will_read_compressed_kv`
                        // branches.
                        let use_f16 = state.kv_f16;

                        // Pre-reserve KV cache to avoid repeated reallocations.
                        // Keys and values are handled independently — whichever
                        // side is compressed reserves the packed buffers;
                        // the other side reserves the f32 flat cache.
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
                        // Processes all n tokens sequentially (O(n) per token).
                        // After this loop, q_mat contains post-RoPE Q and the
                        // KV cache is fully populated through start_pos + n - 1.
                        for j in 0..n {
                            let pos = start_pos + j;
                            let q = &mut state.scratch.q[..hs];
                            let k = &mut state.scratch.k[..kv_dim];
                            let v = &mut state.scratch.v[..kv_dim];
                            for i in 0..hs {
                                q[i] = q_mat[i * n + j];
                            }
                            for i in 0..kv_dim {
                                k[i] = k_mat[i * n + j];
                                v[i] = v_mat[i * n + j];
                            }

                            // QK norm
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

                            // Write processed Q back to q_mat so flash attention
                            // can read it. K/V go into the cache below.
                            for i in 0..hs {
                                q_mat[i * n + j] = q[i];
                            }

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
                                match (will_compress_kv, compressed_values.as_mut()) {
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
                        }

                        // ── Pass B: attention ────────────────────────────────
                        // The KV cache is now fully populated. Branch on
                        // whether TurboQuant compressed KV is active.
                        let use_tq = will_read_compressed_kv
                            && match &state.layers[layer] {
                                LayerState::Attention {
                                    compressed_keys,
                                    compressed_values,
                                    ..
                                } => compressed_keys.is_some() || compressed_values.is_some(),
                                _ => false,
                            };

                        // Flash attention (tiled + rayon) is faster at longer
                        // prompts. Below the threshold the overhead of the
                        // two-pass decomposition + online softmax exceeds the
                        // naive NEON path, so fall back.
                        // Flash attention (tiled + rayon) is faster than the naive
                        // NEON path only for longer prompts. The crossover is around
                        // pp200 on Apple Silicon (measured: naive wins at pp128 by 5%,
                        // flash wins at pp252 by 6%). Use 256 to avoid regressions.
                        const FLASH_ATTN_THRESHOLD: usize = 256;
                        let use_flash = !use_tq && n >= FLASH_ATTN_THRESHOLD;

                        // f16 mode: widen the half KV cache into the reused f32
                        // scratch ONCE (mirrors the dense path) so the f32-only
                        // flash/naive kernels below can read it. Only one of
                        // flash/tq/naive runs per layer, and TQ is never f16, so
                        // a single widen here covers whichever branch is taken.
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
                            // f32 path: flash attention over the full KV cache,
                            // parallel across *query heads* via `par_rows_n_chunked`
                            // — the pinned RowPool on native, rayon on wasm32. On
                            // native this shares the one prefill pool with the GEMM
                            // instead of a second full-width pool spin-waiting
                            // through attention's phase (the oversubscription the
                            // GEMM consolidation removed).
                            //
                            // Splitting per KV head (the pre-widening layout)
                            // capped parallelism at n_kv_heads — half-idle on a
                            // core-count host, the same under-utilization #289
                            // fixed for the dense transformers. One row per query
                            // head gives n_heads-way; group members of one KV head
                            // re-read that head's K/V (an L3 hit at these sizes),
                            // and full core use more than pays for it.
                            //
                            // Byte-identical to the per-KV-head split: KV head
                            // kv_h's block was [group_size, n, head_dim] at
                            // kv_h*group_size*n*head_dim, member g at +g*n*head_dim
                            // — i.e. head h = kv_h*group_size + g at exactly
                            // h*n*head_dim. So per-head chunking writes the same
                            // bytes and the scatter collapses to a flat h loop.
                            // Bit-identical: each (head, query) output is computed
                            // independently.
                            //
                            // f16 reads the pre-widened f32 scratch (above); f32
                            // reads the cache directly. Flash kernel stays f32-only.
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

                            // `min_chunk_rows = 1`: a head is a heavy row and there
                            // are only n_heads of them, so the default steal floor
                            // would hand all heads to a couple of workers. One head
                            // per steal unit lets every worker take a head.
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
                                    n,
                                    kv_dim,
                                    kv_h * head_dim,
                                    head_dim,
                                    scale,
                                    start_pos,
                                );
                            });

                            // Scatter-copy: flash_buf [n_heads, n, head_dim]
                            // → out_proj_input [hs, n] stride-n. Loop order
                            // d-then-j gives sequential writes to out_proj_input
                            // (stride 1) and small-stride reads from flash_buf
                            // (stride head_dim). Head h's block sits at
                            // h*n*head_dim, so the old kv_h/g nesting collapses to
                            // a flat h loop.
                            for h in 0..n_heads {
                                let src_base = h * n * head_dim;
                                for d in 0..head_dim {
                                    let row_idx = (h * head_dim + d) * n;
                                    for j in 0..n {
                                        out_proj_input[row_idx + j] =
                                            flash_buf[src_base + j * head_dim + d];
                                    }
                                }
                            }
                        } else if use_tq {
                            // TurboQuant path: per-token attention using the
                            // compressed KV cache. Re-extract post-RoPE Q from
                            // q_mat for each token.
                            // `reserve` counts from the current length, and a
                            // preceding decode may have left this buffer laid
                            // out as one row per head. Drop that layout first so
                            // the hint means "capacity for this prefill" rather
                            // than "decode arena plus this prefill"; the loop
                            // below resizes and the kernels fully overwrite.
                            state.scratch.scores.clear();
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let q = &mut state.scratch.q[..hs];
                                for i in 0..hs {
                                    q[i] = q_mat[i * n + j];
                                }

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
                                let attn_out = &mut state.scratch.attn_out[..hs];
                                let q = &state.scratch.q[..hs];
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

                                for i in 0..hs {
                                    out_proj_input[i * n + j] = attn_out[i];
                                }
                            }
                        } else {
                            // Short-prompt f32 fallback: naive per-token
                            // attention (no tiling, no rayon). Faster than
                            // flash attention when n < FLASH_ATTN_THRESHOLD
                            // because the attention portion is trivially small.
                            // f16 reads the pre-widened f32 scratch (above); f32
                            // reads the cache directly. The widened slice's
                            // len()/kv_dim equals the real seq_len, so the
                            // per-token seq_len clamp below is unchanged.
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
                            // `reserve` counts from the current length, and a
                            // preceding decode may have left this buffer laid
                            // out as one row per head. Drop that layout first so
                            // the hint means "capacity for this prefill" rather
                            // than "decode arena plus this prefill"; the loop
                            // below resizes and the kernels fully overwrite.
                            state.scratch.scores.clear();
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let seq_len = (start_pos + j + 1).min(k_cache.len() / kv_dim);
                                // Q is already post-RoPE in q_mat from Pass A;
                                // re-extract into scratch for the naive path.
                                for i in 0..hs {
                                    state.scratch.q[i] = q_mat[i * n + j];
                                }
                                let q = &state.scratch.q[..hs];
                                let attn_out = &mut state.scratch.attn_out[..hs];
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

                                for i in 0..hs {
                                    out_proj_input[i * n + j] = attn_out[i];
                                }
                            }
                        }

                        // Phase 3: Batch output projection GEMM
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &out_proj_input,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_output_ref,
                                &out_proj_input,
                                &mut block_out,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        transformer::gemm_preq(
                            &self.gguf,
                            attn_output_ref,
                            &bq_scales,
                            &bq_quants,
                            &mut block_out,
                            hs,
                            n,
                            hs,
                        );

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
                        self.forward_conv_block(layer, &col, state);
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

            // Residual: hidden += block_out
            for i in 0..hs * n {
                hidden[i] += block_out[i];
            }
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(&format!("layer {layer} ({block_kind}) post-block"), &hidden);
            }

            // FFN pre-norm each column
            for j in 0..n {
                for i in 0..hs {
                    ffn_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut ffn_col,
                    &self.ffn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    ffn_input[i * n + j] = ffn_col[i];
                }
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

                    // Gate + Up via batched GEMM
                    #[cfg(feature = "blas")]
                    {
                        transformer::try_blas_prefill_gemm(
                            &self.gguf,
                            &dense.gate,
                            &ffn_input,
                            &mut gate_mat,
                            is,
                            n,
                            hs,
                            &mut state.scratch.dequant_weight_scratch,
                        );
                        transformer::try_blas_prefill_gemm(
                            &self.gguf,
                            &dense.up,
                            &ffn_input,
                            &mut up_mat,
                            is,
                            n,
                            hs,
                            &mut state.scratch.dequant_weight_scratch,
                        );
                    }
                    #[cfg(not(feature = "blas"))]
                    {
                        transformer::gemm_preq(
                            &self.gguf,
                            &dense.gate,
                            &bq_scales,
                            &bq_quants,
                            &mut gate_mat,
                            is,
                            n,
                            hs,
                        );
                        transformer::gemm_preq(
                            &self.gguf,
                            &dense.up,
                            &bq_scales,
                            &bq_quants,
                            &mut up_mat,
                            is,
                            n,
                            hs,
                        );
                    }

                    // LoRA on gate/up — BEFORE the SiLU+mul, input is the normed FFN
                    // input `[hs×n]`.
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

                    // Fused SiLU+mul (row-major is×n)
                    cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);

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
                        transformer::try_blas_prefill_gemm(
                            &self.gguf,
                            &dense.down,
                            &gate_mat,
                            &mut ffn_out,
                            hs,
                            n,
                            is,
                            &mut state.scratch.dequant_weight_scratch,
                        );
                    }
                    #[cfg(not(feature = "blas"))]
                    transformer::gemm_preq(
                        &self.gguf,
                        &dense.down,
                        &dq_scales,
                        &dq_quants,
                        &mut ffn_out,
                        hs,
                        n,
                        is,
                    );

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
            // Second residual
            for i in 0..hs * n {
                hidden[i] += ffn_out[i];
            }
            if debug_hidden {
                log_rms(&format!("layer {layer} post-ffn"), &hidden);
            }
        }

        // seq_len tracks total tokens processed. The conv/attn blocks handle
        // per-token KV cache growth internally. We need seq_len = start_pos + n
        // at the end for the decode phase to continue from the right position.
        // Note: seq_len was NOT incremented inside the block functions — only
        // the single-token forward() does that. So set it here:
        state.seq_len = start_pos + n;

        // Extract last token, apply output norm + projection
        let mut last_hidden = vec![0.0f32; hs];
        for i in 0..hs {
            last_hidden[i] = hidden[i * n + (n - 1)];
        }
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

        // Embed all tokens → hidden[hs × n] with stride n (token j at
        // indices [j, n+j, 2n+j, ...]). Layer loop + output projection
        // is shared with `forward_prefill_from_embeddings` via
        // `prefill_layers_and_logits`.
        let mut hidden = vec![0.0f32; hs * n];
        let mut emb_buf = vec![0.0f32; hs];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut emb_buf);
            for i in 0..hs {
                hidden[i * n + j] = emb_buf[i];
            }
        }

        self.prefill_layers_and_logits(hidden, n, start_pos, state)
    }
}

impl Model for Lfm2Model {
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
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        self.run_layers(&mut hidden, pos, state);

        // 2. Output projection (tied embeddings)
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

        // Transpose row-major embeddings (frame j at [j*hs..(j+1)*hs])
        // into column-major hidden (token j's channel i at [i*n + j]) —
        // same layout `forward_prefill` builds via the embed-table
        // lookup. After this, the layer loop + output projection in
        // `prefill_layers_and_logits` is identical to the token path.
        let mut hidden = vec![0.0f32; hs * n];
        for j in 0..n {
            let frame = &embeddings[j * hs..(j + 1) * hs];
            for i in 0..hs {
                hidden[i * n + j] = frame[i];
            }
        }

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

//! LoRA (Low-Rank Adaptation) adapters.
//!
//! An adapter adds a low-rank delta to selected weight matrices: for a base
//! projection `y = W·x`, the adapted output is
//!
//! ```text
//! y = W·x + scale · B·(A·x)
//! ```
//!
//! where `A` is `[rank × k]` (down-projection, input width `k`), `B` is
//! `[d × rank]` (up-projection, output width `d`), and `scale = alpha / rank`.
//! Applying it at runtime (rather than merging into `W`) keeps the base weights
//! quantized and untouched, so adapters can be hot-swapped / unloaded per session.
//!
//! This module is the **loader + math**: it parses adapter files (GGUF from
//! llama.cpp's `convert_lora_to_gguf`, or PEFT `.safetensors`) into f32 factors
//! and exposes the pure apply helpers. Wiring an adapter into the model forward
//! passes lives in the backends (a later PR).
//!
//! Factors are stored **row-major, pre-dequantized to f32** (`a[i·k + j]` is
//! `A[i][j]`, `b[i·rank + j]` is `B[i][j]`). An ordinary projection's factors are
//! tiny (rank ≤ ~64), so f32 keeps the correction exact and gives one shared
//! apply path across every backend with no dtype dispatch.
//!
//! **Mixture-of-experts adapters are not tiny.** A routed projection carries one
//! factor pair *per expert*, so its footprint scales with the expert count: on
//! LFM2.5-8B-A1B (32 experts, 22 routed layers, three projections each) a rank-32
//! adapter is roughly a gigabyte of f32 once loaded. The caps here bound the two
//! factors separately ([`MAX_LORA_RANK`] the rank, `MAX_LORA_EXPERTS` the expert
//! count) and both sit far above any real adapter, so neither constrains the
//! product in practice: an embedder on a memory budget has to size the adapter
//! itself. Storing the factors quantized would change the apply path on every
//! backend, so it is deliberately not done here.

#[cfg(any(feature = "mmap", not(target_arch = "wasm32")))]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::gguf::GgufFile;

/// Number of distinct [`LoraTarget`]s, i.e. the length of `LoraTarget::ALL` and
/// the width of every per-target array indexed by [`LoraTarget::index`].
///
/// Public because [`LoraTarget::index`] and [`LoraTarget::ALL`] are: anything
/// that indexes by `index()` needs the width, and spelling it as a literal is a
/// runtime out-of-bounds rather than a compile error once a target is added.
/// The GPU backends (`model/gpu_lfm2.rs`, `model/metal_lfm2.rs`) size their own
/// per-target arrays from it for exactly that reason, though being in this crate
/// they would not need it to be public.
pub const LORA_TARGET_COUNT: usize = 13;

/// The linear-projection targets an adapter can modify: the four attention
/// projections, the three FFN projections, LFM2's two gated-conv (`shortconv`)
/// projections, and the mixture-of-experts router plus its three per-expert
/// projections. llama.cpp's `convert_lora_to_gguf` emits deltas for all of
/// these, so cera adapts them all; otherwise an adapter trained against
/// llama.cpp is only partially applied and the resulting hidden states diverge.
///
/// Declaration order is `index()` order, which `Ord` and `ALL` both rely on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum LoraTarget {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnGate,
    FfnUp,
    FfnDown,
    /// LFM2 gated-conv input projection (`shortconv.in_proj`), `hidden → 3·hidden`.
    ShortconvInProj,
    /// LFM2 gated-conv output projection (`shortconv.out_proj`), `hidden → hidden`.
    ShortconvOutProj,
    /// MoE router projection (`ffn_gate_inp`), `hidden → n_expert`. Adapted
    /// because llama.cpp routes through `build_lora_mm(gate_inp, cur)`: skipping
    /// it would leave expert *selection* unadapted while the experts themselves
    /// were adapted, which diverges without ever looking wrong.
    FfnGateInp,
    /// MoE per-expert gate projection (`ffn_gate_exps`), `hidden → expert_ff_len`.
    FfnGateExps,
    /// MoE per-expert up projection (`ffn_up_exps`), `hidden → expert_ff_len`.
    FfnUpExps,
    /// MoE per-expert down projection (`ffn_down_exps`), `expert_ff_len → hidden`.
    FfnDownExps,
}

impl LoraTarget {
    /// Every target, in `index()` order.
    pub const ALL: [LoraTarget; LORA_TARGET_COUNT] = [
        LoraTarget::AttnQ,
        LoraTarget::AttnK,
        LoraTarget::AttnV,
        LoraTarget::AttnOutput,
        LoraTarget::FfnGate,
        LoraTarget::FfnUp,
        LoraTarget::FfnDown,
        LoraTarget::ShortconvInProj,
        LoraTarget::ShortconvOutProj,
        LoraTarget::FfnGateInp,
        LoraTarget::FfnGateExps,
        LoraTarget::FfnUpExps,
        LoraTarget::FfnDownExps,
    ];

    /// Dense array index (`0..LORA_TARGET_COUNT`) for `LoraLayer::targets`.
    pub fn index(self) -> usize {
        match self {
            LoraTarget::AttnQ => 0,
            LoraTarget::AttnK => 1,
            LoraTarget::AttnV => 2,
            LoraTarget::AttnOutput => 3,
            LoraTarget::FfnGate => 4,
            LoraTarget::FfnUp => 5,
            LoraTarget::FfnDown => 6,
            LoraTarget::ShortconvInProj => 7,
            LoraTarget::ShortconvOutProj => 8,
            LoraTarget::FfnGateInp => 9,
            LoraTarget::FfnGateExps => 10,
            LoraTarget::FfnUpExps => 11,
            LoraTarget::FfnDownExps => 12,
        }
    }

    /// Whether this target's base weight is a *stacked* per-expert tensor, and
    /// so carries one set of low-rank factors per expert rather than one set
    /// overall. Drives both the load-time split and which accessor
    /// ([`LoraAdapterWeights::get`] vs [`LoraAdapterWeights::get_expert`])
    /// returns the delta.
    pub fn is_expert(self) -> bool {
        matches!(
            self,
            LoraTarget::FfnGateExps | LoraTarget::FfnUpExps | LoraTarget::FfnDownExps
        )
    }

    /// Whether this target only exists on a routed (mixture-of-experts) layer.
    ///
    /// Wider than [`Self::is_expert`] by exactly the router, `ffn_gate_inp`,
    /// which is not a stacked tensor but is just as meaningless on a dense
    /// layer. Shared by `validate_dims` and
    /// [`LoraAdapterWeights::has_moe_deltas`] so the set cannot drift between
    /// them.
    fn is_routed_ffn(self) -> bool {
        self.is_expert() || self == LoraTarget::FfnGateInp
    }

    /// The GGUF base-weight stem, e.g. `attn_q` in `blk.N.attn_q.weight`.
    fn gguf_stem(self) -> &'static str {
        match self {
            LoraTarget::AttnQ => "attn_q",
            LoraTarget::AttnK => "attn_k",
            LoraTarget::AttnV => "attn_v",
            LoraTarget::AttnOutput => "attn_output",
            LoraTarget::FfnGate => "ffn_gate",
            LoraTarget::FfnUp => "ffn_up",
            LoraTarget::FfnDown => "ffn_down",
            LoraTarget::ShortconvInProj => "shortconv.in_proj",
            LoraTarget::ShortconvOutProj => "shortconv.out_proj",
            LoraTarget::FfnGateInp => "ffn_gate_inp",
            LoraTarget::FfnGateExps => "ffn_gate_exps",
            LoraTarget::FfnUpExps => "ffn_up_exps",
            LoraTarget::FfnDownExps => "ffn_down_exps",
        }
    }

    /// The GGUF stem → target. `None` for stems we don't adapt in v1.
    fn from_gguf_stem(stem: &str) -> Option<LoraTarget> {
        LoraTarget::ALL.into_iter().find(|t| t.gguf_stem() == stem)
    }

    /// The PEFT sub-module name, e.g. `self_attn.q_proj`.
    fn from_peft_module(module: &str) -> Option<LoraTarget> {
        match module {
            "self_attn.q_proj" => Some(LoraTarget::AttnQ),
            "self_attn.k_proj" => Some(LoraTarget::AttnK),
            "self_attn.v_proj" => Some(LoraTarget::AttnV),
            "self_attn.o_proj" => Some(LoraTarget::AttnOutput),
            "mlp.gate_proj" => Some(LoraTarget::FfnGate),
            "mlp.up_proj" => Some(LoraTarget::FfnUp),
            "mlp.down_proj" => Some(LoraTarget::FfnDown),
            // LFM2 gated-conv (`Lfm2ShortConv` submodule `conv`) projections.
            "conv.in_proj" => Some(LoraTarget::ShortconvInProj),
            "conv.out_proj" => Some(LoraTarget::ShortconvOutProj),
            _ => None,
        }
    }
}

/// One target's low-rank factors, pre-dequantized to f32 (row-major).
#[derive(Clone)]
pub struct LoraTargetWeights {
    /// Down-projection `A`, `[rank × k]` row-major.
    pub a: Vec<f32>,
    /// Up-projection `B`, `[d × rank]` row-major.
    pub b: Vec<f32>,
    /// Low rank `r`.
    pub rank: usize,
    /// Input width (base projection's input dim).
    pub k: usize,
    /// Output width (base projection's output dim).
    pub d: usize,
    /// `alpha / rank` (folded into the apply).
    pub scale: f32,
}

impl LoraTargetWeights {
    fn new(
        a: Vec<f32>,
        rank_a: usize,
        k: usize,
        b: Vec<f32>,
        d: usize,
        rank_b: usize,
        alpha: f32,
    ) -> Result<Self> {
        ensure!(
            rank_a == rank_b,
            "LoRA rank mismatch between A ({rank_a}) and B ({rank_b})"
        );
        ensure!(rank_a > 0 && k > 0 && d > 0, "LoRA dims must be non-zero");
        // Cap the rank so backends can size fixed rank-width scratch (e.g. the
        // Metal `lora_tmp` buffer) without an out-of-bounds risk. Real adapters
        // are rank <= ~64; this bound is generous.
        ensure!(
            rank_a <= MAX_LORA_RANK,
            "LoRA rank {rank_a} exceeds the supported maximum ({MAX_LORA_RANK})"
        );
        // checked_mul so absurd dims from a malformed adapter error rather than
        // wrapping (which could make a wrong size compare equal).
        let ak = rank_a.checked_mul(k).context("LoRA A dims overflow")?;
        let dr = d.checked_mul(rank_a).context("LoRA B dims overflow")?;
        ensure!(a.len() == ak, "LoRA A size {} != rank*k {ak}", a.len());
        ensure!(b.len() == dr, "LoRA B size {} != d*rank {dr}", b.len());
        Ok(Self {
            a,
            b,
            rank: rank_a,
            k,
            d,
            scale: alpha / rank_a as f32,
        })
    }
}

/// One target's delta: a single low-rank pair, or one pair per expert.
#[derive(Clone)]
enum TargetDelta {
    Dense(LoraTargetWeights),
    /// One entry per expert, index-aligned with the base weight's expert slices.
    Experts(Vec<LoraTargetWeights>),
}

/// The (up to [`LORA_TARGET_COUNT`]) target deltas for one transformer layer.
#[derive(Default, Clone)]
pub struct LoraLayer {
    targets: [Option<TargetDelta>; LORA_TARGET_COUNT],
}

/// A loaded LoRA adapter: per-layer low-rank deltas plus scaling, with optional classifier head.
pub struct LoraAdapterWeights {
    layers: Vec<LoraLayer>,
    default_scale: f32,
    /// Classification head weight matrix if this adapter includes a token classification head.
    pub classifier_weight: Option<Vec<f32>>,
    /// Classification head bias vector if this adapter includes a token classification head.
    pub classifier_bias: Option<Vec<f32>>,
    /// Classification class label strings in index order.
    pub class_labels: Vec<String>,
    /// Number of token classification classes (0 if not a classifier).
    pub num_classes: usize,
}

impl LoraAdapterWeights {
    /// Whether this adapter includes a token classification head.
    pub fn is_classifier(&self) -> bool {
        self.classifier_weight.is_some()
    }

    /// Number of token classification classes (0 if not a classifier).
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Attach or replace class labels on an Arc-wrapped adapter.
    pub fn with_class_labels(mut self: Arc<Self>, labels: Vec<String>) -> Arc<Self> {
        let n_cls = labels.len();
        if let Some(s) = Arc::get_mut(&mut self) {
            s.num_classes = n_cls;
            s.class_labels = labels;
            self
        } else {
            Arc::new(Self {
                layers: self.layers.clone(),
                default_scale: self.default_scale,
                classifier_weight: self.classifier_weight.clone(),
                classifier_bias: self.classifier_bias.clone(),
                class_labels: labels,
                num_classes: n_cls,
            })
        }
    }

    /// Construct a classifier adapter instance for testing.
    #[cfg(test)]
    pub fn new_classifier_for_testing(
        classifier_weight: Vec<f32>,
        classifier_bias: Option<Vec<f32>>,
        class_labels: Vec<String>,
    ) -> Arc<Self> {
        let num_classes = if !class_labels.is_empty() {
            class_labels.len()
        } else if let Some(ref b) = classifier_bias {
            b.len()
        } else {
            0
        };
        Arc::new(Self {
            layers: Vec::new(),
            default_scale: 1.0,
            classifier_weight: Some(classifier_weight),
            classifier_bias,
            class_labels,
            num_classes,
        })
    }
    /// The delta for `(layer, target)`, or `None` if the adapter doesn't touch it.
    ///
    /// Always `None` for an expert target ([`LoraTarget::is_expert`]), whose
    /// delta is per-expert; use [`Self::get_expert`]. That is why this returns
    /// the weights directly rather than the internal `TargetDelta`: an expert delta
    /// cannot reach a caller that is not asking for a specific expert.
    pub fn get(&self, layer: usize, target: LoraTarget) -> Option<&LoraTargetWeights> {
        match self.layers.get(layer)?.targets[target.index()].as_ref()? {
            TargetDelta::Dense(t) => Some(t),
            TargetDelta::Experts(_) => None,
        }
    }

    /// The delta for `(layer, target, expert)` of a stacked per-expert
    /// projection, or `None` if the adapter doesn't touch it.
    ///
    /// Always `None` for a non-expert target; use [`Self::get`].
    pub fn get_expert(
        &self,
        layer: usize,
        target: LoraTarget,
        expert: usize,
    ) -> Option<&LoraTargetWeights> {
        match self.layers.get(layer)?.targets[target.index()].as_ref()? {
            TargetDelta::Experts(per_expert) => per_expert.get(expert),
            TargetDelta::Dense(_) => None,
        }
    }

    /// Whether any layer carries a delta only the routed-FFN path can apply:
    /// the router projection or a per-expert factor.
    ///
    /// Both halves go missing in different ways on a backend without those
    /// hooks, which is why one predicate covers them. [`Self::get`] returns
    /// `None` for an expert target by design, so a per-target upload loop skips
    /// per-expert factors without a word. The router is worse: structurally it
    /// is an ordinary dense target, so the same loop uploads it happily and
    /// then never reaches a hook that would apply it. Such a backend rejects
    /// the whole adapter on this rather than running a silently partial one.
    pub fn has_moe_deltas(&self) -> bool {
        self.layers.iter().any(|l| {
            LoraTarget::ALL
                .into_iter()
                .filter(|t| t.is_routed_ffn())
                .any(|t| l.targets[t.index()].is_some())
        })
    }

    /// Number of layers the adapter spans (one past the highest layer index seen).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// `alpha / rank` reported by the adapter (or derived), for diagnostics.
    pub fn default_scale(&self) -> f32 {
        self.default_scale
    }

    /// Total number of `(layer, target)` deltas present. A per-expert delta
    /// counts once, not once per expert: it adapts one projection.
    pub fn target_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.targets.iter().filter(|t| t.is_some()).count())
            .sum()
    }

    /// Verify every target's `(k, d)` matches what `config`'s projections expect,
    /// so an adapter built for a *different* model is rejected up front with a
    /// clear error rather than silently truncating (mis-`zip`ing) in the apply
    /// hot path and corrupting the output. Called at attach time.
    pub fn validate_dims(&self, config: &crate::model::ModelConfig) -> Result<()> {
        let n_layers = config
            .block_types
            .len()
            .max(config.kv_heads_per_layer.len());
        let q_dim = config.n_heads * config.head_dim;
        for (layer, l) in self.layers.iter().enumerate() {
            for target in LoraTarget::ALL {
                let Some(delta) = l.targets[target.index()].as_ref() else {
                    continue;
                };
                ensure!(
                    layer < n_layers,
                    "LoRA references layer {layer} but the model has {n_layers} layers"
                );
                // Whether *this layer's* FFN is routed. Per-layer, not
                // per-model: `lfm2moe` runs dense leading blocks and MoE ones in
                // the same file, so a whole-model flag would mislabel both ends.
                let moe = config
                    .moe
                    .as_ref()
                    .filter(|m| m.is_moe_layer.get(layer).copied().unwrap_or(false));
                // Reject a delta the forward pass would silently drop. A dense
                // `ffn_gate` adapter on a routed layer has plausible dims (the
                // model's leading blocks really are that wide), so without this
                // it validates, loads, and then adapts nothing: a partially
                // applied adapter that still generates fluent text.
                match (target, moe.is_some()) {
                    (LoraTarget::FfnGate | LoraTarget::FfnUp | LoraTarget::FfnDown, true) => bail!(
                        "LoRA target {target:?} on layer {layer} adapts a dense feed-forward \
                         block, but that layer is mixture-of-experts; it needs the stacked \
                         per-expert tensors (`ffn_*_exps.weight.lora_{{a,b}}`)"
                    ),
                    (t, false) if t.is_routed_ffn() => bail!(
                        "LoRA target {target:?} on layer {layer} adapts a mixture-of-experts \
                         feed-forward block, but that layer is dense"
                    ),
                    _ => {}
                }
                // Per-layer KV width (0 / absent ⇒ fall back to the global count).
                let kv_heads = config
                    .kv_heads_per_layer
                    .get(layer)
                    .copied()
                    .filter(|&h| h > 0)
                    .unwrap_or(config.n_kv_heads);
                let kv_dim = kv_heads * config.head_dim;
                let (want_k, want_d) = match target {
                    LoraTarget::AttnQ => (config.hidden_size, q_dim),
                    LoraTarget::AttnK | LoraTarget::AttnV => (config.hidden_size, kv_dim),
                    LoraTarget::AttnOutput => (q_dim, config.hidden_size),
                    LoraTarget::FfnGate | LoraTarget::FfnUp => {
                        (config.hidden_size, config.intermediate_size)
                    }
                    LoraTarget::FfnDown => (config.intermediate_size, config.hidden_size),
                    // LFM2 shortconv `in_proj` fans hidden → 3·hidden (the B/C/x
                    // gates); `out_proj` maps hidden → hidden.
                    LoraTarget::ShortconvInProj => (config.hidden_size, 3 * config.hidden_size),
                    LoraTarget::ShortconvOutProj => (config.hidden_size, config.hidden_size),
                    // MoE. The `moe` binding is `Some` for all four of these:
                    // the match above bailed on the dense-layer case, so the
                    // fallbacks here are unreachable rather than a silent
                    // default. Experts are `expert_ff_len` wide, *not*
                    // `intermediate_size` (1792 vs 7168 on LFM2.5-8B-A1B), so
                    // reusing the dense arm would accept an adapter four times
                    // too wide.
                    LoraTarget::FfnGateInp => (config.hidden_size, moe.map_or(0, |m| m.n_expert)),
                    LoraTarget::FfnGateExps | LoraTarget::FfnUpExps => {
                        (config.hidden_size, moe.map_or(0, |m| m.expert_ff_len))
                    }
                    LoraTarget::FfnDownExps => {
                        (moe.map_or(0, |m| m.expert_ff_len), config.hidden_size)
                    }
                };

                // One `(k, d)` rule, checked against every low-rank pair the
                // target carries: one for a dense projection, `n_expert` for a
                // stacked one. Written as a slice so the two cases cannot drift.
                // The count check belongs to the stacked arm alone; a dense
                // delta is one pair by construction, so checking it there would
                // compare 1 to 1 and read as a guard that cannot fire.
                let pairs = match delta {
                    TargetDelta::Dense(t) => std::slice::from_ref(t),
                    TargetDelta::Experts(per_expert) => {
                        let want_experts = moe.map_or(0, |m| m.n_expert);
                        ensure!(
                            per_expert.len() == want_experts,
                            "LoRA target {target:?} on layer {layer} carries {} expert deltas, \
                             but the model has {want_experts} experts",
                            per_expert.len()
                        );
                        per_expert.as_slice()
                    }
                };
                for t in pairs {
                    ensure!(
                        t.k == want_k && t.d == want_d,
                        "LoRA target {target:?} on layer {layer} has dims (in={}, out={}), \
                         but the model expects (in={want_k}, out={want_d}). Adapter built for a \
                         different model?",
                        t.k,
                        t.d
                    );
                }
            }
        }
        if let Some(ref w) = self.classifier_weight {
            ensure!(
                w.len() % config.hidden_size == 0,
                "classifier weight length {} is not a multiple of hidden_size {}",
                w.len(),
                config.hidden_size
            );
            let num_classes = w.len() / config.hidden_size;
            if let Some(ref b) = self.classifier_bias {
                ensure!(
                    b.len() == num_classes,
                    "classifier bias length {} does not match num_classes {}",
                    b.len(),
                    num_classes
                );
            }
            if !self.class_labels.is_empty() {
                ensure!(
                    self.class_labels.len() == num_classes,
                    "class_labels count {} does not match num_classes {}",
                    self.class_labels.len(),
                    num_classes
                );
            }
        }
        Ok(())
    }

    // ── GGUF ────────────────────────────────────────────────────────────────

    /// Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
    /// a file. Tensors are named `blk.{N}.{stem}.weight.lora_a` / `.lora_b`;
    /// `alpha` is read from the `adapter.lora.alpha` metadata (falling back to
    /// `rank`, i.e. `scale = 1`). Requires the `mmap` feature (for
    /// `GgufFile::open`); otherwise use [`Self::from_gguf_bytes`].
    #[cfg(feature = "mmap")]
    pub fn from_gguf(path: &Path) -> Result<Arc<Self>> {
        let gguf = GgufFile::open(path).with_context(|| format!("open adapter {path:?}"))?;
        Self::from_gguf_file(&gguf)
    }

    /// Load a GGUF adapter from in-memory bytes (no filesystem — WASM).
    pub fn from_gguf_bytes(bytes: Arc<[u8]>) -> Result<Arc<Self>> {
        let gguf = GgufFile::from_bytes(bytes).context("parse adapter GGUF bytes")?;
        Self::from_gguf_file(&gguf)
    }

    fn from_gguf_file(gguf: &GgufFile) -> Result<Arc<Self>> {
        // llama.cpp's convention (`adapter.lora.alpha`); missing ⇒ scale 1.0.
        let alpha_meta = gguf.get_f32("adapter.lora.alpha");

        let mut builder = AdapterBuilder::new();
        for (name, info) in &gguf.tensors {
            let Some((layer, target, is_a)) = parse_gguf_lora_name(name) else {
                continue;
            };
            // GGUF `ne` is fastest-varying first, so a `[k, rank]` factor is
            // `rank` rows of `k`. A stacked per-expert factor adds a third
            // dimension holding the expert count, its slices contiguous, which
            // is why `n_slices` can simply chunk the flat data below.
            //
            // Read straight from the shape rather than through
            // `GgufFile::tensor_meta`, which is deliberately rank-2-only: it
            // backs the 2D `WeightRef` kernels, where quietly flattening a
            // rank-3 tensor would hand a GEMV every expert at once.
            let (rows, cols, n_slices) = match info.shape[..] {
                [cols, rows] => (rows, cols, 1),
                [cols, rows, n_slices] => (rows, cols, n_slices),
                _ => bail!(
                    "LoRA tensor {name} has rank {}, expected 2 (dense) or 3 (per-expert)",
                    info.shape.len()
                ),
            };
            let data = gguf.get_tensor(name)?.to_f32_vec();
            let factor = Factor::new(data, rows, cols, n_slices)
                .with_context(|| format!("LoRA tensor {name}"))?;
            builder.add_factor(layer, target, is_a, factor);
        }

        let (classifier_weight, classifier_bias, num_classes) =
            if let Ok(tensor) = gguf.get_tensor("classifier.weight") {
                let w = tensor.to_f32_vec();
                let b = gguf
                    .get_tensor("classifier.bias")
                    .ok()
                    .map(|t| t.to_f32_vec());
                let n_cls = b
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or_else(|| tensor.shape().get(1).copied().unwrap_or(0));
                (Some(w), b, n_cls)
            } else {
                (None, None, 0)
            };
        let class_labels = gguf
            .get_string_array("token_classifier.labels")
            .map(|arr| arr.into_iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        builder.finish(
            alpha_meta,
            classifier_weight,
            classifier_bias,
            class_labels,
            num_classes,
        )
    }

    // ── safetensors (PEFT) ────────────────────────────────────────────────────

    /// Load a PEFT `.safetensors` adapter from a file. Tensors are named
    /// `base_model.model.model.layers.{N}.{module}.lora_A.weight` /
    /// `lora_B.weight`. PEFT stores `alpha` in a sibling `adapter_config.json`,
    /// not in the tensor file (pass it via `alpha`, `None` defaults to scale 1).
    /// Native only (WASM uses [`Self::from_safetensors_bytes`]).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_safetensors(path: &Path, alpha: Option<f32>) -> Result<Arc<Self>> {
        let bytes = std::fs::read(path).with_context(|| format!("read adapter {path:?}"))?;
        let alpha = alpha.or_else(|| {
            path.parent().and_then(|dir| {
                let p = dir.join("adapter_config.json");
                let content = std::fs::read_to_string(p).ok()?;
                let val: serde_json::Value = serde_json::from_str(&content).ok()?;
                val.get("lora_alpha")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32)
            })
        });

        let mut adapter = Self::from_safetensors_bytes(&bytes, alpha)?;

        if let Some(parent) = path
            .parent()
            .filter(|_| adapter.is_classifier() && adapter.class_labels.is_empty())
        {
            let labels = load_labels_from_dir(parent);
            if !labels.is_empty() {
                adapter = adapter.with_class_labels(labels);
            }
        }
        Ok(adapter)
    }

    /// Load an adapter from a file path (auto-detecting GGUF vs SafeTensors).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_path(path: &Path) -> Result<Arc<Self>> {
        let is_gguf = if let Ok(mut f) = std::fs::File::open(path) {
            use std::io::Read;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic).is_ok() && &magic == b"GGUF"
        } else {
            false
        };
        if is_gguf {
            Self::from_gguf(path)
        } else {
            Self::from_safetensors(path, None)
        }
    }

    /// Load a PEFT safetensors adapter from in-memory bytes.
    pub fn from_safetensors_bytes(bytes: &[u8], alpha: Option<f32>) -> Result<Arc<Self>> {
        let st = SafeTensors::parse(bytes)?;
        let mut builder = AdapterBuilder::new();
        let mut classifier_weight = None;
        let mut classifier_bias = None;
        let mut num_classes = 0;

        for (name, entry) in st.tensors() {
            if name.ends_with("classifier.weight") {
                if let Ok((rows, _cols)) = entry.shape2() {
                    num_classes = rows;
                }
                classifier_weight = Some(st.dequantize(entry, bytes)?);
                continue;
            }
            if name.ends_with("classifier.bias") {
                let b = st.dequantize(entry, bytes)?;
                num_classes = b.len();
                classifier_bias = Some(b);
                continue;
            }

            // Mixture-of-experts deltas are only read from GGUF, where the
            // per-expert factors arrive stacked into one tensor with a known
            // layout. PEFT stores them as one module per expert under a naming
            // scheme that varies by architecture, so rather than guess at it and
            // drop what doesn't match, say so. Detected on the path containing
            // `experts` (which is what makes this an assertion about the file
            // rather than a guess about the scheme) and only for tensors that
            // are LoRA factors at all.
            if name.contains("experts") && (name.contains("lora_A") || name.contains("lora_B")) {
                bail!(
                    "PEFT adapter tensor {name} targets a mixture-of-experts projection, which \
                     cera only loads from GGUF. Convert the adapter with llama.cpp's \
                     `convert_lora_to_gguf.py` and load the result."
                );
            }
            let Some((layer, target, is_a)) = parse_peft_lora_name(name) else {
                continue;
            };
            // PEFT weights are row-major `[out, in]`: `lora_A` is `[rank, k]`,
            // `lora_B` is `[d, rank]` (same (rows, cols) convention as GGUF).
            let (rows, cols) = entry
                .shape2()
                .with_context(|| format!("tensor {name} not 2-D"))?;
            let data = st.dequantize(entry, bytes)?;
            // `shape2` already rejected anything but rank 2, so every PEFT
            // factor is a single unstacked matrix.
            let factor =
                Factor::new(data, rows, cols, 1).with_context(|| format!("LoRA tensor {name}"))?;
            builder.add_factor(layer, target, is_a, factor);
        }
        // PEFT keeps alpha out-of-band; default to alpha == rank (scale 1).
        builder.finish(
            alpha,
            classifier_weight,
            classifier_bias,
            Vec::new(),
            num_classes,
        )
    }
}

/// Helper to load class labels from sibling JSON files in a directory.
#[cfg(not(target_arch = "wasm32"))]
fn load_labels_from_dir(dir: &Path) -> Vec<String> {
    // 1. Try label_schema.json (BIOES entity types schema)
    let schema_path = dir.join("label_schema.json");
    if let Ok(content) = std::fs::read_to_string(&schema_path) {
        let val_opt = serde_json::from_str::<serde_json::Value>(&content).ok();
        if let Some(types) = val_opt
            .as_ref()
            .and_then(|v| v.get("types_in_order")?.as_array())
        {
            let mut labels = vec!["O".to_string()];
            for item in types {
                if let Some(t_str) = item.as_str() {
                    labels.push(format!("B-{t_str}"));
                    labels.push(format!("I-{t_str}"));
                    labels.push(format!("E-{t_str}"));
                    labels.push(format!("S-{t_str}"));
                }
            }
            return labels;
        }
    }

    // 2. Try config.json or adapter_config.json for "id2label"
    for filename in &["config.json", "adapter_config.json"] {
        let p = dir.join(filename);
        if let Ok(content) = std::fs::read_to_string(&p) {
            let val_opt = serde_json::from_str::<serde_json::Value>(&content).ok();
            if let Some(id2label) = val_opt
                .as_ref()
                .and_then(|v| v.get("id2label")?.as_object())
            {
                let mut pairs = Vec::new();
                for (k, v) in id2label {
                    if let (Ok(idx), Some(lbl)) = (k.parse::<usize>(), v.as_str()) {
                        pairs.push((idx, lbl.to_string()));
                    }
                }
                if !pairs.is_empty() {
                    pairs.sort_by_key(|&(idx, _)| idx);
                    return pairs.into_iter().map(|(_, l)| l).collect();
                }
            }
        }
    }

    Vec::new()
}

/// Sanity cap on an adapter's layer index — real models have well under this
/// many layers; a larger index means a malformed/hostile tensor name, which we
/// reject rather than let it size a huge allocation.
const MAX_LORA_LAYERS: usize = 8192;

/// Sanity cap on the expert count of a stacked adapter factor.
///
/// Not a defence against a crafted header: `GgufFile::tensor_range` already
/// requires a tensor's full byte range to be present, so a `[1, 1, 10^9]` factor
/// needs ~4 GB of real file behind it or the read fails first. What this bounds
/// is the *amplification* on a file that does have the bytes: the split turns
/// one flat tensor into `n_slices` `LoraTargetWeights`, each two separately
/// allocated `Vec`s. It runs before anything knows the model's real expert
/// count, since `validate_dims` is the only thing that does and it runs at
/// attach time. Real MoE models are far below this; llama.cpp's own
/// `LLAMA_MAX_EXPERTS` is 512.
const MAX_LORA_EXPERTS: usize = 4096;

/// Maximum supported LoRA rank. Adapters above this are rejected at load
/// (`LoraTargetWeights::new`), which lets GPU backends size a fixed rank-width
/// scratch buffer (the Metal `lora_tmp`) with no out-of-bounds risk. Real
/// adapters are rank ≤ ~64; this bound is deliberately generous.
pub const MAX_LORA_RANK: usize = 512;

/// Accumulates loose A/B factors keyed by (layer, target), then validates + pairs
/// them into a `LoraAdapterWeights`.
#[derive(Default)]
struct AdapterBuilder {
    /// (layer, target) → (A?, B?).
    factors: std::collections::HashMap<(usize, LoraTarget), FactorPair>,
    max_layer: usize,
}

/// One loaded LoRA factor: `n_slices` contiguous `[rows × cols]` row-major
/// matrices. `n_slices` is 1 for an ordinary projection and the expert count for
/// a stacked mixture-of-experts one, so both shapes flow through one type.
struct Factor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    n_slices: usize,
}

impl Factor {
    /// Validate that `data` is exactly `n_slices` whole `[rows × cols]` matrices,
    /// which is what makes [`Self::slice`] in-bounds by construction.
    fn new(data: Vec<f32>, rows: usize, cols: usize, n_slices: usize) -> Result<Self> {
        // checked_mul so a malformed header's absurd dims error instead of
        // wrapping to a small product that happens to match `data.len()`.
        let want = rows
            .checked_mul(cols)
            .and_then(|m| m.checked_mul(n_slices))
            .context("LoRA factor dims overflow")?;
        // Checked separately from the size: a zero-slice factor satisfies
        // `data.len() == want` at 0 == 0, so folding the two together would
        // report a failure by printing an equality that holds.
        ensure!(n_slices > 0, "LoRA factor has no slices");
        ensure!(
            n_slices <= MAX_LORA_EXPERTS,
            "LoRA factor is stacked {n_slices} deep, over the sane maximum of {MAX_LORA_EXPERTS} experts"
        );
        ensure!(
            data.len() == want,
            "LoRA factor has {} elements, expected {n_slices}×{rows}×{cols} = {want}",
            data.len()
        );
        Ok(Self {
            data,
            rows,
            cols,
            n_slices,
        })
    }

    /// The `i`-th `[rows × cols]` matrix, copied out. Called once per expert at
    /// load time, so the copy is not on any hot path.
    fn slice(&self, i: usize) -> Vec<f32> {
        let n = self.rows * self.cols;
        self.data[i * n..(i + 1) * n].to_vec()
    }
}

#[derive(Default)]
struct FactorPair {
    a: Option<Factor>,
    b: Option<Factor>,
}

impl AdapterBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn add_factor(&mut self, layer: usize, target: LoraTarget, is_a: bool, factor: Factor) {
        self.max_layer = self.max_layer.max(layer);
        let slot = self.factors.entry((layer, target)).or_default();
        if is_a {
            slot.a = Some(factor);
        } else {
            slot.b = Some(factor);
        }
    }

    fn finish(
        self,
        alpha: Option<f32>,
        classifier_weight: Option<Vec<f32>>,
        classifier_bias: Option<Vec<f32>>,
        class_labels: Vec<String>,
        num_classes: usize,
    ) -> Result<Arc<LoraAdapterWeights>> {
        ensure!(
            !self.factors.is_empty() || classifier_weight.is_some(),
            "adapter contains no LoRA or classifier tensors"
        );
        let n_cls = if !class_labels.is_empty() {
            class_labels.len()
        } else if let Some(ref b) = classifier_bias {
            b.len()
        } else {
            num_classes
        };
        if self.factors.is_empty() {
            return Ok(Arc::new(LoraAdapterWeights {
                layers: Vec::new(),
                default_scale: 1.0,
                classifier_weight,
                classifier_bias,
                class_labels,
                num_classes: n_cls,
            }));
        }
        // Reject an absurd layer index from a malformed/hostile name before it
        // sizes the `layers` Vec — otherwise `blk.9999999999...` would try to
        // allocate ~10^10 entries and OOM-abort instead of erroring.
        ensure!(
            self.max_layer < MAX_LORA_LAYERS,
            "adapter layer index {} exceeds the sane maximum ({MAX_LORA_LAYERS})",
            self.max_layer
        );
        let n_layers = self.max_layer + 1;
        let mut layers: Vec<LoraLayer> = (0..n_layers).map(|_| LoraLayer::default()).collect();

        // Iterate in a deterministic (layer, target) order so `default_scale`
        // (taken from the first pair) is stable across runs — HashMap order isn't.
        let mut factors: Vec<_> = self.factors.into_iter().collect();
        factors.sort_by_key(|&(key, _)| key);

        // A single global scale: alpha/rank of the first (lowest-index) pair.
        let mut default_scale = 1.0f32;
        let mut scale_set = false;

        for ((layer, target), pair) in factors {
            let a = pair
                .a
                .with_context(|| format!("layer {layer} target {target:?}: missing lora_a"))?;
            let b = pair
                .b
                .with_context(|| format!("layer {layer} target {target:?}: missing lora_b"))?;
            // A and B are separate tensors, so nothing in the file forces them to
            // agree on the expert count; a mismatch would otherwise pair expert
            // `i`'s A with a B that isn't its own.
            ensure!(
                a.n_slices == b.n_slices,
                "layer {layer} target {target:?}: lora_a has {} slices but lora_b has {}",
                a.n_slices,
                b.n_slices
            );
            // Rank is `alpha`'s denominator. Read from A's rows here;
            // llama.cpp reads the same number off B (`lw->b->ne[0]`), and the
            // two agree because `LoraTargetWeights::new` rejects the pair below
            // unless A's rank equals B's.
            let alpha = alpha.unwrap_or(a.rows as f32);
            // Whether the delta is per-expert follows from the *target*, not from
            // the slice count: a one-expert model would otherwise load its
            // stacked tensors as dense ones and never reach `get_expert`.
            let delta = if target.is_expert() {
                let per_expert = (0..a.n_slices)
                    .map(|e| {
                        LoraTargetWeights::new(
                            a.slice(e),
                            a.rows,
                            a.cols,
                            b.slice(e),
                            b.rows,
                            b.cols,
                            alpha,
                        )
                        .with_context(|| format!("layer {layer} target {target:?} expert {e}"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                TargetDelta::Experts(per_expert)
            } else {
                ensure!(
                    a.n_slices == 1,
                    "layer {layer} target {target:?} is a single projection carrying one \
                     low-rank pair, but its factors are stacked {} deep",
                    a.n_slices
                );
                TargetDelta::Dense(
                    LoraTargetWeights::new(a.data, a.rows, a.cols, b.data, b.rows, b.cols, alpha)
                        .with_context(|| format!("layer {layer} target {target:?}"))?,
                )
            };
            if !scale_set {
                // `Experts` is never empty: `Factor::new` rejects `n_slices == 0`
                // and the vec is built with exactly `a.n_slices` entries, so the
                // `map_or` default is a total-function formality, not a fallback.
                default_scale = match &delta {
                    TargetDelta::Dense(t) => t.scale,
                    TargetDelta::Experts(per_expert) => per_expert.first().map_or(1.0, |t| t.scale),
                };
                scale_set = true;
            }
            layers[layer].targets[target.index()] = Some(delta);
        }

        Ok(Arc::new(LoraAdapterWeights {
            layers,
            default_scale,
            classifier_weight,
            classifier_bias,
            class_labels,
            num_classes: n_cls,
        }))
    }
}

/// Parse a GGUF LoRA tensor name → `(layer, target, is_a)`.
/// e.g. `blk.12.attn_q.weight.lora_a` → `(12, AttnQ, true)`.
fn parse_gguf_lora_name(name: &str) -> Option<(usize, LoraTarget, bool)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;
    let (stem, suffix) = rest.split_once(".weight.")?;
    let is_a = match suffix {
        "lora_a" => true,
        "lora_b" => false,
        _ => return None,
    };
    let target = LoraTarget::from_gguf_stem(stem)?;
    Some((layer, target, is_a))
}

/// Parse a PEFT safetensors LoRA tensor name → `(layer, target, is_a)`.
/// e.g. `base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight`
/// → `(7, AttnQ, true)`.
fn parse_peft_lora_name(name: &str) -> Option<(usize, LoraTarget, bool)> {
    // Find the `layers.{N}.` segment (prefix depth varies by export tooling).
    let idx = name.find("layers.")?;
    let after = &name[idx + "layers.".len()..];
    let (layer_str, rest) = after.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;
    // rest = `{module}.lora_{A,B}.weight`
    let rest = rest.strip_suffix(".weight")?;
    let (module, ab) = rest.rsplit_once('.')?;
    let is_a = match ab {
        "lora_A" => true,
        "lora_B" => false,
        _ => return None,
    };
    let target = LoraTarget::from_peft_module(module)?;
    Some((layer, target, is_a))
}

// ── apply (pure math) ────────────────────────────────────────────────────────

/// Decode-path apply: `y += scale · B·(A·x)`, in place. `x` is length `k`, `y`
/// length `d`; `tmp` is scratch resized to `rank`. Alloc-free given a reused `tmp`.
///
/// This is the reference order that [`apply_prefill`] must reproduce exactly;
/// see that function for why "exactly" and not "closely".
pub fn apply_decode(t: &LoraTargetWeights, x: &[f32], y: &mut [f32], tmp: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), t.k);
    debug_assert_eq!(y.len(), t.d);
    // A zero-scale adapter is a guaranteed no-op — skip the loops (and the
    // `+= 0.0`, which could otherwise flip a `-0.0` to `+0.0`).
    if t.scale == 0.0 {
        return;
    }
    tmp.clear();
    tmp.resize(t.rank, 0.0);
    // tmp = scale · (A · x)   (A is [rank × k] row-major; fold scale into the
    // small r-vector, the cheapest place).
    for (row, tmp_r) in t.a.chunks_exact(t.k).zip(tmp.iter_mut()) {
        let acc: f32 = row.iter().zip(x).map(|(w, &xi)| w * xi).sum();
        *tmp_r = acc * t.scale;
    }
    // y += B · tmp   (B is [d × rank] row-major). The dot product is summed on
    // its own, from zero, and added to `y` once.
    for (row, yi) in t.b.chunks_exact(t.rank).zip(y.iter_mut()) {
        let acc: f32 = row.iter().zip(tmp.iter()).map(|(w, &ti)| w * ti).sum();
        *yi += acc;
    }
}

/// Prefill-path apply: `Y += scale · B·(A·X)` for `n` tokens at once. `x` is the
/// projection input `[k × n]` **channel-major** (`x[i*n + j]` = input channel `i`
/// of token `j`), `y` the projection output `[d × n]` in the same layout,
/// accumulated in place — matching the batched-prefill buffer layout so this
/// drops in right after a base projection GEMM. `tmp` is scratch resized to
/// `rank · n + n`.
///
/// **Must be bit-identical to [`apply_decode`] on each column**, which
/// `apply_prefill_matches_per_column_decode` asserts on the raw bits rather than
/// to a tolerance. Closeness is not enough: the projections around an adapter
/// are quantized, so a one-ulp difference in the delta can flip an int8 bucket
/// in the next layer, and from there the paths diverge by a quantization step
/// rather than an ulp. On LFM2-350M-Q4_0 that turned a 1e-8 seed into logits
/// 0.49 apart, ten layers on.
///
/// The trap is in the second stage, and it is invisible if you only compare the
/// formulas. Accumulating `y[o][j] += B[o][r]·tmp[r][j]` inside the `r` loop,
/// which is the natural way to write it against this layout, computes
/// `((y + b₀t₀) + b₁t₁) + …`, while the decode path sums the dot product from
/// zero and adds it to `y` once. Those differ in the rounding of every partial
/// sum. So the `r` loop accumulates into a zeroed scratch row, and only the
/// finished dot product reaches `y`.
///
/// The first stage needs no such care: it already accumulates from zero over the
/// input dimension in ascending order, exactly as the decode path's dot product
/// does.
pub fn apply_prefill(
    t: &LoraTargetWeights,
    x: &[f32],
    y: &mut [f32],
    n: usize,
    tmp: &mut Vec<f32>,
) {
    debug_assert_eq!(x.len(), t.k * n);
    debug_assert_eq!(y.len(), t.d * n);
    // Nothing to apply for zero columns; also guards `chunks_exact_mut(0)`, which
    // panics (this is a `pub` helper, so a caller could pass n == 0).
    if n == 0 || t.scale == 0.0 {
        return;
    }
    // `rank × n` for the low-rank vectors, plus one `n`-wide row to accumulate
    // each output's dot product before it touches `y`.
    tmp.clear();
    tmp.resize(t.rank * n + n, 0.0);
    let (tmp_rank, acc_row) = tmp.split_at_mut(t.rank * n);

    // Tmp[rank × n] = scale · (A · X).  A is [rank × k] row-major.
    for (r, tmp_row) in tmp_rank.chunks_exact_mut(n).enumerate() {
        let a_row = &t.a[r * t.k..(r + 1) * t.k];
        for (kk, &a_val) in a_row.iter().enumerate() {
            let x_row = &x[kk * n..(kk + 1) * n];
            for (t_j, &x_j) in tmp_row.iter_mut().zip(x_row) {
                *t_j += a_val * x_j;
            }
        }
        for t_j in tmp_row.iter_mut() {
            *t_j *= t.scale;
        }
    }
    // Y[d × n] += B · Tmp.  B is [d × rank] row-major.
    for (o, y_row) in y.chunks_exact_mut(n).enumerate() {
        let b_row = &t.b[o * t.rank..(o + 1) * t.rank];
        acc_row.fill(0.0);
        for (r, &b_val) in b_row.iter().enumerate() {
            let tmp_row = &tmp_rank[r * n..(r + 1) * n];
            for (a_j, &t_j) in acc_row.iter_mut().zip(tmp_row) {
                *a_j += b_val * t_j;
            }
        }
        for (y_j, &a_j) in y_row.iter_mut().zip(acc_row.iter()) {
            *y_j += a_j;
        }
    }
}

/// Apply the Q/K/V attention-projection LoRAs for one layer: `q/k/v` are the
/// base projection outputs (share input `x`), each gets `+= scale·B·(A·x)` if the
/// adapter targets it. Shared by both `forward_attn_block` implementations
/// (dense transformer + LFM2) so the two can't drift out of sync.
pub fn apply_attn_qkv(
    lora: &LoraAdapterWeights,
    layer: usize,
    x: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    tmp: &mut Vec<f32>,
) {
    if let Some(t) = lora.get(layer, LoraTarget::AttnQ) {
        apply_decode(t, x, q, tmp);
    }
    if let Some(t) = lora.get(layer, LoraTarget::AttnK) {
        apply_decode(t, x, k, tmp);
    }
    if let Some(t) = lora.get(layer, LoraTarget::AttnV) {
        apply_decode(t, x, v, tmp);
    }
}

// ── minimal safetensors reader ────────────────────────────────────────────────

/// A parsed safetensors header entry.
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

impl StEntry {
    fn shape2(&self) -> Result<(usize, usize)> {
        ensure!(self.shape.len() == 2, "expected 2-D, got {:?}", self.shape);
        Ok((self.shape[0], self.shape[1]))
    }
}

/// A minimal safetensors reader: `u64-LE header length + JSON header + tensor
/// bytes`. Only the tiny LoRA factors are decoded, so this stays simple.
struct SafeTensors {
    entries: Vec<(String, StEntry)>,
    data_start: usize,
}

impl SafeTensors {
    fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 8, "safetensors: truncated header length");
        // `try_from` (not `as usize`) so an oversized length is REJECTED rather
        // than truncated to a small in-range value on 32-bit targets (wasm).
        let header_len = usize::try_from(u64::from_le_bytes(bytes[0..8].try_into().unwrap()))
            .context("safetensors: header length too large for this platform")?;
        let header_end = 8usize
            .checked_add(header_len)
            .context("safetensors: header length overflow")?;
        ensure!(
            header_end <= bytes.len(),
            "safetensors: header exceeds file"
        );
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])
            .context("safetensors: bad JSON header")?;
        let obj = header
            .as_object()
            .context("safetensors: header is not an object")?;

        let mut entries = Vec::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .with_context(|| format!("{name}: missing dtype"))?
                .to_string();
            let shape = v
                .get("shape")
                .and_then(|s| s.as_array())
                .with_context(|| format!("{name}: missing shape"))?
                .iter()
                .map(|n| n.as_u64().and_then(|u| usize::try_from(u).ok()))
                .collect::<Option<Vec<_>>>()
                .with_context(|| format!("{name}: bad shape (or a dim too large)"))?;
            let offsets = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .with_context(|| format!("{name}: missing data_offsets"))?;
            ensure!(
                offsets.len() == 2,
                "{name}: data_offsets must be [begin, end]"
            );
            let to_usize = |v: &serde_json::Value| -> Result<usize> {
                usize::try_from(v.as_u64().context("bad data_offset")?)
                    .context("data_offset too large for this platform")
            };
            let begin = to_usize(&offsets[0])?;
            let end = to_usize(&offsets[1])?;
            entries.push((
                name.clone(),
                StEntry {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            ));
        }
        Ok(Self {
            entries,
            data_start: header_end,
        })
    }

    fn tensors(&self) -> impl Iterator<Item = (&str, &StEntry)> {
        self.entries.iter().map(|(n, e)| (n.as_str(), e))
    }

    /// Decode one entry's bytes → f32 (F32 / F16 / BF16).
    fn dequantize(&self, e: &StEntry, bytes: &[u8]) -> Result<Vec<f32>> {
        let start = self
            .data_start
            .checked_add(e.begin)
            .context("safetensors: offset overflow")?;
        let end = self
            .data_start
            .checked_add(e.end)
            .context("safetensors: offset overflow")?;
        ensure!(
            end <= bytes.len() && start <= end,
            "safetensors: tensor slice out of range"
        );
        let raw = &bytes[start..end];
        // Checked product so a crafted shape (e.g. [6e9, 6e9]) yields a typed
        // error, not an overflow panic (debug) / silent wrap (release).
        let n = e
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .context("safetensors: shape product overflows usize")?;
        let expect_bytes = |elt: usize| n.checked_mul(elt).context("safetensors: size overflow");
        match e.dtype.as_str() {
            "F32" => {
                ensure!(raw.len() == expect_bytes(4)?, "F32 byte count mismatch");
                Ok(raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            }
            "F16" => {
                ensure!(raw.len() == expect_bytes(2)?, "F16 byte count mismatch");
                Ok(raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| crate::quant::f16_to_f32(u16::from_le_bytes(*c)))
                    .collect())
            }
            "BF16" => {
                ensure!(raw.len() == expect_bytes(2)?, "BF16 byte count mismatch");
                Ok(raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| crate::quant::bf16_to_f32(u16::from_le_bytes(*c)))
                    .collect())
            }
            other => bail!("unsupported safetensors dtype for LoRA: {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_name_parse() {
        assert_eq!(
            parse_gguf_lora_name("blk.12.attn_q.weight.lora_a"),
            Some((12, LoraTarget::AttnQ, true))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.0.ffn_down.weight.lora_b"),
            Some((0, LoraTarget::FfnDown, false))
        );
        // LFM2 gated-conv (shortconv) projections — the dotted stem must round-trip.
        assert_eq!(
            parse_gguf_lora_name("blk.4.shortconv.in_proj.weight.lora_a"),
            Some((4, LoraTarget::ShortconvInProj, true))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.15.shortconv.out_proj.weight.lora_b"),
            Some((15, LoraTarget::ShortconvOutProj, false))
        );
        // MoE. `ffn_gate`, `ffn_gate_inp` and `ffn_gate_exps` are three
        // different projections sharing a prefix, so a `starts_with` match
        // would collapse them; each must land on its own target.
        assert_eq!(
            parse_gguf_lora_name("blk.2.ffn_gate_exps.weight.lora_a"),
            Some((2, LoraTarget::FfnGateExps, true))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.2.ffn_up_exps.weight.lora_b"),
            Some((2, LoraTarget::FfnUpExps, false))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.23.ffn_down_exps.weight.lora_a"),
            Some((23, LoraTarget::FfnDownExps, true))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.2.ffn_gate_inp.weight.lora_a"),
            Some((2, LoraTarget::FfnGateInp, true))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.2.ffn_gate.weight.lora_a"),
            Some((2, LoraTarget::FfnGate, true))
        );
        // Non-lora / unknown target / malformed → None.
        assert_eq!(parse_gguf_lora_name("blk.3.attn_q.weight"), None);
        assert_eq!(parse_gguf_lora_name("blk.3.attn_norm.weight.lora_a"), None);
        assert_eq!(parse_gguf_lora_name("token_embd.weight"), None);
    }

    #[test]
    fn peft_name_parse() {
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight"),
            Some((7, LoraTarget::AttnQ, true))
        );
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.31.mlp.up_proj.lora_B.weight"),
            Some((31, LoraTarget::FfnUp, false))
        );
        assert_eq!(
            parse_peft_lora_name("model.layers.2.self_attn.o_proj.lora_A.weight"),
            Some((2, LoraTarget::AttnOutput, true))
        );
        // LFM2 gated-conv projections — dotted module name must round-trip.
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.0.conv.in_proj.lora_A.weight"),
            Some((0, LoraTarget::ShortconvInProj, true))
        );
        assert_eq!(
            parse_peft_lora_name("model.layers.3.conv.out_proj.lora_B.weight"),
            Some((3, LoraTarget::ShortconvOutProj, false))
        );
        // Not a lora tensor / unknown module.
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.0.input_layernorm.weight"),
            None
        );
    }

    /// Build a minimal PEFT safetensors buffer with one q_proj adapter on layer 0.
    fn synth_safetensors(rank: usize, k: usize, d: usize, a_val: f32, b_val: f32) -> Vec<u8> {
        let a: Vec<f32> = vec![a_val; rank * k];
        let b: Vec<f32> = vec![b_val; d * rank];
        let a_bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
        let a_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let b_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
        let header = serde_json::json!({
            a_name: { "dtype": "F32", "shape": [rank, k], "data_offsets": [0, a_bytes.len()] },
            b_name: { "dtype": "F32", "shape": [d, rank], "data_offsets": [a_bytes.len(), a_bytes.len() + b_bytes.len()] },
        });
        let header_str = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_str);
        out.extend_from_slice(&a_bytes);
        out.extend_from_slice(&b_bytes);
        out
    }

    #[test]
    fn safetensors_load_shapes_and_scale() {
        let (rank, k, d) = (8, 64, 128);
        let buf = synth_safetensors(rank, k, d, 0.5, 0.25);
        // alpha = 16 → scale = alpha/rank = 2.0
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(16.0)).unwrap();
        assert_eq!(adapter.n_layers(), 1);
        assert_eq!(adapter.target_count(), 1);
        let t = adapter.get(0, LoraTarget::AttnQ).expect("q_proj present");
        assert_eq!((t.rank, t.k, t.d), (rank, k, d));
        assert_eq!(t.a.len(), rank * k);
        assert_eq!(t.b.len(), d * rank);
        assert!((t.scale - 2.0).abs() < 1e-6, "scale {}", t.scale);
        // No adapter on an untouched target.
        assert!(adapter.get(0, LoraTarget::FfnDown).is_none());
        // Default alpha (None) ⇒ scale 1.0.
        let a2 = LoraAdapterWeights::from_safetensors_bytes(&buf, None).unwrap();
        assert!((a2.get(0, LoraTarget::AttnQ).unwrap().scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_math_and_noop() {
        let (rank, k, d) = (2, 3, 4);
        // A = all 1.0, B = all 0.0 → delta is zero (no-op) regardless of scale.
        let buf = synth_safetensors(rank, k, d, 1.0, 0.0);
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(4.0)).unwrap();
        let t = adapter.get(0, LoraTarget::AttnQ).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];
        let before = y.clone();
        let mut tmp = Vec::new();
        apply_decode(t, &x, &mut y, &mut tmp);
        assert_eq!(y, before, "B=0 must be a no-op");

        // Now B = 1.0: delta_o = scale * sum_r( sum_j A[r][j] x[j] ) with A=1 →
        // A·x = sum(x) per rank; tmp[r] = scale*sum(x); B·tmp = rank*scale*sum(x).
        let buf = synth_safetensors(rank, k, d, 1.0, 1.0);
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(4.0)).unwrap();
        let t = adapter.get(0, LoraTarget::AttnQ).unwrap();
        let mut y = vec![0.0; d];
        apply_decode(t, &x, &mut y, &mut tmp);
        let sum_x: f32 = x.iter().sum();
        let expected = rank as f32 * (t.scale * sum_x); // scale = 4/2 = 2
        for &yi in &y {
            assert!((yi - expected).abs() < 1e-5, "{yi} != {expected}");
        }
    }

    /// The batched `apply_prefill` must equal `apply_decode` on each column
    /// **bit for bit**, not to a tolerance.
    ///
    /// A tolerance is not good enough here and this test used to allow one. The
    /// projections around the adapter are quantized, so a one-ulp difference in
    /// the delta can flip an int8 bucket in the next layer, and from there the
    /// two paths diverge by a quantization step rather than an ulp. On
    /// LFM2-350M-Q4_0 a shortconv adapter seeded exactly that way ended up
    /// 0.49 apart in the logits.
    #[test]
    fn apply_prefill_matches_per_column_decode() {
        let (rank, k, d, n) = (3, 4, 5, 3);
        let buf = synth_safetensors(rank, k, d, 0.5, 0.25);
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(6.0)).unwrap();
        let t = adapter.get(0, LoraTarget::AttnQ).unwrap();

        // Channel-major X [k×n], distinct per column; nonzero base Y to exercise
        // the in-place accumulate.
        let mut x = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                x[i * n + j] = (i as f32 + 1.0) * (j as f32 + 1.0) * 0.1;
            }
        }
        let mut y_batched: Vec<f32> = (0..d * n).map(|i| i as f32 * 0.01).collect();
        let mut y_ref = y_batched.clone();

        let mut tmp = Vec::new();
        apply_prefill(t, &x, &mut y_batched, n, &mut tmp);

        for j in 0..n {
            let x_col: Vec<f32> = (0..k).map(|i| x[i * n + j]).collect();
            let mut y_col: Vec<f32> = (0..d).map(|o| y_ref[o * n + j]).collect();
            apply_decode(t, &x_col, &mut y_col, &mut tmp);
            for (o, &v) in y_col.iter().enumerate() {
                y_ref[o * n + j] = v;
            }
        }
        for (i, (a, b)) in y_batched.iter().zip(&y_ref).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: batched {a} != per-column {b}"
            );
        }
    }

    #[test]
    fn empty_adapter_errors() {
        // A safetensors buffer with no LoRA tensors → typed error, not a panic.
        let header = serde_json::json!({
            "some.other.weight": { "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16] },
        });
        let hs = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hs);
        buf.extend_from_slice(&[0u8; 16]);
        assert!(LoraAdapterWeights::from_safetensors_bytes(&buf, None).is_err());
    }

    /// Wrap a JSON header into a safetensors buffer with `data_len` trailing bytes.
    fn st_buf(header: serde_json::Value, data_len: usize) -> Vec<u8> {
        let hs = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hs);
        buf.extend_from_slice(&vec![0u8; data_len]);
        buf
    }

    #[test]
    fn rejects_absurd_layer_index() {
        // A hostile PEFT name with a giant layer index must ERROR, not try to
        // allocate ~10^10 layers and OOM-abort.
        let a = "base_model.model.model.layers.9999999999.self_attn.q_proj.lora_A.weight";
        let b = "base_model.model.model.layers.9999999999.self_attn.q_proj.lora_B.weight";
        let buf = st_buf(
            serde_json::json!({
                a: { "dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4] },
                b: { "dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8] },
            }),
            8,
        );
        assert!(LoraAdapterWeights::from_safetensors_bytes(&buf, None).is_err());
    }

    // ── mixture-of-experts adapters ─────────────────────────────────────────

    /// `ALL` and `index()` are two hand-written spellings of the same order, and
    /// `index()` is what sizes the per-target arrays in `model/gpu_lfm2.rs` and
    /// `model/metal_lfm2.rs`. A target added to one and not the other indexes
    /// out of bounds at runtime, so pin them to each other.
    ///
    /// `ALL.len() == LORA_TARGET_COUNT` is deliberately *not* asserted: `ALL` is
    /// declared as `[LoraTarget; LORA_TARGET_COUNT]`, so that equality is the
    /// type and an assertion on it can never fail.
    #[test]
    fn all_targets_are_in_index_order() {
        for (i, target) in LoraTarget::ALL.into_iter().enumerate() {
            assert_eq!(target.index(), i, "{target:?}");
        }
        // Stems must be distinct, or `from_gguf_stem` silently resolves a
        // tensor to whichever target `ALL` happens to list first.
        let mut stems: Vec<&str> = LoraTarget::ALL.iter().map(|t| t.gguf_stem()).collect();
        stems.sort_unstable();
        stems.dedup();
        assert_eq!(
            stems.len(),
            LORA_TARGET_COUNT,
            "duplicate gguf stems: {stems:?}"
        );
    }

    /// The error text of a load that must fail, *including the cause chain*.
    ///
    /// `LoraAdapterWeights` has no `Debug`, so `unwrap_err` is unavailable on
    /// the loader's `Result`. Formatted with `{:?}` rather than `to_string()`
    /// because the loader wraps failures in `with_context` (the tensor name), so
    /// `to_string()` returns only that outermost line and an assertion on the
    /// real reason would never match.
    fn load_err(r: Result<Arc<LoraAdapterWeights>>) -> String {
        match r {
            Ok(_) => panic!("expected the adapter to be rejected"),
            Err(e) => format!("{e:?}"),
        }
    }

    /// Append a GGUF-format length-prefixed string.
    fn push_gguf_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// Serialize a minimal GGUF v3 file of F32 tensors, `(name, ne, data)` with
    /// `ne` fastest-varying first (GGUF's own order), plus an optional
    /// `adapter.lora.alpha`.
    fn synth_gguf(tensors: &[(&str, Vec<usize>, Vec<f32>)], alpha: Option<f32>) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(alpha.is_some() as u64).to_le_bytes());
        if let Some(a) = alpha {
            push_gguf_string(&mut out, "adapter.lora.alpha");
            out.extend_from_slice(&6u32.to_le_bytes()); // GGUF_TYPE_FLOAT32
            out.extend_from_slice(&a.to_le_bytes());
        }
        let mut offset = 0u64;
        for (name, ne, data) in tensors {
            push_gguf_string(&mut out, name);
            out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
            ne.iter()
                .for_each(|&d| out.extend_from_slice(&(d as u64).to_le_bytes()));
            out.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
            out.extend_from_slice(&offset.to_le_bytes());
            offset += (data.len() * 4) as u64;
        }
        // The data section starts at the next multiple of the default alignment.
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        for (_, _, data) in tensors {
            out.extend(data.iter().flat_map(|x| x.to_le_bytes()));
        }
        out
    }

    /// A stacked expert adapter for one layer: `n_expert` slices where expert
    /// `e`'s `A` is filled with `e + 1` and its `B` with `1 / (e + 1)`, so a
    /// wrong slice is a wrong *value*, not just a wrong shape.
    ///
    /// Both factors vary with `e` deliberately. Holding `B` constant would leave
    /// a bad offset or stride in `b.slice(e)` invisible, since every slice would
    /// then hold the same bytes.
    fn synth_expert_gguf(n_expert: usize, rank: usize, k: usize, d: usize) -> Vec<u8> {
        let a: Vec<f32> = (0..n_expert)
            .flat_map(|e| std::iter::repeat_n(e as f32 + 1.0, rank * k))
            .collect();
        let b: Vec<f32> = (0..n_expert)
            .flat_map(|e| std::iter::repeat_n(1.0 / (e as f32 + 1.0), d * rank))
            .collect();
        synth_gguf(
            &[
                // ne is [k, rank, n_expert] for A and [rank, d, n_expert] for B,
                // matching llama.cpp's `A.shape[-2] == B.shape[-1]` assertion.
                (
                    "blk.0.ffn_gate_exps.weight.lora_a",
                    vec![k, rank, n_expert],
                    a,
                ),
                (
                    "blk.0.ffn_gate_exps.weight.lora_b",
                    vec![rank, d, n_expert],
                    b,
                ),
            ],
            Some(rank as f32),
        )
    }

    /// A stacked adapter must split into one independent low-rank pair per
    /// expert, in expert order, reachable only through `get_expert`.
    #[test]
    fn gguf_expert_adapter_splits_per_expert() {
        let (n_expert, rank, k, d) = (4, 2, 3, 5);
        let buf = synth_expert_gguf(n_expert, rank, k, d);
        let adapter = LoraAdapterWeights::from_gguf_bytes(Arc::from(buf.into_boxed_slice()))
            .expect("expert adapter loads");

        // One projection adapted, not one per expert.
        assert_eq!(adapter.target_count(), 1);
        // A per-expert delta is invisible to the dense accessor, so a caller
        // that forgot to route by expert reads `None` instead of expert 0.
        assert!(adapter.get(0, LoraTarget::FfnGateExps).is_none());

        for e in 0..n_expert {
            let t = adapter
                .get_expert(0, LoraTarget::FfnGateExps, e)
                .unwrap_or_else(|| panic!("expert {e} present"));
            assert_eq!((t.rank, t.k, t.d), (rank, k, d));
            // Expert e's A was filled with e+1 and its B with 1/(e+1): proves
            // the split took slice e of *each* factor, not slice 0 repeated or
            // a stride off by one. Checked on both because a bad offset in one
            // of the two is invisible if only the other varies.
            assert!(
                t.a.iter().all(|&x| x == e as f32 + 1.0),
                "expert {e} got A = {:?}",
                &t.a[..t.a.len().min(4)]
            );
            assert!(
                t.b.iter().all(|&x| x == 1.0 / (e as f32 + 1.0)),
                "expert {e} got B = {:?}",
                &t.b[..t.b.len().min(4)]
            );
        }
        // One past the last expert is `None`, not a wrapped index.
        assert!(
            adapter
                .get_expert(0, LoraTarget::FfnGateExps, n_expert)
                .is_none()
        );
    }

    /// Expert `e`'s `A` must be paired with expert `e`'s `B`, not with another
    /// expert's.
    ///
    /// `A` is filled with `e + 1` and `B` with `1 / (e + 1)`, so a correctly
    /// paired expert yields the *same* value for every `e` and the expectation
    /// is a constant. That is the sharp shape here: cross-pairing A from one
    /// slice with B from another breaks the cancellation and the constant fails.
    ///
    /// This does not test that the *forward pass* looks up the routed expert:
    /// the expert index is passed in explicitly. That is pinned by
    /// `expert_factors_are_indexed_by_the_routed_expert` in
    /// `tests/moe_lora_parity.rs`, which drives a real model.
    #[test]
    fn expert_deltas_apply_independently() {
        let (n_expert, rank, k, d) = (3, 2, 4, 3);
        let buf = synth_expert_gguf(n_expert, rank, k, d);
        let adapter =
            LoraAdapterWeights::from_gguf_bytes(Arc::from(buf.into_boxed_slice())).unwrap();

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let sum_x: f32 = x.iter().sum();
        let mut tmp = Vec::new();
        for e in 0..n_expert {
            let t = adapter.get_expert(0, LoraTarget::FfnGateExps, e).unwrap();
            let mut y = vec![0.0f32; d];
            apply_decode(t, &x, &mut y, &mut tmp);
            // A is all (e+1), B all 1/(e+1), alpha == rank so scale == 1:
            // tmp[r] = (e+1)·sum(x); y[o] = rank · (e+1) · sum(x) / (e+1).
            // The (e+1) factors cancel, which is the point: an implementation
            // that paired expert e's A with a different expert's B would NOT
            // cancel, so the constant expectation is the sharp check here.
            let expected = rank as f32 * sum_x;
            assert!(
                y.iter().all(|&v| (v - expected).abs() < 1e-4),
                "expert {e}: {y:?} != {expected}"
            );
        }
    }

    /// A dense (rank-2) factor under an expert target is a malformed adapter:
    /// it describes one expert where the model has many. The loader cannot know
    /// the expert count on its own, so it accepts it as a one-expert stack; what
    /// must not happen is that it silently *adapts* expert 0 and leaves the rest
    /// bare. `validate_dims` is the one that has the model, so it is the one
    /// that has to reject it.
    #[test]
    fn an_unstacked_expert_factor_is_rejected_against_the_model() {
        let (rank, k, d) = (2, 8, 16);
        let buf = synth_gguf(
            &[
                (
                    "blk.2.ffn_gate_exps.weight.lora_a",
                    vec![k, rank],
                    vec![1.0; rank * k],
                ),
                (
                    "blk.2.ffn_gate_exps.weight.lora_b",
                    vec![rank, d],
                    vec![1.0; d * rank],
                ),
            ],
            Some(rank as f32),
        );
        let adapter = LoraAdapterWeights::from_gguf_bytes(Arc::from(buf.into_boxed_slice()))
            .expect("a rank-2 expert factor loads as a one-expert stack");
        // `moe_config()`'s layer 2 is routed with 3 experts, so a one-expert
        // stack must not pass. Dims are otherwise correct (8 -> 16), which is
        // what makes the expert-count check the only thing that can catch it.
        let err = adapter
            .validate_dims(&moe_config())
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 expert deltas"), "{err}");
    }

    /// A stacked factor under a *dense* target must be refused rather than
    /// flattened: nothing downstream would ever index its experts.
    #[test]
    fn rejects_stacked_factors_on_a_dense_target() {
        let (n_expert, rank, k, d) = (2, 2, 3, 5);
        let buf = synth_gguf(
            &[
                (
                    "blk.0.ffn_gate.weight.lora_a",
                    vec![k, rank, n_expert],
                    vec![1.0; n_expert * rank * k],
                ),
                (
                    "blk.0.ffn_gate.weight.lora_b",
                    vec![rank, d, n_expert],
                    vec![1.0; n_expert * d * rank],
                ),
            ],
            Some(rank as f32),
        );
        let err = load_err(LoraAdapterWeights::from_gguf_bytes(Arc::from(
            buf.into_boxed_slice(),
        )));
        assert!(err.contains("stacked"), "{err}");
    }

    /// An expert count past anything real must be refused at load, before the
    /// split turns one flat tensor into two `Vec`s per slice.
    #[test]
    fn rejects_absurd_expert_count() {
        let n = MAX_LORA_EXPERTS + 1;
        let buf = synth_gguf(
            &[
                (
                    "blk.0.ffn_gate_exps.weight.lora_a",
                    vec![1, 1, n],
                    vec![1.0; n],
                ),
                (
                    "blk.0.ffn_gate_exps.weight.lora_b",
                    vec![1, 1, n],
                    vec![1.0; n],
                ),
            ],
            Some(1.0),
        );
        let err = load_err(LoraAdapterWeights::from_gguf_bytes(Arc::from(
            buf.into_boxed_slice(),
        )));
        assert!(err.contains("over the sane maximum"), "{err}");
    }

    /// A and B are separate tensors; disagreeing expert counts would pair
    /// expert `i`'s A with someone else's B.
    #[test]
    fn rejects_mismatched_expert_counts() {
        let (rank, k, d) = (2, 3, 5);
        let buf = synth_gguf(
            &[
                (
                    "blk.0.ffn_gate_exps.weight.lora_a",
                    vec![k, rank, 4],
                    vec![1.0; 4 * rank * k],
                ),
                (
                    "blk.0.ffn_gate_exps.weight.lora_b",
                    vec![rank, d, 2],
                    vec![1.0; 2 * d * rank],
                ),
            ],
            Some(rank as f32),
        );
        let err = load_err(LoraAdapterWeights::from_gguf_bytes(Arc::from(
            buf.into_boxed_slice(),
        )));
        assert!(err.contains("slices"), "{err}");
    }

    /// PEFT stores expert modules one per expert under an architecture-specific
    /// name. Rather than guess and drop what doesn't match, loading must fail
    /// with a pointer at the supported route.
    #[test]
    fn peft_expert_adapter_is_refused_not_dropped() {
        let name = "base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_A.weight";
        let buf = st_buf(
            serde_json::json!({
                name: { "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16] },
            }),
            16,
        );
        let err = load_err(LoraAdapterWeights::from_safetensors_bytes(&buf, None));
        assert!(err.contains("convert_lora_to_gguf"), "{err}");
    }

    /// A four-layer config: layers 0-1 dense, 2-3 routed, shaped like
    /// LFM2.5-8B-A1B's split but small.
    fn moe_config() -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            architecture: "lfm2moe".to_string(),
            n_layers: 4,
            hidden_size: 8,
            intermediate_size: 32,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            vocab_size: 16,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            block_types: vec![crate::model::BlockType::Attention; 4],
            conv_kernel_size: None,
            kv_heads_per_layer: vec![2; 4],
            scalars: crate::model::ScalarMultipliers::default(),
            moe: Some(crate::model::MoeConfig {
                n_expert: 3,
                n_expert_used: 2,
                expert_ff_len: 16,
                is_moe_layer: vec![false, false, true, true],
            }),
            is_causal: true,
            class_labels: Vec::new(),
        }
    }

    /// `has_moe_deltas` covers both halves of "only the routed FFN can apply
    /// this", and the router is the half worth a test.
    ///
    /// A per-expert factor is at least *structurally* distinct, so a backend
    /// without expert kernels skips it (`get` returns `None` for an expert
    /// target). The router is an ordinary dense target with ordinary dims, so
    /// nothing about its shape stops a per-target upload loop from taking it
    /// and then never applying it. If this predicate ever stopped covering
    /// `FfnGateInp`, `Session::attach_lora_adapters` would start admitting
    /// router adapters onto backends that silently drop them, and no output
    /// would look wrong.
    #[test]
    fn has_moe_deltas_covers_the_router_and_the_experts() {
        assert!(
            adapter_for(0, LoraTarget::FfnGateInp, 8, 4, 1).has_moe_deltas(),
            "a router delta is a routed-FFN delta"
        );
        assert!(
            adapter_for(0, LoraTarget::FfnGateExps, 8, 16, 3).has_moe_deltas(),
            "a per-expert delta is a routed-FFN delta"
        );
        assert!(
            !adapter_for(0, LoraTarget::FfnGate, 8, 16, 1).has_moe_deltas(),
            "a dense FFN delta must not trip the gate; every GPU backend applies it"
        );
        assert!(
            !adapter_for(0, LoraTarget::AttnQ, 8, 8, 1).has_moe_deltas(),
            "an attention delta must not trip the gate"
        );
    }

    /// Build a one-layer adapter carrying `target` with the given dims, for
    /// `n_slices` experts (1 = dense).
    fn adapter_for(
        layer: usize,
        target: LoraTarget,
        k: usize,
        d: usize,
        n_slices: usize,
    ) -> Arc<LoraAdapterWeights> {
        let rank = 2;
        let stem = target.gguf_stem();
        let (ne_a, ne_b) = if n_slices > 1 {
            (vec![k, rank, n_slices], vec![rank, d, n_slices])
        } else {
            (vec![k, rank], vec![rank, d])
        };
        let buf = synth_gguf(
            &[
                (
                    &format!("blk.{layer}.{stem}.weight.lora_a"),
                    ne_a,
                    vec![0.5; n_slices * rank * k],
                ),
                (
                    &format!("blk.{layer}.{stem}.weight.lora_b"),
                    ne_b,
                    vec![0.5; n_slices * d * rank],
                ),
            ],
            Some(rank as f32),
        );
        LoraAdapterWeights::from_gguf_bytes(Arc::from(buf.into_boxed_slice()))
            .expect("synthetic adapter loads")
    }

    /// The silent-drop case the expert path exists to close: a dense `ffn_gate`
    /// adapter aimed at a routed layer has entirely plausible dims (the model's
    /// own leading blocks are that wide), so only the layer kind catches it.
    #[test]
    fn validate_dims_rejects_a_dense_ffn_adapter_on_a_routed_layer() {
        let cfg = moe_config();
        // Layer 1 is dense: the same adapter is accepted there.
        adapter_for(1, LoraTarget::FfnGate, 8, 32, 1)
            .validate_dims(&cfg)
            .expect("dense adapter on a dense layer");

        let err = adapter_for(2, LoraTarget::FfnGate, 8, 32, 1)
            .validate_dims(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("mixture-of-experts"), "{err}");
    }

    /// ...and the mirror image: expert deltas aimed at a dense layer.
    #[test]
    fn validate_dims_rejects_an_expert_adapter_on_a_dense_layer() {
        let cfg = moe_config();
        adapter_for(3, LoraTarget::FfnGateExps, 8, 16, 3)
            .validate_dims(&cfg)
            .expect("expert adapter on a routed layer");

        let err = adapter_for(0, LoraTarget::FfnGateExps, 8, 16, 3)
            .validate_dims(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is dense"), "{err}");
    }

    /// Experts are `expert_ff_len` wide (16), not `intermediate_size` (32).
    /// Validating against the dense width would accept an adapter twice too big
    /// and then read past every expert's factors.
    #[test]
    fn validate_dims_uses_the_expert_width_not_the_dense_one() {
        let cfg = moe_config();
        let err = adapter_for(2, LoraTarget::FfnGateExps, 8, 32, 3)
            .validate_dims(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out=16"), "{err}");
    }

    /// The router is `hidden → n_expert`, and it is a real target: an adapter
    /// that moves a token across a selection boundary must not be dropped.
    #[test]
    fn validate_dims_accepts_the_router_and_checks_its_width() {
        let cfg = moe_config();
        adapter_for(2, LoraTarget::FfnGateInp, 8, 3, 1)
            .validate_dims(&cfg)
            .expect("router adapter validates");
        let err = adapter_for(2, LoraTarget::FfnGateInp, 8, 4, 1)
            .validate_dims(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out=3"), "{err}");
    }

    /// An adapter trained on a differently-sized expert set must be rejected
    /// before the forward pass indexes an expert it doesn't have.
    #[test]
    fn validate_dims_rejects_wrong_expert_count() {
        let cfg = moe_config();
        let err = adapter_for(2, LoraTarget::FfnGateExps, 8, 16, 2)
            .validate_dims(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 expert deltas"), "{err}");
    }

    #[test]
    fn rejects_overflow_shape() {
        // A crafted shape whose product overflows usize must ERROR, not panic
        // (debug overflow-check) nor silently wrap (release).
        let big = (u64::MAX / 2) as usize;
        let name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let buf = st_buf(
            serde_json::json!({
                name: { "dtype": "F32", "shape": [big, big], "data_offsets": [0, 4] },
            }),
            4,
        );
        assert!(LoraAdapterWeights::from_safetensors_bytes(&buf, None).is_err());
    }

    #[test]
    fn peft_classifier_adapter_loads_and_validates() {
        // 3 classes, hidden_size 8: weight [3, 8] = 24 floats = 96 bytes, bias [3] = 3 floats = 12 bytes
        let w_bytes = 24 * 4;
        let b_bytes = 3 * 4;
        let total_bytes = w_bytes + b_bytes;

        let buf = st_buf(
            serde_json::json!({
                "base_model.model.classifier.weight": {
                    "dtype": "F32",
                    "shape": [3, 8],
                    "data_offsets": [0, w_bytes]
                },
                "base_model.model.classifier.bias": {
                    "dtype": "F32",
                    "shape": [3],
                    "data_offsets": [w_bytes, total_bytes]
                }
            }),
            total_bytes,
        );

        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, None).unwrap();
        assert!(adapter.is_classifier());
        assert_eq!(adapter.num_classes(), 3); // parsed from classifier.weight shape and bias length

        let labels = vec![
            "O".to_string(),
            "B-EMAIL".to_string(),
            "I-EMAIL".to_string(),
        ];
        let adapter = adapter.with_class_labels(labels);
        assert_eq!(adapter.num_classes(), 3);

        let cfg = crate::model::ModelConfig {
            architecture: "lfm2".to_string(),
            n_layers: 2,
            hidden_size: 8,
            intermediate_size: 32,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            vocab_size: 16,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            block_types: vec![crate::model::BlockType::Attention; 2],
            conv_kernel_size: None,
            kv_heads_per_layer: vec![2; 2],
            scalars: crate::model::ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };
        assert!(adapter.validate_dims(&cfg).is_ok());

        // Mismatched hidden size should fail validation
        let bad_cfg = crate::model::ModelConfig {
            hidden_size: 16,
            ..cfg
        };
        assert!(adapter.validate_dims(&bad_cfg).is_err());
    }
}

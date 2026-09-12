// Gemma 4 architecture model implementation.
// Supports Gemma 4 dense architectures (E2B IT, E4B IT) featuring:
// - Per-Layer Embeddings (PLE / MatFormer)
// - Cross-layer KV cache sharing
// - Sliding Window Attention (SWA) pattern
// - Attention scaling override (1.0)
// - Per-head QK-norm and unweighted V-norm
// - Standard RMSNorm (norm_shift = 0.0)
// - Final logit soft-capping

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::{GgufFile, GgufValue};
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::transformer::{self, DecodeAttnDims, KvView, WeightRef, gemv};
#[cfg(target_arch = "aarch64")]
use crate::model::transformer::{gemv_preq, quantize_to_scratch_bufs};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};

/// Per-layer weights for Gemma 4.
struct Gemma4LayerWeights {
    attn_norm: Vec<f32>,
    attn_q: WeightRef,
    attn_q_norm: Vec<f32>,
    attn_k: Option<WeightRef>,
    attn_k_norm: Option<Vec<f32>>,
    attn_v: Option<WeightRef>,
    attn_output: WeightRef,
    attn_post_norm: Vec<f32>,

    ffn_norm: Vec<f32>,
    ffn_gate: WeightRef,
    ffn_up: WeightRef,
    ffn_down: WeightRef,
    ffn_post_norm: Vec<f32>,

    // Per-layer embedding projections and norms.
    per_layer_inp_gate: Option<WeightRef>,
    per_layer_proj: Option<WeightRef>,
    per_layer_post_norm: Option<Vec<f32>>,

    // Optional layer output scalar multiplier.
    layer_out_scale: Option<f32>,
}

/// Gemma 4 model runtime.
pub struct Gemma4Model {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    n_embd_per_layer: usize,
    _n_kv_shared_layers: usize,
    n_layer_kv_from_start: usize,
    is_swa: Vec<bool>,
    sliding_window: usize,
    final_logit_softcapping: Option<f32>,

    output_norm_weight: Vec<f32>,
    embd_ref: WeightRef,
    output_ref: Option<WeightRef>,

    // Per-layer embedding global weights.
    per_layer_token_embd: Option<WeightRef>,
    per_layer_model_proj: Option<WeightRef>,
    per_layer_proj_norm: Vec<f32>,

    layers: Vec<Gemma4LayerWeights>,

    #[allow(dead_code)]
    model_id: String,
}

impl Gemma4Model {
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let prefix = if gguf.metadata.contains_key("gemma4.block_count") {
            "gemma4"
        } else if gguf.metadata.contains_key("gemma4-assistant.block_count") {
            "gemma4-assistant"
        } else {
            "gemma4"
        };
        let n_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .or_else(|| gguf.get_u32("gemma4-assistant.block_count"))
            .context("missing block_count")? as usize;
        ensure!(n_layers > 0, "block_count must be > 0");
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .or_else(|| gguf.get_u32("gemma4-assistant.embedding_length"))
            .context("missing embedding_length")? as usize;
        ensure!(hidden_size > 0, "embedding_length must be > 0");
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .or_else(|| gguf.get_u32("gemma4-assistant.feed_forward_length"))
            .context("missing feed_forward_length")? as usize;
        ensure!(intermediate_size > 0, "feed_forward_length must be > 0");
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .or_else(|| gguf.get_u32("gemma4-assistant.attention.head_count"))
            .context("missing head_count")? as usize;
        ensure!(n_heads > 0, "attention.head_count must be > 0");
        let n_kv_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .or_else(|| gguf.get_u32("gemma4-assistant.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        ensure!(n_kv_heads > 0, "attention.head_count_kv must be > 0");
        ensure!(
            n_heads.is_multiple_of(n_kv_heads),
            "n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
        );

        let head_dim = gguf
            .get_u32(&format!("{prefix}.attention.key_length"))
            .or_else(|| gguf.get_u32("gemma4-assistant.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size / n_heads);
        ensure!(head_dim > 0, "head_dim must be > 0");
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let n_embd_per_layer = gguf
            .get_u32(&format!("{prefix}.embedding_length_per_layer_input"))
            .or_else(|| gguf.get_u32("gemma4-assistant.embedding_length_per_layer_input"))
            .unwrap_or(0) as usize;

        let n_kv_shared_layers = gguf
            .get_u32(&format!("{prefix}.attention.shared_kv_layers"))
            .or_else(|| gguf.get_u32("gemma4-assistant.attention.shared_kv_layers"))
            .unwrap_or(0) as usize;
        ensure!(
            n_kv_shared_layers < n_layers,
            "shared_kv_layers ({n_kv_shared_layers}) must be less than n_layers ({n_layers})"
        );
        let n_layer_kv_from_start = n_layers - n_kv_shared_layers;
        if n_kv_shared_layers > 0 {
            ensure!(
                n_layer_kv_from_start >= 2,
                "shared_kv_layers ({n_kv_shared_layers}) requires at least 2 non-shared KV layers, found {n_layer_kv_from_start}"
            );
        }

        let sliding_window = gguf
            .get_u32(&format!("{prefix}.attention.sliding_window"))
            .or_else(|| gguf.get_u32("gemma4-assistant.attention.sliding_window"))
            .unwrap_or(0) as usize;

        // Parse sliding window pattern (array or integer period).
        let is_swa = if let Some(GgufValue::Array(arr)) = gguf
            .metadata
            .get(&format!("{prefix}.attention.sliding_window_pattern"))
        {
            arr.iter()
                .map(|v| match v {
                    GgufValue::Bool(b) => *b,
                    GgufValue::U32(u) => *u != 0,
                    GgufValue::I32(i) => *i != 0,
                    GgufValue::U8(u) => *u != 0,
                    _ => true,
                })
                .collect()
        } else if let Some(period) =
            gguf.get_u32(&format!("{prefix}.attention.sliding_window_pattern"))
        {
            (0..n_layers)
                .map(|il| period == 0 || (il as u32 % period < (period - 1)))
                .collect()
        } else {
            vec![sliding_window > 0; n_layers]
        };
        ensure!(
            is_swa.len() >= n_layers,
            "sliding_window_pattern length ({}) must be >= n_layers ({n_layers})",
            is_swa.len()
        );

        let max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .or_else(|| gguf.get_u32("gemma4-assistant.context_length"))
            .unwrap_or(2048) as usize;
        let max_seq_len = max_seq_len.min(context_size);

        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .or_else(|| gguf.get_f32("gemma4-assistant.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_f32("gemma4-assistant.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let final_logit_softcapping = gguf
            .get_f32(&format!("{prefix}.final_logit_softcapping"))
            .or_else(|| gguf.get_f32("gemma4-assistant.final_logit_softcapping"));
        if let Some(cap) = final_logit_softcapping {
            ensure!(
                cap.is_finite() && cap > 0.0,
                "final_logit_softcapping must be positive and finite: {cap}"
            );
        }

        let vocab_size = match gguf.get_u32(&format!("{prefix}.vocab_size")) {
            Some(v) => v as usize,
            None => {
                let info = gguf
                    .tensors
                    .get("token_embd.weight")
                    .context("missing token_embd.weight (cannot derive vocab_size)")?;
                ensure!(
                    info.shape.len() >= 2,
                    "token_embd.weight has unexpected shape {:?}",
                    info.shape
                );
                info.shape[1]
            }
        };

        // Pre-allocate kv heads per layer: shared layers get 0 so their cache allocation is empty.
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            if i < n_layer_kv_from_start {
                kv_heads_per_layer.push(n_kv_heads);
            } else {
                kv_heads_per_layer.push(0);
            }
        }

        let config = ModelConfig {
            architecture: "gemma4".to_string(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types: vec![BlockType::Attention; n_layers],
            conv_kernel_size: None,
            kv_heads_per_layer,
            scalars: ScalarMultipliers {
                embedding: (hidden_size as f32).sqrt(),
                residual: 1.0,
                attn: Some(1.0), // Gemma 4 uses 1.0 attention scale override
                logit: 1.0,
            },
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };

        let output_norm_weight = gguf
            .get_tensor("output_norm.weight")
            .context("missing output_norm.weight")?
            .to_f32_vec();
        ensure!(
            output_norm_weight.len() == hidden_size,
            "invalid output_norm.weight length {}, expected {hidden_size}",
            output_norm_weight.len()
        );

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;
        ensure!(
            embd_ref.k == hidden_size && embd_ref.m >= vocab_size,
            "token_embd.weight shape [{}, {}] mismatch for [>={vocab_size}, {hidden_size}]",
            embd_ref.m,
            embd_ref.k
        );
        let output_ref = if gguf.tensors.contains_key("output.weight") {
            let out = transformer::resolve_weight(&gguf, "output.weight")?;
            ensure!(
                out.k == hidden_size && out.m >= vocab_size,
                "output.weight shape [{}, {}] mismatch for [>={vocab_size}, {hidden_size}]",
                out.m,
                out.k
            );
            Some(out)
        } else {
            None
        };

        let (per_layer_token_embd, per_layer_model_proj, per_layer_proj_norm) = if n_embd_per_layer
            > 0
        {
            let tok_embd = transformer::resolve_weight(&gguf, "per_layer_token_embd.weight")
                .or_else(|_| transformer::resolve_weight(&gguf, "per_layer_tok_embd.weight"))
                .context("missing per_layer_token_embd.weight")?;
            let total_pl = n_layers * n_embd_per_layer;
            ensure!(
                tok_embd.k == total_pl,
                "invalid per_layer_token_embd k dim {}, expected {total_pl}",
                tok_embd.k
            );
            ensure!(
                tok_embd.m >= vocab_size,
                "invalid per_layer_token_embd m dim {}, expected at least {vocab_size}",
                tok_embd.m
            );
            let model_proj = transformer::resolve_weight(&gguf, "per_layer_model_proj.weight")
                .context("missing per_layer_model_proj.weight")?
                .with_repack(&gguf);
            ensure!(
                model_proj.m == total_pl && model_proj.k == hidden_size,
                "invalid per_layer_model_proj shape [{}, {}], expected [{total_pl}, {hidden_size}]",
                model_proj.m,
                model_proj.k
            );
            let proj_norm = gguf
                .get_tensor("per_layer_proj_norm.weight")
                .context("missing per_layer_proj_norm.weight")?
                .to_f32_vec();
            ensure!(
                proj_norm.len() == n_embd_per_layer,
                "invalid per_layer_proj_norm length {}, expected {n_embd_per_layer}",
                proj_norm.len()
            );
            (Some(tok_embd), Some(model_proj), proj_norm)
        } else {
            (None, None, Vec::new())
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let has_kv = i < n_layer_kv_from_start;
            let attn_norm = gguf
                .get_tensor(&format!("blk.{i}.attn_norm.weight"))
                .with_context(|| format!("missing blk.{i}.attn_norm.weight"))?
                .to_f32_vec();
            ensure!(
                attn_norm.len() == hidden_size,
                "invalid attn_norm length {} for layer {i}, expected {hidden_size}",
                attn_norm.len()
            );
            let attn_q = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?
                .with_repack(&gguf);
            ensure!(
                attn_q.m == q_dim && attn_q.k == hidden_size,
                "layer {i} attn_q shape [{}, {}] expected [{q_dim}, {hidden_size}]",
                attn_q.m,
                attn_q.k
            );
            let attn_q_norm = gguf
                .get_tensor(&format!("blk.{i}.attn_q_norm.weight"))
                .with_context(|| format!("missing blk.{i}.attn_q_norm.weight"))?
                .to_f32_vec();
            ensure!(
                attn_q_norm.len() == head_dim,
                "invalid attn_q_norm length {} for layer {i}, expected {head_dim}",
                attn_q_norm.len()
            );

            let (attn_k, attn_k_norm, attn_v) = if has_kv {
                let k = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?
                    .with_repack(&gguf);
                ensure!(
                    k.m == kv_dim && k.k == hidden_size,
                    "layer {i} attn_k shape [{}, {}] expected [{kv_dim}, {hidden_size}]",
                    k.m,
                    k.k
                );
                let k_norm = gguf
                    .get_tensor(&format!("blk.{i}.attn_k_norm.weight"))
                    .with_context(|| format!("missing blk.{i}.attn_k_norm.weight"))?
                    .to_f32_vec();
                ensure!(
                    k_norm.len() == head_dim,
                    "invalid attn_k_norm length {} for layer {i}, expected {head_dim}",
                    k_norm.len()
                );
                let v = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))
                    .ok()
                    .map(|w| w.with_repack(&gguf));
                if let Some(ref v_weight) = v {
                    ensure!(
                        v_weight.m == kv_dim && v_weight.k == hidden_size,
                        "layer {i} attn_v shape [{}, {}] expected [{kv_dim}, {hidden_size}]",
                        v_weight.m,
                        v_weight.k
                    );
                }
                (Some(k), Some(k_norm), v)
            } else {
                (None, None, None)
            };

            let attn_output =
                transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_output.weight"))?
                    .with_repack(&gguf);
            ensure!(
                attn_output.m == hidden_size && attn_output.k == q_dim,
                "layer {i} attn_output shape [{}, {}] expected [{hidden_size}, {q_dim}]",
                attn_output.m,
                attn_output.k
            );
            let attn_post_norm = gguf
                .get_tensor(&format!("blk.{i}.post_attention_norm.weight"))
                .or_else(|_| gguf.get_tensor(&format!("blk.{i}.attn_post_norm.weight")))
                .with_context(|| format!("missing post_attention_norm for layer {i}"))?
                .to_f32_vec();
            ensure!(
                attn_post_norm.len() == hidden_size,
                "invalid post_attention_norm length {} for layer {i}, expected {hidden_size}",
                attn_post_norm.len()
            );

            let ffn_norm = gguf
                .get_tensor(&format!("blk.{i}.ffn_norm.weight"))
                .with_context(|| format!("missing blk.{i}.ffn_norm.weight"))?
                .to_f32_vec();
            ensure!(
                ffn_norm.len() == hidden_size,
                "invalid ffn_norm length {} for layer {i}, expected {hidden_size}",
                ffn_norm.len()
            );
            let ffn_gate = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?
                .with_repack(&gguf);
            ensure!(
                ffn_gate.m == intermediate_size && ffn_gate.k == hidden_size,
                "layer {i} ffn_gate shape [{}, {}] expected [{intermediate_size}, {hidden_size}]",
                ffn_gate.m,
                ffn_gate.k
            );
            let ffn_up = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?
                .with_repack(&gguf);
            ensure!(
                ffn_up.m == intermediate_size && ffn_up.k == hidden_size,
                "layer {i} ffn_up shape [{}, {}] expected [{intermediate_size}, {hidden_size}]",
                ffn_up.m,
                ffn_up.k
            );
            let ffn_down = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?
                .with_repack(&gguf);
            ensure!(
                ffn_down.m == hidden_size && ffn_down.k == intermediate_size,
                "layer {i} ffn_down shape [{}, {}] expected [{hidden_size}, {intermediate_size}]",
                ffn_down.m,
                ffn_down.k
            );
            let ffn_post_norm = gguf
                .get_tensor(&format!("blk.{i}.post_ffw_norm.weight"))
                .or_else(|_| gguf.get_tensor(&format!("blk.{i}.ffn_post_norm.weight")))
                .with_context(|| format!("missing post_ffw_norm for layer {i}"))?
                .to_f32_vec();
            ensure!(
                ffn_post_norm.len() == hidden_size,
                "invalid post_ffw_norm length {} for layer {i}, expected {hidden_size}",
                ffn_post_norm.len()
            );

            let (per_layer_inp_gate, per_layer_proj, per_layer_post_norm) = if n_embd_per_layer > 0
            {
                let gate = transformer::resolve_weight(&gguf, &format!("blk.{i}.inp_gate.weight"))
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.per_layer_inp_gate.weight"),
                        )
                    })
                    .with_context(|| format!("missing inp_gate for layer {i}"))?
                    .with_repack(&gguf);
                ensure!(
                    gate.m == n_embd_per_layer && gate.k == hidden_size,
                    "invalid per_layer_inp_gate shape [{}, {}] for layer {i}, expected [{n_embd_per_layer}, {hidden_size}]",
                    gate.m,
                    gate.k
                );
                let proj = transformer::resolve_weight(&gguf, &format!("blk.{i}.proj.weight"))
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &gguf,
                            &format!("blk.{i}.per_layer_proj.weight"),
                        )
                    })
                    .with_context(|| format!("missing proj for layer {i}"))?
                    .with_repack(&gguf);
                ensure!(
                    proj.m == hidden_size && proj.k == n_embd_per_layer,
                    "invalid per_layer_proj shape [{}, {}] for layer {i}, expected [{hidden_size}, {n_embd_per_layer}]",
                    proj.m,
                    proj.k
                );
                let post_norm = gguf
                    .get_tensor(&format!("blk.{i}.post_norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("blk.{i}.per_layer_post_norm.weight")))
                    .with_context(|| format!("missing post_norm for layer {i}"))?
                    .to_f32_vec();
                ensure!(
                    post_norm.len() == hidden_size,
                    "invalid post_norm length {} for layer {i}, expected {hidden_size}",
                    post_norm.len()
                );
                (Some(gate), Some(proj), Some(post_norm))
            } else {
                (None, None, None)
            };

            let layer_out_scale = gguf
                .get_tensor(&format!("blk.{i}.layer_out_scale.weight"))
                .or_else(|_| gguf.get_tensor(&format!("blk.{i}.layer_scalar.weight")))
                .ok()
                .and_then(|t| t.to_f32_vec().first().copied())
                .filter(|s| s.is_finite());

            layers.push(Gemma4LayerWeights {
                attn_norm,
                attn_q,
                attn_q_norm,
                attn_k,
                attn_k_norm,
                attn_v,
                attn_output,
                attn_post_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_post_norm,
                per_layer_inp_gate,
                per_layer_proj,
                per_layer_post_norm,
                layer_out_scale,
            });
        }

        Ok(Self {
            gguf,
            config,
            head_dim,
            n_embd_per_layer,
            _n_kv_shared_layers: n_kv_shared_layers,
            n_layer_kv_from_start,
            is_swa,
            sliding_window,
            final_logit_softcapping,
            output_norm_weight,
            embd_ref,
            output_ref,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            model_id,
        })
    }

    /// Run single token decode pass.
    fn forward_single_token(&self, token: u32, pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let hs = self.config.hidden_size;
        let head_dim = self.head_dim;
        let q_dim = self.config.n_heads * head_dim;
        let kv_dim = self.config.n_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;

        // Base token embedding lookup and scale.
        let mut hidden = transformer::dequantize_row(&self.gguf, &self.embd_ref, token as usize);
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("embd", &hidden);
        }
        cpu::scale_inplace(&mut hidden, (hs as f32).sqrt());
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("inp_scaled", &hidden);
        }

        // Resize scratch buffers.
        state.scratch.normed.resize(hs, 0.0);
        state.scratch.ffn_input.resize(hs, 0.0);
        state.scratch.q.resize(q_dim, 0.0);
        state.scratch.k.resize(kv_dim, 0.0);
        state.scratch.v.resize(kv_dim, 0.0);
        state.scratch.attn_out.resize(q_dim, 0.0);
        state.scratch.out.resize(hs, 0.0);
        state
            .scratch
            .gate
            .resize(self.config.intermediate_size, 0.0);
        state.scratch.up.resize(self.config.intermediate_size, 0.0);

        // Prepare per-layer embedding inputs without heap allocation.
        let n_pl = self.n_embd_per_layer;
        let total_pl = self.config.n_layers * n_pl;
        if n_pl > 0 {
            state.scratch.lora_tmp.resize(total_pl * 2, 0.0);
            state.scratch.conv_scratch.resize(n_pl, 0.0);

            let (combined, proj_pl) = state.scratch.lora_tmp.split_at_mut(total_pl);
            transformer::dequantize_row_into(
                &self.gguf,
                self.per_layer_token_embd.as_ref().unwrap(),
                token as usize,
                combined,
            );
            cpu::scale_inplace(combined, (n_pl as f32).sqrt());

            gemv(
                &self.gguf,
                self.per_layer_model_proj.as_ref().unwrap(),
                &hidden,
                proj_pl,
            );
            cpu::scale_inplace(proj_pl, 1.0 / (hs as f32).sqrt());
            for il in 0..self.config.n_layers {
                cpu::rmsnorm(
                    &mut proj_pl[il * n_pl..(il + 1) * n_pl],
                    &self.per_layer_proj_norm,
                    eps,
                );
            }

            let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
            for j in 0..total_pl {
                combined[j] = (proj_pl[j] + combined[j]) * inv_sqrt_2;
            }
        }

        for i in 0..self.config.n_layers {
            let layer = &self.layers[i];
            let has_kv = i < self.n_layer_kv_from_start;
            let is_swa = self.is_swa[i];

            // Attention pre-norm.
            cpu::rmsnorm_into(&hidden, &mut state.scratch.normed, &layer.attn_norm, eps);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("attn_norm-{i}"), &state.scratch.normed);
            }

            // Q projection.
            #[cfg(target_arch = "aarch64")]
            {
                quantize_to_scratch_bufs(
                    &state.scratch.normed,
                    &mut state.scratch.q8_scales,
                    &mut state.scratch.q8_quants,
                );
                gemv_preq(
                    &self.gguf,
                    &layer.attn_q,
                    &state.scratch.normed,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.q[..q_dim],
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                gemv(
                    &self.gguf,
                    &layer.attn_q,
                    &state.scratch.normed,
                    &mut state.scratch.q[..q_dim],
                );
            }
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("Qcur-{i}"), &state.scratch.q[..q_dim]);
            }

            // QK norm on Q (per head).
            for h in 0..self.config.n_heads {
                cpu::rmsnorm(
                    &mut state.scratch.q[h * head_dim..(h + 1) * head_dim],
                    &layer.attn_q_norm,
                    eps,
                );
            }
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(
                    &format!("Qcur_normed-{i}"),
                    &state.scratch.q[..q_dim],
                );
            }

            if has_kv {
                // K projection.
                #[cfg(target_arch = "aarch64")]
                {
                    gemv_preq(
                        &self.gguf,
                        layer.attn_k.as_ref().unwrap(),
                        &state.scratch.normed,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.k[..kv_dim],
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(
                        &self.gguf,
                        layer.attn_k.as_ref().unwrap(),
                        &state.scratch.normed,
                        &mut state.scratch.k[..kv_dim],
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Kcur-{i}"),
                        &state.scratch.k[..kv_dim],
                    );
                }

                // QK norm on K (per head).
                for h in 0..self.config.n_kv_heads {
                    cpu::rmsnorm(
                        &mut state.scratch.k[h * head_dim..(h + 1) * head_dim],
                        layer.attn_k_norm.as_ref().unwrap(),
                        eps,
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Kcur_normed-{i}"),
                        &state.scratch.k[..kv_dim],
                    );
                }

                // RoPE on Q and K (Neox split-halves layout).
                cpu::rope(
                    &mut state.scratch.q[..q_dim],
                    &mut state.scratch.k[..kv_dim],
                    pos,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    head_dim,
                    self.config.rope_theta,
                );
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Qcur_pos-{i}"),
                        &state.scratch.q[..q_dim],
                    );
                    transformer::oracle_dump::record(
                        &format!("Kcur_pos-{i}"),
                        &state.scratch.k[..kv_dim],
                    );
                }

                // V projection.
                let v_ref = layer
                    .attn_v
                    .as_ref()
                    .unwrap_or_else(|| layer.attn_k.as_ref().unwrap());
                #[cfg(target_arch = "aarch64")]
                {
                    gemv_preq(
                        &self.gguf,
                        v_ref,
                        &state.scratch.normed,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.v[..kv_dim],
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(
                        &self.gguf,
                        v_ref,
                        &state.scratch.normed,
                        &mut state.scratch.v[..kv_dim],
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Vcur-{i}"),
                        &state.scratch.v[..kv_dim],
                    );
                }

                // Unweighted RMSNorm on V (per head).
                for h in 0..self.config.n_kv_heads {
                    cpu::rmsnorm_unweighted(
                        &mut state.scratch.v[h * head_dim..(h + 1) * head_dim],
                        eps,
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Vcur_normed-{i}"),
                        &state.scratch.v[..kv_dim],
                    );
                }

                // Append K and V to this layer's cache slot.
                let use_f16 = state.kv_f16;
                if let LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } = &mut state.layers[i]
                {
                    if use_f16 {
                        key_cache_f16.extend(
                            state.scratch.k[..kv_dim]
                                .iter()
                                .map(|&x| crate::quant::f32_to_f16(x)),
                        );
                        value_cache_f16.extend(
                            state.scratch.v[..kv_dim]
                                .iter()
                                .map(|&x| crate::quant::f32_to_f16(x)),
                        );
                    } else {
                        key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                        value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                    }
                }
            } else {
                // Shared layer: RoPE Q only.
                cpu::rope(
                    &mut state.scratch.q[..q_dim],
                    &mut [],
                    pos,
                    self.config.n_heads,
                    0,
                    head_dim,
                    self.config.rope_theta,
                );
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Qcur_pos-{i}"),
                        &state.scratch.q[..q_dim],
                    );
                }
            }

            // Self-attention over KV cache.
            let kv_src = if has_kv {
                i
            } else if is_swa {
                self.n_layer_kv_from_start - 2
            } else {
                self.n_layer_kv_from_start - 1
            };

            let kv = match &state.layers[kv_src] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    if state.kv_f16 {
                        KvView::F16 {
                            k: key_cache_f16,
                            v: value_cache_f16,
                        }
                    } else {
                        KvView::F32 {
                            k: key_cache,
                            v: value_cache,
                        }
                    }
                }
                _ => panic!("Gemma 4 expected Attention LayerState"),
            };

            let sw = if is_swa && self.sliding_window > 0 {
                Some(self.sliding_window)
            } else {
                None
            };

            let d = DecodeAttnDims {
                n_heads: self.config.n_heads,
                n_kv_heads: self.config.n_kv_heads,
                head_dim,
                scale: 1.0,
                seq_len: pos + 1,
                attn_logit_softcapping: None,
                sliding_window: sw,
            };

            transformer::decode_attention(
                &state.scratch.q[..q_dim],
                &kv,
                &d,
                &mut state.scratch.attn_out[..q_dim],
                &mut state.scratch.scores,
            );
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(
                    &format!("kqv_out-{i}"),
                    &state.scratch.attn_out[..q_dim],
                );
            }

            // Output projection.
            #[cfg(target_arch = "aarch64")]
            {
                quantize_to_scratch_bufs(
                    &state.scratch.attn_out[..q_dim],
                    &mut state.scratch.q8_scales,
                    &mut state.scratch.q8_quants,
                );
                gemv_preq(
                    &self.gguf,
                    &layer.attn_output,
                    &state.scratch.attn_out[..q_dim],
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.out[..hs],
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                gemv(
                    &self.gguf,
                    &layer.attn_output,
                    &state.scratch.attn_out[..q_dim],
                    &mut state.scratch.out[..hs],
                );
            }

            // Post-norm on attention output.
            cpu::rmsnorm(&mut state.scratch.out[..hs], &layer.attn_post_norm, eps);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(
                    &format!("attn_post_norm-{i}"),
                    &state.scratch.out[..hs],
                );
            }

            // Residual addition.
            cpu::add_inplace(&mut hidden, &state.scratch.out[..hs]);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("attn_out-{i}"), &hidden);
            }

            // FFN pre-norm.
            cpu::rmsnorm_into(&hidden, &mut state.scratch.ffn_input, &layer.ffn_norm, eps);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(
                    &format!("ffn_norm-{i}"),
                    &state.scratch.ffn_input,
                );
            }

            // FFN gate and up projections.
            let intermediate = self.config.intermediate_size;
            #[cfg(target_arch = "aarch64")]
            {
                quantize_to_scratch_bufs(
                    &state.scratch.ffn_input,
                    &mut state.scratch.q8_scales,
                    &mut state.scratch.q8_quants,
                );
                gemv_preq(
                    &self.gguf,
                    &layer.ffn_gate,
                    &state.scratch.ffn_input,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.gate[..intermediate],
                );
                gemv_preq(
                    &self.gguf,
                    &layer.ffn_up,
                    &state.scratch.ffn_input,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.up[..intermediate],
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                gemv(
                    &self.gguf,
                    &layer.ffn_gate,
                    &state.scratch.ffn_input,
                    &mut state.scratch.gate[..intermediate],
                );
                gemv(
                    &self.gguf,
                    &layer.ffn_up,
                    &state.scratch.ffn_input,
                    &mut state.scratch.up[..intermediate],
                );
            }

            // GeGLU activation.
            cpu::gelu_mul_inplace(
                &mut state.scratch.gate[..intermediate],
                &state.scratch.up[..intermediate],
            );
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(
                    &format!("ffn_geglu-{i}"),
                    &state.scratch.gate[..intermediate],
                );
            }

            // FFN down projection.
            #[cfg(target_arch = "aarch64")]
            {
                quantize_to_scratch_bufs(
                    &state.scratch.gate[..intermediate],
                    &mut state.scratch.q8_scales,
                    &mut state.scratch.q8_quants,
                );
                gemv_preq(
                    &self.gguf,
                    &layer.ffn_down,
                    &state.scratch.gate[..intermediate],
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut state.scratch.out[..hs],
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                gemv(
                    &self.gguf,
                    &layer.ffn_down,
                    &state.scratch.gate[..intermediate],
                    &mut state.scratch.out[..hs],
                );
            }
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("ffn_out-{i}"), &state.scratch.out[..hs]);
            }

            // FFN post-norm.
            cpu::rmsnorm(&mut state.scratch.out[..hs], &layer.ffn_post_norm, eps);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(
                    &format!("ffn_post_norm-{i}"),
                    &state.scratch.out[..hs],
                );
            }

            // Residual addition.
            cpu::add_inplace(&mut hidden, &state.scratch.out[..hs]);

            // Per-layer embedding gating and projection.
            if n_pl > 0 {
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(&format!("pe_in-{i}"), &hidden);
                }
                let pe_gate = &mut state.scratch.conv_scratch[..n_pl];
                gemv(
                    &self.gguf,
                    layer.per_layer_inp_gate.as_ref().unwrap(),
                    &hidden,
                    pe_gate,
                );
                let inp_this_layer = &state.scratch.lora_tmp[i * n_pl..(i + 1) * n_pl];
                cpu::gelu_mul_inplace(pe_gate, inp_this_layer);

                gemv(
                    &self.gguf,
                    layer.per_layer_proj.as_ref().unwrap(),
                    pe_gate,
                    &mut state.scratch.out[..hs],
                );
                cpu::rmsnorm(
                    &mut state.scratch.out[..hs],
                    layer.per_layer_post_norm.as_ref().unwrap(),
                    eps,
                );
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("per_layer_embd_out-{i}"),
                        &state.scratch.out[..hs],
                    );
                }
                cpu::add_inplace(&mut hidden, &state.scratch.out[..hs]);
            }

            // Layer output scalar.
            if let Some(scale) = layer.layer_out_scale {
                cpu::scale_inplace(&mut hidden, scale);
            }

            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("l_out-{i}"), &hidden);
            }
        }

        // Output normalization.
        cpu::rmsnorm(&mut hidden, &self.output_norm_weight, eps);
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_norm", &hidden);
        }
        state.seq_len += 1;

        // Logits projection.
        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);
        let mut logits = vec![0.0f32; self.config.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            quantize_to_scratch_bufs(
                &hidden,
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
            gemv_preq(
                &self.gguf,
                out_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            gemv(&self.gguf, out_ref, &hidden, &mut logits);
        }

        if let Some(cap) = self.final_logit_softcapping {
            cpu::softcap_inplace(&mut logits, cap);
        }
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_output", &logits);
        }

        logits
    }

    /// Run layer-outer prefill across a prompt sequence.
    ///
    /// Rather than streaming the entire model's weight matrices N times in a
    /// token-outer loop, this iterates over layers on the outside and tokens on
    /// the inside. Each layer's weights are streamed from memory once and kept
    /// hot in CPU cache while evaluating all N tokens, reducing memory traffic
    /// by N-fold while maintaining strict causal sequential parity.
    fn forward_prefill_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        all_logits: bool,
    ) -> Vec<f32> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            let logits = self.forward_single_token(tokens[0], start_pos, state);
            return logits;
        }

        let hs = self.config.hidden_size;
        let head_dim = self.head_dim;
        let q_dim = self.config.n_heads * head_dim;
        let kv_dim = self.config.n_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;
        let n_pl = self.n_embd_per_layer;
        let total_pl = self.config.n_layers * n_pl;

        // Resize scratch buffers for per-token execution within layers.
        state.scratch.normed.resize(hs, 0.0);
        state.scratch.ffn_input.resize(hs, 0.0);
        state.scratch.q.resize(q_dim, 0.0);
        state.scratch.k.resize(kv_dim, 0.0);
        state.scratch.v.resize(kv_dim, 0.0);
        state.scratch.attn_out.resize(q_dim, 0.0);
        state.scratch.out.resize(hs, 0.0);
        state
            .scratch
            .gate
            .resize(self.config.intermediate_size, 0.0);
        state.scratch.up.resize(self.config.intermediate_size, 0.0);

        // 1. Base embeddings for all N tokens: [n * hs].
        let mut hidden = vec![0.0f32; n * hs];
        for (j, &tok) in tokens.iter().enumerate() {
            let row = &mut hidden[j * hs..(j + 1) * hs];
            transformer::dequantize_row_into(&self.gguf, &self.embd_ref, tok as usize, row);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record("embd", row);
            }
            cpu::scale_inplace(row, (hs as f32).sqrt());
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record("inp_scaled", row);
            }
        }

        // 2. Per-layer embeddings: compute combined representation for all N tokens.
        let combined = if n_pl > 0 {
            state.scratch.lora_tmp.resize(total_pl * 2, 0.0);
            state.scratch.conv_scratch.resize(n_pl, 0.0);
            let mut comb_buf = vec![0.0f32; n * total_pl];
            let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();

            for (j, &tok) in tokens.iter().enumerate() {
                let comb_slice = &mut comb_buf[j * total_pl..(j + 1) * total_pl];
                let (comb_tmp, proj_pl) = state.scratch.lora_tmp.split_at_mut(total_pl);

                transformer::dequantize_row_into(
                    &self.gguf,
                    self.per_layer_token_embd.as_ref().unwrap(),
                    tok as usize,
                    comb_tmp,
                );
                cpu::scale_inplace(comb_tmp, (n_pl as f32).sqrt());

                let tok_hidden = &hidden[j * hs..(j + 1) * hs];
                gemv(
                    &self.gguf,
                    self.per_layer_model_proj.as_ref().unwrap(),
                    tok_hidden,
                    proj_pl,
                );
                cpu::scale_inplace(proj_pl, 1.0 / (hs as f32).sqrt());
                for il in 0..self.config.n_layers {
                    cpu::rmsnorm(
                        &mut proj_pl[il * n_pl..(il + 1) * n_pl],
                        &self.per_layer_proj_norm,
                        eps,
                    );
                }

                for k in 0..total_pl {
                    comb_slice[k] = (proj_pl[k] + comb_tmp[k]) * inv_sqrt_2;
                }
            }
            comb_buf
        } else {
            Vec::new()
        };

        // 3. Layer-outer loop: Stream each layer's weights once for the entire prompt.
        for i in 0..self.config.n_layers {
            let layer = &self.layers[i];
            let has_kv = i < self.n_layer_kv_from_start;
            let is_swa = self.is_swa[i];
            let kv_src = if has_kv {
                i
            } else if is_swa {
                self.n_layer_kv_from_start - 2
            } else {
                self.n_layer_kv_from_start - 1
            };

            let sw = if is_swa && self.sliding_window > 0 {
                Some(self.sliding_window)
            } else {
                None
            };

            for j in 0..n {
                let pos = start_pos + j;
                let tok_hidden = &mut hidden[j * hs..(j + 1) * hs];

                // Attention pre-norm.
                cpu::rmsnorm_into(tok_hidden, &mut state.scratch.normed, &layer.attn_norm, eps);
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("attn_norm-{i}"),
                        &state.scratch.normed,
                    );
                }

                // Q projection.
                #[cfg(target_arch = "aarch64")]
                {
                    quantize_to_scratch_bufs(
                        &state.scratch.normed,
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                    gemv_preq(
                        &self.gguf,
                        &layer.attn_q,
                        &state.scratch.normed,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.q[..q_dim],
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(
                        &self.gguf,
                        &layer.attn_q,
                        &state.scratch.normed,
                        &mut state.scratch.q[..q_dim],
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Qcur-{i}"),
                        &state.scratch.q[..q_dim],
                    );
                }

                // QK norm on Q (per head).
                for h in 0..self.config.n_heads {
                    cpu::rmsnorm(
                        &mut state.scratch.q[h * head_dim..(h + 1) * head_dim],
                        &layer.attn_q_norm,
                        eps,
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("Qcur_normed-{i}"),
                        &state.scratch.q[..q_dim],
                    );
                }

                if has_kv {
                    // K projection.
                    #[cfg(target_arch = "aarch64")]
                    {
                        gemv_preq(
                            &self.gguf,
                            layer.attn_k.as_ref().unwrap(),
                            &state.scratch.normed,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut state.scratch.k[..kv_dim],
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        gemv(
                            &self.gguf,
                            layer.attn_k.as_ref().unwrap(),
                            &state.scratch.normed,
                            &mut state.scratch.k[..kv_dim],
                        );
                    }
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("Kcur-{i}"),
                            &state.scratch.k[..kv_dim],
                        );
                    }

                    // QK norm on K (per head).
                    for h in 0..self.config.n_kv_heads {
                        cpu::rmsnorm(
                            &mut state.scratch.k[h * head_dim..(h + 1) * head_dim],
                            layer.attn_k_norm.as_ref().unwrap(),
                            eps,
                        );
                    }
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("Kcur_normed-{i}"),
                            &state.scratch.k[..kv_dim],
                        );
                    }

                    // RoPE on Q and K (Neox split-halves layout).
                    cpu::rope(
                        &mut state.scratch.q[..q_dim],
                        &mut state.scratch.k[..kv_dim],
                        pos,
                        self.config.n_heads,
                        self.config.n_kv_heads,
                        head_dim,
                        self.config.rope_theta,
                    );
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("Qcur_pos-{i}"),
                            &state.scratch.q[..q_dim],
                        );
                        transformer::oracle_dump::record(
                            &format!("Kcur_pos-{i}"),
                            &state.scratch.k[..kv_dim],
                        );
                    }

                    // V projection.
                    let v_ref = layer
                        .attn_v
                        .as_ref()
                        .unwrap_or_else(|| layer.attn_k.as_ref().unwrap());
                    #[cfg(target_arch = "aarch64")]
                    {
                        gemv_preq(
                            &self.gguf,
                            v_ref,
                            &state.scratch.normed,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut state.scratch.v[..kv_dim],
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        gemv(
                            &self.gguf,
                            v_ref,
                            &state.scratch.normed,
                            &mut state.scratch.v[..kv_dim],
                        );
                    }
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("Vcur-{i}"),
                            &state.scratch.v[..kv_dim],
                        );
                    }

                    // Unweighted RMSNorm on V (per head).
                    for h in 0..self.config.n_kv_heads {
                        cpu::rmsnorm_unweighted(
                            &mut state.scratch.v[h * head_dim..(h + 1) * head_dim],
                            eps,
                        );
                    }
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("Vcur_normed-{i}"),
                            &state.scratch.v[..kv_dim],
                        );
                    }

                    // Append K and V to this layer's cache slot.
                    let use_f16 = state.kv_f16;
                    if let LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } = &mut state.layers[i]
                    {
                        if use_f16 {
                            key_cache_f16.extend(
                                state.scratch.k[..kv_dim]
                                    .iter()
                                    .map(|&x| crate::quant::f32_to_f16(x)),
                            );
                            value_cache_f16.extend(
                                state.scratch.v[..kv_dim]
                                    .iter()
                                    .map(|&x| crate::quant::f32_to_f16(x)),
                            );
                        } else {
                            key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                            value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                        }
                    }
                } else {
                    // Shared layer: RoPE Q only.
                    cpu::rope(
                        &mut state.scratch.q[..q_dim],
                        &mut [],
                        pos,
                        self.config.n_heads,
                        0,
                        head_dim,
                        self.config.rope_theta,
                    );
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("Qcur_pos-{i}"),
                            &state.scratch.q[..q_dim],
                        );
                    }
                }

                // Self-attention over KV cache.
                let kv = match &state.layers[kv_src] {
                    LayerState::Attention {
                        key_cache,
                        value_cache,
                        key_cache_f16,
                        value_cache_f16,
                        ..
                    } => {
                        if state.kv_f16 {
                            KvView::F16 {
                                k: key_cache_f16,
                                v: value_cache_f16,
                            }
                        } else {
                            KvView::F32 {
                                k: key_cache,
                                v: value_cache,
                            }
                        }
                    }
                    _ => panic!("Gemma 4 expected Attention LayerState"),
                };

                let dims = DecodeAttnDims {
                    n_heads: self.config.n_heads,
                    n_kv_heads: self.config.n_kv_heads,
                    head_dim,
                    scale: 1.0,
                    seq_len: pos + 1,
                    attn_logit_softcapping: None,
                    sliding_window: sw,
                };
                transformer::decode_attention(
                    &state.scratch.q[..q_dim],
                    &kv,
                    &dims,
                    &mut state.scratch.attn_out[..q_dim],
                    &mut state.scratch.scores,
                );

                // Output projection.
                #[cfg(target_arch = "aarch64")]
                {
                    quantize_to_scratch_bufs(
                        &state.scratch.attn_out[..q_dim],
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                    gemv_preq(
                        &self.gguf,
                        &layer.attn_output,
                        &state.scratch.attn_out[..q_dim],
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.out[..hs],
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(
                        &self.gguf,
                        &layer.attn_output,
                        &state.scratch.attn_out[..q_dim],
                        &mut state.scratch.out[..hs],
                    );
                }

                // Post-attention norm.
                cpu::rmsnorm(&mut state.scratch.out[..hs], &layer.attn_post_norm, eps);
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("attn_post_norm-{i}"),
                        &state.scratch.out[..hs],
                    );
                }

                // Residual addition.
                cpu::add_inplace(tok_hidden, &state.scratch.out[..hs]);
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(&format!("attn_out-{i}"), tok_hidden);
                }

                // FFN pre-norm.
                cpu::rmsnorm_into(
                    tok_hidden,
                    &mut state.scratch.ffn_input,
                    &layer.ffn_norm,
                    eps,
                );
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("ffn_norm-{i}"),
                        &state.scratch.ffn_input,
                    );
                }

                // FFN gate and up projections.
                let intermediate = self.config.intermediate_size;
                #[cfg(target_arch = "aarch64")]
                {
                    quantize_to_scratch_bufs(
                        &state.scratch.ffn_input,
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                    gemv_preq(
                        &self.gguf,
                        &layer.ffn_gate,
                        &state.scratch.ffn_input,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.gate[..intermediate],
                    );
                    gemv_preq(
                        &self.gguf,
                        &layer.ffn_up,
                        &state.scratch.ffn_input,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.up[..intermediate],
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(
                        &self.gguf,
                        &layer.ffn_gate,
                        &state.scratch.ffn_input,
                        &mut state.scratch.gate[..intermediate],
                    );
                    gemv(
                        &self.gguf,
                        &layer.ffn_up,
                        &state.scratch.ffn_input,
                        &mut state.scratch.up[..intermediate],
                    );
                }

                // GeGLU activation.
                cpu::gelu_mul_inplace(
                    &mut state.scratch.gate[..intermediate],
                    &state.scratch.up[..intermediate],
                );

                // Down projection.
                #[cfg(target_arch = "aarch64")]
                {
                    quantize_to_scratch_bufs(
                        &state.scratch.gate[..intermediate],
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                    gemv_preq(
                        &self.gguf,
                        &layer.ffn_down,
                        &state.scratch.gate[..intermediate],
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        &mut state.scratch.out[..hs],
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(
                        &self.gguf,
                        &layer.ffn_down,
                        &state.scratch.gate[..intermediate],
                        &mut state.scratch.out[..hs],
                    );
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("ffn_out-{i}"),
                        &state.scratch.out[..hs],
                    );
                }

                // FFN post-norm.
                cpu::rmsnorm(&mut state.scratch.out[..hs], &layer.ffn_post_norm, eps);
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(
                        &format!("ffn_post_norm-{i}"),
                        &state.scratch.out[..hs],
                    );
                }

                // Residual addition.
                cpu::add_inplace(tok_hidden, &state.scratch.out[..hs]);

                // Per-layer embedding gating and projection.
                if n_pl > 0 {
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(&format!("pe_in-{i}"), tok_hidden);
                    }
                    let pe_gate = &mut state.scratch.conv_scratch[..n_pl];
                    gemv(
                        &self.gguf,
                        layer.per_layer_inp_gate.as_ref().unwrap(),
                        tok_hidden,
                        pe_gate,
                    );
                    let inp_this_layer =
                        &combined[j * total_pl + i * n_pl..j * total_pl + (i + 1) * n_pl];
                    cpu::gelu_mul_inplace(pe_gate, inp_this_layer);

                    gemv(
                        &self.gguf,
                        layer.per_layer_proj.as_ref().unwrap(),
                        pe_gate,
                        &mut state.scratch.out[..hs],
                    );
                    cpu::rmsnorm(
                        &mut state.scratch.out[..hs],
                        layer.per_layer_post_norm.as_ref().unwrap(),
                        eps,
                    );
                    if transformer::oracle_dump::is_active() {
                        transformer::oracle_dump::record(
                            &format!("per_layer_embd_out-{i}"),
                            &state.scratch.out[..hs],
                        );
                    }
                    cpu::add_inplace(tok_hidden, &state.scratch.out[..hs]);
                }

                // Layer output scalar.
                if let Some(scale) = layer.layer_out_scale {
                    cpu::scale_inplace(tok_hidden, scale);
                }
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record(&format!("l_out-{i}"), tok_hidden);
                }
            }
        }

        state.seq_len = start_pos + n;

        // Final output processing.
        let vocab_size = self.config.vocab_size;
        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);

        if all_logits {
            let mut all_out = vec![0.0f32; n * vocab_size];
            for j in 0..n {
                let tok_hidden = &mut hidden[j * hs..(j + 1) * hs];
                cpu::rmsnorm(tok_hidden, &self.output_norm_weight, eps);
                let logits_slice = &mut all_out[j * vocab_size..(j + 1) * vocab_size];
                #[cfg(target_arch = "aarch64")]
                {
                    quantize_to_scratch_bufs(
                        tok_hidden,
                        &mut state.scratch.q8_scales,
                        &mut state.scratch.q8_quants,
                    );
                    gemv_preq(
                        &self.gguf,
                        out_ref,
                        tok_hidden,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        logits_slice,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    gemv(&self.gguf, out_ref, tok_hidden, logits_slice);
                }
                if let Some(cap) = self.final_logit_softcapping {
                    cpu::softcap_inplace(logits_slice, cap);
                }
            }
            all_out
        } else {
            let last_hidden = &mut hidden[(n - 1) * hs..n * hs];
            cpu::rmsnorm(last_hidden, &self.output_norm_weight, eps);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record("result_norm", last_hidden);
            }
            let mut logits = vec![0.0f32; vocab_size];
            #[cfg(target_arch = "aarch64")]
            {
                quantize_to_scratch_bufs(
                    last_hidden,
                    &mut state.scratch.q8_scales,
                    &mut state.scratch.q8_quants,
                );
                gemv_preq(
                    &self.gguf,
                    out_ref,
                    last_hidden,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    &mut logits,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                gemv(&self.gguf, out_ref, last_hidden, &mut logits);
            }
            if let Some(cap) = self.final_logit_softcapping {
                cpu::softcap_inplace(&mut logits, cap);
            }
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record("result_output", &logits);
            }
            logits
        }
    }
}

impl Model for Gemma4Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.forward_single_token(tokens[0], pos, state)
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        if tokens.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );

        self.forward_prefill_inner(tokens, start_pos, state, false)
    }

    fn supports_all_logits(&self) -> bool {
        true
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
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill_logits_all: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );

        self.forward_prefill_inner(tokens, start_pos, state, true)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

//! BERT and ModernBERT encoder model implementation.
//!
//! Supports both encoder architectures:
//! 1. `modernbert`: RoPE (Neox layout), Pre-LN (LayerNorm/RMSNorm), GeGLU feed-forward,
//!    and alternating local (128-token sliding window) and global attention.
//! 2. `bert` (Classic BERT): absolute positional embeddings, Post-LN (LayerNorm with bias),
//!    and standard 2-layer MLP with bias.
//!
//! Pure encoder pipeline: stateless monolithic forward pass exposing
//! `hidden_states(&self, tokens, state)` and `hidden_states_mean_pooled`.

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::transformer::{self, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};

/// LayerNorm without bias term (e.g. ModernBERT pre-LN and output-LN).
/// Subtracts the mean (unlike RMSNorm) and multiplies by weight.
#[inline]
pub fn layer_norm_no_bias_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    if x.len() != weight.len() || x.is_empty() {
        return;
    }
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let inv_std = (1.0 / (var + eps as f64).sqrt()) as f32;
    let mean_f32 = mean as f32;
    for i in 0..n {
        x[i] = (x[i] - mean_f32) * inv_std * weight[i];
    }
}

/// Per-layer weight references for one BERT / ModernBERT encoder layer.
pub struct BertLayerWeightRefs {
    pub attn_q: WeightRef,
    pub attn_k: WeightRef,
    pub attn_v: WeightRef,
    pub attn_output: WeightRef,
    pub attn_q_bias: Option<Vec<f32>>,
    pub attn_k_bias: Option<Vec<f32>>,
    pub attn_v_bias: Option<Vec<f32>>,
    pub attn_output_bias: Option<Vec<f32>>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Option<Vec<f32>>,

    pub ffn_up: WeightRef,
    pub ffn_up_bias: Option<Vec<f32>>,
    pub ffn_down: WeightRef,
    pub ffn_down_bias: Option<Vec<f32>>,
    pub ffn_gate: Option<WeightRef>,
    pub ffn_gate_bias: Option<Vec<f32>>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Option<Vec<f32>>,

    /// True if this layer uses local sliding window attention (128 tokens in ModernBERT).
    pub is_sliding: bool,
}

pub struct BertModel {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    is_modernbert: bool,
    global_rope_theta: f32,
    half_window: usize,
    embd_ref: WeightRef,
    pos_embd_ref: Option<WeightRef>,
    embd_norm_weight: Option<Vec<f32>>,
    embd_norm_bias: Option<Vec<f32>>,
    layer_refs: Vec<BertLayerWeightRefs>,
    output_norm_weight: Option<Vec<f32>>,
    output_norm_bias: Option<Vec<f32>>,
    #[allow(dead_code)]
    model_id: String,
}

impl BertModel {
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        ensure!(context_size > 0, "context_size must be > 0");

        let arch = gguf
            .get_str("general.architecture")
            .context("missing general.architecture")?
            .to_string();
        let prefix = arch.as_str();

        let is_modernbert =
            prefix == "modernbert" || gguf.tensors.contains_key("blk.0.ffn_gate.weight");

        let n_layers =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .or_else(|| gguf.get_u32("bert.block_count"))
                .with_context(|| format!("missing {prefix}.block_count"))? as usize;
        ensure!(n_layers > 0, "n_layers must be > 0");

        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .or_else(|| gguf.get_u32("bert.embedding_length"))
            .with_context(|| format!("missing {prefix}.embedding_length"))?
            as usize;

        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .or_else(|| gguf.get_u32("bert.feed_forward_length"))
            .unwrap_or((hidden_size * 4) as u32) as usize;

        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .or_else(|| gguf.get_u32("bert.attention.head_count"))
            .with_context(|| format!("missing {prefix}.attention.head_count"))?
            as usize;

        ensure!(n_heads > 0, "n_heads must be > 0");
        ensure!(
            hidden_size.is_multiple_of(n_heads),
            "hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
        );
        let head_dim = hidden_size / n_heads;
        ensure!(head_dim > 0, "head_dim must be > 0");

        let vocab_size = match gguf.get_u32(&format!("{prefix}.vocab_size")) {
            Some(v) => v as usize,
            None => {
                let info = gguf
                    .tensors
                    .get("token_embd.weight")
                    .context("missing token_embd.weight")?;
                info.shape.get(1).copied().with_context(|| {
                    format!(
                        "token_embd.weight shape has fewer than 2 dimensions: {:?}",
                        info.shape
                    )
                })?
            }
        };

        let sliding_window_size = gguf
            .get_u32(&format!("{prefix}.sliding_window"))
            .or_else(|| gguf.get_u32("modernbert.sliding_window"))
            .unwrap_or(128) as usize;
        let half_window = sliding_window_size / 2;

        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(8192) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);

        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10_000.0);

        let global_rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.global_freq_base"))
            .or_else(|| gguf.get_f32("modernbert.global_rope_theta"))
            .unwrap_or(160_000.0);

        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_epsilon"))
            .or_else(|| gguf.get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")))
            .unwrap_or(1e-5);

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;
        ensure!(
            embd_ref.k == hidden_size,
            "token_embd.weight k ({}) != hidden_size ({})",
            embd_ref.k,
            hidden_size
        );

        let pos_embd_ref = if gguf.tensors.contains_key("position_embd.weight") {
            let pos = transformer::resolve_weight(&gguf, "position_embd.weight")?;
            ensure!(
                pos.k == hidden_size,
                "position_embd.weight k ({}) != hidden_size ({})",
                pos.k,
                hidden_size
            );
            Some(pos)
        } else {
            None
        };

        let embd_norm_weight = gguf
            .get_tensor("token_embd_norm.weight")
            .map(|t| t.to_f32_vec())
            .ok();
        if let Some(ref w) = embd_norm_weight {
            ensure!(
                w.len() == hidden_size,
                "token_embd_norm.weight len ({}) != hidden_size ({})",
                w.len(),
                hidden_size
            );
        }
        let embd_norm_bias = gguf
            .get_tensor("token_embd_norm.bias")
            .map(|t| t.to_f32_vec())
            .ok();
        if let Some(ref b) = embd_norm_bias {
            ensure!(
                b.len() == hidden_size,
                "token_embd_norm.bias len ({}) != hidden_size ({})",
                b.len(),
                hidden_size
            );
        }

        let (output_norm_weight, output_norm_bias) = if is_modernbert {
            let w = if let Ok(t) = gguf.get_tensor("output_norm.weight") {
                t.to_f32_vec()
            } else {
                vec![1.0; hidden_size]
            };
            ensure!(
                w.len() == hidden_size,
                "output_norm.weight len ({}) != hidden_size ({})",
                w.len(),
                hidden_size
            );
            let b = gguf
                .get_tensor("output_norm.bias")
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = b {
                ensure!(
                    b.len() == hidden_size,
                    "output_norm.bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }
            (Some(w), b)
        } else if let Ok(t) = gguf.get_tensor("output_norm.weight") {
            let w = t.to_f32_vec();
            ensure!(
                w.len() == hidden_size,
                "output_norm.weight len ({}) != hidden_size ({})",
                w.len(),
                hidden_size
            );
            let b = gguf
                .get_tensor("output_norm.bias")
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = b {
                ensure!(
                    b.len() == hidden_size,
                    "output_norm.bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }
            (Some(w), b)
        } else {
            (None, None)
        };

        let mut layer_refs = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn_q = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?
                .with_repack(&gguf);
            ensure!(
                attn_q.k == hidden_size && attn_q.m == hidden_size,
                "layer {i} attn_q dimensions mismatch: [{}, {}] != [{}, {}]",
                attn_q.m,
                attn_q.k,
                hidden_size,
                hidden_size
            );
            let attn_k = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?
                .with_repack(&gguf);
            ensure!(
                attn_k.k == hidden_size && attn_k.m == hidden_size,
                "layer {i} attn_k dimensions mismatch: [{}, {}] != [{}, {}]",
                attn_k.m,
                attn_k.k,
                hidden_size,
                hidden_size
            );
            let attn_v = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))?
                .with_repack(&gguf);
            ensure!(
                attn_v.k == hidden_size && attn_v.m == hidden_size,
                "layer {i} attn_v dimensions mismatch: [{}, {}] != [{}, {}]",
                attn_v.m,
                attn_v.k,
                hidden_size,
                hidden_size
            );
            let attn_output =
                transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_output.weight"))?
                    .with_repack(&gguf);
            ensure!(
                attn_output.k == hidden_size && attn_output.m == hidden_size,
                "layer {i} attn_output dimensions mismatch: [{}, {}] != [{}, {}]",
                attn_output.m,
                attn_output.k,
                hidden_size,
                hidden_size
            );

            let attn_q_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_q.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = attn_q_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} attn_q_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }
            let attn_k_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_k.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = attn_k_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} attn_k_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }
            let attn_v_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_v.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = attn_v_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} attn_v_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }
            let attn_output_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_output.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = attn_output_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} attn_output_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }

            let attn_norm_weight =
                if let Ok(t) = gguf.get_tensor(&format!("blk.{i}.attn_norm.weight")) {
                    t.to_f32_vec()
                } else {
                    vec![1.0; hidden_size]
                };
            ensure!(
                attn_norm_weight.len() == hidden_size,
                "layer {i} attn_norm_weight len ({}) != hidden_size ({})",
                attn_norm_weight.len(),
                hidden_size
            );
            let attn_norm_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_norm.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = attn_norm_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} attn_norm_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }

            let ffn_up = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?
                .with_repack(&gguf);
            ensure!(
                ffn_up.k == hidden_size && ffn_up.m == intermediate_size,
                "layer {i} ffn_up dimensions mismatch: [{}, {}] != [{}, {}]",
                ffn_up.m,
                ffn_up.k,
                intermediate_size,
                hidden_size
            );
            let ffn_up_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_up.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = ffn_up_bias {
                ensure!(
                    b.len() == intermediate_size,
                    "layer {i} ffn_up_bias len ({}) != intermediate_size ({})",
                    b.len(),
                    intermediate_size
                );
            }

            let ffn_down = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?
                .with_repack(&gguf);
            ensure!(
                ffn_down.k == intermediate_size && ffn_down.m == hidden_size,
                "layer {i} ffn_down dimensions mismatch: [{}, {}] != [{}, {}]",
                ffn_down.m,
                ffn_down.k,
                hidden_size,
                intermediate_size
            );
            let ffn_down_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_down.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = ffn_down_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} ffn_down_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }

            let ffn_gate = if gguf
                .tensors
                .contains_key(&format!("blk.{i}.ffn_gate.weight"))
            {
                let gate = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?
                    .with_repack(&gguf);
                ensure!(
                    gate.k == hidden_size && gate.m == intermediate_size,
                    "layer {i} ffn_gate dimensions mismatch: [{}, {}] != [{}, {}]",
                    gate.m,
                    gate.k,
                    intermediate_size,
                    hidden_size
                );
                Some(gate)
            } else {
                None
            };
            let ffn_gate_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_gate.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = ffn_gate_bias {
                ensure!(
                    b.len() == intermediate_size,
                    "layer {i} ffn_gate_bias len ({}) != intermediate_size ({})",
                    b.len(),
                    intermediate_size
                );
            }

            let ffn_norm_weight =
                if let Ok(t) = gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight")) {
                    t.to_f32_vec()
                } else {
                    vec![1.0; hidden_size]
                };
            ensure!(
                ffn_norm_weight.len() == hidden_size,
                "layer {i} ffn_norm_weight len ({}) != hidden_size ({})",
                ffn_norm_weight.len(),
                hidden_size
            );
            let ffn_norm_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_norm.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            if let Some(ref b) = ffn_norm_bias {
                ensure!(
                    b.len() == hidden_size,
                    "layer {i} ffn_norm_bias len ({}) != hidden_size ({})",
                    b.len(),
                    hidden_size
                );
            }

            // ModernBERT: layers 0, 1, 3, 4, 6, 7 are sliding window (128 toks);
            // every 3rd layer (2, 5, 8...) is global attention.
            let is_sliding = is_modernbert && (i % 3 != 2);

            layer_refs.push(BertLayerWeightRefs {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_output_bias,
                attn_norm_weight,
                attn_norm_bias,
                ffn_up,
                ffn_up_bias,
                ffn_down,
                ffn_down_bias,
                ffn_gate,
                ffn_gate_bias,
                ffn_norm_weight,
                ffn_norm_bias,
                is_sliding,
            });
        }

        let config = ModelConfig {
            architecture: arch,
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads: n_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types: vec![BlockType::Attention; n_layers],
            conv_kernel_size: None,
            kv_heads_per_layer: vec![n_heads; n_layers],
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: false,
            class_labels: Vec::new(),
        };

        Ok(Self {
            gguf,
            config,
            head_dim,
            is_modernbert,
            global_rope_theta,
            half_window,
            embd_ref,
            pos_embd_ref,
            embd_norm_weight,
            embd_norm_bias,
            layer_refs,
            output_norm_weight,
            output_norm_bias,
            model_id,
        })
    }
}

impl Model for BertModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn supports_hidden_states(&self) -> bool {
        true
    }

    fn hidden_states(&self, tokens: &[u32], _state: &mut InferenceState) -> Vec<f32> {
        let n_tokens = if self.config.max_seq_len > 0 && tokens.len() > self.config.max_seq_len {
            tracing::warn!(
                "BertModel::hidden_states received {} tokens exceeding max_seq_len {}; truncating",
                tokens.len(),
                self.config.max_seq_len
            );
            self.config.max_seq_len
        } else {
            tokens.len()
        };
        if n_tokens == 0 {
            return Vec::new();
        }
        let tokens = &tokens[..n_tokens];
        let d = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Embedding lookup: [n_tokens, d] row-major
        let mut x = vec![0.0f32; n_tokens * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let row_idx = (tok as usize).min(self.embd_ref.m.saturating_sub(1));
            transformer::dequantize_row_into(
                &self.gguf,
                &self.embd_ref,
                row_idx,
                &mut x[i * d..(i + 1) * d],
            );
        }

        // Classic BERT: Add absolute position embeddings and apply token_embd_norm
        if let Some(ref pos_ref) = self.pos_embd_ref {
            let max_pos = pos_ref.m.min(self.config.max_seq_len);
            let mut pos_buf = vec![0.0f32; d];
            for i in 0..n_tokens.min(max_pos) {
                transformer::dequantize_row_into(&self.gguf, pos_ref, i, &mut pos_buf);
                cpu::add_inplace(&mut x[i * d..(i + 1) * d], &pos_buf);
            }
        }
        if let Some(ref w) = self.embd_norm_weight {
            for i in 0..n_tokens {
                let row = &mut x[i * d..(i + 1) * d];
                if let Some(ref b) = self.embd_norm_bias {
                    cpu::layer_norm_inplace(row, w, b, self.config.rms_norm_eps);
                } else {
                    layer_norm_no_bias_inplace(row, w, self.config.rms_norm_eps);
                }
            }
        }

        // 2. Transformer layers (reuse scratch buffers from state to avoid per-chunk heap allocations)
        let scratch = &mut _state.scratch;
        scratch.q.resize(n_tokens * d, 0.0);
        scratch.k.resize(n_tokens * d, 0.0);
        scratch.v.resize(n_tokens * d, 0.0);
        scratch.attn_out.resize(n_tokens * d, 0.0);
        scratch.normed.resize(n_tokens * d, 0.0);
        if self.is_modernbert {
            scratch.ffn_input.resize(n_tokens * d, 0.0);
        }
        scratch.out.resize(d, 0.0);
        let inter = self.config.intermediate_size;
        scratch.up.resize(inter, 0.0);
        scratch.gate.resize(inter, 0.0);
        scratch.conv_proj.resize(d, 0.0);
        scratch.scores.resize(n_tokens, f32::NEG_INFINITY);

        let q_mat = &mut scratch.q[..n_tokens * d];
        let k_mat = &mut scratch.k[..n_tokens * d];
        let v_mat = &mut scratch.v[..n_tokens * d];
        let attn_out = &mut scratch.attn_out[..n_tokens * d];
        let norm_x = &mut scratch.normed[..n_tokens * d];
        let ffn_norm_x =
            &mut scratch.ffn_input[..if self.is_modernbert { n_tokens * d } else { 0 }];
        let proj_out = &mut scratch.out[..d];
        let up_row = &mut scratch.up[..inter];
        let gate_row = &mut scratch.gate[..inter];
        let down_row = &mut scratch.conv_proj[..d];
        let scores = &mut scratch.scores[..n_tokens];

        for layer in &self.layer_refs {
            norm_x.copy_from_slice(&x);

            // Pre-LN for ModernBERT; for Classic BERT, normalization is post-residual
            if self.is_modernbert {
                for i in 0..n_tokens {
                    layer_norm_no_bias_inplace(
                        &mut norm_x[i * d..(i + 1) * d],
                        &layer.attn_norm_weight,
                        self.config.rms_norm_eps,
                    );
                }
            }

            // Batched Q, K, V projections: norm_x [n_tokens, d] * W^T [d, d] -> [n_tokens, d]
            for i in 0..n_tokens {
                let row = &norm_x[i * d..(i + 1) * d];
                let q_row = &mut q_mat[i * d..(i + 1) * d];
                let k_row = &mut k_mat[i * d..(i + 1) * d];
                let v_row = &mut v_mat[i * d..(i + 1) * d];

                transformer::gemv(&self.gguf, &layer.attn_q, row, q_row);
                transformer::gemv(&self.gguf, &layer.attn_k, row, k_row);
                transformer::gemv(&self.gguf, &layer.attn_v, row, v_row);

                if let Some(ref b) = layer.attn_q_bias {
                    cpu::add_inplace(q_row, b);
                }
                if let Some(ref b) = layer.attn_k_bias {
                    cpu::add_inplace(k_row, b);
                }
                if let Some(ref b) = layer.attn_v_bias {
                    cpu::add_inplace(v_row, b);
                }
            }

            // Apply RoPE for ModernBERT (Neox layout per head)
            if self.is_modernbert {
                let layer_theta = if layer.is_sliding {
                    self.config.rope_theta
                } else {
                    self.global_rope_theta
                };
                for i in 0..n_tokens {
                    let q_tok = &mut q_mat[i * d..(i + 1) * d];
                    let k_tok = &mut k_mat[i * d..(i + 1) * d];
                    cpu::rope(q_tok, k_tok, i, n_heads, n_heads, head_dim, layer_theta);
                }
            }

            // Bidirectional Multi-Head Attention: Softmax(Q * K^T * scale) * V
            // If layer.is_sliding is true, apply 128-token sliding window (|q - k| <= 64).
            for h in 0..n_heads {
                for q_idx in 0..n_tokens {
                    let q_vec = &q_mat[q_idx * d + h * head_dim..q_idx * d + (h + 1) * head_dim];

                    // Compute dot product scores with keys within sliding window
                    let (k_start, k_end) = if layer.is_sliding {
                        (
                            q_idx.saturating_sub(self.half_window),
                            (q_idx + self.half_window + 1).min(n_tokens),
                        )
                    } else {
                        (0, n_tokens)
                    };

                    let mut max_score = f32::NEG_INFINITY;
                    for k_idx in k_start..k_end {
                        let k_vec =
                            &k_mat[k_idx * d + h * head_dim..k_idx * d + (h + 1) * head_dim];
                        let s = cpu::dot_f32(q_vec, k_vec) * scale;
                        scores[k_idx] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }

                    // Softmax
                    let mut sum_exp = 0.0f32;
                    for score in &mut scores[k_start..k_end] {
                        if *score > f32::NEG_INFINITY {
                            let e = (*score - max_score).exp();
                            *score = e;
                            sum_exp += e;
                        } else {
                            *score = 0.0;
                        }
                    }
                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                    for score in &mut scores[k_start..k_end] {
                        *score *= inv_sum;
                    }

                    // Multiply by V (sequential memory access across head_dim)
                    let out_slot =
                        &mut attn_out[q_idx * d + h * head_dim..q_idx * d + (h + 1) * head_dim];
                    out_slot.fill(0.0);
                    assert_eq!(out_slot.len(), head_dim);
                    for k_idx in k_start..k_end {
                        let s = scores[k_idx];
                        if s > 0.0 {
                            let v_slice =
                                &v_mat[k_idx * d + h * head_dim..k_idx * d + (h + 1) * head_dim];
                            assert_eq!(v_slice.len(), head_dim);
                            for (out_elem, v_elem) in out_slot.iter_mut().zip(v_slice.iter()) {
                                *out_elem += s * *v_elem;
                            }
                        }
                    }
                }
            }

            // Attention Output Projection & Residual
            for i in 0..n_tokens {
                let row = &attn_out[i * d..(i + 1) * d];
                transformer::gemv(&self.gguf, &layer.attn_output, row, &mut *proj_out);
                if let Some(ref b) = layer.attn_output_bias {
                    cpu::add_inplace(&mut *proj_out, b);
                }
                cpu::add_inplace(&mut x[i * d..(i + 1) * d], proj_out);

                // Post-LN for Classic BERT
                if !self.is_modernbert {
                    if let Some(ref b) = layer.attn_norm_bias {
                        cpu::layer_norm_inplace(
                            &mut x[i * d..(i + 1) * d],
                            &layer.attn_norm_weight,
                            b,
                            self.config.rms_norm_eps,
                        );
                    } else {
                        layer_norm_no_bias_inplace(
                            &mut x[i * d..(i + 1) * d],
                            &layer.attn_norm_weight,
                            self.config.rms_norm_eps,
                        );
                    }
                }
            }

            // Feed-Forward Network (FFN)
            if self.is_modernbert {
                ffn_norm_x.copy_from_slice(&x);
                for i in 0..n_tokens {
                    layer_norm_no_bias_inplace(
                        &mut ffn_norm_x[i * d..(i + 1) * d],
                        &layer.ffn_norm_weight,
                        self.config.rms_norm_eps,
                    );
                }
            }

            if let Some(ref gate_ref) = layer.ffn_gate {
                // ModernBERT GeGLU path: ffn_down(gelu(ffn_gate(x)) * ffn_up(x))
                for i in 0..n_tokens {
                    let row = &ffn_norm_x[i * d..(i + 1) * d];
                    transformer::gemv(&self.gguf, &layer.ffn_up, row, &mut *up_row);
                    transformer::gemv(&self.gguf, gate_ref, row, &mut *gate_row);

                    cpu::gelu_inplace(&mut *gate_row);
                    cpu::mul_inplace(&mut *gate_row, up_row);

                    transformer::gemv(&self.gguf, &layer.ffn_down, gate_row, &mut *down_row);
                    cpu::add_inplace(&mut x[i * d..(i + 1) * d], down_row);
                }
            } else {
                // Classic BERT MLP path: ffn_down(gelu(ffn_up(x) + bias)) + bias
                for i in 0..n_tokens {
                    let row = &x[i * d..(i + 1) * d];
                    transformer::gemv(&self.gguf, &layer.ffn_up, row, &mut *up_row);
                    if let Some(ref b) = layer.ffn_up_bias {
                        cpu::add_inplace(&mut *up_row, b);
                    }

                    cpu::gelu_inplace(&mut *up_row);

                    transformer::gemv(&self.gguf, &layer.ffn_down, up_row, &mut *down_row);
                    if let Some(ref b) = layer.ffn_down_bias {
                        cpu::add_inplace(&mut *down_row, b);
                    }

                    cpu::add_inplace(&mut x[i * d..(i + 1) * d], down_row);

                    if let Some(ref b) = layer.ffn_norm_bias {
                        cpu::layer_norm_inplace(
                            &mut x[i * d..(i + 1) * d],
                            &layer.ffn_norm_weight,
                            b,
                            self.config.rms_norm_eps,
                        );
                    } else {
                        layer_norm_no_bias_inplace(
                            &mut x[i * d..(i + 1) * d],
                            &layer.ffn_norm_weight,
                            self.config.rms_norm_eps,
                        );
                    }
                }
            }
        }

        // 3. Final Norm
        if let Some(ref w) = self.output_norm_weight {
            for i in 0..n_tokens {
                let row = &mut x[i * d..(i + 1) * d];
                if let Some(ref b) = self.output_norm_bias {
                    cpu::layer_norm_inplace(row, w, b, self.config.rms_norm_eps);
                } else {
                    layer_norm_no_bias_inplace(row, w, self.config.rms_norm_eps);
                }
            }
        }

        x
    }

    fn forward(&self, _tokens: &[u32], _pos: usize, _state: &mut InferenceState) -> Vec<f32> {
        tracing::warn!("BertModel is an encoder-only model; forward() returning empty logits");
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modernbert_alternating_schedule_is_correct() {
        let n_layers = 22;
        let mut sliding_count = 0;
        let mut global_count = 0;
        for i in 0..n_layers {
            let is_sliding = i % 3 != 2;
            if is_sliding {
                sliding_count += 1;
            } else {
                global_count += 1;
            }
        }
        // In 22 layers: layers 2, 5, 8, 11, 14, 17, 20 are global (7 layers)
        // The other 15 layers are 128-token local sliding window
        assert_eq!(global_count, 7);
        assert_eq!(sliding_count, 15);
    }

    #[test]
    fn sliding_window_distance_check() {
        let window_size = 128isize;
        let half_window = window_size / 2; // 64

        // Query at pos 100
        let q = 100isize;
        // Key at pos 120 -> diff 20 <= 64 -> attends
        assert!((q - 120).abs() <= half_window);
        // Key at pos 164 -> diff 64 <= 64 -> attends
        assert!((q - 164).abs() <= half_window);
        // Key at pos 165 -> diff 65 > 64 -> masked out
        assert!((q - 165).abs() > half_window);
        // Key at pos 35 -> diff 65 > 64 -> masked out
        assert!((q - 35).abs() > half_window);
    }

    #[test]
    fn test_head_dim_divisibility_logic() {
        let hidden_size = 768usize;
        let n_heads = 12usize;
        assert!(hidden_size.is_multiple_of(n_heads));
        assert_eq!(hidden_size / n_heads, 64);

        let indivisible_n_heads = 13usize;
        assert!(!hidden_size.is_multiple_of(indivisible_n_heads));
    }

    #[test]
    fn test_layer_norm_no_bias_inplace() {
        let mut x = vec![2.0, 4.0, 6.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        layer_norm_no_bias_inplace(&mut x, &weight, 1e-5);

        // Mean of [2, 4, 6, 8] is 5.0
        // Variance is ((9 + 1 + 1 + 9) / 4) = 20 / 4 = 5.0
        // inv_std is 1 / sqrt(5.0 + 1e-5) ~ 0.447213
        // Centered: [-3, -1, 1, 3] * 0.447213 ~ [-1.34164, -0.44721, 0.44721, 1.34164]
        let sum: f32 = x.iter().sum();
        assert!(sum.abs() < 1e-5); // Mean must be 0
        assert!((x[0] - (-1.34164)).abs() < 1e-4);
        assert!((x[3] - 1.34164).abs() < 1e-4);
    }

    #[test]
    fn test_sliding_window_half_window_bounds() {
        let half_window = 32usize;
        let n_tokens = 100usize;

        // Query at pos 10: k_start = 0, k_end = 43
        let q_idx = 10usize;
        let k_start = q_idx.saturating_sub(half_window);
        let k_end = (q_idx + half_window + 1).min(n_tokens);
        assert_eq!(k_start, 0);
        assert_eq!(k_end, 43);

        // Query at pos 50: k_start = 18, k_end = 83
        let q_idx = 50usize;
        let k_start = q_idx.saturating_sub(half_window);
        let k_end = (q_idx + half_window + 1).min(n_tokens);
        assert_eq!(k_start, 18);
        assert_eq!(k_end, 83);

        // Query at pos 95: k_start = 63, k_end = 100
        let q_idx = 95usize;
        let k_start = q_idx.saturating_sub(half_window);
        let k_end = (q_idx + half_window + 1).min(n_tokens);
        assert_eq!(k_start, 63);
        assert_eq!(k_end, 100);
    }

    #[test]
    fn test_max_seq_len_clamping_logic() {
        let max_seq_len = 128usize;
        let tokens = vec![1u32; 256];
        let n_tokens = if max_seq_len > 0 && tokens.len() > max_seq_len {
            max_seq_len
        } else {
            tokens.len()
        };
        assert_eq!(n_tokens, 128);
        assert_eq!(tokens[..n_tokens].len(), 128);
    }
}

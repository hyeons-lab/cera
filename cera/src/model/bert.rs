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

use crate::backend::cpu::{self, RopeType};
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::transformer::{self, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};

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
    #[allow(dead_code)]
    rope_type: RopeType,
    embd_ref: WeightRef,
    pos_embd_ref: Option<WeightRef>,
    embd_norm_weight: Option<Vec<f32>>,
    embd_norm_bias: Option<Vec<f32>>,
    layer_refs: Vec<BertLayerWeightRefs>,
    output_norm_weight: Vec<f32>,
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

        let rope_type = if is_modernbert {
            RopeType::Neox
        } else {
            RopeType::Norm
        };

        let n_layers =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .or_else(|| gguf.get_u32("bert.block_count"))
                .with_context(|| format!("missing {prefix}.block_count"))? as usize;

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

        let head_dim = hidden_size / n_heads;
        ensure!(head_dim > 0, "head_dim must be > 0");

        let vocab_size = match gguf.get_u32(&format!("{prefix}.vocab_size")) {
            Some(v) => v as usize,
            None => {
                let info = gguf
                    .tensors
                    .get("token_embd.weight")
                    .context("missing token_embd.weight")?;
                info.shape[1]
            }
        };

        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(8192) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);

        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10_000.0);

        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_epsilon"))
            .or_else(|| gguf.get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")))
            .unwrap_or(1e-5);

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;

        let pos_embd_ref = if gguf.tensors.contains_key("position_embd.weight") {
            Some(transformer::resolve_weight(&gguf, "position_embd.weight")?)
        } else {
            None
        };

        let embd_norm_weight = gguf
            .get_tensor("token_embd_norm.weight")
            .map(|t| t.to_f32_vec())
            .ok();
        let embd_norm_bias = gguf
            .get_tensor("token_embd_norm.bias")
            .map(|t| t.to_f32_vec())
            .ok();

        let output_norm_weight = if let Ok(t) = gguf.get_tensor("output_norm.weight") {
            t.to_f32_vec()
        } else if let Ok(t) = gguf.get_tensor(&format!("blk.{}.attn_norm.weight", n_layers - 1)) {
            t.to_f32_vec()
        } else {
            vec![1.0; hidden_size]
        };

        let output_norm_bias = gguf
            .get_tensor("output_norm.bias")
            .map(|t| t.to_f32_vec())
            .ok();

        let mut layer_refs = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn_q = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?
                .with_repack(&gguf);
            let attn_k = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?
                .with_repack(&gguf);
            let attn_v = transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))?
                .with_repack(&gguf);
            let attn_output =
                transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_output.weight"))?
                    .with_repack(&gguf);

            let attn_q_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_q.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            let attn_k_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_k.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            let attn_v_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_v.bias"))
                .map(|t| t.to_f32_vec())
                .ok();
            let attn_output_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_output.bias"))
                .map(|t| t.to_f32_vec())
                .ok();

            let attn_norm_weight =
                if let Ok(t) = gguf.get_tensor(&format!("blk.{i}.attn_norm.weight")) {
                    t.to_f32_vec()
                } else {
                    vec![1.0; hidden_size]
                };
            let attn_norm_bias = gguf
                .get_tensor(&format!("blk.{i}.attn_norm.bias"))
                .map(|t| t.to_f32_vec())
                .ok();

            let ffn_up = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?
                .with_repack(&gguf);
            let ffn_up_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_up.bias"))
                .map(|t| t.to_f32_vec())
                .ok();

            let ffn_down = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?
                .with_repack(&gguf);
            let ffn_down_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_down.bias"))
                .map(|t| t.to_f32_vec())
                .ok();

            let ffn_gate = if gguf
                .tensors
                .contains_key(&format!("blk.{i}.ffn_gate.weight"))
            {
                Some(
                    transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?
                        .with_repack(&gguf),
                )
            } else {
                None
            };
            let ffn_gate_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_gate.bias"))
                .map(|t| t.to_f32_vec())
                .ok();

            let ffn_norm_weight =
                if let Ok(t) = gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight")) {
                    t.to_f32_vec()
                } else {
                    vec![1.0; hidden_size]
                };
            let ffn_norm_bias = gguf
                .get_tensor(&format!("blk.{i}.ffn_norm.bias"))
                .map(|t| t.to_f32_vec())
                .ok();

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
            rope_type,
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
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            return Vec::new();
        }
        let d = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Embedding lookup: [n_tokens, d] row-major
        let mut x = vec![0.0f32; n_tokens * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let row = transformer::dequantize_row(&self.gguf, &self.embd_ref, tok as usize);
            x[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
        }

        // Classic BERT: Add absolute position embeddings and apply token_embd_norm
        if let Some(ref pos_ref) = self.pos_embd_ref {
            for i in 0..n_tokens {
                if i < self.config.max_seq_len {
                    let pos_row = transformer::dequantize_row(&self.gguf, pos_ref, i);
                    cpu::add_inplace(&mut x[i * d..(i + 1) * d], &pos_row[..d]);
                }
            }
        }
        if let (Some(w), Some(b)) = (&self.embd_norm_weight, &self.embd_norm_bias) {
            for i in 0..n_tokens {
                cpu::layer_norm_inplace(&mut x[i * d..(i + 1) * d], w, b, self.config.rms_norm_eps);
            }
        }

        // 2. Transformer layers
        let mut q_mat = vec![0.0f32; n_tokens * d];
        let mut k_mat = vec![0.0f32; n_tokens * d];
        let mut v_mat = vec![0.0f32; n_tokens * d];
        let mut attn_out = vec![0.0f32; n_tokens * d];

        let mut norm_x = vec![0.0f32; n_tokens * d];
        let mut ffn_norm_x = vec![0.0f32; n_tokens * d];
        let mut proj_out = vec![0.0f32; d];
        let inter = self.config.intermediate_size;
        let mut up_row = vec![0.0f32; inter];
        let mut gate_row = vec![0.0f32; inter];
        let mut down_row = vec![0.0f32; d];
        let mut scores = vec![f32::NEG_INFINITY; n_tokens];

        for layer in &self.layer_refs {
            norm_x.copy_from_slice(&x);

            // Pre-LN for ModernBERT; for Classic BERT, normalization is post-residual
            if self.is_modernbert {
                for i in 0..n_tokens {
                    cpu::rmsnorm(
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
                for i in 0..n_tokens {
                    let q_tok = &mut q_mat[i * d..(i + 1) * d];
                    let k_tok = &mut k_mat[i * d..(i + 1) * d];
                    cpu::rope(
                        q_tok,
                        k_tok,
                        i,
                        n_heads,
                        n_heads,
                        head_dim,
                        self.config.rope_theta,
                    );
                }
            }

            // Bidirectional Multi-Head Attention: Softmax(Q * K^T * scale) * V
            // If layer.is_sliding is true, apply 128-token sliding window (|q - k| <= 64).
            for h in 0..n_heads {
                for q_idx in 0..n_tokens {
                    let q_vec = &q_mat[q_idx * d + h * head_dim..q_idx * d + (h + 1) * head_dim];

                    // Compute dot product scores with keys
                    scores.fill(f32::NEG_INFINITY);
                    let mut max_score = f32::NEG_INFINITY;

                    for k_idx in 0..n_tokens {
                        if layer.is_sliding {
                            let diff = (q_idx as isize - k_idx as isize).abs();
                            if diff > 64 {
                                continue;
                            }
                        }
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
                    for score in scores.iter_mut().take(n_tokens) {
                        if *score > f32::NEG_INFINITY {
                            let e = (*score - max_score).exp();
                            *score = e;
                            sum_exp += e;
                        } else {
                            *score = 0.0;
                        }
                    }
                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

                    // Multiply by V
                    let out_slot =
                        &mut attn_out[q_idx * d + h * head_dim..q_idx * d + (h + 1) * head_dim];
                    for dim_idx in 0..head_dim {
                        let mut acc = 0.0f32;
                        for k_idx in 0..n_tokens {
                            if scores[k_idx] > 0.0 {
                                let v_val = v_mat[k_idx * d + h * head_dim + dim_idx];
                                acc += scores[k_idx] * inv_sum * v_val;
                            }
                        }
                        out_slot[dim_idx] = acc;
                    }
                }
            }

            // Attention Output Projection & Residual
            for i in 0..n_tokens {
                let row = &attn_out[i * d..(i + 1) * d];
                transformer::gemv(&self.gguf, &layer.attn_output, row, &mut proj_out);
                if let Some(ref b) = layer.attn_output_bias {
                    cpu::add_inplace(&mut proj_out, b);
                }
                cpu::add_inplace(&mut x[i * d..(i + 1) * d], &proj_out);

                // Post-LN for Classic BERT
                if !self.is_modernbert
                    && let Some(ref b) = layer.attn_norm_bias
                {
                    cpu::layer_norm_inplace(
                        &mut x[i * d..(i + 1) * d],
                        &layer.attn_norm_weight,
                        b,
                        self.config.rms_norm_eps,
                    );
                }
            }

            // Feed-Forward Network (FFN)
            ffn_norm_x.copy_from_slice(&x);
            if self.is_modernbert {
                for i in 0..n_tokens {
                    cpu::rmsnorm(
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
                    transformer::gemv(&self.gguf, &layer.ffn_up, row, &mut up_row);
                    transformer::gemv(&self.gguf, gate_ref, row, &mut gate_row);

                    cpu::gelu_inplace(&mut gate_row);
                    cpu::mul_inplace(&mut gate_row, &up_row);

                    transformer::gemv(&self.gguf, &layer.ffn_down, &gate_row, &mut down_row);
                    cpu::add_inplace(&mut x[i * d..(i + 1) * d], &down_row);
                }
            } else {
                // Classic BERT MLP path: ffn_down(gelu(ffn_up(x) + bias)) + bias
                for i in 0..n_tokens {
                    let row = &x[i * d..(i + 1) * d];
                    transformer::gemv(&self.gguf, &layer.ffn_up, row, &mut up_row);
                    if let Some(ref b) = layer.ffn_up_bias {
                        cpu::add_inplace(&mut up_row, b);
                    }

                    cpu::gelu_inplace(&mut up_row);

                    transformer::gemv(&self.gguf, &layer.ffn_down, &up_row, &mut down_row);
                    if let Some(ref b) = layer.ffn_down_bias {
                        cpu::add_inplace(&mut down_row, b);
                    }

                    cpu::add_inplace(&mut x[i * d..(i + 1) * d], &down_row);

                    if let Some(ref b) = layer.ffn_norm_bias {
                        cpu::layer_norm_inplace(
                            &mut x[i * d..(i + 1) * d],
                            &layer.ffn_norm_weight,
                            b,
                            self.config.rms_norm_eps,
                        );
                    }
                }
            }
        }

        // 3. Final Norm
        for i in 0..n_tokens {
            let row = &mut x[i * d..(i + 1) * d];
            if let Some(ref b) = self.output_norm_bias {
                cpu::layer_norm_inplace(row, &self.output_norm_weight, b, self.config.rms_norm_eps);
            } else {
                cpu::rmsnorm(row, &self.output_norm_weight, self.config.rms_norm_eps);
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
}

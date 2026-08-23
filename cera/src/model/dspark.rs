//! DSpark / DFlash speculative decoding draft model implementation.
//!
//! DSpark is a compact draft model sidecar architecture (e.g. 5 layers) designed to
//! predict multi-token speculative drafts (`block_size` up to 9 tokens) with high acceptance
//! rates. It shares the vocabulary, embedding table (`token_embd.weight`), and LM head
//! with the base target model (e.g. LFM2.5).

use anyhow::{Context, Result, anyhow, ensure};
use std::sync::Arc;

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::transformer::{
    self, AttnDims, AttnExtras, AttnWeights, FfnWeights, WeightRef, forward_attn_block,
    forward_ffn_block, gemv,
};
use crate::model::{BlockType, ModelConfig, ScalarMultipliers};
use crate::spec::Drafter;

/// Configuration parameters for DSpark draft sidecar models.
#[derive(Debug, Clone)]
pub struct DSparkConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl DSparkConfig {
    /// Parse DSpark configuration from a GGUF file.
    pub fn from_gguf(gguf: &GgufFile, base_vocab: usize, base_hidden: usize) -> Result<Self> {
        let arch = gguf.get_str("general.architecture").unwrap_or("dflash");

        let pfx = |key: &str| -> String { format!("{arch}.{key}") };

        let hidden_size = gguf
            .get_u32(&pfx("embedding_length"))
            .or_else(|| gguf.get_u32("dflash.embedding_length"))
            .or_else(|| gguf.get_u32("dspark.embedding_length"))
            .or_else(|| gguf.get_u32(&pfx("hidden_size")))
            .or_else(|| gguf.get_u32("dspark.hidden_size"))
            .or_else(|| gguf.get_u32("dflash.hidden_size"))
            .map(|v| v as usize)
            .unwrap_or(base_hidden);

        let num_layers = gguf
            .get_u32(&pfx("block_count"))
            .or_else(|| gguf.get_u32("dflash.block_count"))
            .or_else(|| gguf.get_u32("dspark.block_count"))
            .or_else(|| gguf.get_u32(&pfx("num_layers")))
            .or_else(|| gguf.get_u32("dspark.num_layers"))
            .or_else(|| gguf.get_u32("dflash.num_layers"))
            .map(|v| v as usize)
            .unwrap_or(5);

        let num_heads = gguf
            .get_u32(&pfx("attention.head_count"))
            .or_else(|| gguf.get_u32("dflash.attention.head_count"))
            .or_else(|| gguf.get_u32("dspark.attention.head_count"))
            .or_else(|| gguf.get_u32(&pfx("num_heads")))
            .or_else(|| gguf.get_u32("dspark.num_heads"))
            .or_else(|| gguf.get_u32("dflash.num_heads"))
            .map(|v| v as usize)
            .unwrap_or(16);

        let num_kv_heads = gguf
            .get_u32(&pfx("attention.head_count_kv"))
            .or_else(|| gguf.get_u32("dflash.attention.head_count_kv"))
            .or_else(|| gguf.get_u32("dspark.attention.head_count_kv"))
            .or_else(|| gguf.get_u32(&pfx("num_kv_heads")))
            .or_else(|| gguf.get_u32("dspark.num_kv_heads"))
            .or_else(|| gguf.get_u32("dflash.num_kv_heads"))
            .map(|v| v as usize)
            .unwrap_or(8);

        let head_dim = gguf
            .get_u32(&pfx("attention.key_length"))
            .or_else(|| gguf.get_u32("dflash.attention.key_length"))
            .or_else(|| gguf.get_u32("dspark.attention.key_length"))
            .or_else(|| gguf.get_u32(&pfx("head_dim")))
            .or_else(|| gguf.get_u32("dspark.head_dim"))
            .or_else(|| gguf.get_u32("dflash.head_dim"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_heads.max(1));

        let intermediate_size = gguf
            .get_u32(&pfx("feed_forward_length"))
            .or_else(|| gguf.get_u32("dflash.feed_forward_length"))
            .or_else(|| gguf.get_u32("dspark.feed_forward_length"))
            .or_else(|| gguf.get_u32(&pfx("intermediate_size")))
            .or_else(|| gguf.get_u32("dspark.intermediate_size"))
            .or_else(|| gguf.get_u32("dflash.intermediate_size"))
            .map(|v| v as usize)
            .unwrap_or_else(|| (hidden_size * 8) / 3);

        let vocab_size = gguf
            .get_u32(&pfx("vocab_size"))
            .or_else(|| gguf.get_u32("dflash.vocab_size"))
            .or_else(|| gguf.get_u32("dspark.vocab_size"))
            .map(|v| v as usize)
            .unwrap_or(base_vocab);

        if vocab_size != base_vocab {
            anyhow::bail!(
                "draft model vocab_size ({vocab_size}) does not match base model vocab_size ({base_vocab})"
            );
        }

        let block_size = gguf
            .get_u32(&pfx("block_size"))
            .or_else(|| gguf.get_u32("dspark.block_size"))
            .or_else(|| gguf.get_u32("dflash.block_size"))
            .map(|v| v as usize)
            .unwrap_or(9);

        let markov_rank = gguf
            .get_u32(&pfx("markov_rank"))
            .or_else(|| gguf.get_u32("dspark.markov_rank"))
            .or_else(|| gguf.get_u32("dflash.markov_rank"))
            .map(|v| v as usize)
            .unwrap_or(256);

        let rms_norm_eps = gguf
            .get_f32(&pfx("attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_f32("dflash.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_f32("dspark.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_f32("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let rope_theta = gguf
            .get_f32(&pfx("rope.freq_base"))
            .or_else(|| gguf.get_f32("dflash.rope.freq_base"))
            .or_else(|| gguf.get_f32("dspark.rope.freq_base"))
            .or_else(|| gguf.get_f32("rope.freq_base"))
            .unwrap_or(500000.0);

        ensure!(hidden_size > 0, "DSpark hidden_size must be positive");
        ensure!(
            hidden_size.is_multiple_of(32),
            "DSpark hidden_size ({hidden_size}) must be divisible by 32 for SIMD alignment",
        );
        ensure!(num_layers > 0, "DSpark num_layers must be positive");
        ensure!(num_heads > 0, "DSpark num_heads must be positive");
        ensure!(num_kv_heads > 0, "DSpark num_kv_heads must be positive");
        ensure!(
            num_heads.is_multiple_of(num_kv_heads),
            "DSpark num_heads ({num_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
        );
        ensure!(head_dim > 0, "DSpark head_dim must be positive");
        ensure!(
            intermediate_size > 0,
            "DSpark intermediate_size must be positive"
        );
        ensure!(vocab_size > 0, "DSpark vocab_size must be positive");
        ensure!(block_size > 0, "DSpark block_size must be positive");

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            block_size,
            markov_rank,
            rms_norm_eps,
            rope_theta,
        })
    }

    /// Convert to standard ModelConfig for InferenceState allocation.
    pub fn to_model_config(&self, max_seq_len: usize) -> ModelConfig {
        ModelConfig {
            architecture: "dflash".to_string(),
            n_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            n_heads: self.num_heads,
            n_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_seq_len,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            block_types: vec![BlockType::Attention; self.num_layers],
            conv_kernel_size: None,
            kv_heads_per_layer: vec![self.num_kv_heads; self.num_layers],
            scalars: ScalarMultipliers::default(),
            moe: None,
        }
    }
}

/// Pre-resolved layer weights for a single DSpark transformer layer.
#[derive(Clone)]
pub(crate) struct DSparkLayer {
    pub(crate) attn_norm: Arc<[f32]>,
    pub(crate) attn_q: WeightRef,
    pub(crate) attn_k: WeightRef,
    pub(crate) attn_v: WeightRef,
    pub(crate) attn_output: WeightRef,
    pub(crate) ffn_norm: Arc<[f32]>,
    pub(crate) ffn_gate: WeightRef,
    pub(crate) ffn_up: WeightRef,
    pub(crate) ffn_down: WeightRef,
}

/// Pure-Rust CPU DSpark draft sidecar model.
#[derive(Clone)]
pub struct DSparkDraftModel {
    pub(crate) gguf: Arc<GgufFile>,
    pub(crate) base_gguf: Arc<GgufFile>,
    pub config: DSparkConfig,
    pub(crate) layers: Arc<[DSparkLayer]>,
    pub(crate) output_norm_weight: Arc<[f32]>,
    pub(crate) base_embd_ref: WeightRef,
    pub(crate) base_output_ref: Option<WeightRef>,
    pub(crate) markov_a: Option<WeightRef>,
    pub(crate) markov_b: Option<WeightRef>,
    pub(crate) confidence_weight: Option<Arc<[f32]>>,
}

/// Session-isolated DSpark drafter holding stateful KV cache and preallocated scratch buffers.
pub struct DSparkSessionDrafter {
    model: Arc<DSparkDraftModel>,
    state: Option<InferenceState>,
    capacity: usize,
    synced_pos: usize,
    synced_tokens: Vec<u32>,
    scratch_normed: Vec<f32>,
    scratch_hidden: Vec<f32>,
    scratch_logits: Vec<f32>,
    scratch_markov: Vec<f32>,
    scratch_markov_logits: Vec<f32>,
    q8_scales: Vec<f32>,
    q8_quants: Vec<i8>,
    q8_markov_scales: Vec<f32>,
    q8_markov_quants: Vec<i8>,
}

impl DSparkSessionDrafter {
    pub fn new(model: Arc<DSparkDraftModel>) -> Self {
        let hs = model.config.hidden_size;
        let vocab = model.config.vocab_size;
        let has_markov = model.markov_a.is_some() && model.markov_b.is_some();
        let markov_rank = if has_markov {
            model.config.markov_rank
        } else {
            0
        };
        let markov_vocab = if has_markov { vocab } else { 0 };
        Self {
            model,
            state: None,
            capacity: 0,
            synced_pos: 0,
            synced_tokens: Vec::with_capacity(4096),
            scratch_normed: vec![0.0f32; hs],
            scratch_hidden: vec![0.0f32; hs],
            scratch_logits: vec![0.0f32; vocab],
            scratch_markov: vec![0.0f32; markov_rank],
            scratch_markov_logits: vec![0.0f32; markov_vocab],
            q8_scales: vec![0.0f32; hs / 32],
            q8_quants: vec![0i8; hs],
            q8_markov_scales: vec![0.0f32; markov_rank.div_ceil(32)],
            q8_markov_quants: vec![0i8; markov_rank.div_ceil(32) * 32],
        }
    }

    fn forward_token_id(
        model: &DSparkDraftModel,
        tok: usize,
        pos: usize,
        state: &mut InferenceState,
        scratch_hidden: &mut [f32],
        scratch_normed: &mut [f32],
    ) {
        if tok < model.base_embd_ref.m {
            transformer::dequantize_row_into(
                &model.base_gguf,
                &model.base_embd_ref,
                tok,
                scratch_hidden,
            );
        } else {
            scratch_hidden.fill(0.0);
        }
        model.forward_token(scratch_hidden, pos, state, scratch_normed);
    }
}

impl DSparkDraftModel {
    /// Load a DSpark draft model from a GGUF file, paired with a base model's GGUF and weights.
    pub(crate) fn from_gguf(
        draft_gguf: Arc<GgufFile>,
        base_gguf: Arc<GgufFile>,
        base_embd_ref: WeightRef,
        base_output_ref: Option<WeightRef>,
    ) -> Result<Self> {
        let config = DSparkConfig::from_gguf(&draft_gguf, base_embd_ref.m, base_embd_ref.k)?;
        let mut layers = Vec::with_capacity(config.num_layers);

        let arch = draft_gguf
            .get_str("general.architecture")
            .unwrap_or("dflash");

        let resolve_layer_weight = |l: usize, name: &str| -> Result<WeightRef> {
            let mut res =
                transformer::resolve_weight(&draft_gguf, &format!("blk.{l}.{name}.weight"))
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("layers.{l}.{name}.weight"),
                        )
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("{arch}.blk.{l}.{name}.weight"),
                        )
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("model.layers.{l}.{name}.weight"),
                        )
                    });

            if res.is_err() && name == "attn_output" {
                res = transformer::resolve_weight(&draft_gguf, &format!("blk.{l}.attn_out.weight"))
                    .or_else(|_| {
                        transformer::resolve_weight(&draft_gguf, &format!("blk.{l}.attn_wo.weight"))
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("layers.{l}.attn_out.weight"),
                        )
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("layers.{l}.attn_wo.weight"),
                        )
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("{arch}.blk.{l}.attn_out.weight"),
                        )
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("model.layers.{l}.attn_out.weight"),
                        )
                    })
                    .or_else(|_| {
                        transformer::resolve_weight(
                            &draft_gguf,
                            &format!("model.layers.{l}.attn_wo.weight"),
                        )
                    });
            }

            res
        };
        let get_layer_norm = |l: usize, name: &str| -> Result<Vec<f32>> {
            draft_gguf
                .get_tensor(&format!("blk.{l}.{name}.weight"))
                .or_else(|_| draft_gguf.get_tensor(&format!("layers.{l}.{name}.weight")))
                .or_else(|_| draft_gguf.get_tensor(&format!("{arch}.blk.{l}.{name}.weight")))
                .or_else(|_| draft_gguf.get_tensor(&format!("model.layers.{l}.{name}.weight")))
                .map(|t| t.to_f32_vec())
                .context(format!("missing {name} for layer {l}"))
        };

        for l in 0..config.num_layers {
            let attn_norm = get_layer_norm(l, "attn_norm")?;
            ensure!(
                attn_norm.len() == config.hidden_size,
                "attn_norm length {} does not match hidden size {}",
                attn_norm.len(),
                config.hidden_size
            );

            let attn_q = resolve_layer_weight(l, "attn_q")?;
            let attn_k = resolve_layer_weight(l, "attn_k")?;
            let attn_v = resolve_layer_weight(l, "attn_v")?;
            let attn_output = resolve_layer_weight(l, "attn_output")?;

            ensure!(
                attn_q.k == config.hidden_size && attn_q.m == config.num_heads * config.head_dim,
                "layer {l} attn_q shape mismatch: expected [{}, {}], got [{}, {}]",
                config.hidden_size,
                config.num_heads * config.head_dim,
                attn_q.k,
                attn_q.m
            );
            ensure!(
                attn_k.k == config.hidden_size && attn_k.m == config.num_kv_heads * config.head_dim,
                "layer {l} attn_k shape mismatch: expected [{}, {}], got [{}, {}]",
                config.hidden_size,
                config.num_kv_heads * config.head_dim,
                attn_k.k,
                attn_k.m
            );
            ensure!(
                attn_v.k == config.hidden_size && attn_v.m == config.num_kv_heads * config.head_dim,
                "layer {l} attn_v shape mismatch: expected [{}, {}], got [{}, {}]",
                config.hidden_size,
                config.num_kv_heads * config.head_dim,
                attn_v.k,
                attn_v.m
            );
            ensure!(
                attn_output.k == config.num_heads * config.head_dim
                    && attn_output.m == config.hidden_size,
                "layer {l} attn_output shape mismatch: expected [{}, {}], got [{}, {}]",
                config.num_heads * config.head_dim,
                config.hidden_size,
                attn_output.k,
                attn_output.m
            );

            let ffn_norm = get_layer_norm(l, "ffn_norm")?;
            ensure!(
                ffn_norm.len() == config.hidden_size,
                "ffn_norm length {} does not match hidden size {}",
                ffn_norm.len(),
                config.hidden_size
            );

            let ffn_gate = resolve_layer_weight(l, "ffn_gate")?;
            let ffn_up = resolve_layer_weight(l, "ffn_up")?;
            let ffn_down = resolve_layer_weight(l, "ffn_down")?;

            ensure!(
                ffn_gate.k == config.hidden_size && ffn_gate.m == config.intermediate_size,
                "layer {l} ffn_gate shape mismatch: expected [{}, {}], got [{}, {}]",
                config.hidden_size,
                config.intermediate_size,
                ffn_gate.k,
                ffn_gate.m
            );
            ensure!(
                ffn_up.k == config.hidden_size && ffn_up.m == config.intermediate_size,
                "layer {l} ffn_up shape mismatch: expected [{}, {}], got [{}, {}]",
                config.hidden_size,
                config.intermediate_size,
                ffn_up.k,
                ffn_up.m
            );
            ensure!(
                ffn_down.k == config.intermediate_size && ffn_down.m == config.hidden_size,
                "layer {l} ffn_down shape mismatch: expected [{}, {}], got [{}, {}]",
                config.intermediate_size,
                config.hidden_size,
                ffn_down.k,
                ffn_down.m
            );

            layers.push(DSparkLayer {
                attn_norm: Arc::from(attn_norm),
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_norm: Arc::from(ffn_norm),
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let output_norm_weight = draft_gguf
            .get_tensor("output_norm.weight")
            .or_else(|_| draft_gguf.get_tensor("dflash.output_norm.weight"))
            .or_else(|_| draft_gguf.get_tensor("dspark.output_norm.weight"))
            .or_else(|_| draft_gguf.get_tensor(&format!("{arch}.output_norm.weight")))
            .or_else(|_| draft_gguf.get_tensor("token_embd_norm.weight"))
            .or_else(|_| draft_gguf.get_tensor("norm.weight"))
            .or_else(|_| draft_gguf.get_tensor("model.norm.weight"))
            .or_else(|_| draft_gguf.get_tensor("model.output_norm.weight"))
            .map(|t| t.to_f32_vec())
            .unwrap_or_else(|_| vec![1.0; config.hidden_size]);
        ensure!(
            output_norm_weight.len() == config.hidden_size,
            "output_norm_weight length {} does not match hidden size {}",
            output_norm_weight.len(),
            config.hidden_size
        );

        let head_vocab = base_output_ref.as_ref().unwrap_or(&base_embd_ref).m;
        ensure!(
            config.vocab_size == head_vocab,
            "DSpark draft model vocab size ({}) != base model head vocab size ({})",
            config.vocab_size,
            head_vocab
        );
        ensure!(
            config.hidden_size == base_embd_ref.k,
            "DSpark draft model hidden size ({}) != base model hidden size ({})",
            config.hidden_size,
            base_embd_ref.k
        );
        if let Some(out_ref) = &base_output_ref {
            ensure!(
                out_ref.k == config.hidden_size,
                "DSpark draft model hidden size ({}) != base model output head hidden size ({})",
                config.hidden_size,
                out_ref.k
            );
        }

        let markov_a = transformer::resolve_weight(&draft_gguf, "dflash.markov_a.weight")
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "dspark.markov_a.weight"))
            .or_else(|_| {
                transformer::resolve_weight(&draft_gguf, &format!("{arch}.markov_a.weight"))
            })
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "markov_a.weight"))
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "markov_w1.weight"))
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "dflash.markov_w1.weight"))
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "dspark.markov_w1.weight"))
            .or_else(|_| {
                transformer::resolve_weight(&draft_gguf, "model.markov_head.markov_w1.weight")
            })
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "model.markov_head.markov_w1"))
            .ok();

        let markov_b = transformer::resolve_weight(&draft_gguf, "dflash.markov_b.weight")
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "dspark.markov_b.weight"))
            .or_else(|_| {
                transformer::resolve_weight(&draft_gguf, &format!("{arch}.markov_b.weight"))
            })
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "markov_b.weight"))
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "markov_w2.weight"))
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "dflash.markov_w2.weight"))
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "dspark.markov_w2.weight"))
            .or_else(|_| {
                transformer::resolve_weight(&draft_gguf, "model.markov_head.markov_w2.weight")
            })
            .or_else(|_| transformer::resolve_weight(&draft_gguf, "model.markov_head.markov_w2"))
            .ok();

        ensure!(
            markov_a.is_some() == markov_b.is_some(),
            "malformed DSpark draft model: markov_a and markov_b must both be present or both absent"
        );

        if let (Some(ma), Some(mb)) = (&markov_a, &markov_b) {
            ensure!(
                config.markov_rank > 0,
                "Markov rank must be positive when Markov weights are present"
            );
            ensure!(
                ma.k == mb.k,
                "Markov rank mismatch: markov_a.k ({}) != markov_b.k ({})",
                ma.k,
                mb.k
            );
            ensure!(
                ma.k == config.markov_rank,
                "Markov rank mismatch: markov_a.k ({}) != config.markov_rank ({})",
                ma.k,
                config.markov_rank
            );
            ensure!(
                mb.m == config.vocab_size,
                "Markov vocab mismatch: markov_b.m ({}) != config.vocab_size ({})",
                mb.m,
                config.vocab_size
            );
            ensure!(
                ma.m == config.vocab_size,
                "Markov vocab mismatch: markov_a.m ({}) != config.vocab_size ({})",
                ma.m,
                config.vocab_size
            );
        }

        let confidence_weight: Option<Arc<[f32]>> = draft_gguf
            .get_tensor("confidence.weight")
            .or_else(|_| draft_gguf.get_tensor("confidence_head.proj.weight"))
            .or_else(|_| draft_gguf.get_tensor("conf_proj.weight"))
            .or_else(|_| draft_gguf.get_tensor("conf_proj"))
            .or_else(|_| draft_gguf.get_tensor("model.confidence_head.proj.weight"))
            .or_else(|_| draft_gguf.get_tensor("dflash.confidence.weight"))
            .or_else(|_| draft_gguf.get_tensor("dspark.confidence.weight"))
            .or_else(|_| draft_gguf.get_tensor(&format!("{arch}.confidence.weight")))
            .map(|t| {
                let data = t.to_f32_vec();
                if data.len() >= config.hidden_size {
                    Ok(Arc::from(&data[..config.hidden_size]))
                } else {
                    Err(anyhow!(
                        "confidence_weight length ({}) < hidden_size ({})",
                        data.len(),
                        config.hidden_size
                    ))
                }
            })
            .ok()
            .transpose()?;

        if let Some(cw) = &confidence_weight {
            ensure!(
                cw.len() == config.hidden_size,
                "confidence_weight length ({}) != hidden_size ({})",
                cw.len(),
                config.hidden_size
            );
        }

        Ok(Self {
            gguf: draft_gguf,
            base_gguf,
            config,
            layers: Arc::from(layers),
            output_norm_weight: Arc::from(output_norm_weight),
            base_embd_ref,
            base_output_ref,
            markov_a,
            markov_b,
            confidence_weight,
        })
    }

    /// Single-token forward pass through the DSpark layers with zero allocation.
    pub fn forward_token(
        &self,
        hidden: &mut [f32],
        pos: usize,
        state: &mut InferenceState,
        normed: &mut [f32],
    ) {
        let hs = self.config.hidden_size;
        let dims = AttnDims {
            hidden_size: self.config.hidden_size,
            n_heads: self.config.num_heads,
            n_kv_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
            rms_norm_eps: self.config.rms_norm_eps,
            rope_theta: self.config.rope_theta,
            rope_type: cpu::RopeType::Norm,
            attn_scale: None,
            rope_freqs: None,
        };
        let attn_extras = AttnExtras {
            qkv_bias: None,
            qk_norm: None,
        };

        for (l, layer) in self.layers.iter().enumerate() {
            // 1. Pre-attention norm
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(normed, &layer.attn_norm, self.config.rms_norm_eps);

            // 2. Self-Attention Block
            let attn_weights = AttnWeights {
                attn_q: &layer.attn_q,
                attn_k: &layer.attn_k,
                attn_v: &layer.attn_v,
                attn_output: &layer.attn_output,
            };

            #[cfg(target_arch = "aarch64")]
            transformer::quantize_to_scratch(normed, state);

            forward_attn_block(
                &self.gguf,
                l,
                &attn_weights,
                &attn_extras,
                dims,
                normed,
                pos,
                state,
            );

            // Residual connection 1
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            // 3. Pre-FFN norm
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(normed, &layer.ffn_norm, self.config.rms_norm_eps);

            // 4. SwiGLU / MLP FFN Block
            let ffn_weights = FfnWeights {
                ffn_gate: &layer.ffn_gate,
                ffn_up: &layer.ffn_up,
                ffn_down: &layer.ffn_down,
            };

            #[cfg(target_arch = "aarch64")]
            transformer::quantize_to_scratch(normed, state);

            forward_ffn_block(
                &self.gguf,
                l,
                &ffn_weights,
                self.config.hidden_size,
                self.config.intermediate_size,
                normed,
                state,
            );

            // Residual connection 2
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);
        }

        // Final output norm
        cpu::rmsnorm(hidden, &self.output_norm_weight, self.config.rms_norm_eps);
        state.seq_len = pos + 1;
    }
}

impl Drafter for DSparkDraftModel {
    fn clone_drafter(&self) -> Box<dyn Drafter> {
        Box::new(DSparkSessionDrafter::new(Arc::new(self.clone())))
    }

    fn reset(&mut self) {}

    fn draft(&mut self, tokens: &[u32], max_k: usize) -> Vec<u32> {
        let mut drafter = self.clone_drafter();
        drafter.draft(tokens, max_k)
    }

    fn suggested_k(&self) -> Option<usize> {
        Some(self.config.block_size)
    }
}

impl Drafter for DSparkSessionDrafter {
    fn clone_drafter(&self) -> Box<dyn Drafter> {
        Box::new(Self::new(Arc::clone(&self.model)))
    }

    fn reset(&mut self) {
        self.state = None;
        self.capacity = 0;
        self.synced_pos = 0;
        self.synced_tokens.clear();
    }

    fn suggested_k(&self) -> Option<usize> {
        Some(self.model.config.block_size)
    }

    fn draft(&mut self, tokens: &[u32], max_k: usize) -> Vec<u32> {
        let k = max_k.min(self.model.config.block_size);
        if k == 0 || tokens.is_empty() {
            return Vec::new();
        }

        let prefix_len = tokens.len();

        // Initialize state or reallocate if capacity is exceeded
        let needs_realloc = self.state.is_none() || prefix_len + k >= self.capacity;
        if needs_realloc {
            let target_cap = (prefix_len + k + 4096).max(8192);
            self.state =
                InferenceState::from_config(&self.model.config.to_model_config(target_cap)).ok();
            self.capacity = target_cap;
            self.synced_pos = 0;
            self.synced_tokens.clear();
        }

        let state = match &mut self.state {
            Some(s) => s,
            None => return Vec::new(),
        };

        // Detect history divergence, truncations, or rewind.
        // If previous prefix diverged, invalidate synced cache and re-sync from 0.
        let prefix_diverged = self.synced_pos > prefix_len
            || self.synced_tokens.get(..self.synced_pos) != Some(&tokens[..self.synced_pos]);

        if prefix_diverged {
            self.synced_pos = 0;
            self.synced_tokens.clear();
            state.truncate_to(0);
        } else if self.synced_pos == prefix_len {
            self.synced_pos = prefix_len.saturating_sub(1);
            if state.seq_len > self.synced_pos {
                state.truncate_to(self.synced_pos);
            }
        }

        // Forward newly appended tokens incrementally (delta forward)
        for (pos, &raw_tok) in
            (self.synced_pos..prefix_len).zip(&tokens[self.synced_pos..prefix_len])
        {
            Self::forward_token_id(
                &self.model,
                raw_tok as usize,
                pos,
                state,
                &mut self.scratch_hidden,
                &mut self.scratch_normed,
            );
        }
        self.synced_pos = prefix_len;
        self.synced_tokens.clear();
        self.synced_tokens.extend_from_slice(tokens);

        let mut drafted = Vec::with_capacity(k);
        let mut last_tok = tokens.last().copied().unwrap_or(0);
        let head_ref = self
            .model
            .base_output_ref
            .as_ref()
            .unwrap_or(&self.model.base_embd_ref);

        // Generate up to k draft tokens
        for step in 0..k {
            let pos = prefix_len + step;

            // Check confidence threshold if confidence head is present (evaluated on draft steps > 0)
            if let Some(conf_w) = self.model.confidence_weight.as_ref().filter(|_| step > 0) {
                let conf_dot = cpu::dot_f32(conf_w, &self.scratch_hidden);
                if !conf_dot.is_finite() {
                    break;
                }
                let clamped_neg_dot = (-conf_dot).clamp(-80.0, 80.0);
                let prob = 1.0 / (1.0 + clamped_neg_dot.exp());
                if !prob.is_finite() || prob < 0.35 {
                    break;
                }
            }

            // Project logits via base model LM head
            #[cfg(target_arch = "aarch64")]
            {
                let nb = self.scratch_hidden.len() / 32;
                self.q8_scales.resize(nb, 0.0);
                self.q8_quants.resize(self.scratch_hidden.len(), 0);
                unsafe {
                    crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                        &self.scratch_hidden,
                        &mut self.q8_scales,
                        &mut self.q8_quants,
                    );
                }
                transformer::gemv_preq(
                    &self.model.base_gguf,
                    head_ref,
                    &self.scratch_hidden,
                    &self.q8_scales,
                    &self.q8_quants,
                    &mut self.scratch_logits,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                gemv(
                    &self.model.base_gguf,
                    head_ref,
                    &self.scratch_hidden,
                    &mut self.scratch_logits,
                );
            }

            // Add Markov transition boost if markov weights are present
            let prev_tok_idx = last_tok as usize;
            if let (Some(ma), Some(mb)) = (&self.model.markov_a, &self.model.markov_b)
                && prev_tok_idx < ma.m
            {
                transformer::dequantize_row_into(
                    &self.model.gguf,
                    ma,
                    prev_tok_idx,
                    &mut self.scratch_markov,
                );
                #[cfg(target_arch = "aarch64")]
                {
                    if self.scratch_markov.len().is_multiple_of(32) {
                        let nb_m = self.scratch_markov.len() / 32;
                        self.q8_markov_scales.resize(nb_m, 0.0);
                        self.q8_markov_quants.resize(self.scratch_markov.len(), 0);
                        unsafe {
                            crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                                &self.scratch_markov,
                                &mut self.q8_markov_scales,
                                &mut self.q8_markov_quants,
                            );
                        }
                        transformer::gemv_preq(
                            &self.model.gguf,
                            mb,
                            &self.scratch_markov,
                            &self.q8_markov_scales,
                            &self.q8_markov_quants,
                            &mut self.scratch_markov_logits,
                        );
                    } else {
                        gemv(
                            &self.model.gguf,
                            mb,
                            &self.scratch_markov,
                            &mut self.scratch_markov_logits,
                        );
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                gemv(
                    &self.model.gguf,
                    mb,
                    &self.scratch_markov,
                    &mut self.scratch_markov_logits,
                );
                cpu::add_inplace(&mut self.scratch_logits, &self.scratch_markov_logits);
            }

            let next_tok = crate::sampler::argmax(&self.scratch_logits);
            drafted.push(next_tok);
            last_tok = next_tok;

            // Forward next draft token (only if more draft steps remain)
            if step + 1 < k {
                Self::forward_token_id(
                    &self.model,
                    next_tok as usize,
                    pos,
                    state,
                    &mut self.scratch_hidden,
                    &mut self.scratch_normed,
                );
            }
        }

        // Roll back state to the verified prefix length before returning so draft tokens
        // never linger in the drafter's KV cache
        if state.seq_len > prefix_len {
            state.truncate_to(prefix_len);
        }

        drafted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn dspark_config_to_model_config_sets_dimensions() {
        let cfg = DSparkConfig {
            hidden_size: 2048,
            num_layers: 5,
            num_heads: 16,
            num_kv_heads: 4,
            intermediate_size: 5632,
            head_dim: 128,
            vocab_size: 32000,
            block_size: 9,
            markov_rank: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 500000.0,
        };

        let model_cfg = cfg.to_model_config(2048);
        assert_eq!(model_cfg.hidden_size, 2048);
        assert_eq!(model_cfg.n_layers, 5);
        assert_eq!(model_cfg.n_heads, 16);
        assert_eq!(model_cfg.n_kv_heads, 4);
        assert_eq!(model_cfg.head_dim, 128);
        assert_eq!(model_cfg.vocab_size, 32000);
        assert_eq!(model_cfg.max_seq_len, 2048);
    }

    #[test]
    fn dspark_config_validates_field_bounds() {
        let cfg = DSparkConfig {
            hidden_size: 1024,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 2,
            intermediate_size: 2048,
            head_dim: 128,
            vocab_size: 1000,
            block_size: 5,
            markov_rank: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        let mcfg = cfg.to_model_config(512);
        assert_eq!(mcfg.n_layers, 4);
        assert_eq!(mcfg.hidden_size, 1024);
        assert_eq!(mcfg.intermediate_size, 2048);
    }

    fn empty_gguf() -> Arc<GgufFile> {
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        data.resize(data.len() + 4096, 0);
        Arc::new(GgufFile::from_bytes(Arc::from(data.into_boxed_slice())).unwrap())
    }

    #[test]
    fn dspark_session_drafter_reset_clears_synced_pos_and_state() {
        let cfg = DSparkConfig {
            hidden_size: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 128,
            head_dim: 32,
            vocab_size: 100,
            block_size: 4,
            markov_rank: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        let mut drafter = DSparkSessionDrafter {
            model: Arc::new(DSparkDraftModel {
                gguf: empty_gguf(),
                base_gguf: empty_gguf(),
                config: cfg.clone(),
                layers: Arc::from([]),
                output_norm_weight: Arc::from([]),
                base_embd_ref: transformer::WeightRef::new(0, 100 * 64 * 4, DType::F32, 100, 64),
                base_output_ref: None,
                markov_a: None,
                markov_b: None,
                confidence_weight: None,
            }),
            state: Some(InferenceState::from_config(&cfg.to_model_config(256)).unwrap()),
            capacity: 256,
            synced_pos: 42,
            synced_tokens: Vec::new(),
            scratch_normed: vec![0.0f32; 64],
            scratch_hidden: vec![0.0f32; 64],
            scratch_logits: vec![0.0f32; 100],
            scratch_markov: vec![0.0f32; 16],
            scratch_markov_logits: vec![0.0f32; 100],
            q8_scales: Vec::new(),
            q8_quants: Vec::new(),
            q8_markov_scales: Vec::new(),
            q8_markov_quants: Vec::new(),
        };

        assert!(drafter.state.is_some());
        assert_eq!(drafter.synced_pos, 42);

        drafter.reset();

        assert!(drafter.state.is_none());
        assert_eq!(drafter.synced_pos, 0);
    }

    #[test]
    fn dspark_draft_refreshes_scratch_hidden_when_prefix_unchanged() {
        let cfg = DSparkConfig {
            hidden_size: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            head_dim: 16,
            vocab_size: 10,
            block_size: 2,
            markov_rank: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        let model = Arc::new(DSparkDraftModel {
            gguf: empty_gguf(),
            base_gguf: empty_gguf(),
            config: cfg,
            layers: Arc::from([]),
            output_norm_weight: Arc::from(vec![1.0f32; 32]),
            base_embd_ref: transformer::WeightRef::new(0, 10 * 32 * 4, DType::F32, 10, 32),
            base_output_ref: None,
            markov_a: None,
            markov_b: None,
            confidence_weight: None,
        });

        let mut drafter = DSparkSessionDrafter::new(model);
        assert_eq!(drafter.synced_pos, 0);

        let tokens = vec![1, 2, 3];
        let _ = drafter.draft(&tokens, 2);
        assert_eq!(drafter.synced_pos, 3);

        // Calling draft again with the same prefix must maintain consistent synced_pos
        let _ = drafter.draft(&tokens, 2);
        assert_eq!(drafter.synced_pos, 3);

        // Calling draft with a shortened history (e.g. chat rewind) must safely reset and re-sync
        let shortened = vec![1];
        let _ = drafter.draft(&shortened, 2);
        assert_eq!(drafter.synced_pos, 1);
    }

    #[test]
    fn dspark_confidence_head_terminates_early() {
        let cfg = DSparkConfig {
            hidden_size: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            head_dim: 16,
            vocab_size: 10,
            block_size: 5,
            markov_rank: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        // Force a large negative confidence weight so sigmoid(dot) ~ 0.0 < 0.35
        let conf_w = Arc::from(vec![-100.0f32; 32]);
        let model = Arc::new(DSparkDraftModel {
            gguf: empty_gguf(),
            base_gguf: empty_gguf(),
            config: cfg,
            layers: Arc::from([]),
            output_norm_weight: Arc::from(vec![1.0f32; 32]),
            base_embd_ref: transformer::WeightRef::new(0, 10 * 32 * 4, DType::F32, 10, 32),
            base_output_ref: None,
            markov_a: None,
            markov_b: None,
            confidence_weight: Some(conf_w),
        });

        let mut drafter = DSparkSessionDrafter::new(model);
        // Step 0 produces token 0 without checking confidence; step 1 encounters negative confidence and terminates early
        let drafted = drafter.draft(&[1], 2);
        assert_eq!(drafted.len(), 1);
    }

    #[test]
    fn dspark_draft_detects_divergent_prefix() {
        let cfg = DSparkConfig {
            hidden_size: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            head_dim: 16,
            vocab_size: 10,
            block_size: 2,
            markov_rank: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        let model = Arc::new(DSparkDraftModel {
            gguf: empty_gguf(),
            base_gguf: empty_gguf(),
            config: cfg,
            layers: Arc::from([]),
            output_norm_weight: Arc::from(vec![1.0f32; 32]),
            base_embd_ref: transformer::WeightRef::new(0, 10 * 32 * 4, DType::F32, 10, 32),
            base_output_ref: None,
            markov_a: None,
            markov_b: None,
            confidence_weight: None,
        });

        let mut drafter = DSparkSessionDrafter::new(model);
        let _ = drafter.draft(&[1, 2, 3], 2);
        assert_eq!(drafter.synced_pos, 3);

        // A divergent sequence of the same length must be detected and re-synced from 0
        let _ = drafter.draft(&[4, 5, 6], 2);
        assert_eq!(drafter.synced_pos, 3);
        assert_eq!(drafter.synced_tokens, vec![4, 5, 6]);
    }

    fn gguf_with_payload(payload: &[u8]) -> (Arc<GgufFile>, usize) {
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        let offset = data.len();
        data.extend_from_slice(payload);
        (
            Arc::new(GgufFile::from_bytes(Arc::from(data.into_boxed_slice())).unwrap()),
            offset,
        )
    }

    #[test]
    fn dspark_markov_head_boosts_transition_logits() {
        let cfg = DSparkConfig {
            hidden_size: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            head_dim: 16,
            vocab_size: 4,
            block_size: 1,
            markov_rank: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        // Construct mock Markov weights with large boost for transition token 3
        let mut markov_a_data = [0.0f32; 4 * 2]; // 4 vocab x 2 rank
        markov_a_data[2] = 10.0; // from token 1 (idx 1*2), emit rank factor 10.0
        let mut markov_b_data = [0.0f32; 4 * 2]; // 4 vocab x 2 rank
        markov_b_data[6] = 10.0; // to token 3 (idx 3*2), dot product = 100.0

        let mut payload = Vec::new();
        for &f in markov_a_data.iter().chain(markov_b_data.iter()) {
            payload.extend_from_slice(&f.to_ne_bytes());
        }
        let (draft_gguf, offset) = gguf_with_payload(&payload);

        let model = Arc::new(DSparkDraftModel {
            gguf: draft_gguf,
            base_gguf: empty_gguf(),
            config: cfg,
            layers: Arc::from([]),
            output_norm_weight: Arc::from(vec![1.0f32; 32]),
            base_embd_ref: transformer::WeightRef::new(0, 4 * 32 * 4, DType::F32, 4, 32),
            base_output_ref: None,
            markov_a: Some(transformer::WeightRef::new(offset, 4 * 2 * 4, DType::F32, 4, 2)),
            markov_b: Some(transformer::WeightRef::new(offset + 4 * 2 * 4, 4 * 2 * 4, DType::F32, 4, 2)),
            confidence_weight: None,
        });

        let mut drafter = DSparkSessionDrafter::new(model);
        // Prefix ending in token 1 will boost candidate token 3 by +100.0
        let drafted = drafter.draft(&[1], 1);
        assert_eq!(drafted, vec![3]);
    }

    #[test]
    fn dspark_confidence_head_handles_non_finite_activations() {
        let cfg = DSparkConfig {
            hidden_size: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            head_dim: 16,
            vocab_size: 4,
            block_size: 4,
            markov_rank: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        // Confidence weights initialized to +INFINITY values
        let conf_w = Arc::from(vec![f32::INFINITY; 32]);

        let model = Arc::new(DSparkDraftModel {
            gguf: empty_gguf(),
            base_gguf: empty_gguf(),
            config: cfg,
            layers: Arc::from([]),
            output_norm_weight: Arc::from(vec![1.0f32; 32]),
            base_embd_ref: transformer::WeightRef::new(0, 4 * 32 * 4, DType::F32, 4, 32),
            base_output_ref: None,
            markov_a: None,
            markov_b: None,
            confidence_weight: Some(conf_w),
        });

        let mut drafter = DSparkSessionDrafter::new(model);
        // Step 0 produces token 0 without checking confidence; step 1 encounters +INFINITY and safely terminates
        let drafted = drafter.draft(&[1], 3);
        assert_eq!(drafted.len(), 1);
    }

    #[test]
    fn dspark_drafter_dynamically_expands_capacity_on_long_sequences() {
        let cfg = DSparkConfig {
            hidden_size: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            head_dim: 16,
            vocab_size: 4,
            block_size: 4,
            markov_rank: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        let model = Arc::new(DSparkDraftModel {
            gguf: empty_gguf(),
            base_gguf: empty_gguf(),
            config: cfg,
            layers: Arc::from([]),
            output_norm_weight: Arc::from(vec![1.0f32; 32]),
            base_embd_ref: transformer::WeightRef::new(0, 4 * 32 * 4, DType::F32, 4, 32),
            base_output_ref: None,
            markov_a: None,
            markov_b: None,
            confidence_weight: None,
        });

        let mut drafter = DSparkSessionDrafter::new(model);
        // Initially draft with 1 token (creates state with cap 8192)
        let _ = drafter.draft(&[1], 2);
        assert_eq!(drafter.capacity, 8192);

        // Subsequent draft with 9000 tokens dynamically expands capacity
        let long_prompt = vec![1u32; 9000];
        let _ = drafter.draft(&long_prompt, 2);
        assert!(drafter.capacity >= 9000 + 4096);
    }
}

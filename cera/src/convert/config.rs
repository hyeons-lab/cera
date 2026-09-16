//! Mapping from Hugging Face `config.json` to GGUF model metadata.

use crate::convert::writer::GgufWriter;
use crate::session::CeraError;
use serde::Deserialize;
use serde_json::Value;

/// HF Transformers `config.json` schema representation.
#[derive(Debug, Clone, Deserialize)]
pub struct HfModelConfig {
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    #[serde(alias = "layer_norm_eps", alias = "layer_norm_epsilon")]
    pub rms_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
    #[serde(alias = "head_dim", alias = "key_length")]
    pub head_dim: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub bos_token_id: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub eos_token_id: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub pad_token_id: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn parse_token_id_value(val: &Value) -> Option<u32> {
    match val {
        Value::Number(n) => n.as_u64().map(|v| v as u32),
        Value::Array(arr) => arr.first().and_then(parse_token_id_value),
        Value::String(s) => s.parse::<u32>().ok(),
        _ => None,
    }
}

fn deserialize_token_id<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<Value> = Option::deserialize(deserializer)?;
    Ok(val.as_ref().and_then(parse_token_id_value))
}

impl HfModelConfig {
    /// Parse from JSON bytes.
    pub fn parse_from_bytes(bytes: &[u8]) -> Result<Self, CeraError> {
        let mut cfg: Self = serde_json::from_slice(bytes)
            .map_err(|e| CeraError::Backend(format!("failed to parse model config.json: {e}")))?;
        cfg.resolve_nested_text_config();
        Ok(cfg)
    }

    /// Parse from JSON string.
    pub fn from_json_str(json_str: &str) -> Result<Self, CeraError> {
        Self::parse_from_bytes(json_str.as_bytes())
    }

    fn resolve_nested_text_config(&mut self) {
        if let Some(Value::Object(text_cfg)) = self.extra.get("text_config") {
            if let Some(mt) = text_cfg.get("model_type").and_then(|v| v.as_str()) {
                let lower = self.model_type.to_ascii_lowercase();
                if self.model_type.is_empty()
                    || lower.contains("vl")
                    || lower.contains("multimodal")
                    || lower.contains("vision")
                {
                    self.model_type = mt.to_string();
                }
            }
            if self.hidden_size.is_none() {
                self.hidden_size = text_cfg
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.num_hidden_layers.is_none() {
                self.num_hidden_layers = text_cfg
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.num_attention_heads.is_none() {
                self.num_attention_heads = text_cfg
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.num_key_value_heads.is_none() {
                self.num_key_value_heads = text_cfg
                    .get("num_key_value_heads")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.intermediate_size.is_none() {
                self.intermediate_size = text_cfg
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.vocab_size.is_none() {
                self.vocab_size = text_cfg
                    .get("vocab_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.max_position_embeddings.is_none() {
                self.max_position_embeddings = text_cfg
                    .get("max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.rms_norm_eps.is_none() {
                self.rms_norm_eps = text_cfg
                    .get("rms_norm_eps")
                    .or_else(|| text_cfg.get("layer_norm_eps"))
                    .or_else(|| text_cfg.get("layer_norm_epsilon"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
            }
            if self.rope_theta.is_none() {
                self.rope_theta = text_cfg
                    .get("rope_theta")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
            }
            if self.head_dim.is_none() {
                self.head_dim = text_cfg
                    .get("head_dim")
                    .or_else(|| text_cfg.get("key_length"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.bos_token_id.is_none() {
                self.bos_token_id = text_cfg.get("bos_token_id").and_then(parse_token_id_value);
            }
            if self.eos_token_id.is_none() {
                self.eos_token_id = text_cfg.get("eos_token_id").and_then(parse_token_id_value);
            }
            if self.pad_token_id.is_none() {
                self.pad_token_id = text_cfg.get("pad_token_id").and_then(parse_token_id_value);
            }
        }
    }

    /// Determine the canonical GGUF architecture name.
    pub fn gguf_architecture(&self) -> &str {
        match self.model_type.to_ascii_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => "llama",
            "qwen35" | "qwen3_5" | "qwen3.5" => "qwen35",
            "bailingmoe3" | "bailingmoe" | "bailingmoe2" | "bailing_moe_v3" | "bailing_moe"
            | "bailing" => "bailingmoe3",
            "qwen2" | "qwen" => "qwen2",
            "qwen3" => "qwen3",
            "mistral3" | "ministral3" | "ministral" => "mistral3",
            "mistral" => "llama", // Classic Mistral maps to llama architecture in GGUF
            "gemma" => "gemma",
            "gemma2" => "gemma2",
            "minicpm" | "minicpm3" => "minicpm",
            "nanbeige" | "nanbeige2" | "nanbeige4" => "nanbeige",
            "olmo" => "olmo",
            "olmo2" | "olmo3" => "olmo2",
            "olmoe" => "olmoe",
            "mamba2" | "falcon_mamba" => "mamba2",
            "mamba" => "mamba",
            "phi3" | "phi" | "phi-3" | "phi4" | "phi-4" => "phi3",
            "lfm" | "lfm2" | "lfm2.5" | "liquid" => "lfm2",
            "whisper" => "whisper",
            _ => {
                if let Some(first_arch) = self.architectures.first() {
                    let arch_lower = first_arch.to_ascii_lowercase();
                    if arch_lower.contains("whisper") {
                        "whisper"
                    } else if arch_lower.contains("lfm") || arch_lower.contains("liquid") {
                        "lfm2"
                    } else if arch_lower.contains("qwen35")
                        || arch_lower.contains("qwen3_5")
                        || arch_lower.contains("qwen3.5")
                    {
                        "qwen35"
                    } else if arch_lower.contains("qwen3") {
                        "qwen3"
                    } else if arch_lower.contains("bailing") {
                        "bailingmoe3"
                    } else if arch_lower.contains("qwen2") || arch_lower.contains("qwen") {
                        "qwen2"
                    } else if arch_lower.contains("nanbeige") {
                        "nanbeige"
                    } else if arch_lower.contains("mistral3") || arch_lower.contains("ministral") {
                        "mistral3"
                    } else if arch_lower.contains("minicpm") {
                        "minicpm"
                    } else if arch_lower.contains("olmoe") {
                        "olmoe"
                    } else if arch_lower.contains("olmo2") || arch_lower.contains("olmo3") {
                        "olmo2"
                    } else if arch_lower.contains("olmo") {
                        "olmo"
                    } else if arch_lower.contains("gemma2") {
                        "gemma2"
                    } else if arch_lower.contains("gemma") {
                        "gemma"
                    } else if arch_lower.contains("mamba2") {
                        "mamba2"
                    } else if arch_lower.contains("mamba") {
                        "mamba"
                    } else if !arch_lower.contains("moe")
                        && !arch_lower.contains("phi2")
                        && !arch_lower.contains("phi-2")
                        && !arch_lower.contains("phi_2")
                        && (arch_lower.contains("phi3")
                            || arch_lower.contains("phi-3")
                            || arch_lower.contains("phi4")
                            || arch_lower.contains("phi-4")
                            || arch_lower == "phi"
                            || arch_lower.starts_with("phi-")
                            || arch_lower.starts_with("phi_"))
                    {
                        "phi3"
                    } else {
                        "llama"
                    }
                } else {
                    "llama"
                }
            }
        }
    }

    /// Apply architecture metadata keys to a [`GgufWriter`].
    pub fn apply_to_gguf_writer(&self, writer: &mut GgufWriter, model_name: &str) {
        let arch = self.gguf_architecture();

        writer.add_string("general.architecture", arch);
        writer.add_string("general.name", model_name);

        if arch == "whisper" {
            writer.add_u32("whisper.sampling_rate", 16000);
            let d_model = self
                .extra
                .get("d_model")
                .and_then(|v| v.as_u64())
                .or(self.hidden_size.map(|h| h as u64))
                .unwrap_or(384) as u32;
            writer.add_u32("whisper.audio.embedding_length", d_model);
            writer.add_u32("whisper.text.embedding_length", d_model);

            let enc_heads = self
                .extra
                .get("encoder_attention_heads")
                .and_then(|v| v.as_u64())
                .or(self.num_attention_heads.map(|h| h as u64))
                .unwrap_or(6) as u32;
            let dec_heads = self
                .extra
                .get("decoder_attention_heads")
                .and_then(|v| v.as_u64())
                .or(self.num_attention_heads.map(|h| h as u64))
                .unwrap_or(6) as u32;
            writer.add_u32("whisper.audio.attention.head_count", enc_heads);
            writer.add_u32("whisper.text.attention.head_count", dec_heads);
            writer.add_u32("whisper.audio.head_count", enc_heads);
            writer.add_u32("whisper.text.head_count", dec_heads);

            let enc_layers = self
                .extra
                .get("encoder_layers")
                .and_then(|v| v.as_u64())
                .or(self.num_hidden_layers.map(|l| l as u64))
                .unwrap_or(4) as u32;
            let dec_layers = self
                .extra
                .get("decoder_layers")
                .and_then(|v| v.as_u64())
                .or(self.num_hidden_layers.map(|l| l as u64))
                .unwrap_or(4) as u32;
            writer.add_u32("whisper.audio.block_count", enc_layers);
            writer.add_u32("whisper.text.block_count", dec_layers);
            writer.add_u32("whisper.audio.layer_count", enc_layers);
            writer.add_u32("whisper.text.layer_count", dec_layers);

            let num_mel_bins = self
                .extra
                .get("num_mel_bins")
                .and_then(|v| v.as_u64())
                .unwrap_or(80) as u32;
            writer.add_u32("whisper.audio.num_mel_bins", num_mel_bins);
            writer.add_u32("whisper.audio.n_mel", num_mel_bins);

            let src_ctx = self
                .extra
                .get("max_source_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(1500) as u32;
            let tgt_ctx = self
                .extra
                .get("max_target_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(448) as u32;
            writer.add_u32("whisper.audio.context_length", src_ctx);
            writer.add_u32("whisper.text.context_length", tgt_ctx);
            writer.add_u32("whisper.audio.ctx", src_ctx);
            writer.add_u32("whisper.text.ctx", tgt_ctx);

            if let Some(v) = self.vocab_size {
                writer.add_u32("whisper.vocab_size", v as u32);
            }
            if let Some(bos) = self.bos_token_id {
                writer.add_u32("tokenizer.ggml.bos_token_id", bos);
                writer.add_u32("general.bos_token_id", bos);
            }
            if let Some(eos) = self.eos_token_id {
                writer.add_u32("tokenizer.ggml.eos_token_id", eos);
                writer.add_u32("general.eos_token_id", eos);
            }
            if let Some(pad) = self.pad_token_id {
                writer.add_u32("tokenizer.ggml.padding_token_id", pad);
                writer.add_u32("general.pad_token_id", pad);
            }
            return;
        }

        if let Some(v) = self.vocab_size {
            writer.add_u32(format!("{arch}.vocab_size"), v as u32);
        }
        if let Some(ctx) = self.max_position_embeddings {
            writer.add_u32(format!("{arch}.context_length"), ctx as u32);
        }
        if let Some(h) = self.hidden_size {
            writer.add_u32(format!("{arch}.embedding_length"), h as u32);
        }
        if let Some(layers) = self.num_hidden_layers {
            writer.add_u32(format!("{arch}.block_count"), layers as u32);
        }
        let layers = self.num_hidden_layers.unwrap_or(0);
        if let Some(heads) = self.num_attention_heads {
            writer.add_u32(format!("{arch}.attention.head_count"), heads as u32);
        }
        if let Some(kv_heads) = self.num_key_value_heads.or(self.num_attention_heads) {
            if arch == "lfm2" {
                let kv_array = vec![kv_heads as i32; layers.max(1)];
                writer.add_i32_array(format!("{arch}.attention.head_count_kv"), kv_array);
            } else {
                writer.add_u32(format!("{arch}.attention.head_count_kv"), kv_heads as u32);
            }
        }
        if let Some(ffn) = self.intermediate_size {
            writer.add_u32(format!("{arch}.feed_forward_length"), ffn as u32);
        }
        if let Some(dim) = self.head_dim {
            writer.add_u32(format!("{arch}.attention.key_length"), dim as u32);
        }
        if let Some(eps) = self.rms_norm_eps {
            writer.add_f32(format!("{arch}.attention.layer_norm_rms_epsilon"), eps);
        }
        if let Some(rope) = self.rope_theta {
            writer.add_f32(format!("{arch}.rope.freq_base"), rope);
        }

        if let Some(bos) = self.bos_token_id {
            writer.add_u32("tokenizer.ggml.bos_token_id", bos);
            writer.add_u32("general.bos_token_id", bos);
        }
        if let Some(eos) = self.eos_token_id {
            writer.add_u32("tokenizer.ggml.eos_token_id", eos);
            writer.add_u32("general.eos_token_id", eos);
        }
        if let Some(pad) = self.pad_token_id {
            writer.add_u32("tokenizer.ggml.padding_token_id", pad);
            writer.add_u32("general.pad_token_id", pad);
        }

        if arch == "nanbeige" {
            let num_loops = self
                .extra
                .get("num_loops")
                .and_then(|v| {
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                })
                .unwrap_or(1) as u32;
            writer.add_u32("nanbeige.num_loops", num_loops);

            let skip_loop_final_norm = self
                .extra
                .get("skip_loop_final_norm")
                .and_then(|v| {
                    v.as_bool().or_else(|| match v {
                        Value::Number(n) => n.as_u64().map(|x| x != 0),
                        Value::String(s) => match s.trim() {
                            "true" | "1" => Some(true),
                            "false" | "0" => Some(false),
                            _ => None,
                        },
                        _ => None,
                    })
                })
                .unwrap_or(false);
            writer.add_bool("nanbeige.skip_loop_final_norm", skip_loop_final_norm);
        }

        if arch == "minicpm" || arch == "minicpm3" {
            if let Some(scale_emb) = self.extra.get("scale_emb").and_then(|v| v.as_f64()) {
                writer.add_f32("minicpm.embedding_scale", scale_emb as f32);
            }
            if let Some(scale_depth) = self.extra.get("scale_depth").and_then(|v| v.as_f64()) {
                let n_layers = self.num_hidden_layers.unwrap_or(1).max(1) as f64;
                let residual_scale = scale_depth / n_layers.sqrt();
                writer.add_f32("minicpm.residual_scale", residual_scale as f32);
            }
            if let (Some(h), Some(dim_base)) = (
                self.hidden_size,
                self.extra.get("dim_model_base").and_then(|v| v.as_f64()),
            ) && h > 0
                && dim_base > 0.0
            {
                let logit_scale = dim_base / (h as f64);
                writer.add_f32("minicpm.logit_scale", logit_scale as f32);
            }
        }

        if arch == "gemma2" {
            if let Some(cap) = self
                .extra
                .get("attn_logit_softcapping")
                .and_then(|v| v.as_f64())
            {
                writer.add_f32("gemma2.attn_logit_softcapping", cap as f32);
            }
            if let Some(cap) = self
                .extra
                .get("final_logit_softcapping")
                .and_then(|v| v.as_f64())
            {
                writer.add_f32("gemma2.final_logit_softcapping", cap as f32);
            }
        }

        if let Some(sw) = self
            .extra
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .filter(|&w| w > 0)
        {
            writer.add_u32(format!("{arch}.attention.sliding_window"), sw as u32);
        }

        if arch == "qwen35" {
            if let Some(conv_kernel) = self
                .extra
                .get("linear_conv_kernel_dim")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.conv_kernel", conv_kernel as u32);
            }
            if let Some(inner_size) = self.extra.get("linear_inner_size").and_then(Value::as_u64) {
                writer.add_u32("qwen35.ssm.inner_size", inner_size as u32);
            }
            if let Some(state_size) = self
                .extra
                .get("linear_key_head_dim")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.state_size", state_size as u32);
            }
            if let Some(dt_rank) = self
                .extra
                .get("linear_num_value_heads")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.time_step_rank", dt_rank as u32);
            }
            if let Some(group_count) = self
                .extra
                .get("linear_num_key_heads")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.group_count", group_count as u32);
            }
            if let Some(full_attn_interval) = self
                .extra
                .get("full_attention_interval")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.full_attention_interval", full_attn_interval as u32);
            }
        }

        if arch == "bailingmoe3" {
            if let Some(conv_kernel) = self
                .extra
                .get("short_conv_kernel_size")
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.ssm.conv_kernel", conv_kernel as u32);
            }
            if let Some(head_dim) = self.extra.get("head_dim").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.kda.head_dim", head_dim as u32);
            }
            if let Some(safe_gate) = self.extra.get("kda_safe_gate").and_then(Value::as_bool) {
                writer.add_bool("bailingmoe3.kda.safe_gate", safe_gate);
            }
            if let Some(lower_bound) = self.extra.get("kda_lower_bound").and_then(Value::as_f64) {
                writer.add_f32("bailingmoe3.kda.gate_lower_bound", lower_bound as f32);
            }
            if let Some(kv_lora_rank) = self.extra.get("kv_lora_rank").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.attention.kv_lora_rank", kv_lora_rank as u32);
            }
            if let Some(q_lora_rank) = self.extra.get("q_lora_rank").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.attention.q_lora_rank", q_lora_rank as u32);
            }
            let qk_rope_dim = self.extra.get("qk_rope_head_dim").and_then(Value::as_u64);
            if let Some(rope_dim) = qk_rope_dim {
                writer.add_u32("bailingmoe3.rope.dimension_count", rope_dim as u32);
            }
            let kv_lora_rank_val = self
                .extra
                .get("kv_lora_rank")
                .and_then(Value::as_u64)
                .unwrap_or(512);
            let qk_nope_dim_val = self
                .extra
                .get("qk_nope_head_dim")
                .and_then(Value::as_u64)
                .unwrap_or(128);
            let qk_rope_dim_val = qk_rope_dim.unwrap_or(64);
            writer.add_u32(
                "bailingmoe3.attention.key_length",
                (kv_lora_rank_val + qk_rope_dim_val) as u32,
            );
            writer.add_u32(
                "bailingmoe3.attention.key_length_mla",
                (qk_nope_dim_val + qk_rope_dim_val) as u32,
            );
            if let Some(v_head_dim) = self.extra.get("v_head_dim").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.attention.value_length_mla", v_head_dim as u32);
            }
            if let Some(n_exp) = self
                .extra
                .get("num_experts")
                .or_else(|| self.extra.get("n_routed_experts"))
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.expert_count", n_exp as u32);
            }
            if let Some(n_used) = self
                .extra
                .get("num_experts_per_tok")
                .or_else(|| self.extra.get("n_activated_experts"))
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.expert_used_count", n_used as u32);
            }
            if let Some(moe_ff) = self
                .extra
                .get("moe_intermediate_size")
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.expert_feed_forward_length", moe_ff as u32);
            }
            if let Some(shexp_ff) = self
                .extra
                .get("moe_shared_expert_intermediate_size")
                .and_then(Value::as_u64)
            {
                writer.add_u32(
                    "bailingmoe3.expert_shared_feed_forward_length",
                    shexp_ff as u32,
                );
            }
            if let Some(shexp_n) = self.extra.get("num_shared_experts").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.expert_shared_count", shexp_n as u32);
            }
            if let Some(lead_dense) = self
                .extra
                .get("first_k_dense_replace")
                .or_else(|| self.extra.get("leading_dense_block_count"))
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.leading_dense_block_count", lead_dense as u32);
            }
            if let Some(scale) = self
                .extra
                .get("routed_scaling_factor")
                .and_then(Value::as_f64)
            {
                writer.add_f32("bailingmoe3.expert_weights_scale", scale as f32);
            }
        }

        if arch == "mistral3" {
            let text_config = self.extra.get("text_config");
            let rope_scaling = self
                .extra
                .get("rope_parameters")
                .or_else(|| self.extra.get("rope_scaling"))
                .or_else(|| {
                    text_config
                        .and_then(|tc| tc.get("rope_parameters").or_else(|| tc.get("rope_scaling")))
                });
            if let Some(scale_type) = rope_scaling
                .and_then(|v| v.get("type").or_else(|| v.get("rope_type")))
                .and_then(Value::as_str)
            {
                writer.add_string("mistral3.rope.scaling.type", scale_type);
            }
            if let Some(factor) = rope_scaling
                .and_then(|v| v.get("factor"))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.factor", factor as f32);
            }
            if let Some(orig_ctx) = rope_scaling
                .and_then(|v| {
                    v.get("original_max_position_embeddings")
                        .or_else(|| v.get("original_context_length"))
                })
                .and_then(Value::as_u64)
                .filter(|&v| v > 0)
            {
                writer.add_u32(
                    "mistral3.rope.scaling.original_context_length",
                    orig_ctx as u32,
                );
                writer.add_u32("mistral3.attention.temperature_length", orig_ctx as u32);
            }
            if let Some(scale) = self
                .extra
                .get("attention_temperature_scale")
                .or_else(|| self.extra.get("attn_temp_scale"))
                .or_else(|| rope_scaling.and_then(|v| v.get("llama_4_scaling_beta")))
                .or_else(|| {
                    self.extra
                        .get("llama_4_scaling")
                        .and_then(|v| v.get("beta"))
                })
                .or_else(|| {
                    text_config.and_then(|tc| {
                        tc.get("attention_temperature_scale")
                            .or_else(|| tc.get("attn_temp_scale"))
                            .or_else(|| tc.get("llama_4_scaling").and_then(|v| v.get("beta")))
                    })
                })
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.attention.temperature_scale", scale as f32);
            }
            if let Some(log_mul) = self
                .extra
                .get("yarn_log_multiplier")
                .or_else(|| {
                    rope_scaling.and_then(|v| {
                        v.get("yarn_log_multiplier")
                            .or_else(|| v.get("mscale_all_dim"))
                    })
                })
                .or_else(|| text_config.and_then(|tc| tc.get("yarn_log_multiplier")))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v >= 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_log_multiplier", log_mul as f32);
            }
            if let Some(beta_fast) = rope_scaling
                .and_then(|v| {
                    v.get("yarn_beta_fast")
                        .or_else(|| v.get("beta_fast"))
                        .or_else(|| v.get("beta"))
                })
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_beta_fast", beta_fast as f32);
            }
            if let Some(beta_slow) = rope_scaling
                .and_then(|v| {
                    v.get("yarn_beta_slow")
                        .or_else(|| v.get("beta_slow"))
                        .or_else(|| v.get("alpha"))
                })
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_beta_slow", beta_slow as f32);
            }
            if let Some(ext_factor) = rope_scaling
                .and_then(|v| v.get("yarn_ext_factor").or_else(|| v.get("ext_factor")))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v >= 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_ext_factor", ext_factor as f32);
            }
            if let Some(attn_factor) = rope_scaling
                .and_then(|v| v.get("attn_factor").or_else(|| v.get("yarn_attn_factor")))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.attn_factor", attn_factor as f32);
            }
        }

        // Token classification labels (e.g. LiquidAI/pii-detect)
        if let Some(Value::Object(id2label)) = self.extra.get("id2label") {
            let mut label_pairs: Vec<(usize, String)> = Vec::new();
            for (k, v) in id2label {
                if let (Ok(idx), Some(s)) = (k.parse::<usize>(), v.as_str()) {
                    label_pairs.push((idx, s.to_string()));
                }
            }
            label_pairs.sort_by_key(|p| p.0);
            let labels: Vec<String> = label_pairs.into_iter().map(|p| p.1).collect();
            if !labels.is_empty() {
                writer.add_u32("token_classifier.num_labels", labels.len() as u32);
                writer.add_string_array("token_classifier.labels", labels);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::writer::MetadataValue;

    #[test]
    fn test_token_classifier_id2label_metadata() {
        let json_data = r#"{
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirP2ForTokenClassification"],
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "id2label": {
                "0": "O",
                "1": "B-NAME",
                "2": "I-NAME",
                "3": "S-NAME"
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "pii-detect");

        assert_eq!(
            writer.get_metadata("token_classifier.num_labels"),
            Some(&MetadataValue::Uint32(4))
        );
        let expected_labels = vec![
            "O".to_string(),
            "B-NAME".to_string(),
            "I-NAME".to_string(),
            "S-NAME".to_string(),
        ];
        assert_eq!(
            writer.get_metadata("token_classifier.labels"),
            Some(&MetadataValue::StringArray(expected_labels))
        );
    }

    #[test]
    fn test_nanbeige_config_mapping() {
        let json_data = r#"{
            "model_type": "nanbeige",
            "architectures": ["NanbeigeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_loops": 2,
            "skip_loop_final_norm": false
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "nanbeige");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "nanbeige-test");
        assert_eq!(
            writer.get_metadata("nanbeige.num_loops"),
            Some(&MetadataValue::Uint32(2))
        );
        assert_eq!(
            writer.get_metadata("nanbeige.skip_loop_final_norm"),
            Some(&MetadataValue::Bool(false))
        );
    }

    #[test]
    fn test_minicpm_config_mapping() {
        let json_data = r#"{
            "model_type": "minicpm",
            "architectures": ["MiniCPMForCausalLM"],
            "hidden_size": 2304,
            "num_hidden_layers": 36,
            "scale_emb": 12.0,
            "scale_depth": 1.4,
            "dim_model_base": 256.0
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "minicpm");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "minicpm-test");
        assert_eq!(
            writer.get_metadata("minicpm.embedding_scale"),
            Some(&MetadataValue::Float32(12.0))
        );
        let expected_residual = (1.4 / (36.0f64).sqrt()) as f32;
        assert_eq!(
            writer.get_metadata("minicpm.residual_scale"),
            Some(&MetadataValue::Float32(expected_residual))
        );
        let expected_logit = (256.0 / 2304.0) as f32;
        assert_eq!(
            writer.get_metadata("minicpm.logit_scale"),
            Some(&MetadataValue::Float32(expected_logit))
        );
    }

    #[test]
    fn test_gemma2_config_mapping() {
        let json_data = r#"{
            "model_type": "gemma2",
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "sliding_window": 4096
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "gemma2");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "gemma2-test");
        assert_eq!(
            writer.get_metadata("gemma2.attn_logit_softcapping"),
            Some(&MetadataValue::Float32(50.0))
        );
        assert_eq!(
            writer.get_metadata("gemma2.attention.sliding_window"),
            Some(&MetadataValue::Uint32(4096))
        );
    }

    #[test]
    fn test_olmoe_config_mapping() {
        let json_data = r#"{
            "model_type": "olmoe",
            "architectures": ["OlmoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "olmoe");
    }

    #[test]
    fn test_qwen3_architecture_detection() {
        let json_data = r#"{
            "model_type": "custom",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "qwen3");
    }

    #[test]
    fn test_sliding_window_generalized_emission() {
        let json_data = r#"{
            "model_type": "qwen2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "sliding_window": 32768
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "qwen2-test");
        assert_eq!(
            writer.get_metadata("qwen2.attention.sliding_window"),
            Some(&MetadataValue::Uint32(32768))
        );
    }

    #[test]
    fn test_qwen35_config_mapping() {
        let json_data = r#"{
            "model_type": "qwen35",
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "linear_conv_kernel_dim": 4,
            "linear_inner_size": 2048,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 16,
            "linear_num_key_heads": 8,
            "full_attention_interval": 4
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "qwen35");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "qwen35-test");
        assert_eq!(
            writer.get_metadata("qwen35.ssm.conv_kernel"),
            Some(&MetadataValue::Uint32(4))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.inner_size"),
            Some(&MetadataValue::Uint32(2048))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.state_size"),
            Some(&MetadataValue::Uint32(128))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.time_step_rank"),
            Some(&MetadataValue::Uint32(16))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.group_count"),
            Some(&MetadataValue::Uint32(8))
        );
        assert_eq!(
            writer.get_metadata("qwen35.full_attention_interval"),
            Some(&MetadataValue::Uint32(4))
        );
    }

    #[test]
    fn test_mistral3_config_mapping() {
        let json_data = r#"{
            "model_type": "mistral3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "attention_temperature_scale": 0.1,
            "yarn_log_multiplier": 0.0707
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "mistral3-test");

        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_scale"),
            Some(&MetadataValue::Float32(0.1))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_log_multiplier"),
            Some(&MetadataValue::Float32(0.0707))
        );
    }

    #[test]
    fn test_mistral3_nested_rope_scaling_mapping() {
        let json_data = r#"{
            "model_type": "mistral3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "attention_temperature_scale": 0.1,
            "rope_scaling": {
                "type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
                "yarn_log_multiplier": 0.0707,
                "yarn_beta_fast": 32.0,
                "yarn_beta_slow": 1.0
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "mistral3-nested-test");

        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.type"),
            Some(&MetadataValue::String("yarn".to_string()))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.factor"),
            Some(&MetadataValue::Float32(2.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.original_context_length"),
            Some(&MetadataValue::Uint32(32768))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_log_multiplier"),
            Some(&MetadataValue::Float32(0.0707))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_fast"),
            Some(&MetadataValue::Float32(32.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_slow"),
            Some(&MetadataValue::Float32(1.0))
        );
    }

    #[test]
    fn test_mistral3_rope_parameters_canonical_mapping() {
        let json_data = r#"{
            "model_type": "mistral3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 2.5,
                "original_max_position_embeddings": 32768,
                "mscale_all_dim": 0.0707,
                "llama_4_scaling_beta": 0.125,
                "beta": 32.0,
                "alpha": 1.0
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "mistral3-rope-params-test");

        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.type"),
            Some(&MetadataValue::String("yarn".to_string()))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.factor"),
            Some(&MetadataValue::Float32(2.5))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.original_context_length"),
            Some(&MetadataValue::Uint32(32768))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_log_multiplier"),
            Some(&MetadataValue::Float32(0.0707))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_scale"),
            Some(&MetadataValue::Float32(0.125))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_fast"),
            Some(&MetadataValue::Float32(32.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_slow"),
            Some(&MetadataValue::Float32(1.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_length"),
            Some(&MetadataValue::Uint32(32768))
        );
    }

    #[test]
    fn test_mistral3_multimodal_text_config_mapping() {
        let json_data = r#"{
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "pixtral",
            "text_config": {
                "model_type": "mistral3",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "vocab_size": 32768,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 16384,
                    "mscale_all_dim": 0.1,
                    "llama_4_scaling_beta": 0.2,
                    "beta": 64.0,
                    "alpha": 2.0
                }
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "pixtral-test");

        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.type"),
            Some(&MetadataValue::String("yarn".to_string()))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.factor"),
            Some(&MetadataValue::Float32(4.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.original_context_length"),
            Some(&MetadataValue::Uint32(16384))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_length"),
            Some(&MetadataValue::Uint32(16384))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_scale"),
            Some(&MetadataValue::Float32(0.2))
        );
    }

    #[test]
    fn test_phi3_hf_model_config() {
        let json_data = r#"{
            "model_type": "phi3",
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32064,
            "max_position_embeddings": 4096,
            "sliding_window": 2048,
            "rms_norm_eps": 1e-5
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "phi3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "phi3-mini-test");

        assert_eq!(
            writer.get_metadata("general.architecture"),
            Some(&MetadataValue::String("phi3".to_string()))
        );
        assert_eq!(
            writer.get_metadata("phi3.embedding_length"),
            Some(&MetadataValue::Uint32(3072))
        );
        assert_eq!(
            writer.get_metadata("phi3.feed_forward_length"),
            Some(&MetadataValue::Uint32(8192))
        );
        assert_eq!(
            writer.get_metadata("phi3.attention.sliding_window"),
            Some(&MetadataValue::Uint32(2048))
        );
    }

    #[test]
    fn test_phi2_not_classified_as_phi3() {
        let json_data = r#"{
            "architectures": ["PhiForCausalLM"],
            "model_type": "phi-msft",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_attention_heads": 32,
            "num_hidden_layers": 32
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_ne!(cfg.gguf_architecture(), "phi3");

        let json_phi2 = r#"{
            "architectures": ["Phi-2"],
            "model_type": "phi2",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_attention_heads": 32,
            "num_hidden_layers": 32
        }"#;
        let cfg_phi2 = HfModelConfig::from_json_str(json_phi2).unwrap();
        assert_ne!(cfg_phi2.gguf_architecture(), "phi3");
    }
}

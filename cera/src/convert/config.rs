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
            "qwen2" | "qwen" => "qwen2",
            "mistral" => "llama", // Mistral maps to llama architecture in GGUF
            "gemma" | "gemma2" => "gemma",
            "phi3" | "phi" => "phi3",
            "lfm" | "lfm2" | "lfm2.5" | "liquid" => "lfm2",
            "whisper" => "whisper",
            _ => {
                if let Some(first_arch) = self.architectures.first() {
                    let arch_lower = first_arch.to_ascii_lowercase();
                    if arch_lower.contains("whisper") {
                        "whisper"
                    } else if arch_lower.contains("lfm") || arch_lower.contains("liquid") {
                        "lfm2"
                    } else if arch_lower.contains("qwen2") || arch_lower.contains("qwen") {
                        "qwen2"
                    } else if arch_lower.contains("llama") || arch_lower.contains("mistral") {
                        "llama"
                    } else if arch_lower.contains("gemma") {
                        "gemma"
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
}

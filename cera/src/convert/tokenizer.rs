//! Tokenizer parser and converter from Hugging Face `tokenizer.json` to GGUF metadata.

use crate::convert::writer::GgufWriter;
use crate::session::CeraError;
use serde::Deserialize;
use std::collections::BTreeMap;

/// HF `tokenizer.json` top-level structure.
#[derive(Debug, Clone, Deserialize)]
pub struct HfTokenizerJson {
    #[serde(default)]
    pub model: HfTokenizerModel,
    #[serde(default)]
    pub added_tokens: Vec<HfAddedToken>,
    #[serde(default)]
    pub pre_tokenizer: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum HfVocab {
    Map(BTreeMap<String, u32>),
    List(Vec<(String, serde_json::Value)>),
}

impl Default for HfVocab {
    fn default() -> Self {
        Self::Map(BTreeMap::new())
    }
}

impl HfVocab {
    pub fn for_each_token_with_score<F: FnMut(&str, u32, Option<f32>)>(&self, mut f: F) {
        match self {
            Self::Map(m) => {
                for (tok, &id) in m {
                    f(tok, id, None);
                }
            }
            Self::List(l) => {
                for (i, (tok, val)) in l.iter().enumerate() {
                    let score = val.as_f64().map(|v| v as f32);
                    f(tok, i as u32, score);
                }
            }
        }
    }

    pub fn for_each_token<F: FnMut(&str, u32)>(&self, mut f: F) {
        self.for_each_token_with_score(|tok, id, _| f(tok, id));
    }

    pub fn max_id(&self) -> u32 {
        match self {
            Self::Map(m) => m.values().copied().max().unwrap_or(0),
            Self::List(l) => l.len().saturating_sub(1) as u32,
        }
    }

    pub fn to_id_map(&self) -> BTreeMap<String, u32> {
        let mut map = BTreeMap::new();
        self.for_each_token(|tok, id| {
            map.insert(tok.to_string(), id);
        });
        map
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HfTokenizerModel {
    #[serde(rename = "type", default)]
    pub model_type: String,
    #[serde(default)]
    pub vocab: HfVocab,
    #[serde(default)]
    pub merges: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfAddedToken {
    pub id: u32,
    pub content: String,
    #[serde(default)]
    pub special: bool,
}

impl HfTokenizerJson {
    /// Parse from JSON bytes.
    pub fn parse_from_bytes(bytes: &[u8]) -> Result<Self, CeraError> {
        serde_json::from_slice(bytes)
            .map_err(|e| CeraError::Backend(format!("failed to parse tokenizer.json: {e}")))
    }

    /// Parse from JSON string.
    pub fn from_json_str(json_str: &str) -> Result<Self, CeraError> {
        serde_json::from_str(json_str)
            .map_err(|e| CeraError::Backend(format!("failed to parse tokenizer.json: {e}")))
    }

    /// Convert tokenizer vocab, merges, and added tokens to GGUF metadata KVs.
    pub fn apply_to_gguf_writer(&self, writer: &mut GgufWriter, chat_template: Option<&str>) {
        let model_type = match self.model.model_type.to_ascii_lowercase().as_str() {
            "bpe" => "gpt2",
            "unigram" => "llama",
            "wordpiece" => "bert",
            _ => "gpt2",
        };
        writer.add_string("tokenizer.ggml.model", model_type);

        let pre_str = self
            .pre_tokenizer
            .as_ref()
            .map(|v| v.to_string().to_ascii_lowercase())
            .unwrap_or_default();

        let pre_type = if pre_str.contains("llama-v3") || pre_str.contains("llama3") {
            "llama3"
        } else if pre_str.contains("qwen2") || pre_str.contains("qwen") {
            "qwen2"
        } else if pre_str.contains("deepseek") {
            "deepseek-llm"
        } else if pre_str.contains("chatglm") {
            "chatglm"
        } else if pre_str.contains("tekken") {
            "tekken"
        } else {
            match self.model.model_type.to_ascii_lowercase().as_str() {
                "unigram" | "spm" => "default",
                "llama" | "llama3" => "llama3",
                "qwen2" | "qwen" => "qwen2",
                _ => "gpt2",
            }
        };
        writer.add_string("tokenizer.ggml.pre", pre_type);

        // Invert vocab mapping ID -> token string
        let max_vocab = self.model.vocab.max_id();
        let max_added = self.added_tokens.iter().map(|t| t.id).max().unwrap_or(0);
        let max_id = max_vocab.max(max_added);

        let vocab_size = (max_id.min(1_000_000) + 1) as usize;
        let mut tokens = vec![String::new(); vocab_size];
        let mut scores = vec![0.0f32; vocab_size];
        let mut token_types = vec![1i32; vocab_size]; // 1 = normal token

        self.model
            .vocab
            .for_each_token_with_score(|tok, id, score| {
                if (id as usize) < vocab_size {
                    tokens[id as usize] = tok.to_string();
                    if let Some(s) = score {
                        scores[id as usize] = s;
                    }
                }
            });

        for tok in &self.added_tokens {
            if (tok.id as usize) < vocab_size {
                tokens[tok.id as usize] = tok.content.clone();
                token_types[tok.id as usize] = if tok.special { 3 } else { 4 }; // 3 = control/special, 4 = user defined
            }
        }

        // Fill any empty gaps with placeholder
        for (i, t) in tokens.iter_mut().enumerate() {
            if t.is_empty() {
                *t = format!("<token_{i}>");
                token_types[i] = 2; // 2 = unknown
            }
        }

        writer.add_string_array("tokenizer.ggml.tokens", tokens);
        writer.add_f32_array("tokenizer.ggml.scores", scores);
        writer.add_i32_array("tokenizer.ggml.token_type", token_types);

        if !self.model.merges.is_empty() {
            writer.add_string_array("tokenizer.ggml.merges", self.model.merges.clone());
        }

        if let Some(tmpl) = chat_template {
            writer.add_string("tokenizer.chat_template", tmpl);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tokenizer_map_vocab() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": { "<pad>": 0, "<s>": 1, "hello": 2 },
                "merges": []
            },
            "added_tokens": [
                { "id": 3, "content": "<unk>", "special": true }
            ]
        }"#;

        let tok = HfTokenizerJson::from_json_str(json).unwrap();
        let map = tok.model.vocab.to_id_map();
        assert_eq!(map.get("hello"), Some(&2));
        assert_eq!(tok.added_tokens.len(), 1);
    }

    #[test]
    fn test_parse_tokenizer_list_vocab() {
        let json = r#"{
            "model": {
                "type": "Unigram",
                "vocab": [
                    ["<unk>", 0.0],
                    ["<s>", -1.5],
                    ["</s>", -2.0],
                    ["world", -3.2]
                ]
            }
        }"#;

        let tok = HfTokenizerJson::from_json_str(json).unwrap();
        let map = tok.model.vocab.to_id_map();
        assert_eq!(map.get("<unk>"), Some(&0));
        assert_eq!(map.get("world"), Some(&3));
    }
}

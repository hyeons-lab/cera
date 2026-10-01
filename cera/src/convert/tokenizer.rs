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
    #[serde(default)]
    pub post_processor: Option<serde_json::Value>,
}

/// How [`HfTokenizerJson::apply_to_gguf_writer_with`] lays the vocabulary out.
///
/// The default reproduces the historical layout. `llama_cpp_layout` switches to
/// what llama.cpp's `convert_hf_to_gguf.py` writes, which llama.cpp's loader relies
/// on and which the LFM family's GGUFs are checked against.
#[derive(Debug, Clone, Default)]
pub struct VocabOptions<'a> {
    /// Follow llama.cpp's converter: pad the token list to `pad_to` with unused
    /// `[PAD<i>]` entries (the embedding has that many rows and llama.cpp sizes the
    /// vocabulary from the list), type added tokens that look like control tokens
    /// as control, and write `add_bos_token` / `add_eos_token`.
    pub llama_cpp_layout: bool,
    /// The model's `vocab_size`.
    pub pad_to: Option<usize>,
    /// `tokenizer_config.json`: the BOS/EOS token names and its `add_*_token` overrides.
    pub tokenizer_config: Option<&'a serde_json::Value>,
}

/// GGUF `token_type` values (`llama_token_type`).
const TOKEN_NORMAL: i32 = 1;
const TOKEN_UNKNOWN: i32 = 2;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const TOKEN_UNUSED: i32 = 5;

/// llama.cpp's `does_token_look_special`: added tokens that the tokenizer does not
/// flag `special` but that are control tokens by shape (`<|im_start|>`).
fn token_looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<\u{ff5c}") && token.ends_with("\u{ff5c}>"))
        || (token.starts_with("<unused") && token.ends_with('>'))
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

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum HfMerges {
    Strings(Vec<String>),
    Pairs(Vec<Vec<String>>),
}

impl Default for HfMerges {
    fn default() -> Self {
        Self::Strings(Vec::new())
    }
}

impl HfMerges {
    pub fn to_string_vec(&self) -> Vec<String> {
        match self {
            Self::Strings(s) => s.clone(),
            Self::Pairs(pairs) => pairs.iter().map(|p| p.join(" ")).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Strings(s) => s.is_empty(),
            Self::Pairs(p) => p.is_empty(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HfTokenizerModel {
    #[serde(rename = "type", default)]
    pub model_type: String,
    #[serde(default)]
    pub vocab: HfVocab,
    #[serde(default)]
    pub merges: HfMerges,
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

    /// Whether the post-processor adds BOS / EOS around a single sequence.
    ///
    /// The same crude reading of `TemplateProcessing` that gguf-py uses: a `single`
    /// template that opens with the BOS token adds BOS, one that closes with the EOS
    /// token adds EOS. `None` means the tokenizer does not say. An unnamed BOS/EOS
    /// (`bos` / `eos` of `None`) matches whatever the template uses.
    fn template_adds_specials(
        &self,
        bos: Option<&str>,
        eos: Option<&str>,
    ) -> (Option<bool>, Option<bool>) {
        let Some(post) = self.post_processor.as_ref() else {
            return (None, None);
        };
        let processors: Vec<&serde_json::Value> = match post.get("processors") {
            Some(serde_json::Value::Array(list)) => list.iter().collect(),
            _ => vec![post],
        };
        let (mut add_bos, mut add_eos) = (None, None);
        for processor in processors {
            if processor.get("type").and_then(|t| t.as_str()) != Some("TemplateProcessing") {
                continue;
            }
            let Some(single) = processor.get("single").and_then(|v| v.as_array()) else {
                continue;
            };
            if single.len() < 2 {
                continue;
            }
            let special_id = |entry: &serde_json::Value| {
                entry
                    .get("SpecialToken")
                    .and_then(|t| t.get("id"))
                    .and_then(|id| id.as_str())
                    .map(str::to_owned)
            };
            // When the config does not name the token, the template's own is taken to
            // be it (gguf-py does the same), so a missing `tokenizer_config.json` does
            // not read as "adds no BOS".
            if let Some(first) = single.first().and_then(special_id) {
                add_bos = Some(bos.is_none_or(|bos| bos == first));
            }
            // (gguf-py would re-point EOS at the template's token here; the EOS id
            // comes from the config, so a mismatch is left unstated instead)
            if let Some(last) = single.last().and_then(special_id)
                && eos.is_none_or(|eos| eos == last)
            {
                add_eos = Some(true);
            }
        }
        (add_bos, add_eos)
    }

    /// Convert tokenizer vocab, merges, and added tokens to GGUF metadata KVs.
    pub fn apply_to_gguf_writer(&self, writer: &mut GgufWriter, chat_template: Option<&str>) {
        self.apply_to_gguf_writer_with(writer, chat_template, &VocabOptions::default());
    }

    /// [`Self::apply_to_gguf_writer`] with explicit layout options.
    pub fn apply_to_gguf_writer_with(
        &self,
        writer: &mut GgufWriter,
        chat_template: Option<&str>,
        options: &VocabOptions<'_>,
    ) {
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
        } else if pre_str.contains("(?i:'s|'t|'re")
            || pre_str.contains("\\p{l}\\p{n}")
            || pre_str.contains("lfm")
        {
            "lfm2"
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

        let mut vocab_size = (max_id.min(1_000_000) + 1) as usize;
        if options.llama_cpp_layout {
            vocab_size = vocab_size.max(options.pad_to.unwrap_or(0).min(1_000_000));
        }
        let mut tokens = vec![String::new(); vocab_size];
        let mut scores = vec![0.0f32; vocab_size];
        let mut token_types = vec![TOKEN_NORMAL; vocab_size];

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
                let control =
                    tok.special || (options.llama_cpp_layout && token_looks_special(&tok.content));
                token_types[tok.id as usize] = if control {
                    TOKEN_CONTROL
                } else {
                    TOKEN_USER_DEFINED
                };
            }
        }

        // Fill any empty gaps with placeholder
        for (i, t) in tokens.iter_mut().enumerate() {
            if t.is_empty() {
                if options.llama_cpp_layout {
                    *t = format!("[PAD{i}]");
                    token_types[i] = TOKEN_UNUSED;
                } else {
                    *t = format!("<token_{i}>");
                    token_types[i] = TOKEN_UNKNOWN;
                }
            }
        }

        writer.add_string_array("tokenizer.ggml.tokens", tokens);
        writer.add_f32_array("tokenizer.ggml.scores", scores);
        writer.add_i32_array("tokenizer.ggml.token_type", token_types);

        if !self.model.merges.is_empty() {
            writer.add_string_array("tokenizer.ggml.merges", self.model.merges.to_string_vec());
        }

        if let Some(tmpl) = chat_template {
            writer.add_string("tokenizer.chat_template", tmpl);
        }

        if options.llama_cpp_layout {
            let name_of = |key: &str| {
                options
                    .tokenizer_config
                    .and_then(|c| c.get(key))
                    .and_then(|t| match t {
                        serde_json::Value::String(s) => Some(s.as_str()),
                        serde_json::Value::Object(o) => o.get("content")?.as_str(),
                        _ => None,
                    })
            };
            let (mut add_bos, mut add_eos) =
                self.template_adds_specials(name_of("bos_token"), name_of("eos_token"));
            // an explicit `add_*_token` in tokenizer_config.json wins over the template
            let explicit = |key: &str| {
                options
                    .tokenizer_config
                    .and_then(|c| c.get(key))
                    .and_then(|v| v.as_bool())
            };
            add_bos = explicit("add_bos_token").or(add_bos);
            add_eos = explicit("add_eos_token").or(add_eos);
            if let Some(v) = add_bos {
                writer.add_bool("tokenizer.ggml.add_bos_token", v);
            }
            if let Some(v) = add_eos {
                writer.add_bool("tokenizer.ggml.add_eos_token", v);
            }
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

    fn templated_tokenizer(single: &str) -> HfTokenizerJson {
        HfTokenizerJson::from_json_str(&format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{"<s>": 0, "</s>": 1, "a": 2}}, "merges": []}},
                "added_tokens": [{{"id": 0, "content": "<s>", "special": true}},
                                 {{"id": 1, "content": "</s>", "special": true}}],
                "post_processor": {{"type": "TemplateProcessing", "single": {single}}}
            }}"#
        ))
        .unwrap()
    }

    fn bool_key(writer: &GgufWriter, key: &str) -> Option<bool> {
        match writer.get_metadata(key) {
            Some(crate::convert::writer::MetadataValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    #[test]
    fn llama_cpp_layout_reads_bos_and_eos_from_the_template() {
        let config = serde_json::json!({"bos_token": "<s>", "eos_token": "</s>"});
        let options = VocabOptions {
            llama_cpp_layout: true,
            pad_to: None,
            tokenizer_config: Some(&config),
        };
        // BOS A EOS
        let tok = templated_tokenizer(
            r#"[{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}},
                {"SpecialToken": {"id": "</s>"}}]"#,
        );
        let mut writer = GgufWriter::new();
        tok.apply_to_gguf_writer_with(&mut writer, None, &options);
        assert_eq!(
            bool_key(&writer, "tokenizer.ggml.add_bos_token"),
            Some(true)
        );
        assert_eq!(
            bool_key(&writer, "tokenizer.ggml.add_eos_token"),
            Some(true)
        );

        // a template that does not open with the BOS token adds none
        let tok = templated_tokenizer(r#"[{"Sequence": {"id": "A"}}, {"Sequence": {"id": "B"}}]"#);
        let mut writer = GgufWriter::new();
        tok.apply_to_gguf_writer_with(&mut writer, None, &options);
        assert_eq!(bool_key(&writer, "tokenizer.ggml.add_bos_token"), None);
    }

    #[test]
    fn explicit_add_token_flags_override_the_template() {
        let config = serde_json::json!({
            "bos_token": "<s>", "eos_token": "</s>", "add_bos_token": false,
        });
        let tok =
            templated_tokenizer(r#"[{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}}]"#);
        let mut writer = GgufWriter::new();
        let options = VocabOptions {
            llama_cpp_layout: true,
            pad_to: None,
            tokenizer_config: Some(&config),
        };
        tok.apply_to_gguf_writer_with(&mut writer, None, &options);
        assert_eq!(
            bool_key(&writer, "tokenizer.ggml.add_bos_token"),
            Some(false)
        );
    }

    #[test]
    fn default_layout_is_unchanged() {
        // no padding, no add_*_token keys, the historical placeholder for gaps
        let tok =
            templated_tokenizer(r#"[{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}}]"#);
        let mut writer = GgufWriter::new();
        tok.apply_to_gguf_writer(&mut writer, None);
        assert_eq!(bool_key(&writer, "tokenizer.ggml.add_bos_token"), None);
        let tokens = match writer.get_metadata("tokenizer.ggml.tokens") {
            Some(crate::convert::writer::MetadataValue::StringArray(t)) => t.clone(),
            other => panic!("tokens: {other:?}"),
        };
        assert_eq!(tokens, ["<s>", "</s>", "a"]);
    }

    #[test]
    fn control_tokens_are_recognised_by_shape() {
        assert!(token_looks_special("<|im_start|>"));
        assert!(token_looks_special("<pad>"));
        assert!(token_looks_special("<unused12>"));
        assert!(!token_looks_special("<think>"));
        assert!(!token_looks_special("hello"));
    }

    #[test]
    fn an_unnamed_bos_token_is_the_templates_own() {
        // no tokenizer_config.json at all
        let tok =
            templated_tokenizer(r#"[{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}}]"#);
        let mut writer = GgufWriter::new();
        let options = VocabOptions {
            llama_cpp_layout: true,
            ..VocabOptions::default()
        };
        tok.apply_to_gguf_writer_with(&mut writer, None, &options);
        assert_eq!(
            bool_key(&writer, "tokenizer.ggml.add_bos_token"),
            Some(true)
        );
        // a named BOS that is a different token still reads as "no BOS added"
        let config = serde_json::json!({"bos_token": "<other>"});
        let mut writer = GgufWriter::new();
        let options = VocabOptions {
            llama_cpp_layout: true,
            tokenizer_config: Some(&config),
            ..VocabOptions::default()
        };
        tok.apply_to_gguf_writer_with(&mut writer, None, &options);
        assert_eq!(
            bool_key(&writer, "tokenizer.ggml.add_bos_token"),
            Some(false)
        );
    }

    #[test]
    fn a_named_eos_that_differs_from_the_templates_last_token_is_left_unstated() {
        let tok = templated_tokenizer(
            r#"[{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}},
                {"SpecialToken": {"id": "</s>"}}]"#,
        );
        let config = serde_json::json!({"bos_token": "<s>", "eos_token": "<|im_end|>"});
        let mut writer = GgufWriter::new();
        let options = VocabOptions {
            llama_cpp_layout: true,
            tokenizer_config: Some(&config),
            ..VocabOptions::default()
        };
        tok.apply_to_gguf_writer_with(&mut writer, None, &options);
        assert_eq!(
            bool_key(&writer, "tokenizer.ggml.add_bos_token"),
            Some(true)
        );
        assert_eq!(bool_key(&writer, "tokenizer.ggml.add_eos_token"), None);
    }
}

//! The GGUF that `cera::convert` writes for an LFM2 checkpoint must be one llama.cpp's
//! loader and cera's own loader both accept, so it has to match what llama.cpp's
//! `convert_hf_to_gguf.py` writes for the same checkpoint.
//!
//! This builds a tiny Hugging Face LFM2 checkpoint in the Transformers 5 layout (the
//! one `LiquidAI/LFM2.5-*` ship): `block_ff_dim` with auto-adjust, `norm_eps`,
//! `rope_parameters`, `layer_types`, a `[channels, 1, taps]` conv kernel, a
//! `model.embedding_norm` tensor, a tokenizer with fewer entries than `vocab_size`,
//! and the chat template in `chat_template.jinja`. It converts it and checks the
//! points where cera's output used to differ from llama.cpp's (each of them made
//! llama.cpp refuse the file, and the missing embedding norm made cera refuse it
//! too), then loads the result and runs a forward pass.
//!
//! The reference values were read off the llama.cpp converter's output for
//! `LFM2.5-350M`: a byte-for-byte diff of every tensor, the token list and its types
//! came out identical after these fixes.

#![cfg(all(feature = "std-fs", feature = "mmap"))]

use std::path::{Path, PathBuf};

use cera::convert::pipeline::quantize_safetensors_to_gguf;
use cera::convert::quantize::TargetQuant;
use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::Model;
use cera::model::lfm2::Lfm2Model;
use cera::tokenizer::BpeTokenizer;
use serde_json::json;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
/// `block_ff_dim` as the config states it; the model's MLP is narrower (see
/// `ADJUSTED_FF`).
const BLOCK_FF_DIM: usize = 192;
/// `int(2 * 192 / 3) = 128`, scaled by 1.0 and rounded up to a multiple of 64.
const ADJUSTED_FF: usize = 128;
const L_CACHE: usize = 3;
const VOCAB_SIZE: usize = 300;
/// conv, conv, attention, conv
const LAYER_TYPES: [&str; 4] = ["conv", "conv", "full_attention", "conv"];
const BOS: &str = "<|startoftext|>";

/// GPT-2 byte-to-unicode table, the alphabet of a byte-level BPE vocabulary.
fn byte_level_alphabet() -> Vec<char> {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    printable.extend(0xA1..=0xAC);
    printable.extend(0xAE..=0xFF);
    let mut next_extra = 0u32;
    (0u32..256)
        .map(|b| {
            if printable.contains(&b) {
                char::from_u32(b).unwrap()
            } else {
                next_extra += 1;
                char::from_u32(255 + next_extra).unwrap()
            }
        })
        .collect()
}

/// The checkpoint's tokenizer: five added tokens (two of them not flagged `special`),
/// the 256 byte tokens and one merged token, 262 ids in all, short of `VOCAB_SIZE`.
fn tokenizer_json() -> serde_json::Value {
    let added = [
        ("<|pad|>", true),
        (BOS, true),
        ("<|im_end|>", true),
        // not flagged special, but a control token by its shape
        ("<|tool_call_start|>", false),
        // neither special nor control-shaped: stays a user-defined token
        ("<think>", false),
    ];
    let mut vocab = serde_json::Map::new();
    let mut added_tokens = Vec::new();
    for (id, (content, special)) in added.iter().enumerate() {
        vocab.insert(content.to_string(), json!(id));
        added_tokens.push(json!({
            "id": id, "content": content, "special": special, "normalized": false,
        }));
    }
    for (b, c) in byte_level_alphabet().into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(added.len() + b));
    }
    let merged_id = added.len() + 256;
    vocab.insert("he".into(), json!(merged_id));
    assert_eq!(merged_id + 1, 262);
    json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e"]},
        "added_tokens": added_tokens,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "post_processor": {"type": "Sequence", "processors": [
            {"type": "ByteLevel", "add_prefix_space": true},
            {"type": "TemplateProcessing",
             "single": [{"SpecialToken": {"id": BOS, "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}}],
             "pair": []},
        ]},
    })
}

fn config_json() -> serde_json::Value {
    json!({
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "block_auto_adjust_ff_dim": true,
        "block_ff_dim": BLOCK_FF_DIM,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 64,
        "block_norm_eps": 1e-5,
        "norm_eps": 1e-5,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "conv_L_cache": L_CACHE,
        "conv_bias": false,
        "hidden_size": HIDDEN,
        "intermediate_size": BLOCK_FF_DIM,
        "layer_types": LAYER_TYPES,
        "max_position_embeddings": 512,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYER_TYPES.len(),
        "num_key_value_heads": KV_HEADS,
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "vocab_size": VOCAB_SIZE,
    })
}

/// A tensor's row-major shape and its values.
type Values = (Vec<usize>, Vec<f32>);

/// Deterministic pseudo-random values in `[-scale, scale]`.
struct Weights(u64);

impl Weights {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
    }

    fn tensor(&mut self, shape: &[usize], scale: f32) -> Values {
        let n = shape.iter().product();
        (shape.to_vec(), (0..n).map(|_| self.next(scale)).collect())
    }

    /// Norm weights near one.
    fn norm(&mut self, n: usize) -> Values {
        (vec![n], (0..n).map(|_| 1.0 + self.next(0.1)).collect())
    }
}

fn hf_tensors() -> Vec<(String, Values)> {
    let mut w = Weights(0x9E37_79B9_7F4A_7C15);
    let mut t = vec![
        (
            "model.embed_tokens.weight".to_string(),
            w.tensor(&[VOCAB_SIZE, HIDDEN], 0.3),
        ),
        // LFM2 applies this one after the last block, whatever Hugging Face calls it
        ("model.embedding_norm.weight".to_string(), w.norm(HIDDEN)),
    ];
    for (i, kind) in LAYER_TYPES.iter().enumerate() {
        let p = format!("model.layers.{i}");
        t.push((format!("{p}.operator_norm.weight"), w.norm(HIDDEN)));
        t.push((format!("{p}.ffn_norm.weight"), w.norm(HIDDEN)));
        t.push((
            format!("{p}.feed_forward.w1.weight"),
            w.tensor(&[ADJUSTED_FF, HIDDEN], 0.08),
        ));
        t.push((
            format!("{p}.feed_forward.w3.weight"),
            w.tensor(&[ADJUSTED_FF, HIDDEN], 0.08),
        ));
        t.push((
            format!("{p}.feed_forward.w2.weight"),
            w.tensor(&[HIDDEN, ADJUSTED_FF], 0.08),
        ));
        if *kind == "conv" {
            t.push((
                format!("{p}.conv.conv.weight"),
                w.tensor(&[HIDDEN, 1, L_CACHE], 0.3),
            ));
            t.push((
                format!("{p}.conv.in_proj.weight"),
                w.tensor(&[3 * HIDDEN, HIDDEN], 0.08),
            ));
            t.push((
                format!("{p}.conv.out_proj.weight"),
                w.tensor(&[HIDDEN, HIDDEN], 0.08),
            ));
        } else {
            t.push((
                format!("{p}.self_attn.q_proj.weight"),
                w.tensor(&[HEADS * HEAD_DIM, HIDDEN], 0.08),
            ));
            t.push((
                format!("{p}.self_attn.k_proj.weight"),
                w.tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], 0.08),
            ));
            t.push((
                format!("{p}.self_attn.v_proj.weight"),
                w.tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], 0.08),
            ));
            t.push((
                format!("{p}.self_attn.out_proj.weight"),
                w.tensor(&[HIDDEN, HEADS * HEAD_DIM], 0.08),
            ));
            t.push((
                format!("{p}.self_attn.q_layernorm.weight"),
                w.norm(HEAD_DIM),
            ));
            t.push((
                format!("{p}.self_attn.k_layernorm.weight"),
                w.norm(HEAD_DIM),
            ));
        }
    }
    t
}

fn write_safetensors(path: &Path, tensors: &[(String, Values)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (shape, values)) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

/// A checkpoint directory that deletes itself. Unique per call so parallel tests do
/// not share files.
struct Checkpoint(PathBuf);

impl Checkpoint {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cera_convert_lfm2_{tag}_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config_json().to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            json!({"bos_token": BOS, "eos_token": "<|im_end|>", "pad_token": "<|pad|>"})
                .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
        )
        .unwrap();
        write_safetensors(&dir.join("model.safetensors"), &hf_tensors());
        Self(dir)
    }

    fn convert(&self, quant: TargetQuant) -> GgufFile {
        let out = self.0.join("out.gguf");
        quantize_safetensors_to_gguf(&self.0, &out, quant).expect("convert");
        GgufFile::open(&out).expect("open converted gguf")
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn metadata_matches_llama_cpp_converter() {
    let ckpt = Checkpoint::new("meta");
    let gguf = ckpt.convert(TargetQuant::F32);

    assert_eq!(gguf.architecture(), Some("lfm2"));
    // 0 on the conv layers is how llama.cpp tells the block kinds apart
    assert_eq!(
        gguf.get_i32_array("lfm2.attention.head_count_kv"),
        Some(vec![0, 0, KV_HEADS as i32, 0]),
    );
    // the width the MLP really has, not `block_ff_dim`
    assert_eq!(
        gguf.get_u32("lfm2.feed_forward_length"),
        Some(ADJUSTED_FF as u32)
    );
    assert_eq!(gguf.get_u32("lfm2.shortconv.l_cache"), Some(L_CACHE as u32));
    assert_eq!(
        gguf.get_f32("lfm2.attention.layer_norm_rms_epsilon"),
        Some(1e-5)
    );
    assert_eq!(gguf.get_f32("lfm2.rope.freq_base"), Some(1_000_000.0));
    assert_eq!(
        gguf.get_u32("lfm2.attention.head_count"),
        Some(HEADS as u32)
    );
}

#[test]
fn tensors_use_llama_cpp_names_and_shapes() {
    let ckpt = Checkpoint::new("tensors");
    let gguf = ckpt.convert(TargetQuant::F32);

    assert!(gguf.tensors.contains_key("token_embd_norm.weight"));
    assert!(
        !gguf.tensors.keys().any(|n| n.starts_with("model.")),
        "an HF tensor name leaked through: {:?}",
        gguf.tensors
            .keys()
            .filter(|n| n.starts_with("model."))
            .collect::<Vec<_>>()
    );
    // llama.cpp's loader wants the conv kernel as 2-D [taps, channels]
    for layer in [0, 1, 3] {
        assert_eq!(
            gguf.tensors[&format!("blk.{layer}.shortconv.conv.weight")].shape,
            vec![L_CACHE, HIDDEN],
            "layer {layer}"
        );
    }
}

#[test]
fn vocabulary_is_padded_and_typed_like_llama_cpp() {
    let ckpt = Checkpoint::new("vocab");
    let gguf = ckpt.convert(TargetQuant::F32);

    let tokens = gguf.get_string_array("tokenizer.ggml.tokens").unwrap();
    let types = gguf.get_i32_array("tokenizer.ggml.token_type").unwrap();
    // the embedding has VOCAB_SIZE rows, and llama.cpp sizes the vocabulary from this list
    assert_eq!(tokens.len(), VOCAB_SIZE);
    assert_eq!(types.len(), VOCAB_SIZE);
    assert_eq!(tokens[262], "[PAD262]");
    assert_eq!(tokens[VOCAB_SIZE - 1], "[PAD299]");
    assert!(types[262..].iter().all(|&t| t == 5), "padding is UNUSED");

    assert_eq!(&types[..5], &[3, 3, 3, 3, 4]);
    assert_eq!(types[5], 1, "byte tokens are normal");
}

#[test]
fn bos_and_template_come_from_the_checkpoint() {
    let ckpt = Checkpoint::new("bos");
    let gguf = ckpt.convert(TargetQuant::F32);

    // the post-processor's template opens with the BOS token
    assert_eq!(gguf.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
    assert_eq!(gguf.get_bool("tokenizer.ggml.add_eos_token"), None);
    // Transformers 5 keeps the template in chat_template.jinja, not tokenizer_config.json
    assert!(
        gguf.get_str("tokenizer.chat_template")
            .is_some_and(|t| t.contains("messages"))
    );
}

/// What each Hugging Face tensor is called in a llama.cpp LFM2 GGUF, written out
/// independently of the converter's own mapping.
fn llama_cpp_name(hf: &str) -> String {
    let fixed = [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.embedding_norm.weight", "token_embd_norm.weight"),
    ];
    if let Some((_, gguf)) = fixed.iter().find(|(name, _)| *name == hf) {
        return gguf.to_string();
    }
    let rest = hf.strip_prefix("model.layers.").expect(hf);
    let (layer, tail) = rest.split_once('.').unwrap();
    let suffix = match tail {
        "operator_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "feed_forward.w1.weight" => "ffn_gate.weight",
        "feed_forward.w3.weight" => "ffn_up.weight",
        "feed_forward.w2.weight" => "ffn_down.weight",
        "conv.conv.weight" => "shortconv.conv.weight",
        "conv.in_proj.weight" => "shortconv.in_proj.weight",
        "conv.out_proj.weight" => "shortconv.out_proj.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.out_proj.weight" => "attn_output.weight",
        "self_attn.q_layernorm.weight" => "attn_q_norm.weight",
        "self_attn.k_layernorm.weight" => "attn_k_norm.weight",
        other => panic!("no llama.cpp name for {other}"),
    };
    format!("blk.{layer}.{suffix}")
}

#[test]
fn every_tensor_carries_its_source_weights_under_the_llama_cpp_name() {
    let ckpt = Checkpoint::new("bytes");
    let gguf = ckpt.convert(TargetQuant::F32);
    let source = hf_tensors();
    assert_eq!(
        gguf.tensors.len(),
        source.len(),
        "no tensor added or dropped"
    );
    for (hf_name, (_, values)) in &source {
        let name = llama_cpp_name(hf_name);
        let want: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let got = gguf
            .tensor_data(&name)
            .unwrap_or_else(|e| panic!("{name} (from {hf_name}): {e}"));
        assert!(
            got == want.as_slice(),
            "{name} does not hold {hf_name}'s data"
        );
    }
}

#[test]
fn a_missing_tokenizer_config_still_reads_bos_from_the_template() {
    let ckpt = Checkpoint::new("noconfig");
    std::fs::remove_file(ckpt.0.join("tokenizer_config.json")).unwrap();
    let gguf = ckpt.convert(TargetQuant::F32);
    assert_eq!(gguf.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
}

#[test]
fn converted_model_loads_and_runs() {
    for quant in [TargetQuant::F32, TargetQuant::Q8_0] {
        let ckpt = Checkpoint::new("run");
        let path = {
            ckpt.convert(quant);
            ckpt.0.join("out.gguf")
        };
        let tokenizer =
            BpeTokenizer::from_gguf(&GgufFile::open(&path).unwrap()).expect("tokenizer");
        let model = Lfm2Model::from_gguf(GgufFile::open(&path).unwrap(), 256)
            .unwrap_or_else(|e| panic!("cera cannot load its own {quant:?} conversion: {e}"));
        let mut state =
            InferenceState::from_config_with_compression(model.config(), &KvCompression::None)
                .unwrap();
        let tokens = tokenizer.encode("hello there");
        assert!(!tokens.is_empty());
        let logits = tokens
            .iter()
            .map(|&t| model.forward(&[t], state.seq_len, &mut state))
            .last()
            .unwrap();
        assert_eq!(logits.len(), VOCAB_SIZE);
        assert!(logits.iter().all(|v| v.is_finite()), "{quant:?}");
    }
}

#[test]
fn moe_checkpoints_are_refused_not_mislabelled() {
    let ckpt = Checkpoint::new("moe");
    let mut config = config_json();
    config["model_type"] = json!("lfm2_moe");
    config["architectures"] = json!(["Lfm2MoeForCausalLM"]);
    std::fs::write(ckpt.0.join("config.json"), config.to_string()).unwrap();
    let err = quantize_safetensors_to_gguf(&ckpt.0, &ckpt.0.join("out.gguf"), TargetQuant::F32)
        .expect_err("MoE conversion is not implemented");
    assert!(err.to_string().contains("LFM2-MoE"), "{err}");
}

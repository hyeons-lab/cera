//! `cera::convert` of an LFM3 checkpoint must write the GGUF llama.cpp's converter
//! writes for it, so both runtimes load it.
//!
//! A tiny Hugging Face LFM3 checkpoint (the layout of the 1.7B: sandwich norms, an
//! attention gate, conv / sliding / global layers, tied embeddings, a null `head_dim`,
//! partial rotary on the global layers) is converted and checked against what the
//! llama.cpp converter was observed to write for the real 1.7B checkpoint (every one
//! of its 287 tensors, the token list and all `lfm3.*` keys came out identical), then
//! loaded and run. The audio-checkpoint layout (language model under `lfm.`, next to
//! audio tensors, config under `lfm`) must convert to the same file.

#![cfg(all(feature = "std-fs", feature = "mmap"))]

use std::path::{Path, PathBuf};

use cera::convert::pipeline::quantize_safetensors_to_gguf;
use cera::convert::quantize::TargetQuant;
use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::Model;
use cera::model::lfm2::LfmModel;
use cera::tokenizer::BpeTokenizer;
use serde_json::json;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FF: usize = 128;
const L_CACHE: usize = 7;
const WINDOW: usize = 4;
const VOCAB_SIZE: usize = 300;
const LAYER_TYPES: [&str; 6] = [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "sliding_attention",
    "full_attention",
];
const BOS: &str = "<|startoftext|>";

type Values = (Vec<usize>, Vec<f32>);

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

/// Five added tokens, 256 byte tokens and one merge: 262 ids, short of `VOCAB_SIZE`.
/// Its post-processor adds no BOS, like the real checkpoint's.
fn tokenizer_json() -> serde_json::Value {
    let added = [
        ("<|pad|>", true),
        (BOS, true),
        ("<|im_end|>", true),
        ("<|tool_call_start|>", false),
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
    vocab.insert("he".into(), json!(added.len() + 256));
    json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e"]},
        "added_tokens": added_tokens,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true},
    })
}

fn text_config() -> serde_json::Value {
    json!({
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "conv_L_cache": L_CACHE,
        "conv_bias": false,
        "head_dim": null,
        "hidden_size": HIDDEN,
        "intermediate_size": FF,
        "layer_types": LAYER_TYPES,
        "max_position_embeddings": 512,
        "norm_eps": 1e-5,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYER_TYPES.len(),
        "num_key_value_heads": KV_HEADS,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default",
            },
            // not the loader's 10000 default, so a loader that ignores the key moves the goldens
            "sliding_attention": {"rope_theta": 20000, "rope_type": "default"},
        },
        "sliding_window": WINDOW,
        "tie_word_embeddings": true,
        "use_sandwich_norm": true,
        "vocab_size": VOCAB_SIZE,
    })
}

/// The checkpoint of an audio model: the language model's config under `lfm`.
fn audio_config() -> serde_json::Value {
    json!({
        "architectures": ["Lfm3AudioModel"],
        "model_type": "lfm3_audio",
        "codebooks": 12,
        "lfm": text_config(),
    })
}

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

    fn norm(&mut self, n: usize) -> Values {
        (vec![n], (0..n).map(|_| 1.0 + self.next(0.1)).collect())
    }
}

/// The checkpoint's tensors under their Hugging Face names.
fn hf_tensors() -> Vec<(String, Values)> {
    let mut w = Weights(0x9E37_79B9_7F4A_7C15);
    let mut t = vec![
        (
            "model.embed_tokens.weight".to_string(),
            w.tensor(&[VOCAB_SIZE, HIDDEN], 0.3),
        ),
        // the final norm, under the name LFM2 also gives it
        ("model.embedding_norm.weight".to_string(), w.norm(HIDDEN)),
    ];
    for (i, kind) in LAYER_TYPES.iter().enumerate() {
        let p = format!("model.layers.{i}");
        for norm in [
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ] {
            t.push((format!("{p}.{norm}.weight"), w.norm(HIDDEN)));
        }
        t.push((
            format!("{p}.feed_forward.w1.weight"),
            w.tensor(&[FF, HIDDEN], 0.08),
        ));
        t.push((
            format!("{p}.feed_forward.w3.weight"),
            w.tensor(&[FF, HIDDEN], 0.08),
        ));
        t.push((
            format!("{p}.feed_forward.w2.weight"),
            w.tensor(&[HIDDEN, FF], 0.08),
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
                format!("{p}.self_attn.o_proj.weight"),
                w.tensor(&[HIDDEN, HEADS * HEAD_DIM], 0.08),
            ));
            t.push((format!("{p}.self_attn.q_norm.weight"), w.norm(HEAD_DIM)));
            t.push((format!("{p}.self_attn.k_norm.weight"), w.norm(HEAD_DIM)));
            // a strong gate, so a mis-routed one cannot hide
            t.push((
                format!("{p}.self_attn.gate_proj.weight"),
                w.tensor(&[HEADS, HIDDEN], 0.5),
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

/// A checkpoint directory that deletes itself.
struct Checkpoint(PathBuf);

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    /// A standalone `Lfm3ForCausalLM` checkpoint.
    Text,
    /// The audio model's: `lfm.`-prefixed language model, `lfm` config, audio tensors.
    Audio,
}

impl Checkpoint {
    fn new(tag: &str, layout: Layout) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cera_convert_lfm3_{tag}_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // owns the directory from here, so a write that fails part-way still removes it
        let checkpoint = Self(dir.clone());
        let config = if layout == Layout::Audio {
            audio_config()
        } else {
            text_config()
        };
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            json!({"bos_token": BOS, "eos_token": "<|im_end|>"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
        )
        .unwrap();
        let mut tensors = hf_tensors();
        if layout == Layout::Audio {
            for (name, _) in tensors.iter_mut() {
                *name = format!("lfm.{name}");
            }
            let mut w = Weights(7);
            // an untied head, and the audio side of the checkpoint
            tensors.push((
                "lfm.lm_head.weight".into(),
                w.tensor(&[VOCAB_SIZE, HIDDEN], 0.3),
            ));
            for name in [
                "encoder.lfm.layers.0.operator_norm.weight",
                "mlp.model.0.weight",
                "codebook_embedding.weight",
                "depth_embeddings.0.embedding_norm.weight",
                "depth_linear.weight",
                "depthformer.model.layers.0.ffn_norm.weight",
                "mask_embedding",
            ] {
                tensors.push((name.into(), w.tensor(&[HIDDEN], 1.0)));
            }
        }
        write_safetensors(&dir.join("model.safetensors"), &tensors);
        if layout == Layout::Audio {
            // the audio bundle's detokenizer ships beside the language model; it is not a
            // shard of it, so a converter that read it would fail on this content
            std::fs::write(
                dir.join("detokenizer.safetensors"),
                b"not a safetensors file",
            )
            .unwrap();
        }
        checkpoint
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
    let gguf = Checkpoint::new("meta", Layout::Text).convert(TargetQuant::F32);

    assert_eq!(gguf.architecture(), Some("lfm3"));
    assert_eq!(
        gguf.get_i32_array("lfm3.attention.head_count_kv"),
        Some(vec![0, 0, 2, 0, 2, 2])
    );
    // lfm3 does not auto-adjust the MLP width
    assert_eq!(gguf.get_u32("lfm3.feed_forward_length"), Some(FF as u32));
    assert_eq!(gguf.get_u32("lfm3.shortconv.l_cache"), Some(L_CACHE as u32));
    // `head_dim: null` falls back to hidden_size / heads
    assert_eq!(
        gguf.get_u32("lfm3.attention.key_length"),
        Some(HEAD_DIM as u32)
    );
    assert_eq!(
        gguf.get_u32("lfm3.attention.value_length"),
        Some(HEAD_DIM as u32)
    );
    // the whole head is "rotary"; partial rotary lives in rope_freqs.weight
    assert_eq!(
        gguf.get_u32("lfm3.rope.dimension_count"),
        Some(HEAD_DIM as u32)
    );
    assert_eq!(gguf.get_f32("lfm3.rope.freq_base"), Some(1_000_000.0));
    assert_eq!(gguf.get_f32("lfm3.rope.freq_base_swa"), Some(20_000.0));
    assert_eq!(
        gguf.get_u32("lfm3.attention.sliding_window"),
        Some(WINDOW as u32)
    );
    assert_eq!(
        gguf.get_bool_array("lfm3.attention.sliding_window_pattern"),
        Some(vec![false, false, false, false, true, false])
    );
    assert_eq!(
        gguf.get_f32("lfm3.attention.layer_norm_rms_epsilon"),
        Some(1e-5)
    );
    // the tokenizer does not add BOS; the chat template does
    assert_eq!(gguf.get_bool("tokenizer.ggml.add_bos_token"), Some(false));
    assert!(gguf.get_str("tokenizer.chat_template").is_some());
}

/// The GGUF name of each Hugging Face tensor, written out independently of the
/// converter's own mapping.
fn llama_cpp_name(hf: &str) -> String {
    let fixed = [
        ("model.embed_tokens.weight", "token_embd.weight"),
        // lfm3 calls the tensor LFM2 names token_embd_norm its output norm
        ("model.embedding_norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ];
    if let Some((_, gguf)) = fixed.iter().find(|(name, _)| *name == hf) {
        return gguf.to_string();
    }
    let rest = hf.strip_prefix("model.layers.").expect(hf);
    let (layer, tail) = rest.split_once('.').unwrap();
    let suffix = match tail {
        "operator_norm.weight" => "attn_norm.weight",
        "post_operator_norm.weight" => "post_attention_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "post_ffn_norm.weight" => "post_ffw_norm.weight",
        "feed_forward.w1.weight" => "ffn_gate.weight",
        "feed_forward.w3.weight" => "ffn_up.weight",
        "feed_forward.w2.weight" => "ffn_down.weight",
        "conv.conv.weight" => "shortconv.conv.weight",
        "conv.in_proj.weight" => "shortconv.in_proj.weight",
        "conv.out_proj.weight" => "shortconv.out_proj.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "self_attn.gate_proj.weight" => "attn_gate.weight",
        other => panic!("no llama.cpp name for {other}"),
    };
    format!("blk.{layer}.{suffix}")
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Every source tensor, under its llama.cpp name, with its own data; plus exactly one
/// generated tensor, `rope_freqs.weight`.
fn check_tensors(gguf: &GgufFile, source: &[(String, Values)]) {
    assert_eq!(
        gguf.tensors.len(),
        source.len() + 1,
        "the source tensors plus rope_freqs.weight"
    );
    for (hf_name, (_, values)) in source {
        let name = llama_cpp_name(hf_name.strip_prefix("lfm.").unwrap_or(hf_name));
        let got = gguf
            .tensor_data(&name)
            .unwrap_or_else(|e| panic!("{name} (from {hf_name}): {e}"));
        assert!(
            got == le_bytes(values).as_slice(),
            "{name} does not hold {hf_name}'s data"
        );
    }
    for layer in [0, 1, 3] {
        assert_eq!(
            gguf.tensors[&format!("blk.{layer}.shortconv.conv.weight")].shape,
            vec![L_CACHE, HIDDEN],
            "llama.cpp wants the conv kernel 2-D"
        );
    }
    assert_eq!(
        gguf.tensors["blk.2.attn_gate.weight"].shape,
        vec![HIDDEN, HEADS]
    );
}

#[test]
fn tensors_use_llama_cpp_names_and_carry_the_source_weights() {
    let gguf = Checkpoint::new("tensors", Layout::Text).convert(TargetQuant::F32);
    check_tensors(&gguf, &hf_tensors());
    // tied embeddings: no output.weight
    assert!(!gguf.tensors.contains_key("output.weight"));
}

#[test]
fn rope_freqs_masks_the_unrotated_pairs() {
    // partial_rotary_factor 0.5 on a 16-wide head: 4 of 8 pairs rotate
    for quant in [TargetQuant::F32, TargetQuant::F16, TargetQuant::Q8_0] {
        let gguf = Checkpoint::new("rope", Layout::Text).convert(quant);
        let info = &gguf.tensors["rope_freqs.weight"];
        assert_eq!(info.shape, vec![HEAD_DIM / 2], "{quant:?}");
        let want = le_bytes(&[1.0, 1.0, 1.0, 1.0, 1e30, 1e30, 1e30, 1e30]);
        // F32 even in an F16 file: 1e30 does not fit F16
        assert_eq!(
            gguf.tensor_data("rope_freqs.weight").unwrap(),
            want.as_slice(),
            "{quant:?}"
        );
    }
}

#[test]
fn the_audio_checkpoint_layout_converts_to_the_same_language_model() {
    let ckpt = Checkpoint::new("audio", Layout::Audio);
    let gguf = ckpt.convert(TargetQuant::F32);
    assert_eq!(gguf.architecture(), Some("lfm3"));
    assert_eq!(
        gguf.get_i32_array("lfm3.attention.head_count_kv"),
        Some(vec![0, 0, 2, 0, 2, 2]),
        "config read from the nested `lfm`"
    );

    // the language model's tensors, the untied head, and nothing of the audio side
    let mut source: Vec<(String, Values)> = hf_tensors();
    for (name, _) in source.iter_mut() {
        *name = format!("lfm.{name}");
    }
    let head = Weights(7).tensor(&[VOCAB_SIZE, HIDDEN], 0.3);
    source.push(("lfm.lm_head.weight".into(), head));
    check_tensors(&gguf, &source);
    for audio in ["encoder", "mlp", "depth", "codebook", "mask", "lfm."] {
        assert!(
            !gguf.tensors.keys().any(|n| n.starts_with(audio)),
            "{audio}* leaked into the language-model GGUF"
        );
    }
}

#[test]
fn converted_models_load_and_run() {
    for (layout, quant) in [
        (Layout::Text, TargetQuant::F32),
        (Layout::Text, TargetQuant::Q8_0),
        (Layout::Text, TargetQuant::F16),
        (Layout::Audio, TargetQuant::F32),
    ] {
        let ckpt = Checkpoint::new("run", layout);
        ckpt.convert(quant);
        let path = ckpt.0.join("out.gguf");
        let tokenizer =
            BpeTokenizer::from_gguf(&GgufFile::open(&path).unwrap()).expect("tokenizer");
        let model = LfmModel::from_gguf(GgufFile::open(&path).unwrap(), 256)
            .unwrap_or_else(|e| panic!("cera cannot load its own {quant:?} conversion: {e}"));
        let mut state =
            InferenceState::from_config_with_compression(model.config(), &KvCompression::None)
                .unwrap();
        // longer than the 4-token window, so the sliding layer masks something
        let tokens = tokenizer.encode("hello there, this is a longer prompt");
        assert!(tokens.len() > WINDOW);
        let logits = tokens
            .iter()
            .map(|&t| model.forward(&[t], state.seq_len, &mut state))
            .last()
            .unwrap();
        assert_eq!(logits.len(), VOCAB_SIZE);
        assert!(logits.iter().all(|v| v.is_finite()), "{quant:?}");
        // what lfm3 cannot honour, pinned here without python (the oracle test that also
        // checks it skips in CI): LoRA and the n_keep shift are refused, TurboQuant falls back
        assert!(!model.supports_lora(), "{quant:?}");
        assert!(!model.supports_kv_shift(), "{quant:?}");
        assert!(!model.turboquant_supported(), "{quant:?}");
        // the request itself is served (with the plain cache) rather than failing the session
        model
            .configure_kv_compression(&KvCompression::turboquant(1))
            .unwrap_or_else(|e| panic!("{quant:?}: a TurboQuant request failed: {e}"));
    }
}

/// Largest absolute difference between two logit rows, NaN when either holds a NaN or
/// infinity (`f32::max` would drop it, and a `worst < tol` gate would then pass a NaN row).
/// The rows must be the same length.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared rows differ in length");
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return f32::NAN;
        }
        worst = worst.max((x - y).abs());
    }
    worst
}

#[test]
fn max_abs_diff_is_nan_for_non_finite_rows() {
    assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.5, 2.0]), 0.5);
    assert!(max_abs_diff(&[f32::NAN, 2.0], &[1.0, 2.0]).is_nan());
    assert!(max_abs_diff(&[1.0], &[f32::INFINITY]).is_nan());
    // a NaN is unordered, so the `worst < tol` gates cannot pass it
    assert!(
        max_abs_diff(&[f32::NAN], &[0.0])
            .partial_cmp(&0.0)
            .is_none()
    );
}

/// Without `has_blas` (or on `F32` weights), batched prefill and per-token decode run the same
/// arithmetic and agree within `1e-3`. With `has_blas`, `Q8_0` prefill runs `f32` SGEMM while
/// decode quantizes activations to `Q8_0` for int8 GEMV (see `lfm2_decode_prefill_identity`).
fn prefill_decode_tol(quant: TargetQuant) -> f32 {
    if cfg!(has_blas) && quant == TargetQuant::Q8_0 {
        0.2
    } else {
        1e-3
    }
}

/// The converted model's batched prefill must agree with its token-by-token decode on a
/// prompt longer than the window: the converted metadata (window, rope bases, the
/// partial-rotary mask) drives both, and they diverge if any of it is read wrongly.
#[test]
fn converted_model_prefill_agrees_with_decode() {
    let ckpt = Checkpoint::new("prefill", Layout::Text);
    ckpt.convert(TargetQuant::Q8_0);
    let path = ckpt.0.join("out.gguf");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    let model = LfmModel::from_gguf(GgufFile::open(&path).unwrap(), 256).unwrap();
    let tokens = tokenizer.encode("hello there, this is a longer prompt for the window");
    let fresh = || {
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap()
    };
    let mut decode = fresh();
    let decoded = tokens
        .iter()
        .map(|&t| model.forward(&[t], decode.seq_len, &mut decode))
        .last()
        .unwrap();
    let mut prefill = fresh();
    let prefilled = model.forward_prefill(&tokens, 0, &mut prefill);
    let worst = max_abs_diff(&decoded, &prefilled);
    let tol = prefill_decode_tol(TargetQuant::Q8_0);
    assert!(worst < tol, "prefill and decode logits differ by {worst}");
}

/// Prefill the 48-token golden prompt through a deterministic F32 conversion of `layout`
/// and return the top-3 token ids, the first four logits and the sum of all logits.
fn golden_logits(tag: &str, layout: Layout) -> Golden {
    let ckpt = Checkpoint::new(tag, layout);
    ckpt.convert(TargetQuant::F32);
    let path = ckpt.0.join("out.gguf");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    let model = LfmModel::from_gguf(GgufFile::open(&path).unwrap(), 256).unwrap();
    let tokens = tokenizer.encode("hello there, this is a longer prompt for the window");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits = model.forward_prefill(&tokens, 0, &mut state);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    Golden {
        n_tokens: tokens.len(),
        top3: order[..3].to_vec(),
        head: logits[..4].to_vec(),
        sum: logits.iter().sum(),
    }
}

/// What the golden prompt produces: its length, the top-3 token ids, the first four
/// logits and the sum of all logits.
struct Golden {
    n_tokens: usize,
    top3: Vec<usize>,
    head: Vec<f32>,
    sum: f32,
}

fn assert_golden(got: Golden, top3: [usize; 3], head: [f32; 4], sum: f32) {
    assert_eq!(
        got.n_tokens, 48,
        "the prompt must outgrow the 4-token window"
    );
    // as a set: the top two logits sit 1e-4 apart, inside the per-logit tolerance below
    let (mut got_ids, mut want_ids) = (got.top3.clone(), top3.to_vec());
    got_ids.sort_unstable();
    want_ids.sort_unstable();
    assert_eq!(got_ids, want_ids, "top-3 token ids");
    for (i, (got, want)) in got.head.iter().zip(head).enumerate() {
        assert!((got - want).abs() < 2e-3, "logit {i}: {got} vs {want}");
    }
    assert!(
        (got.sum - sum).abs() < 2e-2,
        "sum of logits {} vs {sum}",
        got.sum
    );
}

/// The CPU `lfm3` graph pinned without python, on a deterministic F32 conversion of the
/// tied-embedding text checkpoint: the sliding-window mask, the per-head attention gate,
/// the partial-rotary frequency mask and the sliding layers' own rope base each change the
/// logits of this prompt by far more than the tolerance, while the prefill-vs-decode test
/// above cannot see them (both paths share the attention block). The values are a
/// regression pin captured from this build (the tolerance also holds on x86_64); the
/// llama.cpp comparison is `lfm3_oracle_parity`, wherever python has numpy and gguf.
#[test]
fn converted_f32_model_matches_pinned_logits() {
    assert_golden(
        golden_logits("golden", Layout::Text),
        [140, 296, 117],
        [-1.969926, 1.1184531, 1.0317479, -1.0103271],
        11.101996,
    );
}

/// The same pin on the audio checkpoint's layout, whose language model has an untied
/// `output.weight`: a loader that ignored it (silently tying the head to the embedding)
/// moves these logits, which the tied text checkpoint cannot show.
#[test]
fn converted_f32_untied_head_model_matches_pinned_logits() {
    assert_golden(
        golden_logits("golden-untied", Layout::Audio),
        [195, 287, 229],
        [-0.92816746, -0.4836746, -0.45933115, -0.87254345],
        16.896694,
    );
}

/// The untied `output.weight` reaches every logits entry point, not only batched prefill:
/// decode, greedy decode and the all-logits verify each read `logits_ref` separately.
#[test]
fn untied_head_reaches_every_logits_path() {
    // the F32 file takes the plain gemv sites; Q8_0 takes the batched quantized ones
    for quant in [TargetQuant::F32, TargetQuant::Q8_0] {
        let ckpt = Checkpoint::new(&format!("untied-paths-{quant:?}"), Layout::Audio);
        ckpt.convert(quant);
        let path = ckpt.0.join("out.gguf");
        let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
        let model = LfmModel::from_gguf(GgufFile::open(&path).unwrap(), 256).unwrap();
        let tokens = tokenizer.encode("hello there, this is a longer prompt for the window");
        let n = tokens.len();
        let fresh = || {
            InferenceState::from_config_with_compression(model.config(), &KvCompression::None)
                .unwrap()
        };
        let diff = max_abs_diff;
        let prefilled = model.forward_prefill(&tokens, 0, &mut fresh());

        let tol = prefill_decode_tol(quant);
        let mut st = fresh();
        let decoded = tokens
            .iter()
            .map(|&t| model.forward(&[t], st.seq_len, &mut st))
            .last()
            .unwrap();
        assert!(diff(&decoded, &prefilled) < tol, "decode head");

        let mut st = fresh();
        let _ = model.forward_prefill(&tokens[..n - 1], 0, &mut st);
        let greedy = model.forward_greedy(&tokens[n - 1..], n - 1, &mut st);
        // The greedy head and the prefill head are the same arithmetic up to `tol` (a Q8_0 decode
        // and prefill differ under BLAS), so a near tie may flip the pick: require the picked
        // token to be a top-scoring one within that tolerance, not the identical index.
        let best = prefilled[cera::sampler::argmax(&prefilled) as usize];
        let picked = prefilled[greedy as usize];
        assert!(
            picked >= best - tol * best.abs().max(1.0),
            "greedy head picked a token {picked} below the best {best}"
        );

        let v = model.config().vocab_size;
        let all = model.forward_prefill_logits_all(&tokens, 0, &mut fresh());
        assert!(
            diff(&all[(n - 1) * v..], &prefilled) < 1e-3,
            "all-logits head"
        );

        // the embedding-input path (audio and vision hosts) projects through the same head
        let hs = model.config().hidden_size;
        let table = GgufFile::open(&path)
            .unwrap()
            .get_tensor("token_embd.weight")
            .unwrap()
            .to_f32_vec();
        let row = tokens[n - 1] as usize * hs;
        let mut st = fresh();
        let _ = model.forward_prefill(&tokens[..n - 1], 0, &mut st);
        let embedded = model.forward_from_embedding(&table[row..row + hs], n - 1, &mut st);
        assert!(diff(&embedded, &prefilled) < tol, "embedding-input head");
    }
}

/// Two prefill chunks (the second at `start_pos > 0`) then a rewind into the second chunk,
/// then decode the tail: the multi-tap conv's `ConvHistory` must carry absolute positions.
#[test]
fn chunked_prefill_then_rewind_matches_decode() {
    let ckpt = Checkpoint::new("chunkrew", Layout::Text);
    ckpt.convert(TargetQuant::Q8_0);
    let path = ckpt.0.join("out.gguf");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    let model = LfmModel::from_gguf(GgufFile::open(&path).unwrap(), 256).unwrap();
    let tokens = tokenizer.encode("hello there, this is a longer prompt for the window");
    let fresh = || {
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap()
    };
    let mut decode = fresh();
    let decoded = tokens
        .iter()
        .map(|&t| model.forward(&[t], decode.seq_len, &mut decode))
        .last()
        .unwrap();
    let split = 20;
    let mut st = fresh();
    let _ = model.forward_prefill(&tokens[..split], 0, &mut st);
    let _ = model.forward_prefill(&tokens[split..], split, &mut st);
    model.try_truncate_kv(&mut st, split + 3).unwrap();
    let mut last = Vec::new();
    for &t in &tokens[split + 3..] {
        last = model.forward(&[t], st.seq_len, &mut st);
    }
    let worst = max_abs_diff(&decoded, &last);
    let tol = prefill_decode_tol(TargetQuant::Q8_0);
    assert!(worst < tol, "chunked prefill + rewind differs by {worst}");
}

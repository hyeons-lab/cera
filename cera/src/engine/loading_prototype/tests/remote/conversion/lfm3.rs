//! The streaming converter writes the llama.cpp LFM3 layout: the same checks
//! `tests/convert_lfm3_layout.rs` makes of the local converter, through the HTTP path.

use super::super::http::Context;
use super::lfm2::{options, routes_with};
use super::*;
use crate::bundle::HfSpec;
use crate::convert::stream_quantize_hf_repo;

const REPO: &str = "lfm3";
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FF: usize = 64;
const L_CACHE: usize = 7;

/// A tiny conv + global-attention checkpoint under the Hugging Face names, plus one
/// tensor of the audio model's encoder that must not reach the GGUF.
fn shard() -> Vec<u8> {
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![8, HIDDEN]),
        ("model.embedding_norm.weight".into(), vec![HIDDEN]),
        (
            "encoder.lfm.layers.0.operator_norm.weight".into(),
            vec![HIDDEN],
        ),
    ];
    for (layer, conv) in [(0, true), (1, false)] {
        let p = format!("model.layers.{layer}");
        for norm in [
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ] {
            tensors.push((format!("{p}.{norm}.weight"), vec![HIDDEN]));
        }
        tensors.push((format!("{p}.feed_forward.w1.weight"), vec![FF, HIDDEN]));
        tensors.push((format!("{p}.feed_forward.w3.weight"), vec![FF, HIDDEN]));
        tensors.push((format!("{p}.feed_forward.w2.weight"), vec![HIDDEN, FF]));
        if conv {
            tensors.push((format!("{p}.conv.conv.weight"), vec![HIDDEN, 1, L_CACHE]));
            tensors.push((format!("{p}.conv.in_proj.weight"), vec![3 * HIDDEN, HIDDEN]));
            tensors.push((format!("{p}.conv.out_proj.weight"), vec![HIDDEN, HIDDEN]));
        } else {
            tensors.push((format!("{p}.self_attn.q_proj.weight"), vec![HIDDEN, HIDDEN]));
            tensors.push((
                format!("{p}.self_attn.k_proj.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
            ));
            tensors.push((
                format!("{p}.self_attn.v_proj.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
            ));
            tensors.push((format!("{p}.self_attn.o_proj.weight"), vec![HIDDEN, HIDDEN]));
            tensors.push((format!("{p}.self_attn.q_norm.weight"), vec![HEAD_DIM]));
            tensors.push((format!("{p}.self_attn.k_norm.weight"), vec![HEAD_DIM]));
            tensors.push((
                format!("{p}.self_attn.gate_proj.weight"),
                vec![HEADS, HIDDEN],
            ));
        }
    }
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let start = data.len();
        let count: usize = shape.iter().product();
        for i in 0..count {
            data.extend_from_slice(&((i % 7) as f32 * 0.25 + 1.0).to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype":"F32", "shape":shape, "data_offsets":[start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    bytes
}

fn hf_config() -> serde_json::Value {
    json!({
        "model_type": "lfm3", "architectures": ["Lfm3ForCausalLM"],
        "hidden_size": HIDDEN, "num_hidden_layers": 2,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": null, "intermediate_size": FF, "vocab_size": 8,
        "norm_eps": 1e-5, "conv_L_cache": L_CACHE, "max_position_embeddings": 256,
        "layer_types": ["conv", "full_attention"],
        "rope_parameters": {"full_attention": {
            "rope_theta": 1000000.0, "partial_rotary_factor": 0.5, "rope_type": "default"}},
        "tie_word_embeddings": true,
        "bos_token_id": 1, "eos_token_id": 2,
    })
}

fn routes() -> HashMap<String, Response> {
    routes_with(
        REPO,
        vec![
            ("model.safetensors".into(), shard()),
            // a side model next to the language model (the audio bundle's detokenizer): not a
            // shard of it, so a converter that fetched and parsed it would fail on this content
            (
                "detokenizer/model.safetensors".into(),
                b"not a safetensors file".to_vec(),
            ),
        ],
        hf_config(),
    )
}

fn convert(ctx: &Context) -> GgufFile {
    let progress = Arc::new(Progress::default());
    let cfg = config(&ctx.root, &progress);
    let spec = HfSpec::parse(&format!("fixture/{REPO}:F32")).unwrap();
    stream_quantize_hf_repo(&spec, options(&cfg, progress)).unwrap();
    GgufFile::open(&cache_dir(&cfg, REPO, "F32").join("model.gguf")).unwrap()
}

#[test]
fn streaming_conversion_writes_the_llama_cpp_lfm3_layout() {
    isolated_with_ranges(
        "conversion::lfm3::streaming_conversion_writes_the_llama_cpp_lfm3_layout",
        routes,
        |ctx| {
            let gguf = convert(&ctx);

            assert_eq!(gguf.architecture(), Some("lfm3"));
            assert_eq!(
                gguf.get_i32_array("lfm3.attention.head_count_kv"),
                Some(vec![0, KV_HEADS as i32])
            );
            assert_eq!(gguf.get_u32("lfm3.shortconv.l_cache"), Some(L_CACHE as u32));
            assert_eq!(gguf.get_f32("lfm3.rope.freq_base"), Some(1_000_000.0));
            // the tokenizer's post-processor adds a BOS, but lfm3 is written without one
            assert_eq!(gguf.get_bool("tokenizer.ggml.add_bos_token"), Some(false));
            assert_eq!(
                gguf.get_u32("lfm3.attention.key_length"),
                Some(HEAD_DIM as u32)
            );

            // llama.cpp's names, the generated partial-rotary table (the unrotated
            // pairs carry a huge factor) and none of the audio model's tensors
            for name in [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.post_attention_norm.weight",
                "blk.0.post_ffw_norm.weight",
                "blk.1.attn_gate.weight",
            ] {
                assert!(gguf.tensors.contains_key(name), "missing {name}");
            }
            assert_eq!(
                gguf.tensors["blk.0.shortconv.conv.weight"].shape,
                vec![L_CACHE, HIDDEN]
            );
            let want: Vec<u8> = [1.0f32, 1.0, 1.0, 1.0, 1e30, 1e30, 1e30, 1e30]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(gguf.tensor_data("rope_freqs.weight").unwrap(), &want[..]);
            assert!(
                gguf.tensors.keys().all(|n| !n.contains("encoder")),
                "an audio tensor reached the GGUF: {:?}",
                gguf.tensors.keys().collect::<Vec<_>>()
            );
        },
        |_requests, _ranges| {},
    );
}

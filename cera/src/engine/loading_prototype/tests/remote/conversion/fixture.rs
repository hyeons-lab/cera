use super::*;

pub(super) const TEMPLATE: &str = "{% for message in messages %}{{ message.content }}{% endfor %}";

// Explicit pairs keep the expected GGUF independent of the converter's name mapper.
const NAMES: [(&str, &str); 11] = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.norm.weight", "output_norm.weight"),
    (
        "model.layers.0.input_layernorm.weight",
        "blk.0.attn_norm.weight",
    ),
    (
        "model.layers.0.post_attention_layernorm.weight",
        "blk.0.ffn_norm.weight",
    ),
    (
        "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_q.weight",
    ),
    (
        "model.layers.0.self_attn.k_proj.weight",
        "blk.0.attn_k.weight",
    ),
    (
        "model.layers.0.self_attn.v_proj.weight",
        "blk.0.attn_v.weight",
    ),
    (
        "model.layers.0.self_attn.o_proj.weight",
        "blk.0.attn_output.weight",
    ),
    (
        "model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_gate.weight",
    ),
    ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
    (
        "model.layers.0.mlp.down_proj.weight",
        "blk.0.ffn_down.weight",
    ),
];

pub(super) fn source() -> GgufFile {
    GgufFile::from_bytes(super::super::super::companion_fixture::primary()).unwrap()
}

pub(super) fn shards(split: bool) -> Vec<(String, Vec<u8>)> {
    let source = source();
    let groups: Vec<_> = if split {
        vec![&NAMES[..5], &NAMES[5..]]
    } else {
        vec![&NAMES[..]]
    };
    groups
        .iter()
        .enumerate()
        .map(|(i, group)| {
            let mut header = serde_json::Map::new();
            let mut data = Vec::new();
            for &(hf, gguf) in *group {
                let start = data.len();
                data.extend_from_slice(source.tensor_data(gguf).unwrap());
                let shape: Vec<_> = source.tensors[gguf].shape.iter().rev().copied().collect();
                header.insert(
                    hf.into(),
                    json!({"dtype":"F32", "shape":shape, "data_offsets":[start,data.len()]}),
                );
            }
            let header = serde_json::to_vec(&header).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(&data);
            let name = if split {
                format!("model-{:05}-of-00002.safetensors", i + 1)
            } else {
                "model.safetensors".into()
            };
            (name, bytes)
        })
        .collect()
}

pub(super) fn ranges(bytes: &[u8]) -> Vec<String> {
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: std::collections::BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let mut ranges = vec!["bytes=0-7".into(), format!("bytes=8-{}", 7 + len)];
    for value in header.values() {
        let offsets = value["data_offsets"].as_array().unwrap();
        ranges.push(format!(
            "bytes={}-{}",
            8 + len + offsets[0].as_u64().unwrap() as usize,
            7 + len + offsets[1].as_u64().unwrap() as usize
        ));
    }
    ranges
}

pub(super) fn commit(revision: &str) -> &str {
    match revision {
        "main" => "1111111111111111111111111111111111111111",
        "release" => "2222222222222222222222222222222222222222",
        _ => revision,
    }
}

pub(super) fn routes(
    repo: &str,
    revision: &str,
    shards: &[(String, Vec<u8>)],
    partial: bool,
) -> HashMap<String, Response> {
    let resolved = commit(revision);
    let prefix = format!("/fixture/{repo}/resolve/{resolved}/");
    let suffix = if revision == "main" {
        String::new()
    } else {
        format!("/revision/{revision}")
    };
    let mut routes = HashMap::from([
        (format!("/api/models/fixture/{repo}{suffix}"), Response::bytes(json!({"id":format!("fixture/{repo}"),"sha":resolved,"siblings": shards.iter().rev().map(|(name,_)| json!({"rfilename":name})).collect::<Vec<_>>(), "pipeline_tag":"text-generation"}).to_string())),
        (format!("{prefix}config.json"), Response::bytes(json!({"model_type":"llama", "hidden_size":32, "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":32,"vocab_size":2,"max_position_embeddings":256}).to_string())),
        (format!("{prefix}tokenizer.json"), Response::bytes(json!({"model":{"type":"BPE", "vocab":{"a":0,"b":1},"merges":[]}}).to_string())),
        (format!("{prefix}tokenizer_config.json"), Response::bytes(json!({"chat_template":TEMPLATE}).to_string())),
        (format!("{prefix}generation_config.json"), Response::bytes(json!({"temperature":0.25,"min_p":0.15,"top_p":0.8,"top_k":3,"repetition_penalty":1.2}).to_string())),
    ]);
    for (name, bytes) in shards {
        routes.insert(
            format!("{prefix}{name}"),
            if partial {
                Response::ranged(bytes)
            } else {
                Response::bytes(bytes)
            },
        );
    }
    routes
}

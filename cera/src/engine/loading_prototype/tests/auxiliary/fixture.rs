use super::*;

// Full one-block, nonzero F32 weights. The draft shares this base's embedding
// and tied output table. A larger context also fits a 64-token image and text.
pub(super) fn primary() -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");
    writer.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
    dimensions(&mut writer, "llama");
    writer.add_u32("llama.context_length", 256);
    let mut tensors = transformer();
    tensors.push(("token_embd.weight".into(), vec![32, 2]));
    finish(writer, tensors, 7, None)
}

// LFM2 implements embedding input; the dense Llama draft target does not.
pub(super) fn vision_primary() -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "lfm2");
    writer.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
    dimensions(&mut writer, "lfm2");
    writer.add_u32("lfm2.context_length", 256);
    let mut tensors = transformer();
    tensors[0].0 = "token_embd_norm.weight".into();
    tensors.push(("token_embd.weight".into(), vec![32, 2]));
    tensors.push(("blk.0.attn_q_norm.weight".into(), vec![32]));
    tensors.push(("blk.0.attn_k_norm.weight".into(), vec![32]));
    finish(writer, tensors, 7, None)
}

fn dimensions(writer: &mut GgufWriter, arch: &str) {
    for (key, value) in [
        ("block_count", 1),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("vocab_size", 2),
    ] {
        if arch == "lfm2" && key == "attention.head_count_kv" {
            writer.add_i32_array(format!("{arch}.{key}"), vec![value as i32]);
        } else {
            writer.add_u32(format!("{arch}.{key}"), value);
        }
    }
}

fn transformer() -> Vec<(String, Vec<u64>)> {
    let mut tensors = vec![
        ("output_norm.weight".into(), vec![32]),
        ("blk.0.attn_norm.weight".into(), vec![32]),
        ("blk.0.ffn_norm.weight".into(), vec![32]),
    ];
    for name in [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ] {
        tensors.push((format!("blk.0.{name}.weight"), vec![32, 32]));
    }
    tensors
}

// A rank-one Markov head makes the two drafts observably select different
// tokens while still executing the real attention/FFN and base output head.
pub(super) fn draft(token: u32, block_size: u32) -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "dspark");
    dimensions(&mut writer, "dspark");
    writer.add_u32("dspark.block_size", block_size);
    writer.add_u32("dspark.markov_rank", 1);
    let mut tensors = transformer();
    tensors.push(("dspark.markov_a.weight".into(), vec![1, 2]));
    tensors.push(("dspark.markov_b.weight".into(), vec![1, 2]));
    finish(writer, tensors, 29, Some(token))
}

pub(super) fn vision(seed: u32) -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "clip");
    writer.add_bool("clip.has_vision_encoder", true);
    for (key, value) in [
        ("block_count", 1),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("image_size", 8),
        ("patch_size", 1),
        ("projection_dim", 32),
        ("projector.scale_factor", 1),
    ] {
        writer.add_u32(format!("clip.vision.{key}"), value);
    }
    writer.add_f32("clip.vision.attention.layer_norm_epsilon", 1e-5);
    writer.add_f32_array("clip.vision.image_mean", vec![0.0; 3]);
    writer.add_f32_array("clip.vision.image_std", vec![1.0; 3]);
    let mut tensors = vec![
        ("v.patch_embd.weight".into(), vec![1, 1, 3, 32]),
        ("v.patch_embd.bias".into(), vec![32]),
        ("v.position_embd.weight".into(), vec![32, 64]),
        ("v.post_ln.weight".into(), vec![32]),
        ("v.post_ln.bias".into(), vec![32]),
    ];
    for name in ["ln1", "ln2"] {
        tensors.push((format!("v.blk.0.{name}.weight"), vec![32]));
        tensors.push((format!("v.blk.0.{name}.bias"), vec![32]));
    }
    for name in [
        "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down",
    ] {
        tensors.push((format!("v.blk.0.{name}.weight"), vec![32, 32]));
        tensors.push((format!("v.blk.0.{name}.bias"), vec![32]));
    }
    for name in ["mm.1", "mm.2"] {
        tensors.push((format!("{name}.weight"), vec![32, 32]));
        tensors.push((format!("{name}.bias"), vec![32]));
    }
    finish(writer, tensors, seed, None)
}

fn finish(
    mut writer: GgufWriter,
    tensors: Vec<(String, Vec<u64>)>,
    mut seed: u32,
    draft_token: Option<u32>,
) -> Arc<[u8]> {
    let mut payloads = Vec::new();
    for (name, dims) in tensors {
        let data: Vec<u8> = (0..dims.iter().product::<u64>())
            .flat_map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = if name == "dspark.markov_a.weight" {
                    1.0
                } else if name == "dspark.markov_b.weight" {
                    if Some(i as u32) == draft_token {
                        100.0
                    } else {
                        -100.0
                    }
                } else if name.contains("norm")
                    || name.ends_with("ln.weight")
                    || name.ends_with("ln1.weight")
                    || name.ends_with("ln2.weight")
                {
                    1.0
                } else if name.ends_with("bias") {
                    0.0
                } else {
                    ((seed >> 16) as f32 / 65_535.0 - 0.5) / 4.0
                };
                value.to_le_bytes()
            })
            .collect();
        writer.add_tensor(name, dims, GGML_TYPE_F32, data.len());
        payloads.push(data);
    }
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    for payload in payloads {
        writer.write_tensor_data(&mut bytes, &payload).unwrap();
    }
    bytes.into()
}

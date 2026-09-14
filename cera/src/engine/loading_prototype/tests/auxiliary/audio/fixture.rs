use super::*;

// Complete CPU geometries: one Conformer block, one depthformer block, and
// the detokenizer's fixed eight-block layout. No downloaded weights.
#[derive(Default)]
struct Fixture {
    writer: GgufWriter,
    tensors: Vec<(String, Vec<u64>)>,
}

impl Fixture {
    fn tensor(&mut self, name: impl Into<String>, dims: &[u64]) {
        self.tensors.push((name.into(), dims.to_vec()));
    }

    fn affine(&mut self, name: &str, input: u64, output: u64) {
        self.tensor(format!("{name}.weight"), &[input, output]);
        self.tensor(format!("{name}.bias"), &[output]);
    }

    fn norm(&mut self, name: &str) {
        self.tensor(format!("{name}.weight"), &[32]);
        self.tensor(format!("{name}.bias"), &[32]);
    }

    fn finish(mut self, mut seed: u32) -> Arc<[u8]> {
        let mut payloads = Vec::new();
        for (name, dims) in self.tensors {
            let data: Vec<u8> = (0..dims.iter().product::<u64>())
                .flat_map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let value = if name.ends_with("weight")
                        && (name.contains("norm") || name.contains(".ln"))
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
            self.writer
                .add_tensor(name, dims, GGML_TYPE_F32, data.len());
            payloads.push(data);
        }
        let mut bytes = Vec::new();
        self.writer
            .write_header_and_tensor_info(&mut bytes)
            .unwrap();
        for data in payloads {
            self.writer.write_tensor_data(&mut bytes, &data).unwrap();
        }
        bytes.into()
    }
}

pub(super) fn encoder(seed: u32, hidden: u64) -> Arc<[u8]> {
    let mut f = Fixture::default();
    f.writer.add_string("general.architecture", "clip");
    f.writer.add_bool("clip.has_audio_encoder", true);
    for (key, value) in [
        ("block_count", 1),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("num_mel_bins", 8),
    ] {
        f.writer.add_u32(format!("clip.audio.{key}"), value);
    }
    f.writer
        .add_f32("clip.audio.attention.layer_norm_epsilon", 1e-5);
    for i in [0, 2, 3, 5, 6] {
        let kernel = if i == 3 || i == 6 { 1 } else { 3 };
        f.tensor(format!("a.conv1d.{i}.weight"), &[kernel, kernel, 1, 1]);
        f.tensor(format!("a.conv1d.{i}.bias"), &[1]);
    }
    f.affine("a.pre_encode.out", 1, 32);
    for name in [
        "ffn_norm",
        "ffn_norm_1",
        "ln1",
        "ln2",
        "norm_conv",
        "conv_norm",
    ] {
        f.norm(&format!("a.blk.0.{name}"));
    }
    for name in [
        "ffn_up",
        "ffn_down",
        "ffn_up_1",
        "ffn_down_1",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "conv_pw2",
    ] {
        f.affine(&format!("a.blk.0.{name}"), 32, 32);
    }
    f.affine("a.blk.0.conv_pw1", 32, 64);
    f.tensor("a.blk.0.conv_dw.weight", &[3, 32]);
    f.tensor("a.blk.0.conv_dw.bias", &[32]);
    f.tensor("a.blk.0.linear_pos.weight", &[512, 32]);
    f.tensor("a.blk.0.pos_bias_u", &[32]);
    f.tensor("a.blk.0.pos_bias_v", &[32]);
    f.norm("mm.a.mlp.0");
    f.affine("mm.a.mlp.1", 32, 32);
    f.affine("mm.a.mlp.3", 32, hidden);
    f.finish(seed)
}

pub(super) fn vocoder(seed: u32, hidden: u64, decoder: bool, detok: bool) -> Arc<[u8]> {
    let mut f = Fixture::default();
    f.writer.add_string("general.architecture", "audio-fixture");
    if decoder {
        f.writer.add_u32("depthformer_n_layer", 1);
        f.writer.add_u32("depthformer_n_embd", 64);
        // The CPU loader fixes 32 query / 8 KV heads; head width two keeps RoPE active.
        let p = "depthformer.layers.0";
        for name in ["operator_norm", "ffn_norm"] {
            f.tensor(format!("{p}.{name}.weight"), &[64]);
        }
        for name in ["q", "k"] {
            f.tensor(
                format!("{p}.operator.attention.{name}_layernorm.weight"),
                &[2],
            );
        }
        f.tensor(format!("{p}.operator.qkv_proj.weight"), &[64, 96]);
        f.tensor(format!("{p}.operator.out_proj.weight"), &[64, 64]);
        for name in ["w1", "w2", "w3"] {
            f.tensor(format!("{p}.feed_forward.{name}.weight"), &[64, 64]);
        }
        f.affine("depth_linear", hidden, 8 * 64);
        // A single valid code makes Session's fixed stochastic audio sampler deterministic.
        for i in 0..8 {
            f.tensor(format!("depth_embeddings.{i}.embedding.weight"), &[64, 1]);
            f.tensor(format!("depth_embeddings.{i}.embedding_norm.weight"), &[64]);
            f.tensor(format!("depth_embeddings.{i}.to_logits.weight"), &[64, 1]);
        }
        f.tensor("audio_embedding.embedding.weight", &[hidden, 8 * 2049]);
        f.tensor("audio_embedding.embedding_norm.weight", &[hidden]);
        f.tensor("audio_embedding.to_logits.weight", &[hidden, 1]);
    }
    if detok {
        f.tensor("emb.emb.weight", &[32, 8 * 2048]);
        f.tensor("lfm.embedding_norm.weight", &[32]);
        f.affine("lin", 32, 1282);
        for i in 0..8 {
            let p = format!("lfm.layers.{i}");
            for name in ["operator_norm", "ffn_norm"] {
                f.tensor(format!("{p}.{name}.weight"), &[32]);
            }
            for name in ["w1", "w2", "w3"] {
                f.tensor(format!("{p}.feed_forward.{name}.weight"), &[32, 32]);
            }
            if [0, 1, 3, 5, 7].contains(&i) {
                f.tensor(format!("{p}.conv.in_proj.weight"), &[32, 96]);
                f.tensor(format!("{p}.conv.out_proj.weight"), &[32, 32]);
                f.tensor(format!("{p}.conv.conv.weight"), &[3, 32]);
            } else {
                for name in ["q", "k", "v", "out"] {
                    f.tensor(format!("{p}.self_attn.{name}_proj.weight"), &[32, 32]);
                }
                for name in ["q", "k"] {
                    f.tensor(format!("{p}.self_attn.{name}_layernorm.weight"), &[32]);
                }
            }
        }
    }
    f.finish(seed)
}

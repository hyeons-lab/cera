use cera::convert::writer::{GGML_TYPE_F32, GgufWriter};

/// Complete tiny weights exercise preprocessing and decoding, but deliberately
/// ignore acoustic content. The output projection selects only `a` or `b`.
pub fn model(output_b: bool, multilingual: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "whisper");
    let mut tokens = vec![
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|en|>",
        "<|es|>",
        "<|translate|>",
        "<|transcribe|>",
        "<|notimestamps|>",
        "<|nospeech|>",
        "a",
        "b",
        "<|0.00|>",
        "<|0.02|>",
    ];
    if !multilingual {
        tokens[5] = "unused";
    }
    writer.add_string_array(
        "tokenizer.ggml.tokens",
        tokens.into_iter().map(str::to_owned).collect(),
    );
    for (key, value) in [
        ("audio.block_count", 1),
        ("audio.embedding_length", 4),
        ("audio.attention.head_count", 1),
        ("audio.mel_bins", 80),
        ("audio.context_length", 1500),
        ("text.block_count", 1),
        ("text.embedding_length", 4),
        ("text.attention.head_count", 1),
        ("text.context_length", 16),
        ("vocab_size", 12),
    ] {
        writer.add_u32(format!("whisper.{key}"), value);
    }
    let mut tensors = Vec::<(String, Vec<u64>, Vec<f32>)>::new();
    let mut add = |name: String, shape: Vec<u64>, value: f32| {
        let count = shape.iter().product::<u64>() as usize;
        tensors.push((name, shape, vec![value; count]));
    };
    for (conv, input) in [("conv1", 80), ("conv2", 4)] {
        add(format!("encoder.{conv}.weight"), vec![3, input, 4], 0.0);
        add(format!("encoder.{conv}.bias"), vec![4], 0.0);
    }
    add("encoder.positional_embedding".into(), vec![4, 1500], 0.0);
    add("decoder.positional_embedding".into(), vec![4, 16], 0.0);
    for side in ["encoder", "decoder"] {
        for norm in ["attn_ln", "mlp_ln"] {
            add(format!("{side}.blocks.0.{norm}.weight"), vec![4], 1.0);
            add(format!("{side}.blocks.0.{norm}.bias"), vec![4], 0.0);
        }
        for projection in ["query", "key", "value", "out"] {
            add(
                format!("{side}.blocks.0.attn.{projection}.weight"),
                vec![4, 4],
                0.0,
            );
        }
        add(format!("{side}.blocks.0.mlp.0.weight"), vec![4, 16], 0.0);
        add(format!("{side}.blocks.0.mlp.2.weight"), vec![16, 4], 0.0);
        add(format!("{side}.ln_post.weight"), vec![4], 1.0);
        add(format!("{side}.ln_post.bias"), vec![4], 0.0);
    }
    add("decoder.blocks.0.cross_attn_ln.weight".into(), vec![4], 1.0);
    add("decoder.blocks.0.cross_attn_ln.bias".into(), vec![4], 0.0);
    for projection in ["query", "key", "value", "out"] {
        add(
            format!("decoder.blocks.0.cross_attn.{projection}.weight"),
            vec![4, 4],
            0.0,
        );
    }
    tensors.push((
        "decoder.token_embeddings.weight".into(),
        vec![4, 12],
        [1.0, -1.0, 0.0, 0.0].repeat(12),
    ));
    let mut output = vec![0.0; 4 * 12];
    let row = if output_b { 9 } else { 8 };
    output[row * 4..row * 4 + 4].copy_from_slice(&[4.0, -4.0, 0.0, 0.0]);
    tensors.push(("decoder.proj.weight".into(), vec![4, 12], output));
    let mut payloads = Vec::new();
    for (name, shape, values) in tensors {
        let bytes: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        writer.add_tensor(name, shape, GGML_TYPE_F32, bytes.len());
        payloads.push(bytes);
    }
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    for payload in payloads {
        writer.write_tensor_data(&mut bytes, &payload).unwrap();
    }
    bytes
}

pub fn pcm() -> Vec<f32> {
    (0..1600).map(|i| ((i as f32) * 0.13).sin() * 0.2).collect()
}

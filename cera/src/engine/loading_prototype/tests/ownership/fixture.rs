use super::super::{GGML_TYPE_F32, GgufWriter};
use crate::lora::LoraAdapterWeights;
use std::sync::Arc;

fn finish(mut writer: GgufWriter, tensors: Vec<(String, Vec<u64>, Vec<f32>)>) -> Arc<[u8]> {
    let mut payloads = Vec::new();
    for (name, shape, values) in tensors {
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        writer.add_tensor(name, shape, GGML_TYPE_F32, bytes.len());
        payloads.push(bytes);
    }
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    for payload in payloads {
        writer.write_tensor_data(&mut bytes, &payload).unwrap();
    }
    bytes.into()
}

// One convolution and one attention block exercise both kinds of live CPU state.
pub(super) fn tiny_hybrid() -> Arc<[u8]> {
    tiny_hybrid_with_seed(7)
}

pub(super) fn tiny_hybrid_with_seed(seed: u32) -> Arc<[u8]> {
    hybrid(seed, false)
}

#[cfg(all(
    not(target_arch = "wasm32"),
    any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    )
))]
pub(super) fn tiny_audio() -> Arc<[u8]> {
    hybrid(7, true)
}

pub(super) fn tiny_hybrid_with_context(context: u32) -> Arc<[u8]> {
    hybrid_with_context(7, false, context)
}

fn hybrid(seed: u32, audio: bool) -> Arc<[u8]> {
    hybrid_with_context(seed, audio, if audio { 512 } else { 64 })
}

fn hybrid_with_context(seed: u32, audio: bool, context: u32) -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "lfm2");
    writer.add_string("general.name", "ownership-hybrid-fixture");
    let vocab_size = if audio { 3 } else { 2 };
    let tokens = if audio {
        writer.add_i32_array("tokenizer.ggml.token_type", vec![1, 1, 3]);
        writer.add_string(
            "tokenizer.chat_template",
            "ab{{ messages[1]['content'] }}ba",
        );
        vec!["a".into(), "b".into(), "<|reserved_4|>".into()]
    } else {
        vec!["a".into(), "b".into()]
    };
    writer.add_string_array("tokenizer.ggml.tokens", tokens);
    for (key, value) in [
        ("block_count", 2),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("context_length", context),
        ("vocab_size", vocab_size),
        ("shortconv.l_cache", 3),
    ] {
        writer.add_u32(format!("lfm2.{key}"), value);
    }
    writer.add_i32_array("lfm2.attention.head_count_kv", vec![0, 1]);
    let mut shapes = vec![
        ("token_embd.weight".into(), vec![32, vocab_size as u64]),
        ("token_embd_norm.weight".into(), vec![32]),
        ("blk.0.shortconv.in_proj.weight".into(), vec![32, 96]),
        ("blk.0.shortconv.out_proj.weight".into(), vec![32, 32]),
        ("blk.0.shortconv.conv.weight".into(), vec![3, 32]),
    ];
    for block in 0..2 {
        for norm in ["attn_norm", "ffn_norm"] {
            shapes.push((format!("blk.{block}.{norm}.weight"), vec![32]));
        }
        for projection in ["ffn_gate", "ffn_up", "ffn_down"] {
            shapes.push((format!("blk.{block}.{projection}.weight"), vec![32, 32]));
        }
    }
    for norm in ["attn_q_norm", "attn_k_norm"] {
        shapes.push((format!("blk.1.{norm}.weight"), vec![32]));
    }
    for projection in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        shapes.push((format!("blk.1.{projection}.weight"), vec![32, 32]));
    }
    random_weights(writer, shapes, seed)
}

pub(super) fn tiny_dense() -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");
    writer.add_string("general.name", "ownership-dense-fixture");
    writer.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
    for (key, value) in [
        ("block_count", 1),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("context_length", 64),
        ("vocab_size", 2),
    ] {
        writer.add_u32(format!("llama.{key}"), value);
    }
    let mut shapes = vec![
        ("token_embd.weight".into(), vec![32, 2]),
        ("output_norm.weight".into(), vec![32]),
        ("blk.0.attn_norm.weight".into(), vec![32]),
        ("blk.0.ffn_norm.weight".into(), vec![32]),
    ];
    for projection in [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ] {
        shapes.push((format!("blk.0.{projection}.weight"), vec![32, 32]));
    }
    random_weights(writer, shapes, 7)
}

// Two attention layers make suffix-dependent prefix KV observable when
// bidirectional attention is enabled in the model or by classifier LoRA.
pub(super) fn tiny_attention_lfm2(causal: bool) -> Arc<[u8]> {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "lfm2");
    writer.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
    writer.add_bool("lfm2.is_causal", causal);
    for (key, value) in [
        ("block_count", 2),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("context_length", 64),
        ("vocab_size", 2),
    ] {
        writer.add_u32(format!("lfm2.{key}"), value);
    }
    writer.add_i32_array("lfm2.attention.head_count_kv", vec![1, 1]);
    let mut shapes = vec![
        ("token_embd.weight".into(), vec![32, 2]),
        ("token_embd_norm.weight".into(), vec![32]),
    ];
    for block in 0..2 {
        for norm in ["attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"] {
            shapes.push((format!("blk.{block}.{norm}.weight"), vec![32]));
        }
        for projection in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            shapes.push((format!("blk.{block}.{projection}.weight"), vec![32, 32]));
        }
    }
    random_weights_impl(writer, shapes, 7, true)
}

fn random_weights(writer: GgufWriter, shapes: Vec<(String, Vec<u64>)>, seed: u32) -> Arc<[u8]> {
    random_weights_impl(writer, shapes, seed, false)
}

fn random_weights_impl(
    mut writer: GgufWriter,
    shapes: Vec<(String, Vec<u64>)>,
    mut seed: u32,
    quantized: bool,
) -> Arc<[u8]> {
    let tensors = shapes
        .into_iter()
        .map(|(name, shape)| {
            let values = (0..shape.iter().product::<u64>())
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    if name.contains("norm") {
                        1.0
                    } else {
                        ((seed >> 16) as f32 / 65_535.0 - 0.5) / 4.0
                    }
                })
                .collect();
            (name, shape, values)
        })
        .collect::<Vec<_>>();
    if !quantized {
        return finish(writer, tensors);
    }
    let mut payloads = Vec::new();
    for (name, shape, values) in tensors {
        let (dtype, bytes) = if shape.len() == 2 {
            let mut bytes = vec![0; values.len() / 32 * 34];
            crate::convert::quantize::quantize_q8_0(&values, &mut bytes).unwrap();
            (crate::convert::writer::GGML_TYPE_Q8_0, bytes)
        } else {
            (
                GGML_TYPE_F32,
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            )
        };
        writer.add_tensor(name, shape, dtype, bytes.len());
        payloads.push(bytes);
    }
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    for payload in payloads {
        writer.write_tensor_data(&mut bytes, &payload).unwrap();
    }
    bytes.into()
}

pub(super) fn adapter(width: usize) -> Arc<LoraAdapterWeights> {
    let tensors = [("a", vec![width as u64, 1]), ("b", vec![1, width as u64])]
        .into_iter()
        .map(|(suffix, shape)| {
            let values = (0..width).map(|i| ((i % 7) as f32 - 3.0) / 2.0).collect();
            (
                format!("blk.0.ffn_down.weight.lora_{suffix}"),
                shape,
                values,
            )
        })
        .collect();
    LoraAdapterWeights::from_gguf_bytes(finish(GgufWriter::new(), tensors)).unwrap()
}

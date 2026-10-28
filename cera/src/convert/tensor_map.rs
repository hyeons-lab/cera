//! Tensor name classification, expert stacking, and fused weight splitting for
//! SafeTensors-to-GGUF conversion.

use crate::convert::safetensors::translate_hf_to_gguf_tensor_name_with_arch;
use crate::session::CeraError;

/// How a Hugging Face tensor should be handled when converting a language model to GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorAction {
    /// Map 1-to-1 to the given GGUF tensor name.
    Map(String),
    /// Skip this tensor (for example, vision or audio encoder weights in a multimodal checkpoint).
    Skip,
    /// Accumulate a per-expert weight slice (`experts.{expert_idx}.{w1|w2|w3}.weight`) to stack
    /// into a 3D `blk.{block_idx}.ffn_{gate|down|up}_exps.weight` tensor.
    StackExpert {
        block_idx: usize,
        expert_idx: usize,
        gguf_name: String,
    },
    /// Split a fused LFM3-VL `self_attn.qkv.weight` tensor into `attn_q`, `attn_k`, and `attn_v`.
    SplitVlQkv { block_idx: usize },
    /// Split a fused LFM3-VL `feed_forward.gate_up.weight` tensor into `ffn_gate` and `ffn_up`.
    SplitVlGateUp { block_idx: usize },
}

const MULTIMODAL_SKIP_PREFIXES: [&str; 20] = [
    "encoder.",
    "audio_encoder.",
    "mlp.",
    "audio_projector.",
    "codebook_embedding.",
    "depth_decoder.",
    "depth_embeddings.",
    "depth_linear.",
    "depthformer.",
    "mask_embedding",
    "vision_tower.",
    "vision_encoder.",
    "vision_projector.",
    "multi_modal_projector.",
    "vision_feature_layer.",
    "conformer.",
    "audio_tower.",
    "audio_adapter.",
    "code_generator.",
    "detokenizer.",
];

/// Return true if `hf_name` belongs to a multimodal tower/adapter that is excluded from
/// the language-model GGUF.
pub fn is_multimodal_skipped_tensor(hf_name: &str) -> bool {
    let stripped = hf_name.strip_prefix("model.").unwrap_or(hf_name);
    MULTIMODAL_SKIP_PREFIXES
        .iter()
        .any(|p| hf_name.starts_with(p) || stripped.starts_with(p))
}

/// Parse a per-expert tensor name of the form:
/// `[language_model.][lfm.][model.]layers.{bid}.{feed_forward|mlp}.experts.{xid}.{w1|w2|w3|gate_proj|down_proj|up_proj}.weight`
pub fn parse_lfm3_expert_tensor(hf_name: &str) -> Option<(usize, usize, String)> {
    let stripped = hf_name
        .strip_prefix("model.language_model.")
        .or_else(|| hf_name.strip_prefix("language_model."))
        .unwrap_or(hf_name);
    let stripped = stripped.strip_prefix("lfm.").unwrap_or(stripped);
    let layer_rest = stripped
        .strip_prefix("model.layers.")
        .or_else(|| stripped.strip_prefix("layers."))?;
    let (layer_str, sub) = layer_rest.split_once('.')?;
    let block_idx: usize = layer_str.parse().ok()?;
    let exp_rest = sub
        .strip_prefix("feed_forward.experts.")
        .or_else(|| sub.strip_prefix("mlp.experts."))?;
    let (exp_str, proj_tail) = exp_rest.split_once('.')?;
    let expert_idx: usize = exp_str.parse().ok()?;
    let gguf_suffix = match proj_tail {
        "w1.weight" | "gate_proj.weight" => "ffn_gate_exps.weight",
        "w2.weight" | "down_proj.weight" => "ffn_down_exps.weight",
        "w3.weight" | "up_proj.weight" => "ffn_up_exps.weight",
        _ => return None,
    };
    Some((
        block_idx,
        expert_idx,
        format!("blk.{block_idx}.{gguf_suffix}"),
    ))
}

/// Classify a Hugging Face tensor name for the given target GGUF architecture.
pub fn classify_tensor(hf_name: &str, arch: &str) -> TensorAction {
    if arch == "lfm3" {
        if is_multimodal_skipped_tensor(hf_name) {
            return TensorAction::Skip;
        }
        if let Some((block_idx, expert_idx, gguf_name)) = parse_lfm3_expert_tensor(hf_name) {
            return TensorAction::StackExpert {
                block_idx,
                expert_idx,
                gguf_name,
            };
        }
        let stripped = hf_name
            .strip_prefix("model.language_model.")
            .or_else(|| hf_name.strip_prefix("language_model."))
            .unwrap_or(hf_name);
        let stripped = stripped.strip_prefix("lfm.").unwrap_or(stripped);
        let layer_rest = stripped
            .strip_prefix("model.layers.")
            .or_else(|| stripped.strip_prefix("layers."));
        if let Some((layer_str, sub)) = layer_rest.and_then(|r| r.split_once('.'))
            && let Ok(block_idx) = layer_str.parse::<usize>()
        {
            match sub {
                "self_attn.qkv.weight" => return TensorAction::SplitVlQkv { block_idx },
                "feed_forward.gate_up.weight" => return TensorAction::SplitVlGateUp { block_idx },
                _ => {}
            }
        }
    }
    TensorAction::Map(translate_hf_to_gguf_tensor_name_with_arch(hf_name, arch))
}

/// Stack per-expert 2D tensors `[rows, cols]` in ascending `expert_idx` order `[0..n_experts)`
/// into a single contiguous 3D tensor `[n_experts, rows, cols]`.
pub fn stack_expert_tensors(
    experts: &mut [(usize, Vec<usize>, Vec<f32>)],
    expected_experts: Option<usize>,
    gguf_name: &str,
) -> Result<(Vec<usize>, Vec<f32>), CeraError> {
    if experts.is_empty() {
        return Err(CeraError::Backend(format!(
            "no expert tensors found for `{gguf_name}`"
        )));
    }
    experts.sort_by_key(|(idx, _, _)| *idx);
    if let Some(expected) = expected_experts
        && experts.len() != expected
    {
        return Err(CeraError::Backend(format!(
            "`{gguf_name}` has {} expert tensors, expected {expected}",
            experts.len()
        )));
    }
    let first_shape = &experts[0].1;
    if first_shape.len() != 2 {
        return Err(CeraError::Backend(format!(
            "`{gguf_name}` expert 0 has shape {first_shape:?}; expected 2D [rows, cols]"
        )));
    }
    let (rows, cols) = (first_shape[0], first_shape[1]);
    let slice_len = rows
        .checked_mul(cols)
        .ok_or_else(|| CeraError::Backend(format!("`{gguf_name}` expert dimensions overflow")))?;
    let total_len = slice_len.checked_mul(experts.len()).ok_or_else(|| {
        CeraError::Backend(format!("`{gguf_name}` stacked expert buffer size overflow"))
    })?;
    let mut stacked = Vec::with_capacity(total_len);
    for (expected_idx, (idx, shape, data)) in experts.iter().enumerate() {
        if *idx != expected_idx {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` missing expert {expected_idx} (found expert {idx})"
            )));
        }
        if shape.as_slice() != [rows, cols] {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` expert {idx} shape {shape:?} does not match [{rows}, {cols}]"
            )));
        }
        if data.len() != slice_len {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` expert {idx} has {} elements, expected {slice_len}",
                data.len()
            )));
        }
        stacked.extend_from_slice(data);
    }
    Ok((vec![experts.len(), rows, cols], stacked))
}

/// `(q_rows, kv_rows, total_rows)` of a fused LFM3-VL qkv weight, or `None` when a row count
/// does not fit in `usize`. The head counts come from an untrusted `config.json`, so the
/// products are checked: a wrapped row count could match a crafted tensor shape.
pub(crate) fn lfm3_vl_qkv_rows(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Option<(usize, usize, usize)> {
    let q_rows = n_head.checked_mul(head_dim)?;
    let kv_rows = n_head_kv.checked_mul(head_dim)?;
    let total = q_rows.checked_add(kv_rows.checked_mul(2)?)?;
    Some((q_rows, kv_rows, total))
}

/// A `(gguf_name, shape, values)` tuple produced by splitting a fused tensor.
pub type SplitTensor = (String, Vec<usize>, Vec<f32>);

/// Split a fused LFM3-VL `self_attn.qkv.weight` tensor of shape
/// `[(n_head + 2 * n_head_kv) * head_dim, hidden]` into `attn_q`, `attn_k`, and `attn_v`.
pub fn split_lfm3_vl_qkv(
    qkv: &[f32],
    shape: &[usize],
    block_idx: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Result<[SplitTensor; 3], CeraError> {
    if shape.len() != 2 {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused qkv.weight must be 2D, got {shape:?}"
        )));
    }
    let (total_rows, cols) = (shape[0], shape[1]);
    let Some((q_rows, kv_rows, expected_rows)) = lfm3_vl_qkv_rows(n_head, n_head_kv, head_dim)
    else {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused qkv.weight: {n_head} heads and {n_head_kv} kv heads of \
             width {head_dim} overflow the row count"
        )));
    };
    if total_rows != expected_rows || qkv.len() != total_rows.saturating_mul(cols) {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused qkv.weight shape {shape:?} does not match \
             q_rows={q_rows} + 2*kv_rows={kv_rows}"
        )));
    }
    let q_elems = q_rows * cols;
    let kv_elems = kv_rows * cols;
    let q = qkv[..q_elems].to_vec();
    let k = qkv[q_elems..q_elems + kv_elems].to_vec();
    let v = qkv[q_elems + kv_elems..].to_vec();
    Ok([
        (
            format!("blk.{block_idx}.attn_q.weight"),
            vec![q_rows, cols],
            q,
        ),
        (
            format!("blk.{block_idx}.attn_k.weight"),
            vec![kv_rows, cols],
            k,
        ),
        (
            format!("blk.{block_idx}.attn_v.weight"),
            vec![kv_rows, cols],
            v,
        ),
    ])
}

/// Split a fused LFM3-VL `feed_forward.gate_up.weight` tensor of shape `[2 * ff, hidden]`
/// into `blk.{block_idx}.ffn_gate.weight` and `blk.{block_idx}.ffn_up.weight`.
pub fn split_lfm3_vl_gate_up(
    gate_up: &[f32],
    shape: &[usize],
    block_idx: usize,
) -> Result<[SplitTensor; 2], CeraError> {
    if shape.len() != 2 || !shape[0].is_multiple_of(2) {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused gate_up.weight must have even 2D rows, got {shape:?}"
        )));
    }
    let (total_rows, cols) = (shape[0], shape[1]);
    if gate_up.len() != total_rows.saturating_mul(cols) {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused gate_up.weight element count mismatch"
        )));
    }
    let half_rows = total_rows / 2;
    let half_elems = half_rows * cols;
    let gate = gate_up[..half_elems].to_vec();
    let up = gate_up[half_elems..].to_vec();
    Ok([
        (
            format!("blk.{block_idx}.ffn_gate.weight"),
            vec![half_rows, cols],
            gate,
        ),
        (
            format!("blk.{block_idx}.ffn_up.weight"),
            vec![half_rows, cols],
            up,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qkv_split_refuses_head_counts_whose_rows_overflow() {
        // 2 * kv_rows wrapped (release) or panicked (debug) on a huge `num_key_value_heads`
        let huge = usize::MAX / 2 + 1;
        let err = split_lfm3_vl_qkv(&[0.0; 4], &[2, 2], 0, 1, huge, 1).unwrap_err();
        assert!(err.to_string().contains("fused qkv"), "{err}");
        let err = split_lfm3_vl_qkv(&[0.0; 4], &[2, 2], 0, huge, 1, 2).unwrap_err();
        assert!(err.to_string().contains("fused qkv"), "{err}");
    }

    #[test]
    fn multimodal_tensors_are_skipped_for_lfm3() {
        for name in [
            "vision_tower.embeddings.patch_embeddings.weight",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            "vision_encoder.blocks.0.weight",
            "vision_projector.linear_1.weight",
            "multi_modal_projector.linear_1.weight",
            "model.multi_modal_projector.linear_2.weight",
            "vision_feature_layer.norm.weight",
            "conformer.layers.0.conv.weight",
            "audio_tower.layers.0.weight",
            "audio_adapter.mlp.0.weight",
            "code_generator.layers.0.weight",
            "detokenizer.lfm.layers.0.weight",
            "encoder.lfm.layers.0.operator_norm.weight",
            "mlp.model.0.weight",
            "codebook_embedding.weight",
            "depth_embeddings.0.weight",
            "depth_linear.weight",
            "depthformer.model.layers.0.ffn_norm.weight",
            "mask_embedding",
        ] {
            assert_eq!(classify_tensor(name, "lfm3"), TensorAction::Skip, "{name}");
        }
    }

    #[test]
    fn language_model_prefix_is_stripped_and_mapped() {
        assert_eq!(
            classify_tensor("language_model.model.embed_tokens.weight", "lfm3"),
            TensorAction::Map("token_embd.weight".into())
        );
        assert_eq!(
            classify_tensor("language_model.layers.3.operator_norm.weight", "lfm3"),
            TensorAction::Map("blk.3.attn_norm.weight".into())
        );
        assert_eq!(
            classify_tensor("language_model.layers.1.self_attn.qkv.weight", "lfm3"),
            TensorAction::SplitVlQkv { block_idx: 1 }
        );
        assert_eq!(
            classify_tensor(
                "language_model.layers.1.feed_forward.gate_up.weight",
                "lfm3"
            ),
            TensorAction::SplitVlGateUp { block_idx: 1 }
        );
    }

    #[test]
    fn moe_router_bias_and_expert_tensors_are_classified() {
        assert_eq!(
            classify_tensor("model.layers.2.feed_forward.gate.weight", "lfm3"),
            TensorAction::Map("blk.2.ffn_gate_inp.weight".into())
        );
        assert_eq!(
            classify_tensor("model.layers.2.feed_forward.expert_bias", "lfm3"),
            TensorAction::Map("blk.2.exp_probs_b.bias".into())
        );
        assert_eq!(
            classify_tensor("model.layers.2.feed_forward.experts.5.w1.weight", "lfm3"),
            TensorAction::StackExpert {
                block_idx: 2,
                expert_idx: 5,
                gguf_name: "blk.2.ffn_gate_exps.weight".into(),
            }
        );
        assert_eq!(
            classify_tensor("model.layers.2.feed_forward.experts.5.w2.weight", "lfm3"),
            TensorAction::StackExpert {
                block_idx: 2,
                expert_idx: 5,
                gguf_name: "blk.2.ffn_down_exps.weight".into(),
            }
        );
        assert_eq!(
            classify_tensor("model.layers.2.feed_forward.experts.5.w3.weight", "lfm3"),
            TensorAction::StackExpert {
                block_idx: 2,
                expert_idx: 5,
                gguf_name: "blk.2.ffn_up_exps.weight".into(),
            }
        );
    }

    #[test]
    fn stack_expert_tensors_orders_by_index_and_produces_3d_shape() {
        let mut experts = vec![
            (1, vec![2, 3], vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            (0, vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ];
        let (shape, data) =
            stack_expert_tensors(&mut experts, Some(2), "blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(shape, vec![2, 2, 3]);
        assert_eq!(
            data,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
            ]
        );
    }

    #[test]
    fn split_vl_qkv_and_gate_up_slice_rows_cleanly() {
        let qkv: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let parts = split_lfm3_vl_qkv(&qkv, &[6, 4], 2, 4, 1, 1).unwrap();
        assert_eq!(parts[0].0, "blk.2.attn_q.weight");
        assert_eq!(parts[0].1, vec![4, 4]);
        assert_eq!(parts[0].2, (0..16).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(parts[1].0, "blk.2.attn_k.weight");
        assert_eq!(parts[1].1, vec![1, 4]);
        assert_eq!(parts[1].2, (16..20).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(parts[2].0, "blk.2.attn_v.weight");
        assert_eq!(parts[2].1, vec![1, 4]);
        assert_eq!(parts[2].2, (20..24).map(|x| x as f32).collect::<Vec<_>>());

        let gate_up: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let gu = split_lfm3_vl_gate_up(&gate_up, &[4, 4], 3).unwrap();
        assert_eq!(gu[0].0, "blk.3.ffn_gate.weight");
        assert_eq!(gu[0].1, vec![2, 4]);
        assert_eq!(gu[0].2, (0..8).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(gu[1].0, "blk.3.ffn_up.weight");
        assert_eq!(gu[1].1, vec![2, 4]);
        assert_eq!(gu[1].2, (8..16).map(|x| x as f32).collect::<Vec<_>>());
    }

    #[test]
    fn canonical_omni_and_audio_tensors_are_skipped_for_llm_conversion() {
        for name in [
            "model.audio_encoder.downsample.conv_in.weight",
            "model.audio_encoder.lfm.layers.0.operator_norm.weight",
            "model.audio_projector.layer_norm.weight",
            "model.audio_projector.linear_1.weight",
            "model.codebook_embedding.weight",
            "model.depth_decoder.depth_linear.weight",
            "model.depth_decoder.mask_embedding",
            "model.depth_decoder.model.layers.0.operator_norm.weight",
            "model.depth_decoder.depth_embeddings.0.embedding.weight",
        ] {
            assert_eq!(
                classify_tensor(name, "lfm3"),
                TensorAction::Skip,
                "expected `{name}` to be skipped during LLM conversion"
            );
        }
    }
}

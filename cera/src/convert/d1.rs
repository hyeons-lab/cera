//! Conversion of the open `d1-omni` decision checkpoint (`model_type = "d1_omni"`).
//!
//! The checkpoint keeps a bidirectional LFM2 trunk, a decision head, a SigLIP2 vision tower with
//! its projector and (not converted here yet) a FastConformer audio tower in one
//! `model.safetensors`. It is
//! written as an ordinary non-causal `lfm2` GGUF, so the trunk loads and runs as every other
//! bidirectional LFM2 encoder does, plus the head's tensors and settings under the `d1.` prefix:
//!
//! | Checkpoint tensor                                  | GGUF tensor                      |
//! |----------------------------------------------------|----------------------------------|
//! | `encoder.*`                                        | the LFM2 names of `model.*`      |
//! | `head.type_emb.weight`                             | `d1.question_type.weight`        |
//! | `head.head.layers.N.norm1.{weight,bias}`           | `d1.blk.N.attn_norm.*`           |
//! | `head.head.layers.N.self_attn.in_proj_{weight,bias}` | `d1.blk.N.attn_qkv.{weight,bias}` |
//! | `head.head.layers.N.self_attn.out_proj.*`          | `d1.blk.N.attn_output.*`         |
//! | `head.head.layers.N.norm2.*`                       | `d1.blk.N.ffn_norm.*`            |
//! | `head.head.layers.N.linear1.*` / `linear2.*`       | `d1.blk.N.ffn_up.*` / `ffn_down.*` |
//! | `head.scorer.0.*` / `.1.*` / `.3.*`                | `d1.cls.norm.*` / `d1.cls.*` / `d1.cls.output.*` |
//! | `vision.tower.vision_model.embeddings.patch_embedding.*` | `d1.v.patch_embd.*`        |
//! | `vision.tower.vision_model.embeddings.position_embedding.weight` | `d1.v.position_embd.weight` |
//! | `vision.tower.vision_model.encoder.layers.N.layer_norm{1,2}.*` | `d1.v.blk.N.ln{1,2}.*` |
//! | `vision.tower.vision_model.encoder.layers.N.self_attn.{q,k,v,out}_proj.*` | `d1.v.blk.N.attn_{q,k,v,out}.*` |
//! | `vision.tower.vision_model.encoder.layers.N.mlp.fc{1,2}.*` | `d1.v.blk.N.ffn_{up,down}.*` |
//! | `vision.tower.vision_model.post_layernorm.*`       | `d1.v.post_ln.*`                 |
//! | `vision.projector.linear_{1,2}.*`                  | `d1.v.mm.{1,2}.*`                |
//!
//! Metadata, all optional for a reader that only wants the trunk:
//!
//! * `d1.head.block_count`, `d1.head.attention.head_count`, `d1.head.feed_forward_length`,
//!   `d1.head.layer_norm_epsilon`: the head's pre-norm transformer layers;
//! * `d1.vision.block_count`, `d1.vision.embedding_length`, `d1.vision.feed_forward_length`,
//!   `d1.vision.attention.head_count`, `d1.vision.patch_size`, `d1.vision.position_side`,
//!   `d1.vision.layer_norm_epsilon`, `d1.vision.projector_hidden_length`: the tower;
//! * `d1.context_length`: the longest prompt, media prefix included;
//! * `d1.image_text_length`, `d1.audio_text_length`: the room an image or audio request leaves
//!   its text;
//! * `d1.temperature.keys` and `d1.temperature.values`: the calibration of text answers, by
//!   question type and option count, as the checkpoint names them (`choice:3-5`).

use serde_json::Value;

use crate::convert::safetensors::translate_hf_to_gguf_tensor_name_with_arch;
use crate::convert::writer::GgufWriter;
use crate::session::CeraError;

/// `model_type` of a d1-omni `config.json`.
pub const MODEL_TYPE: &str = "d1_omni";

/// GGUF names that stay F32 whatever the target: the scorer and the question-type table are
/// tiny and everything the answers rest on.
pub fn keeps_f32(gguf_name: &str) -> bool {
    gguf_name.starts_with("d1.cls")
        || gguf_name.starts_with("d1.question_type")
        || gguf_name.starts_with("d1.v.patch_embd")
        || gguf_name.starts_with("d1.v.position_embd")
}

/// A vision tower or projector tensor under its GGUF name, `None` for one that does not belong.
fn vision_tensor_name(hf_name: &str) -> Option<String> {
    if let Some(rest) = hf_name.strip_prefix("vision.projector.") {
        let (layer, kind) = rest.split_once('.')?;
        let n = layer.strip_prefix("linear_")?;
        return matches!(n, "1" | "2").then(|| format!("d1.v.mm.{n}.{kind}"));
    }
    let rest = hf_name.strip_prefix("vision.tower.vision_model.")?;
    match rest {
        "embeddings.patch_embedding.weight" => return Some("d1.v.patch_embd.weight".into()),
        "embeddings.patch_embedding.bias" => return Some("d1.v.patch_embd.bias".into()),
        "embeddings.position_embedding.weight" => return Some("d1.v.position_embd.weight".into()),
        "post_layernorm.weight" => return Some("d1.v.post_ln.weight".into()),
        "post_layernorm.bias" => return Some("d1.v.post_ln.bias".into()),
        _ => {}
    }
    let (index, suffix) = rest.strip_prefix("encoder.layers.")?.split_once('.')?;
    index.parse::<usize>().ok()?;
    let (module, kind) = suffix.rsplit_once('.')?;
    let name = match module {
        "layer_norm1" => "ln1",
        "layer_norm2" => "ln2",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.out_proj" => "attn_out",
        "mlp.fc1" => "ffn_up",
        "mlp.fc2" => "ffn_down",
        _ => return None,
    };
    matches!(kind, "weight" | "bias").then(|| format!("d1.v.blk.{index}.{name}.{kind}"))
}

/// The GGUF name of a d1-omni checkpoint tensor, or `None` for a tensor this conversion leaves
/// out (the audio tower).
pub fn tensor_name(hf_name: &str) -> Option<String> {
    if hf_name.starts_with("vision.") {
        return vision_tensor_name(hf_name);
    }
    if let Some(rest) = hf_name.strip_prefix("encoder.") {
        return Some(translate_hf_to_gguf_tensor_name_with_arch(
            &format!("model.{rest}"),
            "lfm2",
        ));
    }
    let rest = hf_name.strip_prefix("head.")?;
    if rest == "type_emb.weight" {
        return Some("d1.question_type.weight".to_string());
    }
    if let Some(scorer) = rest.strip_prefix("scorer.") {
        let (index, kind) = scorer.split_once('.')?;
        let part = match index {
            "0" => "d1.cls.norm",
            "1" => "d1.cls",
            "3" => "d1.cls.output",
            _ => return None,
        };
        return Some(format!("{part}.{kind}"));
    }
    let (index, suffix) = rest.strip_prefix("head.layers.")?.split_once('.')?;
    index.parse::<usize>().ok()?;
    let name = match suffix {
        "norm1.weight" => "attn_norm.weight",
        "norm1.bias" => "attn_norm.bias",
        "self_attn.in_proj_weight" => "attn_qkv.weight",
        "self_attn.in_proj_bias" => "attn_qkv.bias",
        "self_attn.out_proj.weight" => "attn_output.weight",
        "self_attn.out_proj.bias" => "attn_output.bias",
        "norm2.weight" => "ffn_norm.weight",
        "norm2.bias" => "ffn_norm.bias",
        "linear1.weight" => "ffn_up.weight",
        "linear1.bias" => "ffn_up.bias",
        "linear2.weight" => "ffn_down.weight",
        "linear2.bias" => "ffn_down.bias",
        _ => return None,
    };
    Some(format!("d1.blk.{index}.{name}"))
}

fn count(config: &Value, key: &str) -> Result<u32, CeraError> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| CeraError::Backend(format!("d1-omni config.json: `{key}` is missing")))
}

/// Write the `d1.*` metadata of a d1-omni `config.json`.
///
/// # Errors
///
/// Refuses a config without the counts the head and the prompt budget are read from.
pub fn apply_metadata(config: &Value, writer: &mut GgufWriter) -> Result<(), CeraError> {
    let hidden = config
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| {
            CeraError::Backend("d1-omni config.json: no `text_config.hidden_size`".into())
        })?;
    let layers = count(config, "head_layers")?;
    writer.add_u32("d1.head.block_count", layers);
    // the head is `nn.TransformerEncoderLayer(d, d // 64, 4 * d)`
    writer.add_u32("d1.head.attention.head_count", hidden / 64);
    writer.add_u32("d1.head.feed_forward_length", 4 * hidden);
    writer.add_f32("d1.head.layer_norm_epsilon", 1e-5);
    writer.add_u32("d1.context_length", count(config, "max_length")?);
    writer.add_u32("d1.image_text_length", count(config, "image_text_length")?);
    writer.add_u32("d1.audio_text_length", count(config, "audio_text_length")?);
    if let Some(vision) = config.get("vision_config") {
        let field = |key: &str| {
            vision
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    CeraError::Backend(format!("d1-omni config.json: no `vision_config.{key}`"))
                })
        };
        writer.add_u32("d1.vision.block_count", field("num_hidden_layers")?);
        writer.add_u32("d1.vision.embedding_length", field("hidden_size")?);
        writer.add_u32("d1.vision.feed_forward_length", field("intermediate_size")?);
        writer.add_u32(
            "d1.vision.attention.head_count",
            field("num_attention_heads")?,
        );
        writer.add_u32("d1.vision.patch_size", field("patch_size")?);
        // the position table is square: `num_patches` is its area
        let patches = field("num_patches")?;
        let side = (f64::from(patches).sqrt().round()) as u32;
        if side * side != patches {
            return Err(CeraError::Backend(format!(
                "d1-omni vision_config.num_patches ({patches}) is not a square"
            )));
        }
        writer.add_u32("d1.vision.position_side", side);
        writer.add_f32(
            "d1.vision.layer_norm_epsilon",
            vision
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
        );
        writer.add_u32(
            "d1.vision.projector_hidden_length",
            count(config, "projector_hidden_size")?,
        );
    }
    if let Some(temperatures) = config.get("temperatures").and_then(Value::as_object) {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (key, value) in temperatures {
            let value = value.as_f64().ok_or_else(|| {
                CeraError::Backend(format!("d1-omni temperature `{key}` is not a number"))
            })?;
            keys.push(key.clone());
            values.push(value as f32);
        }
        writer.add_string_array("d1.temperature.keys", keys);
        writer.add_f32_array("d1.temperature.values", values);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trunk_tensors_take_the_lfm2_names() {
        assert_eq!(
            tensor_name("encoder.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            tensor_name("encoder.embedding_norm.weight").as_deref(),
            Some("token_embd_norm.weight")
        );
        assert_eq!(
            tensor_name("encoder.layers.2.self_attn.q_proj.weight").as_deref(),
            Some("blk.2.attn_q.weight")
        );
    }

    #[test]
    fn head_tensors_live_under_d1() {
        let cases = [
            ("head.type_emb.weight", "d1.question_type.weight"),
            ("head.scorer.0.weight", "d1.cls.norm.weight"),
            ("head.scorer.1.bias", "d1.cls.bias"),
            ("head.scorer.3.weight", "d1.cls.output.weight"),
            ("head.head.layers.1.norm1.bias", "d1.blk.1.attn_norm.bias"),
            (
                "head.head.layers.0.self_attn.in_proj_weight",
                "d1.blk.0.attn_qkv.weight",
            ),
            (
                "head.head.layers.0.self_attn.out_proj.bias",
                "d1.blk.0.attn_output.bias",
            ),
            (
                "head.head.layers.1.linear2.weight",
                "d1.blk.1.ffn_down.weight",
            ),
        ];
        for (hf, gguf) in cases {
            assert_eq!(tensor_name(hf).as_deref(), Some(gguf), "{hf}");
        }
    }

    #[test]
    fn vision_tensors_take_the_tower_names() {
        let cases = [
            (
                "vision.tower.vision_model.embeddings.patch_embedding.weight",
                "d1.v.patch_embd.weight",
            ),
            (
                "vision.tower.vision_model.embeddings.position_embedding.weight",
                "d1.v.position_embd.weight",
            ),
            (
                "vision.tower.vision_model.encoder.layers.3.layer_norm2.bias",
                "d1.v.blk.3.ln2.bias",
            ),
            (
                "vision.tower.vision_model.encoder.layers.11.self_attn.out_proj.weight",
                "d1.v.blk.11.attn_out.weight",
            ),
            (
                "vision.tower.vision_model.encoder.layers.0.mlp.fc1.bias",
                "d1.v.blk.0.ffn_up.bias",
            ),
            (
                "vision.tower.vision_model.post_layernorm.bias",
                "d1.v.post_ln.bias",
            ),
            ("vision.projector.linear_2.weight", "d1.v.mm.2.weight"),
        ];
        for (hf, gguf) in cases {
            assert_eq!(tensor_name(hf).as_deref(), Some(gguf), "{hf}");
        }
        for bad in [
            "vision.projector.linear_3.weight",
            "vision.tower.vision_model.encoder.layers.0.mlp.fc3.weight",
            "vision.tower.vision_model.encoder.layers.x.layer_norm1.weight",
            "vision.tower.vision_model.encoder.layers.0.layer_norm1.running_mean",
        ] {
            assert_eq!(tensor_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn towers_and_unknown_tensors_are_left_out() {
        for hf in [
            "audio.encoder.layers.0.conv.batch_norm.num_batches_tracked",
            "audio.adapter.norm.weight",
            "head.head.layers.0.self_attn.mystery",
            "head.scorer.2.weight",
            "head.head.layers.x.norm1.weight",
        ] {
            assert_eq!(tensor_name(hf), None, "{hf}");
        }
    }

    #[test]
    fn the_scorer_and_question_types_stay_f32() {
        assert!(keeps_f32("d1.cls.output.weight"));
        assert!(keeps_f32("d1.cls.weight"));
        assert!(keeps_f32("d1.question_type.weight"));
        assert!(keeps_f32("d1.v.patch_embd.weight"));
        assert!(keeps_f32("d1.v.position_embd.weight"));
        assert!(!keeps_f32("d1.v.blk.0.ffn_up.weight"));
        assert!(!keeps_f32("d1.blk.0.ffn_up.weight"));
        assert!(!keeps_f32("blk.0.ffn_up.weight"));
    }

    #[test]
    fn metadata_carries_the_head_and_the_calibration() {
        let config = serde_json::json!({
            "head_layers": 2,
            "max_length": 16384,
            "image_text_length": 896,
            "audio_text_length": 15360,
            "projector_hidden_size": 2048,
            "text_config": {"hidden_size": 1024},
            "vision_config": {
                "num_hidden_layers": 12, "hidden_size": 768, "intermediate_size": 3072,
                "num_attention_heads": 12, "patch_size": 16, "num_patches": 256,
                "layer_norm_eps": 1e-6
            },
            "temperatures": {"noul:2": 1.5, "choice": 1.0},
        });
        let mut writer = GgufWriter::new();
        apply_metadata(&config, &mut writer).unwrap();
        assert!(writer.get_metadata("d1.head.block_count").is_some());
        assert!(writer.get_metadata("d1.temperature.keys").is_some());
        assert!(writer.get_metadata("d1.vision.position_side").is_some());
        assert!(
            apply_metadata(
                &serde_json::json!({"head_layers": 2}),
                &mut GgufWriter::new()
            )
            .is_err()
        );
    }
}

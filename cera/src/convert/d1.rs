//! Conversion of the open `d1-omni` decision checkpoint (`model_type = "d1_omni"`).
//!
//! The checkpoint keeps a bidirectional LFM2 trunk, a decision head, a SigLIP2 vision tower with
//! its projector and a FastConformer audio tower in one `model.safetensors`. It is
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
//! | `audio.encoder.*`, `audio.adapter.*`               | the LFM2-Audio encoder layout (`a.*`, `mm.a.mlp.*`) |
//! | `audio.residual.{ln,down,up}.*`                    | `d1.a.res.{norm,down,up}.*`      |
//!
//! The audio tower is written in the layout `crate::model::audio_encoder` already loads (the
//! same FastConformer lineage as LFM2-Audio), including its `clip.audio.*` settings, so that
//! encoder runs it unchanged. Two things are rewritten on the way: every batch norm is folded
//! into a scale and a shift (`a.blk.N.conv_norm.{weight,bias}`), and the singleton axis of the
//! 1-D convolution kernels is dropped.
//!
//! Metadata, all optional for a reader that only wants the trunk:
//!
//! * `d1.head.block_count`, `d1.head.attention.head_count`, `d1.head.feed_forward_length`,
//!   `d1.head.layer_norm_epsilon`: the head's pre-norm transformer layers;
//! * `d1.vision.block_count`, `d1.vision.embedding_length`, `d1.vision.feed_forward_length`,
//!   `d1.vision.attention.head_count`, `d1.vision.patch_size`, `d1.vision.position_side`,
//!   `d1.vision.layer_norm_epsilon`, `d1.vision.projector_hidden_length`: the tower;
//! * `clip.audio.*`: the audio encoder's depth, width, heads and mel bins, as that loader reads
//!   them; `d1.audio.residual_width`: the hidden width of the residual block after the adapter;
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
        || gguf_name.starts_with("d1.a.res")
        // the audio convolutions and the folded batch norms are small and numerically touchy
        || gguf_name.starts_with("a.conv1d.")
        || gguf_name.contains(".conv_dw.")
        || gguf_name.contains(".conv_pw")
        || gguf_name.contains(".conv_norm.")
}

/// Epsilon of `torch.nn.BatchNorm1d`, which the conformer's convolution module uses.
pub const BATCH_NORM_EPS: f64 = 1e-5;

/// What to do with a checkpoint tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorPlan {
    /// Leave it out.
    Skip,
    /// Write it under this GGUF name.
    Name(String),
    /// One of the four statistics of the batch norm of conformer layer `layer`; the four are
    /// folded into `a.blk.<layer>.conv_norm.{weight,bias}`.
    BatchNorm { layer: usize, stat: BatchNormStat },
}

/// The tensors of a batch norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchNormStat {
    Weight,
    Bias,
    Mean,
    Var,
}

/// Which half of a folded batch norm a derived tensor holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    /// `weight / sqrt(var + eps)`
    Scale,
    /// `bias - mean * scale`
    Shift,
}

/// Fold a batch norm (`y = (x - mean) / sqrt(var + eps) * weight + bias`) into `y = x * scale +
/// shift`, in f64.
pub fn fold_batch_norm(
    kind: Fold,
    weight: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
) -> Vec<f32> {
    (0..weight.len())
        .map(|i| {
            let scale = f64::from(weight[i]) / (f64::from(var[i]) + BATCH_NORM_EPS).sqrt();
            match kind {
                Fold::Scale => scale as f32,
                Fold::Shift => (f64::from(bias[i]) - f64::from(mean[i]) * scale) as f32,
            }
        })
        .collect()
}

/// The GGUF dimensions of a tensor: the reversed shape, with the singleton axis of the audio
/// 1-D convolution kernels dropped (a depthwise `[c, 1, k]` becomes `[k, c]`, a pointwise
/// `[o, i, 1]` becomes `[i, o]`).
pub fn tensor_dims(gguf_name: &str, shape: &[usize]) -> Vec<u64> {
    let mut dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
    if shape.len() == 3 {
        if gguf_name.ends_with(".conv_dw.weight") && shape[1] == 1 {
            dims.remove(1);
        } else if (gguf_name.ends_with(".conv_pw1.weight")
            || gguf_name.ends_with(".conv_pw2.weight"))
            && shape[2] == 1
        {
            dims.remove(0);
        }
    }
    dims
}

/// An audio tower tensor, or `None` for one that does not belong.
fn audio_plan(hf_name: &str) -> Option<TensorPlan> {
    if let Some(rest) = hf_name.strip_prefix("audio.residual.") {
        let (layer, kind) = rest.split_once('.')?;
        let name = match layer {
            "ln" => "norm",
            "down" | "up" => layer,
            _ => return None,
        };
        return matches!(kind, "weight" | "bias")
            .then(|| TensorPlan::Name(format!("d1.a.res.{name}.{kind}")));
    }
    if let Some(rest) = hf_name.strip_prefix("audio.adapter.") {
        let (layer, kind) = rest.split_once('.')?;
        let index = match layer {
            "norm" => 0,
            "linear_1" => 1,
            "linear_2" => 3,
            _ => return None,
        };
        return matches!(kind, "weight" | "bias")
            .then(|| TensorPlan::Name(format!("mm.a.mlp.{index}.{kind}")));
    }
    let rest = hf_name.strip_prefix("audio.encoder.")?;
    if let Some(rest) = rest.strip_prefix("pre_encode.") {
        if let Some(conv) = rest.strip_prefix("conv.") {
            let (index, kind) = conv.split_once('.')?;
            index.parse::<usize>().ok()?;
            return matches!(kind, "weight" | "bias")
                .then(|| TensorPlan::Name(format!("a.conv1d.{index}.{kind}")));
        }
        let kind = rest.strip_prefix("out.")?;
        return matches!(kind, "weight" | "bias")
            .then(|| TensorPlan::Name(format!("a.pre_encode.out.{kind}")));
    }
    let (layer, suffix) = rest.strip_prefix("layers.")?.split_once('.')?;
    let index: usize = layer.parse().ok()?;
    // the two positional biases carry no `.weight` suffix
    if matches!(suffix, "self_attn.pos_bias_u" | "self_attn.pos_bias_v") {
        let name = suffix.strip_prefix("self_attn.")?;
        return Some(TensorPlan::Name(format!("a.blk.{layer}.{name}")));
    }
    let (module, kind) = suffix.rsplit_once('.')?;
    if module == "conv.batch_norm" {
        let stat = match kind {
            "weight" => BatchNormStat::Weight,
            "bias" => BatchNormStat::Bias,
            "running_mean" => BatchNormStat::Mean,
            "running_var" => BatchNormStat::Var,
            "num_batches_tracked" => return Some(TensorPlan::Skip),
            _ => return None,
        };
        return Some(TensorPlan::BatchNorm { layer: index, stat });
    }
    if !matches!(kind, "weight" | "bias") {
        return None;
    }
    let name = match module {
        "feed_forward1.linear1" => "ffn_up",
        "feed_forward1.linear2" => "ffn_down",
        "norm_feed_forward1" => "ffn_norm",
        "feed_forward2.linear1" => "ffn_up_1",
        "feed_forward2.linear2" => "ffn_down_1",
        "norm_feed_forward2" => "ffn_norm_1",
        "norm_self_att" => "ln1",
        "norm_out" => "ln2",
        "norm_conv" => "norm_conv",
        "self_attn.linear_q" => "attn_q",
        "self_attn.linear_k" => "attn_k",
        "self_attn.linear_v" => "attn_v",
        "self_attn.linear_out" => "attn_out",
        "self_attn.linear_pos" => "linear_pos",
        "conv.pointwise_conv1" => "conv_pw1",
        "conv.depthwise_conv" => "conv_dw",
        "conv.pointwise_conv2" => "conv_pw2",
        _ => return None,
    };
    Some(TensorPlan::Name(format!("a.blk.{layer}.{name}.{kind}")))
}

/// What to do with a d1-omni checkpoint tensor.
pub fn tensor_plan(hf_name: &str) -> TensorPlan {
    if hf_name.starts_with("audio.") {
        return audio_plan(hf_name).unwrap_or(TensorPlan::Skip);
    }
    tensor_name(hf_name).map_or(TensorPlan::Skip, TensorPlan::Name)
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

/// The GGUF name of a d1-omni trunk, head or vision tensor, or `None` for any other (the audio
/// tower is planned by [`tensor_plan`]).
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
    if let Some(audio) = config.get("audio_config") {
        let field = |key: &str| {
            audio
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    CeraError::Backend(format!("d1-omni config.json: no `audio_config.{key}`"))
                })
        };
        let width = field("d_model")?;
        writer.add_bool("clip.has_audio_encoder", true);
        writer.add_u32("clip.audio.block_count", field("n_layers")?);
        writer.add_u32("clip.audio.embedding_length", width);
        writer.add_u32(
            "clip.audio.feed_forward_length",
            width * field("ff_expansion_factor")?,
        );
        writer.add_u32("clip.audio.attention.head_count", field("n_heads")?);
        writer.add_f32("clip.audio.attention.layer_norm_epsilon", 1e-5);
        writer.add_u32("clip.audio.num_mel_bins", field("feat_in")?);
        writer.add_u32("d1.audio.residual_width", field("residual_width")?);
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
            "head.head.layers.0.self_attn.mystery",
            "head.scorer.2.weight",
            "head.head.layers.x.norm1.weight",
        ] {
            assert_eq!(tensor_name(hf), None, "{hf}");
        }
    }

    #[test]
    fn audio_tensors_take_the_encoder_layout() {
        let name = |hf: &str| match tensor_plan(hf) {
            TensorPlan::Name(n) => Some(n),
            _ => None,
        };
        let cases = [
            (
                "audio.encoder.pre_encode.conv.3.weight",
                "a.conv1d.3.weight",
            ),
            ("audio.encoder.pre_encode.out.bias", "a.pre_encode.out.bias"),
            (
                "audio.encoder.layers.4.feed_forward1.linear1.weight",
                "a.blk.4.ffn_up.weight",
            ),
            (
                "audio.encoder.layers.4.feed_forward2.linear2.bias",
                "a.blk.4.ffn_down_1.bias",
            ),
            (
                "audio.encoder.layers.4.norm_self_att.weight",
                "a.blk.4.ln1.weight",
            ),
            ("audio.encoder.layers.16.norm_out.bias", "a.blk.16.ln2.bias"),
            (
                "audio.encoder.layers.0.self_attn.linear_pos.weight",
                "a.blk.0.linear_pos.weight",
            ),
            (
                "audio.encoder.layers.2.self_attn.pos_bias_u",
                "a.blk.2.pos_bias_u",
            ),
            (
                "audio.encoder.layers.2.conv.depthwise_conv.weight",
                "a.blk.2.conv_dw.weight",
            ),
            (
                "audio.encoder.layers.2.conv.pointwise_conv2.bias",
                "a.blk.2.conv_pw2.bias",
            ),
            ("audio.adapter.norm.weight", "mm.a.mlp.0.weight"),
            ("audio.adapter.linear_1.bias", "mm.a.mlp.1.bias"),
            ("audio.adapter.linear_2.weight", "mm.a.mlp.3.weight"),
            ("audio.residual.ln.weight", "d1.a.res.norm.weight"),
            ("audio.residual.up.bias", "d1.a.res.up.bias"),
        ];
        for (hf, gguf) in cases {
            assert_eq!(name(hf).as_deref(), Some(gguf), "{hf}");
        }
        for skip in [
            "audio.encoder.layers.0.conv.batch_norm.num_batches_tracked",
            "audio.encoder.layers.0.conv.mystery.weight",
            "audio.adapter.linear_3.weight",
            "audio.residual.gate.weight",
        ] {
            assert_eq!(tensor_plan(skip), TensorPlan::Skip, "{skip}");
        }
    }

    #[test]
    fn batch_norm_statistics_are_collected_per_layer() {
        for (hf, stat) in [
            ("weight", BatchNormStat::Weight),
            ("bias", BatchNormStat::Bias),
            ("running_mean", BatchNormStat::Mean),
            ("running_var", BatchNormStat::Var),
        ] {
            assert_eq!(
                tensor_plan(&format!("audio.encoder.layers.7.conv.batch_norm.{hf}")),
                TensorPlan::BatchNorm { layer: 7, stat }
            );
        }
    }

    #[test]
    fn a_folded_batch_norm_is_the_same_affine_map() {
        let (w, b, mean, var) = (
            [2.0f32, 0.5],
            [0.1f32, -0.3],
            [1.0f32, -2.0],
            [3.0f32, 0.25],
        );
        let scale = fold_batch_norm(Fold::Scale, &w, &b, &mean, &var);
        let shift = fold_batch_norm(Fold::Shift, &w, &b, &mean, &var);
        for i in 0..2 {
            for x in [-1.5f32, 0.0, 0.7, 4.0] {
                let direct = (x - mean[i]) / (var[i] + 1e-5).sqrt() * w[i] + b[i];
                let folded = x * scale[i] + shift[i];
                assert!((direct - folded).abs() < 1e-5, "{direct} vs {folded}");
            }
        }
    }

    #[test]
    fn the_singleton_axis_of_audio_kernels_is_dropped() {
        assert_eq!(
            tensor_dims("a.blk.0.conv_dw.weight", &[512, 1, 9]),
            vec![9, 512]
        );
        assert_eq!(
            tensor_dims("a.blk.0.conv_pw1.weight", &[1024, 512, 1]),
            vec![512, 1024]
        );
        assert_eq!(
            tensor_dims("a.blk.0.conv_pw2.weight", &[512, 512, 1]),
            vec![512, 512]
        );
        // the stem keeps its four axes, and ordinary matrices are just reversed
        assert_eq!(
            tensor_dims("a.conv1d.3.weight", &[256, 256, 1, 1]),
            vec![1, 1, 256, 256]
        );
        assert_eq!(
            tensor_dims("a.conv1d.0.weight", &[256, 1, 3, 3]),
            vec![3, 3, 1, 256]
        );
        assert_eq!(
            tensor_dims("a.blk.0.ffn_up.weight", &[2048, 512]),
            vec![512, 2048]
        );
    }

    #[test]
    fn the_scorer_and_question_types_stay_f32() {
        assert!(keeps_f32("d1.cls.output.weight"));
        assert!(keeps_f32("d1.cls.weight"));
        assert!(keeps_f32("d1.question_type.weight"));
        assert!(keeps_f32("d1.v.patch_embd.weight"));
        assert!(keeps_f32("a.conv1d.0.weight"));
        assert!(keeps_f32("a.blk.3.conv_dw.weight"));
        assert!(keeps_f32("a.blk.3.conv_norm.bias"));
        assert!(keeps_f32("d1.a.res.down.weight"));
        assert!(!keeps_f32("a.blk.3.ffn_up.weight"));
        assert!(!keeps_f32("mm.a.mlp.1.weight"));
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
            "audio_config": {
                "d_model": 512, "ff_expansion_factor": 4, "n_heads": 8, "n_layers": 17,
                "feat_in": 128, "residual_width": 512
            },
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
        assert!(writer.get_metadata("clip.audio.block_count").is_some());
        assert!(writer.get_metadata("d1.audio.residual_width").is_some());
        assert!(
            apply_metadata(
                &serde_json::json!({"head_layers": 2}),
                &mut GgufWriter::new()
            )
            .is_err()
        );
    }
}

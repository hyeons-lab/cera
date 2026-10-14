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

use crate::convert::safetensors::{gguf_tensor_dims, translate_hf_to_gguf_tensor_name_with_arch};
use crate::convert::writer::GgufWriter;
use crate::d1::head::HEAD_DIM;
use crate::d1::prompt::MIN_OPTION_BUDGET;
use crate::d1::vision::{PATCH, POSITION_SIDE};
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

/// The GGUF dimensions of a tensor: the canonical reversed shape (including the LFM2
/// short-convolution singleton squeeze), with the singleton axis of the audio 1-D
/// convolution kernels dropped on top (a depthwise `[c, 1, k]` becomes `[k, c]`, a
/// pointwise `[o, i, 1]` becomes `[i, o]`).
pub fn tensor_dims(gguf_name: &str, shape: &[usize]) -> Vec<u64> {
    let mut dims = gguf_tensor_dims(gguf_name, shape);
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
        return (matches!(n, "1" | "2") && matches!(kind, "weight" | "bias"))
            .then(|| format!("d1.v.mm.{n}.{kind}"));
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
        return matches!(kind, "weight" | "bias").then(|| format!("{part}.{kind}"));
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
    let Some(value) = config.get(key) else {
        return Err(CeraError::Backend(format!(
            "d1-omni config.json: `{key}` is missing"
        )));
    };
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| {
            CeraError::Backend(format!(
                "d1-omni config.json: `{key}` must be an integer 0..=u32::MAX"
            ))
        })
}

/// A count from a config section, with the same missing-vs-wrong-type split as [`count`].
fn section_count(section: &Value, section_name: &str, key: &str) -> Result<u32, CeraError> {
    let value = section.get(key).ok_or_else(|| {
        CeraError::Backend(format!("d1-omni config.json: no `{section_name}.{key}`"))
    })?;
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| {
            CeraError::Backend(format!(
                "d1-omni config.json: `{section_name}.{key}` must be an integer 0..=u32::MAX"
            ))
        })
}

/// Refuse a zero count the loaders cannot use, naming the config key as `count`
/// would (`section.key` for section fields, bare for top-level ones). Per-site,
/// not blanket: some block counts and unblocked widths may legitimately be 0
/// (a block-less vision tower), while the audio encoder needs both.
fn nonzero(value: u32, key: &str) -> Result<u32, CeraError> {
    if value == 0 {
        return Err(CeraError::Backend(format!(
            "d1-omni config.json: {key} is 0"
        )));
    }
    Ok(value)
}

/// Write the `d1.*` metadata of a d1-omni `config.json`.
///
/// # Errors
///
/// Refuses a config without the counts the head and the prompt budget are read from.
pub fn apply_metadata(config: &Value, writer: &mut GgufWriter) -> Result<(), CeraError> {
    let text = config.get("text_config").ok_or_else(|| {
        CeraError::Backend("d1-omni config.json: no `text_config.hidden_size`".into())
    })?;
    let hidden = section_count(text, "text_config", "hidden_size")?;
    let layers = count(config, "head_layers")?;
    writer.add_u32("d1.head.block_count", layers);
    // the head is `nn.TransformerEncoderLayer(d, d // HEAD_DIM, 4 * d)`
    let hidden = nonzero(hidden, "text_config.hidden_size")?;
    if hidden % (HEAD_DIM as u32) != 0 {
        return Err(CeraError::Backend(format!(
            "d1-omni config.json: text_config.hidden_size ({hidden}) is not a multiple of {HEAD_DIM}"
        )));
    }
    writer.add_u32("d1.head.attention.head_count", hidden / (HEAD_DIM as u32));
    let ffn = hidden.checked_mul(4).ok_or_else(|| {
        CeraError::Backend(format!(
            "d1-omni config.json: text_config.hidden_size ({hidden}) overflows u32 * 4"
        ))
    })?;
    writer.add_u32("d1.head.feed_forward_length", ffn);
    writer.add_f32("d1.head.layer_norm_epsilon", 1e-5);
    let max_length = count(config, "max_length")?;
    if max_length < MIN_OPTION_BUDGET as u32 {
        return Err(CeraError::Backend(format!(
            "d1-omni config.json: max_length ({max_length}) is below the {MIN_OPTION_BUDGET} the prompt needs"
        )));
    }
    writer.add_u32("d1.context_length", max_length);
    let image_text_length = nonzero(count(config, "image_text_length")?, "image_text_length")?;
    writer.add_u32("d1.image_text_length", image_text_length);
    let audio_text_length = nonzero(count(config, "audio_text_length")?, "audio_text_length")?;
    writer.add_u32("d1.audio_text_length", audio_text_length);
    if let Some(vision) = config.get("vision_config") {
        let field = |key: &str| section_count(vision, "vision_config", key);
        writer.add_u32("d1.vision.block_count", field("num_hidden_layers")?);
        let vision_width = nonzero(field("hidden_size")?, "vision_config.hidden_size")?;
        writer.add_u32("d1.vision.embedding_length", vision_width);
        // strict even for a block-less section: a 0 here is a config typo, while
        // the loader accepts a 0 width only where no block reads it
        let vision_ffn = nonzero(
            field("intermediate_size")?,
            "vision_config.intermediate_size",
        )?;
        writer.add_u32("d1.vision.feed_forward_length", vision_ffn);
        let vision_heads = nonzero(
            field("num_attention_heads")?,
            "vision_config.num_attention_heads",
        )?;
        writer.add_u32("d1.vision.attention.head_count", vision_heads);
        let patch = field("patch_size")?;
        if patch as usize != PATCH {
            return Err(CeraError::Backend(format!(
                "d1-omni config.json: vision_config.patch_size ({patch}) is not the {PATCH} px the tower reads"
            )));
        }
        writer.add_u32("d1.vision.patch_size", patch);
        // the position table is square: `num_patches` is its area
        let patches = field("num_patches")?;
        let side = (f64::from(patches).sqrt().round()) as u32;
        if u64::from(side) * u64::from(side) != u64::from(patches) {
            return Err(CeraError::Backend(format!(
                "d1-omni config.json: vision_config.num_patches ({patches}) is not a square"
            )));
        }
        if side as usize != POSITION_SIDE {
            return Err(CeraError::Backend(format!(
                "d1-omni config.json: vision_config.num_patches ({patches}) is not the {POSITION_SIDE} x {POSITION_SIDE} table the tower reads"
            )));
        }
        writer.add_u32("d1.vision.position_side", side);
        // missing means the default, but a present epsilon must be a positive
        // finite f32: a bad one is refused here with a clear message instead of
        // a confusing refusal at the final guard (and a +inf epsilon would
        // pass every guard silently)
        let eps = match vision.get("layer_norm_eps") {
            None => 1e-6,
            Some(value) => {
                let f = value.as_f64().ok_or_else(|| {
                    CeraError::Backend(
                        "d1-omni config.json: `vision_config.layer_norm_eps` must be a number"
                            .into(),
                    )
                })?;
                let e = f as f32;
                if !(e.is_finite() && e > 0.0) {
                    return Err(CeraError::Backend(format!(
                        "d1-omni config.json: `vision_config.layer_norm_eps` must be positive and finite, got {f}"
                    )));
                }
                e
            }
        };
        writer.add_f32("d1.vision.layer_norm_epsilon", eps);
        let projector_hidden = nonzero(
            count(config, "projector_hidden_size")?,
            "projector_hidden_size",
        )?;
        writer.add_u32("d1.vision.projector_hidden_length", projector_hidden);
    }
    if let Some(audio) = config.get("audio_config") {
        let field = |key: &str| section_count(audio, "audio_config", key);
        let width = nonzero(field("d_model")?, "audio_config.d_model")?;
        writer.add_bool("clip.has_audio_encoder", true);
        writer.add_u32(
            "clip.audio.block_count",
            nonzero(field("n_layers")?, "audio_config.n_layers")?,
        );
        writer.add_u32("clip.audio.embedding_length", width);
        let factor = field("ff_expansion_factor")?;
        let audio_ffn = width.checked_mul(factor).ok_or_else(|| {
            CeraError::Backend(format!(
                "d1-omni config.json: audio_config overflows u32: d_model ({width}) * ff_expansion_factor ({factor})"
            ))
        })?;
        writer.add_u32("clip.audio.feed_forward_length", audio_ffn);
        let audio_heads = nonzero(field("n_heads")?, "audio_config.n_heads")?;
        // past the zero gate, so the remainder is safe
        if width % audio_heads != 0 {
            return Err(CeraError::Backend(format!(
                "d1-omni config.json: audio_config.d_model ({width}) is not a multiple of n_heads ({audio_heads})"
            )));
        }
        writer.add_u32("clip.audio.attention.head_count", audio_heads);
        writer.add_f32("clip.audio.attention.layer_norm_epsilon", 1e-5);
        let feat_in = nonzero(field("feat_in")?, "audio_config.feat_in")?;
        writer.add_u32("clip.audio.num_mel_bins", feat_in);
        let residual = nonzero(field("residual_width")?, "audio_config.residual_width")?;
        writer.add_u32("d1.audio.residual_width", residual);
    }
    if let Some(temperatures) = config.get("temperatures") {
        let temperatures = temperatures.as_object().ok_or_else(|| {
            CeraError::Backend("d1-omni config.json: `temperatures` must be an object".into())
        })?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (key, value) in temperatures {
            let value = value.as_f64().ok_or_else(|| {
                CeraError::Backend(format!(
                    "d1-omni config.json: temperature `{key}` is not a number"
                ))
            })?;
            // the loader refuses what is not positive and finite, so refuse it
            // here: the check runs on the f32 the file stores, since 1e300 is a
            // finite f64 but an infinite f32
            let stored = value as f32;
            if !(stored.is_finite() && stored > 0.0) {
                return Err(CeraError::Backend(format!(
                    "d1-omni config.json: temperature `{key}` must be positive and finite, got {value}"
                )));
            }
            keys.push(key.clone());
            values.push(stored);
        }
        writer.add_string_array("d1.temperature.keys", keys);
        writer.add_f32_array("d1.temperature.values", values);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::writer::MetadataValue;

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
            "vision.projector.linear_1.running_mean",
            "vision.projector.linear_1.weight.extra",
            "vision.tower.vision_model.encoder.layers.0.mlp.fc3.weight",
            "vision.tower.vision_model.encoder.layers.x.layer_norm1.weight",
            "vision.tower.vision_model.encoder.layers.0.layer_norm1.running_mean",
        ] {
            assert_eq!(tensor_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn unknown_tensors_are_left_out() {
        for hf in [
            "head.head.layers.0.self_attn.mystery",
            "head.scorer.2.weight",
            "head.scorer.1.num_batches_tracked",
            "head.scorer.1.weight.extra",
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
        // the trunk inherits the canonical short-convolution squeeze
        assert_eq!(
            tensor_dims("blk.0.shortconv.conv.weight", &[512, 1, 9]),
            vec![9, 512]
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
        // the derived counts, not just their presence: 1024 wide means 16 heads
        assert_eq!(
            writer.get_metadata("d1.head.attention.head_count"),
            Some(&MetadataValue::Uint32(16))
        );
        assert_eq!(
            writer.get_metadata("d1.head.feed_forward_length"),
            Some(&MetadataValue::Uint32(4096))
        );
        assert_eq!(
            writer.get_metadata("d1.vision.position_side"),
            Some(&MetadataValue::Uint32(16))
        );
        assert_eq!(
            writer.get_metadata("clip.audio.feed_forward_length"),
            Some(&MetadataValue::Uint32(2048))
        );
        assert_eq!(
            writer.get_metadata("d1.temperature.keys"),
            Some(&MetadataValue::StringArray(vec![
                "choice".into(),
                "noul:2".into()
            ]))
        );
        assert_eq!(
            writer.get_metadata("d1.temperature.values"),
            Some(&MetadataValue::Float32Array(vec![1.0, 1.5]))
        );
        // the vision block count is read, never defaulted: 12 layers in, 12 out
        assert_eq!(
            writer.get_metadata("d1.vision.block_count"),
            Some(&MetadataValue::Uint32(12))
        );
        // ... and a missing epsilon defaults instead of refusing
        let mut noeps = config.clone();
        noeps["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_norm_eps");
        let mut writer = GgufWriter::new();
        apply_metadata(&noeps, &mut writer).unwrap();
        assert_eq!(
            writer.get_metadata("d1.vision.layer_norm_epsilon"),
            Some(&MetadataValue::Float32(1e-6))
        );
        // ... while a present one is carried, not defaulted over
        let mut custom = config.clone();
        custom["vision_config"]["layer_norm_eps"] = serde_json::json!(1e-5);
        let mut writer = GgufWriter::new();
        apply_metadata(&custom, &mut writer).unwrap();
        assert_eq!(
            writer.get_metadata("d1.vision.layer_norm_epsilon"),
            Some(&MetadataValue::Float32(1e-5))
        );
        assert!(
            apply_metadata(
                &serde_json::json!({"head_layers": 2}),
                &mut GgufWriter::new()
            )
            .is_err()
        );
    }

    #[test]
    fn metadata_refuses_counts_the_loaders_cannot_use() {
        // head_layers, max_length and the text rooms are valid in every leg
        let base = serde_json::json!({
            "head_layers": 2,
            "max_length": 16384,
            "image_text_length": 896,
            "audio_text_length": 15360,
            "text_config": {"hidden_size": 2048},
        });
        let with = |key: &str, value: serde_json::Value| {
            let mut config = base.clone();
            config[key] = value;
            config
        };
        let without = |key: &str| {
            let mut config = base.clone();
            config.as_object_mut().unwrap().remove(key);
            config
        };
        // every leg pins the guard that fires it, so a refactor cannot hand a leg to
        // a different guard while its target rots
        // one base section per tower, one-field patches per leg: a new required key
        // is added once instead of once per leg
        let vision_base = || {
            serde_json::json!({"num_hidden_layers": 1, "hidden_size": 64,
                "intermediate_size": 64, "num_attention_heads": 1,
                "patch_size": 16, "num_patches": 256})
        };
        let vision_with = |key: &str, value: serde_json::Value| {
            let mut section = vision_base();
            section[key] = value;
            with("vision_config", section)
        };
        let vision_without = |key: &str| {
            let mut section = vision_base();
            section.as_object_mut().unwrap().remove(key);
            with("vision_config", section)
        };
        let audio_base = || {
            serde_json::json!({"d_model": 64, "ff_expansion_factor": 4,
                "n_heads": 1, "n_layers": 1, "feat_in": 1, "residual_width": 1})
        };
        let audio_with = |key: &str, value: serde_json::Value| {
            let mut section = audio_base();
            section[key] = value;
            with("audio_config", section)
        };
        let audio_without = |key: &str| {
            let mut section = audio_base();
            section.as_object_mut().unwrap().remove(key);
            with("audio_config", section)
        };
        for (config, want) in [
            // a wrong-typed count names the type error, not a missing key
            (
                with("head_layers", serde_json::json!("2")),
                "must be an integer",
            ),
            // ... and a missing count names the missing key
            (without("head_layers"), "is missing"),
            // a zero hidden width passes the % 64 gate but loads nothing
            (
                with("text_config", serde_json::json!({"hidden_size": 0})),
                "is 0",
            ),
            // a missing hidden width names the missing key
            (
                with("text_config", serde_json::json!({})),
                "no `text_config.hidden_size`",
            ),
            // the prompt needs 96 positions of context
            (with("max_length", serde_json::json!(64)), "below the 96"),
            // ... one below the floor refuses too: the gate is `<`, not `<=`
            (with("max_length", serde_json::json!(95)), "below the 96"),
            // a zero text room would load, then blame the prompt for every refusal
            (
                with("image_text_length", serde_json::json!(0)),
                "image_text_length is 0",
            ),
            (
                with("audio_text_length", serde_json::json!(0)),
                "audio_text_length is 0",
            ),
            // hidden width that is not a multiple of 64 loses heads to truncation
            (
                with("text_config", serde_json::json!({"hidden_size": 1000})),
                "not a multiple of 64",
            ),
            // 4 * hidden must fit u32 (a multiple of 64, so the % 64 gate passes first)
            (
                with(
                    "text_config",
                    serde_json::json!({"hidden_size": u32::MAX - 63}),
                ),
                "overflows u32",
            ),
            // a wrong-typed hidden width names the type error too
            (
                with("text_config", serde_json::json!({"hidden_size": "2048"})),
                "must be an integer",
            ),
            // the position table must be square ...
            (
                vision_with("num_patches", serde_json::json!(257)),
                "not a square",
            ),
            // ... and 16 x 16, the only table the tower reads
            (
                vision_with("num_patches", serde_json::json!(100)),
                "16 x 16",
            ),
            // u32::MAX patches: the square itself must not overflow the check
            (
                vision_with("num_patches", serde_json::json!(u32::MAX)),
                "not a square",
            ),
            // the tower reads 16 px patches
            (vision_with("patch_size", serde_json::json!(32)), "16 px"),
            // a wrong-typed vision field names the type error, not a missing key
            (
                vision_with("patch_size", serde_json::json!("16")),
                "must be an integer",
            ),
            // a missing vision field names the missing key
            (
                vision_without("patch_size"),
                "no `vision_config.patch_size`",
            ),
            // the tower needs at least one head
            (
                vision_with("num_attention_heads", serde_json::json!(0)),
                "num_attention_heads is 0",
            ),
            // a zero feed-forward width loads 0-element block tensors
            (
                vision_with("intermediate_size", serde_json::json!(0)),
                "intermediate_size is 0",
            ),
            // a zero vision width converts every tensor, then refuses at load
            (
                vision_with("hidden_size", serde_json::json!(0)),
                "vision_config.hidden_size is 0",
            ),
            // a present-but-bad epsilon would NaN every vision prefix
            (
                vision_with("layer_norm_eps", serde_json::json!(-1.0)),
                "layer_norm_eps` must be positive and finite",
            ),
            // 1e300 is a finite f64 but an infinite f32: the check runs on the stored value
            (
                vision_with("layer_norm_eps", serde_json::json!(1e300)),
                "layer_norm_eps` must be positive and finite",
            ),
            // ... and a wrong-typed one must name the type error, not silently default
            (
                vision_with("layer_norm_eps", serde_json::json!("1e-6")),
                "layer_norm_eps` must be a number",
            ),
            // a zero projector width loads 0-element projector tensors (the guard
            // lives in the vision section, so the leg needs it present)
            (
                {
                    let mut config = with("vision_config", vision_base());
                    config["projector_hidden_size"] = serde_json::json!(0);
                    config
                },
                "projector_hidden_size is 0",
            ),
            // audio width times expansion must fit u32
            (
                audio_with("d_model", serde_json::json!(u32::MAX)),
                "overflows u32",
            ),
            // a zero audio width converts every tensor, then refuses at load
            (
                audio_with("d_model", serde_json::json!(0)),
                "audio_config.d_model is 0",
            ),
            // the audio encoder needs its blocks: zero loads nothing usable
            (
                audio_with("n_layers", serde_json::json!(0)),
                "audio_config.n_layers is 0",
            ),
            // a non-numeric temperature names the key, not a missing count
            (
                with("temperatures", serde_json::json!({"noul": "hot"})),
                "temperature `noul` is not a number",
            ),
            // a zero temperature converts cleanly, then refuses at load
            (
                with("temperatures", serde_json::json!({"noul": 0})),
                "temperature `noul` must be positive and finite",
            ),
            // ... as does a negative one
            (
                with("temperatures", serde_json::json!({"noul": -1.0})),
                "temperature `noul` must be positive and finite",
            ),
            // 1e300 is a finite f64 but an infinite f32: the check runs on the stored value
            (
                with("temperatures", serde_json::json!({"noul": 1e300})),
                "temperature `noul` must be positive and finite",
            ),
            // a missing vision block count names the key, never a default
            (
                vision_without("num_hidden_layers"),
                "no `vision_config.num_hidden_layers`",
            ),
            // ... and a missing audio expansion factor names its key too
            (
                audio_without("ff_expansion_factor"),
                "no `audio_config.ff_expansion_factor`",
            ),
            // a wrong-typed audio field names the type error, not a missing key
            (
                audio_with("n_heads", serde_json::json!("1")),
                "must be an integer",
            ),
            // a missing audio field names the missing key
            (audio_without("n_heads"), "no `audio_config.n_heads`"),
            // the encoder needs at least one head
            (audio_with("n_heads", serde_json::json!(0)), "n_heads is 0"),
            // heads that do not divide the width would panic in attention
            // (the base d_model of 64 is indivisible by 3)
            (
                audio_with("n_heads", serde_json::json!(3)),
                "not a multiple of n_heads",
            ),
            // no mel bins means no frames for any clip, scored as an empty prefix
            (audio_with("feat_in", serde_json::json!(0)), "feat_in is 0"),
            // a zero residual width would load, then answer through a degenerate adapter
            (
                audio_with("residual_width", serde_json::json!(0)),
                "residual_width is 0",
            ),
        ] {
            let err = apply_metadata(&config, &mut GgufWriter::new()).unwrap_err();
            assert!(err.to_string().contains(want), "unexpected message: {err}");
        }
        // ... and exactly the floor converts
        apply_metadata(
            &with("max_length", serde_json::json!(96)),
            &mut GgufWriter::new(),
        )
        .unwrap();
    }
}

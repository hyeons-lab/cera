//! Multimodal projector (`mmproj`) SafeTensors-to-GGUF converter for LFM3-VL and LFM3-Audio.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::convert::quantize::{QuantStrategy, TargetQuant, quantize_tensor_data_with_strategy};
use crate::convert::safetensors::{
    SafeTensorsHeader, decode_safetensor_to_f32_into, gguf_tensor_dims,
};
use crate::convert::writer::{GGML_TYPE_F16, GGML_TYPE_F32, GgufWriter};
use crate::session::CeraError;

/// Prepared GGUF tensor ready to be written into an `mmproj` file.
#[derive(Debug, Clone)]
pub struct MmprojTensor {
    pub name: String,
    /// Row-major shape `[d0, d1, ...]` before reversing to GGUF dims.
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub force_f32: bool,
}

impl MmprojTensor {
    pub fn gguf_dims(&self) -> Vec<u64> {
        if (self.name.ends_with("shortconv.conv.weight") || self.name.ends_with("conv_dw.weight"))
            && self.shape.len() == 3
            && self.shape[1] == 1
        {
            vec![self.shape[2] as u64, self.shape[0] as u64]
        } else {
            gguf_tensor_dims(&self.name, &self.shape)
        }
    }
}

/// Validate that `shape` product does not overflow `usize` and equals `data.len()`.
fn validate_tensor_buffer(name: &str, shape: &[usize], data: &[f32]) -> Result<(), CeraError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| CeraError::Backend(format!("shape overflow for `{name}`: {shape:?}")))?;
    if data.len() != expected {
        return Err(CeraError::Backend(format!(
            "`{name}` data len {} != shape product {expected} ({shape:?})",
            data.len()
        )));
    }
    Ok(())
}

/// Convert the vision or audio multimodal components of an LFM3-VL or LFM3-Audio
/// checkpoint directory into a standalone `mmproj` GGUF file.
pub fn convert_mmproj(
    input_path: &Path,
    output_gguf_path: &Path,
    quant: TargetQuant,
) -> Result<(), CeraError> {
    if !input_path.exists() {
        return Err(CeraError::Backend(format!(
            "input path `{}` does not exist",
            input_path.display()
        )));
    }
    let dir = if input_path.is_dir() {
        input_path.to_path_buf()
    } else {
        input_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };

    let config_path = dir.join("config.json");
    if !config_path.exists() {
        return Err(CeraError::Backend(format!(
            "config.json not found in `{}`",
            dir.display()
        )));
    }
    let config_bytes = fs::read(&config_path).map_err(|e| {
        CeraError::Backend(format!("failed to read `{}`: {e}", config_path.display()))
    })?;
    let root_cfg: Value = serde_json::from_slice(&config_bytes).map_err(|e| {
        CeraError::Backend(format!("failed to parse `{}`: {e}", config_path.display()))
    })?;

    let preproc_cfg = read_preprocessor_config(&dir)?;
    let detok_cfg = match read_optional_json(&dir.join("audio_detokenizer").join("config.json"))? {
        Some(cfg) => Some(cfg),
        None => read_optional_json(&dir.join("audio_tokenizer").join("config.json"))?,
    };

    let mut raw_tensors = load_checkpoint_tensors(&dir, input_path)?;
    // Also load `audio_detokenizer/model.safetensors` or `audio_tokenizer/model.safetensors`
    // under `detokenizer.` prefix if present.
    for sub in ["audio_detokenizer", "audio_tokenizer"] {
        let detok_st = dir.join(sub).join("model.safetensors");
        if detok_st.is_file() {
            for (name, (shape, data)) in load_safetensors_file(&detok_st, &|_| true)? {
                raw_tensors.insert(format!("detokenizer.{name}"), (shape, data));
            }
            break;
        }
    }

    let model_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("mmproj");
    let mut writer = GgufWriter::new();
    let kind = detect_mmproj_kind(&root_cfg)?;
    let tensors = match kind {
        MmprojKind::Vision => {
            apply_lfm3_vl_mmproj_metadata(
                &root_cfg,
                preproc_cfg.as_ref(),
                &mut writer,
                model_name,
            )?;
            build_lfm3_vl_mmproj_tensors(&root_cfg, &raw_tensors)?
        }
        MmprojKind::Audio => {
            let has_detok =
                detok_cfg.is_some() || raw_tensors.keys().any(|k| k.starts_with("detokenizer."));
            apply_lfm3_audio_mmproj_metadata(
                &root_cfg,
                detok_cfg.as_ref(),
                has_detok,
                &mut writer,
                model_name,
            )?;
            build_lfm3_audio_mmproj_tensors(&root_cfg, detok_cfg.as_ref(), &raw_tensors)?
        }
        MmprojKind::Omni => {
            let has_detok =
                detok_cfg.is_some() || raw_tensors.keys().any(|k| k.starts_with("detokenizer."));
            apply_lfm3_audio_mmproj_metadata(
                &root_cfg,
                detok_cfg.as_ref(),
                has_detok,
                &mut writer,
                model_name,
            )?;
            apply_lfm3_vl_mmproj_metadata_inner(
                &root_cfg,
                preproc_cfg.as_ref(),
                &mut writer,
                model_name,
                true,
            )?;
            let mut all =
                build_lfm3_audio_mmproj_tensors(&root_cfg, detok_cfg.as_ref(), &raw_tensors)?;
            all.extend(build_lfm3_vl_mmproj_tensors(&root_cfg, &raw_tensors)?);
            all
        }
    };

    write_mmproj_gguf(output_gguf_path, &mut writer, &tensors, quant)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmprojKind {
    Vision,
    Audio,
    Omni,
}

pub fn detect_mmproj_kind(root_cfg: &Value) -> Result<MmprojKind, CeraError> {
    let model_type = root_cfg
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let arch = root_cfg
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    let has_vision = model_type.contains("vl")
        || arch.contains("vl")
        || root_cfg.get("vision_config").is_some()
        || root_cfg.get("vision_tower").is_some()
        || root_cfg.get("vision").is_some();
    let has_audio = model_type.contains("audio")
        || arch.contains("audio")
        || root_cfg.get("encoder").is_some()
        || root_cfg.get("audio_encoder_config").is_some()
        || root_cfg.get("depth_config").is_some()
        || root_cfg.get("conformer").is_some()
        || root_cfg.get("audio_tower").is_some();

    if model_type.contains("omni") || arch.contains("omni") || (has_vision && has_audio) {
        return Ok(MmprojKind::Omni);
    }
    if has_vision {
        return Ok(MmprojKind::Vision);
    }
    if has_audio {
        return Ok(MmprojKind::Audio);
    }
    Err(CeraError::Backend(
        "config.json is neither an LFM3-VL nor an LFM3-Audio multimodal checkpoint".into(),
    ))
}

/// Apply GGUF metadata for an LFM3-VL vision projector (`clip` architecture).
pub fn apply_lfm3_vl_mmproj_metadata(
    root_cfg: &Value,
    preproc_cfg: Option<&Value>,
    writer: &mut GgufWriter,
    model_name: &str,
) -> Result<(), CeraError> {
    apply_lfm3_vl_mmproj_metadata_inner(root_cfg, preproc_cfg, writer, model_name, false)
}

fn apply_lfm3_vl_mmproj_metadata_inner(
    root_cfg: &Value,
    preproc_cfg: Option<&Value>,
    writer: &mut GgufWriter,
    model_name: &str,
    is_omni: bool,
) -> Result<(), CeraError> {
    let vcfg = root_cfg
        .get("vision_config")
        .or_else(|| root_cfg.get("vision_tower"))
        .or_else(|| root_cfg.get("vision"))
        .unwrap_or(root_cfg);
    let tcfg = root_cfg
        .get("text_config")
        .or_else(|| root_cfg.get("language_config"))
        .or_else(|| root_cfg.get("lfm"))
        .unwrap_or(root_cfg);

    if !is_omni {
        writer.add_string("general.architecture", "clip");
        writer.add_string("general.name", model_name);
        writer.add_bool("clip.has_vision_encoder", true);
        writer.add_bool("clip.has_audio_encoder", false);
        writer.add_string("clip.projector_type", "lfm3vl");

        let is_swiglu = vcfg
            .get("hidden_act")
            .and_then(Value::as_str)
            .is_some_and(|a| a.contains("silu") || a.contains("swiglu"));
        writer.add_bool("clip.use_gelu", !is_swiglu);
        writer.add_bool("clip.use_silu", false);
    } else {
        writer.add_bool("clip.has_vision_encoder", true);
        writer.add_string("clip.vision.projector_type", "lfm3o");
    }

    let hidden_size = narrowed(
        "hidden_size",
        vcfg.get("vision_hidden_size")
            .or_else(|| vcfg.get("hidden_size")),
        768,
    )?;
    let ffn_size = narrowed(
        "intermediate_size",
        vcfg.get("vision_intermediate_size")
            .or_else(|| vcfg.get("intermediate_size")),
        3072,
    )?;
    let layers = narrowed(
        "num_hidden_layers",
        vcfg.get("vision_num_hidden_layers")
            .or_else(|| vcfg.get("num_hidden_layers")),
        12,
    )?;
    let heads = narrowed(
        "num_attention_heads",
        vcfg.get("vision_num_attention_heads")
            .or_else(|| vcfg.get("num_attention_heads")),
        12,
    )?;
    let kv_heads = narrowed(
        "num_key_value_heads",
        vcfg.get("vision_num_key_value_heads")
            .or_else(|| vcfg.get("num_key_value_heads")),
        heads,
    )?;
    let rope_theta = vcfg
        .get("rope_parameters")
        .and_then(|rp| rp.get("rope_theta").or_else(|| rp.get("theta")))
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or_else(|| f32_or(vcfg, &["rope_theta", "vision_rope_theta"], 1_000_000.0));
    let eps = f32_or(vcfg, &["layer_norm_eps", "rms_norm_eps", "norm_eps"], 1e-5);
    let proj_dim = narrowed(
        "hidden_size",
        tcfg.get("hidden_size")
            .or_else(|| vcfg.get("projection_dim")),
        2048,
    )?;
    let patch_size = narrowed(
        "patch_size",
        vcfg.get("patch_size")
            .or_else(|| vcfg.get("vision_patch_size")),
        16,
    )?;
    let image_size = narrowed("image_size", vcfg.get("image_size"), 512)?;

    writer.add_u32("clip.vision.embedding_length", hidden_size);
    writer.add_u32("clip.vision.feed_forward_length", ffn_size);
    writer.add_u32("clip.vision.block_count", layers);
    writer.add_u32("clip.vision.attention.head_count", heads);
    writer.add_u32("clip.vision.attention.head_count_kv", kv_heads);
    writer.add_f32("clip.vision.rope_theta", rope_theta);
    writer.add_f32("clip.vision.attention.layer_norm_epsilon", eps);
    writer.add_u32("clip.vision.projection_dim", proj_dim);
    writer.add_u32("clip.vision.patch_size", patch_size);
    writer.add_u32("clip.vision.image_size", image_size);
    writer.add_bool("clip.vision.use_silu", true);
    writer.add_string("clip.vision.image_resize_algo", "bilinear");

    let ip_cfg = preproc_cfg
        .and_then(|p| p.get("image_processor").or(Some(p)))
        .unwrap_or(vcfg);
    let mean = f32_vec_or(ip_cfg, "image_mean", &[0.5, 0.5, 0.5]);
    let std = f32_vec_or(ip_cfg, "image_std", &[0.5, 0.5, 0.5]);
    writer.add_f32_array("clip.vision.image_mean", mean);
    writer.add_f32_array("clip.vision.image_std", std);

    let downsample = narrowed(
        "downsample_factor",
        ip_cfg
            .get("downsample_factor")
            .or_else(|| ip_cfg.get("merge_size"))
            .or_else(|| vcfg.get("downsample_factor"))
            .or_else(|| vcfg.get("spatial_merge_size"))
            .or_else(|| vcfg.get("vision_spatial_merge_size"))
            .or_else(|| root_cfg.get("downsample_factor")),
        2,
    )?;
    writer.add_u32("clip.vision.spatial_merge_size", downsample);
    writer.add_u32("clip.vision.projector.scale_factor", downsample);

    // a token budget is `tokens * (patch * downsample)^2` pixels: checked, as the config is
    // untrusted and a wrapped product would write a different pixel bound
    let side = patch_size.checked_mul(downsample);
    let token_pixels = |key: &str, tokens: u32| -> Result<u32, CeraError> {
        side.and_then(|side| tokens.checked_mul(side)?.checked_mul(side))
            .ok_or_else(|| {
                CeraError::Backend(format!(
                    "mmproj config `{key}` ({tokens}) overflows u32 pixels at patch {patch_size} x \
                     downsample {downsample}"
                ))
            })
    };
    let pixel_bound =
        |direct: &str, edge: &str, tokens_key: &str| -> Result<Option<u32>, CeraError> {
            if let Some(v) = ip_cfg
                .get(direct)
                .or_else(|| ip_cfg.get("size").and_then(|s| s.get(edge)))
            {
                return narrowed_opt(direct, Some(v));
            }
            match narrowed_opt(tokens_key, ip_cfg.get(tokens_key))? {
                Some(tokens) => token_pixels(tokens_key, tokens).map(Some),
                None => Ok(None),
            }
        };
    if let Some(px) = pixel_bound("min_pixels", "shortest_edge", "min_image_tokens")? {
        writer.add_u32("clip.vision.image_min_pixels", px);
    }
    if let Some(px) = pixel_bound("max_pixels", "longest_edge", "max_image_tokens")? {
        writer.add_u32("clip.vision.image_max_pixels", px);
    }
    if let Some(tile_size) = narrowed_opt("tile_size", ip_cfg.get("tile_size"))? {
        writer.add_u32("clip.vision.preproc_image_size", tile_size);
    }
    if let Some(max_tiles) = narrowed_opt("max_tiles", ip_cfg.get("max_tiles"))? {
        writer.add_u32("clip.vision.max_tiles", max_tiles);
    }
    if let Some(min_tiles) = narrowed_opt("min_tiles", ip_cfg.get("min_tiles"))? {
        writer.add_u32("clip.vision.min_tiles", min_tiles);
    }
    if let Some(use_thumb) = ip_cfg.get("use_thumbnail").and_then(Value::as_bool) {
        writer.add_bool("clip.vision.use_thumbnail", use_thumb);
    }
    Ok(())
}

/// Transform LFM3-VL vision encoder and multimodal projector tensors into GGUF `mmproj` tensors.
pub fn build_lfm3_vl_mmproj_tensors(
    root_cfg: &Value,
    raw_tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
) -> Result<Vec<MmprojTensor>, CeraError> {
    let vcfg = root_cfg
        .get("vision_config")
        .or_else(|| root_cfg.get("vision_tower"))
        .or_else(|| root_cfg.get("vision"))
        .unwrap_or(root_cfg);
    let patch_size = narrowed(
        "patch_size",
        vcfg.get("patch_size")
            .or_else(|| vcfg.get("vision_patch_size")),
        16,
    )? as usize;
    let num_channels = narrowed(
        "num_channels",
        vcfg.get("num_channels")
            .or_else(|| vcfg.get("in_channels"))
            .or_else(|| vcfg.get("vision_num_channels")),
        3,
    )? as usize;
    if patch_size == 0 || num_channels == 0 {
        return Err(CeraError::Backend(format!(
            "mmproj vision config `patch_size` ({patch_size}) and `num_channels` ({num_channels}) must be > 0"
        )));
    }
    let patch_area = patch_size.checked_mul(patch_size).ok_or_else(|| {
        CeraError::Backend(format!(
            "mmproj vision config `patch_size` ({patch_size}) overflows patch_area"
        ))
    })?;
    let patch_vol = num_channels.checked_mul(patch_area).ok_or_else(|| {
        CeraError::Backend(format!(
            "mmproj vision config `num_channels * patch_area` ({num_channels} * {patch_area}) overflows usize"
        ))
    })?;

    let mut out = Vec::new();
    for (raw_name, (shape, data)) in raw_tensors {
        validate_tensor_buffer(raw_name, shape, data)?;
        let name_after_model = raw_name.strip_prefix("model.").unwrap_or(raw_name);
        let in_vision_scope = name_after_model.starts_with("vision_tower.")
            || name_after_model.starts_with("vision_encoder.");
        let name = name_after_model
            .strip_prefix("vision_tower.")
            .unwrap_or(name_after_model);

        // Native LFM3-VL temporal patchifier `[hidden, 2 * patch_dim]` -> split into `v.patch_embd.weight` and `v.patch_embd.weight.1`
        if name == "vision_encoder.patchifier.weight"
            || name == "vision_encoder.patchifier.proj.weight"
            || (in_vision_scope
                && (name == "patchifier.weight" || name == "patchifier.proj.weight"))
        {
            if shape.len() != 2 || !shape[1].is_multiple_of(2) {
                return Err(CeraError::Backend(format!(
                    "`{raw_name}` expected 2D shape with even columns, got {shape:?}"
                )));
            }
            let (rows, cols) = (shape[0], shape[1]);
            let half = cols / 2;
            let mut w0 = Vec::with_capacity(rows * half);
            let mut w1 = Vec::with_capacity(rows * half);
            if (name == "vision_encoder.patchifier.weight" || name == "patchifier.weight")
                && half == patch_vol
            {
                // `[n_embd, 3, 2, p, p]` layout -> split temporal slice 0 and slice 1 per channel
                for r in 0..rows {
                    for c in 0..num_channels {
                        let base = (r * num_channels + c) * (2 * patch_area);
                        w0.extend_from_slice(&data[base..base + patch_area]);
                        w1.extend_from_slice(&data[base + patch_area..base + 2 * patch_area]);
                    }
                }
            } else {
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    w0.extend_from_slice(&row[..half]);
                    w1.extend_from_slice(&row[half..]);
                }
            }
            let pe_shape = if half == patch_vol {
                vec![rows, num_channels, patch_size, patch_size]
            } else {
                vec![rows, half]
            };
            out.push(MmprojTensor {
                name: "v.patch_embd.weight".into(),
                shape: pe_shape.clone(),
                data: w0,
                force_f32: false,
            });
            out.push(MmprojTensor {
                name: "v.patch_embd.weight.1".into(),
                shape: pe_shape,
                data: w1,
                force_f32: false,
            });
            continue;
        }
        if name == "vision_encoder.patchifier.bias"
            || name == "vision_encoder.patchifier.proj.bias"
            || name == "vision_tower.embeddings.patch_embeddings.bias"
            || name == "embeddings.patch_embeddings.bias"
            || (in_vision_scope && (name == "patchifier.bias" || name == "patchifier.proj.bias"))
        {
            out.push(MmprojTensor {
                name: "v.patch_embd.bias".into(),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: true,
            });
            continue;
        }
        if name == "vision_encoder.embedding_norm.weight"
            || name == "vision_encoder.patchifier.norm.weight"
            || name == "vision_tower.embeddings.layer_norm.weight"
            || name == "embeddings.layer_norm.weight"
            || (in_vision_scope
                && (name == "embedding_norm.weight" || name == "patchifier.norm.weight"))
        {
            out.push(MmprojTensor {
                name: "v.embedding_norm.weight".into(),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: true,
            });
            continue;
        }
        if name == "vision_encoder.embedding_norm.bias"
            || name == "vision_encoder.patchifier.norm.bias"
            || name == "vision_tower.embeddings.layer_norm.bias"
            || name == "embeddings.layer_norm.bias"
            || (in_vision_scope
                && (name == "embedding_norm.bias" || name == "patchifier.norm.bias"))
        {
            out.push(MmprojTensor {
                name: "v.embedding_norm.bias".into(),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: true,
            });
            continue;
        }
        if name == "vision_encoder.pos_emb"
            || name == "vision_tower.embeddings.position_embedding.weight"
            || name == "embeddings.position_embedding.weight"
            || (in_vision_scope && name == "pos_emb")
        {
            let squeezed_shape = if shape.len() == 3 && shape[0] == 1 {
                vec![shape[1], shape[2]]
            } else {
                shape.clone()
            };
            out.push(MmprojTensor {
                name: "v.position_embd.weight".into(),
                shape: squeezed_shape,
                data: data.clone(),
                force_f32: true,
            });
            continue;
        }
        if name == "vision_tower.embeddings.patch_embeddings.weight"
            || name == "embeddings.patch_embeddings.weight"
        {
            // Reshape 2D `[hidden, C * P * P]` or 3D `[hidden, C, P * P]` to 4D `[hidden, C, P, P]`
            let target_shape = if shape.len() == 2 && shape[1] == patch_vol {
                vec![shape[0], num_channels, patch_size, patch_size]
            } else if shape.len() == 3 && shape[2] == patch_area {
                vec![shape[0], shape[1], patch_size, patch_size]
            } else {
                shape.clone()
            };
            out.push(MmprojTensor {
                name: "v.patch_embd.weight".into(),
                shape: target_shape,
                data: data.clone(),
                force_f32: false,
            });
            continue;
        }

        // Spatial conv2d tensor: reshape 3D `[out_c, in_c, k*k]` to 4D `[out_c, in_c, k, k]`
        if let Some(tail) = name.strip_prefix("vision_feature_layer.conv.") {
            let target_shape = reshape_spatial_conv_3d_to_4d(shape);
            out.push(MmprojTensor {
                name: format!("mm.conv.{tail}"),
                shape: target_shape,
                data: data.clone(),
                force_f32: tail == "bias",
            });
            continue;
        }

        // Vision encoder blocks (`vision_encoder.layers.{i}.*`, `model.vision_tower.layers.{i}.*`, `vision_tower.encoder.layers.{i}.*`, or `vision_encoder.blocks.{i}.*`)
        if let Some(rest) = name
            .strip_prefix("vision_encoder.layers.")
            .or_else(|| name.strip_prefix("vision_tower.encoder.layers."))
            .or_else(|| name.strip_prefix("encoder.layers."))
            .or_else(|| name.strip_prefix("vision_encoder.blocks."))
            .or_else(|| {
                in_vision_scope
                    .then(|| name.strip_prefix("layers."))
                    .flatten()
            })
            .or_else(|| {
                in_vision_scope
                    .then(|| name.strip_prefix("blocks."))
                    .flatten()
            })
            && let Some((layer_str, sub)) = rest.split_once('.')
            && let Ok(bid) = layer_str.parse::<usize>()
        {
            if let Some(kind) = sub
                .strip_prefix("attn.qkv_proj.")
                .or_else(|| sub.strip_prefix("self_attn.qkv_proj."))
            {
                if shape.is_empty() || !shape[0].is_multiple_of(3) {
                    return Err(CeraError::Backend(format!(
                        "`{raw_name}` expected first dim divisible by 3, got {shape:?}"
                    )));
                }
                let third_rows = shape[0] / 3;
                let cols: usize = shape[1..].iter().product();
                let chunk_len = third_rows * cols;
                let mut sub_shape = shape.clone();
                sub_shape[0] = third_rows;
                for (idx, tag) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
                    out.push(MmprojTensor {
                        name: format!("v.blk.{bid}.{tag}.{kind}"),
                        shape: sub_shape.clone(),
                        data: data[idx * chunk_len..(idx + 1) * chunk_len].to_vec(),
                        force_f32: kind == "bias",
                    });
                }
                continue;
            }
            if sub == "ffn.gate_up.weight" || sub == "feed_forward.gate_up.weight" {
                if shape.len() != 2 || !shape[0].is_multiple_of(2) {
                    return Err(CeraError::Backend(format!(
                        "`{raw_name}` expected 2D even rows, got {shape:?}"
                    )));
                }
                let half_rows = shape[0] / 2;
                let half_elems = half_rows * shape[1];
                out.push(MmprojTensor {
                    name: format!("v.blk.{bid}.ffn_gate.weight"),
                    shape: vec![half_rows, shape[1]],
                    data: data[..half_elems].to_vec(),
                    force_f32: false,
                });
                out.push(MmprojTensor {
                    name: format!("v.blk.{bid}.ffn_up.weight"),
                    shape: vec![half_rows, shape[1]],
                    data: data[half_elems..].to_vec(),
                    force_f32: false,
                });
                continue;
            }
            let mapped_sub = match sub {
                "self_attn.q_proj.weight" | "attn.q_proj.weight" => Some("attn_q.weight"),
                "self_attn.q_proj.bias" | "attn.q_proj.bias" => Some("attn_q.bias"),
                "self_attn.k_proj.weight" | "attn.k_proj.weight" => Some("attn_k.weight"),
                "self_attn.k_proj.bias" | "attn.k_proj.bias" => Some("attn_k.bias"),
                "self_attn.v_proj.weight" | "attn.v_proj.weight" => Some("attn_v.weight"),
                "self_attn.v_proj.bias" | "attn.v_proj.bias" => Some("attn_v.bias"),
                "self_attn.out_proj.weight" | "attn.out_proj.weight" => Some("attn_out.weight"),
                "self_attn.out_proj.bias" | "attn.out_proj.bias" => Some("attn_out.bias"),
                "attn.q_norm.weight"
                | "self_attn.q_norm.weight"
                | "self_attn.q_layernorm.weight" => Some("attn_q_norm.weight"),
                "attn.q_norm.bias" | "self_attn.q_norm.bias" | "self_attn.q_layernorm.bias" => {
                    Some("attn_q_norm.bias")
                }
                "attn.k_norm.weight"
                | "self_attn.k_norm.weight"
                | "self_attn.k_layernorm.weight" => Some("attn_k_norm.weight"),
                "attn.k_norm.bias" | "self_attn.k_norm.bias" | "self_attn.k_layernorm.bias" => {
                    Some("attn_k_norm.bias")
                }
                "conv.in_proj.weight" | "conv2d_in_proj.weight" => Some("conv2d_in_proj.weight"),
                "conv.in_proj.bias" | "conv2d_in_proj.bias" => Some("conv2d_in_proj.bias"),
                "conv.conv.weight"
                | "conv.spatial_conv"
                | "conv.spatial_conv.weight"
                | "conv2d_spatial.weight" => Some("conv2d_spatial.weight"),
                "conv.conv.bias" | "conv.spatial_conv.bias" | "conv2d_spatial.bias" => {
                    Some("conv2d_spatial.bias")
                }
                "conv.out_proj.weight" | "conv2d_out_proj.weight" => Some("conv2d_out_proj.weight"),
                "conv.out_proj.bias" | "conv2d_out_proj.bias" => Some("conv2d_out_proj.bias"),
                "layer_norm1.weight" | "operator_norm.weight" => Some("ln1.weight"),
                "layer_norm1.bias" | "operator_norm.bias" => Some("ln1.bias"),
                "layer_norm2.weight" | "ffn_norm.weight" => Some("ln2.weight"),
                "layer_norm2.bias" | "ffn_norm.bias" => Some("ln2.bias"),
                "mlp.gate_proj.weight" | "ffn.w1.weight" | "feed_forward.w1.weight" => {
                    Some("ffn_gate.weight")
                }
                "mlp.gate_proj.bias" | "ffn.w1.bias" | "feed_forward.w1.bias" => {
                    Some("ffn_gate.bias")
                }
                "mlp.fc1.weight"
                | "mlp.up_proj.weight"
                | "ffn.w3.weight"
                | "feed_forward.w3.weight" => Some("ffn_up.weight"),
                "mlp.fc1.bias" | "mlp.up_proj.bias" | "ffn.w3.bias" | "feed_forward.w3.bias" => {
                    Some("ffn_up.bias")
                }
                "mlp.fc2.weight"
                | "mlp.down_proj.weight"
                | "ffn.down.weight"
                | "ffn.w2.weight"
                | "feed_forward.w2.weight" => Some("ffn_down.weight"),
                "mlp.fc2.bias"
                | "mlp.down_proj.bias"
                | "ffn.down.bias"
                | "ffn.w2.bias"
                | "feed_forward.w2.bias" => Some("ffn_down.bias"),
                _ => None,
            };
            if let Some(suffix) = mapped_sub {
                let target_shape = if suffix == "conv2d_spatial.weight" {
                    reshape_spatial_conv_3d_to_4d(shape)
                } else {
                    shape.clone()
                };
                let is_1d = target_shape.len() == 1;
                out.push(MmprojTensor {
                    name: format!("v.blk.{bid}.{suffix}"),
                    shape: target_shape,
                    data: data.clone(),
                    force_f32: is_1d || suffix == "conv2d_spatial.weight",
                });
            }
            continue;
        }

        let direct = match name {
            "vision_tower.post_layernorm.weight"
            | "post_layernorm.weight"
            | "vision_encoder.final_norm.weight" => Some("v.post_ln.weight"),
            "final_norm.weight" if in_vision_scope => Some("v.post_ln.weight"),
            "vision_tower.post_layernorm.bias"
            | "post_layernorm.bias"
            | "vision_encoder.final_norm.bias" => Some("v.post_ln.bias"),
            "final_norm.bias" if in_vision_scope => Some("v.post_ln.bias"),
            "multi_modal_projector.layer_norm.weight"
            | "vision_projector.layer_norm.weight"
            | "vision_projector.norm.weight" => Some("mm.input_norm.weight"),
            "multi_modal_projector.layer_norm.bias"
            | "vision_projector.layer_norm.bias"
            | "vision_projector.norm.bias" => Some("mm.input_norm.bias"),
            "multi_modal_projector.linear_1.weight"
            | "vision_projector.linear_1.weight"
            | "vision_projector.fc1.weight" => Some("mm.0.weight"),
            "multi_modal_projector.linear_1.bias"
            | "vision_projector.linear_1.bias"
            | "vision_projector.fc1.bias" => Some("mm.0.bias"),
            "multi_modal_projector.linear_2.weight"
            | "vision_projector.linear_2.weight"
            | "vision_projector.fc2.weight" => Some("mm.2.weight"),
            "multi_modal_projector.linear_2.bias"
            | "vision_projector.linear_2.bias"
            | "vision_projector.fc2.bias" => Some("mm.2.bias"),
            _ => None,
        };
        if let Some(gguf_name) = direct {
            out.push(MmprojTensor {
                name: gguf_name.into(),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: shape.len() == 1,
            });
        }
    }
    Ok(out)
}

fn reshape_spatial_conv_3d_to_4d(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 3 {
        let k = (shape[2] as f64).sqrt().round() as usize;
        if k * k == shape[2] {
            return vec![shape[0], shape[1], k, k];
        }
    }
    shape.to_vec()
}

fn compute_lfm_ffn_dim(cfg: &Value, default: u32) -> Result<u32, CeraError> {
    let base = narrowed("intermediate_size", cfg.get("intermediate_size"), default)?;
    if cfg.get("block_auto_adjust_ff_dim").and_then(Value::as_bool) == Some(true) {
        let mut dim = (2 * base as usize) / 3;
        if let Some(mult) = cfg.get("block_ffn_dim_multiplier").and_then(Value::as_f64) {
            if !mult.is_finite() || mult <= 0.0 {
                return Err(CeraError::Backend(format!(
                    "mmproj config `block_ffn_dim_multiplier` must be finite and positive, got {mult}"
                )));
            }
            let scaled = dim as f64 * mult;
            if !scaled.is_finite() || scaled < 1.0 || scaled > u32::MAX as f64 {
                return Err(CeraError::Backend(format!(
                    "mmproj config scaled ffn dim ({scaled}) does not fit in u32"
                )));
            }
            dim = scaled as usize;
        }
        let mult_of =
            narrowed("block_multiple_of", cfg.get("block_multiple_of"), 256)?.max(1) as usize;
        let rounded = dim
            .div_ceil(mult_of)
            .checked_mul(mult_of)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                CeraError::Backend(format!(
                    "mmproj config rounded ffn dim (dim={dim}, block_multiple_of={mult_of}) does not fit in u32"
                ))
            })?;
        Ok(rounded)
    } else {
        Ok(base)
    }
}

/// Apply GGUF metadata for an LFM3-Audio multimodal projector (`clip` architecture).
pub fn apply_lfm3_audio_mmproj_metadata(
    root_cfg: &Value,
    detok_cfg: Option<&Value>,
    has_detok: bool,
    writer: &mut GgufWriter,
    model_name: &str,
) -> Result<(), CeraError> {
    let enc_outer = root_cfg
        .get("encoder")
        .or_else(|| root_cfg.get("audio_encoder_config"))
        .or_else(|| root_cfg.get("conformer"))
        .or_else(|| root_cfg.get("audio_tower"))
        .unwrap_or(root_cfg);
    let enc_cfg = enc_outer
        .get("lfm")
        .or_else(|| enc_outer.get("text_config"))
        .unwrap_or(enc_outer);
    let lfm_cfg = root_cfg
        .get("lfm")
        .or_else(|| root_cfg.get("text_config"))
        .unwrap_or(root_cfg);
    let depth_cfg = root_cfg
        .get("depthformer")
        .or_else(|| root_cfg.get("depth_config"))
        .or_else(|| root_cfg.get("code_generator"))
        .unwrap_or(root_cfg);

    let has_gen_audio = has_detok
        || root_cfg.get("depthformer").is_some()
        || root_cfg.get("depth_config").is_some()
        || root_cfg.get("code_generator").is_some();

    writer.add_string("general.architecture", "clip");
    writer.add_string("general.name", model_name);
    writer.add_bool("clip.has_vision_encoder", false);
    writer.add_bool("clip.has_audio_encoder", true);
    if has_gen_audio {
        writer.add_bool("clip.has_gen_audio_encoder", true);
    }
    if has_detok {
        writer.add_bool("clip.has_lfm3_audio_detokenizer", true);
    }
    writer.add_string("clip.projector_type", "lfm3a");
    writer.add_bool("clip.use_gelu", false);
    writer.add_bool("clip.use_silu", true);

    let enc_hidden = narrowed("hidden_size", enc_cfg.get("hidden_size"), 512)?;
    let enc_ffn = compute_lfm_ffn_dim(enc_cfg, 2048)?;
    let enc_layers = narrowed("num_hidden_layers", enc_cfg.get("num_hidden_layers"), 16)?;
    let enc_heads = narrowed("num_attention_heads", enc_cfg.get("num_attention_heads"), 8)?;
    let enc_kv_heads = narrowed(
        "num_key_value_heads",
        enc_cfg.get("num_key_value_heads"),
        enc_heads,
    )?;
    let enc_head_dim = narrowed_opt("head_dim", enc_cfg.get("head_dim"))?
        .unwrap_or_else(|| enc_hidden / enc_heads.max(1));
    let enc_eps = f32_or(
        enc_cfg,
        &["norm_eps", "rms_norm_eps", "layer_norm_eps"],
        1e-5,
    );
    let enc_l_cache = narrowed("conv_L_cache", enc_cfg.get("conv_L_cache"), 7)?;
    let enc_window = narrowed(
        "sliding_window",
        enc_outer
            .get("sliding_window")
            .or_else(|| enc_cfg.get("sliding_window")),
        150,
    )?;
    let proj_dim = narrowed("hidden_size", lfm_cfg.get("hidden_size"), 2048)?;
    let n_mel = narrowed(
        "num_mel_bins",
        enc_outer
            .get("dim_in")
            .or_else(|| enc_outer.get("features_dim"))
            .or_else(|| enc_outer.get("num_mel_bins"))
            .or_else(|| enc_cfg.get("num_mel_bins")),
        128,
    )?;

    writer.add_u32("clip.audio.embedding_length", enc_hidden);
    writer.add_u32("clip.audio.feed_forward_length", enc_ffn);
    writer.add_u32("clip.audio.block_count", enc_layers);
    writer.add_u32("clip.audio.attention.head_count", enc_heads);
    writer.add_u32("clip.audio.attention.head_count_kv", enc_kv_heads);
    writer.add_u32("clip.audio.attention.key_length", enc_head_dim);
    writer.add_u32("clip.audio.attention.value_length", enc_head_dim);
    writer.add_f32("clip.audio.attention.layer_norm_epsilon", enc_eps);
    writer.add_f32("clip.audio.attention.layer_norm_rms_epsilon", enc_eps);
    writer.add_u32("clip.audio.projection_dim", proj_dim);
    writer.add_u32("clip.audio.num_mel_bins", n_mel);
    writer.add_u32("clip.audio.conv_l_cache", enc_l_cache);
    writer.add_u32("clip.audio.shortconv.l_cache", enc_l_cache);
    writer.add_u32("clip.audio.window_size", enc_window);
    writer.add_u32("clip.audio.rope.dimension_count", enc_head_dim);
    let enc_theta = enc_cfg
        .get("rope_parameters")
        .and_then(|p| p.get("full_attention").or(Some(p)))
        .and_then(|p| p.get("rope_theta").or_else(|| p.get("theta")))
        .or_else(|| enc_cfg.get("rope_theta"))
        .and_then(Value::as_f64)
        .unwrap_or(1_000_000.0) as f32;
    writer.add_f32("clip.audio.rope.freq_base", enc_theta);

    // Code generator / depthformer metadata (both `clip.audio_gen.code.*` for runtime loader
    // and `clip.audio.code_generator.*` for schema inspection).
    let depth_layers = narrowed(
        "layers",
        depth_cfg
            .get("layers")
            .or_else(|| depth_cfg.get("num_hidden_layers")),
        4,
    )?;
    let depth_dim = narrowed(
        "dim",
        depth_cfg
            .get("dim")
            .or_else(|| depth_cfg.get("hidden_size")),
        1024,
    )?;
    let depth_ffn = narrowed(
        "intermediate_size",
        depth_cfg.get("intermediate_size"),
        1792,
    )?;
    let depth_heads = narrowed(
        "num_heads",
        depth_cfg
            .get("num_heads")
            .or_else(|| depth_cfg.get("num_attention_heads")),
        8,
    )?;
    let depth_kv_heads = narrowed(
        "num_kv_heads",
        depth_cfg
            .get("num_kv_heads")
            .or_else(|| depth_cfg.get("num_key_value_heads")),
        2,
    )?;
    let depth_eps = f32_or(
        depth_cfg,
        &["norm_eps", "rms_norm_eps", "layer_norm_eps"],
        1e-5,
    );
    let depth_theta = depth_cfg
        .get("rope_parameters")
        .and_then(|p| p.get("full_attention").or(Some(p)))
        .and_then(|p| p.get("rope_theta").or_else(|| p.get("theta")))
        .or_else(|| depth_cfg.get("rope_theta"))
        .and_then(Value::as_f64)
        .unwrap_or(1_000_000.0) as f32;
    let codebooks = narrowed(
        "num_codebooks",
        root_cfg
            .get("codebooks")
            .or_else(|| root_cfg.get("num_codebooks")),
        12,
    )?;
    let audio_vocab = narrowed(
        "audio_vocab_size",
        root_cfg
            .get("audio_vocab_size")
            .or_else(|| root_cfg.get("codebook_size"))
            .or_else(|| root_cfg.get("vocab_size")),
        257,
    )?;
    if let Some(n_text) =
        narrowed_opt("interleaved_n_text", root_cfg.get("interleaved_n_text"))?.filter(|&v| v > 0)
    {
        writer.add_u32("interleaved_n_text", n_text);
        writer.add_u32("clip.audio.interleaved_n_text", n_text);
    }
    if let Some(n_audio) =
        narrowed_opt("interleaved_n_audio", root_cfg.get("interleaved_n_audio"))?.filter(|&v| v > 0)
    {
        writer.add_u32("interleaved_n_audio", n_audio);
        writer.add_u32("clip.audio.interleaved_n_audio", n_audio);
    }

    writer.add_u32("clip.audio_gen.code.block_count", depth_layers);
    writer.add_u32("clip.audio_gen.code.embedding_length", depth_dim);
    writer.add_u32("clip.audio_gen.code.feed_forward_length", depth_ffn);
    writer.add_u32("clip.audio_gen.code.attention.head_count", depth_heads);
    writer.add_u32(
        "clip.audio_gen.code.attention.head_count_kv",
        depth_kv_heads,
    );
    writer.add_f32(
        "clip.audio_gen.code.attention.layer_norm_epsilon",
        depth_eps,
    );
    writer.add_f32("clip.audio_gen.code.rope.freq_base", depth_theta);
    writer.add_u32("clip.audio_gen.code.num_codebooks", codebooks);
    writer.add_u32("clip.audio_gen.code.vocab_size", audio_vocab);

    writer.add_u32("clip.audio.code_generator.block_count", depth_layers);
    writer.add_u32("clip.audio.code_generator.embedding_length", depth_dim);
    writer.add_u32("clip.audio.code_generator.feed_forward_length", depth_ffn);
    writer.add_u32("clip.audio.code_generator.head_count", depth_heads);
    writer.add_u32("clip.audio.code_generator.head_count_kv", depth_kv_heads);
    writer.add_u32("clip.audio.code_generator.num_codebooks", codebooks);
    writer.add_u32("clip.audio.code_generator.vocab_size", audio_vocab);

    if let Some(raw_dcfg) = detok_cfg {
        let dcfg = raw_dcfg.get("model").unwrap_or(raw_dcfg);
        let dlfm_opt = raw_dcfg
            .get("decoder_lfm_config")
            .or_else(|| raw_dcfg.get("lfm"))
            .or_else(|| dcfg.get("decoder_lfm_config"))
            .or_else(|| dcfg.get("lfm"));
        let dlfm = dlfm_opt.unwrap_or(dcfg);
        let is_codec_cfg = dlfm_opt.is_none()
            && (raw_dcfg.get("fsq_levels").is_some()
                || dcfg.get("fsq_levels").is_some()
                || dcfg.get("encoder_dim").is_some()
                || dcfg.get("stft_n_fft").is_some());
        let d_hidden = match narrowed_opt("encoder_dim", dcfg.get("encoder_dim"))? {
            Some(v) => v,
            None => narrowed("hidden_size", dlfm.get("hidden_size"), 512)?,
        };
        let d_ffn = compute_lfm_ffn_dim(dlfm, if is_codec_cfg { 2304 } else { 2048 })?;
        let d_layers = narrowed("num_hidden_layers", dlfm.get("num_hidden_layers"), 8)?;
        let d_heads = narrowed(
            "num_attention_heads",
            dlfm.get("num_attention_heads"),
            if is_codec_cfg { 8 } else { 16 },
        )?;
        let d_kv_heads = narrowed(
            "num_key_value_heads",
            dlfm.get("num_key_value_heads"),
            if is_codec_cfg { 2 } else { 4 },
        )?;
        let d_head_dim = narrowed_opt("head_dim", dlfm.get("head_dim"))?
            .unwrap_or_else(|| d_hidden / d_heads.max(1));
        let d_eps = f32_or(dlfm, &["norm_eps", "rms_norm_eps"], 1e-5);
        let d_l_cache = narrowed("conv_L_cache", dlfm.get("conv_L_cache"), 7)?;
        let d_window = narrowed(
            "decoder_window",
            dcfg.get("decoder_window")
                .or_else(|| dlfm.get("sliding_window")),
            30,
        )?;
        let d_theta =
            dlfm.get("rope_parameters")
                .and_then(|p| {
                    p.get("sliding_attention")
                        .or_else(|| p.get("full_attention"))
                        .or(Some(p))
                })
                .and_then(|p| p.get("rope_theta").or_else(|| p.get("theta")))
                .or_else(|| dlfm.get("rope_theta"))
                .and_then(Value::as_f64)
                .unwrap_or(if is_codec_cfg { 1_000_000.0 } else { 10_000.0 }) as f32;
        let d_n_fft = narrowed(
            "stft_n_fft",
            dcfg.get("stft_n_fft").or_else(|| dcfg.get("n_fft")),
            1280,
        )?;
        let d_hop = narrowed(
            "stft_hop_length",
            dcfg.get("stft_hop_length")
                .or_else(|| dcfg.get("hop_length")),
            320,
        )?;

        writer.add_u32("clip.audio_gen.wav.embedding_length", d_hidden);
        writer.add_u32("clip.audio_gen.wav.feed_forward_length", d_ffn);
        writer.add_u32("clip.audio_gen.wav.block_count", d_layers);
        writer.add_u32("clip.audio_gen.wav.attention.head_count", d_heads);
        writer.add_u32("clip.audio_gen.wav.attention.head_count_kv", d_kv_heads);
        writer.add_f32("clip.audio_gen.wav.attention.layer_norm_epsilon", d_eps);
        writer.add_u32("clip.audio_gen.wav.window_size", d_window);
        writer.add_f32("clip.audio_gen.wav.rope.freq_base", d_theta);
        writer.add_u32("clip.audio_gen.wav.n_fft", d_n_fft);
        writer.add_u32("clip.audio_gen.wav.hop_length", d_hop);

        writer.add_u32("clip.audio.detokenizer.embedding_length", d_hidden);
        writer.add_u32("clip.audio.detokenizer.feed_forward_length", d_ffn);
        writer.add_u32("clip.audio.detokenizer.block_count", d_layers);
        writer.add_u32("clip.audio.detokenizer.attention.head_count", d_heads);
        if let Some(types) = dlfm.get("layer_types").and_then(Value::as_array) {
            let d_kv_i32 = i32::try_from(d_kv_heads).map_err(|_| {
                CeraError::Backend(format!(
                    "mmproj detokenizer `num_key_value_heads` ({d_kv_heads}) does not fit in i32"
                ))
            })?;
            let kv_arr: Vec<i32> = types
                .iter()
                .map(|t| {
                    if t.as_str() == Some("conv") {
                        0
                    } else {
                        d_kv_i32
                    }
                })
                .collect();
            writer.add_i32_array("clip.audio.detokenizer.attention.head_count_kv", kv_arr);
        }
        writer.add_u32("clip.audio.detokenizer.attention.key_length", d_head_dim);
        writer.add_u32("clip.audio.detokenizer.attention.value_length", d_head_dim);
        writer.add_f32(
            "clip.audio.detokenizer.attention.layer_norm_rms_epsilon",
            d_eps,
        );
        writer.add_u32("clip.audio.detokenizer.shortconv.l_cache", d_l_cache);
        writer.add_u32("clip.audio.detokenizer.sliding_window", d_window);
        writer.add_f32("clip.audio.detokenizer.rope.freq_base", d_theta);
        writer.add_u32("clip.audio.detokenizer.rope.dimension_count", d_head_dim);
        writer.add_u32("clip.audio.detokenizer.n_fft", d_n_fft);
        writer.add_u32("clip.audio.detokenizer.hop_length", d_hop);
        let rates_arr = parse_positive_detok_i32_array(raw_dcfg, "upsample_rates")?;
        let kernels_arr = parse_positive_detok_i32_array(raw_dcfg, "upsample_kernel_sizes")?;
        if let (Some(r), Some(k)) = (&rates_arr, &kernels_arr)
            && r.len() != k.len()
        {
            return Err(CeraError::Backend(format!(
                "mmproj detokenizer `upsample_rates` len ({}) != `upsample_kernel_sizes` len ({})",
                r.len(),
                k.len()
            )));
        }
        if let Some(arr) = rates_arr {
            writer.add_i32_array("clip.audio.detokenizer.upsample_rates", arr);
        }
        if let Some(arr) = kernels_arr {
            writer.add_i32_array("clip.audio.detokenizer.upsample_kernels", arr);
        }
    }
    Ok(())
}

/// `key` of a detokenizer config: the nested `model` object when it has it, else the top
/// level, the same precedence on the metadata and tensor paths.
fn detok_value<'a>(cfg: &'a Value, key: &str) -> Option<&'a Value> {
    cfg.get("model")
        .and_then(|m| m.get(key))
        .or_else(|| cfg.get(key))
}

/// Parse an optional detokenizer integer array (`upsample_rates` or `upsample_kernel_sizes`),
/// requiring every element to be a positive `i32` (`>= 1`).
fn parse_positive_detok_i32_array(cfg: &Value, key: &str) -> Result<Option<Vec<i32>>, CeraError> {
    let Some(val) = detok_value(cfg, key) else {
        return Ok(None);
    };
    if val.is_null() {
        return Ok(None);
    }
    let Some(arr) = val.as_array() else {
        return Err(CeraError::Backend(format!(
            "mmproj detokenizer `{key}` must be an array, got {val}"
        )));
    };
    let mut out = Vec::with_capacity(arr.len());
    for (idx, elem) in arr.iter().enumerate() {
        let Some(v) = elem.as_i64() else {
            return Err(CeraError::Backend(format!(
                "mmproj detokenizer `{key}[{idx}]` ({elem}) must be an integer"
            )));
        };
        let v_i32 = i32::try_from(v).ok().filter(|&x| x > 0).ok_or_else(|| {
            CeraError::Backend(format!(
                "mmproj detokenizer `{key}[{idx}]` ({v}) must be in 1..={}",
                i32::MAX
            ))
        })?;
        out.push(v_i32);
    }
    Ok(Some(out))
}

/// The detokenizer's `upsample_kernel_sizes` (causal kernels the folded convs are sliced to).
fn detok_upsample_kernels(detok_cfg: Option<&Value>) -> Result<Vec<usize>, CeraError> {
    let Some(cfg) = detok_cfg else {
        return Ok(Vec::new());
    };
    Ok(
        parse_positive_detok_i32_array(cfg, "upsample_kernel_sizes")?
            .unwrap_or_default()
            .into_iter()
            .map(|v| v as usize)
            .collect(),
    )
}

/// Most entries a config-defined FSQ table may have (65,536): the runtime's own table has 256.
const MAX_FSQ_ENTRIES: usize = 1 << 16;

/// The FSQ codebook for `levels` quantisation levels per dimension, as the runtime's built-in
/// table is: a level `c` of `L` maps to `2c / (L - 1) - 1`, symmetric around zero (a single
/// level maps to 0), and dimension 0 varies fastest. `levels` come from an untrusted
/// config, so the entry count is checked and bounded.
fn fsq_codebook_from_levels(levels: &[usize]) -> Result<(Vec<usize>, Vec<f32>), CeraError> {
    let bad =
        |why: String| CeraError::Backend(format!("detokenizer `fsq_levels` {levels:?}: {why}"));
    if levels.is_empty() || levels.contains(&0) {
        return Err(bad("every level count must be at least 1".into()));
    }
    let mut entries = 1usize;
    for &l in levels {
        entries = entries
            .checked_mul(l)
            .filter(|&n| n <= MAX_FSQ_ENTRIES)
            .ok_or_else(|| bad(format!("more than {MAX_FSQ_ENTRIES} entries")))?;
    }
    let dim = levels.len();
    let mut basis = vec![1usize; dim];
    for d in 1..dim {
        basis[d] = basis[d - 1] * levels[d - 1];
    }
    let mut table = Vec::with_capacity(entries * dim);
    for idx in 0..entries {
        for d in 0..dim {
            let l = levels[d];
            let coord = ((idx / basis[d]) % l) as f32;
            table.push(if l == 1 {
                0.0
            } else {
                2.0 * coord / (l - 1) as f32 - 1.0
            });
        }
    }
    Ok((vec![entries, dim], table))
}

/// Build the 4096x6 FSQ codebook (`6^4 = 4096` levels in
/// `[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]`, padded with 2 trailing zeros per row).
pub fn build_lfm3_fsq_codebook() -> (Vec<usize>, Vec<f32>) {
    const LEVELS: [f32; 6] = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0];
    let mut data = Vec::with_capacity(4096 * 6);
    for idx in 0..4096usize {
        let d0 = LEVELS[idx % 6];
        let d1 = LEVELS[(idx / 6) % 6];
        let d2 = LEVELS[(idx / 36) % 6];
        let d3 = LEVELS[(idx / 216) % 6];
        data.extend_from_slice(&[d0, d1, d2, d3, 0.0, 0.0]);
    }
    (vec![4096, 6], data)
}

/// Fold PyTorch WeightNorm parameters `original0` (`g`, shape `[out_c, 1, 1]`) and
/// `original1` (`v`, shape `[out_c, in_c, k_padded]`) into `w = g * v / ||v||_2` and
/// slice causal padding along the last axis to `[:, :, :kernel_size]`.
pub fn fold_weight_norm_and_slice_causal(
    g: &[f32],
    v: &[f32],
    v_shape: &[usize],
    kernel_size: Option<usize>,
) -> Result<(Vec<usize>, Vec<f32>), CeraError> {
    if v_shape.len() != 3 {
        return Err(CeraError::Backend(format!(
            "WeightNorm original1 must be 3D [out_c, in_c, k], got {v_shape:?}"
        )));
    }
    let (out_c, in_c, k_padded) = (v_shape[0], v_shape[1], v_shape[2]);
    let expected_v = out_c
        .checked_mul(in_c)
        .and_then(|p| p.checked_mul(k_padded))
        .ok_or_else(|| CeraError::Backend(format!("WeightNorm shape overflow: {v_shape:?}")))?;
    if g.len() != out_c || v.len() != expected_v {
        return Err(CeraError::Backend(format!(
            "WeightNorm shape mismatch: g len {} vs out_c {out_c}, v len {} vs {v_shape:?}",
            g.len(),
            v.len()
        )));
    }
    let k_keep = kernel_size.unwrap_or(k_padded).min(k_padded);
    let slice_len = in_c * k_padded;
    let mut out = Vec::with_capacity(out_c * in_c * k_keep);
    for (oc, &g_val) in g.iter().enumerate() {
        let ch_slice = &v[oc * slice_len..(oc + 1) * slice_len];
        let norm_sq: f32 = ch_slice.iter().map(|&x| x * x).sum();
        let scale = g_val / (norm_sq.sqrt().max(1e-12));
        for ic in 0..in_c {
            let row = &ch_slice[ic * k_padded..ic * k_padded + k_keep];
            out.extend(row.iter().map(|&x| x * scale));
        }
    }
    Ok((vec![out_c, in_c, k_keep], out))
}

/// Transform LFM3-Audio encoder, adapter, depthformer, and detokenizer tensors into GGUF `mmproj` tensors.
pub fn build_lfm3_audio_mmproj_tensors(
    root_cfg: &Value,
    detok_cfg: Option<&Value>,
    raw_tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
) -> Result<Vec<MmprojTensor>, CeraError> {
    let mut out = Vec::new();

    // Collect indexed embeddings for concatenation:
    // - `codebook_embedding.emb.{idx}.weight` or `audio_out_embeddings.{idx}.weight` -> `a.gen.code.out_embd.weight`
    // - `depth_embeddings.{idx}.weight` -> `a.gen.code.tok_embd.weight`
    // - `depth_embeddings.{idx}.embedding_norm.weight` -> `a.gen.code.in_norm.weight`
    // - `depth_embeddings.{idx}.linear.weight` -> `a.gen.code.out_proj.weight`
    let mut codebook_embs: BTreeMap<usize, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut depth_embs: BTreeMap<usize, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut depth_norms: BTreeMap<usize, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut depth_linears: BTreeMap<usize, (Vec<usize>, Vec<f32>)> = BTreeMap::new();

    // WeightNorm pairs in detokenizer: prefix -> (Option<g>, Option<(v_shape, v)>)
    type WeightNormPair = (Option<Vec<f32>>, Option<(Vec<usize>, Vec<f32>)>);
    let mut weight_norms: BTreeMap<String, WeightNormPair> = BTreeMap::new();

    for (raw_name, (shape, data)) in raw_tensors {
        validate_tensor_buffer(raw_name, shape, data)?;
        let name = raw_name.strip_prefix("model.").unwrap_or(raw_name.as_str());

        if let Some(rest) = name
            .strip_prefix("codebook_embedding.emb.")
            .or_else(|| name.strip_prefix("audio_out_embeddings."))
            && let Some(idx_str) = rest.strip_suffix(".weight")
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            codebook_embs.insert(idx, (shape.clone(), data.clone()));
            continue;
        }
        if let Some(rest) = name
            .strip_prefix("depth_embeddings.")
            .or_else(|| name.strip_prefix("depth_decoder.depth_embeddings."))
            && let Some((idx_str, tail)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            match tail {
                "weight" | "embedding.weight" => {
                    depth_embs.insert(idx, (shape.clone(), data.clone()));
                    continue;
                }
                "embedding_norm.weight" => {
                    depth_norms.insert(idx, (shape.clone(), data.clone()));
                    continue;
                }
                "linear.weight" | "to_logits.weight" => {
                    depth_linears.insert(idx, (shape.clone(), data.clone()));
                    continue;
                }
                _ => {}
            }
        }
        if let Some(prefix) = name.strip_suffix(".parametrizations.weight.original0") {
            weight_norms.entry(prefix.to_string()).or_default().0 = Some(data.clone());
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".parametrizations.weight.original1") {
            weight_norms.entry(prefix.to_string()).or_default().1 =
                Some((shape.clone(), data.clone()));
            continue;
        }

        // Encoder 5-layer depthwise-separable downsample stem (`encoder.downsample.*` or `audio_encoder.downsample.*` -> `a.conv1d.{0,2,3,5,6}` / `a.pre_encode.out`)
        if let Some(rest) = name
            .strip_prefix("encoder.downsample.")
            .or_else(|| name.strip_prefix("audio_encoder.downsample."))
        {
            let mapped = match rest {
                "conv_in.weight" => Some(("a.conv1d.0.weight", shape.clone(), false)),
                "conv_in.bias" => Some(("a.conv1d.0.bias", shape.clone(), true)),
                "conv_mid.weight" => Some(("a.conv1d.2.weight", shape.clone(), false)),
                "conv_mid.bias" => Some(("a.conv1d.2.bias", shape.clone(), true)),
                "lin_mid.weight" => Some(("a.conv1d.3.weight", shape.clone(), false)),
                "lin_mid.bias" => Some(("a.conv1d.3.bias", shape.clone(), true)),
                "conv_end.weight" => Some(("a.conv1d.5.weight", shape.clone(), false)),
                "conv_end.bias" => Some(("a.conv1d.5.bias", shape.clone(), true)),
                "lin_end.weight" => Some(("a.conv1d.6.weight", shape.clone(), false)),
                "lin_end.bias" => Some(("a.conv1d.6.bias", shape.clone(), true)),
                "lin_out.weight" => {
                    let s = if shape.len() == 3 && shape[2] == 1 {
                        vec![shape[0], shape[1]]
                    } else {
                        shape.clone()
                    };
                    Some(("a.pre_encode.out.weight", s, false))
                }
                "lin_out.bias" => Some(("a.pre_encode.out.bias", shape.clone(), true)),
                _ => None,
            };
            if let Some((gguf_name, target_shape, force_f32)) = mapped {
                out.push(MmprojTensor {
                    name: gguf_name.into(),
                    shape: target_shape,
                    data: data.clone(),
                    force_f32,
                });
                continue;
            }
        }

        // Encoder conv stem (`encoder.pre_encode.conv.{idx}.{weight,bias}` or `conformer.pre_encode.conv.{idx}.{weight,bias}`)
        if let Some(rest) = name
            .strip_prefix("encoder.pre_encode.conv.")
            .or_else(|| name.strip_prefix("conformer.pre_encode.conv."))
            && let Some((idx_str, kind)) = rest.split_once('.')
            && idx_str.parse::<u32>().is_ok()
            && (kind == "weight" || kind == "bias")
        {
            out.push(MmprojTensor {
                name: format!("a.conv1d.{idx_str}.{kind}"),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: kind == "bias",
            });
            continue;
        }

        // Encoder blocks (`encoder.lfm.lfm.layers.{i}.*`, `encoder.lfm.layers.{i}.*`, or `audio_encoder.lfm.layers.{i}.*`)
        if let Some(rest) = name
            .strip_prefix("encoder.lfm.lfm.layers.")
            .or_else(|| name.strip_prefix("encoder.lfm.layers."))
            .or_else(|| name.strip_prefix("audio_encoder.lfm.layers."))
            && let Some((layer_str, sub)) = rest.split_once('.')
            && let Ok(bid) = layer_str.parse::<usize>()
        {
            if let Some(suffix) = map_lfm3_block_sub(sub) {
                out.push(MmprojTensor {
                    name: format!("a.blk.{bid}.{suffix}"),
                    shape: shape.clone(),
                    data: data.clone(),
                    force_f32: shape.len() == 1,
                });
            }
            continue;
        }
        // Depthformer blocks (`depthformer.model.layers.{i}.*`, `depth_decoder.model.layers.{i}.*`, or `code_generator.layers.{i}.*`)
        if let Some(rest) = name
            .strip_prefix("depthformer.model.layers.")
            .or_else(|| name.strip_prefix("depth_decoder.model.layers."))
            .or_else(|| name.strip_prefix("code_generator.layers."))
            && let Some((layer_str, sub)) = rest.split_once('.')
            && let Ok(bid) = layer_str.parse::<usize>()
        {
            if let Some(suffix) = map_lfm3_block_sub(sub) {
                out.push(MmprojTensor {
                    name: format!("a.gen.code.blk.{bid}.{suffix}"),
                    shape: shape.clone(),
                    data: data.clone(),
                    force_f32: shape.len() == 1,
                });
            }
            continue;
        }
        // Detokenizer LFM blocks (`detokenizer.decoder.lfm.layers.{i}.*` or `detokenizer.lfm.layers.{i}.*`)
        if let Some(rest) = name
            .strip_prefix("detokenizer.decoder.lfm.layers.")
            .or_else(|| name.strip_prefix("detokenizer.lfm.layers."))
            && let Some((layer_str, sub)) = rest.split_once('.')
            && let Ok(bid) = layer_str.parse::<usize>()
        {
            if let Some(suffix) = map_lfm3_block_sub(sub) {
                out.push(MmprojTensor {
                    name: format!("a.detok.blk.{bid}.{suffix}"),
                    shape: shape.clone(),
                    data: data.clone(),
                    force_f32: shape.len() == 1,
                });
            }
            continue;
        }
        // Canonical HF audio MLP projector (`audio_projector.{layer_norm,linear_1,linear_2}.{weight,bias}`)
        if let Some(rest) = name.strip_prefix("audio_projector.") {
            let mapped = match rest {
                "layer_norm.weight" | "norm.weight" => Some("mm.a.mlp.0.weight"),
                "layer_norm.bias" | "norm.bias" => Some("mm.a.mlp.0.bias"),
                "linear_1.weight" | "fc1.weight" => Some("mm.a.mlp.1.weight"),
                "linear_1.bias" | "fc1.bias" => Some("mm.a.mlp.1.bias"),
                "linear_2.weight" | "fc2.weight" => Some("mm.a.mlp.3.weight"),
                "linear_2.bias" | "fc2.bias" => Some("mm.a.mlp.3.bias"),
                _ => None,
            };
            if let Some(gguf_name) = mapped {
                out.push(MmprojTensor {
                    name: gguf_name.into(),
                    shape: shape.clone(),
                    data: data.clone(),
                    force_f32: shape.len() == 1,
                });
                continue;
            }
        }
        // Audio MLP projector (`mlp.model.{idx}.{weight,bias}` or `audio_adapter.mlp.{idx}.{weight,bias}`)
        if let Some(rest) = name
            .strip_prefix("mlp.model.")
            .or_else(|| name.strip_prefix("audio_adapter.mlp."))
        {
            // In 3-entry sequential modules (`[LayerNorm(0), Linear(1), GELU_fn, Linear(2)]`),
            // map index 2 to `mm.a.mlp.3.{weight,bias}` so `load_lfm3a_encoder` finds `mm.a.mlp.3.*`.
            let mapped_rest = match rest {
                "2.weight"
                    if !raw_tensors.contains_key("mlp.model.3.weight")
                        && !raw_tensors.contains_key("model.mlp.model.3.weight")
                        && !raw_tensors.contains_key("audio_adapter.mlp.3.weight") =>
                {
                    "3.weight"
                }
                "2.bias"
                    if !raw_tensors.contains_key("mlp.model.3.bias")
                        && !raw_tensors.contains_key("model.mlp.model.3.bias")
                        && !raw_tensors.contains_key("audio_adapter.mlp.3.bias") =>
                {
                    "3.bias"
                }
                other => other,
            };
            out.push(MmprojTensor {
                name: format!("mm.a.mlp.{mapped_rest}"),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: shape.len() == 1,
            });
            continue;
        }
        // Detokenizer upsample / conv biases
        if let Some(rest) = name.strip_prefix("detokenizer.ups.")
            && let Some((idx_str, tail)) = rest.split_once('.')
            && tail == "0.bias"
        {
            out.push(MmprojTensor {
                name: format!("detok.ups.{idx_str}.bias"),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: true,
            });
            continue;
        }
        if let Some(rest) = name.strip_prefix("detokenizer.convs.")
            && let Some((idx_str, tail)) = rest.split_once('.')
            && tail == "1.bias"
        {
            out.push(MmprojTensor {
                name: format!("detok.convs.{idx_str}.bias"),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: true,
            });
            continue;
        }

        let direct = match name {
            "encoder.pre_encode.out.weight" | "conformer.pre_encode.out.weight" => {
                Some("a.pre_encode.out.weight")
            }
            "encoder.pre_encode.out.bias" | "conformer.pre_encode.out.bias" => {
                Some("a.pre_encode.out.bias")
            }
            "encoder.embedding_norm.weight"
            | "encoder.lfm.embedding_norm.weight"
            | "encoder.lfm.lfm.embedding_norm.weight"
            | "audio_encoder.lfm.embedding_norm.weight"
            | "conformer.norm.weight" => Some("a.post_ln.weight"),
            "depth_linear.weight"
            | "depth_decoder.depth_linear.weight"
            | "code_generator.in_proj.weight" => Some("a.gen.code.in_proj.weight"),
            "depth_linear.bias"
            | "depth_decoder.depth_linear.bias"
            | "code_generator.in_proj.bias" => Some("a.gen.code.in_proj.bias"),
            "codebook_embedding.weight" => Some("a.gen.code.out_embd.weight"),
            "mask_embedding" | "depth_decoder.mask_embedding" => Some("a.gen.code.mask_embd"),
            "codebook_embedding.embedding_norm.weight"
            | "depthformer.model.embedding_norm.weight"
            | "depth_decoder.model.embedding_norm.weight"
            | "code_generator.output_norm.weight" => Some("a.gen.code.output_norm.weight"),
            "detokenizer.decoder.lfm.embedding_norm.weight"
            | "detokenizer.lfm.embedding_norm.weight"
            | "detokenizer.output_norm.weight"
            | "detokenizer.tfm_output_norm.weight" => Some("a.detok.output_norm.weight"),
            "detokenizer.from_quantizer.weight"
            | "detokenizer.tfm_in_proj.weight"
            | "detokenizer.quant.out_proj.weight" => Some("a.detok.tfm_in_proj.weight"),
            "detokenizer.from_quantizer.bias"
            | "detokenizer.tfm_in_proj.bias"
            | "detokenizer.quant.out_proj.bias" => Some("a.detok.tfm_in_proj.bias"),
            "detokenizer.quantizer.project_out.weight"
            | "detokenizer.quantizer.fsq.project_out.weight"
            | "detokenizer.quant_first_out.weight"
            | "detokenizer.quant.first.out_proj.weight" => Some("a.detok.quant_first_out.weight"),
            "detokenizer.quantizer.project_out.bias"
            | "detokenizer.quantizer.fsq.project_out.bias"
            | "detokenizer.quant_first_out.bias"
            | "detokenizer.quant.first.out_proj.bias" => Some("a.detok.quant_first_out.bias"),
            "detokenizer.quant_rest_out.weight" | "detokenizer.quant.rest.out_proj.weight" => {
                Some("a.detok.quant_rest_out.weight")
            }
            "detokenizer.quant_rest_out.bias" | "detokenizer.quant.rest.out_proj.bias" => {
                Some("a.detok.quant_rest_out.bias")
            }
            "detokenizer.decoder.lin.weight"
            | "detokenizer.tfm_out_proj.weight"
            | "detokenizer.istft_head.weight" => Some("a.detok.tfm_out_proj.weight"),
            "detokenizer.decoder.lin.bias"
            | "detokenizer.tfm_out_proj.bias"
            | "detokenizer.istft_head.bias" => Some("a.detok.tfm_out_proj.bias"),
            "detokenizer.lin_in.bias" => Some("detok.lin_in.bias"),
            "detokenizer.lin_out.bias" => Some("a.detok.tfm_out_proj.bias"),
            _ => None,
        };
        if let Some(gguf_name) = direct {
            out.push(MmprojTensor {
                name: gguf_name.into(),
                shape: shape.clone(),
                data: data.clone(),
                force_f32: shape.len() == 1,
            });
        }
    }

    // Concatenate indexed embeddings along axis 0
    for (gguf_name, map) in [
        ("a.gen.code.out_embd.weight", codebook_embs),
        ("a.gen.code.tok_embd.weight", depth_embs),
        ("a.gen.code.in_norm.weight", depth_norms),
        ("a.gen.code.out_proj.weight", depth_linears),
    ] {
        if map.is_empty() {
            continue;
        }
        let (cat_shape, cat_data) = concat_indexed_tensors(&map, gguf_name)?;
        let is_norm = gguf_name.ends_with("in_norm.weight");
        out.push(MmprojTensor {
            name: gguf_name.into(),
            shape: cat_shape,
            data: cat_data,
            force_f32: is_norm,
        });
    }

    // Fold WeightNorm pairs and slice causal kernels for detokenizer convs
    let upsample_kernels = detok_upsample_kernels(detok_cfg)?;

    let has_detok_weights = !weight_norms.is_empty()
        || out
            .iter()
            .any(|t| t.name.starts_with("a.detok.") || t.name.starts_with("detok."));
    for (prefix, (g_opt, v_opt)) in weight_norms {
        let (Some(g), Some((v_shape, v))) = (g_opt, v_opt) else {
            return Err(CeraError::Backend(format!(
                "incomplete WeightNorm pair for `{prefix}`"
            )));
        };
        let (gguf_name, kernel_size) = if prefix == "detokenizer.lin_in" {
            ("detok.lin_in.weight".to_string(), Some(3usize))
        } else if prefix == "detokenizer.lin_out" {
            ("detok.lin_out.weight".to_string(), Some(1usize))
        } else if let Some(rest) = prefix.strip_prefix("detokenizer.ups.")
            && let Some(idx_str) = rest.strip_suffix(".0")
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            let k = upsample_kernels.get(idx).copied();
            (format!("detok.ups.{idx}.weight"), k)
        } else if let Some(rest) = prefix.strip_prefix("detokenizer.convs.")
            && let Some(idx_str) = rest.strip_suffix(".1")
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            (format!("detok.convs.{idx}.weight"), Some(3usize))
        } else {
            (format!("{prefix}.weight"), None)
        };
        let (folded_shape, folded_data) =
            fold_weight_norm_and_slice_causal(&g, &v, &v_shape, kernel_size)?;
        if gguf_name == "detok.lin_out.weight"
            && folded_shape.len() == 3
            && folded_shape[2] == 1
            && !out.iter().any(|t| t.name == "a.detok.tfm_out_proj.weight")
        {
            out.push(MmprojTensor {
                name: "a.detok.tfm_out_proj.weight".into(),
                shape: vec![folded_shape[0], folded_shape[1]],
                data: folded_data.clone(),
                force_f32: false,
            });
        }
        out.push(MmprojTensor {
            name: gguf_name,
            shape: folded_shape,
            data: folded_data,
            force_f32: false,
        });
    }

    if has_detok_weights {
        let (cb_shape, cb_data) = build_lfm3_fsq_codebook();
        out.push(MmprojTensor {
            name: "detok.fsq.codebook".into(),
            shape: cb_shape,
            data: cb_data,
            force_f32: true,
        });
        if !out.iter().any(|t| t.name == "a.detok.fsq_codebook") {
            let levels_arr = detok_cfg
                .and_then(|c| detok_value(c, "fsq_levels"))
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty());
            let (fsq_shape, fsq_data) = if let Some(levels_arr) = levels_arr {
                let levels = levels_arr
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(|| {
                                CeraError::Backend(format!(
                                    "detokenizer `fsq_levels` entry {v} is not a level count"
                                ))
                            })
                    })
                    .collect::<Result<Vec<usize>, CeraError>>()?;
                fsq_codebook_from_levels(&levels)?
            } else {
                (
                    vec![256, 4],
                    crate::model::audio_decoder::build_default_fsq_codebook(),
                )
            };
            out.push(MmprojTensor {
                name: "a.detok.fsq_codebook".into(),
                shape: fsq_shape,
                data: fsq_data,
                force_f32: true,
            });
        }
    }

    let _ = root_cfg;
    Ok(out)
}

fn map_lfm3_block_sub(sub: &str) -> Option<&'static str> {
    match sub {
        "operator_norm.weight" | "attn_norm.weight" | "ln1.weight" => Some("ln1.weight"),
        "post_operator_norm.weight" | "post_attention_norm.weight" | "attn_post_norm.weight" => {
            Some("attn_post_norm.weight")
        }
        "ffn_norm.weight" | "ln2.weight" => Some("ln2.weight"),
        "post_ffn_norm.weight" | "post_ffw_norm.weight" | "ffn_post_norm.weight" => {
            Some("ffn_post_norm.weight")
        }
        "self_attn.q_proj.weight" | "attn_q.weight" => Some("attn_q.weight"),
        "self_attn.k_proj.weight" | "attn_k.weight" => Some("attn_k.weight"),
        "self_attn.v_proj.weight" | "attn_v.weight" => Some("attn_v.weight"),
        "self_attn.o_proj.weight" | "self_attn.out_proj.weight" | "attn_out.weight" => {
            Some("attn_out.weight")
        }
        "self_attn.q_norm.weight" | "self_attn.q_layernorm.weight" | "attn_q_norm.weight" => {
            Some("attn_q_norm.weight")
        }
        "self_attn.k_norm.weight" | "self_attn.k_layernorm.weight" | "attn_k_norm.weight" => {
            Some("attn_k_norm.weight")
        }
        "self_attn.gate_proj.weight" | "attn_gate.weight" => Some("attn_gate.weight"),
        "feed_forward.w1.weight" | "ffn_gate.weight" => Some("ffn_gate.weight"),
        "feed_forward.w2.weight" | "ffn_down.weight" => Some("ffn_down.weight"),
        "feed_forward.w3.weight" | "ffn_up.weight" => Some("ffn_up.weight"),
        "conv.conv.weight" | "conv.conv" | "shortconv.conv.weight" | "conv_dw.weight" => {
            Some("conv_dw.weight")
        }
        "conv.in_proj.weight" | "shortconv.in_proj.weight" | "conv_pw1.weight" => {
            Some("conv_pw1.weight")
        }
        "conv.out_proj.weight" | "shortconv.out_proj.weight" | "conv_pw2.weight" => {
            Some("conv_pw2.weight")
        }
        _ => None,
    }
}

fn concat_indexed_tensors(
    map: &BTreeMap<usize, (Vec<usize>, Vec<f32>)>,
    gguf_name: &str,
) -> Result<(Vec<usize>, Vec<f32>), CeraError> {
    let (_, (first_shape, _)) = map
        .iter()
        .next()
        .ok_or_else(|| CeraError::Backend(format!("`{gguf_name}` has no slices to concatenate")))?;
    if first_shape.is_empty() {
        return Err(CeraError::Backend(format!(
            "`{gguf_name}` cannot concatenate 0D scalar tensors"
        )));
    }
    let tail_dims = &first_shape[1..];
    let mut total_dim0 = 0usize;
    let mut combined = Vec::new();
    for (expected_idx, (&idx, (shape, data))) in map.iter().enumerate() {
        if idx != expected_idx {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` missing index {expected_idx} (found {idx})"
            )));
        }
        if shape.len() != first_shape.len() || &shape[1..] != tail_dims {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` slice {idx} shape {shape:?} disagrees with {first_shape:?}"
            )));
        }
        total_dim0 = total_dim0
            .checked_add(shape[0])
            .ok_or_else(|| CeraError::Backend(format!("`{gguf_name}` dim0 overflow")))?;
        combined.extend_from_slice(data);
    }
    let mut out_shape = Vec::with_capacity(first_shape.len().max(2));
    if first_shape.len() == 1 {
        out_shape.push(map.len());
        out_shape.push(first_shape[0]);
    } else {
        out_shape.push(total_dim0);
        out_shape.extend_from_slice(tail_dims);
    }
    Ok((out_shape, combined))
}

/// A config number as `u32`: absent (or not a number) gives `None`, and a number that does not
/// fit is an error naming the key, not a wrapped width written into the GGUF.
fn narrowed_opt(key: &str, v: Option<&Value>) -> Result<Option<u32>, CeraError> {
    if let Some(i) = v.and_then(Value::as_i64)
        && i < 0
    {
        return Err(CeraError::Backend(format!(
            "mmproj config `{key}` ({i}) does not fit in u32"
        )));
    }
    match v.and_then(Value::as_u64) {
        None => Ok(None),
        Some(n) => u32::try_from(n).map(Some).map_err(|_| {
            CeraError::Backend(format!("mmproj config `{key}` ({n}) does not fit in u32"))
        }),
    }
}

/// [`narrowed_opt`] with a default for an absent key.
fn narrowed(key: &str, v: Option<&Value>, default: u32) -> Result<u32, CeraError> {
    Ok(narrowed_opt(key, v)?.unwrap_or(default))
}

fn f32_or(cfg: &Value, keys: &[&str], default: f32) -> f32 {
    keys.iter()
        .find_map(|k| cfg.get(*k).and_then(Value::as_f64))
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn f32_vec_or(cfg: &Value, key: &str, default: &[f32]) -> Vec<f32> {
    cfg.get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_f64)
                .map(|v| v as f32)
                .collect::<Vec<f32>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

/// The image preprocessing config of a checkpoint directory: `processor_config.json` when it
/// carries image fields, else `preprocessor_config.json`. A processor config that only names
/// its processor class (no image fields) must not hide the preprocessor config, or the image
/// mean, std and tile metadata silently fall back to defaults.
fn read_preprocessor_config(dir: &Path) -> Result<Option<Value>, CeraError> {
    const IMAGE_KEYS: [&str; 13] = [
        "image_processor",
        "image_mean",
        "image_std",
        "min_pixels",
        "max_pixels",
        "size",
        "downsample_factor",
        "merge_size",
        "tile_size",
        "max_tiles",
        "min_tiles",
        "min_image_tokens",
        "max_image_tokens",
    ];
    let processor = read_optional_json(&dir.join("processor_config.json"))?;
    if processor
        .as_ref()
        .is_some_and(|p| IMAGE_KEYS.iter().any(|k| p.get(k).is_some()))
    {
        return Ok(processor);
    }
    Ok(read_optional_json(&dir.join("preprocessor_config.json"))?.or(processor))
}

fn read_optional_json(path: &Path) -> Result<Option<Value>, CeraError> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(path)
        .map_err(|e| CeraError::Backend(format!("failed to read `{}`: {e}", path.display())))?;
    let val = serde_json::from_slice(&bytes)
        .map_err(|e| CeraError::Backend(format!("failed to parse `{}`: {e}", path.display())))?;
    Ok(Some(val))
}

type TensorMap = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn load_checkpoint_tensors(dir: &Path, input_path: &Path) -> Result<TensorMap, CeraError> {
    let mut files: Vec<PathBuf> = Vec::new();
    if input_path.is_file()
        && input_path.extension().and_then(|e| e.to_str()) == Some("safetensors")
    {
        files.push(input_path.to_path_buf());
    } else {
        for entry in
            fs::read_dir(dir).map_err(|e| CeraError::Backend(format!("failed to read dir: {e}")))?
        {
            let entry =
                entry.map_err(|e| CeraError::Backend(format!("failed to read entry: {e}")))?;
            let p = entry.path();
            let name_lower = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if p.is_file()
                && p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                && name_lower != "detokenizer.safetensors"
            {
                files.push(p);
            }
        }
        files.sort();
    }
    let mut map = BTreeMap::new();
    for f in files {
        for (k, v) in load_safetensors_file(&f, &is_projector_tensor)? {
            map.insert(k, v);
        }
    }
    Ok(map)
}

/// Whether a checkpoint tensor can belong to the projector. The language model's tensors
/// (the backbone layers, embeddings, head) are never read by the builders, so they are
/// neither decoded nor held in memory, and an exotic dtype in them cannot fail the conversion.
fn is_projector_tensor(name: &str) -> bool {
    const BACKBONE_PREFIXES: [&str; 8] = [
        "model.layers.",
        "model.embed_tokens.",
        "model.embedding_norm.",
        "lm_head.",
        "lfm.",
        "model.lfm.",
        "model.language_model.",
        "language_model.",
    ];
    !BACKBONE_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Read the tensors of one SafeTensors file that `keep` accepts, decoded to `f32`.
fn load_safetensors_file(path: &Path, keep: &dyn Fn(&str) -> bool) -> Result<TensorMap, CeraError> {
    let mut file = File::open(path)
        .map_err(|e| CeraError::Backend(format!("failed to open `{}`: {e}", path.display())))?;
    let header = SafeTensorsHeader::parse_from_reader(&mut file)?;
    let file_len = file
        .metadata()
        .map_err(|e| CeraError::Backend(format!("failed to stat `{}`: {e}", path.display())))?
        .len();
    let mut out = BTreeMap::new();
    for (name, info) in header.tensors.iter().filter(|(name, _)| keep(name)) {
        let start = header
            .header_size_bytes
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| CeraError::Backend(format!("offset overflow for `{name}`")))?;
        let end = header
            .header_size_bytes
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| CeraError::Backend(format!("offset overflow for `{name}`")))?;
        if end < start {
            return Err(CeraError::Backend(format!(
                "invalid data_offsets ({}, {}) for `{name}`",
                info.data_offsets.0, info.data_offsets.1
            )));
        }
        // the span comes from the file's own header: bound it before sizing a buffer from it
        if end as u64 > file_len {
            return Err(CeraError::Backend(format!(
                "`{name}` data range {start}..{end} extends past the end of its {file_len}-byte file"
            )));
        }
        file.seek(SeekFrom::Start(start as u64))
            .map_err(|e| CeraError::Backend(format!("seek failed: {e}")))?;
        let mut raw = vec![0u8; end - start];
        file.read_exact(&mut raw)
            .map_err(|e| CeraError::Backend(format!("read failed: {e}")))?;
        let mut values = Vec::new();
        decode_safetensor_to_f32_into(&raw, &info.dtype, &mut values).map_err(|e| {
            CeraError::Backend(format!("tensor `{name}` in `{}`: {e}", path.display()))
        })?;
        validate_tensor_buffer(name, &info.shape, &values)?;
        out.insert(name.clone(), (info.shape.clone(), values));
    }
    Ok(out)
}

struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn write_mmproj_gguf(
    output_gguf_path: &Path,
    writer: &mut GgufWriter,
    tensors: &[MmprojTensor],
    quant: TargetQuant,
) -> Result<(), CeraError> {
    // An empty set means the checkpoint held none of the projector's tensors; two tensors under
    // one GGUF name (checkpoint aliases mapping to the same slot) would write a corrupt file.
    if tensors.is_empty() {
        return Err(CeraError::Backend(
            "no projector tensors were found in the checkpoint (check the model type and the \
             tensor names)"
                .into(),
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for t in tensors {
        if !seen.insert(t.name.as_str()) {
            return Err(CeraError::Backend(format!(
                "two checkpoint tensors map to the projector tensor `{}`",
                t.name
            )));
        }
    }
    let mut types = Vec::with_capacity(tensors.len());
    for t in tensors {
        validate_tensor_buffer(&t.name, &t.shape, &t.data)?;
        let n_elems = t.data.len();
        let ggml_type = if t.force_f32 || t.shape.len() <= 1 {
            GGML_TYPE_F32
        } else {
            match quant {
                TargetQuant::F32 => GGML_TYPE_F32,
                TargetQuant::F16 => GGML_TYPE_F16,
                _ => quant.select_ggml_type(
                    &t.name,
                    t.shape.len(),
                    n_elems,
                    t.shape.last().copied().unwrap_or(1),
                ),
            }
        };
        let bytes = TargetQuant::compute_tensor_bytes(ggml_type, n_elems);
        writer.add_tensor(&t.name, t.gguf_dims(), ggml_type, bytes);
        types.push(ggml_type);
    }

    use std::sync::atomic::{AtomicU64, Ordering};
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let temp_path =
        output_gguf_path.with_extension(format!("gguf.tmp.{}.{}", std::process::id(), seq));
    let mut guard = TempFileGuard::new(temp_path.clone());

    let out_file = File::create(&temp_path).map_err(|e| {
        CeraError::Backend(format!("failed to create `{}`: {e}", temp_path.display()))
    })?;
    let mut buf_writer = BufWriter::new(out_file);
    writer.write_header_and_tensor_info(&mut buf_writer)?;

    let mut quant_buf = Vec::new();
    for (t, &ggml_type) in tensors.iter().zip(&types) {
        let target_size = TargetQuant::compute_tensor_bytes(ggml_type, t.data.len());
        quant_buf.resize(target_size, 0);
        quantize_tensor_data_with_strategy(
            &t.data,
            ggml_type,
            QuantStrategy::Auto,
            &mut quant_buf,
        )?;
        writer.write_tensor_data(&mut buf_writer, &quant_buf)?;
    }
    buf_writer
        .flush()
        .map_err(|e| CeraError::Backend(format!("failed to flush GGUF output: {e}")))?;
    buf_writer
        .get_ref()
        .sync_data()
        .map_err(|e| CeraError::Backend(format!("failed to sync GGUF output: {e}")))?;
    drop(buf_writer);

    fs::rename(&temp_path, output_gguf_path).map_err(|e| {
        CeraError::Backend(format!(
            "failed to rename `{}` to `{}`: {e}",
            temp_path.display(),
            output_gguf_path.display()
        ))
    })?;
    guard.disarm();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// File-declared counts size allocations (`Vec::with_capacity(n_layer)`, the sliding-window
    /// KV ring): the detokenizer loader bounds them like the depthformer loader bounds its own.
    #[test]
    fn the_detokenizer_loader_bounds_its_declared_counts() {
        DetokenizerWeights::from_gguf(&audio_gguf_with(|_| {})).unwrap();
        for (key, value) in [
            ("clip.audio_gen.wav.block_count", 1u32 << 30),
            ("clip.audio_gen.wav.window_size", 1 << 30),
        ] {
            let gguf = audio_gguf_with(|w| w.add_u32(key, value));
            let err = DetokenizerWeights::from_gguf(&gguf)
                .err()
                .unwrap_or_else(|| panic!("{key}={value} was accepted"));
            assert!(err.to_string().contains("must be in"), "{key}: {err}");
        }
    }

    /// A width that does not fit `u32` used to wrap (`as u32`) into a plausible small number in
    /// the written GGUF; it is refused with the key named.
    #[test]
    fn an_audio_config_number_that_does_not_fit_u32_is_refused() {
        let too_big = serde_json::json!(u64::from(u32::MAX) + 7);
        for (path, key) in [
            (vec!["encoder", "sliding_window"], "sliding_window"),
            (vec!["encoder", "dim_in"], "num_mel_bins"),
            (vec!["codebooks"], "num_codebooks"),
            (vec!["audio_vocab_size"], "audio_vocab_size"),
            (vec!["interleaved_n_text"], "interleaved_n_text"),
        ] {
            let (mut cfg, detok, _) = audio_roundtrip_fixture();
            let mut at = &mut cfg;
            for step in &path[..path.len() - 1] {
                at = at
                    .as_object_mut()
                    .unwrap()
                    .entry(*step)
                    .or_insert_with(|| serde_json::json!({}));
            }
            at.as_object_mut()
                .unwrap()
                .insert(path[path.len() - 1].to_string(), too_big.clone());
            let mut writer = GgufWriter::new();
            let err = apply_lfm3_audio_mmproj_metadata(&cfg, Some(&detok), true, &mut writer, "x")
                .err()
                .unwrap_or_else(|| panic!("{path:?} = {too_big} was accepted"));
            assert!(
                err.to_string().contains(key) && err.to_string().contains("does not fit"),
                "{path:?}: {err}"
            );
        }
    }

    /// The converted GGUF of the LFM3-Audio fixture, with `edit` applied to its metadata.
    fn audio_gguf_with(edit: impl FnOnce(&mut GgufWriter)) -> Arc<GgufFile> {
        let (cfg, detok, raw) = audio_roundtrip_fixture();
        let mut writer = GgufWriter::new();
        apply_lfm3_audio_mmproj_metadata(&cfg, Some(&detok), true, &mut writer, "audio-edit")
            .unwrap();
        edit(&mut writer);
        let tensors = build_lfm3_audio_mmproj_tensors(&cfg, Some(&detok), &raw).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("mmproj.gguf");
        write_mmproj_gguf(&out, &mut writer, &tensors, TargetQuant::F32).unwrap();
        Arc::new(GgufFile::from_bytes(Arc::from(fs::read(&out).unwrap())).unwrap())
    }

    /// A key that is present with the wrong type used to read as absent and take its default,
    /// so a corrupt file loaded a different graph without an error.
    #[test]
    fn the_audio_encoder_loader_refuses_mistyped_metadata() {
        AudioEncoderWeights::from_gguf(&audio_gguf_with(|_| {})).unwrap();
        for key in [
            "clip.audio.attention.layer_norm_epsilon",
            "clip.audio.num_mel_bins",
            "clip.audio.rope.freq_base",
            "clip.audio.window_size",
            "clip.audio.attention.head_count_kv",
        ] {
            let gguf = audio_gguf_with(|w| w.add_string(key, "not a number"));
            let err = AudioEncoderWeights::from_gguf(&gguf)
                .err()
                .unwrap_or_else(|| panic!("{key} as a string was accepted"));
            assert!(err.to_string().contains(key), "{key}: {err}");
        }
    }

    /// The converted GGUF of the LFM3-VL fixture, after `edit` changed its configs.
    fn vl_gguf_with(edit: impl FnOnce(&mut Value, &mut Value)) -> Arc<GgufFile> {
        let (mut cfg, mut preproc, raw) = vl_roundtrip_fixture();
        edit(&mut cfg, &mut preproc);
        let mut writer = GgufWriter::new();
        apply_lfm3_vl_mmproj_metadata(&cfg, Some(&preproc), &mut writer, "vl-edit").unwrap();
        let tensors = build_lfm3_vl_mmproj_tensors(&cfg, &raw).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("mmproj.gguf");
        write_mmproj_gguf(&out, &mut writer, &tensors, TargetQuant::F32).unwrap();
        Arc::new(GgufFile::from_bytes(Arc::from(fs::read(&out).unwrap())).unwrap())
    }

    #[test]
    fn the_vision_loader_refuses_metadata_its_tensors_contradict() {
        // the feed-forward width in the metadata is what the int8 scratch is sized from: a
        // block whose weights are another width used to load and then panic in the forward
        let gguf = vl_gguf_with(|cfg, _| cfg["vision_config"]["intermediate_size"] = json!(32));
        let err = VisionEncoderWeights::from_gguf(&gguf)
            .err()
            .expect("accepted");
        assert!(err.to_string().contains("feed_forward_length"), "{err}");

        // pixel bounds that cannot be a resize range
        for (min, max) in [(0u64, 1024u64), (4096, 1024), (1024, 0), (16, 1 << 31)] {
            let gguf = vl_gguf_with(|_, p| {
                p["min_pixels"] = json!(min);
                p["max_pixels"] = json!(max);
            });
            let err = VisionEncoderWeights::from_gguf(&gguf)
                .err()
                .unwrap_or_else(|| panic!("min={min} max={max} accepted"));
            assert!(err.to_string().contains("image_m"), "{min}/{max}: {err}");
        }
        // the fixture itself loads
        VisionEncoderWeights::from_gguf(&vl_gguf_with(|_, _| {})).unwrap();
    }

    /// SafeTensors bytes for `tensors` (all F32), enough for the converter's reader.
    fn safetensors_bytes(tensors: &TensorMap) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut body = Vec::new();
        for (name, (shape, data)) in tensors {
            let start = body.len();
            for v in data {
                body.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, body.len()]}),
            );
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&body);
        out
    }

    /// The whole file path, `convert_mmproj` itself, for an Omni checkpoint: the vision and
    /// the audio halves land in one GGUF that all four loaders accept.
    #[test]
    fn omni_checkpoint_converts_to_one_file_all_loaders_accept() {
        let (vl_cfg, preproc, vl_raw) = vl_roundtrip_fixture();
        let (mut cfg, detok, audio_raw) = audio_roundtrip_fixture();
        cfg["model_type"] = json!("lfm3_omni");
        cfg["architectures"] = json!(["Lfm3OmniForConditionalGeneration"]);
        cfg["vision_config"] = vl_cfg["vision_config"].clone();
        cfg["text_config"] = vl_cfg["text_config"].clone();
        let mut raw = vl_raw;
        raw.extend(audio_raw);

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            preproc.to_string(),
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("audio_detokenizer")).unwrap();
        std::fs::write(
            dir.path().join("audio_detokenizer").join("config.json"),
            detok.to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors"),
            safetensors_bytes(&raw),
        )
        .unwrap();

        let out = dir.path().join("mmproj.gguf");
        convert_mmproj(dir.path(), &out, TargetQuant::F32).unwrap();
        let gguf = Arc::new(GgufFile::from_bytes(Arc::from(fs::read(&out).unwrap())).unwrap());

        let vision = VisionEncoderWeights::from_gguf(&gguf).expect("vision half");
        assert!(vision.config.projector_type.is_lfm3());
        assert_eq!(vision.config.n_embd, 8);
        AudioEncoderWeights::from_gguf(&gguf).expect("audio encoder half");
        AudioDecoderWeights::from_gguf(&gguf).expect("audio depthformer half");
        DetokenizerWeights::from_gguf(&gguf).expect("audio detokenizer half");
    }

    /// Values, not shapes: the two patchifier spellings store the temporal pair differently.
    /// `patchifier.weight` is `[n_embd, channels, 2, p, p]` (the two time slices sit inside
    /// each channel); `patchifier.proj.weight` is `[n_embd, 2 * channels * p * p]` with the
    /// first time slice in the first half of the row. One channel pair of one pixel makes
    /// the difference observable.
    #[test]
    fn patchifier_spellings_split_the_temporal_pair_by_their_layout() {
        let cfg = json!({"vision_config": {"patch_size": 1, "num_channels": 2}});
        let split = |name: &str| {
            let mut raw = BTreeMap::new();
            // channel 0 at t0 and t1, then channel 1 at t0 and t1
            raw.insert(name.to_string(), (vec![1, 4], vec![10.0, 11.0, 20.0, 21.0]));
            let out = build_lfm3_vl_mmproj_tensors(&cfg, &raw).unwrap();
            let get = |n: &str| out.iter().find(|t| t.name == n).unwrap().data.clone();
            (get("v.patch_embd.weight"), get("v.patch_embd.weight.1"))
        };
        // channel-major: t0 takes the first value of each channel pair, t1 the second
        assert_eq!(
            split("vision_encoder.patchifier.weight"),
            (vec![10.0, 20.0], vec![11.0, 21.0])
        );
        // flat halves: t0 is the first half of the row, t1 the second
        assert_eq!(
            split("vision_encoder.patchifier.proj.weight"),
            (vec![10.0, 11.0], vec![20.0, 21.0])
        );
    }

    #[test]
    fn an_empty_or_duplicated_tensor_set_is_refused_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("mmproj.gguf");
        let tensor = |name: &str| MmprojTensor {
            name: name.into(),
            shape: vec![1],
            data: vec![0.0],
            force_f32: true,
        };
        let mut writer = GgufWriter::new();
        let err = write_mmproj_gguf(&out, &mut writer, &[], TargetQuant::F32).unwrap_err();
        assert!(err.to_string().contains("no projector tensors"), "{err}");
        let mut writer = GgufWriter::new();
        let dup = [tensor("a.x"), tensor("a.y"), tensor("a.x")];
        let err = write_mmproj_gguf(&out, &mut writer, &dup, TargetQuant::F32).unwrap_err();
        assert!(err.to_string().contains("a.x"), "{err}");
        assert!(!out.exists(), "nothing is written for a refused set");
    }

    #[test]
    fn language_model_tensors_are_not_read_for_the_projector() {
        // A backbone tensor in a dtype the converter cannot decode (here F8) must not fail a
        // projector conversion that never uses it; the projector's own tensors still load.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let header = br#"{"model.layers.0.mlp.weight":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]},"vision_tower.proj.weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let tensors = load_safetensors_file(&path, &is_projector_tensor).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["vision_tower.proj.weight"].1, vec![1.5]);
        // a tensor the projector does need keeps its error, naming the tensor
        let err = load_safetensors_file(&path, &|_| true).unwrap_err();
        assert!(
            err.to_string().contains("model.layers.0.mlp.weight"),
            "{err}"
        );
    }

    #[test]
    fn upsample_kernels_are_read_from_the_nested_model_object_too() {
        let top = json!({"upsample_kernel_sizes": [4, 4]});
        let nested = json!({"model": {"upsample_kernel_sizes": [8, 2]}});
        assert_eq!(detok_upsample_kernels(Some(&top)).unwrap(), vec![4, 4]);
        assert_eq!(detok_upsample_kernels(Some(&nested)).unwrap(), vec![8, 2]);
        assert!(detok_upsample_kernels(None).unwrap().is_empty());
    }

    #[test]
    fn config_fsq_levels_agree_with_the_builtin_codebook_and_are_bounded() {
        // four levels of four: the same table the runtime builds by default
        let (shape, data) = fsq_codebook_from_levels(&[4, 4, 4, 4]).unwrap();
        assert_eq!(shape, vec![256, 4]);
        let reference = crate::model::audio_decoder::build_default_fsq_codebook();
        assert_eq!(data.len(), reference.len());
        for (i, (got, want)) in data.iter().zip(&reference).enumerate() {
            assert!((got - want).abs() < 1e-6, "entry {i}: {got} != {want}");
        }
        // an odd level count is symmetric around zero
        let (_, odd) = fsq_codebook_from_levels(&[5]).unwrap();
        assert_eq!(odd, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        // untrusted input: a product that overflows or explodes, a zero level, no levels
        assert!(fsq_codebook_from_levels(&[usize::MAX, 2]).is_err());
        assert!(fsq_codebook_from_levels(&[1 << 20, 1 << 20]).is_err());
        assert!(fsq_codebook_from_levels(&[4, 0]).is_err());
        assert!(fsq_codebook_from_levels(&[]).is_err());
    }

    fn vl_cfg() -> Value {
        json!({
            "model_type": "lfm3_vl",
            "vision_config": {
                "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
                "num_attention_heads": 12, "num_key_value_heads": 4, "patch_size": 16,
                "image_size": 512, "layer_norm_eps": 1e-6
            },
            "text_config": {"hidden_size": 2048}
        })
    }

    #[test]
    fn vl_metadata_refuses_config_numbers_that_do_not_fit_u32() {
        let wide = (1u64 << 32) + 768;
        for (path, value) in [
            ("hidden_size", json!(wide)),
            ("num_hidden_layers", json!(wide)),
            ("patch_size", json!(wide)),
        ] {
            let mut cfg = vl_cfg();
            cfg["vision_config"][path] = value;
            let mut writer = GgufWriter::new();
            let err = apply_lfm3_vl_mmproj_metadata(&cfg, None, &mut writer, "m").expect_err(path);
            assert!(err.to_string().contains(path), "{path}: {err}");
        }
        for (key, value) in [
            ("max_pixels", json!(1u64 << 32)),
            ("min_pixels", json!(1u64 << 32)),
            ("tile_size", json!(1u64 << 32)),
            // 4,194,304 tokens of 32 x 32 pixels is 2^32 pixels
            ("min_image_tokens", json!(1u64 << 22)),
            ("max_image_tokens", json!(1u64 << 22)),
        ] {
            let mut writer = GgufWriter::new();
            let preproc = json!({ key: value });
            let err = apply_lfm3_vl_mmproj_metadata(&vl_cfg(), Some(&preproc), &mut writer, "m")
                .expect_err(key);
            assert!(err.to_string().contains(key), "{key}: {err}");
        }
    }

    #[test]
    fn a_processor_config_without_image_fields_does_not_hide_the_preprocessor_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("processor_config.json"),
            r#"{"processor_class": "Lfm3VlProcessor"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            r#"{"image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6]}"#,
        )
        .unwrap();
        let cfg = read_preprocessor_config(dir.path()).unwrap().unwrap();
        assert_eq!(cfg["image_mean"][0], 0.1);
    }

    #[test]
    fn a_tensor_span_past_the_end_of_the_file_is_refused_before_allocating() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,1125899906842624]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(&path, bytes).unwrap();
        let err = load_safetensors_file(&path, &|_| true).unwrap_err();
        assert!(err.to_string().contains("past the end"), "{err}");
    }
    use crate::convert::writer::MetadataValue;
    use crate::gguf::GgufFile;
    use crate::model::audio_decoder::{AudioDecoderWeights, DetokenizerWeights};
    use crate::model::audio_encoder::AudioEncoderWeights;
    use crate::model::vision_encoder::VisionEncoderWeights;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn vl_mmproj_metadata_and_tensor_transforms() {
        let cfg = json!({
            "architectures": ["Lfm3VlForConditionalGeneration"],
            "model_type": "lfm3_vl",
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "rope_theta": 10000.0,
                "patch_size": 2,
                "image_size": 512,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "hidden_size": 2048
            }
        });
        let preproc = json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "downsample_factor": 2,
            "tile_size": 512,
            "max_tiles": 10,
            "min_tiles": 2,
            "use_thumbnail": true
        });
        let mut writer = GgufWriter::new();
        apply_lfm3_vl_mmproj_metadata(&cfg, Some(&preproc), &mut writer, "vl-mmproj").unwrap();
        assert_eq!(
            writer.get_metadata("general.architecture"),
            Some(&MetadataValue::String("clip".into()))
        );
        assert_eq!(
            writer.get_metadata("clip.has_vision_encoder"),
            Some(&MetadataValue::Bool(true))
        );
        assert_eq!(
            writer.get_metadata("clip.has_audio_encoder"),
            Some(&MetadataValue::Bool(false))
        );
        assert_eq!(
            writer.get_metadata("clip.projector_type"),
            Some(&MetadataValue::String("lfm3vl".into()))
        );
        assert_eq!(
            writer.get_metadata("clip.vision.embedding_length"),
            Some(&MetadataValue::Uint32(768))
        );
        assert_eq!(
            writer.get_metadata("clip.vision.attention.head_count_kv"),
            Some(&MetadataValue::Uint32(4))
        );
        assert_eq!(
            writer.get_metadata("clip.vision.projection_dim"),
            Some(&MetadataValue::Uint32(2048))
        );
        assert_eq!(
            writer.get_metadata("clip.vision.spatial_merge_size"),
            Some(&MetadataValue::Uint32(2))
        );
        assert_eq!(
            writer.get_metadata("clip.vision.projector.scale_factor"),
            Some(&MetadataValue::Uint32(2))
        );

        let mut raw = BTreeMap::new();
        // 2D patch_embeddings `[4, 3 * 2 * 2 = 12]` -> reshaped to 4D `[4, 3, 2, 2]`
        raw.insert(
            "vision_tower.embeddings.patch_embeddings.weight".into(),
            (vec![4, 12], vec![1.0; 48]),
        );
        // Temporal patchifier `[4, 24]` (`half = 12 == 3 * 2 * 2`) -> split into two 4D `[4, 3, 2, 2]` tensors
        raw.insert(
            "vision_encoder.patchifier.proj.weight".into(),
            (vec![4, 24], (0..96).map(|x| x as f32).collect()),
        );
        // 3D spatial conv `[8, 4, 9]` -> reshaped to 4D `[8, 4, 3, 3]`
        raw.insert(
            "vision_feature_layer.conv.weight".into(),
            (vec![8, 4, 9], vec![0.5; 288]),
        );
        // Fused qkv `[6, 4]` -> three `[2, 4]` tensors
        raw.insert(
            "vision_encoder.blocks.0.attn.qkv_proj.weight".into(),
            (vec![6, 4], (0..24).map(|x| x as f32).collect()),
        );
        // Fused gate_up `[4, 4]` -> two `[2, 4]` tensors
        raw.insert(
            "vision_encoder.blocks.0.ffn.gate_up.weight".into(),
            (vec![4, 4], (0..16).map(|x| x as f32).collect()),
        );

        let tensors = build_lfm3_vl_mmproj_tensors(&cfg, &raw).unwrap();
        let by_name: BTreeMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        assert_eq!(by_name["mm.conv.weight"].shape, vec![8, 4, 3, 3]);
        assert_eq!(by_name["v.patch_embd.weight.1"].shape, vec![4, 3, 2, 2]);
        assert_eq!(by_name["v.blk.0.attn_q.weight"].shape, vec![2, 4]);
        assert_eq!(by_name["v.blk.0.attn_k.weight"].shape, vec![2, 4]);
        assert_eq!(by_name["v.blk.0.attn_v.weight"].shape, vec![2, 4]);
        assert_eq!(by_name["v.blk.0.ffn_gate.weight"].shape, vec![2, 4]);
        assert_eq!(by_name["v.blk.0.ffn_up.weight"].shape, vec![2, 4]);
    }

    #[test]
    fn audio_mmproj_metadata_embeddings_weightnorm_and_fsq() {
        let cfg = json!({
            "architectures": ["Lfm3AudioModel"],
            "model_type": "lfm3_audio",
            "codebooks": 2,
            "vocab_size": 4097,
            "encoder": {
                "features_dim": 128,
                "lfm": {
                    "hidden_size": 512,
                    "intermediate_size": 2048,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "conv_L_cache": 7,
                    "layer_types": ["conv", "full_attention"]
                }
            },
            "depthformer": {
                "layers": 6,
                "dim": 1024
            },
            "lfm": {
                "hidden_size": 2048
            }
        });
        let detok = json!({
            "n_fft": 1280,
            "hop_length": 320,
            "upsample_rates": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "lfm": {
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "conv_L_cache": 7,
                "sliding_window": 30,
                "layer_types": ["conv", "sliding_attention"]
            }
        });
        let mut writer = GgufWriter::new();
        apply_lfm3_audio_mmproj_metadata(&cfg, Some(&detok), true, &mut writer, "audio-mmproj")
            .unwrap();
        assert_eq!(
            writer.get_metadata("clip.has_audio_encoder"),
            Some(&MetadataValue::Bool(true))
        );
        assert_eq!(
            writer.get_metadata("clip.has_gen_audio_encoder"),
            Some(&MetadataValue::Bool(true))
        );
        assert_eq!(
            writer.get_metadata("clip.has_lfm3_audio_detokenizer"),
            Some(&MetadataValue::Bool(true))
        );
        assert_eq!(
            writer.get_metadata("clip.projector_type"),
            Some(&MetadataValue::String("lfm3a".into()))
        );
        assert_eq!(
            writer.get_metadata("clip.audio_gen.code.num_codebooks"),
            Some(&MetadataValue::Uint32(2))
        );

        let mut raw = BTreeMap::new();
        raw.insert(
            "codebook_embedding.emb.0.weight".into(),
            (vec![10, 4], vec![1.0; 40]),
        );
        raw.insert(
            "codebook_embedding.emb.1.weight".into(),
            (vec![10, 4], vec![2.0; 40]),
        );
        // WeightNorm on detokenizer.lin_in: original0 `[2, 1, 1]`, original1 `[2, 1, 5]` -> sliced to `[2, 1, 3]`
        raw.insert(
            "detokenizer.lin_in.parametrizations.weight.original0".into(),
            (vec![2, 1, 1], vec![2.0, 4.0]),
        );
        raw.insert(
            "detokenizer.lin_in.parametrizations.weight.original1".into(),
            (
                vec![2, 1, 5],
                vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            ),
        );

        let tensors = build_lfm3_audio_mmproj_tensors(&cfg, Some(&detok), &raw).unwrap();
        let by_name: BTreeMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        assert_eq!(by_name["a.gen.code.out_embd.weight"].shape, vec![20, 4]);
        let lin_in = by_name["detok.lin_in.weight"];
        assert_eq!(lin_in.shape, vec![2, 1, 3]);
        assert!((lin_in.data[0] - 1.2).abs() < 1e-5);
        assert!((lin_in.data[1] - 1.6).abs() < 1e-5);
        assert!((lin_in.data[2] - 0.0).abs() < 1e-5);
        assert!((lin_in.data[3] - 0.0).abs() < 1e-5);
        assert!((lin_in.data[4] - 4.0).abs() < 1e-5);
        let fsq = by_name["detok.fsq.codebook"];
        assert_eq!(fsq.shape, vec![4096, 6]);
        let fsq_runtime = by_name["a.detok.fsq_codebook"];
        assert_eq!(fsq_runtime.shape, vec![256, 4]);
    }

    /// A tiny LFM3-VL checkpoint: `(config, preprocessor config, tensors)`.
    fn vl_roundtrip_fixture() -> (Value, Value, TensorMap) {
        let cfg = json!({
            "architectures": ["Lfm3VlForConditionalGeneration"],
            "model_type": "lfm3_vl",
            "vision_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rope_theta": 10000.0,
                "patch_size": 2,
                "image_size": 8,
                "num_channels": 3,
                "layer_norm_eps": 1e-5
            },
            "text_config": {
                "hidden_size": 12
            }
        });
        let preproc = json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "downsample_factor": 2
        });

        let mut raw = BTreeMap::new();
        // Patchifier proj: `[8, 2 * (3 * 2 * 2) = 24]` + bias `[8]` + norm `[8]`
        raw.insert(
            "vision_encoder.patchifier.proj.weight".into(),
            (vec![8, 24], vec![0.05; 8 * 24]),
        );
        raw.insert(
            "vision_encoder.patchifier.proj.bias".into(),
            (vec![8], vec![0.01; 8]),
        );
        raw.insert(
            "vision_encoder.patchifier.norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "vision_encoder.patchifier.norm.bias".into(),
            (vec![8], vec![0.0; 8]),
        );

        // Layer 0: Conv2d block (`conv.in_proj`, `conv.conv` 3D `[8, 1, 9]`, `conv.out_proj`, `ffn.gate_up`, `ffn.down`)
        raw.insert(
            "vision_encoder.blocks.0.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "vision_encoder.blocks.0.conv.in_proj.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.0.conv.conv.weight".into(),
            (vec![8, 1, 9], vec![0.05; 8 * 9]),
        );
        raw.insert(
            "vision_encoder.blocks.0.conv.out_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.0.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "vision_encoder.blocks.0.ffn.gate_up.weight".into(),
            (vec![32, 8], vec![0.02; 32 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.0.ffn.down.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );

        // Layer 1: Attention block (`self_attn.q/k/v_proj`, `attn.q/k_norm`, `attn.out_proj`, `ffn.gate_up`, `ffn.down`)
        raw.insert(
            "vision_encoder.blocks.1.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.self_attn.q_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.self_attn.k_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.self_attn.v_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.attn.q_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "vision_encoder.blocks.1.attn.k_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "vision_encoder.blocks.1.attn.out_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.ffn.gate_up.weight".into(),
            (vec![32, 8], vec![0.02; 32 * 8]),
        );
        raw.insert(
            "vision_encoder.blocks.1.ffn.down.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );

        // Post norm + projector (`mm1_in_dim = 8 * 2^2 = 32 -> 16 -> 12`)
        raw.insert(
            "vision_encoder.final_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "vision_projector.norm.weight".into(),
            (vec![32], vec![1.0; 32]),
        );
        raw.insert(
            "vision_projector.norm.bias".into(),
            (vec![32], vec![0.0; 32]),
        );
        raw.insert(
            "vision_projector.fc1.weight".into(),
            (vec![16, 32], vec![0.02; 16 * 32]),
        );
        raw.insert(
            "vision_projector.fc1.bias".into(),
            (vec![16], vec![0.0; 16]),
        );
        raw.insert(
            "vision_projector.fc2.weight".into(),
            (vec![12, 16], vec![0.02; 12 * 16]),
        );
        raw.insert(
            "vision_projector.fc2.bias".into(),
            (vec![12], vec![0.0; 12]),
        );

        (cfg, preproc, raw)
    }

    /// A tiny LFM3-Audio checkpoint: `(config, detokenizer config, tensors)`.
    fn audio_roundtrip_fixture() -> (Value, Value, TensorMap) {
        let cfg = json!({
            "architectures": ["Lfm3AudioModel"],
            "model_type": "lfm3_audio",
            "codebooks": 2,
            "vocab_size": 16,
            "encoder": {
                "features_dim": 128,
                "lfm": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "conv_L_cache": 3,
                    "layer_types": ["conv", "full_attention"]
                }
            },
            "depthformer": {
                "layers": 1,
                "dim": 8,
                "intermediate_size": 16,
                "num_heads": 2,
                "num_kv_heads": 1
            },
            "lfm": {
                "hidden_size": 12
            }
        });
        let detok = json!({
            "n_fft": 16,
            "hop_length": 4,
            "lfm": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "conv_L_cache": 3,
                "sliding_window": 4,
                "layer_types": ["conv", "sliding_attention"]
            }
        });

        let mut raw = BTreeMap::new();
        // Encoder 3-layer conv stem (`a.conv1d.{0,1,2}` + `encoder.pre_encode.out`)
        for idx in 0..3 {
            let in_c = if idx == 0 { 1 } else { 4 };
            raw.insert(
                format!("encoder.pre_encode.conv.{idx}.weight"),
                (vec![4, in_c, 3, 3], vec![0.02; 4 * in_c * 9]),
            );
            raw.insert(
                format!("encoder.pre_encode.conv.{idx}.bias"),
                (vec![4], vec![0.0; 4]),
            );
        }
        raw.insert(
            "encoder.pre_encode.out.weight".into(),
            (vec![8, 64], vec![0.02; 8 * 64]),
        );
        raw.insert(
            "encoder.pre_encode.out.bias".into(),
            (vec![8], vec![0.0; 8]),
        );
        // Encoder block 0 (ShortConv) + block 1 (Attention)
        raw.insert(
            "encoder.lfm.layers.0.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "encoder.lfm.layers.0.conv.in_proj.weight".into(),
            (vec![24, 8], vec![0.02; 24 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.0.conv.conv.weight".into(),
            (vec![8, 1, 3], vec![0.05; 8 * 3]),
        );
        raw.insert(
            "encoder.lfm.layers.0.conv.out_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.0.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "encoder.lfm.layers.0.feed_forward.w1.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.0.feed_forward.w2.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );
        raw.insert(
            "encoder.lfm.layers.0.feed_forward.w3.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );

        raw.insert(
            "encoder.lfm.layers.1.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.q_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.k_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.v_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.o_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.q_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.k_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "encoder.lfm.layers.1.self_attn.gate_proj.weight".into(),
            (vec![2, 8], vec![0.02; 2 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.feed_forward.w1.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "encoder.lfm.layers.1.feed_forward.w2.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );
        raw.insert(
            "encoder.lfm.layers.1.feed_forward.w3.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "encoder.embedding_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );

        // MLP adapter (`mlp.model.{0,1,2}` -> `mm.a.mlp.{0,1,3}`)
        raw.insert("mlp.model.0.weight".into(), (vec![8], vec![1.0; 8]));
        raw.insert("mlp.model.0.bias".into(), (vec![8], vec![0.0; 8]));
        raw.insert(
            "mlp.model.1.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert("mlp.model.1.bias".into(), (vec![16], vec![0.0; 16]));
        raw.insert(
            "mlp.model.2.weight".into(),
            (vec![12, 16], vec![0.02; 12 * 16]),
        );
        raw.insert("mlp.model.2.bias".into(), (vec![12], vec![0.0; 12]));

        // Depthformer (`depth_linear`, 1 attention layer, 2 codebooks of vocab 16)
        raw.insert(
            "depth_linear.weight".into(),
            (vec![8, 12], vec![0.02; 8 * 12]),
        );
        raw.insert("depth_linear.bias".into(), (vec![8], vec![0.0; 8]));
        raw.insert(
            "depthformer.model.layers.0.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.self_attn.q_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.self_attn.k_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.self_attn.v_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.self_attn.o_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.self_attn.q_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "depthformer.model.layers.0.self_attn.k_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "depthformer.model.layers.0.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.feed_forward.w1.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "depthformer.model.layers.0.feed_forward.w2.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );
        raw.insert(
            "depthformer.model.layers.0.feed_forward.w3.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "codebook_embedding.embedding_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        for cb in 0..2 {
            raw.insert(
                format!("codebook_embedding.emb.{cb}.weight"),
                (vec![16, 12], vec![0.02; 16 * 12]),
            );
            raw.insert(
                format!("depth_embeddings.{cb}.weight"),
                (vec![16, 8], vec![0.02; 16 * 8]),
            );
            raw.insert(
                format!("depth_embeddings.{cb}.linear.weight"),
                (vec![16, 8], vec![0.02; 16 * 8]),
            );
        }

        // Detokenizer (`fsq_proj_dim = 4`, `n_embd = 8`, `n_fft = 16 -> single_frame_bins = 18`)
        raw.insert(
            "detokenizer.tfm_in_proj.weight".into(),
            (vec![8, 4], vec![0.02; 8 * 4]),
        );
        raw.insert(
            "detokenizer.tfm_in_proj.bias".into(),
            (vec![8], vec![0.0; 8]),
        );
        raw.insert(
            "detokenizer.quant_first_out.weight".into(),
            (vec![4, 16], vec![0.02; 4 * 16]),
        );
        raw.insert(
            "detokenizer.quant_first_out.bias".into(),
            (vec![4], vec![0.0; 4]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.conv.in_proj.weight".into(),
            (vec![24, 8], vec![0.02; 24 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.conv.conv.weight".into(),
            (vec![8, 1, 3], vec![0.05; 8 * 3]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.conv.out_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.feed_forward.w1.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.feed_forward.w2.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );
        raw.insert(
            "detokenizer.lfm.layers.0.feed_forward.w3.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );

        raw.insert(
            "detokenizer.lfm.layers.1.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.q_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.k_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.v_proj.weight".into(),
            (vec![4, 8], vec![0.02; 4 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.o_proj.weight".into(),
            (vec![8, 8], vec![0.02; 8 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.q_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.k_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.self_attn.gate_proj.weight".into(),
            (vec![2, 8], vec![0.02; 2 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.ffn_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.feed_forward.w1.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.feed_forward.w2.weight".into(),
            (vec![8, 16], vec![0.02; 8 * 16]),
        );
        raw.insert(
            "detokenizer.lfm.layers.1.feed_forward.w3.weight".into(),
            (vec![16, 8], vec![0.02; 16 * 8]),
        );
        raw.insert(
            "detokenizer.lfm.embedding_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "detokenizer.tfm_out_proj.weight".into(),
            (vec![18, 8], vec![0.02; 18 * 8]),
        );
        raw.insert(
            "detokenizer.tfm_out_proj.bias".into(),
            (vec![18], vec![0.0; 18]),
        );

        (cfg, detok, raw)
    }

    #[test]
    fn vl_mmproj_round_trips_through_vision_encoder_loader() {
        let (cfg, preproc, raw) = vl_roundtrip_fixture();

        let mut writer = GgufWriter::new();
        apply_lfm3_vl_mmproj_metadata(&cfg, Some(&preproc), &mut writer, "vl-rt").unwrap();
        let tensors = build_lfm3_vl_mmproj_tensors(&cfg, &raw).unwrap();

        let tmp =
            std::env::temp_dir().join(format!("cera_vl_mmproj_rt_{}.gguf", std::process::id()));
        write_mmproj_gguf(&tmp, &mut writer, &tensors, TargetQuant::F32).unwrap();
        let gguf = Arc::new(GgufFile::from_bytes(Arc::from(fs::read(&tmp).unwrap())).unwrap());
        let _ = fs::remove_file(&tmp);

        let loaded = VisionEncoderWeights::from_gguf(&gguf).expect("load converted LFM3-VL mmproj");
        assert!(loaded.config.projector_type.is_lfm3());
        assert_eq!(loaded.config.n_layer, 2);
        assert_eq!(loaded.config.n_embd, 8);
        assert_eq!(loaded.config.n_head, 2);
        assert_eq!(loaded.config.n_head_kv, 1);
        assert_eq!(loaded.config.projection_dim, 12);
        assert!(loaded.patch_embed_1.is_some());
    }

    #[test]
    fn audio_mmproj_round_trips_through_audio_encoder_decoder_and_detokenizer_loaders() {
        let (cfg, detok, raw) = audio_roundtrip_fixture();

        let mut writer = GgufWriter::new();
        apply_lfm3_audio_mmproj_metadata(&cfg, Some(&detok), true, &mut writer, "audio-rt")
            .unwrap();
        let tensors = build_lfm3_audio_mmproj_tensors(&cfg, Some(&detok), &raw).unwrap();

        let tmp =
            std::env::temp_dir().join(format!("cera_audio_mmproj_rt_{}.gguf", std::process::id()));
        write_mmproj_gguf(&tmp, &mut writer, &tensors, TargetQuant::F32).unwrap();
        let gguf = Arc::new(GgufFile::from_bytes(Arc::from(fs::read(&tmp).unwrap())).unwrap());
        let _ = fs::remove_file(&tmp);

        let enc = AudioEncoderWeights::from_gguf(&gguf).expect("load converted LFM3A encoder");
        assert!(enc.is_lfm3a());
        assert_eq!(enc.config.n_layer, 2);
        assert_eq!(enc.config.n_embd, 8);
        assert_eq!(enc.config.llm_hidden_size, 12);

        let dec = AudioDecoderWeights::from_gguf(&gguf).expect("load converted LFM3A depthformer");
        assert_eq!(dec.decoder_config.n_codebook, 2);
        // the schema decides the sampler, not the codebook count (this file has 2, not 12)
        assert!(dec.is_lfm3a_gen());
        assert_eq!(dec.decoder_config.n_vocab, 16);
        assert_eq!(dec.depthformer_config.n_layer, 1);

        let det = DetokenizerWeights::from_gguf(&gguf).expect("load converted LFM3A detokenizer");
        assert!(det.is_lfm3a_gen());
        assert_eq!(det.config.n_layer, 2);
        assert_eq!(det.config.n_embd, 8);
    }

    #[test]
    fn canonical_hf_omni_and_vl_tensors_convert_cleanly() {
        let cfg = json!({
            "architectures": ["Lfm3OmniForConditionalGeneration"],
            "model_type": "lfm3_omni",
            "num_codebooks": 2,
            "codebook_size": 16,
            "downsample_factor": 2,
            "projector_hidden_size": 16,
            "vision_config": {
                "model_type": "lfm3_vl_vision",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "patch_size": 2,
                "num_channels": 3
            },
            "audio_encoder_config": {
                "num_mel_bins": 128,
                "conv_channels": 4,
                "hidden_size": 8,
                "sliding_window": 32,
                "text_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "conv_L_cache": 3
                }
            },
            "depth_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1
            },
            "text_config": {
                "hidden_size": 12
            }
        });
        assert_eq!(detect_mmproj_kind(&cfg).unwrap(), MmprojKind::Omni);

        let mut raw = BTreeMap::new();
        // Canonical vision tower (`model.vision_tower.*` without `vision_encoder.`)
        raw.insert(
            "model.vision_tower.patchifier.weight".into(),
            (vec![4, 24], (0..96).map(|x| x as f32).collect()),
        );
        raw.insert(
            "model.vision_tower.embedding_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "model.vision_tower.layers.0.operator_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "model.vision_tower.layers.0.conv.in_proj.weight".into(),
            (vec![8, 4], vec![0.1; 32]),
        );
        raw.insert(
            "model.vision_tower.layers.0.conv.spatial_conv.weight".into(),
            (vec![4, 1, 3, 3], vec![0.1; 36]),
        );
        raw.insert(
            "model.vision_tower.layers.0.conv.out_proj.weight".into(),
            (vec![4, 4], vec![0.1; 16]),
        );
        raw.insert(
            "model.vision_tower.layers.0.ffn_norm.weight".into(),
            (vec![4], vec![1.0; 4]),
        );
        raw.insert(
            "model.vision_tower.layers.0.feed_forward.w1.weight".into(),
            (vec![8, 4], vec![0.1; 32]),
        );
        raw.insert(
            "model.vision_tower.layers.0.feed_forward.w2.weight".into(),
            (vec![4, 8], vec![0.1; 32]),
        );
        raw.insert(
            "model.vision_tower.layers.0.feed_forward.w3.weight".into(),
            (vec![8, 4], vec![0.1; 32]),
        );
        raw.insert(
            "model.multi_modal_projector.layer_norm.weight".into(),
            (vec![16], vec![1.0; 16]),
        );
        raw.insert(
            "model.multi_modal_projector.linear_1.weight".into(),
            (vec![16, 16], vec![0.1; 256]),
        );
        raw.insert(
            "model.multi_modal_projector.linear_2.weight".into(),
            (vec![12, 16], vec![0.1; 192]),
        );
        // Ensure LLM layer is NOT mistaken for a vision layer
        raw.insert(
            "model.language_model.layers.0.operator_norm.weight".into(),
            (vec![12], vec![1.0; 12]),
        );
        raw.insert(
            "model.layers.0.operator_norm.weight".into(),
            (vec![12], vec![1.0; 12]),
        );

        let v_tensors = build_lfm3_vl_mmproj_tensors(&cfg, &raw).unwrap();
        let v_by_name: BTreeMap<_, _> = v_tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        assert_eq!(v_by_name["v.patch_embd.weight"].shape, vec![4, 3, 2, 2]);
        assert_eq!(v_by_name["v.patch_embd.weight.1"].shape, vec![4, 3, 2, 2]);
        assert_eq!(v_by_name["v.embedding_norm.weight"].shape, vec![4]);
        assert_eq!(v_by_name["v.blk.0.ln1.weight"].shape, vec![4]);
        assert_eq!(
            v_by_name["v.blk.0.conv2d_spatial.weight"].shape,
            vec![4, 1, 3, 3]
        );
        assert_eq!(v_by_name["v.blk.0.ffn_gate.weight"].shape, vec![8, 4]);
        assert_eq!(v_by_name["v.blk.0.ffn_up.weight"].shape, vec![8, 4]);
        assert_eq!(v_by_name["v.blk.0.ffn_down.weight"].shape, vec![4, 8]);
        assert_eq!(v_by_name["mm.input_norm.weight"].shape, vec![16]);
        assert_eq!(v_by_name["mm.0.weight"].shape, vec![16, 16]);
        assert_eq!(v_by_name["mm.2.weight"].shape, vec![12, 16]);

        // Canonical audio encoder, projector, codebook_embedding, and depth_decoder
        raw.insert(
            "model.audio_encoder.downsample.conv_in.weight".into(),
            (vec![4, 1, 3, 3], vec![0.1; 36]),
        );
        raw.insert(
            "model.audio_encoder.lfm.layers.0.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "model.audio_encoder.lfm.embedding_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "model.audio_projector.layer_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "model.audio_projector.linear_1.weight".into(),
            (vec![16, 8], vec![0.1; 128]),
        );
        raw.insert(
            "model.audio_projector.linear_2.weight".into(),
            (vec![12, 16], vec![0.1; 192]),
        );
        raw.insert(
            "model.codebook_embedding.weight".into(),
            (vec![32, 12], vec![0.1; 384]),
        );
        raw.insert(
            "model.depth_decoder.depth_linear.weight".into(),
            (vec![8, 12], vec![0.1; 96]),
        );
        raw.insert(
            "model.depth_decoder.mask_embedding".into(),
            (vec![8], vec![0.2; 8]),
        );
        raw.insert(
            "model.depth_decoder.model.layers.0.operator_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        raw.insert(
            "model.depth_decoder.model.embedding_norm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        for cb in 0..2 {
            raw.insert(
                format!("model.depth_decoder.depth_embeddings.{cb}.embedding.weight"),
                (vec![16, 8], vec![0.1; 128]),
            );
            raw.insert(
                format!("model.depth_decoder.depth_embeddings.{cb}.embedding_norm.weight"),
                (vec![8], vec![1.0; 8]),
            );
        }
        raw.insert(
            "detokenizer.quantizer.project_out.weight".into(),
            (vec![8, 48], vec![0.1; 384]),
        );

        let a_tensors = build_lfm3_audio_mmproj_tensors(&cfg, None, &raw).unwrap();
        let a_by_name: BTreeMap<_, _> = a_tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        assert_eq!(a_by_name["a.conv1d.0.weight"].shape, vec![4, 1, 3, 3]);
        assert_eq!(a_by_name["a.blk.0.ln1.weight"].shape, vec![8]);
        assert_eq!(a_by_name["a.post_ln.weight"].shape, vec![8]);
        assert_eq!(a_by_name["mm.a.mlp.0.weight"].shape, vec![8]);
        assert_eq!(a_by_name["mm.a.mlp.1.weight"].shape, vec![16, 8]);
        assert_eq!(a_by_name["mm.a.mlp.3.weight"].shape, vec![12, 16]);
        assert_eq!(a_by_name["a.gen.code.out_embd.weight"].shape, vec![32, 12]);
        assert_eq!(a_by_name["a.gen.code.in_proj.weight"].shape, vec![8, 12]);
        assert_eq!(a_by_name["a.gen.code.mask_embd"].shape, vec![8]);
        assert_eq!(a_by_name["a.gen.code.blk.0.ln1.weight"].shape, vec![8]);
        assert_eq!(a_by_name["a.gen.code.output_norm.weight"].shape, vec![8]);
        assert_eq!(a_by_name["a.gen.code.tok_embd.weight"].shape, vec![32, 8]);
        assert_eq!(a_by_name["a.gen.code.in_norm.weight"].shape, vec![2, 8]);
        assert_eq!(
            a_by_name["a.detok.quant_first_out.weight"].shape,
            vec![8, 48]
        );
    }

    #[test]
    fn mmproj_bounds_and_backbone_prefix_checks() {
        assert!(!is_projector_tensor(
            "model.lfm.layers.0.operator_norm.weight"
        ));
        assert!(!is_projector_tensor(
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(is_projector_tensor("model.vision_tower.patch_embed.weight"));

        assert!(
            compute_lfm_ffn_dim(
                &serde_json::json!({
                    "intermediate_size": 512,
                    "block_auto_adjust_ff_dim": true,
                    "block_ffn_dim_multiplier": 0.0,
                }),
                512,
            )
            .is_err()
        );
        assert!(
            compute_lfm_ffn_dim(
                &serde_json::json!({
                    "intermediate_size": 512,
                    "block_auto_adjust_ff_dim": true,
                    "block_ffn_dim_multiplier": -1.0,
                }),
                512,
            )
            .is_err()
        );
        assert!(
            compute_lfm_ffn_dim(
                &serde_json::json!({
                    "intermediate_size": 512,
                    "block_auto_adjust_ff_dim": true,
                    "block_ffn_dim_multiplier": 1e30,
                }),
                512,
            )
            .is_err()
        );
        assert!(
            compute_lfm_ffn_dim(
                &serde_json::json!({
                    "intermediate_size": -1,
                }),
                512,
            )
            .is_err()
        );

        let raw = BTreeMap::new();
        let zero_patch_cfg = serde_json::json!({
            "vision_config": { "patch_size": 0, "num_channels": 3 }
        });
        assert!(build_lfm3_vl_mmproj_tensors(&zero_patch_cfg, &raw).is_err());

        let zero_chan_cfg = serde_json::json!({
            "vision_config": { "patch_size": 16, "num_channels": 0 }
        });
        assert!(build_lfm3_vl_mmproj_tensors(&zero_chan_cfg, &raw).is_err());

        for bad_detok in [
            serde_json::json!({ "upsample_kernel_sizes": [0, 16] }),
            serde_json::json!({ "upsample_kernel_sizes": [-4, 16] }),
            serde_json::json!({ "upsample_kernel_sizes": ["bad", 16] }),
            serde_json::json!({ "upsample_kernel_sizes": 16 }),
        ] {
            assert!(
                detok_upsample_kernels(Some(&bad_detok)).is_err(),
                "expected error for {bad_detok}"
            );
        }
        assert_eq!(
            detok_upsample_kernels(Some(&serde_json::json!({
                "model": { "upsample_kernel_sizes": [16, 10, 8] }
            })))
            .unwrap(),
            vec![16, 10, 8]
        );
    }
}

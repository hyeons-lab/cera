//! Mapping from Hugging Face `config.json` to GGUF model metadata.

use crate::convert::writer::GgufWriter;
use crate::session::CeraError;
use serde::Deserialize;
use serde_json::Value;

/// A tensor the GGUF needs that the checkpoint does not store.
#[derive(Debug, Clone)]
pub struct GeneratedTensor {
    pub name: String,
    /// GGUF dimensions (fastest-varying first).
    pub dims: Vec<u64>,
    pub data: Vec<f32>,
}

/// HF Transformers `config.json` schema representation.
#[derive(Debug, Clone, Deserialize)]
pub struct HfModelConfig {
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    #[serde(alias = "layer_norm_eps", alias = "layer_norm_epsilon")]
    pub rms_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
    #[serde(alias = "head_dim", alias = "key_length")]
    pub head_dim: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub bos_token_id: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub eos_token_id: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub pad_token_id: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn parse_token_id_value(val: &Value) -> Option<u32> {
    match val {
        Value::Number(n) => n.as_u64().and_then(|v| u32::try_from(v).ok()),
        Value::Array(arr) => arr.first().and_then(parse_token_id_value),
        Value::String(s) => s.parse::<u32>().ok(),
        _ => None,
    }
}

fn deserialize_token_id<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<Value> = Option::deserialize(deserializer)?;
    Ok(val.as_ref().and_then(parse_token_id_value))
}

/// The most blocks a converted LFM2 may declare (enforced by `lfm2_layout`).
const MAX_LFM2_LAYERS: usize = 4096;

/// The most tokens the llama.cpp vocabulary layout will pad to; a larger `vocab_size`
/// is refused rather than written as a token list shorter than the embedding.
pub(crate) const MAX_PADDED_VOCAB: usize = 1_000_000;

/// Widest attention head an `lfm3` conversion accepts: the generated `rope_freqs.weight`
/// has one `f32` per rotary pair, so the width must not come unchecked from a remote config.
const MAX_LFM3_HEAD_DIM: usize = 4096;

impl HfModelConfig {
    /// Parse from JSON bytes.
    pub fn parse_from_bytes(bytes: &[u8]) -> Result<Self, CeraError> {
        let mut cfg: Self = serde_json::from_slice(bytes)
            .map_err(|e| CeraError::Backend(format!("failed to parse model config.json: {e}")))?;
        // the audio and multimodal checkpoints keep their language model under `lfm` or `language_config`
        if !cfg.extra.contains_key("text_config")
            && let Some(lfm) = cfg
                .extra
                .get("lfm")
                .or_else(|| cfg.extra.get("language_config"))
                .filter(|v| v.is_object() && cfg.names_lfm3(v))
                .cloned()
        {
            cfg.extra.insert("text_config".into(), lfm);
        }
        cfg.resolve_nested_text_config();
        cfg.resolve_alternate_spellings();
        Ok(cfg)
    }

    /// Whether the top level or the nested language-model config names an `lfm3`
    /// architecture or model type.
    fn names_lfm3(&self, lfm: &Value) -> bool {
        if self.model_type.to_ascii_lowercase().contains("lfm3")
            || lfm
                .get("model_type")
                .and_then(Value::as_str)
                .is_some_and(|m| m.to_ascii_lowercase().contains("lfm3"))
        {
            return true;
        }
        let nested = lfm
            .get("architectures")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str);
        self.architectures
            .iter()
            .map(String::as_str)
            .chain(nested)
            .any(|a| a.to_ascii_lowercase().contains("lfm3"))
    }

    /// Parse from JSON string.
    pub fn from_json_str(json_str: &str) -> Result<Self, CeraError> {
        Self::parse_from_bytes(json_str.as_bytes())
    }

    /// Which of the `layers` blocks are attention (the rest are gated convolutions):
    /// `layer_types` when the config has it, else the `full_attn_idxs` list of the
    /// first LFM2 checkpoints. `None` rather than a guess when the layout is unusable;
    /// [`Self::lfm2_layout`] says why.
    pub(crate) fn lfm2_attention_layers(&self, layers: usize) -> Option<Vec<bool>> {
        self.lfm2_layout(layers).ok()
    }

    /// [`Self::lfm2_attention_layers`] with the cause of a refusal. The layout is
    /// refused when it is missing, is not one entry per block, holds an entry it cannot
    /// read, or has no attention block at all: any of those would write a plausible but
    /// wrong `head_count_kv`.
    fn lfm2_layout(&self, layers: usize) -> Result<Vec<bool>, String> {
        // `num_hidden_layers` is untrusted, and sizes what is built here and what the
        // writer builds from it; no real LFM2 comes near the bound
        if layers == 0 {
            return Err("`num_hidden_layers` is missing or zero".into());
        }
        if layers > MAX_LFM2_LAYERS {
            return Err(format!(
                "`num_hidden_layers` is {layers}; the converter accepts at most {MAX_LFM2_LAYERS}"
            ));
        }
        let attention = if let Some(types) = self.hparam("layer_types") {
            let types = types.as_array().ok_or("`layer_types` is not a list")?;
            if types.len() != layers {
                return Err(format!(
                    "`layer_types` has {} entries but `num_hidden_layers` is {layers}",
                    types.len()
                ));
            }
            types
                .iter()
                .enumerate()
                .map(|(i, t)| match t.as_str() {
                    Some("conv") => Ok(false),
                    Some(t) if t.ends_with("attention") => Ok(true),
                    _ => Err(format!(
                        "`layer_types[{i}]` is {t}, not `conv` or an attention type"
                    )),
                })
                .collect::<Result<Vec<bool>, String>>()?
        } else {
            let idxs = self
                .hparam("full_attn_idxs")
                .ok_or("neither `layer_types` nor `full_attn_idxs` is present")?
                .as_array()
                .ok_or("`full_attn_idxs` is not a list")?;
            let mut attention = vec![false; layers];
            for idx in idxs {
                let i = idx
                    .as_u64()
                    .and_then(|i| usize::try_from(i).ok())
                    .filter(|&i| i < layers)
                    .ok_or_else(|| {
                        format!("`full_attn_idxs` entry {idx} is not an index below {layers}")
                    })?;
                attention[i] = true;
            }
            attention
        };
        if !attention.contains(&true) {
            return Err("the layout has no attention layer".into());
        }
        Ok(attention)
    }

    /// The attention head width: `head_dim`, or `hidden_size / num_attention_heads`
    /// when the config leaves it null (the 1.7B does in some exports).
    pub(crate) fn lfm3_head_dim(&self) -> Option<usize> {
        self.head_dim.filter(|&d| d > 0).or_else(|| {
            self.hidden_size?
                .checked_div(self.num_attention_heads?)
                .filter(|&d| d > 0)
        })
    }

    /// Per `layer_types` entry of an `lfm3` config: whether it is `sliding_attention`
    /// (empty when the key is absent or not a list).
    fn lfm3_sliding_pattern(&self) -> Vec<bool> {
        self.hparam("layer_types")
            .and_then(Value::as_array)
            .map(|types| {
                types
                    .iter()
                    .map(|t| t.as_str() == Some("sliding_attention"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether any layer of an `lfm3` config slides.
    fn lfm3_has_sliding_layers(&self) -> bool {
        self.lfm3_sliding_pattern().into_iter().any(|s| s)
    }

    /// How many rotary pairs of a `head_dim`-wide head rotate at `factor`: the share is
    /// rounded down to whole pairs. The refusal of a factor that rotates none and the
    /// generated `rope_freqs.weight` both use this one rule.
    fn lfm3_rotated_pairs(factor: f64, head_dim: usize) -> usize {
        (((factor * head_dim as f64) as usize) / 2).min(head_dim / 2)
    }

    /// The `sliding_window` of an `lfm3` config as a usable width (1..=u32::MAX).
    fn lfm3_sliding_window(&self) -> Option<u32> {
        self.hparam("sliding_window")
            .and_then(Value::as_u64)
            .and_then(|w| u32::try_from(w).ok())
            .filter(|&w| w > 0)
    }

    /// What the `lfm3` loader would refuse in a routed-expert config, found at conversion
    /// time: `apply_lfm3_moe_metadata` writes these values as they are, and the loader
    /// requires `expert_used_count` and `expert_feed_forward_length`, a unit gate scale and
    /// normalised routing. `None` for a dense model.
    fn lfm3_moe_problem(&self) -> Option<String> {
        let (count_key, count) = Self::LFM3_EXPERT_COUNT_KEYS
            .iter()
            .find_map(|k| self.hparam(k).map(|v| (*k, v)))?;
        let Some(count) = count.as_u64().and_then(|v| u32::try_from(v).ok()) else {
            return Some(format!("`{count_key}` must be an integer in 0..=u32::MAX"));
        };
        if count == 0 {
            return None;
        }
        let u32_of = |keys: &[&'static str]| {
            keys.iter().find_map(|k| {
                self.hparam(k)
                    .map(|v| (*k, v.as_u64().and_then(|v| u32::try_from(v).ok())))
            })
        };
        match u32_of(&["num_experts_per_tok", "n_activated_experts"]) {
            None => return Some("`num_experts_per_tok` is missing".into()),
            Some((key, None)) => {
                return Some(format!("`{key}` must be an integer in 1..=u32::MAX"));
            }
            Some((key, Some(used))) if used == 0 || used > count => {
                return Some(format!("`{key}` ({used}) must be in 1..={count}"));
            }
            Some(_) => {}
        }
        match u32_of(&["moe_intermediate_size"]) {
            None => return Some("`moe_intermediate_size` is missing".into()),
            Some((key, None)) => {
                return Some(format!("`{key}` must be an integer in 1..=u32::MAX"));
            }
            Some((key, Some(ff))) => {
                if ff == 0 || ff % 32 != 0 {
                    return Some(format!("`{key}` ({ff}) must be a positive multiple of 32"));
                }
                if self
                    .intermediate_size
                    .is_some_and(|dense| ff as usize > dense)
                {
                    return Some(format!(
                        "`{key}` ({ff}) exceeds `intermediate_size`, which sizes the FFN scratch"
                    ));
                }
            }
        }
        if let Some(scale) = self.hparam("routed_scaling_factor")
            && !scale
                .as_f64()
                .is_some_and(|f| f.is_finite() && (f - 1.0).abs() <= 1e-6)
        {
            return Some(format!(
                "`routed_scaling_factor` ({scale}) is not supported (only 1.0)"
            ));
        }
        if let Some(norm) = self.hparam("norm_topk_prob")
            && norm.as_bool() != Some(true)
        {
            return Some(format!(
                "`norm_topk_prob` ({norm}) is not supported (only true: normalised routing)"
            ));
        }
        if let Some((key, lead)) = u32_of(&[
            "num_dense_layers",
            "first_k_dense_replace",
            "leading_dense_block_count",
        ]) {
            match lead {
                None => return Some(format!("`{key}` must be an integer in 0..=u32::MAX")),
                Some(lead) if self.num_hidden_layers.is_some_and(|n| lead as usize > n) => {
                    return Some(format!("`{key}` ({lead}) exceeds `num_hidden_layers`"));
                }
                Some(_) => {}
            }
        }
        None
    }

    /// The routed expert count the metadata advertises (`lfm3.expert_count`) and the
    /// converters enforce: the first of `num_experts` / `n_routed_experts` present at the top
    /// level or in `text_config`, when it is a positive integer. One lookup for both, so the
    /// completeness check can never disagree with the metadata.
    pub(crate) fn expert_count(&self) -> Option<usize> {
        Self::LFM3_EXPERT_COUNT_KEYS
            .iter()
            .find_map(|k| self.hparam(k))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
    }

    /// The config keys the expert count is read from, in order.
    const LFM3_EXPERT_COUNT_KEYS: [&'static str; 2] = ["num_experts", "n_routed_experts"];

    /// Why an `lfm3` config cannot be converted faithfully, for members the converter and
    /// the loaders do not read: rope scaling (a `rope_type` other than `default`, or a
    /// `factor`), a `partial_rotary_factor` on the sliding layers (only the global layers'
    /// is honoured) and `conv_bias`. Ignoring any of them would write a file that loads
    /// and produces plausible, wrong text.
    fn lfm3_unhonoured_member(&self) -> Option<String> {
        if self.hparam("conv_bias").and_then(Value::as_bool) == Some(true) {
            return Some("`conv_bias` is true; lfm3 short convolutions here have no bias".into());
        }
        let params = self.hparam("rope_parameters");
        let sliding = params.and_then(|p| p.get("sliding_attention"));
        let swa = self.hparam("swa_rope_parameters");
        for (place, s) in [
            ("rope_parameters.sliding_attention", sliding),
            ("swa_rope_parameters", swa),
        ] {
            if let Some(v) = s
                .and_then(|p| p.get("partial_rotary_factor"))
                .filter(|v| !v.is_null() && v.as_f64() != Some(1.0))
            {
                return Some(format!(
                    "`{place}.partial_rotary_factor` ({v}) is set; \
                     only the global layers' is read"
                ));
            }
        }
        // `rope_scaling` is the older spelling of `rope_parameters`
        for (place, p) in [
            ("rope_parameters", params),
            (
                "rope_parameters.full_attention",
                params.and_then(|p| p.get("full_attention")),
            ),
            ("rope_parameters.sliding_attention", sliding),
            ("swa_rope_parameters", swa),
            ("rope_scaling", self.hparam("rope_scaling")),
        ] {
            let Some(p) = p else { continue };
            let kind = p.get("rope_type").or_else(|| p.get("type"));
            if let Some(k) = kind.filter(|k| !k.is_null() && k.as_str() != Some("default")) {
                return Some(format!(
                    "`{place}.rope_type` is {k}; rope scaling is not supported"
                ));
            }
            if let Some(f) = p.get("factor").filter(|f| !f.is_null()) {
                return Some(format!(
                    "`{place}.factor` is {f}; rope scaling is not supported"
                ));
            }
        }
        None
    }

    fn is_lfm3_moe_or_vl(&self) -> bool {
        let mt = self.model_type.to_ascii_lowercase();
        if mt.contains("moe") || mt.contains("vl") || mt.contains("omni") || mt.contains("audio") {
            return true;
        }
        self.architectures.iter().any(|a| {
            let al = a.to_ascii_lowercase();
            al.contains("moe") || al.contains("vl") || al.contains("omni") || al.contains("audio")
        }) || self.hparam("num_experts").is_some()
            || self.hparam("n_routed_experts").is_some()
            || self.hparam("swa_rope_parameters").is_some()
            || self.hparam("full_attention_rope_theta").is_some()
    }

    /// The share of each head's rotary pairs that rotate on the global-attention layers
    /// (`rope_parameters.full_attention.partial_rotary_factor`, or flat on MoE/VL configs):
    /// 1.0 when the key is absent or null, otherwise a finite number in (0, 1]. Anything
    /// else would write a `rope_freqs.weight` that rotates nothing or everything.
    fn lfm3_partial_rotary_factor(&self) -> Result<f64, String> {
        let params = self.hparam("rope_parameters");
        let allow_flat = self.is_lfm3_moe_or_vl()
            || params
                .and_then(|p| p.get("theta"))
                .is_some_and(|v| !v.is_null());
        let nested_val = params
            .and_then(|p| p.get("full_attention"))
            .and_then(|p| p.get("partial_rotary_factor"))
            .filter(|v| !v.is_null());
        let flat_val = params
            .and_then(|p| p.get("partial_rotary_factor"))
            .or_else(|| self.hparam("partial_rotary_factor"))
            .filter(|v| !v.is_null());
        let Some(value) = nested_val.or_else(|| allow_flat.then_some(flat_val).flatten()) else {
            // on dense text checkpoints, only `rope_parameters.full_attention` is read;
            // the same key anywhere else would be ignored and the model converted as fully rotary
            return match flat_val {
                Some(v) => Err(format!(
                    "`partial_rotary_factor` ({v}) is set outside \
                     `rope_parameters.full_attention`, where it is read"
                )),
                None => Ok(1.0),
            };
        };
        let factor = value
            .as_f64()
            .filter(|f| f.is_finite() && *f > 0.0 && *f <= 1.0)
            .ok_or_else(|| format!("`partial_rotary_factor` is {value} (must be in (0, 1])"))?;
        if let Some(dim) = self.lfm3_head_dim()
            && Self::lfm3_rotated_pairs(factor, dim) == 0
        {
            return Err(format!(
                "`partial_rotary_factor` {factor:e} rotates no pair of a {dim}-wide head"
            ));
        }
        Ok(factor)
    }

    /// Head width, the sliding-window layout and the rotary width of an `lfm3` model.
    fn apply_lfm3_attention_metadata(&self, writer: &mut GgufWriter) {
        if let Some(dim) = self.lfm3_head_dim() {
            writer.add_u32("lfm3.attention.key_length", dim as u32);
            writer.add_u32("lfm3.attention.value_length", dim as u32);
            // the whole head counts as rotary; partial rotary is carried by the
            // `rope_freqs.weight` tensor instead (see `lfm3_rope_freqs`)
            writer.add_u32("lfm3.rope.dimension_count", dim as u32);
        }
        let sliding = self.lfm3_sliding_pattern();
        if !sliding.is_empty() {
            writer.add_bool_array("lfm3.attention.sliding_window_pattern", sliding.clone());
        }
        // `ensure_convertible` refuses a sliding layer without a usable window
        if sliding.iter().any(|&s| s)
            && let Some(window) = self.lfm3_sliding_window()
        {
            writer.add_u32("lfm3.attention.sliding_window", window);
        }
    }

    /// Routed expert metadata for an `lfm3` MoE model (`Lfm3MoeForCausalLM`).
    /// Emitted only when `num_experts` is present and > 0.
    fn apply_lfm3_moe_metadata(&self, writer: &mut GgufWriter) {
        let Some(n_exp) = self.expert_count().and_then(|n| u32::try_from(n).ok()) else {
            return;
        };
        writer.add_u32("lfm3.expert_count", n_exp);
        if let Some(n_used) = self
            .hparam("num_experts_per_tok")
            .or_else(|| self.hparam("n_activated_experts"))
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            writer.add_u32("lfm3.expert_used_count", n_used);
        }
        if let Some(lead_dense) = self
            .hparam("num_dense_layers")
            .or_else(|| self.hparam("first_k_dense_replace"))
            .or_else(|| self.hparam("leading_dense_block_count"))
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            writer.add_u32("lfm3.leading_dense_block_count", lead_dense);
        }
        if let Some(moe_ff) = self
            .hparam("moe_intermediate_size")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            writer.add_u32("lfm3.expert_feed_forward_length", moe_ff);
        }
        // Sigmoid expert gating function (ExpertGatingFuncType.SIGMOID = 2)
        writer.add_u32("lfm3.expert_gating_func", 2);
        if let Some(scale) = self
            .hparam("routed_scaling_factor")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .filter(|v| v.is_finite())
        {
            writer.add_f32("lfm3.expert_weights_scale", scale);
        }
        if let Some(norm_topk) = self.hparam("norm_topk_prob").and_then(Value::as_bool) {
            writer.add_bool("lfm3.expert_weights_norm", norm_topk);
        }
    }

    /// The `rope_freqs.weight` an `lfm3` GGUF carries: one factor per rotary pair of
    /// the global-attention layers. Partial rotary is expressed by dividing the
    /// trailing pairs' frequency by 1e30, so they never rotate. `None` when the
    /// config has no head width.
    pub(crate) fn lfm3_rope_freqs(&self) -> Option<Vec<f32>> {
        // bounded here too: this is reachable through the public `generated_tensors`
        let head_dim = self
            .lfm3_head_dim()
            .filter(|&d| d <= MAX_LFM3_HEAD_DIM && d.is_multiple_of(2))?;
        let partial = self.lfm3_partial_rotary_factor().ok()?;
        let pairs = head_dim / 2;
        let rotated = Self::lfm3_rotated_pairs(partial, head_dim);
        let mut freqs = vec![1.0f32; rotated];
        freqs.resize(pairs, 1e30);
        Some(freqs)
    }

    /// The SwiGLU width the LFM2 MLP really has. `intermediate_size` (`block_ff_dim`)
    /// is the pre-adjustment value: with `block_auto_adjust_ff_dim` the model takes
    /// two thirds of it, scales it and rounds up to a multiple of `block_multiple_of`
    /// (Transformers' `Lfm2MLP`, and llama.cpp's converter).
    pub(crate) fn lfm2_feed_forward_length(&self) -> Option<usize> {
        // `config.json` is untrusted (the streaming converter fetches it from a remote
        // repo): reject what cannot be a width, and never overflow on the way. `as`
        // saturates, so a negative, non-finite or huge float lands on 0 or `u64::MAX`,
        // and the final gate below rejects both.
        let base = match self.hparam("block_ff_dim") {
            // present but not a width: corrupt, so no fallback to `intermediate_size`
            Some(v) => v
                .as_u64()
                .or_else(|| v.as_f64().map(|f| f as u64))
                .and_then(|v| usize::try_from(v).ok())?,
            None => self.intermediate_size?,
        };
        // a key that is present (and not null) must be of its type: a wrong-typed value
        // falling back to the default would write a width the tensors do not have
        let auto_adjust = match self.hparam("block_auto_adjust_ff_dim") {
            Some(v) => v.as_bool()?,
            None => true,
        };
        let width = if auto_adjust {
            let mut width = base.checked_mul(2)? / 3;
            if let Some(multiplier) = self.hparam("block_ffn_dim_multiplier") {
                width = (multiplier.as_f64()? * width as f64) as usize;
            }
            let multiple_of = match self.hparam("block_multiple_of") {
                Some(v) => usize::try_from(v.as_u64()?).ok().filter(|&m| m > 0)?,
                None => 256,
            };
            width.div_ceil(multiple_of).checked_mul(multiple_of)?
        } else {
            base
        };
        // written as a `u32` metadata value; zero is not a width
        (width > 0 && u32::try_from(width).is_ok()).then_some(width)
    }

    /// The short-convolution kernel width (`conv_L_cache`), if the config has a usable one.
    /// cera's loader accepts 2 to 4 (every shipped LFM2 has 3), so anything else would
    /// convert into a file that is rejected only when it is loaded.
    pub(crate) fn lfm2_conv_l_cache(&self) -> Option<u32> {
        u32::try_from(self.hparam("conv_L_cache")?.as_u64()?)
            .ok()
            .filter(|l| (2..=4).contains(l))
    }

    /// [`Self::lfm2_conv_l_cache`] for `lfm3`, whose short conv is 7 taps wide: cera's
    /// CPU loader accepts 2 to 16 (`CPU_CONV_KERNEL_MAX`; the NPU path stops at 9).
    pub(crate) fn lfm3_conv_l_cache(&self) -> Option<u32> {
        u32::try_from(self.hparam("conv_L_cache")?.as_u64()?)
            .ok()
            .filter(|l| (2..=crate::model::lfm2::CPU_CONV_KERNEL_MAX as u32).contains(l))
    }

    /// A hyperparameter that has no typed field: the top level of `config.json`, else
    /// the nested `text_config` of a multimodal checkpoint. A JSON `null` counts as
    /// absent, as it does in Transformers configs.
    fn hparam(&self, key: &str) -> Option<&Value> {
        self.extra.get(key).filter(|v| !v.is_null()).or_else(|| {
            self.extra
                .get("text_config")?
                .get(key)
                .filter(|v| !v.is_null())
        })
    }

    /// An encoder checkpoint (`Lfm2BidirectionalModel`, `Lfm2BidirP2ForTokenClassification`):
    /// full attention and a centred short convolution instead of the causal ones.
    fn is_bidirectional(&self) -> bool {
        // the d1 decision models run their trunk bidirectionally too
        self.model_type
            .eq_ignore_ascii_case(crate::convert::d1::MODEL_TYPE)
            || self
                .architectures
                .first()
                .is_some_and(|a| a.to_ascii_lowercase().contains("bidir"))
    }

    fn is_lfm_family(&self) -> bool {
        matches!(self.gguf_architecture(), "lfm2" | "lfm2moe" | "lfm3")
    }

    /// For the LFM family, read the RMSNorm epsilon and RoPE base from the spellings its
    /// configs use, in the order llama.cpp's converter does: `norm_eps` /
    /// `block_norm_eps` over `rms_norm_eps`, and `rope_parameters` (Transformers 5, where
    /// the base moved out of the top level; the `full_attention` entry when the parameters
    /// are keyed by attention type, else the flat `rope_theta`) over the top-level `rope_theta`.
    /// Configs carry both spellings with equal values; a fine-tune that edits only one
    /// converts to the reference converter's base.
    fn resolve_alternate_spellings(&mut self) {
        // checked against llama.cpp's converter for the LFM family only
        if !self.is_lfm_family() {
            return;
        }
        // the first spelling the config has: a present but unreadable one is refused by
        // `ensure_convertible`, not skipped in favour of its sibling
        if let Some(eps) = ["norm_eps", "block_norm_eps"]
            .iter()
            .find_map(|k| self.hparam(k))
        {
            self.rms_norm_eps = eps.as_f64().map(|v| v as f32);
        }
        // likewise a `rope_theta` that is there but not a number: it replaces the top-level
        // value with `None`, which `ensure_convertible` refuses. `rope_parameters` that
        // simply carries no base (only a `rope_type`) leaves the top-level value alone.
        if let Some(theta) = self.rope_parameter_theta("full_attention") {
            self.rope_theta = theta.as_f64().map(|v| v as f32);
        }
        // For pure MoE checkpoints (`num_dense_layers == 0`) that omit `intermediate_size`,
        // fall back to `moe_intermediate_size`.
        if self.intermediate_size.is_none()
            && self.gguf_architecture() == "lfm3"
            && self.hparam("num_dense_layers").and_then(Value::as_u64) == Some(0)
            && let Some(moe_ff) = self
                .hparam("moe_intermediate_size")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
        {
            self.intermediate_size = Some(moe_ff);
        }
    }

    /// The `rope_theta` (or `theta`) nested under `layer_type` when `rope_parameters` is
    /// keyed by attention type, or `swa_rope_parameters` / flat VL keys, else the flat
    /// `rope_parameters.rope_theta`, as the JSON value it is written as.
    fn rope_parameter_theta(&self, layer_type: &str) -> Option<&Value> {
        fn get_theta(p: &Value) -> Option<&Value> {
            p.get("rope_theta")
                .or_else(|| p.get("theta"))
                .filter(|v| !v.is_null())
        }
        let params = self.hparam("rope_parameters");
        if let Some(v) = params.and_then(|p| p.get(layer_type)).and_then(get_theta) {
            return Some(v);
        }
        if layer_type == "sliding_attention" {
            if let Some(v) = self.hparam("swa_rope_parameters").and_then(get_theta) {
                return Some(v);
            }
            if let Some(v) = self.hparam("sliding_attention_rope_theta") {
                return Some(v);
            }
        } else if layer_type == "full_attention"
            && let Some(v) = self.hparam("full_attention_rope_theta")
        {
            return Some(v);
        }
        params.and_then(get_theta)
    }

    fn resolve_nested_text_config(&mut self) {
        if let Some(Value::Object(text_cfg)) = self.extra.get("text_config") {
            if let Some(mt) = text_cfg.get("model_type").and_then(|v| v.as_str()) {
                let lower = self.model_type.to_ascii_lowercase();
                // A d1-omni checkpoint's own type selects the decision head and towers;
                // its text model's `lfm2` type must not replace it.
                let is_d1 = self
                    .model_type
                    .eq_ignore_ascii_case(crate::convert::d1::MODEL_TYPE);
                if !is_d1
                    && (self.model_type.is_empty()
                        || lower.contains("vl")
                        || lower.contains("multimodal")
                        || lower.contains("vision")
                        || lower.contains("omni")
                        || lower.contains("audio"))
                {
                    self.model_type = mt.to_string();
                }
            }
            if self.hidden_size.is_none() {
                self.hidden_size = text_cfg
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.num_hidden_layers.is_none() {
                self.num_hidden_layers = text_cfg
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.num_attention_heads.is_none() {
                self.num_attention_heads = text_cfg
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.num_key_value_heads.is_none() {
                self.num_key_value_heads = text_cfg
                    .get("num_key_value_heads")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.intermediate_size.is_none() {
                self.intermediate_size = text_cfg
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.vocab_size.is_none() {
                self.vocab_size = text_cfg
                    .get("vocab_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.max_position_embeddings.is_none() {
                self.max_position_embeddings = text_cfg
                    .get("max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.rms_norm_eps.is_none() {
                self.rms_norm_eps = text_cfg
                    .get("rms_norm_eps")
                    .or_else(|| text_cfg.get("layer_norm_eps"))
                    .or_else(|| text_cfg.get("layer_norm_epsilon"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
            }
            if self.rope_theta.is_none() {
                self.rope_theta = text_cfg
                    .get("rope_theta")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
            }
            if self.head_dim.is_none() {
                self.head_dim = text_cfg
                    .get("head_dim")
                    .or_else(|| text_cfg.get("key_length"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
            }
            if self.bos_token_id.is_none() {
                self.bos_token_id = text_cfg.get("bos_token_id").and_then(parse_token_id_value);
            }
            if self.eos_token_id.is_none() {
                self.eos_token_id = text_cfg.get("eos_token_id").and_then(parse_token_id_value);
            }
            if self.pad_token_id.is_none() {
                self.pad_token_id = text_cfg.get("pad_token_id").and_then(parse_token_id_value);
            }
        }
    }

    /// Determine the canonical GGUF architecture name.
    pub fn gguf_architecture(&self) -> &str {
        match self.model_type.to_ascii_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => "llama",
            "qwen35" | "qwen3_5" | "qwen3.5" => "qwen35",
            "bailingmoe3" | "bailingmoe" | "bailingmoe2" | "bailing_moe_v3" | "bailing_moe"
            | "bailing" => "bailingmoe3",
            "qwen2" | "qwen" => "qwen2",
            "qwen3" => "qwen3",
            "mistral3" | "ministral3" | "ministral" => "mistral3",
            "mistral" => "llama", // Classic Mistral maps to llama architecture in GGUF
            "gemma" => "gemma",
            "gemma2" => "gemma2",
            "minicpm" | "minicpm3" => "minicpm",
            "nanbeige" | "nanbeige2" | "nanbeige4" => "nanbeige",
            "olmo" => "olmo",
            "olmo2" | "olmo3" => "olmo2",
            "olmoe" => "olmoe",
            "mamba2" | "falcon_mamba" => "mamba2",
            "mamba" => "mamba",
            "phi3" | "phi" | "phi-3" | "phi4" | "phi-4" => "phi3",
            "lfm" | "lfm2" | "lfm2.5" | "liquid" | "d1_omni" => "lfm2",
            "lfm2_moe" | "lfm2moe" => "lfm2moe",
            "lfm3" | "lfm3_moe" | "lfm3moe" | "lfm3_vl" | "lfm3vl" | "lfm3_audio" | "lfm3_omni" => {
                "lfm3"
            }
            "whisper" => "whisper",
            _ => {
                if let Some(first_arch) = self.architectures.first() {
                    let arch_lower = first_arch.to_ascii_lowercase();
                    if arch_lower.contains("whisper") {
                        "whisper"
                    } else if arch_lower.contains("lfm2moe") {
                        "lfm2moe"
                    } else if arch_lower.contains("lfm3") {
                        "lfm3"
                    } else if arch_lower.contains("lfm") || arch_lower.contains("liquid") {
                        "lfm2"
                    } else if arch_lower.contains("qwen35")
                        || arch_lower.contains("qwen3_5")
                        || arch_lower.contains("qwen3.5")
                    {
                        "qwen35"
                    } else if arch_lower.contains("qwen3") {
                        "qwen3"
                    } else if arch_lower.contains("bailing") {
                        "bailingmoe3"
                    } else if arch_lower.contains("qwen2") || arch_lower.contains("qwen") {
                        "qwen2"
                    } else if arch_lower.contains("nanbeige") {
                        "nanbeige"
                    } else if arch_lower.contains("mistral3") || arch_lower.contains("ministral") {
                        "mistral3"
                    } else if arch_lower.contains("minicpm") {
                        "minicpm"
                    } else if arch_lower.contains("olmoe") {
                        "olmoe"
                    } else if arch_lower.contains("olmo2") || arch_lower.contains("olmo3") {
                        "olmo2"
                    } else if arch_lower.contains("olmo") {
                        "olmo"
                    } else if arch_lower.contains("gemma2") {
                        "gemma2"
                    } else if arch_lower.contains("gemma") {
                        "gemma"
                    } else if arch_lower.contains("mamba2") {
                        "mamba2"
                    } else if arch_lower.contains("mamba") {
                        "mamba"
                    } else if !arch_lower.contains("moe")
                        && !arch_lower.contains("phi2")
                        && !arch_lower.contains("phi-2")
                        && !arch_lower.contains("phi_2")
                        && (arch_lower.contains("phi3")
                            || arch_lower.contains("phi-3")
                            || arch_lower.contains("phi4")
                            || arch_lower.contains("phi-4")
                            || arch_lower == "phi"
                            || arch_lower.starts_with("phi-")
                            || arch_lower.starts_with("phi_"))
                    {
                        "phi3"
                    } else {
                        "llama"
                    }
                } else {
                    "llama"
                }
            }
        }
    }

    /// Refuse a checkpoint this converter would write a wrong GGUF for.
    pub fn ensure_convertible(&self) -> Result<(), CeraError> {
        if self.gguf_architecture() == "lfm2moe" {
            // The routed experts must be stacked into per-layer `ffn_*_exps` tensors,
            // which this converter does not do; without that the file would carry
            // the dense architecture name over expert tensors it cannot load.
            return Err(CeraError::Backend(
                "converting LFM2-MoE checkpoints is not supported; convert them with \
                 llama.cpp's convert_hf_to_gguf.py"
                    .into(),
            ));
        }
        let arch = self.gguf_architecture();
        if arch == "lfm3" {
            let supported = [
                "lfm3forcausallm",
                "lfm3audiomodel",
                "lfm3audioforconditionalgeneration",
                "lfm3omnimodel",
                "lfm3omniforconditionalgeneration",
                "lfm3moeforcausallm",
                "lfm3vlforconditionalgeneration",
            ];
            if let Some(declared) = self.architectures.first()
                && !supported.contains(&declared.to_ascii_lowercase().as_str())
            {
                return Err(CeraError::Backend(format!(
                    "converting `{declared}` checkpoints is not supported; convert them with \
                     llama.cpp's convert_hf_to_gguf.py"
                )));
            }
        }
        if matches!(arch, "lfm2" | "lfm3") {
            let refuse = |what: String| {
                Err(CeraError::Backend(format!(
                    "cannot convert this {} config: {what}",
                    arch.to_ascii_uppercase()
                )))
            };
            // llama.cpp tells the two block kinds apart by a per-layer `head_count_kv`,
            // so a layout that cannot be read per block would write a file it loads wrongly
            let layout = match self.lfm2_layout(self.num_hidden_layers.unwrap_or(0)) {
                Ok(layout) => layout,
                Err(why) => return refuse(why),
            };
            // what the loaders require: a config that gives none of these a usable value
            // would convert into a file that is rejected only when it is loaded
            let shown = |v: Option<String>| v.unwrap_or_else(|| "missing".into());
            if self
                .hidden_size
                .is_none_or(|h| u32::try_from(h).is_err() || h == 0)
            {
                return refuse(format!(
                    "`hidden_size` is {} (must be 1..=u32::MAX)",
                    shown(self.hidden_size.map(|v| v.to_string()))
                ));
            }
            if self
                .vocab_size
                .is_none_or(|v| v == 0 || v > MAX_PADDED_VOCAB)
            {
                return refuse(format!(
                    "`vocab_size` is {} (must be 1..={MAX_PADDED_VOCAB})",
                    shown(self.vocab_size.map(|v| v.to_string()))
                ));
            }
            match arch {
                "lfm2" => {
                    if self.lfm2_feed_forward_length().is_none() {
                        return refuse(
                            "no usable `block_ff_dim` / `intermediate_size` (a present \
                             `block_ff_dim`, `block_multiple_of`, `block_ffn_dim_multiplier` or \
                             `block_auto_adjust_ff_dim` must be a valid number or boolean)"
                                .into(),
                        );
                    }
                    if self.lfm2_conv_l_cache().is_none() {
                        return refuse(format!(
                            "`conv_L_cache` is {}; it must be 2 to 4",
                            shown(self.hparam("conv_L_cache").map(|v| v.to_string()))
                        ));
                    }
                }
                "lfm3" => {
                    if self
                        .intermediate_size
                        .is_none_or(|n| n == 0 || u32::try_from(n).is_err())
                    {
                        return refuse(format!(
                            "`intermediate_size` is {} (must be 1..=u32::MAX)",
                            shown(self.intermediate_size.map(|v| v.to_string()))
                        ));
                    }
                    if let Some(why) = self.lfm3_unhonoured_member() {
                        return refuse(why);
                    }
                    if let Some(why) = self.lfm3_moe_problem() {
                        return refuse(why);
                    }
                    if self.lfm3_conv_l_cache().is_none() {
                        return refuse(format!(
                            "`conv_L_cache` is {}; it must be 2 to {}",
                            shown(self.hparam("conv_L_cache").map(|v| v.to_string())),
                            crate::model::lfm2::CPU_CONV_KERNEL_MAX
                        ));
                    }
                }
                _ => {}
            }
            // per-layer `i32`s: a count that does not fit would wrap, and 0 marks a conv layer
            for (key, heads) in [
                ("num_attention_heads", self.num_attention_heads),
                (
                    "num_key_value_heads",
                    self.num_key_value_heads.or(self.num_attention_heads),
                ),
            ] {
                if heads.is_none_or(|k| i32::try_from(k).ok().filter(|&k| k > 0).is_none()) {
                    return refuse(format!(
                        "`{key}` is {} (must be 1..=i32::MAX)",
                        shown(heads.map(|v| v.to_string()))
                    ));
                }
            }
            // cera's loader requires every attention layer's kv heads to divide the heads
            if let (Some(heads), Some(kv)) = (
                self.num_attention_heads,
                self.num_key_value_heads.or(self.num_attention_heads),
            ) && heads % kv != 0
            {
                return refuse(format!(
                    "`num_attention_heads` ({heads}) is not a multiple of \
                     `num_key_value_heads` ({kv})"
                ));
            }
            // LFM2 splits the hidden width evenly across the heads; LFM3 allows a decoupled
            // explicit `head_dim` (e.g. LFM3-VL has hidden_size 3072, 32 heads, head_dim 64),
            // requiring `hidden % heads == 0` only when `head_dim` is omitted.
            if let (Some(hidden), Some(heads)) = (self.hidden_size, self.num_attention_heads) {
                let bad_split = if arch == "lfm3" {
                    self.head_dim.is_none() && hidden % heads != 0
                } else {
                    hidden % heads != 0
                        || self
                            .head_dim
                            .is_some_and(|d| d.checked_mul(heads) != Some(hidden))
                };
                if bad_split {
                    return refuse(format!(
                        "`hidden_size` ({hidden}) must equal `num_attention_heads` ({heads}) \
                         times `head_dim`{}",
                        self.head_dim.map_or(String::new(), |d| format!(" ({d})"))
                    ));
                }
                let has_conv = layout.contains(&false);
                if hidden % 4 != 0 && has_conv {
                    return refuse(format!(
                        "`hidden_size` ({hidden}) is not a multiple of 4, which the \
                         short-convolution layers require"
                    ));
                }
            }
            // llama.cpp requires the context length and the norm epsilon, and defaults the
            // RoPE base to a value that is not cera's; a config that leaves any of them out,
            // or gives one that is not a usable number, converts into a file the two loaders
            // read differently or reject
            if self
                .max_position_embeddings
                .is_none_or(|n| n == 0 || u32::try_from(n).is_err())
            {
                return refuse(format!(
                    "`max_position_embeddings` is {} (must be 1..=u32::MAX)",
                    shown(self.max_position_embeddings.map(|v| v.to_string()))
                ));
            }
            for (key, value, also) in [
                (
                    "norm_eps",
                    self.rms_norm_eps,
                    "`block_norm_eps` or `rms_norm_eps`",
                ),
                (
                    "rope_theta",
                    self.rope_theta,
                    "`rope_parameters.rope_theta`",
                ),
            ] {
                if value.is_none_or(|v| !(v.is_finite() && v > 0.0)) {
                    return refuse(format!(
                        "`{key}` is {} (also read from {also}; must be positive and finite)",
                        shown(value.map(|v| v.to_string()))
                    ));
                }
            }
            if arch == "lfm3" {
                // the loader splits the hidden width over at most MAX_LFM3_HEADS heads and
                // rotates pairs, so an odd or enormous head width is a file it refuses
                // (or, for `rope_freqs.weight`, an allocation sized from untrusted input)
                if self
                    .num_attention_heads
                    .is_some_and(|h| h > crate::model::lfm2::MAX_LFM3_HEADS)
                {
                    return refuse(format!(
                        "`num_attention_heads` is {} (cera runs at most {})",
                        self.num_attention_heads.unwrap_or(0),
                        crate::model::lfm2::MAX_LFM3_HEADS
                    ));
                }
                if self.head_dim == Some(0) {
                    return refuse(format!(
                        "the head width (`head_dim`) is 0 (must be even and at most {MAX_LFM3_HEAD_DIM})"
                    ));
                }
                if let Some(dim) = self.lfm3_head_dim() {
                    if !dim.is_multiple_of(2) || dim > MAX_LFM3_HEAD_DIM {
                        return refuse(format!(
                            "the head width (`head_dim`) is {dim} (must be even and at most {MAX_LFM3_HEAD_DIM})"
                        ));
                    }
                    if let Some(heads) = self.num_attention_heads
                        && dim.checked_mul(heads).is_none_or(|q| !q.is_multiple_of(32))
                    {
                        return refuse(format!(
                            "`head_dim` ({dim}) times `num_attention_heads` ({heads}) is not a multiple of 32, \
                             which the quantized attention projections require"
                        ));
                    }
                }
                // a sliding layer needs the window llama.cpp reads, or the file loads nowhere
                if self.lfm3_has_sliding_layers() && self.lfm3_sliding_window().is_none() {
                    return refuse(format!(
                        "`sliding_window` is {} (must be 1..=u32::MAX when a layer slides)",
                        self.hparam("sliding_window")
                            .map_or("missing".to_string(), |v| v.to_string())
                    ));
                }
                if let Err(why) = self.lfm3_partial_rotary_factor() {
                    return refuse(why);
                }
                // the loader refuses a global base that is not finite and above 1
                if let Some(v) = self.rope_theta.filter(|v| !v.is_finite() || *v <= 1.0) {
                    return refuse(format!(
                        "`rope_theta` is {v} (must be a finite number above 1)"
                    ));
                }
                // the loader requires a finite sliding-layer base above 1 (and defaults a
                // missing one): an unreadable or out-of-range value must not be written
                if let Some(base) = self.rope_parameter_theta("sliding_attention")
                    && base
                        .as_f64()
                        // compared as the f32 the writer stores and the loader reads
                        .is_none_or(|b| {
                            let b = b as f32;
                            !(b.is_finite() && b > 1.0)
                        })
                {
                    return refuse(format!(
                        "`rope_parameters.sliding_attention.rope_theta` is {base} \
                         (must be a finite number above 1)"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Whether a checkpoint tensor is not part of the language-model GGUF being written.
    /// Multimodal `lfm3` checkpoints store vision/audio encoders, depth decoders, and
    /// adapters alongside the language model; only the language model is converted here.
    pub fn skips_tensor(&self, hf_name: &str) -> bool {
        self.gguf_architecture() == "lfm3"
            && crate::convert::tensor_map::is_multimodal_skipped_tensor(hf_name)
    }

    /// Tensors derived from the config (`rope_freqs.weight` for `lfm3`). Call
    /// [`Self::ensure_convertible`] first: a config it refuses may yield none.
    pub fn generated_tensors(&self) -> Vec<GeneratedTensor> {
        if self.gguf_architecture() != "lfm3" {
            return Vec::new();
        }
        self.lfm3_rope_freqs()
            .map(|data| GeneratedTensor {
                name: "rope_freqs.weight".into(),
                dims: vec![data.len() as u64],
                data,
            })
            .into_iter()
            .collect()
    }

    /// A value for `tokenizer.ggml.add_bos_token` that wins over the tokenizer's own.
    /// `lfm3`: the tokenizer does not insert BOS and the chat template adds it
    /// explicitly, so the flag is always off (llama.cpp's converter does the same).
    pub fn forced_add_bos_token(&self) -> Option<bool> {
        (self.gguf_architecture() == "lfm3").then_some(false)
    }

    /// Whether the vocabulary should be laid out the way llama.cpp's converter does
    /// (see [`crate::convert::tokenizer::VocabOptions`]). Verified for the LFM family only.
    pub fn uses_llama_cpp_vocab_layout(&self) -> bool {
        matches!(self.gguf_architecture(), "lfm2" | "lfm3")
    }

    /// Apply architecture metadata keys to a [`GgufWriter`]. Call [`Self::ensure_convertible`]
    /// first: for an LFM2 config it refuses what this would write incompletely (an
    /// unreadable layer layout leaves `head_count_kv` out) rather than erroring here.
    pub fn apply_to_gguf_writer(&self, writer: &mut GgufWriter, model_name: &str) {
        let arch = self.gguf_architecture();

        writer.add_string("general.architecture", arch);
        writer.add_string("general.name", model_name);

        if arch == "whisper" {
            writer.add_u32("whisper.sampling_rate", 16000);
            let d_model = self
                .extra
                .get("d_model")
                .and_then(|v| v.as_u64())
                .or(self.hidden_size.map(|h| h as u64))
                .unwrap_or(384) as u32;
            writer.add_u32("whisper.audio.embedding_length", d_model);
            writer.add_u32("whisper.text.embedding_length", d_model);

            let enc_heads = self
                .extra
                .get("encoder_attention_heads")
                .and_then(|v| v.as_u64())
                .or(self.num_attention_heads.map(|h| h as u64))
                .unwrap_or(6) as u32;
            let dec_heads = self
                .extra
                .get("decoder_attention_heads")
                .and_then(|v| v.as_u64())
                .or(self.num_attention_heads.map(|h| h as u64))
                .unwrap_or(6) as u32;
            writer.add_u32("whisper.audio.attention.head_count", enc_heads);
            writer.add_u32("whisper.text.attention.head_count", dec_heads);
            writer.add_u32("whisper.audio.head_count", enc_heads);
            writer.add_u32("whisper.text.head_count", dec_heads);

            let enc_layers = self
                .extra
                .get("encoder_layers")
                .and_then(|v| v.as_u64())
                .or(self.num_hidden_layers.map(|l| l as u64))
                .unwrap_or(4) as u32;
            let dec_layers = self
                .extra
                .get("decoder_layers")
                .and_then(|v| v.as_u64())
                .or(self.num_hidden_layers.map(|l| l as u64))
                .unwrap_or(4) as u32;
            writer.add_u32("whisper.audio.block_count", enc_layers);
            writer.add_u32("whisper.text.block_count", dec_layers);
            writer.add_u32("whisper.audio.layer_count", enc_layers);
            writer.add_u32("whisper.text.layer_count", dec_layers);

            let num_mel_bins = self
                .extra
                .get("num_mel_bins")
                .and_then(|v| v.as_u64())
                .unwrap_or(80) as u32;
            writer.add_u32("whisper.audio.num_mel_bins", num_mel_bins);
            writer.add_u32("whisper.audio.n_mel", num_mel_bins);

            let src_ctx = self
                .extra
                .get("max_source_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(1500) as u32;
            let tgt_ctx = self
                .extra
                .get("max_target_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(448) as u32;
            writer.add_u32("whisper.audio.context_length", src_ctx);
            writer.add_u32("whisper.text.context_length", tgt_ctx);
            writer.add_u32("whisper.audio.ctx", src_ctx);
            writer.add_u32("whisper.text.ctx", tgt_ctx);

            if let Some(v) = self.vocab_size {
                writer.add_u32("whisper.vocab_size", v as u32);
            }
            if let Some(bos) = self.bos_token_id {
                writer.add_u32("tokenizer.ggml.bos_token_id", bos);
                writer.add_u32("general.bos_token_id", bos);
            }
            if let Some(eos) = self.eos_token_id {
                writer.add_u32("tokenizer.ggml.eos_token_id", eos);
                writer.add_u32("general.eos_token_id", eos);
            }
            if let Some(pad) = self.pad_token_id {
                writer.add_u32("tokenizer.ggml.padding_token_id", pad);
                writer.add_u32("general.pad_token_id", pad);
            }
            return;
        }

        if let Some(v) = self.vocab_size {
            writer.add_u32(format!("{arch}.vocab_size"), v as u32);
        }
        if let Some(ctx) = self.max_position_embeddings {
            writer.add_u32(format!("{arch}.context_length"), ctx as u32);
        }
        if let Some(h) = self.hidden_size {
            writer.add_u32(format!("{arch}.embedding_length"), h as u32);
        }
        if let Some(layers) = self.num_hidden_layers {
            writer.add_u32(format!("{arch}.block_count"), layers as u32);
        }
        let layers = self.num_hidden_layers.unwrap_or(0);
        if let Some(heads) = self.num_attention_heads {
            writer.add_u32(format!("{arch}.attention.head_count"), heads as u32);
        }
        if let Some(kv_heads) = self.num_key_value_heads.or(self.num_attention_heads) {
            if matches!(arch, "lfm2" | "lfm3") {
                // Per layer, and 0 on the conv layers: that is how llama.cpp tells the
                // two block kinds apart, so a uniform array reads as all-attention there.
                // `ensure_convertible` refuses a layout it cannot read; a caller that
                // skipped it gets no key, which fails in the reader rather than loading
                // as a model whose every layer is attention
                if let (Some(attention), Ok(kv_heads)) =
                    (self.lfm2_attention_layers(layers), i32::try_from(kv_heads))
                {
                    let kv_array = attention
                        .iter()
                        .map(|&is_attention| if is_attention { kv_heads } else { 0 })
                        .collect();
                    writer.add_i32_array(format!("{arch}.attention.head_count_kv"), kv_array);
                }
            } else {
                writer.add_u32(format!("{arch}.attention.head_count_kv"), kv_heads as u32);
            }
        }
        if arch == "lfm3"
            && let Some(l_cache) = self.lfm3_conv_l_cache()
        {
            writer.add_u32(format!("{arch}.shortconv.l_cache"), l_cache);
        }
        if arch == "lfm2" {
            if let Some(ffn) = self.lfm2_feed_forward_length() {
                // `lfm2_feed_forward_length` bounds it to a `u32`
                writer.add_u32(format!("{arch}.feed_forward_length"), ffn as u32);
            }
            if let Some(l_cache) = self.lfm2_conv_l_cache() {
                writer.add_u32(format!("{arch}.shortconv.l_cache"), l_cache);
            }
        } else if let Some(ffn) = self.intermediate_size {
            writer.add_u32(format!("{arch}.feed_forward_length"), ffn as u32);
        }
        if arch == "lfm2" && self.is_bidirectional() {
            // `is_causal` is the key cera reads, `attention.causal` the one llama.cpp does
            writer.add_bool(format!("{arch}.is_causal"), false);
            writer.add_bool(format!("{arch}.attention.causal"), false);
        }
        if arch == "lfm3" {
            self.apply_lfm3_attention_metadata(writer);
            self.apply_lfm3_moe_metadata(writer);
        } else if let Some(dim) = self.head_dim {
            writer.add_u32(format!("{arch}.attention.key_length"), dim as u32);
        }
        if let Some(eps) = self.rms_norm_eps {
            writer.add_f32(format!("{arch}.attention.layer_norm_rms_epsilon"), eps);
        }
        if let Some(rope) = self.rope_theta {
            writer.add_f32(format!("{arch}.rope.freq_base"), rope);
        }
        if arch == "lfm3"
            && let Some(swa) = self
                .rope_parameter_theta("sliding_attention")
                .and_then(Value::as_f64)
        {
            writer.add_f32("lfm3.rope.freq_base_swa", swa as f32);
        }

        if let Some(bos) = self.bos_token_id {
            writer.add_u32("tokenizer.ggml.bos_token_id", bos);
            writer.add_u32("general.bos_token_id", bos);
        }
        if let Some(eos) = self.eos_token_id {
            writer.add_u32("tokenizer.ggml.eos_token_id", eos);
            writer.add_u32("general.eos_token_id", eos);
        }
        if let Some(pad) = self.pad_token_id {
            writer.add_u32("tokenizer.ggml.padding_token_id", pad);
            writer.add_u32("general.pad_token_id", pad);
        }

        if arch == "nanbeige" {
            let num_loops = self
                .extra
                .get("num_loops")
                .and_then(|v| {
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                })
                .unwrap_or(1) as u32;
            writer.add_u32("nanbeige.num_loops", num_loops);

            let skip_loop_final_norm = self
                .extra
                .get("skip_loop_final_norm")
                .and_then(|v| {
                    v.as_bool().or_else(|| match v {
                        Value::Number(n) => n.as_u64().map(|x| x != 0),
                        Value::String(s) => match s.trim() {
                            "true" | "1" => Some(true),
                            "false" | "0" => Some(false),
                            _ => None,
                        },
                        _ => None,
                    })
                })
                .unwrap_or(false);
            writer.add_bool("nanbeige.skip_loop_final_norm", skip_loop_final_norm);
        }

        if arch == "minicpm" || arch == "minicpm3" {
            if let Some(scale_emb) = self.extra.get("scale_emb").and_then(|v| v.as_f64()) {
                writer.add_f32("minicpm.embedding_scale", scale_emb as f32);
            }
            if let Some(scale_depth) = self.extra.get("scale_depth").and_then(|v| v.as_f64()) {
                let n_layers = self.num_hidden_layers.unwrap_or(1).max(1) as f64;
                let residual_scale = scale_depth / n_layers.sqrt();
                writer.add_f32("minicpm.residual_scale", residual_scale as f32);
            }
            if let (Some(h), Some(dim_base)) = (
                self.hidden_size,
                self.extra.get("dim_model_base").and_then(|v| v.as_f64()),
            ) && h > 0
                && dim_base > 0.0
            {
                let logit_scale = dim_base / (h as f64);
                writer.add_f32("minicpm.logit_scale", logit_scale as f32);
            }
        }

        if arch == "gemma2" {
            if let Some(cap) = self
                .extra
                .get("attn_logit_softcapping")
                .and_then(|v| v.as_f64())
            {
                writer.add_f32("gemma2.attn_logit_softcapping", cap as f32);
            }
            if let Some(cap) = self
                .extra
                .get("final_logit_softcapping")
                .and_then(|v| v.as_f64())
            {
                writer.add_f32("gemma2.final_logit_softcapping", cap as f32);
            }
        }

        // lfm3 writes its window only with a sliding layer, in `apply_lfm3_attention_metadata`
        if let Some(sw) = self
            .extra
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .and_then(|w| u32::try_from(w).ok())
            .filter(|&w| w > 0 && arch != "lfm3")
        {
            writer.add_u32(format!("{arch}.attention.sliding_window"), sw);
        }

        if arch == "qwen35" {
            if let Some(conv_kernel) = self
                .extra
                .get("linear_conv_kernel_dim")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.conv_kernel", conv_kernel as u32);
            }
            if let Some(inner_size) = self.extra.get("linear_inner_size").and_then(Value::as_u64) {
                writer.add_u32("qwen35.ssm.inner_size", inner_size as u32);
            }
            if let Some(state_size) = self
                .extra
                .get("linear_key_head_dim")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.state_size", state_size as u32);
            }
            if let Some(dt_rank) = self
                .extra
                .get("linear_num_value_heads")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.time_step_rank", dt_rank as u32);
            }
            if let Some(group_count) = self
                .extra
                .get("linear_num_key_heads")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.ssm.group_count", group_count as u32);
            }
            if let Some(full_attn_interval) = self
                .extra
                .get("full_attention_interval")
                .and_then(Value::as_u64)
            {
                writer.add_u32("qwen35.full_attention_interval", full_attn_interval as u32);
            }
        }

        if arch == "bailingmoe3" {
            if let Some(conv_kernel) = self
                .extra
                .get("short_conv_kernel_size")
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.ssm.conv_kernel", conv_kernel as u32);
            }
            if let Some(head_dim) = self.extra.get("head_dim").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.kda.head_dim", head_dim as u32);
            }
            if let Some(safe_gate) = self.extra.get("kda_safe_gate").and_then(Value::as_bool) {
                writer.add_bool("bailingmoe3.kda.safe_gate", safe_gate);
            }
            if let Some(lower_bound) = self.extra.get("kda_lower_bound").and_then(Value::as_f64) {
                writer.add_f32("bailingmoe3.kda.gate_lower_bound", lower_bound as f32);
            }
            if let Some(kv_lora_rank) = self.extra.get("kv_lora_rank").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.attention.kv_lora_rank", kv_lora_rank as u32);
            }
            if let Some(q_lora_rank) = self.extra.get("q_lora_rank").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.attention.q_lora_rank", q_lora_rank as u32);
            }
            let qk_rope_dim = self.extra.get("qk_rope_head_dim").and_then(Value::as_u64);
            if let Some(rope_dim) = qk_rope_dim {
                writer.add_u32("bailingmoe3.rope.dimension_count", rope_dim as u32);
            }
            let kv_lora_rank_val = self
                .extra
                .get("kv_lora_rank")
                .and_then(Value::as_u64)
                .unwrap_or(512);
            let qk_nope_dim_val = self
                .extra
                .get("qk_nope_head_dim")
                .and_then(Value::as_u64)
                .unwrap_or(128);
            let qk_rope_dim_val = qk_rope_dim.unwrap_or(64);
            writer.add_u32(
                "bailingmoe3.attention.key_length",
                (kv_lora_rank_val + qk_rope_dim_val) as u32,
            );
            writer.add_u32(
                "bailingmoe3.attention.key_length_mla",
                (qk_nope_dim_val + qk_rope_dim_val) as u32,
            );
            if let Some(v_head_dim) = self.extra.get("v_head_dim").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.attention.value_length_mla", v_head_dim as u32);
            }
            if let Some(n_exp) = self
                .extra
                .get("num_experts")
                .or_else(|| self.extra.get("n_routed_experts"))
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.expert_count", n_exp as u32);
            }
            if let Some(n_used) = self
                .extra
                .get("num_experts_per_tok")
                .or_else(|| self.extra.get("n_activated_experts"))
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.expert_used_count", n_used as u32);
            }
            if let Some(moe_ff) = self
                .extra
                .get("moe_intermediate_size")
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.expert_feed_forward_length", moe_ff as u32);
            }
            if let Some(shexp_ff) = self
                .extra
                .get("moe_shared_expert_intermediate_size")
                .and_then(Value::as_u64)
            {
                writer.add_u32(
                    "bailingmoe3.expert_shared_feed_forward_length",
                    shexp_ff as u32,
                );
            }
            if let Some(shexp_n) = self.extra.get("num_shared_experts").and_then(Value::as_u64) {
                writer.add_u32("bailingmoe3.expert_shared_count", shexp_n as u32);
            }
            if let Some(lead_dense) = self
                .extra
                .get("first_k_dense_replace")
                .or_else(|| self.extra.get("leading_dense_block_count"))
                .and_then(Value::as_u64)
            {
                writer.add_u32("bailingmoe3.leading_dense_block_count", lead_dense as u32);
            }
            if let Some(scale) = self
                .extra
                .get("routed_scaling_factor")
                .and_then(Value::as_f64)
            {
                writer.add_f32("bailingmoe3.expert_weights_scale", scale as f32);
            }
        }

        if arch == "mistral3" {
            let text_config = self.extra.get("text_config");
            let rope_scaling = self
                .extra
                .get("rope_parameters")
                .or_else(|| self.extra.get("rope_scaling"))
                .or_else(|| {
                    text_config
                        .and_then(|tc| tc.get("rope_parameters").or_else(|| tc.get("rope_scaling")))
                });
            if let Some(scale_type) = rope_scaling
                .and_then(|v| v.get("type").or_else(|| v.get("rope_type")))
                .and_then(Value::as_str)
            {
                writer.add_string("mistral3.rope.scaling.type", scale_type);
            }
            if let Some(factor) = rope_scaling
                .and_then(|v| v.get("factor"))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.factor", factor as f32);
            }
            if let Some(orig_ctx) = rope_scaling
                .and_then(|v| {
                    v.get("original_max_position_embeddings")
                        .or_else(|| v.get("original_context_length"))
                })
                .and_then(Value::as_u64)
                .filter(|&v| v > 0)
            {
                writer.add_u32(
                    "mistral3.rope.scaling.original_context_length",
                    orig_ctx as u32,
                );
                writer.add_u32("mistral3.attention.temperature_length", orig_ctx as u32);
            }
            if let Some(scale) = self
                .extra
                .get("attention_temperature_scale")
                .or_else(|| self.extra.get("attn_temp_scale"))
                .or_else(|| rope_scaling.and_then(|v| v.get("llama_4_scaling_beta")))
                .or_else(|| {
                    self.extra
                        .get("llama_4_scaling")
                        .and_then(|v| v.get("beta"))
                })
                .or_else(|| {
                    text_config.and_then(|tc| {
                        tc.get("attention_temperature_scale")
                            .or_else(|| tc.get("attn_temp_scale"))
                            .or_else(|| tc.get("llama_4_scaling").and_then(|v| v.get("beta")))
                    })
                })
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.attention.temperature_scale", scale as f32);
            }
            if let Some(log_mul) = self
                .extra
                .get("yarn_log_multiplier")
                .or_else(|| {
                    rope_scaling.and_then(|v| {
                        v.get("yarn_log_multiplier")
                            .or_else(|| v.get("mscale_all_dim"))
                    })
                })
                .or_else(|| text_config.and_then(|tc| tc.get("yarn_log_multiplier")))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v >= 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_log_multiplier", log_mul as f32);
            }
            if let Some(beta_fast) = rope_scaling
                .and_then(|v| {
                    v.get("yarn_beta_fast")
                        .or_else(|| v.get("beta_fast"))
                        .or_else(|| v.get("beta"))
                })
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_beta_fast", beta_fast as f32);
            }
            if let Some(beta_slow) = rope_scaling
                .and_then(|v| {
                    v.get("yarn_beta_slow")
                        .or_else(|| v.get("beta_slow"))
                        .or_else(|| v.get("alpha"))
                })
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_beta_slow", beta_slow as f32);
            }
            if let Some(ext_factor) = rope_scaling
                .and_then(|v| v.get("yarn_ext_factor").or_else(|| v.get("ext_factor")))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v >= 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.yarn_ext_factor", ext_factor as f32);
            }
            if let Some(attn_factor) = rope_scaling
                .and_then(|v| v.get("attn_factor").or_else(|| v.get("yarn_attn_factor")))
                .and_then(Value::as_f64)
                .filter(|&v| v.is_finite() && v > 0.0)
            {
                writer.add_f32("mistral3.rope.scaling.attn_factor", attn_factor as f32);
            }
        }

        // Token classification labels (e.g. pii-detect)
        if let Some(Value::Object(id2label)) = self.extra.get("id2label") {
            let mut label_pairs: Vec<(usize, String)> = Vec::new();
            for (k, v) in id2label {
                if let (Ok(idx), Some(s)) = (k.parse::<usize>(), v.as_str()) {
                    label_pairs.push((idx, s.to_string()));
                }
            }
            label_pairs.sort_by_key(|p| p.0);
            let labels: Vec<String> = label_pairs.into_iter().map(|p| p.1).collect();
            if !labels.is_empty() {
                writer.add_u32("token_classifier.num_labels", labels.len() as u32);
                writer.add_string_array("token_classifier.labels", labels);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::writer::MetadataValue;
    use serde_json::json;

    #[test]
    fn test_token_classifier_id2label_metadata() {
        let json_data = r#"{
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirP2ForTokenClassification"],
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "id2label": {
                "0": "O",
                "1": "B-NAME",
                "2": "I-NAME",
                "3": "S-NAME"
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "pii-detect");

        assert_eq!(
            writer.get_metadata("token_classifier.num_labels"),
            Some(&MetadataValue::Uint32(4))
        );
        let expected_labels = vec![
            "O".to_string(),
            "B-NAME".to_string(),
            "I-NAME".to_string(),
            "S-NAME".to_string(),
        ];
        assert_eq!(
            writer.get_metadata("token_classifier.labels"),
            Some(&MetadataValue::StringArray(expected_labels))
        );
    }

    #[test]
    fn liquid_model_type_and_architecture_still_map_to_lfm2() {
        let by_type = HfModelConfig::from_json_str(
            r#"{"model_type": "liquid", "hidden_size": 64, "num_hidden_layers": 2}"#,
        )
        .unwrap();
        assert_eq!(by_type.gguf_architecture(), "lfm2");

        let by_arch = HfModelConfig::from_json_str(
            r#"{"model_type": "unknown_type", "architectures": ["LiquidForCausalLM"],
                "hidden_size": 64, "num_hidden_layers": 2}"#,
        )
        .unwrap();
        assert_eq!(by_arch.gguf_architecture(), "lfm2");
    }

    #[test]
    fn d1_omni_keeps_its_model_type_when_text_config_names_one() {
        // The text model inside a d1-omni checkpoint says `lfm2`; hoisting that over
        // `d1_omni` would drop the d1 head, towers and folds and convert a causal LFM2.
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "d1_omni",
                "text_config": {"model_type": "lfm2", "hidden_size": 1024,
                                "num_hidden_layers": 16}}"#,
        )
        .unwrap();
        assert_eq!(cfg.model_type, crate::convert::d1::MODEL_TYPE);
        assert!(cfg.is_bidirectional());
    }

    #[test]
    fn test_nanbeige_config_mapping() {
        let json_data = r#"{
            "model_type": "nanbeige",
            "architectures": ["NanbeigeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_loops": 2,
            "skip_loop_final_norm": false
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "nanbeige");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "nanbeige-test");
        assert_eq!(
            writer.get_metadata("nanbeige.num_loops"),
            Some(&MetadataValue::Uint32(2))
        );
        assert_eq!(
            writer.get_metadata("nanbeige.skip_loop_final_norm"),
            Some(&MetadataValue::Bool(false))
        );
    }

    #[test]
    fn test_minicpm_config_mapping() {
        let json_data = r#"{
            "model_type": "minicpm",
            "architectures": ["MiniCPMForCausalLM"],
            "hidden_size": 2304,
            "num_hidden_layers": 36,
            "scale_emb": 12.0,
            "scale_depth": 1.4,
            "dim_model_base": 256.0
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "minicpm");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "minicpm-test");
        assert_eq!(
            writer.get_metadata("minicpm.embedding_scale"),
            Some(&MetadataValue::Float32(12.0))
        );
        let expected_residual = (1.4 / (36.0f64).sqrt()) as f32;
        assert_eq!(
            writer.get_metadata("minicpm.residual_scale"),
            Some(&MetadataValue::Float32(expected_residual))
        );
        let expected_logit = (256.0 / 2304.0) as f32;
        assert_eq!(
            writer.get_metadata("minicpm.logit_scale"),
            Some(&MetadataValue::Float32(expected_logit))
        );
    }

    #[test]
    fn test_gemma2_config_mapping() {
        let json_data = r#"{
            "model_type": "gemma2",
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "sliding_window": 4096
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "gemma2");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "gemma2-test");
        assert_eq!(
            writer.get_metadata("gemma2.attn_logit_softcapping"),
            Some(&MetadataValue::Float32(50.0))
        );
        assert_eq!(
            writer.get_metadata("gemma2.attention.sliding_window"),
            Some(&MetadataValue::Uint32(4096))
        );
    }

    #[test]
    fn test_olmoe_config_mapping() {
        let json_data = r#"{
            "model_type": "olmoe",
            "architectures": ["OlmoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "olmoe");
    }

    #[test]
    fn test_qwen3_architecture_detection() {
        let json_data = r#"{
            "model_type": "custom",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "qwen3");
    }

    #[test]
    fn test_sliding_window_generalized_emission() {
        let json_data = r#"{
            "model_type": "qwen2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "sliding_window": 32768
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "qwen2-test");
        assert_eq!(
            writer.get_metadata("qwen2.attention.sliding_window"),
            Some(&MetadataValue::Uint32(32768))
        );
    }

    #[test]
    fn test_qwen35_config_mapping() {
        let json_data = r#"{
            "model_type": "qwen35",
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "linear_conv_kernel_dim": 4,
            "linear_inner_size": 2048,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 16,
            "linear_num_key_heads": 8,
            "full_attention_interval": 4
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "qwen35");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "qwen35-test");
        assert_eq!(
            writer.get_metadata("qwen35.ssm.conv_kernel"),
            Some(&MetadataValue::Uint32(4))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.inner_size"),
            Some(&MetadataValue::Uint32(2048))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.state_size"),
            Some(&MetadataValue::Uint32(128))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.time_step_rank"),
            Some(&MetadataValue::Uint32(16))
        );
        assert_eq!(
            writer.get_metadata("qwen35.ssm.group_count"),
            Some(&MetadataValue::Uint32(8))
        );
        assert_eq!(
            writer.get_metadata("qwen35.full_attention_interval"),
            Some(&MetadataValue::Uint32(4))
        );
    }

    #[test]
    fn test_mistral3_config_mapping() {
        let json_data = r#"{
            "model_type": "mistral3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "attention_temperature_scale": 0.1,
            "yarn_log_multiplier": 0.0707
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "mistral3-test");

        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_scale"),
            Some(&MetadataValue::Float32(0.1))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_log_multiplier"),
            Some(&MetadataValue::Float32(0.0707))
        );
    }

    #[test]
    fn test_mistral3_nested_rope_scaling_mapping() {
        let json_data = r#"{
            "model_type": "mistral3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "attention_temperature_scale": 0.1,
            "rope_scaling": {
                "type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
                "yarn_log_multiplier": 0.0707,
                "yarn_beta_fast": 32.0,
                "yarn_beta_slow": 1.0
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "mistral3-nested-test");

        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.type"),
            Some(&MetadataValue::String("yarn".to_string()))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.factor"),
            Some(&MetadataValue::Float32(2.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.original_context_length"),
            Some(&MetadataValue::Uint32(32768))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_log_multiplier"),
            Some(&MetadataValue::Float32(0.0707))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_fast"),
            Some(&MetadataValue::Float32(32.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_slow"),
            Some(&MetadataValue::Float32(1.0))
        );
    }

    #[test]
    fn test_mistral3_rope_parameters_canonical_mapping() {
        let json_data = r#"{
            "model_type": "mistral3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 2.5,
                "original_max_position_embeddings": 32768,
                "mscale_all_dim": 0.0707,
                "llama_4_scaling_beta": 0.125,
                "beta": 32.0,
                "alpha": 1.0
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "mistral3-rope-params-test");

        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.type"),
            Some(&MetadataValue::String("yarn".to_string()))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.factor"),
            Some(&MetadataValue::Float32(2.5))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.original_context_length"),
            Some(&MetadataValue::Uint32(32768))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_log_multiplier"),
            Some(&MetadataValue::Float32(0.0707))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_scale"),
            Some(&MetadataValue::Float32(0.125))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_fast"),
            Some(&MetadataValue::Float32(32.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.yarn_beta_slow"),
            Some(&MetadataValue::Float32(1.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_length"),
            Some(&MetadataValue::Uint32(32768))
        );
    }

    #[test]
    fn test_mistral3_multimodal_text_config_mapping() {
        let json_data = r#"{
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "pixtral",
            "text_config": {
                "model_type": "mistral3",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "vocab_size": 32768,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 16384,
                    "mscale_all_dim": 0.1,
                    "llama_4_scaling_beta": 0.2,
                    "beta": 64.0,
                    "alpha": 2.0
                }
            }
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "mistral3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "pixtral-test");

        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.type"),
            Some(&MetadataValue::String("yarn".to_string()))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.factor"),
            Some(&MetadataValue::Float32(4.0))
        );
        assert_eq!(
            writer.get_metadata("mistral3.rope.scaling.original_context_length"),
            Some(&MetadataValue::Uint32(16384))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_length"),
            Some(&MetadataValue::Uint32(16384))
        );
        assert_eq!(
            writer.get_metadata("mistral3.attention.temperature_scale"),
            Some(&MetadataValue::Float32(0.2))
        );
    }

    #[test]
    fn test_phi3_hf_model_config() {
        let json_data = r#"{
            "model_type": "phi3",
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32064,
            "max_position_embeddings": 4096,
            "sliding_window": 2048,
            "rms_norm_eps": 1e-5
        }"#;

        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_eq!(cfg.gguf_architecture(), "phi3");

        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "phi3-mini-test");

        assert_eq!(
            writer.get_metadata("general.architecture"),
            Some(&MetadataValue::String("phi3".to_string()))
        );
        assert_eq!(
            writer.get_metadata("phi3.embedding_length"),
            Some(&MetadataValue::Uint32(3072))
        );
        assert_eq!(
            writer.get_metadata("phi3.feed_forward_length"),
            Some(&MetadataValue::Uint32(8192))
        );
        assert_eq!(
            writer.get_metadata("phi3.attention.sliding_window"),
            Some(&MetadataValue::Uint32(2048))
        );
    }

    #[test]
    fn test_phi2_not_classified_as_phi3() {
        let json_data = r#"{
            "architectures": ["PhiForCausalLM"],
            "model_type": "phi-msft",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_attention_heads": 32,
            "num_hidden_layers": 32
        }"#;
        let cfg = HfModelConfig::from_json_str(json_data).unwrap();
        assert_ne!(cfg.gguf_architecture(), "phi3");

        let json_phi2 = r#"{
            "architectures": ["Phi-2"],
            "model_type": "phi2",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_attention_heads": 32,
            "num_hidden_layers": 32
        }"#;
        let cfg_phi2 = HfModelConfig::from_json_str(json_phi2).unwrap();
        assert_ne!(cfg_phi2.gguf_architecture(), "phi3");
    }

    fn lfm2_config(extra: &str) -> HfModelConfig {
        HfModelConfig::from_json_str(&format!(
            r#"{{"model_type": "lfm2", "hidden_size": 1024, "num_hidden_layers": 4,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "intermediate_size": 6656, "vocab_size": 65536 {extra}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn lfm2_feed_forward_length_follows_the_auto_adjust_rule() {
        // LFM2.5-350M: int(2 * 6656 / 3) = 4437, rounded up to a multiple of 256
        let cfg = lfm2_config(
            r#", "block_ff_dim": 6656, "block_auto_adjust_ff_dim": true,
                 "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256"#,
        );
        assert_eq!(cfg.lfm2_feed_forward_length(), Some(4608));
        // the multiplier applies before the rounding
        let cfg = lfm2_config(
            r#", "block_ff_dim": 6656, "block_auto_adjust_ff_dim": true,
                 "block_ffn_dim_multiplier": 1.5, "block_multiple_of": 256"#,
        );
        assert_eq!(cfg.lfm2_feed_forward_length(), Some(6656));
        // without auto-adjust the stated width is the width
        let cfg = lfm2_config(r#", "block_ff_dim": 6656, "block_auto_adjust_ff_dim": false"#);
        assert_eq!(cfg.lfm2_feed_forward_length(), Some(6656));
        // a config with no block_* keys takes the Transformers defaults (adjust, 256)
        assert_eq!(lfm2_config("").lfm2_feed_forward_length(), Some(4608));
    }

    #[test]
    fn lfm2_attention_layers_come_from_layer_types_or_full_attn_idxs() {
        let cfg =
            lfm2_config(r#", "layer_types": ["conv", "full_attention", "conv", "full_attention"]"#);
        assert_eq!(
            cfg.lfm2_attention_layers(4),
            Some(vec![false, true, false, true])
        );
        // the first LFM2 checkpoints list the attention layers instead
        let cfg = lfm2_config(r#", "full_attn_idxs": [2]"#);
        assert_eq!(
            cfg.lfm2_attention_layers(4),
            Some(vec![false, false, true, false])
        );
        // neither: no guess
        assert_eq!(lfm2_config("").lfm2_attention_layers(4), None);
    }

    #[test]
    fn lfm2_metadata_marks_conv_layers_with_zero_kv_heads() {
        let cfg = lfm2_config(
            r#", "layer_types": ["conv", "full_attention", "conv", "full_attention"],
                 "conv_L_cache": 3, "norm_eps": 1e-5,
                 "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}"#,
        );
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(
            writer.get_metadata("lfm2.attention.head_count_kv"),
            Some(&MetadataValue::Int32Array(vec![0, 8, 0, 8]))
        );
        assert_eq!(
            writer.get_metadata("lfm2.shortconv.l_cache"),
            Some(&MetadataValue::Uint32(3))
        );
        assert_eq!(
            writer.get_metadata("lfm2.rope.freq_base"),
            Some(&MetadataValue::Float32(1_000_000.0))
        );
        assert_eq!(
            writer.get_metadata("lfm2.attention.layer_norm_rms_epsilon"),
            Some(&MetadataValue::Float32(1e-5))
        );
    }

    #[test]
    fn rope_base_and_epsilon_follow_the_reference_converters_spellings() {
        // llama.cpp's converter takes `rope_parameters` over a top-level `rope_theta`, and
        // `norm_eps` over `rms_norm_eps`: a fine-tune that edits only one must not
        // convert to a different base than the reference
        let cfg = lfm2_config(
            r#", "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
                 "rope_parameters": {"rope_theta": 5.0}, "norm_eps": 1e-5"#,
        );
        assert_eq!(cfg.rope_theta, Some(5.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-5));
        // the top-level spellings are used when the newer ones are absent
        let cfg = lfm2_config(r#", "rope_theta": 10000.0, "rms_norm_eps": 1e-6"#);
        assert_eq!(cfg.rope_theta, Some(10000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-6));
        // a `norm_eps` that is present but not a number is not skipped for its sibling
        let cfg = lfm2_config(r#", "norm_eps": "bad", "block_norm_eps": 0.5"#);
        assert_eq!(cfg.rms_norm_eps, None);
        // keyed by attention type, the full-attention base is the model's base
        let cfg = lfm2_config(
            r#", "block_norm_eps": 2e-5,
                 "rope_parameters": {"full_attention": {"rope_theta": 1000000.0},
                                     "sliding_attention": {"rope_theta": 10000.0}}"#,
        );
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        assert_eq!(cfg.rms_norm_eps, Some(2e-5));
    }

    #[test]
    fn lfm2_moe_is_its_own_architecture_and_is_not_convertible() {
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]}"#,
        )
        .unwrap();
        assert_eq!(cfg.gguf_architecture(), "lfm2moe");
        assert!(cfg.ensure_convertible().is_err());
        valid_lfm2_config().ensure_convertible().unwrap();
    }

    #[test]
    fn other_architectures_do_not_pick_up_the_lfm_spellings() {
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
                "norm_eps": 1e-5, "rope_parameters": {"rope_theta": 500000.0}}"#,
        )
        .unwrap();
        assert_eq!(cfg.rope_theta, None);
        assert_eq!(cfg.rms_norm_eps, None);
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(writer.get_metadata("llama.rope.freq_base"), None);
        assert_eq!(
            writer.get_metadata("llama.attention.layer_norm_rms_epsilon"),
            None
        );
    }

    #[test]
    fn lfm2_writer_leaves_head_count_kv_out_when_the_layout_is_unreadable() {
        // `ensure_convertible` refuses this config; a caller that skips it must not get a
        // uniform array, which llama.cpp would read as an all-attention model
        let mut writer = GgufWriter::new();
        lfm2_config("").apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(writer.get_metadata("lfm2.attention.head_count_kv"), None);
        // nor a head count that wraps in the per-layer `i32`
        let mut cfg = valid_lfm2();
        cfg["num_key_value_heads"] = json!(4294967298u64);
        let mut writer = GgufWriter::new();
        HfModelConfig::from_json_str(&cfg.to_string())
            .unwrap()
            .apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(writer.get_metadata("lfm2.attention.head_count_kv"), None);
    }

    /// A config `ensure_convertible` accepts; the tests below change one key at a time, so
    /// a refusal is for that key and not for something else the config lacks.
    fn valid_lfm2() -> serde_json::Value {
        json!({
            "model_type": "lfm2", "hidden_size": 64, "vocab_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "intermediate_size": 192, "conv_L_cache": 3,
            "max_position_embeddings": 128000, "norm_eps": 1e-5, "rope_theta": 1000000.0,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"]
        })
    }

    fn valid_lfm2_config() -> HfModelConfig {
        HfModelConfig::from_json_str(&valid_lfm2().to_string()).unwrap()
    }

    /// The refusal for `valid_lfm2()` with `key` set to `value` (`null` removes it),
    /// or `None` when the config is accepted.
    fn refusal(key: &str, value: serde_json::Value) -> Option<String> {
        refusal_in(valid_lfm2(), key, value)
    }

    /// [`refusal`] for any base config.
    fn refusal_in(
        mut cfg: serde_json::Value,
        key: &str,
        value: serde_json::Value,
    ) -> Option<String> {
        if value.is_null() {
            cfg.as_object_mut().unwrap().remove(key);
        } else {
            cfg[key] = value;
        }
        refusal_of(cfg)
    }

    fn refusal_of(cfg: serde_json::Value) -> Option<String> {
        HfModelConfig::from_json_str(&cfg.to_string())
            .unwrap()
            .ensure_convertible()
            .err()
            .map(|e| e.to_string())
    }

    #[test]
    fn lfm2_the_base_config_is_accepted() {
        valid_lfm2_config().ensure_convertible().unwrap();
    }

    #[test]
    fn lfm2_layout_is_refused_when_it_cannot_be_read_per_block() {
        let layout = |types: serde_json::Value| refusal("layer_types", types);
        // not one entry per block
        let err = layout(json!(["conv", "full_attention"])).unwrap_or_default();
        assert!(err.contains("2 entries"), "{err}");
        assert!(layout(json!(["conv", "conv", "conv", "conv", "full_attention"])).is_some());
        // an entry that is not a block kind, or a value that is not a list
        let err = layout(json!(["conv", "mamba", "conv", "full_attention"])).unwrap_or_default();
        assert!(err.contains("layer_types[1]"), "{err}");
        assert!(layout(json!([null, 1, "conv", true])).is_some());
        assert!(layout(json!("conv")).is_some());
        // no attention block at all
        assert!(layout(json!(["conv", "conv", "conv", "conv"])).is_some());
        // llama.cpp treats any non-conv type as attention
        assert_eq!(
            layout(json!([
                "conv",
                "sliding_attention",
                "conv",
                "full_attention"
            ])),
            None
        );
        // neither `layer_types` nor `full_attn_idxs`
        assert!(layout(json!(null)).unwrap_or_default().contains("neither"));
        // `full_attn_idxs` entries that are not in-range indices are not dropped
        let idxs = |idxs: serde_json::Value| {
            let mut cfg = valid_lfm2();
            cfg.as_object_mut().unwrap().remove("layer_types");
            cfg["full_attn_idxs"] = idxs;
            refusal_of(cfg)
        };
        assert_eq!(idxs(json!([1, 3])), None);
        for bad in [
            json!([]),
            json!([4]),
            json!([9, 10]),
            json!(["2"]),
            json!([1, -1]),
            json!([1, 2.5]),
        ] {
            assert!(idxs(bad.clone()).is_some(), "{bad}");
        }
        // JSON null is how Transformers writes an unset field
        let mut cfg = valid_lfm2();
        cfg["full_attn_idxs"] = json!([1]);
        cfg["layer_types"] = json!(null);
        assert_eq!(refusal_of(cfg), None);
    }

    #[test]
    fn lfm2_block_count_is_bounded_before_it_sizes_anything() {
        // untrusted `num_hidden_layers` must not drive an allocation or a quadratic scan;
        // `full_attn_idxs` is the layout whose size follows `num_hidden_layers`
        let sized = |layers: serde_json::Value| {
            let mut cfg = valid_lfm2();
            cfg.as_object_mut().unwrap().remove("layer_types");
            cfg["full_attn_idxs"] = json!([1]);
            cfg["num_hidden_layers"] = layers;
            refusal_of(cfg)
        };
        assert_eq!(sized(json!(4096)), None);
        for layers in [
            json!(4097),
            json!(4611686018427387904u64),
            json!(18446744073709551615u64),
        ] {
            let err = sized(layers.clone()).unwrap_or_default();
            assert!(err.contains("at most 4096"), "{layers}: {err}");
        }
        assert!(sized(json!(0)).unwrap_or_default().contains("zero"));
        assert!(
            refusal("num_hidden_layers", json!(null))
                .unwrap_or_default()
                .contains("zero")
        );
    }

    #[test]
    fn lfm2_head_counts_are_checked_for_width_and_divisibility() {
        for (key, value) in [
            ("num_attention_heads", json!(0)),
            ("num_attention_heads", json!(2147483648u64)),
            ("num_attention_heads", json!(4294967297u64)), // wraps to 1 under `as i32`
            ("num_attention_heads", json!(5)),             // 5 % 2 != 0
            ("num_key_value_heads", json!(0)),
            ("num_key_value_heads", json!(4294967296u64)),
            ("num_key_value_heads", json!(8)), // 4 % 8 != 0
        ] {
            assert!(refusal(key, value.clone()).is_some(), "{key} {value}");
        }
        // both wrap to 4 and 4 divides 4: only the width check can refuse this
        let mut cfg = valid_lfm2();
        cfg["num_attention_heads"] = json!(4294967300u64);
        cfg["num_key_value_heads"] = json!(4294967300u64);
        assert!(refusal_of(cfg).unwrap_or_default().contains("i32::MAX"));
        // missing kv heads default to the attention heads
        assert_eq!(refusal("num_key_value_heads", json!(null)), None);
        assert!(refusal("num_attention_heads", json!(null)).is_some());
    }

    #[test]
    fn lfm2_hidden_width_must_split_evenly_across_the_heads() {
        let err = refusal("hidden_size", json!(66)).unwrap_or_default();
        assert!(
            err.contains("`hidden_size` (66)") && err.contains("num_attention_heads"),
            "{err}"
        );
        // an explicit head_dim has to account for the whole width
        let err = refusal("head_dim", json!(7)).unwrap_or_default();
        assert!(err.contains("head_dim` (7)"), "{err}");
        assert_eq!(refusal("head_dim", json!(16)), None);
        for bad in [json!(0), json!(4294967296u64)] {
            assert!(refusal("head_dim", bad).is_some());
        }
        // the short convolution wants a width divisible by 4 (3 heads x 7 = 21)
        let mut cfg = valid_lfm2();
        cfg["hidden_size"] = json!(21);
        cfg["num_attention_heads"] = json!(3);
        cfg["num_key_value_heads"] = json!(3);
        assert!(
            refusal_of(cfg)
                .unwrap_or_default()
                .contains("multiple of 4")
        );
    }

    #[test]
    fn lfm2_without_what_the_loader_requires_is_refused() {
        // missing
        for key in [
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "norm_eps",
            "rope_theta",
            "conv_L_cache",
        ] {
            let err = refusal(key, json!(null)).unwrap_or_else(|| panic!("{key} is required"));
            assert!(err.contains("cannot convert"), "{key}: {err}");
        }
        // present but unusable
        for (key, value) in [
            ("hidden_size", json!(0)),
            ("hidden_size", json!(4294967296u64)),
            ("vocab_size", json!(0)),
            ("vocab_size", json!(1_000_001)),
            ("max_position_embeddings", json!(0)),
            ("max_position_embeddings", json!(4294967296u64)),
            ("norm_eps", json!(0)),
            ("norm_eps", json!(-1e-5)),
            ("norm_eps", json!(1e39)), // finite as f64, `inf` as f32
            ("rope_theta", json!(0)),
            ("rope_theta", json!(-5)),
            ("rope_theta", json!(1e39)),
            ("conv_L_cache", json!(1)),
            ("conv_L_cache", json!(5)),
            ("conv_L_cache", json!(4294967299u64)), // wraps to 3 under `as u32`
            ("conv_L_cache", json!("3")),
        ] {
            assert!(refusal(key, value.clone()).is_some(), "{key} {value}");
        }
        // and the edges that are fine
        for (key, value) in [
            ("vocab_size", json!(1_000_000)),
            ("max_position_embeddings", json!(1)),
            ("conv_L_cache", json!(2)),
            ("conv_L_cache", json!(4)),
        ] {
            assert_eq!(refusal(key, value.clone()), None, "{key} {value}");
        }
    }

    #[test]
    fn lfm2_refusals_show_the_value_they_found() {
        let err = refusal("vocab_size", json!(2_000_000)).unwrap_or_default();
        assert!(err.contains("`vocab_size` is 2000000"), "{err}");
        let err = refusal("vocab_size", json!(null)).unwrap_or_default();
        assert!(err.contains("`vocab_size` is missing"), "{err}");
        let err = refusal("rope_theta", json!(-5)).unwrap_or_default();
        assert!(
            err.contains("`rope_theta` is -5") && err.contains("rope_parameters"),
            "{err}"
        );
        let err = refusal("norm_eps", json!(null)).unwrap_or_default();
        assert!(
            err.contains("`norm_eps` is missing") && err.contains("block_norm_eps"),
            "{err}"
        );
        let err = refusal("conv_L_cache", json!(5)).unwrap_or_default();
        assert!(err.contains("`conv_L_cache` is 5"), "{err}");
    }

    #[test]
    fn lfm2_feed_forward_width_is_refused_when_a_present_key_is_corrupt() {
        // a present `block_ff_dim` that is not a width is corrupt, not absent
        for ff in [
            json!(0),
            json!("6656"),
            json!(-5),
            json!(1e30),
            json!(u64::MAX),
        ] {
            assert!(refusal("block_ff_dim", ff.clone()).is_some(), "{ff}");
        }
        // wrong-typed siblings must not fall back to their defaults
        for (key, value) in [
            ("block_ffn_dim_multiplier", json!("2.0")),
            ("block_ffn_dim_multiplier", json!(-1.0)),
            ("block_multiple_of", json!("64")),
            ("block_multiple_of", json!(-64)),
            ("block_multiple_of", json!(0)),
            ("block_auto_adjust_ff_dim", json!("false")),
        ] {
            assert!(refusal(key, value.clone()).is_some(), "{key} {value}");
        }
        // null is absent
        for key in ["block_multiple_of", "block_ffn_dim_multiplier"] {
            let mut cfg = valid_lfm2();
            cfg[key] = json!(null);
            assert_eq!(refusal_of(cfg), None, "{key}");
        }
    }

    #[test]
    fn lfm2_feed_forward_length_never_overflows() {
        let width = |extra: &str| lfm2_config(extra).lfm2_feed_forward_length();
        // 2^63 + 1024 doubles past `usize::MAX` to 2048: overflow, not a width
        assert_eq!(width(r#", "block_ff_dim": 9223372036854776832"#), None);
        // a multiplier that saturates the width, with a multiple that is not a power of two
        // (wrapping the round-up would land on a plausible 384)
        assert_eq!(
            width(
                r#", "block_ff_dim": 6656, "block_ffn_dim_multiplier": 1e30, "block_multiple_of": 1000"#
            ),
            None
        );
        assert_eq!(width(r#", "block_ff_dim": 9223372036854775808"#), None);
        // a float width is a width
        assert_eq!(width(r#", "block_ff_dim": 6656.0"#), Some(4608));
        // a width past `u32` has no metadata slot
        assert_eq!(
            width(r#", "block_ff_dim": 8589934592, "block_auto_adjust_ff_dim": false"#),
            None
        );
    }

    /// A config `ensure_convertible` accepts, with `extra` JSON members appended.
    fn lfm3_json(extra: &str) -> String {
        format!(
            r#"{{"model_type": "lfm3", "architectures": ["Lfm3ForCausalLM"],
                "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
                "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 300,
                "conv_L_cache": 7, "max_position_embeddings": 512, "norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "layer_types": ["conv", "full_attention", "conv"] {extra}}}"#
        )
    }

    fn lfm3_config(extra: &str) -> HfModelConfig {
        HfModelConfig::from_json_str(&lfm3_json(extra)).unwrap()
    }

    /// The same config as JSON, so a test can change one key.
    fn valid_lfm3() -> serde_json::Value {
        serde_json::from_str(&lfm3_json("")).unwrap()
    }

    fn lfm3_refusal(key: &str, value: serde_json::Value) -> Option<String> {
        refusal_in(valid_lfm3(), key, value)
    }

    /// The shared LFM validation covers lfm3 too: each of these used to convert into a
    /// file with a wrapped or missing value that only the loader rejected, or allocate
    /// from an untrusted width. Every refusal names the key it is about. (Rows can be refused
    /// by more than one guard; the head width and head count guards are pinned one by one in
    /// `lfm3_head_widths_are_bounded_and_even`.)
    #[test]
    fn lfm3_malformed_configs_are_refused() {
        assert_eq!(refusal_of(valid_lfm3()), None);
        for (key, value) in [
            ("conv_L_cache", json!(1)),
            ("conv_L_cache", json!(17)),
            ("conv_L_cache", json!(4294967303u64)),
            ("conv_L_cache", json!(null)),
            ("layer_types", json!(["conv", "full_attention"])),
            ("layer_types", json!(["conv", "mamba", "conv"])),
            ("layer_types", json!(null)),
            ("head_dim", json!(134217728)),
            ("head_dim", json!(15)),
            ("num_attention_heads", json!(0)),
            ("num_attention_heads", json!(256)),
            ("num_key_value_heads", json!(0)),
            ("num_key_value_heads", json!(3)),
            ("hidden_size", json!(63)),
            ("hidden_size", json!(4294967360u64)),
            ("intermediate_size", json!(null)),
            ("intermediate_size", json!(0)),
            ("intermediate_size", json!(4294967296u64)),
            ("max_position_embeddings", json!(null)),
            ("norm_eps", json!(null)),
            ("rope_theta", json!(1.0)),
            ("rope_theta", json!(0.5)),
            ("rope_theta", json!(null)),
        ] {
            let err = lfm3_refusal(key, value.clone());
            assert!(
                err.as_deref().is_some_and(|e| e.contains(key)),
                "{key} = {value}: {err:?}"
            );
        }
        for (key, value) in [
            ("conv_L_cache", json!(2)),
            ("conv_L_cache", json!(16)),
            ("head_dim", json!(16)),
        ] {
            assert_eq!(lfm3_refusal(key, value.clone()), None, "{key} = {value}");
        }
    }

    /// The head width feeds an allocation (`rope_freqs.weight`) and the rotary pairs: an
    /// odd one, one past the cap, or a head count past what cera runs is refused, whichever
    /// hidden size and head count produce it.
    #[test]
    fn lfm3_head_widths_are_bounded_and_even() {
        let with = |hidden: u64, heads: u64| {
            let mut cfg = valid_lfm3();
            cfg["hidden_size"] = json!(hidden);
            cfg["num_attention_heads"] = json!(heads);
            cfg["num_key_value_heads"] = json!(heads);
            refusal_of(cfg)
        };
        // 8192 / 1 = 8192 > 4096, 4294967292 / 1 is the OOM probe, 60 / 4 = 15 is odd
        for (hidden, heads, expect) in [
            (8192u64, 1u64, "head width"),
            (4294967292, 1, "head width"),
            (60, 4, "head width"),
            (512, 256, "num_attention_heads"),
        ] {
            let err = with(hidden, heads);
            assert!(
                err.as_deref().is_some_and(|e| e.contains(expect)),
                "hidden {hidden}, heads {heads}: {err:?}"
            );
        }
        // the public `generated_tensors` is bounded on its own: a refused config yields no
        // table, so an embedder that skips `ensure_convertible` allocates nothing
        let mut hostile = valid_lfm3();
        hostile["hidden_size"] = json!(134217728u64);
        hostile["num_attention_heads"] = json!(1);
        hostile["num_key_value_heads"] = json!(1);
        let cfg = HfModelConfig::from_json_str(&hostile.to_string()).unwrap();
        assert!(cfg.generated_tensors().is_empty());
        assert_eq!(with(4096, 1), None, "the widest accepted head");
        assert_eq!(with(2048, 128), None, "the most heads cera runs");
    }

    /// The sliding-layer base must be readable and finite above 1, as the loader requires;
    /// anything else used to be written as is (or as `inf`, or dropped so the loader
    /// quietly assumed 10000).
    #[test]
    fn lfm3_sliding_rope_base_must_be_usable() {
        let with = |base: serde_json::Value| {
            let mut cfg = valid_lfm3();
            cfg["rope_parameters"] = json!({
                "full_attention": {"rope_theta": 1000000.0},
                "sliding_attention": {"rope_theta": base},
            });
            refusal_of(cfg)
        };
        // 1.00000001 is above 1 as an f64 but is stored (and read back) as exactly 1.0
        for base in [
            json!(0.5),
            json!(0),
            json!(-3),
            json!(1e300),
            json!("x"),
            json!(1.00000001),
        ] {
            let err = with(base.clone());
            assert!(
                err.as_deref()
                    .is_some_and(|e| e.contains("sliding_attention.rope_theta")),
                "{base}: {err:?}"
            );
        }
        assert_eq!(with(json!(10000)), None);
        // the first f32 above 1 survives the round trip through the writer and the loader
        assert_eq!(with(json!(1.0000001)), None);
    }

    /// A sliding layer needs a window llama.cpp and cera can read: a missing, zero,
    /// negative or `u32`-overflowing one is refused (it used to be wrapped to a small
    /// window, or dropped so the file loaded nowhere), and a good one is written as is.
    #[test]
    fn lfm3_sliding_layers_need_a_usable_window() {
        let sliding = || {
            let mut cfg = valid_lfm3();
            cfg["layer_types"] = json!(["conv", "sliding_attention", "conv"]);
            cfg
        };
        for window in [
            json!(null),
            json!(0),
            json!(-5),
            json!(4294967304u64),
            json!("wide"),
        ] {
            let err = refusal_in(sliding(), "sliding_window", window.clone());
            assert!(
                err.as_deref()
                    .is_some_and(|e| e.contains("`sliding_window`")),
                "sliding_window = {window}: {err:?}"
            );
        }
        // a config without a sliding layer does not need one
        assert_eq!(lfm3_refusal("sliding_window", json!(null)), None);
        let mut cfg = sliding();
        cfg["sliding_window"] = json!(8);
        assert_eq!(refusal_of(cfg.clone()), None);
        let mut writer = GgufWriter::new();
        HfModelConfig::from_json_str(&cfg.to_string())
            .unwrap()
            .apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(
            writer.get_metadata("lfm3.attention.sliding_window"),
            Some(&MetadataValue::Uint32(8))
        );
    }

    /// `partial_rotary_factor` decides which rotary pairs the generated
    /// `rope_freqs.weight` freezes: a value outside (0, 1] or not a number would freeze
    /// every pair or none, and the text would still look plausible.
    #[test]
    fn lfm3_partial_rotary_factor_must_be_a_share() {
        let with = |factor: serde_json::Value| {
            let mut cfg = valid_lfm3();
            cfg["rope_parameters"] = json!({"full_attention": {
                "rope_theta": 1000000.0, "partial_rotary_factor": factor}});
            cfg
        };
        // a factor too small to rotate one pair of the 16-wide head freezes every pair
        for factor in [
            json!(0.001),
            json!(-0.5),
            json!(0),
            json!(1.5),
            json!("half"),
        ] {
            let err = refusal_of(with(factor.clone()));
            assert!(
                err.as_deref()
                    .is_some_and(|e| e.contains("partial_rotary_factor")),
                "partial_rotary_factor = {factor}: {err:?}"
            );
        }
        for factor in [json!(0.5), json!(1), json!(null)] {
            assert_eq!(refusal_of(with(factor.clone())), None, "{factor}");
        }
        // the same key anywhere but `rope_parameters.full_attention` would be ignored
        let top = |cfg: &mut serde_json::Value| cfg["partial_rotary_factor"] = json!(0.5);
        let flat = |cfg: &mut serde_json::Value| {
            cfg["rope_parameters"] = json!({"rope_theta": 1e6, "partial_rotary_factor": 0.5});
        };
        for place in [&top as &dyn Fn(&mut serde_json::Value), &flat] {
            let mut cfg = valid_lfm3();
            place(&mut cfg);
            let err = refusal_of(cfg);
            assert!(
                err.as_deref().is_some_and(|e| e.contains("outside")),
                "{err:?}"
            );
        }
    }

    /// Members the converter does not honour are refused rather than dropped.
    #[test]
    fn lfm3_unhonoured_members_are_refused() {
        let params = |v: serde_json::Value| {
            let mut cfg = valid_lfm3();
            cfg["rope_parameters"] = v;
            cfg
        };
        let rows = [
            ("conv_bias", {
                let mut c = valid_lfm3();
                c["conv_bias"] = json!(true);
                c
            }),
            (
                "sliding partial_rotary_factor",
                params(json!({"full_attention": {"rope_theta": 1e6},
                    "sliding_attention": {"rope_theta": 1e4, "partial_rotary_factor": 0.5}})),
            ),
            (
                "full yarn",
                params(
                    json!({"full_attention": {"rope_theta": 1e6, "rope_type": "yarn",
                    "factor": 4.0}}),
                ),
            ),
            (
                "sliding linear",
                params(json!({"full_attention": {"rope_theta": 1e6},
                    "sliding_attention": {"rope_theta": 1e4, "rope_type": "linear",
                    "factor": 8.0}})),
            ),
            (
                "a rope type alone",
                params(json!({"full_attention": {"rope_theta": 1e6, "rope_type": "yarn"}})),
            ),
            (
                "a sliding `type` alone",
                params(json!({"full_attention": {"rope_theta": 1e6},
                    "sliding_attention": {"rope_theta": 1e4, "type": "linear"}})),
            ),
            ("legacy rope_scaling yarn", {
                let mut c = valid_lfm3();
                c["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0});
                c
            }),
            ("legacy rope_scaling type alone", {
                let mut c = valid_lfm3();
                c["rope_scaling"] = json!({"type": "linear"});
                c
            }),
            ("text_config rope_scaling", {
                let mut c = valid_lfm3();
                c["text_config"] = json!({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}});
                c
            }),
            (
                "flat factor",
                params(json!({"rope_theta": 1e6, "factor": 2.0})),
            ),
        ];
        for (label, cfg) in rows {
            assert!(refusal_of(cfg).is_some(), "{label}: converted");
        }
        // what real checkpoints carry
        let mut ok = valid_lfm3();
        ok["conv_bias"] = json!(false);
        ok["rope_parameters"] = json!({
            "full_attention": {"rope_theta": 1e6, "rope_type": "default",
                "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default",
                "partial_rotary_factor": 1.0}});
        assert_eq!(refusal_of(ok), None);
    }

    /// A per-attention-type `rope_theta` beats a flat one for lfm2 as well (llama.cpp's
    /// converter reads `rope_parameters["full_attention"]` when it is there).
    #[test]
    fn lfm2_prefers_the_full_attention_rope_base_over_a_flat_one() {
        let mut cfg = valid_lfm2();
        cfg["rope_parameters"] = json!({
            "rope_theta": 7.0,
            "full_attention": {"rope_theta": 2000000.0},
        });
        let cfg = HfModelConfig::from_json_str(&cfg.to_string()).unwrap();
        // not the fixture's top-level base, and not the flat entry
        assert_eq!(cfg.rope_theta, Some(2_000_000.0));
    }

    #[test]
    fn lfm3_writes_the_window_only_when_a_layer_slides() {
        let mut writer = GgufWriter::new();
        lfm3_config(r#", "sliding_window": 8"#).apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(writer.get_metadata("lfm3.attention.sliding_window"), None);
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm3", "architectures": ["Lfm3ForCausalLM"], "sliding_window": 8,
                "layer_types": ["conv", "sliding_attention"], "hidden_size": 64,
                "num_attention_heads": 4}"#,
        )
        .unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(
            writer.get_metadata("lfm3.attention.sliding_window"),
            Some(&MetadataValue::Uint32(8))
        );
    }

    #[test]
    fn lfm3_reads_its_rope_bases_from_the_per_layer_entries() {
        // a stray top-level rope_theta does not beat rope_parameters; typed entries beat flat ones
        let cfg = lfm3_config(
            r#", "rope_parameters": {"rope_theta": 7.0,
                   "full_attention": {"rope_theta": 1000000.0},
                   "sliding_attention": {"rope_theta": 10000.0}}"#,
        );
        assert_eq!(cfg.rope_theta, Some(1_000_000.0));
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "t");
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base_swa"),
            Some(&MetadataValue::Float32(10_000.0))
        );
    }

    #[test]
    fn a_null_in_the_nested_text_config_is_absent_too() {
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm2_vl", "text_config": {"model_type": "lfm2",
                "layer_types": null, "full_attn_idxs": [1]}}"#,
        )
        .unwrap();
        assert_eq!(cfg.lfm2_attention_layers(2), Some(vec![false, true]));
    }

    #[test]
    fn lfm2_hyperparameters_are_read_from_a_nested_text_config() {
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm2_vl", "text_config": {"model_type": "lfm2",
                "layer_types": ["conv", "full_attention"], "block_ff_dim": 192,
                "block_multiple_of": 64, "conv_L_cache": 3}}"#,
        )
        .unwrap();
        assert_eq!(cfg.lfm2_attention_layers(2), Some(vec![false, true]));
        assert_eq!(cfg.lfm2_feed_forward_length(), Some(128));
    }

    #[test]
    fn token_ids_that_do_not_fit_a_u32_are_dropped_not_wrapped() {
        let cfg = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm2", "bos_token_id": 4294967296, "eos_token_id": [2, 3],
                "pad_token_id": "0"}"#,
        )
        .unwrap();
        assert_eq!(cfg.bos_token_id, None);
        assert_eq!(cfg.eos_token_id, Some(2));
        assert_eq!(cfg.pad_token_id, Some(0));
    }

    #[test]
    fn a_sliding_window_that_does_not_fit_a_u32_is_not_wrapped() {
        let write = |window: u64| {
            let cfg = lfm2_config(&format!(r#", "sliding_window": {window}"#));
            let mut writer = GgufWriter::new();
            cfg.apply_to_gguf_writer(&mut writer, "t");
            writer
                .get_metadata("lfm2.attention.sliding_window")
                .cloned()
        };
        assert_eq!(write(512), Some(MetadataValue::Uint32(512)));
        assert_eq!(write(4294967297), None);
    }

    #[test]
    fn bidirectional_lfm2_encoders_are_written_non_causal() {
        let causal_keys = |architecture: &str| {
            let mut cfg = valid_lfm2();
            cfg["architectures"] = json!([architecture]);
            let mut writer = GgufWriter::new();
            HfModelConfig::from_json_str(&cfg.to_string())
                .unwrap()
                .apply_to_gguf_writer(&mut writer, "t");
            (
                writer.get_metadata("lfm2.is_causal").cloned(),
                writer.get_metadata("lfm2.attention.causal").cloned(),
            )
        };
        // cera reads `is_causal`, llama.cpp reads `attention.causal`
        let non_causal = (
            Some(MetadataValue::Bool(false)),
            Some(MetadataValue::Bool(false)),
        );
        assert_eq!(causal_keys("Lfm2BidirectionalModel"), non_causal);
        assert_eq!(causal_keys("Lfm2BidirP2ForTokenClassification"), non_causal);
        // a decoder keeps the default
        assert_eq!(causal_keys("Lfm2ForCausalLM"), (None, None));
    }

    #[test]
    fn a_rope_base_that_is_present_but_unreadable_is_not_replaced_by_the_top_level_one() {
        // `valid_lfm2()` has a top-level `rope_theta`
        let err = refusal("rope_parameters", json!({"rope_theta": "invalid"})).unwrap_or_default();
        assert!(err.contains("`rope_theta` is missing"), "{err}");
        // `rope_parameters` that carries no base leaves the top-level one in force
        assert_eq!(
            refusal("rope_parameters", json!({"rope_type": "default"})),
            None
        );
        // null is absent
        assert_eq!(
            refusal("rope_parameters", json!({"rope_theta": null})),
            None
        );
    }

    #[test]
    fn lfm3_rope_freqs_follow_the_partial_rotary_factor() {
        let half = lfm3_config(
            r#", "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5}}"#,
        );
        let freqs = half.lfm3_rope_freqs().unwrap();
        assert_eq!(freqs.len(), 8); // head_dim 16 from 64 / 4
        assert_eq!(&freqs[..4], &[1.0; 4]);
        assert!(freqs[4..].iter().all(|&f| f == 1e30));
        // no factor: every pair rotates
        assert!(
            lfm3_config("")
                .lfm3_rope_freqs()
                .unwrap()
                .iter()
                .all(|&f| f == 1.0)
        );
    }

    #[test]
    fn a_head_dim_that_does_not_fill_the_hidden_width_is_refused_at_both_ends() {
        // the equality check also bounds `head_dim`: 0 and 2^32 never reach a `u32` cast
        for d in [json!(0), json!(4294967296u64), json!(17)] {
            assert!(refusal("head_dim", d.clone()).is_some(), "{d}");
        }
    }

    #[test]
    fn the_lfm_block_is_the_text_config_only_for_lfm3() {
        let audio3 = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm3_audio", "architectures": ["Lfm3AudioModel"],
                "lfm": {"hidden_size": 64, "num_hidden_layers": 2}}"#,
        )
        .unwrap();
        assert_eq!(audio3.hidden_size, Some(64));
        // an LFM2 audio checkpoint keeps its old behaviour
        let audio2 = HfModelConfig::from_json_str(
            r#"{"model_type": "lfm2_audio", "architectures": ["Lfm2AudioModel"],
                "lfm": {"hidden_size": 64}}"#,
        )
        .unwrap();
        assert_eq!(audio2.hidden_size, None);
    }

    #[test]
    fn lfm3_variants_the_converter_cannot_handle_are_refused() {
        assert!(lfm3_config("").ensure_convertible().is_ok());
        // Bare VL models without a CausalLM head remain refused
        {
            let arch = "Lfm3VlModel";
            let err = refusal_in(valid_lfm3(), "architectures", json!([arch]));
            assert!(
                err.as_deref()
                    .is_some_and(|e| e.contains("convert them with llama.cpp")),
                "{arch}: {err:?}"
            );
        }
        // Supported LFM3 architectures convert
        for arch in [
            "Lfm3ForCausalLM",
            "Lfm3AudioModel",
            "Lfm3OmniModel",
            "Lfm3MoeForCausalLM",
            "Lfm3VlForConditionalGeneration",
        ] {
            assert_eq!(
                refusal_in(valid_lfm3(), "architectures", json!([arch])),
                None,
                "{arch}"
            );
        }
    }

    fn valid_lfm3_moe() -> serde_json::Value {
        json!({
            "architectures": ["Lfm3MoeForCausalLM"],
            "model_type": "lfm3_moe",
            "hidden_size": 2048,
            "intermediate_size": 7168,
            "moe_intermediate_size": 1792,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 65536,
            "max_position_embeddings": 128000,
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "sliding_window": 4096,
            "layer_types": ["conv", "sliding_attention", "conv", "full_attention"],
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "num_dense_layers": 2,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": true,
            "rope_parameters": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
            "swa_rope_parameters": {"rope_theta": 100000.0}
        })
    }

    /// The loader refuses these routed-expert values (or needs them present), so the
    /// converter must refuse them too instead of writing a file only the loader rejects.
    #[test]
    fn lfm3_moe_metadata_the_loader_refuses_is_refused() {
        assert_eq!(refusal_of(valid_lfm3_moe()), None);
        for (key, value) in [
            ("num_experts", json!(4294967296u64)),
            ("num_experts", json!("32")),
            ("num_experts_per_tok", json!(null)),
            ("num_experts_per_tok", json!(0)),
            ("num_experts_per_tok", json!(33)),
            ("moe_intermediate_size", json!(null)),
            ("moe_intermediate_size", json!(0)),
            ("moe_intermediate_size", json!(1000)),
            ("moe_intermediate_size", json!(8192)),
            ("routed_scaling_factor", json!(2.5)),
            ("norm_topk_prob", json!(false)),
            ("norm_topk_prob", json!("yes")),
            ("num_dense_layers", json!(5)),
        ] {
            let why = refusal_in(valid_lfm3_moe(), key, value.clone())
                .unwrap_or_else(|| panic!("{key}={value} was accepted"));
            assert!(
                why.contains(key),
                "{key}={value}: refusal {why:?} names another key"
            );
        }
    }

    #[test]
    fn lfm3_moe_config_emits_expert_metadata_and_reads_swa_rope() {
        let cfg_json = json!({
            "architectures": ["Lfm3MoeForCausalLM"],
            "model_type": "lfm3_moe",
            "hidden_size": 2048,
            "intermediate_size": 7168,
            "moe_intermediate_size": 1792,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 65536,
            "max_position_embeddings": 128000,
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "sliding_window": 4096,
            "layer_types": ["conv", "sliding_attention", "conv", "full_attention"],
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "num_dense_layers": 2,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": true,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25
            },
            "swa_rope_parameters": {
                "rope_theta": 100000.0
            }
        });
        let cfg = HfModelConfig::from_json_str(&cfg_json.to_string()).unwrap();
        assert_eq!(cfg.gguf_architecture(), "lfm3");
        cfg.ensure_convertible().unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "lfm3-moe");
        assert_eq!(
            writer.get_metadata("lfm3.expert_count"),
            Some(&MetadataValue::Uint32(32))
        );
        assert_eq!(
            writer.get_metadata("lfm3.expert_used_count"),
            Some(&MetadataValue::Uint32(4))
        );
        assert_eq!(
            writer.get_metadata("lfm3.leading_dense_block_count"),
            Some(&MetadataValue::Uint32(2))
        );
        assert_eq!(
            writer.get_metadata("lfm3.expert_feed_forward_length"),
            Some(&MetadataValue::Uint32(1792))
        );
        assert_eq!(
            writer.get_metadata("lfm3.expert_gating_func"),
            Some(&MetadataValue::Uint32(2))
        );
        assert_eq!(
            writer.get_metadata("lfm3.expert_weights_scale"),
            Some(&MetadataValue::Float32(1.0))
        );
        assert_eq!(
            writer.get_metadata("lfm3.expert_weights_norm"),
            Some(&MetadataValue::Bool(true))
        );
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base"),
            Some(&MetadataValue::Float32(1_000_000.0))
        );
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base_swa"),
            Some(&MetadataValue::Float32(100_000.0))
        );
        let freqs = cfg.lfm3_rope_freqs().unwrap();
        assert_eq!(freqs.len(), 32);
        assert_eq!(&freqs[..8], &[1.0; 8]);
        assert!(freqs[8..].iter().all(|&f| f == 1e30));
    }

    #[test]
    fn lfm3_moe_theta_spelling_is_supported() {
        let cfg_json = json!({
            "architectures": ["Lfm3MoeForCausalLM"],
            "model_type": "lfm3_moe",
            "hidden_size": 64,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 300,
            "max_position_embeddings": 512,
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "sliding_window": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_dense_layers": 0,
            "rope_parameters": {
                "full_attention": {"theta": 1000000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"theta": 10000.0}
            }
        });
        let cfg = HfModelConfig::from_json_str(&cfg_json.to_string()).unwrap();
        assert_eq!(cfg.gguf_architecture(), "lfm3");
        assert_eq!(cfg.intermediate_size, Some(128));
        cfg.ensure_convertible().unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "dev-moe");
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base"),
            Some(&MetadataValue::Float32(1_000_000.0))
        );
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base_swa"),
            Some(&MetadataValue::Float32(10_000.0))
        );
        assert_eq!(
            writer.get_metadata("lfm3.leading_dense_block_count"),
            Some(&MetadataValue::Uint32(0))
        );
    }

    #[test]
    fn lfm3_vl_allows_decoupled_head_dim_and_flat_rope_keys() {
        let cfg_json = json!({
            "architectures": ["Lfm3VlForConditionalGeneration"],
            "model_type": "lfm3_vl",
            "text_config": {
                "model_type": "lfm3",
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 2,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "vocab_size": 65536,
                "max_position_embeddings": 32768,
                "norm_eps": 1e-5,
                "conv_L_cache": 3,
                "sliding_window": 1024,
                "layer_types": ["sliding_attention", "full_attention"],
                "full_attention_rope_theta": 1000000.0,
                "sliding_attention_rope_theta": 100000.0,
                "partial_rotary_factor": 0.5
            }
        });
        let cfg = HfModelConfig::from_json_str(&cfg_json.to_string()).unwrap();
        assert_eq!(cfg.gguf_architecture(), "lfm3");
        assert_eq!(cfg.lfm3_head_dim(), Some(64));
        cfg.ensure_convertible().unwrap();
        let mut writer = GgufWriter::new();
        cfg.apply_to_gguf_writer(&mut writer, "lfm3-vl");
        assert_eq!(
            writer.get_metadata("lfm3.attention.key_length"),
            Some(&MetadataValue::Uint32(64))
        );
        assert_eq!(
            writer.get_metadata("lfm3.attention.value_length"),
            Some(&MetadataValue::Uint32(64))
        );
        assert_eq!(
            writer.get_metadata("lfm3.rope.dimension_count"),
            Some(&MetadataValue::Uint32(64))
        );
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base"),
            Some(&MetadataValue::Float32(1_000_000.0))
        );
        assert_eq!(
            writer.get_metadata("lfm3.rope.freq_base_swa"),
            Some(&MetadataValue::Float32(100_000.0))
        );

        let mut bad_json = cfg_json;
        bad_json["text_config"]["num_attention_heads"] = json!(4);
        bad_json["text_config"]["num_key_value_heads"] = json!(2);
        bad_json["text_config"]["head_dim"] = json!(6);
        let bad_cfg = HfModelConfig::from_json_str(&bad_json.to_string()).unwrap();
        let err = bad_cfg.ensure_convertible().unwrap_err().to_string();
        assert!(
            err.contains("is not a multiple of 32"),
            "unexpected error: {err}"
        );
    }
}

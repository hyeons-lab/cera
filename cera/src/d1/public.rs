//! The public d1 files: the llama.cpp format Liquid publishes on Hugging Face.
//!
//! A cera-converted file ([`crate::convert::d1`]) is one GGUF with `d1.*` tensors and settings.
//! The public `d1-omni-600M` is two: the model (`general.architecture = lfm2`) and a `clip`
//! architecture mmproj that carries the vision and speech towers. The model file stores the
//! head blocks after the trunk's own (`blk.16`, `blk.17` beside the trunk's `blk.0` to `blk.15`,
//! all counted in `lfm2.block_count`), the head's scorer as `cls.*`, the question-type table as
//! `token_types.weight`, and the calibration as `lfm2.decision.temperature.*` settings.
//!
//! This module only names where things are; the loaders in [`super::head`],
//! [`super::vision`] and [`super::audio`] read the tensors.

use std::collections::HashMap;

use anyhow::{Result, ensure};

use crate::gguf::GgufFile;

/// `lfm2.decision.type` of the public non-causal d1-omni model.
pub const OMNI_TYPE: &str = "lfm2-d1-omni";

/// `lfm2.decision.type` of the public causal d1 model.
pub const CAUSAL_TYPE: &str = "lfm2-d1";

/// `lfm3.decision.type` of the public LFM3 d1 model (`d1-2B`).
pub const LFM3_CAUSAL_TYPE: &str = "lfm3-d1";

/// `lfm3.decision.type` of the public LFM3 d1-omni model (`d1-Omni-2B`).
pub const LFM3_OMNI_TYPE: &str = "lfm3-d1-omni";

const TYPE_KEY: &str = "lfm2.decision.type";
const LFM3_TYPE_KEY: &str = "lfm3.decision.type";
const BLOCK_COUNT_KEY: &str = "lfm2.decision.block_count";
const TEMPERATURE_PREFIX: &str = "lfm2.decision.temperature.";
const LFM3_TEMPERATURE_PREFIX: &str = "lfm3.decision.temperature.";
const OPTION_CODE_SCHEME_KEY: &str = "lfm2.decision.option_code_scheme";
const LFM3_OPTION_CODE_SCHEME_KEY: &str = "lfm3.decision.option_code_scheme";

/// Longest prompt of the public d1-omni model, in tokens (llama.cpp's `D1OMNI_MAX_TOKENS`).
pub(crate) const OMNI_MAX_LENGTH: usize = 16384;
/// Room an image request leaves its text (`D1OMNI_MAX_TOKENS_IMAGE`).
pub(crate) const OMNI_IMAGE_TEXT_LENGTH: usize = 896;
/// Room a speech request leaves its text (`D1OMNI_MAX_TOKENS_AUDIO`).
pub(crate) const OMNI_AUDIO_TEXT_LENGTH: usize = 15360;

/// The `lfm2.decision.type` or `lfm3.decision.type` of a public d1 file, `None` for any other file.
pub fn decision_type(gguf: &GgufFile) -> Option<&str> {
    gguf.get_str(TYPE_KEY)
        .or_else(|| gguf.get_str(LFM3_TYPE_KEY))
        .filter(|t| {
            matches!(
                *t,
                OMNI_TYPE | CAUSAL_TYPE | LFM3_CAUSAL_TYPE | LFM3_OMNI_TYPE
            )
        })
}

/// Whether `gguf` is any public d1 model file (`lfm2-d1`, `lfm2-d1-omni`, `lfm3-d1`, `lfm3-d1-omni`).
pub fn is_public_d1(gguf: &GgufFile) -> bool {
    decision_type(gguf).is_some()
}

/// Whether `gguf` is a public non-omni d1 model file (`lfm2-d1` or `lfm3-d1`).
pub fn is_causal(gguf: &GgufFile) -> bool {
    matches!(decision_type(gguf), Some(CAUSAL_TYPE | LFM3_CAUSAL_TYPE))
}

/// Whether `gguf` is a public d1-omni model file (`lfm2-d1-omni` or `lfm3-d1-omni`).
pub fn is_omni(gguf: &GgufFile) -> bool {
    matches!(decision_type(gguf), Some(OMNI_TYPE | LFM3_OMNI_TYPE))
}

/// Whether `gguf` is the head-based public `lfm2-d1-omni` file: an encoder trunk with a d1 head
/// and no language-model head. Unlike `lfm3-d1-omni`, which is prompt-based and scores logits.
#[cfg(feature = "hexagon")]
pub(crate) fn is_head_based_omni(gguf: &GgufFile) -> bool {
    decision_type(gguf) == Some(OMNI_TYPE)
}

/// Whether `gguf` is a public LFM3 d1 model file (`lfm3-d1` or `lfm3-d1-omni`).
pub fn is_lfm3_d1(gguf: &GgufFile) -> bool {
    matches!(decision_type(gguf), Some(LFM3_CAUSAL_TYPE | LFM3_OMNI_TYPE))
}

/// Whether `gguf` is a prompt-based public d1 model (`lfm2-d1`, `lfm3-d1`, or `lfm3-d1-omni`)
/// that scores next-token logits from `tokenizer.chat_template.systemone`.
pub fn is_prompt_d1(gguf: &GgufFile) -> bool {
    matches!(
        decision_type(gguf),
        Some(CAUSAL_TYPE | LFM3_CAUSAL_TYPE | LFM3_OMNI_TYPE)
    )
}

/// The option-code scheme (`prompt.py`'s `d1_codes`): 2 (`A`..`Z` then `AA`..`ZZ` past 26
/// options) for `lfm3` decision models unless overridden, 1 (`00`..`99` past 26 options)
/// otherwise.
pub fn option_code_scheme(gguf: &GgufFile) -> u32 {
    gguf.get_u32(LFM3_OPTION_CODE_SCHEME_KEY)
        .or_else(|| gguf.get_u32(OPTION_CODE_SCHEME_KEY))
        .unwrap_or(if is_lfm3_d1(gguf) { 2 } else { 1 })
}

/// How many blocks at the end of `{prefix}.block_count` are the d1 head's, not the trunk's:
/// `lfm2.decision.block_count` for a public `lfm2-d1-omni` file, 0 for everything else.
pub(crate) fn stored_head_blocks(gguf: &GgufFile, prefix: &str) -> usize {
    if prefix == "lfm2" && decision_type(gguf) == Some(OMNI_TYPE) {
        gguf.get_u32(BLOCK_COUNT_KEY).unwrap_or(0) as usize
    } else {
        0
    }
}

/// The calibration temperatures, keyed the way [`super::prompt::Question::temperature_key`]
/// spells a bucket. llama.cpp stores `choice.3_5` where the converted files say `choice:3-5`
/// (and `choice.11` for `choice:11+`); the per-type base `choice` is the same in both.
///
/// # Errors
///
/// Fails on a temperature that is not positive and finite.
pub(crate) fn temperatures(gguf: &GgufFile) -> Result<HashMap<String, f32>> {
    gguf.metadata
        .iter()
        .filter_map(|(key, value)| {
            let name = key
                .strip_prefix(TEMPERATURE_PREFIX)
                .or_else(|| key.strip_prefix(LFM3_TEMPERATURE_PREFIX))?;
            let value = match value {
                crate::gguf::GgufValue::F32(v) => *v,
                crate::gguf::GgufValue::F64(v) => *v as f32,
                _ => return None,
            };
            Some((bucket_key(name), value))
        })
        .map(|(key, value)| {
            ensure!(
                value.is_finite() && value > 0.0,
                "invalid d1 temperature {key} = {value}"
            );
            Ok((key, value))
        })
        .collect()
}

/// `choice.3_5` becomes `choice:3-5`; a name without a bucket is unchanged.
fn bucket_key(name: &str) -> String {
    match name.split_once('.') {
        Some((kind, "2")) => format!("{kind}:2"),
        Some((kind, "3_5")) => format!("{kind}:3-5"),
        Some((kind, "6_10")) => format!("{kind}:6-10"),
        Some((kind, "11")) => format!("{kind}:11+"),
        _ => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufBuilder;

    /// Only the head-based `lfm2-d1-omni` trunk is an encoder without a language-model head. The
    /// `lfm3-d1-omni` file is prompt-based and scores next-token logits, so its head must be planned.
    #[cfg(feature = "hexagon")]
    #[test]
    fn only_the_lfm2_omni_file_is_an_encoder_trunk() {
        let file = |key: &str, ty: &str| GgufBuilder::new().kv_str(key, ty).build();
        assert!(crate::model::is_encoder_trunk(&file(TYPE_KEY, OMNI_TYPE)));
        assert!(!crate::model::is_encoder_trunk(&file(
            LFM3_TYPE_KEY,
            LFM3_OMNI_TYPE
        )));
        assert!(!crate::model::is_encoder_trunk(&file(
            TYPE_KEY,
            CAUSAL_TYPE
        )));
        assert!(!crate::model::is_encoder_trunk(&file(
            LFM3_TYPE_KEY,
            LFM3_CAUSAL_TYPE
        )));
    }

    #[test]
    fn buckets_are_spelled_the_converted_way() {
        assert_eq!(bucket_key("choice"), "choice");
        assert_eq!(bucket_key("choice.2"), "choice:2");
        assert_eq!(bucket_key("score.3_5"), "score:3-5");
        assert_eq!(bucket_key("choice.6_10"), "choice:6-10");
        assert_eq!(bucket_key("choice.11"), "choice:11+");
        // a bucket this reader does not know is kept as the file spells it
        assert_eq!(bucket_key("choice.x"), "choice.x");
    }

    #[test]
    fn temperatures_read_the_public_keys() {
        let gguf = GgufBuilder::new()
            .kv_f32("lfm2.decision.temperature.choice", 1.0)
            .kv_f32("lfm2.decision.temperature.choice.3_5", 1.5)
            .kv_f32("lfm2.decision.temperature.noul.2", 1.25)
            .kv_u32("lfm2.decision.block_count", 2)
            .build();
        let t = temperatures(&gguf).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!(t["choice"], 1.0);
        assert_eq!(t["choice:3-5"], 1.5);
        assert_eq!(t["noul:2"], 1.25);
    }

    #[test]
    fn a_bad_temperature_is_refused() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let gguf = GgufBuilder::new()
                .kv_f32("lfm2.decision.temperature.choice", bad)
                .build();
            let err = temperatures(&gguf).expect_err("expected a refusal");
            assert!(err.to_string().contains("invalid d1 temperature"), "{err}");
        }
    }

    #[test]
    fn head_blocks_are_the_trunks_tail_only_in_a_public_omni_file() {
        let omni = GgufBuilder::new()
            .kv_str("lfm2.decision.type", OMNI_TYPE)
            .kv_u32("lfm2.decision.block_count", 2)
            .build();
        assert_eq!(stored_head_blocks(&omni, "lfm2"), 2);
        assert_eq!(stored_head_blocks(&omni, "lfm2moe"), 0);
        let plain = GgufBuilder::new()
            .kv_u32("lfm2.decision.block_count", 2)
            .build();
        assert_eq!(stored_head_blocks(&plain, "lfm2"), 0);
        let causal = GgufBuilder::new()
            .kv_str("lfm2.decision.type", CAUSAL_TYPE)
            .kv_u32("lfm2.decision.block_count", 2)
            .build();
        assert_eq!(stored_head_blocks(&causal, "lfm2"), 0);
        assert_eq!(decision_type(&causal), Some(CAUSAL_TYPE));
        assert!(is_omni(&omni) && !is_omni(&causal));
        assert!(is_causal(&causal) && !is_causal(&omni));
        assert!(is_public_d1(&omni) && is_public_d1(&causal) && !is_public_d1(&plain));
        assert!(!is_lfm3_d1(&omni) && !is_lfm3_d1(&causal));
        assert_eq!(option_code_scheme(&causal), 1);
    }

    #[test]
    fn lfm3_d1_and_d1_omni_keys_are_recognized() {
        let lfm3_d1 = GgufBuilder::new()
            .kv_str("lfm3.decision.type", LFM3_CAUSAL_TYPE)
            .kv_f32("lfm3.decision.temperature.choice", 1.337)
            .kv_f32("lfm3.decision.temperature.noul", 1.337)
            .kv_f32("lfm3.decision.temperature.score", 1.337)
            .build();
        assert_eq!(decision_type(&lfm3_d1), Some(LFM3_CAUSAL_TYPE));
        assert!(is_public_d1(&lfm3_d1));
        assert!(is_causal(&lfm3_d1));
        assert!(!is_omni(&lfm3_d1));
        assert!(is_lfm3_d1(&lfm3_d1));
        assert!(is_prompt_d1(&lfm3_d1));
        assert_eq!(stored_head_blocks(&lfm3_d1, "lfm3"), 0);
        assert_eq!(option_code_scheme(&lfm3_d1), 2);
        let t = temperatures(&lfm3_d1).unwrap();
        assert_eq!(t.len(), 3);
        assert!((t["choice"] - 1.337).abs() < 1e-6);

        let lfm3_omni = GgufBuilder::new()
            .kv_str("lfm3.decision.type", LFM3_OMNI_TYPE)
            .kv_f32("lfm3.decision.temperature.choice", 3.383)
            .kv_u32("lfm3.decision.option_code_scheme", 1)
            .build();
        assert_eq!(decision_type(&lfm3_omni), Some(LFM3_OMNI_TYPE));
        assert!(is_public_d1(&lfm3_omni));
        assert!(!is_causal(&lfm3_omni));
        assert!(is_omni(&lfm3_omni));
        assert!(is_lfm3_d1(&lfm3_omni));
        assert!(is_prompt_d1(&lfm3_omni));
        assert_eq!(stored_head_blocks(&lfm3_omni, "lfm3"), 0);
        assert_eq!(stored_head_blocks(&lfm3_omni, "lfm2"), 0);
        assert_eq!(option_code_scheme(&lfm3_omni), 1);
    }
}

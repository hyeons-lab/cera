use super::{LoadError, SamplingDefaults};
use cera::manifest::GenerationDefaults as CoreDefaults;

#[derive(Clone, uniffi::Enum)]
pub enum GenerationDefaults {
    Text {
        sampling: SamplingDefaults,
    },
    Audio {
        sampling: SamplingDefaults,
        number_of_decoding_threads: Option<u32>,
        audio_temperature: Option<f32>,
        audio_top_k: Option<u32>,
    },
    Other {
        raw_json: String,
    },
}

impl GenerationDefaults {
    pub(super) fn into_core(self) -> Result<CoreDefaults, LoadError> {
        Ok(match self {
            Self::Text { sampling } => CoreDefaults::Text {
                temperature: sampling.temperature,
                min_p: sampling.min_p,
                top_p: sampling.top_p,
                top_k: sampling.top_k,
                repetition_penalty: sampling.repetition_penalty,
            },
            Self::Audio {
                sampling,
                number_of_decoding_threads,
                audio_temperature,
                audio_top_k,
            } => CoreDefaults::Audio {
                number_of_decoding_threads,
                audio_temperature,
                audio_top_k,
                temperature: sampling.temperature,
                min_p: sampling.min_p,
                top_p: sampling.top_p,
                top_k: sampling.top_k,
                repetition_penalty: sampling.repetition_penalty,
            },
            Self::Other { raw_json } => CoreDefaults::Other {
                raw: serde_json::from_str(&raw_json).map_err(|error| LoadError::InvalidConfig {
                    field: "generation_defaults.raw_json".into(),
                    value: raw_json,
                    reason: "invalid_json".into(),
                    detail: error.to_string(),
                })?,
            },
        })
    }
}

impl From<&CoreDefaults> for GenerationDefaults {
    fn from(value: &CoreDefaults) -> Self {
        match value {
            CoreDefaults::Text {
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
            } => Self::Text {
                sampling: SamplingDefaults {
                    temperature: *temperature,
                    min_p: *min_p,
                    top_p: *top_p,
                    top_k: *top_k,
                    repetition_penalty: *repetition_penalty,
                },
            },
            CoreDefaults::Audio {
                number_of_decoding_threads,
                audio_temperature,
                audio_top_k,
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
            } => Self::Audio {
                sampling: SamplingDefaults {
                    temperature: *temperature,
                    min_p: *min_p,
                    top_p: *top_p,
                    top_k: *top_k,
                    repetition_penalty: *repetition_penalty,
                },
                number_of_decoding_threads: *number_of_decoding_threads,
                audio_temperature: *audio_temperature,
                audio_top_k: *audio_top_k,
            },
            CoreDefaults::Other { raw } => Self::Other {
                raw_json: raw.to_string(),
            },
        }
    }
}

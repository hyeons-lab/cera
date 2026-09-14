//! Explicit generative loading for native bindings.
//!
//! Existing CeraEngine constructors and Session errors retain their contracts.
use crate::{CeraEngine, EngineConfig, FfiError, Session, SessionConfig};
use cera as core;
use cera::ModelBytes;
use std::sync::{Arc, Mutex};

mod defaults;
pub use defaults::GenerationDefaults;

#[derive(Clone, uniffi::Record)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Clone, uniffi::Record)]
pub struct ModelParts {
    pub model: Vec<u8>,
    pub multimodal_projector: Option<Vec<u8>>,
    pub audio_decoder: Option<Vec<u8>>,
    pub audio_tokenizer: Option<Vec<u8>>,
    pub draft_model: Option<Vec<u8>>,
    pub inference_type: Option<String>,
    pub chat_template: Option<String>,
    pub generation_defaults: Option<crate::GenerationDefaults>,
}

#[derive(uniffi::Enum)]
pub enum ModelSource {
    BundleId {
        id: String,
        quant: String,
    },

    HuggingFace {
        spec: String,
        quant: Option<String>,
        strategy: Option<String>,
    },
    Bytes {
        bytes: Vec<u8>,
    },
    Parts {
        parts: ModelParts,
    },

    Path {
        path: String,
    },

    Files {
        files: ModelFiles,
    },
}

#[derive(Clone, uniffi::Record)]
pub struct ModelFiles {
    pub model: String,
    pub multimodal_projector: Option<String>,
    pub audio_decoder: Option<String>,
    pub audio_tokenizer: Option<String>,
    pub draft_model: Option<String>,
    pub extras: std::collections::HashMap<String, String>,
    pub inference_type: Option<String>,
    pub chat_template: Option<String>,
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum LoadError {
    #[error("expected {expected}, found {actual}: {architecture}")]
    KindMismatch {
        expected: String,
        actual: String,
        architecture: String,
    },
    #[error("unsupported architecture: {architecture}")]
    UnsupportedArchitecture { architecture: String },
    #[error("unsupported inference type: {inference_type}")]
    UnsupportedInferenceType { inference_type: String },
    #[error("{detail}")]
    Source { source_kind: String, detail: String },
    #[error("{detail}")]
    Assembly { backend: String, detail: String },
    #[error("{detail}")]
    InvalidConfig {
        field: String,
        value: String,
        reason: String,
        detail: String,
    },
    #[error("{detail}")]
    Engine { detail: String },
    #[error("loader already consumed")]
    Consumed,
}

impl From<core::LoadError> for LoadError {
    fn from(error: core::LoadError) -> Self {
        match error {
            core::LoadError::KindMismatch {
                expected,
                actual,
                architecture,
            } => Self::KindMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
                architecture,
            },
            core::LoadError::UnsupportedArchitecture { architecture } => {
                Self::UnsupportedArchitecture { architecture }
            }
            core::LoadError::Source {
                error: cera::CeraError::UnsupportedInferenceType(inference_type),
                ..
            }
            | core::LoadError::Assembly {
                error: cera::CeraError::UnsupportedInferenceType(inference_type),
                ..
            }
            | core::LoadError::Engine(cera::CeraError::UnsupportedInferenceType(inference_type)) => {
                Self::UnsupportedInferenceType { inference_type }
            }
            core::LoadError::Source { source_kind, error } => Self::Source {
                source_kind: source_kind.into(),
                detail: error.to_string(),
            },
            core::LoadError::Assembly { backend, error } => Self::Assembly {
                backend: format!("{backend:?}"),
                detail: error.to_string(),
            },
            other => Self::Engine {
                detail: other.to_string(),
            },
        }
    }
}

fn source(source: ModelSource) -> Result<core::ModelSource<'static>, LoadError> {
    let source = match source {
        ModelSource::BundleId { id, quant } => core::ModelSource::bundle_id(id, quant),

        ModelSource::HuggingFace {
            spec,
            quant,
            strategy,
        } => core::ModelSource::hugging_face(spec, quant.as_deref(), strategy.as_deref()),
        ModelSource::Bytes { bytes } => core::ModelSource::bytes(bytes),
        ModelSource::Parts { parts } => core::ModelSource::parts(ModelBytes {
            model: Arc::from(parts.model),
            multimodal_projector: parts.multimodal_projector.map(Arc::from),
            audio_decoder: parts.audio_decoder.map(Arc::from),
            audio_tokenizer: parts.audio_tokenizer.map(Arc::from),
            draft_model: parts.draft_model.map(Arc::from),
            inference_type: parts
                .inference_type
                .as_deref()
                .map(cera::manifest::InferenceType::parse_str),
            chat_template: parts.chat_template,
            generation_defaults: parts
                .generation_defaults
                .map(crate::GenerationDefaults::into_core)
                .transpose()?,
        }),

        ModelSource::Path { path } => core::ModelSource::path(path),

        ModelSource::Files { files } => core::ModelSource::files(cera::ModelFiles {
            model: files.model.into(),
            multimodal_projector: files.multimodal_projector.map(Into::into),
            audio_decoder: files.audio_decoder.map(Into::into),
            audio_tokenizer: files.audio_tokenizer.map(Into::into),
            draft_model: files.draft_model.map(Into::into),
            extras: files
                .extras
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            inference_type: files
                .inference_type
                .as_deref()
                .map(cera::manifest::InferenceType::parse_str),
            chat_template: files.chat_template,
        }),
    };
    Ok(source)
}

/// Synchronous, single-use model loader. Both build methods consume the source,
/// including on failure. Dispatch remote or expensive loads off the UI thread.
#[derive(uniffi::Object)]
pub struct ModelLoader {
    source: Mutex<Option<ModelSource>>,
    config: EngineConfig,
}

impl ModelLoader {
    fn take(&self) -> Result<core::ModelLoader<'static>, LoadError> {
        let source_input = self
            .source
            .lock()
            .map_err(|_| LoadError::Engine {
                detail: "loader source lock poisoned".into(),
            })?
            .take()
            .ok_or(LoadError::Consumed)?;
        let config: cera::EngineConfig =
            self.config
                .clone()
                .try_into()
                .map_err(|error: FfiError| LoadError::InvalidConfig {
                    field: "context_size".into(),
                    value: self.config.context_size.to_string(),
                    reason: "out_of_range".into(),
                    detail: error.to_string(),
                })?;
        Ok(core::ModelLoader::new(source(source_input)?).config(config))
    }
}

#[uniffi::export]
impl ModelLoader {
    /// Retain explicit source data and the existing production engine options.
    /// Construction does not load weights or contact a remote service.
    #[uniffi::constructor]
    pub fn new(source: ModelSource, config: EngineConfig) -> Self {
        Self {
            source: Mutex::new(Some(source)),
            config,
        }
    }

    /// Load a dynamic model handle. Generative loading is currently supported.
    pub fn build(&self) -> Result<Arc<ModelHandle>, LoadError> {
        Ok(Arc::new(ModelHandle {
            inner: self.take()?.build()?,
        }))
    }

    /// Load a generative model, reporting other known kinds before assembly.
    pub fn build_generative(&self) -> Result<Arc<GenerativeModel>, LoadError> {
        Ok(Arc::new(GenerativeModel {
            inner: self.take()?.build_generative()?,
        }))
    }
}

/// Dynamic loaded-model handle. Typed accessors share ownership.
#[derive(uniffi::Object)]
pub struct ModelHandle {
    inner: core::ModelHandle,
}

#[uniffi::export]
impl ModelHandle {
    /// Kind of the loaded model. A string allows future kinds without enum decoding.
    pub fn kind(&self) -> String {
        format!("{:?}", self.inner.kind())
    }

    /// Share a generative model if present; the result can outlive this handle.
    pub fn as_generative(&self) -> Option<Arc<GenerativeModel>> {
        self.inner
            .as_generative()
            .map(|inner| Arc::new(GenerativeModel { inner }))
    }
}

/// A shared generative engine. Creating handles never reloads the source or copies live KV.
#[derive(uniffi::Object)]
pub struct GenerativeModel {
    inner: core::GenerativeModel,
}

#[uniffi::export]
impl GenerativeModel {
    /// Access all retained engine operations through the already loaded engine.
    pub fn engine(&self) -> Arc<CeraEngine> {
        Arc::new(CeraEngine {
            inner: self.inner.engine(),
        })
    }

    /// Create an existing production Session with the caller's full configuration.
    /// Sessions retain their resources after all loader/model/engine handles close.
    /// Existing backend sharing restrictions and Session/FfiError behavior apply.
    pub fn create_session(&self, config: SessionConfig) -> Result<Arc<Session>, FfiError> {
        self.engine().new_session(config)
    }
}

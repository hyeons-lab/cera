use cera as core;
use cera::{
    EngineConfig as CoreEngineConfig, GenerateOpts, ModalitySink, ModelBytes,
    SessionConfig as CoreSessionConfig,
};
use std::sync::Arc;
#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

#[path = "defaults.rs"]
mod defaults;
pub use defaults::GenerationDefaults;

#[derive(Clone)]
#[cfg_attr(feature = "native", derive(uniffi::Record))]
#[cfg_attr(feature = "web", wasm_bindgen(getter_with_clone))]
pub struct LoadConfig {
    #[cfg(feature = "native")]
    #[uniffi(default = 4096)]
    pub context_size: u64,
    #[cfg(not(feature = "native"))]
    pub context_size: u32,
    pub backend: String,
    #[cfg_attr(feature = "native", uniffi(default = None))]
    pub draft_model: Option<String>,
    #[cfg_attr(feature = "native", uniffi(default = false))]
    pub gpu_depthformer: bool,
    #[cfg(feature = "native")]
    #[uniffi(default = None)]
    pub bundle_repo: Option<Arc<crate::remote::ProbeBundleRepo>>,
}

#[derive(Clone)]
#[cfg_attr(feature = "native", derive(uniffi::Record))]
#[cfg_attr(feature = "web", wasm_bindgen(getter_with_clone))]
#[cfg_attr(feature = "native", uniffi(name = "ProbeSamplingDefaults"))]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Clone)]
#[cfg_attr(feature = "native", derive(uniffi::Record))]
#[cfg_attr(feature = "web", wasm_bindgen(getter_with_clone))]
#[cfg_attr(feature = "native", uniffi(name = "ProbeModelParts"))]
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

#[cfg_attr(feature = "native", derive(uniffi::Enum))]
pub enum Source {
    #[cfg(feature = "native")]
    BundleId {
        id: String,
        quant: String,
    },
    #[cfg(feature = "native")]
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
    #[cfg(feature = "native")]
    Path {
        path: String,
    },
    #[cfg(feature = "native")]
    Files {
        files: ModelFiles,
    },
}

#[cfg(feature = "native")]
#[derive(Clone, uniffi::Record)]
#[cfg_attr(feature = "native", uniffi(name = "ProbeModelFiles"))]
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

#[derive(Debug, thiserror::Error)]
#[cfg_attr(feature = "native", derive(uniffi::Error))]
#[cfg_attr(feature = "native", uniffi(name = "ProbeLoadError"))]
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

pub fn engine_error(error: cera::CeraError) -> LoadError {
    LoadError::Engine {
        detail: error.to_string(),
    }
}

#[path = "context.rs"]
mod context;

// Mirrors the existing native FFI conversion; also exercised on wasm32.
pub fn native_context_size(context_size: u64) -> Result<usize, LoadError> {
    context::native_context_size(context_size).map_err(|detail| LoadError::InvalidConfig {
        field: "context_size".into(),
        value: context_size.to_string(),
        reason: "out_of_range".into(),
        detail,
    })
}

pub fn source(source: Source) -> Result<core::ModelSource<'static>, LoadError> {
    let source = match source {
        #[cfg(feature = "native")]
        Source::BundleId { id, quant } => core::ModelSource::bundle_id(id, quant),
        #[cfg(feature = "native")]
        Source::HuggingFace {
            spec,
            quant,
            strategy,
        } => core::ModelSource::hugging_face(spec, quant.as_deref(), strategy.as_deref()),
        Source::Bytes { bytes } => core::ModelSource::bytes(bytes),
        Source::Parts { parts } => core::ModelSource::parts(ModelBytes {
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
        #[cfg(feature = "native")]
        Source::Path { path } => core::ModelSource::path(path),
        #[cfg(feature = "native")]
        Source::Files { files } => core::ModelSource::files(cera::ModelFiles {
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

pub fn loader(input: Source, config: LoadConfig) -> Result<core::ModelLoader<'static>, LoadError> {
    #[cfg(feature = "native")]
    let context_size = native_context_size(config.context_size)?;
    #[cfg(not(feature = "native"))]
    let context_size = config.context_size as usize;
    let source = source(input)?;
    Ok(core::ModelLoader::new(source).config(CoreEngineConfig {
        context_size,
        backend: cera::BackendPreference::parse_str(&config.backend).map_err(|error| {
            LoadError::InvalidConfig {
                field: "backend".into(),
                value: config.backend.clone(),
                reason: "unknown_backend".into(),
                detail: error.to_string(),
            }
        })?,
        draft_model: config.draft_model.map(Into::into),
        gpu_depthformer: config.gpu_depthformer,
        #[cfg(feature = "native")]
        bundle_repo: config.bundle_repo.map(|repo| repo.inner.clone()),
    }))
}

#[derive(Clone)]
#[cfg_attr(feature = "native", derive(uniffi::Record))]
#[cfg_attr(feature = "web", wasm_bindgen(getter_with_clone))]
pub struct ModelInfo {
    #[cfg(feature = "native")]
    pub requested_context: u64,
    #[cfg(not(feature = "native"))]
    pub requested_context: u32,
    pub capacity: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub backend: String,
    pub draft_model: Option<String>,
    pub gpu_depthformer: bool,
    pub chat_template: Option<String>,
}

pub fn info(model: &core::GenerativeModel) -> ModelInfo {
    let defaults = model.default_generate_opts();
    ModelInfo {
        #[cfg(feature = "native")]
        requested_context: if model.config().context_size == usize::MAX {
            model.metadata().max_seq_len as u64
        } else {
            model.config().context_size as u64
        },
        #[cfg(not(feature = "native"))]
        requested_context: model.config().context_size as u32,
        capacity: model.model().config().max_seq_len as u32,
        temperature: defaults.temperature,
        top_p: defaults.top_p,
        top_k: defaults.top_k,
        min_p: defaults.min_p,
        repetition_penalty: defaults.repetition_penalty,
        backend: format!("{:?}", model.config().backend),
        draft_model: model
            .config()
            .draft_model
            .as_ref()
            .map(|p| p.display().to_string()),
        gpu_depthformer: model.config().gpu_depthformer,
        chat_template: model.manifest().chat_template.clone(),
    }
}

pub fn session(model: &core::GenerativeModel) -> Result<cera::Session, LoadError> {
    model
        .create_session(CoreSessionConfig {
            seed: Some(42),
            ..CoreSessionConfig::default()
        })
        .map_err(engine_error)
}

pub fn generate(session: &mut cera::Session) -> Result<Vec<u32>, LoadError> {
    #[derive(Default)]
    struct Sink(Vec<u32>, usize);
    impl ModalitySink for Sink {
        fn on_text_tokens(&mut self, tokens: &[u32]) {
            self.0.extend_from_slice(tokens);
        }
        fn on_done(&mut self, _: cera::session::FinishReason) {
            self.1 += 1;
        }
    }
    let mut sink = Sink::default();
    session
        .generate(
            &GenerateOpts {
                temperature: 0.0,
                max_tokens: 3,
                ignore_eos: true,
                ..GenerateOpts::default()
            },
            &mut sink,
        )
        .map_err(engine_error)?;
    if sink.1 != 1 {
        return Err(LoadError::Engine {
            detail: "expected one terminal callback".into(),
        });
    }
    Ok(sink.0)
}

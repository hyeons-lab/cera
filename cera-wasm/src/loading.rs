//! Explicit CPU WASM loading. Browser async and WebGPU entry points retain their existing homes.
use cera as core;
use cera::ModelBytes;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

mod defaults;
mod errors;
pub use defaults::GenerationDefaults;
use errors::{LoadError, error};

/// CPU loading options. A zero context uses the core engine's existing semantics.
#[derive(Clone)]
#[wasm_bindgen(getter_with_clone)]
pub struct LoadConfig {
    pub context_size: u32,
    pub backend: String,
    pub draft_model: Option<String>,
    pub gpu_depthformer: bool,
}

#[derive(Clone)]
#[wasm_bindgen(getter_with_clone)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

/// Multi-component model parts for WASM loading.
///
/// Constructed from JavaScript and passed by value into `ModelSource::parts`.
/// Property getters use `getter_with_clone` to conform to the TypeScript binding
/// contract in `tests/api_contracts/wasm_loading.json`; JS consumers should avoid
/// reading `.model` directly to prevent cloning the byte buffer into JS memory.
#[derive(Clone)]
#[wasm_bindgen(getter_with_clone)]
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

#[wasm_bindgen]
impl LoadConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(context_size: u32, backend: String) -> Self {
        Self {
            context_size,
            backend,
            draft_model: None,
            gpu_depthformer: false,
        }
    }
}

#[wasm_bindgen]
impl SamplingDefaults {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
        }
    }
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ModelParts {
    #[wasm_bindgen(constructor)]
    pub fn new(model: Vec<u8>) -> Self {
        Self {
            model,
            multimodal_projector: None,
            audio_decoder: None,
            audio_tokenizer: None,
            draft_model: None,
            inference_type: None,
            chat_template: None,
            generation_defaults: None,
        }
    }
}

#[wasm_bindgen]
pub struct ModelSource {
    inner: Source,
}

#[wasm_bindgen]
impl ModelSource {
    pub fn bytes(bytes: Vec<u8>) -> Self {
        Self {
            inner: Source::Bytes { bytes },
        }
    }
    pub fn parts(parts: ModelParts) -> Self {
        Self {
            inner: Source::Parts { parts },
        }
    }
}

enum Source {
    Bytes { bytes: Vec<u8> },
    Parts { parts: ModelParts },
}

fn source(source: Source) -> Result<core::ModelSource<'static>, LoadError> {
    Ok(match source {
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
    })
}

/// Single-use synchronous loader. Both build methods consume the source even on failure.
#[wasm_bindgen]
pub struct ModelLoader {
    source: Option<Source>,
    config: LoadConfig,
}

impl ModelLoader {
    fn take(&mut self) -> Result<core::ModelLoader<'static>, LoadError> {
        let input = self.source.take().ok_or(LoadError::Consumed)?;
        let source = source(input)?;
        // Optional core features can add fields; match the retained constructors.
        #[allow(clippy::needless_update)]
        let config = cera::EngineConfig {
            context_size: self.config.context_size as usize,
            backend: cera::BackendPreference::parse_str(&self.config.backend).map_err(|err| {
                LoadError::InvalidConfig {
                    field: "backend".into(),
                    value: self.config.backend.clone(),
                    reason: "unknown_backend".into(),
                    detail: err.to_string(),
                }
            })?,
            draft_model: self.config.draft_model.clone().map(Into::into),
            gpu_depthformer: self.config.gpu_depthformer,
            ..Default::default()
        };
        Ok(core::ModelLoader::new(source).config(config))
    }
}

#[wasm_bindgen]
impl ModelLoader {
    #[wasm_bindgen(constructor)]
    pub fn new(source: ModelSource, config: LoadConfig) -> Self {
        Self {
            source: Some(source.inner),
            config,
        }
    }
    pub fn build(&mut self) -> Result<ModelHandle, JsValue> {
        self.take()
            .and_then(|loader| loader.build().map_err(Into::into))
            .map(|inner| ModelHandle { inner })
            .map_err(error)
    }
    #[wasm_bindgen(js_name = buildGenerative)]
    pub fn build_generative(&mut self) -> Result<GenerativeModel, JsValue> {
        self.take()
            .and_then(|loader| loader.build_generative().map_err(Into::into))
            .map(|inner| GenerativeModel { inner })
            .map_err(error)
    }
}

/// Dynamic loaded model; its typed accessor shares ownership.
#[wasm_bindgen]
pub struct ModelHandle {
    inner: core::ModelHandle,
}

#[wasm_bindgen]
impl ModelHandle {
    pub fn kind(&self) -> String {
        format!("{:?}", self.inner.kind())
    }
    #[wasm_bindgen(js_name = asGenerative)]
    pub fn as_generative(&self) -> Option<GenerativeModel> {
        self.inner
            .as_generative()
            .map(|inner| GenerativeModel { inner })
    }
}

/// A loaded generative engine with shared weights and independent CPU sessions.
#[wasm_bindgen]
pub struct GenerativeModel {
    inner: core::GenerativeModel,
}

#[wasm_bindgen]
impl GenerativeModel {
    /// Share the already loaded engine, including its tokenizer and cache.
    pub fn engine(&self) -> crate::CeraEngine {
        crate::CeraEngine {
            inner: self.inner.engine(),
        }
    }
    /// Create a production Session that can outlive all loading and engine handles.
    #[wasm_bindgen(js_name = createSession)]
    pub fn create_session(&self, config: &crate::SessionConfig) -> Result<crate::Session, JsError> {
        self.engine().new_session(config)
    }
}

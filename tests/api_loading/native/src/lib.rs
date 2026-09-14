mod remote;
#[path = "../../shared.rs"]
mod shared;
use cera as core;
pub use cera_ffi::{BackendPreference, EngineConfig, SessionConfig};
pub use remote::{ProbeBundleRepo, ProbeDownloadProgressSink};
pub use shared::{
    GenerationDefaults, LoadConfig, LoadError, ModelFiles, ModelInfo, ModelParts, SamplingDefaults,
    Source,
};
use std::sync::{Arc, Mutex};

uniffi::setup_scaffolding!();

#[uniffi::export]
pub fn session_config_for_probe(
    session: Arc<cera_ffi::Session>,
) -> Result<SessionConfig, cera_ffi::FfiError> {
    session.config_for_loading_probe()
}

#[uniffi::export]
pub fn engines_share_for_probe(
    first: Arc<cera_ffi::CeraEngine>,
    second: Arc<cera_ffi::CeraEngine>,
) -> bool {
    first.shares_wrapper_for_probe(&second)
}

#[derive(uniffi::Object)]
pub struct ProbeModelLoader {
    source: Mutex<Option<Source>>,
    config: Configuration,
}

enum Configuration {
    Legacy(LoadConfig),
    Typed(EngineConfig),
}

impl ProbeModelLoader {
    fn take(&self) -> Result<core::ModelLoader<'static>, LoadError> {
        let source = self
            .source
            .lock()
            .unwrap()
            .take()
            .ok_or(LoadError::Consumed)?;
        match &self.config {
            Configuration::Typed(config) => {
                let options: cera::EngineConfig =
                    config
                        .clone()
                        .try_into()
                        .map_err(|error: cera_ffi::FfiError| LoadError::InvalidConfig {
                            field: "context_size".into(),
                            value: config.context_size.to_string(),
                            reason: "out_of_range".into(),
                            detail: error.to_string(),
                        })?;
                Ok(core::ModelLoader::new(shared::source(source)?).config(options))
            }
            Configuration::Legacy(config) => shared::loader(source, config.clone()),
        }
    }
}

#[uniffi::export]
impl ProbeModelLoader {
    #[uniffi::constructor]
    pub fn new(source: Source, config: LoadConfig) -> Self {
        Self {
            source: Mutex::new(Some(source)),
            config: Configuration::Legacy(config),
        }
    }
    pub fn build(&self) -> Result<Arc<ProbeModelHandle>, LoadError> {
        Ok(Arc::new(ProbeModelHandle {
            inner: Some(self.take()?.build()?),
        }))
    }
    pub fn build_generative(&self) -> Result<Arc<ProbeGenerativeModel>, LoadError> {
        Ok(Arc::new(ProbeGenerativeModel {
            inner: self.take()?.build_generative()?,
        }))
    }
}

#[uniffi::export]
pub fn model_loader_with_engine_config(
    source: Source,
    config: EngineConfig,
) -> Arc<ProbeModelLoader> {
    Arc::new(ProbeModelLoader {
        source: Mutex::new(Some(source)),
        config: Configuration::Typed(config),
    })
}

#[derive(uniffi::Object)]
pub struct ProbeModelHandle {
    inner: Option<core::ModelHandle>,
}

#[uniffi::export]
impl ProbeModelHandle {
    pub fn kind(&self) -> String {
        self.inner
            .as_ref()
            .map_or_else(|| "future-probe".into(), |h| format!("{:?}", h.kind()))
    }
    pub fn as_generative(&self) -> Option<Arc<ProbeGenerativeModel>> {
        self.inner
            .as_ref()?
            .as_generative()
            .map(|inner| Arc::new(ProbeGenerativeModel { inner }))
    }
}

// Synthetic representation control only; no future core model is loaded.
#[uniffi::export]
pub fn future_handle_for_probe() -> Arc<ProbeModelHandle> {
    Arc::new(ProbeModelHandle { inner: None })
}

#[derive(uniffi::Object)]
pub struct ProbeGenerativeModel {
    inner: core::GenerativeModel,
}

#[uniffi::export]
impl ProbeGenerativeModel {
    // Observe the actual core manifest after loading, not the input record.
    pub fn generation_defaults_for_probe(&self) -> GenerationDefaults {
        (&self.inner.manifest().generation_defaults).into()
    }
    // Observe the repository retained by the core model, not the input record.
    pub fn repository_for_probe(&self) -> Option<Arc<ProbeBundleRepo>> {
        self.inner
            .config()
            .bundle_repo
            .clone()
            .map(|inner| Arc::new(ProbeBundleRepo { inner }))
    }
    pub fn info(&self) -> ModelInfo {
        shared::info(&self.inner)
    }
    // Probe observation of the existing resolved manifest, not input echoing.
    pub fn files(&self) -> ModelFiles {
        let manifest = self.inner.manifest();
        let files = &manifest.files;
        ModelFiles {
            model: files.model.clone(),
            multimodal_projector: files.multimodal_projector.clone(),
            audio_decoder: files.audio_decoder.clone(),
            audio_tokenizer: files.audio_tokenizer.clone(),
            draft_model: files.draft_model.clone(),
            extras: files.extras.clone(),
            inference_type: Some(manifest.inference_type.as_str().to_owned()),
            chat_template: manifest.chat_template.clone(),
        }
    }
    pub fn create_session(&self) -> Result<Arc<ProbeSession>, LoadError> {
        Ok(Arc::new(ProbeSession {
            inner: Mutex::new(shared::session(&self.inner)?),
        }))
    }
    pub fn engine(&self) -> Arc<cera_ffi::CeraEngine> {
        Arc::new(cera_ffi::CeraEngine::from_shared_for_probe(
            self.inner.engine(),
        ))
    }
    pub fn shares_engine_for_probe(&self, engine: Arc<cera_ffi::CeraEngine>) -> bool {
        engine.shares_core_for_probe(self.inner.engine())
    }
    pub fn create_session_with_config(
        &self,
        config: SessionConfig,
    ) -> Result<Arc<cera_ffi::Session>, cera_ffi::FfiError> {
        self.engine().new_session(config)
    }
}

#[derive(uniffi::Object)]
pub struct ProbeSession {
    inner: Mutex<cera::Session>,
}

#[uniffi::export]
impl ProbeSession {
    pub fn append(&self, tokens: Vec<u32>) -> Result<(), LoadError> {
        self.inner
            .lock()
            .unwrap()
            .append_tokens(&tokens)
            .map_err(shared::engine_error)
    }
    pub fn generate(&self) -> Result<Vec<u32>, LoadError> {
        shared::generate(&mut self.inner.lock().unwrap())
    }
    pub fn position(&self) -> u32 {
        self.inner.lock().unwrap().position()
    }
}

#[uniffi::export]
pub fn production_defaults_for_probe(
    engine: Arc<cera_ffi::CeraEngine>,
) -> cera_ffi::GenerationDefaults {
    engine.defaults_for_loading_probe()
}

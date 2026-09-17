//! Existing engine operations retained through a shared generative model.

use super::{GenerativeModel, LoadConfig};
use crate::engine::ModelMetadata;
use crate::kv_cache::KvCacheConfig;
use crate::manifest::Manifest;
use crate::model::Model;
use crate::tokenizer::BpeTokenizer;
use crate::{CeraError, GenerateOpts, ModalityCapabilities};
use std::sync::Arc;

impl GenerativeModel {
    /// Share the already loaded engine; no source resolution or KV copy occurs.
    pub fn engine(&self) -> Arc<crate::CeraEngine> {
        self.engine.clone()
    }

    pub fn model(&self) -> &dyn Model {
        self.engine.model()
    }

    pub fn model_arc(&self) -> Arc<dyn Model> {
        self.engine.model_arc()
    }

    pub fn tokenizer(&self) -> &BpeTokenizer {
        self.engine.tokenizer()
    }

    pub fn tokenizer_arc(&self) -> Arc<BpeTokenizer> {
        self.engine.tokenizer_arc()
    }

    pub fn manifest(&self) -> &Manifest {
        self.engine.manifest()
    }

    pub fn metadata(&self) -> &ModelMetadata {
        self.engine.metadata()
    }

    pub fn config(&self) -> &LoadConfig {
        self.engine.config()
    }

    // This preserves the manifest-derived declaration. It is not a successful
    // auxiliary-load probe; callers still need the legacy encoder accessors.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.engine.capabilities()
    }

    pub fn default_generate_opts(&self) -> GenerateOpts {
        self.engine.default_generate_opts()
    }

    pub fn configure_cache(&self, config: KvCacheConfig) {
        self.engine.configure_cache(config);
    }

    pub fn clear_warm_cache(&self) {
        self.engine.clear_warm_cache();
    }

    pub fn clear_cache(&self) {
        self.engine.clear_cache();
    }

    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<String, CeraError> {
        self.engine.transcribe(pcm, sample_rate)
    }
}

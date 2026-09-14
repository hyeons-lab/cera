// Appended only to the isolated production FFI mirror.
impl CeraEngine {
    pub fn defaults_for_loading_probe(&self) -> GenerationDefaults {
        (&self.inner.manifest().generation_defaults).into()
    }
    pub fn from_shared_for_probe(inner: Arc<cera::CeraEngine>) -> Self {
        Self { inner }
    }
    pub fn shares_core_for_probe(&self, other: Arc<cera::CeraEngine>) -> bool {
        Arc::ptr_eq(&self.inner, &other)
    }
    pub fn shares_wrapper_for_probe(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Session {
    pub fn config_for_loading_probe(&self) -> Result<SessionConfig, FfiError> {
        let config = self.lock_inner()?.config_for_loading_probe();
        Ok(SessionConfig {
            max_seq_len: config.max_seq_len,
            kv_compression: Some(config.kv_compression.into()),
            n_keep: config.n_keep,
            seed: config.seed,
            ubatch_size: config.ubatch_size,
            gpu_depthformer: config.gpu_depthformer,
        })
    }
}

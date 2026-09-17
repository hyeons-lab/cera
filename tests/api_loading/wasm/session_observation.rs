impl CeraEngine {
    pub fn shares_core_for_probe(&self, other: std::sync::Arc<cera::CeraEngine>) -> bool {
        std::sync::Arc::ptr_eq(&self.inner, &other)
    }
}

impl Session {
    pub fn config_for_loading_probe(&self) -> SessionConfig {
        SessionConfig {
            inner: self.inner.config_for_loading_probe(),
        }
    }
}

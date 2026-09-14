use crate::LoadError;
use std::sync::Arc;

#[derive(Debug, uniffi::Object)]
pub struct ProbeBundleRepo {
    pub(crate) inner: cera::bundle::BundleRepo,
}

#[uniffi::export]
impl ProbeBundleRepo {
    #[uniffi::constructor]
    pub fn new(store_dir: String) -> Self {
        Self {
            inner: cera::bundle::BundleRepo::new(store_dir),
        }
    }

    #[uniffi::constructor]
    pub fn with_progress(store_dir: String, progress: Arc<dyn ProbeDownloadProgressSink>) -> Self {
        Self {
            inner: cera::bundle::BundleRepo::with_progress(
                store_dir,
                Arc::new(ProgressAdapter(progress)),
            ),
        }
    }

    pub fn store_dir(&self) -> String {
        self.inner.store_dir().to_string_lossy().into_owned()
    }

    // Trigger real I/O through the retained core repository to observe its callback.
    pub fn resolve_for_probe(&self, url: String) -> Result<String, LoadError> {
        self.inner
            .resolve_url(&url, None)
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(|error| LoadError::Source {
                source_kind: "repository".into(),
                detail: error.to_string(),
            })
    }
}

#[uniffi::export(with_foreign)]
pub trait ProbeDownloadProgressSink: Send + Sync {
    fn on_progress(&self, url: String, bytes_downloaded: u64, total_bytes: Option<u64>);
}

struct ProgressAdapter(Arc<dyn ProbeDownloadProgressSink>);

impl std::fmt::Debug for ProgressAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProgressAdapter")
    }
}

impl cera::bundle::DownloadProgress for ProgressAdapter {
    fn on_progress(&self, url: &str, bytes_downloaded: u64, total_bytes: Option<u64>) {
        self.0
            .on_progress(url.to_owned(), bytes_downloaded, total_bytes);
    }
}

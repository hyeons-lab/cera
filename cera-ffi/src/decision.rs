//! Foreign-language bindings to the public d1 decision models ([`cera::d1`]).
//!
//! A d1 model answers named, typed questions (`choice`, `score`, `noul`) about a state, with
//! optional images or a speech clip, in one forward pass per question and with nothing
//! generated. The surface is JSON in, JSON out: [`DecisionEngine::decide_json`] takes the
//! `/v1/systemone` request body llama.cpp serves and returns its response body.
//!
//! The public models are `d1-omni-600M` (a GGUF plus its `mmproj` GGUF, which carries the vision
//! and speech towers) and `d1-3B`. Both run on the CPU in this version: `open` accepts
//! `Auto` and `Cpu` and refuses an accelerator by name.
//!
//! Threading: one request runs at a time per engine (a second call waits for the first), and a
//! request uses the engine's worker pool for its matrix products. [`DecisionEngine::decide_json`]
//! blocks the calling thread; call it off the main thread or use
//! [`DecisionEngine::decide_json_async`].
//!
//! Cancellation: dropping the future of [`DecisionEngine::decide_json_async`] (cancelling the
//! Kotlin coroutine) stops the request before its next question. The forward pass in flight
//! finishes first, so the worker is busy for up to one question after the cancel: a few hundred
//! milliseconds for text, a few seconds for a large image.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{BackendPreference, FfiError, spawn_blocking_guarded};

/// A loaded public d1 decision model.
#[derive(uniffi::Object)]
pub struct DecisionEngine {
    inner: Arc<cera::d1::D1Engine>,
}

fn to_ffi(error: anyhow::Error) -> FfiError {
    if error.downcast_ref::<cera::d1::Cancelled>().is_some() {
        return FfiError::Cancelled;
    }
    FfiError::Backend {
        detail: format!("{error:#}"),
    }
}

impl DecisionEngine {
    fn build(
        model_path: &str,
        mmproj_path: Option<&str>,
        backend: BackendPreference,
    ) -> Result<Arc<Self>, FfiError> {
        let inner = cera::d1::D1Engine::open(
            std::path::Path::new(model_path),
            mmproj_path.map(std::path::Path::new),
            backend.into(),
        )
        .map_err(to_ffi)?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }
}

#[uniffi::export]
impl DecisionEngine {
    /// Load a d1 decision model from a GGUF path, plus the `mmproj` GGUF of a `d1-omni` model
    /// for image and speech requests (`None` answers text only).
    ///
    /// Blocks while the weights load; use [`DecisionEngine::open_async`] from a UI thread.
    /// `backend` must be `Cpu` or `Auto` (which runs on the CPU).
    #[uniffi::constructor]
    pub fn open(
        model_path: String,
        mmproj_path: Option<String>,
        backend: BackendPreference,
    ) -> Result<Arc<Self>, FfiError> {
        Self::build(&model_path, mmproj_path.as_deref(), backend)
    }

    /// Answer a `/v1/systemone` request given as JSON (`state`, `questions`, optionally `images`
    /// as data URLs, an `audio/` data URL being a speech clip) and return the response JSON
    /// (`answers` and `usage`).
    ///
    /// Blocks for the duration of the forward passes. An invalid request is a
    /// [`FfiError::Backend`] whose detail says what is wrong.
    pub fn decide_json(&self, request_json: String) -> Result<String, FfiError> {
        self.inner.decide_json(&request_json).map_err(to_ffi)
    }

    /// The model type from the file: `lfm2-d1-omni`, `lfm2-d1` or `converted`.
    pub fn decision_type(&self) -> String {
        self.inner.kind().to_string()
    }

    /// Whether image requests can be answered (the vision tower is loaded).
    pub fn supports_images(&self) -> bool {
        self.inner.supports_images()
    }

    /// Whether speech requests can be answered (the speech tower is loaded).
    pub fn supports_audio(&self) -> bool {
        self.inner.supports_audio()
    }

    /// The backend the model runs on: `cpu`.
    pub fn backend_name(&self) -> String {
        self.inner.backend_name().to_string()
    }
}

/// Raises its flag when dropped while armed: the future of an async call was dropped before the
/// work finished.
struct CancelOnDrop {
    flag: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.flag.store(true, Ordering::Relaxed);
        }
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl DecisionEngine {
    /// Async [`DecisionEngine::open`]: the load runs on a tokio blocking worker, so a UI thread
    /// never waits on it. Dropping the future aborts the load only while it is still queued.
    #[uniffi::constructor]
    pub async fn open_async(
        model_path: String,
        mmproj_path: Option<String>,
        backend: BackendPreference,
    ) -> Result<Arc<Self>, FfiError> {
        spawn_blocking_guarded("decision_open_async", move || {
            Self::build(&model_path, mmproj_path.as_deref(), backend)
        })
        .await
    }

    /// Async [`DecisionEngine::decide_json`] on a tokio blocking worker. Dropping the future
    /// cancels the request before its next question (see the module documentation).
    pub async fn decide_json_async(
        self: Arc<Self>,
        request_json: String,
    ) -> Result<String, FfiError> {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut guard = CancelOnDrop {
            flag: Arc::clone(&cancel),
            armed: true,
        };
        let result = spawn_blocking_guarded("decide_json_async", move || {
            self.inner
                .decide_json_cancellable(&request_json, &cancel)
                .map_err(to_ffi)
        })
        .await;
        guard.armed = false;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_refuses_a_missing_file_with_its_path() {
        let err = DecisionEngine::open("/no/such/d1.gguf".into(), None, BackendPreference::Cpu)
            .err()
            .expect("expected a refusal");
        assert!(err.to_string().contains("/no/such/d1.gguf"), "{err}");
    }

    #[test]
    fn open_refuses_an_accelerator_before_touching_any_file() {
        for backend in [
            BackendPreference::Gpu,
            BackendPreference::Metal,
            BackendPreference::Hexagon,
            BackendPreference::Npu,
        ] {
            let err = DecisionEngine::open("/no/such/d1.gguf".into(), None, backend)
                .err()
                .expect("expected a refusal");
            assert!(
                err.to_string().contains("run on the CPU"),
                "{backend:?}: {err}"
            );
        }
    }

    #[test]
    fn a_cancelled_request_maps_to_the_cancelled_error() {
        let error = anyhow::Error::new(cera::d1::Cancelled);
        assert!(matches!(to_ffi(error), FfiError::Cancelled));
        assert!(matches!(
            to_ffi(anyhow::anyhow!("boom")),
            FfiError::Backend { .. }
        ));
    }

    #[test]
    fn dropping_an_armed_guard_raises_the_flag_and_a_disarmed_one_does_not() {
        let flag = Arc::new(AtomicBool::new(false));
        drop(CancelOnDrop {
            flag: Arc::clone(&flag),
            armed: false,
        });
        assert!(!flag.load(Ordering::Relaxed));
        drop(CancelOnDrop {
            flag: Arc::clone(&flag),
            armed: true,
        });
        assert!(flag.load(Ordering::Relaxed));
    }
}

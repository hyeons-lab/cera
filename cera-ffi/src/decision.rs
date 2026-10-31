//! Foreign-language bindings to the public d1 decision models ([`cera::d1`]).
//!
//! A d1 model answers named, typed questions (`choice`, `score`, `noul`) about a state, with
//! optional images or a speech clip, in one forward pass per question and with nothing
//! generated. The surface is JSON in, JSON out: [`DecisionEngine::decide_json`] takes the
//! `/v1/systemone` request body llama.cpp serves and returns its response body.
//!
//! The public models are `d1-omni-600M` (a GGUF plus its `mmproj` GGUF, which carries the vision
//! and speech towers) and `d1-3B` (a GGUF plus its `mmproj` GGUF for images).
//!
//! Backends: `open` takes a [`BackendPreference`]. `Cpu`, `Gpu` (wgpu: Vulkan, or Metal through
//! wgpu), `Metal` and `Hexagon` / `Npu` open on that backend or fail with the reason, and nothing
//! falls back behind the caller's back. `Auto` takes the best accelerator that loads the model and
//! survives a warm-up request, and otherwise the CPU; [`DecisionEngine::backend_report`] says where
//! each part runs and why `Auto` landed on the CPU if it did. What runs where:
//!
//! - d1-omni, trunk: Metal, wgpu or the Hexagon NPU (a bidirectional pass over all rows).
//! - d1-omni, head and the vision and speech towers: on the trunk's backend (Metal, wgpu or the
//!   Hexagon NPU; no mixing, so an NPU phone runs all of d1-omni on the NPU). A part the NPU
//!   cannot take stays on the CPU and the report says which and why. Small host-side steps remain
//!   (patch embedding, log-mel, the head scorer; the head takes 4096 rows a pass on the NPU).
//! - d1-3B, trunk and vision tower: Metal, wgpu or the Hexagon NPU, as the engine's causal
//!   LFM2-VL graph runs them. It has no head.
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
    // keep the typed variants (out of memory, context overflow, busy, ...) the other entry
    // points expose, so a caller can branch on them; a plain `Backend` keeps its full context
    // chain in the formatted error below
    if let Some(typed) = error.downcast_ref::<cera::CeraError>()
        && !matches!(
            typed,
            cera::CeraError::Backend(_)
                | cera::CeraError::Format(_)
                | cera::CeraError::Io(_)
                | cera::CeraError::LoraDimMismatch(_)
                | cera::CeraError::LoraCompose(_)
        )
    {
        return FfiError::from(typed);
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
    /// Load a d1 decision model from a GGUF path, plus the `mmproj` GGUF of the model for image
    /// (and, for `d1-omni`, speech) requests (`None` answers text only).
    ///
    /// Blocks while the weights load and, for `Auto`, run a warm-up request; use
    /// [`DecisionEngine::open_async`] from a UI thread. `backend` is a request, see the module
    /// documentation.
    #[uniffi::constructor]
    pub fn open(
        model_path: String,
        mmproj_path: Option<String>,
        backend: BackendPreference,
    ) -> Result<Arc<Self>, FfiError> {
        let _pin = crate::CallerPinScope;
        Self::build(&model_path, mmproj_path.as_deref(), backend)
    }

    /// Answer a `/v1/systemone` request given as JSON (`state`, `questions`, optionally `images`
    /// as data URLs, an `audio/` data URL being a speech clip) and return the response JSON
    /// (`answers` and `usage`).
    ///
    /// Blocks for the duration of the forward passes. An invalid request is a
    /// [`FfiError::Backend`] whose detail says what is wrong.
    pub fn decide_json(&self, request_json: String) -> Result<String, FfiError> {
        let _pin = crate::CallerPinScope;
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

    /// The backend the trunk runs on: `cpu`, `metal`, `wgpu` or `hexagon`.
    pub fn backend_name(&self) -> String {
        self.inner.backend_name().to_string()
    }

    /// Where each part of the model runs, for example `requested Auto: trunk hexagon, head
    /// hexagon, vision hexagon, audio hexagon`, with the reason in brackets when `Auto` fell back to the
    /// CPU. For logs and diagnostics; the wording is not a stable format.
    pub fn backend_report(&self) -> String {
        self.inner.report().to_string()
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
            let _pin = crate::CallerPinScope;
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
            let _pin = crate::CallerPinScope;
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
    fn typed_engine_errors_keep_their_variant_at_the_boundary() {
        use anyhow::Context as _;
        let overflow = anyhow::Error::new(cera::CeraError::ContextOverflow {
            max_seq_len: 64,
            by: 3,
        })
        .context("answering");
        assert!(matches!(
            to_ffi(overflow),
            FfiError::ContextOverflow {
                max_seq_len: 64,
                by: 3
            }
        ));
        let oom = Err::<(), _>(cera::CeraError::OutOfMemory { requested_bytes: 1 })
            .context("loading")
            .unwrap_err();
        assert!(matches!(to_ffi(oom), FfiError::OutOfMemory { .. }));
        // a plain backend error keeps its whole message
        let backend =
            anyhow::Error::new(cera::CeraError::Backend("no device".into())).context("opening");
        match to_ffi(backend) {
            FfiError::Backend { detail } => {
                assert!(
                    detail.contains("opening") && detail.contains("no device"),
                    "{detail}"
                )
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            to_ffi(anyhow::Error::new(cera::d1::Cancelled)),
            FfiError::Cancelled
        ));
    }

    #[test]
    fn open_refuses_a_missing_file_with_its_path() {
        let err = DecisionEngine::open("/no/such/d1.gguf".into(), None, BackendPreference::Cpu)
            .err()
            .expect("expected a refusal");
        assert!(err.to_string().contains("/no/such/d1.gguf"), "{err}");
    }

    #[test]
    fn every_backend_preference_reaches_the_engine() {
        // a missing file is the first refusal whatever the backend, so the path is named
        for backend in [
            BackendPreference::Auto,
            BackendPreference::Cpu,
            BackendPreference::Gpu,
            BackendPreference::Metal,
            BackendPreference::Hexagon,
            BackendPreference::Npu,
        ] {
            let err = DecisionEngine::open("/no/such/d1.gguf".into(), None, backend)
                .err()
                .expect("expected a refusal");
            assert!(
                err.to_string().contains("/no/such/d1.gguf"),
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

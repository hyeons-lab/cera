//! A d1 model opened from files and answering `/v1/systemone` JSON, the one-call surface the
//! FFI and the CLI share.
//!
//! [`D1Engine`] owns the trunk engine and, by model type, either the d1-omni head and towers
//! ([`D1Model`]) or the causal d1 prompt ([`CausalD1`]). A request is one blocking call: the
//! questions run one after another on the calling thread (each is a single forward pass, using
//! the engine's own worker pool for the matrix products), and the call returns the response JSON.
//!
//! Everything runs on the CPU in this version: an explicit GPU, Metal or NPU preference is
//! refused at [`D1Engine::open`] rather than silently ignored.

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};

use super::causal::{self, CausalD1};
use super::{Cancelled, D1Model, D1Request, D1Response, public};
use crate::engine::{BackendPreference, CeraEngine, EngineConfig, ModelFiles};
use crate::gguf::GgufFile;
use crate::session::SessionConfig;

/// A loaded d1 model.
pub struct D1Engine {
    engine: CeraEngine,
    model: Model,
    /// The model type: `lfm2-d1-omni`, `lfm2-d1`, or `converted` for a cera-converted file.
    kind: String,
    /// One request at a time: the trunk's scratch state is per model, and two requests would
    /// only contend for the same cores.
    busy: Mutex<()>,
}

/// The model type's half of the work.
enum Model {
    /// A public d1-omni model or a cera-converted file.
    Omni(Box<D1Model>),
    /// A public causal d1 model; `images` is whether an mmproj attached its vision tower.
    Causal { model: Box<CausalD1>, images: bool },
}

impl D1Engine {
    /// Open a d1 model: a public `d1-omni` GGUF (plus its `mmproj` GGUF for images and speech), a
    /// public `d1-3B` GGUF (plus its `mmproj` GGUF for images), or a cera-converted file (which
    /// has its towers inside and takes no `mmproj`).
    ///
    /// `backend` must be `Cpu` or `Auto` (which picks the CPU): the d1 head and towers have no
    /// accelerated path in this version.
    ///
    /// # Errors
    ///
    /// Fails when a file does not open, is not a d1 model, is missing a tensor or setting, or
    /// the backend is an accelerator.
    pub fn open(
        model: &Path,
        mmproj_path: Option<&Path>,
        backend: BackendPreference,
    ) -> Result<Self> {
        ensure!(
            matches!(backend, BackendPreference::Cpu | BackendPreference::Auto),
            "d1 models run on the CPU in this version (asked for {backend:?})"
        );
        let gguf = GgufFile::open_arc(model)
            .with_context(|| format!("opening the model `{}`", model.display()))?;
        let mmproj = mmproj_path
            .map(|path| {
                GgufFile::open_arc(path)
                    .with_context(|| format!("opening the mmproj `{}`", path.display()))
            })
            .transpose()?;
        // refuse a non-d1 file before the engine reads its weights
        ensure!(
            D1Model::is_d1(&gguf) || public::decision_type(&gguf).is_some(),
            "`{}` is not a d1 decision model",
            model.display()
        );
        let kind =
            public::decision_type(&gguf).map_or_else(|| "converted".to_string(), str::to_string);
        let is_causal = public::decision_type(&gguf) == Some(public::CAUSAL_TYPE);
        let cfg = EngineConfig {
            context_size: if is_causal {
                causal::CONTEXT_TOKENS
            } else {
                public::OMNI_MAX_LENGTH
            },
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        };
        let (engine, model) = if is_causal {
            // the mmproj goes to the engine, which attaches its vision tower to every session
            let engine = match mmproj_path {
                Some(path) => CeraEngine::from_files(
                    ModelFiles {
                        model: model.to_path_buf(),
                        multimodal_projector: Some(path.to_path_buf()),
                        audio_decoder: None,
                        audio_tokenizer: None,
                        draft_model: None,
                        extras: std::collections::HashMap::new(),
                        // an `lfm2` model with an mmproj is not taken for a vision model unless
                        // it is said to be one
                        inference_type: Some(crate::manifest::InferenceType::LlamaCppImageToText),
                        chat_template: None,
                    },
                    cfg,
                )?,
                None => CeraEngine::from_path(model, cfg)?,
            };
            let causal_model = CausalD1::from_gguf(&gguf, &engine.tokenizer_arc())?;
            let model = Model::Causal {
                model: Box::new(causal_model),
                images: mmproj_path.is_some(),
            };
            (engine, model)
        } else {
            let engine = CeraEngine::from_path(model, cfg)?;
            let d1 = D1Model::open(&gguf, mmproj.as_ref(), &engine.tokenizer_arc())?;
            (engine, Model::Omni(Box::new(d1)))
        };
        Ok(Self {
            engine,
            model,
            kind,
            busy: Mutex::new(()),
        })
    }

    /// The model type: `lfm2-d1-omni`, `lfm2-d1` or `converted`.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The backend the model runs on (`cpu`).
    pub fn backend_name(&self) -> &'static str {
        "cpu"
    }

    /// Whether the model can read images (its vision tower is loaded).
    pub fn supports_images(&self) -> bool {
        match &self.model {
            Model::Omni(model) => model.supports_images(),
            Model::Causal { images, .. } => *images,
        }
    }

    /// Whether the model can read a speech clip (its speech tower is loaded).
    pub fn supports_audio(&self) -> bool {
        match &self.model {
            Model::Omni(model) => model.supports_audio(),
            Model::Causal { .. } => false,
        }
    }

    /// Answer a parsed request. Blocks until every question is answered; a second concurrent
    /// call waits for the first.
    ///
    /// # Errors
    ///
    /// As [`D1Model::answer`] or [`CausalD1::answer`].
    pub fn decide(&self, request: &D1Request) -> Result<D1Response> {
        self.decide_cancellable(request, &AtomicBool::new(false))
    }

    /// As [`Self::decide`], returning [`Cancelled`] once `cancel` is set. The flag is read
    /// before each question's forward pass, so a pass in flight finishes (a few hundred
    /// milliseconds for a text question, seconds for a large image) before the call returns.
    ///
    /// # Errors
    ///
    /// As [`Self::decide`], and [`Cancelled`] (downcast it from the error) when the flag is set.
    pub fn decide_cancellable(
        &self,
        request: &D1Request,
        cancel: &AtomicBool,
    ) -> Result<D1Response> {
        let _turn = self
            .busy
            .lock()
            .map_err(|_| anyhow::anyhow!("a previous d1 request panicked"))?;
        // a request cancelled while it waited for its turn never touches the model
        ensure!(!cancel.load(Ordering::Relaxed), Cancelled);
        let mut session = self.engine.new_session(SessionConfig::default())?;
        let tokenizer = self.engine.tokenizer_arc();
        match &self.model {
            Model::Omni(model) => {
                model.answer_cancellable(&mut session, &tokenizer, request, cancel)
            }
            Model::Causal { model, images } => {
                ensure!(
                    *images || request.images.is_empty(),
                    "this model has no vision tower (load its mmproj)"
                );
                model.answer(&mut session, &tokenizer, request, cancel)
            }
        }
    }

    /// Answer a `/v1/systemone` request given as JSON and return the response JSON
    /// (`{"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}`).
    ///
    /// # Errors
    ///
    /// Fails on a request the model cannot accept (the message says what is wrong) or a trunk
    /// failure.
    pub fn decide_json(&self, request_json: &str) -> Result<String> {
        self.decide_json_cancellable(request_json, &AtomicBool::new(false))
    }

    /// As [`Self::decide_json`], with the cancellation of [`Self::decide_cancellable`].
    ///
    /// # Errors
    ///
    /// As [`Self::decide_json`], and [`Cancelled`] when the flag is set.
    pub fn decide_json_cancellable(
        &self,
        request_json: &str,
        cancel: &AtomicBool,
    ) -> Result<String> {
        let request = D1Request::parse(request_json)?;
        Ok(self.decide_cancellable(&request, cancel)?.to_json().dumps())
    }
}

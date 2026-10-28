//! A d1 model opened from files and answering `/v1/systemone` JSON, the one-call surface the
//! FFI and the CLI share.
//!
//! [`D1Engine`] owns the trunk engine and, by model type, either the d1-omni head and towers
//! ([`D1Model`]) or the causal d1 prompt ([`CausalD1`]). A request is one blocking call: the
//! questions run one after another on the calling thread (each is a single forward pass, using
//! the engine's own worker pool for the matrix products), and the call returns the response JSON.
//!
//! The backend is a request, not a hope. An explicit `Cpu`, `Gpu` (wgpu), `Metal` or `Hexagon`
//! (`Npu`) opens on that backend or fails with the reason; nothing falls back behind the
//! caller's back. `Auto` takes the best backend that loads this model and survives a one-question
//! warm-up request, and otherwise the CPU, with the reason in [`D1Engine::report`].

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};

use super::causal::{self, CausalD1};
use super::{Cancelled, D1Model, D1Request, D1Response, public};
use crate::engine::{BackendPreference, CeraEngine, EngineConfig, ModelFiles};
use crate::gguf::GgufFile;
use crate::session::SessionConfig;

/// The files of a model being opened.
#[derive(Clone, Copy)]
struct Files<'a> {
    model: &'a Path,
    mmproj_path: Option<&'a Path>,
    gguf: &'a std::sync::Arc<GgufFile>,
    mmproj: Option<&'a std::sync::Arc<GgufFile>>,
}

/// The backend to place the head and towers on, given where the trunk landed: the same one (an
/// NPU trunk keeps them on the NPU, no GPU mixing), so an `Auto` load that ended on the CPU leaves
/// them there too.
fn tower_preference(trunk: &str, load_as: BackendPreference) -> BackendPreference {
    match trunk {
        "metal" => BackendPreference::Metal,
        "wgpu" => BackendPreference::Gpu,
        "hexagon" => BackendPreference::Hexagon,
        _ if load_as == BackendPreference::Auto => BackendPreference::Cpu,
        _ => load_as,
    }
}

/// Puts a model's device waits to sleep for the length of a request.
struct LowCpuWaits<'a>(&'a dyn crate::model::Model, bool);

impl<'a> LowCpuWaits<'a> {
    fn enter(model: &'a dyn crate::model::Model) -> Self {
        let acts = model.set_encoder_low_cpu(true);
        Self(model, acts)
    }
}

impl Drop for LowCpuWaits<'_> {
    fn drop(&mut self) {
        if self.1 {
            self.0.set_encoder_low_cpu(false);
        }
    }
}

/// A loaded d1 model.
pub struct D1Engine {
    engine: CeraEngine,
    model: Model,
    /// The model type: `lfm2-d1-omni`, `lfm2-d1`, or `converted` for a cera-converted file.
    kind: String,
    /// One request at a time: the trunk's scratch state is per model, and two requests would
    /// only contend for the same cores.
    busy: Mutex<()>,
    report: BackendReport,
}

/// Where each part of a loaded d1 model runs. Labels are `cpu`, `metal`, `wgpu` or `hexagon`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendReport {
    /// What the caller asked for.
    pub requested: BackendPreference,
    /// The trunk (the LFM2 decoder or the bidirectional encoder).
    pub trunk: &'static str,
    /// The d1 decision head (d1-omni only; `cpu` for d1-3B, which has none).
    pub head: &'static str,
    /// The vision tower, `None` when the model has none loaded.
    pub vision: Option<&'static str>,
    /// The speech tower, `None` when the model has none loaded.
    pub audio: Option<&'static str>,
    /// Why an `Auto` request ended up somewhere other than the best accelerator.
    pub note: Option<String>,
}

impl std::fmt::Display for BackendReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "requested {:?}: trunk {}, head {}",
            self.requested, self.trunk, self.head
        )?;
        if let Some(vision) = self.vision {
            write!(f, ", vision {vision}")?;
        }
        if let Some(audio) = self.audio {
            write!(f, ", audio {audio}")?;
        }
        if let Some(note) = &self.note {
            write!(f, " ({note})")?;
        }
        Ok(())
    }
}

/// A request small enough to cost nothing and real enough to run every trunk kernel: `Auto`
/// opens with it, so a device that loads the weights but cannot run them is caught there.
const WARMUP_REQUEST: &str = r#"{"state": "The parcel arrived on Monday.", "questions": {"q": {"type": "noul", "instructions": "Did the parcel arrive?"}}}"#;

/// The model type's half of the work.
enum Model {
    /// A public d1-omni model or a cera-converted file.
    Omni(Box<D1Model>),
    /// A public causal d1 model; `images` is whether an mmproj attached its vision tower.
    Causal { model: Box<CausalD1>, images: bool },
}

/// Whether an `Auto` load of an image-capable d1-3B must leave wgpu: on Android a phone GPU
/// loses the device on large images, and the warm-up request has no image to show it. The vision
/// tower can sit on wgpu under a CPU trunk, so both placements count; a model without a vision
/// tower has no image path to protect.
fn keep_off_wgpu(android: bool, takes_images: bool, trunk: &str, vision: Option<&str>) -> bool {
    android && takes_images && (trunk == "wgpu" || vision == Some("wgpu"))
}

impl D1Engine {
    /// Open a d1 model: a public `d1-omni` GGUF (plus its `mmproj` GGUF for images and speech), a
    /// public `d1-3B` GGUF (plus its `mmproj` GGUF for images), or a cera-converted file (which
    /// has its towers inside and takes no `mmproj`).
    ///
    /// `backend` picks where the model runs (see the module documentation). `Auto` falls back to
    /// the CPU when no accelerator loads the model or answers the warm-up request, and says why in
    /// [`Self::report`]; every other value is strict.
    ///
    /// # Errors
    ///
    /// Fails when a file does not open, is not a d1 model, is missing a tensor or setting, or
    /// (for an explicit backend) the backend cannot run the model.
    pub fn open(
        model: &Path,
        mmproj_path: Option<&Path>,
        backend: BackendPreference,
    ) -> Result<Self> {
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
        let files = Files {
            model,
            mmproj_path,
            gguf: &gguf,
            mmproj: mmproj.as_ref(),
        };
        if backend != BackendPreference::Auto {
            return Self::build(&files, backend, backend, None);
        }
        // Auto: the best accelerator that loads the model and answers a request, else the CPU
        match Self::build(&files, BackendPreference::Auto, backend, None) {
            // d1-3B on a phone GPU runs text, but an image of 1024 px or more loses the device
            // (the mobile GPU watchdog) and a 512 px one is slower than the CPU; the warm-up
            // request has no image, so it cannot catch that. Keep an image-capable 3B off wgpu
            // on Android: the vision tower can land on wgpu even when the trunk is on the CPU
            // (an `Auto` load follows the device, not the trunk), so test both. A text-only 3B
            // has no image path to protect and keeps its GPU.
            Ok(engine)
                if keep_off_wgpu(
                    cfg!(target_os = "android"),
                    matches!(engine.model, Model::Causal { images: true, .. }),
                    engine.report.trunk,
                    engine.report.vision,
                ) =>
            {
                let note =
                    "wgpu on a phone loses the device on large images for d1-3B; using the CPU"
                        .to_string();
                drop(engine);
                Self::build(&files, BackendPreference::Cpu, backend, Some(note))
            }
            Ok(mut engine) if engine.report.trunk == "cpu" => {
                // the engine's own Auto order found no accelerator for the trunk (none present, or
                // none could hold the model); say so rather than leave a bare "cpu"
                if engine.report.note.is_none() {
                    engine.report.note = Some(
                        "no accelerator took the trunk (none present, or the model did not fit one; the engine log has the reason); using the CPU"
                            .into(),
                    );
                }
                Ok(engine)
            }
            Ok(engine) => match engine.decide_json(WARMUP_REQUEST) {
                Ok(_) => Ok(engine),
                Err(e) => {
                    let note = format!(
                        "{} failed its warm-up request ({e:#}); using the CPU",
                        engine.report.trunk
                    );
                    drop(engine);
                    Self::build(&files, BackendPreference::Cpu, backend, Some(note))
                }
            },
            Err(e) => Self::build(
                &files,
                BackendPreference::Cpu,
                backend,
                Some(format!(
                    "no accelerator could load this model ({e:#}); using the CPU"
                )),
            ),
        }
    }

    /// Load on `load_as`; `requested` is what the caller asked for, for the report.
    fn build(
        files: &Files<'_>,
        load_as: BackendPreference,
        requested: BackendPreference,
        note: Option<String>,
    ) -> Result<Self> {
        let Files {
            model,
            mmproj_path,
            gguf,
            mmproj,
        } = *files;
        let kind =
            public::decision_type(gguf).map_or_else(|| "converted".to_string(), str::to_string);
        let is_causal = public::is_prompt_d1(gguf);
        let cfg = EngineConfig {
            context_size: if is_causal {
                causal::CONTEXT_TOKENS
            } else {
                public::OMNI_MAX_LENGTH
            },
            backend: load_as,
            ..EngineConfig::default()
        };
        let (engine, model, report) = if is_causal {
            // the mmproj goes to the engine, which attaches its vision/audio towers to every session
            let is_audio_only_mmproj = mmproj.is_some_and(|g| {
                crate::model::audio_encoder::is_audio_encoder_gguf(g)
                    && g.get_bool("clip.has_vision_encoder") != Some(true)
            });
            let mut extras = std::collections::HashMap::new();
            if public::decision_type(gguf) == Some(public::LFM3_OMNI_TYPE)
                && !is_audio_only_mmproj
                && let Some(dir) = mmproj_path.or(Some(model)).and_then(Path::parent)
                && let Ok(entries) = std::fs::read_dir(dir)
            {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("gguf")
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("audioproj"))
                    {
                        extras.insert("audio_encoder".to_string(), p);
                        break;
                    }
                }
            }
            let engine = match mmproj_path {
                Some(path) => CeraEngine::from_files(
                    ModelFiles {
                        model: model.to_path_buf(),
                        multimodal_projector: Some(path.to_path_buf()),
                        audio_decoder: None,
                        audio_tokenizer: None,
                        draft_model: None,
                        extras,
                        // an `lfm2` model with an mmproj is not taken for a vision model unless
                        // it is said to be one
                        inference_type: Some(if is_audio_only_mmproj {
                            crate::manifest::InferenceType::LlamaCppLfm2AudioV1
                        } else {
                            crate::manifest::InferenceType::LlamaCppImageToText
                        }),
                        chat_template: None,
                    },
                    cfg,
                )?,
                None if !extras.is_empty() => CeraEngine::from_files(
                    ModelFiles {
                        model: model.to_path_buf(),
                        multimodal_projector: None,
                        audio_decoder: None,
                        audio_tokenizer: None,
                        draft_model: None,
                        extras,
                        inference_type: Some(crate::manifest::InferenceType::LlamaCppTextToText),
                        chat_template: None,
                    },
                    cfg,
                )?,
                None => CeraEngine::from_path(model, cfg)?,
            };
            let trunk = engine.backend_name();
            let vision = if mmproj_path.is_some() && !is_audio_only_mmproj {
                Some(engine.vision_backend_name().unwrap_or("cpu"))
            } else {
                None
            };
            let audio = engine.audio_encoder().map(|_| {
                if engine.has_gpu_audio_encoder() {
                    trunk
                } else {
                    "cpu"
                }
            });
            // a tower the accelerator did not take runs on the CPU: say so
            let mut note = note;
            if vision == Some("cpu") && trunk != "cpu" {
                let msg = format!(
                    "the vision tower did not load on {trunk} (see the cera-hexagon or wgpu lines in the log); on the CPU"
                );
                note = Some(match note {
                    Some(n) => format!("{n}; {msg}"),
                    None => msg,
                });
            }
            let report = BackendReport {
                requested,
                trunk,
                // prompt-based d1 models have no head layers: their answer is read off the trunk's logits on the host
                head: "cpu (no head layers; the answer is read from the trunk's logits)",
                vision,
                audio,
                note,
            };
            let causal_model = CausalD1::from_gguf(gguf, &engine.tokenizer_arc())?;
            let model = Model::Causal {
                model: Box::new(causal_model),
                images: vision.is_some(),
            };
            (engine, model, report)
        } else {
            let engine = CeraEngine::from_path(model, cfg)?;
            let trunk = engine.backend_name();
            let mut d1 = D1Model::open(gguf, mmproj, &engine.tokenizer_arc())?;
            let towers = tower_preference(trunk, load_as);
            d1.accelerate_with(gguf, mmproj, towers);
            let (head, vision, audio) = d1.accelerated();
            let tower_label = match towers {
                BackendPreference::Metal => "metal",
                BackendPreference::Gpu => "wgpu",
                BackendPreference::Hexagon | BackendPreference::Npu => "hexagon",
                BackendPreference::Cpu | BackendPreference::Auto => "cpu",
            };
            let place = |on: bool| if on { tower_label } else { "cpu" };
            // the NPU head takes up to `LONG_STACK_ROWS` rows a pass; a longer prompt runs on the host
            let head_label = if head && towers == BackendPreference::Hexagon {
                "hexagon (host above 4096 rows)"
            } else {
                place(head)
            };
            // an NPU part that did not load runs on the host: say which, never silently
            let mut note = note;
            if towers == BackendPreference::Hexagon {
                let missed: Vec<&str> = [
                    ("head", !head),
                    ("vision tower", d1.supports_images() && !vision),
                    ("speech tower", d1.supports_audio() && !audio),
                ]
                .iter()
                .filter_map(|&(name, off)| off.then_some(name))
                .collect();
                if !missed.is_empty() {
                    let msg = format!(
                        "the NPU did not take the {} (see the cera-hexagon lines in the log for why); on the CPU",
                        missed.join(", ")
                    );
                    note = Some(match note {
                        Some(n) => format!("{n}; {msg}"),
                        None => msg,
                    });
                }
            }
            let report = BackendReport {
                requested,
                trunk,
                head: head_label,
                vision: d1.supports_images().then(|| place(vision)),
                audio: d1.supports_audio().then(|| place(audio)),
                note,
            };
            (engine, Model::Omni(Box::new(d1)), report)
        };
        Ok(Self {
            engine,
            model,
            kind,
            busy: Mutex::new(()),
            report,
        })
    }

    /// Where each part of the model runs.
    pub fn report(&self) -> &BackendReport {
        &self.report
    }

    /// The model type: `lfm2-d1-omni`, `lfm2-d1` or `converted`.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The backend the trunk runs on: `cpu`, `metal`, `wgpu` or `hexagon`. The head and the
    /// towers may be elsewhere; see [`Self::report`].
    pub fn backend_name(&self) -> &'static str {
        self.report.trunk
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
        // a request is batch work: a trunk that waits on a device sleeps between polls, which
        // costs a millisecond or two of wall time and most of the CPU time of a short pass
        let _sleepy = LowCpuWaits::enter(self.engine.model());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn towers_follow_the_trunk() {
        use BackendPreference::*;
        assert_eq!(tower_preference("metal", Auto), Metal);
        assert_eq!(tower_preference("metal", Metal), Metal);
        assert_eq!(tower_preference("wgpu", Gpu), Gpu);
        // an NPU trunk keeps the head and the towers on the NPU, whatever was asked for
        assert_eq!(tower_preference("hexagon", Auto), Hexagon);
        assert_eq!(tower_preference("hexagon", Hexagon), Hexagon);
        assert_eq!(tower_preference("hexagon", Npu), Hexagon);
        // an Auto request that ended on the CPU keeps the towers there
        assert_eq!(tower_preference("cpu", Auto), Cpu);
        assert_eq!(tower_preference("cpu", Cpu), Cpu);
    }

    #[test]
    fn an_image_capable_3b_leaves_wgpu_on_android_whichever_part_is_on_it() {
        // the trunk on wgpu, the vision tower on wgpu under a CPU trunk, or both
        assert!(keep_off_wgpu(true, true, "wgpu", Some("wgpu")));
        assert!(keep_off_wgpu(true, true, "wgpu", None));
        assert!(keep_off_wgpu(true, true, "cpu", Some("wgpu")));
        // nothing on wgpu, no image path, or not a phone: leave the placement alone
        assert!(!keep_off_wgpu(true, true, "hexagon", Some("hexagon")));
        assert!(!keep_off_wgpu(true, true, "cpu", Some("cpu")));
        assert!(!keep_off_wgpu(true, false, "wgpu", None));
        assert!(!keep_off_wgpu(false, true, "wgpu", Some("wgpu")));
    }

    #[test]
    fn the_npu_head_label_names_its_row_limit() {
        assert_eq!(crate::model::vision_encoder_gpu::LONG_STACK_ROWS, 4096);
    }

    #[test]
    fn a_report_names_every_part_and_the_reason_for_a_fallback() {
        let report = BackendReport {
            requested: BackendPreference::Auto,
            trunk: "cpu",
            head: "cpu",
            vision: Some("cpu"),
            audio: None,
            note: Some("no accelerator could load this model (boom); using the CPU".into()),
        };
        assert_eq!(
            report.to_string(),
            "requested Auto: trunk cpu, head cpu, vision cpu (no accelerator could load this model (boom); using the CPU)"
        );
    }

    #[test]
    fn a_file_that_is_not_a_d1_model_is_refused_on_every_backend() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-d1.gguf");
        std::fs::write(&path, b"not a gguf").unwrap();
        for backend in [
            BackendPreference::Auto,
            BackendPreference::Cpu,
            BackendPreference::Metal,
            BackendPreference::Gpu,
            BackendPreference::Hexagon,
        ] {
            let Err(err) = D1Engine::open(&path, None, backend) else {
                panic!("{backend:?} opened a file that is not a GGUF");
            };
            assert!(
                format!("{err:#}").contains("not-d1.gguf"),
                "{backend:?}: {err:#}"
            );
        }
    }
}

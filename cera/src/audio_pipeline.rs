//! Unified audio processing pipeline combining VAD, Hotword, and Whisper ASR.
//!
//! Connects Voice Activity Detection (Silero VAD v5), Keyword Spotting (Hotword detector),
//! and Automatic Speech Recognition (Whisper model) into a streaming voice pipeline.
//!
//! # Example
//! ```ignore
//! use cera::audio_pipeline::{AudioPipeline, AudioPipelineBuilder, AudioPipelineConfig};
//!
//! let pipeline = AudioPipelineBuilder::new()
//!     .with_vad_from_file("models/silero_vad.gguf")?
//!     .with_hotword_from_file("models/hey_liquid.gguf", None)?
//!     .with_whisper_from_file("models/whisper_base.gguf")?
//!     .build()?;
//! ```

use std::collections::{HashMap, VecDeque};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::hotword::{HotwordConfig, HotwordDetector, HotwordIterator};
use crate::live_diarizer::{LiveDiarizer, LiveDiarizerSession, Nemotron3LiveDiarizer};
use crate::model::nemotron3_diarization::{
    Nemotron3Model, StreamingParams as Nemotron3StreamingParams,
};
use crate::model::sortformer::{SortformerModel, StreamingParams};
use crate::model::whisper::{WhisperModel, WhisperTranscribeOpts};
use crate::speaker_labeler::{LabeledUtterance, SpeakerLabelerConfig};
use crate::tokenizer::BpeTokenizer;
use crate::vad::{SileroVad, VadConfig, VadEvent, VadIterator, VadSampleRate};

#[cfg(test)]
mod tests;

/// Active state of the streaming audio pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioPipelineState {
    /// Awaiting a keyword spotting wake word before activating speech recording.
    ListeningForHotword,
    /// Evaluating incoming audio frames to detect speech onset.
    ListeningForSpeech,
    /// Speech onset detected; accumulating utterance samples in the audio buffer.
    SpeechActive,
    /// Transcribing the accumulated speech utterance using Whisper.
    Transcribing,
}

/// An event emitted by the unified audio pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioPipelineEvent {
    /// Keyword spotting detected a wake word.
    WakeWordDetected {
        /// The triggered keyword name.
        keyword: String,
        /// Detection confidence score between 0.0 and 1.0.
        confidence: f32,
        /// Detection timestamp in milliseconds from stream start.
        timestamp_ms: f32,
        /// Sample offset where the detection hop completed.
        sample_offset: u64,
    },
    /// Voice Activity Detection identified speech onset.
    SpeechStart {
        /// Sample index where speech began.
        sample: u64,
        /// Timestamp in milliseconds from stream start.
        ms: f32,
    },
    /// Voice Activity Detection identified speech termination.
    SpeechEnd {
        /// Starting sample index of the speech segment.
        start_sample: u64,
        /// Ending sample index of the speech segment.
        end_sample: u64,
        /// Start timestamp in milliseconds.
        start_ms: f32,
        /// End timestamp in milliseconds.
        end_ms: f32,
    },
    /// Whisper transcription completed for a speech utterance.
    UtteranceTranscribed {
        /// Recognized text output.
        text: String,
        /// Start timestamp of the utterance in milliseconds.
        start_ms: f32,
        /// End timestamp of the utterance in milliseconds.
        end_ms: f32,
        /// Number of 16 kHz audio samples transcribed.
        sample_count: usize,
    },
    /// The attached diarizer has covered an utterance and assigned it a speaker. Emitted once
    /// per utterance, after its [`UtteranceTranscribed`](Self::UtteranceTranscribed) (which
    /// comes first, with the text): a chunk plus its lookahead later, so seconds later with
    /// the checkpoint's default preset. Needs [`AudioPipelineBuilder::with_diarizer`].
    UtteranceLabeled {
        /// The utterance text, as in its `UtteranceTranscribed` event.
        text: String,
        /// Start of the utterance in milliseconds.
        start_ms: f32,
        /// End of the utterance in milliseconds.
        end_ms: f32,
        /// The most active speaker's slot (`0..4` for the Sortformer diarizer, `0..8` for
        /// Nemotron-3), or `None` when the diarizer found no speaker
        /// active over the span, or the labeler had to give the utterance up (see `dropped`).
        speaker: Option<u32>,
        /// The speaker's share of all speakers' active time over the span, in `(0, 1]`.
        confidence: Option<f32>,
        /// A second speaker who was also clearly active over the span, if any.
        overlapping: Option<u32>,
        /// True when the labeler gave the utterance up instead of labeling it (history expiry,
        /// queue overflow, or non-finite times): `None` speaker with `dropped` set is a stalled
        /// diarizer, not silence.
        dropped: bool,
    },
}

/// Configuration options for the unified audio pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioPipelineConfig {
    /// Whether a keyword spotting wake word must be detected before speech tracking begins.
    pub require_hotword: bool,
    /// Whether to automatically run Whisper transcription upon speech completion.
    pub auto_transcribe: bool,
    /// Audio pre-roll duration in milliseconds to retain prior to wake word or speech onset.
    pub pre_roll_ms: usize,
    /// Maximum allowed utterance duration in milliseconds before forcing a boundary.
    pub max_utterance_ms: usize,
    /// Voice Activity Detection configuration.
    pub vad_config: VadConfig,
    /// Keyword Spotting configuration. Overrides detector defaults; an explicit
    /// configuration passed to `with_hotword*` or an attached iterator takes precedence.
    pub hotword_config: Option<HotwordConfig>,
    /// Whisper transcription options.
    pub whisper_opts: Option<WhisperTranscribeOpts>,
    /// Keep the VAD on the CPU even in `hexagon` builds (default false: try the NPU).
    pub vad_on_cpu: bool,
}

impl Default for AudioPipelineConfig {
    fn default() -> Self {
        Self {
            require_hotword: false,
            auto_transcribe: true,
            pre_roll_ms: 200,
            max_utterance_ms: 30_000,
            vad_config: VadConfig::default(),
            hotword_config: None,
            whisper_opts: None,
            vad_on_cpu: false,
        }
    }
}

impl AudioPipelineConfig {
    /// Returns a sanitized configuration with bounded durations and clamped thresholds.
    pub fn sanitized(&self) -> Self {
        Self {
            require_hotword: self.require_hotword,
            auto_transcribe: self.auto_transcribe,
            pre_roll_ms: self.pre_roll_ms.clamp(0, 5000),
            max_utterance_ms: self.max_utterance_ms.clamp(500, 300_000),
            vad_config: self.vad_config.sanitized(),
            hotword_config: self.hotword_config.as_ref().map(|c| c.sanitized()),
            whisper_opts: self.whisper_opts.clone(),
            vad_on_cpu: self.vad_on_cpu,
        }
    }
}

/// Fluent builder for constructing an [`AudioPipeline`].
#[derive(Default)]
pub struct AudioPipelineBuilder {
    vad: Option<SileroVad>,
    vad_config: Option<VadConfig>,
    vad_sample_rate: Option<VadSampleRate>,
    hotword: Option<HotwordIterator>,
    hotword_config_explicit: bool,
    hotword_on_cpu: bool,
    whisper: Option<WhisperModel>,
    whisper_tokenizer: Option<BpeTokenizer>,
    config: Option<AudioPipelineConfig>,
    cancel: Option<Arc<AtomicBool>>,
    vad_on_cpu: bool,
    diarizer: Option<DiarizerKind>,
    labeler_config: Option<SpeakerLabelerConfig>,
}

/// Which speaker diarizer a pipeline attaches. Only one: the second `with_diarizer*` call
/// replaces the first.
enum DiarizerKind {
    Sortformer(SortformerModel, StreamingParams),
    Nemotron3(Nemotron3Model, Nemotron3StreamingParams),
}

/// What [`AudioPipelineBuilder::build`] does with an attached VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadBackendAction {
    /// Opted out: drop any accelerator the VAD already carries.
    ForceCpu,
    /// Default: try the NPU when the VAD has no accelerator yet.
    TryHexagon,
    /// No VAD attached, or already accelerated: leave it alone.
    Keep,
}

/// The pure `build` decision behind [`AudioPipelineBuilder::with_vad_on_cpu`], truth-tabled in
/// `tests` so the wiring (not just the flag) is pinned.
fn vad_backend_action(vad_on_cpu: bool, has_vad: bool, accelerated: bool) -> VadBackendAction {
    if !has_vad {
        VadBackendAction::Keep
    } else if vad_on_cpu {
        VadBackendAction::ForceCpu
    } else if accelerated {
        VadBackendAction::Keep
    } else {
        VadBackendAction::TryHexagon
    }
}

impl AudioPipelineBuilder {
    /// Create a new empty audio pipeline builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a Silero VAD session.
    pub fn with_vad(mut self, vad: SileroVad) -> Self {
        self.vad = Some(vad);
        self
    }

    /// Keep the VAD on the CPU (`true`) or let [`Self::build`] move it to the NPU (the
    /// default, `false`). In a build with the `hexagon` feature the pipeline otherwise runs the
    /// VAD's 16 kHz windows on the Hexagon NPU when one is available: the VAD runs for every
    /// 32 ms of audio, speech or not, so it is the model an always-on service keeps busiest, and
    /// on the NPU it costs the CPU on the order of a tenth of what it does on the CPU (S25 Ultra,
    /// release build; re-measure on your SoC with `cera/examples/hexagon_vad_probe.rs`) while
    /// escaping Android's demotion of background CPU work. Opting out also drops an accelerator
    /// a [`Self::with_vad`] session already carries. Equivalent to setting
    /// [`AudioPipelineConfig::vad_on_cpu`]; either opts out, so [`AudioPipeline::from_files`]
    /// callers (who only carry the config) can opt out too.
    pub fn with_vad_on_cpu(mut self, on_cpu: bool) -> Self {
        self.vad_on_cpu = on_cpu;
        self
    }

    /// Attach a custom VAD configuration.
    pub fn with_vad_config(mut self, config: VadConfig) -> Self {
        self.vad_config = Some(config);
        self
    }

    /// Keep the keyword spotter on the CPU. In a build with the `hexagon` feature the pipeline
    /// otherwise runs its backbone on the Hexagon NPU when one is available: the detector
    /// re-evaluates a 1.2 s window every 80 ms whether or not anyone is speaking, so on the CPU it
    /// is the largest always-on cost of a wake word, and Android demotes background CPU work and
    /// not the NPU.
    pub fn with_hotword_on_cpu(mut self) -> Self {
        self.hotword_on_cpu = true;
        self
    }

    /// Attach a Keyword Spotting detector.
    pub fn with_hotword(
        mut self,
        detector: HotwordDetector,
        config: Option<HotwordConfig>,
    ) -> Self {
        self.set_hotword_detector(detector, config);
        self
    }

    /// Record a detector plus whether its config was explicit (explicit wins
    /// over pipeline defaults in [`Self::build`]).
    fn set_hotword_detector(&mut self, detector: HotwordDetector, config: Option<HotwordConfig>) {
        self.hotword_config_explicit = config.is_some();
        let cfg = config.unwrap_or_else(|| detector.default_config());
        self.hotword = Some(HotwordIterator::new(detector, None, cfg));
    }

    /// Attach an existing Keyword Spotting iterator.
    pub fn with_hotword_iterator(mut self, iterator: HotwordIterator) -> Self {
        self.hotword_config_explicit = true;
        self.hotword = Some(iterator);
        self
    }

    /// Attach a Whisper model and its corresponding tokenizer.
    pub fn with_whisper(mut self, model: WhisperModel, tokenizer: BpeTokenizer) -> Self {
        self.whisper = Some(model);
        self.whisper_tokenizer = Some(tokenizer);
        self
    }

    /// Attach custom Whisper transcription options.
    pub fn with_whisper_opts(mut self, opts: WhisperTranscribeOpts) -> Self {
        let mut cfg = self.config.unwrap_or_default();
        cfg.whisper_opts = Some(opts);
        self.config = Some(cfg);
        self
    }

    /// Set full pipeline configuration.
    pub fn with_config(mut self, config: AudioPipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Configure whether a wake word is required.
    pub fn with_require_hotword(mut self, require: bool) -> Self {
        let mut cfg = self.config.unwrap_or_default();
        cfg.require_hotword = require;
        self.config = Some(cfg);
        self
    }

    /// Configure whether to automatically transcribe utterances on completion.
    pub fn with_auto_transcribe(mut self, auto: bool) -> Self {
        let mut cfg = self.config.unwrap_or_default();
        cfg.auto_transcribe = auto;
        self.config = Some(cfg);
        self
    }

    /// Attach an external cooperative cancellation latch.
    pub fn with_cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Attach a Sortformer speaker diarizer. The pipeline then feeds it every sample it sees
    /// (whatever its state: the diarizer needs the continuous signal, so with a required wake word
    /// it also hears the audio before the wake word) and follows each
    /// [`AudioPipelineEvent::UtteranceTranscribed`] with an
    /// [`AudioPipelineEvent::UtteranceLabeled`] once the diarizer has covered the utterance.
    ///
    /// Put the model on the NPU first with `try_hexagon_sortformer` if it should run there: the
    /// accelerator belongs to the model. `params` are the streaming parameters, usually
    /// [`SortformerModel::default_streaming`]; they set the delay before a label (see
    /// [`SortformerLive::latency_frames`](crate::model::sortformer::SortformerLive::latency_frames)).
    pub fn with_diarizer(mut self, model: SortformerModel, params: StreamingParams) -> Self {
        self.diarizer = Some(DiarizerKind::Sortformer(model, params));
        self
    }

    /// Attach a Nemotron-3-Diarization speaker diarizer (8 speakers, 10 ms frames). Behaves
    /// like [`Self::with_diarizer`]; `params` are usually
    /// [`Nemotron3Model::default_streaming`](crate::model::nemotron3_diarization::Nemotron3Model::default_streaming).
    pub fn with_diarizer_nemotron3(
        mut self,
        model: Nemotron3Model,
        params: Nemotron3StreamingParams,
    ) -> Self {
        self.diarizer = Some(DiarizerKind::Nemotron3(model, params));
        self
    }

    /// Load a converted Nemotron-3 GGUF and attach it with the checkpoint's own streaming
    /// parameters (see [`Self::with_diarizer_nemotron3`]).
    #[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
    pub fn with_diarizer_nemotron3_from_file<P: AsRef<Path>>(self, path: P) -> Result<Self> {
        let model = Nemotron3Model::from_file(path)
            .context("failed to load the Nemotron-3 diarizer from file")?;
        let params = model.default_streaming().clone();
        Ok(self.with_diarizer_nemotron3(model, params))
    }

    /// Load a converted Nemotron-3 GGUF from in-memory bytes and attach it with the
    /// checkpoint's own streaming parameters (see [`Self::with_diarizer_nemotron3`]).
    pub fn with_diarizer_nemotron3_from_bytes(self, bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let model = Nemotron3Model::from_bytes(bytes)
            .context("failed to load the Nemotron-3 diarizer from bytes")?;
        let params = model.default_streaming().clone();
        Ok(self.with_diarizer_nemotron3(model, params))
    }

    /// Load a converted Sortformer GGUF and attach it with the checkpoint's own streaming
    /// parameters (see [`Self::with_diarizer`]).
    #[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
    pub fn with_diarizer_from_file<P: AsRef<Path>>(self, path: P) -> Result<Self> {
        let model = SortformerModel::from_file(path)
            .context("failed to load the Sortformer diarizer from file")?;
        let params = model.default_streaming().clone();
        Ok(self.with_diarizer(model, params))
    }

    /// Load a converted Sortformer GGUF from in-memory bytes and attach it with the
    /// checkpoint's own streaming parameters (see [`Self::with_diarizer`]). The bytes path
    /// for callers without filesystem access (wasm32 has no `from_file` loaders).
    pub fn with_diarizer_from_bytes(self, bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let model = SortformerModel::from_bytes(bytes)
            .context("failed to load the Sortformer diarizer from bytes")?;
        let params = model.default_streaming().clone();
        Ok(self.with_diarizer(model, params))
    }

    /// How utterances are matched with the diarizer's speaker activity (see
    /// [`SpeakerLabelerConfig`]). Only used with [`Self::with_diarizer`] or
    /// [`Self::with_diarizer_nemotron3`].
    pub fn with_speaker_labeler_config(mut self, config: SpeakerLabelerConfig) -> Self {
        self.labeler_config = Some(config);
        self
    }

    /// Load Silero VAD from a GGUF file path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_vad_from_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let vad = SileroVad::from_file(path).context("failed to load Silero VAD from file")?;
        self.vad = Some(vad);
        Ok(self)
    }

    /// Load Silero VAD from in-memory GGUF bytes.
    pub fn with_vad_from_bytes(mut self, bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let vad = SileroVad::from_bytes(bytes).context("failed to load Silero VAD from bytes")?;
        self.vad = Some(vad);
        Ok(self)
    }

    /// Load Keyword Spotting model from a GGUF file path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_hotword_from_file<P: AsRef<Path>>(
        mut self,
        path: P,
        config: Option<HotwordConfig>,
    ) -> Result<Self> {
        let detector = HotwordDetector::from_file(path)
            .context("failed to load Hotword detector from file")?;
        self.set_hotword_detector(detector, config);
        Ok(self)
    }

    /// Load Keyword Spotting model from in-memory GGUF bytes.
    pub fn with_hotword_from_bytes(
        mut self,
        bytes: impl Into<Arc<[u8]>>,
        config: Option<HotwordConfig>,
    ) -> Result<Self> {
        let detector = HotwordDetector::from_bytes(bytes)
            .context("failed to load Hotword detector from bytes")?;
        self.set_hotword_detector(detector, config);
        Ok(self)
    }

    /// Load Whisper model and tokenizer from a GGUF file path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_whisper_from_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let (model, tokenizer) =
            WhisperModel::from_file(path).context("failed to load Whisper model from file")?;
        self.whisper = Some(model);
        self.whisper_tokenizer = Some(tokenizer);
        Ok(self)
    }

    /// Load Whisper model and tokenizer from in-memory GGUF bytes.
    pub fn with_whisper_from_bytes(mut self, bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let (model, tokenizer) =
            WhisperModel::from_bytes(bytes).context("failed to load Whisper model from bytes")?;
        self.whisper = Some(model);
        self.whisper_tokenizer = Some(tokenizer);
        Ok(self)
    }

    /// Construct the initialized [`AudioPipeline`].
    pub fn build(self) -> Result<AudioPipeline> {
        let has_hotword = self.hotword.is_some();
        let explicit_config = self.config.is_some();
        let mut config = self.config.unwrap_or_default().sanitized();

        // If hotword is attached and caller did not explicitly override require_hotword, default to true.
        if has_hotword && !config.require_hotword && !explicit_config {
            config.require_hotword = true;
        }

        let vad_config = self
            .vad_config
            .unwrap_or(config.vad_config.clone())
            .sanitized();
        config.vad_config = vad_config.clone();
        #[cfg_attr(not(feature = "hexagon"), allow(unused_mut))]
        let mut hotword = self.hotword.map(|iterator| {
            if !self.hotword_config_explicit
                && let Some(cfg) = config.hotword_config.clone()
            {
                iterator.with_config(cfg)
            } else {
                iterator
            }
        });
        #[cfg(feature = "hexagon")]
        if !self.hotword_on_cpu
            && let Some(h) = hotword.as_mut()
            && !h.detector().is_accelerated()
        {
            h.detector_mut().try_enable_hexagon();
        }
        let vad = {
            let mut vad = self.vad;
            // Either the builder flag or the config opts out (fail-safe); `from_files` only
            // carries the config, so the knob must live in both.
            let action = vad_backend_action(
                self.vad_on_cpu || config.vad_on_cpu,
                vad.is_some(),
                vad.as_ref().is_some_and(|v| v.is_accelerated()),
            );
            match (vad.as_mut(), action) {
                (Some(v), VadBackendAction::ForceCpu) => v.clear_accelerator(),
                #[cfg(feature = "hexagon")]
                (Some(v), VadBackendAction::TryHexagon) => {
                    v.try_enable_hexagon();
                }
                _ => {}
            }
            vad
        };
        let vad_sample_rate = self.vad_sample_rate.unwrap_or(VadSampleRate::Rate16kHz);
        let vad_iter = if vad.is_some() {
            Some(VadIterator::new(vad_sample_rate, vad_config))
        } else {
            None
        };

        let initial_state = if config.require_hotword && has_hotword {
            AudioPipelineState::ListeningForHotword
        } else {
            AudioPipelineState::ListeningForSpeech
        };

        let cancel = self
            .cancel
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let labeler_config = self.labeler_config.unwrap_or_default();
        let diarizer: Option<Box<dyn PipelineDiarizerDriver>> = self
            .diarizer
            .map(|kind| -> Result<Box<dyn PipelineDiarizerDriver>> {
                Ok(match kind {
                    DiarizerKind::Sortformer(model, params) => {
                        Box::new(PipelineDiarizer::<LiveDiarizer>::new(
                            model,
                            params,
                            labeler_config.clone(),
                        )?)
                    }
                    DiarizerKind::Nemotron3(model, params) => {
                        Box::new(PipelineDiarizer::<Nemotron3LiveDiarizer>::new(
                            model,
                            params,
                            labeler_config.clone(),
                        )?)
                    }
                })
            })
            .transpose()
            .context("failed to start the speaker diarizer")?;

        Ok(AudioPipeline {
            vad,
            vad_iter,
            hotword,
            whisper: self.whisper,
            whisper_tokenizer: self.whisper_tokenizer,
            config,
            state: initial_state,
            current_sample: 0,
            hotword_paused_at: 0,
            vad_sample_offset: 0,
            speech_history: VecDeque::new(),
            utterance_buffer: Vec::with_capacity(32_000),
            last_utterance: Vec::with_capacity(32_000),
            current_utterance_start_sample: 0,
            pending_events: VecDeque::with_capacity(8),
            cancel,
            diarizer,
        })
    }
}

/// What the pipeline remembers about an utterance while the diarizer has not yet covered it.
struct PendingUtterance {
    text: String,
    start_ms: f32,
    end_ms: f32,
}

/// A labeled utterance with the speaker-slot count erased: everything the pipeline's events
/// carry, whichever diarizer labeled it.
struct PipelineLabeledUtterance {
    id: u64,
    label: Option<(usize, f32, Option<usize>)>,
    dropped: bool,
}

impl PipelineLabeledUtterance {
    fn from_labeled<const S: usize>(u: LabeledUtterance<S>) -> Self {
        Self {
            id: u.id,
            label: u.label.map(|l| (l.speaker, l.confidence, l.overlapping)),
            dropped: u.dropped,
        }
    }
}

impl<const S: usize> From<LabeledUtterance<S>> for PipelineLabeledUtterance {
    fn from(u: LabeledUtterance<S>) -> Self {
        Self::from_labeled(u)
    }
}

/// What the pipeline drives a diarizer with, whichever model it runs. The two structs below
/// implement it; the pipeline never sees which one it holds. `Send` because the FFI
/// pipeline crosses threads behind a mutex.
trait PipelineDiarizerDriver: Send {
    /// Feed audio; on failure the diarizer stops and the pipeline continues without labels.
    fn push_audio(&mut self, pcm: &[f32]);
    /// Register an utterance for labeling. Returns whether a diarizer will label it.
    fn register(&mut self, text: String, start_ms: f32, end_ms: f32) -> bool;
    /// Poll covered utterances into `events`.
    fn poll(&mut self, events: &mut Vec<AudioPipelineEvent>);
    /// End of audio at `at_sample`: label everything waiting, then restart the session there.
    fn finish_and_restart(&mut self, events: &mut Vec<AudioPipelineEvent>, at_sample: u64);
    /// Drop the session and everything waiting on it.
    fn reset(&mut self);
    /// Whether the diarizer is still running (it stops, with a warning, if it fails).
    fn is_live(&self) -> bool;
    /// Test introspection: utterances waiting on a poll.
    #[cfg(test)]
    fn debug_pending_len(&self) -> usize;
    /// Test introspection: pipeline-clock time of the session's sample 0.
    #[cfg(test)]
    fn debug_origin_ms(&self) -> f64;
    /// Test introspection: debug rendering of the streaming parameters.
    #[cfg(test)]
    fn debug_params(&self) -> String;
}

/// The diarizer attached to a pipeline and the bookkeeping that ties its labels back to the
/// pipeline's events, written once over the session type: [`LiveDiarizer`] (4 slots, 80 ms
/// frames) or [`Nemotron3LiveDiarizer`] (8 slots, 10 ms frames).
///
/// The diarizer counts time from the first sample it was fed. After a [`flush`](AudioPipeline::flush)
/// (end of audio for the diarizer, which then labels everything still waiting) a fresh session
/// starts at the pipeline's current sample, so `origin_ms` is the pipeline-clock time of that
/// session's sample 0. Speaker slots are per session: a slot after a flush is not the same
/// speaker as before it.
struct PipelineDiarizer<S: LiveDiarizerSession> {
    model: S::Model,
    params: S::Params,
    cfg: SpeakerLabelerConfig,
    /// `None` after a failure: the pipeline keeps transcribing without labels.
    session: Option<S>,
    origin_ms: f64,
    next_id: u64,
    pending: HashMap<u64, PendingUtterance>,
}

/// Pipeline-clock `ms` in session-local time: an utterance that began before this session
/// did (the span straddles a flush) starts at the session's first frame.
///
/// Non-finite times pass through untouched: `f64::max` maps NaN and `-inf` to `0.0`,
/// which would smuggle an unplaceable utterance past the labeler's non-finite guard and
/// emit it `dropped=false` with NaN timestamps instead of `dropped`.
fn session_local_ms(ms: f32, origin_ms: f64) -> f64 {
    if !ms.is_finite() {
        return ms as f64;
    }
    (ms as f64 - origin_ms).max(0.0)
}

/// The `UtteranceLabeled` event for a registered utterance the diarizer has covered.
fn labeled_event(p: PendingUtterance, u: &PipelineLabeledUtterance) -> AudioPipelineEvent {
    AudioPipelineEvent::UtteranceLabeled {
        text: p.text,
        start_ms: p.start_ms,
        end_ms: p.end_ms,
        speaker: u.label.map(|(speaker, _, _)| speaker as u32),
        confidence: u.label.map(|(_, confidence, _)| confidence),
        overlapping: u
            .label
            .and_then(|(_, _, overlapping)| overlapping)
            .map(|o| o as u32),
        dropped: u.dropped,
    }
}

/// Take a covered utterance out of `pending` and label it. Unknown ids (already taken, or
/// never registered) yield nothing.
fn take_labeled_event(
    pending: &mut HashMap<u64, PendingUtterance>,
    u: PipelineLabeledUtterance,
) -> Option<AudioPipelineEvent> {
    let p = pending.remove(&u.id)?;
    Some(labeled_event(p, &u))
}

impl<S: LiveDiarizerSession> PipelineDiarizer<S>
where
    S::Labeled: Into<PipelineLabeledUtterance>,
{
    fn new(model: S::Model, params: S::Params, cfg: SpeakerLabelerConfig) -> Result<Self> {
        let session = S::open(&model, params.clone(), cfg.clone())?;
        Ok(Self {
            model,
            params,
            cfg,
            session: Some(session),
            origin_ms: 0.0,
            next_id: 0,
            pending: HashMap::new(),
        })
    }

    /// Drop the session and everything waiting on it; the diarizer is out for good.
    fn fail(&mut self, what: &str, e: &anyhow::Error) {
        tracing::warn!("speaker diarizer {what} failed ({e:#}); continuing without speaker labels");
        // No `tracing` subscriber on the shipping mobile/FFI platforms; without this
        // the reason for the label loss goes nowhere.
        eprintln!(
            "cera-audio-pipeline: speaker diarizer {what} failed ({e:#}); continuing without speaker labels"
        );
        self.session = None;
        self.pending.clear();
    }

    fn push_audio(&mut self, pcm: &[f32]) {
        if let Some(session) = &mut self.session
            && let Err(e) = session.push_audio(pcm)
        {
            self.fail("audio", &e);
        }
    }

    /// Register an utterance for labeling. Returns whether a diarizer will label it.
    ///
    /// Refuses (returns false) once `max_pending` utterances are waiting on a poll: the
    /// labeler parks at most that many, so an unbounded grant would return true for an
    /// utterance that can never emit `UtteranceLabeled` while pinning its text.
    fn register(&mut self, text: String, start_ms: f32, end_ms: f32) -> bool {
        let Some(session) = &mut self.session else {
            return false;
        };
        if self.pending.len() >= self.cfg.max_pending.max(1) {
            return false;
        }
        let id = self.next_id;
        self.next_id += 1;
        session.add_utterance(
            id,
            session_local_ms(start_ms, self.origin_ms),
            session_local_ms(end_ms, self.origin_ms),
        );
        self.pending.insert(
            id,
            PendingUtterance {
                text,
                start_ms,
                end_ms,
            },
        );
        true
    }

    fn labeled(&mut self, u: S::Labeled) -> Option<AudioPipelineEvent> {
        take_labeled_event(&mut self.pending, u.into())
    }

    /// Utterances the diarizer has covered by now.
    fn poll(&mut self, events: &mut Vec<AudioPipelineEvent>) {
        let Some(session) = &mut self.session else {
            return;
        };
        let done = session.poll();
        events.extend(done.into_iter().filter_map(|u| self.labeled(u)));
    }

    /// End of audio: label everything waiting with the frames that exist, then start a new
    /// session whose sample 0 is the pipeline's `at_sample`.
    fn finish_and_restart(&mut self, events: &mut Vec<AudioPipelineEvent>, at_sample: u64) {
        let Some(mut session) = self.session.take() else {
            return;
        };
        match session.finish() {
            Ok(done) => events.extend(done.into_iter().filter_map(|u| self.labeled(u))),
            Err(e) => {
                self.fail("flush", &e);
                return;
            }
        }
        self.origin_ms = sample_ms(at_sample) as f64;
        match S::open(&self.model, self.params.clone(), self.cfg.clone()) {
            Ok(fresh) => self.session = Some(fresh),
            Err(e) => self.fail("restart", &e),
        }
    }

    /// A new stream: forget the session and everything pending, restart at time zero.
    fn reset(&mut self) {
        self.pending.clear();
        self.origin_ms = 0.0;
        match S::open(&self.model, self.params.clone(), self.cfg.clone()) {
            Ok(fresh) => self.session = Some(fresh),
            Err(e) => self.fail("reset", &e),
        }
    }
}

impl<S: LiveDiarizerSession> PipelineDiarizerDriver for PipelineDiarizer<S>
where
    S::Labeled: Into<PipelineLabeledUtterance>,
{
    fn push_audio(&mut self, pcm: &[f32]) {
        PipelineDiarizer::<S>::push_audio(self, pcm);
    }

    fn register(&mut self, text: String, start_ms: f32, end_ms: f32) -> bool {
        PipelineDiarizer::<S>::register(self, text, start_ms, end_ms)
    }

    fn poll(&mut self, events: &mut Vec<AudioPipelineEvent>) {
        PipelineDiarizer::<S>::poll(self, events);
    }

    fn finish_and_restart(&mut self, events: &mut Vec<AudioPipelineEvent>, at_sample: u64) {
        PipelineDiarizer::<S>::finish_and_restart(self, events, at_sample);
    }

    fn reset(&mut self) {
        PipelineDiarizer::<S>::reset(self);
    }

    fn is_live(&self) -> bool {
        self.session.is_some()
    }

    #[cfg(test)]
    fn debug_pending_len(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    fn debug_origin_ms(&self) -> f64 {
        self.origin_ms
    }

    #[cfg(test)]
    fn debug_params(&self) -> String {
        format!("{:?}", self.params)
    }
}

/// Maximum queued pending events before oldest events are dropped.
const MAX_PENDING_EVENTS: usize = 128;

/// Unified audio facade coordinating VAD, Hotword, and Whisper ASR.
pub struct AudioPipeline {
    vad: Option<SileroVad>,
    vad_iter: Option<VadIterator>,
    hotword: Option<HotwordIterator>,
    whisper: Option<WhisperModel>,
    whisper_tokenizer: Option<BpeTokenizer>,
    config: AudioPipelineConfig,
    state: AudioPipelineState,
    current_sample: u64,
    hotword_paused_at: u64,
    vad_sample_offset: u64,
    speech_history: VecDeque<f32>,
    utterance_buffer: Vec<f32>,
    last_utterance: Vec<f32>,
    current_utterance_start_sample: u64,
    pending_events: VecDeque<AudioPipelineEvent>,
    cancel: Arc<AtomicBool>,
    diarizer: Option<Box<dyn PipelineDiarizerDriver>>,
}

impl AudioPipeline {
    /// Return a new builder for constructing an [`AudioPipeline`].
    pub fn builder() -> AudioPipelineBuilder {
        AudioPipelineBuilder::new()
    }

    /// Construct a pipeline from filesystem model paths.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_files(
        vad_path: Option<&Path>,
        hotword_path: Option<&Path>,
        whisper_path: Option<&Path>,
        config: Option<AudioPipelineConfig>,
    ) -> Result<Self> {
        let mut builder = Self::builder();
        if let Some(cfg) = config {
            builder = builder.with_config(cfg);
        }
        if let Some(vp) = vad_path {
            builder = builder.with_vad_from_file(vp)?;
        }
        if let Some(hp) = hotword_path {
            builder = builder.with_hotword_from_file(hp, None)?;
        }
        if let Some(wp) = whisper_path {
            builder = builder.with_whisper_from_file(wp)?;
        }
        builder.build()
    }

    /// The current lifecycle state of the pipeline.
    pub fn state(&self) -> AudioPipelineState {
        self.state
    }

    /// Current configuration of the pipeline.
    pub fn config(&self) -> &AudioPipelineConfig {
        &self.config
    }

    /// Whether speech activity is currently ongoing.
    pub fn is_speech_active(&self) -> bool {
        self.state == AudioPipelineState::SpeechActive
    }

    /// Whether the pipeline is currently awaiting a wake word trigger.
    pub fn is_listening_for_hotword(&self) -> bool {
        self.state == AudioPipelineState::ListeningForHotword
    }

    /// Total audio samples processed by the pipeline since start or reset.
    pub fn current_sample(&self) -> u64 {
        self.current_sample
    }

    /// Whether a keyword spotting model is attached.
    pub fn has_hotword(&self) -> bool {
        self.hotword.is_some()
    }

    /// Whether a Voice Activity Detection model is attached.
    pub fn has_vad(&self) -> bool {
        self.vad.is_some()
    }

    /// Reference to the attached Voice Activity Detection session if configured.
    pub fn vad(&self) -> Option<&SileroVad> {
        self.vad.as_ref()
    }

    /// Whether a Whisper speech recognition model and tokenizer are attached.
    pub fn has_whisper(&self) -> bool {
        self.whisper.is_some() && self.whisper_tokenizer.is_some()
    }

    /// Clone the cooperative cancellation latch.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// Trigger cooperative cancellation of any active transcription.
    ///
    /// Cancellation is sticky across utterances. Call [`clear_cancel`](Self::clear_cancel)
    /// or [`reset`](Self::reset) before subsequent speech segments to resume transcription.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Clear the cooperative cancellation latch.
    pub fn clear_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    /// Return a view of the most recently finished utterance audio samples.
    pub fn last_utterance(&self) -> &[f32] {
        &self.last_utterance
    }

    /// Take ownership of the most recently completed utterance audio samples.
    pub fn take_last_utterance(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.last_utterance)
    }

    /// Pop a queued event emitted by previous chunk evaluations.
    pub fn pop_event(&mut self) -> Option<AudioPipelineEvent> {
        self.pending_events.pop_front()
    }

    /// Reset stream state, VAD recurrent state, KWS ring buffer, and speech accumulators.
    pub fn reset(&mut self) {
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
        if let Some(vad_iter) = &mut self.vad_iter {
            vad_iter.reset();
        }
        if let Some(hotword) = &mut self.hotword {
            hotword.reset();
        }
        self.current_sample = 0;
        self.hotword_paused_at = 0;
        self.vad_sample_offset = 0;
        self.speech_history.clear();
        self.utterance_buffer.clear();
        self.last_utterance.clear();
        self.current_utterance_start_sample = 0;
        self.pending_events.clear();
        if let Some(d) = &mut self.diarizer {
            d.reset();
        }
        self.cancel.store(false, Ordering::Relaxed);
        self.state = if self.config.require_hotword && self.hotword.is_some() {
            AudioPipelineState::ListeningForHotword
        } else {
            AudioPipelineState::ListeningForSpeech
        };
    }

    fn enqueue_event(&mut self, ev: AudioPipelineEvent) {
        if self.pending_events.len() >= MAX_PENDING_EVENTS {
            self.pending_events.pop_front();
        }
        self.pending_events.push_back(ev);
    }

    /// Process a streaming chunk of 16 kHz mono PCM audio samples.
    ///
    /// Evaluates wake words, speech boundaries, and automatic transcription according to
    /// the active state machine, returning all newly triggered pipeline events. The attached
    /// diarizer hears a chunk only when the pipeline accepts it: on failure its clock stays
    /// with the pipeline's instead of running past the rejected audio.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<AudioPipelineEvent>> {
        if chunk.is_empty() {
            return Ok(Vec::new());
        }

        let sanitized: Option<Vec<f32>> = chunk.iter().any(|s| !s.is_finite()).then(|| {
            chunk
                .iter()
                .map(|&s| if s.is_finite() { s } else { 0.0 })
                .collect()
        });
        let chunk = sanitized.as_deref().unwrap_or(chunk);
        let mut events = Vec::new();
        let before = self.current_sample;
        let res = self.process_chunk_inner(chunk, &mut events);
        // The diarizer hears the samples the pipeline accepted, so the two clocks agree even
        // when inner failed after advancing `current_sample` (an `end_utterance` Whisper
        // error): feeding nothing would skew every later label until flush, feeding the whole
        // chunk would run it ahead. `current_sample` only moves forward inside inner, so the
        // subtraction is exact; on success `accepted` is the whole chunk.
        let accepted = (self.current_sample - before).min(chunk.len() as u64) as usize;
        if accepted > 0 {
            self.diarize(&chunk[..accepted], &mut events);
        }
        for ev in &events {
            self.enqueue_event(ev.clone());
        }
        res?;
        Ok(events)
    }

    /// Feed the attached diarizer `chunk`, register the utterances transcribed meanwhile (those
    /// already in `events`) and append the labels that came due.
    fn diarize(&mut self, chunk: &[f32], events: &mut Vec<AudioPipelineEvent>) {
        self.register_transcribed(events);
        let Some(d) = &mut self.diarizer else {
            return;
        };
        d.push_audio(chunk);
        d.poll(events);
    }

    /// Register every `UtteranceTranscribed` in `events` with the diarizer.
    fn register_transcribed(&mut self, events: &[AudioPipelineEvent]) {
        let Some(d) = &mut self.diarizer else {
            return;
        };
        for ev in events {
            if let AudioPipelineEvent::UtteranceTranscribed {
                text,
                start_ms,
                end_ms,
                ..
            } = ev
            {
                d.register(text.clone(), *start_ms, *end_ms);
            }
        }
    }

    /// Whether a speaker diarizer is attached and running (it stops, with a warning, if it
    /// fails; the pipeline then keeps going without labels).
    pub fn has_diarizer(&self) -> bool {
        self.diarizer.as_ref().is_some_and(|d| d.is_live())
    }

    /// Register an utterance transcribed outside the pipeline (a custom recognizer, or
    /// `auto_transcribe` off with [`Self::transcribe_pcm`] called by the caller) so it gets an
    /// [`AudioPipelineEvent::UtteranceLabeled`] like the ones the pipeline transcribes itself.
    /// `start_ms` and `end_ms` are on the pipeline's own clock, as in `UtteranceTranscribed`.
    /// Returns whether a diarizer will label it (false when none is attached, it has stopped,
    /// or `max_pending` utterances are already waiting on a poll).
    pub fn add_utterance(&mut self, text: String, start_ms: f32, end_ms: f32) -> bool {
        self.diarizer
            .as_mut()
            .is_some_and(|d| d.register(text, start_ms, end_ms))
    }

    fn process_chunk_inner(
        &mut self,
        chunk: &[f32],
        events: &mut Vec<AudioPipelineEvent>,
    ) -> Result<()> {
        let mut remaining = chunk;
        let max_samples = self.config.max_utterance_ms * 16;
        // Loop-invariant: attachment presence and config never change mid-chunk.
        let has_vad = self.vad.is_some() && self.vad_iter.is_some();
        let history_limit =
            (self.config.pre_roll_ms + self.config.vad_config.speech_pad_ms) * 16 + 512;
        while !remaining.is_empty() {
            if self.state == AudioPipelineState::ListeningForHotword {
                let Some(hotword) = &mut self.hotword else {
                    self.state = AudioPipelineState::ListeningForSpeech;
                    continue;
                };
                let count = remaining.len().min(hotword.samples_until_hop());
                let (part, rest) = remaining.split_at(count);
                let detected = hotword.process_chunk(part)?;
                self.current_sample += count as u64;
                remaining = rest;
                if let Some(detected) = detected {
                    self.hotword_paused_at = self.current_sample;
                    events.push(AudioPipelineEvent::WakeWordDetected {
                        keyword: detected.keyword,
                        confidence: detected.confidence,
                        timestamp_ms: sample_ms(self.current_sample),
                        sample_offset: self.current_sample,
                    });
                    let pre_roll = (self.config.pre_roll_ms * 16)
                        .min(48_000)
                        .min(max_samples.saturating_sub(512))
                        .min(self.current_sample as usize)
                        .min(hotword.available_samples());
                    self.speech_history = hotword.read_last_samples(pre_roll).into();
                    self.utterance_buffer.clear();
                    self.reset_vad_clock();
                    self.state = AudioPipelineState::ListeningForSpeech;
                }
                continue;
            }

            if !has_vad && self.state == AudioPipelineState::ListeningForSpeech {
                self.current_utterance_start_sample = self
                    .current_sample
                    .saturating_sub(self.speech_history.len() as u64);
                self.utterance_buffer.extend(self.speech_history.drain(..));
                self.state = AudioPipelineState::SpeechActive;
                events.push(AudioPipelineEvent::SpeechStart {
                    sample: self.current_utterance_start_sample,
                    ms: sample_ms(self.current_utterance_start_sample),
                });
            }

            let mut count = remaining.len();
            if let Some(vad_iter) = &self.vad_iter {
                count = count.min(vad_iter.samples_until_window());
            }
            if self.state == AudioPipelineState::SpeechActive {
                // Saturating: the cap is enforced below, so a buffer already at
                // the cap yields an empty part and the utterance-end path clears
                // it instead of stalling or underflowing here.
                count = count.min(max_samples.saturating_sub(self.utterance_buffer.len()));
            }
            let (part, rest) = remaining.split_at(count);
            if has_vad {
                self.speech_history.extend(part.iter().copied());
                let excess = self.speech_history.len().saturating_sub(history_limit);
                self.speech_history.drain(..excess);
            }
            if self.state == AudioPipelineState::SpeechActive {
                self.utterance_buffer.extend_from_slice(part);
            }
            let vad_event = if let (Some(vad), Some(vad_iter)) = (&mut self.vad, &mut self.vad_iter)
            {
                vad_iter.process_chunk(vad, part)?
            } else {
                None
            };
            self.current_sample += count as u64;
            remaining = rest;

            match vad_event {
                Some(VadEvent::SpeechStart { sample, .. }) => {
                    let start = (self.vad_sample_offset + sample)
                        .saturating_sub((self.config.pre_roll_ms * 16) as u64)
                        .max(
                            self.current_sample
                                .saturating_sub(self.speech_history.len() as u64),
                        )
                        .max(self.current_sample.saturating_sub(max_samples as u64));
                    // `start` is clamped to `<= current_sample` above and to the
                    // retained history window, so these saturate only if a future
                    // VAD stride ever breaks that clock invariant (cf. SpeechEnd).
                    let samples = (self.current_sample.saturating_sub(start) as usize)
                        .min(self.speech_history.len());
                    self.utterance_buffer.clear();
                    self.utterance_buffer.extend(
                        self.speech_history
                            .iter()
                            .skip(self.speech_history.len() - samples)
                            .copied(),
                    );
                    self.current_utterance_start_sample = start;
                    self.state = AudioPipelineState::SpeechActive;
                    events.push(AudioPipelineEvent::SpeechStart {
                        sample: start,
                        ms: sample_ms(start),
                    });
                }
                Some(VadEvent::SpeechEnd { end_sample, .. })
                    if self.state == AudioPipelineState::SpeechActive =>
                {
                    let end = (self.vad_sample_offset + end_sample)
                        .clamp(self.current_utterance_start_sample, self.current_sample);
                    self.utterance_buffer
                        .truncate((end - self.current_utterance_start_sample) as usize);
                    self.end_utterance(events, end, false)?;
                }
                _ => {}
            }
            if self.state == AudioPipelineState::SpeechActive
                && self.utterance_buffer.len() >= max_samples
            {
                self.end_utterance(events, self.current_sample, has_vad)?;
            }
        }
        Ok(())
    }

    fn reset_vad_clock(&mut self) {
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
        if let Some(vad_iter) = &mut self.vad_iter {
            vad_iter.reset();
        }
        self.vad_sample_offset = self.current_sample;
    }

    fn end_utterance(
        &mut self,
        events: &mut Vec<AudioPipelineEvent>,
        end: u64,
        continue_speech: bool,
    ) -> Result<()> {
        events.push(AudioPipelineEvent::SpeechEnd {
            start_sample: self.current_utterance_start_sample,
            end_sample: end,
            start_ms: sample_ms(self.current_utterance_start_sample),
            end_ms: sample_ms(end),
        });
        self.finish_utterance(events, end, continue_speech)
    }

    /// Flush any active speech segment at the end of the audio stream.
    ///
    /// Closes in-flight speech boundaries, runs transcription if configured, and returns
    /// terminal events.
    pub fn flush(&mut self) -> Result<Vec<AudioPipelineEvent>> {
        let mut events = Vec::new();
        let res = self.flush_inner(&mut events);
        // End of audio for the diarizer too: it labels everything still waiting, then a new
        // session starts at the current sample.
        self.register_transcribed(&events);
        let at = self.current_sample;
        if let Some(d) = &mut self.diarizer {
            d.finish_and_restart(&mut events, at);
        }
        for ev in &events {
            self.enqueue_event(ev.clone());
        }
        res?;
        Ok(events)
    }

    fn flush_inner(&mut self, events: &mut Vec<AudioPipelineEvent>) -> Result<()> {
        if self.state == AudioPipelineState::SpeechActive {
            let end = match self.vad_iter.as_mut().and_then(|iter| iter.flush()) {
                Some(VadEvent::SpeechEnd { end_sample, .. }) => (self.vad_sample_offset
                    + end_sample)
                    .clamp(self.current_utterance_start_sample, self.current_sample),
                _ => self.current_sample,
            };
            self.utterance_buffer
                .truncate((end - self.current_utterance_start_sample) as usize);
            let result = self.end_utterance(events, end, false);
            self.reset_vad_clock();
            self.speech_history.clear();
            result?;
        }
        Ok(())
    }

    /// Complete one bounded utterance while retaining the continuous VAD clock.
    fn finish_utterance(
        &mut self,
        events: &mut Vec<AudioPipelineEvent>,
        end_sample: u64,
        continue_speech: bool,
    ) -> Result<()> {
        let has_audio = !self.utterance_buffer.is_empty();
        if has_audio {
            std::mem::swap(&mut self.last_utterance, &mut self.utterance_buffer);
            self.utterance_buffer.clear();
            if self.utterance_buffer.capacity() < 32_000 {
                self.utterance_buffer
                    .reserve(32_000 - self.utterance_buffer.capacity());
            }
        }
        let mut transcription_result = Ok(());
        if has_audio
            && self.config.auto_transcribe
            && let (Some(whisper), Some(tokenizer)) = (&self.whisper, &self.whisper_tokenizer)
        {
            self.state = AudioPipelineState::Transcribing;
            let mut opts = self.config.whisper_opts.clone().unwrap_or_default();
            opts.cancel = Some(self.cancel.clone());
            match whisper.transcribe(tokenizer, &self.last_utterance, &opts) {
                Ok(text) => events.push(AudioPipelineEvent::UtteranceTranscribed {
                    text,
                    start_ms: sample_ms(self.current_utterance_start_sample),
                    end_ms: sample_ms(end_sample),
                    sample_count: self.last_utterance.len(),
                }),
                Err(e) if e.to_string().contains("cancelled") => {
                    tracing::debug!("Whisper transcription was cancelled cooperatively");
                }
                Err(e) => transcription_result = Err(e),
            }
        }
        if continue_speech {
            self.state = AudioPipelineState::SpeechActive;
            self.current_utterance_start_sample = end_sample;
            events.push(AudioPipelineEvent::SpeechStart {
                sample: end_sample,
                ms: sample_ms(end_sample),
            });
        } else {
            self.state = if self.config.require_hotword && self.hotword.is_some() {
                self.speech_history.clear();
                // KWS was paused during speech. Its old ring and local clock
                // cannot describe a contiguous prefix of the next wake cycle.
                if let Some(hotword) = &mut self.hotword {
                    hotword.resume_after_pause(self.current_sample - self.hotword_paused_at);
                }
                AudioPipelineState::ListeningForHotword
            } else {
                AudioPipelineState::ListeningForSpeech
            };
        }
        transcription_result
    }

    /// Transcribe an arbitrary buffer of 16 kHz mono PCM audio samples using the attached Whisper model.
    pub fn transcribe_pcm(&self, pcm: &[f32]) -> Result<String> {
        let whisper = self
            .whisper
            .as_ref()
            .context("Whisper model is not attached")?;
        let tokenizer = self
            .whisper_tokenizer
            .as_ref()
            .context("Whisper tokenizer is not attached")?;

        let mut opts = self.config.whisper_opts.clone().unwrap_or_default();
        opts.cancel = Some(self.cancel.clone());

        whisper.transcribe(tokenizer, pcm, &opts)
    }
}

fn sample_ms(sample: u64) -> f32 {
    (sample as f64 / 16.0) as f32
}

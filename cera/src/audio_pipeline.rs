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

use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::hotword::{HotwordConfig, HotwordDetector, HotwordIterator};
use crate::model::whisper::{WhisperModel, WhisperTranscribeOpts};
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
    /// Keyword Spotting configuration.
    pub hotword_config: Option<HotwordConfig>,
    /// Whisper transcription options.
    pub whisper_opts: Option<WhisperTranscribeOpts>,
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
    whisper: Option<WhisperModel>,
    whisper_tokenizer: Option<BpeTokenizer>,
    config: Option<AudioPipelineConfig>,
    cancel: Option<Arc<AtomicBool>>,
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

    /// Attach a custom VAD configuration.
    pub fn with_vad_config(mut self, config: VadConfig) -> Self {
        self.vad_config = Some(config);
        self
    }

    /// Attach a Keyword Spotting detector.
    pub fn with_hotword(
        mut self,
        detector: HotwordDetector,
        config: Option<HotwordConfig>,
    ) -> Self {
        let cfg = config.unwrap_or_else(|| detector.default_config());
        self.hotword = Some(HotwordIterator::new(detector, None, cfg));
        self
    }

    /// Attach an existing Keyword Spotting iterator.
    pub fn with_hotword_iterator(mut self, iterator: HotwordIterator) -> Self {
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
        let cfg = config.unwrap_or_else(|| detector.default_config());
        self.hotword = Some(HotwordIterator::new(detector, None, cfg));
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
        let cfg = config.unwrap_or_else(|| detector.default_config());
        self.hotword = Some(HotwordIterator::new(detector, None, cfg));
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

        let vad_config = self.vad_config.unwrap_or(config.vad_config.clone());
        let vad_sample_rate = self.vad_sample_rate.unwrap_or(VadSampleRate::Rate16kHz);
        let vad_iter = if self.vad.is_some() {
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

        Ok(AudioPipeline {
            vad: self.vad,
            vad_iter,
            hotword: self.hotword,
            whisper: self.whisper,
            whisper_tokenizer: self.whisper_tokenizer,
            config,
            state: initial_state,
            current_sample: 0,
            utterance_buffer: Vec::with_capacity(32_000),
            last_utterance: Vec::new(),
            current_utterance_start_sample: 0,
            pending_events: VecDeque::with_capacity(8),
            cancel,
        })
    }
}

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
    utterance_buffer: Vec<f32>,
    last_utterance: Vec<f32>,
    current_utterance_start_sample: u64,
    pending_events: VecDeque<AudioPipelineEvent>,
    cancel: Arc<AtomicBool>,
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
        self.utterance_buffer.clear();
        self.last_utterance.clear();
        self.current_utterance_start_sample = 0;
        self.pending_events.clear();
        self.cancel.store(false, Ordering::Relaxed);
        self.state = if self.config.require_hotword && self.hotword.is_some() {
            AudioPipelineState::ListeningForHotword
        } else {
            AudioPipelineState::ListeningForSpeech
        };
    }

    /// Process a streaming chunk of 16 kHz mono PCM audio samples.
    ///
    /// Evaluates wake words, speech boundaries, and automatic transcription according to
    /// the active state machine, returning all newly triggered pipeline events.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<Vec<AudioPipelineEvent>> {
        if chunk.is_empty() {
            return Ok(Vec::new());
        }

        let mut events = Vec::new();
        let chunk_len = chunk.len() as u64;

        // Sanitize incoming PCM samples against non-finite values (NaN / Inf)
        let sanitized: Option<Vec<f32>> = if chunk.iter().any(|s| !s.is_finite()) {
            Some(
                chunk
                    .iter()
                    .map(|&s| if s.is_finite() { s } else { 0.0 })
                    .collect(),
            )
        } else {
            None
        };
        let chunk_slice = sanitized.as_deref().unwrap_or(chunk);

        match self.state {
            AudioPipelineState::ListeningForHotword => {
                if let Some(hotword) = &mut self.hotword {
                    let maybe_event = hotword.process_chunk(chunk_slice)?;
                    self.current_sample = self.current_sample.saturating_add(chunk_len);

                    if let Some(hw_ev) = maybe_event {
                        let ev = AudioPipelineEvent::WakeWordDetected {
                            keyword: hw_ev.keyword,
                            confidence: hw_ev.confidence,
                            timestamp_ms: hw_ev.timestamp_ms,
                            sample_offset: hw_ev.sample_offset,
                        };
                        events.push(ev);

                        // Extract pre-roll audio from the circular buffer
                        let pre_roll_samples = (self.config.pre_roll_ms * 16).min(48_000);
                        let pre_roll = hotword.read_last_samples(pre_roll_samples);

                        self.utterance_buffer.clear();
                        self.utterance_buffer.extend_from_slice(&pre_roll);
                        self.current_utterance_start_sample =
                            self.current_sample.saturating_sub(pre_roll.len() as u64);

                        // Reset VAD so speech tracking starts fresh on wake word transition
                        if let Some(vad) = &mut self.vad {
                            vad.reset();
                        }
                        if let Some(vad_iter) = &mut self.vad_iter {
                            vad_iter.reset();
                        }

                        self.state = AudioPipelineState::ListeningForSpeech;
                    }
                } else {
                    // No hotword attached; transition directly to speech listening
                    self.state = AudioPipelineState::ListeningForSpeech;
                    return self.process_chunk(chunk);
                }
            }

            AudioPipelineState::ListeningForSpeech => {
                if let (Some(vad), Some(vad_iter)) = (&mut self.vad, &mut self.vad_iter) {
                    let initial_event = vad_iter.process_chunk(vad, chunk_slice)?;
                    self.current_sample = self.current_sample.saturating_add(chunk_len);

                    let mut vad_events = Vec::new();
                    if let Some(ev) = initial_event {
                        vad_events.push(ev);
                    }
                    while let Some(ev) = vad_iter.pop_event() {
                        vad_events.push(ev);
                    }

                    for v_ev in vad_events {
                        match v_ev {
                            VadEvent::SpeechStart { sample, ms } => {
                                self.state = AudioPipelineState::SpeechActive;
                                events.push(AudioPipelineEvent::SpeechStart { sample, ms });
                                self.current_utterance_start_sample = sample;
                                self.utterance_buffer.extend_from_slice(chunk_slice);
                            }
                            VadEvent::SpeechEnd {
                                start_sample,
                                end_sample,
                                start_ms,
                                end_ms,
                            } => {
                                if self.state == AudioPipelineState::SpeechActive {
                                    events.push(AudioPipelineEvent::SpeechEnd {
                                        start_sample,
                                        end_sample,
                                        start_ms,
                                        end_ms,
                                    });
                                    self.finish_utterance(&mut events, true)?;
                                }
                            }
                        }
                    }

                    // Keep pre-roll window populated while awaiting speech start
                    if self.state == AudioPipelineState::ListeningForSpeech {
                        let pre_roll_samples = (self.config.pre_roll_ms * 16).min(48_000);
                        self.utterance_buffer.extend_from_slice(chunk_slice);
                        if self.utterance_buffer.len() > pre_roll_samples {
                            let excess = self.utterance_buffer.len() - pre_roll_samples;
                            self.utterance_buffer.drain(..excess);
                        }
                    }
                } else {
                    // No VAD attached: treat incoming audio as immediate speech onset
                    self.state = AudioPipelineState::SpeechActive;
                    let ms = (self.current_sample as f64 * 1000.0 / 16000.0) as f32;
                    events.push(AudioPipelineEvent::SpeechStart {
                        sample: self.current_sample,
                        ms,
                    });
                    self.current_utterance_start_sample = self.current_sample;
                    self.current_sample = self.current_sample.saturating_add(chunk_len);
                    self.utterance_buffer.extend_from_slice(chunk_slice);

                    let max_samples = (self.config.max_utterance_ms * 16).max(16_000);
                    if self.utterance_buffer.len() >= max_samples {
                        let start_s = self.current_utterance_start_sample;
                        let end_s = self.current_sample;
                        let start_m = (start_s as f64 * 1000.0 / 16000.0) as f32;
                        let end_m = (end_s as f64 * 1000.0 / 16000.0) as f32;

                        events.push(AudioPipelineEvent::SpeechEnd {
                            start_sample: start_s,
                            end_sample: end_s,
                            start_ms: start_m,
                            end_ms: end_m,
                        });

                        self.finish_utterance(&mut events, true)?;
                    }
                }
            }

            AudioPipelineState::SpeechActive => {
                self.utterance_buffer.extend_from_slice(chunk_slice);

                let max_samples = (self.config.max_utterance_ms * 16).max(16_000);
                let duration_exceeded = self.utterance_buffer.len() >= max_samples;

                if let (Some(vad), Some(vad_iter)) = (&mut self.vad, &mut self.vad_iter) {
                    let initial_event = vad_iter.process_chunk(vad, chunk_slice)?;
                    self.current_sample = self.current_sample.saturating_add(chunk_len);

                    let mut speech_ended = false;
                    let mut end_payload = None;

                    if let Some(VadEvent::SpeechEnd {
                        start_sample,
                        end_sample,
                        start_ms,
                        end_ms,
                    }) = initial_event
                    {
                        speech_ended = true;
                        end_payload = Some((start_sample, end_sample, start_ms, end_ms));
                    }

                    while let Some(ev) = vad_iter.pop_event() {
                        if let VadEvent::SpeechEnd {
                            start_sample,
                            end_sample,
                            start_ms,
                            end_ms,
                        } = ev
                        {
                            speech_ended = true;
                            end_payload = Some((start_sample, end_sample, start_ms, end_ms));
                        }
                    }

                    if speech_ended || duration_exceeded {
                        let (start_sample, end_sample, start_ms, end_ms) = match end_payload {
                            Some((start_s, end_s, _, end_m)) => {
                                let clamped_start =
                                    start_s.max(self.current_utterance_start_sample);
                                let clamped_start_m =
                                    (clamped_start as f64 * 1000.0 / 16000.0) as f32;
                                (clamped_start, end_s, clamped_start_m, end_m)
                            }
                            None => {
                                let end_s = self.current_sample;
                                let start_s = self.current_utterance_start_sample;
                                let start_m = (start_s as f64 * 1000.0 / 16000.0) as f32;
                                let end_m = (end_s as f64 * 1000.0 / 16000.0) as f32;
                                (start_s, end_s, start_m, end_m)
                            }
                        };

                        events.push(AudioPipelineEvent::SpeechEnd {
                            start_sample,
                            end_sample,
                            start_ms,
                            end_ms,
                        });

                        let reset_vad = speech_ended;
                        self.finish_utterance(&mut events, reset_vad)?;
                    }
                } else {
                    self.current_sample = self.current_sample.saturating_add(chunk_len);
                    if duration_exceeded {
                        let start_s = self.current_utterance_start_sample;
                        let end_s = self.current_sample;
                        let start_m = (start_s as f64 * 1000.0 / 16000.0) as f32;
                        let end_m = (end_s as f64 * 1000.0 / 16000.0) as f32;

                        events.push(AudioPipelineEvent::SpeechEnd {
                            start_sample: start_s,
                            end_sample: end_s,
                            start_ms: start_m,
                            end_ms: end_m,
                        });

                        self.finish_utterance(&mut events, true)?;
                    }
                }
            }

            AudioPipelineState::Transcribing => {
                // If chunk arrives while already transcribing, queue sample position advance
                self.current_sample = self.current_sample.saturating_add(chunk_len);
            }
        }

        // Store any excess events in pending queue
        for ev in &events {
            self.pending_events.push_back(ev.clone());
        }

        Ok(events)
    }

    /// Flush any active speech segment at the end of the audio stream.
    ///
    /// Closes in-flight speech boundaries, runs transcription if configured, and returns
    /// terminal events.
    pub fn flush(&mut self) -> Result<Vec<AudioPipelineEvent>> {
        let mut events = Vec::new();

        if self.state == AudioPipelineState::SpeechActive {
            let mut end_payload = None;
            if let Some(vad_iter) = &mut self.vad_iter
                && let Some(VadEvent::SpeechEnd {
                    start_sample,
                    end_sample,
                    start_ms,
                    end_ms,
                }) = vad_iter.flush()
            {
                end_payload = Some((start_sample, end_sample, start_ms, end_ms));
            }

            let (start_sample, end_sample, start_ms, end_ms) = end_payload.unwrap_or_else(|| {
                let end_s = self.current_sample;
                let start_s = self.current_utterance_start_sample;
                let start_m = (start_s as f64 * 1000.0 / 16000.0) as f32;
                let end_m = (end_s as f64 * 1000.0 / 16000.0) as f32;
                (start_s, end_s, start_m, end_m)
            });

            events.push(AudioPipelineEvent::SpeechEnd {
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            });

            self.finish_utterance(&mut events, true)?;
        }

        for ev in &events {
            self.pending_events.push_back(ev.clone());
        }

        Ok(events)
    }

    /// Internal helper: finalize an utterance buffer, perform optional Whisper transcription,
    /// and transition back to listening or continue active speech.
    fn finish_utterance(
        &mut self,
        events: &mut Vec<AudioPipelineEvent>,
        reset_vad: bool,
    ) -> Result<()> {
        self.last_utterance = std::mem::take(&mut self.utterance_buffer);
        let sample_count = self.last_utterance.len();

        if self.config.auto_transcribe
            && let (Some(whisper), Some(tokenizer)) = (&self.whisper, &self.whisper_tokenizer)
        {
            self.state = AudioPipelineState::Transcribing;

            let mut opts = self.config.whisper_opts.clone().unwrap_or_default();
            opts.cancel = Some(self.cancel.clone());

            match whisper.transcribe(tokenizer, &self.last_utterance, &opts) {
                Ok(text) => {
                    let start_ms =
                        (self.current_utterance_start_sample as f64 * 1000.0 / 16000.0) as f32;
                    let end_ms = (self.current_sample as f64 * 1000.0 / 16000.0) as f32;

                    events.push(AudioPipelineEvent::UtteranceTranscribed {
                        text,
                        start_ms,
                        end_ms,
                        sample_count,
                    });
                }
                Err(e) => {
                    // Check if failure was cooperative cancellation
                    if e.to_string().contains("cancelled") {
                        tracing::debug!("Whisper transcription was cancelled cooperatively");
                    } else {
                        self.state = if reset_vad {
                            if self.config.require_hotword && self.hotword.is_some() {
                                AudioPipelineState::ListeningForHotword
                            } else {
                                AudioPipelineState::ListeningForSpeech
                            }
                        } else {
                            self.current_utterance_start_sample = self.current_sample;
                            AudioPipelineState::SpeechActive
                        };
                        if reset_vad {
                            if let Some(vad) = &mut self.vad {
                                vad.reset();
                            }
                            if let Some(vad_iter) = &mut self.vad_iter {
                                vad_iter.reset();
                            }
                        }
                        return Err(e);
                    }
                }
            }
        }

        if reset_vad {
            // Return to listening state
            self.state = if self.config.require_hotword && self.hotword.is_some() {
                AudioPipelineState::ListeningForHotword
            } else {
                AudioPipelineState::ListeningForSpeech
            };

            if let Some(vad) = &mut self.vad {
                vad.reset();
            }
            if let Some(vad_iter) = &mut self.vad_iter {
                vad_iter.reset();
            }
        } else {
            // Speech is still active across max utterance duration cutoff:
            // cycle the utterance buffer while preserving the VAD model's internal
            // hidden states and trigger state so continuations do not lose leading phonemes.
            self.state = AudioPipelineState::SpeechActive;
            self.current_utterance_start_sample = self.current_sample;
            let ms = (self.current_sample as f64 * 1000.0 / 16000.0) as f32;
            events.push(AudioPipelineEvent::SpeechStart {
                sample: self.current_sample,
                ms,
            });
        }

        Ok(())
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

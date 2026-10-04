//! UniFFI lowering for the unified AudioPipeline facade.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::{FfiError, FfiHotwordConfig, FfiVadConfig, FfiWhisperTranscribeOpts};

#[cfg(test)]
mod tests;

/// Active state of the streaming audio pipeline.
#[derive(uniffi::Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiAudioPipelineState {
    /// Awaiting a keyword spotting wake word before activating speech recording.
    ListeningForHotword,
    /// Evaluating incoming audio frames to detect speech onset.
    ListeningForSpeech,
    /// Speech onset detected; accumulating utterance samples in the audio buffer.
    SpeechActive,
    /// Transcribing the accumulated speech utterance using Whisper.
    Transcribing,
}

impl From<cera::audio_pipeline::AudioPipelineState> for FfiAudioPipelineState {
    fn from(s: cera::audio_pipeline::AudioPipelineState) -> Self {
        match s {
            cera::audio_pipeline::AudioPipelineState::ListeningForHotword => {
                Self::ListeningForHotword
            }
            cera::audio_pipeline::AudioPipelineState::ListeningForSpeech => {
                Self::ListeningForSpeech
            }
            cera::audio_pipeline::AudioPipelineState::SpeechActive => Self::SpeechActive,
            cera::audio_pipeline::AudioPipelineState::Transcribing => Self::Transcribing,
        }
    }
}

/// An event emitted by the unified audio pipeline.
#[derive(uniffi::Enum, Debug, Clone, PartialEq)]
pub enum FfiAudioPipelineEvent {
    /// Keyword spotting detected a wake word.
    WakeWordDetected {
        /// Triggered keyword.
        keyword: String,
        /// Confidence probability between 0.0 and 1.0.
        confidence: f32,
        /// Timestamp in milliseconds from stream start.
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
        sample_count: u64,
    },
    /// The attached speaker diarizer has covered an utterance and assigned it a speaker. One per
    /// utterance, after its `UtteranceTranscribed`: a chunk plus its lookahead later (seconds with
    /// the default preset). Needs a pipeline built with `from_files_with_diarizer`.
    UtteranceLabeled {
        /// The utterance text, as in its `UtteranceTranscribed` event.
        text: String,
        /// Start timestamp of the utterance in milliseconds.
        start_ms: f32,
        /// End timestamp of the utterance in milliseconds.
        end_ms: f32,
        /// The most active speaker's slot (0 to 3), or `None` when no speaker was active over the
        /// span or the labeler had to give the utterance up (see `dropped`).
        speaker: Option<u32>,
        /// The speaker's share of all speakers' active time over the span, in (0, 1].
        confidence: Option<f32>,
        /// A second speaker who was also clearly active over the span, if any.
        overlapping: Option<u32>,
        /// True when the labeler gave the utterance up instead of labeling it (history expiry,
        /// queue overflow, or non-finite times): `None` speaker with `dropped` set is a stalled
        /// diarizer, not silence.
        dropped: bool,
    },
}

impl From<cera::audio_pipeline::AudioPipelineEvent> for FfiAudioPipelineEvent {
    fn from(ev: cera::audio_pipeline::AudioPipelineEvent) -> Self {
        match ev {
            cera::audio_pipeline::AudioPipelineEvent::WakeWordDetected {
                keyword,
                confidence,
                timestamp_ms,
                sample_offset,
            } => Self::WakeWordDetected {
                keyword,
                confidence,
                timestamp_ms,
                sample_offset,
            },
            cera::audio_pipeline::AudioPipelineEvent::SpeechStart { sample, ms } => {
                Self::SpeechStart { sample, ms }
            }
            cera::audio_pipeline::AudioPipelineEvent::SpeechEnd {
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            } => Self::SpeechEnd {
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            },
            cera::audio_pipeline::AudioPipelineEvent::UtteranceTranscribed {
                text,
                start_ms,
                end_ms,
                sample_count,
            } => Self::UtteranceTranscribed {
                text,
                start_ms,
                end_ms,
                sample_count: sample_count as u64,
            },
            cera::audio_pipeline::AudioPipelineEvent::UtteranceLabeled {
                text,
                start_ms,
                end_ms,
                speaker,
                confidence,
                overlapping,
                dropped,
            } => Self::UtteranceLabeled {
                text,
                start_ms,
                end_ms,
                speaker,
                confidence,
                overlapping,
                dropped,
            },
        }
    }
}

/// Configuration options for the unified audio pipeline.
#[derive(uniffi::Record, Debug, Clone, PartialEq)]
pub struct FfiAudioPipelineConfig {
    /// Whether a keyword spotting wake word must be detected before speech tracking begins.
    pub require_hotword: bool,
    /// Whether to automatically run Whisper transcription upon speech completion.
    pub auto_transcribe: bool,
    /// Audio pre-roll duration in milliseconds to retain prior to wake word or speech onset.
    pub pre_roll_ms: u32,
    /// Maximum allowed utterance duration in milliseconds before forcing a boundary.
    pub max_utterance_ms: u32,
    /// Voice Activity Detection configuration.
    pub vad_config: Option<FfiVadConfig>,
    /// Keyword Spotting configuration.
    pub hotword_config: Option<FfiHotwordConfig>,
    /// Whisper transcription options.
    pub whisper_opts: Option<FfiWhisperTranscribeOpts>,
}

impl Default for FfiAudioPipelineConfig {
    fn default() -> Self {
        Self {
            require_hotword: false,
            auto_transcribe: true,
            pre_roll_ms: 200,
            max_utterance_ms: 30_000,
            vad_config: Some(FfiVadConfig::default()),
            hotword_config: None,
            whisper_opts: None,
        }
    }
}

impl From<FfiAudioPipelineConfig> for cera::audio_pipeline::AudioPipelineConfig {
    fn from(cfg: FfiAudioPipelineConfig) -> Self {
        Self {
            require_hotword: cfg.require_hotword,
            auto_transcribe: cfg.auto_transcribe,
            pre_roll_ms: cfg.pre_roll_ms as usize,
            max_utterance_ms: cfg.max_utterance_ms as usize,
            vad_config: cfg.vad_config.map(Into::into).unwrap_or_default(),
            hotword_config: cfg.hotword_config.map(Into::into),
            whisper_opts: cfg.whisper_opts.map(Into::into),
        }
    }
}

/// Returns default configuration for the audio pipeline.
#[uniffi::export]
pub fn audio_pipeline_default_config() -> FfiAudioPipelineConfig {
    FfiAudioPipelineConfig::default()
}

/// Unified audio facade coordinating VAD, Hotword, and Whisper ASR.
#[derive(uniffi::Object)]
pub struct FfiAudioPipeline {
    pub(crate) inner: Mutex<cera::audio_pipeline::AudioPipeline>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// Whether the diarizer was staged on the Hexagon NPU (false: the CPU, or no diarizer).
    pub(crate) diarizer_on_npu: bool,
}

impl FfiAudioPipeline {
    /// Stage `model` on the Hexagon NPU when asked and possible; reports whether it runs
    /// there (false on builds without the `hexagon` feature, or when staging fails).
    fn stage_diarizer(
        model: &cera::model::sortformer::SortformerModel,
        window_frames: usize,
        prefer_npu: bool,
    ) -> bool {
        #[cfg(feature = "hexagon")]
        {
            prefer_npu
                && cera::model::sortformer_hexagon::try_hexagon_sortformer(model, window_frames)
                    .is_some()
        }
        #[cfg(not(feature = "hexagon"))]
        {
            let _ = (model, window_frames, prefer_npu);
            false
        }
    }

    fn build_from_files(
        vad_path: Option<String>,
        hotword_path: Option<String>,
        whisper_path: Option<String>,
        diarizer: Option<(String, bool)>,
        config: Option<FfiAudioPipelineConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        let mut builder = cera::audio_pipeline::AudioPipelineBuilder::new();
        if let Some(cfg) = config {
            builder = builder.with_config(cfg.into());
        }
        if let Some(vp) = vad_path {
            builder = builder
                .with_vad_from_file(&vp)
                .map_err(|e| FfiError::Backend {
                    detail: format!("failed to load VAD model from {vp}: {e}"),
                })?;
        }
        if let Some(hp) = hotword_path {
            builder = builder
                .with_hotword_from_file(&hp, None)
                .map_err(|e| FfiError::Backend {
                    detail: format!("failed to load Hotword model from {hp}: {e}"),
                })?;
        }
        if let Some(wp) = whisper_path {
            builder = builder
                .with_whisper_from_file(&wp)
                .map_err(|e| FfiError::Backend {
                    detail: format!("failed to load Whisper model from {wp}: {e}"),
                })?;
        }
        let diarizer_on_npu = match diarizer {
            None => false,
            Some((dp, prefer_npu)) => {
                let model =
                    cera::model::sortformer::SortformerModel::from_file(&dp).map_err(|e| {
                        FfiError::Backend {
                            detail: format!(
                                "failed to load the Sortformer diarizer from {dp}: {e:#}"
                            ),
                        }
                    })?;
                let params = model.default_streaming().clone();
                let on_npu = Self::stage_diarizer(&model, params.window_frames(), prefer_npu);
                builder = builder.with_diarizer(model, params);
                on_npu
            }
        };
        let pipeline = builder.build().map_err(|e| FfiError::Backend {
            detail: format!("failed to build audio pipeline: {e}"),
        })?;
        let cancel = pipeline.cancel_handle();
        Ok(Arc::new(Self {
            inner: Mutex::new(pipeline),
            cancel,
            diarizer_on_npu,
        }))
    }

    fn build_from_bytes(
        vad_bytes: Option<Vec<u8>>,
        hotword_bytes: Option<Vec<u8>>,
        whisper_bytes: Option<Vec<u8>>,
        diarizer: Option<(Vec<u8>, bool)>,
        config: Option<FfiAudioPipelineConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        let mut builder = cera::audio_pipeline::AudioPipelineBuilder::new();
        if let Some(cfg) = config {
            builder = builder.with_config(cfg.into());
        }
        if let Some(vb) = vad_bytes {
            builder = builder
                .with_vad_from_bytes(vb)
                .map_err(|e| FfiError::Backend {
                    detail: format!("failed to load VAD model from bytes: {e}"),
                })?;
        }
        if let Some(hb) = hotword_bytes {
            builder = builder
                .with_hotword_from_bytes(hb, None)
                .map_err(|e| FfiError::Backend {
                    detail: format!("failed to load Hotword model from bytes: {e}"),
                })?;
        }
        if let Some(wb) = whisper_bytes {
            builder = builder
                .with_whisper_from_bytes(wb)
                .map_err(|e| FfiError::Backend {
                    detail: format!("failed to load Whisper model from bytes: {e}"),
                })?;
        }
        let diarizer_on_npu = match diarizer {
            None => false,
            Some((db, prefer_npu)) => {
                let model =
                    cera::model::sortformer::SortformerModel::from_bytes(db).map_err(|e| {
                        FfiError::Backend {
                            detail: format!(
                                "failed to load the Sortformer diarizer from bytes: {e:#}"
                            ),
                        }
                    })?;
                let params = model.default_streaming().clone();
                let on_npu = Self::stage_diarizer(&model, params.window_frames(), prefer_npu);
                builder = builder.with_diarizer(model, params);
                on_npu
            }
        };
        let pipeline = builder.build().map_err(|e| FfiError::Backend {
            detail: format!("failed to build audio pipeline: {e}"),
        })?;
        let cancel = pipeline.cancel_handle();
        Ok(Arc::new(Self {
            inner: Mutex::new(pipeline),
            cancel,
            diarizer_on_npu,
        }))
    }

    fn lock_inner(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, cera::audio_pipeline::AudioPipeline>, FfiError> {
        self.inner.lock().map_err(|_| FfiError::Backend {
            detail: "FfiAudioPipeline inner mutex poisoned".to_string(),
        })
    }
}

#[uniffi::export]
impl FfiAudioPipeline {
    /// Construct a pipeline from filesystem model paths.
    #[uniffi::constructor]
    pub fn from_files(
        vad_path: Option<String>,
        hotword_path: Option<String>,
        whisper_path: Option<String>,
        config: Option<FfiAudioPipelineConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        Self::build_from_files(vad_path, hotword_path, whisper_path, None, config)
    }

    /// Construct a pipeline from filesystem model paths with a Sortformer speaker diarizer
    /// (`diarizer_path`, a converted Sortformer GGUF). Every transcribed utterance then gets an
    /// `UtteranceLabeled` event with its speaker, once the diarizer has covered it.
    ///
    /// With `prefer_npu` the diarizer runs on the Hexagon NPU when this build has it and the
    /// device offers it (the GGUF must have been converted with `--tail-outtype q8_0`); otherwise,
    /// or if staging fails, it runs on the CPU. `diarizer_on_npu()` says which.
    #[uniffi::constructor]
    pub fn from_files_with_diarizer(
        vad_path: Option<String>,
        hotword_path: Option<String>,
        whisper_path: Option<String>,
        diarizer_path: String,
        prefer_npu: bool,
        config: Option<FfiAudioPipelineConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        Self::build_from_files(
            vad_path,
            hotword_path,
            whisper_path,
            Some((diarizer_path, prefer_npu)),
            config,
        )
    }

    /// Construct a pipeline from in-memory GGUF byte buffers.
    #[uniffi::constructor]
    pub fn from_bytes(
        vad_bytes: Option<Vec<u8>>,
        hotword_bytes: Option<Vec<u8>>,
        whisper_bytes: Option<Vec<u8>>,
        config: Option<FfiAudioPipelineConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        Self::build_from_bytes(vad_bytes, hotword_bytes, whisper_bytes, None, config)
    }

    /// Construct a pipeline from in-memory GGUF byte buffers with a Sortformer speaker
    /// diarizer (`diarizer_bytes`, a converted Sortformer GGUF). Every transcribed utterance
    /// then gets an `UtteranceLabeled` event with its speaker, once the diarizer covers it.
    ///
    /// With `prefer_npu` the diarizer runs on the Hexagon NPU when this build has it and the
    /// device offers it (the GGUF must have been converted with `--tail-outtype q8_0`);
    /// otherwise, or if staging fails, it runs on the CPU. `diarizer_on_npu()` says which.
    #[uniffi::constructor]
    pub fn from_bytes_with_diarizer(
        vad_bytes: Option<Vec<u8>>,
        hotword_bytes: Option<Vec<u8>>,
        whisper_bytes: Option<Vec<u8>>,
        diarizer_bytes: Vec<u8>,
        prefer_npu: bool,
        config: Option<FfiAudioPipelineConfig>,
    ) -> Result<Arc<Self>, FfiError> {
        Self::build_from_bytes(
            vad_bytes,
            hotword_bytes,
            whisper_bytes,
            Some((diarizer_bytes, prefer_npu)),
            config,
        )
    }

    /// Current lifecycle state of the pipeline.
    pub fn state(&self) -> Result<FfiAudioPipelineState, FfiError> {
        let pipeline = self.lock_inner()?;
        Ok(pipeline.state().into())
    }

    /// Whether speech activity is currently ongoing.
    pub fn is_speech_active(&self) -> Result<bool, FfiError> {
        let pipeline = self.lock_inner()?;
        Ok(pipeline.is_speech_active())
    }

    /// Whether the pipeline is currently awaiting a wake word trigger.
    pub fn is_listening_for_hotword(&self) -> Result<bool, FfiError> {
        let pipeline = self.lock_inner()?;
        Ok(pipeline.is_listening_for_hotword())
    }

    /// Total audio samples processed since start or reset.
    pub fn current_sample(&self) -> Result<u64, FfiError> {
        let pipeline = self.lock_inner()?;
        Ok(pipeline.current_sample())
    }

    /// Process a streaming chunk of 16 kHz mono PCM audio samples.
    pub fn process_chunk(&self, chunk: Vec<f32>) -> Result<Vec<FfiAudioPipelineEvent>, FfiError> {
        let mut pipeline = self.lock_inner()?;
        let events = pipeline
            .process_chunk(&chunk)
            .map_err(|e| FfiError::Backend {
                detail: format!("failed to process audio chunk: {e}"),
            })?;
        Ok(events.into_iter().map(Into::into).collect())
    }

    /// Flush any in-flight speech segment at the end of the audio stream.
    pub fn flush(&self) -> Result<Vec<FfiAudioPipelineEvent>, FfiError> {
        let mut pipeline = self.lock_inner()?;
        let events = pipeline.flush().map_err(|e| FfiError::Backend {
            detail: format!("failed to flush audio pipeline: {e}"),
        })?;
        Ok(events.into_iter().map(Into::into).collect())
    }

    /// Reset stream state, VAD recurrent state, KWS ring buffer, and speech accumulators.
    pub fn reset(&self) -> Result<(), FfiError> {
        let mut pipeline = self.lock_inner()?;
        pipeline.reset();
        self.cancel.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Cooperatively cancel any active transcription.
    ///
    /// Cancellation is sticky across utterances. Call `clear_cancel()` or `reset()`
    /// before subsequent speech segments to resume transcription.
    pub fn cancel(&self) -> Result<(), FfiError> {
        self.cancel.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Clear cooperative cancellation flag.
    pub fn clear_cancel(&self) -> Result<(), FfiError> {
        self.cancel.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Return a copy of the most recently finished utterance audio samples.
    pub fn last_utterance(&self) -> Result<Vec<f32>, FfiError> {
        let pipeline = self.lock_inner()?;
        Ok(pipeline.last_utterance().to_vec())
    }

    /// Take ownership of the most recently completed utterance audio samples.
    pub fn take_last_utterance(&self) -> Result<Vec<f32>, FfiError> {
        let mut pipeline = self.lock_inner()?;
        Ok(pipeline.take_last_utterance())
    }

    /// Transcribe an arbitrary buffer of 16 kHz mono PCM audio samples.
    pub fn transcribe_pcm(&self, pcm: Vec<f32>) -> Result<String, FfiError> {
        let pipeline = self.lock_inner()?;
        pipeline
            .transcribe_pcm(&pcm)
            .map_err(|e| FfiError::Backend {
                detail: format!("failed to transcribe audio: {e}"),
            })
    }

    /// Pop a queued event emitted by previous chunk evaluations.
    pub fn pop_event(&self) -> Result<Option<FfiAudioPipelineEvent>, FfiError> {
        let mut pipeline = self.lock_inner()?;
        Ok(pipeline.pop_event().map(Into::into))
    }

    /// Whether a speaker diarizer is attached and running. It stops, with a warning in the log,
    /// if it fails; the pipeline then keeps transcribing without speaker labels.
    pub fn has_diarizer(&self) -> Result<bool, FfiError> {
        let pipeline = self.lock_inner()?;
        Ok(pipeline.has_diarizer())
    }

    /// Whether the diarizer was staged on the Hexagon NPU and has not stopped (false: the
    /// CPU, or no diarizer). Steps the NPU declines or that fail there still run on the CPU.
    pub fn diarizer_on_npu(&self) -> bool {
        // A stopped diarizer reports false even when staging succeeded: the flag alone would
        // claim the NPU exactly when the labels stop. A poisoned mutex reads as no diarizer.
        self.diarizer_on_npu && self.inner.lock().is_ok_and(|p| p.has_diarizer())
    }

    /// Register an utterance transcribed outside the pipeline so it gets an `UtteranceLabeled`
    /// event too. `start_ms` and `end_ms` are on the pipeline's clock, as in
    /// `UtteranceTranscribed`. Returns whether a diarizer will label it.
    pub fn add_utterance(
        &self,
        text: String,
        start_ms: f32,
        end_ms: f32,
    ) -> Result<bool, FfiError> {
        let mut pipeline = self.lock_inner()?;
        Ok(pipeline.add_utterance(text, start_ms, end_ms))
    }
}

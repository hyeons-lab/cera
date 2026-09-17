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
}

impl FfiAudioPipeline {
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
        let pipeline = builder.build().map_err(|e| FfiError::Backend {
            detail: format!("failed to build audio pipeline: {e}"),
        })?;
        let cancel = pipeline.cancel_handle();
        Ok(Arc::new(Self {
            inner: Mutex::new(pipeline),
            cancel,
        }))
    }

    /// Construct a pipeline from in-memory GGUF byte buffers.
    #[uniffi::constructor]
    pub fn from_bytes(
        vad_bytes: Option<Vec<u8>>,
        hotword_bytes: Option<Vec<u8>>,
        whisper_bytes: Option<Vec<u8>>,
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
        let pipeline = builder.build().map_err(|e| FfiError::Backend {
            detail: format!("failed to build audio pipeline: {e}"),
        })?;
        let cancel = pipeline.cancel_handle();
        Ok(Arc::new(Self {
            inner: Mutex::new(pipeline),
            cancel,
        }))
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
}

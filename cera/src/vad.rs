//! Voice Activity Detection (VAD) using Silero VAD v5.
//!
//! Provides streaming and batch speech activity detection for 16 kHz and 8 kHz
//! audio using GGUF-packaged Silero VAD models, executed natively in Cera with
//! zero external runtime dependencies.
//!
//! # Example
//! ```ignore
//! use cera::vad::{SileroVad, VadSampleRate};
//!
//! let mut vad = SileroVad::from_file("models/silero_vad.gguf")?;
//!
//! // Audio chunk of 512 samples @ 16kHz (32ms)
//! let chunk = [0.0f32; 512];
//! let speech_prob = vad.process_chunk(&chunk, VadSampleRate::Rate16kHz)?;
//! println!("Speech probability: {:.3}", speech_prob);
//! # Ok::<(), anyhow::Error>(())
//! ```

#[cfg(feature = "mmap")]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::gguf::GgufFile;
use crate::tensor::{DType, Tensor};

/// Audio sample rate supported by Silero VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VadSampleRate {
    /// 16,000 Hz (standard). Window size: 512 samples (32 ms), context: 64 samples.
    Rate16kHz,
    /// 8,000 Hz (telephony). Window size: 256 samples (32 ms), context: 32 samples.
    Rate8kHz,
}

impl VadSampleRate {
    /// The required audio chunk / frame size in samples.
    pub fn window_size(&self) -> usize {
        match self {
            VadSampleRate::Rate16kHz => 512,
            VadSampleRate::Rate8kHz => 256,
        }
    }

    /// The context size in samples prepended from the previous chunk.
    pub fn context_size(&self) -> usize {
        match self {
            VadSampleRate::Rate16kHz => 64,
            VadSampleRate::Rate8kHz => 32,
        }
    }

    /// Sample rate in Hz.
    pub fn hz(&self) -> usize {
        match self {
            VadSampleRate::Rate16kHz => 16000,
            VadSampleRate::Rate8kHz => 8000,
        }
    }
}

/// A detected speech segment with sample and millisecond boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechTimestamp {
    /// Starting sample index in the original audio buffer.
    pub start_sample: u64,
    /// Ending sample index in the original audio buffer.
    pub end_sample: u64,
    /// Start time in milliseconds.
    pub start_ms: f32,
    /// End time in milliseconds.
    pub end_ms: f32,
}

/// Configuration options for batch speech detection and segmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadConfig {
    /// Probability threshold to classify a frame as speech (default: 0.5).
    pub threshold: f32,
    /// Probability threshold to classify a frame as silence (default: 0.35).
    pub neg_threshold: f32,
    /// Minimum duration of speech segment to be retained, in milliseconds (default: 64 ms).
    pub min_speech_duration_ms: usize,
    /// Minimum duration of silence before splitting speech segments, in milliseconds (default: 100 ms).
    pub min_silence_duration_ms: usize,
    /// Amount of padding to prepend and append to speech segments, in milliseconds (default: 30 ms).
    pub speech_pad_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_duration_ms: 64,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

impl VadConfig {
    /// Validates configuration parameters and clamps thresholds defensively.
    #[inline]
    pub fn sanitized(&self) -> Self {
        let threshold = if self.threshold.is_finite() {
            self.threshold.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let neg_threshold = if self.neg_threshold.is_finite() {
            self.neg_threshold.clamp(0.0, threshold)
        } else {
            (threshold - 0.15).max(0.0)
        };
        Self {
            threshold,
            neg_threshold,
            min_speech_duration_ms: self.min_speech_duration_ms,
            min_silence_duration_ms: self.min_silence_duration_ms,
            speech_pad_ms: self.speech_pad_ms,
        }
    }
}

/// A speech boundary event emitted during streaming audio processing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VadEvent {
    /// Speech has started at the given sample index and millisecond timestamp.
    SpeechStart { sample: u64, ms: f32 },
    /// Speech has ended with the given start/end sample indices and millisecond timestamps.
    SpeechEnd {
        start_sample: u64,
        end_sample: u64,
        start_ms: f32,
        end_ms: f32,
    },
}

/// Helper for stateful streaming Voice Activity Detection.
///
/// Wraps a `SileroVad` session and detects continuous speech segments across chunks.
///
/// ### Streaming Event Model
/// - **Zero-Latency Start**: To provide real-time streaming event delivery, `VadIterator` emits `SpeechStart`
///   immediately upon crossing `config.threshold`. In streaming mode, `config.min_speech_duration_ms`
///   is not used for lookahead buffering; consumers may filter short speech events upon receiving
///   `SpeechEnd` if desired.
/// - **Hysteresis & Silence Accumulation**: When speech is active (`triggered = true`), speech probability
///   falling into the hysteresis region (`neg_threshold <= prob < threshold`) keeps speech active while
///   allowing the silence counter to track consecutive unvoiced samples. Crossing `threshold` resets
///   pending silence, while dropping below `neg_threshold` for `min_silence_duration_ms` concludes the
///   segment and emits `SpeechEnd`.
#[derive(Debug, Clone)]
pub struct VadIterator {
    config: VadConfig,
    rate: VadSampleRate,
    min_silence_samples: u64,
    speech_pad_samples: u64,
    triggered: bool,
    temp_end: Option<u64>,
    current_speech_start: Option<u64>,
    current_sample: u64,
}

impl VadIterator {
    /// Create a new streaming `VadIterator` with the given sample rate and configuration.
    pub fn new(rate: VadSampleRate, config: VadConfig) -> Self {
        let config = config.sanitized();
        let samples_per_ms = rate.hz() as u64 / 1000;
        let min_silence_samples = config.min_silence_duration_ms as u64 * samples_per_ms;
        let speech_pad_samples = config.speech_pad_ms as u64 * samples_per_ms;
        Self {
            config,
            rate,
            min_silence_samples,
            speech_pad_samples,
            triggered: false,
            temp_end: None,
            current_speech_start: None,
            current_sample: 0,
        }
    }

    /// Reset stream state (sample counter, triggers, pending speech boundaries).
    pub fn reset(&mut self) {
        self.triggered = false;
        self.temp_end = None;
        self.current_speech_start = None;
        self.current_sample = 0;
    }

    /// Flush any in-flight active speech segment at the end of the audio stream.
    pub fn flush(&mut self) -> Option<VadEvent> {
        if self.triggered {
            let hz = self.rate.hz() as f64;
            let start = self.current_speech_start.unwrap_or(0);
            let end = (self.temp_end.unwrap_or(self.current_sample) + self.speech_pad_samples)
                .min(self.current_sample);
            self.triggered = false;
            self.temp_end = None;
            self.current_speech_start = None;

            let start_ms = ((start as f64 * 1000.0) / hz) as f32;
            let end_ms = ((end as f64 * 1000.0) / hz) as f32;
            Some(VadEvent::SpeechEnd {
                start_sample: start,
                end_sample: end,
                start_ms,
                end_ms,
            })
        } else {
            None
        }
    }

    /// Process a streaming chunk of audio using `vad` and return any detected speech start or end event.
    ///
    /// - `vad`: mutable reference to `SileroVad` session.
    /// - `chunk`: audio samples (512 samples for 16kHz, 256 for 8kHz).
    pub fn process_chunk(
        &mut self,
        vad: &mut SileroVad,
        chunk: &[f32],
    ) -> Result<Option<VadEvent>> {
        let prob = vad.process_chunk(chunk, self.rate)?;
        let hz = self.rate.hz() as f64;

        let cur_sample = self.current_sample;
        self.current_sample += chunk.len() as u64;

        if prob >= self.config.threshold {
            self.temp_end = None;
            if !self.triggered {
                self.triggered = true;
                let start_sample = cur_sample.saturating_sub(self.speech_pad_samples);
                self.current_speech_start = Some(start_sample);
                let ms = ((start_sample as f64 * 1000.0) / hz) as f32;
                return Ok(Some(VadEvent::SpeechStart {
                    sample: start_sample,
                    ms,
                }));
            }
        }

        if prob < self.config.neg_threshold && self.triggered {
            let temp_end_val = *self.temp_end.get_or_insert(cur_sample);
            if cur_sample.saturating_sub(temp_end_val) >= self.min_silence_samples {
                let start = self.current_speech_start.unwrap_or(0);
                let end = (temp_end_val + self.speech_pad_samples).min(self.current_sample);
                self.triggered = false;
                self.temp_end = None;
                self.current_speech_start = None;

                let start_ms = ((start as f64 * 1000.0) / hz) as f32;
                let end_ms = ((end as f64 * 1000.0) / hz) as f32;
                return Ok(Some(VadEvent::SpeechEnd {
                    start_sample: start,
                    end_sample: end,
                    start_ms,
                    end_ms,
                }));
            }
        }

        Ok(None)
    }
}

/// Weight tensors for Silero VAD v5.
struct VadWeights {
    // 16kHz network
    stft_16k_basis: Tensor,
    encoder_16k_0_w: Tensor,
    encoder_16k_0_b: Tensor,
    encoder_16k_1_w: Tensor,
    encoder_16k_1_b: Tensor,
    encoder_16k_2_w: Tensor,
    encoder_16k_2_b: Tensor,
    encoder_16k_3_w: Tensor,
    encoder_16k_3_b: Tensor,
    decoder_16k_rnn_w_ih: Tensor,
    decoder_16k_rnn_w_hh: Tensor,
    decoder_16k_rnn_b_ih: Tensor,
    decoder_16k_rnn_b_hh: Tensor,
    decoder_16k_head_w: Tensor,
    decoder_16k_head_b: Tensor,

    // 8kHz network
    stft_8k_basis: Tensor,
    encoder_8k_0_w: Tensor,
    encoder_8k_0_b: Tensor,
    encoder_8k_1_w: Tensor,
    encoder_8k_1_b: Tensor,
    encoder_8k_2_w: Tensor,
    encoder_8k_2_b: Tensor,
    encoder_8k_3_w: Tensor,
    encoder_8k_3_b: Tensor,
    decoder_8k_rnn_w_ih: Tensor,
    decoder_8k_rnn_w_hh: Tensor,
    decoder_8k_rnn_b_ih: Tensor,
    decoder_8k_rnn_b_hh: Tensor,
    decoder_8k_head_w: Tensor,
    decoder_8k_head_b: Tensor,
}

impl VadWeights {
    fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn get_f32_tensor(gguf: &GgufFile, name: &str, expected_len: usize) -> Result<Tensor> {
            let t = gguf
                .get_tensor(name)
                .with_context(|| format!("missing tensor '{name}'"))?;
            ensure!(
                t.dtype() == DType::F32,
                "tensor '{name}' expected F32 dtype, found {:?}",
                t.dtype()
            );
            ensure!(
                t.numel() == expected_len,
                "tensor '{name}' expected {} elements, found {}",
                expected_len,
                t.numel()
            );
            Ok(t)
        }

        Ok(Self {
            stft_16k_basis: get_f32_tensor(gguf, "stft.16k.basis", 258 * 256)?,
            encoder_16k_0_w: get_f32_tensor(gguf, "encoder.16k.0.weight", 128 * 129 * 3)?,
            encoder_16k_0_b: get_f32_tensor(gguf, "encoder.16k.0.bias", 128)?,
            encoder_16k_1_w: get_f32_tensor(gguf, "encoder.16k.1.weight", 64 * 128 * 3)?,
            encoder_16k_1_b: get_f32_tensor(gguf, "encoder.16k.1.bias", 64)?,
            encoder_16k_2_w: get_f32_tensor(gguf, "encoder.16k.2.weight", 64 * 64 * 3)?,
            encoder_16k_2_b: get_f32_tensor(gguf, "encoder.16k.2.bias", 64)?,
            encoder_16k_3_w: get_f32_tensor(gguf, "encoder.16k.3.weight", 128 * 64 * 3)?,
            encoder_16k_3_b: get_f32_tensor(gguf, "encoder.16k.3.bias", 128)?,
            decoder_16k_rnn_w_ih: get_f32_tensor(gguf, "decoder.16k.rnn.weight_ih", 512 * 128)?,
            decoder_16k_rnn_w_hh: get_f32_tensor(gguf, "decoder.16k.rnn.weight_hh", 512 * 128)?,
            decoder_16k_rnn_b_ih: get_f32_tensor(gguf, "decoder.16k.rnn.bias_ih", 512)?,
            decoder_16k_rnn_b_hh: get_f32_tensor(gguf, "decoder.16k.rnn.bias_hh", 512)?,
            decoder_16k_head_w: get_f32_tensor(gguf, "decoder.16k.head.weight", 128)?,
            decoder_16k_head_b: get_f32_tensor(gguf, "decoder.16k.head.bias", 1)?,

            stft_8k_basis: get_f32_tensor(gguf, "stft.8k.basis", 130 * 128)?,
            encoder_8k_0_w: get_f32_tensor(gguf, "encoder.8k.0.weight", 128 * 65 * 3)?,
            encoder_8k_0_b: get_f32_tensor(gguf, "encoder.8k.0.bias", 128)?,
            encoder_8k_1_w: get_f32_tensor(gguf, "encoder.8k.1.weight", 64 * 128 * 3)?,
            encoder_8k_1_b: get_f32_tensor(gguf, "encoder.8k.1.bias", 64)?,
            encoder_8k_2_w: get_f32_tensor(gguf, "encoder.8k.2.weight", 64 * 64 * 3)?,
            encoder_8k_2_b: get_f32_tensor(gguf, "encoder.8k.2.bias", 64)?,
            encoder_8k_3_w: get_f32_tensor(gguf, "encoder.8k.3.weight", 128 * 64 * 3)?,
            encoder_8k_3_b: get_f32_tensor(gguf, "encoder.8k.3.bias", 128)?,
            decoder_8k_rnn_w_ih: get_f32_tensor(gguf, "decoder.8k.rnn.weight_ih", 512 * 128)?,
            decoder_8k_rnn_w_hh: get_f32_tensor(gguf, "decoder.8k.rnn.weight_hh", 512 * 128)?,
            decoder_8k_rnn_b_ih: get_f32_tensor(gguf, "decoder.8k.rnn.bias_ih", 512)?,
            decoder_8k_rnn_b_hh: get_f32_tensor(gguf, "decoder.8k.rnn.bias_hh", 512)?,
            decoder_8k_head_w: get_f32_tensor(gguf, "decoder.8k.head.weight", 128)?,
            decoder_8k_head_b: get_f32_tensor(gguf, "decoder.8k.head.bias", 1)?,
        })
    }
}

/// Stateful Silero Voice Activity Detector (VAD) session.
pub struct SileroVad {
    weights: VadWeights,
    /// Hidden state `h` of the LSTM cell (size 128).
    h: [f32; 128],
    /// Cell state `c` of the LSTM cell (size 128).
    c: [f32; 128],
    /// Trailing context buffer for 16 kHz streams (64 samples).
    context_16k: [f32; 64],
    /// Trailing context buffer for 8 kHz streams (32 samples).
    context_8k: [f32; 32],
}

impl SileroVad {
    /// Load a Silero VAD session from a parsed GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let weights = VadWeights::from_gguf(gguf)?;
        Ok(Self {
            weights,
            h: [0.0; 128],
            c: [0.0; 128],
            context_16k: [0.0; 64],
            context_8k: [0.0; 32],
        })
    }

    /// Load a Silero VAD model from a `.gguf` file path.
    #[cfg(feature = "mmap")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf = GgufFile::open(path.as_ref())?;
        Self::from_gguf(&gguf)
    }

    /// Load a Silero VAD model from in-memory GGUF bytes.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let gguf = GgufFile::from_bytes(bytes.into())?;
        Self::from_gguf(&gguf)
    }

    /// Reset recurrent state tensors and streaming context to zeros.
    ///
    /// Always call this when starting a new stream or switching audio contexts.
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.context_16k.fill(0.0);
        self.context_8k.fill(0.0);
    }

    /// Process a single chunk of audio and return the speech probability in `[0.0, 1.0]`.
    ///
    /// - For [`VadSampleRate::Rate16kHz`], `chunk` must have exactly 512 samples.
    /// - For [`VadSampleRate::Rate8kHz`], `chunk` must have exactly 256 samples.
    pub fn process_chunk(&mut self, chunk: &[f32], rate: VadSampleRate) -> Result<f32> {
        let expected_size = rate.window_size();
        ensure!(
            chunk.len() == expected_size,
            "expected chunk of {} samples for {:?}, got {}",
            expected_size,
            rate,
            chunk.len()
        );
        ensure!(
            chunk.iter().all(|s| s.is_finite()),
            "input audio chunk contains non-finite values (NaN or Inf)"
        );

        let prob = match rate {
            VadSampleRate::Rate16kHz => self.forward_16k(chunk),
            VadSampleRate::Rate8kHz => self.forward_8k(chunk),
        };
        Ok(prob)
    }

    /// Process an entire audio buffer and return speech timestamps.
    pub fn get_speech_timestamps(
        &mut self,
        audio: &[f32],
        rate: VadSampleRate,
        config: &VadConfig,
    ) -> Result<Vec<SpeechTimestamp>> {
        self.reset();

        let window_size = rate.window_size();
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let config = config.sanitized();
        let samples_per_ms = rate.hz() as u64 / 1000;
        let min_speech_samples = config.min_speech_duration_ms as u64 * samples_per_ms;
        let min_silence_samples = config.min_silence_duration_ms as u64 * samples_per_ms;
        let speech_pad_samples = config.speech_pad_ms as u64 * samples_per_ms;
        let hz = rate.hz() as f64;

        let num_chunks = audio.len().div_ceil(window_size);
        let mut speech_probs = Vec::with_capacity(num_chunks);
        let mut chunks = audio.chunks_exact(window_size);

        for chunk in &mut chunks {
            let prob = self.process_chunk(chunk, rate)?;
            speech_probs.push(prob);
        }

        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut padded = [0.0f32; 512];
            padded[..rem.len()].copy_from_slice(rem);
            let prob = self.process_chunk(&padded[..window_size], rate)?;
            speech_probs.push(prob);
        }

        let mut speeches: Vec<(u64, u64)> = Vec::new();
        let mut current_speech_start: Option<u64> = None;
        let mut temp_end: Option<u64> = None;
        let mut triggered = false;

        for (i, &prob) in speech_probs.iter().enumerate() {
            let cur_sample = (i * window_size) as u64;

            if prob >= config.threshold {
                temp_end = None;
                if !triggered {
                    triggered = true;
                    current_speech_start = Some(cur_sample);
                    continue;
                }
            } else if prob < config.neg_threshold && triggered {
                let temp_end_val = *temp_end.get_or_insert(cur_sample);
                let sil_dur = cur_sample.saturating_sub(temp_end_val);
                if sil_dur >= min_silence_samples {
                    if let Some(start) = current_speech_start
                        && temp_end_val > start
                        && (temp_end_val - start) >= min_speech_samples
                    {
                        speeches.push((start, temp_end_val));
                    }
                    current_speech_start = None;
                    temp_end = None;
                    triggered = false;
                }
            }
        }

        if let Some(start) = current_speech_start {
            let end = temp_end.unwrap_or(audio.len() as u64);
            if end > start && (end - start) >= min_speech_samples {
                speeches.push((start, end));
            }
        }

        let mut timestamps = Vec::with_capacity(speeches.len());
        let audio_len = audio.len() as u64;

        for i in 0..speeches.len() {
            if i == 0 {
                speeches[i].0 = speeches[i].0.saturating_sub(speech_pad_samples);
            }
            if i != speeches.len() - 1 {
                let next_start = speeches[i + 1].0;
                let sil_dur = next_start.saturating_sub(speeches[i].1);
                if sil_dur < 2 * speech_pad_samples {
                    speeches[i].1 += sil_dur / 2;
                    speeches[i + 1].0 = speeches[i + 1].0.saturating_sub(sil_dur / 2);
                } else {
                    speeches[i].1 = (speeches[i].1 + speech_pad_samples).min(audio_len);
                    speeches[i + 1].0 = speeches[i + 1].0.saturating_sub(speech_pad_samples);
                }
            } else {
                speeches[i].1 = (speeches[i].1 + speech_pad_samples).min(audio_len);
            }

            let start_ms = ((speeches[i].0 as f64 * 1000.0) / hz) as f32;
            let end_ms = ((speeches[i].1 as f64 * 1000.0) / hz) as f32;
            timestamps.push(SpeechTimestamp {
                start_sample: speeches[i].0,
                end_sample: speeches[i].1,
                start_ms,
                end_ms,
            });
        }

        self.reset();
        Ok(timestamps)
    }

    // ── Internal forward passes ───────────────────────────────────────────────

    fn forward_16k(&mut self, chunk: &[f32]) -> f32 {
        // Direct context prepend and right reflect-pad:
        // Context (64) + Chunk (512) -> 576 samples
        // Padded with 64 right reflected -> 640 samples
        let mut padded = [0.0f32; 640];
        padded[..64].copy_from_slice(&self.context_16k);
        padded[64..576].copy_from_slice(chunk);
        self.context_16k.copy_from_slice(&padded[512..576]);

        for i in 0..64 {
            padded[576 + i] = padded[575 - 1 - i];
        }

        // 3. STFT Conv: in_ch=1, out_ch=258, kernel=256, stride=128
        // Input length 640 -> 4 frames: (640 - 256) / 128 + 1 = 4
        let basis = self.weights.stft_16k_basis.as_f32_slice();
        let mut mag = [0.0f32; 129 * 4]; // [129 channels, 4 frames]

        for frame in 0..4 {
            let window = &padded[frame * 128..frame * 128 + 256];
            for ch in 0..129 {
                let real_filter = &basis[ch * 256..(ch + 1) * 256];
                let imag_filter = &basis[(ch + 129) * 256..(ch + 130) * 256];

                let real_val = dot_product(window, real_filter);
                let imag_val = dot_product(window, imag_filter);

                let m = (real_val * real_val + imag_val * imag_val).sqrt();
                mag[ch * 4 + frame] = m;
            }
        }

        // 4. Encoder
        // Layer 0: Conv1D(129 -> 128, k=3, s=1, p=1) + ReLU -> [128, 4]
        let mut enc0 = [0.0f32; 128 * 4];
        conv1d_relu(
            &mag,
            &mut enc0,
            129,
            4,
            128,
            1,
            1,
            self.weights.encoder_16k_0_w.as_f32_slice(),
            self.weights.encoder_16k_0_b.as_f32_slice(),
        );

        // Layer 1: Conv1D(128 -> 64, k=3, s=2, p=1) + ReLU -> [64, 2]
        let mut enc1 = [0.0f32; 64 * 2];
        conv1d_relu(
            &enc0,
            &mut enc1,
            128,
            4,
            64,
            2,
            1,
            self.weights.encoder_16k_1_w.as_f32_slice(),
            self.weights.encoder_16k_1_b.as_f32_slice(),
        );

        // Layer 2: Conv1D(64 -> 64, k=3, s=2, p=1) + ReLU -> [64, 1]
        let mut enc2 = [0.0f32; 64];
        conv1d_relu(
            &enc1,
            &mut enc2,
            64,
            2,
            64,
            2,
            1,
            self.weights.encoder_16k_2_w.as_f32_slice(),
            self.weights.encoder_16k_2_b.as_f32_slice(),
        );

        // Layer 3: Conv1D(64 -> 128, k=3, s=1, p=1) + ReLU -> [128, 1]
        let mut enc3 = [0.0f32; 128];
        conv1d_relu(
            &enc2,
            &mut enc3,
            64,
            1,
            128,
            1,
            1,
            self.weights.encoder_16k_3_w.as_f32_slice(),
            self.weights.encoder_16k_3_b.as_f32_slice(),
        );

        // 5. LSTM Cell & Decoder Head (16kHz)
        self.lstm_and_head(&enc3, VadSampleRate::Rate16kHz)
    }

    fn forward_8k(&mut self, chunk: &[f32]) -> f32 {
        // Direct context prepend and right reflect-pad:
        // Context (32) + Chunk (256) -> 288 samples
        // Padded with 32 right reflected -> 320 samples
        let mut padded = [0.0f32; 320];
        padded[..32].copy_from_slice(&self.context_8k);
        padded[32..288].copy_from_slice(chunk);
        self.context_8k.copy_from_slice(&padded[256..288]);

        for i in 0..32 {
            padded[288 + i] = padded[287 - 1 - i];
        }

        // 3. STFT Conv: in_ch=1, out_ch=130, kernel=128, stride=64
        // Input length 320 -> 4 frames: (320 - 128) / 64 + 1 = 4
        let basis = self.weights.stft_8k_basis.as_f32_slice();
        let mut mag = [0.0f32; 65 * 4]; // [65 channels, 4 frames]

        for frame in 0..4 {
            let window = &padded[frame * 64..frame * 64 + 128];
            for ch in 0..65 {
                let real_filter = &basis[ch * 128..(ch + 1) * 128];
                let imag_filter = &basis[(ch + 65) * 128..(ch + 66) * 128];

                let real_val = dot_product(window, real_filter);
                let imag_val = dot_product(window, imag_filter);

                let m = (real_val * real_val + imag_val * imag_val).sqrt();
                mag[ch * 4 + frame] = m;
            }
        }

        // 4. Encoder (8kHz)
        // Layer 0: Conv1D(65 -> 128, k=3, s=1, p=1) + ReLU -> [128, 4]
        let mut enc0 = [0.0f32; 128 * 4];
        conv1d_relu(
            &mag,
            &mut enc0,
            65,
            4,
            128,
            1,
            1,
            self.weights.encoder_8k_0_w.as_f32_slice(),
            self.weights.encoder_8k_0_b.as_f32_slice(),
        );

        // Layer 1: Conv1D(128 -> 64, k=3, s=2, p=1) + ReLU -> [64, 2]
        let mut enc1 = [0.0f32; 64 * 2];
        conv1d_relu(
            &enc0,
            &mut enc1,
            128,
            4,
            64,
            2,
            1,
            self.weights.encoder_8k_1_w.as_f32_slice(),
            self.weights.encoder_8k_1_b.as_f32_slice(),
        );

        // Layer 2: Conv1D(64 -> 64, k=3, s=2, p=1) + ReLU -> [64, 1]
        let mut enc2 = [0.0f32; 64];
        conv1d_relu(
            &enc1,
            &mut enc2,
            64,
            2,
            64,
            2,
            1,
            self.weights.encoder_8k_2_w.as_f32_slice(),
            self.weights.encoder_8k_2_b.as_f32_slice(),
        );

        // Layer 3: Conv1D(64 -> 128, k=3, s=1, p=1) + ReLU -> [128, 1]
        let mut enc3 = [0.0f32; 128];
        conv1d_relu(
            &enc2,
            &mut enc3,
            64,
            1,
            128,
            1,
            1,
            self.weights.encoder_8k_3_w.as_f32_slice(),
            self.weights.encoder_8k_3_b.as_f32_slice(),
        );

        // 5. LSTM Cell & Decoder Head (8kHz)
        self.lstm_and_head(&enc3, VadSampleRate::Rate8kHz)
    }

    fn lstm_and_head(&mut self, enc3: &[f32; 128], rate: VadSampleRate) -> f32 {
        let (w_ih, w_hh, b_ih, b_hh, dec_w, dec_b) = match rate {
            VadSampleRate::Rate16kHz => (
                self.weights.decoder_16k_rnn_w_ih.as_f32_slice(),
                self.weights.decoder_16k_rnn_w_hh.as_f32_slice(),
                self.weights.decoder_16k_rnn_b_ih.as_f32_slice(),
                self.weights.decoder_16k_rnn_b_hh.as_f32_slice(),
                self.weights.decoder_16k_head_w.as_f32_slice(),
                self.weights.decoder_16k_head_b.as_f32_slice()[0],
            ),
            VadSampleRate::Rate8kHz => (
                self.weights.decoder_8k_rnn_w_ih.as_f32_slice(),
                self.weights.decoder_8k_rnn_w_hh.as_f32_slice(),
                self.weights.decoder_8k_rnn_b_ih.as_f32_slice(),
                self.weights.decoder_8k_rnn_b_hh.as_f32_slice(),
                self.weights.decoder_8k_head_w.as_f32_slice(),
                self.weights.decoder_8k_head_b.as_f32_slice()[0],
            ),
        };
        // 4 gates of size 128: [i, f, g, o]
        let mut gates = [0.0f32; 512];
        let (rows_ih, _) = w_ih.as_chunks::<128>();
        let (rows_hh, _) = w_hh.as_chunks::<128>();
        for (g, (row_ih, row_hh)) in rows_ih.iter().zip(rows_hh.iter()).enumerate() {
            gates[g] = dot_product(row_ih, enc3) + b_ih[g] + dot_product(row_hh, &self.h) + b_hh[g];
        }

        // LSTM activations
        for idx in 0..128 {
            let i_gate = sigmoid(gates[idx]);
            let f_gate = sigmoid(gates[128 + idx]);
            let g_gate = gates[256 + idx].tanh();
            let o_gate = sigmoid(gates[384 + idx]);

            let c_next = f_gate * self.c[idx] + i_gate * g_gate;
            let h_next = o_gate * c_next.tanh();

            self.c[idx] = c_next;
            self.h[idx] = h_next;
        }

        // Decoder Head: ReLU(h) -> Linear(128 -> 1) -> Sigmoid
        assert_eq!(dec_w.len(), self.h.len());
        let logit = self
            .h
            .iter()
            .zip(dec_w.iter())
            .fold(dec_b, |acc, (&h_val, &w)| acc + w * h_val.max(0.0));

        sigmoid(logit)
    }
}

// ── Math & Convolution Helpers ────────────────────────────────────────────────

#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 1D Convolution with stride, padding, bias and ReLU activation.
///
/// Input layout: `[in_channels, in_len]` row-major.
/// Output layout: `[out_channels, out_len]` row-major.
/// Weight layout: `[out_channels, in_channels, kernel_size]` (3D contiguous).
#[allow(clippy::too_many_arguments)]
#[inline]
fn conv1d_relu(
    input: &[f32],
    output: &mut [f32],
    in_channels: usize,
    in_len: usize,
    out_channels: usize,
    stride: usize,
    pad: usize,
    weights: &[f32],
    bias: &[f32],
) {
    debug_assert!(stride > 0, "stride must be strictly positive");
    debug_assert_eq!(weights.len(), out_channels * in_channels * 3);
    debug_assert_eq!(bias.len(), out_channels);
    debug_assert_eq!(input.len(), in_channels * in_len);
    let padded_len = in_len + 2 * pad;
    let out_len = if padded_len >= 3 {
        (padded_len - 3) / stride + 1
    } else {
        0
    };
    debug_assert_eq!(output.len(), out_channels * out_len);

    for (out_c, (w_out_c, &b)) in weights
        .chunks_exact(in_channels * 3)
        .zip(bias.iter())
        .enumerate()
    {
        let (chunks_3, _) = w_out_c.as_chunks::<3>();
        for out_idx in 0..out_len {
            let in_center = out_idx * stride;
            let mut acc = b;

            for (k, in_pos) in (in_center..in_center + 3).enumerate() {
                if in_pos >= pad && in_pos < in_len + pad {
                    let in_offset = in_pos - pad;
                    for (in_c, w_in_c) in chunks_3.iter().enumerate() {
                        let sample = input[in_c * in_len + in_offset];
                        acc += w_in_c[k] * sample;
                    }
                }
            }

            output[out_c * out_len + out_idx] = acc.max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_relu_underflow_guard() {
        let input = [1.0f32];
        let mut output = [];
        let weights = [0.1f32; 3];
        let bias = [0.0f32; 1];
        // in_len = 1, pad = 0 -> padded_len = 1 < 3 -> out_len = 0
        conv1d_relu(&input, &mut output, 1, 1, 1, 1, 0, &weights, &bias);
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_vad_init_and_reset() {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest_dir.join("models/silero_vad.gguf"),
            manifest_dir.join("../models/silero_vad.gguf"),
            std::path::PathBuf::from("models/silero_vad.gguf"),
        ];
        let model_path = candidates.into_iter().find(|p| p.exists());
        let Some(path) = model_path else {
            eprintln!("Skipping test_vad_init_and_reset: models/silero_vad.gguf not found");
            return;
        };
        let mut vad = SileroVad::from_file(&path).expect("load SileroVad fixture");
        vad.reset();
        assert_eq!(vad.h.len(), 128);
        assert_eq!(vad.c.len(), 128);

        // Test 16kHz chunk
        let chunk_16k = [0.0f32; 512];
        let prob_16k = vad
            .process_chunk(&chunk_16k, VadSampleRate::Rate16kHz)
            .unwrap();
        assert!((0.0..=1.0).contains(&prob_16k));

        // Test 8kHz chunk
        vad.reset();
        let chunk_8k = [0.0f32; 256];
        let prob_8k = vad
            .process_chunk(&chunk_8k, VadSampleRate::Rate8kHz)
            .unwrap();
        assert!((0.0..=1.0).contains(&prob_8k));

        // Test non-finite chunk rejection
        let mut nan_chunk = [0.0f32; 512];
        nan_chunk[10] = f32::NAN;
        assert!(
            vad.process_chunk(&nan_chunk, VadSampleRate::Rate16kHz)
                .is_err()
        );

        // Test short audio buffer (< 512 samples)
        let short_audio = [0.0f32; 128];
        let timestamps = vad
            .get_speech_timestamps(
                &short_audio,
                VadSampleRate::Rate16kHz,
                &VadConfig::default(),
            )
            .unwrap();
        assert!(timestamps.is_empty());
    }
}

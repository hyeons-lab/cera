//! Audio-aware generation loop with text ↔ audio modality switching.

use std::sync::atomic::AtomicBool;

use anyhow::Result;

use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::Model;
use crate::model::audio_decoder::{
    AudioAccelerator, AudioDecoderWeights, DepthformerState, DetokenizerState, DetokenizerWeights,
    detokenize_to_spectrum, embed_audio_codes, embed_audio_token, istft_to_pcm, sample_audio_frame,
    sample_frame,
};
use crate::sampler::{Sampler, SamplerConfig};
use crate::time::{Duration, Instant};
use crate::tokenizer::BpeTokenizer;

/// Audio generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioGenerateConfig {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    /// Audio sampling temperature (0.0 = greedy, >0 = stochastic).
    pub audio_temperature: f32,
    /// Audio top-k for stochastic sampling.
    pub audio_top_k: usize,
    /// Generation mode.
    pub mode: AudioMode,
    /// Opt in to an experimental accelerated depthformer (Metal, wgpu): off by
    /// default because GEMV accumulation order differences can produce
    /// different codes. Accelerators trusted by default (Hexagon) ignore it.
    pub gpu_depthformer: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioMode {
    /// All text first, then audio when <|audio_start|> (128) is emitted.
    Sequential,
    /// Alternate: 6 text tokens, 12 audio frames, repeat.
    Interleaved,
}

/// Result of audio generation.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioGenerateResult {
    pub text_tokens: usize,
    pub audio_frames: usize,
    pub audio_samples: usize,
    pub elapsed_secs: f64,
    pub depthformer_secs: f64,
    pub detokenizer_secs: f64,
}

/// Special token IDs for modality control.
pub const TOKEN_AUDIO_START: u32 = 128;
pub const TOKEN_AUDIO_END: u32 = 129;
pub const TOKEN_TEXT_END: u32 = 130;
pub const TOKEN_MIXED_START: u32 = 131;
pub const AUDIO_END_CODE: i32 = 2048;
pub const LFM3A_AUDIO_END_CODE: i32 = 256;

/// The end-of-speech code of an `lfm3a_gen` decoder with `n_vocab` entries: the sampler's own
/// mask/EOS id, the last one. (`LFM3A_AUDIO_END_CODE` is that id for the 257-entry vocabulary.)
fn lfm3a_eos_code(n_vocab: usize) -> i32 {
    i32::try_from(n_vocab.saturating_sub(1)).unwrap_or(i32::MAX)
}

/// Whether `code0` ends an `lfm3a_gen` audio span.
fn lfm3a_is_end_code(n_vocab: usize, code0: i32) -> bool {
    code0 == lfm3a_eos_code(n_vocab)
}
pub const DEFAULT_INTERLEAVED_TEXT_BUDGET: usize = 6;
pub const DEFAULT_INTERLEAVED_AUDIO_BUDGET: usize = 12;

/// Modality control token IDs resolved dynamically from the tokenizer vocabulary,
/// falling back to the standard LFM2/LFM3 token IDs (`128..=131`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModalityControlTokens {
    pub audio_start: u32,
    pub audio_end: u32,
    pub text_end: u32,
    pub mixed_start: u32,
}

impl Default for ModalityControlTokens {
    fn default() -> Self {
        Self {
            audio_start: TOKEN_AUDIO_START,
            audio_end: TOKEN_AUDIO_END,
            text_end: TOKEN_TEXT_END,
            mixed_start: TOKEN_MIXED_START,
        }
    }
}

impl ModalityControlTokens {
    /// Resolve `<|audio_start|>`, `<|audio_end|>`, `<|text_end|>`, and `<|mixed_start|>`
    /// dynamically from `tokenizer`, falling back to `128, 129, 130, 131` when absent.
    pub fn from_tokenizer(tokenizer: &BpeTokenizer) -> Self {
        Self {
            audio_start: tokenizer
                .token_to_id("<|audio_start|>")
                .unwrap_or(TOKEN_AUDIO_START),
            audio_end: tokenizer
                .token_to_id("<|audio_end|>")
                .unwrap_or(TOKEN_AUDIO_END),
            text_end: tokenizer
                .token_to_id("<|text_end|>")
                .unwrap_or(TOKEN_TEXT_END),
            mixed_start: tokenizer
                .token_to_id("<|mixed_start|>")
                .unwrap_or(TOKEN_MIXED_START),
        }
    }
}

/// How many text tokens and audio frames an interleaved turn alternates
/// between. A vocoder GGUF may declare its own cadence
/// (`interleaved_n_text` / `interleaved_n_audio`; the Japanese LFM2.5-Audio
/// declares 6 and 9), and the llama.cpp reference honors it. A vocoder without
/// those keys, such as the merged LEAP English one, gets the 6/12 defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterleaveCadence {
    /// Text tokens per interleave round.
    pub text: usize,
    /// Audio frames per interleave round.
    pub audio: usize,
}

impl Default for InterleaveCadence {
    fn default() -> Self {
        Self {
            text: DEFAULT_INTERLEAVED_TEXT_BUDGET,
            audio: DEFAULT_INTERLEAVED_AUDIO_BUDGET,
        }
    }
}

impl InterleaveCadence {
    /// Read the cadence from a vocoder GGUF's metadata. A missing key keeps its
    /// default; a zero (which would spin the interleave loop) is rejected with
    /// a warning and also keeps the default.
    pub fn from_gguf(gguf: &crate::gguf::GgufFile) -> Self {
        let read = |key: &str, alt_key: &str, default: usize| match gguf
            .get_u32(key)
            .or_else(|| gguf.get_u32(alt_key))
        {
            Some(0) => {
                tracing::warn!(
                    target: "cera::audio",
                    key,
                    "vocoder declares a zero interleave budget; using {default}"
                );
                default
            }
            Some(v) => v as usize,
            None => default,
        };
        let d = Self::default();
        Self {
            text: read(
                "interleaved_n_text",
                "clip.audio_gen.interleaved_n_text",
                d.text,
            ),
            audio: read(
                "interleaved_n_audio",
                "clip.audio_gen.interleaved_n_audio",
                d.audio,
            ),
        }
    }
}
pub const AUDIO_SAFETY_FRAME_LIMIT: usize = 4096;

/// Standard silence detection watchdog for voice synthesis across native and WebAssembly engines.
/// Tracks RMS energy across frames and detects natural trailing speech termination.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSilenceWatchdog {
    pub total_voiced_frames: usize,
    pub consecutive_silent_frames: usize,
    pub audio_frames_count: usize,
    pub rms_threshold: f32,
    pub min_voiced_frames: usize,
    pub silent_frames_cutoff: usize,
    pub silent_frames_cutoff_voiced: usize,
}

impl Default for AudioSilenceWatchdog {
    fn default() -> Self {
        Self {
            total_voiced_frames: 0,
            consecutive_silent_frames: 0,
            audio_frames_count: 0,
            rms_threshold: 0.001,
            min_voiced_frames: 5,
            silent_frames_cutoff: 30,
            silent_frames_cutoff_voiced: 25,
        }
    }
}

impl AudioSilenceWatchdog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_thresholds(
        rms_threshold: f32,
        min_voiced_frames: usize,
        silent_frames_cutoff: usize,
        silent_frames_cutoff_voiced: usize,
    ) -> Self {
        Self {
            total_voiced_frames: 0,
            consecutive_silent_frames: 0,
            audio_frames_count: 0,
            rms_threshold,
            min_voiced_frames,
            silent_frames_cutoff,
            silent_frames_cutoff_voiced,
        }
    }

    /// Observe a new decoded PCM buffer and update silence/voiced counters.
    /// Returns the computed RMS energy.
    #[inline]
    pub fn observe_pcm(&mut self, pcm: &[f32]) -> f32 {
        self.audio_frames_count += 1;
        if pcm.is_empty() {
            self.consecutive_silent_frames += 1;
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut valid_samples = 0usize;
        for &x in pcm {
            if x.is_finite() {
                sum += x * x;
                valid_samples += 1;
            }
        }
        if valid_samples == 0 {
            self.consecutive_silent_frames += 1;
            return 0.0;
        }
        let rms = (sum / valid_samples as f32).sqrt();
        if rms >= self.rms_threshold {
            self.total_voiced_frames += 1;
            self.consecutive_silent_frames = 0;
        } else {
            self.consecutive_silent_frames += 1;
        }
        rms
    }

    /// Check if trailing silence has reached the termination threshold.
    #[inline]
    pub fn is_silence_terminated(&self) -> bool {
        self.consecutive_silent_frames >= self.silent_frames_cutoff
            || (self.total_voiced_frames >= self.min_voiced_frames
                && self.consecutive_silent_frames >= self.silent_frames_cutoff_voiced)
    }

    /// Check if the safety frame ceiling has been reached.
    #[inline]
    pub fn is_safety_limit_reached(&self) -> bool {
        self.audio_frames_count >= AUDIO_SAFETY_FRAME_LIMIT
    }
}

#[derive(PartialEq)]
enum Modality {
    Text,
    Audio,
}

// ---------------------------------------------------------------------------
// AudioOutputDecoder
// ---------------------------------------------------------------------------

/// Per-frame outcome from [`AudioOutputDecoder::decode_frame`].
#[derive(Debug, Clone, PartialEq)]
pub enum FrameOutcome {
    /// Codes sampled + detokenized; `audio_embedding` is the feedback
    /// embedding the caller should pass back through the main LLM.
    /// `pcm` contains the progressive real-time audio chunk for this frame.
    /// `codes` contains the 8 discrete acoustic codebook indices for this frame.
    Codes {
        audio_embedding: Vec<f32>,
        pcm: Vec<f32>,
        codes: [i32; 8],
    },
    /// Audio stream terminated (`codes[0] == AUDIO_END_CODE`). No
    /// spectrum produced for this frame; caller should return control
    /// to the text modality per its mode's exit convention.
    End,
    /// The accelerator backend faulted mid-frame (see
    /// `AudioAccelerator::take_audio_error`) and the stage cannot continue on
    /// CPU fallback with coherent state, so the frame produced nothing usable.
    /// Callers must abort generation with the error, not continue over the hole:
    /// both sampling faults and mid-utterance or lost-device detokenize faults
    /// arrive here (while batch-mode detokenize/lost-device ISTFT faults
    /// propagate from `try_finish`). Carries the fault detail string (`CeraError`
    /// is neither `Clone` nor `PartialEq`, which this enum derives); callers
    /// re-wrap with `CeraError::Backend`.
    Fault(String),
}

/// Owns the audio output-decoder state and exposes per-frame operations.
/// Extracted from `generate_audio` so the same per-frame logic is shared
/// between the Sequential and Interleaved paths and across CPU/WebGPU sessions.
pub struct AudioOutputDecoder<'a> {
    weights: &'a AudioDecoderWeights,
    detok_weights: &'a DetokenizerWeights,
    accelerator: Option<&'a dyn AudioAccelerator>,
    df_state: DepthformerState,
    detok_state: DetokenizerState,
    streamer: crate::model::audio_decoder::IstftStreamer,
    /// Accumulated spectrum across the entire generate_audio call.
    all_spectrum: Vec<f32>,
    audio_frames: usize,
    streamed_samples: usize,
    time_depthformer: Duration,
    time_detokenizer: Duration,
    audio_temperature: f32,
    audio_top_k: usize,
    /// Precomputed: hardware depthformer acceleration is active.
    use_accelerated_df: bool,
    /// Latched on frame-0 non-device-loss detokenizer CPU fallback so all later frames stay on CPU `detok_state`.
    detok_fallback_cpu: bool,
    /// Whether to detokenize and stream audio incrementally per frame,
    /// or buffer sampled codes and perform a single fused detokenization + iSTFT pass at finish.
    streaming: bool,
    /// Buffered sampled codes across frames (used when `streaming == false`).
    all_codes: Vec<i32>,
    /// Full codebook indices from the most recently decoded frame (8 for LFM2A, 12 for LFM3A).
    last_codes: Vec<i32>,
    /// Unified silence and loop safety watchdog.
    pub watchdog: AudioSilenceWatchdog,
}

impl<'a> Drop for AudioOutputDecoder<'a> {
    fn drop(&mut self) {
        if let Some(acc) = self.accelerator {
            acc.release_session();
        }
    }
}

fn is_device_loss_cera_err(err: &crate::CeraError) -> bool {
    err.to_string().to_ascii_lowercase().contains("device lost")
}

fn format_cera_err_detail(err: crate::CeraError) -> String {
    match err {
        crate::CeraError::Backend(detail) => detail,
        other => other.to_string(),
    }
}

impl<'a> AudioOutputDecoder<'a> {
    /// Total audio frames decoded so far in this session.
    pub fn audio_frames(&self) -> usize {
        self.audio_frames
    }

    /// Total audio samples streamed so far in this session.
    pub fn streamed_samples(&self) -> usize {
        self.streamed_samples
    }

    /// Full codebook indices of the most recently decoded frame (8 for LFM2A, 12 for LFM3A).
    pub fn last_frame_codes(&self) -> &[i32] {
        &self.last_codes
    }

    /// Sample rate of the vocoder in Hz (typically 24000).
    pub fn sample_rate(&self) -> u32 {
        self.detok_weights.config.sample_rate as u32
    }

    /// Total duration spent in depthformer frame sampling.
    pub fn time_depthformer(&self) -> Duration {
        self.time_depthformer
    }

    /// Total duration spent in detokenizer spectral processing.
    pub fn time_detokenizer(&self) -> Duration {
        self.time_detokenizer
    }

    /// Configure whether audio is detokenized per-frame or batched at finish.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Observe PCM output from a decoded frame and update silence/voiced counters.
    /// Returns the computed RMS energy.
    #[inline]
    pub fn observe_pcm(&mut self, pcm: &[f32]) -> f32 {
        if !self.streaming {
            self.watchdog.audio_frames_count += 1;
            return 0.0;
        }
        self.watchdog.observe_pcm(pcm)
    }

    /// Check if trailing silence has reached the termination threshold.
    #[inline]
    pub fn is_silence_terminated(&self) -> bool {
        self.streaming && self.watchdog.is_silence_terminated()
    }

    /// Check if the safety frame limit (4096 frames) has been reached.
    #[inline]
    pub fn is_safety_limit_reached(&self) -> bool {
        self.watchdog.is_safety_limit_reached()
    }

    #[inline]
    fn is_end_code(&self, code0: i32) -> bool {
        if self.weights.is_lfm3a_gen() {
            lfm3a_is_end_code(self.weights.decoder_config.n_vocab, code0)
        } else {
            code0 == AUDIO_END_CODE
        }
    }

    /// For `lfm3a_gen` (LFM3-Audio / LFM3-Omni), return the summed codebook feedback embedding
    /// for the end-of-speech frame `[eos_code; n_codebook]`, which closes an `audio_out` span
    /// in the LLM context so the backbone predicts `<|im_end|>`. Returns `None` for `lfm2a`.
    pub fn eos_feedback_embedding(&self) -> Option<Vec<f32>> {
        if self.weights.is_lfm3a_gen() {
            let eos = lfm3a_eos_code(self.weights.decoder_config.n_vocab);
            let n_cb = self.weights.decoder_config.n_codebook.max(1);
            let eos_codes = vec![eos; n_cb];
            Some(embed_audio_codes(self.weights, &eos_codes))
        } else {
            None
        }
    }

    /// Seed the internal Depthformer PRNG for deterministic audio codebook sampling.
    pub fn seed_rng(&mut self, seed: u64) {
        self.df_state.seed_rng(seed);
        if let Some(acc) = self.accelerator {
            acc.seed_rng(seed);
        }
    }

    pub fn new(
        weights: &'a AudioDecoderWeights,
        detok_weights: &'a DetokenizerWeights,
        accelerator: Option<&'a dyn AudioAccelerator>,
        audio_temperature: f32,
        audio_top_k: usize,
        gpu_depthformer: bool,
    ) -> Self {
        let initial_seed = crate::model::audio_decoder::non_zero_rng_seed(rand::random::<u64>());
        let active_accelerator = if (weights.is_lfm3a_gen() || detok_weights.is_lfm3a_gen())
            && !accelerator.is_some_and(|a| a.supports_lfm3a_gen_detokenizer())
        {
            None
        } else {
            match accelerator {
                Some(a) if a.try_acquire_session() => {
                    a.reset_detokenizer();
                    a.reset_depthformer();
                    a.seed_rng(initial_seed);
                    Some(a)
                }
                Some(_) => {
                    tracing::warn!(
                        "Audio accelerator busy with another session, falling back to CPU detokenization"
                    );
                    None
                }
                None => None,
            }
        };
        let mut df_state = DepthformerState::new(&weights.depthformer_config);
        df_state.seed_rng(initial_seed);
        let detok_state = DetokenizerState::new(&detok_weights.config);
        let streamer = crate::model::audio_decoder::IstftStreamer::new(
            detok_weights.config.n_fft,
            detok_weights.config.hop_length,
        );
        Self {
            weights,
            detok_weights,
            accelerator: active_accelerator,
            df_state,
            detok_state,
            streamer,
            all_spectrum: Vec::new(),
            audio_frames: 0,
            streamed_samples: 0,
            time_depthformer: Duration::ZERO,
            time_detokenizer: Duration::ZERO,
            audio_temperature,
            audio_top_k,
            use_accelerated_df: active_accelerator.is_some_and(|a| {
                let df_ok = if weights.is_lfm3a_gen() {
                    a.supports_lfm3a_gen_depthformer()
                } else {
                    a.supports_depthformer()
                };
                crate::model::audio_decoder::accelerated_depthformer_enabled(
                    df_ok,
                    a.depthformer_default_on(),
                    gpu_depthformer,
                )
            }),
            detok_fallback_cpu: false,
            streaming: true,
            all_codes: Vec::new(),
            last_codes: Vec::new(),
            watchdog: AudioSilenceWatchdog::new(),
        }
    }

    /// Whether this decoder is using hardware-accelerated depthformer.
    pub fn supports_accelerated_depthformer(&self) -> bool {
        self.use_accelerated_df
    }

    /// Backward-compatibility alias for [`Self::supports_accelerated_depthformer`].
    pub fn supports_gpu_depthformer(&self) -> bool {
        self.supports_accelerated_depthformer()
    }

    /// Sample one audio frame via depthformer, detect end-of-stream,
    /// detokenize into spectrum, and produce the feedback embedding the
    /// caller feeds back through the main LLM.
    ///
    /// `embed` is the main LLM's hidden state / embedding to condition
    /// this frame on (the audio_start token embedding on the first
    /// frame, or the prior frame's feedback embedding afterward).
    pub fn decode_frame(&mut self, embed: &[f32]) -> FrameOutcome {
        if self.weights.is_lfm3a_gen() {
            let t0 = Instant::now();
            let codes_vec = match (self.use_accelerated_df, self.accelerator) {
                (true, Some(a)) => {
                    let _ = a.take_audio_error();
                    let sampled =
                        a.sample_audio_frame_lfm3a(embed, self.audio_temperature, self.audio_top_k);
                    if let Some(e) = a.take_audio_error() {
                        let detail = match e {
                            crate::CeraError::Backend(detail) => detail,
                            other => other.to_string(),
                        };
                        return FrameOutcome::Fault(detail);
                    }
                    sampled.unwrap_or_else(|| {
                        sample_frame(
                            self.weights,
                            &mut self.df_state,
                            embed,
                            self.audio_temperature,
                            self.audio_top_k,
                        )
                    })
                }
                _ => sample_frame(
                    self.weights,
                    &mut self.df_state,
                    embed,
                    self.audio_temperature,
                    self.audio_top_k,
                ),
            };
            self.time_depthformer += t0.elapsed();

            if codes_vec
                .first()
                .copied()
                .is_some_and(|c| self.is_end_code(c))
            {
                self.last_codes.clone_from(&codes_vec);
                return FrameOutcome::End;
            }

            self.audio_frames += 1;
            self.last_codes.clone_from(&codes_vec);
            let mut codes = [0i32; 8];
            for (dst, &src) in codes.iter_mut().zip(codes_vec.iter()) {
                *dst = src;
            }
            let audio_embedding = embed_audio_codes(self.weights, &codes_vec);

            if !self.streaming {
                self.all_codes.extend_from_slice(&codes_vec);
                return FrameOutcome::Codes {
                    audio_embedding,
                    pcm: vec![],
                    codes,
                };
            }

            let t1 = Instant::now();
            let spectrum = match self.detokenize_frame_sync(&codes_vec) {
                Ok(s) => s,
                Err(detail) => return FrameOutcome::Fault(detail),
            };
            self.time_detokenizer += t1.elapsed();

            let pcm = self.streamer.feed_frames(&spectrum);
            self.streamed_samples += pcm.len();

            return FrameOutcome::Codes {
                audio_embedding,
                pcm,
                codes,
            };
        }

        let t0 = Instant::now();
        let codes = match (self.use_accelerated_df, self.accelerator) {
            (true, Some(a)) => {
                // Drain the backend fault slot around the bare sample:
                // a fault here must abort (the async path `?`s it), since
                // CPU state went cold while the accelerator drove sampling.
                let _ = a.take_audio_error();
                let codes = a.sample_audio_frame(embed, self.audio_temperature, self.audio_top_k);
                if let Some(e) = a.take_audio_error() {
                    // Backend details round-trip exactly; anything else
                    // degrades to its display text (audio faults are
                    // `Backend` by construction: every recorder passes it).
                    let detail = match e {
                        crate::CeraError::Backend(detail) => detail,
                        other => other.to_string(),
                    };
                    return FrameOutcome::Fault(detail);
                }
                codes
            }
            _ => sample_audio_frame(
                self.weights,
                &mut self.df_state,
                embed,
                self.audio_temperature,
                self.audio_top_k,
            ),
        };
        self.time_depthformer += t0.elapsed();

        if codes[0] == AUDIO_END_CODE {
            return FrameOutcome::End;
        }

        self.audio_frames += 1;
        self.last_codes.clear();
        self.last_codes.extend_from_slice(&codes);
        let audio_embedding = embed_audio_token(self.weights, &codes);

        if !self.streaming {
            self.all_codes.extend_from_slice(&codes);
            return FrameOutcome::Codes {
                audio_embedding,
                pcm: vec![],
                codes,
            };
        }

        let t1 = Instant::now();
        let spectrum = match self.detokenize_frame_sync(&codes) {
            Ok(s) => s,
            Err(detail) => return FrameOutcome::Fault(detail),
        };
        self.time_detokenizer += t1.elapsed();

        let pcm = self.streamer.feed_frames(&spectrum);
        self.streamed_samples += pcm.len();

        FrameOutcome::Codes {
            audio_embedding,
            pcm,
            codes,
        }
    }

    /// Async version of [`Self::decode_frame`] for WebGPU / browser wasm.
    pub async fn decode_frame_async(&mut self, embed: &[f32]) -> anyhow::Result<FrameOutcome> {
        if self.weights.is_lfm3a_gen() {
            return Ok(self.decode_frame(embed));
        }
        let t0 = Instant::now();
        let codes = match (self.use_accelerated_df, self.accelerator) {
            (true, Some(a)) => {
                a.sample_audio_frame_async(embed, self.audio_temperature, self.audio_top_k)
                    .await?
            }
            _ => sample_audio_frame(
                self.weights,
                &mut self.df_state,
                embed,
                self.audio_temperature,
                self.audio_top_k,
            ),
        };
        self.time_depthformer += t0.elapsed();

        if codes[0] == AUDIO_END_CODE {
            return Ok(FrameOutcome::End);
        }

        self.audio_frames += 1;
        self.last_codes.clear();
        self.last_codes.extend_from_slice(&codes);
        let audio_embedding = embed_audio_token(self.weights, &codes);

        if !self.streaming {
            self.all_codes.extend_from_slice(&codes);
            return Ok(FrameOutcome::Codes {
                audio_embedding,
                pcm: vec![],
                codes,
            });
        }

        let t1 = Instant::now();
        let spectrum = if !self.detok_fallback_cpu
            && let Some(a) = self.accelerator
        {
            a.detokenize_to_spectrum_async(self.detok_weights, &codes)
                .await?
        } else {
            detokenize_to_spectrum(
                self.detok_weights,
                self.weights,
                &mut self.detok_state,
                &codes,
            )
        };
        self.time_detokenizer += t1.elapsed();

        let pcm = self.streamer.feed_frames(&spectrum);
        self.streamed_samples += pcm.len();

        Ok(FrameOutcome::Codes {
            audio_embedding,
            pcm,
            codes,
        })
    }

    /// Async version of [`Self::decode_frame_async`] taking a GPU hidden buffer directly.
    #[cfg(feature = "gpu")]
    pub async fn decode_frame_from_gpu_hidden_async(
        &mut self,
        hidden_buf: &wgpu::Buffer,
    ) -> anyhow::Result<FrameOutcome> {
        let t0 = Instant::now();
        let codes = if let (true, Some(a)) = (self.use_accelerated_df, self.accelerator) {
            a.sample_audio_frame_from_gpu_hidden_async(
                hidden_buf,
                self.audio_temperature,
                self.audio_top_k,
            )
            .await?
        } else {
            anyhow::bail!("GPU hidden buffer handoff requires GPU depthformer");
        };
        self.time_depthformer += t0.elapsed();

        if codes[0] == AUDIO_END_CODE {
            return Ok(FrameOutcome::End);
        }

        self.audio_frames += 1;
        self.last_codes.clear();
        self.last_codes.extend_from_slice(&codes);
        let audio_embedding = embed_audio_token(self.weights, &codes);

        if !self.streaming {
            self.all_codes.extend_from_slice(&codes);
            return Ok(FrameOutcome::Codes {
                audio_embedding,
                pcm: vec![],
                codes,
            });
        }

        let t1 = Instant::now();
        let spectrum = if !self.detok_fallback_cpu
            && let Some(a) = self.accelerator
        {
            a.detokenize_to_spectrum_async(self.detok_weights, &codes)
                .await?
        } else {
            detokenize_to_spectrum(
                self.detok_weights,
                self.weights,
                &mut self.detok_state,
                &codes,
            )
        };
        self.time_detokenizer += t1.elapsed();

        let pcm = self.streamer.feed_frames(&spectrum);
        self.streamed_samples += pcm.len();

        Ok(FrameOutcome::Codes {
            audio_embedding,
            pcm,
            codes,
        })
    }

    /// Drain any remaining audio through the ISTFT pass, returning an error if
    /// batch detokenization or iSTFT aborts due to a mid-utterance or lost-device fault.
    /// Flushes any buffered streaming samples to the caller and returns the
    /// accumulated streamed sample count.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn try_finish(&mut self, mut sink: impl FnMut(&[f32], u32)) -> anyhow::Result<usize> {
        if self.streaming {
            let remaining = self.streamer.flush();
            if !remaining.is_empty() {
                sink(&remaining, self.detok_weights.config.sample_rate as u32);
                self.streamed_samples += remaining.len();
            }
            return Ok(self.streamed_samples);
        }

        if !self.all_codes.is_empty() && self.all_spectrum.is_empty() {
            let t1 = Instant::now();
            let frame_codebooks = if self.weights.is_lfm3a_gen() {
                self.weights.decoder_config.n_codebook.max(1)
            } else {
                8
            };
            let chunks = self.all_codes.chunks_exact(frame_codebooks);
            let remainder = chunks.remainder();
            let mut fallback_cpu = self.detok_fallback_cpu;
            for (chunk_idx, codes) in chunks.enumerate() {
                let spectrum = if !fallback_cpu && let Some(a) = self.accelerator {
                    let _ = a.take_audio_error();
                    let spectrum = a.detokenize_to_spectrum(self.detok_weights, codes);
                    if let Some(err) = a.take_audio_error() {
                        if chunk_idx > 0 || is_device_loss_cera_err(&err) {
                            tracing::warn!(
                                "GPU batch detokenize aborted at chunk {chunk_idx}: {err}"
                            );
                            eprintln!(
                                "[cera::audio] GPU batch detokenize aborted at chunk {chunk_idx}: {err}"
                            );
                            self.all_spectrum.clear();
                            self.all_codes.clear();
                            return Err(err.into());
                        }
                        tracing::warn!(
                            "GPU batch detokenize fault on chunk 0 ({err}); falling back to CPU"
                        );
                        eprintln!(
                            "[cera::audio] GPU batch detokenize fault on chunk 0 ({err}); falling back to CPU"
                        );
                        fallback_cpu = true;
                        self.detok_fallback_cpu = true;
                        detokenize_to_spectrum(
                            self.detok_weights,
                            self.weights,
                            &mut self.detok_state,
                            codes,
                        )
                    } else {
                        spectrum
                    }
                } else {
                    detokenize_to_spectrum(
                        self.detok_weights,
                        self.weights,
                        &mut self.detok_state,
                        codes,
                    )
                };
                self.all_spectrum.extend_from_slice(&spectrum);
            }
            if !remainder.is_empty() {
                tracing::warn!(
                    "AudioOutputDecoder: discarding {} trailing unaligned audio codes (expected multiple of {})",
                    remainder.len(),
                    frame_codebooks
                );
            }
            self.time_detokenizer += t1.elapsed();
        }

        if self.all_spectrum.is_empty() {
            return Ok(self.streamed_samples);
        }
        let n_fft = self.detok_weights.config.n_fft;
        let hop = self.detok_weights.config.hop_length;
        let pcm = match self.accelerator {
            Some(a) => {
                // ISTFT is stateless, so a non-lost transient accelerator fault
                // recomputes on CPU cleanly; device loss aborts via `try_finish`.
                let _ = a.take_audio_error();
                let pcm = a.istft_to_pcm(&self.all_spectrum, n_fft, hop);
                if let Some(err) = a.take_audio_error() {
                    if is_device_loss_cera_err(&err) {
                        self.all_spectrum.clear();
                        self.all_codes.clear();
                        return Err(err.into());
                    }
                    istft_to_pcm(&self.all_spectrum, n_fft, hop)
                } else {
                    pcm
                }
            }
            None => istft_to_pcm(&self.all_spectrum, n_fft, hop),
        };
        self.all_spectrum.clear();
        self.all_codes.clear();
        if pcm.is_empty() {
            return Ok(self.streamed_samples);
        }
        let n = pcm.len();
        sink(&pcm, self.detok_weights.config.sample_rate as u32);
        self.streamed_samples += n;
        Ok(self.streamed_samples)
    }

    /// Infallible compatibility wrapper around [`Self::try_finish`]; returns the
    /// accumulated streamed sample count even if batch detokenization aborted.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn finish(&mut self, sink: impl FnMut(&[f32], u32)) -> usize {
        self.try_finish(sink).unwrap_or(self.streamed_samples)
    }

    fn detokenize_frame_sync(&mut self, codes: &[i32]) -> Result<Vec<f32>, String> {
        if !self.detok_fallback_cpu
            && let Some(a) = self.accelerator
        {
            // Drain around the bare call; mid-utterance or lost-device faults
            // abort instead of running cold CPU `detok_state`, while frame-0
            // transient faults warn, latch CPU fallback, and stay on CPU.
            let _ = a.take_audio_error();
            let spectrum = a.detokenize_to_spectrum(self.detok_weights, codes);
            if let Some(err) = a.take_audio_error() {
                if self.audio_frames > 1 || is_device_loss_cera_err(&err) {
                    return Err(format_cera_err_detail(err));
                }
                self.detok_fallback_cpu = true;
                tracing::warn!("GPU detokenize fault on frame 0 ({err}); falling back to CPU");
                eprintln!(
                    "[cera::audio] GPU detokenize fault on frame 0 ({err}); falling back to CPU"
                );
                return Ok(detokenize_to_spectrum(
                    self.detok_weights,
                    self.weights,
                    &mut self.detok_state,
                    codes,
                ));
            }
            return Ok(spectrum);
        }
        Ok(detokenize_to_spectrum(
            self.detok_weights,
            self.weights,
            &mut self.detok_state,
            codes,
        ))
    }

    /// Async version of [`Self::try_finish`] for WebGPU / browser wasm.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub async fn finish_async(
        &mut self,
        mut sink: impl FnMut(&[f32], u32),
    ) -> anyhow::Result<usize> {
        if self.streaming {
            let remaining = self.streamer.flush();
            if !remaining.is_empty() {
                sink(&remaining, self.detok_weights.config.sample_rate as u32);
                self.streamed_samples += remaining.len();
            }
            return Ok(self.streamed_samples);
        }

        if !self.all_codes.is_empty() && self.all_spectrum.is_empty() {
            let t1 = Instant::now();
            let frame_codebooks = if self.weights.is_lfm3a_gen() {
                self.weights.decoder_config.n_codebook.max(1)
            } else {
                8
            };
            let chunks = self.all_codes.chunks_exact(frame_codebooks);
            let remainder = chunks.remainder();
            for codes in chunks {
                let spectrum = if !self.detok_fallback_cpu
                    && let Some(a) = self.accelerator
                {
                    match a
                        .detokenize_to_spectrum_async(self.detok_weights, codes)
                        .await
                    {
                        Ok(s) => s,
                        Err(err) => {
                            self.all_spectrum.clear();
                            self.all_codes.clear();
                            return Err(err);
                        }
                    }
                } else {
                    detokenize_to_spectrum(
                        self.detok_weights,
                        self.weights,
                        &mut self.detok_state,
                        codes,
                    )
                };
                self.all_spectrum.extend_from_slice(&spectrum);
            }
            if !remainder.is_empty() {
                tracing::warn!(
                    "AudioOutputDecoder: discarding {} trailing unaligned audio codes (expected multiple of {})",
                    remainder.len(),
                    frame_codebooks
                );
            }
            self.time_detokenizer += t1.elapsed();
        }

        if self.all_spectrum.is_empty() {
            return Ok(self.streamed_samples);
        }
        let n_fft = self.detok_weights.config.n_fft;
        let hop = self.detok_weights.config.hop_length;
        let pcm = match self.accelerator {
            Some(a) => match a.istft_to_pcm_async(&self.all_spectrum, n_fft, hop).await {
                Ok(pcm) => pcm,
                Err(err) => {
                    self.all_spectrum.clear();
                    self.all_codes.clear();
                    return Err(err);
                }
            },
            None => istft_to_pcm(&self.all_spectrum, n_fft, hop),
        };
        self.all_spectrum.clear();
        self.all_codes.clear();
        if pcm.is_empty() {
            return Ok(self.streamed_samples);
        }
        let n = pcm.len();
        sink(&pcm, self.detok_weights.config.sample_rate as u32);
        self.streamed_samples += n;
        Ok(self.streamed_samples)
    }
}

// ---------------------------------------------------------------------------
// generate_audio
// ---------------------------------------------------------------------------

/// Fail `generate_audio` on a recorded backend fault: the `Session`
/// `check_decode_error` equivalent for this standalone driver, which feeds
/// `&dyn Model` straight in and would otherwise sample zero logits as real
/// output and advance `pos` over the hole.
fn check_audio_decode_error(model: &dyn Model) -> Result<()> {
    if let Some(e) = model.take_decode_error() {
        return Err(e.into());
    }
    Ok(())
}

/// Generate text + audio from a model with vocoder.
///
/// `accelerator`: optional hardware accelerator backend for depthformer + detokenizer acceleration.
#[allow(unused_assignments, clippy::too_many_arguments)]
pub fn generate_audio(
    model: &dyn Model,
    decoder_weights: &AudioDecoderWeights,
    detok_weights: &DetokenizerWeights,
    tokenizer: &BpeTokenizer,
    prompt_tokens: &[u32],
    config: &AudioGenerateConfig,
    accelerator: Option<&dyn AudioAccelerator>,
    mut text_callback: impl FnMut(&str),
    mut audio_callback: impl FnMut(&[f32], u32),
) -> Result<AudioGenerateResult> {
    anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
    // Discard any stale decode error (mirrors `Session`): only this
    // attempt's own faults may fail it at the checks below.
    let _ = model.take_decode_error();

    let ctrl = ModalityControlTokens::from_tokenizer(tokenizer);
    let model_config = model.config();
    let mut state = InferenceState::from_config(model_config)?;
    // A fresh state starts at position 0, but backends that keep KV or
    // recurrent state on the model or device (Hexagon) would carry the last
    // generation's, or stay failed-closed after a panic. Reset them to match.
    // Best effort: backends without a checked reset return `Err` and are
    // unaffected; a real failure resurfaces on the first forward pass.
    let _ = model.try_reset_kv(&mut state, &KvCompression::None, model_config.max_seq_len);
    let mut sampler = Sampler::new(config.sampler.clone());
    let mut decoder = AudioOutputDecoder::new(
        decoder_weights,
        detok_weights,
        accelerator,
        config.audio_temperature,
        config.audio_top_k,
        config.gpu_depthformer,
    );

    let start = Instant::now();

    // Prefill through the chunked entry point (one chunk: `ubatch == 0`
    // disables session-level chunking) so a backend fault stops `consumed`
    // at the KV prefix that exists instead of advancing `pos` over a hole:
    // the `Session::append_tokens` accounting, minus cancellation (this
    // standalone driver has no cancel flag to poll).
    let no_cancel = AtomicBool::new(false);
    let (consumed, prefill_logits) =
        model.forward_prefill_chunked(prompt_tokens, 0, &mut state, 0, &no_cancel);
    check_audio_decode_error(model)?;
    if consumed < prompt_tokens.len() {
        anyhow::bail!(
            "prefill fault after {consumed}/{} tokens",
            prompt_tokens.len()
        );
    }
    let mut logits = match prefill_logits {
        Some(logits) => logits,
        None => anyhow::bail!(
            "prefill produced no logits for {} tokens",
            prompt_tokens.len()
        ),
    };

    let mut modality = Modality::Text;
    let mut generated = 0usize;
    let mut text_tokens = 0usize;
    let mut pos = consumed;

    // Interleaved mode counters.
    let cadence = decoder_weights.interleave;
    let mut interleaved = matches!(config.mode, AudioMode::Interleaved);
    let mut modality_budget = if interleaved {
        cadence.text // start with the vocoder's text budget
    } else {
        usize::MAX
    };
    let mut text_done = false;

    let mut next_token = sampler.sample(&mut logits);

    'outer: loop {
        if generated >= config.max_tokens || pos >= model_config.max_seq_len {
            break;
        }

        if modality == Modality::Text {
            // Check for EOG.
            if tokenizer.eos_token() == Some(next_token) {
                break;
            }

            if next_token == ctrl.mixed_start {
                interleaved = true;
                modality_budget = cadence.text;
                // a sampled token like any other: without it a run of this token is bounded
                // only by `max_seq_len`, not by `max_tokens`
                generated += 1;
                if generated >= config.max_tokens {
                    break;
                }
                logits = model.forward(&[next_token], pos, &mut state);
                check_audio_decode_error(model)?;
                next_token = sampler.sample(&mut logits);
                pos += 1;
                continue;
            }

            // Sequential mode: switch on audio_start token.
            if next_token == ctrl.audio_start {
                modality = Modality::Audio;
                modality_budget = if interleaved {
                    cadence.audio
                } else {
                    usize::MAX
                };
                continue;
            }

            if next_token == ctrl.text_end {
                text_done = true;
            }

            // Emit text token (excluding modality control markers).
            if next_token != ctrl.text_end
                && next_token != ctrl.audio_end
                && next_token != ctrl.mixed_start
            {
                let piece = tokenizer.decode(&[next_token]);
                text_callback(&piece);
                text_tokens += 1;
            }

            generated += 1;
            modality_budget = modality_budget.saturating_sub(1);

            if generated >= config.max_tokens {
                break;
            }

            // Interleaved: check budget AFTER consuming the current token.
            // When budget hits 0, use forward_embedding on this token to
            // extract the audio embedding. This matches the reference which
            // extracts from the decode of the LAST text token.
            if interleaved && (modality_budget == 0 || text_done) {
                let mut emb = model.forward_embedding(&[next_token], pos, &mut state);
                check_audio_decode_error(model)?;
                pos += 1;

                modality = Modality::Audio;
                modality_budget = cadence.audio;

                // Run audio loop with this embedding.
                loop {
                    if pos >= model_config.max_seq_len || decoder.is_safety_limit_reached() {
                        break 'outer;
                    }
                    let outcome = decoder.decode_frame(&emb);
                    let audio_emb = match outcome {
                        // Sampling faults and mid-utterance/lost-device detokenize faults
                        // abort generation cleanly instead of continuing on cold CPU state.
                        FrameOutcome::Fault(detail) => {
                            return Err(crate::CeraError::Backend(detail).into());
                        }
                        FrameOutcome::End => {
                            if let Some(eos_emb) = decoder.eos_feedback_embedding() {
                                logits = model.forward_from_embedding(&eos_emb, pos, &mut state);
                                check_audio_decode_error(model)?;
                                pos += 1;
                                if text_done {
                                    break;
                                }
                                next_token = sampler.sample(&mut logits);
                                break;
                            }
                            if text_done {
                                break;
                            }
                            logits = model.forward(&[ctrl.text_end], pos, &mut state);
                            check_audio_decode_error(model)?;
                            next_token = sampler.sample(&mut logits);
                            pos += 1;
                            break;
                        }
                        FrameOutcome::Codes {
                            audio_embedding,
                            pcm,
                            ..
                        } => {
                            decoder.observe_pcm(&pcm);
                            if !pcm.is_empty() {
                                audio_callback(&pcm, decoder.sample_rate());
                            }
                            if text_done && decoder.is_silence_terminated() {
                                break;
                            }
                            audio_embedding
                        }
                    };
                    modality_budget = modality_budget.saturating_sub(1);

                    if generated >= config.max_tokens || pos >= model_config.max_seq_len {
                        break 'outer;
                    }
                    if modality_budget == 0 && !text_done {
                        // Switch back to text. The reference transitions by
                        // decoding the last audio code embedding and sampling
                        // text from those logits (not by injecting TEXT_END).
                        logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
                        check_audio_decode_error(model)?;
                        next_token = sampler.sample(&mut logits);
                        pos += 1;
                        break;
                    }

                    emb = model.forward_hidden_from_embedding(&audio_emb, pos, &mut state);
                    check_audio_decode_error(model)?;
                    pos += 1;
                }

                if text_done {
                    break;
                }

                // Switch back to text.
                modality = Modality::Text;
                modality_budget = cadence.text;
                continue;
            }

            // Normal text: forward and sample next token.
            logits = model.forward(&[next_token], pos, &mut state);
            check_audio_decode_error(model)?;
            next_token = sampler.sample(&mut logits);
            pos += 1;
        } else {
            // Sequential audio mode: embedding from the audio_start token.
            let mut emb = model.forward_embedding(&[next_token], pos, &mut state);
            check_audio_decode_error(model)?;
            // The output norm naturally produces the right scale (~0.14 RMS)
            // when the hidden state has the activation outlier at channel 1455.
            pos += 1;
            generated += 1;

            loop {
                if pos >= model_config.max_seq_len || decoder.is_safety_limit_reached() {
                    break 'outer;
                }
                let outcome = decoder.decode_frame(&emb);
                let audio_emb = match outcome {
                    // Sampling faults and mid-utterance/lost-device detokenize faults
                    // abort generation cleanly instead of continuing on cold CPU state.
                    FrameOutcome::Fault(detail) => {
                        return Err(crate::CeraError::Backend(detail).into());
                    }
                    FrameOutcome::End => {
                        if !interleaved {
                            // Sequential TTS: audio is the final output.
                            // On `lfm3a_gen`, record the EOS frame embedding in the LLM
                            // state before exiting cleanly.
                            if let Some(eos_emb) = decoder.eos_feedback_embedding()
                                && pos < model_config.max_seq_len
                            {
                                let _ = model.forward_from_embedding(&eos_emb, pos, &mut state);
                                check_audio_decode_error(model)?;
                            }
                            break 'outer;
                        } else {
                            // Interleaved: transition back to text. Trailing
                            // silence detection and safety frame limits bound
                            // runaway cycles.
                            modality = Modality::Text;
                            text_done = true;
                            modality_budget = cadence.text;
                            if let Some(eos_emb) = decoder.eos_feedback_embedding() {
                                logits = model.forward_from_embedding(&eos_emb, pos, &mut state);
                            } else {
                                logits = model.forward(&[ctrl.text_end], pos, &mut state);
                            }
                            check_audio_decode_error(model)?;
                            next_token = sampler.sample(&mut logits);
                            pos += 1;
                            break;
                        }
                    }
                    FrameOutcome::Codes {
                        audio_embedding,
                        pcm,
                        ..
                    } => {
                        decoder.observe_pcm(&pcm);
                        if !pcm.is_empty() {
                            audio_callback(&pcm, decoder.sample_rate());
                        }
                        if (!interleaved || text_done) && decoder.is_silence_terminated() {
                            break 'outer;
                        }
                        audio_embedding
                    }
                };
                modality_budget = modality_budget.saturating_sub(1);

                if generated >= config.max_tokens || pos >= model_config.max_seq_len {
                    break;
                }
                if interleaved && modality_budget == 0 && !text_done {
                    modality = Modality::Text;
                    modality_budget = cadence.text;
                    logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
                    check_audio_decode_error(model)?;
                    next_token = sampler.sample(&mut logits);
                    pos += 1;
                    break;
                }

                // Feed codes back as embedding -> next hidden state.
                emb = model.forward_hidden_from_embedding(&audio_emb, pos, &mut state);
                check_audio_decode_error(model)?;
                pos += 1;
            }
        }
    }

    // Drain any remaining buffered samples or execute batch ISTFT, returning batch/device-loss errors.
    let audio_samples = decoder.try_finish(&mut audio_callback)?;

    Ok(AudioGenerateResult {
        text_tokens,
        audio_frames: decoder.audio_frames,
        audio_samples,
        elapsed_secs: start.elapsed().as_secs_f64(),
        depthformer_secs: decoder.time_depthformer.as_secs_f64(),
        detokenizer_secs: decoder.time_detokenizer.as_secs_f64(),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The end-of-speech code is the sampler's own mask/EOS id, `n_vocab - 1`. A fixed 256
    /// ended turns early on a larger vocabulary (256 is a real code there) and named a row
    /// past the table on a smaller one.
    #[test]
    fn lfm3a_end_code_is_the_last_vocabulary_id() {
        assert_eq!(lfm3a_eos_code(257), 256);
        assert_eq!(lfm3a_eos_code(2049), 2048);
        assert_eq!(lfm3a_eos_code(100), 99);
        assert!(lfm3a_is_end_code(257, 256));
        assert!(!lfm3a_is_end_code(257, 255));
        assert!(lfm3a_is_end_code(2049, 2048));
        assert!(
            !lfm3a_is_end_code(2049, 256),
            "256 is an ordinary code in a 2049 vocabulary"
        );
        assert!(lfm3a_is_end_code(100, 99));
        assert!(
            !lfm3a_is_end_code(100, 256),
            "256 is outside a 100-entry vocabulary"
        );
    }

    #[test]
    fn test_audio_silence_watchdog_initial_state() {
        let watchdog = AudioSilenceWatchdog::new();
        assert_eq!(watchdog.audio_frames_count, 0);
        assert_eq!(watchdog.total_voiced_frames, 0);
        assert_eq!(watchdog.consecutive_silent_frames, 0);
        assert!(!watchdog.is_silence_terminated());
        assert!(!watchdog.is_safety_limit_reached());
    }

    #[test]
    fn test_audio_silence_watchdog_unvoiced_cutoff() {
        let mut watchdog = AudioSilenceWatchdog::new();
        let silent_pcm = vec![0.0f32; 1920];

        for _ in 0..29 {
            watchdog.observe_pcm(&silent_pcm);
            assert!(!watchdog.is_silence_terminated());
        }
        watchdog.observe_pcm(&silent_pcm);
        assert!(watchdog.is_silence_terminated());
    }

    #[test]
    fn test_audio_silence_watchdog_voiced_cutoff() {
        let mut watchdog = AudioSilenceWatchdog::new();
        let voiced_pcm = vec![0.05f32; 1920];
        let silent_pcm = vec![0.0f32; 1920];

        for _ in 0..5 {
            watchdog.observe_pcm(&voiced_pcm);
        }
        assert_eq!(watchdog.total_voiced_frames, 5);
        assert_eq!(watchdog.consecutive_silent_frames, 0);
        assert!(!watchdog.is_silence_terminated());

        for _ in 0..24 {
            watchdog.observe_pcm(&silent_pcm);
            assert!(!watchdog.is_silence_terminated());
        }
        watchdog.observe_pcm(&silent_pcm);
        assert!(watchdog.is_silence_terminated());
    }

    #[test]
    fn test_audio_silence_watchdog_safety_limit() {
        let mut watchdog = AudioSilenceWatchdog::new();
        watchdog.audio_frames_count = AUDIO_SAFETY_FRAME_LIMIT - 1;
        assert!(!watchdog.is_safety_limit_reached());
        watchdog.audio_frames_count = AUDIO_SAFETY_FRAME_LIMIT;
        assert!(watchdog.is_safety_limit_reached());
    }

    #[test]
    fn test_audio_silence_watchdog_empty_pcm_resilience() {
        let mut watchdog = AudioSilenceWatchdog::new();
        let empty_pcm: [f32; 0] = [];
        let rms = watchdog.observe_pcm(&empty_pcm);
        assert_eq!(rms, 0.0);
        assert_eq!(watchdog.audio_frames_count, 1);
        assert_eq!(watchdog.consecutive_silent_frames, 1);
        assert_eq!(watchdog.total_voiced_frames, 0);
    }

    #[test]
    fn test_audio_silence_watchdog_custom_thresholds() {
        let mut watchdog = AudioSilenceWatchdog::new_with_thresholds(0.002, 3, 10, 8);
        assert_eq!(watchdog.rms_threshold, 0.002);
        assert_eq!(watchdog.min_voiced_frames, 3);
        assert_eq!(watchdog.silent_frames_cutoff, 10);
        assert_eq!(watchdog.silent_frames_cutoff_voiced, 8);

        let silent_pcm = vec![0.0f32; 1920];
        for _ in 0..9 {
            watchdog.observe_pcm(&silent_pcm);
            assert!(!watchdog.is_silence_terminated());
        }
        watchdog.observe_pcm(&silent_pcm);
        assert!(watchdog.is_silence_terminated());
    }

    #[test]
    fn test_audio_silence_watchdog_non_finite_resilience() {
        let mut watchdog = AudioSilenceWatchdog::new();
        let corrupted_pcm = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let rms = watchdog.observe_pcm(&corrupted_pcm);
        assert_eq!(rms, 0.0);
        assert_eq!(watchdog.audio_frames_count, 1);
        assert_eq!(watchdog.consecutive_silent_frames, 1);

        let mixed_pcm = vec![0.05f32, f32::NAN, 0.05f32];
        let rms_mixed = watchdog.observe_pcm(&mixed_pcm);
        assert!(rms_mixed > 0.04 && rms_mixed < 0.06);
        assert_eq!(watchdog.total_voiced_frames, 1);
        assert_eq!(watchdog.consecutive_silent_frames, 0);
    }

    /// Minimal `Model` with a poisonable decode-error slot, for pinning
    /// `check_audio_decode_error` without audio weight fixtures.
    struct PoisonableModel {
        slot: std::sync::Mutex<Option<crate::CeraError>>,
    }

    impl PoisonableModel {
        fn with_slot(err: Option<crate::CeraError>) -> Self {
            Self {
                slot: std::sync::Mutex::new(err),
            }
        }
    }

    impl Model for PoisonableModel {
        fn forward(&self, _tokens: &[u32], _pos: usize, _state: &mut InferenceState) -> Vec<f32> {
            Vec::new()
        }
        fn config(&self) -> &crate::model::ModelConfig {
            // Never read by the drain under test; the mock exists only to
            // carry the slot.
            unimplemented!("PoisonableModel carries no config")
        }
        fn take_decode_error(&self) -> Option<crate::CeraError> {
            crate::model::take_fault(&self.slot)
        }
    }

    #[test]
    fn audio_decode_error_drains_sticky_slot() {
        // Poisoned drains to Err once, then is clean (sticky until taken).
        let poisoned = PoisonableModel::with_slot(Some(crate::CeraError::Backend(
            "injected audio fault".into(),
        )));
        assert!(check_audio_decode_error(&poisoned).is_err());
        assert!(check_audio_decode_error(&poisoned).is_ok());
        // A clean model never trips the drain.
        let clean = PoisonableModel::with_slot(None);
        assert!(check_audio_decode_error(&clean).is_ok());
    }

    /// Fault-injecting `AudioAccelerator` double: scripted `take_audio_error`
    /// outcomes (one per take, in order; the script running dry reads as
    /// clean) plus garbage outputs no real computation produces, so a
    /// missed drain is observable.
    struct ScriptedFaultGpu {
        takes: std::sync::Mutex<std::collections::VecDeque<Option<crate::CeraError>>>,
        /// What `depthformer_default_on` reports (an opt-in accelerator by
        /// default, like Metal and wgpu).
        default_on: bool,
    }

    impl ScriptedFaultGpu {
        fn with_script(outcomes: Vec<Option<crate::CeraError>>) -> Self {
            Self {
                takes: std::sync::Mutex::new(outcomes.into()),
                default_on: false,
            }
        }

        /// One fault on the first drain (covers a discard-then-drain pair).
        fn fault_once() -> Self {
            Self::with_script(vec![
                None,
                Some(crate::CeraError::Backend("injected audio fault".into())),
            ])
        }
    }

    impl crate::model::audio_decoder::AudioAccelerator for ScriptedFaultGpu {
        fn sample_audio_frame(&self, _: &[f32], _: f32, _: usize) -> [i32; 8] {
            [1; 8]
        }

        fn detokenize_to_spectrum(
            &self,
            _: &crate::model::audio_decoder::DetokenizerWeights,
            _: &[i32],
        ) -> Vec<f32> {
            vec![7.0; 3]
        }

        fn istft_to_pcm(&self, _: &[f32], _: usize, _: usize) -> Vec<f32> {
            vec![999.0; 5]
        }

        fn reset_depthformer(&self) {}

        fn reset_detokenizer(&self) {}

        fn supports_depthformer(&self) -> bool {
            true
        }

        fn depthformer_default_on(&self) -> bool {
            self.default_on
        }

        fn take_audio_error(&self) -> Option<crate::CeraError> {
            self.takes.lock().unwrap().pop_front().flatten()
        }
    }

    #[test]
    fn gpu_sample_fault_is_frame_fault_not_codes() {
        // A sampling fault aborts the frame: no codes are trusted, and no
        // weight or streamer state is touched before the `Fault` return.
        let (dec, detok) = empty_vocoder_weights();
        let gpu = ScriptedFaultGpu::fault_once();
        let mut decoder = AudioOutputDecoder::new(&dec, &detok, Some(&gpu), 1.0, 1, true);
        let outcome = decoder.decode_frame(&[0.0; 4]);
        assert_eq!(outcome, FrameOutcome::Fault("injected audio fault".into()));
    }

    /// The accelerated depthformer needs support and either default trust or
    /// an explicit request: an opt-in accelerator (Metal, wgpu) must not run
    /// it without `gpu_depthformer`, a trusted one (Hexagon) always does.
    #[test]
    fn accelerated_depthformer_is_opt_in_unless_trusted() {
        let (dec, detok) = empty_vocoder_weights();
        let built = |gpu: &ScriptedFaultGpu, requested: bool| {
            AudioOutputDecoder::new(&dec, &detok, Some(gpu), 1.0, 1, requested)
                .supports_accelerated_depthformer()
        };
        let opt_in = ScriptedFaultGpu::with_script(vec![]);
        assert!(
            !built(&opt_in, false),
            "opt-in accelerator without the flag"
        );
        assert!(built(&opt_in, true), "opt-in accelerator with the flag");
        let trusted = ScriptedFaultGpu {
            default_on: true,
            ..ScriptedFaultGpu::with_script(vec![])
        };
        assert!(built(&trusted, false), "trusted accelerator needs no flag");
    }

    #[test]
    fn gpu_detok_fault_falls_back_to_cpu() {
        // Minimal head so the CPU fallback runs: 1-row zero codebook (all
        // codes embed to zero), unit output norm, zero 10-wide head
        // (matches the n_fft=8 streamer's 10-float frames).
        let (dec, mut detok) = empty_vocoder_weights();
        detok.output_norm = vec![1.0; 4];
        detok.emb_weight = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 4], 1, 4);
        detok.lin_w = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 40], 10, 4);
        detok.lin_b = vec![0.0; 10];
        // Sample clean (discard + drain), detokenize faults (discard + drain).
        let gpu = ScriptedFaultGpu::with_script(vec![
            None,
            None,
            None,
            Some(crate::CeraError::Backend("injected audio fault".into())),
        ]);
        let mut decoder = AudioOutputDecoder::new(&dec, &detok, Some(&gpu), 1.0, 1, true);
        let outcome = decoder.decode_frame(&[0.0; 4]);
        let FrameOutcome::Codes { pcm, codes, .. } = outcome else {
            panic!("detok fault must fall back, not abort: {outcome:?}");
        };
        assert_eq!(codes, [1; 8]);
        // Only the CPU recompute (all-zero spectrum) yields framed zero
        // PCM; the double's 3-float garbage cannot produce this shape.
        assert!(!pcm.is_empty() && pcm.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn gpu_istft_fault_recomputes_on_cpu() {
        let (dec, detok) = empty_vocoder_weights();
        let gpu = ScriptedFaultGpu::fault_once();
        let mut decoder =
            AudioOutputDecoder::new(&dec, &detok, Some(&gpu), 1.0, 1, true).with_streaming(false);
        // Preset spectrum with empty codes skips the detokenize loop, so
        // only the ISTFT drain runs.
        let spectrum = vec![0.5; 40];
        decoder.all_spectrum = spectrum.clone();
        let expected = crate::model::audio_decoder::istft_to_pcm(
            &spectrum,
            detok.config.n_fft,
            detok.config.hop_length,
        );
        let mut got = Vec::new();
        let n = decoder.finish(|pcm, _| got.extend_from_slice(pcm));
        assert_eq!(got, expected);
        assert_eq!(n, expected.len());
        assert!(!expected.is_empty());
    }

    /// Poll a future that never pends (straight-line test bodies only).
    fn block_on_ready<F: std::future::Future>(mut fut: F) -> F::Output {
        use std::task::{Context, Poll};
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        // SAFETY: the future is never moved after pinning.
        let mut pinned = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(out) => out,
            Poll::Pending => panic!("test future pended unexpectedly"),
        }
    }

    #[test]
    fn async_default_detok_drain_fails_loud() {
        // The trait's async defaults drain before `Ok`: a recorded fault
        // (the Metal path, which has no explicit async override) must
        // surface as `Err`, and a clean run passes the output through.
        let (_dec, detok) = empty_vocoder_weights();
        let gpu = ScriptedFaultGpu::fault_once();
        let out = block_on_ready(gpu.detokenize_to_spectrum_async(&detok, &[0; 8]));
        assert!(out.is_err());
        let clean = ScriptedFaultGpu::with_script(vec![None, None]);
        let out = block_on_ready(clean.detokenize_to_spectrum_async(&detok, &[0; 8]));
        assert_eq!(out.unwrap(), vec![7.0; 3]);
    }

    /// `Model` with a real config and a scripted `forward_prefill_chunked`,
    /// for driving `generate_audio`'s prefill-fault branches without silicon.
    struct ScriptedPrefillModel {
        config: crate::model::ModelConfig,
        consumed: usize,
        logits: Option<Vec<f32>>,
        /// How many times `generate_audio` asked for a checked KV reset.
        resets: std::sync::atomic::AtomicUsize,
        forwards: std::sync::atomic::AtomicUsize,
    }

    impl Model for ScriptedPrefillModel {
        fn forward(&self, tokens: &[u32], _pos: usize, state: &mut InferenceState) -> Vec<f32> {
            self.forwards
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            state.seq_len += tokens.len();
            vec![0.0; self.config.vocab_size]
        }
        fn config(&self) -> &crate::model::ModelConfig {
            &self.config
        }
        fn forward_prefill_chunked(
            &self,
            _tokens: &[u32],
            _start_pos: usize,
            _state: &mut InferenceState,
            _ubatch: usize,
            _cancel: &AtomicBool,
        ) -> (usize, Option<Vec<f32>>) {
            (self.consumed, self.logits.clone())
        }
        fn try_reset_kv(
            &self,
            _state: &mut InferenceState,
            _compression: &crate::kv_cache::KvCompression,
            _max_seq_len: usize,
        ) -> Result<(), crate::CeraError> {
            self.resets
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    fn scripted_model(consumed: usize, logits: Option<Vec<f32>>) -> ScriptedPrefillModel {
        ScriptedPrefillModel {
            config: crate::model::ModelConfig {
                architecture: "mock".into(),
                n_layers: 0,
                hidden_size: 0,
                intermediate_size: 0,
                n_heads: 0,
                n_kv_heads: 0,
                head_dim: 0,
                vocab_size: 100,
                max_seq_len: 1024,
                rope_theta: 0.0,
                rms_norm_eps: 0.0,
                block_types: Vec::new(),
                conv_kernel_size: None,
                ssm: None,
                kv_heads_per_layer: Vec::new(),
                scalars: crate::model::ScalarMultipliers::default(),
                moe: None,
                is_causal: true,
                class_labels: Vec::new(),
            },
            consumed,
            logits,
            resets: std::sync::atomic::AtomicUsize::new(0),
            forwards: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Vocoder weights with real (tiny) configs and empty owned tensors:
    /// `generate_audio`'s prefill bails fire before any decode, so only
    /// `AudioOutputDecoder::new` must construct (it reads configs only).
    /// `n_fft`/`hop_length` are nonzero because `IstftStreamer::new`
    /// plans a real FFT.
    pub(crate) fn empty_vocoder_weights() -> (AudioDecoderWeights, DetokenizerWeights) {
        use crate::model::audio_decoder::{
            CodebookWeights, DecoderConfig, DepthformerConfig, DetokenizerConfig,
        };
        use crate::model::weights::MmapWeight;
        let dec = AudioDecoderWeights {
            lfm3a_gen: false,
            depthformer_config: DepthformerConfig {
                n_layer: 0,
                n_embd: 4,
                n_head: 1,
                n_head_kv: 1,
                n_embd_head: 4,
                ffn_dim: 4,
                rms_norm_eps: 1e-5,
                rope_freq_base: 10000.0,
                max_seq_len: 8,
            },
            decoder_config: DecoderConfig {
                n_codebook: 1,
                n_vocab: 4,
                n_embd: 4,
                rms_norm_eps: 1e-5,
            },
            depthformer_layers: Vec::new(),
            depth_linear_w: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            depth_linear_b: Vec::new(),
            depth_embeddings: Vec::new(),
            audio_embedding: CodebookWeights {
                embedding: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
                norm: Vec::new(),
                to_logits: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            },
            interleave: InterleaveCadence::default(),
        };
        let detok = DetokenizerWeights {
            config: DetokenizerConfig {
                n_layer: 0,
                n_embd: 4,
                n_head: 1,
                n_head_kv: 1,
                n_embd_head: 4,
                ffn_dim: 4,
                d_conv: 0,
                rms_norm_eps: 1e-5,
                rope_freq_base: 10000.0,
                swa_window_size: 0,
                n_codes: 1,
                n_fft: 8,
                hop_length: 2,
                sample_rate: 24000,
                layer_is_conv: Vec::new(),
            },
            output_norm: Vec::new(),
            emb_weight: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            lin_w: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            lin_b: Vec::new(),
            layers: Vec::new(),
            lfm3_fsq: None,
        };
        (dec, detok)
    }

    fn audio_test_config() -> AudioGenerateConfig {
        AudioGenerateConfig {
            max_tokens: 4,
            sampler: SamplerConfig::default(),
            audio_temperature: 0.0,
            audio_top_k: 1,
            mode: AudioMode::Sequential,
            gpu_depthformer: false,
        }
    }

    #[test]
    fn mixed_start_at_the_token_limit_does_not_forward_again() {
        let mut logits = vec![-1.0; 132];
        logits[TOKEN_MIXED_START as usize] = 1.0;
        let mut model = scripted_model(1, Some(logits));
        model.config.vocab_size = 132;
        let (dec_w, detok_w) = empty_vocoder_weights();
        let tokenizer =
            BpeTokenizer::from_vocab((0..132).map(|n| n.to_string().into_bytes()).collect());
        let mut config = audio_test_config();
        config.max_tokens = 1;
        config.sampler.temperature = 0.0;
        generate_audio(
            &model,
            &dec_w,
            &detok_w,
            &tokenizer,
            &[1],
            &config,
            None,
            |_| {},
            |_, _| {},
        )
        .unwrap();
        assert_eq!(model.forwards.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn generate_audio_short_prefill_is_fault_not_silent() {
        let model = scripted_model(1, Some(vec![0.0; 100]));
        let (dec_w, detok_w) = empty_vocoder_weights();
        let tokenizer = BpeTokenizer::empty_for_test();
        let err = generate_audio(
            &model,
            &dec_w,
            &detok_w,
            &tokenizer,
            &[1, 2, 3],
            &audio_test_config(),
            None,
            |_| {},
            |_, _| {},
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("prefill fault after 1/3"),
            "{err:?}"
        );
    }

    /// A fresh generation must reset device-resident model state (Hexagon
    /// conv state, or a fail-closed poisoned device) before its first pass.
    #[test]
    fn generate_audio_resets_model_state_first() {
        let model = scripted_model(3, None);
        let (dec_w, detok_w) = empty_vocoder_weights();
        let tokenizer = BpeTokenizer::empty_for_test();
        let _ = generate_audio(
            &model,
            &dec_w,
            &detok_w,
            &tokenizer,
            &[1, 2, 3],
            &audio_test_config(),
            None,
            |_| {},
            |_, _| {},
        );
        assert_eq!(model.resets.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn generate_audio_missing_logits_bails() {
        let model = scripted_model(3, None);
        let (dec_w, detok_w) = empty_vocoder_weights();
        let tokenizer = BpeTokenizer::empty_for_test();
        let err = generate_audio(
            &model,
            &dec_w,
            &detok_w,
            &tokenizer,
            &[1, 2, 3],
            &audio_test_config(),
            None,
            |_| {},
            |_, _| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("produced no logits"), "{err:?}");
    }

    fn cadence_gguf(meta: &[(&str, u32)]) -> crate::gguf::GgufFile {
        let mut w = crate::convert::writer::GgufWriter::new();
        for (k, v) in meta {
            w.add_u32(*k, *v);
        }
        let mut bytes = Vec::new();
        w.write_header_and_tensor_info(&mut bytes).unwrap();
        crate::gguf::GgufFile::from_bytes(bytes.into()).unwrap()
    }

    #[test]
    fn interleave_cadence_reads_the_vocoders_declared_values() {
        let g = cadence_gguf(&[("interleaved_n_text", 6), ("interleaved_n_audio", 9)]);
        assert_eq!(
            InterleaveCadence::from_gguf(&g),
            InterleaveCadence { text: 6, audio: 9 }
        );
    }

    #[test]
    fn interleave_cadence_defaults_when_absent_and_per_key() {
        assert_eq!(
            InterleaveCadence::from_gguf(&cadence_gguf(&[])),
            InterleaveCadence::default()
        );
        assert_eq!(
            InterleaveCadence::default(),
            InterleaveCadence { text: 6, audio: 12 }
        );
        // One key present keeps the other's default.
        assert_eq!(
            InterleaveCadence::from_gguf(&cadence_gguf(&[("interleaved_n_audio", 9)])),
            InterleaveCadence { text: 6, audio: 9 }
        );
    }

    #[test]
    fn interleave_cadence_rejects_zero_budgets() {
        // A zero budget would switch modality on every token forever.
        let g = cadence_gguf(&[("interleaved_n_text", 0), ("interleaved_n_audio", 0)]);
        assert_eq!(
            InterleaveCadence::from_gguf(&g),
            InterleaveCadence::default()
        );
    }

    #[test]
    fn interleave_cadence_reads_clip_audio_gen_keys() {
        let g = cadence_gguf(&[
            ("clip.audio_gen.interleaved_n_text", 6),
            ("clip.audio_gen.interleaved_n_audio", 12),
        ]);
        assert_eq!(
            InterleaveCadence::from_gguf(&g),
            InterleaveCadence { text: 6, audio: 12 }
        );
    }

    #[test]
    fn modality_control_tokens_fallback_to_standard_ids() {
        let tok = BpeTokenizer::empty_for_test();
        let ctrl = ModalityControlTokens::from_tokenizer(&tok);
        assert_eq!(ctrl.audio_start, 128);
        assert_eq!(ctrl.audio_end, 129);
        assert_eq!(ctrl.text_end, 130);
        assert_eq!(ctrl.mixed_start, 131);
    }

    #[test]
    fn async_and_mid_utterance_detok_faults_abort_instead_of_running_cold_cpu_state() {
        let (dec, mut detok) = empty_vocoder_weights();
        detok.output_norm = vec![1.0; 4];
        detok.emb_weight = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 4], 1, 4);
        detok.lin_w = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 40], 10, 4);
        detok.lin_b = vec![0.0; 10];

        // 1. Async decode_frame_async must propagate detok Err via `?` (not fall back to CPU).
        let gpu_async = ScriptedFaultGpu::with_script(vec![
            None,
            None,
            None,
            Some(crate::CeraError::Backend(
                "GPU device lost (async detok)".into(),
            )),
        ]);
        let mut dec_async = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_async), 1.0, 1, true);
        let async_err = block_on_ready(dec_async.decode_frame_async(&[0.0; 4])).unwrap_err();
        assert!(format!("{async_err:#}").contains("async detok"));

        // 2. Sync decode_frame must abort (FrameOutcome::Fault) on frame-0 device loss.
        let gpu_sync_lost = ScriptedFaultGpu::with_script(vec![
            None,
            None,
            None,
            Some(crate::CeraError::Backend(
                "GPU device lost (sync detok)".into(),
            )),
        ]);
        let mut dec_lost =
            AudioOutputDecoder::new(&dec, &detok, Some(&gpu_sync_lost), 1.0, 1, true);
        assert_eq!(
            dec_lost.decode_frame(&[0.0; 4]),
            FrameOutcome::Fault("GPU device lost (sync detok)".into())
        );

        // 3. Sync decode_frame mid-utterance (frame 2) must abort rather than running cold CPU
        //    `detok_state` after frame 1 executed on GPU.
        let gpu_mid = ScriptedFaultGpu::with_script(vec![
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(crate::CeraError::Backend(
                "mid-utterance detok fault".into(),
            )),
        ]);
        let mut dec_mid = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_mid), 1.0, 1, true);
        let _first = dec_mid.decode_frame(&[0.0; 4]);
        assert_eq!(
            dec_mid.decode_frame(&[0.0; 4]),
            FrameOutcome::Fault("mid-utterance detok fault".into())
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn decode_frame_from_gpu_hidden_async_propagates_detok_error() {
        let Some(ctx) = crate::backend::wgpu::gpu_or_skip() else {
            return;
        };
        let hidden_buf = ctx.create_storage_rw(16, "test_audio_hidden");
        let (dec, mut detok) = empty_vocoder_weights();
        detok.output_norm = vec![1.0; 4];
        detok.emb_weight = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 4], 1, 4);
        detok.lin_w = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 40], 10, 4);
        detok.lin_b = vec![0.0; 10];

        struct HiddenSampleFaultDetokGpu {
            takes: std::sync::Mutex<std::collections::VecDeque<Option<crate::CeraError>>>,
        }
        impl crate::model::audio_decoder::AudioAccelerator for HiddenSampleFaultDetokGpu {
            fn sample_audio_frame(&self, _: &[f32], _: f32, _: usize) -> [i32; 8] {
                [1; 8]
            }
            fn sample_audio_frame_from_gpu_hidden_async<'a>(
                &'a self,
                _: &'a wgpu::Buffer,
                _: f32,
                _: usize,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = anyhow::Result<[i32; 8]>> + Send + 'a>,
            > {
                Box::pin(async { Ok([1; 8]) })
            }
            fn detokenize_to_spectrum(
                &self,
                _: &crate::model::audio_decoder::DetokenizerWeights,
                _: &[i32],
            ) -> Vec<f32> {
                vec![7.0; 3]
            }
            fn istft_to_pcm(&self, _: &[f32], _: usize, _: usize) -> Vec<f32> {
                Vec::new()
            }
            fn reset_depthformer(&self) {}
            fn reset_detokenizer(&self) {}
            fn supports_depthformer(&self) -> bool {
                true
            }
            fn take_audio_error(&self) -> Option<crate::CeraError> {
                self.takes.lock().unwrap().pop_front().flatten()
            }
        }
        let gpu = HiddenSampleFaultDetokGpu {
            takes: std::sync::Mutex::new(std::collections::VecDeque::from([
                None,
                Some(crate::CeraError::Backend(
                    "hidden async detok failed".into(),
                )),
            ])),
        };
        let mut dec_hidden = AudioOutputDecoder::new(&dec, &detok, Some(&gpu), 1.0, 1, true);
        let err =
            block_on_ready(dec_hidden.decode_frame_from_gpu_hidden_async(&hidden_buf)).unwrap_err();
        assert!(format!("{err:#}").contains("hidden async detok failed"));
    }

    #[test]
    fn frame0_detok_fallback_latches_cpu_and_skips_accelerator_on_later_frames() {
        let (dec, mut detok) = empty_vocoder_weights();
        detok.output_norm = vec![1.0; 4];
        detok.emb_weight = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 4], 1, 4);
        detok.lin_w = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 40], 10, 4);
        detok.lin_b = vec![0.0; 10];

        struct CountingDetokGpu {
            takes: std::sync::Mutex<std::collections::VecDeque<Option<crate::CeraError>>>,
            detok_calls: std::sync::atomic::AtomicUsize,
        }
        impl crate::model::audio_decoder::AudioAccelerator for CountingDetokGpu {
            fn sample_audio_frame(&self, _: &[f32], _: f32, _: usize) -> [i32; 8] {
                [1; 8]
            }
            fn detokenize_to_spectrum(
                &self,
                _: &crate::model::audio_decoder::DetokenizerWeights,
                _: &[i32],
            ) -> Vec<f32> {
                self.detok_calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                vec![7.0; 3]
            }
            fn istft_to_pcm(&self, _: &[f32], _: usize, _: usize) -> Vec<f32> {
                Vec::new()
            }
            fn reset_depthformer(&self) {}
            fn reset_detokenizer(&self) {}
            fn supports_depthformer(&self) -> bool {
                true
            }
            fn take_audio_error(&self) -> Option<crate::CeraError> {
                self.takes.lock().unwrap().pop_front().flatten()
            }
        }

        let gpu = CountingDetokGpu {
            takes: std::sync::Mutex::new(std::collections::VecDeque::from([
                None,
                None,
                None,
                Some(crate::CeraError::Backend(
                    "transient frame 0 shader fault".into(),
                )),
                // Frame 1 and frame 2 keep sampling clean while skipping accelerator detokenization.
                None,
                None,
                None,
                None,
            ])),
            detok_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut decoder = AudioOutputDecoder::new(&dec, &detok, Some(&gpu), 1.0, 1, true);
        let f0 = decoder.decode_frame(&[0.0; 4]);
        assert!(matches!(f0, FrameOutcome::Codes { .. }));
        assert_eq!(
            gpu.detok_calls.load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        let f1 = decoder.decode_frame(&[0.0; 4]);
        assert!(matches!(f1, FrameOutcome::Codes { .. }));
        assert_eq!(
            gpu.detok_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "frame 1 must stay on CPU detok_state once frame 0 fell back"
        );

        let f2 = block_on_ready(decoder.decode_frame_async(&[0.0; 4])).unwrap();
        assert!(matches!(f2, FrameOutcome::Codes { .. }));
        assert_eq!(
            gpu.detok_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "async decode_frame_async must also stay on CPU detok_state once latched"
        );
    }

    #[test]
    fn try_finish_batch_mode_aborts_on_mid_utterance_chunk_or_device_loss() {
        let (dec, mut detok) = empty_vocoder_weights();
        detok.output_norm = vec![1.0; 4];
        detok.emb_weight = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 4], 1, 4);
        detok.lin_w = crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 40], 10, 4);
        detok.lin_b = vec![0.0; 10];

        // 1. Mid-utterance batch chunk 1 transient fault returns Err and clears buffers.
        let gpu_chunk1 = ScriptedFaultGpu::with_script(vec![
            None,
            None, // sample frame 0
            None,
            None, // sample frame 1
            None,
            None, // batch detok chunk 0 ok
            None,
            Some(crate::CeraError::Backend("chunk 1 detok failed".into())),
        ]);
        let mut dec1 = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_chunk1), 1.0, 1, true)
            .with_streaming(false);
        assert!(matches!(
            dec1.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        assert!(matches!(
            dec1.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        let err1 = dec1.try_finish(|_, _| {}).unwrap_err();
        assert!(format!("{err1:#}").contains("chunk 1 detok failed"));
        assert!(dec1.all_codes.is_empty());
        assert!(dec1.all_spectrum.is_empty());

        // 2. Chunk 0 device-loss fault returns Err (never falls back onto dead device).
        let gpu_chunk0_lost = ScriptedFaultGpu::with_script(vec![
            None,
            None, // sample frame 0
            None,
            Some(crate::CeraError::Backend(
                "GPU device lost during chunk 0 detok".into(),
            )),
        ]);
        let mut dec2 = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_chunk0_lost), 1.0, 1, true)
            .with_streaming(false);
        assert!(matches!(
            dec2.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        let err2 = dec2.try_finish(|_, _| {}).unwrap_err();
        assert!(format!("{err2:#}").contains("device lost"));
        assert!(dec2.all_codes.is_empty());

        // 3. Batch iSTFT device-loss returns Err; transient non-lost iSTFT fault recomputes on CPU.
        let gpu_istft_lost = ScriptedFaultGpu::with_script(vec![
            None,
            None, // sample frame 0
            None,
            None, // chunk 0 detok ok
            None,
            Some(crate::CeraError::Backend(
                "GPU device lost during istft".into(),
            )),
        ]);
        let mut dec3 = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_istft_lost), 1.0, 1, true)
            .with_streaming(false);
        assert!(matches!(
            dec3.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        let err3 = dec3.try_finish(|_, _| {}).unwrap_err();
        assert!(format!("{err3:#}").contains("istft"));

        let gpu_istft_transient = ScriptedFaultGpu::fault_once();
        let mut dec4 =
            AudioOutputDecoder::new(&dec, &detok, Some(&gpu_istft_transient), 1.0, 1, true)
                .with_streaming(false);
        dec4.all_spectrum = vec![0.5; 40];
        let mut got_pcm = Vec::new();
        let ok_samples = dec4
            .try_finish(|pcm, _| got_pcm.extend_from_slice(pcm))
            .unwrap();
        assert!(ok_samples > 0);
        assert_eq!(got_pcm.len(), ok_samples);
    }

    #[test]
    fn finish_async_batch_mode_clears_buffers_on_detok_and_istft_error() {
        let (dec, detok) = empty_vocoder_weights();

        // 1. Mid-loop async batch detokenize failure clears partial `all_spectrum` and `all_codes`.
        let gpu_detok_err = ScriptedFaultGpu::with_script(vec![
            None,
            None, // sample frame 0
            None,
            None, // sample frame 1
            None,
            None, // async detok chunk 0 ok (pushes partial spectrum)
            None,
            Some(crate::CeraError::Backend(
                "async batch chunk 1 detok failed".into(),
            )),
        ]);
        let mut dec1 = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_detok_err), 1.0, 1, true)
            .with_streaming(false);
        assert!(matches!(
            dec1.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        assert!(matches!(
            dec1.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        let err1 = block_on_ready(dec1.finish_async(|_, _| {})).unwrap_err();
        assert!(format!("{err1:#}").contains("chunk 1 detok failed"));
        assert!(dec1.all_spectrum.is_empty());
        assert!(dec1.all_codes.is_empty());
        assert_eq!(dec1.try_finish(|_, _| {}).unwrap(), 0);

        // 2. Async batch iSTFT failure clears `all_spectrum` and `all_codes`.
        let gpu_istft_err = ScriptedFaultGpu::with_script(vec![
            None,
            None, // sample frame 0
            None,
            None, // async detok chunk 0 ok
            None,
            Some(crate::CeraError::Backend("async batch istft failed".into())),
        ]);
        let mut dec2 = AudioOutputDecoder::new(&dec, &detok, Some(&gpu_istft_err), 1.0, 1, true)
            .with_streaming(false);
        assert!(matches!(
            dec2.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        let err2 = block_on_ready(dec2.finish_async(|_, _| {})).unwrap_err();
        assert!(format!("{err2:#}").contains("async batch istft failed"));
        assert!(dec2.all_spectrum.is_empty());
        assert!(dec2.all_codes.is_empty());
        assert_eq!(dec2.finish(|_, _| {}), 0);
    }

    #[test]
    fn lfm3a_gen_accelerated_depthformer_and_fault_handling() {
        struct MockLfm3aAccel {
            fail: bool,
            calls: std::sync::atomic::AtomicUsize,
            err: std::sync::Mutex<Option<crate::CeraError>>,
        }
        impl crate::model::audio_decoder::AudioAccelerator for MockLfm3aAccel {
            fn sample_audio_frame(&self, _: &[f32], _: f32, _: usize) -> [i32; 8] {
                unreachable!("lfm3a_gen path must call sample_audio_frame_lfm3a");
            }
            fn supports_lfm3a_gen_depthformer(&self) -> bool {
                true
            }
            fn sample_audio_frame_lfm3a(&self, _: &[f32], _: f32, _: usize) -> Option<Vec<i32>> {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if self.fail {
                    *self.err.lock().unwrap() =
                        Some(crate::CeraError::Backend("lfm3a df fault".into()));
                    None
                } else {
                    Some(vec![12; 12])
                }
            }
            fn detokenize_to_spectrum(
                &self,
                _: &crate::model::audio_decoder::DetokenizerWeights,
                _: &[i32],
            ) -> Vec<f32> {
                vec![0.0; 10]
            }
            fn reset_depthformer(&self) {}
            fn reset_detokenizer(&self) {}
            fn supports_depthformer(&self) -> bool {
                true
            }
            fn supports_lfm3a_gen_detokenizer(&self) -> bool {
                true
            }
            fn depthformer_default_on(&self) -> bool {
                true
            }
            fn take_audio_error(&self) -> Option<crate::CeraError> {
                self.err.lock().unwrap().take()
            }
        }

        let (mut dec, mut detok) = empty_vocoder_weights();
        dec.lfm3a_gen = true;
        dec.decoder_config.n_codebook = 12;
        dec.decoder_config.n_vocab = 257;
        dec.audio_embedding.embedding =
            crate::model::weights::MmapWeight::from_owned_f32(vec![0.1; 12 * 257 * 4], 12 * 257, 4);
        detok.config.n_codes = 12;
        detok.emb_weight =
            crate::model::weights::MmapWeight::from_owned_f32(vec![0.0; 12 * 257 * 4], 12 * 257, 4);

        let ok_acc = MockLfm3aAccel {
            fail: false,
            calls: std::sync::atomic::AtomicUsize::new(0),
            err: std::sync::Mutex::new(None),
        };
        let mut decoder = AudioOutputDecoder::new(&dec, &detok, Some(&ok_acc), 0.0, 1, false)
            .with_streaming(false);
        assert!(decoder.supports_accelerated_depthformer());
        assert!(matches!(
            decoder.decode_frame(&[0.0; 4]),
            FrameOutcome::Codes { .. }
        ));
        assert_eq!(ok_acc.calls.load(std::sync::atomic::Ordering::Relaxed), 1);

        let err_acc = MockLfm3aAccel {
            fail: true,
            calls: std::sync::atomic::AtomicUsize::new(0),
            err: std::sync::Mutex::new(None),
        };
        let mut fault_dec = AudioOutputDecoder::new(&dec, &detok, Some(&err_acc), 0.0, 1, false);
        assert_eq!(
            fault_dec.decode_frame(&[0.0; 4]),
            FrameOutcome::Fault("lfm3a df fault".into())
        );
    }
}

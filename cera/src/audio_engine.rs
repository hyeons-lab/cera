//! Audio-aware generation loop with text ↔ audio modality switching.

use anyhow::Result;

use crate::kv_cache::InferenceState;
use crate::model::Model;
use crate::model::audio_decoder::{
    AudioDecoderWeights, AudioGpu, DepthformerState, DetokenizerState, DetokenizerWeights,
    detokenize_to_spectrum, embed_audio_token, istft_to_pcm, sample_audio_frame,
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
    /// Use GPU for depthformer (code sampling). Disabled by default because
    /// GEMV accumulation order differences can produce different codes.
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
pub const TOKEN_TEXT_END: u32 = 130;
pub const AUDIO_END_CODE: i32 = 2048;
pub const DEFAULT_INTERLEAVED_TEXT_BUDGET: usize = 6;
pub const DEFAULT_INTERLEAVED_AUDIO_BUDGET: usize = 12;
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
}

/// Owns the audio output-decoder state and exposes per-frame operations.
/// Extracted from `generate_audio` so the same per-frame logic is shared
/// between the Sequential and Interleaved paths and across CPU/WebGPU sessions.
pub struct AudioOutputDecoder<'a> {
    weights: &'a AudioDecoderWeights,
    detok_weights: &'a DetokenizerWeights,
    gpu: Option<&'a dyn AudioGpu>,
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
    /// Precomputed: `gpu.is_some() && config.gpu_depthformer`.
    use_gpu_df: bool,
    /// Whether to detokenize and stream audio incrementally per frame,
    /// or buffer sampled codes and perform a single fused detokenization + iSTFT pass at finish.
    streaming: bool,
    /// Buffered sampled codes across frames (used when `streaming == false`).
    all_codes: Vec<i32>,
    /// Unified silence and loop safety watchdog.
    pub watchdog: AudioSilenceWatchdog,
}

impl<'a> Drop for AudioOutputDecoder<'a> {
    fn drop(&mut self) {
        if let Some(g) = self.gpu {
            g.release_session();
        }
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

    pub fn new(
        weights: &'a AudioDecoderWeights,
        detok_weights: &'a DetokenizerWeights,
        gpu: Option<&'a dyn AudioGpu>,
        audio_temperature: f32,
        audio_top_k: usize,
        gpu_depthformer: bool,
    ) -> Self {
        let active_gpu = match gpu {
            Some(g) if g.try_acquire_session() => {
                g.reset_detokenizer();
                g.reset_depthformer();
                Some(g)
            }
            Some(_) => {
                tracing::warn!(
                    "GPU audio decoder busy with another session, falling back to CPU detokenization"
                );
                None
            }
            None => None,
        };
        let df_state = DepthformerState::new(&weights.depthformer_config);
        let detok_state = DetokenizerState::new(&detok_weights.config);
        let streamer = crate::model::audio_decoder::IstftStreamer::new(
            detok_weights.config.n_fft,
            detok_weights.config.hop_length,
        );
        Self {
            weights,
            detok_weights,
            gpu: active_gpu,
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
            use_gpu_df: gpu_depthformer && active_gpu.is_some_and(|g| g.supports_depthformer()),
            streaming: true,
            all_codes: Vec::new(),
            watchdog: AudioSilenceWatchdog::new(),
        }
    }

    /// Whether this decoder is using the GPU depthformer.
    pub fn supports_gpu_depthformer(&self) -> bool {
        self.use_gpu_df
    }

    /// Sample one audio frame via depthformer, detect end-of-stream,
    /// detokenize into spectrum, and produce the feedback embedding the
    /// caller feeds back through the main LLM.
    ///
    /// `embed` is the main LLM's hidden state / embedding to condition
    /// this frame on (the audio_start token embedding on the first
    /// frame, or the prior frame's feedback embedding afterward).
    pub fn decode_frame(&mut self, embed: &[f32]) -> FrameOutcome {
        let t0 = Instant::now();
        let codes = match (self.use_gpu_df, self.gpu) {
            (true, Some(g)) => {
                g.sample_audio_frame(embed, self.audio_temperature, self.audio_top_k)
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
        let spectrum = if let Some(g) = self.gpu {
            g.detokenize_to_spectrum(self.detok_weights, &codes)
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

        FrameOutcome::Codes {
            audio_embedding,
            pcm,
            codes,
        }
    }

    /// Async version of [`Self::decode_frame`] for WebGPU / browser wasm.
    pub async fn decode_frame_async(&mut self, embed: &[f32]) -> anyhow::Result<FrameOutcome> {
        let t0 = Instant::now();
        let codes = match (self.use_gpu_df, self.gpu) {
            (true, Some(g)) => {
                g.sample_audio_frame_async(embed, self.audio_temperature, self.audio_top_k)
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
        let spectrum = if let Some(g) = self.gpu {
            match g
                .detokenize_to_spectrum_async(self.detok_weights, &codes)
                .await
            {
                Ok(spec) => spec,
                Err(_) => detokenize_to_spectrum(
                    self.detok_weights,
                    self.weights,
                    &mut self.detok_state,
                    &codes,
                ),
            }
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
        let codes = if let (true, Some(g)) = (self.use_gpu_df, self.gpu) {
            g.sample_audio_frame_from_gpu_hidden_async(
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
        let spectrum = if let Some(g) = self.gpu {
            match g
                .detokenize_to_spectrum_async(self.detok_weights, &codes)
                .await
            {
                Ok(spec) => spec,
                Err(_) => detokenize_to_spectrum(
                    self.detok_weights,
                    self.weights,
                    &mut self.detok_state,
                    &codes,
                ),
            }
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

    /// Drain any remaining audio through the ISTFT pass.
    /// Flushes any buffered streaming samples to the caller and returns the
    /// accumulated streamed sample count.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn finish(&mut self, mut sink: impl FnMut(&[f32], u32)) -> usize {
        if self.streaming {
            let remaining = self.streamer.flush();
            if !remaining.is_empty() {
                sink(&remaining, self.detok_weights.config.sample_rate as u32);
                self.streamed_samples += remaining.len();
            }
            return self.streamed_samples;
        }

        if !self.all_codes.is_empty() && self.all_spectrum.is_empty() {
            let t1 = Instant::now();
            let chunks = self.all_codes.chunks_exact(8);
            let remainder = chunks.remainder();
            for codes in chunks {
                let spectrum = if let Some(g) = self.gpu {
                    g.detokenize_to_spectrum(self.detok_weights, codes)
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
                    "AudioOutputDecoder: discarding {} trailing unaligned audio codes (expected multiple of 8)",
                    remainder.len()
                );
            }
            self.time_detokenizer += t1.elapsed();
        }

        if self.all_spectrum.is_empty() {
            return self.streamed_samples;
        }
        let n_fft = self.detok_weights.config.n_fft;
        let hop = self.detok_weights.config.hop_length;
        let pcm = match self.gpu {
            Some(g) => g.istft_to_pcm(&self.all_spectrum, n_fft, hop),
            None => istft_to_pcm(&self.all_spectrum, n_fft, hop),
        };
        self.all_spectrum.clear();
        self.all_codes.clear();
        if pcm.is_empty() {
            return self.streamed_samples;
        }
        let n = pcm.len();
        sink(&pcm, self.detok_weights.config.sample_rate as u32);
        self.streamed_samples += n;
        self.streamed_samples
    }

    /// Async version of [`Self::finish`] for WebGPU / browser wasm.
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
            let chunks = self.all_codes.chunks_exact(8);
            let remainder = chunks.remainder();
            for codes in chunks {
                let spectrum = if let Some(g) = self.gpu {
                    g.detokenize_to_spectrum_async(self.detok_weights, codes)
                        .await?
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
                    "AudioOutputDecoder: discarding {} trailing unaligned audio codes (expected multiple of 8)",
                    remainder.len()
                );
            }
            self.time_detokenizer += t1.elapsed();
        }

        if self.all_spectrum.is_empty() {
            return Ok(self.streamed_samples);
        }
        let n_fft = self.detok_weights.config.n_fft;
        let hop = self.detok_weights.config.hop_length;
        let pcm = match self.gpu {
            Some(g) => g.istft_to_pcm_async(&self.all_spectrum, n_fft, hop).await?,
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

/// Generate text + audio from a model with vocoder.
///
/// `gpu`: optional GPU backend for depthformer + detokenizer acceleration.
#[allow(unused_assignments, clippy::too_many_arguments)]
pub fn generate_audio(
    model: &dyn Model,
    decoder_weights: &AudioDecoderWeights,
    detok_weights: &DetokenizerWeights,
    tokenizer: &BpeTokenizer,
    prompt_tokens: &[u32],
    config: &AudioGenerateConfig,
    gpu: Option<&dyn AudioGpu>,
    mut text_callback: impl FnMut(&str),
    mut audio_callback: impl FnMut(&[f32], u32),
) -> Result<AudioGenerateResult> {
    anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");

    let model_config = model.config();
    let mut state = InferenceState::from_config(model_config)?;
    let mut sampler = Sampler::new(config.sampler.clone());
    let mut decoder = AudioOutputDecoder::new(
        decoder_weights,
        detok_weights,
        gpu,
        config.audio_temperature,
        config.audio_top_k,
        config.gpu_depthformer,
    );

    let start = Instant::now();

    // Prefill.
    let mut logits = model.forward_prefill(prompt_tokens, 0, &mut state);

    let mut modality = Modality::Text;
    let mut generated = 0usize;
    let mut text_tokens = 0usize;
    let mut pos = prompt_tokens.len();

    // Interleaved mode counters.
    let mut modality_budget = match config.mode {
        AudioMode::Interleaved => DEFAULT_INTERLEAVED_TEXT_BUDGET, // start with default text tokens
        AudioMode::Sequential => usize::MAX,
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

            // Sequential mode: switch on audio_start token.
            if next_token == TOKEN_AUDIO_START {
                modality = Modality::Audio;
                modality_budget = match config.mode {
                    AudioMode::Interleaved => DEFAULT_INTERLEAVED_AUDIO_BUDGET,
                    AudioMode::Sequential => usize::MAX,
                };
                continue;
            }

            if next_token == TOKEN_TEXT_END {
                text_done = true;
            }

            // Emit text token.
            if next_token != TOKEN_TEXT_END {
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
            if matches!(config.mode, AudioMode::Interleaved) && (modality_budget == 0 || text_done)
            {
                let mut emb = model.forward_embedding(&[next_token], pos, &mut state);
                pos += 1;

                modality = Modality::Audio;
                modality_budget = DEFAULT_INTERLEAVED_AUDIO_BUDGET;

                // Run audio loop with this embedding.
                loop {
                    if pos >= model_config.max_seq_len || decoder.is_safety_limit_reached() {
                        break;
                    }
                    let outcome = decoder.decode_frame(&emb);
                    let audio_emb = match outcome {
                        FrameOutcome::End => {
                            if text_done {
                                break;
                            }
                            logits = model.forward(&[TOKEN_TEXT_END], pos, &mut state);
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
                        break;
                    }
                    if modality_budget == 0 && !text_done {
                        // Switch back to text. The reference transitions by
                        // decoding the last audio code embedding and sampling
                        // text from those logits (not by injecting TEXT_END).
                        logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
                        next_token = sampler.sample(&mut logits);
                        pos += 1;
                        break;
                    }

                    emb = model.forward_hidden_from_embedding(&audio_emb, pos, &mut state);
                    pos += 1;
                }

                if text_done {
                    break;
                }

                // Switch back to text.
                modality = Modality::Text;
                modality_budget = DEFAULT_INTERLEAVED_TEXT_BUDGET;
                continue;
            }

            // Normal text: forward and sample next token.
            logits = model.forward(&[next_token], pos, &mut state);
            next_token = sampler.sample(&mut logits);
            pos += 1;
        } else {
            // Sequential audio mode: embedding from the audio_start token.
            let mut emb = model.forward_embedding(&[next_token], pos, &mut state);
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
                    FrameOutcome::End => match config.mode {
                        AudioMode::Sequential => {
                            // Sequential TTS: audio is the final output.
                            // Returning to text + forwarding TEXT_END +
                            // resampling produces runaway garbage tokens (the
                            // model has nothing useful left to say) until
                            // max_tokens caps. Exit cleanly.
                            break 'outer;
                        }
                        AudioMode::Interleaved => {
                            // Interleaved: transition back to text. Trailing
                            // silence detection and safety frame limits bound
                            // runaway cycles.
                            modality = Modality::Text;
                            text_done = true;
                            modality_budget = DEFAULT_INTERLEAVED_TEXT_BUDGET;
                            logits = model.forward(&[TOKEN_TEXT_END], pos, &mut state);
                            next_token = sampler.sample(&mut logits);
                            pos += 1;
                            break;
                        }
                    },
                    FrameOutcome::Codes {
                        audio_embedding,
                        pcm,
                        ..
                    } => {
                        decoder.observe_pcm(&pcm);
                        if !pcm.is_empty() {
                            audio_callback(&pcm, decoder.sample_rate());
                        }
                        if (matches!(config.mode, AudioMode::Sequential) || text_done)
                            && decoder.is_silence_terminated()
                        {
                            break 'outer;
                        }
                        audio_embedding
                    }
                };
                modality_budget = modality_budget.saturating_sub(1);

                if generated >= config.max_tokens || pos >= model_config.max_seq_len {
                    break;
                }
                if matches!(config.mode, AudioMode::Interleaved)
                    && modality_budget == 0
                    && !text_done
                {
                    modality = Modality::Text;
                    modality_budget = DEFAULT_INTERLEAVED_TEXT_BUDGET;
                    logits = model.forward_from_embedding(&audio_emb, pos, &mut state);
                    next_token = sampler.sample(&mut logits);
                    pos += 1;
                    break;
                }

                // Feed codes back as embedding → next hidden state.
                emb = model.forward_hidden_from_embedding(&audio_emb, pos, &mut state);
                pos += 1;
            }
        }
    }

    // Drain any remaining buffered samples or execute batch ISTFT.
    let audio_samples = decoder.finish(&mut audio_callback);

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
mod tests {
    use super::*;

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
}

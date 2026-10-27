//! Speech to prefix embeddings.
//!
//! A clip of 16 kHz mono speech (up to 30 s; a shorter one than 0.5 s is padded) goes through
//! 128 log-mel features every 10 ms, a 17-layer FastConformer that subsamples by 8, an MLP
//! adapter from 512 to the trunk's width, and a residual correction:
//!
//! ```text
//! y = x + up(GELU(down(LayerNorm(x))))
//! ```
//!
//! One embedding comes out per 80 ms. The front end, the encoder and the adapter are the
//! FastConformer of [`crate::model::audio_encoder`] (the same lineage as LFM2-Audio, with the
//! same `a.*` tensors), so this module only adds the residual and the clip handling.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::engine::BackendPreference;
use crate::gguf::GgufFile;
use crate::model::audio_encoder::{
    AudioEncoderWeights, RESAMPLE_RATE_MAX, RESAMPLE_RATE_MIN, audio_encoder_forward,
    resample_linear,
};
use crate::model::audio_encoder_gpu::{AudioGpuEncode, build_gpu_audio_encoder};
use crate::model::audio_preprocessor::log_mel_spectrogram;
use crate::model::pii::matmul_nt_f32;

/// Sample rate the model reads.
pub const SAMPLE_RATE: u32 = 16_000;
/// Longest clip, in seconds; the rest is cut.
pub(crate) const MAX_SECONDS: usize = 30;
/// Shortest clip, in samples (0.5 s); a shorter one is padded with silence.
const MIN_SAMPLES: usize = 8_000;
/// LayerNorm epsilon of the residual block.
const EPS: f32 = 1e-5;
/// Samples between two mel frames (10 ms).
const HOP: usize = 160;

/// The mel frames the encoder reads for a clip of `samples`: `samples / 160`. The STFT makes one
/// more frame than that; the reference drops it, and feeding it to the encoder changes the last
/// rows and, for some lengths, the row count.
pub fn valid_frames(samples: usize) -> usize {
    samples / HOP
}

/// How many prefix embeddings a clip of `samples` (already prepared) becomes: the frames, halved
/// (rounding up) by each of the three stride-2 convolutions of the subsampling.
pub fn prefix_rows(samples: usize) -> usize {
    let mut rows = valid_frames(samples);
    for _ in 0..3 {
        rows = rows.div_ceil(2);
    }
    rows
}

/// A WAV file as mono samples in `[-1, 1]` and its sample rate.
///
/// Reads 16, 24 and 32-bit integer PCM and 32-bit float, any number of channels (averaged).
/// Unbounded: materializes the full clip, so prefer a capped decode for untrusted bytes.
///
/// # Errors
///
/// Fails on anything that is not a WAV of one of those encodings, a WAV that
/// holds no samples, or float samples that are not finite.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_wav_impl(bytes, None)
}

/// A WAV file as mono samples, keeping at most `max_seconds` of audio.
///
/// The pipeline decodes through this so a long clip never materializes past the
/// 30 s [`prepare_clip`] keeps; [`decode_wav`] is the uncapped twin.
///
/// # Errors
///
/// Fails when [`decode_wav`] fails, and on sample rates outside what
/// [`prepare_clip`] accepts (refused before any sample decodes, so the cap
/// cannot scale with a rate the pipeline rejects). A zero cap fails the no-samples guard.
pub(crate) fn decode_wav_capped(bytes: &[u8], max_seconds: usize) -> Result<(Vec<f32>, u32)> {
    decode_wav_impl(bytes, Some(max_seconds))
}

fn decode_wav_impl(bytes: &[u8], cap_seconds: Option<usize>) -> Result<(Vec<f32>, u32)> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV file"
    );
    let mut format: Option<(u16, u16, u32, u16)> = None; // tag, channels, rate, bits
    let mut data: Option<&[u8]> = None;
    let mut pos: usize = 12;
    while pos.saturating_add(8) <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8).saturating_add(size).min(bytes.len())];
        match id {
            b"fmt " => {
                ensure!(body.len() >= 16, "the WAV `fmt ` chunk is too short");
                let mut tag = u16::from_le_bytes(body[0..2].try_into()?);
                if tag == 0xFFFE && body.len() >= 26 {
                    // WAVE_FORMAT_EXTENSIBLE: the real tag opens the sub-format GUID
                    tag = u16::from_le_bytes(body[24..26].try_into()?);
                }
                format = Some((
                    tag,
                    u16::from_le_bytes(body[2..4].try_into()?),
                    u32::from_le_bytes(body[4..8].try_into()?),
                    u16::from_le_bytes(body[14..16].try_into()?),
                ));
            }
            b"data" => {
                ensure!(
                    body.len() >= size,
                    "the WAV `data` chunk declares {size} bytes but holds {}",
                    body.len()
                );
                data = Some(body);
            }
            _ => {}
        }
        pos = pos
            .saturating_add(8)
            .saturating_add(size)
            .saturating_add(size & 1);
    }
    let (tag, channels, rate, bits) = format.context("the WAV has no `fmt ` chunk")?;
    let data = data.context("the WAV has no `data` chunk")?;
    ensure!(
        channels >= 1 && rate > 0,
        "the WAV has no channels or no sample rate"
    );
    let sample = |chunk: &[u8]| -> f32 {
        match (tag, bits) {
            (1, 16) => f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 32768.0,
            (1, 24) => {
                let v = i32::from_le_bytes([0, chunk[0], chunk[1], chunk[2]]) >> 8;
                v as f32 / 8_388_608.0
            }
            (1, 32) => {
                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                    / 2_147_483_648.0
            }
            _ => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        }
    };
    if !matches!((tag, bits), (1, 16 | 24 | 32) | (3, 32)) {
        bail!(
            "unsupported WAV encoding (format {tag}, {bits} bits); use 16, 24 or 32-bit PCM or 32-bit float"
        );
    }
    let width = usize::from(bits / 8);
    let frame = width * usize::from(channels);
    // the cap counts frames, so a capped decode keeps the clip's head exactly as
    // the uncapped one plus `prepare_clip`'s cut would
    let take = match cap_seconds {
        Some(s) => {
            // the pipeline refuses this rate in `prepare_clip` after the samples
            // materialize; refuse here instead, where the rate is already parsed
            // and the take would otherwise scale with the refused rate
            check_sample_rate(rate)?;
            s.saturating_mul(rate as usize)
        }
        None => usize::MAX,
    };
    let samples: Vec<f32> = data
        .chunks_exact(frame)
        .take(take)
        .map(|f| f.chunks_exact(width).map(&sample).sum::<f32>() / f32::from(channels))
        .collect();
    ensure!(!samples.is_empty(), "the WAV `data` chunk holds no samples");
    // int PCM cannot be non-finite, but float bytes can; a NaN here is refused
    // here with a clear message instead of a confusing refusal at the final guard
    ensure!(
        samples.iter().all(|s| s.is_finite()),
        "the WAV `data` chunk holds non-finite samples"
    );
    Ok((samples, rate))
}

/// Refuse a sample rate outside what [`resample_linear`] accepts.
fn check_sample_rate(rate: u32) -> Result<()> {
    ensure!(
        (RESAMPLE_RATE_MIN..=RESAMPLE_RATE_MAX).contains(&rate),
        "unsupported sample rate {rate} Hz (want {RESAMPLE_RATE_MIN}..={RESAMPLE_RATE_MAX})"
    );
    Ok(())
}

/// The samples of a clip as the model reads them: resampled to 16 kHz when it is not already
/// (linearly, so another rate is not bit-exact with a high-quality resampler), cut to 30 s and
/// padded to 0.5 s. The input is cut to 30 s before resampling, so a low rate cannot amplify
/// a long clip past the cap.
///
/// # Errors
///
/// Fails when the clip holds no samples, when its samples are not finite, or when
/// the rate is outside what the resampler accepts: answering those as padded
/// silence would sound confident about nothing.
pub fn prepare_clip(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    check_sample_rate(rate)?;
    ensure!(!samples.is_empty(), "the clip holds no samples");
    ensure!(
        samples.iter().all(|s| s.is_finite()),
        "the clip holds non-finite samples"
    );
    let bounded = &samples[..samples.len().min(MAX_SECONDS * rate as usize)];
    let mut pcm = if rate == SAMPLE_RATE {
        bounded.to_vec()
    } else {
        resample_linear(bounded, rate, SAMPLE_RATE)
    };
    pcm.truncate(MAX_SECONDS * SAMPLE_RATE as usize);
    if pcm.len() < MIN_SAMPLES {
        pcm.resize(MIN_SAMPLES, 0.0);
    }
    Ok(pcm)
}

/// The speech encoder and its residual block.
pub struct AudioTower {
    encoder: Arc<AudioEncoderWeights>,
    /// The encoder on a GPU, when [`Self::accelerate`] found one.
    gpu: Option<Arc<dyn AudioGpuEncode>>,
    width: usize,
    hidden: usize,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    down_w: Vec<f32>,
    down_b: Vec<f32>,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
}

/// A PCM WAV for tests, shared by the decode tests here and the pipeline tests
/// in `super` (one writer, so the header layout is asserted once).
#[cfg(test)]
pub(crate) fn wav_bytes_for_test(
    tag: u16,
    channels: u16,
    rate: u32,
    bits: u16,
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    let block = channels * bits / 8;
    out.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// 16-bit PCM mono at 16 kHz for tests; legs that vary the format keep the
/// full [`wav_bytes_for_test`] instead of swapping the same-typed pair.
#[cfg(test)]
pub(crate) fn wav_pcm16_mono_16k_for_test(data: &[u8]) -> Vec<u8> {
    wav_bytes_for_test(1, 1, 16_000, 16, data)
}

/// Where a file keeps the residual block after the speech adapter.
#[derive(Debug, Clone, Copy)]
struct ResidualNames {
    norm_w: &'static str,
    norm_b: &'static str,
    down: &'static str,
    down_b: &'static str,
    up: &'static str,
    up_b: &'static str,
}

impl ResidualNames {
    /// A cera-converted file.
    const CONVERTED: Self = Self {
        norm_w: "d1.a.res.norm.weight",
        norm_b: "d1.a.res.norm.bias",
        down: "d1.a.res.down.weight",
        down_b: "d1.a.res.down.bias",
        up: "d1.a.res.up.weight",
        up_b: "d1.a.res.up.bias",
    };
    /// The public mmproj, where the residual continues the adapter's `mm.a.mlp` numbering.
    const PUBLIC: Self = Self {
        norm_w: "mm.a.mlp.4.weight",
        norm_b: "mm.a.mlp.4.bias",
        down: "mm.a.mlp.5.weight",
        down_b: "mm.a.mlp.5.bias",
        up: "mm.a.mlp.6.weight",
        up_b: "mm.a.mlp.6.bias",
    };
}

impl AudioTower {
    /// A tower over an all-zero 8-bin encoder (no conformer blocks) for tests: any
    /// clip encodes to zeros, so the pipeline legs answer evenly through it.
    /// Shared by the encode tests here and the pipeline tests in `super`.
    #[cfg(test)]
    pub(crate) fn silent_tower_for_test(trunk_width: usize) -> Self {
        use crate::model::audio_encoder::{
            AudioEncoderConfig, AudioEncoderWeights, AudioMlpAdapterWeights, ConvLayerWeights,
            ConvStemWeights,
        };
        use crate::model::weights::MmapWeight;
        // the stem's five layers in positional order (GGUF 0, 2, 3, 5, 6): three
        // stride-2 3x3 downsamplers and two 1x1 pointwise, as in the stem smoke
        // test; zero kernels and biases, so the stem output is zeros
        let out_ch = 4;
        let zero_kernel = |kw: usize, kh: usize, in_per_group: usize| ConvLayerWeights {
            name: "silent".into(),
            weight: vec![0.0f32; out_ch * in_per_group * kh * kw],
            bias: vec![0.0; out_ch],
            shape: vec![kw, kh, in_per_group, out_ch],
        };
        let identity_pointwise = || ConvLayerWeights {
            name: "silent".into(),
            weight: {
                let mut w = vec![0.0f32; out_ch * out_ch];
                for c in 0..out_ch {
                    w[c * out_ch + c] = 1.0;
                }
                w
            },
            bias: vec![0.0; out_ch],
            shape: vec![1, 1, out_ch, out_ch],
        };
        let plane = out_ch; // F_out is 1 for 8 mel bins after three halvings
        Self {
            encoder: Arc::new(AudioEncoderWeights {
                config: AudioEncoderConfig {
                    n_layer: 0,
                    n_embd: plane,
                    n_ff: 0,
                    n_head: 0,
                    eps: 1e-5,
                    n_mel_bins: 8,
                    llm_hidden_size: trunk_width,
                },
                conv_stem: ConvStemWeights {
                    layers: vec![
                        zero_kernel(3, 3, 1),
                        zero_kernel(3, 3, 1),
                        identity_pointwise(),
                        zero_kernel(3, 3, 1),
                        identity_pointwise(),
                    ],
                    pre_encode_out_w: MmapWeight::from_owned_f32(
                        vec![0.0; plane * plane],
                        plane,
                        plane,
                    ),
                    pre_encode_out_b: vec![0.0; plane],
                },
                layers: Vec::new(),
                mlp_adapter: AudioMlpAdapterWeights {
                    norm_w: vec![0.0; plane],
                    norm_b: vec![0.0; plane],
                    up_w: MmapWeight::from_owned_f32(vec![0.0; 8 * plane], 8, plane),
                    up_b: vec![0.0; 8],
                    down_w: MmapWeight::from_owned_f32(vec![0.0; trunk_width * 8], trunk_width, 8),
                    down_b: vec![0.0; trunk_width],
                },
            }),
            gpu: None,
            width: trunk_width,
            hidden: 8,
            norm_w: vec![0.0; trunk_width],
            norm_b: vec![0.0; trunk_width],
            down_w: vec![0.0; 8 * trunk_width],
            down_b: vec![0.0; 8],
            up_w: vec![0.0; trunk_width * 8],
            up_b: vec![0.0; trunk_width],
        }
    }

    /// Whether a GGUF carries a speech tower.
    pub fn is_present(gguf: &GgufFile) -> bool {
        gguf.get_u32("clip.audio.block_count").is_some()
            && gguf.get_u32("d1.audio.residual_width").is_some()
    }

    /// Read the encoder and the residual block.
    ///
    /// # Errors
    ///
    /// Fails when a setting or a tensor is missing or has the wrong size, a
    /// width is zero or mismatched, or when the adapter's width is not the
    /// trunk's.
    pub fn from_gguf(gguf: &Arc<GgufFile>, out_width: usize) -> Result<Self> {
        // with the cheap scalar reads, before the encoder: a zero residual width
        // would otherwise load and then answer through a degenerate adapter
        let hidden = super::setting_u32(gguf, "d1.audio.residual_width")?;
        Self::load(gguf, out_width, hidden, ResidualNames::CONVERTED)
    }

    /// Whether `mmproj` is the `clip` projector of a public d1-omni model with a speech tower.
    pub fn is_present_public(mmproj: &GgufFile) -> bool {
        mmproj.get_bool("clip.has_audio_encoder") == Some(true)
            && mmproj.get_str("clip.audio.projector_type") == Some("d1omni_a")
    }

    /// Read the speech tower from the mmproj of a public d1-omni model: the standard `clip`
    /// audio encoder and adapter, then the residual block the model adds after them
    /// (`mm.a.mlp.4` to `mm.a.mlp.6`).
    ///
    /// # Errors
    ///
    /// As [`Self::from_gguf`].
    pub fn from_public(mmproj: &Arc<GgufFile>, out_width: usize) -> Result<Self> {
        // the residual's inner width is the rows of its down projection (GGUF lists a matrix's
        // dimensions innermost first, so a linear layer is `[inputs, outputs]` and `shape[1]` is its
        // output width)
        let hidden = mmproj
            .tensors
            .get(ResidualNames::PUBLIC.down)
            .and_then(|t| t.shape.get(1).copied())
            .with_context(|| {
                format!(
                    "the speech tower needs the tensor `{}`",
                    ResidualNames::PUBLIC.down
                )
            })?;
        Self::load(mmproj, out_width, hidden, ResidualNames::PUBLIC)
    }

    fn load(
        gguf: &Arc<GgufFile>,
        out_width: usize,
        hidden: usize,
        names: ResidualNames,
    ) -> Result<Self> {
        // before the encoder: a zero residual width would otherwise load and then answer
        // through a degenerate adapter
        ensure!(hidden >= 1, "the speech tower has no residual width");
        let encoder = AudioEncoderWeights::from_gguf(gguf).context("the speech encoder")?;
        let width = encoder.config.llm_hidden_size;
        ensure!(
            width == out_width,
            "the speech adapter is {width} wide, the trunk {out_width}"
        );
        // element counts multiply file-declared widths, so they route
        // through the checked product: the plain multiply wraps on 32-bit
        let t = |name: &str, factors: &[usize]| {
            let elements = super::checked_len(name, factors)?;
            super::load_f32(gguf, "speech tower", name, elements)
        };
        Ok(Self {
            encoder: Arc::new(encoder),
            gpu: None,
            width,
            hidden,
            norm_w: t(names.norm_w, &[width])?,
            norm_b: t(names.norm_b, &[width])?,
            down_w: t(names.down, &[hidden, width])?,
            down_b: t(names.down_b, &[hidden])?,
            up_w: t(names.up, &[width, hidden])?,
            up_b: t(names.up_b, &[width])?,
        })
    }

    /// Run the encoder on the accelerator `backend` names, when there is one; the log-mel
    /// front end stays on the host so that exactly the reference's frames are read. Does
    /// nothing for the CPU or when no device opens.
    pub fn accelerate(&mut self, backend: BackendPreference) {
        // A backend that cannot build must not clear a working encoder: only `Some` sticks.
        if backend != BackendPreference::Cpu
            && let Some(gpu) = build_gpu_audio_encoder(&self.encoder, backend)
        {
            self.gpu = Some(gpu);
        }
    }

    /// Whether the encoder runs on an accelerator (a GPU or the NPU).
    pub fn is_accelerated(&self) -> bool {
        self.gpu.is_some()
    }

    /// `x + up(GELU(down(LayerNorm(x))))` over `rows` embeddings.
    fn residual(&self, x: &mut [f32], rows: usize) {
        let mut normed = x.to_vec();
        for row in normed.chunks_exact_mut(self.width) {
            cpu::layer_norm_inplace(row, &self.norm_w, &self.norm_b, EPS);
        }
        let mut mid = vec![0f32; rows * self.hidden];
        matmul_nt_f32(
            rows,
            self.hidden,
            self.width,
            &normed,
            &self.down_w,
            Some(&self.down_b),
            &mut mid,
        );
        cpu::gelu_erf_inplace(&mut mid);
        let mut out = vec![0f32; rows * self.width];
        matmul_nt_f32(
            rows,
            self.width,
            self.hidden,
            &mid,
            &self.up_w,
            Some(&self.up_b),
            &mut out,
        );
        for (a, b) in x.iter_mut().zip(&out) {
            *a += b;
        }
    }

    /// The prefix embeddings of a prepared clip ([`prepare_clip`]): `[rows * width]`.
    ///
    /// # Errors
    ///
    /// Fails if the encoder returns a different number of rows than [`prefix_rows`] says: the
    /// prompt's position budget and the reference's lengths both rest on that rule.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let bins = self.encoder.config.n_mel_bins;
        let (mel, stft_frames) = log_mel_spectrogram(pcm, bins);
        let frames = valid_frames(pcm.len()).min(stft_frames);
        if frames == 0 {
            // the predictor counts rows from the clip length, so no frames for a
            // counted clip is a disagreement, not an empty result (a 0-bin encoder
            // would otherwise answer speech questions from text alone)
            ensure!(
                prefix_rows(pcm.len()) == 0,
                "the speech encoder returned 0 rows for {} samples, expected {}",
                pcm.len(),
                prefix_rows(pcm.len())
            );
            return Ok(Vec::new());
        }
        let mel = &mel[..frames * bins];
        let on_gpu = self.gpu.as_ref().and_then(|gpu| {
            gpu.encode_mel(mel, frames)
                .map_err(|e| {
                    crate::d1::warn_cpu_fallback(format_args!(
                        "the speech encoder runs on the CPU: {e:#}"
                    ))
                })
                .ok()
        });
        let (mut embeddings, rows) =
            on_gpu.unwrap_or_else(|| audio_encoder_forward(mel, frames, &self.encoder));
        ensure!(
            rows == prefix_rows(pcm.len()),
            "the speech encoder returned {rows} rows for {} samples, expected {}",
            pcm.len(),
            prefix_rows(pcm.len())
        );
        self.residual(&mut embeddings, rows);
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_decodes_to_the_reference_scaling() {
        let data: Vec<u8> = [0i16, 16384, -32768, 32767]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (samples, rate) = decode_wav(&wav_pcm16_mono_16k_for_test(&data)).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(samples, [0.0, 0.5, -1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn stereo_is_averaged_and_float_and_24_bit_are_read() {
        let stereo: Vec<u8> = [1000i16, 3000, -2000, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (mono, _) = decode_wav(&wav_bytes_for_test(1, 2, 8_000, 16, &stereo)).unwrap();
        assert_eq!(mono, [2000.0 / 32768.0, -1000.0 / 32768.0]);
        let floats: Vec<u8> = [0.25f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (f, _) = decode_wav(&wav_bytes_for_test(3, 1, 16_000, 32, &floats)).unwrap();
        assert_eq!(f, [0.25, -0.5]);
        let v24: Vec<u8> = [0x40_0000i32, -0x40_0000]
            .iter()
            .flat_map(|v| v.to_le_bytes()[..3].to_vec())
            .collect();
        let (s24, _) = decode_wav(&wav_bytes_for_test(1, 1, 16_000, 24, &v24)).unwrap();
        assert_eq!(s24, [0.5, -0.5]);
    }

    #[test]
    fn chunks_other_than_data_are_skipped_and_odd_ones_padded() {
        let mut bytes = wav_pcm16_mono_16k_for_test(&[0, 0, 0, 64]);
        // a LIST chunk of odd size, with its pad byte, between `fmt ` and `data`
        let at = bytes.windows(4).position(|w| w == b"data").unwrap();
        let mut list = Vec::new();
        list.extend_from_slice(b"LIST");
        list.extend_from_slice(&3u32.to_le_bytes());
        list.extend_from_slice(&[1, 2, 3, 0]);
        bytes.splice(at..at, list);
        let (samples, _) = decode_wav(&bytes).unwrap();
        assert_eq!(samples, [0.0, 0.5]);
    }

    #[test]
    fn what_is_not_a_supported_wav_is_refused() {
        let mut no_data = wav_pcm16_mono_16k_for_test(&[]);
        no_data.truncate(36);
        // each leg pins its guard's message: `is_err` alone would still pass if
        // a deleted guard fell through to a later refusal (or panicked)
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (b"not a wav".to_vec(), "not a WAV file"),
            (
                wav_bytes_for_test(1, 1, 16_000, 8, &[0, 0]),
                "unsupported WAV encoding",
            ),
            (
                wav_bytes_for_test(2, 1, 16_000, 16, &[0, 0]),
                "unsupported WAV encoding",
            ),
            (wav_bytes_for_test(1, 0, 16_000, 16, &[0, 0]), "no channels"),
            (no_data, "has no `data` chunk"),
        ];
        for (bytes, want) in &cases {
            let err = decode_wav(bytes).unwrap_err();
            assert!(err.to_string().contains(want), "unexpected message: {err}");
        }
        // a present-but-empty data chunk answers silence with confidence unless refused
        let empty = wav_pcm16_mono_16k_for_test(&[]);
        let err = decode_wav(&empty).unwrap_err();
        assert!(
            err.to_string().contains("no samples"),
            "unexpected message: {err}"
        );
        // and a chunk cut short of its declared size is truncated, not partial audio
        let mut cut = wav_pcm16_mono_16k_for_test(&[0, 1, 2, 3]);
        cut.truncate(cut.len() - 2);
        let err = decode_wav(&cut).unwrap_err();
        assert!(
            err.to_string().contains("declares 4 bytes but holds 2"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn the_prefix_length_follows_the_valid_frames() {
        // lengths where the extra STFT frame would have added a row: a multiple of eight frames
        assert_eq!(prefix_rows(32_000), 25); // 200 frames
        assert_eq!(prefix_rows(480_000), 375); // 3000 frames
        assert_eq!(prefix_rows(8_000), 7); // 50 frames
        assert_eq!(prefix_rows(166_960), 131); // 1043 frames
        assert_eq!(prefix_rows(48_000), 38); // 300 frames
        assert_eq!(valid_frames(166_960), 1043);
        assert_eq!(prefix_rows(0), 0);
    }

    #[test]
    fn a_clip_is_padded_to_half_a_second_and_cut_to_thirty() {
        assert_eq!(
            prepare_clip(&[0.5; 100], 16_000).unwrap().len(),
            MIN_SAMPLES
        );
        assert_eq!(prepare_clip(&[0.5; 100], 16_000).unwrap()[0], 0.5);
        assert_eq!(MIN_SAMPLES, 8_000); // the policy value; the legs below pin the mechanics
        assert_eq!(prepare_clip(&[0.1; 20_000], 16_000).unwrap().len(), 20_000);
        assert_eq!(
            prepare_clip(&vec![0.1; 31 * 16_000], 16_000).unwrap().len(),
            30 * 16_000
        );
        // the edges themselves: exactly the pad floor stays unpadded, one below
        // pads; exactly the cut stays whole, one above cuts
        let floor = prepare_clip(&[0.5; MIN_SAMPLES], 16_000).unwrap();
        assert_eq!(floor.len(), MIN_SAMPLES);
        assert!(floor.iter().all(|&s| s == 0.5));
        assert_eq!(
            prepare_clip(&[0.5; MIN_SAMPLES - 1], 16_000).unwrap().len(),
            MIN_SAMPLES
        );
        assert_eq!(
            prepare_clip(&vec![0.1; 30 * 16_000], 16_000).unwrap().len(),
            30 * 16_000
        );
        assert_eq!(
            prepare_clip(&vec![0.1; 30 * 16_000 + 1], 16_000)
                .unwrap()
                .len(),
            30 * 16_000
        );
        // another rate is resampled to 16 kHz first
        assert_eq!(
            prepare_clip(&vec![0.1; 32_000], 32_000).unwrap().len(),
            16_000
        );
    }

    #[test]
    fn a_zero_residual_width_is_refused_before_the_encoder_loads() {
        // no encoder keys at all: the scalar check fires first
        let file = Arc::new(
            crate::gguf::GgufBuilder::new()
                .kv_u32("d1.audio.residual_width", 0)
                .build(),
        );
        // a match, not unwrap_err: the tower holds encoder weights without Debug
        let err = match AudioTower::from_gguf(&file, 64) {
            Ok(_) => panic!("expected the guard to refuse"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("no residual width"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn prepare_refuses_rates_the_resampler_cannot_read() {
        for rate in [0, 999, 192_001, u32::MAX] {
            let err = prepare_clip(&[0.1; 100], rate).unwrap_err();
            assert!(
                err.to_string().contains("unsupported sample rate"),
                "unexpected message: {err}"
            );
        }
        assert!(prepare_clip(&[0.1; 100], 1_000).is_ok());
        assert!(prepare_clip(&[0.1; 100], 192_000).is_ok());
        let err = prepare_clip(&[], 16_000).unwrap_err();
        assert!(
            err.to_string().contains("holds no samples"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn prepare_clip_refuses_non_finite_samples() {
        let err = prepare_clip(&[0.1, f32::NAN], 16_000).unwrap_err();
        assert!(
            err.to_string().contains("non-finite"),
            "unexpected message: {err}"
        );
    }

    // No test pins that the 30 s cut runs before resampling: both are prefix
    // operations, so the output is the same either way (up to one edge sample where
    // the resampler clamps). The ordering only bounds transient memory.

    /// A tower whose encoder reads 0 mel bins: the front end yields no frames for
    /// any clip, so the weights below are never touched (the early return fires first).
    fn zero_bin_tower() -> AudioTower {
        use crate::model::audio_encoder::{
            AudioEncoderConfig, AudioEncoderWeights, AudioMlpAdapterWeights, ConvStemWeights,
        };
        use crate::model::weights::MmapWeight;
        let noweight = || MmapWeight::from_owned_f32(Vec::new(), 0, 0);
        AudioTower {
            encoder: Arc::new(AudioEncoderWeights {
                config: AudioEncoderConfig {
                    n_layer: 0,
                    n_embd: 0,
                    n_ff: 0,
                    n_head: 0,
                    eps: 1e-5,
                    n_mel_bins: 0,
                    llm_hidden_size: 0,
                },
                conv_stem: ConvStemWeights {
                    layers: Vec::new(),
                    pre_encode_out_w: noweight(),
                    pre_encode_out_b: Vec::new(),
                },
                layers: Vec::new(),
                mlp_adapter: AudioMlpAdapterWeights {
                    norm_w: Vec::new(),
                    norm_b: Vec::new(),
                    up_w: noweight(),
                    up_b: Vec::new(),
                    down_w: noweight(),
                    down_b: Vec::new(),
                },
            }),
            gpu: None,
            width: 0,
            hidden: 0,
            norm_w: Vec::new(),
            norm_b: Vec::new(),
            down_w: Vec::new(),
            down_b: Vec::new(),
            up_w: Vec::new(),
            up_b: Vec::new(),
        }
    }

    #[test]
    fn no_frames_for_a_counted_clip_is_refused_not_empty() {
        // half a second counts 7 rows but yields no frames: answering that as an
        // empty prefix would score a speech question from text alone
        let err = zero_bin_tower()
            .encode(&vec![0.0; MIN_SAMPLES])
            .unwrap_err();
        assert!(
            err.to_string().contains("returned 0 rows"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn no_frames_for_an_uncounted_clip_is_still_empty() {
        assert_eq!(zero_bin_tower().encode(&[]).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn a_silent_tower_encodes_the_predicted_rows_of_zeros() {
        // half a second counts 7 rows, and the all-zero encoder agrees with zeros
        let pcm = vec![0.0; MIN_SAMPLES];
        let out = AudioTower::silent_tower_for_test(64).encode(&pcm).unwrap();
        assert_eq!(out.len(), prefix_rows(pcm.len()) * 64);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn a_capped_decode_keeps_the_clips_head() {
        // 31 s of mono 16-bit 16 kHz with varying samples, so head and tail differ
        let pcm: Vec<u8> = (0..31 * 16_000)
            .flat_map(|i| (i as i16).to_le_bytes())
            .collect();
        let bytes = wav_pcm16_mono_16k_for_test(&pcm);
        let (full, rate) = decode_wav(&bytes).unwrap();
        let (capped, capped_rate) = decode_wav_capped(&bytes, MAX_SECONDS).unwrap();
        assert_eq!(capped_rate, rate);
        assert_eq!(capped.len(), 30 * 16_000);
        assert_eq!(capped, full[..30 * 16_000]);
        // a short clip decodes whole, cap or not
        let short = wav_pcm16_mono_16k_for_test(&pcm[..16_000 * 2]);
        assert_eq!(
            decode_wav_capped(&short, MAX_SECONDS).unwrap().0,
            decode_wav(&short).unwrap().0
        );
        // ... and exactly the cap decodes whole: the cut is past the edge
        let exact = wav_pcm16_mono_16k_for_test(&pcm[..30 * 16_000 * 2]);
        let (capped, _) = decode_wav_capped(&exact, MAX_SECONDS).unwrap();
        assert_eq!(capped.len(), 30 * 16_000);
        assert_eq!(capped, decode_wav(&exact).unwrap().0);
    }

    #[test]
    fn a_capped_decode_refuses_a_rate_the_pipeline_rejects() {
        // just over the ceiling: the take would scale with the refused rate
        let bytes = wav_bytes_for_test(1, 1, 192_001, 16, &[0, 0]);
        let err = decode_wav_capped(&bytes, MAX_SECONDS).unwrap_err();
        assert!(
            err.to_string().contains("unsupported sample rate"),
            "unexpected message: {err}"
        );
        // ... and just under the floor, through the capped path too
        let bytes = wav_bytes_for_test(1, 1, 999, 16, &[0, 0]);
        let err = decode_wav_capped(&bytes, MAX_SECONDS).unwrap_err();
        assert!(
            err.to_string().contains("unsupported sample rate"),
            "unexpected message: {err}"
        );
        // rate 0 never reaches the range check: the header guard fires first
        let bytes = wav_bytes_for_test(1, 1, 0, 16, &[0, 0]);
        let err = decode_wav_capped(&bytes, MAX_SECONDS).unwrap_err();
        assert!(
            err.to_string().contains("no sample rate"),
            "unexpected message: {err}"
        );
        // the documented zero-cap failure: take(0) yields no samples
        let bytes = wav_pcm16_mono_16k_for_test(&[0, 0]);
        let err = decode_wav_capped(&bytes, 0).unwrap_err();
        assert!(
            err.to_string().contains("no samples"),
            "unexpected message: {err}"
        );
        // the uncapped twin keeps its contract: any rate over 0 decodes
        let bytes = wav_bytes_for_test(1, 1, 192_001, 16, &[0, 0]);
        assert!(decode_wav(&bytes).is_ok());
    }

    #[test]
    fn a_capped_decode_accepts_the_boundary_rates() {
        // the capped twin shares `prepare_clip`'s range, endpoints included: a
        // half-open mutant here would refuse clips the pipeline accepts
        for rate in [RESAMPLE_RATE_MIN, RESAMPLE_RATE_MAX] {
            let bytes = wav_bytes_for_test(1, 1, rate, 16, &[0, 0]);
            assert!(
                decode_wav_capped(&bytes, MAX_SECONDS).is_ok(),
                "rate {rate} refused"
            );
        }
    }

    #[test]
    fn float_samples_that_are_not_finite_are_refused() {
        // int PCM cannot be non-finite, but a float WAV can carry NaN or infinities
        // straight from its bytes; both twins must refuse them, not answer NaN
        for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bytes = wav_bytes_for_test(3, 1, 16_000, 32, &sample.to_le_bytes());
            for (name, result) in [
                ("decode_wav", decode_wav(&bytes)),
                ("decode_wav_capped", decode_wav_capped(&bytes, MAX_SECONDS)),
            ] {
                let err = result.unwrap_err();
                assert!(
                    err.to_string().contains("non-finite"),
                    "{name}: unexpected message: {err}"
                );
            }
        }
    }

    /// AudioTower acceleration honors BackendPreference::Cpu and keeps acceleration off.
    #[test]
    fn audio_tower_accelerate_handles_cpu_and_refusals() {
        use crate::engine::BackendPreference;
        use crate::model::audio_encoder::{
            AudioEncoderConfig, AudioEncoderWeights, AudioMlpAdapterWeights, ConvStemWeights,
        };
        use crate::model::weights::MmapWeight;

        let config = AudioEncoderConfig {
            n_layer: 0,
            n_embd: 64,
            n_ff: 128,
            n_head: 2,
            eps: 1e-5,
            n_mel_bins: 128,
            llm_hidden_size: 64,
        };
        let weights = AudioEncoderWeights {
            config,
            conv_stem: ConvStemWeights {
                layers: Vec::new(),
                pre_encode_out_w: MmapWeight::from_owned_f32(vec![0.0; 64 * 64], 64, 64),
                pre_encode_out_b: vec![0.0; 64],
            },
            layers: Vec::new(),
            mlp_adapter: AudioMlpAdapterWeights {
                norm_w: vec![0.0; 64],
                norm_b: vec![0.0; 64],
                up_w: MmapWeight::from_owned_f32(vec![0.0; 64 * 64], 64, 64),
                up_b: vec![0.0; 64],
                down_w: MmapWeight::from_owned_f32(vec![0.0; 64 * 64], 64, 64),
                down_b: vec![0.0; 64],
            },
        };
        let mut tower = AudioTower {
            encoder: Arc::new(weights),
            gpu: None,
            width: 64,
            hidden: 64,
            norm_w: vec![1.0; 64],
            norm_b: vec![0.0; 64],
            down_w: vec![0.0; 64 * 64],
            down_b: vec![0.0; 64],
            up_w: vec![0.0; 64 * 64],
            up_b: vec![0.0; 64],
        };
        assert!(!tower.is_accelerated());
        tower.accelerate(BackendPreference::Cpu);
        assert!(!tower.is_accelerated());
    }
}

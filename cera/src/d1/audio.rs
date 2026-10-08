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
use crate::model::audio_encoder::{AudioEncoderWeights, audio_encoder_forward, resample_linear};
use crate::model::audio_encoder_gpu::{AudioGpuEncode, build_gpu_audio_encoder};
use crate::model::audio_preprocessor::log_mel_spectrogram;
use crate::model::pii::matmul_nt_f32;

/// Sample rate the model reads.
pub const SAMPLE_RATE: u32 = 16_000;
/// Longest clip, in seconds; the rest is cut.
const MAX_SECONDS: usize = 30;
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
///
/// # Errors
///
/// Fails on anything that is not a WAV of one of those encodings.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV file"
    );
    let mut format: Option<(u16, u16, u32, u16)> = None; // tag, channels, rate, bits
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
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
            b"data" => data = Some(body),
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
    let samples = data
        .chunks_exact(frame)
        .map(|f| f.chunks_exact(width).map(&sample).sum::<f32>() / f32::from(channels))
        .collect();
    Ok((samples, rate))
}

/// The samples of a clip as the model reads them: resampled to 16 kHz when it is not already
/// (linearly, so another rate is not bit-exact with a high-quality resampler), cut to 30 s and
/// padded to 0.5 s.
pub fn prepare_clip(samples: &[f32], rate: u32) -> Vec<f32> {
    let mut pcm = if rate == SAMPLE_RATE {
        samples.to_vec()
    } else {
        resample_linear(samples, rate, SAMPLE_RATE)
    };
    pcm.truncate(MAX_SECONDS * SAMPLE_RATE as usize);
    if pcm.len() < MIN_SAMPLES {
        pcm.resize(MIN_SAMPLES, 0.0);
    }
    pcm
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

impl AudioTower {
    /// Whether a GGUF carries a speech tower.
    pub fn is_present(gguf: &GgufFile) -> bool {
        gguf.get_u32("clip.audio.block_count").is_some()
            && gguf.get_u32("d1.audio.residual_width").is_some()
    }

    /// Read the encoder and the residual block.
    ///
    /// # Errors
    ///
    /// Fails when a setting or a tensor is missing or has the wrong size, or when the adapter's
    /// width is not the trunk's.
    pub fn from_gguf(gguf: &Arc<GgufFile>, out_width: usize) -> Result<Self> {
        let encoder = AudioEncoderWeights::from_gguf(gguf).context("the speech encoder")?;
        let width = encoder.config.llm_hidden_size;
        ensure!(
            width == out_width,
            "the speech adapter is {width} wide, the trunk {out_width}"
        );
        let hidden = gguf
            .get_u32("d1.audio.residual_width")
            .context("missing d1.audio.residual_width")? as usize;
        let t = |name: &str, elements: usize| -> Result<Vec<f32>> {
            let v = gguf
                .get_tensor(name)
                .with_context(|| format!("the speech tower needs the tensor `{name}`"))?
                .to_f32_vec();
            ensure!(
                v.len() == elements,
                "`{name}` has {} values, expected {elements}",
                v.len()
            );
            Ok(v)
        };
        Ok(Self {
            encoder: Arc::new(encoder),
            gpu: None,
            width,
            hidden,
            norm_w: t("d1.a.res.norm.weight", width)?,
            norm_b: t("d1.a.res.norm.bias", width)?,
            down_w: t("d1.a.res.down.weight", hidden * width)?,
            down_b: t("d1.a.res.down.bias", hidden)?,
            up_w: t("d1.a.res.up.weight", width * hidden)?,
            up_b: t("d1.a.res.up.bias", width)?,
        })
    }

    /// Run the encoder on the GPU `backend` names, when there is one; the log-mel front end
    /// stays on the host so that exactly the reference's frames are read. Does nothing for the
    /// CPU or when no device opens.
    pub fn accelerate(&mut self, backend: BackendPreference) {
        if backend != BackendPreference::Cpu {
            self.gpu = build_gpu_audio_encoder(&self.encoder, backend);
        }
    }

    /// Whether the encoder runs on a GPU.
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
            return Ok(Vec::new());
        }
        let mel = &mel[..frames * bins];
        let on_gpu = self.gpu.as_ref().and_then(|gpu| {
            gpu.encode_mel(mel, frames)
                .map_err(|e| tracing::warn!("the speech encoder runs on the CPU: {e:#}"))
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

    fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
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

    #[test]
    fn pcm16_decodes_to_the_reference_scaling() {
        let data: Vec<u8> = [0i16, 16384, -32768, 32767]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (samples, rate) = decode_wav(&wav(1, 1, 16_000, 16, &data)).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(samples, [0.0, 0.5, -1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn stereo_is_averaged_and_float_and_24_bit_are_read() {
        let stereo: Vec<u8> = [1000i16, 3000, -2000, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (mono, _) = decode_wav(&wav(1, 2, 8_000, 16, &stereo)).unwrap();
        assert_eq!(mono, [2000.0 / 32768.0, -1000.0 / 32768.0]);
        let floats: Vec<u8> = [0.25f32, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (f, _) = decode_wav(&wav(3, 1, 16_000, 32, &floats)).unwrap();
        assert_eq!(f, [0.25, -0.5]);
        let v24: Vec<u8> = [0x40_0000i32, -0x40_0000]
            .iter()
            .flat_map(|v| v.to_le_bytes()[..3].to_vec())
            .collect();
        let (s24, _) = decode_wav(&wav(1, 1, 16_000, 24, &v24)).unwrap();
        assert_eq!(s24, [0.5, -0.5]);
    }

    #[test]
    fn chunks_other_than_data_are_skipped_and_odd_ones_padded() {
        let mut bytes = wav(1, 1, 16_000, 16, &[0, 0, 0, 64]);
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
        assert!(decode_wav(b"not a wav").is_err());
        assert!(decode_wav(&wav(1, 1, 16_000, 8, &[0, 0])).is_err());
        assert!(decode_wav(&wav(2, 1, 16_000, 16, &[0, 0])).is_err());
        assert!(decode_wav(&wav(1, 0, 16_000, 16, &[0, 0])).is_err());
        let mut no_data = wav(1, 1, 16_000, 16, &[]);
        no_data.truncate(36);
        assert!(decode_wav(&no_data).is_err());
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
        assert_eq!(prepare_clip(&[0.5; 100], 16_000).len(), MIN_SAMPLES);
        assert_eq!(prepare_clip(&[0.5; 100], 16_000)[0], 0.5);
        assert_eq!(prepare_clip(&[0.1; 20_000], 16_000).len(), 20_000);
        assert_eq!(
            prepare_clip(&vec![0.1; 31 * 16_000], 16_000).len(),
            30 * 16_000
        );
        // another rate is resampled to 16 kHz first
        assert_eq!(prepare_clip(&vec![0.1; 32_000], 32_000).len(), 16_000);
    }
}

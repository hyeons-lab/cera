//! Whisper audio preprocessor: PCM samples -> log-mel spectrogram.
//!
//! Implements OpenAI Whisper log-mel feature extraction:
//! 1. Audio pad/trim to 30.00 seconds (480,000 samples at 16,000 Hz).
//! 2. Center padding by N_FFT / 2 = 200 samples.
//! 3. Periodic Hann window (400 samples) + 400-point real FFT.
//! 4. Power spectrum squared magnitude (|Re|^2 + |Im|^2).
//! 5. Slaney-scale triangular mel filterbank (80 or 128 bins).
//! 6. Log10 scaling with 1e-10 floor.
//! 7. Dynamic range clamping to [max - 8.0, max] and scaling via (x + 4.0) / 4.0 (nominal range [-1.0, 1.0] when max is near 0 dB).
//!
//! Output is [n_mels x 3000] channel-major (or available as [3000 x n_mels] time-major),
//! matching Whisper's 1D conv subsampling stem expectations.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::model::audio_preprocessor::{build_hann_window, build_mel_filterbank};

/// Sampling rate expected by Whisper (16 kHz).
pub const SAMPLE_RATE: usize = 16_000;
/// FFT window length in samples (25 ms at 16 kHz).
pub const N_FFT: usize = 400;
/// Hop size between consecutive frames in samples (10 ms at 16 kHz).
pub const HOP_LEN: usize = 160;
/// Number of samples in a standard 30-second Whisper chunk.
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE; // 480,000
/// Expected number of spectrogram frames in a 30-second chunk.
pub const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP_LEN; // 3,000
/// Number of unique FFT bins for a 400-point real FFT.
pub const N_FFT_BINS: usize = N_FFT / 2 + 1; // 201

/// Compute the log-mel spectrogram for 16 kHz mono audio samples.
///
/// If `pcm` length is not exactly 480,000 samples (30 seconds), it will be
/// padded with zeros or trimmed to 480,000 samples.
///
/// Output layout: `[n_mels x 3000]` row-major (channel-major outer, time inner).
/// Element `[m, t]` is at index `m * CHUNK_FRAMES + t`.
pub fn extract_whisper_mel(pcm: &[f32], n_mels: usize) -> Vec<f32> {
    if n_mels != 80 && n_mels != 128 {
        return Vec::new();
    }

    // Center padding by N_FFT / 2 = 200 samples on each side directly into CHUNK_SAMPLES buffer
    let pad = N_FFT / 2;
    let mut padded_audio = vec![0.0f32; CHUNK_SAMPLES + 2 * pad];
    let copy_len = pcm.len().min(CHUNK_SAMPLES);
    if copy_len > 0 {
        for (dst, &src) in padded_audio[pad..pad + copy_len]
            .iter_mut()
            .zip(&pcm[..copy_len])
        {
            *dst = if src.is_finite() { src } else { 0.0 };
        }
        // PyTorch reflect padding on left flank (mirrors without duplicating audio[0])
        for i in 0..pad {
            if i + 1 < copy_len {
                let v = pcm[i + 1];
                padded_audio[pad - 1 - i] = if v.is_finite() { v } else { 0.0 };
            }
        }
        // Right flank reflection around the 30s (CHUNK_SAMPLES) boundary
        if copy_len == CHUNK_SAMPLES {
            for i in 0..pad {
                let src_idx = CHUNK_SAMPLES.saturating_sub(2 + i);
                let v = pcm[src_idx];
                padded_audio[pad + CHUNK_SAMPLES + i] = if v.is_finite() { v } else { 0.0 };
            }
        }
        // When copy_len < CHUNK_SAMPLES, samples at the 30s chunk boundary are 0.0,
        // so right-flank padding correctly remains 0.0.
    }

    let hann = build_hann_window(N_FFT);
    let mel_filters = build_mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    thread_local! {
        static FFT_PLANNER: std::cell::RefCell<FftPlanner<f32>> =
            std::cell::RefCell::new(FftPlanner::new());
    }
    let fft = FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(N_FFT));
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); N_FFT];

    // Temporary storage for power spectrogram [CHUNK_FRAMES x N_FFT_BINS]
    let mut power_spec = vec![0.0f32; CHUNK_FRAMES * N_FFT_BINS];

    for frame in 0..CHUNK_FRAMES {
        let start = frame * HOP_LEN;
        for i in 0..N_FFT {
            fft_buf[i] = Complex32::new(padded_audio[start + i] * hann[i], 0.0);
        }

        fft.process(&mut fft_buf);

        let row = &mut power_spec[frame * N_FFT_BINS..(frame + 1) * N_FFT_BINS];
        for (k, slot) in row.iter_mut().enumerate().take(N_FFT_BINS) {
            *slot = fft_buf[k].norm_sqr();
        }
    }

    // Multiply with mel filterbank and apply log10:
    // mel_spec[m, t] = sum_k filter[m, k] * power[t, k]
    let mut log_mel = vec![0.0f32; n_mels * CHUNK_FRAMES];
    let mut max_val = f32::NEG_INFINITY;

    for m in 0..n_mels {
        let filter_row = &mel_filters[m * N_FFT_BINS..(m + 1) * N_FFT_BINS];
        let k_first = filter_row.iter().position(|&v| v > 0.0).unwrap_or(0);
        let k_last = filter_row
            .iter()
            .rposition(|&v| v > 0.0)
            .map_or(0, |i| i + 1);
        let active_filter = &filter_row[k_first..k_last];

        for t in 0..CHUNK_FRAMES {
            let power_row = &power_spec[t * N_FFT_BINS..(t + 1) * N_FFT_BINS];
            let sum = if !active_filter.is_empty() {
                crate::backend::cpu::dot_f32(active_filter, &power_row[k_first..k_last])
            } else {
                0.0
            };
            // clamp(min = 1e-10) and log10
            let val = (sum.max(1e-10)).log10();
            if val > max_val {
                max_val = val;
            }
            log_mel[m * CHUNK_FRAMES + t] = val;
        }
    }

    // Dynamic range shift and scaling:
    // log_spec = max(log_spec, max_val - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    let min_allowed = max_val - 8.0;
    for slot in log_mel.iter_mut() {
        let clamped = slot.max(min_allowed);
        *slot = (clamped + 4.0) / 4.0;
    }

    log_mel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_whisper_mel_silence() {
        let silence = vec![0.0f32; 16000]; // 1 second of silence
        let mel = extract_whisper_mel(&silence, 80);

        assert_eq!(mel.len(), 80 * CHUNK_FRAMES);
        for (idx, &v) in mel.iter().enumerate() {
            assert!(v.is_finite(), "mel[{idx}] is not finite: {v}");
            assert!(
                (-2.0..=2.0).contains(&v),
                "mel[{idx}] out of expected range: {v}"
            );
        }
    }

    #[test]
    fn test_extract_whisper_mel_sine_wave() {
        // 1 second of 440 Hz sine wave
        let mut pcm = Vec::with_capacity(16000);
        for i in 0..16000 {
            let t = i as f32 / 16000.0;
            pcm.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        let mel_80 = extract_whisper_mel(&pcm, 80);
        assert_eq!(mel_80.len(), 80 * 3000);

        let mel_128 = extract_whisper_mel(&pcm, 128);
        assert_eq!(mel_128.len(), 128 * 3000);

        // Max value is finite and dynamic range is exactly bounded by 8.0 / 4.0 = 2.0
        let max_val_80 = mel_80.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val_80 = mel_80.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_val_80.is_finite());
        assert!(min_val_80.is_finite());
        assert!(
            (max_val_80 - min_val_80) <= 2.0001,
            "range exceeds 2.0: {} to {}",
            min_val_80,
            max_val_80
        );
        assert!(
            (max_val_80 - min_val_80) >= 1.9999,
            "range less than 2.0: {} to {}",
            min_val_80,
            max_val_80
        );
    }

    #[test]
    fn test_extract_whisper_mel_invalid_bins() {
        let pcm = vec![0.0f32; 16000];
        let mel = extract_whisper_mel(&pcm, 64);
        assert!(mel.is_empty(), "invalid mel bins should yield empty output");
    }

    #[test]
    fn test_extract_whisper_mel_non_finite_pcm() {
        let pcm = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];
        let mel = extract_whisper_mel(&pcm, 80);
        assert_eq!(mel.len(), 80 * CHUNK_FRAMES);
        for &val in &mel {
            assert!(val.is_finite(), "mel value must be finite: {val}");
        }
    }

    #[test]
    fn test_extract_whisper_mel_single_sample() {
        let pcm = vec![0.5f32];
        let mel = extract_whisper_mel(&pcm, 80);
        assert_eq!(mel.len(), 80 * CHUNK_FRAMES);
        for &val in &mel {
            assert!(
                val.is_finite(),
                "mel value for single-sample input must be finite: {val}"
            );
        }
    }
}

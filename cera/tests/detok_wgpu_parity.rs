#![cfg(feature = "gpu")]

//! GPU↔CPU parity tests for the WGPU detokenizer.
//!
//! Mirrors `detok_metal_parity.rs` for the WGPU backend. Both paths dequantize
//! the vocoder's Q4_0 weights to f32; unlike Metal, the WGPU `flash_attention`
//! kernel reads an f32 KV cache (no f16 cast), so the GPU attention is a touch
//! more accurate than Metal's. The log-magnitude `max_diff` gate is the primary
//! correctness check; the per-frame cosine gate stays at 0.98 to leave headroom
//! for the reg-tile-GEMM-vs-CPU-dot rounding on low-energy frames.
//!
//! Gating: needs a GPU adapter (wgpu → Metal/Vulkan/DX) and the vocoder GGUF in
//! `~/.leap/models`; skips cleanly if either is absent.

use cera::backend::wgpu::GpuContext;

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    // Two zero vectors agree exactly, so the gates below should read that as a
    // match rather than as the worst possible score. Only a one-sided zero is
    // real disagreement. Callers pair this with a non-silence assertion on the
    // CPU reference, since "both sides emitted nothing" is a detokenizer bug
    // that agreement alone cannot see.
    match (na == 0.0, nb == 0.0) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.0,
        (false, false) => dot / (na * nb),
    }
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn band_energy(log_abs: &[f32], start: usize, end: usize) -> f64 {
    log_abs[start..end].iter().map(|&v| (v as f64).exp()).sum()
}

fn load_vocoder() -> Option<(std::sync::Arc<cera::gguf::GgufFile>, std::path::PathBuf)> {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("vocoder not found at {}, skipping", path.display());
        return None;
    }
    let gguf = cera::gguf::GgufFile::open_arc(&path).unwrap();
    Some((gguf, path))
}

/// Skip cleanly when no adapter is available (native-only blocking init).
fn gpu_available() -> bool {
    match GpuContext::new() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("skipping: no wgpu adapter ({e})");
            false
        }
    }
}

const N_FFT_BINS: usize = 641;
const SPECTRUM_PER_FRAME: usize = N_FFT_BINS * 2; // 1282

#[test]
fn spectrum_parity() {
    if !gpu_available() {
        return;
    }
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_gguf(&gguf, &path).unwrap();

    let code_sets: &[&[i32]] = &[
        &[100, 200, 300, 400, 500, 600, 700, 800],
        &[1, 1000, 2000, 500, 1500, 750, 1250, 2047],
    ];

    for (ci, codes) in code_sets.iter().enumerate() {
        let mut cpu_state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);
        gpu.reset();

        let cpu_spec = cera::model::audio_decoder::detokenize_to_spectrum(
            &detok_w,
            &dec_w,
            &mut cpu_state,
            codes,
        );
        let gpu_spec = gpu.detokenize_to_spectrum(&detok_w, codes);

        assert_eq!(
            cpu_spec.len(),
            gpu_spec.len(),
            "code set {ci}: length mismatch"
        );
        assert_eq!(
            cpu_spec.len(),
            6 * SPECTRUM_PER_FRAME,
            "code set {ci}: expected 6×1282"
        );

        for f in 0..6 {
            let off = f * SPECTRUM_PER_FRAME;
            let cpu_frame = &cpu_spec[off..off + SPECTRUM_PER_FRAME];
            let gpu_frame = &gpu_spec[off..off + SPECTRUM_PER_FRAME];

            assert!(
                cpu_frame.iter().all(|v| v.is_finite()),
                "cs {ci} f {f}: CPU NaN/Inf"
            );
            assert!(
                gpu_frame.iter().all(|v| v.is_finite()),
                "cs {ci} f {f}: GPU NaN/Inf"
            );
            // The reference has to carry signal for the two gates below to mean
            // anything: a pair of all-zero frames scores a perfect cosine and a
            // zero max_diff, which is the same green as a real match.
            assert!(
                cpu_frame.iter().any(|&v| v != 0.0),
                "cs {ci} f {f}: CPU reference frame is all zeros"
            );

            let cos = cosine_sim(cpu_frame, gpu_frame);
            let max_diff = max_abs_diff(&cpu_frame[..N_FFT_BINS], &gpu_frame[..N_FFT_BINS]);
            eprintln!("  code_set={ci} frame={f}: cos={cos:.6} max_diff={max_diff:.4}");
            assert!(
                cos > 0.98,
                "code set {ci}, frame {f}: cosine {cos:.6} < 0.98"
            );
            assert!(
                max_diff < 0.5,
                "code set {ci}, frame {f}: max_diff {max_diff:.4} >= 0.5"
            );
        }
    }
    eprintln!("spectrum_parity: PASSED");
}

#[test]
fn pcm_parity() {
    if !gpu_available() {
        return;
    }
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_gguf(&gguf, &path).unwrap();

    let codes: &[i32] = &[100, 200, 300, 400, 500, 600, 700, 800];

    let mut cpu_state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    gpu.reset();

    let cpu_spec =
        cera::model::audio_decoder::detokenize_to_spectrum(&detok_w, &dec_w, &mut cpu_state, codes);
    let gpu_spec = gpu.detokenize_to_spectrum(&detok_w, codes);

    let cpu_pcm = cera::model::audio_decoder::istft_to_pcm(
        &cpu_spec,
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );
    let gpu_pcm = cera::model::audio_decoder::istft_to_pcm(
        &gpu_spec,
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );

    let cpu_rms = rms(&cpu_pcm);
    let gpu_rms = rms(&gpu_pcm);
    assert!(cpu_rms > 0.0, "CPU PCM is silent");
    assert!(gpu_rms > 0.0, "GPU PCM is silent");
    assert!(cpu_pcm.iter().all(|v| v.is_finite()), "CPU PCM NaN/Inf");
    assert!(gpu_pcm.iter().all(|v| v.is_finite()), "GPU PCM NaN/Inf");

    let rms_ratio = gpu_rms / cpu_rms;
    eprintln!("  cpu_rms={cpu_rms:.2} gpu_rms={gpu_rms:.2} ratio={rms_ratio:.4}");
    assert!(
        (0.8..=1.25).contains(&rms_ratio),
        "PCM RMS ratio {rms_ratio:.4} outside [0.8, 1.25]"
    );

    let bands = [(0, 160), (160, 320), (320, 480), (480, N_FFT_BINS)];
    let cpu_log_abs = &cpu_spec[..N_FFT_BINS];
    let gpu_log_abs = &gpu_spec[..N_FFT_BINS];
    for &(start, end) in &bands {
        let cpu_e = band_energy(cpu_log_abs, start, end);
        let gpu_e = band_energy(gpu_log_abs, start, end);
        if cpu_e > 1e-10 {
            let ratio = gpu_e / cpu_e;
            eprintln!("  band [{start}-{end}]: cpu={cpu_e:.2} gpu={gpu_e:.2} ratio={ratio:.4}");
            assert!(
                (0.5..=2.0).contains(&ratio),
                "Band [{start}-{end}] energy ratio {ratio:.4} outside [0.5, 2.0]"
            );
        }
    }
    eprintln!("pcm_parity: PASSED");
}

/// The GPU ISTFT (`exp_polar` → iDFT-matmul → `overlap_add`) must reproduce the
/// CPU `istft_to_pcm` on the *same* spectrum, isolating the ISTFT from any
/// spectrum divergence. The iDFT-as-matmul differs from rustfft's radix only by
/// float rounding, so the bar is tight.
#[test]
fn gpu_istft_matches_cpu() {
    if !gpu_available() {
        return;
    }
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_gguf(&gguf, &path).unwrap();

    // Accumulate several frames (like the engine's `all_spectrum`) so overlap-add
    // runs well past the n_fft/hop overlap depth and through the tail.
    let mut cpu_state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    let mut spectrum = Vec::new();
    for f in 0..4 {
        let base = f * 111 + 20;
        let codes: Vec<i32> = (0..8).map(|c| base + c * 90).collect();
        let s = cera::model::audio_decoder::detokenize_to_spectrum(
            &detok_w,
            &dec_w,
            &mut cpu_state,
            &codes,
        );
        spectrum.extend_from_slice(&s);
    }

    let n_fft = detok_w.config.n_fft;
    let hop = detok_w.config.hop_length;
    let cpu_pcm = cera::model::audio_decoder::istft_to_pcm(&spectrum, n_fft, hop);
    let gpu_pcm = gpu.istft_to_pcm(&spectrum, n_fft, hop);

    assert_eq!(cpu_pcm.len(), gpu_pcm.len(), "PCM length mismatch");
    assert!(gpu_pcm.iter().all(|v| v.is_finite()), "GPU PCM NaN/Inf");

    let cos = cosine_sim(&cpu_pcm, &gpu_pcm);
    let cpu_rms = rms(&cpu_pcm);
    let worst = max_abs_diff(&cpu_pcm, &gpu_pcm);
    eprintln!(
        "  gpu_istft: cos={cos:.6} cpu_rms={cpu_rms:.4} worst_abs_diff={worst:.4} rel={:.4e}",
        worst as f64 / cpu_rms.max(1e-9)
    );
    assert!(cos > 0.999, "GPU ISTFT cosine {cos:.6} < 0.999");
    assert!(
        (worst as f64) < 0.05 * cpu_rms.max(1e-6),
        "GPU ISTFT worst abs diff {worst:.4} > 5% of CPU RMS {cpu_rms:.4}"
    );
    eprintln!("gpu_istft_matches_cpu: PASSED");
}

#[test]
fn multi_frame_stability() {
    if !gpu_available() {
        return;
    }
    let Some((gguf, path)) = load_vocoder() else {
        return;
    };

    let detok_w = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();
    let dec_w = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let gpu = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_gguf(&gguf, &path).unwrap();

    let mut cpu_state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    gpu.reset();

    // 8 frames = 48 tokens → wraps the SWA window (30) 1.6 times.
    let all_codes: Vec<[i32; 8]> = (0..8)
        .map(|i| {
            let base = i * 100 + 50;
            [
                base,
                base + 100,
                base + 200,
                base + 300,
                base + 400,
                base + 500,
                base + 600,
                base + 700,
            ]
        })
        .collect();

    let mut first_rms_cpu = 0.0f64;
    let mut first_rms_gpu = 0.0f64;

    for (fi, codes) in all_codes.iter().enumerate() {
        let cpu_spec = cera::model::audio_decoder::detokenize_to_spectrum(
            &detok_w,
            &dec_w,
            &mut cpu_state,
            codes,
        );
        let gpu_spec = gpu.detokenize_to_spectrum(&detok_w, codes);

        let cpu_f0 = &cpu_spec[..SPECTRUM_PER_FRAME];
        let gpu_f0 = &gpu_spec[..SPECTRUM_PER_FRAME];
        assert!(
            cpu_f0.iter().all(|v| v.is_finite()),
            "frame {fi}: CPU NaN/Inf"
        );
        assert!(
            gpu_f0.iter().all(|v| v.is_finite()),
            "frame {fi}: GPU NaN/Inf"
        );

        let cos = cosine_sim(cpu_f0, gpu_f0);
        let cpu_r = rms(cpu_f0);
        let gpu_r = rms(gpu_f0);
        // Same reason as `spectrum_parity`: a silent reference makes the cosine
        // gate vacuous rather than failing it.
        assert!(cpu_r > 0.0, "frame {fi}: CPU reference frame is all zeros");
        if fi == 0 {
            first_rms_cpu = cpu_r;
            first_rms_gpu = gpu_r;
        }
        eprintln!("  frame {fi}: cos={cos:.6} cpu_rms={cpu_r:.4} gpu_rms={gpu_r:.4}");

        let threshold = if fi < 3 { 0.99 } else { 0.95 };
        assert!(cos > threshold, "frame {fi}: cosine {cos:.6} < {threshold}");
        if first_rms_cpu > 1e-10 {
            assert!(
                cpu_r / first_rms_cpu < 100.0,
                "frame {fi}: CPU RMS exploded"
            );
        }
        if first_rms_gpu > 1e-10 {
            assert!(
                gpu_r / first_rms_gpu < 100.0,
                "frame {fi}: GPU RMS exploded"
            );
        }
    }
    eprintln!("multi_frame_stability: PASSED");
}

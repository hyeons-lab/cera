#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

//! GPU↔CPU parity for the wgpu LFM2A Conformer audio encoder.
//!
//! The wgpu ops are the WGSL halves of the Slang kernels the Metal encoder dispatches
//! (`tests/audio_encoder_metal_parity.rs` pins those), plus the ViT's wgpu GEMM, LayerNorm and
//! elementwise kernels. This pins the pieces that are specific to wgpu:
//!
//! 1. **A convolution too long for one dispatch.** A dispatch is at most 65535 workgroups
//!    (16.7M elements); the first stem convolution of a clip past about 20 s has more outputs
//!    than that, so `conv2d` splits it over output channels. Checked against a direct reference,
//!    dense and depthwise.
//! 2. **The whole encoder** on a real model: `encode_audio_mel_gpu` against
//!    `audio_encoder_forward`, for a clip short enough for one dispatch and one long enough to
//!    split, and `encode_audio_pcm_gpu` (the on-GPU log-mel front end) end to end.
//!
//! The model is a d1-omni GGUF (`CERA_AUDIO_GGUF`, else `~/.leap/models/d1-omni-public/d1-f16.gguf`)
//! or the LFM2.5-Audio mmproj. A GGUF whose linears are F16 gives the CPU an exact f32 reference
//! and a tight gate; a quantized one makes the CPU the lossy side (it quantizes activations to
//! Q8_0 for a quantized weight), so the gate is loose. Skips when no model or no adapter exists;
//! `CERA_REQUIRE_GPU=1` turns the missing adapter into a failure.

use cera::backend::wgpu::GpuContext;
use cera::model::audio_encoder::{AudioEncoderWeights, audio_encoder_forward};
use cera::model::audio_encoder_gpu::{
    AudioEncoderGpuOps, Conv2dSpec, GpuAudioWeights, WgpuAudioOps, encode_audio_mel_gpu,
    encode_audio_pcm_gpu,
};
use cera::model::audio_preprocessor::log_mel_spectrogram;

mod common;
use common::{noise, test_pcm};

fn gpu_ops() -> Option<WgpuAudioOps> {
    match GpuContext::new() {
        Ok(ctx) => Some(WgpuAudioOps::new(ctx).expect("build wgpu audio ops")),
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but no GPU adapter is available: {e}"
            );
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

fn rel_l2(want: &[f32], got: &[f32]) -> f64 {
    assert_eq!(want.len(), got.len(), "length");
    let num: f64 = want
        .iter()
        .zip(got)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let den: f64 = want.iter().map(|a| f64::from(*a).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// `[out_ch][h_out][w_out]` by the definition of the convolution.
fn conv2d_reference(input: &[f32], weight: &[f32], bias: &[f32], s: &Conv2dSpec) -> Vec<f32> {
    let per_group_in = s.in_ch / s.groups;
    let per_group_out = s.out_ch / s.groups;
    let mut out = vec![0f32; s.out_len()];
    for oc in 0..s.out_ch {
        let g = oc / per_group_out;
        for oy in 0..s.h_out {
            for ox in 0..s.w_out {
                let mut acc = bias[oc];
                for ic in 0..per_group_in {
                    for ky in 0..s.kh {
                        for kx in 0..s.kw {
                            let y = (oy * s.stride_h + ky) as isize - s.pad_h as isize;
                            let x = (ox * s.stride_w + kx) as isize - s.pad_w as isize;
                            if y < 0 || x < 0 || y >= s.h_in as isize || x >= s.w_in as isize {
                                continue;
                            }
                            let c = g * per_group_in + ic;
                            acc += input[(c * s.h_in + y as usize) * s.w_in + x as usize]
                                * weight[((oc * per_group_in + ic) * s.kh + ky) * s.kw + kx];
                        }
                    }
                }
                out[(oc * s.h_out + oy) * s.w_out + ox] = acc;
            }
        }
    }
    out
}

#[test]
fn a_convolution_over_the_dispatch_limit_is_split_over_channels() {
    let Some(ops) = gpu_ops() else {
        return;
    };
    // 96 channels of 2 x 90_000 outputs: 17.3M, over the 16.7M one dispatch covers, so it runs as
    // a first chunk of 64 channels and a second of 32. (A 3 x 3 kernel, stride 1, same padding.)
    for (in_ch, groups) in [(1usize, 1usize), (96, 96)] {
        let spec = Conv2dSpec::padded(in_ch, 96, 2, 90_000, (3, 3), (1, 1), (1, 1), (1, 1), groups)
            .expect("a runnable convolution");
        assert!(
            spec.out_len() > cera::backend::wgpu::MAX_WG as usize * 256,
            "the case must need a split"
        );
        let input = noise(in_ch * spec.h_in * spec.w_in, 1, 1.0);
        let weight = noise(96 * (in_ch / groups) * 9, 2, 0.5);
        let bias = noise(96, 3, 0.5);
        let want = conv2d_reference(&input, &weight, &bias, &spec);
        let got = ops.conv2d(
            &ops.upload(&input),
            &ops.upload(&weight),
            &ops.upload(&bias),
            &spec,
        );
        let got = ops.download(&got, spec.out_len());
        let err = rel_l2(&want, &got);
        assert!(err < 1e-5, "groups={groups}: rel-L2 {err}");
    }
}

/// A gate-admitted clip whose first stem convolution no dispatch can bind is refused with
/// an error, not an abort: 8191 frames of 128 bins stay under `MAX_AUDIO_TOKENS` post-stem
/// but need a 262144-wide plane per channel, past what the channel splitter binds.
#[test]
fn an_unsplittable_stem_is_refused_not_aborted() {
    let Some((weights, _)) = load_encoder() else {
        return;
    };
    let Some(ops) = gpu_ops() else {
        return;
    };
    let bins = weights.config.n_mel_bins;
    if bins != 128 {
        eprintln!("skipping: this model has {bins} mel bins, not the 128 the case needs");
        return;
    }
    let gpu_w = GpuAudioWeights::build(&ops, &weights).expect("upload the audio encoder");
    let n_frames = 8191;
    let mel = vec![0.0f32; n_frames * bins];
    let err = encode_audio_mel_gpu(&ops, &gpu_w, &mel, n_frames).expect_err(
        "an 8191-frame clip must be refused: its first stem convolution cannot be split",
    );
    assert!(
        err.to_string().contains("channel splitter"),
        "unexpected refusal: {err}"
    );
}

/// The encoder, and whether the CPU reference is exact (F16 or F32 linears).
fn load_encoder() -> Option<(AudioEncoderWeights, bool)> {
    let home = std::path::PathBuf::from(std::env::var("HOME").ok()?);
    let candidates = [
        (
            Some(home.join(".leap/models/d1-omni-public/d1-f16.gguf")),
            true,
        ),
        (
            Some(
                home.join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"),
            ),
            false,
        ),
    ];
    // an explicit model is set-but-invalid when it cannot load: fail loud rather than
    // silently skipping, which would report green while testing nothing
    if let Ok(explicit) = std::env::var("CERA_AUDIO_GGUF") {
        let path = std::path::PathBuf::from(&explicit);
        assert!(path.exists(), "CERA_AUDIO_GGUF={explicit} does not exist");
        let gguf = cera::gguf::GgufFile::open_arc(&path).expect("open the audio model");
        let weights =
            AudioEncoderWeights::from_gguf(&gguf).expect("CERA_AUDIO_GGUF holds no audio encoder");
        return Some((weights, true));
    }
    for (path, exact) in candidates {
        let Some(path) = path else { continue };
        if !path.exists() {
            continue;
        }
        let gguf = cera::gguf::GgufFile::open_arc(&path).expect("open the audio model");
        if let Ok(weights) = AudioEncoderWeights::from_gguf(&gguf) {
            return Some((weights, exact));
        }
    }
    assert!(
        std::env::var("CERA_REQUIRE_MODEL")
            .unwrap_or_default()
            .is_empty(),
        "CERA_REQUIRE_MODEL is set but no audio encoder model was found"
    );
    eprintln!("no audio encoder model found, skipping");
    None
}

#[test]
fn the_wgpu_encoder_matches_the_cpu_encoder() {
    let Some((weights, exact)) = load_encoder() else {
        return;
    };
    let Some(ops) = gpu_ops() else {
        return;
    };
    let gpu_w = GpuAudioWeights::build(&ops, &weights).expect("upload the audio encoder");
    // 3 s fits one dispatch everywhere; 25 s makes the first stem convolution split.
    for secs in [3.0f32, 25.0] {
        let pcm = test_pcm(secs);
        let bins = weights.config.n_mel_bins;
        let (mel, frames) = log_mel_spectrogram(&pcm, bins);
        let (want, want_rows) = audio_encoder_forward(&mel[..frames * bins], frames, &weights);
        let (got, rows) =
            encode_audio_mel_gpu(&ops, &gpu_w, &mel[..frames * bins], frames).expect("encode");
        assert_eq!(rows, want_rows, "{secs} s: rows");
        let tolerance = if exact { 2e-3 } else { 6e-2 };
        let err = rel_l2(&want, &got);
        // Log the margin: the quantized tier is 30x looser than the exact one, so a pass
        // should show how close it was rather than just passing.
        eprintln!("{secs} s (exact={exact}): rel-L2 {err:.3e} against limit {tolerance:.1e}");
        assert!(
            err < tolerance,
            "{secs} s: rel-L2 {err} (limit {tolerance})"
        );
        // the on-GPU front end too, against the CPU front end into the CPU encoder
        let (pcm_got, pcm_rows) = encode_audio_pcm_gpu(&ops, &gpu_w, &pcm).expect("encode pcm");
        assert_eq!(pcm_rows, want_rows, "{secs} s (pcm): rows");
        let err = rel_l2(&want, &pcm_got);
        eprintln!("{secs} s (pcm, exact={exact}): rel-L2 {err:.3e} against limit {tolerance:.1e}");
        assert!(
            err < tolerance.max(5e-3),
            "{secs} s (pcm): rel-L2 {err} (limit {})",
            tolerance.max(5e-3)
        );
    }
}

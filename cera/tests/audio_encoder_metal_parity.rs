#![cfg(all(feature = "metal", target_os = "macos"))]

//! GPU↔CPU parity for the Metal LFM2A Conformer audio encoder.
//!
//! Pins `model::audio_encoder_gpu` against `model::audio_encoder` (the CPU
//! encoder is the numerical reference) on the real LFM2.5-Audio mmproj rather
//! than synthesized weights. The Conformer has ~40 tensors per block across 17
//! blocks; a synthetic fixture large enough to exercise the depthwise stem, the
//! Transformer-XL attention *and* the Q4_0 GEMM path would be most of a model
//! file, and it would not catch the class of bug that matters most here (a
//! layout assumption that happens to hold for a square, uniform fixture and not
//! for the shipped shapes).
//!
//! Checks two stage boundaries, not just the end:
//!
//! 1. **Conv stem**: `encoder_input_gpu` vs `conv_stem_forward`. Covers
//!    `conv2d_direct` in all three stem modes (regular, depthwise, pointwise),
//!    `relu_inplace`, `transpose_blocked` with a non-unit inner block, and the
//!    Q4_0 `pre_encode_out` projection.
//! 2. **Full encoder**: `encode_audio_mel_gpu` vs `audio_encoder_forward`.
//!    Everything above plus 17 blocks of macaron FFN, XL attention, and the conv
//!    module, then the erf-GELU adapter.
//!
//! ## Tolerances, and why the CPU is the approximate side
//!
//! The obvious gate, GPU against the shipping CPU encoder, has to be loose,
//! and the reason is worth stating because it is the opposite of what it looks
//! like. On a Q4_0 weight (which is every linear in this mmproj)
//! `MmapWeight::gemv` dispatches to `gemv_q4_0_f32`, which quantizes the
//! **activations** to Q8_0 blocks and does an int8 dot. The GPU keeps
//! activations in f32 on both its GEMM paths. Across ~140 linear layers that
//! makes the CPU reference, not the GPU, the lossy one.
//!
//! So each test carries a second comparison against the *same* CPU forward pass
//! run with every linear weight dequantized to F32 (`to_f32_encoder`), which
//! removes the activation quantization and leaves only the CPU's f64 attention
//! accumulation and `ggml_expf`. Measured on an M1 Max, whole encoder, 17
//! blocks:
//!
//! | comparison                        | rel-L2  |
//! |-----------------------------------|---------|
//! | GPU vs f32 reference              | 1.1e-6  |
//! | shipping Q4_0 CPU vs f32 reference| 1.7e-2  |
//!
//! The GPU is ~15000x closer to the exact answer than the path it is being
//! checked against. That is why the tight gates are against the f32 reference
//! and the loose ones against the shipping path, and why each test also asserts
//! the GPU is *closer* to the reference than the CPU is. A real kernel bug
//! fails that ordering, activation quantization cannot.
//!
//! Skips cleanly when the mmproj is absent or no Metal device exists;
//! `CERA_REQUIRE_METAL=1` turns the missing device into a failure, per the
//! project-wide convention.

mod common;

use cera::model::audio_encoder::{
    AudioEncoderWeights, SAMPLE_RATE, audio_encoder_forward, conv_stem_forward,
};
use cera::model::audio_encoder_gpu::{
    GpuAudioWeights, MetalAudioOps, encode_audio_mel_gpu, encoder_input_gpu,
};
use cera::model::audio_preprocessor::log_mel_spectrogram;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Cosine similarity, or `None` when a side has no direction to compare.
///
/// A zero vector points nowhere, so no return value honestly describes its
/// agreement with anything: 0.0 claims the two are orthogonal, 1.0 claims they
/// match, and both are inventions. Handing the caller the ambiguity keeps that
/// decision at the gate, where the two zero cases are not alike. One side zero
/// is the shape of a kernel that never ran and left its output buffer as
/// allocated, which is the single most likely real failure here. Both sides zero
/// means the reference itself carries no signal, so the comparison proves
/// nothing rather than proving agreement, and answering 1.0 would turn a
/// measurement of nothing into a pass.
fn cosine_sim(a: &[f32], b: &[f32]) -> Option<f64> {
    assert_eq!(a.len(), b.len(), "cosine_sim on mismatched lengths");
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na * nb))
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Assert `got` matches `want` per element within `atol + rtol * |want|`, and
/// that the two agree in direction.
///
/// The per-element mixed bound is the form `vision_encoder_gpu`'s `run_parity`
/// uses, for the reason documented there: GEMM error scales with the magnitude
/// of the element being computed, so a flat absolute bound is simultaneously too
/// loose for the small elements and too tight for the large ones. A
/// bound expressed against the *tensor's* RMS has the same problem in reverse:
/// these activations have a long tail, and one large element drifting inside its
/// own relative budget looks like a huge deviation next to the RMS.
///
/// Cosine is checked as well, and it is not redundant: the mixed bound alone
/// would pass an output that drifted the same direction everywhere, which is
/// what a systematically wrong residual scale looks like.
fn assert_parity(label: &str, want: &[f32], got: &[f32], min_cos: f64, atol: f32, rtol: f32) {
    assert_eq!(
        want.len(),
        got.len(),
        "{label}: length {} != reference {}",
        got.len(),
        want.len()
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{label}: GPU output has non-finite values"
    );

    let cos = cosine_sim(want, got);
    let diff = max_abs_diff(want, got);
    // Element with the largest budget overrun, so the failure names the value
    // that actually broke rather than the largest raw difference.
    let (worst_i, worst_over) = want
        .iter()
        .zip(got)
        .enumerate()
        .map(|(i, (&c, &g))| (i, (c - g).abs() - (atol + rtol * c.abs())))
        .fold((0usize, f32::NEG_INFINITY), |acc, x| {
            if x.1 > acc.1 { x } else { acc }
        });
    eprintln!(
        "{label}: cosine {}, max_abs_diff {diff:.3e}, ref rms {:.3e}, \
         worst element [{worst_i}] cpu={:.4} gpu={:.4} (budget overrun {worst_over:.3e})",
        match cos {
            Some(c) => format!("{c:.6}"),
            None => "undefined (a side is all zeros)".to_string(),
        },
        rms(want),
        want[worst_i],
        got[worst_i],
    );
    // Split from the gate below so the undefined case reports what is actually
    // wrong. Folded into a number it reads as "cosine 0.000000 < 0.999", which
    // describes two vectors that disagree about direction and sends the reader
    // looking for a math error in a kernel that may never have run at all.
    let Some(cos) = cos else {
        panic!(
            "{label}: cosine is undefined because a side is all zeros \
             (reference rms {:.3e}, gpu rms {:.3e}). An all-zero GPU output is an \
             output buffer that was allocated and never written; an all-zero \
             reference means this comparison has no signal to check against.",
            rms(want),
            rms(got),
        );
    };
    assert!(
        cos >= min_cos,
        "{label}: cosine {cos:.6} < {min_cos} (max_abs_diff {diff:.3e})"
    );
    assert!(
        worst_over <= 0.0,
        "{label}: element {worst_i} is outside atol {atol} + rtol {rtol} * |{}| \
         (cpu {}, gpu {}, diff {})",
        want[worst_i],
        want[worst_i],
        got[worst_i],
        (want[worst_i] - got[worst_i]).abs(),
    );
}

/// Dequantize a linear weight to an F32 [`MmapWeight`] of the same shape.
///
/// The CPU encoder's `gemv` on a quantized weight goes through `gemv_q4_0_f32`,
/// which quantizes the *activations* to Q8_0 blocks before an int8 dot. That is
/// the single largest source of CPU↔GPU disagreement in this encoder, and it is
/// on the CPU side. Rebuilding the weights as F32 makes every CPU projection an
/// exact f32 gemv, which turns the parity check from "the two roughly agree" into
/// "the GPU matches a reference that is not itself approximating".
fn to_f32_weight(w: &cera::model::weights::MmapWeight) -> cera::model::weights::MmapWeight {
    cera::model::weights::MmapWeight::from_owned_bytes(
        bytemuck::cast_slice(&w.to_dense_f32()).to_vec(),
        cera::tensor::DType::F32,
        w.rows,
        w.cols,
    )
}

/// The whole encoder with every linear weight dequantized to F32; see
/// [`to_f32_weight`]. Everything else (norms, biases, conv kernels) is already
/// F32 in the GGUF and is cloned as-is.
fn to_f32_encoder(w: &AudioEncoderWeights) -> AudioEncoderWeights {
    use cera::model::audio_encoder::{
        AudioMlpAdapterWeights, ConformerLayerWeights, ConvLayerWeights, ConvStemWeights,
    };
    AudioEncoderWeights {
        config: w.config.clone(),
        conv_stem: ConvStemWeights {
            layers: w
                .conv_stem
                .layers
                .iter()
                .map(|l| ConvLayerWeights {
                    name: l.name.clone(),
                    weight: l.weight.clone(),
                    bias: l.bias.clone(),
                    shape: l.shape.clone(),
                })
                .collect(),
            pre_encode_out_w: to_f32_weight(&w.conv_stem.pre_encode_out_w),
            pre_encode_out_b: w.conv_stem.pre_encode_out_b.clone(),
        },
        layers: w
            .layers
            .iter()
            .map(|b| ConformerLayerWeights {
                ffn_norm_w: b.ffn_norm_w.clone(),
                ffn_norm_b: b.ffn_norm_b.clone(),
                ffn_up_w: to_f32_weight(&b.ffn_up_w),
                ffn_up_b: b.ffn_up_b.clone(),
                ffn_down_w: to_f32_weight(&b.ffn_down_w),
                ffn_down_b: b.ffn_down_b.clone(),
                ln1_w: b.ln1_w.clone(),
                ln1_b: b.ln1_b.clone(),
                attn_q_w: to_f32_weight(&b.attn_q_w),
                attn_q_b: b.attn_q_b.clone(),
                attn_k_w: to_f32_weight(&b.attn_k_w),
                attn_k_b: b.attn_k_b.clone(),
                attn_v_w: to_f32_weight(&b.attn_v_w),
                attn_v_b: b.attn_v_b.clone(),
                attn_o_w: to_f32_weight(&b.attn_o_w),
                attn_o_b: b.attn_o_b.clone(),
                pos_bias_u: b.pos_bias_u.clone(),
                pos_bias_v: b.pos_bias_v.clone(),
                linear_pos_w: to_f32_weight(&b.linear_pos_w),
                norm_conv_w: b.norm_conv_w.clone(),
                norm_conv_b: b.norm_conv_b.clone(),
                conv_pw1_w: to_f32_weight(&b.conv_pw1_w),
                conv_pw1_b: b.conv_pw1_b.clone(),
                conv_dw_w: b.conv_dw_w.clone(),
                conv_dw_b: b.conv_dw_b.clone(),
                conv_dw_shape: b.conv_dw_shape.clone(),
                conv_norm_w: b.conv_norm_w.clone(),
                conv_norm_b: b.conv_norm_b.clone(),
                conv_pw2_w: to_f32_weight(&b.conv_pw2_w),
                conv_pw2_b: b.conv_pw2_b.clone(),
                ffn_norm_1_w: b.ffn_norm_1_w.clone(),
                ffn_norm_1_b: b.ffn_norm_1_b.clone(),
                ffn_up_1_w: to_f32_weight(&b.ffn_up_1_w),
                ffn_up_1_b: b.ffn_up_1_b.clone(),
                ffn_down_1_w: to_f32_weight(&b.ffn_down_1_w),
                ffn_down_1_b: b.ffn_down_1_b.clone(),
                ln2_w: b.ln2_w.clone(),
                ln2_b: b.ln2_b.clone(),
            })
            .collect(),
        mlp_adapter: AudioMlpAdapterWeights {
            norm_w: w.mlp_adapter.norm_w.clone(),
            norm_b: w.mlp_adapter.norm_b.clone(),
            up_w: to_f32_weight(&w.mlp_adapter.up_w),
            up_b: w.mlp_adapter.up_b.clone(),
            down_w: to_f32_weight(&w.mlp_adapter.down_w),
            down_b: w.mlp_adapter.down_b.clone(),
        },
    }
}

/// Relative L2 error `||got - want|| / ||want||` over the whole tensor.
///
/// The global gate for the encoder output. Unlike a per-element bound it is not
/// hostage to one near-cancellation value, and unlike cosine it does catch a
/// uniform scale error.
fn rel_l2(want: &[f32], got: &[f32]) -> f64 {
    let num: f64 = want
        .iter()
        .zip(got)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = want.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
    if den == 0.0 { num } else { num / den }
}

/// The audio mmproj, or `None` to skip. Same `~/.leap/models` convention the
/// detokenizer parity suites use.
fn load_encoder() -> Option<AudioEncoderWeights> {
    let path = std::path::PathBuf::from(std::env::var("HOME").ok()?)
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but the audio mmproj is absent at {}",
            path.display(),
        );
        eprintln!("audio mmproj not found at {}, skipping", path.display());
        return None;
    }
    let gguf = cera::gguf::GgufFile::open_arc(&path).expect("open audio mmproj");
    Some(AudioEncoderWeights::from_gguf(&gguf).expect("parse audio encoder"))
}

/// A deterministic, broadband test signal: three tones plus a chirp, so the mel
/// spectrogram has structure across the whole filterbank rather than energy in
/// one bin. `secs` of mono PCM at the encoder's sample rate.
///
/// Broadband matters: a single tone leaves most mel bins near the log floor,
/// where post-norm values are tiny and a cosine gate passes almost anything.
fn test_pcm(secs: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let chirp = (std::f32::consts::TAU * (200.0 + 900.0 * t) * t).sin();
            0.30 * (std::f32::consts::TAU * 440.0 * t).sin()
                + 0.20 * (std::f32::consts::TAU * 1320.0 * t).sin()
                + 0.15 * (std::f32::consts::TAU * 3000.0 * t).sin()
                + 0.25 * chirp
        })
        .collect()
}

/// Load the encoder, a Metal context, and the uploaded GPU weights, or `None` to
/// skip.
type Fixture = (
    AudioEncoderWeights,
    MetalAudioOps,
    GpuAudioWeights<MetalAudioOps>,
);
fn fixture() -> Option<Fixture> {
    let weights = load_encoder()?;
    let ctx = common::metal_context()?;
    let ops = MetalAudioOps::new(ctx).expect("build Metal audio ops");
    let gpu_w = GpuAudioWeights::build(&ops, &weights).expect("upload audio encoder weights");
    Some((weights, ops, gpu_w))
}

// ── Test 1: conv subsampling stem ───────────────────────────────────────────

/// The stem, against both the shipping CPU path and a higher-precision variant
/// of it.
///
/// The plain GPU-vs-CPU gate here has to be loose (`rtol` 6e-2) and that needs
/// justifying, because the convolutions themselves are exact. See
/// [`conv2d_layer_parity`], which pins every stem layer to ~3e-6.
///
/// The gap is the `pre_encode_out` projection, and it is the **CPU** that is
/// approximating: `MmapWeight::gemv` on a Q4_0 weight goes through
/// `gemv_q4_0_f32`, which quantizes the *activations* to Q8_0 blocks and does an
/// int8 dot. Over a 4096-wide reduction of post-ReLU features with a wide
/// in-block dynamic range, that costs a few percent on individual outputs. The
/// GPU keeps activations in f32 on both its GEMM paths, so it does not have that
/// error at all: the two disagree because the reference is the lossy side.
///
/// Rather than assert that in a comment, the test proves it: it re-runs the stem
/// with `pre_encode_out` dequantized to F32 (making the CPU projection an exact
/// f32 gemv) and requires the GPU to sit *closer* to that reference than the
/// shipping Q4_0 CPU path does. A real GPU bug fails that comparison; activation
/// quantization in the reference passes it.
#[test]
fn conv_stem_parity() {
    use cera::model::audio_encoder::{ConvLayerWeights, ConvStemWeights};

    let Some((weights, ops, gpu_w)) = fixture() else {
        return;
    };

    let pcm = test_pcm(1.5);
    let (mel, n_frames) = log_mel_spectrogram(&pcm, weights.config.n_mel_bins);
    assert!(n_frames > 0, "test signal produced no mel frames");

    let (cpu_out, cpu_t) = conv_stem_forward(&mel, n_frames, &weights.conv_stem, &weights.config);
    let (gpu_out, gpu_t) =
        encoder_input_gpu(&ops, &gpu_w, &mel, n_frames).expect("GPU conv stem should run");

    assert_eq!(gpu_t, cpu_t, "stem output length disagrees");
    assert_eq!(
        gpu_t,
        gpu_w.predict_t_out(n_frames).expect("stem is runnable"),
        "predict_t_out disagrees with the stem it predicts"
    );
    // Loose: the reference's Q8_0 activation quantization has an absolute noise
    // floor of ~1.7 on this tensor (RMS 108). The tight gate is below.
    assert_parity("conv stem", &cpu_out, &gpu_out, 0.9999, 2.5, 5e-3);

    // ── The precision cross-check ──
    let packed = &weights.conv_stem.pre_encode_out_w;
    let f32_stem = ConvStemWeights {
        layers: weights
            .conv_stem
            .layers
            .iter()
            .map(|l| ConvLayerWeights {
                name: l.name.clone(),
                weight: l.weight.clone(),
                bias: l.bias.clone(),
                shape: l.shape.clone(),
            })
            .collect(),
        pre_encode_out_w: to_f32_weight(packed),
        pre_encode_out_b: weights.conv_stem.pre_encode_out_b.clone(),
    };
    let (exact_out, _) = conv_stem_forward(&mel, n_frames, &f32_stem, &weights.config);

    let gpu_err = max_abs_diff(&exact_out, &gpu_out);
    let cpu_err = max_abs_diff(&exact_out, &cpu_out);
    eprintln!("conv stem vs f32 reference: gpu {gpu_err:.3e}, shipping cpu {cpu_err:.3e}");
    assert!(
        gpu_err < cpu_err,
        "GPU stem ({gpu_err:.3e} from the f32 reference) is no closer than the Q4_0 CPU \
         path ({cpu_err:.3e}); the disagreement is not just the reference's activation \
         quantization"
    );
    // And it must be close in absolute terms, not merely closer than a lossy
    // baseline: f32 convolutions plus one f32-activation GEMM.
    assert_parity(
        "conv stem vs f32 reference",
        &exact_out,
        &gpu_out,
        0.999999,
        5e-3,
        1e-5,
    );
}

// ── Test 1b: per-stem-layer convolution ─────────────────────────────────────

/// Each stem convolution against `cpu::conv2d` in isolation, on the real
/// weights.
///
/// The stem chains five convolutions in three different modes; a fault in any
/// one of them reaches [`conv_stem_parity`] only as a blurred aggregate. Running
/// them layer by layer against the same input says *which* mode is wrong, which
/// is the difference between a grouping bug and a padding bug.
#[test]
fn conv2d_layer_parity() {
    use cera::model::audio_encoder_gpu::{AudioEncoderGpuOps, Conv2dSpec};

    let Some((weights, ops, _gpu_w)) = fixture() else {
        return;
    };

    let pcm = test_pcm(0.8);
    let (mel, n_frames) = log_mel_spectrogram(&pcm, weights.config.n_mel_bins);
    assert!(n_frames > 0);

    // The encoder's own tables, not a copy: a restated table drifts silently,
    // and this test's whole claim is that it walks the same stem.
    use cera::model::audio_encoder_gpu::{STEM_LAYER_MODES, STEM_RELU_AFTER};

    let mut cur = mel.clone();
    let (mut ch, mut h, mut w) = (1usize, n_frames, weights.config.n_mel_bins);

    for (pos, (layer, &(depthwise, stride, pad))) in weights
        .conv_stem
        .layers
        .iter()
        .zip(&STEM_LAYER_MODES)
        .enumerate()
    {
        let (kw, kh, _in_per_group, out_ch) = (
            layer.shape[0],
            layer.shape[1],
            layer.shape[2],
            layer.shape[3],
        );
        let groups = if depthwise { ch } else { 1 };
        let spec = Conv2dSpec::padded(
            ch,
            out_ch,
            h,
            w,
            (kh, kw),
            (stride, stride),
            (pad, pad),
            (pad, pad),
            groups,
        )
        .expect("stem layer geometry is runnable");

        let mut cpu_out = vec![0.0f32; spec.out_len()];
        cera::backend::cpu::conv2d(
            &cur,
            &layer.weight,
            Some(&layer.bias),
            &mut cpu_out,
            ch,
            out_ch,
            h,
            w,
            kh,
            kw,
            stride,
            stride,
            pad,
            pad,
            groups,
        );

        let gpu_in = ops.upload(&cur);
        let gpu_wt = ops.upload(&layer.weight);
        let gpu_b = ops.upload(&layer.bias);
        let gpu_buf = ops.conv2d(&gpu_in, &gpu_wt, &gpu_b, &spec);
        let mut gpu_out = ops.download(&gpu_buf, spec.out_len());

        assert_parity(
            &format!("stem conv layer {pos} ({}, groups {groups})", layer.name),
            &cpu_out,
            &gpu_out,
            0.999999,
            1e-4,
            1e-5,
        );

        if STEM_RELU_AFTER[pos] {
            cera::backend::cpu::relu_inplace(&mut cpu_out);
            ops.relu(&gpu_buf, spec.out_len());
            gpu_out = ops.download(&gpu_buf, spec.out_len());
            assert_parity(
                &format!("stem conv layer {pos} + relu"),
                &cpu_out,
                &gpu_out,
                0.999999,
                1e-4,
                1e-5,
            );
        }

        cur = cpu_out;
        ch = spec.out_ch;
        h = spec.h_out;
        w = spec.w_out;
    }
}

// ── Test 2: full encoder ────────────────────────────────────────────────────

/// End-to-end: mel in, LLM-hidden-size embeddings out, through all 17 Conformer
/// blocks and the adapter. This is the check that the XL attention's rel-shift,
/// the conv module's transpose round-trip, and the half-weight macaron residuals
/// are all right at once.
#[test]
fn full_encoder_parity() {
    let Some((weights, ops, gpu_w)) = fixture() else {
        return;
    };

    let pcm = test_pcm(1.5);
    let (mel, n_frames) = log_mel_spectrogram(&pcm, weights.config.n_mel_bins);
    assert!(n_frames > 0, "test signal produced no mel frames");

    let (cpu_out, cpu_t) = audio_encoder_forward(&mel, n_frames, &weights);
    let (gpu_out, gpu_t) =
        encode_audio_mel_gpu(&ops, &gpu_w, &mel, n_frames).expect("GPU encoder should run");

    assert_eq!(gpu_t, cpu_t, "encoder output length disagrees");
    assert_eq!(
        cpu_out.len(),
        cpu_t * weights.config.llm_hidden_size,
        "CPU reference is not [t_out x llm_hidden_size]"
    );
    // Against the shipping CPU path: loose, because that path quantizes its
    // activations to Q8_0 at every one of the ~140 linear layers and this is the
    // accumulated result. The tight check is the f32 reference below.
    assert_parity("full encoder", &cpu_out, &gpu_out, 0.999, 5e-2, 5e-2);

    // ── The check that actually constrains the kernels ──
    //
    // Same forward pass with every linear weight dequantized to F32, so the
    // reference is not itself approximating. What remains between the two is the
    // CPU's f64 attention accumulation and `ggml_expf`, which is small and
    // bounded, so this gate is ~50x tighter than the one above, and a genuine
    // bug in any kernel has nowhere to hide behind quantization noise.
    let exact = to_f32_encoder(&weights);
    let (exact_out, exact_t) = audio_encoder_forward(&mel, n_frames, &exact);
    assert_eq!(exact_t, cpu_t);

    let gpu_l2 = rel_l2(&exact_out, &gpu_out);
    let cpu_l2 = rel_l2(&exact_out, &cpu_out);
    eprintln!("full encoder vs f32 reference: gpu rel-L2 {gpu_l2:.3e}, shipping cpu {cpu_l2:.3e}");
    assert!(
        gpu_l2 < cpu_l2,
        "GPU output (rel-L2 {gpu_l2:.3e} from the f32 reference) is no closer than the \
         quantized CPU path ({cpu_l2:.3e}); the disagreement is not just the reference's \
         activation quantization"
    );
    assert!(
        gpu_l2 <= 1e-4,
        "GPU rel-L2 vs the f32 reference is {gpu_l2:.3e}, over 1e-4"
    );
    assert_parity(
        "full encoder vs f32 reference",
        &exact_out,
        &gpu_out,
        0.99999,
        1e-4,
        1e-4,
    );
}

// ── Test 3: sequence-length sweep ───────────────────────────────────────────

/// Parity must not depend on the sequence length.
///
/// The XL attention indexes the position embedding as `(t-1) + key - q`, so an
/// off-by-one in the rel-shift is invisible at one length and obvious at
/// another; the stem's stride-2 layers likewise change their padding behaviour
/// with the input parity (odd vs even frame counts). Three durations chosen to
/// land on different `t_out` parities.
#[test]
fn parity_holds_across_lengths() {
    let Some((weights, ops, gpu_w)) = fixture() else {
        return;
    };
    // Built once and reused: dequantizing every linear weight is the expensive
    // part of this suite, and it does not depend on the input length.
    let exact = to_f32_encoder(&weights);

    for secs in [0.35f32, 0.8, 2.1] {
        let pcm = test_pcm(secs);
        let (mel, n_frames) = log_mel_spectrogram(&pcm, weights.config.n_mel_bins);
        assert!(n_frames > 0, "{secs}s produced no mel frames");

        let (exact_out, exact_t) = audio_encoder_forward(&mel, n_frames, &exact);
        let (gpu_out, gpu_t) =
            encode_audio_mel_gpu(&ops, &gpu_w, &mel, n_frames).expect("GPU encoder should run");
        assert_eq!(gpu_t, exact_t, "{secs}s: output length disagrees");

        let label = format!("full encoder @ {secs}s (t_out {exact_t})");
        let l2 = rel_l2(&exact_out, &gpu_out);
        eprintln!("{label}: rel-L2 vs f32 reference {l2:.3e}");
        assert!(
            l2 <= 1e-4,
            "{label}: rel-L2 {l2:.3e} vs f32 reference, over 1e-4"
        );
        assert_parity(&label, &exact_out, &gpu_out, 0.99999, 1e-4, 1e-4);
    }
}

// ── Test 4: capacity guard ──────────────────────────────────────────────────

/// A chunk past the attention kernel's `MAX_AUDIO_TOKENS` must be refused before
/// anything is dispatched, so the session falls back to the CPU encoder instead
/// of the kernel writing past its workgroup scratch.
///
/// Checked through `predict_t_out` plus a real oversized call: the guard is only
/// worth anything if the forward pass actually consults it.
#[test]
fn oversized_chunk_is_refused_not_truncated() {
    let Some((weights, ops, gpu_w)) = fixture() else {
        return;
    };

    let n_mel = weights.config.n_mel_bins;
    // The stem downsamples time by 8x, so this lands well past the 1024 cap
    // without allocating an unreasonable buffer.
    let n_frames = cera::model::audio_encoder_gpu::MAX_AUDIO_TOKENS * 8 + 64;
    assert!(
        gpu_w.predict_t_out(n_frames).expect("stem is runnable")
            > cera::model::audio_encoder_gpu::MAX_AUDIO_TOKENS,
        "test fixture no longer exceeds the cap"
    );

    let mel = vec![0.0f32; n_frames * n_mel];
    let err = encode_audio_mel_gpu(&ops, &gpu_w, &mel, n_frames)
        .expect_err("oversized chunk must be refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("MAX_AUDIO_TOKENS"),
        "refusal should name the cap it hit, got: {msg}"
    );
}

// ── Test 5: empty input ─────────────────────────────────────────────────────

/// Zero frames is a valid no-op, not an error. `Session::append_audio` maps the
/// empty result to `EmptyInput` itself, and an `Err` here would send it down the
/// CPU fallback for no reason.
#[test]
fn empty_input_is_empty_output() {
    let Some((_weights, ops, gpu_w)) = fixture() else {
        return;
    };
    let (out, t) = encode_audio_mel_gpu(&ops, &gpu_w, &[], 0).expect("empty input is not an error");
    assert_eq!(t, 0);
    assert!(out.is_empty());
}

// ── Test 6: the attention kernel's strided path ─────────────────────────────

/// Reference Conformer XL attention, f64 accumulation, matching
/// `cpu::conformer_self_attention_forward` steps 4-6.
#[allow(clippy::too_many_arguments)]
fn xl_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    p: &[f32],
    bu: &[f32],
    bv: &[f32],
    tokens: usize,
    n_head: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = n_head * head_dim;
    let scale = 1.0f64 / (head_dim as f64).sqrt();
    let mut out = vec![0.0f32; tokens * dim];
    for h in 0..n_head {
        let hb = h * head_dim;
        for qi in 0..tokens {
            let qu: Vec<f32> = (0..head_dim)
                .map(|d| q[qi * dim + hb + d] + bu[hb + d])
                .collect();
            let qv: Vec<f32> = (0..head_dim)
                .map(|d| q[qi * dim + hb + d] + bv[hb + d])
                .collect();
            let mut scores = vec![0.0f32; tokens];
            for (ki, score) in scores.iter_mut().enumerate() {
                let pos = tokens - 1 + ki - qi;
                let (mut ac, mut bd) = (0.0f64, 0.0f64);
                for d in 0..head_dim {
                    ac += qu[d] as f64 * k[ki * dim + hb + d] as f64;
                    bd += qv[d] as f64 * p[pos * dim + hb + d] as f64;
                }
                *score = ((ac + bd) * scale) as f32;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = scores.iter().map(|&x| ((x - max) as f64).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..head_dim {
                let acc: f64 = exps
                    .iter()
                    .enumerate()
                    .map(|(ki, &e)| e * v[ki * dim + hb + d] as f64)
                    .sum();
                out[qi * dim + hb + d] = (acc / sum) as f32;
            }
        }
    }
    out
}

/// Exercise `audio_xl_attention` above the workgroup size.
///
/// Every other test here runs a couple of seconds of audio, so `t_out` stays
/// under 30 and each of the kernel's 256 threads handles at most one key. Real
/// input does not look like that: a 30 s chunk is ~375 post-stem tokens, so the
/// grid-strided `for (key = tid; key < tokens; key += 256)` loop is the regime
/// production always runs in, and it was the one regime nothing covered. Driving
/// the kernel directly rather than through a 22 s encoder run keeps this cheap
/// (one dispatch, no model, no 17-block CPU reference).
#[test]
fn xl_attention_strided_path_parity() {
    use cera::model::audio_encoder_gpu::AudioEncoderGpuOps;

    let Some(ctx) = common::metal_context() else {
        return;
    };
    let ops = MetalAudioOps::new(ctx).expect("build Metal audio ops");

    // 300 > 256, so every thread takes at least two keys and some take three.
    let (tokens, n_head, head_dim) = (300usize, 8usize, 64usize);
    let dim = n_head * head_dim;
    let seq_len = 2 * tokens - 1;
    let wave = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.017 + seed).sin() * 0.6 + (i as f32 * 0.003).cos() * 0.3)
            .collect()
    };
    let (q, k, v) = (
        wave(tokens * dim, 0.0),
        wave(tokens * dim, 1.7),
        wave(tokens * dim, 3.1),
    );
    let p = wave(seq_len * dim, 5.9);
    let (bu, bv) = (wave(dim, 7.3), wave(dim, 9.5));

    let want = xl_attention_ref(&q, &k, &v, &p, &bu, &bv, tokens, n_head, head_dim);
    let got = ops.download(
        &ops.xl_attention(
            &ops.upload(&q),
            &ops.upload(&k),
            &ops.upload(&v),
            &ops.upload(&p),
            &ops.upload(&bu),
            &ops.upload(&bv),
            tokens,
            n_head,
            head_dim,
        ),
        tokens * dim,
    );

    assert_parity(
        "xl_attention @ 300 tokens",
        &want,
        &got,
        0.999999,
        1e-4,
        1e-4,
    );
}

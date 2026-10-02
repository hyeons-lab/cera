//! The fused decode kernel (`rmsnorm_and_quantize_q8_0`) and the prefill route
//! (`rmsnorm_into`, then `quantize_f32_to_q8_0_into`) must produce the same
//! normalized values, Q8_0 scales and int8 quants, bit for bit.
//!
//! They did not on aarch64: the fused NEON kernel summed squares in f32 while
//! `rmsnorm_neon` sums in f64 (as ggml does), and it quantized with a different
//! reciprocal than the standalone quantizer. The resulting `inv_rms` differed by
//! a few ulps, changing most normalized values, and ~1% of the int8 quants
//! flipped, so batched prefill and per-token decode drifted apart from the first
//! layer (cosine ~0.999 on LFM2.5-230M). Both now use the stored f16 scale
//! (`quant::q8_0_activation_scale`) and its reciprocal.

use cera::backend::cpu;

/// Deterministic pseudo-random floats in `[-1, 1)`.
fn noise(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

/// Assert the two routes agree on `x`.
fn assert_routes_agree(x: &[f32], w: &[f32], label: &str) {
    let n = x.len();
    let (mut sc_f, mut q_f, mut norm_f) = (vec![0f32; n / 32], vec![0i8; n], vec![0f32; n]);
    cpu::rmsnorm_and_quantize_q8_0(x, w, 1e-5, &mut sc_f, &mut q_f, Some(&mut norm_f));

    let mut norm_s = vec![0f32; n];
    cpu::rmsnorm_into(x, &mut norm_s, w, 1e-5);
    let (mut sc_s, mut q_s) = (vec![0f32; n / 32], vec![0i8; n]);
    cpu::quantize_f32_to_q8_0_into(&norm_s, &mut sc_s, &mut q_s);

    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&norm_f), bits(&norm_s), "{label}: normalized values");
    assert_eq!(bits(&sc_f), bits(&sc_s), "{label}: scales");
    assert_eq!(q_f, q_s, "{label}: int8 quants");
}

#[test]
fn fused_rmsnorm_quantize_matches_normalize_then_quantize() {
    let mut rnd = noise(12345);
    // 2048 is a real hidden size; 32 is a single block; 96 is not a power of two.
    [32usize, 96, 1024, 2048].into_iter().for_each(|n| {
        (0..200).for_each(|i| {
            let x: Vec<f32> = (0..n).map(|_| rnd() * 3.0).collect();
            let w: Vec<f32> = (0..n).map(|_| 1.0 + rnd() * 0.3).collect();
            assert_routes_agree(&x, &w, &format!("n={n} trial={i}"));
        });
    });
}

#[test]
fn fused_rmsnorm_quantize_matches_with_outliers_and_zero_blocks() {
    let mut rnd = noise(99);
    let n = 256;
    let w = vec![1.0f32; n];
    // One huge outlier per row: the block scale it sets stresses the rounding.
    (0..100).for_each(|i| {
        let mut x: Vec<f32> = (0..n).map(|_| rnd()).collect();
        x[(i * 7) % n] = 500.0 * if i % 2 == 0 { 1.0 } else { -1.0 };
        assert_routes_agree(&x, &w, &format!("outlier trial={i}"));
    });
    // A block that is exactly zero next to live blocks (amax == 0).
    let mut x: Vec<f32> = (0..n).map(|_| rnd()).collect();
    x[64..96].fill(0.0);
    assert_routes_agree(&x, &w, "zero block");
}

#[test]
fn fused_rmsnorm_quantize_matches_on_small_magnitude_blocks() {
    // Weights spanning ten orders of magnitude give blocks whose scale is an f16
    // subnormal (and, at the small end, the smallest subnormal): the range where
    // the stored scale and the quantizer's reciprocal used to disagree, and where
    // the fused NEON kernel and the standalone quantizer must still agree.
    let mut rnd = noise(2026);
    let n = 320;
    (0..100).for_each(|i| {
        let x: Vec<f32> = (0..n).map(|_| rnd() * 3.0).collect();
        let w: Vec<f32> = (0..n)
            .map(|j| 10f32.powi(-(((j / 32) + i) % 10)) * (1.0 + rnd() * 0.3))
            .collect();
        assert_routes_agree(&x, &w, &format!("small-magnitude trial={i}"));
    });
}

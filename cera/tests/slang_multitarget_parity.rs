//! Parity for the multi-target Slang pipeline: one `.slang` source compiled to
//! **both** WGSL and MSL by build.rs, each checked against the same CPU
//! reference here.
//!
//! `softmax` is the pilot. It was picked because the two handwritten kernels it
//! shadows already agree on their binding contract (x in-place at binding 0,
//! params at binding 1) *and* deliberately disagree on their reduction:
//! `softmax.metal` uses a two-stage `simd_max`/`simd_sum` while `softmax.wgsl`
//! walks a shared-memory tree
//! because cera does not request `wgpu::Features::SUBGROUP`. The Slang source
//! keeps both via `__target_switch`, so this suite is really asking whether a
//! generated kernel can preserve a per-target fast path rather than flattening
//! to the portable one.
//!
//! Neither generated kernel is on the production path yet. Nothing regresses if
//! they are wrong; the point is to find out on real hardware first, since the
//! MSL half cannot be executed on a Linux or Windows dev box at all.
//!
//! No GGUF and no network: synthetic inputs only, so these run wherever a GPU
//! exists.

// Every test here needs a GPU backend (wgpu or Metal); with neither feature the
// reference helpers below are unused. Gate the whole file so CI's featureless
// `cargo clippy --workspace --all-targets -- -D warnings` sees nothing rather
// than dead code.
#![cfg(any(feature = "gpu", feature = "metal"))]

/// Reference softmax over `x`, matching what both kernels should produce.
///
/// Max-shifted like the kernels are, so the comparison is not measuring a
/// difference in numerical strategy that both sides intended.
fn softmax_ref(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

/// Inputs spanning several orders of magnitude and both signs, including a
/// large positive that would overflow `exp` without the max shift. A kernel
/// that dropped the shift still passes on tame inputs, so the fixture has to
/// carry the hazard itself.
fn softmax_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            match i % 4 {
                0 => (t * 0.011).sin() * 6.0,
                1 => -(t * 0.007).cos() * 12.0,
                2 => (t % 17.0) - 8.0,
                _ => 0.25 * t.sqrt(),
            }
        })
        .collect()
}

/// Broadcast bias add reference: `x[t*dim + j] += bias[j]`. A single f32 add, so
/// the generated kernel must match it bit-for-bit (no tolerance).
fn bias_add_ref(x: &[f32], bias: &[f32], dim: usize) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, v)| v + bias[i % dim])
        .collect()
}

/// `rows*dim` x values and `dim` bias values, deterministic and spanning both
/// signs so a dropped or mis-indexed bias term shows up.
fn bias_add_inputs(rows: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..rows * dim)
        .map(|i| (i as f32 * 0.013).sin() * 3.0)
        .collect();
    let bias = (0..dim).map(|j| (j as f32) * 0.5 - 4.0).collect();
    (x, bias)
}

/// Compare against `softmax_ref` on a relative tolerance, and independently
/// assert the outputs sum to 1: a reduction bug that scaled every element
/// equally would satisfy a loose elementwise check but not this one.
fn assert_softmax(label: &str, input: &[f32], got: &[f32]) {
    let expect = softmax_ref(input);
    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(&expect) {
        assert!(a.is_finite(), "{label}: non-finite output");
        max_abs = max_abs.max((a - b).abs());
    }
    let max_ref = expect.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tol = 1e-5 * max_ref.max(1e-6);
    assert!(
        max_abs <= tol,
        "{label}: max_abs_err={max_abs:.3e} > tol={tol:.3e} (max_ref={max_ref:.3e})"
    );

    let sum: f32 = got.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-4,
        "{label}: outputs sum to {sum}, not 1"
    );
    eprintln!("{label}: max_abs_err={max_abs:.3e} sum={sum:.6} OK");
}

/// Reference tanh-approximation GELU, matching `gelu.slang` (and `cpu::
/// gelu_inplace`) including the tanh-argument clamp: without it a GPU tanh
/// computed as (exp(2a)-1)/(exp(2a)+1) goes NaN on the saturated tail, so the
/// reference has to clamp identically or the fixture's large inputs disagree for
/// the wrong reason.
fn gelu_ref(x: &[f32]) -> Vec<f32> {
    // sqrt(2/pi), truncated to the shortest literal that is the same f32 the
    // kernel's `0.7978845608f` compiles to (clippy::excessive_precision).
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    x.iter()
        .map(|&xv| {
            let inner = (SQRT_2_OVER_PI * (xv + 0.044715 * xv * xv * xv)).clamp(-15.0, 15.0);
            0.5 * xv * (1.0 + inner.tanh())
        })
        .collect()
}

/// Deterministic, both signs, and deliberately including magnitudes past the
/// clamp knee (|arg| > 15 around |x| ~ 17) so the saturated tail is exercised;
/// a kernel that dropped the clamp returns NaN there and fails.
fn gelu_inputs(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            match i % 5 {
                0 => (t * 0.013).sin() * 4.0,
                1 => -(t * 0.009).cos() * 3.0,
                2 => (t % 11.0) - 5.0,
                3 => 20.0,
                _ => -18.0,
            }
        })
        .collect()
}

/// tanh is transcendental, so the generated kernel is held to a relative bound
/// (scaled by the largest reference magnitude) rather than bit-equality: the
/// GPU's tanh and the host's differ by a few ULP, amplified by the `0.5 * x`
/// factor on the large-|x| fixture entries.
fn assert_gelu(label: &str, input: &[f32], got: &[f32]) {
    assert_close(label, got, &gelu_ref(input), 1e-4);
}

/// Deterministic a/b inputs for the elementwise ops, both signs, `b` kept away
/// from zero so `mul` cannot pass by accident on an all-zero addend.
fn elementwise_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..n)
        .map(|i| (i as f32 * 0.017).sin() * 3.0 - 1.0)
        .collect();
    let b = (0..n)
        .map(|i| (i as f32 * 0.011).cos() * 2.0 + 0.5)
        .collect();
    (a, b)
}

fn elementwise_add_ref(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn elementwise_mul_ref(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}
fn elementwise_scaled_add_ref(a: &[f32], b: &[f32], s: f32) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + s * y).collect()
}
fn elementwise_silu_mul_ref(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(g, y)| (g / (1.0 + (-g).exp())) * y)
        .collect()
}

/// Relative bound (scaled by the largest reference magnitude) for the
/// elementwise ops that are not bit-exact: `scaled_add` because the GPU may
/// contract `a + s*b` to an FMA the host mul-then-add does not, and `silu_mul`
/// because of the `exp`.
fn assert_close(label: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch (got {}, want {})",
        got.len(),
        want.len()
    );
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "{label}: non-finite output");
        assert!(w.is_finite(), "{label}: non-finite reference");
        max_abs = max_abs.max((g - w).abs());
    }
    let max_ref = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tol = rel * max_ref.max(1e-6);
    assert!(
        max_abs <= tol,
        "{label}: max_abs_err={max_abs:.3e} > tol={tol:.3e} (max_ref={max_ref:.3e})"
    );
    eprintln!("{label}: max_abs_err={max_abs:.3e} OK");
}

/// Deterministic Q and K vectors, both signs. `n_heads`/`n_kv_heads` model GQA,
/// where K has fewer heads than Q.
fn rope_inputs(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let q = (0..n_heads * head_dim)
        .map(|i| (i as f32 * 0.021).sin() * 2.0 - 0.5)
        .collect();
    let k = (0..n_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.017).cos() * 1.5 + 0.25)
        .collect();
    (q, k)
}

/// Non-trivial `freq_factors` (all != 1), so a kernel that ignored them diverges.
/// Only the wgsl branch has a freq_factors path (the metal branch is NEOX-only),
/// so this is used solely by the `gpu` module.
#[cfg(feature = "gpu")]
fn rope_freq_factors(half: usize) -> Vec<f32> {
    (0..half).map(|d| 1.0 + 0.1 * d as f32).collect()
}

/// Apply RoPE in-place to `n_h` heads of `buf`, the reference for `rope.slang`.
/// `rope_type` 0 = NEOX (split-halves), 1 = interleaved; `ff`, when present,
/// divides each pair's angle (Llama-3). Angle uses `powf(-2d/head_dim)`, which is
/// what both kernel branches compute (the metal branch as `1 / powr(.., 2d/..)`,
/// the wgsl branch as `pow(.., -2d/..)`).
fn rope_apply(
    buf: &mut [f32],
    n_h: usize,
    pos: u32,
    head_dim: usize,
    freq_base: f32,
    rope_type: u32,
    ff: Option<&[f32]>,
) {
    let half = head_dim / 2;
    for head in 0..n_h {
        for d in 0..half {
            let mut angle = pos as f32 * freq_base.powf(-2.0 * d as f32 / head_dim as f32);
            if let Some(ff) = ff {
                angle /= ff[d];
            }
            let (i0, i1) = if rope_type == 0 {
                (head * head_dim + d, head * head_dim + d + half)
            } else {
                (head * head_dim + 2 * d, head * head_dim + 2 * d + 1)
            };
            let (x0, x1) = (buf[i0], buf[i1]);
            buf[i0] = x0 * angle.cos() - x1 * angle.sin();
            buf[i1] = x0 * angle.sin() + x1 * angle.cos();
        }
    }
}

/// Deterministic per-head input and a shared per-element weight. Values span
/// both signs so a dropped square or a mis-indexed weight shows up.
fn per_head_rmsnorm_inputs(n_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..n_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin() * 2.0)
        .collect();
    let weight = (0..head_dim)
        .map(|j| 1.0 + 0.1 * (j as f32 * 0.05).cos())
        .collect();
    (x, weight)
}

/// Reference per-head RMSnorm: for each head, `x /= sqrt(mean(x^2) + eps)` then
/// scale by the shared weight. Sums sequentially, so the generated kernel (simd
/// or tree reduction) is held to a relative bound, not bit-equality.
fn per_head_rmsnorm_ref(
    x: &[f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for h in 0..n_heads {
        let off = h * head_dim;
        let sum_sq: f32 = x[off..off + head_dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            out[off + i] = x[off + i] * inv_rms * weight[i];
        }
    }
    out
}

/// Deterministic src/weight/bias for batched LayerNorm. `src` has a non-zero mean
/// and a healthy spread so the mean-subtraction and variance are both exercised.
fn layernorm_batch_inputs(rows: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let src = (0..rows * n)
        .map(|i| (i as f32 * 0.02).sin() * 3.0 + 0.5)
        .collect();
    let weight = (0..n)
        .map(|j| 1.0 + 0.05 * (j as f32 * 0.03).cos())
        .collect();
    let bias = (0..n).map(|j| 0.1 * (j as f32 * 0.07).sin()).collect();
    (src, weight, bias)
}

/// Reference batched affine LayerNorm, tightly packed (src_stride = dst_stride =
/// n). Accumulates mean and variance in f64 (the "true" answer), so the generated
/// f32 two-pass kernel is held to a relative bound.
fn layernorm_batch_ref(
    src: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * n];
    for r in 0..rows {
        let off = r * n;
        let mean = (0..n).map(|i| src[off + i] as f64).sum::<f64>() / n as f64;
        let var = (0..n)
            .map(|i| {
                let d = src[off + i] as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for i in 0..n {
            dst[off + i] =
                ((src[off + i] as f64 - mean) * inv_std * weight[i] as f64 + bias[i] as f64) as f32;
        }
    }
    dst
}

/// Deterministic src / weight / residual for batched RMSnorm, both signs.
fn rmsnorm_batch_inputs(rows: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let src = (0..rows * n)
        .map(|i| (i as f32 * 0.02).sin() * 2.0)
        .collect();
    let w = (0..n)
        .map(|j| 1.0 + 0.1 * (j as f32 * 0.04).cos())
        .collect();
    let residual = (0..rows * n)
        .map(|i| (i as f32 * 0.015).cos() * 1.5)
        .collect();
    (src, w, residual)
}

/// Reference batched RMSnorm (tightly packed): `dst = src * inv_rms(src) * w` per
/// row, sum of squares in f64 so the generated f32 kernel is held to a relative
/// bound.
fn rmsnorm_batch_ref(src: &[f32], w: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * n];
    for r in 0..rows {
        let off = r * n;
        let sum_sq: f64 = (0..n)
            .map(|i| {
                let v = src[off + i] as f64;
                v * v
            })
            .sum();
        let inv_rms = 1.0 / (sum_sq / n as f64 + eps as f64).sqrt();
        for i in 0..n {
            dst[off + i] = (src[off + i] as f64 * inv_rms * w[i] as f64) as f32;
        }
    }
    dst
}

/// Reference for the fused `add_rmsnorm_batch`: `src' = src + res_scale*residual`
/// (in-place), then rmsnorm(src') -> dst. Returns (post-add src, dst). The add is
/// f32 to mirror the kernel.
fn add_rmsnorm_batch_ref(
    src: &[f32],
    residual: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
    res_scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut new_src = src.to_vec();
    for r in 0..rows {
        let off = r * n;
        for i in 0..n {
            new_src[off + i] = src[off + i] + res_scale * residual[off + i];
        }
    }
    let dst = rmsnorm_batch_ref(&new_src, w, rows, n, eps);
    (new_src, dst)
}

/// Logits with a single, unambiguous maximum injected at `peak`, so the argmax
/// is well defined and the metal (strict `>`) and wgsl (explicit tie-break)
/// branches must agree. The base values span a small range and never reach the
/// injected peak.
fn argmax_inputs(n: usize, peak: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
    x[peak] = 10.0;
    x
}

/// Reference argmax: strict `>` so the lower index wins on a tie. This matches
/// the WGSL branch and the CPU `argmax` exactly; the Metal branch keeps the
/// lower lane on a tie (not necessarily the lowest index), but the fixtures
/// inject a unique peak so no tie is exercised.
fn argmax_f32_ref(x: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (i, &v) in x.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i as u32;
        }
    }
    best_i
}

/// Deterministic single-vector input and per-element weight for RMSnorm.
fn rmsnorm_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..n).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let weight = (0..n)
        .map(|j| 1.0 + 0.1 * (j as f32 * 0.04).cos())
        .collect();
    (x, weight)
}

/// Reference RMSnorm: `x * inv_rms(x) * weight`, sum of squares in f64 so the
/// generated f32 kernel is held to a relative bound.
fn rmsnorm_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f64 = x
        .iter()
        .map(|&v| {
            let v = v as f64;
            v * v
        })
        .sum();
    let inv_rms = 1.0 / (sum_sq / n as f64 + eps as f64).sqrt();
    (0..n)
        .map(|i| (x[i] as f64 * inv_rms * weight[i] as f64) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Q8_0 GEMM fixtures
//
// The second kernel in the tree, and the first production-shaped one: the Metal
// branch is a `linalg::CoopMat` port of `shaders/gemm_q8_0.metal`, the WGSL
// branch the source's untiled reference. Both are checked here against the same
// CPU dot product.
//
// This suite can only establish correctness. `softmax.slang` passed a suite
// exactly like it while carrying an extra threadgroup barrier that cost ~24% at
// the size where barrier latency dominates, so `examples/slang_gemm_bench.rs`
// is the other half of the check, not an optional extra.
// ---------------------------------------------------------------------------

/// Deterministic, sign-mixed, and not tile-aligned in period, so a kernel that
/// transposed an axis or dropped a k block does not accidentally agree.
fn gemm_data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Pack a `[m, k]` row-major f32 matrix as Q8_0: 34 bytes per 32-element block,
/// an f16 scale followed by 32 int8. Never a `#[repr(C)]` struct round-trip on
/// the host side either; the point is to write the exact byte layout the shader
/// indexes into.
fn q8_0_pack(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % 32, 0, "Q8_0 needs whole 32-element blocks");
    let mut bytes = Vec::with_capacity(data.len() / 32 * 34);
    for block in data.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for &x in block {
            bytes.push((x * id).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    bytes
}

/// `dst[row + col * m] = sum_k w[row, k] * x[col, k]`, computed from the
/// *dequantized* weights rather than the pre-quantization ones. Referencing the
/// originals would fold quantization error into the tolerance, which is exactly
/// the band a real kernel bug would hide in.
///
/// Returns the products alongside a per-element bound `sum |w * x|`, so the
/// assertion can scale with the cancellation the shape actually has instead of
/// with an arbitrary absolute epsilon.
fn gemm_ref(w: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; m * n];
    let mut mag = vec![0.0f32; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f64;
            let mut abs = 0.0f64;
            for i in 0..k {
                let p = f64::from(w[row * k + i]) * f64::from(x[col * k + i]);
                acc += p;
                abs += p.abs();
            }
            out[row + col * m] = acc as f32;
            mag[row + col * m] = abs as f32;
        }
    }
    (out, mag)
}

/// `tol` is relative to `sum |w * x|` because the two targets carry different
/// error: the Metal branch stages dequantized weights as `half` (as the
/// handwritten kernel does), so it inherits fp16 rounding the WGSL reference
/// branch never sees. Comparing both at the looser bound would let a genuine
/// WGSL bug through.
fn assert_gemm(label: &str, got: &[f32], want: &[f32], mag: &[f32], tol: f32) {
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}: non-finite output at {i}");
        let bound = tol * mag[i].max(1e-6);
        let err = (g - w).abs();
        if err / bound > worst {
            worst = err / bound;
            worst_at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{label}: worst err is {worst:.2}x tolerance at index {worst_at} \
         (got {}, want {}, bound {:.3e})",
        got[worst_at],
        want[worst_at],
        tol * mag[worst_at].max(1e-6)
    );
    eprintln!("{label}: worst={worst:.3} of tolerance OK");
}

/// `(m, k, n)`. Three shapes, each here for a different path.
///
/// * `(64, 64, 32)`: exactly one 64x32 tile, so the direct `simdgroup_store`
///   fast path runs and the ragged epilogue never does.
/// * `(70, 96, 45)`: overhangs on both axes, so the two-round bounce through
///   `sb` runs and the clamped `thread_row`/`thread_col` reads are exercised.
/// * `(10, 32, 3)`: smaller than a tile in every dimension, one k block.
const GEMM_SHAPES: &[(usize, usize, usize)] = &[(64, 64, 32), (70, 96, 45), (10, 32, 3)];

#[cfg(feature = "gpu")]
mod wgsl {
    use super::*;
    use cera::backend::wgpu::{DevicePollExt, GpuContext, shaders};

    fn setup() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                assert!(
                    std::env::var("CERA_REQUIRE_GPU")
                        .unwrap_or_default()
                        .is_empty(),
                    "CERA_REQUIRE_GPU is set but no GPU is available: {e}"
                );
                eprintln!("[slang-multitarget] SKIP (no GPU): {e}");
                None
            }
        }
    }

    fn run_softmax(ctx: &GpuContext, input: &[f32]) -> Vec<f32> {
        let n = input.len() as u32;
        let pipeline = ctx.create_pipeline(shaders::SOFTMAX_SLANG, "softmax", "softmax_slang");

        let x = ctx.upload_f32(input, "softmax_x");
        let params = ctx.upload_storage(bytemuck::cast_slice(&[n, 0u32]), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Single workgroup by design: the kernel grid-strides over n.
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&x, input.len())
    }

    /// n < the 256-wide workgroup, so most threads contribute only the identity
    /// to both reductions. A `-inf`/`0` seed that leaked into the wrong branch
    /// shows up here and nowhere else.
    #[test]
    fn softmax_slang_shorter_than_workgroup() {
        let Some(ctx) = setup() else { return };
        let input = softmax_input(100);
        let got = run_softmax(&ctx, &input);
        assert_softmax("wgsl softmax n=100", &input, &got);
    }

    /// n not a multiple of 256, so the grid-stride loop is ragged.
    #[test]
    fn softmax_slang_ragged() {
        let Some(ctx) = setup() else { return };
        let input = softmax_input(1000);
        let got = run_softmax(&ctx, &input);
        assert_softmax("wgsl softmax n=1000", &input, &got);
    }

    /// Exact multiple of the workgroup, several strides deep.
    #[test]
    fn softmax_slang_exact_multiple() {
        let Some(ctx) = setup() else { return };
        let input = softmax_input(2048);
        let got = run_softmax(&ctx, &input);
        assert_softmax("wgsl softmax n=2048", &input, &got);
    }

    fn run_bias_add(ctx: &GpuContext, x: &[f32], bias: &[f32], dim: u32) -> Vec<f32> {
        let total = x.len() as u32;
        let pipeline = ctx.create_pipeline(shaders::BIAS_ADD_SLANG, "bias_add", "bias_add_slang");

        let x_buf = ctx.upload_f32(x, "bias_x");
        let bias_buf = ctx.upload_f32(bias, "bias_bias");
        let params = ctx.upload_storage(bytemuck::cast_slice(&[total, dim]), "bias_params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Flat grid over `total` elements, 256 per workgroup.
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&x_buf, x.len())
    }

    #[test]
    fn bias_add_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        // Ragged total (not a multiple of 256) exercises the bounds check.
        let (rows, dim) = (7usize, 40usize);
        let (x, bias) = bias_add_inputs(rows, dim);
        let got = run_bias_add(&ctx, &x, &bias, dim as u32);
        let want = bias_add_ref(&x, &bias, dim);
        assert_eq!(got, want, "wgsl bias_add rows={rows} dim={dim}");
    }

    fn run_gelu(ctx: &GpuContext, input: &[f32]) -> Vec<f32> {
        let n = input.len() as u32;
        let pipeline = ctx.create_pipeline(shaders::GELU_SLANG, "gelu_inplace", "gelu_slang");

        let x = ctx.upload_f32(input, "gelu_x");
        let params = ctx.upload_storage(bytemuck::cast_slice(&[n, 0u32]), "gelu_params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Flat grid over `n` elements, 256 per workgroup.
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&x, input.len())
    }

    #[test]
    fn gelu_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        // Ragged length exercises the bounds check; the fixture spans the clamp.
        let input = gelu_inputs(1000);
        let got = run_gelu(&ctx, &input);
        assert_gelu("wgsl gelu n=1000", &input, &got);
    }

    fn run_elementwise(
        ctx: &GpuContext,
        entry: &str,
        a: &[f32],
        b: &[f32],
        scale_bits: u32,
    ) -> Vec<f32> {
        let n = a.len() as u32;
        let pipeline = ctx.create_pipeline(shaders::ELEMENTWISE_SLANG, entry, entry);

        let a_buf = ctx.upload_f32(a, "ew_a");
        let b_buf = ctx.upload_f32(b, "ew_b");
        let params = ctx.upload_storage(bytemuck::cast_slice(&[n, scale_bits]), "ew_params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&a_buf, a.len())
    }

    /// All four shared entry points against their references. add/mul are
    /// bit-exact; scaled_add and silu_mul use a relative bound (FMA / exp).
    #[test]
    fn elementwise_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let (a, b) = elementwise_inputs(1000);
        let s = 0.75f32;

        let got = run_elementwise(&ctx, "add_inplace", &a, &b, 0);
        assert_eq!(got, elementwise_add_ref(&a, &b), "wgsl elementwise add");

        let got = run_elementwise(&ctx, "mul_inplace", &a, &b, 0);
        assert_eq!(got, elementwise_mul_ref(&a, &b), "wgsl elementwise mul");

        let got = run_elementwise(&ctx, "scaled_add_inplace", &a, &b, s.to_bits());
        assert_close(
            "wgsl elementwise scaled_add",
            &got,
            &elementwise_scaled_add_ref(&a, &b, s),
            1e-6,
        );

        let got = run_elementwise(&ctx, "silu_mul_inplace", &a, &b, 0);
        assert_close(
            "wgsl elementwise silu_mul",
            &got,
            &elementwise_silu_mul_ref(&a, &b),
            1e-5,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rope(
        ctx: &GpuContext,
        q: &[f32],
        k: &[f32],
        pos: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        freq_base: f32,
        rope_type: u32,
        ff: &[f32],
        has_ff: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let pipeline = ctx.create_pipeline(shaders::ROPE_SLANG, "rope", "rope_slang");

        let q_buf = ctx.upload_f32(q, "rope_q");
        let k_buf = ctx.upload_f32(k, "rope_k");
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[
                pos,
                n_heads,
                n_kv_heads,
                head_dim,
                freq_base.to_bits(),
                rope_type,
                has_ff,
            ]),
            "rope_params",
        );
        let ff_buf = ctx.upload_f32(ff, "rope_ff");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ff_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total = n_heads.max(n_kv_heads) * (head_dim / 2);
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        (
            ctx.download_f32(&q_buf, q.len()),
            ctx.download_f32(&k_buf, k.len()),
        )
    }

    /// The wgsl branch carries NEOX + interleaved + freq_factors, so all three
    /// are exercised. Bounds are relative (cos/sin/pow are transcendental); a
    /// wrong pairing or a dropped freq_factors term diverges far past them.
    #[test]
    fn rope_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let (n_heads, n_kv_heads, head_dim) = (8u32, 2u32, 32usize);
        let half = head_dim / 2;
        let pos = 7u32;
        let freq_base = 10000.0f32;
        let ff = rope_freq_factors(half);
        let (q, k) = rope_inputs(n_heads as usize, n_kv_heads as usize, head_dim);

        for (label, rope_type, has_ff) in [
            ("wgsl rope neox", 0u32, 0u32),
            ("wgsl rope interleaved", 1u32, 0u32),
            ("wgsl rope neox+freq", 0u32, 1u32),
        ] {
            let (gq, gk) = run_rope(
                &ctx,
                &q,
                &k,
                pos,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                freq_base,
                rope_type,
                &ff,
                has_ff,
            );
            let ffopt = (has_ff == 1).then_some(ff.as_slice());
            let mut wq = q.clone();
            let mut wk = k.clone();
            rope_apply(
                &mut wq,
                n_heads as usize,
                pos,
                head_dim,
                freq_base,
                rope_type,
                ffopt,
            );
            rope_apply(
                &mut wk,
                n_kv_heads as usize,
                pos,
                head_dim,
                freq_base,
                rope_type,
                ffopt,
            );
            assert_close(&format!("{label} q"), &gq, &wq, 1e-4);
            assert_close(&format!("{label} k"), &gk, &wk, 1e-4);
        }
    }

    fn run_per_head_rmsnorm(
        ctx: &GpuContext,
        x: &[f32],
        weight: &[f32],
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipeline = ctx.create_pipeline(
            shaders::PER_HEAD_RMSNORM_SLANG,
            "per_head_rmsnorm",
            "per_head_rmsnorm_slang",
        );
        let x_buf = ctx.upload_f32(x, "phr_x");
        let w_buf = ctx.upload_f32(weight, "phr_w");
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[head_dim, eps.to_bits(), 0u32, 0u32]),
            "phr_params",
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // One workgroup per head.
            pass.dispatch_workgroups(n_heads, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&x_buf, x.len())
    }

    #[test]
    fn per_head_rmsnorm_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let eps = 1e-5f32;
        // Typical (head_dim < the 256-wide workgroup) and a grid-strided case.
        for (n_heads, head_dim) in [(32usize, 128usize), (4, 1024)] {
            let (x, weight) = per_head_rmsnorm_inputs(n_heads, head_dim);
            let got = run_per_head_rmsnorm(&ctx, &x, &weight, n_heads as u32, head_dim as u32, eps);
            let want = per_head_rmsnorm_ref(&x, &weight, n_heads, head_dim, eps);
            assert_close(
                &format!("wgsl per_head_rmsnorm {n_heads}x{head_dim}"),
                &got,
                &want,
                1e-5,
            );
        }
    }

    fn run_layernorm_batch(
        ctx: &GpuContext,
        src: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: u32,
        n: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipeline = ctx.create_pipeline(
            shaders::LAYERNORM_BATCH_SLANG,
            "layernorm_batch",
            "layernorm_batch_slang",
        );
        let src_buf = ctx.upload_f32(src, "ln_src");
        let dst_buf = ctx.upload_f32(&vec![0.0f32; (rows * n) as usize], "ln_dst");
        let w_buf = ctx.upload_f32(weight, "ln_w");
        let b_buf = ctx.upload_f32(bias, "ln_b");
        // src_stride = dst_stride = n (tightly packed).
        let params =
            ctx.upload_storage(bytemuck::cast_slice(&[n, eps.to_bits(), n, n]), "ln_params");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // One workgroup per row.
            pass.dispatch_workgroups(rows, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&dst_buf, (rows * n) as usize)
    }

    #[test]
    fn layernorm_batch_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let eps = 1e-5f32;
        // Realistic hidden size, and a ragged n (not a multiple of 256).
        for (rows, n) in [(8usize, 4096usize), (4, 2049)] {
            let (src, weight, bias) = layernorm_batch_inputs(rows, n);
            let got = run_layernorm_batch(&ctx, &src, &weight, &bias, rows as u32, n as u32, eps);
            let want = layernorm_batch_ref(&src, &weight, &bias, rows, n, eps);
            assert_close(
                &format!("wgsl layernorm_batch {rows}x{n}"),
                &got,
                &want,
                1e-4,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rmsnorm_batch(
        ctx: &GpuContext,
        entry: &str,
        src: &[f32],
        w: &[f32],
        residual: Option<&[f32]>,
        rows: u32,
        n: u32,
        eps: f32,
        res_scale: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let pipeline = ctx.create_pipeline(shaders::RMSNORM_BATCH_SLANG, entry, entry);
        let src_buf = ctx.upload_f32(src, "rb_src");
        let dst_buf = ctx.upload_f32(&vec![0.0f32; (rows * n) as usize], "rb_dst");
        let w_buf = ctx.upload_f32(w, "rb_w");
        // params: n, eps, src_stride = n, dst_stride = n, res_scale.
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[n, eps.to_bits(), n, n, res_scale.to_bits()]),
            "rb_params",
        );
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ];
        // add_rmsnorm_batch reads residual at binding 4; rmsnorm_batch's generated
        // layout omits it, so bind it only when present.
        let res_buf = residual.map(|r| ctx.upload_f32(r, "rb_res"));
        if let Some(rb) = &res_buf {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: rb.as_entire_binding(),
            });
        }
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(rows, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        (
            ctx.download_f32(&src_buf, (rows * n) as usize),
            ctx.download_f32(&dst_buf, (rows * n) as usize),
        )
    }

    #[test]
    fn rmsnorm_batch_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let eps = 1e-5f32;
        for (rows, n) in [(8usize, 4096usize), (4, 2049)] {
            let (src, w, residual) = rmsnorm_batch_inputs(rows, n);

            // Plain: reads src, writes dst; src unchanged.
            let (_, got_dst) = run_rmsnorm_batch(
                &ctx,
                "rmsnorm_batch",
                &src,
                &w,
                None,
                rows as u32,
                n as u32,
                eps,
                1.0,
            );
            let want_dst = rmsnorm_batch_ref(&src, &w, rows, n, eps);
            assert_close(
                &format!("wgsl rmsnorm_batch {rows}x{n}"),
                &got_dst,
                &want_dst,
                1e-4,
            );

            // Fused add + rmsnorm with a Granite-style residual scale.
            let res_scale = 0.75f32;
            let (got_src, got_dst) = run_rmsnorm_batch(
                &ctx,
                "add_rmsnorm_batch",
                &src,
                &w,
                Some(&residual),
                rows as u32,
                n as u32,
                eps,
                res_scale,
            );
            let (want_src, want_dst) =
                add_rmsnorm_batch_ref(&src, &residual, &w, rows, n, eps, res_scale);
            assert_close(
                &format!("wgsl add_rmsnorm_batch src {rows}x{n}"),
                &got_src,
                &want_src,
                1e-5,
            );
            assert_close(
                &format!("wgsl add_rmsnorm_batch dst {rows}x{n}"),
                &got_dst,
                &want_dst,
                1e-4,
            );
        }
    }

    fn run_argmax_f32(ctx: &GpuContext, x: &[f32]) -> u32 {
        let n = x.len() as u32;
        let pipeline =
            ctx.create_pipeline(shaders::ARGMAX_F32_SLANG, "argmax_f32", "argmax_f32_slang");
        let x_buf = ctx.upload_f32(x, "am_x");
        let out_buf = ctx.upload_storage(bytemuck::cast_slice(&[0u32]), "am_out");
        let params = ctx.upload_storage(bytemuck::cast_slice(&[n, 0u32]), "am_params");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Single workgroup by design: the kernel grid-strides over n.
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_u32(&out_buf, 1)[0]
    }

    #[test]
    fn argmax_f32_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        // Realistic vocab sizes; peak at the start, interior, and the grid-stride
        // tail (last element) so the stride loop's end is exercised.
        for (n, peak) in [(32768usize, 12345usize), (131072, 0), (131072, 131071)] {
            let x = argmax_inputs(n, peak);
            let got = run_argmax_f32(&ctx, &x);
            assert_eq!(got, argmax_f32_ref(&x), "wgsl argmax n={n} peak={peak}");
            assert_eq!(
                got, peak as u32,
                "wgsl argmax n={n}: got {got}, want peak {peak}"
            );
        }
    }

    // The wgsl rmsnorm branch is in-place: 3 bindings (x rw, weight, params).
    fn run_rmsnorm(ctx: &GpuContext, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as u32;
        let pipeline = ctx.create_pipeline(shaders::RMSNORM_SLANG, "rmsnorm", "rmsnorm_slang");
        let x_buf = ctx.upload_f32(x, "rn_x");
        let w_buf = ctx.upload_f32(weight, "rn_w");
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[n, eps.to_bits(), 0u32, 0u32]),
            "rn_params",
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Single workgroup by design; the kernel grid-strides over n.
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&x_buf, x.len())
    }

    #[test]
    fn rmsnorm_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let eps = 1e-5f32;
        // Realistic hidden sizes plus a ragged n (not a multiple of 256).
        for n in [4096usize, 8192, 2049] {
            let (x, weight) = rmsnorm_inputs(n);
            let got = run_rmsnorm(&ctx, &x, &weight, eps);
            let want = rmsnorm_ref(&x, &weight, eps);
            assert_close(&format!("wgsl rmsnorm n={n}"), &got, &want, 1e-4);
        }
    }

    fn run_gemm(
        ctx: &GpuContext,
        packed: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let pipeline =
            ctx.create_pipeline(shaders::GEMM_Q8_0_SLANG, "gemm_q8_0", "gemm_q8_0_slang");

        let src0 = ctx.upload_storage(packed, "gemm_w");
        let src1 = ctx.upload_f32(x, "gemm_x");
        let dst = ctx.upload_f32(&vec![0.0f32; m * n], "gemm_dst");
        // x_stride = k, y_stride = m: both operands are tightly packed here.
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[m as u32, k as u32, n as u32, k as u32, m as u32, 0u32]),
            "gemm_params",
        );

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(32) as u32, m.div_ceil(64) as u32, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&dst, m * n)
    }

    /// The WGSL branch computes in f32 throughout with no half staging, so it is
    /// held to a much tighter bound than the Metal twin.
    #[test]
    fn gemm_q8_0_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        for &(m, k, n) in GEMM_SHAPES {
            let w = gemm_data(m * k, 7);
            let x = gemm_data(n * k, 13);
            let packed = q8_0_pack(&w);
            let mut deq = vec![0.0f32; m * k];
            cera::quant::dequantize_q8_0_matrix(&packed, m, k, &mut deq);

            let (want, mag) = gemm_ref(&deq, &x, m, k, n);
            let got = run_gemm(&ctx, &packed, &x, m, k, n);
            assert_gemm(
                &format!("wgsl gemm_q8_0 {m}x{k}x{n}"),
                &got,
                &want,
                &mag,
                1e-5,
            );
        }
    }
}

// Declared at the root, as every other suite does, so the gate is the shared
// one rather than a per-file copy that could drift weaker than the CI leg
// assumes. Gated to match `metal_context`'s own cfg so a non-Metal build does
// not pull in an empty module.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod common;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod msl {
    use super::*;
    use crate::common;
    use cera::backend::metal::{MetalContext, shaders};

    fn run_softmax(ctx: &MetalContext, input: &[f32]) -> Vec<f32> {
        let n = input.len() as u32;
        let pipeline = ctx
            .create_pipeline(shaders::SOFTMAX_SLANG, "softmax")
            .expect("compile generated MSL");

        let x = ctx.upload_f32(input);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n, 0u32]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&x), 0);
        enc.set_buffer(1, Some(&params), 0);
        // One threadgroup of 256, matching the kernel's grid-stride contract.
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&x, input.len())
    }

    #[test]
    fn softmax_slang_shorter_than_workgroup() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let input = softmax_input(100);
        let got = run_softmax(&ctx, &input);
        assert_softmax("msl softmax n=100", &input, &got);
    }

    #[test]
    fn softmax_slang_ragged() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let input = softmax_input(1000);
        let got = run_softmax(&ctx, &input);
        assert_softmax("msl softmax n=1000", &input, &got);
    }

    #[test]
    fn softmax_slang_exact_multiple() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let input = softmax_input(2048);
        let got = run_softmax(&ctx, &input);
        assert_softmax("msl softmax n=2048", &input, &got);
    }

    fn run_bias_add(ctx: &MetalContext, x: &[f32], bias: &[f32], dim: u32) -> Vec<f32> {
        let total = x.len() as u32;
        let pipeline = ctx
            .create_pipeline(shaders::BIAS_ADD_SLANG, "bias_add")
            .expect("compile generated MSL");

        let x_buf = ctx.upload_f32(x);
        let bias_buf = ctx.upload_f32(bias);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[total, dim]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&bias_buf), 0);
        enc.set_buffer(2, Some(&params), 0);
        // Flat grid over `total` elements, 256 per threadgroup.
        let groups = total.div_ceil(256);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&x_buf, x.len())
    }

    #[test]
    fn bias_add_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        // Ragged total (not a multiple of 256) exercises the bounds check.
        let (rows, dim) = (7usize, 40usize);
        let (x, bias) = bias_add_inputs(rows, dim);
        let got = run_bias_add(&ctx, &x, &bias, dim as u32);
        let want = bias_add_ref(&x, &bias, dim);
        assert_eq!(got, want, "msl bias_add rows={rows} dim={dim}");
    }

    fn run_gelu(ctx: &MetalContext, input: &[f32]) -> Vec<f32> {
        let n = input.len() as u32;
        let pipeline = ctx
            .create_pipeline(shaders::GELU_SLANG, "gelu_inplace")
            .expect("compile generated MSL");

        let x = ctx.upload_f32(input);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n, 0u32]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&x), 0);
        enc.set_buffer(1, Some(&params), 0);
        // Flat grid over `n` elements, 256 per threadgroup.
        let groups = n.div_ceil(256);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&x, input.len())
    }

    #[test]
    fn gelu_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        // Ragged length exercises the bounds check; the fixture spans the clamp.
        let input = gelu_inputs(1000);
        let got = run_gelu(&ctx, &input);
        assert_gelu("msl gelu n=1000", &input, &got);
    }

    fn run_elementwise(
        ctx: &MetalContext,
        entry: &str,
        a: &[f32],
        b: &[f32],
        scale_bits: u32,
    ) -> Vec<f32> {
        let n = a.len() as u32;
        let pipeline = ctx
            .create_pipeline(shaders::ELEMENTWISE_SLANG, entry)
            .expect("compile generated MSL");

        let a_buf = ctx.upload_f32(a);
        let b_buf = ctx.upload_f32(b);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n, scale_bits]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&params), 0);
        let groups = n.div_ceil(256);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&a_buf, a.len())
    }

    /// All four shared entry points against their references. add/mul are
    /// bit-exact; scaled_add and silu_mul use a relative bound (FMA / exp).
    #[test]
    fn elementwise_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let (a, b) = elementwise_inputs(1000);
        let s = 0.75f32;

        let got = run_elementwise(&ctx, "add_inplace", &a, &b, 0);
        assert_eq!(got, elementwise_add_ref(&a, &b), "msl elementwise add");

        let got = run_elementwise(&ctx, "mul_inplace", &a, &b, 0);
        assert_eq!(got, elementwise_mul_ref(&a, &b), "msl elementwise mul");

        let got = run_elementwise(&ctx, "scaled_add_inplace", &a, &b, s.to_bits());
        assert_close(
            "msl elementwise scaled_add",
            &got,
            &elementwise_scaled_add_ref(&a, &b, s),
            1e-6,
        );

        let got = run_elementwise(&ctx, "silu_mul_inplace", &a, &b, 0);
        assert_close(
            "msl elementwise silu_mul",
            &got,
            &elementwise_silu_mul_ref(&a, &b),
            1e-5,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rope(
        ctx: &MetalContext,
        q: &[f32],
        k: &[f32],
        pos: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        freq_base: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let pipeline = ctx
            .create_pipeline(shaders::ROPE_SLANG, "rope")
            .expect("compile generated MSL");

        let q_buf = ctx.upload_f32(q);
        let k_buf = ctx.upload_f32(k);
        // The metal branch reads only params[0..4]: no rope_type / freq_factors.
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            freq_base.to_bits(),
        ]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&params), 0);
        let total = n_heads.max(n_kv_heads) * (head_dim / 2);
        let groups = total.div_ceil(256);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        (ctx.read_f32(&q_buf, q.len()), ctx.read_f32(&k_buf, k.len()))
    }

    /// The metal branch is NEOX-only (mirrors rope.metal); the interleaved and
    /// freq_factors paths live only in the wgsl branch and are checked there.
    #[test]
    fn rope_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let (n_heads, n_kv_heads, head_dim) = (8u32, 2u32, 32usize);
        let pos = 7u32;
        let freq_base = 10000.0f32;
        let (q, k) = rope_inputs(n_heads as usize, n_kv_heads as usize, head_dim);

        let (gq, gk) = run_rope(
            &ctx,
            &q,
            &k,
            pos,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            freq_base,
        );
        let mut wq = q.clone();
        let mut wk = k.clone();
        rope_apply(&mut wq, n_heads as usize, pos, head_dim, freq_base, 0, None);
        rope_apply(
            &mut wk,
            n_kv_heads as usize,
            pos,
            head_dim,
            freq_base,
            0,
            None,
        );
        assert_close("msl rope neox q", &gq, &wq, 1e-4);
        assert_close("msl rope neox k", &gk, &wk, 1e-4);
    }

    fn run_per_head_rmsnorm(
        ctx: &MetalContext,
        x: &[f32],
        weight: &[f32],
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::PER_HEAD_RMSNORM_SLANG, "per_head_rmsnorm")
            .expect("compile generated MSL");
        let x_buf = ctx.upload_f32(x);
        let w_buf = ctx.upload_f32(weight);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[head_dim, eps.to_bits(), 0u32, 0u32]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&params), 0);
        // One threadgroup per head.
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n_heads as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&x_buf, x.len())
    }

    #[test]
    fn per_head_rmsnorm_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let eps = 1e-5f32;
        for (n_heads, head_dim) in [(32usize, 128usize), (4, 1024)] {
            let (x, weight) = per_head_rmsnorm_inputs(n_heads, head_dim);
            let got = run_per_head_rmsnorm(&ctx, &x, &weight, n_heads as u32, head_dim as u32, eps);
            let want = per_head_rmsnorm_ref(&x, &weight, n_heads, head_dim, eps);
            assert_close(
                &format!("msl per_head_rmsnorm {n_heads}x{head_dim}"),
                &got,
                &want,
                1e-5,
            );
        }
    }

    fn run_layernorm_batch(
        ctx: &MetalContext,
        src: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: u32,
        n: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::LAYERNORM_BATCH_SLANG, "layernorm_batch")
            .expect("compile generated MSL");
        let src_buf = ctx.upload_f32(src);
        let dst_buf = ctx.upload_f32(&vec![0.0f32; (rows * n) as usize]);
        let w_buf = ctx.upload_f32(weight);
        let b_buf = ctx.upload_f32(bias);
        // src_stride = dst_stride = n (tightly packed).
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n, eps.to_bits(), n, n]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&dst_buf), 0);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_buffer(3, Some(&b_buf), 0);
        enc.set_buffer(4, Some(&params), 0);
        // One threadgroup per row.
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&dst_buf, (rows * n) as usize)
    }

    #[test]
    fn layernorm_batch_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let eps = 1e-5f32;
        for (rows, n) in [(8usize, 4096usize), (4, 2049)] {
            let (src, weight, bias) = layernorm_batch_inputs(rows, n);
            let got = run_layernorm_batch(&ctx, &src, &weight, &bias, rows as u32, n as u32, eps);
            let want = layernorm_batch_ref(&src, &weight, &bias, rows, n, eps);
            assert_close(
                &format!("msl layernorm_batch {rows}x{n}"),
                &got,
                &want,
                1e-4,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rmsnorm_batch(
        ctx: &MetalContext,
        entry: &str,
        src: &[f32],
        w: &[f32],
        residual: Option<&[f32]>,
        rows: u32,
        n: u32,
        eps: f32,
        res_scale: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let pipeline = ctx
            .create_pipeline(shaders::RMSNORM_BATCH_SLANG, entry)
            .expect("compile generated MSL");
        let src_buf = ctx.upload_f32(src);
        let dst_buf = ctx.upload_f32(&vec![0.0f32; (rows * n) as usize]);
        let w_buf = ctx.upload_f32(w);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[
            n,
            eps.to_bits(),
            n,
            n,
            res_scale.to_bits(),
        ]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&dst_buf), 0);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_buffer(3, Some(&params), 0);
        // add_rmsnorm_batch reads residual at buffer 4. The plain rmsnorm_batch
        // entry still declares buffer 4 in its generated MSL signature (Slang
        // emits the shared binding for both entries) but never reads it, so bind
        // a valid buffer there regardless, so a debug-layer run does not flag an
        // unset buffer.
        let res_buf = residual.map(|r| ctx.upload_f32(r));
        enc.set_buffer(4, Some(res_buf.as_ref().unwrap_or(&w_buf)), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        (
            ctx.read_f32(&src_buf, (rows * n) as usize),
            ctx.read_f32(&dst_buf, (rows * n) as usize),
        )
    }

    #[test]
    fn rmsnorm_batch_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let eps = 1e-5f32;
        for (rows, n) in [(8usize, 4096usize), (4, 2049)] {
            let (src, w, residual) = rmsnorm_batch_inputs(rows, n);

            let (_, got_dst) = run_rmsnorm_batch(
                &ctx,
                "rmsnorm_batch",
                &src,
                &w,
                None,
                rows as u32,
                n as u32,
                eps,
                1.0,
            );
            let want_dst = rmsnorm_batch_ref(&src, &w, rows, n, eps);
            assert_close(
                &format!("msl rmsnorm_batch {rows}x{n}"),
                &got_dst,
                &want_dst,
                1e-4,
            );

            let res_scale = 0.75f32;
            let (got_src, got_dst) = run_rmsnorm_batch(
                &ctx,
                "add_rmsnorm_batch",
                &src,
                &w,
                Some(&residual),
                rows as u32,
                n as u32,
                eps,
                res_scale,
            );
            let (want_src, want_dst) =
                add_rmsnorm_batch_ref(&src, &residual, &w, rows, n, eps, res_scale);
            assert_close(
                &format!("msl add_rmsnorm_batch src {rows}x{n}"),
                &got_src,
                &want_src,
                1e-5,
            );
            assert_close(
                &format!("msl add_rmsnorm_batch dst {rows}x{n}"),
                &got_dst,
                &want_dst,
                1e-4,
            );
        }
    }

    fn run_argmax_f32(ctx: &MetalContext, x: &[f32]) -> u32 {
        let n = x.len() as u32;
        let pipeline = ctx
            .create_pipeline(shaders::ARGMAX_F32_SLANG, "argmax_f32")
            .expect("compile generated MSL");
        let x_buf = ctx.upload_f32(x);
        let out_buf = ctx.upload_bytes(bytemuck::cast_slice(&[0u32]));
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n, 0u32]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_buffer(2, Some(&params), 0);
        // Single threadgroup of 256; the kernel grid-strides over n.
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_u32(&out_buf, 1)[0]
    }

    #[test]
    fn argmax_f32_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        for (n, peak) in [(32768usize, 12345usize), (131072, 0), (131072, 131071)] {
            let x = argmax_inputs(n, peak);
            let got = run_argmax_f32(&ctx, &x);
            assert_eq!(got, argmax_f32_ref(&x), "msl argmax n={n} peak={peak}");
            assert_eq!(
                got, peak as u32,
                "msl argmax n={n}: got {got}, want peak {peak}"
            );
        }
    }

    // The metal rmsnorm branch is out-of-place: 4 buffers (src, dst, w, params).
    fn run_rmsnorm(ctx: &MetalContext, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as u32;
        let pipeline = ctx
            .create_pipeline(shaders::RMSNORM_SLANG, "rmsnorm")
            .expect("compile generated MSL");
        let src_buf = ctx.upload_f32(x);
        let dst_buf = ctx.upload_f32(&vec![0.0f32; x.len()]);
        let w_buf = ctx.upload_f32(weight);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n, eps.to_bits(), 0u32, 0u32]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&dst_buf), 0);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_buffer(3, Some(&params), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&dst_buf, x.len())
    }

    #[test]
    fn rmsnorm_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let eps = 1e-5f32;
        for n in [4096usize, 8192, 2049] {
            let (x, weight) = rmsnorm_inputs(n);
            let got = run_rmsnorm(&ctx, &x, &weight, eps);
            let want = rmsnorm_ref(&x, &weight, eps);
            assert_close(&format!("msl rmsnorm n={n}"), &got, &want, 1e-4);
        }
    }

    fn run_gemm(
        ctx: &MetalContext,
        packed: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::GEMM_Q8_0_SLANG, "gemm_q8_0")
            .expect("compile generated MSL");

        let src0 = ctx.upload_bytes(packed);
        let src1 = ctx.upload_f32(x);
        let dst = ctx.upload_f32(&vec![0.0f32; m * n]);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[
            m as u32, k as u32, n as u32, k as u32, m as u32, 0u32,
        ]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&src0), 0);
        enc.set_buffer(1, Some(&src1), 0);
        enc.set_buffer(2, Some(&dst), 0);
        enc.set_buffer(3, Some(&params), 0);
        // No `set_threadgroup_memory_length`: unlike the handwritten kernel,
        // which takes its scratch as a `threadgroup char *` argument, the Slang
        // source declares both staging arrays statically.
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n.div_ceil(32) as u64,
                height: m.div_ceil(64) as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&dst, m * n)
    }

    /// Looser than the WGSL bound by design: this branch stages dequantized
    /// weights as `half` before the MMA, exactly as `gemm_q8_0.metal` does, so
    /// it carries fp16 rounding on every term.
    #[test]
    fn gemm_q8_0_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        for &(m, k, n) in GEMM_SHAPES {
            let w = gemm_data(m * k, 7);
            let x = gemm_data(n * k, 13);
            let packed = q8_0_pack(&w);
            let mut deq = vec![0.0f32; m * k];
            cera::quant::dequantize_q8_0_matrix(&packed, m, k, &mut deq);

            let (want, mag) = gemm_ref(&deq, &x, m, k, n);
            let got = run_gemm(&ctx, &packed, &x, m, k, n);
            assert_gemm(
                &format!("msl gemm_q8_0 {m}x{k}x{n}"),
                &got,
                &want,
                &mag,
                2e-3,
            );
        }
    }

    /// The generated MSL must reach the MMA hardware, not a scalar loop. Same
    /// silent-failure shape as `generated_msl_keeps_simd_reduction`: a fallback
    /// here is still numerically correct, so every test above keeps passing
    /// while the kernel loses the only reason it exists.
    #[test]
    fn generated_gemm_keeps_simdgroup_matrix() {
        let src = shaders::GEMM_Q8_0_SLANG;
        assert!(
            src.contains("simdgroup_multiply_accumulate"),
            "generated MSL lost the MMA; __target_switch selected the portable branch"
        );
        assert!(
            src.contains("simdgroup_matrix<half"),
            "generated MSL lost the half weight operand, so the MMA is no longer \
             the mixed float x half form gemm_q8_0.metal uses"
        );
        // 8 accumulators + 4 weight + 2 input operands must survive as arrays;
        // a scalarized rewrite would drop the array types entirely.
        assert!(
            src.contains("array<simdgroup_matrix<float, int(8), int(8)>, int(8)>"),
            "generated MSL no longer holds 8 live simdgroup accumulators"
        );
    }

    /// The generated MSL must keep Metal's two-stage simd reduction, not fall
    /// back to the portable tree. This is the whole premise of
    /// `__target_switch`, and it is a silent failure otherwise: the tree is
    /// correct, so every test above would still pass while the kernel quietly
    /// got slower. Asserted on the shader text, since there is no runtime way
    /// to observe which branch survived.
    #[test]
    fn generated_msl_keeps_simd_reduction() {
        let src = shaders::SOFTMAX_SLANG;
        assert!(
            src.contains("simd_max"),
            "generated MSL lost simd_max; __target_switch selected the portable tree"
        );
        assert!(
            src.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// per_head_rmsnorm's whole reason to be a `__target_switch` port is the
    /// Metal simd reduction; a silent fall-through to the tree is correct but
    /// slow and no numeric test sees it.
    #[test]
    fn generated_per_head_rmsnorm_keeps_simd_sum() {
        assert!(
            shaders::PER_HEAD_RMSNORM_SLANG.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// layernorm_batch's two reductions must stay simd on Metal; a silent tree
    /// fall-through is correct but slow and invisible to the numeric tests.
    #[test]
    fn generated_layernorm_batch_keeps_simd_sum() {
        assert!(
            shaders::LAYERNORM_BATCH_SLANG.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// Both rmsnorm_batch entry points must keep the metal simd reduction; a
    /// silent tree fall-through is correct but slow and invisible to the tests.
    #[test]
    fn generated_rmsnorm_batch_keeps_simd_sum() {
        assert!(
            shaders::RMSNORM_BATCH_SLANG.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// argmax_f32's metal branch reduces the (value, index) pair with
    /// `simd_shuffle_down`; a silent fall-through to the tree is correct but slow
    /// and invisible to the numeric test.
    #[test]
    fn generated_argmax_keeps_simd_shuffle_down() {
        assert!(
            shaders::ARGMAX_F32_SLANG.contains("simd_shuffle_down"),
            "generated MSL lost simd_shuffle_down; __target_switch selected the portable tree"
        );
    }

    /// rmsnorm's metal branch keeps the simd reduction; a silent tree
    /// fall-through is correct but slow and invisible to the numeric test.
    #[test]
    fn generated_rmsnorm_keeps_simd_sum() {
        assert!(
            shaders::RMSNORM_SLANG.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }
}

/// Slang reaches Metal's `simdgroup_matrix` hardware through `linalg::CoopMat`.
///
/// This is the finding that decides whether the eight hand-tuned
/// `simdgroup_matrix` GEMMs (the hot path, and the bulk of any migration) are
/// portable at all. Asserted on the emitted text because nothing dispatches the
/// probe and there is no runtime way to observe which instructions were chosen.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn coopmat_probe_reaches_metal_mma() {
    let src = cera::backend::metal::shaders::COOPMAT_PROBE_SLANG;
    for op in [
        "simdgroup_multiply_accumulate",
        "simdgroup_load",
        "simdgroup_store",
    ] {
        assert!(
            src.contains(op),
            "generated MSL is missing {op}; Slang no longer lowers linalg::CoopMat to Metal MMA"
        );
    }
}

/// The same source still emits valid WGSL, even though WGSL has no
/// cooperative-matrix type.
///
/// Unguarded, the `CoopMat` path fails WGSL emission outright with
/// `E36107 unavailable features in entry point`. It compiles only because
/// `__target_switch` eliminates the metal branch **before** entry-point
/// capability validation runs, which is the non-obvious part worth pinning: a
/// Slang release that reordered those two would break this and nothing else.
#[cfg(feature = "gpu")]
#[test]
fn coopmat_probe_wgsl_falls_back_cleanly() {
    let src = cera::backend::wgpu::shaders::COOPMAT_PROBE_SLANG;
    // The entry point is *named* coopmat_probe, so match on syntax the
    // cooperative-matrix path would actually emit, not on the substring "coop".
    for leaked in ["CoopMat<", "simdgroup", "subgroupMatrix"] {
        assert!(
            !src.contains(leaked),
            "generated WGSL leaked {leaked}; the metal branch was not eliminated"
        );
    }
    assert!(
        src.contains("coopmat_probe"),
        "generated WGSL is missing its entry point"
    );
}

/// `half` operands make the WGSL emission require `enable f16`, and cera asks
/// for `SHADER_F16` only when the adapter reports it (`GpuContext::new`). That
/// is why nothing builds a pipeline from this probe, and it is the constraint a
/// real GEMM port has to design around. Pinned so the coupling is not
/// rediscovered as a device-specific pipeline failure.
#[cfg(feature = "gpu")]
#[test]
fn coopmat_probe_documents_the_f16_requirement() {
    assert!(
        cera::backend::wgpu::shaders::COOPMAT_PROBE_SLANG.contains("enable f16"),
        "probe no longer requires f16; if the operands became f32 this test and the warning it guards can go"
    );
}

/// The Q8_0 GEMM is the *counter*example to the probe, and that is the point of
/// how it is bound. Its `half` staging lives entirely inside the metal branch,
/// so the eliminated WGSL never sees an f16 type and the emission stays legal on
/// adapters without `SHADER_F16`. This is the constraint
/// `coopmat_probe_documents_the_f16_requirement` warns about, designed around
/// rather than worked around, and it only holds while `src0`/`sa` keep their
/// current types.
#[cfg(feature = "gpu")]
#[test]
fn generated_gemm_wgsl_needs_no_f16() {
    let src = cera::backend::wgpu::shaders::GEMM_Q8_0_SLANG;
    assert!(
        !src.contains("enable f16"),
        "generated GEMM WGSL now requires f16; cera enables SHADER_F16 only when \
         the adapter reports it, so this would fail pipeline creation elsewhere"
    );
    for leaked in ["CoopMat<", "simdgroup", "subgroupMatrix"] {
        assert!(
            !src.contains(leaked),
            "generated WGSL leaked {leaked}; the metal branch was not eliminated"
        );
    }
}

/// The generated WGSL must **not** contain subgroup ops: cera never requests
/// `wgpu::Features::SUBGROUP`, so a wave intrinsic leaking into this target
/// fails pipeline creation on every device. Text-level because the failure
/// would otherwise surface as an opaque validation error far from its cause.
#[cfg(feature = "gpu")]
#[test]
fn generated_wgsl_has_no_subgroup_ops() {
    for (name, src) in [
        ("softmax", cera::backend::wgpu::shaders::SOFTMAX_SLANG),
        ("gemm_q8_0", cera::backend::wgpu::shaders::GEMM_Q8_0_SLANG),
        (
            "per_head_rmsnorm",
            cera::backend::wgpu::shaders::PER_HEAD_RMSNORM_SLANG,
        ),
        (
            "layernorm_batch",
            cera::backend::wgpu::shaders::LAYERNORM_BATCH_SLANG,
        ),
        (
            "rmsnorm_batch",
            cera::backend::wgpu::shaders::RMSNORM_BATCH_SLANG,
        ),
        ("argmax_f32", cera::backend::wgpu::shaders::ARGMAX_F32_SLANG),
        ("rmsnorm", cera::backend::wgpu::shaders::RMSNORM_SLANG),
    ] {
        assert!(
            !src.contains("subgroup"),
            "generated WGSL for {name} uses a subgroup op, but cera does not enable Features::SUBGROUP"
        );
    }
}

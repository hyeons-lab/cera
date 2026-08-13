//! Parity for the multi-target Slang pipeline: one `.slang` source compiled to
//! **both** WGSL and MSL by build.rs, each checked against the same CPU
//! reference here.
//!
//! `softmax` was the pilot. It was picked because the two handwritten kernels it
//! replaced already agreed on their binding contract (x in-place at binding 0,
//! params at binding 1) *and* deliberately disagreed on their reduction: the
//! Metal one used a two-stage `simd_max`/`simd_sum` while the WGSL one walked a
//! shared-memory tree, because cera does not request
//! `wgpu::Features::SUBGROUP`. The Slang source keeps both via
//! `__target_switch`, so this suite is really asking whether a generated kernel
//! can preserve a per-target fast path rather than flattening to the portable
//! one.
//!
//! **These kernels are on the production path.** They used to sit beside their
//! handwritten twins, and a wrong one regressed nothing; the twins are now
//! deleted and this suite is what stands between a wrong `.slang` and shipped
//! inference. It matters most for the MSL half, which cannot be executed on a
//! Linux or Windows dev box at all.
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
// Conv1d fixtures (LFM2 short-conv tier)
//
// Unlike the norm tier these are clean single-body ports with no
// `__target_switch`, so the question is not whether a per-target fast path
// survived but whether the shared body still writes *both* of its outputs: each
// kernel produces an output vector and advances the rolling buffer in place. A
// port that dropped the rolling-buffer update would still pass an output-only
// check, so every conv assertion here covers the returned buffer too.
//
// Every conv test sweeps [`CONV_SHAPES`], which is what makes the predicated
// `[ForceUnroll]` loops in `conv1d_fused_batch.slang` testable at all: at the
// maximum shape (ks=4, d_conv=3) every predicate in those loops is
// unconditionally true, so replacing them all with `true` would still pass. The
// shape that actually ships is the smaller one.
// ---------------------------------------------------------------------------

/// `(kernel_size, d_conv)` pairs every conv test sweeps.
///
/// `(3, 2)` is the production shape: shipped LFM2 GGUFs set
/// `lfm2.shortconv.l_cache = 3`, and the Metal and wgpu hosts both derive
/// `kernel_size = l_cache`, `d_conv = kernel_size - 1`. `(4, 3)` is the maximum
/// the `conv1d_fused_batch` register arrays are sized for, so it pins the
/// boundary. `(2, 1)` degenerates the rolling buffer to a single slot, which is
/// where an off-by-one in the shift loop shows up.
const CONV_SHAPES: &[(usize, usize)] = &[(3, 2), (4, 3), (2, 1)];

/// Deterministic conv input, rolling-buffer state and per-channel weights.
/// `weight` is `hs x ks` with the current-input tap at column `d_conv`.
fn conv_inputs(hs: usize, ks: usize, d_conv: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input = (0..hs).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let rbuffer = (0..d_conv * hs)
        .map(|i| (i as f32 * 0.017).cos() * 0.5)
        .collect();
    let weight = (0..hs * ks)
        .map(|i| 0.1 + 0.05 * (i as f32 * 0.03).sin())
        .collect();
    (input, rbuffer, weight)
}

/// Packed `[x | c | b]` conv projection for `n_tokens` tokens, the layout both
/// fused kernels read (`proj_stride = 3 * hs`).
fn conv_proj_inputs(hs: usize, n_tokens: usize) -> Vec<f32> {
    (0..n_tokens * 3 * hs)
        .map(|i| (i as f32 * 0.007).sin() * 1.5)
        .collect()
}

/// Reference depthwise conv1d: one output element per channel, then the rolling
/// buffer shifted left one slot with `input` appended. Returns
/// `(output, rbuffer)`.
fn conv1d_ref(
    input: &[f32],
    rbuffer: &[f32],
    weight: &[f32],
    hs: usize,
    ks: usize,
    d_conv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; hs];
    let mut rb = rbuffer.to_vec();
    for ch in 0..hs {
        let mut sum = 0.0f32;
        for k in 0..d_conv {
            sum += rb[k * hs + ch] * weight[ch * ks + k];
        }
        sum += input[ch] * weight[ch * ks + d_conv];
        out[ch] = sum;
        // Shift after accumulating, ascending so the in-place move is a left
        // shift, exactly as the kernel does it.
        for k in 0..d_conv.saturating_sub(1) {
            rb[k * hs + ch] = rb[(k + 1) * hs + ch];
        }
        if d_conv > 0 {
            rb[(d_conv - 1) * hs + ch] = input[ch];
        }
    }
    (out, rb)
}

/// Reference batched fused conv: per token `bx = x * b`, conv over the rolling
/// state, then `c * sum`, walking tokens in order because the state is
/// sequential. Returns `(output, rbuffer)`. The single-token `conv1d_fused` is
/// this with `n_tokens = 1`, so both kernels are pinned to one reference.
fn conv1d_fused_batch_ref(
    proj: &[f32],
    rbuffer: &[f32],
    weight: &[f32],
    hs: usize,
    ks: usize,
    d_conv: usize,
    n_tokens: usize,
) -> (Vec<f32>, Vec<f32>) {
    let proj_stride = 3 * hs;
    let mut out = vec![0.0f32; n_tokens * hs];
    let mut rb = rbuffer.to_vec();
    for ch in 0..hs {
        // The kernel keeps this channel's state in registers across tokens and
        // writes it back once at the end; mirror that so the two agree on the
        // intermediate values, not just the final ones.
        let mut state: Vec<f32> = (0..d_conv).map(|k| rb[k * hs + ch]).collect();
        for t in 0..n_tokens {
            let base = t * proj_stride;
            let bx = proj[base + ch] * proj[base + 2 * hs + ch];
            let mut sum = 0.0f32;
            for k in 0..d_conv {
                sum += state[k] * weight[ch * ks + k];
            }
            sum += bx * weight[ch * ks + d_conv];
            for k in 0..d_conv.saturating_sub(1) {
                state[k] = state[k + 1];
            }
            if d_conv > 0 {
                state[d_conv - 1] = bx;
            }
            out[t * hs + ch] = proj[base + hs + ch] * sum;
        }
        for k in 0..d_conv {
            rb[k * hs + ch] = state[k];
        }
    }
    (out, rb)
}

/// Realistic LFM2 hidden sizes plus a ragged one (not a multiple of 256), which
/// is what exercises the `ch >= hs` guard.
const CONV_HIDDEN_SIZES: &[usize] = &[1024, 2048, 1537];

/// Params for both single-token conv kernels: `(hs, kernel_size, d_conv, pad)`.
fn conv_params(hs: usize, ks: usize, d_conv: usize) -> [u32; 4] {
    [hs as u32, ks as u32, d_conv as u32, 0]
}

/// Params for the batched kernel, which adds the token count and the two
/// strides.
fn conv_batch_params(hs: usize, ks: usize, d_conv: usize, n_tokens: usize) -> [u32; 6] {
    [
        hs as u32,
        ks as u32,
        d_conv as u32,
        n_tokens as u32,
        (3 * hs) as u32,
        hs as u32,
    ]
}

/// `(hidden_size, n_tokens)` for the batched kernel: a realistic prefill batch,
/// a short one, and a ragged pair. Token count is the axis that matters most
/// there, since the rolling state has to stay correct across every token.
const CONV_BATCH_SHAPES: &[(usize, usize)] = &[(1024, 64), (2048, 7), (1537, 33)];

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

// ── GPU ISTFT kernels (exp_polar, overlap_add) ──────────────────────────────

/// `exp_polar` reference: per frame, `[log_abs | angle]` (each `bins` wide) maps
/// to `[re | im]` with `re_j = exp(log_abs_j)·cos(angle_j)`, `im_j =
/// exp(log_abs_j)·sin(angle_j)`. Input and output share the `2·bins` stride.
fn exp_polar_ref(spectrum: &[f32], n_frames: usize, bins: usize) -> Vec<f32> {
    let fs = 2 * bins;
    let mut out = vec![0.0f32; n_frames * fs];
    for f in 0..n_frames {
        let base = f * fs;
        for j in 0..bins {
            let mag = spectrum[base + j].exp();
            out[base + j] = mag * spectrum[base + bins + j].cos();
            out[base + bins + j] = mag * spectrum[base + bins + j].sin();
        }
    }
    out
}

/// `overlap_add` reference: window each length-`n_fft` frame with `hann`, lay it
/// at offset `frame·hop`, sum overlaps, and normalize by the accumulated window
/// energy. Output length is `n_frames·hop` (no startup-pad strip here; that is
/// the decoder's CPU-side tail). Matches the kernel's per-position formula.
fn overlap_add_ref(
    time_domain: &[f32],
    hann: &[f32],
    n_frames: usize,
    n_fft: usize,
    hop: usize,
) -> Vec<f32> {
    let total = n_frames * hop;
    (0..total)
        .map(|g| {
            let i_hi = (g / hop).min(n_frames - 1);
            let i_lo = if g >= n_fft { (g - n_fft) / hop + 1 } else { 0 };
            let mut numer = 0.0f32;
            let mut denom = 0.0f32;
            for i in i_lo..=i_hi {
                let local = g - i * hop;
                let w = hann[local];
                numer += time_domain[i * n_fft + local] * w;
                denom += w * w;
            }
            if denom > 1e-8 { numer / denom } else { numer }
        })
        .collect()
}

// ── Mixture-of-experts (lfm2moe) ────────────────────────────────────────────

// Shape of the MoE fixtures. Small enough to keep the reference cheap, but with
// every dimension distinct, so a kernel that read the slot dimension as the
// token dimension (or the reverse) lands on different data instead of aliasing
// onto the right answer.

/// Experts per routed layer.
const MOE_N_EXPERT: usize = 6;
/// Experts activated per token. Deliberately unequal to `MOE_N_TOKENS`.
const MOE_N_USED: usize = 3;
/// Tokens per fixture batch.
const MOE_N_TOKENS: usize = 4;
/// Rows per expert projection.
const MOE_M: usize = 10;
/// Inner dimension, chosen so `nb = k / 32 = 35` is both *greater* than the
/// GEMV's 32-thread workgroup and not a multiple of it.
///
/// Both halves matter and an earlier `k = 96` had neither. `nb > WG` is what
/// makes threads take a second iteration of the grid-stride block loop, so the
/// cross-block accumulation (`sum += acc * scale`) is actually executed twice
/// by a thread; at `nb = 3` no thread ever loops, and a kernel that overwrote
/// `sum` instead of accumulating passed. `nb % WG != 0` then leaves that loop
/// ragged, so the threads that stop early are exercised too. The real down
/// projection has both of those properties as well (`1792 / 32 = 56`); what the
/// fixture reproduces is the properties, not the size.
const MOE_K: usize = 1120;

/// Pack a `[.., k]` row-major f32 buffer as Q4_0: 18 bytes per 32-element block,
/// an f16 scale followed by 16 bytes of nibble pairs, where byte `i` holds
/// element `i` in its low nibble and element `i + 16` in its high one.
///
/// Written out by hand rather than reusing `cera::quant`'s quantizer so the
/// fixture states the byte layout the shader indexes, independently of the
/// production packer it is meant to agree with. The *dequantizer* is not
/// duplicated: [`q4_0_dequant`] calls into `cera::quant`, so the reference the
/// kernels are scored against is the crate's own.
fn q4_0_pack(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % 32, 0, "Q4_0 needs whole 32-element blocks");
    let mut bytes = Vec::with_capacity(data.len() / 32 * 18);
    for block in data.as_chunks::<32>().0 {
        // Q4_0 is symmetric around 8, so the scale comes from the most negative
        // value, matching `quant::BlockQ4_0`'s layout and `cera-wasm`'s packer.
        let amax = block
            .iter()
            .fold(0f32, |m, &x| if x.abs() > m.abs() { x } else { m });
        let d = amax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let nib = |x: f32| ((x * id + 8.5) as i32).clamp(0, 15) as u8;
        bytes.extend((0..16).map(|i| nib(block[i]) | (nib(block[i + 16]) << 4)));
    }
    bytes
}

/// The f32 values a Q4_0 buffer packed by [`q4_0_pack`] decodes back to.
///
/// The references below contract against *these*, not the pre-quantization
/// floats: comparing against the originals would fold quantization error into
/// the tolerance, which is the band a real kernel bug would hide in.
///
/// Delegates to `cera::quant`, as the Q8_0 cases in this file already do, so
/// what the kernels are scored against is the crate's own dequantizer rather
/// than a second copy of the nibble unpacking that could drift away from it.
fn q4_0_dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; bytes.len() / 18 * 32];
    cera::quant::dequantize_q4_0_row(bytes, &mut out);
    out
}

/// Router logits and per-expert selection bias.
///
/// The bias is what makes this fixture worth having: `lfm2moe` trains it to
/// balance expert load, which pushes the ranking scores toward each other, so
/// the bias here is scaled to land ties and near-ties on the top-k boundary.
/// A kernel that ranked by the unbiased probability, or that broke ties toward
/// the higher index, disagrees here and nowhere else.
fn moe_route_inputs() -> (Vec<f32>, Vec<f32>) {
    let mut logits = gemm_data(MOE_N_TOKENS * MOE_N_EXPERT, 7);
    let mut bias: Vec<f32> = (0..MOE_N_EXPERT).map(|e| (e as f32 - 2.5) * 0.05).collect();

    // Plant an exact tie, so the documented "ties go to the lower expert index"
    // is a property the fixture can actually falsify. Pseudo-random logits never
    // produce one: the smallest pairwise score gap here is ~2e-3, so flipping
    // the kernel's `>` to `>=` left every parity test green.
    //
    // Constructed rather than searched for, because the tie has to survive in
    // f32 on two backends: experts 2 and 3 are given equal biases and, for token
    // 1, equal logits, so `sigmoid(l) + b` is not merely close but bit-identical.
    // Both logits are large enough to put the pair inside the top `MOE_N_USED`,
    // where the tie-break decides which *slot* each takes; since `sel_expert` is
    // compared elementwise, swapping them fails the test.
    bias[3] = bias[2];
    logits[MOE_N_EXPERT + 2] = 3.0;
    logits[MOE_N_EXPERT + 3] = 3.0;

    (logits, bias)
}

/// Sigmoid, rank by `prob + bias`, weight by the renormalized *unbiased* prob.
///
/// Mirrors `lfm2::select_experts`, including its two details that are invisible
/// in aggregate output: ties go to the lower expert index, and the divisor is
/// clamped to f16's smallest positive normal.
fn moe_route_ref(logits: &[f32], bias: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(MOE_N_TOKENS * MOE_N_USED);
    let mut weights = Vec::with_capacity(MOE_N_TOKENS * MOE_N_USED);
    for tok in 0..MOE_N_TOKENS {
        let probs: Vec<f32> = (0..MOE_N_EXPERT)
            .map(|e| 1.0 / (1.0 + (-logits[tok * MOE_N_EXPERT + e]).exp()))
            .collect();
        let mut chosen: Vec<usize> = Vec::with_capacity(MOE_N_USED);
        for _ in 0..MOE_N_USED {
            let best = (0..MOE_N_EXPERT)
                .filter(|e| !chosen.contains(e))
                .max_by(|&a, &b| {
                    (probs[a] + bias[a])
                        .total_cmp(&(probs[b] + bias[b]))
                        .then(b.cmp(&a))
                })
                .expect("n_used <= n_expert");
            chosen.push(best);
        }
        let denom = chosen
            .iter()
            .map(|&e| probs[e])
            .sum::<f32>()
            .max(1.0 / 16384.0);
        ids.extend(chosen.iter().map(|&e| e as u32));
        weights.extend(chosen.iter().map(|&e| probs[e] / denom));
    }
    (ids, weights)
}

/// `y[entry, row] = W[sel[entry]][row, :] . x[x_row(entry), :]`, over the
/// dequantized weights. `x_by_entry` mirrors the kernel's parameter: `false`
/// reads the token's row (shared by its slots), `true` reads the entry's own.
fn moe_gemv_ref(
    w: &[f32],
    x: &[f32],
    sel: &[u32],
    n_entries: usize,
    x_by_entry: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; n_entries * MOE_M];
    let mut mag = vec![0.0f32; n_entries * MOE_M];
    for entry in 0..n_entries {
        let x_row = if x_by_entry {
            entry
        } else {
            entry / MOE_N_USED
        };
        let w_base = sel[entry] as usize * MOE_M * MOE_K;
        for row in 0..MOE_M {
            let (acc, abs) = (0..MOE_K).fold((0.0f64, 0.0f64), |(a, b), i| {
                let p = f64::from(w[w_base + row * MOE_K + i]) * f64::from(x[x_row * MOE_K + i]);
                (a + p, b + p.abs())
            });
            out[entry * MOE_M + row] = acc as f32;
            mag[entry * MOE_M + row] = abs as f32;
        }
    }
    (out, mag)
}

/// `out[tok, row] (+)= sum over the token's slots of weight[entry] * z[entry, row]`.
fn moe_combine_ref(
    z: &[f32],
    weights: &[f32],
    out: &[f32],
    accumulate: bool,
    hidden: usize,
) -> Vec<f32> {
    (0..MOE_N_TOKENS)
        .flat_map(|tok| {
            (0..hidden).map(move |row| {
                let acc: f32 = (0..MOE_N_USED)
                    .map(|s| {
                        let entry = tok * MOE_N_USED + s;
                        weights[entry] * z[entry * hidden + row]
                    })
                    .sum();
                if accumulate {
                    out[tok * hidden + row] + acc
                } else {
                    acc
                }
            })
        })
        .collect()
}

/// Deterministic polar spectrum: log-magnitudes spanning a wide dynamic range
/// (so `exp` is meaningfully exercised) and angles across `[-π, π]`.
fn exp_polar_input(n_frames: usize, bins: usize) -> Vec<f32> {
    let fs = 2 * bins;
    (0..n_frames * fs)
        .map(|i| {
            let frame = i / fs;
            let j = i % fs;
            if j < bins {
                // log-magnitude
                -6.0 + 5.0 * (0.03 * (frame * bins + j) as f32).sin()
            } else {
                // angle in [-π, π]
                std::f32::consts::PI * (0.05 * (frame + j) as f32).cos()
            }
        })
        .collect()
}

/// Deterministic time-domain frames for the overlap-add fixture.
fn overlap_add_input(n_frames: usize, n_fft: usize) -> Vec<f32> {
    (0..n_frames * n_fft)
        .map(|i| (0.017 * i as f32).sin() * 2.0 - (0.003 * i as f32).cos())
        .collect()
}

/// Hidden widths `moe_combine` is scored at.
///
/// `MOE_M` fits in one 256-thread workgroup, so it only ever runs with
/// `grp.x == 0` and cannot see a wrong group stride in
/// `row = grp.x * WG + lid.x`; production always dispatches several groups
/// (`2048 / 256 = 8`). The second width crosses that boundary and is not a
/// multiple of 256, so the ragged last group and its `row >= hidden` early-out
/// are exercised in the same case.
const MOE_COMBINE_WIDTHS: &[usize] = &[MOE_M, 300];

/// Expert ids for `entries` (token, slot) pairs.
///
/// The stride `5` is coprime with `MOE_N_EXPERT`, so the ids cycle through every
/// stacked slice. That is the whole point of the constant: a stride sharing a
/// factor with the expert count (this was `3`, and `gcd(3, 6) = 3`) walks only a
/// subset, leaving most of the stacked tensor never addressed and a wrong expert
/// stride free to pass. Shared by every GEMV case, MSL and WGSL, so it cannot
/// be "simplified" back in one of them.
fn moe_sel(entries: usize) -> Vec<u32> {
    (0..entries)
        .map(|e| ((e * 5 + 1) % MOE_N_EXPERT) as u32)
        .collect()
}

/// `[m, k, n_used, n_entries, expert_stride_bytes, x_by_entry, 0, 0]`, the two
/// `uint4`s `moe_gemv_q4_0` reads.
fn moe_gemv_params(n_entries: usize, x_by_entry: bool) -> [u32; 8] {
    let expert_stride = (MOE_M * MOE_K / 32 * 18) as u32;
    [
        MOE_M as u32,
        MOE_K as u32,
        MOE_N_USED as u32,
        n_entries as u32,
        expert_stride,
        u32::from(x_by_entry),
        0,
        0,
    ]
>>>>>>> 13e2a0f (feat(lfm2moe): run the routed FFN on the Metal and wgpu backends)
}

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
        let pipeline = ctx.create_pipeline(shaders::SOFTMAX, "softmax", "softmax_slang");

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
        let pipeline = ctx.create_pipeline(shaders::BIAS_ADD, "bias_add", "bias_add_slang");

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
        let pipeline = ctx.create_pipeline(shaders::GELU, "gelu_inplace", "gelu_slang");

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
        let pipeline = ctx.create_pipeline(shaders::ROPE, "rope", "rope_slang");

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
            shaders::PER_HEAD_RMSNORM,
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
            shaders::LAYERNORM_BATCH,
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
        let pipeline = ctx.create_pipeline(shaders::RMSNORM_BATCH, entry, entry);
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
        let pipeline = ctx.create_pipeline(shaders::ARGMAX_F32, "argmax_f32", "argmax_f32_slang");
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
        let pipeline = ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm_slang");
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

    /// Bind five storage buffers as bindings 0..4 for the conv kernels, which
    /// all share one binding contract (in/proj, rbuffer, weight, output,
    /// params).
    fn conv_bind_group(
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        bufs: [&wgpu::Buffer; 5],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Dispatch one conv kernel over `hs` channels and read back both of its
    /// outputs. All three conv kernels share a five-slot binding contract
    /// (in/proj, rbuffer, weight, output, params), so one runner serves them
    /// all; only the params word count differs.
    #[allow(clippy::too_many_arguments)]
    fn run_conv(
        ctx: &GpuContext,
        shader: &str,
        entry: &str,
        first: &[f32],
        rbuffer: &[f32],
        weight: &[f32],
        hs: usize,
        out_len: usize,
        params: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
        let pipeline = ctx.create_pipeline(shader, entry, entry);
        let first_buf = ctx.upload_f32(first, "cv_first");
        let rb_buf = ctx.upload_f32(rbuffer, "cv_rb");
        let w_buf = ctx.upload_f32(weight, "cv_w");
        let out_buf = ctx.upload_f32(&vec![0.0f32; out_len], "cv_out");
        let par_buf = ctx.upload_storage(bytemuck::cast_slice(params), "cv_params");
        let bg = conv_bind_group(
            ctx,
            &pipeline,
            [&first_buf, &rb_buf, &w_buf, &out_buf, &par_buf],
        );

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(hs.div_ceil(256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        (
            ctx.download_f32(&out_buf, out_len),
            ctx.download_f32(&rb_buf, rbuffer.len()),
        )
    }

    /// Run one shader over every `CONV_SHAPES` x `CONV_HIDDEN_SIZES` pair and
    /// check the output and the advanced rolling buffer against the reference.
    /// `shader` is either the generated kernel or its handwritten twin, so the
    /// same sweep pins both.
    fn check_conv1d(ctx: &GpuContext, tag: &str, shader: &str) {
        for &(ks, d_conv) in CONV_SHAPES {
            for &hs in CONV_HIDDEN_SIZES {
                let (input, rbuffer, weight) = conv_inputs(hs, ks, d_conv);
                let (got_out, got_rb) = run_conv(
                    ctx,
                    shader,
                    "conv1d_depthwise",
                    &input,
                    &rbuffer,
                    &weight,
                    hs,
                    hs,
                    &conv_params(hs, ks, d_conv),
                );
                let (want_out, want_rb) = conv1d_ref(&input, &rbuffer, &weight, hs, ks, d_conv);
                let label = format!("wgsl {tag} conv1d hs={hs} ks={ks} d_conv={d_conv}");
                assert_close(&label, &got_out, &want_out, 1e-5);
                assert_close(&format!("{label} rbuffer"), &got_rb, &want_rb, 1e-5);
            }
        }
    }

    /// Single-token fused conv sweep, shared by the generated kernel and the
    /// handwritten twin.
    fn check_conv1d_fused(ctx: &GpuContext, tag: &str, shader: &str) {
        for &(ks, d_conv) in CONV_SHAPES {
            for &hs in CONV_HIDDEN_SIZES {
                let (_, rbuffer, weight) = conv_inputs(hs, ks, d_conv);
                let proj = conv_proj_inputs(hs, 1);
                let (got_out, got_rb) = run_conv(
                    ctx,
                    shader,
                    "conv1d_fused",
                    &proj,
                    &rbuffer,
                    &weight,
                    hs,
                    hs,
                    &conv_params(hs, ks, d_conv),
                );
                let (want_out, want_rb) =
                    conv1d_fused_batch_ref(&proj, &rbuffer, &weight, hs, ks, d_conv, 1);
                let label = format!("wgsl {tag} conv1d_fused hs={hs} ks={ks} d_conv={d_conv}");
                assert_close(&label, &got_out, &want_out, 1e-5);
                assert_close(&format!("{label} rbuffer"), &got_rb, &want_rb, 1e-5);
            }
        }
    }

    /// Batched fused conv sweep, shared by the generated kernel and the
    /// handwritten twin.
    fn check_conv1d_fused_batch(ctx: &GpuContext, tag: &str, shader: &str) {
        for &(ks, d_conv) in CONV_SHAPES {
            for &(hs, n_tokens) in CONV_BATCH_SHAPES {
                let (_, rbuffer, weight) = conv_inputs(hs, ks, d_conv);
                let proj = conv_proj_inputs(hs, n_tokens);
                let (got_out, got_rb) = run_conv(
                    ctx,
                    shader,
                    "conv1d_fused_batch",
                    &proj,
                    &rbuffer,
                    &weight,
                    hs,
                    n_tokens * hs,
                    &conv_batch_params(hs, ks, d_conv, n_tokens),
                );
                let (want_out, want_rb) =
                    conv1d_fused_batch_ref(&proj, &rbuffer, &weight, hs, ks, d_conv, n_tokens);
                let label = format!(
                    "wgsl {tag} conv1d_fused_batch hs={hs} n={n_tokens} ks={ks} d_conv={d_conv}"
                );
                assert_close(&label, &got_out, &want_out, 1e-5);
                assert_close(&format!("{label} rbuffer"), &got_rb, &want_rb, 1e-5);
            }
        }
    }

    #[test]
    fn conv1d_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        check_conv1d(&ctx, "generated", shaders::CONV1D);
    }

    #[test]
    fn conv1d_fused_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        check_conv1d_fused(&ctx, "generated", shaders::CONV1D_FUSED);
    }

    #[test]
    fn conv1d_fused_batch_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        check_conv1d_fused_batch(&ctx, "generated", shaders::CONV1D_FUSED_BATCH);
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

    fn run_exp_polar(ctx: &GpuContext, spectrum: &[f32], n_frames: u32, bins: u32) -> Vec<f32> {
        let pipeline = ctx.create_pipeline(shaders::EXP_POLAR, "exp_polar", "exp_polar_slang");
        let spec = ctx.upload_f32(spectrum, "exp_polar_spec");
        let out = ctx.create_storage_rw((spectrum.len() * 4) as u64, "exp_polar_out");
        let params =
            ctx.upload_storage(bytemuck::cast_slice(&[n_frames, bins]), "exp_polar_params");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
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
            pass.dispatch_workgroups((n_frames * bins).div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&out, spectrum.len())
    }

    #[test]
    fn exp_polar_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        let (n_frames, bins) = (5usize, 641usize);
        let spec = exp_polar_input(n_frames, bins);
        let got = run_exp_polar(&ctx, &spec, n_frames as u32, bins as u32);
        let want = exp_polar_ref(&spec, n_frames, bins);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-4,
            "wgsl exp_polar worst abs err {worst:.3e} > 1e-4"
        );
    }

    fn run_overlap_add(
        ctx: &GpuContext,
        time_domain: &[f32],
        hann: &[f32],
        n_frames: u32,
        n_fft: u32,
        hop: u32,
    ) -> Vec<f32> {
        let pipeline =
            ctx.create_pipeline(shaders::OVERLAP_ADD, "overlap_add", "overlap_add_slang");
        let td = ctx.upload_f32(time_domain, "oa_td");
        let hann_buf = ctx.upload_f32(hann, "oa_hann");
        let total = (n_frames * hop) as usize;
        let out = ctx.create_storage_rw((total * 4) as u64, "oa_out");
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[n_frames, n_fft, hop, 0u32]),
            "oa_params",
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: td.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hann_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
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
            pass.dispatch_workgroups((n_frames * hop).div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
        ctx.download_f32(&out, total)
    }

    #[test]
    fn overlap_add_slang_matches_reference() {
        let Some(ctx) = setup() else { return };
        // 8 frames wraps well past the n_fft/hop overlap depth (1280/320 = 4).
        let (n_frames, n_fft, hop) = (8usize, 1280usize, 320usize);
        let td = overlap_add_input(n_frames, n_fft);
        let hann = cera::model::audio_decoder::build_hann(n_fft);
        let got = run_overlap_add(&ctx, &td, &hann, n_frames as u32, n_fft as u32, hop as u32);
        let want = overlap_add_ref(&td, &hann, n_frames, n_fft, hop);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-3,
            "wgsl overlap_add worst abs err {worst:.3e} > 1e-3"
        );
    }

    /// Bind `buffers` at consecutive slots 0.., dispatch `groups`, and return
    /// the pipeline's work as done. The MoE kernels all take a flat,
    /// consecutively-numbered binding set, so a single helper covers the three
    /// of them rather than three near-identical `create_bind_group` blocks.
    fn run_moe_kernel(
        ctx: &GpuContext,
        src: &str,
        entry: &str,
        buffers: &[&wgpu::Buffer],
        groups: (u32, u32),
    ) {
        let pipeline = ctx.create_pipeline(src, entry, entry);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
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
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll_wait();
    }

    /// WGSL half of `msl::moe_route_matches_cpu_selection`. Scored on synthetic
    /// data with a planted boundary tie, which is what
    /// `tests/wgpu_moe_oracle.rs` cannot do: that one drives a real model, where
    /// the selection is whatever the weights produce.
    #[test]
    fn moe_route_matches_cpu_selection() {
        let Some(ctx) = setup() else { return };
        let entries = MOE_N_TOKENS * MOE_N_USED;
        let (logits, bias) = moe_route_inputs();
        let (want_ids, want_w) = moe_route_ref(&logits, &bias);

        let logits_buf = ctx.upload_f32(&logits, "moe_logits");
        let bias_buf = ctx.upload_f32(&bias, "moe_bias");
        let sel_expert = ctx.upload_storage(bytemuck::cast_slice(&vec![0u32; entries]), "moe_sel");
        let sel_weight = ctx.upload_f32(&vec![0.0f32; entries], "moe_selw");
        let params = ctx.upload_storage(
            bytemuck::cast_slice(&[
                MOE_N_EXPERT as u32,
                MOE_N_USED as u32,
                MOE_N_TOKENS as u32,
                0,
            ]),
            "moe_route_params",
        );
        run_moe_kernel(
            &ctx,
            shaders::MOE_ROUTE,
            "moe_route",
            &[&logits_buf, &bias_buf, &sel_expert, &sel_weight, &params],
            (MOE_N_TOKENS as u32, 1),
        );

        assert_eq!(
            ctx.download_u32(&sel_expert, entries),
            want_ids,
            "wgsl moe_route picked different experts"
        );
        assert_close(
            "wgsl moe_route weights",
            &ctx.download_f32(&sel_weight, entries),
            &want_w,
            1e-5,
        );
    }

    /// WGSL half of the two `msl::moe_gemv_*` tests, covering both activation
    /// indexings in one body since only the flag and the fixtures differ.
    #[test]
    fn moe_gemv_matches_cpu() {
        let Some(ctx) = setup() else { return };
        let entries = MOE_N_TOKENS * MOE_N_USED;
        for (x_by_entry, seed) in [(false, 11u32), (true, 13)] {
            let packed = q4_0_pack(&gemm_data(MOE_N_EXPERT * MOE_M * MOE_K, seed));
            let w = q4_0_dequant(&packed);
            let x_rows = if x_by_entry { entries } else { MOE_N_TOKENS };
            let x = gemm_data(x_rows * MOE_K, seed + 1);
            let sel = moe_sel(entries);
            let (want, mag) = moe_gemv_ref(&w, &x, &sel, entries, x_by_entry);

            // Same trailing pad as the MSL path: a block start that is not
            // word-aligned makes the kernel read one `u32` past the last byte.
            let mut w_bytes = packed.clone();
            w_bytes.extend_from_slice(&[0u8; 8]);
            let w_buf = ctx.upload_storage(&w_bytes, "moe_w");
            let x_buf = ctx.upload_f32(&x, "moe_x");
            let y_buf = ctx.upload_f32(&vec![0.0f32; entries * MOE_M], "moe_y");
            let sel_buf = ctx.upload_storage(bytemuck::cast_slice(&sel), "moe_sel");
            let params = ctx.upload_storage(
                bytemuck::cast_slice(&moe_gemv_params(entries, x_by_entry)),
                "moe_gemv_params",
            );
            run_moe_kernel(
                &ctx,
                shaders::MOE_GEMV_Q4_0,
                "moe_gemv_q4_0",
                &[&w_buf, &x_buf, &y_buf, &sel_buf, &params],
                (MOE_M as u32, entries as u32),
            );

            assert_gemm(
                &format!("wgsl moe_gemv_q4_0 x_by_entry={x_by_entry}"),
                &ctx.download_f32(&y_buf, entries * MOE_M),
                &want,
                &mag,
                1e-5,
            );
        }
    }

    /// WGSL half of `msl::moe_combine_matches_cpu`, both output conventions.
    #[test]
    fn moe_combine_matches_cpu() {
        let Some(ctx) = setup() else { return };
        let entries = MOE_N_TOKENS * MOE_N_USED;
        let weights: Vec<f32> = gemm_data(entries, 16).iter().map(|v| v.abs()).collect();
        for &hidden in MOE_COMBINE_WIDTHS {
            let z = gemm_data(entries * hidden, 15);
            let seed_out = gemm_data(MOE_N_TOKENS * hidden, 17);
            for accumulate in [false, true] {
                let want = moe_combine_ref(&z, &weights, &seed_out, accumulate, hidden);
                let z_buf = ctx.upload_f32(&z, "moe_z");
                let w_buf = ctx.upload_f32(&weights, "moe_selw");
                let out_buf = ctx.upload_f32(&seed_out, "moe_out");
                let params = ctx.upload_storage(
                    bytemuck::cast_slice(&[
                        hidden as u32,
                        MOE_N_USED as u32,
                        MOE_N_TOKENS as u32,
                        u32::from(accumulate),
                    ]),
                    "moe_combine_params",
                );
                run_moe_kernel(
                    &ctx,
                    shaders::MOE_COMBINE,
                    "moe_combine",
                    &[&z_buf, &w_buf, &out_buf, &params],
                    (hidden.div_ceil(256) as u32, MOE_N_TOKENS as u32),
                );
                assert_close(
                    &format!("wgsl moe_combine hidden={hidden} accumulate={accumulate}"),
                    &ctx.download_f32(&out_buf, MOE_N_TOKENS * hidden),
                    &want,
                    1e-5,
                );
            }
        }
    }
>>>>>>> 13e2a0f (feat(lfm2moe): run the routed FFN on the Metal and wgpu backends)
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
            .create_pipeline(shaders::SOFTMAX, "softmax")
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
            .create_pipeline(shaders::BIAS_ADD, "bias_add")
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
            .create_pipeline(shaders::GELU, "gelu_inplace")
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
            .create_pipeline(shaders::ROPE, "rope")
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

    /// The metal branch is NEOX-only; the interleaved and freq_factors paths
    /// live only in the wgsl branch and are checked there.
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
            .create_pipeline(shaders::PER_HEAD_RMSNORM, "per_head_rmsnorm")
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
            .create_pipeline(shaders::LAYERNORM_BATCH, "layernorm_batch")
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
            .create_pipeline(shaders::RMSNORM_BATCH, entry)
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
            .create_pipeline(shaders::ARGMAX_F32, "argmax_f32")
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
            .create_pipeline(shaders::RMSNORM, "rmsnorm")
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

    /// Dispatch one conv kernel over `hs` channels and read back both of its
    /// outputs. All three conv kernels share a five-slot binding contract
    /// (in/proj, rbuffer, weight, output, params at buffers 0..4), so one runner
    /// serves them all; only the params word count differs.
    #[allow(clippy::too_many_arguments)]
    fn run_conv(
        ctx: &MetalContext,
        shader: &'static str,
        entry: &str,
        first: &[f32],
        rbuffer: &[f32],
        weight: &[f32],
        hs: usize,
        out_len: usize,
        params: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
        let pipeline = ctx
            .create_pipeline(shader, entry)
            .expect("compile generated MSL");

        let first_buf = ctx.upload_f32(first);
        let rb_buf = ctx.upload_f32(rbuffer);
        let w_buf = ctx.upload_f32(weight);
        let out_buf = ctx.upload_f32(&vec![0.0f32; out_len]);
        let par_buf = ctx.upload_bytes(bytemuck::cast_slice(params));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&first_buf), 0);
        enc.set_buffer(1, Some(&rb_buf), 0);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_buffer(4, Some(&par_buf), 0);
        // Flat grid over `hs` channels, 256 per threadgroup.
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: hs.div_ceil(256) as u64,
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
            ctx.read_f32(&out_buf, out_len),
            ctx.read_f32(&rb_buf, rbuffer.len()),
        )
    }

    /// Run one shader over every `CONV_SHAPES` x `CONV_HIDDEN_SIZES` pair and
    /// check the output and the advanced rolling buffer against the reference.
    /// `shader` is either the generated kernel or its handwritten twin, so the
    /// same sweep pins both.
    fn check_conv1d(ctx: &MetalContext, tag: &str, shader: &'static str) {
        for &(ks, d_conv) in CONV_SHAPES {
            for &hs in CONV_HIDDEN_SIZES {
                let (input, rbuffer, weight) = conv_inputs(hs, ks, d_conv);
                let (got_out, got_rb) = run_conv(
                    ctx,
                    shader,
                    "conv1d_depthwise",
                    &input,
                    &rbuffer,
                    &weight,
                    hs,
                    hs,
                    &conv_params(hs, ks, d_conv),
                );
                let (want_out, want_rb) = conv1d_ref(&input, &rbuffer, &weight, hs, ks, d_conv);
                let label = format!("msl {tag} conv1d hs={hs} ks={ks} d_conv={d_conv}");
                assert_close(&label, &got_out, &want_out, 1e-5);
                assert_close(&format!("{label} rbuffer"), &got_rb, &want_rb, 1e-5);
            }
        }
    }

    /// Single-token fused conv sweep, shared by the generated kernel and the
    /// handwritten twin.
    fn check_conv1d_fused(ctx: &MetalContext, tag: &str, shader: &'static str) {
        for &(ks, d_conv) in CONV_SHAPES {
            for &hs in CONV_HIDDEN_SIZES {
                let (_, rbuffer, weight) = conv_inputs(hs, ks, d_conv);
                let proj = conv_proj_inputs(hs, 1);
                let (got_out, got_rb) = run_conv(
                    ctx,
                    shader,
                    "conv1d_fused",
                    &proj,
                    &rbuffer,
                    &weight,
                    hs,
                    hs,
                    &conv_params(hs, ks, d_conv),
                );
                let (want_out, want_rb) =
                    conv1d_fused_batch_ref(&proj, &rbuffer, &weight, hs, ks, d_conv, 1);
                let label = format!("msl {tag} conv1d_fused hs={hs} ks={ks} d_conv={d_conv}");
                assert_close(&label, &got_out, &want_out, 1e-5);
                assert_close(&format!("{label} rbuffer"), &got_rb, &want_rb, 1e-5);
            }
        }
    }

    /// Batched fused conv sweep, shared by the generated kernel and the
    /// handwritten twin.
    fn check_conv1d_fused_batch(ctx: &MetalContext, tag: &str, shader: &'static str) {
        for &(ks, d_conv) in CONV_SHAPES {
            for &(hs, n_tokens) in CONV_BATCH_SHAPES {
                let (_, rbuffer, weight) = conv_inputs(hs, ks, d_conv);
                let proj = conv_proj_inputs(hs, n_tokens);
                let (got_out, got_rb) = run_conv(
                    ctx,
                    shader,
                    "conv1d_fused_batch",
                    &proj,
                    &rbuffer,
                    &weight,
                    hs,
                    n_tokens * hs,
                    &conv_batch_params(hs, ks, d_conv, n_tokens),
                );
                let (want_out, want_rb) =
                    conv1d_fused_batch_ref(&proj, &rbuffer, &weight, hs, ks, d_conv, n_tokens);
                let label = format!(
                    "msl {tag} conv1d_fused_batch hs={hs} n={n_tokens} ks={ks} d_conv={d_conv}"
                );
                assert_close(&label, &got_out, &want_out, 1e-5);
                assert_close(&format!("{label} rbuffer"), &got_rb, &want_rb, 1e-5);
            }
        }
    }

    #[test]
    fn conv1d_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        check_conv1d(&ctx, "generated", shaders::CONV1D);
    }

    #[test]
    fn conv1d_fused_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        check_conv1d_fused(&ctx, "generated", shaders::CONV1D_FUSED);
    }

    #[test]
    fn conv1d_fused_batch_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        check_conv1d_fused_batch(&ctx, "generated", shaders::CONV1D_FUSED_BATCH);
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
        // 4 KB of weights staged as half plus 4 KB of input staged as float.
        // Slang declares both arrays statically, but `build.rs` rewrites them
        // into slices of a `[[threadgroup(0)]]` parameter, because static
        // groupshared is one of three things that stop the native AGX compiler
        // folding load displacements. Setting this is harmless if that rewrite
        // declined and the arrays are still static, so it is unconditional
        // rather than gated on the shader text. The size is pinned by
        // `POSTPASS_SCRATCH_BYTES` in `build_support/msl_postpass.rs`, which
        // declines rather than rewrite a shader that outgrows it.
        enc.set_threadgroup_memory_length(0, 8192);
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

    /// The MSL post-pass must actually have run on the shader that ships.
    ///
    /// It is worth ~5% on the large shapes and it is designed to fail *softly*:
    /// when slangc moves an anchor it declines with a `cargo:warning` and emits
    /// correct-but-slower MSL, which every other test here still passes. Without
    /// this assertion the regression is invisible until someone re-runs
    /// `slang_gemm_bench` by hand. `tests/msl_postpass.rs` covers the transform
    /// itself; this covers the wiring, so it deliberately reads the OUT_DIR
    /// artifact rather than the committed one.
    #[test]
    fn generated_gemm_is_post_processed() {
        // Kept in sync with POSTPASS_MARKER in `cera/build_support/msl_postpass.rs`.
        // If that constant was deliberately bumped, update this literal to match.
        assert!(
            shaders::GEMM_Q8_0_SLANG.starts_with("// cera:msl-postpass=v"),
            "the MSL post-pass declined; check the build log for its cargo:warning. \
             The shader is still correct, but the simdgroup GEMM lost ~5%."
        );
        // Equality on the first line, not `starts_with`: the latter would also
        // accept a future `v10`, which is the trap `POSTPASS_MARKER` is matched
        // exactly to avoid.
        assert_eq!(
            shaders::GEMM_Q8_0_SLANG.lines().next(),
            Some("// cera:msl-postpass=v1"),
            "the post-pass ran but at an unexpected marker version; update this test"
        );
        // The three conditions the pass exists to impose.
        for needed in [
            "threadgroup char* shmem_p_0 [[threadgroup(0)]]",
            "threadgroup const half* pa_1 = pa_0 + ",
            "#pragma unroll(",
        ] {
            assert!(
                shaders::GEMM_Q8_0_SLANG.contains(needed),
                "post-processed MSL is missing {needed:?}"
            );
        }
        // Condition 2 asserted at the loads themselves, not just at the
        // declaration of the pointer walk. Declaring `pa_1` while every load
        // still reads `(_sa)[...]` is the half-patched shape the pass is
        // required never to emit, and it is invisible to the check above.
        for load in [
            "_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>(pa_0",
            "_slang_simdgroup_load<simdgroup_matrix<float, int(8), int(8)>>(pb_0",
        ] {
            assert!(
                shaders::GEMM_Q8_0_SLANG.contains(load),
                "post-processed MSL has the pointer walk but no load consuming it: {load:?}"
            );
        }
    }

    /// The generated MSL must keep Metal's two-stage simd reduction, not fall
    /// back to the portable tree. This is the whole premise of
    /// `__target_switch`, and it is a silent failure otherwise: the tree is
    /// correct, so every test above would still pass while the kernel quietly
    /// got slower. Asserted on the shader text, since there is no runtime way
    /// to observe which branch survived.
    #[test]
    fn generated_msl_keeps_simd_reduction() {
        let src = shaders::SOFTMAX;
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
            shaders::PER_HEAD_RMSNORM.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// layernorm_batch's two reductions must stay simd on Metal; a silent tree
    /// fall-through is correct but slow and invisible to the numeric tests.
    #[test]
    fn generated_layernorm_batch_keeps_simd_sum() {
        assert!(
            shaders::LAYERNORM_BATCH.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// Both rmsnorm_batch entry points must keep the metal simd reduction; a
    /// silent tree fall-through is correct but slow and invisible to the tests.
    #[test]
    fn generated_rmsnorm_batch_keeps_simd_sum() {
        assert!(
            shaders::RMSNORM_BATCH.contains("simd_sum"),
            "generated MSL lost simd_sum; __target_switch selected the portable tree"
        );
    }

    /// argmax_f32's metal branch reduces the (value, index) pair with
    /// `simd_shuffle_down`; a silent fall-through to the tree is correct but slow
    /// and invisible to the numeric test.
    #[test]
    fn generated_argmax_keeps_simd_shuffle_down() {
        assert!(
            shaders::ARGMAX_F32.contains("simd_shuffle_down"),
            "generated MSL lost simd_shuffle_down; __target_switch selected the portable tree"
        );
    }

    /// rmsnorm's metal branch keeps the simd reduction; a silent tree
    /// fall-through is correct but slow and invisible to the numeric test.
    #[test]
    fn generated_rmsnorm_keeps_simd_sum() {
           fn run_exp_polar(ctx: &MetalContext, spectrum: &[f32], n_frames: u32, bins: u32) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::EXP_POLAR, "exp_polar")
            .expect("compile generated MSL");
        let spec = ctx.upload_f32(spectrum);
        let out = ctx.create_buffer((spectrum.len() * 4) as u64);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n_frames, bins]));
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&spec), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_buffer(2, Some(&params), 0);
        let groups = (n_frames * bins).div_ceil(256);
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
        ctx.read_f32(&out, spectrum.len())
    }

    #[test]
    fn exp_polar_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let (n_frames, bins) = (5usize, 641usize);
        let spec = exp_polar_input(n_frames, bins);
        let got = run_exp_polar(&ctx, &spec, n_frames as u32, bins as u32);
        let want = exp_polar_ref(&spec, n_frames, bins);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-4,
            "msl exp_polar worst abs err {worst:.3e} > 1e-4"
        );
    }

    fn run_overlap_add(
        ctx: &MetalContext,
        time_domain: &[f32],
        hann: &[f32],
        n_frames: u32,
        n_fft: u32,
        hop: u32,
    ) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::OVERLAP_ADD, "overlap_add")
            .expect("compile generated MSL");
        let td = ctx.upload_f32(time_domain);
        let hann_buf = ctx.upload_f32(hann);
        let total = (n_frames * hop) as usize;
        let out = ctx.create_buffer((total * 4) as u64);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n_frames, n_fft, hop, 0u32]));
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&td), 0);
        enc.set_buffer(1, Some(&hann_buf), 0);
        enc.set_buffer(2, Some(&out), 0);
        enc.set_buffer(3, Some(&params), 0);
        let groups = (n_frames * hop).div_ceil(256);
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
        ctx.read_f32(&out, total)
    }

    #[test]
    fn overlap_add_slang_matches_reference() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let (n_frames, n_fft, hop) = (8usize, 1280usize, 320usize);
        let td = overlap_add_input(n_frames, n_fft);
        let hann = cera::model::audio_decoder::build_hann(n_fft);
        let got = run_overlap_add(&ctx, &td, &hann, n_frames as u32, n_fft as u32, hop as u32);
        let want = overlap_add_ref(&td, &hann, n_frames, n_fft, hop);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-3,
            "msl overlap_add worst abs err {worst:.3e} > 1e-3"
        );
    }

    fn run_moe_route(ctx: &MetalContext, logits: &[f32], bias: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let pipeline = ctx
            .create_pipeline(shaders::MOE_ROUTE, "moe_route")
            .expect("compile generated MSL");
        let entries = MOE_N_TOKENS * MOE_N_USED;
        let logits_buf = ctx.upload_f32(logits);
        let bias_buf = ctx.upload_f32(bias);
        let sel_expert = ctx.create_buffer((entries * 4) as u64);
        let sel_weight = ctx.create_buffer((entries * 4) as u64);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[
            MOE_N_EXPERT as u32,
            MOE_N_USED as u32,
            MOE_N_TOKENS as u32,
            0,
        ]));
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&logits_buf), 0);
        enc.set_buffer(1, Some(&bias_buf), 0);
        enc.set_buffer(2, Some(&sel_expert), 0);
        enc.set_buffer(3, Some(&sel_weight), 0);
        enc.set_buffer(4, Some(&params), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: MOE_N_TOKENS as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        (
            ctx.read_u32(&sel_expert, entries),
            ctx.read_f32(&sel_weight, entries),
        )
    }

    /// The routing rule, ties and all.
    ///
    /// Expert *ids* are compared exactly, not within a tolerance: a flipped
    /// selection is a different expert's weights, which no epsilon covers. The
    /// weights then get the ordinary float tolerance.
    #[test]
    fn moe_route_matches_cpu_selection() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let (logits, bias) = moe_route_inputs();
        let (want_ids, want_w) = moe_route_ref(&logits, &bias);
        let (got_ids, got_w) = run_moe_route(&ctx, &logits, &bias);
        assert_eq!(got_ids, want_ids, "msl moe_route picked different experts");
        assert_close("msl moe_route weights", &got_w, &want_w, 1e-5);
    }

    fn run_moe_gemv(
        ctx: &MetalContext,
        packed: &[u8],
        x: &[f32],
        sel: &[u32],
        n_entries: usize,
        x_by_entry: bool,
    ) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::MOE_GEMV_Q4_0, "moe_gemv_q4_0")
            .expect("compile generated MSL");
        // The kernel reads up to a whole `uint` past the last block byte when a
        // block start is not word-aligned, exactly as `gemv_q4_0` does; pad so
        // that read stays inside the allocation.
        let mut w_bytes = packed.to_vec();
        w_bytes.extend_from_slice(&[0u8; 8]);
        let w_buf = ctx.upload_bytes(&w_bytes);
        let x_buf = ctx.upload_f32(x);
        let sel_buf = ctx.upload_bytes(bytemuck::cast_slice(sel));
        let y_buf = ctx.create_buffer((n_entries * MOE_M * 4) as u64);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&moe_gemv_params(
            n_entries, x_by_entry,
        )));
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_buffer(3, Some(&sel_buf), 0);
        enc.set_buffer(4, Some(&params), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: MOE_M as u64,
                height: n_entries as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        ctx.read_f32(&y_buf, n_entries * MOE_M)
    }

    /// The gate/up shape: every slot of a token contracts against the *same*
    /// activation row, so only the expert slice differs between them.
    ///
    /// The selection deliberately repeats an expert across tokens and gives one
    /// token two different experts, so a kernel that ignored `sel_expert` and
    /// used `entry % n_expert` (or expert 0 throughout) disagrees.
    #[test]
    fn moe_gemv_by_token_matches_cpu() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let entries = MOE_N_TOKENS * MOE_N_USED;
        let packed = q4_0_pack(&gemm_data(MOE_N_EXPERT * MOE_M * MOE_K, 11));
        let w = q4_0_dequant(&packed);
        let x = gemm_data(MOE_N_TOKENS * MOE_K, 12);
        let sel = moe_sel(entries);
        let (want, mag) = moe_gemv_ref(&w, &x, &sel, entries, false);
        let got = run_moe_gemv(&ctx, &packed, &x, &sel, entries, false);
        assert_gemm("msl moe_gemv_q4_0 by-token", &got, &want, &mag, 1e-5);
    }

    /// The down-projection shape: each entry contracts against its own row, so
    /// swapping `x_by_entry` collapses a token's slots onto one input.
    #[test]
    fn moe_gemv_by_entry_matches_cpu() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let entries = MOE_N_TOKENS * MOE_N_USED;
        let packed = q4_0_pack(&gemm_data(MOE_N_EXPERT * MOE_M * MOE_K, 13));
        let w = q4_0_dequant(&packed);
        let x = gemm_data(entries * MOE_K, 14);
        let sel = moe_sel(entries);
        let (want, mag) = moe_gemv_ref(&w, &x, &sel, entries, true);
        let got = run_moe_gemv(&ctx, &packed, &x, &sel, entries, true);
        assert_gemm("msl moe_gemv_q4_0 by-entry", &got, &want, &mag, 1e-5);
    }

    fn run_moe_combine(
        ctx: &MetalContext,
        z: &[f32],
        weights: &[f32],
        out: &[f32],
        accumulate: bool,
        hidden: usize,
    ) -> Vec<f32> {
        let pipeline = ctx
            .create_pipeline(shaders::MOE_COMBINE, "moe_combine")
            .expect("compile generated MSL");
        let z_buf = ctx.upload_f32(z);
        let w_buf = ctx.upload_f32(weights);
        let out_buf = ctx.upload_f32(out);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[
            hidden as u32,
            MOE_N_USED as u32,
            MOE_N_TOKENS as u32,
            u32::from(accumulate),
        ]));

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&z_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_buffer(3, Some(&params), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: hidden.div_ceil(256) as u64,
                height: MOE_N_TOKENS as u64,
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

        ctx.read_f32(&out_buf, MOE_N_TOKENS * hidden)
    }

    /// Both output conventions, because the two callers disagree: decode adds
    /// into the residual stream, batched prefill overwrites a scratch buffer the
    /// next layer's fused residual add consumes. The pre-seeded `out` is
    /// non-zero so an `accumulate = false` that silently added would fail.
    #[test]
    fn moe_combine_matches_cpu() {
        let Some(ctx) = common::metal_context() else {
            return;
        };
        let entries = MOE_N_TOKENS * MOE_N_USED;
        let weights = gemm_data(entries, 16)
            .iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>();
        for &hidden in MOE_COMBINE_WIDTHS {
            let z = gemm_data(entries * hidden, 15);
            let seed_out = gemm_data(MOE_N_TOKENS * hidden, 17);
            for accumulate in [false, true] {
                let want = moe_combine_ref(&z, &weights, &seed_out, accumulate, hidden);
                let got = run_moe_combine(&ctx, &z, &weights, &seed_out, accumulate, hidden);
                assert_close(
                    &format!("msl moe_combine hidden={hidden} accumulate={accumulate}"),
                    &got,
                    &want,
                    1e-5,
                );
            }
        }
    }       &want,
                    1e-5,
                );
            }
        }
>>>>>>> 13e2a0f (feat(lfm2moe): run the routed FFN on the Metal and wgpu backends)
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
        ("softmax", cera::backend::wgpu::shaders::SOFTMAX),
        ("gemm_q8_0", cera::backend::wgpu::shaders::GEMM_Q8_0_SLANG),
        (
            "per_head_rmsnorm",
            cera::backend::wgpu::shaders::PER_HEAD_RMSNORM,
        ),
        (
            "layernorm_batch",
            cera::backend::wgpu::shaders::LAYERNORM_BATCH,
        ),
        ("rmsnorm_batch", cera::backend::wgpu::shaders::RMSNORM_BATCH),
        ("argmax_f32", cera::backend::wgpu::shaders::ARGMAX_F32),
        ("rmsnorm", cera::backend::wgpu::shaders::RMSNORM),
        ("conv1d", cera::backend::wgpu::shaders::CONV1D),
        ("conv1d_fused", cera::backend::wgpu::shaders::CONV1D_FUSED),
        (
            "conv1d_fused_batch",
            cera::backend::wgpu::shaders::CONV1D_FUSED_BATCH,
        ),
    ] {
        assert!(
            !src.contains("subgroup"),
            "generated WGSL for {name} uses a subgroup op, but cera does not enable Features::SUBGROUP"
        );
    }
    // The audio tier goes through the shared lookup rather than a fourth copy of
    // its name list. `audio_xl_attention` is the one that can actually regress
    // here: its softmax reduction is a `__target_switch` whose metal arm uses
    // `WaveActiveMax`/`WaveActiveSum`, so a failed branch elimination would put a
    // wave intrinsic in the WGSL.
    for name in AUDIO_KERNELS {
        let Some(src) = audio_kernel_sources(name).0 else {
            continue;
        };
        assert!(
            !src.contains("subgroup"),
            "generated WGSL for {name} uses a subgroup op, but cera does not enable Features::SUBGROUP"
        );
    }
}

/// The audio kernels' WGSL must not require `enable f16` either: cera requests
/// `SHADER_F16` only when the adapter reports it, so an f16 type reaching this
/// emission fails pipeline creation on the devices that lack it. Same constraint
/// `generated_gemm_wgsl_needs_no_f16` pins for the GEMM, checked here because
/// these are the sources the wgpu audio encoder will build pipelines from.
#[cfg(feature = "gpu")]
#[test]
fn generated_audio_wgsl_needs_no_f16() {
    for name in AUDIO_KERNELS {
        let Some(src) = audio_kernel_sources(name).0 else {
            continue;
        };
        assert!(
            !src.contains("enable f16"),
            "generated WGSL for {name} requires f16; cera enables SHADER_F16 only when \
             the adapter reports it, so this would fail pipeline creation elsewhere"
        );
    }
}

/// Every LFM2A audio-encoder kernel, by `.slang` basename.
///
/// One list, consumed by all three checks below. It used to be written out per
/// check, which meant a seventh kernel had to be added in three places and the
/// checks failed open if it was added to fewer.
const AUDIO_KERNELS: &[&str] = &[
    "activations",
    "conv2d_direct",
    "transpose_blocked",
    "glu_split",
    "chan_affine_silu",
    "audio_xl_attention",
    "stft_frame",
    "power_spec",
    "mel_project",
    "mel_norm",
];

/// The `(wgsl, msl)` sources for one audio kernel, `None` where the feature is
/// off.
///
/// A single lookup rather than a `match` per call site: the earlier form ended
/// in a `_ =>` catch-all, so a seventh kernel added to the caller's table would
/// have silently re-tested `audio_xl_attention` and passed. Panicking on an
/// unknown name makes that a failure instead.
fn audio_kernel_sources(name: &str) -> (Option<&'static str>, Option<&'static str>) {
    #[cfg(feature = "gpu")]
    let wgsl = Some(match name {
        "activations" => cera::backend::wgpu::shaders::ACTIVATIONS,
        "conv2d_direct" => cera::backend::wgpu::shaders::CONV2D_DIRECT,
        "transpose_blocked" => cera::backend::wgpu::shaders::TRANSPOSE_BLOCKED,
        "glu_split" => cera::backend::wgpu::shaders::GLU_SPLIT,
        "chan_affine_silu" => cera::backend::wgpu::shaders::CHAN_AFFINE_SILU,
        "audio_xl_attention" => cera::backend::wgpu::shaders::AUDIO_XL_ATTENTION,
        "stft_frame" => cera::backend::wgpu::shaders::STFT_FRAME,
        "power_spec" => cera::backend::wgpu::shaders::POWER_SPEC,
        "mel_project" => cera::backend::wgpu::shaders::MEL_PROJECT,
        "mel_norm" => cera::backend::wgpu::shaders::MEL_NORM,
        other => panic!("no WGSL source registered for audio kernel {other}"),
    });
    #[cfg(not(feature = "gpu"))]
    let wgsl = None::<&'static str>;

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    let msl = Some(match name {
        "activations" => cera::backend::metal::shaders::ACTIVATIONS,
        "conv2d_direct" => cera::backend::metal::shaders::CONV2D_DIRECT,
        "transpose_blocked" => cera::backend::metal::shaders::TRANSPOSE_BLOCKED,
        "glu_split" => cera::backend::metal::shaders::GLU_SPLIT,
        "chan_affine_silu" => cera::backend::metal::shaders::CHAN_AFFINE_SILU,
        "audio_xl_attention" => cera::backend::metal::shaders::AUDIO_XL_ATTENTION,
        "stft_frame" => cera::backend::metal::shaders::STFT_FRAME,
        "power_spec" => cera::backend::metal::shaders::POWER_SPEC,
        "mel_project" => cera::backend::metal::shaders::MEL_PROJECT,
        "mel_norm" => cera::backend::metal::shaders::MEL_NORM,
        other => panic!("no MSL source registered for audio kernel {other}"),
    });
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    let msl = None::<&'static str>;

    (wgsl, msl)
}

/// Every audio kernel's entry points must survive into each target this build
/// actually emits.
///
/// These kernels ship in this change but only the Metal backend drives them; the
/// WGSL halves sit inert until the wgpu audio encoder lands. An entry point
/// silently missing from one target would therefore surface as a pipeline
/// failure in a later change rather than here, which is exactly the drift this
/// suite exists to prevent.
///
/// "Each target this build emits" and not "both": the sources are cfg'd, so a
/// `--features metal` build has no WGSL to look at and a featureless one has
/// neither. The gate below is what the wording change is really about. Without
/// it this test ran in every configuration, and the featureless one CI runs
/// twice (`cargo test --workspace` and `-p cera --no-default-features`) walked
/// all six kernels, looked at nothing and reported a pass. Skipped in a build
/// that emits no sources is honest; green there is not.
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
#[test]
fn audio_kernel_entry_points_reach_every_enabled_target() {
    // Only `activations` exposes entry points that differ from its basename; the
    // rest are looked up by name, so the list stays derived from AUDIO_KERNELS.
    let cases: Vec<(&str, Vec<&str>)> = AUDIO_KERNELS
        .iter()
        .map(|&name| match name {
            "activations" => (
                name,
                vec!["relu_inplace", "silu_inplace", "gelu_erf_inplace"],
            ),
            _ => (name, vec![name]),
        })
        .collect();
    let checked = cases
        .iter()
        .map(|(name, entries)| {
            let (wgsl, msl) = audio_kernel_sources(name);
            entries
                .iter()
                .map(|e| {
                    if let Some(src) = wgsl {
                        assert!(src.contains(e), "{name}.wgsl is missing entry point {e}");
                    }
                    if let Some(src) = msl {
                        assert!(src.contains(e), "{name}.metal is missing entry point {e}");
                    }
                    usize::from(wgsl.is_some()) + usize::from(msl.is_some())
                })
                .sum::<usize>()
        })
        .sum::<usize>();

    // Belt and braces behind the cfg gate, and not the same check restated: the
    // gate says a source table should exist, this says entry points were actually
    // compared. It is what still catches an emptied `AUDIO_KERNELS` or an entry
    // list that lost its names, neither of which the cfg can see.
    assert!(
        checked > 0,
        "checked no entry points at all, so this test proves nothing: \
         AUDIO_KERNELS or the per-kernel entry lists are empty"
    );
}

/// The conv tier is the first clean single-body port since Phase 1a: one body,
/// no `__target_switch`, and all three kernels on the same five-slot binding
/// contract. That only holds because the Metal `conv1d_fused` twin was first
/// consolidated onto the packed `proj` buffer; its old signature took x, b and c
/// as three separate buffers and ran to `buffer(6)`. Pin the emitted binding
/// count so reintroducing a per-target split cannot pass silently.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn generated_conv_msl_binds_five_buffers() {
    for (name, src) in [
        ("conv1d", cera::backend::metal::shaders::CONV1D),
        ("conv1d_fused", cera::backend::metal::shaders::CONV1D_FUSED),
        (
            "conv1d_fused_batch",
            cera::backend::metal::shaders::CONV1D_FUSED_BATCH,
        ),
    ] {
        for i in 0..5 {
            assert!(
                src.contains(&format!("[[buffer({i})]]")),
                "generated MSL for {name} is missing buffer({i})"
            );
        }
        assert!(
            !src.contains("[[buffer(5)]]"),
            "generated MSL for {name} binds more than the five conv slots"
        );
    }
}

/// WGSL half of [`generated_conv_msl_binds_five_buffers`].
#[cfg(feature = "gpu")]
#[test]
fn generated_conv_wgsl_binds_five_slots() {
    for (name, src) in [
        ("conv1d", cera::backend::wgpu::shaders::CONV1D),
        ("conv1d_fused", cera::backend::wgpu::shaders::CONV1D_FUSED),
        (
            "conv1d_fused_batch",
            cera::backend::wgpu::shaders::CONV1D_FUSED_BATCH,
        ),
    ] {
        for i in 0..5 {
            assert!(
                src.contains(&format!("@binding({i})")),
                "generated WGSL for {name} is missing binding {i}"
            );
        }
        assert!(
            !src.contains("@binding(5)"),
            "generated WGSL for {name} binds more than the five conv slots"
        );
    }
}

/// `conv1d_fused_batch` gets its speed from having all five loops over its
/// `w_local[4]` / `rb[3]` state written with a literal trip count and
/// `[ForceUnroll]`. Losing that unroll measured 0.72x against the handwritten
/// kernel, which no numeric test can see: the kernel stays bit-identical, just
/// slower.
///
/// This asserts the unroll, which is the part that regressed, not register
/// residency as such: the emission still indexes `w_local[d_conv]` and
/// `rb[d_conv - 1]` by a runtime value, and predicating those away to make every
/// index constant measured slightly slower, so they stay.
///
/// Assert it structurally rather than by name: after unrolling, the only loop
/// left in either emission is the per-token one. Written the compound way
/// (`k < d_conv && k < 3`), Slang lowers `&&` to a branchy short-circuit that
/// does not unroll, and all five reappear.
#[test]
fn generated_conv_batch_unrolls_its_register_loops() {
    // Tracks that at least one arm compiled in. Both are cfg-gated, and with
    // `--features metal` on a non-Apple target neither would, leaving the sole
    // guard for the 0.72x regression passing while asserting nothing.
    let mut checked = 0usize;
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        checked += 1;
        let msl = cera::backend::metal::shaders::CONV1D_FUSED_BATCH;
        let loops = msl.matches("for(;;)").count();
        assert_eq!(
            loops, 1,
            "generated MSL for conv1d_fused_batch has {loops} loops, expected only the \
             per-token one; the register-array loops lost their unroll"
        );
    }
    #[cfg(feature = "gpu")]
    {
        checked += 1;
        let wgsl = cera::backend::wgpu::shaders::CONV1D_FUSED_BATCH;
        let loops = wgsl.matches("for(;;)").count() + wgsl.matches("loop {").count();
        assert_eq!(
            loops, 1,
            "generated WGSL for conv1d_fused_batch has {loops} loops, expected only the \
             per-token one; the register-array loops lost their unroll"
        );
    }
    assert!(
        checked > 0,
        "neither emission was checked; this test cannot guard the unroll in this \
         feature/target combination"
    );
}

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
    ] {
        assert!(
            !src.contains("subgroup"),
            "generated WGSL for {name} uses a subgroup op, but cera does not enable Features::SUBGROUP"
        );
    }
}

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
        "probe no longer requires f16; if the operands became f32 this test and the          warning it guards can go"
    );
}

/// The generated WGSL must **not** contain subgroup ops: cera never requests
/// `wgpu::Features::SUBGROUP`, so a wave intrinsic leaking into this target
/// fails pipeline creation on every device. Text-level because the failure
/// would otherwise surface as an opaque validation error far from its cause.
#[cfg(feature = "gpu")]
#[test]
fn generated_wgsl_has_no_subgroup_ops() {
    let src = cera::backend::wgpu::shaders::SOFTMAX_SLANG;
    assert!(
        !src.contains("subgroup"),
        "generated WGSL uses a subgroup op, but cera does not enable Features::SUBGROUP"
    );
}

//! Shader-only oracle test for the wgpu `kv_shift` kernel. Verifies the GPU
//! RoPE-delta math against the CPU `apply_rope_to_head` / `apply_rope_norm_to_head`
//! references, **without** loading any model weights or running attention. The
//! wgpu counterpart of `metal_kv_shift_oracle.rs`.
//!
//! ## What is being tested
//!
//! 1. Initialize a small synthetic K cache `[seq_len × kv_dim]` in f32 GPU
//!    memory by writing, for each absolute position `t`,
//!    `apply_rope_*_to_head(initial, t)` — i.e. the cache holds the output of
//!    the forward-time RoPE (the wgpu forward path RoPEs K before caching it).
//! 2. Dispatch `kv_shift` with `delta = -shift`. The kernel reads each retained
//!    cell at OLD position `t_old = n_keep + t_off + shift`, applies `R(-shift)`,
//!    and writes to scratch at `t_off`.
//! 3. Read scratch back and compare each cell to the CPU oracle:
//!    `apply_rope_*_to_head(initial, n_keep + t_off)` — the value the cell would
//!    have if RoPE were re-run directly for the cell's NEW position. A correct
//!    shift leaves each retained cell numerically equal to a fresh rotation at
//!    the new position.
//!
//! Both pair layouts are covered: NeoX (`rope_type == 0`, split-halves) and
//! NORM (`rope_type == 1`, interleaved). The NORM case also exercises the
//! Llama-3 `freq_factors` divide.
//!
//! ## Tolerance
//!
//! The wgpu KV cache is **f32** (Metal's is f16), so there is no half-precision
//! round-trip floor here — only the difference between `cos/sin` evaluated at
//! `pos` directly vs. at `pos+shift` then composed with `-shift`. `5e-5` is
//! comfortably above that reassociation noise while still catching any sign
//! error, dim-pair swap, or magnitude drift.
//!
//! Gating: needs only a GPU adapter (wgpu → Metal/Vulkan/DX). Runs whenever the
//! `gpu` feature is on; skips cleanly if no adapter is available.

#![cfg(feature = "gpu")]

use cera::backend::cpu::{apply_rope_norm_to_head, apply_rope_to_head};
use cera::backend::wgpu::{GpuContext, KvShiftParams, shaders};

// Dimensions are chosen so the dispatch spans MORE THAN ONE workgroup in X:
// `total = RETAINED * N_KV_HEADS * (HEAD_DIM/2) = 19 * 2 * 8 = 304 > 256`, so the
// grid rounds up to 2 workgroups → `(2, 1, 1)`. That exercises the kernel's
// cross-workgroup `get_wid` index recovery along X (`wid.x == 1` does real work).
// It does NOT exercise the Y spill (`wid.y > 0`), which only occurs once the
// workgroup count exceeds MAX_WG (65535) — a ~16.7M-thread dispatch no unit test
// can afford to allocate a buffer for. The host-side grid sizing for that Y-spill
// path (the exact logic the 2-D flatten added) is covered directly, without a
// GPU, by the `kv_shift_workgroups` unit tests in `backend::wgpu`.
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const SEQ_LEN: usize = 24;
const N_KEEP: usize = 2;
const SHIFT: usize = 3;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const NEW_SEQ_LEN: usize = SEQ_LEN - SHIFT;
const RETAINED: usize = NEW_SEQ_LEN - N_KEEP;
const FREQ_BASE: f32 = 10_000.0;
const TOL: f32 = 5e-5;

/// Per-head initial K vector. The `+ h*0.5` term makes every KV head hold
/// DISTINCT data, so a kernel regression in the read-side head offset
/// (`head_off = h*head_dim`) is observable — with identical per-head data a
/// dropped/wrong `h` would still read correct values and pass silently.
fn head_initial(h: usize) -> Vec<f32> {
    (0..HEAD_DIM)
        .map(|i| (i as f32 + 1.0) * 0.1 + h as f32 * 0.5)
        .collect()
}

/// Run the `kv_shift` kernel over a synthetic, RoPE'd K cache and return the
/// retained-region scratch contents (`RETAINED × KV_DIM` floats). `rope_type`
/// selects the pair layout; `freq_factors` (NORM only) is divided into each
/// pair's angle when `Some`.
fn run_kv_shift(
    ctx: &GpuContext,
    rope_type: u32,
    freq_factors: Option<&[f32]>,
    populate: impl Fn(&mut [f32], usize),
) -> Vec<f32> {
    // Build the K cache: cell at absolute position `t` holds the forward-RoPE'd
    // head for that position. Each head starts from distinct data (see
    // `head_initial`) so head addressing is actually under test.
    let mut k_cache = vec![0.0f32; SEQ_LEN * KV_DIM];
    for t in 0..SEQ_LEN {
        for h in 0..N_KV_HEADS {
            let mut rotated = head_initial(h);
            populate(&mut rotated, t);
            let off = t * KV_DIM + h * HEAD_DIM;
            k_cache[off..off + HEAD_DIM].copy_from_slice(&rotated);
        }
    }

    let k_buf = ctx.upload_f32(&k_cache, "k_cache");
    let scratch = ctx.create_storage_rw((RETAINED * KV_DIM * 4) as u64, "scratch");
    let ff = freq_factors.unwrap_or(&[1.0]);
    let ff_buf = ctx.upload_f32(ff, "freq_factors");

    let params = KvShiftParams {
        n_keep: N_KEEP as u32,
        shift: SHIFT as u32,
        retained: RETAINED as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        freq_base_bits: FREQ_BASE.to_bits(),
        rope_type,
        has_freq_factors: u32::from(freq_factors.is_some()),
    };
    let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params.to_u32_array()), "params");

    let pipeline = ctx.create_pipeline(shaders::KV_SHIFT, "kv_shift", "kv_shift");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kv_shift"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scratch.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ff_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kv_shift"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // 2-D grid via the shared `dispatch_dims` helper — identical to
        // production (`encode_kv_shift_layers`) so the test drives the kernel
        // exactly as the engine does, and the grid-sizing logic lives in one place.
        let (gx, gy, gz) = params.dispatch_dims();
        pass.dispatch_workgroups(gx, gy, gz);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.download_f32(&scratch, RETAINED * KV_DIM)
}

/// Assert each retained cell matches a fresh rotation at its NEW position.
fn assert_matches_oracle(scratch: &[f32], oracle: impl Fn(&mut [f32], usize)) {
    let mut max_diff = 0.0f32;
    for t_off in 0..RETAINED {
        let t_new = N_KEEP + t_off;
        for h in 0..N_KV_HEADS {
            let mut expected = head_initial(h);
            oracle(&mut expected, t_new);
            let off = t_off * KV_DIM + h * HEAD_DIM;
            let got = &scratch[off..off + HEAD_DIM];
            for i in 0..HEAD_DIM {
                let diff = (got[i] - expected[i]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < TOL,
                    "mismatch at t_off={t_off} (t_new={t_new}) h={h} i={i}: \
                     got={} expected={} diff={diff} (tol={TOL})",
                    got[i],
                    expected[i],
                );
            }
        }
    }
    eprintln!("max abs diff = {max_diff:.3e} (tol = {TOL:.0e})");
}

/// Acquire a GPU context, or signal skip. When `CERA_REQUIRE_GPU` is set (the CI
/// lavapipe job sets it), a missing adapter is a hard FAILURE rather than a
/// silent green skip — mirrors the `CERA_REQUIRE_SIMD` gate so a CI job that is
/// supposed to run the kernel cannot pass by skipping it.
fn gpu_ctx_or_skip() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            let required = std::env::var("CERA_REQUIRE_GPU").unwrap_or_default();
            assert!(
                required.is_empty(),
                "CERA_REQUIRE_GPU is set but no GPU adapter is available: {e}"
            );
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

#[test]
fn kv_shift_neox_matches_cpu_rope_oracle() {
    let Some(ctx) = gpu_ctx_or_skip() else {
        return;
    };
    let scratch = run_kv_shift(&ctx, 0, None, |head, t| {
        apply_rope_to_head(head, t, HEAD_DIM, FREQ_BASE);
    });
    assert_matches_oracle(&scratch, |head, t| {
        apply_rope_to_head(head, t, HEAD_DIM, FREQ_BASE);
    });
}

#[test]
fn kv_shift_norm_matches_cpu_rope_oracle() {
    let Some(ctx) = gpu_ctx_or_skip() else {
        return;
    };
    let scratch = run_kv_shift(&ctx, 1, None, |head, t| {
        apply_rope_norm_to_head(head, t, HEAD_DIM, FREQ_BASE, None);
    });
    assert_matches_oracle(&scratch, |head, t| {
        apply_rope_norm_to_head(head, t, HEAD_DIM, FREQ_BASE, None);
    });
}

#[test]
fn kv_shift_norm_freq_factors_matches_cpu_rope_oracle() {
    let Some(ctx) = gpu_ctx_or_skip() else {
        return;
    };
    // Llama-3-style per-pair frequency factors (head_dim/2 = 8 values).
    let ff: Vec<f32> = vec![1.0, 1.5, 2.0, 4.0, 1.25, 1.75, 3.0, 5.0];
    let scratch = run_kv_shift(&ctx, 1, Some(&ff), |head, t| {
        apply_rope_norm_to_head(head, t, HEAD_DIM, FREQ_BASE, Some(&ff));
    });
    assert_matches_oracle(&scratch, |head, t| {
        apply_rope_norm_to_head(head, t, HEAD_DIM, FREQ_BASE, Some(&ff));
    });
}

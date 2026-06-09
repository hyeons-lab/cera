//! Shader-only oracle test for the Metal `kv_shift_k_to_scratch`
//! kernel. Verifies the GPU RoPE-delta math against the CPU
//! `apply_rope_to_head` reference, **without** loading any model
//! weights or running attention. Complements
//! `tests/shift_real_model.rs` (end-to-end smoke; "no NaN / no
//! panic") and `tests/n_keep_shift.rs` (CPU-only shift correctness)
//! by closing the gap on whether the Metal kernel itself produces
//! the right numerical output.
//!
//! ## What is being tested
//!
//! 1. Initialize a small synthetic K cache `[seq_len × kv_dim]` in
//!    f16 GPU memory by writing, for each absolute position `t`,
//!    `apply_rope_to_head(initial, t)` — i.e. the K cache holds the
//!    output of the forward-time RoPE.
//! 2. Dispatch `kv_shift_k_to_scratch` with `delta = -shift`. The
//!    kernel reads each retained cell at OLD position `t_old =
//!    n_keep + t_off + shift`, applies `R(-shift)`, and writes to
//!    scratch at `t_off`.
//! 3. Read the scratch buffer back and compare each cell to the
//!    CPU oracle: `apply_rope_to_head(initial, n_keep + t_off)` —
//!    i.e. the value the cell would have if we re-ran RoPE
//!    directly for the cell's NEW position. A correct shift kernel
//!    leaves each retained cell numerically equivalent to a fresh
//!    rotation at the new position, modulo the f16-storage round-
//!    trip that happens twice (once in step 1, once in step 3).
//!
//! ## Tolerance
//!
//! f16 has ~11 bits of mantissa precision, so a single store-then-
//! load round-trip introduces ~2^-10 ≈ 1e-3 relative error for
//! values bounded by ~1. The shader does TWO such round-trips
//! (input store at init, output store at write-back), plus the
//! difference between `cos/sin` evaluated at `pos` directly vs at
//! `pos+shift` then composed with `-shift`. We use an absolute
//! tolerance of `5e-3` — ample headroom over the round-trip floor
//! while still catching any sign error, dim-pair swap, or
//! significant magnitude drift.
//!
//! ## Why an `#[ignore]`-free test
//!
//! Unlike `shift_real_model.rs` (downloads ~210 MB) and
//! `attention_metal_parity.rs` (needs a real GGUF in
//! `~/.leap/models/`), this test only needs a Metal device. It runs
//! whenever the `metal` feature is on and the host is macOS — the
//! same gate the GPU code itself uses.

#![cfg(all(feature = "metal", target_os = "macos"))]

use cera::backend::cpu::apply_rope_to_head;
use cera::backend::metal::{MetalContext, shaders};
use half::f16;
use metal::{Buffer, MTLSize};

/// Pack an `f32` slice into the head positions of an f16 GPU buffer
/// at byte offset `dst_off`. `data.len()` floats are written as
/// `data.len() * 2` bytes.
fn write_f32_as_f16(buf: &Buffer, dst_off_bytes: usize, data: &[f32]) {
    let dst_ptr = unsafe { (buf.contents() as *mut u8).add(dst_off_bytes) as *mut f16 };
    for (i, &v) in data.iter().enumerate() {
        unsafe { dst_ptr.add(i).write(f16::from_f32(v)) };
    }
}

/// Read `count` f16 elements starting at byte offset `src_off` and
/// return as a freshly-allocated `Vec<f32>`. Mirrors the f16-read
/// path used by `MetalLfm2Model`'s attention kernels.
fn read_f16_as_f32(buf: &Buffer, src_off_bytes: usize, count: usize) -> Vec<f32> {
    let src_ptr = unsafe { (buf.contents() as *const u8).add(src_off_bytes) as *const f16 };
    (0..count)
        .map(|i| f32::from(unsafe { src_ptr.add(i).read() }))
        .collect()
}

#[test]
fn kv_shift_k_kernel_matches_cpu_rope_oracle() {
    let ctx = MetalContext::new().expect("Metal context");
    let pipeline = ctx
        .create_pipeline(shaders::KV_SHIFT, "kv_shift_k_to_scratch")
        .expect("compile kv_shift_k_to_scratch");

    // Synthetic shape — small enough to inspect on failure but
    // exercises every dim-pair (head_dim/2 = 4 angles) and both
    // KV heads. Choosing seq_len=8 lets us pick a non-trivial
    // n_keep + shift split with retained cells on both sides of
    // the boundary.
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const SEQ_LEN: usize = 8;
    const N_KEEP: usize = 2;
    const SHIFT: usize = 3;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
    const NEW_SEQ_LEN: usize = SEQ_LEN - SHIFT;
    const RETAINED: usize = NEW_SEQ_LEN - N_KEEP;
    const FREQ_BASE: f32 = 10_000.0;

    // A fixed, non-uniform per-head input. `apply_rope_to_head`
    // rotates pairs `(d, d + half_dim)`, so the asymmetric values
    // here force every dim-pair to do non-trivial work — masks
    // sign-bug regressions where (e.g.) the first half were always
    // zero.
    let initial: Vec<f32> = (0..HEAD_DIM).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Build the K cache: each cell at absolute position `t` stores
    // `apply_rope_to_head(initial, t, head_dim, freq_base)`. Stored
    // f16 — same format the production K cache uses, so the test's
    // round-trip drift matches what the shipped path sees.
    let k_cache_bytes = SEQ_LEN * KV_DIM * 2;
    let k_cache = ctx.create_buffer(k_cache_bytes as u64);
    for t in 0..SEQ_LEN {
        for h in 0..N_KV_HEADS {
            let mut rotated = initial.clone();
            apply_rope_to_head(&mut rotated, t, HEAD_DIM, FREQ_BASE);
            let head_off_elements = t * KV_DIM + h * HEAD_DIM;
            write_f32_as_f16(&k_cache, head_off_elements * 2, &rotated);
        }
    }

    // Scratch sized for the full retained region.
    let scratch_bytes = RETAINED * KV_DIM * 2;
    let scratch = ctx.create_buffer(scratch_bytes as u64);

    // Params struct must match the shader's `KParams` layout exactly.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct KParams {
        n_keep: u32,
        shift: u32,
        new_seq_len: u32,
        n_kv_heads: u32,
        head_dim: u32,
        freq_base_bits: u32,
        delta_pos: i32,
        _pad: u32,
    }
    let kparams = KParams {
        n_keep: N_KEEP as u32,
        shift: SHIFT as u32,
        new_seq_len: NEW_SEQ_LEN as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        freq_base_bits: FREQ_BASE.to_bits(),
        delta_pos: -(SHIFT as i32),
        _pad: 0,
    };

    let cmd_buf = ctx.queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&k_cache), 0);
    enc.set_buffer(1, Some(&scratch), 0);
    enc.set_bytes(
        2,
        std::mem::size_of::<KParams>() as u64,
        &kparams as *const _ as *const _,
    );
    let half_dim = HEAD_DIM / 2;
    let total = (RETAINED * N_KV_HEADS * half_dim) as u64;
    let groups = MTLSize {
        width: total.div_ceil(256),
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(groups, threads);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Compare each retained cell against the CPU oracle:
    // `apply_rope_to_head(initial, t_new=N_KEEP + t_off)`. The
    // shader composed `R(-shift)` with `R(t_old)` to get `R(t_new)`
    // — the oracle computes `R(t_new)` directly. They must agree
    // within the f16-storage round-trip floor.
    //
    // 1e-3 is the empirical worst-case observed locally; that's
    // tight to the f16 mantissa quantum (~9.8e-4 for the |x| ≤ 1
    // values here). Tightening it to 1e-3 means a future regression
    // that adds even one extra round-trip — or that drops a step
    // out of f32 — fails the assert. Loose tolerances would let
    // sign flips on small-magnitude pairs pass.
    const TOL: f32 = 1e-3;
    let scratch_f32 = read_f16_as_f32(&scratch, 0, RETAINED * KV_DIM);
    let mut max_diff: f32 = 0.0;
    for t_off in 0..RETAINED {
        let t_new = N_KEEP + t_off;
        for h in 0..N_KV_HEADS {
            let mut expected = initial.clone();
            apply_rope_to_head(&mut expected, t_new, HEAD_DIM, FREQ_BASE);
            let head_off = t_off * KV_DIM + h * HEAD_DIM;
            let got = &scratch_f32[head_off..head_off + HEAD_DIM];
            for i in 0..HEAD_DIM {
                let diff = (got[i] - expected[i]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < TOL,
                    "mismatch at t_off={t_off} (t_new={t_new}) h={h} i={i}: \
                     got={} expected={} diff={diff} (tol={TOL}; \
                     max-seen so far={max_diff})",
                    got[i],
                    expected[i],
                );
            }
        }
    }
    eprintln!("max abs diff = {max_diff:.3e} (tol = {TOL:.0e})");
}

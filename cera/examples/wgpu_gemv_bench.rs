//! Isolated wgpu **decode GEMV** microbenchmark.
//!
//! The GEMM twin (`wgpu_gemm_bench`) is what produced the prefill diagnosis in
//! PR #311: an end-to-end tok/s number cannot tell "this kernel is slow" from
//! "there are too many of them", and it cannot be swept. This is the decode
//! equivalent — it times one GEMV kernel on the real projection shapes and
//! reports **achieved bandwidth**.
//!
//! Bandwidth is the right *unit* — a GEMV streams weights, so the roofline is
//! bandwidth and GFLOP/s obscures how far off peak it is — but read the number
//! as "how much of the roofline is being used", **not** as "this kernel is
//! bandwidth-bound". A quantized GEMV here can be instruction-bound: unrolling
//! `gemv_q4_k`'s dequant loop was worth 1.42x with no change in bytes moved.
//! (The stronger claim this doc used to make — that q4_k took the *same wall
//! time* as the dense f32 GEMV while moving 7x fewer bytes — came from the
//! positional bug described below and does not hold — q4_k is several times
//! faster than the f32 GEMV at every shape measured since.)
//!
//! `BASELINE.md`'s "time tracks bytes" result (1.89x bytes -> 2.10x time) came
//! from an in-model Q8_0-vs-Q4_0 A/B, and does not generalize across the family.
//! It is also **currently unreconciled with this bench**, which makes q8_0 look
//! substantially further behind q4_0 per byte than the in-model A/B's ~10%.
//! Both cannot describe the same pair.
//!
//! **This harness has three times reported kernel differences that were its own,
//! and two of those reached `BASELINE.md` as published findings.**
//!
//! 1. A compute pass opened per iteration, charging a boundary to every
//!    measurement (fixed in #321; see `run`).
//! 2. A cell's number depending on its **position in the table** — whichever
//!    kernel ran first looked ~3x slow. Published as an "FFN per-dispatch floor"
//!    and as a q4_k-specific defect; neither was real (see the two-round loop in
//!    `main`).
//! 3. One blocking submit + `poll(Wait)` per measurement — a ~1.0-1.5 ms GPU
//!    round trip — amortised over `ITERS` and never subtracted, so a cell read
//!    `T + C/ITERS` (see the two-point timing in the cell body).
//!
//! All three are fixed. **Do not treat this bench's absolute numbers as
//! authoritative until its variance has been characterised** — differencing two
//! timings removes defect 3's bias at the cost of higher variance, and per-cell
//! results do not yet replicate tightly. The durable guidance:
//!
//! - Prefer an **A/B within one process**, ABBA-ordered, over comparing absolute
//!   numbers across runs. That is what the kernel-variant results in
//!   `BASELINE.md` rest on, and why they survived all three defects.
//! - Keep the **f32 control** in the table; it is what separates a harness
//!   artifact from a kernel property.
//! - Validate the instrument, not just the result: **permute the kernel order and
//!   sweep `ITERS`** — a real difference survives both, and those two checks are
//!   exactly what catch defects 2 and 3 respectively. Neither was found by reading
//!   this file: defect 2 surfaced when a kernel rewrite failed to move the number
//!   it targeted, defect 3 when the `ITERS` sweep was finally run during review.
//!
//! ```text
//! cargo run --release -p cera --features gpu --example wgpu_gemv_bench
//! ITERS=1000 cargo run --release -p cera --features gpu --example wgpu_gemv_bench
//! ```
//!
//! Cells that print `noise` had a two-point difference too small to resolve; raise
//! `ITERS`. Take medians of >=3 runs, and permute the kernel order to confirm.

use anyhow::{Context, ensure};
use cera::backend::wgpu::{DevicePollExt, GpuContext, shaders};
use cera::tensor::DType;

/// Reads a **positive** `u32` from the environment, falling back to `default`
/// only when the variable is absent.
///
/// A variable that is set but unparseable is a hard error rather than a silent
/// fall back. The shape knobs are self-correcting — the table prints the `m`/`k`
/// it actually ran — but **`ITERS` is echoed nowhere**, so a mistyped `ITERS=100O`
/// silently re-measures the default and an `ITERS` sweep comes back looking
/// ITERS-independent — which is exactly the check the two-point timing below has
/// to survive.
///
/// Zero is rejected because it is not meaningful at any of the call sites:
/// `ITERS=0` divides the two-point difference by zero, and `M=0` / `K=0` build
/// empty buffers.
fn env_u32(key: &str, default: u32) -> anyhow::Result<u32> {
    let Some(raw) = std::env::var_os(key) else {
        return Ok(default);
    };
    let raw = raw.to_string_lossy();
    let value: u32 = raw
        .trim()
        .parse()
        .with_context(|| format!("{key}={raw:?} is not a u32"))?;
    ensure!(value > 0, "{key} must be at least 1, got 0");
    Ok(value)
}

/// [`env_u32`] for the one `f64` knob, with the same fail-fast contract. The
/// footer does print the effective figure, so this is about not *accepting* a
/// mistyped `PEAK_GBS` rather than about it being invisible.
fn env_f64(key: &str, default: f64) -> anyhow::Result<f64> {
    let Some(raw) = std::env::var_os(key) else {
        return Ok(default);
    };
    let raw = raw.to_string_lossy();
    let value: f64 = raw
        .trim()
        .parse()
        .with_context(|| format!("{key}={raw:?} is not an f64"))?;
    ensure!(
        value.is_finite() && value > 0.0,
        "{key} must be a finite positive number, got {value}"
    );
    Ok(value)
}

/// One kernel under test.
struct Kernel {
    name: &'static str,
    dtype: DType,
    shader: &'static str,
    entry: &'static str,
    /// Must match the shader's `NR` / `ROWS_PER_WG`.
    rows_per_wg: u32,
}

/// Block geometry: `(elements_per_block, bytes_per_block)`.
fn block_geom(dtype: DType) -> (usize, usize) {
    match dtype {
        DType::Q4_0 => (32, 18),
        DType::Q8_0 => (32, 34),
        DType::Q4KM => (256, 144),
        DType::Q5KM => (256, 176),
        DType::Q6K => (256, 210),
        // Dense f32: the control, and the reason to keep it in the table. It is
        // the best-behaved kernel here, by a wide margin at the LM-head shape, so
        // when it moves with a quantized kernel the cause is the harness, not the
        // kernel — which is how the positional bug in `main` was caught.
        DType::F32 => (1, 4),
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Weight bytes of the right size and block alignment for `dtype`.
///
/// This measures **bandwidth**, so the decoded values are irrelevant — but the
/// bytes are not arbitrary: every byte is kept in `0x30..=0x37` so that whatever
/// pair of them a kernel reads as an f16 scale is a small normal positive
/// number. Random bytes would hit the f16 NaN/Inf encodings (exponent all ones),
/// and NaN propagating through the accumulators is a real risk of measuring
/// something other than the kernel.
fn synth_weights(dtype: DType, m: usize, k: usize) -> Vec<u8> {
    let (elems, bytes) = block_geom(dtype);
    assert_eq!(k % elems, 0, "k must be a multiple of the block size");
    let total = m * (k / elems) * bytes;
    let mut out: Vec<u8> = (0..total).map(|i| 0x30u8 + (i * 7 % 8) as u8).collect();
    // Pad to a whole `u32`. The shaders bind this as `array<u32>`, and the
    // 2-byte-aligned block sizes make the total a non-multiple of 4 for some
    // shapes — e.g. Q6_K (210 B) at an odd `m`. `upload_storage` already rounds
    // up to COPY_BUFFER_ALIGNMENT and zeroes the tail, so this is not fixing an
    // observed failure (m=3 k=256 runs fine). It is here so the benchmark does
    // not silently depend on that, given `M=`/`K=` invite arbitrary shapes.
    out.resize(total.next_multiple_of(4), 0);
    out
}

fn main() -> anyhow::Result<()> {
    let ctx = GpuContext::new()?;
    eprintln!("adapter: {} ({})", ctx.adapter_name, ctx.backend);

    // 200, not 50: two-point timing differences two measurements, so the kernel
    // work has to be large enough next to the fixed round trip for the difference
    // to resolve. At 50 several cells per run came back unresolvable.
    let iters = env_u32("ITERS", 200)?;
    // The second timing point. Derived once here rather than as `iters * 2` at
    // the call site, so the bound is checked where it is computed.
    let iters_2n = iters
        .checked_mul(2)
        .context("ITERS is too large: the second timing point runs 2x ITERS")?;

    // `M=..K=..` overrides the shape table with a single custom shape, for
    // sweeping one dimension while the other is held fixed.
    let custom = if std::env::var_os("M").is_some() || std::env::var_os("K").is_some() {
        let (m, k) = (env_u32("M", 2816)?, env_u32("K", 1024)?);
        // Checked here, before the run starts. `synth_weights` would otherwise
        // catch it partway through, but its message names only "the block size"
        // and not which one, so `K=1000` panics with an unexplained "left: 232".
        // 256 is the strictest block size in the table (Q4_K, Q5_K and Q6_K)
        // and a multiple of every other one.
        ensure!(
            k.is_multiple_of(256),
            "K must be a multiple of 256 (the Q4_K/Q5_K/Q6_K super-block), got {k}"
        );
        Some(vec![(m, k, "custom")])
    } else {
        None
    };

    // Real LFM2 projection shapes. `(m, k, label)`.
    let default_shapes: &[(u32, u32, &str)] = &[
        (2816, 1024, "ffn gate/up"),
        (1024, 2816, "ffn down"),
        (1024, 1024, "attn qkv/out"),
        (65536, 1024, "lm head"),
    ];
    let shapes: &[(u32, u32, &str)] = custom.as_deref().unwrap_or(default_shapes);

    let kernels = &[
        Kernel {
            name: "q4_k",
            dtype: DType::Q4KM,
            shader: shaders::GEMV_Q4_K,
            entry: "gemv_q4_k",
            rows_per_wg: 2,
        },
        Kernel {
            name: "q5_k",
            dtype: DType::Q5KM,
            shader: shaders::GEMV_Q5_K,
            entry: "gemv_q5_k",
            rows_per_wg: 2,
        },
        Kernel {
            name: "q4_0",
            dtype: DType::Q4_0,
            shader: shaders::GEMV_Q4_0_FAST,
            entry: "gemv_q4_0_fast",
            rows_per_wg: 4,
        },
        Kernel {
            name: "q8_0",
            dtype: DType::Q8_0,
            shader: shaders::GEMV_Q8_0,
            entry: "gemv_q8_0",
            rows_per_wg: 8,
        },
        Kernel {
            name: "f32",
            dtype: DType::F32,
            shader: shaders::GEMV_F32,
            entry: "gemv_f32",
            rows_per_wg: 8,
        },
        Kernel {
            name: "q6_k",
            dtype: DType::Q6K,
            shader: shaders::GEMV_Q6_K,
            entry: "gemv_q6_k",
            rows_per_wg: 2,
        },
    ];

    println!(
        "\n{:<6} {:<14} {:>6} {:>6} {:>10} {:>10} {:>9}",
        "kernel", "shape", "m", "k", "ms", "GB/s", "% peak*"
    );
    println!("{}", "-".repeat(70));

    // M1 Max unified memory. Only used for the "% peak" column; override for
    // other hardware.
    let peak_gbs = env_f64("PEAK_GBS", 400.0)?;

    // The whole table runs TWICE and only the second round is reported.
    //
    // A cell's number otherwise depends on where it sits in the table, and the
    // per-cell `run(2)` warmup does not fix that. Measured: `q6_k` at ffn gate/up
    // reports 0.0305 ms when its kernel runs last and 0.0952 ms when it runs
    // first, and reversing the kernel order moves the penalty to whatever is now
    // first — so it is positional, not a property of the kernel. q4_k was listed
    // first, so it wore that penalty in every table this bench printed before
    // this fix, which is how it came to be written up as ~3x slower than q4_0 at
    // identical bytes/element. It is not.
    //
    // A 750 ms global clock-ramp warmup was tried first and measurably did NOT
    // remove the effect, so the mechanism is not clock ramp and is still
    // unidentified. A discarded first pass sidesteps it without needing the
    // mechanism — by the time anything is reported, every cell has already run in
    // full. So do not replace this with a cheaper-looking global warmup; that was
    // measured and rejected.
    //
    // It is not fully fixed, only reduced: forward and reversed runs now agree
    // within ~3-5% for most kernels, but one per run still moves by up to ~19%.
    // The verification is a forward vs reversed kernel-order run — if a change
    // here makes those disagree by more than that, it made things worse.
    //
    // Separately: one pipeline per kernel, built once. Each cell used to compile
    // its own, so a 5x4 table cost 40 shader compiles across the two rounds and
    // every measurement leaned on `run(2)` to absorb a fresh compile — one more
    // uncontrolled per-cell variable in a harness whose residual positional
    // effect is still unexplained.
    let pipelines: Vec<wgpu::ComputePipeline> = kernels
        .iter()
        .map(|kern| ctx.create_pipeline(kern.shader, kern.entry, kern.entry))
        .collect();

    for round in 0..2 {
        for (kern, pipeline) in kernels.iter().zip(&pipelines) {
            for &(m, k, label) in shapes {
                let weight_bytes = synth_weights(kern.dtype, m as usize, k as usize);
                let x: Vec<f32> = (0..k)
                    .map(|i| ((i * 13 + 7) % 251) as f32 * 0.01 - 1.25)
                    .collect();

                let a_buf = ctx.upload_storage(&weight_bytes, "A");
                let x_buf = ctx.upload_f32(&x, "x");
                let y_buf = ctx.create_storage_rw(u64::from(m) * 4, "y");
                let params = [m, k, 0u32, 0u32];
                let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

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
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: y_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let groups = m.div_ceil(kern.rows_per_wg);
                // All iterations share ONE compute pass. A pass per iteration charged
                // a boundary to every measurement, which is why this bench used to
                // report q4_0 at 53 GB/s for the LM-head shape where #321 then
                // measured 113. Dispatches inside a pass still execute in order, so
                // the dependency chain is unchanged.
                //
                // Deliberately no flat per-boundary cost quoted here. An earlier
                // version of this comment asserted ~38 us, carried over from #318's
                // conv-block result, and that does not survive its own example: at
                // this shape #321 moved q4_0 from 53 to 113 GB/s, i.e. 37.7 MB in
                // 712 us against 334 us, so the boundary it removed was worth
                // ~378 us — an order of magnitude off 38. BASELINE.md retracts the
                // generalization directly: a boundary in the attention path measured
                // ~6 us. The cost is shape-dependent, not a constant of the machine.
                //
                // Both figures above are from the single-point timing this file has
                // since replaced, so treat them as the order-of-magnitude argument
                // they are and not as current measurements.
                //
                // Rotating several output buffers was also tried, on the theory that
                // identical repeated dispatches serialize write-after-write and turn
                // this into a latency measurement. It changed nothing, so WAW is not
                // what limits the small-m rows; not kept.
                //
                // That experiment was chasing "the FFN rows read far below what
                // production sustains", which was itself the positional bug the
                // two-round loop in `main` now fixes — so treat its premise, and the
                // ~13 GB/s FFN figures this comment used to quote, as withdrawn.
                let run = |count: u32| {
                    let mut enc = ctx.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        for _ in 0..count {
                            pass.dispatch_workgroups(groups, 1, 1);
                        }
                    }
                    ctx.queue.submit(Some(enc.finish()));
                    ctx.device.poll_wait();
                };

                // First touch of this cell's freshly uploaded buffers. It no longer
                // absorbs a shader compile — the pipelines are built once, above —
                // and it cannot fix table position or the fixed round-trip cost;
                // those are the two-round loop's and the two-point timing's jobs
                // respectively. Keep it for the buffer warm-up alone.
                run(2);

                // Two-point timing: `run(n)` and `run(2n)` differ by exactly `n`
                // dispatches, so the difference cancels every fixed per-measurement
                // cost. Each `run` ends in one blocking submit + `poll(Wait)`, and a
                // GPU round trip costs ~1.0-1.5 ms whatever it carries. Timing a
                // single `run(iters)` and dividing buries that whole round trip in
                // the per-iteration number.
                //
                // It is not a rounding error, and it is worst exactly where the
                // kernel is cheapest: `q6_k` at ffn gate/up measured 46.0 / 63.6 /
                // 79.3 GB/s at ITERS = 50 / 200 / 1000 under the old single-point
                // timing. Fitting `per_iter = T + C/iters` to that gives C ~ 1.16 ms,
                // i.e. one round trip. A table built that way is ITERS-dependent and
                // silently compresses cheap shapes against expensive ones — the same
                // artifact as the "per-dispatch floor" this file's history retracts.
                let t_n = {
                    let t0 = std::time::Instant::now();
                    run(iters);
                    t0.elapsed()
                };
                let t_2n = {
                    let t0 = std::time::Instant::now();
                    run(iters_2n);
                    t0.elapsed()
                };
                // Saturating: under noise the difference can come out below zero,
                // which would otherwise panic on `Duration` subtraction.
                let per_iter = t_2n.saturating_sub(t_n).as_secs_f64() / f64::from(iters);

                // Round 0 exists only to have run; nothing about it is reported.
                if round == 0 {
                    continue;
                }

                // Reject cells whose two-point difference is not resolvable, rather
                // than printing the reciprocal of noise as a measurement.
                //
                // `t_2n / t_n = (C + 2nT) / (C + nT)`: it tends to 2 when the kernel
                // work `nT` dominates the fixed round trip `C`, and to 1 when `C`
                // dominates. Near 1 the difference is mostly noise and `bytes /
                // per_iter` explodes. This is not theoretical — at the old default of
                // `ITERS=50`, `q4_k ffn down` printed **8927.9 GB/s, 2232% of an M1
                // Max's 400**, sitting in the table looking like a result.
                //
                // Requiring a 1.2x growth means `nT` is at least roughly a quarter of
                // `C`. The test is on the ratio, not on GB/s against `peak_gbs`,
                // because a small weight buffer can legitimately exceed DRAM peak out
                // of cache — so "above peak" is not by itself proof of nonsense,
                // whereas "doubling the work did not lengthen the run" is.
                //
                // If cells come back `noise`, raise `ITERS`.
                let resolvable = t_2n.as_secs_f64() >= t_n.as_secs_f64() * 1.2;
                if per_iter <= 0.0 || !resolvable {
                    println!(
                        "{:<6} {:<14} {:>6} {:>6} {:>10} {:>10} {:>9}",
                        kern.name, label, m, k, "noise", "noise", "noise"
                    );
                    continue;
                }

                let (be, bb) = block_geom(kern.dtype);
                let bytes = f64::from(m) * f64::from(k) * (bb as f64 / be as f64);
                let gbs = bytes / per_iter / 1e9;
                println!(
                    "{:<6} {:<14} {:>6} {:>6} {:>10.4} {:>10.1} {:>8.1}%",
                    kern.name,
                    label,
                    m,
                    k,
                    per_iter * 1e3,
                    gbs,
                    100.0 * gbs / peak_gbs,
                );
            }
        }
    }

    println!("\n* % of {peak_gbs} GB/s (override with PEAK_GBS)");
    Ok(())
}

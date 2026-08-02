//! Generated-vs-handwritten MSL microbenchmark for the Slang multi-target pilot.
//!
//! The parity suite answers "is the generated kernel correct". This answers the
//! question that actually decides adoption: **is it as fast as the kernel it
//! would replace**. Those are different, and the gap between them is silent:
//! `__target_switch` selecting the portable shared-memory tree instead of
//! Metal's two-stage `simd_max`/`simd_sum` produces a correct kernel that is
//! simply slower, which no correctness test can see.
//!
//! Times `shaders::SOFTMAX` (handwritten) against `shaders::SOFTMAX_SLANG`
//! (generated from `shaders/slang/softmax.slang`) on identical input, and checks
//! they agree numerically so a "faster" arm cannot be one that skipped work.
//!
//! Metal only. The wgpu twin is measurable on any host but not interesting: on
//! this branch nothing about the WGSL emission is in question, and the only
//! Vulkan device available to CI is a software rasterizer.
//!
//! ```sh
//! cargo run -p cera --features metal --release --example slang_softmax_bench
//! ```
//!
//! ## Measurement discipline
//!
//! Absolute GPU throughput is not stable across a session, so the design leans
//! entirely on the ratio:
//!
//! - one warm-up round per size, discarded, because the first dispatch after a
//!   pipeline is built pays compilation and power-state ramp;
//! - arms alternated each round, so a drift over the run cannot be read as a
//!   difference between the kernels;
//! - many dispatches per timed encoder, so submit and wait overhead is
//!   amortized rather than measured;
//! - the median round reported, not the mean, so one descheduled round does not
//!   move the answer.

use cera::backend::metal::{MetalContext, shaders};
use metal::MTLSize;

/// Dispatches per timed encoder. Softmax on a few thousand elements is far
/// shorter than a command-buffer round trip, so timing one dispatch would mostly
/// measure the submit.
const ITERS: u64 = 200;
/// Timed rounds per arm, after the discarded warm-up. Odd so the median is a
/// real sample rather than an average of two.
const ROUNDS: usize = 7;

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

/// One timed run: `ITERS` dispatches in a single command buffer. Returns
/// microseconds per dispatch.
///
/// The kernel is in-place and each dispatch consumes the previous one's output.
/// That is deliberate: the work per dispatch is data-independent, so repeated
/// application costs the same, and it avoids re-uploading inside the timed
/// region. Correctness is checked separately from a fresh buffer.
fn time_dispatches(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    x: &metal::Buffer,
    params: &metal::Buffer,
) -> f64 {
    let start = std::time::Instant::now();
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(params), 0);
    for _ in 0..ITERS {
        enc.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    start.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

/// Single dispatch from a fresh upload, returning the kernel's output.
fn run_once(ctx: &MetalContext, pipeline: &metal::ComputePipelineState, input: &[f32]) -> Vec<f32> {
    let x = ctx.upload_f32(input);
    let params = ctx.upload_bytes(bytemuck::cast_slice(&[input.len() as u32, 0u32]));
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&x), 0);
    enc.set_buffer(1, Some(&params), 0);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

fn main() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            std::process::exit(1);
        }
    };

    let hand = ctx
        .create_pipeline(shaders::SOFTMAX, "softmax")
        .expect("compile handwritten softmax.metal");
    let slang = ctx
        .create_pipeline(shaders::SOFTMAX_SLANG, "softmax")
        .expect("compile generated softmax.metal");

    // Establish the two kernels compute the same thing before comparing their
    // speed. A faster arm that disagrees is not a faster arm.
    println!("== agreement ==");
    let mut agree = true;
    for &n in &[100usize, 1000, 4096] {
        let input = softmax_input(n);
        let a = run_once(&ctx, &hand, &input);
        let b = run_once(&ctx, &slang, &input);
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let max_ref = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let ok = max_abs <= 1e-6 * max_ref.max(1e-6);
        agree &= ok;
        println!(
            "  n={n:<6} max_abs_diff={max_abs:.3e} (max|ref|={max_ref:.3e})  {}",
            if ok { "MATCH" } else { "MISMATCH" }
        );
    }
    if !agree {
        println!("\nkernels disagree; timings below are not comparable");
    }

    // `simd_max`/`simd_sum` in the generated source is the whole premise. If it
    // is absent the timings still print, but they are measuring the portable
    // tree and the comparison means something different, so say so up front.
    let kept_simd =
        shaders::SOFTMAX_SLANG.contains("simd_max") && shaders::SOFTMAX_SLANG.contains("simd_sum");
    if !kept_simd {
        println!(
            "\nWARNING: generated MSL has no simd_max/simd_sum. __target_switch took the\n\
             portable tree, so this is a tree-vs-simd comparison, not generated-vs-handwritten."
        );
    }

    println!(
        "\n== timing (us per dispatch, median of {ROUNDS} rounds, {ITERS} dispatches each) =="
    );
    println!(
        "{:>8}  {:>12}  {:>12}  {:>8}",
        "n", "handwritten", "generated", "ratio"
    );

    for &n in &[1024usize, 4096, 16384, 65536] {
        let input = softmax_input(n);
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[n as u32, 0u32]));
        let x_hand = ctx.upload_f32(&input);
        let x_slang = ctx.upload_f32(&input);

        // Discarded: first dispatch after pipeline creation pays compilation and
        // power-state ramp, which is not what this is measuring.
        time_dispatches(&ctx, &hand, &x_hand, &params);
        time_dispatches(&ctx, &slang, &x_slang, &params);

        let mut t_hand = Vec::with_capacity(ROUNDS);
        let mut t_slang = Vec::with_capacity(ROUNDS);
        for r in 0..ROUNDS {
            // Alternate order so a monotonic drift across the run cannot be
            // attributed to whichever arm happens to run second.
            if r % 2 == 0 {
                t_hand.push(time_dispatches(&ctx, &hand, &x_hand, &params));
                t_slang.push(time_dispatches(&ctx, &slang, &x_slang, &params));
            } else {
                t_slang.push(time_dispatches(&ctx, &slang, &x_slang, &params));
                t_hand.push(time_dispatches(&ctx, &hand, &x_hand, &params));
            }
        }

        let h = median(t_hand);
        let s = median(t_slang);
        println!("{n:>8}  {h:>12.2}  {s:>12.2}  {:>7.3}x", h / s);
    }

    println!(
        "\nratio > 1.00 means the generated kernel is faster; < 1.00 means the handwritten\n\
         one is. Treat anything within a few percent of 1.00 as no difference: this is a\n\
         wall-clock measurement of a kernel that runs in microseconds."
    );
}

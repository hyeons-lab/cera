//! Generated-vs-handwritten MSL microbenchmark for the Phase 1b-ii kernels:
//! argmax_f32 and rmsnorm. Same question and discipline as the other slang_*
//! benches, applied to the two special-case reductions.
//!
//! argmax_f32 reduces a (value, index) pair via `simd_shuffle_down`; rmsnorm
//! diverges in I/O model (metal is out-of-place src -> dst) as well as the
//! reduction. The parity suite checks correctness against a CPU reference; this
//! checks that the generated kernel is as fast as the handwritten one and that
//! the two agree directly.
//!
//! Metal only, for the same reason as the other benches.
//!
//! ```sh
//! cargo run -p cera --features metal --release --example slang_norm2_bench
//! ```

use cera::backend::metal::{MetalContext, shaders};
use metal::MTLSize;

const ITERS: u64 = 200;
const ROUNDS: usize = 7;

fn size1d(w: u64) -> MTLSize {
    MTLSize {
        width: w,
        height: 1,
        depth: 1,
    }
}

fn time_dispatches(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    bufs: &[&metal::Buffer],
    groups: u64,
) -> f64 {
    let start = std::time::Instant::now();
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    for _ in 0..ITERS {
        enc.dispatch_thread_groups(size1d(groups), size1d(256));
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    start.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn run_once(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    bufs: &[&metal::Buffer],
    groups: u64,
) {
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.dispatch_thread_groups(size1d(groups), size1d(256));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

fn compare(
    ctx: &MetalContext,
    hand: &metal::ComputePipelineState,
    slang: &metal::ComputePipelineState,
    hand_bufs: &[&metal::Buffer],
    slang_bufs: &[&metal::Buffer],
    groups: u64,
) -> (f64, f64) {
    time_dispatches(ctx, hand, hand_bufs, groups);
    time_dispatches(ctx, slang, slang_bufs, groups);
    let mut th = Vec::with_capacity(ROUNDS);
    let mut ts = Vec::with_capacity(ROUNDS);
    for r in 0..ROUNDS {
        if r % 2 == 0 {
            th.push(time_dispatches(ctx, hand, hand_bufs, groups));
            ts.push(time_dispatches(ctx, slang, slang_bufs, groups));
        } else {
            ts.push(time_dispatches(ctx, slang, slang_bufs, groups));
            th.push(time_dispatches(ctx, hand, hand_bufs, groups));
        }
    }
    (median(th), median(ts))
}

fn print_row(label: &str, h: f64, s: f64) {
    println!("{label:<22}  {h:>10.2}  {s:>10.2}  {:>7.3}x", h / s);
}

fn main() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            std::process::exit(1);
        }
    };

    println!("== agreement (generated vs handwritten, single dispatch) ==");
    let mut ok = true;

    // ---- argmax_f32: single workgroup, out[0] = index ----
    let am_n = 131072usize;
    let am_peak = 98765usize;
    let mut am_x: Vec<f32> = (0..am_n).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
    am_x[am_peak] = 10.0;
    let am_x_buf = ctx.upload_f32(&am_x);
    let am_params = ctx.upload_bytes(bytemuck::cast_slice(&[am_n as u32, 0u32]));
    let am_hand = ctx
        .create_pipeline(shaders::ARGMAX_F32, "argmax_f32")
        .expect("argmax_f32.metal");
    let am_slang = ctx
        .create_pipeline(shaders::ARGMAX_F32_SLANG, "argmax_f32")
        .expect("argmax_f32 slang");
    {
        let out_h = ctx.upload_bytes(bytemuck::cast_slice(&[0u32]));
        run_once(&ctx, &am_hand, &[&am_x_buf, &out_h, &am_params], 1);
        let ih = ctx.read_u32(&out_h, 1)[0];
        let out_s = ctx.upload_bytes(bytemuck::cast_slice(&[0u32]));
        run_once(&ctx, &am_slang, &[&am_x_buf, &out_s, &am_params], 1);
        let is = ctx.read_u32(&out_s, 1)[0];
        let match_ = ih == is && ih == am_peak as u32;
        ok &= match_;
        println!(
            "  argmax_f32            hand={ih} slang={is} (peak={am_peak})  {}",
            if match_ { "MATCH" } else { "MISMATCH" }
        );
    }

    // ---- rmsnorm: single workgroup, metal out-of-place src -> dst ----
    let rn_n = 8192usize;
    let rn_x: Vec<f32> = (0..rn_n).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let rn_w: Vec<f32> = (0..rn_n)
        .map(|j| 1.0 + 0.1 * (j as f32 * 0.04).cos())
        .collect();
    let rn_x_buf = ctx.upload_f32(&rn_x);
    let rn_w_buf = ctx.upload_f32(&rn_w);
    let rn_params = ctx.upload_bytes(bytemuck::cast_slice(&[
        rn_n as u32,
        1e-5f32.to_bits(),
        0u32,
        0u32,
    ]));
    let rn_hand = ctx
        .create_pipeline(shaders::RMSNORM, "rmsnorm")
        .expect("rmsnorm.metal");
    let rn_slang = ctx
        .create_pipeline(shaders::RMSNORM_SLANG, "rmsnorm")
        .expect("rmsnorm slang");
    {
        let dh = ctx.upload_f32(&vec![0.0f32; rn_n]);
        run_once(&ctx, &rn_hand, &[&rn_x_buf, &dh, &rn_w_buf, &rn_params], 1);
        let a = ctx.read_f32(&dh, rn_n);
        let ds = ctx.upload_f32(&vec![0.0f32; rn_n]);
        run_once(&ctx, &rn_slang, &[&rn_x_buf, &ds, &rn_w_buf, &rn_params], 1);
        let b = ctx.read_f32(&ds, rn_n);
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let max_ref = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let match_ = max_abs <= 1e-5 * max_ref.max(1e-6);
        ok &= match_;
        println!(
            "  rmsnorm               max_abs_diff={max_abs:.3e} (max|ref|={max_ref:.3e})  {}",
            if match_ { "MATCH" } else { "MISMATCH" }
        );
    }

    if !ok {
        println!("\nkernels disagree; timings below are not comparable");
    }

    println!(
        "\n== timing (us per dispatch, median of {ROUNDS} rounds, {ITERS} dispatches each) =="
    );
    println!(
        "{:<22}  {:>10}  {:>10}  {:>8}",
        "kernel", "handwritten", "generated", "ratio"
    );

    // argmax timing: reads x, writes out[0]; data-independent cost.
    {
        let out_h = ctx.upload_bytes(bytemuck::cast_slice(&[0u32]));
        let out_s = ctx.upload_bytes(bytemuck::cast_slice(&[0u32]));
        let (h, s) = compare(
            &ctx,
            &am_hand,
            &am_slang,
            &[&am_x_buf, &out_h, &am_params],
            &[&am_x_buf, &out_s, &am_params],
            1,
        );
        print_row(&format!("argmax_f32 n={am_n}"), h, s);
    }

    // rmsnorm timing: src shared, separate dst per arm; data-independent cost.
    {
        let dh = ctx.upload_f32(&vec![0.0f32; rn_n]);
        let ds = ctx.upload_f32(&vec![0.0f32; rn_n]);
        let (h, s) = compare(
            &ctx,
            &rn_hand,
            &rn_slang,
            &[&rn_x_buf, &dh, &rn_w_buf, &rn_params],
            &[&rn_x_buf, &ds, &rn_w_buf, &rn_params],
            1,
        );
        print_row(&format!("rmsnorm n={rn_n}"), h, s);
    }

    println!(
        "\nratio > 1.00 means the generated kernel is faster; < 1.00 the handwritten one is.\n\
         Treat anything within a few percent of 1.00 as no difference."
    );
}

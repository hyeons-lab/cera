//! Generated-vs-handwritten MSL microbenchmark for the Phase 1b-i norm kernels:
//! per_head_rmsnorm, layernorm_batch, rmsnorm_batch, and add_rmsnorm_batch. Same
//! question and discipline as slang_softmax_bench / slang_elementwise_bench,
//! applied to the reduction-divergent tier.
//!
//! These are `__target_switch` ports where the metal branch keeps a two-stage
//! `simd_sum` and the wgsl branch a shared-memory tree. The parity suite checks
//! correctness against a CPU reference; this checks the other thing that decides
//! adoption: that the generated kernel is as fast as the handwritten one, and
//! that the two agree directly (a signal the CPU-reference parity cannot give,
//! which is how the rope pow/powr divergence was caught in Phase 1a).
//!
//! Metal only, for the same reason as the other benches.
//!
//! ```sh
//! cargo run -p cera --features metal --release --example slang_norm_bench
//! ```
//!
//! Discipline (identical to the other benches): a discarded warm-up per kernel,
//! arms alternated per round, `ITERS` dispatches per timed encoder, median of
//! `ROUNDS`. In-place repeated application is data-independent in cost. The one
//! exception is add_rmsnorm_batch, which mutates src every dispatch: it is timed
//! with res_scale = 0 (a no-op add that still pays the residual read) so src
//! stays bounded; its agreement check uses a real res_scale.

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
    println!("{label:<26}  {h:>10.2}  {s:>10.2}  {:>7.3}x", h / s);
}

fn report_agreement(label: &str, a: &[f32], b: &[f32]) -> bool {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let max_ref = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let ok = max_abs <= 1e-5 * max_ref.max(1e-6);
    println!(
        "  {label:<24} max_abs_diff={max_abs:.3e} (max|ref|={max_ref:.3e})  {}",
        if ok { "MATCH" } else { "MISMATCH" }
    );
    ok
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

fn main() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            std::process::exit(1);
        }
    };

    let eps = 1e-5f32;
    println!("== agreement (generated vs handwritten, single dispatch) ==");
    let mut ok = true;

    // ---- per_head_rmsnorm: in-place x, [n_heads * head_dim] ----
    let (n_heads, head_dim) = (32u32, 128usize);
    let phr_hand = ctx
        .create_pipeline(shaders::PER_HEAD_RMSNORM, "per_head_rmsnorm")
        .expect("per_head_rmsnorm.metal");
    let phr_slang = ctx
        .create_pipeline(shaders::PER_HEAD_RMSNORM_SLANG, "per_head_rmsnorm")
        .expect("per_head_rmsnorm slang");
    let phr_x: Vec<f32> = (0..n_heads as usize * head_dim)
        .map(|i| (i as f32 * 0.01).sin() * 2.0)
        .collect();
    let phr_w: Vec<f32> = (0..head_dim)
        .map(|j| 1.0 + 0.1 * (j as f32 * 0.05).cos())
        .collect();
    let phr_params = ctx.upload_bytes(bytemuck::cast_slice(&[
        head_dim as u32,
        eps.to_bits(),
        0u32,
        0u32,
    ]));
    let phr_w_buf = ctx.upload_f32(&phr_w);
    {
        let xa = ctx.upload_f32(&phr_x);
        run_once(
            &ctx,
            &phr_hand,
            &[&xa, &phr_w_buf, &phr_params],
            n_heads as u64,
        );
        let a = ctx.read_f32(&xa, phr_x.len());
        let xb = ctx.upload_f32(&phr_x);
        run_once(
            &ctx,
            &phr_slang,
            &[&xb, &phr_w_buf, &phr_params],
            n_heads as u64,
        );
        let b = ctx.read_f32(&xb, phr_x.len());
        ok &= report_agreement("per_head_rmsnorm", &a, &b);
    }

    // ---- layernorm_batch: out-of-place, rows x n ----
    let (rows, n) = (64u32, 4096usize);
    let ln_params = ctx.upload_bytes(bytemuck::cast_slice(&[
        n as u32,
        eps.to_bits(),
        n as u32,
        n as u32,
    ]));
    let ln_src: Vec<f32> = (0..rows as usize * n)
        .map(|i| (i as f32 * 0.02).sin() * 3.0 + 0.5)
        .collect();
    let ln_w: Vec<f32> = (0..n)
        .map(|j| 1.0 + 0.05 * (j as f32 * 0.03).cos())
        .collect();
    let ln_b: Vec<f32> = (0..n).map(|j| 0.1 * (j as f32 * 0.07).sin()).collect();
    let ln_src_buf = ctx.upload_f32(&ln_src);
    let ln_w_buf = ctx.upload_f32(&ln_w);
    let ln_b_buf = ctx.upload_f32(&ln_b);
    let ln_hand = ctx
        .create_pipeline(shaders::LAYERNORM_BATCH, "layernorm_batch")
        .expect("layernorm_batch.metal");
    let ln_slang = ctx
        .create_pipeline(shaders::LAYERNORM_BATCH_SLANG, "layernorm_batch")
        .expect("layernorm_batch slang");
    {
        let da = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        run_once(
            &ctx,
            &ln_hand,
            &[&ln_src_buf, &da, &ln_w_buf, &ln_b_buf, &ln_params],
            rows as u64,
        );
        let a = ctx.read_f32(&da, rows as usize * n);
        let db = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        run_once(
            &ctx,
            &ln_slang,
            &[&ln_src_buf, &db, &ln_w_buf, &ln_b_buf, &ln_params],
            rows as u64,
        );
        let b = ctx.read_f32(&db, rows as usize * n);
        ok &= report_agreement("layernorm_batch", &a, &b);
    }

    // ---- rmsnorm_batch + add_rmsnorm_batch: out-of-place, rows x n ----
    let rb_src: Vec<f32> = (0..rows as usize * n)
        .map(|i| (i as f32 * 0.02).sin() * 2.0)
        .collect();
    let rb_w: Vec<f32> = (0..n)
        .map(|j| 1.0 + 0.1 * (j as f32 * 0.04).cos())
        .collect();
    let rb_res: Vec<f32> = (0..rows as usize * n)
        .map(|i| (i as f32 * 0.015).cos() * 1.5)
        .collect();
    let rb_w_buf = ctx.upload_f32(&rb_w);
    let rb_res_buf = ctx.upload_f32(&rb_res);
    // params with res_scale = 0.75 for the agreement check.
    let rb_params_075 = ctx.upload_bytes(bytemuck::cast_slice(&[
        n as u32,
        eps.to_bits(),
        n as u32,
        n as u32,
        0.75f32.to_bits(),
    ]));
    let rb_hand = ctx
        .create_pipeline(shaders::RMSNORM_BATCH, "rmsnorm_batch")
        .expect("rmsnorm_batch.metal");
    let rb_slang = ctx
        .create_pipeline(shaders::RMSNORM_BATCH_SLANG, "rmsnorm_batch")
        .expect("rmsnorm_batch slang");
    let arb_hand = ctx
        .create_pipeline(shaders::RMSNORM_BATCH, "add_rmsnorm_batch")
        .expect("add_rmsnorm_batch.metal");
    let arb_slang = ctx
        .create_pipeline(shaders::RMSNORM_BATCH_SLANG, "add_rmsnorm_batch")
        .expect("add_rmsnorm_batch slang");
    {
        let sa = ctx.upload_f32(&rb_src);
        let da = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        run_once(
            &ctx,
            &rb_hand,
            &[&sa, &da, &rb_w_buf, &rb_params_075],
            rows as u64,
        );
        let a = ctx.read_f32(&da, rows as usize * n);
        let sb = ctx.upload_f32(&rb_src);
        let db = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        run_once(
            &ctx,
            &rb_slang,
            &[&sb, &db, &rb_w_buf, &rb_params_075],
            rows as u64,
        );
        let b = ctx.read_f32(&db, rows as usize * n);
        ok &= report_agreement("rmsnorm_batch", &a, &b);

        // add_rmsnorm_batch mutates src, so check both the post-add src and dst.
        let sa = ctx.upload_f32(&rb_src);
        let da = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        run_once(
            &ctx,
            &arb_hand,
            &[&sa, &da, &rb_w_buf, &rb_params_075, &rb_res_buf],
            rows as u64,
        );
        let (asrc, adst) = (
            ctx.read_f32(&sa, rows as usize * n),
            ctx.read_f32(&da, rows as usize * n),
        );
        let sb = ctx.upload_f32(&rb_src);
        let db = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        run_once(
            &ctx,
            &arb_slang,
            &[&sb, &db, &rb_w_buf, &rb_params_075, &rb_res_buf],
            rows as u64,
        );
        let (bsrc, bdst) = (
            ctx.read_f32(&sb, rows as usize * n),
            ctx.read_f32(&db, rows as usize * n),
        );
        ok &= report_agreement("add_rmsnorm_batch (src)", &asrc, &bsrc);
        ok &= report_agreement("add_rmsnorm_batch (dst)", &adst, &bdst);
    }

    if !ok {
        println!("\nkernels disagree; timings below are not comparable");
    }

    println!(
        "\n== timing (us per dispatch, median of {ROUNDS} rounds, {ITERS} dispatches each) =="
    );
    println!(
        "{:<26}  {:>10}  {:>10}  {:>8}",
        "kernel", "handwritten", "generated", "ratio"
    );

    // per_head_rmsnorm timing (in-place; separate x per arm).
    {
        let xh = ctx.upload_f32(&phr_x);
        let xs = ctx.upload_f32(&phr_x);
        let (h, s) = compare(
            &ctx,
            &phr_hand,
            &phr_slang,
            &[&xh, &phr_w_buf, &phr_params],
            &[&xs, &phr_w_buf, &phr_params],
            n_heads as u64,
        );
        print_row(&format!("per_head_rmsnorm {n_heads}x{head_dim}"), h, s);
    }

    // layernorm_batch timing (out-of-place; src shared, separate dst per arm).
    {
        let dh = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        let ds = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        let (h, s) = compare(
            &ctx,
            &ln_hand,
            &ln_slang,
            &[&ln_src_buf, &dh, &ln_w_buf, &ln_b_buf, &ln_params],
            &[&ln_src_buf, &ds, &ln_w_buf, &ln_b_buf, &ln_params],
            rows as u64,
        );
        print_row(&format!("layernorm_batch {rows}x{n}"), h, s);
    }

    // rmsnorm_batch timing (out-of-place; src shared, separate dst per arm).
    {
        let sh = ctx.upload_f32(&rb_src);
        let dh = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        let ss = ctx.upload_f32(&rb_src);
        let ds = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        let (h, s) = compare(
            &ctx,
            &rb_hand,
            &rb_slang,
            &[&sh, &dh, &rb_w_buf, &rb_params_075],
            &[&ss, &ds, &rb_w_buf, &rb_params_075],
            rows as u64,
        );
        print_row(&format!("rmsnorm_batch {rows}x{n}"), h, s);
    }

    // add_rmsnorm_batch timing: res_scale = 0 so the repeated in-place add is a
    // no-op (src stays bounded) while still paying the residual read.
    {
        let params0 = ctx.upload_bytes(bytemuck::cast_slice(&[
            n as u32,
            eps.to_bits(),
            n as u32,
            n as u32,
            0.0f32.to_bits(),
        ]));
        let sh = ctx.upload_f32(&rb_src);
        let dh = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        let ss = ctx.upload_f32(&rb_src);
        let ds = ctx.upload_f32(&vec![0.0f32; rows as usize * n]);
        let (h, s) = compare(
            &ctx,
            &arb_hand,
            &arb_slang,
            &[&sh, &dh, &rb_w_buf, &params0, &rb_res_buf],
            &[&ss, &ds, &rb_w_buf, &params0, &rb_res_buf],
            rows as u64,
        );
        print_row(&format!("add_rmsnorm_batch {rows}x{n}"), h, s);
    }

    println!(
        "\nratio > 1.00 means the generated kernel is faster; < 1.00 the handwritten one is.\n\
         Treat anything within a few percent of 1.00 as no difference."
    );
}

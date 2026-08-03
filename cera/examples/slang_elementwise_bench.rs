//! Generated-vs-handwritten MSL microbenchmark for the Phase 1a Slang kernels:
//! gelu, the four shared elementwise ops, and rope. Same question as
//! `slang_softmax_bench` (is the generated kernel as fast as the one it would
//! replace) and the same measurement discipline, extended to the
//! elementwise/math tier.
//!
//! Unlike softmax and the GEMM, these kernels have no reduction, no barrier and
//! no tiling: each is a branchless per-element map (or, for rope, a per-pair
//! rotation). A generated kernel that lowered them differently is not expected to
//! change throughput, but "not expected" is a hypothesis, and the whole point of
//! the pilot is to measure rather than assume.
//!
//! Metal only, for the same reason as the other two: the WGSL half is verified by
//! parity and the only Vulkan device CI can reach is a software rasterizer.
//!
//! ```sh
//! cargo run -p cera --features metal --release --example slang_elementwise_bench
//! ```
//!
//! Discipline (identical to `slang_softmax_bench`): a discarded warm-up round per
//! kernel, arms alternated each round so drift is not read as a kernel
//! difference, `ITERS` dispatches per timed encoder so submit/wait is amortized,
//! and the median round reported. In-place repeated application is
//! data-independent in cost, so it stays off the upload path inside the timed
//! region; correctness is checked separately from a fresh buffer.

use cera::backend::metal::{MetalContext, shaders};
use metal::MTLSize;

/// Dispatches per timed encoder. A single elementwise pass over ~1M floats still
/// runs in tens of microseconds, comparable to a command-buffer round trip, so
/// timing one dispatch would fold in the submit.
const ITERS: u64 = 200;
/// Timed rounds per arm after the discarded warm-up. Odd so the median is a real
/// sample, not an average of two.
const ROUNDS: usize = 7;
/// Element count for the elementwise/gelu kernels: large enough to be
/// bandwidth-bound and dwarf per-dispatch launch cost.
const N: usize = 1 << 20;

fn size1d(w: u64) -> MTLSize {
    MTLSize {
        width: w,
        height: 1,
        depth: 1,
    }
}

/// One timed run: `ITERS` in-place dispatches in a single command buffer, over
/// `groups` threadgroups of 256. Returns microseconds per dispatch.
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

/// Single dispatch of `pipeline` over `bufs`; no timing.
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

fn rel_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let max_ref = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    (max_abs, max_ref)
}

/// Median us/dispatch for both arms, warm-up discarded, order alternated.
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

fn report_agreement(label: &str, a: &[f32], b: &[f32]) -> bool {
    let (max_abs, max_ref) = rel_diff(a, b);
    let ok = max_abs <= 1e-5 * max_ref.max(1e-6);
    println!(
        "  {label:<20} max_abs_diff={max_abs:.3e} (max|ref|={max_ref:.3e})  {}",
        if ok { "MATCH" } else { "MISMATCH" }
    );
    ok
}

fn main() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            std::process::exit(1);
        }
    };

    let groups = (N as u64).div_ceil(256);
    // `a`/x span both signs; `b` is kept in [0.5, 1.0] so the in-place timing
    // loop cannot overflow to inf (the failure mode that would change per-op
    // cost). `mul` does shrink toward zero over the loop, which is harmless: a
    // branchless f32 multiply costs the same on any finite or zero operand.
    let x: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.001).sin() * 3.0).collect();
    let bvals: Vec<f32> = (0..N)
        .map(|i| ((i as f32) * 0.0017).cos() * 0.25 + 0.75)
        .collect();

    println!("== agreement (generated vs handwritten, single dispatch) ==");

    // gelu
    let gelu_hand = ctx
        .create_pipeline(shaders::GELU, "gelu_inplace")
        .expect("gelu.metal");
    let gelu_slang = ctx
        .create_pipeline(shaders::GELU_SLANG, "gelu_inplace")
        .expect("gelu slang");
    let gparams = ctx.upload_bytes(bytemuck::cast_slice(&[N as u32, 0u32]));
    let mut ok = true;
    {
        let xa = ctx.upload_f32(&x);
        run_once(&ctx, &gelu_hand, &[&xa, &gparams], groups);
        let a = ctx.read_f32(&xa, N);
        let xb = ctx.upload_f32(&x);
        run_once(&ctx, &gelu_slang, &[&xb, &gparams], groups);
        let b = ctx.read_f32(&xb, N);
        ok &= report_agreement("gelu", &a, &b);
    }

    // elementwise: (entry, scale_bits)
    let ew_ops: [(&str, u32); 4] = [
        ("add_inplace", 0),
        ("mul_inplace", 0),
        ("scaled_add_inplace", 1.0f32.to_bits()),
        ("silu_mul_inplace", 0),
    ];
    let ew_hand = shaders::ELEMENTWISE;
    let ew_slang = shaders::ELEMENTWISE_SLANG;
    for (entry, scale) in ew_ops {
        let hand = ctx
            .create_pipeline(ew_hand, entry)
            .expect("elementwise.metal");
        let slang = ctx
            .create_pipeline(ew_slang, entry)
            .expect("elementwise slang");
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[N as u32, scale]));
        let bh = ctx.upload_f32(&bvals);
        let ah = ctx.upload_f32(&x);
        run_once(&ctx, &hand, &[&ah, &bh, &params], groups);
        let a = ctx.read_f32(&ah, N);
        let bs = ctx.upload_f32(&bvals);
        let as_ = ctx.upload_f32(&x);
        run_once(&ctx, &slang, &[&as_, &bs, &params], groups);
        let b = ctx.read_f32(&as_, N);
        ok &= report_agreement(entry, &a, &b);
    }

    // rope: realistic single-token GQA shape.
    let (n_heads, n_kv_heads, head_dim) = (32u32, 8u32, 128usize);
    let half = head_dim / 2;
    let rope_groups = (n_heads.max(n_kv_heads) * half as u32).div_ceil(256) as u64;
    let rope_hand = ctx
        .create_pipeline(shaders::ROPE, "rope")
        .expect("rope.metal");
    let rope_slang = ctx
        .create_pipeline(shaders::ROPE_SLANG, "rope")
        .expect("rope slang");
    let rparams = ctx.upload_bytes(bytemuck::cast_slice(&[
        1000u32,
        n_heads,
        n_kv_heads,
        head_dim as u32,
        10000.0f32.to_bits(),
    ]));
    let qv: Vec<f32> = (0..n_heads as usize * head_dim)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();
    let kv: Vec<f32> = (0..n_kv_heads as usize * head_dim)
        .map(|i| ((i as f32) * 0.02).cos())
        .collect();
    {
        let qh = ctx.upload_f32(&qv);
        let kh = ctx.upload_f32(&kv);
        run_once(&ctx, &rope_hand, &[&qh, &kh, &rparams], rope_groups);
        let aq = ctx.read_f32(&qh, qv.len());
        let ak = ctx.read_f32(&kh, kv.len());
        let qs = ctx.upload_f32(&qv);
        let ks = ctx.upload_f32(&kv);
        run_once(&ctx, &rope_slang, &[&qs, &ks, &rparams], rope_groups);
        let bq = ctx.read_f32(&qs, qv.len());
        let bk = ctx.read_f32(&ks, kv.len());
        // Both rotated buffers: the kernel writes q and k, so check both.
        ok &= report_agreement("rope (q)", &aq, &bq);
        ok &= report_agreement("rope (k)", &ak, &bk);
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

    // gelu timing
    {
        let xh = ctx.upload_f32(&x);
        let xs = ctx.upload_f32(&x);
        let (h, s) = compare(
            &ctx,
            &gelu_hand,
            &gelu_slang,
            &[&xh, &gparams],
            &[&xs, &gparams],
            groups,
        );
        print_row(&format!("gelu (n={N})"), h, s);
    }

    // elementwise timing
    for (entry, scale) in ew_ops {
        let hand = ctx
            .create_pipeline(ew_hand, entry)
            .expect("elementwise.metal");
        let slang = ctx
            .create_pipeline(ew_slang, entry)
            .expect("elementwise slang");
        let params = ctx.upload_bytes(bytemuck::cast_slice(&[N as u32, scale]));
        let ah = ctx.upload_f32(&x);
        let bh = ctx.upload_f32(&bvals);
        let as_ = ctx.upload_f32(&x);
        let bs = ctx.upload_f32(&bvals);
        let (h, s) = compare(
            &ctx,
            &hand,
            &slang,
            &[&ah, &bh, &params],
            &[&as_, &bs, &params],
            groups,
        );
        print_row(entry, h, s);
    }

    // rope timing
    {
        let qh = ctx.upload_f32(&qv);
        let kh = ctx.upload_f32(&kv);
        let qs = ctx.upload_f32(&qv);
        let ks = ctx.upload_f32(&kv);
        let (h, s) = compare(
            &ctx,
            &rope_hand,
            &rope_slang,
            &[&qh, &kh, &rparams],
            &[&qs, &ks, &rparams],
            rope_groups,
        );
        print_row("rope (1 token, GQA)", h, s);
    }

    println!(
        "\nratio > 1.00 means the generated kernel is faster; < 1.00 the handwritten one is.\n\
         These are branchless maps running in microseconds, so treat anything within a few\n\
         percent of 1.00 as no difference."
    );
}

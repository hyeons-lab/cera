//! Generated-vs-handwritten MSL microbenchmark for the conv tier: conv1d,
//! conv1d_fused and conv1d_fused_batch. Same question and discipline as the
//! other slang_* benches.
//!
//! These are the first clean single-body ports since Phase 1a (no
//! `__target_switch`, no `__target_intrinsic`), so the expectation is that the
//! generated kernels are indistinguishable from the handwritten ones. The
//! parity suite checks correctness against a CPU reference; this checks speed
//! and, just as importantly, compares the two kernels' outputs *directly*, the
//! signal a CPU-reference test cannot give (it is what caught rope's pow/powr
//! divergence in Phase 1a).
//!
//! Every kernel here mutates its rolling buffer, so each timed arm gets its own
//! state buffer and the agreement check compares both outputs.
//!
//! Metal only, for the same reason as the other benches.
//!
//! ```sh
//! cargo run -p cera --features metal --release --example slang_conv_bench
//! ```

use cera::backend::metal::{MetalContext, shaders};
use metal::MTLSize;

const ITERS: u64 = 200;
const ROUNDS: usize = 7;

// The production shape: shipped LFM2 GGUFs set `lfm2.shortconv.l_cache = 3`, so
// the hosts derive kernel_size = 3 and d_conv = 2. Measure what actually runs.
const KS: usize = 3;
const D_CONV: usize = 2;

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
    println!("{label:<32}  {h:>10.2}  {s:>10.2}  {:>7.3}x", h / s);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// One kernel's fixtures: the shared read-only inputs plus a private rolling
/// buffer and output per arm, since both are written.
struct Case {
    first: metal::Buffer,
    weight: metal::Buffer,
    params: metal::Buffer,
    rb_hand: metal::Buffer,
    rb_slang: metal::Buffer,
    out_hand: metal::Buffer,
    out_slang: metal::Buffer,
    rb_len: usize,
    out_len: usize,
    groups: u64,
}

impl Case {
    /// `n_tokens` of 1 builds the four-word params both single-token kernels
    /// take; anything larger builds the batched kernel's six. Deriving the
    /// params here rather than accepting them keeps them from drifting away from
    /// the buffer sizes, which are shaped by the same `ks` / `d_conv`.
    fn new(ctx: &MetalContext, first: &[f32], hs: usize, n_tokens: usize) -> Self {
        let rbuffer: Vec<f32> = (0..D_CONV * hs)
            .map(|i| (i as f32 * 0.017).cos() * 0.5)
            .collect();
        let weight: Vec<f32> = (0..hs * KS)
            .map(|i| 0.1 + 0.05 * (i as f32 * 0.03).sin())
            .collect();
        let out_len = n_tokens * hs;
        let params: Vec<u32> = if n_tokens == 1 {
            vec![hs as u32, KS as u32, D_CONV as u32, 0]
        } else {
            vec![
                hs as u32,
                KS as u32,
                D_CONV as u32,
                n_tokens as u32,
                (3 * hs) as u32,
                hs as u32,
            ]
        };
        Self {
            first: ctx.upload_f32(first),
            weight: ctx.upload_f32(&weight),
            params: ctx.upload_bytes(bytemuck::cast_slice(&params)),
            rb_hand: ctx.upload_f32(&rbuffer),
            rb_slang: ctx.upload_f32(&rbuffer),
            out_hand: ctx.upload_f32(&vec![0.0f32; out_len]),
            out_slang: ctx.upload_f32(&vec![0.0f32; out_len]),
            rb_len: rbuffer.len(),
            out_len,
            groups: hs.div_ceil(256) as u64,
        }
    }

    fn hand_bufs(&self) -> [&metal::Buffer; 5] {
        [
            &self.first,
            &self.rb_hand,
            &self.weight,
            &self.out_hand,
            &self.params,
        ]
    }

    fn slang_bufs(&self) -> [&metal::Buffer; 5] {
        [
            &self.first,
            &self.rb_slang,
            &self.weight,
            &self.out_slang,
            &self.params,
        ]
    }

    /// Dispatch each arm once and report whether the outputs and the advanced
    /// rolling buffers agree.
    ///
    /// Both output buffers start zeroed and both rolling buffers start
    /// identical, so "the two arms agree" is not on its own evidence of
    /// anything: if both kernels early-returned (wrong params, wrong entry
    /// point) every difference would be 0 and this would print MATCH while
    /// measuring two no-ops. The `wrote_output` check closes that.
    fn agree(
        &self,
        ctx: &MetalContext,
        label: &str,
        hand: &metal::ComputePipelineState,
        slang: &metal::ComputePipelineState,
    ) -> bool {
        run_once(ctx, hand, &self.hand_bufs(), self.groups);
        run_once(ctx, slang, &self.slang_bufs(), self.groups);
        let oh = ctx.read_f32(&self.out_hand, self.out_len);
        let os = ctx.read_f32(&self.out_slang, self.out_len);
        let rh = ctx.read_f32(&self.rb_hand, self.rb_len);
        let rs = ctx.read_f32(&self.rb_slang, self.rb_len);
        let d_out = max_abs_diff(&oh, &os);
        let d_rb = max_abs_diff(&rh, &rs);
        let mag = oh.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // Both arms must actually have written something, or "agreement" is two
        // untouched zero buffers comparing equal.
        let wrote_output = mag > 0.0 && os.iter().any(|v| *v != 0.0);
        let ok = wrote_output && d_out <= 1e-5 * mag.max(1e-6) && d_rb <= 1e-5;
        let verdict = if !wrote_output {
            "NO OUTPUT"
        } else if ok {
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!("  {label:<24} out_diff={d_out:.3e} rbuffer_diff={d_rb:.3e}  {verdict}");
        ok
    }
}

fn main() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            std::process::exit(1);
        }
    };

    // Above LFM2's real hidden size (1024-2048) so the kernels get some actual
    // work. Read the two single-token rows with care even so: they do one pass
    // over `hs` channels and stay launch- and bandwidth-bound at any size worth
    // allocating (measured 10.6us at hs=8192 and 14.1us at hs=131072, i.e. 16x
    // the work for 1.3x the time), so a few percent there is not a codegen
    // signal. The batched row is the one that scales with its work and can
    // resolve small differences; it is also the kernel where the register-array
    // codegen actually matters.
    let hs = 8192usize;
    let n_tokens = 64usize;

    let input: Vec<f32> = (0..hs).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let proj1: Vec<f32> = (0..3 * hs)
        .map(|i| (i as f32 * 0.007).sin() * 1.5)
        .collect();
    let proj_n: Vec<f32> = (0..n_tokens * 3 * hs)
        .map(|i| (i as f32 * 0.007).sin() * 1.5)
        .collect();

    let cases = [
        ("conv1d", Case::new(&ctx, &input, hs, 1)),
        ("conv1d_fused", Case::new(&ctx, &proj1, hs, 1)),
        ("conv1d_fused_batch", Case::new(&ctx, &proj_n, hs, n_tokens)),
    ];

    let pipelines: Vec<(metal::ComputePipelineState, metal::ComputePipelineState)> = [
        (shaders::CONV1D, shaders::CONV1D_SLANG, "conv1d_depthwise"),
        (
            shaders::CONV1D_FUSED,
            shaders::CONV1D_FUSED_SLANG,
            "conv1d_fused",
        ),
        (
            shaders::CONV1D_FUSED_BATCH,
            shaders::CONV1D_FUSED_BATCH_SLANG,
            "conv1d_fused_batch",
        ),
    ]
    .iter()
    .map(|(hand_src, slang_src, entry)| {
        (
            ctx.create_pipeline(hand_src, entry)
                .unwrap_or_else(|e| panic!("handwritten {entry}: {e}")),
            ctx.create_pipeline(slang_src, entry)
                .unwrap_or_else(|e| panic!("slang {entry}: {e}")),
        )
    })
    .collect();

    println!("== agreement (generated vs handwritten, single dispatch) ==");
    let mut ok = true;
    for ((label, case), (hand, slang)) in cases.iter().zip(&pipelines) {
        ok &= case.agree(&ctx, label, hand, slang);
    }
    if !ok {
        println!("\nkernels disagree; timings below are not comparable");
    }

    println!(
        "\n== timing (us per dispatch, median of {ROUNDS} rounds, {ITERS} dispatches each) =="
    );
    println!(
        "{:<32}  {:>10}  {:>10}  {:>8}",
        "kernel", "handwritten", "generated", "ratio"
    );
    for ((label, case), (hand, slang)) in cases.iter().zip(&pipelines) {
        let (h, s) = compare(
            &ctx,
            hand,
            slang,
            &case.hand_bufs(),
            &case.slang_bufs(),
            case.groups,
        );
        let suffix = if *label == "conv1d_fused_batch" {
            format!(" hs={hs} n={n_tokens}")
        } else {
            format!(" hs={hs}")
        };
        print_row(&format!("{label}{suffix}"), h, s);
    }

    println!(
        "\nratio > 1.00 means the generated kernel is faster; < 1.00 the handwritten one is.\n\
         The batched row is the informative one: it scales with its work, and a few\n\
         percent there is real. The two single-token rows are launch- and\n\
         bandwidth-bound and have measured anywhere from 0.86x to 1.09x run to run\n\
         on an idle M-series host, so read them only for large regressions."
    );

    // Exit nonzero on disagreement so this is usable as a check, not just a
    // report: a divergent port should not look like a clean run to a script.
    if !ok {
        std::process::exit(1);
    }
}

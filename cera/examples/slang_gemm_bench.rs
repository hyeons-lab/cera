//! Generated-vs-handwritten MSL benchmark for the Slang Q8_0 GEMM.
//!
//! This is the measurement the whole Slang branch turns on. `softmax.slang`
//! proved a reduction can be shared and `coopmat_probe.slang` proved
//! `linalg::CoopMat` reaches Metal's `simdgroup_matrix` at all, but neither says
//! whether a *generated* simdgroup GEMM keeps the speed of the hand-tuned one.
//! Every hot kernel in the Metal backend is that shape, so if the answer here is
//! no, the migration stops being interesting regardless of how clean the source
//! is.
//!
//! Times `shaders::GEMM_Q8_0` (handwritten, `shaders/gemm_q8_0.metal`) against
//! `shaders::GEMM_Q8_0_SLANG` (generated from `shaders/slang/gemm_q8_0.slang`)
//! on identical buffers.
//!
//! ```sh
//! cargo run -p cera --features metal --release --example slang_gemm_bench
//! ```
//!
//! ## What a regression here would most likely be
//!
//! The port carries two known divergences, both documented in the .slang: no
//! `simdgroup_barrier` between the operand loads and the MMAs, and a two-round
//! ragged epilogue instead of a single reinterpreted scratch buffer. Only the
//! first can touch these numbers, since every shape below is a whole multiple of
//! the 64x32 tile and takes the fast-path store.
//!
//! A third possibility is invisible in the source: eight live 8x8 float
//! accumulators plus six operand registers may or may not survive as registers
//! after Slang's codegen. If they spill, it shows up here and nowhere else.
//!
//! This is the same failure mode `slang_softmax_bench` caught: that kernel
//! passed its whole parity suite while carrying one extra threadgroup barrier
//! and running ~24% slower at the size where barrier latency dominates.
//!
//! ## Measurement discipline
//!
//! Same as `slang_softmax_bench`: a discarded warm-up per shape, arms alternated
//! each round so drift cannot be read as a kernel difference, several dispatches
//! per timed encoder to amortize the submit, and the median round rather than
//! the mean.

use cera::backend::metal::{MetalContext, shaders};
use metal::MTLSize;

/// Dispatches per timed encoder. Lower than the softmax bench because a GEMM
/// dispatch is milliseconds, not microseconds, so submit overhead is already
/// negligible and more iterations only lengthen the run.
const ITERS: u64 = 20;
/// Timed rounds per arm, after the discarded warm-up. Odd so the median is a
/// real sample.
const ROUNDS: usize = 7;

/// Threadgroup scratch the handwritten kernel takes as an argument: 4 KB of half
/// weights + 4 KB of float input. The Slang port declares the same 8 KB
/// statically, which is why only one arm sets this.
const SHMEM_BYTES: u64 = 8192;

/// `(m, k, n)`: output rows, reduction depth, token count. All whole multiples
/// of the 64x32 tile so both kernels take the fast-path store and the comparison
/// is of the inner loop rather than of two different epilogues.
const SHAPES: &[(usize, usize, usize)] = &[
    (1024, 1024, 128),
    (2048, 2048, 256),
    (4096, 4096, 256),
    (2048, 2048, 512),
];

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Pack `[m, k]` row-major f32 as Q8_0: 34 bytes per 32-element block.
fn q8_0_pack(src: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(src.len() / 32 * 34);
    for block in src.as_chunks::<32>().0 {
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

struct Bufs {
    src0: metal::Buffer,
    src1: metal::Buffer,
    dst: metal::Buffer,
    params: metal::Buffer,
    grid: MTLSize,
}

fn make_bufs(ctx: &MetalContext, m: usize, k: usize, n: usize) -> Bufs {
    let packed = q8_0_pack(&data(m * k, 7));
    let x = data(n * k, 13);
    Bufs {
        src0: ctx.upload_bytes(&packed),
        src1: ctx.upload_f32(&x),
        dst: ctx.upload_f32(&vec![0.0f32; m * n]),
        // x_stride = k, y_stride = m: both operands tightly packed.
        params: ctx.upload_bytes(bytemuck::cast_slice(&[
            m as u32, k as u32, n as u32, k as u32, m as u32, 0u32,
        ])),
        grid: MTLSize {
            width: n.div_ceil(32) as u64,
            height: m.div_ceil(64) as u64,
            depth: 1,
        },
    }
}

fn encode(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    b: &Bufs,
    needs_shmem: bool,
    iters: u64,
) {
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&b.src0), 0);
    enc.set_buffer(1, Some(&b.src1), 0);
    enc.set_buffer(2, Some(&b.dst), 0);
    enc.set_buffer(3, Some(&b.params), 0);
    if needs_shmem {
        enc.set_threadgroup_memory_length(0, SHMEM_BYTES);
    }
    for _ in 0..iters {
        enc.dispatch_thread_groups(
            b.grid,
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

/// One timed run: `ITERS` dispatches in a single command buffer. Microseconds
/// per dispatch.
///
/// Every dispatch writes the same output from the same inputs. That is fine for
/// timing (the work is identical and data-independent) and it keeps re-uploads
/// out of the timed region.
fn time_dispatches(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    b: &Bufs,
    needs_shmem: bool,
) -> f64 {
    let start = std::time::Instant::now();
    encode(ctx, pipeline, b, needs_shmem, ITERS);
    start.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

/// GFLOP/s for a `m x k` by `k x n` product, counting the usual 2 flops per MAC.
fn gflops(m: usize, k: usize, n: usize, us: f64) -> f64 {
    2.0 * (m * k * n) as f64 / (us * 1e-6) / 1e9
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
        .create_pipeline(shaders::GEMM_Q8_0, "gemm_q8_0")
        .expect("compile handwritten gemm_q8_0.metal");
    let slang = ctx
        .create_pipeline(shaders::GEMM_Q8_0_SLANG, "gemm_q8_0")
        .expect("compile generated gemm_q8_0.metal");

    // maxTotalThreadsPerThreadgroup is a register-pressure proxy: the driver
    // sets it from how many registers each thread needs, so a lower number means
    // a heavier kernel and fewer resident threads (lower occupancy). This was the
    // signal that located the dequant as the bottleneck: the naive per-byte
    // extraction cut the generated kernel to 768 while the handwritten held 832.
    println!("== occupancy (higher = lower register pressure) ==");
    println!(
        "  handwritten  maxThreadsPerTG={:<5} execWidth={}",
        hand.max_total_threads_per_threadgroup(),
        hand.thread_execution_width(),
    );
    println!(
        "  generated    maxThreadsPerTG={:<5} execWidth={}\n",
        slang.max_total_threads_per_threadgroup(),
        slang.thread_execution_width(),
    );

    // The generated kernel is supposed to reach the same instructions, not just
    // the same answers. If it did not, the timings still print but they are
    // measuring the portable branch and mean something else entirely.
    if !shaders::GEMM_Q8_0_SLANG.contains("simdgroup_multiply_accumulate") {
        println!(
            "WARNING: generated MSL has no simdgroup_multiply_accumulate. __target_switch\n\
             took the portable branch, so this is scalar-vs-MMA, not generated-vs-handwritten.\n"
        );
    }

    // Both kernels stage weights as half and accumulate in float over the same
    // tile order, so they should agree to the last bit. A difference is not
    // automatically a bug, but it means the two are no longer doing the same
    // arithmetic and the timings need that caveat attached.
    println!("== agreement ==");
    let mut agree = true;
    for &(m, k, n) in &[(128usize, 256usize, 64usize), (192, 128, 96)] {
        let a = make_bufs(&ctx, m, k, n);
        let b = make_bufs(&ctx, m, k, n);
        encode(&ctx, &hand, &a, true, 1);
        encode(&ctx, &slang, &b, false, 1);
        let ra = ctx.read_f32(&a.dst, m * n);
        let rb = ctx.read_f32(&b.dst, m * n);

        let max_abs = ra
            .iter()
            .zip(&rb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let max_ref = ra.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let ok = max_abs <= 1e-5 * max_ref.max(1e-6);
        agree &= ok;
        println!(
            "  {m}x{k}x{n:<5} max_abs_diff={max_abs:.3e} (max|ref|={max_ref:.3e})  {}{}",
            if ok { "MATCH" } else { "MISMATCH" },
            if max_abs == 0.0 {
                " (bit-identical)"
            } else {
                ""
            }
        );
    }
    if !agree {
        println!("\nkernels disagree; timings below are not comparable");
    }

    println!(
        "\n== timing (us per dispatch, median of {ROUNDS} rounds, {ITERS} dispatches each) =="
    );
    println!(
        "{:>16}  {:>12}  {:>12}  {:>9}  {:>9}  {:>8}",
        "m x k x n", "handwritten", "generated", "hand GF/s", "gen GF/s", "ratio"
    );

    for &(m, k, n) in SHAPES {
        let bh = make_bufs(&ctx, m, k, n);
        let bs = make_bufs(&ctx, m, k, n);

        // Discarded: the first dispatch after pipeline creation pays compilation
        // and power-state ramp.
        time_dispatches(&ctx, &hand, &bh, true);
        time_dispatches(&ctx, &slang, &bs, false);

        let mut t_hand = Vec::with_capacity(ROUNDS);
        let mut t_slang = Vec::with_capacity(ROUNDS);
        for r in 0..ROUNDS {
            // Alternate order so a monotonic drift across the run cannot be
            // attributed to whichever arm happens to run second.
            if r % 2 == 0 {
                t_hand.push(time_dispatches(&ctx, &hand, &bh, true));
                t_slang.push(time_dispatches(&ctx, &slang, &bs, false));
            } else {
                t_slang.push(time_dispatches(&ctx, &slang, &bs, false));
                t_hand.push(time_dispatches(&ctx, &hand, &bh, true));
            }
        }

        let h = median(t_hand);
        let s = median(t_slang);
        let shape = format!("{m}x{k}x{n}");
        println!(
            "{shape:>16}  {h:>12.1}  {s:>12.1}  {:>9.1}  {:>9.1}  {:>7.3}x",
            gflops(m, k, n, h),
            gflops(m, k, n, s),
            h / s
        );
    }

    println!(
        "\nratio > 1.00 means the generated kernel is faster; < 1.00 means the handwritten\n\
         one is. Unlike the softmax bench these dispatches are milliseconds, so a few\n\
         percent here is a real difference rather than timer noise."
    );
}

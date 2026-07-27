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
//! bandwidth-bound". Decode GEMVs here are frequently bound by something else:
//!
//! - `gemv_q4_k` was *instruction*-bound. It took the same wall time as the
//!   dense f32 GEMV while moving 7x fewer bytes; unrolling its dequant loop was
//!   worth 1.42x with no change in bytes moved.
//! - At small `m` every kernel — including the dense f32 control — sits in a
//!   per-dispatch overhead floor, so the bandwidth column there measures the
//!   dispatch, not the kernel.
//!
//! `BASELINE.md`'s "time tracks bytes" result (1.89x bytes -> 2.10x time) came
//! from a Q8_0-vs-Q4_0 A/B. Those two are bandwidth-limited; it does not
//! generalize across the family.
//!
//! Practical consequence: **ablate at large `m`** (65536), where the floor is
//! amortized, and keep the f32 control in the table — it is what distinguishes a
//! harness floor from a kernel defect.
//!
//! ```text
//! cargo run --release --features gpu --example wgpu_gemv_bench
//! ITERS=50 cargo run --release --features gpu --example wgpu_gemv_bench
//! ```

use cera::backend::wgpu::{GpuContext, shaders};
use cera::tensor::DType;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
        DType::Q6K => (256, 210),
        // Dense f32: the control. `gemv_f32` is the best-behaved kernel in the
        // family (106 GB/s as `gemv_f16` in BASELINE.md), so if it shows the
        // same per-dispatch floor the floor is the harness, not the kernel.
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
    (0..total).map(|i| 0x30u8 + (i * 7 % 8) as u8).collect()
}

fn main() -> anyhow::Result<()> {
    let ctx = GpuContext::new()?;
    eprintln!("adapter: {} ({})", ctx.adapter_name, ctx.backend);

    let iters = env_u32("ITERS", 50);

    // `M=..K=..` overrides the shape table with a single custom shape, for
    // sweeping one dimension while the other is held fixed.
    let custom = (std::env::var("M").is_ok() || std::env::var("K").is_ok())
        .then(|| vec![(env_u32("M", 2816), env_u32("K", 1024), "custom")]);

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
    let peak_gbs = std::env::var("PEAK_GBS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(400.0);

    for kern in kernels {
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

            let pipeline = ctx.create_pipeline(kern.shader, kern.entry, kern.entry);
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
            let run = |count: u32| {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                for _ in 0..count {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                }
                ctx.queue.submit(Some(enc.finish()));
                ctx.device.poll(wgpu::Maintain::Wait);
            };

            run(2); // warmup: shader compile + clock ramp
            let t0 = std::time::Instant::now();
            run(iters);
            let per_iter = t0.elapsed().as_secs_f64() / f64::from(iters);

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
    println!("\n* % of {peak_gbs} GB/s (override with PEAK_GBS)");
    Ok(())
}

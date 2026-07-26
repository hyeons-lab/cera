//! Standalone microbenchmark for the wgpu `mul_mat_reg_tile` prefill GEMM.
//!
//! The batched-prefill profile attributes ~98% of wgpu prefill time to this one
//! kernel, so tuning it needs a loop tighter than a full model run. This example
//! dispatches the kernel on the real LFM2 projection shapes and reports achieved
//! GFLOP/s per dtype, with the tile geometry overridable from the environment.
//!
//! ```text
//! cargo run --release -p cera --features gpu --example wgpu_gemm_bench
//! WG_M=32 WG_N=8 ITERS=20 ./target/release/examples/wgpu_gemm_bench
//! ```
//!
//! Only the workgroup dims and TILE_K are tunable. TILE_M/TILE_N are fixed at 4
//! because the kernel hand-unrolls for a 4x4 thread tile — a different value
//! silently computes a fraction of the dispatched tile, so this validates them
//! rather than letting a sweep report a fast wrong answer.
//!
//! `SHADER=<path>` swaps in an experimental kernel with the same four bindings
//! and `MulMatParams` layout. In that mode every result is also diffed against
//! the production kernel on the same inputs, so a rewrite can't be "fast" by
//! computing the wrong thing.

use anyhow::{Context, ensure};
use cera::backend::wgpu::{GpuContext, shaders};

/// One benchmarked matmul: `dst[n, m] = src0[m, k] * src1[n, k]`.
struct Shape {
    label: &'static str,
    m: u32,
    k: u32,
    n: u32,
}

/// The distinct projection shapes in an LFM2.5-350M 512-token prefill.
const SHAPES: &[Shape] = &[
    Shape {
        label: "ffn_gate",
        m: 4608,
        k: 1024,
        n: 512,
    },
    Shape {
        label: "ffn_down",
        m: 1024,
        k: 4608,
        n: 512,
    },
    Shape {
        label: "conv_in",
        m: 3072,
        k: 1024,
        n: 512,
    },
    Shape {
        label: "attn_out",
        m: 1024,
        k: 1024,
        n: 512,
    },
];

struct Variant {
    name: &'static str,
    loader: &'static str,
    /// Packed size in bytes of `elems` weights in this dtype.
    src0_bytes: fn(u64) -> u64,
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "f32",
        loader: "INIT_SRC0_SHMEM_FLOAT",
        src0_bytes: |e| e * 4,
    },
    Variant {
        name: "q4_0",
        loader: "INIT_SRC0_SHMEM_Q4_0",
        src0_bytes: |e| e / 32 * 18,
    },
    Variant {
        name: "q8_0",
        loader: "INIT_SRC0_SHMEM_Q8_0",
        src0_bytes: |e| e / 32 * 34,
    },
    Variant {
        name: "q4_k",
        loader: "INIT_SRC0_SHMEM_Q4_K",
        src0_bytes: |e| e / 256 * 144,
    },
    Variant {
        name: "q6_k",
        loader: "INIT_SRC0_SHMEM_Q6_K",
        src0_bytes: |e| e / 256 * 210,
    },
];

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic pseudo-random packed-quant bytes — a constant fill would let a
/// kernel that drops terms still match the reference.
///
/// Bit 6 is cleared on every byte at an odd offset. All four quant formats store
/// their scales as f16 at an even offset within an even-length block, so every
/// f16 scale's high byte lands at an odd offset (the converse is not true — most
/// odd bytes are quant payload, which this harmlessly perturbs). Clearing bit 6
/// caps the 5-bit exponent at 15, keeping it out of the all-ones Inf/NaN
/// encoding. Without it ~3% of Q4_0 blocks decode to a non-finite scale, which
/// poisons the output and makes the `SHADER=` diff compare NaN against NaN — no
/// check at all.
///
/// Bit 7 — the f16 SIGN — is deliberately preserved. ggml derives Q4_0's scale
/// as `max / -8` and Q6_K's as `max_scale / -128`, so roughly half of all real
/// blocks carry a negative scale. Masking the sign off (an earlier version used
/// `& 0x3F`) made every scale positive and left a candidate kernel that drops
/// the f16 sign diffing clean at `rel_err = 0`.
fn fill_quant_bytes(len: usize) -> Vec<u8> {
    let mut s = 0x12345678u32;
    (0..len)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (s >> 16) as u8;
            if i % 2 == 1 { b & 0xBF } else { b }
        })
        .collect()
}

fn fill_f32(len: usize) -> Vec<f32> {
    let mut s = 0x9E3779B9u32;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 16) as f32 / 32768.0) - 1.0
        })
        .collect()
}

struct Tile {
    wg_m: u32,
    wg_n: u32,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
}

impl Tile {
    fn from_env() -> Self {
        Self {
            wg_m: env_u32("WG_M", 16),
            wg_n: env_u32("WG_N", 16),
            tile_m: env_u32("TILE_M", 4),
            tile_n: env_u32("TILE_N", 4),
            tile_k: env_u32("TILE_K", 32),
        }
    }

    fn defines(&self, v: &Variant) -> Vec<(String, String)> {
        vec![
            (
                "SRC0_INNER_TYPE".into(),
                if v.loader == "INIT_SRC0_SHMEM_FLOAT" {
                    "f32".into()
                } else {
                    "u32".into()
                },
            ),
            (v.loader.into(), String::new()),
            ("WORKGROUP_SIZE_M".into(), format!("{}u", self.wg_m)),
            ("WORKGROUP_SIZE_N".into(), format!("{}u", self.wg_n)),
            ("TILE_M".into(), format!("{}u", self.tile_m)),
            ("TILE_N".into(), format!("{}u", self.tile_n)),
            ("TILE_K".into(), format!("{}u", self.tile_k)),
        ]
    }

    fn grid(&self, s: &Shape) -> (u32, u32, u32) {
        (
            s.m.div_ceil(self.wg_m * self.tile_m),
            s.n.div_ceil(self.wg_n * self.tile_n),
            1,
        )
    }
}

fn build(ctx: &GpuContext, src: &str, tile: &Tile, v: &Variant) -> wgpu::ComputePipeline {
    let defs = tile.defines(v);
    let refs: Vec<(&str, &str)> = defs
        .iter()
        .map(|(k, val)| (k.as_str(), val.as_str()))
        .collect();
    ctx.create_pipeline_with_defines(src, "main", v.name, &refs)
}

fn main() -> anyhow::Result<()> {
    let ctx = GpuContext::new()?;
    let experimental = std::env::var("SHADER").ok();
    let source = match &experimental {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read SHADER={path}"))?,
        None => shaders::MUL_MAT_REG_TILE.to_string(),
    };
    let tile = Tile::from_env();
    ensure!(
        tile.tile_m == 4 && tile.tile_n == 4,
        "TILE_M/TILE_N must be 4: mul_mat_reg_tile.wgsl hand-unrolls a 4x4 thread \
         tile, and any other value computes only part of the dispatched tile \
         (fast, and wrong) — got {}x{}",
        tile.tile_m,
        tile.tile_n,
    );
    ensure!(
        tile.tile_k.is_multiple_of(8),
        "TILE_K must be a multiple of 8: the Q4_0 loader stages 8 consecutive k \
         per thread — got {}",
        tile.tile_k,
    );
    let iters = env_u32("ITERS", 20);
    // Mirrors the production shader's shmem sizing; an experimental kernel may
    // lay its tiles out differently, so this is a guide, not a guarantee.
    let shmem = (tile.tile_k * (tile.wg_m * tile.tile_m + 4)
        + tile.tile_k * (tile.wg_n * tile.tile_n + 4))
        * 4;

    println!(
        "adapter: {} ({})\ntile: WG_M={} WG_N={} TILE_M={} TILE_N={} TILE_K={} \
         -> {} threads/wg, {}x{} out-tile/wg, shmem ~{:.1} KB{}\n",
        ctx.adapter_name,
        ctx.backend,
        tile.wg_m,
        tile.wg_n,
        tile.tile_m,
        tile.tile_n,
        tile.tile_k,
        tile.wg_m * tile.wg_n,
        tile.wg_m * tile.tile_m,
        tile.wg_n * tile.tile_n,
        shmem as f64 / 1024.0,
        experimental
            .as_ref()
            .map(|p| format!("\nshader: {p} (diffed vs production)"))
            .unwrap_or_default(),
    );

    // `ONLY=f32,q4_0` restricts the run — an experimental kernel usually
    // implements one dequant path before the rest are ported.
    let only = std::env::var("ONLY").unwrap_or_default();
    for v in VARIANTS
        .iter()
        .filter(|v| only.is_empty() || only.split(',').any(|o| o == v.name))
    {
        let pipeline = build(&ctx, &source, &tile, v);
        let reference = experimental
            .is_some()
            .then(|| build(&ctx, shaders::MUL_MAT_REG_TILE, &tile, v));

        for s in SHAPES {
            // The f32 variant's src0 is a real f32 buffer, not packed quant
            // bytes: filling it from the byte stream would reinterpret random
            // words as floats, giving a uniform exponent and (but for the
            // incidental bit-6 masking) Inf/NaN.
            let src0 = if v.loader == "INIT_SRC0_SHMEM_FLOAT" {
                ctx.upload_f32(&fill_f32(s.m as usize * s.k as usize), "src0")
            } else {
                ctx.upload_storage(
                    &fill_quant_bytes((v.src0_bytes)(s.m as u64 * s.k as u64) as usize),
                    "src0",
                )
            };
            let src1 = ctx.upload_f32(&fill_f32(s.n as usize * s.k as usize), "src1");
            let params =
                ctx.upload_storage(bytemuck::cast_slice(&[s.m, s.k, s.n, s.k, s.m]), "params");
            let dst_len = s.n as u64 * s.m as u64;

            let dispatch = |pipe: &wgpu::ComputePipeline, dst: &wgpu::Buffer, count: u32| {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src0.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: src1.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dst.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
                let grid = tile.grid(s);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                for _ in 0..count {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
                ctx.queue.submit(Some(enc.finish()));
                ctx.device.poll(wgpu::Maintain::Wait);
            };

            let dst = ctx.create_storage_rw(dst_len * 4, "dst");
            dispatch(&pipeline, &dst, 2); // warmup: shader compile + clock ramp
            let t0 = std::time::Instant::now();
            dispatch(&pipeline, &dst, iters);
            let per_iter = t0.elapsed().as_secs_f64() / iters as f64;

            // Relative error against the production kernel over the WHOLE
            // output. An earlier version sampled the first 64K floats to keep
            // the readback cheap, but dst is column-major with `y_stride = m`,
            // so 64K floats is under 64 columns on every shape here — entirely
            // inside column-tile 0, i.e. only `wg_id.y == 0`. A candidate whose
            // column-tile mapping (`offset_n`, the `wg_linear` split, `sb`
            // indexing) is wrong for any later n-tile diffed clean. The readback
            // only happens in `SHADER=` mode and once per shape, so paying for
            // all of it is the right trade.
            let diff = reference.as_ref().map(|r| {
                let dst_ref = ctx.create_storage_rw(dst_len * 4, "dst_ref");
                dispatch(r, &dst_ref, 1);
                let n = dst_len as usize;
                let a = ctx.download_f32(&dst, n);
                let b = ctx.download_f32(&dst_ref, n);
                // A non-finite anywhere makes the comparison meaningless, not
                // passing: `f32::max` returns the other operand for NaN, so a
                // fold over NaN silently reports the running max and every
                // mismatch disappears. Fail outright instead.
                assert!(
                    a.iter().chain(&b).all(|x| x.is_finite()),
                    "{} {}: non-finite output — the diff validates nothing",
                    v.name,
                    s.label,
                );
                let scale = b.iter().fold(0f32, |acc, x| acc.max(x.abs())).max(1e-6);
                a.iter()
                    .zip(&b)
                    .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()))
                    / scale
            });

            // Both kernels accumulate k in the same order, so a correct
            // candidate is bit-identical — hence `== 0.0` rather than a
            // tolerance. Relax this deliberately (and say why) if a candidate
            // intentionally reorders the reduction. Anything else must stop the
            // sweep rather than scroll past as a printed number: the whole point
            // of `SHADER=` mode is that a rewrite cannot be "fast" by computing
            // the wrong thing.
            if let Some(d) = diff {
                assert!(
                    d == 0.0,
                    "{} {}: rel_err {d:.3e} against the production kernel — the \
                     candidate computes something different",
                    v.name,
                    s.label,
                );
            }

            let gflop = 2.0 * s.m as f64 * s.k as f64 * s.n as f64 / 1e9;
            let grid = tile.grid(s);
            println!(
                "{:4} {:8} m={:5} k={:5} n={:4}  grid={:4}x{:<3}  {:8.3} ms  {:8.1} GFLOP/s{}",
                v.name,
                s.label,
                s.m,
                s.k,
                s.n,
                grid.0,
                grid.1,
                per_iter * 1e3,
                gflop / per_iter,
                diff.map(|d| format!("  rel_err={d:.2e}"))
                    .unwrap_or_default(),
            );
        }
        println!();
    }
    Ok(())
}

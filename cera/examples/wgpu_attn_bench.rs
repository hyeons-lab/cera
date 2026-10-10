//! Throughput and error of the causal prefill attention kernels, on an LFM2-shaped call (16 query
//! heads, 8 KV heads, head_dim 64, unit-variance Q/K/V like the post-norm activations).
//!
//! Times the f32 WGSL kernel in the tree (`attention_prefill_hd64.wgsl`) and any number of SPIR-V
//! variants (`scripts/attn-fp16/gen.py` makes the fp16 ones) on the same buffers, and scores each
//! against an f64 reference computed on the host from the same f16-rounded K/V:
//!
//! ```text
//! python3 scripts/attn-fp16/gen.py /tmp/attn_f16
//! cargo ndk -t arm64-v8a build --release -p cera --features gpu --example wgpu_attn_bench
//! adb push target/aarch64-linux-android/release/examples/wgpu_attn_bench /data/local/tmp/
//! adb push /tmp/attn_f16 /data/local/tmp/attn_f16
//! adb shell /data/local/tmp/wgpu_attn_bench /data/local/tmp/attn_f16/*.spv
//! ```
//!
//! Each dispatch is far below the seconds a phone's hang detector tolerates. A variant's bindings
//! are the kernel's own (q f32, K and V packed f16 words, out f32, 14 parameter words).

use cera::backend::wgpu::{DevicePollExt, GpuContext, shaders};
use std::time::Instant;

const N_HEADS: u32 = 16;
const N_KV: u32 = 8;
const HD: u32 = 64;
const KV_DIM: u32 = N_KV * HD;
const Q_STRIDE: u32 = N_HEADS * HD;

/// Deterministic standard-normal stream (xorshift64 + Box-Muller).
struct Normal(u64);
impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next(&mut self) -> f32 {
        let (u1, u2) = (self.uniform(), self.uniform());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

struct Case {
    n: u32,
    start: u32,
    q: Vec<f32>,
    k: Vec<f32>, // f16-rounded
    v: Vec<f32>,
}

fn make_case(n: u32, start: u32) -> Case {
    let mut rng = Normal(0x9E37_79B9_7F4A_7C15 ^ u64::from(n) << 20 ^ u64::from(start));
    let max_seq = (start + n) as usize;
    // `ATTN_BENCH_QK_SCALE` multiplies Q and K, widening the score distribution (real models have
    // heads with much larger logits than unit-variance data gives).
    let qk: f32 = std::env::var("ATTN_BENCH_QK_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let q: Vec<f32> = (0..n as usize * Q_STRIDE as usize)
        .map(|_| rng.next() * qk)
        .collect();
    let round = |x: f32| half::f16::from_f32(x).to_f32();
    let k: Vec<f32> = (0..max_seq * KV_DIM as usize)
        .map(|_| round(rng.next() * qk))
        .collect();
    let v: Vec<f32> = (0..max_seq * KV_DIM as usize)
        .map(|_| round(rng.next()))
        .collect();
    Case { n, start, q, k, v }
}

fn pack_f16(x: &[f32]) -> Vec<u32> {
    x.chunks(2)
        .map(|p| {
            let lo = half::f16::from_f32(p[0]).to_bits() as u32;
            let hi = half::f16::from_f32(*p.get(1).unwrap_or(&0.0)).to_bits() as u32;
            lo | (hi << 16)
        })
        .collect()
}

/// f64 causal attention on the host.
fn reference(c: &Case) -> Vec<f32> {
    let scale = 1.0 / f64::from(HD).sqrt();
    let mut out = vec![0.0f32; c.n as usize * Q_STRIDE as usize];
    let group = (N_HEADS / N_KV) as usize;
    std::thread::scope(|s| {
        let chunks: Vec<_> = out
            .chunks_mut(Q_STRIDE as usize)
            .enumerate()
            .collect::<Vec<_>>();
        let per = chunks.len().div_ceil(8);
        let mut it = chunks.into_iter();
        let mut handles = Vec::new();
        loop {
            let batch: Vec<_> = it.by_ref().take(per).collect();
            if batch.is_empty() {
                break;
            }
            handles.push(s.spawn(move || {
                for (row, dst) in batch {
                    let pos = c.start as usize + row;
                    for h in 0..N_HEADS as usize {
                        let kvh = h / group;
                        let q = &c.q[row * Q_STRIDE as usize + h * HD as usize..][..HD as usize];
                        let mut scores = Vec::with_capacity(pos + 1);
                        for t in 0..=pos {
                            let k = &c.k[t * KV_DIM as usize + kvh * HD as usize..][..HD as usize];
                            let dot: f64 = q
                                .iter()
                                .zip(k)
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum();
                            scores.push(dot * scale);
                        }
                        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let mut denom = 0.0;
                        for s in &mut scores {
                            *s = (*s - m).exp();
                            denom += *s;
                        }
                        for d in 0..HD as usize {
                            let acc: f64 = scores
                                .iter()
                                .enumerate()
                                .map(|(t, &p)| {
                                    p * f64::from(c.v[t * KV_DIM as usize + kvh * HD as usize + d])
                                })
                                .sum();
                            dst[h * HD as usize + d] = (acc / denom) as f32;
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    out
}

struct Variant {
    name: String,
    pipeline: wgpu::ComputePipeline,
    /// Queries per workgroup (the dispatch's X extent is `ceil(n / queries_per_group)`).
    queries_per_group: u32,
}

fn load_variant(ctx: &GpuContext, path: &str) -> Variant {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bgl"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, true),
            ],
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attn_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(path),
                spirv: Some(std::borrow::Cow::Owned(words)),
                entry_points: std::borrow::Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main"),
                    workgroup_size: (0, 0, 0),
                }]),
                ..Default::default()
            })
    };
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(path),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let name = std::path::Path::new(path)
        .file_stem()
        .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    // a kernel with "q64" in its file name takes 64 queries per workgroup; the rest take 32
    let queries_per_group = if name.contains("q64") { 64 } else { 32 };
    Variant {
        name,
        pipeline,
        queries_per_group,
    }
}

fn main() {
    let ctx = GpuContext::new().expect("no GPU");
    assert!(
        ctx.supports_spirv_passthrough(),
        "needs a Vulkan device that takes SPIR-V passthrough"
    );
    println!("adapter: {} ({})", ctx.adapter_name, ctx.backend);
    let mut variants = vec![Variant {
        name: "f32 wgsl (tree)".into(),
        pipeline: ctx.create_pipeline(
            shaders::ATTENTION_PREFILL_HD64,
            "main",
            "attention_prefill_hd64",
        ),
        queries_per_group: 32,
    }];
    for path in std::env::args().skip(1) {
        variants.push(load_variant(&ctx, &path));
    }

    for &(n, start) in &[(512u32, 0u32), (512, 1536)] {
        let c = make_case(n, start);
        let max_seq = start + n;
        let want = reference(&c);
        let ref_rms =
            (want.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        let q = ctx.upload_f32(&c.q, "attn.q");
        let k = ctx.upload_storage(bytemuck::cast_slice(&pack_f16(&c.k)), "attn.k");
        let v = ctx.upload_storage(bytemuck::cast_slice(&pack_f16(&c.v)), "attn.v");
        let params: [u32; 14] = [
            N_HEADS,
            N_KV,
            HD,
            KV_DIM,
            max_seq,
            (1.0 / (HD as f32).sqrt()).to_bits(),
            start,
            start + n,
            Q_STRIDE,
            Q_STRIDE,
            0,
            n,
            0,
            0,
        ];
        let p = ctx.upload_storage(bytemuck::cast_slice(&params), "attn.params");
        // causal flops: 2 (QK) + 2 (PV) per key per dim
        let keys: f64 = (0..n).map(|i| f64::from(start + i + 1)).sum();
        let flops = 4.0 * f64::from(HD) * keys * f64::from(N_HEADS);
        println!(
            "\nn={n} start={start} ({:.2} GFLOP causal)  reference rms {ref_rms:.4}",
            flops / 1e9
        );
        for var in &variants {
            let out = ctx.create_storage_rw(u64::from(n * Q_STRIDE) * 4, "attn.out");
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &var.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.as_entire_binding(),
                    },
                ],
            });
            let run = |reps: u32| -> f64 {
                let t0 = Instant::now();
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&var.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    for _ in 0..reps {
                        pass.dispatch_workgroups(n.div_ceil(var.queries_per_group), N_HEADS, 1);
                    }
                }
                ctx.queue.submit([enc.finish()]);
                ctx.device.poll_wait();
                t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
            };
            run(2); // warm
            let mut ms: Vec<f64> = (0..7).map(|_| run(4)).collect();
            ms.sort_by(f64::total_cmp);
            let med = ms[3];
            let got = ctx.download_f32(&out, (n * Q_STRIDE) as usize);
            let (mut max_abs, mut sq) = (0.0f64, 0.0f64);
            let mut nan = 0usize;
            for (g, w) in got.iter().zip(&want) {
                if !g.is_finite() {
                    nan += 1;
                    continue;
                }
                let e = f64::from(g - w).abs();
                max_abs = max_abs.max(e);
                sq += e * e;
            }
            let rms = (sq / got.len() as f64).sqrt();
            println!(
                "  {:<22} {:>7.2} ms  {:>5.2} TFLOPS   max_abs_err {:.2e}  rms_err {:.2e} ({:.3}% of rms){}",
                var.name,
                med,
                flops / (med * 1e-3) / 1e12,
                max_abs,
                rms,
                100.0 * rms / ref_rms,
                if nan > 0 {
                    format!("  NON-FINITE x{nan}")
                } else {
                    String::new()
                }
            );
        }
    }
}

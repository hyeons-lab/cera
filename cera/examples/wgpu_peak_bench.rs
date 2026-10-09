//! Peak arithmetic and memory throughput of a Vulkan GPU, through the same SPIR-V passthrough path
//! the production GEMMs use.
//!
//! The prefill GEMMs and attention kernels report achieved TFLOPS, but a number only means
//! something against the hardware's ceiling. This example measures the ceiling with kernels that do
//! nothing but dependent-free FMA chains (fp32 and fp16, 8 and 16 chains per thread), a packed
//! int8 dot product (`SPV_KHR_integer_dot_product`), and a coalesced streaming read:
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --features gpu --example wgpu_peak_bench
//! adb push target/aarch64-linux-android/release/examples/wgpu_peak_bench /data/local/tmp/
//! adb shell /data/local/tmp/wgpu_peak_bench
//! ```
//!
//! Kernels live in `examples/peak/*.slang` (generated, see `peak/gen.py`) with their compiled
//! `.spv` checked in next to them; rebuild one with
//! `slangc k.slang -target spirv -O3 -entry main -stage compute -o k.spv` (slangc 2026.19).
//!
//! fp16 uses SPIR-V `Float16`, which wgpu does not advertise as `SHADER_F16` on Adreno 830 (it
//! needs 16-bit storage as well) but the driver runs; a WGSL kernel cannot use it, a passthrough
//! one can. The integer-dot kernel needs a device extension wgpu does not enable, so it is only
//! attempted with `PEAK_DOT=1` (a device without the feature may lose the context).
//!
//! Each kernel is calibrated to run about 25 ms per dispatch, then timed over 8 dispatches
//! submitted together, 5 rounds, median reported. A dispatch is far below the seconds a phone's
//! hang detector tolerates.

use cera::backend::wgpu::{DevicePollExt, GpuContext};
use std::time::Instant;

const GROUPS: u32 = 8192; // 8192 x 256 = 2M threads: enough to fill any mobile GPU many times over
const THREADS: u64 = GROUPS as u64 * 256;

struct Kernel {
    name: &'static str,
    /// SPIR-V bytes.
    spv: &'static [u8],
    /// FLOPs (or integer ops) per thread per loop iteration; 0 for the bandwidth kernel.
    ops_per_iter: u64,
}

macro_rules! k {
    ($name:literal, $ops:expr) => {
        Kernel {
            name: $name,
            spv: include_bytes!(concat!("peak/", $name, ".spv")),
            ops_per_iter: $ops,
        }
    };
}

fn main() {
    let ctx = GpuContext::new().expect("no GPU");
    assert!(
        ctx.supports_spirv_passthrough(),
        "needs a Vulkan device that takes SPIR-V passthrough"
    );
    println!(
        "adapter: {} ({})   storage binding limit {} MiB",
        ctx.adapter_name,
        ctx.backend,
        ctx.max_storage_buffer_binding_size >> 20
    );

    let fma = [
        k!("fma_f32_c8", 8 * 4 * 2),
        k!("fma_f32_c16", 16 * 4 * 2),
        k!("fma_f16_c8", 8 * 4 * 2),
        k!("fma_f16_c16", 16 * 4 * 2),
    ];
    for kernel in &fma {
        let t = time_kernel(&ctx, kernel, None);
        println!(
            "{:<12} {:>8.3} TFLOPS   ({:.2} ms per dispatch, {} iters)",
            kernel.name, t.rate_t, t.ms, t.iters
        );
    }
    if std::env::var("PEAK_DOT").is_ok_and(|v| v == "1") {
        let t = time_kernel(&ctx, &k!("dot_i8_c8", 8 * 4 * 2), None);
        println!(
            "{:<12} {:>8.3} TOPS     ({:.2} ms per dispatch, {} iters)",
            "dot_i8_c8", t.rate_t, t.ms, t.iters
        );
    }

    // Streaming read: one pass over the largest buffer a binding may hold (capped at 256 MiB).
    let bytes = ctx.max_storage_buffer_binding_size.min(256 << 20);
    let n_vec = (bytes / 16) as u32;
    let src = ctx.create_storage_rw(u64::from(n_vec) * 16, "peak.src");
    let t = time_kernel(&ctx, &k!("bw_read", 0), Some((&src, n_vec)));
    println!(
        "{:<12} {:>8.1} GB/s      ({:.2} ms per {} MiB pass)",
        "bw_read",
        t.rate_t,
        t.ms,
        (u64::from(n_vec) * 16) >> 20
    );
}

struct Timing {
    /// TFLOPS (arithmetic kernels) or GB/s (bandwidth).
    rate_t: f64,
    ms: f64,
    iters: u32,
}

fn time_kernel(ctx: &GpuContext, kernel: &Kernel, bw: Option<(&wgpu::Buffer, u32)>) -> Timing {
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
            label: Some("peak_bgl"),
            entries: &[storage(0, false), storage(1, true), storage(2, true)],
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("peak_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let words: Vec<u32> = kernel
        .spv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(kernel.name),
                spirv: Some(std::borrow::Cow::Owned(words)),
                entry_points: std::borrow::Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main"),
                    workgroup_size: (0, 0, 0), // unused for SPIR-V
                }]),
                ..Default::default()
            })
    };
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(kernel.name),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let out = ctx.create_storage_rw(THREADS * 4, "peak.out");
    let dummy = ctx.create_storage_rw(16, "peak.dummy");
    let src: &wgpu::Buffer = bw.map_or(&dummy, |(b, _)| b);
    let n_vec = bw.map_or(1, |(_, n)| n);

    // params: x, y, iters, n_vec, a, b, c, d  (x = 0.5, y = 0.25: a fixed point of a*x+y, so the
    // chains stay finite; a/b are packed int8 words for the dot kernel, a is the thread count for
    // the bandwidth kernel)
    let run = |iters: u32, reps: u32| -> f64 {
        let params: [u32; 8] = [
            0.5f32.to_bits(),
            0.25f32.to_bits(),
            iters,
            n_vec,
            if bw.is_some() {
                THREADS as u32
            } else {
                0x0102_0304
            },
            0x0201_0403,
            0,
            0,
        ];
        let p = ctx.upload_storage(bytemuck::cast_slice(&params), "peak.params");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src.as_entire_binding(),
                },
            ],
        });
        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            for _ in 0..reps {
                pass.dispatch_workgroups(GROUPS, 1, 1);
            }
        }
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll_wait();
        t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
    };

    // The bandwidth kernel reads the whole buffer once per dispatch: iters = n_vec / threads.
    let (iters, reps) = if bw.is_some() {
        (n_vec.div_ceil(THREADS as u32).max(1), 8)
    } else {
        // Calibrate: double the loop until one dispatch takes about 25 ms.
        let mut iters = 64u32;
        loop {
            let ms = run(iters, 1);
            if ms >= 25.0 || iters >= 1 << 24 {
                break;
            }
            iters = (iters * 2).max((f64::from(iters) * 25.0 / ms.max(0.05)) as u32 / 2 * 2);
            iters = iters.min(1 << 24);
        }
        (iters, 8)
    };
    run(iters, 2); // warm
    let mut ms: Vec<f64> = (0..5).map(|_| run(iters, reps)).collect();
    ms.sort_by(f64::total_cmp);
    let med = ms[2];
    let rate = if bw.is_some() {
        u64::from(n_vec) as f64 * 16.0 / (med * 1e-3) / 1e9
    } else {
        THREADS as f64 * f64::from(iters) * kernel.ops_per_iter as f64 / (med * 1e-3) / 1e12
    };
    Timing {
        rate_t: rate,
        ms: med,
        iters,
    }
}

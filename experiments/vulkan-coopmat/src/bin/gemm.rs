//! int8 GEMM microbench: Slang-authored cooperative-matrix kernel vs a
//! shared-memory-tiled baseline, timed with GPU timestamp queries.
//!
//! C[M,N] = A[M,K] * B[K,N], int8 x int8 -> int32, row-major. Verifies both GPU
//! outputs against a CPU reference, then reports GPU-time / GFLOPS for each.

use ash::vk;
use std::time::Instant;

const COOPMAT_SPV: &[u8] = include_bytes!("../../shaders/gemm_coopmat.spv");
const TILED_SPV: &[u8] = include_bytes!("../../shaders/gemm_tiled.spv");

// (M, K, N) — M multiple of 64, N/K multiples of 16 (coopmat tile constraints).
const SHAPES: &[(u32, u32, u32)] = &[(256, 1024, 1024), (512, 1024, 1024), (512, 1024, 3072)];
const ITERS: usize = 10;
const WARMUP: usize = 2;

fn main() {
    unsafe { run() }
}

unsafe fn run() {
    let entry = ash::Entry::load().expect("load libvulkan");
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let instance = entry
        .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
        .expect("create_instance");

    let pdev = instance.enumerate_physical_devices().unwrap()[0];
    let props = instance.get_physical_device_properties(pdev);
    let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
    let ts_period = props.limits.timestamp_period; // ns per tick

    // Subgroup size (coopmat is subgroup-scoped; the kernel is LocalSize=32).
    let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
    instance.get_physical_device_properties2(pdev, &mut p2);
    println!("device: {name}  subgroupSize={}  tsPeriod={ts_period}ns", sg.subgroup_size);

    // Compute queue family.
    let qfams = instance.get_physical_device_queue_family_properties(pdev);
    let qf = qfams
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .expect("no compute queue") as u32;
    assert!(qfams[qf as usize].timestamp_valid_bits > 0, "queue lacks timestamps");

    // Device with the features coopmat needs.
    let mut f_cm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let mut f_mm = vk::PhysicalDeviceVulkanMemoryModelFeatures::default().vulkan_memory_model(true);
    let mut f_i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_int8(true);
    let mut f_s8 = vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(true);
    let qprio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qf)
        .queue_priorities(&qprio)];
    let ext = [ash::khr::cooperative_matrix::NAME.as_ptr()];
    let device = instance
        .create_device(
            pdev,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_extension_names(&ext)
                .push_next(&mut f_cm)
                .push_next(&mut f_mm)
                .push_next(&mut f_i8)
                .push_next(&mut f_s8),
            None,
        )
        .expect("create_device (coopmat features)");
    let queue = device.get_device_queue(qf, 0);

    let mem_props = instance.get_physical_device_memory_properties(pdev);
    let harness = Harness::new(&device, &mem_props, qf, ts_period);

    let cm_pipe = harness.pipeline(&device, COOPMAT_SPV);
    let tl_pipe = harness.pipeline(&device, TILED_SPV);

    println!("\n{:>5} {:>5} {:>5} | {:>14} {:>10} | {:>14} {:>10} | {:>7}",
        "M", "K", "N", "coopmat_ms", "GFLOP/s", "tiled_ms", "GFLOP/s", "speedup");

    let mut first = true;
    for &(m, k, n) in SHAPES {
        let (a, b) = gen_inputs(m, k, n);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        // CPU reference (once, on the first/smallest shape — 268M MACs).
        let cpu_ref = if first { Some(cpu_gemm(&a, &b, m, k, n)) } else { None };

        let cm = harness.run(&device, queue, &cm_pipe, &a, &b, m, k, n,
            (n / 16, m / 64, 1), cpu_ref.as_deref(), "coopmat");
        let tl = harness.run(&device, queue, &tl_pipe, &a, &b, m, k, n,
            (n.div_ceil(16), m.div_ceil(16), 1), cpu_ref.as_deref(), "tiled");

        let cm_ms = cm * 1e-6;
        let tl_ms = tl * 1e-6;
        println!("{m:>5} {k:>5} {n:>5} | {:>14.4} {:>10.1} | {:>14.4} {:>10.1} | {:>6.2}x",
            cm_ms, flops / cm, tl_ms, flops / tl, tl / cm);
        use std::io::Write;
        let _ = std::io::stdout().flush();
        first = false;
    }

    // (leak the rest — process exits)
}

/// Reusable Vulkan objects (descriptor layout, pipeline layout, cmd pool, query pool).
struct Harness {
    dsl: vk::DescriptorSetLayout,
    pl: vk::PipelineLayout,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    qpool: vk::QueryPool,
    ts_period: f32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl Harness {
    unsafe fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        qf: u32,
        ts_period: f32,
    ) -> Self {
        let bindings: Vec<_> = (0..3)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let pcr = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(12)];
        let dsls = [dsl];
        let pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&dsls)
                    .push_constant_ranges(&pcr),
                None,
            )
            .unwrap();
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        let qpool = device
            .create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
            .unwrap();
        Self {
            dsl,
            pl,
            pool,
            cmd,
            qpool,
            ts_period,
            mem_props: *mem_props,
        }
    }

    unsafe fn pipeline(&self, device: &ash::Device, spv: &[u8]) -> vk::Pipeline {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).unwrap();
        let module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .unwrap();
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(self.pl)],
                None,
            )
            .unwrap()[0]
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn run(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        pipeline: &vk::Pipeline,
        a: &[i8],
        b: &[i8],
        m: u32,
        k: u32,
        n: u32,
        groups: (u32, u32, u32),
        verify: Option<&[i32]>,
        label: &str,
    ) -> f64 {
        let a_buf = self.buffer_i8(device, a);
        let b_buf = self.buffer_i8(device, b);
        let c_buf = self.buffer_i32(device, (m * n) as usize);

        let dpool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(3)]),
                None,
            )
            .unwrap();
        let dsls = [self.dsl];
        let dset = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(dpool)
                    .set_layouts(&dsls),
            )
            .unwrap()[0];
        let infos = [
            vk::DescriptorBufferInfo::default().buffer(a_buf.0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(b_buf.0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(c_buf.0).range(vk::WHOLE_SIZE),
        ];
        let writes: Vec<_> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect();
        device.update_descriptor_sets(&writes, &[]);

        let push = [m, n, k];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();

        // PowerVR compute timestamps are unreliable (report a fixed floor), so time
        // wall-clock over a batch of REP serialized dispatches recorded once. A
        // compute→compute barrier between dispatches prevents overlap, so the total
        // is REP × single-dispatch; the fixed submit/fence latency is amortized by
        // the large REP.
        const REP: u32 = 40;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        device
            .begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
        device.cmd_bind_descriptor_sets(
            self.cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pl,
            0,
            &[dset],
            &[],
        );
        device.cmd_push_constants(
            self.cmd,
            self.pl,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck_bytes(&push),
        );
        for r in 0..REP {
            device.cmd_dispatch(self.cmd, groups.0, groups.1, groups.2);
            if r + 1 < REP {
                device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        }
        device.end_command_buffer(self.cmd).unwrap();

        let submit = |dev: &ash::Device| {
            dev.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd])],
                fence,
            )
            .unwrap();
            dev.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            dev.reset_fences(&[fence]).unwrap();
        };

        for _ in 0..WARMUP {
            submit(device);
        }
        let mut best = f64::MAX;
        for _ in 0..ITERS {
            let t = Instant::now();
            submit(device);
            let per = t.elapsed().as_nanos() as f64 / REP as f64;
            best = best.min(per);
        }

        if let Some(reference) = verify {
            let got = self.read_i32(device, &c_buf, (m * n) as usize);
            let mism = got.iter().zip(reference).filter(|(a, b)| a != b).count();
            if mism == 0 {
                println!("  [{label}] verified OK ({m}x{k}x{n})");
            } else {
                println!("  [{label}] MISMATCH: {mism}/{} elements differ (first: got {} want {})",
                    got.len(),
                    got.iter().zip(reference).find(|(a, b)| a != b).map(|(g, _)| *g).unwrap_or(0),
                    reference.iter().zip(&got).find(|(r, g)| r != g).map(|(r, _)| *r).unwrap_or(0));
            }
        }

        device.destroy_fence(fence, None);
        device.destroy_descriptor_pool(dpool, None);
        self.free(device, a_buf);
        self.free(device, b_buf);
        self.free(device, c_buf);
        best
    }

    unsafe fn buffer_i8(&self, device: &ash::Device, data: &[i8]) -> (vk::Buffer, vk::DeviceMemory) {
        let bytes: &[u8] = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len());
        self.buffer(device, bytes)
    }
    unsafe fn buffer_i32(&self, device: &ash::Device, len: usize) -> (vk::Buffer, vk::DeviceMemory) {
        self.buffer(device, &vec![0u8; len * 4])
    }

    unsafe fn buffer(&self, device: &ash::Device, bytes: &[u8]) -> (vk::Buffer, vk::DeviceMemory) {
        let size = bytes.len().max(4) as u64;
        let buf = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
                None,
            )
            .unwrap();
        let req = device.get_buffer_memory_requirements(buf);
        // Prefer HOST_VISIBLE|COHERENT|DEVICE_LOCAL (UMA); fall back to host-visible.
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let mt = self
            .find_mem(req.memory_type_bits, want)
            .or_else(|| {
                self.find_mem(
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .expect("no host-visible memory type");
        let mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buf, mem, 0).unwrap();
        let ptr = device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).unwrap();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
        device.unmap_memory(mem);
        (buf, mem)
    }

    unsafe fn read_i32(&self, device: &ash::Device, buf: &(vk::Buffer, vk::DeviceMemory), len: usize) -> Vec<i32> {
        let ptr = device.map_memory(buf.1, 0, (len * 4) as u64, vk::MemoryMapFlags::empty()).unwrap();
        let out = std::slice::from_raw_parts(ptr as *const i32, len).to_vec();
        device.unmap_memory(buf.1);
        out
    }

    unsafe fn free(&self, device: &ash::Device, buf: (vk::Buffer, vk::DeviceMemory)) {
        device.destroy_buffer(buf.0, None);
        device.free_memory(buf.1, None);
    }

    fn find_mem(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            (bits & (1 << i)) != 0
                && self.mem_props.memory_types[i as usize].property_flags.contains(flags)
        })
    }
}

fn bytemuck_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn gen_inputs(m: u32, k: u32, n: u32) -> (Vec<i8>, Vec<i8>) {
    let a: Vec<i8> = (0..(m * k)).map(|i| (((i * 7 + 3) % 17) as i32 - 8) as i8).collect();
    let b: Vec<i8> = (0..(k * n)).map(|i| (((i * 5 + 1) % 13) as i32 - 6) as i8).collect();
    (a, b)
}

fn cpu_gemm(a: &[i8], b: &[i8], m: u32, k: u32, n: u32) -> Vec<i32> {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let t = Instant::now();
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = acc;
        }
    }
    eprintln!("  cpu ref {m}x{k}x{n} in {:.1}s", t.elapsed().as_secs_f64());
    c
}

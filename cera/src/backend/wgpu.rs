// wgpu GPU compute backend.
//
// GPU inference: dequantize weights to f32 at load time, run all ops via WGSL
// compute shaders. Full forward pass in a single CommandEncoder — only logits
// are read back to CPU.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::backend::wgsl_pp::Preprocessor;
use crate::tensor::DType;
use half::f16;

/// GPU compute context: device, queue, and optional timestamp profiling.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    pub max_storage_buffer_binding_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub preprocessor: Preprocessor,
    /// Timestamp profiling (None if TIMESTAMP_QUERY not supported).
    pub profiler: Option<GpuProfiler>,
    /// Pre-allocated staging buffer for download_f32. Resized on demand.
    /// `Mutex` (not `RefCell`) so `GpuContext` is `Sync`, which is the
    /// prerequisite for `Arc<dyn Model>: Send + Sync` through the FFI.
    staging: std::sync::Mutex<Option<wgpu::Buffer>>,
    staging_size: std::sync::atomic::AtomicU64,
}

/// A tensor stored on the GPU.
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl GpuTensor {
    /// Return the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return the size of the tensor data in bytes.
    pub fn size_bytes(&self) -> usize {
        let block_size = self.dtype.block_size();
        // Use div_ceil to ensure sufficient buffer size even if not perfectly aligned
        // to block boundaries (though in practice LLM tensors usually are).
        let raw_size = self.numel().div_ceil(block_size) * self.dtype.block_bytes();
        // Ensure 4-byte alignment for compatibility with wgpu copy operations.
        raw_size.div_ceil(4) * 4
    }
}

/// GPU timestamp profiler — records per-dispatch timing.
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    timestamp_period: f32, // nanoseconds per tick
    /// (label, start_idx, end_idx) for each recorded span.
    spans: std::sync::Mutex<Vec<(String, u32, u32)>>,
    next_query: std::sync::atomic::AtomicU32,
    max_queries: u32,
}

/// A GPU→CPU readback whose copy + map request have already been submitted.
///
/// Split from the `.await` so the caller can submit it while holding a lock
/// (e.g. `infer_lock`) — serialising the GPU work against other forwards —
/// and then release the lock before awaiting the map completion. Holding a
/// `std::sync::Mutex` guard across `.await` would trip `await_holding_lock`;
/// the per-call staging buffer isolates the readback so releasing early is safe
/// (the bytes read are the ones copied at submit time, not whatever the GPU
/// buffer holds later).
pub(crate) struct PendingReadback {
    // Only needed to drive the queue on native; on wasm the JS event loop
    // fires the map callback, so the field would be dead.
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    staging: wgpu::Buffer,
    size: u64,
    rx: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl PendingReadback {
    /// Await map completion and return the mapped bytes. Errors (e.g. a lost
    /// GPU device) surface as `anyhow::Error` instead of panicking, so async
    /// callers can propagate them to the JS boundary.
    pub(crate) async fn recv(self) -> Result<Vec<u8>> {
        // Native: drive the queue so the map callback fires before the await
        // resolves (keeps this usable from a blocking executor). wasm: the
        // WebGPU backend ignores `poll`; the await suspends to the JS event
        // loop, which fires the callback.
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);
        self.rx
            .await
            .map_err(|_| anyhow::anyhow!("GPU readback channel closed"))?
            .map_err(|e| anyhow::anyhow!("GPU readback failed: {e:?}"))?;

        let slice = self.staging.slice(0..self.size);
        let data = slice.get_mapped_range();
        let bytes = data.to_vec();
        drop(data);
        self.staging.unmap();
        Ok(bytes)
    }
}

impl GpuContext {
    /// Initialize the GPU: request a high-performance adapter + device
    /// (blocking). Native convenience wrapper around [`Self::new_async`].
    /// Not available on wasm32 — the WebGPU backend can only be driven from
    /// the JS event loop, so `pollster::block_on` would deadlock there; use
    /// [`Self::new_async`].await instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// wasm32 stub: the blocking [`Self::new`] cannot exist on wasm (it would
    /// deadlock the JS event loop). Kept so callers that haven't migrated fail
    /// with a clear message pointing at the async entry point.
    #[cfg(target_arch = "wasm32")]
    pub fn new() -> Result<Self> {
        anyhow::bail!(
            "GpuContext::new() is unavailable on wasm32; use GpuContext::new_async().await"
        )
    }

    /// Initialize the GPU asynchronously: request a high-performance adapter
    /// and device. This is the wasm-compatible entry point — WebGPU's
    /// `requestAdapter` / `requestDevice` resolve on the JS event loop, so
    /// the host must `.await` rather than block. Native callers can use the
    /// blocking [`Self::new`] wrapper.
    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no GPU adapter found")?;

        let adapter_name = adapter.get_info().name.clone();
        let backend = format!("{:?}", adapter.get_info().backend);

        let profile_requested = std::env::var("CERA_GPU_PROFILE").as_deref() == Ok("1");
        let has_timestamps =
            profile_requested && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut features = wgpu::Features::empty();
        if has_timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }

        // Use the adapter's actual limits instead of hardcoding. This avoids
        // failures on GPUs with smaller max_buffer_size (integrated, mobile).
        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cera-gpu"),
                    required_features: features,
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("failed to request GPU device: {e}"))?;

        let profiler = if has_timestamps {
            let max_queries = 512u32; // enough for ~16 layers × ~16 dispatches
            let timestamp_period = queue.get_timestamp_period();
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profiler"),
                ty: wgpu::QueryType::Timestamp,
                count: max_queries,
            });
            let buf_size = (max_queries as u64) * 8; // u64 per timestamp
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profiler-resolve"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profiler-read"),
                size: buf_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            tracing::info!("GPU timestamp profiling enabled (period={timestamp_period}ns/tick)");
            Some(GpuProfiler {
                query_set,
                resolve_buf,
                read_buf,
                timestamp_period,
                spans: std::sync::Mutex::new(Vec::new()),
                next_query: std::sync::atomic::AtomicU32::new(0),
                max_queries,
            })
        } else {
            tracing::info!("GPU timestamp profiling not available");
            None
        };

        let mut preprocessor = Preprocessor::new();
        preprocessor.add_include("common_decls.tmpl", shaders::COMMON_DECLS);
        preprocessor.add_include("mul_mat_decls.tmpl", shaders::MUL_MAT_DECLS);

        tracing::info!(
            adapter = %adapter_name,
            backend = %backend,
            min_subgroup_size = adapter_limits.min_subgroup_size,
            "GPU initialized"
        );

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size as u64,
            min_storage_buffer_offset_alignment: adapter_limits.min_storage_buffer_offset_alignment
                as u64,
            preprocessor,
            profiler,
            staging: std::sync::Mutex::new(None),
            staging_size: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Upload data to a GPU storage buffer.
    pub fn upload_storage(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Upload f32 data to a GPU storage buffer.
    pub fn upload_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.upload_storage(bytemuck::cast_slice(data), label)
    }

    /// Upload f32 data to a GPU storage buffer, converting to f16.
    ///
    /// Uses a chunked approach to avoid materializing the full f16 vector
    /// on the host, reducing peak memory usage.
    pub fn upload_f32_as_f16(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        let byte_size = (data.len() * 2) as u64;
        let aligned_size = byte_size.div_ceil(4) * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Use 1MB chunks for conversion
        let chunk_size = 512 * 1024;
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let f16_chunk: Vec<f16> = chunk.iter().map(|&x| f16::from_f32(x)).collect();
            let chunk_byte_size = (f16_chunk.len() * 2) as u64;
            let aligned_chunk_size = chunk_byte_size.div_ceil(4) * 4;
            if aligned_chunk_size > chunk_byte_size {
                let mut padded = f16_chunk;
                padded.push(f16::ZERO);
                self.queue.write_buffer(
                    &buffer,
                    (i * chunk_size * 2) as u64,
                    bytemuck::cast_slice(&padded),
                );
            } else {
                self.queue.write_buffer(
                    &buffer,
                    (i * chunk_size * 2) as u64,
                    bytemuck::cast_slice(&f16_chunk),
                );
            }
        }
        buffer
    }

    /// Upload f16 data to a GPU storage buffer.
    pub fn upload_f16(&self, data: &[f16], label: &str) -> wgpu::Buffer {
        let size = (data.len() * 2) as u64;
        let aligned_size = size.div_ceil(4) * 4;
        let buffer = self.create_storage_rw(aligned_size, label);
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }

    /// Create a zeroed GPU buffer with read-write storage usage.
    pub fn create_storage_rw(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Read f32 data back from a GPU buffer (blocking). Reuses a cached
    /// staging buffer to avoid per-token allocation.
    pub fn download_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        use std::sync::atomic::Ordering;
        let size = (count * std::mem::size_of::<f32>()) as u64;
        // Grow staging buffer if needed (typically allocated once for
        // vocab_size). Size check + possible re-allocation happen under
        // a single mutex acquisition so two racing callers can't both
        // hit the !sufficient branch and reallocate twice.
        let staging_guard = {
            let mut guard = self.staging.lock().expect("staging mutex poisoned");
            if guard.as_ref().map(|b| b.size() < size).unwrap_or(true) {
                *guard = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging-download"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.staging_size.store(size, Ordering::Relaxed);
            }
            guard
        };
        let staging = staging_guard.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Map only the requested `0..size` byte range, not the full
        // staging buffer. The cached staging is sized to the largest
        // historical request, so `slice(..)` would map+copy that
        // entire size every call (e.g. a small `count` after a prior
        // `vocab_size` download would still pay the vocab-sized cost
        // and stale tail bytes would leak into the returned `Vec`).
        let slice = staging.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Submit a GPU→CPU readback of `size` bytes and return a
    /// [`PendingReadback`] to `.await` — the wasm/WebGPU-compatible analog of
    /// the blocking `download_*` helpers. WebGPU can only be driven from the JS
    /// event loop, so completion is awaited via a oneshot channel instead of
    /// blocking on `device.poll(Maintain::Wait)` + `mpsc::recv`.
    ///
    /// The copy + map request are issued here (synchronously), so a caller can
    /// invoke this under a lock and await the result after releasing it.
    /// Allocates a fresh staging buffer per call (vs. the cached staging the
    /// sync path reuses) — per-token readback alloc is fine for the current
    /// prototype (a staging-reuse perf pass is a follow-up).
    pub(crate) fn begin_download(&self, buffer: &wgpu::Buffer, size: u64) -> PendingReadback {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-download-async"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download-async"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = futures_channel::oneshot::channel();
        staging
            .slice(0..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        PendingReadback {
            #[cfg(not(target_arch = "wasm32"))]
            device: self.device.clone(),
            staging,
            size,
            rx,
        }
    }

    /// Async f32 readback. See `Self::begin_download`.
    pub async fn download_f32_async(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>> {
        let size = (count * std::mem::size_of::<f32>()) as u64;
        let bytes = self.begin_download(buffer, size).recv().await?;
        // Copy into a properly aligned Vec<f32> rather than `cast_slice`-ing the
        // 1-byte-aligned Vec<u8> (which panics when the allocation isn't
        // 4-aligned).
        let mut out = vec![0.0f32; count];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes);
        Ok(out)
    }

    /// Async u32 readback (argmax token id). See `Self::begin_download`.
    pub async fn download_u32_async(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<u32>> {
        let size = (count * std::mem::size_of::<u32>()) as u64;
        let bytes = self.begin_download(buffer, size).recv().await?;
        // Aligned copy — see `download_f32_async`.
        let mut out = vec![0u32; count];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes);
        Ok(out)
    }

    /// Read u32 data back from a GPU buffer (blocking). Mirrors
    /// `download_f32` but reinterprets the staging bytes as `u32`.
    /// Used by the argmax kernel which writes `out: array<u32>`.
    pub fn download_u32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        use std::sync::atomic::Ordering;
        let size = (count * std::mem::size_of::<u32>()) as u64;
        let staging_guard = {
            let mut guard = self.staging.lock().expect("staging mutex poisoned");
            if guard.as_ref().map(|b| b.size() < size).unwrap_or(true) {
                *guard = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging-download"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.staging_size.store(size, Ordering::Relaxed);
            }
            guard
        };
        let staging = staging_guard.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_u32"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Map only the requested `0..size` range — see download_f32
        // above for the same reasoning. With a vocab-sized cached
        // staging buffer, mapping the full range turns the 4-byte
        // argmax readback into a vocab-sized copy on every greedy
        // step, defeating the optimization.
        let slice = staging.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Read f16 data back from a GPU buffer and convert to f32 (blocking).
    pub fn download_f16_as_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        use std::sync::atomic::Ordering;
        let size = (count * std::mem::size_of::<f16>()) as u64;
        // copy_buffer_to_buffer requires 4-byte alignment for size and offsets.
        let aligned_size = size.div_ceil(4) * 4;

        let staging_guard = {
            let mut guard = self.staging.lock().expect("staging mutex poisoned");
            if guard
                .as_ref()
                .map(|b| b.size() < aligned_size)
                .unwrap_or(true)
            {
                *guard = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging-download"),
                    size: aligned_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.staging_size.store(aligned_size, Ordering::Relaxed);
            }
            guard
        };
        let staging = staging_guard.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_f16"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, aligned_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(0..aligned_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        // Slicing to exact byte count before casting to handle potential 2-byte padding.
        let f16_data: &[f16] = bytemuck::cast_slice(&data[0..size as usize]);
        let result: Vec<f32> = f16_data.iter().map(|&x| x.to_f32()).collect();
        drop(data);
        staging.unmap();
        result
    }

    /// Create a compute pipeline from WGSL source.
    pub fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        self.create_pipeline_with_defines(shader_source, entry_point, label, &[])
    }

    /// Create a compute pipeline from WGSL source with preprocessor defines.
    pub fn create_pipeline_with_defines(
        &self,
        shader_source: &str,
        entry_point: &str,
        label: &str,
        defines: &[(&str, &str)],
    ) -> wgpu::ComputePipeline {
        let preprocessed = self
            .preprocessor
            .preprocess(shader_source, defines)
            .with_context(|| format!("failed to preprocess shader: {label}"))
            .expect("shader preprocessing failed");

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(preprocessed.into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // auto-infer from shader
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Get timestamp_writes for a compute pass (if profiling enabled).
    /// Returns (begin_query_idx, end_query_idx) for later resolution.
    pub fn begin_profile_span(&self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        use std::sync::atomic::Ordering;
        let profiler = self.profiler.as_ref()?;
        let idx = profiler.next_query.load(Ordering::Relaxed);
        if idx + 2 > profiler.max_queries {
            return None; // out of query slots
        }
        profiler.next_query.store(idx + 2, Ordering::Relaxed);
        profiler
            .spans
            .lock()
            .expect("profiler mutex poisoned")
            .push((label.to_string(), idx, idx + 1));
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &profiler.query_set,
            beginning_of_pass_write_index: Some(idx),
            end_of_pass_write_index: Some(idx + 1),
        })
    }

    /// Reset profiler for a new forward pass.
    pub fn reset_profiler(&self) {
        use std::sync::atomic::Ordering;
        if let Some(profiler) = &self.profiler {
            profiler.next_query.store(0, Ordering::Relaxed);
            profiler
                .spans
                .lock()
                .expect("profiler mutex poisoned")
                .clear();
        }
    }

    /// Resolve timestamps and print per-span timings.
    pub fn finish_profiler(&self) {
        use std::sync::atomic::Ordering;
        let profiler = match &self.profiler {
            Some(p) => p,
            None => return,
        };
        let n_queries = profiler.next_query.load(Ordering::Relaxed);
        if n_queries == 0 {
            return;
        }

        // Resolve queries → resolve_buf, then copy → read_buf
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.resolve_query_set(&profiler.query_set, 0..n_queries, &profiler.resolve_buf, 0);
        enc.copy_buffer_to_buffer(
            &profiler.resolve_buf,
            0,
            &profiler.read_buf,
            0,
            (n_queries as u64) * 8,
        );
        self.queue.submit(Some(enc.finish()));

        let slice = profiler.read_buf.slice(..((n_queries as u64) * 8));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let period_ns = profiler.timestamp_period as f64;
        let spans = profiler.spans.lock().expect("profiler mutex poisoned");

        // Aggregate by label
        let mut totals: std::collections::HashMap<String, (f64, usize)> =
            std::collections::HashMap::new();
        for (label, start_idx, end_idx) in spans.iter() {
            let start = timestamps[*start_idx as usize];
            let end = timestamps[*end_idx as usize];
            let us = (end.wrapping_sub(start)) as f64 * period_ns / 1000.0;
            let entry = totals.entry(label.clone()).or_insert((0.0, 0));
            entry.0 += us;
            entry.1 += 1;
        }

        let mut sorted: Vec<_> = totals.into_iter().collect();
        sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        let total_us: f64 = sorted.iter().map(|(_, (us, _))| us).sum();

        eprintln!("── GPU Profile ({total_us:.0}µs total) ──");
        for (label, (us, count)) in &sorted {
            let pct = us / total_us * 100.0;
            eprintln!("  {label:20} {us:8.0}µs ({count:3}×) {pct:5.1}%");
        }

        drop(data);
        profiler.read_buf.unmap();
    }
}

// ── Shaders (embedded at compile time) ─────────────────────────────────────

pub mod shaders {
    pub const COMMON_DECLS: &str = include_str!("shaders/common_decls.tmpl");
    pub const MUL_MAT_DECLS: &str = include_str!("shaders/mul_mat_decls.tmpl");
    pub const MUL_MAT_REG_TILE: &str = include_str!("shaders/mul_mat_reg_tile.wgsl");
    pub const GEMV_F32: &str = include_str!("shaders/gemv_f32.wgsl");
    pub const GEMM_F32: &str = include_str!("shaders/gemm_f32.wgsl");
    pub const GEMV_Q4_0: &str = include_str!("shaders/gemv_q4_0.wgsl");
    pub const GEMV_Q4_0_FAST: &str = include_str!("shaders/gemv_q4_0_fast.wgsl");
    pub const GEMV_Q4_K: &str = include_str!("shaders/gemv_q4_k.wgsl");
    pub const GEMV_Q5_K: &str = include_str!("shaders/gemv_q5_k.wgsl");
    pub const GEMV_Q6_K: &str = include_str!("shaders/gemv_q6_k.wgsl");
    pub const GEMV_Q8_0: &str = include_str!("shaders/gemv_q8_0.wgsl");
    pub const ELEMENTWISE: &str = include_str!("shaders/elementwise.wgsl");
    pub const SCALE_F32: &str = include_str!("shaders/scale_f32.wgsl");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.wgsl");
    pub const RMSNORM_BATCH: &str = include_str!("shaders/rmsnorm_batch.wgsl");
    pub const QK_NORM_ROPE_BATCH: &str = include_str!("shaders/qk_norm_rope_batch.wgsl");
    pub const CONV1D_FUSED_BATCH: &str = include_str!("shaders/conv1d_fused_batch.wgsl");
    pub const GEMM_Q4_0: &str = include_str!("shaders/gemm_q4_0.wgsl");
    pub const GEMM_Q4_K: &str = include_str!("shaders/gemm_q4_k.wgsl");
    pub const GEMM_Q8_0: &str = include_str!("shaders/gemm_q8_0.wgsl");
    pub const PER_HEAD_RMSNORM: &str = include_str!("shaders/per_head_rmsnorm.wgsl");
    pub const SOFTMAX: &str = include_str!("shaders/softmax.wgsl");
    pub const ARGMAX_F32: &str = include_str!("shaders/argmax_f32.wgsl");
    pub const ROPE: &str = include_str!("shaders/rope.wgsl");
    pub const KV_SHIFT: &str = include_str!("shaders/kv_shift.wgsl");
    pub const ATTENTION: &str = include_str!("shaders/attention.wgsl");
    pub const ATTENTION_PREFILL: &str = include_str!("shaders/attention_prefill.wgsl");
    pub const CONV1D: &str = include_str!("shaders/conv1d.wgsl");
    pub const CONV1D_FUSED: &str = include_str!("shaders/conv1d_fused.wgsl");
    // Vision-encoder (ViT) kernels.
    pub const LAYERNORM_BATCH: &str = include_str!("shaders/layernorm_batch.wgsl");
    pub const GELU: &str = include_str!("shaders/gelu.wgsl");
    pub const BIAS_ADD: &str = include_str!("shaders/bias_add.wgsl");
    pub const VIT_ATTENTION: &str = include_str!("shaders/vit_attention.wgsl");
    pub const VIT_ATTENTION_TILED: &str = include_str!("shaders/vit_attention_tiled.wgsl");
}

/// Maximum workgroups per grid dimension. Mirrors `MAX_WG` in
/// `shaders/common_decls.tmpl` — the two MUST stay equal: the kernel's `get_wid`
/// recovers the flat workgroup index as `wid.x + wid.y * MAX_WG`, which only
/// tiles correctly when the host pins the X extent to exactly this value once
/// the grid spills into Y. 65535 is the WebGPU `maxComputeWorkgroupsPerDimension`
/// floor (D3D12 / many Vulkan drivers).
pub const MAX_WG: u32 = 65535;

/// Flatten a 256-thread-per-workgroup dispatch of `total_threads` into a grid
/// that respects the [`MAX_WG`] per-dimension cap. Returns `(x, y, 1)` with the
/// X extent pinned to exactly `MAX_WG` whenever the count spills into Y, so the
/// kernel's `get_wid = wid.x + wid.y * MAX_WG` recovers a gap-free, overlap-free
/// linear index over `[0, ceil(total_threads / 256))`. Shared by the production
/// `kv_shift` dispatch and the `wgpu_kv_shift_oracle` test so the two can't drift.
pub fn kv_shift_workgroups(total_threads: u32) -> (u32, u32, u32) {
    let wg = total_threads.div_ceil(256);
    (wg.min(MAX_WG), wg.div_ceil(MAX_WG), 1)
}

/// Typed parameters for the wgpu `kv_shift` WGSL kernel (`shaders/kv_shift.wgsl`).
///
/// Marshalled to an 8-element `array<u32>` storage buffer via
/// [`Self::to_u32_array`]. Named fields are the single source of truth for *this
/// (wgpu) kernel's* positional `params[i]` reads — keep this struct, the kernel's
/// `params` unpacking, and the kernel's header comment in lockstep. Used by both
/// the production shift (`GpuLfm2Model::encode_kv_shift_layers`) and the
/// `wgpu_kv_shift_oracle` test so the wgpu layout cannot drift between them. (The
/// Metal backend has its own `KParams`; this is only the wgpu-side definition.)
#[derive(Copy, Clone)]
pub struct KvShiftParams {
    /// Cells `[0, n_keep)` are kept verbatim; the shift drops `[n_keep, n_keep+shift)`.
    pub n_keep: u32,
    /// Number of cells dropped; the RoPE delta applied to retained K is `-shift`.
    pub shift: u32,
    /// `new_seq_len - n_keep` — count of retained (re-rotated) cells.
    pub retained: u32,
    /// KV heads in this layer (GQA-aware; can differ per layer).
    pub n_kv_heads: u32,
    /// Per-head dimension (RoPE pairs span `head_dim/2`).
    pub head_dim: u32,
    /// `rope_theta.to_bits()` — the RoPE frequency base, as f32 bits.
    pub freq_base_bits: u32,
    /// RoPE pair layout: 0 = NeoX (split-halves), 1 = NORM (interleaved).
    pub rope_type: u32,
    /// 1 when `freq_factors` holds real Llama-3 factors, 0 for the dummy buffer.
    pub has_freq_factors: u32,
}

impl KvShiftParams {
    /// Flatten to the `array<u32, 8>` layout the kernel reads positionally.
    pub fn to_u32_array(self) -> [u32; 8] {
        [
            self.n_keep,
            self.shift,
            self.retained,
            self.n_kv_heads,
            self.head_dim,
            self.freq_base_bits,
            self.rope_type,
            self.has_freq_factors,
        ]
    }

    /// One kernel thread per (retained cell, KV head, RoPE pair) — matches the
    /// kernel's `total = retained * n_kv_heads * (head_dim / 2)`.
    pub fn total_threads(self) -> u32 {
        self.retained * self.n_kv_heads * (self.head_dim / 2)
    }

    /// 2-D-flattened workgroup grid for dispatching the `kv_shift` kernel over
    /// [`Self::total_threads`]; see [`kv_shift_workgroups`].
    pub fn dispatch_dims(self) -> (u32, u32, u32) {
        kv_shift_workgroups(self.total_threads())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_shift_workgroups_recovers_flat_index_bijectively() {
        // The kernel recovers its flat workgroup index as `wid.x + wid.y*MAX_WG`.
        // Verify the host grid sizing makes that a gap-free, overlap-free cover
        // of `[0, ceil(total/256))` — INCLUDING totals large enough to spill into
        // Y (`wid.y > 0`), the exact path the 2-D flatten fix added and the one
        // the GPU oracle can't afford to allocate a buffer for. Runs on CPU.
        let totals = [
            1u32,
            255,
            256,
            257,
            304,                    // the oracle's total → grid (2,1,1), Y == 1
            MAX_WG * 256,           // exactly MAX_WG workgroups, still Y == 1
            MAX_WG * 256 + 1,       // first total that spills into Y
            (MAX_WG + 7) * 256,     // small Y spill, non-multiple of MAX_WG
            3 * MAX_WG * 256 - 100, // ~3 rows in Y
        ];
        for &total in &totals {
            let wg = total.div_ceil(256);
            let (gx, gy, gz) = kv_shift_workgroups(total);
            assert_eq!(gz, 1, "z extent is always 1 (total={total})");
            // The invariant the kernel relies on: once the grid spills into Y,
            // X must be exactly MAX_WG or `get_wid`'s stride mis-tiles.
            if gy > 1 {
                assert_eq!(
                    gx, MAX_WG,
                    "X must be pinned to MAX_WG when Y>1 (total={total})"
                );
            }
            assert!(
                (gx as u64) * (gy as u64) >= wg as u64,
                "grid {gx}x{gy} under-covers wg={wg} (total={total})",
            );
            // Every needed flat index in [0, wg) maps to a workgroup the grid
            // actually dispatches, and flat->(x,y)->flat round-trips. O(wg).
            for fw in 0..wg {
                let x = fw % MAX_WG;
                let y = fw / MAX_WG;
                assert!(
                    x < gx && y < gy,
                    "flat {fw} maps to ({x},{y}) outside grid {gx}x{gy} (total={total})",
                );
                assert_eq!(
                    x + y * MAX_WG,
                    fw,
                    "get_wid recovery is not the inverse (total={total})"
                );
            }
        }
    }

    #[test]
    fn kv_shift_params_dispatch_dims_match_total_threads() {
        // Pin `total_threads` to the kernel's per-thread granularity and confirm
        // the typed params route through the shared `kv_shift_workgroups` helper.
        let p = KvShiftParams {
            n_keep: 2,
            shift: 3,
            retained: 19,
            n_kv_heads: 2,
            head_dim: 16,
            freq_base_bits: 10_000.0f32.to_bits(),
            rope_type: 0,
            has_freq_factors: 0,
        };
        assert_eq!(p.total_threads(), 19 * 2 * (16 / 2)); // 304
        assert_eq!(p.dispatch_dims(), kv_shift_workgroups(p.total_threads()));
        assert_eq!(p.dispatch_dims(), (2, 1, 1));
    }

    #[test]
    fn test_gpu_context_init() {
        let ctx = GpuContext::new();
        match ctx {
            Ok(ctx) => {
                println!("GPU: {} ({})", ctx.adapter_name, ctx.backend);
                assert!(!ctx.adapter_name.is_empty());
            }
            Err(e) => {
                println!("No GPU available (expected in CI): {e}");
            }
        }
    }

    #[test]
    fn test_gpu_upload_download_roundtrip() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return, // skip if no GPU
        };

        // Use odd number of elements to test alignment/padding logic.
        let data: Vec<f32> = (0..257).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data, "test");
        let result = ctx.download_f32(&buf, data.len());
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_async_download_roundtrip() {
        // Validates the wasm-facing async readback primitives on native by
        // driving the futures with pollster. Same data path as the blocking
        // roundtrip above, through `new_async` + `download_f32_async`.
        let ctx = match pollster::block_on(GpuContext::new_async()) {
            Ok(ctx) => ctx,
            Err(_) => return, // skip if no GPU
        };
        let data: Vec<f32> = (0..257).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32(&data, "test_async");
        let result = pollster::block_on(ctx.download_f32_async(&buf, data.len()))
            .expect("async readback failed");
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_f16_roundtrip() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return, // skip if no GPU
        };

        // Use odd number of elements to test alignment/padding logic.
        let data: Vec<f32> = (0..257).map(|i| i as f32 * 0.1).collect();
        let buf = ctx.upload_f32_as_f16(&data, "test_f16");
        let result = ctx.download_f16_as_f32(&buf, data.len());

        for i in 0..data.len() {
            let diff = (data[i] - result[i]).abs();
            // F16 precision is limited, relative error ~1e-3.
            // For values up to 25.6, absolute error can be up to ~0.02.
            assert!(
                diff < 2e-2,
                "f16 mismatch at {i}: {} vs {}",
                data[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemv_f32() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Small 4×8 matrix × 8-element vector
        let m = 4u32;
        let k = 8u32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.5).collect();

        // CPU reference
        let mut expected = vec![0.0f32; m as usize];
        for i in 0..m as usize {
            for j in 0..k as usize {
                expected[i] += a[i * k as usize + j] * x[j];
            }
        }

        // GPU
        let a_buf = ctx.upload_f32(&a, "A");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv_f32"),
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

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv_f32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per row (simple V1)
            pass.dispatch_workgroups(m, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);

        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-3,
                "GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    // ── ViT vision-encoder kernel parity tests ──────────────────────────────
    //
    // Each validates a new shader (layernorm_batch, gelu, bias_add,
    // vit_attention) against its CPU reference in `backend::cpu` /
    // hand-rolled math. All skip cleanly when no GPU is present (CI).

    /// Dispatch a single compute pipeline over `bufs` (in binding order) and
    /// `workgroups`, returning nothing — caller reads back via `download_f32`.
    fn run_kernel(
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }

    #[test]
    fn test_gpu_layernorm_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let rows = 5usize;
        let dim = 320usize; // not a multiple of 256, > one wave
        let eps = 1e-6f32;
        let src: Vec<f32> = (0..rows * dim)
            .map(|i| ((i * 31 + 7) % 197) as f32 * 0.03 - 2.9)
            .collect();
        let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..dim).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();

        // CPU reference (per row).
        let mut expected = src.clone();
        for r in 0..rows {
            crate::backend::cpu::layer_norm_inplace(
                &mut expected[r * dim..(r + 1) * dim],
                &weight,
                &bias,
                eps,
            );
        }

        let src_buf = ctx.upload_f32(&src, "src");
        let dst_buf = ctx.create_storage_rw((rows * dim * 4) as u64, "dst");
        let w_buf = ctx.upload_f32(&weight, "w");
        let b_buf = ctx.upload_f32(&bias, "b");
        let params = [dim as u32, eps.to_bits(), dim as u32, dim as u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(
            shaders::LAYERNORM_BATCH,
            "layernorm_batch",
            "layernorm_batch",
        );
        run_kernel(
            &ctx,
            &pipeline,
            &[&src_buf, &dst_buf, &w_buf, &b_buf, &p_buf],
            (rows as u32, 1, 1),
        );

        let result = ctx.download_f32(&dst_buf, rows * dim);
        for i in 0..rows * dim {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-3,
                "layernorm mismatch at {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gelu_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Span [-25, 25): includes large |x| where a naive GPU tanh
        // (exp(2a)/...) overflows to NaN via gelu's cubic term — the clamp in
        // gelu.wgsl must keep parity with the CPU's saturating f32::tanh.
        let n = 1000usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.05).collect();

        let mut expected = x.clone();
        crate::backend::cpu::gelu_inplace(&mut expected);

        let x_buf = ctx.upload_f32(&x, "x");
        let params = [n as u32, 0u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GELU, "gelu_inplace", "gelu");
        run_kernel(
            &ctx,
            &pipeline,
            &[&x_buf, &p_buf],
            (n.div_ceil(256) as u32, 1, 1),
        );

        let result = ctx.download_f32(&x_buf, n);
        for i in 0..n {
            assert!(
                result[i].is_finite(),
                "gelu produced non-finite at {i} (x={}): {} — tanh overflow?",
                x[i],
                result[i]
            );
            let diff = (expected[i] - result[i]).abs();
            // tanh on GPU vs CPU differs slightly; activation magnitudes ~O(1).
            assert!(
                diff < 1e-4,
                "gelu mismatch at {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_bias_add_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let rows = 7usize;
        let dim = 130usize;
        let x: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.05 - 3.0).collect();

        let mut expected = x.clone();
        for r in 0..rows {
            for j in 0..dim {
                expected[r * dim + j] += bias[j];
            }
        }

        let x_buf = ctx.upload_f32(&x, "x");
        let b_buf = ctx.upload_f32(&bias, "bias");
        let total = (rows * dim) as u32;
        let params = [total, dim as u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::BIAS_ADD, "bias_add", "bias_add");
        run_kernel(
            &ctx,
            &pipeline,
            &[&x_buf, &b_buf, &p_buf],
            ((total as usize).div_ceil(256) as u32, 1, 1),
        );

        let result = ctx.download_f32(&x_buf, rows * dim);
        for i in 0..rows * dim {
            assert!(
                (expected[i] - result[i]).abs() < 1e-5,
                "bias_add mismatch at {i}: cpu={}, gpu={}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_vit_attention_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let tokens = 20usize;
        let n_head = 3usize;
        let head_dim = 8usize;
        let dim = n_head * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mk = |seed: usize| -> Vec<f32> {
            (0..tokens * dim)
                .map(|i| (((i + seed) * 37 + 11) % 101) as f32 * 0.02 - 1.0)
                .collect()
        };
        let q = mk(1);
        let k = mk(2);
        let v = mk(3);

        // CPU reference: bidirectional softmax(QKᵀ·scale)·V per head.
        let mut expected = vec![0.0f32; tokens * dim];
        for h in 0..n_head {
            for qi in 0..tokens {
                let q_off = qi * dim + h * head_dim;
                let mut scores = vec![0.0f32; tokens];
                for (ki, si) in scores.iter_mut().enumerate() {
                    let k_off = ki * dim + h * head_dim;
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q[q_off + d] * k[k_off + d];
                    }
                    *si = s * scale;
                }
                crate::backend::cpu::softmax_inplace(&mut scores);
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..tokens {
                        acc += scores[ki] * v[ki * dim + h * head_dim + d];
                    }
                    expected[q_off + d] = acc;
                }
            }
        }

        let q_buf = ctx.upload_f32(&q, "q");
        let k_buf = ctx.upload_f32(&k, "k");
        let v_buf = ctx.upload_f32(&v, "v");
        let out_buf = ctx.create_storage_rw((tokens * dim * 4) as u64, "out");
        let params = [
            tokens as u32,
            n_head as u32,
            head_dim as u32,
            scale.to_bits(),
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline =
            ctx.create_pipeline(shaders::VIT_ATTENTION, "vit_attention", "vit_attention");
        run_kernel(
            &ctx,
            &pipeline,
            &[&q_buf, &k_buf, &v_buf, &out_buf, &p_buf],
            (tokens as u32, n_head as u32, 1),
        );

        let result = ctx.download_f32(&out_buf, tokens * dim);
        for i in 0..tokens * dim {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-4,
                "vit_attention mismatch at {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    /// Helper: pack `m × k` f32 weights into Q4_0 block layout (row-major,
    /// 18 bytes per 32-element block: f16 scale + 16 packed nibbles).
    fn quantize_q4_0_for_test(m: usize, k: usize, weights: &[f32]) -> Vec<u8> {
        let mut q4_bytes: Vec<u8> = Vec::with_capacity(m * (k / 32) * 18);
        for row in 0..m {
            for b in 0..(k / 32) {
                let start = row * k + b * 32;
                let chunk = &weights[start..start + 32];
                let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 7.0;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let d_bits = half::f16::from_f32(scale).to_bits();
                q4_bytes.push((d_bits & 0xFF) as u8);
                q4_bytes.push((d_bits >> 8) as u8);
                for qi in 0..16 {
                    let lo = ((chunk[qi] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    let hi = ((chunk[qi + 16] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    q4_bytes.push(lo | (hi << 4));
                }
            }
        }
        q4_bytes
    }

    /// Helper: CPU reference for Q4_0 matmul (mirrors the shader's
    /// dequantize-then-multiply path).
    fn cpu_matmul_q4_0(m: usize, k: usize, n: usize, q4_bytes: &[u8], x_batch: &[f32]) -> Vec<f32> {
        let mut expected = vec![0.0f32; n * m];
        for t in 0..n {
            let x_slice = &x_batch[t * k..(t + 1) * k];
            for row in 0..m {
                let mut acc = 0.0f32;
                for b in 0..(k / 32) {
                    let block_off = (row * (k / 32) + b) * 18;
                    let d_bits = u16::from_le_bytes([q4_bytes[block_off], q4_bytes[block_off + 1]]);
                    let delta = half::f16::from_bits(d_bits).to_f32();
                    for qi in 0..16 {
                        let byte = q4_bytes[block_off + 2 + qi];
                        let lo = (byte & 0xF) as f32 - 8.0;
                        let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                        acc += lo * delta * x_slice[b * 32 + qi];
                        acc += hi * delta * x_slice[b * 32 + qi + 16];
                    }
                }
                expected[t * m + row] = acc;
            }
        }
        expected
    }

    #[test]
    fn test_gpu_mul_mat_tile_q4_0_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m: u32 = 32;
        let k: u32 = 128;
        let n: u32 = 16; // tokens

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let q4_bytes = quantize_q4_0_for_test(m as usize, k as usize, &weights_f32);

        let mut x_batch: Vec<f32> = Vec::with_capacity((n * k) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 64.0)) * 0.05);
            }
        }

        let expected = cpu_matmul_q4_0(m as usize, k as usize, n as usize, &q4_bytes, &x_batch);

        // GPU run
        let a_buf = ctx.upload_storage(&q4_bytes, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * m) as u64) * 4, "y_batch");

        // MulMatParams: m, k, n, x_stride, y_stride.
        let params: [u32; 5] = [m, k, n, k, m];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline_with_defines(
            shaders::MUL_MAT_REG_TILE,
            "main",
            "mul_mat_q4_0_tile_test",
            &[
                ("VEC", ""),
                ("SRC0_INNER_TYPE", "u32"),
                ("SRC1_INNER_TYPE", "f32"),
                ("INIT_SRC0_SHMEM_Q4_0", ""),
                ("INIT_SRC1_SHMEM_FLOAT", ""),
                ("WORKGROUP_SIZE_M", "8u"),
                ("WORKGROUP_SIZE_N", "8u"),
                ("TILE_M", "4u"),
                ("TILE_N", "4u"),
                ("TILE_K", "32u"),
            ],
        );
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg_m = m.div_ceil(8 * 4);
            let wg_n = n.div_ceil(8 * 4);
            pass.dispatch_workgroups(wg_m, wg_n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let result = ctx.download_f32(&y_buf, (n * m) as usize);
        for i in 0..(n * m) as usize {
            let diff = (result[i] - expected[i]).abs();
            // Q4_0 precision is lower, but should be within noise
            assert!(
                diff < 1e-2,
                "mismatch at {}: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_gpu_mul_mat_tile_scalar_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // m=30 is NOT a multiple of 4, forcing SCALAR variant
        let m: u32 = 30;
        let k: u32 = 128;
        let n: u32 = 16;
        let x_stride: u32 = k + 3;
        let y_stride: u32 = m + 5;

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let mut x_batch: Vec<f32> = Vec::with_capacity((n * x_stride) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 64.0)) * 0.05);
            }
            x_batch.resize(x_batch.len() + (x_stride - k) as usize, -999.0);
        }

        let mut expected = vec![0.0f32; (n * y_stride) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..t * x_stride as usize + k as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for col in 0..k as usize {
                    acc += weights_f32[row * k as usize + col] * x_slice[col];
                }
                expected[t * y_stride as usize + row] = acc;
            }
        }

        let a_buf = ctx.upload_f32(&weights_f32, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "y_batch");

        let params: [u32; 5] = [m, k, n, x_stride, y_stride];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline_with_defines(
            shaders::MUL_MAT_REG_TILE,
            "main",
            "mul_mat_tile_scalar_test",
            &[
                ("SCALAR", ""),
                ("SRC0_INNER_TYPE", "f32"),
                ("SRC1_INNER_TYPE", "f32"),
                ("INIT_SRC0_SHMEM_FLOAT", ""),
                ("INIT_SRC1_SHMEM_FLOAT", ""),
                ("WORKGROUP_SIZE_M", "8u"),
                ("WORKGROUP_SIZE_N", "8u"),
                ("TILE_M", "4u"),
                ("TILE_N", "4u"),
                ("TILE_K", "32u"),
            ],
        );
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg_m = m.div_ceil(8 * 4);
            let wg_n = n.div_ceil(8 * 4);
            pass.dispatch_workgroups(wg_m, wg_n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let result = ctx.download_f32(&y_buf, (n * y_stride) as usize);
        for t in 0..n as usize {
            for row in 0..m as usize {
                let i = t * y_stride as usize + row;
                let diff = (result[i] - expected[i]).abs();
                assert!(
                    diff < 1e-4,
                    "mismatch at {}: {} vs {}",
                    i,
                    result[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn test_gpu_mul_mat_tile_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m: u32 = 32;
        let k: u32 = 128;
        let n: u32 = 16; // tokens

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let mut x_batch: Vec<f32> = Vec::with_capacity((n * k) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 64.0)) * 0.05);
            }
        }

        // CPU reference
        let mut expected = vec![0.0f32; (n * m) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * k as usize..(t + 1) * k as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for col in 0..k as usize {
                    acc += weights_f32[row * k as usize + col] * x_slice[col];
                }
                expected[t * m as usize + row] = acc;
            }
        }

        // GPU run
        let a_buf = ctx.upload_f32(&weights_f32, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * m) as u64) * 4, "y_batch");

        // MulMatParams: m, k, n, x_stride, y_stride.
        let params: [u32; 5] = [m, k, n, k, m];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline_with_defines(
            shaders::MUL_MAT_REG_TILE,
            "main",
            "mul_mat_tile_test",
            &[
                ("VEC", ""),
                ("SRC0_INNER_TYPE", "f32"),
                ("SRC1_INNER_TYPE", "f32"),
                ("INIT_SRC0_SHMEM_FLOAT", ""),
                ("INIT_SRC1_SHMEM_FLOAT", ""),
                ("WORKGROUP_SIZE_M", "8u"),
                ("WORKGROUP_SIZE_N", "8u"),
                ("TILE_M", "4u"),
                ("TILE_N", "4u"),
                ("TILE_K", "32u"),
            ],
        );
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Tests build pipelines with WORKGROUP_SIZE_*=8 and TILE_*=4,
            // so each workgroup covers 8*4 = 32 rows AND cols.
            let wg_m = m.div_ceil(8 * 4);
            let wg_n = n.div_ceil(8 * 4);
            pass.dispatch_workgroups(wg_m, wg_n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let result = ctx.download_f32(&y_buf, (n * m) as usize);
        for i in 0..(n * m) as usize {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "mismatch at {}: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    /// Q4_0 SCALAR parity — exercises the variant the production
    /// fallback uses when `m` isn't a multiple of 4. The VEC test
    /// above leaves this path untouched.
    #[test]
    fn test_gpu_mul_mat_tile_q4_0_scalar_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // m=30 forces the SCALAR pipeline; k stays block-aligned (Q4_0
        // requires k % 32 == 0).
        let m: u32 = 30;
        let k: u32 = 128;
        let n: u32 = 16;

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let q4_bytes = quantize_q4_0_for_test(m as usize, k as usize, &weights_f32);

        let mut x_batch: Vec<f32> = Vec::with_capacity((n * k) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 64.0)) * 0.05);
            }
        }

        let expected = cpu_matmul_q4_0(m as usize, k as usize, n as usize, &q4_bytes, &x_batch);

        let a_buf = ctx.upload_storage(&q4_bytes, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * m) as u64) * 4, "y_batch");

        let params: [u32; 5] = [m, k, n, k, m];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline_with_defines(
            shaders::MUL_MAT_REG_TILE,
            "main",
            "mul_mat_q4_0_scalar_test",
            &[
                ("SCALAR", ""),
                ("SRC0_INNER_TYPE", "u32"),
                ("SRC1_INNER_TYPE", "f32"),
                ("INIT_SRC0_SHMEM_Q4_0", ""),
                ("INIT_SRC1_SHMEM_FLOAT", ""),
                ("WORKGROUP_SIZE_M", "8u"),
                ("WORKGROUP_SIZE_N", "8u"),
                ("TILE_M", "4u"),
                ("TILE_N", "4u"),
                ("TILE_K", "32u"),
            ],
        );
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg_m = m.div_ceil(8 * 4);
            let wg_n = n.div_ceil(8 * 4);
            pass.dispatch_workgroups(wg_m, wg_n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let result = ctx.download_f32(&y_buf, (n * m) as usize);
        for i in 0..(n * m) as usize {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-2,
                "mismatch at {}: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemv_f32_realistic() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Realistic LFM2 FFN gate: 2816 × 1024
        let m = 2816u32;
        let k = 1024u32;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 997) as f32 * 0.001 - 0.5)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 13 + 7) % 251) as f32 * 0.01 - 1.25)
            .collect();

        // CPU reference
        let mut expected = vec![0.0f32; m as usize];
        for i in 0..m as usize {
            for j in 0..k as usize {
                expected[i] += a[i * k as usize + j] * x[j];
            }
        }

        // GPU
        let a_buf = ctx.upload_f32(&a, "A");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(m, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);

        let mut max_diff = 0.0f32;
        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 0.1, // wider tolerance for large dot products
                "GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
        println!(
            "GPU GEMV 2816×1024: max_diff={max_diff:.6}, all {} rows match",
            m
        );
    }

    #[test]
    #[ignore] // slow microbenchmark — run explicitly with --ignored
    fn bench_gpu_gemv_f32() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Benchmark at realistic FFN sizes: 2816×1024 (gate projection)
        let m = 2816u32;
        let k = 1024u32;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 997) as f32 * 0.001 - 0.5)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 13 + 7) % 251) as f32 * 0.01 - 1.25)
            .collect();

        let a_buf = ctx.upload_f32(&a, "A");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        // Warmup
        for _ in 0..5 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(m, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::Maintain::Wait);

        // Timed: 100 iterations of single GEMV dispatch
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(m, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed();

        let us_per_gemv = elapsed.as_micros() as f64 / iters as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (us_per_gemv * 1e3);

        // CPU reference timing (use black_box to prevent dead-code elimination)
        let mut cpu_y = vec![0.0f32; m as usize];
        let cpu_iters = 1000;
        let cpu_start = std::time::Instant::now();
        for _ in 0..cpu_iters {
            for i in 0..m as usize {
                let mut sum = 0.0f32;
                for j in 0..k as usize {
                    sum += a[i * k as usize + j] * x[j];
                }
                cpu_y[i] = sum;
            }
            std::hint::black_box(&cpu_y);
        }
        let cpu_elapsed = cpu_start.elapsed();
        let cpu_us = cpu_elapsed.as_micros() as f64 / cpu_iters as f64;
        let cpu_gflops = (2.0 * m as f64 * k as f64) / (cpu_us * 1e3);

        // NEON Q4_0 GEMV timing (the actual decode hot path)
        #[cfg(target_arch = "aarch64")]
        let neon_q4_us = {
            // Build synthetic Q4_0 weight data inline
            let nb = k as usize / 32;
            let mut q4_bytes = Vec::new();
            for row in 0..m as usize {
                for b in 0..nb {
                    let start = row * k as usize + b * 32;
                    let block = &a[start..start + 32];
                    let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let d = amax / 7.0;
                    let d_f16 = half::f16::from_f32(d);
                    q4_bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let mut qs = [0u8; 16];
                    for i in 0..16 {
                        let lo = ((block[i] * id + 8.5) as u8).min(15);
                        let hi = ((block[16 + i] * id + 8.5) as u8).min(15);
                        qs[i] = lo | (hi << 4);
                    }
                    q4_bytes.extend_from_slice(&qs);
                }
            }
            let mut q4_y = vec![0.0f32; m as usize];
            let mut q8s = Vec::new();
            let mut q8q = Vec::new();
            let q4_iters = 1000;
            let q4_start = std::time::Instant::now();
            for _ in 0..q4_iters {
                unsafe {
                    crate::backend::simd::neon::gemv_q4_0_f32_neon(
                        &q4_bytes, &x, &mut q4_y, m as usize, k as usize, &mut q8s, &mut q8q,
                    );
                }
                std::hint::black_box(&q4_y);
            }
            let q4_elapsed = q4_start.elapsed();
            q4_elapsed.as_micros() as f64 / q4_iters as f64
        };
        #[cfg(not(target_arch = "aarch64"))]
        let neon_q4_us = 0.0;
        let neon_q4_gflops = if neon_q4_us > 0.0 {
            (2.0 * m as f64 * k as f64) / (neon_q4_us * 1e3)
        } else {
            0.0
        };

        println!(
            "GEMV {m}×{k}:\n  GPU(f32 Metal)     = {us_per_gemv:.0}µs ({gflops:.1} GFLOPS)\n  CPU(scalar f32)    = {cpu_us:.0}µs ({cpu_gflops:.1} GFLOPS)\n  CPU(NEON Q4_0)     = {neon_q4_us:.0}µs ({neon_q4_gflops:.1} GFLOPS)\n  GPU vs scalar:  {:.1}x\n  GPU vs NEON Q4_0: {:.1}x",
            cpu_us / us_per_gemv,
            if neon_q4_us > 0.0 {
                neon_q4_us / us_per_gemv
            } else {
                0.0
            },
        );
    }

    #[test]
    fn test_gpu_elementwise_add() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 1024u32;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let a_buf = ctx.create_storage_rw((n as u64) * 4, "a");
        ctx.queue.write_buffer(&a_buf, 0, bytemuck::cast_slice(&a));
        let b_buf = ctx.upload_f32(&b, "b");
        let params = [n, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "add_inplace");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&a_buf, n as usize);
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(diff < 1e-5, "add mismatch at {i}: {diff}");
        }
    }

    #[test]
    fn test_gpu_silu_mul() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 512u32;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.02).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let gate_buf = ctx.create_storage_rw((n as u64) * 4, "gate");
        ctx.queue
            .write_buffer(&gate_buf, 0, bytemuck::cast_slice(&gate));
        let up_buf = ctx.upload_f32(&up, "up");
        let params = [n, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::ELEMENTWISE, "silu_mul_inplace", "silu_mul");
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: up_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = ctx.download_f32(&gate_buf, n as usize);
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-4,
                "silu_mul mismatch at {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_rmsnorm() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 1024u32;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 7.0) * 0.05).collect();

        // CPU reference
        let mut expected = x.clone();
        crate::backend::cpu::rmsnorm(&mut expected, &weight, eps);

        // GPU
        let x_buf = ctx.create_storage_rw((n as u64) * 4, "x");
        ctx.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
        let w_buf = ctx.upload_f32(&weight, "w");
        let params = [n, eps.to_bits(), 0u32, 0u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&x_buf, n as usize);
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-3,
                "rmsnorm mismatch at {i}: cpu={}, gpu={}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_softmax() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n = 128u32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();

        // CPU reference
        let mut expected = x.clone();
        crate::backend::cpu::softmax_inplace(&mut expected);

        // GPU
        let x_buf = ctx.create_storage_rw((n as u64) * 4, "x");
        ctx.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
        let params = [n, 0u32];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::SOFTMAX, "softmax", "softmax");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&x_buf, n as usize);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum should be 1.0, got {sum}"
        );
        for i in 0..n as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-5,
                "softmax mismatch at {i}: cpu={}, gpu={}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemv_q4_0() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Build a small Q4_0 weight matrix (8 rows × 64 elements — matches shader ROWS_PER_WG)
        let m = 8u32;
        let k = 64u32;
        let nb = k / 32;

        // Generate f32 weights, quantize to Q4_0
        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();

        // Q4_0 quantize each row
        let mut q4_bytes: Vec<u8> = Vec::new();
        for row in 0..m as usize {
            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let block = &weights_f32[start..start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = amax / 7.0;
                let d_f16 = half::f16::from_f32(d);
                q4_bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                for qi in 0..16 {
                    let lo = ((block[qi] * id + 8.5) as u8).min(15);
                    let hi = ((block[16 + qi] * id + 8.5) as u8).min(15);
                    q4_bytes.push(lo | (hi << 4));
                }
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 32.0) * 0.05).collect();

        // CPU reference: dequant + matmul
        let mut expected = vec![0.0f32; m as usize];
        for (row, exp) in expected.iter_mut().enumerate() {
            for b in 0..nb as usize {
                let block_off = (row * nb as usize + b) * 18;
                let d_bits = u16::from_le_bytes([q4_bytes[block_off], q4_bytes[block_off + 1]]);
                let delta = half::f16::from_bits(d_bits).to_f32();
                for qi in 0..16 {
                    let byte = q4_bytes[block_off + 2 + qi];
                    let lo = (byte & 0xF) as f32 - 8.0;
                    let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                    *exp += lo * delta * x[b * 32 + qi];
                    *exp += hi * delta * x[b * 32 + qi + 16];
                }
            }
        }

        // GPU
        let a_buf = ctx.upload_storage(&q4_bytes, "A_q4");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0", "gemv_q4_0");
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

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Shader processes 8 rows per workgroup
            pass.dispatch_workgroups(m.div_ceil(8), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);
        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 0.5,
                "Q4_0 GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
        println!("Q4_0 GEMV {m}×{k}: all rows match");
    }

    #[test]
    fn test_gpu_gemv_q8_0() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m = 9u32;
        let k = 64u32;
        let nb = k / 32;

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 + 5) % 41) as f32 * 0.07 - 1.3)
            .collect();

        let mut q8_bytes: Vec<u8> = Vec::new();
        let mut expected = vec![0.0f32; m as usize];
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 17.0) * 0.03125).collect();

        for (row, exp) in expected.iter_mut().enumerate() {
            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let block = &weights_f32[start..start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = if amax != 0.0 { amax / 127.0 } else { 0.0 };
                let d_f16 = half::f16::from_f32(d);
                q8_bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };

                for (qi, &value) in block.iter().enumerate() {
                    let quant = (value * id).round().clamp(-127.0, 127.0) as i8;
                    q8_bytes.push(quant as u8);
                    *exp += f32::from(quant) * d_f16.to_f32() * x[b * 32 + qi];
                }
            }
        }

        let a_buf = ctx.upload_storage(&q8_bytes, "A_q8");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_Q8_0, "gemv_q8_0", "gemv_q8_0");
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

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(8), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);
        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            assert!(
                diff < 1e-3,
                "Q8_0 GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    /// Q6_K GEMV parity: synthesize Q6_K blocks, dequantize on CPU, GEMV on the
    /// wgpu `gemv_q6_k` kernel, compare. Mirrors the Metal `test_gemv_q6_k`
    /// parity test; guards the B1 wiring (Q6K stays quantized on the GPU and is
    /// served by `gemv_q6_k` in `gemv_pipeline_rows_label`). The kernel uses
    /// NR=2 rows per workgroup (32 threads), so the dispatch is `ceil(m/2)`.
    #[test]
    fn test_gpu_gemv_q6_k() {
        use crate::quant::{BlockQ6K, dequantize_q6_k_block};
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // m odd so the ceil(m/2) tail workgroup (which writes only row 0) is
        // exercised; k a multiple of 256 as Q6_K always is.
        let m = 65u32;
        let k = 512u32; // 2 super-blocks per row
        let qk_k = 256usize;
        let nb = k as usize / qk_k;

        // Synthetic Q6_K weights: deterministic ql/qh/scales, serialized into
        // the exact GGUF block layout the shader reads (ql | qh | scales | d).
        let mut raw = Vec::with_capacity(m as usize * nb * 210);
        let mut expected_f32 = vec![0.0f32; m as usize * k as usize];
        for row in 0..m as usize {
            for b in 0..nb {
                let mut blk = BlockQ6K {
                    ql: [0u8; 128],
                    qh: [0u8; 64],
                    scales: [0i8; 16],
                    d: half::f16::from_f32(0.01 + (row as f32 * 0.003).sin() * 0.002).to_bits(),
                };
                for (i, v) in blk.ql.iter_mut().enumerate() {
                    *v = ((row * 37 + b * 13 + i) & 0xFF) as u8;
                }
                for (i, v) in blk.qh.iter_mut().enumerate() {
                    *v = ((row * 11 + b * 7 + i) & 0xFF) as u8;
                }
                for (i, v) in blk.scales.iter_mut().enumerate() {
                    *v = (((row * 3 + b * 5 + i) as i32 & 0x7F) - 32) as i8;
                }
                let dq = dequantize_q6_k_block(&blk);
                let row_off = row * k as usize + b * qk_k;
                expected_f32[row_off..row_off + qk_k].copy_from_slice(&dq);
                raw.extend_from_slice(&blk.ql);
                raw.extend_from_slice(&blk.qh);
                raw.extend_from_slice(bytemuck::cast_slice(&blk.scales));
                raw.extend_from_slice(&blk.d.to_le_bytes());
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();

        // CPU reference: y[r] = Σ weight_f32[r][i] × x[i].
        let mut expected = vec![0.0f32; m as usize];
        for (row, exp) in expected.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for i in 0..k as usize {
                s += expected_f32[row * k as usize + i] * x[i];
            }
            *exp = s;
        }

        let a_buf = ctx.upload_storage(&raw, "A_q6k");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_Q6_K, "gemv_q6_k", "gemv_q6_k");
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

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(2), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);
        for i in 0..m as usize {
            let denom = expected[i].abs().max(1.0);
            let rel = (expected[i] - result[i]).abs() / denom;
            assert!(
                rel < 5e-3,
                "Q6_K GEMV mismatch at row {i}: cpu={}, gpu={}, rel={rel:.2e}",
                expected[i],
                result[i]
            );
        }
    }

    /// Q4_K (Q4_K_M) GEMV parity: synthesize Q4_K blocks, dequantize on CPU with
    /// the production `dequantize_q4_k_m_block`, GEMV on the wgpu `gemv_q4_k`
    /// kernel, compare. Guards the B2 wiring (Q4KM stays quantized on the GPU and
    /// is served by `gemv_q4_k` in `gemv_pipeline_rows_label`, NR=2 → ceil(m/2)).
    #[test]
    fn test_gpu_gemv_q4_k() {
        use crate::quant::{BlockQ4KM, dequantize_q4_k_m_block};
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // m odd to exercise the ceil(m/2) tail workgroup; k a multiple of 256.
        let m = 65u32;
        let k = 512u32; // 2 super-blocks per row
        let qk_k = 256usize;
        let nb = k as usize / qk_k;

        // Synthetic Q4_K weights serialized into the exact GGUF block layout the
        // shader reads (d | dmin | scales[12] | qs[128] = 144 bytes).
        let mut raw = Vec::with_capacity(m as usize * nb * 144);
        let mut expected_f32 = vec![0.0f32; m as usize * k as usize];
        for row in 0..m as usize {
            for b in 0..nb {
                let mut blk = BlockQ4KM {
                    d: half::f16::from_f32(0.02 + (row as f32 * 0.004).sin() * 0.003).to_bits(),
                    dmin: half::f16::from_f32(0.01 + (b as f32 * 0.002)).to_bits(),
                    scales: [0u8; 12],
                    qs: [0u8; 128],
                };
                for (i, v) in blk.scales.iter_mut().enumerate() {
                    *v = ((row * 5 + b * 7 + i * 3) & 0xFF) as u8;
                }
                for (i, v) in blk.qs.iter_mut().enumerate() {
                    *v = ((row * 37 + b * 13 + i) & 0xFF) as u8;
                }
                let dq = dequantize_q4_k_m_block(&blk);
                let row_off = row * k as usize + b * qk_k;
                expected_f32[row_off..row_off + qk_k].copy_from_slice(&dq);
                raw.extend_from_slice(&blk.d.to_le_bytes());
                raw.extend_from_slice(&blk.dmin.to_le_bytes());
                raw.extend_from_slice(&blk.scales);
                raw.extend_from_slice(&blk.qs);
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.017).cos()).collect();

        // CPU reference: y[r] = Σ weight_f32[r][i] × x[i].
        let mut expected = vec![0.0f32; m as usize];
        for (row, exp) in expected.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for i in 0..k as usize {
                s += expected_f32[row * k as usize + i] * x[i];
            }
            *exp = s;
        }

        let a_buf = ctx.upload_storage(&raw, "A_q4k");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_Q4_K, "gemv_q4_k", "gemv_q4_k");
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

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(2), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);
        for i in 0..m as usize {
            let denom = expected[i].abs().max(1.0);
            let rel = (expected[i] - result[i]).abs() / denom;
            assert!(
                rel < 5e-3,
                "Q4_K GEMV mismatch at row {i}: cpu={}, gpu={}, rel={rel:.2e}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemv_q5_k() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // 5 rows × 512 cols = 2 Q5_K blocks per row.
        let m = 5u32;
        let k = 512u32;
        let nb = (k / 256) as usize;
        let bpb = 176usize; // Q5_K block bytes

        // Synthesize deterministic Q5_K blocks (raw GGUF bytes). We compare the
        // GPU kernel against the CPU `dequantize_q5_k_row` reference, so we don't
        // need a real quantizer — arbitrary well-formed bytes exercise every
        // field (d/dmin, 6-bit scales+mins, qh 5th bit, qs nibbles).
        let n_blocks = m as usize * nb;
        let mut a_bytes = vec![0u8; n_blocks * bpb];
        for (bi, chunk) in a_bytes.chunks_mut(bpb).enumerate() {
            let d = half::f16::from_f32(0.015 + (bi % 5) as f32 * 0.004);
            let dmin = half::f16::from_f32(0.008 + (bi % 3) as f32 * 0.003);
            chunk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            chunk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            for (i, b) in chunk[4..16].iter_mut().enumerate() {
                *b = ((bi * 7 + i * 13 + 1) % 256) as u8; // scales (full 8-bit; decode masks to 6)
            }
            for (i, b) in chunk[16..48].iter_mut().enumerate() {
                *b = ((bi * 29 + i * 7) % 256) as u8; // qh
            }
            for (i, b) in chunk[48..176].iter_mut().enumerate() {
                *b = ((bi * 17 + i * 5) % 256) as u8; // qs
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 100.0) * 0.01).collect();

        // CPU reference: dequantize each row and dot with x.
        let mut expected = vec![0.0f32; m as usize];
        for (row, exp) in expected.iter_mut().enumerate() {
            let row_bytes = &a_bytes[row * nb * bpb..(row + 1) * nb * bpb];
            let mut deq = vec![0.0f32; k as usize];
            crate::quant::dequantize_q5_k_row(row_bytes, &mut deq);
            *exp = deq.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }

        let a_buf = ctx.upload_storage(&a_bytes, "A_q5k");
        let x_buf = ctx.upload_f32(&x, "x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "y");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMV_Q5_K, "gemv_q5_k", "gemv_q5_k");
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

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(2), 1, 1); // NR=2
        }
        ctx.queue.submit(Some(enc.finish()));

        let result = ctx.download_f32(&y_buf, m as usize);
        for i in 0..m as usize {
            let diff = (expected[i] - result[i]).abs();
            let tol = 1e-3 * expected[i].abs().max(1.0);
            assert!(
                diff <= tol,
                "Q5_K GEMV mismatch at row {i}: cpu={}, gpu={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gpu_gemm_q8_0_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m = 11u32;
        let k = 64u32;
        let n = 3u32;
        let x_stride = k + 4;
        let y_stride = m + 5;
        let nb = k / 32;

        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 53) as f32 * 0.045 - 1.1)
            .collect();

        let mut q8_bytes: Vec<u8> = Vec::new();
        for row in 0..m as usize {
            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let block = &weights_f32[start..start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = if amax != 0.0 { amax / 127.0 } else { 0.0 };
                let d_f16 = half::f16::from_f32(d);
                q8_bytes.extend_from_slice(&d_f16.to_bits().to_le_bytes());
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                for &value in block {
                    let quant = (value * id).round().clamp(-127.0, 127.0) as i8;
                    q8_bytes.push(quant as u8);
                }
            }
        }

        let mut x_batch = vec![0.0f32; (n * x_stride) as usize];
        for t in 0..n as usize {
            for i in 0..k as usize {
                x_batch[t * x_stride as usize + i] = ((t as f32 + 1.0) * (i as f32 - 19.0)) * 0.021;
            }
        }

        let mut expected = vec![0.0f32; (n * y_stride) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..t * x_stride as usize + k as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for b in 0..nb as usize {
                    let block_off = (row * nb as usize + b) * 34;
                    let d_bits = u16::from_le_bytes([q8_bytes[block_off], q8_bytes[block_off + 1]]);
                    let d = half::f16::from_bits(d_bits).to_f32();
                    for qi in 0..32 {
                        let quant = q8_bytes[block_off + 2 + qi] as i8;
                        acc += f32::from(quant) * d * x_slice[b * 32 + qi];
                    }
                }
                expected[t * y_stride as usize + row] = acc;
            }
        }

        let a_buf = ctx.upload_storage(&q8_bytes, "gemm_q8_weights");
        let x_buf = ctx.upload_f32(&x_batch, "gemm_q8_x");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "gemm_q8_y");
        let params: [u32; 6] = [m, k, n, x_stride, y_stride, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "gemm_q8_params");

        let pipeline = ctx.create_pipeline(shaders::GEMM_Q8_0, "gemm_q8_0", "gemm_q8_0");
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(8), n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&y_buf, (n * y_stride) as usize);
        for t in 0..n as usize {
            for row in 0..m as usize {
                let idx = t * y_stride as usize + row;
                let diff = (expected[idx] - got[idx]).abs();
                assert!(
                    diff < 1e-3,
                    "Q8_0 GEMM mismatch at token {t}, row {row}: cpu={}, gpu={}, diff={diff}",
                    expected[idx],
                    got[idx]
                );
            }
        }
    }

    /// Q4_K (Q4_K_M) batched GEMM parity: synthesize Q4_K blocks, dequantize on
    /// CPU with `dequantize_q4_k_m_block`, run the wgpu `gemm_q4_k` kernel over a
    /// multi-token batch with padded x_stride/y_stride, compare. Guards the B2
    /// batched-prefill wiring (ROWS_PER_WG=8 → dispatch ceil(m/8)); m not a
    /// multiple of 8 exercises the partial row tile.
    #[test]
    fn test_gpu_gemm_q4_k_parity() {
        use crate::quant::{BlockQ4KM, dequantize_q4_k_m_block};
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m = 11u32;
        let k = 512u32; // 2 super-blocks per row
        let n = 3u32;
        let x_stride = k + 4;
        let y_stride = m + 5;
        let qk_k = 256usize;
        let nb = k as usize / qk_k;

        // Synthetic Q4_K weights + their f32 dequantization (CPU reference).
        let mut raw = Vec::with_capacity(m as usize * nb * 144);
        let mut w_f32 = vec![0.0f32; m as usize * k as usize];
        for row in 0..m as usize {
            for b in 0..nb {
                let mut blk = BlockQ4KM {
                    d: half::f16::from_f32(0.02 + (row as f32 * 0.004).sin() * 0.003).to_bits(),
                    dmin: half::f16::from_f32(0.01 + (b as f32 * 0.002)).to_bits(),
                    scales: [0u8; 12],
                    qs: [0u8; 128],
                };
                for (i, v) in blk.scales.iter_mut().enumerate() {
                    *v = ((row * 5 + b * 7 + i * 3) & 0xFF) as u8;
                }
                for (i, v) in blk.qs.iter_mut().enumerate() {
                    *v = ((row * 37 + b * 13 + i) & 0xFF) as u8;
                }
                let dq = dequantize_q4_k_m_block(&blk);
                let off = row * k as usize + b * qk_k;
                w_f32[off..off + qk_k].copy_from_slice(&dq);
                raw.extend_from_slice(&blk.d.to_le_bytes());
                raw.extend_from_slice(&blk.dmin.to_le_bytes());
                raw.extend_from_slice(&blk.scales);
                raw.extend_from_slice(&blk.qs);
            }
        }

        let mut x_batch = vec![0.0f32; (n * x_stride) as usize];
        for t in 0..n as usize {
            for i in 0..k as usize {
                x_batch[t * x_stride as usize + i] =
                    ((t as f32 + 1.0) * (i as f32 - 200.0)) * 0.0007;
            }
        }

        let mut expected = vec![0.0f32; (n * y_stride) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..t * x_stride as usize + k as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for i in 0..k as usize {
                    acc += w_f32[row * k as usize + i] * x_slice[i];
                }
                expected[t * y_stride as usize + row] = acc;
            }
        }

        let a_buf = ctx.upload_storage(&raw, "gemm_q4k_weights");
        let x_buf = ctx.upload_f32(&x_batch, "gemm_q4k_x");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "gemm_q4k_y");
        let params: [u32; 6] = [m, k, n, x_stride, y_stride, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "gemm_q4k_params");

        let pipeline = ctx.create_pipeline(shaders::GEMM_Q4_K, "gemm_q4_k", "gemm_q4_k");
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(8), n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&y_buf, (n * y_stride) as usize);
        for t in 0..n as usize {
            for row in 0..m as usize {
                let idx = t * y_stride as usize + row;
                let denom = expected[idx].abs().max(1.0);
                let rel = (expected[idx] - got[idx]).abs() / denom;
                assert!(
                    rel < 5e-3,
                    "Q4_K GEMM mismatch at token {t}, row {row}: cpu={}, gpu={}, rel={rel:.2e}",
                    expected[idx],
                    got[idx]
                );
            }
        }
    }

    #[test]
    fn test_gpu_gemv_quant_dispatch_smoke() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m = 9u32;
        let k = 64u32;
        let nb = k / 32;
        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 11 + 7) % 37) as f32 * 0.08 - 1.2)
            .collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 23.0) * 0.025).collect();

        let mut q4_bytes = Vec::new();
        let mut q4_expected = vec![0.0f32; m as usize];
        let mut q8_bytes = Vec::new();
        let mut q8_expected = vec![0.0f32; m as usize];
        let mut f32_expected = vec![0.0f32; m as usize];

        for row in 0..m as usize {
            for col in 0..k as usize {
                f32_expected[row] += weights_f32[row * k as usize + col] * x[col];
            }

            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let block = &weights_f32[start..start + 32];
                let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

                let d4 = if amax != 0.0 { amax / 7.0 } else { 0.0 };
                let d4_f16 = half::f16::from_f32(d4);
                q4_bytes.extend_from_slice(&d4_f16.to_bits().to_le_bytes());
                let id4 = if d4 != 0.0 { 1.0 / d4 } else { 0.0 };
                for qi in 0..16 {
                    let lo = ((block[qi] * id4 + 8.5) as u8).min(15);
                    let hi = ((block[16 + qi] * id4 + 8.5) as u8).min(15);
                    q4_bytes.push(lo | (hi << 4));
                    q4_expected[row] += (f32::from(lo) - 8.0) * d4_f16.to_f32() * x[b * 32 + qi];
                    q4_expected[row] +=
                        (f32::from(hi) - 8.0) * d4_f16.to_f32() * x[b * 32 + 16 + qi];
                }

                let d8 = if amax != 0.0 { amax / 127.0 } else { 0.0 };
                let d8_f16 = half::f16::from_f32(d8);
                q8_bytes.extend_from_slice(&d8_f16.to_bits().to_le_bytes());
                let id8 = if d8 != 0.0 { 1.0 / d8 } else { 0.0 };
                for (qi, &value) in block.iter().enumerate() {
                    let quant = (value * id8).round().clamp(-127.0, 127.0) as i8;
                    q8_bytes.push(quant as u8);
                    q8_expected[row] += f32::from(quant) * d8_f16.to_f32() * x[b * 32 + qi];
                }
            }
        }

        struct Case<'a> {
            name: &'static str,
            dtype: DType,
            shader: &'static str,
            entry: &'static str,
            rows_per_wg: u32,
            weight_bytes: &'a [u8],
            expected: &'a [f32],
            tolerance: f32,
        }

        let f32_weight_bytes = bytemuck::cast_slice(&weights_f32);
        let cases = [
            Case {
                name: "f32",
                dtype: DType::F32,
                shader: shaders::GEMV_F32,
                entry: "gemv_f32",
                rows_per_wg: 8,
                weight_bytes: f32_weight_bytes,
                expected: &f32_expected,
                tolerance: 1e-4,
            },
            Case {
                name: "q4_0",
                dtype: DType::Q4_0,
                shader: shaders::GEMV_Q4_0_FAST,
                entry: "gemv_q4_0_fast",
                rows_per_wg: 4,
                weight_bytes: &q4_bytes,
                expected: &q4_expected,
                tolerance: 5e-2,
            },
            Case {
                name: "q8_0",
                dtype: DType::Q8_0,
                shader: shaders::GEMV_Q8_0,
                entry: "gemv_q8_0",
                rows_per_wg: 8,
                weight_bytes: &q8_bytes,
                expected: &q8_expected,
                tolerance: 1e-3,
            },
        ];

        let x_buf = ctx.upload_f32(&x, "gemv_dispatch_x");
        let params = [m, k, 0u32, 0u32];
        let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "gemv_dispatch_params");

        for case in cases {
            let pipeline = ctx.create_pipeline(case.shader, case.entry, case.name);
            let a_buf = ctx.upload_storage(case.weight_bytes, case.name);
            let y_buf = ctx.create_storage_rw((m as u64) * 4, "gemv_dispatch_y");
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(case.name),
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

            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(m.div_ceil(case.rows_per_wg), 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));

            let result = ctx.download_f32(&y_buf, m as usize);
            for (i, (&exp, &got)) in case.expected.iter().zip(&result).enumerate() {
                let diff = (exp - got).abs();
                assert!(
                    diff < case.tolerance,
                    "{:?} {} GEMV mismatch at row {i}: cpu={exp}, gpu={got}, diff={diff}",
                    case.dtype,
                    case.name,
                );
            }
        }
    }

    /// Argmax kernel correctness across three shapes that exercise the
    /// stride loop (`n > 256`), the trivial single-stride case (`n < 256`),
    /// and the boundary (`n == 256`). Plants known maxima to verify both
    /// the value picked and the lower-idx tie-break.
    #[test]
    fn test_gpu_argmax_f32() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let pipeline = ctx.create_pipeline(shaders::ARGMAX_F32, "argmax_f32", "argmax_f32");

        let cases: &[(usize, usize)] = &[
            (32, 17),       // n < workgroup size; expected idx 17
            (256, 200),     // n == workgroup size
            (2048, 1733),   // typical multi-stride n
            (50000, 12345), // vocab-sized
        ];
        for &(n, plant_idx) in cases {
            // Build a non-monotonic vector so `argmax`-style tie-break
            // edge cases don't accidentally pass via "highest index wins".
            let mut x: Vec<f32> = (0..n)
                .map(|i| ((i as i32 * 31 + 7) % 211) as f32 / 211.0)
                .collect();
            x[plant_idx] = 99.0; // unambiguous global max

            let x_buf = ctx.upload_f32(&x, "argmax_in");
            let out_buf = ctx.create_storage_rw(4, "argmax_out");
            let params =
                ctx.upload_storage(bytemuck::cast_slice(&[n as u32, 0u32]), "argmax_params");
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });

            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));

            let out = ctx.download_u32(&out_buf, 1);
            assert_eq!(
                out[0] as usize, plant_idx,
                "argmax(n={n}) returned {}, expected {plant_idx}",
                out[0]
            );
        }

        // Lower-index tie-break: two equal maxima, lower index must win.
        let n: usize = 1024;
        let mut x = vec![0.0f32; n];
        x[100] = 5.0;
        x[700] = 5.0; // equal-magnitude tie
        let x_buf = ctx.upload_f32(&x, "argmax_tie_in");
        let out_buf = ctx.create_storage_rw(4, "argmax_tie_out");
        let params =
            ctx.upload_storage(bytemuck::cast_slice(&[n as u32, 0u32]), "argmax_tie_params");
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let out = ctx.download_u32(&out_buf, 1);
        assert_eq!(
            out[0], 100,
            "tie-break: lower index must win, got {}",
            out[0]
        );
    }

    /// Spike: confirm WGSL `override` constants flow through the wgpu 24
    /// pipeline-creation API on this machine. If this passes, per-head-dim
    /// specialization in Phase 3 of the wgpu kernel-parity plan can ride on
    /// `override` rather than separate shader files / string templating.
    #[test]
    fn spike_wgsl_override_constants() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Single shader, single entry point. `HEAD_DIM` is an
        // override-able u32 with a default value of 1; the kernel writes
        // its current value to `out[0]` so we can read it back.
        let src = r#"
            override HEAD_DIM: u32 = 1u;
            @group(0) @binding(0) var<storage, read_write> out: array<u32>;
            @compute @workgroup_size(1)
            fn main() { out[0] = HEAD_DIM; }
        "#;

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("override_spike"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });

        let make_pipeline = |head_dim: u32| {
            let mut consts: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            consts.insert("HEAD_DIM".to_string(), head_dim as f64);
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("override_spike_hd{head_dim}")),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &consts,
                        zero_initialize_workgroup_memory: true,
                    },
                    cache: None,
                })
        };

        let dispatch_and_read = |pipeline: &wgpu::ComputePipeline| -> u32 {
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("override_spike_out"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
            // u32 readback via the same staging path used for f32 elsewhere.
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, 4);
            ctx.queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            staging.unmap();
            v
        };

        let pipeline_64 = make_pipeline(64);
        let pipeline_128 = make_pipeline(128);
        let v64 = dispatch_and_read(&pipeline_64);
        let v128 = dispatch_and_read(&pipeline_128);
        assert_eq!(v64, 64, "override HEAD_DIM=64 not honored");
        assert_eq!(v128, 128, "override HEAD_DIM=128 not honored");
        println!("WGSL override spike OK: same module → HEAD_DIM={v64} and {v128}");
    }

    /// Parity check: `rmsnorm_batch` on N vectors must match the
    /// per-vector `rmsnorm.wgsl` invoked N times. Same fixture, same
    /// weights, byte-close output. Covers the contract that PR 2.C-full
    /// will lean on — batched dispatch is a no-op rewrite of the
    /// per-token loop.
    #[test]
    fn test_gpu_rmsnorm_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n: u32 = 1024; // hidden_size
        let batch: u32 = 7; // N tokens — non-power-of-two on purpose
        let eps = 1e-5f32;

        // Build N distinct vectors. Each token gets its own pseudo-random
        // pattern so no two share a sum_sq → catches per-workgroup
        // offset bugs.
        let mut src: Vec<f32> = Vec::with_capacity((n * batch) as usize);
        for b in 0..batch {
            for i in 0..n {
                src.push(((b as f32 + 1.0) * (i as f32 - 512.0)) * 0.001);
            }
        }
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 7.0) * 0.05).collect();

        // ─── Reference: run the per-vector rmsnorm.wgsl N times ───
        let pipeline_per = ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm_ref");
        let w_buf = ctx.upload_f32(&weight, "w");
        let params_per = [n, eps.to_bits(), 0u32, 0u32];
        let p_buf_per = ctx.upload_storage(bytemuck::cast_slice(&params_per), "params_per");

        let mut reference = vec![0.0f32; (n * batch) as usize];
        for b in 0..batch {
            let row_start = (b * n) as usize;
            let row_end = row_start + n as usize;
            let scratch = ctx.create_storage_rw((n as u64) * 4, "rmsnorm_ref_scratch");
            ctx.queue
                .write_buffer(&scratch, 0, bytemuck::cast_slice(&src[row_start..row_end]));
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline_per.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scratch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buf_per.as_entire_binding(),
                    },
                ],
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline_per);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
            let out = ctx.download_f32(&scratch, n as usize);
            reference[row_start..row_end].copy_from_slice(&out);
        }

        // ─── Batched run: one dispatch with N workgroups ───
        let pipeline_batch =
            ctx.create_pipeline(shaders::RMSNORM_BATCH, "rmsnorm_batch", "rmsnorm_batch");
        let src_buf = ctx.create_storage_rw((src.len() as u64) * 4, "src");
        ctx.queue
            .write_buffer(&src_buf, 0, bytemuck::cast_slice(&src));
        let dst_buf = ctx.create_storage_rw((src.len() as u64) * 4, "dst");
        // params: (n, eps_bits, src_stride, dst_stride, res_scale_bits). Strides
        // are both `n` here; res_scale is unused by the no-residual entry point.
        let params_batch = [n, eps.to_bits(), n, n, 1.0f32.to_bits()];
        let p_buf_batch = ctx.upload_storage(bytemuck::cast_slice(&params_batch), "params_batch");
        // The `rmsnorm_batch` entry point doesn't read `residual`, and
        // naga's auto-layout drops binding 4 from the inferred layout
        // accordingly — the bind group has only 4 entries.
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_batch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf_batch.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline_batch);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(batch, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let batched = ctx.download_f32(&dst_buf, (n * batch) as usize);

        // Allow ~1e-3 absolute slack — same threshold the per-vector
        // test uses against the CPU reference.
        for i in 0..(n * batch) as usize {
            let diff = (reference[i] - batched[i]).abs();
            assert!(
                diff < 1e-3,
                "rmsnorm_batch mismatch at idx {i} (token {}, dim {}): \
                 ref={}, batched={}, diff={diff}",
                i / n as usize,
                i % n as usize,
                reference[i],
                batched[i]
            );
        }
    }

    /// `qk_norm_rope_batch` parity: per-head rmsnorm + RoPE on a batch
    /// of N tokens must match running CPU rmsnorm + CPU rope per token
    /// at `pos = start_pos + token_idx`. Both Q and K are checked.
    #[test]
    fn test_gpu_qk_norm_rope_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n_heads: u32 = 4;
        let n_kv_heads: u32 = 2;
        let head_dim: u32 = 64;
        let n_tokens: u32 = 3;
        let start_pos: u32 = 5;
        let eps = 1e-5f32;
        let freq_base = 10000.0f32;
        let rope_type: u32 = 0;
        let q_stride = n_heads * head_dim;
        let k_stride = n_kv_heads * head_dim;

        // Build N tokens of Q and K activations.
        let mut q_batch: Vec<f32> = Vec::with_capacity((n_tokens * q_stride) as usize);
        let mut k_batch: Vec<f32> = Vec::with_capacity((n_tokens * k_stride) as usize);
        for t in 0..n_tokens {
            for i in 0..q_stride {
                q_batch.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.01);
            }
            for i in 0..k_stride {
                k_batch.push(((t as f32 + 2.0) * (i as f32 - 16.0)) * 0.013);
            }
        }
        let q_norm_w: Vec<f32> = (0..head_dim)
            .map(|i| 0.9 + (i as f32 % 5.0) * 0.04)
            .collect();
        let k_norm_w: Vec<f32> = (0..head_dim)
            .map(|i| 1.1 - (i as f32 % 5.0) * 0.03)
            .collect();

        // ─── CPU reference: per-token rmsnorm-then-rope on each head ──────
        let mut ref_q = q_batch.clone();
        let mut ref_k = k_batch.clone();
        for t in 0..n_tokens {
            let q_off = (t * q_stride) as usize;
            let k_off = (t * k_stride) as usize;
            // rmsnorm each Q head
            for h in 0..n_heads as usize {
                let head_start = q_off + h * head_dim as usize;
                let head_end = head_start + head_dim as usize;
                crate::backend::cpu::rmsnorm(&mut ref_q[head_start..head_end], &q_norm_w, eps);
            }
            // rmsnorm each K head
            for h in 0..n_kv_heads as usize {
                let head_start = k_off + h * head_dim as usize;
                let head_end = head_start + head_dim as usize;
                crate::backend::cpu::rmsnorm(&mut ref_k[head_start..head_end], &k_norm_w, eps);
            }
            // rope at pos = start_pos + t over the per-token Q/K slabs
            let q_end = q_off + (n_heads * head_dim) as usize;
            let k_end = k_off + (n_kv_heads * head_dim) as usize;
            crate::backend::cpu::rope(
                &mut ref_q[q_off..q_end],
                &mut ref_k[k_off..k_end],
                (start_pos + t) as usize,
                n_heads as usize,
                n_kv_heads as usize,
                head_dim as usize,
                freq_base,
            );
        }

        // ─── Batched run: one dispatch over (n_tokens × heads_per_token) ──
        let pipeline = ctx.create_pipeline(
            shaders::QK_NORM_ROPE_BATCH,
            "qk_norm_rope_batch",
            "qk_norm_rope_batch",
        );
        let q_buf = ctx.create_storage_rw((q_batch.len() as u64) * 4, "q");
        ctx.queue
            .write_buffer(&q_buf, 0, bytemuck::cast_slice(&q_batch));
        let k_buf = ctx.create_storage_rw((k_batch.len() as u64) * 4, "k");
        ctx.queue
            .write_buffer(&k_buf, 0, bytemuck::cast_slice(&k_batch));
        let qw_buf = ctx.upload_f32(&q_norm_w, "q_norm_w");
        let kw_buf = ctx.upload_f32(&k_norm_w, "k_norm_w");
        // has_freq_factors = 0 (no Llama-3 scaling), has_qk_norm = 1 (this test
        // exercises the rmsnorm+rope fused path). The rope-only and freq-factors
        // variants are covered end-to-end by the differential prefill tests.
        let params = [
            start_pos,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
            eps.to_bits(),
            freq_base.to_bits(),
            rope_type,
            q_stride,
            k_stride,
            0, // has_freq_factors
            1, // has_qk_norm
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");
        // freq_factors: 1-element dummy (unused when has_freq_factors == 0).
        let ff_buf = ctx.upload_f32(&[1.0f32], "freq_factors_dummy");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ff_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_tokens * (n_heads + n_kv_heads), 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got_q = ctx.download_f32(&q_buf, q_batch.len());
        let got_k = ctx.download_f32(&k_buf, k_batch.len());

        // RoPE introduces sin/cos differences between iterative theta
        // (CPU) and per-d `pow` (shader); the residual is well below
        // 1e-3 in practice but allow a generous slack.
        let tol = 2e-3f32;
        for i in 0..ref_q.len() {
            let diff = (ref_q[i] - got_q[i]).abs();
            assert!(
                diff < tol,
                "Q mismatch at idx {i} (token {}, dim {}): cpu={}, gpu={}, diff={diff}",
                i / q_stride as usize,
                i % q_stride as usize,
                ref_q[i],
                got_q[i]
            );
        }
        for i in 0..ref_k.len() {
            let diff = (ref_k[i] - got_k[i]).abs();
            assert!(
                diff < tol,
                "K mismatch at idx {i} (token {}, dim {}): cpu={}, gpu={}, diff={diff}",
                i / k_stride as usize,
                i % k_stride as usize,
                ref_k[i],
                got_k[i]
            );
        }
    }

    /// `qk_norm_rope_batch` parity for the two variants the dense-transformer
    /// prefill path added but the test above doesn't cover (flagged in review):
    ///   • `has_qk_norm = 0` — rope-only (llama/qwen2/mistral/granite); the
    ///     Phase-1 rmsnorm uniform branch is skipped and slots 2/3 are dummies.
    ///   • `has_freq_factors = 1` + `rope_type = 1` (NORM) — Llama-3 RoPE
    ///     frequency scaling through `binding(5)`.
    /// Both run rope-only against a CPU reference that does NOT rmsnorm.
    #[test]
    fn test_gpu_qk_norm_rope_batch_rope_only_and_freq_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n_heads: u32 = 4;
        let n_kv_heads: u32 = 2;
        let head_dim: u32 = 64;
        let n_tokens: u32 = 3;
        let start_pos: u32 = 5;
        let eps = 1e-5f32; // unused when has_qk_norm = 0, but still uploaded.
        let freq_base = 10000.0f32;
        let q_stride = n_heads * head_dim;
        let k_stride = n_kv_heads * head_dim;

        // Llama-3 freq factors (length head_dim/2), monotonically rising so the
        // per-pair division has a visible, non-trivial effect.
        let freqs: Vec<f32> = (0..head_dim / 2).map(|i| 1.0 + (i as f32) * 0.05).collect();

        // (label, rope_type, has_freq_factors)
        let cases: [(&str, u32, bool); 2] = [("neox_rope_only", 0, false), ("norm_freq", 1, true)];

        for (label, rope_type, has_freq_factors) in cases {
            let mut q_batch: Vec<f32> = Vec::with_capacity((n_tokens * q_stride) as usize);
            let mut k_batch: Vec<f32> = Vec::with_capacity((n_tokens * k_stride) as usize);
            for t in 0..n_tokens {
                for i in 0..q_stride {
                    q_batch.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.01);
                }
                for i in 0..k_stride {
                    k_batch.push(((t as f32 + 2.0) * (i as f32 - 16.0)) * 0.013);
                }
            }

            // CPU reference: rope-only at pos = start_pos + token, no rmsnorm.
            let ff = if has_freq_factors {
                Some(freqs.as_slice())
            } else {
                None
            };
            let mut ref_q = q_batch.clone();
            let mut ref_k = k_batch.clone();
            for t in 0..n_tokens {
                let q_off = (t * q_stride) as usize;
                let k_off = (t * k_stride) as usize;
                let q_end = q_off + q_stride as usize;
                let k_end = k_off + k_stride as usize;
                let pos = (start_pos + t) as usize;
                if rope_type == 0 {
                    crate::backend::cpu::rope(
                        &mut ref_q[q_off..q_end],
                        &mut ref_k[k_off..k_end],
                        pos,
                        n_heads as usize,
                        n_kv_heads as usize,
                        head_dim as usize,
                        freq_base,
                    );
                } else {
                    crate::backend::cpu::rope_norm(
                        &mut ref_q[q_off..q_end],
                        &mut ref_k[k_off..k_end],
                        pos,
                        n_heads as usize,
                        n_kv_heads as usize,
                        head_dim as usize,
                        freq_base,
                        ff,
                    );
                }
            }

            let pipeline = ctx.create_pipeline(
                shaders::QK_NORM_ROPE_BATCH,
                "qk_norm_rope_batch",
                "qk_norm_rope_batch",
            );
            let q_buf = ctx.create_storage_rw((q_batch.len() as u64) * 4, "q");
            ctx.queue
                .write_buffer(&q_buf, 0, bytemuck::cast_slice(&q_batch));
            let k_buf = ctx.create_storage_rw((k_batch.len() as u64) * 4, "k");
            ctx.queue
                .write_buffer(&k_buf, 0, bytemuck::cast_slice(&k_batch));
            // has_qk_norm = 0 ⇒ slots 2/3 are dummies; reuse the freq buffer.
            let dummy = ctx.upload_f32(&[1.0f32], "qk_norm_dummy");
            let ff_buf = if has_freq_factors {
                ctx.upload_f32(&freqs, "freq_factors")
            } else {
                ctx.upload_f32(&[1.0f32], "freq_factors_dummy")
            };
            let params = [
                start_pos,
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                eps.to_bits(),
                freq_base.to_bits(),
                rope_type,
                q_stride,
                k_stride,
                has_freq_factors as u32,
                0, // has_qk_norm
            ];
            let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dummy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dummy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ff_buf.as_entire_binding(),
                    },
                ],
            });
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(n_tokens * (n_heads + n_kv_heads), 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));

            let got_q = ctx.download_f32(&q_buf, q_batch.len());
            let got_k = ctx.download_f32(&k_buf, k_batch.len());

            let tol = 2e-3f32;
            for i in 0..ref_q.len() {
                let diff = (ref_q[i] - got_q[i]).abs();
                assert!(
                    diff < tol,
                    "[{label}] Q mismatch at idx {i}: cpu={}, gpu={}, diff={diff}",
                    ref_q[i],
                    got_q[i]
                );
            }
            for i in 0..ref_k.len() {
                let diff = (ref_k[i] - got_k[i]).abs();
                assert!(
                    diff < tol,
                    "[{label}] K mismatch at idx {i}: cpu={}, gpu={}, diff={diff}",
                    ref_k[i],
                    got_k[i]
                );
            }
        }
    }

    /// `conv1d_fused_batch` parity: walking N tokens through the
    /// fused (x⊙b → conv → c⊙conv) pipeline must match the same
    /// sequence performed step-by-step on the CPU. Verifies the
    /// rolling-buffer carry-over across token boundaries — the
    /// non-trivial part vs. the per-token shader.
    #[test]
    fn test_gpu_conv1d_fused_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let hs: usize = 64;
        let kernel_size: usize = 4;
        let d_conv: usize = kernel_size - 1; // 3
        let n_tokens: usize = 5;
        let proj_stride = 3 * hs;
        let out_stride = hs;

        // Build proj as N tokens × (x | c | b), each segment hs floats.
        let mut proj: Vec<f32> = Vec::with_capacity(n_tokens * proj_stride);
        for t in 0..n_tokens {
            // x
            for i in 0..hs {
                proj.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.011);
            }
            // c
            for i in 0..hs {
                proj.push(((t as f32 + 2.0) * (i as f32 + 5.0)) * 0.007);
            }
            // b
            for i in 0..hs {
                proj.push(((t as f32 + 3.0) * (i as f32 - 16.0)) * 0.013);
            }
        }
        // Initial rolling buffer (d_conv × hs) — non-zero so the
        // first token's conv reads real prior context.
        let mut rb_initial: Vec<f32> = Vec::with_capacity(d_conv * hs);
        for k in 0..d_conv {
            for i in 0..hs {
                rb_initial.push(((k as f32 + 1.0) * (i as f32 - 8.0)) * 0.005);
            }
        }
        // Conv weights: hs × kernel_size, layout `weight[ch * ks + k]`.
        let mut weight: Vec<f32> = Vec::with_capacity(hs * kernel_size);
        for ch in 0..hs {
            for k in 0..kernel_size {
                weight.push(0.1 + (ch as f32 % 7.0) * 0.02 - (k as f32) * 0.03);
            }
        }

        // ─── CPU reference ────────────────────────────────────────
        let mut ref_out = vec![0.0f32; n_tokens * out_stride];
        let mut ref_rb = rb_initial.clone();
        for t in 0..n_tokens {
            let base = t * proj_stride;
            for ch in 0..hs {
                let x = proj[base + ch];
                let c = proj[base + hs + ch];
                let b = proj[base + 2 * hs + ch];
                let bx = x * b;

                let mut sum = 0.0f32;
                for k in 0..d_conv {
                    sum += ref_rb[k * hs + ch] * weight[ch * kernel_size + k];
                }
                sum += bx * weight[ch * kernel_size + d_conv];

                // Shift rolling buffer left; append bx at the tail.
                if d_conv > 1 {
                    for k in 0..d_conv - 1 {
                        ref_rb[k * hs + ch] = ref_rb[(k + 1) * hs + ch];
                    }
                }
                if d_conv > 0 {
                    ref_rb[(d_conv - 1) * hs + ch] = bx;
                }

                ref_out[t * out_stride + ch] = c * sum;
            }
        }

        // ─── Batched GPU run ──────────────────────────────────────
        let pipeline = ctx.create_pipeline(
            shaders::CONV1D_FUSED_BATCH,
            "conv1d_fused_batch",
            "conv1d_fused_batch",
        );
        let proj_buf = ctx.upload_f32(&proj, "proj");
        let rb_buf = ctx.create_storage_rw((rb_initial.len() as u64) * 4, "rb");
        ctx.queue
            .write_buffer(&rb_buf, 0, bytemuck::cast_slice(&rb_initial));
        let weight_buf = ctx.upload_f32(&weight, "weight");
        let out_buf = ctx.create_storage_rw((ref_out.len() as u64) * 4, "out");
        let params: [u32; 6] = [
            hs as u32,
            kernel_size as u32,
            d_conv as u32,
            n_tokens as u32,
            proj_stride as u32,
            out_stride as u32,
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(hs.div_ceil(256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got_out = ctx.download_f32(&out_buf, ref_out.len());
        let got_rb = ctx.download_f32(&rb_buf, ref_rb.len());

        let tol = 1e-4f32;
        for i in 0..ref_out.len() {
            let diff = (ref_out[i] - got_out[i]).abs();
            assert!(
                diff < tol,
                "out mismatch at idx {i} (token {}, ch {}): cpu={}, gpu={}, diff={diff}",
                i / hs,
                i % hs,
                ref_out[i],
                got_out[i]
            );
        }
        for i in 0..ref_rb.len() {
            let diff = (ref_rb[i] - got_rb[i]).abs();
            assert!(
                diff < tol,
                "rolling-buffer mismatch at idx {i}: cpu={}, gpu={}, diff={diff}",
                ref_rb[i],
                got_rb[i]
            );
        }
    }

    /// Single-token fused conv parity: `conv1d_fused.wgsl` (decode
    /// path) must match the CPU reference of `bx = x*b → conv → c*sum`
    /// plus the rolling-buffer update. Same scaffold as the batched
    /// twin's parity test, with `n_tokens = 1` and the new shader.
    /// Catches regressions in the body or the layout (proj packed
    /// [x, c, b] at offsets 0/hs/2*hs).
    #[test]
    fn test_gpu_conv1d_fused_decode_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // hs=1024 matches LFM2-VL-450M's decode-time hidden_size, so the
        // dispatch grid spans 4 workgroups (`(hs/256, 1, 1)`) — the actual
        // production shape. Smaller fixtures only exercise the single-WG
        // path and miss any cross-workgroup correctness issues.
        let hs: usize = 1024;
        let kernel_size: usize = 4;
        let d_conv: usize = kernel_size - 1; // 3

        // Single-token proj: [x | c | b], each segment hs floats.
        let mut proj: Vec<f32> = Vec::with_capacity(3 * hs);
        for i in 0..hs {
            proj.push((i as f32 - 32.0) * 0.011);
        }
        for i in 0..hs {
            proj.push((i as f32 + 5.0) * 0.007);
        }
        for i in 0..hs {
            proj.push((i as f32 - 16.0) * 0.013);
        }
        // Rolling buffer (d_conv × hs) — non-zero so the conv reads
        // real prior context, not just `bx * weight[d_conv]`.
        let mut rb_initial: Vec<f32> = Vec::with_capacity(d_conv * hs);
        for k in 0..d_conv {
            for i in 0..hs {
                rb_initial.push(((k as f32 + 1.0) * (i as f32 - 8.0)) * 0.005);
            }
        }
        // Weights: hs × kernel_size, layout `weight[ch * ks + k]`.
        let mut weight: Vec<f32> = Vec::with_capacity(hs * kernel_size);
        for ch in 0..hs {
            for k in 0..kernel_size {
                weight.push(0.1 + (ch as f32 % 7.0) * 0.02 - (k as f32) * 0.03);
            }
        }

        // ─── CPU reference ────────────────────────────────────────
        let mut ref_out = vec![0.0f32; hs];
        let mut ref_rb = rb_initial.clone();
        for ch in 0..hs {
            let x = proj[ch];
            let c = proj[hs + ch];
            let b = proj[2 * hs + ch];
            let bx = x * b;

            let mut sum = 0.0f32;
            for k in 0..d_conv {
                sum += ref_rb[k * hs + ch] * weight[ch * kernel_size + k];
            }
            sum += bx * weight[ch * kernel_size + d_conv];

            if d_conv > 1 {
                for k in 0..d_conv - 1 {
                    ref_rb[k * hs + ch] = ref_rb[(k + 1) * hs + ch];
                }
            }
            if d_conv > 0 {
                ref_rb[(d_conv - 1) * hs + ch] = bx;
            }

            ref_out[ch] = c * sum;
        }

        // ─── GPU run ──────────────────────────────────────────────
        let pipeline = ctx.create_pipeline(shaders::CONV1D_FUSED, "conv1d_fused", "conv1d_fused");
        let proj_buf = ctx.upload_f32(&proj, "proj");
        let rb_buf = ctx.create_storage_rw((rb_initial.len() as u64) * 4, "rb");
        ctx.queue
            .write_buffer(&rb_buf, 0, bytemuck::cast_slice(&rb_initial));
        let weight_buf = ctx.upload_f32(&weight, "weight");
        let out_buf = ctx.create_storage_rw((ref_out.len() as u64) * 4, "out");
        let params: [u32; 4] = [hs as u32, kernel_size as u32, d_conv as u32, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(hs.div_ceil(256) as u32, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got_out = ctx.download_f32(&out_buf, ref_out.len());
        let got_rb = ctx.download_f32(&rb_buf, ref_rb.len());

        let tol = 1e-4f32;
        for i in 0..ref_out.len() {
            let diff = (ref_out[i] - got_out[i]).abs();
            assert!(
                diff < tol,
                "out mismatch at ch {i}: cpu={}, gpu={}, diff={diff}",
                ref_out[i],
                got_out[i]
            );
        }
        for i in 0..ref_rb.len() {
            let diff = (ref_rb[i] - got_rb[i]).abs();
            assert!(
                diff < tol,
                "rolling-buffer mismatch at idx {i}: cpu={}, gpu={}, diff={diff}",
                ref_rb[i],
                got_rb[i]
            );
        }
    }

    /// `gemm_q4_0` parity: batched output[token, row] must match the
    /// CPU-side dequant + matmul at every (row, token) cell. Uses the
    /// same Q4_0 layout the gemv tests do (8 rows × small K).
    #[test]
    fn test_gpu_gemm_q4_0_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let m: u32 = 8;
        let k: u32 = 64;
        let n: u32 = 5; // tokens
        let nb = k / 32;
        let x_stride = k;
        let y_stride = m;

        // Build f32 weights, quantize to Q4_0.
        let weights_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.4)
            .collect();
        let mut q4_bytes: Vec<u8> = Vec::new();
        for row in 0..m as usize {
            for b in 0..nb as usize {
                let start = row * k as usize + b * 32;
                let chunk = &weights_f32[start..start + 32];
                let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 7.0;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let d_bits = half::f16::from_f32(scale).to_bits();
                q4_bytes.push((d_bits & 0xFF) as u8);
                q4_bytes.push((d_bits >> 8) as u8);
                for qi in 0..16 {
                    let lo = ((chunk[qi] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    let hi = ((chunk[qi + 16] * inv).round() + 8.0).clamp(0.0, 15.0) as u8;
                    q4_bytes.push(lo | (hi << 4));
                }
            }
        }

        // N input vectors of K floats each, each vector distinct.
        let mut x_batch: Vec<f32> = Vec::with_capacity((n * x_stride) as usize);
        for t in 0..n {
            for i in 0..k {
                x_batch.push(((t as f32 + 1.0) * (i as f32 - 32.0)) * 0.05);
            }
        }

        // ─── CPU reference: dequant Q4_0 + matmul row × x for each token ───
        let mut expected = vec![0.0f32; (n * m) as usize];
        for t in 0..n as usize {
            let x_slice = &x_batch[t * x_stride as usize..(t + 1) * x_stride as usize];
            for row in 0..m as usize {
                let mut acc = 0.0f32;
                for b in 0..nb as usize {
                    let block_off = (row * nb as usize + b) * 18;
                    let d_bits = u16::from_le_bytes([q4_bytes[block_off], q4_bytes[block_off + 1]]);
                    let delta = half::f16::from_bits(d_bits).to_f32();
                    for qi in 0..16 {
                        let byte = q4_bytes[block_off + 2 + qi];
                        let lo = (byte & 0xF) as f32 - 8.0;
                        let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                        acc += lo * delta * x_slice[b * 32 + qi];
                        acc += hi * delta * x_slice[b * 32 + qi + 16];
                    }
                }
                expected[t * m as usize + row] = acc;
            }
        }

        // ─── Batched GPU run ──────────────────────────────────────────────
        let a_buf = ctx.upload_storage(&q4_bytes, "weights");
        let x_buf = ctx.upload_f32(&x_batch, "x_batch");
        let y_buf = ctx.create_storage_rw(((n * y_stride) as u64) * 4, "y_batch");
        let params: [u32; 6] = [m, k, n, x_stride, y_stride, 0];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let pipeline = ctx.create_pipeline(shaders::GEMM_Q4_0, "gemm_q4_0", "gemm_q4_0");
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
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(4), n, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&y_buf, (n * y_stride) as usize);

        // Q4_0 quantization noise — same threshold the per-token test uses.
        for t in 0..n as usize {
            for row in 0..m as usize {
                let idx = t * m as usize + row;
                let diff = (expected[idx] - got[idx]).abs();
                assert!(
                    diff < 0.5,
                    "GEMM Q4_0 mismatch at (token {t}, row {row}): cpu={}, gpu={}, diff={diff}",
                    expected[idx],
                    got[idx]
                );
            }
        }
    }

    /// `add_rmsnorm_batch` parity: identical to running `add_inplace`
    /// on each vector + residual, then `rmsnorm_batch`. Confirms the
    /// fused kernel matches the unfused two-step sequence.
    #[test]
    fn test_gpu_add_rmsnorm_batch_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n: u32 = 1024;
        let batch: u32 = 5;
        let eps = 1e-5f32;

        let mut src: Vec<f32> = Vec::with_capacity((n * batch) as usize);
        let mut residual: Vec<f32> = Vec::with_capacity((n * batch) as usize);
        for b in 0..batch {
            for i in 0..n {
                src.push(((b + 1) as f32 * (i as f32 - 512.0)) * 0.001);
                residual.push(((b + 2) as f32 * ((i as f32 + 17.0) % 13.0)) * 0.002);
            }
        }
        let weight: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 7.0) * 0.05).collect();

        // Non-identity residual scale exercises Granite's `residual_multiplier`
        // fold (every other arch passes 1.0 ⇒ plain add).
        let res_scale = 0.7f32;

        // CPU reference: src += res_scale·residual, then rmsnorm with eps.
        let mut reference = src.clone();
        for i in 0..reference.len() {
            reference[i] += res_scale * residual[i];
        }
        for b in 0..batch {
            let row_start = (b * n) as usize;
            let row_end = row_start + n as usize;
            crate::backend::cpu::rmsnorm(&mut reference[row_start..row_end], &weight, eps);
        }

        // Batched fused run.
        let pipeline = ctx.create_pipeline(
            shaders::RMSNORM_BATCH,
            "add_rmsnorm_batch",
            "add_rmsnorm_batch",
        );
        let src_buf = ctx.create_storage_rw((src.len() as u64) * 4, "src");
        ctx.queue
            .write_buffer(&src_buf, 0, bytemuck::cast_slice(&src));
        let dst_buf = ctx.create_storage_rw((src.len() as u64) * 4, "dst");
        let res_buf = ctx.upload_f32(&residual, "residual");
        let w_buf = ctx.upload_f32(&weight, "w");
        let params = [n, eps.to_bits(), n, n, res_scale.to_bits()];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: res_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(batch, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let batched = ctx.download_f32(&dst_buf, (n * batch) as usize);

        for i in 0..(n * batch) as usize {
            let diff = (reference[i] - batched[i]).abs();
            assert!(
                diff < 1e-3,
                "add_rmsnorm_batch mismatch at idx {i} (token {}, dim {}): \
                 ref={}, batched={}, diff={diff}",
                i / n as usize,
                i % n as usize,
                reference[i],
                batched[i]
            );
        }
    }

    /// `attention_prefill` parity: batched attention over N queries
    /// matches a CPU reference (Q × K^T → causal-masked softmax → V) at
    /// every (token, head, dim) cell. Covers GQA (n_kv_heads < n_heads),
    /// non-zero start_pos, and a multi-query prefill.
    #[test]
    fn test_gpu_attention_prefill_parity() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let n_heads: u32 = 4;
        let n_kv_heads: u32 = 2; // GQA
        let head_dim: u32 = 32;
        let kv_dim = n_kv_heads * head_dim;
        let n_queries: u32 = 5;
        let start_pos: u32 = 3;
        // Tight `max_seq` (exactly `start_pos + n_queries`): the last
        // workgroup's `seq_len = pos_q + 1u = max_seq`, exercising the
        // boundary of the clamp added to address the PR review feedback.
        let max_seq = start_pos + n_queries;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_stride = n_heads * head_dim;
        let out_stride = n_heads * head_dim;
        let group_size = n_heads / n_kv_heads;

        // Build Q (per-token × per-head), K cache (max_seq × kv_dim), V cache.
        let mut q_batch = vec![0.0f32; (n_queries * q_stride) as usize];
        for q in 0..n_queries {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    let v = ((q as f32 + 1.0) * ((h as f32 + 1.0) * (d as f32 + 1.0))) * 0.013;
                    q_batch[(q * q_stride + h * head_dim + d) as usize] = v;
                }
            }
        }
        let mut k_cache = vec![0.0f32; (max_seq * kv_dim) as usize];
        let mut v_cache = vec![0.0f32; (max_seq * kv_dim) as usize];
        for t in 0..max_seq {
            for kh in 0..n_kv_heads {
                for d in 0..head_dim {
                    let kv = ((t as f32 + 1.0) * (kh as f32 + 1.0) * (d as f32 + 0.5)) * 0.011;
                    let vv = ((t as f32 + 1.0) * (kh as f32 + 2.0) * (d as f32 + 1.5)) * 0.017;
                    k_cache[(t * kv_dim + kh * head_dim + d) as usize] = kv;
                    v_cache[(t * kv_dim + kh * head_dim + d) as usize] = vv;
                }
            }
        }

        // ─── CPU reference: per-query, per-head attention with causal mask ──
        let mut ref_out = vec![0.0f32; (n_queries * out_stride) as usize];
        for q in 0..n_queries as usize {
            let pos_q = start_pos as usize + q;
            let seq_len = pos_q + 1;
            for h in 0..n_heads as usize {
                let kv_head = h / group_size as usize;
                let kv_h_off = kv_head * head_dim as usize;
                let q_off = q * q_stride as usize + h * head_dim as usize;

                // Q × K^T scores up to seq_len with scale.
                let mut scores = vec![0.0f32; seq_len];
                for t in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim as usize {
                        dot += q_batch[q_off + d] * k_cache[t * kv_dim as usize + kv_h_off + d];
                    }
                    scores[t] = dot * scale;
                }
                // Softmax.
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let inv = 1.0f32 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }
                // Weighted V → output.
                let out_off = q * out_stride as usize + h * head_dim as usize;
                for d in 0..head_dim as usize {
                    let mut val = 0.0f32;
                    for t in 0..seq_len {
                        val += scores[t] * v_cache[t * kv_dim as usize + kv_h_off + d];
                    }
                    ref_out[out_off + d] = val;
                }
            }
        }

        // ─── Batched GPU run ───────────────────────────────────────────────
        let pipeline = ctx.create_pipeline(
            shaders::ATTENTION_PREFILL,
            "attention_prefill",
            "attention_prefill",
        );
        let q_buf = ctx.upload_f32(&q_batch, "q");
        let k_buf = ctx.upload_f32(&k_cache, "k");
        let v_buf = ctx.upload_f32(&v_cache, "v");
        let out_buf = ctx.create_storage_rw((ref_out.len() as u64) * 4, "out");
        // Per-(query, head) scratch slab; sized to max_seq even though most
        // queries use less.
        let scores_buf =
            ctx.create_storage_rw(((n_queries * n_heads * max_seq) as u64) * 4, "scores");
        let params: [u32; 12] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            max_seq,
            scale.to_bits(),
            start_pos,
            n_queries,
            q_stride,
            out_stride,
            0,
            0,
        ];
        let p_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scores_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_heads, n_queries, 1);
        }
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&out_buf, ref_out.len());

        let tol = 1e-4f32;
        for i in 0..ref_out.len() {
            let diff = (ref_out[i] - got[i]).abs();
            assert!(
                diff < tol,
                "attention_prefill mismatch at idx {i} \
                 (token {}, head {}, dim {}): cpu={}, gpu={}, diff={diff}",
                i / out_stride as usize,
                (i % out_stride as usize) / head_dim as usize,
                i % head_dim as usize,
                ref_out[i],
                got[i]
            );
        }
    }
}

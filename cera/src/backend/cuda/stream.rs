// CUDA stream, event, and graph execution primitives.
//
// Wraps cudarc asynchronous execution queues with non-blocking stream creation,
// event recording and timing, and 1-token-per-launch CUDA Graph capture and replay.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaEvent as RawCudaEvent, CudaGraph as RawCudaGraph, CudaStream as RawCudaStream,
    sys,
};

/// Asynchronous CUDA execution stream.
#[derive(Debug, Clone)]
pub struct CudaStream {
    pub(crate) stream: Arc<RawCudaStream>,
}

impl CudaStream {
    /// Create a new non-blocking CUDA stream.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx
            .new_stream()
            .context("failed to create non-blocking CUDA stream")?;
        Ok(Self { stream })
    }

    /// Get the default (legacy/null) stream for the context.
    pub fn default_stream(ctx: &Arc<CudaContext>) -> Self {
        Self {
            stream: ctx.default_stream(),
        }
    }

    /// Synchronize the stream, blocking host CPU until all queued work finishes.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .context("failed to synchronize CUDA stream")?;
        Ok(())
    }

    /// Make this stream wait on an event before executing subsequent commands.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        self.stream
            .wait(&event.event)
            .context("failed to enqueue stream wait on event")?;
        Ok(())
    }

    /// Begin capturing CUDA operations on this stream into a CUDA Graph.
    pub fn begin_capture(&self) -> Result<()> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .context("failed to begin CUDA stream capture")?;
        Ok(())
    }

    /// End graph capture and instantiate an executable CUDA Graph.
    pub fn end_capture(&self) -> Result<CudaGraph> {
        let raw_graph = self
            .stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .context("failed to instantiate CUDA Graph")?
            .context("CUDA graph capture returned null graph")?;
        Ok(CudaGraph { graph: raw_graph })
    }

    /// Underlying CUstream handle.
    pub fn cu_stream(&self) -> sys::CUstream {
        self.stream.cu_stream()
    }

    /// Create a kernel launch builder for submitting work to this stream.
    pub fn launch_builder<'a>(
        &'a self,
        kernel: &'a super::module::CudaKernel,
    ) -> cudarc::driver::LaunchArgs<'a> {
        self.stream.launch_builder(&kernel.func)
    }

    /// Underlying cudarc stream reference.
    pub fn inner(&self) -> &Arc<RawCudaStream> {
        &self.stream
    }
}

/// Synchronization and timing event on a CUDA stream.
pub struct CudaEvent {
    pub(crate) event: RawCudaEvent,
}

impl CudaEvent {
    /// Create a new CUDA event.
    ///
    /// If `enable_timing` is false, the event disables timing overhead.
    pub fn new(ctx: &Arc<CudaContext>, enable_timing: bool) -> Result<Self> {
        let flags = if enable_timing {
            Some(sys::CUevent_flags::CU_EVENT_DEFAULT)
        } else {
            Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
        };
        let event = ctx
            .new_event(flags)
            .context("failed to create CUDA event")?;
        Ok(Self { event })
    }

    /// Record this event on the specified stream.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        self.event
            .record(&stream.stream)
            .context("failed to record CUDA event")?;
        Ok(())
    }

    /// Synchronize the calling thread until this event completes.
    pub fn synchronize(&self) -> Result<()> {
        self.event
            .synchronize()
            .context("failed to synchronize CUDA event")?;
        Ok(())
    }

    /// Calculate elapsed time in milliseconds between `self` (start) and `end`.
    pub fn elapsed_ms(&self, end: &CudaEvent) -> Result<f32> {
        self.event
            .elapsed_ms(&end.event)
            .context("failed to compute elapsed time between CUDA events")
    }

    /// Check if all work prior to this event has completed without blocking.
    pub fn is_complete(&self) -> bool {
        self.event.is_complete()
    }
}

/// Instantiated CUDA Graph ready for fast deterministic replay.
///
/// Used for single-token autoregressive decoding to bypass host API launch overhead
/// while streaming tokens immediately to downstream audio/TTS pipelines.
pub struct CudaGraph {
    pub(crate) graph: RawCudaGraph,
}

impl CudaGraph {
    /// Launch the captured graph on its assigned stream.
    pub fn launch(&self) -> Result<()> {
        self.graph.launch().context("failed to launch CUDA Graph")?;
        Ok(())
    }

    /// Pre-upload graph resources to the GPU to eliminate first-launch latency spikes.
    pub fn upload(&self) -> Result<()> {
        self.graph
            .upload()
            .context("failed to upload CUDA Graph resources")?;
        Ok(())
    }
}

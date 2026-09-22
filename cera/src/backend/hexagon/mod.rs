//! Native Qualcomm Hexagon NPU backend for Cera.
//!
//! Provides hardware-accelerated tensor operations on Snapdragon Hexagon Tensor Processors (HTP)
//! through FastRPC shared memory (`rpcmem`) and asynchronous command queues (`dspqueue`).

pub mod device;
pub mod params;
pub mod queue;
pub mod repack;
pub mod rpcmem;
pub mod sys;
pub mod types;

pub use device::{HexagonArch, HexagonDevice};
pub use params::*;
pub use queue::{HexagonOpBatch, HexagonQueueSession};
pub use repack::*;
pub use rpcmem::RpcmemBuffer;
pub use sys::FastRpcDriver;
pub use types::*;

use std::sync::Arc;
use crate::session::CeraError;

/// Shared Hexagon compute context managing the FastRPC driver and device session.
pub struct HexagonContext {
    driver: Arc<FastRpcDriver>,
}

// HexagonContext manages thread-safe driver access.
unsafe impl Send for HexagonContext {}
unsafe impl Sync for HexagonContext {}

impl HexagonContext {
    /// Initialize the Hexagon backend by loading the FastRPC userspace driver.
    pub fn new() -> Result<Arc<Self>, CeraError> {
        let driver = FastRpcDriver::load()?;
        Ok(Arc::new(Self { driver }))
    }

    /// Access the underlying FastRPC driver handle.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }
}

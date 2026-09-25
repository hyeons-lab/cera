//! Native Qualcomm Hexagon NPU backend for Cera.
//!
//! Provides hardware-accelerated tensor operations on Snapdragon Hexagon Tensor Processors (HTP)
//! through FastRPC shared memory (`rpcmem`) and asynchronous command queues (`dspqueue`).

pub mod adpf;
pub mod device;
pub mod params;
pub mod queue;
pub mod repack;
pub mod rpcmem;
pub mod skels;
pub mod sys;
pub mod types;

pub use adpf::AdpfSession;
pub use device::{HexagonArch, HexagonDevice};
pub use params::*;
pub use queue::HexagonQueueSession;
pub use repack::*;
pub use rpcmem::RpcmemBuffer;
pub use skels::{HexagonProbe, PROBE_ARCHS, embedded_skel, install_skels, probe};
pub use sys::FastRpcDriver;
pub use types::*;

use crate::session::CeraError;
use std::sync::Arc;

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
        if std::env::var("CERA_HEXAGON_SPIN")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            // Debug knob: park a spinner to hold CPU clocks across DSP-bound
            // waits (validates governor effects; burns a core — ADPF is the
            // production answer). Detached: runs until process exit. `OnceLock`:
            // without it every model reload parks another thread and the
            // knob skews the runs it was meant to stabilize; the stored
            // `Result` keeps a spawn failure visible to every caller instead
            // of only the first.
            static SPIN_RESULT: std::sync::OnceLock<Result<(), CeraError>> =
                std::sync::OnceLock::new();
            SPIN_RESULT
                .get_or_init(|| {
                    tracing::warn!(
                        "cera-hexagon: CERA_HEXAGON_SPIN=1 parking a keepalive spinner thread"
                    );
                    std::thread::Builder::new()
                        .name("cera-hex-spin".into())
                        .spawn(|| {
                            loop {
                                std::hint::spin_loop();
                            }
                        })
                        .map(|_| ())
                        .map_err(|e| CeraError::Backend(format!("spinner spawn failed: {e}")))
                })
                .as_ref()
                .map_err(|e| CeraError::Backend(e.to_string()))?;
        }
        Ok(Arc::new(Self { driver }))
    }

    /// Access the underlying FastRPC driver handle.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }
}

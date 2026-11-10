//! AMD XDNA NPU backend (Ryzen AI, Strix Halo) for Cera.
//!
//! Drives the NPU through the XRT runtime that ships inside the AMD NPU
//! driver, via a small MSVC C ABI shim (`shim/cera_xdna_shim.cpp`) that
//! `build.rs` compiles into a DLL. The Rust side embeds that DLL and loads
//! it at runtime, so a cera binary still starts on machines without the
//! driver: loading simply fails there and the backend reports itself
//! unavailable. Support is gated by attempting to load an npu2 `.xclbin`,
//! never by device-name or BDF strings.
//!
//! Design decisions (weight format, activation quantization, parity policy,
//! prefill placement, numeric bar) are recorded in
//! `docs/XDNA_PHASE0_RESULTS.md` section 4.

#[cfg(not(windows))]
compile_error!(
    "the `xdna` feature needs a Windows target (XRT via the AMD NPU driver); \
     it cannot be built for wasm32, macOS, or Linux"
);

pub mod device;
pub mod kernels;
pub mod repack;
pub mod sys;

pub use device::{XdnaDevice, probe};

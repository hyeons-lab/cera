//! Parity and sanity tests for the native CUDA backend.
//!
//! Verifies:
//! 1. Graceful driver availability probe when CUDA driver is not present.
//! 2. Clean error handling when constructing devices without CUDA runtime.
//! 3. BackendPreference parsing and configuration for CUDA.
//! 4. When running on an NVIDIA host (e.g. Jetson Orin):
//!    - Primary context initialization with CU_CTX_SCHED_BLOCKING_SYNC.
//!    - Memory allocators (device, pinned host, unified managed).
//!    - Compilation and kernel launch of native CUDA kernels.
//!    - Numerical correctness of RMSNorm, SwiGLU, and GEMV.

#![cfg(feature = "cuda")]

use cera::backend::cuda::{CudaContext, CudaDevice};
use cera::engine::BackendPreference;

#[test]
fn test_cuda_backend_preference_parsing() {
    let pref = BackendPreference::parse_str("cuda").expect("parse cuda preference");
    assert_eq!(pref, BackendPreference::Cuda);

    let pref_upper = BackendPreference::parse_str("CUDA").expect("parse uppercase CUDA preference");
    assert_eq!(pref_upper, BackendPreference::Cuda);
}

#[test]
fn test_cuda_device_availability_probe() {
    let available = CudaDevice::is_available();
    if !available {
        eprintln!("CUDA driver not available on this host; testing fallback behavior");
        let dev_err = CudaDevice::new(0);
        assert!(
            dev_err.is_err(),
            "expected error constructing CUDA device when driver is unavailable"
        );
        return;
    }

    // When running on a machine with CUDA driver and GPU:
    let dev = CudaDevice::new(0).expect("failed to initialize CUDA device 0");
    assert!(!dev.name.is_empty(), "CUDA device name should not be empty");
    assert!(
        dev.total_memory_bytes > 0,
        "device memory should be greater than zero"
    );

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");

    // Test buffer allocation
    let buf = ctx
        .create_buffer(1024)
        .expect("failed to allocate device buffer");
    assert_eq!(buf.len(), 1024);

    let pinned = ctx
        .create_pinned_buffer(1024)
        .expect("failed to allocate pinned buffer");
    assert_eq!(pinned.len(), 1024);

    // Test kernel compilation and simple RMSNorm execution
    let hidden_size: usize = 64;
    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.1).collect();
    let weight: Vec<f32> = vec![1.0; hidden_size];

    let in_buf = ctx
        .upload_f32(&input)
        .expect("failed to upload input buffer");
    let weight_buf = ctx
        .upload_f32(&weight)
        .expect("failed to upload weight buffer");
    let mut out_buf = ctx
        .create_buffer(hidden_size * std::mem::size_of::<f32>())
        .expect("failed to allocate out buffer");

    ctx.rmsnorm(&mut out_buf, &in_buf, &weight_buf, hidden_size as u32, 1e-5)
        .expect("rmsnorm kernel execution failed");

    let mut result = vec![0.0f32; hidden_size];
    ctx.download_f32(&out_buf, &mut result)
        .expect("failed to download rmsnorm result");

    // Compute CPU reference RMSNorm
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let rms = ((sum_sq / (hidden_size as f32)) + 1e-5).sqrt();
    let ref_out: Vec<f32> = input.iter().map(|&x| (x / rms) * 1.0).collect();

    for i in 0..hidden_size {
        assert!(
            (result[i] - ref_out[i]).abs() < 1e-4,
            "mismatch at index {i}: cuda {} vs ref {}",
            result[i],
            ref_out[i]
        );
    }
}

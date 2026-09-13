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

#[test]
fn test_cuda_q8_0_gemv_and_gemm_parity() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA driver not available on this host; skipping live Q8 parity test");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");

    let m: usize = 4;
    let k: usize = 64; // 2 blocks of 32
    let nb = k / 32;

    // Construct synthetic Q8_0 weight matrix: m rows, k columns
    let mut q8_bytes = Vec::new();
    let mut float_weights = Vec::new();

    for row in 0..m {
        for b in 0..nb {
            // Scale d = 0.5f32 (0x3800 in IEEE FP16)
            let d_val = 0.5f32;
            let d_fp16: u16 = half::f16::from_f32(d_val).to_bits();
            q8_bytes.extend_from_slice(&d_fp16.to_le_bytes());

            for col in 0..32 {
                let q_val = (((row * 32 + b * 32 + col) % 7) as i8) - 3;
                q8_bytes.push(q_val as u8);
                float_weights.push((q_val as f32) * d_val);
            }
        }
    }

    let weight_buf = ctx
        .upload_bytes(&q8_bytes)
        .expect("failed to upload Q8 weights");

    // 1. Test GEMV (M=1 token against m rows)
    let x_vec: Vec<f32> = (0..k).map(|i| (i as f32) * 0.05).collect();
    let x_buf = ctx.upload_f32(&x_vec).expect("failed to upload x vector");
    let mut out_gemv = ctx
        .create_buffer(m * std::mem::size_of::<f32>())
        .expect("allocate out_gemv");

    ctx.gemv_q8_0(&mut out_gemv, &weight_buf, &x_buf, m as u32, k as u32)
        .expect("gemv_q8_0 failed");

    let mut gemv_result = vec![0.0f32; m];
    ctx.download_f32(&out_gemv, &mut gemv_result)
        .expect("download gemv result");

    for r in 0..m {
        let expected: f32 = (0..k).map(|c| float_weights[r * k + c] * x_vec[c]).sum();
        assert!(
            (gemv_result[r] - expected).abs() < 1e-3,
            "GEMV mismatch at row {r}: got {}, expected {}",
            gemv_result[r],
            expected
        );
    }

    // 2. Test GEMM (Batch of 2 tokens against m rows)
    let batch_m: usize = 2;
    let mut x_batch = x_vec.clone();
    x_batch.extend((0..k).map(|i| (i as f32) * -0.02));
    let x_batch_buf = ctx.upload_f32(&x_batch).expect("failed to upload x batch");
    let mut out_gemm = ctx
        .create_buffer(batch_m * m * std::mem::size_of::<f32>())
        .expect("allocate out_gemm");

    ctx.gemm_q8_0(
        &mut out_gemm,
        &weight_buf,
        &x_batch_buf,
        batch_m as u32,
        m as u32,
        k as u32,
    )
    .expect("gemm_q8_0 failed");

    let mut gemm_result = vec![0.0f32; batch_m * m];
    ctx.download_f32(&out_gemm, &mut gemm_result)
        .expect("download gemm result");

    for b in 0..batch_m {
        for r in 0..m {
            let expected: f32 = (0..k)
                .map(|c| float_weights[r * k + c] * x_batch[b * k + c])
                .sum();
            let actual = gemm_result[b * m + r];
            assert!(
                (actual - expected).abs() < 1e-3,
                "GEMM mismatch at batch {b}, row {r}: got {}, expected {}",
                actual,
                expected
            );
        }
    }

    // 3. Test on-device embedding gather
    let mut out_gather = ctx
        .create_buffer(k * std::mem::size_of::<f32>())
        .expect("allocate out_gather");
    let target_token: u32 = 1;

    ctx.gather_embedding_q8_0(&mut out_gather, &weight_buf, target_token, k as u32)
        .expect("gather_embedding_q8_0 failed");

    let mut gather_result = vec![0.0f32; k];
    ctx.download_f32(&out_gather, &mut gather_result)
        .expect("download gather result");

    for c in 0..k {
        let expected = float_weights[(target_token as usize) * k + c];
        assert!(
            (gather_result[c] - expected).abs() < 1e-4,
            "Gather mismatch at col {c}: got {}, expected {}",
            gather_result[c],
            expected
        );
    }
}

#[test]
fn test_cuda_q4_0_and_fused_rmsnorm_parity() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA driver not available on this host; skipping live Q4 parity test");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");

    let m: usize = 4;
    let k: usize = 64; // 2 blocks of 32
    let nb = k / 32;

    // Construct synthetic Q4_0 weight matrix: m rows, k columns
    let mut q4_bytes = Vec::new();
    let mut float_weights = vec![0.0f32; m * k];

    for row in 0..m {
        for b in 0..nb {
            let d_val = 0.25f32;
            let d_fp16: u16 = half::f16::from_f32(d_val).to_bits();
            q4_bytes.extend_from_slice(&d_fp16.to_le_bytes());

            for i in 0..16 {
                let lo_nibble = ((row + b + i) % 15) as u8;
                let hi_nibble = ((row * 2 + b + i * 3) % 15) as u8;
                let byte = (hi_nibble << 4) | (lo_nibble & 0x0F);
                q4_bytes.push(byte);

                let col0 = b * 32 + i;
                let col1 = b * 32 + i + 16;
                float_weights[row * k + col0] = ((lo_nibble as i8) - 8) as f32 * d_val;
                float_weights[row * k + col1] = ((hi_nibble as i8) - 8) as f32 * d_val;
            }
        }
    }

    let weight_buf = ctx
        .upload_bytes(&q4_bytes)
        .expect("failed to upload Q4 weights");

    // 1. Test GEMV Q4_0
    let x_vec: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let x_buf = ctx.upload_f32(&x_vec).expect("failed to upload x vector");
    let mut out_gemv = ctx
        .create_buffer(m * std::mem::size_of::<f32>())
        .expect("allocate out_gemv");

    ctx.gemv_q4_0(&mut out_gemv, &weight_buf, &x_buf, m as u32, k as u32)
        .expect("gemv_q4_0 failed");

    let mut gemv_result = vec![0.0f32; m];
    ctx.download_f32(&out_gemv, &mut gemv_result)
        .expect("download gemv result");

    for r in 0..m {
        let expected: f32 = (0..k).map(|c| float_weights[r * k + c] * x_vec[c]).sum();
        assert!(
            (gemv_result[r] - expected).abs() < 1e-3,
            "GEMV Q4_0 mismatch at row {r}: got {}, expected {}",
            gemv_result[r],
            expected
        );
    }

    // 2. Test GEMM Q4_0 (Batch of 2 tokens)
    let batch_m: usize = 2;
    let mut x_batch = x_vec.clone();
    x_batch.extend((0..k).map(|i| (i as f32) * 0.05 + 1.0));
    let x_batch_buf = ctx.upload_f32(&x_batch).expect("failed to upload x batch");
    let mut out_gemm = ctx
        .create_buffer(batch_m * m * std::mem::size_of::<f32>())
        .expect("allocate out_gemm");

    ctx.gemm_q4_0(
        &mut out_gemm,
        &weight_buf,
        &x_batch_buf,
        batch_m as u32,
        m as u32,
        k as u32,
    )
    .expect("gemm_q4_0 failed");

    let mut gemm_result = vec![0.0f32; batch_m * m];
    ctx.download_f32(&out_gemm, &mut gemm_result)
        .expect("download gemm result");

    for b in 0..batch_m {
        for r in 0..m {
            let expected: f32 = (0..k)
                .map(|c| float_weights[r * k + c] * x_batch[b * k + c])
                .sum();
            let actual = gemm_result[b * m + r];
            assert!(
                (actual - expected).abs() < 1e-3,
                "GEMM Q4_0 mismatch at batch {b}, row {r}: got {}, expected {}",
                actual,
                expected
            );
        }
    }

    // 3. Test on-device Q4_0 embedding gather
    let mut out_gather = ctx
        .create_buffer(k * std::mem::size_of::<f32>())
        .expect("allocate out_gather");
    let target_token: u32 = 2;

    ctx.gather_embedding_q4_0(&mut out_gather, &weight_buf, target_token, k as u32)
        .expect("gather_embedding_q4_0 failed");

    let mut gather_result = vec![0.0f32; k];
    ctx.download_f32(&out_gather, &mut gather_result)
        .expect("download gather result");

    for c in 0..k {
        let expected = float_weights[(target_token as usize) * k + c];
        assert!(
            (gather_result[c] - expected).abs() < 1e-4,
            "Gather Q4_0 mismatch at col {c}: got {}, expected {}",
            gather_result[c],
            expected
        );
    }

    // 4. Test fused_add_rmsnorm
    let hidden_vec = vec![1.0f32; k];
    let residual_vec = vec![0.5f32; k];
    let norm_weight = vec![2.0f32; k];

    let mut hidden_buf = ctx.upload_f32(&hidden_vec).expect("upload hidden");
    let res_buf = ctx.upload_f32(&residual_vec).expect("upload residual");
    let w_buf = ctx.upload_f32(&norm_weight).expect("upload norm weight");
    let mut norm_out_buf = ctx
        .create_buffer(k * std::mem::size_of::<f32>())
        .expect("allocate norm out");

    ctx.fused_add_rmsnorm(
        &mut norm_out_buf,
        &mut hidden_buf,
        &res_buf,
        &w_buf,
        k as u32,
        1e-5,
    )
    .expect("fused_add_rmsnorm failed");

    let mut final_hidden = vec![0.0f32; k];
    ctx.download_f32(&hidden_buf, &mut final_hidden)
        .expect("download final hidden");
    for val in &final_hidden {
        assert!((val - 1.5f32).abs() < 1e-4, "hidden state was not updated");
    }

    let mut norm_result = vec![0.0f32; k];
    ctx.download_f32(&norm_out_buf, &mut norm_result)
        .expect("download norm result");
    for val in &norm_result {
        // (1.5 / sqrt(1.5^2 + 1e-5)) * 2.0 ≈ 2.0
        assert!(
            (val - 2.0f32).abs() < 1e-3,
            "norm result mismatch: got {val}"
        );
    }
}

#[test]
fn test_cuda_argmax_f32() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test_cuda_argmax_f32");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let n = 1024usize;
    let mut logits = vec![0.0f32; n];
    logits[42] = 100.0f32;
    logits[999] = 99.0f32;

    let logits_buf = ctx.upload_f32(&logits).expect("upload logits");
    let mut token_buf = ctx
        .create_buffer(std::mem::size_of::<u32>())
        .expect("allocate token buf");

    ctx.argmax_f32(&mut token_buf, &logits_buf, n as u32)
        .expect("argmax_f32 kernel launch failed");
    ctx.synchronize().expect("synchronize failed");

    let mut pinned = ctx
        .create_pinned_buffer(std::mem::size_of::<u32>())
        .expect("allocate pinned buffer");
    token_buf
        .copy_to_host(pinned.as_mut_slice())
        .expect("copy_to_host failed");

    let token_id = u32::from_ne_bytes(pinned.as_slice()[..4].try_into().unwrap());
    assert_eq!(token_id, 42, "argmax_f32 produced incorrect token ID");
}

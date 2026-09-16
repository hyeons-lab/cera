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

use cera::backend::cuda::{CudaContext, CudaDevice, QkNormRopeParams};
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

    // Test argmax_f32_pinned zero-copy UMA direct write
    let mut pinned_direct = ctx
        .create_pinned_buffer(std::mem::size_of::<u32>())
        .expect("allocate pinned buffer for direct write");
    ctx.argmax_f32_pinned(&mut pinned_direct, &logits_buf, n as u32)
        .expect("argmax_f32_pinned kernel launch failed");
    ctx.synchronize().expect("synchronize failed");

    let direct_token_id = u32::from_ne_bytes(pinned_direct.as_slice()[..4].try_into().unwrap());
    assert_eq!(
        direct_token_id, 42,
        "argmax_f32_pinned produced incorrect token ID"
    );
}

#[test]
fn test_cuda_argmax_masked_logits() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test_cuda_argmax_masked_logits");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let n = 2048usize;
    // Set all logits to -f32::MAX (masked out), except the very last one
    let mut logits = vec![-f32::MAX; n];
    let winning_idx = n - 1;
    logits[winning_idx] = -100.0f32;

    let logits_buf = ctx.upload_f32(&logits).expect("upload logits");
    let mut pinned_direct = ctx
        .create_pinned_buffer(std::mem::size_of::<u32>())
        .expect("allocate pinned buffer");

    ctx.argmax_f32_pinned(&mut pinned_direct, &logits_buf, n as u32)
        .expect("argmax_f32_pinned kernel launch failed");
    ctx.synchronize().expect("synchronize failed");

    let direct_token_id = u32::from_ne_bytes(pinned_direct.as_slice()[..4].try_into().unwrap());
    assert_eq!(
        direct_token_id as usize, winning_idx,
        "argmax_f32_pinned failed tie-breaker on masked logits: expected {winning_idx}, got {direct_token_id}"
    );
}

#[test]
fn test_cuda_pinned_buffer_zero_length() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test_cuda_pinned_buffer_zero_length");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let pinned = ctx
        .create_pinned_buffer(0)
        .expect("allocate zero-len pinned buffer");
    assert_eq!(pinned.len(), 0);
    assert!(pinned.is_empty());
    assert!(pinned.as_slice().is_empty());
}

#[test]
fn test_cuda_append_kv_cache_f16() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test_cuda_append_kv_cache_f16");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let n = 64usize;
    let k_vec: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let v_vec: Vec<f32> = (0..n).map(|i| (i as f32 * -0.2) + 1.0).collect();

    let k_src = ctx.upload_f32(&k_vec).expect("upload k");
    let v_src = ctx.upload_f32(&v_vec).expect("upload v");

    let max_len = 2;
    let k_cache = ctx
        .create_buffer(max_len * n * std::mem::size_of::<u16>())
        .expect("create k cache");
    let v_cache = ctx
        .create_buffer(max_len * n * std::mem::size_of::<u16>())
        .expect("create v cache");

    let offset = n;
    ctx.append_kv_cache_f16(&k_src, &v_src, &k_cache, &v_cache, offset, n as u32)
        .expect("append_kv_cache_f16 failed");
    ctx.synchronize().expect("synchronize failed");

    let mut k_out = vec![0u16; max_len * n];
    k_cache
        .copy_to_host(bytemuck::cast_slice_mut(&mut k_out))
        .expect("download k_cache");

    let mut v_out = vec![0u16; max_len * n];
    v_cache
        .copy_to_host(bytemuck::cast_slice_mut(&mut v_out))
        .expect("download v_cache");

    for i in 0..n {
        let expected_k = half::f16::from_f32(k_vec[i]).to_bits();
        let expected_v = half::f16::from_f32(v_vec[i]).to_bits();
        assert_eq!(k_out[offset + i], expected_k, "k mismatch at index {i}");
        assert_eq!(v_out[offset + i], expected_v, "v mismatch at index {i}");
    }
}

#[test]
fn test_cuda_qk_norm_rope_precomputed_inv_freq() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test_cuda_qk_norm_rope_precomputed_inv_freq");
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let n_heads = 2u32;
    let n_kv_heads = 1u32;
    let head_dim = 64u32;
    let half_dim = (head_dim / 2).min(64) as usize;
    let freq_base = 10000.0f32;
    let pos = 5u32;

    // Precompute inverse frequencies on CPU
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let inv_freqs: Vec<f32> = (0..half_dim).map(|i| theta_scale.powf(i as f32)).collect();
    let rope_inv_freq_buf = ctx.upload_f32(&inv_freqs).expect("upload rope_inv_freq");

    let q_len = (n_heads * head_dim) as usize;
    let k_len = (n_kv_heads * head_dim) as usize;
    let q_init: Vec<f32> = (0..q_len).map(|i| (i as f32) * 0.05 + 0.1).collect();
    let k_init: Vec<f32> = (0..k_len).map(|i| (i as f32) * -0.03 + 0.5).collect();

    let mut q_buf = ctx.upload_f32(&q_init).expect("upload q");
    let mut k_buf = ctx.upload_f32(&k_init).expect("upload k");

    let params = QkNormRopeParams {
        pos,
        n_heads,
        n_kv_heads,
        head_dim,
        eps: 1e-5,
        freq_base,
        rope_type: 0, // NeoX
        has_freq_factors: 0,
        has_qk_norm: 0,
    };

    ctx.qk_norm_rope(
        &mut q_buf,
        &mut k_buf,
        None,
        None,
        Some(&rope_inv_freq_buf),
        params,
    )
    .expect("qk_norm_rope failed");
    ctx.synchronize().expect("synchronize failed");

    let mut q_out = vec![0.0f32; q_len];
    ctx.download_f32(&q_buf, &mut q_out).expect("download q");

    // Check against CPU reference RoPE
    for h in 0..n_heads as usize {
        for d in 0..half_dim {
            let theta = (pos as f32) * inv_freqs[d];
            let cos_a = theta.cos();
            let sin_a = theta.sin();
            let x0 = q_init[h * head_dim as usize + d];
            let x1 = q_init[h * head_dim as usize + d + half_dim];
            let expected_0 = x0 * cos_a - x1 * sin_a;
            let expected_1 = x0 * sin_a + x1 * cos_a;

            let actual_0 = q_out[h * head_dim as usize + d];
            let actual_1 = q_out[h * head_dim as usize + d + half_dim];
            assert!(
                (actual_0 - expected_0).abs() < 1e-4,
                "Q head {h} dim {d} mismatch: {actual_0} vs {expected_0}"
            );
            assert!(
                (actual_1 - expected_1).abs() < 1e-4,
                "Q head {h} dim {d}+half mismatch: {actual_1} vs {expected_1}"
            );
        }
    }
}

#[test]
fn test_cuda_q4_0_concat3_and_swiglu_parity() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA driver not available on this host; skipping live Q4_0 concat3/swiglu test");
        return;
    }

    let ctx = CudaContext::new(0).expect("initialize CUDA context");

    let m1: usize = 4;
    let m2: usize = 2;
    let m3: usize = 2;
    let k: usize = 64;
    let nb = k / 32;

    let gen_q4_0 = |m: usize, offset: usize| -> (Vec<u8>, Vec<f32>) {
        let mut raw = Vec::new();
        let mut floats = Vec::new();
        for r in 0..m {
            for b in 0..nb {
                let d_val = 0.25f32 + ((r + offset) as f32) * 0.05;
                let d_fp16: u16 = half::f16::from_f32(d_val).to_bits();
                raw.extend_from_slice(&d_fp16.to_le_bytes());

                let mut q_nibbles = [0i8; 32];
                for (i, slot) in q_nibbles.iter_mut().enumerate() {
                    let q = (((r * 32 + b * 32 + i + offset) % 15) as i8) - 7;
                    *slot = q;
                    floats.push((q as f32) * d_val);
                }
                for i in 0..16 {
                    let lo = (q_nibbles[i] + 8) as u8 & 0x0F;
                    let hi = (q_nibbles[i + 16] + 8) as u8 & 0x0F;
                    raw.push(lo | (hi << 4));
                }
            }
        }
        (raw, floats)
    };

    let (q1_raw, _) = gen_q4_0(m1, 1);
    let (q2_raw, _) = gen_q4_0(m2, 5);
    let (q3_raw, _) = gen_q4_0(m3, 9);

    let b1 = ctx.upload_bytes(&q1_raw).expect("upload b1");
    let b2 = ctx.upload_bytes(&q2_raw).expect("upload b2");
    let b3 = ctx.upload_bytes(&q3_raw).expect("upload b3");

    let x_vec: Vec<f32> = (0..k).map(|i| (i as f32) * 0.02 + 0.1).collect();
    let x_buf = ctx.upload_f32(&x_vec).expect("upload x");

    let mut sep_y1 = ctx.create_buffer(m1 * 4).expect("alloc sep_y1");
    let mut sep_y2 = ctx.create_buffer(m2 * 4).expect("alloc sep_y2");
    let mut sep_y3 = ctx.create_buffer(m3 * 4).expect("alloc sep_y3");

    ctx.gemv_q4_0(&mut sep_y1, &b1, &x_buf, m1 as u32, k as u32)
        .expect("gemv 1");
    ctx.gemv_q4_0(&mut sep_y2, &b2, &x_buf, m2 as u32, k as u32)
        .expect("gemv 2");
    ctx.gemv_q4_0(&mut sep_y3, &b3, &x_buf, m3 as u32, k as u32)
        .expect("gemv 3");

    let mut cat_y1 = ctx.create_buffer(m1 * 4).expect("alloc cat_y1");
    let mut cat_y2 = ctx.create_buffer(m2 * 4).expect("alloc cat_y2");
    let mut cat_y3 = ctx.create_buffer(m3 * 4).expect("alloc cat_y3");

    ctx.gemv_q4_0_concat3(
        &mut cat_y1,
        &mut cat_y2,
        &mut cat_y3,
        &b1,
        &b2,
        &b3,
        &x_buf,
        m1 as u32,
        m2 as u32,
        m3 as u32,
        k as u32,
    )
    .expect("gemv_q4_0_concat3");
    ctx.synchronize().expect("sync");

    let mut r_sep1 = vec![0.0f32; m1];
    let mut r_cat1 = vec![0.0f32; m1];
    ctx.download_f32(&sep_y1, &mut r_sep1).expect("dl sep1");
    ctx.download_f32(&cat_y1, &mut r_cat1).expect("dl cat1");
    for i in 0..m1 {
        assert!(
            (r_sep1[i] - r_cat1[i]).abs() < 1e-4,
            "concat3 y1 mismatch at {i}: {} vs {}",
            r_sep1[i],
            r_cat1[i]
        );
    }

    let mut r_sep2 = vec![0.0f32; m2];
    let mut r_cat2 = vec![0.0f32; m2];
    ctx.download_f32(&sep_y2, &mut r_sep2).expect("dl sep2");
    ctx.download_f32(&cat_y2, &mut r_cat2).expect("dl cat2");
    for i in 0..m2 {
        assert!(
            (r_sep2[i] - r_cat2[i]).abs() < 1e-4,
            "concat3 y2 mismatch at {i}: {} vs {}",
            r_sep2[i],
            r_cat2[i]
        );
    }

    let mut r_sep3 = vec![0.0f32; m3];
    let mut r_cat3 = vec![0.0f32; m3];
    ctx.download_f32(&sep_y3, &mut r_sep3).expect("dl sep3");
    ctx.download_f32(&cat_y3, &mut r_cat3).expect("dl cat3");
    for i in 0..m3 {
        assert!(
            (r_sep3[i] - r_cat3[i]).abs() < 1e-4,
            "concat3 y3 mismatch at {i}: {} vs {}",
            r_sep3[i],
            r_cat3[i]
        );
    }

    // SwiGLU: separate gate, up, silu_mul vs fused gemv_q4_0_swiglu
    let (gate_raw, _) = gen_q4_0(m1, 2);
    let (up_raw, _) = gen_q4_0(m1, 7);
    let gate_buf = ctx.upload_bytes(&gate_raw).expect("upload gate");
    let up_buf = ctx.upload_bytes(&up_raw).expect("upload up");

    let mut sep_gate = ctx.create_buffer(m1 * 4).expect("alloc sep_gate");
    let mut sep_up = ctx.create_buffer(m1 * 4).expect("alloc sep_up");
    ctx.gemv_q4_0(&mut sep_gate, &gate_buf, &x_buf, m1 as u32, k as u32)
        .expect("gemv gate");
    ctx.gemv_q4_0(&mut sep_up, &up_buf, &x_buf, m1 as u32, k as u32)
        .expect("gemv up");
    ctx.silu_mul_inplace(&mut sep_gate, &sep_up, m1 as u32)
        .expect("silu_mul");

    let mut fused_swiglu = ctx.create_buffer(m1 * 4).expect("alloc fused_swiglu");
    ctx.gemv_q4_0_swiglu(
        &mut fused_swiglu,
        &gate_buf,
        &up_buf,
        &x_buf,
        m1 as u32,
        k as u32,
    )
    .expect("gemv_q4_0_swiglu");
    ctx.synchronize().expect("sync");

    let mut r_sep_swiglu = vec![0.0f32; m1];
    let mut r_fused_swiglu = vec![0.0f32; m1];
    ctx.download_f32(&sep_gate, &mut r_sep_swiglu)
        .expect("dl sep swiglu");
    ctx.download_f32(&fused_swiglu, &mut r_fused_swiglu)
        .expect("dl fused swiglu");
    for i in 0..m1 {
        assert!(
            (r_sep_swiglu[i] - r_fused_swiglu[i]).abs() < 1e-4,
            "swiglu mismatch at {i}: {} vs {}",
            r_sep_swiglu[i],
            r_fused_swiglu[i]
        );
    }
}

#[test]
fn test_cuda_q8_0_concat3_and_swiglu_parity() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA driver not available on this host; skipping live Q8_0 concat3/swiglu test");
        return;
    }

    let ctx = CudaContext::new(0).expect("initialize CUDA context");

    let m1: usize = 4;
    let m2: usize = 2;
    let m3: usize = 2;
    let k: usize = 64;
    let nb = k / 32;

    let gen_q8_0 = |m: usize, offset: usize| -> Vec<u8> {
        let mut raw = Vec::new();
        for r in 0..m {
            for b in 0..nb {
                let d_val = 0.5f32 + ((r + offset) as f32) * 0.02;
                let d_fp16: u16 = half::f16::from_f32(d_val).to_bits();
                raw.extend_from_slice(&d_fp16.to_le_bytes());

                for i in 0..32 {
                    let q = (((r * 32 + b * 32 + i + offset) % 11) as i8) - 5;
                    raw.push(q as u8);
                }
            }
        }
        raw
    };

    let q1_raw = gen_q8_0(m1, 1);
    let q2_raw = gen_q8_0(m2, 4);
    let q3_raw = gen_q8_0(m3, 8);

    let b1 = ctx.upload_bytes(&q1_raw).expect("upload b1");
    let b2 = ctx.upload_bytes(&q2_raw).expect("upload b2");
    let b3 = ctx.upload_bytes(&q3_raw).expect("upload b3");

    let x_vec: Vec<f32> = (0..k).map(|i| (i as f32) * 0.03 - 0.5).collect();
    let x_buf = ctx.upload_f32(&x_vec).expect("upload x");

    let mut sep_y1 = ctx.create_buffer(m1 * 4).expect("alloc sep_y1");
    let mut sep_y2 = ctx.create_buffer(m2 * 4).expect("alloc sep_y2");
    let mut sep_y3 = ctx.create_buffer(m3 * 4).expect("alloc sep_y3");

    ctx.gemv_q8_0(&mut sep_y1, &b1, &x_buf, m1 as u32, k as u32)
        .expect("gemv 1");
    ctx.gemv_q8_0(&mut sep_y2, &b2, &x_buf, m2 as u32, k as u32)
        .expect("gemv 2");
    ctx.gemv_q8_0(&mut sep_y3, &b3, &x_buf, m3 as u32, k as u32)
        .expect("gemv 3");

    let mut cat_y1 = ctx.create_buffer(m1 * 4).expect("alloc cat_y1");
    let mut cat_y2 = ctx.create_buffer(m2 * 4).expect("alloc cat_y2");
    let mut cat_y3 = ctx.create_buffer(m3 * 4).expect("alloc cat_y3");

    ctx.gemv_q8_0_concat3(
        &mut cat_y1,
        &mut cat_y2,
        &mut cat_y3,
        &b1,
        &b2,
        &b3,
        &x_buf,
        m1 as u32,
        m2 as u32,
        m3 as u32,
        k as u32,
    )
    .expect("gemv_q8_0_concat3");
    ctx.synchronize().expect("sync");

    let mut r_sep1 = vec![0.0f32; m1];
    let mut r_cat1 = vec![0.0f32; m1];
    ctx.download_f32(&sep_y1, &mut r_sep1).expect("dl sep1");
    ctx.download_f32(&cat_y1, &mut r_cat1).expect("dl cat1");
    for i in 0..m1 {
        assert!(
            (r_sep1[i] - r_cat1[i]).abs() < 1e-4,
            "Q8 concat3 y1 mismatch at {i}: {} vs {}",
            r_sep1[i],
            r_cat1[i]
        );
    }

    let mut r_sep2 = vec![0.0f32; m2];
    let mut r_cat2 = vec![0.0f32; m2];
    ctx.download_f32(&sep_y2, &mut r_sep2).expect("dl sep2");
    ctx.download_f32(&cat_y2, &mut r_cat2).expect("dl cat2");
    for i in 0..m2 {
        assert!(
            (r_sep2[i] - r_cat2[i]).abs() < 1e-4,
            "Q8 concat3 y2 mismatch at {i}: {} vs {}",
            r_sep2[i],
            r_cat2[i]
        );
    }

    let mut r_sep3 = vec![0.0f32; m3];
    let mut r_cat3 = vec![0.0f32; m3];
    ctx.download_f32(&sep_y3, &mut r_sep3).expect("dl sep3");
    ctx.download_f32(&cat_y3, &mut r_cat3).expect("dl cat3");
    for i in 0..m3 {
        assert!(
            (r_sep3[i] - r_cat3[i]).abs() < 1e-4,
            "Q8 concat3 y3 mismatch at {i}: {} vs {}",
            r_sep3[i],
            r_cat3[i]
        );
    }

    // SwiGLU
    let gate_raw = gen_q8_0(m1, 2);
    let up_raw = gen_q8_0(m1, 6);
    let gate_buf = ctx.upload_bytes(&gate_raw).expect("upload gate");
    let up_buf = ctx.upload_bytes(&up_raw).expect("upload up");

    let mut sep_gate = ctx.create_buffer(m1 * 4).expect("alloc sep_gate");
    let mut sep_up = ctx.create_buffer(m1 * 4).expect("alloc sep_up");
    ctx.gemv_q8_0(&mut sep_gate, &gate_buf, &x_buf, m1 as u32, k as u32)
        .expect("gemv gate");
    ctx.gemv_q8_0(&mut sep_up, &up_buf, &x_buf, m1 as u32, k as u32)
        .expect("gemv up");
    ctx.silu_mul_inplace(&mut sep_gate, &sep_up, m1 as u32)
        .expect("silu_mul");

    let mut fused_swiglu = ctx.create_buffer(m1 * 4).expect("alloc fused_swiglu");
    ctx.gemv_q8_0_swiglu(
        &mut fused_swiglu,
        &gate_buf,
        &up_buf,
        &x_buf,
        m1 as u32,
        k as u32,
    )
    .expect("gemv_q8_0_swiglu");
    ctx.synchronize().expect("sync");

    let mut r_sep_swiglu = vec![0.0f32; m1];
    let mut r_fused_swiglu = vec![0.0f32; m1];
    ctx.download_f32(&sep_gate, &mut r_sep_swiglu)
        .expect("dl sep swiglu");
    ctx.download_f32(&fused_swiglu, &mut r_fused_swiglu)
        .expect("dl fused swiglu");
    for i in 0..m1 {
        assert!(
            (r_sep_swiglu[i] - r_fused_swiglu[i]).abs() < 1e-4,
            "Q8 swiglu mismatch at {i}: {} vs {}",
            r_sep_swiglu[i],
            r_fused_swiglu[i]
        );
    }
}

#[test]
fn test_cuda_q4k_gemv_gemm_swiglu_and_gather_parity() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA driver not available on this host; skipping live Q4_K parity test");
        return;
    }

    use cera::quant::{BlockQ4KM, dequantize_q4_k_m_block};

    let ctx = CudaContext::new(0).expect("initialize CUDA context");

    let m: usize = 4;
    let k: usize = 256;
    let nb = k / 256;

    let mut raw_bytes = Vec::with_capacity(m * nb * 144);
    let mut ref_weights = vec![0.0f32; m * k];

    for r in 0..m {
        for b in 0..nb {
            let mut blk = BlockQ4KM {
                d: half::f16::from_f32(0.025 + (r as f32) * 0.005).to_bits(),
                dmin: half::f16::from_f32(0.01 + (b as f32) * 0.003).to_bits(),
                scales: [0u8; 12],
                qs: [0u8; 128],
            };
            for (i, v) in blk.scales.iter_mut().enumerate() {
                *v = ((r * 5 + b * 7 + i * 3) & 0xFF) as u8;
            }
            for (i, v) in blk.qs.iter_mut().enumerate() {
                *v = ((r * 37 + b * 13 + i) & 0xFF) as u8;
            }
            let dq = dequantize_q4_k_m_block(&blk);
            let row_off = r * k + b * 256;
            ref_weights[row_off..row_off + 256].copy_from_slice(&dq);
            raw_bytes.extend_from_slice(&blk.d.to_le_bytes());
            raw_bytes.extend_from_slice(&blk.dmin.to_le_bytes());
            raw_bytes.extend_from_slice(&blk.scales);
            raw_bytes.extend_from_slice(&blk.qs);
        }
    }

    let weight_buf = ctx.upload_bytes(&raw_bytes).expect("upload Q4K weights");

    // 1. GEMV: test against CPU reference dot product
    let x_vec: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.2).collect();
    let x_buf = ctx.upload_f32(&x_vec).expect("upload x");
    let mut out_gemv = ctx.create_buffer(m * 4).expect("alloc out_gemv");

    ctx.gemv_q4k(&mut out_gemv, &weight_buf, &x_buf, m as u32, k as u32)
        .expect("gemv_q4k");
    ctx.synchronize().expect("sync");

    let mut gemv_result = vec![0.0f32; m];
    ctx.download_f32(&out_gemv, &mut gemv_result)
        .expect("download gemv");

    for r in 0..m {
        let expected: f32 = (0..k).map(|c| ref_weights[r * k + c] * x_vec[c]).sum();
        assert!(
            (gemv_result[r] - expected).abs() < 1e-3,
            "Q4K GEMV mismatch at row {r}: got {}, expected {}",
            gemv_result[r],
            expected
        );
    }

    // 2. Batched GEMM: batch of 2 tokens
    let batch_m: usize = 2;
    let mut x_batch = x_vec.clone();
    x_batch.extend((0..k).map(|i| (i as f32) * -0.015 + 0.3));
    let x_batch_buf = ctx.upload_f32(&x_batch).expect("upload x_batch");
    let mut out_gemm = ctx.create_buffer(batch_m * m * 4).expect("alloc out_gemm");

    ctx.gemm_q4k(
        &mut out_gemm,
        &weight_buf,
        &x_batch_buf,
        batch_m as u32,
        m as u32,
        k as u32,
    )
    .expect("gemm_q4k");
    ctx.synchronize().expect("sync");

    let mut gemm_result = vec![0.0f32; batch_m * m];
    ctx.download_f32(&out_gemm, &mut gemm_result)
        .expect("download gemm");

    for b in 0..batch_m {
        for r in 0..m {
            let expected: f32 = (0..k)
                .map(|c| ref_weights[r * k + c] * x_batch[b * k + c])
                .sum();
            let actual = gemm_result[b * m + r];
            assert!(
                (actual - expected).abs() < 1e-3,
                "Q4K GEMM mismatch at batch {b}, row {r}: got {actual}, expected {expected}"
            );
        }
    }

    // 3. Embedding gather: row 1
    let mut out_gather = ctx.create_buffer(k * 4).expect("alloc out_gather");
    ctx.gather_embedding_q4k(&mut out_gather, &weight_buf, 1, k as u32)
        .expect("gather_embedding_q4k");
    ctx.synchronize().expect("sync");

    let mut gather_result = vec![0.0f32; k];
    ctx.download_f32(&out_gather, &mut gather_result)
        .expect("download gather");
    for c in 0..k {
        let expected = ref_weights[k + c];
        assert!(
            (gather_result[c] - expected).abs() < 1e-4,
            "Q4K gather mismatch at col {c}: got {}, expected {}",
            gather_result[c],
            expected
        );
    }
}

#[test]
fn test_cuda_rope_layout_parity() {
    if !CudaDevice::is_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let n_heads = 2;
    let n_kv_heads = 1;
    let head_dim = 64;
    let pos = 5;
    let freq_base = 10000.0f32;

    // Test both 0 (NeoX) and 1 (Norm / interleaved)
    for rope_type in [0u32, 1u32] {
        let mut q_host: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.05).collect();
        let mut k_host: Vec<f32> = (0..n_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.03)
            .collect();

        // Compute CPU reference
        let mut q_ref = q_host.clone();
        let mut k_ref = k_host.clone();
        let half_dim = head_dim / 2;
        let theta_scale = freq_base.powf(-2.0 / head_dim as f32);

        // Reference RoPE application
        for h in 0..n_heads {
            let offset = h * head_dim;
            for d in 0..half_dim {
                let theta = (pos as f32) * theta_scale.powf(d as f32);
                let (sin_a, cos_a) = theta.sin_cos();
                if rope_type == 0 {
                    let x0 = q_ref[offset + d];
                    let x1 = q_ref[offset + d + half_dim];
                    q_ref[offset + d] = x0 * cos_a - x1 * sin_a;
                    q_ref[offset + d + half_dim] = x0 * sin_a + x1 * cos_a;
                } else {
                    let x0 = q_ref[offset + 2 * d];
                    let x1 = q_ref[offset + 2 * d + 1];
                    q_ref[offset + 2 * d] = x0 * cos_a - x1 * sin_a;
                    q_ref[offset + 2 * d + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }

        for h in 0..n_kv_heads {
            let offset = h * head_dim;
            for d in 0..half_dim {
                let theta = (pos as f32) * theta_scale.powf(d as f32);
                let (sin_a, cos_a) = theta.sin_cos();
                if rope_type == 0 {
                    let x0 = k_ref[offset + d];
                    let x1 = k_ref[offset + d + half_dim];
                    k_ref[offset + d] = x0 * cos_a - x1 * sin_a;
                    k_ref[offset + d + half_dim] = x0 * sin_a + x1 * cos_a;
                } else {
                    let x0 = k_ref[offset + 2 * d];
                    let x1 = k_ref[offset + 2 * d + 1];
                    k_ref[offset + 2 * d] = x0 * cos_a - x1 * sin_a;
                    k_ref[offset + 2 * d + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }

        let mut q_buf = ctx.upload_f32(&q_host).expect("upload q");
        let mut k_buf = ctx.upload_f32(&k_host).expect("upload k");

        let params = QkNormRopeParams {
            pos: pos as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            eps: 1e-5,
            freq_base,
            rope_type,
            has_freq_factors: 0,
            has_qk_norm: 0,
        };

        ctx.qk_norm_rope(&mut q_buf, &mut k_buf, None, None, None, params)
            .expect("qk_norm_rope kernel");
        ctx.synchronize().expect("sync");

        ctx.download_f32(&q_buf, &mut q_host).expect("download q");
        ctx.download_f32(&k_buf, &mut k_host).expect("download k");

        for i in 0..q_host.len() {
            assert!(
                (q_host[i] - q_ref[i]).abs() < 1e-4,
                "rope_type {rope_type} q mismatch at {i}: cuda {} vs ref {}",
                q_host[i],
                q_ref[i]
            );
        }
        for i in 0..k_host.len() {
            assert!(
                (k_host[i] - k_ref[i]).abs() < 1e-4,
                "rope_type {rope_type} k mismatch at {i}: cuda {} vs ref {}",
                k_host[i],
                k_ref[i]
            );
        }
    }
}

#[test]
fn test_cuda_zero_dimension_guards() {
    if !CudaDevice::is_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("failed to initialize CUDA context");
    let mut buf = ctx.create_buffer(64).expect("alloc buffer");
    let weights = ctx.create_buffer(64).expect("alloc weights");
    let x = ctx.create_buffer(64).expect("alloc x");

    // Zero m or k should return Ok(()) cleanly without driver launch errors
    assert!(ctx.gemv_q4_0(&mut buf, &weights, &x, 0, 64).is_ok());
    assert!(ctx.gemv_q4_0(&mut buf, &weights, &x, 64, 0).is_ok());
    assert!(ctx.gemv_q8_0(&mut buf, &weights, &x, 0, 64).is_ok());
    assert!(ctx.gemv_q4k(&mut buf, &weights, &x, 0, 64).is_ok());

    // Zero GEMM dimensions
    assert!(ctx.gemm_q4_0(&mut buf, &weights, &x, 0, 16, 64).is_ok());
    assert!(ctx.gemm_q8_0(&mut buf, &weights, &x, 16, 0, 64).is_ok());
    assert!(ctx.gemm_q4k(&mut buf, &weights, &x, 16, 16, 0).is_ok());

    // Zero elementwise and rmsnorm
    assert!(ctx.rmsnorm(&mut buf, &x, &weights, 0, 1e-5).is_ok());
    assert!(ctx.silu_mul_inplace(&mut buf, &x, 0).is_ok());
    assert!(ctx.add_inplace(&mut buf, &x, 0).is_ok());
    assert!(ctx.scale_inplace(&mut buf, 1.0, 0).is_ok());
    assert!(ctx.softmax(&mut buf, 0).is_ok());
    assert!(ctx.argmax_f32(&mut buf, &x, 0).is_ok());
}

//! Minimal CPU GEMV probe: kernel throughput in isolation.
//!
//! Times `gemv_q4_0_with_q8` on an LM-head-shaped matrix (65536x1024,
//! 37.8 MB, larger than any mobile L3, so every iteration streams from
//! DRAM exactly like decode). Compares against the GB/s implied by
//! end-to-end decode to split "slow kernel" from "slow framework".
//!
//! Run on device, single-threaded:
//!   CERA_THREADS=1 taskset 80 ./simd_gemv_bench

#[cfg(target_arch = "aarch64")]
mod bench {
    use std::time::Instant;

    use cera::backend::cpu::{
        gemm_preq_dispatch, gemv_q4_0_gate_up_swiglu_with_q8, gemv_q4_0_with_q8,
        quantize_f32_to_q8_0, silu_mul_inplace,
    };
    use cera::quant::BlockQ4_0;
    use cera::tensor::DType;

    static mut RNG_STATE: u64 = 0x12345678;

    fn next_u8() -> u8 {
        unsafe {
            RNG_STATE = RNG_STATE.wrapping_mul(6364136223846793005).wrapping_add(1);
            (RNG_STATE >> 33) as u8
        }
    }

    fn random_q4_0(m: usize, k: usize) -> Vec<u8> {
        let mut aquant = vec![0u8; m * (k / 32) * size_of::<BlockQ4_0>()];
        for blk in aquant.chunks_mut(size_of::<BlockQ4_0>()) {
            let b = unsafe { &mut *(blk.as_mut_ptr() as *mut BlockQ4_0) };
            b.d = 0x3C00; // f16 1.0
            for q in b.qs.iter_mut() {
                *q = next_u8();
            }
        }
        aquant
    }

    fn quantize_input(k: usize) -> (Vec<f32>, Vec<i8>) {
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        quantize_f32_to_q8_0(&x)
    }

    /// Run `f` `warmup` times untimed, then `iters` times timed; return the
    /// median seconds. One scaffold for every probe below, so a statistics
    /// fix lands once instead of once per bench.
    fn time_median(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        let mut dts = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            f();
            dts.push(t.elapsed().as_secs_f64());
        }
        dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dts[iters / 2]
    }

    fn bench_plain(m: usize, k: usize) {
        let aquant = random_q4_0(m, k);
        let (xs, xq) = quantize_input(k);
        let mut y = vec![0.0f32; m];
        let med = time_median(3, 20, || {
            gemv_q4_0_with_q8(&aquant, &xs, &xq, &mut y, m, k);
        });
        let chk: f64 = y.iter().step_by(1024).map(|v| *v as f64).sum();
        println!(
            "plain m={m} k={k}: median {:.6}s, {:.1} GB/s (chk {chk:.3})",
            med,
            aquant.len() as f64 / med / 1e9
        );
    }

    fn bench_fused(m: usize, k: usize) {
        let gate = random_q4_0(m, k);
        let up = random_q4_0(m, k);
        let (xs, xq) = quantize_input(k);
        let mut y = vec![0.0f32; m];
        let med = time_median(3, 20, || {
            gemv_q4_0_gate_up_swiglu_with_q8(&gate, &up, &xs, &xq, &mut y, m, k);
        });
        let chk: f64 = y.iter().step_by(1024).map(|v| *v as f64).sum();
        println!(
            "fused m={m} k={k}: median {:.6}s, {:.1} GB/s (chk {chk:.3})",
            med,
            (gate.len() + up.len()) as f64 / med / 1e9
        );
    }

    fn bench_fused_same_ptr(m: usize, k: usize) {
        // Same matrix for gate AND up: if the fused loop is intrinsically slow
        // this stays slow; if the two-region ping-pong is the problem this flies.
        let gate = random_q4_0(m, k);
        let (xs, xq) = quantize_input(k);
        let mut y = vec![0.0f32; m];
        let med = time_median(3, 20, || {
            gemv_q4_0_gate_up_swiglu_with_q8(&gate, &gate, &xs, &xq, &mut y, m, k);
        });
        println!(
            "fused-sameptr m={m} k={k}: median {:.6}s, {:.1} GB/s",
            med,
            (2 * gate.len()) as f64 / med / 1e9
        );
    }

    fn bench_unfused(m: usize, k: usize) {
        // The candidate fix measured directly: two plain GEMVs + SiLU pass.
        let gate = random_q4_0(m, k);
        let up = random_q4_0(m, k);
        let (xs, xq) = quantize_input(k);
        let mut yg = vec![0.0f32; m];
        let mut yu = vec![0.0f32; m];
        let med = time_median(3, 20, || {
            gemv_q4_0_with_q8(&gate, &xs, &xq, &mut yg, m, k);
            gemv_q4_0_with_q8(&up, &xs, &xq, &mut yu, m, k);
            silu_mul_inplace(&mut yg, &yu);
        });
        println!(
            "unfused m={m} k={k}: median {:.6}s, {:.1} GB/s",
            med,
            (gate.len() + up.len()) as f64 / med / 1e9
        );
    }

    fn bench_gemm(m: usize, k: usize, n: usize) {
        let aquant = random_q4_0(m, k);
        // B packed column-major to match `quantize_columns`: column j occupies
        // quants[j*k..(j+1)*k], scales[j*nb..(j+1)*nb].
        let nb = k / 32;
        let mut b_quants = vec![0i8; n * k];
        let mut b_scales = vec![0f32; n * nb];
        for j in 0..n {
            let col: Vec<f32> = (0..k)
                .map(|i| (((i * 2654435761 + j * 40503) % 1000) as f32) / 1000.0 - 0.5)
                .collect();
            let (s, q) = quantize_f32_to_q8_0(&col);
            b_scales[j * nb..(j + 1) * nb].copy_from_slice(&s);
            b_quants[j * k..(j + 1) * k].copy_from_slice(&q);
        }
        let mut out = vec![0.0f32; m * n];
        let med = time_median(2, 10, || {
            let ran = gemm_preq_dispatch(
                DType::Q4_0,
                &aquant,
                &b_scales,
                &b_quants,
                &mut out,
                m,
                n,
                k,
            );
            assert!(ran, "no GEMM kernel ran for Q4_0");
        });
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let chk: f64 = out.iter().step_by(100003).map(|v| *v as f64).sum();
        println!(
            "gemm m={m} k={k} n={n}: median {:.6}s, {:.1} GFLOPS (chk {chk:.3})",
            med,
            flops / med / 1e9
        );
    }

    pub(crate) fn run() {
        // LM-head shape (DRAM-realistic) + FFN shapes (may sit in L3; isolates
        // shape dependence, not absolute bandwidth).
        bench_plain(65536, 1024);
        bench_plain(4608, 1024);
        bench_plain(1024, 4608);
        bench_fused(4608, 1024);
        bench_fused_same_ptr(4608, 1024);
        bench_unfused(4608, 1024);
        // FFN-gate prefill shape: pool-scaling isolation (no serial gaps).
        bench_gemm(4608, 1024, 512);
    }
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    bench::run();
    #[cfg(not(target_arch = "aarch64"))]
    eprintln!("simd_gemv_bench requires an aarch64 target");
}

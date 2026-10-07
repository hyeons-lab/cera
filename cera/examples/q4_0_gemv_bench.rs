//! The Q4_0 decode GEMV on the LFM2-450M's real shapes: the standard layout against the decode repack.
//!
//! Random Q4_0 weights, timed through `gemv_with_preq` (standard layout) and `gemv_q4_0_dec4_with_q8`
//! (4-row interleave), as ms and GB/s of weight bytes streamed. Run under `simpleperf stat -e
//! cpu-cycles:u,instructions:u` for the instruction counts, which do not depend on the phone's clock state.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example q4_0_gemv_bench
//! adb push target/aarch64-linux-android/release/examples/q4_0_gemv_bench /data/local/tmp/cmp-cera/
//! adb shell 'cd /data/local/tmp/cmp-cera && ./q4_0_gemv_bench [iters]'
//! ```

#[cfg(all(target_arch = "aarch64", not(has_blas)))]
fn main() {
    use std::time::Instant;

    use cera::backend::cpu::{
        gemv_q4_0_dec4_with_q8, gemv_with_preq, q4_0_dec4_supported, quantize_f32_to_q8_0,
        repack_q4_0_dec4,
    };
    use cera::tensor::DType;

    let iters: usize = std::env::args()
        .nth(1)
        .map_or(400, |v| v.parse().expect("iters"));
    // (name, rows, k) of the biggest decode GEMVs: FFN gate/up, FFN down, conv in_proj, out_proj.
    let shapes = [
        ("ffn_gate/up", 4608usize, 1024usize),
        ("ffn_down", 1024, 4608),
        ("conv in_proj", 3072, 1024),
        ("out_proj", 1024, 1024),
    ];
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    if !q4_0_dec4_supported(16, 32) {
        eprintln!("this host has no dot-product + fp16 support; nothing to compare");
        return;
    }
    for (name, m, k) in shapes {
        let nb = k / 32;
        let mut w = vec![0u8; m * nb * 18];
        for blk in w.chunks_mut(18) {
            blk[..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
            for b in &mut blk[2..] {
                *b = next() as u8;
            }
        }
        let packed = repack_q4_0_dec4(&w, m, k);
        let x: Vec<f32> = (0..k)
            .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let (xs, xq) = quantize_f32_to_q8_0(&x);
        let mut y = vec![0.0f32; m];
        let gb = (m * nb * 18) as f64 / 1e9;
        let mut run = |label: &str, f: &mut dyn FnMut(&mut [f32])| {
            for _ in 0..50 {
                f(&mut y);
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                f(&mut y);
                samples.push(t.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = samples[iters / 2];
            println!(
                "{name:13} {label:9} p50 {p50:7.3} ms ({:5.1} GB/s)  p10 {:7.3}  p90 {:7.3}",
                gb / (p50 / 1e3),
                samples[iters / 10],
                samples[iters * 9 / 10]
            );
        };
        run("standard", &mut |y| {
            gemv_with_preq(DType::Q4_0, &w, &xs, &xq, &x, y, m, k)
        });
        run("dec4", &mut |y| {
            gemv_q4_0_dec4_with_q8(&packed, &xs, &xq, y, m, k)
        });
    }
}

#[cfg(not(all(target_arch = "aarch64", not(has_blas))))]
fn main() {
    eprintln!("aarch64 only");
}

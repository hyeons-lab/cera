//! The Q6_K lm_head GEMV on its own: how many GB/s the decode step's output projection streams.
//!
//! Random Q6_K weights at the real shape (65536 x 1024 by default), timed through the fused-argmax
//! path (greedy decode) and the full-logits path (sampling). Compare against the Q4_0 feed-forward,
//! which streams at about 50 GB/s on the same phone.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example q6k_gemv_bench
//! adb push target/aarch64-linux-android/release/examples/q6k_gemv_bench /data/local/tmp/cmp-cera/
//! adb shell 'cd /data/local/tmp/cmp-cera && ./q6k_gemv_bench [rows] [k] [iters]'
//! ```

use std::time::Instant;

use cera::backend::cpu::{gemv_with_preq, gemv_with_preq_argmax, quantize_f32_to_q8_0_into};
use cera::tensor::DType;

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().map_or(65536, |v| v.parse().expect("rows"));
    let k: usize = args.next().map_or(1024, |v| v.parse().expect("k"));
    let iters: usize = args.next().map_or(300, |v| v.parse().expect("iters"));
    assert_eq!(k % 256, 0);
    let row_bytes = k / 256 * 210;
    let mut st = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    // Block layout: ql[128] qh[64] scales[16] d(f16): random bytes, a small finite d, and scales
    // kept in range so the sums stay finite.
    let mut w = vec![0u8; rows * row_bytes];
    for blk in w.chunks_mut(210) {
        for b in &mut blk[..192] {
            *b = next() as u8;
        }
        for b in &mut blk[192..208] {
            *b = (next() % 40) as u8;
        }
        blk[208..210].copy_from_slice(&half::f16::from_f32(0.003).to_bits().to_le_bytes());
    }
    let x: Vec<f32> = (0..k)
        .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let mut xs = vec![0.0f32; x.len() / 32];
    let mut xq = vec![0i8; x.len()];
    quantize_f32_to_q8_0_into(&x, &mut xs, &mut xq);

    let mut logits = vec![0.0f32; rows];
    let gb = (rows * row_bytes) as f64 / 1e9;
    let time = |label: &str, f: &mut dyn FnMut() -> usize| {
        for _ in 0..30 {
            std::hint::black_box(f());
        }
        let mut samples = Vec::with_capacity(iters);
        let mut last = 0;
        for _ in 0..iters {
            let t = Instant::now();
            last = std::hint::black_box(f());
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (p10, p50, p90) = (
            samples[iters / 10],
            samples[iters / 2],
            samples[iters * 9 / 10],
        );
        println!(
            "{label:8} p50 {p50:.3} ms ({:.1} GB/s)  p10 {p10:.3}  p90 {p90:.3}  -> {last}",
            gb / (p50 / 1e3)
        );
    };
    println!("{rows} x {k} Q6_K = {:.1} MB", gb * 1e3);
    time("argmax", &mut || {
        gemv_with_preq_argmax(DType::Q6K, &w, &xs, &xq, &x, rows, k)
    });
    time("logits", &mut || {
        gemv_with_preq(DType::Q6K, &w, &xs, &xq, &x, &mut logits, rows, k);
        cera::sampler::argmax(&logits) as usize
    });
}

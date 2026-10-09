//! What one fork-join on the decode `RowPool` costs: the barrier tax every decode GEMV pays.
//!
//! A decode token issues on the order of a hundred dispatches, so a few microseconds each is a
//! visible slice of a 4.5 ms token. Times an empty dispatch back to back, then one with a little
//! serial work between dispatches (the caller's norm/residual/conv between matmuls).
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example pool_dispatch_bench
//! adb push target/aarch64-linux-android/release/examples/pool_dispatch_bench /data/local/tmp/cmp-cera/
//! adb shell 'cd /data/local/tmp/cmp-cera && ./pool_dispatch_bench'
//! ```

use std::hint::black_box;
use std::time::Instant;

use cera::backend::cpu::{decode_par_threads, par_range};

fn serial_work(ns: u32) {
    // A dependent chain the optimizer cannot fold: about one ns per iteration on this class of core.
    let mut x = black_box(1u64);
    for _ in 0..ns {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    black_box(x);
}

fn main() {
    println!("decode pool: {} threads", decode_par_threads());
    let n = 200_000;
    for (label, gap_ns, rows_work) in [
        ("empty dispatch", 0u32, 0u32),
        ("1us serial gap", 300, 0),
        ("1us work per chunk", 0, 300),
        ("5us serial gap", 1500, 0),
    ] {
        for _ in 0..2000 {
            par_range(64, 1, |s, c| {
                black_box((s, c));
            });
        }
        let t = Instant::now();
        for _ in 0..n {
            par_range(64, 1, |s, c| {
                if rows_work > 0 {
                    serial_work(rows_work);
                }
                black_box((s, c));
            });
            if gap_ns > 0 {
                serial_work(gap_ns);
            }
        }
        let per = t.elapsed().as_secs_f64() * 1e9 / n as f64;
        let mut t0 = 0.0;
        if gap_ns > 0 {
            let t = Instant::now();
            for _ in 0..n {
                serial_work(gap_ns);
            }
            t0 = t.elapsed().as_secs_f64() * 1e9 / n as f64;
        }
        println!(
            "{label:20} {per:8.0} ns per iteration (serial gap alone {t0:.0} ns, dispatch cost ~{:.0} ns)",
            per - t0
        );
    }
}

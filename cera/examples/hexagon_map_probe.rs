//! Probe how the CDSP address space behaves under map / unmap, on a device.
//!
//! The NPU backend keeps every weight in one mapped `rpcmem` buffer, and the
//! CDSP maps only about 3 to 4 GiB in total, so a larger model cannot load.
//! Paging weights through the mapping works only if three things hold, and
//! this measures each of them:
//!
//! 1. unmapping a buffer gives its address space back (so a sliding window of
//!    mappings can cover more than the ceiling over time);
//! 2. a sliding window over buffers that all stay allocated (resident in RAM)
//!    keeps working, with no copy per page-in;
//! 3. what a map / unmap costs per buffer size, which sets how often a model
//!    can afford to swap.
//!
//! Run on an Android device with the DSP skels staged:
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_map_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_map_probe /data/local/tmp/
//! adb shell 'echo 0 > /proc/$$/oom_score_adj; /data/local/tmp/hexagon_map_probe'
//! ```
//!
//! The `oom_score_adj` write matters: adb-shell children are unkillable by
//! default, so a probe that exhausts memory wedges the phone instead of dying.

#[cfg(feature = "hexagon")]
fn main() {
    use std::time::Instant;

    use cera::backend::hexagon::{FastRpcDriver, RpcmemBuffer, probe_device};

    const MIB: usize = 1 << 20;

    let driver = FastRpcDriver::load().expect("load the FastRPC driver");
    // A mapping is only meaningful once a DSP session exists.
    let _device = probe_device(&driver, None).expect("open a DSP session");

    let alloc = |size: usize| RpcmemBuffer::alloc(driver.clone(), size, false).expect("rpcmem");
    let map = |b: &RpcmemBuffer| driver.fastrpc_mmap(b.fd(), b.as_mut_ptr(), b.size());
    let unmap = |b: &RpcmemBuffer| driver.fastrpc_munmap(b.fd(), b.as_mut_ptr(), b.size());

    println!("== 1. map until refused, then release (chunk 256 MiB)");
    let mut held = Vec::new();
    let mut mapped_mib = 0usize;
    loop {
        let b = alloc(256 * MIB);
        match map(&b) {
            Ok(()) => {
                mapped_mib += 256;
                held.push(b);
            }
            Err(e) => {
                println!("refused at {mapped_mib} MiB mapped: {e}");
                break;
            }
        }
        if mapped_mib >= 6144 {
            println!("still mapping at {mapped_mib} MiB; stopping the probe here");
            break;
        }
    }
    let ceiling = mapped_mib;
    for b in &held {
        unmap(b).expect("unmap");
    }
    println!("released {} buffers", held.len());
    drop(held);

    println!("== 2. does unmapping return the space? map+unmap 1 GiB eight times");
    for i in 0..8 {
        let b = alloc(1024 * MIB);
        match map(&b) {
            Ok(()) => {
                unmap(&b).expect("unmap");
                println!("cycle {i}: ok");
            }
            Err(e) => {
                println!("cycle {i}: refused: {e}");
                break;
            }
        }
    }

    println!("== 3. sliding window: 9 resident 512 MiB buffers, 4 mapped at a time");
    let pool: Vec<RpcmemBuffer> = (0..9).map(|_| alloc(512 * MIB)).collect();
    let mut window: std::collections::VecDeque<usize> = Default::default();
    let (mut map_ms, mut unmap_ms, mut maps) = (0.0, 0.0, 0);
    for step in 0..27 {
        let idx = step % pool.len();
        if window.len() == 4 {
            let old = window.pop_front().unwrap();
            let t = Instant::now();
            unmap(&pool[old]).expect("unmap");
            unmap_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        let t = Instant::now();
        match map(&pool[idx]) {
            Ok(()) => {
                map_ms += t.elapsed().as_secs_f64() * 1e3;
                maps += 1;
                window.push_back(idx);
            }
            Err(e) => {
                println!("step {step}: refused with {} mapped: {e}", window.len());
                break;
            }
        }
    }
    println!(
        "{maps} maps over a {}-MiB window: avg map {:.2} ms, avg unmap {:.2} ms per 512 MiB",
        4 * 512,
        map_ms / maps.max(1) as f64,
        unmap_ms / maps.max(1) as f64
    );
    for i in window {
        unmap(&pool[i]).expect("unmap");
    }

    println!("== 4. map/unmap latency by size");
    for size_mib in [16, 64, 256, 1024] {
        let b = alloc(size_mib * MIB);
        let t = Instant::now();
        let mapped = map(&b);
        let m = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        if mapped.is_ok() {
            unmap(&b).expect("unmap");
        }
        let u = t.elapsed().as_secs_f64() * 1e3;
        println!("{size_mib:>5} MiB: map {m:.2} ms, unmap {u:.2} ms ({mapped:?})");
    }

    println!("summary: ceiling with 256 MiB chunks ~ {ceiling} MiB");
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon");
}

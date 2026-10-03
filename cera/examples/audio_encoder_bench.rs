//! Time the LFM2-Audio Conformer encoder on its own: wall clock and CPU
//! seconds (user + system) per second of audio.
//!
//! Speech input is the one stage of an LFM2-Audio turn that still runs on the
//! CPU when the backbone is on the NPU. For background work on a phone what
//! matters is how much CPU it burns (heat, battery, contention with the app),
//! not only how long it takes, so this reports both. The input is synthetic
//! (the encoder's cost does not depend on the content).
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example audio_encoder_bench
//! adb push target/aarch64-linux-android/release/examples/audio_encoder_bench /data/local/tmp/
//! adb shell '/data/local/tmp/audio_encoder_bench mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf 10'
//! ```
//!
//! Arguments: the mmproj GGUF, then the clip length in seconds (default 10)
//! and the number of timed repetitions (default 5, after one warm-up).

use std::path::Path;
use std::time::Instant;

use cera::gguf::GgufFile;
use cera::model::audio_encoder::{AudioEncoderWeights, SAMPLE_RATE, encode_audio_pcm};

/// User plus system CPU time of this process, in seconds.
fn cpu_seconds() -> f64 {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` fills the struct it is given; RUSAGE_SELF is valid.
    let ru = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr());
        ru.assume_init()
    };
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    secs(ru.ru_utime) + secs(ru.ru_stime)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: audio_encoder_bench <mmproj.gguf> [seconds] [reps]");
    let seconds: f64 = args.next().map_or(10.0, |s| s.parse().expect("seconds"));
    let reps: usize = args.next().map_or(5, |s| s.parse().expect("reps"));

    // The CLI does this; an embedder that skips it gets a differently sized pool.
    cera::backend::cpu::configure_thread_pool();

    let gguf = GgufFile::open_arc(Path::new(&path)).expect("open mmproj");
    let weights = AudioEncoderWeights::from_gguf(&gguf).expect("audio encoder weights");
    let cfg = &weights.config;
    println!(
        "encoder: {} layers, n_embd {}, n_ff {}, {} heads, {} mel bins -> llm {}",
        cfg.n_layer, cfg.n_embd, cfg.n_ff, cfg.n_head, cfg.n_mel_bins, cfg.llm_hidden_size
    );

    // A few harmonics plus noise: a stand-in for speech.
    let n = (seconds * SAMPLE_RATE as f64) as usize;
    let mut seed = 0x2545F491u32;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let t = i as f32 / SAMPLE_RATE as f32;
            0.3 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 540.0 * t).sin()
                + 0.05 * noise
        })
        .collect();

    // Warm-up: page in the mmap'd weights and spin up the thread pools.
    let (_, frames) = encode_audio_pcm(&pcm, &weights);
    println!("{seconds:.1} s of audio -> {frames} frames");

    let (mut wall, mut cpu) = (0.0, 0.0);
    for _ in 0..reps {
        let (c0, t0) = (cpu_seconds(), Instant::now());
        let _ = encode_audio_pcm(&pcm, &weights);
        wall += t0.elapsed().as_secs_f64();
        cpu += cpu_seconds() - c0;
    }
    let (wall, cpu) = (wall / reps as f64, cpu / reps as f64);
    println!(
        "per encode: wall {:.0} ms ({:.3} s per audio second), cpu {:.0} ms ({:.3} cpu-s per audio second, {:.1} cores busy)",
        wall * 1e3,
        wall / seconds,
        cpu * 1e3,
        cpu / seconds,
        cpu / wall
    );
}

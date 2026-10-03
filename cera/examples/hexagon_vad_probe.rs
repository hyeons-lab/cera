//! Silero VAD on the Hexagon NPU against the CPU: the speech probability of every 32 ms window
//! of a clip, and what each costs the CPU.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_vad_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_vad_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && ./hexagon_vad_probe silero_vad.gguf clip.wav'
//! ```

#[cfg(feature = "hexagon")]
fn main() {
    use cera::vad::{SileroVad, VadSampleRate};

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: hexagon_vad_probe <silero_vad.gguf> <clip.wav>");
    let wav = std::fs::read(args.next().expect("a WAV path")).expect("read the WAV");
    let data = wav
        .windows(4)
        .position(|w| w == b"data")
        .expect("data chunk");
    let pcm: Vec<f32> = wav[data + 8..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect();
    let windows: Vec<&[f32]> = pcm.as_chunks::<512>().0.iter().map(|w| &w[..]).collect();

    let cpu_seconds = || {
        let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `getrusage` fills the struct; RUSAGE_SELF is valid.
        let ru = unsafe {
            libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr());
            ru.assume_init()
        };
        let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
        (secs(ru.ru_utime), secs(ru.ru_stime))
    };

    let mut cpu = SileroVad::from_file(&model).expect("load the VAD");
    let mut npu = SileroVad::from_file(&model).expect("load the VAD");
    assert!(npu.try_enable_hexagon(), "the NPU is not available");
    assert!(npu.is_accelerated());

    let run = |vad: &mut SileroVad| -> (Vec<f32>, f64, f64, f64) {
        vad.reset();
        let ((u0, s0), t0) = (cpu_seconds(), std::time::Instant::now());
        let probs: Vec<f32> = windows
            .iter()
            .map(|w| vad.process_chunk(w, VadSampleRate::Rate16kHz).unwrap())
            .collect();
        let (u1, s1) = cpu_seconds();
        (probs, t0.elapsed().as_secs_f64(), u1 - u0, s1 - s0)
    };

    let (p_cpu, wall_cpu, u_cpu, s_cpu) = run(&mut cpu);
    let _ = run(&mut npu); // warm-up: builds and uploads the batch
    let (p_npu, wall_npu, u_npu, s_npu) = run(&mut npu);
    assert!(npu.is_accelerated(), "the NPU dropped out during the run");

    let worst = p_cpu
        .iter()
        .zip(&p_npu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let flips = p_cpu
        .iter()
        .zip(&p_npu)
        .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
        .count();
    let n = windows.len() as f64;
    println!(
        "{} windows ({:.1} s): max |prob diff| {worst:.5}, decisions at 0.5 flipped: {flips}",
        windows.len(),
        n * 0.032
    );
    println!(
        "CPU VAD: wall {:.2} ms/window, cpu {:.0} us/window (user {:.0}, sys {:.0})",
        wall_cpu / n * 1e3,
        (u_cpu + s_cpu) / n * 1e6,
        u_cpu / n * 1e6,
        s_cpu / n * 1e6
    );
    println!(
        "NPU VAD: wall {:.2} ms/window, cpu {:.0} us/window (user {:.0}, sys {:.0})",
        wall_npu / n * 1e3,
        (u_npu + s_npu) / n * 1e6,
        u_npu / n * 1e6,
        s_npu / n * 1e6
    );
    println!(
        "per audio second: CPU {:.4} cpu-s, NPU {:.4} cpu-s",
        (u_cpu + s_cpu) / (n * 0.032),
        (u_npu + s_npu) / (n * 0.032)
    );
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon");
}

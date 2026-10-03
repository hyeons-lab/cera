//! The keyword spotter's backbone on the Hexagon NPU against the CPU: the keyword scores of every
//! detection window of a clip, and what each costs the CPU. A synthetic model (random weights,
//! the shipped shapes) stands in for a trained one when no GGUF is given: the arithmetic is what
//! is compared, and it does not depend on the weights being trained.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_hotword_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_hotword_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && ./hexagon_hotword_probe clip60.wav [hey_liquid.gguf]'
//! ```

#[cfg(feature = "hexagon")]
fn synthetic_model() -> Vec<u8> {
    use cera::convert::writer::{GGML_TYPE_F32, GgufWriter};
    let mut seed = 0x1234_5678u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
    };
    let mut w = GgufWriter::new();
    w.add_string("general.architecture", "kws");
    w.add_string("general.name", "synthetic KWS");
    w.add_u32("kws.keyword_count", 1);
    w.add_string_array("kws.keywords", vec!["Hey Synthetic".to_string()]);
    for (k, v) in [
        ("kws.sample_rate", 16_000),
        ("kws.window_samples", 19_200),
        ("kws.hop_samples", 1_280),
        ("kws.mel_bins", 32),
        ("kws.mel_window_samples", 400),
        ("kws.mel_hop_samples", 160),
        ("kws.fft_size", 512),
        ("kws.embedding_dim", 64),
    ] {
        w.add_u32(k, v);
    }
    w.add_f32("kws.default_threshold", 0.75);
    let tensors: [(&str, usize, f32); 12] = [
        ("kws.backbone.conv0.weight", 64 * 32 * 3, 0.08),
        ("kws.backbone.conv0.bias", 64, 0.1),
        ("kws.backbone.conv1.weight", 64 * 64 * 3, 0.06),
        ("kws.backbone.conv1.bias", 64, 0.1),
        ("kws.backbone.conv2.weight", 64 * 64 * 3, 0.06),
        ("kws.backbone.conv2.bias", 64, 0.1),
        ("kws.backbone.conv3.weight", 64 * 64 * 3, 0.06),
        ("kws.backbone.conv3.bias", 64, 0.1),
        ("kws.head.dense1.weight", 32 * 64, 40.0),
        ("kws.head.dense1.bias", 32, 0.1),
        ("kws.head.dense2.weight", 32, 12.0),
        ("kws.head.dense2.bias", 1, 0.1),
    ];
    let data: Vec<Vec<f32>> = tensors
        .iter()
        .map(|&(_, n, scale)| (0..n).map(|_| next() * scale).collect())
        .collect();
    for (name, n, _) in &tensors {
        w.add_tensor(*name, vec![*n as u64], GGML_TYPE_F32, n * 4);
    }
    let mut bytes = Vec::new();
    w.write_header_and_tensor_info(&mut bytes).unwrap();
    for d in &data {
        w.write_tensor_data(&mut bytes, bytemuck::cast_slice(d))
            .unwrap();
    }
    bytes
}

#[cfg(feature = "hexagon")]
fn main() {
    use cera::hotword::HotwordDetector;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args().skip(1);
    let wav = std::fs::read(
        args.next()
            .expect("usage: hexagon_hotword_probe <clip.wav> [model.gguf]"),
    )
    .expect("read the WAV");
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
    let model_arg: Vec<String> = std::env::args().skip(2).collect();
    let make = |_: ()| {
        if let Some(p) = model_arg.first() {
            HotwordDetector::from_file(p).expect("load the model")
        } else {
            HotwordDetector::from_bytes(synthetic_model()).expect("synthetic model")
        }
    };
    let mut cpu = make(());
    let mut npu = make(());
    assert!(npu.try_enable_hexagon(), "the NPU is not available");

    let (window, hop) = (19_200usize, 1_280usize);
    let windows: Vec<&[f32]> = (0..)
        .map(|i| i * hop)
        .take_while(|&s| s + window <= pcm.len())
        .map(|s| &pcm[s..s + window])
        .collect();

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
    // Scores and pooled embeddings of every window.
    type Run = (Vec<f32>, Vec<Vec<f32>>, f64, f64);
    let run = |d: &mut HotwordDetector| -> Run {
        let ((u0, s0), t0) = (cpu_seconds(), std::time::Instant::now());
        let mut scores = Vec::new();
        let mut embeddings = Vec::new();
        for w in &windows {
            scores.push(d.process_window(w).expect("window")[0]);
            embeddings.push(d.last_embedding().to_vec());
        }
        let (u1, s1) = cpu_seconds();
        (
            scores,
            embeddings,
            t0.elapsed().as_secs_f64(),
            u1 - u0 + s1 - s0,
        )
    };
    let (p_cpu, e_cpu, wall_cpu, cpu_cpu) = run(&mut cpu);
    let _ = run(&mut npu); // warm-up: builds and uploads the batch
    let (p_npu, e_npu, wall_npu, cpu_npu) = run(&mut npu);
    assert!(npu.is_accelerated(), "the NPU dropped out during the run");
    let (mut worst_emb, mut rms_emb) = (0f32, 0f64);
    for (a, b) in e_cpu.iter().flatten().zip(e_npu.iter().flatten()) {
        worst_emb = worst_emb.max((a - b).abs());
        rms_emb += (*a as f64).powi(2);
    }
    rms_emb = (rms_emb / e_cpu.iter().map(Vec::len).sum::<usize>() as f64).sqrt();
    println!("embedding: max |diff| {worst_emb:.6} (rms of the CPU embedding {rms_emb:.4})");

    let worst = p_cpu
        .iter()
        .zip(&p_npu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let (lo, hi) = p_cpu
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &p| (l.min(p), h.max(p)));
    let n = windows.len() as f64;
    println!(
        "{} windows: max |score diff| {worst:.6} (CPU scores span {lo:.3} to {hi:.3})",
        windows.len()
    );
    println!(
        "CPU: wall {:.2} ms/window, cpu {:.0} us/window",
        wall_cpu / n * 1e3,
        cpu_cpu / n * 1e6
    );
    println!(
        "NPU: wall {:.2} ms/window, cpu {:.0} us/window",
        wall_npu / n * 1e3,
        cpu_npu / n * 1e6
    );
    // One window every 80 ms of audio.
    println!(
        "per audio second (12.5 windows): CPU {:.4} cpu-s, NPU {:.4} cpu-s",
        cpu_cpu / n * 12.5,
        cpu_npu / n * 12.5
    );
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon");
}

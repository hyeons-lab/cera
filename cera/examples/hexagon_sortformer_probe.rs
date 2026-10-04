//! Check the Hexagon FastConformer on Sortformer's weights against the CPU, and time what each
//! costs the CPU.
//!
//! Sortformer's encoder is the same NeMo FastConformer family as LFM2-Audio's, so the NPU
//! encoder stages it through `HexagonAudioEncoder::from_parts` (no MLP adapter, no LFM2 mel
//! tables). Everything runs on the same input on both sides: the CPU pre-encode embeddings of
//! a clip, looped or trimmed to the requested length.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_sortformer_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_sortformer_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && echo 0 > /proc/$$/oom_score_adj && \
//!   ./hexagon_sortformer_probe sortformer-q8_0.gguf clip.wav 30'
//! ```
//!
//! Arguments: the Sortformer GGUF, a mono 16 kHz 16-bit WAV, and the clip length in seconds
//! (default: the WAV's own length; at most 608 encoder frames, about 48.6 s).

#[cfg(feature = "hexagon")]
fn read_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read the WAV");
    assert!(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV: {} bytes with no RIFF/WAVE header",
        bytes.len()
    );
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            assert!(
                body.len() >= 16,
                "WAV fmt chunk holds {} bytes, need 16",
                body.len()
            );
            let tag = u16::from_le_bytes([body[0], body[1]]);
            let ch = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            assert_eq!(
                (tag, ch, rate, bits),
                (1, 1, 16_000, 16),
                "need mono 16 kHz s16"
            );
        } else if id == b"data" {
            return body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

#[cfg(feature = "hexagon")]
fn main() {
    use std::sync::{Arc, Mutex};

    use cera::backend::hexagon::{FastRpcDriver, probe_device};
    use cera::model::audio_encoder_hexagon::HexagonAudioEncoder;
    use cera::model::sortformer::SortformerModel;
    use cera::model::sortformer_hexagon::{HexagonSortformerTail, TailStage};

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: hexagon_sortformer_probe <sortformer.gguf> <clip.wav> [seconds]");
    let wav_path = args.next().expect("a WAV path");
    let wav = read_wav(&wav_path);
    assert!(!wav.is_empty(), "the WAV has no audio samples");
    let seconds: f64 = args
        .next()
        .map_or(wav.len() as f64 / 16_000.0, |s| s.parse().expect("seconds"));
    let n = (seconds * 16_000.0) as usize;
    let pcm: Vec<f32> = (0..n).map(|i| wav[i % wav.len()]).collect();

    // The checkpoint's default streaming step encodes up to 608 frames (speaker cache 188,
    // FIFO 188, chunk 188 and its contexts); the low-latency preset needs about 390.
    const MAX_FRAMES: usize = 608;
    let model = SortformerModel::from_file(&model_path).expect("load Sortformer");
    let (mel, n_frames) = model.log_mel(&pcm);
    let (emb, t) = model.pre_encode(&mel, n_frames);
    println!("clip {seconds:.1} s -> {n_frames} mel frames -> {t} encoder frames");
    assert!(
        t <= MAX_FRAMES,
        "{t} frames exceed the NPU encoder's {MAX_FRAMES}"
    );

    // CPU reference: the FastConformer output after block 0 and after the last block, from
    // the model's own `predict`, with the time the blocks took.
    let n_layer = model.encoder_parts().layers.len();
    let mut xscaled = Vec::new();
    let mut first = Vec::new();
    let mut last = Vec::new();
    let mut stamps: Vec<(String, std::time::Instant)> = Vec::new();
    let (mut enc_proj, mut tf0, mut tf_last) = (Vec::new(), Vec::new(), Vec::new());
    let n_tf = model.config().tf_layers;
    let cpu_preds = model.predict_with_taps(&emb, t, &mut |name, v| {
        stamps.push((name.to_string(), std::time::Instant::now()));
        if name == "xscaled" {
            xscaled = v.to_vec();
        } else if name == "enc.layer0" {
            first = v.to_vec();
        } else if name == format!("enc.layer{}", n_layer - 1) {
            last = v.to_vec();
        } else if name == "enc_proj" {
            enc_proj = v.to_vec();
        } else if name == "tf.layer0" {
            tf0 = v.to_vec();
        } else if name == format!("tf.layer{}", n_tf - 1) {
            tf_last = v.to_vec();
        }
    });
    let at = |name: &str| stamps.iter().find(|(n, _)| n == name).map(|(_, i)| *i);
    if let (Some(a), Some(b)) = (at("xscaled"), at(&format!("enc.layer{}", n_layer - 1))) {
        println!("CPU blocks: {:.0} ms wall", (b - a).as_secs_f64() * 1e3);
    }

    let driver = FastRpcDriver::load().expect("load the FastRPC driver");
    let device = probe_device(&driver, None).expect("open a DSP session");
    let device = Arc::new(Mutex::new(device));
    device
        .lock()
        .unwrap()
        .queue_session_mut()
        .set_blocking_wait(true);
    let enc_driver = Arc::clone(&driver);
    let enc = HexagonAudioEncoder::from_parts(
        driver,
        Arc::clone(&device),
        &model.encoder_parts(),
        MAX_FRAMES,
        false,
    )
    .expect("stage the encoder");

    let tail = HexagonSortformerTail::new(enc_driver, Arc::clone(&device), &model, MAX_FRAMES)
        .expect("stage the tail");

    let report = |name: &str, cpu: &[f32], npu: &[f32]| {
        assert_eq!(cpu.len(), npu.len(), "{name}: length");
        let (mut dot, mut a2, mut b2, mut max) = (0f64, 0f64, 0f64, 0f32);
        for (c, n) in cpu.iter().zip(npu) {
            dot += *c as f64 * *n as f64;
            a2 += (*c as f64).powi(2);
            b2 += (*n as f64).powi(2);
            max = max.max((c - n).abs());
        }
        println!(
            "{name:<28} cosine {:.6}  max |diff| {max:.4}  (rms cpu {:.3})",
            dot / (a2.sqrt() * b2.sqrt()).max(1e-30),
            (a2 / cpu.len() as f64).sqrt()
        );
    };

    // Stem on the NPU against the CPU pre-encode.
    match enc.stem_output(&mel, n_frames, 64) {
        Ok(out) => report("conv stem output", &emb, &out),
        Err(e) => println!("conv stem: NPU error: {e}"),
    }
    // Blocks from the CPU's own x-scaled input.
    match enc.run_blocks(1, &xscaled, t) {
        Ok(out) => report("block 0", &first, &out),
        Err(e) => println!("block 0: NPU error: {e}"),
    }
    match enc.run_blocks(n_layer, &xscaled, t) {
        Ok(out) => report(&format!("blocks 0..{n_layer}"), &last, &out),
        Err(e) => println!("blocks: NPU error: {e}"),
    }

    // The tail, from the CPU's own FastConformer output.
    for (name, stage, want) in [
        ("encoder_proj", TailStage::Proj, &enc_proj),
        ("transformer layer 0", TailStage::Layers(1), &tf0),
        ("transformer last layer", TailStage::Layers(n_tf), &tf_last),
        ("speaker activities", TailStage::Full, &cpu_preds),
    ] {
        match tail.run(&last, t, stage) {
            Ok(out) => report(name, want, &out),
            Err(e) => println!("{name}: NPU error: {e}"),
        }
    }

    // Everything on the NPU: stem, x-scale (host), blocks, tail.
    let chain = || -> Vec<f32> {
        let emb = enc.stem_output(&mel, n_frames, 64).expect("NPU stem");
        let scale = model.encoder_input_scale();
        let x: Vec<f32> = emb.iter().map(|v| v * scale).collect();
        let enc_out = enc.run_blocks(n_layer, &x, t).expect("NPU blocks");
        tail.predict(&enc_out, t).expect("NPU tail")
    };
    let npu_preds = chain();
    report("end to end (all NPU)", &cpu_preds, &npu_preds);
    let hard = |v: &[f32]| v.iter().map(|&p| p > 0.5).collect::<Vec<_>>();
    let (a, b) = (hard(&cpu_preds), hard(&npu_preds));
    let agree = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    println!(
        "speaker decisions at 0.5: {agree}/{} agree ({:.3}%)",
        a.len(),
        100.0 * agree as f64 / a.len() as f64
    );

    // CPU seconds of this process: what the NPU path costs the CPU.
    let cpu_split = || {
        let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `getrusage` fills the struct; RUSAGE_SELF is valid.
        let ru = unsafe {
            libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr());
            ru.assume_init()
        };
        let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
        secs(ru.ru_utime) + secs(ru.ru_stime)
    };
    let reps = 5;
    let measure = |name: &str, run: &mut dyn FnMut()| {
        run();
        let (c0, t0) = (cpu_split(), std::time::Instant::now());
        for _ in 0..reps {
            run();
        }
        let (wall, cpu) = (
            t0.elapsed().as_secs_f64() / reps as f64,
            (cpu_split() - c0) / reps as f64,
        );
        println!(
            "{name:<26} wall {:6.0} ms ({:.3} per audio s)  cpu {:6.0} ms ({:.3} cpu-s per audio s)",
            wall * 1e3,
            wall / seconds,
            cpu * 1e3,
            cpu / seconds
        );
    };
    // The log-mel front end, and a whole live run, on a second model with the NPU as its
    // accelerator (mel, stem, blocks and tail) against the CPU model above.
    let npu_model = SortformerModel::from_file(&model_path).expect("load Sortformer again");
    let _npu = cera::model::sortformer_hexagon::try_hexagon_sortformer(&npu_model, MAX_FRAMES)
        .expect("stage the NPU diarizer");
    let (npu_mel, npu_n) = npu_model.log_mel(&pcm);
    assert_eq!(npu_n, n_frames);
    let worst = mel
        .iter()
        .zip(&npu_mel)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    report("log-mel (natural log)", &mel, &npu_mel);
    println!("log-mel worst |diff| {worst:.4} (log units; the floor is ln 2^-24 = -16.6)");
    let params = model.default_streaming().clone();
    let run_live = |m: &SortformerModel| {
        let mut live = m.new_live(params.clone()).unwrap();
        let mut out = Vec::new();
        for piece in pcm.chunks(1600) {
            out.extend(live.push_audio(piece).unwrap());
        }
        out.extend(live.finish().unwrap());
        out
    };
    let (cpu_live, npu_live) = (run_live(&model), run_live(&npu_model));
    report("live run, all NPU", &cpu_live, &npu_live);
    let (a, b) = (hard(&cpu_live), hard(&npu_live));
    let agree = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    println!(
        "live decisions at 0.5: {agree}/{} agree ({:.3}%)",
        a.len(),
        100.0 * agree as f64 / a.len() as f64
    );
    measure("CPU log-mel", &mut || drop(model.log_mel(&pcm)));
    measure("NPU log-mel", &mut || drop(npu_model.log_mel(&pcm)));
    measure("live run (CPU model)", &mut || drop(run_live(&model)));
    measure("live run (all NPU)", &mut || drop(run_live(&npu_model)));
    measure("CPU pre-encode (stem)", &mut || {
        drop(model.pre_encode(&mel, n_frames))
    });
    measure("NPU stem", &mut || {
        drop(enc.stem_output(&mel, n_frames, 64).expect("NPU stem"))
    });
    measure("CPU predict (all)", &mut || drop(model.predict(&emb, t)));
    measure("NPU tail", &mut || {
        drop(tail.predict(&last, t).expect("NPU tail"))
    });
    measure("NPU everything", &mut || drop(chain()));
    measure("NPU blocks", &mut || {
        drop(enc.run_blocks(n_layer, &xscaled, t).expect("NPU blocks"))
    });
}

//! Check the Hexagon Nemotron-3 diarizer against the CPU, and time what each costs the
//! CPU.
//!
//! Everything runs on the same input on both sides: the CPU log-mel of a clip, looped or
//! trimmed to the requested length, then the embedder and the predictor stage by stage
//! (input norm, first/last block, final norm, proj, upsampled, logits, full), then a whole
//! live run with the NPU as the model's accelerator.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_nemotron3_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_nemotron3_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && echo 0 > /proc/$$/oom_score_adj && \
//!   ./hexagon_nemotron3_probe nemotron3-q8_0.gguf clip.wav 30'
//! ```
//!
//! Arguments: the Nemotron-3 GGUF (converted with `--tail-outtype q8_0`), a mono 16 kHz
//! 16-bit WAV, and the clip length in seconds (default: the WAV's own length; at most 608
//! encoder frames, about 48.6 s).

#[cfg(feature = "hexagon")]
fn main() {
    use std::sync::{Arc, Mutex};

    use cera::backend::hexagon::{FastRpcDriver, probe_device};
    use cera::model::nemotron3_diarization::{Nemotron3Model, enc_frames};
    use cera::model::nemotron3_diarization_hexagon::{
        HexagonNemotron3Embed, HexagonNemotron3Predict, PredictStage, try_hexagon_nemotron3,
    };

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: hexagon_nemotron3_probe <nemotron3.gguf> <clip.wav> [seconds]");
    let wav_path = args.next().expect("a WAV path");
    let wav = cera::wav::read_wav_mono_16k(&wav_path).expect("read the WAV");
    assert!(!wav.is_empty(), "the WAV has no audio samples");
    let seconds: f64 = args
        .next()
        .map_or(wav.len() as f64 / 16_000.0, |s| s.parse().expect("seconds"));
    let n = (seconds * 16_000.0) as usize;
    let pcm: Vec<f32> = (0..n).map(|i| wav[i % wav.len()]).collect();

    // DSP-memory cap for one staging; the fit against the checkpoint's own window is
    // checked below, so a checkpoint change fails loud instead of declining silent.
    const MAX_FRAMES: usize = 608;
    let model = Nemotron3Model::from_file(&model_path).expect("load Nemotron-3");
    let window = model.default_streaming().window_frames();
    let (mel, n_frames) = model.log_mel(&pcm);
    let (emb, t) = model.embed(&mel, n_frames);
    // The mel tail rarely fills its group: pad groups exercise the key mask.
    let valid = enc_frames(n_frames);
    println!("clip {seconds:.1} s -> {n_frames} mel frames -> {t} encoder frames ({valid} valid)");
    assert!(
        t <= MAX_FRAMES && window <= MAX_FRAMES,
        "clip t={t} or checkpoint window={window} exceeds staging {MAX_FRAMES}"
    );

    // CPU reference: the embedder plus every `predict_with_taps` tap.
    let n_layer = model.config().n_layer;
    let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
    let cpu_preds = model.predict_with_taps(&emb, n_frames, &mut |name, v| {
        taps.push((name.to_string(), v.to_vec()));
    });
    let tap = |name: &str| {
        taps.iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no tap {name}"))
            .1
            .clone()
    };

    let driver = FastRpcDriver::load().expect("load the FastRPC driver");
    let device = probe_device(&driver, None).expect("open a DSP session");
    let device = Arc::new(Mutex::new(device));
    device
        .lock()
        .unwrap()
        .queue_session_mut()
        .set_blocking_wait(true);
    let enc_driver = Arc::clone(&driver);
    let embed =
        HexagonNemotron3Embed::new(Arc::clone(&driver), Arc::clone(&device), &model, MAX_FRAMES)
            .expect("stage the embedder");
    let predict = HexagonNemotron3Predict::new(enc_driver, Arc::clone(&device), &model, MAX_FRAMES)
        .expect("stage the predictor");

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

    // Embedder on the NPU against the CPU embeddings.
    match embed.embed(&mel, n_frames) {
        Ok(out) => report("embedder", &emb, &out),
        Err(e) => println!("embedder: NPU error: {e}"),
    }
    // Predictor stages from the CPU's own embeddings.
    for (name, stage, want) in [
        ("input norm", PredictStage::InputNorm, tap("input_norm")),
        (
            "encoder layer 0",
            PredictStage::Layers(1),
            tap("enc.layer0"),
        ),
        (
            "encoder last layer",
            PredictStage::Layers(n_layer),
            tap(&format!("enc.layer{}", n_layer - 1)),
        ),
        ("final norm", PredictStage::FinalNorm, tap("final_norm")),
        ("proj", PredictStage::Proj, tap("enc_proj")),
        ("upsampled", PredictStage::Upsampled, tap("upsampled")),
        ("logits", PredictStage::Logits, tap("logits")),
    ] {
        match predict.run(&emb, t, valid, stage) {
            Ok(out) => report(name, &want, &out),
            Err(e) => println!("{name}: NPU error: {e}"),
        }
    }

    // A padded run exercises the key mask: pad groups past the valid ones must neither
    // attract attention weight nor emit (offline and padded streaming pad to `pad_to`).
    let pad_to = 128;
    let padded_n = n_frames.next_multiple_of(pad_to);
    if padded_n != n_frames {
        let mut padded_mel = mel.clone();
        padded_mel.resize(padded_n * model.config().n_mel_bins, 0.0);
        let (padded_emb, padded_t) = model.embed(&padded_mel, padded_n);
        let cpu_padded = model.predict_cpu(&padded_emb, n_frames);
        match predict.predict(&padded_emb, n_frames) {
            Ok(out) => report("padded full (key mask)", &cpu_padded, &out),
            Err(e) => println!("padded full: NPU error: {e}"),
        }
        assert_eq!(padded_t, padded_n.div_ceil(8));
    }

    // Everything on the NPU: mel, embed, predict.
    let npu_model = Nemotron3Model::from_file(&model_path).expect("load Nemotron-3 again");
    let _npu = try_hexagon_nemotron3(&npu_model, MAX_FRAMES).expect("stage the NPU diarizer");
    let (npu_mel, npu_n) = npu_model.log_mel(&pcm);
    assert_eq!(npu_n, n_frames);
    report("log-mel (natural log)", &mel, &npu_mel);
    let (npu_emb, npu_t) = npu_model.embed(&npu_mel, npu_n);
    assert_eq!(npu_t, t);
    report("embedder (chained)", &emb, &npu_emb);
    let npu_preds = npu_model.predict(&npu_emb, n_frames);
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
    let params = model.default_streaming().clone();
    let run_live = |m: &Nemotron3Model| {
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
    // Any flip's margin: a decision both sides place away from 0.5 by more than their
    // difference would be a systematic divergence, not NPU noise on a borderline frame.
    let mut worst_margin = 0f32;
    let mut worst_at = 0;
    for (i, (&c, &n)) in cpu_live.iter().zip(&npu_live).enumerate() {
        if (c > 0.5) != (n > 0.5) {
            let margin = (0.5 - c).abs().max((0.5 - n).abs());
            if margin > worst_margin {
                worst_margin = margin;
                worst_at = i;
            }
        }
    }
    if agree != a.len() {
        println!(
            "worst flip margin {worst_margin:.4} at flat index {worst_at} (cpu {:.4}, npu {:.4})",
            cpu_live[worst_at], npu_live[worst_at]
        );
    }
    measure("CPU log-mel", &mut || drop(model.log_mel(&pcm)));
    measure("NPU log-mel", &mut || drop(npu_model.log_mel(&pcm)));
    measure("live run (CPU model)", &mut || drop(run_live(&model)));
    measure("live run (all NPU)", &mut || drop(run_live(&npu_model)));
    measure("CPU embed", &mut || drop(model.embed(&mel, n_frames)));
    measure("NPU embed", &mut || {
        drop(embed.embed(&mel, n_frames).expect("NPU embed"))
    });
    measure("CPU predict (all)", &mut || {
        drop(model.predict_cpu(&emb, n_frames))
    });
    measure("NPU predict", &mut || {
        drop(predict.predict(&emb, n_frames).expect("NPU predict"))
    });
}

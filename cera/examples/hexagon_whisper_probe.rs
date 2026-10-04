//! Where a Whisper transcription spends its CPU time when the model runs on the Hexagon NPU:
//! the log-mel (host) against the whole call (mel, encoder and decoder).
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_whisper_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_whisper_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && ./hexagon_whisper_probe whisper-tiny-q8_0.gguf clip.wav [seconds]'
//! ```
//!
//! Arguments: a Whisper GGUF (Q8_0 or Q4_0 to run on the NPU), a mono 16 kHz 16-bit WAV, and
//! how many seconds of it to transcribe (default 3, a typical VAD utterance).

#[cfg(feature = "hexagon")]
fn main() {
    use cera::engine::BackendPreference;
    use cera::model::whisper::{WhisperModel, WhisperTranscribeOpts};
    use cera::model::whisper_preprocessor::extract_whisper_mel;

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: hexagon_whisper_probe <whisper.gguf> <clip.wav> [seconds]");
    let wav = std::fs::read(args.next().expect("a WAV path")).expect("read the WAV");
    let seconds: f64 = args.next().map_or(3.0, |s| s.parse().expect("seconds"));
    let data = wav
        .windows(4)
        .position(|w| w == b"data")
        .expect("data chunk");
    let all: Vec<f32> = wav[data + 8..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect();
    let pcm = &all[..((seconds * 16_000.0) as usize).min(all.len())];

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

    let (model, tokenizer) =
        WhisperModel::from_file_with_backend(&model_path, BackendPreference::Auto).expect("load");
    println!(
        "{} s utterance, Hexagon NPU: {}",
        pcm.len() as f64 / 16_000.0,
        model.is_hexagon()
    );
    let opts = WhisperTranscribeOpts {
        language: Some("en".into()),
        ..Default::default()
    };
    let n_mel = model.weights.config.n_audio_mel_bins;
    let reps = 10;
    let measure = |name: &str, run: &mut dyn FnMut()| {
        run();
        let ((u0, s0), t0) = (cpu_seconds(), std::time::Instant::now());
        for _ in 0..reps {
            run();
        }
        let (u1, s1) = cpu_seconds();
        let per = |d: f64| d / reps as f64 * 1e3;
        println!(
            "{name:<26} wall {:7.2} ms  cpu {:7.2} ms (user {:7.2}, sys {:7.2})",
            per(t0.elapsed().as_secs_f64()),
            per(u1 - u0 + s1 - s0),
            per(u1 - u0),
            per(s1 - s0)
        );
    };
    let text = model
        .transcribe(&tokenizer, pcm, &opts)
        .expect("transcribe");
    println!("text (~{} tokens): {text}", tokenizer.encode(&text).len());
    measure("token_to_id", &mut || {
        let _ = tokenizer.token_to_id("<|transcribe|>");
    });
    measure("assemble prompt", &mut || {
        drop(cera::model::whisper::assemble_whisper_prompt(
            &model.special_tokens,
            Some(&tokenizer),
            Some("en"),
            false,
            false,
        ))
    });
    measure("log-mel (host)", &mut || {
        drop(extract_whisper_mel(pcm, n_mel))
    });

    // The NPU-side costs on their own (log-mel, encoder window, decoder steps, greedy
    // sampling) against the host work around them.
    if model.is_hexagon() {
        use cera::model::whisper_hexagon::init_hexagon_whisper;
        let hex = init_hexagon_whisper(&model.weights, &tokenizer).expect("stage the NPU model");
        assert!(
            hex.mel_on_dsp(),
            "log-mel is not staged on the DSP (stays on the host); the NPU timings below would measure the host"
        );
        let mel = extract_whisper_mel(pcm, n_mel);
        let mel_fb = hex.mel_fallbacks();
        let npu_mel = hex.log_mel(pcm);
        let worst = mel
            .iter()
            .zip(&npu_mel)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "log-mel on the NPU vs the host: max |diff| {worst:.5} over {} values",
            mel.len()
        );
        measure("log-mel (NPU)", &mut || drop(hex.log_mel(pcm)));
        assert_eq!(
            hex.mel_fallbacks(),
            mel_fb,
            "log-mel fell back to the host mid-probe; the NPU timings measured the host"
        );
        measure("encoder window (NPU)", &mut || {
            hex.encode_audio(&mel).expect("encode")
        });
        let mut logits = vec![0.0f32; model.weights.config.n_vocab];
        let sot = model.special_tokens.sot;
        let mut pos = 0usize;
        measure("one decoder step (NPU)", &mut || {
            hex.decode_step(sot, pos % 8, &mut logits).expect("decode");
            pos += 1;
        });
        // The host work around a decoder step: suppressing control tokens and the argmax.
        let special = model.special_tokens.clone();
        let mut sampler = cera::sampler::Sampler::new(cera::sampler::SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        });
        measure("suppress + sample", &mut || {
            cera::model::whisper::suppress_whisper_special_tokens(&mut logits, &special, false);
            let _ = sampler.sample(&mut logits);
        });
        // The greedy step against the host's suppress + argmax, over a real decode.
        hex.encode_audio(&mel).expect("encode");
        let prompt = cera::model::whisper::assemble_whisper_prompt(
            &special,
            Some(&tokenizer),
            Some("en"),
            false,
            false,
        );
        for (p, &tok) in prompt.iter().enumerate().take(prompt.len() - 1) {
            hex.decode_step(tok, p, &mut logits).expect("prefill");
        }
        let (mut cur, mut steps, mut mismatches) = (*prompt.last().unwrap(), 0, 0);
        for pos in prompt.len() - 1..prompt.len() + 40 {
            hex.decode_step(cur, pos, &mut logits).expect("decode");
            cera::model::whisper::suppress_whisper_special_tokens(&mut logits, &special, false);
            // Greedy sampling is stateless, so the sampler above is reused per step.
            let host = sampler.sample(&mut logits);
            let dsp = hex.decode_step_greedy(cur, pos, false).expect("greedy");
            steps += 1;
            mismatches += usize::from(host != dsp);
            if host == special.eot {
                break;
            }
            cur = host;
        }
        println!("greedy on the DSP vs the host: {mismatches} of {steps} tokens differ");
        assert_eq!(
            mismatches, 0,
            "DSP greedy diverged from host suppress+argmax"
        );
        let mut gpos = 0usize;
        measure("one greedy step (NPU)", &mut || {
            let _ = hex
                .decode_step_greedy(sot, gpos % 8, false)
                .expect("greedy");
            gpos += 1;
        });
        let tokens: Vec<u32> = (1000..1020).collect();
        measure("tokenizer.decode(20)", &mut || {
            drop(tokenizer.decode(&tokens))
        });
    }
    measure("whole transcribe", &mut || {
        drop(
            model
                .transcribe(&tokenizer, pcm, &opts)
                .expect("transcribe"),
        )
    });
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon");
}

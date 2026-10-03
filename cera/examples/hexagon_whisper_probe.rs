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
            "{name:<22} wall {:6.0} ms  cpu {:6.0} ms (user {:6.0}, sys {:6.0})",
            per(t0.elapsed().as_secs_f64()),
            per(u1 - u0 + s1 - s0),
            per(u1 - u0),
            per(s1 - s0)
        );
    };
    measure("log-mel (host)", &mut || {
        drop(extract_whisper_mel(pcm, n_mel))
    });

    // The NPU model's two phases on their own: the encoder window and one decoder step.
    if model.is_hexagon() {
        use cera::model::whisper_hexagon::init_hexagon_whisper;
        let hex = init_hexagon_whisper(&model.weights, &tokenizer).expect("stage the NPU model");
        let mel = extract_whisper_mel(pcm, n_mel);
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

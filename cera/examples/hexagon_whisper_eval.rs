//! Transcribe a list of 16 kHz mono WAV clips with a Whisper model on the Hexagon NPU and print
//! each transcript with its wall time, so settings (`CERA_WHISPER_AUDIO_CTX`,
//! `CERA_HEXAGON_WHISPER_FA`, `CERA_HEXAGON_WHISPER_MM`) can be compared on the same clips.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_whisper_eval --features hexagon
//! adb shell 'cd /data/local/tmp/whisper-eval && ./hexagon_whisper_eval whisper.gguf a.wav b.wav ...'
//! ```
//!
//! Every output line starts with `EVAL`: `EVAL <file> <seconds> s <wall ms> ms <transcript>`.

#[cfg(feature = "hexagon")]
fn main() {
    use cera::engine::BackendPreference;
    use cera::model::whisper::{WhisperModel, WhisperTranscribeOpts};

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: hexagon_whisper_eval <whisper.gguf> <clip.wav>...");
    let (model, tokenizer) =
        WhisperModel::from_file_with_backend(&model_path, BackendPreference::Auto).expect("load");
    println!("EVAL hexagon {}", model.is_hexagon());
    let opts = WhisperTranscribeOpts {
        language: Some("en".into()),
        ..Default::default()
    };
    let mut first = true;
    for path in args {
        let wav = std::fs::read(&path).expect("read the WAV");
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
        if first {
            // The first call pays one-time costs; keep it out of the timings.
            let _ = model.transcribe(&tokenizer, &pcm, &opts);
            first = false;
        }
        let t = std::time::Instant::now();
        let text = model
            .transcribe(&tokenizer, &pcm, &opts)
            .expect("transcribe");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let name = path.rsplit('/').next().unwrap_or(&path);
        println!(
            "EVAL {name} {:.2} s {ms:.0} ms {}",
            pcm.len() as f64 / 16_000.0,
            text.trim()
        );
    }
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon to run this example");
}

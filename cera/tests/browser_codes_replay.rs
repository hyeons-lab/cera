//! Replays a code sequence captured from a browser WebGPU run through the
//! native CPU vocoder.
//!
//! The browser occasionally emits a single audio frame whose PCM runs far
//! outside [-1, 1] (absmax 2.7 and 11.9 in the run this fixture came from)
//! while its neighbours sit around 0.1-0.5. Native runs of the same pipeline
//! never showed it, so the question is whether those codes genuinely render
//! that loud (the vocoder is faithful and the sampler picked a bad code) or
//! whether only the wasm build mis-renders them.
//!
//! The detokenizer is stateful, so the whole prefix has to be replayed in
//! order for frame N to mean anything.
//!
//! Fixture: `CERA_BROWSER_CODES` points at a file of
//! `c0,..,c7|browser_absmax` lines. Skips when unset.

use cera::model::audio_decoder::{AudioDecoderWeights, DetokenizerWeights};

/// The vocoder has to be the same quant the capture ran on, so this is an
/// explicit path rather than the usual `~/.leap` default.
fn load_vocoder() -> Option<std::sync::Arc<cera::gguf::GgufFile>> {
    let path = std::path::PathBuf::from(std::env::var("CERA_VOCODER").ok()?);
    if !path.exists() {
        eprintln!("vocoder not found at {}, skipping", path.display());
        return None;
    }
    cera::gguf::GgufFile::open_arc(&path).ok()
}

#[test]
fn browser_codes_replay_matches_native() {
    let Ok(fixture) = std::env::var("CERA_BROWSER_CODES") else {
        eprintln!("CERA_BROWSER_CODES unset, skipping");
        return;
    };
    let Some(gguf) = load_vocoder() else {
        return;
    };
    let text = std::fs::read_to_string(&fixture).expect("reading the code fixture");

    let rows: Vec<([i32; 8], f32)> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (codes, absmax) = l.split_once('|').expect("each line is codes|absmax");
            let v: Vec<i32> = codes
                .split(',')
                .map(|c| c.trim().parse().expect("code is an integer"))
                .collect();
            let mut arr = [0i32; 8];
            arr.copy_from_slice(&v);
            (arr, absmax.trim().parse().expect("absmax is a float"))
        })
        .collect();

    let detok_w = DetokenizerWeights::from_gguf(&gguf).expect("detok weights");
    let dec_w = AudioDecoderWeights::from_gguf(&gguf).expect("decoder weights");
    let mut state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    let mut streamer = cera::model::audio_decoder::IstftStreamer::new(
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );

    let mut worst = (0usize, 0.0f32, 0.0f32);
    for (i, (codes, browser_absmax)) in rows.iter().enumerate() {
        let spectrum =
            cera::model::audio_decoder::detokenize_to_spectrum(&detok_w, &dec_w, &mut state, codes);
        // The two halves of a frame are (log_abs, angle) and only the first
        // feeds `exp()`, so they have to be reported apart: a large angle is
        // harmless (cos/sin wrap it), a large log-magnitude is not.
        let bins = detok_w.config.n_fft / 2 + 1;
        let frame_size = bins * 2;
        let mut mag_max = f32::NEG_INFINITY;
        let mut ang_max = 0.0f32;
        for f in spectrum.chunks_exact(frame_size) {
            for &v in &f[..bins] {
                mag_max = mag_max.max(v);
            }
            for &v in &f[bins..] {
                ang_max = ang_max.max(v.abs());
            }
        }
        let pcm = streamer.feed_frames(&spectrum);
        let native_absmax = pcm.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let flag = if native_absmax > 1.0 {
            " <== NATIVE OVER 1.0"
        } else {
            ""
        };
        let mismatch = if (native_absmax - browser_absmax).abs() > 0.05 {
            " <== DIVERGES FROM BROWSER"
        } else {
            ""
        };
        println!(
            "frame {i:3}: native_absmax={native_absmax:8.4}  browser_absmax={browser_absmax:8.4}  \
             log_abs_max={mag_max:7.3}(exp {:9.3e})  ang_max={ang_max:8.2}  n={}{flag}{mismatch}",
            mag_max.exp(),
            pcm.len()
        );
        if native_absmax > worst.1 {
            worst = (i, native_absmax, *browser_absmax);
        }
    }
    println!(
        "worst native frame: {} at {:.4} (browser said {:.4})",
        worst.0, worst.1, worst.2
    );
}

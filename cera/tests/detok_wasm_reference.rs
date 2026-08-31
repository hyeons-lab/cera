//! Emits the native detokenizer's spectrum for a fixed code sequence, so a
//! wasm build can be diffed against it bin for bin.
//!
//! Replaying captured codes through the native vocoder showed it disagreeing
//! with the browser on most frames, not only the ones whose PCM blew past
//! [-1, 1] -- same codes, same weights, same quant, same state trajectory. That
//! puts the detokenizer itself under suspicion rather than the ISTFT
//! downstream, but "the whole pipeline disagrees" is not a localization.
//!
//! This test is the reference half of a two-sided comparison. It takes codes
//! rather than generating them, so no LLM, sampler, or GPU is in the loop and a
//! difference can only come from the detokenizer. The wasm half is
//! `WebGpuSession.debugDetokenizeSpectrum`, which runs the same function over
//! the same codes; `audio-harness.html` fetches this output and diffs.
//!
//! ```text
//! CERA_VOCODER=~/models/liquid-ci/vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf \
//! CERA_SPECTRUM_OUT=.../native_spectrum.bin \
//!   cargo test -p cera --test detok_wasm_reference -- --nocapture
//! ```
//!
//! The payload is little-endian f32, frames concatenated in order. A sidecar
//! `.json` carries the shape and the codes so the browser does not have to be
//! told them separately.

use cera::model::audio_decoder::{AudioDecoderWeights, DetokenizerWeights};

/// A fixed, arbitrary code sequence. Values are in range for the 2049-entry
/// codebooks and span low and high indices; what matters is only that both
/// sides run the identical list.
const CODES: &[[i32; 8]] = &[
    [1049, 811, 1626, 290, 457, 478, 712, 1533],
    [127, 1470, 457, 1422, 481, 1509, 976, 2008],
    [1880, 1050, 1400, 142, 457, 1720, 666, 1477],
    [1156, 893, 1616, 1229, 839, 242, 1249, 633],
    [1792, 1032, 696, 857, 1985, 388, 1764, 685],
    [1914, 1431, 17, 1493, 1482, 473, 1190, 1490],
    [288, 1478, 1289, 1408, 1256, 241, 7, 1051],
    [1834, 533, 968, 1897, 822, 1372, 158, 1797],
    [15, 985, 728, 1381, 381, 366, 515, 1804],
    [78, 1510, 889, 1085, 726, 896, 927, 1289],
    [2047, 0, 2047, 0, 2047, 0, 2047, 0],
    [0, 1, 2, 3, 4, 5, 6, 7],
];

#[test]
fn emit_native_spectrum_reference() {
    let Ok(vocoder) = std::env::var("CERA_VOCODER") else {
        eprintln!("CERA_VOCODER unset, skipping");
        return;
    };
    let path = std::path::PathBuf::from(&vocoder);
    if !path.exists() {
        eprintln!("vocoder not found at {}, skipping", path.display());
        return;
    }
    let gguf = cera::gguf::GgufFile::open_arc(&path).expect("opening the vocoder");
    // The depthformer always comes from the vocoder, but the detokenizer does
    // not: `cera-wasm` prefers the bundle's *audio tokenizer* GGUF and only
    // falls back to the vocoder when that fails to parse. `CERA_DETOK` selects
    // the same source so the reference can be pointed at either one.
    let detok_gguf = match std::env::var("CERA_DETOK") {
        Ok(p) => cera::gguf::GgufFile::open_arc(std::path::Path::new(&p))
            .expect("opening the detokenizer GGUF"),
        Err(_) => std::sync::Arc::clone(&gguf),
    };
    let detok_w = DetokenizerWeights::from_gguf(&detok_gguf).expect("detok weights");
    let dec_w = AudioDecoderWeights::from_gguf(&gguf).expect("decoder weights");

    let bins = detok_w.config.n_fft / 2 + 1;
    let frame_size = bins * 2;

    // One state across the whole sequence: the detokenizer carries conv buffers
    // and a KV cache, so frame N is only reproducible after frames 0..N.
    let mut state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);
    let mut spectrum = Vec::new();
    for codes in CODES {
        spectrum.extend_from_slice(&cera::model::audio_decoder::detokenize_to_spectrum(
            &detok_w, &dec_w, &mut state, codes,
        ));
    }

    let sub_frames = spectrum.len() / frame_size;
    println!(
        "codes={} sub_frames={} bins={} floats={}",
        CODES.len(),
        sub_frames,
        bins,
        spectrum.len()
    );
    for (i, f) in spectrum.chunks_exact(frame_size).enumerate().take(6) {
        let mag = f[..bins].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let ang = f[bins..].iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("  sub_frame {i:3}: log_abs_max={mag:8.4}  ang_absmax={ang:8.3}");
    }
    assert!(
        spectrum.iter().all(|v| v.is_finite()),
        "native spectrum has NaN/Inf"
    );

    // Same fingerprint the wasm side reports, so a weight-level difference can
    // be separated from an arithmetic one. Both fields are plain f32 in the
    // GGUF, so a mismatch means the file was read differently, not dequantized
    // differently.
    for (name, v) in [
        ("output_norm", &detok_w.output_norm),
        ("lin_b", &detok_w.lin_b),
    ] {
        println!(
            "{name}: len={} sum={:.6} absmax={:.6} first4={:?}",
            v.len(),
            v.iter().sum::<f32>(),
            v.iter().fold(0.0f32, |a, &b| a.max(b.abs())),
            &v[..4.min(v.len())]
        );
    }

    // Quantized weights, probed the same way the wasm side does: a row that
    // dequantizes identically on both backends clears the GGUF read and the
    // block decode, leaving only the matmul kernels.
    for (name, w) in [
        ("emb_weight", &detok_w.emb_weight),
        ("lin_w", &detok_w.lin_w),
    ] {
        let mut row = vec![0.0f32; w.cols];
        w.dequantize_row(0, &mut row);
        println!(
            "{name}: rows={} cols={} dtype={:?} row0_sum={:.6} row0_absmax={:.6} first4={:?}",
            w.rows,
            w.cols,
            w.dtype,
            row.iter().sum::<f32>(),
            row.iter().fold(0.0f32, |a, &b| a.max(b.abs())),
            &row[..4.min(row.len())]
        );
    }

    if let Ok(out) = std::env::var("CERA_SPECTRUM_OUT") {
        let bytes: Vec<u8> = spectrum.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out, &bytes).expect("writing the spectrum payload");
        let meta = serde_json::json!({
            "bins": bins,
            "frame_size": frame_size,
            "sub_frames": sub_frames,
            "floats": spectrum.len(),
            "codes": CODES.iter().flatten().copied().collect::<Vec<i32>>(),
        });
        std::fs::write(
            format!("{out}.json"),
            serde_json::to_string_pretty(&meta).expect("serializing metadata"),
        )
        .expect("writing the spectrum metadata");
        println!("wrote {out} ({} bytes) and {out}.json", bytes.len());
    } else {
        println!("CERA_SPECTRUM_OUT unset, not writing a payload");
    }
}

//! Parity of cera's CPU Sortformer against NVIDIA's NeMo implementation.
//!
//! The reference is `cera/tests/fixtures/sortformer/golden.json` (committed) plus
//! `golden.safetensors` (every intermediate tensor and per-step streaming state, ~15 MB, NOT
//! committed). Both are produced by `scripts/sortformer/gen_golden.py` from the committed
//! `clip.wav`; the models by `scripts/sortformer/convert_sortformer.py`. Everything model-backed
//! lives in `~/.leap/models/sortformer/` and skips with a message when absent
//! (`CERA_REQUIRE_MODEL=1` turns a skip into a failure).
//!
//! Three layers, each isolating a different kind of bug:
//!
//! * `stages_teacher_forced`: every stage is fed NeMo's *own* input for that stage and compared
//!   with NeMo's output, so the first stage that differs names the bug (mel, stem, x-scale, a
//!   FastConformer block, `encoder_proj`, a Transformer layer, the head).
//! * `offline_end_to_end`: PCM to sigmoids with nothing from NeMo but the answer.
//! * `streaming_*`: the streaming loop for four presets, comparing the predictions and, for
//!   the two tiny-cache presets that overflow the speaker cache, the speaker cache, FIFO and
//!   silence profile after every step.
//!
//! The file also pins the live path (`SortformerLive`, `MelStream`), the loader's refusal of
//! hostile files, and the argument checks of `step`.
//!
//! The CPU model is not batched, so these are slow in a debug build: run them with `--release`.
//! The 33-step low-latency preset is slower still and is `#[ignore]`d: run it with `--ignored`.

#![cfg(feature = "mmap")] // `SortformerModel::from_file`

use std::collections::HashMap;
use std::path::PathBuf;

use cera::convert::safetensors::SafeTensorsHeader;
use cera::model::sortformer::{SortformerModel, StreamingParams};

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".leap/models/sortformer")
}

/// The committed fixtures. `SORTFORMER_FIXTURES` overrides the compile-time path so a test binary
/// cross-built for a device (where the source tree does not exist) can run against pushed copies.
fn fixtures_dir() -> PathBuf {
    match std::env::var_os("SORTFORMER_FIXTURES") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sortformer"),
    }
}

/// A model-backed file, or `None` to skip.
fn local(rel: &str) -> Option<PathBuf> {
    let path = models_dir().join(rel);
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display()
        );
        eprintln!("{} not found, skipping", path.display());
        return None;
    }
    Some(path)
}

fn model(file: &str) -> Option<SortformerModel> {
    let path = local(file)?;
    Some(SortformerModel::from_file(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display())))
}

/// 16-bit mono PCM WAV to f32, the way `soundfile.read(dtype="float32")` does it.
fn read_clip() -> Vec<f32> {
    let bytes = std::fs::read(fixtures_dir().join("clip.wav")).expect("clip.wav");
    assert_eq!(&bytes[..4], b"RIFF");
    let mut pos = 12;
    let mut fmt_ok = false;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..pos + 8 + len];
        if id == b"fmt " {
            let (tag, ch, rate, bits) = (
                u16::from_le_bytes(body[0..2].try_into().unwrap()),
                u16::from_le_bytes(body[2..4].try_into().unwrap()),
                u32::from_le_bytes(body[4..8].try_into().unwrap()),
                u16::from_le_bytes(body[14..16].try_into().unwrap()),
            );
            assert_eq!((tag, ch, rate, bits), (1, 1, 16_000, 16), "clip.wav format");
            fmt_ok = true;
        } else if id == b"data" {
            assert!(fmt_ok, "data before fmt");
            return body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk in clip.wav");
}

struct Golden {
    json: serde_json::Value,
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Golden {
    fn load() -> Option<Self> {
        let path = local("golden/golden.safetensors")?;
        let bytes = std::fs::read(&path).expect("read golden.safetensors");
        let header = SafeTensorsHeader::parse_from_bytes(&bytes).expect("safetensors header");
        let base = header.header_size_bytes;
        let mut tensors = HashMap::new();
        for (name, info) in &header.tensors {
            let raw = &bytes[base + info.data_offsets.0..base + info.data_offsets.1];
            let vals: Vec<f32> = match info.dtype.as_str() {
                "F32" => raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
                "I64" => raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b) as f32)
                    .collect(),
                other => panic!("{name}: unexpected dtype {other}"),
            };
            tensors.insert(name.clone(), (info.shape.clone(), vals));
        }
        let json = serde_json::from_slice(
            &std::fs::read(fixtures_dir().join("golden.json")).expect("golden.json"),
        )
        .expect("parse golden.json");
        Some(Self { json, tensors })
    }

    fn t(&self, name: &str) -> &[f32] {
        &self
            .tensors
            .get(name)
            .unwrap_or_else(|| {
                panic!("golden tensor {name} missing; regenerate with gen_golden.py")
            })
            .1
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.tensors[name].0
    }

    fn n_frames(&self) -> usize {
        self.json["n_frames"].as_u64().unwrap() as usize
    }

    fn mel_len(&self) -> usize {
        self.json["mel_len"].as_u64().unwrap() as usize
    }
}

struct Diff {
    max_abs: f32,
    cosine: f64,
}

fn diff(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let (mut dot, mut a2, mut b2, mut max_abs) = (0f64, 0f64, 0f64, 0f32);
    for (&a, &b) in got.iter().zip(want) {
        dot += a as f64 * b as f64;
        a2 += (a as f64).powi(2);
        b2 += (b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    Diff {
        max_abs,
        cosine: dot / (a2.sqrt() * b2.sqrt()).max(1e-30),
    }
}

#[track_caller]
fn check(stage: &str, got: &[f32], want: &[f32], max_abs: f32, min_cos: f64) {
    let d = diff(got, want);
    eprintln!(
        "  {stage:<14} max|d| {:.3e}  cos {:.8}",
        d.max_abs, d.cosine
    );
    assert!(
        d.max_abs <= max_abs && d.cosine >= min_cos,
        "{stage}: max|d| {:.3e} (limit {max_abs:.0e}), cosine {:.8} (limit {min_cos})",
        d.max_abs,
        d.cosine
    );
}

/// NeMo's `[128 x T_padded]` mel to our time-major `[n x 128]`.
fn mel_time_major(golden: &Golden, n: usize) -> Vec<f32> {
    let g = golden.t("mel");
    let t_pad = golden.shape("mel")[1];
    let mut out = vec![0.0f32; n * 128];
    for f in 0..n {
        for m in 0..128 {
            out[f * 128 + m] = g[m * t_pad + f];
        }
    }
    out
}

#[test]
fn stages_teacher_forced() {
    let (Some(m), Some(golden)) = (model("sortformer-4spk-v2.1-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let n_mel = golden.mel_len();
    let t = golden.n_frames();

    // Front end from PCM.
    let (mel, n) = m.log_mel(&pcm);
    assert_eq!(n, n_mel, "mel frame count");
    check("mel", &mel, &mel_time_major(&golden, n), 2e-3, 0.99999);

    // Stem from NeMo's mel.
    let nemo_mel = mel_time_major(&golden, n_mel);
    let (pre, t_pre) = m.pre_encode(&nemo_mel, n_mel);
    assert_eq!(t_pre, t, "encoder frame count");
    let want_pre = &golden.t("pre_encode")[..t * 512];
    check("pre_encode", &pre, want_pre, 2e-3, 0.99999);

    // Everything after the stem from NeMo's pre-encode embeddings.
    let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
    let preds = m.predict_with_taps(want_pre, t, &mut |name, v| {
        taps.push((name.to_string(), v.to_vec()))
    });
    for (name, got) in &taps {
        let width = if name.starts_with("enc.") || name == "xscaled" {
            512
        } else {
            192
        };
        let want = &golden.t(name)[..t * width];
        // Tolerance scales with the activation size: the FastConformer residual stream is
        // large, so an absolute bound on it would be either vacuous or flaky.
        let scale = want.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1.0);
        check(name, got, want, 2e-4 * scale, 0.999999);
    }
    check(
        "preds",
        &preds,
        &golden.t("preds_offline")[..t * 4],
        2e-4,
        0.99999,
    );
}

#[test]
fn offline_end_to_end() {
    let Some(golden) = Golden::load() else { return };
    let pcm = read_clip();
    let t = golden.n_frames();
    let want = &golden.t("preds_offline")[..t * 4];

    // (file, max |d| on the sigmoids, decisions allowed to flip at 0.5)
    for (file, tol, flips) in [
        ("sortformer-4spk-v2.1-f32.gguf", 2e-3f32, 0usize),
        ("sortformer-4spk-v2.1-q8_0.gguf", 5e-2, 3),
    ] {
        let Some(m) = model(file) else { continue };
        let got = m.diarize_offline(&pcm).unwrap();
        assert_eq!(got.len(), t * 4, "{file}");
        eprintln!("{file}");
        let d = diff(&got, want);
        let flipped = got
            .iter()
            .zip(want)
            .filter(|&(&a, &b)| (a > 0.5) != (b > 0.5))
            .count();
        eprintln!(
            "  max|d| {:.3e}  cos {:.8}  flips {flipped}",
            d.max_abs, d.cosine
        );
        assert!(
            d.max_abs <= tol,
            "{file}: sigmoid max|d| {:.3e} > {tol:e}",
            d.max_abs
        );
        assert!(
            flipped <= flips,
            "{file}: {flipped} decisions flipped (allowed {flips})"
        );
    }
}

/// The preset's parameters from the golden JSON, over the checkpoint's score hyper-parameters.
fn preset(m: &SortformerModel, golden: &Golden, name: &str) -> StreamingParams {
    let p = &golden.json["presets"][name]["params"];
    let get = |k: &str| p[k].as_u64().unwrap() as usize;
    m.default_streaming().with_chunking(
        get("chunk_len"),
        get("chunk_left_context"),
        get("chunk_right_context"),
        get("fifo_len"),
        get("spkcache_len"),
        get("spkcache_update_period"),
    )
}

/// Run one preset and compare with NeMo. `state` also compares the cache/FIFO/silence state after
/// every step but the last: the last step covers NeMo's padded frames, whose embeddings NeMo
/// keeps and cera does not compute (zeros), so only that step's state is not comparable.
fn run_streaming(file: &str, name: &str, state: bool, tol_preds: f32, tol_state: f32) {
    let (Some(m), Some(golden)) = (model(file), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let params = preset(&m, &golden, name);
    let n_steps = golden.json["presets"][name]["n_steps"].as_u64().unwrap() as usize;
    let steps = golden.json["presets"][name]["steps"].as_array().unwrap();
    let recorded = golden.json["presets"][name]["state_recorded"]
        .as_bool()
        .unwrap();
    eprintln!("{file} / {name}: {n_steps} steps");

    let (mel, n) = m.log_mel(&pcm);
    let mut stream = m.new_stream(params).unwrap();
    let mut worst_state = 0f32;
    let (mut compressed_steps, mut max_sil) = (0usize, 0usize);
    let got = stream
        .diarize_features_with(&mel, n, &mut |i, s, chunk_preds| {
            let bounds = steps[i]["chunk_frames"].as_array().unwrap();
            let (lo, hi) = (
                bounds[0].as_u64().unwrap() as usize,
                bounds[1].as_u64().unwrap() as usize,
            );
            assert_eq!(
                chunk_preds.len(),
                (hi - lo) * 4,
                "step {i}: chunk frame count"
            );
            if !(state && recorded) || i + 1 == n_steps {
                return;
            }
            let key = |what: &str| format!("{name}.step{i}.{what}");
            let mut cmp = |what: &str, got: &[f32], want_name: String| {
                if !golden.tensors.contains_key(&want_name) {
                    assert!(
                        got.is_empty(),
                        "step {i}: {what} is {} values, NeMo's is empty",
                        got.len()
                    );
                    return;
                }
                let want = golden.t(&want_name);
                assert_eq!(got.len(), want.len(), "step {i}: {what} length");
                let d = diff(got, want);
                worst_state = worst_state.max(d.max_abs);
                assert!(
                    d.max_abs <= tol_state,
                    "step {i}: {what} max|d| {:.3e} > {tol_state:e}",
                    d.max_abs
                );
            };
            compressed_steps += usize::from(s.spkcache_preds().is_some());
            max_sil = max_sil.max(s.n_sil_frames());
            cmp("spkcache", s.spkcache(), key("spkcache"));
            cmp("fifo", s.fifo(), key("fifo"));
            cmp("fifo_preds", s.fifo_preds(), key("fifo_preds"));
            cmp("mean_sil_emb", s.mean_sil_emb(), key("mean_sil_emb"));
            match s.spkcache_preds() {
                Some(sp) => cmp("spkcache_preds", sp, key("spkcache_preds")),
                None => assert!(
                    !golden.tensors.contains_key(&key("spkcache_preds")),
                    "step {i}: NeMo has spkcache_preds, cera does not yet"
                ),
            }
            let n_sil = golden.t(&key("n_sil_frames"))[0] as usize;
            assert_eq!(s.n_sil_frames(), n_sil, "step {i}: silence frame count");
        })
        .expect("streaming");
    if state {
        eprintln!(
            "  worst cache/FIFO/silence max|d| over steps: {worst_state:.3e} \
             ({compressed_steps} steps with a compressed cache, {max_sil} silence frames profiled)"
        );
        if name.starts_with("tiny") {
            // The point of these presets: without compression and silence slots in play, the
            // state comparison above would pass without touching the code worth testing.
            assert!(
                compressed_steps > 0,
                "{name}: the speaker cache never compressed"
            );
            assert!(
                max_sil > 0,
                "{name}: no silence frames reached the silence profile"
            );
        }
    }

    let want = golden.t(&format!("{name}.total_preds"));
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: total frames (NeMo includes its padded frames)"
    );
    let d = diff(&got, want);
    let flipped = got
        .iter()
        .zip(want)
        .filter(|&(&a, &b)| (a > 0.5) != (b > 0.5))
        .count();
    eprintln!(
        "  predictions max|d| {:.3e}  cos {:.8}  flips {flipped}",
        d.max_abs, d.cosine
    );
    assert!(
        d.max_abs <= tol_preds,
        "{name}: max|d| {:.3e} > {tol_preds:e}",
        d.max_abs
    );
}

#[test]
fn streaming_tiny_overflows_the_cache() {
    run_streaming("sortformer-4spk-v2.1-f32.gguf", "tiny", true, 5e-3, 5e-3);
}

#[test]
fn streaming_tiny_nofifo() {
    run_streaming(
        "sortformer-4spk-v2.1-f32.gguf",
        "tiny_nofifo",
        true,
        5e-3,
        5e-3,
    );
}

/// Predictions only in effect: two steps and the last is excluded, so no step has a compressed
/// cache to compare. The `tiny*` presets carry the cache, FIFO and silence-profile coverage.
#[test]
fn streaming_default_preset() {
    run_streaming("sortformer-4spk-v2.1-f32.gguf", "default", true, 2e-3, 2e-3);
}

#[test]
#[ignore = "33 steps, about a minute in release; run with --ignored"]
fn streaming_low_latency_preset() {
    run_streaming(
        "sortformer-4spk-v2.1-f32.gguf",
        "low_latency",
        false,
        5e-3,
        5e-3,
    );
}

/// `step` takes offsets in mel frames like NeMo's loader: left context is a whole number of
/// encoder frames, right context rounds UP (a 12-frame lookahead is 2 encoder frames, not 1). A
/// padded padding-tail yields zero predictions for the frames past the audio. The parity
/// presets only ever see right offsets that are multiples of 8, so this is where rounding shows.
#[test]
fn step_bookkeeping_follows_nemo() {
    let Some(m) = model("sortformer-4spk-v2.1-f32.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mel, _) = m.log_mel(&pcm);
    let params = m.default_streaming().clone();

    // 100 mel frames -> 13 encoder frames; 12 frames of lookahead -> ceil(12 / 8) = 2 of them.
    let mut s = m.new_stream(params.clone()).unwrap();
    let preds = s.step(&mel[..100 * 128], 100, 100, 0, 12).unwrap();
    assert_eq!(preds.len(), (13 - 2) * 4, "right context rounds up");

    let mut s = m.new_stream(params.clone()).unwrap();
    let preds = s.step(&mel[..100 * 128], 100, 100, 16, 0).unwrap();
    assert_eq!(
        preds.len(),
        (13 - 2) * 4,
        "16 frames of left context are 2 encoder frames"
    );

    // 112 frames padded, 100 of them audio: 14 encoder frames, the last past the end.
    let mut padded = mel[..100 * 128].to_vec();
    padded.resize(112 * 128, 0.0);
    let mut s = m.new_stream(params).unwrap();
    let preds = s.step(&padded, 112, 100, 0, 0).unwrap();
    assert_eq!(preds.len(), 14 * 4);
    assert!(
        preds[13 * 4..].iter().all(|&p| p == 0.0),
        "frame past the audio predicts nothing"
    );
    assert!(
        preds[..13 * 4].iter().all(|&p| p > 0.0 && p < 1.0),
        "audio frames are sigmoids"
    );
}

/// Push `pcm` in pieces of `piece` samples through a live diarizer; returns every prediction.
fn run_live(m: &SortformerModel, pcm: &[f32], params: StreamingParams, piece: usize) -> Vec<f32> {
    let mut live = m.new_live(params).unwrap();
    let mut out = Vec::new();
    for part in pcm.chunks(piece) {
        out.extend(live.push_audio(part).unwrap());
    }
    out.extend(live.finish().unwrap());
    assert_eq!(live.frames_emitted() * 4, out.len());
    out
}

/// The incremental mel must be the whole-clip mel exactly, for any way of cutting the audio:
/// every frame is the same arithmetic on the same samples.
#[test]
fn mel_stream_is_bit_identical_to_the_whole_clip_mel() {
    let Some(m) = model("sortformer-4spk-v2.1-f32.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (want, n) = m.log_mel(&pcm);
    for piece in [1usize, 7, 160, 161, 777, 8000, pcm.len()] {
        let mut ms = m.new_mel_stream();
        let mut got = Vec::new();
        for part in pcm.chunks(piece) {
            got.extend(ms.push(part).unwrap());
        }
        got.extend(ms.finish());
        assert_eq!(
            got.len(),
            want.len(),
            "piece {piece}: frame count (want {n})"
        );
        assert_eq!(ms.frames(), n);
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "piece {piece}: mel differs from the whole-clip mel"
        );
    }
    // Audio shorter than one hop has no valid frames, like the offline path.
    let mut ms = m.new_mel_stream();
    assert!(ms.push(&pcm[..100]).unwrap().is_empty() && ms.finish().is_empty());
    assert_eq!(m.log_mel(&pcm[..100]).1, 0);
}

/// Live diarization is the feature-level streaming loop run on audio as it arrives: same
/// predictions to the bit, however the audio is cut, and (apart from the final chunk's padding)
/// the same as NeMo's.
#[test]
fn live_matches_the_offline_streaming_loop_and_nemo() {
    let (Some(m), Some(golden)) = (model("sortformer-4spk-v2.1-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let t = golden.n_frames();
    for name in ["tiny", "tiny_nofifo", "default"] {
        let params = preset(&m, &golden, name);
        let (mel, n) = m.log_mel(&pcm);
        let want = m
            .new_stream(params.clone())
            .unwrap()
            .diarize_features_unpadded(&mel, n)
            .unwrap();
        assert_eq!(want.len(), t * 4, "{name}: one prediction per valid frame");
        for piece in [777usize, pcm.len()] {
            let got = run_live(&m, &pcm, params.clone(), piece);
            assert!(
                got.len() == want.len()
                    && got
                        .iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: live output differs from the offline loop (pieces of {piece})"
            );
        }
        // Against NeMo: the unpadded loop chunks the last frames differently from NeMo's padded one,
        // but every valid frame sees the same audio.
        let nemo = &golden.t(&format!("{name}.total_preds"))[..t * 4];
        let d = diff(&want, nemo);
        eprintln!("  {name}: live vs NeMo max|d| {:.3e}", d.max_abs);
        assert!(
            d.max_abs <= 5e-3,
            "{name}: live vs NeMo max|d| {:.3e}",
            d.max_abs
        );
    }
}

/// A chunk is released as soon as it and its lookahead exist, not before and not later.
#[test]
fn live_releases_a_chunk_when_its_lookahead_arrives() {
    let (Some(m), Some(golden)) = (model("sortformer-4spk-v2.1-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let params = preset(&m, &golden, "tiny"); // chunk 12 frames, 1 frame of lookahead
    let mut live = m.new_live(params).unwrap();
    assert_eq!(live.latency_frames(), 13);

    // 12 chunk frames + 1 lookahead frame = 104 mel frames; the 104th (index 103) needs audio
    // through sample 103 * 160 + 256 = 16736.
    let ready = 103 * 160 + 256;
    assert!(live.push_audio(&pcm[..ready - 1]).unwrap().is_empty());
    assert_eq!(live.frames_emitted(), 0);
    let first = live.push_audio(&pcm[ready - 1..ready]).unwrap();
    assert_eq!(first.len(), 12 * 4, "the first chunk is 12 frames");

    // The tail comes out at finish, and the totals are one prediction per valid frame.
    let rest: usize = live.push_audio(&pcm[ready..]).unwrap().len() + live.finish().unwrap().len();
    assert_eq!(live.frames_emitted(), golden.n_frames());
    assert_eq!((first.len() + rest) / 4, golden.n_frames());
    // `SortformerLive`'s own check, not the `MelStream` underneath it, which refuses too.
    let err = live.push_audio(&[0.0; 160]).unwrap_err().to_string();
    assert!(
        err.contains("SortformerLive: push_audio after finish"),
        "{err}"
    );
    assert!(live.finish().unwrap().is_empty(), "finish is idempotent");
}

/// A live stream holds a bounded window of mel, whatever its length: the buffers do not grow
/// with the audio. Loops the clip so the stream runs far longer than any window.
#[test]
fn live_buffers_stay_bounded() {
    let (Some(m), Some(golden)) = (model("sortformer-4spk-v2.1-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let params = preset(&m, &golden, "tiny");
    let (nm_window, ss) = (
        params.left_context * 8 + params.chunk_len * 8 + params.right_context * 8,
        8,
    );
    let mut live = m.new_live(params).unwrap();
    let mut mel = m.new_mel_stream();
    let mut peak_frames = 0;
    for _ in 0..4 {
        for piece in pcm.chunks(777) {
            live.push_audio(piece).unwrap();
            mel.push(piece).unwrap();
            peak_frames = peak_frames.max(live.buffered_frames());
            // One FFT window and one hop at most, however much has been pushed.
            assert!(
                mel.buffered_samples() <= 512 + 160 + 777,
                "{}",
                mel.buffered_samples()
            );
        }
    }
    assert!(live.frames_emitted() > 3 * golden.n_frames());
    // Left context + chunk + lookahead, plus a chunk that was still waiting.
    assert!(
        peak_frames <= nm_window + ss * 12,
        "held {peak_frames} mel frames after {} predictions",
        live.frames_emitted()
    );
}

/// A NaN or infinite sample is refused with nothing consumed: the stream carries on as if the
/// bad piece had never been pushed instead of poisoning every later frame.
#[test]
fn live_refuses_non_finite_pcm_without_consuming_it() {
    let Some(m) = model("sortformer-4spk-v2.1-f32.gguf") else {
        return;
    };
    let pcm = read_clip();
    let params = m.default_streaming().with_chunking(12, 1, 1, 20, 40, 12);
    let mut clean = m.new_live(params.clone()).unwrap();
    let mut dirty = m.new_live(params).unwrap();
    let (a, b) = pcm.split_at(8_000);
    let mut want = clean.push_audio(a).unwrap();
    want.extend(clean.push_audio(b).unwrap());
    want.extend(clean.finish().unwrap());

    let mut got = dirty.push_audio(a).unwrap();
    let mut bad = b.to_vec();
    bad[100] = f32::NAN;
    let err = dirty.push_audio(&bad).unwrap_err().to_string();
    assert!(err.contains("non-finite") && err.contains("8100"), "{err}");
    bad[100] = f32::INFINITY;
    assert!(dirty.push_audio(&bad).is_err());
    got.extend(dirty.push_audio(b).unwrap());
    got.extend(dirty.finish().unwrap());
    assert_eq!(got, want, "a refused piece must leave no trace");
}

/// Chunk lengths that are not a multiple of the `pad_to` padding put the first chunk of the
/// padded tail entirely past the last real frame; that chunk must come out as zero predictions
/// rather than a slice panic.
#[test]
fn odd_chunk_lengths_with_a_padded_tail_do_not_panic() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    // 305 mel frames: 305 % 16 = 1, so the padded length is 320.
    let pcm = &pcm[..48_800];
    let (mel, n) = m.log_mel(pcm);
    assert_eq!(n % 16, 1);
    let base = m.default_streaming();
    for (chunk, left) in [(1, 0), (3, 0), (2, 0), (1, 1)] {
        let params = base.with_chunking(chunk, left, 4, 8, 16, 4);
        let out = m
            .new_stream(params)
            .unwrap()
            .diarize_features(&mel, n)
            .unwrap_or_else(|e| panic!("chunk {chunk} left {left}: {e:#}"));
        assert_eq!(
            out.len(),
            n.div_ceil(16) * 16 / 8 * 4,
            "chunk {chunk} left {left}"
        );
    }
}

/// `step` refuses a left context that is not a whole number of encoder frames instead of
/// silently flooring it.
#[test]
fn step_refuses_a_misaligned_left_context() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mel, _) = m.log_mel(&pcm);
    let mut s = m.new_stream(m.default_streaming().clone()).unwrap();
    let err = s
        .step(&mel[..112 * 128], 112, 112, 11, 48)
        .unwrap_err()
        .to_string();
    assert!(err.contains("left_offset 11"), "{err}");
    assert!(s.step(&mel[..112 * 128], 112, 112, 8, 48).is_ok());
}

/// Overwrite the u32 value of metadata key `key` in a GGUF's bytes.
fn patch_u32(bytes: &mut [u8], key: &str, value: u32) {
    let k = key.as_bytes();
    let at = bytes
        .windows(k.len())
        .position(|w| w == k)
        .unwrap_or_else(|| panic!("{key} not in the file"))
        + k.len();
    assert_eq!(
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()),
        4,
        "{key} is not a u32"
    );
    bytes[at + 4..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// A GGUF whose metadata is hostile or foreign is an error naming the key, never a panic, a
/// divide by zero at first audio, or a front end that silently differs from NeMo's.
#[test]
fn loader_refuses_bad_metadata() {
    let Some(path) = local("sortformer-4spk-v2.1-q4_0.gguf") else {
        return;
    };
    let original = std::fs::read(path).unwrap();
    let load = |key: &str, value: u32| {
        let mut bytes = original.clone();
        patch_u32(&mut bytes, key, value);
        let g = std::sync::Arc::new(cera::gguf::GgufFile::from_bytes(bytes.into()).unwrap());
        SortformerModel::from_gguf(&g)
            .err()
            .map(|e| format!("{e:#}"))
    };
    for (key, value, want) in [
        ("sortformer.tf_head_count", 0, "tf_head_count 0"),
        ("sortformer.tf_head_count", 5, "tf_head_count 5"),
        ("clip.audio.attention.head_count", 0, "head_count 0"),
        ("clip.audio.attention.head_count", 3, "head_count 3"),
        ("clip.audio.block_count", 100_000, "layer counts"),
        ("sortformer.tf_layer_count", 0, "layer counts"),
        ("sortformer.mel.n_fft", 1024, "n_fft"),
        ("sortformer.max_speakers", 8, "max_speakers"),
        ("sortformer.conv_kernel_size", 5, "depthwise kernel"),
        ("clip.audio.block_count", 0, "layer counts"),
        ("sortformer.tf_layer_count", 100_000, "layer counts"),
        ("sortformer.mel.pad_to", 0, "pad_to"),
        ("sortformer.mel.pad_to", 100_000, "pad_to"),
        ("sortformer.mel.pad_to", u32::MAX, "pad_to"),
        ("clip.audio.embedding_length", 256, "embedding_length"),
        ("sortformer.stream.chunk_len", 0, "chunk_len must be > 0"),
        ("sortformer.stream.max_index", 5, "max_index"),
        ("sortformer.stream.chunk_left_context", 100_000, "exceeds"),
    ] {
        let err = load(key, value).unwrap_or_else(|| panic!("{key}={value} was accepted"));
        assert!(err.contains(want), "{key}={value}: {err}");
    }
}

/// Overwrite the f32 value (GGUF type 6) or a same-length string value (type 8) of metadata key
/// `key`.
fn patch_meta(bytes: &mut [u8], key: &str, value: &[u8], type_tag: u32) {
    let k = key.as_bytes();
    let at = bytes
        .windows(k.len())
        .position(|w| w == k)
        .unwrap_or_else(|| panic!("{key} not in the file"))
        + k.len();
    assert_eq!(
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()),
        type_tag,
        "{key}: wrong GGUF type"
    );
    let body = if type_tag == 8 { at + 4 + 8 } else { at + 4 };
    assert!(bytes[body..].len() > value.len());
    bytes[body..body + value.len()].copy_from_slice(value);
}

#[test]
fn loader_refuses_foreign_front_end_metadata() {
    let Some(path) = local("sortformer-4spk-v2.1-q4_0.gguf") else {
        return;
    };
    let original = std::fs::read(path).unwrap();
    let load = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut bytes = original.clone();
        edit(&mut bytes);
        let g = std::sync::Arc::new(cera::gguf::GgufFile::from_bytes(bytes.into()).unwrap());
        SortformerModel::from_gguf(&g)
            .err()
            .map(|e| format!("{e:#}"))
    };
    let u = |key: &'static str, v: u32| move |b: &mut Vec<u8>| patch_u32(b, key, v);
    for (key, v, want) in [
        ("sortformer.mel.win_length", 399, "win_length"),
        ("sortformer.mel.hop_length", 161, "hop_length"),
        ("sortformer.sample_rate", 8_000, "sample_rate"),
        ("sortformer.subsampling_factor", 4, "subsampling_factor"),
        ("sortformer.fc_d_model", 256, "fc_d_model"),
    ] {
        let err = load(&u(key, v)).unwrap_or_else(|| panic!("{key}={v} was accepted"));
        assert!(err.contains(want), "{key}={v}: {err}");
    }
    for (key, v, want) in [
        ("sortformer.mel.preemph", 0.9f32, "preemph"),
        ("sortformer.mel.mag_power", 1.0, "mag_power"),
        ("sortformer.mel.log_zero_guard", 1e-3, "log_zero_guard"),
    ] {
        let err = load(&|b| patch_meta(b, key, &v.to_le_bytes(), 6))
            .unwrap_or_else(|| panic!("{key}={v} was accepted"));
        assert!(err.contains(want), "{key}={v}: {err}");
    }
    let err = load(&|b| patch_meta(b, "sortformer.mel.normalize", b"PT", 8)).expect("accepted");
    assert!(err.contains("mel normalize"), "{err}");
    let err = load(&|b| patch_meta(b, "sortformer.tf_activation", b"gelu", 8)).expect("accepted");
    assert!(err.contains("tf_activation"), "{err}");
}

/// Overwrite dimension `dim` of tensor `name` in a GGUF's tensor-info table (the name, a u32
/// dimension count, then one u64 per dimension).
fn patch_tensor_dim(bytes: &mut [u8], name: &str, dim: usize, value: u64) {
    let mut needle = (name.len() as u64).to_le_bytes().to_vec();
    needle.extend_from_slice(name.as_bytes());
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap_or_else(|| panic!("{name} not in the file"))
        + needle.len();
    let n_dims = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    assert!(dim < n_dims, "{name} has {n_dims} dims");
    let p = at + 4 + dim * 8;
    bytes[p..p + 8].copy_from_slice(&value.to_le_bytes());
}

/// A tensor whose shape disagrees with the metadata is refused at load, naming where: the
/// kernels only `debug_assert` their shapes, so a release build would otherwise run a partial
/// bias or a mis-shaped matrix, or panic at the first audio.
#[test]
fn loader_refuses_inconsistent_tensor_shapes() {
    let Some(path) = local("sortformer-4spk-v2.1-q4_0.gguf") else {
        return;
    };
    let original = std::fs::read(path).unwrap();
    for (name, dim, value, want) in [
        ("a.blk.3.ffn_up.bias", 0, 2047, "FastConformer block 3"),
        ("a.blk.0.attn_q.bias", 0, 511, "FastConformer block 0"),
        ("a.blk.16.conv_pw2.weight", 1, 256, "FastConformer block 16"),
        ("sf.blk.5.attn_k.bias", 0, 191, "transformer layer 5"),
        ("sf.blk.2.ffn_up.weight", 1, 767, "transformer layer 2"),
        ("sf.head.out.weight", 1, 3, "speaker head"),
        ("sf.enc_proj.bias", 0, 191, "sf.enc_proj.bias"),
        (
            "a.pre_encode.out.weight",
            1,
            4096,
            "a.pre_encode.out.weight",
        ),
        ("a.pre_encode.out.bias", 0, 511, "a.pre_encode.out.bias"),
        ("a.conv1d.3.bias", 2, 255, "conv stem layer"),
    ] {
        let mut bytes = original.clone();
        patch_tensor_dim(&mut bytes, name, dim, value);
        let g = std::sync::Arc::new(cera::gguf::GgufFile::from_bytes(bytes.into()).unwrap());
        let err = SortformerModel::from_gguf(&g)
            .err()
            .unwrap_or_else(|| panic!("{name} dim {dim} = {value} was accepted"));
        assert!(format!("{err:#}").contains(want), "{name}: {err:#}");
    }
}

/// Every entry point that takes PCM refuses NaN and infinity, and a refused `step` leaves the
/// stream untouched: a later clean chunk gives what a stream that never saw the bad one gives.
#[test]
fn non_finite_input_is_refused_everywhere_and_leaves_no_trace() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let mut bad = pcm[..16_000].to_vec();
    bad[5_000] = f32::NAN;
    // The message, not just `is_err`: `diarize_streaming` would also fail later, at the first
    // `step`, so a bare `is_err` passes with the PCM check deleted.
    let want = "PCM sample at index 5000";
    assert!(
        m.diarize_offline(&bad)
            .unwrap_err()
            .to_string()
            .contains(want)
    );
    assert!(
        m.diarize_streaming(&bad, m.default_streaming().clone())
            .unwrap_err()
            .to_string()
            .contains(want)
    );

    let params = m.default_streaming().with_chunking(12, 1, 1, 20, 40, 12);
    let (mel, _) = m.log_mel(&pcm);
    let chunk = &mel[..100 * 128];
    let mut poisoned = mel[..100 * 128].to_vec();
    poisoned[60 * 128 + 3] = f32::INFINITY;
    let mut a = m.new_stream(params.clone()).unwrap();
    let err = a.step(&poisoned, 100, 100, 0, 8).unwrap_err().to_string();
    assert!(
        err.contains("non-finite") && err.contains("frame 60"),
        "{err}"
    );
    assert_eq!(
        (a.spkcache().len(), a.fifo().len(), a.n_sil_frames()),
        (0, 0, 0)
    );
    let mut b = m.new_stream(params).unwrap();
    assert_eq!(
        a.step(chunk, 100, 100, 0, 8).unwrap(),
        b.step(chunk, 100, 100, 0, 8).unwrap()
    );
}

/// A bare `MelStream` refuses audio after `finish`, not only through `SortformerLive`.
#[test]
fn a_bare_mel_stream_refuses_audio_after_finish() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let mut ms = m.new_mel_stream();
    ms.push(&pcm[..4_000]).unwrap();
    ms.finish();
    assert!(ms.push(&pcm[..160]).is_err(), "no audio after finish");
    assert!(ms.finish().is_empty(), "finish is idempotent");
}

/// Every argument check in `step` and `diarize_features` has its own failing input, asserted by
/// message: without the check the call would slice out of bounds or wrap an unsigned subtraction.
#[test]
fn step_refuses_inconsistent_arguments() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mel, _) = m.log_mel(&pcm);
    let feats = &mel[..112 * 128];
    let mut s = m.new_stream(m.default_streaming().clone()).unwrap();
    let refused = |r: anyhow::Result<Vec<f32>>, want: &str| {
        let err = r.unwrap_err().to_string();
        assert!(err.contains(want), "want `{want}`: {err}");
    };
    refused(s.step(&feats[..111 * 128], 112, 112, 0, 0), "feats is not");
    // 2^57 x 128 wraps to 0 in a usize: an empty slice must not pass for that many frames.
    refused(s.step(&[], 1usize << 57, 0, 0, 0), "feats is not");
    refused(s.step(feats, 112, 113, 0, 0), "valid_feat 113 > n_feat 112");
    // 112 frames are 14 encoder frames; 10 of left and 6 of right context leave none for the chunk.
    refused(s.step(feats, 112, 112, 80, 48), "shorter than its contexts");
    refused(s.diarize_features(&mel[..100 * 128], 101), "mel is not");
    // A bogus frame count whose padded length overflows is an error, not an overflow panic.
    refused(s.diarize_features(&[], usize::MAX), "cannot be padded");
    // A window past the offline bound is refused before the stem runs.
    let big = 8 * 7_600;
    refused(s.step(&vec![0.0; big * 128], big, big, 0, 0), "step window");
    assert_eq!(
        (s.spkcache().len(), s.fifo().len(), s.n_sil_frames()),
        (0, 0, 0)
    );
}

/// Every tensor of the loader's inputs, one dimension at a time and one element short, is refused:
/// pins each shape check and each length in the per-layer tables, not just the ten above. Only
/// layer 0 of each stack is probed (the checks are per layer). The f32 file is used so that a
/// shrunk matrix is still a readable tensor and only the loader's own checks can refuse it.
#[test]
fn loader_refuses_every_tensor_one_element_short() {
    let Some(path) = local("sortformer-4spk-v2.1-f32.gguf") else {
        return;
    };
    let original = std::fs::read(path).unwrap();
    let g = cera::gguf::GgufFile::from_bytes(original.clone().into()).unwrap();
    let mut names: Vec<&String> = g
        .tensors
        .keys()
        .filter(|n| {
            let blk = n.starts_with("a.blk.") || n.starts_with("sf.blk.");
            !blk || n.starts_with("a.blk.0.") || n.starts_with("sf.blk.0.")
        })
        .collect();
    names.sort();
    let (mut probed, mut accepted) = (0, Vec::new());
    for name in names {
        let shape = g.get_tensor(name).unwrap().shape().to_vec();
        for (dim, &v) in shape.iter().enumerate().filter(|&(_, &v)| v > 1) {
            let mut bytes = original.clone();
            patch_tensor_dim(&mut bytes, name, dim, v as u64 - 1);
            probed += 1;
            let g = std::sync::Arc::new(cera::gguf::GgufFile::from_bytes(bytes.into()).unwrap());
            if SortformerModel::from_gguf(&g).is_ok() {
                accepted.push(format!("{name} dim {dim}"));
            }
        }
    }
    assert!(probed > 100, "only {probed} probes");
    assert!(accepted.is_empty(), "the loader accepted {accepted:?}");
}

/// Finite but absurd samples (reinterpreted bytes, near `f32::MAX`) overflow the mel to infinity
/// just as a NaN would, so they are refused the same way: whole, leaving no trace.
#[test]
fn absurdly_large_pcm_is_refused_and_a_live_stream_carries_on() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let mut bad = pcm[..16_000].to_vec();
    for (i, x) in bad[5_000..5_100].iter_mut().enumerate() {
        *x = if i % 2 == 0 { 3e38 } else { -3e38 };
    }
    assert!(m.diarize_offline(&bad).is_err());

    let params = m.default_streaming().with_chunking(12, 1, 1, 20, 40, 12);
    let mut clean = m.new_live(params.clone()).unwrap();
    let mut dirty = m.new_live(params).unwrap();
    let mut want = Vec::new();
    let mut got = Vec::new();
    for piece in pcm.chunks(4_000) {
        want.extend(clean.push_audio(piece).unwrap());
        let mut with_garbage = piece.to_vec();
        with_garbage[0] = f32::MAX;
        assert!(dirty.push_audio(&with_garbage).is_err());
        got.extend(dirty.push_audio(piece).unwrap());
    }
    want.extend(clean.finish().unwrap());
    got.extend(dirty.finish().unwrap());
    assert_eq!(got, want, "refused pushes must leave no trace");
}

/// A bad value in a late chunk of a whole clip is refused before any chunk runs, naming its
/// frame in the clip, and the stream is left untouched.
#[test]
fn a_bad_mel_value_late_in_a_clip_is_refused_before_any_chunk_runs() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mut mel, n) = m.log_mel(&pcm);
    mel[1_200 * 128 + 7] = f32::NAN;
    let mut s = m
        .new_stream(m.default_streaming().with_chunking(12, 1, 1, 20, 40, 12))
        .unwrap();
    let err = s.diarize_features(&mel, n).unwrap_err().to_string();
    assert!(err.contains("frame 1200"), "{err}");
    assert_eq!(
        (s.spkcache().len(), s.fifo().len(), s.n_sil_frames()),
        (0, 0, 0)
    );
}

/// Offline attention is quadratic in the clip, so a clip past the documented limit is an error,
/// not an allocation that aborts the process.
#[test]
fn offline_diarization_refuses_a_clip_past_its_limit() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let err = m
        .diarize_offline(&vec![0.0; 16_000 * 700])
        .unwrap_err()
        .to_string();
    assert!(err.contains("offline"), "{err}");
}

/// Overwrite the storage type field of tensor `name` in a GGUF's tensor-info table (after the
/// name, the u32 dimension count and one u64 per dimension).
fn patch_tensor_type(bytes: &mut [u8], name: &str, ty: u32) {
    let mut needle = (name.len() as u64).to_le_bytes().to_vec();
    needle.extend_from_slice(name.as_bytes());
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap_or_else(|| panic!("{name} not in the file"))
        + needle.len();
    let n_dims = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let p = at + 4 + n_dims * 8;
    bytes[p..p + 4].copy_from_slice(&ty.to_le_bytes());
}

/// A matrix stored as a type with no matmul kernel would run as zeros in a release build
/// (`gemv_dispatch` only debug-asserts), so the loader refuses it. GGUF type 26 is I32, the same
/// element width as F32, so the file stays well formed.
#[test]
fn loader_refuses_matrices_with_no_matmul_kernel() {
    let Some(path) = local("sortformer-4spk-v2.1-f32.gguf") else {
        return;
    };
    let original = std::fs::read(path).unwrap();
    let g = cera::gguf::GgufFile::from_bytes(original.clone().into()).unwrap();
    // Every matrix the loader hands to a matmul kernel: the rank-2 weights of the stem output,
    // the head, and layer 0 of each stack (the checks are per layer). The depthwise kernel is
    // rank 2 too but is not a matmul.
    let mut names: Vec<String> = g
        .tensors
        .keys()
        .filter(|n| {
            let blk = n.starts_with("a.blk.") || n.starts_with("sf.blk.");
            (!blk || n.starts_with("a.blk.0.") || n.starts_with("sf.blk.0."))
                && n.ends_with(".weight")
                && !n.contains("conv_dw")
                && !n.starts_with("a.conv1d")
                && g.get_tensor(n).unwrap().shape().len() == 2
        })
        .cloned()
        .collect();
    names.sort();
    assert!(
        names.len() >= 20,
        "only {} matrices: {names:?}",
        names.len()
    );
    for name in &names {
        let mut bytes = original.clone();
        patch_tensor_type(&mut bytes, name, 26);
        let g = std::sync::Arc::new(cera::gguf::GgufFile::from_bytes(bytes.into()).unwrap());
        let err = SortformerModel::from_gguf(&g)
            .err()
            .unwrap_or_else(|| panic!("{name} stored as I32 was accepted"));
        assert!(
            format!("{err:#}").contains("no matmul kernel"),
            "{name}: {err:#}"
        );
    }
}

/// A finite mel value large enough to overflow the stem is refused like a NaN: whole, before
/// anything is consumed, and a later clean chunk matches a stream that never saw it.
#[test]
fn a_huge_finite_mel_value_is_refused_and_leaves_no_trace() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mel, n) = m.log_mel(&pcm);
    let params = m.default_streaming().with_chunking(12, 1, 1, 20, 40, 12);
    let chunk = &mel[..100 * 128];
    let mut poisoned = chunk.to_vec();
    poisoned[60 * 128 + 3] = 1e9;
    let mut a = m.new_stream(params.clone()).unwrap();
    let err = a.step(&poisoned, 100, 100, 0, 8).unwrap_err().to_string();
    assert!(
        err.contains("out-of-range") && err.contains("frame 60"),
        "{err}"
    );
    assert_eq!(
        (a.spkcache().len(), a.fifo().len(), a.n_sil_frames()),
        (0, 0, 0)
    );
    let mut b = m.new_stream(params.clone()).unwrap();
    assert_eq!(
        a.step(chunk, 100, 100, 0, 8).unwrap(),
        b.step(chunk, 100, 100, 0, 8).unwrap()
    );
    let mut late = mel.clone();
    late[1_200 * 128] = -3e38;
    let mut s = m.new_stream(params).unwrap();
    let err = s.diarize_features(&late, n).unwrap_err().to_string();
    assert!(
        err.contains("out-of-range") && err.contains("frame 1200"),
        "{err}"
    );
    assert_eq!(
        (s.spkcache().len(), s.fifo().len(), s.n_sil_frames()),
        (0, 0, 0)
    );
}

/// `validate` ties `max_index` to the configured chunk, but `step` takes whatever the caller
/// feeds: a longer step could put a real flat index at or past a tight `max_index`, where it
/// would read as a disabled slot, so `step` refuses it.
#[test]
fn step_refuses_a_window_its_max_index_cannot_index() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mel, _) = m.log_mel(&pcm);
    let base = m.default_streaming().with_chunking(6, 1, 7, 20, 24, 8);
    let sil = base.sil_frames_per_spk;
    let tight = cera::model::sortformer::StreamingParams {
        max_index: 4 * (24 + 20 + 6 + sil),
        ..base.clone()
    };
    // Exactly enough for the configured chunk: accepted.
    assert!(m.new_stream(tight.clone()).is_ok());
    // A 150-frame step needs far more.
    let mut s = m.new_stream(tight).unwrap();
    let err = s
        .step(&mel[..1_200 * 128], 1_200, 1_200, 0, 0)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("max_index") && err.contains("150-frame"),
        "{err}"
    );
    assert_eq!(s.fifo().len(), 0);
    // The default max_index covers it.
    let mut ok = m.new_stream(base).unwrap();
    assert!(ok.step(&mel[..1_200 * 128], 1_200, 1_200, 0, 0).is_ok());
}

/// `validate` ties `max_index` to the configured chunk, and `step` re-checks it against what it
/// is really handed, counting only the rows compression can see (not the contexts): a config at
/// exactly the `validate` minimum must stream a whole clip, and give what a roomy `max_index`
/// gives, rather than fail once the cache and FIFO have filled.
#[test]
fn a_tight_but_valid_max_index_streams_a_whole_clip() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let pcm = read_clip();
    let roomy = m.default_streaming().with_chunking(6, 1, 7, 20, 24, 8);
    let tight = cera::model::sortformer::StreamingParams {
        max_index: 4 * (24 + 20 + 6 + roomy.sil_frames_per_spk),
        ..roomy.clone()
    };
    assert_eq!(
        run_live(&m, &pcm, tight, 16_000),
        run_live(&m, &pcm, roomy, 16_000)
    );
}

/// `predict` is a public driver with a caller-chosen size: past the attention window the other
/// entry points enforce, it panics with a message instead of allocating quadratic memory.
#[test]
fn predict_refuses_a_window_past_the_attention_bound() {
    let Some(m) = model("sortformer-4spk-v2.1-q8_0.gguf") else {
        return;
    };
    let t = 7_501;
    let emb = vec![0.0f32; t * 512];
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.predict(&emb, t)))
        .expect_err("a 7501-frame window was accepted");
    let msg = panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains("exceeds") && msg.contains("7500"), "{msg}");
}

/// The service-shaped path: PCM goes in as it arrives, each utterance is registered when its
/// text would be ready, and it comes back labeled once the diarizer covers it, which for a
/// short-latency preset is long before the clip ends. The spans sit inside the clip's three
/// voices (slots 0, 1, 2 in arrival order).
#[test]
fn live_diarizer_labels_utterances_as_the_audio_arrives() {
    let (Some(m), Some(golden)) = (model("sortformer-4spk-v2.1-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let params = preset(&m, &golden, "tiny"); // 12-frame chunks, 1 frame of lookahead
    let mut d = cera::live_diarizer::LiveDiarizer::new(
        &m,
        params,
        cera::speaker_labeler::SpeakerLabelerConfig::default(),
    )
    .unwrap();
    // (id, start ms, end ms): voices A, B, C, C. Slots are arrival ordered, but which slot the
    // third voice lands in depends on the preset's cache (NeMo's own output for this preset puts
    // it in slot 3), so only the pattern is pinned.
    let utterances = [
        (10u64, 700.0, 3_800.0),
        (11, 4_700.0, 7_800.0),
        (12, 8_700.0, 10_300.0),
        (13, 13_700.0, 15_200.0),
    ];
    let mut registered = 0;
    let mut released: Vec<(u64, usize, f64)> = Vec::new(); // (id, slot, audio ms pushed)
    let mut pushed = 0usize;
    for piece in pcm.chunks(1_600) {
        d.push_audio(piece).unwrap();
        pushed += piece.len();
        let now_ms = pushed as f64 / 16.0;
        // Whisper would hand the text over when the utterance ends.
        while registered < utterances.len() && utterances[registered].2 <= now_ms {
            let (id, a, b) = utterances[registered];
            d.add_utterance(id, a, b);
            registered += 1;
        }
        for u in d.poll() {
            let label = u
                .label
                .as_ref()
                .unwrap_or_else(|| panic!("{u:?} unlabeled"));
            assert!(!u.dropped);
            released.push((u.id, label.speaker, now_ms));
        }
    }
    for u in d.finish().unwrap() {
        let label = u
            .label
            .as_ref()
            .unwrap_or_else(|| panic!("{u:?} unlabeled"));
        released.push((u.id, label.speaker, f64::INFINITY));
    }
    assert_eq!(released.len(), utterances.len(), "{released:?}");
    let slot = |id: u64| released.iter().find(|r| r.0 == id).unwrap().1;
    assert_eq!((slot(10), slot(11)), (0, 1), "{released:?}");
    assert_eq!(
        slot(12),
        slot(13),
        "the same voice keeps its slot: {released:?}"
    );
    assert!(
        ![0, 1].contains(&slot(12)),
        "a new voice takes a new slot: {released:?}"
    );
    for (id, _, at_ms) in &released {
        // Labeled after it ended (and, for all but the last, not at the end of the clip).
        let end = utterances.iter().find(|u| u.0 == *id).unwrap().2;
        assert!(*at_ms >= end, "utterance {id} released before it ended");
    }
    // The first utterance does not wait for the end of the clip.
    let first = released.iter().find(|r| r.0 == 10).unwrap();
    assert!(
        first.2 < 8_000.0,
        "first utterance released at {} ms",
        first.2
    );
}

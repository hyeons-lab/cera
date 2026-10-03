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
//! The CPU model is not batched, so these are slow in a debug build: run them with `--release`.
//! The 33-step low-latency preset is slower still and runs only with `SORTFORMER_FULL=1`.

use std::collections::HashMap;
use std::path::PathBuf;

use cera::convert::safetensors::SafeTensorsHeader;
use cera::model::sortformer::{SortformerModel, StreamingParams};

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".leap/models/sortformer")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sortformer")
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
        let got = m.diarize_offline(&pcm);
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

#[test]
fn streaming_default_preset() {
    run_streaming("sortformer-4spk-v2.1-f32.gguf", "default", true, 2e-3, 2e-3);
}

#[test]
fn streaming_low_latency_preset() {
    if std::env::var("SORTFORMER_FULL").as_deref() != Ok("1") {
        eprintln!("skipping the 33-step low-latency preset; set SORTFORMER_FULL=1 to run it");
        return;
    }
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
    assert!(
        live.push_audio(&[0.0; 160]).is_err(),
        "no audio after finish"
    );
    assert!(live.finish().unwrap().is_empty(), "finish is idempotent");
}

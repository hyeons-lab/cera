//! Parity of cera's CPU Nemotron-3-Diarization against NVIDIA's NeMo implementation.
//!
//! The reference is `cera/tests/fixtures/nemotron3/golden.json` (committed) plus
//! `golden.safetensors` (every intermediate tensor and per-step streaming state, ~19 MB, NOT
//! committed). Both are produced by `scripts/nemotron3_diarization/gen_golden.py` from the
//! committed `clip.wav` (shared with the 4spk fixtures); the models by
//! `scripts/nemotron3_diarization/convert.py`. Everything model-backed lives in
//! `~/.leap/models/nemotron3-diarization/` and skips with a message when absent
//! (`CERA_REQUIRE_MODEL=1` turns a skip into a failure).
//!
//! Three layers, each isolating a different kind of bug:
//!
//! * `stages_teacher_forced`: every stage is fed NeMo's *own* input for that stage and compared
//!   with NeMo's output, so the first stage that differs names the bug (mel, stacking, input
//!   norm, an encoder block, final norm, `proj`, the upsampler, the classifier).
//! * `offline_end_to_end`: PCM to sigmoids with nothing from NeMo but the answer.
//! * `streaming_*`: the streaming loop for five presets, comparing the predictions and, for
//!   the presets that record state, the speaker cache, FIFO and flags after every step.
//!
//! The file also pins the live path (`Nemotron3Live`, `MelStream`), the loader's refusal of
//! hostile files, and the argument checks of `step`.
//!
//! The CPU model is not batched, so these are slow in a debug build: run them with `--release`.
//! The 65-step ultra-low-latency preset is slower still and is `#[ignore]`d: run it with `--ignored`.

#![cfg(feature = "mmap")] // `Nemotron3Model::from_file`

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use cera::convert::safetensors::SafeTensorsHeader;
use cera::model::nemotron3_diarization::{Nemotron3Model, StreamingParams, enc_frames};

/// The committed fixtures. `NEMOTRON3_FIXTURES` overrides the compile-time path so a test binary
/// cross-built for a device (where the source tree does not exist) can run against pushed copies.
fn fixtures_dir() -> PathBuf {
    match std::env::var_os("NEMOTRON3_FIXTURES") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nemotron3"),
    }
}

/// The committed clip, shared with the 4spk fixtures (an override dir is expected to carry a
/// copy next to this model's golden files).
fn clip_path() -> PathBuf {
    match std::env::var_os("NEMOTRON3_FIXTURES") {
        Some(dir) => PathBuf::from(dir).join("clip.wav"),
        None => {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sortformer/clip.wav")
        }
    }
}

/// A model-backed file, or `None` to skip.
fn local(rel: &str) -> Option<PathBuf> {
    common::local_file("NEMOTRON3_MODELS_DIR", "nemotron3-diarization", rel)
}

fn model(file: &str) -> Option<Nemotron3Model> {
    let path = local(file)?;
    Some(Nemotron3Model::from_file(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display())))
}

/// 16-bit mono PCM WAV to f32, the way `soundfile.read(dtype="float32")` does it.
fn read_clip() -> Vec<f32> {
    common::read_wav_f32(&clip_path())
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

    fn n_mel_frames(&self) -> usize {
        self.json["n_mel_frames"].as_u64().unwrap() as usize
    }

    fn n_enc_frames(&self) -> usize {
        self.json["n_enc_frames"].as_u64().unwrap() as usize
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
fn loader_reads_config_and_streaming_defaults() {
    let Some(m) = model("nemotron3-diarization-f32.gguf") else {
        return;
    };
    let c = m.config();
    assert_eq!(c.n_layer, 31);
    assert_eq!(c.n_embd, 512);
    assert_eq!(c.n_ff, 2048);
    assert_eq!(c.n_head, 8);
    assert_eq!(c.n_mel_bins, 128);
    assert_eq!(c.tf_d, 192);
    assert_eq!(c.n_spk, 8);
    assert_eq!(c.subsampling, 8);
    assert_eq!(c.rope_theta, 10_000.0);
    let s = m.default_streaming();
    assert_eq!(
        (
            s.chunk_len,
            s.right_context,
            s.fifo_len,
            s.spkcache_len,
            s.update_period
        ),
        (264, 0, 0, 264, 264)
    );
    assert_eq!(s.sil_frames_per_spk, 1);
    assert_eq!(s.pred_score_threshold, 0.25);
    assert_eq!(s.max_index, 99999);
    // The card presets are runtime overrides of these.
    assert_eq!(s.low_latency().window_frames(), 9 + 4 + 264 + 264);
    assert_eq!(s.very_low_latency().window_frames(), 6 + 2 + 264 + 264);
    assert_eq!(s.ultra_low_latency().window_frames(), 3 + 1 + 264 + 264);
}

/// The committed golden stands alone: schema keys, frame counts, per-preset prediction
/// shapes, and preset chunkings that pass `validate` over the checkpoint's score
/// hyper-parameters. Hermetic (no models dir, no safetensors): the one parity test that runs
/// in CI, so a corrupted or half-regenerated `golden.json` fails there instead of skipping
/// every consumer green.
#[test]
fn golden_json_is_self_consistent() {
    let json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("golden.json")).expect("committed golden.json"),
    )
    .expect("parse golden.json");
    for key in [
        "meta",
        "mel_len",
        "n_mel_frames",
        "n_enc_frames",
        "preds_offline",
        "presets",
        "stages",
    ] {
        assert!(json.get(key).is_some(), "golden.json lacks `{key}`");
    }
    let n_mel = json["n_mel_frames"].as_u64().unwrap() as usize;
    assert_eq!(json["mel_len"].as_u64().unwrap() as usize, n_mel);
    assert_eq!(
        json["n_enc_frames"].as_u64().unwrap() as usize,
        enc_frames(n_mel),
        "encoder frame count"
    );
    let rows = |v: &serde_json::Value| {
        let rows = v.as_array().unwrap();
        assert!(
            rows.iter()
                .all(|r| r.as_array().is_some_and(|r| r.len() == 8)),
            "every prediction row holds 8 speaker slots"
        );
        rows.len()
    };
    assert_eq!(rows(&json["preds_offline"]), n_mel);
    let sm = &json["meta"]["streaming_modules"];
    let f = |k: &str| sm[k].as_f64().unwrap() as f32;
    let u = |k: &str| sm[k].as_u64().unwrap() as usize;
    for name in [
        "default",
        "low_latency",
        "tiny",
        "tiny_nofifo",
        "ultra_low_latency",
    ] {
        let preset = &json["presets"][name];
        assert_eq!(rows(&preset["total_preds"]), n_mel, "{name} frames");
        let p = &preset["params"];
        let g = |k: &str| p[k].as_u64().unwrap() as usize;
        StreamingParams {
            chunk_len: g("chunk_len"),
            right_context: g("chunk_right_context"),
            fifo_len: g("fifo_len"),
            spkcache_len: g("spkcache_len"),
            update_period: g("spkcache_update_period"),
            sil_frames_per_spk: u("spkcache_sil_frames_per_spk"),
            pred_score_threshold: f("pred_score_threshold"),
            scores_boost_latest: f("scores_boost_latest"),
            sil_threshold: f("sil_threshold"),
            strong_boost_rate: f("strong_boost_rate"),
            weak_boost_rate: f("weak_boost_rate"),
            min_pos_scores_rate: f("min_pos_scores_rate"),
            max_index: u("max_index"),
        }
        .validate()
        .unwrap_or_else(|e| panic!("{name}: {e:#}"));
    }
}

#[test]
fn loader_refuses_other_architectures() {
    // A 4spk Sortformer GGUF must fail fast, naming the architecture mismatch (skips when the
    // old models are absent; this pins the failure, not the old model).
    let Some(path) = common::local_file(
        "SORTFORMER_MODELS_DIR",
        "sortformer",
        "sortformer-4spk-v2.1-f32.gguf",
    ) else {
        return;
    };
    match Nemotron3Model::from_file(&path) {
        Ok(_) => panic!("{} loaded as Nemotron-3", path.display()),
        Err(err) => assert!(
            format!("{err:#}").contains("not a Nemotron-3 GGUF"),
            "unexpected error: {err:#}"
        ),
    }
}

#[test]
fn stages_teacher_forced() {
    let (Some(m), Some(golden)) = (model("nemotron3-diarization-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let n_mel = golden.n_mel_frames();
    let t = golden.n_enc_frames();
    assert_eq!(t, enc_frames(n_mel), "encoder frame count");

    // Front end from PCM.
    let (mel, n) = m.log_mel(&pcm);
    assert_eq!(n, n_mel, "mel frame count");
    check("mel", &mel, &mel_time_major(&golden, n), 2e-3, 0.99999);

    // Embedder from NeMo's mel (same tolerance class as the 4spk stem: f32 summation
    // order in the frontend matmul). The clip mel pads to `pad_to` first, like NeMo's front
    // end; tensors cover the padded total, pad groups included.
    let mut nemo_mel = mel_time_major(&golden, n_mel);
    let feat_len = n_mel.div_ceil(16) * 16;
    nemo_mel.resize(feat_len * 128, 0.0);
    let (stacked, total) = m.embed(&nemo_mel, feat_len);
    assert_eq!(total, golden.shape("stacked")[0], "padded group count");
    check("stacked", &stacked, golden.t("stacked"), 2e-3, 0.99999);

    // Everything after the embedder from NeMo's stacked embeddings.
    let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
    let preds = m.predict_with_taps(golden.t("stacked"), n_mel, &mut |name, v| {
        taps.push((name.to_string(), v.to_vec()));
    });
    // The tap set itself: a deleted `tap()` emission must fail here, not silently shrink the
    // loop below (which only iterates what was produced).
    let mut expected = vec!["input_norm".to_string()];
    expected.extend((0..31).map(|i| format!("enc.layer{i}")));
    expected.extend(["final_norm", "enc_proj", "upsampled", "logits"].map(str::to_string));
    assert_eq!(
        taps.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        expected,
        "tap set"
    );
    for (name, got) in &taps {
        let want = golden.t(name);
        // Tolerance scales with the activation size: the residual stream is large, so an
        // absolute bound on it would be either vacuous or flaky.
        let scale = want.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1.0);
        check(name, got, want, 2e-4 * scale, 0.999999);
    }
    check("preds", &preds, golden.t("preds_offline"), 2e-4, 0.99999);
}

#[test]
fn offline_end_to_end() {
    let Some(golden) = Golden::load() else { return };
    let pcm = read_clip();
    let n_mel = golden.n_mel_frames();
    let want = &golden.t("preds_offline")[..n_mel * 8];

    // (file, max |d| on the sigmoids, decisions allowed to flip at 0.5)
    let mut ran = 0;
    for (file, tol, flips) in [
        ("nemotron3-diarization-f32.gguf", 2e-3f32, 0usize),
        ("nemotron3-diarization-q8_0.gguf", 5e-2, 3),
    ] {
        let Some(m) = model(file) else { continue };
        ran += 1;
        let got = m.diarize_offline(&pcm).unwrap();
        assert_eq!(got.len(), n_mel * 8, "{file}");
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
    assert!(
        ran > 0,
        "no GGUF present; the tensors loaded but nothing was compared"
    );
}

/// The preset's parameters from the golden JSON, over the checkpoint's score hyper-parameters.
fn preset(m: &Nemotron3Model, golden: &Golden, name: &str) -> StreamingParams {
    let p = &golden.json["presets"][name]["params"];
    let get = |k: &str| p[k].as_u64().unwrap() as usize;
    assert_eq!(get("chunk_left_context"), 0, "the port has no left context");
    m.default_streaming().with_chunking(
        get("chunk_len"),
        get("chunk_right_context"),
        get("fifo_len"),
        get("spkcache_len"),
        get("spkcache_update_period"),
    )
}

/// Run one preset and compare with NeMo, predictions and — for the presets that record it —
/// the speaker cache, FIFO and flags after every step, including the last: unlike the 4spk
/// port this one computes the padded groups, so the last step's state is comparable too.
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
    let (mut compressed_steps, mut max_sil_slots) = (0usize, 0usize);
    let got = stream
        .diarize_features_with(&mel, n, &mut |i, s, chunk_preds| {
            let bounds = steps[i]["chunk_frames"].as_array().unwrap();
            let (lo, hi) = (
                bounds[0].as_u64().unwrap() as usize,
                bounds[1].as_u64().unwrap() as usize,
            );
            assert_eq!(
                chunk_preds.len(),
                (hi - lo) * 8,
                "step {i}: chunk frame count"
            );
            assert_eq!(
                steps[i]["left_offset"].as_u64().unwrap(),
                0,
                "step {i}: the port has no left context"
            );
            if !(state && recorded) {
                return;
            }
            // NeMo keeps a speaker permutation and a silence profile in eval too, but only as
            // untouched initial state (permutation is train-only, silence is learned): pin that.
            assert!(
                steps[i]["spk_perm"].is_null(),
                "step {i}: NeMo set a speaker permutation"
            );
            assert_eq!(
                s.spkcache_compressed(),
                steps[i]["spkcache_compressed"].as_bool().unwrap(),
                "step {i}: compression flag"
            );
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
            compressed_steps += usize::from(s.spkcache_compressed());
            // Disabled cache slots predict exact zeros (torch's `where` with 0.0); a live
            // sigmoid is never exactly 0, so all-zero rows count the silence slots.
            max_sil_slots = max_sil_slots.max(
                s.spkcache_preds()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .filter(|r| r.iter().all(|&p| p == 0.0))
                    .count(),
            );
            cmp("spkcache", s.spkcache(), key("spkcache"));
            cmp("spkcache_preds", s.spkcache_preds(), key("spkcache_preds"));
            cmp("fifo", s.fifo(), key("fifo"));
            cmp("fifo_preds", s.fifo_preds(), key("fifo_preds"));
            // NeMo never maintains lengths on the sync path (the tensor row counts, compared
            // above, are the lengths); pin that they stay `None`.
            assert!(
                steps[i]["spkcache_lengths"].is_null() && steps[i]["fifo_lengths"].is_null(),
                "step {i}: NeMo set lengths on the sync path"
            );
            assert!(
                golden.t(&key("mean_sil_emb")).iter().all(|&v| v == 0.0),
                "step {i}: NeMo touched the silence profile"
            );
            assert_eq!(
                golden.t(&key("n_sil_frames"))[0] as usize,
                0,
                "step {i}: NeMo counted silence frames"
            );
        })
        .expect("streaming");
    if state {
        eprintln!(
            "  worst cache/FIFO max|d| over steps: {worst_state:.3e} \
             ({compressed_steps} steps with a compressed cache, {max_sil_slots} silence slots max)"
        );
        if name.starts_with("tiny") {
            // The point of these presets: without compression and silence slots in play, the
            // state comparison above would pass without touching the code worth testing.
            assert!(
                compressed_steps > 0,
                "{name}: the speaker cache never compressed"
            );
            assert!(
                max_sil_slots > 0,
                "{name}: no silence slot ever entered the cache"
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
    run_streaming("nemotron3-diarization-f32.gguf", "tiny", true, 5e-3, 5e-3);
}

#[test]
fn streaming_tiny_nofifo() {
    run_streaming(
        "nemotron3-diarization-f32.gguf",
        "tiny_nofifo",
        true,
        5e-3,
        5e-3,
    );
}

/// Predictions only in effect: one step, so no step has a compressed cache to compare. The
/// `tiny*` presets carry the cache and FIFO coverage.
#[test]
fn streaming_default_preset() {
    run_streaming(
        "nemotron3-diarization-f32.gguf",
        "default",
        true,
        2e-3,
        2e-3,
    );
}

#[test]
fn streaming_low_latency_preset() {
    run_streaming(
        "nemotron3-diarization-f32.gguf",
        "low_latency",
        false,
        5e-3,
        5e-3,
    );
}

#[test]
#[ignore = "65 steps, several minutes in release; run with --ignored"]
fn streaming_ultra_low_latency_preset() {
    run_streaming(
        "nemotron3-diarization-f32.gguf",
        "ultra_low_latency",
        false,
        5e-3,
        5e-3,
    );
}

/// `step` takes the lookahead in mel frames like NeMo's loader: it rounds UP to whole groups
/// (a 12-frame lookahead is 2 groups, not 1). A padded tail yields zero predictions for the
/// sub-frames past the valid groups. The parity presets only ever see lookaheads that are
/// multiples of 8, so this is where rounding shows.
#[test]
fn step_bookkeeping_follows_nemo() {
    let Some(m) = model("nemotron3-diarization-f32.gguf") else {
        return;
    };
    let pcm = read_clip();
    let (mel, _) = m.log_mel(&pcm);
    let params = m.default_streaming().clone();

    // 100 mel frames -> 13 stacked groups; 12 frames of lookahead -> ceil(12 / 8) = 2.
    let mut s = m.new_stream(params.clone()).unwrap();
    let preds = s.step(&mel[..100 * 128], 100, 100, 12).unwrap();
    assert_eq!(preds.len(), (13 - 2) * 8 * 8, "right context rounds up");

    // 112 frames padded, 100 of them audio: 14 groups, the last partly past the end.
    let mut padded = mel[..100 * 128].to_vec();
    padded.resize(112 * 128, 0.0);
    let mut s = m.new_stream(params).unwrap();
    let preds = s.step(&padded, 112, 100, 0).unwrap();
    assert_eq!(preds.len(), 14 * 8 * 8);
    // 100 valid mel frames are 13 valid groups: sub-frames past 13 * 8 read zero, like NeMo's
    // masked output, while the partial group's own pad sub-frames stay live.
    assert!(
        preds[13 * 8 * 8..].iter().all(|&p| p == 0.0),
        "sub-frames past the valid groups must be zero"
    );
}

fn run_live(m: &Nemotron3Model, pcm: &[f32], params: StreamingParams, piece: usize) -> Vec<f32> {
    let mut live = m.new_live(params).unwrap();
    let mut out = Vec::new();
    for part in pcm.chunks(piece) {
        out.extend(live.push_audio(part).unwrap());
    }
    out.extend(live.finish().unwrap());
    assert_eq!(live.frames_emitted() * 8, out.len());
    out
}

/// The incremental mel must be the whole-clip mel exactly, for any way of cutting the audio:
/// every frame is the same arithmetic on the same samples.
#[test]
fn mel_stream_is_bit_identical_to_the_whole_clip_mel() {
    let Some(m) = model("nemotron3-diarization-f32.gguf") else {
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
    let (Some(m), Some(golden)) = (model("nemotron3-diarization-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let n_mel = golden.n_mel_frames();
    for name in ["tiny", "tiny_nofifo", "default"] {
        let params = preset(&m, &golden, name);
        let (mel, n) = m.log_mel(&pcm);
        let want = m
            .new_stream(params.clone())
            .unwrap()
            .diarize_features_unpadded(&mel, n)
            .unwrap();
        assert!(
            want.len() >= n_mel * 8,
            "{name}: one prediction per valid frame"
        );
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
        // Against NeMo: the unpadded loop chunks the last frames differently from NeMo's
        // padded one. Every valid frame but the last sees the same audio; the last valid
        // sub-frame comes from a different step (unpadded: the previous step's lookahead
        // group, padded: the last step's chunk), and NeMo itself differs there by 3.4e-2
        // between its own padded and unpadded runs. The padded streaming tests above compare
        // that frame, so only it is excluded here.
        let nemo = &golden.t(&format!("{name}.total_preds"))[..(n_mel - 1) * 8];
        let d = diff(&want[..(n_mel - 1) * 8], nemo);
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
    let (Some(m), Some(golden)) = (model("nemotron3-diarization-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let params = preset(&m, &golden, "tiny"); // chunk 6 groups, 2 groups of lookahead
    let mut live = m.new_live(params).unwrap();
    assert_eq!(live.latency_frames(), 8);

    // 6 chunk groups + 2 lookahead groups = 64 mel frames; the 64th (index 63) needs audio
    // through sample 63 * 160 + 256 = 10336.
    let ready = 63 * 160 + 256;
    assert!(live.push_audio(&pcm[..ready - 1]).unwrap().is_empty());
    assert_eq!(live.frames_emitted(), 0);
    let first = live.push_audio(&pcm[ready - 1..ready]).unwrap();
    assert_eq!(first.len(), 6 * 8 * 8, "the first chunk is 6 groups");

    // The tail comes out at finish, and the totals match the unpadded loop exactly.
    let rest: usize = live.push_audio(&pcm[ready..]).unwrap().len() + live.finish().unwrap().len();
    let (mel, n) = m.log_mel(&pcm);
    let want = m
        .new_stream(preset(&m, &golden, "tiny"))
        .unwrap()
        .diarize_features_unpadded(&mel, n)
        .unwrap();
    assert_eq!(first.len() + rest, want.len());
    // `Nemotron3Live`'s own check, not the `MelStream` underneath it, which refuses too.
    let err = live.push_audio(&[0.0; 160]).unwrap_err().to_string();
    assert!(
        err.contains("Nemotron3Live: push_audio after finish"),
        "{err}"
    );
    assert!(live.finish().unwrap().is_empty(), "finish is idempotent");
}

/// A live stream holds a bounded window of mel, whatever its length: the buffers do not grow
/// with the audio. Loops the clip so the stream runs far longer than any window.
#[test]
fn live_buffers_stay_bounded() {
    let (Some(m), Some(golden)) = (model("nemotron3-diarization-f32.gguf"), Golden::load()) else {
        return;
    };
    let pcm = read_clip();
    let params = preset(&m, &golden, "tiny");
    let nm_window = params.chunk_len * 8 + params.right_context * 8;
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
    assert!(live.frames_emitted() > 3 * golden.n_mel_frames());
    // Chunk + lookahead, plus a chunk that was still waiting.
    assert!(
        peak_frames <= nm_window + 8 * 6,
        "held {peak_frames} mel frames after {} predictions",
        live.frames_emitted()
    );
}

/// A NaN or infinite sample is refused with nothing consumed: the stream carries on as if the
/// bad piece had never been pushed instead of poisoning every later frame.
#[test]
fn live_refuses_non_finite_pcm_without_consuming_it() {
    let Some(m) = model("nemotron3-diarization-f32.gguf") else {
        return;
    };
    let pcm = read_clip();
    let params = m.default_streaming().with_chunking(12, 1, 20, 40, 12);
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
    let err = dirty.push_audio(&bad).unwrap_err().to_string();
    assert!(err.contains("non-finite") && err.contains("8100"), "{err}");
    got.extend(dirty.push_audio(b).unwrap());
    got.extend(dirty.finish().unwrap());
    assert_eq!(got, want, "a refused piece must leave no trace");
}

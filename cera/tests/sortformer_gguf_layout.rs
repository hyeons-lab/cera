//! Layout contract for the Streaming Sortformer GGUFs written by
//! `scripts/sortformer/convert_sortformer.py`.
//!
//! The FastConformer half deliberately reuses the tensor names and shapes of the LFM2-Audio mmproj
//! (`a.conv1d.*`, `a.pre_encode.*`, `a.blk.N.*`) so `AudioEncoderWeights`' block loader can read it;
//! everything Sortformer-specific lives under `sf.*` / `sortformer.*`. These tests pin that contract
//! from cera's side, against the files the converter actually produced:
//!
//! * `inventory_and_metadata`: every expected tensor exists with the expected GGUF shape (ne order),
//!   the metadata the loader will need is present and has the checkpoint's values, and nothing
//!   unexpected is there (the unused `hidden_to_spks` must not be exported).
//! * `matrices_agree_across_storage_types`: the same tensor read from the F32 file and from the
//!   F16 / Q8_0 files gives the same GEMV result through cera's own kernels, which catches a block
//!   layout or type mix-up between gguf-py's writer and cera's reader. (The numerical fidelity of
//!   the conversion against NeMo is `scripts/sortformer/verify_gguf.py`'s job.)
//!
//! The models live in `~/.leap/models/sortformer/` (see the script's docstring for the commands);
//! without them these tests skip with a message, and `CERA_REQUIRE_MODEL=1` turns the skip into a
//! failure, like the other model-backed suites.

#![cfg(feature = "mmap")] // `GgufFile::open_arc`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use cera::gguf::GgufFile;
use cera::model::weights::MmapWeight;
use cera::tensor::DType;

const N_ENC: usize = 17;
const N_TF: usize = 18;
const D: usize = 512;
const FF: usize = 2048;
const TD: usize = 192;
const TFF: usize = 768;

fn model_path(file: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".leap/models/sortformer")
        .join(file);
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display(),
        );
        eprintln!("{} not found, skipping", path.display());
        return None;
    }
    Some(path)
}

fn open(file: &str) -> Option<Arc<GgufFile>> {
    let path = model_path(file)?;
    Some(GgufFile::open_arc(&path).unwrap_or_else(|e| panic!("open {}: {e:#}", path.display())))
}

/// Every tensor the converter writes, with its GGUF shape (ne order: fastest axis first).
fn expected_tensors() -> BTreeMap<String, Vec<usize>> {
    let mut t: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut put = |name: String, shape: &[usize]| {
        assert!(
            t.insert(name.clone(), shape.to_vec()).is_none(),
            "{name} listed twice"
        );
    };

    // Stem: depthwise-separable 3x3 convs with 256 channels; biases are [C, 1, 1] (ne [1, 1, C]).
    for (i, w) in [
        (0, [3, 3, 1, 256]),
        (2, [3, 3, 1, 256]),
        (3, [1, 1, 256, 256]),
        (5, [3, 3, 1, 256]),
        (6, [1, 1, 256, 256]),
    ] {
        put(format!("a.conv1d.{i}.weight"), &w);
        put(format!("a.conv1d.{i}.bias"), &[1, 1, 256]);
    }
    put("a.pre_encode.out.weight".into(), &[4096, D]);
    put("a.pre_encode.out.bias".into(), &[D]);

    for n in 0..N_ENC {
        let a = |s: &str| format!("a.blk.{n}.{s}");
        for norm in [
            "ffn_norm",
            "ffn_norm_1",
            "ln1",
            "ln2",
            "norm_conv",
            "conv_norm",
        ] {
            put(a(&format!("{norm}.weight")), &[D]);
            put(a(&format!("{norm}.bias")), &[D]);
        }
        for ffn in ["ffn_up", "ffn_up_1"] {
            put(a(&format!("{ffn}.weight")), &[D, FF]);
            put(a(&format!("{ffn}.bias")), &[FF]);
        }
        for ffn in ["ffn_down", "ffn_down_1"] {
            put(a(&format!("{ffn}.weight")), &[FF, D]);
            put(a(&format!("{ffn}.bias")), &[D]);
        }
        for proj in ["attn_q", "attn_k", "attn_v", "attn_out", "conv_pw2"] {
            put(a(&format!("{proj}.weight")), &[D, D]);
            put(a(&format!("{proj}.bias")), &[D]);
        }
        put(a("linear_pos.weight"), &[D, D]);
        put(a("pos_bias_u"), &[64, 8]);
        put(a("pos_bias_v"), &[64, 8]);
        put(a("conv_pw1.weight"), &[D, 2 * D]);
        put(a("conv_pw1.bias"), &[2 * D]);
        put(a("conv_dw.weight"), &[9, D]);
        put(a("conv_dw.bias"), &[D]);
    }

    put("sf.enc_proj.weight".into(), &[D, TD]);
    put("sf.enc_proj.bias".into(), &[TD]);
    for n in 0..N_TF {
        let s = |x: &str| format!("sf.blk.{n}.{x}");
        for norm in ["ln1", "ln2"] {
            put(s(&format!("{norm}.weight")), &[TD]);
            put(s(&format!("{norm}.bias")), &[TD]);
        }
        for proj in ["attn_q", "attn_k", "attn_v", "attn_out"] {
            put(s(&format!("{proj}.weight")), &[TD, TD]);
            put(s(&format!("{proj}.bias")), &[TD]);
        }
        put(s("ffn_up.weight"), &[TD, TFF]);
        put(s("ffn_up.bias"), &[TFF]);
        put(s("ffn_down.weight"), &[TFF, TD]);
        put(s("ffn_down.bias"), &[TD]);
    }
    put("sf.head.hidden.weight".into(), &[TD, TD]);
    put("sf.head.hidden.bias".into(), &[TD]);
    put("sf.head.out.weight".into(), &[TD, 4]);
    put("sf.head.out.bias".into(), &[4]);
    put("sf.mel.window".into(), &[400]);
    put("sf.mel.fb".into(), &[257, 128]);
    t
}

#[test]
fn inventory_and_metadata() {
    let expected = expected_tensors();
    assert_eq!(
        expected.len(),
        937,
        "the converter's tensor count changed; update this test with it"
    );

    for file in [
        "sortformer-4spk-v2.1-f32.gguf",
        "sortformer-4spk-v2.1-q8_0.gguf",
    ] {
        let Some(g) = open(file) else { return };
        assert_eq!(g.architecture(), Some("sortformer"), "{file}");

        for (name, shape) in &expected {
            let t = g
                .get_tensor(name)
                .unwrap_or_else(|e| panic!("{file}: {name}: {e:#}"));
            assert_eq!(t.shape(), shape.as_slice(), "{file}: {name}");
        }
        // No extra tensors either: the file holds exactly the expected inventory.
        let mut extra: Vec<&String> = g
            .tensors
            .keys()
            .filter(|k| !expected.contains_key(*k))
            .collect();
        extra.sort();
        assert!(extra.is_empty(), "{file}: unexpected tensors {extra:?}");
        assert_eq!(g.tensors.len(), expected.len(), "{file}");

        // Metadata the loader needs: encoder keys under the names AudioEncoderWeights already reads
        // (with the TRUE ffn width, unlike the mmproj's stale 512), plus the Sortformer ones.
        for (key, want) in [
            ("clip.audio.block_count", N_ENC as u32),
            ("clip.audio.embedding_length", D as u32),
            ("clip.audio.feed_forward_length", FF as u32),
            ("clip.audio.attention.head_count", 8),
            ("clip.audio.num_mel_bins", 128),
            ("sortformer.max_speakers", 4),
            ("sortformer.fc_d_model", D as u32),
            ("sortformer.tf_d_model", TD as u32),
            ("sortformer.tf_layer_count", N_TF as u32),
            ("sortformer.tf_head_count", 8),
            ("sortformer.tf_inner_size", TFF as u32),
            ("sortformer.subsampling_factor", 8),
            ("sortformer.conv_kernel_size", 9),
            ("sortformer.sample_rate", 16_000),
            ("sortformer.mel.n_fft", 512),
            ("sortformer.mel.win_length", 400),
            ("sortformer.mel.hop_length", 160),
            ("sortformer.mel.n_mels", 128),
            ("sortformer.mel.pad_to", 16),
            ("sortformer.stream.chunk_len", 188),
            ("sortformer.stream.chunk_left_context", 1),
            ("sortformer.stream.chunk_right_context", 1),
            ("sortformer.stream.fifo_len", 0),
            ("sortformer.stream.spkcache_len", 188),
            ("sortformer.stream.spkcache_update_period", 188),
            ("sortformer.stream.spkcache_sil_frames_per_spk", 3),
            ("sortformer.stream.max_index", 99_999),
        ] {
            assert_eq!(g.get_u32(key), Some(want), "{file}: {key}");
        }
        assert_eq!(
            g.get_str("sortformer.tf_activation"),
            Some("relu"),
            "{file}"
        );
        assert_eq!(g.get_str("sortformer.mel.normalize"), Some("NA"), "{file}");
        assert_eq!(g.get_bool("sortformer.xscaling"), Some(true), "{file}");
        assert_eq!(g.get_bool("clip.has_audio_encoder"), Some(true), "{file}");
        for (key, want) in [
            ("sortformer.mel.preemph", 0.97f32),
            ("sortformer.mel.mag_power", 2.0),
            ("sortformer.stream.pred_score_threshold", 0.25),
            ("sortformer.stream.scores_boost_latest", 0.05),
            ("sortformer.stream.sil_threshold", 0.2),
            ("sortformer.stream.strong_boost_rate", 0.75),
            ("sortformer.stream.weak_boost_rate", 1.5),
            ("sortformer.stream.min_pos_scores_rate", 0.5),
        ] {
            let got = g
                .get_f32(key)
                .unwrap_or_else(|| panic!("{file}: {key} missing"));
            assert!(
                (got - want).abs() < 1e-6,
                "{file}: {key} = {got}, want {want}"
            );
        }
        assert!(
            (g.get_f32("sortformer.mel.log_zero_guard").unwrap() - 2.0f32.powi(-24)).abs() < 1e-12,
            "{file}: log_zero_guard"
        );
        // NeMo's `hidden_to_spks` is unused at inference; the exact-inventory check above is what
        // keeps it out of the file.
    }
}

/// Deterministic, zero-mean test vector.
fn probe(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37).sin() + 0.25 * ((i as f32) * 0.011).cos())
        .collect()
}

fn gemv(g: &Arc<GgufFile>, name: &str) -> Vec<f32> {
    let w = MmapWeight::from_gguf(g, name).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    let x = probe(w.cols);
    let mut y = vec![0f32; w.rows];
    w.gemv(&x, &mut y);
    y
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = want.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
    num / den
}

#[test]
fn matrices_agree_across_storage_types() {
    let Some(f32g) = open("sortformer-4spk-v2.1-f32.gguf") else {
        return;
    };
    // (file, tensors that file stores in that type, tolerance on the relative L2 of the GEMV)
    let cases: [(&str, DType, &[&str], f64); 3] = [
        (
            "sortformer-4spk-v2.1-q8_0.gguf",
            DType::Q8_0,
            &[
                "a.pre_encode.out.weight",
                "a.blk.0.ffn_up.weight",
                "a.blk.0.ffn_down_1.weight",
                "a.blk.7.attn_q.weight",
                "a.blk.16.linear_pos.weight",
                "a.blk.3.conv_pw1.weight",
                "a.blk.16.conv_pw2.weight",
            ],
            2e-2,
        ),
        (
            "sortformer-4spk-v2.1-q8_0.gguf",
            // the Transformer head is F16 in every non-f32 file
            DType::F16,
            &[
                "sf.enc_proj.weight",
                "sf.blk.0.attn_q.weight",
                "sf.blk.17.ffn_up.weight",
                "sf.blk.5.ffn_down.weight",
                "sf.head.hidden.weight",
            ],
            1e-3,
        ),
        (
            "sortformer-4spk-v2.1-f16.gguf",
            DType::F16,
            &[
                "a.blk.0.ffn_up.weight",
                "a.blk.9.attn_out.weight",
                "a.pre_encode.out.weight",
            ],
            1e-3,
        ),
    ];
    for (file, stored, names, tol) in cases {
        let Some(g) = open(file) else { return };
        for name in names {
            // Without this the comparison could be F32 against F32 and prove nothing.
            let w = MmapWeight::from_gguf(&g, name).unwrap();
            assert_eq!(
                w.dtype, stored,
                "{file}: {name} is not stored as {stored:?}"
            );
            let got = gemv(&g, name);
            let want = gemv(&f32g, name);
            let err = rel_l2(&got, &want);
            assert!(
                err < tol,
                "{file}: {name}: GEMV relative L2 {err:.3e} vs the f32 file, want < {tol:e}"
            );
        }
    }
}

//! The Sortformer loader's metadata guards, without any model file.
//!
//! Every check here fires before the loader reads its first tensor, so a GGUF with the right
//! metadata and no tensors is enough: the valid baseline gets past every guard and stops only
//! at the first missing tensor, and each hostile variant is refused earlier, by name. These
//! tests need nothing from `~/.leap/models`, so unlike the model-backed suites in
//! `sortformer_parity.rs` they run in CI.

use std::sync::Arc;

use cera::convert::writer::GgufWriter;
use cera::gguf::GgufFile;
use cera::model::sortformer::SortformerModel;

#[derive(Clone)]
enum Val {
    U(u32),
    F(f32),
    S(&'static str),
    B(bool),
}

/// The metadata a converted checkpoint carries, as the loader reads it.
fn baseline() -> Vec<(&'static str, Val)> {
    use Val::*;
    vec![
        ("general.architecture", S("sortformer")),
        ("sortformer.mel.normalize", S("NA")),
        ("sortformer.mel.n_fft", U(512)),
        ("sortformer.mel.win_length", U(400)),
        ("sortformer.mel.hop_length", U(160)),
        ("sortformer.sample_rate", U(16_000)),
        ("sortformer.mel.preemph", F(0.97)),
        ("sortformer.mel.mag_power", F(2.0)),
        ("sortformer.mel.log_zero_guard", F(2.0f32.powi(-24))),
        ("sortformer.mel.pad_to", U(16)),
        ("sortformer.tf_activation", S("relu")),
        ("clip.audio.block_count", U(17)),
        ("clip.audio.embedding_length", U(512)),
        ("clip.audio.attention.head_count", U(8)),
        ("clip.audio.num_mel_bins", U(128)),
        ("clip.audio.attention.layer_norm_epsilon", F(1e-5)),
        ("sortformer.tf_layer_count", U(18)),
        ("sortformer.tf_d_model", U(192)),
        ("sortformer.tf_head_count", U(8)),
        ("sortformer.tf_inner_size", U(768)),
        ("sortformer.tf_layer_norm_epsilon", F(1e-5)),
        ("sortformer.max_speakers", U(4)),
        ("sortformer.fc_d_model", U(512)),
        ("sortformer.subsampling_factor", U(8)),
        ("sortformer.xscaling", B(true)),
    ]
}

/// The loader's error text for `baseline()` with `over` replacing keys. There are no tensors,
/// so it always fails: past every metadata guard, at the first tensor; before them, by name.
fn load(over: &[(&str, Val)]) -> String {
    let mut w = GgufWriter::new();
    for (k, v) in baseline() {
        let v = over
            .iter()
            .find(|(ok, _)| *ok == k)
            .map_or(v, |(_, ov)| ov.clone());
        match v {
            Val::U(x) => w.add_u32(k, x),
            Val::F(x) => w.add_f32(k, x),
            Val::S(x) => w.add_string(k, x),
            Val::B(x) => w.add_bool(k, x),
        }
    }
    let mut bytes = Vec::new();
    w.write_header_and_tensor_info(&mut bytes).unwrap();
    let g = Arc::new(GgufFile::from_bytes(bytes.into()).unwrap());
    match SortformerModel::from_gguf(&g) {
        Ok(_) => panic!("a GGUF without tensors loaded"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn the_valid_metadata_gets_past_every_guard() {
    // The first thing after the metadata guards is the stem's first tensor.
    let err = load(&[]);
    assert!(err.contains("a.conv1d"), "{err}");
}

#[test]
fn hostile_or_foreign_metadata_is_refused_by_name() {
    use Val::*;
    let cases: &[(&str, Val, &str)] = &[
        ("general.architecture", S("llama"), "not a Sortformer GGUF"),
        (
            "sortformer.mel.normalize",
            S("per_feature"),
            "mel normalize",
        ),
        ("sortformer.mel.n_fft", U(1024), "n_fft"),
        ("sortformer.mel.win_length", U(399), "win_length"),
        ("sortformer.mel.hop_length", U(161), "hop_length"),
        ("sortformer.sample_rate", U(8_000), "sample_rate"),
        ("sortformer.mel.preemph", F(0.9), "preemph"),
        ("sortformer.mel.mag_power", F(1.0), "mag_power"),
        ("sortformer.mel.log_zero_guard", F(1e-3), "log_zero_guard"),
        ("sortformer.mel.pad_to", U(0), "pad_to"),
        ("sortformer.mel.pad_to", U(65), "pad_to"),
        ("sortformer.mel.pad_to", U(u32::MAX), "pad_to"),
        ("sortformer.tf_activation", S("gelu"), "tf_activation"),
        ("clip.audio.block_count", U(0), "layer counts"),
        ("clip.audio.block_count", U(257), "layer counts"),
        ("sortformer.tf_layer_count", U(0), "layer counts"),
        ("sortformer.tf_layer_count", U(257), "layer counts"),
        ("clip.audio.embedding_length", U(256), "embedding_length"),
        ("clip.audio.attention.head_count", U(0), "head_count 0"),
        ("clip.audio.attention.head_count", U(3), "head_count 3"),
        ("sortformer.tf_head_count", U(0), "tf_head_count 0"),
        ("sortformer.tf_head_count", U(5), "tf_head_count 5"),
        ("sortformer.max_speakers", U(8), "max_speakers"),
        ("sortformer.fc_d_model", U(256), "fc_d_model"),
        ("sortformer.subsampling_factor", U(4), "subsampling_factor"),
    ];
    for (key, v, want) in cases {
        let err = load(&[(key, v.clone())]);
        assert!(err.contains(want), "{key}: want `{want}` in `{err}`");
        assert!(!err.contains("a.conv1d"), "{key} got past its guard: {err}");
    }
}

#[test]
fn the_bounded_fields_accept_both_ends_of_their_range() {
    use Val::*;
    for (key, v) in [
        ("sortformer.mel.pad_to", U(1)),
        ("sortformer.mel.pad_to", U(64)),
        ("clip.audio.block_count", U(1)),
        ("clip.audio.block_count", U(256)),
        ("sortformer.tf_layer_count", U(1)),
        ("sortformer.tf_layer_count", U(256)),
    ] {
        let err = load(&[(key, v)]);
        assert!(
            err.contains("a.conv1d"),
            "{key} was refused by a guard: {err}"
        );
    }
}

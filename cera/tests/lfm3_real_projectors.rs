//! The stricter projector loaders against the published LFM3 files.
//!
//! The unit tests build small synthetic files; these check that the real ones still load, so a
//! validation that is too tight for a shipped checkpoint fails here and not on a user's machine.
//! They need `mmproj-LFM3-Audio-2B-Q8_0.gguf` and `mmproj-LFM3-Omni-2B-Q8_0.gguf` (from the
//! `LiquidAI/LFM3-Audio-2B-GGUF` and `LiquidAI/LFM3-Omni-2B-GGUF` repositories) under
//! `CERA_ORACLE_MODELS_DIR`, and skip when a file is absent unless `CERA_REQUIRE_MODEL` is set.
#![cfg(all(feature = "std-fs", feature = "mmap"))]

mod common;

use std::sync::Arc;

use cera::gguf::GgufFile;
use cera::model::audio_decoder::{AudioDecoderWeights, DetokenizerWeights};
use cera::model::audio_encoder::AudioEncoderWeights;
use cera::model::vision_encoder::VisionEncoderWeights;

fn open(rel: &str) -> Option<Arc<GgufFile>> {
    let path = common::fixture_or_skip(rel, "lfm3-projectors")?;
    Some(Arc::new(
        GgufFile::open(&path).expect("open the projector GGUF"),
    ))
}

#[test]
fn the_published_lfm3_audio_projector_loads() {
    let Some(gguf) = open("LFM3-Audio-2B-GGUF/mmproj-LFM3-Audio-2B-Q8_0.gguf") else {
        return;
    };
    AudioEncoderWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("audio encoder: {e:#}"));
    AudioDecoderWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("audio decoder: {e:#}"));
    let detok =
        DetokenizerWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("audio detokenizer: {e:#}"));
    assert!(
        detok.is_lfm3a_gen(),
        "the published audio file is the lfm3a_gen schema"
    );
}

#[test]
fn the_published_lfm3_omni_projector_loads() {
    let Some(gguf) = open("LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-Q8_0.gguf") else {
        return;
    };
    AudioEncoderWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("audio encoder: {e:#}"));
    AudioDecoderWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("audio decoder: {e:#}"));
    DetokenizerWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("audio detokenizer: {e:#}"));
    let vision =
        VisionEncoderWeights::from_gguf(&gguf).unwrap_or_else(|e| panic!("vision encoder: {e:#}"));
    assert!(
        vision.is_lfm3(),
        "the omni file's vision tower is the lfm3 one"
    );
}

use super::*;
use crate::model::audio_decoder::{
    AudioDecoderWeights, DepthformerState, DetokenizerState, DetokenizerWeights,
    depthformer_forward, detokenize_to_spectrum, embed_audio_token, istft_to_pcm,
    sample_audio_frame,
};
use crate::model::audio_encoder::{AudioEncoderWeights, encode_audio_pcm, resample_linear};

use super::super::audio_fixture as fixture;
mod input;
mod output;

fn primary() -> Arc<[u8]> {
    super::fixture::vision_primary()
}

fn parts() -> ModelBytes {
    let mut parts = ModelBytes::text(primary());
    parts.inference_type = Some(InferenceType::LlamaCppLfm2AudioV1);
    parts
}

fn gguf(bytes: Arc<[u8]>) -> Arc<GgufFile> {
    Arc::new(GgufFile::from_bytes(bytes).unwrap())
}

fn encoder(bytes: Arc<[u8]>) -> AudioEncoderWeights {
    AudioEncoderWeights::from_gguf(&gguf(bytes)).unwrap()
}

fn decoder(bytes: Arc<[u8]>) -> AudioDecoderWeights {
    AudioDecoderWeights::from_gguf(&gguf(bytes)).unwrap()
}

fn detok(bytes: Arc<[u8]>) -> DetokenizerWeights {
    DetokenizerWeights::from_gguf(&gguf(bytes)).unwrap()
}

fn control() -> crate::Session {
    CeraEngine::from_bytes(primary(), config())
        .unwrap()
        .new_session(SessionConfig::default())
        .unwrap()
}

#[cfg(feature = "mmap")]
fn paths(parts: &ModelBytes, missing: bool) -> (Vec<CeraEngine>, Option<tempfile::TempDir>) {
    let dir = tempfile::tempdir().unwrap();
    let primary = dir.path().join("primary.gguf");
    std::fs::write(&primary, &parts.model).unwrap();
    let mut files = crate::ModelFiles::text(&primary);
    files.inference_type = parts.inference_type.clone();
    let mut manifest = super::super::filesystem::manifest("primary.gguf");
    manifest["inference_type"] = parts.inference_type.as_ref().unwrap().as_str().into();
    for (name, bytes, slot) in [
        (
            "multimodal_projector",
            &parts.multimodal_projector,
            &mut files.multimodal_projector,
        ),
        (
            "audio_decoder",
            &parts.audio_decoder,
            &mut files.audio_decoder,
        ),
        (
            "audio_tokenizer",
            &parts.audio_tokenizer,
            &mut files.audio_tokenizer,
        ),
    ] {
        if let Some(bytes) = bytes {
            let filename = format!("{name}.gguf");
            if !missing {
                std::fs::write(dir.path().join(&filename), bytes).unwrap();
            }
            *slot = Some(dir.path().join(&filename));
            manifest["load_time_parameters"][name] = filename.into();
        }
    }
    let mut engines = file_engines(files, config());
    let json = super::super::filesystem::write_manifest(dir.path(), &manifest);
    engines.extend(path_engines(&json, config()));
    engines.extend(path_engines(dir.path(), config()));
    #[cfg(unix)]
    {
        dir.close().unwrap();
        (engines, None)
    }
    #[cfg(not(unix))]
    (engines, Some(dir))
}

fn all_engines(parts: ModelBytes) -> (Vec<CeraEngine>, Option<tempfile::TempDir>) {
    let engines = parts_engines(parts.clone(), config());
    #[cfg(feature = "mmap")]
    {
        let mut engines = engines;
        let (file_engines, guard) = paths(&parts, false);
        engines.extend(file_engines);
        (engines, guard)
    }
    #[cfg(not(feature = "mmap"))]
    (engines, None)
}

// Manual companion to the external walkthrough; normal tests stay hermetic.
#[test]
#[ignore = "exports audio walkthrough fixtures into a new temporary directory"]
fn export_audio_walkthrough_fixture() {
    let dir = tempfile::Builder::new()
        .prefix("cera-audio-example-")
        .tempdir()
        .unwrap();
    for (name, bytes) in [
        ("primary.gguf", primary()),
        ("encoder.gguf", fixture::encoder(7, 32)),
        ("vocoder.gguf", fixture::vocoder(7, 32, true, true)),
    ] {
        std::fs::write(dir.path().join(name), bytes).unwrap();
    }
    println!("AUDIO_EXAMPLE_DIR={}", dir.keep().display());
}

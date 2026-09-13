#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use cera_ffi::{
    FfiSileroVad, FfiVadConfig, FfiVadIterator, FfiVadSampleRate, silero_vad_default_config,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn find_vad_model() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../models/silero_vad.gguf"),
        manifest_dir.join("models/silero_vad.gguf"),
        PathBuf::from("models/silero_vad.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[test]
fn test_ffi_vad_config_and_conversions() {
    let def = silero_vad_default_config();
    assert_eq!(def.threshold, 0.5);
    assert_eq!(def.neg_threshold, 0.35);
    assert_eq!(def.min_speech_duration_ms, 64);
    assert_eq!(def.min_silence_duration_ms, 100);
    assert_eq!(def.speech_pad_ms, 30);
    assert_eq!(def.frame_stride, None);

    let custom = FfiVadConfig {
        threshold: 0.6,
        neg_threshold: 0.4,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 150,
        speech_pad_ms: 20,
        frame_stride: Some(320),
    };

    let core: cera::vad::VadConfig = custom.clone().into();
    assert_eq!(core.threshold, 0.6);
    assert_eq!(core.neg_threshold, 0.4);
    assert_eq!(core.min_speech_duration_ms, 100);
    assert_eq!(core.min_silence_duration_ms, 150);
    assert_eq!(core.speech_pad_ms, 20);
    assert_eq!(core.frame_stride, Some(320));

    let roundtrip: FfiVadConfig = core.into();
    assert_eq!(roundtrip, custom);
}

#[test]
fn test_ffi_vad_iterator_and_model() -> Result<()> {
    let Some(model_path) = find_vad_model() else {
        eprintln!("Skipping test: models/silero_vad.gguf not found");
        return Ok(());
    };

    let model_str = model_path.to_str().unwrap().to_string();
    let vad = FfiSileroVad::from_file(model_str)?;
    vad.reset()?;

    // Test process_chunk_with_stride on FfiSileroVad
    let silence = vec![0.0f32; 512];
    let prob = vad.process_chunk_with_stride(silence.clone(), FfiVadSampleRate::Rate16kHz, 320)?;
    assert!(prob < 0.05);

    // Test FfiVadIterator configured with 20ms frame stride (320 samples)
    let config = FfiVadConfig {
        threshold: 0.5,
        neg_threshold: 0.35,
        min_speech_duration_ms: 64,
        min_silence_duration_ms: 100,
        speech_pad_ms: 30,
        frame_stride: Some(320),
    };
    let iterator = FfiVadIterator::new(FfiVadSampleRate::Rate16kHz, Some(config));
    assert_eq!(iterator.frame_stride()?, 320);
    assert!(!iterator.is_speech_active()?);

    // Feed silence in 320-sample chunks
    let chunk_20ms = vec![0.0f32; 320];
    for _ in 0..10 {
        let event = iterator.process_chunk(&vad, chunk_20ms.clone())?;
        assert!(event.is_none());
    }

    assert!(!iterator.is_speech_active()?);
    assert!(iterator.pop_event()?.is_none());
    let flushed = iterator.flush()?;
    assert!(flushed.is_none());

    iterator.reset()?;
    assert!(!iterator.is_speech_active()?);

    Ok(())
}

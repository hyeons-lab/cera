//! Native ASR shares immutable inputs while preserving a live GPU conversation.
use super::*;

fn parts() -> ModelBytes {
    let mut parts = ModelBytes::text(fixture::tiny_audio());
    parts.inference_type = Some(InferenceType::LlamaCppLfm2AudioV1);
    parts.multimodal_projector = Some(super::super::audio_fixture::encoder(7, 32));
    parts
}

fn audio_engine(backend: BackendPreference) -> CeraEngine {
    CeraEngine::from_parts(
        parts(),
        EngineConfig {
            backend,
            context_size: 512,
            ..EngineConfig::default()
        },
    )
    .unwrap()
}

fn pcm() -> Vec<f32> {
    (0..3200).map(|i| (i as f32 * 0.1).sin() * 0.2).collect()
}

fn exercise(backend: BackendPreference) {
    for compression in [KvCompression::None, KvCompression::turboquant(42)] {
        let mut engine = Arc::new(audio_engine(backend));
        let config = SessionConfig {
            kv_compression: compression,
            seed: Some(42),
            ..SessionConfig::default()
        };
        let mut live = engine.new_session(config.clone()).unwrap();
        let control_engine = audio_engine(backend);
        let mut control = control_engine.new_session(config).unwrap();
        live.append_tokens(&[0, 1, 0]).unwrap();
        control.append_tokens(&[0, 1, 0]).unwrap();
        same_live(&live, &control);
        assert!(matches!(
            engine.new_session(SessionConfig::default()),
            Err(CeraError::Busy)
        ));
        assert!(engine.transcription_model.lock().unwrap().is_none());

        // Failed helper construction must leave the slot retryable and the
        // original conversation intact. Inject an unsupported retained model.
        let retained = engine.primary_gguf.clone();
        let mut invalid = GgufWriter::new();
        invalid.add_string("general.architecture", "unsupported-test-architecture");
        let mut bytes = Vec::new();
        invalid.write_header_and_tensor_info(&mut bytes).unwrap();
        Arc::get_mut(&mut engine).unwrap().primary_gguf =
            Some(Arc::new(GgufFile::from_bytes(bytes.into()).unwrap()));
        assert!(engine.transcribe(&pcm(), 16000).is_err());
        assert!(engine.transcription_model.lock().unwrap().is_none());
        same_live(&live, &control);
        Arc::get_mut(&mut engine).unwrap().primary_gguf = retained;

        let samples = pcm();
        let expected = audio_engine(backend).transcribe(&samples, 16000).unwrap();
        assert!(!expected.is_empty());
        assert_eq!(engine.transcribe(&samples, 16000).unwrap(), expected);
        let helper = engine
            .transcription_model
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        assert!(!Arc::ptr_eq(&helper, &engine.model));
        assert_eq!(helper.warm_cache_usage(), Some((0, 0)));
        // A failed ASR releases its helper lease and the queue guard.
        assert!(engine.transcribe(&[], 16000).is_err());
        assert_eq!(engine.transcribe(&samples, 16000).unwrap(), expected);
        engine.configure_cache(KvCacheConfig {
            max_warm_entries: 0,
            ..KvCacheConfig::default()
        });
        engine.clear_warm_cache();
        engine.clear_cache();
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let engine = Arc::clone(&engine);
                let samples = samples.clone();
                std::thread::spawn(move || engine.transcribe(&samples, 16000).unwrap())
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), expected);
        }
        assert_eq!(helper.warm_cache_usage(), Some((0, 0)));
        assert!(Arc::ptr_eq(
            &helper,
            engine.transcription_model.lock().unwrap().as_ref().unwrap()
        ));
        same_live(&live, &control);
        drop(helper);
        drop(engine);
        live.append_tokens(&[1, 0, 1]).unwrap();
        control.append_tokens(&[1, 0, 1]).unwrap();
        same_live(&live, &control);
        assert_eq!(generate(&mut live).tokens, generate(&mut control).tokens);
        same_live(&live, &control);
    }

    #[cfg(all(feature = "mmap", unix))]
    {
        // The helper is first loaded after the original files disappear.
        let dir = tempfile::tempdir().unwrap();
        let parts = parts();
        let primary = dir.path().join("primary.gguf");
        let encoder = dir.path().join("encoder.gguf");
        std::fs::write(&primary, parts.model).unwrap();
        std::fs::write(&encoder, parts.multimodal_projector.unwrap()).unwrap();
        let mut files = crate::ModelFiles::text(&primary);
        files.multimodal_projector = Some(encoder);
        files.inference_type = parts.inference_type;
        let engine = CeraEngine::from_files(
            files,
            EngineConfig {
                backend,
                context_size: 512,
                ..EngineConfig::default()
            },
        )
        .unwrap();
        let _live = engine.new_session(SessionConfig::default()).unwrap();
        dir.close().unwrap();
        assert!(!primary.exists());
        let samples = pcm();
        assert_eq!(
            engine.transcribe(&samples, 16000).unwrap(),
            audio_engine(backend).transcribe(&samples, 16000).unwrap()
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a Metal device; no CPU fallback or successful skip"]
fn metal_transcription_preserves_live_conversation() {
    exercise(BackendPreference::Metal);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a wgpu device; no CPU fallback or successful skip"]
fn wgpu_transcription_preserves_live_conversation() {
    exercise(BackendPreference::Gpu);
}

#[test]
#[ignore = "exports tiny GPU ownership fixtures into a new temporary directory"]
fn export_gpu_ownership_fixtures() {
    let dir = tempfile::Builder::new()
        .prefix("cera-gpu-ownership-")
        .tempdir()
        .unwrap();
    std::fs::write(dir.path().join("primary.gguf"), fixture::tiny_audio()).unwrap();
    std::fs::write(dir.path().join("conversation.gguf"), fixture::tiny_hybrid()).unwrap();
    std::fs::write(
        dir.path().join("encoder.gguf"),
        super::super::audio_fixture::encoder(7, 32),
    )
    .unwrap();
    println!("GPU_OWNERSHIP_FIXTURES={}", dir.keep().display());
}

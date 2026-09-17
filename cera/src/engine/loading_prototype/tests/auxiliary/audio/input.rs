use super::*;

fn pcm() -> Vec<f32> {
    (0..1600).map(|i| (i as f32 * 0.13).sin() * 0.2).collect()
}

fn check(engine: CeraEngine, capability: bool, hidden: Option<usize>) {
    assert_eq!(engine.capabilities().audio_in, capability);
    assert_eq!(engine.capabilities().audio_out, capability);
    assert_eq!(
        engine.audio_encoder().map(|w| w.config.llm_hidden_size),
        hidden
    );
    assert!(!engine.has_gpu_audio_encoder());
    let mut session = engine.new_session(SessionConfig::default()).unwrap();
    drop(engine);
    let result = session.append_audio(&pcm(), 16_000);
    if !capability {
        assert!(matches!(result, Err(CeraError::UnsupportedModality)));
    } else if hidden.is_none() {
        assert!(
            matches!(result, Err(CeraError::Backend(message)) if message.contains("no audio encoder attached"))
        );
    } else if hidden != Some(32) {
        assert!(
            matches!(result, Err(CeraError::Backend(message)) if message.contains("does not match the LLM"))
        );
    } else {
        result.unwrap();
        let (expected, frames) = encode_audio_pcm(&pcm(), &encoder(fixture::encoder(7, 32)));
        assert!(frames > 0);
        assert_eq!(session.position() as usize, frames);
        let mut reference = control();
        reference.append_embeddings(&expected, frames).unwrap();
        session.append_tokens(&[0, 1]).unwrap();
        reference.append_tokens(&[0, 1]).unwrap();
        close(
            session.last_logits().unwrap(),
            reference.last_logits().unwrap(),
        );
        return;
    }
    assert_eq!(session.position(), 0);
    assert_eq!(generate(&mut session), generate(&mut control()));
}

#[test]
fn encoder_loading_honors_modality_and_keeps_optional_failures_local() {
    for (bytes, hidden) in [
        (None, None),
        (Some(Arc::from(&b"invalid GGUF"[..])), None),
        (Some(header("clip")), None),
        (Some(fixture::encoder(7, 32)), Some(32)),
        (Some(fixture::encoder(7, 64)), Some(64)),
    ] {
        for inference in [
            None,
            Some(InferenceType::LlamaCppTextToText),
            Some(InferenceType::LlamaCppLfm2AudioV1),
        ] {
            let capability = inference == Some(InferenceType::LlamaCppLfm2AudioV1);
            let mut source = parts();
            source.inference_type = inference;
            source.multimodal_projector = bytes.clone();
            for engine in parts_engines(source.clone(), config()) {
                check(engine, capability, hidden.filter(|_| capability));
            }
            #[cfg(feature = "mmap")]
            {
                // Manifests require a declared type. Files' inferred primary is
                // text here; supply that equivalent declaration to the path set.
                source
                    .inference_type
                    .get_or_insert(InferenceType::LlamaCppTextToText);
                let (engines, _directory) = paths(&source, false);
                for engine in engines {
                    check(engine, capability, hidden.filter(|_| capability));
                }
            }
        }
    }
    #[cfg(feature = "mmap")]
    {
        let mut source = parts();
        source.multimodal_projector = Some(fixture::encoder(7, 32));
        let (engines, _directory) = paths(&source, true);
        for engine in engines {
            check(engine, true, None);
        }
    }
}

#[test]
fn retained_audio_input_executes_resampling_and_isolates_live_sessions() {
    let samples = pcm();
    let weights = encoder(fixture::encoder(7, 32));
    let (expected, frames) = encode_audio_pcm(&samples, &weights);
    let (silence, silence_frames) = encode_audio_pcm(&vec![0.0; samples.len()], &weights);
    assert_eq!(frames, silence_frames);
    distinct(&expected, &silence);
    let (other, other_frames) = encode_audio_pcm(&samples, &encoder(fixture::encoder(29, 32)));
    assert_eq!(frames, other_frames);
    distinct(&expected, &other);
    let resampled = resample_linear(&samples, 8_000, 16_000);
    let (expected_resampled, resampled_frames) = encode_audio_pcm(&resampled, &weights);
    assert!(resampled_frames > frames);
    let mut source = parts();
    source.multimodal_projector = Some(fixture::encoder(7, 32));
    let (engines, _directory) = all_engines(source);
    for engine in engines {
        let mut a = engine.new_session(SessionConfig::default()).unwrap();
        let mut b = engine.new_session(SessionConfig::default()).unwrap();
        drop(engine);
        let mut ca = control();
        let mut cb = control();
        a.append_audio(&samples, 16_000).unwrap();
        ca.append_embeddings(&expected, frames).unwrap();
        b.append_tokens(&[1, 0, 0]).unwrap();
        cb.append_tokens(&[1, 0, 0]).unwrap();
        a.append_tokens(&[0, 1]).unwrap();
        ca.append_tokens(&[0, 1]).unwrap();
        assert_eq!(a.position() as usize, frames + 2);
        close(a.last_logits().unwrap(), ca.last_logits().unwrap());
        a.reset().unwrap();
        b.append_audio(&samples, 8_000).unwrap();
        cb.append_embeddings(&expected_resampled, resampled_frames)
            .unwrap();
        b.append_tokens(&[1]).unwrap();
        cb.append_tokens(&[1]).unwrap();
        assert_eq!(b.position() as usize, resampled_frames + 4);
        close(b.last_logits().unwrap(), cb.last_logits().unwrap());
        a.append_audio(&samples, 16_000).unwrap();
        a.append_tokens(&[0, 1]).unwrap();
        close(a.last_logits().unwrap(), ca.last_logits().unwrap());
    }
}

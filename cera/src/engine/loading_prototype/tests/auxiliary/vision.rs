use super::*;
use crate::model::vision_encoder::VisionEncoderWeights;

fn weights(bytes: Arc<[u8]>) -> VisionEncoderWeights {
    VisionEncoderWeights::from_gguf(&Arc::new(GgufFile::from_bytes(bytes).unwrap())).unwrap()
}

fn pixels() -> Vec<f32> {
    (0..12).map(|i| i as f32 / 12.0).collect()
}

fn check(engine: CeraEngine, raw: bool, typed: bool, capability: bool) {
    assert_eq!(engine.capabilities().image_in, capability);
    assert_eq!(engine.vision_encoder_gguf().is_some(), raw);
    assert_eq!(engine.vision_encoder().is_some(), typed);
    assert!(!engine.has_gpu_vision_encoder());
    if typed {
        let actual = engine
            .vision_encoder()
            .unwrap()
            .encode_image(&pixels(), 2, 2)
            .unwrap();
        let expected = weights(fixture::vision(7))
            .encode_image(&pixels(), 2, 2)
            .unwrap();
        let other = weights(fixture::vision(29))
            .encode_image(&pixels(), 2, 2)
            .unwrap();
        assert_eq!(actual.len(), 4 * 32);
        close(&actual, &expected);
        distinct(&actual, &other);
        #[cfg(feature = "vl-preprocess")]
        if !capability {
            assert!(matches!(
                engine
                    .new_session(SessionConfig::default())
                    .unwrap()
                    .append_image(&[]),
                Err(CeraError::UnsupportedModality)
            ));
        }
        let mut control = CeraEngine::from_bytes(fixture::vision_primary(), config())
            .unwrap()
            .new_session(SessionConfig::default())
            .unwrap();
        let mut session = engine.new_session(SessionConfig::default()).unwrap();
        drop(engine);
        session.append_embeddings(&actual, 4).unwrap();
        control.append_embeddings(&expected, 4).unwrap();
        session.append_tokens(&[0, 1]).unwrap();
        control.append_tokens(&[0, 1]).unwrap();
        assert_eq!(session.position(), 6);
        close(
            session.last_logits().unwrap(),
            control.last_logits().unwrap(),
        );
    } else {
        let mut session = engine.new_session(SessionConfig::default()).unwrap();
        drop(engine);
        #[cfg(feature = "vl-preprocess")]
        if capability {
            assert!(
                matches!(session.append_image(&[]), Err(CeraError::Backend(message)) if message.contains("no vision encoder attached"))
            );
        } else {
            assert!(matches!(
                session.append_image(&[]),
                Err(CeraError::UnsupportedModality)
            ));
        }
        // Optional companion failures must still leave usable text inference.
        let mut control = CeraEngine::from_bytes(fixture::vision_primary(), config())
            .unwrap()
            .new_session(SessionConfig::default())
            .unwrap();
        assert_eq!(generate(&mut session), generate(&mut control));
    }
}

#[test]
fn multipart_distinguishes_declared_raw_and_usable_vision() {
    for (sidecar, parsed, usable) in [
        (None, false, false),
        (Some(Arc::from(&b"bad GGUF"[..])), false, false),
        (Some(header("clip")), true, false),
        (Some(fixture::vision(7)), true, true),
    ] {
        for inference in [
            None,
            Some(InferenceType::LlamaCppTextToText),
            Some(InferenceType::LlamaCppImageToText),
        ] {
            let capability = match inference {
                None => parsed,
                Some(InferenceType::LlamaCppImageToText) => true,
                _ => false,
            };
            let mut parts = ModelBytes::text(fixture::vision_primary());
            parts.multimodal_projector = sidecar.clone();
            parts.inference_type = inference;
            for engine in parts_engines(parts, config()) {
                check(
                    engine,
                    parsed && capability,
                    usable && capability,
                    capability,
                );
            }
        }
    }
}

#[cfg(feature = "mmap")]
#[test]
fn files_and_manifests_load_real_vision_without_changing_inference_rules() {
    use std::fs;
    let dir = tempfile::tempdir().unwrap();
    let primary = dir.path().join("primary.gguf");
    fs::write(&primary, fixture::vision_primary()).unwrap();
    for (name, bytes, raw, usable) in [
        ("valid.gguf", Some(fixture::vision(7)), true, true),
        ("header.gguf", Some(header("clip")), true, false),
        (
            "corrupt.gguf",
            Some(Arc::from(&b"invalid"[..])),
            false,
            false,
        ),
        ("missing.gguf", None, false, false),
    ] {
        let sidecar = dir.path().join(name);
        if let Some(bytes) = bytes {
            fs::write(&sidecar, bytes).unwrap();
        }
        for inference in [
            None,
            Some(InferenceType::LlamaCppTextToText),
            Some(InferenceType::LlamaCppImageToText),
        ] {
            // Files infer only from the primary, unlike multipart bytes. They
            // still open/attach a supplied projector even in text mode.
            let capability = inference == Some(InferenceType::LlamaCppImageToText);
            let mut files = crate::ModelFiles::text(&primary);
            files.multimodal_projector = Some(sidecar.clone());
            files.inference_type = inference;
            for engine in file_engines(files, config()) {
                check(engine, raw, usable, capability);
            }
        }
        let mut manifest = super::super::filesystem::manifest("primary.gguf");
        manifest["inference_type"] = "llama.cpp/image-to-text".into();
        manifest["load_time_parameters"]["multimodal_projector"] = name.into();
        let json = super::super::filesystem::write_manifest(dir.path(), &manifest);
        for path in [&json, dir.path()] {
            for engine in path_engines(path, config()) {
                check(engine, raw, usable, true);
            }
        }
    }
}

#[cfg(feature = "vl-preprocess")]
#[test]
fn attached_vision_executes_after_parent_release_and_keeps_sessions_independent() {
    let image = image::RgbImage::from_fn(8, 8, |x, y| {
        image::Rgb([(x * 31) as u8, (y * 29) as u8, 73])
    });
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png).unwrap();
    let image = png.into_inner();
    let encoder = weights(fixture::vision(7));
    let pre = crate::model::vision_preprocessor::preprocess_image(&image, &encoder.config).unwrap();
    let embeddings = encoder
        .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
        .unwrap();
    assert_eq!(embeddings.len(), 64 * 32);
    let other = weights(fixture::vision(29))
        .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
        .unwrap();
    distinct(&embeddings, &other);
    let mut parts = ModelBytes::text(fixture::vision_primary());
    parts.multimodal_projector = Some(fixture::vision(7));
    parts.inference_type = Some(InferenceType::LlamaCppImageToText);
    let (engines, _source_directory) = executable_engines(parts);
    for engine in engines {
        let mut a = engine.new_session(SessionConfig::default()).unwrap();
        let mut b = engine.new_session(SessionConfig::default()).unwrap();
        let mut control = CeraEngine::from_bytes(fixture::vision_primary(), config())
            .unwrap()
            .new_session(SessionConfig::default())
            .unwrap();
        drop(engine);
        a.append_image(&image).unwrap();
        b.append_tokens(&[1, 0, 0]).unwrap();
        control.append_embeddings(&embeddings, 64).unwrap();
        a.append_tokens(&[0, 1]).unwrap();
        control.append_tokens(&[0, 1]).unwrap();
        assert_eq!(a.position(), 66);
        close(a.last_logits().unwrap(), control.last_logits().unwrap());
        a.reset().unwrap();
        b.append_image(&image).unwrap();
        control.reset().unwrap();
        control.append_tokens(&[1, 0, 0]).unwrap();
        control.append_embeddings(&embeddings, 64).unwrap();
        b.append_tokens(&[1]).unwrap();
        control.append_tokens(&[1]).unwrap();
        assert_eq!(b.position(), 68);
        close(b.last_logits().unwrap(), control.last_logits().unwrap());
    }
}

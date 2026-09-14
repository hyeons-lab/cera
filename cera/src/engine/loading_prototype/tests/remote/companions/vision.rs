use super::*;
use crate::model::vision_encoder::VisionEncoderWeights;

fn weights(seed: u32) -> VisionEncoderWeights {
    VisionEncoderWeights::from_gguf(&Arc::new(
        GgufFile::from_bytes(fixture::vision(seed)).unwrap(),
    ))
    .unwrap()
}

fn execute(engine: CeraEngine, seed: u32) {
    assert!(engine.capabilities().image_in);
    let expected = weights(seed);
    let pixels: Vec<_> = (0..12).map(|i| i as f32 / 12.0).collect();
    let embeddings = expected.encode_image(&pixels, 2, 2).unwrap();
    close(
        &engine
            .vision_encoder()
            .unwrap()
            .encode_image(&pixels, 2, 2)
            .unwrap(),
        &embeddings,
    );
    distinct(
        &embeddings,
        &weights(if seed == 7 { 29 } else { 7 })
            .encode_image(&pixels, 2, 2)
            .unwrap(),
    );
    let mut session = engine.new_session(SessionConfig::default()).unwrap();
    let mut control = CeraEngine::from_bytes(
        fixture::vision_primary(),
        LoadConfig {
            context_size: 256,
            ..cpu_config()
        },
    )
    .unwrap()
    .new_session(SessionConfig::default())
    .unwrap();
    drop(engine);
    session.append_embeddings(&embeddings, 4).unwrap();
    control.append_embeddings(&embeddings, 4).unwrap();
    #[cfg(feature = "vl-preprocess")]
    {
        let image = image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 31) as u8, (y * 29) as u8, 73])
        });
        let mut png = Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let pre =
            crate::model::vision_preprocessor::preprocess_image(png.get_ref(), &expected.config)
                .unwrap();
        let encoded = expected
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .unwrap();
        session.append_image(png.get_ref()).unwrap();
        control.append_embeddings(&encoded, 64).unwrap();
    }
    session.append_tokens(&[0, 1]).unwrap();
    control.append_tokens(&[0, 1]).unwrap();
    assert_eq!(session.position(), control.position());
    close(
        session.last_logits().unwrap(),
        control.last_logits().unwrap(),
    );
}

#[test]
fn hf_vision_selection_repairs_cache_and_executes_retained_sessions() {
    isolated_with_routes(
        "companions::vision::hf_vision_selection_repairs_cache_and_executes_retained_sessions",
        || {
            routes(
                "vision",
                "image-text-to-text",
                &[
                    ("model-Q4_K_M.gguf", fixture::vision_primary()),
                    ("model-Q8_0.gguf", fixture::vision_primary()),
                    ("model-F16.gguf", fixture::vision_primary()),
                    ("mmproj-Q4_K_M.gguf", fixture::vision(7)),
                    ("mmproj-Q8_0.gguf", fixture::vision(29)),
                ],
            )
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let url = format!(
                "{}/fixture/vision/resolve/{MAIN_COMMIT}/mmproj-Q4_K_M.gguf",
                ctx.url
            );
            let wrong = fixture::vision(29);
            assert_eq!(wrong.len(), fixture::vision(7).len());
            assert_ne!(digest(&wrong), digest(&fixture::vision(7)));
            let path = seed(repo, &url, &wrong);
            assert!(!crate::bundle::download::sidecar_path(&path).exists());
            for (quant, chosen, seed) in [
                ("Q4_K_M", "Q4_K_M", 7),
                ("Q8_0", "Q8_0", 29),
                ("F16", "Q8_0", 29),
            ] {
                for engine in hf_engines(&format!("fixture/vision:{quant}"), &cfg) {
                    let url = format!(
                        "{}/fixture/vision/resolve/{MAIN_COMMIT}/mmproj-{chosen}.gguf",
                        ctx.url
                    );
                    assert_eq!(
                        Path::new(
                            engine
                                .manifest()
                                .files
                                .multimodal_projector
                                .as_ref()
                                .unwrap()
                        ),
                        repo.fixture_path(&url).unwrap()
                    );
                    execute(engine, seed);
                    assert_cached(repo, &progress, &url, &fixture::vision(seed));
                }
            }
        },
        |requests| {
            for file in [
                "model-Q4_K_M.gguf",
                "model-Q8_0.gguf",
                "model-F16.gguf",
                "mmproj-Q4_K_M.gguf",
                "mmproj-Q8_0.gguf",
            ] {
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/fixture/vision/resolve/{MAIN_COMMIT}/{file}")
                    ),
                    1
                );
            }
        },
    );
}

#[test]
fn remote_optional_parse_failures_allow_text_but_download_failures_are_fatal() {
    isolated_with_routes(
        "companions::vision::remote_optional_parse_failures_allow_text_but_download_failures_are_fatal",
        || {
            let mut mismatch = Response::gguf(fixture::vision(7));
            mismatch.hash = Some("0".repeat(64));
            HashMap::from([
                (
                    "/assets/primary.gguf".into(),
                    Response::gguf(fixture::vision_primary()),
                ),
                ("/assets/bad.gguf".into(), Response::gguf(b"invalid GGUF")),
                ("/assets/header.gguf".into(), Response::gguf(header("clip"))),
                ("/assets/mismatch.gguf".into(), mismatch),
            ])
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let path = ctx.root.join("manifest.json");
            for asset in ["bad", "header", "mismatch", "missing"] {
                let mut manifest = manifest(&format!("{}/assets/primary.gguf", ctx.url));
                manifest["inference_type"] = "llama.cpp/image-to-text".into();
                manifest["load_time_parameters"]["multimodal_projector"] =
                    format!("{}/assets/{asset}.gguf", ctx.url).into();
                fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
                if matches!(asset, "bad" | "header") {
                    let model = load(ModelSource::path(&path), cfg.clone());
                    let legacy = CeraEngine::from_path(&path, cfg.clone()).unwrap();
                    for engine in [unwrap_model(model), legacy] {
                        assert!(engine.capabilities().image_in);
                        assert!(engine.vision_encoder().is_none());
                        assert_eq!(engine.vision_encoder_gguf().is_some(), asset == "header");
                        let mut session = engine.new_session(SessionConfig::default()).unwrap();
                        drop(engine);
                        let mut control =
                            CeraEngine::from_bytes(fixture::vision_primary(), cfg.clone())
                                .unwrap()
                                .new_session(SessionConfig::default())
                                .unwrap();
                        assert_eq!(generate(&mut session), generate(&mut control));
                    }
                } else {
                    let actual = engine_error(
                        ModelLoader::new(ModelSource::path(&path))
                            .config(cfg.clone())
                            .build(),
                        "path",
                    );
                    same_error(
                        actual,
                        CeraEngine::from_path(&path, cfg.clone()).err().unwrap(),
                    );
                    let dest = repo
                        .fixture_path(&format!("{}/assets/{asset}.gguf", ctx.url))
                        .unwrap();
                    assert!(!dest.exists());
                    assert!(!crate::bundle::download::partial_path(&dest).exists());
                    assert!(!crate::bundle::download::sidecar_path(&dest).exists());
                }
            }
        },
        |requests| {
            for (asset, n) in [
                ("primary", 1),
                ("bad", 1),
                ("header", 1),
                ("mismatch", 2),
                ("missing", 2),
            ] {
                assert_eq!(count(requests, "GET", &format!("/assets/{asset}.gguf")), n);
            }
        },
    );
}

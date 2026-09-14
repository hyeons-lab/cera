use super::*;

fn execute(mut engine: CeraEngine, token: u32, k: usize) {
    assert_eq!(engine.drafter.as_ref().unwrap().suggested_k(), Some(k));
    let calls = observe_drafter(&mut engine);
    let mut session = engine.new_session(SessionConfig::default()).unwrap();
    drop(engine);
    let mut control = CeraEngine::from_bytes(
        fixture::primary(),
        LoadConfig {
            context_size: 256,
            ..cpu_config()
        },
    )
    .unwrap()
    .new_session(SessionConfig::default())
    .unwrap();
    assert_eq!(generate(&mut session), generate(&mut control));
    let proposals = calls.lock().unwrap();
    assert!(
        !proposals.is_empty(),
        "generation bypassed the remote drafter"
    );
    assert!(proposals.iter().all(|tokens| tokens == &vec![token; k]));
    session.append_tokens(&[1]).unwrap();
    control.append_tokens(&[1]).unwrap();
    assert_eq!(session.position(), control.position());
    close(
        session.last_logits().unwrap(),
        control.last_logits().unwrap(),
    );
}

#[test]
fn hf_and_manifest_drafts_execute_with_configuration_precedence() {
    isolated_with_routes(
        "companions::draft::hf_and_manifest_drafts_execute_with_configuration_precedence",
        || {
            routes(
                "draft",
                "text-generation",
                &[
                    ("model-Q4_K_M.gguf", fixture::primary()),
                    ("draft-Q4_K_M.gguf", fixture::draft(0, 2)),
                    ("draft-Q8_0.gguf", fixture::draft(1, 3)),
                ],
            )
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let base = format!("{}/fixture/draft/resolve/{MAIN_COMMIT}", ctx.url);
            let mut manifest = manifest(&format!("{base}/model-Q4_K_M.gguf"));
            manifest["load_time_parameters"]["draft_model"] =
                format!("{base}/draft-Q4_K_M.gguf").into();
            let path = ctx.root.join("manifest.json");
            let bytes = serde_json::to_vec(&manifest).unwrap();
            fs::write(&path, &bytes).unwrap();
            let catalog_url =
                crate::bundle::leap_bundles_manifest_url("fixture-remote-draft", "Q4_K_M").unwrap();
            seed(repo, &catalog_url, &bytes);
            let override_path = ctx.root.join("override.gguf");
            fs::write(&override_path, fixture::draft(1, 3)).unwrap();
            for use_override in [false, true] {
                let cfg = LoadConfig {
                    draft_model: use_override.then(|| override_path.clone()),
                    ..cfg.clone()
                };
                let mut engines = hf_engines("fixture/draft:Q4_K_M", &cfg);
                engines.push(unwrap_model(load(ModelSource::path(&path), cfg.clone())));
                engines.push(CeraEngine::from_path(&path, cfg.clone()).unwrap());
                engines.push(unwrap_model(load(
                    ModelSource::bundle_id("fixture-remote-draft", "Q4_K_M"),
                    cfg.clone(),
                )));
                engines.push(
                    CeraEngine::from_bundle_id("fixture-remote-draft", "Q4_K_M", cfg.clone())
                        .unwrap(),
                );
                for engine in engines {
                    // Manifest metadata retains the discovered path even when
                    // explicit configuration supplies the executable drafter.
                    assert_eq!(
                        Path::new(engine.manifest().files.draft_model.as_ref().unwrap()),
                        repo.fixture_path(&format!("{base}/draft-Q4_K_M.gguf"))
                            .unwrap()
                    );
                    execute(
                        engine,
                        u32::from(use_override),
                        if use_override { 3 } else { 2 },
                    );
                }
            }
            assert_cached(
                repo,
                &progress,
                &format!("{base}/draft-Q4_K_M.gguf"),
                &fixture::draft(0, 2),
            );
        },
        |requests| {
            for (file, expected) in [
                ("model-Q4_K_M.gguf", 1),
                ("draft-Q4_K_M.gguf", 1),
                ("draft-Q8_0.gguf", 0),
            ] {
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/fixture/draft/resolve/{MAIN_COMMIT}/{file}")
                    ),
                    expected
                );
            }
            assert!(count(requests, "CONNECT", "huggingface.co:443") > 0);
        },
    );
}

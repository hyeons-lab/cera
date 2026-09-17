use super::*;

#[track_caller]
fn selected(engine: &CeraEngine, expected: Option<(u32, usize)>) {
    assert_eq!(engine.drafter.is_some(), expected.is_some());
    if let Some((token, k)) = expected {
        let source = engine.drafter.as_ref().unwrap();
        assert_eq!(source.suggested_k(), Some(k));
        let mut a = source.clone_drafter();
        let mut b = source.clone_drafter();
        assert_eq!(a.draft(&[0, 1], 9), vec![token; k]);
        assert_eq!(b.draft(&[1, 0, 0], 9), vec![token; k]);
        a.reset();
        assert_eq!(b.draft(&[1, 0, 0, 1], 9), vec![token; k]);
        assert_eq!(a.draft(&[1], 9), vec![token; k]);
    }
    let mut session = engine.new_session(SessionConfig::default()).unwrap();
    let mut control = CeraEngine::from_bytes(fixture::primary(), config())
        .unwrap()
        .new_session(SessionConfig::default())
        .unwrap();
    assert_eq!(generate(&mut session), generate(&mut control));
}

#[test]
fn multipart_loads_real_drafts_and_ignores_optional_failures() {
    for (draft, expected) in [
        (None, None),
        (Some(Arc::from(&b"bad GGUF"[..])), None),
        (Some(header("dspark")), None),
        (Some(fixture::draft(0, 2)), Some((0, 2))),
        (Some(fixture::draft(1, 3)), Some((1, 3))),
    ] {
        let mut parts = ModelBytes::text(fixture::primary());
        parts.draft_model = draft;
        for engine in parts_engines(parts, config()) {
            selected(&engine, expected);
        }
    }
}

#[cfg(feature = "mmap")]
#[test]
fn bytes_then_config_then_manifest_draft_precedence_is_executable() {
    use std::fs;
    let dir = tempfile::tempdir().unwrap();
    let primary = dir.path().join("primary.gguf");
    fs::write(&primary, fixture::primary()).unwrap();
    let config_path = dir.path().join("config.gguf");
    let manifest_path = dir.path().join("manifest.gguf");
    let broken_path = dir.path().join("broken.gguf");
    let missing_path = dir.path().join("missing.gguf");
    fs::write(&config_path, fixture::draft(1, 3)).unwrap();
    fs::write(&manifest_path, fixture::draft(0, 2)).unwrap();
    fs::write(&broken_path, header("dspark")).unwrap();
    for (cfg_path, expected_path) in [
        (None, Some((0, 2))),
        (Some(config_path.clone()), Some((1, 3))),
        (Some(broken_path), None),
        (Some(missing_path), None),
    ] {
        let cfg = LoadConfig {
            draft_model: cfg_path.clone(),
            ..config()
        };
        let mut files = crate::ModelFiles::text(&primary);
        files.draft_model = Some(manifest_path.clone());
        for engine in file_engines(files, cfg.clone()) {
            selected(&engine, expected_path);
        }
        let mut manifest = super::super::filesystem::manifest("primary.gguf");
        manifest["load_time_parameters"]["draft_model"] = "manifest.gguf".into();
        let json = super::super::filesystem::write_manifest(dir.path(), &manifest);
        for path in [&json, dir.path()] {
            for engine in path_engines(path, cfg.clone()) {
                selected(&engine, expected_path);
            }
        }
        let fallback = if cfg_path == Some(config_path.clone()) {
            Some((1, 3))
        } else {
            None
        };
        for (bytes, expected) in [
            (Some(fixture::draft(0, 2)), Some((0, 2))),
            (None, fallback),
            (Some(header("dspark")), fallback),
            (Some(Arc::from(&b"invalid"[..])), fallback),
        ] {
            let mut parts = ModelBytes::text(fixture::primary());
            parts.draft_model = bytes;
            for engine in parts_engines(parts, cfg.clone()) {
                selected(&engine, expected);
            }
        }
    }
    // Byte and reader constructors also honor an explicitly supplied draft path.
    let cfg = LoadConfig {
        draft_model: Some(config_path),
        ..config()
    };
    selected(
        &CeraEngine::from_bytes(fixture::primary(), cfg.clone()).unwrap(),
        Some((1, 3)),
    );
    selected(
        &CeraEngine::from_reader(Cursor::new(fixture::primary()), cfg.clone()).unwrap(),
        Some((1, 3)),
    );
    for source in [
        ModelSource::bytes(fixture::primary()),
        ModelSource::reader(Cursor::new(fixture::primary())),
    ] {
        selected(
            &ModelLoader::new(source)
                .config(cfg.clone())
                .build_generative()
                .unwrap()
                .engine,
            Some((1, 3)),
        );
    }
}

#[cfg(not(feature = "mmap"))]
#[test]
fn draft_paths_are_inert_without_mmap_but_valid_bytes_still_execute() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid-draft.gguf");
    std::fs::write(&path, fixture::draft(1, 3)).unwrap();
    let cfg = LoadConfig {
        draft_model: Some(path),
        ..config()
    };
    for (bytes, expected) in [
        (None, None),
        (Some(header("dspark")), None),
        (Some(fixture::draft(0, 2)), Some((0, 2))),
    ] {
        let mut parts = ModelBytes::text(fixture::primary());
        parts.draft_model = bytes;
        for engine in parts_engines(parts, cfg.clone()) {
            selected(&engine, expected);
        }
    }
}

#[test]
fn sessions_execute_loaded_drafts_after_parent_release() {
    let mut parts = ModelBytes::text(fixture::primary());
    parts.draft_model = Some(fixture::draft(1, 3));
    let (engines, _source_directory) = executable_engines(parts);
    for mut engine in engines {
        let calls = observe_drafter(&mut engine);
        let mut a = engine.new_session(SessionConfig::default()).unwrap();
        let mut b = engine.new_session(SessionConfig::default()).unwrap();
        drop(engine);
        let mut ca = CeraEngine::from_bytes(fixture::primary(), config())
            .unwrap()
            .new_session(SessionConfig::default())
            .unwrap();
        let mut cb = CeraEngine::from_bytes(fixture::primary(), config())
            .unwrap()
            .new_session(SessionConfig::default())
            .unwrap();
        a.append_tokens(&[0]).unwrap();
        ca.append_tokens(&[0]).unwrap();
        b.append_tokens(&[1, 0, 0]).unwrap();
        cb.append_tokens(&[1, 0, 0]).unwrap();
        for (session, control) in [(&mut a, &mut ca), (&mut b, &mut cb)] {
            calls.lock().unwrap().clear();
            assert_eq!(generate(session), generate(control));
            assert!(
                !calls.lock().unwrap().is_empty(),
                "generation bypassed the loaded drafter"
            );
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|tokens| tokens == &[1, 1, 1])
            );
            session.append_tokens(&[1]).unwrap();
            control.append_tokens(&[1]).unwrap();
            assert_eq!(session.position(), control.position());
            close(
                session.last_logits().unwrap(),
                control.last_logits().unwrap(),
            );
        }
        a.reset().unwrap();
        ca.reset().unwrap();
        assert_eq!(generate(&mut a), generate(&mut ca));
        assert_eq!(generate(&mut b), generate(&mut cb));
    }
}

#[test]
fn real_dspark_session_state_is_history_sensitive_and_isolated() {
    use crate::model::dspark::{DSparkDraftModel, DSparkSessionDrafter};
    let model = || {
        Arc::new(
            DSparkDraftModel::from_ggufs(
                Arc::new(GgufFile::from_bytes(fixture::draft(1, 3)).unwrap()),
                Arc::new(GgufFile::from_bytes(fixture::primary()).unwrap()),
            )
            .unwrap(),
        )
    };
    let shared = model();
    let mut a = DSparkSessionDrafter::new(shared.clone());
    let mut b = DSparkSessionDrafter::new(shared.clone());
    let mut ca = DSparkSessionDrafter::new(model());
    let mut cb = DSparkSessionDrafter::new(model());
    drop(shared);
    // Same length and final token: differences must depend on earlier context.
    let ah = a.prepare_draft_step(&[0, 0, 1]).unwrap().to_vec();
    let bh = b.prepare_draft_step(&[1, 0, 1]).unwrap().to_vec();
    distinct(&ah, &bh);
    close(&ah, ca.prepare_draft_step(&[0, 0, 1]).unwrap());
    close(&bh, cb.prepare_draft_step(&[1, 0, 1]).unwrap());
    // The large Markov head deliberately makes token selection stable. Hidden
    // states, before that head, supply a numerical guard against shared KV.
    a.draft(&[0, 0, 1], 3);
    ca.draft(&[0, 0, 1], 3);
    close(
        a.prepare_draft_step(&[0, 0, 1, 0]).unwrap(),
        ca.prepare_draft_step(&[0, 0, 1, 0]).unwrap(),
    );
    a.reset();
    ca.reset();
    close(
        b.prepare_draft_step(&[1, 0, 1, 1]).unwrap(),
        cb.prepare_draft_step(&[1, 0, 1, 1]).unwrap(),
    );
    close(
        a.prepare_draft_step(&[1, 1]).unwrap(),
        ca.prepare_draft_step(&[1, 1]).unwrap(),
    );
}

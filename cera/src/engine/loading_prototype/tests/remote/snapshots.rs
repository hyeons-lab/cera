//! Direct GGUF discovery binds files and cache entries to a resolved commit.

use super::super::{auxiliary::close, companion_fixture};
use super::*;
use http::{Response, isolated_with_routes};
use std::collections::HashMap;

const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const FILE: &str = "nested/model-F32.gguf";

fn metadata(sha: Option<&str>) -> Response {
    Response::bytes(
        json!({"id":"fixture/snapshot", "sha":sha,
            "siblings":[{"rfilename":FILE},{"rfilename":"generation_config.json"}]})
        .to_string(),
    )
    .authenticated("cera-loopback-fixture")
}

fn weights(changed: bool) -> Arc<[u8]> {
    let source = companion_fixture::primary();
    if !changed {
        return source;
    }
    let gguf = GgufFile::from_bytes(source.clone()).unwrap();
    let mut bytes = source.to_vec();
    for tensor in gguf.tensors.values() {
        let start = tensor.offset as usize;
        let (values, tail) = bytes[start..start + tensor.size_bytes].as_chunks_mut::<4>();
        assert!(tail.is_empty());
        for value in values {
            let number = f32::from_le_bytes(*value);
            value.copy_from_slice(&(number * 0.5).to_le_bytes());
        }
    }
    assert_ne!(bytes.as_slice(), source.as_ref());
    bytes.into()
}

fn routes(sequence: Vec<Response>) -> HashMap<String, Response> {
    let mut routes = HashMap::from([(
        "/api/models/fixture/snapshot".into(),
        Response::sequence(sequence),
    )]);
    // Mutable URLs serve B even when metadata resolves A.
    for (rev, changed) in [(A, false), (B, true), ("main", true)] {
        routes.insert(
            format!("/fixture/snapshot/resolve/{rev}/{FILE}"),
            Response::gguf(weights(changed)).authenticated("cera-loopback-fixture"),
        );
        routes.insert(
            format!("/fixture/snapshot/resolve/{rev}/generation_config.json"),
            Response::bytes(json!({"temperature":if changed {0.75} else {0.25}}).to_string())
                .authenticated("cera-loopback-fixture"),
        );
    }
    routes
}

fn cfg(root: &Path, progress: &Arc<Progress>) -> LoadConfig {
    LoadConfig {
        context_size: 256,
        ..config(root, progress)
    }
}

fn check_model(engine: &CeraEngine, changed: bool) {
    assert_eq!(
        fs::read(&engine.manifest().files.model).unwrap(),
        weights(changed).as_ref()
    );
    assert_eq!(
        engine.default_generate_opts().temperature,
        if changed { 0.75 } else { 0.25 }
    );
    let mut actual = engine.new_session(SessionConfig::default()).unwrap();
    let mut expected = control(changed);
    for tokens in [&[0, 1][..], &[1, 0, 1][..]] {
        actual.append_tokens(tokens).unwrap();
        expected.append_tokens(tokens).unwrap();
        close(
            actual.last_logits().unwrap(),
            expected.last_logits().unwrap(),
        );
    }
}

fn control(changed: bool) -> crate::session::Session {
    CeraEngine::from_bytes(
        weights(changed),
        LoadConfig {
            context_size: 256,
            ..cpu_config()
        },
    )
    .unwrap()
    .new_session(SessionConfig::default())
    .unwrap()
}

#[test]
fn moved_hf_branch_uses_distinct_snapshots_and_keeps_live_session() {
    isolated_with_routes(
        "snapshots::moved_hf_branch_uses_distinct_snapshots_and_keeps_live_session",
        || {
            routes(vec![
                metadata(Some(A)),
                metadata(Some(B)),
                metadata(Some(B)),
                metadata(Some(A)),
            ])
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            // A pre-existing branch entry must not satisfy a resolved commit URL.
            let old = seed(
                repo,
                &format!("{}/fixture/snapshot/resolve/main/{FILE}", ctx.url),
                weights(true),
            );
            let first = load(
                ModelSource::hugging_face("fixture/snapshot:F32", None, None),
                cfg.clone(),
            );
            check_model(&first.engine, false);
            let a_path = repo
                .fixture_path(&format!("{}/fixture/snapshot/resolve/{A}/{FILE}", ctx.url))
                .unwrap();
            assert_eq!(Path::new(&first.engine.manifest().files.model), a_path);
            let mut retained = first.create_session(SessionConfig::default()).unwrap();
            let mut expected = control(false);
            retained.append_tokens(&[0, 1]).unwrap();
            expected.append_tokens(&[0, 1]).unwrap();
            drop(first);

            let second = CeraEngine::from_hf("fixture/snapshot:F32", None, cfg.clone()).unwrap();
            check_model(&second, true);
            let b_path = PathBuf::from(&second.manifest().files.model);
            assert_ne!(a_path, b_path);
            let events = progress.0.lock().unwrap().len();
            let handle = ModelLoader::new(ModelSource::hugging_face(
                "fixture/snapshot:F32",
                None,
                None,
            ))
            .config(cfg.clone())
            .build()
            .unwrap();
            check_model(&handle.as_generative().unwrap().engine, true);
            assert_eq!(progress.0.lock().unwrap().len(), events);
            let restored =
                CeraEngine::from_hf_url(&format!("{}/fixture/snapshot", ctx.url), Some("F32"), cfg)
                    .unwrap();
            check_model(&restored, false);
            assert_eq!(progress.0.lock().unwrap().len(), events);
            assert_eq!(fs::read(&old).unwrap(), weights(true).as_ref());
            assert_eq!(fs::read(&a_path).unwrap(), weights(false).as_ref());
            assert_eq!(fs::read(&b_path).unwrap(), weights(true).as_ref());
            retained.append_tokens(&[1, 0, 1]).unwrap();
            expected.append_tokens(&[1, 0, 1]).unwrap();
            assert_eq!(retained.position(), 5);
            close(
                retained.last_logits().unwrap(),
                expected.last_logits().unwrap(),
            );
        },
        |requests| {
            assert_eq!(count(requests, "GET", "/api/models/fixture/snapshot"), 4);
            for rev in [A, B] {
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/fixture/snapshot/resolve/{rev}/{FILE}")
                    ),
                    1
                );
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/fixture/snapshot/resolve/{rev}/generation_config.json")
                    ),
                    2
                );
            }
            assert!(
                !requests
                    .iter()
                    .any(|(_, path)| path.contains("/resolve/main/"))
            );
        },
    );
}

#[test]
fn invalid_hf_snapshot_cannot_reuse_downloaded_model() {
    isolated_with_routes(
        "snapshots::invalid_hf_snapshot_cannot_reuse_downloaded_model",
        || {
            routes(vec![
                metadata(Some(A)),
                metadata(None),
                metadata(Some("short")),
                metadata(Some(&"z".repeat(40))),
                Response::status(404),
                metadata(Some(A)),
            ])
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let first = CeraEngine::from_hf("fixture/snapshot:F32", None, cfg.clone()).unwrap();
            let path = first.manifest().files.model.clone();
            let bytes = fs::read(&path).unwrap();
            let events = progress.0.lock().unwrap().len();
            for _ in 0..4 {
                let error = engine_error(
                    ModelLoader::new(ModelSource::hugging_face(
                        "fixture/snapshot:F32",
                        None,
                        None,
                    ))
                    .config(cfg.clone())
                    .build(),
                    "hf",
                );
                assert!(matches!(error, CeraError::Backend(_)));
                assert_eq!(fs::read(&path).unwrap(), bytes);
                assert_eq!(progress.0.lock().unwrap().len(), events);
            }
            check_model(
                &CeraEngine::from_hf("fixture/snapshot:F32", None, cfg).unwrap(),
                false,
            );
            assert_eq!(progress.0.lock().unwrap().len(), events);
        },
        |requests| {
            assert_eq!(count(requests, "GET", "/api/models/fixture/snapshot"), 6);
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/snapshot/resolve/{A}/{FILE}")
                ),
                1
            );
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/snapshot/resolve/{A}/generation_config.json")
                ),
                2
            );
            assert!(
                !requests
                    .iter()
                    .any(|(_, path)| path.contains("/resolve/main/"))
            );
        },
    );
}

#[test]
fn explicit_hf_snapshot_checks_commit_and_preserves_subpath() {
    isolated_with_routes(
        "snapshots::explicit_hf_snapshot_checks_commit_and_preserves_subpath",
        || {
            let mut routes = routes(vec![metadata(None)]);
            routes.insert(
                format!(
                    "/api/models/fixture/snapshot/revision/{}",
                    A.to_ascii_uppercase()
                ),
                metadata(Some(A)),
            );
            routes.insert(
                format!("/api/models/fixture/snapshot/revision/{B}"),
                metadata(Some(A)),
            );
            routes
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let url = format!(
                "{}/fixture/snapshot/resolve/{}/{FILE}",
                ctx.url,
                A.to_ascii_uppercase()
            );
            // The explicit file wins over a conflicting quant preference.
            check_model(
                &CeraEngine::from_hf_url(&url, Some("Q8_0"), cfg.clone()).unwrap(),
                false,
            );
            let events = progress.0.lock().unwrap().len();
            let error = CeraEngine::from_hf(&format!("fixture/snapshot@{B}"), None, cfg.clone())
                .err()
                .unwrap();
            assert!(
                matches!(error, CeraError::Backend(message) if message.contains("does not match"))
            );
            assert_eq!(progress.0.lock().unwrap().len(), events);
            let b_path = cfg
                .bundle_repo
                .as_ref()
                .unwrap()
                .fixture_path(&format!("{}/fixture/snapshot/resolve/{B}/{FILE}", ctx.url))
                .unwrap();
            assert!(!b_path.exists());
            // Standalone metadata inspection remains permissive about absent SHA.
            let info = crate::bundle::hf::fetch_model_info(
                &crate::bundle::HfSpec::parse("fixture/snapshot").unwrap(),
            )
            .unwrap();
            assert_eq!(info.siblings.len(), 2);
        },
        |requests| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!(
                        "/api/models/fixture/snapshot/revision/{}",
                        A.to_ascii_uppercase()
                    )
                ),
                1
            );
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/api/models/fixture/snapshot/revision/{B}")
                ),
                1
            );
            assert_eq!(count(requests, "GET", "/api/models/fixture/snapshot"), 1);
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/snapshot/resolve/{A}/{FILE}")
                ),
                1
            );
            assert!(
                !requests
                    .iter()
                    .any(|(_, path)| path.contains(&format!("/resolve/{B}/")))
            );
        },
    );
}

#[test]
fn external_dspark_snapshot_refreshes_independently_of_primary() {
    use super::super::auxiliary::observe_drafter;

    const PRIMARY: &str = "fixture/LFM2.5-2.6B";
    const DRAFT: &str = "LiquidAI/LFM2.5-2.6B-DSpark-GGUF";
    const DRAFT_FILE: &str = "LFM2.5-2.6B-DSpark-Q4_K_M.gguf";
    isolated_with_routes(
        "snapshots::external_dspark_snapshot_refreshes_independently_of_primary",
        || {
            let draft_metadata = |sha: Option<&str>| {
                Response::bytes(
                    json!({"id":DRAFT,"sha":sha,
                    "siblings":[{"rfilename":DRAFT_FILE}]})
                    .to_string(),
                )
                .authenticated("cera-loopback-fixture")
            };
            let mut routes = HashMap::from([
                (
                    format!("/api/models/{PRIMARY}"),
                    Response::bytes(
                        json!({"id":PRIMARY,"sha":MAIN_COMMIT,
                    "siblings":[{"rfilename":"model-Q4_K_M.gguf"}]})
                        .to_string(),
                    ),
                ),
                (
                    format!("/{PRIMARY}/resolve/{MAIN_COMMIT}/model-Q4_K_M.gguf"),
                    Response::gguf(companion_fixture::primary()),
                ),
                (
                    format!("/api/models/{DRAFT}"),
                    Response::sequence(vec![
                        draft_metadata(Some(A)),
                        draft_metadata(Some(B)),
                        draft_metadata(Some(B)),
                        draft_metadata(None),
                        draft_metadata(Some(A)),
                    ]),
                ),
            ]);
            for (revision, token, k) in [(A, 0, 2), (B, 1, 3), ("main", 1, 3)] {
                routes.insert(
                    format!("/{DRAFT}/resolve/{revision}/{DRAFT_FILE}"),
                    Response::gguf(companion_fixture::draft(token, k))
                        .authenticated("cera-loopback-fixture"),
                );
            }
            routes
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let old = seed(
                repo,
                &format!("{}/{DRAFT}/resolve/main/{DRAFT_FILE}", ctx.url),
                companion_fixture::draft(1, 3),
            );
            let mut retained = None;
            let mut events_after_b = 0;
            for (index, revision, token, k) in
                [(0, A, 0, 2), (1, B, 1, 3), (2, B, 1, 3), (3, A, 0, 2)]
            {
                if index == 3 {
                    let error = CeraEngine::from_hf(PRIMARY, Some("Q4_K_M"), cfg.clone())
                        .err()
                        .unwrap();
                    assert!(matches!(error, CeraError::Backend(_)));
                    assert_eq!(progress.0.lock().unwrap().len(), events_after_b);
                }
                let mut engine = match index {
                    0 => Arc::try_unwrap(
                        load(
                            ModelSource::hugging_face(PRIMARY, Some("Q4_K_M"), None),
                            cfg.clone(),
                        )
                        .engine,
                    )
                    .ok()
                    .unwrap(),
                    2 => {
                        let handle = ModelLoader::new(ModelSource::hugging_face(
                            PRIMARY,
                            Some("Q4_K_M"),
                            None,
                        ))
                        .config(cfg.clone())
                        .build()
                        .unwrap();
                        let model = handle.as_generative().unwrap();
                        drop(handle);
                        Arc::try_unwrap(model.engine).ok().unwrap()
                    }
                    _ => CeraEngine::from_hf(PRIMARY, Some("Q4_K_M"), cfg.clone()).unwrap(),
                };
                let path = repo
                    .fixture_path(&format!(
                        "{}/{DRAFT}/resolve/{revision}/{DRAFT_FILE}",
                        ctx.url
                    ))
                    .unwrap();
                assert_eq!(
                    fs::read(engine.manifest().files.draft_model.as_ref().unwrap()).unwrap(),
                    companion_fixture::draft(token, k).as_ref()
                );
                assert_eq!(
                    Path::new(engine.manifest().files.draft_model.as_ref().unwrap()),
                    path
                );
                let draft_count = usize::try_from(k).unwrap();
                assert_eq!(
                    engine.drafter.as_ref().unwrap().suggested_k(),
                    Some(draft_count)
                );
                let calls = observe_drafter(&mut engine);
                let mut session = engine.new_session(SessionConfig::default()).unwrap();
                drop(engine);
                if index == 0 {
                    retained = Some((session, calls));
                } else {
                    assert_eq!(generate(&mut session), generate(&mut control(false)));
                    let proposals = calls.lock().unwrap();
                    assert!(!proposals.is_empty());
                    assert!(
                        proposals
                            .iter()
                            .all(|tokens| tokens == &vec![token; draft_count])
                    );
                }
                if index == 1 {
                    events_after_b = progress.0.lock().unwrap().len();
                } else if index >= 2 {
                    assert_eq!(progress.0.lock().unwrap().len(), events_after_b);
                }
            }
            let (mut session, calls) = retained.unwrap();
            assert_eq!(generate(&mut session), generate(&mut control(false)));
            let proposals = calls.lock().unwrap();
            assert!(!proposals.is_empty());
            assert!(proposals.iter().all(|tokens| tokens == &[0, 0]));
            assert_eq!(
                fs::read(old).unwrap(),
                companion_fixture::draft(1, 3).as_ref()
            );
        },
        |requests| {
            assert_eq!(count(requests, "GET", &format!("/api/models/{PRIMARY}")), 5);
            assert_eq!(count(requests, "GET", &format!("/api/models/{DRAFT}")), 5);
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/{PRIMARY}/resolve/{MAIN_COMMIT}/model-Q4_K_M.gguf")
                ),
                1
            );
            for revision in [A, B] {
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/{DRAFT}/resolve/{revision}/{DRAFT_FILE}")
                    ),
                    1
                );
            }
            assert!(
                !requests
                    .iter()
                    .any(|(_, path)| path.contains("/resolve/main/"))
            );
        },
    );
}

#[test]
fn prefixed_hf_endpoint_loads_co_located_and_external_drafts() {
    use super::super::auxiliary::observe_drafter;

    const PREFIX: &str = "/hub/mirror";
    const EXTERNAL: &str = "LiquidAI/LFM2.5-2.6B-DSpark-GGUF";
    const EXTERNAL_FILE: &str = "LFM2.5-2.6B-DSpark-Q4_K_M.gguf";
    http::isolated_with_endpoint_path(
        "snapshots::prefixed_hf_endpoint_loads_co_located_and_external_drafts",
        PREFIX,
        || {
            let mut routes = HashMap::new();
            for (repo, colocated) in [("colocated", true), ("LFM2.5-2.6B", false)] {
                let mut files = vec![json!({"rfilename":"model-Q4_K_M.gguf"})];
                if colocated {
                    files.push(json!({"rfilename":"draft-Q4_K_M.gguf"}));
                    routes.insert(
                        format!("{PREFIX}/fixture/{repo}/resolve/{MAIN_COMMIT}/draft-Q4_K_M.gguf"),
                        Response::gguf(companion_fixture::draft(0, 2)),
                    );
                }
                routes.insert(
                    format!("{PREFIX}/api/models/fixture/{repo}"),
                    Response::bytes(
                        json!({"id":format!("fixture/{repo}"),"sha":MAIN_COMMIT,"siblings":files})
                            .to_string(),
                    ),
                );
                routes.insert(
                    format!("{PREFIX}/fixture/{repo}/resolve/{MAIN_COMMIT}/model-Q4_K_M.gguf"),
                    Response::gguf(companion_fixture::primary()),
                );
            }
            routes.insert(
                format!("{PREFIX}/api/models/{EXTERNAL}"),
                Response::bytes(
                    json!({"id":EXTERNAL,"sha":A,"siblings":[{"rfilename":EXTERNAL_FILE}]})
                        .to_string(),
                ),
            );
            routes.insert(
                format!("{PREFIX}/{EXTERNAL}/resolve/{A}/{EXTERNAL_FILE}"),
                Response::gguf(companion_fixture::draft(1, 3)),
            );
            routes
        },
        |ctx| {
            assert!(ctx.url.ends_with(PREFIX));
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            for (repo, draft_path, token, k) in [
                (
                    "colocated",
                    format!("fixture/colocated/resolve/{MAIN_COMMIT}/draft-Q4_K_M.gguf"),
                    0,
                    2,
                ),
                (
                    "LFM2.5-2.6B",
                    format!("{EXTERNAL}/resolve/{A}/{EXTERNAL_FILE}"),
                    1,
                    3,
                ),
            ] {
                let mut engine =
                    CeraEngine::from_hf(&format!("fixture/{repo}"), Some("Q4_K_M"), cfg.clone())
                        .unwrap();
                let path = cfg
                    .bundle_repo
                    .as_ref()
                    .unwrap()
                    .fixture_path(&format!("{}/{draft_path}", ctx.url))
                    .unwrap();
                assert_eq!(
                    Path::new(engine.manifest().files.draft_model.as_ref().unwrap()),
                    path
                );
                assert_eq!(
                    fs::read(path).unwrap(),
                    companion_fixture::draft(token, k).as_ref()
                );
                let calls = observe_drafter(&mut engine);
                let mut session = engine.new_session(SessionConfig::default()).unwrap();
                drop(engine);
                assert_eq!(generate(&mut session), generate(&mut control(false)));
                let proposals = calls.lock().unwrap();
                assert!(!proposals.is_empty());
                let draft_count = usize::try_from(k).unwrap();
                assert!(
                    proposals
                        .iter()
                        .all(|tokens| tokens == &vec![token; draft_count])
                );
            }
        },
        |requests| {
            assert!(
                requests
                    .iter()
                    .all(|(_, path)| path.starts_with(&format!("{PREFIX}/")))
            );
            for repo in ["fixture/colocated", "fixture/LFM2.5-2.6B", EXTERNAL] {
                assert_eq!(
                    count(requests, "GET", &format!("{PREFIX}/api/models/{repo}")),
                    1
                );
            }
            assert_eq!(
                requests
                    .iter()
                    .filter(|(method, path)| method == "GET" && path.ends_with(".gguf"))
                    .count(),
                4
            );
            assert!(
                !requests
                    .iter()
                    .any(|(_, path)| path.contains("/resolve/main/"))
            );
        },
    );
}

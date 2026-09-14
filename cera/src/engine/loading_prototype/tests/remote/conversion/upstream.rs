use super::*;
use crate::bundle::HfSpec;
use crate::convert::{QuantizeOptions, TargetQuant, stream_quantize_hf_repo};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const A: &str = "1111111111111111111111111111111111111111";
const B: &str = "2222222222222222222222222222222222222222";

fn metadata(repo: &str, sha: Option<&str>) -> Response {
    Response::bytes(
        json!({"id":format!("fixture/{repo}"), "sha":sha,
        "siblings":[{"rfilename":"model.safetensors"}]})
        .to_string(),
    )
}

fn halve(bytes: &mut [u8]) {
    let (values, tail) = bytes.as_chunks_mut::<4>();
    assert!(
        tail.is_empty(),
        "fixture F32 payload must contain complete values"
    );
    for value in values {
        let number = f32::from_le_bytes(*value);
        value.copy_from_slice(&(number * 0.5).to_le_bytes());
    }
}

fn changed_source() -> Arc<[u8]> {
    let source = fixture::source();
    let mut bytes = source.mmap_data().to_vec();
    for tensor in source.tensors.values() {
        let start = tensor.offset as usize;
        halve(&mut bytes[start..start + tensor.size_bytes]);
    }
    bytes.into()
}

fn sources(repo: &str) -> HashMap<String, Response> {
    let original = fixture::shards(false);
    let mut changed = original.clone();
    let bytes = &mut changed[0].1;
    let start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    halve(&mut bytes[start..]);
    assert_eq!(&original[0].1[..start], &bytes[..start]); // identical source header/layout
    let mut routes = fixture::routes(repo, A, &original, true);
    routes.extend(fixture::routes(repo, B, &changed, true));
    // Mutable input URLs deliberately serve different bytes than commit A.
    let aliases: Vec<_> = routes
        .iter()
        .filter_map(|(key, value)| {
            let prefix = format!("/fixture/{repo}/resolve/{B}/");
            key.strip_prefix(&prefix).map(|tail| {
                (
                    format!("/fixture/{repo}/resolve/main/{tail}"),
                    value.clone(),
                )
            })
        })
        .collect();
    routes.extend(aliases);
    routes
}

fn options(cfg: &LoadConfig, progress: Arc<dyn DownloadProgress>) -> QuantizeOptions {
    QuantizeOptions {
        target_quant: TargetQuant::F32,
        cache_dir: cfg.bundle_repo.as_ref().unwrap().store_dir().into(),
        auth_token: None,
        progress: Some(progress),
        ..Default::default()
    }
}

fn check_weights(path: &Path, changed: bool) {
    let actual = GgufFile::open(path).unwrap();
    let bytes = if changed {
        changed_source()
    } else {
        super::super::super::companion_fixture::primary()
    };
    let expected = GgufFile::from_bytes(bytes.clone()).unwrap();
    assert_eq!(actual.tensors.len(), expected.tensors.len());
    for (name, tensor) in &expected.tensors {
        assert_eq!(actual.tensors[name].shape, tensor.shape);
        assert_eq!(
            actual.tensor_data(name).unwrap(),
            expected.tensor_data(name).unwrap(),
            "{name}"
        );
    }
    execute(CeraEngine::from_path(path, cpu_config()).unwrap(), bytes);
}

#[test]
fn conversion_pins_every_input_to_resolved_commit() {
    isolated_with_ranges(
        "conversion::upstream::conversion_pins_every_input_to_resolved_commit",
        || {
            let mut routes = sources("pin");
            routes.insert("/api/models/fixture/pin".into(), metadata("pin", Some(A)));
            routes
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let model = load(
                ModelSource::hugging_face("fixture/pin:F32", None, None),
                cfg.clone(),
            );
            let path = cache_dir(&cfg, "pin", "F32").join("model.gguf");
            check_weights(&path, false);
            execute(
                Arc::try_unwrap(model.engine).ok().unwrap(),
                super::super::super::companion_fixture::primary(),
            );
            let receipt_path = path.with_extension("gguf.receipt.json");
            let mut receipt: serde_json::Value =
                serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
            assert_eq!(receipt["request"]["resolved_revision"], A);
            assert_eq!(receipt["request"]["revision"], "main");
            // Legacy completed records have no resolved source; rebuild exactly once.
            receipt["request"]
                .as_object_mut()
                .unwrap()
                .remove("resolved_revision");
            receipt["request"]["version"] = json!(1);
            fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
            let spec = HfSpec::parse("fixture/pin:F32").unwrap();
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 24);
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 24);
            check_weights(&path, false);
        },
        |requests, ranges| {
            assert!(!requests.iter().any(|(_, p)| p.contains("/resolve/main/")));
            assert!(
                ranges
                    .iter()
                    .all(|(p, _)| p.contains(&format!("/resolve/{A}/")))
            );
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
            ] {
                assert_eq!(
                    count(requests, "GET", &format!("/fixture/pin/resolve/{A}/{name}")),
                    2
                );
            }
        },
    );
}

#[derive(Debug)]
struct StopAfterFive {
    cancel: Arc<AtomicBool>,
    calls: AtomicUsize,
}
impl DownloadProgress for StopAfterFive {
    fn on_progress(&self, _: &str, _: u64, _: Option<u64>) {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 4 {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }
}

#[test]
fn moved_branch_restarts_checkpoint_and_refreshes_completed_cache() {
    isolated_with_ranges(
        "conversion::upstream::moved_branch_restarts_checkpoint_and_refreshes_completed_cache",
        || {
            let mut routes = sources("move");
            routes.insert(
                "/api/models/fixture/move".into(),
                Response::sequence(vec![
                    metadata("move", Some(A)),
                    metadata("move", Some(B)),
                    metadata("move", Some(B)),
                    metadata("move", Some(A)),
                    metadata("move", Some(A)),
                ]),
            );
            routes
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse("fixture/move:F32").unwrap();
            let cancel = Arc::new(AtomicBool::new(false));
            let stopper = Arc::new(StopAfterFive {
                cancel: cancel.clone(),
                calls: AtomicUsize::new(0),
            });
            let mut opts = options(&cfg, stopper.clone());
            opts.cancel = Some(cancel);
            assert!(matches!(
                stream_quantize_hf_repo(&spec, opts),
                Err(CeraError::Cancelled)
            ));
            assert_eq!(stopper.calls.load(Ordering::Relaxed), 5);
            let path = cache_dir(&cfg, "move", "F32").join("model.gguf");
            let checkpoint: serde_json::Value = serde_json::from_slice(
                &fs::read(path.with_extension("gguf.checkpoint.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(checkpoint["request"]["resolved_revision"], A);
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 12); // restart all eleven, same layout
            check_weights(&path, true);
            let retained = CeraEngine::from_path(&path, cpu_config()).unwrap();
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 12); // same B reuses
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 24); // A refreshes completed B
            check_weights(&path, false);
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 24); // same A reuses
            execute(retained, changed_source());
        },
        |requests, ranges| {
            assert_eq!(count(requests, "GET", "/api/models/fixture/move"), 5);
            for (commit, expected) in [(A, 20), (B, 13)] {
                let path = format!("/fixture/move/resolve/{commit}/model.safetensors");
                assert_eq!(count(requests, "GET", &path), expected);
            }
            assert!(ranges.iter().all(|(p, _)| !p.contains("/resolve/main/")));
        },
    );
}

#[test]
fn invalid_or_unavailable_revision_does_not_reuse_completed_output() {
    isolated_with_ranges(
        "conversion::upstream::invalid_or_unavailable_revision_does_not_reuse_completed_output",
        || {
            let mut routes = sources("reject");
            routes.insert(
                "/api/models/fixture/reject".into(),
                Response::sequence(vec![
                    metadata("reject", Some(A)),
                    metadata("reject", None),
                    metadata("reject", Some("bad")),
                    metadata("reject", Some(&"z".repeat(40))),
                    Response::status(404),
                    metadata("reject", Some(A)),
                ]),
            );
            routes
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse("fixture/reject:F32").unwrap();
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            let path = cache_dir(&cfg, "reject", "F32").join("model.gguf");
            let before = fs::read(&path).unwrap();
            for _ in 0..4 {
                assert!(matches!(
                    stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())),
                    Err(CeraError::Backend(_))
                ));
                assert_eq!(fs::read(&path).unwrap(), before);
                assert_eq!(progress.0.lock().unwrap().len(), 12);
            }
            stream_quantize_hf_repo(&spec, options(&cfg, progress.clone())).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), 12); // recovered metadata, reuse A
            check_weights(&path, false);
        },
        |requests, _| {
            assert_eq!(count(requests, "GET", "/api/models/fixture/reject"), 6);
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/reject/resolve/{A}/model.safetensors")
                ),
                13
            );
        },
    );
}

#[test]
fn explicit_revision_and_authentication_pin_inputs_and_reject_wrong_commit() {
    isolated_with_ranges(
        "conversion::upstream::explicit_revision_and_authentication_pin_inputs_and_reject_wrong_commit",
        || {
            let mut routes = sources("explicit");
            routes.insert(
                "/api/models/fixture/explicit/revision/release".into(),
                metadata("explicit", Some(A)).authenticated("cera-explicit-fixture"),
            );
            routes.insert(
                format!("/api/models/fixture/explicit/revision/{A}"),
                metadata("explicit", Some(B)),
            );
            routes.insert(
                format!("/api/models/fixture/explicit/revision/{B}"),
                metadata("explicit", Some(B)).authenticated("cera-explicit-fixture"),
            );
            for (path, response) in &mut routes {
                if path.contains("/resolve/") {
                    *response = response.clone().authenticated("cera-explicit-fixture");
                }
            }
            routes
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let mut opts = options(&cfg, progress.clone());
            opts.auth_token = Some("cera-explicit-fixture".into());
            for (revision, changed) in [("release", false), (B, true)] {
                let spec = HfSpec::parse(&format!("fixture/explicit:F32@{revision}")).unwrap();
                let manifest = stream_quantize_hf_repo(&spec, opts.clone()).unwrap();
                check_weights(Path::new(&manifest.files.model), changed);
                let receipt: serde_json::Value = serde_json::from_slice(
                    &fs::read(Path::new(&manifest.files.model).with_extension("gguf.receipt.json"))
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(receipt["request"]["revision"], revision);
                assert_eq!(
                    receipt["request"]["resolved_revision"],
                    if changed { B } else { A }
                );
            }
            let spec = HfSpec::parse(&format!("fixture/explicit:F32@{A}")).unwrap();
            let error = stream_quantize_hf_repo(&spec, opts).err().unwrap();
            assert!(
                error
                    .to_string()
                    .contains("does not match the requested commit"),
                "{error}"
            );
            assert!(!cache_dir(&cfg, "explicit", &format!("F32@{A}")).exists());
            assert_eq!(progress.0.lock().unwrap().len(), 24);
        },
        |requests, _| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    "/api/models/fixture/explicit/revision/release"
                ),
                1
            );
            assert!(
                !requests
                    .iter()
                    .any(|(_, p)| p.contains('?') || p.contains("/resolve/release/"))
            );
        },
    );
}

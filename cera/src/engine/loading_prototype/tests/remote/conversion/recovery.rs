use super::*;
use crate::bundle::HfSpec;
use crate::convert::{QuantStrategy, QuantizeOptions, TargetQuant, stream_quantize_hf_repo};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

const RECOVERY_CASES: &[(&str, bool)] = &[
    ("resume", false),
    ("restart", true),
    ("legacy", true),
    ("legacy-source", true),
    ("changed-options", true),
    ("payload-damage", true),
    ("header-damage", true),
    ("short-prefix", true),
    ("missing-digest", true),
    ("invalid-digest", true),
    ("wrong-digest", true),
    ("missing-layout", true),
    ("wrong-layout", true),
    ("wrong-count", true),
    ("wrong-boundary", true),
];

#[test]
fn interrupted_conversion_resumes_and_invalid_checkpoint_restarts() {
    isolated_with_ranges(
        "conversion::recovery::interrupted_conversion_resumes_and_invalid_checkpoint_restarts",
        || {
            RECOVERY_CASES
                .iter()
                .map(|(repo, _)| *repo)
                .flat_map(|repo| fixture::routes(repo, "main", &fixture::shards(false), true))
                .collect()
        },
        |ctx| {
            for &(repo, restart) in RECOVERY_CASES {
                let progress = Arc::new(Progress::default());
                let cfg = config(&ctx.root, &progress);
                let spec = HfSpec::parse(&format!("fixture/{repo}:F32")).unwrap();
                let cancel = Arc::new(AtomicBool::new(false));
                let stopper = Arc::new(StopAfterFive {
                    cancel: cancel.clone(),
                    calls: AtomicUsize::new(0),
                });
                // Prime the existing converter's recovery files. Loading itself has no cancel option.
                let result = stream_quantize_hf_repo(
                    &spec,
                    QuantizeOptions {
                        target_quant: TargetQuant::F32,
                        strategy: QuantStrategy::Auto,
                        cache_dir: cfg.bundle_repo.as_ref().unwrap().store_dir().to_path_buf(),
                        auth_token: None,
                        progress: Some(stopper.clone()),
                        cancel: Some(cancel),
                        tensor_overrides: if repo == "changed-options" {
                            vec![("*".into(), TargetQuant::F16)]
                        } else {
                            Vec::new()
                        },
                    },
                );
                assert!(matches!(result, Err(CeraError::Cancelled)), "{result:?}");
                assert_eq!(stopper.calls.load(Ordering::Relaxed), 5);
                let dir = cache_dir(&cfg, repo, "F32");
                let checkpoint_path = dir.join("model.gguf.checkpoint.json");
                let mut checkpoint: serde_json::Value =
                    serde_json::from_slice(&fs::read(&checkpoint_path).unwrap()).unwrap();
                assert_eq!(checkpoint["completed_tensors"], 5);
                assert_eq!(checkpoint["total_tensors"], 11);
                let completed_bytes = checkpoint["file_bytes"].as_u64().unwrap();
                let tmp = dir.join("model.gguf.tmp");
                assert_eq!(fs::metadata(&tmp).unwrap().len(), completed_bytes);
                assert!(!dir.join("model.gguf").exists());
                assert!(!dir.join("F32.json").exists());
                assert!(!dir.join("model.gguf.sha256").exists());
                // Extend beyond the complete fixture output, so later writes cannot
                // hide a missing truncate at the durable checkpoint boundary.
                fs::OpenOptions::new()
                    .append(true)
                    .open(&tmp)
                    .unwrap()
                    .write_all(&vec![0xa5; 128 * 1024])
                    .unwrap();
                match repo {
                    "restart" => checkpoint["total_tensors"] = json!(99),
                    "legacy" => {
                        checkpoint.as_object_mut().unwrap().remove("request");
                    }
                    "legacy-source" => {
                        checkpoint["request"]
                            .as_object_mut()
                            .unwrap()
                            .remove("resolved_revision");
                        checkpoint["request"]["version"] = json!(1);
                    }
                    "payload-damage" | "header-damage" => {
                        let mut bytes = fs::read(&tmp).unwrap();
                        let index = if repo == "header-damage" {
                            0
                        } else {
                            completed_bytes as usize - 64
                        };
                        bytes[index] ^= 0xff;
                        fs::write(&tmp, bytes).unwrap();
                    }
                    "short-prefix" => fs::OpenOptions::new()
                        .write(true)
                        .open(&tmp)
                        .unwrap()
                        .set_len(completed_bytes - 1)
                        .unwrap(),
                    "missing-digest" => {
                        checkpoint.as_object_mut().unwrap().remove("prefix_sha256");
                    }
                    "invalid-digest" => checkpoint["prefix_sha256"] = json!("invalid"),
                    "wrong-digest" => checkpoint["prefix_sha256"] = json!("0".repeat(64)),
                    "missing-layout" => {
                        checkpoint.as_object_mut().unwrap().remove("header_sha256");
                    }
                    "wrong-layout" => checkpoint["header_sha256"] = json!("0".repeat(64)),
                    "wrong-count" => checkpoint["completed_tensors"] = json!(4),
                    "wrong-boundary" => {
                        checkpoint["file_bytes"] = json!(0);
                        checkpoint["prefix_sha256"] = json!(digest(&[]));
                    }
                    _ => {}
                }
                fs::write(&checkpoint_path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
                let model = load(
                    ModelSource::hugging_face(format!("fixture/{repo}:F32"), None, None),
                    cfg,
                );
                let path = dir.join("model.gguf");
                check_output(&path, "F32");
                let events = progress.0.lock().unwrap();
                assert_eq!(events.len(), if restart { 12 } else { 7 }, "{repo}");
                if restart {
                    assert!(events[0].1 < completed_bytes);
                } else {
                    assert_eq!(events[0].1, completed_bytes);
                }
                assert!(events.windows(2).all(|p| p[0].1 < p[1].1));
                assert_eq!(events.last().unwrap().1, fs::metadata(&path).unwrap().len());
                execute(
                    Arc::try_unwrap(model.engine).ok().unwrap(),
                    super::super::super::companion_fixture::primary(),
                );
            }
        },
        |requests, ranges| {
            let shards = fixture::shards(false);
            let expected = fixture::ranges(&shards[0].1);
            for &(repo, restart) in RECOVERY_CASES {
                let path = format!(
                    "/fixture/{repo}/resolve/1111111111111111111111111111111111111111/model.safetensors"
                );
                for (index, range) in expected.iter().enumerate() {
                    let times = if index < 2 || (restart && index < 7) {
                        2
                    } else {
                        1
                    };
                    assert_eq!(
                        ranges
                            .iter()
                            .filter(|(t, r)| *t == path && r == range)
                            .count(),
                        times,
                        "{repo}: {range}"
                    );
                }
                assert_eq!(count(requests, "GET", &path), if restart { 20 } else { 15 });
            }
        },
    );
}

#[test]
fn repeated_interruption_preserves_the_entire_verified_prefix() {
    isolated_with_ranges(
        "conversion::recovery::repeated_interruption_preserves_the_entire_verified_prefix",
        || fixture::routes("repeat", "main", &fixture::shards(false), true),
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse("fixture/repeat:F32").unwrap();
            let dir = cache_dir(&cfg, "repeat", "F32");
            let tmp = dir.join("model.gguf.tmp");
            let mut previous_prefix = Vec::new();
            let mut previous_header_digest = None;
            for completed in [5, 10] {
                let cancel = Arc::new(AtomicBool::new(false));
                let stopper = Arc::new(StopAfterFive {
                    cancel: cancel.clone(),
                    calls: AtomicUsize::new(0),
                });
                let result = stream_quantize_hf_repo(
                    &spec,
                    QuantizeOptions {
                        target_quant: TargetQuant::F32,
                        strategy: QuantStrategy::Auto,
                        cache_dir: cfg.bundle_repo.as_ref().unwrap().store_dir().to_path_buf(),
                        auth_token: None,
                        progress: Some(stopper.clone()),
                        cancel: Some(cancel),
                        tensor_overrides: Vec::new(),
                    },
                );
                assert!(matches!(result, Err(CeraError::Cancelled)), "{result:?}");
                assert_eq!(stopper.calls.load(Ordering::Relaxed), 5);
                let checkpoint: serde_json::Value = serde_json::from_slice(
                    &fs::read(dir.join("model.gguf.checkpoint.json")).unwrap(),
                )
                .unwrap();
                let bytes = fs::read(&tmp).unwrap();
                assert_eq!(checkpoint["completed_tensors"], completed);
                assert_eq!(checkpoint["total_tensors"], 11);
                assert_eq!(checkpoint["file_bytes"], bytes.len());
                // Hash actual disk bytes independently, including the first run's
                // header and tensors on the second interruption.
                assert_eq!(checkpoint["prefix_sha256"], digest(&bytes));
                assert!(bytes.starts_with(&previous_prefix));
                assert!(bytes.len() > previous_prefix.len());
                let header_digest = checkpoint["header_sha256"].as_str().unwrap().to_owned();
                assert_eq!(header_digest.len(), 64);
                if let Some(previous) = previous_header_digest {
                    assert_eq!(header_digest, previous);
                }
                previous_header_digest = Some(header_digest);
                previous_prefix = bytes;
                for name in [
                    "model.gguf",
                    "F32.json",
                    "model.gguf.sha256",
                    "model.gguf.receipt.json",
                ] {
                    assert!(!dir.join(name).exists(), "{completed}: {name}");
                }
                fs::OpenOptions::new()
                    .append(true)
                    .open(&tmp)
                    .unwrap()
                    .write_all(&vec![0xa5; 128 * 1024])
                    .unwrap();
            }
            let model = load(
                ModelSource::hugging_face("fixture/repeat:F32", None, None),
                cfg,
            );
            let path = dir.join("model.gguf");
            check_output(&path, "F32");
            assert!(fs::read(&path).unwrap().starts_with(&previous_prefix));
            let events = progress.0.lock().unwrap();
            assert_eq!(events.len(), 2); // one remaining tensor and final completion
            assert_eq!(events[0].1, previous_prefix.len() as u64);
            assert!(events[0].1 < events[1].1);
            assert_eq!(events[1].1, fs::metadata(&path).unwrap().len());
            execute(
                Arc::try_unwrap(model.engine).ok().unwrap(),
                super::super::super::companion_fixture::primary(),
            );
        },
        |requests, ranges| {
            let path = "/fixture/repeat/resolve/1111111111111111111111111111111111111111/model.safetensors";
            for (index, range) in fixture::ranges(&fixture::shards(false)[0].1)
                .iter()
                .enumerate()
            {
                assert_eq!(
                    ranges
                        .iter()
                        .filter(|(t, r)| t == path && r == range)
                        .count(),
                    if index < 2 { 3 } else { 1 },
                    "{range}"
                );
            }
            assert_eq!(count(requests, "GET", path), 17); // six headers, eleven tensors
        },
    );
}

#[test]
fn malformed_conversion_preserves_errors_and_does_not_publish_output() {
    isolated_with_ranges(
        "conversion::recovery::malformed_conversion_preserves_errors_and_does_not_publish_output",
        || {
            let mut routes = fixture::routes(
                "bad-header",
                "main",
                &[("model.safetensors".into(), vec![0; 8])],
                true,
            );
            let mut shards = fixture::shards(false);
            // The same byte count interpreted as F16 doubles the decoded element count.
            let bytes = &mut shards[0].1;
            let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header = String::from_utf8(bytes[8..8 + len].to_vec())
                .unwrap()
                .replace("F32", "F16");
            bytes[8..8 + len].copy_from_slice(header.as_bytes());
            routes.extend(fixture::routes("bad-count", "main", &shards, true));
            routes
        },
        |ctx| {
            for (repo, message) in [
                ("bad-header", "invalid SafeTensors header length 0"),
                ("bad-count", "element count mismatch"),
            ] {
                let progress = Arc::new(Progress::default());
                let cfg = config(&ctx.root, &progress);
                let spec = format!("fixture/{repo}:F32");
                let typed = engine_error(
                    ModelLoader::new(ModelSource::hugging_face(&spec, None, None))
                        .config(cfg.clone())
                        .build(),
                    "hf",
                );
                assert!(typed.to_string().contains(message), "{typed}");
                let legacy = CeraEngine::from_hf(&spec, None, cfg.clone()).err().unwrap();
                same_error(typed, legacy);
                let dir = cache_dir(&cfg, repo, "F32");
                for filename in [
                    "F32.json",
                    "model.gguf",
                    "model.gguf.sha256",
                    "model.gguf.checkpoint.json",
                ] {
                    assert!(!dir.join(filename).exists(), "{repo}: {filename}");
                }
                // The converter intentionally retains its temporary file on a tensor failure.
                assert_eq!(dir.join("model.gguf.tmp").exists(), repo == "bad-count");
            }
        },
        |requests, ranges| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    "/fixture/bad-header/resolve/1111111111111111111111111111111111111111/model.safetensors"
                ),
                2
            );
            assert_eq!(
                count(
                    requests,
                    "GET",
                    "/fixture/bad-count/resolve/1111111111111111111111111111111111111111/model.safetensors"
                ),
                6
            );
            assert!(ranges.iter().all(|(_, r)| r.starts_with("bytes=")));
        },
    );
}

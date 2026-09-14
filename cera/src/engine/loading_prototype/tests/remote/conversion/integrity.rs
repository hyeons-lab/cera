use super::*;
use crate::bundle::HfSpec;
use crate::convert::{QuantizeOptions, TargetQuant, stream_quantize_hf_repo};
use std::sync::atomic::AtomicBool;

const RECEIPT: &str = "model.gguf.receipt.json";

fn options(cfg: &LoadConfig) -> QuantizeOptions {
    QuantizeOptions {
        target_quant: TargetQuant::F32,
        cache_dir: cfg.bundle_repo.as_ref().unwrap().store_dir().into(),
        auth_token: None,
        ..Default::default()
    }
}

// Replace the directory entry, preserving any live mmap of the old model.
fn replace(path: &Path, bytes: &[u8]) {
    let temp = path.with_extension("replacement");
    fs::write(&temp, bytes).unwrap();
    fs::rename(temp, path).unwrap();
}

#[test]
fn corrupt_completed_artifacts_reconvert_and_preserve_retained_models() {
    isolated_with_ranges(
        "conversion::integrity::corrupt_completed_artifacts_reconvert_and_preserve_retained_models",
        || fixture::routes("integrity", "main", &fixture::shards(false), true),
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let source = || ModelSource::hugging_face("fixture/integrity:F32", None, None);
            let original = load(source(), cfg.clone());
            let dir = cache_dir(&cfg, "integrity", "F32");
            let path = dir.join("model.gguf");
            let manifest = dir.join("F32.json");
            let pristine = fs::read(&path).unwrap();
            let pristine_manifest = fs::read(&manifest).unwrap();
            let gguf = GgufFile::from_bytes(pristine.clone().into()).unwrap();
            let offset = gguf.tensors["blk.0.ffn_down.weight"].offset as usize;

            for damage in [
                "same-size",
                "truncated",
                "manifest",
                "missing-receipt",
                "bad-receipt",
            ] {
                match damage {
                    "same-size" => {
                        let mut bad = pristine.clone();
                        bad[offset..offset + 4].copy_from_slice(&9.0f32.to_le_bytes());
                        assert_ne!(digest(&bad), digest(&pristine));
                        replace(&path, &bad);
                    }
                    "truncated" => replace(&path, &pristine[..pristine.len() / 2]),
                    "manifest" => {
                        let mut value: serde_json::Value =
                            serde_json::from_slice(&pristine_manifest).unwrap();
                        value["generation_time_parameters"]["sampling_parameters"]["temperature"] =
                            json!(0.95);
                        let damaged = serde_json::to_vec(&value).unwrap();
                        let parsed = Manifest::from_bytes(&damaged).unwrap();
                        let GenerationDefaults::Text { temperature, .. } =
                            parsed.generation_defaults
                        else {
                            panic!("expected text defaults")
                        };
                        assert_eq!(temperature, Some(0.95));
                        fs::write(&manifest, damaged).unwrap();
                    }
                    "missing-receipt" => fs::remove_file(dir.join(RECEIPT)).unwrap(),
                    "bad-receipt" => fs::write(dir.join(RECEIPT), b"{}").unwrap(),
                    _ => unreachable!(),
                }
                let before = progress.0.lock().unwrap().len();
                let repaired = load(source(), cfg.clone());
                assert_eq!(
                    progress.0.lock().unwrap().len() - before,
                    12,
                    "{damage} must reconvert"
                );
                assert_eq!(fs::read(&path).unwrap(), pristine, "{damage}");
                assert_eq!(fs::read(&manifest).unwrap(), pristine_manifest, "{damage}");
                check_output(&path, "F32");
                assert_eq!(repaired.engine.default_generate_opts().temperature, 0.25);
                execute(
                    Arc::try_unwrap(repaired.engine).ok().unwrap(),
                    super::super::super::companion_fixture::primary(),
                );
            }
            // A receipt-verified artifact can repair its auxiliary sidecar without fetching tensors.
            for content in [
                None,
                Some("invalid"),
                Some("0000000000000000000000000000000000000000000000000000000000000000"),
            ] {
                let sidecar = path.with_extension("gguf.sha256");
                if let Some(text) = content {
                    fs::write(&sidecar, text).unwrap();
                } else {
                    fs::remove_file(&sidecar).unwrap();
                }
                let before = progress.0.lock().unwrap().len();
                let _ = load(source(), cfg.clone());
                assert_eq!(progress.0.lock().unwrap().len(), before);
                check_output(&path, "F32");
            }
            execute(
                Arc::try_unwrap(original.engine).ok().unwrap(),
                super::super::super::companion_fixture::primary(),
            );
        },
        |requests, ranges| {
            let target = "/fixture/integrity/resolve/1111111111111111111111111111111111111111/model.safetensors";
            let expected = fixture::ranges(&fixture::shards(false)[0].1);
            assert_eq!(count(requests, "GET", target), 6 * expected.len());
            for range in expected {
                assert_eq!(
                    ranges
                        .iter()
                        .filter(|(p, r)| p == target && r == &range)
                        .count(),
                    6
                );
            }
        },
    );
}

#[test]
fn interrupted_repair_does_not_revalidate_old_output() {
    isolated_with_ranges(
        "conversion::integrity::interrupted_repair_does_not_revalidate_old_output",
        || fixture::routes("repair", "main", &fixture::shards(false), true),
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse("fixture/repair:F32").unwrap();
            let original = stream_quantize_hf_repo(&spec, options(&cfg)).unwrap();
            let path = Path::new(&original.files.model);
            let pristine = fs::read(path).unwrap();
            replace(path, &pristine[..100]);
            let mut opts = options(&cfg);
            opts.cancel = Some(Arc::new(AtomicBool::new(true)));
            assert!(matches!(
                stream_quantize_hf_repo(&spec, opts),
                Err(CeraError::Cancelled)
            ));
            assert!(!path.parent().unwrap().join(RECEIPT).exists());
            assert_eq!(fs::read(path).unwrap().len(), 100);
            let model = load(
                ModelSource::hugging_face("fixture/repair:F32", None, None),
                cfg,
            );
            assert_eq!(progress.0.lock().unwrap().len(), 12);
            assert_eq!(fs::read(path).unwrap(), pristine);
            execute(
                Arc::try_unwrap(model.engine).ok().unwrap(),
                super::super::super::companion_fixture::primary(),
            );
        },
        |requests, _| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    "/fixture/repair/resolve/1111111111111111111111111111111111111111/model.safetensors"
                ),
                28
            );
        },
    );
}

#[test]
fn changed_tensor_overrides_reconvert_and_identical_overrides_reuse() {
    isolated_with_ranges(
        "conversion::integrity::changed_tensor_overrides_reconvert_and_identical_overrides_reuse",
        || fixture::routes("overrides", "main", &fixture::shards(false), true),
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse("fixture/overrides:F32").unwrap();
            for (override_type, expected_events) in [
                (None, 12),
                (Some(TargetQuant::F16), 24),
                (Some(TargetQuant::F16), 24),
                (None, 36),
            ] {
                let mut opts = options(&cfg);
                opts.progress = Some(progress.clone());
                if let Some(kind) = override_type {
                    opts.tensor_overrides
                        .push(("blk.0.ffn_down.weight".into(), kind));
                }
                let manifest = stream_quantize_hf_repo(&spec, opts).unwrap();
                let gguf = GgufFile::open(Path::new(&manifest.files.model)).unwrap();
                assert_eq!(
                    gguf.tensors["blk.0.ffn_down.weight"].ggml_type_id,
                    if override_type.is_some() {
                        GGML_TYPE_F16
                    } else {
                        GGML_TYPE_F32
                    }
                );
                assert_eq!(progress.0.lock().unwrap().len(), expected_events);
                let got = gguf
                    .get_tensor("blk.0.ffn_down.weight")
                    .unwrap()
                    .to_f32_vec();
                let want = fixture::source()
                    .get_tensor("blk.0.ffn_down.weight")
                    .unwrap()
                    .to_f32_vec();
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.001));
            }
        },
        |requests, _| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    "/fixture/overrides/resolve/1111111111111111111111111111111111111111/model.safetensors"
                ),
                39
            )
        },
    );
}

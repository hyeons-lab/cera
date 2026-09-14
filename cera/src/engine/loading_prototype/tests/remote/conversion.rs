//! Actual streaming conversion through the shared HF loading boundary.

use super::super::auxiliary::{close, distinct};
use super::*;
use crate::convert::writer::{GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0};
use http::{Response, isolated_with_ranges};
use std::collections::HashMap;

mod fixture;
mod integrity;
mod recovery;
mod upstream;

fn cache_dir(cfg: &LoadConfig, repo: &str, leaf: &str) -> PathBuf {
    cfg.bundle_repo
        .as_ref()
        .unwrap()
        .store_dir()
        .join("huggingface.co/fixture")
        .join(repo)
        .join("quantized")
        .join(leaf)
}

fn check_output(path: &Path, quant: &str) {
    let actual = GgufFile::open(path).unwrap();
    let expected = fixture::source();
    assert_eq!(actual.architecture(), Some("llama"));
    assert_eq!(actual.get_u32("llama.context_length"), Some(256));
    assert_eq!(
        actual.get_string_array("tokenizer.ggml.tokens").unwrap(),
        ["a", "b"]
    );
    assert_eq!(
        actual.get_str("tokenizer.chat_template"),
        Some(fixture::TEMPLATE)
    );
    assert_eq!(actual.tensors.len(), expected.tensors.len());
    let alignment = actual.get_u32("general.alignment").unwrap_or(32) as usize;
    let expected_len = actual
        .tensors
        .values()
        .map(|t| t.offset as usize + t.size_bytes)
        .max()
        .unwrap()
        .next_multiple_of(alignment);
    assert_eq!(
        fs::metadata(path).unwrap().len() as usize,
        expected_len,
        "unexpected trailing conversion bytes"
    );
    for (name, tensor) in &expected.tensors {
        let got = &actual.tensors[name];
        assert_eq!(got.shape, tensor.shape, "{name}");
        let expected_type = match quant {
            "F16" => GGML_TYPE_F16,
            "Q8_0" if tensor.shape.len() == 2 && tensor.size_bytes >= 1024 => GGML_TYPE_Q8_0,
            "Q4_0" if tensor.shape.len() == 2 && tensor.size_bytes >= 1024 => GGML_TYPE_Q4_0,
            _ => GGML_TYPE_F32,
        };
        assert_eq!(got.ggml_type_id, expected_type, "{name}");
        if expected_type == GGML_TYPE_F32 {
            assert_eq!(
                actual.tensor_data(name).unwrap(),
                expected.tensor_data(name).unwrap(),
                "{name}"
            );
        } else {
            let got = actual.get_tensor(name).unwrap().to_f32_vec();
            let want = expected.get_tensor(name).unwrap().to_f32_vec();
            assert_eq!(got.len(), want.len(), "{name}");
            let signal_energy: f32 = want.iter().map(|x| x * x).sum();
            let error_energy: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum();
            assert!(
                signal_energy > 0.0,
                "fixture must have nonzero weights: {name}"
            );
            let relative_error = (error_energy / signal_energy).sqrt();
            assert!(
                relative_error < 0.15,
                "{name}: relative error {relative_error}"
            );
            let max_error = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let tolerance = match quant {
                "F16" => 0.001,
                "Q8_0" => 0.01,
                "Q4_0" => 0.15,
                _ => unreachable!(),
            };
            assert!(
                got.iter().all(|x| x.is_finite()) && max_error < tolerance,
                "{name}: {max_error}"
            );
        }
    }
    assert_eq!(
        fs::read_to_string(path.with_extension("gguf.sha256"))
            .unwrap()
            .trim(),
        digest(&fs::read(path).unwrap())
    );
    let dir = path.parent().unwrap();
    assert!(!dir.join("model.gguf.tmp").exists());
    assert!(!dir.join("model.gguf.checkpoint.json").exists());
}

fn execute(engine: CeraEngine, expected_bytes: Arc<[u8]>) {
    let mut session = engine.new_session(SessionConfig::default()).unwrap();
    drop(engine);
    let mut control = CeraEngine::from_bytes(expected_bytes.clone(), cpu_config())
        .unwrap()
        .new_session(SessionConfig::default())
        .unwrap();
    session.append_tokens(&[0, 1]).unwrap();
    control.append_tokens(&[0, 1]).unwrap();
    close(
        session.last_logits().unwrap(),
        control.last_logits().unwrap(),
    );
    let mut other = CeraEngine::from_bytes(expected_bytes, cpu_config())
        .unwrap()
        .new_session(SessionConfig::default())
        .unwrap();
    other.append_tokens(&[1, 1]).unwrap();
    distinct(session.last_logits().unwrap(), other.last_logits().unwrap());
    assert_eq!(generate(&mut session), generate(&mut control));
    session.append_tokens(&[1]).unwrap();
    control.append_tokens(&[1]).unwrap();
    assert_eq!(session.position(), 7);
    close(
        session.last_logits().unwrap(),
        control.last_logits().unwrap(),
    );
}

#[test]
fn single_and_sharded_conversion_execute_and_reuse_cache() {
    isolated_with_ranges(
        "conversion::single_and_sharded_conversion_execute_and_reuse_cache",
        || {
            let mut routes = fixture::routes("single", "main", &fixture::shards(false), true);
            routes.extend(fixture::routes(
                "single",
                "release",
                &fixture::shards(false),
                true,
            ));
            // Exercise servers that ignore Range and return bounded HTTP 200 bodies too.
            routes.extend(fixture::routes(
                "sharded",
                "main",
                &fixture::shards(true),
                false,
            ));
            routes
        },
        |ctx| {
            for (repo, revision, quant, strategy, leaf) in [
                ("single", "main", "F32", None, "F32"),
                ("single", "release", "F32", None, "F32@release"),
                ("sharded", "main", "F32", None, "F32"),
                (
                    "single",
                    "release",
                    "Q8_0",
                    Some("fast-mse"),
                    "Q8_0-fast-mse@release",
                ),
                ("single", "main", "Q4_0", Some("hqq"), "Q4_0-hqq"),
                ("single", "main", "Q4_0", None, "Q4_0"),
                ("single", "main", "F16", Some("unknown"), "F16"),
            ] {
                let progress = Arc::new(Progress::default());
                let cfg = config(&ctx.root, &progress);
                // Explicit quant wins over the spec's deliberately invalid format.
                let spec = format!("fixture/{repo}:invalid@{revision}");
                let source = || ModelSource::hugging_face(&spec, Some(quant), strategy);
                let typed = load(source(), cfg.clone());
                let dir = cache_dir(&cfg, repo, leaf);
                let path = dir.join("model.gguf");
                assert_eq!(Path::new(&typed.engine.manifest().files.model), path);
                check_output(&path, quant);
                let events = progress.0.lock().unwrap().clone();
                assert_eq!(events.len(), 12); // eleven tensors plus final output
                assert!(events.windows(2).all(|pair| pair[0].1 < pair[1].1));
                let final_event = events.last().unwrap();
                assert_eq!(
                    final_event.0,
                    format!(
                        "{}/fixture/{repo}/resolve/{}/model.gguf",
                        ctx.url,
                        fixture::commit(revision)
                    )
                );
                assert_eq!(final_event.1, fs::metadata(&path).unwrap().len());
                assert!(
                    events
                        .iter()
                        .all(|(_, bytes, total)| *total == Some(final_event.1)
                            && *bytes <= final_event.1)
                );
                let output_hash = digest(&fs::read(&path).unwrap());
                let dynamic = ModelLoader::new(source())
                    .config(cfg.clone())
                    .build()
                    .unwrap();
                let dynamic_model = dynamic.as_generative().unwrap();
                drop(dynamic);
                let legacy =
                    CeraEngine::from_hf_with_strategy(&spec, Some(quant), strategy, cfg).unwrap();
                assert_eq!(progress.0.lock().unwrap().len(), events.len());
                assert_eq!(output_hash, digest(&fs::read(&path).unwrap()));
                let expected: Arc<[u8]> = if quant == "F32" {
                    super::super::companion_fixture::primary()
                } else {
                    fs::read(&path).unwrap().into()
                };
                for engine in [
                    Arc::try_unwrap(typed.engine).ok().unwrap(),
                    Arc::try_unwrap(dynamic_model.engine).ok().unwrap(),
                    legacy,
                ] {
                    assert_eq!(Path::new(&engine.manifest().files.model), path);
                    assert_eq!(engine.manifest().raw["quant"], quant);
                    assert_eq!(
                        engine.manifest().raw["strategy"],
                        strategy.filter(|s| *s != "unknown").unwrap_or("auto")
                    );
                    assert_eq!(
                        engine.manifest().chat_template.as_deref(),
                        Some(fixture::TEMPLATE)
                    );
                    let defaults = engine.default_generate_opts();
                    assert_eq!(
                        (
                            defaults.temperature,
                            defaults.min_p,
                            defaults.top_p,
                            defaults.top_k,
                            defaults.repetition_penalty
                        ),
                        (0.25, 0.15, 0.8, 3, 1.2)
                    );
                    execute(engine, expected.clone());
                }
            }
            let cfg = config(&ctx.root, &Arc::new(Progress::default()));
            let auto = cache_dir(&cfg, "single", "Q4_0").join("model.gguf");
            let hqq = cache_dir(&cfg, "single", "Q4_0-hqq").join("model.gguf");
            assert_ne!(
                digest(&fs::read(auto).unwrap()),
                digest(&fs::read(hqq).unwrap()),
                "strategy must affect converted weights"
            );
        },
        |requests, ranges| {
            for (repo, revision, split, conversions) in [
                ("single", "main", false, 4),
                ("single", "release", false, 2),
                ("sharded", "main", true, 1),
            ] {
                let prefix = format!("/fixture/{repo}/resolve/{}/", fixture::commit(revision));
                for (name, bytes) in fixture::shards(split) {
                    let target = format!("{prefix}{name}");
                    let expected = fixture::ranges(&bytes);
                    assert_eq!(
                        count(requests, "GET", &target),
                        expected.len() * conversions
                    );
                    for range in expected {
                        assert_eq!(
                            ranges
                                .iter()
                                .filter(|(t, r)| *t == target && *r == range)
                                .count(),
                            conversions,
                            "{target}: {range}"
                        );
                    }
                }
                for name in [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                ] {
                    assert_eq!(
                        count(requests, "GET", &format!("{prefix}{name}")),
                        conversions
                    );
                }
                let suffix = if revision == "main" {
                    String::new()
                } else {
                    format!("/revision/{revision}")
                };
                // Each load resolves discovery plus the converter's current commit, even on a cache hit.
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/api/models/fixture/{repo}{suffix}")
                    ),
                    conversions * 6
                );
            }
        },
    );
}

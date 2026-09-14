//! Remote adapters use real HTTP/download/cache paths against isolated fixtures.

use super::*;
use crate::bundle::{BundleRepo, DownloadProgress};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::sync::Mutex;

mod companions;
mod conversion;
mod http;
mod snapshots;
use http::{MAIN_COMMIT, RELEASE_COMMIT, count, isolated};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Default)]
struct Progress(Mutex<Vec<(String, u64, Option<u64>)>>);
impl DownloadProgress for Progress {
    fn on_progress(&self, url: &str, bytes: u64, total: Option<u64>) {
        self.0.lock().unwrap().push((url.into(), bytes, total));
    }
}

fn config(root: &Path, progress: &Arc<Progress>) -> LoadConfig {
    LoadConfig {
        bundle_repo: Some(BundleRepo::with_progress(
            root.join("store"),
            progress.clone(),
        )),
        ..cpu_config()
    }
}

fn load(source: ModelSource<'_>, config: LoadConfig) -> GenerativeModel {
    ModelLoader::new(source)
        .config(config)
        .build_generative()
        .unwrap()
}

fn engine_error<T>(result: Result<T, LoadError>, expected_source: &str) -> CeraError {
    match result.err().expect("expected failure") {
        LoadError::Source { source_kind, error } => {
            assert_eq!(source_kind, expected_source, "wrong loading source kind");
            error
        }
        error => panic!("expected source-resolution failure, got {error:?}"),
    }
}

fn same_error(actual: CeraError, expected: CeraError) {
    assert_eq!(
        std::mem::discriminant(&actual),
        std::mem::discriminant(&expected)
    );
    assert_eq!(actual.to_string(), expected.to_string());
}

fn assert_parity(model: &GenerativeModel, legacy: &CeraEngine) {
    assert_eq!(model.engine.manifest().raw, legacy.manifest().raw);
    assert_eq!(
        model.engine.manifest().files.model,
        legacy.manifest().files.model
    );
    assert_eq!(
        model.engine.manifest().inference_type,
        legacy.manifest().inference_type
    );
    let actual = generate(&mut model.create_session(SessionConfig::default()).unwrap());
    assert_eq!(actual.len(), 2);
    assert_eq!(
        actual,
        generate(&mut legacy.new_session(SessionConfig::default()).unwrap())
    );
}

// Use the shared URL addressing primitive without exposing BundleRepo internals.
trait FixtureCachePath {
    fn fixture_path(&self, url: &str) -> Result<PathBuf, CeraError>;
}

impl FixtureCachePath for BundleRepo {
    fn fixture_path(&self, url: &str) -> Result<PathBuf, CeraError> {
        let mut path = self.store_dir().to_path_buf();
        for segment in crate::bundle::cache_key::cache_relative_segments(url)? {
            path.push(segment);
        }
        Ok(path)
    }
}

fn seed(repo: &BundleRepo, url: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
    let path = repo.fixture_path(url).unwrap();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, bytes).unwrap();
    path
}

fn manifest(model: &str) -> serde_json::Value {
    json!({"schema_version":"1.0.0", "inference_type":"llama.cpp/text-to-text",
        "load_time_parameters":{"model":model}})
}

#[test]
fn hf_selection_defaults_cache_progress_and_ownership() {
    isolated(
        "hf_selection_defaults_cache_progress_and_ownership",
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let cases = [
                (
                    "fixture/text".to_string(),
                    None,
                    None,
                    "main/nested/model-Q4_K_M.gguf",
                ),
                (
                    "fixture/text:Q8_0".to_string(),
                    None,
                    None,
                    "main/model-Q8_0.gguf",
                ),
                (
                    "fixture/text:Q4_K_M@release".to_string(),
                    Some("Q8_0"),
                    Some("hqq"),
                    "release/model-Q8_0.gguf",
                ),
                (
                    format!(
                        "{}/fixture/text/resolve/release/nested/model-Q4_K_M.gguf",
                        ctx.url
                    ),
                    Some("Q8_0"),
                    None,
                    "release/nested/model-Q4_K_M.gguf",
                ),
                (
                    format!("{}/fixture/text", ctx.url),
                    None,
                    Some("unknown-strategy"),
                    "main/nested/model-Q4_K_M.gguf",
                ),
            ];
            for (spec, quant, strategy, selected) in cases {
                let selected = selected
                    .replace("main/", &format!("{MAIN_COMMIT}/"))
                    .replace("release/", &format!("{RELEASE_COMMIT}/"));
                let model = load(
                    ModelSource::hugging_face(&spec, quant, strategy),
                    cfg.clone(),
                );
                let event_count = progress.0.lock().unwrap().len();
                let legacy = if spec == format!("{}/fixture/text", ctx.url) {
                    CeraEngine::from_hf_url(&spec, quant, cfg.clone()).unwrap()
                } else {
                    CeraEngine::from_hf_with_strategy(&spec, quant, strategy, cfg.clone()).unwrap()
                };
                assert_eq!(
                    progress.0.lock().unwrap().len(),
                    event_count,
                    "cache hit emitted progress"
                );
                assert_parity(&model, &legacy);
                let expected = repo
                    .fixture_path(&format!("{}/fixture/text/resolve/{selected}", ctx.url))
                    .unwrap();
                assert_eq!(Path::new(&model.engine.manifest().files.model), expected);
                assert_eq!(fs::read(&expected).unwrap(), http::hf_model_bytes());
                assert_eq!(
                    fs::read_to_string(crate::bundle::download::sidecar_path(&expected)).unwrap(),
                    digest(&http::hf_model_bytes())
                );
                let opts = model.engine.default_generate_opts();
                assert_eq!(opts.temperature, 0.25);
                assert_eq!(opts.min_p, 0.15);
                assert_eq!(opts.top_p, 0.8);
                assert_eq!(opts.top_k, 3);
                assert_eq!(opts.repetition_penalty, 1.2);
                let retained = model.engine.config().bundle_repo.as_ref().unwrap();
                assert_eq!(retained.store_dir(), repo.store_dir());
                assert!(Arc::ptr_eq(
                    &retained.progress().unwrap(),
                    &repo.progress().unwrap()
                ));
                let mut session = model.create_session(SessionConfig::default()).unwrap();
                drop(model);
                drop(legacy);
                assert_eq!(generate(&mut session).len(), 2);
            }
            let events = progress.0.lock().unwrap();
            for selected in [
                "main/nested/model-Q4_K_M.gguf",
                "main/model-Q8_0.gguf",
                "release/model-Q8_0.gguf",
                "release/nested/model-Q4_K_M.gguf",
            ] {
                let selected = selected
                    .replace("main/", &format!("{MAIN_COMMIT}/"))
                    .replace("release/", &format!("{RELEASE_COMMIT}/"));
                let url = format!("{}/fixture/text/resolve/{selected}", ctx.url);
                let stream: Vec<_> = events.iter().filter(|(u, _, _)| *u == url).collect();
                assert!(stream.len() > 1, "expected intermediate download progress");
                assert!(stream.first().unwrap().1 < stream.last().unwrap().1);
                assert!(stream.windows(2).all(|w| w[0].1 <= w[1].1));
                assert_eq!(
                    stream.last().unwrap().1,
                    http::hf_model_bytes().len() as u64
                );
                assert!(
                    stream
                        .iter()
                        .all(|(_, _, total)| *total == Some(http::hf_model_bytes().len() as u64))
                );
            }
        },
        |requests| {
            for selected in [
                "main/nested/model-Q4_K_M.gguf",
                "main/model-Q8_0.gguf",
                "release/model-Q8_0.gguf",
                "release/nested/model-Q4_K_M.gguf",
            ] {
                let selected = selected
                    .replace("main/", &format!("{MAIN_COMMIT}/"))
                    .replace("release/", &format!("{RELEASE_COMMIT}/"));
                assert_eq!(
                    count(
                        requests,
                        "GET",
                        &format!("/fixture/text/resolve/{selected}")
                    ),
                    1
                );
            }
            assert_eq!(count(requests, "GET", "/api/models/fixture/text"), 6);
            assert_eq!(
                count(requests, "GET", "/api/models/fixture/text/revision/release"),
                4
            );
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/text/resolve/{MAIN_COMMIT}/generation_config.json")
                ),
                6
            );
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/text/resolve/{RELEASE_COMMIT}/generation_config.json")
                ),
                4
            );
        },
    );
}

#[test]
fn remote_errors_preserve_order_and_check_final_kind() {
    isolated(
        "remote_errors_preserve_order_and_check_final_kind",
        |ctx| {
            same_error(
                engine_error(
                    ModelLoader::new(ModelSource::hugging_face("", None, None)).build(),
                    "hf",
                ),
                CeraEngine::from_hf("", None, cpu_config()).err().unwrap(),
            );
            same_error(
                engine_error(
                    ModelLoader::new(ModelSource::bundle_id("..", "")).build(),
                    "bundle",
                ),
                CeraEngine::from_bundle_id("..", "", cpu_config())
                    .err()
                    .unwrap(),
            );
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            for (spec, quant) in [
                ("", None),
                ("fixture/missing", None),
                ("fixture/private", None),
                ("fixture/bad-json", None),
                ("fixture/text", Some("Q2_K")),
                ("fixture/safetensors", Some("invalid")),
            ] {
                let error = engine_error(
                    ModelLoader::new(ModelSource::hugging_face(spec, quant, Some("hqq")))
                        .config(cfg.clone())
                        .build(),
                    "hf",
                );
                same_error(
                    error,
                    CeraEngine::from_hf_with_strategy(spec, quant, Some("hqq"), cfg.clone())
                        .err()
                        .unwrap(),
                );
            }
            assert!(progress.0.lock().unwrap().is_empty());
            for (arch, kind) in [
                ("bert", ModelKind::Encoder),
                ("modernbert", ModelKind::Encoder),
                ("whisper", ModelKind::Whisper),
                ("silero_vad", ModelKind::Vad),
                ("kws", ModelKind::Hotword),
            ] {
                let mut cfg = cfg.clone();
                cfg.backend = BackendPreference::Metal;
                let error = ModelLoader::new(ModelSource::hugging_face(
                    format!("fixture/{arch}"),
                    None,
                    None,
                ))
                .config(cfg)
                .build()
                .err()
                .unwrap();
                assert!(
                    matches!(error, LoadError::KindMismatch {expected:ModelKind::Generative,actual,architecture} if actual==kind && architecture==arch)
                );
            }
            let error = ModelLoader::new(ModelSource::hugging_face("fixture/future", None, None))
                .config(cfg)
                .build()
                .err()
                .unwrap();
            assert!(
                matches!(error,LoadError::UnsupportedArchitecture {architecture} if architecture=="future")
            );
        },
        |requests| {
            assert_eq!(count(requests, "GET", "/api/models/fixture/missing"), 2);
            assert_eq!(count(requests, "GET", "/api/models/fixture/private"), 2);
            assert!(
                !requests
                    .iter()
                    .any(|(_, target)| target.ends_with(".safetensors"))
            );
            assert!(!requests.iter().any(|(_, target)| {
                target.contains("/fixture/text/resolve/") && target.ends_with(".gguf")
            }));
            // Legacy discovery fetches defaults before primary quant selection.
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!("/fixture/text/resolve/{MAIN_COMMIT}/generation_config.json")
                ),
                2
            );
        },
    );
}

#[test]
fn remote_manifest_payloads_and_download_failures() {
    isolated(
        "remote_manifest_payloads_and_download_failures",
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let mut value = manifest(&format!("{}/assets/primary.gguf", ctx.url));
            for (field, asset) in [
                ("multimodal_projector", "mmproj"),
                ("audio_decoder", "decoder"),
                ("audio_tokenizer", "tokenizer"),
                ("draft_model", "draft"),
                ("future_file", "extra"),
            ] {
                value["load_time_parameters"][field] =
                    json!(format!("{}/assets/{asset}.gguf", ctx.url));
            }
            value["load_time_parameters"]["chat_template"] = json!("retained remote metadata");
            let path = ctx.root.join("manifest.json");
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            let model = load(ModelSource::path(&path), cfg.clone());
            let count = progress.0.lock().unwrap().len();
            let legacy = CeraEngine::from_path(&path, cfg.clone()).unwrap();
            assert_eq!(progress.0.lock().unwrap().len(), count);
            assert_parity(&model, &legacy);
            assert_eq!(model.engine.manifest().raw, value);
            let files = &model.engine.manifest().files;
            for (path, asset) in [
                (&files.model, "primary"),
                (files.multimodal_projector.as_ref().unwrap(), "mmproj"),
                (files.audio_decoder.as_ref().unwrap(), "decoder"),
                (files.audio_tokenizer.as_ref().unwrap(), "tokenizer"),
                (files.draft_model.as_ref().unwrap(), "draft"),
                (&files.extras["future_file"], "extra"),
            ] {
                assert_eq!(
                    Path::new(path),
                    cfg.bundle_repo
                        .as_ref()
                        .unwrap()
                        .fixture_path(&format!("{}/assets/{asset}.gguf", ctx.url))
                        .unwrap()
                );
            }
            for asset in ["bad", "hash-mismatch", "missing"] {
                fs::write(
                    &path,
                    serde_json::to_vec(&manifest(&format!("{}/assets/{asset}.gguf", ctx.url)))
                        .unwrap(),
                )
                .unwrap();
                let error = engine_error(
                    ModelLoader::new(ModelSource::path(&path))
                        .config(cfg.clone())
                        .build(),
                    "path",
                );
                same_error(
                    error,
                    CeraEngine::from_path(&path, cfg.clone()).err().unwrap(),
                );
            }
            let dest = cfg
                .bundle_repo
                .as_ref()
                .unwrap()
                .fixture_path(&format!("{}/assets/hash-mismatch.gguf", ctx.url))
                .unwrap();
            assert!(!dest.exists());
            assert!(!crate::bundle::download::partial_path(&dest).exists());
            assert!(!crate::bundle::download::sidecar_path(&dest).exists());
        },
        |requests| {
            for asset in [
                "primary",
                "mmproj",
                "decoder",
                "tokenizer",
                "draft",
                "extra",
                "bad",
            ] {
                assert_eq!(count(requests, "GET", &format!("/assets/{asset}.gguf")), 1);
            }
            assert_eq!(count(requests, "GET", "/assets/hash-mismatch.gguf"), 2);
            assert_eq!(count(requests, "GET", "/assets/missing.gguf"), 2);
        },
    );
}

#[test]
fn cached_bundle_relative_manifest_known_bundle_and_dspark() {
    isolated(
        "cached_bundle_relative_manifest_known_bundle_and_dspark",
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            for id in ["fixture-model", "LFM2.5-1.2B-Instruct-GGUF"] {
                let url = crate::bundle::leap_bundles_manifest_url(id, "Q4_K_M").unwrap();
                let path = seed(
                    repo,
                    &url,
                    serde_json::to_vec(&manifest("relative.gguf")).unwrap(),
                );
                fs::write(path.parent().unwrap().join("relative.gguf"), tiny_llama()).unwrap();
                let plain = load(ModelSource::bundle_id(id, "Q4_K_M"), cfg.clone());
                assert!(plain.engine.manifest().files.draft_model.is_none());
                for quant in ["Q4_K_M+DSpark", "Q4_K_M dspark"] {
                    let model = load(ModelSource::bundle_id(id, quant), cfg.clone());
                    let legacy = CeraEngine::from_bundle_id(id, quant, cfg.clone()).unwrap();
                    assert_parity(&model, &legacy);
                    assert_eq!(
                        Path::new(&model.engine.manifest().files.model),
                        path.parent().unwrap().join("relative.gguf")
                    );
                    if id.starts_with("LFM2.5") {
                        let url =
                            crate::bundle::hf::known_companion_dspark_url(id, "Q4_K_M").unwrap();
                        assert_eq!(
                            Path::new(model.engine.manifest().files.draft_model.as_ref().unwrap()),
                            repo.fixture_path(&url).unwrap()
                        );
                    } else {
                        assert!(model.engine.manifest().files.draft_model.is_none());
                    }
                }
            }
            // Known VL bundle bypasses the catalog. The tiny text primary tests
            // resolution/assembly only; the companion is intentionally unusable.
            let known = crate::bundle::known_bundle_manifest("LFM2.5-VL-3B-GGUF", "Q8_0").unwrap();
            seed(repo, &known.files.model, tiny_llama());
            seed(
                repo,
                known.files.multimodal_projector.as_ref().unwrap(),
                b"invalid optional projector",
            );
            let model = load(
                ModelSource::bundle_id("LiquidAI/LFM2.5-VL-3B-GGUF", "Q8_0"),
                cfg.clone(),
            );
            let legacy =
                CeraEngine::from_bundle_id("LiquidAI/LFM2.5-VL-3B-GGUF", "Q8_0", cfg.clone())
                    .unwrap();
            assert_parity(&model, &legacy);
            assert_eq!(
                model.engine.manifest().inference_type,
                InferenceType::LlamaCppImageToText
            );
            assert_eq!(model.engine.default_generate_opts().temperature, 0.1);
            // Explicit draft wins over the suffix's known companion.
            let id = "LFM2.5-1.2B-Instruct-GGUF";
            let mut explicit = manifest("relative.gguf");
            explicit["load_time_parameters"]["draft_model"] = json!("explicit-draft.gguf");
            let path = seed(
                repo,
                &crate::bundle::leap_bundles_manifest_url(id, "Q4_K_M").unwrap(),
                serde_json::to_vec(&explicit).unwrap(),
            );
            let model = load(ModelSource::bundle_id(id, "Q4_K_M+dspark"), cfg.clone());
            assert_eq!(
                Path::new(model.engine.manifest().files.draft_model.as_ref().unwrap()),
                path.parent().unwrap().join("explicit-draft.gguf")
            );
            same_error(
                engine_error(
                    ModelLoader::new(ModelSource::bundle_id("..", "Q4_0"))
                        .config(cfg.clone())
                        .build(),
                    "bundle",
                ),
                CeraEngine::from_bundle_id("..", "Q4_0", cfg).err().unwrap(),
            );
        },
        |requests| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    "/LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF/resolve/main/LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf"
                ),
                1
            );
            assert!(requests.iter().any(|(method, _)| method == "CONNECT"));
            assert!(
                requests
                    .iter()
                    .filter(|(method, _)| method == "GET")
                    .all(|(_, target)| target.contains("DSpark"))
            );
        },
    );
}

#[test]
fn cache_rehash_repair_pinned_hash_and_failed_head_reuse() {
    isolated(
        "cache_rehash_repair_pinned_hash_and_failed_head_reuse",
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let url = format!("{}/assets/stale.gguf", ctx.url);
            // Equal length forces integrity validation to distinguish corruption
            // from a size-only cache hit.
            let path = seed(repo, &url, vec![0; tiny_llama().len()]);
            assert_eq!(repo.resolve_url(&url, None).unwrap(), path);
            assert_eq!(fs::read(&path).unwrap(), tiny_llama().as_ref());
            let count = progress.0.lock().unwrap().len();
            fs::remove_file(crate::bundle::download::sidecar_path(&path)).unwrap();
            assert_eq!(repo.resolve_url(&url, None).unwrap(), path);
            assert_eq!(
                fs::read_to_string(crate::bundle::download::sidecar_path(&path)).unwrap(),
                digest(&tiny_llama())
            );
            assert_eq!(
                repo.resolve_url(&url, Some(&digest(&tiny_llama())))
                    .unwrap(),
                path
            );
            let offline = format!("{}/assets/offline.gguf", ctx.url);
            let offline_path = seed(repo, &offline, tiny_llama());
            assert_eq!(repo.resolve_url(&offline, None).unwrap(), offline_path);
            assert_eq!(progress.0.lock().unwrap().len(), count);
        },
        |requests| {
            assert_eq!(count(requests, "GET", "/assets/stale.gguf"), 1);
            assert_eq!(count(requests, "HEAD", "/assets/stale.gguf"), 2);
            assert_eq!(count(requests, "HEAD", "/assets/offline.gguf"), 1);
            assert_eq!(count(requests, "GET", "/assets/offline.gguf"), 0);
        },
    );
}

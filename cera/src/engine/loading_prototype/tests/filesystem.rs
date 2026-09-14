//! Local filesystem contracts for the private loader and legacy constructors.

use super::*;
use crate::ModelFiles;
use serde_json::{Value, json};
use std::fs;

pub(super) fn manifest(model: impl AsRef<Path>) -> Value {
    json!({
        "schema_version": "1.0.0",
        "inference_type": "llama.cpp/text-to-text",
        "load_time_parameters": {"model": model.as_ref().to_str().unwrap()},
    })
}

pub(super) fn write_manifest(directory: &Path, value: &Value) -> PathBuf {
    let path = directory.join("bundle.JSON");
    fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    path
}

fn load(source: ModelSource<'_>, config: LoadConfig) -> GenerativeModel {
    ModelLoader::new(source)
        .config(config)
        .build_generative()
        .unwrap()
}

fn assert_engine_parity(actual: &CeraEngine, legacy: &CeraEngine) {
    assert_eq!(actual.manifest().raw, legacy.manifest().raw);
    assert_eq!(actual.manifest().files.model, legacy.manifest().files.model);
    assert_eq!(
        actual.manifest().inference_type,
        legacy.manifest().inference_type
    );
    assert_eq!(
        actual.tokenizer().chat_template(),
        legacy.tokenizer().chat_template()
    );
    assert_eq!(
        actual.model().config().max_seq_len,
        legacy.model().config().max_seq_len
    );
    assert_eq!(
        generate(&mut actual.new_session(SessionConfig::default()).unwrap()),
        generate(&mut legacy.new_session(SessionConfig::default()).unwrap()),
    );
}

#[test]
fn path_manifest_directory_and_files_keep_generation_and_ownership() {
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("model.GGUF");
    fs::write(&primary, tiny_llama()).unwrap();
    let json_path = write_manifest(directory.path(), &manifest("model.GGUF"));
    for path in [&primary, &json_path, directory.path()] {
        let legacy = CeraEngine::from_path(path, cpu_config()).unwrap();
        let handle = ModelLoader::new(ModelSource::path(path))
            .config(cpu_config())
            .build()
            .unwrap();
        let model = handle.as_generative().unwrap();
        assert_engine_parity(&model.engine, &legacy);
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        drop(handle);
        drop(model);
        assert_eq!(
            generate(&mut session),
            generate(&mut legacy.new_session(SessionConfig::default()).unwrap())
        );
    }
    for inference in [None, Some(InferenceType::LlamaCppTextToText)] {
        let mut files = ModelFiles::text(&primary);
        files.inference_type = inference;
        let legacy = CeraEngine::from_files(files.clone(), cpu_config()).unwrap();
        let model = load(ModelSource::files(files), cpu_config());
        assert_engine_parity(&model.engine, &legacy);
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        drop(model);
        drop(legacy);
        assert_eq!(generate(&mut session).len(), 2);
    }
}

#[test]
fn manifest_keeps_relative_auxiliary_paths_raw_fields_and_defaults() {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("weights")).unwrap();
    let primary = directory.path().join("weights/model.gguf");
    fs::write(&primary, tiny_llama()).unwrap();
    let absolute_aux = directory.path().join("absolute-decoder.gguf");
    let mut value = manifest("weights/model.gguf");
    value["load_time_parameters"]
        .as_object_mut()
        .unwrap()
        .extend(
            json!({
                "multimodal_projector": "relative-projector.gguf",
                "audio_decoder": absolute_aux,
                "audio_tokenizer": "relative-tokenizer.gguf",
                "draft_model": "missing-draft.gguf",
                "future_file": "future.bin",
                "future_object": {"preserved": true},
                "chat_template": "manifest override",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
    value["generation_time_parameters"] = json!({"sampling_parameters": {
        "temperature": 0.25, "min_p": 0.15, "top_p": 0.8, "top_k": 3, "repetition_penalty": 1.2,
    }});
    let json_path = write_manifest(directory.path(), &value);
    let legacy = CeraEngine::from_path(&json_path, cpu_config()).unwrap();
    let model = load(ModelSource::path(json_path), cpu_config());
    assert_engine_parity(&model.engine, &legacy);
    let resolved = model.engine.manifest();
    assert_eq!(resolved.raw, value);
    assert_eq!(resolved.files.model, primary.to_str().unwrap());
    assert_eq!(
        resolved.files.audio_decoder.as_deref(),
        absolute_aux.to_str()
    );
    for (actual, relative) in [
        (
            &resolved.files.multimodal_projector,
            "relative-projector.gguf",
        ),
        (&resolved.files.audio_tokenizer, "relative-tokenizer.gguf"),
        (&resolved.files.draft_model, "missing-draft.gguf"),
    ] {
        assert_eq!(actual.as_deref(), directory.path().join(relative).to_str());
    }
    assert_eq!(
        resolved.files.extras["future_file"],
        directory.path().join("future.bin").to_str().unwrap()
    );
    assert_eq!(resolved.chat_template.as_deref(), Some("manifest override"));
    // The manifest override stays separate from the tokenizer's embedded value.
    assert_eq!(
        model.engine.tokenizer().chat_template(),
        Some("embedded template")
    );
    let messages = [crate::tokenizer::ChatMessage {
        role: "user".into(),
        content: "hello".into(),
    }];
    for engine in [model.engine.as_ref(), &legacy] {
        assert_eq!(
            crate::tokenizer::apply_chat_template(engine.tokenizer(), &messages, true).unwrap(),
            "embedded template",
        );
    }
    let defaults = model.engine.default_generate_opts();
    assert_eq!(defaults.temperature, 0.25);
    assert_eq!(defaults.min_p, 0.15);
    assert_eq!(defaults.top_p, 0.8);
    assert_eq!(defaults.top_k, 3);
    assert_eq!(defaults.repetition_penalty, 1.2);
}

#[test]
fn explicit_files_keep_every_payload_and_the_original_relative_primary() {
    // A relative primary in a subdirectory catches accidentally opening the
    // normalized manifest field instead of the caller's original primary path.
    let directory = tempfile::tempdir_in(".").unwrap();
    let relative_dir = Path::new(directory.path().file_name().unwrap());
    let primary = relative_dir.join("model.gguf");
    assert!(primary.is_relative());
    fs::write(&primary, tiny_llama()).unwrap();
    let mut files = ModelFiles::text(&primary);
    files.multimodal_projector = Some("missing-projector.gguf".into());
    files.audio_decoder = Some("missing-decoder.gguf".into());
    files.audio_tokenizer = Some("missing-tokenizer.gguf".into());
    files.draft_model = Some("missing-draft.gguf".into());
    files
        .extras
        .insert("future_file".into(), "future.bin".into());
    files.chat_template = Some("caller override".into());
    let legacy = CeraEngine::from_files(files.clone(), cpu_config()).unwrap();
    let model = load(ModelSource::files(files), cpu_config());
    assert_engine_parity(&model.engine, &legacy);
    let resolved = model.engine.manifest();
    for (actual, relative) in [
        (
            &resolved.files.multimodal_projector,
            "missing-projector.gguf",
        ),
        (&resolved.files.audio_decoder, "missing-decoder.gguf"),
        (&resolved.files.audio_tokenizer, "missing-tokenizer.gguf"),
        (&resolved.files.draft_model, "missing-draft.gguf"),
    ] {
        assert_eq!(actual.as_deref(), relative_dir.join(relative).to_str());
    }
    assert_eq!(
        resolved.files.extras["future_file"],
        relative_dir.join("future.bin").to_str().unwrap()
    );
    assert_eq!(resolved.chat_template.as_deref(), Some("caller override"));
    assert_eq!(
        resolved.raw["load_time_parameters"]["model"],
        primary.to_str().unwrap()
    );
    assert!(!model.engine.capabilities().image_in);
}

#[test]
fn path_kinds_fail_before_tokenizer_weights_or_backend() {
    for (architecture, expected) in [
        ("bert", ModelKind::Encoder),
        ("modernbert", ModelKind::Encoder),
        ("whisper", ModelKind::Whisper),
        ("silero_vad", ModelKind::Vad),
        ("kws", ModelKind::Hotword),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("model.gguf");
        fs::write(&primary, header(architecture)).unwrap();
        let json_path = write_manifest(directory.path(), &manifest("model.gguf"));
        let mut inferred_files = ModelFiles::text(&primary);
        inferred_files.inference_type = None;
        for source in [
            ModelSource::files(inferred_files),
            ModelSource::path(&primary),
            ModelSource::path(json_path),
            ModelSource::path(directory.path()),
            ModelSource::files(ModelFiles::text(&primary)),
        ] {
            let config = LoadConfig {
                backend: BackendPreference::Metal,
                ..cpu_config()
            };
            let failure = error(ModelLoader::new(source).config(config).build_generative());
            assert!(
                matches!(
                    &failure,
                    LoadError::KindMismatch { actual, architecture: arch, .. }
                        if *actual == expected && arch == architecture
                ),
                "{architecture}: {failure:?}"
            );
        }
    }
}

#[test]
fn unknown_path_architecture_is_not_treated_as_generative() {
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("future.gguf");
    fs::write(&primary, header("future-model")).unwrap();
    for source in [
        ModelSource::path(&primary),
        ModelSource::files(ModelFiles::text(&primary)),
    ] {
        assert!(matches!(error(ModelLoader::new(source).build_generative()),
            LoadError::UnsupportedArchitecture { architecture } if architecture == "future-model"));
    }
}

fn assert_path_error(path: &Path, message: &str) {
    let legacy = CeraEngine::from_path(path, cpu_config())
        .err()
        .expect("legacy must reject");
    let LoadError::Source {
        source_kind: "path",
        error: actual,
    } = error(
        ModelLoader::new(ModelSource::path(path))
            .config(cpu_config())
            .build_generative(),
    )
    else {
        panic!("source failure must retain its source kind and legacy cause")
    };
    assert_eq!(
        std::mem::discriminant(&actual),
        std::mem::discriminant(&legacy)
    );
    assert_eq!(actual.to_string(), legacy.to_string());
    assert!(actual.to_string().contains(message), "{actual}");
}

#[test]
fn directory_extension_and_manifest_failures_keep_legacy_order_and_errors() {
    let directory = tempfile::tempdir().unwrap();
    assert_path_error(directory.path(), "no .json manifest");
    assert_path_error(
        &directory.path().join("missing.gguf"),
        "inference-type auto-detect",
    );
    assert_path_error(&directory.path().join("model.bin"), "expected a .gguf file");
    assert_path_error(&directory.path().join("missing.json"), "parsing manifest");
    let json_path = directory.path().join("a.json");
    fs::write(&json_path, b"not JSON").unwrap();
    assert_path_error(&json_path, "parsing manifest");
    fs::write(directory.path().join("z.JSON"), b"not JSON").unwrap();
    assert_path_error(directory.path(), "a.json, z.JSON");
    let mut value = manifest("missing.gguf");
    value["inference_type"] = json!("future/inference");
    fs::write(&json_path, serde_json::to_vec(&value).unwrap()).unwrap();
    // Unsupported inference wins over a missing primary, as in the legacy API.
    assert_path_error(&json_path, "future/inference");
    for primary in ["https://example.invalid/model.gguf", "file:///model.gguf"] {
        fs::write(&json_path, serde_json::to_vec(&manifest(primary)).unwrap()).unwrap();
        assert_path_error(
            &json_path,
            if primary.starts_with("https") {
                "remote URL"
            } else {
                "file://"
            },
        );
    }
}

#[test]
fn load_capacity_and_session_limit_remain_separate_and_config_is_retained() {
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("model.gguf");
    fs::write(&primary, tiny_llama()).unwrap();
    for (requested, capacity) in [(8, 8), (128, 64)] {
        let config = LoadConfig {
            context_size: requested,
            gpu_depthformer: true,
            draft_model: Some(directory.path().join("optional-missing-draft.gguf")),
            ..cpu_config()
        };
        let model = load(ModelSource::path(&primary), config);
        assert_eq!(model.engine.config().context_size, requested);
        assert_eq!(model.engine.config().backend, BackendPreference::Cpu);
        assert!(model.engine.config().gpu_depthformer);
        assert_eq!(
            model.engine.config().draft_model.as_deref(),
            Some(
                directory
                    .path()
                    .join("optional-missing-draft.gguf")
                    .as_path()
            )
        );
        assert_eq!(model.engine.model().config().max_seq_len, capacity);
        for (limit, expected) in [(None, capacity), (Some(4), 4), (Some(256), capacity)] {
            let mut session = model
                .create_session(SessionConfig {
                    max_seq_len: limit,
                    ..SessionConfig::default()
                })
                .unwrap();
            assert!(matches!(session.append_tokens(&vec![0; expected + 1]),
                Err(CeraError::ContextOverflow { max_seq_len, .. }) if max_seq_len as usize == expected));
        }
    }
}

#[cfg(feature = "remote")]
#[test]
fn local_load_retains_configured_repository_and_progress_without_network() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[derive(Debug, Default)]
    struct Progress(AtomicUsize);
    impl crate::bundle::DownloadProgress for Progress {
        fn on_progress(&self, _: &str, _: u64, _: Option<u64>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("model.gguf");
    fs::write(&primary, tiny_llama()).unwrap();
    let progress = Arc::new(Progress::default());
    let callback: Arc<dyn crate::bundle::DownloadProgress> = progress.clone();
    let store = directory.path().join("store");
    let config = LoadConfig {
        bundle_repo: Some(crate::bundle::BundleRepo::with_progress(
            &store,
            callback.clone(),
        )),
        ..cpu_config()
    };
    let model = load(ModelSource::files(ModelFiles::text(&primary)), config);
    let repo = model.engine.config().bundle_repo.as_ref().unwrap();
    assert_eq!(repo.store_dir(), store);
    assert!(Arc::ptr_eq(&repo.progress().unwrap(), &callback));
    assert_eq!(progress.0.load(Ordering::SeqCst), 0);
    assert!(!store.exists());
}

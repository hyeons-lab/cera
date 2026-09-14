//! Source selection contracts consumed by legacy and typed remote loading.

use super::*;

fn info(names: &[&str]) -> HfModelInfo {
    serde_json::from_value(serde_json::json!({
        "id": "fixture/model",
        "siblings": names.iter().map(|name| serde_json::json!({"rfilename":name})).collect::<Vec<_>>()
    })).unwrap()
}

#[test]
fn vision_pairing_preserves_quant_and_fallback_order() {
    let spec = HfSpec::parse("fixture/vision@release").unwrap();
    for (companions, expected) in [
        (
            vec![
                "mmproj-Q4_0.gguf",
                "mmproj-F16.gguf",
                "mmproj-Q8_0.gguf",
                "mmproj-Q4_K_M.gguf",
            ],
            "mmproj-Q4_K_M.gguf",
        ),
        (
            vec!["mmproj-Q4_0.gguf", "mmproj-BF16.gguf", "mmproj-Q8_0.gguf"],
            "mmproj-Q8_0.gguf",
        ),
        (
            vec!["mmproj-Q4_0.gguf", "mmproj-BF16.gguf"],
            "mmproj-BF16.gguf",
        ),
        (
            vec!["mmproj-Q4_0.gguf", "mmproj-Q5_K_M.gguf"],
            "mmproj-Q4_0.gguf",
        ),
    ] {
        let mut names = vec!["model-Q8_0.gguf", "model-Q4_K_M.gguf"];
        names.extend(companions);
        let manifest = resolve_hf_manifest(&spec, &info(&names), Some("q4-k-m"), None).unwrap();
        assert_eq!(
            manifest.files.model,
            spec.file_download_url("model-Q4_K_M.gguf")
        );
        assert_eq!(
            manifest.files.multimodal_projector.as_deref(),
            Some(spec.file_download_url(expected).as_str())
        );
        assert_eq!(manifest.inference_type, InferenceType::LlamaCppImageToText);
    }
}

#[test]
fn audio_and_draft_pairing_preserve_distinct_fallbacks_and_extras() {
    let spec = HfSpec::parse("fixture/audio").unwrap();
    let names = [
        "model-Q6_K.gguf",
        "audio_decoder-F16.gguf",
        "audio_decoder-Q4_0.gguf",
        "vocoder-F16.gguf",
        "tokenizer-F16.gguf",
        "tokenizer-Q4_0.gguf",
        "draft-F16.gguf",
        "draft-Q8_0.gguf",
        "draft-Q4_K_M.gguf",
    ];
    let manifest = resolve_hf_manifest(&spec, &info(&names), None, None).unwrap();
    assert_eq!(manifest.inference_type, InferenceType::LlamaCppLfm2AudioV1);
    assert_eq!(
        manifest.files.audio_decoder,
        Some(spec.file_download_url("audio_decoder-Q4_0.gguf"))
    );
    assert_eq!(
        manifest.files.audio_tokenizer,
        Some(spec.file_download_url("tokenizer-Q4_0.gguf"))
    );
    assert_eq!(
        manifest.files.draft_model,
        Some(spec.file_download_url("draft-Q4_K_M.gguf"))
    );
    // Legacy extras retain the first classified candidate, even when the
    // selected audio-tokenizer slot uses another quantization.
    assert_eq!(
        manifest.files.extras["vocoder"],
        spec.file_download_url("vocoder-F16.gguf")
    );
    assert_eq!(
        manifest.files.extras["tokenizer"],
        spec.file_download_url("tokenizer-F16.gguf")
    );
    let fallback = resolve_hf_manifest(
        &spec,
        &info(&[
            "model-Q6_K.gguf",
            "vocoder-F16.gguf",
            "audiotokenizer.safetensors",
        ]),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        fallback.files.audio_decoder,
        Some(spec.file_download_url("vocoder-F16.gguf"))
    );
    assert_eq!(
        fallback.files.audio_tokenizer,
        Some(spec.file_download_url("audiotokenizer.safetensors"))
    );
}

#[test]
fn explicit_nested_file_wins_and_requires_a_path_boundary() {
    let mut spec = HfSpec::parse("fixture/text").unwrap();
    let info = info(&["prefixmodel-Q8_0.gguf", "nested/model-Q4_K_M.gguf"]);
    spec.subpath = Some("model-Q4_K_M.gguf".into());
    let manifest = resolve_hf_manifest(&spec, &info, Some("Q8_0"), None).unwrap();
    assert_eq!(
        manifest.files.model,
        spec.file_download_url("nested/model-Q4_K_M.gguf")
    );
    spec.subpath = Some("model-Q8_0.gguf".into());
    assert!(
        resolve_hf_manifest(&spec, &info, None, None)
            .err()
            .unwrap()
            .to_string()
            .contains("requested file `model-Q8_0.gguf` not found")
    );
}

#[cfg(feature = "remote")]
#[test]
fn streaming_options_keep_strategy_defaults_and_callback_ownership() {
    use crate::convert::{QuantStrategy, TargetQuant};
    use std::sync::Arc;
    #[derive(Debug)]
    struct Progress;
    impl crate::bundle::DownloadProgress for Progress {
        fn on_progress(&self, _: &str, _: u64, _: Option<u64>) {
            panic!("option construction must not report download progress");
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("not-created");
    let progress: Arc<dyn crate::bundle::DownloadProgress> = Arc::new(Progress);
    for (name, expected) in [
        ("auto", QuantStrategy::Auto),
        ("fast-mse", QuantStrategy::FastMse),
        ("hqq", QuantStrategy::Hqq),
        ("quarot", QuantStrategy::QuaRot),
        ("unknown", QuantStrategy::Auto),
    ] {
        let opts = streaming_quantize_options(
            Some("Q8_0"),
            Some(name),
            Some(&cache),
            Some(progress.clone()),
            || Some("fixture-token".into()),
        )
        .unwrap();
        assert_eq!(opts.target_quant, TargetQuant::Q8_0);
        assert_eq!(opts.strategy, expected);
        assert_eq!(opts.cache_dir, cache);
        assert_eq!(opts.auth_token.as_deref(), Some("fixture-token"));
        assert!(Arc::ptr_eq(&opts.progress.unwrap(), &progress));
        assert!(opts.cancel.is_none());
        assert!(opts.tensor_overrides.is_empty());
    }
    let defaults = streaming_quantize_options(None, None, None, None, || None).unwrap();
    assert_eq!(defaults.target_quant, TargetQuant::Q4_K_M);
    assert_eq!(defaults.strategy, QuantStrategy::Auto);
    assert_eq!(defaults.cache_dir, default_cache_dir());
    assert!(defaults.auth_token.is_none());
    assert!(defaults.progress.is_none());
    assert!(
        streaming_quantize_options(Some("invalid"), None, Some(&cache), None, || panic!(
            "invalid quant must fail before reading auth"
        ))
        .is_err()
    );
    assert!(!cache.exists());
}

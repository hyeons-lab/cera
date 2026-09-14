//! Primary-parser counts use CPU fixtures without auxiliary or backend reopens.

use super::*;

#[track_caller]
fn once<T>(load: impl FnOnce() -> T) -> T {
    parses(1, load)
}

#[track_caller]
fn parses<T>(expected: usize, load: impl FnOnce() -> T) -> T {
    let before = crate::gguf::parse_probe::count();
    let result = load();
    assert_eq!(crate::gguf::parse_probe::count() - before, expected);
    result
}

#[test]
fn multipart_kind_check_precedes_auxiliary_parsing() {
    let mut parts = ModelBytes::text(header("bert"));
    parts.inference_type = None;
    parts.multimodal_projector = Some(header("llama"));
    // Legacy multipart loading parses the sidecar before deciding inference
    // type. Typed loading must reject the primary before parsing that sidecar.
    let legacy = parses(2, || CeraEngine::from_parts(parts.clone(), cpu_config()));
    assert!(matches!(
        legacy,
        Err(CeraError::UnsupportedInferenceType(_))
    ));
    let failure = once(|| {
        error(
            ModelLoader::new(ModelSource::parts(parts))
                .config(cpu_config())
                .build_generative(),
        )
    });
    assert!(matches!(
        failure,
        LoadError::KindMismatch {
            actual: ModelKind::Encoder,
            ..
        }
    ));
}

#[cfg(feature = "mmap")]
#[test]
fn file_kind_and_inference_gates_keep_auxiliary_error_order() {
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("model.gguf");
    std::fs::write(&primary, header("bert")).unwrap();
    let mut files = crate::ModelFiles::text(&primary);
    files.inference_type = None;
    files.multimodal_projector = Some("file:///unsupported-sidecar.gguf".into());
    let legacy = once(|| CeraEngine::from_files(files.clone(), cpu_config()));
    assert!(matches!(legacy, Err(CeraError::Backend(message)) if message.contains("file://")));
    let typed =
        once(|| error(ModelLoader::new(ModelSource::files(files.clone())).build_generative()));
    assert!(matches!(
        typed,
        LoadError::KindMismatch {
            actual: ModelKind::Encoder,
            ..
        }
    ));

    // Explicit inference defers the primary open until after file resolution.
    files.inference_type = Some(InferenceType::LlamaCppTextToText);
    let typed = parses(0, || {
        error(ModelLoader::new(ModelSource::files(files.clone())).build_generative())
    });
    assert!(
        matches!(typed, LoadError::Source { source_kind: "files", error: CeraError::Backend(message) } if message.contains("file://"))
    );

    files.multimodal_projector = None;
    files.inference_type = Some(InferenceType::Unknown("future/inference".into()));
    files.model = directory.path().join("missing.gguf");
    let legacy = parses(0, || CeraEngine::from_files(files.clone(), cpu_config()));
    assert!(
        matches!(legacy, Err(CeraError::UnsupportedInferenceType(it)) if it == "future/inference")
    );
    let typed = parses(0, || {
        error(ModelLoader::new(ModelSource::files(files)).build_generative())
    });
    assert!(
        matches!(typed, LoadError::Source { source_kind: "files", error: CeraError::UnsupportedInferenceType(it) } if it == "future/inference")
    );
}

// APFS rejects these filenames. Linux exercises the legacy distinction between
// a manifest's lossy string path and ModelFiles' original byte path.
#[cfg(all(feature = "mmap", target_os = "linux"))]
#[test]
fn non_utf8_primary_preserves_bare_path_failure_and_explicit_files_success() {
    use std::os::unix::ffi::OsStrExt;

    let directory = tempfile::tempdir().unwrap();
    let primary = directory
        .path()
        .join(std::ffi::OsStr::from_bytes(b"model-\xff.gguf"));
    std::fs::write(&primary, tiny_llama()).unwrap();
    let expected = format!("opening `{}`:", primary.to_string_lossy());
    let legacy = CeraEngine::from_path(&primary, cpu_config()).err().unwrap();
    assert!(
        matches!(&legacy, CeraError::Backend(message) if message.starts_with(&expected)),
        "{legacy}"
    );
    let typed = error(
        ModelLoader::new(ModelSource::path(&primary))
            .config(cpu_config())
            .build_generative(),
    );
    assert!(
        matches!(typed, LoadError::Source { source_kind: "path", error: e } if e.to_string() == legacy.to_string())
    );
    let mut files = crate::ModelFiles::text(&primary);
    files.inference_type = None;
    once(|| CeraEngine::from_files(files.clone(), cpu_config())).unwrap();
    once(|| {
        ModelLoader::new(ModelSource::files(files))
            .config(cpu_config())
            .build_generative()
    })
    .unwrap();
}

#[test]
fn memory_primary_is_parsed_once() {
    let bytes = tiny_llama();
    once(|| CeraEngine::from_bytes(bytes.clone(), cpu_config())).unwrap();
    once(|| CeraEngine::from_reader(Cursor::new(bytes.clone()), cpu_config())).unwrap();
    once(|| CeraEngine::from_parts(ModelBytes::text(bytes.clone()), cpu_config())).unwrap();

    for dynamic in [false, true] {
        for source in [
            ModelSource::bytes(bytes.clone()),
            ModelSource::reader(Cursor::new(bytes.clone())),
            ModelSource::parts(ModelBytes::text(bytes.clone())),
        ] {
            let loader = ModelLoader::new(source).config(cpu_config());
            let model = once(|| {
                if dynamic {
                    loader.build().unwrap().as_generative().unwrap()
                } else {
                    loader.build_generative().unwrap()
                }
            });
            let mut session = model.create_session(SessionConfig::default()).unwrap();
            drop(model);
            assert_eq!(generate(&mut session).len(), 2);
        }
    }
}

#[cfg(feature = "mmap")]
#[test]
fn legacy_filesystem_primary_is_parsed_once() {
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("model.gguf");
    std::fs::write(&primary, tiny_llama()).unwrap();
    let manifest =
        filesystem::write_manifest(directory.path(), &filesystem::manifest("model.gguf"));
    for path in [&primary, &manifest, directory.path()] {
        once(|| CeraEngine::from_path(path, cpu_config())).unwrap();
    }
    for inference in [None, Some(InferenceType::LlamaCppTextToText)] {
        let mut files = crate::ModelFiles::text(&primary);
        files.inference_type = inference;
        once(|| CeraEngine::from_files(files, cpu_config())).unwrap();
    }
}

#[cfg(feature = "mmap")]
#[test]
fn typed_filesystem_primary_is_parsed_once() {
    let directory = tempfile::tempdir().unwrap();
    let primary = directory.path().join("model.gguf");
    std::fs::write(&primary, tiny_llama()).unwrap();
    let manifest =
        filesystem::write_manifest(directory.path(), &filesystem::manifest("model.gguf"));
    for dynamic in [false, true] {
        let mut inferred = crate::ModelFiles::text(&primary);
        inferred.inference_type = None;
        for source in [
            ModelSource::path(&primary),
            ModelSource::path(&manifest),
            ModelSource::path(directory.path()),
            ModelSource::files(inferred),
            ModelSource::files(crate::ModelFiles::text(&primary)),
        ] {
            let loader = ModelLoader::new(source).config(cpu_config());
            let model = once(|| {
                if dynamic {
                    loader.build().unwrap().as_generative().unwrap()
                } else {
                    loader.build_generative().unwrap()
                }
            });
            let mut session = model.create_session(SessionConfig::default()).unwrap();
            drop(model);
            assert_eq!(generate(&mut session).len(), 2);
        }
    }
}

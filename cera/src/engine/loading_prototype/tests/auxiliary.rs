//! Real CPU companion loading, selection and session execution contracts.

use super::*;
use crate::spec::Drafter;
use std::sync::Mutex;

mod audio;
mod draft;
use super::companion_fixture as fixture;
mod vision;

fn config() -> LoadConfig {
    LoadConfig {
        context_size: 256,
        ..cpu_config()
    }
}

fn parts_engines(parts: ModelBytes, config: LoadConfig) -> Vec<CeraEngine> {
    let legacy = CeraEngine::from_parts(parts.clone(), config.clone()).unwrap();
    let typed = ModelLoader::new(ModelSource::parts(parts.clone()))
        .config(config.clone())
        .build_generative()
        .unwrap();
    let handle = ModelLoader::new(ModelSource::parts(parts))
        .config(config)
        .build()
        .unwrap();
    let dynamic = handle.as_generative().unwrap();
    drop(handle);
    vec![
        legacy,
        Arc::try_unwrap(typed.engine).ok().unwrap(),
        Arc::try_unwrap(dynamic.engine).ok().unwrap(),
    ]
}

#[cfg(feature = "mmap")]
fn file_engines(files: crate::ModelFiles, config: LoadConfig) -> Vec<CeraEngine> {
    let legacy = CeraEngine::from_files(files.clone(), config.clone()).unwrap();
    let typed = ModelLoader::new(ModelSource::files(files.clone()))
        .config(config.clone())
        .build_generative()
        .unwrap();
    let handle = ModelLoader::new(ModelSource::files(files))
        .config(config)
        .build()
        .unwrap();
    let dynamic = handle.as_generative().unwrap();
    drop(handle);
    vec![
        legacy,
        Arc::try_unwrap(typed.engine).ok().unwrap(),
        Arc::try_unwrap(dynamic.engine).ok().unwrap(),
    ]
}

#[cfg(feature = "mmap")]
fn path_engines(path: &Path, config: LoadConfig) -> Vec<CeraEngine> {
    let legacy = CeraEngine::from_path(path, config.clone()).unwrap();
    let typed = ModelLoader::new(ModelSource::path(path))
        .config(config.clone())
        .build_generative()
        .unwrap();
    let handle = ModelLoader::new(ModelSource::path(path))
        .config(config)
        .build()
        .unwrap();
    let dynamic = handle.as_generative().unwrap();
    drop(handle);
    vec![
        legacy,
        Arc::try_unwrap(typed.engine).ok().unwrap(),
        Arc::try_unwrap(dynamic.engine).ok().unwrap(),
    ]
}

// Unix ownership cases remove the source directory before execution to rule
// out reopening. Other platforms retain its guard until all mappings drop.
fn executable_engines(parts: ModelBytes) -> (Vec<CeraEngine>, Option<tempfile::TempDir>) {
    let engines = parts_engines(parts.clone(), config());
    #[cfg(feature = "mmap")]
    {
        use std::fs;
        let mut engines = engines;
        let dir = tempfile::tempdir().unwrap();
        let primary = dir.path().join("primary.gguf");
        fs::write(&primary, &parts.model).unwrap();
        let mut files = crate::ModelFiles::text(&primary);
        files.inference_type = parts.inference_type;
        let mut manifest = super::filesystem::manifest("primary.gguf");
        if let Some(bytes) = parts.multimodal_projector {
            fs::write(dir.path().join("vision.gguf"), bytes).unwrap();
            files.multimodal_projector = Some(dir.path().join("vision.gguf"));
            manifest["inference_type"] = "llama.cpp/image-to-text".into();
            manifest["load_time_parameters"]["multimodal_projector"] = "vision.gguf".into();
        }
        if let Some(bytes) = parts.draft_model {
            fs::write(dir.path().join("draft.gguf"), bytes).unwrap();
            files.draft_model = Some(dir.path().join("draft.gguf"));
            manifest["load_time_parameters"]["draft_model"] = "draft.gguf".into();
        }
        engines.extend(file_engines(files, config()));
        let json = super::filesystem::write_manifest(dir.path(), &manifest);
        engines.extend(path_engines(&json, config()));
        engines.extend(path_engines(dir.path(), config()));
        #[cfg(unix)]
        {
            dir.close().unwrap();
            (engines, None)
        }
        #[cfg(not(unix))]
        (engines, Some(dir))
    }
    #[cfg(not(feature = "mmap"))]
    (engines, None)
}

#[track_caller]
pub(super) fn close(actual: &[f32], expected: &[f32]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} != {b}");
    }
}

#[track_caller]
pub(super) fn distinct(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
    assert!(
        actual
            .iter()
            .zip(expected)
            .any(|(a, b)| (a - b).abs() > 1e-3)
    );
}

struct ObservedDraft {
    inner: Box<dyn Drafter>,
    calls: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl Drafter for ObservedDraft {
    fn clone_drafter(&self) -> Box<dyn Drafter> {
        Box::new(Self {
            inner: self.inner.clone_drafter(),
            calls: self.calls.clone(),
        })
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn suggested_k(&self) -> Option<usize> {
        self.inner.suggested_k()
    }
    fn draft(&mut self, tokens: &[u32], max_k: usize) -> Vec<u32> {
        let result = self.inner.draft(tokens, max_k);
        self.calls.lock().unwrap().push(result.clone());
        result
    }
}

pub(super) fn observe_drafter(engine: &mut CeraEngine) -> Arc<Mutex<Vec<Vec<u32>>>> {
    let calls = Arc::new(Mutex::new(Vec::new()));
    engine.drafter = Some(Arc::new(ObservedDraft {
        inner: engine.drafter.take().unwrap().clone_drafter(),
        calls: calls.clone(),
    }));
    calls
}

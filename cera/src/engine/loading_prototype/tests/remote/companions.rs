//! Join remote discovery/integrity with executable CPU companion contracts.

use super::super::auxiliary::{close, distinct, observe_drafter};
use super::super::{audio_fixture, companion_fixture as fixture};
use super::*;
use http::{Response, isolated_with_routes};
use std::collections::HashMap;

mod audio;
mod draft;
mod vision;

fn cfg(root: &Path, progress: &Arc<Progress>) -> LoadConfig {
    LoadConfig {
        context_size: 256,
        ..config(root, progress)
    }
}

// Quant names exercise selection; payloads deliberately contain F32 tensors.
fn routes(repo: &str, pipeline: &str, files: &[(&str, Arc<[u8]>)]) -> HashMap<String, Response> {
    let mut routes = HashMap::from([(
        format!("/api/models/fixture/{repo}"),
        Response::bytes(json!({
            "id": format!("fixture/{repo}"), "pipeline_tag": pipeline, "sha": MAIN_COMMIT,
            "siblings": files.iter().map(|(name, _)| json!({"rfilename":name})).collect::<Vec<_>>()
        }).to_string()),
    )]);
    for (name, bytes) in files {
        routes.insert(
            format!("/fixture/{repo}/resolve/{MAIN_COMMIT}/{name}"),
            Response::gguf(bytes),
        );
    }
    routes
}

fn unwrap_model(model: GenerativeModel) -> CeraEngine {
    Arc::try_unwrap(model.engine).ok().unwrap()
}

fn hf_engines(spec: &str, cfg: &LoadConfig) -> Vec<CeraEngine> {
    let typed = load(ModelSource::hugging_face(spec, None, None), cfg.clone());
    let handle = ModelLoader::new(ModelSource::hugging_face(spec, None, None))
        .config(cfg.clone())
        .build()
        .unwrap();
    let dynamic = handle.as_generative().unwrap();
    drop(handle);
    vec![
        unwrap_model(typed),
        unwrap_model(dynamic),
        CeraEngine::from_hf(spec, None, cfg.clone()).unwrap(),
    ]
}

fn assert_cached(repo: &BundleRepo, progress: &Progress, url: &str, bytes: &[u8]) {
    let path = repo.fixture_path(url).unwrap();
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(
        fs::read_to_string(crate::bundle::download::sidecar_path(&path))
            .unwrap()
            .trim(),
        digest(bytes)
    );
    let events = progress.0.lock().unwrap();
    let events: Vec<_> = events.iter().filter(|(u, _, _)| u == url).collect();
    assert!(!events.is_empty(), "no progress for {url}");
    assert!(events.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    assert_eq!(events.last().unwrap().1, bytes.len() as u64);
}

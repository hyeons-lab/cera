use super::*;
use crate::kv_cache::{InferenceState, KvPrefixCache};
use crate::model::cache_identity::HASHED_BYTES;
use crate::model::lfm2::PREFILLED_TOKENS;
use std::collections::BTreeMap;
use std::path::Path;

fn named(path: &Path) -> GenerativeModel {
    ModelLoader::new(ModelSource::path(path))
        .config(cpu_config())
        .build_generative()
        .unwrap()
}

fn cold(dir: &Path) -> KvCacheConfig {
    KvCacheConfig {
        cache_dir: Some(dir.to_owned()),
        max_warm_entries: 0,
        ..KvCacheConfig::default()
    }
}

fn files(dir: &Path) -> BTreeMap<std::ffi::OsString, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), std::fs::read(entry.path()).unwrap())
        })
        .collect()
}

fn control(bytes: Arc<[u8]>) -> GenerativeModel {
    let model = load(bytes);
    model.configure_cache(KvCacheConfig {
        max_warm_entries: 0,
        ..KvCacheConfig::default()
    });
    model
}

fn append_count(session: &mut Session, tokens: &[u32]) -> usize {
    let before = PREFILLED_TOKENS.get();
    session.append_tokens(tokens).unwrap();
    PREFILLED_TOKENS.get() - before
}

fn changed_late_ffn(bytes: &Arc<[u8]>) -> Arc<[u8]> {
    let original = GgufFile::from_bytes(bytes.clone()).unwrap();
    let changed = GgufFile::from_bytes(fixture::tiny_hybrid_with_seed(29)).unwrap();
    // Preserve the header, embeddings, length and every other tensor.
    let tensor = "blk.1.ffn_down.weight";
    let info = &original.tensors[tensor];
    let mut bytes = bytes.to_vec();
    bytes[info.offset as usize..info.offset as usize + info.size_bytes]
        .copy_from_slice(changed.tensor_data(tensor).unwrap());
    bytes.into()
}

#[test]
fn same_path_replacement_separates_cold_state_and_preserves_live_models() {
    for compression in [KvCompression::None, KvCompression::F16] {
        let weights = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = weights.path().join("model.gguf");
        let first_bytes = fixture::tiny_hybrid();
        let other_bytes = changed_late_ffn(&first_bytes);
        assert_eq!(first_bytes.len(), other_bytes.len());
        std::fs::write(&path, &first_bytes).unwrap();
        let first = named(&path);
        first.configure_cache(cold(dir.path()));
        let first_control = control(first_bytes);
        let mut first_live = session(&first, compression.clone());
        let mut first_expected = session(&first_control, compression.clone());
        for s in [&mut first_live, &mut first_expected] {
            s.append_tokens(&[0, 1]).unwrap();
        }
        let original_files = files(dir.path());
        assert_eq!(original_files.len(), 1);

        // Rename a new inode over the path; never truncate a live mapped file.
        let replacement = weights.path().join("replacement.gguf");
        std::fs::write(&replacement, &other_bytes).unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let other = named(&path);
        other.configure_cache(cold(dir.path()));
        let other_control = control(other_bytes);
        let mut actual = session(&other, compression.clone());
        let mut expected = session(&other_control, compression);
        assert_eq!(append_count(&mut actual, &[0, 1, 0]), 3);
        expected.append_tokens(&[0, 1, 0]).unwrap();
        same_live(&actual, &expected);
        for s in [&mut first_live, &mut first_expected] {
            s.append_tokens(&[0]).unwrap();
        }
        same_live(&first_live, &first_expected);
        distinct(
            first_live.last_logits().unwrap(),
            actual.last_logits().unwrap(),
        );
        let combined = files(dir.path());
        assert_eq!(combined.len(), 2);
        assert!(
            original_files
                .iter()
                .all(|(key, value)| combined.get(key) == Some(value))
        );
        other.clear_cache();
        assert_eq!(files(dir.path()), original_files);
        first.clear_cache();
        assert!(files(dir.path()).is_empty());
        drop(first);
        drop(other);
        for (live, control) in [
            (&mut first_live, &mut first_expected),
            (&mut actual, &mut expected),
        ] {
            live.append_tokens(&[1]).unwrap();
            control.append_tokens(&[1]).unwrap();
            same_live(live, control);
        }
    }
}

#[test]
fn unchanged_weights_reload_cold_prefix_without_rehashing_on_append() {
    for compression in [KvCompression::None, KvCompression::F16] {
        let weights = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = weights.path().join("model.gguf");
        let bytes = fixture::tiny_hybrid();
        std::fs::write(&path, &bytes).unwrap();
        let before = HASHED_BYTES.get();
        let model = named(&path);
        model.configure_cache(KvCacheConfig::default());
        let mut warm = session(&model, compression.clone());
        warm.append_tokens(&[0, 1]).unwrap();
        assert_eq!(HASHED_BYTES.get(), before, "warm use must not hash weights");
        model.configure_cache(cold(dir.path()));
        assert_eq!(HASHED_BYTES.get() - before, bytes.len());
        let mut populate = session(&model, compression.clone());
        populate.append_tokens(&[0, 1]).unwrap();
        assert_eq!(files(dir.path()).len(), 1);
        model.configure_cache(cold(dir.path()));
        assert_eq!(HASHED_BYTES.get() - before, bytes.len());
        drop(populate);
        drop(warm);
        drop(model);

        let reloaded = named(&path);
        reloaded.configure_cache(cold(dir.path()));
        assert_eq!(HASHED_BYTES.get() - before, bytes.len() * 2);
        let expected_model = control(bytes);
        let mut actual = session(&reloaded, compression.clone());
        let mut expected = session(&expected_model, compression);
        let hashed = HASHED_BYTES.get();
        assert_eq!(
            append_count(&mut actual, &[0, 1, 0]),
            1,
            "restore two tokens from disk"
        );
        expected.append_tokens(&[0, 1, 0]).unwrap();
        same_live(&actual, &expected);
        assert_eq!(generate(&mut actual).tokens, generate(&mut expected).tokens);
        same_live(&actual, &expected);
        assert_eq!(
            HASHED_BYTES.get(),
            hashed,
            "append/generate must not scan weights"
        );
    }
}

#[test]
fn old_path_only_files_are_ignored_and_preserved() {
    for compression in [KvCompression::None, KvCompression::F16] {
        let weights = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = weights.path().join("model.gguf");
        let source = control(fixture::tiny_hybrid());
        let mut state =
            InferenceState::from_config_with_compression(source.model().config(), &compression)
                .unwrap();
        source.model().forward_prefill(&[0, 1], 0, &mut state);
        let mut legacy = KvPrefixCache::new(
            cold(dir.path()),
            source.model().config(),
            &format!("cpu:{}{}", compression.cache_tag(), path.to_string_lossy()),
        );
        legacy.insert(&[0, 1], state.snapshot().unwrap());
        assert!(legacy.find_longest_prefix(&[0, 1, 0]).is_some());
        let old = files(dir.path());
        assert_eq!(old.len(), 1);
        let bytes = fixture::tiny_hybrid_with_seed(29);
        std::fs::write(&path, &bytes).unwrap();
        let model = named(&path);
        model.configure_cache(cold(dir.path()));
        let expected_model = control(bytes);
        let mut actual = session(&model, compression.clone());
        let mut expected = session(&expected_model, compression);
        assert_eq!(append_count(&mut actual, &[0, 1, 0]), 3);
        expected.append_tokens(&[0, 1, 0]).unwrap();
        same_live(&actual, &expected);
        model.clear_cache();
        assert_eq!(files(dir.path()), old);
    }
}

#[test]
fn first_hash_allows_session_creation_and_uses_the_latest_compression_tag() {
    use crate::model::cache_identity::HASH_PAUSE;
    use std::sync::mpsc;
    use std::time::Duration;

    let weights = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = weights.path().join("model.gguf");
    std::fs::write(&path, fixture::tiny_hybrid()).unwrap();
    let model = named(&path);
    let configured = model.clone();
    let config = cold(dir.path());
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let configuring = std::thread::spawn(move || {
        HASH_PAUSE.with_borrow_mut(|hook| {
            *hook = Some(Box::new(move || {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            }));
        });
        configured.configure_cache(config);
    });
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let creating = std::thread::spawn(move || {
        let session = session(&model, KvCompression::F16);
        ready_tx.send(()).unwrap();
        session
    });
    let ready = ready_rx.recv_timeout(Duration::from_secs(5));
    // Release and join both workers before reporting a failure.
    resume_tx.send(()).unwrap();
    configuring.join().unwrap();
    let mut first = creating.join().unwrap();
    assert!(ready.is_ok(), "hashing blocked session creation: {ready:?}");
    first.append_tokens(&[0, 1]).unwrap();
    let reloaded = named(&path);
    reloaded.configure_cache(cold(dir.path()));
    let mut next = session(&reloaded, KvCompression::F16);
    assert_eq!(append_count(&mut next, &[0, 1, 0]), 1);
    let expected_model = control(fixture::tiny_hybrid());
    let mut expected = session(&expected_model, KvCompression::F16);
    expected.append_tokens(&[0, 1, 0]).unwrap();
    same_live(&next, &expected);
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn shared_metal_mapping_keeps_the_parsed_inode_after_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    let original = fixture::tiny_hybrid();
    std::fs::write(&path, &original).unwrap();
    let parsed = GgufFile::open(&path).unwrap();
    let replacement = dir.path().join("replacement.gguf");
    std::fs::write(&replacement, fixture::tiny_hybrid_with_seed(29)).unwrap();
    std::fs::rename(replacement, &path).unwrap();
    let backing = parsed.mapped_backing().unwrap();
    assert_eq!(backing.as_ptr(), parsed.mmap_data().as_ptr());
    assert_ne!(backing.as_ref().as_ref(), std::fs::read(path).unwrap());
    drop(parsed);
    assert_eq!(backing.as_ref().as_ref(), original.as_ref());
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a Metal device; executes the replaced-path ownership control"]
fn metal_execution_uses_parsed_weights_after_path_replacement() {
    use crate::model::Model;
    use crate::model::metal_lfm2::MetalLfm2Model;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    let original = fixture::tiny_hybrid();
    let changed = changed_late_ffn(&original);
    std::fs::write(&path, &original).unwrap();
    let parsed = GgufFile::open(&path).unwrap();
    let replacement = dir.path().join("replacement.gguf");
    std::fs::write(&replacement, &changed).unwrap();
    std::fs::rename(replacement, &path).unwrap();
    let actual = MetalLfm2Model::from_gguf(parsed, Some(&path), 64).unwrap();
    let control =
        MetalLfm2Model::from_gguf(GgufFile::from_bytes(original).unwrap(), None, 64).unwrap();
    let other =
        MetalLfm2Model::from_gguf(GgufFile::from_bytes(changed).unwrap(), None, 64).unwrap();
    let execute = |model: &MetalLfm2Model| {
        model.configure_cache(KvCacheConfig {
            max_warm_entries: 0,
            ..KvCacheConfig::default()
        });
        model
            .configure_kv_compression(&KvCompression::None)
            .unwrap();
        let mut state = InferenceState::from_config(model.config()).unwrap();
        model.forward_prefill(&[0, 1, 0], 0, &mut state)
    };
    let expected = execute(&control);
    distinct(&expected, &execute(&other));
    close(&execute(&actual), &expected);
}

#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
#[test]
fn dspark_gpu_identity_includes_base_weights_and_draft_weights() {
    use crate::model::cache_identity::for_gpu_source;
    use crate::model::dspark::{DSparkDraftModel, DSparkGpuWeightSource};
    use crate::model::gpu_weight_source::GpuWeightSource;

    let base = super::super::companion_fixture::primary();
    let parsed = GgufFile::from_bytes(base.clone()).unwrap();
    let embedding = &parsed.tensors["token_embd.weight"];
    let mut changed = base.to_vec();
    for byte in
        &mut changed[embedding.offset as usize..embedding.offset as usize + embedding.size_bytes]
    {
        *byte = 0;
    }
    let source = |draft, base| {
        let dspark = Arc::new(
            DSparkDraftModel::from_ggufs(
                Arc::new(GgufFile::from_bytes(draft).unwrap()),
                Arc::new(GgufFile::from_bytes(base).unwrap()),
            )
            .unwrap(),
        );
        DSparkGpuWeightSource {
            config: dspark.config.to_model_config(64),
            dspark,
        }
    };
    let draft = super::super::companion_fixture::draft(1, 3);
    let original = source(draft.clone(), base.clone());
    let changed_base = source(draft.clone(), changed.into());
    let changed_draft = source(super::super::companion_fixture::draft(0, 3), base.clone());
    let same = source(draft, base);
    assert_eq!(original.gguf().mmap_data(), changed_base.gguf().mmap_data());
    assert_ne!(
        original.embedding_tensor_data().unwrap(),
        changed_base.embedding_tensor_data().unwrap()
    );
    let original_id = for_gpu_source(&original, "same-draft-path");
    assert_eq!(original_id, for_gpu_source(&same, "same-draft-path"));
    assert_ne!(
        original_id,
        for_gpu_source(&changed_base, "same-draft-path")
    );
    assert_ne!(
        original_id,
        for_gpu_source(&changed_draft, "same-draft-path")
    );
    assert_ne!(original_id, for_gpu_source(&original, "other-namespace"));
    let before = HASHED_BYTES.get();
    assert_eq!(for_gpu_source(&original, ""), "");
    assert_eq!(HASHED_BYTES.get(), before);
}

use super::*;
use crate::kv_cache::{InferenceState, KvPrefixCache};
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

type MemoryLoader = fn(Arc<[u8]>) -> GenerativeModel;

fn loaders() -> [MemoryLoader; 6] {
    [
        |bytes| GenerativeModel {
            engine: Arc::new(CeraEngine::from_bytes(bytes, cpu_config()).unwrap()),
        },
        |bytes| GenerativeModel {
            engine: Arc::new(CeraEngine::from_reader(Cursor::new(bytes), cpu_config()).unwrap()),
        },
        |bytes| GenerativeModel {
            engine: Arc::new(
                CeraEngine::from_parts(ModelBytes::text(bytes), cpu_config()).unwrap(),
            ),
        },
        load,
        |bytes| {
            ModelLoader::new(ModelSource::reader(Cursor::new(bytes)))
                .config(cpu_config())
                .build_generative()
                .unwrap()
        },
        |bytes| {
            ModelLoader::new(ModelSource::parts(ModelBytes::text(bytes)))
                .config(cpu_config())
                .build_generative()
                .unwrap()
        },
    ]
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

#[test]
fn anonymous_models_do_not_share_persistent_state() {
    for compression in [KvCompression::None, KvCompression::F16] {
        for loader in loaders() {
            let dir = tempfile::tempdir().unwrap();
            let first = loader(fixture::tiny_hybrid());
            // Identical metadata, including model name; every non-norm tensor
            // changes. Neither architecture nor display name identifies weights.
            let other_bytes = fixture::tiny_hybrid_with_seed(29);
            let other = loader(other_bytes.clone());
            let control = load(other_bytes);
            control.configure_cache(KvCacheConfig {
                max_warm_entries: 0,
                ..KvCacheConfig::default()
            });
            for model in [&first, &other] {
                model.configure_cache(KvCacheConfig {
                    cache_dir: Some(dir.path().to_owned()),
                    ..KvCacheConfig::default()
                });
            }
            let mut first_session = session(&first, compression.clone());
            first_session.append_tokens(&[0, 1]).unwrap();
            drop(first_session);
            drop(first);
            let mut actual = session(&other, compression.clone());
            let mut expected = session(&control, compression.clone());
            for s in [&mut actual, &mut expected] {
                s.append_tokens(&[0, 1, 0]).unwrap();
            }
            same_live(&actual, &expected);
            assert!(files(dir.path()).is_empty());
        }
    }
}

#[test]
fn anonymous_loaders_ignore_old_files_and_preserve_live_state() {
    for compression in [KvCompression::None, KvCompression::F16] {
        for loader in loaders() {
            let dir = tempfile::tempdir().unwrap();
            let source = load(fixture::tiny_hybrid());
            let mut state =
                InferenceState::from_config_with_compression(source.model().config(), &compression)
                    .unwrap();
            source.model().forward_prefill(&[0, 1], 0, &mut state);
            // Write a real snapshot under the old anonymous CPU namespace.
            let mut old = KvPrefixCache::new(
                KvCacheConfig {
                    cache_dir: Some(dir.path().to_owned()),
                    ..KvCacheConfig::default()
                },
                source.model().config(),
                &format!("cpu:{}", compression.cache_tag()),
            );
            old.insert(&[0, 1], state.snapshot().unwrap());
            let before = files(dir.path());
            assert_eq!(before.len(), 1);
            assert!(before.values().all(|bytes| !bytes.is_empty()));
            old.clear_warm();
            assert!(old.find_longest_prefix(&[0, 1, 0]).is_some());

            let bytes = fixture::tiny_hybrid_with_seed(29);
            let model = loader(bytes.clone());
            let control = load(bytes);
            control.configure_cache(KvCacheConfig {
                max_warm_entries: 0,
                ..KvCacheConfig::default()
            });
            model.configure_cache(KvCacheConfig {
                cache_dir: Some(dir.path().to_owned()),
                ..KvCacheConfig::default()
            });
            let mut actual = session(&model, compression.clone());
            let mut expected = session(&control, compression.clone());
            for s in [&mut actual, &mut expected] {
                s.append_tokens(&[0, 1, 0]).unwrap();
            }
            same_live(&actual, &expected);
            let mut original = session(&source, compression.clone());
            original.append_tokens(&[0, 1, 0]).unwrap();
            distinct(
                original.last_logits().unwrap(),
                expected.last_logits().unwrap(),
            );
            assert_eq!(files(dir.path()), before);

            // Warm clearing must not expose the old cold entry to a fresh session.
            model.clear_warm_cache();
            let mut after_clear = session(&model, compression.clone());
            after_clear.append_tokens(&[0, 1, 0]).unwrap();
            same_live(&after_clear, &expected);
            model.clear_cache();
            assert_eq!(files(dir.path()), before);

            // Reconfiguration after the first compression tag must not enable
            // disk access, even when warm entries are disabled as well.
            let unused = dir.path().join("must-not-create");
            model.configure_cache(KvCacheConfig {
                cache_dir: Some(unused.clone()),
                max_warm_entries: 0,
                ..KvCacheConfig::default()
            });
            let mut reconfigured = session(&model, compression.clone());
            reconfigured.append_tokens(&[0, 1, 0]).unwrap();
            same_live(&reconfigured, &expected);
            model.clear_cache();
            assert!(!unused.exists());
            assert_eq!(files(dir.path()), before);
            drop(model);
            for s in [&mut actual, &mut expected] {
                s.append_tokens(&[1]).unwrap();
            }
            same_live(&actual, &expected);
        }
    }
}

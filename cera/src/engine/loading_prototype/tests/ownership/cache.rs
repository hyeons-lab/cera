use super::*;
use std::collections::BTreeMap;
use std::path::Path;

fn files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

#[test]
fn cache_controls_retain_live_state_and_forward_cold_tier_effects() {
    for compression in [KvCompression::None, KvCompression::F16] {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let bytes = fixture::tiny_hybrid();
        let model_dir = tempfile::tempdir().unwrap();
        let path = model_dir.path().join("model.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let shared = ModelLoader::new(ModelSource::path(&path))
            .config(cpu_config())
            .build_generative()
            .unwrap();
        let isolated = load(bytes);
        isolated.configure_cache(KvCacheConfig {
            max_warm_entries: 0,
            ..KvCacheConfig::default()
        });
        let configure = |dir: &Path| {
            shared.configure_cache(KvCacheConfig {
                cache_dir: Some(dir.to_owned()),
                ..KvCacheConfig::default()
            });
        };
        configure(first_dir.path());
        let mut live = session(&shared, compression.clone());
        let mut control = session(&isolated, compression.clone());
        for s in [&mut live, &mut control] {
            s.append_tokens(&[0, 1]).unwrap();
        }
        same_live(&live, &control);
        let initial = files(first_dir.path());
        assert_eq!(initial.len(), 1);
        assert!(
            initial
                .iter()
                .all(|(name, bytes)| { name.ends_with(".kvcache") && !bytes.is_empty() })
        );
        shared.clone().clear_warm_cache();
        assert_eq!(files(first_dir.path()), initial);

        // A new session can continue a cached strict prefix with numerical
        // parity against an independent model with prefix caching disabled.
        let mut extended = session(&shared, compression.clone());
        let mut extended_control = session(&isolated, compression.clone());
        for s in [&mut extended, &mut extended_control] {
            s.append_tokens(&[0, 1, 0]).unwrap();
        }
        same_live(&extended, &extended_control);
        let populated = files(first_dir.path());
        assert_eq!(populated.len(), 2);
        let mut adapted = session(&shared, compression.clone());
        adapted.attach_lora_adapters(fixture::adapter(32)).unwrap();
        adapted.append_tokens(&[1, 1, 0]).unwrap();
        live.hidden_states_for_tokens(&[1, 0, 1]).unwrap();
        assert_eq!(files(first_dir.path()), populated);

        let sentinel = "unrelated-namespace.kvcache";
        std::fs::write(first_dir.path().join(sentinel), b"preserve").unwrap();
        shared.clone().clear_cache();
        let sentinel_only = BTreeMap::from([(sentinel.to_owned(), b"preserve".to_vec())]);
        assert_eq!(files(first_dir.path()), sentinel_only);
        for (actual, expected) in [
            (&mut live, &mut control),
            (&mut extended, &mut extended_control),
        ] {
            actual.append_tokens(&[1]).unwrap();
            expected.append_tokens(&[1]).unwrap();
            same_live(actual, expected);
        }

        let mut refill = session(&shared, compression.clone());
        refill.append_tokens(&[1, 0]).unwrap();
        let before_reconfigure = files(first_dir.path());
        assert_eq!(before_reconfigure.len(), 2);
        configure(second_dir.path());
        let mut new_root = session(&shared, compression);
        new_root.append_tokens(&[1, 0, 1]).unwrap();
        assert_eq!(files(second_dir.path()).len(), 1);
        assert_eq!(files(first_dir.path()), before_reconfigure);
        shared.clear_cache();
        assert!(files(second_dir.path()).is_empty());
        assert_eq!(files(first_dir.path()), before_reconfigure);
        live.append_tokens(&[0]).unwrap();
        control.append_tokens(&[0]).unwrap();
        same_live(&live, &control);
    }
}

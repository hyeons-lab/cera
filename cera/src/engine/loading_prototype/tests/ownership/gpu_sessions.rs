//! Run explicitly with --ignored on a native GPU host; unavailable devices fail.
use super::*;

fn load_gpu(backend: BackendPreference) -> GenerativeModel {
    let bytes = fixture::tiny_hybrid();
    let backing = Arc::downgrade(&bytes);
    let loaded = ModelLoader::new(ModelSource::bytes(bytes))
        .config(LoadConfig {
            backend,
            context_size: 64,
            ..LoadConfig::default()
        })
        .build_generative()
        .unwrap();
    assert!(
        backing.upgrade().is_none(),
        "text-only GPU loads must release their owned GGUF staging allocation"
    );
    // Explicit backend preferences return errors instead of falling back to CPU.
    loaded.engine.configure_cache(KvCacheConfig {
        max_warm_entries: 8,
        ..KvCacheConfig::default()
    });
    loaded
}

fn exercise(backend: BackendPreference) {
    for compression in [KvCompression::None, KvCompression::turboquant(42)] {
        let shared = load_gpu(backend);
        let cloned = shared.clone();
        let config = SessionConfig {
            kv_compression: compression.clone(),
            seed: Some(42),
            ..SessionConfig::default()
        };
        let mut active = shared.create_session(config.clone()).unwrap();
        let control_model = load_gpu(backend);
        let mut control = control_model.create_session(config.clone()).unwrap();
        active.append_tokens(&[0, 1, 0]).unwrap();
        control.append_tokens(&[0, 1, 0]).unwrap();
        same_live(&active, &control);

        // Both the prototype's cloned handle and the public engine must reject.
        assert!(matches!(
            cloned.create_session(config.clone()),
            Err(CeraError::Busy)
        ));
        assert!(matches!(
            shared.engine.new_session(config.clone()),
            Err(CeraError::Busy)
        ));
        let model = shared.engine.model_arc();
        let tokenizer = shared.engine.tokenizer_arc();
        let capabilities = shared.engine.capabilities();
        assert!(matches!(
            Session::new(
                model.clone(),
                tokenizer.clone(),
                capabilities,
                config.clone()
            ),
            Err(CeraError::Busy)
        ));

        // Busy must win before an incompatible mode can reconfigure live state.
        let conflict = SessionConfig {
            kv_compression: KvCompression::turboquant(999),
            ..config.clone()
        };
        assert!(matches!(
            shared.engine.new_session(conflict.clone()),
            Err(CeraError::Busy)
        ));
        drop(cloned);
        drop(shared);
        active.append_tokens(&[1, 1]).unwrap();
        control.append_tokens(&[1, 1]).unwrap();
        same_live(&active, &control);
        assert_eq!(generate(&mut active).tokens, generate(&mut control).tokens);
        same_live(&active, &control);

        active.cancel();
        assert!(matches!(
            Session::new(
                model.clone(),
                tokenizer.clone(),
                capabilities,
                config.clone()
            ),
            Err(CeraError::Busy)
        ));
        active.reset().unwrap();
        assert!(matches!(
            Session::new(
                model.clone(),
                tokenizer.clone(),
                capabilities,
                config.clone()
            ),
            Err(CeraError::Busy)
        ));
        active.append_tokens(&[1]).unwrap();
        let pristine_model = load_gpu(backend);
        let mut pristine = pristine_model.create_session(config.clone()).unwrap();
        pristine.append_tokens(&[1]).unwrap();
        same_live(&active, &pristine);
        drop(pristine);
        drop(pristine_model);
        drop(active);

        // The lifetime lease releases, while per-model compression stays fixed.
        assert!(matches!(
            Session::new(model.clone(), tokenizer.clone(), capabilities, conflict),
            Err(CeraError::KvCompressionConflict { .. })
        ));
        let mut successor =
            Session::new(model.clone(), tokenizer, capabilities, config.clone()).unwrap();
        successor.append_tokens(&[1, 0, 0]).unwrap();
        let fresh = load_gpu(backend);
        let mut expected = fresh.create_session(config).unwrap();
        expected.append_tokens(&[1, 0, 0]).unwrap();
        same_live(&successor, &expected);
        assert_eq!(
            generate(&mut successor).tokens,
            generate(&mut expected).tokens
        );
        same_live(&successor, &expected);
        println!(
            "{backend:?} {compression:?}: Busy, live continuation, reset, mode-error cleanup and successor passed"
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a Metal device; no CPU fallback or successful skip"]
fn metal_sessions_keep_exclusive_live_state() {
    exercise(BackendPreference::Metal);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a wgpu device; no CPU fallback or successful skip"]
fn wgpu_sessions_keep_exclusive_live_state() {
    exercise(BackendPreference::Gpu);
}

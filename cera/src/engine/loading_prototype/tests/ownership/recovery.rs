#![allow(deprecated)]

use super::*;
use crate::session::RecoveryOutcome;
use crate::tokenizer::UserMessage;

fn config(compression: KvCompression, chunk: u32, max: u32) -> SessionConfig {
    SessionConfig {
        kv_compression: compression,
        seed: Some(42),
        ubatch_size: chunk,
        max_seq_len: Some(max),
        ..Default::default()
    }
}

fn message(text: &str) -> UserMessage {
    UserMessage {
        text: Some(text.into()),
        ..Default::default()
    }
}

#[test]
fn recovery_real_causal_models_restore_and_continue_after_cancelled_message() {
    for bytes in [
        fixture::tiny_dense(),
        fixture::tiny_hybrid(),
        fixture::tiny_attention_lfm2(true),
    ] {
        for compression in [KvCompression::None, KvCompression::F16] {
            let loaded = load(bytes.clone());
            let config = config(compression, 1, 16);
            let mut actual = loaded.create_session(config.clone()).unwrap();
            let mut reference = load(bytes.clone()).create_session(config).unwrap();
            for s in [&mut actual, &mut reference] {
                s.append_tokens(&[0, 1]).unwrap();
            }
            actual.cancel();
            assert!(matches!(
                actual.append_user_message(&message("baba")),
                Err(CeraError::Cancelled)
            ));
            assert_eq!(
                actual.last_ingest_recovery().unwrap().outcome,
                RecoveryOutcome::Restored
            );
            same_live(&actual, &reference);
            assert!(
                actual
                    .cancel_handle()
                    .load(std::sync::atomic::Ordering::Relaxed)
            );
            actual.clear_cancel();
            for s in [&mut actual, &mut reference] {
                s.append_user_message(&message("a")).unwrap();
            }
            assert!(actual.last_ingest_recovery().is_none());
            same_live(&actual, &reference);
            assert_eq!(
                generate(&mut actual).tokens,
                generate(&mut reference).tokens
            );
            same_live(&actual, &reference);
        }
    }
}

#[test]
fn recovery_compressed_rejection_preserves_context_and_cancelled_append_resets() {
    for (keys, values) in [(true, false), (false, true), (true, true)] {
        let bytes = fixture::tiny_hybrid();
        let loaded = load(bytes.clone());
        let config = config(
            KvCompression::TurboQuant {
                seed: 7,
                keys,
                values,
            },
            1,
            8,
        );
        let mut actual = loaded.create_session(config.clone()).unwrap();
        let mut reference = load(bytes).create_session(config).unwrap();
        let adapter = fixture::adapter(32);
        for s in [&mut actual, &mut reference] {
            s.attach_lora_adapters(adapter.clone()).unwrap();
            s.append_tokens(&[0, 1]).unwrap();
        }
        // This reaches the old unconditional rollback arm despite no forward.
        // That arm asserted even for a no-op on a compressed cache.
        assert!(matches!(
            actual.append_user_message(&message("abababab")),
            Err(CeraError::ContextOverflow { .. })
        ));
        assert_eq!(
            actual.last_ingest_recovery().unwrap().outcome,
            RecoveryOutcome::Unchanged
        );
        same_live(&actual, &reference);
        for s in [&mut actual, &mut reference] {
            s.append_tokens(&[1]).unwrap();
        }
        same_live(&actual, &reference);
        actual.cancel();
        assert!(matches!(
            actual.append_user_message(&message("baba")),
            Err(CeraError::Cancelled)
        ));
        assert_eq!(
            actual.last_ingest_recovery().unwrap().outcome,
            RecoveryOutcome::Reset
        );
        assert!(actual.is_usable() && actual.has_lora_adapters());
        assert_eq!(actual.position(), 0);
        assert!(actual.last_logits().is_none());
        assert!(
            actual
                .cancel_handle()
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        actual.clear_cancel();
        reference.reset().unwrap();
        for s in [&mut actual, &mut reference] {
            s.append_tokens(&[1, 0]).unwrap();
        }
        same_live(&actual, &reference);
        assert_eq!(
            generate(&mut actual).tokens,
            generate(&mut reference).tokens
        );
        same_live(&actual, &reference);
    }
}

#[test]
fn recovery_expired_convolution_checkpoint_resets_before_reuse() {
    let load = |bytes| {
        ModelLoader::new(ModelSource::bytes(bytes))
            .config(EngineConfig {
                context_size: 128,
                ..cpu_config()
            })
            .build_generative()
            .unwrap()
    };
    for compression in [KvCompression::None, KvCompression::F16] {
        let bytes = fixture::tiny_hybrid_with_context(128);
        let loaded = load(bytes.clone());
        let config = config(compression, 64, 128);
        let mut actual = loaded.create_session(config.clone()).unwrap();
        let mut reference = load(bytes).create_session(config).unwrap();
        actual.append_tokens(&[0, 1]).unwrap();
        actual.cancel();
        let input = "ab".repeat(33);
        let tokens = actual.tokenizer().encode(&input);
        assert_eq!(tokens.len(), 66);
        let result = actual.append_user_message(&message(&input));
        assert!(
            matches!(result, Err(CeraError::Cancelled)),
            "{result:?}, position {}",
            actual.position()
        );
        let recovery = actual.last_ingest_recovery().unwrap();
        assert_eq!(recovery.outcome, RecoveryOutcome::Reset);
        assert!(matches!(
            recovery.rewind_error,
            Some(crate::kv_cache::KvRewindError::MissingConvolutionCheckpoint { position: 2, .. })
        ));
        assert_eq!(actual.position(), 0);
        actual.clear_cancel();
        for s in [&mut actual, &mut reference] {
            s.append_tokens(&[1, 0]).unwrap();
        }
        same_live(&actual, &reference);
        assert_eq!(
            generate(&mut actual).tokens,
            generate(&mut reference).tokens
        );
        same_live(&actual, &reference);
    }
}

#[test]
fn recovery_noncausal_models_reset_instead_of_claiming_restoration() {
    let bytes = fixture::tiny_attention_lfm2(false);
    let loaded = load(bytes.clone());
    let config = config(KvCompression::None, 1, 16);
    let mut actual = loaded.create_session(config.clone()).unwrap();
    let mut reference = load(bytes).create_session(config).unwrap();
    actual.append_tokens(&[0, 1]).unwrap();
    actual.cancel();
    assert!(matches!(
        actual.append_user_message(&message("abab")),
        Err(CeraError::Cancelled)
    ));
    let recovery = actual.last_ingest_recovery().unwrap();
    assert_eq!(recovery.outcome, RecoveryOutcome::Reset);
    assert_eq!(
        recovery.rewind_error,
        Some(crate::kv_cache::KvRewindError::NonCausal)
    );
    actual.clear_cancel();
    for s in [&mut actual, &mut reference] {
        s.append_tokens(&[1, 0]).unwrap();
    }
    same_live(&actual, &reference);
}

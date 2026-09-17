use super::*;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use std::sync::Arc;

struct RefusingModel(ModelConfig);

impl Model for RefusingModel {
    fn config(&self) -> &ModelConfig {
        &self.0
    }

    fn forward(&self, tokens: &[u32], _: usize, state: &mut InferenceState) -> Vec<f32> {
        state.seq_len += tokens.len();
        vec![1.0, 0.0]
    }

    fn try_reset_kv(
        &self,
        _: &mut InferenceState,
        _: &KvCompression,
        _: usize,
    ) -> Result<(), cera::CeraError> {
        Err(cera::CeraError::OutOfMemory {
            requested_bytes: (1 << 40) + 7,
        })
    }
}

fn session() -> Arc<Session> {
    let config = ModelConfig {
        architecture: "ffi-recovery-test".into(),
        n_layers: 1,
        hidden_size: 2,
        intermediate_size: 2,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 2,
        vocab_size: 2,
        max_seq_len: 16,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types: vec![BlockType::Attention],
        conv_kernel_size: None,
        ssm: None,
        kv_heads_per_layer: vec![1],
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    };
    let inner = cera::Session::new(
        Arc::new(RefusingModel(config)),
        Arc::new(cera::tokenizer::BpeTokenizer::from_vocab(vec![
            b"a".to_vec(),
            b"b".to_vec(),
        ])),
        cera::ModalityCapabilities::text_only(),
        cera::SessionConfig {
            ubatch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    Session::from_core(inner)
}

#[test]
fn status_preserves_original_error_and_typed_secondary_failure() {
    let session = session();
    let initial = session.recovery_status().unwrap();
    assert!(initial.usable);
    assert_eq!(initial.position, 0);
    assert!(initial.last_ingest_recovery.is_none());
    session.append_tokens(vec![0, 1]).unwrap();
    session.cancel();
    assert!(matches!(
        session.send_message(crate::UserMessage {
            text: Some("baba".into()),
            ..Default::default()
        }),
        Err(FfiError::Cancelled)
    ));
    for _ in 0..2 {
        let status = session.recovery_status().unwrap();
        assert!(!status.usable);
        assert_eq!(status.position, 3);
        let report = status.last_ingest_recovery.unwrap();
        assert_eq!(report.outcome, RecoveryOutcome::Unusable);
        assert_eq!(
            report.rewind_error,
            Some(KvRewindFailure::BackendUnsupported)
        );
        assert!(matches!(
            report.reset_error,
            Some(FfiError::OutOfMemory { requested_bytes }) if requested_bytes == (1 << 40) + 7
        ));
        assert!(session.cancel.load(std::sync::atomic::Ordering::Relaxed));
    }
    session.clear_cancel();
    assert!(!session.recovery_status().unwrap().usable);
    assert!(matches!(
        session.append_tokens(vec![0]),
        Err(FfiError::Backend { .. })
    ));
    assert!(matches!(session.reset(), Err(FfiError::OutOfMemory { .. })));
    assert!(!session.recovery_status().unwrap().usable);
}

#[test]
fn diagnostic_observation_never_blocks_on_session_lock() {
    let session = session();
    let guard = session.inner.lock().unwrap();
    // Same-thread reentrancy models the enclosing lock in a streaming callback.
    assert!(matches!(session.recovery_status(), Err(FfiError::Busy)));
    let other = session.clone();
    assert!(
        std::thread::spawn(move || matches!(other.recovery_status(), Err(FfiError::Busy)))
            .join()
            .unwrap()
    );
    drop(guard);
    assert!(session.recovery_status().unwrap().usable);
}

#[test]
fn poisoned_status_requires_recreation_without_returning_stale_snapshot() {
    let session = session();
    let other = session.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = other.inner.lock().unwrap();
            panic!("injected enclosing-operation panic");
        })
        .join()
        .is_err()
    );
    assert!(matches!(
        session.recovery_status(),
        Err(FfiError::Backend { detail }) if detail.contains("poisoned")
    ));
}

#[test]
fn rewind_diagnostics_keep_wide_positions_and_layout_detail() {
    use cera::kv_cache::KvRewindError as E;
    let cases = [
        (
            E::OutOfBounds {
                requested: usize::MAX,
                current: 3,
            },
            KvRewindFailure::OutOfBounds {
                requested: usize::MAX as u64,
                current: 3,
            },
        ),
        (E::Compressed, KvRewindFailure::Compressed),
        (E::NonCausal, KvRewindFailure::NonCausal),
        (
            E::MissingConvolutionCheckpoint {
                layer: 2,
                position: usize::MAX,
            },
            KvRewindFailure::MissingConvolutionCheckpoint {
                layer: 2,
                position: usize::MAX as u64,
            },
        ),
        (
            E::InvalidCacheLayout {
                layer: 5,
                detail: "unequal rows",
            },
            KvRewindFailure::InvalidCacheLayout {
                layer: 5,
                detail: "unequal rows".into(),
            },
        ),
        (E::BackendUnsupported, KvRewindFailure::BackendUnsupported),
    ];
    for (core, expected) in cases {
        assert_eq!(KvRewindFailure::from(&core), expected);
    }
}

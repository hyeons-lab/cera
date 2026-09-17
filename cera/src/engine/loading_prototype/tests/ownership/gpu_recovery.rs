//! Actual device proof; ignored unless explicitly selected on a GPU host.
#![allow(deprecated)]
use super::*;
use crate::kv_cache::{InferenceState, LayerSnapshot, StateSnapshot};
use crate::session::RecoveryOutcome;
use crate::tokenizer::UserMessage;
use std::sync::atomic::Ordering;

fn load_device(bytes: Arc<[u8]>, backend: BackendPreference) -> GenerativeModel {
    ModelLoader::new(ModelSource::bytes(bytes))
        .config(EngineConfig {
            backend,
            context_size: 64,
            ..Default::default()
        })
        .build_generative()
        .unwrap()
}

fn assert_empty_layer(layer: &LayerSnapshot) {
    match layer {
        LayerSnapshot::Attention { k_data, v_data }
        | LayerSnapshot::AttentionF16 { k_data, v_data } => {
            assert!(k_data.is_empty() && v_data.is_empty());
        }
        LayerSnapshot::AttentionCompressed { keys, values } => {
            assert_eq!(
                crate::turboquant::decode_compressed_keys(keys)
                    .unwrap()
                    .seq_len(),
                0
            );
            assert_eq!(
                crate::turboquant::decode_compressed_values(values)
                    .unwrap()
                    .seq_len(),
                0
            );
        }
        LayerSnapshot::Conv { buffer } => {
            assert!(!buffer.is_empty());
            assert!(
                buffer.iter().all(|&byte| byte == 0),
                "device convolution was not cleared"
            );
        }
        LayerSnapshot::Mamba2 {
            conv_state,
            ssm_state,
        }
        | LayerSnapshot::DeltaNet {
            conv_state,
            ssm_state,
        } => {
            assert!(conv_state.iter().all(|&byte| byte == 0));
            assert!(ssm_state.iter().all(|&byte| byte == 0));
        }
        LayerSnapshot::ParallelAttentionMamba2 {
            snap,
            conv_state,
            ssm_state,
        } => {
            assert_empty_layer(snap);
            assert!(conv_state.iter().all(|&byte| byte == 0));
            assert!(ssm_state.iter().all(|&byte| byte == 0));
        }
    }
}

fn assert_empty(snapshot: &StateSnapshot) {
    assert_eq!(snapshot.seq_len, 0);
    for layer in &snapshot.layers {
        assert_empty_layer(layer);
    }
}

fn exercise(backend: BackendPreference) {
    for (architecture, bytes) in [
        ("hybrid", fixture::tiny_hybrid()),
        ("dense", fixture::tiny_dense()),
    ] {
        for compression in [
            KvCompression::None,
            KvCompression::F16,
            KvCompression::turboquant(7),
        ] {
            let loaded = load_device(bytes.clone(), backend);
            let control = load_device(bytes.clone(), backend);
            let model = loaded.model();
            let reference = control.model();
            let config = SessionConfig {
                kv_compression: compression.clone(),
                ubatch_size: 1,
                seed: Some(42),
                ..Default::default()
            };
            model.configure_kv_compression(&compression).unwrap();
            reference.configure_kv_compression(&compression).unwrap();
            let mut state =
                InferenceState::from_config_with_compression(model.config(), &compression).unwrap();
            let mut expected =
                InferenceState::from_config_with_compression(reference.config(), &compression)
                    .unwrap();
            let adapter = matches!(compression, KvCompression::TurboQuant { .. })
                .then(|| fixture::adapter(32));
            state.lora = adapter.clone();
            expected.lora = adapter.clone();
            model.forward_prefill(&[0, 1, 0], 0, &mut state);
            let before = model.snapshot_state();
            assert_eq!(before.seq_len, 3);
            assert_eq!(before.is_compressed(), adapter.is_some());
            if architecture == "hybrid" {
                assert!(before.layers.iter().any(|layer| matches!(layer, LayerSnapshot::Conv { buffer } if buffer.iter().any(|&byte| byte != 0))));
            }
            // A rejected mode change must not clear any live state.
            let conflict = if adapter.is_some() {
                KvCompression::None
            } else {
                KvCompression::turboquant(99)
            };
            assert!(matches!(
                model.try_reset_kv(&mut state, &conflict, 64),
                Err(CeraError::KvCompressionConflict { .. })
            ));
            assert_eq!(state.seq_len, 3);
            assert_eq!(
                format!("{:?}", model.snapshot_state()),
                format!("{before:?}")
            );

            model.try_reset_kv(&mut state, &compression, 64).unwrap();
            assert_eq!(state.seq_len, 0);
            if let Some(adapter) = &adapter {
                assert!(Arc::ptr_eq(state.lora.as_ref().unwrap(), adapter));
            }
            // This is deliberately BEFORE another prefill can repair device KV.
            assert_empty(&model.snapshot_state());
            let input = vec![0.1; model.config().hidden_size];
            close(
                &model.forward_from_embedding(&input, 0, &mut state),
                &reference.forward_from_embedding(&input, 0, &mut expected),
            );
            close(
                &model.forward(&[1], 1, &mut state),
                &reference.forward(&[1], 1, &mut expected),
            );

            // Exercise the public Session automatic fallback and ownership.
            let mut active = loaded.create_session(config.clone()).unwrap();
            let mut fresh = control.create_session(config.clone()).unwrap();
            for s in [&mut active, &mut fresh] {
                if let Some(adapter) = &adapter {
                    s.attach_lora_adapters(adapter.clone()).unwrap();
                }
            }
            active.append_tokens(&[0, 1]).unwrap();
            let position = active.position_handle();
            let cancel = active.cancel_handle();
            active.cancel();
            let message = UserMessage {
                text: Some("baba".into()),
                ..Default::default()
            };
            assert!(matches!(
                active.append_user_message(&message),
                Err(CeraError::Cancelled)
            ));
            let recovery = active.last_ingest_recovery().unwrap();
            assert_eq!(recovery.outcome, RecoveryOutcome::Reset);
            assert!(recovery.reset_error.is_none());
            assert!(active.is_usable());
            assert_eq!(position.load(Ordering::Relaxed), 0);
            assert!(cancel.load(Ordering::Relaxed));
            assert!(Arc::ptr_eq(&position, &active.position_handle()));
            assert!(Arc::ptr_eq(&cancel, &active.cancel_handle()));
            assert_empty(&model.snapshot_state());
            assert!(matches!(
                loaded.create_session(config.clone()),
                Err(CeraError::Busy)
            ));
            active.clear_cancel();
            for s in [&mut active, &mut fresh] {
                s.append_tokens(&[1, 0]).unwrap();
                s.append_user_message(&message).unwrap();
                s.append_embeddings(&input, 1).unwrap();
            }
            same_live(&active, &fresh);
            assert_eq!(generate(&mut active).tokens, generate(&mut fresh).tokens);
            same_live(&active, &fresh);
            drop(active);
            assert!(loaded.create_session(config).is_ok());
            println!(
                "{backend:?} {architecture} {compression:?}: immediate reset, embeddings, cancellation, replay, adapter and ownership passed"
            );
        }
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a Metal device; unavailable device is a failure"]
fn metal_checked_device_recovery() {
    exercise(BackendPreference::Metal);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a native wgpu device; unavailable device is a failure"]
fn wgpu_checked_device_recovery() {
    exercise(BackendPreference::Gpu);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a native wgpu device; deliberately destroys an isolated device"]
fn wgpu_destroyed_device_cannot_report_checked_reset() {
    use crate::model::Model;
    let ctx = crate::backend::wgpu::GpuContext::new().unwrap();
    let device = ctx.device.clone();
    let gguf = crate::gguf::GgufFile::from_bytes(fixture::tiny_hybrid()).unwrap();
    let model =
        crate::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_ctx(gguf, 64, String::new(), ctx)
            .unwrap();
    model
        .configure_kv_compression(&KvCompression::None)
        .unwrap();
    let mut state = InferenceState::from_config(model.config()).unwrap();
    model.forward_prefill(&[0, 1], 0, &mut state);
    assert_eq!(state.seq_len, 2);
    device.destroy();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.try_reset_kv(&mut state, &KvCompression::None, 64)
    }));
    assert!(
        matches!(&result, Ok(Err(CeraError::Backend(error))) if error.contains("checked wgpu reset: GPU readback failed")),
        "destroyed device must return a checked reset error: {result:?}"
    );
    assert_eq!(
        state.seq_len, 2,
        "caller state committed before device completion"
    );
    match result {
        Ok(Err(error)) => println!("destroyed device: checked reset error: {error}"),
        Err(_) => unreachable!(),
        Ok(Ok(())) => unreachable!(),
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires native wgpu; destroys an isolated device after a prefill chunk"]
fn wgpu_device_loss_preserves_primary_and_secondary_session_errors() {
    use crate::model::{Model, ModelConfig, ModelSessionLease};
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    struct LoseDevice {
        model: crate::model::gpu_lfm2::GpuLfm2Model,
        device: wgpu::Device,
        armed: AtomicBool,
        forwards: AtomicUsize,
    }
    impl Model for LoseDevice {
        fn config(&self) -> &ModelConfig {
            self.model.config()
        }
        fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
            self.model.acquire_session()
        }
        fn configure_kv_compression(&self, compression: &KvCompression) -> Result<(), CeraError> {
            self.model.configure_kv_compression(compression)
        }
        fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
            self.forwards.fetch_add(1, Ordering::Relaxed);
            let logits = self.model.forward(tokens, pos, state);
            if self.armed.swap(false, Ordering::Relaxed) {
                self.device.destroy();
            }
            logits
        }
        fn try_reset_kv(
            &self,
            state: &mut InferenceState,
            compression: &KvCompression,
            max: usize,
        ) -> Result<(), CeraError> {
            self.model.try_reset_kv(state, compression, max)
        }
    }
    let ctx = crate::backend::wgpu::GpuContext::new().unwrap();
    let device = ctx.device.clone();
    let gguf = crate::gguf::GgufFile::from_bytes(fixture::tiny_hybrid()).unwrap();
    let tokenizer = Arc::new(crate::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap());
    let model = Arc::new(LoseDevice {
        model: crate::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_ctx(
            gguf,
            64,
            String::new(),
            ctx,
        )
        .unwrap(),
        device,
        armed: AtomicBool::new(false),
        forwards: AtomicUsize::new(0),
    });
    let mut session = Session::new(
        model.clone(),
        tokenizer,
        crate::session::ModalityCapabilities::text_only(),
        SessionConfig {
            ubatch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    session.append_tokens(&[0, 1]).unwrap();
    let cancel = session.cancel_handle();
    let position = session.position_handle();
    model.armed.store(true, Ordering::Relaxed);
    session.cancel();
    let message = UserMessage {
        text: Some("baba".into()),
        ..Default::default()
    };
    assert!(matches!(
        session.append_user_message(&message),
        Err(CeraError::Cancelled)
    ));
    let recovery = session.last_ingest_recovery().unwrap();
    assert_eq!(recovery.outcome, RecoveryOutcome::Unusable);
    assert!(
        matches!(&recovery.reset_error, Some(CeraError::Backend(error)) if error.contains("checked wgpu reset: GPU readback failed"))
    );
    assert!(cancel.load(Ordering::Relaxed));
    assert_eq!(position.load(Ordering::Relaxed), 3);
    assert!(Arc::ptr_eq(&cancel, &session.cancel_handle()));
    assert!(Arc::ptr_eq(&position, &session.position_handle()));
    assert!(!session.is_usable());
    assert!(session.append_tokens(&[1]).is_err());
    assert_eq!(model.forwards.load(Ordering::Relaxed), 3);
    assert!(session.reset().is_err());
    assert!(!session.is_usable());
    assert!(cancel.load(Ordering::Relaxed));
    assert_eq!(position.load(Ordering::Relaxed), 3);
    println!(
        "device loss: primary Cancelled, secondary readback error, unusable state, handles and cancellation preserved"
    );
}

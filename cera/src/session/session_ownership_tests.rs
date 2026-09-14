use super::*;
use crate::model::{ModelConfig, ModelSessionGate, ScalarMultipliers};
use std::sync::atomic::{AtomicU8, AtomicUsize};

struct SharedModel(ModelConfig);

impl Model for SharedModel {
    fn config(&self) -> &ModelConfig {
        &self.0
    }
    fn forward(&self, tokens: &[u32], _: usize, state: &mut InferenceState) -> Vec<f32> {
        state.seq_len += tokens.len();
        vec![0.0, 1.0]
    }
}

fn config() -> ModelConfig {
    ModelConfig {
        architecture: "ownership-test".into(),
        n_layers: 0,
        hidden_size: 2,
        intermediate_size: 2,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 2,
        vocab_size: 2,
        max_seq_len: 16,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types: Vec::new(),
        conv_kernel_size: None,
        kv_heads_per_layer: Vec::new(),
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}

struct ReservedModel {
    inner: SharedModel,
    gate: ModelSessionGate,
    configured: AtomicUsize,
    failure: AtomicU8,
}

impl ReservedModel {
    fn new() -> Self {
        Self {
            inner: SharedModel(config()),
            gate: ModelSessionGate::default(),
            configured: AtomicUsize::new(0),
            failure: AtomicU8::new(0),
        }
    }
}

impl Model for ReservedModel {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.inner.forward(tokens, pos, state)
    }
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.gate.try_acquire().map(Some)
    }
    fn configure_kv_compression(&self, _: &KvCompression) -> Result<(), CeraError> {
        self.configured.fetch_add(1, Ordering::Relaxed);
        match self.failure.load(Ordering::Relaxed) {
            1 => Err(CeraError::Backend("configuration rejected".into())),
            2 => panic!("configuration panic"),
            _ => Ok(()),
        }
    }
}

fn create(model: Arc<dyn Model>) -> Result<Session, CeraError> {
    Session::new(
        model,
        Arc::new(BpeTokenizer::empty_for_test()),
        ModalityCapabilities::text_only(),
        SessionConfig::default(),
    )
}

#[test]
fn busy_precedes_configuration_and_reset_retains_ownership() {
    let model = Arc::new(ReservedModel::new());
    let mut active = create(model.clone()).unwrap();
    active.append_tokens(&[0, 1]).unwrap();
    model.failure.store(1, Ordering::Relaxed);
    assert!(matches!(create(model.clone()), Err(CeraError::Busy)));
    assert_eq!(model.configured.load(Ordering::Relaxed), 1);
    assert_eq!(active.position(), 2);
    assert!(active.reset().is_err());
    assert!(matches!(create(model.clone()), Err(CeraError::Busy)));
    model.failure.store(0, Ordering::Relaxed);
    active.reset().unwrap();
    active.cancel();
    assert!(matches!(create(model.clone()), Err(CeraError::Busy)));
    active.clear_cancel();
    active.append_tokens(&[1]).unwrap();
    let observed_position = active.position_handle();
    let cancel = active.cancel_handle();
    drop(active);
    let successor = create(model).unwrap();
    assert_eq!(successor.position(), 0);
    // Observation/cancellation handles do not own the inference context.
    assert_eq!(observed_position.load(Ordering::Relaxed), 1);
    assert!(!cancel.load(Ordering::Relaxed));
}

#[test]
fn failed_and_panicking_construction_release_ownership() {
    for failure in [1, 2] {
        let model = Arc::new(ReservedModel::new());
        model.failure.store(failure, Ordering::Relaxed);
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| create(model.clone())));
        if failure == 1 {
            assert!(matches!(outcome, Ok(Err(CeraError::Backend(_)))));
        } else {
            assert!(outcome.is_err());
        }
        model.failure.store(0, Ordering::Relaxed);
        assert!(create(model).is_ok());
    }
    let mut invalid = ReservedModel::new();
    invalid.inner.0.n_heads = usize::MAX;
    let model = Arc::new(invalid);
    assert!(matches!(
        create(model.clone()),
        Err(CeraError::OutOfMemory { .. })
    ));
    assert!(model.gate.try_acquire().is_ok());
}

#[test]
fn default_model_hook_preserves_multiple_live_sessions() {
    let model = Arc::new(SharedModel(config()));
    let mut a = create(model.clone()).unwrap();
    let mut b = create(model).unwrap();
    a.append_tokens(&[0, 1]).unwrap();
    b.append_tokens(&[1]).unwrap();
    a.reset().unwrap();
    assert_eq!(b.position(), 1);
    b.append_tokens(&[0]).unwrap();
    assert_eq!(b.position(), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn racing_session_constructors_have_one_owner_and_release_across_threads() {
    let model = Arc::new(ReservedModel::new());
    let start = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let model = model.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                create(model)
            })
        })
        .collect();
    // Returned Sessions stay alive in their JoinHandles until collected here.
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, Err(CeraError::Busy)))
            .count(),
        7
    );
    assert_eq!(model.configured.load(Ordering::Relaxed), 1);
    drop(results);
    assert!(create(model).is_ok());
}

struct DropProbe {
    model: Arc<ReservedModel>,
    saw_busy: Arc<AtomicBool>,
}

impl crate::spec::Drafter for DropProbe {
    fn clone_drafter(&self) -> Box<dyn crate::spec::Drafter> {
        unreachable!()
    }
    fn reset(&mut self) {}
    fn draft(&mut self, _: &[u32], _: usize) -> Vec<u32> {
        Vec::new()
    }
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.saw_busy.store(
            matches!(self.model.gate.try_acquire(), Err(CeraError::Busy)),
            Ordering::Relaxed,
        );
    }
}

#[test]
fn session_drops_owned_resources_before_releasing_the_lease() {
    let model = Arc::new(ReservedModel::new());
    let mut active = create(model.clone()).unwrap();
    let saw_busy = Arc::new(AtomicBool::new(false));
    active.drafter = Some(Box::new(DropProbe {
        model: model.clone(),
        saw_busy: saw_busy.clone(),
    }));
    drop(active);
    assert!(saw_busy.load(Ordering::Relaxed));
    assert!(create(model).is_ok());
}

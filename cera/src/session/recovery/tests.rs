use super::*;
use crate::kv_cache::LayerState;
use crate::model::{ModelConfig, ScalarMultipliers};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicUsize};

struct FaultModel {
    config: ModelConfig,
    rewind: AtomicBool,
    reset: AtomicBool,
    // 1: mutate but report zero consumed; 2: panic after forward;
    // 3: panic during rewind; 4: fail reset; 5: panic during reset.
    fault: AtomicU8,
    rewinds: AtomicUsize,
    resets: AtomicUsize,
    reset_hook: Mutex<Option<Box<dyn Fn() + Send>>>,
}

impl FaultModel {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            config: ModelConfig {
                architecture: "recovery-test".into(),
                n_layers: 1,
                hidden_size: 2,
                intermediate_size: 2,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                vocab_size: 2,
                max_seq_len: 256,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
                block_types: vec![crate::model::BlockType::Attention],
                conv_kernel_size: None,
                kv_heads_per_layer: vec![1],
                scalars: ScalarMultipliers::default(),
                moe: None,
                is_causal: true,
                class_labels: Vec::new(),
            },
            rewind: AtomicBool::new(true),
            reset: AtomicBool::new(true),
            fault: AtomicU8::new(0),
            rewinds: AtomicUsize::new(0),
            resets: AtomicUsize::new(0),
            reset_hook: Mutex::new(None),
        })
    }
}

impl Model for FaultModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(pos, state.seq_len);
        for layer in &mut state.layers {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = layer
            {
                for token in tokens {
                    key_cache.extend([*token as f32, pos as f32]);
                    value_cache.extend([*token as f32, 1.0]);
                }
            }
        }
        state.seq_len += tokens.len();
        assert_ne!(self.fault.load(Ordering::Relaxed), 2, "forward panic");
        vec![pos as f32, tokens[0] as f32]
    }
    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
        ubatch: usize,
        cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        let chunk = if ubatch == 0 { tokens.len() } else { ubatch };
        let mut consumed = 0;
        let mut logits = None;
        for tokens in tokens.chunks(chunk) {
            logits = Some(self.forward(tokens, pos + consumed, state));
            if self.fault.load(Ordering::Relaxed) == 1 {
                return (0, logits);
            }
            consumed += tokens.len();
            if cancel.load(Ordering::Relaxed) {
                break;
            }
        }
        (consumed, logits)
    }
    fn supports_embedding_input(&self) -> bool {
        true
    }
    fn forward_from_embedding(
        &self,
        _: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.forward(&[1], pos, state)
    }
    fn supports_kv_shift(&self) -> bool {
        true
    }
    fn shift_kv(&self, state: &mut InferenceState, keep: usize, shift: usize) {
        for layer in &mut state.layers {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = layer
            {
                key_cache.drain(keep * 2..(keep + shift) * 2);
                value_cache.drain(keep * 2..(keep + shift) * 2);
            }
        }
        state.seq_len -= shift;
    }
    fn truncate_kv(&self, _: &mut InferenceState, _: usize) {
        panic!("legacy rewind must never be called by recovery");
    }
    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        if !self.rewind.load(Ordering::Relaxed) {
            return Err(KvRewindError::BackendUnsupported);
        }
        state.check_truncate_to(len)
    }
    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        self.rewinds.fetch_add(1, Ordering::Relaxed);
        assert_ne!(self.fault.load(Ordering::Relaxed), 3, "rewind panic");
        self.check_kv_rewind(state, len)?;
        state.try_truncate_to(len)
    }
    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max: usize,
    ) -> Result<(), CeraError> {
        self.resets.fetch_add(1, Ordering::Relaxed);
        if let Some(hook) = self.reset_hook.lock().unwrap().as_ref() {
            hook();
        }
        assert_ne!(self.fault.load(Ordering::Relaxed), 5, "reset panic");
        if !self.reset.load(Ordering::Relaxed) || self.fault.load(Ordering::Relaxed) == 4 {
            return Err(CeraError::Backend("injected reset failure".into()));
        }
        crate::model::reset_cpu_kv(self, state, compression, max)
    }
}

fn session(model: Arc<dyn Model>, max: u32, keep: u32) -> Session {
    Session::new(
        model,
        Arc::new(BpeTokenizer::empty_for_test()),
        ModalityCapabilities::text_only(),
        SessionConfig {
            max_seq_len: Some(max),
            n_keep: keep,
            ubatch_size: 1,
            seed: Some(17),
            ..Default::default()
        },
    )
    .unwrap()
}

fn snapshot(session: &Session) -> String {
    format!(
        "{:?}|{}|{}|{:?}|{:?}|{}|{:?}",
        session.state.snapshot(),
        session.current_pos,
        session.position(),
        session.token_history,
        session.last_logits,
        session.prefill_tokens,
        session.prefill_elapsed
    )
}

fn outcome(session: &Session) -> RecoveryOutcome {
    session.last_ingest_recovery().unwrap().outcome
}

#[test]
fn validation_failure_does_not_touch_execution_or_invoke_recovery() {
    let model = FaultModel::new();
    let mut active = session(model.clone(), 16, 0);
    active.append_tokens(&[0, 1]).unwrap();
    active.cancel();
    let before = snapshot(&active);
    for message in [
        crate::tokenizer::UserMessage::default(),
        crate::tokenizer::UserMessage {
            images: vec![vec![1]],
            ..Default::default()
        },
    ] {
        assert!(active.append_user_message(&message).is_err());
        assert_eq!(outcome(&active), RecoveryOutcome::Unchanged);
        assert_eq!(snapshot(&active), before);
        assert!(active.is_usable());
        assert!(active.cancel.load(Ordering::Relaxed));
    }
    assert_eq!(model.rewinds.load(Ordering::Relaxed), 0);
    assert_eq!(model.resets.load(Ordering::Relaxed), 0);
}

#[derive(Clone)]
struct DraftProbe {
    resets: Arc<AtomicUsize>,
    panic: bool,
}
impl crate::spec::Drafter for DraftProbe {
    fn clone_drafter(&self) -> Box<dyn crate::spec::Drafter> {
        Box::new(self.clone())
    }
    fn draft(&mut self, _: &[u32], _: usize) -> Vec<u32> {
        vec![0]
    }
    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::Relaxed);
        assert!(!self.panic, "drafter panic");
    }
}

#[test]
fn late_mixed_input_failure_restores_metadata_rng_drafter_and_cache() {
    let model = FaultModel::new();
    let mut active = session(model.clone(), 16, 0);
    let mut reference = session(model.clone(), 16, 0);
    for s in [&mut active, &mut reference] {
        s.append_tokens(&[0, 1]).unwrap();
        for _ in 0..7 {
            s.sampler.sample(&mut [0.0, 0.0]);
        }
    }
    let resets = Arc::new(AtomicUsize::new(0));
    active.drafter = Some(Box::new(DraftProbe {
        resets: resets.clone(),
        panic: false,
    }));
    let before = snapshot(&active);
    let result = active.with_ingest_recovery(|s| {
        s.append_tokens(&[1])?;
        s.append_embeddings(&[0.0, 1.0], 1)?;
        s.append_tokens(&[2])
    });
    assert!(matches!(result, Err(CeraError::InvalidToken { id: 2, .. })));
    assert_eq!(outcome(&active), RecoveryOutcome::Restored);
    assert_eq!(snapshot(&active), before);
    assert_eq!(resets.load(Ordering::Relaxed), 0);
    for _ in 0..40 {
        assert_eq!(
            active.sampler.sample(&mut [0.0, 0.0]),
            reference.sampler.sample(&mut [0.0, 0.0])
        );
    }
    active.append_tokens(&[0]).unwrap();
    reference.append_tokens(&[0]).unwrap();
    assert_eq!(active.last_logits(), reference.last_logits());
    assert_eq!(
        format!("{:?}", active.state.snapshot()),
        format!("{:?}", reference.state.snapshot())
    );
}

#[test]
fn destructive_shift_cannot_restore_even_when_reported_position_matches() {
    for embeddings in [false, true] {
        let model = FaultModel::new();
        let mut active = session(model.clone(), 4, 1);
        active.append_tokens(&[0, 1, 0]).unwrap();
        active.cancel();
        let resets = Arc::new(AtomicUsize::new(0));
        active.drafter = Some(Box::new(DraftProbe {
            resets: resets.clone(),
            panic: false,
        }));
        let result = active.with_ingest_recovery(|s| {
            let result = if embeddings {
                s.append_embeddings(&[0.0; 4], 2)
            } else {
                s.append_tokens(&[1, 1])
            };
            assert_eq!(s.position(), 3); // one cell shifted out, one consumed
            result
        });
        assert!(matches!(result, Err(CeraError::Cancelled)));
        assert_eq!(outcome(&active), RecoveryOutcome::Reset);
        assert!(active.is_usable());
        assert_eq!(active.position(), 0);
        assert!(active.token_history.is_empty() && active.last_logits().is_none());
        assert_eq!(active.prefill_tokens, 0);
        assert_eq!(active.prefill_elapsed, Duration::ZERO);
        assert_eq!(model.rewinds.load(Ordering::Relaxed), 0);
        assert_eq!(resets.load(Ordering::Relaxed), 2); // shift and recovery reset
        assert!(active.cancel.load(Ordering::Relaxed));
    }
}

#[test]
fn zero_reported_progress_is_still_a_mutation_attempt() {
    let model = FaultModel::new();
    let mut active = session(model.clone(), 16, 0);
    active.append_tokens(&[0, 1]).unwrap();
    let before = snapshot(&active);
    model.fault.store(1, Ordering::Relaxed);
    assert!(matches!(
        active.with_ingest_recovery(|s| s.append_tokens(&[0, 1])),
        Err(CeraError::Cancelled)
    ));
    assert_eq!(outcome(&active), RecoveryOutcome::Restored);
    assert_eq!(snapshot(&active), before);
    assert_eq!(model.rewinds.load(Ordering::Relaxed), 1);
}

#[test]
fn failed_reset_preserves_both_errors_and_blocks_every_execution_entry() {
    let model = FaultModel::new();
    let mut active = session(model.clone(), 16, 0);
    active.append_tokens(&[0, 1]).unwrap();
    model.rewind.store(false, Ordering::Relaxed);
    model.reset.store(false, Ordering::Relaxed);
    active.cancel();
    assert!(matches!(
        active.with_ingest_recovery(|s| s.append_tokens(&[0, 1])),
        Err(CeraError::Cancelled)
    ));
    let report = active.last_ingest_recovery().unwrap();
    assert_eq!(report.outcome, RecoveryOutcome::Unusable);
    assert_eq!(report.rewind_error, Some(KvRewindError::BackendUnsupported));
    assert!(
        matches!(&report.reset_error, Some(CeraError::Backend(s)) if s == "injected reset failure")
    );
    assert!(!active.is_usable());
    struct Sink;
    impl ModalitySink for Sink {
        fn on_done(&mut self, _: FinishReason) {}
    }
    let errors = [
        active.append_tokens(&[]),
        active.append_text(""),
        active.append_embeddings(&[], 0),
        active.append_audio(&[], 0),
        active.append_image(&[]),
        active.append_image_with_opts(&[], None),
        active.append_chat_with_images(&[], &[], false),
        active.append_user_message(&Default::default()),
        active.hidden_states_for_tokens(&[]).map(|_| ()),
        active.hidden_states_mean_pooled(&[]).map(|_| ()),
        active.hidden_states_for_text("").map(|_| ()),
        active.generate(&Default::default(), &mut Sink).map(|_| ()),
    ];
    for error in errors {
        assert!(matches!(error, Err(CeraError::Backend(s)) if s.contains("unusable")));
    }
    assert!(active.cancel.load(Ordering::Relaxed)); // generate's cleanup never ran
    assert!(active.reset().is_err());
    assert!(!active.is_usable());
    active.clear_cancel();
    assert!(!active.is_usable());
    model.reset.store(true, Ordering::Relaxed);
    active.reset().unwrap();
    assert!(active.is_usable());
    assert!(active.last_ingest_recovery().is_none());
    assert_eq!(active.position(), 0);
    active.append_tokens(&[1]).unwrap();
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn cancellation_arriving_during_reset_survives_and_handles_keep_identity() {
    let model = FaultModel::new();
    model.rewind.store(false, Ordering::Relaxed);
    let mut active = session(model.clone(), 16, 0);
    active.append_tokens(&[0, 1]).unwrap();
    let cancel = active.cancel_handle();
    let position = active.position_handle();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    *model.reset_hook.lock().unwrap() = Some(Box::new({
        let entered = entered.clone();
        let resume = resume.clone();
        move || {
            entered.wait();
            resume.wait();
        }
    }));
    let thread = std::thread::spawn({
        let cancel = cancel.clone();
        move || {
            entered.wait();
            cancel.store(true, Ordering::Relaxed);
            resume.wait();
        }
    });
    let result = active.with_ingest_recovery(|s| {
        s.append_tokens(&[1])?;
        Err(CeraError::Backend("late encoder failure".into()))
    });
    thread.join().unwrap();
    assert!(matches!(result, Err(CeraError::Backend(s)) if s == "late encoder failure"));
    assert_eq!(outcome(&active), RecoveryOutcome::Reset);
    assert!(cancel.load(Ordering::Relaxed));
    assert!(Arc::ptr_eq(&cancel, &active.cancel_handle()));
    assert!(Arc::ptr_eq(&position, &active.position_handle()));
    assert_eq!(position.load(Ordering::Relaxed), 0);
    *model.reset_hook.lock().unwrap() = None;
    active.reset().unwrap();
    assert!(!cancel.load(Ordering::Relaxed)); // explicit caller reset retains old behavior
}

#[test]
fn forward_and_recovery_unwinds_leave_the_session_unusable() {
    for fault in [2, 3, 5, 6] {
        let model = FaultModel::new();
        let mut active = session(model.clone(), 16, 0);
        active.append_tokens(&[0, 1]).unwrap();
        model.fault.store(fault, Ordering::Relaxed);
        if fault >= 5 {
            model.rewind.store(false, Ordering::Relaxed);
        }
        if fault == 6 {
            active.drafter = Some(Box::new(DraftProbe {
                resets: Arc::new(AtomicUsize::new(0)),
                panic: true,
            }));
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            active.with_ingest_recovery(|s| {
                s.append_tokens(&[1])?;
                Err(CeraError::Cancelled)
            })
        }));
        assert!(result.is_err(), "fault {fault}");
        assert_eq!(outcome(&active), RecoveryOutcome::Unusable);
        assert!(!active.is_usable());
        assert!(active.ingest_mutation.is_none());
        assert!(active.append_tokens(&[0]).is_err());
    }
}

#[test]
fn raw_partial_append_retains_legacy_progress_and_does_not_report_recovery() {
    let mut active = session(FaultModel::new(), 16, 0);
    active.cancel();
    assert!(matches!(
        active.append_tokens(&[0, 1]),
        Err(CeraError::Cancelled)
    ));
    assert_eq!(active.position(), 1);
    assert!(active.last_logits().is_none());
    assert!(active.is_usable());
    assert!(active.last_ingest_recovery().is_none());
}

#[test]
fn reset_reseeds_sampler_and_preserves_session_configuration() {
    let model = FaultModel::new();
    let mut active = session(model.clone(), 16, 1);
    let mut fresh = session(model.clone(), 16, 1);
    active.append_tokens(&[0, 1]).unwrap();
    active.set_image_max_long_size(Some(256));
    for _ in 0..30 {
        active.sampler.sample(&mut [0.0, 0.0]);
    }
    model.rewind.store(false, Ordering::Relaxed);
    assert!(
        active
            .with_ingest_recovery(|s| {
                s.append_tokens(&[1])?;
                Err(CeraError::Cancelled)
            })
            .is_err()
    );
    assert_eq!(outcome(&active), RecoveryOutcome::Reset);
    assert_eq!(active.max_seq_len, 16);
    assert_eq!(active.config.n_keep, 1);
    assert_eq!(active.image_max_long_size, Some(256));
    for _ in 0..40 {
        assert_eq!(
            active.sampler.sample(&mut [0.0, 0.0]),
            fresh.sampler.sample(&mut [0.0, 0.0])
        );
    }
}

#[test]
fn explicit_reset_unwind_cannot_enable_inference_with_partial_metadata() {
    let mut active = session(FaultModel::new(), 16, 0);
    active.append_tokens(&[0, 1]).unwrap();
    active.cancel();
    active.drafter = Some(Box::new(DraftProbe {
        resets: Arc::new(AtomicUsize::new(0)),
        panic: true,
    }));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| active.reset()));
    assert!(result.is_err());
    assert!(!active.is_usable());
    assert!(active.cancel.load(Ordering::Relaxed));
    assert!(active.append_tokens(&[0]).is_err());
    active.drafter = None;
    active.reset().unwrap();
    assert!(active.is_usable());
    assert!(!active.cancel.load(Ordering::Relaxed));
}

#[test]
fn default_backend_recovery_succeeds_via_fallback_reset() {
    struct Unproven(Arc<FaultModel>);
    impl Model for Unproven {
        fn config(&self) -> &ModelConfig {
            self.0.config()
        }
        fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
            self.0.forward(tokens, pos, state)
        }
        fn truncate_kv(&self, _: &mut InferenceState, _: usize) {
            panic!("legacy rewind is not a checked capability");
        }
    }
    let mut active = session(Arc::new(Unproven(FaultModel::new())), 16, 0);
    active.append_tokens(&[0, 1]).unwrap();
    active.reset().unwrap(); // still supported for a previously usable session
    active.cancel();
    assert!(matches!(
        active.with_ingest_recovery(|s| s.append_tokens(&[0, 1])),
        Err(CeraError::Cancelled)
    ));
    assert_eq!(outcome(&active), RecoveryOutcome::Unusable);
    assert!(active.last_ingest_recovery().unwrap().reset_error.is_some());
    // Backends without checked reset fall back to state re-allocation on explicit reset:
    assert!(active.reset().is_ok());
    assert!(active.is_usable());
    assert_eq!(active.position(), 0);
}

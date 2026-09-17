use super::*;
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use crate::session::{
    FinishReason, GenerateOpts, ModalityCapabilities, ModalitySink, Session, SessionConfig,
    SpecDecode,
};
use crate::tokenizer::BpeTokenizer;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct ScriptModel {
    config: ModelConfig,
    predictions: Vec<u32>,
    forwards: AtomicUsize,
    verifications: AtomicUsize,
    truncations: Mutex<Vec<usize>>,
    panic_forward: AtomicBool,
    speculative: bool,
}

impl ScriptModel {
    fn session(predictions: Vec<u32>, speculative: bool, max: u32) -> (Arc<Self>, Session) {
        let vocab_size = predictions.iter().copied().max().unwrap_or(3).max(3) as usize + 1;
        let model = Arc::new(Self {
            config: ModelConfig {
                architecture: "decode-observation-test".into(),
                n_layers: 1,
                hidden_size: 2,
                intermediate_size: 2,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                vocab_size,
                max_seq_len: 128,
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
            },
            predictions,
            forwards: AtomicUsize::new(0),
            verifications: AtomicUsize::new(0),
            truncations: Mutex::new(Vec::new()),
            panic_forward: AtomicBool::new(false),
            speculative,
        });
        let mut vocab = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"<eos>".to_vec(),
            b"p".to_vec(),
        ];
        vocab.extend((4..vocab_size).map(|id| format!("unused{id}").into_bytes()));
        let tokenizer = BpeTokenizer::from_vocab(vocab).with_special_tokens_for_testing(
            None,
            Some(2),
            false,
            false,
        );
        let session = Session::new(
            model.clone(),
            Arc::new(tokenizer),
            ModalityCapabilities::text_only(),
            SessionConfig {
                max_seq_len: Some(max),
                seed: Some(42),
                ubatch_size: 0,
                ..Default::default()
            },
        )
        .unwrap();
        (model, session)
    }
}

impl Model for ScriptModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(pos, state.seq_len);
        self.forwards.fetch_add(tokens.len(), Ordering::Relaxed);
        for layer in &mut state.layers {
            if let LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } = layer
            {
                for (index, &token) in tokens.iter().enumerate() {
                    key_cache.extend([token as f32, (pos + index) as f32]);
                    value_cache.extend([token as f32, 1.0]);
                }
            }
        }
        state.seq_len += tokens.len();
        assert!(
            !self.panic_forward.load(Ordering::Relaxed),
            "injected forward unwind"
        );
        let token = self.predictions.get(state.seq_len).copied().unwrap_or(0);
        let mut logits = vec![0.0; self.config.vocab_size];
        logits[token as usize] = 1000.0;
        logits
    }
    fn supports_all_logits(&self) -> bool {
        self.speculative
    }
    fn forward_embedding(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.forward(tokens, pos, state);
        vec![0.0; self.config.hidden_size]
    }
    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.verifications.fetch_add(1, Ordering::Relaxed);
        tokens
            .iter()
            .enumerate()
            .flat_map(|(index, token)| self.forward(&[*token], pos + index, state))
            .collect()
    }
    fn check_kv_rewind(
        &self,
        state: &InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        state.check_truncate_to(len)
    }
    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        self.truncations.lock().unwrap().push(len);
        state.try_truncate_to(len).unwrap();
    }
}

#[derive(Clone)]
struct FixedDraft(Vec<u32>);

impl crate::spec::Drafter for FixedDraft {
    fn clone_drafter(&self) -> Box<dyn crate::spec::Drafter> {
        Box::new(self.clone())
    }
    fn reset(&mut self) {}
    fn draft(&mut self, _: &[u32], max: usize) -> Vec<u32> {
        self.0.iter().copied().take(max).collect()
    }
}

#[derive(Default)]
struct Sink {
    tokens: Vec<u32>,
    done: Vec<FinishReason>,
    cancel: Option<Arc<AtomicBool>>,
    panic_tokens: bool,
    panic_done: bool,
}

impl ModalitySink for Sink {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        if let Some(cancel) = &self.cancel {
            cancel.store(true, Ordering::Relaxed);
        }
        assert!(!self.panic_tokens, "injected token callback unwind");
    }
    fn on_done(&mut self, reason: FinishReason) {
        self.done.push(reason);
        assert!(!self.panic_done, "injected terminal callback unwind");
    }
}

fn opts() -> GenerateOpts {
    GenerateOpts {
        max_tokens: 8,
        temperature: 0.0,
        flush_every_tokens: 1,
        flush_every_ms: 0,
        ..Default::default()
    }
}

fn resident(session: &Session) -> Vec<u32> {
    match &session.state.layers[0] {
        LayerState::Attention { key_cache, .. } => {
            let (rows, remainder) = key_cache.as_chunks::<2>();
            assert!(remainder.is_empty());
            rows.iter().map(|v| v[0] as u32).collect()
        }
        _ => panic!("expected actual attention cache"),
    }
}

#[test]
fn ordinary_stops_record_actual_uncommitted_token_in_both_sampling_modes() {
    for temperature in [0.0, 0.7] {
        for stop in [1, 2] {
            let (_, mut active) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
            active.append_tokens(&[3]).unwrap();
            let mut config = opts();
            config.temperature = temperature;
            if stop == 1 {
                config.stop_tokens = vec![1];
            }
            let mut sink = Sink::default();
            let result = active.generate_observed(&config, &mut sink);
            assert_eq!(
                result.observation,
                DecodeObservation::TokenStop { token: stop }
            );
            let summary = result.result.unwrap();
            assert_eq!(summary.finish_reason, FinishReason::Stop);
            assert_eq!(summary.tokens_generated as usize, sink.tokens.len());
            assert_eq!(summary.prompt_eval_tokens, 1);
            assert_eq!(sink.tokens, if stop == 1 { vec![0] } else { vec![0, 1] });
            assert_eq!(resident(&active), [vec![3], sink.tokens].concat());
            assert_eq!(active.position() as usize, resident(&active).len());
            assert_eq!(sink.done, [FinishReason::Stop]);
        }
    }
}

#[test]
fn no_progress_exits_preserve_logits_history_kv_and_rng() {
    for case in ["zero", "cancel", "full", "empty"] {
        let (model, mut active) = ScriptModel::session(
            vec![0, 0, 1, 2],
            false,
            if case == "full" { 1 } else { 128 },
        );
        let (_, mut reference) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
        if case != "empty" {
            active.append_tokens(&[3]).unwrap();
        }
        let before = resident(&active);
        let logits = active.last_logits.clone();
        let history = active.token_history.clone();
        let forwards = model.forwards.load(Ordering::Relaxed);
        if case == "cancel" {
            active.cancel();
        }
        let mut config = opts();
        config.temperature = 0.7;
        if case == "zero" {
            config.max_tokens = 0;
        }
        let mut sink = Sink::default();
        let observed = active.generate_observed(&config, &mut sink);
        assert_eq!(
            observed.observation,
            DecodeObservation::NoProgress,
            "{case}"
        );
        if case == "empty" {
            assert!(matches!(observed.result, Err(CeraError::EmptyInput)));
            assert!(sink.done.is_empty());
        } else {
            assert_eq!(observed.result.unwrap().tokens_generated, 0);
            assert_eq!(sink.done.len(), 1);
        }
        assert_eq!(resident(&active), before);
        assert_eq!(active.last_logits, logits);
        assert_eq!(active.token_history, history);
        assert_eq!(model.forwards.load(Ordering::Relaxed), forwards);
        assert!(!active.cancel.load(Ordering::Relaxed));
        for _ in 0..16 {
            assert_eq!(
                active.sampler.sample(&mut [0.0; 4]),
                reference.sampler.sample(&mut [0.0; 4])
            );
        }
    }
}

#[test]
fn zero_output_terminal_still_consumes_stochastic_rng() {
    let (_, mut active) = ScriptModel::session(vec![0, 2], false, 128);
    let (_, mut reference) = ScriptModel::session(vec![0, 2], false, 128);
    active.append_tokens(&[3]).unwrap();
    let mut config = opts();
    config.temperature = 0.7;
    let observed = active.generate_observed(&config, &mut Sink::default());
    assert_eq!(
        observed.observation,
        DecodeObservation::TokenStop { token: 2 }
    );
    assert_eq!(observed.result.unwrap().tokens_generated, 0);
    reference.sync_sampler_from_opts(&config);
    assert_eq!(reference.sampler.sample(&mut [0.0, 0.0, 1000.0, 0.0]), 2);
    let (_, mut untouched) = ScriptModel::session(vec![0, 2], false, 128);
    untouched.sync_sampler_from_opts(&config);
    let actual: Vec<_> = (0..16)
        .map(|_| active.sampler.sample(&mut [0.0; 4]))
        .collect();
    let expected: Vec<_> = (0..16)
        .map(|_| reference.sampler.sample(&mut [0.0; 4]))
        .collect();
    let unadvanced: Vec<_> = (0..16)
        .map(|_| untouched.sampler.sample(&mut [0.0; 4]))
        .collect();
    assert_eq!(actual, expected);
    assert_ne!(
        actual, unadvanced,
        "RNG observation must distinguish an omitted sampling step"
    );
}

#[test]
fn speculative_stop_is_observed_after_accepted_draft_rewind() {
    for stop in [1, 2] {
        let (model, mut active) = ScriptModel::session(vec![0, 0, 1, 2, 3, 0], true, 128);
        active.append_tokens(&[3]).unwrap();
        active.attach_drafter(&FixedDraft(vec![1, 2, 0]));
        let mut config = opts();
        config.spec = Some(SpecDecode { ngram: 2, k: 3 });
        if stop == 1 {
            config.stop_tokens = vec![1];
        }
        let mut sink = Sink::default();
        let observed = active.generate_observed(&config, &mut sink);
        assert_eq!(
            observed.observation,
            DecodeObservation::TokenStop { token: stop }
        );
        assert_eq!(observed.result.unwrap().finish_reason, FinishReason::Stop);
        assert_eq!(model.verifications.load(Ordering::Relaxed), 1);
        assert_eq!(
            *model.truncations.lock().unwrap(),
            if stop == 1 { vec![4, 2] } else { vec![4, 3] }
        );
        assert_eq!(resident(&active), [vec![3], sink.tokens.clone()].concat());
        assert_eq!(active.token_history, resident(&active));
        assert_eq!(active.position() as usize, resident(&active).len());
        assert_eq!(sink.tokens, if stop == 1 { vec![0] } else { vec![0, 1] });
    }
}

#[test]
fn speculative_initial_stop_does_not_claim_prefill_or_draft_progress() {
    let (model, mut active) = ScriptModel::session(vec![0, 2], true, 128);
    active.append_tokens(&[3]).unwrap();
    let mut config = opts();
    config.spec = Some(SpecDecode::default());
    let observed = active.generate_observed(&config, &mut Sink::default());
    assert_eq!(
        observed.observation,
        DecodeObservation::TokenStop { token: 2 }
    );
    assert_eq!(observed.result.unwrap().tokens_generated, 0);
    assert_eq!(resident(&active), [3]);
    assert_eq!(model.verifications.load(Ordering::Relaxed), 0);
    assert_eq!(model.forwards.load(Ordering::Relaxed), 1);
}

#[test]
fn grammar_dead_end_and_completed_grammar_have_different_observations() {
    for (grammar, expected, finish) in [
        (
            "root ::= \"z\"",
            DecodeObservation::Interrupted,
            FinishReason::GrammarDeadEnd,
        ),
        (
            "root ::= \"ab\"",
            DecodeObservation::TokenStop { token: 2 },
            FinishReason::Stop,
        ),
    ] {
        let (_, mut active) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
        active.append_tokens(&[3]).unwrap();
        let mut config = opts();
        config.grammar = Some(Arc::new(crate::grammar::Grammar::parse(grammar).unwrap()));
        let observed = active.generate_observed(&config, &mut Sink::default());
        assert_eq!(observed.observation, expected);
        assert_eq!(observed.result.unwrap().finish_reason, finish);
    }
}

#[test]
fn cancellation_between_entry_and_first_spec_step_is_proven_no_progress() {
    struct CancelBeforeSpec(Arc<AtomicBool>);
    impl crate::spec::Drafter for CancelBeforeSpec {
        fn clone_drafter(&self) -> Box<dyn crate::spec::Drafter> {
            Box::new(Self(self.0.clone()))
        }
        fn reset(&mut self) {}
        fn draft(&mut self, _: &[u32], _: usize) -> Vec<u32> {
            panic!("cancel must precede any draft")
        }
        fn suggested_k(&self) -> Option<usize> {
            self.0.store(true, Ordering::Relaxed);
            Some(3)
        }
    }
    let (model, mut active) = ScriptModel::session(vec![0, 0, 1, 2], true, 128);
    active.append_tokens(&[3]).unwrap();
    active.attach_drafter(&CancelBeforeSpec(active.cancel_handle()));
    let before = resident(&active);
    let history = active.token_history.clone();
    let logits = active.last_logits.clone();
    let observed = active.generate_observed(&opts(), &mut Sink::default());
    assert_eq!(observed.observation, DecodeObservation::NoProgress);
    assert_eq!(
        observed.result.unwrap().finish_reason,
        FinishReason::Cancelled
    );
    assert_eq!(resident(&active), before);
    assert_eq!(active.token_history, history);
    assert_eq!(active.last_logits, logits);
    assert_eq!(model.forwards.load(Ordering::Relaxed), 1);
    assert_eq!(model.verifications.load(Ordering::Relaxed), 0);
    assert!(!active.cancel.load(Ordering::Relaxed));
}

#[test]
fn zero_tokens_and_pre_step_cancel_yield_no_progress_observation() {
    for case in ["max_tokens_zero", "cancel_armed"] {
        let (model, mut active) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
        active.append_tokens(&[3]).unwrap();
        let before = resident(&active);
        let history = active.token_history.clone();
        let logits = active.last_logits.clone();
        let mut config = opts();
        if case == "max_tokens_zero" {
            config.max_tokens = 0;
        } else {
            active.cancel();
        }
        let observed = active.generate_observed(&config, &mut Sink::default());
        assert_eq!(
            observed.observation,
            DecodeObservation::NoProgress,
            "{case}"
        );
        let expected_finish = if case == "max_tokens_zero" {
            FinishReason::MaxTokens
        } else {
            FinishReason::Cancelled
        };
        assert_eq!(observed.result.unwrap().finish_reason, expected_finish);
        assert_eq!(resident(&active), before);
        assert_eq!(active.token_history, history);
        assert_eq!(active.last_logits, logits);
        assert_eq!(model.forwards.load(Ordering::Relaxed), 1);
        assert!(!active.cancel.load(Ordering::Relaxed));
    }
}

#[test]
fn grammar_controls_stop_eligibility_before_recording_a_terminal() {
    for ignore_eos in [false, true] {
        let (_, mut active) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
        active.append_tokens(&[3]).unwrap();
        let mut config = opts();
        config.grammar = Some(Arc::new(
            crate::grammar::Grammar::parse("root ::= \"ab\"").unwrap(),
        ));
        config.stop_tokens = vec![0]; // Not terminal while the grammar still requires "ab".
        config.ignore_eos = ignore_eos;
        let mut sink = Sink::default();
        let observed = active.generate_observed(&config, &mut sink);
        assert_eq!(
            observed.observation,
            DecodeObservation::TokenStop { token: 2 }
        );
        assert_eq!(observed.result.unwrap().finish_reason, FinishReason::Stop);
        assert_eq!(sink.tokens, [0, 1]);
    }
    let (_, mut active) = ScriptModel::session(vec![0, 2, 0, 1, 2, 0, 1], false, 128);
    active.append_tokens(&[3]).unwrap();
    let mut config = opts();
    config.grammar = Some(Arc::new(
        crate::grammar::Grammar::parse("root ::= \"ab\"").unwrap(),
    ));
    config.grammar_trigger_tokens = vec![2];
    config.max_tokens = 6;
    let mut sink = Sink::default();
    let observed = active.generate_observed(&config, &mut sink);
    assert_eq!(observed.observation, DecodeObservation::Interrupted);
    assert_eq!(
        observed.result.unwrap().finish_reason,
        FinishReason::MaxTokens
    );
    assert_eq!(sink.tokens, [2, 0, 1, 2, 0, 1]);
    assert_eq!(resident(&active), [vec![3], sink.tokens].concat());
}

#[test]
fn budget_context_and_partial_cancel_are_interruptions() {
    for case in ["budget", "context", "cancel", "ignore-eos"] {
        let (_, mut active) = ScriptModel::session(
            vec![0, 0, 1, 2, 0],
            false,
            if case == "context" { 2 } else { 128 },
        );
        active.append_tokens(&[3]).unwrap();
        let mut config = opts();
        config.max_tokens = if case == "ignore-eos" { 4 } else { 1 };
        config.ignore_eos = case == "ignore-eos";
        if case == "context" {
            config.max_tokens = 8;
        }
        let mut sink = Sink {
            cancel: (case == "cancel").then(|| active.cancel_handle()),
            ..Default::default()
        };
        let observed = active.generate_observed(&config, &mut sink);
        assert_eq!(
            observed.observation,
            DecodeObservation::Interrupted,
            "{case}"
        );
        let expected = match case {
            "context" => FinishReason::ContextFull,
            "cancel" => FinishReason::Cancelled,
            _ => FinishReason::MaxTokens,
        };
        assert_eq!(observed.result.unwrap().finish_reason, expected);
        assert!(!sink.tokens.is_empty());
        assert_eq!(resident(&active), [vec![3], sink.tokens].concat());
        assert!(!active.cancel.load(Ordering::Relaxed));
    }
}

#[test]
fn errors_and_unwinds_never_return_a_positive_observation() {
    let (_, mut active) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
    active.usable = false;
    let observed = active.generate_observed(&opts(), &mut Sink::default());
    assert_eq!(observed.observation, DecodeObservation::Unproven);
    assert!(observed.result.is_err());
    for fault in ["forward", "tokens", "done"] {
        let (model, mut active) = ScriptModel::session(vec![0, 0, 1, 2], false, 128);
        active.append_tokens(&[3]).unwrap();
        model
            .panic_forward
            .store(fault == "forward", Ordering::Relaxed);
        let mut sink = Sink {
            panic_tokens: fault == "tokens",
            panic_done: fault == "done",
            ..Default::default()
        };
        let mut returned = None;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                returned = Some(active.generate_observed(&opts(), &mut sink));
            }))
            .is_err()
        );
        assert!(returned.is_none());
        assert!(!active.cancel.load(Ordering::Relaxed));
    }
}

mod audio;

mod rewind;

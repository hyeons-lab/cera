use super::*;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError, LayerState};
use crate::model::ModelConfig;
use crate::session::{FinishReason, ModalityCapabilities, RecoveryOutcome, SessionConfig};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

// Injected faults, armed through `StateModel::fault`.
const NO_FAULT: u8 = 0;
const PARTIAL_PREFILL: u8 = 1;
const FORWARD_PANIC: u8 = 2;
const RESET_ERROR: u8 = 3;
const RESET_BACKEND_UNSUPPORTED: u8 = 4;

struct StateModel {
    config: ModelConfig,
    answer: u32,
    calls: Mutex<Vec<(usize, Vec<u32>)>>,
    rewinds: AtomicUsize,
    resets: AtomicUsize,
    rewind_supported: AtomicBool,
    fault: AtomicU8,
}

fn state_model(tokenizer: &BpeTokenizer) -> Arc<StateModel> {
    Arc::new(StateModel {
        config: fixtures::model_config("chat-transaction-test", tokenizer.vocab_size()),
        answer: tokenizer.encode("a")[0],
        calls: Mutex::new(Vec::new()),
        rewinds: AtomicUsize::new(0),
        resets: AtomicUsize::new(0),
        rewind_supported: AtomicBool::new(true),
        fault: AtomicU8::new(NO_FAULT),
    })
}

fn session_config() -> SessionConfig {
    SessionConfig {
        seed: Some(42),
        ..Default::default()
    }
}

fn setup(tokenizer: Arc<BpeTokenizer>) -> (Arc<StateModel>, Chat<CoreExecution>) {
    let model = state_model(&tokenizer);
    let session = Session::new(
        model.clone(),
        tokenizer,
        ModalityCapabilities::text_only(),
        session_config(),
    )
    .unwrap();
    (model, core_chat(session).unwrap())
}

/// The chat and its Session are both disabled and every entry point except
/// reset is refused. Clears the injected fault, then proves a successful
/// checked reset is the one way back to Idle.
fn assert_unusable_until_reset(chat: &mut Chat<CoreExecution>, model: &StateModel) {
    assert_eq!(chat.phase(), SessionPhase::Unusable);
    assert!(!chat.execution_for_test().session.is_usable());
    assert!(chat.raw().is_err());
    assert!(chat.ingest(&user("refused")).is_err());
    assert!(chat.replace_messages(&[user("refused")]).is_err());
    assert!(chat.complete(&opts(0.0)).is_err());
    model.fault.store(NO_FAULT, Ordering::Relaxed);
    chat.reset().unwrap();
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert!(chat.execution_for_test().session.is_usable());
}

impl Model for StateModel {
    fn is_classifier(&self) -> bool {
        !self.config.class_labels.is_empty()
    }
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(pos, state.seq_len);
        self.calls.lock().unwrap().push((pos, tokens.to_vec()));
        let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[0]
        else {
            unreachable!()
        };
        key_cache.extend(
            tokens
                .iter()
                .enumerate()
                .flat_map(|(i, &token)| [token as f32, (pos + i) as f32]),
        );
        value_cache.extend(tokens.iter().flat_map(|&token| [token as f32, 1.0]));
        state.seq_len += tokens.len();
        assert_ne!(
            self.fault.load(Ordering::Relaxed),
            FORWARD_PANIC,
            "injected forward panic"
        );
        fixtures::scripted_logits(tokens, self.answer, self.config.vocab_size)
    }
    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
        _: usize,
        cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        let mut last = None;
        for (i, &token) in tokens.iter().enumerate() {
            last = Some(self.forward(&[token], pos + i, state));
            if self.fault.load(Ordering::Relaxed) == PARTIAL_PREFILL {
                return (0, last);
            }
            if cancel.load(Ordering::Relaxed) {
                return (i + 1, last);
            }
        }
        (tokens.len(), last)
    }
    fn supports_all_logits(&self) -> bool {
        true
    }
    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        tokens
            .iter()
            .enumerate()
            .flat_map(|(i, t)| self.forward(&[*t], pos + i, state))
            .collect()
    }
    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        if !self.rewind_supported.load(Ordering::Relaxed) {
            return Err(KvRewindError::BackendUnsupported);
        }
        state.check_truncate_to(len)
    }
    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        self.rewinds.fetch_add(1, Ordering::Relaxed);
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
        if self.fault.load(Ordering::Relaxed) == RESET_ERROR {
            return Err(CeraError::OutOfMemory {
                requested_bytes: 1234,
            });
        }
        if self.fault.load(Ordering::Relaxed) == RESET_BACKEND_UNSUPPORTED {
            return Err(CeraError::Backend(
                "checked KV reset is not supported by this backend".into(),
            ));
        }
        crate::model::reset_cpu_kv(self, state, compression, max)
    }
}

fn opts(temperature: f32) -> GenerateOpts {
    GenerateOpts {
        temperature,
        max_tokens: 8,
        flush_every_tokens: 1,
        flush_every_ms: 0,
        ..Default::default()
    }
}
fn user(text: &str) -> Message {
    Message::text(Role::User, text)
}
/// Physical forward inputs recorded since call index `from`, as owned data so
/// no lock is held while the chat runs the next operation.
fn calls_since(model: &StateModel, from: usize) -> Vec<(usize, Vec<u32>)> {
    model.calls.lock().unwrap()[from..].to_vec()
}
fn resident(chat: &mut Chat<CoreExecution>) -> Vec<u32> {
    let session = &chat.execution_for_test().session;
    let LayerState::Attention { key_cache, .. } = &session.state.layers[0] else {
        unreachable!()
    };
    let (rows, rest) = key_cache.as_chunks::<2>();
    assert!(rest.is_empty());
    let tokens: Vec<_> = rows.iter().map(|r| r[0] as u32).collect();
    assert_eq!(tokens, session.token_history);
    assert_eq!(tokens.len(), session.current_pos);
    tokens
}
fn snapshot(chat: &mut Chat<CoreExecution>) -> String {
    let s = &chat.execution_for_test().session;
    format!(
        "{:?}",
        (
            s.state.snapshot(),
            &s.token_history,
            &s.last_logits,
            s.current_pos,
            s.prefill_tokens,
            s.prefill_elapsed
        )
    )
}

#[test]
fn actual_ten_turns_append_only_new_boundary_and_input() {
    run_ten_turns(fixtures::tokenizer(), false);
}

fn run_ten_turns(tokenizer: Arc<BpeTokenizer>, unicode: bool) {
    for temperature in [0.0, 0.7] {
        let (model, mut chat) = setup(tokenizer.clone());
        let mut expected = Vec::new();
        for turn in 0..10 {
            let text = if unicode {
                format!("turn{turn}: café 日本語")
            } else {
                format!("turn{turn}")
            };
            let rendered = format!(
                "{}<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
                if turn == 0 {
                    "<|startoftext|>"
                } else {
                    "<|im_end|>\n"
                }
            );
            let delta = tokenizer.encode(&rendered);
            let before = chat.position();
            let calls_before = model.calls.lock().unwrap().len();
            let input = chat.ingest(&user(&text)).unwrap();
            assert_eq!(input.input_tokens, delta.len());
            assert_eq!(input.position_before, before);
            expected.extend(&delta);
            assert_eq!(resident(&mut chat), expected);
            let calls = calls_since(&model, calls_before);
            let added: Vec<_> = calls.iter().flat_map(|(_, t)| t.iter().copied()).collect();
            assert_eq!(added, delta);
            assert_eq!(calls[0].0, before);
            let result = chat.complete(&opts(temperature)).unwrap();
            assert_eq!(result.text, "a");
            assert_eq!(result.tokens, [model.answer]);
            assert_eq!(result.summary.finish_reason, FinishReason::Stop);
            assert_eq!(result.summary.prompt_eval_tokens as usize, delta.len());
            assert_eq!(chat.phase(), SessionPhase::TurnComplete);
            expected.push(model.answer);
            assert_eq!(resident(&mut chat), expected);
            assert!(matches!(
                chat.complete(&opts(temperature)),
                Err(CompleteError::Validation(ValidationError::Phase(
                    SessionPhase::TurnComplete
                )))
            ));
        }
        assert_eq!(model.resets.load(Ordering::Relaxed), 0);
        assert_eq!(model.rewinds.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn actual_recovery_restores_pending_boundary_and_full_metadata() {
    for temperature in [0.0, 0.7] {
        let (model, mut chat) = setup(fixtures::tokenizer());
        chat.ingest(&user("first")).unwrap();
        chat.complete(&opts(temperature)).unwrap();
        let before = snapshot(&mut chat);
        model.fault.store(PARTIAL_PREFILL, Ordering::Relaxed);
        let error = chat.ingest(&user("next")).unwrap_err();
        assert!(matches!(
            error.cause,
            IngestCause::Execution(CeraError::Cancelled)
        ));
        assert_eq!(error.recovery, RecoveryOutcome::Restored);
        assert!(error.rewind_error.is_none() && error.recovery_error.is_none());
        assert_eq!(snapshot(&mut chat), before);
        assert_eq!(chat.phase(), SessionPhase::TurnComplete);
        model.fault.store(NO_FAULT, Ordering::Relaxed);
        let pos = chat.position();
        chat.ingest(&user("next")).unwrap();
        // Exactly one pending EOS, the template newline, then `<|im_start|>`.
        assert_eq!(resident(&mut chat)[pos..pos + 3], [7, 9, 6]);
    }
}

#[test]
fn actual_cancelled_prefill_resets_honestly_and_preserves_handle_identity() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    chat.complete(&opts(0.0)).unwrap();
    model.rewind_supported.store(false, Ordering::Relaxed);
    let cancel = chat.execution_for_test().session.cancel_handle();
    let position = chat.execution_for_test().session.position_handle();
    cancel.store(true, Ordering::Relaxed);
    let error = chat.ingest(&user("next")).unwrap_err();
    assert_eq!(error.recovery, RecoveryOutcome::Reset);
    assert_eq!(
        error.rewind_error.as_deref(),
        Some(&KvRewindError::BackendUnsupported)
    );
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert!(resident(&mut chat).is_empty());
    assert_eq!(position.load(Ordering::Relaxed), 0);
    // Clearing cancel through the chat surface restores the latch non-destructively.
    chat.clear_cancel();
    chat.ingest(&user("retry")).unwrap();
    assert!(position.load(Ordering::Relaxed) > 0);
    assert!(Arc::ptr_eq(
        &cancel,
        &chat.execution_for_test().session.cancel_handle()
    ));
}

#[test]
fn actual_reset_failure_keeps_primary_and_typed_secondary_errors() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    chat.complete(&opts(0.0)).unwrap();
    model.rewind_supported.store(false, Ordering::Relaxed);
    model.fault.store(RESET_ERROR, Ordering::Relaxed);
    chat.execution_for_test().session.cancel();
    let error = chat.ingest(&user("next")).unwrap_err();
    assert!(matches!(
        error.cause,
        IngestCause::Execution(CeraError::Cancelled)
    ));
    assert_eq!(error.recovery, RecoveryOutcome::Unusable);
    assert_eq!(
        error.rewind_error.as_deref(),
        Some(&KvRewindError::BackendUnsupported)
    );
    assert!(matches!(
        error.recovery_error,
        Some(CeraError::OutOfMemory {
            requested_bytes: 1234
        })
    ));
    assert_unusable_until_reset(&mut chat, &model);
}

#[test]
fn direct_reset_failure_from_a_healthy_turn_is_unusable_until_reset_succeeds() {
    for through_replacement in [false, true] {
        let (model, mut chat) = setup(fixtures::tokenizer());
        chat.ingest(&user("first")).unwrap();
        chat.complete(&opts(0.0)).unwrap();
        model.fault.store(RESET_ERROR, Ordering::Relaxed);
        if through_replacement {
            let error = chat.replace_messages(&[user("replacement")]).unwrap_err();
            assert!(matches!(
                error.cause,
                IngestCause::Execution(CeraError::OutOfMemory {
                    requested_bytes: 1234
                })
            ));
            assert_eq!(error.recovery, RecoveryOutcome::Unusable);
            assert!(error.rewind_error.is_none() && error.recovery_error.is_none());
        } else {
            assert!(matches!(
                chat.reset(),
                Err(CeraError::OutOfMemory {
                    requested_bytes: 1234
                })
            ));
        }
        // The backend contract allows a failed checked reset to leave state
        // invalid, so the intact CPU rows are not certified either, and the
        // stale prefill logits are dropped as in the recovery path.
        assert!(chat.execution_for_test().session.last_logits.is_none());
        assert_eq!(model.resets.load(Ordering::Relaxed), 1);
        assert_unusable_until_reset(&mut chat, &model);
        // Only the helper's successful reset reached the backend again.
        assert_eq!(model.resets.load(Ordering::Relaxed), 2);
        assert!(resident(&mut chat).is_empty());
        chat.ingest(&user("next")).unwrap();
        assert_eq!(chat.complete(&opts(0.0)).unwrap().text, "a");
    }
}

#[test]
fn actual_validation_and_replacement_preserve_old_context_until_reset() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    let before = snapshot(&mut chat);
    for messages in [
        vec![],
        vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image(vec![1])],
        }],
    ] {
        assert!(chat.replace_messages(&messages).is_err());
        assert_eq!(snapshot(&mut chat), before);
        assert_eq!(chat.phase(), SessionPhase::PromptReady);
        assert_eq!(model.resets.load(Ordering::Relaxed), 0);
    }
    chat.execution_for_test().session.cancel();
    let error = chat.replace_messages(&[user("replacement")]).unwrap_err();
    assert_eq!(error.recovery, RecoveryOutcome::Reset);
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert!(resident(&mut chat).is_empty());
    assert!(
        chat.execution_for_test()
            .session
            .cancel
            .load(Ordering::Relaxed)
    );
    chat.reset().unwrap();
    assert!(
        !chat
            .execution_for_test()
            .session
            .cancel
            .load(Ordering::Relaxed)
    );
}

#[test]
fn raw_replacement_cannot_reuse_the_old_profile_identity() {
    // One shared tokenizer, so only the model identity distinguishes the two.
    let tokenizer = fixtures::tokenizer();
    let (_, mut chat) = setup(tokenizer.clone());
    let (other_model, mut other) = setup(tokenizer);
    std::mem::swap(
        &mut chat.raw().unwrap().session,
        &mut other.raw().unwrap().session,
    );
    let error = chat.replace_messages(&[user("next")]).unwrap_err();
    assert!(matches!(
        error.cause,
        IngestCause::Validation(ValidationError::UnsupportedProfile)
    ));
    assert_eq!(error.recovery, RecoveryOutcome::Unchanged);
    assert_eq!(chat.phase(), SessionPhase::RawContext);
    assert!(other_model.calls.lock().unwrap().is_empty());
    assert_eq!(other_model.resets.load(Ordering::Relaxed), 0);
    // An explicit reset does not consult the stale profile, so the chat is not
    // stranded; the classifier-adapter case below exercises the full undo path.
    chat.reset().unwrap();
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert_eq!(other_model.resets.load(Ordering::Relaxed), 1);
    assert!(other_model.calls.lock().unwrap().is_empty());
}

#[test]
fn raw_swap_cannot_install_a_sliding_context_under_the_same_identity() {
    // Same model and tokenizer Arcs, so only the re-checked `n_keep` differs.
    let tokenizer = fixtures::tokenizer();
    let (model, mut chat) = setup(tokenizer.clone());
    chat.ingest(&user("first")).unwrap();
    let mut sliding = Session::new(
        model.clone(),
        tokenizer,
        ModalityCapabilities::text_only(),
        SessionConfig {
            n_keep: 4,
            ..session_config()
        },
    )
    .unwrap();
    std::mem::swap(&mut chat.raw().unwrap().session, &mut sliding);
    let error = chat.replace_messages(&[user("next")]).unwrap_err();
    assert!(matches!(
        error.cause,
        IngestCause::Validation(ValidationError::SlidingContext)
    ));
    assert_eq!(error.recovery, RecoveryOutcome::Unchanged);
    assert_eq!(chat.phase(), SessionPhase::RawContext);
    assert_eq!(model.resets.load(Ordering::Relaxed), 0);
    std::mem::swap(&mut chat.raw().unwrap().session, &mut sliding);
    chat.replace_messages(&[user("next")]).unwrap();
    assert_eq!(chat.complete(&opts(0.0)).unwrap().text, "a");
}

#[test]
fn construction_rejects_unsupported_profiles_before_inference() {
    let tokenizer = fixtures::tokenizer();
    let text = ModalityCapabilities::text_only();
    type Case = (
        fn(&mut ModelConfig),
        ModalityCapabilities,
        SessionConfig,
        ValidationError,
    );
    let cases: [Case; 6] = [
        (
            |config| config.is_causal = false,
            text,
            session_config(),
            ValidationError::UnsupportedProfile,
        ),
        (
            |config| config.class_labels.push("class".into()),
            text,
            session_config(),
            ValidationError::UnsupportedProfile,
        ),
        (
            // A model vocabulary smaller than the tokenizer's can index out of range.
            |config| config.vocab_size -= 1,
            text,
            session_config(),
            ValidationError::UnsupportedProfile,
        ),
        (
            |_| {},
            ModalityCapabilities {
                text_in: false,
                ..text
            },
            session_config(),
            ValidationError::UnsupportedProfile,
        ),
        (
            |_| {},
            ModalityCapabilities {
                text_out: false,
                ..text
            },
            session_config(),
            ValidationError::UnsupportedProfile,
        ),
        (
            |_| {},
            text,
            SessionConfig {
                n_keep: 4,
                ..session_config()
            },
            ValidationError::SlidingContext,
        ),
    ];
    for (mutate, capabilities, config, expected) in cases {
        let mut model = state_model(&tokenizer);
        mutate(&mut Arc::get_mut(&mut model).unwrap().config);
        let session = Session::new(model.clone(), tokenizer.clone(), capabilities, config).unwrap();
        let (recovered_session, err) = core_chat(session).unwrap_err();
        assert_eq!(err, expected);
        assert!(recovered_session.is_usable());
        assert!(model.calls.lock().unwrap().is_empty());
        assert_eq!(model.resets.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn no_progress_error_keeps_execution_usable_and_opens_replacement() {
    // A prompt whose logits vanished cannot be certified, but the observation
    // proves nothing was mutated: the cursor is stale, the Session is fine.
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    let before = resident(&mut chat);
    chat.execution_for_test().session.last_logits = None;
    assert!(matches!(
        chat.complete(&opts(0.0)),
        Err(CompleteError::Execution(CeraError::EmptyInput))
    ));
    assert_eq!(chat.phase(), SessionPhase::RawContext);
    assert!(chat.execution_for_test().session.is_usable());
    assert_eq!(resident(&mut chat), before);
    assert_eq!(model.resets.load(Ordering::Relaxed), 0);
    chat.replace_messages(&[user("again")]).unwrap();
    assert_eq!(chat.complete(&opts(0.0)).unwrap().text, "a");
}

#[test]
#[ignore = "requires pinned public GGUF; run tests/api_chat/run.py --core-transactions"]
fn public_tokenizer_actual_session_ten_turns() {
    let path =
        std::env::var("CERA_CHAT_PROFILE_MODEL").expect("runner must supply the pinned model");
    let gguf = crate::gguf::GgufFile::from_bytes(std::fs::read(path).unwrap().into()).unwrap();
    run_ten_turns(Arc::new(BpeTokenizer::from_gguf(&gguf).unwrap()), true);
}

fn load_real_lfm2(bytes: &[u8]) -> (Arc<dyn Model>, Arc<BpeTokenizer>) {
    let arc_bytes: Arc<[u8]> = Arc::from(bytes);
    let gguf_tok = crate::gguf::GgufFile::from_bytes(arc_bytes.clone()).unwrap();
    let tokenizer = Arc::new(BpeTokenizer::from_gguf(&gguf_tok).unwrap());
    let gguf_model = crate::gguf::GgufFile::from_bytes(arc_bytes).unwrap();
    let model: Arc<dyn Model> =
        Arc::from(crate::model::load_model(gguf_model, None, 4096).unwrap());
    (model, tokenizer)
}

fn real_lfm2_session_from_model(
    model: Arc<dyn Model>,
    tokenizer: Arc<BpeTokenizer>,
    seed: Option<u64>,
) -> Session {
    Session::new(
        model,
        tokenizer,
        ModalityCapabilities::text_only(),
        SessionConfig {
            seed,
            ..Default::default()
        },
    )
    .unwrap()
}

fn real_lfm2_session(bytes: &[u8], seed: Option<u64>) -> (Session, Arc<BpeTokenizer>) {
    let (model, tokenizer) = load_real_lfm2(bytes);
    let session = real_lfm2_session_from_model(model, tokenizer.clone(), seed);
    (session, tokenizer)
}

#[test]
#[ignore = "requires pinned public GGUF; run tests/api_chat/run.py --core-transactions"]
fn real_model_r1_ten_warm_turns_and_kv_retention() {
    let path =
        std::env::var("CERA_CHAT_PROFILE_MODEL").expect("runner must supply the pinned model");
    let bytes = std::fs::read(path).unwrap();
    let (model, tokenizer) = load_real_lfm2(&bytes);
    let session = real_lfm2_session_from_model(model.clone(), tokenizer.clone(), Some(42));
    let mut chat = core_chat(session).unwrap();

    let kv_dim = model.config().n_kv_heads * model.config().head_dim;
    let mut prior_kv_snapshots: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let mut turn_outputs: Vec<(String, Vec<u32>, String)> = Vec::new();
    let grammar =
        Arc::new(crate::grammar::Grammar::parse(r#"root ::= [a-zA-Z]{1,16} "\n""#).unwrap());
    let opts = GenerateOpts {
        max_tokens: 64,
        temperature: 0.7,
        grammar: Some(grammar),
        ..Default::default()
    };

    let prompts = [
        "Say hi.",
        "Say ok.",
        "Say yes.",
        "Say cool.",
        "Say wow.",
        "Say done.",
        "Say bye.",
        "Say hello.",
        "Say fine.",
        "Say good.",
    ];

    for (turn, prompt) in prompts.iter().enumerate() {
        let pos_before = chat.position();
        let user_msg = user(prompt);
        let summary = if turn == 0 {
            chat.ingest_messages(&[Message::text(Role::System, "Be concise."), user_msg.clone()])
                .unwrap()
        } else {
            chat.ingest(&user_msg).unwrap()
        };
        assert_eq!(summary.position_before, pos_before);
        assert!(summary.input_tokens > 0);
        assert_eq!(chat.position(), pos_before + summary.input_tokens);
        assert_eq!(chat.phase(), SessionPhase::PromptReady);

        if turn == 0 {
            // Verify cold reference prefill parity on turn 0:
            let cold_session =
                real_lfm2_session_from_model(model.clone(), tokenizer.clone(), Some(42));
            let mut cold_chat = core_chat(cold_session).unwrap();
            cold_chat
                .ingest_messages(&[Message::text(Role::System, "Be concise."), user_msg.clone()])
                .unwrap();
            let warm_logits = chat
                .execution_for_test()
                .session
                .last_logits()
                .unwrap()
                .to_vec();
            let cold_logits = cold_chat
                .execution_for_test()
                .session
                .last_logits()
                .unwrap()
                .to_vec();
            assert_eq!(
                warm_logits, cold_logits,
                "turn 0 warm prefill logits must be identical to cold reference"
            );
        } else {
            // Verify physical KV immutability: delta ingestion must leave prior KV intact:
            let warm_sess = &chat.execution_for_test().session;
            let mut attn_layer_idx = 0;
            for layer in &warm_sess.state.layers {
                if let LayerState::Attention {
                    key_cache,
                    value_cache,
                    ..
                } = layer
                {
                    let (prev_k, prev_v) = &prior_kv_snapshots[attn_layer_idx];
                    assert!(
                        key_cache.len() >= pos_before * kv_dim,
                        "layer {attn_layer_idx} key cache must contain all prior positions"
                    );
                    assert_eq!(
                        &key_cache[..pos_before * kv_dim],
                        &prev_k[..pos_before * kv_dim],
                        "layer {attn_layer_idx} key cache before position {pos_before} must be bitwise identical"
                    );
                    assert_eq!(
                        &value_cache[..pos_before * kv_dim],
                        &prev_v[..pos_before * kv_dim],
                        "layer {attn_layer_idx} value cache before position {pos_before} must be bitwise identical"
                    );
                    attn_layer_idx += 1;
                }
            }
        }

        let result = chat.complete(&opts).unwrap();
        assert!(result.summary.tokens_generated > 0);
        assert_eq!(result.summary.finish_reason, FinishReason::Stop);
        assert_eq!(chat.phase(), SessionPhase::TurnComplete);
        assert_eq!(
            chat.position(),
            pos_before + summary.input_tokens + result.summary.tokens_generated as usize
        );
        turn_outputs.push((prompt.to_string(), result.tokens, result.text));

        // Snapshot full KV cache for all attention layers up to the end of this turn:
        let warm_sess = &chat.execution_for_test().session;
        prior_kv_snapshots = warm_sess
            .state
            .layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    ..
                } => Some((key_cache.clone(), value_cache.clone())),
                _ => None,
            })
            .collect();
        assert!(
            !prior_kv_snapshots.is_empty(),
            "model must have attention layers"
        );

        // Verify that convolution layers retain live recurrent state across turns:
        let conv_layers: Vec<&Vec<f32>> = warm_sess
            .state
            .layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Conv { buffer, .. } => Some(buffer),
                _ => None,
            })
            .collect();
        assert!(
            !conv_layers.is_empty(),
            "model must have convolution layers"
        );
        for (idx, buf) in conv_layers.iter().enumerate() {
            assert!(
                buf.iter().any(|&x| x != 0.0),
                "conv layer {idx} buffer must maintain live recurrent activation state across turns"
            );
        }
    }

    // Proof of determinism across consecutive runs with identical seed:
    let session2 = real_lfm2_session_from_model(model.clone(), tokenizer.clone(), Some(42));
    let mut chat2 = core_chat(session2).unwrap();
    for (turn, (prompt, expected_tokens, expected_text)) in turn_outputs.iter().enumerate() {
        let summary2 = if turn == 0 {
            chat2
                .ingest_messages(&[Message::text(Role::System, "Be concise."), user(prompt)])
                .unwrap()
        } else {
            chat2.ingest(&user(prompt)).unwrap()
        };
        assert!(summary2.input_tokens > 0);
        let result = chat2.complete(&opts).unwrap();
        assert_eq!(&result.tokens, expected_tokens, "turn {turn} tokens match");
        assert_eq!(&result.text, expected_text, "turn {turn} text matches");
        assert_eq!(result.summary.finish_reason, FinishReason::Stop);
    }
    assert_eq!(chat2.position(), chat.position());
}

#[test]
#[ignore = "requires pinned public GGUF; run tests/api_chat/run.py --core-transactions"]
fn real_model_r1_stochastic_rng_determinism_and_divergence() {
    let path =
        std::env::var("CERA_CHAT_PROFILE_MODEL").expect("runner must supply the pinned model");
    let bytes = std::fs::read(path).unwrap();
    let (model, tokenizer) = load_real_lfm2(&bytes);

    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.7,
        ..Default::default()
    };

    let prompt = [
        Message::text(Role::System, "You are a creative storyteller."),
        user("Once upon a time in a distant galaxy, there was a"),
    ];

    let s1 = real_lfm2_session_from_model(model.clone(), tokenizer.clone(), Some(777));
    let mut chat1 = core_chat(s1).unwrap();
    chat1.ingest_messages(&prompt).unwrap();
    let r1 = chat1.complete(&opts).unwrap();

    let s2 = real_lfm2_session_from_model(model.clone(), tokenizer.clone(), Some(777));
    let mut chat2 = core_chat(s2).unwrap();
    chat2.ingest_messages(&prompt).unwrap();
    let r2 = chat2.complete(&opts).unwrap();

    assert_eq!(
        r1.tokens, r2.tokens,
        "identical seeds must produce identical stochastic tokens"
    );
    assert_eq!(r1.text, r2.text);

    let s3 = real_lfm2_session_from_model(model.clone(), tokenizer.clone(), Some(888));
    let mut chat3 = core_chat(s3).unwrap();
    chat3.ingest_messages(&prompt).unwrap();
    let r3 = chat3.complete(&opts).unwrap();

    assert_ne!(
        r1.tokens, r3.tokens,
        "different seeds must produce divergent stochastic tokens"
    );
}

#[test]
#[ignore = "requires pinned public GGUF; run tests/api_chat/run.py --core-transactions"]
fn real_model_r1_interrupted_turn_and_replacement_recovery() {
    let path =
        std::env::var("CERA_CHAT_PROFILE_MODEL").expect("runner must supply the pinned model");
    let bytes = std::fs::read(path).unwrap();
    let (model, tokenizer) = load_real_lfm2(&bytes);

    let session = real_lfm2_session_from_model(model, tokenizer, Some(42));
    let mut chat = core_chat(session).unwrap();

    // 1. Ingest turn and execute with max_tokens: 2, causing FinishReason::MaxTokens:
    let _summary = chat.ingest(&user("What is 2 + 2?")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    let opts_short = GenerateOpts {
        max_tokens: 2,
        temperature: 0.0,
        ..Default::default()
    };
    let result = chat.complete(&opts_short).unwrap();
    assert_eq!(result.tokens.len(), 2);
    assert_eq!(result.summary.finish_reason, FinishReason::MaxTokens);
    assert_eq!(chat.phase(), SessionPhase::Interrupted);

    // 2. Interrupted turn rejects subsequent ingest:
    let err = chat.ingest(&user("Next question")).unwrap_err();
    assert!(matches!(
        err.cause,
        IngestCause::Validation(ValidationError::Phase(SessionPhase::Interrupted))
    ));
    assert_eq!(chat.phase(), SessionPhase::Interrupted);

    // 3. Replacement reset cleanly restores prompt ready state and rewinds KV cache:
    let interrupted_pos = chat.position();
    let rep_summary = chat
        .replace_messages(&[Message::text(Role::System, "Be concise."), user("Say hi.")])
        .unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    assert_eq!(rep_summary.position_before, interrupted_pos);
    assert_eq!(rep_summary.position_after, rep_summary.input_tokens);
    assert_eq!(chat.position(), rep_summary.input_tokens);

    let opts_full = GenerateOpts {
        max_tokens: 64,
        temperature: 0.7,
        ..Default::default()
    };
    let fresh_result = chat.complete(&opts_full).unwrap();
    assert_eq!(fresh_result.summary.finish_reason, FinishReason::Stop);
    assert_eq!(chat.phase(), SessionPhase::TurnComplete);

    // Verify against a cold independent session with the same replacement prompt:
    let (cold_sess, _) = real_lfm2_session(&bytes, Some(42));
    let mut cold_chat = core_chat(cold_sess).unwrap();
    cold_chat
        .ingest_messages(&[Message::text(Role::System, "Be concise."), user("Say hi.")])
        .unwrap();
    let cold_result = cold_chat.complete(&opts_full).unwrap();

    assert_eq!(fresh_result.tokens, cold_result.tokens);
    assert_eq!(fresh_result.text, cold_result.text);
    assert_eq!(cold_result.summary.finish_reason, FinishReason::Stop);
}

#[test]
fn no_progress_keeps_prompt_ready_and_preserves_future_rng() {
    for temperature in [0.0, 0.7] {
        let (_, mut chat) = setup(fixtures::tokenizer());
        let (_, mut reference) = setup(fixtures::tokenizer());
        for c in [&mut chat, &mut reference] {
            c.ingest(&user("first")).unwrap();
        }
        let before = resident(&mut chat);
        let logits = chat.execution_for_test().session.last_logits.clone();
        let zero = GenerateOpts {
            max_tokens: 0,
            ..opts(temperature)
        };
        let result = chat.complete(&zero).unwrap();
        assert_eq!(result.summary.finish_reason, FinishReason::MaxTokens);
        assert_eq!(chat.phase(), SessionPhase::PromptReady);
        chat.cancel();
        let result = chat.complete(&opts(temperature)).unwrap();
        assert_eq!(result.summary.finish_reason, FinishReason::Cancelled);
        assert_eq!(chat.phase(), SessionPhase::PromptReady);
        assert!(!chat.cancel_handle().unwrap().load(Ordering::Relaxed));
        assert_eq!(resident(&mut chat), before);
        assert_eq!(chat.execution_for_test().session.last_logits, logits);
        for c in [&mut chat, &mut reference] {
            assert_eq!(c.complete(&opts(temperature)).unwrap().text, "a");
        }
        for _ in 0..16 {
            let left = chat
                .execution_for_test()
                .session
                .sampler
                .sample(&mut [0.0, 0.0]);
            let right = reference
                .execution_for_test()
                .session
                .sampler
                .sample(&mut [0.0, 0.0]);
            assert_eq!(left, right);
        }
    }
}

struct PanicSink {
    tokens: bool,
    done: bool,
}
impl ModalitySink for PanicSink {
    fn on_text_tokens(&mut self, _: &[u32]) {
        assert!(!self.tokens, "injected token callback panic");
    }
    fn on_done(&mut self, _: FinishReason) {
        assert!(!self.done, "injected done callback panic");
    }
}

#[test]
fn actual_decode_unwinds_disable_chat_and_raw_execution() {
    for case in ["forward", "tokens", "done"] {
        let (model, mut chat) = setup(fixtures::tokenizer());
        chat.ingest(&user("first")).unwrap();
        if case == "forward" {
            model.fault.store(FORWARD_PANIC, Ordering::Relaxed);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            chat.generate_into(
                &opts(0.0),
                &mut PanicSink {
                    tokens: case == "tokens",
                    done: case == "done",
                },
            )
        }));
        assert!(result.is_err());
        assert_unusable_until_reset(&mut chat, &model);
    }
}

#[test]
fn actual_prefill_unwind_disables_chat_and_raw_execution() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    chat.complete(&opts(0.0)).unwrap();
    model.fault.store(FORWARD_PANIC, Ordering::Relaxed);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| chat.ingest(&user("next"))));
    assert!(result.is_err());
    // The recovery guard recorded the unwind; the adapter never got to take it.
    assert_eq!(
        chat.execution_for_test()
            .session
            .last_ingest_recovery()
            .map(|r| r.outcome),
        Some(RecoveryOutcome::Unusable)
    );
    assert_unusable_until_reset(&mut chat, &model);
    assert!(
        chat.execution_for_test()
            .session
            .last_ingest_recovery()
            .is_none()
    );
}

#[derive(Clone)]
struct Draft {
    tokens: Vec<u32>,
    resets: Arc<AtomicUsize>,
}
impl crate::spec::Drafter for Draft {
    fn clone_drafter(&self) -> Box<dyn crate::spec::Drafter> {
        Box::new(self.clone())
    }
    fn draft(&mut self, _: &[u32], max: usize) -> Vec<u32> {
        self.tokens.iter().copied().take(max).collect()
    }
    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn actual_speculative_observation_and_recovery_preserve_drafter_contract() {
    for checked in [true, false] {
        let (model, mut chat) = setup(fixtures::tokenizer());
        chat.ingest(&user("first")).unwrap();
        let resets = Arc::new(AtomicUsize::new(0));
        chat.execution_for_test().session.attach_drafter(&Draft {
            tokens: vec![7, model.answer, 9],
            resets: resets.clone(),
        });
        model.rewind_supported.store(checked, Ordering::Relaxed);
        let result = chat.complete(&opts(0.0)).unwrap();
        assert_eq!(result.text, "a");
        assert_eq!(result.summary.finish_reason, FinishReason::Stop);
        if checked {
            assert_eq!(chat.phase(), SessionPhase::TurnComplete);
            model.fault.store(PARTIAL_PREFILL, Ordering::Relaxed);
            let error = chat.ingest(&user("next")).unwrap_err();
            assert_eq!(error.recovery, RecoveryOutcome::Restored);
            assert_eq!(chat.phase(), SessionPhase::TurnComplete);
            assert_eq!(resets.load(Ordering::Relaxed), 0);
        } else {
            assert_eq!(chat.phase(), SessionPhase::Unusable);
            assert!(!chat.execution_for_test().session.is_usable());
            assert!(chat.ingest(&user("next")).is_err());
        }
        model.fault.store(NO_FAULT, Ordering::Relaxed);
        chat.reset().unwrap();
        assert_eq!(resets.load(Ordering::Relaxed), 1);
        assert_eq!(chat.phase(), SessionPhase::Idle);
    }
}

#[test]
fn actual_batch_has_one_final_prefix_and_rejects_a_second_ingest() {
    let tokenizer = fixtures::tokenizer();
    let (model, mut chat) = setup(tokenizer.clone());
    let messages = [
        Message::text(Role::System, "system"),
        user("one"),
        Message::text(Role::Assistant, "answer"),
        user("two"),
    ];
    let raw = [
        ("system", "system"),
        ("user", "one"),
        ("assistant", "answer"),
        ("user", "two"),
    ]
    .map(|(role, content)| crate::tokenizer::ChatMessage {
        role: role.into(),
        content: content.into(),
    });
    let expected =
        tokenizer.encode(&crate::tokenizer::apply_chat_template(&tokenizer, &raw, true).unwrap());
    let summary = chat.ingest_messages(&messages).unwrap();
    assert_eq!(summary.input_tokens, expected.len());
    assert_eq!(resident(&mut chat), expected);
    let calls = model.calls.lock().unwrap().len();
    assert!(chat.ingest(&user("extra")).is_err());
    assert_eq!(model.calls.lock().unwrap().len(), calls);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
}

#[test]
fn actual_raw_append_requires_replacement_before_chat() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    chat.raw()
        .unwrap()
        .session
        .append_tokens(&[model.answer])
        .unwrap();
    assert_eq!(chat.phase(), SessionPhase::RawContext);
    assert!(chat.ingest(&user("next")).is_err());
    assert!(chat.complete(&opts(0.0)).is_err());
    chat.replace_messages(&[user("replacement")]).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    assert_eq!(chat.complete(&opts(0.0)).unwrap().text, "a");
}

#[test]
fn a_zero_output_custom_stop_is_interrupted_in_both_sampling_modes() {
    for temperature in [0.0, 0.7] {
        let (model, mut chat) = setup(fixtures::tokenizer());
        chat.ingest(&user("first")).unwrap();
        let config = GenerateOpts {
            stop_tokens: vec![model.answer],
            ..opts(temperature)
        };
        let result = chat.complete(&config).unwrap();
        assert!(result.tokens.is_empty());
        assert_eq!(result.summary.finish_reason, FinishReason::Stop);
        assert_eq!(chat.phase(), SessionPhase::Interrupted);
        assert!(chat.ingest(&user("next")).is_err());
        assert!(chat.complete(&opts(temperature)).is_err());
    }
}

#[test]
fn classifier_adapter_is_rejected_before_replacement_reset() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    let adapter = crate::lora::LoraAdapterWeights::new_classifier_for_testing(
        vec![0.0; 2],
        None,
        vec!["class".into()],
    );
    chat.raw()
        .unwrap()
        .session
        .attach_lora_adapters(adapter)
        .unwrap();
    let before = snapshot(&mut chat);
    let calls = model.calls.lock().unwrap().len();
    let error = chat.replace_messages(&[user("next")]).unwrap_err();
    assert!(matches!(
        error.cause,
        IngestCause::Validation(ValidationError::UnsupportedProfile)
    ));
    assert_eq!(error.recovery, RecoveryOutcome::Unchanged);
    assert_eq!(chat.phase(), SessionPhase::RawContext);
    assert_eq!(snapshot(&mut chat), before);
    assert_eq!(model.calls.lock().unwrap().len(), calls);
    assert_eq!(model.resets.load(Ordering::Relaxed), 0);
    // Identity drift must not strand the chat: an explicit reset still runs
    // the checked KV reset, and Idle keeps raw access open to remove the adapter.
    chat.reset().unwrap();
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert_eq!(model.resets.load(Ordering::Relaxed), 1);
    assert!(chat.execution_for_test().session.has_lora_adapters());
    let error = chat.ingest(&user("next")).unwrap_err();
    assert!(matches!(
        error.cause,
        IngestCause::Validation(ValidationError::UnsupportedProfile)
    ));
    assert_eq!(chat.phase(), SessionPhase::Idle);
    chat.raw().unwrap().session.remove_lora_adapters();
    chat.replace_messages(&[user("next")]).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    assert_eq!(chat.complete(&opts(0.0)).unwrap().text, "a");
}

#[test]
fn chat_into_session_reclaims_usable_session() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    let summary = chat.ingest(&user("hello world")).unwrap();
    assert!(summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let session = chat.into_session();
    assert!(session.is_usable());
    assert_eq!(session.position() as usize, summary.input_tokens);

    let roundtrip_chat = session
        .into_chat()
        .expect("roundtrip into_chat must succeed");
    assert_eq!(roundtrip_chat.phase(), SessionPhase::RawContext);
    assert_eq!(roundtrip_chat.position(), summary.input_tokens);
}

#[test]
fn chat_cancellation_methods_and_handle() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    let handle = chat.cancel_handle().expect("cancel handle must be present");
    assert!(!handle.load(Ordering::Relaxed));

    chat.cancel();
    assert!(handle.load(Ordering::Relaxed));

    chat.clear_cancel();
    assert!(!handle.load(Ordering::Relaxed));
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert_eq!(chat.position(), 0);
}

#[test]
fn fallback_reset_for_backend_without_try_reset_kv() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("initial message")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    assert!(chat.position() > 0);

    // Arm backend fault so try_reset_kv returns the default unsupported Backend error.
    model
        .fault
        .store(RESET_BACKEND_UNSUPPORTED, Ordering::Relaxed);

    // Explicit reset must succeed via fallback re-allocation.
    chat.reset().unwrap();
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert_eq!(chat.position(), 0);
    assert!(chat.execution_for_test().session.is_usable());

    // Ingest a fresh message after fallback reset.
    chat.ingest(&user("after reset")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    assert!(chat.position() > 0);

    // Replacement reset also uses reset() internally and must succeed via fallback.
    chat.replace_messages(&[user("replacement message")])
        .unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
    assert!(chat.execution_for_test().session.is_usable());

    // Verify generation completes successfully.
    let turn = chat.complete(&opts(0.0)).unwrap();
    assert_eq!(turn.text, "a");
}

#[test]
fn chat_into_session_reclaims_unusable_session_and_recovers_via_reset() {
    let (model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("first")).unwrap();
    // Arm forward panic to force SessionPhase::Unusable.
    model.fault.store(FORWARD_PANIC, Ordering::Relaxed);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| chat.complete(&opts(0.0))));
    assert!(result.is_err());
    assert_eq!(chat.phase(), SessionPhase::Unusable);

    // Reclaim session from unusable chat.
    let mut session = chat.into_session();
    assert!(!session.is_usable());

    // Disarm fault and recover session via reset.
    model.fault.store(NO_FAULT, Ordering::Relaxed);
    session.reset().unwrap();
    assert!(session.is_usable());
    assert_eq!(session.position(), 0);
}

#[test]
fn core_chat_refusal_on_unsupported_tokenizer_profile_preserves_session() {
    let tokenizer = fixtures::tokenizer();
    let model = state_model(&tokenizer);

    // Create session with minimal tokenizer that has no ChatML template or special tokens.
    let empty_tokenizer = Arc::new(BpeTokenizer::from_vocab(vec![b"hello".to_vec()]));
    let empty_session = Session::new(
        model,
        empty_tokenizer,
        ModalityCapabilities::text_only(),
        session_config(),
    )
    .unwrap();

    let (recovered_session, err) = core_chat(empty_session).unwrap_err();
    assert_eq!(err, ValidationError::UnsupportedProfile);
    assert!(recovered_session.is_usable());
    assert_eq!(recovered_session.position(), 0);
}

#[test]
fn chat_cancel_handle_mid_turn_cancels_and_recovers() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("long query")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let handle = chat.cancel_handle().expect("cancel handle must exist");
    // Pre-arm cancellation so decode terminates on first token check.
    handle.store(true, Ordering::Relaxed);

    let turn = chat.complete(&opts(0.0)).unwrap();
    assert_eq!(turn.summary.finish_reason, FinishReason::Cancelled);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    // Clear cancel and complete again.
    chat.clear_cancel();
    assert!(!handle.load(Ordering::Relaxed));
    let turn2 = chat.complete(&opts(0.0)).unwrap();
    assert_eq!(turn2.text, "a");
    assert_eq!(chat.phase(), SessionPhase::TurnComplete);
}

#[test]
fn chat_stream_text_emits_fragments_and_returns_complete_turn() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("stream me")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let mut emitted = Vec::new();
    let turn = chat
        .stream_text(&opts(0.0), |delta| {
            emitted.push(delta.to_string());
        })
        .unwrap();

    assert_eq!(turn.text, "a");
    assert_eq!(turn.summary.finish_reason, FinishReason::Stop);
    assert_eq!(chat.phase(), SessionPhase::TurnComplete);
    assert_eq!(emitted.concat(), "a");
}

struct JsonScriptedModel {
    config: ModelConfig,
    tokens: Vec<u32>,
    step: AtomicUsize,
}

impl Model for JsonScriptedModel {
    fn is_classifier(&self) -> bool {
        false
    }
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[0]
        else {
            unreachable!()
        };
        key_cache.extend(
            tokens
                .iter()
                .enumerate()
                .flat_map(|(i, &token)| [token as f32, (pos + i) as f32]),
        );
        value_cache.extend(tokens.iter().flat_map(|&token| [token as f32, 1.0]));
        state.seq_len += tokens.len();

        let mut logits = vec![-100.0; self.config.vocab_size];
        let idx = self.step.fetch_add(1, Ordering::Relaxed);
        let next = if idx < self.tokens.len() {
            self.tokens[idx]
        } else {
            7 // EOS
        };
        logits[next as usize] = 10.0;
        logits
    }
    fn forward_prefill_chunked(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
        _: usize,
        _cancel: &AtomicBool,
    ) -> (usize, Option<Vec<f32>>) {
        let mut last = None;
        for (i, &token) in tokens.iter().enumerate() {
            self.step.store(0, Ordering::Relaxed);
            last = Some(self.forward(&[token], pos + i, state));
        }
        (tokens.len(), last)
    }
}

#[test]
fn chat_complete_json_compiles_schema_and_enforces_grammar() {
    let tokenizer = fixtures::tokenizer();
    // In fixtures::tokenizer():
    // token 11 = '"'
    // token 74 = 'a'
    // token 7 = '<|im_end|>' (EOS)
    let scripted = Arc::new(JsonScriptedModel {
        config: fixtures::model_config("json-scripted-test", tokenizer.vocab_size()),
        tokens: vec![11, 74, 11],
        step: AtomicUsize::new(0),
    });
    let session = Session::new(
        scripted,
        tokenizer,
        ModalityCapabilities::text_only(),
        session_config(),
    )
    .unwrap();
    let mut chat = core_chat(session).unwrap();

    chat.ingest(&user("give json")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let schema = r#"{"type": "string", "enum": ["a"]}"#;
    let turn = chat.complete_json(&opts(0.0), schema).unwrap();

    assert_eq!(turn.text, "\"a\"");
    assert_eq!(turn.tokens, vec![11, 74, 11]);
    assert_eq!(turn.summary.finish_reason, FinishReason::Stop);
    assert_eq!(chat.phase(), SessionPhase::TurnComplete);
}

#[test]
fn chat_complete_json_rejects_invalid_schema() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    chat.ingest(&user("give json")).unwrap();

    let invalid_schema = r#"{"type": "unsupported_xyz"}"#;
    let err = chat.complete_json(&opts(0.0), invalid_schema).unwrap_err();
    match err {
        CompleteError::Validation(ValidationError::Generation(msg)) => {
            assert!(msg.contains("invalid JSON schema"));
        }
        other => panic!("expected Validation(Generation), got: {other:?}"),
    }
}

#[test]
fn chat_tool_registration_and_accessors() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    assert!(chat.tools().is_empty());
    assert_eq!(chat.tool_format(), ToolFormat::Lfm2Pythonic);

    let tool = ToolDef {
        name: "calculator".into(),
        description: Some("Perform math calculations".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "expr": { "type": "string" }
            },
            "required": ["expr"]
        }),
    };

    chat.set_tools(vec![tool.clone()]);
    assert_eq!(chat.tools().len(), 1);
    assert_eq!(chat.tools()[0].name, "calculator");
    assert_eq!(
        chat.tools()[0].description.as_deref(),
        Some("Perform math calculations")
    );

    chat.set_tool_format(ToolFormat::Hermes);
    assert_eq!(chat.tool_format(), ToolFormat::Hermes);
}

#[test]
fn chat_initial_render_with_tools_injects_tool_definitions() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    let tool = ToolDef {
        name: "get_weather".into(),
        description: Some("Get weather for a city".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            }
        }),
    };
    chat.set_tools(vec![tool]);

    let summary = chat.ingest(&user("What is the weather in Paris?")).unwrap();
    assert!(summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
}

#[test]
fn chat_tool_execution_loop_parses_call_and_ingests_response() {
    let tokenizer = fixtures::tokenizer();
    let tool_call_text = "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>";
    let call_tokens = tokenizer.encode(tool_call_text);

    let scripted = Arc::new(JsonScriptedModel {
        config: fixtures::model_config("tool-scripted-test", tokenizer.vocab_size()),
        tokens: call_tokens,
        step: AtomicUsize::new(0),
    });
    let session = Session::new(
        scripted,
        tokenizer,
        ModalityCapabilities::text_only(),
        session_config(),
    )
    .unwrap();
    let mut chat = core_chat(session).unwrap();

    let tool = ToolDef {
        name: "get_weather".into(),
        description: Some("Get weather for city".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            }
        }),
    };
    chat.set_tools(vec![tool]);

    chat.ingest(&user("weather in Paris?")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let mut generate_opts = opts(0.0);
    generate_opts.max_tokens = 128;
    let turn = chat.complete(&generate_opts).unwrap();
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].name, "get_weather");
    assert_eq!(
        turn.tool_calls[0].arguments,
        serde_json::json!({ "city": "Paris" })
    );
    assert_eq!(chat.phase(), SessionPhase::TurnComplete);

    let ingest_summary = chat
        .ingest_tool_response("get_weather", "{\"temperature\": 20}")
        .unwrap();
    assert!(ingest_summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);
}

#[test]
fn chat_tool_validation_rules() {
    let (_model, mut chat) = setup(fixtures::tokenizer());

    // Without tools, Role::Tool is an UnsupportedRole:
    let err = chat
        .replace_messages(&[Message::tool("tool response")])
        .unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(val, ValidationError::UnsupportedRole { message: 0 });
        }
        other => panic!("expected validation error, got {other:?}"),
    }

    // With tools registered, Role::Tool at the start of initial messages violates role order:
    chat.set_tools(vec![ToolDef {
        name: "test".into(),
        description: None,
        parameters: serde_json::json!({}),
    }]);
    let err = chat
        .replace_messages(&[Message::tool("tool response")])
        .unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(val, ValidationError::RoleOrder { message: 0 });
        }
        other => panic!("expected validation error, got {other:?}"),
    }
}

#[derive(Debug, Default)]
struct MockMultimodalExecution {
    tokens: Vec<u32>,
    images: Vec<Vec<u8>>,
    audio: Vec<(Vec<f32>, u32)>,
    image_capable: bool,
    audio_capable: bool,
}

impl Execution for MockMultimodalExecution {
    fn position(&self) -> usize {
        self.tokens.len()
    }
    fn capacity(&self) -> usize {
        4096
    }
    fn audio_output(&self) -> bool {
        false
    }
    fn image_input(&self) -> bool {
        self.image_capable
    }
    fn audio_input(&self) -> bool {
        self.audio_capable
    }
    fn validate_decode(&self, _: &GenerateOpts) -> Result<(), ValidationError> {
        Ok(())
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError> {
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }
    fn append_segments(&mut self, segments: &[IngestSegment<'_>]) -> Result<(), IngestError> {
        for seg in segments {
            match *seg {
                IngestSegment::Tokens(toks) => self.tokens.extend_from_slice(toks),
                IngestSegment::Image(bytes) => {
                    self.images.push(bytes.to_vec());
                    self.tokens.push(9999);
                }
                IngestSegment::Audio { pcm, sample_rate } => {
                    self.audio.push((pcm.to_vec(), sample_rate));
                    self.tokens.push(8888);
                }
            }
        }
        Ok(())
    }
    fn reset(&mut self, _explicit: bool) -> Result<(), CeraError> {
        self.tokens.clear();
        self.images.clear();
        self.audio.clear();
        Ok(())
    }
    fn decode(&mut self, _: &GenerateOpts, _: &mut dyn ModalitySink) -> DecodeReport {
        DecodeReport {
            result: Ok(GenerateSummary {
                tokens_generated: 1,
                prompt_eval_tokens: 0,
                prompt_eval_ms: 0,
                decode_ms: 0,
                finish_reason: FinishReason::Stop,
            }),
            state: DecodeState::Terminal {
                token: 7,
                committed: true,
            },
        }
    }
}

#[test]
fn chat_multimodal_image_rejected_on_text_only_session() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    let err = chat.ingest(&Message::image(vec![1, 2, 3, 4])).unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(
                val,
                ValidationError::UnsupportedContent {
                    message: 0,
                    part: 0,
                }
            );
        }
        other => panic!("expected validation error, got {other:?}"),
    }
    assert_eq!(chat.phase(), SessionPhase::Idle);
}

#[test]
fn chat_multimodal_audio_rejected_on_text_only_session() {
    let (_model, mut chat) = setup(fixtures::tokenizer());
    let err = chat
        .ingest(&Message::audio(vec![0.1, 0.2, 0.3], 16000))
        .unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(
                val,
                ValidationError::UnsupportedContent {
                    message: 0,
                    part: 0,
                }
            );
        }
        other => panic!("expected validation error, got {other:?}"),
    }
    assert_eq!(chat.phase(), SessionPhase::Idle);
}

#[test]
fn chat_multimodal_convenience_constructors_and_profile_accessors() {
    let text_part = ContentPart::text("hello");
    assert_eq!(text_part, ContentPart::Text("hello".into()));
    let img_part = ContentPart::image(vec![1, 2, 3]);
    assert_eq!(img_part, ContentPart::Image(vec![1, 2, 3]));
    let aud_part = ContentPart::audio(vec![0.5], 16000);
    assert_eq!(
        aud_part,
        ContentPart::Audio {
            pcm: vec![0.5],
            sample_rate: 16000,
        }
    );

    let img_msg = Message::image(vec![10, 20]);
    assert_eq!(img_msg.role, Role::User);
    assert_eq!(img_msg.content, vec![ContentPart::Image(vec![10, 20])]);

    let user_img = Message::user_with_image("look", vec![30, 40]);
    assert_eq!(user_img.role, Role::User);
    assert_eq!(
        user_img.content,
        vec![
            ContentPart::Image(vec![30, 40]),
            ContentPart::Text("look".into()),
        ]
    );

    let aud_msg = Message::audio(vec![0.25], 16000);
    assert_eq!(aud_msg.role, Role::User);

    let user_aud = Message::user_with_audio("listen", vec![0.75], 16000);
    assert_eq!(user_aud.role, Role::User);

    let profile = Profile::discover(fixtures::tokenizer()).unwrap();
    assert!(profile.image_marker().is_some());
    assert!(profile.image_start().is_some());
    assert!(profile.image_end().is_some());
    assert!(profile.audio_marker().is_some());
}

#[test]
fn chat_multimodal_image_ingestion_with_image_capable_execution() {
    let profile = Profile::discover(fixtures::tokenizer()).unwrap();
    let exec = MockMultimodalExecution {
        image_capable: true,
        ..Default::default()
    };
    let mut chat = Chat::new(exec, profile, 0).ok().unwrap();

    let image_payload = vec![1, 2, 3, 4, 5];
    let summary = chat
        .ingest(&Message::user_with_image(
            "Describe this image",
            image_payload.clone(),
        ))
        .unwrap();
    assert!(summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let inner = chat.into_inner();
    assert_eq!(inner.images.len(), 1);
    assert_eq!(inner.images[0], image_payload);
    assert!(inner.tokens.contains(&2));
    assert!(inner.tokens.contains(&3));
    assert!(inner.tokens.contains(&9999));
}

#[test]
fn chat_multimodal_audio_ingestion_with_audio_capable_execution() {
    let profile = Profile::discover(fixtures::tokenizer()).unwrap();
    let exec = MockMultimodalExecution {
        audio_capable: true,
        ..Default::default()
    };
    let mut chat = Chat::new(exec, profile, 0).ok().unwrap();

    let pcm = vec![0.1, 0.2, 0.3, 0.4];
    let summary = chat
        .ingest(&Message::user_with_audio(
            "Transcribe speech",
            pcm.clone(),
            16000,
        ))
        .unwrap();
    assert!(summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let inner = chat.into_inner();
    assert_eq!(inner.audio.len(), 1);
    assert_eq!(inner.audio[0], (pcm, 16000));
    assert!(inner.tokens.contains(&8888));
}

#[test]
fn chat_multimodal_validation_rules() {
    let profile = Profile::discover(fixtures::tokenizer()).unwrap();
    let exec = MockMultimodalExecution {
        image_capable: true,
        audio_capable: true,
        ..Default::default()
    };
    let mut chat = Chat::new(exec, profile, 0).ok().unwrap();

    // Image in System role is rejected:
    let err = chat
        .replace_messages(&[Message::with_parts(
            Role::System,
            vec![ContentPart::Image(vec![1, 2])],
        )])
        .unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(
                val,
                ValidationError::UnsupportedContent {
                    message: 0,
                    part: 0,
                }
            );
        }
        other => panic!("expected validation error, got {other:?}"),
    }

    // Audio in Assistant role is rejected:
    let err = chat
        .replace_messages(&[
            Message::user("hi"),
            Message::with_parts(
                Role::Assistant,
                vec![ContentPart::Audio {
                    pcm: vec![0.1],
                    sample_rate: 16000,
                }],
            ),
        ])
        .unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(
                val,
                ValidationError::UnsupportedContent {
                    message: 1,
                    part: 0,
                }
            );
        }
        other => panic!("expected validation error, got {other:?}"),
    }

    // Mixing image and audio in the same turn is rejected:
    let err = chat
        .replace_messages(&[Message::with_parts(
            Role::User,
            vec![
                ContentPart::Image(vec![1, 2]),
                ContentPart::Audio {
                    pcm: vec![0.1],
                    sample_rate: 16000,
                },
            ],
        )])
        .unwrap_err();
    match err.cause {
        IngestCause::Validation(val) => {
            assert_eq!(
                val,
                ValidationError::UnsupportedContent {
                    message: 0,
                    part: 1,
                }
            );
        }
        other => panic!("expected validation error, got {other:?}"),
    }
}

#[test]
fn chat_checkpoint_roundtrip_persistence_and_continuation() {
    let tok = fixtures::tokenizer();
    let (model, mut chat) = setup(tok.clone());

    chat.ingest(&user("hello")).unwrap();
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let result = chat.complete(&opts(0.0)).unwrap();
    assert_eq!(chat.phase(), SessionPhase::TurnComplete);
    assert_eq!(result.text, "a");
    let pos_before_checkpoint = chat.position();

    let tools = vec![ToolDef {
        name: "calculator".to_string(),
        description: Some("Evaluates arithmetic expressions".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "expr": { "type": "string" }
            },
            "required": ["expr"]
        }),
    }];
    chat.set_tools(tools.clone());
    chat.set_tool_format(ToolFormat::Hermes);

    let cp = chat.checkpoint().unwrap();
    assert_eq!(cp.phase, SessionPhase::TurnComplete);
    assert_eq!(cp.tools.len(), 1);
    assert_eq!(cp.tool_format, ToolFormat::Hermes);
    assert_eq!(cp.session_checkpoint.position, pos_before_checkpoint);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_chat.chk");
    chat.save_checkpoint(&path).unwrap();

    let session2 = Session::new(
        model,
        tok,
        ModalityCapabilities::text_only(),
        session_config(),
    )
    .unwrap();
    let mut chat2 = core_chat(session2).unwrap();
    assert_eq!(chat2.phase(), SessionPhase::Idle);
    assert!(chat2.tools().is_empty());

    chat2.load_checkpoint(&path).unwrap();
    assert_eq!(chat2.phase(), SessionPhase::TurnComplete);
    assert_eq!(chat2.tools(), tools.as_slice());
    assert_eq!(chat2.tool_format(), ToolFormat::Hermes);
    assert_eq!(chat2.position(), pos_before_checkpoint);

    // Clear tools for normal dialogue continuation:
    chat2.set_tools(Vec::new());
    chat2.ingest(&user("follow up")).unwrap();
    assert_eq!(chat2.phase(), SessionPhase::PromptReady);
    let result2 = chat2.complete(&opts(0.0)).unwrap();
    assert_eq!(chat2.phase(), SessionPhase::TurnComplete);
    assert_eq!(result2.text, "a");
    assert!(chat2.position() > pos_before_checkpoint);
}

#[test]
fn chat_checkpoint_rejects_incompatible_model_architecture() {
    let tok = fixtures::tokenizer();
    let (_model, mut chat) = setup(tok.clone());

    chat.ingest(&user("hello world")).unwrap();
    chat.complete(&opts(0.0)).unwrap();

    let cp = chat.checkpoint().unwrap();

    let mut cfg2 = fixtures::model_config("chat-transaction-test", tok.vocab_size());
    cfg2.hidden_size = 4;
    cfg2.head_dim = 4;
    let model2 = Arc::new(StateModel {
        config: cfg2,
        answer: tok.encode("a")[0],
        calls: Mutex::new(Vec::new()),
        rewinds: AtomicUsize::new(0),
        resets: AtomicUsize::new(0),
        rewind_supported: AtomicBool::new(true),
        fault: AtomicU8::new(NO_FAULT),
    });
    let session2 = Session::new(
        model2,
        tok,
        ModalityCapabilities::text_only(),
        session_config(),
    )
    .unwrap();
    let mut chat2 = core_chat(session2).unwrap();

    let err = chat2.restore(&cp).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(msg.contains("model fingerprint mismatch"));
        }
        other => panic!("expected format error for fingerprint mismatch, got {other:?}"),
    }
}

#[test]
fn chat_checkpoint_rejects_unusable_phase() {
    let tok = fixtures::tokenizer();
    let (_model, mut chat) = setup(tok.clone());

    chat.ingest(&user("test unusable checkpoint rejection"))
        .unwrap();
    let mut cp = chat.checkpoint().unwrap();
    cp.phase = SessionPhase::Unusable;

    let (_model2, mut chat2) = setup(tok);
    let err = chat2.restore(&cp).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(msg.contains("cannot restore chat checkpoint in Unusable phase"));
        }
        other => panic!("expected format error for unusable phase, got {other:?}"),
    }
}

#[test]
fn chat_checkpoint_rejects_mismatched_position_and_seq_len() {
    let tok = fixtures::tokenizer();
    let (_model, mut chat) = setup(tok.clone());

    chat.ingest(&user("test position sync")).unwrap();
    let mut cp = chat.checkpoint().unwrap();
    cp.session_checkpoint.position += 1;

    let (_model2, mut chat2) = setup(tok);
    let err = chat2.restore(&cp).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(msg.contains("does not match KV state sequence length"));
        }
        other => panic!("expected format error for position mismatch, got {other:?}"),
    }
}

#[test]
fn chat_checkpoint_rejects_layer_variant_mismatch_without_panic() {
    let tok = fixtures::tokenizer();
    let (_model, mut chat) = setup(tok.clone());

    chat.ingest(&user("test layer mismatch")).unwrap();
    let mut cp = chat.checkpoint().unwrap();
    // Replace layer 0 with a Conv snapshot when model expects Attention
    if let Some(first_layer) = cp.session_checkpoint.kv_state.layers.first_mut() {
        *first_layer = crate::kv_cache::LayerSnapshot::Conv {
            buffer: vec![0, 0, 0, 0],
        };
    }

    let (_model2, mut chat2) = setup(tok);
    let err = chat2.restore(&cp).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(msg.contains("snapshot layer kind does not match") || msg.contains("variant"));
        }
        other => panic!("expected format error for layer kind mismatch, got {other:?}"),
    }
}

#[test]
fn profile_discovery_for_llama3_template_and_framing() {
    let tok = BpeTokenizer::llama3_for_test();
    let profile = Profile::discover(Arc::new(tok)).unwrap();
    assert_eq!(profile.family(), TemplateFamily::Llama3);
    assert_eq!(profile.turn_end(), "<|eot_id|>");
    assert_eq!(profile.eos(), 128009);

    let messages = [Message::user("Hello Llama")];
    let rendered_initial = profile.render(&messages, true, &[]).unwrap();
    assert!(
        rendered_initial
            .contains("<|start_header_id|>user<|end_header_id|>\n\nHello Llama<|eot_id|>")
    );
    assert!(rendered_initial.contains("<|start_header_id|>assistant<|end_header_id|>\n\n"));

    let rendered_cont = profile.render(&messages, false, &[]).unwrap();
    assert!(!rendered_cont.starts_with("<|begin_of_text|>"));
    assert!(
        rendered_cont.contains("<|start_header_id|>user<|end_header_id|>\n\nHello Llama<|eot_id|>")
    );
}

#[test]
fn profile_discovery_for_gemma_template_and_framing() {
    let tok = BpeTokenizer::gemma_for_test();
    let profile = Profile::discover(Arc::new(tok)).unwrap();
    assert_eq!(profile.family(), TemplateFamily::Gemma);
    assert_eq!(profile.turn_end(), "<end_of_turn>");
    assert_eq!(profile.eos(), 107);

    let messages = [Message::user("Hello Gemma")];
    let rendered_initial = profile.render(&messages, true, &[]).unwrap();
    assert!(rendered_initial.contains("<start_of_turn>user\nHello Gemma<end_of_turn>\n"));
    assert!(rendered_initial.contains("<start_of_turn>model\n"));

    let rendered_cont = profile.render(&messages, false, &[]).unwrap();
    assert!(!rendered_cont.starts_with("<bos>"));
    assert!(rendered_cont.contains("<start_of_turn>user\nHello Gemma<end_of_turn>\n"));
}

#[test]
fn profile_custom_builder_and_continuation() {
    let tok = Arc::new(BpeTokenizer::with_custom_template_for_test(
        TEMPLATE,
        Some(7),
    ));

    let profile = Profile::builder(tok)
        .family(TemplateFamily::Custom)
        .turn_prefix(">>>")
        .turn_end("<<<")
        .eos(99)
        .build()
        .unwrap();

    assert_eq!(profile.family(), TemplateFamily::Custom);
    assert_eq!(profile.turn_prefix(), ">>>");
    assert_eq!(profile.turn_end(), "<<<");
    assert_eq!(profile.eos(), 99);
}

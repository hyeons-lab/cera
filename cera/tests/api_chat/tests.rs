use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::core_api::CeraError;
use super::core_api::session::{
    FinishReason, GenerateOpts, GenerateSummary, ModalitySink, RecoveryOutcome,
};
use super::core_api::tokenizer::{BpeTokenizer, ChatMessage, UserMessage, apply_chat_template};

use super::contract::*;
use super::fixtures::{self, Sink, TraceModel};

#[derive(Debug)]
struct TraceExecution {
    tokens: Vec<u32>,
    capacity: usize,
    resets: Vec<bool>,
    append_calls: usize,
    decode_calls: usize,
    failure: Option<RecoveryOutcome>,
    reset_fails: bool,
    panic_append: bool,
    audio: bool,
    invalid_decode: bool,
    cancel: Arc<AtomicBool>,
    decode: VecDeque<DecodeReport>,
    output: VecDeque<Vec<u32>>,
}

impl Default for TraceExecution {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            capacity: 8192,
            resets: Vec::new(),
            append_calls: 0,
            decode_calls: 0,
            failure: None,
            reset_fails: false,
            panic_append: false,
            audio: false,
            invalid_decode: false,
            cancel: Arc::new(AtomicBool::new(false)),
            decode: VecDeque::new(),
            output: VecDeque::new(),
        }
    }
}

impl Execution for TraceExecution {
    fn position(&self) -> usize {
        self.tokens.len()
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn audio_output(&self) -> bool {
        self.audio
    }
    fn validate_decode(&self, _: &GenerateOpts) -> Result<(), ValidationError> {
        if self.invalid_decode {
            Err(ValidationError::Generation(
                "injected invalid grammar".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError> {
        self.append_calls += 1;
        assert!(!self.panic_append, "injected append unwind");
        if let Some(recovery) = self.failure.take() {
            if recovery == RecoveryOutcome::Reset {
                self.tokens.clear();
            }
            return Err(IngestError {
                cause: IngestCause::Execution(CeraError::Cancelled),
                recovery,
                rewind_error: None,
                recovery_error: (recovery == RecoveryOutcome::Unusable)
                    .then(|| CeraError::Backend("reset failed".into())),
            });
        }
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }
    fn reset(&mut self, explicit: bool) -> Result<(), CeraError> {
        self.resets.push(explicit);
        if self.reset_fails {
            return Err(CeraError::Backend("reset failed".into()));
        }
        self.tokens.clear();
        Ok(())
    }
    fn decode(&mut self, _: &GenerateOpts, sink: &mut dyn ModalitySink) -> DecodeReport {
        self.decode_calls += 1;
        let mut report = self
            .decode
            .pop_front()
            .expect("scripted decode observation");
        if let Some(tokens) = self.output.pop_front() {
            self.tokens.extend_from_slice(&tokens);
            sink.on_text_tokens(&tokens);
            if let Ok(summary) = &mut report.result {
                summary.tokens_generated = tokens.len() as u32;
            }
        }
        if let DecodeState::Terminal {
            token,
            committed: true,
        } = report.state
        {
            self.tokens.push(token);
        }
        if let Ok(summary) = &report.result {
            sink.on_done(summary.finish_reason.clone());
        }
        report
    }
    fn cancel_handle(&self) -> Option<Arc<AtomicBool>> {
        Some(Arc::clone(&self.cancel))
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    fn clear_cancel(&mut self) {
        self.cancel.store(false, Ordering::Relaxed);
    }
}

fn report(state: DecodeState, finish: FinishReason) -> DecodeReport {
    DecodeReport {
        state,
        result: Ok(GenerateSummary {
            tokens_generated: 0,
            prompt_eval_tokens: 11,
            prompt_eval_ms: 13,
            decode_ms: 17,
            finish_reason: finish,
        }),
    }
}

fn chat(execution: TraceExecution) -> Chat<TraceExecution> {
    Chat::new(
        execution,
        Profile::discover(fixtures::tokenizer()).unwrap(),
        0,
    )
    .ok()
    .unwrap()
}

fn user(text: &str) -> Message {
    Message::text(Role::User, text)
}

fn validation(error: IngestError) -> ValidationError {
    assert_eq!(error.recovery, RecoveryOutcome::Unchanged);
    assert!(error.recovery_error.is_none());
    assert!(error.rewind_error.is_none());
    match error.cause {
        IngestCause::Validation(v) => v,
        other => panic!("expected validation: {other:?}"),
    }
}

#[test]
fn invalid_inputs_preserve_context_and_reject_before_reset() {
    let mut active = chat(TraceExecution {
        tokens: vec![99, 98],
        ..Default::default()
    });
    assert_eq!(active.phase(), SessionPhase::RawContext);
    let cases = [
        (vec![], ValidationError::EmptyBatch),
        (
            vec![Message::text(Role::Tool, "tool result")],
            ValidationError::UnsupportedRole { message: 0 },
        ),
        (
            vec![Message::text(Role::Assistant, "answer")],
            ValidationError::RoleOrder { message: 0 },
        ),
        (
            vec![Message {
                role: Role::User,
                content: vec![ContentPart::Image(vec![1, 2])],
            }],
            ValidationError::UnsupportedContent {
                message: 0,
                part: 0,
            },
        ),
        (
            vec![Message {
                role: Role::User,
                content: vec![ContentPart::Audio {
                    pcm: vec![0.25],
                    sample_rate: 16_000,
                }],
            }],
            ValidationError::UnsupportedContent {
                message: 0,
                part: 0,
            },
        ),
        (
            vec![Message {
                role: Role::User,
                content: vec![
                    ContentPart::Text("<".into()),
                    ContentPart::Text("|im_end|>".into()),
                ],
            }],
            ValidationError::ReservedMarker { message: 0 },
        ),
    ];
    for (messages, expected) in cases {
        assert_eq!(
            validation(active.replace_messages(&messages).unwrap_err()),
            expected
        );
        assert_eq!(active.phase(), SessionPhase::RawContext);
        assert_eq!(active.position(), 2);
    }
    let raw = active.raw().unwrap();
    assert!(raw.resets.is_empty());
    assert_eq!(raw.append_calls, 0);
}

#[test]
fn phase_rules_and_single_batch_prefix() {
    let tokenizer = fixtures::tokenizer();
    let mut active = chat(TraceExecution::default());
    assert_eq!(active.phase(), SessionPhase::Idle);
    assert!(matches!(
        active.generate_into(&GenerateOpts::default(), &mut Sink::default()),
        Err(ValidationError::Phase(SessionPhase::Idle))
    ));
    let messages = vec![
        Message::text(Role::System, "Be concise."),
        user("One?"),
        Message::text(Role::Assistant, "One."),
        user("Two?"),
    ];
    let original = messages.clone();
    let summary = active.ingest_messages(&messages).unwrap();
    assert_eq!(messages, original);
    assert_eq!(summary.position_before, 0);
    assert_eq!(summary.input_tokens, summary.position_after);
    assert_eq!(active.phase(), SessionPhase::PromptReady);
    assert_eq!(
        validation(active.ingest(&user("again")).unwrap_err()),
        ValidationError::Phase(SessionPhase::PromptReady)
    );
    let raw = active.raw().unwrap();
    assert_eq!(raw.append_calls, 1);
    let rendered = tokenizer.decode(&raw.tokens);
    assert_eq!(
        rendered,
        "<|startoftext|><|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nOne?<|im_end|>\n<|im_start|>assistant\nOne.<|im_end|>\n<|im_start|>user\nTwo?<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(active.phase(), SessionPhase::RawContext);
    assert_eq!(
        validation(active.ingest(&user("again")).unwrap_err()),
        ValidationError::Phase(SessionPhase::RawContext)
    );
    active.replace_messages(&messages).unwrap();
    assert_eq!(active.phase(), SessionPhase::PromptReady);
}

#[test]
fn stop_reason_does_not_prove_a_turn_boundary() {
    for (state, finish, expected) in [
        (
            DecodeState::NoProgress,
            FinishReason::MaxTokens,
            SessionPhase::PromptReady,
        ),
        (
            DecodeState::NoProgress,
            FinishReason::Cancelled,
            SessionPhase::PromptReady,
        ),
        (
            DecodeState::Terminal {
                token: 7,
                committed: false,
            },
            FinishReason::Stop,
            SessionPhase::TurnComplete,
        ),
        (
            DecodeState::Terminal {
                token: 8,
                committed: false,
            },
            FinishReason::Stop,
            SessionPhase::Interrupted,
        ),
        (
            DecodeState::Interrupted,
            FinishReason::Stop,
            SessionPhase::Interrupted,
        ),
        (
            DecodeState::Interrupted,
            FinishReason::MaxTokens,
            SessionPhase::Interrupted,
        ),
        (
            DecodeState::Interrupted,
            FinishReason::Cancelled,
            SessionPhase::Interrupted,
        ),
        (
            DecodeState::Unusable,
            FinishReason::Stop,
            SessionPhase::Unusable,
        ),
    ] {
        for temperature in [0.0, 0.7] {
            let mut active = chat(TraceExecution {
                decode: VecDeque::from([report(state, finish.clone())]),
                ..Default::default()
            });
            active.ingest(&user("Hello")).unwrap();
            let mut sink = Sink::default();
            let result = active
                .generate_into(
                    &GenerateOpts {
                        temperature,
                        ..Default::default()
                    },
                    &mut sink,
                )
                .unwrap();
            assert_eq!(result.result.unwrap().finish_reason, finish);
            assert_eq!(sink.done, std::slice::from_ref(&finish));
            assert_eq!(active.phase(), expected);
            if expected != SessionPhase::PromptReady {
                assert!(
                    matches!(active.generate_into(&GenerateOpts::default(), &mut sink), Err(ValidationError::Phase(p)) if p == expected)
                );
            }
        }
    }
}

#[test]
fn pending_boundary_survives_whole_batch_restoration() {
    for committed in [false, true] {
        for recovery in [RecoveryOutcome::Unchanged, RecoveryOutcome::Restored] {
            let mut active = chat(TraceExecution {
                decode: VecDeque::from([report(
                    DecodeState::Terminal {
                        token: 7,
                        committed,
                    },
                    FinishReason::Stop,
                )]),
                ..Default::default()
            });
            active.ingest(&user("one")).unwrap();
            active
                .generate_into(&GenerateOpts::default(), &mut Sink::default())
                .unwrap();
            // Schedule failure without going through raw escape: this is fault
            // injection into the test bridge, not an application operation.
            active.set_failure(recovery);
            let before = active.position();
            let error = active.ingest(&user("two")).unwrap_err();
            assert_eq!(error.recovery, recovery);
            assert!(matches!(
                error.cause,
                IngestCause::Execution(CeraError::Cancelled)
            ));
            assert_eq!(active.phase(), SessionPhase::TurnComplete);
            assert_eq!(active.position(), before);
            active.ingest(&user("two")).unwrap();
            let text = fixtures::tokenizer().decode(&active.raw().unwrap().tokens);
            assert!(
                text.contains("assistant\n<|im_end|>\n<|im_start|>user\ntwo"),
                "{text}"
            );
            assert_eq!(text.matches("<|startoftext|>").count(), 1);
        }
    }
}

#[test]
fn replacement_never_reports_restoring_the_discarded_context() {
    for recovery in [
        RecoveryOutcome::Unchanged,
        RecoveryOutcome::Restored,
        RecoveryOutcome::Reset,
        RecoveryOutcome::Unusable,
    ] {
        let mut active = chat(TraceExecution {
            tokens: vec![99],
            failure: Some(recovery),
            ..Default::default()
        });
        let error = active.replace_messages(&[user("new")]).unwrap_err();
        if recovery == RecoveryOutcome::Unusable {
            assert_eq!(error.recovery, RecoveryOutcome::Unusable);
            assert!(matches!(error.recovery_error, Some(CeraError::Backend(_))));
            assert_eq!(active.phase(), SessionPhase::Unusable);
            assert!(active.raw().is_err());
            assert!(active.replace_messages(&[user("retry")]).is_err());
            active.reset().unwrap();
        } else {
            assert_eq!(error.recovery, RecoveryOutcome::Reset);
        }
        assert_eq!(active.phase(), SessionPhase::Idle);
        assert_eq!(active.position(), 0);
        let raw = active.raw().unwrap();
        assert!(!raw.resets[0]);
        if recovery == RecoveryOutcome::Unusable {
            assert_eq!(raw.resets, [false, true]);
        }
    }
}

#[test]
fn reset_failure_unwind_and_capacity_are_explicit() {
    let mut active = chat(TraceExecution {
        tokens: vec![1, 2],
        capacity: 3,
        ..Default::default()
    });
    assert!(matches!(
        validation(active.replace_messages(&[user("too large")]).unwrap_err()),
        ValidationError::Capacity { .. }
    ));
    assert_eq!(active.position(), 2);
    assert!(active.raw().unwrap().resets.is_empty());
    let mut active = chat(TraceExecution {
        reset_fails: true,
        ..Default::default()
    });
    assert_eq!(
        active
            .replace_messages(&[user("new")])
            .unwrap_err()
            .recovery,
        RecoveryOutcome::Unusable
    );
    assert!(active.reset().is_err());
    assert_eq!(active.phase(), SessionPhase::Unusable);
    let mut active = chat(TraceExecution {
        panic_append: true,
        ..Default::default()
    });
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| active.ingest(&user("new"))))
            .is_err()
    );
    assert_eq!(active.phase(), SessionPhase::Unusable);
}

#[test]
fn unsupported_profile_and_sliding_rejected() {
    assert!(matches!(
        Profile::discover(Arc::new(BpeTokenizer::from_vocab(vec![b"a".to_vec()]))),
        Err(ValidationError::UnsupportedProfile)
    ));
    let (exec, err) = Chat::new(
        TraceExecution::default(),
        Profile::discover(fixtures::tokenizer()).unwrap(),
        1,
    )
    .err()
    .unwrap();
    assert_eq!(err, ValidationError::SlidingContext);
    assert_eq!(exec.position(), 0);
}

#[test]
fn collection_shares_decode_and_rejects_audio_and_invalid_options() {
    let (exec, err) = Chat::new(
        TraceExecution {
            audio: true,
            ..Default::default()
        },
        Profile::discover(fixtures::tokenizer()).unwrap(),
        0,
    )
    .err()
    .unwrap();
    assert_eq!(err, ValidationError::AudioOutput);
    assert!(exec.audio);
    let mut active = chat(TraceExecution {
        invalid_decode: true,
        ..Default::default()
    });
    active.ingest(&user("hello")).unwrap();
    assert!(matches!(
        active.complete(&GenerateOpts::default()),
        Err(CompleteError::Validation(ValidationError::Generation(_)))
    ));
    assert_eq!(active.phase(), SessionPhase::PromptReady);
    assert_eq!(active.execution_for_test().decode_calls, 0);
    let mut active = chat(TraceExecution {
        decode: VecDeque::from([report(
            DecodeState::Terminal {
                token: 7,
                committed: false,
            },
            FinishReason::Stop,
        )]),
        output: VecDeque::from([fixtures::tokenizer().encode("answer")]),
        ..Default::default()
    });
    active.ingest(&user("hello")).unwrap();
    let result = active.complete(&GenerateOpts::default()).unwrap();
    assert_eq!(result.text, "answer");
    assert_eq!(result.tokens, fixtures::tokenizer().encode("answer"));
    assert_eq!(
        result.summary.tokens_generated as usize,
        result.tokens.len()
    );
    assert_eq!(result.summary.finish_reason, FinishReason::Stop);
    assert_eq!(result.summary.prompt_eval_tokens, 11);
    assert_eq!(result.summary.prompt_eval_ms, 13);
    assert_eq!(result.summary.decode_ms, 17);
    assert_eq!(active.phase(), SessionPhase::TurnComplete);
    assert_eq!(active.execution_for_test().decode_calls, 1);
    assert!(matches!(
        active.complete(&GenerateOpts::default()),
        Err(CompleteError::Validation(ValidationError::Phase(
            SessionPhase::TurnComplete
        )))
    ));
    let mut active = chat(TraceExecution {
        decode: VecDeque::from([DecodeReport {
            result: Err(CeraError::Backend("decode failed".into())),
            state: DecodeState::Unusable,
        }]),
        ..Default::default()
    });
    active.ingest(&user("hello")).unwrap();
    assert!(matches!(
        active.complete(&GenerateOpts::default()),
        Err(CompleteError::Execution(CeraError::Backend(_)))
    ));
    assert_eq!(active.phase(), SessionPhase::Unusable);
}

#[test]
fn raw_replacement_cannot_bypass_output_capabilities() {
    let mut active = chat(TraceExecution::default());
    active.ingest(&user("hello")).unwrap();
    *active.raw().unwrap() = TraceExecution {
        audio: true,
        tokens: vec![99],
        ..Default::default()
    };
    assert_eq!(
        validation(active.replace_messages(&[user("new")]).unwrap_err()),
        ValidationError::AudioOutput
    );
    assert_eq!(active.phase(), SessionPhase::RawContext);
    let raw = active.raw().unwrap();
    assert_eq!(raw.tokens, [99]);
    assert_eq!(raw.append_calls, 0);
    assert!(raw.resets.is_empty());
    raw.audio = false;
    active.replace_messages(&[user("new")]).unwrap();
    // Fault injection also checks capability drift inside an executor without
    // explicit raw escape: validation must still happen before decode.
    active.execution_for_test().audio = true;
    assert!(matches!(
        active.complete(&GenerateOpts::default()),
        Err(CompleteError::Validation(ValidationError::AudioOutput))
    ));
    assert_eq!(active.phase(), SessionPhase::PromptReady);
    assert_eq!(active.execution_for_test().decode_calls, 0);
    // Execution validity is the first error at both entry points, before the
    // audio capability, so a caller sees the same ordering from prepare/decode.
    active.execution_for_test().invalid_decode = true;
    assert!(matches!(
        active.complete(&GenerateOpts::default()),
        Err(CompleteError::Validation(ValidationError::Generation(_)))
    ));
    assert_eq!(active.execution_for_test().decode_calls, 0);
}

#[test]
fn no_progress_error_leaves_the_cursor_stale_not_the_execution_unusable() {
    for temperature in [0.0, 0.7] {
        let mut active = chat(TraceExecution {
            decode: VecDeque::from([DecodeReport {
                result: Err(CeraError::EmptyInput),
                state: DecodeState::NoProgress,
            }]),
            ..Default::default()
        });
        active.ingest(&user("hello")).unwrap();
        let position = active.position();
        assert!(matches!(
            active.complete(&GenerateOpts {
                temperature,
                ..Default::default()
            }),
            Err(CompleteError::Execution(CeraError::EmptyInput))
        ));
        // Nothing was mutated, so replacement and raw access stay open and the
        // stale prompt is not certified again.
        assert_eq!(active.phase(), SessionPhase::RawContext);
        assert_eq!(active.position(), position);
        assert!(matches!(
            active.ingest(&user("again")).unwrap_err().cause,
            IngestCause::Validation(ValidationError::Phase(SessionPhase::RawContext))
        ));
        assert!(active.raw().unwrap().resets.is_empty());
        active.replace_messages(&[user("again")]).unwrap();
        assert_eq!(active.raw().unwrap().resets, [false]);
    }
}

fn legacy_boundary(tokenizer: Arc<BpeTokenizer>) {
    for temperature in [0.0, 0.7] {
        let (model, mut session) = TraceModel::session(tokenizer.clone());
        session
            .append_user_message(&UserMessage {
                text: Some("Hello".into()),
                ..Default::default()
            })
            .unwrap();
        let prepared = model.resident.lock().unwrap().clone();
        let mut sink = Sink::default();
        let summary = session
            .generate(
                &GenerateOpts {
                    temperature,
                    max_tokens: 4,
                    ..Default::default()
                },
                &mut sink,
            )
            .unwrap();
        assert_eq!(summary.finish_reason, FinishReason::Stop);
        assert_eq!(sink.tokens, tokenizer.encode("a"));
        let resident = model.resident.lock().unwrap().clone();
        assert_eq!(resident, [prepared, sink.tokens.clone()].concat());
        assert_ne!(resident.last(), Some(&7));
        assert_eq!(session.position() as usize, resident.len());
        let next = UserMessage {
            text: Some("World".into()),
            ..Default::default()
        };
        session.append_user_message(&next).unwrap();
        let legacy = model.resident.lock().unwrap().clone();
        let history = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "a".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "World".into(),
            },
        ];
        let canonical = tokenizer.encode(&apply_chat_template(&tokenizer, &history, true).unwrap());
        assert_ne!(legacy, canonical);
        assert_eq!(legacy.iter().filter(|&&id| id == 1).count(), 2);
        let corrected = [
            resident,
            tokenizer
                .encode("<|im_end|>\n<|im_start|>user\nWorld<|im_end|>\n<|im_start|>assistant\n"),
        ]
        .concat();
        assert_eq!(corrected, canonical);
    }
}

#[test]
fn real_session_legacy_decode_omits_eos_and_repeats_bos() {
    legacy_boundary(fixtures::tokenizer());
}

#[test]
fn profile_discovery_and_turn_framing_for_lfm2_5_template() {
    let tokenizer = fixtures::lfm2_5_tokenizer();
    let profile = Profile::discover(tokenizer.clone()).unwrap();

    let execution = TraceExecution {
        decode: VecDeque::from([report(
            DecodeState::Terminal {
                token: 7,
                committed: false,
            },
            FinishReason::Stop,
        )]),
        ..Default::default()
    };
    let mut chat = Chat::new(execution, profile, 0).unwrap();

    let summary = chat
        .ingest_messages(&[Message::text(Role::System, "Be concise."), user("Hello")])
        .unwrap();
    assert!(summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let mut history = vec![
        ChatMessage {
            role: "system".into(),
            content: "Be concise.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        },
    ];
    let canonical = tokenizer.encode(&apply_chat_template(&tokenizer, &history, true).unwrap());
    assert_eq!(chat.tokens(), canonical);

    chat.generate_into(&GenerateOpts::default(), &mut Sink::default())
        .unwrap();
    history.push(ChatMessage {
        role: "assistant".into(),
        content: String::new(),
    });

    let cont_summary = chat.ingest(&user("Second turn")).unwrap();
    assert!(cont_summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    history.push(ChatMessage {
        role: "user".into(),
        content: "Second turn".into(),
    });
    let canonical2 = tokenizer.encode(&apply_chat_template(&tokenizer, &history, true).unwrap());
    assert_eq!(chat.tokens(), canonical2);
}

#[test]
#[ignore = "requires the pinned public GGUF; tests/api_chat/run.py verifies its SHA-256 and runs this test"]
fn public_lfm2_tokenizer_boundary_and_ten_turns() {
    let path = std::env::var("CERA_CHAT_PROFILE_MODEL")
        .expect("run tests/api_chat/run.py --model <GGUF> --output <dir>");
    let gguf =
        super::core_api::gguf::GgufFile::from_bytes(std::fs::read(path).unwrap().into()).unwrap();
    assert_eq!(gguf.get_str("general.architecture"), Some("lfm2"));
    assert_eq!(gguf.get_str("tokenizer.ggml.pre"), Some("lfm2"));
    let tokenizer = Arc::new(BpeTokenizer::from_gguf(&gguf).unwrap());
    legacy_boundary(tokenizer.clone());
    let reports = (0..10)
        .map(|_| {
            report(
                DecodeState::Terminal {
                    token: 7,
                    committed: false,
                },
                FinishReason::Stop,
            )
        })
        .collect();
    let answers: Vec<_> = (0..10)
        .map(|turn| format!("Answer {turn}: reçu."))
        .collect();
    let mut active = Chat::new(
        TraceExecution {
            decode: reports,
            output: answers.iter().map(|text| tokenizer.encode(text)).collect(),
            ..Default::default()
        },
        Profile::discover(tokenizer.clone()).unwrap(),
        0,
    )
    .ok()
    .unwrap();
    let mut history = Vec::new();
    for (turn, answer) in answers.iter().enumerate() {
        let text = format!("Turn {turn}: café 東京 🙂\nKeep spaces.\n");
        active.ingest(&user(&text)).unwrap();
        history.push(ChatMessage {
            role: "user".into(),
            content: text,
        });
        let canonical = tokenizer.encode(&apply_chat_template(&tokenizer, &history, true).unwrap());
        assert_eq!(active.tokens(), canonical);
        assert_eq!(active.phase(), SessionPhase::PromptReady);
        active
            .generate_into(&GenerateOpts::default(), &mut Sink::default())
            .unwrap();
        history.push(ChatMessage {
            role: "assistant".into(),
            content: answer.clone(),
        });
    }
    let raw = active.raw().unwrap();
    assert!(raw.resets.is_empty());
    assert_eq!(raw.append_calls, 10);
    assert_eq!(raw.decode_calls, 10);
    assert_eq!(raw.tokens.iter().filter(|&&id| id == 1).count(), 1);
}

#[test]
fn non_destructive_session_reclaim_and_into_inner() {
    let execution = TraceExecution::default();
    let mut chat = Chat::new(
        execution,
        Profile::discover(fixtures::tokenizer()).unwrap(),
        0,
    )
    .ok()
    .unwrap();
    let summary = chat.ingest(&user("hello")).unwrap();
    assert!(summary.input_tokens > 0);
    assert_eq!(chat.phase(), SessionPhase::PromptReady);

    let reclaimed = chat.into_inner();
    assert_eq!(reclaimed.position(), summary.input_tokens);
    assert_eq!(reclaimed.append_calls, 1);
}

#[test]
fn cancellation_handle_and_clear_cancel_on_chat() {
    let execution = TraceExecution::default();
    let mut chat = Chat::new(
        execution,
        Profile::discover(fixtures::tokenizer()).unwrap(),
        0,
    )
    .ok()
    .unwrap();

    let handle = chat.cancel_handle().expect("cancel handle must exist");
    assert!(!handle.load(Ordering::Relaxed));

    chat.cancel();
    assert!(handle.load(Ordering::Relaxed));

    // Clearing cancel does not affect phase or cursor.
    chat.clear_cancel();
    assert!(!handle.load(Ordering::Relaxed));
    assert_eq!(chat.phase(), SessionPhase::Idle);
    assert_eq!(chat.position(), 0);
}

// Test-only access does not enter the eventual application-facing contract.
trait ChatTraceExt {
    fn set_failure(&mut self, recovery: RecoveryOutcome);
    fn tokens(&mut self) -> &[u32];
}

impl ChatTraceExt for Chat<TraceExecution> {
    fn set_failure(&mut self, recovery: RecoveryOutcome) {
        self.execution_for_test().failure = Some(recovery);
    }
    fn tokens(&mut self) -> &[u32] {
        &self.execution_for_test().tokens
    }
}

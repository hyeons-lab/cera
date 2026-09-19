use super::*;
use std::sync::Arc;

use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use cera::tokenizer::BpeTokenizer;

struct TestModel {
    config: ModelConfig,
    token_to_emit: u32,
}

impl TestModel {
    fn new(token_to_emit: u32) -> Self {
        let config = ModelConfig {
            architecture: "ffi-chat-test".into(),
            n_layers: 1,
            hidden_size: 4,
            intermediate_size: 4,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            vocab_size: 32,
            max_seq_len: 256,
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
        Self {
            config,
            token_to_emit,
        }
    }
}

impl Model for TestModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(&self, tokens: &[u32], _: usize, state: &mut InferenceState) -> Vec<f32> {
        let rows = vec![0.0; tokens.len() * self.config.n_kv_heads * self.config.head_dim];
        state.append_kv(0, &rows, &rows);
        state.seq_len += tokens.len();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        if (self.token_to_emit as usize) < logits.len() {
            logits[self.token_to_emit as usize] = 10.0;
        }
        logits
    }

    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max_seq_len: usize,
    ) -> Result<(), cera::CeraError> {
        let mut config = self.config.clone();
        config.max_seq_len = config.max_seq_len.min(max_seq_len);
        *state = InferenceState::from_config_with_compression(&config, compression)?;
        Ok(())
    }
}

fn test_session_with_token(n_keep: u32, token_to_emit: u32) -> Arc<Session> {
    let model = Arc::new(TestModel::new(token_to_emit));
    let tokenizer = Arc::new(BpeTokenizer::chat_for_test());
    let inner = cera::Session::new(
        model,
        tokenizer,
        cera::ModalityCapabilities::text_only(),
        cera::SessionConfig {
            n_keep,
            ubatch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    Session::from_core(inner)
}

fn test_session(n_keep: u32) -> Arc<Session> {
    test_session_with_token(n_keep, 7) // Emits EOS (7)
}

struct TestSink {
    chunks: std::sync::Mutex<Vec<String>>,
    done: std::sync::Mutex<Option<FinishReason>>,
    done_count: std::sync::atomic::AtomicUsize,
}

impl TestSink {
    fn new() -> Self {
        Self {
            chunks: std::sync::Mutex::new(Vec::new()),
            done: std::sync::Mutex::new(None),
            done_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl ModalitySink for TestSink {
    fn on_thought_chunk(&self, _: String) {}
    fn on_text_chunk(&self, text: String) {
        self.chunks.lock().unwrap().push(text);
    }
    fn on_audio_frames(&self, _: Vec<f32>, _: u32) {}
    fn on_done(&self, reason: FinishReason) {
        self.done_count.fetch_add(1, Ordering::Relaxed);
        *self.done.lock().unwrap() = Some(reason);
    }
}

#[tokio::test]
async fn invalid_json_stream_notifies_once_for_both_entry_points() {
    let chat = test_session(0).into_chat().unwrap();
    let sink = Arc::new(TestSink::new());
    assert!(
        chat.generate_streaming_json(GenerateOpts::default(), "{".into(), sink.clone())
            .is_err()
    );
    assert_eq!(sink.done_count.load(Ordering::Relaxed), 1);
    assert!(matches!(
        *sink.done.lock().unwrap(),
        Some(FinishReason::Error { .. })
    ));
    let sink = Arc::new(TestSink::new());
    assert!(
        chat.generate_streaming_async_json(GenerateOpts::default(), "{".into(), sink.clone())
            .await
            .is_err()
    );
    assert_eq!(sink.done_count.load(Ordering::Relaxed), 1);
    assert!(matches!(
        *sink.done.lock().unwrap(),
        Some(FinishReason::Error { .. })
    ));
}

#[test]
fn moved_chat_terminal_callback_can_reenter() {
    struct ReentrantSink(Arc<ChatSession>, std::sync::atomic::AtomicUsize);
    impl ModalitySink for ReentrantSink {
        fn on_thought_chunk(&self, _: String) {}
        fn on_text_chunk(&self, _: String) {}
        fn on_audio_frames(&self, _: Vec<f32>, _: u32) {}
        fn on_done(&self, reason: FinishReason) {
            assert!(matches!(reason, FinishReason::Error { .. }));
            // phase uses try_lock: this assertion fails promptly if the callback
            // still holds the mutex, instead of hanging the regression test.
            assert!(matches!(self.0.phase(), Err(FfiError::Backend { .. })));
            assert!(self.0.reset().is_err());
            self.1.fetch_add(1, Ordering::Relaxed);
        }
    }
    let chat = test_session(0).into_chat().unwrap();
    let _session = chat.into_session().unwrap();
    let sink = Arc::new(ReentrantSink(
        chat.clone(),
        std::sync::atomic::AtomicUsize::new(0),
    ));
    assert!(
        chat.generate_streaming(GenerateOpts::default(), sink.clone())
            .is_err()
    );
    assert_eq!(sink.1.load(Ordering::Relaxed), 1);
}

#[test]
fn role_and_message_conversions() {
    assert_eq!(Role::from(cera::session::chat::Role::System), Role::System);
    assert_eq!(Role::from(cera::session::chat::Role::User), Role::User);
    assert_eq!(
        Role::from(cera::session::chat::Role::Assistant),
        Role::Assistant
    );
    assert_eq!(Role::from(cera::session::chat::Role::Tool), Role::Tool);

    let msg = chat_message_user("hello".into());
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.content, "hello");

    let core_msg: cera::session::chat::Message = msg.into();
    assert_eq!(core_msg.role, cera::session::chat::Role::User);
    assert_eq!(core_msg.content.len(), 1);

    let sys = chat_message_system("sys prompt".into());
    assert_eq!(sys.role, Role::System);
    let asst = chat_message_assistant("response".into());
    assert_eq!(asst.role, Role::Assistant);
    let tool = chat_message_tool("tool result".into());
    assert_eq!(tool.role, Role::Tool);
}

#[test]
fn session_phase_conversions() {
    use cera::session::chat::SessionPhase as C;
    assert_eq!(SessionPhase::from(C::Idle), SessionPhase::Idle);
    assert_eq!(
        SessionPhase::from(C::PromptReady),
        SessionPhase::PromptReady
    );
    assert_eq!(
        SessionPhase::from(C::TurnComplete),
        SessionPhase::TurnComplete
    );
    assert_eq!(
        SessionPhase::from(C::Interrupted),
        SessionPhase::Interrupted
    );
    assert_eq!(SessionPhase::from(C::RawContext), SessionPhase::RawContext);
    assert_eq!(SessionPhase::from(C::Unusable), SessionPhase::Unusable);
}

#[test]
fn validation_error_mapping_to_ffi_error() {
    let err = ValidationError::UnsupportedProfile;
    let ffi_err = FfiError::from(err);
    match ffi_err {
        FfiError::ChatValidation { error } => {
            assert_eq!(error, ValidationError::UnsupportedProfile);
        }
        _ => panic!("unexpected FfiError variant"),
    }
}

#[test]
fn session_into_chat_lifecycle_and_turn_completion() {
    let session = test_session(0);
    let chat = session
        .into_chat()
        .expect("chat construction should succeed");

    assert_eq!(chat.phase().unwrap(), SessionPhase::Idle);
    assert_eq!(chat.position().unwrap(), 0);

    let user_msg = chat_message_user("What is the capital of France?".into());
    let ingest_summary = chat.ingest(user_msg).expect("ingest should succeed");
    assert!(ingest_summary.input_tokens > 0);
    assert_eq!(chat.phase().unwrap(), SessionPhase::PromptReady);

    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let turn = chat.complete(opts).expect("completion should succeed");
    assert_eq!(turn.summary.finish_reason, FinishReason::Stop);
    assert_eq!(chat.phase().unwrap(), SessionPhase::TurnComplete);

    chat.reset().expect("reset should succeed");
    assert_eq!(chat.phase().unwrap(), SessionPhase::Idle);
    assert_eq!(chat.position().unwrap(), 0);
}

#[test]
fn session_into_chat_refusal_preserves_session() {
    // n_keep = 4 is sliding context, which Chat coordinator must refuse
    let session = test_session(4);
    let result = session.into_chat();
    assert!(result.is_err());

    match result.unwrap_err() {
        FfiError::ChatValidation { error } => {
            assert_eq!(error, ValidationError::SlidingContext);
        }
        other => panic!("expected SlidingContext validation error, got: {other:?}"),
    }

    // Original session must still be intact and usable
    assert_eq!(session.position(), 0);
    let rec = session
        .recovery_status()
        .expect("session should still be usable");
    assert!(rec.usable);
}

#[test]
fn moved_session_returns_error_on_subsequent_calls() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    session.cancel();
    assert!(!chat.cancel.load(Ordering::Relaxed));
    chat.cancel();
    session.clear_cancel();
    assert!(chat.cancel.load(Ordering::Relaxed));
    chat.clear_cancel().unwrap();
    chat.ingest(chat_message_user("still usable".into()))
        .unwrap();
    assert_ne!(
        chat.complete(GenerateOpts::default())
            .unwrap()
            .summary
            .finish_reason,
        FinishReason::Cancelled
    );

    let err = session.append_text("test".into()).unwrap_err();
    match err {
        FfiError::Backend { detail } => {
            assert!(detail.contains("session has been moved into a ChatSession"));
        }
        other => panic!("expected Backend moved session error, got: {other:?}"),
    }
}

#[test]
fn chat_session_into_session_reclaims_usable_session() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    let user_msg = chat_message_user("Hello".into());
    chat.ingest(user_msg).expect("ingest succeeds");

    let reclaimed = chat.into_session().expect("into_session succeeds");
    assert!(reclaimed.position() > 0);

    let err = chat.phase().unwrap_err();
    match err {
        FfiError::Backend { detail } => {
            assert!(detail.contains("chat session has been moved back into a Session"));
        }
        other => panic!("expected Backend moved chat error, got: {other:?}"),
    }
}

#[test]
fn chat_session_cancellation_and_clear() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    chat.cancel();
    chat.clear_cancel().expect("clear_cancel succeeds");
}

#[test]
fn chat_session_streaming_generation() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    let user_msg = chat_message_user("Streaming test".into());
    chat.ingest(user_msg).expect("ingest succeeds");

    let sink = Arc::new(TestSink::new());
    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let summary = chat
        .generate_streaming(opts, sink.clone())
        .expect("streaming should succeed");
    assert_eq!(summary.finish_reason, FinishReason::Stop);
    assert_eq!(*sink.done.lock().unwrap(), Some(FinishReason::Stop));
    assert_eq!(chat.phase().unwrap(), SessionPhase::TurnComplete);
}

struct CancelSink {
    chat: std::sync::Mutex<Option<Arc<ChatSession>>>,
    cancelled: std::sync::atomic::AtomicBool,
    done: std::sync::Mutex<Option<FinishReason>>,
}

impl CancelSink {
    fn new() -> Self {
        Self {
            chat: std::sync::Mutex::new(None),
            cancelled: std::sync::atomic::AtomicBool::new(false),
            done: std::sync::Mutex::new(None),
        }
    }

    fn attach(&self, chat: Arc<ChatSession>) {
        *self.chat.lock().unwrap() = Some(chat);
    }
}

impl ModalitySink for CancelSink {
    fn on_thought_chunk(&self, _: String) {}
    fn on_text_chunk(&self, _: String) {
        if !self.cancelled.swap(true, Ordering::Relaxed)
            && let Some(ref chat) = *self.chat.lock().unwrap()
        {
            chat.cancel();
        }
    }
    fn on_audio_frames(&self, _: Vec<f32>, _: u32) {}
    fn on_done(&self, reason: FinishReason) {
        *self.done.lock().unwrap() = Some(reason);
    }
}

#[test]
fn chat_session_reentrant_cancellation_in_streaming_callback() {
    let session = test_session_with_token(0, 10);
    let chat = session.into_chat().expect("into_chat succeeds");

    let user_msg = chat_message_user("Streaming cancel test".into());
    chat.ingest(user_msg).expect("ingest succeeds");

    let sink = Arc::new(CancelSink::new());
    sink.attach(chat.clone());

    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let result = chat.generate_streaming(opts, sink.clone());
    match result {
        Ok(summary) => assert_eq!(summary.finish_reason, FinishReason::Cancelled),
        Err(FfiError::Cancelled) => {}
        other => panic!("expected Cancelled summary or error, got: {other:?}"),
    }
    assert_eq!(*sink.done.lock().unwrap(), Some(FinishReason::Cancelled));
    assert_eq!(chat.phase().unwrap(), SessionPhase::Interrupted);
}

#[test]
fn chat_session_recovery_status_nonblocking_when_locked() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    // Acquire inner lock directly to simulate active decode or ingest
    let guard = chat.inner.lock().unwrap();

    // recovery_status must not block and return Busy
    let err = chat.recovery_status().unwrap_err();
    match err {
        FfiError::Busy => {}
        other => panic!("expected Busy error, got: {other:?}"),
    }
    drop(guard);

    // After dropping the lock, recovery_status returns Ok
    let status = chat.recovery_status().expect("recovery status succeeds");
    assert!(status.usable);
    assert_eq!(status.position, 0);
}

#[test]
fn chat_session_streaming_early_error_dispatches_on_done() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    // Calling generate_streaming in Idle phase without ingest fails validation
    let sink = Arc::new(TestSink::new());
    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let err = chat.generate_streaming(opts, sink.clone()).unwrap_err();
    match err {
        FfiError::ChatValidation { error } => {
            assert_eq!(
                error,
                ValidationError::Phase {
                    phase: SessionPhase::Idle
                }
            );
        }
        other => panic!("expected ChatValidation Phase error, got: {other:?}"),
    }
    // Terminal event must have been dispatched to sink
    let done = sink.done.lock().unwrap().clone();
    assert!(matches!(done, Some(FinishReason::Error { .. })));
}

#[test]
fn chat_message_bidirectional_conversion() {
    let core_msg =
        cera::session::chat::Message::text(cera::session::chat::Role::Assistant, "Hello from core");
    let ffi_msg = Message::from(&core_msg);
    assert_eq!(ffi_msg.role, Role::Assistant);
    assert_eq!(ffi_msg.content, "Hello from core");
    assert!(ffi_msg.image_bytes.is_none());
    assert!(ffi_msg.audio_pcm.is_none());

    let core_roundtrip: cera::session::chat::Message = ffi_msg.into();
    assert_eq!(core_roundtrip, core_msg);

    let multimodal_img = cera::session::chat::Message {
        role: cera::session::chat::Role::User,
        content: vec![
            cera::session::chat::ContentPart::Image(vec![1, 2, 3]),
            cera::session::chat::ContentPart::Text("what is this?".into()),
        ],
    };
    let ffi_img = Message::from(&multimodal_img);
    assert_eq!(ffi_img.role, Role::User);
    assert_eq!(ffi_img.content, "what is this?");
    assert_eq!(ffi_img.image_bytes, Some(vec![1, 2, 3]));
    let img_roundtrip: cera::session::chat::Message = ffi_img.into();
    assert_eq!(img_roundtrip, multimodal_img);

    let user_img_factory = chat_message_user_image(vec![4, 5, 6], Some("describe".into()));
    assert_eq!(user_img_factory.role, Role::User);
    assert_eq!(user_img_factory.content, "describe");
    assert_eq!(user_img_factory.image_bytes, Some(vec![4, 5, 6]));

    let user_aud_factory = chat_message_user_audio(vec![0.1, 0.2], 16000, Some("listen".into()));
    assert_eq!(user_aud_factory.role, Role::User);
    assert_eq!(user_aud_factory.content, "listen");
    assert_eq!(user_aud_factory.audio_pcm, Some(vec![0.1, 0.2]));
    assert_eq!(user_aud_factory.audio_sample_rate, Some(16000));
}

#[test]
fn chat_session_recovery_status_retains_ingest_recovery() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    // Single message ingestion with system role fails validation because
    // Profile::render requires the message batch to end with a user role.
    let sys_msg = chat_message_system("System instructions".to_string());
    let err = chat.ingest(sys_msg).unwrap_err();
    assert!(matches!(err, FfiError::ChatValidation { .. }));

    // recovery_status must retain the IngestRecovery diagnostic
    let status = chat.recovery_status().expect("recovery status succeeds");
    assert!(status.usable);
    assert!(status.last_ingest_recovery.is_some());

    // Valid user turn clears the retained diagnostic
    let user_msg = chat_message_user("Hello!".to_string());
    chat.ingest(user_msg).expect("valid ingest succeeds");

    let status_after = chat.recovery_status().expect("recovery status succeeds");
    assert!(status_after.usable);
    assert!(status_after.last_ingest_recovery.is_none());

    // Ingest failure again to set diagnostic
    let invalid_assistant = chat_message_assistant("Premature assistant reply".to_string());
    chat.ingest(invalid_assistant).unwrap_err();
    assert!(
        chat.recovery_status()
            .unwrap()
            .last_ingest_recovery
            .is_some()
    );

    // Explicit reset clears the diagnostic
    chat.reset().expect("reset succeeds");
    assert!(
        chat.recovery_status()
            .unwrap()
            .last_ingest_recovery
            .is_none()
    );
}

#[test]
fn chat_session_double_into_session_fails() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    // First into_session reclamation succeeds
    let reclaimed = chat.into_session().expect("first reclamation succeeds");
    assert_eq!(reclaimed.position(), 0);

    // Second into_session fails fail-closed
    let err = chat.into_session().unwrap_err();
    match err {
        FfiError::Backend { detail } => {
            assert!(detail.contains("already been moved"));
        }
        other => panic!("expected Backend error on double move, got: {other:?}"),
    }
}

#[test]
fn chat_session_phase_nonblocking_when_locked() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    // Acquire inner lock directly to simulate active decode or ingest
    let guard = chat.inner.lock().unwrap();

    // phase must not block and return Busy
    let err = chat.phase().unwrap_err();
    match err {
        FfiError::Busy => {}
        other => panic!("expected Busy error, got: {other:?}"),
    }
    drop(guard);

    // After dropping the lock, phase returns Ok
    assert_eq!(chat.phase().unwrap(), SessionPhase::Idle);
}

#[test]
fn chat_session_cancel_after_into_session_does_not_cancel_reclaimed_session() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    let reclaimed = chat.into_session().expect("reclamation succeeds");

    // Calling cancel on the moved chat session must be a no-op
    chat.cancel();

    // Reclaimed session's cancellation atomic must remain false
    assert!(!reclaimed.cancel.load(std::sync::atomic::Ordering::Relaxed));
}

#[tokio::test]
async fn chat_session_complete_async_produces_turn_result() {
    let session = test_session_with_token(0, 10);
    let chat = session.into_chat().expect("into_chat succeeds");

    chat.ingest(chat_message_user("hello async".into()))
        .expect("ingest succeeds");

    let opts = GenerateOpts {
        max_tokens: 1,
        temperature: 0.0,
        ..Default::default()
    };

    let result = Arc::clone(&chat)
        .complete_async(opts)
        .await
        .expect("complete_async succeeds");
    assert_eq!(result.text, "hi ");
    assert_eq!(result.summary.finish_reason, FinishReason::MaxTokens);
    assert_eq!(chat.phase().unwrap(), SessionPhase::Interrupted);
}

#[tokio::test]
async fn chat_session_generate_streaming_async_produces_stream() {
    let session = test_session_with_token(0, 10);
    let chat = session.into_chat().expect("into_chat succeeds");

    chat.ingest(chat_message_user("stream async".into()))
        .expect("ingest succeeds");

    let opts = GenerateOpts {
        max_tokens: 1,
        temperature: 0.0,
        ..Default::default()
    };

    let sink = Arc::new(TestSink::new());

    let summary = Arc::clone(&chat)
        .generate_streaming_async(opts, sink.clone())
        .await
        .expect("generate_streaming_async succeeds");

    assert_eq!(summary.finish_reason, FinishReason::MaxTokens);
    assert_eq!(sink.chunks.lock().unwrap().concat(), "hi ");
    assert_eq!(*sink.done.lock().unwrap(), Some(FinishReason::MaxTokens));
    assert_eq!(chat.phase().unwrap(), SessionPhase::Interrupted);
}

#[tokio::test]
async fn chat_session_consecutive_streaming_async_generations_succeed() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    // Turn 1 completes cleanly to TurnComplete
    chat.ingest(chat_message_user("turn 1".into()))
        .expect("turn 1 ingest succeeds");
    let opts = GenerateOpts {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let sink1 = Arc::new(TestSink::new());
    let summary1 = Arc::clone(&chat)
        .generate_streaming_async(opts.clone(), sink1.clone())
        .await
        .expect("turn 1 streaming succeeds");
    assert_eq!(summary1.finish_reason, FinishReason::Stop);
    assert_eq!(*sink1.done.lock().unwrap(), Some(FinishReason::Stop));
    assert_eq!(chat.phase().unwrap(), SessionPhase::TurnComplete);

    // Turn 2 on same session proves stream exhaustion does not set sticky cancellation
    chat.ingest(chat_message_user("turn 2".into()))
        .expect("turn 2 ingest succeeds");
    let sink2 = Arc::new(TestSink::new());
    let summary2 = Arc::clone(&chat)
        .generate_streaming_async(opts, sink2.clone())
        .await
        .expect("turn 2 streaming succeeds");
    assert_eq!(summary2.finish_reason, FinishReason::Stop);
    assert_eq!(*sink2.done.lock().unwrap(), Some(FinishReason::Stop));
    assert_eq!(chat.phase().unwrap(), SessionPhase::TurnComplete);
}

#[test]
fn json_schema_to_grammar_compiles_valid_schema() {
    let schema = r#"{"type": "string", "enum": ["apple", "banana"]}"#;
    let grammar = json_schema_to_grammar(schema.to_string()).expect("compiles valid schema");
    assert!(grammar.contains("root ::="));
    assert!(grammar.contains("apple"));
    assert!(grammar.contains("banana"));
}

#[test]
fn json_schema_to_grammar_rejects_invalid_schema() {
    let schema = r#"{"type": "invalid_type_123"}"#;
    let err = json_schema_to_grammar(schema.to_string()).unwrap_err();
    match err {
        FfiError::GrammarParse { detail } => {
            assert!(detail.contains("invalid JSON schema"));
        }
        other => panic!("expected GrammarParse, got {other:?}"),
    }
}

#[test]
fn chat_session_complete_json_rejects_invalid_schema() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");
    chat.ingest(chat_message_user("give me json".into()))
        .expect("ingest succeeds");

    let opts = GenerateOpts::default();
    let err = chat
        .complete_json(opts, r#"{"type": "unsupported"}"#.into())
        .unwrap_err();
    match err {
        FfiError::GrammarParse { detail } => {
            assert!(detail.contains("invalid JSON schema"));
        }
        other => panic!("expected GrammarParse, got {other:?}"),
    }
}

#[test]
fn chat_session_tool_registration_and_accessors() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    assert!(chat.tools().unwrap().is_empty());
    assert_eq!(chat.tool_format().unwrap(), ToolFormat::Lfm2Pythonic);

    let tool = ToolDef {
        name: "get_weather".into(),
        description: Some("Get weather for a given city".into()),
        parameters_json: r#"{"type": "object", "properties": {"city": {"type": "string"}}}"#.into(),
    };

    chat.set_tools(vec![tool]).expect("set_tools succeeds");
    let tools = chat.tools().expect("tools query succeeds");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(
        tools[0].description.as_deref(),
        Some("Get weather for a given city")
    );

    chat.set_tool_format(ToolFormat::Hermes)
        .expect("set_tool_format succeeds");
    assert_eq!(chat.tool_format().unwrap(), ToolFormat::Hermes);
}

#[test]
fn chat_session_tool_execution_flow() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    let tool = ToolDef {
        name: "calculator".into(),
        description: Some("Calculate mathematical expression".into()),
        parameters_json: r#"{"type": "object", "properties": {"expr": {"type": "string"}}}"#.into(),
    };
    chat.set_tools(vec![tool]).expect("set_tools succeeds");

    chat.ingest(chat_message_user("Calculate 2 + 2".into()))
        .expect("user message ingests");
    assert_eq!(chat.phase().unwrap(), SessionPhase::PromptReady);

    // Ingesting tool response directly in Idle/PromptReady phase returns validation error:
    let err = chat
        .ingest_tool_response("calculator".into(), "4".into())
        .unwrap_err();
    assert!(matches!(err, FfiError::ChatValidation { .. }));
}

#[test]
fn turn_result_tool_calls_field() {
    let core_turn = cera::session::chat::TurnResult {
        text: "response".into(),
        tokens: vec![1, 2, 3],
        summary: cera::session::GenerateSummary {
            tokens_generated: 3,
            prompt_eval_tokens: 10,
            prompt_eval_ms: 5,
            decode_ms: 10,
            finish_reason: cera::session::FinishReason::Stop,
        },
        tool_calls: vec![cera::tools::ToolCall {
            name: "test_call".into(),
            arguments: serde_json::json!({"arg": "val"}),
        }],
    };

    let ffi_turn = TurnResult::from(core_turn);
    assert_eq!(ffi_turn.text, "response");
    assert_eq!(ffi_turn.tool_calls.len(), 1);
    assert_eq!(ffi_turn.tool_calls[0].name, "test_call");
    assert!(
        ffi_turn.tool_calls[0]
            .arguments_json
            .contains("\"arg\":\"val\"")
    );
}

#[test]
fn chat_session_checkpoint_export_import_and_file_persistence() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");

    chat.ingest(chat_message_user("hello".into()))
        .expect("ingest succeeds");
    chat.complete(GenerateOpts::default())
        .expect("complete succeeds");
    assert_eq!(chat.phase().unwrap(), SessionPhase::TurnComplete);
    let expected_pos = chat.position().unwrap();

    let tool = ToolDef {
        name: "math".into(),
        description: Some("evaluate math".into()),
        parameters_json: r#"{"type":"object"}"#.into(),
    };
    chat.set_tools(vec![tool]).expect("set_tools succeeds");
    chat.set_tool_format(ToolFormat::Hermes)
        .expect("set_tool_format succeeds");

    let bytes = chat
        .export_checkpoint()
        .expect("export_checkpoint succeeds");
    assert!(!bytes.is_empty());

    let dir = tempfile::tempdir().expect("tempdir succeeds");
    let file_path = dir
        .path()
        .join("chat_ffi.chk")
        .to_str()
        .unwrap()
        .to_string();
    chat.save_checkpoint(file_path.clone())
        .expect("save_checkpoint succeeds");

    let session2 = test_session(0);
    let chat2 = session2.into_chat().expect("into_chat succeeds");
    assert_eq!(chat2.phase().unwrap(), SessionPhase::Idle);

    chat2
        .load_checkpoint(file_path)
        .expect("load_checkpoint succeeds");
    assert_eq!(chat2.phase().unwrap(), SessionPhase::TurnComplete);
    assert_eq!(chat2.position().unwrap(), expected_pos);
    assert_eq!(chat2.tool_format().unwrap(), ToolFormat::Hermes);
    assert_eq!(chat2.tools().unwrap().len(), 1);

    let session3 = test_session(0);
    let chat3 = session3.into_chat().expect("into_chat succeeds");
    chat3
        .import_checkpoint(bytes)
        .expect("import_checkpoint succeeds");
    assert_eq!(chat3.phase().unwrap(), SessionPhase::TurnComplete);
    assert_eq!(chat3.position().unwrap(), expected_pos);

    chat3.set_tools(Vec::new()).expect("clear tools succeeds");
    chat3
        .ingest(chat_message_user("next turn".into()))
        .expect("ingest turn 2 succeeds");
    let turn2 = chat3
        .complete(GenerateOpts::default())
        .expect("complete turn 2 succeeds");
    assert_eq!(turn2.text, "");
    assert!(chat3.position().unwrap() > expected_pos);
}

#[test]
fn session_checkpoint_export_import_and_file_persistence() {
    let session = test_session(0);
    session
        .append_text("testing persistence".into())
        .expect("append_text succeeds");
    let expected_pos = session.position();

    let bytes = session
        .export_checkpoint()
        .expect("export_checkpoint succeeds");
    let dir = tempfile::tempdir().expect("tempdir succeeds");
    let file_path = dir
        .path()
        .join("sess_ffi.chk")
        .to_str()
        .unwrap()
        .to_string();
    session
        .save_checkpoint(file_path.clone())
        .expect("save_checkpoint succeeds");

    let session2 = test_session(0);
    assert_eq!(session2.position(), 0);
    session2
        .load_checkpoint(file_path)
        .expect("load_checkpoint succeeds");
    assert_eq!(session2.position(), expected_pos);

    let session3 = test_session(0);
    session3
        .import_checkpoint(bytes)
        .expect("import_checkpoint succeeds");
    assert_eq!(session3.position(), expected_pos);
}

#[test]
fn chat_session_checkpoint_rejects_corrupted_data() {
    let session = test_session(0);
    let chat = session.into_chat().expect("into_chat succeeds");
    let err = chat.import_checkpoint(vec![1, 2, 3, 4]).unwrap_err();
    assert!(matches!(err, FfiError::Backend { .. }));
}

#[test]
fn tool_def_try_from_validates_parameters_json() {
    use crate::ToolDef;

    // Empty parameters_json defaults to empty object
    let empty_tool = ToolDef {
        name: "test_empty".to_string(),
        description: None,
        parameters_json: "".to_string(),
    };
    let core_empty: cera::tools::ToolDef = empty_tool.try_into().unwrap();
    assert_eq!(core_empty.name, "test_empty");
    assert!(core_empty.parameters.is_object());

    // Valid object schema succeeds
    let valid_tool = ToolDef {
        name: "test_valid".to_string(),
        description: Some("valid tool".to_string()),
        parameters_json: r#"{"type":"object","properties":{"location":{"type":"string"}}}"#
            .to_string(),
    };
    let core_valid: cera::tools::ToolDef = valid_tool.try_into().unwrap();
    assert_eq!(core_valid.name, "test_valid");
    assert_eq!(core_valid.description.as_deref(), Some("valid tool"));

    // Invalid JSON fails
    let invalid_json_tool = ToolDef {
        name: "test_invalid_json".to_string(),
        description: None,
        parameters_json: "{not_valid_json}".to_string(),
    };
    let err = cera::tools::ToolDef::try_from(invalid_json_tool).unwrap_err();
    assert!(matches!(err, FfiError::Backend { .. }));

    // Scalar JSON fails
    let scalar_tool = ToolDef {
        name: "test_scalar".to_string(),
        description: None,
        parameters_json: "42".to_string(),
    };
    let err = cera::tools::ToolDef::try_from(scalar_tool).unwrap_err();
    assert!(matches!(err, FfiError::Backend { .. }));

    // Array JSON fails
    let array_tool = ToolDef {
        name: "test_array".to_string(),
        description: None,
        parameters_json: "[]".to_string(),
    };
    let err = cera::tools::ToolDef::try_from(array_tool).unwrap_err();
    assert!(matches!(err, FfiError::Backend { .. }));
}

//! Foreign-language bindings to [`cera::session::chat`] via UniFFI.
//!
//! Exposes the stateful [`ChatSession`] coordinator, turn execution methods,
//! message structures, and typed validation errors to Swift, Kotlin, and Python.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::{
    FfiError, FinishReason, ForeignSinkAdapter, GenerateOpts, GenerateSummary, ModalitySink,
    Session, SessionRecoveryStatus, ToolCall, ToolDef, ToolFormat,
};

/// Message author role in conversational chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Role {
    /// System prompt setting instructions and context.
    System,
    /// User prompt input.
    User,
    /// Assistant model response.
    Assistant,
    /// Tool result or response payload.
    Tool,
}

impl From<cera::session::chat::Role> for Role {
    fn from(role: cera::session::chat::Role) -> Self {
        match role {
            cera::session::chat::Role::System => Self::System,
            cera::session::chat::Role::User => Self::User,
            cera::session::chat::Role::Assistant => Self::Assistant,
            cera::session::chat::Role::Tool => Self::Tool,
        }
    }
}

impl From<Role> for cera::session::chat::Role {
    fn from(role: Role) -> Self {
        match role {
            Role::System => Self::System,
            Role::User => Self::User,
            Role::Assistant => Self::Assistant,
            Role::Tool => Self::Tool,
        }
    }
}

/// A structured conversational turn message.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct Message {
    /// Author role.
    pub role: Role,
    /// Message text content.
    pub content: String,
}

impl From<Message> for cera::session::chat::Message {
    fn from(m: Message) -> Self {
        cera::session::chat::Message::text(m.role.into(), m.content)
    }
}

impl From<&Message> for cera::session::chat::Message {
    fn from(m: &Message) -> Self {
        cera::session::chat::Message::text(m.role.into(), m.content.clone())
    }
}

impl TryFrom<&cera::session::chat::Message> for Message {
    type Error = FfiError;

    fn try_from(msg: &cera::session::chat::Message) -> Result<Self, Self::Error> {
        let mut text = String::new();
        for part in &msg.content {
            match part {
                cera::session::chat::ContentPart::Text(s) => text.push_str(s),
                cera::session::chat::ContentPart::Image(_)
                | cera::session::chat::ContentPart::Audio { .. } => {
                    return Err(FfiError::UnsupportedModality);
                }
            }
        }
        Ok(Message {
            role: msg.role.into(),
            content: text,
        })
    }
}

/// Convenience factory for a user text message.
#[uniffi::export]
pub fn chat_message_user(content: String) -> Message {
    Message {
        role: Role::User,
        content,
    }
}

/// Convenience factory for a system text message.
#[uniffi::export]
pub fn chat_message_system(content: String) -> Message {
    Message {
        role: Role::System,
        content,
    }
}

/// Convenience factory for an assistant text message.
#[uniffi::export]
pub fn chat_message_assistant(content: String) -> Message {
    Message {
        role: Role::Assistant,
        content,
    }
}

/// Convenience factory for a tool text message.
#[uniffi::export]
pub fn chat_message_tool(content: String) -> Message {
    Message {
        role: Role::Tool,
        content,
    }
}

/// Compile a JSON Schema definition string into a GBNF grammar string.
#[uniffi::export]
pub fn json_schema_to_grammar(schema_json: String) -> Result<String, FfiError> {
    cera::grammar::json_schema_to_gbnf_str(&schema_json).map_err(|e| FfiError::GrammarParse {
        detail: format!("invalid JSON schema: {e}"),
    })
}

/// Lifecycle phase of a stateful chat coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum SessionPhase {
    /// Clean session at position 0, ready for initial message ingestion.
    Idle,
    /// Input messages have been appended and prefilled; ready for decode.
    PromptReady,
    /// A turn finished with a terminal end-of-sequence stop marker.
    TurnComplete,
    /// Generation was interrupted by cancellation or custom nonterminal stop.
    Interrupted,
    /// Underlying execution state was modified outside chat rules; replacement required.
    RawContext,
    /// Unrecoverable execution fault or unwind; checked reset required to restore usability.
    Unusable,
}

impl From<cera::session::chat::SessionPhase> for SessionPhase {
    fn from(phase: cera::session::chat::SessionPhase) -> Self {
        match phase {
            cera::session::chat::SessionPhase::Idle => Self::Idle,
            cera::session::chat::SessionPhase::PromptReady => Self::PromptReady,
            cera::session::chat::SessionPhase::TurnComplete => Self::TurnComplete,
            cera::session::chat::SessionPhase::Interrupted => Self::Interrupted,
            cera::session::chat::SessionPhase::RawContext => Self::RawContext,
            cera::session::chat::SessionPhase::Unusable => Self::Unusable,
        }
    }
}

impl From<SessionPhase> for cera::session::chat::SessionPhase {
    fn from(phase: SessionPhase) -> Self {
        match phase {
            SessionPhase::Idle => Self::Idle,
            SessionPhase::PromptReady => Self::PromptReady,
            SessionPhase::TurnComplete => Self::TurnComplete,
            SessionPhase::Interrupted => Self::Interrupted,
            SessionPhase::RawContext => Self::RawContext,
            SessionPhase::Unusable => Self::Unusable,
        }
    }
}

/// Validation failure during chat construction, preparation, or decode.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum ValidationError {
    /// Model or tokenizer configuration does not match a supported chat profile.
    UnsupportedProfile,
    /// Sliding context configuration (n_keep != 0) is not supported for chat.
    SlidingContext,
    /// Model audio output is not supported for text chat.
    AudioOutput,
    /// Generation parameter validation error.
    Generation { detail: String },
    /// Operation refused in the current session phase.
    Phase { phase: SessionPhase },
    /// Message batch provided to ingest was empty.
    EmptyBatch,
    /// Message role sequence violates chat rules.
    RoleOrder { message: u32 },
    /// Message role is not supported in the active profile.
    UnsupportedRole { message: u32 },
    /// Content part is not supported in the active profile.
    UnsupportedContent { message: u32, part: u32 },
    /// Message text contains a reserved ChatML marker sequence.
    ReservedMarker { message: u32 },
    /// Context tokens required exceed available capacity in the KV cache.
    Capacity { required: u32, available: u32 },
    /// Chat template rendering error.
    Template { detail: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedProfile => f.write_str("unsupported chat template profile"),
            Self::SlidingContext => {
                f.write_str("sliding context (n_keep != 0) is not supported for chat")
            }
            Self::AudioOutput => f.write_str("audio output is not supported for text chat"),
            Self::Generation { detail } => write!(f, "generation error: {detail}"),
            Self::Phase { phase } => write!(f, "invalid session phase: {phase:?}"),
            Self::EmptyBatch => f.write_str("empty message batch"),
            Self::RoleOrder { message } => {
                write!(f, "invalid role order at message index {message}")
            }
            Self::UnsupportedRole { message } => {
                write!(f, "unsupported role at message index {message}")
            }
            Self::UnsupportedContent { message, part } => {
                write!(
                    f,
                    "unsupported content part {part} in message index {message}"
                )
            }
            Self::ReservedMarker { message } => {
                write!(f, "reserved marker found in message index {message}")
            }
            Self::Capacity {
                required,
                available,
            } => write!(
                f,
                "required context capacity {required} exceeds available capacity {available}"
            ),
            Self::Template { detail } => write!(f, "template error: {detail}"),
        }
    }
}

impl std::error::Error for ValidationError {}

impl From<cera::session::chat::ValidationError> for ValidationError {
    fn from(err: cera::session::chat::ValidationError) -> Self {
        use cera::session::chat::ValidationError as E;
        match err {
            E::UnsupportedProfile => Self::UnsupportedProfile,
            E::SlidingContext => Self::SlidingContext,
            E::AudioOutput => Self::AudioOutput,
            E::Generation(detail) => Self::Generation { detail },
            E::Phase(phase) => Self::Phase {
                phase: phase.into(),
            },
            E::EmptyBatch => Self::EmptyBatch,
            E::RoleOrder { message } => Self::RoleOrder {
                message: message.min(u32::MAX as usize) as u32,
            },
            E::UnsupportedRole { message } => Self::UnsupportedRole {
                message: message.min(u32::MAX as usize) as u32,
            },
            E::UnsupportedContent { message, part } => Self::UnsupportedContent {
                message: message.min(u32::MAX as usize) as u32,
                part: part.min(u32::MAX as usize) as u32,
            },
            E::ReservedMarker { message } => Self::ReservedMarker {
                message: message.min(u32::MAX as usize) as u32,
            },
            E::Capacity {
                required,
                available,
            } => Self::Capacity {
                required: required.min(u32::MAX as usize) as u32,
                available: available.min(u32::MAX as usize) as u32,
            },
            E::Template(detail) => Self::Template { detail },
        }
    }
}

/// Summary of a successful message ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct IngestSummary {
    /// Number of tokens encoded and appended to context.
    pub input_tokens: u32,
    /// KV position before ingestion.
    pub position_before: u32,
    /// KV position after ingestion.
    pub position_after: u32,
}

impl From<cera::session::chat::IngestSummary> for IngestSummary {
    fn from(s: cera::session::chat::IngestSummary) -> Self {
        Self {
            input_tokens: s.input_tokens.min(u32::MAX as usize) as u32,
            position_before: s.position_before.min(u32::MAX as usize) as u32,
            position_after: s.position_after.min(u32::MAX as usize) as u32,
        }
    }
}

/// Result of a completed chat turn.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TurnResult {
    /// Decoded assistant response text.
    pub text: String,
    /// Token identifiers emitted during the turn.
    pub tokens: Vec<u32>,
    /// Generation summary metrics.
    pub summary: GenerateSummary,
    /// Parsed tool calls emitted by the model during the turn.
    #[uniffi(default = [])]
    pub tool_calls: Vec<ToolCall>,
}

impl From<cera::session::chat::TurnResult> for TurnResult {
    fn from(res: cera::session::chat::TurnResult) -> Self {
        Self {
            text: res.text,
            tokens: res.tokens,
            summary: res.summary.into(),
            tool_calls: res.tool_calls.into_iter().map(Into::into).collect(),
        }
    }
}

/// Stateful chat coordinator wrapping an inference session.
#[derive(Debug, uniffi::Object)]
pub struct ChatSession {
    inner: std::sync::Mutex<Option<cera::session::chat::SessionChat>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    position: Arc<std::sync::atomic::AtomicU32>,
    moved: std::sync::atomic::AtomicBool,
    last_ingest_recovery: std::sync::Mutex<Option<crate::IngestRecovery>>,
}

impl ChatSession {
    pub(crate) fn from_core_session(session: cera::Session) -> Result<Arc<Self>, FfiError> {
        let position = session.position_handle();
        let chat =
            cera::session::chat::core_chat(session).map_err(|(_, err)| FfiError::from(err))?;
        let cancel = chat
            .cancel_handle()
            .unwrap_or_else(|| Arc::new(std::sync::atomic::AtomicBool::new(false)));
        Ok(Arc::new(Self {
            inner: std::sync::Mutex::new(Some(chat)),
            cancel,
            position,
            moved: std::sync::atomic::AtomicBool::new(false),
            last_ingest_recovery: std::sync::Mutex::new(None),
        }))
    }

    fn lock_inner(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<cera::session::chat::SessionChat>>, FfiError> {
        self.inner.lock().map_err(|e| FfiError::Backend {
            detail: format!(
                "chat session mutex poisoned (a prior call panicked mid-lock; session state is \
                 inconsistent): {e}"
            ),
        })
    }

    fn with_chat<R>(
        &self,
        f: impl FnOnce(&mut cera::session::chat::SessionChat) -> Result<R, FfiError>,
    ) -> Result<R, FfiError> {
        let mut guard = self.lock_inner()?;
        let chat = guard.as_mut().ok_or_else(|| FfiError::Backend {
            detail: "chat session has been moved back into a Session".into(),
        })?;
        f(chat)
    }

    fn set_last_ingest_recovery(&self, recovery: Option<crate::IngestRecovery>) {
        let mut guard = self
            .last_ingest_recovery
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = recovery;
    }
}

#[uniffi::export]
impl ChatSession {
    /// Construct a ChatSession from an existing Session, taking ownership of its state.
    ///
    /// If validation fails, the session remains intact and usable on the caller side.
    #[uniffi::constructor]
    pub fn from_session(session: &Session) -> Result<Arc<Self>, FfiError> {
        let mut guard = session
            .inner_mutex()
            .lock()
            .map_err(|e| FfiError::Backend {
                detail: format!("session mutex poisoned: {e}"),
            })?;
        let core_session = guard.take().ok_or_else(|| FfiError::Backend {
            detail: "session has already been moved into a ChatSession".into(),
        })?;
        let position = core_session.position_handle();
        match cera::session::chat::core_chat(core_session) {
            Ok(chat) => {
                let cancel = chat
                    .cancel_handle()
                    .unwrap_or_else(|| Arc::new(std::sync::atomic::AtomicBool::new(false)));
                Ok(Arc::new(Self {
                    inner: std::sync::Mutex::new(Some(chat)),
                    cancel,
                    position,
                    moved: std::sync::atomic::AtomicBool::new(false),
                    last_ingest_recovery: std::sync::Mutex::new(None),
                }))
            }
            Err((returned_session, validation_err)) => {
                *guard = Some(returned_session);
                Err(FfiError::from(validation_err))
            }
        }
    }

    /// Current session lifecycle phase.
    ///
    /// Non-blocking observation; returns `FfiError::Busy` if another operation is active.
    pub fn phase(&self) -> Result<SessionPhase, FfiError> {
        let guard = self.inner.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => FfiError::Busy,
            std::sync::TryLockError::Poisoned(_) => FfiError::Backend {
                detail: "chat session mutex poisoned; recreate the session".into(),
            },
        })?;
        let chat = guard.as_ref().ok_or_else(|| FfiError::Backend {
            detail: "chat session has been moved back into a Session".into(),
        })?;
        Ok(chat.phase().into())
    }

    /// Current token position in the execution context.
    ///
    /// Lock-free and safe to query concurrently while generation is in flight.
    pub fn position(&self) -> Result<u32, FfiError> {
        if self.moved.load(Ordering::Acquire) {
            return Err(FfiError::Backend {
                detail: "chat session has been moved back into a Session".into(),
            });
        }
        Ok(self.position.load(Ordering::Relaxed))
    }

    /// Ingest a single message into the chat context.
    ///
    /// Single-message ingestion requires a user message to trigger assistant turn
    /// completion. To start a multi-turn conversation with a system prompt, supply both
    /// messages via [`ChatSession::ingest_messages`].
    pub fn ingest(&self, message: Message) -> Result<IngestSummary, FfiError> {
        let core_msg = cera::session::chat::Message::from(message);
        self.with_chat(|chat| match chat.ingest(&core_msg) {
            Ok(summary) => {
                self.set_last_ingest_recovery(None);
                Ok(summary.into())
            }
            Err(err) => {
                self.set_last_ingest_recovery(Some(crate::IngestRecovery {
                    outcome: err.recovery.into(),
                    rewind_error: err.rewind_error.as_deref().map(Into::into),
                    reset_error: err.recovery_error.as_ref().map(Into::into),
                }));
                Err(match err.cause {
                    cera::session::chat::IngestCause::Validation(val) => FfiError::from(val),
                    cera::session::chat::IngestCause::Execution(exec) => FfiError::from(exec),
                })
            }
        })
    }

    /// Ingest a batch of messages into the chat context.
    pub fn ingest_messages(&self, messages: Vec<Message>) -> Result<IngestSummary, FfiError> {
        let core_msgs: Vec<cera::session::chat::Message> =
            messages.into_iter().map(Into::into).collect();
        self.with_chat(|chat| match chat.ingest_messages(&core_msgs) {
            Ok(summary) => {
                self.set_last_ingest_recovery(None);
                Ok(summary.into())
            }
            Err(err) => {
                self.set_last_ingest_recovery(Some(crate::IngestRecovery {
                    outcome: err.recovery.into(),
                    rewind_error: err.rewind_error.as_deref().map(Into::into),
                    reset_error: err.recovery_error.as_ref().map(Into::into),
                }));
                Err(match err.cause {
                    cera::session::chat::IngestCause::Validation(val) => FfiError::from(val),
                    cera::session::chat::IngestCause::Execution(exec) => FfiError::from(exec),
                })
            }
        })
    }

    /// Replace conversational history with a fresh message batch.
    pub fn replace_messages(&self, messages: Vec<Message>) -> Result<IngestSummary, FfiError> {
        let core_msgs: Vec<cera::session::chat::Message> =
            messages.into_iter().map(Into::into).collect();
        self.with_chat(|chat| match chat.replace_messages(&core_msgs) {
            Ok(summary) => {
                self.set_last_ingest_recovery(None);
                Ok(summary.into())
            }
            Err(err) => {
                self.set_last_ingest_recovery(Some(crate::IngestRecovery {
                    outcome: err.recovery.into(),
                    rewind_error: err.rewind_error.as_deref().map(Into::into),
                    reset_error: err.recovery_error.as_ref().map(Into::into),
                }));
                Err(match err.cause {
                    cera::session::chat::IngestCause::Validation(val) => FfiError::from(val),
                    cera::session::chat::IngestCause::Execution(exec) => FfiError::from(exec),
                })
            }
        })
    }

    /// Complete generation synchronously and return the assistant response.
    pub fn complete(&self, opts: GenerateOpts) -> Result<TurnResult, FfiError> {
        let core_opts = cera::GenerateOpts::try_from(opts)?;
        self.with_chat(|chat| {
            let result = chat.complete(&core_opts).map_err(|err| match err {
                cera::session::chat::CompleteError::Validation(val) => FfiError::from(val),
                cera::session::chat::CompleteError::Execution(exec) => FfiError::from(exec),
            })?;
            Ok(result.into())
        })
    }

    /// Stream generation output tokens into the specified sink.
    pub fn generate_streaming(
        &self,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let core = match cera::GenerateOpts::try_from(opts) {
            Ok(core) => core,
            Err(err) => {
                sink.on_done(FinishReason::Error {
                    message: err.to_string(),
                });
                return Err(err);
            }
        };
        let (mut adapter, report_result) = {
            let mut guard = match self.lock_inner() {
                Ok(guard) => guard,
                Err(err) => {
                    sink.on_done(FinishReason::Error {
                        message: err.to_string(),
                    });
                    return Err(err);
                }
            };
            let chat = match guard.as_mut() {
                Some(chat) => chat,
                None => {
                    let err = FfiError::Backend {
                        detail: "chat session has been moved back into a Session".into(),
                    };
                    sink.on_done(FinishReason::Error {
                        message: err.to_string(),
                    });
                    return Err(err);
                }
            };
            let tokenizer = chat.profile().tokenizer().clone();
            let mut adapter = ForeignSinkAdapter::new(sink, tokenizer);
            let report_result = chat
                .generate_into(&core, &mut adapter)
                .map_err(FfiError::from);
            (adapter, report_result)
        };
        match report_result {
            Ok(report) => match report.result {
                Ok(summary) => {
                    adapter.notify_done(None);
                    Ok(summary.into())
                }
                Err(cera_err) => {
                    let err = FfiError::from(cera_err);
                    if !adapter.done_called {
                        adapter.flush_pending();
                        let finish_reason = match &err {
                            FfiError::Cancelled => FinishReason::Cancelled,
                            _ => FinishReason::Error {
                                message: err.to_string(),
                            },
                        };
                        adapter.notify_done(Some(finish_reason));
                    } else {
                        adapter.notify_done(None);
                    }
                    Err(err)
                }
            },
            Err(val_err) => {
                if !adapter.done_called {
                    adapter.flush_pending();
                    adapter.notify_done(Some(FinishReason::Error {
                        message: val_err.to_string(),
                    }));
                } else {
                    adapter.notify_done(None);
                }
                Err(val_err)
            }
        }
    }

    /// Complete generation synchronously constrained by a JSON Schema.
    pub fn complete_json(
        &self,
        opts: GenerateOpts,
        schema_json: String,
    ) -> Result<TurnResult, FfiError> {
        let grammar_str = cera::grammar::json_schema_to_gbnf_str(&schema_json).map_err(|e| {
            FfiError::GrammarParse {
                detail: format!("invalid JSON schema: {e}"),
            }
        })?;
        let mut constrained_opts = opts;
        constrained_opts.grammar = Some(grammar_str);
        self.complete(constrained_opts)
    }

    /// Stream generation output tokens into the specified sink, constrained by a JSON Schema.
    pub fn generate_streaming_json(
        &self,
        opts: GenerateOpts,
        schema_json: String,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let grammar_str = cera::grammar::json_schema_to_gbnf_str(&schema_json).map_err(|e| {
            FfiError::GrammarParse {
                detail: format!("invalid JSON schema: {e}"),
            }
        })?;
        let mut constrained_opts = opts;
        constrained_opts.grammar = Some(grammar_str);
        self.generate_streaming(constrained_opts, sink)
    }

    /// Register tools for function calling.
    pub fn set_tools(&self, tools: Vec<ToolDef>) -> Result<(), FfiError> {
        let core_tools = crate::to_core_tools(tools)?;
        self.with_chat(|chat| {
            chat.set_tools(core_tools);
            Ok(())
        })
    }

    /// Currently registered tools for function calling.
    pub fn tools(&self) -> Result<Vec<ToolDef>, FfiError> {
        self.with_chat(|chat| Ok(chat.tools().iter().cloned().map(Into::into).collect()))
    }

    /// Set tool wire format explicitly.
    pub fn set_tool_format(&self, format: ToolFormat) -> Result<(), FfiError> {
        self.with_chat(|chat| {
            chat.set_tool_format(format.into());
            Ok(())
        })
    }

    /// Current tool wire format.
    pub fn tool_format(&self) -> Result<ToolFormat, FfiError> {
        self.with_chat(|chat| Ok(chat.tool_format().into()))
    }

    /// Ingest a tool execution response back into the conversation.
    pub fn ingest_tool_response(
        &self,
        name: String,
        content: String,
    ) -> Result<IngestSummary, FfiError> {
        self.with_chat(|chat| match chat.ingest_tool_response(&name, &content) {
            Ok(summary) => {
                self.set_last_ingest_recovery(None);
                Ok(summary.into())
            }
            Err(err) => {
                self.set_last_ingest_recovery(Some(crate::IngestRecovery {
                    outcome: err.recovery.into(),
                    rewind_error: err.rewind_error.as_deref().map(Into::into),
                    reset_error: err.recovery_error.as_ref().map(Into::into),
                }));
                Err(match err.cause {
                    cera::session::chat::IngestCause::Validation(val) => FfiError::from(val),
                    cera::session::chat::IngestCause::Execution(exec) => FfiError::from(exec),
                })
            }
        })
    }

    /// Reset execution state and return to Idle phase.
    pub fn reset(&self) -> Result<(), FfiError> {
        self.with_chat(|chat| {
            self.set_last_ingest_recovery(None);
            chat.reset().map_err(FfiError::from)
        })
    }

    /// Flip cancellation flag to interrupt in-flight prefill or decode.
    ///
    /// Wait-free and safe from any thread. If the session has already been reclaimed
    /// via `into_session()`, this call is a no-op to prevent cross-session cancellation.
    pub fn cancel(&self) {
        if self.moved.load(Ordering::Acquire) {
            return;
        }
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Clear pending cancellation.
    pub fn clear_cancel(&self) -> Result<(), FfiError> {
        if self.moved.load(Ordering::Acquire) {
            return Err(FfiError::Backend {
                detail: "chat session has been moved back into a Session".into(),
            });
        }
        self.cancel.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Observe recovery status after an ingestion failure.
    ///
    /// Non-blocking observation; returns `FfiError::Busy` if another operation is active.
    pub fn recovery_status(&self) -> Result<SessionRecoveryStatus, FfiError> {
        let guard = self.inner.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => FfiError::Busy,
            std::sync::TryLockError::Poisoned(_) => FfiError::Backend {
                detail: "chat session mutex poisoned; recreate the session".into(),
            },
        })?;
        let chat = guard.as_ref().ok_or_else(|| FfiError::Backend {
            detail: "chat session has been moved back into a Session".into(),
        })?;
        let session = chat.session();
        let ingest_recovery = self
            .last_ingest_recovery
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        Ok(SessionRecoveryStatus {
            usable: session.is_usable(),
            position: session.position(),
            last_ingest_recovery: ingest_recovery
                .or_else(|| session.last_ingest_recovery().map(Into::into)),
        })
    }

    /// Reclaim the underlying Session, consuming this ChatSession.
    pub fn into_session(&self) -> Result<Arc<Session>, FfiError> {
        let mut guard = self.lock_inner()?;
        let chat = guard.take().ok_or_else(|| FfiError::Backend {
            detail: "chat session has already been moved into a Session".into(),
        })?;
        self.moved.store(true, Ordering::Release);
        self.cancel.store(false, Ordering::Relaxed);
        let session = chat.into_session();
        Ok(Session::from_core(session))
    }
}

struct AsyncChatCancelGuard {
    chat: Arc<ChatSession>,
    abort: tokio::task::AbortHandle,
    armed: bool,
}

impl Drop for AsyncChatCancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.abort.abort();
            self.chat.cancel();
        }
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl ChatSession {
    /// Async variant of [`ChatSession::complete`].
    pub async fn complete_async(
        self: Arc<Self>,
        opts: GenerateOpts,
    ) -> Result<TurnResult, FfiError> {
        let chat_for_guard = Arc::clone(&self);
        let handle = tokio::task::spawn_blocking(move || self.complete(opts));
        let mut guard = AsyncChatCancelGuard {
            chat: chat_for_guard,
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result.map_err(|e| FfiError::Backend {
            detail: format!("complete_async join error: {e}"),
        })?
    }

    /// Async variant of [`ChatSession::generate_streaming`].
    pub async fn generate_streaming_async(
        self: Arc<Self>,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let chat_for_guard = Arc::clone(&self);
        let handle = tokio::task::spawn_blocking(move || self.generate_streaming(opts, sink));
        let mut guard = AsyncChatCancelGuard {
            chat: chat_for_guard,
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result.map_err(|e| FfiError::Backend {
            detail: format!("generate_streaming_async join error: {e}"),
        })?
    }

    /// Async variant of [`ChatSession::complete_json`].
    pub async fn complete_async_json(
        self: Arc<Self>,
        opts: GenerateOpts,
        schema_json: String,
    ) -> Result<TurnResult, FfiError> {
        let grammar_str = cera::grammar::json_schema_to_gbnf_str(&schema_json).map_err(|e| {
            FfiError::GrammarParse {
                detail: format!("invalid JSON schema: {e}"),
            }
        })?;
        let mut constrained_opts = opts;
        constrained_opts.grammar = Some(grammar_str);
        self.complete_async(constrained_opts).await
    }

    /// Async variant of [`ChatSession::generate_streaming_json`].
    pub async fn generate_streaming_async_json(
        self: Arc<Self>,
        opts: GenerateOpts,
        schema_json: String,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let grammar_str = cera::grammar::json_schema_to_gbnf_str(&schema_json).map_err(|e| {
            FfiError::GrammarParse {
                detail: format!("invalid JSON schema: {e}"),
            }
        })?;
        let mut constrained_opts = opts;
        constrained_opts.grammar = Some(grammar_str);
        self.generate_streaming_async(constrained_opts, sink).await
    }
}

#[cfg(test)]
mod tests;

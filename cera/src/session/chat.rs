//! Stateful, multi-turn conversational chat coordinator.
//!
//! This module provides a transactional chat interface over [`Session`],
//! managing turn framing, chat template rendering, phase transitions, and
//! KV cache reuse across conversational turns.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};

use thiserror::Error;

use super::{
    CeraError, DecodeObservation, FinishReason, GenerateOpts, GenerateSummary, IngestRecovery,
    ModalitySink, RecoveryOutcome, Session,
};
#[cfg(test)]
use crate as core_api;
use crate::kv_cache::KvRewindError;
use crate::model::Model;
use crate::tokenizer::{BpeTokenizer, ChatMessage, apply_chat_template};

/// ChatML chat template used by LFM2 models.
pub const TEMPLATE: &str = "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// ChatML chat template used by LFM2.5 models with thinking support.
pub const LFM2_5_TEMPLATE: &str = "{{- bos_token -}}\n{%- set keep_past_thinking = keep_past_thinking | default(false) -%}\n{%- set ns = namespace(system_prompt=\"\") -%}\n{%- if messages[0][\"role\"] == \"system\" -%}\n    {%- set sys_content = messages[0][\"content\"] -%}\n    {%- if sys_content is not string -%}\n        {%- for item in sys_content -%}\n            {%- if item[\"type\"] == \"text\" -%}\n                {%- set ns.system_prompt = ns.system_prompt + item[\"text\"] -%}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {%- set ns.system_prompt = sys_content -%}\n    {%- endif -%}\n    {%- set messages = messages[1:] -%}\n{%- endif -%}\n{%- if tools -%}\n    {%- set ns.system_prompt = ns.system_prompt + (\"\\n\" if ns.system_prompt else \"\") + \"List of tools: [\" -%}\n    {%- for tool in tools -%}\n        {%- if tool is not string -%}\n            {%- set tool = tool | tojson -%}\n        {%- endif -%}\n        {%- set ns.system_prompt = ns.system_prompt + tool -%}\n        {%- if not loop.last -%}\n            {%- set ns.system_prompt = ns.system_prompt + \", \" -%}\n        {%- endif -%}\n    {%- endfor -%}\n    {%- set ns.system_prompt = ns.system_prompt + \"]\" -%}\n{%- endif -%}\n{%- if ns.system_prompt -%}\n    {{- \"<|im_start|>system\\n\" + ns.system_prompt + \"<|im_end|>\\n\" -}}\n{%- endif -%}\n{%- set ns.last_assistant_index = -1 -%}\n{%- for message in messages -%}\n    {%- if message[\"role\"] == \"assistant\" -%}\n        {%- set ns.last_assistant_index = loop.index0 -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- for message in messages -%}\n    {{- \"<|im_start|>\" + message[\"role\"] + \"\\n\" -}}\n    {%- set content = message[\"content\"] -%}\n    {%- if content is not string -%}\n        {%- set ns.content = \"\" -%}\n        {%- for item in content -%}\n            {%- if item[\"type\"] == \"image\" -%}\n                {%- set ns.content = ns.content + \"<image>\" -%}\n            {%- elif item[\"type\"] == \"text\" -%}\n                {%- set ns.content = ns.content + item[\"text\"] -%}\n            {%- else -%}\n                {%- set ns.content = ns.content + item | tojson -%}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- set content = ns.content -%}\n    {%- endif -%}\n    {%- if message[\"role\"] == \"assistant\" and not keep_past_thinking and loop.index0 != ns.last_assistant_index -%}\n        {%- if \"</think>\" in content -%}\n            {%- set content = content.split(\"</think>\")[-1] | trim -%}\n        {%- endif -%}\n    {%- endif -%}\n    {{- content + \"<|im_end|>\\n\" -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- \"<|im_start|>assistant\\n\" -}}\n{%- endif -%}";

/// Beginning-of-sequence special token text.
pub const BOS: &str = "<|startoftext|>";

/// End-of-sequence special token text.
pub const END: &str = "<|im_end|>";

/// Message author role in conversational chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Role {
    /// String identifier of this role.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Content piece within a conversational message.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    /// Text message chunk.
    Text(String),
    /// Image payload encoded as byte buffer.
    Image(Vec<u8>),
    /// Audio PCM waveform sample buffer.
    Audio { pcm: Vec<f32>, sample_rate: u32 },
}

/// A structured conversational turn message.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// Author role.
    pub role: Role,
    /// Ordered sequence of content parts.
    pub content: Vec<ContentPart>,
}

impl Message {
    /// Create a message with the specified role and content parts.
    pub fn with_parts(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    /// Create a single-part text message for the specified role.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    /// Create a single-part user text message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    /// Create a single-part system text message.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(Role::System, text)
    }

    /// Create a single-part assistant text message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    /// Create a single-part tool text message.
    pub fn tool(text: impl Into<String>) -> Self {
        Self::text(Role::Tool, text)
    }
}

/// Lifecycle phase of a stateful chat coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl SessionPhase {
    /// String representation of the phase.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::PromptReady => "PromptReady",
            Self::TurnComplete => "TurnComplete",
            Self::Interrupted => "Interrupted",
            Self::RawContext => "RawContext",
            Self::Unusable => "Unusable",
        }
    }
}

impl std::fmt::Display for SessionPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Validation failure during chat construction, preparation, or decode.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError {
    /// Model or tokenizer configuration does not match a supported chat profile.
    #[error("unsupported chat template profile")]
    UnsupportedProfile,
    /// Sliding context configuration (`n_keep != 0`) is not supported for chat.
    #[error("sliding context (n_keep != 0) is not supported for chat")]
    SlidingContext,
    /// Model audio output is not supported for text chat.
    #[error("audio output is not supported for text chat")]
    AudioOutput,
    /// Generation parameter validation error.
    #[error("generation error: {0}")]
    Generation(String),
    /// Operation refused in the current session phase.
    #[error("invalid session phase: {0:?}")]
    Phase(SessionPhase),
    /// Message batch provided to ingest was empty.
    #[error("empty message batch")]
    EmptyBatch,
    /// Message role sequence violates chat rules (e.g. system not first, or missing user turn).
    #[error("invalid role order at message index {message}")]
    RoleOrder { message: usize },
    /// Message role is not supported in the active profile.
    #[error("unsupported role at message index {message}")]
    UnsupportedRole { message: usize },
    /// Content part is not supported in the active profile.
    #[error("unsupported content part {part} in message index {message}")]
    UnsupportedContent { message: usize, part: usize },
    /// Message text contains a reserved ChatML marker sequence (`<|`).
    #[error("reserved marker found in message index {message}")]
    ReservedMarker { message: usize },
    /// Context tokens required exceed available capacity in the KV cache.
    #[error("required context capacity {required} exceeds available capacity {available}")]
    Capacity { required: usize, available: usize },
    /// Chat template rendering error.
    #[error("template error: {0}")]
    Template(String),
}

/// Cause of an ingest failure.
#[derive(Debug, Error)]
pub enum IngestCause {
    /// Pre-mutation or format validation error.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// Runtime execution error during token prefill.
    #[error(transparent)]
    Execution(#[from] CeraError),
}

/// Error returned when message ingestion fails.
#[derive(Debug, Error)]
#[error("ingest failed ({cause}), recovery outcome: {recovery:?}")]
pub struct IngestError {
    /// Underlying validation or execution cause.
    pub cause: IngestCause,
    /// Recovery state established after the error.
    pub recovery: RecoveryOutcome,
    /// Diagnostic from KV rewind if attempted.
    pub rewind_error: Option<Box<KvRewindError>>,
    /// Secondary error if checked reset was attempted during recovery.
    pub recovery_error: Option<CeraError>,
}

impl From<ValidationError> for IngestError {
    fn from(error: ValidationError) -> Self {
        Self {
            cause: IngestCause::Validation(error),
            recovery: RecoveryOutcome::Unchanged,
            rewind_error: None,
            recovery_error: None,
        }
    }
}

/// Summary of a successful message ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IngestSummary {
    /// Number of tokens encoded and appended to the context.
    pub input_tokens: usize,
    /// KV position before ingestion.
    pub position_before: usize,
    /// KV position after ingestion.
    pub position_after: usize,
}

/// Observed state transition during decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeState {
    /// No progress was made (e.g. zero tokens generated or pre-step cancel).
    NoProgress,
    /// Terminal stop token was encountered.
    Terminal { token: u32, committed: bool },
    /// Decode was interrupted by cancellation or custom stop condition.
    Interrupted,
    /// Execution state was left uncertified or invalid.
    Unusable,
}

/// Full execution report from a decode pass.
#[derive(Debug)]
pub struct DecodeReport {
    /// Result of the generation pass.
    pub result: Result<GenerateSummary, CeraError>,
    /// Observed decode state transition.
    pub state: DecodeState,
}

/// Execution backend contract for chat coordination.
pub trait Execution: std::fmt::Debug {
    /// Current token position in the context.
    fn position(&self) -> usize;
    /// Maximum context capacity.
    fn capacity(&self) -> usize;
    /// Whether audio output decoding is active.
    fn audio_output(&self) -> bool;
    /// Pre-mutation validation before preparing context.
    fn validate_prepare(&self) -> Result<(), ValidationError> {
        Ok(())
    }
    /// Validation before decode starts.
    fn validate_decode(&self, opts: &GenerateOpts) -> Result<(), ValidationError>;
    /// Append a batch of tokens atomically to context.
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError>;
    /// Checked reset of execution state.
    fn reset(&mut self, explicit: bool) -> Result<(), CeraError>;
    /// Execute decode pass driving token generation into sink.
    fn decode(&mut self, opts: &GenerateOpts, sink: &mut dyn ModalitySink) -> DecodeReport;
    /// Handle to external cancellation atomic flag.
    fn cancel_handle(&self) -> Option<Arc<AtomicBool>> {
        None
    }
    /// Trigger cancellation.
    fn cancel(&self) {}
    /// Clear pending cancellation.
    fn clear_cancel(&mut self) {}
}

/// Result of a completed chat turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnResult {
    /// Decoded assistant response text.
    pub text: String,
    /// Token identifiers emitted during the turn.
    pub tokens: Vec<u32>,
    /// Generation summary metrics.
    pub summary: GenerateSummary,
}

/// Error returned by `complete()`.
#[derive(Debug, Error)]
pub enum CompleteError {
    /// Validation error prior to decode.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// Execution error during decode.
    #[error(transparent)]
    Execution(#[from] CeraError),
}

#[derive(Default)]
struct Collector(Vec<u32>);

impl Collector {
    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }
}

impl ModalitySink for Collector {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.0.extend_from_slice(tokens);
    }
    fn on_done(&mut self, _: FinishReason) {}
}

/// Chat rendering profile holding tokenizer and template rules.
#[derive(Clone)]
pub struct Profile {
    tokenizer: Arc<BpeTokenizer>,
    eos: u32,
    newline_tokens: Vec<u32>,
}

impl std::fmt::Debug for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Profile")
            .field("eos", &self.eos)
            .field("newline_tokens", &self.newline_tokens)
            .finish()
    }
}

impl Profile {
    /// Discover and validate a chat profile from a tokenizer's metadata.
    pub fn discover(tokenizer: Arc<BpeTokenizer>) -> Result<Self, ValidationError> {
        let template_ok = matches!(
            tokenizer.chat_template(),
            Some(t) if t == TEMPLATE || t == LFM2_5_TEMPLATE
        );
        if !template_ok
            || tokenizer.bos_token() != Some(1)
            || tokenizer.eos_token() != Some(7)
            || tokenizer.encode(BOS) != [1]
            || tokenizer.encode(END) != [7]
            || tokenizer.encode("<|im_start|>") != [6]
        {
            return Err(ValidationError::UnsupportedProfile);
        }
        let newline_tokens = tokenizer.encode("\n");
        Ok(Self {
            tokenizer,
            eos: 7,
            newline_tokens,
        })
    }

    /// End-of-sequence token ID.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Pre-encoded newline token sequence for turn boundaries.
    pub fn newline_tokens(&self) -> &[u32] {
        &self.newline_tokens
    }

    /// Reference to the tokenizer used by this profile.
    pub fn tokenizer(&self) -> &Arc<BpeTokenizer> {
        &self.tokenizer
    }

    fn render(&self, messages: &[Message], initial: bool) -> Result<String, ValidationError> {
        if messages.is_empty() {
            return Err(ValidationError::EmptyBatch);
        }
        let mut serialized = Vec::with_capacity(messages.len());
        let mut expected = Role::User;
        for (index, message) in messages.iter().enumerate() {
            if message.role == Role::Tool {
                return Err(ValidationError::UnsupportedRole { message: index });
            }
            let role = if initial && index == 0 && message.role == Role::System {
                "system"
            } else {
                match message.role {
                    Role::User if expected == Role::User => {
                        expected = Role::Assistant;
                        "user"
                    }
                    Role::Assistant if expected == Role::Assistant => {
                        expected = Role::User;
                        "assistant"
                    }
                    _ => return Err(ValidationError::RoleOrder { message: index }),
                }
            };
            let mut text = String::new();
            for (part, content) in message.content.iter().enumerate() {
                match content {
                    ContentPart::Text(value) => text.push_str(value),
                    _ => {
                        return Err(ValidationError::UnsupportedContent {
                            message: index,
                            part,
                        });
                    }
                }
            }
            if text.contains("<|") {
                return Err(ValidationError::ReservedMarker { message: index });
            }
            serialized.push(ChatMessage {
                role: role.into(),
                content: text,
            });
        }
        if messages.last().is_none_or(|msg| msg.role != Role::User) {
            return Err(ValidationError::RoleOrder {
                message: messages.len().saturating_sub(1),
            });
        }
        let rendered = apply_chat_template(&self.tokenizer, &serialized, true)
            .map_err(|e| ValidationError::Template(e.to_string()))?;
        if initial {
            Ok(rendered)
        } else if rendered.starts_with(BOS) {
            let mut s = rendered;
            s.drain(..BOS.len());
            Ok(s)
        } else {
            Err(ValidationError::UnsupportedProfile)
        }
    }
}

/// Stateful chat coordinator over an execution engine.
pub struct Chat<E = CoreExecution> {
    execution: E,
    profile: Profile,
    phase: SessionPhase,
    terminal_committed: Option<bool>,
}

impl<E: Execution> std::fmt::Debug for Chat<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chat")
            .field("execution", &self.execution)
            .field("phase", &self.phase)
            .finish()
    }
}

impl<E: Execution> Chat<E> {
    /// Create a new chat session wrapping the given execution engine and profile.
    pub fn new(execution: E, profile: Profile, n_keep: u32) -> Result<Self, (E, ValidationError)> {
        if let Err(err) = execution.validate_prepare() {
            return Err((execution, err));
        }
        if n_keep != 0 {
            return Err((execution, ValidationError::SlidingContext));
        }
        if execution.audio_output() {
            return Err((execution, ValidationError::AudioOutput));
        }
        let phase = if execution.position() == 0 {
            SessionPhase::Idle
        } else {
            SessionPhase::RawContext
        };
        Ok(Self {
            execution,
            profile,
            phase,
            terminal_committed: None,
        })
    }

    /// Extract the underlying execution engine.
    pub fn into_inner(self) -> E {
        self.execution
    }

    /// Shared handle to the cancellation flag, if supported.
    pub fn cancel_handle(&self) -> Option<Arc<AtomicBool>> {
        self.execution.cancel_handle()
    }

    /// Flip the cancellation flag to interrupt ongoing prefill or decode.
    pub fn cancel(&self) {
        self.execution.cancel();
    }

    /// Clear pending cancellation non-destructively.
    pub fn clear_cancel(&mut self) {
        self.execution.clear_cancel();
    }

    /// Current session lifecycle phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Active chat rendering profile.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Current token cursor position in the execution engine.
    pub fn position(&self) -> usize {
        self.execution.position()
    }

    #[doc(hidden)]
    pub fn execution_for_test(&mut self) -> &mut E {
        &mut self.execution
    }

    /// Ingest a single message into the chat context.
    pub fn ingest(&mut self, message: &Message) -> Result<IngestSummary, IngestError> {
        self.ingest_messages(std::slice::from_ref(message))
    }

    /// Ingest a batch of messages into the chat context.
    pub fn ingest_messages(&mut self, messages: &[Message]) -> Result<IngestSummary, IngestError> {
        if !matches!(self.phase, SessionPhase::Idle | SessionPhase::TurnComplete) {
            return Err(ValidationError::Phase(self.phase).into());
        }
        self.prepare(messages, false)
    }

    /// Replace the conversational history with a fresh message batch.
    pub fn replace_messages(&mut self, messages: &[Message]) -> Result<IngestSummary, IngestError> {
        if self.phase == SessionPhase::Unusable {
            return Err(ValidationError::Phase(self.phase).into());
        }
        self.prepare(messages, true)
    }

    fn prepare(
        &mut self,
        messages: &[Message],
        replace: bool,
    ) -> Result<IngestSummary, IngestError> {
        self.execution.validate_prepare()?;
        if self.execution.audio_output() {
            return Err(ValidationError::AudioOutput.into());
        }
        let before = self.position();
        let initial = replace || self.phase == SessionPhase::Idle;
        let rendered = self.profile.render(messages, initial)?;
        let mut tokens = Vec::new();
        if !initial {
            if self.terminal_committed == Some(false) {
                tokens.push(self.profile.eos);
            }
            tokens.extend_from_slice(&self.profile.newline_tokens);
        }
        tokens.extend(self.profile.tokenizer.encode(&rendered));
        let available = self
            .execution
            .capacity()
            .saturating_sub(if replace { 0 } else { before });
        if tokens.len() > available {
            return Err(ValidationError::Capacity {
                required: tokens.len(),
                available,
            }
            .into());
        }
        // Enter Unusable phase before mutating execution state.
        // If an append fails or panics, the session remains safely marked Unusable
        // unless a recovery path restores consistency.
        let prior_phase = self.phase;
        let prior_terminal = self.terminal_committed;
        self.phase = SessionPhase::Unusable;
        self.terminal_committed = None;
        if replace && let Err(error) = self.execution.reset(false) {
            return Err(IngestError {
                cause: IngestCause::Execution(error),
                recovery: RecoveryOutcome::Unusable,
                rewind_error: None,
                recovery_error: None,
            });
        }
        match self.execution.append(&tokens) {
            Ok(()) => {
                self.phase = SessionPhase::PromptReady;
                self.terminal_committed = None;
                Ok(IngestSummary {
                    input_tokens: tokens.len(),
                    position_before: before,
                    position_after: self.position(),
                })
            }
            Err(mut error) => {
                if replace
                    && matches!(
                        error.recovery,
                        RecoveryOutcome::Unchanged | RecoveryOutcome::Restored
                    )
                {
                    error.recovery = RecoveryOutcome::Reset;
                }
                match error.recovery {
                    RecoveryOutcome::Unchanged | RecoveryOutcome::Restored => {
                        self.phase = prior_phase;
                        self.terminal_committed = prior_terminal;
                    }
                    RecoveryOutcome::Reset => {
                        self.phase = SessionPhase::Idle;
                        self.terminal_committed = None;
                    }
                    _ => {
                        self.phase = SessionPhase::Unusable;
                        self.terminal_committed = None;
                    }
                }
                Err(error)
            }
        }
    }

    /// Stream generation output tokens into the specified sink.
    pub fn generate_into(
        &mut self,
        opts: &GenerateOpts,
        sink: &mut dyn ModalitySink,
    ) -> Result<DecodeReport, ValidationError> {
        if self.phase != SessionPhase::PromptReady {
            return Err(ValidationError::Phase(self.phase));
        }
        self.execution.validate_decode(opts)?;
        if self.execution.audio_output() {
            return Err(ValidationError::AudioOutput);
        }
        self.phase = SessionPhase::Unusable;
        let report = self.execution.decode(opts, sink);
        self.phase = match report.state {
            // A zero-token generation or pre-step cancel leaves the prompt ready for retry.
            // An execution error marks the session RawContext for inspection.
            DecodeState::NoProgress => match &report.result {
                Ok(summary) if summary.tokens_generated == 0 => SessionPhase::PromptReady,
                Err(_) => SessionPhase::RawContext,
                Ok(_) => SessionPhase::Unusable,
            },
            DecodeState::Terminal { token, committed }
                if token == self.profile.eos
                    && report
                        .result
                        .as_ref()
                        .is_ok_and(|r| r.finish_reason == FinishReason::Stop) =>
            {
                self.terminal_committed = Some(committed);
                SessionPhase::TurnComplete
            }
            DecodeState::Terminal { .. } | DecodeState::Interrupted => SessionPhase::Interrupted,
            DecodeState::Unusable => SessionPhase::Unusable,
        };
        Ok(report)
    }

    /// Complete generation synchronously and return the assistant response.
    pub fn complete(&mut self, opts: &GenerateOpts) -> Result<TurnResult, CompleteError> {
        let capacity = (opts.max_tokens as usize).min(4096);
        let mut collector = Collector::with_capacity(capacity);
        let report = self
            .generate_into(opts, &mut collector)
            .map_err(CompleteError::Validation)?;
        let summary = report.result.map_err(CompleteError::Execution)?;
        Ok(TurnResult {
            text: self.profile.tokenizer.decode(&collector.0),
            tokens: collector.0,
            summary,
        })
    }

    /// Reset execution state and return to `SessionPhase::Idle`.
    pub fn reset(&mut self) -> Result<(), CeraError> {
        self.phase = SessionPhase::Unusable;
        self.terminal_committed = None;
        self.execution.reset(true)?;
        self.phase = SessionPhase::Idle;
        Ok(())
    }

    /// Escape to the underlying execution engine, marking the phase `RawContext`.
    pub fn raw(&mut self) -> Result<&mut E, ValidationError> {
        if self.phase == SessionPhase::Unusable {
            return Err(ValidationError::Phase(self.phase));
        }
        self.phase = SessionPhase::RawContext;
        self.terminal_committed = None;
        Ok(&mut self.execution)
    }
}

/// Execution adapter connecting a [`Session`] to the [`Execution`] trait.
pub struct CoreExecution {
    /// The wrapped inference session.
    pub session: Session,
    tokenizer: Arc<BpeTokenizer>,
    // Weak pointer to model prevents circular references while allowing
    // pointer equality checks against session.model to verify structural identity.
    model: Weak<dyn Model>,
}

/// Chat coordinator using standard Session inference execution.
pub type SessionChat = Chat<CoreExecution>;

impl Chat<CoreExecution> {
    /// Reclaim ownership of the underlying [`Session`].
    pub fn into_session(self) -> Session {
        self.into_inner().into_session()
    }

    /// Borrow the underlying [`Session`].
    pub fn session(&self) -> &Session {
        self.execution.session()
    }
}

/// Wrap a [`Session`] in a [`Chat`] coordinator.
///
/// On failure, the original session is returned intact along with the validation error.
#[allow(clippy::result_large_err)]
pub fn core_chat(session: Session) -> Result<SessionChat, (Session, ValidationError)> {
    let tokenizer = session.tokenizer_arc();
    let profile = match Profile::discover(tokenizer.clone()) {
        Ok(p) => p,
        Err(err) => return Err((session, err)),
    };
    let keep = session.config.n_keep;
    let execution = CoreExecution {
        tokenizer,
        model: Arc::downgrade(&session.model),
        session,
    };
    Chat::new(execution, profile, keep).map_err(|(exec, err)| (exec.into_session(), err))
}

impl std::fmt::Debug for CoreExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreExecution")
            .field("pos", &self.session.current_pos)
            .field("usable", &self.session.usable)
            .finish()
    }
}

impl CoreExecution {
    /// Consume this adapter and return the underlying [`Session`].
    pub fn into_session(self) -> Session {
        self.session
    }

    /// Borrow the underlying [`Session`].
    pub fn session(&self) -> &Session {
        &self.session
    }

    fn validate_identity(&self) -> Result<(), ValidationError> {
        if !Arc::ptr_eq(&self.tokenizer, &self.session.tokenizer)
            || !std::ptr::addr_eq(self.model.as_ptr(), Arc::as_ptr(&self.session.model))
        {
            return Err(ValidationError::UnsupportedProfile);
        }
        let config = self.session.model.config();
        if config.vocab_size < self.tokenizer.vocab_size()
            || !config.is_causal
            || self.session.model.is_classifier()
            || self
                .session
                .lora
                .as_ref()
                .is_some_and(|adapter| adapter.is_classifier())
            || !self.session.capabilities.text_in
            || !self.session.capabilities.text_out
        {
            return Err(ValidationError::UnsupportedProfile);
        }
        if self.session.config.n_keep != 0 {
            return Err(ValidationError::SlidingContext);
        }
        Ok(())
    }
}

// Guard ensuring that unexpected unwinds or panics mark the session unusable
// and purge potentially corrupted logits.
struct DecodeGuard<'a> {
    session: &'a mut Session,
    finished: bool,
}

impl Drop for DecodeGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.session.usable = false;
            self.session.last_logits = None;
        }
    }
}

impl Execution for CoreExecution {
    fn position(&self) -> usize {
        self.session.current_pos
    }
    fn capacity(&self) -> usize {
        self.session.max_seq_len
    }
    fn audio_output(&self) -> bool {
        self.session.capabilities.audio_out || self.session.audio_decoder.is_some()
    }
    fn validate_prepare(&self) -> Result<(), ValidationError> {
        self.validate_identity()?;
        if !self.session.is_usable() {
            return Err(ValidationError::Phase(SessionPhase::Unusable));
        }
        Ok(())
    }
    fn validate_decode(&self, _: &GenerateOpts) -> Result<(), ValidationError> {
        self.validate_prepare()
    }
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError> {
        self.session.last_ingest_recovery = None;
        self.session
            .with_ingest_recovery(|session| session.append_tokens(tokens))
            .map_err(|cause| {
                let IngestRecovery {
                    outcome,
                    rewind_error,
                    reset_error,
                } = self
                    .session
                    .last_ingest_recovery
                    .take()
                    .unwrap_or(IngestRecovery {
                        outcome: RecoveryOutcome::Unusable,
                        rewind_error: None,
                        reset_error: None,
                    });
                IngestError {
                    cause: IngestCause::Execution(cause),
                    recovery: outcome,
                    rewind_error: rewind_error.map(Box::new),
                    recovery_error: reset_error,
                }
            })
    }
    fn reset(&mut self, explicit: bool) -> Result<(), CeraError> {
        self.session.usable = false;
        self.session.last_logits = None;
        match self.session.reset_execution_checked() {
            Ok(()) => {
                self.session.usable = true;
                self.session.last_ingest_recovery = None;
            }
            Err(ref err) if err.is_checked_kv_reset_unsupported() => {
                self.session.reset_realloc_state()?;
            }
            Err(err) => return Err(err),
        }
        // Implicit resets during message replacement preserve pending cancellation flags,
        // while explicit coordinator resets clear them to prepare for a fresh turn.
        if explicit {
            self.session.clear_cancel();
        }
        Ok(())
    }
    fn decode(&mut self, opts: &GenerateOpts, sink: &mut dyn ModalitySink) -> DecodeReport {
        let mut guard = DecodeGuard {
            session: &mut self.session,
            finished: false,
        };
        let observed = guard.session.generate_observed(opts, sink);
        let state = match observed.observation {
            // If tokens were generated despite a NoProgress observation, the state is inconsistent.
            DecodeObservation::NoProgress
                if observed
                    .result
                    .as_ref()
                    .is_ok_and(|r| r.tokens_generated != 0) =>
            {
                DecodeState::Unusable
            }
            DecodeObservation::NoProgress => DecodeState::NoProgress,
            DecodeObservation::TokenStop { token } if observed.result.is_ok() => {
                DecodeState::Terminal {
                    token,
                    committed: false,
                }
            }
            DecodeObservation::Interrupted | DecodeObservation::Audio
                if observed.result.is_ok() =>
            {
                DecodeState::Interrupted
            }
            _ => DecodeState::Unusable,
        };
        guard.finished = !matches!(state, DecodeState::Unusable);
        DecodeReport {
            result: observed.result,
            state,
        }
    }
    fn cancel_handle(&self) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        Some(self.session.cancel_handle())
    }
    fn cancel(&self) {
        self.session.cancel();
    }
    fn clear_cancel(&mut self) {
        self.session.clear_cancel();
    }
}

#[cfg(test)]
#[path = "../../tests/api_chat/contract.rs"]
mod contract;

#[cfg(test)]
#[path = "../../tests/api_chat/tests.rs"]
mod contract_tests;

#[cfg(test)]
#[path = "../../tests/api_chat/fixtures.rs"]
mod fixtures;

#[cfg(test)]
mod tests;

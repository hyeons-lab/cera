//! Isolated P0.1 contract prototype. `cera/tests/chat_contract.rs` compiles it
//! against the scripted executor; `cera/src/session/chat.rs` includes the same
//! file in the unit-test build so the private core adapter can implement it
//! without public test hooks. Both binders alias the crate as `core_api`.
//! This is not a published API or a production decode adapter.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::core_api::CeraError;
use super::core_api::kv_cache::KvRewindError;
use super::core_api::session::{
    FinishReason, GenerateOpts, GenerateSummary, ModalitySink, RecoveryOutcome,
};
use super::core_api::tokenizer::{BpeTokenizer, ChatMessage, apply_chat_template};

pub const TEMPLATE: &str = "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
pub const LFM2_5_TEMPLATE: &str = "{{- bos_token -}}\n{%- set keep_past_thinking = keep_past_thinking | default(false) -%}\n{%- set ns = namespace(system_prompt=\"\") -%}\n{%- if messages[0][\"role\"] == \"system\" -%}\n    {%- set sys_content = messages[0][\"content\"] -%}\n    {%- if sys_content is not string -%}\n        {%- for item in sys_content -%}\n            {%- if item[\"type\"] == \"text\" -%}\n                {%- set ns.system_prompt = ns.system_prompt + item[\"text\"] -%}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {%- set ns.system_prompt = sys_content -%}\n    {%- endif -%}\n    {%- set messages = messages[1:] -%}\n{%- endif -%}\n{%- if tools -%}\n    {%- set ns.system_prompt = ns.system_prompt + (\"\\n\" if ns.system_prompt else \"\") + \"List of tools: [\" -%}\n    {%- for tool in tools -%}\n        {%- if tool is not string -%}\n            {%- set tool = tool | tojson -%}\n        {%- endif -%}\n        {%- set ns.system_prompt = ns.system_prompt + tool -%}\n        {%- if not loop.last -%}\n            {%- set ns.system_prompt = ns.system_prompt + \", \" -%}\n        {%- endif -%}\n    {%- endfor -%}\n    {%- set ns.system_prompt = ns.system_prompt + \"]\" -%}\n{%- endif -%}\n{%- if ns.system_prompt -%}\n    {{- \"<|im_start|>system\\n\" + ns.system_prompt + \"<|im_end|>\\n\" -}}\n{%- endif -%}\n{%- set ns.last_assistant_index = -1 -%}\n{%- for message in messages -%}\n    {%- if message[\"role\"] == \"assistant\" -%}\n        {%- set ns.last_assistant_index = loop.index0 -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- for message in messages -%}\n    {{- \"<|im_start|>\" + message[\"role\"] + \"\\n\" -}}\n    {%- set content = message[\"content\"] -%}\n    {%- if content is not string -%}\n        {%- set ns.content = \"\" -%}\n        {%- for item in content -%}\n            {%- if item[\"type\"] == \"image\" -%}\n                {%- set ns.content = ns.content + \"<image>\" -%}\n            {%- elif item[\"type\"] == \"text\" -%}\n                {%- set ns.content = ns.content + item[\"text\"] -%}\n            {%- else -%}\n                {%- set ns.content = ns.content + item | tojson -%}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- set content = ns.content -%}\n    {%- endif -%}\n    {%- if message[\"role\"] == \"assistant\" and not keep_past_thinking and loop.index0 != ns.last_assistant_index -%}\n        {%- if \"</think>\" in content -%}\n            {%- set content = content.split(\"</think>\")[-1] | trim -%}\n        {%- endif -%}\n    {%- endif -%}\n    {{- content + \"<|im_end|>\\n\" -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- \"<|im_start|>assistant\\n\" -}}\n{%- endif -%}";
const BOS: &str = "<|startoftext|>";
const END: &str = "<|im_end|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text(String),
    Image(Vec<u8>),
    Audio { pcm: Vec<f32>, sample_rate: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(text.into())],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    Idle,
    PromptReady,
    TurnComplete,
    Interrupted,
    RawContext,
    Unusable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    UnsupportedProfile,
    SlidingContext,
    AudioOutput,
    Generation(String),
    Phase(SessionPhase),
    EmptyBatch,
    RoleOrder { message: usize },
    UnsupportedRole { message: usize },
    UnsupportedContent { message: usize, part: usize },
    ReservedMarker { message: usize },
    Capacity { required: usize, available: usize },
    Template(String),
}

#[derive(Debug)]
pub enum IngestCause {
    Validation(ValidationError),
    Execution(CeraError),
}

#[derive(Debug)]
pub struct IngestError {
    pub cause: IngestCause,
    pub recovery: RecoveryOutcome,
    pub rewind_error: Option<Box<KvRewindError>>,
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

#[derive(Debug, PartialEq, Eq)]
pub struct IngestSummary {
    pub input_tokens: usize,
    pub position_before: usize,
    pub position_after: usize,
}

/// A bridge must account for execution state, not just emitted token counters.
/// In particular, a sampled EOS can consume RNG without emitting or committing.
#[derive(Debug, Clone, Copy)]
pub enum DecodeState {
    NoProgress,
    Terminal { token: u32, committed: bool },
    Interrupted,
    Unusable,
}

#[derive(Debug)]
pub struct DecodeReport {
    pub result: Result<GenerateSummary, CeraError>,
    pub state: DecodeState,
}

/// Required backend contract. No production implementation is published; the
/// only implementor besides the scripted test executor is the private core
/// adapter in `cera/src/session/chat.rs`, whose identity and usability checks
/// are what `validate_prepare` exists for. `append` must recover the entire
/// supplied batch as one transaction, including metadata. `reset` must be
/// checked, must not depend on the chat profile, and must preserve external
/// cancellation when `explicit` is false. A failed reset establishes no usable
/// state.
pub trait Execution: std::fmt::Debug {
    fn position(&self) -> usize;
    fn capacity(&self) -> usize;
    fn audio_output(&self) -> bool;
    fn validate_prepare(&self) -> Result<(), ValidationError> {
        Ok(())
    }
    fn validate_decode(&self, opts: &GenerateOpts) -> Result<(), ValidationError>;
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError>;
    fn reset(&mut self, explicit: bool) -> Result<(), CeraError>;
    fn decode(&mut self, opts: &GenerateOpts, sink: &mut dyn ModalitySink) -> DecodeReport;
    fn cancel_handle(&self) -> Option<Arc<AtomicBool>> {
        None
    }
    fn cancel(&self) {}
    fn clear_cancel(&mut self) {}
}

#[derive(Debug)]
pub struct TurnResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub summary: GenerateSummary,
}

#[derive(Debug)]
pub enum CompleteError {
    Validation(ValidationError),
    Execution(CeraError),
}

#[derive(Default)]
struct Collector(Vec<u32>);

impl ModalitySink for Collector {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.0.extend_from_slice(tokens);
    }
    fn on_done(&mut self, _: FinishReason) {}
}

/// One frozen rendering profile. Discovery checks rendering prerequisites;
/// the public-model runner separately verifies the full artifact SHA-256.
/// Matching this small signature is not a general model/backend support claim.
pub struct Profile {
    tokenizer: Arc<BpeTokenizer>,
    eos: u32,
}

impl Profile {
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
        Ok(Self { tokenizer, eos: 7 })
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
                if message.role != expected {
                    return Err(ValidationError::RoleOrder { message: index });
                }
                expected = if expected == Role::User {
                    Role::Assistant
                } else {
                    Role::User
                };
                match message.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    _ => unreachable!(),
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
            // This text-only profile rejects token marker syntax, including a
            // marker split across adjacent text parts, instead of inventing an
            // escape convention that the model/template does not define.
            if text.contains("<|") {
                return Err(ValidationError::ReservedMarker { message: index });
            }
            serialized.push(ChatMessage {
                role: role.into(),
                content: text,
            });
        }
        if messages.last().unwrap().role != Role::User {
            return Err(ValidationError::RoleOrder {
                message: messages.len() - 1,
            });
        }
        let rendered = apply_chat_template(&self.tokenizer, &serialized, true)
            .map_err(|e| ValidationError::Template(e.to_string()))?;
        if initial {
            Ok(rendered)
        } else {
            rendered
                .strip_prefix(BOS)
                .map(str::to_owned)
                .ok_or(ValidationError::UnsupportedProfile)
        }
    }
}

/// Bounded chat bookkeeping plus execution; never retains application messages.
pub struct Chat<E> {
    execution: E,
    profile: Profile,
    phase: SessionPhase,
    // None unless a terminal token was positively observed. True means it is
    // already resident; the following template newline is still pending.
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

    pub fn into_inner(self) -> E {
        self.execution
    }

    pub fn cancel_handle(&self) -> Option<Arc<AtomicBool>> {
        self.execution.cancel_handle()
    }

    pub fn cancel(&self) {
        self.execution.cancel();
    }

    pub fn clear_cancel(&mut self) {
        self.execution.clear_cancel();
    }

    pub fn phase(&self) -> SessionPhase {
        self.phase
    }
    pub fn position(&self) -> usize {
        self.execution.position()
    }

    #[cfg(test)]
    pub(super) fn execution_for_test(&mut self) -> &mut E {
        &mut self.execution
    }

    pub fn ingest(&mut self, message: &Message) -> Result<IngestSummary, IngestError> {
        self.ingest_messages(std::slice::from_ref(message))
    }

    pub fn ingest_messages(&mut self, messages: &[Message]) -> Result<IngestSummary, IngestError> {
        if !matches!(self.phase, SessionPhase::Idle | SessionPhase::TurnComplete) {
            return Err(ValidationError::Phase(self.phase).into());
        }
        self.prepare(messages, false)
    }

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
            tokens.extend(self.profile.tokenizer.encode("\n"));
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
        // The phase is Unusable for the whole time execution may mutate; an
        // unwind anywhere below leaves it there. Only a non-replacing append
        // can restore the prior state, so it is captured once, here.
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
                // After replacement reset, the previous context cannot be
                // restored: Unchanged/Restored refer only to the empty state.
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

    pub fn generate_into(
        &mut self,
        opts: &GenerateOpts,
        sink: &mut dyn ModalitySink,
    ) -> Result<DecodeReport, ValidationError> {
        if self.phase != SessionPhase::PromptReady {
            return Err(ValidationError::Phase(self.phase));
        }
        // Execution validity is reported before the audio capability, as in
        // `prepare`, so both entry points name the same first error.
        self.execution.validate_decode(opts)?;
        if self.execution.audio_output() {
            return Err(ValidationError::AudioOutput);
        }
        // An unwind or unclassified backend exit must not preserve PromptReady.
        self.phase = SessionPhase::Unusable;
        let report = self.execution.decode(opts, sink);
        self.phase = match report.state {
            DecodeState::NoProgress => match &report.result {
                Ok(summary) if summary.tokens_generated == 0 => SessionPhase::PromptReady,
                // A no-progress error (for example missing prefill logits)
                // proves the execution did not mutate, so only the cursor is
                // stale: replacement and raw access stay open, like after
                // `raw()`, instead of forcing a destructive reset.
                Err(_) => SessionPhase::RawContext,
                // Emitted tokens contradict the executor's own observation.
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

    /// Text collection invokes the same decode entry exactly once. Use the sink
    /// surface when partial output must remain available after an execution error.
    pub fn complete(&mut self, opts: &GenerateOpts) -> Result<TurnResult, CompleteError> {
        let mut collector = Collector::default();
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

    pub fn reset(&mut self) -> Result<(), CeraError> {
        self.phase = SessionPhase::Unusable;
        self.terminal_committed = None;
        self.execution.reset(true)?;
        self.phase = SessionPhase::Idle;
        Ok(())
    }

    /// Any raw escape conservatively invalidates the chat cursor before access.
    pub fn raw(&mut self) -> Result<&mut E, ValidationError> {
        if self.phase == SessionPhase::Unusable {
            return Err(ValidationError::Phase(self.phase));
        }
        self.phase = SessionPhase::RawContext;
        self.terminal_committed = None;
        Ok(&mut self.execution)
    }
}

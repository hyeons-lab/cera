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
    ModalitySink, RecoveryOutcome, Session, checkpoint,
};
#[cfg(test)]
use crate as core_api;
use crate::kv_cache::KvRewindError;
use crate::model::Model;
use crate::tokenizer::{
    BpeTokenizer, ChatMessage, apply_chat_template, apply_chat_template_with_tools,
};
use crate::tools::{ToolCall, ToolDef, ToolFormat, parse_tool_calls, tool_grammar};

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

impl ContentPart {
    /// Convenience constructor for a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Convenience constructor for an image part.
    pub fn image(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Image(bytes.into())
    }

    /// Convenience constructor for an audio PCM part.
    pub fn audio(pcm: impl Into<Vec<f32>>, sample_rate: u32) -> Self {
        Self::Audio {
            pcm: pcm.into(),
            sample_rate,
        }
    }
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

    /// Create a single-part user message with an image payload.
    pub fn image(bytes: impl Into<Vec<u8>>) -> Self {
        Self::with_parts(Role::User, vec![ContentPart::Image(bytes.into())])
    }

    /// Create a user message with an image payload and accompanying text prompt.
    pub fn user_with_image(text: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self::with_parts(
            Role::User,
            vec![
                ContentPart::Image(bytes.into()),
                ContentPart::Text(text.into()),
            ],
        )
    }

    /// Create a single-part user message with audio PCM waveform samples.
    pub fn audio(pcm: impl Into<Vec<f32>>, sample_rate: u32) -> Self {
        Self::with_parts(
            Role::User,
            vec![ContentPart::Audio {
                pcm: pcm.into(),
                sample_rate,
            }],
        )
    }

    /// Create a user message with audio PCM samples and accompanying text prompt.
    pub fn user_with_audio(
        text: impl Into<String>,
        pcm: impl Into<Vec<f32>>,
        sample_rate: u32,
    ) -> Self {
        let text = text.into();
        let mut parts = vec![ContentPart::Audio {
            pcm: pcm.into(),
            sample_rate,
        }];
        if !text.is_empty() {
            parts.push(ContentPart::Text(format!("\n{text}")));
        }
        Self::with_parts(Role::User, parts)
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

/// A segment of prefill content to be ingested into the execution context.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestSegment<'a> {
    /// Token IDs to be evaluated.
    Tokens(&'a [u32]),
    /// Image payload bytes.
    Image(&'a [u8]),
    /// Audio PCM samples and sample rate.
    Audio { pcm: &'a [f32], sample_rate: u32 },
}

/// Execution backend contract for chat coordination.
pub trait Execution: std::fmt::Debug {
    /// Current token position in the context.
    fn position(&self) -> usize;
    /// Maximum context capacity.
    fn capacity(&self) -> usize;
    /// Whether audio output decoding is active.
    fn audio_output(&self) -> bool;
    /// Whether image input is supported.
    fn image_input(&self) -> bool {
        false
    }
    /// Whether audio input is supported.
    fn audio_input(&self) -> bool {
        false
    }
    /// Pre-mutation validation before preparing context.
    fn validate_prepare(&self) -> Result<(), ValidationError> {
        Ok(())
    }
    /// Validation before decode starts.
    fn validate_decode(&self, opts: &GenerateOpts) -> Result<(), ValidationError>;
    /// Append a batch of tokens atomically to context.
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError>;
    /// Append a sequence of multimodal segments (tokens, images, audio) atomically to context.
    fn append_segments(&mut self, segments: &[IngestSegment<'_>]) -> Result<(), IngestError> {
        for seg in segments {
            match *seg {
                IngestSegment::Tokens(tokens) => self.append(tokens)?,
                IngestSegment::Image(_) | IngestSegment::Audio { .. } => {
                    return Err(ValidationError::UnsupportedContent {
                        message: 0,
                        part: 0,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
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
    /// Capture a snapshot of underlying session state if supported.
    fn checkpoint(&self) -> Result<checkpoint::SessionCheckpoint, CeraError> {
        Err(CeraError::Format(
            "execution backend does not support checkpointing".to_string(),
        ))
    }
    /// Restore a snapshot of underlying session state if supported.
    fn restore(&mut self, checkpoint: &checkpoint::SessionCheckpoint) -> Result<(), CeraError> {
        let _ = checkpoint;
        Err(CeraError::Format(
            "execution backend does not support restoring checkpoints".to_string(),
        ))
    }
}

/// Result of a completed chat turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnResult {
    /// Decoded assistant response text.
    pub text: String,
    /// Token identifiers emitted during the turn.
    pub tokens: Vec<u32>,
    /// Generation summary metrics.
    pub summary: GenerateSummary,
    /// Parsed tool calls if emitted by the model.
    pub tool_calls: Vec<ToolCall>,
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

/// Chat template family identifying turn framing semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TemplateFamily {
    /// ChatML, Hermes, Qwen style with `<|im_start|>` and `<|im_end|>`.
    ChatML,
    /// Llama 3 style with `<|start_header_id|>` and `<|eot_id|>`.
    Llama3,
    /// Gemma style with `<start_of_turn>` and `<end_of_turn>`.
    Gemma,
    /// Custom or dynamically probed template.
    Custom,
}

/// Builder for constructing custom chat profiles.
#[derive(Clone)]
pub struct ProfileBuilder {
    tokenizer: Arc<BpeTokenizer>,
    family: TemplateFamily,
    turn_prefix: Option<String>,
    turn_end: Option<String>,
    eos: Option<u32>,
}

impl ProfileBuilder {
    /// Create a new profile builder for the given tokenizer.
    pub fn new(tokenizer: Arc<BpeTokenizer>) -> Self {
        Self {
            tokenizer,
            family: TemplateFamily::ChatML,
            turn_prefix: None,
            turn_end: None,
            eos: None,
        }
    }

    /// Set the template family.
    pub fn family(mut self, family: TemplateFamily) -> Self {
        self.family = family;
        self
    }

    /// Set the turn prefix text stripped on continuation turns.
    pub fn turn_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.turn_prefix = Some(prefix.into());
        self
    }

    /// Set the turn end delimiter text.
    pub fn turn_end(mut self, end: impl Into<String>) -> Self {
        self.turn_end = Some(end.into());
        self
    }

    /// Set the end-of-sequence token ID.
    pub fn eos(mut self, eos: u32) -> Self {
        self.eos = Some(eos);
        self
    }

    /// Build the profile.
    pub fn build(self) -> Result<Profile, ValidationError> {
        let (turn_prefix, turn_end, default_eos) = match self.family {
            TemplateFamily::ChatML => (BOS.to_string(), END.to_string(), 7),
            TemplateFamily::Llama3 => (
                "<|begin_of_text|>".to_string(),
                "<|eot_id|>".to_string(),
                self.tokenizer
                    .special_token_id("<|eot_id|>")
                    .unwrap_or(128009),
            ),
            TemplateFamily::Gemma => (
                "<bos>".to_string(),
                "<end_of_turn>".to_string(),
                self.tokenizer
                    .special_token_id("<end_of_turn>")
                    .unwrap_or(1),
            ),
            TemplateFamily::Custom => (
                String::new(),
                "\n".to_string(),
                self.tokenizer.eos_token().unwrap_or(2),
            ),
        };
        let turn_prefix = self.turn_prefix.unwrap_or(turn_prefix);
        let turn_end = self.turn_end.unwrap_or(turn_end);
        let eos = self
            .eos
            .or_else(|| self.tokenizer.eos_token())
            .unwrap_or(default_eos);
        Profile::new_with_family(self.tokenizer, self.family, turn_prefix, turn_end, eos)
    }
}

/// Chat rendering profile holding tokenizer and template rules.
#[derive(Clone)]
pub struct Profile {
    tokenizer: Arc<BpeTokenizer>,
    eos: u32,
    family: TemplateFamily,
    turn_prefix: String,
    turn_end: String,
    newline_tokens: Vec<u32>,
    image_marker: Option<u32>,
    image_start: Option<u32>,
    image_end: Option<u32>,
    audio_marker: Option<(u32, &'static str)>,
}

impl std::fmt::Debug for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Profile")
            .field("eos", &self.eos)
            .field("family", &self.family)
            .field("turn_prefix", &self.turn_prefix)
            .field("turn_end", &self.turn_end)
            .field("newline_tokens", &self.newline_tokens)
            .field("image_marker", &self.image_marker)
            .field("image_start", &self.image_start)
            .field("image_end", &self.image_end)
            .field("audio_marker", &self.audio_marker)
            .finish()
    }
}

impl Profile {
    /// Discover and validate a chat profile from a tokenizer's metadata.
    pub fn discover(tokenizer: Arc<BpeTokenizer>) -> Result<Self, ValidationError> {
        let template_str = tokenizer
            .chat_template()
            .ok_or(ValidationError::UnsupportedProfile)?;

        // 1. Check ChatML (LFM2, LFM2.5, Qwen, DeepSeek, Hermes)
        if template_str == TEMPLATE
            || template_str == LFM2_5_TEMPLATE
            || (template_str.contains("<|im_start|>") && template_str.contains("<|im_end|>"))
        {
            if template_str == TEMPLATE || template_str == LFM2_5_TEMPLATE {
                if tokenizer.bos_token() != Some(1)
                    || tokenizer.eos_token() != Some(7)
                    || tokenizer.encode(BOS) != [1]
                    || tokenizer.encode(END) != [7]
                    || tokenizer.encode("<|im_start|>") != [6]
                {
                    return Err(ValidationError::UnsupportedProfile);
                }
            } else if tokenizer.encode("<|im_start|>").is_empty()
                || tokenizer.encode("<|im_end|>").is_empty()
            {
                return Err(ValidationError::UnsupportedProfile);
            }
            let eos = tokenizer
                .eos_token()
                .or_else(|| tokenizer.special_token_id("<|im_end|>"))
                .unwrap_or(7);
            return Self::new_with_family(
                tokenizer,
                TemplateFamily::ChatML,
                BOS.to_string(),
                END.to_string(),
                eos,
            );
        }

        // 2. Check Llama 3 / 3.1 / 3.2
        if template_str.contains("<|start_header_id|>")
            && (template_str.contains("<|eot_id|>") || template_str.contains("<|end_header_id|>"))
        {
            let turn_end = "<|eot_id|>";
            let turn_prefix = if tokenizer.bos_token().is_some() {
                "<|begin_of_text|>"
            } else {
                "<|start_header_id|>"
            };
            let eos = tokenizer
                .special_token_id(turn_end)
                .or_else(|| tokenizer.eos_token())
                .ok_or(ValidationError::UnsupportedProfile)?;
            return Self::new_with_family(
                tokenizer,
                TemplateFamily::Llama3,
                turn_prefix.to_string(),
                turn_end.to_string(),
                eos,
            );
        }

        // 3. Check Gemma 2 / 3
        if template_str.contains("<start_of_turn>") && template_str.contains("<end_of_turn>") {
            let turn_end = "<end_of_turn>";
            let turn_prefix = if tokenizer.bos_token().is_some() {
                "<bos>"
            } else {
                "<start_of_turn>"
            };
            let eos = tokenizer
                .special_token_id(turn_end)
                .or_else(|| tokenizer.eos_token())
                .ok_or(ValidationError::UnsupportedProfile)?;
            return Self::new_with_family(
                tokenizer,
                TemplateFamily::Gemma,
                turn_prefix.to_string(),
                turn_end.to_string(),
                eos,
            );
        }

        // 4. Dynamic probe for generic Jinja templates
        let probe = [ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        if let (Ok(rendered), Some(eos)) = (
            apply_chat_template(&tokenizer, &probe, true),
            tokenizer.eos_token(),
        ) {
            let turn_end = if rendered.contains("<|im_end|>") {
                "<|im_end|>"
            } else if rendered.contains("<|eot_id|>") {
                "<|eot_id|>"
            } else if rendered.contains("<end_of_turn>") {
                "<end_of_turn>"
            } else {
                "\n"
            };
            return Self::new_with_family(
                tokenizer,
                TemplateFamily::Custom,
                String::new(),
                turn_end.to_string(),
                eos,
            );
        }

        Err(ValidationError::UnsupportedProfile)
    }

    fn new_with_family(
        tokenizer: Arc<BpeTokenizer>,
        family: TemplateFamily,
        turn_prefix: String,
        turn_end: String,
        eos: u32,
    ) -> Result<Self, ValidationError> {
        let newline_tokens = tokenizer.encode("\n");
        let image_marker = {
            let probed = tokenizer.encode("<image>");
            if probed.len() == 1 {
                Some(probed[0])
            } else {
                None
            }
        };
        let image_start = tokenizer.special_token_id("<|image_start|>");
        let image_end = tokenizer.special_token_id("<|image_end|>");
        let audio_marker = crate::engine::CeraEngine::AUDIO_MARKER_CANDIDATES
            .into_iter()
            .find_map(|name| tokenizer.special_token_id(name).map(|id| (id, name)));
        Ok(Self {
            tokenizer,
            eos,
            family,
            turn_prefix,
            turn_end,
            newline_tokens,
            image_marker,
            image_start,
            image_end,
            audio_marker,
        })
    }

    /// Builder for configuring custom profiles.
    pub fn builder(tokenizer: Arc<BpeTokenizer>) -> ProfileBuilder {
        ProfileBuilder::new(tokenizer)
    }

    /// Create a profile with explicit family and delimiters.
    pub fn custom(
        tokenizer: Arc<BpeTokenizer>,
        family: TemplateFamily,
        turn_prefix: impl Into<String>,
        turn_end: impl Into<String>,
        eos: Option<u32>,
    ) -> Result<Self, ValidationError> {
        let eos = eos
            .or_else(|| tokenizer.eos_token())
            .ok_or(ValidationError::UnsupportedProfile)?;
        Self::new_with_family(tokenizer, family, turn_prefix.into(), turn_end.into(), eos)
    }

    /// Template family identifying turn framing semantics.
    pub fn family(&self) -> TemplateFamily {
        self.family
    }

    /// Turn prefix text stripped on continuation turns.
    pub fn turn_prefix(&self) -> &str {
        &self.turn_prefix
    }

    /// Turn end delimiter text.
    pub fn turn_end(&self) -> &str {
        &self.turn_end
    }

    /// End-of-sequence token ID.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Pre-encoded newline token sequence for turn boundaries.
    pub fn newline_tokens(&self) -> &[u32] {
        &self.newline_tokens
    }

    /// Token ID for `<image>` marker if supported by tokenizer.
    pub fn image_marker(&self) -> Option<u32> {
        self.image_marker
    }

    /// Token ID for `<|image_start|>` envelope if supported by tokenizer.
    pub fn image_start(&self) -> Option<u32> {
        self.image_start
    }

    /// Token ID for `<|image_end|>` envelope if supported by tokenizer.
    pub fn image_end(&self) -> Option<u32> {
        self.image_end
    }

    /// Audio marker special token ID and candidate name if supported.
    pub fn audio_marker(&self) -> Option<(u32, &'static str)> {
        self.audio_marker
    }

    /// Reference to the tokenizer used by this profile.
    pub fn tokenizer(&self) -> &Arc<BpeTokenizer> {
        &self.tokenizer
    }

    fn render(
        &self,
        messages: &[Message],
        initial: bool,
        tools: &[ToolDef],
    ) -> Result<String, ValidationError> {
        if messages.is_empty() {
            return Err(ValidationError::EmptyBatch);
        }
        let mut serialized = Vec::with_capacity(messages.len());
        #[derive(Clone, Copy, PartialEq)]
        enum MessageState {
            Start,
            System,
            User,
            Assistant,
            Tool,
        }
        let mut state = MessageState::Start;
        let assistant_role = match self.family {
            TemplateFamily::Gemma => "model",
            _ => "assistant",
        };
        for (index, message) in messages.iter().enumerate() {
            if message.role == Role::Tool && tools.is_empty() {
                return Err(ValidationError::UnsupportedRole { message: index });
            }
            let role = match (state, message.role) {
                (MessageState::Start, Role::System) if initial && index == 0 => {
                    state = MessageState::System;
                    "system"
                }
                (
                    MessageState::Start | MessageState::System | MessageState::Assistant,
                    Role::User,
                ) => {
                    state = MessageState::User;
                    "user"
                }
                (
                    MessageState::Start | MessageState::Assistant | MessageState::Tool,
                    Role::Tool,
                ) if !initial => {
                    state = MessageState::Tool;
                    "tool"
                }
                (MessageState::Assistant | MessageState::Tool, Role::Tool) => {
                    state = MessageState::Tool;
                    "tool"
                }
                (MessageState::User | MessageState::Tool, Role::Assistant) => {
                    state = MessageState::Assistant;
                    assistant_role
                }
                _ => return Err(ValidationError::RoleOrder { message: index }),
            };
            let mut text = String::new();
            let mut user_text = String::new();
            for (part, content) in message.content.iter().enumerate() {
                match content {
                    ContentPart::Text(value) => {
                        user_text.push_str(value);
                        text.push_str(value);
                    }
                    ContentPart::Image(_) => {
                        if message.role != Role::User {
                            return Err(ValidationError::UnsupportedContent {
                                message: index,
                                part,
                            });
                        }
                        if self.image_marker.is_none()
                            || self.image_start.is_none()
                            || self.image_end.is_none()
                        {
                            return Err(ValidationError::UnsupportedContent {
                                message: index,
                                part,
                            });
                        }
                        text.push_str("<image>");
                    }
                    ContentPart::Audio { .. } => {
                        if message.role != Role::User {
                            return Err(ValidationError::UnsupportedContent {
                                message: index,
                                part,
                            });
                        }
                        let Some((_, marker_name)) = self.audio_marker else {
                            return Err(ValidationError::UnsupportedContent {
                                message: index,
                                part,
                            });
                        };
                        text.push_str(marker_name);
                    }
                }
            }
            let has_reserved = match self.family {
                TemplateFamily::ChatML | TemplateFamily::Llama3 => user_text.contains("<|"),
                TemplateFamily::Gemma => {
                    user_text.contains("<start_of_turn>") || user_text.contains("<end_of_turn>")
                }
                TemplateFamily::Custom => {
                    (!self.turn_prefix.is_empty() && user_text.contains(&self.turn_prefix))
                        || (!self.turn_end.is_empty() && user_text.contains(&self.turn_end))
                }
            };
            if has_reserved {
                return Err(ValidationError::ReservedMarker { message: index });
            }
            serialized.push(ChatMessage {
                role: role.into(),
                content: text,
            });
        }
        if messages
            .last()
            .is_none_or(|msg| msg.role != Role::User && msg.role != Role::Tool)
        {
            return Err(ValidationError::RoleOrder {
                message: messages.len().saturating_sub(1),
            });
        }
        let template_has_tools = self
            .tokenizer
            .chat_template()
            .is_some_and(|t| t.contains("tools"));
        if initial && !tools.is_empty() && !template_has_tools {
            let tools_json = tools
                .iter()
                .map(|t| serde_json::to_string(t).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(", ");
            let tool_block = format!("List of tools: [{tools_json}]");
            if let Some(first) = serialized.first_mut()
                && first.role == "system"
            {
                first.content.push('\n');
                first.content.push_str(&tool_block);
            } else {
                serialized.insert(
                    0,
                    ChatMessage {
                        role: "system".into(),
                        content: tool_block,
                    },
                );
            }
        }
        let rendered = if initial && !tools.is_empty() {
            apply_chat_template_with_tools(&self.tokenizer, &serialized, tools, true)
        } else {
            apply_chat_template(&self.tokenizer, &serialized, true)
        }
        .map_err(|e| ValidationError::Template(e.to_string()))?;
        if initial {
            Ok(rendered)
        } else {
            let mut s = rendered;
            let candidate_prefixes = [
                self.turn_prefix.as_str(),
                BOS,
                "<|begin_of_text|>",
                "<bos>",
                "<s>",
            ];
            for prefix in candidate_prefixes {
                if !prefix.is_empty() && s.starts_with(prefix) {
                    s.drain(..prefix.len());
                    break;
                }
            }
            if s.starts_with("\r\n") {
                s.drain(..2);
            } else if s.starts_with('\n') {
                s.drain(..1);
            }
            Ok(s)
        }
    }
}

/// Stateful chat coordinator over an execution engine.
pub struct Chat<E = CoreExecution> {
    execution: E,
    profile: Profile,
    phase: SessionPhase,
    terminal_committed: Option<bool>,
    tools: Vec<ToolDef>,
    tool_format: ToolFormat,
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
            tools: Vec::new(),
            tool_format: ToolFormat::Lfm2Pythonic,
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

    /// Capture a checkpoint of current conversational state and underlying execution.
    pub fn checkpoint(&self) -> Result<checkpoint::ChatCheckpoint, CeraError> {
        let session_checkpoint = self.execution.checkpoint()?;
        Ok(checkpoint::ChatCheckpoint {
            session_checkpoint,
            phase: self.phase,
            tool_format: self.tool_format,
            tools: self.tools.clone(),
            terminal_committed: self.terminal_committed,
        })
    }

    /// Restore a previously captured checkpoint into this chat session.
    pub fn restore(&mut self, checkpoint: &checkpoint::ChatCheckpoint) -> Result<(), CeraError> {
        if checkpoint.phase == SessionPhase::Unusable {
            return Err(CeraError::Format(
                "cannot restore chat checkpoint in Unusable phase".to_string(),
            ));
        }
        self.execution.restore(&checkpoint.session_checkpoint)?;
        self.phase = checkpoint.phase;
        self.tool_format = checkpoint.tool_format;
        self.tools = checkpoint.tools.clone();
        self.terminal_committed = checkpoint.terminal_committed.or_else(|| {
            if checkpoint.phase == SessionPhase::TurnComplete {
                Some(false)
            } else {
                None
            }
        });
        Ok(())
    }

    /// Save chat session checkpoint to file.
    pub fn save_checkpoint(&self, path: impl AsRef<std::path::Path>) -> Result<(), CeraError> {
        let cp = self.checkpoint()?;
        cp.save_to_file(path)
    }

    /// Load and restore chat session checkpoint from file.
    pub fn load_checkpoint(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), CeraError> {
        let cp = checkpoint::ChatCheckpoint::load_from_file(path)?;
        self.restore(&cp)
    }

    /// Current session lifecycle phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Active chat rendering profile.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Currently registered tools for function calling.
    pub fn tools(&self) -> &[ToolDef] {
        &self.tools
    }

    /// Register tools for function calling.
    pub fn set_tools(&mut self, tools: Vec<ToolDef>) {
        self.tools = tools;
    }

    /// Active tool wire format.
    pub fn tool_format(&self) -> ToolFormat {
        self.tool_format
    }

    /// Set the tool wire format explicitly.
    pub fn set_tool_format(&mut self, format: ToolFormat) {
        self.tool_format = format;
    }

    /// Ingest a tool execution response back into the conversation.
    pub fn ingest_tool_response(
        &mut self,
        _name: &str,
        content: &str,
    ) -> Result<IngestSummary, IngestError> {
        self.ingest(&Message::tool(content))
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

        // Scan messages for multimodal parts and check capabilities upfront:
        let mut images: Vec<&[u8]> = Vec::new();
        let mut audio: Option<(&[f32], u32)> = None;
        for (m_idx, msg) in messages.iter().enumerate() {
            for (p_idx, part) in msg.content.iter().enumerate() {
                match part {
                    ContentPart::Text(_) => {}
                    ContentPart::Image(bytes) => {
                        if !self.execution.image_input() {
                            return Err(ValidationError::UnsupportedContent {
                                message: m_idx,
                                part: p_idx,
                            }
                            .into());
                        }
                        if audio.is_some() {
                            return Err(ValidationError::UnsupportedContent {
                                message: m_idx,
                                part: p_idx,
                            }
                            .into());
                        }
                        images.push(bytes);
                    }
                    ContentPart::Audio { pcm, sample_rate } => {
                        if !self.execution.audio_input() {
                            return Err(ValidationError::UnsupportedContent {
                                message: m_idx,
                                part: p_idx,
                            }
                            .into());
                        }
                        if audio.is_some() || !images.is_empty() {
                            return Err(ValidationError::UnsupportedContent {
                                message: m_idx,
                                part: p_idx,
                            }
                            .into());
                        }
                        audio = Some((pcm, *sample_rate));
                    }
                }
            }
        }

        let before = self.position();
        let initial = replace || self.phase == SessionPhase::Idle;
        let rendered = self.profile.render(messages, initial, &self.tools)?;
        let mut tokens = Vec::with_capacity(
            rendered.len().saturating_div(2) + self.profile.newline_tokens.len() + 1,
        );
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

        // Build prefill segments:
        let img_start_arr;
        let img_end_arr;
        let mut template_segments_opt = None;
        let mut audio_split_opt = None;

        if !images.is_empty() {
            let image_marker_id = self.profile.image_marker.ok_or_else(|| {
                ValidationError::Template("missing image marker token in profile".into())
            })?;
            let img_start_id = self.profile.image_start.ok_or_else(|| {
                ValidationError::Template("missing image start token in profile".into())
            })?;
            let img_end_id = self.profile.image_end.ok_or_else(|| {
                ValidationError::Template("missing image end token in profile".into())
            })?;
            img_start_arr = [img_start_id];
            img_end_arr = [img_end_id];

            let segs = crate::session::splice_image_markers(&tokens, image_marker_id);
            let marker_count = segs
                .iter()
                .filter(|s| matches!(s, crate::session::ChatTemplateSegment::Image))
                .count();
            if marker_count != images.len() {
                return Err(ValidationError::Template(format!(
                    "rendered template has {marker_count} image markers but {} images were supplied",
                    images.len()
                ))
                .into());
            }
            template_segments_opt = Some(segs);
        } else if audio.is_some() {
            img_start_arr = [0];
            img_end_arr = [0];
            let (marker_id, marker_name) = self.profile.audio_marker.ok_or_else(|| {
                ValidationError::Template("missing audio marker token in profile".into())
            })?;
            let split =
                crate::engine::CeraEngine::split_tokens_at_marker(&tokens, marker_id, marker_name)
                    .map_err(|e| ValidationError::Template(e.to_string()))?;
            audio_split_opt = Some(split);
        } else {
            img_start_arr = [0];
            img_end_arr = [0];
        }

        let mut segments = Vec::new();
        if let Some(template_segments) = template_segments_opt {
            let mut img_idx = 0;
            for seg in &template_segments {
                match *seg {
                    crate::session::ChatTemplateSegment::Text { start, end } => {
                        segments.push(IngestSegment::Tokens(&tokens[start..end]));
                    }
                    crate::session::ChatTemplateSegment::Image => {
                        segments.push(IngestSegment::Tokens(&img_start_arr));
                        segments.push(IngestSegment::Image(images[img_idx]));
                        segments.push(IngestSegment::Tokens(&img_end_arr));
                        img_idx += 1;
                    }
                }
            }
        } else if let (Some(split), Some((pcm, sample_rate))) = (audio_split_opt, audio) {
            if split > 0 {
                segments.push(IngestSegment::Tokens(&tokens[..split]));
            }
            segments.push(IngestSegment::Audio { pcm, sample_rate });
            if split + 1 < tokens.len() {
                segments.push(IngestSegment::Tokens(&tokens[split + 1..]));
            }
        } else {
            segments.push(IngestSegment::Tokens(&tokens));
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
        match self.execution.append_segments(&segments) {
            Ok(()) => {
                self.phase = SessionPhase::PromptReady;
                self.terminal_committed = None;
                Ok(IngestSummary {
                    input_tokens: self.position().saturating_sub(before),
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
        let effective_opts;
        let opts = if opts.grammar.is_none() && !self.tools.is_empty() {
            if let Ok(gbnf) = tool_grammar(&self.tools, self.tool_format) {
                if let Ok(grammar) = crate::grammar::Grammar::parse(&gbnf) {
                    let mut modified = opts.clone();
                    modified.grammar = Some(Arc::new(grammar));
                    if modified.grammar_trigger_tokens.is_empty() {
                        let trigger = self
                            .profile
                            .tokenizer
                            .special_token_id(self.tool_format.call_start_marker())
                            .or_else(|| {
                                let t = self
                                    .profile
                                    .tokenizer
                                    .encode(self.tool_format.call_start_marker());
                                if t.len() == 1 { Some(t[0]) } else { None }
                            });
                        if let Some(tok) = trigger {
                            modified.grammar_trigger_tokens = vec![tok];
                        }
                    }
                    effective_opts = modified;
                    &effective_opts
                } else {
                    opts
                }
            } else {
                opts
            }
        } else {
            opts
        };
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
        let text = self.profile.tokenizer.decode(&collector.0);
        let tool_calls = parse_tool_calls(&text, self.tool_format).unwrap_or_default();
        Ok(TurnResult {
            text,
            tokens: collector.0,
            summary,
            tool_calls,
        })
    }

    /// Stream generated tokens into a text callback, returning the final turn result.
    pub fn stream_text<F>(
        &mut self,
        opts: &GenerateOpts,
        on_text: F,
    ) -> Result<TurnResult, CompleteError>
    where
        F: FnMut(&str),
    {
        struct StreamingCollector<F> {
            tokens: Vec<u32>,
            tokenizer: Arc<BpeTokenizer>,
            on_text: F,
            last_decoded_len: usize,
        }

        impl<F: FnMut(&str)> ModalitySink for StreamingCollector<F> {
            fn on_text_tokens(&mut self, new_tokens: &[u32]) {
                self.tokens.extend_from_slice(new_tokens);
                let current_text = self.tokenizer.decode(&self.tokens);
                if current_text.len() > self.last_decoded_len
                    && let Some(delta) = current_text.get(self.last_decoded_len..)
                {
                    (self.on_text)(delta);
                    self.last_decoded_len = current_text.len();
                }
            }

            fn on_done(&mut self, _reason: FinishReason) {}
        }

        let capacity = (opts.max_tokens as usize).min(4096);
        let tokenizer = Arc::clone(&self.profile.tokenizer);
        let mut collector = StreamingCollector {
            tokens: Vec::with_capacity(capacity),
            tokenizer,
            on_text,
            last_decoded_len: 0,
        };
        let report = self
            .generate_into(opts, &mut collector)
            .map_err(CompleteError::Validation)?;
        let summary = report.result.map_err(CompleteError::Execution)?;
        let full_text = self.profile.tokenizer.decode(&collector.tokens);
        let tool_calls = parse_tool_calls(&full_text, self.tool_format).unwrap_or_default();
        Ok(TurnResult {
            text: full_text,
            tokens: collector.tokens,
            summary,
            tool_calls,
        })
    }

    /// Complete generation constrained by a JSON Schema, returning the assistant response.
    pub fn complete_json(
        &mut self,
        opts: &GenerateOpts,
        schema_str: &str,
    ) -> Result<TurnResult, CompleteError> {
        let grammar = crate::grammar::Grammar::from_json_schema_str(schema_str).map_err(|e| {
            CompleteError::Validation(ValidationError::Generation(format!(
                "invalid JSON schema: {e}"
            )))
        })?;
        let mut constrained_opts = opts.clone();
        constrained_opts.grammar = Some(Arc::new(grammar));
        self.complete(&constrained_opts)
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
    fn image_input(&self) -> bool {
        self.session.capabilities.image_in
    }
    fn audio_input(&self) -> bool {
        self.session.capabilities.audio_in
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
        self.append_segments(&[IngestSegment::Tokens(tokens)])
    }
    fn append_segments(&mut self, segments: &[IngestSegment<'_>]) -> Result<(), IngestError> {
        self.session.last_ingest_recovery = None;
        self.session
            .with_ingest_recovery(|session| {
                for seg in segments {
                    match *seg {
                        IngestSegment::Tokens(toks) => {
                            if !toks.is_empty() {
                                session.append_tokens(toks)?;
                            }
                        }
                        IngestSegment::Image(bytes) => {
                            session.append_image(bytes)?;
                        }
                        IngestSegment::Audio { pcm, sample_rate } => {
                            session.append_audio(pcm, sample_rate)?;
                        }
                    }
                }
                Ok(())
            })
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
    fn checkpoint(&self) -> Result<checkpoint::SessionCheckpoint, CeraError> {
        self.session.checkpoint()
    }
    fn restore(&mut self, checkpoint: &checkpoint::SessionCheckpoint) -> Result<(), CeraError> {
        self.session.restore(checkpoint)
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

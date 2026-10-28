//! Tool (function) calling: tool definitions, chat-template plumbing, output
//! parsing, and grammar generation for constrained tool-call decoding.
//!
//! # Overview
//!
//! Tool calling has three moving parts, and cera already ships the hard one:
//!
//! 1. **Prompt side** — the tool schemas are injected into the chat template.
//!    Tool-trained models (LFM2, Qwen3, Llama-3.x) have a `{% if tools %}`
//!    branch in their GGUF chat template; [`crate::tokenizer::apply_chat_template_with_tools`]
//!    passes a `tools` array so that branch renders.
//! 2. **Output side** — the model emits a tool call in a *model-family-specific*
//!    wire format. [`parse_tool_calls`] turns that text back into structured
//!    [`ToolCall`]s.
//! 3. **Constraint (optional)** — [`tool_grammar`] builds a GBNF grammar that
//!    forces the output to a valid call for the declared tools, fed to the
//!    existing grammar-constrained decoder ([`crate::grammar`]).
//!
//! # Formats are not interchangeable
//!
//! There is no single "tool call format". LFM2 emits **Pythonic** calls
//! (`[get_weather(city="Paris")]`), while Hermes/Qwen emit JSON wrapped in
//! `<tool_call>…</tool_call>`. [`ToolFormat`] captures the family; parsing and
//! grammar generation branch on it.

use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value};

fn is_false(b: &bool) -> bool {
    !*b
}

/// A tool the model may call. Serializes to the JSON object shape chat
/// templates expect under the OpenAI "function" convention:
/// `{"name": ..., "description": ..., "parameters": {JSON Schema}}`.
///
/// `parameters` is a JSON Schema object describing the arguments (`type:
/// object`, `properties`, `required`). It is passed through verbatim to the
/// template and used to derive a constraint grammar.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments object. Defaults to an empty object
    /// schema when absent.
    #[serde(default = "empty_params")]
    pub parameters: Value,
    /// When `true`, this tool is withheld from the initial system prompt's
    /// `<tools>` block and only revealed mid-conversation through
    /// `<additional_tools>` or `tool_reference`, while remaining active for
    /// tool-call parsing and grammar generation.
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_loading: bool,
}

impl Default for ToolDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: None,
            parameters: empty_params(),
            defer_loading: false,
        }
    }
}

impl ToolDef {
    /// Construct a new non-deferred tool definition.
    pub fn new(name: impl Into<String>, description: Option<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description,
            parameters,
            defer_loading: false,
        }
    }

    /// Builder helper to mark this tool as deferred (`defer_loading`).
    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = defer_loading;
        self
    }
}

impl<'de> serde::Deserialize<'de> for ToolDef {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        let Some(obj) = v.as_object() else {
            return Err(serde::de::Error::custom("ToolDef must be a JSON object"));
        };
        // A field that is present (and not null) must be of its type: coercing a wrong-typed
        // `defer_loading` to false would show a tool the caller meant to withhold.
        let typed = |obj: &serde_json::Map<String, Value>,
                     key: &'static str,
                     want: &str,
                     ok: fn(&Value) -> bool|
         -> std::result::Result<Option<Value>, D::Error> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) if ok(v) => Ok(Some(v.clone())),
                Some(_) => Err(serde::de::Error::custom(format!(
                    "ToolDef `{key}` must be {want}"
                ))),
            }
        };
        let outer_defer =
            typed(obj, "defer_loading", "a boolean", Value::is_boolean)?.and_then(|d| d.as_bool());
        let fn_obj = if let Some(f) = obj.get("function").and_then(|f| f.as_object()) {
            f
        } else {
            obj
        };
        let name = fn_obj
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| serde::de::Error::missing_field("name"))?
            .to_string();
        let description = typed(fn_obj, "description", "a string", Value::is_string)?
            .and_then(|d| d.as_str().map(ToOwned::to_owned));
        // `parameters` is the schema as given: any JSON value round-trips through serde (the FFI
        // layer is where a non-object schema is refused), so it is not type-checked here.
        let parameters = fn_obj
            .get("parameters")
            .filter(|p| !p.is_null())
            .or_else(|| fn_obj.get("input_schema").filter(|p| !p.is_null()))
            .cloned()
            .unwrap_or_else(empty_params);
        let defer_loading = typed(fn_obj, "defer_loading", "a boolean", Value::is_boolean)?
            .and_then(|d| d.as_bool())
            .or(outer_defer)
            .unwrap_or(false);
        Ok(ToolDef {
            name,
            description,
            parameters,
            defer_loading,
        })
    }
}

fn empty_params() -> Value {
    serde_json::json!({ "type": "object", "properties": {} })
}

/// A tool call parsed from model output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// The call's arguments. Normally a JSON object (`{arg_name: value}`): the
    /// LFM2 Pythonic and LFM3 XML parsers always produce one, with each argument
    /// keeping its parsed JSON type. The Hermes parser passes through whatever
    /// the model emitted under `"arguments"`/`"parameters"`, so a malformed
    /// model output could in principle yield a non-object value here; consumers
    /// that require an object should check with [`Value::as_object`].
    pub arguments: Value,
}

/// Result of parsing a complete assistant output that may contain `<think>`
/// reasoning, prose content, and one or more tool calls.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct ParsedOutput {
    /// Assistant prose text outside `<tool_call>` and `<think>` blocks.
    pub content: String,
    /// Optional reasoning trace extracted from `<think>...</think>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls extracted from the output in emission order.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// A typed content part inside a [`ToolChatMessage`] (used for `tool` role
/// responses that carry `tool_reference` entries alongside or instead of text).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolContentPart {
    /// A plain text fragment.
    Text { text: String },
    /// Reference to a tool by name (typically a deferred tool with
    /// `defer_loading: true` whose schema should be revealed in an
    /// `<additional_tools>` block).
    ToolReference {
        #[serde(alias = "tool_name")]
        name: String,
    },
}

impl ToolContentPart {
    /// Construct a plain text content part.
    pub fn text(text: impl Into<String>) -> Self {
        ToolContentPart::Text { text: text.into() }
    }

    /// Construct a `tool_reference` content part naming a tool in `tools`.
    pub fn tool_reference(name: impl Into<String>) -> Self {
        ToolContentPart::ToolReference { name: name.into() }
    }
}

/// A chat message that can carry tool calls / tool results, for replaying a
/// tool-calling conversation into the chat template.
///
/// This is a *separate* struct from [`crate::tokenizer::ChatMessage`] on
/// purpose: [`crate::tokenizer::apply_chat_template`] is generic over
/// `Serialize`, and the multimodal path uses the same trick
/// ([`crate::tokenizer::ChatMessageMultimodal`]). Keeping tool fields here
/// means the plain text path stays a two-field struct with no churn at its
/// dozens of call sites.
///
/// Field semantics follow the OpenAI convention every tool-trained template
/// understands:
/// - `role = "assistant"` with `tool_calls` set -> a turn where the model
///   called tools (Hermes/Qwen/LFM3 templates read `message.tool_calls`; LFM2
///   keeps the call text in `content` instead, so set whichever the model needs).
/// - `role = "tool"` with `content` = the JSON result string (or
///   `content_parts` with [`ToolContentPart::ToolReference`] entries) -> a tool
///   result fed back to the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolChatMessage {
    pub role: String,
    pub content: String,
    pub content_parts: Vec<ToolContentPart>,
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Links a `tool` result back to the call it answers, when the template
    /// (or a downstream consumer) uses it. Omitted when `None`.
    pub tool_call_id: Option<String>,
    /// Optional reasoning/thinking trace (`<think>...</think>`) for `assistant` turns.
    pub reasoning_content: Option<String>,
    /// Optional deferred tool definitions revealed in `<additional_tools>` for `tool` turns.
    pub additional_tools: Option<Vec<ToolDef>>,
}

impl serde::Serialize for ToolChatMessage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("role", &self.role)?;

        if !self.content_parts.is_empty() {
            if self.content.is_empty() {
                map.serialize_entry("content", &self.content_parts)?;
            } else {
                let mut combined = Vec::with_capacity(self.content_parts.len() + 1);
                combined.push(ToolContentPart::Text {
                    text: self.content.clone(),
                });
                combined.extend(self.content_parts.iter().cloned());
                map.serialize_entry("content", &combined)?;
            }
        } else if self.role == "tool" {
            // a tool's own output: never a reason to fail the render
            match lenient_tool_content_parts(&self.content) {
                Some(parts) => map.serialize_entry("content", &parts)?,
                None => map.serialize_entry("content", &self.content)?,
            }
        } else {
            map.serialize_entry("content", &self.content)?;
        }

        if let Some(ref calls) = self.tool_calls {
            map.serialize_entry("tool_calls", calls)?;
        }
        if let Some(ref id) = self.tool_call_id {
            map.serialize_entry("tool_call_id", id)?;
        }
        if let Some(ref rc) = self.reasoning_content {
            map.serialize_entry("reasoning_content", rc)?;
        }
        if let Some(ref at) = self.additional_tools {
            map.serialize_entry("additional_tools", at)?;
        }
        map.end()
    }
}

impl ToolChatMessage {
    /// A plain role+content message (no tool calls, no id).
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ToolChatMessage {
            role: role.into(),
            content: content.into(),
            content_parts: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            additional_tools: None,
        }
    }

    /// An `assistant` turn that issued `calls`. `content` is any text the model
    /// produced alongside the calls (often empty).
    pub fn assistant_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        ToolChatMessage {
            role: "assistant".into(),
            content: content.into(),
            content_parts: Vec::new(),
            tool_calls: Some(calls),
            tool_call_id: None,
            reasoning_content: None,
            additional_tools: None,
        }
    }

    /// A `tool` result message answering `tool_call_id` (if known).
    pub fn tool_result(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        ToolChatMessage {
            role: "tool".into(),
            content: content.into(),
            content_parts: Vec::new(),
            tool_calls: None,
            tool_call_id,
            reasoning_content: None,
            additional_tools: None,
        }
    }

    /// A `tool` result message carrying typed content parts (e.g. `tool_reference`).
    pub fn tool_result_parts(parts: Vec<ToolContentPart>, tool_call_id: Option<String>) -> Self {
        ToolChatMessage {
            role: "tool".into(),
            content: String::new(),
            content_parts: parts,
            tool_calls: None,
            tool_call_id,
            reasoning_content: None,
            additional_tools: None,
        }
    }

    /// A `tool` result message revealing one or more deferred tools by `tool_reference` name.
    pub fn tool_references<I, S>(names: I, tool_call_id: Option<String>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let parts = names
            .into_iter()
            .map(ToolContentPart::tool_reference)
            .collect();
        Self::tool_result_parts(parts, tool_call_id)
    }

    /// Attach reasoning/thinking text (`<think>...</think>`) to this message.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }

    /// Attach `additional_tools` to reveal in a `tool` turn.
    pub fn with_additional_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.additional_tools = Some(tools);
        self
    }
}

/// Wire format a model family uses for tool calls. Determines both how output
/// is parsed and what constraint grammar is generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// LFM2 / LFM2.5: Pythonic call list wrapped in tool-call markers -
    /// `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>`.
    Lfm2Pythonic,
    /// LFM3: XML function and parameter blocks wrapped in `<tool_call>` -
    /// `<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>`.
    Lfm3Xml,
    /// Hermes-style (Qwen2.5/Qwen3, many fine-tunes): one or more
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` JSON blocks.
    Hermes,
}

impl ToolFormat {
    /// Best-effort detection from the model architecture string
    /// ([`crate::model::ModelConfig::architecture`]) or chat template source.
    /// Returns `None` when the input has no known tool-call convention.
    pub fn detect(architecture_or_template: &str) -> Option<ToolFormat> {
        match architecture_or_template {
            "lfm2" | "lfm2moe" => Some(ToolFormat::Lfm2Pythonic),
            "lfm3" | "lfm3moe" => Some(ToolFormat::Lfm3Xml),
            "qwen2" | "qwen2.5" | "qwen3" | "qwen3moe" | "qwen35" | "qwen3_5" | "qwen3.5" => {
                Some(ToolFormat::Hermes)
            }
            other => Self::detect_from_template(other),
        }
    }

    /// Detect the tool-call format from a GGUF Jinja chat template string.
    ///
    /// Order matters:
    /// 1. LFM3 XML templates contain `<function=` (or `<parameter=`) inside
    ///    `<tool_call>` and do not use `<|tool_call_start|>`.
    /// 2. LFM2 Pythonic templates contain `<|tool_call_start|>`.
    /// 3. Hermes templates contain `<tool_call>` with JSON payloads.
    pub fn detect_from_template(template: &str) -> Option<ToolFormat> {
        if (template.contains("<function=") || template.contains("<parameter="))
            && !template.contains("<|tool_call_start|>")
        {
            return Some(ToolFormat::Lfm3Xml);
        }
        if template.contains("<|tool_call_start|>") {
            return Some(ToolFormat::Lfm2Pythonic);
        }
        if template.contains("<tool_call>") {
            return Some(ToolFormat::Hermes);
        }
        None
    }

    /// Alias for [`Self::detect`] that checks both architecture names and chat
    /// template strings.
    pub fn detect_format(template_or_arch: &str) -> Option<ToolFormat> {
        Self::detect(template_or_arch)
    }

    /// The literal string that opens a tool-call section for this format. Used
    /// as the lazy grammar trigger and as a parse anchor.
    pub fn call_start_marker(self) -> &'static str {
        match self {
            ToolFormat::Lfm2Pythonic => "<|tool_call_start|>",
            ToolFormat::Lfm3Xml | ToolFormat::Hermes => "<tool_call>",
        }
    }

    /// The literal string that closes a tool-call section for this format.
    pub fn call_end_marker(self) -> &'static str {
        match self {
            ToolFormat::Lfm2Pythonic => "<|tool_call_end|>",
            ToolFormat::Lfm3Xml | ToolFormat::Hermes => "</tool_call>",
        }
    }
}

/// Detect the [`ToolFormat`] from a chat template string or model architecture
/// identifier.
pub fn detect_format(template_or_arch: &str) -> Option<ToolFormat> {
    ToolFormat::detect(template_or_arch)
}

/// Parse tool calls out of generated model text.
///
/// Tolerant by design: a missing closing marker (common when generation stops
/// on EOS right after the call) still parses. Returns an empty vec when the
/// text contains no tool call: that is the normal "the model answered in prose"
/// case, not an error. Returns `Err` only when a call *section* is present but
/// malformed enough that no call can be recovered.
pub fn parse_tool_calls(text: &str, format: ToolFormat) -> Result<Vec<ToolCall>> {
    parse_tool_calls_with_tools(text, format, &[])
}

/// Parse tool calls out of generated model text, using `tools` schemas when
/// present to guide parameter type coercion (for example, keeping a
/// `"type": "string"` parameter as a JSON string in `ToolFormat::Lfm3Xml` even
/// when its raw value looks like a number or boolean).
pub fn parse_tool_calls_with_tools(
    text: &str,
    format: ToolFormat,
    tools: &[ToolDef],
) -> Result<Vec<ToolCall>> {
    // The same view of the text as `parse_output`: a call quoted inside the model's reasoning
    // is not a call, whichever format is asked for.
    let (_, after_think) = extract_thinking(text);
    match format {
        ToolFormat::Lfm2Pythonic => parse_lfm2_pythonic(&after_think),
        ToolFormat::Lfm3Xml => parse_lfm3_xml(&after_think, tools),
        ToolFormat::Hermes => parse_hermes(&after_think),
    }
}

/// Parse a complete assistant output into assistant prose (`content`), optional
/// `<think>...</think>` reasoning (`reasoning_content`), and [`ToolCall`]s.
pub fn parse_output(text: &str, format: ToolFormat) -> Result<ParsedOutput> {
    parse_output_with_tools(text, format, &[])
}

/// Like [`parse_output`], but uses `tools` schemas to coerce parameter types
/// in [`ToolFormat::Lfm3Xml`].
pub fn parse_output_with_tools(
    text: &str,
    format: ToolFormat,
    tools: &[ToolDef],
) -> Result<ParsedOutput> {
    let (reasoning_content, after_think) = extract_thinking(text);
    let (content, tool_calls) = match format {
        ToolFormat::Lfm2Pythonic => split_and_parse_lfm2(&after_think)?,
        ToolFormat::Lfm3Xml => split_and_parse_lfm3(&after_think, tools)?,
        ToolFormat::Hermes => split_and_parse_hermes(&after_think)?,
    };
    Ok(ParsedOutput {
        content,
        reasoning_content,
        tool_calls,
    })
}

// ── Prompt formatting helpers ───────────────────────────────────────────────

/// Format a single [`ToolDef`] as a single-line JSON specification that is character for
/// character what the LFM3 `tool_to_json` template macro prints: sorted keys, `, ` and `: `
/// separators, HTML-significant characters escaped (`defer_loading` is always stripped so it
/// never leaks into the rendered prompt). A test renders the template and compares.
pub fn format_tool_def(tool: &ToolDef) -> String {
    // The template prints the definition through `tojson`: keys in sorted order, `, ` and `: `
    // separators at every depth, and HTML-significant characters as `\u00XX` escapes.
    let mut def = serde_json::Map::new();
    def.insert("name".into(), Value::String(tool.name.clone()));
    if let Some(desc) = &tool.description {
        def.insert("description".into(), Value::String(desc.clone()));
    }
    def.insert("parameters".into(), tool.parameters.clone());
    let mut out = String::new();
    write_template_json(&Value::Object(def), &mut out);
    out
}

/// `value` as the chat template's `tojson` filter prints it (see [`format_tool_def`]).
fn write_template_json(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_template_json_str(key, out);
                out.push_str(": ");
                write_template_json(&map[key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_template_json(item, out);
            }
            out.push(']');
        }
        Value::String(s) => write_template_json_str(s, out),
        other => out.push_str(&other.to_string()),
    }
}

fn write_template_json_str(s: &str, out: &mut String) {
    let quoted = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    for c in quoted.chars() {
        match c {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\'' => out.push_str("\\u0027"),
            c => out.push(c),
        }
    }
}

/// Format the initial tool-definition block for a system prompt, excluding any
/// tools marked with `defer_loading: true`. Returns an empty string when no
/// non-deferred tools remain.
pub fn format_tools_for_prompt(tools: &[ToolDef], format: ToolFormat) -> String {
    let visible: Vec<&ToolDef> = tools.iter().filter(|t| !t.defer_loading).collect();
    if visible.is_empty() {
        return String::new();
    }
    match format {
        ToolFormat::Lfm2Pythonic => {
            let items: Vec<String> = visible.into_iter().map(format_tool_def).collect();
            format!("<|tool_list_start|>[{}]<|tool_list_end|>", items.join(", "))
        }
        ToolFormat::Lfm3Xml | ToolFormat::Hermes => {
            let lines: Vec<String> = visible.into_iter().map(format_tool_def).collect();
            format!("<tools>\n{}\n</tools>", lines.join("\n"))
        }
    }
}

/// Format a single [`ToolCall`] in the wire format for `format`.
pub fn format_tool_call(call: &ToolCall, format: ToolFormat) -> String {
    match format {
        ToolFormat::Lfm3Xml => {
            let mut out = format!("<tool_call>\n<function={}>\n", call.name);
            if let Some(obj) = call.arguments.as_object() {
                for (k, v) in obj {
                    out.push_str(&format!(
                        "<parameter={k}>\n{}\n</parameter>\n",
                        format_lfm3_value(v)
                    ));
                }
            }
            out.push_str("</function>\n</tool_call>");
            out
        }
        ToolFormat::Hermes => {
            let payload = serde_json::json!({
                "name": call.name,
                "arguments": call.arguments,
            });
            format!("<tool_call>{payload}</tool_call>")
        }
        ToolFormat::Lfm2Pythonic => {
            format!(
                "<|tool_call_start|>[{}]<|tool_call_end|>",
                format_pythonic_call(call)
            )
        }
    }
}

/// Format one or more [`ToolCall`]s in the wire format for `format`.
pub fn format_tool_calls(calls: &[ToolCall], format: ToolFormat) -> String {
    if calls.is_empty() {
        return String::new();
    }
    match format {
        ToolFormat::Lfm3Xml => calls
            .iter()
            .map(|c| format_tool_call(c, format))
            .collect::<Vec<_>>()
            .join("\n\n"),
        ToolFormat::Hermes => calls
            .iter()
            .map(|c| format_tool_call(c, format))
            .collect::<Vec<_>>()
            .join("\n"),
        ToolFormat::Lfm2Pythonic => {
            let inner = calls
                .iter()
                .map(format_pythonic_call)
                .collect::<Vec<_>>()
                .join(", ");
            format!("<|tool_call_start|>[{inner}]<|tool_call_end|>")
        }
    }
}

fn format_lfm3_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "true".into()
            } else {
                "false".into()
            }
        }
        Value::Null => "null".into(),
        Value::Number(n) => n.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
    }
}

fn format_pythonic_call(call: &ToolCall) -> String {
    let mut args = Vec::new();
    if let Some(obj) = call.arguments.as_object() {
        for (k, v) in obj {
            args.push(format!("{k}={}", format_pythonic_value(v)));
        }
    }
    format!("{}({})", call.name, args.join(", "))
}

fn format_pythonic_value(v: &Value) -> String {
    match v {
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

/// Format a tool response string in `<tool_response>\n...\n</tool_response>`.
///
/// If `content` already contains `<tool_response>`, it is returned as-is.
/// If `content` is a JSON-encoded `tool_reference` item or list of items, each
/// referenced tool name is resolved against `tools` and rendered inside an
/// `<additional_tools>` block.
pub fn format_tool_response(content: &str, tools: &[ToolDef]) -> Result<String> {
    if content.contains("<tool_response>") {
        return Ok(content.to_string());
    }
    if let Some(parts) = try_parse_tool_content_parts(content)? {
        return format_tool_response_parts(&parts, tools);
    }
    Ok(format!("<tool_response>\n{content}\n</tool_response>"))
}

/// Format a slice of [`ToolContentPart`]s into a `<tool_response>` block,
/// expanding any [`ToolContentPart::ToolReference`] items into
/// `<additional_tools>` definitions looked up from `tools`. Consecutive
/// `tool_reference` entries share a single `<additional_tools>` block.
pub fn format_tool_response_parts(parts: &[ToolContentPart], tools: &[ToolDef]) -> Result<String> {
    let mut body = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            body.push('\n');
        }
        match part {
            ToolContentPart::Text { text } => {
                body.push_str(text);
            }
            ToolContentPart::ToolReference { name } => {
                if name.is_empty() {
                    bail!("tool_reference part is missing a non-empty tool name");
                }
                let prev_is_ref = i > 0
                    && matches!(
                        parts.get(i - 1),
                        Some(ToolContentPart::ToolReference { .. })
                    );
                let next_is_ref = matches!(
                    parts.get(i + 1),
                    Some(ToolContentPart::ToolReference { .. })
                );
                if !prev_is_ref {
                    body.push_str("<additional_tools>\n");
                }
                let tool = tools.iter().find(|t| t.name == *name).ok_or_else(|| {
                    anyhow::anyhow!("tool_reference names a tool that is not in tools: {name}")
                })?;
                body.push_str(&format_tool_def(tool));
                if !next_is_ref {
                    body.push_str("\n</additional_tools>");
                }
            }
        }
    }
    Ok(format!("<tool_response>\n{body}\n</tool_response>"))
}

/// Inspect `s` to see if it is JSON containing one or more `tool_reference`
/// content parts. Returns `Ok(None)` when `s` is ordinary text or unrelated
/// JSON, `Ok(Some(parts))` when valid `tool_reference` parts are present, and
/// `Err` when a `tool_reference` object is malformed (e.g. missing `"name"`).
pub(crate) fn try_parse_tool_content_parts(s: &str) -> Result<Option<Vec<ToolContentPart>>> {
    parse_tool_content_parts(s, false)
}

/// [`try_parse_tool_content_parts`] for the render path, where the content is a tool's own
/// output and must never fail the whole prompt: a part that is malformed or of a kind this crate
/// does not model (an image block, a reference without a name) is kept as text of its JSON, and
/// content that cannot be read as parts at all stays as it is.
pub(crate) fn lenient_tool_content_parts(s: &str) -> Option<Vec<ToolContentPart>> {
    parse_tool_content_parts(s, true).ok().flatten()
}

fn parse_tool_content_parts(s: &str, lenient: bool) -> Result<Option<Vec<ToolContentPart>>> {
    let trimmed = s.trim();
    if !(trimmed.starts_with('[') || trimmed.starts_with('{')) {
        return Ok(None);
    }
    let Ok(val) = serde_json::from_str::<Value>(trimmed) else {
        return Ok(None);
    };
    let items: Vec<Value> = match val {
        Value::Array(arr) => arr,
        Value::Object(_) => vec![val],
        _ => return Ok(None),
    };
    let has_tool_ref = items.iter().any(|item| {
        item.as_object()
            .and_then(|o| o.get("type"))
            .and_then(|t| t.as_str())
            == Some("tool_reference")
    });
    if !has_tool_ref {
        return Ok(None);
    }
    let mut parts = Vec::with_capacity(items.len());
    for item in items {
        match tool_content_part(&item) {
            Ok(part) => parts.push(part),
            Err(_) if lenient => parts.push(ToolContentPart::Text {
                text: item.to_string(),
            }),
            Err(e) => return Err(e),
        }
    }
    Ok(Some(parts))
}

/// One content part of a tool result.
fn tool_content_part(item: &Value) -> Result<ToolContentPart> {
    let Some(obj) = item.as_object() else {
        bail!("content part must be a JSON object");
    };
    let Some(ty) = obj.get("type").and_then(|t| t.as_str()) else {
        bail!("content part is missing 'type'");
    };
    match ty {
        "tool_reference" => {
            let name = obj
                .get("name")
                .or_else(|| obj.get("tool_name"))
                .and_then(|n| n.as_str())
                .filter(|n| !n.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool_reference part is missing 'name'"))?;
            Ok(ToolContentPart::ToolReference {
                name: name.to_string(),
            })
        }
        "text" => {
            let text = obj.get("text").and_then(|t| t.as_str()).unwrap_or_default();
            Ok(ToolContentPart::Text {
                text: text.to_string(),
            })
        }
        other => bail!("unsupported tool content part type: '{other}'"),
    }
}

// ── Grammar generation (constrained tool calls) ─────────────────────────────

/// Build a GBNF grammar that constrains output to a valid call for one of
/// `tools`, in the given `format`. Feed the compiled grammar to
/// [`crate::grammar::Grammar::parse`] and set it on
/// [`crate::session::GenerateOpts::grammar`].
///
/// The grammar constrains the tool-call **interior** — the marker tokens
/// (`<|tool_call_start|>` etc.) are special tokens the grammar mask cannot emit,
/// so they are handled by the lazy trigger instead (see
/// [`crate::session::GenerateOpts::grammar_trigger_tokens`]). Under a lazy
/// trigger the model emits the start marker on its own, this grammar constrains
/// the call, and it deactivates on completion so the model can close the marker.
///
/// What is enforced: the call structure, the function name (must be a declared
/// tool), each argument name (must be one of that tool's schema properties), and
/// each argument's outer value syntax (string/integer/number/boolean/array/object,
/// plus scalar `enum` literal sets). Nested array item and object property schemas
/// are not enforced. Neither are required arguments or no-duplicate/ordering
/// constraints: arguments may appear in any order and any subset. This compiler
/// is separate from the general JSON Schema compiler. A token limit can still
/// truncate output, and lazy triggering permits prose; validate parsed calls
/// and their arguments before execution.
///
/// For the LFM2 Pythonic format, tool and argument names must be valid Python
/// identifiers (`[A-Za-z_][A-Za-z0-9_]*`) — that is what the model was trained
/// on and what [`parse_tool_calls`] reads back. A name containing e.g. `-` or
/// `.` would be emitted verbatim by the grammar but not round-trip through the
/// parser; keep tool/argument names identifier-safe for that format.
///
/// For the LFM3 XML format, names sit inside `<function=NAME>` and `<parameter=NAME>` tags:
/// they must be non-empty, free of `<`, `>` and line breaks, and have no leading or trailing
/// space (anything else is refused here). A `string` parameter's value may hold any text except
/// the closing `</parameter>` tag; a string `enum` is emitted as written (quoted only when an
/// untyped variant would read back as JSON). The Hermes JSON format quotes names, so it has no
/// such restriction.
pub fn tool_grammar(tools: &[ToolDef], format: ToolFormat) -> Result<String> {
    if tools.is_empty() {
        bail!("tool_grammar: no tools provided");
    }
    match format {
        ToolFormat::Lfm2Pythonic => {
            // The Pythonic grammar emits names verbatim (`name(arg=...)`), and the
            // parser reads them back as Python identifiers. A name with `-`, `.`,
            // a leading digit, etc. would compile into a grammar the parser can't
            // round-trip: the model could satisfy the grammar yet produce output
            // `parse_tool_calls` rejects. Fail loudly at build time instead.
            for tool in tools {
                ensure!(
                    is_py_ident(&tool.name),
                    "tool name `{}` is not a valid Python identifier; the LFM2 \
                     Pythonic format requires [A-Za-z_][A-Za-z0-9_]* names",
                    tool.name
                );
                for (name, _) in properties(tool) {
                    ensure!(
                        is_py_ident(&name),
                        "argument name `{name}` of tool `{}` is not a valid Python \
                         identifier; the LFM2 Pythonic format requires \
                         [A-Za-z_][A-Za-z0-9_]* names",
                        tool.name
                    );
                }
            }
            Ok(lfm2_grammar(tools))
        }
        ToolFormat::Lfm3Xml => {
            // Names sit between `=` and `>` on one line and are read back trimmed: one with
            // `<`, `>`, a line break or edge spaces would compile into a grammar the parser
            // cannot read back (the model could satisfy it and still produce an unparseable call).
            for tool in tools {
                ensure!(
                    is_lfm3_name(&tool.name),
                    "tool name `{}` cannot be written in the LFM3 XML format (it must be \
                     non-empty, have no `<`, `>` or line break, and no leading or trailing space)",
                    tool.name
                );
                for (name, _) in properties(tool) {
                    ensure!(
                        is_lfm3_name(&name),
                        "argument name `{name}` of tool `{}` cannot be written in the LFM3 XML \
                         format (it must be non-empty, have no `<`, `>` or line break, and no \
                         leading or trailing space)",
                        tool.name
                    );
                }
            }
            Ok(lfm3_xml_grammar(tools))
        }
        ToolFormat::Hermes => Ok(hermes_grammar(tools)),
    }
}

/// Whether `name` round-trips through `<function=NAME>` / `<parameter=NAME>` tags.
fn is_lfm3_name(name: &str) -> bool {
    !name.is_empty() && name.trim() == name && !name.contains(['<', '>', '\n', '\r'])
}

/// Alias for [`tool_grammar`]: generate a GBNF grammar constraining tool-call
/// output for `tools` in `format`.
pub fn generate_gbnf(tools: &[ToolDef], format: ToolFormat) -> Result<String> {
    tool_grammar(tools, format)
}

/// True when `s` is a valid Python identifier for our purposes:
/// `[A-Za-z_][A-Za-z0-9_]*` (ASCII). Matches what the Pythonic grammar can emit
/// and what the Pythonic parser's `parse_ident` reads back.
fn is_py_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Shared value/whitespace rules appended to every generated grammar. `bt`
/// selects boolean/null spelling: Pythonic (`True`/`False`/`None`) vs JSON
/// (`true`/`false`/`null`). String syntax (double-quoted, JSON escapes) is
/// common to both.
fn value_rules(pythonic: bool) -> String {
    let (t, f, n) = if pythonic {
        ("\"True\"", "\"False\"", "\"None\"")
    } else {
        ("\"true\"", "\"false\"", "\"null\"")
    };
    format!(
        r#"tc-value ::= tc-str | tc-num | tc-bool | tc-null | tc-array | tc-object
tc-str ::= "\"" tc-char* "\""
tc-char ::= [^"\\\x00-\x1F] | "\\" tc-esc
tc-esc ::= ["\\/bfnrt] | "u" tc-hex tc-hex tc-hex tc-hex
tc-hex ::= [0-9a-fA-F]
tc-int ::= "-"? ("0" | [1-9] [0-9]*)
tc-num ::= tc-int ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
tc-bool ::= {t} | {f}
tc-null ::= {n}
tc-array ::= "[" tc-ws ( tc-value ( tc-ws "," tc-ws tc-value )* )? tc-ws "]"
tc-object ::= "{{" tc-ws ( tc-str tc-ws ":" tc-ws tc-value ( tc-ws "," tc-ws tc-str tc-ws ":" tc-ws tc-value )* )? tc-ws "}}"
tc-ws ::= [ \t\n\r]*
"#,
    )
}

/// A GBNF literal that matches the JSON encoding of `s`. `s` is first
/// JSON-escaped (quotes, backslashes, control chars) via serde_json, then the
/// resulting quoted string is wrapped as a GBNF literal. Correct for both JSON
/// (Hermes) and pythonic (LFM2) string emission, which agree on the
/// `\" \\ \n \t` escapes. Without this, a value like `C:\path` would compile to
/// a grammar literal matching invalid JSON (`"C:\path"`, one backslash) that no
/// correct emitter can produce — making that value impossible under constraint.
fn json_str_gbnf(s: &str) -> String {
    // `serde_json::to_string` on a string is infallible; it yields the quoted,
    // escaped JSON form (e.g. `"C:\\path"`).
    let json = serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""));
    gbnf_lit(&json)
}

/// Escape a string to appear as a GBNF double-quoted literal body.
fn gbnf_lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// GBNF alternation matching a single JSON value as a literal (for `enum`).
/// Strings become quoted literals; scalars their literal text; anything
/// exotic falls through to the generic `tc-value`.
fn enum_literal(v: &Value, pythonic: bool) -> Option<String> {
    match v {
        Value::String(s) => Some(json_str_gbnf(s)),
        Value::Bool(b) => Some(gbnf_lit(match (b, pythonic) {
            (true, true) => "True",
            (false, true) => "False",
            (true, false) => "true",
            (false, false) => "false",
        })),
        Value::Number(n) => Some(gbnf_lit(&n.to_string())),
        Value::Null => Some(gbnf_lit(if pythonic { "None" } else { "null" })),
        _ => None,
    }
}

/// The value rule (a GBNF fragment, not a named rule) for one property schema.
fn value_for_schema(schema: &Value, pythonic: bool) -> String {
    // `enum` overrides type: constrain to the literal set — but only when
    // *every* variant is representable as a literal. If any variant is
    // non-scalar (array/object), fall back to the type rule rather than
    // silently forbidding those variants (which dropping them would do).
    if let Some(Value::Array(variants)) = schema.get("enum")
        && !variants.is_empty()
    {
        let lits: Vec<Option<String>> =
            variants.iter().map(|v| enum_literal(v, pythonic)).collect();
        if lits.iter().all(Option::is_some) {
            let joined = lits.into_iter().flatten().collect::<Vec<_>>().join(" | ");
            return format!("( {joined} )");
        }
    }
    let ty = schema.get("type").and_then(|t| t.as_str());
    match ty {
        Some("string") => "tc-str".into(),
        Some("integer") => "tc-int".into(),
        Some("number") => "tc-num".into(),
        Some("boolean") => "tc-bool".into(),
        Some("array") => "tc-array".into(),
        Some("object") => "tc-object".into(),
        Some("null") => "tc-null".into(),
        _ => "tc-value".into(),
    }
}

/// Properties of a tool's parameter schema, as `(name, subschema)` pairs.
fn properties(tool: &ToolDef) -> Vec<(String, Value)> {
    tool.parameters
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

/// LFM2 Pythonic grammar: `[ name(arg=value, …), … ]`.
fn lfm2_grammar(tools: &[ToolDef]) -> String {
    let mut out = String::new();
    // A tool call section is a Python list of one or more calls.
    out.push_str(
        "root ::= tc-ws \"[\" tc-ws tc-call ( tc-ws \",\" tc-ws tc-call )* tc-ws \"]\" tc-ws\n",
    );
    let call_alts: Vec<String> = (0..tools.len()).map(|i| format!("tc-call-{i}")).collect();
    out.push_str(&format!("tc-call ::= {}\n", call_alts.join(" | ")));

    for (i, tool) in tools.iter().enumerate() {
        let props = properties(tool);
        out.push_str(&format!(
            "tc-call-{i} ::= {} tc-ws \"(\" tc-ws tc-args-{i} tc-ws \")\"\n",
            gbnf_lit(&tool.name)
        ));
        if props.is_empty() {
            // No parameters → empty arg list.
            out.push_str(&format!("tc-args-{i} ::= \"\"\n"));
            continue;
        }
        // Any subset, any order, of this tool's named+typed pairs.
        out.push_str(&format!(
            "tc-args-{i} ::= ( tc-pair-{i} ( tc-ws \",\" tc-ws tc-pair-{i} )* )?\n"
        ));
        let pair_alts: Vec<String> = (0..props.len())
            .map(|j| format!("tc-pair-{i}-{j}"))
            .collect();
        out.push_str(&format!("tc-pair-{i} ::= {}\n", pair_alts.join(" | ")));
        for (j, (name, schema)) in props.iter().enumerate() {
            out.push_str(&format!(
                "tc-pair-{i}-{j} ::= {} tc-ws \"=\" tc-ws {}\n",
                gbnf_lit(name),
                value_for_schema(schema, true)
            ));
        }
    }
    out.push_str(&value_rules(true));
    out
}

/// Hermes grammar: `<tool_call>` JSON blocks: the markers are special
/// tokens handled by the trigger, so this constrains the JSON object body:
/// `{"name": "<one of the tools>", "arguments": { ... }}`.
fn hermes_grammar(tools: &[ToolDef]) -> String {
    let mut out = String::new();
    out.push_str("root ::= tc-ws tc-call tc-ws\n");
    let call_alts: Vec<String> = (0..tools.len()).map(|i| format!("tc-call-{i}")).collect();
    out.push_str(&format!("tc-call ::= {}\n", call_alts.join(" | ")));

    for (i, tool) in tools.iter().enumerate() {
        let props = properties(tool);
        // {"name": "tool", "arguments": <args>}: the name *value* is a JSON
        // string, so it carries its own surrounding quotes.
        out.push_str(&format!(
            "tc-call-{i} ::= \"{{\" tc-ws \"\\\"name\\\"\" tc-ws \":\" tc-ws {} tc-ws \",\" tc-ws \"\\\"arguments\\\"\" tc-ws \":\" tc-ws tc-args-{i} tc-ws \"}}\"\n",
            json_str_gbnf(&tool.name)
        ));
        if props.is_empty() {
            out.push_str(&format!("tc-args-{i} ::= \"{{\" tc-ws \"}}\"\n"));
            continue;
        }
        out.push_str(&format!(
            "tc-args-{i} ::= \"{{\" tc-ws ( tc-pair-{i} ( tc-ws \",\" tc-ws tc-pair-{i} )* )? tc-ws \"}}\"\n"
        ));
        let pair_alts: Vec<String> = (0..props.len())
            .map(|j| format!("tc-pair-{i}-{j}"))
            .collect();
        out.push_str(&format!("tc-pair-{i} ::= {}\n", pair_alts.join(" | ")));
        for (j, (name, schema)) in props.iter().enumerate() {
            // JSON key is a quoted string.
            out.push_str(&format!(
                "tc-pair-{i}-{j} ::= {} tc-ws \":\" tc-ws {}\n",
                json_str_gbnf(name),
                value_for_schema(schema, false)
            ));
        }
    }
    out.push_str(&value_rules(false));
    out
}

/// GBNF for `name`: any text that does not contain `terminator` (a string that starts with
/// `<`). A value may hold `<`, blank lines and carriage returns, as the parser reads them;
/// only the closing tag ends it, so only the closing tag is excluded. The text is split at
/// each `<`; after one, a small chain of states tracks how much of the terminator has been
/// seen, and a character that breaks the match goes back to plain text.
fn lfm3_text_rule(name: &str, terminator: &str) -> String {
    let rest: Vec<char> = terminator.chars().skip(1).collect();
    let n = rest.len();
    let class = |c: char| {
        // inside a negated class a few characters need escaping
        match c {
            '\\' | ']' | '^' | '-' => format!("\\{c}"),
            _ => c.to_string(),
        }
    };
    let mut out = format!("{name} ::= [^<]* ( \"<\" {name}-0 )*\n");
    for (k, &expected) in rest.iter().enumerate() {
        // after "<" + rest[..k]: end the segment, continue with a char that breaks the match,
        // or take the next char of the terminator (not for the last one: that would close it)
        let other = format!("( [^<{}] [^<]* )?", class(expected));
        if k + 1 < n {
            out.push_str(&format!(
                "{name}-{k} ::= {other} | {} {name}-{}\n",
                gbnf_lit(&expected.to_string()),
                k + 1
            ));
        } else {
            out.push_str(&format!("{name}-{k} ::= {other}\n"));
        }
    }
    out
}

fn lfm3_enum_literal(v: &Value, typed_string: bool) -> Option<String> {
    match v {
        // Read back by `parse_lfm3_param_value`: a `string` parameter keeps the text as
        // written; an untyped one is parsed as JSON first, so a variant such as `true` or `3`
        // must be quoted to stay a string, and any other variant is plain text.
        Value::String(s) => Some(
            if typed_string || serde_json::from_str::<Value>(s).is_err() {
                gbnf_lit(s)
            } else {
                json_str_gbnf(s)
            },
        ),
        Value::Bool(b) => Some(gbnf_lit(if *b { "true" } else { "false" })),
        Value::Number(n) => Some(gbnf_lit(&n.to_string())),
        Value::Null => Some(gbnf_lit("null")),
        _ => None,
    }
}

fn lfm3_value_for_schema(schema: &Value) -> String {
    if let Some(Value::Array(variants)) = schema.get("enum")
        && !variants.is_empty()
    {
        let typed_string = schema.get("type").and_then(|t| t.as_str()) == Some("string");
        let lits: Vec<Option<String>> = variants
            .iter()
            .map(|v| lfm3_enum_literal(v, typed_string))
            .collect();
        if lits.iter().all(Option::is_some) {
            let joined = lits.into_iter().flatten().collect::<Vec<_>>().join(" | ");
            return format!("( {joined} )");
        }
    }
    let ty = schema.get("type").and_then(|t| t.as_str());
    match ty {
        Some("string") => "tc-xml-str".into(),
        Some("integer") => "tc-int".into(),
        Some("number") => "tc-num".into(),
        Some("boolean") => "tc-bool".into(),
        Some("array") => "tc-array".into(),
        Some("object") => "tc-object".into(),
        Some("null") => "tc-null".into(),
        _ => "( tc-value | tc-xml-str )".into(),
    }
}

/// LFM3 XML grammar:
/// `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVAL\n</parameter>\n</function>\n</tool_call>`.
/// Also accepts the interior `<function=...>\n...\n</function>` when `<tool_call>`
/// was already consumed by a lazy grammar trigger.
fn lfm3_xml_grammar(tools: &[ToolDef]) -> String {
    let mut out = String::new();
    out.push_str("root ::= tc-ws tc-call-block ( tc-ws tc-call-block )* tc-ws\n");
    out.push_str(
        "tc-call-block ::= \"<tool_call>\\n\" tc-call \"\\n</tool_call>\" | tc-call ( \"\\n</tool_call>\" )?\n",
    );
    let call_alts: Vec<String> = (0..tools.len()).map(|i| format!("tc-call-{i}")).collect();
    out.push_str(&format!("tc-call ::= {}\n", call_alts.join(" | ")));

    for (i, tool) in tools.iter().enumerate() {
        let props = properties(tool);
        out.push_str(&format!(
            "tc-call-{i} ::= {} tc-args-{i} \"</function>\"\n",
            gbnf_lit(&format!("<function={}>\n", tool.name))
        ));
        if props.is_empty() {
            out.push_str(&format!("tc-args-{i} ::= \"\"\n"));
            continue;
        }
        out.push_str(&format!("tc-args-{i} ::= tc-pair-{i}*\n"));
        let pair_alts: Vec<String> = (0..props.len())
            .map(|j| format!("tc-pair-{i}-{j}"))
            .collect();
        out.push_str(&format!("tc-pair-{i} ::= {}\n", pair_alts.join(" | ")));
        for (j, (name, schema)) in props.iter().enumerate() {
            out.push_str(&format!(
                "tc-pair-{i}-{j} ::= {} {} \"\\n</parameter>\\n\"\n",
                gbnf_lit(&format!("<parameter={name}>\n")),
                lfm3_value_for_schema(schema)
            ));
        }
    }
    out.push_str(&lfm3_text_rule("tc-xml-str", "</parameter>"));
    out.push_str(&value_rules(false));
    out
}

// ── LFM2 Pythonic parser ────────────────────────────────────────────────────

/// Parse `<|tool_call_start|>[name(a="x", b=3), other()]<|tool_call_end|>`.
///
/// The interior is a Python list of function calls. Fully-wrapped output (with
/// markers) is the normal case; a bare `[...]` list with no markers is also
/// accepted (e.g. constrained decode run eagerly without a trigger token, or a
/// caller passing just the interior).
fn parse_lfm2_pythonic(text: &str) -> Result<Vec<ToolCall>> {
    // Walk every `<|tool_call_start|>...<|tool_call_end|>` section, not just the
    // first: a model may emit several sections in one turn, and the Hermes
    // parser handles multiples, so this stays consistent. A missing final end
    // marker is tolerated (the tail is treated as the last section).
    let mut calls = Vec::new();
    let mut rest = text;
    let mut saw_marker = false;
    while let Some(start) = rest.find("<|tool_call_start|>") {
        saw_marker = true;
        let after = &rest[start + "<|tool_call_start|>".len()..];
        let (inner, next) = match after.find("<|tool_call_end|>") {
            Some(end) => (&after[..end], &after[end + "<|tool_call_end|>".len()..]),
            None => (after, ""),
        };
        parse_pythonic_section(inner.trim(), &mut calls)?;
        rest = next;
    }
    // No markers at all -> tolerate a bare Pythonic call list, but only when it
    // actually looks like one (starts with `[`) so ordinary prose isn't
    // misparsed as a call. Without markers there is no explicit call intent, so
    // a leading `[` that doesn't parse as a call list (ordinary prose, a JSON
    // array like `[1, 2, 3]`, etc.) is treated as "no tool call": the error is
    // swallowed rather than propagated. A *marked* section still errors on a
    // malformed body (handled above), because the markers assert call intent.
    if !saw_marker {
        let trimmed = text.trim();
        if trimmed.starts_with('[') {
            let mut bare = Vec::new();
            if parse_pythonic_section(trimmed, &mut bare).is_ok() {
                calls.append(&mut bare);
            }
        }
    }
    Ok(calls)
}

/// Parse one `[call, call, ...]` section body, appending to `calls`.
fn parse_pythonic_section(inner: &str, calls: &mut Vec<ToolCall>) -> Result<()> {
    // Strip the outer list brackets if present: `[call, call]` -> `call, call`.
    let body = inner
        .strip_prefix('[')
        .map(|s| s.strip_suffix(']').unwrap_or(s))
        .unwrap_or(inner)
        .trim();
    if body.is_empty() {
        return Ok(());
    }
    let mut p = PyParser::new(body);
    loop {
        p.skip_ws();
        if p.eof() {
            break;
        }
        calls.push(p.parse_call()?);
        p.skip_ws();
        if p.peek() == Some(',') {
            p.bump();
        }
    }
    Ok(())
}

/// Minimal recursive-descent parser for the Pythonic call subset LFM2 emits:
/// `name(key=value, ...)` with values being str / int / float / bool / None /
/// list / dict. Enough for real tool calls without pulling in a Python parser.
struct PyParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> PyParser<'a> {
    fn new(s: &'a str) -> Self {
        PyParser {
            s: s.as_bytes(),
            i: 0,
        }
    }
    fn eof(&self) -> bool {
        self.i >= self.s.len()
    }
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).map(|&b| b as char)
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    /// `name ( args )` -> a [`ToolCall`].
    fn parse_call(&mut self) -> Result<ToolCall> {
        self.skip_ws();
        let name = self.parse_ident();
        if name.is_empty() {
            bail!("expected function name in tool call");
        }
        self.skip_ws();
        if self.bump() != Some('(') {
            bail!("expected '(' after tool name '{name}'");
        }
        let mut args = Map::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(')') {
                self.bump();
                break;
            }
            let key = self.parse_ident();
            if key.is_empty() {
                bail!("expected argument name in call to '{name}'");
            }
            self.skip_ws();
            if self.bump() != Some('=') {
                bail!("expected '=' after argument '{key}' in call to '{name}'");
            }
            self.skip_ws();
            let val = self.parse_value()?;
            args.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some(')') => {
                    self.bump();
                    break;
                }
                _ => bail!("expected ',' or ')' in argument list for '{name}'"),
            }
        }
        Ok(ToolCall {
            name,
            arguments: Value::Object(args),
        })
    }

    fn parse_ident(&mut self) -> String {
        let start = self.i;
        // Python identifiers are `[A-Za-z_][A-Za-z0-9_]*`: a leading digit is
        // not an identifier. Enforcing the start char here matches the module
        // doc and keeps a bracket-leading non-call (`[1, 2, 3]`) from being
        // read as a call whose name is a number.
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => self.i += 1,
            _ => return String::new(),
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.i += 1;
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.s[start..self.i]).into_owned()
    }

    /// A Python literal -> JSON value.
    fn parse_value(&mut self) -> Result<Value> {
        self.skip_ws();
        match self.peek() {
            Some('"') | Some('\'') => self.parse_string(),
            Some('[') => self.parse_list(),
            Some('{') => self.parse_dict(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(_) => self.parse_keyword(),
            None => bail!("unexpected end of input while parsing value"),
        }
    }

    fn parse_string(&mut self) -> Result<Value> {
        // Work at the byte level: string CONTENT may be multibyte UTF-8
        // (`city="Zürich"`, emoji, CJK), so accumulate raw bytes and decode
        // once at the end. `bump()`-as-char would mangle every non-ASCII byte.
        // The opening quote (`'`/`"`) is ASCII; `parse_value` already confirmed
        // it, so index it directly.
        let quote = self.s[self.i];
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        while self.i < self.s.len() {
            let b = self.s[self.i];
            self.i += 1;
            match b {
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        bail!("unterminated escape in string literal");
                    };
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'/' => out.push(b'/'),
                        b'\\' => out.push(b'\\'),
                        b'\'' => out.push(b'\''),
                        b'"' => out.push(b'"'),
                        // `\uXXXX` -> the codepoint's UTF-8 bytes, combining a
                        // `😀`-style UTF-16 surrogate pair into the
                        // astral codepoint. These escapes are permitted by the
                        // generated grammar's `tc-esc`, so the parser must decode
                        // them to agree with constrained output. Lone/unpaired
                        // surrogates fall back to U+FFFD.
                        b'u' => {
                            let cp = self.parse_u_codepoint()?;
                            let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => {
                            out.push(b'\\');
                            out.push(other);
                        }
                    }
                }
                b if b == quote => {
                    return Ok(Value::String(String::from_utf8_lossy(&out).into_owned()));
                }
                b => out.push(b),
            }
        }
        bail!("unterminated string literal")
    }

    /// Parse a `\uXXXX` escape body (the `\u` is already consumed) into a Unicode
    /// scalar value, combining a UTF-16 surrogate pair (`😀`) into the
    /// single astral codepoint it encodes. A lone/unpaired high surrogate is
    /// returned as-is (the caller maps it to U+FFFD).
    fn parse_u_codepoint(&mut self) -> Result<u32> {
        let hi = self.parse_hex4()?;
        // High surrogate directly followed by `\u` -> try to pair it.
        if (0xD800..=0xDBFF).contains(&hi)
            && self.s.get(self.i) == Some(&b'\\')
            && self.s.get(self.i + 1) == Some(&b'u')
        {
            let before_second = self.i; // start of the trailing `\u...`
            self.i += 2; // tentatively consume the second `\u`
            let lo = self.parse_hex4()?;
            if (0xDC00..=0xDFFF).contains(&lo) {
                return Ok(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00));
            }
            // Not a valid low surrogate: rewind so the trailing escape is
            // decoded on its own rather than consumed and dropped. A high
            // surrogate followed by a non-low-surrogate `\u` escape (e.g. the
            // second escape is `A`) then yields U+FFFD (the unpaired high
            // surrogate, via the caller) followed by the second escape's own
            // character, instead of just U+FFFD with the second escape lost.
            self.i = before_second;
        }
        Ok(hi)
    }

    /// Parse exactly four hex digits (a `\uXXXX` escape body) into a codepoint.
    fn parse_hex4(&mut self) -> Result<u32> {
        let mut cp = 0u32;
        for _ in 0..4 {
            let Some(&b) = self.s.get(self.i) else {
                bail!("truncated \\u escape in string literal");
            };
            let d = (b as char)
                .to_digit(16)
                .ok_or_else(|| anyhow::anyhow!("bad hex digit in \\u escape"))?;
            cp = cp * 16 + d;
            self.i += 1;
        }
        Ok(cp)
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.i;
        if self.peek() == Some('-') {
            self.bump();
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.bump();
            } else if c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                is_float = true;
                self.bump();
            } else {
                break;
            }
        }
        let tok = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        if is_float {
            let f: f64 = tok
                .parse()
                .map_err(|_| anyhow::anyhow!("bad float '{tok}'"))?;
            // A literal beyond f64 range (e.g. `1e9999`) parses to +-inf, which
            // `serde_json` would silently turn into `null`; reject it instead of
            // dropping the value: same guard as the integer-overflow path below.
            ensure!(f.is_finite(), "float literal '{tok}' out of range");
            Ok(serde_json::json!(f))
        } else {
            // Prefer an exact integer, but fall back to f64 when the literal
            // exceeds i64 range rather than failing the whole parse (llama.cpp
            // treats an out-of-range integer as a number, not an error).
            match tok.parse::<i64>() {
                Ok(n) => Ok(serde_json::json!(n)),
                Err(_) => {
                    let f: f64 = tok
                        .parse()
                        .map_err(|_| anyhow::anyhow!("bad integer '{tok}'"))?;
                    // A literal beyond f64 range parses to +-inf, which
                    // `serde_json` would silently turn into `null`; reject it
                    // instead of dropping the value.
                    ensure!(f.is_finite(), "integer literal '{tok}' out of range");
                    Ok(serde_json::json!(f))
                }
            }
        }
    }

    /// `True` / `False` / `None` (Python spelling).
    fn parse_keyword(&mut self) -> Result<Value> {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphabetic() {
                self.bump();
            } else {
                break;
            }
        }
        let kw = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        match kw {
            "True" | "true" => Ok(Value::Bool(true)),
            "False" | "false" => Ok(Value::Bool(false)),
            "None" | "null" => Ok(Value::Null),
            other => bail!("unexpected literal '{other}' in tool call"),
        }
    }

    fn parse_list(&mut self) -> Result<Value> {
        self.bump(); // [
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                self.bump();
                break;
            }
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some(']') => {
                    self.bump();
                    break;
                }
                _ => bail!("expected ',' or ']' in list literal"),
            }
        }
        Ok(Value::Array(items))
    }

    fn parse_dict(&mut self) -> Result<Value> {
        self.bump(); // {
        let mut map = Map::new();
        loop {
            self.skip_ws();
            if self.peek() == Some('}') {
                self.bump();
                break;
            }
            let key = match self.parse_value()? {
                Value::String(s) => s,
                other => bail!("dict keys must be strings, got {other}"),
            };
            self.skip_ws();
            if self.bump() != Some(':') {
                bail!("expected ':' in dict literal");
            }
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some('}') => {
                    self.bump();
                    break;
                }
                _ => bail!("expected ',' or '}}' in dict literal"),
            }
        }
        Ok(Value::Object(map))
    }
}

// ── LFM3 XML parser ─────────────────────────────────────────────────────────

/// Strip at most one leading newline (`\r\n` or `\n`) and at most one trailing
/// newline (`\r\n` or `\n`) from `s`.
fn strip_one_newline(s: &str) -> &str {
    let s = s
        .strip_prefix("\r\n")
        .or_else(|| s.strip_prefix('\n'))
        .unwrap_or(s);
    s.strip_suffix("\r\n")
        .or_else(|| s.strip_suffix('\n'))
        .unwrap_or(s)
}

/// Private-use markers that stand for a tool-call body while thinking is extracted.
const MASK_OPEN: char = '\u{E000}';
const MASK_CLOSE: char = '\u{E001}';

/// Replace each tool-call section (`<tool_call>...</tool_call>`, or LFM2's
/// `<|tool_call_start|>...<|tool_call_end|>`) with `MASK_OPEN index MASK_CLOSE` and return the
/// masked text with the sections it hid. A parameter value may contain `<think>` text (code,
/// documentation), which must not be read as the model's reasoning.
fn mask_tool_calls(text: &str) -> (String, Vec<String>) {
    const SECTIONS: [(&str, &str); 2] = [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call_start|>", "<|tool_call_end|>"),
    ];
    let mut masked = String::with_capacity(text.len());
    let mut bodies = Vec::new();
    let mut rest = text;
    loop {
        let next = SECTIONS
            .iter()
            .filter_map(|(open, close)| rest.find(open).map(|at| (at, *open, *close)))
            .min_by_key(|(at, _, _)| *at);
        let Some((at, open, close)) = next else {
            masked.push_str(rest);
            break;
        };
        masked.push_str(&rest[..at]);
        let body_start = at + open.len();
        let close_pos = rest[body_start..].find(close);
        let is_bare_in_think = match close_pos {
            None => rest[body_start..].contains("</think>"),
            Some(c) => {
                let interior = &rest[body_start..body_start + c];
                if let Some(tc) = interior.find("</think>") {
                    let before_tc = &interior[..tc];
                    let after_tc = &interior[tc + "</think>".len()..];
                    after_tc.contains(open)
                        || !(before_tc.contains("<function=")
                            || before_tc.contains('{')
                            || before_tc.contains('['))
                } else {
                    false
                }
            }
        };
        if is_bare_in_think {
            masked.push_str(open);
            rest = &rest[body_start..];
            continue;
        }
        // an unclosed section runs to the end of the text
        let end = close_pos.map_or(rest.len(), |c| body_start + c + close.len());
        masked.push(MASK_OPEN);
        masked.push_str(&bodies.len().to_string());
        masked.push(MASK_CLOSE);
        bodies.push(rest[at..end].to_string());
        rest = &rest[end..];
    }
    (masked, bodies)
}

/// Undo [`mask_tool_calls`] on `text`.
fn unmask_tool_calls(text: &str, bodies: &[String]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(MASK_OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + MASK_OPEN.len_utf8()..];
        let Some(close) = after.find(MASK_CLOSE) else {
            out.push_str(&rest[open..]);
            return out;
        };
        match after[..close]
            .parse::<usize>()
            .ok()
            .and_then(|i| bodies.get(i))
        {
            Some(body) => out.push_str(body),
            None => out
                .push_str(&rest[open..open + MASK_OPEN.len_utf8() + close + MASK_CLOSE.len_utf8()]),
        }
        rest = &after[close + MASK_CLOSE.len_utf8()..];
    }
    out.push_str(rest);
    out
}

/// Extract `<think>...</think>` blocks (or a leading `...</think>` when the
/// prompt already opened `<think>`), returning `(reasoning_content, rest)`.
fn extract_thinking(text: &str) -> (Option<String>, String) {
    let (masked, bodies) = mask_tool_calls(text);
    let (reasoning, rest) = extract_thinking_masked(&masked);
    (
        reasoning.map(|r| unmask_tool_calls(&r, &bodies)),
        unmask_tool_calls(&rest, &bodies),
    )
}

/// The body of [`extract_thinking`], over text whose tool-call sections are masked.
fn extract_thinking_masked(text: &str) -> (Option<String>, String) {
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut remaining = String::new();
    let mut rest = text;

    // Handle the case where `<think>` was part of the prompt prefix, so the
    // generated text closes with `</think>` before any `<think>` opens.
    if let Some(close_pos) = rest.find("</think>") {
        let open_pos = rest.find("<think>");
        if open_pos.is_none_or(|op| close_pos < op) {
            let thought = strip_one_newline(&rest[..close_pos]).trim();
            if !thought.is_empty() {
                reasoning_parts.push(thought.to_string());
            }
            rest = &rest[close_pos + "</think>".len()..];
        }
    }

    while let Some(start) = rest.find("<think>") {
        remaining.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        match after.find("</think>") {
            Some(end) => {
                let thought = strip_one_newline(&after[..end]).trim();
                if !thought.is_empty() {
                    reasoning_parts.push(thought.to_string());
                }
                rest = &after[end + "</think>".len()..];
            }
            None => {
                // a thought that never closes ends where a tool call begins: the call is
                // the model's answer, not part of the reasoning
                let (thinking, tail) = match after.find(MASK_OPEN) {
                    Some(call_at) => after.split_at(call_at),
                    None => (after, ""),
                };
                let thought = strip_one_newline(thinking).trim();
                if !thought.is_empty() {
                    reasoning_parts.push(thought.to_string());
                }
                rest = tail;
                break;
            }
        }
    }
    remaining.push_str(rest);

    let reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n"))
    };
    (reasoning, remaining)
}

fn parse_lfm3_xml(text: &str, tools: &[ToolDef]) -> Result<Vec<ToolCall>> {
    let (_, after_think) = extract_thinking(text);
    let (_, calls) = split_and_parse_lfm3(&after_think, tools)?;
    Ok(calls)
}

fn split_and_parse_lfm3(text: &str, tools: &[ToolDef]) -> Result<(String, Vec<ToolCall>)> {
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut saw_tool_call = false;

    while let Some(start) = rest.find("<tool_call>") {
        saw_tool_call = true;
        let before = &rest[..start];
        if !before.trim().is_empty() {
            content_parts.push(before.trim());
        }
        let after = &rest[start + "<tool_call>".len()..];
        let (tc_body, next_rest) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        let tc_body = strip_one_newline(tc_body);
        parse_lfm3_functions(tc_body, tools, &mut calls)?;
        rest = next_rest;
    }

    if !saw_tool_call {
        let trimmed = rest.trim();
        if trimmed.starts_with("<function=") {
            parse_lfm3_functions(trimmed, tools, &mut calls)?;
        } else if !trimmed.is_empty() {
            content_parts.push(trimmed);
        }
    } else if !rest.trim().is_empty() {
        content_parts.push(rest.trim());
    }

    Ok((content_parts.join("\n"), calls))
}

fn parse_lfm3_functions(body: &str, tools: &[ToolDef], calls: &mut Vec<ToolCall>) -> Result<()> {
    let mut rest = body;
    let mut saw_fn = false;
    while let Some(fn_start) = rest.find("<function=") {
        saw_fn = true;
        if !rest[..fn_start].trim().is_empty() {
            bail!(
                "unexpected text before <function=...> in <tool_call>: {:?}",
                rest[..fn_start].trim()
            );
        }
        let after_prefix = &rest[fn_start + "<function=".len()..];
        let Some(gt_pos) = after_prefix.find('>') else {
            bail!("unterminated <function=...> opening tag in <tool_call>");
        };
        let fn_name = after_prefix[..gt_pos].trim();
        if fn_name.is_empty() {
            bail!("empty function name in <function=> tag");
        }
        let after_fn_open = &after_prefix[gt_pos + 1..];
        let (fn_body, next_rest) = match after_fn_open.find("</function>") {
            Some(end) => (
                &after_fn_open[..end],
                &after_fn_open[end + "</function>".len()..],
            ),
            None => (after_fn_open, ""),
        };
        let fn_body = strip_one_newline(fn_body);
        let tool_def = tools.iter().find(|t| t.name == fn_name);
        let args = parse_lfm3_parameters(fn_body, fn_name, tool_def)?;
        calls.push(ToolCall {
            name: fn_name.to_string(),
            arguments: Value::Object(args),
        });
        rest = next_rest;
    }
    ensure!(saw_fn, "<tool_call> block contains no <function=...> tag");
    if !rest.trim().is_empty() {
        bail!(
            "unexpected trailing text after </function> in <tool_call>: {:?}",
            rest.trim()
        );
    }
    Ok(())
}

fn parse_lfm3_parameters(
    fn_body: &str,
    fn_name: &str,
    tool_def: Option<&ToolDef>,
) -> Result<Map<String, Value>> {
    let mut args = Map::new();
    let mut rest = fn_body;
    while !rest.trim().is_empty() {
        let Some(param_start) = rest.find("<parameter=") else {
            bail!(
                "unexpected content in <function={fn_name}> body: {:?}",
                rest.trim()
            );
        };
        if !rest[..param_start].trim().is_empty() {
            bail!(
                "unexpected text before <parameter=...> in <function={fn_name}>: {:?}",
                rest[..param_start].trim()
            );
        }
        let after_prefix = &rest[param_start + "<parameter=".len()..];
        let Some(gt_pos) = after_prefix.find('>') else {
            bail!("unterminated <parameter=...> opening tag in call to '{fn_name}'");
        };
        let param_name = after_prefix[..gt_pos].trim();
        if param_name.is_empty() {
            bail!("empty parameter name in call to '{fn_name}'");
        }
        let after_param_open = &after_prefix[gt_pos + 1..];
        let (raw_val, next_rest) = match after_param_open.find("</parameter>") {
            Some(end) => (
                &after_param_open[..end],
                &after_param_open[end + "</parameter>".len()..],
            ),
            None => bail!("missing </parameter> for '{param_name}' in call to '{fn_name}'"),
        };
        let val_str = strip_one_newline(raw_val);
        let prop_schema = tool_def
            .and_then(|t| t.parameters.get("properties"))
            .and_then(|p| p.get(param_name));
        let val = parse_lfm3_param_value(val_str, prop_schema);
        args.insert(param_name.to_string(), val);
        rest = next_rest;
    }
    Ok(args)
}

fn parse_lfm3_param_value(val_str: &str, prop_schema: Option<&Value>) -> Value {
    let schema_ty = prop_schema
        .and_then(|s| s.get("type"))
        .and_then(|t| t.as_str());
    if schema_ty == Some("string") {
        return Value::String(val_str.to_string());
    }
    match serde_json::from_str::<Value>(val_str) {
        Ok(v) => v,
        Err(_) => Value::String(val_str.to_string()),
    }
}

fn split_and_parse_lfm2(text: &str) -> Result<(String, Vec<ToolCall>)> {
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut saw_marker = false;
    while let Some(start) = rest.find("<|tool_call_start|>") {
        saw_marker = true;
        let before = &rest[..start];
        if !before.trim().is_empty() {
            content_parts.push(before.trim());
        }
        let after = &rest[start + "<|tool_call_start|>".len()..];
        let (inner, next) = match after.find("<|tool_call_end|>") {
            Some(end) => (&after[..end], &after[end + "<|tool_call_end|>".len()..]),
            None => (after, ""),
        };
        parse_pythonic_section(inner.trim(), &mut calls)?;
        rest = next;
    }
    if !saw_marker {
        let trimmed = rest.trim();
        if trimmed.starts_with('[') {
            let mut bare = Vec::new();
            if parse_pythonic_section(trimmed, &mut bare).is_ok() {
                calls.append(&mut bare);
            } else if !trimmed.is_empty() {
                content_parts.push(trimmed);
            }
        } else if !trimmed.is_empty() {
            content_parts.push(trimmed);
        }
    } else if !rest.trim().is_empty() {
        content_parts.push(rest.trim());
    }
    Ok((content_parts.join("\n"), calls))
}

fn split_and_parse_hermes(text: &str) -> Result<(String, Vec<ToolCall>)> {
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let before = &rest[..start];
        if !before.trim().is_empty() {
            content_parts.push(before.trim());
        }
        let after = &rest[start + "<tool_call>".len()..];
        let (json_str, next) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        let v: Value = serde_json::from_str(json_str.trim())
            .map_err(|e| anyhow::anyhow!("invalid <tool_call> JSON: {e}"))?;
        let name = v
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("<tool_call> missing string 'name'"))?
            .to_string();
        let arguments = v
            .get("arguments")
            .or_else(|| v.get("parameters"))
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        calls.push(ToolCall { name, arguments });
        rest = next;
    }
    if !rest.trim().is_empty() {
        content_parts.push(rest.trim());
    }
    Ok((content_parts.join("\n"), calls))
}

// ── Hermes / JSON parser ────────────────────────────────────────────────────

/// Parse one or more `<tool_call>{json}</tool_call>` blocks. Each block's JSON
/// is `{"name": ..., "arguments": {...}}` (some models use `"parameters"`).
fn parse_hermes(text: &str) -> Result<Vec<ToolCall>> {
    let (_, calls) = split_and_parse_hermes(text)?;
    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_tags_inside_a_tool_call_are_part_of_the_call() {
        // a parameter value may legitimately contain the literal tags (code, docs)
        let body = "<tool_call>\n<function=note>\n<parameter=text>\na <think>b</think> c\n</parameter>\n</function>\n</tool_call>";
        let (reasoning, rest) = extract_thinking(&format!("hello {body} bye"));
        assert_eq!(reasoning, None);
        assert_eq!(rest, format!("hello {body} bye"));
        let calls = parse_tool_calls(body, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert!(
            calls[0].arguments.to_string().contains("<think>b</think>"),
            "{:?}",
            calls[0]
        );
    }

    #[test]
    fn an_unterminated_think_does_not_swallow_the_tool_call_after_it() {
        let call = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        let (reasoning, rest) = extract_thinking(&format!("<think>plan {call}"));
        assert_eq!(reasoning.as_deref(), Some("plan"));
        assert_eq!(rest, call);
    }

    #[test]
    fn every_format_ignores_tool_calls_quoted_inside_thinking() {
        // the two entry points agree: a call mentioned in the model's reasoning is not a call
        let quoted =
            "<think>maybe <tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call></think>done";
        for format in [
            ToolFormat::Hermes,
            ToolFormat::Lfm3Xml,
            ToolFormat::Lfm2Pythonic,
        ] {
            let direct = parse_tool_calls(quoted, format).unwrap();
            let full = parse_output(quoted, format).unwrap().tool_calls;
            assert_eq!(
                direct.len(),
                full.len(),
                "{format:?}: entry points disagree"
            );
            assert!(direct.is_empty(), "{format:?}: a quoted call was parsed");
        }
    }

    #[test]
    fn detect_arch() {
        assert_eq!(ToolFormat::detect("lfm2"), Some(ToolFormat::Lfm2Pythonic));
        assert_eq!(
            ToolFormat::detect("lfm2moe"),
            Some(ToolFormat::Lfm2Pythonic)
        );
        assert_eq!(ToolFormat::detect("lfm3"), Some(ToolFormat::Lfm3Xml));
        assert_eq!(ToolFormat::detect("lfm3moe"), Some(ToolFormat::Lfm3Xml));
        assert_eq!(ToolFormat::detect("qwen2"), Some(ToolFormat::Hermes));
        assert_eq!(ToolFormat::detect("qwen2.5"), Some(ToolFormat::Hermes));
        assert_eq!(ToolFormat::detect("qwen3"), Some(ToolFormat::Hermes));
        assert_eq!(ToolFormat::detect("gpt2"), None);
    }

    #[test]
    fn detect_format_from_templates() {
        let lfm3_tmpl = "<tool_call>\n<function={{ func.name }}>\n<parameter={{ arg_name }}>\n{{ arg_value }}\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(detect_format(lfm3_tmpl), Some(ToolFormat::Lfm3Xml));
        assert_eq!(
            ToolFormat::detect_from_template(lfm3_tmpl),
            Some(ToolFormat::Lfm3Xml)
        );

        let lfm2_tmpl = "<|tool_call_start|>[{{ call }}]<|tool_call_end|>";
        assert_eq!(detect_format(lfm2_tmpl), Some(ToolFormat::Lfm2Pythonic));

        let hermes_tmpl = "<tool_call>{\"name\": \"foo\", \"arguments\": {}}</tool_call>";
        assert_eq!(detect_format(hermes_tmpl), Some(ToolFormat::Hermes));

        assert_eq!(detect_format("plain chat template without tools"), None);
    }

    #[test]
    fn lfm2_marker_less_non_call_is_empty_not_error() {
        // A bare `[...]` that isn't a Pythonic call list (JSON array, prose) has
        // no explicit call intent -> "no tool call" (empty vec), not an error.
        for text in ["[1, 2, 3]", "[just some prose]", "[\"a\", \"b\"]"] {
            let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
            assert!(calls.is_empty(), "expected no calls for {text:?}");
        }
        // But a marker-delimited malformed body still errors (markers assert
        // call intent).
        assert!(
            parse_tool_calls(
                "<|tool_call_start|>[1, 2, 3]<|tool_call_end|>",
                ToolFormat::Lfm2Pythonic
            )
            .is_err()
        );
        // A genuine bare call list still parses.
        let calls =
            parse_tool_calls("[get_weather(city=\"Paris\")]", ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn lfm2_grammar_rejects_non_identifier_names() {
        // A tool name that isn't a Python identifier can't round-trip through
        // the Pythonic parser, so grammar generation must reject it.
        let bad_tool = ToolDef {
            name: "get-weather".into(),
            description: None,
            parameters: empty_params(),
            defer_loading: false,
        };
        assert!(tool_grammar(&[bad_tool], ToolFormat::Lfm2Pythonic).is_err());

        // A bad *argument* name is rejected too.
        let bad_arg = ToolDef {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "2fa": { "type": "string" } }
            }),
            defer_loading: false,
        };
        assert!(tool_grammar(std::slice::from_ref(&bad_arg), ToolFormat::Lfm2Pythonic).is_err());
        // The same non-identifier names are fine for Hermes (JSON quotes them).
        assert!(tool_grammar(&[bad_arg], ToolFormat::Hermes).is_ok());
    }

    #[test]
    fn lfm2_single_call() {
        let text = "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));
    }

    #[test]
    fn lfm2_multi_arg_types() {
        let text = "<|tool_call_start|>[f(a=1, b=2.5, c=True, d=None, e=\"hi\")]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            serde_json::json!({"a": 1, "b": 2.5, "c": true, "d": null, "e": "hi"})
        );
    }

    #[test]
    fn lfm2_nested_and_multiple_calls() {
        let text = "<|tool_call_start|>[a(x=[1, 2, 3]), b(y={\"k\": \"v\"})]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, serde_json::json!({"x": [1, 2, 3]}));
        assert_eq!(calls[1].arguments, serde_json::json!({"y": {"k": "v"}}));
    }

    #[test]
    fn lfm2_non_ascii_string_arg() {
        // Multibyte UTF-8 (umlaut, CJK, emoji) must survive parsing.
        let text = "<|tool_call_start|>[f(city=\"Zürich\", note=\"日本語 👍\")]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls[0].arguments["city"], "Zürich");
        assert_eq!(calls[0].arguments["note"], "日本語 👍");
    }

    #[test]
    fn lfm2_unicode_and_slash_escapes() {
        let text = "<|tool_call_start|>[open(url=\"http:\\/\\/x.com\\u002Fp\")]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls[0].arguments["url"], "http://x.com/p");
    }

    #[test]
    fn lfm2_big_integer_falls_back_to_float() {
        // > i64::MAX must not fail the whole parse.
        let text = "<|tool_call_start|>[wire(amount=10000000000000000000)]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert!(calls[0].arguments["amount"].is_number());
        // A value beyond f64 range must error, not become a silent null.
        let huge = "9".repeat(400);
        let text = format!("<|tool_call_start|>[wire(amount={huge})]<|tool_call_end|>");
        assert!(parse_tool_calls(&text, ToolFormat::Lfm2Pythonic).is_err());
    }

    #[test]
    fn lfm2_surrogate_pair_escape() {
        // `😀` is the UTF-16 encoding of 😀 (U+1F600).
        let text = "<|tool_call_start|>[react(emoji=\"\\uD83D\\uDE00\")]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls[0].arguments["emoji"], "\u{1F600}");
    }

    #[test]
    fn lfm2_out_of_range_float_is_error_not_null() {
        // A float literal beyond f64 range parses to +-inf; serde_json would turn
        // that into `null`, silently corrupting the argument. Reject instead.
        let text = "<|tool_call_start|>[f(x=1e9999)]<|tool_call_end|>";
        assert!(parse_tool_calls(text, ToolFormat::Lfm2Pythonic).is_err());
        // A normal float still parses.
        let ok = "<|tool_call_start|>[f(x=1.5)]<|tool_call_end|>";
        let calls = parse_tool_calls(ok, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls[0].arguments["x"], 1.5);
    }

    #[test]
    fn lfm2_unpaired_high_surrogate_keeps_following_escape() {
        // A high surrogate followed by a `\u` escape that is NOT a low surrogate:
        // the high surrogate becomes U+FFFD, but the second escape (`A`) must
        // still be decoded: not consumed and dropped.
        let text = "<|tool_call_start|>[f(s=\"\\uD83D\\u0041\")]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls[0].arguments["s"], "\u{FFFD}A");
    }

    #[test]
    fn lfm2_multiple_sections() {
        // Two separate marker sections -> both calls recovered.
        let text = "<|tool_call_start|>[a(x=1)]<|tool_call_end|> then \
                    <|tool_call_start|>[b(y=2)]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn enum_with_backslash_value_is_emittable() {
        // A string enum value with a backslash must compile to a grammar
        // literal matching VALID JSON (`"C:\\path"`), not the raw bytes.
        let tools = vec![ToolDef {
            name: "open".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"p": {"type": "string", "enum": ["C:\\path"]}}
            }),
            defer_loading: false,
        }];
        let g = compile(&tool_grammar(&tools, ToolFormat::Hermes).unwrap());
        // The JSON-correct form (two backslashes) must be accepted.
        assert!(accepts_complete(
            &g,
            br#"{"name": "open", "arguments": {"p": "C:\\path"}}"#
        ));
    }

    #[test]
    fn non_scalar_enum_falls_back_to_type() {
        // enum containing an array variant -> must not forbid the array; falls
        // back to the declared type instead of dropping the non-scalar member.
        let tools = vec![ToolDef {
            name: "f".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "array", "enum": [["a"], ["b"]]}}
            }),
            defer_loading: false,
        }];
        let g = compile(&tool_grammar(&tools, ToolFormat::Lfm2Pythonic).unwrap());
        assert!(accepts_complete(&g, b"[f(x=[\"a\"])]"));
    }

    #[test]
    fn lfm2_missing_end_marker_is_tolerated() {
        let text = "<|tool_call_start|>[ping()]";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn lfm2_single_quotes() {
        let text = "<|tool_call_start|>[echo(msg='it\\'s ok')]<|tool_call_end|>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls[0].arguments, serde_json::json!({"msg": "it's ok"}));
    }

    #[test]
    fn lfm2_bare_list_without_markers() {
        // No markers at all, but a bare Pythonic list -> parsed.
        let calls =
            parse_tool_calls("[get_weather(city=\"Paris\")]", ToolFormat::Lfm2Pythonic).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        // Prose that isn't a call -> empty, not an error.
        let calls = parse_tool_calls("no tools here", ToolFormat::Lfm2Pythonic).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn no_call_is_empty_not_error() {
        let calls = parse_tool_calls("The weather is sunny.", ToolFormat::Lfm2Pythonic).unwrap();
        assert!(calls.is_empty());
        let calls = parse_tool_calls("Just prose.", ToolFormat::Lfm3Xml).unwrap();
        assert!(calls.is_empty());
        let calls = parse_tool_calls("Just prose.", ToolFormat::Hermes).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn hermes_single_and_parameters_alias() {
        let text = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
        let calls = parse_tool_calls(text, ToolFormat::Hermes).unwrap();
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));

        let alias = "<tool_call>{\"name\": \"f\", \"parameters\": {\"a\": 1}}</tool_call>";
        let calls = parse_tool_calls(alias, ToolFormat::Hermes).unwrap();
        assert_eq!(calls[0].arguments, serde_json::json!({"a": 1}));
    }

    #[test]
    fn hermes_multiple_blocks() {
        let text = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n\
                    <tool_call>{\"name\": \"b\", \"arguments\": {\"x\": true}}</tool_call>";
        let calls = parse_tool_calls(text, ToolFormat::Hermes).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn lfm3_xml_single_and_multiple_params() {
        let text = "<tool_call>\n\
                    <function=markItemPurchased>\n\
                    <parameter=id>\n\
                    abc\n\
                    </parameter>\n\
                    <parameter=qty>\n\
                    3\n\
                    </parameter>\n\
                    </function>\n\
                    </tool_call>";
        let parsed = parse_output(text, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.reasoning_content, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "markItemPurchased");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({"id": "abc", "qty": 3})
        );
    }

    #[test]
    fn lfm3_xml_all_param_value_types_and_schema_string_override() {
        let text = "<tool_call>\n\
                    <function=configure>\n\
                    <parameter=label>\n\
                    blue widget\n\
                    </parameter>\n\
                    <parameter=numeric_string_id>\n\
                    12345\n\
                    </parameter>\n\
                    <parameter=count>\n\
                    42\n\
                    </parameter>\n\
                    <parameter=ratio>\n\
                    2.5\n\
                    </parameter>\n\
                    <parameter=verbose>\n\
                    true\n\
                    </parameter>\n\
                    <parameter=tags>\n\
                    [\"alpha\", \"beta\"]\n\
                    </parameter>\n\
                    <parameter=meta>\n\
                    {\"env\": \"prod\", \"retries\": 2}\n\
                    </parameter>\n\
                    </function>\n\
                    </tool_call>";
        let tools = vec![ToolDef {
            name: "configure".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "numeric_string_id": {"type": "string"},
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "verbose": {"type": "boolean"},
                    "tags": {"type": "array"},
                    "meta": {"type": "object"}
                }
            }),
            defer_loading: false,
        }];
        let parsed = parse_output_with_tools(text, ToolFormat::Lfm3Xml, &tools).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({
                "label": "blue widget",
                "numeric_string_id": "12345",
                "count": 42,
                "ratio": 2.5,
                "verbose": true,
                "tags": ["alpha", "beta"],
                "meta": {"env": "prod", "retries": 2}
            })
        );
    }

    #[test]
    fn lfm3_xml_multiple_calls_and_surrounding_text_and_thinking() {
        let text = "<think>Need to search and then purchase.</think>Marking it now.\n\
                    <tool_call>\n\
                    <function=searchCatalog>\n\
                    <parameter=query>\n\
                    blue widget\n\
                    </parameter>\n\
                    </function>\n\
                    </tool_call>\n\
                    <tool_call>\n\
                    <function=markItemPurchased>\n\
                    <parameter=id>\n\
                    abc\n\
                    </parameter>\n\
                    <parameter=qty>\n\
                    3\n\
                    </parameter>\n\
                    </function>\n\
                    </tool_call>";
        let parsed = parse_output(text, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(
            parsed.reasoning_content.as_deref(),
            Some("Need to search and then purchase.")
        );
        assert_eq!(parsed.content, "Marking it now.");
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].name, "searchCatalog");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({"query": "blue widget"})
        );
        assert_eq!(parsed.tool_calls[1].name, "markItemPurchased");
        assert_eq!(
            parsed.tool_calls[1].arguments,
            serde_json::json!({"id": "abc", "qty": 3})
        );
    }

    #[test]
    fn lfm3_xml_preserves_multiline_string_body_after_single_newline_strip() {
        let text = "<tool_call>\n\
                    <function=write_note>\n\
                    <parameter=body>\n\
                    line 1\n\
                    line 2\n\
                    </parameter>\n\
                    </function>\n\
                    </tool_call>";
        let calls = parse_tool_calls(text, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["body"], "line 1\nline 2");
    }

    #[test]
    fn bare_tool_call_marker_inside_thinking_does_not_swallow_closing_think() {
        let with_call = "<think>I will emit a <tool_call> block now.</think>\
            <tool_call>\n<function=ping>\n<parameter=host>\nlocalhost\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_output(with_call, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(
            parsed.reasoning_content.as_deref(),
            Some("I will emit a <tool_call> block now.")
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "ping");
        assert_eq!(parsed.tool_calls[0].arguments["host"], "localhost");

        let prose_only =
            "<think>The syntax uses <tool_call> tags.</think>Here is a plain text reply.";
        let parsed_prose = parse_output(prose_only, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(
            parsed_prose.reasoning_content.as_deref(),
            Some("The syntax uses <tool_call> tags.")
        );
        assert_eq!(parsed_prose.content, "Here is a plain text reply.");
        assert!(parsed_prose.tool_calls.is_empty());
    }

    #[test]
    fn lfm3_xml_formatting_and_deferred_tools() {
        let search_tool = ToolDef {
            name: "tool_search".into(),
            description: Some("Search for available tools by keyword".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }),
            defer_loading: false,
        };
        let purchase_tool = ToolDef {
            name: "markItemPurchased".into(),
            description: Some("Mark an item as purchased".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "qty": {"type": "integer"}
                }
            }),
            defer_loading: true,
        };
        let details_tool = ToolDef {
            name: "getItemDetails".into(),
            description: Some("Fetch details for an item".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verbose": {"type": "boolean"}
                }
            }),
            defer_loading: true,
        };
        let all_tools = vec![
            search_tool.clone(),
            purchase_tool.clone(),
            details_tool.clone(),
        ];

        // Deferred tools are excluded from the initial <tools> block, and
        // `defer_loading` never appears in the rendered block.
        let prompt_tools = format_tools_for_prompt(&all_tools, ToolFormat::Lfm3Xml);
        assert!(prompt_tools.starts_with("<tools>\n"));
        assert!(prompt_tools.ends_with("\n</tools>"));
        assert!(prompt_tools.contains("\"name\": \"tool_search\""));
        assert!(!prompt_tools.contains("markItemPurchased"));
        assert!(!prompt_tools.contains("getItemDetails"));
        assert!(!prompt_tools.contains("defer_loading"));

        // When all tools are deferred, the <tools> block is omitted entirely.
        let only_deferred = vec![purchase_tool.clone(), details_tool.clone()];
        assert_eq!(
            format_tools_for_prompt(&only_deferred, ToolFormat::Lfm3Xml),
            ""
        );

        // Formatting a tool call in LFM3 XML round-trips through parse_output.
        let call = ToolCall {
            name: "markItemPurchased".into(),
            arguments: serde_json::json!({"id": "abc", "qty": 3}),
        };
        let formatted_call = format_tool_call(&call, ToolFormat::Lfm3Xml);
        assert!(formatted_call.starts_with("<tool_call>\n<function=markItemPurchased>\n"));
        assert!(formatted_call.ends_with("</function>\n</tool_call>"));
        let reparsed = parse_output(&formatted_call, ToolFormat::Lfm3Xml).unwrap();
        assert_eq!(reparsed.tool_calls, vec![call]);

        // Expanding a single tool_reference inside <tool_response>.
        let single_ref = format_tool_response_parts(
            &[ToolContentPart::tool_reference("markItemPurchased")],
            &all_tools,
        )
        .unwrap();
        assert!(
            single_ref.starts_with(
                "<tool_response>\n<additional_tools>\n{\"description\": \"Mark an item as purchased\""
            ),
            "unexpected single_ref: {single_ref}"
        );
        assert!(single_ref.ends_with("\n</additional_tools>\n</tool_response>"));
        assert!(!single_ref.contains("defer_loading"));
        assert!(!single_ref.contains("getItemDetails"));

        // Consecutive tool_reference entries share one <additional_tools> block.
        let multi_ref = format_tool_response(
            r#"[{"type": "tool_reference", "name": "markItemPurchased"}, {"type": "tool_reference", "name": "getItemDetails"}]"#,
            &all_tools,
        )
        .unwrap();
        assert!(multi_ref.contains("\"name\": \"markItemPurchased\""));
        assert!(multi_ref.contains("\"name\": \"getItemDetails\""));
        assert_eq!(multi_ref.matches("<additional_tools>").count(), 1);
        assert_eq!(multi_ref.matches("</additional_tools>").count(), 1);

        // Unknown tool_reference name errors.
        assert!(
            format_tool_response(
                r#"[{"type": "tool_reference", "name": "nonexistentTool"}]"#,
                &all_tools,
            )
            .is_err()
        );

        // Missing name on tool_reference errors.
        assert!(format_tool_response(r#"[{"type": "tool_reference"}]"#, &all_tools).is_err());
    }

    fn compile(src: &str) -> std::sync::Arc<crate::grammar::Grammar> {
        std::sync::Arc::new(
            crate::grammar::Grammar::parse(src)
                .unwrap_or_else(|e| panic!("grammar failed to compile: {e}\n---\n{src}")),
        )
    }

    /// True iff the grammar accepts `bytes` and reaches a complete (terminable)
    /// state with no leftover requirement.
    fn accepts_complete(g: &std::sync::Arc<crate::grammar::Grammar>, bytes: &[u8]) -> bool {
        let mut st = crate::grammar::GrammarState::new(g.clone());
        if !st.accepts(bytes) {
            return false;
        }
        st.accept(bytes);
        st.is_complete()
    }

    fn weather() -> ToolDef {
        ToolDef {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }),
            defer_loading: false,
        }
    }

    #[test]
    fn lfm2_grammar_accepts_valid_calls() {
        let g = compile(&tool_grammar(&[weather()], ToolFormat::Lfm2Pythonic).unwrap());
        assert!(accepts_complete(&g, b"[get_weather(city=\"Paris\")]"));
        assert!(accepts_complete(
            &g,
            b"[get_weather(city=\"Paris\", days=3)]"
        ));
        assert!(accepts_complete(
            &g,
            b"[get_weather(units=\"celsius\", city=\"Rome\")]"
        ));
        assert!(accepts_complete(&g, b"[get_weather()]"));
    }

    #[test]
    fn lfm2_grammar_rejects_invalid_calls() {
        let g = compile(&tool_grammar(&[weather()], ToolFormat::Lfm2Pythonic).unwrap());
        // Unknown function name.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"[get_stocks("));
        // Unknown argument name.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"[get_weather(country="));
        // Wrong type for `days` (integer schema, string given).
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"[get_weather(days=\""));
        // Enum violation for `units`.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"[get_weather(units=\"kelvin\")"));
    }

    #[test]
    fn lfm3_xml_grammar_accepts_valid_and_rejects_invalid_calls() {
        let g = compile(&generate_gbnf(&[weather()], ToolFormat::Lfm3Xml).unwrap());
        assert!(accepts_complete(
            &g,
            b"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
        ));
        assert!(accepts_complete(
            &g,
            b"<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=units>\ncelsius\n</parameter>\n</function>\n</tool_call>"
        ));
        assert!(accepts_complete(
            &g,
            b"<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        ));
        // Interior after lazy `<tool_call>` trigger.
        assert!(accepts_complete(
            &g,
            b"<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>"
        ));

        // Unknown function name.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"<tool_call>\n<function=get_stocks>\n"));
        // Unknown parameter name.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"<tool_call>\n<function=get_weather>\n<parameter=country>\n"));
        // Wrong type for integer parameter `days`.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(b"<tool_call>\n<function=get_weather>\n<parameter=days>\nnot_an_int"));
        // Invalid enum variant for `units`.
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(
            b"<tool_call>\n<function=get_weather>\n<parameter=units>\nkelvin\n</parameter>"
        ));
    }

    /// What the parser accepts as a string parameter the grammar must be able to emit: text
    /// with `<`, code, blank lines (paragraphs) and CRs. Only the closing tag ends a value.
    #[test]
    fn lfm3_xml_grammar_string_values_can_hold_any_text_but_the_closing_tag() {
        let g = compile(&generate_gbnf(&[weather()], ToolFormat::Lfm3Xml).unwrap());
        let call = |value: &str| {
            format!(
                "<tool_call>\n<function=get_weather>\n<parameter=city>\n{value}\n</parameter>\n</function>\n</tool_call>"
            )
        };
        for value in [
            "a < b",
            "x<y>z",
            "<<<",
            "first paragraph\n\nsecond paragraph",
            "line one\r\nline two",
            "<div>html</div>",
            "</param",
            "</parameter",
            "tail <",
        ] {
            assert!(
                accepts_complete(&g, call(value).as_bytes()),
                "the grammar rejects the value {value:?}"
            );
            // and the parser reads back the same text
            let calls = parse_tool_calls(&call(value), ToolFormat::Lfm3Xml).unwrap();
            assert_eq!(
                calls[0].arguments["city"],
                serde_json::json!(value),
                "{value:?}"
            );
        }
        // the closing tag inside a value would end it early, so it cannot be emitted there
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(
            !st.accepts(b"<tool_call>\n<function=get_weather>\n<parameter=city>\nab</parameter>cd")
        );
    }

    /// A string enum is emitted the way the parser reads it back: raw text for a `string`
    /// parameter (the parser keeps the text as written, so a quoted form would become a value
    /// with quotes in it, outside the enum), and quoted only for an untyped enum variant that
    /// would otherwise parse as JSON (`true`, `3`).
    #[test]
    fn lfm3_xml_string_enums_are_emitted_as_the_parser_reads_them() {
        let tool = |schema: serde_json::Value| ToolDef {
            name: "set".into(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": {"v": schema}}),
            ..Default::default()
        };
        let call = |value: &str| {
            format!(
                "<tool_call>\n<function=set>\n<parameter=v>\n{value}\n</parameter>\n</function>\n</tool_call>"
            )
        };
        // typed: raw only
        let typed = tool(serde_json::json!({"type": "string", "enum": ["celsius", "kelvin"]}));
        let g = compile(&generate_gbnf(std::slice::from_ref(&typed), ToolFormat::Lfm3Xml).unwrap());
        assert!(accepts_complete(&g, call("celsius").as_bytes()));
        assert!(
            !accepts_complete(&g, call("\"celsius\"").as_bytes()),
            "quoted form emitted"
        );
        let parsed =
            parse_tool_calls_with_tools(&call("celsius"), ToolFormat::Lfm3Xml, &[typed]).unwrap();
        assert_eq!(parsed[0].arguments["v"], "celsius");
        // untyped: a variant that reads as JSON must be quoted to stay a string
        let untyped = tool(serde_json::json!({"enum": ["true", "x"]}));
        let g =
            compile(&generate_gbnf(std::slice::from_ref(&untyped), ToolFormat::Lfm3Xml).unwrap());
        assert!(accepts_complete(&g, call("x").as_bytes()));
        assert!(accepts_complete(&g, call("\"true\"").as_bytes()));
        assert!(
            !accepts_complete(&g, call("true").as_bytes()),
            "raw `true` reads back as a bool"
        );
        let parsed =
            parse_tool_calls_with_tools(&call("\"true\""), ToolFormat::Lfm3Xml, &[untyped])
                .unwrap();
        assert_eq!(parsed[0].arguments["v"], "true");
    }

    #[test]
    fn lfm3_xml_grammar_rejects_names_the_parser_cannot_read_back() {
        let named = |tool: &str, arg: &str| ToolDef {
            name: tool.into(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": {arg: {"type": "string"}}}),
            ..Default::default()
        };
        for tool in ["a>b", "a<b", "line\nbreak", " lead", "trail ", ""] {
            assert!(
                tool_grammar(&[named(tool, "x")], ToolFormat::Lfm3Xml).is_err(),
                "tool {tool:?}"
            );
        }
        for arg in ["a>b", "x\ny", " lead", ""] {
            assert!(
                tool_grammar(&[named("ok", arg)], ToolFormat::Lfm3Xml).is_err(),
                "arg {arg:?}"
            );
        }
        // names the XML tags can hold are fine, including ones Python would refuse
        assert!(tool_grammar(&[named("get-weather.v2", "max-days")], ToolFormat::Lfm3Xml).is_ok());
    }

    #[test]
    fn a_tool_definition_with_a_wrong_typed_field_is_rejected() {
        let parse = |json: &str| serde_json::from_str::<ToolDef>(json);
        for bad in [
            r#"{"name":"t","defer_loading":"true"}"#,
            r#"{"name":"t","defer_loading":1}"#,
            r#"{"name":"t","description":5}"#,
            r#"{"function":{"name":"t","defer_loading":"yes"}}"#,
        ] {
            assert!(parse(bad).is_err(), "{bad} was accepted");
        }
        // every documented shape still reads, and null counts as absent
        // a definition always reads back as it serialized, whatever its schema is
        let odd = ToolDef::new("t", None, serde_json::json!("not an object"));
        let back: ToolDef = serde_json::from_str(&serde_json::to_string(&odd).unwrap()).unwrap();
        assert_eq!(back, odd);
        assert!(parse(r#"{"name":"t"}"#).is_ok());
        assert!(parse(r#"{"name":"t","description":null,"parameters":null}"#).is_ok());
        let nested = parse(
            r#"{"function":{"name":"t","parameters":{"type":"object"}},"defer_loading":true}"#,
        )
        .unwrap();
        assert!(nested.defer_loading);
        let anthropic = parse(r#"{"name":"t","input_schema":{"type":"object"}}"#).unwrap();
        assert_eq!(anthropic.parameters, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn a_tool_result_with_unmodelled_parts_does_not_fail_the_render() {
        // a reference next to an image block (a part kind this crate does not model)
        let content =
            r#"[{"type":"tool_reference","name":"lookup"},{"type":"image","source":"x"}]"#;
        let msg = ToolChatMessage {
            role: "tool".into(),
            content: content.into(),
            content_parts: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            additional_tools: None,
        };
        let v = serde_json::to_value(&msg).expect("rendering must not fail on tool output");
        let parts = v["content"].as_array().expect("parts");
        assert_eq!(parts[0]["type"], "tool_reference");
        assert_eq!(parts[0]["name"], "lookup");
        assert_eq!(parts[1]["type"], "text", "the image block is kept as text");
        assert!(parts[1]["text"].as_str().unwrap().contains("\"image\""));
        // a reference with no name is text too, and malformed JSON stays as written
        let nameless = ToolChatMessage {
            content: r#"[{"type":"tool_reference"}]"#.into(),
            ..msg.clone()
        };
        assert!(serde_json::to_value(&nameless).is_ok());
        // the explicit API still reports the malformed part
        assert!(try_parse_tool_content_parts(content).is_err());
    }

    #[test]
    fn hermes_grammar_accepts_json_call() {
        let g = compile(&tool_grammar(&[weather()], ToolFormat::Hermes).unwrap());
        assert!(accepts_complete(
            &g,
            br#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
        ));
        assert!(accepts_complete(
            &g,
            br#"{"name": "get_weather", "arguments": {"city": "Rome", "days": 5}}"#
        ));
    }

    #[test]
    fn hermes_grammar_rejects_bad_name() {
        let g = compile(&tool_grammar(&[weather()], ToolFormat::Hermes).unwrap());
        let st = crate::grammar::GrammarState::new(g.clone());
        assert!(!st.accepts(br#"{"name": "get_stocks"#));
    }

    #[test]
    fn multi_tool_grammar_alternates() {
        let tools = vec![
            weather(),
            ToolDef {
                name: "add".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}
                }),
                defer_loading: false,
            },
        ];
        let g = compile(&tool_grammar(&tools, ToolFormat::Lfm2Pythonic).unwrap());
        assert!(accepts_complete(&g, b"[get_weather(city=\"Paris\")]"));
        assert!(accepts_complete(&g, b"[add(a=1, b=2)]"));
        // Multiple calls in one section.
        assert!(accepts_complete(
            &g,
            b"[add(a=1, b=2), get_weather(city=\"X\")]"
        ));
    }

    #[test]
    fn empty_tools_is_error() {
        assert!(tool_grammar(&[], ToolFormat::Lfm2Pythonic).is_err());
        assert!(tool_grammar(&[], ToolFormat::Lfm3Xml).is_err());
    }

    #[test]
    fn tool_def_serializes_to_function_shape() {
        let tool = ToolDef {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
            defer_loading: false,
        };
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["description"], "Get weather");
        assert_eq!(v["parameters"]["properties"]["city"]["type"], "string");
        assert!(v.get("defer_loading").is_none());

        // Round-trip from OpenAI wrapper with defer_loading: true.
        let oai_json = r#"{"type": "function", "function": {"name": "markItemPurchased", "description": "Mark item", "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}, "defer_loading": true}}"#;
        let parsed: ToolDef = serde_json::from_str(oai_json).unwrap();
        assert_eq!(parsed.name, "markItemPurchased");
        assert!(parsed.defer_loading);
        let v_deferred = serde_json::to_value(&parsed).unwrap();
        assert_eq!(v_deferred["defer_loading"], true);
    }
}

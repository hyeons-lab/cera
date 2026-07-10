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

use anyhow::{Result, bail};
use serde_json::{Map, Value};

/// A tool the model may call. Serializes to the JSON object shape chat
/// templates expect under the OpenAI "function" convention:
/// `{"name": …, "description": …, "parameters": {JSON Schema}}`.
///
/// `parameters` is a JSON Schema object describing the arguments (`type:
/// object`, `properties`, `required`). It is passed through verbatim to the
/// template and used to derive a constraint grammar.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments object. Defaults to an empty object
    /// schema when absent.
    #[serde(default = "empty_params")]
    pub parameters: Value,
}

fn empty_params() -> Value {
    serde_json::json!({ "type": "object", "properties": {} })
}

/// A tool call parsed from model output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// Argument object (`{arg_name: value}`). Always a JSON object; scalar
    /// argument values keep their parsed JSON type (string/number/bool/null/
    /// array/object).
    pub arguments: Value,
}

/// A chat message that can carry tool calls / tool results, for replaying a
/// tool-calling conversation into the chat template.
///
/// This is a *separate* struct from [`crate::tokenizer::ChatMessage`] on
/// purpose — [`crate::tokenizer::apply_chat_template`] is generic over
/// `Serialize`, and the multimodal path uses the same trick
/// ([`crate::tokenizer::ChatMessageMultimodal`]). Keeping tool fields here
/// means the plain text path stays a two-field struct with no churn at its
/// dozens of call sites.
///
/// Field semantics follow the OpenAI convention every tool-trained template
/// understands:
/// - `role = "assistant"` with `tool_calls` set → a turn where the model
///   called tools (Hermes/Qwen templates read `message.tool_calls`; LFM2 keeps
///   the call text in `content` instead, so set whichever the model needs).
/// - `role = "tool"` with `content` = the JSON result string → a tool result
///   fed back to the model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Links a `tool` result back to the call it answers, when the template
    /// (or a downstream consumer) uses it. Omitted when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ToolChatMessage {
    /// A plain role+content message (no tool calls, no id).
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ToolChatMessage {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// An `assistant` turn that issued `calls`. `content` is any text the model
    /// produced alongside the calls (often empty).
    pub fn assistant_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        ToolChatMessage {
            role: "assistant".into(),
            content: content.into(),
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    /// A `tool` result message answering `tool_call_id` (if known).
    pub fn tool_result(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        ToolChatMessage {
            role: "tool".into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id,
        }
    }
}

/// Wire format a model family uses for tool calls. Determines both how output
/// is parsed and what constraint grammar is generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// LFM2 / LFM2.5: Pythonic call list wrapped in tool-call markers —
    /// `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>`.
    Lfm2Pythonic,
    /// Hermes-style (Qwen2.5/Qwen3, many fine-tunes): one or more
    /// `<tool_call>{"name": …, "arguments": {…}}</tool_call>` JSON blocks.
    Hermes,
}

impl ToolFormat {
    /// Best-effort detection from the model architecture string
    /// ([`crate::model::ModelConfig::architecture`]). Returns `None` when the
    /// architecture has no known tool-call convention — callers may still set a
    /// format explicitly.
    pub fn detect(architecture: &str) -> Option<ToolFormat> {
        match architecture {
            "lfm2" | "lfm2moe" => Some(ToolFormat::Lfm2Pythonic),
            // Qwen2/Qwen3/most llama.cpp Hermes fine-tunes use the
            // `<tool_call>` JSON convention. Plain "llama" is ambiguous
            // (Llama-3.1 uses bare JSON), so we don't claim it here.
            "qwen2" | "qwen3" | "qwen3moe" => Some(ToolFormat::Hermes),
            _ => None,
        }
    }

    /// The literal string that opens a tool-call section for this format. Used
    /// as the lazy grammar trigger and as a parse anchor.
    pub fn call_start_marker(self) -> &'static str {
        match self {
            ToolFormat::Lfm2Pythonic => "<|tool_call_start|>",
            ToolFormat::Hermes => "<tool_call>",
        }
    }

    /// The literal string that closes a tool-call section for this format.
    pub fn call_end_marker(self) -> &'static str {
        match self {
            ToolFormat::Lfm2Pythonic => "<|tool_call_end|>",
            ToolFormat::Hermes => "</tool_call>",
        }
    }
}

/// Parse tool calls out of generated model text.
///
/// Tolerant by design: a missing closing marker (common when generation stops
/// on EOS right after the call) still parses. Returns an empty vec when the
/// text contains no tool call — that is the normal "the model answered in prose"
/// case, not an error. Returns `Err` only when a call *section* is present but
/// malformed enough that no call can be recovered.
pub fn parse_tool_calls(text: &str, format: ToolFormat) -> Result<Vec<ToolCall>> {
    match format {
        ToolFormat::Lfm2Pythonic => parse_lfm2_pythonic(text),
        ToolFormat::Hermes => parse_hermes(text),
    }
}

// ── LFM2 Pythonic parser ────────────────────────────────────────────────────

/// Parse `<|tool_call_start|>[name(a="x", b=3), other()]<|tool_call_end|>`.
///
/// The interior is a Python list of function calls. We accept the bare list
/// with or without the surrounding markers so the same routine handles both
/// streamed fragments and fully wrapped output.
fn parse_lfm2_pythonic(text: &str) -> Result<Vec<ToolCall>> {
    let Some(inner) = extract_between(text, "<|tool_call_start|>", "<|tool_call_end|>") else {
        return Ok(Vec::new());
    };
    let inner = inner.trim();
    // Strip the outer list brackets if present: `[call, call]` → `call, call`.
    let body = inner
        .strip_prefix('[')
        .map(|s| s.strip_suffix(']').unwrap_or(s))
        .unwrap_or(inner)
        .trim();
    if body.is_empty() {
        return Ok(Vec::new());
    }
    let mut p = PyParser::new(body);
    let mut calls = Vec::new();
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
    Ok(calls)
}

/// Minimal recursive-descent parser for the Pythonic call subset LFM2 emits:
/// `name(key=value, …)` with values being str / int / float / bool / None /
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

    /// `name ( args )` → a [`ToolCall`].
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
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.i += 1;
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.s[start..self.i]).into_owned()
    }

    /// A Python literal → JSON value.
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
        let quote = self.bump().unwrap(); // ' or "
        let mut out = String::new();
        while let Some(c) = self.bump() {
            match c {
                '\\' => match self.bump() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('\\') => out.push('\\'),
                    Some('\'') => out.push('\''),
                    Some('"') => out.push('"'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => bail!("unterminated escape in string literal"),
                },
                c if c == quote => return Ok(Value::String(out)),
                c => out.push(c),
            }
        }
        bail!("unterminated string literal")
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
            Ok(serde_json::json!(f))
        } else {
            let n: i64 = tok
                .parse()
                .map_err(|_| anyhow::anyhow!("bad integer '{tok}'"))?;
            Ok(serde_json::json!(n))
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

// ── Hermes / JSON parser ────────────────────────────────────────────────────

/// Parse one or more `<tool_call>{json}</tool_call>` blocks. Each block's JSON
/// is `{"name": …, "arguments": {…}}` (some models use `"parameters"`).
fn parse_hermes(text: &str) -> Result<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        // Tolerate a missing closing tag on the final block.
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
    Ok(calls)
}

// ── shared helpers ──────────────────────────────────────────────────────────

/// Return the substring between the first `open` and the first following
/// `close`. If `open` is present but `close` is not, return everything after
/// `open` (tolerant tail). Returns `None` only when `open` is absent.
fn extract_between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.find(open)? + open.len();
    let after = &text[start..];
    match after.find(close) {
        Some(end) => Some(&after[..end]),
        None => Some(after),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_arch() {
        assert_eq!(ToolFormat::detect("lfm2"), Some(ToolFormat::Lfm2Pythonic));
        assert_eq!(ToolFormat::detect("qwen3"), Some(ToolFormat::Hermes));
        assert_eq!(ToolFormat::detect("gpt2"), None);
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
    fn no_call_is_empty_not_error() {
        let calls = parse_tool_calls("The weather is sunny.", ToolFormat::Lfm2Pythonic).unwrap();
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
    fn tool_def_serializes_to_function_shape() {
        let tool = ToolDef {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        };
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["description"], "Get weather");
        assert_eq!(v["parameters"]["properties"]["city"]["type"], "string");
    }
}

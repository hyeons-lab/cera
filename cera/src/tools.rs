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
/// each argument's value **type** (string/integer/number/boolean/array/object,
/// plus `enum` literal sets). Not yet enforced: that *required* arguments are
/// present, and no-duplicate/ordering constraints — arguments may appear in any
/// order and any subset. This is a deliberate v1 scope: it guarantees a
/// well-formed, correctly-typed call without over-constraining the model.
pub fn tool_grammar(tools: &[ToolDef], format: ToolFormat) -> Result<String> {
    if tools.is_empty() {
        bail!("tool_grammar: no tools provided");
    }
    match format {
        ToolFormat::Lfm2Pythonic => Ok(lfm2_grammar(tools)),
        ToolFormat::Hermes => Ok(hermes_grammar(tools)),
    }
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

/// Hermes grammar: `<tool_call>` JSON blocks — but the markers are special
/// tokens handled by the trigger, so this constrains the JSON object body:
/// `{"name": "<one of the tools>", "arguments": { … }}`.
fn hermes_grammar(tools: &[ToolDef]) -> String {
    let mut out = String::new();
    out.push_str("root ::= tc-ws tc-call tc-ws\n");
    let call_alts: Vec<String> = (0..tools.len()).map(|i| format!("tc-call-{i}")).collect();
    out.push_str(&format!("tc-call ::= {}\n", call_alts.join(" | ")));

    for (i, tool) in tools.iter().enumerate() {
        let props = properties(tool);
        // {"name": "tool", "arguments": <args>} — the name *value* is a JSON
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

// ── LFM2 Pythonic parser ────────────────────────────────────────────────────

/// Parse `<|tool_call_start|>[name(a="x", b=3), other()]<|tool_call_end|>`.
///
/// The interior is a Python list of function calls. We accept the bare list
/// with or without the surrounding markers so the same routine handles both
/// streamed fragments and fully wrapped output.
fn parse_lfm2_pythonic(text: &str) -> Result<Vec<ToolCall>> {
    // Walk every `<|tool_call_start|>…<|tool_call_end|>` section, not just the
    // first — a model may emit several sections in one turn, and the Hermes
    // parser handles multiples, so this stays consistent. A missing final end
    // marker is tolerated (the tail is treated as the last section).
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<|tool_call_start|>") {
        let after = &rest[start + "<|tool_call_start|>".len()..];
        let (inner, next) = match after.find("<|tool_call_end|>") {
            Some(end) => (&after[..end], &after[end + "<|tool_call_end|>".len()..]),
            None => (after, ""),
        };
        parse_pythonic_section(inner.trim(), &mut calls)?;
        rest = next;
    }
    Ok(calls)
}

/// Parse one `[call, call, …]` section body, appending to `calls`.
fn parse_pythonic_section(inner: &str, calls: &mut Vec<ToolCall>) -> Result<()> {
    // Strip the outer list brackets if present: `[call, call]` → `call, call`.
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
                        // `\uXXXX` → the codepoint's UTF-8 bytes. These escapes
                        // are permitted by the generated grammar's `tc-esc`, so
                        // the parser must decode them to agree with constrained
                        // output. Lone surrogates fall back to U+FFFD.
                        b'u' => {
                            let cp = self.parse_hex4()?;
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
    }

    #[test]
    fn lfm2_multiple_sections() {
        // Two separate marker sections → both calls recovered.
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
        // enum containing an array variant → must not forbid the array; falls
        // back to the declared type instead of dropping the non-scalar member.
        let tools = vec![ToolDef {
            name: "f".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "array", "enum": [["a"], ["b"]]}}
            }),
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

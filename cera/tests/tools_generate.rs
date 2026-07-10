//! End-to-end tool-calling check: render a tools prompt, generate on a real
//! tool-trained model, and parse the emitted call.
//!
//! Skips unless `$CERA_TOOLS_TEST_MODEL` (or a repo-root `models/` candidate)
//! points at a real GGUF — CI has none, so it stays green. Greedy decode
//! (`temperature = 0`) keeps it deterministic.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use cera::engine::{CeraEngine, EngineConfig};
use cera::session::{FinishReason, GenerateOpts, ModalitySink, SessionConfig};
use cera::tokenizer::{ChatMessage, apply_chat_template_with_tools};
use cera::tools::{ToolDef, ToolFormat, parse_tool_calls};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CERA_TOOLS_TEST_MODEL") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "../models/LFM2.5-350M-Q4_0.gguf",
        "../models/LFM2.5-350M-Q4_K_M.gguf",
    ] {
        let p = manifest.join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Collects decoded text across the generation.
struct CollectSink {
    tokens: Vec<u32>,
    done: Option<FinishReason>,
}
impl ModalitySink for CollectSink {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
    }
    fn on_done(&mut self, reason: FinishReason) {
        self.done = Some(reason);
    }
}

#[test]
fn model_emits_parseable_tool_call() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no tool-trained GGUF (set CERA_TOOLS_TEST_MODEL)");
        return;
    };

    let engine = CeraEngine::from_path(&path, EngineConfig::default()).expect("load engine");
    let arch = engine.metadata().architecture.clone();
    let format = ToolFormat::detect(&arch).unwrap_or(ToolFormat::Lfm2Pythonic);

    let tools = vec![ToolDef {
        name: "get_weather".into(),
        description: Some("Get the current weather for a given city".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
    }];
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "What is the weather in Paris right now?".into(),
    }];

    let prompt = apply_chat_template_with_tools(engine.tokenizer(), &messages, &tools, true)
        .expect("render tools prompt");

    let mut session = engine
        .new_session(SessionConfig::default())
        .expect("session");
    session.append_text(&prompt).expect("append prompt");

    let sink = Rc::new(RefCell::new(CollectSink {
        tokens: Vec::new(),
        done: None,
    }));
    let opts = GenerateOpts {
        max_tokens: 128,
        temperature: 0.0, // greedy → deterministic
        ..Default::default()
    };
    {
        let mut s = sink.borrow_mut();
        session.generate(&opts, &mut *s).expect("generate");
    }

    let out = engine.tokenizer().decode(&sink.borrow().tokens);
    eprintln!("--- arch={arch} format={format:?} ---\n{out}\n--- end ---");

    let calls = parse_tool_calls(&out, format).expect("parse tool calls");
    assert!(
        !calls.is_empty(),
        "model did not emit a parseable tool call:\n{out}"
    );
    assert_eq!(
        calls[0].name, "get_weather",
        "unexpected tool name in {calls:?}"
    );
    assert!(
        calls[0].arguments.get("city").is_some(),
        "call missing 'city' argument: {:?}",
        calls[0]
    );
}

/// The lazy-trigger + constrained path: free text until `<|tool_call_start|>`,
/// then the generated grammar forces a well-formed, correctly-typed call.
#[test]
fn constrained_tool_call_with_lazy_trigger() {
    use std::sync::Arc;

    use cera::grammar::Grammar;
    use cera::tools::tool_grammar;

    let Some(path) = model_path() else {
        eprintln!("skipping: no tool-trained GGUF (set CERA_TOOLS_TEST_MODEL)");
        return;
    };

    let engine = CeraEngine::from_path(&path, EngineConfig::default()).expect("load engine");
    let arch = engine.metadata().architecture.clone();
    let format = ToolFormat::detect(&arch).unwrap_or(ToolFormat::Lfm2Pythonic);

    let tools = vec![ToolDef {
        name: "get_weather".into(),
        description: Some("Get the current weather for a given city".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "units": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["city"]
        }),
    }];
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "What is the weather in Tokyo?".into(),
    }];

    let prompt = apply_chat_template_with_tools(engine.tokenizer(), &messages, &tools, true)
        .expect("render tools prompt");

    // Build the constraint grammar + resolve the start-marker trigger token.
    let gbnf = tool_grammar(&tools, format).expect("tool grammar");
    let grammar = Arc::new(Grammar::parse(&gbnf).expect("compile grammar"));
    // `special_token_id` is fallible by design: a model whose tool-call markers
    // aren't registered as special tokens can't drive the lazy trigger. Skip
    // (don't panic) so this gated test tolerates any `$CERA_TOOLS_TEST_MODEL`.
    let Some(trigger) = engine
        .tokenizer()
        .special_token_id(format.call_start_marker())
    else {
        eprintln!(
            "skipping: model has no `{}` special token for the {format:?} format",
            format.call_start_marker()
        );
        return;
    };

    let mut session = engine
        .new_session(SessionConfig::default())
        .expect("session");
    session.append_text(&prompt).expect("append prompt");

    let sink = Rc::new(RefCell::new(CollectSink {
        tokens: Vec::new(),
        done: None,
    }));
    let opts = GenerateOpts {
        max_tokens: 128,
        temperature: 0.0,
        grammar: Some(grammar),
        grammar_trigger_tokens: vec![trigger],
        ..Default::default()
    };
    {
        let mut s = sink.borrow_mut();
        session.generate(&opts, &mut *s).expect("generate");
    }

    let out = engine.tokenizer().decode(&sink.borrow().tokens);
    eprintln!("--- constrained arch={arch} ---\n{out}\n--- end ---");

    let calls = parse_tool_calls(&out, format).expect("parse tool calls");
    assert!(!calls.is_empty(), "no tool call under constraint:\n{out}");
    assert_eq!(calls[0].name, "get_weather");
    // The grammar guarantees only declared arg names appear.
    if let Some(obj) = calls[0].arguments.as_object() {
        for k in obj.keys() {
            assert!(
                k == "city" || k == "units",
                "grammar allowed an undeclared argument '{k}': {out}"
            );
        }
    }
}

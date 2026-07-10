//! Verifies the chat-template `tools` plumbing against a real tool-trained
//! GGUF. Loading only touches metadata (no weights), so this is cheap.
//!
//! The model is resolved from `$CERA_TOOLS_TEST_MODEL`, else a few candidate
//! paths (the repo-root `models/` dir). Absent → the test skips, so CI (which
//! has no models) stays green.

use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::tokenizer::{BpeTokenizer, ChatMessage, apply_chat_template_with_tools};
use cera::tools::ToolDef;

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

fn weather_tool() -> ToolDef {
    ToolDef {
        name: "get_weather".into(),
        description: Some("Get the current weather for a city".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
    }
}

#[test]
fn tools_render_into_lfm2_template() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no tool-trained GGUF found (set CERA_TOOLS_TEST_MODEL)");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tok = BpeTokenizer::from_gguf(&gguf).expect("build tokenizer");

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "What's the weather in Paris?".into(),
    }];

    // With a tool, the template's `{% if tools %}` branch must fire.
    let tools = vec![weather_tool()];
    let with = apply_chat_template_with_tools(&tok, &messages, &tools, true).expect("render");
    assert!(
        with.contains("List of tools") || with.contains("tool_list_start"),
        "tool block missing from rendered prompt:\n{with}"
    );
    assert!(with.contains("get_weather"), "tool name missing:\n{with}");

    // Without tools the block must be absent — proves we didn't change the
    // no-tools rendering.
    let without =
        apply_chat_template_with_tools(&tok, &messages, &[], true).expect("render no tools");
    assert!(
        !without.contains("get_weather"),
        "tool leaked into no-tools render:\n{without}"
    );
    assert!(
        !without.contains("List of tools"),
        "tool block present with empty tools:\n{without}"
    );
}

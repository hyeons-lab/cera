//! Integration tests for GBNF grammar-constrained decoding, including validation of the
//! JSON grammar that ships with `cera-cli` (`--json`).

use std::sync::Arc;

use cera::grammar::{Grammar, GrammarMask, GrammarState};

/// The grammar bundled with `cera-cli --json`. Tested here so the shipped artifact can't
/// silently rot.
const JSON_GBNF: &str = include_str!("../../cera-cli/grammars/json.gbnf");

fn json() -> Arc<Grammar> {
    Arc::new(Grammar::parse(JSON_GBNF).expect("bundled json.gbnf must parse"))
}

/// Feed `input` byte-by-byte; returns the final state iff every byte was accepted.
fn run(g: &Arc<Grammar>, input: &[u8]) -> Option<GrammarState> {
    let mut st = GrammarState::new(g.clone());
    for &b in input {
        if !st.accepts(&[b]) {
            return None;
        }
        st.accept(&[b]);
    }
    Some(st)
}

fn accepts_complete(g: &Arc<Grammar>, s: &str) -> bool {
    run(g, s.as_bytes())
        .map(|st| st.is_complete())
        .unwrap_or(false)
}

#[test]
fn json_grammar_accepts_valid() {
    let g = json();
    for s in [
        "{}",
        "[]",
        r#"{"a":1}"#,
        "[1,2,3]",
        r#"{ "name" : "ada" , "age" : 36 }"#,
        r#"{"nested":{"a":[1,2,{"b":true}]},"x":null,"y":false}"#,
        "-12.5e+10",
        r#""a string with \"escapes\" and é and / and \\""#,
        "  \n  {\n  \"k\" : [ 1 , 2 ]\n}",
        "1234567890",
        "true",
    ] {
        assert!(accepts_complete(&g, s), "should accept valid JSON: {s:?}");
    }
}

#[test]
fn json_grammar_rejects_invalid() {
    let g = json();
    for s in [
        r#"{"k":}"#,  // missing value
        r#"{"k" 1}"#, // missing colon
        "[1,]",       // trailing comma
        "{,}",        // bogus
        r#"{key:1}"#, // unquoted key
        "[1,2,3",     // unterminated
        "01",         // leading zero
        r#""unterminated"#,
        "nul", // not a complete literal
    ] {
        assert!(
            !accepts_complete(&g, s),
            "should reject invalid JSON: {s:?}"
        );
    }
}

#[test]
fn json_incomplete_prefix_is_not_complete_but_extensible() {
    let g = json();
    // A partial object `{"a":` — parsed up to (and including) the colon, so the grammar
    // now expects a value. Accepted (the bytes parse) but not a complete value.
    let st = run(&g, b"{\"a\":").expect("prefix should parse");
    assert!(!st.is_complete());
    // ...and the next byte can begin any JSON value: a number, a string, or an array.
    assert!(st.accepts(b"1")); // number
    assert!(st.accepts(b"\"")); // string
    assert!(st.accepts(b"[")); // array
}

#[test]
fn mask_constrains_to_json_openers_then_advances() {
    let g = json();
    // Stub vocab: tokens that a JSON value can/can't start with.
    // 0="{", 1="[", 2="x" (invalid start), 3="\"", 4=EOS, 5=true.
    let token_bytes = vec![
        b"{".to_vec(),
        b"[".to_vec(),
        b"x".to_vec(),
        b"\"".to_vec(),
        Vec::new(),
        b"true".to_vec(),
    ];
    let special = vec![false, false, false, false, true, false];
    let mask = GrammarMask::new(token_bytes, Some(4), special);

    let mut st = GrammarState::new(g.clone());
    let mut logits = vec![0.0f32; 6];
    let allowed = mask.apply(&st, &mut logits);
    assert!(
        allowed >= 3,
        "‘{{’, ‘[’, ‘\"’, ‘true’ are all valid JSON openers"
    );
    assert_eq!(logits[0], 0.0, "'{{' allowed");
    assert_eq!(logits[1], 0.0, "'[' allowed");
    assert_eq!(
        logits[2],
        f32::NEG_INFINITY,
        "'x' is not a valid JSON start"
    );
    assert_eq!(logits[4], f32::NEG_INFINITY, "EOS masked before any value");

    // After committing a complete value, EOS becomes allowed.
    st.accept(b"true");
    let mut logits = vec![0.0f32; 6];
    mask.apply(&st, &mut logits);
    assert_eq!(logits[4], 0.0, "EOS allowed once the value is complete");
}

#[test]
fn custom_grammar_enum_choice() {
    // A grammar that only permits one of three exact words — the classic
    // classification / tool-routing use case.
    let g = Arc::new(Grammar::parse(r#"root ::= "yes" | "no" | "maybe""#).unwrap());
    assert!(accepts_complete(&g, "yes"));
    assert!(accepts_complete(&g, "maybe"));
    assert!(!accepts_complete(&g, "y"));
    assert!(!accepts_complete(&g, "yesno"));
}

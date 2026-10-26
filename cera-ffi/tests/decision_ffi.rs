//! The d1 decision bindings against the public `d1-omni-600M` files.
//!
//! Needs the real model, so the tests are `#[ignore]`d; run them with
//! `CERA_D1_OMNI_GGUF=<model.gguf> CERA_D1_OMNI_MMPROJ=<mmproj.gguf> \
//!  cargo test -p cera-ffi --test decision_ffi -- --ignored`.

use cera_ffi::{BackendPreference, DecisionEngine, FfiError};

const TEXT_REQUEST: &str = r#"{"state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}"#;

fn open() -> std::sync::Arc<DecisionEngine> {
    let model = std::env::var("CERA_D1_OMNI_GGUF").expect("set CERA_D1_OMNI_GGUF");
    let mmproj = std::env::var("CERA_D1_OMNI_MMPROJ").ok();
    DecisionEngine::open(model, mmproj, BackendPreference::Cpu).expect("opening the model")
}

fn field(json: &serde_json::Value, path: &[&str]) -> f64 {
    path.iter()
        .fold(json, |v, k| &v[*k])
        .as_f64()
        .unwrap_or_else(|| panic!("no number at {path:?} in {json}"))
}

#[test]
#[ignore = "needs CERA_D1_OMNI_GGUF"]
fn a_text_request_is_answered_in_json() {
    let engine = open();
    assert_eq!(engine.decision_type(), "lfm2-d1-omni");
    assert_eq!(engine.backend_name(), "cpu");
    let response: serde_json::Value =
        serde_json::from_str(&engine.decide_json(TEXT_REQUEST.into()).unwrap()).unwrap();
    assert!(field(&response, &["answers", "refund", "noul"]) > 0.9);
    assert_eq!(response["answers"]["team"]["choice"], "billing");
    let urgency = field(&response, &["answers", "urgency", "score"]);
    assert!((urgency - 1.805).abs() < 0.06, "urgency {urgency}");
    assert_eq!(response["usage"]["output_tokens"], 0);
}

#[test]
#[ignore = "needs CERA_D1_OMNI_GGUF"]
fn a_bad_request_is_an_error_that_says_what_is_wrong() {
    let engine = open();
    for (request, needle) in [
        ("not json", "JSON"),
        (r#"{"state": "x"}"#, "questions"),
        (
            r#"{"state": "x", "questions": {"q": {"type": "choice", "instructions": "i"}}}"#,
            "choice",
        ),
    ] {
        let err = engine
            .decide_json(request.into())
            .expect_err("expected a refusal");
        assert!(matches!(err, FfiError::Backend { .. }), "{err:?}");
        assert!(err.to_string().contains(needle), "{request}: {err}");
    }
    // the engine still answers after a refusal
    engine.decide_json(TEXT_REQUEST.into()).unwrap();
}

#[test]
#[ignore = "needs CERA_D1_OMNI_GGUF and CERA_D1_OMNI_MMPROJ"]
fn media_without_a_tower_is_refused_and_with_one_answered() {
    let model = std::env::var("CERA_D1_OMNI_GGUF").expect("set CERA_D1_OMNI_GGUF");
    let text_only = DecisionEngine::open(model, None, BackendPreference::Auto).unwrap();
    assert!(!text_only.supports_images() && !text_only.supports_audio());
    let request = r#"{"state": null, "images": ["data:audio/wav;base64,AAAA"],
        "questions": {"q": {"type": "noul", "instructions": "Is it speech?"}}}"#;
    let err = text_only.decide_json(request.into()).unwrap_err();
    assert!(err.to_string().contains("speech tower"), "{err}");
    let full = open();
    assert!(full.supports_images() && full.supports_audio());
}

#[tokio::test]
#[ignore = "needs CERA_D1_OMNI_GGUF"]
async fn the_async_call_answers_and_a_dropped_one_leaves_the_engine_usable() {
    let engine = open();
    let json = std::sync::Arc::clone(&engine)
        .decide_json_async(TEXT_REQUEST.into())
        .await
        .unwrap();
    assert!(json.contains("\"answers\""));
    // drop a call mid-flight: the request stops at its next question and the next one runs
    let pending = std::sync::Arc::clone(&engine).decide_json_async(TEXT_REQUEST.into());
    let _ = tokio::time::timeout(std::time::Duration::from_millis(5), pending).await;
    let again = engine.decide_json_async(TEXT_REQUEST.into()).await.unwrap();
    assert_eq!(json, again);
}

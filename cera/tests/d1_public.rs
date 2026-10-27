//! The public `d1-omni` and `d1-3B` files against llama.cpp's `/v1/systemone`.
//!
//! These need the real models, so every test is `#[ignore]`d and reads its inputs from the
//! environment; run them with `cargo test -p cera --test d1_public -- --ignored`:
//!
//! - `CERA_D1_OMNI_GGUF`: the model, e.g. `d1-omni-600M-Q8_0.gguf`.
//! - `CERA_D1_OMNI_MMPROJ`: its `mmproj-d1-omni-600M-*.gguf` (image and speech requests are
//!   skipped without it).
//! - `CERA_D1_GOLDEN_DIR`: a directory of `req-{text,image,audio}.json` requests and the
//!   `resp-omni-{text,image,audio}.json` / `resp-3b-{text,image}.json` answers `llama-server`
//!   gave for them.
//! - `CERA_D1_3B_GGUF`, `CERA_D1_3B_MMPROJ`: the `d1-3B` model and its mmproj, for the 3B tests.
//! - `CERA_D1_BACKEND` (optional, default `cpu`): `cpu`, `gpu`, `metal`, `npu` or `auto`.
//! - `CERA_D1_TOLERANCE` (optional, default 0.05): the largest absolute difference allowed
//!   between a probability, a `noul` or a `score` here and in the golden answer. On Q8_0 files
//!   the two engines differ by up to 0.043 without BLAS and 0.023 with `--features blas` (each
//!   rounds its activations its own way, and llama.cpp's own Q8_0 answer sits 0.008 from its
//!   F16 one); on the F16 files they agree to 0.004. The choices must match exactly.

#![cfg(feature = "mmap")]

use std::path::{Path, PathBuf};

use cera::d1::{D1Engine, Json};
use cera::engine::BackendPreference;

/// The backend under test: `CERA_D1_BACKEND` (`cpu`, `gpu`, `metal`, `npu`, `auto`), default `cpu`.
fn backend() -> BackendPreference {
    std::env::var("CERA_D1_BACKEND")
        .ok()
        .map_or(BackendPreference::Cpu, |v| {
            BackendPreference::parse_str(&v).expect("a backend name")
        })
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn tolerance() -> f64 {
    std::env::var("CERA_D1_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05)
}

fn number(v: &Json) -> f64 {
    match v {
        Json::Float(f) => *f,
        Json::Int(i) => i.parse().expect("an integer"),
        other => panic!("not a number: {other:?}"),
    }
}

/// Every number of an answer that is a probability-like quantity, by path.
fn probabilities(answer: &Json) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    if let Some(v) = answer.get("noul") {
        out.push(("noul".to_string(), number(v)));
    }
    if let Some(v) = answer.get("score") {
        out.push(("score".to_string(), number(v)));
    }
    if let Some(Json::Object(entries)) = answer.get("probabilities") {
        out.extend(entries.iter().map(|(k, v)| (format!("p[{k}]"), number(v))));
    }
    out
}

fn compare(name: &str, got: &Json, want: &Json, tol: f64) -> f64 {
    let (Some(Json::Object(got_answers)), Some(Json::Object(want_answers))) =
        (got.get("answers"), want.get("answers"))
    else {
        panic!("{name}: a response without answers");
    };
    assert_eq!(
        got_answers.len(),
        want_answers.len(),
        "{name}: question count"
    );
    let mut worst = 0f64;
    for (id, want_answer) in want_answers {
        let got_answer = got_answers
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("{name}: no answer for `{id}`"));
        assert_eq!(
            got_answer.get("choice"),
            want_answer.get("choice"),
            "{name}.{id}: the choice"
        );
        for ((label, want_v), (_, got_v)) in probabilities(want_answer)
            .into_iter()
            .zip(probabilities(got_answer))
        {
            let diff = (want_v - got_v).abs();
            worst = worst.max(diff);
            assert!(
                diff <= tol,
                "{name}.{id}.{label}: golden {want_v:.5}, got {got_v:.5} (tolerance {tol})"
            );
        }
        // the confidence is derived from the probabilities, so it is held to the same bar
        if let (Some(w), Some(g)) = (want_answer.get("confidence"), got_answer.get("confidence")) {
            assert!(
                (number(w) - number(g)).abs() <= 2.0 * tol,
                "{name}.{id}.confidence: golden {}, got {}",
                number(w),
                number(g)
            );
        }
    }
    assert_eq!(
        got.get("usage").and_then(|u| u.get("input_tokens")),
        want.get("usage").and_then(|u| u.get("input_tokens")),
        "{name}: input tokens (the prompt layout)"
    );
    worst
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

#[test]
#[ignore = "needs CERA_D1_OMNI_GGUF, CERA_D1_GOLDEN_DIR and the public d1-omni files"]
fn the_public_omni_files_answer_like_llama_server() {
    let (Some(model), Some(golden)) = (
        env_path("CERA_D1_OMNI_GGUF"),
        env_path("CERA_D1_GOLDEN_DIR"),
    ) else {
        panic!("set CERA_D1_OMNI_GGUF and CERA_D1_GOLDEN_DIR");
    };
    let mmproj = env_path("CERA_D1_OMNI_MMPROJ");
    let engine = D1Engine::open(&model, mmproj.as_deref(), backend())
        .expect("opening the public d1-omni model");
    assert_eq!(engine.kind(), "lfm2-d1-omni");
    let tol = tolerance();
    let cases: Vec<&str> = ["text", "image", "audio"]
        .into_iter()
        .filter(|case| *case == "text" || mmproj.is_some())
        .collect();
    for case in cases {
        let request = read(&golden.join(format!("req-{case}.json")));
        let want = Json::parse(&read(&golden.join(format!("resp-omni-{case}.json")))).unwrap();
        let got = Json::parse(&engine.decide_json(&request).unwrap()).unwrap();
        let worst = compare(case, &got, &want, tol);
        eprintln!("{case}: worst absolute difference {worst:.5}");
    }
}

fn engine_3b(with_mmproj: bool) -> D1Engine {
    let model = env_path("CERA_D1_3B_GGUF").expect("set CERA_D1_3B_GGUF");
    let mmproj = env_path("CERA_D1_3B_MMPROJ").filter(|_| with_mmproj);
    D1Engine::open(&model, mmproj.as_deref(), backend()).expect("opening the public d1-3B model")
}

#[test]
#[ignore = "needs CERA_D1_3B_GGUF, CERA_D1_3B_MMPROJ and CERA_D1_GOLDEN_DIR"]
fn the_public_3b_files_answer_like_llama_server() {
    let golden = env_path("CERA_D1_GOLDEN_DIR").expect("set CERA_D1_GOLDEN_DIR");
    let engine = engine_3b(true);
    assert_eq!(engine.kind(), "lfm2-d1");
    assert!(engine.supports_images() && !engine.supports_audio());
    let tol = tolerance();
    for case in ["text", "image"] {
        let request = read(&golden.join(format!("req-{case}.json")));
        let want = Json::parse(&read(&golden.join(format!("resp-3b-{case}.json")))).unwrap();
        let got = Json::parse(&engine.decide_json(&request).unwrap()).unwrap();
        let worst = compare(case, &got, &want, tol);
        eprintln!("3b {case}: worst absolute difference {worst:.5}");
    }
}

/// The questions of a request share the image and state prefix and continue from a checkpoint
/// of it: each answers as it does asked alone.
#[test]
#[ignore = "needs CERA_D1_3B_GGUF, CERA_D1_3B_MMPROJ and CERA_D1_GOLDEN_DIR"]
fn the_3b_shared_prefix_does_not_change_an_answer() {
    let golden = env_path("CERA_D1_GOLDEN_DIR").expect("set CERA_D1_GOLDEN_DIR");
    let engine = engine_3b(true);
    let request = Json::parse(&read(&golden.join("req-image.json"))).unwrap();
    let together = Json::parse(&engine.decide_json(&request.dumps()).unwrap()).unwrap();
    let Some(Json::Object(questions)) = request.get("questions") else {
        panic!("no questions");
    };
    for (id, question) in questions {
        let Json::Object(mut alone) = request.clone() else {
            panic!("not an object");
        };
        for (key, value) in &mut alone {
            if key == "questions" {
                *value = Json::Object(vec![(id.clone(), question.clone())]);
            }
        }
        let single =
            Json::parse(&engine.decide_json(&Json::Object(alone).dumps()).unwrap()).unwrap();
        let (Some(Json::Object(a)), Some(Json::Object(b))) =
            (single.get("answers"), together.get("answers"))
        else {
            panic!("no answers");
        };
        let want = Json::Object(vec![("answers".into(), Json::Object(a.clone()))]);
        let got = Json::Object(vec![(
            "answers".into(),
            Json::Object(b.iter().filter(|(k, _)| k == id).cloned().collect()),
        )]);
        let worst = compare_answers(id, &got, &want);
        assert!(
            worst < 2e-3,
            "{id}: shared-prefix answer differs by {worst}"
        );
    }
}

/// Same as [`compare`] without the usage check, for answers taken from different requests.
fn compare_answers(name: &str, got: &Json, want: &Json) -> f64 {
    let strip = |j: &Json| {
        let usage = Json::Object(vec![("input_tokens".into(), Json::Int("0".into()))]);
        Json::Object(vec![
            ("answers".into(), j.get("answers").cloned().unwrap()),
            ("usage".into(), usage),
        ])
    };
    compare(name, &strip(got), &strip(want), 1.0)
}

#[test]
#[ignore = "needs CERA_D1_3B_GGUF"]
fn the_3b_refuses_what_it_cannot_read() {
    let text_only = engine_3b(false);
    assert!(!text_only.supports_images());
    let with_image = r#"{"state": "s", "images": ["data:image/png;base64,AAAA"],
        "questions": {"q": {"type": "noul", "instructions": "i"}}}"#;
    let err = text_only.decide_json(with_image).unwrap_err();
    assert!(err.to_string().contains("no vision tower"), "{err}");
    let audio = r#"{"state": "s", "images": ["data:audio/wav;base64,AAAA"],
        "questions": {"q": {"type": "noul", "instructions": "i"}}}"#;
    let err = text_only.decide_json(audio).unwrap_err();
    assert!(err.to_string().contains("no speech input"), "{err}");
    // and it still answers
    text_only
        .decide_json(r#"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}"#)
        .unwrap();
}

/// The prompt cera builds is the one the model's own `systemone` template renders: the same
/// token ids and the same marker positions, for every question type and media kind.
#[test]
#[ignore = "needs CERA_D1_OMNI_GGUF (the template and tokenizer are read from it)"]
fn the_prompt_matches_the_systemone_template() {
    use cera::d1::prompt::{Media, PromptTokens, Question, encode};
    use minijinja::Environment;

    let model = env_path("CERA_D1_OMNI_GGUF").expect("set CERA_D1_OMNI_GGUF");
    let gguf = cera::gguf::GgufFile::open(&model).unwrap();
    let template = gguf
        .get_str("tokenizer.chat_template.systemone")
        .expect("the systemone template")
        .to_string();
    let tokenizer = cera::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let ids = PromptTokens::from_tokenizer(&tokenizer).unwrap();

    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("systemone", &template).unwrap();

    const SEP: &str = "<<d1omni:sep>>";
    const MARKS: [&str; 3] = [
        "<<d1omni:state>>",
        "<<d1omni:question>>",
        "<<d1omni:option>>",
    ];
    // `(question JSON, state)` pairs; the state is plain text in all of them
    let questions = [
        r#"{"type":"noul","instructions":"Is it a refund?"}"#,
        r#"{"type":"noul","instructions":"Is it a refund?","criteria":{"true":"A refund is asked","false":"Nothing is asked"}}"#,
        r#"{"type":"noul","instructions":"Is it a refund?","criteria":{"yes":"Y"}}"#,
        r#"{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges","tech":null,"fraud":""}}"#,
        r#"{"type":"choice","instructions":"Which team?","criteria":{"a":null,"b":null}}"#,
        r#"{"type":"score","instructions":"How urgent?","criteria":["low","mid","high"]}"#,
    ];
    for state in [
        "I was charged twice.",
        "",
        "Ünïcode, \"quotes\" and <b>tags</b>",
    ] {
        for (media, has_images, audio) in [
            (Media::None, false, false),
            (Media::Image, true, false),
            (Media::Audio, false, true),
        ] {
            for text in questions {
                let json = Json::parse(text).unwrap();
                let question = Question::from_json(&json).unwrap();
                // the options as the server hands them to the template (noul: false, true)
                let options: Vec<serde_json::Value> = match question.kind {
                    cera::d1::QuestionType::Noul => {
                        let side = |names: [&str; 2]| {
                            names
                                .iter()
                                .find_map(|n| question.criteria.iter().find(|(k, _)| k == n))
                                .map(|(_, v)| to_serde(v))
                                .unwrap_or(serde_json::Value::Null)
                        };
                        vec![
                            serde_json::json!({"key": "false", "description": side(["false", "no"])}),
                            serde_json::json!({"key": "true", "description": side(["true", "yes"])}),
                        ]
                    }
                    _ => question
                        .criteria
                        .iter()
                        .map(|(k, v)| serde_json::json!({"key": k, "description": to_serde(v)}))
                        .collect(),
                };
                let state_value = if state.is_empty() && audio {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(state.to_string())
                };
                let ctx = minijinja::value::Serde(serde_json::json!({
                    "state": state_value,
                    "instructions": question.instructions,
                    "type": question.kind.name(),
                    "options": options,
                    "images": if has_images { vec!["<media>"] } else { vec![] },
                    "audio": audio,
                    "sep": SEP,
                    "mark_state": MARKS[0],
                    "mark_question": MARKS[1],
                    "mark_option": MARKS[2],
                }));
                let rendered = env.get_template("systemone").unwrap().render(ctx).unwrap();
                // llama.cpp tokenizes every piece between the separators on its own, with the
                // delimiter tokens parsed, after dropping the marker that tags its budget
                let from_template: Vec<u32> = rendered
                    .split(SEP)
                    .skip(1)
                    .flat_map(|piece| {
                        let piece = MARKS
                            .iter()
                            .find_map(|m| piece.strip_prefix(m))
                            .unwrap_or(piece);
                        tokenizer.encode(piece)
                    })
                    .collect();
                let state_json = if state.is_empty() && audio {
                    Json::Object(Vec::new())
                } else {
                    Json::Str(state.to_string())
                };
                let (ours, markers) =
                    encode(&tokenizer, &ids, &state_json, &question, 16384, media).unwrap();
                assert_eq!(
                    ours, from_template,
                    "state {state:?}, {media:?}, question {text}\n{rendered}"
                );
                assert_eq!(markers.len(), question.options());
                assert!(markers.iter().all(|&m| ours[m] == ids.marker));
            }
        }
    }
}

fn to_serde(v: &Json) -> serde_json::Value {
    serde_json::from_str(&v.dumps()).expect("Json::dumps is JSON")
}

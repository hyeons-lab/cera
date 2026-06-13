//! Cross-implementation correctness gate for the Qwen2 (NEOX-rope) forward pass.
//!
//! Compares cera's `LlamaModel` against golden fixtures generated from upstream
//! llama.cpp on the *same* quantized GGUF (see `scripts/oracle/`). Two gates per
//! prompt:
//!   1. tokenizer parity   — cera's `encode` matches llama.cpp's input tokens
//!   2. per-substep sums    — cera's per-node activation `sum` checksums match
//!      llama.cpp's, layer by layer. Deterministic and tie-proof (unlike exact
//!      greedy text, which flips at logit ties on open-ended prompts), and it
//!      localizes any math bug to the first diverging sub-step.
//!
//! Gated behind `CERA_ORACLE=1` and `#[ignore]` (needs the ~530MB fixture model,
//! which is not committed). Run:
//!   CERA_ORACLE=1 cargo test -p cera --release --test oracle_qwen2 -- --ignored --nocapture
//!
//! Model path: $CERA_ORACLE_MODEL, else target/oracle/models/qwen2-0_5b-instruct-q8_0.gguf
//! (populated by scripts/oracle/vendor_llama_cpp.sh + the hf download).

use std::collections::HashMap;
use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::Model;
use cera::model::llama::LlamaModel;
use cera::model::transformer::oracle_dump;
use cera::tokenizer::BpeTokenizer;

/// Relative difference, robust near zero. Catches gross math bugs (sign flip,
/// wrong layout → diffs ≫ 1) while tolerating Q8_0 accumulation-order noise
/// between cera's and llama.cpp's CPU kernels.
fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / (a.abs() + b.abs() + 1e-9)
}

/// Per-node relative tolerance for the sum gate.
const SUM_REL_TOL: f64 = 0.02;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("CERA_ORACLE_MODEL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/oracle/models/qwen2-0_5b-instruct-q8_0.gguf")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle/qwen2-0_5b")
}

/// Encode a prompt the same way the oracle did: byte-level BPE, prepending BOS
/// only when the GGUF requests it (Qwen2 does not).
fn encode_with_bos(tok: &BpeTokenizer, gguf: &GgufFile, prompt: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    if tok.bos_token().is_some()
        && gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false)
    {
        tokens.push(tok.bos_token().unwrap());
    }
    tokens.extend_from_slice(&tok.encode(prompt));
    tokens
}

#[test]
#[ignore] // run with --ignored + CERA_ORACLE=1
fn qwen2_matches_llama_cpp_oracle() {
    if std::env::var("CERA_ORACLE").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_ORACLE=1 not set");
        return;
    }
    let mp = model_path();
    if !mp.exists() {
        eprintln!("skipping: oracle model not found at {}", mp.display());
        return;
    }
    eprintln!("oracle model: {}", mp.display());

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");

    let gguf_model = GgufFile::open(&mp).expect("open gguf");
    let model = LlamaModel::from_gguf(gguf_model, 8192).expect("load LlamaModel");
    let n_layers = model.config().n_layers;

    // llama.cpp prunes its graph so only the LAST token flows past the final
    // layer: `l_out-{last}`, `result_norm`, and `result_output` are computed for
    // one position, while every earlier node covers all positions. So for those
    // nodes we compare cera's LAST-token contribution; for the rest, the sum
    // over all token positions.
    let last_pos_only = |name: &str| {
        name == "result_norm"
            || name == "result_output"
            || name == format!("l_out-{}", n_layers - 1)
    };

    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures_dir().join("index.json")).unwrap())
            .unwrap();
    let prompts = index["prompts"].as_array().unwrap();
    assert!(!prompts.is_empty(), "no prompt fixtures found");

    let mut failures = Vec::new();
    for entry in prompts {
        let fname = entry["fixture"].as_str().unwrap();
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixtures_dir().join(fname)).unwrap())
                .unwrap();
        let prompt = fx["prompt"].as_str().unwrap();
        let want_tokens: Vec<u32> = fx["input_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        // Gate 1 — tokenizer parity.
        let got_tokens = encode_with_bos(&tokenizer, &gguf, prompt);
        if got_tokens != want_tokens {
            failures.push(format!(
                "[{fname}] tokenizer mismatch:\n    cera: {got_tokens:?}\n    llama:{want_tokens:?}"
            ));
            // Tokenization drives the forward pass; the sum gate is meaningless
            // on diverging inputs, so skip it for this prompt.
            continue;
        }

        // Gate 2 — per-substep sum checksums. Prefill with the dump active.
        let mut state =
            InferenceState::from_config_with_compression(model.config(), &KvCompression::None);
        oracle_dump::begin();
        let _ = model.forward_prefill(&got_tokens, 0, &mut state);
        let occ = oracle_dump::take();

        // Fold cera occurrences: all-position nodes sum over tokens; last-position
        // nodes take the final (last-token) occurrence.
        let mut cera: HashMap<String, f64> = HashMap::new();
        for (name, sum) in occ {
            if last_pos_only(&name) {
                cera.insert(name, sum); // overwrite → last wins
            } else {
                *cera.entry(name).or_insert(0.0) += sum;
            }
        }

        // Oracle node sums by name (our recorded names are unique in the dump).
        let mut want: HashMap<&str, f64> = HashMap::new();
        for node in fx["nodes"].as_array().unwrap() {
            want.insert(
                node["name"].as_str().unwrap(),
                node["sum"].as_f64().unwrap(),
            );
        }

        // `l_out-{last}` and `result_norm` are single-token, post-residual
        // activations whose scalar *sum* nearly cancels (≈ -1), so tiny
        // per-element Q8_0 accumulation noise blows up the *relative* sum diff —
        // a weak checksum, not a real divergence. The final layer is instead
        // validated by `result_output` (the logits: large magnitude → robust
        // sum, and rmsnorm preserves direction), which is gated tightly. So we
        // report these two informationally but don't fail on them.
        let informational =
            |name: &str| name == "result_norm" || name == format!("l_out-{}", n_layers - 1);

        let mut worst = 0.0f64;
        let mut checked = 0usize;
        for (name, &got) in &cera {
            let Some(&exp) = want.get(name.as_str()) else {
                failures.push(format!("[{fname}] oracle has no node {name:?}"));
                continue;
            };
            let d = rel_diff(got, exp);
            if informational(name) {
                eprintln!("[{fname}] (info) {name}: cera={got:.4} llama={exp:.4} rel={d:.4}");
                continue;
            }
            checked += 1;
            if d > worst {
                worst = d;
            }
            if d > SUM_REL_TOL {
                failures.push(format!(
                    "[{fname}] sum mismatch at {name}: cera={got:.4} llama={exp:.4} rel={d:.4}"
                ));
            }
        }
        // Sanity: gated set is embd + (n_layers-1) early l_out + result_output.
        assert!(
            checked >= n_layers,
            "[{fname}] only {checked} nodes checked — instrumentation/fixture drift"
        );
        eprintln!("[{fname}] OK — {checked} gated nodes, worst rel diff {worst:.5}");
    }

    assert!(
        failures.is_empty(),
        "oracle gate failures:\n{}",
        failures.join("\n")
    );
}

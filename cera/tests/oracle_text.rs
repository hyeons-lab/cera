//! Cross-implementation correctness gate for NEOX-rope text models (Qwen2,
//! Qwen3) against golden fixtures generated from upstream llama.cpp on the *same*
//! quantized GGUF (see `scripts/oracle/`).
//!
//! Iterates every fixture set under `tests/fixtures/oracle/<model>/`; each
//! `index.json` names its `model_file`, looked up under `target/oracle/models/`
//! (override the dir with `CERA_ORACLE_MODELS_DIR`). A model whose GGUF is absent
//! is skipped, so CI (which has neither the models nor llama.cpp) passes.
//!
//! Two gates per prompt:
//!   1. tokenizer parity   — cera's `encode` matches llama.cpp's input tokens
//!   2. per-layer sums      — cera's activation `sum` checksums match llama.cpp's,
//!      keyed by (node name, op) so repeated node names don't collide. Covers the
//!      embedding, every layer's residual-stream output (`l_out-{i}` — captures
//!      that layer's attention/rope/bias/FFN), and the final logits.
//!      Deterministic and tie-proof (unlike exact greedy text, which flips at
//!      logit ties on open-ended prompts), and localizes any math bug to the
//!      first diverging layer.
//!
//! Gated behind `CERA_ORACLE=1` and `#[ignore]`. Run:
//!   CERA_ORACLE=1 cargo test -p cera --release --test oracle_text -- --ignored --nocapture

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

/// Per-node relative tolerance for the sum gate. Sized to clear the observed
/// Q8_0 cross-implementation accumulation noise (≤ ~3.5% on early small-sum
/// residuals across Qwen2 + Qwen3, all prompts) with margin, while still
/// catching real math bugs — a sign flip / wrong layout / misapplied bias
/// shifts a node's sum by tens of percent to >100%, far above this.
const SUM_REL_TOL: f64 = 0.05;

/// Below this absolute sum, cancellation makes the *relative* diff meaningless,
/// so the node is reported but not gated. A real bug propagates to the many
/// large-magnitude nodes downstream, so this loses no coverage.
const SUM_MAG_FLOOR: f64 = 10.0;

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle")
}

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

/// Encode a prompt the same way the oracle did: byte-level BPE, prepending BOS
/// only when the GGUF requests it (Qwen does not).
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

/// Validate one model's fixture set. Returns failure strings (empty = pass);
/// `None` if the model GGUF is absent (skipped).
fn check_model(fixture_dir: &std::path::Path) -> Option<Vec<String>> {
    // Committed fixtures must parse — corruption is a hard failure, not a skip.
    // The *only* intended skip is the model GGUF being absent (CI / no download).
    let index_path = fixture_dir.join("index.json");
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&index_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", index_path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", index_path.display()));
    let model_file = index["model_file"].as_str().unwrap();
    let mp = models_dir().join(model_file);
    if !mp.exists() {
        eprintln!("skipping {model_file}: not found at {}", mp.display());
        return None;
    }
    eprintln!("=== oracle model: {} ===", mp.display());

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let model = LlamaModel::from_gguf(GgufFile::open(&mp).expect("open gguf"), 8192)
        .expect("load LlamaModel");
    let n_layers = model.config().n_layers;
    let last = n_layers - 1;

    // Each cera-recorded node maps to exactly one llama.cpp graph op, so the
    // oracle lookup is keyed by (name, op): names like `Qcur-{i}` repeat across
    // ops (MUL_MAT/ADD/RESHAPE/ROPE) in the fixture, and a name-only key would
    // silently collide. cera records the embedding, each layer's residual-stream
    // output (`l_out-{i}`), the final norm, and the logits.
    //
    // Why the residual stream and not finer sub-steps: `l_out-{i}` is the layer's
    // output, so any bug in that layer's attention/rope/bias/FFN surfaces here
    // and localizes to the layer. The finer post-rope Q/K sums were evaluated as
    // gate nodes and rejected — rope rotation makes them cancel toward zero, so
    // their relative sum diff is a noisy checksum (>10% under pure Q8_0 noise).
    let expected_op = |name: &str| -> &'static str {
        if name == "embd" {
            "GET_ROWS"
        } else if name.starts_with("l_out-") {
            "ADD"
        } else if name == "result_norm" {
            "MUL"
        } else if name == "result_output" {
            "MUL_MAT"
        } else {
            panic!("unmapped oracle node name {name:?}")
        }
    };

    // llama.cpp prunes its graph so only the LAST token flows past the final
    // layer: every node of layer `last` (and the result_* nodes) covers one
    // position; earlier layers cover all positions. Fold cera occurrences to
    // match: sum all-position nodes over tokens, take the last occurrence for
    // last-position nodes.
    let last_pos_only =
        |name: &str| name.starts_with("result_") || name.ends_with(&format!("-{last}"));
    // `l_out-{last}` and `result_norm` are single-token, post-residual sums that
    // nearly cancel (≈ 0), and `result_norm` derives from that noisy residual —
    // tiny Q8_0 accumulation noise blows up their *relative* sum diff, a weak
    // checksum. The final layer is validated instead by `result_output` (logits:
    // large magnitude, rmsnorm preserves direction). Report these two
    // informationally rather than gating them.
    let informational = |name: &str| name == "result_norm" || name == format!("l_out-{last}");

    let mut failures = Vec::new();
    for entry in index["prompts"].as_array().unwrap() {
        let fname = entry["fixture"].as_str().unwrap();
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture_dir.join(fname)).unwrap())
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
            continue; // tokenization drives the forward pass; sum gate moot
        }

        // Gate 2 — per-substep sum checksums. Prefill with the dump active.
        let mut state =
            InferenceState::from_config_with_compression(model.config(), &KvCompression::None);
        oracle_dump::begin();
        let _ = model.forward_prefill(&got_tokens, 0, &mut state);
        let occ = oracle_dump::take();

        let mut cera: HashMap<String, f64> = HashMap::new();
        for (name, sum) in occ {
            if last_pos_only(&name) {
                cera.insert(name, sum); // last wins
            } else {
                *cera.entry(name).or_insert(0.0) += sum;
            }
        }
        // Oracle sums keyed by (name, op) so per-substep nodes don't collide.
        let mut want: HashMap<(&str, &str), f64> = HashMap::new();
        for node in fx["nodes"].as_array().unwrap() {
            want.insert(
                (node["name"].as_str().unwrap(), node["op"].as_str().unwrap()),
                node["sum"].as_f64().unwrap(),
            );
        }

        let mut worst = 0.0f64;
        let mut checked = 0usize;
        for (name, &got) in &cera {
            let op = expected_op(name);
            let Some(&exp) = want.get(&(name.as_str(), op)) else {
                failures.push(format!("[{fname}] oracle has no node ({name:?}, {op:?})"));
                continue;
            };
            let d = rel_diff(got, exp);
            // Skip nodes that can't be a reliable checksum: explicitly noisy
            // ones, and any whose oracle sum is near zero (cancellation makes the
            // relative diff meaningless). Real bugs still surface on the many
            // large-magnitude nodes downstream.
            if informational(name) || exp.abs() < SUM_MAG_FLOOR {
                eprintln!("[{fname}] (info) {name}/{op}: cera={got:.4} llama={exp:.4} rel={d:.4}");
                continue;
            }
            checked += 1;
            worst = worst.max(d);
            if d > SUM_REL_TOL {
                failures.push(format!(
                    "[{fname}] sum mismatch at {name}/{op}: cera={got:.4} llama={exp:.4} rel={d:.4}"
                ));
            }
        }
        // Most per-layer residuals should be gated (a handful of near-zero-sum
        // ones are legitimately skipped by the magnitude floor). A much smaller
        // count means the instrumentation or fixtures drifted.
        assert!(
            checked >= n_layers / 2,
            "[{fname}] only {checked} nodes checked — instrumentation/fixture drift"
        );
        eprintln!("[{fname}] OK — {checked} gated nodes, worst rel diff {worst:.5}");
    }
    Some(failures)
}

#[test]
#[ignore] // run with --ignored + CERA_ORACLE=1
fn text_models_match_llama_cpp_oracle() {
    if std::env::var("CERA_ORACLE").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_ORACLE=1 not set");
        return;
    }

    let mut dirs: Vec<PathBuf> = std::fs::read_dir(fixtures_root())
        .expect("read fixtures dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no oracle fixture sets found");

    let mut all_failures = Vec::new();
    let mut ran = 0usize;
    for dir in &dirs {
        if let Some(failures) = check_model(dir) {
            ran += 1;
            all_failures.extend(failures);
        }
    }

    if ran == 0 {
        eprintln!(
            "skipping: no oracle models under {}",
            models_dir().display()
        );
        return;
    }
    assert!(
        all_failures.is_empty(),
        "oracle gate failures:\n{}",
        all_failures.join("\n")
    );
}

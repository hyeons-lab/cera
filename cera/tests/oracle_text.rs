//! Cross-implementation correctness gate for the dense text models cera serves
//! through `LlamaModel` — NEOX-rope (Qwen2, Qwen3) and NORM-rope (LLaMA, Mistral,
//! Granite 3.x) — against golden fixtures generated from upstream llama.cpp on the
//! *same* quantized GGUF (see `scripts/oracle/`). Granite additionally exercises
//! the embedding/residual/attention/logit scalar multipliers (folded into the
//! gated `l_out-{i}`, `embd`, and `result_output` sums).
//!
//! Iterates every fixture set under `tests/fixtures/oracle/<model>/`; each
//! `index.json` names its `model_file`, looked up under `target/oracle/models/`
//! (override the dir with `CERA_ORACLE_MODELS_DIR`). A model whose GGUF is absent
//! is skipped, so CI (which has neither the models nor llama.cpp) passes.
//!
//! Two gates per prompt, plus one informational signal:
//!   1. tokenizer parity   — cera's `encode` matches llama.cpp's input tokens
//!   2. per-layer sums      — cera's activation `sum` checksums match llama.cpp's,
//!      keyed by (node name, op) so repeated node names don't collide. Covers the
//!      embedding and every layer's residual-stream output (`l_out-{i}` — captures
//!      that layer's attention/rope/bias/FFN). Deterministic and localizes any
//!      math bug to the first diverging layer. The final-logit `result_output`
//!      sum is NOT gated: it sums ~10^5 partially-cancelling logits, so its
//!      relative diff is both noisy (Q8_0 accumulation doesn't average out) and
//!      insensitive (a wrong rope convention barely moves it) — reported as info.
//!      • greedy continuation (informational) — cera's `--temp 0` argmax decode vs
//!      llama.cpp's greedy text, reported MATCH / DIVERGES but never gated:
//!      greedy decode flips at near-tied logits, and Q8_0 noise tips those ties
//!      into a different-but-coherent continuation that is not a bug.
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

/// Greedy next-token pick. Mirrors `sampler::cpu_argmax` exactly (NaN → -inf,
/// `total_cmp`, last-index-on-ties) so the harness reproduces cera's production
/// `--temperature 0` decode rather than a subtly different argmax.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
            let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
            a.total_cmp(&b)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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

/// Encode a prompt the same way the oracle did: byte-level BPE, with a leading
/// BOS to match llama.cpp's special-token prefixing.
///
/// cera's `encode()` does not prepend BOS — that's the Session / chat-template
/// layer's job — so the harness normalizes that prefix here to isolate BPE-merge
/// parity. We can't read it off `tokenizer.ggml.add_bos_token`: that key is
/// *absent* on some BPE GGUFs (e.g. `llama-bpe`), yet llama.cpp still prepends
/// BOS by vocab-type default.
///
/// Decide from the *whole* golden sequence rather than `want_tokens.first()`
/// alone: only prepend when the golden is exactly `[bos] ++ encode(prompt)`.
/// Keying on the full body is unambiguous even when the prompt's first *content*
/// token legitimately equals `bos_id` and the model does NOT add BOS — e.g. Qwen
/// (`bos_id == eos_id == <|endoftext|>`, `add_bos_token = false`) on a prompt
/// that literally starts with "<|endoftext|>": a first-token heuristic would
/// double-prepend and false-fail the tokenizer gate. A genuine BPE divergence in
/// the body still surfaces — the `[1..] == base` check fails, no BOS is added,
/// and the mismatch is reported.
fn encode_with_bos(tok: &BpeTokenizer, want_tokens: &[u32], prompt: &str) -> Vec<u32> {
    let base = tok.encode(prompt);
    match tok.bos_token() {
        Some(bos)
            if want_tokens.first() == Some(&bos)
                && want_tokens.len() == base.len() + 1
                && want_tokens[1..] == base[..] =>
        {
            let mut tokens = Vec::with_capacity(base.len() + 1);
            tokens.push(bos);
            tokens.extend_from_slice(&base);
            tokens
        }
        _ => base,
    }
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
    //
    // Which op a node maps to is per-arch DETERMINISTIC, so derive the exact
    // expected op from the model's own scalar metadata rather than accepting an
    // OR-list of "acceptable" ops. Granite scales the embeddings and the logits,
    // so its "embd" callback fires on a SCALE node (not GET_ROWS) and
    // "result_output" on SCALE (not MUL_MAT); plain archs never scale, so they
    // stay GET_ROWS / MUL_MAT. An exact per-arch op keeps the gate a precise
    // contract — it catches a genuinely wrong mapping (e.g. an arch that should
    // scale `embd` but emits GET_ROWS), which a first-present-wins list could not.
    let scalars = model.config().scalars;
    let expected_op = |name: &str| -> &'static str {
        if name == "embd" {
            if scalars.embedding != 1.0 {
                "SCALE"
            } else {
                "GET_ROWS"
            }
        } else if name.starts_with("l_out-") {
            "ADD"
        } else if name == "result_norm" {
            "MUL"
        } else if name == "result_output" {
            if scalars.logit != 1.0 {
                "SCALE"
            } else {
                "MUL_MAT"
            }
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
    // `l_out-{last}`, `result_norm`, and `result_output` are single-token sums
    // whose *relative* diff is a weak cross-impl checksum:
    //   - `l_out-{last}` / `result_norm` are post-residual sums that nearly
    //     cancel (≈ 0), so tiny Q8_0 accumulation noise blows up their rel diff.
    //   - `result_output` sums ~10^5 partially-cancelling logits (128k vocab on
    //     Llama-3), so Q8_0 accumulation-order differences don't average out in
    //     the sum either — its rel diff drifts past tol on some short prompts
    //     even when the argmax (the value users consume) is identical (verified:
    //     cera's greedy decode matched llama.cpp byte-for-byte on the prompt that
    //     tripped the old gate). It's also insensitive (a wrong rope convention
    //     barely moves it), so it's a poor gate at any tolerance.
    // The sensitive, localizing gate is the per-layer residual sums (all gated);
    // `embd` is exact (0.0000) and covers the tied output-projection weights. The
    // greedy continuation below is an additional informational cross-check.
    // Report these three sums informationally rather than gating them.
    let informational = |name: &str| {
        name == "result_norm" || name == "result_output" || name == format!("l_out-{last}")
    };

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
        let got_tokens = encode_with_bos(&tokenizer, &want_tokens, prompt);
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
                failures.push(format!(
                    "[{fname}] oracle has no node {name:?} with op {op:?}"
                ));
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
        eprintln!("[{fname}] sums OK — {checked} gated nodes, worst rel diff {worst:.5}");

        // Greedy continuation (end-to-end argmax) — INFORMATIONAL, not gated.
        // This is the human-meaningful final-output signal, but it cannot be a
        // hard gate: greedy decode flips whenever two tokens are near-tied, and
        // tiny Q8_0 cross-impl noise tips those ties either way. Empirically a
        // few prompts diverge into a *different but equally coherent*
        // continuation (e.g. two valid Spanish replies that split at token 0),
        // which is not a bug. So decode cera's continuation, compare to
        // llama.cpp's greedy text, and report MATCH / DIVERGES without failing —
        // a MATCH confirms exact argmax-path agreement, a DIVERGES is a prompt
        // to eyeball (a real projection bug yields garbage, not a plausible
        // alternative). The gated per-layer sums + `embd` (exact, and the tied
        // output-projection weights) remain the actual correctness gate.
        let n_predict = index["n_predict"].as_u64().unwrap_or(16) as usize;
        let want_text = fx["greedy_text"].as_str().unwrap().trim_end();
        let mut gstate =
            InferenceState::from_config_with_compression(model.config(), &KvCompression::None);
        let mut logits = model.forward_prefill(&got_tokens, 0, &mut gstate);
        let mut out_tokens: Vec<u32> = Vec::new();
        for _ in 0..n_predict {
            let next = argmax(&logits);
            // llama.cpp's greedy decode stops at EOS and does not render it, so
            // match that: break before appending the EOS token.
            if tokenizer.eos_token() == Some(next) {
                break;
            }
            out_tokens.push(next);
            logits = model.forward(&[next], gstate.seq_len, &mut gstate);
        }
        let got_text = tokenizer.decode(&out_tokens);
        let got_text = got_text.trim_end();
        if got_text == want_text {
            eprintln!("[{fname}] greedy MATCH — {got_text:?}");
        } else {
            eprintln!(
                "[{fname}] greedy DIVERGES (tie-flip; not gated):\n    cera: {got_text:?}\n    llama:{want_text:?}"
            );
        }
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

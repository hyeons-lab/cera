//! Cross-implementation parity tests for Mamba-2 SSM hybrid models
//! (Granite 4.0-h interleaved hybrid and Falcon H1 parallel hybrid)
//! against upstream llama.cpp oracle reference nodes.

#![cfg(feature = "mmap")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::load_model;
use cera::model::transformer::oracle_dump;

/// Relative difference, robust near zero.
fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / (a.abs() + b.abs() + 1e-9)
}

const SUM_REL_TOL: f64 = 0.01;

fn find_or_create_fixture(model_name: &str) -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("/tmp").join(model_name),
        std::env::temp_dir().join(model_name),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script_path = manifest_dir.join("../scripts/oracle/create_mamba2_test_models.py");
    if script_path.exists() {
        let status = Command::new("python3").arg(&script_path).status().ok()?;
        if status.success() {
            for c in &candidates {
                if c.exists() {
                    return Some(c.clone());
                }
            }
        }
    }

    None
}

#[test]
fn test_granite_hybrid_oracle_parity() {
    let Some(model_path) = find_or_create_fixture("test_granite_hybrid.gguf") else {
        if std::env::var("CERA_STRICT_TESTS").is_ok() {
            panic!("test_granite_hybrid.gguf not found and could not be generated");
        }
        eprintln!("Skipping test_granite_hybrid_oracle_parity: fixture not available");
        return;
    };

    let gguf = GgufFile::open(&model_path).expect("open granite hybrid gguf");
    let model = load_model(gguf, None, 256).expect("load granite hybrid model");
    let mut state = InferenceState::from_config(model.config()).expect("create inference state");

    // Expected oracle node sums for prompt "A" (token id 69) from upstream llama.cpp:
    // l_out-0: -0.454092
    // l_out-1: -0.238577
    // result_norm: -2.949017
    // result_output: 0.475466
    let expected: [(&str, f64); 4] = [
        ("l_out-0", -0.454092),
        ("l_out-1", -0.238577),
        ("result_norm", -2.949017),
        ("result_output", 0.475466),
    ];

    oracle_dump::begin();
    let logits = model.forward(&[69], 0, &mut state);
    let dumped = oracle_dump::take();

    assert_eq!(logits.len(), model.config().vocab_size);
    let dump_map: HashMap<String, f64> = dumped.into_iter().collect();

    for (node, want_sum) in expected {
        let got_sum = dump_map
            .get(node)
            .copied()
            .unwrap_or_else(|| panic!("missing dump node {node}"));
        let diff = rel_diff(got_sum, want_sum);
        println!("Granite {node}: got {got_sum:.6}, expected {want_sum:.6}, rel_diff = {diff:.6}");
        assert!(
            diff < SUM_REL_TOL,
            "Granite hybrid node {node} divergence: got {got_sum}, expected {want_sum}, rel_diff {diff} >= {SUM_REL_TOL}"
        );
    }

    // Prefill test: forward_prefill on same token yields matching logits
    let mut state2 = InferenceState::from_config(model.config()).expect("create inference state 2");
    oracle_dump::begin();
    let prefill_logits = model.forward_prefill(&[69], 0, &mut state2);
    let prefill_dump = oracle_dump::take();
    let prefill_map: HashMap<String, f64> = prefill_dump.into_iter().collect();

    for (node, want_sum) in expected {
        let got_sum = prefill_map
            .get(node)
            .copied()
            .unwrap_or_else(|| panic!("missing prefill dump node {node}"));
        let diff = rel_diff(got_sum, want_sum);
        assert!(
            diff < SUM_REL_TOL,
            "Granite prefill node {node} divergence: got {got_sum}, expected {want_sum}"
        );
    }

    assert_eq!(logits.len(), prefill_logits.len());
    let logit_max_diff = logits
        .iter()
        .zip(prefill_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        logit_max_diff < 1e-5,
        "prefill logits diverged from single-token forward: max diff {logit_max_diff}"
    );

    // Multi-token decode vs prefill equivalence
    let logits_step2 = model.forward(&[70], 1, &mut state);
    let mut state_multi =
        InferenceState::from_config(model.config()).expect("create multi-token inference state");
    let prefill_multi_logits = model.forward_prefill(&[69, 70], 0, &mut state_multi);
    let multi_diff = logits_step2
        .iter()
        .zip(prefill_multi_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        multi_diff < 1e-5,
        "multi-token prefill diverged from sequential decode: max diff {multi_diff}"
    );

    // State snapshot and restore rollback
    let snapshot = state.snapshot().expect("take state snapshot at pos 2");
    let logits_step3 = model.forward(&[71], 2, &mut state);
    state.restore(&snapshot);
    assert_eq!(state.seq_len, 2);
    let logits_step3_restored = model.forward(&[71], 2, &mut state);
    let rollback_diff = logits_step3
        .iter()
        .zip(logits_step3_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        rollback_diff < 1e-5,
        "snapshot rollback diverged: max diff {rollback_diff}"
    );

    // State reset for reuse
    state.clear_for_reuse();
    assert_eq!(state.seq_len, 0);
    let clean_logits = model.forward(&[69], 0, &mut state);
    let reuse_diff = logits
        .iter()
        .zip(clean_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        reuse_diff < 1e-5,
        "clear_for_reuse diverged: max diff {reuse_diff}"
    );

    // Truncate test: calling truncate_to with safe_len > 0 safely resets state to 0
    // because Mamba-2 recurrent layers lack step history.
    let _ = model.forward(&[70], 1, &mut state);
    assert_eq!(state.seq_len, 2);
    state.truncate_to(1);
    assert_eq!(state.seq_len, 0);
    let clean_after_trunc = model.forward(&[69], 0, &mut state);
    let trunc_diff = logits
        .iter()
        .zip(clean_after_trunc.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        trunc_diff < 1e-5,
        "state after safe_len > 0 truncation diverged from clean forward: max diff {trunc_diff}"
    );
}

#[test]
fn test_falcon_h1_oracle_parity() {
    let Some(model_path) = find_or_create_fixture("test_falcon_h1.gguf") else {
        if std::env::var("CERA_STRICT_TESTS").is_ok() {
            panic!("test_falcon_h1.gguf not found and could not be generated");
        }
        eprintln!("Skipping test_falcon_h1_oracle_parity: fixture not available");
        return;
    };

    let gguf = GgufFile::open(&model_path).expect("open falcon h1 gguf");
    let model = load_model(gguf, None, 256).expect("load falcon h1 model");
    let mut state = InferenceState::from_config(model.config()).expect("create inference state");

    // Expected oracle node sums for prompt "A" (token id 69) from upstream llama.cpp:
    // l_out-0: 0.504395
    // l_out-1: -0.219094
    // result_norm: -0.516712
    // result_output: 9.447885
    let expected: [(&str, f64); 4] = [
        ("l_out-0", 0.504395),
        ("l_out-1", -0.219094),
        ("result_norm", -0.516712),
        ("result_output", 9.447885),
    ];

    oracle_dump::begin();
    let logits = model.forward(&[69], 0, &mut state);
    let dumped = oracle_dump::take();

    assert_eq!(logits.len(), model.config().vocab_size);
    let dump_map: HashMap<String, f64> = dumped.into_iter().collect();

    for (node, want_sum) in expected {
        let got_sum = dump_map
            .get(node)
            .copied()
            .unwrap_or_else(|| panic!("missing dump node {node}"));
        let diff = rel_diff(got_sum, want_sum);
        println!("Falcon {node}: got {got_sum:.6}, expected {want_sum:.6}, rel_diff = {diff:.6}");
        assert!(
            diff < SUM_REL_TOL,
            "Falcon H1 node {node} divergence: got {got_sum}, expected {want_sum}, rel_diff {diff} >= {SUM_REL_TOL}"
        );
    }

    // Prefill test: forward_prefill on same token yields matching logits
    let mut state2 = InferenceState::from_config(model.config()).expect("create inference state 2");
    oracle_dump::begin();
    let prefill_logits = model.forward_prefill(&[69], 0, &mut state2);
    let prefill_dump = oracle_dump::take();
    let prefill_map: HashMap<String, f64> = prefill_dump.into_iter().collect();

    for (node, want_sum) in expected {
        let got_sum = prefill_map
            .get(node)
            .copied()
            .unwrap_or_else(|| panic!("missing prefill dump node {node}"));
        let diff = rel_diff(got_sum, want_sum);
        assert!(
            diff < SUM_REL_TOL,
            "Falcon prefill node {node} divergence: got {got_sum}, expected {want_sum}"
        );
    }

    assert_eq!(logits.len(), prefill_logits.len());
    let logit_max_diff = logits
        .iter()
        .zip(prefill_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        logit_max_diff < 1e-5,
        "prefill logits diverged from single-token forward: max diff {logit_max_diff}"
    );

    // Multi-token decode vs prefill equivalence
    let logits_step2 = model.forward(&[70], 1, &mut state);
    let mut state_multi =
        InferenceState::from_config(model.config()).expect("create multi-token inference state");
    let prefill_multi_logits = model.forward_prefill(&[69, 70], 0, &mut state_multi);
    let multi_diff = logits_step2
        .iter()
        .zip(prefill_multi_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        multi_diff < 1e-5,
        "multi-token prefill diverged from sequential decode: max diff {multi_diff}"
    );

    // State snapshot and restore rollback
    let snapshot = state.snapshot().expect("take state snapshot at pos 2");
    let logits_step3 = model.forward(&[71], 2, &mut state);
    state.restore(&snapshot);
    assert_eq!(state.seq_len, 2);
    let logits_step3_restored = model.forward(&[71], 2, &mut state);
    let rollback_diff = logits_step3
        .iter()
        .zip(logits_step3_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        rollback_diff < 1e-5,
        "snapshot rollback diverged: max diff {rollback_diff}"
    );

    // State reset for reuse
    state.clear_for_reuse();
    assert_eq!(state.seq_len, 0);
    let clean_logits = model.forward(&[69], 0, &mut state);
    let reuse_diff = logits
        .iter()
        .zip(clean_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        reuse_diff < 1e-5,
        "clear_for_reuse diverged: max diff {reuse_diff}"
    );

    // Truncate test: calling truncate_to with safe_len > 0 safely resets state to 0
    // because Mamba-2 recurrent layers lack step history.
    let _ = model.forward(&[70], 1, &mut state);
    assert_eq!(state.seq_len, 2);
    state.truncate_to(1);
    assert_eq!(state.seq_len, 0);
    let clean_after_trunc = model.forward(&[69], 0, &mut state);
    let trunc_diff = logits
        .iter()
        .zip(clean_after_trunc.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        trunc_diff < 1e-5,
        "state after safe_len > 0 truncation diverged from clean forward: max diff {trunc_diff}"
    );
}

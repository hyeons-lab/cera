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
    if let Ok(dir) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        let candidate = PathBuf::from(dir).join(model_name);
        if candidate.exists() && !candidate.is_symlink() {
            return Some(candidate);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.join("../target/oracle/models");
    let _ = std::fs::create_dir_all(&target_dir);
    let target = target_dir.join(model_name);
    if target.exists() && !target.is_symlink() {
        return Some(target);
    }

    let temp_dir = std::env::temp_dir();
    let temp_target = temp_dir.join(model_name);
    if temp_target.exists() && !temp_target.is_symlink() {
        return Some(temp_target);
    }

    let script_path = manifest_dir.join("../scripts/oracle/create_mamba2_test_models.py");
    if script_path.exists() {
        let status = Command::new("python3")
            .arg(&script_path)
            .arg(&target_dir)
            .status()
            .ok()?;
        if status.success() && target.exists() && !target.is_symlink() {
            return Some(target);
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

#[test]
fn test_mamba2_truncate_to_clears_state() {
    let config = cera::model::ModelConfig {
        architecture: "granitehybrid".into(),
        n_layers: 2,
        hidden_size: 64,
        intermediate_size: 128,
        n_heads: 4,
        n_kv_heads: 4,
        head_dim: 16,
        vocab_size: 256,
        max_seq_len: 128,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        block_types: vec![
            cera::model::BlockType::Mamba2,
            cera::model::BlockType::Attention,
        ],
        kv_heads_per_layer: vec![0, 4],
        conv_kernel_size: None,
        ssm: Some(cera::model::SsmConfig {
            d_conv: 4,
            d_inner: 64,
            d_state: 16,
            dt_rank: 4,
            n_group: 1,
        }),
        scalars: Default::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    };

    let mut state = InferenceState::from_config(&config).expect("create inference state");
    state.seq_len = 8;
    if let cera::kv_cache::LayerState::Mamba2 { conv_state, .. } = &mut state.layers[0] {
        conv_state.fill(1.0);
    }

    state.truncate_to(4);
    assert_eq!(state.seq_len, 0);
    if let cera::kv_cache::LayerState::Mamba2 {
        conv_state,
        ssm_state,
    } = &state.layers[0]
    {
        assert!(conv_state.iter().all(|&v| v == 0.0));
        assert!(ssm_state.iter().all(|&v| v == 0.0));
    }
}

#[test]
fn test_mamba2_session_rollback_clears_last_logits() {
    let Some(model_path) = find_or_create_fixture("test_granite_hybrid.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&model_path).expect("open gguf");
    let tokenizer = std::sync::Arc::new(cera::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap());
    let model: std::sync::Arc<dyn cera::model::Model> =
        load_model(gguf, None, 128).expect("load model").into();
    let mut session = cera::session::Session::new(
        model,
        tokenizer,
        cera::session::ModalityCapabilities::text_only(),
        cera::session::SessionConfig {
            ubatch_size: 1,
            ..Default::default()
        },
    )
    .expect("create session");

    session.append_tokens(&[69, 70]).expect("append tokens");
    assert!(session.last_logits().is_some());
    assert_eq!(session.position(), 2);

    session.cancel();
    let msg = cera::tokenizer::UserMessage {
        text: Some("test test".into()),
        images: Vec::new(),
        audio: None,
    };
    let append_res = session.append_user_message(&msg);
    assert!(append_res.is_err());
    assert_eq!(session.position(), 0);
    assert!(session.last_logits().is_none());
}

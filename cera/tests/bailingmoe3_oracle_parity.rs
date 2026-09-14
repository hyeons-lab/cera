//! Oracle parity and functional verification tests for Ling 3.0 Tiny (bailingmoe3)
//! hybrid linear, latent attention, and MoE architecture.

#![cfg(feature = "mmap")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCacheConfig, KvCompression, KvPrefixCache, LayerState};
use cera::model::bailingmoe3::BailingMoe3Model;
use cera::model::transformer::oracle_dump;
use cera::model::{BlockType, Model};
use cera::tokenizer::BpeTokenizer;

/// Cosine similarity between two f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn ensure_test_fixture() -> Option<PathBuf> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let target_dir = manifest_dir.join("../target/tmp/cera_test_bailingmoe3");
            let _ = std::fs::create_dir_all(&target_dir);
            let path = target_dir.join("test_bailingmoe3.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }

            let script = manifest_dir.join("../scripts/oracle/create_bailingmoe3_test_model.py");
            if !script.exists() {
                eprintln!(
                    "skipping: fixture generator script not found at {}",
                    script.display()
                );
                return None;
            }

            let probe = Command::new("python3")
                .args(["-c", "import numpy, gguf"])
                .output();
            let python_has_deps = matches!(probe, Ok(out) if out.status.success());

            for py in ["python3", "python"] {
                let status = Command::new(py).arg(&script).arg(&path).status();
                if matches!(status, Ok(st) if st.success())
                    && path.is_file()
                    && path.metadata().map(|m| m.len() > 1024).unwrap_or(false)
                {
                    return Some(path);
                }
            }

            eprintln!("skipping: python3/python failed to generate test_bailingmoe3.gguf fixture");
            let require_fixture = std::env::var("CERA_REQUIRE_MODEL").as_deref() == Ok("1")
                || std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
            if require_fixture
                || (python_has_deps && matches!(std::env::var("CI").as_deref(), Ok("true" | "1")))
            {
                panic!("test_bailingmoe3.gguf generation failed on CI runner despite python dependencies being present");
            }
            None
        })
        .clone()
}

#[test]
fn bailingmoe3_architecture_metadata_and_loading() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load bailingmoe3 model");
    let cfg = model.config();

    assert_eq!(cfg.architecture, "bailingmoe3");
    assert_eq!(cfg.n_layers, 4);
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.intermediate_size, 128);
    assert_eq!(cfg.n_heads, 4);
    assert_eq!(cfg.n_kv_heads, 1);
    assert_eq!(cfg.vocab_size, 261);
    assert_eq!(cfg.max_seq_len, 256);
    assert_eq!(
        cfg.block_types,
        vec![
            BlockType::DeltaNet,
            BlockType::DeltaNet,
            BlockType::DeltaNet,
            BlockType::Attention,
        ]
    );
    assert_eq!(cfg.kv_heads_per_layer, vec![0, 0, 0, 1]);

    let ssm = cfg.ssm.as_ref().expect("ssm config present");
    assert_eq!(ssm.d_conv, 4);
    assert_eq!(ssm.d_inner, 64);
    assert_eq!(ssm.d_state, 16);
    assert_eq!(ssm.dt_rank, 4);
    assert_eq!(ssm.n_group, 4);
}

#[test]
fn bailingmoe3_prefill_matches_sequential_decode() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load bailingmoe3 model");

    let tokens = [1u32, 65, 66, 67];

    // 1. Sequential decode
    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut seq_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        seq_logits = model.forward(&[tok], pos, &mut seq_state);
    }

    // 2. Prefill
    let mut prefill_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let prefill_logits = model.forward_prefill(&tokens, 0, &mut prefill_state);

    let sim = cosine_similarity(&seq_logits, &prefill_logits);
    assert!(
        sim > 0.9999,
        "prefill logits diverged from sequential decode: similarity {sim:.6}"
    );
}

#[test]
fn bailingmoe3_f16_kv_cache_parity() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load bailingmoe3 model");
    assert!(model.f16_kv_supported());

    let tokens = [1u32, 65, 66, 67];

    let mut f32_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let f32_logits = model.forward_prefill(&tokens, 0, &mut f32_state);

    let mut f16_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::F16).unwrap();
    let f16_logits = model.forward_prefill(&tokens, 0, &mut f16_state);

    let sim = cosine_similarity(&f32_logits, &f16_logits);
    assert!(
        sim > 0.999,
        "F16 KV cache logits diverged from F32: similarity {sim:.6}"
    );
}

#[test]
fn bailingmoe3_prefix_cache_roundtrip() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load bailingmoe3 model");

    let prefix = [1u32, 65, 66];
    let next_tok = 67u32;
    let full_tokens = [1u32, 65, 66, 67];

    // Run prefix of 3 tokens
    let mut ref_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let _ = model.forward_prefill(&prefix, 0, &mut ref_state);
    assert_eq!(ref_state.seq_len, prefix.len());

    // Save snapshot of prefix state
    let snap = ref_state.snapshot().expect("snapshot state");
    let mut cache = KvPrefixCache::new(
        KvCacheConfig::default(),
        model.config(),
        "cpu:test_bailingmoe3",
    );
    cache.insert(&prefix, snap);

    // Continue decode with next token on reference state
    let ref_logits = model.forward(&[next_tok], prefix.len(), &mut ref_state);
    assert_eq!(ref_state.seq_len, full_tokens.len());

    // Restore into fresh state
    let mut restored_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let (hit_snap, hit_len) = cache
        .find_longest_prefix(&full_tokens)
        .expect("cache match");
    assert_eq!(hit_len, prefix.len());
    restored_state.restore(&hit_snap);
    assert_eq!(restored_state.seq_len, prefix.len());

    let restored_logits = model.forward(&[next_tok], prefix.len(), &mut restored_state);
    assert_eq!(restored_state.seq_len, full_tokens.len());

    let sim = cosine_similarity(&ref_logits, &restored_logits);
    assert!(
        sim > 0.99999,
        "restored prefix cache logits diverged from reference: similarity {sim:.6}"
    );
}

#[test]
fn bailingmoe3_truncate_kv_and_zero_reset() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load bailingmoe3 model");

    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let tokens = [1u32, 65, 66, 67];
    let _ = model.forward_prefill(&tokens, 0, &mut state);
    assert_eq!(state.seq_len, 4);

    // Truncating a recurrent model rolls back to 0 to prevent state corruption
    model.truncate_kv(&mut state, 2);
    assert_eq!(state.seq_len, 0);

    for layer in 0..3 {
        let (conv, ssm) = state.deltanet_state(layer);
        assert!(conv.iter().all(|&x| x == 0.0));
        assert!(ssm.iter().all(|&x| x == 0.0));
    }
    match &state.layers[3] {
        LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } => {
            assert!(key_cache.is_empty());
            assert!(value_cache.is_empty());
        }
        _ => panic!("expected attention layer at index 3"),
    }
}

#[test]
fn bailingmoe3_tokenizer_roundtrip() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("load tokenizer");

    let text = "AB";
    let encoded = tokenizer.encode(text);
    assert_eq!(encoded, vec![260]);
    let decoded = tokenizer.decode(&encoded);
    assert_eq!(decoded, text);
}

#[test]
fn bailingmoe3_rejects_positive_gate_lower_bound() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.kda.gate_lower_bound".to_string(),
        cera::gguf::GgufValue::F32(1.0),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject non-negative gate lower bound");
}

#[test]
fn bailingmoe3_rejects_zero_block_count() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.block_count".to_string(),
        cera::gguf::GgufValue::U32(0),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject zero block count");
}

#[test]
fn bailingmoe3_rejects_zero_context_size() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.context_length".to_string(),
        cera::gguf::GgufValue::U32(0),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject zero context size");
}

#[test]
fn bailingmoe3_state_clear_and_snapshot_restore() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load bailingmoe3 model");

    let tokens = [1u32, 65, 66];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_step0 = model.forward(&[tokens[0]], 0, &mut state);
    let snapshot_step0 = state.snapshot().expect("state snapshot");

    // Advance state with more tokens
    let _logits_step1 = model.forward(&[tokens[1]], 1, &mut state);
    let logits_step2 = model.forward(&[tokens[2]], 2, &mut state);
    assert_eq!(state.seq_len, 3);

    // Restore back to step 0
    state.restore(&snapshot_step0);
    assert_eq!(state.seq_len, 1);

    // Re-run step 1 from restored state
    let _rerun_logits_step1 = model.forward(&[tokens[1]], 1, &mut state);
    let rerun_logits_step2 = model.forward(&[tokens[2]], 2, &mut state);
    assert_eq!(state.seq_len, 3);

    let sim = cosine_similarity(&logits_step2, &rerun_logits_step2);
    assert!(
        sim > 0.99999,
        "restored state decode diverged from original: cosine similarity {sim:.8}"
    );

    // Test clear_for_reuse()
    state.clear_for_reuse();
    assert_eq!(state.seq_len, 0);
    for layer in 0..3 {
        let (conv, ssm) = state.deltanet_state(layer);
        assert!(
            conv.iter().all(|&x| x == 0.0),
            "conv state should be cleared"
        );
        assert!(ssm.iter().all(|&x| x == 0.0), "ssm state should be cleared");
    }
    match &state.layers[3] {
        LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } => {
            assert!(key_cache.is_empty(), "key cache should be cleared");
            assert!(value_cache.is_empty(), "value cache should be cleared");
        }
        _ => panic!("expected attention layer at index 3"),
    }

    // After clear, decoding token 0 from position 0 must produce identical logits
    let cleared_logits = model.forward(&[tokens[0]], 0, &mut state);
    let clear_sim = cosine_similarity(&logits_step0, &cleared_logits);
    assert!(
        clear_sim > 0.99999,
        "cleared state decode diverged from initial run: cosine similarity {clear_sim:.8}"
    );
}

#[test]
fn bailingmoe3_oracle_dump_activations() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load bailingmoe3 model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let _logits = model.forward(&[1], 0, &mut state);
    let dumped = oracle_dump::take();

    let dump_names: Vec<String> = dumped.into_iter().map(|(name, _sum)| name).collect();
    assert!(
        dump_names.contains(&"embd".to_string()),
        "oracle dump must contain 'embd'"
    );
    assert!(
        dump_names.contains(&"l_out-0".to_string()),
        "oracle dump must contain 'l_out-0'"
    );
    assert!(
        dump_names.contains(&"l_out-1".to_string()),
        "oracle dump must contain 'l_out-1'"
    );
    assert!(
        dump_names.contains(&"l_out-2".to_string()),
        "oracle dump must contain 'l_out-2'"
    );
    assert!(
        dump_names.contains(&"l_out-3".to_string()),
        "oracle dump must contain 'l_out-3'"
    );
    assert!(
        dump_names.contains(&"result_norm".to_string()),
        "oracle dump must contain 'result_norm'"
    );
    assert!(
        dump_names.contains(&"result_output".to_string()),
        "oracle dump must contain 'result_output'"
    );
}

#[test]
fn bailingmoe3_rejects_zero_kda_head_dim() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.kda.head_dim".to_string(),
        cera::gguf::GgufValue::U32(0),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject zero kda head dim");
}

#[test]
fn bailingmoe3_rejects_zero_expert_count() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.expert_count".to_string(),
        cera::gguf::GgufValue::U32(0),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject zero expert count");
}

#[test]
fn bailingmoe3_rejects_zero_expert_used_count() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.expert_used_count".to_string(),
        cera::gguf::GgufValue::U32(0),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject zero expert used count");
}

#[test]
fn bailingmoe3_rejects_excessive_expert_used_count() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    gguf.metadata.insert(
        "bailingmoe3.expert_used_count".to_string(),
        cera::gguf::GgufValue::U32(999),
    );

    let res = BailingMoe3Model::from_gguf(gguf, 256);
    assert!(res.is_err(), "should reject excessive expert used count");
}

#[test]
fn bailingmoe3_alias_loading_bailingmoe_and_bailingmoe2() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    for alias in ["bailingmoe", "bailingmoe2", "bailingmoe3"] {
        let mut gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
        gguf.metadata.insert(
            "general.architecture".to_string(),
            cera::gguf::GgufValue::String(alias.to_string()),
        );
        let model = cera::model::load_model(gguf, Some(&path), 256);
        assert!(model.is_ok(), "load_model failed for alias {alias}");
        assert_eq!(model.unwrap().config().architecture, alias);
    }
}

#[test]
fn bailingmoe3_rejects_out_of_vocab_token_gracefully() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let oov_token = model.config().vocab_size as u32 + 100;
    let forward_logits = model.forward(&[oov_token], 0, &mut state);
    assert!(
        forward_logits.is_empty(),
        "out of vocab token must return empty logits without panicking"
    );

    let prefill_logits = model.forward_prefill(&[1, oov_token, 2], 0, &mut state);
    assert!(
        prefill_logits.is_empty(),
        "out of vocab token during prefill must return empty logits without panicking"
    );
}

#[test]
fn bailingmoe3_empty_token_slice_graceful_return() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let forward_logits = model.forward(&[], 0, &mut state);
    assert!(
        forward_logits.is_empty(),
        "empty token slice must return empty logits without panicking"
    );

    let prefill_logits = model.forward_prefill(&[], 0, &mut state);
    assert!(
        prefill_logits.is_empty(),
        "empty token slice in prefill must return empty logits without panicking"
    );
}

#[test]
fn bailingmoe3_moe_config_populated() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load model");
    let cfg = model.config();

    let moe = cfg.moe.as_ref().expect("moe config should be populated");
    assert_eq!(moe.n_expert, 4);
    assert_eq!(moe.n_expert_used, 2);
    assert_eq!(moe.expert_ff_len, 32);
    assert_eq!(moe.is_moe_layer.len(), 4);
    assert!(!moe.is_moe_layer[0], "layer 0 should be dense");
    assert!(moe.is_moe_layer[1], "layer 1 should be MoE");
    assert!(moe.is_moe_layer[2], "layer 2 should be MoE");
    assert!(moe.is_moe_layer[3], "layer 3 should be MoE");
}

#[test]
fn bailingmoe3_select_experts_non_finite_fallback() {
    let mut selected = Vec::new();

    // 1. NaN in probs: should produce finite normalized weights
    let probs = [f32::NAN, 0.5, 0.2, 0.1];
    let biases = [0.0, 0.0, 0.0, 0.0];
    cera::model::bailingmoe3::select_bailingmoe_experts(&probs, &biases, 2, 2.5, &mut selected);
    assert!(!selected.is_empty());
    for &(_, w) in &selected {
        assert!(w.is_finite(), "expert weights must be finite: got {w}");
    }

    // 2. Infinity in biases: should remain finite or zero
    let probs2 = [0.5, 0.4, 0.3, 0.2];
    let biases2 = [f32::INFINITY, 0.0, 0.0, 0.0];
    cera::model::bailingmoe3::select_bailingmoe_experts(&probs2, &biases2, 2, 2.5, &mut selected);
    assert!(!selected.is_empty());
    for &(_, w) in &selected {
        assert!(w.is_finite(), "expert weights must be finite: got {w}");
    }

    // 3. NaN in scaling factor: all weights zeroed
    cera::model::bailingmoe3::select_bailingmoe_experts(
        &probs2,
        &[0.0; 4],
        2,
        f32::NAN,
        &mut selected,
    );
    for &(_, w) in &selected {
        assert_eq!(w, 0.0, "weights must be zeroed when scale is NaN");
    }
}

#[test]
fn bailingmoe3_prefill_atomic_state_on_invalid_token() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let oov_token = model.config().vocab_size as u32 + 500;
    let tokens = [1u32, 65, oov_token, 66];
    let logits = model.forward_prefill(&tokens, 0, &mut state);
    assert!(
        logits.is_empty(),
        "must return empty logits on invalid token"
    );
    assert_eq!(
        state.seq_len, 0,
        "state.seq_len must not advance if prefill failed validation"
    );
}

#[test]
fn bailingmoe3_single_token_prefill_matches_forward() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load model");
    let mut state_forward =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut state_prefill =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_forward = model.forward(&[1], 0, &mut state_forward);
    let logits_prefill = model.forward_prefill(&[1], 0, &mut state_prefill);

    assert_eq!(logits_forward.len(), model.config().vocab_size);
    assert_eq!(logits_prefill.len(), model.config().vocab_size);

    let sim = cosine_similarity(&logits_forward, &logits_prefill);
    assert!(
        sim > 0.99999,
        "single-token prefill must match forward: sim = {sim}"
    );
    assert_eq!(state_forward.seq_len, 1);
    assert_eq!(state_prefill.seq_len, 1);
}

#[test]
fn bailingmoe3_select_experts_zero_count_boundary() {
    let mut selected = Vec::new();
    let probs = [0.2, 0.5, 0.1, 0.8];
    let bias = [0.0; 4];

    // n_used == 0
    cera::model::bailingmoe3::select_bailingmoe_experts(&probs, &bias, 0, 1.0, &mut selected);
    assert!(
        selected.is_empty(),
        "selected experts must be empty when n_used is 0"
    );

    // Empty probabilities slice
    cera::model::bailingmoe3::select_bailingmoe_experts(&[], &[], 2, 1.0, &mut selected);
    assert!(
        selected.is_empty(),
        "selected experts must be empty when probs is empty"
    );
}

#[test]
fn bailingmoe3_prefill_rejects_divergent_start_pos() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_bailingmoe3.gguf");
    let model = BailingMoe3Model::from_gguf(gguf, 256).expect("load model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let tokens = [1u32, 65];
    // state.seq_len is 0, but start_pos is 5
    let logits = model.forward_prefill(&tokens, 5, &mut state);
    assert!(
        logits.is_empty(),
        "must return empty logits when start_pos diverges from state.seq_len"
    );
    assert_eq!(
        state.seq_len, 0,
        "state.seq_len must remain untouched on sequence monotonicity failure"
    );
}

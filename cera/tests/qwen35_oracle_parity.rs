//! Oracle parity and functional verification tests for Qwen 3.5 / Ornith 1.0 (qwen35)
//! interleaved hybrid architecture.

#![cfg(feature = "mmap")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCacheConfig, KvCompression, KvPrefixCache, LayerState};
use cera::model::qwen35::Qwen35Model;
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
            let target_dir = manifest_dir.join("../target/tmp/cera_test_qwen35");
            let _ = std::fs::create_dir_all(&target_dir);
            let path = target_dir.join("test_qwen35.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }

            let script = manifest_dir.join("../scripts/oracle/create_qwen35_test_model.py");
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

            eprintln!("skipping: python3/python failed to generate test_qwen35.gguf fixture");
            let require_fixture = std::env::var("CERA_REQUIRE_MODEL").as_deref() == Ok("1")
                || std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
            if require_fixture
                || (python_has_deps && matches!(std::env::var("CI").as_deref(), Ok("true" | "1")))
            {
                panic!("test_qwen35.gguf generation failed on CI runner despite python dependencies being present");
            }
            None
        })
        .clone()
}

#[test]
fn qwen35_architecture_metadata_and_loading() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load qwen35 model");
    let cfg = model.config();

    assert_eq!(cfg.architecture, "qwen35");
    assert_eq!(cfg.n_layers, 4);
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.intermediate_size, 128);
    assert_eq!(cfg.n_heads, 4);
    assert_eq!(cfg.n_kv_heads, 2);
    assert_eq!(cfg.head_dim, 16);
    assert!(cfg.is_causal);

    // Verify interleaved layer types: layers 0, 1, 2 are DeltaNet, layer 3 is Attention
    assert_eq!(cfg.block_types.len(), 4);
    assert_eq!(cfg.block_types[0], BlockType::DeltaNet);
    assert_eq!(cfg.block_types[1], BlockType::DeltaNet);
    assert_eq!(cfg.block_types[2], BlockType::DeltaNet);
    assert_eq!(cfg.block_types[3], BlockType::Attention);

    assert_eq!(cfg.kv_heads_per_layer, vec![0, 0, 0, 2]);

    let ssm = cfg.ssm.as_ref().expect("ssm configuration present");
    assert_eq!(ssm.d_conv, 4);
    assert_eq!(ssm.d_inner, 64);
    assert_eq!(ssm.d_state, 16);
    assert_eq!(ssm.dt_rank, 4);
    assert_eq!(ssm.n_group, 2);
}

#[test]
fn qwen35_single_token_decode_finite_logits() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let token = 69u32;
    let logits = model.forward(&[token], 0, &mut state);

    assert_eq!(logits.len(), model.config().vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit at index {i} is not finite: {l}");
    }
    assert_eq!(state.seq_len, 1);

    // Verify DeltaNet recurrent states were updated and non-zero
    for layer in 0..3 {
        let (conv, ssm) = state.deltanet_state(layer);
        let conv_norm: f32 = conv.iter().map(|x| x * x).sum();
        let ssm_norm: f32 = ssm.iter().map(|x| x * x).sum();
        assert!(
            conv_norm > 0.0,
            "layer {layer} conv state should be non-zero after decode"
        );
        assert!(
            ssm_norm > 0.0,
            "layer {layer} ssm state should be non-zero after decode"
        );
    }

    // Layer 3 is Attention: verify attention KV cache has exactly 1 token
    match &state.layers[3] {
        LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } => {
            let kv_dim = model.config().n_kv_heads * model.config().head_dim;
            let k_slice = &key_cache[..kv_dim];
            let v_slice = &value_cache[..kv_dim];
            let k_norm: f32 = k_slice.iter().map(|x| x * x).sum();
            let v_norm: f32 = v_slice.iter().map(|x| x * x).sum();
            assert!(k_norm > 0.0, "layer 3 key cache should be non-zero");
            assert!(v_norm > 0.0, "layer 3 value cache should be non-zero");
        }
        _ => panic!("expected Attention layer at index 3"),
    }
}

#[test]
fn qwen35_prefill_matches_sequential_decode() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let tokens = [69u32, 112, 109, 45, 88];

    // 1. Sequential single-token decode
    let mut decode_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut last_decode_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        last_decode_logits = model.forward(&[tok], pos, &mut decode_state);
    }
    assert_eq!(decode_state.seq_len, tokens.len());

    // 2. Prefill forward
    let mut prefill_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let prefill_logits = model.forward_prefill(&tokens, 0, &mut prefill_state);
    assert_eq!(prefill_state.seq_len, tokens.len());

    // Compare logits
    assert_eq!(prefill_logits.len(), last_decode_logits.len());
    let sim = cosine_similarity(&prefill_logits, &last_decode_logits);
    assert!(
        sim > 0.99999,
        "prefill logits diverged from sequential decode: cosine similarity {sim:.8} < 0.99999"
    );

    let max_abs_diff = prefill_logits
        .iter()
        .zip(last_decode_logits.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff < 1e-4,
        "max absolute difference between prefill and decode logits ({max_abs_diff:.6e}) exceeds tolerance"
    );
}

#[test]
fn qwen35_state_clear_and_snapshot_restore() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let tokens = [69u32, 112, 109];
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

    // After clear, decoding token 0 from position 0 must produce identical logits
    let cleared_logits = model.forward(&[tokens[0]], 0, &mut state);
    let clear_sim = cosine_similarity(&logits_step0, &cleared_logits);
    assert!(
        clear_sim > 0.99999,
        "cleared state decode diverged from initial run: cosine similarity {clear_sim:.8}"
    );
}

#[test]
fn qwen35_oracle_dump_activations() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let _logits = model.forward(&[69], 0, &mut state);
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
fn qwen35_tokenizer_roundtrip() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("load tokenizer from qwen35 fixture");

    let text = "Hello world";
    let token_ids = tokenizer.encode(text);
    assert!(
        !token_ids.is_empty(),
        "encoded token IDs should not be empty"
    );

    let decoded = tokenizer.decode(&token_ids);
    assert_eq!(
        decoded, text,
        "roundtrip tokenizer decoding failed: expected {text:?}, got {decoded:?}"
    );
}

#[test]
fn qwen35_f16_kv_cache_parity() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let tokens = [69u32, 112, 109, 45];

    // 1. Full precision (F32)
    let mut state_f32 =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits_f32 = model.forward_prefill(&tokens, 0, &mut state_f32);

    // 2. Half precision (F16)
    let mut state_f16 =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::F16).unwrap();
    let logits_f16 = model.forward_prefill(&tokens, 0, &mut state_f16);

    let sim = cosine_similarity(&logits_f32, &logits_f16);
    assert!(
        sim > 0.999,
        "F16 KV cache prefill diverged from F32: cosine similarity {sim:.8} < 0.999"
    );

    // Decode subsequent token
    let next_tok = 88u32;
    let next_f32 = model.forward(&[next_tok], tokens.len(), &mut state_f32);
    let next_f16 = model.forward(&[next_tok], tokens.len(), &mut state_f16);

    let next_sim = cosine_similarity(&next_f32, &next_f16);
    assert!(
        next_sim > 0.999,
        "F16 KV cache decode diverged from F32: cosine similarity {next_sim:.8} < 0.999"
    );
}

#[test]
fn qwen35_truncate_kv_and_zero_reset() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let tokens = [69u32, 112, 109];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let initial_logits = model.forward(&[tokens[0]], 0, &mut state);
    let _ = model.forward(&[tokens[1]], 1, &mut state);
    let _ = model.forward(&[tokens[2]], 2, &mut state);
    assert_eq!(state.seq_len, 3);

    // Truncate to length 2: because recurrent DeltaNet layers lack intermediate
    // history without snapshots, truncate_to safely falls back to safe_len = 0.
    model.truncate_kv(&mut state, 2);
    assert_eq!(state.seq_len, 0);

    for layer in 0..3 {
        let (conv, ssm) = state.deltanet_state(layer);
        assert!(
            conv.iter().all(|&x| x == 0.0),
            "layer {layer} conv state should be 0.0 after truncate_kv fallback"
        );
        assert!(
            ssm.iter().all(|&x| x == 0.0),
            "layer {layer} ssm state should be 0.0 after truncate_kv fallback"
        );
    }

    // Decoding token 0 from position 0 must match initial decode
    let rerun_logits = model.forward(&[tokens[0]], 0, &mut state);
    let sim = cosine_similarity(&initial_logits, &rerun_logits);
    assert!(
        sim > 0.99999,
        "decode after truncate_kv fallback diverged: cosine similarity {sim:.8}"
    );

    // Explicit truncate to length 0
    let _ = model.forward(&[tokens[1]], 1, &mut state);
    model.truncate_kv(&mut state, 0);
    assert_eq!(state.seq_len, 0);
    for layer in 0..3 {
        let (conv, ssm) = state.deltanet_state(layer);
        assert!(conv.iter().all(|&x| x == 0.0));
        assert!(ssm.iter().all(|&x| x == 0.0));
    }
}

#[test]
fn qwen35_prefix_cache_roundtrip() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let tokens = [69u32, 112, 109, 45];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    // Run prefix of 3 tokens
    let _ = model.forward_prefill(&tokens[..3], 0, &mut state);
    assert_eq!(state.seq_len, 3);

    let snap = state.snapshot().expect("snapshot state");
    let mut cache = KvPrefixCache::new(KvCacheConfig::default(), model.config(), "cpu:test_qwen35");
    cache.insert(&tokens[..3], snap);

    // Look up prefix for sequence of 4 tokens
    let (hit_snap, hit_len) = cache
        .find_longest_prefix(&tokens[..4])
        .expect("cache prefix hit");
    assert_eq!(hit_len, 3);

    let mut restored_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    restored_state.restore(&hit_snap);
    assert_eq!(restored_state.seq_len, 3);

    // Continue decode on original and restored states
    let orig_next = model.forward(&[tokens[3]], 3, &mut state);
    let restored_next = model.forward(&[tokens[3]], 3, &mut restored_state);

    let sim = cosine_similarity(&orig_next, &restored_next);
    assert!(
        sim > 0.99999,
        "prefix cache restored decode diverged: cosine similarity {sim:.8}"
    );
}

#[test]
fn qwen35_rejects_zero_full_attention_interval() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    gguf.metadata.insert(
        "qwen35.full_attention_interval".to_string(),
        GgufValue::U32(0),
    );
    let res = Qwen35Model::from_gguf(gguf, 256);
    let err_msg = match res {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected error for full_attention_interval = 0"),
    };
    assert!(
        err_msg.contains("full_attention_interval must be > 0"),
        "expected error message about full_attention_interval > 0, got: {err_msg}"
    );
}

#[test]
fn qwen35_ffn_norm_alias_loading() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    if let Some(tensor) = gguf.tensors.remove("blk.0.attn_post_norm.weight") {
        gguf.tensors
            .insert("blk.0.ffn_norm.weight".to_string(), tensor);
    }
    let model = Qwen35Model::from_gguf(gguf, 256);
    assert!(
        model.is_ok(),
        "model should load with ffn_norm.weight alias"
    );
}

#[test]
fn qwen35_rejects_truncated_recurrent_layers() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    // block_count is 4, but provide only 2 booleans
    gguf.metadata.insert(
        "qwen35.attention.recurrent_layers".to_string(),
        GgufValue::Array(vec![GgufValue::Bool(true), GgufValue::Bool(false)]),
    );
    let res = Qwen35Model::from_gguf(gguf, 256);
    let err_msg = match res {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected error for truncated recurrent_layers"),
    };
    assert!(
        err_msg.contains("must be >= block_count"),
        "expected error message about recurrent_layers length >= block_count, got: {err_msg}"
    );
}

#[test]
fn qwen35_rejects_non_divisible_gqa_heads() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    gguf.metadata
        .insert("qwen35.attention.head_count".to_string(), GgufValue::U32(7));
    gguf.metadata.insert(
        "qwen35.attention.head_count_kv".to_string(),
        GgufValue::U32(2),
    );
    let res = Qwen35Model::from_gguf(gguf, 256);
    let err_msg = match res {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected error for non-divisible GQA heads"),
    };
    assert!(
        err_msg.contains("must be a multiple of head_count_kv"),
        "expected error message about head_count multiple of head_count_kv, got: {err_msg}"
    );
}

#[test]
fn qwen35_rejects_zero_context_size() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let res = Qwen35Model::from_gguf(gguf, 0);
    let err_msg = match res {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected error for context_size = 0"),
    };
    assert!(
        err_msg.contains("context_size must be > 0"),
        "expected error message about context_size > 0, got: {err_msg}"
    );
}

#[test]
fn qwen35_rejects_odd_head_dim() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    gguf.metadata.insert(
        "qwen35.attention.key_length".to_string(),
        GgufValue::U32(15),
    );
    let res = Qwen35Model::from_gguf(gguf, 256);
    let err_msg = match res {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected error for odd head_dim"),
    };
    assert!(
        err_msg.contains("must be an even integer for RoPE rotation"),
        "expected error message about even head_dim, got: {err_msg}"
    );
}

#[test]
fn qwen35_rejects_zero_head_count() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    gguf.metadata
        .insert("qwen35.attention.head_count".to_string(), GgufValue::U32(0));
    gguf.metadata.remove("qwen35.attention.key_length");
    let res = Qwen35Model::from_gguf(gguf, 256);
    let err_msg = match res {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("expected error for head_count = 0"),
    };
    assert!(
        err_msg.contains("head_count must be > 0"),
        "expected error message about head_count > 0, got: {err_msg}"
    );
}

#[test]
fn qwen35_missing_inner_size_metadata_fallback() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    gguf.metadata.remove("qwen35.ssm.inner_size");
    let model = Qwen35Model::from_gguf(gguf, 256);
    assert!(
        model.is_ok(),
        "model should load successfully when ssm.inner_size is omitted"
    );
    let model = model.unwrap();
    let ssm = model.config().ssm.as_ref().expect("ssm config");
    assert_eq!(
        ssm.d_inner,
        ssm.dt_rank * ssm.d_state,
        "default inner_size should match dt_rank * d_state"
    );
}

#[test]
fn qwen35_alias_loading_qwen3_5_and_dot() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    for alias in ["qwen3_5", "qwen3.5"] {
        let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
        gguf.metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String(alias.to_string()),
        );
        let model = Qwen35Model::from_gguf(gguf, 256);
        assert!(
            model.is_ok(),
            "model should load successfully with architecture alias '{alias}'"
        );
    }
}

#[test]
fn qwen35_ssm_output_weight_alias_loading() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    // Swap blk.0.ssm_out.weight with blk.0.ssm_output.weight
    if let Some(tensor) = gguf.tensors.remove("blk.0.ssm_out.weight") {
        gguf.tensors
            .insert("blk.0.ssm_output.weight".to_string(), tensor);
    }
    let model = Qwen35Model::from_gguf(gguf, 256);
    assert!(
        model.is_ok(),
        "model should load successfully with ssm_output.weight alias"
    );
}

#[test]
fn qwen35_load_model_dispatches_for_all_aliases() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    for alias in ["qwen35", "qwen3_5", "qwen3.5"] {
        let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
        gguf.metadata.insert(
            "general.architecture".to_string(),
            cera::gguf::GgufValue::String(alias.to_string()),
        );
        let model = cera::model::load_model(gguf, None, 256);
        assert!(
            model.is_ok(),
            "load_model should dispatch successfully for arch '{alias}'"
        );
    }
}

#[test]
fn qwen35_attn_output_weight_alias_loading() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    // Swap blk.3.attn_output.weight with blk.3.attn_out.weight
    if let Some(tensor) = gguf.tensors.remove("blk.3.attn_output.weight") {
        gguf.tensors
            .insert("blk.3.attn_out.weight".to_string(), tensor);
    }
    let model = Qwen35Model::from_gguf(gguf, 256);
    assert!(
        model.is_ok(),
        "model should load successfully with attn_out.weight alias"
    );
}

#[test]
fn qwen35_oversized_snapshot_restore_truncates_to_exact_dimension() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let _logits = model.forward(&[69], 0, &mut state);
    let mut snap = state.snapshot().expect("state snapshot");

    let (orig_conv_len, orig_ssm_len) = {
        let (c, s) = state.deltanet_state(0);
        (c.len(), s.len())
    };

    // Make snapshot layer 0 oversized with extra float bytes
    if let cera::kv_cache::LayerSnapshot::DeltaNet {
        ref mut conv_state,
        ref mut ssm_state,
    } = snap.layers[0]
    {
        conv_state.extend_from_slice(&[0u8; 64]);
        ssm_state.extend_from_slice(&[0u8; 64]);
    }

    state.restore(&snap);

    let (c_restored, s_restored) = state.deltanet_state(0);
    assert_eq!(
        c_restored.len(),
        orig_conv_len,
        "oversized snapshot conv_state must be truncated to expected dimension"
    );
    assert_eq!(
        s_restored.len(),
        orig_ssm_len,
        "oversized snapshot ssm_state must be truncated to expected dimension"
    );
}

#[test]
fn qwen35_layer_early_return_zero_residual_delta() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    // Corrupt layer 0's conv_state to be shorter than expected
    if let cera::kv_cache::LayerState::DeltaNet {
        ref mut conv_state, ..
    } = state.layers[0]
    {
        conv_state.truncate(1);
    }

    // Decoding should safely early return on layer 0 without panic or propagating corrupted residual
    let logits = model.forward(&[69], 0, &mut state);
    assert_eq!(logits.len(), model.config().vocab_size);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn qwen35_f16_kv_supported_reports_true() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");
    assert!(
        model.f16_kv_supported(),
        "Qwen35Model must report f16_kv_supported = true"
    );
}

#[test]
fn qwen35_undersized_snapshot_restore_pads_to_exact_dimension() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let _logits = model.forward(&[69], 0, &mut state);
    let mut snap = state.snapshot().expect("state snapshot");

    let (orig_conv_len, orig_ssm_len) = {
        let (c, s) = state.deltanet_state(0);
        (c.len(), s.len())
    };

    // Truncate snapshot layer 0's conv_state and ssm_state to be undersized
    if let cera::kv_cache::LayerSnapshot::DeltaNet {
        ref mut conv_state,
        ref mut ssm_state,
    } = snap.layers[0]
    {
        conv_state.truncate(conv_state.len() / 2);
        let aligned_conv = (conv_state.len() / 4) * 4;
        conv_state.truncate(aligned_conv);

        ssm_state.truncate(ssm_state.len() / 2);
        let aligned_ssm = (ssm_state.len() / 4) * 4;
        ssm_state.truncate(aligned_ssm);
    }

    state.restore(&snap);

    let (c_restored, s_restored) = state.deltanet_state(0);
    assert_eq!(
        c_restored.len(),
        orig_conv_len,
        "undersized snapshot conv_state must be padded to expected dimension"
    );
    assert_eq!(
        s_restored.len(),
        orig_ssm_len,
        "undersized snapshot ssm_state must be padded to expected dimension"
    );
}

#[test]
fn qwen35_prefix_cache_roundtrip_empty_conv_state() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let prefix_tokens = [42u32, 43, 44];
    let query_tokens = [42u32, 43, 44, 45];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let _ = model.forward_prefill(&prefix_tokens, 0, &mut state);

    let mut snap = state.snapshot().expect("snapshot state");
    // Clear conv_state to 0 bytes (simulating d_conv = 1)
    if let cera::kv_cache::LayerSnapshot::DeltaNet {
        ref mut conv_state, ..
    } = snap.layers[0]
    {
        conv_state.clear();
    }

    let tempdir = tempfile::tempdir().expect("tempdir");
    let config = KvCacheConfig {
        cache_dir: Some(tempdir.path().to_path_buf()),
        ..Default::default()
    };
    let mut cache = KvPrefixCache::new(config, model.config(), "cpu:test_qwen35_empty_conv");
    cache.insert(&prefix_tokens, snap);

    let (hit_snap, hit_len) = cache
        .find_longest_prefix(&query_tokens)
        .expect("cache prefix hit for empty conv state");
    assert_eq!(hit_len, prefix_tokens.len());

    let mut restored_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    restored_state.restore(&hit_snap);
    assert_eq!(restored_state.seq_len, prefix_tokens.len());
}

#[test]
fn qwen35_forward_rejects_divergent_pos() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    // state.seq_len is 0, passing pos = 5 should panic
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.forward(&[42], 5, &mut state);
    }));
    assert!(
        res.is_err(),
        "forward with divergent pos must panic with assertion"
    );
}

#[test]
fn qwen35_decay_factor_bounded_no_explosion() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    // Decode 10 steps and ensure logits and state norms remain strictly finite and bounded
    for step in 0..10 {
        let logits = model.forward(&[42 + (step as u32 % 5)], step, &mut state);
        assert_eq!(logits.len(), model.config().vocab_size);
        for &val in &logits {
            assert!(val.is_finite(), "logits must remain finite at step {step}");
            assert!(val.abs() < 1e6, "logits exploded at step {step}: {val}");
        }
    }

    for (l_idx, layer) in state.layers.iter().enumerate() {
        if let LayerState::DeltaNet { ssm_state, .. } = layer {
            for &val in ssm_state {
                assert!(val.is_finite(), "ssm_state in layer {l_idx} is not finite");
                assert!(
                    val.abs() < 1e6,
                    "ssm_state in layer {l_idx} exploded: {val}"
                );
            }
        }
    }
}

#[test]
fn qwen35_multi_turn_truncate_and_prefill_continuation() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_qwen35.gguf");
    let model = Qwen35Model::from_gguf(gguf, 256).expect("load Qwen35Model");

    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    // Step 1: decode 5 tokens
    for i in 0..5 {
        let _ = model.forward(&[10 + i], i as usize, &mut state);
    }
    assert_eq!(state.seq_len, 5);

    // Step 2: truncate to 0
    model.truncate_kv(&mut state, 0);
    assert_eq!(state.seq_len, 0);

    // Step 3: prefill fresh sequence
    let test_prompt = [42u32, 43, 44, 45];
    let truncated_logits = model.forward_prefill(&test_prompt, 0, &mut state);

    // Compare with clean fresh state
    let mut fresh_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let fresh_logits = model.forward_prefill(&test_prompt, 0, &mut fresh_state);

    let sim = cosine_similarity(&truncated_logits, &fresh_logits);
    assert!(
        sim > 0.99999,
        "cosine similarity between truncated-and-reprefilled and fresh state should be > 0.99999, got {sim}"
    );
}

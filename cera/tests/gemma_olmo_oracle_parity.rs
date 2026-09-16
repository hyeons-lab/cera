//! Cross-validation test comparing Gemma 2 and Olmo 2 dense architectures
//! in Cera against upstream llama.cpp oracle outputs.

#![cfg(feature = "mmap")]

use std::collections::HashMap;
use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::Model;
use cera::model::llama::LlamaModel;
use cera::model::transformer::oracle_dump;

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / (a.abs() + b.abs() + 1e-9)
}

fn find_model_path(env_var: &str, default_filename: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        let path = PathBuf::from(d).join(default_filename);
        if path.exists() {
            return Some(path);
        }
    }
    let target_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/oracle/models")
        .join(default_filename);
    if target_path.exists() {
        return Some(target_path);
    }
    let root_target_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../target/oracle/models")
        .join(default_filename);
    if root_target_path.exists() {
        return Some(root_target_path);
    }
    let tmp_path = PathBuf::from("/tmp").join(default_filename);
    if tmp_path.exists() {
        return Some(tmp_path);
    }
    let sys_tmp = std::env::temp_dir().join(default_filename);
    if sys_tmp.exists() {
        return Some(sys_tmp);
    }
    None
}

fn gemma2_path() -> Option<PathBuf> {
    find_model_path("CERA_TEST_GEMMA2_GGUF", "test_gemma2.gguf")
}

fn olmo2_path() -> Option<PathBuf> {
    find_model_path("CERA_TEST_OLMO2_GGUF", "test_olmo2.gguf")
}

#[test]
fn gemma2_matches_llama_cpp_oracle() {
    let Some(path) = gemma2_path() else {
        eprintln!(
            "skipping: test_gemma2.gguf not found in CERA_TEST_GEMMA2_GGUF, CERA_ORACLE_MODELS_DIR, target/oracle/models, or /tmp"
        );
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_gemma2.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load gemma2 model");

    // Token IDs in test GGUF fixture (4 special tokens + ASCII byte value): 'T' (84 + 4 = 88), 'h' (104 + 4 = 108), 'e' (101 + 4 = 105).
    let tokens = vec![88u32, 108, 105];
    let mut dump_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let _logits_dump = model.forward_prefill(&tokens, 0, &mut dump_state);
    let occ = oracle_dump::take();

    let mut cera_sums: HashMap<String, f64> = HashMap::new();
    let last = model.config().n_layers - 1;
    for (name, sum) in occ {
        if name.starts_with("result_") || name.ends_with(&format!("-{last}")) {
            cera_sums.insert(name, sum);
        } else {
            *cera_sums.entry(name).or_insert(0.0) += sum;
        }
    }

    // Reference sums from llama.cpp llama-eval-callback on test_gemma2.gguf:
    // inp_scaled: 1.266444
    // l_out-0: -13.983496
    // l_out-1: -15.288410
    // result_norm: -7.945115
    // result_output: -1.675211
    let expected = [
        ("embd", 1.266444),
        ("l_out-0", -13.983496),
        ("l_out-1", -15.288410),
        ("result_norm", -7.945115),
        ("result_output", -1.675211),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node}"));
        let diff = rel_diff(got, exp);
        eprintln!("[gemma2] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.01,
            "gemma2 divergence at {node}: cera={got}, llama={exp}, rel_diff={diff}"
        );
    }

    // Verify batched prefill matches sequential path (executed outside oracle_dump to engage batched kernel)
    let mut batched_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits_batched = model.forward_prefill(&tokens, 0, &mut batched_state);

    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_seq = model.forward(&[t], i, &mut seq_state);
    }

    assert_eq!(logits_batched.len(), logits_seq.len());
    let dot: f32 = logits_batched
        .iter()
        .zip(&logits_seq)
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = logits_batched.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = logits_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[gemma2] batched vs sequential prefill cosine: {cos:.6}");
    assert!(cos > 0.9999, "gemma2 prefill cosine too low: {cos}");

    // Verify decode step after batched prefill matches decode step after sequential prefill
    let next_token = logits_batched
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    let decode_from_batched = model.forward(&[next_token], tokens.len(), &mut batched_state);
    let decode_from_seq = model.forward(&[next_token], tokens.len(), &mut seq_state);

    let d_dot: f32 = decode_from_batched
        .iter()
        .zip(&decode_from_seq)
        .map(|(a, b)| a * b)
        .sum();
    let d_na: f32 = decode_from_batched
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let d_nb: f32 = decode_from_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let d_cos = d_dot / (d_na * d_nb);
    eprintln!("[gemma2] decode after batched vs sequential cosine: {d_cos:.6}");
    assert!(d_cos > 0.9999, "gemma2 decode cosine too low: {d_cos}");
}

#[test]
fn olmo2_matches_llama_cpp_oracle() {
    let Some(path) = olmo2_path() else {
        eprintln!(
            "skipping: test_olmo2.gguf not found in CERA_TEST_OLMO2_GGUF, CERA_ORACLE_MODELS_DIR, target/oracle/models, or /tmp"
        );
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_olmo2.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load olmo2 model");

    // Token IDs in test GGUF fixture (4 special tokens + ASCII byte value): 'T' (84 + 4 = 88), 'h' (104 + 4 = 108), 'e' (101 + 4 = 105).
    let tokens = vec![88u32, 108, 105];
    let mut dump_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let _logits_dump = model.forward_prefill(&tokens, 0, &mut dump_state);
    let occ = oracle_dump::take();

    let mut cera_sums: HashMap<String, f64> = HashMap::new();
    let last = model.config().n_layers - 1;
    for (name, sum) in occ {
        if name.starts_with("result_") || name.ends_with(&format!("-{last}")) {
            cera_sums.insert(name, sum);
        } else {
            *cera_sums.entry(name).or_insert(0.0) += sum;
        }
    }

    // Reference sums from llama.cpp llama-eval-callback on test_olmo2.gguf:
    // embd: 0.158306
    // l_out-0: 3.747856
    // l_out-1: 11.481936
    // result_norm: 6.047365
    // result_output: 3.290789
    let expected = [
        ("embd", 0.158306),
        ("l_out-0", 3.747856),
        ("l_out-1", 11.481936),
        ("result_norm", 6.047365),
        ("result_output", 3.290789),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node}"));
        let diff = rel_diff(got, exp);
        eprintln!("[olmo2] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.01,
            "olmo2 divergence at {node}: cera={got}, llama={exp}, rel_diff={diff}"
        );
    }

    // Verify batched prefill matches sequential path (executed outside oracle_dump to engage batched kernel)
    let mut batched_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits_batched = model.forward_prefill(&tokens, 0, &mut batched_state);

    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_seq = model.forward(&[t], i, &mut seq_state);
    }

    assert_eq!(logits_batched.len(), logits_seq.len());
    let dot: f32 = logits_batched
        .iter()
        .zip(&logits_seq)
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = logits_batched.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = logits_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[olmo2] batched vs sequential prefill cosine: {cos:.6}");
    assert!(cos > 0.9999, "olmo2 prefill cosine too low: {cos}");

    // Verify decode step after batched prefill matches decode step after sequential prefill
    let next_token = logits_batched
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    let decode_from_batched = model.forward(&[next_token], tokens.len(), &mut batched_state);
    let decode_from_seq = model.forward(&[next_token], tokens.len(), &mut seq_state);

    let d_dot: f32 = decode_from_batched
        .iter()
        .zip(&decode_from_seq)
        .map(|(a, b)| a * b)
        .sum();
    let d_na: f32 = decode_from_batched
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let d_nb: f32 = decode_from_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let d_cos = d_dot / (d_na * d_nb);
    eprintln!("[olmo2] decode after batched vs sequential cosine: {d_cos:.6}");
    assert!(d_cos > 0.9999, "olmo2 decode cosine too low: {d_cos}");
}

#[test]
fn gemma2_long_prefill_bypasses_flash_attn_and_matches_decode() {
    let Some(path) = gemma2_path() else {
        eprintln!(
            "skipping: test_gemma2.gguf not found in CERA_TEST_GEMMA2_GGUF, CERA_ORACLE_MODELS_DIR, target/oracle/models, or /tmp"
        );
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_gemma2.gguf");
    // Size cache to 512 tokens so 280 tokens fit (> FLASH_ATTN_THRESHOLD = 256).
    let model = LlamaModel::from_gguf(gguf, 512).expect("load gemma2 model");

    // Prompt with 280 tokens (> 256 threshold where Flash Attention normally activates,
    // but Flash Attention is bypassed because attn_logit_softcapping is present).
    let tokens: Vec<u32> = (0..280).map(|i| 88 + (i % 20) as u32).collect();
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
    assert_eq!(logits_prefill.len(), model.config().vocab_size);
    assert!(logits_prefill.iter().all(|v| v.is_finite()));
    assert_eq!(state.seq_len, 280);

    // Verify batched execution matches sequential execution
    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_seq = model.forward(&[t], i, &mut seq_state);
    }
    assert_eq!(logits_prefill.len(), logits_seq.len());
    let dot: f32 = logits_prefill
        .iter()
        .zip(&logits_seq)
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = logits_prefill.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = logits_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[gemma2] batched vs sequential long prefill cosine: {cos:.6}");
    assert!(
        cos > 0.999,
        "gemma2 batched vs sequential long prefill cosine too low: {cos}"
    );

    // Verify a subsequent decode step succeeds.
    let next_logits = model.forward(&[88], 280, &mut state);
    assert_eq!(next_logits.len(), model.config().vocab_size);
    assert!(next_logits.iter().all(|v| v.is_finite()));
    assert_eq!(state.seq_len, 281);
}

#[test]
fn olmo2_long_prefill_engages_flash_attn_and_matches_decode() {
    let Some(path) = olmo2_path() else {
        eprintln!(
            "skipping: test_olmo2.gguf not found in CERA_TEST_OLMO2_GGUF, CERA_ORACLE_MODELS_DIR, target/oracle/models, or /tmp"
        );
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_olmo2.gguf");
    // Size cache to 512 tokens so 280 tokens fit (> FLASH_ATTN_THRESHOLD = 256).
    let model = LlamaModel::from_gguf(gguf, 512).expect("load olmo2 model");

    // Prompt with 280 tokens (> 256 threshold where Flash Attention engages
    // because Olmo 2 does not use attention logit soft-capping).
    let tokens: Vec<u32> = (0..280).map(|i| 88 + (i % 20) as u32).collect();
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
    assert_eq!(logits_prefill.len(), model.config().vocab_size);
    assert!(logits_prefill.iter().all(|v| v.is_finite()));
    assert_eq!(state.seq_len, 280);

    // Verify Flash Attention output matches sequential execution
    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_seq = model.forward(&[t], i, &mut seq_state);
    }
    assert_eq!(logits_prefill.len(), logits_seq.len());
    let dot: f32 = logits_prefill
        .iter()
        .zip(&logits_seq)
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = logits_prefill.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = logits_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[olmo2] flash vs sequential prefill cosine: {cos:.6}");
    assert!(
        cos > 0.999,
        "olmo2 flash vs sequential cosine too low: {cos}"
    );

    // Verify a subsequent decode step succeeds.
    let next_logits = model.forward(&[88], 280, &mut state);
    assert_eq!(next_logits.len(), model.config().vocab_size);
    assert!(next_logits.iter().all(|v| v.is_finite()));
    assert_eq!(state.seq_len, 281);
}

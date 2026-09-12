//! Cross-validation test comparing Gemma 2 and Olmo 2 dense architectures
//! in Cera against upstream llama.cpp oracle outputs.

#![cfg(feature = "mmap")]

use std::collections::HashMap;
use std::path::Path;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::Model;
use cera::model::llama::LlamaModel;
use cera::model::transformer::oracle_dump;

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / (a.abs() + b.abs() + 1e-9)
}

#[test]
fn gemma2_matches_llama_cpp_oracle() {
    let path = Path::new("/tmp/test_gemma2.gguf");
    if !path.exists() {
        eprintln!("skipping: /tmp/test_gemma2.gguf does not exist");
        return;
    }

    let gguf = GgufFile::open(path).expect("open test_gemma2.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load gemma2 model");

    // Token IDs in test GGUF fixture (4 special tokens + ASCII byte value): 'T' (84 + 4 = 88), 'h' (104 + 4 = 108), 'e' (101 + 4 = 105).
    let tokens = vec![88u32, 108, 105];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
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

    // Verify batched prefill matches sequential path
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
    eprintln!("[gemma2] batched vs sequential prefill cosine: {cos:.6}");
    assert!(cos > 0.9999, "gemma2 prefill cosine too low: {cos}");

    // Verify decode step after batched prefill matches decode step after sequential prefill
    let next_token = logits_prefill
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    let decode_from_batched = model.forward(&[next_token], tokens.len(), &mut state);
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
    let path = Path::new("/tmp/test_olmo2.gguf");
    if !path.exists() {
        eprintln!("skipping: /tmp/test_olmo2.gguf does not exist");
        return;
    }

    let gguf = GgufFile::open(path).expect("open test_olmo2.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load olmo2 model");

    // Token IDs in test GGUF fixture (4 special tokens + ASCII byte value): 'T' (84 + 4 = 88), 'h' (104 + 4 = 108), 'e' (101 + 4 = 105).
    let tokens = vec![88u32, 108, 105];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
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

    // Verify batched prefill matches sequential path
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
    eprintln!("[olmo2] batched vs sequential prefill cosine: {cos:.6}");
    assert!(cos > 0.9999, "olmo2 prefill cosine too low: {cos}");

    // Verify decode step after batched prefill matches decode step after sequential prefill
    let next_token = logits_prefill
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    let decode_from_batched = model.forward(&[next_token], tokens.len(), &mut state);
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

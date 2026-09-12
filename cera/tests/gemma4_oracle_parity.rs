//! Cross-validation test comparing Gemma 4 dense architecture
//! in Cera against upstream llama.cpp oracle outputs.

#![cfg(feature = "mmap")]

use std::collections::HashMap;
use std::path::Path;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression, LayerState};
use cera::model::transformer::oracle_dump;

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / (a.abs() + b.abs() + 1e-9)
}

fn get_test_model_path() -> std::path::PathBuf {
    std::env::temp_dir().join("test_gemma4.gguf")
}

fn ensure_test_model(path: &Path) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if !path.exists() {
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let script_path = if manifest_dir
                .join("../scripts/oracle/create_gemma4_test_model.py")
                .exists()
            {
                manifest_dir.join("../scripts/oracle/create_gemma4_test_model.py")
            } else {
                manifest_dir.join("scripts/oracle/create_gemma4_test_model.py")
            };
            eprintln!("generating {:?} via {:?}", path, script_path);
            let status = std::process::Command::new("python3")
                .arg(&script_path)
                .arg(path)
                .status()
                .expect("execute create_gemma4_test_model.py");
            assert!(status.success(), "failed to generate {:?}", path);
        }
    });
}

#[test]
fn gemma4_matches_llama_cpp_oracle() {
    let path = get_test_model_path();
    ensure_test_model(&path);

    let gguf = GgufFile::open(&path).expect("open test_gemma4.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load gemma4 model");

    // Token IDs in test GGUF fixture (4 special tokens + ASCII byte value):
    // 'T' (84 + 4 = 88), 'h' (104 + 4 = 108), 'e' (101 + 4 = 105).
    let tokens = vec![88u32, 108, 105];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
    let occ = oracle_dump::take();

    let mut cera_sums: HashMap<String, f64> = HashMap::new();
    for (name, sum) in occ {
        if name.starts_with("result_") {
            // For result_norm and result_output, keep the final token value.
            cera_sums.insert(name, sum);
        } else {
            // For per-layer activations across prefill tokens, accumulate the total sum.
            *cera_sums.entry(name).or_insert(0.0) += sum;
        }
    }

    // Reference node sums from upstream llama-eval-callback on /tmp/test_gemma4.gguf:
    // embd: 0.158305
    // inp_scaled: 1.266444
    // l_out-0: 16.341734
    // l_out-1: -4.514472
    // l_out-2: -34.523163
    // l_out-3: -48.512215
    // result_norm: -2.891024
    // result_output: 8.814717
    let expected = [
        ("embd", 0.158305),
        ("inp_scaled", 1.266444),
        ("l_out-0", 16.341734),
        ("l_out-1", -4.514472),
        ("l_out-2", -34.523163),
        ("l_out-3", -48.512215),
        ("result_norm", -2.891024),
        ("result_output", 8.814717),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node} in cera oracle dump"));
        let diff = rel_diff(got, exp);
        eprintln!("[gemma4] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.005,
            "gemma4 divergence at {node}: cera={got}, llama={exp}, rel_diff={diff}"
        );
    }

    // Verify cross-layer KV cache invariants:
    // Layers 0..3: layers 0, 1, 2 own KV caches and hold 3 tokens of KV entries.
    // Layer 3 is a shared KV layer and must have 0 KV entries in its own slot.
    let head_dim = model.config().head_dim;
    let kv_dim = model.config().n_kv_heads * head_dim;
    let expected_owned_entries = tokens.len() * kv_dim;

    for (layer_idx, layer_state) in state.layers.iter().enumerate() {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = layer_state
        {
            if layer_idx < 3 {
                assert_eq!(
                    key_cache.len(),
                    expected_owned_entries,
                    "layer {layer_idx} expected {expected_owned_entries} key entries, got {}",
                    key_cache.len()
                );
                assert_eq!(
                    value_cache.len(),
                    expected_owned_entries,
                    "layer {layer_idx} expected {expected_owned_entries} value entries, got {}",
                    value_cache.len()
                );
            } else {
                assert_eq!(
                    key_cache.len(),
                    0,
                    "shared layer {layer_idx} should have empty owned key cache, got {}",
                    key_cache.len()
                );
                assert_eq!(
                    value_cache.len(),
                    0,
                    "shared layer {layer_idx} should have empty owned value cache, got {}",
                    value_cache.len()
                );
            }
        } else {
            panic!("expected Attention layer state for layer {layer_idx}");
        }
    }

    // Verify sequential forward produces identical output to forward_prefill.
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
    eprintln!("[gemma4] prefill vs sequential cosine: {cos:.6}");
    assert!(cos > 0.99999, "gemma4 prefill cosine too low: {cos}");

    // Verify subsequent single-token decode after prefill matches decode after sequential.
    let next_token = logits_prefill
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    let decode_from_prefill = model.forward(&[next_token], 3, &mut state);
    let decode_from_seq = model.forward(&[next_token], 3, &mut seq_state);

    let decode_dot: f32 = decode_from_prefill
        .iter()
        .zip(&decode_from_seq)
        .map(|(a, b)| a * b)
        .sum();
    let decode_na: f32 = decode_from_prefill
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let decode_nb: f32 = decode_from_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let decode_cos = decode_dot / (decode_na * decode_nb);
    eprintln!("[gemma4] decode step cosine: {decode_cos:.6}");
    assert!(
        decode_cos > 0.99999,
        "gemma4 decode cosine too low: {decode_cos}"
    );
}

#[test]
fn gemma4_truncate_and_resume() {
    let path = get_test_model_path();
    ensure_test_model(&path);

    let gguf = GgufFile::open(&path).expect("open test_gemma4.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load gemma4 model");

    let tokens = vec![88u32, 108, 105];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    // Prefill 3 tokens.
    let _ = model.forward_prefill(&tokens, 0, &mut state);
    assert_eq!(state.seq_len, 3);

    // Truncate back to position 1.
    state.truncate_to(1);
    assert_eq!(state.seq_len, 1);

    // Run token 108 at position 1 on truncated state.
    let resumed_logits = model.forward(&[108], 1, &mut state);

    // Compare against fresh state evaluated on [88, 108].
    let mut fresh_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let _ = model.forward(&[88], 0, &mut fresh_state);
    let fresh_logits = model.forward(&[108], 1, &mut fresh_state);

    let dot: f32 = resumed_logits
        .iter()
        .zip(&fresh_logits)
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = resumed_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = fresh_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[gemma4] truncate and resume cosine: {cos:.6}");
    assert!(cos > 0.99999, "truncate and resume cosine too low: {cos}");
}

#[test]
fn gemma4_f16_kv_compression_parity() {
    let path = get_test_model_path();
    ensure_test_model(&path);

    let gguf = GgufFile::open(&path).expect("open test_gemma4.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load gemma4 model");

    let tokens = vec![88u32, 108, 105];
    let mut state_f32 =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut state_f16 =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::F16).unwrap();

    let logits_f32 = model.forward_prefill(&tokens, 0, &mut state_f32);
    let logits_f16 = model.forward_prefill(&tokens, 0, &mut state_f16);

    assert_eq!(logits_f32.len(), logits_f16.len());
    let dot: f32 = logits_f32.iter().zip(&logits_f16).map(|(a, b)| a * b).sum();
    let na: f32 = logits_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = logits_f16.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[gemma4] f32 vs f16 kv compression cosine: {cos:.6}");
    assert!(cos > 0.999, "f32 vs f16 cosine too low: {cos}");
}

#[test]
fn gemma4_empty_prefill() {
    let path = get_test_model_path();
    ensure_test_model(&path);

    let gguf = GgufFile::open(&path).expect("open test_gemma4.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load gemma4 model");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let empty = model.forward_prefill(&[], 0, &mut state);
    assert!(empty.is_empty());
    assert_eq!(state.seq_len, 0);
}

#[test]
fn gemma4_all_logits_prefill_matches_last() {
    let path = get_test_model_path();
    ensure_test_model(&path);

    let gguf = GgufFile::open(&path).expect("open test_gemma4.gguf");
    let model = cera::model::load_model(gguf, Some(&path), 256).expect("load gemma4 model");

    // Prompt of 20 tokens to verify across N > 16.
    let tokens: Vec<u32> = (0..20).map(|i| 80 + (i % 30)).collect();
    let vocab_size = model.config().vocab_size;

    let mut state_last =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut state_all =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_last = model.forward_prefill(&tokens, 0, &mut state_last);
    let logits_all = model.forward_prefill_logits_all(&tokens, 0, &mut state_all);

    assert_eq!(logits_last.len(), vocab_size);
    assert_eq!(logits_all.len(), tokens.len() * vocab_size);

    let last_slice = &logits_all[(tokens.len() - 1) * vocab_size..];
    for (i, (&a, &b)) in logits_last.iter().zip(last_slice).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logits mismatch at vocab idx {i}: forward_prefill={a}, all_logits={b}"
        );
    }
}

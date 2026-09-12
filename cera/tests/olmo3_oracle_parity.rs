//! Cross-validation test comparing Olmo 3 (olmo2 with SWA and YaRN RoPE)
//! architecture in Cera against upstream llama.cpp oracle outputs.

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

fn ensure_test_fixture() -> Option<std::path::PathBuf> {
    static FIXTURE: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("cera_test_olmo3_{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join("test_olmo3.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let script = manifest_dir.join("../scripts/oracle/create_olmo3_test_model.py");
            if !script.exists() {
                eprintln!(
                    "skipping: fixture generator script not found at {}",
                    script.display()
                );
                return None;
            }
            for py in ["python3", "python"] {
                let status = std::process::Command::new(py)
                    .arg(&script)
                    .arg(&path)
                    .status();
                if matches!(status, Ok(st) if st.success()) {
                    return Some(path);
                }
            }
            eprintln!("skipping: python3/python failed to generate test_olmo3.gguf fixture");
            None
        })
        .clone()
}

#[test]
fn olmo3_matches_llama_cpp_oracle() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_olmo3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load olmo3 model");

    // Token IDs in test GGUF fixture (4 special tokens + ASCII byte value): 'T' (84 + 4 = 88), 'h' (104 + 4 = 108), 'e' (101 + 4 = 105).
    let tokens = vec![88u32, 108, 105];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let _logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
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

    // Reference sums from llama.cpp llama-eval-callback on test_olmo3.gguf:
    // embd: 0.158305
    // l_out-0: 7.324851
    // l_out-1: 32.913902
    // l_out-2: 47.925575
    // l_out-3: 33.984318
    // result_norm: 13.150345
    // result_output: 4.379386
    let expected = [
        ("embd", 0.158305),
        ("l_out-0", 7.324851),
        ("l_out-1", 32.913902),
        ("l_out-2", 47.925575),
        ("l_out-3", 33.984318),
        ("result_norm", 13.150345),
        ("result_output", 4.379386),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node}"));
        let diff = rel_diff(got, exp);
        eprintln!("[olmo3] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.01,
            "olmo3 divergence at {node}: cera={got}, llama={exp}, rel_diff={diff}"
        );
    }

    // Verify batched prefill matches sequential path
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
    eprintln!("[olmo3] batched vs sequential prefill cosine: {cos:.6}");
    assert!(cos > 0.9999, "olmo3 prefill cosine too low: {cos}");

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
    eprintln!("[olmo3] decode after batched vs sequential cosine: {d_cos:.6}");
    assert!(d_cos > 0.9999, "olmo3 decode cosine too low: {d_cos}");
}

#[test]
fn olmo3_long_prompt_prefill_and_multi_decode() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_olmo3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load olmo3 model");

    // 20 tokens (> window of 2, exercising repeated SWA patterns across 5 periods).
    let tokens: Vec<u32> = (0..20).map(|i| 88 + (i % 30) as u32).collect();
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
    assert_eq!(logits_prefill.len(), model.config().vocab_size);
    assert!(logits_prefill.iter().all(|v| v.is_finite()));
    assert_eq!(state.seq_len, 20);

    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_seq = model.forward(&[t], i, &mut seq_state);
    }

    let dot: f32 = logits_prefill
        .iter()
        .zip(&logits_seq)
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = logits_prefill.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = logits_seq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("[olmo3 long] batched vs sequential cosine: {cos:.6}");
    assert!(cos > 0.9999, "long prompt prefill cosine too low: {cos}");

    // Multi-step decode
    let mut next_token = 88u32;
    for step in 0..5 {
        let dec_b = model.forward(&[next_token], state.seq_len, &mut state);
        let dec_s = model.forward(&[next_token], seq_state.seq_len, &mut seq_state);

        let d_dot: f32 = dec_b.iter().zip(&dec_s).map(|(a, b)| a * b).sum();
        let d_na: f32 = dec_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let d_nb: f32 = dec_s.iter().map(|x| x * x).sum::<f32>().sqrt();
        let d_cos = d_dot / (d_na * d_nb);
        eprintln!("[olmo3 step {step}] decode cosine: {d_cos:.6}");
        assert!(d_cos > 0.9999, "step {step} decode cosine too low: {d_cos}");

        next_token = dec_b
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
    }
}

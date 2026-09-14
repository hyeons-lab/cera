//! Cross-validation test comparing Mistral 3 / Ministral 3 architecture in Cera
//! against upstream llama.cpp oracle outputs.

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
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let target_dir = manifest_dir.join("../target/tmp/cera_test_mistral3");
            let _ = std::fs::create_dir_all(&target_dir);
            let path = target_dir.join("test_mistral3.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }
            let script = manifest_dir.join("../scripts/oracle/create_mistral3_test_model.py");
            if !script.exists() {
                eprintln!(
                    "skipping: fixture generator script not found at {}",
                    script.display()
                );
                return None;
            }
            let probe = std::process::Command::new("python3")
                .args(["-c", "import numpy, gguf"])
                .output();
            let python_has_deps = matches!(probe, Ok(out) if out.status.success());

            for py in ["python3", "python"] {
                let status = std::process::Command::new(py)
                    .arg(&script)
                    .arg(&path)
                    .status();
                if matches!(status, Ok(st) if st.success())
                    && path.is_file()
                    && path.metadata().map(|m| m.len() > 1024).unwrap_or(false)
                {
                    return Some(path);
                }
            }
            eprintln!("skipping: python3/python failed to generate test_mistral3.gguf fixture");
            let require_fixture = std::env::var("CERA_REQUIRE_MODEL").as_deref() == Ok("1")
                || std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
            if require_fixture
                || (python_has_deps && matches!(std::env::var("CI").as_deref(), Ok("true" | "1")))
            {
                panic!(
                    "test_mistral3.gguf generation failed on CI runner despite python dependencies being present (check generator script)"
                );
            }
            None
        })
        .clone()
}

#[test]
fn mistral3_matches_llama_cpp_oracle() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    // Verify direct load_model dispatch
    let generic_model = cera::model::load_model(
        GgufFile::open(&path).expect("open test_mistral3.gguf"),
        Some(&path),
        256,
    )
    .expect("load mistral3 via load_model");
    assert_eq!(generic_model.config().architecture, "mistral3");
    assert_eq!(generic_model.config().n_layers, 4);

    let gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load mistral3 model");

    let tokens = vec![2u32, 69, 36, 70];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let _logits_dump = model.forward_prefill(&tokens, 0, &mut state);
    let occ = oracle_dump::take();

    let mut cera_sums: HashMap<String, f64> = HashMap::new();
    let last = model.config().n_layers - 1;
    let last_suffix = format!("-{last}");
    for (name, sum) in occ {
        if name.starts_with("result_") || name.ends_with(&last_suffix) {
            cera_sums.insert(name, sum);
        } else {
            *cera_sums.entry(name).or_insert(0.0) += sum;
        }
    }

    // Reference sums from llama.cpp llama-eval-callback on test_mistral3.gguf with prompt "A B":
    let expected = [
        ("embd", 0.206930),
        ("l_out-0", 2.218392),
        ("l_out-1", 1.467810),
        ("l_out-2", -2.326116),
        ("l_out-3", -0.460856),
        ("result_norm", -1.491950),
        ("result_output", 6.279382),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node}"));
        let diff = rel_diff(got, exp);
        eprintln!("[mistral3] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.01,
            "node {node} diverged: cera={got} llama={exp} diff={diff}"
        );
    }

    // Verify batched prefill against sequential decode (run without oracle_dump active)
    let mut state_batched =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits_batched = model.forward_prefill(&tokens, 0, &mut state_batched);

    let mut state_seq =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut last_logits = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        last_logits = model.forward(&[t], pos, &mut state_seq);
    }

    assert_eq!(logits_batched.len(), last_logits.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in logits_batched.iter().zip(last_logits.iter()) {
        dot += a as f64 * b as f64;
        norm_a += (a as f64).powi(2);
        norm_b += (b as f64).powi(2);
    }
    assert!(
        norm_a > 0.0 && norm_b > 0.0,
        "logits norms must be positive"
    );
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    eprintln!("[mistral3] batched prefill vs sequential decode cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.9999,
        "prefill vs decode diverged: cosine = {cosine}"
    );
}

#[test]
fn mistral3_empty_tokens_prefill_panics() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load mistral3");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.forward_prefill(&[], 0, &mut state);
    }));
    assert!(
        result.is_err(),
        "forward_prefill must panic on empty token slice"
    );
}

#[test]
fn mistral3_tokenizer_loads_and_roundtrips() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("load tokenizer from mistral3 gguf");
    let prompt = "A B";
    let encoded = tokenizer.encode(prompt);
    assert!(!encoded.is_empty(), "encoded tokens must not be empty");
    let decoded = tokenizer.decode(&encoded);
    assert_eq!(decoded, prompt, "tokenizer roundtrip failed");
}
#[test]
fn mistral3_long_prompt_prefill_and_temp_scaling() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load mistral3");

    // 150 tokens (crosses orig_ctx_len = 64 and 2 * orig_ctx_len = 128, exercising multiple attention temperature scaling plateaus)
    let tokens: Vec<u32> = (0..150).map(|i| 4 + (i % 200) as u32).collect();
    let mut batched_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits_batched = model.forward_prefill(&tokens, 0, &mut batched_state);
    assert_eq!(logits_batched.len(), model.config().vocab_size);
    assert!(logits_batched.iter().all(|v| v.is_finite()));
    assert_eq!(batched_state.seq_len, 150);

    let mut seq_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_seq = model.forward(&[t], i, &mut seq_state);
    }
    assert_eq!(seq_state.seq_len, 150);

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in logits_batched.iter().zip(logits_seq.iter()) {
        dot += a as f64 * b as f64;
        norm_a += (a as f64).powi(2);
        norm_b += (b as f64).powi(2);
    }
    assert!(
        norm_a > 0.0 && norm_b > 0.0,
        "logits norms must be positive"
    );
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    eprintln!("[mistral3 long] batched prefill vs sequential decode cosine: {cosine:.6}");
    assert!(
        cosine > 0.9999,
        "long prompt prefill vs decode diverged: cosine = {cosine}"
    );

    // Multi-step decode after batched prefill vs sequential prefill
    let mut next_token = logits_batched
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap();

    for step in 0..4 {
        let dec_b = model.forward(&[next_token], batched_state.seq_len, &mut batched_state);
        let dec_s = model.forward(&[next_token], seq_state.seq_len, &mut seq_state);

        let mut d_dot = 0.0f64;
        let mut d_na = 0.0f64;
        let mut d_nb = 0.0f64;
        for (&a, &b) in dec_b.iter().zip(dec_s.iter()) {
            d_dot += a as f64 * b as f64;
            d_na += (a as f64).powi(2);
            d_nb += (b as f64).powi(2);
        }
        let d_cos = d_dot / (d_na.sqrt() * d_nb.sqrt());
        eprintln!("[mistral3 step {step}] decode cosine: {d_cos:.6}");
        assert!(
            d_cos > 0.9999,
            "step {step} decode diverged: cosine = {d_cos}"
        );

        next_token = dec_b
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn mistral3_metal_loader_rejects_unsupported_arch() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    let result = cera::model::load_model_metal(gguf, Some(&path), 256);
    assert!(
        result.is_err(),
        "load_model_metal must reject mistral3 as unsupported architecture"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn mistral3_gpu_loader_rejects_unsupported_arch() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    let result = cera::model::load_model_gpu(gguf, Some(&path), 256);
    assert!(
        result.is_err(),
        "load_model_gpu must reject mistral3 as unsupported architecture"
    );
}

#[test]
fn mistral3_pretokenizer_fallback_without_pre_key() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    gguf.metadata.remove("tokenizer.ggml.pre");
    let tokenizer = cera::tokenizer::BpeTokenizer::from_gguf(&gguf)
        .expect("load tokenizer with fallback pretokenizer");
    let prompt = "A B";
    let encoded = tokenizer.encode(prompt);
    assert!(!encoded.is_empty());
    assert_eq!(tokenizer.decode(&encoded), prompt);
}

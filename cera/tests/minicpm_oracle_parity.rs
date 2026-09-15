//! Cross-validation test comparing MiniCPM architecture in Cera against
//! upstream llama.cpp oracle outputs.

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
            if let Ok(p) = std::env::var("CERA_TEST_MINICPM_GGUF") {
                let path = std::path::PathBuf::from(p);
                if path.exists() {
                    return Some(path);
                }
            }
            if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
                let path = std::path::PathBuf::from(d).join("test_minicpm.gguf");
                if path.exists() {
                    return Some(path);
                }
            }
            let target_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/oracle/models/test_minicpm.gguf");
            if target_path.exists() {
                return Some(target_path);
            }
            let root_target_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../target/oracle/models/test_minicpm.gguf");
            if root_target_path.exists() {
                return Some(root_target_path);
            }

            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let target_dir = manifest_dir.join("../target/tmp/cera_test_minicpm");
            if let Err(e) = std::fs::create_dir_all(&target_dir) {
                eprintln!(
                    "skipping: failed to create fixture directory {}: {e}",
                    target_dir.display()
                );
                return None;
            }
            let path = target_dir.join("test_minicpm.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }
            let script = manifest_dir.join("../scripts/oracle/create_minicpm_test_model.py");
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
                if matches!(status, Ok(st) if st.success()) {
                    return Some(path);
                }
            }
            eprintln!("skipping: python3/python failed to generate test_minicpm.gguf fixture");
            let require_fixture = std::env::var("CERA_REQUIRE_MODEL").as_deref() == Ok("1")
                || std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
            if require_fixture
                || (python_has_deps && matches!(std::env::var("CI").as_deref(), Ok("true" | "1")))
            {
                panic!("test_minicpm.gguf generation failed on CI runner despite python dependencies being present (check generator script)");
            }
            None
        })
        .clone()
}

#[test]
fn minicpm_matches_llama_cpp_oracle() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    // Verify direct load_model dispatch
    let generic_model = cera::model::load_model(
        GgufFile::open(&path).expect("open test_minicpm.gguf"),
        Some(&path),
        256,
    )
    .expect("load minicpm via load_model");
    assert_eq!(generic_model.config().architecture, "minicpm");

    let gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load minicpm model");

    let tokens = vec![88u32, 108, 105];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    oracle_dump::begin();
    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state);
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

    // Reference sums from llama.cpp llama-eval-callback on test_minicpm.gguf:
    // embd: 1.899666
    // l_out-0: 2.456619
    // l_out-1: -0.877008
    // result_norm: -1.249908
    // result_output: 42.378071
    let expected = [
        ("embd", 1.899666),
        ("l_out-0", 2.456619),
        ("l_out-1", -0.877008),
        ("result_norm", -1.249908),
        ("result_output", 42.378071),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node}"));
        let diff = rel_diff(got, exp);
        eprintln!("[minicpm] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.01,
            "node {node} diverged: cera={got} llama={exp} diff={diff}"
        );
    }

    // Verify sequential single-token decode against batched prefill
    let mut state_seq =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut last_logits = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        last_logits = model.forward(&[t], pos, &mut state_seq);
    }

    assert_eq!(logits_prefill.len(), last_logits.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in logits_prefill.iter().zip(last_logits.iter()) {
        dot += a as f64 * b as f64;
        norm_a += (a as f64).powi(2);
        norm_b += (b as f64).powi(2);
    }
    assert!(
        norm_a > 0.0 && norm_b > 0.0,
        "logits norms must be positive"
    );
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    eprintln!("[minicpm] batched prefill vs sequential decode cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.9999,
        "prefill vs decode diverged: cosine = {cosine}"
    );
}

#[test]
fn minicpm_scalar_fallback_when_keys_absent() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    let gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    let model_explicit = LlamaModel::from_gguf(gguf.clone(), 256).expect("load minicpm model");

    // The fixture sets:
    // embedding_scale = 12.0
    // residual_scale = 1.4 / sqrt(2) ≈ 0.9899495
    // logit_scale = 64.0 / 256.0 = 0.25
    let scalars = model_explicit.config().scalars;
    assert!((scalars.embedding - 12.0).abs() < 1e-5);
    assert!((scalars.residual - 0.9899495).abs() < 1e-5);
    assert!((scalars.logit - 0.25).abs() < 1e-5);

    // Strip scalar keys from metadata to exercise absent-key fallback branch
    let mut stripped_gguf = gguf;
    stripped_gguf.metadata.remove("minicpm.embedding_scale");
    stripped_gguf.metadata.remove("minicpm.residual_scale");
    stripped_gguf.metadata.remove("minicpm.logit_scale");
    let model_stripped =
        LlamaModel::from_gguf(stripped_gguf, 256).expect("load stripped minicpm model");
    let scalars_fallback = model_stripped.config().scalars;
    assert!((scalars_fallback.embedding - 12.0).abs() < 1e-5);
    assert!((scalars_fallback.residual - 0.9899495).abs() < 1e-5);
    assert!((scalars_fallback.logit - 0.25).abs() < 1e-5);

    // Verify that forward passes produce identical logits between explicit and fallback
    let tokens = vec![88u32, 108, 105];
    let mut state_exp =
        InferenceState::from_config_with_compression(model_explicit.config(), &KvCompression::None)
            .unwrap();
    let mut state_fallback =
        InferenceState::from_config_with_compression(model_stripped.config(), &KvCompression::None)
            .unwrap();
    let logits_exp = model_explicit.forward_prefill(&tokens, 0, &mut state_exp);
    let logits_fallback = model_stripped.forward_prefill(&tokens, 0, &mut state_fallback);
    assert_eq!(logits_exp.len(), logits_fallback.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in logits_exp.iter().zip(logits_fallback.iter()) {
        dot += a as f64 * b as f64;
        norm_a += (a as f64).powi(2);
        norm_b += (b as f64).powi(2);
    }
    assert!(
        norm_a > 0.0 && norm_b > 0.0,
        "logits norms must be positive"
    );
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    eprintln!("[minicpm] explicit vs fallback logits cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.999999,
        "fallback vs explicit logits diverged: cosine = {cosine}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn minicpm_metal_forward_runs() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    let cpu_model = LlamaModel::from_gguf(gguf.clone(), 256).expect("load cpu minicpm");
    let metal_model = match cera::model::load_model_metal(gguf, Some(&path), 256) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_METAL").as_deref() != Ok("1"),
                "CERA_REQUIRE_METAL is set but Metal model failed to load: {e}"
            );
            eprintln!("skipping: Metal device unavailable ({e})");
            return;
        }
    };
    let tokens = vec![88u32, 108, 105];
    let mut cpu_state =
        InferenceState::from_config_with_compression(cpu_model.config(), &KvCompression::None)
            .unwrap();
    let mut metal_state =
        InferenceState::from_config_with_compression(metal_model.config(), &KvCompression::None)
            .unwrap();
    let cpu_logits = cpu_model.forward_prefill(&tokens, 0, &mut cpu_state);
    let metal_logits = metal_model.forward_prefill(&tokens, 0, &mut metal_state);
    assert_eq!(metal_logits.len(), metal_model.config().vocab_size);
    assert_eq!(cpu_logits.len(), metal_logits.len());
    let mut dot = 0.0f64;
    let mut norm_cpu = 0.0f64;
    let mut norm_metal = 0.0f64;
    for (&c, &m) in cpu_logits.iter().zip(metal_logits.iter()) {
        dot += c as f64 * m as f64;
        norm_cpu += (c as f64).powi(2);
        norm_metal += (m as f64).powi(2);
    }
    assert!(
        norm_cpu > 0.0 && norm_metal > 0.0,
        "logits norms must be positive"
    );
    let cosine = dot / (norm_cpu.sqrt() * norm_metal.sqrt());
    eprintln!("[minicpm] CPU vs Metal prefill logits cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.999,
        "CPU vs Metal logits diverged: cosine = {cosine}"
    );

    // Single-token decode on Metal
    let token_next = [42u32];
    let cpu_dec = cpu_model.forward(&token_next, cpu_state.seq_len, &mut cpu_state);
    let metal_dec = metal_model.forward(&token_next, metal_state.seq_len, &mut metal_state);
    assert_eq!(metal_dec.len(), metal_model.config().vocab_size);
    assert_eq!(cpu_dec.len(), metal_dec.len());
    let mut dot_dec = 0.0f64;
    let mut norm_cpu_dec = 0.0f64;
    let mut norm_metal_dec = 0.0f64;
    for (&c, &m) in cpu_dec.iter().zip(metal_dec.iter()) {
        dot_dec += c as f64 * m as f64;
        norm_cpu_dec += (c as f64).powi(2);
        norm_metal_dec += (m as f64).powi(2);
    }
    assert!(
        norm_cpu_dec > 0.0 && norm_metal_dec > 0.0,
        "decode logits norms must be positive"
    );
    let cosine_dec = dot_dec / (norm_cpu_dec.sqrt() * norm_metal_dec.sqrt());
    eprintln!("[minicpm] CPU vs Metal decode logits cosine similarity: {cosine_dec:.6}");
    assert!(
        cosine_dec > 0.999,
        "CPU vs Metal decode logits diverged: cosine = {cosine_dec}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn minicpm_gpu_forward_runs() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    let cpu_model = LlamaModel::from_gguf(gguf.clone(), 256).expect("load cpu minicpm");
    let model = match cera::model::load_model_gpu(gguf, Some(&path), 256) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU").as_deref() != Ok("1"),
                "CERA_REQUIRE_GPU is set but WebGPU model failed to load: {e}"
            );
            eprintln!("skipping: WebGPU adapter unavailable ({e})");
            return;
        }
    };
    assert_eq!(model.config().architecture, "minicpm");

    let tokens = vec![88u32, 108, 105];
    let mut cpu_state =
        InferenceState::from_config_with_compression(cpu_model.config(), &KvCompression::None)
            .unwrap();
    let mut gpu_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let cpu_logits = cpu_model.forward_prefill(&tokens, 0, &mut cpu_state);
    let gpu_logits = model.forward_prefill(&tokens, 0, &mut gpu_state);
    assert_eq!(gpu_logits.len(), model.config().vocab_size);
    assert_eq!(cpu_logits.len(), gpu_logits.len());

    let mut dot = 0.0f64;
    let mut norm_cpu = 0.0f64;
    let mut norm_gpu = 0.0f64;
    for (&c, &g) in cpu_logits.iter().zip(gpu_logits.iter()) {
        dot += c as f64 * g as f64;
        norm_cpu += (c as f64).powi(2);
        norm_gpu += (g as f64).powi(2);
    }
    assert!(
        norm_cpu > 0.0 && norm_gpu > 0.0,
        "prefill logits norms must be positive"
    );
    let cosine = dot / (norm_cpu.sqrt() * norm_gpu.sqrt());
    eprintln!("[minicpm] CPU vs WebGPU prefill logits cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.999,
        "CPU vs WebGPU prefill logits diverged: cosine = {cosine}"
    );

    let token_next = [42u32];
    let cpu_dec = cpu_model.forward(&token_next, cpu_state.seq_len, &mut cpu_state);
    let gpu_dec = model.forward(&token_next, gpu_state.seq_len, &mut gpu_state);
    assert_eq!(gpu_dec.len(), model.config().vocab_size);
    assert_eq!(cpu_dec.len(), gpu_dec.len());

    let mut dot_dec = 0.0f64;
    let mut norm_cpu_dec = 0.0f64;
    let mut norm_gpu_dec = 0.0f64;
    for (&c, &g) in cpu_dec.iter().zip(gpu_dec.iter()) {
        dot_dec += c as f64 * g as f64;
        norm_cpu_dec += (c as f64).powi(2);
        norm_gpu_dec += (g as f64).powi(2);
    }
    assert!(
        norm_cpu_dec > 0.0 && norm_gpu_dec > 0.0,
        "decode logits norms must be positive"
    );
    let cosine_dec = dot_dec / (norm_cpu_dec.sqrt() * norm_gpu_dec.sqrt());
    eprintln!("[minicpm] CPU vs WebGPU decode logits cosine similarity: {cosine_dec:.6}");
    assert!(
        cosine_dec > 0.999,
        "CPU vs WebGPU decode logits diverged: cosine = {cosine_dec}"
    );
}

#[test]
fn minicpm_empty_tokens_prefill_panics() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load minicpm");
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
fn minicpm_tokenizer_loads_and_roundtrips() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("load tokenizer from minicpm gguf");
    let text = "Hello world";
    let tokens = tokenizer.encode(text);
    assert!(!tokens.is_empty(), "encoded tokens must not be empty");
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, text, "tokenizer encode/decode must round-trip");
}

#[test]
fn test_scalar_multipliers_rejects_non_positive_and_nan() {
    use cera::gguf::GgufValue;
    use cera::model::ScalarMultipliers;
    use std::sync::Arc;

    let base_gguf = GgufFile::from_bytes(Arc::from(vec![
        0x47, 0x47, 0x55, 0x46, // magic "GGUF"
        0x03, 0x00, 0x00, 0x00, // version 3
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // tensor_count = 0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // kv_count = 0
    ]))
    .expect("parse minimal in-memory GGUF");

    // 0. Base empty metadata verifies defaults:
    // minicpm defaults
    let scalars_mcpm = ScalarMultipliers::from_gguf(&base_gguf, "minicpm", 2, 64).unwrap();
    assert_eq!(scalars_mcpm.embedding, 12.0);
    assert!((scalars_mcpm.residual - 1.4 / (2.0f32).sqrt()).abs() < 1e-5);
    assert_eq!(scalars_mcpm.logit, 64.0 / 256.0);
    assert!(scalars_mcpm.attn.is_none());

    // llama defaults
    let scalars_llama = ScalarMultipliers::from_gguf(&base_gguf, "llama", 2, 64).unwrap();
    assert_eq!(scalars_llama.embedding, 1.0);
    assert_eq!(scalars_llama.residual, 1.0);
    assert_eq!(scalars_llama.logit, 1.0);
    assert!(scalars_llama.attn.is_none());

    // 1. Zero, negative, NaN, or infinity embedding_scale must be rejected
    let mut bad_gguf = base_gguf.clone();
    bad_gguf
        .metadata
        .insert("minicpm.embedding_scale".to_string(), GgufValue::F32(0.0));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf
        .metadata
        .insert("minicpm.embedding_scale".to_string(), GgufValue::F32(-1.0));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.embedding_scale".to_string(),
        GgufValue::F32(f32::NAN),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.embedding_scale".to_string(),
        GgufValue::F32(f32::INFINITY),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    // 2. Non-positive, NaN, or infinity residual_scale must be rejected
    let mut bad_gguf = base_gguf.clone();
    bad_gguf
        .metadata
        .insert("minicpm.residual_scale".to_string(), GgufValue::F32(0.0));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf
        .metadata
        .insert("minicpm.residual_scale".to_string(), GgufValue::F32(-0.5));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.residual_scale".to_string(),
        GgufValue::F32(f32::NAN),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.residual_scale".to_string(),
        GgufValue::F32(f32::INFINITY),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    // 3. Non-positive, NaN, subnormal, or infinity logit_scale must be rejected
    let mut bad_gguf = base_gguf.clone();
    bad_gguf
        .metadata
        .insert("minicpm.logit_scale".to_string(), GgufValue::F32(0.0));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf
        .metadata
        .insert("minicpm.logit_scale".to_string(), GgufValue::F32(-2.0));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf
        .metadata
        .insert("minicpm.logit_scale".to_string(), GgufValue::F32(f32::NAN));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.logit_scale".to_string(),
        GgufValue::F32(f32::INFINITY),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf
        .metadata
        .insert("minicpm.logit_scale".to_string(), GgufValue::F32(1e-40));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    // 4. Negative, NaN, or infinity attention.scale must be rejected if present
    let mut bad_gguf = base_gguf.clone();
    bad_gguf
        .metadata
        .insert("minicpm.attention.scale".to_string(), GgufValue::F32(-0.1));
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.attention.scale".to_string(),
        GgufValue::F32(f32::NAN),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    bad_gguf.metadata.insert(
        "minicpm.attention.scale".to_string(),
        GgufValue::F32(f32::INFINITY),
    );
    assert!(ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err());

    // 5. Multipliers outside symmetric bounds [1e-4, 1e4] must be rejected
    for (key, too_small, too_large) in [
        ("minicpm.embedding_scale", 1e-5, 1e5),
        ("minicpm.residual_scale", 1e-5, 1e5),
        ("minicpm.logit_scale", 1e-5, 1e5),
        ("minicpm.attention.scale", 1e-5, 1e5),
    ] {
        let mut bad_gguf = base_gguf.clone();
        bad_gguf
            .metadata
            .insert(key.to_string(), GgufValue::F32(too_small));
        assert!(
            ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err(),
            "{key} with {too_small} must be rejected"
        );

        let mut bad_gguf = base_gguf.clone();
        bad_gguf
            .metadata
            .insert(key.to_string(), GgufValue::F32(too_large));
        assert!(
            ScalarMultipliers::from_gguf(&bad_gguf, "minicpm", 2, 64).is_err(),
            "{key} with {too_large} must be rejected"
        );
    }
}

#[test]
fn minicpm_odd_head_dim_is_rejected() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    // Insert an odd key_length (15) which should be rejected because RoPE requires even head_dim
    gguf.metadata.insert(
        "minicpm.attention.key_length".to_string(),
        cera::gguf::GgufValue::U32(15),
    );
    let result = LlamaModel::from_gguf(gguf, 256);
    assert!(
        result.is_err(),
        "model loader must reject odd head_dim for RoPE rotation"
    );
}

#[test]
fn minicpm5_arch_alias_loads_and_evaluates() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_minicpm.gguf");
    gguf.metadata.insert(
        "general.architecture".to_string(),
        cera::gguf::GgufValue::String("minicpm5".to_string()),
    );
    // Remap minicpm.* metadata keys to minicpm5.*
    let keys: Vec<String> = gguf.metadata.keys().cloned().collect();
    for k in keys {
        if let Some(suffix) = k.strip_prefix("minicpm.") {
            let val = gguf.metadata.remove(&k).unwrap();
            gguf.metadata.insert(format!("minicpm5.{suffix}"), val);
        }
    }
    let model =
        cera::model::load_model(gguf.clone(), Some(&path), 256).expect("load minicpm5 alias");
    assert_eq!(model.config().architecture, "minicpm5");
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits = model.forward(&[88u32], 0, &mut state);
    assert_eq!(logits.len(), model.config().vocab_size);

    #[cfg(feature = "metal")]
    if let Ok(metal_model) = cera::model::load_model_metal(gguf.clone(), Some(&path), 256) {
        assert_eq!(metal_model.config().architecture, "minicpm5");
        let mut m_state = InferenceState::from_config_with_compression(
            metal_model.config(),
            &KvCompression::None,
        )
        .unwrap();
        let m_logits = metal_model.forward(&[88u32], 0, &mut m_state);
        assert_eq!(m_logits.len(), metal_model.config().vocab_size);
    }

    #[cfg(feature = "gpu")]
    if let Ok(gpu_model) = cera::model::load_model_gpu(gguf, Some(&path), 256) {
        assert_eq!(gpu_model.config().architecture, "minicpm5");
        let mut g_state =
            InferenceState::from_config_with_compression(gpu_model.config(), &KvCompression::None)
                .unwrap();
        let g_logits = gpu_model.forward(&[88u32], 0, &mut g_state);
        assert_eq!(g_logits.len(), gpu_model.config().vocab_size);
    }
}

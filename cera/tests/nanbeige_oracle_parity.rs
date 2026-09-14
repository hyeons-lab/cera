//! Cross-validation test comparing Nanbeige architecture in Cera against
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
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let target_dir = manifest_dir.join("../target/tmp/cera_test_nanbeige");
            let _ = std::fs::create_dir_all(&target_dir);
            let path = target_dir.join("test_nanbeige.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }
            let script = manifest_dir.join("../scripts/oracle/create_nanbeige_test_model.py");
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
            eprintln!("skipping: python3/python failed to generate test_nanbeige.gguf fixture");
            let require_fixture = std::env::var("CERA_REQUIRE_MODEL").as_deref() == Ok("1")
                || std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
            if require_fixture
                || (python_has_deps && matches!(std::env::var("CI").as_deref(), Ok("true" | "1")))
            {
                panic!("test_nanbeige.gguf generation failed on CI runner despite python dependencies being present (check generator script)");
            }
            None
        })
        .clone()
}

#[test]
fn nanbeige_matches_llama_cpp_oracle() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    // Verify direct load_model dispatch
    let generic_model = cera::model::load_model(
        GgufFile::open(&path).expect("open test_nanbeige.gguf"),
        Some(&path),
        256,
    )
    .expect("load nanbeige via load_model");
    assert_eq!(generic_model.config().architecture, "nanbeige");
    assert_eq!(generic_model.config().n_layers, 4);

    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load nanbeige model");

    let tokens = vec![69u32, 112, 109];
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

    // Reference sums from llama.cpp llama-eval-callback on test_nanbeige.gguf:
    // embd: 0.451272
    // l_out-0: 3.278652
    // l_out-1: -3.064673
    // loop_norm-1: -14.905140
    // l_out-2: -15.897154
    // l_out-3: -2.910775
    // result_norm: -2.766613
    // result_output: -16.223349
    let expected = [
        ("embd", 0.451272),
        ("l_out-0", 3.278652),
        ("l_out-1", -3.064673),
        ("loop_norm-1", -14.905140),
        ("l_out-2", -15.897154),
        ("l_out-3", -2.910775),
        ("result_norm", -2.766613),
        ("result_output", -16.223349),
    ];

    for (node, exp) in expected {
        let got = *cera_sums
            .get(node)
            .unwrap_or_else(|| panic!("missing node {node}"));
        let diff = rel_diff(got, exp);
        eprintln!("[nanbeige] {node}: cera={got:.6} llama={exp:.6} rel_diff={diff:.6}");
        assert!(
            diff < 0.0002,
            "node {node} diverged: cera={got} llama={exp} diff={diff}"
        );
    }

    // Verify actual batched prefill against sequential decode (run without oracle_dump active)
    let mut state_batched =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits_batched = model.forward_prefill(&tokens, 0, &mut state_batched);

    // Verify sequential single-token decode against batched prefill
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
    eprintln!("[nanbeige] batched prefill vs sequential decode cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.9999,
        "prefill vs decode diverged: cosine = {cosine}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn nanbeige_metal_forward_runs() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let cpu_model = LlamaModel::from_gguf(gguf.clone(), 256).expect("load cpu nanbeige");
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
    let tokens = vec![69u32, 112, 109];
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
    eprintln!("[nanbeige] CPU vs Metal prefill logits cosine similarity: {cosine:.6}");
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
    eprintln!("[nanbeige] CPU vs Metal decode logits cosine similarity: {cosine_dec:.6}");
    assert!(
        cosine_dec > 0.999,
        "CPU vs Metal decode logits diverged: cosine = {cosine_dec}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn nanbeige_gpu_forward_runs() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let cpu_model = LlamaModel::from_gguf(gguf.clone(), 256).expect("load cpu nanbeige");
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
    assert_eq!(model.config().architecture, "nanbeige");

    let tokens = vec![69u32, 112, 109];
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
    eprintln!("[nanbeige] CPU vs WebGPU prefill logits cosine similarity: {cosine:.6}");
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
    eprintln!("[nanbeige] CPU vs WebGPU decode logits cosine similarity: {cosine_dec:.6}");
    assert!(
        cosine_dec > 0.999,
        "CPU vs WebGPU decode logits diverged: cosine = {cosine_dec}"
    );
}

#[test]
fn nanbeige_empty_tokens_prefill_panics() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load nanbeige");
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
fn nanbeige_tokenizer_loads_and_roundtrips() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("load tokenizer from nanbeige gguf");
    let prompt = "Ali";
    let encoded = tokenizer.encode(prompt);
    assert!(!encoded.is_empty(), "encoded tokens must not be empty");
    let decoded = tokenizer.decode(&encoded);
    assert_eq!(decoded, prompt, "tokenizer roundtrip failed");
}

#[test]
fn nanbeige_rejects_invalid_loops_metadata() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    gguf.metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::U32(0));
    let err = match LlamaModel::from_gguf(gguf, 256) {
        Ok(_) => panic!("expected failure for num_loops = 0"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nanbeige.num_loops must be >= 1"),
        "expected error message about num_loops >= 1, got: {msg}"
    );
}

#[test]
fn nanbeige_handles_integer_variants_and_rejects_malformed_metadata() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");

    // I32 num_loops should be accepted
    let mut gguf_i32 = gguf.clone();
    gguf_i32
        .metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::I32(3));
    let model_i32 = LlamaModel::from_gguf(gguf_i32, 256).expect("load with i32 loops");
    assert_eq!(model_i32.config().n_layers, 6);

    // Negative I32 should be rejected
    let mut gguf_neg = gguf.clone();
    gguf_neg
        .metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::I32(-1));
    let err_neg = match LlamaModel::from_gguf(gguf_neg, 256) {
        Ok(_) => panic!("expected failure for negative num_loops"),
        Err(e) => e,
    };
    assert!(format!("{err_neg:#}").contains("nanbeige.num_loops must be >= 1"));

    // Unexpected type (e.g. String) for num_loops should be rejected
    let mut gguf_str = gguf.clone();
    gguf_str.metadata.insert(
        "nanbeige.num_loops".to_string(),
        GgufValue::String("two".into()),
    );
    let err_str = match LlamaModel::from_gguf(gguf_str, 256) {
        Ok(_) => panic!("expected failure for non-integer num_loops"),
        Err(e) => e,
    };
    assert!(format!("{err_str:#}").contains("unexpected metadata type"));

    // U8 and U16 num_loops should be accepted
    let mut gguf_u8 = gguf.clone();
    gguf_u8
        .metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::U8(2));
    let model_u8 = LlamaModel::from_gguf(gguf_u8, 256).expect("load with u8 loops");
    assert_eq!(model_u8.config().n_layers, 4);

    let mut gguf_u16 = gguf.clone();
    gguf_u16
        .metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::U16(2));
    let model_u16 = LlamaModel::from_gguf(gguf_u16, 256).expect("load with u16 loops");
    assert_eq!(model_u16.config().n_layers, 4);

    // Case-insensitivity: uppercase architecture "Nanbeige" should resolve correctly
    let mut gguf_case = gguf.clone();
    gguf_case.metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("Nanbeige".into()),
    );
    let model_case = LlamaModel::from_gguf(gguf_case, 256).expect("load with mixed-case arch");
    assert_eq!(model_case.config().architecture, "nanbeige");

    // Unexpected type (e.g. String) for skip_loop_final_norm should be rejected
    let mut gguf_skip_str = gguf.clone();
    gguf_skip_str.metadata.insert(
        "nanbeige.skip_loop_final_norm".to_string(),
        GgufValue::String("false".into()),
    );
    let err_skip_str = match LlamaModel::from_gguf(gguf_skip_str, 256) {
        Ok(_) => panic!("expected failure for non-bool skip_loop_final_norm"),
        Err(e) => e,
    };
    assert!(
        format!("{err_skip_str:#}")
            .contains("nanbeige.skip_loop_final_norm has unexpected metadata type")
    );

    // Excessive logical layer count (> 512) should be rejected
    let mut gguf_excess = gguf.clone();
    gguf_excess
        .metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::U32(300));
    let err_excess = match LlamaModel::from_gguf(gguf_excess, 256) {
        Ok(_) => panic!("expected failure for layer count > 512"),
        Err(e) => e,
    };
    assert!(format!("{err_excess:#}").contains("exceeds maximum supported layers"));
}

#[test]
fn nanbeige_respects_skip_loop_final_norm_and_single_loop() {
    use cera::gguf::GgufValue;
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let model_default = LlamaModel::from_gguf(gguf.clone(), 256).expect("load default");
    assert_eq!(model_default.loop_norm_interval(), Some(2));

    // When skip_loop_final_norm is true, loop_norm_interval should be None
    let mut gguf_skip = gguf.clone();
    gguf_skip.metadata.insert(
        "nanbeige.skip_loop_final_norm".to_string(),
        GgufValue::Bool(true),
    );
    let model_skip = LlamaModel::from_gguf(gguf_skip, 256).expect("load with skip_loop_final_norm");
    assert_eq!(model_skip.loop_norm_interval(), None);

    // When skip_loop_final_norm is U8(1), loop_norm_interval should also be None
    let mut gguf_skip_u8 = gguf.clone();
    gguf_skip_u8.metadata.insert(
        "nanbeige.skip_loop_final_norm".to_string(),
        GgufValue::U8(1),
    );
    let model_skip_u8 =
        LlamaModel::from_gguf(gguf_skip_u8, 256).expect("load with skip_loop_final_norm as u8");
    assert_eq!(model_skip_u8.loop_norm_interval(), None);

    // When num_loops is 1, loop_norm_interval should be None
    let mut gguf_1loop = gguf;
    gguf_1loop
        .metadata
        .insert("nanbeige.num_loops".to_string(), GgufValue::U32(1));
    let model_1loop = LlamaModel::from_gguf(gguf_1loop, 256).expect("load with 1 loop");
    assert_eq!(model_1loop.loop_norm_interval(), None);
    assert_eq!(model_1loop.config().n_layers, 2);
}

#[test]
fn nanbeige_forward_prefill_logits_all_matches_last_position() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_nanbeige.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load nanbeige model");

    let tokens = vec![69u32, 112, 109];
    let mut state_all =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let all_logits = model.forward_prefill_logits_all(&tokens, 0, &mut state_all);

    let mut state_last =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let last_logits = model.forward_prefill(&tokens, 0, &mut state_last);

    let vocab = model.config().vocab_size;
    assert_eq!(all_logits.len(), tokens.len() * vocab);
    let last_slice = &all_logits[(tokens.len() - 1) * vocab..];
    assert_eq!(last_slice.len(), last_logits.len());

    let mut max_diff = 0.0f32;
    for (&a, &b) in last_slice.iter().zip(last_logits.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < 1e-5,
        "forward_prefill_logits_all last slice diverged from forward_prefill: max_diff={max_diff}"
    );
}

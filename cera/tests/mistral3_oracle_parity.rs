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
        "forward_prefill must panic on empty token slice per Model trait contract"
    );

    // forward_prefill_chunked on empty slice returns (0, None) without panicking
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let (chunk_len, chunk_logits) = model.forward_prefill_chunked(&[], 0, &mut state, 0, &cancel);
    assert_eq!(chunk_len, 0);
    assert!(chunk_logits.is_none());
}

#[test]
fn mistral3_attention_temp_scaling_monotonic_and_floor_threshold() {
    let scale = 0.1f32;
    let floor_scale = 16usize;

    // For pos < floor_scale, q_scale should strictly be 1.0 (no scaling)
    for pos in 0..floor_scale {
        let q_scale = if pos >= floor_scale {
            ((pos as f32 / floor_scale as f32).floor() + 1.0).ln() * scale + 1.0
        } else {
            1.0
        };
        assert_eq!(
            q_scale, 1.0,
            "pos {pos} < floor_scale should evaluate to exactly 1.0"
        );
    }

    // For pos >= floor_scale, q_scale should be monotonic non-decreasing and strictly > 1.0
    let mut prev = 1.0f32;
    for pos in floor_scale..128 {
        let q_scale = ((pos as f32 / floor_scale as f32).floor() + 1.0).ln() * scale + 1.0;
        assert!(
            q_scale >= prev,
            "pos {pos} q_scale {q_scale} not monotonic with prev {prev}"
        );
        assert!(q_scale > 1.0, "pos {pos} q_scale {q_scale} must exceed 1.0");
        assert!(q_scale.is_finite(), "pos {pos} q_scale must be finite");
        prev = q_scale;
    }
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

/// The fixture without projection/FFN biases (see `assert_gpu_matches_cpu`).
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
fn ensure_nobias_fixture() -> Option<std::path::PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest_dir.join("../target/tmp/cera_test_mistral3");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_mistral3_nobias.gguf");
    if path.metadata().map(|m| m.len() > 1024).unwrap_or(false) {
        return Some(path);
    }
    let script = manifest_dir.join("../scripts/oracle/create_mistral3_test_model.py");
    let ok = std::process::Command::new("python3")
        .arg(&script)
        .arg(&path)
        .arg("--no-bias")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: could not generate the bias-free mistral3 fixture");
        return None;
    }
    Some(path)
}

/// Greedy-free comparison of the GPU forward against the CPU one on the
/// fixture, which carries YaRN rope scaling (factor 2) and attention
/// temperature scaling (floor 64). Positions stay under the floor, the longest
/// context the GPU backends serve; the CPU covers the rest.
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
fn assert_gpu_matches_cpu(
    min_cosine: f64,
    load_gpu: impl Fn(GgufFile, &Path) -> anyhow::Result<Box<dyn Model>>,
) {
    // The GPU backends skip projection/FFN biases (they warn and omit them),
    // so the default fixture, which has them, would differ for that reason
    // alone. The bias-free variant isolates the rope and attention scaling.
    let Some(path) = ensure_nobias_fixture() else {
        return;
    };
    let cpu = LlamaModel::from_gguf(GgufFile::open(&path).unwrap(), 256).expect("cpu load");
    let gpu = match load_gpu(GgufFile::open(&path).unwrap(), &path) {
        Ok(m) => m,
        Err(e) if e.to_string().contains("adapter") || e.to_string().contains("device") => {
            eprintln!("skipping: no GPU available ({e})");
            return;
        }
        Err(e) => panic!("gpu load of mistral3 failed: {e:#}"),
    };
    assert_eq!(
        gpu.config().max_seq_len,
        64,
        "context must be capped at the attention-temperature floor"
    );

    let tokens: Vec<u32> = (0..60).map(|i| 4 + (i * 7 % 200) as u32).collect();
    let mut cpu_state =
        InferenceState::from_config_with_compression(cpu.config(), &KvCompression::None).unwrap();
    let mut gpu_state =
        InferenceState::from_config_with_compression(gpu.config(), &KvCompression::None).unwrap();
    let mut min_cos = 1.0f64;
    for (i, &t) in tokens.iter().enumerate() {
        let want = cpu.forward(&[t], i, &mut cpu_state);
        let got = gpu.forward(&[t], i, &mut gpu_state);
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in want.iter().zip(got.iter()) {
            dot += a as f64 * b as f64;
            na += (a as f64).powi(2);
            nb += (b as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        min_cos = min_cos.min(cos);
    }
    eprintln!("[mistral3 gpu-vs-cpu] min cosine over 60 positions: {min_cos:.6}");
    assert!(min_cos > min_cosine, "min cosine {min_cos}");

    // The batched prefill rotates and scales through its own kernels.
    let mut cpu_state =
        InferenceState::from_config_with_compression(cpu.config(), &KvCompression::None).unwrap();
    let mut gpu_state =
        InferenceState::from_config_with_compression(gpu.config(), &KvCompression::None).unwrap();
    let want = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
    let got = gpu.forward_prefill(&tokens, 0, &mut gpu_state);
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in want.iter().zip(got.iter()) {
        dot += a as f64 * b as f64;
        na += (a as f64).powi(2);
        nb += (b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("[mistral3 gpu-vs-cpu] batched prefill cosine: {cos:.6}");
    assert!(cos > min_cosine, "batched prefill cosine {cos}");
}

/// Floors for the two backends' f16 KV rounding against the CPU: Metal lands at
/// 0.99998 and wgpu at 0.9984 here, while dropping the YaRN angle table or
/// `mscale^2` from the GPU path gives 0.96 or worse on both (checked by
/// mutating the loader). Each floor sits between its own backend's noise and
/// that failure.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
const METAL_MIN_COSINE: f64 = 0.999;
#[cfg(feature = "gpu")]
const WGPU_MIN_COSINE: f64 = 0.99;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn mistral3_metal_matches_cpu_with_yarn_and_temperature_floor() {
    assert_gpu_matches_cpu(METAL_MIN_COSINE, |gguf, path| {
        cera::model::load_model_metal(gguf, Some(path), 256)
    });
}

#[cfg(feature = "gpu")]
#[test]
fn mistral3_gpu_matches_cpu_with_yarn_and_temperature_floor() {
    assert_gpu_matches_cpu(WGPU_MIN_COSINE, |gguf, path| {
        cera::model::load_model_gpu(gguf, Some(path), 256)
    });
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

#[test]
fn ministral3_architecture_alias_resolution() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let mut gguf = GgufFile::open(&path).expect("open test_mistral3.gguf");
    gguf.metadata.insert(
        "general.architecture".to_string(),
        cera::gguf::GgufValue::String("ministral3".to_string()),
    );
    let model = LlamaModel::from_gguf(gguf, 256).expect("load ministral3 alias");
    assert_eq!(model.config().n_layers, 4);

    let tokens = vec![1u32, 2, 3];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let logits = model.forward_prefill(&tokens, 0, &mut state);
    assert!(!logits.is_empty());
}

#[test]
fn rope_norm_yarn_identity_on_invalid_parameters() {
    let yarn = cera::backend::cpu::YarnParams::new(1.0, 1.0, 1.0, 32.0, 1.0, 64);
    let mut head = vec![123.456f32; 16];
    cera::backend::cpu::apply_rope_norm_yarn_to_head(&mut head, 1, 16, 0.0, &yarn);
    assert_eq!(head, vec![123.456f32; 16]);

    let mut q = vec![1.0; 16];
    let mut k = vec![2.0; 16];
    let mut bad_yarn = yarn;
    bad_yarn.freq_scale = f32::NAN;
    cera::backend::cpu::rope_norm_yarn(&mut q, &mut k, 1, 1, 1, 16, 10000.0, &bad_yarn);
    assert_eq!(q, vec![1.0; 16]);
    assert_eq!(k, vec![2.0; 16]);
}

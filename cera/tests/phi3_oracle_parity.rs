//! Cross-validation test for Phi-3 / Phi-4-mini architecture in Cera.

#![cfg(feature = "mmap")]

use std::path::Path;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCacheConfig, KvCompression, KvPrefixCache};
use cera::model::Model;
use cera::model::llama::LlamaModel;

fn ensure_test_fixture() -> Option<std::path::PathBuf> {
    static FIXTURE: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            let target_dir = manifest_dir.join("../target/tmp/cera_test_phi3");
            let _ = std::fs::create_dir_all(&target_dir);
            let path = target_dir.join("test_phi3.gguf");
            if path
                .symlink_metadata()
                .map(|m| !m.file_type().is_symlink() && m.len() > 1024)
                .unwrap_or(false)
            {
                return Some(path);
            }
            let script = manifest_dir.join("../scripts/oracle/create_phi3_test_model.py");
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
            eprintln!("skipping: python3/python failed to generate test_phi3.gguf fixture");
            let require_fixture = std::env::var("CERA_REQUIRE_MODEL").as_deref() == Ok("1")
                || std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
            if require_fixture
                || (python_has_deps && matches!(std::env::var("CI").as_deref(), Ok("true" | "1")))
            {
                panic!("test_phi3.gguf generation failed on CI runner despite python dependencies being present");
            }
            None
        })
        .clone()
}

fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector lengths must match");
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        norm_a += (x as f64).powi(2);
        norm_b += (y as f64).powi(2);
    }
    assert!(
        norm_a > 0.0 && norm_b > 0.0,
        "vector norms must be positive"
    );
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[test]
fn phi3_load_model_and_forward_prefill_and_decode_parity() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };

    // 1. Verify generic load_model dispatch
    let generic_model = cera::model::load_model(
        GgufFile::open(&path).expect("open test_phi3.gguf"),
        Some(&path),
        256,
    )
    .expect("load phi3 via load_model");
    assert_eq!(generic_model.config().architecture, "phi3");
    assert_eq!(generic_model.config().hidden_size, 64);
    assert_eq!(generic_model.config().intermediate_size, 128);
    assert_eq!(generic_model.config().n_heads, 4);
    assert_eq!(generic_model.config().n_kv_heads, 2);
    assert_eq!(generic_model.config().head_dim, 16);

    // 2. Verify LlamaModel constructor
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load phi3 model");

    let tokens = vec![88u32, 108, 105];
    let mut state_prefill =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();

    let logits_prefill = model.forward_prefill(&tokens, 0, &mut state_prefill);
    assert_eq!(logits_prefill.len(), model.config().vocab_size);

    // 3. Verify sequential single-token decode against batched prefill
    let mut state_seq =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut last_logits = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        last_logits = model.forward(&[t], pos, &mut state_seq);
    }

    assert_eq!(logits_prefill.len(), last_logits.len());
    let cosine = compute_cosine_similarity(&logits_prefill, &last_logits);
    eprintln!("[phi3] batched prefill vs sequential decode cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.9999,
        "prefill vs decode diverged: cosine = {cosine}"
    );
}

#[test]
fn phi3_f16_kv_cache_parity() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load phi3 model");

    let tokens = vec![12u32, 45, 99, 130];
    let mut state_f32 =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let mut state_f16 =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::F16).unwrap();

    let logits_f32 = model.forward_prefill(&tokens, 0, &mut state_f32);
    let logits_f16 = model.forward_prefill(&tokens, 0, &mut state_f16);

    let cosine = compute_cosine_similarity(&logits_f32, &logits_f16);
    eprintln!("[phi3] F32 vs F16 KV cache cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.9999,
        "F32 vs F16 KV cache diverged: cosine = {cosine}"
    );
}

#[test]
fn phi3_prefix_cache_roundtrip() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load phi3 model");

    let prompt = vec![10u32, 20, 30];
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    let _ = model.forward_prefill(&prompt, 0, &mut state);

    let snap = state.snapshot().expect("snapshot state");
    let mut cache = KvPrefixCache::new(KvCacheConfig::default(), model.config(), "cpu:test_phi3");
    cache.insert(&prompt, snap);

    let (hit_snap, hit_len) = cache
        .find_longest_prefix(&[prompt[0], prompt[1], prompt[2], 40])
        .expect("cache prefix hit");
    assert_eq!(hit_len, prompt.len());

    let mut restored_state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
    restored_state.restore(&hit_snap);
    assert_eq!(restored_state.seq_len, prompt.len());

    let next_token = [40u32];
    let logits_original = model.forward(&next_token, state.seq_len, &mut state);
    let logits_restored = model.forward(&next_token, restored_state.seq_len, &mut restored_state);

    let cosine = compute_cosine_similarity(&logits_original, &logits_restored);
    assert!(
        cosine > 0.999999,
        "restored state decode diverged: cosine = {cosine}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn phi3_metal_forward_runs() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let cpu_model = LlamaModel::from_gguf(gguf.clone(), 256).expect("load cpu phi3");
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

    let cosine = compute_cosine_similarity(&cpu_logits, &metal_logits);
    eprintln!("[phi3] CPU vs Metal prefill logits cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.999,
        "CPU vs Metal prefill logits diverged: cosine = {cosine}"
    );

    let token_next = [42u32];
    let cpu_dec = cpu_model.forward(&token_next, cpu_state.seq_len, &mut cpu_state);
    let metal_dec = metal_model.forward(&token_next, metal_state.seq_len, &mut metal_state);
    assert_eq!(metal_dec.len(), metal_model.config().vocab_size);
    assert_eq!(cpu_dec.len(), metal_dec.len());

    let cosine_dec = compute_cosine_similarity(&cpu_dec, &metal_dec);
    eprintln!("[phi3] CPU vs Metal decode logits cosine similarity: {cosine_dec:.6}");
    assert!(
        cosine_dec > 0.999,
        "CPU vs Metal decode logits diverged: cosine = {cosine_dec}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn phi3_gpu_forward_runs() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let cpu_model = LlamaModel::from_gguf(gguf.clone(), 256).expect("load cpu phi3");
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
    assert_eq!(model.config().architecture, "phi3");

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

    let cosine = compute_cosine_similarity(&cpu_logits, &gpu_logits);
    eprintln!("[phi3] CPU vs WebGPU prefill logits cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.999,
        "CPU vs WebGPU prefill logits diverged: cosine = {cosine}"
    );

    let token_next = [42u32];
    let cpu_dec = cpu_model.forward(&token_next, cpu_state.seq_len, &mut cpu_state);
    let gpu_dec = model.forward(&token_next, gpu_state.seq_len, &mut gpu_state);
    assert_eq!(gpu_dec.len(), model.config().vocab_size);
    assert_eq!(cpu_dec.len(), gpu_dec.len());

    let cosine_dec = compute_cosine_similarity(&cpu_dec, &gpu_dec);
    eprintln!("[phi3] CPU vs WebGPU decode logits cosine similarity: {cosine_dec:.6}");
    assert!(
        cosine_dec > 0.999,
        "CPU vs WebGPU decode logits diverged: cosine = {cosine_dec}"
    );
}

#[test]
fn phi3_empty_tokens_prefill_panics() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let model = LlamaModel::from_gguf(gguf, 256).expect("load phi3");
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
fn phi3_tokenizer_loads_and_roundtrips() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("load tokenizer from phi3 gguf");

    let text = "Hello world";
    let tokens = tokenizer.encode(text);
    assert!(!tokens.is_empty(), "encoded tokens must not be empty");
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, text, "tokenizer encode/decode must round-trip");
}

#[test]
fn phi3_fused_and_packed_dimension_validations() {
    let Some(path) = ensure_test_fixture() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open test_phi3.gguf");

    // 1. Corrupt fused QKV weight shape (mismatched rows)
    let mut bad_qkv_gguf = gguf.clone();
    if let Some(info) = bad_qkv_gguf.tensors.get_mut("blk.0.attn_qkv.weight") {
        info.shape = vec![64, 100]; // Expected 128 rows, provided 100
    }
    assert!(
        LlamaModel::from_gguf(bad_qkv_gguf, 256).is_err(),
        "mismatched fused QKV rows must be rejected"
    );

    // 2. Corrupt fused QKV weight inner dimension k
    let mut bad_qkv_k_gguf = gguf.clone();
    if let Some(info) = bad_qkv_k_gguf.tensors.get_mut("blk.0.attn_qkv.weight") {
        info.shape = vec![32, 128]; // Expected k=64, provided 32
    }
    assert!(
        LlamaModel::from_gguf(bad_qkv_k_gguf, 256).is_err(),
        "mismatched fused QKV inner dimension k must be rejected"
    );

    // 3. Corrupt packed FFN up weight shape (mismatched rows)
    let mut bad_ffn_gguf = gguf.clone();
    if let Some(info) = bad_ffn_gguf.tensors.get_mut("blk.0.ffn_up.weight") {
        info.shape = vec![64, 200]; // Expected 256 rows, provided 200
    }
    assert!(
        LlamaModel::from_gguf(bad_ffn_gguf, 256).is_err(),
        "mismatched packed FFN up rows must be rejected"
    );

    // 4. Corrupt packed FFN up weight inner dimension k
    let mut bad_ffn_k_gguf = gguf.clone();
    if let Some(info) = bad_ffn_k_gguf.tensors.get_mut("blk.0.ffn_up.weight") {
        info.shape = vec![32, 256]; // Expected k=64, provided 32
    }
    assert!(
        LlamaModel::from_gguf(bad_ffn_k_gguf, 256).is_err(),
        "mismatched packed FFN up inner dimension k must be rejected"
    );

    // 5. Corrupt attn_output dimensions
    let mut bad_attn_out_gguf = gguf.clone();
    if let Some(info) = bad_attn_out_gguf
        .tensors
        .get_mut("blk.0.attn_output.weight")
    {
        info.shape = vec![32, 64]; // Expected k=64, provided 32
    }
    assert!(
        LlamaModel::from_gguf(bad_attn_out_gguf, 256).is_err(),
        "mismatched attn_output dimensions must be rejected"
    );

    // 6. Corrupt ffn_down dimensions
    let mut bad_ffn_down_gguf = gguf.clone();
    if let Some(info) = bad_ffn_down_gguf.tensors.get_mut("blk.0.ffn_down.weight") {
        info.shape = vec![32, 64]; // Expected k=128, provided 32
    }
    assert!(
        LlamaModel::from_gguf(bad_ffn_down_gguf, 256).is_err(),
        "mismatched ffn_down dimensions must be rejected"
    );
}

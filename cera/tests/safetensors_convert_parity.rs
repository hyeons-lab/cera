//! Cross-validation test suite for SafeTensors to GGUF quantization parity.
//!
//! Verifies metadata preservation, tensor name translation, quantization fidelity,
//! and inference logit parity between Cera-converted GGUF models and reference models.

#![cfg(all(feature = "std-fs", feature = "mmap"))]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cera::convert::parity::audit_gguf_parity;
use cera::convert::pipeline::quantize_safetensors_to_gguf;
use cera::convert::quantize::TargetQuant;

struct PairedFixture {
    hf_dir: PathBuf,
    reference_gguf: PathBuf,
    arch: String,
}

fn ensure_paired_fixture(arch: &str) -> Option<PairedFixture> {
    static FIXTURES: OnceLock<std::collections::HashMap<String, Option<PairedFixture>>> =
        OnceLock::new();

    let map = FIXTURES.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        let archs = ["llama", "nanbeige", "minicpm", "gemma2", "olmo2"];
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = manifest_dir.join("../scripts/oracle/create_paired_safetensors_gguf.py");

        if !script.exists() {
            eprintln!(
                "skipping: paired fixture generator script not found at {}",
                script.display()
            );
            for a in archs {
                m.insert(a.to_string(), None);
            }
            return m;
        }

        let probe = std::process::Command::new("python3")
            .args(["-c", "import numpy, gguf, safetensors"])
            .output();
        let python_has_deps = matches!(probe, Ok(out) if out.status.success());
        if !python_has_deps {
            eprintln!("skipping: python3 lacks numpy, gguf, or safetensors dependencies");
            for a in archs {
                m.insert(a.to_string(), None);
            }
            return m;
        }

        for a in archs {
            let target_dir = manifest_dir.join(format!("../target/tmp/cera_test_paired_{a}"));
            let hf_dir = target_dir.join("hf");
            let ref_gguf = target_dir.join("reference.gguf");

            let need_generate = !hf_dir.join("model.safetensors").exists()
                || !hf_dir.join("config.json").exists()
                || !ref_gguf.exists();

            if need_generate {
                let _ = std::fs::create_dir_all(&target_dir);
                let status = std::process::Command::new("python3")
                    .arg(&script)
                    .arg("--out-dir")
                    .arg(&target_dir)
                    .arg("--arch")
                    .arg(a)
                    .status();
                if !matches!(status, Ok(st) if st.success()) {
                    eprintln!("failed to generate paired fixture for arch {a}");
                    m.insert(a.to_string(), None);
                    continue;
                }
            }

            m.insert(
                a.to_string(),
                Some(PairedFixture {
                    hf_dir,
                    reference_gguf: ref_gguf,
                    arch: a.to_string(),
                }),
            );
        }

        m
    });

    map.get(arch).and_then(|f| {
        f.as_ref().map(|fixture| PairedFixture {
            hf_dir: fixture.hf_dir.clone(),
            reference_gguf: fixture.reference_gguf.clone(),
            arch: fixture.arch.clone(),
        })
    })
}

#[test]
fn test_llama_safetensors_f32_parity() {
    let Some(fixture) = ensure_paired_fixture("llama") else {
        eprintln!("skipping test_llama_safetensors_f32_parity: python fixtures not available");
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_gguf =
        manifest_dir.join("../target/tmp/cera_test_paired_llama/cera_converted_f32.gguf");

    quantize_safetensors_to_gguf(&fixture.hf_dir, &out_gguf, TargetQuant::F32)
        .expect("SafeTensors F32 conversion must succeed");

    let report = audit_gguf_parity(
        &out_gguf,
        &fixture.reference_gguf,
        Some("Hello, world!"),
        None,
    )
    .expect("parity audit must succeed");

    assert_eq!(
        report.metadata_diff.mismatch_count, 0,
        "metadata mismatches: {:?}",
        report.metadata_diff.keys
    );
    assert_eq!(
        report.metadata_diff.missing_in_cera_count, 0,
        "missing metadata keys in Cera: {:?}",
        report.metadata_diff.keys
    );

    assert_eq!(report.total_tensors_compared, 21);
    assert!(
        report.min_cosine_similarity >= 0.999999,
        "min cosine similarity too low: {}",
        report.min_cosine_similarity
    );
    assert!(
        report.mean_snr_db >= 90.0,
        "mean SNR too low: {} dB",
        report.mean_snr_db
    );

    let inf = report
        .inference_parity
        .as_ref()
        .expect("inference parity result must be present");
    assert!(
        inf.logit_cosine_similarity >= 0.999999,
        "inference logit similarity too low: {}",
        inf.logit_cosine_similarity
    );
    assert_eq!(
        inf.top_k_overlap, 5,
        "top-k tokens should agree completely on F32 identical weights"
    );
}

#[test]
fn test_llama_safetensors_q4_0_parity() {
    let Some(fixture) = ensure_paired_fixture("llama") else {
        eprintln!("skipping test_llama_safetensors_q4_0_parity: python fixtures not available");
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_gguf =
        manifest_dir.join("../target/tmp/cera_test_paired_llama/cera_converted_q4_0.gguf");

    quantize_safetensors_to_gguf(&fixture.hf_dir, &out_gguf, TargetQuant::Q4_0)
        .expect("SafeTensors Q4_0 conversion must succeed");

    let report = audit_gguf_parity(
        &out_gguf,
        &fixture.reference_gguf,
        Some("Hello, world!"),
        None,
    )
    .expect("parity audit must succeed");

    assert_eq!(report.total_tensors_compared, 21);
    assert!(
        report.min_cosine_similarity >= 0.990,
        "Q4_0 min weight cosine similarity too low: {}",
        report.min_cosine_similarity
    );
    assert!(
        report.mean_cosine_similarity >= 0.995,
        "Q4_0 mean weight cosine similarity too low: {}",
        report.mean_cosine_similarity
    );

    let inf = report
        .inference_parity
        .as_ref()
        .expect("inference parity result must be present");
    assert!(
        inf.logit_cosine_similarity >= 0.950,
        "Q4_0 inference logit similarity too low: {}",
        inf.logit_cosine_similarity
    );
    assert!(
        inf.top_k_overlap >= 3,
        "top-k overlap should be >= 3 under 4-bit quantization, got {}",
        inf.top_k_overlap
    );
}

#[test]
fn test_nanbeige_safetensors_f32_parity() {
    let Some(fixture) = ensure_paired_fixture("nanbeige") else {
        eprintln!("skipping test_nanbeige_safetensors_f32_parity: python fixtures not available");
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_gguf =
        manifest_dir.join("../target/tmp/cera_test_paired_nanbeige/cera_converted_f32.gguf");

    quantize_safetensors_to_gguf(&fixture.hf_dir, &out_gguf, TargetQuant::F32)
        .expect("SafeTensors Nanbeige F32 conversion must succeed");

    let report = audit_gguf_parity(
        &out_gguf,
        &fixture.reference_gguf,
        Some("Hello, world!"),
        None,
    )
    .expect("parity audit must succeed");

    assert_eq!(
        report.metadata_diff.mismatch_count, 0,
        "metadata mismatches: {:?}",
        report.metadata_diff.keys
    );
    assert_eq!(
        report.metadata_diff.missing_in_cera_count, 0,
        "missing metadata keys in Cera: {:?}",
        report.metadata_diff.keys
    );

    assert_eq!(report.total_tensors_compared, 21);
    assert!(
        report.min_cosine_similarity >= 0.999999,
        "min cosine similarity too low: {}",
        report.min_cosine_similarity
    );

    let inf = report
        .inference_parity
        .as_ref()
        .expect("inference parity result must be present");
    assert!(
        inf.logit_cosine_similarity >= 0.999999,
        "inference logit similarity too low: {}",
        inf.logit_cosine_similarity
    );
    assert_eq!(inf.top_k_overlap, 5);
}

#[test]
fn test_minicpm_safetensors_f32_parity() {
    let Some(fixture) = ensure_paired_fixture("minicpm") else {
        eprintln!("skipping test_minicpm_safetensors_f32_parity: python fixtures not available");
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_gguf =
        manifest_dir.join("../target/tmp/cera_test_paired_minicpm/cera_converted_f32.gguf");

    quantize_safetensors_to_gguf(&fixture.hf_dir, &out_gguf, TargetQuant::F32)
        .expect("SafeTensors MiniCPM F32 conversion must succeed");

    let report = audit_gguf_parity(
        &out_gguf,
        &fixture.reference_gguf,
        Some("Hello, world!"),
        None,
    )
    .expect("parity audit must succeed");

    assert_eq!(
        report.metadata_diff.mismatch_count, 0,
        "metadata mismatches: {:?}",
        report.metadata_diff.keys
    );
    assert_eq!(
        report.metadata_diff.missing_in_cera_count, 0,
        "missing metadata keys in Cera: {:?}",
        report.metadata_diff.keys
    );

    assert_eq!(report.total_tensors_compared, 21);
    assert!(
        report.min_cosine_similarity >= 0.999999,
        "min cosine similarity too low: {}",
        report.min_cosine_similarity
    );

    let inf = report
        .inference_parity
        .as_ref()
        .expect("inference parity result must be present");
    assert!(
        inf.logit_cosine_similarity >= 0.999999,
        "inference logit similarity too low: {}",
        inf.logit_cosine_similarity
    );
    assert_eq!(inf.top_k_overlap, 5);
}

#[test]
fn test_gemma2_safetensors_f32_parity() {
    let Some(fixture) = ensure_paired_fixture("gemma2") else {
        eprintln!("skipping test_gemma2_safetensors_f32_parity: python fixtures not available");
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_gguf =
        manifest_dir.join("../target/tmp/cera_test_paired_gemma2/cera_converted_f32.gguf");

    quantize_safetensors_to_gguf(&fixture.hf_dir, &out_gguf, TargetQuant::F32)
        .expect("SafeTensors Gemma2 F32 conversion must succeed");

    let report = audit_gguf_parity(
        &out_gguf,
        &fixture.reference_gguf,
        Some("Hello, world!"),
        None,
    )
    .expect("parity audit must succeed");

    assert_eq!(
        report.metadata_diff.mismatch_count, 0,
        "metadata mismatches: {:?}",
        report.metadata_diff.keys
    );
    assert_eq!(
        report.metadata_diff.missing_in_cera_count, 0,
        "missing metadata keys in Cera: {:?}",
        report.metadata_diff.keys
    );

    assert_eq!(report.total_tensors_compared, 25);
    assert!(
        report.min_cosine_similarity >= 0.999999,
        "min cosine similarity too low: {}",
        report.min_cosine_similarity
    );

    let inf = report
        .inference_parity
        .as_ref()
        .expect("inference parity result must be present");
    assert!(
        inf.logit_cosine_similarity >= 0.999999,
        "inference logit similarity too low: {}",
        inf.logit_cosine_similarity
    );
    assert_eq!(inf.top_k_overlap, 5);
}

#[test]
fn test_olmo2_safetensors_f32_parity() {
    let Some(fixture) = ensure_paired_fixture("olmo2") else {
        eprintln!("skipping test_olmo2_safetensors_f32_parity: python fixtures not available");
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_gguf =
        manifest_dir.join("../target/tmp/cera_test_paired_olmo2/cera_converted_f32.gguf");

    quantize_safetensors_to_gguf(&fixture.hf_dir, &out_gguf, TargetQuant::F32)
        .expect("SafeTensors OLMo2 F32 conversion must succeed");

    let report = audit_gguf_parity(
        &out_gguf,
        &fixture.reference_gguf,
        Some("Hello, world!"),
        None,
    )
    .expect("parity audit must succeed");

    assert_eq!(
        report.metadata_diff.mismatch_count, 0,
        "metadata mismatches: {:?}",
        report.metadata_diff.keys
    );
    assert_eq!(
        report.metadata_diff.missing_in_cera_count, 0,
        "missing metadata keys in Cera: {:?}",
        report.metadata_diff.keys
    );

    assert_eq!(report.total_tensors_compared, 21);
    assert!(
        report.min_cosine_similarity >= 0.999999,
        "min cosine similarity too low: {}",
        report.min_cosine_similarity
    );

    let inf = report
        .inference_parity
        .as_ref()
        .expect("inference parity result must be present");
    assert!(
        inf.logit_cosine_similarity >= 0.999999,
        "inference logit similarity too low: {}",
        inf.logit_cosine_similarity
    );
    assert_eq!(inf.top_k_overlap, 5);
}

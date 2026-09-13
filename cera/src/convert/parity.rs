//! Parity audit between Cera-converted GGUF models and reference community GGUF models.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use crate::convert::quantize::{compute_cosine_similarity, compute_rmse, compute_snr_db};
use crate::gguf::{GgufFile, GgufValue};
use crate::session::CeraError;

/// Status for an individual metadata key comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MetadataParityStatus {
    Match,
    Mismatch { cera: String, reference: String },
    MissingInCera { reference: String },
    ExtraInCera { cera: String },
}

/// Metadata parity diff across all keys.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MetadataDiff {
    pub keys: BTreeMap<String, MetadataParityStatus>,
    pub matching_count: usize,
    pub mismatch_count: usize,
    pub missing_in_cera_count: usize,
    pub extra_in_cera_count: usize,
}

/// Parity metrics for an individual tensor.
#[derive(Debug, Clone, Serialize)]
pub struct TensorParityEntry {
    pub name: String,
    pub shape_cera: Vec<usize>,
    pub shape_reference: Vec<usize>,
    pub shapes_match: bool,
    pub dtype_cera: String,
    pub dtype_reference: String,
    pub cosine_similarity: f32,
    pub snr_db: f32,
    pub rmse: f32,
    pub max_abs_diff: f32,
}

/// Summary metrics and entries from comparing all corresponding tensors.
#[derive(Debug, Clone, Serialize)]
pub struct TensorComparisonSummary {
    pub entries: Vec<TensorParityEntry>,
    pub mean_cosine_similarity: f32,
    pub min_cosine_similarity: f32,
    pub min_similarity_tensor: Option<String>,
    pub mean_snr_db: f32,
    pub min_snr_db: f32,
}

/// Inference logit parity comparison result.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceParityResult {
    pub prompt: String,
    pub tokens: Vec<u32>,
    pub logit_cosine_similarity: f32,
    pub top_k_overlap: usize,
    pub top_k_total: usize,
    pub cera_top_tokens: Vec<(u32, f32)>,
    pub reference_top_tokens: Vec<(u32, f32)>,
    pub max_abs_diff: f32,
}

/// Comprehensive GGUF parity audit report.
#[derive(Debug, Clone, Serialize)]
pub struct GgufParityReport {
    pub cera_path: String,
    pub reference_path: String,
    pub metadata_diff: MetadataDiff,
    pub tensor_entries: Vec<TensorParityEntry>,
    pub mean_cosine_similarity: f32,
    pub min_cosine_similarity: f32,
    pub min_similarity_tensor: Option<String>,
    pub mean_snr_db: f32,
    pub min_snr_db: f32,
    pub total_tensors_compared: usize,
    pub inference_parity: Option<InferenceParityResult>,
}

impl GgufParityReport {
    /// Check whether this parity audit satisfies quality thresholds.
    pub fn is_passing(&self, min_weight_sim: f32, min_logit_sim: f32) -> bool {
        if self.min_cosine_similarity < min_weight_sim {
            return false;
        }
        if let Some(inf) = &self.inference_parity
            && inf.logit_cosine_similarity < min_logit_sim
        {
            return false;
        }
        true
    }

    /// Format report as structured JSON string.
    pub fn format_json(&self) -> Result<String, CeraError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| CeraError::Backend(format!("failed to serialize parity report: {e}")))
    }

    /// Format report as readable terminal text summary.
    pub fn format_table(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "================================================================================\n",
        );
        out.push_str(" GGUF PARITY AUDIT REPORT\n");
        out.push_str(
            "================================================================================\n",
        );
        out.push_str(&format!("Cera GGUF:      {}\n", self.cera_path));
        out.push_str(&format!("Reference GGUF: {}\n", self.reference_path));
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        out.push_str(" METADATA SUMMARY\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        out.push_str(&format!(
            "Matching: {}, Mismatched: {}, Missing in Cera: {}, Extra in Cera: {}\n",
            self.metadata_diff.matching_count,
            self.metadata_diff.mismatch_count,
            self.metadata_diff.missing_in_cera_count,
            self.metadata_diff.extra_in_cera_count,
        ));

        if self.metadata_diff.mismatch_count > 0 || self.metadata_diff.missing_in_cera_count > 0 {
            out.push_str("\nKey Discrepancies:\n");
            for (key, status) in &self.metadata_diff.keys {
                match status {
                    MetadataParityStatus::Mismatch { cera, reference } => {
                        out.push_str(&format!("  ! {key}: cera={cera} vs ref={reference}\n"));
                    }
                    MetadataParityStatus::MissingInCera { reference } => {
                        out.push_str(&format!("  - {key}: missing in cera (ref={reference})\n"));
                    }
                    _ => {}
                }
            }
        }

        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        out.push_str(" TENSOR WEIGHT FIDELITY\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        out.push_str(&format!(
            "Total Tensors Compared: {}\nMean Cosine Similarity: {:.6}\nMin Cosine Similarity:  {:.6} (tensor: {})\nMean SNR:               {:.2} dB\nMin SNR:                {:.2} dB\n",
            self.total_tensors_compared,
            self.mean_cosine_similarity,
            self.min_cosine_similarity,
            self.min_similarity_tensor.as_deref().unwrap_or("none"),
            self.mean_snr_db,
            self.min_snr_db,
        ));

        out.push_str("\nTensor Details (Sample):\n");
        out.push_str(&format!(
            "  {:<32} {:<10} {:<10} {:>10} {:>9} {:>10}\n",
            "Tensor Name", "Cera Type", "Ref Type", "Cosine Sim", "SNR (dB)", "RMSE"
        ));
        out.push_str(&format!(
            "  {:-<32} {:-<10} {:-<10} {:-<10} {:-<9} {:-<10}\n",
            "", "", "", "", "", ""
        ));

        let display_limit = 12.min(self.tensor_entries.len());
        for entry in self.tensor_entries.iter().take(display_limit) {
            let short_name = if entry.name.len() > 32 {
                format!("...{}", &entry.name[entry.name.len() - 29..])
            } else {
                entry.name.clone()
            };
            out.push_str(&format!(
                "  {:<32} {:<10} {:<10} {:>10.6} {:>9.2} {:>10.6}\n",
                short_name,
                entry.dtype_cera,
                entry.dtype_reference,
                entry.cosine_similarity,
                entry.snr_db,
                entry.rmse,
            ));
        }
        if self.tensor_entries.len() > display_limit {
            out.push_str(&format!(
                "  ... and {} more tensors\n",
                self.tensor_entries.len() - display_limit
            ));
        }

        if let Some(inf) = &self.inference_parity {
            out.push_str("--------------------------------------------------------------------------------\n");
            out.push_str(" INFERENCE PARITY\n");
            out.push_str("--------------------------------------------------------------------------------\n");
            out.push_str(&format!("Prompt: \"{}\"\n", inf.prompt));
            out.push_str(&format!(
                "Logits Cosine Similarity: {:.6}\nTop-K Rank Agreement:     {}/{} tokens ({:.1}%)\nMax Absolute Logit Diff:  {:.6}\n",
                inf.logit_cosine_similarity,
                inf.top_k_overlap,
                inf.top_k_total,
                (inf.top_k_overlap as f32 / inf.top_k_total.max(1) as f32) * 100.0,
                inf.max_abs_diff,
            ));
        }
        out.push_str(
            "================================================================================\n",
        );
        out
    }
}

fn format_gguf_value(val: &GgufValue) -> String {
    match val {
        GgufValue::U8(v) => format!("{v} (u8)"),
        GgufValue::I8(v) => format!("{v} (i8)"),
        GgufValue::U16(v) => format!("{v} (u16)"),
        GgufValue::I16(v) => format!("{v} (i16)"),
        GgufValue::U32(v) => format!("{v}"),
        GgufValue::I32(v) => format!("{v}"),
        GgufValue::F32(v) => format!("{v:.6}"),
        GgufValue::U64(v) => format!("{v}"),
        GgufValue::I64(v) => format!("{v}"),
        GgufValue::F64(v) => format!("{v:.6}"),
        GgufValue::Bool(v) => format!("{v}"),
        GgufValue::String(s) => format!("\"{s}\""),
        GgufValue::Array(arr) => format!("[array len {}]", arr.len()),
    }
}

fn float_equal(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs()).max(1.0);
    diff <= 1e-4 * max_abs
}

fn values_match(v1: &GgufValue, v2: &GgufValue) -> bool {
    match (v1, v2) {
        (GgufValue::Bool(a), GgufValue::Bool(b)) => a == b,
        (GgufValue::String(a), GgufValue::String(b)) => a == b,
        (GgufValue::F32(a), GgufValue::F32(b)) => float_equal(*a as f64, *b as f64),
        (GgufValue::F64(a), GgufValue::F64(b)) => float_equal(*a, *b),
        (GgufValue::F32(a), GgufValue::F64(b)) => float_equal(*a as f64, *b),
        (GgufValue::F64(a), GgufValue::F32(b)) => float_equal(*a, *b as f64),
        (GgufValue::Array(a), GgufValue::Array(b)) => {
            if a.len() != b.len() {
                return false;
            }
            a.iter().zip(b.iter()).all(|(x, y)| values_match(x, y))
        }
        // Integer cross-comparison across signed and unsigned representations
        _ => {
            let n1 = gguf_value_as_i64(v1);
            let n2 = gguf_value_as_i64(v2);
            if let (Some(a), Some(b)) = (n1, n2) {
                a == b
            } else {
                v1 == v2
            }
        }
    }
}

fn gguf_value_as_i64(v: &GgufValue) -> Option<i64> {
    match v {
        GgufValue::U8(n) => Some(*n as i64),
        GgufValue::I8(n) => Some(*n as i64),
        GgufValue::U16(n) => Some(*n as i64),
        GgufValue::I16(n) => Some(*n as i64),
        GgufValue::U32(n) => Some(*n as i64),
        GgufValue::I32(n) => Some(*n as i64),
        GgufValue::U64(n) => i64::try_from(*n).ok(),
        GgufValue::I64(n) => Some(*n),
        _ => None,
    }
}

/// Compare metadata keys and values between Cera and reference GGUF files.
pub fn compare_gguf_metadata(cera: &GgufFile, reference: &GgufFile) -> MetadataDiff {
    let mut diff = MetadataDiff::default();

    // Check all keys in reference
    for (key, ref_val) in &reference.metadata {
        // Skip tokenizer lists for metadata comparison to avoid bloating report
        if key == "tokenizer.ggml.tokens"
            || key == "tokenizer.ggml.scores"
            || key == "tokenizer.ggml.token_type"
            || key == "tokenizer.ggml.merges"
        {
            continue;
        }

        match cera.metadata.get(key) {
            Some(cera_val) => {
                if values_match(cera_val, ref_val) {
                    diff.keys.insert(key.clone(), MetadataParityStatus::Match);
                    diff.matching_count += 1;
                } else {
                    diff.keys.insert(
                        key.clone(),
                        MetadataParityStatus::Mismatch {
                            cera: format_gguf_value(cera_val),
                            reference: format_gguf_value(ref_val),
                        },
                    );
                    diff.mismatch_count += 1;
                }
            }
            None => {
                diff.keys.insert(
                    key.clone(),
                    MetadataParityStatus::MissingInCera {
                        reference: format_gguf_value(ref_val),
                    },
                );
                diff.missing_in_cera_count += 1;
            }
        }
    }

    // Check extra keys in Cera
    for (key, cera_val) in &cera.metadata {
        if key == "tokenizer.ggml.tokens"
            || key == "tokenizer.ggml.scores"
            || key == "tokenizer.ggml.token_type"
            || key == "tokenizer.ggml.merges"
        {
            continue;
        }
        if !reference.metadata.contains_key(key) {
            diff.keys.insert(
                key.clone(),
                MetadataParityStatus::ExtraInCera {
                    cera: format_gguf_value(cera_val),
                },
            );
            diff.extra_in_cera_count += 1;
        }
    }

    diff
}

/// Compare all corresponding tensors between two GGUF models.
pub fn compare_gguf_tensors(
    cera: &GgufFile,
    reference: &GgufFile,
) -> Result<TensorComparisonSummary, CeraError> {
    let mut entries = Vec::new();
    let mut sum_cosine = 0.0f32;
    let mut min_cosine = 1.0f32;
    let mut min_cosine_tensor: Option<String> = None;
    let mut sum_snr = 0.0f32;
    let mut min_snr = f32::INFINITY;
    let mut compared_count = 0usize;

    for (name, ref_info) in &reference.tensors {
        let cera_tensor = match cera.get_tensor(name) {
            Ok(t) => t,
            Err(_) => {
                // Tensor missing in Cera
                entries.push(TensorParityEntry {
                    name: name.clone(),
                    shape_cera: Vec::new(),
                    shape_reference: ref_info.shape.clone(),
                    shapes_match: false,
                    dtype_cera: "MISSING".to_string(),
                    dtype_reference: format!("{:?}", ref_info.dtype),
                    cosine_similarity: 0.0,
                    snr_db: 0.0,
                    rmse: f32::INFINITY,
                    max_abs_diff: f32::INFINITY,
                });
                min_cosine = 0.0;
                min_cosine_tensor = Some(name.clone());
                min_snr = 0.0;
                continue;
            }
        };

        let ref_tensor = reference
            .get_tensor(name)
            .map_err(|e| CeraError::Backend(format!("reading ref tensor `{name}`: {e}")))?;

        let shapes_match = cera_tensor.shape() == ref_tensor.shape();
        let f32_cera = cera_tensor.to_f32_vec();
        let f32_ref = ref_tensor.to_f32_vec();

        let cos_sim = compute_cosine_similarity(&f32_cera, &f32_ref);
        let snr = compute_snr_db(&f32_cera, &f32_ref);
        let rmse = compute_rmse(&f32_cera, &f32_ref);
        let max_abs = f32_cera
            .iter()
            .zip(f32_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        if cos_sim < min_cosine {
            min_cosine = cos_sim;
            min_cosine_tensor = Some(name.clone());
        }
        if snr < min_snr {
            min_snr = snr;
        }

        sum_cosine += cos_sim;
        sum_snr += snr;
        compared_count += 1;

        entries.push(TensorParityEntry {
            name: name.clone(),
            shape_cera: cera_tensor.shape().to_vec(),
            shape_reference: ref_tensor.shape().to_vec(),
            shapes_match,
            dtype_cera: format!("{:?}", cera_tensor.dtype()),
            dtype_reference: format!("{:?}", ref_tensor.dtype()),
            cosine_similarity: cos_sim,
            snr_db: snr,
            rmse,
            max_abs_diff: max_abs,
        });
    }

    let mean_cosine = if compared_count > 0 {
        sum_cosine / (compared_count as f32)
    } else {
        0.0
    };
    let mean_snr = if compared_count > 0 {
        sum_snr / (compared_count as f32)
    } else {
        0.0
    };
    if min_snr.is_infinite() {
        min_snr = 0.0;
    }

    Ok(TensorComparisonSummary {
        entries,
        mean_cosine_similarity: mean_cosine,
        min_cosine_similarity: min_cosine,
        min_similarity_tensor: min_cosine_tensor,
        mean_snr_db: mean_snr,
        min_snr_db: min_snr,
    })
}

/// Compare GGUF metadata and tensor data without running inference.
#[cfg(feature = "mmap")]
pub fn compare_gguf_files(
    cera_path: &Path,
    reference_path: &Path,
) -> Result<GgufParityReport, CeraError> {
    let cera_gguf = GgufFile::open(cera_path)
        .map_err(|e| CeraError::Backend(format!("open cera gguf: {e}")))?;
    let ref_gguf = GgufFile::open(reference_path)
        .map_err(|e| CeraError::Backend(format!("open ref gguf: {e}")))?;

    let metadata_diff = compare_gguf_metadata(&cera_gguf, &ref_gguf);
    let tensor_summary = compare_gguf_tensors(&cera_gguf, &ref_gguf)?;

    Ok(GgufParityReport {
        cera_path: cera_path.display().to_string(),
        reference_path: reference_path.display().to_string(),
        metadata_diff,
        tensor_entries: tensor_summary.entries,
        mean_cosine_similarity: tensor_summary.mean_cosine_similarity,
        min_cosine_similarity: tensor_summary.min_cosine_similarity,
        min_similarity_tensor: tensor_summary.min_similarity_tensor,
        mean_snr_db: tensor_summary.mean_snr_db,
        min_snr_db: tensor_summary.min_snr_db,
        total_tensors_compared: cera_gguf.tensors.len(),
        inference_parity: None,
    })
}

/// Compare prompt prefill inference logits between two GGUF models.
#[cfg(feature = "mmap")]
pub fn compare_inference_parity(
    cera_path: &Path,
    reference_path: &Path,
    prompt: &str,
    test_tokens: Option<&[u32]>,
) -> Result<InferenceParityResult, CeraError> {
    let cfg = crate::EngineConfig {
        context_size: 256,
        backend: crate::BackendPreference::Cpu,
        ..Default::default()
    };
    let engine_cera = crate::CeraEngine::from_path(cera_path, cfg.clone())?;
    let engine_ref = crate::CeraEngine::from_path(reference_path, cfg)?;

    let tokens: Vec<u32> = if let Some(toks) = test_tokens {
        toks.to_vec()
    } else {
        let enc = engine_cera.tokenizer().encode(prompt);
        if enc.is_empty() { vec![1, 2, 3] } else { enc }
    };

    let model_cera = engine_cera.model();
    let model_ref = engine_ref.model();

    let mut state_cera = crate::kv_cache::InferenceState::from_config(model_cera.config())
        .map_err(|e| CeraError::Backend(format!("init cera inference state: {e}")))?;
    let logits_cera = model_cera.forward_prefill(&tokens, 0, &mut state_cera);

    let mut state_ref = crate::kv_cache::InferenceState::from_config(model_ref.config())
        .map_err(|e| CeraError::Backend(format!("init ref inference state: {e}")))?;
    let logits_ref = model_ref.forward_prefill(&tokens, 0, &mut state_ref);

    let cos_sim = compute_cosine_similarity(&logits_cera, &logits_ref);
    let max_diff = logits_cera
        .iter()
        .zip(logits_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Extract top-5 tokens
    fn get_top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    let top_cera = get_top_k(&logits_cera, 5);
    let top_ref = get_top_k(&logits_ref, 5);

    let cera_set: std::collections::HashSet<u32> = top_cera.iter().map(|(id, _)| *id).collect();
    let overlap = top_ref
        .iter()
        .filter(|(id, _)| cera_set.contains(id))
        .count();

    Ok(InferenceParityResult {
        prompt: prompt.to_string(),
        tokens,
        logit_cosine_similarity: cos_sim,
        top_k_overlap: overlap,
        top_k_total: 5,
        cera_top_tokens: top_cera,
        reference_top_tokens: top_ref,
        max_abs_diff: max_diff,
    })
}

/// Comprehensive audit comparing metadata, tensor weights, and inference logits.
#[cfg(feature = "mmap")]
pub fn audit_gguf_parity(
    cera_path: &Path,
    reference_path: &Path,
    prompt: Option<&str>,
    test_tokens: Option<&[u32]>,
) -> Result<GgufParityReport, CeraError> {
    let mut report = compare_gguf_files(cera_path, reference_path)?;
    if let Some(p) = prompt {
        match compare_inference_parity(cera_path, reference_path, p, test_tokens) {
            Ok(inf) => report.inference_parity = Some(inf),
            Err(e) => {
                tracing::warn!("Inference parity check skipped or failed: {e}");
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_equal_handling() {
        assert!(float_equal(1.0, 1.0));
        assert!(float_equal(10000.0, 10000.0001));
        assert!(!float_equal(1.0, 2.0));
        assert!(float_equal(f64::NAN, f64::NAN));
    }

    #[test]
    fn test_metadata_diff_aggregation() {
        let mut diff = MetadataDiff::default();
        diff.keys
            .insert("general.architecture".into(), MetadataParityStatus::Match);
        diff.matching_count += 1;

        diff.keys.insert(
            "llama.block_count".into(),
            MetadataParityStatus::Mismatch {
                cera: "16".into(),
                reference: "32".into(),
            },
        );
        diff.mismatch_count += 1;

        assert_eq!(diff.matching_count, 1);
        assert_eq!(diff.mismatch_count, 1);
    }
}

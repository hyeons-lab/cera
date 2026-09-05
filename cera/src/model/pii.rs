//! Native Rust Hybrid PII classifier, sensitivity gate, and sliding window scanner.
//!
//! Integrates ModernBERT representations with:
//! 1. `SensitivityHead`: Mean-pooled 2-layer MLP for sub-5us conversation-level gating.
//! 2. `DynamicSpanHead`: Vectorized GLiNER span representations with runtime label cross-attention.
//! 3. `SlidingWindowScanner`: 1024-token stride windowing with greedy non-maximum suppression (NMS).

use anyhow::{Context, Result, ensure};

#[cfg(has_blas)]
use crate::backend::blas;
use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::Model;
use crate::model::bert::BertModel;
use crate::tokenizer::BpeTokenizer;

/// Compute matrix-matrix multiplication C = A * B^T + bias with optional BLAS acceleration.
///
/// Layout:
/// - A: [M, K] row-major
/// - B: [N, K] row-major (transposed to [K, N])
/// - C: [M, N] row-major
/// - bias: optional [N] added along columns
pub fn matmul_nt_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "A dimension mismatch");
    assert_eq!(b.len(), n * k, "B dimension mismatch");
    assert_eq!(c.len(), m * n, "C dimension mismatch");
    if let Some(bias_vec) = bias {
        assert_eq!(bias_vec.len(), n, "Bias dimension mismatch");
    }

    #[cfg(has_blas)]
    {
        blas::sgemm_rowmajor_nt(m, n, k, a, b, c);
        if let Some(bias_vec) = bias {
            for row in 0..m {
                cpu::add_inplace(&mut c[row * n..(row + 1) * n], bias_vec);
            }
        }
    }

    #[cfg(not(has_blas))]
    {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let b_row = &b[j * k..(j + 1) * k];
                let mut dot = 0.0f32;
                for p in 0..k {
                    dot += a_row[p] * b_row[p];
                }
                if let Some(bias_vec) = bias {
                    dot += bias_vec[j];
                }
                c[i * n + j] = dot;
            }
        }
    }
}

/// Configuration and weights for the conversation-level sensitivity gate.
#[derive(Debug, Clone)]
pub struct SensitivityHeadWeights {
    /// Dimension of hidden states (e.g. 768 for ModernBERT-base).
    pub hidden_dim: usize,
    /// Dimension of intermediate projection (e.g. 384).
    pub intermediate_dim: usize,
    /// Row-major weight matrix [intermediate_dim, hidden_dim].
    pub dense_weight: Vec<f32>,
    /// Bias vector [intermediate_dim].
    pub dense_bias: Vec<f32>,
    /// Row-major classifier weight [1, intermediate_dim].
    pub classifier_weight: Vec<f32>,
    /// Scalar classifier bias.
    pub classifier_bias: f32,
    /// Probability threshold for early exit bypass (default: 0.05).
    pub gate_threshold: f32,
}

impl SensitivityHeadWeights {
    /// Run fast sensitivity scoring on sequence representations.
    ///
    /// Returns (probability, should_bypass).
    pub fn score(&self, hidden_states: &[f32], n_tokens: usize) -> (f32, bool) {
        if n_tokens == 0 {
            return (0.0, true);
        }
        let d = self.hidden_dim;
        let inter = self.intermediate_dim;

        // 1. Mean pool token representations across sequence
        let mut mean_pool = vec![0.0f32; d];
        for t in 0..n_tokens {
            let row = &hidden_states[t * d..(t + 1) * d];
            for (m, &val) in mean_pool.iter_mut().zip(row.iter()) {
                *m += val;
            }
        }
        let inv_t = 1.0 / (n_tokens as f32);
        for m in &mut mean_pool {
            *m *= inv_t;
        }

        // 2. Dense layer: inter = dense_weight [inter, d] * mean_pool [d] + dense_bias [inter]
        let mut h1 = vec![0.0f32; inter];
        matmul_nt_f32(
            1,
            inter,
            d,
            &mean_pool,
            &self.dense_weight,
            Some(&self.dense_bias),
            &mut h1,
        );
        cpu::gelu_inplace(&mut h1);

        // 3. Classifier layer: z = classifier_weight [1, inter] * h1 [inter] + classifier_bias
        let mut dot = 0.0f32;
        for (w, h) in self.classifier_weight.iter().zip(h1.iter()) {
            dot += w * h;
        }
        let z = dot + self.classifier_bias;

        // 4. Numerically stable sigmoid: 1 / (1 + exp(-z))
        let prob = 1.0 / (1.0 + (-z).exp());
        let bypass = prob < self.gate_threshold;
        (prob, bypass)
    }
}

/// Configuration and weights for vectorized candidate span matching.
#[derive(Debug, Clone)]
pub struct DynamicSpanHeadWeights {
    /// Dimension of token hidden states (e.g. 768).
    pub hidden_dim: usize,
    /// Maximum span length in tokens (default: 24).
    pub max_span_length: usize,
    /// Span size embedding table [max_span_length + 1, size_dim] (default size_dim: 64).
    pub size_dim: usize,
    pub span_size_embedding: Vec<f32>,
    /// Span projector layer 1: [hidden_dim, 3 * hidden_dim + size_dim].
    pub proj1_weight: Vec<f32>,
    pub proj1_bias: Vec<f32>,
    /// Span projector layer 2: [hidden_dim, hidden_dim].
    pub proj2_weight: Vec<f32>,
    pub proj2_bias: Vec<f32>,
    /// Label query projector: [hidden_dim, hidden_dim].
    pub label_proj_weight: Vec<f32>,
    pub label_proj_bias: Vec<f32>,
    /// Confidence threshold for entity extraction (default: 0.45).
    pub span_threshold: f32,
}

/// Candidate span representation index.
#[derive(Debug, Clone, Copy)]
pub struct CandidateSpan {
    pub start: usize,
    pub end: usize,
    pub length: usize,
}

impl DynamicSpanHeadWeights {
    /// Generate all candidate span tuples up to max_span_length.
    pub fn candidate_spans(&self, n_tokens: usize) -> Vec<CandidateSpan> {
        let mut spans = Vec::new();
        for start in 0..n_tokens {
            let max_len = (self.max_span_length + 1).min(n_tokens - start + 1);
            for length in 1..max_len {
                let end = start + length - 1;
                spans.push(CandidateSpan { start, end, length });
            }
        }
        spans
    }

    /// Pre-project label embeddings for cross-attention dot product matching.
    pub fn project_labels(&self, raw_label_embeddings: &[f32], n_labels: usize) -> Vec<f32> {
        let d = self.hidden_dim;
        assert_eq!(raw_label_embeddings.len(), n_labels * d);
        let mut proj = vec![0.0f32; n_labels * d];
        matmul_nt_f32(
            n_labels,
            d,
            d,
            raw_label_embeddings,
            &self.label_proj_weight,
            Some(&self.label_proj_bias),
            &mut proj,
        );
        proj
    }

    /// Run vectorized span extraction and scoring against projected label queries.
    ///
    /// Returns a list of detected (span_index, label_index, confidence_score).
    pub fn score_spans(
        &self,
        hidden_states: &[f32],
        n_tokens: usize,
        projected_labels: &[f32],
        n_labels: usize,
    ) -> Vec<(CandidateSpan, usize, f32)> {
        if n_tokens == 0 || n_labels == 0 {
            return Vec::new();
        }
        let d = self.hidden_dim;
        let s_dim = self.size_dim;
        let span_in_dim = 3 * d + s_dim;

        let candidates = self.candidate_spans(n_tokens);
        let n_spans = candidates.len();
        if n_spans == 0 {
            return Vec::new();
        }

        let mut detections = Vec::new();
        const CHUNK_SIZE: usize = 1024;
        let mut chunk_features = vec![0.0f32; CHUNK_SIZE * span_in_dim];
        let mut chunk_h_proj = vec![0.0f32; CHUNK_SIZE * d];
        let mut chunk_p_spans = vec![0.0f32; CHUNK_SIZE * d];
        let mut chunk_scores = vec![0.0f32; CHUNK_SIZE * n_labels];

        // Process candidate spans in bounded chunks to prevent large transient heap spikes
        for chunk in candidates.chunks(CHUNK_SIZE) {
            let n_chunk = chunk.len();
            let feat_slice = &mut chunk_features[..n_chunk * span_in_dim];
            for (i, cs) in chunk.iter().enumerate() {
                let out_slice = &mut feat_slice[i * span_in_dim..(i + 1) * span_in_dim];
                let h_start = &hidden_states[cs.start * d..(cs.start + 1) * d];
                let h_end = &hidden_states[cs.end * d..(cs.end + 1) * d];

                out_slice[..d].copy_from_slice(h_start);
                out_slice[d..2 * d].copy_from_slice(h_end);
                let diff_slice = &mut out_slice[2 * d..3 * d];
                for j in 0..d {
                    diff_slice[j] = h_end[j] - h_start[j];
                }
                let size_idx = cs.length.min(self.max_span_length);
                let size_emb = &self.span_size_embedding[size_idx * s_dim..(size_idx + 1) * s_dim];
                out_slice[3 * d..span_in_dim].copy_from_slice(size_emb);
            }

            // 1. Project spans: MLP 1 [n_chunk, span_in_dim] * [d, span_in_dim]^T -> [n_chunk, d]
            let h_proj_slice = &mut chunk_h_proj[..n_chunk * d];
            matmul_nt_f32(
                n_chunk,
                d,
                span_in_dim,
                feat_slice,
                &self.proj1_weight,
                Some(&self.proj1_bias),
                h_proj_slice,
            );
            cpu::gelu_inplace(h_proj_slice);

            // 2. Project spans: MLP 2 [n_chunk, d] * [d, d]^T -> [n_chunk, d]
            let p_spans_slice = &mut chunk_p_spans[..n_chunk * d];
            matmul_nt_f32(
                n_chunk,
                d,
                d,
                h_proj_slice,
                &self.proj2_weight,
                Some(&self.proj2_bias),
                p_spans_slice,
            );

            // 3. Dot product cross-attention scoring: [n_chunk, d] * [n_labels, d]^T -> [n_chunk, n_labels]
            let scores_slice = &mut chunk_scores[..n_chunk * n_labels];
            matmul_nt_f32(
                n_chunk,
                n_labels,
                d,
                p_spans_slice,
                projected_labels,
                None,
                scores_slice,
            );

            // 4. Apply sigmoid and collect thresholded candidates
            for (i, &cs) in chunk.iter().enumerate() {
                for (l_idx, &raw_score) in scores_slice[i * n_labels..(i + 1) * n_labels]
                    .iter()
                    .enumerate()
                {
                    let prob = 1.0 / (1.0 + (-raw_score).exp());
                    if prob >= self.span_threshold {
                        detections.push((cs, l_idx, prob));
                    }
                }
            }
        }
        detections
    }
}

/// A detected PII or secret entity span.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedEntity {
    /// Canonical semantic label (e.g. identity.person_name, financial.credit_card).
    pub label: String,
    /// Model confidence score [0.0, 1.0].
    pub score: f32,
    /// Token start index (inclusive).
    pub token_start: usize,
    /// Token end index (inclusive).
    pub token_end: usize,
    /// Unicode character start index (inclusive).
    pub char_start: usize,
    /// Unicode character end index (exclusive).
    pub char_end: usize,
    /// UTF-8 byte start index (inclusive).
    pub byte_start: usize,
    /// UTF-8 byte end index (exclusive).
    pub byte_end: usize,
    /// Extracted text segment.
    pub text: String,
}

/// Full Hybrid PII classifier combining ModernBERT encoder and dual heads.
pub struct HybridPiiModel {
    pub backbone: BertModel,
    pub sensitivity_head: SensitivityHeadWeights,
    pub span_head: DynamicSpanHeadWeights,
    pub labels: Vec<String>,
    pub projected_labels: Vec<f32>,
}

impl HybridPiiModel {
    /// Load Hybrid PII model directly from a GGUF file containing both backbone and heads.
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        let backbone =
            BertModel::from_gguf_with_id(gguf.clone(), 8192, "modernbert-pii".to_string())
                .context("failed to construct BertModel backbone")?;

        let hidden_dim = backbone.config().hidden_size;

        // Parse sensitivity head weights
        let sens_dense_w = gguf
            .get_tensor("sensitivity_head.dense.weight")
            .context("missing sensitivity_head.dense.weight")?
            .to_f32_vec();
        let sens_dense_b = gguf
            .get_tensor("sensitivity_head.dense.bias")
            .context("missing sensitivity_head.dense.bias")?
            .to_f32_vec();
        let sens_cls_w = gguf
            .get_tensor("sensitivity_head.classifier.weight")
            .context("missing sensitivity_head.classifier.weight")?
            .to_f32_vec();
        let sens_cls_b = gguf
            .get_tensor("sensitivity_head.classifier.bias")
            .context("missing sensitivity_head.classifier.bias")?
            .to_f32_vec()
            .first()
            .copied()
            .context("empty sensitivity_head.classifier.bias")?;

        ensure!(hidden_dim > 0, "hidden_dim must be positive");
        ensure!(
            sens_dense_w.len() % hidden_dim == 0,
            "sens_dense_w shape mismatch"
        );
        let intermediate_dim = sens_dense_w.len() / hidden_dim;
        ensure!(
            sens_dense_b.len() == intermediate_dim,
            "sens_dense_b length mismatch"
        );
        ensure!(
            sens_cls_w.len() == intermediate_dim,
            "sens_cls_w length mismatch"
        );
        let gate_threshold = gguf.get_f32("hybrid_pii.gate_threshold").unwrap_or(0.05);

        let sensitivity_head = SensitivityHeadWeights {
            hidden_dim,
            intermediate_dim,
            dense_weight: sens_dense_w,
            dense_bias: sens_dense_b,
            classifier_weight: sens_cls_w,
            classifier_bias: sens_cls_b,
            gate_threshold,
        };

        // Parse span head weights
        let max_span_length = gguf.get_u32("hybrid_pii.max_span_length").unwrap_or(24) as usize;
        let span_threshold = gguf.get_f32("hybrid_pii.span_threshold").unwrap_or(0.45);

        let span_size_emb = gguf
            .get_tensor("span_head.span_size_embedding.weight")
            .context("missing span_head.span_size_embedding.weight")?
            .to_f32_vec();
        ensure!(
            span_size_emb.len() % (max_span_length + 1) == 0,
            "span_size_embedding shape mismatch"
        );
        let size_dim = span_size_emb.len() / (max_span_length + 1);
        let span_in_dim = 3 * hidden_dim + size_dim;

        let proj1_w = gguf
            .get_tensor("span_head.span_projector.0.weight")
            .context("missing span_head.span_projector.0.weight")?
            .to_f32_vec();
        let proj1_b = gguf
            .get_tensor("span_head.span_projector.0.bias")
            .context("missing span_head.span_projector.0.bias")?
            .to_f32_vec();
        let proj2_w = gguf
            .get_tensor("span_head.span_projector.2.weight")
            .context("missing span_head.span_projector.2.weight")?
            .to_f32_vec();
        let proj2_b = gguf
            .get_tensor("span_head.span_projector.2.bias")
            .context("missing span_head.span_projector.2.bias")?
            .to_f32_vec();

        let label_proj_w = gguf
            .get_tensor("span_head.label_projector.weight")
            .context("missing span_head.label_projector.weight")?
            .to_f32_vec();
        let label_proj_b = gguf
            .get_tensor("span_head.label_projector.bias")
            .context("missing span_head.label_projector.bias")?
            .to_f32_vec();

        ensure!(
            proj1_w.len() == hidden_dim * span_in_dim,
            "proj1_w dimension mismatch"
        );
        ensure!(proj1_b.len() == hidden_dim, "proj1_b dimension mismatch");
        ensure!(
            proj2_w.len() == hidden_dim * hidden_dim,
            "proj2_w dimension mismatch"
        );
        ensure!(proj2_b.len() == hidden_dim, "proj2_b dimension mismatch");
        ensure!(
            label_proj_w.len() == hidden_dim * hidden_dim,
            "label_proj_w dimension mismatch"
        );
        ensure!(
            label_proj_b.len() == hidden_dim,
            "label_proj_b dimension mismatch"
        );

        let span_head = DynamicSpanHeadWeights {
            hidden_dim,
            max_span_length,
            size_dim,
            span_size_embedding: span_size_emb,
            proj1_weight: proj1_w,
            proj1_bias: proj1_b,
            proj2_weight: proj2_w,
            proj2_bias: proj2_b,
            label_proj_weight: label_proj_w,
            label_proj_bias: label_proj_b,
            span_threshold,
        };

        let raw_labels = gguf.get_string_array("hybrid_pii.labels");
        let labels: Vec<String> = if let Some(arr) = raw_labels {
            arr.into_iter().map(|s| s.to_string()).collect()
        } else {
            vec![
                "identity.person_name".to_string(),
                "contact.email".to_string(),
                "contact.phone".to_string(),
                "location.address".to_string(),
                "financial.credit_card".to_string(),
                "government.ssn".to_string(),
                "credential.api_key".to_string(),
                "network.ip_address".to_string(),
            ]
        };

        // Project label queries (if pre-embedded labels exist in GGUF)
        let projected_labels = if let Ok(t) = gguf.get_tensor("hybrid_pii.label_embeddings") {
            let embs = t.to_f32_vec();
            if embs.len() == labels.len() * hidden_dim {
                span_head.project_labels(&embs, labels.len())
            } else {
                tracing::warn!(
                    "hybrid_pii.label_embeddings size mismatch, falling back to zero embeddings"
                );
                vec![0.0f32; labels.len() * hidden_dim]
            }
        } else {
            vec![0.0f32; labels.len() * hidden_dim]
        };

        Ok(Self {
            backbone,
            sensitivity_head,
            span_head,
            labels,
            projected_labels,
        })
    }
}

/// Sliding window scanner executing overlapping chunks with greedy non-maximum suppression.
pub struct SlidingWindowScanner {
    pub model: HybridPiiModel,
    pub tokenizer: BpeTokenizer,
    pub window_size: usize,
    pub stride: usize,
}

impl SlidingWindowScanner {
    /// Initialize scanner with default 1024 window and 768 stride (256 token overlap).
    pub fn new(model: HybridPiiModel, tokenizer: BpeTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            window_size: 1024,
            stride: 768,
        }
    }

    /// Scan text and return list of resolved, non-overlapping detected entities.
    pub fn scan(&self, text: &str) -> Result<Vec<DetectedEntity>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let stride = self.stride.max(1);
        let window_size = self.window_size.max(1);

        let (tokens, offsets) = self.tokenizer.encode_with_offsets(text);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let n_tokens = tokens.len();
        let mut raw_candidates: Vec<DetectedEntity> = Vec::new();
        let mut state = InferenceState::new(self.model.backbone.config().n_layers);

        // Process overlapping sliding windows
        let mut win_start = 0;
        while win_start < n_tokens {
            let win_end = (win_start + window_size).min(n_tokens);
            let win_tokens = &tokens[win_start..win_end];
            let win_len = win_tokens.len();

            // 1. Monolithic encoder prefill
            let hidden_states = self.model.backbone.hidden_states(win_tokens, &mut state);

            // 2. Sensitivity gate early exit
            let (_prob, bypass) = self.model.sensitivity_head.score(&hidden_states, win_len);
            if !bypass {
                // 3. Dynamic span head scoring
                let win_detections = self.model.span_head.score_spans(
                    &hidden_states,
                    win_len,
                    &self.model.projected_labels,
                    self.model.labels.len(),
                );

                for (cs, l_idx, score) in win_detections {
                    let global_tok_start = win_start + cs.start;
                    let global_tok_end = win_start + cs.end;

                    let off_start = offsets[global_tok_start];
                    let off_end = offsets[global_tok_end];

                    let char_start = off_start.char_start;
                    let char_end = off_end.char_end;
                    let byte_start = off_start.byte_start;
                    let byte_end = off_end.byte_end;

                    if byte_start < byte_end
                        && let Some(span_text) = text.get(byte_start..byte_end)
                    {
                        raw_candidates.push(DetectedEntity {
                            label: self.model.labels[l_idx].clone(),
                            score,
                            token_start: global_tok_start,
                            token_end: global_tok_end,
                            char_start,
                            char_end,
                            byte_start,
                            byte_end,
                            text: span_text.to_string(),
                        });
                    }
                }
            }

            if win_end == n_tokens {
                break;
            }
            win_start += stride;
        }

        // 4. Greedy Non-Maximum Suppression (NMS)
        // Sort descending by confidence score
        raw_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut accepted: Vec<DetectedEntity> = Vec::new();
        for cand in raw_candidates {
            let overlaps = accepted
                .iter()
                .any(|acc| !(cand.byte_end <= acc.byte_start || cand.byte_start >= acc.byte_end));
            if !overlaps {
                accepted.push(cand);
            }
        }

        // Sort by position ascending for deterministic consumers
        accepted.sort_by_key(|e| e.byte_start);
        Ok(accepted)
    }

    /// Redact detected entities right-to-left with placeholder tags.
    pub fn redact(&self, text: &str) -> Result<String> {
        let entities = self.scan(text)?;
        if entities.is_empty() {
            return Ok(text.to_string());
        }

        let mut redacted = text.to_string();
        // Redact in reverse byte order to keep prior offsets valid
        for ent in entities.into_iter().rev() {
            if text.is_char_boundary(ent.byte_start) && text.is_char_boundary(ent.byte_end) {
                let tag = format!("[REDACTED_{}]", ent.label.to_uppercase().replace('.', "_"));
                redacted.replace_range(ent.byte_start..ent.byte_end, &tag);
            }
        }
        Ok(redacted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_head_early_exit_logic() {
        let hidden_dim = 4;
        let intermediate_dim = 2;
        let dense_weight = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let dense_bias = vec![0.0, 0.0];
        let classifier_weight = vec![1.0, 1.0];
        let classifier_bias = -10.0; // very negative bias -> near zero probability

        let head = SensitivityHeadWeights {
            hidden_dim,
            intermediate_dim,
            dense_weight,
            dense_bias,
            classifier_weight,
            classifier_bias,
            gate_threshold: 0.05,
        };

        let hidden_states = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let (prob, bypass) = head.score(&hidden_states, 2);
        assert!(prob < 0.01);
        assert!(bypass);
    }

    #[test]
    fn test_dynamic_span_head_candidate_spans() {
        let head = DynamicSpanHeadWeights {
            hidden_dim: 4,
            max_span_length: 3,
            size_dim: 2,
            span_size_embedding: vec![0.0; 8],
            proj1_weight: vec![0.0; 4 * (3 * 4 + 2)],
            proj1_bias: vec![0.0; 4],
            proj2_weight: vec![0.0; 16],
            proj2_bias: vec![0.0; 4],
            label_proj_weight: vec![0.0; 16],
            label_proj_bias: vec![0.0; 4],
            span_threshold: 0.5,
        };

        // For 4 tokens and max_span_length 3:
        // start 0: len 1, 2, 3 -> (0,0), (0,1), (0,2)
        // start 1: len 1, 2, 3 -> (1,1), (1,2), (1,3)
        // start 2: len 1, 2    -> (2,2), (2,3)
        // start 3: len 1       -> (3,3)
        // Total = 3 + 3 + 2 + 1 = 9 spans.
        let spans = head.candidate_spans(4);
        assert_eq!(spans.len(), 9);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 0);
        assert_eq!(spans[0].length, 1);
        assert_eq!(spans[8].start, 3);
        assert_eq!(spans[8].end, 3);
        assert_eq!(spans[8].length, 1);
    }

    #[test]
    fn test_matmul_nt_f32_identity_and_bias() {
        // A is [2, 2]: [[1, 2], [3, 4]]
        // B is [2, 2]: [[1, 0], [0, 1]] (Identity) -> B^T is Identity
        // bias is [2]: [10, 20]
        // C should be [[1+10, 2+20], [3+10, 4+20]] = [[11, 22], [13, 24]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![10.0, 20.0];
        let mut c = vec![0.0; 4];

        matmul_nt_f32(2, 2, 2, &a, &b, Some(&bias), &mut c);
        assert_eq!(c, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_nms_and_redaction_logic() {
        let text = "Contact Alice Smith at alice@example.com for details.";
        let raw_candidates = vec![
            DetectedEntity {
                label: "identity.person_name".to_string(),
                score: 0.95,
                token_start: 1,
                token_end: 2,
                char_start: 8,
                char_end: 19,
                byte_start: 8,
                byte_end: 19,
                text: "Alice Smith".to_string(),
            },
            DetectedEntity {
                label: "identity.person_name".to_string(),
                score: 0.70,
                token_start: 2,
                token_end: 2,
                char_start: 14,
                char_end: 19,
                byte_start: 14,
                byte_end: 19,
                text: "Smith".to_string(),
            },
            DetectedEntity {
                label: "contact.email".to_string(),
                score: 0.99,
                token_start: 4,
                token_end: 4,
                char_start: 23,
                char_end: 40,
                byte_start: 23,
                byte_end: 40,
                text: "alice@example.com".to_string(),
            },
        ];

        // Greedy NMS
        let mut sorted = raw_candidates;
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut accepted: Vec<DetectedEntity> = Vec::new();
        for cand in sorted {
            let overlaps = accepted
                .iter()
                .any(|acc| !(cand.byte_end <= acc.byte_start || cand.byte_start >= acc.byte_end));
            if !overlaps {
                accepted.push(cand);
            }
        }
        accepted.sort_by_key(|e| e.byte_start);

        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[0].text, "Alice Smith");
        assert_eq!(accepted[1].text, "alice@example.com");

        // Reverse-order redaction
        let mut redacted = text.to_string();
        for ent in accepted.into_iter().rev() {
            let tag = format!("[REDACTED_{}]", ent.label.to_uppercase().replace('.', "_"));
            redacted.replace_range(ent.byte_start..ent.byte_end, &tag);
        }
        assert_eq!(
            redacted,
            "Contact [REDACTED_IDENTITY_PERSON_NAME] at [REDACTED_CONTACT_EMAIL] for details."
        );
    }

    #[test]
    #[ignore = "needs a real GGUF via HYBRID_PII_ADAPTER_GGUF; run with --ignored"]
    fn test_load_hybrid_pii_adapter_gguf() {
        let Some(adapter_path) = std::env::var_os("HYBRID_PII_ADAPTER_GGUF") else {
            return;
        };
        let path = std::path::Path::new(&adapter_path);
        if !path.is_file() {
            return;
        }
        let gguf = GgufFile::open(path).expect("failed to open GGUF adapter");
        assert_eq!(gguf.get_str("general.architecture"), Some("modernbert"));
        assert_eq!(gguf.get_u32("modernbert.block_count"), Some(22));
        assert_eq!(gguf.get_u32("modernbert.embedding_length"), Some(768));
        assert_eq!(gguf.get_u32("hybrid_pii.max_span_length"), Some(24));
        assert_eq!(gguf.get_f32("hybrid_pii.gate_threshold"), Some(0.05));
        assert_eq!(gguf.get_f32("hybrid_pii.span_threshold"), Some(0.45));

        let labels = gguf
            .get_string_array("hybrid_pii.labels")
            .expect("missing labels");
        assert_eq!(labels.len(), 11);
        assert_eq!(labels[0], "identity.person_name");

        let dense_w = gguf
            .get_tensor("sensitivity_head.dense.weight")
            .expect("missing dense.weight");
        assert_eq!(dense_w.shape(), &[768, 384]);

        let span_emb = gguf
            .get_tensor("span_head.span_size_embedding.weight")
            .expect("missing size_emb");
        assert_eq!(span_emb.shape(), &[64, 25]);
    }

    #[test]
    fn test_multibyte_utf8_slicing_safety() {
        let text = "User 🦀 alice@example.com reported issue in 東京";
        // Emoji '🦀' is 4 bytes: [5..9]
        // Byte 6 is inside the emoji codepoint
        assert!(!text.is_char_boundary(6));
        assert!(text.get(5..6).is_none());
        assert!(text.get(5..9).is_some());
        assert_eq!(text.get(5..9).unwrap(), "🦀");

        // Unaligned byte slice must safely return None rather than panicking
        let unaligned_start = 6;
        let unaligned_end = 12;
        let span_opt = text.get(unaligned_start..unaligned_end);
        assert!(span_opt.is_none());
    }
}

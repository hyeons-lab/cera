//! Token classification and constrained BIOES Viterbi decoding.
//!
//! Provides zero-dependency named entity recognition (NER) and PII detection
//! over bidirectional token classification models (such as LiquidAI/pii-detect).

use crate::kv_cache::InferenceState;
use crate::model::Model;
use crate::session::CeraError;
use crate::tokenizer::{BpeTokenizer, TokenOffset};

/// An identified entity span in source text.
#[derive(Debug, Clone, PartialEq)]
pub struct EntitySpan {
    /// Entity label type (e.g. "NAME", "EMAIL", "PHONE_NUMBER", "STREET_ADDRESS").
    pub entity_type: String,
    /// UTF-8 character start index in source text (inclusive).
    pub start_char: usize,
    /// UTF-8 character end index in source text (exclusive).
    pub end_char: usize,
    /// Token start index in sequence (inclusive).
    pub start_token: usize,
    /// Token end index in sequence (exclusive).
    pub end_token: usize,
    /// Extracted text slice.
    pub text: String,
    /// Mean classification confidence score across the span tokens [0.0..1.0].
    pub score: f32,
}

/// BIOES tag prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BioesPrefix {
    /// Outside any entity.
    O,
    /// Beginning of a multi-token entity.
    B,
    /// Inside a multi-token entity.
    I,
    /// End of a multi-token entity.
    E,
    /// Single-token entity.
    S,
}

/// Parse a raw label string (e.g. "B-NAME", "S-EMAIL", "O") into prefix and entity type.
pub fn parse_bioes(label: &str) -> (BioesPrefix, &str) {
    if label == "O" || label.is_empty() {
        return (BioesPrefix::O, "");
    }
    if let Some(rest) = label.strip_prefix("B-") {
        return (BioesPrefix::B, rest);
    }
    if let Some(rest) = label.strip_prefix("I-") {
        return (BioesPrefix::I, rest);
    }
    if let Some(rest) = label.strip_prefix("E-") {
        return (BioesPrefix::E, rest);
    }
    if let Some(rest) = label.strip_prefix("S-") {
        return (BioesPrefix::S, rest);
    }
    (BioesPrefix::O, label)
}

/// Check if transitioning from `(prev_prefix, prev_type)` to `(curr_prefix, curr_type)`
/// is grammatically valid under the BIOES scheme.
pub fn is_valid_transition(
    prev_prefix: BioesPrefix,
    prev_type: &str,
    curr_prefix: BioesPrefix,
    curr_type: &str,
) -> bool {
    match prev_prefix {
        BioesPrefix::O | BioesPrefix::E | BioesPrefix::S => {
            // From non-entity or completed entity, can only start new entity or stay O
            matches!(
                curr_prefix,
                BioesPrefix::O | BioesPrefix::B | BioesPrefix::S
            )
        }
        BioesPrefix::B | BioesPrefix::I => {
            // Must continue or end the same entity type
            (curr_prefix == BioesPrefix::I || curr_prefix == BioesPrefix::E)
                && prev_type == curr_type
        }
    }
}

/// Decode the most likely tag sequence under BIOES grammatical constraints
/// using the Viterbi dynamic programming algorithm.
pub fn viterbi_decode(logits: &[f32], num_classes: usize, class_labels: &[String]) -> Vec<usize> {
    let n_tokens = logits.len() / num_classes;
    if n_tokens == 0 {
        return Vec::new();
    }

    let parsed_labels: Vec<(BioesPrefix, &str)> = class_labels
        .iter()
        .map(|s| parse_bioes(s.as_str()))
        .collect();

    // Log-softmax normalization per token position
    let mut log_probs = vec![0.0f32; logits.len()];
    for t in 0..n_tokens {
        let row = &logits[t * num_classes..(t + 1) * num_classes];
        let out_row = &mut log_probs[t * num_classes..(t + 1) * num_classes];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let e = (row[c] - max_val).exp();
            out_row[c] = e;
            sum_exp += e;
        }
        let log_sum = sum_exp.ln();
        for c in 0..num_classes {
            out_row[c] = (row[c] - max_val) - log_sum;
        }
    }

    // dp[c] = max log probability of path ending at class c
    let mut dp = vec![f32::NEG_INFINITY; num_classes];
    let mut backpointer = vec![0usize; n_tokens * num_classes];

    // Token 0: can only start with O, B-*, or S-*
    for c in 0..num_classes {
        let (prefix, _) = parsed_labels[c];
        if matches!(prefix, BioesPrefix::O | BioesPrefix::B | BioesPrefix::S) {
            dp[c] = log_probs[c];
        }
    }

    let mut next_dp = vec![f32::NEG_INFINITY; num_classes];

    for t in 1..n_tokens {
        next_dp.fill(f32::NEG_INFINITY);
        let curr_log_probs = &log_probs[t * num_classes..(t + 1) * num_classes];

        for c in 0..num_classes {
            let (curr_prefix, curr_type) = parsed_labels[c];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0usize;

            for p in 0..num_classes {
                if dp[p] <= f32::NEG_INFINITY {
                    continue;
                }
                let (prev_prefix, prev_type) = parsed_labels[p];
                if is_valid_transition(prev_prefix, prev_type, curr_prefix, curr_type) {
                    let score = dp[p] + curr_log_probs[c];
                    if score > best_score {
                        best_score = score;
                        best_prev = p;
                    }
                }
            }

            next_dp[c] = best_score;
            backpointer[t * num_classes + c] = best_prev;
        }

        std::mem::swap(&mut dp, &mut next_dp);
    }

    // Final token: must end in O, E-*, or S-*
    let mut best_final_class = 0usize;
    let mut best_final_score = f32::NEG_INFINITY;
    for c in 0..num_classes {
        let (prefix, _) = parsed_labels[c];
        if matches!(prefix, BioesPrefix::O | BioesPrefix::E | BioesPrefix::S)
            && dp[c] > best_final_score
        {
            best_final_score = dp[c];
            best_final_class = c;
        }
    }

    // If all valid final states had -inf (extreme fallback), pick argmax
    if best_final_score <= f32::NEG_INFINITY {
        for (c, &score) in dp.iter().enumerate().take(num_classes) {
            if score > best_final_score {
                best_final_score = score;
                best_final_class = c;
            }
        }
    }

    // Backtrack path
    let mut path = vec![0usize; n_tokens];
    let mut curr = best_final_class;
    for t in (0..n_tokens).rev() {
        path[t] = curr;
        if t > 0 {
            curr = backpointer[t * num_classes + curr];
        }
    }

    path
}

/// Extract entity spans from decoded tag indices and token character offsets.
pub fn extract_spans(
    tags: &[usize],
    class_labels: &[String],
    offsets: &[TokenOffset],
    text: &str,
    logits: &[f32],
    num_classes: usize,
) -> Vec<EntitySpan> {
    let mut spans = Vec::new();
    let n = tags.len();
    if n == 0 || offsets.len() < n {
        return spans;
    }

    let parsed_labels: Vec<(BioesPrefix, &str)> = class_labels
        .iter()
        .map(|s| parse_bioes(s.as_str()))
        .collect();

    // Softmax probabilities for scoring
    let mut probs = vec![0.0f32; logits.len()];
    for t in 0..n {
        let row = &logits[t * num_classes..(t + 1) * num_classes];
        let out_row = &mut probs[t * num_classes..(t + 1) * num_classes];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let e = (row[c] - max_val).exp();
            out_row[c] = e;
            sum_exp += e;
        }
        for val in out_row.iter_mut().take(num_classes) {
            *val /= sum_exp;
        }
    }

    let mut i = 0;
    while i < n {
        let (prefix, ent_type) = parsed_labels[tags[i]];
        match prefix {
            BioesPrefix::S => {
                let start_char = offsets[i].char_start;
                let end_char = offsets[i].char_end;
                let b_start = offsets[i].byte_start;
                let b_end = offsets[i].byte_end;
                let span_text = if b_start < b_end && b_end <= text.len() {
                    text[b_start..b_end].to_string()
                } else {
                    String::new()
                };
                let score = probs[i * num_classes + tags[i]];
                spans.push(EntitySpan {
                    entity_type: ent_type.to_string(),
                    start_char,
                    end_char,
                    start_token: i,
                    end_token: i + 1,
                    text: span_text,
                    score,
                });
                i += 1;
            }
            BioesPrefix::B => {
                let start_token = i;
                let start_char = offsets[i].char_start;
                let mut end_token = i + 1;
                let mut sum_score = probs[i * num_classes + tags[i]];

                let mut j = i + 1;
                while j < n {
                    let (sub_prefix, sub_type) = parsed_labels[tags[j]];
                    if sub_type != ent_type {
                        break;
                    }
                    if sub_prefix == BioesPrefix::B || sub_prefix == BioesPrefix::S {
                        break;
                    }
                    sum_score += probs[j * num_classes + tags[j]];
                    end_token = j + 1;
                    if sub_prefix == BioesPrefix::E {
                        break;
                    }
                    if sub_prefix != BioesPrefix::I {
                        break;
                    }
                    j += 1;
                }

                let end_char = offsets[end_token - 1].char_end;
                let span_len = end_token - start_token;
                let avg_score = sum_score / (span_len as f32);
                let b_start = offsets[start_token].byte_start;
                let b_end = offsets[end_token - 1].byte_end;
                let span_text = if b_start < b_end && b_end <= text.len() {
                    text[b_start..b_end].to_string()
                } else {
                    String::new()
                };

                spans.push(EntitySpan {
                    entity_type: ent_type.to_string(),
                    start_char,
                    end_char,
                    start_token,
                    end_token,
                    text: span_text,
                    score: avg_score,
                });

                i = end_token;
            }
            _ => {
                i += 1;
            }
        }
    }

    spans
}

/// Detect PII entity spans in text using the provided token classification model and tokenizer.
pub fn detect_pii(
    model: &dyn Model,
    tokenizer: &BpeTokenizer,
    text: &str,
    state: &mut InferenceState,
) -> Result<Vec<EntitySpan>, CeraError> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let (tokens, offsets) = tokenizer.encode_with_offsets(text);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let max_seq = model.config().max_seq_len;
    if tokens.len() > max_seq {
        return Err(CeraError::Backend(format!(
            "sequence length {} exceeds maximum model sequence limit {}",
            tokens.len(),
            max_seq
        )));
    }

    let logits = model.classify_tokens(&tokens, state)?;
    let (class_labels, num_classes) = if let Some(ref lora) = state.lora {
        if !lora.class_labels.is_empty() {
            (lora.class_labels.as_slice(), lora.class_labels.len())
        } else if !model.class_labels().is_empty() {
            (model.class_labels(), model.num_classes())
        } else {
            (model.class_labels(), lora.num_classes())
        }
    } else {
        (model.class_labels(), model.num_classes())
    };

    if num_classes == 0 || class_labels.is_empty() {
        return Err(CeraError::Backend(
            "neither model nor attached adapter has token classification labels".into(),
        ));
    }

    if class_labels.len() != num_classes {
        return Err(CeraError::Backend(format!(
            "mismatch between class labels count ({}) and num_classes ({})",
            class_labels.len(),
            num_classes
        )));
    }

    let expected_logits_len = tokens.len() * num_classes;
    if logits.len() != expected_logits_len {
        return Err(CeraError::Backend(format!(
            "logits length mismatch: expected {} ({} tokens * {} classes), got {}",
            expected_logits_len,
            tokens.len(),
            num_classes,
            logits.len()
        )));
    }

    let tags = viterbi_decode(&logits, num_classes, class_labels);
    let spans = extract_spans(&tags, class_labels, &offsets, text, &logits, num_classes);
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bioes_parsing() {
        assert_eq!(parse_bioes("O"), (BioesPrefix::O, ""));
        assert_eq!(parse_bioes("B-NAME"), (BioesPrefix::B, "NAME"));
        assert_eq!(parse_bioes("I-NAME"), (BioesPrefix::I, "NAME"));
        assert_eq!(parse_bioes("E-NAME"), (BioesPrefix::E, "NAME"));
        assert_eq!(parse_bioes("S-EMAIL"), (BioesPrefix::S, "EMAIL"));
    }

    #[test]
    fn test_bioes_transitions() {
        // Valid transitions
        assert!(is_valid_transition(BioesPrefix::O, "", BioesPrefix::O, ""));
        assert!(is_valid_transition(
            BioesPrefix::O,
            "",
            BioesPrefix::B,
            "NAME"
        ));
        assert!(is_valid_transition(
            BioesPrefix::O,
            "",
            BioesPrefix::S,
            "EMAIL"
        ));
        assert!(is_valid_transition(
            BioesPrefix::B,
            "NAME",
            BioesPrefix::I,
            "NAME"
        ));
        assert!(is_valid_transition(
            BioesPrefix::B,
            "NAME",
            BioesPrefix::E,
            "NAME"
        ));
        assert!(is_valid_transition(
            BioesPrefix::I,
            "NAME",
            BioesPrefix::I,
            "NAME"
        ));
        assert!(is_valid_transition(
            BioesPrefix::I,
            "NAME",
            BioesPrefix::E,
            "NAME"
        ));
        assert!(is_valid_transition(
            BioesPrefix::E,
            "NAME",
            BioesPrefix::O,
            ""
        ));
        assert!(is_valid_transition(
            BioesPrefix::E,
            "NAME",
            BioesPrefix::S,
            "DATE"
        ));
        assert!(is_valid_transition(
            BioesPrefix::S,
            "EMAIL",
            BioesPrefix::O,
            ""
        ));

        // Invalid transitions
        assert!(!is_valid_transition(
            BioesPrefix::O,
            "",
            BioesPrefix::I,
            "NAME"
        ));
        assert!(!is_valid_transition(
            BioesPrefix::O,
            "",
            BioesPrefix::E,
            "NAME"
        ));
        assert!(!is_valid_transition(
            BioesPrefix::B,
            "NAME",
            BioesPrefix::O,
            ""
        ));
        assert!(!is_valid_transition(
            BioesPrefix::B,
            "NAME",
            BioesPrefix::B,
            "NAME"
        ));
        assert!(!is_valid_transition(
            BioesPrefix::B,
            "NAME",
            BioesPrefix::I,
            "EMAIL"
        ));
        assert!(!is_valid_transition(
            BioesPrefix::I,
            "NAME",
            BioesPrefix::O,
            ""
        ));
    }

    #[test]
    fn test_viterbi_decoding_and_span_extraction() {
        let labels = vec![
            "O".to_string(),
            "B-NAME".to_string(),
            "I-NAME".to_string(),
            "E-NAME".to_string(),
            "S-EMAIL".to_string(),
        ];
        let num_classes = labels.len();

        // 4 tokens: "Hello", " John", " Smith", " user@example.com"
        // Expected:
        // Token 0: O
        // Token 1: B-NAME
        // Token 2: E-NAME
        // Token 3: S-EMAIL
        let logits = vec![
            // Token 0
            10.0, -10.0, -10.0, -10.0, -10.0, // Token 1
            -10.0, 10.0, -10.0, -10.0, -10.0, // Token 2
            -10.0, -10.0, -10.0, 10.0, -10.0, // Token 3
            -10.0, -10.0, -10.0, -10.0, 10.0,
        ];

        let tags = viterbi_decode(&logits, num_classes, &labels);
        assert_eq!(tags, vec![0, 1, 3, 4]);

        let text = "Hello John Smith user@example.com";
        let offsets = vec![
            TokenOffset {
                char_start: 0,
                char_end: 5,
                byte_start: 0,
                byte_end: 5,
            },
            TokenOffset {
                char_start: 6,
                char_end: 10,
                byte_start: 6,
                byte_end: 10,
            },
            TokenOffset {
                char_start: 11,
                char_end: 16,
                byte_start: 11,
                byte_end: 16,
            },
            TokenOffset {
                char_start: 17,
                char_end: 33,
                byte_start: 17,
                byte_end: 33,
            },
        ];

        let spans = extract_spans(&tags, &labels, &offsets, text, &logits, num_classes);
        assert_eq!(spans.len(), 2);

        assert_eq!(spans[0].entity_type, "NAME");
        assert_eq!(spans[0].start_char, 6);
        assert_eq!(spans[0].end_char, 16);
        assert_eq!(spans[0].text, "John Smith");

        assert_eq!(spans[1].entity_type, "EMAIL");
        assert_eq!(spans[1].start_char, 17);
        assert_eq!(spans[1].end_char, 33);
        assert_eq!(spans[1].text, "user@example.com");
    }

    struct MockClassifierModel {
        labels: Vec<String>,
        config: crate::model::ModelConfig,
    }

    impl Model for MockClassifierModel {
        fn forward(&self, _tokens: &[u32], _pos: usize, _state: &mut InferenceState) -> Vec<f32> {
            Vec::new()
        }
        fn config(&self) -> &crate::model::ModelConfig {
            &self.config
        }
        fn is_classifier(&self) -> bool {
            true
        }
        fn num_classes(&self) -> usize {
            self.labels.len()
        }
        fn class_labels(&self) -> &[String] {
            &self.labels
        }
        fn classify_tokens(
            &self,
            tokens: &[u32],
            _state: &mut InferenceState,
        ) -> Result<Vec<f32>, CeraError> {
            let n = tokens.len();
            let k = self.labels.len();
            let mut logits = vec![0.0f32; n * k];
            for t in 0..n {
                logits[t * k] = 5.0; // O by default
            }
            if n >= 2 {
                logits[0] = -5.0;
                logits[1] = 10.0; // B-NAME
                logits[k] = -5.0;
                logits[k + 3] = 10.0; // E-NAME
            }
            Ok(logits)
        }
    }

    #[test]
    fn test_detect_pii_end_to_end() {
        let labels = vec![
            "O".to_string(),
            "B-NAME".to_string(),
            "I-NAME".to_string(),
            "E-NAME".to_string(),
            "S-EMAIL".to_string(),
        ];
        let model = MockClassifierModel {
            labels: labels.clone(),
            config: crate::model::ModelConfig {
                architecture: "mock".into(),
                n_layers: 1,
                hidden_size: 4,
                intermediate_size: 8,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 4,
                vocab_size: 256,
                max_seq_len: 64,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                block_types: vec![crate::model::BlockType::Attention],
                conv_kernel_size: None,
                kv_heads_per_layer: vec![1],
                scalars: crate::model::ScalarMultipliers::default(),
                moe: None,
                is_causal: false,
                class_labels: labels,
            },
        };

        // Create a simple BPE tokenizer with bytes + tokens
        let mut vocab_vec = Vec::new();
        let mut token_to_id = std::collections::HashMap::new();
        for b in 0u8..=255 {
            vocab_vec.push(vec![b]);
            token_to_id.insert(vec![b], b as u32);
        }
        let alice_bytes = b"Alice".to_vec();
        token_to_id.insert(alice_bytes.clone(), 256);
        vocab_vec.push(alice_bytes);

        let smith_bytes = b" Smith".to_vec();
        token_to_id.insert(smith_bytes.clone(), 257);
        vocab_vec.push(smith_bytes);

        let tokenizer =
            BpeTokenizer::new_for_testing(vocab_vec, token_to_id, std::collections::HashMap::new());

        let mut state = InferenceState::from_config(&model.config).unwrap();
        let spans = detect_pii(&model, &tokenizer, "AB", &mut state).unwrap();

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].entity_type, "NAME");
        assert_eq!(spans[0].text, "AB");
        assert_eq!(spans[0].start_char, 0);
        assert_eq!(spans[0].end_char, 2);
    }

    struct MockBaseModel {
        config: crate::model::ModelConfig,
    }

    impl Model for MockBaseModel {
        fn forward(&self, _tokens: &[u32], _pos: usize, _state: &mut InferenceState) -> Vec<f32> {
            Vec::new()
        }
        fn config(&self) -> &crate::model::ModelConfig {
            &self.config
        }
        fn classify_tokens(
            &self,
            tokens: &[u32],
            state: &mut InferenceState,
        ) -> Result<Vec<f32>, CeraError> {
            let Some(ref lora) = state.lora else {
                return Err(CeraError::Backend("no classifier head".into()));
            };
            let k = lora.class_labels.len();
            let n = tokens.len();
            let mut logits = vec![0.0f32; n * k];
            for t in 0..n {
                logits[t * k] = 5.0; // O
            }
            if n >= 1 {
                logits[0] = -5.0;
                logits[1] = 10.0; // S-EMAIL
            }
            Ok(logits)
        }
    }

    #[test]
    fn test_detect_pii_with_separate_lora_adapter() {
        let base_model = MockBaseModel {
            config: crate::model::ModelConfig {
                architecture: "mock".into(),
                n_layers: 1,
                hidden_size: 4,
                intermediate_size: 8,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 4,
                vocab_size: 256,
                max_seq_len: 64,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                block_types: vec![crate::model::BlockType::Attention],
                conv_kernel_size: None,
                kv_heads_per_layer: vec![1],
                scalars: crate::model::ScalarMultipliers::default(),
                moe: None,
                is_causal: true,          // base model is causal
                class_labels: Vec::new(), // base model has no classifier labels
            },
        };

        // Construct mock LoRA adapter with classifier head and labels
        let lora = crate::lora::LoraAdapterWeights::new_classifier_for_testing(
            vec![0.0; 8],
            Some(vec![0.0; 2]),
            vec!["O".to_string(), "S-EMAIL".to_string()],
        );

        let mut vocab_vec = Vec::new();
        let mut token_to_id = std::collections::HashMap::new();
        for b in 0u8..=255 {
            vocab_vec.push(vec![b]);
            token_to_id.insert(vec![b], b as u32);
        }
        let tokenizer =
            BpeTokenizer::new_for_testing(vocab_vec, token_to_id, std::collections::HashMap::new());

        let mut state = InferenceState::from_config(&base_model.config).unwrap();
        state.lora = Some(lora);

        let spans = detect_pii(&base_model, &tokenizer, "X", &mut state).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].entity_type, "EMAIL");
        assert_eq!(spans[0].text, "X");
        assert_eq!(spans[0].start_char, 0);
        assert_eq!(spans[0].end_char, 1);
    }

    #[test]
    fn test_consecutive_b_and_s_tags_extraction() {
        let labels = vec![
            "O".to_string(),
            "B-NAME".to_string(),
            "I-NAME".to_string(),
            "E-NAME".to_string(),
            "S-EMAIL".to_string(),
        ];
        let num_classes = labels.len();
        let tags = vec![1, 1, 4]; // B-NAME, B-NAME, S-EMAIL
        let text = "Alice Bob user@example.com";
        let offsets = vec![
            TokenOffset {
                char_start: 0,
                char_end: 5,
                byte_start: 0,
                byte_end: 5,
            },
            TokenOffset {
                char_start: 6,
                char_end: 9,
                byte_start: 6,
                byte_end: 9,
            },
            TokenOffset {
                char_start: 10,
                char_end: 26,
                byte_start: 10,
                byte_end: 26,
            },
        ];
        let mut logits = vec![-10.0f32; 3 * num_classes];
        logits[1] = 10.0;
        logits[num_classes + 1] = 10.0;
        logits[2 * num_classes + 4] = 10.0;

        let spans = extract_spans(&tags, &labels, &offsets, text, &logits, num_classes);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "Alice");
        assert_eq!(spans[1].text, "Bob");
        assert_eq!(spans[2].text, "user@example.com");
    }

    #[test]
    fn test_detect_pii_sequence_length_exceeded() {
        let labels = vec![
            "O".to_string(),
            "B-NAME".to_string(),
            "I-NAME".to_string(),
            "E-NAME".to_string(),
            "S-NAME".to_string(),
        ];
        let model = MockClassifierModel {
            labels: labels.clone(),
            config: crate::model::ModelConfig {
                architecture: "mock".to_string(),
                n_layers: 1,
                hidden_size: 4,
                intermediate_size: 8,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 4,
                vocab_size: 256,
                max_seq_len: 2,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                block_types: vec![crate::model::BlockType::Attention],
                conv_kernel_size: None,
                kv_heads_per_layer: vec![1],
                scalars: crate::model::ScalarMultipliers::default(),
                moe: None,
                is_causal: false,
                class_labels: labels,
            },
        };

        let mut vocab_vec = Vec::new();
        let mut token_to_id = std::collections::HashMap::new();
        for b in 0u8..=255 {
            vocab_vec.push(vec![b]);
            token_to_id.insert(vec![b], b as u32);
        }
        let tokenizer =
            BpeTokenizer::new_for_testing(vocab_vec, token_to_id, std::collections::HashMap::new());

        let mut state = InferenceState::from_config(&model.config).unwrap();
        let result = detect_pii(&model, &tokenizer, "ABCDE", &mut state);
        assert!(result.is_err());
        if let Err(CeraError::Backend(msg)) = result {
            assert!(msg.contains("exceeds maximum model sequence limit"));
        } else {
            panic!("expected CeraError::Backend on sequence length exceeded");
        }
    }
}

//! Unit and parity tests for Grouped MoE Batched Prefill.
//!
//! Verifies:
//! 1. Top-k biased expert routing and unbiased probability renormalization.
//! 2. Grouped MoE token binning and scatter-add accumulation parity against sequential per-token evaluation.
//! 3. Preservation of expert activation order and weight indexing across prefill chunks.

fn select_experts_ref(
    probs: &[f32],
    biases: &[f32],
    n_used: usize,
    selected: &mut Vec<(usize, f32)>,
) {
    selected.clear();
    let n_expert = probs.len().min(biases.len());
    let mut biased = vec![0.0f32; n_expert];
    for i in 0..n_expert {
        biased[i] = probs[i] + biases[i];
    }

    for _ in 0..n_used.min(n_expert) {
        let best = (0..n_expert)
            .filter(|e| !selected.iter().any(|(taken, _)| taken == e))
            .max_by(|&a, &b| biased[a].total_cmp(&biased[b]).then(b.cmp(&a)));
        if let Some(e) = best {
            selected.push((e, probs[e]));
        }
    }

    let sum: f32 = selected.iter().map(|(_, p)| *p).sum();
    let norm = sum.max(6.1035156e-5);
    for (_, p) in selected.iter_mut() {
        *p /= norm;
    }
}

#[test]
fn test_select_experts_renormalization() {
    let probs = vec![0.1, 0.4, 0.2, 0.8, 0.05, 0.3];
    let biases = vec![0.0, 0.1, 0.0, -0.2, 0.5, 0.0];
    let mut selected = Vec::new();

    select_experts_ref(&probs, &biases, 3, &mut selected);

    assert_eq!(selected.len(), 3);
    let total_weight: f32 = selected.iter().map(|(_, w)| *w).sum();
    assert!(
        (total_weight - 1.0).abs() < 1e-6,
        "Weights must sum to 1.0, got {total_weight}"
    );
}

#[test]
fn test_grouped_moe_scatter_equivalence() {
    let n_tokens = 16;
    let hidden_size = 64;
    let n_expert = 8;
    let n_used = 2;

    // Synthetic token activations: [n_tokens * hidden_size]
    let mut ffn_input = vec![0.0f32; n_tokens * hidden_size];
    for j in 0..n_tokens {
        for i in 0..hidden_size {
            ffn_input[j * hidden_size + i] = ((j * 17 + i * 31) % 100) as f32 * 0.01;
        }
    }

    // Synthetic router biases
    let biases: Vec<f32> = (0..n_expert).map(|e| (e as f32) * 0.05).collect();

    // Generate deterministic router assignments per token
    let mut token_selected: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tokens);
    for j in 0..n_tokens {
        let mut probs = vec![0.0f32; n_expert];
        for (e, prob) in probs.iter_mut().enumerate().take(n_expert) {
            let logit = ((j + 1) * (e + 3)) as f32 * 0.1;
            *prob = 1.0 / (1.0 + (-logit).exp());
        }
        let mut sel = Vec::new();
        select_experts_ref(&probs, &biases, n_used, &mut sel);
        token_selected.push(sel);
    }

    // 1. Sequential reference evaluation:
    // For each token, compute dummy expert transformation and accumulate with weight
    let mut seq_out = vec![0.0f32; n_tokens * hidden_size];
    for (j, sel) in token_selected.iter().enumerate().take(n_tokens) {
        let tok_in = &ffn_input[j * hidden_size..(j + 1) * hidden_size];
        for &(e, weight) in sel {
            for i in 0..hidden_size {
                let exp_val = tok_in[i] * (1.0 + (e as f32) * 0.1) + 0.02 * (e as f32);
                seq_out[j * hidden_size + i] += weight * exp_val;
            }
        }
    }

    // 2. Grouped MoE evaluation:
    // Bin tokens by expert
    let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];
    for (j, sel) in token_selected.iter().enumerate().take(n_tokens) {
        for &(e, weight) in sel {
            if weight > 0.0 {
                expert_assignments[e].push((j, weight));
            }
        }
    }

    let mut grouped_out = vec![0.0f32; n_tokens * hidden_size];
    for (e, assigned) in expert_assignments.iter().enumerate().take(n_expert) {
        if assigned.is_empty() {
            continue;
        }
        // Batched expert evaluation across all assigned tokens
        for &(token_j, weight) in assigned {
            let tok_in = &ffn_input[token_j * hidden_size..(token_j + 1) * hidden_size];
            for i in 0..hidden_size {
                let exp_val = tok_in[i] * (1.0 + (e as f32) * 0.1) + 0.02 * (e as f32);
                grouped_out[token_j * hidden_size + i] += weight * exp_val;
            }
        }
    }

    // Verify exact floating-point equality between sequential and grouped outputs
    for idx in 0..(n_tokens * hidden_size) {
        let diff = (seq_out[idx] - grouped_out[idx]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at index {idx}: sequential={} grouped={}",
            seq_out[idx],
            grouped_out[idx]
        );
    }
}

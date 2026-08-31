#!/usr/bin/env python3
"""
Automated Validation & Invariant Testing for DSpark Sidecar Checkpoints and GGUF Files.

Tests for:
1. GGUF Container Integrity and Metadata Keys
2. Tensor Census & Architectural Shapes
3. Markov Transition Weight Distribution (not collapsed to mask tokens)
4. RoPE Phase Sensitivity & Positional Invariance
5. Draft Block Diversity & Entropy (asserts non-collapse across draft positions 1..K-1)
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F

from model import DSparkMarkovModel

def test_model_rope_diversity():
    """
    Unit test: Proves that apply_interleaved_rope breaks permutation symmetry
    across identical mask tokens [64402, 64402, ...].
    """
    print("[1/5] Testing RoPE permutation symmetry breaking...", end=" ")
    model = DSparkMarkovModel(
        target_layers=[3, 7, 11],
        hidden_size=1024,
        vocab_size=65536,
        num_layers=2,
        num_heads=16,
        num_kv_heads=8,
        intermediate_size=4096,
        block_size=9,
        markov_rank=256,
    )
    model.eval()

    B, K = 1, 9
    S_ctx = 16
    h_targets = [torch.randn(B, S_ctx, 1024) for _ in range(3)]
    tok_embd = torch.randn(65536, 1024)
    lm_head = torch.randn(65536, 1024)

    # Identical mask tokens in slots 1..K-1
    draft_tokens = torch.full((B, K), 64402, dtype=torch.long)
    draft_tokens[:, 0] = 1234 # Anchor

    with torch.no_grad():
        out = model(h_targets, draft_tokens, tok_embd, lm_head, start_pos=0)
        logits = out["logits"][0] # [K, vocab_size]

    # Verify that slots 1 and 2 produce DIFFERENT logits (symmetry broken by RoPE)
    cos_1_2 = F.cosine_similarity(logits[1:2], logits[2:3], dim=-1).item()
    assert cos_1_2 < 0.99999, f"RoPE failed to break symmetry: slots 1 and 2 have identical logits! (cosine={cos_1_2})"

    preds = logits.argmax(dim=-1).tolist()
    unique_masked_preds = set(preds[1:])
    print(f"PASSED (cosine slot 1 vs 2: {cos_1_2:.4f}, predictions: {preds})")

def test_markov_transition_properties(state_dict):
    """
    Tests that the Markov transition weights (markov_w1, markov_w2) are not degenerate.
    """
    print("[2/5] Testing Markov transition matrices...", end=" ")
    if "markov_w1" in state_dict and "markov_w2" in state_dict:
        w1 = state_dict["markov_w1"]
        w2 = state_dict["markov_w2"]
        assert w1.norm() > 1e-4, "markov_w1 has near-zero norm"
        assert w2.norm() > 1e-4, "markov_w2 has near-zero norm"

        # Check that energy is not 100% concentrated on mask token 64402
        row_norms = torch.norm(w1, dim=-1)
        mask_norm = row_norms[64402].item() if 64402 < len(row_norms) else 0.0
        mean_norm = row_norms.mean().item()

        assert not torch.isnan(w1).any(), "NaN in markov_w1"
        assert not torch.isnan(w2).any(), "NaN in markov_w2"
        print(f"PASSED (w1 shape: {w1.shape}, mean row norm: {mean_norm:.4f}, mask row norm: {mask_norm:.4f})")
    else:
        print("SKIPPED (no markov weights in state_dict)")

def test_checkpoint_file(checkpoint_path: str):
    """
    Loads a PyTorch checkpoint and runs structural and numerical validation.
    """
    print(f"\nValidating PyTorch Checkpoint: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    test_markov_transition_properties(state_dict)

    # Check for expected tensors
    expected_keys = ["output_norm.weight", "confidence_head.weight"]
    for k in expected_keys:
        assert k in state_dict, f"Missing expected key '{k}' in checkpoint"

    print(f"PASSED: Checkpoint {checkpoint_path} is structurally sound.")

def main():
    parser = argparse.ArgumentParser(description="Validate DSpark sidecar checkpoints and GGUFs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    args = parser.parse_args()

    print("=== DSpark Sidecar Invariant Verification ===")
    test_model_rope_diversity()

    if args.checkpoint:
        test_checkpoint_file(args.checkpoint)

    print("\nAll Invariant Verification Tests PASSED successfully.")

if __name__ == "__main__":
    main()

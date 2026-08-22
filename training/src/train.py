"""
Training script for LFM2.5-VL-450M DSpark Draft Sidecars on Apple Silicon (MPS).

Supports training:
- Option B Standalone (dspark-standalone for Cera)
- Option B Markov (dspark-markov for llama.cpp --spec-type draft-dspark)
- Both simultaneously with shared teacher forward passes
"""

import os
import time
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from model import DSparkStandaloneModel, DSparkMarkovModel
from data import prepare_dataset, collate_fn

def compute_acceptance_length(drafter, base_model, lm_head_weight, tok_embd_weight, eval_loader, device, block_size=9, num_batches=15, is_markov=False, target_layers=[3, 7, 11]):
    """
    Evaluates empirical speculative acceptance length (tau out of block_size) against the target base model.
    """
    drafter.eval()
    total_accepted = 0
    total_rounds = 0

    with torch.inference_mode():
        for i, batch in enumerate(eval_loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            step_stride = max(1, block_size)
            for t in range(16, S - block_size - 1, step_stride):
                prefix_ids = input_ids[:, :t]
                target_window = input_ids[:, t:t + block_size]

                outputs = base_model(input_ids=prefix_ids, output_hidden_states=True)

                if is_markov:
                    target_hiddens = [outputs.hidden_states[l].float() for l in target_layers]
                    dflash_tokens = torch.full((B, block_size), 64402, dtype=torch.long, device=device)
                    dflash_tokens[:, 0] = prefix_ids[:, -1]
                    draft_out = drafter(target_hiddens, dflash_tokens, tok_embd_weight, lm_head_weight)
                    draft_preds = draft_out["logits"].argmax(dim=-1)
                else:
                    # Standalone drafter forwards prefix tokens autoregressively
                    draft_tokens = []
                    curr_tokens = prefix_ids
                    for _ in range(block_size):
                        draft_out = drafter(curr_tokens, tok_embd_weight, lm_head_weight)
                        next_tok = draft_out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                        draft_tokens.append(next_tok)
                        curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)
                    draft_preds = torch.cat(draft_tokens, dim=1)

                for b in range(B):
                    accepted_in_round = 0
                    for k in range(block_size):
                        if draft_preds[b, k] == target_window[b, k]:
                            accepted_in_round += 1
                        else:
                            break
                    total_accepted += accepted_in_round
                    total_rounds += 1

    avg_acceptance = total_accepted / max(1, total_rounds)
    return avg_acceptance

def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    model_id = "LiquidAI/LFM2.5-VL-450M"
    hidden_size = 1024
    vocab_size = 65536
    num_layers = args.num_layers
    num_heads = 16
    num_kv_heads = 8
    intermediate_size = 4096
    block_size = args.block_size
    markov_rank = 256
    target_layers = [int(x) for x in args.target_layers.split(",")]

    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps
    learning_rate = args.lr
    epochs = args.epochs
    max_seq_len = 256
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading base model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, dtype=torch.float32, trust_remote_code=True).to(device)
    base_model.eval()

    # Freeze base model weights
    for param in base_model.parameters():
        param.requires_grad = False

    tok_embd_weight = base_model.language_model.embed_tokens.weight.float()
    lm_head_weight = tok_embd_weight

    train_standalone = args.mode in ["standalone", "both"]
    train_markov = args.mode in ["markov", "both"]

    models = {}
    optimizers = {}
    schedulers = {}

    if train_standalone:
        print("Initializing DSpark Standalone Model (Decoupled Drafter for Cera)...")
        models["standalone"] = DSparkStandaloneModel(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            block_size=block_size,
        ).to(device)
        optimizers["standalone"] = torch.optim.AdamW(models["standalone"].parameters(), lr=learning_rate, weight_decay=0.01)

    if train_markov:
        print(f"Initializing DSpark Markov Model (DFlash specification for llama.cpp, target layers {target_layers})...")
        models["markov"] = DSparkMarkovModel(
            target_layers=target_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            block_size=block_size,
            markov_rank=markov_rank,
        ).to(device)
        optimizers["markov"] = torch.optim.AdamW(models["markov"].parameters(), lr=learning_rate, weight_decay=0.01)

    # Dataset preparation
    full_dataset = prepare_dataset(tokenizer, max_length=max_seq_len, total_samples=2400)
    train_size = int(0.95 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    for k in optimizers:
        schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[k], T_max=total_steps, eta_min=1e-5)

    print(f"\n=======================================================")
    print(f"Starting Training (Mode: {args.mode}, {epochs} epochs, {len(train_dataset)} samples, {len(train_loader)} batches/epoch)")
    print(f"=======================================================\n")
    best_acceptance = {"standalone": 0.0, "markov": 0.0}

    for epoch in range(1, epochs + 1):
        for m in models.values():
            m.train()
        loss_accum = {k: torch.zeros(1, device=device) for k in models}
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape
            if S <= block_size + 2:
                continue

            # Shared teacher forward pass (run once per batch!)
            with torch.inference_mode():
                base_out = base_model(input_ids=input_ids, output_hidden_states=True)

            t_max = S - block_size - 1
            if t_max <= 0:
                continue

            t = random.randint(0, t_max)
            target_tokens = input_ids[:, t + 1 : t + 1 + block_size]
            prev_tokens = input_ids[:, t : t + block_size]

            # 1. Train Standalone Drafter (if enabled)
            if train_standalone:
                # Standalone causal draft LM trains on input sequence predicting next tokens
                out_s = models["standalone"](input_ids[:, :-1], tok_embd_weight, lm_head_weight)
                target_s = input_ids[:, 1:]
                loss_ce_s = F.cross_entropy(out_s["logits"].reshape(-1, vocab_size), target_s.reshape(-1))
                pred_s = out_s["logits"].argmax(dim=-1)
                is_accepted_s = (pred_s == target_s).float()
                loss_conf_s = F.binary_cross_entropy_with_logits(out_s["confidence_logits"], is_accepted_s)
                loss_s = (loss_ce_s + 0.1 * loss_conf_s) / grad_accum_steps
                loss_s.backward()
                loss_accum["standalone"] += loss_s.detach() * grad_accum_steps

            # 2. Train Markov / DFlash Drafter (if enabled)
            if train_markov:
                target_hiddens = [base_out.hidden_states[l][:, :t+1, :].float() for l in target_layers]
                # In DFlash / llama.cpp, slot 0 is the anchor token, and slots 1..k-1 are mask tokens (64402)
                dflash_tokens = torch.full((B, block_size), 64402, dtype=torch.long, device=device)
                dflash_tokens[:, 0] = input_ids[:, t]
                out_m = models["markov"](target_hiddens, dflash_tokens, tok_embd_weight, lm_head_weight)
                loss_ce_m = F.cross_entropy(out_m["logits"].reshape(-1, vocab_size), target_tokens.reshape(-1))
                pred_m = out_m["logits"].argmax(dim=-1)
                is_accepted_m = (pred_m == target_tokens).float()
                loss_conf_m = F.binary_cross_entropy_with_logits(out_m["confidence_logits"], is_accepted_m)
                loss_m = (loss_ce_m + 0.1 * loss_conf_m) / grad_accum_steps
                loss_m.backward()
                loss_accum["markov"] += loss_m.detach() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                for k in models:
                    torch.nn.utils.clip_grad_norm_(models[k].parameters(), 1.0)
                    optimizers[k].step()
                    schedulers[k].step()
                    optimizers[k].zero_grad()

            if (step + 1) % 50 == 0:
                print(f"  [Epoch {epoch:02d}] Step {step+1}/{len(train_loader)}...", flush=True)

        elapsed = time.time() - start_time
        loss_str = " | ".join([f"{k} Loss: {loss_accum[k].item()/max(1, len(train_loader)):.4f}" for k in models])
        print(f"Epoch {epoch:02d}/{epochs:02d} | {loss_str} | Time: {elapsed:.1f}s", flush=True)

        # Evaluate acceptance length
        if train_standalone:
            acc_s = compute_acceptance_length(models["standalone"], base_model, lm_head_weight, lm_head_weight, eval_loader, device, block_size=block_size, is_markov=False)
            print(f"  [Standalone / Cera] Mean Acceptance: {acc_s:.2f}/{block_size} tokens", flush=True)
            if acc_s > best_acceptance["standalone"]:
                best_acceptance["standalone"] = acc_s
                torch.save(models["standalone"].state_dict(), os.path.join(checkpoint_dir, "best_dspark_standalone.pt"))

        if train_markov:
            acc_m = compute_acceptance_length(models["markov"], base_model, lm_head_weight, lm_head_weight, eval_loader, device, block_size=block_size, is_markov=True, target_layers=target_layers)
            print(f"  [Markov / llama.cpp] Mean Acceptance: {acc_m:.2f}/{block_size} tokens", flush=True)
            if acc_m > best_acceptance["markov"]:
                best_acceptance["markov"] = acc_m
                state_dict_m = models["markov"].state_dict()
                state_dict_m["target_layers"] = target_layers
                torch.save(state_dict_m, os.path.join(checkpoint_dir, "best_dspark_markov.pt"))

    print(f"\n=======================================================", flush=True)
    print("Training Complete!", flush=True)
    if train_standalone:
        print(f"  Best Standalone (Cera): {best_acceptance['standalone']:.2f}/{block_size} tokens -> {checkpoint_dir}/best_dspark_standalone.pt", flush=True)
    if train_markov:
        print(f"  Best Markov (llama.cpp): {best_acceptance['markov']:.2f}/{block_size} tokens -> {checkpoint_dir}/best_dspark_markov.pt", flush=True)
    print(f"=======================================================\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["standalone", "markov", "both"], default="both", help="Training mode")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--block-size", type=int, default=9, help="Draft block size K")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of drafter layers")
    parser.add_argument("--target-layers", type=str, default="3,7,11", help="Comma-separated target layers for Markov drafter")
    args = parser.parse_args()

    train(args)


"""
DSpark Draft Model Architectures for LFM2.5-VL / LFM2.

Supports both:
1. DSparkStandaloneModel: Standalone neural draft transformer (Option B - Standalone for Cera).
2. DSparkMarkovModel: Intermediate target-layer feature extraction + Markov transition head (Option B - Markov for llama.cpp).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class DSparkLayer(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, num_kv_heads: int = 8, intermediate_size: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.attn_norm = RMSNorm(hidden_size)
        self.attn_q = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.attn_k = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.attn_v = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.attn_output = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        norm_x = self.attn_norm(x)
        B, S, _ = x.shape

        q = self.attn_q(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(norm_x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(norm_x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.attn_output(attn_out)

        # SwiGLU FFN
        norm_x = self.ffn_norm(x)
        gate = F.silu(self.ffn_gate(norm_x))
        up = self.ffn_up(norm_x)
        ffn_out = self.ffn_down(gate * up)
        x = x + ffn_out

        return x

class DSparkStandaloneModel(nn.Module):
    """
    Standalone Autoregressive Neural Drafter (Cera contract).
    Lightweight neural draft transformer operating on token embeddings with shared base embeddings and LM head.
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        vocab_size: int = 65536,
        num_layers: int = 5,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        intermediate_size: int = 4096,
        block_size: int = 9,
        markov_rank: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.block_size = block_size
        self.markov_rank = markov_rank

        self.layers = nn.ModuleList([
            DSparkLayer(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(hidden_size)

        # Low-rank Markov transition matrices (w1 and w2)
        self.markov_w1 = nn.Parameter(torch.randn(vocab_size, markov_rank) * 0.02)
        self.markov_w2 = nn.Parameter(torch.randn(vocab_size, markov_rank) * 0.02)

        # Confidence projection head (linear vector projection for dot-product parity)
        self.confidence_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_embd_weight: torch.Tensor,
        base_lm_head_weight: torch.Tensor,
    ):
        """
        input_ids: [B, S]
        token_embd_weight: [vocab_size, hidden_size]
        base_lm_head_weight: [vocab_size, hidden_size]
        """
        B, S = input_ids.shape
        x = F.embedding(input_ids, token_embd_weight) # [B, S, hidden_size]

        causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        draft_features = self.output_norm(x) # [B, S, hidden_size]
        base_logits = F.linear(draft_features, base_lm_head_weight) # [B, S, vocab_size]

        # Markov transition bias
        prev_w1 = F.embedding(input_ids, self.markov_w1) # [B, S, markov_rank]
        markov_logits = torch.matmul(prev_w1, self.markov_w2.t()) # [B, S, vocab_size]

        logits = base_logits + markov_logits
        confidence_logits = self.confidence_head(draft_features).squeeze(-1) # [B, S]
        confidence_probs = torch.sigmoid(confidence_logits)

        return {
            "draft_features": draft_features,
            "base_logits": base_logits,
            "markov_logits": markov_logits,
            "logits": logits,
            "confidence_probs": confidence_probs,
            "confidence_logits": confidence_logits,
        }

class DSparkMarkovModel(nn.Module):
    """
    Feature-Extraction + Markov Transition Drafter (Option B: Markov for llama.cpp dflash.cpp).
    Dual-mode: fused target feature injection into KV cache + non-causal draft block attention over [anchor, MASK...].
    """
    def __init__(
        self,
        target_layers: list[int] = [3, 7, 11],
        hidden_size: int = 1024,
        vocab_size: int = 65536,
        num_layers: int = 5,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        intermediate_size: int = 4096,
        block_size: int = 9,
        markov_rank: int = 256,
    ):
        super().__init__()
        self.target_layers = target_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.block_size = block_size
        self.markov_rank = markov_rank

        # Concatenated target feature projection: (len(target_layers) * hidden_size) -> hidden_size
        in_dim = len(target_layers) * hidden_size
        self.fc = nn.Linear(in_dim, hidden_size, bias=False)
        self.enc_norm = RMSNorm(hidden_size)

        self.layers = nn.ModuleList([
            DSparkLayer(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(hidden_size)

        # Low-rank Markov transition matrices (w1 and w2)
        self.markov_w1 = nn.Parameter(torch.randn(vocab_size, markov_rank) * 0.02)
        self.markov_w2 = nn.Parameter(torch.randn(vocab_size, markov_rank) * 0.02)

        # Confidence projection head (linear vector projection for dot-product parity)
        self.confidence_head = nn.Linear(hidden_size + markov_rank, 1, bias=False)

    def forward(
        self,
        target_layer_hiddens: list[torch.Tensor], # list of [B, S_ctx, hidden_size]
        draft_token_ids: torch.Tensor,           # [B, K] = [anchor, 64402, 64402, ...]
        token_embd_weight: torch.Tensor,         # [vocab_size, hidden_size]
        base_lm_head_weight: torch.Tensor,       # [vocab_size, hidden_size]
    ):
        B, K = draft_token_ids.shape
        S_ctx = target_layer_hiddens[0].shape[1]

        # 1. Project fused context features
        concat_h = torch.cat(target_layer_hiddens, dim=-1) # [B, S_ctx, in_dim]
        fused = self.enc_norm(self.fc(concat_h))          # [B, S_ctx, hidden_size]

        # 2. Draft block tokens
        x_draft = F.embedding(draft_token_ids, token_embd_weight) # [B, K, hidden_size]

        # Injected KV from fused context features (no attn_norm on injected path)
        # Block attends over [Injected KV (context), Block KV (draft)]
        x = x_draft
        for layer in self.layers:
            # Context K/V (injected)
            k_inj = layer.attn_k(fused).view(B, S_ctx, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
            v_inj = layer.attn_v(fused).view(B, S_ctx, layer.num_kv_heads, layer.head_dim).transpose(1, 2)

            # Block Q/K/V
            norm_x = layer.attn_norm(x)
            q_blk = layer.attn_q(norm_x).view(B, K, layer.num_heads, layer.head_dim).transpose(1, 2)
            k_blk = layer.attn_k(norm_x).view(B, K, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
            v_blk = layer.attn_v(norm_x).view(B, K, layer.num_kv_heads, layer.head_dim).transpose(1, 2)

            # Combine KV
            k_total = torch.cat([k_inj, k_blk], dim=-2)
            v_total = torch.cat([v_inj, v_blk], dim=-2)

            if layer.num_kv_groups > 1:
                k_total = k_total.repeat_interleave(layer.num_kv_groups, dim=1)
                v_total = v_total.repeat_interleave(layer.num_kv_groups, dim=1)

            scores = torch.matmul(q_blk, k_total.transpose(-2, -1)) / math.sqrt(layer.head_dim)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_blk.dtype)
            attn_out = torch.matmul(attn_weights, v_total)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, -1)
            x = x + layer.attn_output(attn_out)

            # FFN
            norm_x = layer.ffn_norm(x)
            gate = F.silu(layer.ffn_gate(norm_x))
            up = layer.ffn_up(norm_x)
            x = x + layer.ffn_down(gate * up)

        draft_features = self.output_norm(x) # [B, K, hidden_size]
        base_logits = F.linear(draft_features, base_lm_head_weight)

        # Markov transition bias
        prev_w1 = F.embedding(draft_token_ids, self.markov_w1) # [B, K, markov_rank]
        markov_logits = torch.matmul(prev_w1, self.markov_w2.t()) # [B, K, vocab_size]
        logits = base_logits + markov_logits

        conf_in = torch.cat([draft_features, prev_w1], dim=-1)
        confidence_logits = self.confidence_head(conf_in).squeeze(-1)
        confidence_probs = torch.sigmoid(confidence_logits)

        return {
            "draft_features": draft_features,
            "base_logits": base_logits,
            "markov_logits": markov_logits,
            "logits": logits,
            "confidence_probs": confidence_probs,
            "confidence_logits": confidence_logits,
        }

# Alias for backwards compatibility
DSparkDraftModel = DSparkStandaloneModel

if __name__ == "__main__":
    lm_head = torch.randn(65536, 1024)
    tok_embd = torch.randn(65536, 1024)

    # Test Standalone Model
    standalone = DSparkStandaloneModel()
    seq_tokens = torch.randint(0, 1000, (2, 32))
    out_s = standalone(seq_tokens, tok_embd, lm_head)
    print("Standalone logits shape:", out_s["logits"].shape)

    # Test Markov Model
    markov = DSparkMarkovModel(target_layers=[3, 7, 11])
    h_targets = [torch.randn(2, 32, 1024), torch.randn(2, 32, 1024), torch.randn(2, 32, 1024)]
    dflash_tokens = torch.full((2, 9), 64402, dtype=torch.long)
    dflash_tokens[:, 0] = seq_tokens[:, -1]
    out_m = markov(h_targets, dflash_tokens, tok_embd, lm_head)
    print("Markov logits shape:", out_m["logits"].shape)
    print("Markov sidecar params:", sum(p.numel() for p in markov.parameters()) / 1e6, "M")

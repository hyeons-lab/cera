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
    Standalone Neural Drafter (Option B: Standalone).
    Decoupled neural transformer operating from final hidden state and autoregressive embeddings.
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.block_size = block_size

        self.layers = nn.ModuleList([
            DSparkLayer(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(hidden_size)
        self.block_queries = nn.Parameter(torch.randn(block_size, hidden_size) * 0.02)

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        teacher_final_hidden: torch.Tensor,
        base_lm_head_weight: torch.Tensor,
    ):
        """
        teacher_final_hidden: [B, hidden_size]
        base_lm_head_weight: [vocab_size, hidden_size]
        """
        B = teacher_final_hidden.shape[0]
        K = self.block_size

        queries = self.block_queries.unsqueeze(0).expand(B, -1, -1)
        x = teacher_final_hidden.unsqueeze(1).expand(-1, K, -1) + queries

        causal_mask = torch.triu(torch.full((K, K), float("-inf"), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        draft_features = self.output_norm(x) # [B, K, hidden_size]
        base_logits = F.linear(draft_features, base_lm_head_weight)
        confidence_logits = self.confidence_head(draft_features).squeeze(-1) # [B, K]
        confidence_probs = torch.sigmoid(confidence_logits)

        return {
            "draft_features": draft_features,
            "logits": base_logits,
            "confidence_probs": confidence_probs,
            "confidence_logits": confidence_logits,
        }

class DSparkMarkovModel(nn.Module):
    """
    Feature-Extraction + Markov Transition Drafter (Option B: Markov).
    Liquid AI / DFlash specification compatible with llama.cpp (--spec-type draft-dspark).
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
        self.block_queries = nn.Parameter(torch.randn(block_size, hidden_size) * 0.02)

        # Low-rank Markov transition matrices (w1 and w2)
        self.markov_w1 = nn.Parameter(torch.randn(vocab_size, markov_rank) * 0.02)
        self.markov_w2 = nn.Parameter(torch.randn(vocab_size, markov_rank) * 0.02)

        # Confidence projection head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size + markov_rank, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        target_layer_hiddens: list[torch.Tensor],
        base_lm_head_weight: torch.Tensor,
        input_token_ids: torch.Tensor = None,
    ):
        """
        target_layer_hiddens: list of [B, hidden_size] from target_layers
        base_lm_head_weight: [vocab_size, hidden_size]
        input_token_ids: [B, K]
        """
        B = target_layer_hiddens[0].shape[0]
        K = self.block_size

        # Concatenate intermediate target layers
        concat_h = torch.cat(target_layer_hiddens, dim=-1) # [B, len(target_layers) * hidden_size]
        fused_h = self.enc_norm(self.fc(concat_h)) # [B, hidden_size]

        queries = self.block_queries.unsqueeze(0).expand(B, -1, -1)
        x = fused_h.unsqueeze(1).expand(-1, K, -1) + queries

        causal_mask = torch.triu(torch.full((K, K), float("-inf"), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        draft_features = self.output_norm(x) # [B, K, hidden_size]
        base_logits = F.linear(draft_features, base_lm_head_weight)

        # Compute Markov transition bias if input tokens provided
        markov_emb = None
        markov_logits = None
        if input_token_ids is not None:
            # prev_w1: [B, K, markov_rank]
            prev_w1 = F.embedding(input_token_ids, self.markov_w1)
            markov_logits = torch.matmul(prev_w1, self.markov_w2.t()) # [B, K, vocab_size]
            markov_emb = prev_w1
        else:
            markov_emb = torch.zeros(B, K, self.markov_rank, device=x.device)

        # Total combined logits
        combined_logits = base_logits + (markov_logits if markov_logits is not None else 0)

        # Confidence prediction from fused features + markov representation
        conf_in = torch.cat([draft_features, markov_emb], dim=-1)
        confidence_logits = self.confidence_head(conf_in).squeeze(-1) # [B, K]
        confidence_probs = torch.sigmoid(confidence_logits)

        return {
            "draft_features": draft_features,
            "base_logits": base_logits,
            "markov_logits": markov_logits,
            "logits": combined_logits,
            "confidence_probs": confidence_probs,
            "confidence_logits": confidence_logits,
        }

# Alias for backwards compatibility
DSparkDraftModel = DSparkStandaloneModel

if __name__ == "__main__":
    lm_head = torch.randn(65536, 1024)

    # Test Standalone Model
    standalone = DSparkStandaloneModel()
    h_final = torch.randn(2, 1024)
    out_s = standalone(h_final, lm_head)
    print("Standalone logits shape:", out_s["logits"].shape)

    # Test Markov Model
    markov = DSparkMarkovModel(target_layers=[3, 7, 11])
    h_targets = [torch.randn(2, 1024), torch.randn(2, 1024), torch.randn(2, 1024)]
    prev_tokens = torch.randint(0, 1000, (2, 9))
    out_m = markov(h_targets, lm_head, input_token_ids=prev_tokens)
    print("Markov logits shape:", out_m["logits"].shape)
    print("Markov sidecar params:", sum(p.numel() for p in markov.parameters()) / 1e6, "M")


#!/usr/bin/env python3
"""Create a tiny, deterministic `lfm2` GGUF (random weights, Q8_0 linears).

Used by `lfm2_decode_prefill_identity`: small enough to run anywhere, with the
LFM2 layer mix (conv, conv, attention, conv, attention), 3-tap short conv and a
tied output. Linear weights are Q8_0 so cera's batched-GEMM prefill is taken
(f32 weights fall back to per-token and prove nothing).

Usage: create_lfm2_test_model.py <out.gguf>
"""
import sys

import numpy as np
import gguf

N_EMBD, N_HEAD, N_HEAD_KV, N_FF, VOCAB = 64, 4, 2, 128, 260
HEAD_DIM = N_EMBD // N_HEAD
L_CACHE = 3
LAYERS = ["conv", "conv", "attn", "conv", "attn"]


def rand(rng, *shape, scale=0.08):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def norm_weight(rng, n):
    return (1.0 + 0.2 * rng.standard_normal(n)).astype(np.float32)


def byte_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    return {b: chr(b) for b in bs}


def create(out_path, seed=11):
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(out_path, "lfm2")

    def linear(name, arr):
        w.add_tensor(
            name,
            gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0),
            raw_dtype=gguf.GGMLQuantizationType.Q8_0,
        )

    w.add_context_length(512)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(len(LAYERS))
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv([0 if kind == "conv" else N_HEAD_KV for kind in LAYERS])
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(1_000_000.0)
    w.add_uint32("lfm2.vocab_size", VOCAB)
    w.add_uint32("lfm2.shortconv.l_cache", L_CACHE)

    enc = byte_unicode()
    tokens = [b"<unk>", b"<s>", b"</s>", b"<pad>"] + [
        enc.get(b, f"<0x{b:02X}>").encode("utf-8") for b in range(256)
    ]
    types = [1] * VOCAB
    types[0], types[1], types[2], types[3] = 2, 3, 3, 3
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * VOCAB)
    w.add_token_types(types)
    w.add_token_merges([f"{enc[ord('A')]} {enc[ord('B')]}"])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_pad_token_id(3)

    linear("token_embd.weight", rand(rng, VOCAB, N_EMBD, scale=0.3))
    w.add_tensor("token_embd_norm.weight", norm_weight(rng, N_EMBD))
    for i, kind in enumerate(LAYERS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", norm_weight(rng, N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", norm_weight(rng, N_EMBD))
        linear(f"blk.{i}.ffn_gate.weight", rand(rng, N_FF, N_EMBD))
        linear(f"blk.{i}.ffn_up.weight", rand(rng, N_FF, N_EMBD))
        linear(f"blk.{i}.ffn_down.weight", rand(rng, N_EMBD, N_FF))
        if kind == "conv":
            w.add_tensor(f"blk.{i}.shortconv.conv.weight", rand(rng, N_EMBD, L_CACHE, scale=0.3))
            linear(f"blk.{i}.shortconv.in_proj.weight", rand(rng, 3 * N_EMBD, N_EMBD))
            linear(f"blk.{i}.shortconv.out_proj.weight", rand(rng, N_EMBD, N_EMBD))
        else:
            linear(f"blk.{i}.attn_q.weight", rand(rng, N_HEAD * HEAD_DIM, N_EMBD))
            linear(f"blk.{i}.attn_k.weight", rand(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
            linear(f"blk.{i}.attn_v.weight", rand(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
            linear(f"blk.{i}.attn_output.weight", rand(rng, N_EMBD, N_HEAD * HEAD_DIM))
            w.add_tensor(f"blk.{i}.attn_q_norm.weight", norm_weight(rng, HEAD_DIM))
            w.add_tensor(f"blk.{i}.attn_k_norm.weight", norm_weight(rng, HEAD_DIM))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Created lfm2 test model at {out_path}")


if __name__ == "__main__":
    create(sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_lfm2.gguf")

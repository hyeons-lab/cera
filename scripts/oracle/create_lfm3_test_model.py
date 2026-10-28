#!/usr/bin/env python3
"""Create a tiny, deterministic `lfm3` GGUF (random f32 weights) for the parity test.

The model is small enough to run in a second (CI does not install numpy and gguf, so
the Rust tests skip it there unless python has them) but exercises every way `lfm3` differs from
`lfm2`, so a wrong implementation cannot hide behind fluent text:

  * conv layers with a 7-tap kernel (`shortconv.l_cache = 7`, LFM2 has 3),
  * sandwich norms (`post_attention_norm`, `post_ffw_norm`) with non-unit weights,
  * a per-head attention gate (`attn_gate`),
  * a sliding-window layer (window 4, rope base 10000, whole head rotated) next to
    global layers (rope base 1e6, partial rotary through `rope_freqs.weight`, whose
    trailing pairs carry 1e30),
  * an untied `output.weight` that differs from `token_embd.weight`.

Layer plan: conv, conv, global attn, conv, sliding attn, global attn. A 15 token
prompt is therefore longer than the window.

`--q8_0` stores the linear weights as Q8_0 instead of f32. The f32 file is the
tight graph check against llama.cpp; the Q8_0 file takes cera's batched-GEMM
prefill path (f32 weights may not), which is where the multi-tap conv lives.

Usage: create_lfm3_test_model.py <out.gguf> [--q8_0]
"""
import sys

import numpy as np
import gguf

N_EMBD = 64
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = N_EMBD // N_HEAD  # 16
N_FF = 128
L_CACHE = 7
SLIDING_WINDOW = 4
VOCAB = 260
# conv, conv, global, conv, sliding, global
LAYERS = ["conv", "conv", "full", "conv", "swa", "full"]


def rand(rng, *shape, scale=0.08):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def norm_weight(rng, n):
    # not all ones: a norm that is skipped or applied twice must change the output
    return (1.0 + 0.2 * rng.standard_normal(n)).astype(np.float32)


def byte_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    return {b: chr(b) for b in bs}


def create(out_path, seed=7, q8_0=False):
    rng = np.random.default_rng(seed)
    arch = "lfm3"
    w = gguf.GGUFWriter(out_path, arch)

    def linear(name, arr):
        """Add a linear weight, as Q8_0 when asked (rows are multiples of 32 wide)."""
        if q8_0 and arr.ndim == 2 and arr.shape[1] % 32 == 0:
            w.add_tensor(
                name,
                gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0),
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
        else:
            w.add_tensor(name, arr)

    n_layer = len(LAYERS)
    w.add_context_length(512)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv([0 if kind == "conv" else N_HEAD_KV for kind in LAYERS])
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(1_000_000.0)
    w.add_float32(f"{arch}.rope.freq_base_swa", 10_000.0)
    w.add_array(f"{arch}.attention.sliding_window_pattern", [kind == "swa" for kind in LAYERS])
    w.add_uint32(f"{arch}.attention.sliding_window", SLIDING_WINDOW)
    w.add_uint32(f"{arch}.vocab_size", VOCAB)
    w.add_uint32(f"{arch}.shortconv.l_cache", L_CACHE)

    # Byte-level vocabulary; no merges that fire on the test prompt.
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
    w.add_add_bos_token(False)

    linear("token_embd.weight", rand(rng, VOCAB, N_EMBD, scale=0.3))
    w.add_tensor("output_norm.weight", norm_weight(rng, N_EMBD))
    linear("output.weight", rand(rng, VOCAB, N_EMBD, scale=0.3))
    # partial rotary on the global layers: the last half of the pairs is masked out
    pairs = HEAD_DIM // 2
    w.add_tensor(
        "rope_freqs.weight",
        np.array([1.0] * (pairs // 2) + [1e30] * (pairs - pairs // 2), dtype=np.float32),
    )

    for i, kind in enumerate(LAYERS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", norm_weight(rng, N_EMBD))
        w.add_tensor(f"blk.{i}.post_attention_norm.weight", norm_weight(rng, N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", norm_weight(rng, N_EMBD))
        w.add_tensor(f"blk.{i}.post_ffw_norm.weight", norm_weight(rng, N_EMBD))
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
            # a strong gate so a missing sigmoid-gate is not hidden by small activations
            linear(f"blk.{i}.attn_gate.weight", rand(rng, N_HEAD, N_EMBD, scale=0.5))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Created lfm3 test model at {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a tiny synthetic LFM3 test GGUF.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/tmp/test_lfm3.gguf",
        help="Output GGUF file path (default: /tmp/test_lfm3.gguf)",
    )
    parser.add_argument(
        "--q8_0",
        action="store_true",
        help="Quantize 2D linear weights to Q8_0 instead of F32",
    )
    parsed = parser.parse_args()
    create(parsed.path, q8_0=parsed.q8_0)

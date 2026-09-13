#!/usr/bin/env python3
"""Generate paired synthetic Hugging Face SafeTensors repo and reference GGUF model."""

import argparse
import json
import os
import sys
import numpy as np
import gguf
from safetensors.numpy import save_file


def rand_weight(*shape):
    return (np.random.randn(*shape).astype(np.float32) * 0.05)


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def build_vocab_and_merges():
    byte_encoder = bytes_to_unicode()
    special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    vocab = {}
    for i, tok in enumerate(special_tokens):
        vocab[tok] = i
    for b in range(256):
        vocab[byte_encoder[b]] = len(vocab)
    all_tokens = list(vocab.keys())
    merges = [f"{byte_encoder[ord('A')]} {byte_encoder[ord('B')]}"]
    return vocab, all_tokens, merges


def generate_paired_model(out_dir, arch="llama", seed=42):
    np.random.seed(seed)
    hf_dir = os.path.join(out_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    ref_gguf_path = os.path.join(out_dir, "reference.gguf")

    n_embd = 64
    n_head = 4
    n_head_kv = 2
    head_dim = n_embd // n_head  # 16
    n_ff = 128
    n_layers = 2
    vocab_size = 260

    vocab, all_tokens, merges = build_vocab_and_merges()

    # 1. Write tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": True},
            {"id": 1, "content": "<eos>", "special": True},
            {"id": 2, "content": "<bos>", "special": True},
            {"id": 3, "content": "<unk>", "special": True},
        ],
    }
    with open(os.path.join(hf_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f, indent=2)

    # 2. Write config.json
    config_json = {
        "architectures": [f"{arch.capitalize()}ForCausalLM"],
        "model_type": arch,
        "hidden_size": n_embd,
        "intermediate_size": n_ff,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head_kv,
        "num_hidden_layers": n_layers,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }

    if arch == "nanbeige":
        config_json["num_loops"] = 2
        config_json["skip_loop_final_norm"] = False
    elif arch == "minicpm":
        config_json["scale_emb"] = 1.0
        config_json["scale_depth"] = 1.0
        config_json["dim_model_base"] = n_embd
    elif arch == "gemma2":
        config_json["attn_logit_softcapping"] = 50.0
        config_json["final_logit_softcapping"] = 30.0
        config_json["sliding_window"] = 128
    elif arch == "olmo2":
        config_json["sliding_window"] = 128

    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    # 3. Generate shared weights
    hf_tensors = {}
    gguf_tensors = {}

    # Global weights
    w_emb = rand_weight(vocab_size, n_embd)
    w_out = rand_weight(vocab_size, n_embd)
    w_norm = np.ones((n_embd,), dtype=np.float32)

    hf_tensors["model.embed_tokens.weight"] = w_emb
    hf_tensors["lm_head.weight"] = w_out
    hf_tensors["model.norm.weight"] = w_norm

    gguf_tensors["token_embd.weight"] = w_emb
    gguf_tensors["output.weight"] = w_out
    if arch in ("gemma", "gemma2"):
        gguf_tensors["output_norm.weight"] = w_norm + 1.0
    else:
        gguf_tensors["output_norm.weight"] = w_norm

    for i in range(n_layers):
        attn_norm = np.ones((n_embd,), dtype=np.float32)
        q = rand_weight(n_head * head_dim, n_embd)
        k = rand_weight(n_head_kv * head_dim, n_embd)
        v = rand_weight(n_head_kv * head_dim, n_embd)
        o = rand_weight(n_embd, n_head * head_dim)

        ffn_norm = np.ones((n_embd,), dtype=np.float32)
        gate = rand_weight(n_ff, n_embd)
        up = rand_weight(n_ff, n_embd)
        down = rand_weight(n_embd, n_ff)

        if arch in ("gemma", "gemma2"):
            pre_ffn = np.ones((n_embd,), dtype=np.float32)
            post_ffn = np.ones((n_embd,), dtype=np.float32)
            hf_tensors[f"model.layers.{i}.input_layernorm.weight"] = attn_norm
            hf_tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = ffn_norm
            hf_tensors[f"model.layers.{i}.pre_feedforward_layernorm.weight"] = pre_ffn
            hf_tensors[f"model.layers.{i}.post_feedforward_layernorm.weight"] = post_ffn

            gguf_tensors[f"blk.{i}.attn_norm.weight"] = attn_norm + 1.0
            gguf_tensors[f"blk.{i}.attn_post_norm.weight"] = ffn_norm + 1.0
            gguf_tensors[f"blk.{i}.ffn_norm.weight"] = pre_ffn + 1.0
            gguf_tensors[f"blk.{i}.ffn_post_norm.weight"] = post_ffn + 1.0
        else:
            hf_tensors[f"model.layers.{i}.input_layernorm.weight"] = attn_norm
            hf_tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = ffn_norm
            gguf_tensors[f"blk.{i}.attn_norm.weight"] = attn_norm
            gguf_tensors[f"blk.{i}.ffn_norm.weight"] = ffn_norm

        hf_tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        hf_tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = k
        hf_tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = v
        hf_tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = o

        hf_tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = gate
        hf_tensors[f"model.layers.{i}.mlp.up_proj.weight"] = up
        hf_tensors[f"model.layers.{i}.mlp.down_proj.weight"] = down

        gguf_tensors[f"blk.{i}.attn_q.weight"] = q
        gguf_tensors[f"blk.{i}.attn_k.weight"] = k
        gguf_tensors[f"blk.{i}.attn_v.weight"] = v
        gguf_tensors[f"blk.{i}.attn_output.weight"] = o

        gguf_tensors[f"blk.{i}.ffn_gate.weight"] = gate
        gguf_tensors[f"blk.{i}.ffn_up.weight"] = up
        gguf_tensors[f"blk.{i}.ffn_down.weight"] = down

    # 4. Save SafeTensors file
    save_file(hf_tensors, os.path.join(hf_dir, "model.safetensors"))

    # 5. Save reference GGUF file
    writer = gguf.GGUFWriter(ref_gguf_path, arch)
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    if arch == "nanbeige":
        writer.add_uint32("nanbeige.num_loops", 2)
        writer.add_bool("nanbeige.skip_loop_final_norm", False)
    elif arch == "minicpm":
        writer.add_float32("minicpm.embedding_scale", 1.0)
        writer.add_float32("minicpm.residual_scale", float(1.0 / np.sqrt(n_layers)))
        writer.add_float32("minicpm.logit_scale", float(n_embd / n_embd))
    elif arch == "gemma2":
        writer.add_float32("gemma2.attn_logit_softcapping", 50.0)
        writer.add_float32("gemma2.final_logit_softcapping", 30.0)
        writer.add_uint32("gemma2.attention.sliding_window", 128)
    elif arch == "olmo2":
        writer.add_uint32("olmo2.attention.sliding_window", 128)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt2")
    writer.add_token_list(all_tokens)
    writer.add_token_scores([0.0] * len(all_tokens))
    token_types = [1] * len(all_tokens)
    for idx in range(4):
        token_types[idx] = 3
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(0)

    for name, tensor in gguf_tensors.items():
        writer.add_tensor(name, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Generated paired {arch} model in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired synthetic SafeTensors and reference GGUF model"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Target directory where hf/ and reference.gguf will be created",
    )
    parser.add_argument(
        "--arch",
        default="llama",
        choices=["llama", "nanbeige", "minicpm", "gemma2", "olmo2"],
        help="Model architecture",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialization",
    )
    args = parser.parse_args()
    generate_paired_model(args.out_dir, arch=args.arch, seed=args.seed)


if __name__ == "__main__":
    main()

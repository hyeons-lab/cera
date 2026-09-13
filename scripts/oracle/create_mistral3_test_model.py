#!/usr/bin/env python3
import os
import sys
import numpy as np
import gguf

def rand_weight(*shape):
    return (np.random.randn(*shape).astype(np.float32) * 0.05)

def create_mistral3_model(out_path, seed=42):
    np.random.seed(seed)
    arch = "mistral3"
    parent_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    writer = gguf.GGUFWriter(tmp_path, arch)

    n_embd = 64
    n_head = 4
    n_head_kv = 2
    head_dim = n_embd // n_head  # 16
    n_ff = 128
    n_layers = 4
    vocab_size = 260
    orig_ctx_len = 64

    # Architecture metadata
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    # Mistral 3 YaRN RoPE metadata
    writer.add_string(f"{arch}.rope.scaling.type", "yarn")
    writer.add_float32(f"{arch}.rope.scaling.factor", 2.0)
    writer.add_uint32(f"{arch}.rope.scaling.original_context_length", orig_ctx_len)
    writer.add_float32(f"{arch}.rope.scaling.yarn_log_multiplier", 0.0707)
    writer.add_float32(f"{arch}.rope.scaling.yarn_ext_factor", 1.0)
    writer.add_float32(f"{arch}.rope.scaling.attn_factor", 1.0)
    writer.add_float32(f"{arch}.rope.scaling.yarn_beta_fast", 32.0)
    writer.add_float32(f"{arch}.rope.scaling.yarn_beta_slow", 1.0)

    # Mistral 3 attention temperature scaling metadata
    writer.add_float32(f"{arch}.attention.temperature_scale", 0.1)

    # Vocabulary and Tekken tokenizer setup
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

    byte_encoder = bytes_to_unicode()
    special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    all_tokens = list(special_tokens)
    for b in range(256):
        all_tokens.append(byte_encoder[b])

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("tekken")
    writer.add_token_list(all_tokens)
    token_scores = [0.0] * len(all_tokens)
    writer.add_token_scores(token_scores)
    token_types = [1] * len(all_tokens)
    for i in range(len(special_tokens)):
        token_types[i] = 3
    writer.add_token_types(token_types)
    merge_t = f"{byte_encoder[ord('A')]} {byte_encoder[ord('B')]}"
    writer.add_token_merges([merge_t])
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(0)

    # Base tensors
    writer.add_tensor("token_embd.weight", rand_weight(vocab_size, n_embd))
    writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    writer.add_tensor("output.weight", rand_weight(vocab_size, n_embd))

    for i in range(n_layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(n_head * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(n_head_kv * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(n_head_kv * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_head * head_dim))
        # Optional projection bias
        writer.add_tensor(f"blk.{i}.attn_output.bias", rand_weight(n_embd))

        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_gate.bias", rand_weight(n_ff))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.bias", rand_weight(n_ff))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))
        writer.add_tensor(f"blk.{i}.ffn_down.bias", rand_weight(n_embd))

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_path>")
        sys.exit(1)
    create_mistral3_model(sys.argv[1])

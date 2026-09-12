#!/usr/bin/env python3
import os
import sys
import tempfile
import numpy as np
import gguf

def rand_weight(*shape):
    return (np.random.randn(*shape).astype(np.float32) * 0.05)

def create_olmo3_model(out_path, seed=42):
    np.random.seed(seed)
    arch = "olmo2"
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    writer = gguf.GGUFWriter(tmp_path, arch)

    n_embd = 64
    n_head = 4
    n_head_kv = 2
    head_dim = n_embd // n_head  # 16
    n_ff = 128
    n_layer = 4
    vocab_size = 260
    sliding_window = 2

    # SWA pattern: layers 0, 1, 2 use SWA (window=2); layer 3 is dense (full attention + YaRN)
    swa_pattern = [1, 1, 1, 0]

    # Architecture metadata
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    # SWA metadata
    writer.add_uint32(f"{arch}.attention.sliding_window", sliding_window)
    writer.add_array(f"{arch}.attention.sliding_window_pattern", swa_pattern)

    # YaRN RoPE scaling metadata
    writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
    writer.add_rope_scaling_factor(2.0)
    writer.add_rope_scaling_orig_ctx_len(64)
    writer.add_float32(f"{arch}.rope.scaling.attn_factor", 1.069315)
    writer.add_float32(f"{arch}.rope.scaling.yarn_beta_fast", 32.0)
    writer.add_float32(f"{arch}.rope.scaling.yarn_beta_slow", 1.0)
    writer.add_float32(f"{arch}.rope.scaling.yarn_ext_factor", 1.0)

    # Vocabulary
    def bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
            + [ord(" ")]
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
    writer.add_tokenizer_pre("default")
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

    # Tensors
    writer.add_tensor("token_embd.weight", rand_weight(vocab_size, n_embd))
    writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    writer.add_tensor("output.weight", rand_weight(vocab_size, n_embd))

    for i in range(n_layer):
        # Attention
        writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(n_head * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(n_head_kv * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(n_head_kv * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_head * head_dim))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", np.ones((n_head * head_dim,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", np.ones((n_head_kv * head_dim,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", np.ones((n_embd,), dtype=np.float32))

        # FFN
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))
        writer.add_tensor(f"blk.{i}.post_ffw_norm.weight", np.ones((n_embd,), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.replace(tmp_path, out_path)
    print(f"Created {arch} test model at {out_path}")

if __name__ == "__main__":
    out_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(tempfile.gettempdir(), f"test_olmo3_{os.getpid()}.gguf")
    )
    create_olmo3_model(out_path)

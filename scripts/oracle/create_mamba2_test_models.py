#!/usr/bin/env python3
import sys
import numpy as np
import gguf

def create_granite_hybrid(out_path, seed=42):
    np.random.seed(seed)
    arch = "granitehybrid"
    writer = gguf.GGUFWriter(out_path, arch)

    n_embd = 64
    d_inner = 128 # 2 * n_embd
    d_state = 16
    d_conv = 4
    n_ssm_head = 4 # time_step_rank
    n_group = 1
    head_dim_ssm = d_inner // n_ssm_head # 32
    conv_dim = d_inner + 2 * n_group * d_state # 160
    d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head # 292

    n_head = 4
    head_dim_attn = n_embd // n_head # 16
    n_ff = 128
    n_layer = 2 # layer 0: recurrent, layer 1: attention

    # Architecture metadata
    writer.add_architecture()
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    # kv head count per layer: [0, 2]
    writer.add_array(f"{arch}.attention.head_count_kv", [0, 2])
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    # SSM metadata
    writer.add_uint32(f"{arch}.ssm.conv_kernel", d_conv)
    writer.add_uint32(f"{arch}.ssm.inner_size", d_inner)
    writer.add_uint32(f"{arch}.ssm.state_size", d_state)
    writer.add_uint32(f"{arch}.ssm.time_step_rank", n_ssm_head)
    writer.add_uint32(f"{arch}.ssm.group_count", n_group)

    # Granite scalars
    writer.add_float32(f"{arch}.residual_scale", 0.22)
    writer.add_float32(f"{arch}.logit_scale", 16.0)

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
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    byte_encoder = bytes_to_unicode()
    tokens = [b"<unk>", b"<s>", b"</s>", b"<pad>"]
    for b in range(256):
        tokens.append(byte_encoder[b].encode("utf-8"))

    vocab_size = len(tokens) # 260
    token_types = [1] * vocab_size
    token_types[0] = 2
    token_types[1] = 3
    token_types[2] = 3
    token_types[3] = 3

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    merge_t = f"{byte_encoder[ord('A')]} {byte_encoder[ord('B')]}"
    writer.add_token_merges([merge_t])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    def rand_weight(*shape):
        return (np.random.randn(*shape) * 0.05).astype(np.float32)

    tok_embd = rand_weight(vocab_size, n_embd)
    writer.add_tensor("token_embd.weight", tok_embd)
    writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    writer.add_tensor("output.weight", rand_weight(vocab_size, n_embd))

    for i in range(n_layer):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones((n_embd,), dtype=np.float32))

        if i == 0:
            # Layer 0: Recurrent (Mamba2)
            # In-projection: [d_in_proj, n_embd]
            writer.add_tensor(f"blk.{i}.ssm_in.weight", rand_weight(d_in_proj, n_embd))
            # 1D conv: [conv_dim, d_conv]
            writer.add_tensor(f"blk.{i}.ssm_conv1d.weight", rand_weight(conv_dim, d_conv))
            writer.add_tensor(f"blk.{i}.ssm_conv1d.bias", np.zeros((conv_dim,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.ssm_dt.bias", np.zeros((n_ssm_head,), dtype=np.float32))
            # ssm_a: [1, n_ssm_head] in GGUF -> (n_ssm_head, 1) in numpy
            writer.add_tensor(f"blk.{i}.ssm_a", -np.ones((n_ssm_head, 1), dtype=np.float32))
            # ssm_d: [1, n_ssm_head] in GGUF -> (n_ssm_head, 1) in numpy
            writer.add_tensor(f"blk.{i}.ssm_d", np.ones((n_ssm_head, 1), dtype=np.float32))
            # ssm_norm: [n_group, d_inner // n_group] = [1, 128]
            writer.add_tensor(f"blk.{i}.ssm_norm.weight", np.ones((d_inner,), dtype=np.float32))
            # ssm_out: [n_embd, d_inner]
            writer.add_tensor(f"blk.{i}.ssm_out.weight", rand_weight(n_embd, d_inner))
        else:
            # Layer 1: Attention
            writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(n_embd, n_embd))
            writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(2 * head_dim_attn, n_embd))
            writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(2 * head_dim_attn, n_embd))
            writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_embd))

        # FFN
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Created {arch} model at {out_path}")

def create_falcon_h1(out_path, seed=42):
    np.random.seed(seed)
    arch = "falcon-h1"
    writer = gguf.GGUFWriter(out_path, arch)

    n_embd = 64
    d_inner = 128
    d_state = 16
    d_conv = 4
    n_ssm_head = 4
    n_group = 1
    head_dim_ssm = d_inner // n_ssm_head # 32
    conv_dim = d_inner + 2 * n_group * d_state # 160
    d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head # 292

    n_head = 4
    n_head_kv = 2
    head_dim_attn = n_embd // n_head # 16
    n_ff = 128
    n_layer = 2 # parallel attention + ssm on all layers

    # Architecture metadata
    writer.add_architecture()
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    # SSM metadata
    writer.add_uint32(f"{arch}.ssm.conv_kernel", d_conv)
    writer.add_uint32(f"{arch}.ssm.inner_size", d_inner)
    writer.add_uint32(f"{arch}.ssm.state_size", d_state)
    writer.add_uint32(f"{arch}.ssm.time_step_rank", n_ssm_head)
    writer.add_uint32(f"{arch}.ssm.group_count", n_group)

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
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    byte_encoder = bytes_to_unicode()
    tokens = [b"<unk>", b"<s>", b"</s>", b"<pad>"]
    for b in range(256):
        tokens.append(byte_encoder[b].encode("utf-8"))

    vocab_size = len(tokens)
    token_types = [1] * vocab_size
    token_types[0] = 2
    token_types[1] = 3
    token_types[2] = 3
    token_types[3] = 3

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    merge_t = f"{byte_encoder[ord('A')]} {byte_encoder[ord('B')]}"
    writer.add_token_merges([merge_t])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    def rand_weight(*shape):
        return (np.random.randn(*shape) * 0.05).astype(np.float32)

    tok_embd = rand_weight(vocab_size, n_embd)
    writer.add_tensor("token_embd.weight", tok_embd)
    writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    writer.add_tensor("output.weight", rand_weight(vocab_size, n_embd))

    for i in range(n_layer):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones((n_embd,), dtype=np.float32))

        # Both Attention and Mamba2 on every layer
        writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(n_embd, n_embd))
        writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(n_head_kv * head_dim_attn, n_embd))
        writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(n_head_kv * head_dim_attn, n_embd))
        writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_embd))

        writer.add_tensor(f"blk.{i}.ssm_in.weight", rand_weight(d_in_proj, n_embd))
        writer.add_tensor(f"blk.{i}.ssm_conv1d.weight", rand_weight(conv_dim, d_conv))
        writer.add_tensor(f"blk.{i}.ssm_conv1d.bias", np.zeros((conv_dim,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ssm_dt.bias", np.zeros((n_ssm_head,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ssm_a", -np.ones((n_ssm_head, 1), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ssm_d", np.ones((n_ssm_head, 1), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ssm_norm.weight", np.ones((d_inner,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ssm_out.weight", rand_weight(n_embd, d_inner))

        # FFN
        writer.add_tensor(f"blk.{i}.ffn_norm", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Created {arch} model at {out_path}")

if __name__ == "__main__":
    create_granite_hybrid("/tmp/test_granite_hybrid.gguf")
    create_falcon_h1("/tmp/test_falcon_h1.gguf")

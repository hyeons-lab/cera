import os
import sys
import numpy as np
import gguf

def create_qwen35_test_model(out_path, seed=42):
    np.random.seed(seed)
    arch = "qwen35"
    parent_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    writer = gguf.GGUFWriter(tmp_path, arch)

    n_embd = 64
    head_dim = 16
    n_head = 4
    n_head_kv = 2
    n_layer = 4
    n_ff = 128

    # SSM / Delta Net parameters
    d_state = 16
    n_k_heads = 2
    n_v_heads = 4
    d_inner = n_v_heads * d_state # 64
    d_conv = 4
    key_dim = n_k_heads * d_state # 32
    value_dim = n_v_heads * d_state # 64
    conv_dim = 2 * key_dim + value_dim # 128

    # Metadata
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    writer.add_uint32(f"{arch}.ssm.conv_kernel", d_conv)
    writer.add_uint32(f"{arch}.ssm.inner_size", d_inner)
    writer.add_uint32(f"{arch}.ssm.state_size", d_state)
    writer.add_uint32(f"{arch}.ssm.time_step_rank", n_v_heads)
    writer.add_uint32(f"{arch}.ssm.group_count", n_k_heads)
    writer.add_uint32(f"{arch}.full_attention_interval", 4)

    # Tokenizer: byte-level BPE
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
    writer.add_tokenizer_pre("qwen2")
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
        writer.add_tensor(f"blk.{i}.attn_post_norm.weight", np.ones((n_embd,), dtype=np.float32))

        is_recr = (i + 1) % 4 != 0
        if is_recr:
            # Gated Delta Net recurrent layer
            writer.add_tensor(f"blk.{i}.attn_qkv.weight", rand_weight(conv_dim, n_embd))
            writer.add_tensor(f"blk.{i}.attn_gate.weight", rand_weight(value_dim, n_embd))
            writer.add_tensor(f"blk.{i}.ssm_conv1d.weight", rand_weight(conv_dim, d_conv))
            writer.add_tensor(f"blk.{i}.ssm_conv1d.bias", np.zeros((conv_dim,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.ssm_dt.bias", np.zeros((n_v_heads,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.ssm_a", -np.ones((n_v_heads,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.ssm_beta.weight", rand_weight(n_v_heads, n_embd))
            writer.add_tensor(f"blk.{i}.ssm_alpha.weight", rand_weight(n_v_heads, n_embd))
            writer.add_tensor(f"blk.{i}.ssm_norm.weight", np.ones((d_state,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.ssm_out.weight", rand_weight(n_embd, value_dim))
        else:
            # Full attention layer with Q-gating and QK-norm
            writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(2 * n_head * head_dim, n_embd))
            writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(n_head_kv * head_dim, n_embd))
            writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(n_head_kv * head_dim, n_embd))
            writer.add_tensor(f"blk.{i}.attn_q_norm.weight", np.ones((head_dim,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.attn_k_norm.weight", np.ones((head_dim,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_head * head_dim))

        # FFN
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(tmp_path, out_path)
        print(f"Created {arch} model at {out_path}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: create_qwen35_test_model.py <output.gguf>")
        sys.exit(1)
    create_qwen35_test_model(sys.argv[1])

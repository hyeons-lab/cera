import os
import sys
import numpy as np
import gguf

def create_bailingmoe3_test_model(out_path, seed=42):
    np.random.seed(seed)
    arch = "bailingmoe3"
    parent_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    writer = gguf.GGUFWriter(tmp_path, arch)

    n_embd = 64
    n_head = 4
    n_layer = 4
    n_ff = 128

    # KDA parameters
    kda_head_dim = 16
    d_inner = n_head * kda_head_dim  # 64
    d_conv = 4
    kda_lower_bound = -5.0

    # MLA parameters
    qk_nope_head_dim = 12
    qk_rope_head_dim = 4
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 16
    v_head_dim = 16
    kv_lora_rank = 32
    q_lora_rank = 16
    key_length = kv_lora_rank + qk_rope_head_dim  # 36

    # MoE parameters
    n_expert = 4
    n_expert_used = 2
    moe_ff = 32
    moe_shared_ff = 32
    routed_scaling_factor = 2.5
    leading_dense_block_count = 1

    # Metadata
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv([0, 0, 0, 1])
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)

    writer.add_uint32(f"{arch}.ssm.conv_kernel", d_conv)
    writer.add_uint32(f"{arch}.kda.head_dim", kda_head_dim)
    writer.add_bool(f"{arch}.kda.safe_gate", True)
    writer.add_float32(f"{arch}.kda.gate_lower_bound", kda_lower_bound)

    writer.add_uint32(f"{arch}.attention.kv_lora_rank", kv_lora_rank)
    writer.add_uint32(f"{arch}.attention.q_lora_rank", q_lora_rank)
    writer.add_uint32(f"{arch}.rope.dimension_count", qk_rope_head_dim)
    writer.add_uint32(f"{arch}.attention.key_length", key_length)
    writer.add_uint32(f"{arch}.attention.key_length_mla", qk_head_dim)
    writer.add_uint32(f"{arch}.attention.value_length_mla", v_head_dim)

    writer.add_uint32(f"{arch}.expert_count", n_expert)
    writer.add_uint32(f"{arch}.expert_used_count", n_expert_used)
    writer.add_uint32(f"{arch}.expert_feed_forward_length", moe_ff)
    writer.add_uint32(f"{arch}.expert_shared_feed_forward_length", moe_shared_ff)
    writer.add_uint32(f"{arch}.expert_shared_count", 1)
    writer.add_uint32(f"{arch}.leading_dense_block_count", leading_dense_block_count)
    writer.add_float32(f"{arch}.expert_weights_scale", routed_scaling_factor)
    writer.add_bool(f"{arch}.expert_weights_norm", True)

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
    tokens.append((byte_encoder[ord("A")] + byte_encoder[ord("B")]).encode("utf-8"))

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
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32))

        is_recr = (i + 1) % 4 != 0
        if is_recr:
            # KDA recurrent layer
            writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(d_inner, n_embd))
            writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(d_inner, n_embd))
            writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(d_inner, n_embd))

            writer.add_tensor(f"blk.{i}.ssm_conv1d_q.weight", rand_weight(d_inner, d_conv))
            writer.add_tensor(f"blk.{i}.ssm_conv1d_k.weight", rand_weight(d_inner, d_conv))
            writer.add_tensor(f"blk.{i}.ssm_conv1d_v.weight", rand_weight(d_inner, d_conv))

            writer.add_tensor(f"blk.{i}.ssm_f_a.weight", rand_weight(d_inner, n_embd))
            writer.add_tensor(f"blk.{i}.ssm_dt.bias", rand_weight(d_inner))
            writer.add_tensor(f"blk.{i}.ssm_a", np.ones((n_head,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.ssm_beta.weight", rand_weight(n_head, n_embd))
            writer.add_tensor(f"blk.{i}.ssm_g_a.weight", rand_weight(d_inner, n_embd))
            writer.add_tensor(f"blk.{i}.ssm_norm.weight", np.ones((kda_head_dim,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, d_inner))
        else:
            # MLA full attention layer
            writer.add_tensor(f"blk.{i}.attn_q_a.weight", rand_weight(q_lora_rank, n_embd))
            writer.add_tensor(f"blk.{i}.attn_q_a_norm.weight", np.ones((q_lora_rank,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.attn_q_b.weight", rand_weight(n_head * qk_head_dim, q_lora_rank))

            writer.add_tensor(f"blk.{i}.attn_kv_a_mqa.weight", rand_weight(key_length, n_embd))
            writer.add_tensor(f"blk.{i}.attn_kv_a_norm.weight", np.ones((kv_lora_rank,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.attn_k_b.weight", rand_weight(n_head, kv_lora_rank, qk_nope_head_dim))
            writer.add_tensor(f"blk.{i}.attn_v_b.weight", rand_weight(n_head, v_head_dim, kv_lora_rank))
            writer.add_tensor(f"blk.{i}.attn_gate.weight", rand_weight(n_head, n_embd))
            writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_head * v_head_dim))

        # FFN
        if i < leading_dense_block_count:
            writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
            writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
            writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))
        else:
            writer.add_tensor(f"blk.{i}.ffn_gate_inp.weight", rand_weight(n_expert, n_embd))
            writer.add_tensor(f"blk.{i}.exp_probs_b.bias", rand_weight(n_expert))
            writer.add_tensor(f"blk.{i}.ffn_gate_exps.weight", rand_weight(n_expert, moe_ff, n_embd))
            writer.add_tensor(f"blk.{i}.ffn_up_exps.weight", rand_weight(n_expert, moe_ff, n_embd))
            writer.add_tensor(f"blk.{i}.ffn_down_exps.weight", rand_weight(n_expert, n_embd, moe_ff))
            writer.add_tensor(f"blk.{i}.ffn_gate_shexp.weight", rand_weight(moe_shared_ff, n_embd))
            writer.add_tensor(f"blk.{i}.ffn_up_shexp.weight", rand_weight(moe_shared_ff, n_embd))
            writer.add_tensor(f"blk.{i}.ffn_down_shexp.weight", rand_weight(n_embd, moe_shared_ff))

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
    if len(sys.argv) > 1:
        out = sys.argv[1]
    else:
        out = "test_models/bailingmoe3_test.gguf"
    create_bailingmoe3_test_model(out)
    print(f"Created {out}")

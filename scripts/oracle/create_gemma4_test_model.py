#!/usr/bin/env python3
import sys
import numpy as np
import gguf

def rand_weight(*shape):
    return (np.random.randn(*shape).astype(np.float32) * 0.05)

def create_gemma4_model(out_path, seed=42):
    np.random.seed(seed)
    arch = "gemma4"
    writer = gguf.GGUFWriter(out_path, arch)

    n_embd = 64
    n_head = 4
    n_head_kv = 2
    head_dim = n_embd // n_head # 16
    n_ff = 128
    n_layer = 4
    n_kv_shared_layers = 1
    n_layer_kv_from_start = n_layer - n_kv_shared_layers # 3
    n_embd_per_layer = 16
    vocab_size = 260
    sliding_window = 128

    # SWA pattern: layer 0 (SWA), layer 1 (SWA), layer 2 (Full), layer 3 (SWA shared)
    swa_pattern = [1, 1, 0, 1]

    # Architecture metadata
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_uint32(f"{arch}.attention.key_length_swa", head_dim)
    writer.add_uint32(f"{arch}.attention.value_length_swa", head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_uint32(f"{arch}.rope.dimension_count_swa", head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_float32(f"{arch}.rope.freq_base_swa", 10000.0)

    # Gemma 4 specific metadata
    writer.add_uint32(f"{arch}.attention.shared_kv_layers", n_kv_shared_layers)
    writer.add_uint32(f"{arch}.embedding_length_per_layer_input", n_embd_per_layer)
    writer.add_uint32(f"{arch}.attention.sliding_window", sliding_window)
    writer.add_array(f"{arch}.attention.sliding_window_pattern", swa_pattern)
    writer.add_float32(f"{arch}.final_logit_softcapping", 30.0)

    # Vocabulary
    def bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
            + [ord(" ")]
        )
        return {b: chr(b) for b in bs}

    byte_encoder = bytes_to_unicode()
    tokens = [b"<unk>", b"<s>", b"</s>", b"<pad>"]
    for b in range(256):
        tokens.append(byte_encoder.get(b, f"<0x{b:02X}>").encode("utf-8"))

    token_types = [1] * vocab_size
    token_types[0] = 2
    token_types[1] = 3
    token_types[2] = 3
    token_types[3] = 3

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab_size)
    writer.add_token_types(token_types)
    merge_t = f"{byte_encoder[ord('A')]} {byte_encoder[ord('B')]}"
    writer.add_token_merges([merge_t])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(3)

    # Global tensors
    writer.add_tensor("token_embd.weight", rand_weight(vocab_size, n_embd))
    writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    writer.add_tensor("output.weight", rand_weight(vocab_size, n_embd))
    writer.add_tensor("rope_freqs.weight", np.ones((head_dim // 2,), dtype=np.float32))

    # Per-layer embedding global tensors
    # per_layer_tok_embd: [vocab_size, n_embd_per_layer * n_layer]
    writer.add_tensor("per_layer_token_embd.weight", rand_weight(vocab_size, n_embd_per_layer * n_layer))
    # per_layer_model_proj: [n_embd_per_layer * n_layer, n_embd]
    writer.add_tensor("per_layer_model_proj.weight", rand_weight(n_embd_per_layer * n_layer, n_embd))
    # per_layer_proj_norm: [n_embd_per_layer]
    writer.add_tensor("per_layer_proj_norm.weight", np.ones((n_embd_per_layer,), dtype=np.float32))

    # Per-layer tensors
    for i in range(n_layer):
        has_kv = (i < n_layer_kv_from_start)
        is_swa = bool(swa_pattern[i])

        # Norms
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.post_ffw_norm.weight", np.ones((n_embd,), dtype=np.float32))

        # Attention
        writer.add_tensor(f"blk.{i}.attn_q.weight", rand_weight(n_head * head_dim, n_embd))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", np.ones((head_dim,), dtype=np.float32))
        if has_kv:
            writer.add_tensor(f"blk.{i}.attn_k.weight", rand_weight(n_head_kv * head_dim, n_embd))
            writer.add_tensor(f"blk.{i}.attn_k_norm.weight", np.ones((head_dim,), dtype=np.float32))
            writer.add_tensor(f"blk.{i}.attn_v.weight", rand_weight(n_head_kv * head_dim, n_embd))

        writer.add_tensor(f"blk.{i}.attn_output.weight", rand_weight(n_embd, n_head * head_dim))

        # FFN
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand_weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand_weight(n_embd, n_ff))

        # Per-layer embedding layer tensors
        writer.add_tensor(f"blk.{i}.inp_gate.weight", rand_weight(n_embd_per_layer, n_embd))
        writer.add_tensor(f"blk.{i}.proj.weight", rand_weight(n_embd, n_embd_per_layer))
        writer.add_tensor(f"blk.{i}.post_norm.weight", np.ones((n_embd,), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Created {arch} test model at {out_path}")

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_gemma4.gguf"
    create_gemma4_model(out_path)

"""
Export trained DSpark PyTorch checkpoint to Quantized GGUF format (Q4_0 / Q8_0 / F16).

Supports:
- Option B Standalone (dspark-standalone for Cera)
- Option B Markov (dspark-markov for llama.cpp --spec-type draft-dspark)
"""

import sys
import os
import torch
import numpy as np
import gguf
import gguf.quants as q

def copy_tokenizer_metadata(writer: gguf.GGUFWriter):
    candidates = [
        os.environ.get("BASE_GGUF", ""),
        os.path.expanduser("~/.leap/models/LFM2.5-VL-450M-Q4_0/LFM2.5-VL-450M-Q4_0.gguf"),
        os.path.expanduser("~/.cache/cera/LFM2.5-VL-450M-Q4_0.gguf"),
    ]
    base_gguf_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not base_gguf_path:
        return
    base_reader = gguf.GGUFReader(base_gguf_path)
    for field in base_reader.fields.values():
        if field.name.startswith("tokenizer."):
            val = [field.parts[d] for d in field.data]
            if field.types[0] == gguf.GGUFValueType.STRING:
                writer.add_string(field.name, str(val[0].tobytes().decode("utf-8", errors="ignore")))
            elif field.types[0] == gguf.GGUFValueType.UINT32:
                writer.add_uint32(field.name, int(val[0][0]))
            elif field.types[0] == gguf.GGUFValueType.BOOL:
                writer.add_bool(field.name, bool(val[0][0]))
            elif field.types[0] == gguf.GGUFValueType.ARRAY:
                if field.types[1] == gguf.GGUFValueType.STRING:
                    arr = [str(p.tobytes().decode("utf-8", errors="ignore")) for p in val]
                    writer.add_array(field.name, arr)
                elif field.types[1] == gguf.GGUFValueType.INT32:
                    arr = [int(p[0]) for p in val]
                    writer.add_array(field.name, arr)

def export_quantized_gguf(
    checkpoint_path: str,
    output_gguf_path: str,
    quant_type: str = "Q4_0",
    hidden_size: int = 1024,
    vocab_size: int = 65536,
    num_layers: int = 5,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    intermediate_size: int = 4096,
    block_size: int = 9,
    markov_rank: int = 256,
    target_layers: list[int] | None = None,
):
    if target_layers is None:
        target_layers = [3, 7, 11]
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    is_markov_model = "fc.weight" in state_dict or "markov_w1" in state_dict
    arch_name = "dflash" if is_markov_model else "dspark"

    gguf_type = gguf.GGMLQuantizationType.Q4_0 if quant_type.upper() == "Q4_0" else gguf.GGMLQuantizationType.Q8_0
    gguf_writer = gguf.GGUFWriter(output_gguf_path, arch_name)

    copy_tokenizer_metadata(gguf_writer)

    # Write model architecture metadata
    gguf_writer.add_architecture()
    gguf_writer.add_string("general.architecture", arch_name)
    gguf_writer.add_uint32(f"{arch_name}.block_size", block_size)
    gguf_writer.add_uint32(f"{arch_name}.markov_rank", markov_rank)
    gguf_writer.add_uint32(f"{arch_name}.block_count", num_layers)
    gguf_writer.add_uint32(f"{arch_name}.attention.head_count", num_heads)
    gguf_writer.add_uint32(f"{arch_name}.attention.head_count_kv", num_kv_heads)
    gguf_writer.add_uint32(f"{arch_name}.attention.key_length", hidden_size // num_heads)
    gguf_writer.add_uint32(f"{arch_name}.embedding_length", hidden_size)
    gguf_writer.add_uint32(f"{arch_name}.feed_forward_length", intermediate_size)
    gguf_writer.add_uint32(f"{arch_name}.vocab_size", vocab_size)
    gguf_writer.add_float32(f"{arch_name}.attention.layer_norm_rms_epsilon", 1e-5)
    gguf_writer.add_float32(f"{arch_name}.rope.freq_base", 10000.0)

    def add_maybe_quantized(name: str, tensor: torch.Tensor):
        arr = tensor.float().numpy()
        if arr.ndim >= 2 and arr.size >= 256 and (arr.size % 32 == 0):
            quant_data = q.quantize(arr, gguf_type)
            gguf_writer.add_tensor(name, quant_data, raw_dtype=gguf_type)
        else:
            gguf_writer.add_tensor(name, arr)

    if is_markov_model:
        gguf_writer.add_uint32("dflash.context_length", 2048)
        gguf_writer.add_bool("dflash.sample_from_anchor", True)
        gguf_writer.add_uint32("tokenizer.ggml.mask_token_id", 64402)
        gguf_writer.add_uint32("tokenizer.ggml.padding_token_id", 64402)
        if "target_layers" in state_dict:
            tl = state_dict["target_layers"]
            if isinstance(tl, torch.Tensor):
                target_layers = tl.tolist()
            else:
                target_layers = [int(l) for l in tl]
        gguf_writer.add_array("target_layers", [int(l) for l in target_layers])
        gguf_writer.add_array("dflash.target_layers", [int(l) for l in target_layers])

        if "fc.weight" in state_dict:
            add_maybe_quantized("fc.weight", state_dict["fc.weight"])

        if "enc_norm.weight" in state_dict:
            add_maybe_quantized("enc.output_norm.weight", state_dict["enc_norm.weight"])
    else:
        gguf_writer.add_uint32("dspark.num_layers", num_layers)
        gguf_writer.add_uint32("dspark.hidden_size", hidden_size)

    if "markov_w1" in state_dict and "markov_w2" in state_dict:
        add_maybe_quantized("markov_w1.weight", state_dict["markov_w1"])
        add_maybe_quantized("markov_w2.weight", state_dict["markov_w2"])

    # Confidence projection (supported across standalone and markov models)
    if "confidence_head.weight" in state_dict:
        conf_w = state_dict["confidence_head.weight"].float().numpy().reshape(-1)
        gguf_writer.add_tensor("conf_proj.weight", conf_w)
    elif "confidence_head.0.weight" in state_dict:
        conf_w = state_dict["confidence_head.0.weight"].float().numpy().reshape(-1)
        gguf_writer.add_tensor("conf_proj.weight", conf_w)
        if "confidence_head.0.bias" in state_dict:
            conf_b = state_dict["confidence_head.0.bias"].float().numpy().reshape(-1)
            gguf_writer.add_tensor("conf_proj.bias", conf_b)

    print("Quantizing and packing layers...")
    for l in range(num_layers):
        pfx = f"layers.{l}"
        if f"{pfx}.attn_norm.weight" in state_dict:
            add_maybe_quantized(f"blk.{l}.attn_norm.weight", state_dict[f"{pfx}.attn_norm.weight"])
            add_maybe_quantized(f"blk.{l}.attn_q.weight", state_dict[f"{pfx}.attn_q.weight"])
            add_maybe_quantized(f"blk.{l}.attn_k.weight", state_dict[f"{pfx}.attn_k.weight"])
            add_maybe_quantized(f"blk.{l}.attn_v.weight", state_dict[f"{pfx}.attn_v.weight"])
            add_maybe_quantized(f"blk.{l}.attn_output.weight", state_dict[f"{pfx}.attn_output.weight"])

            if is_markov_model:
                q_norm = state_dict.get(f"{pfx}.attn_q_norm.weight", torch.ones(hidden_size // num_heads)).float().numpy()
                k_norm = state_dict.get(f"{pfx}.attn_k_norm.weight", torch.ones(hidden_size // num_heads)).float().numpy()
                gguf_writer.add_tensor(f"blk.{l}.attn_q_norm.weight", q_norm)
                gguf_writer.add_tensor(f"blk.{l}.attn_k_norm.weight", k_norm)

            add_maybe_quantized(f"blk.{l}.ffn_norm.weight", state_dict[f"{pfx}.ffn_norm.weight"])
            add_maybe_quantized(f"blk.{l}.ffn_gate.weight", state_dict[f"{pfx}.ffn_gate.weight"])
            add_maybe_quantized(f"blk.{l}.ffn_up.weight", state_dict[f"{pfx}.ffn_up.weight"])
            add_maybe_quantized(f"blk.{l}.ffn_down.weight", state_dict[f"{pfx}.ffn_down.weight"])

    # Output norm
    if "output_norm.weight" in state_dict:
        add_maybe_quantized("output_norm.weight", state_dict["output_norm.weight"])
    elif "norm.weight" in state_dict:
        add_maybe_quantized("output_norm.weight", state_dict["norm.weight"])

    print(f"Writing Quantized GGUF sidecar ({arch_name}) to {output_gguf_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Quantized GGUF sidecar export complete: {output_gguf_path}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_dspark_standalone.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/lfm2.5-vl-450m-dspark-standalone-Q4_0.gguf"
    q_type = sys.argv[3] if len(sys.argv) > 3 else "Q4_0"
    export_quantized_gguf(ckpt, out, quant_type=q_type)


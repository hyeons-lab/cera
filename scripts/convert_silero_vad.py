#!/usr/bin/env python3
"""
Convert Silero VAD v5 ONNX model to GGUF format for Cera.
"""

import argparse
import os
import numpy as np
import onnx
from onnx import numpy_helper
import gguf


def extract_branch_weights(graph):
    weights = {}
    for n in graph.node:
        if n.op_type == "Constant" and n.output:
            for attr in n.attribute:
                if attr.t and attr.t.data_type == 1:  # FLOAT
                    arr = numpy_helper.to_array(attr.t)
                    out_name = n.output[0]
                    name = out_name.split("__Inline_0__")[-1] if "__Inline_0__" in out_name else out_name
                    weights[name] = arr
    return weights


def convert_silero_vad(onnx_path: str, output_path: str):
    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)

    # Locate 16kHz (then_branch) and 8kHz (else_branch) subgraphs
    then_branch = None
    else_branch = None
    for n in model.graph.node:
        if n.op_type == "If":
            for attr in n.attribute:
                if attr.name == "then_branch":
                    then_branch = attr.g
                elif attr.name == "else_branch":
                    else_branch = attr.g
            if then_branch and else_branch:
                break

    if not then_branch or not else_branch:
        raise ValueError("Could not find then_branch and else_branch in ONNX model")

    w16 = extract_branch_weights(then_branch)
    w8 = extract_branch_weights(else_branch)

    print(f"Extracted {len(w16)} tensors from 16kHz branch, {len(w8)} tensors from 8kHz branch")

    writer = gguf.GGUFWriter(output_path, arch="silero_vad")

    # Metadata
    writer.add_architecture()
    writer.add_string("general.name", "Silero VAD v5")
    writer.add_string("general.description", "Voice Activity Detector v5 from Silero")
    writer.add_uint32("silero_vad.hidden_size", 128)
    writer.add_uint32("silero_vad.window_size_16k", 512)
    writer.add_uint32("silero_vad.window_size_8k", 256)
    writer.add_uint32("silero_vad.context_size_16k", 64)
    writer.add_uint32("silero_vad.context_size_8k", 32)
    writer.add_uint32("silero_vad.hop_size_16k", 128)
    writer.add_uint32("silero_vad.hop_size_8k", 64)
    writer.add_uint32("silero_vad.stft_win_16k", 256)
    writer.add_uint32("silero_vad.stft_win_8k", 128)

    # Tensor mapping for 16kHz & 8kHz
    tensors_to_write = {
        # 16kHz network
        "stft.16k.basis": w16["stft.forward_basis_buffer"],
        "encoder.16k.0.weight": w16["encoder.0.reparam_conv.weight"],
        "encoder.16k.0.bias": w16["encoder.0.reparam_conv.bias"],
        "encoder.16k.1.weight": w16["encoder.1.reparam_conv.weight"],
        "encoder.16k.1.bias": w16["encoder.1.reparam_conv.bias"],
        "encoder.16k.2.weight": w16["encoder.2.reparam_conv.weight"],
        "encoder.16k.2.bias": w16["encoder.2.reparam_conv.bias"],
        "encoder.16k.3.weight": w16["encoder.3.reparam_conv.weight"],
        "encoder.16k.3.bias": w16["encoder.3.reparam_conv.bias"],
        "decoder.16k.rnn.weight_ih": w16["decoder.rnn.weight_ih"],
        "decoder.16k.rnn.weight_hh": w16["decoder.rnn.weight_hh"],
        "decoder.16k.rnn.bias_ih": w16["decoder.rnn.bias_ih"],
        "decoder.16k.rnn.bias_hh": w16["decoder.rnn.bias_hh"],
        "decoder.16k.head.weight": w16["decoder.decoder.2.weight"].reshape(128),
        "decoder.16k.head.bias": w16["decoder.decoder.2.bias"].reshape(1),

        # 8kHz network
        "stft.8k.basis": w8["stft.forward_basis_buffer"],
        "encoder.8k.0.weight": w8["encoder.0.reparam_conv.weight"],
        "encoder.8k.0.bias": w8["encoder.0.reparam_conv.bias"],
        "encoder.8k.1.weight": w8["encoder.1.reparam_conv.weight"],
        "encoder.8k.1.bias": w8["encoder.1.reparam_conv.bias"],
        "encoder.8k.2.weight": w8["encoder.2.reparam_conv.weight"],
        "encoder.8k.2.bias": w8["encoder.2.reparam_conv.bias"],
        "encoder.8k.3.weight": w8["encoder.3.reparam_conv.weight"],
        "encoder.8k.3.bias": w8["encoder.3.reparam_conv.bias"],
        "decoder.8k.rnn.weight_ih": w8["decoder.rnn.weight_ih"],
        "decoder.8k.rnn.weight_hh": w8["decoder.rnn.weight_hh"],
        "decoder.8k.rnn.bias_ih": w8["decoder.rnn.bias_ih"],
        "decoder.8k.rnn.bias_hh": w8["decoder.rnn.bias_hh"],
        "decoder.8k.head.weight": w8["decoder.decoder.2.weight"].reshape(128),
        "decoder.8k.head.bias": w8["decoder.decoder.2.bias"].reshape(1),
    }

    for name, arr in tensors_to_write.items():
        arr_f32 = arr.astype(np.float32)
        print(f"Adding tensor: {name:30s} shape={str(arr_f32.shape):18s} dtype={arr_f32.dtype}")
        writer.add_tensor(name, arr_f32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nSuccessfully wrote {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Convert Silero VAD ONNX to GGUF")
    parser.add_argument("--onnx", default="models/silero_vad.onnx", help="Path to input silero_vad.onnx")
    parser.add_argument("--output", default="models/silero_vad.gguf", help="Path to output silero_vad.gguf")
    args = parser.parse_args()

    convert_silero_vad(args.onnx, args.output)


if __name__ == "__main__":
    main()

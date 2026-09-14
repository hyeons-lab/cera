"""Deterministic one-block F32 GGUF for native boundary probes; no model download."""

import struct


def tiny_model():
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = []
    for key, value in (
        ("general.architecture", "llama"),
        ("general.name", "native-bridge-probe"),
    ):
        metadata.append(string(key) + struct.pack("<I", 8) + string(value))
    metadata.append(
        string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, 2)
        + string("a")
        + string("b")
    )
    for key, value in (
        ("block_count", 1),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("context_length", 64),
        ("vocab_size", 2),
    ):
        metadata.append(string("llama." + key) + struct.pack("<II", 4, value))
    tensors = [
        ("token_embd.weight", (32, 2)),
        ("output_norm.weight", (32,)),
        ("blk.0.attn_norm.weight", (32,)),
        ("blk.0.ffn_norm.weight", (32,)),
    ]
    tensors += [
        ("blk.0." + name + ".weight", (32, 32))
        for name in (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        )
    ]
    descriptors, payloads = [], []
    offset = 0
    for name, dimensions in tensors:
        count = 1
        for dimension in dimensions:
            count *= dimension
        values = [
            1.0 if "norm" in name else ((i % 17) - 8) / 32.0 for i in range(count)
        ]
        payload = struct.pack(f"<{count}f", *values)
        descriptors.append(
            string(name)
            + struct.pack("<I", len(dimensions))
            + struct.pack(f"<{len(dimensions)}Q", *dimensions)
            + struct.pack("<IQ", 0, offset)
        )
        payloads.append(payload)
        offset += len(payload)
        assert offset % 32 == 0
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(metadata))
        + b"".join(metadata + descriptors)
    )
    return header + bytes((-len(header)) % 32) + b"".join(payloads)

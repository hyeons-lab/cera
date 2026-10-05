#!/usr/bin/env python3
"""Convert NVIDIA Nemotron-3-Diarization to a Cera GGUF.

    python scripts/nemotron3_diarization/convert.py \\
        --safetensors ~/.leap/models/nemotron3-diarization/model.safetensors \\
        --nemo ~/.leap/models/nemotron3-diarization/Nemotron-3-Diarization.nemo \\
        --out ~/.leap/models/nemotron3-diarization/nemotron3-diarization-q8_0.gguf \\
        --outtype q8_0

Needs torch, safetensors, numpy, pyyaml and gguf. It does NOT need NeMo: weights come
from `model.safetensors` (+ `config.json` beside it), and the `.nemo` (a tar of
`model_config.yaml` + `model_weights.ckpt`) is read only for the mel window/filterbank
and the streaming defaults, which `config.json` does not fully carry.

Layout
------
Unlike the 4spk Sortformer (whose FastConformer reuses the LFM2-Audio mmproj names),
nothing here is shared with another model, so everything lives under `nd.*` /
`nemotron3.*`:

* `nd.embed.proj.weight` [512, 1024]: the feature-stacking projection (8 stacked
  128-mel frames, no bias).
* `nd.input_norm`, `nd.final_norm`, `nd.blk.N.*`: 31 pre-LN RoPE Transformer layers
  (`ln1`, `attn_q/k/v` without bias, `attn_o` with bias, `ln2`, `mlp_up/down` with
  bias, erf GELU between the MLP matrices).
* `nd.proj` [192, 512], `nd.upsample` ([1536, 576], the Conv1d(192 -> 1536, k=3)
  weight reshaped row-major from [1536, 192, 3]) + bias, `nd.classifier.dense/out`.
  The Rust port lowers the subpixel convolution to a matmul: unrolled[t, c*3+d] holds
  mel-frame (t + d - 1) of stacked channel c (zero outside), matching the reshape.
* `nd.silence_embeds` [512]: the learned silence embedding filling reserved cache slots.
* `nd.mel.window` / `nd.mel.fb`: the checkpoint's own window and filterbank.

Quantization (`--outtype`): f32 | f16 | q8_0 | q4_0 applies to the embedder projection
and the encoder matrices (attention q/k/v/o, both MLP matrices per layer). Norms,
biases, silence embeds and mel tables stay F32. The tail (`nd.proj`, `nd.upsample`,
`nd.classifier.*`) stays F32 (f32) or F16 (otherwise), unless `--tail-outtype q8_0`,
which the Hexagon NPU build reads.
"""

import argparse
import hashlib
import io
import json
import os
import tarfile

import gguf
import numpy as np
import torch
import yaml
from safetensors import safe_open

F32 = gguf.GGMLQuantizationType.F32
F16 = gguf.GGMLQuantizationType.F16
Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q4_0 = gguf.GGMLQuantizationType.Q4_0
OUTTYPES = {"f32": F32, "f16": F16, "q8_0": Q8_0, "q4_0": Q4_0}

# Encoder matrices that follow --outtype.
QUANT_SUFFIXES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_o.weight",
    "mlp_up.weight",
    "mlp_down.weight",
)


def require(cond, msg):
    """Refuse an unsupported checkpoint with a readable message (an `assert` vanishes under -O)."""
    if not cond:
        raise SystemExit(f"unsupported checkpoint: {msg}")


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_nemo(path):
    """Return (config dict, state dict) from a .nemo archive (mel tables + streaming defaults)."""
    cfg = ckpt = None
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = member.name.split("/")[-1]
            if base == "model_config.yaml":
                cfg = yaml.safe_load(tar.extractfile(member).read())
            elif base == "model_weights.ckpt":
                ckpt = torch.load(
                    io.BytesIO(tar.extractfile(member).read()),
                    map_location="cpu",
                    weights_only=True,
                )
    if cfg is None or ckpt is None:
        raise SystemExit(f"{path}: missing model_config.yaml or model_weights.ckpt")
    return cfg, ckpt


def f32(t):
    return t.detach().to(torch.float32).numpy()


def add_tensor(writer, name, arr, outtype, tail_outtype=None):
    """Write one tensor, choosing its storage type from its name and shape."""
    arr = np.ascontiguousarray(arr)
    if name == "nd.embed.proj.weight" or (
        name.startswith("nd.blk.") and name.endswith(QUANT_SUFFIXES)
    ):
        target = outtype
    elif (
        name.startswith(("nd.proj.", "nd.upsample.", "nd.classifier."))
        and name.endswith(".weight")
        and arr.ndim == 2
    ):
        target = tail_outtype or (F32 if outtype == F32 else F16)
    else:
        target = F32
    if target == F32:
        writer.add_tensor(name, arr.astype(np.float32))
    elif target == F16:
        writer.add_tensor(name, arr.astype(np.float16), raw_dtype=F16)
    else:
        if arr.shape[-1] % 32:
            raise SystemExit(f"{name}: row length {arr.shape[-1]} is not a multiple of 32")
        q = gguf.quants.quantize(arr.astype(np.float32), target)
        writer.add_tensor(name, q, raw_dtype=target)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", choices=sorted(OUTTYPES), default="q8_0")
    ap.add_argument("--tail-outtype", choices=["f16", "q8_0"], default=None,
                    help="storage of the tail matrices (nd.proj, nd.upsample, nd.classifier.*) "
                         "(default: F16, or F32 with --outtype f32); q8_0 is what the Hexagon NPU reads")
    args = ap.parse_args()
    outtype = OUTTYPES[args.outtype]
    tail_outtype = OUTTYPES[args.tail_outtype] if args.tail_outtype else None
    if tail_outtype is not None and outtype == F32:
        raise SystemExit("--tail-outtype needs a quantized --outtype, not f32")

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.safetensors)), "config.json")
    with open(cfg_path) as f:
        hf = json.load(f)
    audio, head, stream = hf["audio_config"], hf["head_config"], hf["streaming_config"]

    ncfg, _ = read_nemo(args.nemo)
    enc, pre = ncfg["encoder"], ncfg["preprocessor"]
    sm = ncfg["sortformer_modules"]
    require(ncfg["target"].endswith("SortformerEncLabelModel"), f"target {ncfg['target']!r} is not SortformerEncLabelModel")
    require(enc["self_attention_model"] == "rope", f"encoder attention {enc['self_attention_model']}, not rope")
    require(enc["subsampling"] == "feature_stacking" and enc["subsampling_factor"] == 8,
            f"subsampling {enc['subsampling']} x{enc['subsampling_factor']}, not feature_stacking x8")
    require(enc["xscaling"] is False, f"encoder xscaling {enc['xscaling']}, not false")
    require(enc["qkv_bias"] is False, f"encoder qkv_bias {enc['qkv_bias']}, not false")
    require(enc["pre_block_norm"] is True, "encoder is not pre-LN")
    require(pre["normalize"] == "NA", f"mel normalize {pre['normalize']!r}, not NA")
    require(ncfg["max_num_of_spks"] == 8, f"max_num_of_spks {ncfg['max_num_of_spks']}, not 8")
    require(ncfg["high_resolution"] is True and ncfg["output_subsampling_factor"] == 1,
            "not the high-resolution (10 ms output) checkpoint")
    require(sm["use_learnable_sil_emb"] is True, "checkpoint has no learnable silence embedding")
    require(sm["chunk_left_context"] == 0, f"chunk_left_context {sm['chunk_left_context']}, not 0")

    n_layer, d_model = audio["num_hidden_layers"], audio["hidden_size"]
    n_heads, n_mel = audio["num_attention_heads"], audio["num_mel_bins"]
    n_ff, tf_d, n_spk = audio["intermediate_size"], head["hidden_size"], head["num_speakers"]
    # The Rust loader refuses all of these; fail here instead of at load.
    require((n_layer, d_model, n_heads, n_ff) == (31, 512, 8, 2048),
            f"encoder {n_layer}x{d_model}x{n_heads}x{n_ff}, not 31x512x8x2048")
    require(audio["hidden_act"] == "gelu", f"hidden_act {audio['hidden_act']}, not gelu")
    require(n_mel == 128 and audio["subsampling_factor"] == 8, "not 128 mel bins subsampled x8")
    require((head["audio_hidden_size"], tf_d, n_spk) == (512, 192, 8),
            f"head {head['audio_hidden_size']}->{tf_d}->{n_spk}, not 512->192->8")
    require(audio["rope_parameters"]["rope_theta"] == 10000.0, "rope theta is not 10000")
    require(audio["rope_parameters"].get("partial_rotary_factor", 1.0) == 1.0,
            "partial rotary factor is not 1.0")
    require(audio["num_key_value_heads"] == n_heads, "not plain multi-head attention")
    require(enc["d_model"] == d_model and enc["n_heads"] == n_heads and enc["n_layers"] == n_layer,
            ".nemo encoder dimensions disagree with config.json")
    require(sm["fc_d_model"] == d_model and sm["tf_d_model"] == tf_d,
            ".nemo head dimensions disagree with config.json")
    require(sm["num_spks"] == n_spk, ".nemo speaker count disagrees with config.json")
    # Only the score policy is checkpoint-intrinsic. config.json's chunking knobs are the
    # low-latency runtime preset (fifo 264, update 222), not the checkpoint's own (0, 264).
    pairs = [
        ("speaker_cache_length", "spkcache_len"),
        ("speaker_cache_silence_frames_per_speaker", "spkcache_sil_frames_per_spk"),
        ("prediction_score_threshold", "pred_score_threshold"),
        ("latest_frames_score_boost", "scores_boost_latest"),
        ("min_positive_scores_rate", "min_pos_scores_rate"),
        ("strong_boost_rate", "strong_boost_rate"),
        ("weak_boost_rate", "weak_boost_rate"),
    ]
    for hk, nk in pairs:
        require(stream[hk] == sm[nk], f"config.json {hk}={stream[hk]} disagrees with .nemo {nk}={sm[nk]}")
    require(stream["num_speakers"] == n_spk and stream["subsampling_factor"] == 8,
            "config.json streaming speaker count / subsampling disagree")
    require(pre["n_fft"] == 512 and round(pre["window_size"] * pre["sample_rate"]) == 400
            and round(pre["window_stride"] * pre["sample_rate"]) == 160 and pre["sample_rate"] == 16000,
            "mel front end is not 16 kHz, n_fft 512, 400-sample window, 160-sample hop")
    require(1 <= n_layer <= 256, f"layer count {n_layer} outside 1..256")
    require(sm["spkcache_len"] // 8 > sm["spkcache_sil_frames_per_spk"],
            "spkcache_len leaves no room for speakers beside their silence frames")
    flat = 8 * (sm["spkcache_len"] + sm["fifo_len"] + sm["chunk_len"] + sm["spkcache_sil_frames_per_spk"])
    require(sm["max_index"] >= flat, f"max_index {sm['max_index']} lies inside the cache's flat index range (needs >= {flat})")
    require(0 < sm["pred_score_threshold"] <= 1, f"pred_score_threshold {sm['pred_score_threshold']} outside (0, 1]")
    require(sm["chunk_len"] > 0 and sm["spkcache_update_period"] > 0, "chunk_len and spkcache_update_period must be > 0")
    stream_keys = ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_len",
                   "spkcache_update_period", "spkcache_sil_frames_per_spk")
    require(all(sm[k] <= 1 << 20 for k in stream_keys), "a streaming length exceeds 2^20")
    window = sum(sm[k] for k in ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_len"))
    require(window <= 7500, f"chunk + right + fifo + spkcache = {window} frames exceeds the loader's 7500")

    st = safe_open(args.safetensors, framework="pt", device="cpu")
    names = set(st.keys())
    need = {"model.audio_tower.embedder.projection.weight", "model.audio_tower.input_layer_norm.weight",
            "model.audio_tower.input_layer_norm.bias", "model.audio_tower.layer_norm.weight",
            "model.audio_tower.layer_norm.bias", "model.proj.weight", "model.proj.bias",
            "model.upsampler.conv.weight", "model.upsampler.conv.bias", "classifier.dense.weight",
            "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias",
            "silence_embeds"}
    for n in range(n_layer):
        for t in ("layer_norm1", "layer_norm2"):
            need.add(f"model.audio_tower.layers.{n}.{t}.weight")
            need.add(f"model.audio_tower.layers.{n}.{t}.bias")
        for t in ("q_proj", "k_proj", "v_proj"):
            need.add(f"model.audio_tower.layers.{n}.self_attn.{t}.weight")
        need.add(f"model.audio_tower.layers.{n}.self_attn.o_proj.weight")
        need.add(f"model.audio_tower.layers.{n}.self_attn.o_proj.bias")
        for t in ("fc1", "fc2"):
            need.add(f"model.audio_tower.layers.{n}.mlp.{t}.weight")
            need.add(f"model.audio_tower.layers.{n}.mlp.{t}.bias")
    # q/k/v must have no bias (the Rust forward adds none); anything else unexpected fails too.
    unexpected = names - need
    require(not unexpected, f"unexpected tensors in {args.safetensors}: {sorted(unexpected)[:8]}")
    require(not (need - names), f"missing tensors in {args.safetensors}: {sorted(need - names)[:8]}")

    def w(key):
        return st.get_slice(key)[:].float().numpy()

    tensors = [("nd.embed.proj.weight", w("model.audio_tower.embedder.projection.weight"))]
    tensors.append(("nd.input_norm.weight", w("model.audio_tower.input_layer_norm.weight")))
    tensors.append(("nd.input_norm.bias", w("model.audio_tower.input_layer_norm.bias")))
    tensors.append(("nd.final_norm.weight", w("model.audio_tower.layer_norm.weight")))
    tensors.append(("nd.final_norm.bias", w("model.audio_tower.layer_norm.bias")))
    for n in range(n_layer):
        p, s = f"model.audio_tower.layers.{n}", f"nd.blk.{n}"
        tensors.append((f"{s}.ln1.weight", w(f"{p}.layer_norm1.weight")))
        tensors.append((f"{s}.ln1.bias", w(f"{p}.layer_norm1.bias")))
        tensors.append((f"{s}.attn_q.weight", w(f"{p}.self_attn.q_proj.weight")))
        tensors.append((f"{s}.attn_k.weight", w(f"{p}.self_attn.k_proj.weight")))
        tensors.append((f"{s}.attn_v.weight", w(f"{p}.self_attn.v_proj.weight")))
        tensors.append((f"{s}.attn_o.weight", w(f"{p}.self_attn.o_proj.weight")))
        tensors.append((f"{s}.attn_o.bias", w(f"{p}.self_attn.o_proj.bias")))
        tensors.append((f"{s}.ln2.weight", w(f"{p}.layer_norm2.weight")))
        tensors.append((f"{s}.ln2.bias", w(f"{p}.layer_norm2.bias")))
        tensors.append((f"{s}.mlp_up.weight", w(f"{p}.mlp.fc1.weight")))
        tensors.append((f"{s}.mlp_up.bias", w(f"{p}.mlp.fc1.bias")))
        tensors.append((f"{s}.mlp_down.weight", w(f"{p}.mlp.fc2.weight")))
        tensors.append((f"{s}.mlp_down.bias", w(f"{p}.mlp.fc2.bias")))
    tensors.append(("nd.proj.weight", w("model.proj.weight")))
    tensors.append(("nd.proj.bias", w("model.proj.bias")))
    up = w("model.upsampler.conv.weight")
    require(up.shape[2] == 3, f"upsampler kernel {tuple(up.shape)} is not k=3")
    tensors.append(("nd.upsample.weight", np.ascontiguousarray(up.reshape(up.shape[0], -1))))
    tensors.append(("nd.upsample.bias", w("model.upsampler.conv.bias")))
    tensors.append(("nd.classifier.dense.weight", w("classifier.dense.weight")))
    tensors.append(("nd.classifier.dense.bias", w("classifier.dense.bias")))
    tensors.append(("nd.classifier.out.weight", w("classifier.out_proj.weight")))
    tensors.append(("nd.classifier.out.bias", w("classifier.out_proj.bias")))
    tensors.append(("nd.silence_embeds", w("silence_embeds")))
    _, nckpt = read_nemo(args.nemo)
    window_t = f32(nckpt["preprocessor.featurizer.window"])
    fb = f32(nckpt["preprocessor.featurizer.fb"])
    require(fb.shape[0] == 1, f"mel filterbank {tuple(fb.shape)} has a batch axis")
    tensors.append(("nd.mel.window", window_t))
    tensors.append(("nd.mel.fb", fb[0]))

    writer = gguf.GGUFWriter(args.out, arch="nemotron3_diarization")
    writer.add_string("general.name", "Nemotron-3 Diarization")
    writer.add_string("general.description", "NVIDIA Nemotron-3 speaker diarizer, converted from safetensors")
    writer.add_string("general.source.sha256", sha256_file(args.safetensors))
    writer.add_string("general.source.nemo.sha256", sha256_file(args.nemo))

    writer.add_uint32("nemotron3.block_count", n_layer)
    writer.add_uint32("nemotron3.embedding_length", d_model)
    writer.add_uint32("nemotron3.feed_forward_length", n_ff)
    writer.add_uint32("nemotron3.attention.head_count", n_heads)
    writer.add_float32("nemotron3.attention.layer_norm_epsilon", 1e-5)
    writer.add_float32("nemotron3.rope.theta", 10000.0)
    writer.add_uint32("nemotron3.rope.max_pos", enc["pos_emb_max_len"])
    writer.add_uint32("nemotron3.num_mel_bins", n_mel)
    writer.add_uint32("nemotron3.subsampling_factor", 8)
    writer.add_uint32("nemotron3.head.hidden_size", tf_d)
    writer.add_uint32("nemotron3.max_speakers", n_spk)
    writer.add_uint32("nemotron3.sample_rate", 16000)

    writer.add_uint32("nemotron3.mel.n_fft", pre["n_fft"])
    writer.add_uint32("nemotron3.mel.win_length", round(pre["window_size"] * pre["sample_rate"]))
    writer.add_uint32("nemotron3.mel.hop_length", round(pre["window_stride"] * pre["sample_rate"]))
    writer.add_uint32("nemotron3.mel.n_mels", pre["features"])
    writer.add_string("nemotron3.mel.normalize", pre["normalize"])
    writer.add_float32("nemotron3.mel.preemph", 0.97)
    writer.add_float32("nemotron3.mel.log_zero_guard", 2.0**-24)
    writer.add_float32("nemotron3.mel.mag_power", 2.0)
    writer.add_uint32("nemotron3.mel.pad_to", 16)

    for key in (
        "chunk_len",
        "chunk_right_context",
        "fifo_len",
        "spkcache_len",
        "spkcache_update_period",
        "spkcache_sil_frames_per_spk",
        "max_index",
    ):
        writer.add_uint32(f"nemotron3.stream.{key}", sm[key])
    for key in (
        "pred_score_threshold",
        "scores_boost_latest",
        "sil_threshold",
        "strong_boost_rate",
        "weak_boost_rate",
        "min_pos_scores_rate",
    ):
        writer.add_float32(f"nemotron3.stream.{key}", float(sm[key]))

    counts = {}
    for name, arr in tensors:
        t = add_tensor(writer, name, arr, outtype, tail_outtype)
        counts[t.name] = counts.get(t.name, 0) + 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}: tensors by type {counts}")


if __name__ == "__main__":
    main()

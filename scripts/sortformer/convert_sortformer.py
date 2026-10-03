#!/usr/bin/env python3
"""Convert NVIDIA Streaming Sortformer (`.nemo`) to a Cera GGUF.

    python scripts/sortformer/convert_sortformer.py \
        --nemo ~/.leap/models/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo \
        --out  ~/.leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf \
        --outtype q8_0

Needs torch, numpy, pyyaml and gguf. It does NOT need NeMo: the `.nemo` is a tar of
`model_config.yaml` + `model_weights.ckpt` (a plain torch state dict).

Layout
------
The FastConformer encoder reuses the tensor names and shapes of the LFM2-Audio mmproj
GGUF (`a.conv1d.*`, `a.pre_encode.*`, `a.blk.N.*`, `clip.audio.*` metadata), because it is the
same NeMo FastConformer: 17 blocks, d_model 512, 8 heads, FFN 2048, conv kernel 9, 128 mel,
dw_striding x8 stem with 256 channels. Everything Sortformer-specific lives under `sf.*` /
`sortformer.*`.

Transforms applied (everything else is copied verbatim, transposes included):

* `conv.batch_norm` is folded into the per-channel affine `conv_norm`
  (`scale = w / sqrt(var + eps)`, `shift = b - mean * scale`, eps 1e-5, done in f64). Cera's conv
  module applies `silu(dw * conv_norm.weight + conv_norm.bias)` after the depthwise bias, which is
  exactly BatchNorm in eval mode.
* `pointwise_conv{1,2}.weight` `[out, in, 1]` and `depthwise_conv.weight` `[ch, 1, k]` lose their
  singleton axis.
* The stem conv biases are written `[C, 1, 1]` (GGUF ne `[1, 1, C]`) like the mmproj.
* `hidden_to_spks` ([4, 384]) is not exported: NeMo's inference path
  (`forward_speaker_sigmoids`) is relu -> first_hidden_to_hidden -> relu -> single_hidden_to_spks ->
  sigmoid, and never touches it.
* The mel window and filterbank are exported from the checkpoint (`sf.mel.window`,
  `sf.mel.fb`) so the front end does not have to re-derive them.

Quantization (`--outtype`): f32 | f16 | q8_0 | q4_0 applies to the FastConformer matrices
(attention q/k/v/out, linear_pos, both FFNs, pointwise convs, pre_encode.out). Norms, biases,
depthwise taps, stem convs, positional biases, the mel tables, `encoder_proj`, the Transformer head
and the speaker head stay F32 (f32) or F16 (otherwise, matrices only), never Q8/Q4: they are small
(~8 M params) and sit on the path to the sigmoids, which the DER depends on.
"""

import argparse
import hashlib
import io
import tarfile

import gguf
import numpy as np
import torch
import yaml

F32 = gguf.GGMLQuantizationType.F32
F16 = gguf.GGMLQuantizationType.F16
Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q4_0 = gguf.GGMLQuantizationType.Q4_0
OUTTYPES = {"f32": F32, "f16": F16, "q8_0": Q8_0, "q4_0": Q4_0}

BN_EPS = 1e-5  # nn.BatchNorm1d default; NeMo's ConformerConvolution does not override it.

# Encoder matrices that follow --outtype.
QUANT_SUFFIXES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "linear_pos.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_up_1.weight",
    "ffn_down_1.weight",
    "conv_pw1.weight",
    "conv_pw2.weight",
)


def read_nemo(path):
    """Return (config dict, state dict, sha256 of the .nemo) from a .nemo archive."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    cfg = ckpt = None
    with tarfile.open(path) as tar:
        for member in tar:
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
    return cfg, ckpt, sha.hexdigest()


def f32(t):
    return t.detach().to(torch.float32).numpy()


def fold_batch_norm(sd, pfx):
    """Fold eval-mode BatchNorm1d into (scale, shift), computed in f64."""
    w = sd[f"{pfx}.weight"].double()
    b = sd[f"{pfx}.bias"].double()
    mean = sd[f"{pfx}.running_mean"].double()
    var = sd[f"{pfx}.running_var"].double()
    scale = w / torch.sqrt(var + BN_EPS)
    shift = b - mean * scale
    return scale.float().numpy(), shift.float().numpy()


def encoder_tensors(sd, n_layer):
    """Yield (gguf name, numpy array) for the FastConformer encoder."""
    for i in (0, 2, 3, 5, 6):
        yield f"a.conv1d.{i}.weight", f32(sd[f"encoder.pre_encode.conv.{i}.weight"])
        yield (
            f"a.conv1d.{i}.bias",
            f32(sd[f"encoder.pre_encode.conv.{i}.bias"]).reshape(-1, 1, 1),
        )
    yield "a.pre_encode.out.weight", f32(sd["encoder.pre_encode.out.weight"])
    yield "a.pre_encode.out.bias", f32(sd["encoder.pre_encode.out.bias"])

    for n in range(n_layer):
        e = f"encoder.layers.{n}"
        a = f"a.blk.{n}"
        for ours, theirs in (("ffn_norm", "norm_feed_forward1"), ("ffn_norm_1", "norm_feed_forward2")):
            yield f"{a}.{ours}.weight", f32(sd[f"{e}.{theirs}.weight"])
            yield f"{a}.{ours}.bias", f32(sd[f"{e}.{theirs}.bias"])
        for ours, theirs in (
            ("ffn_up", "feed_forward1.linear1"),
            ("ffn_down", "feed_forward1.linear2"),
            ("ffn_up_1", "feed_forward2.linear1"),
            ("ffn_down_1", "feed_forward2.linear2"),
        ):
            yield f"{a}.{ours}.weight", f32(sd[f"{e}.{theirs}.weight"])
            yield f"{a}.{ours}.bias", f32(sd[f"{e}.{theirs}.bias"])
        for ours, theirs in (("ln1", "norm_self_att"), ("ln2", "norm_out"), ("norm_conv", "norm_conv")):
            yield f"{a}.{ours}.weight", f32(sd[f"{e}.{theirs}.weight"])
            yield f"{a}.{ours}.bias", f32(sd[f"{e}.{theirs}.bias"])
        for ours, theirs in (
            ("attn_q", "linear_q"),
            ("attn_k", "linear_k"),
            ("attn_v", "linear_v"),
            ("attn_out", "linear_out"),
        ):
            yield f"{a}.{ours}.weight", f32(sd[f"{e}.self_attn.{theirs}.weight"])
            yield f"{a}.{ours}.bias", f32(sd[f"{e}.self_attn.{theirs}.bias"])
        yield f"{a}.linear_pos.weight", f32(sd[f"{e}.self_attn.linear_pos.weight"])
        yield f"{a}.pos_bias_u", f32(sd[f"{e}.self_attn.pos_bias_u"])
        yield f"{a}.pos_bias_v", f32(sd[f"{e}.self_attn.pos_bias_v"])
        for ours, theirs in (("conv_pw1", "pointwise_conv1"), ("conv_pw2", "pointwise_conv2")):
            w = f32(sd[f"{e}.conv.{theirs}.weight"])
            assert w.shape[-1] == 1, (theirs, w.shape)
            yield f"{a}.{ours}.weight", w[..., 0]
            yield f"{a}.{ours}.bias", f32(sd[f"{e}.conv.{theirs}.bias"])
        dw = f32(sd[f"{e}.conv.depthwise_conv.weight"])
        assert dw.shape[1] == 1, dw.shape
        yield f"{a}.conv_dw.weight", dw[:, 0, :]
        yield f"{a}.conv_dw.bias", f32(sd[f"{e}.conv.depthwise_conv.bias"])
        scale, shift = fold_batch_norm(sd, f"{e}.conv.batch_norm")
        yield f"{a}.conv_norm.weight", scale
        yield f"{a}.conv_norm.bias", shift


def head_tensors(sd, n_layer):
    """Yield (gguf name, numpy array) for encoder_proj, the Transformer and the speaker head."""
    m = "sortformer_modules"
    yield "sf.enc_proj.weight", f32(sd[f"{m}.encoder_proj.weight"])
    yield "sf.enc_proj.bias", f32(sd[f"{m}.encoder_proj.bias"])
    for n in range(n_layer):
        t = f"transformer_encoder.layers.{n}"
        s = f"sf.blk.{n}"
        for ours, theirs in (
            ("ln1", "layer_norm_1"),
            ("ln2", "layer_norm_2"),
            ("attn_q", "first_sub_layer.query_net"),
            ("attn_k", "first_sub_layer.key_net"),
            ("attn_v", "first_sub_layer.value_net"),
            ("attn_out", "first_sub_layer.out_projection"),
            ("ffn_up", "second_sub_layer.dense_in"),
            ("ffn_down", "second_sub_layer.dense_out"),
        ):
            yield f"{s}.{ours}.weight", f32(sd[f"{t}.{theirs}.weight"])
            yield f"{s}.{ours}.bias", f32(sd[f"{t}.{theirs}.bias"])
    yield "sf.head.hidden.weight", f32(sd[f"{m}.first_hidden_to_hidden.weight"])
    yield "sf.head.hidden.bias", f32(sd[f"{m}.first_hidden_to_hidden.bias"])
    yield "sf.head.out.weight", f32(sd[f"{m}.single_hidden_to_spks.weight"])
    yield "sf.head.out.bias", f32(sd[f"{m}.single_hidden_to_spks.bias"])


def mel_tensors(sd):
    window = f32(sd["preprocessor.featurizer.window"])
    fb = f32(sd["preprocessor.featurizer.fb"])
    assert fb.shape[0] == 1, fb.shape
    yield "sf.mel.window", window
    yield "sf.mel.fb", fb[0]


def add_tensor(writer, name, arr, outtype):
    """Write one tensor, choosing its storage type from its name and shape."""
    arr = np.ascontiguousarray(arr)
    is_matrix = arr.ndim == 2 and not name.startswith("sf.mel.")
    if name.startswith("a.") and name.endswith(QUANT_SUFFIXES) or name == "a.pre_encode.out.weight":
        target = outtype
    elif name.startswith("sf.") and is_matrix and name.endswith(".weight"):
        target = F32 if outtype == F32 else F16
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
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", choices=sorted(OUTTYPES), default="q8_0")
    args = ap.parse_args()
    outtype = OUTTYPES[args.outtype]

    cfg, sd, sha = read_nemo(args.nemo)
    enc, tf, sm, pre = cfg["encoder"], cfg["transformer_encoder"], cfg["sortformer_modules"], cfg["preprocessor"]
    assert cfg["target"].endswith("SortformerEncLabelModel"), cfg["target"]
    assert enc["self_attention_model"] == "rel_pos" and enc["conv_norm_type"] == "batch_norm"
    assert enc["subsampling"] == "dw_striding" and enc["subsampling_factor"] == 8
    assert pre["normalize"] == "NA", pre["normalize"]
    assert tf["hidden_act"] == "relu" and not tf["pre_ln"], (tf["hidden_act"], tf["pre_ln"])
    assert enc["att_context_size"] == [-1, -1], enc["att_context_size"]
    assert enc["xscaling"] is True

    n_layer, d_model = enc["n_layers"], enc["d_model"]
    n_ff = d_model * enc["ff_expansion_factor"]
    tf_layers, tf_d = tf["num_layers"], tf["hidden_size"]

    writer = gguf.GGUFWriter(args.out, arch="sortformer")
    writer.add_string("general.name", "Streaming Sortformer 4spk v2.1")
    writer.add_string("general.description", "NVIDIA Streaming Sortformer diarizer, converted from .nemo")
    writer.add_string("general.source.sha256", sha)

    # Encoder metadata under the keys cera's AudioEncoderWeights::from_gguf already reads.
    writer.add_bool("clip.has_audio_encoder", True)
    writer.add_uint32("clip.audio.block_count", n_layer)
    writer.add_uint32("clip.audio.embedding_length", d_model)
    writer.add_uint32("clip.audio.feed_forward_length", n_ff)  # the true value, unlike the mmproj's 512
    writer.add_uint32("clip.audio.attention.head_count", enc["n_heads"])
    writer.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-5)
    writer.add_uint32("clip.audio.num_mel_bins", enc["feat_in"])

    writer.add_uint32("sortformer.max_speakers", cfg["max_num_of_spks"])
    writer.add_uint32("sortformer.fc_d_model", sm["fc_d_model"])
    writer.add_uint32("sortformer.tf_d_model", tf_d)
    writer.add_uint32("sortformer.tf_layer_count", tf_layers)
    writer.add_uint32("sortformer.tf_head_count", tf["num_attention_heads"])
    writer.add_uint32("sortformer.tf_inner_size", tf["inner_size"])
    writer.add_string("sortformer.tf_activation", tf["hidden_act"])
    writer.add_float32("sortformer.tf_layer_norm_epsilon", 1e-5)
    writer.add_uint32("sortformer.subsampling_factor", enc["subsampling_factor"])
    writer.add_uint32("sortformer.conv_kernel_size", enc["conv_kernel_size"])
    writer.add_bool("sortformer.xscaling", enc["xscaling"])
    writer.add_uint32("sortformer.sample_rate", cfg["sample_rate"])

    # Front end. Values not in model_config.yaml are NeMo's AudioToMelSpectrogramPreprocessor
    # defaults, read back from the loaded model's featurizer by gen_golden.py (which asserts them).
    writer.add_uint32("sortformer.mel.n_fft", pre["n_fft"])
    writer.add_uint32("sortformer.mel.win_length", round(pre["window_size"] * pre["sample_rate"]))
    writer.add_uint32("sortformer.mel.hop_length", round(pre["window_stride"] * pre["sample_rate"]))
    writer.add_uint32("sortformer.mel.n_mels", pre["features"])
    writer.add_string("sortformer.mel.normalize", pre["normalize"])
    writer.add_float32("sortformer.mel.preemph", 0.97)
    writer.add_float32("sortformer.mel.log_zero_guard", 2.0**-24)
    writer.add_float32("sortformer.mel.mag_power", 2.0)
    writer.add_uint32("sortformer.mel.pad_to", 16)

    # Streaming defaults from the checkpoint (all in 80 ms encoder frames). The model card's
    # presets are runtime overrides of these.
    for key in (
        "chunk_len",
        "chunk_left_context",
        "chunk_right_context",
        "fifo_len",
        "spkcache_len",
        "spkcache_update_period",
        "spkcache_sil_frames_per_spk",
        "max_index",
    ):
        writer.add_uint32(f"sortformer.stream.{key}", sm[key])
    for key in (
        "pred_score_threshold",
        "scores_boost_latest",
        "sil_threshold",
        "strong_boost_rate",
        "weak_boost_rate",
        "min_pos_scores_rate",
    ):
        writer.add_float32(f"sortformer.stream.{key}", float(sm[key]))

    counts = {}
    for gen in (encoder_tensors(sd, n_layer), head_tensors(sd, tf_layers), mel_tensors(sd)):
        for name, arr in gen:
            t = add_tensor(writer, name, arr, outtype)
            counts[t.name] = counts.get(t.name, 0) + 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}: tensors by type {counts}")


if __name__ == "__main__":
    main()

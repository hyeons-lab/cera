#!/usr/bin/env python3
"""Check that a converted Sortformer GGUF carries the same model as the `.nemo` it came from.

    python scripts/sortformer/verify_gguf.py \
        --nemo ~/.leap/models/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo \
        --gguf ~/.leap/models/sortformer/sortformer-4spk-v2.1-f32.gguf \
        --clip cera/tests/fixtures/sortformer/clip.wav \
        --tensors ~/.leap/models/sortformer/golden/golden.safetensors \
        --tol-pred 1e-3

It loads the GGUF's tensors back into NeMo's own modules (undoing the conversion: the folded
batch norm becomes mean 0 / var 1 - eps / affine = the folded scale and shift) and runs NeMo's
offline forward on the golden clip. If the conversion dropped, mis-transposed, mis-folded or
mis-named anything, the stage that first goes wrong shows up here, using NeMo as the executor, so
a mistake in cera's own forward pass cannot hide a converter bug (or the other way round).

Exit status is non-zero when the sigmoid outputs differ from the golden by more than --tol-pred
(max abs) or any stage's cosine similarity drops below --min-cos.
"""

import argparse
import sys
import warnings

import gguf
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

warnings.filterwarnings("ignore")

from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: E402

BN_EPS = 1e-5


def read_gguf(path):
    """name -> float32 torch tensor in the original (numpy / torch) axis order."""
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        shape = tuple(int(d) for d in reversed(t.shape))
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            arr = np.asarray(t.data).astype(np.float32).reshape(shape)
        else:
            arr = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        out[t.name] = torch.from_numpy(np.ascontiguousarray(arr))
    return out, r


def load_into_nemo(m, g, n_layer, tf_layers):
    sd = {}
    for i in (0, 2, 3, 5, 6):
        sd[f"encoder.pre_encode.conv.{i}.weight"] = g[f"a.conv1d.{i}.weight"]
        sd[f"encoder.pre_encode.conv.{i}.bias"] = g[f"a.conv1d.{i}.bias"].reshape(-1)
    sd["encoder.pre_encode.out.weight"] = g["a.pre_encode.out.weight"]
    sd["encoder.pre_encode.out.bias"] = g["a.pre_encode.out.bias"]
    for n in range(n_layer):
        e, a = f"encoder.layers.{n}", f"a.blk.{n}"
        for theirs, ours in (("norm_feed_forward1", "ffn_norm"), ("norm_feed_forward2", "ffn_norm_1"),
                             ("norm_self_att", "ln1"), ("norm_out", "ln2"), ("norm_conv", "norm_conv")):
            sd[f"{e}.{theirs}.weight"] = g[f"{a}.{ours}.weight"]
            sd[f"{e}.{theirs}.bias"] = g[f"{a}.{ours}.bias"]
        for theirs, ours in (("feed_forward1.linear1", "ffn_up"), ("feed_forward1.linear2", "ffn_down"),
                             ("feed_forward2.linear1", "ffn_up_1"), ("feed_forward2.linear2", "ffn_down_1")):
            sd[f"{e}.{theirs}.weight"] = g[f"{a}.{ours}.weight"]
            sd[f"{e}.{theirs}.bias"] = g[f"{a}.{ours}.bias"]
        for theirs, ours in (("linear_q", "attn_q"), ("linear_k", "attn_k"), ("linear_v", "attn_v"),
                             ("linear_out", "attn_out")):
            sd[f"{e}.self_attn.{theirs}.weight"] = g[f"{a}.{ours}.weight"]
            sd[f"{e}.self_attn.{theirs}.bias"] = g[f"{a}.{ours}.bias"]
        sd[f"{e}.self_attn.linear_pos.weight"] = g[f"{a}.linear_pos.weight"]
        sd[f"{e}.self_attn.pos_bias_u"] = g[f"{a}.pos_bias_u"]
        sd[f"{e}.self_attn.pos_bias_v"] = g[f"{a}.pos_bias_v"]
        for theirs, ours in (("pointwise_conv1", "conv_pw1"), ("pointwise_conv2", "conv_pw2")):
            sd[f"{e}.conv.{theirs}.weight"] = g[f"{a}.{ours}.weight"].unsqueeze(-1)
            sd[f"{e}.conv.{theirs}.bias"] = g[f"{a}.{ours}.bias"]
        sd[f"{e}.conv.depthwise_conv.weight"] = g[f"{a}.conv_dw.weight"].unsqueeze(1)
        sd[f"{e}.conv.depthwise_conv.bias"] = g[f"{a}.conv_dw.bias"]
        # Undo the fold: with mean 0 and var 1 - eps, BatchNorm computes x * weight + bias.
        sd[f"{e}.conv.batch_norm.weight"] = g[f"{a}.conv_norm.weight"]
        sd[f"{e}.conv.batch_norm.bias"] = g[f"{a}.conv_norm.bias"]
        sd[f"{e}.conv.batch_norm.running_mean"] = torch.zeros_like(g[f"{a}.conv_norm.weight"])
        sd[f"{e}.conv.batch_norm.running_var"] = torch.full_like(g[f"{a}.conv_norm.weight"], 1.0 - BN_EPS)
    p = "sortformer_modules"
    sd[f"{p}.encoder_proj.weight"] = g["sf.enc_proj.weight"]
    sd[f"{p}.encoder_proj.bias"] = g["sf.enc_proj.bias"]
    sd[f"{p}.first_hidden_to_hidden.weight"] = g["sf.head.hidden.weight"]
    sd[f"{p}.first_hidden_to_hidden.bias"] = g["sf.head.hidden.bias"]
    sd[f"{p}.single_hidden_to_spks.weight"] = g["sf.head.out.weight"]
    sd[f"{p}.single_hidden_to_spks.bias"] = g["sf.head.out.bias"]
    for n in range(tf_layers):
        t, s = f"transformer_encoder.layers.{n}", f"sf.blk.{n}"
        for theirs, ours in (("layer_norm_1", "ln1"), ("layer_norm_2", "ln2"),
                             ("first_sub_layer.query_net", "attn_q"), ("first_sub_layer.key_net", "attn_k"),
                             ("first_sub_layer.value_net", "attn_v"), ("first_sub_layer.out_projection", "attn_out"),
                             ("second_sub_layer.dense_in", "ffn_up"), ("second_sub_layer.dense_out", "ffn_down")):
            sd[f"{t}.{theirs}.weight"] = g[f"{s}.{ours}.weight"]
            sd[f"{t}.{theirs}.bias"] = g[f"{s}.{ours}.bias"]
    sd["preprocessor.featurizer.window"] = g["sf.mel.window"]
    sd["preprocessor.featurizer.fb"] = g["sf.mel.fb"].unsqueeze(0)

    # Every parameter and buffer we replace must exist with the same shape; NeMo's own
    # `hidden_to_spks` (unused at inference) is the only thing we intentionally leave alone.
    own = m.state_dict()
    missing = [k for k in sd if k not in own]
    bad = [(k, tuple(v.shape), tuple(own[k].shape)) for k, v in sd.items() if k in own and v.shape != own[k].shape]
    assert not missing, f"names not in NeMo model: {missing[:5]}"
    assert not bad, f"shape mismatches: {bad[:5]}"
    untouched = sorted(k for k in own if k not in sd and not k.endswith("num_batches_tracked"))
    assert untouched == ["sortformer_modules.hidden_to_spks.bias", "sortformer_modules.hidden_to_spks.weight"], untouched
    m.load_state_dict(sd, strict=False)


def cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--tol-pred", type=float, default=1e-3)
    ap.add_argument("--min-cos", type=float, default=0.99)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    golden = load_file(args.tensors)

    g, reader = read_gguf(args.gguf)
    meta = {f.name: f for f in reader.fields.values()}
    assert bytes(meta["general.architecture"].parts[-1]).decode() == "sortformer"

    m = SortformerEncLabelModel.restore_from(restore_path=args.nemo, map_location="cpu")
    m.eval()
    load_into_nemo(m, g, n_layer=17, tf_layers=18)

    pcm, _ = sf.read(args.clip, dtype="float32")
    audio = torch.from_numpy(pcm).unsqueeze(0)
    mel, mel_len = m.process_signal(audio, torch.tensor([audio.shape[1]]))
    stages = {"mel": mel[0]}

    grab = {}
    hooks = [m.encoder.pos_enc.register_forward_hook(lambda _m, _i, o: grab.__setitem__("xscaled", o[0][0]))]
    for i, layer in enumerate(m.encoder.layers):
        hooks.append(layer.register_forward_hook(lambda _m, _i, o, i=i: grab.__setitem__(f"enc.layer{i}", o[0])))
    for i, layer in enumerate(m.transformer_encoder.layers):
        hooks.append(layer.register_forward_hook(lambda _m, _i, o, i=i: grab.__setitem__(f"tf.layer{i}", o[0])))
    pre, _ = m.encoder.pre_encode(x=mel.transpose(1, 2), lengths=mel_len)
    stages["pre_encode"] = pre[0]
    emb, emb_len = m.frontend_encoder(processed_signal=mel, processed_signal_length=mel_len)
    preds = m.forward_infer(emb_seq=emb, emb_seq_length=emb_len)
    for h in hooks:
        h.remove()
    stages.update(grab)
    stages["enc_proj"] = emb[0]
    stages["preds_offline"] = preds[0]

    worst_cos, rows = 1.0, []
    order = ["mel", "pre_encode", "xscaled"] + [f"enc.layer{i}" for i in range(17)] + ["enc_proj"] \
        + [f"tf.layer{i}" for i in range(18)] + ["preds_offline"]
    for k in order:
        a, b = stages[k].float(), golden[k].float()
        d = float((a - b).abs().max())
        c = cos(a, b)
        worst_cos = min(worst_cos, c)
        rows.append((k, d, c))
    for k, d, c in rows:
        if k in ("mel", "pre_encode", "xscaled", "enc.layer0", "enc.layer8", "enc.layer16", "enc_proj",
                 "tf.layer0", "tf.layer17", "preds_offline"):
            print(f"  {k:14s} max|d|={d:10.3e}  cos={c:.7f}")
    pred_d = float((stages["preds_offline"] - golden["preds_offline"]).abs().max())
    flips = int(((stages["preds_offline"] > 0.5) != (golden["preds_offline"] > 0.5)).sum())
    print(f"{args.gguf.split('/')[-1]}: worst stage cosine {worst_cos:.7f}; sigmoid max|d| {pred_d:.3e}; "
          f"{flips} of {preds[0].numel()} speaker decisions flip at 0.5")
    ok = pred_d <= args.tol_pred and worst_cos >= args.min_cos
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

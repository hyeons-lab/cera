#!/usr/bin/env python3
"""Check that a converted Nemotron-3 GGUF carries the same model as the checkpoint.

    python scripts/nemotron3_diarization/verify_gguf.py \\
        --nemo ~/.leap/models/nemotron3-diarization/Nemotron-3-Diarization.nemo \\
        --gguf ~/.leap/models/nemotron3-diarization/nemotron3-diarization-f32.gguf \\
        --clip cera/tests/fixtures/sortformer/clip.wav \\
        --tensors ~/.leap/models/nemotron3-diarization/golden/golden.safetensors \\
        --tol-pred 1e-4

It loads the GGUF's tensors back into NeMo's own modules (the separate q/k/v become
NeMo's fused `w_qkv`, the reshaped upsampler becomes the Conv1d) and runs NeMo's
offline forward on the golden clip. If the conversion dropped, mis-transposed or
mis-named anything, the stage that first goes wrong shows up here, using NeMo as the
executor, so a mistake in cera's own forward pass cannot hide a converter bug (or the
other way round). The GGUF's weights come from the HuggingFace safetensors, so the f32
row also proves the HF conversion matches NeMo.

Exit status is non-zero when the sigmoid outputs differ from the golden by more than
--tol-pred (max abs) or any stage's cosine similarity drops below --min-cos.
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

N_LAYER = 31


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


def load_into_nemo(m, g):
    sd = {"encoder.pre_encode.proj.weight": g["nd.embed.proj.weight"]}
    for ours, theirs in (("nd.input_norm", "encoder.embed_norm"), ("nd.final_norm", "encoder.final_norm")):
        sd[f"{theirs}.weight"] = g[f"{ours}.weight"]
        sd[f"{theirs}.bias"] = g[f"{ours}.bias"]
    for n in range(N_LAYER):
        e, a = f"encoder.layers.{n}", f"nd.blk.{n}"
        sd[f"{e}.norm1.weight"] = g[f"{a}.ln1.weight"]
        sd[f"{e}.norm1.bias"] = g[f"{a}.ln1.bias"]
        sd[f"{e}.norm2.weight"] = g[f"{a}.ln2.weight"]
        sd[f"{e}.norm2.bias"] = g[f"{a}.ln2.bias"]
        sd[f"{e}.attn.w_qkv.weight"] = torch.cat(
            [g[f"{a}.attn_q.weight"], g[f"{a}.attn_k.weight"], g[f"{a}.attn_v.weight"]], dim=0
        )
        sd[f"{e}.attn.out_proj.weight"] = g[f"{a}.attn_o.weight"]
        sd[f"{e}.attn.out_proj.bias"] = g[f"{a}.attn_o.bias"]
        sd[f"{e}.ffn.net.0.weight"] = g[f"{a}.mlp_up.weight"]
        sd[f"{e}.ffn.net.0.bias"] = g[f"{a}.mlp_up.bias"]
        sd[f"{e}.ffn.net.3.weight"] = g[f"{a}.mlp_down.weight"]
        sd[f"{e}.ffn.net.3.bias"] = g[f"{a}.mlp_down.bias"]
    p = "sortformer_modules"
    sd[f"{p}.encoder_proj.weight"] = g["nd.proj.weight"]
    sd[f"{p}.encoder_proj.bias"] = g["nd.proj.bias"]
    sd[f"{p}.subpixel_upsample.weight"] = g["nd.upsample.weight"].reshape(1536, 192, 3)
    sd[f"{p}.subpixel_upsample.bias"] = g["nd.upsample.bias"]
    sd[f"{p}.first_hidden_to_hidden.weight"] = g["nd.classifier.dense.weight"]
    sd[f"{p}.first_hidden_to_hidden.bias"] = g["nd.classifier.dense.bias"]
    sd[f"{p}.single_hidden_to_spks.weight"] = g["nd.classifier.out.weight"]
    sd[f"{p}.single_hidden_to_spks.bias"] = g["nd.classifier.out.bias"]
    sd[f"{p}.learnable_sil_emb"] = g["nd.silence_embeds"]
    sd["preprocessor.featurizer.window"] = g["nd.mel.window"]
    sd["preprocessor.featurizer.fb"] = g["nd.mel.fb"].unsqueeze(0)

    # Every parameter and buffer we replace must exist with the same shape. The GGUF
    # intentionally leaves the two inference-unused heads alone: `hidden_to_spks` and
    # the auxiliary three-class `activity_head`.
    own = m.state_dict()
    missing = [k for k in sd if k not in own]
    bad = [(k, tuple(v.shape), tuple(own[k].shape)) for k, v in sd.items() if k in own and v.shape != own[k].shape]
    if missing:
        raise SystemExit(f"names not in NeMo model: {missing[:5]}")
    if bad:
        raise SystemExit(f"shape mismatches: {bad[:5]}")
    untouched = sorted(k for k in own if k not in sd and not k.endswith("num_batches_tracked"))
    expect_untouched = sorted(
        [
            "sortformer_modules.hidden_to_spks.weight",
            "sortformer_modules.hidden_to_spks.bias",
            "sortformer_modules.activity_head.0.weight",
            "sortformer_modules.activity_head.0.bias",
            "sortformer_modules.activity_head.1.weight",
            "sortformer_modules.activity_head.1.bias",
        ]
    )
    if untouched != expect_untouched:
        raise SystemExit(f"NeMo parameters the GGUF does not cover: {untouched}")
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
    arch = bytes(meta["general.architecture"].parts[-1]).decode()
    if arch != "nemotron3_diarization":
        raise SystemExit(f"{args.gguf}: general.architecture is {arch!r}, not nemotron3_diarization")

    m = SortformerEncLabelModel.restore_from(restore_path=args.nemo, map_location="cpu")
    m.eval()
    load_into_nemo(m, g)

    pcm, _ = sf.read(args.clip, dtype="float32")
    audio = torch.from_numpy(pcm).unsqueeze(0)
    mel, mel_len = m.process_signal(audio, torch.tensor([audio.shape[1]]))
    stages = {"mel": mel[0]}

    grab = {}
    hooks = [
        m.encoder.pre_encode.register_forward_hook(lambda _m, _i, o: grab.__setitem__("stacked", o[0][0])),
        m.encoder.embed_norm.register_forward_hook(lambda _m, _i, o: grab.__setitem__("input_norm", o[0])),
        m.encoder.final_norm.register_forward_hook(lambda _m, _i, o: grab.__setitem__("final_norm", o[0])),
    ]
    for i, layer in enumerate(m.encoder.layers):
        hooks.append(layer.register_forward_hook(lambda _m, _i, o, i=i: grab.__setitem__(f"enc.layer{i}", o[0])))
    emb, emb_len = m.frontend_encoder(processed_signal=mel, processed_signal_length=mel_len)
    preds = m.forward_infer(emb_seq=emb, emb_seq_length=emb_len)
    for h in hooks:
        h.remove()
    stages.update(grab)
    # The conv output itself: a pre-hook on the first linear would capture post-ReLU values.
    stages["upsampled"] = m.sortformer_modules.upsample_hidden(emb)[0]
    stages["enc_proj"] = emb[0]
    stages["preds_offline"] = preds[0]

    worst_cos, rows = 1.0, []
    order = ["mel", "stacked", "input_norm"] + [f"enc.layer{i}" for i in range(N_LAYER)] \
        + ["final_norm", "enc_proj", "upsampled", "preds_offline"]
    for k in order:
        a, b = stages[k].float(), golden[k].float()
        d = float((a - b).abs().max())
        c = cos(a, b)
        worst_cos = min(worst_cos, c)
        rows.append((k, d, c))
    show = {"mel", "stacked", "input_norm", "enc.layer0", "enc.layer15", "enc.layer30",
            "final_norm", "enc_proj", "upsampled", "preds_offline"}
    for k, d, c in rows:
        if k in show:
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

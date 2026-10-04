#!/usr/bin/env python3
"""Generate golden activations for Streaming Sortformer from NVIDIA's own NeMo implementation.

    python scripts/sortformer/gen_golden.py \
        --nemo ~/.leap/models/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo \
        --clip cera/tests/fixtures/sortformer/clip.wav \
        --json cera/tests/fixtures/sortformer/golden.json \
        --tensors ~/.leap/models/sortformer/golden/golden.safetensors

Needs the NeMo toolkit (`pip install nemo_toolkit[asr]`) and runs on CPU in float32.

Two outputs:

* `--json` (small, committed): config, per-stage statistics (shape, mean, |mean|, l2, min, max, head
  values), the full prediction matrices and per-step frame bounds (a step's chunk predictions are
  `total_preds[chunk_frames[0]:chunk_frames[1]]`). CI-sized and enough to pin the pipeline stage by stage.
* `--tensors` (large, NOT committed): every intermediate tensor, for cosine-level comparison and for
  the per-step cache state (`<preset>.step<N>.spkcache` etc.). Regenerate with this script.

What is captured
----------------
Offline (one pass over the whole clip): mel features, pre-encode embeddings, the x-scaled encoder
input, every FastConformer block output, the encoder output after `encoder_proj`, every Transformer
layer output, and the final sigmoids.

Streaming, for four presets (the checkpoint default, the model card's low-latency preset, and two
tiny-cache presets that make a 15 s clip overflow the speaker cache and exercise compression, the
silence profile and FIFO pop): per step, the chunk predictions and the state after the update
(speaker cache, cache predictions, FIFO, FIFO predictions, mean silence embedding, silence count).
The loop here is `SortformerEncLabelModel.forward_streaming` unrolled so the state can be recorded;
it is asserted equal to the real `forward_streaming` for every preset, and run twice to prove
`torch.topk(sorted=False)` ties do not make the trace non-deterministic.
"""

import argparse
import hashlib
import json
import math
import os
import warnings

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: E402
import nemo  # noqa: E402

PRESETS = {
    # name: (chunk_len, left_ctx, right_ctx, fifo_len, spkcache_len, update_period, record_state)
    "default": (188, 1, 1, 0, 188, 188, True),
    "low_latency": (6, 1, 7, 188, 188, 144, False),
    "tiny": (12, 1, 1, 20, 40, 12, True),
    "tiny_nofifo": (10, 1, 1, 0, 36, 10, True),
}


def stats(t):
    t = t.detach().float().cpu()
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return {"shape": list(t.shape), "empty": True}
    return {
        "shape": list(t.shape),
        "mean": float(flat.mean()),
        "abs_mean": float(flat.abs().mean()),
        "l2": float(flat.norm()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "head": [float(v) for v in flat[:8]],
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_featurizer(m):
    """The converter hard-codes the NeMo defaults that are not in model_config.yaml; pin them."""
    f = m.preprocessor.featurizer
    expect = dict(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=512,
        preemph=0.97,
        log=True,
        log_zero_guard_type="add",
        pad_to=16,
        frame_splicing=1,
        mag_power=2.0,
        normalize="NA",
        nfilt=128,
    )
    for k, v in expect.items():
        got = getattr(f, k)
        assert got == v, f"featurizer.{k}: {got!r} != {v!r}"
    assert abs(f.log_zero_guard_value - 2.0**-24) < 1e-12, f.log_zero_guard_value


def set_preset(sm, p):
    sm.chunk_len, sm.chunk_left_context, sm.chunk_right_context = p[0], p[1], p[2]
    sm.fifo_len, sm.spkcache_len, sm.spkcache_update_period = p[3], p[4], p[5]


def run_stream(m, mel, mel_len, record):
    """Unrolled `forward_streaming`; returns (total_preds, per-step records)."""
    sm = m.sortformer_modules
    state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=m.device)
    total = torch.zeros((1, 0, sm.n_spk))
    loader = sm.streaming_feat_loader(
        feat_seq=mel, feat_seq_length=mel_len, feat_seq_offset=torch.zeros((1,), dtype=torch.long)
    )
    steps = []
    for idx, chunk_t, lens, lo, ro in loader:
        prev = total.shape[1]
        state, total = m.forward_streaming_step(
            processed_signal=chunk_t,
            processed_signal_length=lens,
            streaming_state=state,
            total_preds=total,
            left_offset=lo,
            right_offset=ro,
        )
        rec = {
            "idx": idx,
            "left_offset": lo,
            "right_offset": ro,
            "n_feat": int(chunk_t.shape[1]),
            "len": int(lens[0]),
            "chunk_frames": [prev, total.shape[1]],
            "chunk_preds": total[0, prev:].detach().clone(),
        }
        if record:
            for name in ("spkcache", "spkcache_preds", "fifo", "fifo_preds", "mean_sil_emb"):
                v = getattr(state, name)
                rec[name] = None if v is None else v.detach().clone()
            rec["n_sil_frames"] = state.n_sil_frames.detach().clone()
        steps.append(rec)
    return total, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--tensors", required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    m = SortformerEncLabelModel.restore_from(restore_path=args.nemo, map_location="cpu")
    m.eval()
    assert not m.async_streaming and m.streaming_mode

    check_featurizer(m)

    pcm, sr = sf.read(args.clip, dtype="float32")
    assert sr == 16000 and pcm.ndim == 1, (sr, pcm.shape)
    audio = torch.from_numpy(pcm).unsqueeze(0)
    audio_len = torch.tensor([audio.shape[1]])

    T = {}  # full tensors, saved to --tensors
    J = {"stages": {}, "presets": {}}

    # ---- front end -------------------------------------------------------------------------
    mel, mel_len = m.process_signal(audio, audio_len)  # (1, 128, T_padded), pad_to=16
    T["mel"] = mel[0].contiguous()
    J["mel_len"] = int(mel_len[0])
    J["stages"]["mel"] = stats(mel)

    # ---- offline pass, hooks on every block --------------------------------------------------
    hooks, grab = [], {}

    def hook(name, pick=lambda o: o):
        def fn(_mod, _inp, out):
            grab[name] = pick(out).detach().clone()

        return fn

    hooks.append(m.encoder.pos_enc.register_forward_hook(hook("xscaled", lambda o: o[0])))
    for i, layer in enumerate(m.encoder.layers):
        hooks.append(layer.register_forward_hook(hook(f"enc.layer{i}")))
    for i, layer in enumerate(m.transformer_encoder.layers):
        hooks.append(layer.register_forward_hook(hook(f"tf.layer{i}")))

    pre, pre_len = m.encoder.pre_encode(x=mel.transpose(1, 2), lengths=mel_len)
    T["pre_encode"] = pre[0].contiguous()
    emb, emb_len = m.frontend_encoder(processed_signal=mel, processed_signal_length=mel_len)
    preds = m.forward_infer(emb_seq=emb, emb_seq_length=emb_len)
    for h in hooks:
        h.remove()
    n_valid = int(emb_len[0])
    J["n_frames"] = n_valid

    T["xscaled"] = grab.pop("xscaled")[0].contiguous()
    for i in range(len(m.encoder.layers)):
        T[f"enc.layer{i}"] = grab.pop(f"enc.layer{i}")[0].contiguous()
    T["enc_proj"] = emb[0].contiguous()
    for i in range(len(m.transformer_encoder.layers)):
        T[f"tf.layer{i}"] = grab.pop(f"tf.layer{i}")[0].contiguous()
    T["preds_offline"] = preds[0].contiguous()

    for k in ("pre_encode", "xscaled", "enc_proj", "preds_offline"):
        J["stages"][k] = stats(T[k])
    for i in (0, 8, 16):
        J["stages"][f"enc.layer{i}"] = stats(T[f"enc.layer{i}"])
    for i in (0, 8, 17):
        J["stages"][f"tf.layer{i}"] = stats(T[f"tf.layer{i}"])
    J["preds_offline"] = [[round(float(v), 6) for v in row] for row in preds[0, :n_valid]]

    # ---- streaming presets -------------------------------------------------------------------
    sm = m.sortformer_modules
    saved = (sm.chunk_len, sm.chunk_left_context, sm.chunk_right_context, sm.fifo_len, sm.spkcache_len, sm.spkcache_update_period)
    for name, p in PRESETS.items():
        set_preset(sm, p)
        total, steps = run_stream(m, mel, mel_len, record=p[6])
        total2, _ = run_stream(m, mel, mel_len, record=False)
        assert torch.equal(total, total2), f"{name}: streaming trace is not deterministic"
        ref = m.forward_streaming(mel, mel_len)
        assert torch.equal(total, ref), f"{name}: unrolled loop differs from forward_streaming"
        assert total.shape[1] >= n_valid

        step_json = []
        for i, st in enumerate(steps):
            sj = {k: st[k] for k in ("idx", "left_offset", "right_offset", "n_feat", "len", "chunk_frames")}
            if p[6]:
                T[f"{name}.step{i}.chunk_preds"] = st["chunk_preds"].contiguous()
                for key in ("spkcache", "spkcache_preds", "fifo", "fifo_preds", "mean_sil_emb", "n_sil_frames"):
                    v = st[key]
                    if v is None:
                        sj[key] = None
                        continue
                    sj[key] = stats(v.float())
                    if v.numel():
                        T[f"{name}.step{i}.{key}"] = v[0].contiguous() if v.dim() > 1 else v.contiguous()
            step_json.append(sj)
        T[f"{name}.total_preds"] = total[0].contiguous()
        J["presets"][name] = {
            "params": dict(
                chunk_len=p[0],
                chunk_left_context=p[1],
                chunk_right_context=p[2],
                fifo_len=p[3],
                spkcache_len=p[4],
                spkcache_update_period=p[5],
            ),
            "n_steps": len(steps),
            "state_recorded": p[6],
            "total_preds": [[round(float(v), 6) for v in row] for row in total[0, :n_valid]],
            "steps": step_json,
        }
    (sm.chunk_len, sm.chunk_left_context, sm.chunk_right_context, sm.fifo_len, sm.spkcache_len, sm.spkcache_update_period) = saved

    # ---- metadata ----------------------------------------------------------------------------
    sm_cfg = {
        k: getattr(sm, k)
        for k in (
            "n_spk",
            "spkcache_sil_frames_per_spk",
            "pred_score_threshold",
            "scores_boost_latest",
            "sil_threshold",
            "strong_boost_rate",
            "weak_boost_rate",
            "min_pos_scores_rate",
            "max_index",
        )
    }
    J["meta"] = {
        "nemo_version": nemo.__version__,
        "torch_version": torch.__version__,
        "nemo_sha256": sha256(args.nemo),
        "clip_sha256": sha256(args.clip),
        "clip_samples": int(audio.shape[1]),
        "subsampling_factor": 8,
        "streaming_modules": sm_cfg,
        "note": "preds are sigmoid outputs, [frames x 4 speakers]; frame = 80 ms; golden is the synchronous "
        "(async_streaming=False) NeMo path",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.tensors)), exist_ok=True)
    from safetensors.torch import save_file

    save_file({k: v.float().contiguous() if v.is_floating_point() else v.contiguous() for k, v in T.items()}, args.tensors)
    with open(args.json, "w") as f:
        json.dump(J, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.json} ({os.path.getsize(args.json)} B) and {args.tensors} ({os.path.getsize(args.tensors)} B, {len(T)} tensors)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate golden activations for Nemotron-3-Diarization from NVIDIA's NeMo implementation.

    python scripts/nemotron3_diarization/gen_golden.py \\
        --nemo ~/.leap/models/nemotron3-diarization/Nemotron-3-Diarization.nemo \\
        --clip cera/tests/fixtures/sortformer/clip.wav \\
        --json cera/tests/fixtures/nemotron3/golden.json \\
        --tensors ~/.leap/models/nemotron3-diarization/golden/golden.safetensors

Needs the NeMo toolkit from the Speech repo (PyPI nemo_toolkit 3.0.0 lacks the rope
encoder; install `nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git`)
and runs on CPU in float32.

Two outputs:

* `--json` (small, committed): config, per-stage statistics (shape, mean, |mean|, l2,
  min, max, head values), the full prediction matrices and per-step frame bounds (a
  step's chunk predictions are `total_preds[chunk_frames[0]:chunk_frames[1]]`).
  CI-sized and enough to pin the pipeline stage by stage.
* `--tensors` (large, NOT committed): every intermediate tensor, for cosine-level
  comparison and for the per-step cache state (`<preset>.step<N>.spkcache` etc.).
  Regenerate with this script.

What is captured
----------------
Offline (one pass over the whole clip): mel features, stacked+projected embeddings,
the input-LN output, all 31 encoder block outputs, the final-LN output, the
encoder-projected output, the upsampled classifier input, the logits, and the sigmoids.
There is no transformer head (`transformer_encoder is None`, refused otherwise).

Streaming, for five presets (the checkpoint default, the model card's low-latency and
ultra-low-latency presets, and two tiny-cache presets that make a 15 s clip overflow
the speaker cache and exercise compression, FIFO pop and the learned silence slots):
per step, the chunk predictions and the state after the update (speaker cache, cache
lengths, cache predictions, compression flag, FIFO, FIFO lengths, FIFO predictions,
speaker permutation, mean silence embedding, silence count — the last two stay
untouched because the silence embedding is learned).
The loop here is `SortformerEncLabelModel.forward_streaming` unrolled so the state can
be recorded; it must equal the real `forward_streaming` for every preset,
and run twice to prove `torch.topk(sorted=False)` ties do not make the trace
non-deterministic.
"""

import argparse
import hashlib
import json
import os
import warnings

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: E402
import nemo  # noqa: E402


def require(cond, msg):
    """Refuse with a readable message (an `assert` vanishes under -O). Same as convert.py."""
    if not cond:
        raise SystemExit(f"gen_golden: {msg}")


PRESETS = {
    # name: (chunk_len, left_ctx, right_ctx, fifo_len, spkcache_len, update_period, record_state)
    "default": (264, 0, 0, 0, 264, 264, True),
    "low_latency": (9, 0, 4, 264, 264, 222, False),
    "ultra_low_latency": (3, 0, 1, 264, 264, 222, False),
    "tiny": (6, 0, 2, 8, 12, 6, True),
    "tiny_nofifo": (6, 0, 2, 0, 12, 6, True),
}

STATE_KEYS = (
    "spkcache", "spkcache_lengths", "spkcache_preds", "spkcache_compressed",
    "fifo", "fifo_lengths", "fifo_preds", "spk_perm", "mean_sil_emb", "n_sil_frames",
)


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
        require(got == v, f"featurizer.{k}: {got!r} != {v!r}")
    require(abs(f.log_zero_guard_value - 2.0**-24) < 1e-12,
            f"log_zero_guard_value {f.log_zero_guard_value}")


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
            for name in STATE_KEYS:
                v = getattr(state, name)
                if isinstance(v, bool):
                    rec[name] = v
                else:
                    rec[name] = None if v is None else v.detach().clone()
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
    require(not m.async_streaming and m.streaming_mode,
            "expected streaming_mode without async_streaming")
    require(m.transformer_encoder is None, "expected no transformer head")
    require(m.sortformer_modules.use_learnable_sil_emb,
            "expected learnable silence embeddings")

    check_featurizer(m)

    pcm, sr = sf.read(args.clip, dtype="float32")
    require(sr == 16000 and pcm.ndim == 1,
            f"need mono 16 kHz PCM, got sr={sr} shape={pcm.shape}")
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

    def hook(name, pick=lambda o: o[0] if isinstance(o, tuple) else o):
        def fn(_mod, _inp, out):
            grab[name] = pick(out).detach().clone()

        return fn

    hooks.append(m.encoder.pre_encode.register_forward_hook(hook("stacked")))
    hooks.append(m.encoder.embed_norm.register_forward_hook(hook("input_norm")))
    for i, layer in enumerate(m.encoder.layers):
        hooks.append(layer.register_forward_hook(hook(f"enc.layer{i}")))
    hooks.append(m.encoder.final_norm.register_forward_hook(hook("final_norm")))

    emb, emb_len = m.frontend_encoder(processed_signal=mel, processed_signal_length=mel_len)
    preds, logits, activity = m.forward_infer(emb_seq=emb, emb_seq_length=emb_len, return_logits=True)
    for h in hooks:
        h.remove()
    # The conv output itself, not the classifier input: forward_speaker_logits applies ReLU
    # before its first linear, so a pre-hook there would capture post-ReLU values.
    upsampled = m.sortformer_modules.upsample_hidden(emb)
    n_mel = int(mel_len[0])
    n_enc = int(emb_len[0])
    J["n_mel_frames"] = n_mel
    J["n_enc_frames"] = n_enc

    T["stacked"] = grab.pop("stacked")[0].contiguous()
    T["input_norm"] = grab.pop("input_norm")[0].contiguous()
    for i in range(len(m.encoder.layers)):
        T[f"enc.layer{i}"] = grab.pop(f"enc.layer{i}")[0].contiguous()
    T["final_norm"] = grab.pop("final_norm")[0].contiguous()
    T["enc_proj"] = emb[0].contiguous()
    T["upsampled"] = upsampled[0].contiguous()
    T["logits"] = logits[0].contiguous()
    T["preds_offline"] = preds[0].contiguous()
    if activity is not None:
        T["activity"] = activity[0].contiguous()

    for k in ("stacked", "input_norm", "final_norm", "enc_proj", "upsampled", "logits", "preds_offline"):
        J["stages"][k] = stats(T[k])
    for i in (0, 10, 20, 30):
        J["stages"][f"enc.layer{i}"] = stats(T[f"enc.layer{i}"])
    J["preds_offline"] = [[round(float(v), 6) for v in row] for row in preds[0, :n_mel]]

    # ---- streaming presets -------------------------------------------------------------------
    sm = m.sortformer_modules
    saved = (sm.chunk_len, sm.chunk_left_context, sm.chunk_right_context, sm.fifo_len, sm.spkcache_len, sm.spkcache_update_period)
    for name, p in PRESETS.items():
        set_preset(sm, p)
        total, steps = run_stream(m, mel, mel_len, record=p[6])
        total2, _ = run_stream(m, mel, mel_len, record=False)
        require(torch.equal(total, total2), f"{name}: streaming trace is not deterministic")
        ref = m.forward_streaming(mel, mel_len)
        require(torch.equal(total, ref), f"{name}: unrolled loop differs from forward_streaming")
        require(total.shape[1] >= n_mel,
                f"{name}: {total.shape[1]} frames for {n_mel} mel frames")

        step_json = []
        for i, st in enumerate(steps):
            sj = {k: st[k] for k in ("idx", "left_offset", "right_offset", "n_feat", "len", "chunk_frames")}
            if p[6]:
                T[f"{name}.step{i}.chunk_preds"] = st["chunk_preds"].contiguous()
                for key in STATE_KEYS:
                    v = st[key]
                    if isinstance(v, bool):
                        sj[key] = v
                        continue
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
            "total_preds": [[round(float(v), 6) for v in row] for row in total[0, :n_mel]],
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
        "note": "preds are sigmoid outputs, [frames x 8 speakers]; frames are 10 ms mel frames; "
        "golden is the synchronous (async_streaming=False) NeMo path",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.tensors)), exist_ok=True)
    from safetensors.torch import save_file

    save_file({k: v.float().contiguous() if v.is_floating_point() else v.contiguous() for k, v in T.items()}, args.tensors)
    with open(args.json, "w") as f:
        json.dump(J, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.json} ({os.path.getsize(args.json)} B) and {args.tensors} ({os.path.getsize(args.tensors)} B, {len(T)} tensors)")


if __name__ == "__main__":
    main()

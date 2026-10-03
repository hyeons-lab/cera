#!/usr/bin/env python3
"""Build the 15.4 s multi-voice test clip used by the Sortformer golden fixtures.

The clip is committed (cera/tests/fixtures/sortformer/clip.wav), so this script only documents
how it was made. The four source recordings are not in the repository, so rebuilding needs your
own 16 kHz-or-higher speech files in their place; the committed clip is the fixture of record.

Voices A/B/C are three different recordings; the clip has silence gaps (so the speaker cache sees silence frames) and one overlapped region (A over B).

    python scripts/sortformer/make_clip.py --en models/en.wav --fool fool_me_once_mono.wav \
        --kyoko1 kyoko1.wav --kyoko2 kyoko2.wav --out cera/tests/fixtures/sortformer/clip.wav
"""

import argparse
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000


def load(path, start_s, dur_s):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    seg = x[int(start_s * SR) : int((start_s + dur_s) * SR)]
    if seg.size == 0:
        raise SystemExit(f"{path}: no audio at {start_s}..{start_s + dur_s} s (file is {len(x) / SR:.1f} s long)")
    peak = float(np.abs(seg).max()) or 1.0
    return (0.5 * seg / peak).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True, help="voice A (English TTS, 16 kHz)")
    ap.add_argument("--fool", required=True, help="voice B (44.1 kHz mono)")
    ap.add_argument("--kyoko1", required=True, help="voice C")
    ap.add_argument("--kyoko2", required=True, help="voice C continued")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sil = lambda s: np.zeros(int(s * SR), dtype=np.float32)
    a1 = load(args.en, 0.0, 3.5)
    b1 = load(args.fool, 0.0, 3.5)
    c1 = load(args.kyoko1, 0.0, 2.0)
    a2 = load(args.en, 12.0, 2.0)
    b2 = load(args.fool, 4.0, 2.0)
    c2 = load(args.kyoko2, 0.0, 2.0)

    n = min(len(a2), len(b2))
    overlap = np.clip(a2[:n] + b2[:n], -1.0, 1.0)  # A over B

    clip = np.concatenate([sil(0.5), a1, sil(0.5), b1, sil(0.5), c1, sil(0.5), overlap, sil(0.5), c2])
    clip = clip[: 16 * SR]
    sf.write(args.out, clip, SR, subtype="PCM_16")
    print(f"wrote {args.out}: {len(clip) / SR:.2f} s")


if __name__ == "__main__":
    main()

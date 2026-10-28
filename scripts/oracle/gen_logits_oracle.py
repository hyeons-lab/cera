#!/usr/bin/env python3
"""Add a final-logits summary to an existing oracle fixture set.

`gen_text_oracle.py` records per-node activation sums. For models whose residual
stream sums cancel toward zero (the sandwich-norm `lfm3`), those are weak
checksums, so this records the last-token logits too. It runs llama.cpp's
`llama-debug --save-logits` (CPU only) on each prompt of a set and writes, into
that prompt's fixture JSON, an object (under `--key`, default `logits`):

    {"n": <vocab>, "mean": ..., "std": ..., "top": [[token_id, logit], ...]}

Run it once per reference model: the quantized file under test (`logits`) and,
where the checkpoint allows, a bf16 conversion (`--key logits_bf16`), which has
no activation quantization and so is the tight check on the graph itself.

The full vector is 512 KB per prompt, so only the top `--top-k` entries and the
first two moments are kept: enough to gate top-1 agreement, top-k overlap, and
the distribution's location and scale.

Usage:
  DYLD_LIBRARY_PATH=$BIN python3 scripts/oracle/gen_logits_oracle.py \
    --bin-dir $BIN --model <model.gguf> --fixture-dir cera/tests/fixtures/oracle-lfm3/lfm3-1.7b \
    [--key logits_bf16]
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile


def read_logits(path):
    raw = open(path, "rb").read()
    if len(raw) % 4:
        raise SystemExit(f"{path}: {len(raw)} bytes is not a whole number of f32")
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def summarize(logits, top_k):
    n = len(logits)
    if n == 0:
        raise SystemExit("the logits file is empty; llama-debug wrote nothing to compare against")
    # json.dump would write a bare NaN/Infinity (not JSON, and a reference the parity gate
    # cannot compare against): refuse to record a broken reference
    bad = next((i for i, v in enumerate(logits) if not math.isfinite(v)), None)
    if bad is not None:
        raise SystemExit(f"logit {bad} is {logits[bad]}: the reference run produced non-finite logits")
    mean = sum(logits) / n
    var = sum((v - mean) ** 2 for v in logits) / n
    top = sorted(range(n), key=lambda i: logits[i], reverse=True)[:top_k]
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "top": [[i, logits[i]] for i in top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--key", default="logits", help="fixture field to write")
    args = ap.parse_args()

    env = dict(os.environ, DYLD_LIBRARY_PATH=args.bin_dir, LD_LIBRARY_PATH=args.bin_dir)
    index_path = os.path.join(args.fixture_dir, "index.json")
    index = json.load(open(index_path))
    stem = os.path.splitext(os.path.basename(args.model))[0]

    for entry in index["prompts"]:
        fx_path = os.path.join(args.fixture_dir, entry["fixture"])
        fx = json.load(open(fx_path))
        with tempfile.TemporaryDirectory() as out_dir:
            cmd = [
                os.path.join(args.bin_dir, "llama-debug"),
                "-m", args.model, "-p", fx["prompt"], "-ngl", "0",
                "--save-logits", "--logits-output-dir", out_dir,
            ]
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if out.returncode != 0:
                raise SystemExit(
                    f"command failed ({out.returncode}): {' '.join(cmd)}\n{out.stderr}"
                )
            fx[args.key] = summarize(
                read_logits(os.path.join(out_dir, f"llamacpp-{stem}.bin")), args.top_k
            )
        # write beside the target and rename, so an interrupt cannot truncate a fixture
        tmp_path = fx_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(fx, f, indent=2, allow_nan=False)
            os.replace(tmp_path, fx_path)
        except BaseException:
            # leave no stray temp file in the committed fixture directory
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        top1 = fx[args.key]["top"][0]
        print(f"[oracle] {entry['fixture']}: top1 id={top1[0]} logit={top1[1]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()

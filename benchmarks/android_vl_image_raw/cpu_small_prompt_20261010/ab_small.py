#!/usr/bin/env python3
"""TTFT A/B of two cera binaries on the small-prompt path, alternating them, REPS repetitions, p50 TTFT from `cera bench`.
SERIAL=<adb serial>, OLD / NEW = binary names in /data/local/tmp/cmp-cera (default cera-ttft / cera-smallp, the 2/4-worker steps
against the whole pool); usage: ab_small.py OUT REPS."""
import json, os, re, subprocess, sys, time

if "SERIAL" not in os.environ:
    sys.exit("error: SERIAL environment variable is required (adb device serial)")
if len(sys.argv) < 3:
    sys.exit("usage: ab_small.py OUT REPS")

SERIAL = os.environ["SERIAL"]
OUT = sys.argv[1]
REPS = int(sys.argv[2])
AP_MAX_C = 30.0
CERA = "/data/local/tmp/cmp-cera"
OLD = os.environ.get("OLD", "cera-ttft")
NEW = os.environ.get("NEW", "cera-smallp")
for name, b in [("OLD", OLD), ("NEW", NEW)]:
    if not re.match(r"^[a-zA-Z0-9_.-]+$", b):
        sys.exit(f"error: invalid binary name in {name}: {b!r}")
MODELS = [
    ("llama-3.2-1b-q4_0", "Llama-3.2-1B-Q4_0.gguf"),
    ("lfm2.5-vl-450m-q4_0", "LFM2.5-VL-450M-Q4_0.gguf"),
    ("lfm2.5-350m-q4_k_m", "LFM2.5-350M-Q4_K_M.gguf"),
    ("lfm2.5-350m-q8_0", "LFM2.5-350M-Q8_0.gguf"),
    ("lfm2.5-2.6b-agent-q4_0", "agent-q4_0.gguf"),
    ("lfm2.5-2.6b-q5_k_m", "LFM2.5-2.6B-Q5_K_M.gguf"),
]
NS = [int(x.strip()) for x in os.environ.get("NS", "16,64,512").split(",") if x.strip()]
def sh(cmd, timeout=900):
    r = subprocess.run(
        ["adb", "-s", SERIAL, "shell", cmd],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    return (r.stdout + r.stderr).replace("\r", "")


def ap():
    m = re.search(
        r"mValue=([\d.]+)",
        sh(
            "dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"
        ),
    )
    return float(m.group(1)) if m else None


def cool():
    t = ap()
    if t is None:
        return None
    t0 = time.time()
    while True:
        if t <= AP_MAX_C or time.time() - t0 > 600:
            return t
        time.sleep(4)
        t = ap()
        if t is None:
            return None


def ttft(binary, gguf, n):
    out = sh(
        f"cd {CERA} && timeout 600 ./{binary} bench -m /data/local/tmp/{gguf} --device cpu --runs 5 --warmup 2 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1"
    )
    m = re.search(r"ttft ms: p50=([\d.]+)", out)
    if not m:
        sys.stderr.write(
            f"warning: benchmark failed for {binary} {gguf} n={n}:\n{out}\n"
        )
        return None
    return float(m.group(1))


only_models = (
    {x.strip() for x in os.environ["ONLY"].split(",") if x.strip()}
    if os.environ.get("ONLY")
    else None
)

for rep in range(REPS):
    for label, gguf in MODELS:
        if only_models and label not in only_models:
            continue
        for n in NS:
            order = [OLD, NEW] if (rep + n) % 2 == 0 else [NEW, OLD]
            for b in order:
                t = cool()
                row = dict(
                    model=label,
                    n=n,
                    rep=rep,
                    binary=b,
                    ttft_ms=ttft(b, gguf, n),
                    ap_before=t,
                )
                print(json.dumps(row), flush=True)
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")

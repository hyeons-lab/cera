# Alternates the binary `cera-attnpf` with CERA_ATTN_PREFETCH=0 CERA_ATTN_PREFETCH_L2=0 (off) and its defaults (on); decode tok/s p50 of 3 runs.
# usage: SERIAL=<adb serial> ab_prefetch.py REPS model1,model2 depth1,depth2 (AP gate 36 C: the phone idles at 34 C when warm)
import os
import re
import statistics as st
import subprocess
import sys
import time

if "SERIAL" not in os.environ:
    sys.exit("error: SERIAL environment variable is required (adb device serial)")
SERIAL = os.environ["SERIAL"]


def sh(c, t=900):
    r = subprocess.run(
        ["adb", "-s", SERIAL, "shell", c],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=t,
    )
    return r.stdout + r.stderr


def ap():
    m = re.search(
        r"mValue=([\d.]+)",
        sh(
            "dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"
        ),
    )
    return float(m.group(1)) if m else None


def cool(mx=36.0):
    t = ap()
    if t is None:
        return
    t0 = time.time()
    while time.time() - t0 < 600:
        if t <= mx:
            return
        time.sleep(4)
        t = ap()
        if t is None:
            return


def dec(model, depth, env):
    o = sh(
        f"cd /data/local/tmp/cmp-cera && timeout 600 {env} ./cera-attnpf bench -m /data/local/tmp/{model}.gguf --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {depth} --max-tokens 48 2>&1"
    )
    m = re.search(r"decode tok/s: p50=([\d.]+)", o)
    if not m:
        sys.stderr.write(
            f"warning: benchmark failed for {model} depth={depth} env='{env}':\n{o}\n"
        )
        return None
    return float(m.group(1))


if len(sys.argv) < 4:
    sys.exit("usage: SERIAL=<adb serial> ab_prefetch.py REPS model1,model2 depth1,depth2")

REPS = int(sys.argv[1])
cases = [
    (m, d)
    for m in sys.argv[2].split(",")
    for d in map(int, sys.argv[3].split(","))
]
OFF = "CERA_ATTN_PREFETCH=0 CERA_ATTN_PREFETCH_L2=0"
print(
    f"{'model':24s} {'depth':>5s} | {'off':>7s} {'on':>7s} {'change':>7s} | raw off / on"
)
for m, d in cases:
    r = {"off": [], "on": []}
    for i in range(REPS):
        order = ["off", "on"] if i % 2 == 0 else ["on", "off"]
        for mode in order:
            cool()
            val = dec(m, d, OFF if mode == "off" else "")
            if val is not None:
                r[mode].append(val)
    if not r["off"] or not r["on"]:
        print(f"{m:24s} {d:5d} | FAILED | {r['off']} / {r['on']}", flush=True)
        continue
    o, n = st.median(r["off"]), st.median(r["on"])
    pct = (n / o - 1) * 100 if o > 0 else 0.0
    print(
        f"{m:24s} {d:5d} | {o:7.1f} {n:7.1f} {pct:+6.0f}% | {r['off']} / {r['on']}",
        flush=True,
    )

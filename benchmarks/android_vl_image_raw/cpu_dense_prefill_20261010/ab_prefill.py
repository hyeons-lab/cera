#!/usr/bin/env python3
import os
import re
import statistics as st
import subprocess
import sys
import time

if len(sys.argv) < 5:
    sys.exit("usage: ab_prefill.py BINARY_A BINARY_B REPS PROMPT_TOKENS")

SERIAL = os.environ.get("SERIAL")
ADB = ["adb", "-s", SERIAL] if SERIAL else ["adb"]
A, B = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3])
n = int(sys.argv[4])
M = os.environ.get("MODEL", "/data/local/tmp/Llama-3.2-1B-Q4_0.gguf")
CERA = os.environ.get("CERA", "/data/local/tmp/cmp-cera")
AP_MAX_C = 36.0


def ap():
    cmd = ADB + [
        "shell",
        "dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    m = re.search(r"mValue=([0-9.]+)", p.stdout)
    return float(m.group(1)) if m else None


def cool(timeout=600):
    t = ap()
    if t is None:
        return None
    t0 = time.time()
    while True:
        if t <= AP_MAX_C or (time.time() - t0 > timeout):
            return t
        time.sleep(15)
        t = ap()
        if t is None:
            return None


def run(b):
    cool()
    cmd = ADB + [
        "shell",
        f"cd {CERA} && ./{b} bench -m {M} --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1",
    ]
    o = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"prefill tok/s: p50=([0-9.]+)", o)
    if not m:
        sys.stderr.write(f"warning: benchmark failed for {b}:\n{o}\n")
        return 0.0
    return float(m.group(1))


def main():
    r = {A: [], B: []}
    for i in range(reps):
        for b in [A, B] if i % 2 == 0 else [B, A]:
            r[b].append(run(b))
    for b in (A, B):
        print(b, "p50", round(st.median(r[b]), 1), r[b])
    med_a, med_b = st.median(r[A]), st.median(r[B])
    if med_a > 0:
        print("change %+.1f%%" % ((med_b / med_a - 1) * 100))
    else:
        print("change undefined (baseline median is zero)")


if __name__ == "__main__":
    main()

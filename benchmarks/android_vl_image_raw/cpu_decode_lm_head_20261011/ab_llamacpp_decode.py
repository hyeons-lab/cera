#!/usr/bin/env python3
import os
import re
import shlex
import statistics as st
import subprocess
import sys
import time

if len(sys.argv) < 3:
    sys.exit("usage: ab_llamacpp_decode.py REPS MAX_TOKENS")

reps = int(sys.argv[1])
n = int(sys.argv[2])
if reps <= 0 or n <= 0:
    sys.exit("error: REPS and MAX_TOKENS must be positive integers")

M = os.environ.get("MODEL", "Llama-3.2-1B-Q4_0")
if not re.match(r"^[a-zA-Z0-9_.-]+$", M):
    sys.exit(f"error: invalid model name: {M!r}")

SERIAL = os.environ.get("SERIAL")
ADB = ["adb", "-s", SERIAL] if SERIAL else ["adb"]
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


def cera():
    cool()
    cmd = ADB + [
        "shell",
        f"cd /data/local/tmp/cmp-cera && ./cera-q6 bench -m /data/local/tmp/{shlex.quote(M)}.gguf --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens 16 --max-tokens {n} 2>&1",
    ]
    o = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"decode tok/s: p50=([0-9.]+)", o)
    if not m:
        sys.stderr.write(f"warning: cera benchmark failed:\n{o}\n")
        return 0.0
    return float(m.group(1))


def llama():
    cool()
    cmd = ADB + [
        "shell",
        f"cd /data/local/tmp/cmp-llamacpp && LD_LIBRARY_PATH=lib ./bin/llama-bench -m /data/local/tmp/{shlex.quote(M)}.gguf -t 8 -p 0 -n {n} -r 3 -ngl 0 --device none 2>&1",
    ]
    o = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(rf"tg{n}\s*\|\s*([0-9.]+)", o)
    if not m:
        sys.stderr.write(f"warning: llama-bench failed:\n{o}\n")
        return 0.0
    return float(m.group(1))


def main():
    c = []
    l = []
    for i in range(reps):
        if i % 2 == 0:
            c.append(cera())
            l.append(llama())
        else:
            l.append(llama())
            c.append(cera())
    med_c, med_l = st.median(c), st.median(l)
    print(f"n={n} cera p50 {med_c:.0f} {c}\nn={n} llama.cpp p50 {med_l:.0f} {l}")
    if med_l > 0:
        print(f"ratio cera/llama {med_c / med_l:.3f}")
    else:
        print("ratio cera/llama undefined (llama.cpp median is zero)")


if __name__ == "__main__":
    main()

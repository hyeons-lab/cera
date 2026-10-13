#!/usr/bin/env python3
# usage: ab_env.py MODEL REPS N "[ENV=v ]binA" "[ENV=v ]binB"   (variants alternate)
import os
import re
import shlex
import statistics as st
import subprocess
import sys
import time

if len(sys.argv) < 6:
    sys.exit('usage: ab_env.py MODEL REPS N "[ENV=v ]binA" "[ENV=v ]binB"')

M = sys.argv[1]
reps = int(sys.argv[2])
n = int(sys.argv[3])
A, B = sys.argv[4], sys.argv[5]
if reps <= 0 or n <= 0:
    sys.exit("error: REPS and N must be positive integers")
if not re.match(r"^[a-zA-Z0-9_.-]+$", M):
    sys.exit(f"error: invalid model name: {M!r}")

SERIAL = os.environ.get("SERIAL")
ADB = ["adb", "-s", SERIAL] if SERIAL else ["adb"]
CERA = os.environ.get("CERA", "/data/local/tmp/cmp-cera")
AP_MAX_C = 36.0


def parse_variant(v):
    parts = v.split()
    if not parts:
        sys.exit(f"error: empty variant specification: {v!r}")
    env_parts, binary = parts[:-1], parts[-1]
    for e in env_parts:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=[a-zA-Z0-9_.,:-]+$", e):
            sys.exit(f"error: invalid env assignment {e!r} in variant {v!r}")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", binary):
        sys.exit(f"error: invalid binary name {binary!r} in variant {v!r}")
    prefix = (" ".join(env_parts) + " ") if env_parts else ""
    return f"{prefix}./{binary}"


CMD_A = parse_variant(A)
CMD_B = parse_variant(B)


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


def run(cmd_prefix, label):
    cool()
    cmd = ADB + [
        "shell",
        f"cd {shlex.quote(CERA)} && {cmd_prefix} bench -m /data/local/tmp/{shlex.quote(M)}.gguf --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1",
    ]
    o = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"prefill tok/s: p50=([0-9.]+)", o)
    if not m:
        sys.stderr.write(f"warning: benchmark failed for {label}:\n{o}\n")
        return 0.0
    return float(m.group(1))


def main():
    ra, rb = [], []
    for i in range(reps):
        if i % 2 == 0:
            ra.append(run(CMD_A, A))
            rb.append(run(CMD_B, B))
        else:
            rb.append(run(CMD_B, B))
            ra.append(run(CMD_A, A))
    med_a, med_b = st.median(ra), st.median(rb)
    print(f"{M} n={n} [{A}] p50 {med_a:.1f}", ra)
    print(f"{M} n={n} [{B}] p50 {med_b:.1f}", rb)
    if med_a > 0:
        print("change %+.1f%%" % ((med_b / med_a - 1) * 100))
    else:
        print("change undefined (baseline median is zero)")


if __name__ == "__main__":
    main()

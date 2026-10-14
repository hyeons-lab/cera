#!/usr/bin/env python3
"""Thermally gated CPU A/B of the tip of PR 505 (cera-base) against the top of the CPU stack (cera-top):
prefill and decode tok/s per model, alternating binaries, SERIAL=<adb serial>, usage: ab_stack.py OUT ROUNDS."""
import json, os, re, subprocess, sys, time
SERIAL = os.environ["SERIAL"]; OUT = sys.argv[1]; ROUNDS = int(sys.argv[2])
AP_MAX_C = 28.0
CERA = "/data/local/tmp/cmp-cera"
# (label, gguf, prompt tokens, max tokens, warmup, runs)
MODELS = [
    ("llama-3.2-1b-q4_0", "Llama-3.2-1B-Q4_0.gguf", 512, 48, 1, 3),
    ("lfm2.5-vl-450m-q4_0", "LFM2.5-VL-450M-Q4_0.gguf", 512, 64, 1, 3),
    ("lfm2.5-350m-q4_k_m", "LFM2.5-350M-Q4_K_M.gguf", 512, 64, 1, 3),
    ("lfm2.5-350m-q8_0", "LFM2.5-350M-Q8_0.gguf", 512, 64, 1, 3),
    ("lfm2.5-2.6b-agent-q4_0", "agent-q4_0.gguf", 512, 32, 1, 3),
    ("lfm2.5-2.6b-q5_k_m", "LFM2.5-2.6B-Q5_K_M.gguf", 128, 32, 0, 2),
]
def sh(cmd, timeout=900):
    r = subprocess.run(["adb", "-s", SERIAL, "shell", cmd], capture_output=True, text=True, errors="replace", timeout=timeout)
    return (r.stdout + r.stderr).replace("\r", "")
def ap():
    m = re.search(r"mValue=([\d.]+)", sh("dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"))
    return float(m.group(1)) if m else None
def cool():
    t0 = time.time()
    while True:
        t = ap()
        if (t is not None and t <= AP_MAX_C) or time.time() - t0 > 900: return t
        time.sleep(5)
def run(binary, gguf, pp, tg, warm, runs):
    out = sh(f"echo 0 > /proc/$$/oom_score_adj 2>/dev/null; cd {CERA} && timeout 600 ./{binary} bench -m /data/local/tmp/{gguf} --device cpu --runs {runs} --warmup {warm} --no-cache --prompt-tokens {pp} --max-tokens {tg} 2>&1")
    p = re.search(r"prefill tok/s: p50=([\d.]+)", out); d = re.search(r"decode tok/s: p50=([\d.]+)", out)
    return (float(p.group(1)) if p else None, float(d.group(1)) if d else None)
bins = ["cera-base", "cera-top"]
for rnd in range(ROUNDS):
    for label, gguf, pp, tg, warm, runs in MODELS:
        order = bins if rnd % 2 == 0 else bins[::-1]
        for b in order:
            t = cool()
            pf, dc = run(b, gguf, pp, tg, warm, runs)
            row = dict(model=label, round=rnd, binary=b, prefill=pf, decode=dc, ap_before=t, pp=pp, tg=tg)
            print(json.dumps(row), flush=True); open(OUT, "a").write(json.dumps(row) + "\n")

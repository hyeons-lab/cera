#!/usr/bin/env python3
"""Thermally gated CPU benchmark of Cera against llama.cpp: prefill tok/s, time to first token and decode tok/s at
several prompt lengths, alternating the two engines each round. SERIAL=<adb serial>; usage: bench_all.py OUT ROUNDS.
Cera's TTFT is `cera bench`'s `ttft` (prompt handed to the session until the first generated token). llama-bench has
no TTFT, so it is derived the same way, as prompt tokens / prompt-processing tok/s (llama.cpp samples the first token
from the prompt's last logits, so this is its time to first token up to one sampling step)."""
import csv, io, json, os, re, subprocess, sys, time
SERIAL = os.environ["SERIAL"]; OUT = sys.argv[1]; ROUNDS = int(sys.argv[2])
AP_MAX_C = 28.0
CERA = "/data/local/tmp/cmp-cera"; LL = "/data/local/tmp/cmp-llamacpp"
PPS = [16, 64, 512]; TG = 64
MODELS = [
    ("llama-3.2-1b-q4_0", "Llama-3.2-1B-Q4_0.gguf"),
    ("lfm2.5-vl-450m-q4_0", "LFM2.5-VL-450M-Q4_0.gguf"),
    ("lfm2.5-350m-q4_k_m", "LFM2.5-350M-Q4_K_M.gguf"),
    ("lfm2.5-350m-q8_0", "LFM2.5-350M-Q8_0.gguf"),
    ("lfm2.5-2.6b-agent-q4_0", "agent-q4_0.gguf"),
    ("lfm2.5-2.6b-q5_k_m", "LFM2.5-2.6B-Q5_K_M.gguf"),
]
def sh(cmd, timeout=1200):
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
def cera(gguf, pp):
    out = sh(f"echo 0 > /proc/$$/oom_score_adj 2>/dev/null; cd {CERA} && timeout 900 ./cera-ttft bench -m /data/local/tmp/{gguf} --device cpu --runs 3 --warmup 1 --no-cache --prompt-tokens {pp} --max-tokens {TG} 2>&1")
    g = lambda pat: (float(re.search(pat, out).group(1)) if re.search(pat, out) else None)
    return dict(prefill=g(r"prefill tok/s: p50=([\d.]+)"), decode=g(r"decode tok/s: p50=([\d.]+)"), ttft_ms=g(r"ttft ms: p50=([\d.]+)"))
def llama(gguf, pp):
    out = sh(f"echo 0 > /proc/$$/oom_score_adj 2>/dev/null; cd {LL} && export LD_LIBRARY_PATH=$PWD/lib && timeout 900 ./bin/llama-bench -m /data/local/tmp/{gguf} -dev none -ngl 0 -t 8 -p {pp} -n {TG} -r 3 -o csv 2>/dev/null")
    lines = [l for l in out.splitlines() if l.startswith("build_commit") or l.startswith('"')]
    res = dict(prefill=None, decode=None, ttft_ms=None)
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        if int(r["n_prompt"]) > 0 and int(r["n_gen"]) == 0: res["prefill"] = float(r["avg_ts"])
        if int(r["n_gen"]) > 0 and int(r["n_prompt"]) == 0: res["decode"] = float(r["avg_ts"])
    if res["prefill"]: res["ttft_ms"] = pp / res["prefill"] * 1e3
    return res
for rnd in range(ROUNDS):
    for label, gguf in MODELS:
        for pp in PPS:
            engines = [("cera", cera), ("llama.cpp", llama)]
            if (rnd + pp // 16) % 2: engines = engines[::-1]
            for name, fn in engines:
                t = cool()
                res = fn(gguf, pp)
                row = dict(model=label, round=rnd, engine=name, pp=pp, ap_before=t, **res)
                print(json.dumps(row), flush=True); open(OUT, "a").write(json.dumps(row) + "\n")

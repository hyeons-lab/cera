#!/usr/bin/env python3
"""Thermally gated GPU prefill A/B: the direct-B build, the previous PR's build (transposed B, two-row, fused
gate/up) and the build before that (both switches off), against llama-bench, at several prompt lengths."""
import csv, io, json, os, re, statistics as st, subprocess, sys, time
SERIAL = os.environ["SERIAL"]; OUT = sys.argv[1]; ROUNDS = int(sys.argv[2]); BIN = sys.argv[3]
LENS = [int(x) for x in sys.argv[4].split(",")]
AP_MAX_C = 28.0
MODEL = "/data/local/tmp/LFM2.5-VL-450M-Q4_0.gguf"; CERA = "/data/local/tmp/cmp-cera"; LL = "/data/local/tmp/cmp-llamacpp"
GUARD = "echo 0 > /proc/$$/oom_score_adj 2>/dev/null; "
def sh(cmd, timeout=400):
    r = subprocess.run(["adb","-s",SERIAL,"shell",cmd],capture_output=True,text=True,errors="replace",timeout=timeout)
    return (r.stdout+r.stderr).replace("\r","")
def ap():
    m = re.search(r"mValue=([\d.]+)", sh("dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'"))
    return float(m.group(1)) if m else None
def cool():
    t0=time.time()
    while True:
        t=ap()
        if (t is not None and t<=AP_MAX_C) or time.time()-t0>900: return t
        time.sleep(5)
VARS = {"direct": ("cera-direct", ""), "pr497": ("cera-r2", ""), "previous": ("cera-r2", "CERA_WGPU_FUSE_GATE_UP=0 CERA_WGPU_GEMM_R1=1 ")}
def cera(var, n):
    BIN, env = VARS[var]
    out = sh(f"{GUARD}cd {CERA} && {env}timeout 300 ./{BIN} bench -m {MODEL} --device gpu --runs 5 --warmup 2 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1")
    m = re.search(r"prefill tok/s: p50=([\d.]+)", out)
    return float(m.group(1)) if m else None
def llama(n):
    out = sh(f"{GUARD}cd {LL} && export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib && timeout 300 ./bin/llama-bench -m {MODEL} -dev GPUOpenCL -ngl 99 -t 8 -p {n} -n 0 -r 5 -o csv 2>/dev/null")
    lines=[l for l in out.splitlines() if l.startswith("build_commit") or l.startswith('"')]
    for r in csv.DictReader(io.StringIO("\n".join(lines))): return float(r["avg_ts"])
variants = ["direct","pr497","previous","llama"]
for n in LENS:
    for rnd in range(ROUNDS):
        order = variants[rnd%4:]+variants[:rnd%4]
        for v in order:
            t=cool()
            val = llama(n) if v=="llama" else cera(v,n)
            row=dict(len=n,round=rnd,variant=v,tok_s=val,ap_before=t)
            print(json.dumps(row),flush=True); open(OUT,"a").write(json.dumps(row)+"\n")

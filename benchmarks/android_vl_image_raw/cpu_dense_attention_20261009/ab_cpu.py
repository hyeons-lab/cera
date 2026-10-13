#!/usr/bin/env python3
"""Thermally gated CPU prefill A/B on the dense Llama-3.2-1B Q4_0: previous attention kernel (cera-attn2, CERA_CPU_ATTN_TILED=0), the tiled kernel
over whole heads (cera-attn2, PR 505) and the tiled kernel over (head, 32-query) work items (cera-dense), against llama-bench CPU."""
import argparse, csv, io, json, os, re, subprocess, sys, time

AP_MAX_C = 28.0
MODEL = "/data/local/tmp/Llama-3.2-1B-Q4_0.gguf"
CERA = "/data/local/tmp/cmp-cera"
LL = "/data/local/tmp/cmp-llamacpp"
GUARD = "echo 0 > /proc/$$/oom_score_adj 2>/dev/null; "
VARS = {"items": ("cera-dense", ""), "heads": ("cera-attn2", ""), "old": ("cera-attn2", "CERA_CPU_ATTN_TILED=0 ")}

def sh(cmd, serial, timeout=400):
    r = subprocess.run(["adb", "-s", serial, "shell", cmd], capture_output=True, text=True, errors="replace", timeout=timeout)
    return (r.stdout + r.stderr).replace("\r", "")

def ap(serial):
    m = re.search(r"mValue=([\d.]+)", sh("dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'", serial))
    return float(m.group(1)) if m else None

def cool(serial):
    t0 = time.time()
    while True:
        t = ap(serial)
        if (t is not None and t <= AP_MAX_C) or time.time() - t0 > 900:
            return t
        time.sleep(5)

def cera(var, n, serial):
    bin_name, env = VARS[var]
    out = sh(f"{GUARD}cd {CERA} && {env}timeout 300 ./{bin_name} bench -m {MODEL} --device cpu --runs 5 --warmup 2 --no-cache --prompt-tokens {n} --max-tokens 0 2>&1", serial)
    m = re.search(r"prefill tok/s: p50=([\d.]+)", out)
    return float(m.group(1)) if m else None

def llama(n, serial):
    out = sh(f"{GUARD}cd {LL} && export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib && timeout 300 ./bin/llama-bench -m {MODEL} -dev none -ngl 0 -t 8 -p {n} -n 0 -r 5 -o csv 2>/dev/null", serial)
    lines = [l for l in out.splitlines() if l.startswith("build_commit") or l.startswith('"')]
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        return float(r["avg_ts"])
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Thermally gated CPU prefill A/B on the dense Llama-3.2-1B Q4_0."
    )
    parser.add_argument("out", help="Output JSONL file path")
    parser.add_argument("rounds", type=int, help="Number of benchmark rounds")
    parser.add_argument("lens", help="Comma-separated prompt token lengths")
    parser.add_argument(
        "--serial",
        default=os.environ.get("SERIAL"),
        help="ADB device serial (defaults to SERIAL environment variable)",
    )
    args = parser.parse_args()
    if not args.serial:
        parser.error("device serial must be specified via --serial or SERIAL environment variable")

    out_path = args.out
    rounds = args.rounds
    lens = [int(x) for x in args.lens.split(",")]

    variants = ["items", "heads", "old", "llama"]
    for n in lens:
        for rnd in range(rounds):
            order = variants[rnd % 4:] + variants[:rnd % 4]
            for v in order:
                t = cool(args.serial)
                val = llama(n, args.serial) if v == "llama" else cera(v, n, args.serial)
                row = dict(len=n, round=rnd, variant=v, tok_s=val, ap_before=t)
                print(json.dumps(row), flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()

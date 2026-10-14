#!/usr/bin/env python3
"""Thermally gated text-only prefill (pp512) and decode (tg128): cera bench vs llama-bench, same GGUF.

Before every invocation the harness waits until the SoC (AP sensor) is back under AP_MAX_C, so each run starts
from the same cool state; the engine order alternates between rounds to cancel order bias; the AP temperature
before and after, and the prime core's clock during the run (sampled on the device every 0.5 s), are recorded
with each value. Usage: SERIAL=<adb serial> cool_pp_tg.py <out.jsonl> [rounds]
"""
import csv, io, json, os, re, statistics as st, subprocess, sys, time

SERIAL = os.environ.get("SERIAL") or sys.exit("set SERIAL")
OUT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
AP_MAX_C = float(os.environ.get("AP_MAX_C", "28"))
MODEL = "/data/local/tmp/LFM2.5-VL-450M-Q4_0.gguf"
CERA = "/data/local/tmp/cmp-cera"
LL = "/data/local/tmp/cmp-llamacpp"
GUARD = "echo 0 > /proc/$$/oom_score_adj 2>/dev/null; "
ADSP = f"ADSP_LIBRARY_PATH='{CERA}/skels;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/vendor/dsp' "
FREQ = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq"


def sh(cmd, timeout=300):
    r = subprocess.run(["adb", "-s", SERIAL, "shell", cmd], capture_output=True, text=True, errors="replace", timeout=timeout)
    return (r.stdout + r.stderr).replace("\r", "")


def ap_temp():
    out = sh("dumpsys thermalservice | awk '/Current temperatures from HAL/,/Current cooling devices/' | grep 'mName=AP,'")
    m = re.search(r"mValue=([\d.]+)", out)
    return float(m.group(1)) if m else None


def cool_down():
    t0 = time.time()
    while True:
        t = ap_temp()
        if t is not None and t <= AP_MAX_C:
            return t, round(time.time() - t0)
        if time.time() - t0 > 900:
            return t, round(time.time() - t0)
        time.sleep(5)


def measured(cmd):
    """Run cmd on the device while sampling the prime core's clock; returns (output, freq_samples_mhz)."""
    wrapped = (f"rm -f /data/local/tmp/fs.txt; (while :; do cat {FREQ} >> /data/local/tmp/fs.txt; sleep 0.5; done) & P=$!; "
               f"{cmd}; kill $P 2>/dev/null; echo @@FREQ; cat /data/local/tmp/fs.txt")
    out = sh(wrapped, timeout=600)
    body, _, fs = out.partition("@@FREQ")
    return body, [int(x) // 1000 for x in fs.split() if x.isdigit()]


def cera_cmd(dev, ptok, mtok):
    env = ADSP if dev == "hexagon" else ""
    return (f"{GUARD}cd {CERA} && {env}timeout 250 ./cera bench -m {MODEL} --device {dev} --runs 5 --warmup 2 --no-cache "
            f"--prompt-tokens {ptok} --max-tokens {mtok} 2>&1")


def llama_cmd(dev):
    la = {"cpu": "-dev none -ngl 0", "gpu": "-dev GPUOpenCL -ngl 99", "hexagon": "-dev HTP0 -ngl 99"}[dev]
    return (f"{GUARD}cd {LL} && export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib && "
            f"timeout 300 ./bin/llama-bench -m {MODEL} {la} -t 8 -p 512 -n 128 -r 5 -o csv 2>/dev/null")


def parse_llama(body):
    lines = [l for l in body.splitlines() if l.startswith("build_commit") or l.startswith('"')]
    res = {}
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        k = "pp512" if r["n_prompt"] != "0" else "tg128"
        res[k] = float(r["avg_ts"])
    return res


def run_one(engine, dev, rnd, metric, cmd, parse):
    t_before, waited = cool_down()
    body, freqs = measured(cmd)
    t_after = ap_temp()
    vals = parse(body)
    head = re.search(r"headroom before warmup: ([\d.]+)", body)
    for k, v in vals.items():
        row = dict(engine=engine, dev=dev, round=rnd, metric=k, value=v, ap_before=t_before, ap_after=t_after, waited_s=waited,
                   headroom=float(head.group(1)) if head else None,
                   mhz_median=st.median(freqs) if freqs else None, mhz_min=min(freqs) if freqs else None,
                   mhz_max=max(freqs) if freqs else None)
        print(json.dumps(row), flush=True)
        open(OUT, "a").write(json.dumps(row) + "\n")


def cera_parse(kind):
    def p(body):
        m = re.search(rf"{kind} tok/s: p50=([\d.]+)", body)
        return {("pp512" if kind == "prefill" else "tg128"): float(m.group(1))} if m else {}
    return p


def main():
    lvl = sh("dumpsys battery | grep -m1 level")
    print("start", lvl.strip(), "AP", ap_temp(), flush=True)
    for dev in ("cpu", "gpu", "hexagon"):
        for rnd in range(ROUNDS):
            order = ["cera", "llama"] if rnd % 2 == 0 else ["llama", "cera"]
            for eng in order:
                if eng == "cera":
                    run_one("cera", dev, rnd, "pp512", cera_cmd(dev, 512, 0), cera_parse("prefill"))
                    run_one("cera", dev, rnd, "tg128", cera_cmd(dev, 128, 128), cera_parse("decode"))
                else:
                    run_one("llama", dev, rnd, "both", llama_cmd(dev), parse_llama)


main()

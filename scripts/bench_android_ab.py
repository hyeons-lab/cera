#!/usr/bin/env python3
"""Paired, interleaved, rotating-order A/B on a phone, with thermal annotation.

usage: bench_android_ab.py '<json [[label, kind, binary_or_empty, env, extra_args], ...]>' IMAGE ROUNDS [MAXTOK]
  kind: "cera" (a binary under /data/local/tmp/cmp-cera, run on CPU) or "llama" (llama-mtmd-cli, CPU, -t 8)
  env and extra_args are spliced into the command line (e.g. "CERA_Q4_DEC4=0", "--kv-cache-keys f32").
  The first variant is the baseline the ratios are taken against.
SERIAL=<adb-serial> selects the device. The Cera binary named THERMAL_BIN must be a build that has `cera thermal`.

Why paired: CPU decode flips between a fast and a slow clock regime for seconds to minutes with no change in
the workload, so a before/after pair can show anything from a 2% to a 25% gap. Variants are interleaved a few
seconds apart, in rotating order, and compared per round.

After every run it records skin temperature (dumpsys thermalservice) and the Android thermal headroom
(`cera thermal`; 0 cool, 1.0 throttling). A round is "matched" if its skin spread is <= 0.4 C and its headroom
spread is <= 0.06; both all-round and matched-round medians are printed.

TTFT is everything before the first decoded token for both engines: Cera's `Image prefill` line, and for
llama.cpp the vision tower (`mtmd batch encoding`) plus the prompt eval.
"""
import json, os, re, statistics, subprocess, sys

SERIAL = os.environ.get("SERIAL", "")
ADB = [os.environ.get("ADB", "adb"), "-s", SERIAL]
MANIFEST = "/data/local/tmp/LFM2.5-VL-450M-Q4_0.json"
GGUF = "/data/local/tmp/LFM2.5-VL-450M-Q4_0.gguf"
MMPROJ = "/data/local/tmp/mmproj-LFM2.5-VL-450M-Q8_0.gguf"
PROMPT = "Describe this image in detail."
OOM = "echo 0 > /proc/$$/oom_score_adj 2>/dev/null; "
THERMAL_BIN = os.environ.get("THERMAL_BIN", "/data/local/tmp/cmp-cera/cera")


def sh(cmd, timeout=240):
    r = subprocess.run(ADB + ["shell", cmd], capture_output=True, text=True, errors="replace", timeout=timeout)
    return (r.stdout + r.stderr).replace("\r", "")


def thermal():
    # A hung dumpsys must not abort the whole run: the reading is just missing.
    try:
        skin = re.search(r"mValue=([\d.]+)", sh("dumpsys thermalservice 2>/dev/null | grep -m1 'mName=SKIN'", timeout=30))
        hr = re.search(r"now ([\d.]+)", sh(f"cd /data/local/tmp/cmp-cera && {THERMAL_BIN} thermal 2>&1", timeout=30))
    except subprocess.TimeoutExpired:
        return (None, None)
    return (float(skin.group(1)) if skin else None, float(hr.group(1)) if hr else None)


def cmd(kind, binary, env, extra, image, maxtok):
    if kind == "cera":
        return (f"{OOM}cd /data/local/tmp/cmp-cera && {env} timeout 250 ./{binary} run -m {MANIFEST} --image {image} "
                f"-p '{PROMPT}' --max-tokens {maxtok} --temperature 0 --device cpu {extra} 2>&1")
    return (f"{OOM}cd /data/local/tmp/cmp-llamacpp && export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib && "
            f"timeout 250 ./bin/llama-mtmd-cli -m {GGUF} --mmproj {MMPROJ} --image {image} -p '{PROMPT}' -n {maxtok} --temp 0 "
            f"-dev none -mmdev none -ngl 0 -t 8 {extra} --perf -lv 4 2>&1")


def parse(kind, text):
    body = [l for l in text.splitlines() if l.startswith(("A ", "The ", "This "))]
    if kind == "cera":
        d = re.search(r"Decode: ([\d.]+) tok/s", text)
        m = re.search(r"Image prefill: (\d+) KV tokens, ([\d.]+) ms", text)
        if not (d and m):
            return None
        return dict(dec=float(d.group(1)), ctx=int(m.group(1)), ttft=float(m.group(2)), text=body[0][:200] if body else "")
    ev = re.search(r"perf_context_print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second", text)
    pe = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", text)
    tower = sum(int(x) for x in re.findall(r"mtmd batch encoding done in (\d+) ms", text))
    if not (ev and pe and tower):
        return None
    return dict(dec=float(ev.group(3)), ctx=int(pe.group(2)), ttft=tower + float(pe.group(1)), text=body[0][:200] if body else "")


def main():
    if not SERIAL:
        sys.exit("set SERIAL to the adb serial of the device")
    variants = json.loads(sys.argv[1])
    image, rounds = sys.argv[2], int(sys.argv[3])
    maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    lvl = re.search(r"level:\s*(\d+)", sh("dumpsys battery"))
    print(f"battery {lvl.group(1) if lvl else '?'}%  thermal(skin,headroom) {thermal()}", flush=True)
    res = {v[0]: [] for v in variants}
    texts = {}
    per_round = []
    n = len(variants)
    for r in range(rounds):
        order = [variants[(i + r) % n] for i in range(n)]
        row = {}
        for label, kind, binary, env, extra in order:
            try:
                out = sh(cmd(kind, binary, env, extra, image, maxtok), timeout=300)
            except subprocess.TimeoutExpired:
                out = "TIMEOUT"
            p = parse(kind, out)
            sk, hr = thermal()
            if p is None:
                print(f"r{r} {label}: FAILED  {' | '.join(out.strip().splitlines()[-2:])[:200]}", flush=True)
                continue
            p.update(skin=sk, hr=hr)
            res[label].append(p)
            row[label] = p
            texts.setdefault(label, set()).add(p["text"])
            print(f"r{r} {label}: ctx {p['ctx']} decode {p['dec']:.1f} tok/s  skin {sk} headroom {hr}", flush=True)
        per_round.append(row)

    def matched(row):
        sk = [p["skin"] for p in row.values() if p["skin"] is not None]
        hr = [p["hr"] for p in row.values() if p["hr"] is not None]
        return len(row) == n and (not sk or max(sk) - min(sk) <= 0.4) and (not hr or max(hr) - min(hr) <= 0.06)

    mrounds = [row for row in per_round if matched(row)]
    print(f"\n{len(mrounds)}/{rounds} rounds thermally matched")
    base = variants[0][0]
    for name, rows in (("all rounds", [r for r in per_round if len(r) == n]), ("matched rounds", mrounds)):
        if not rows:
            continue
        print(f"-- {name} ({len(rows)})")
        for label, *_ in variants:
            d = [row[label]["dec"] for row in rows]
            t = [row[label]["ttft"] for row in rows]
            ratio = [row[label]["dec"] / row[base]["dec"] for row in rows]
            print(f"  {label:28s} decode median {statistics.median(d):7.1f}  (min {min(d):.1f} max {max(d):.1f})  "
                  f"vs {base}: x{statistics.median(ratio):.3f}   ttft median {statistics.median(t):.0f} ms")
    allp = [p for v in res.values() for p in v]
    skins = [p["skin"] for p in allp if p["skin"]]
    hrs = [p["hr"] for p in allp if p["hr"] is not None]
    parts = []
    if skins:
        parts.append(f"skin range {min(skins)}-{max(skins)} C")
    if hrs:
        parts.append(f"headroom {min(hrs)}-{max(hrs)}")
    if parts:
        print(", ".join(parts))
    for label, t in texts.items():
        print(f"text[{label}]: {len(t)} distinct; first: {sorted(t)[0][:90]}")


if __name__ == "__main__":
    main()

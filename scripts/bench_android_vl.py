#!/usr/bin/env python3
"""On-device image benchmark: Cera (CPU/GPU/NPU) vs llama.cpp (CPU/GPU/NPU) on a vision-language model.

Same model, mmproj, image, prompt, greedy decoding and token cap for every cell. Methodology follows
scripts/bench_android.sh: pin threads and keep each engine's best CPU placement, discard warm passes so
the SoC is at thermal equilibrium, interleave the cells in every pass (thermal drift hits all engines
alike), keep the raw output of every run, and refuse to run on a low battery. Results and analysis:
benchmarks/ANDROID_VL_IMAGE.md.

Prerequisites (on the device, paths overridable below)
  $CERA_DIR/cera                      cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu,hexagon
  $CERA_DIR/skels/libggml-htp-v*.so   cera/src/backend/hexagon/skels/ (Cera's own DSP skels, never llama.cpp's)
  $LLAMA_DIR/bin/llama-mtmd-cli, $LLAMA_DIR/lib   a llama.cpp Snapdragon build (OpenCL + Hexagon), e.g. the
                                      llama-cpp-android-arm64-snapdragon CI artifact of ggml-org/llama.cpp
  the GGUF, the mmproj GGUF, a Leap-style manifest JSON naming both, and the image

Usage
  python3 scripts/bench_android_vl.py <out-dir> sweep                         # CPU thread/mask sweep
  python3 scripts/bench_android_vl.py <out-dir> matrix <cera-mask> <llama-mask> <llama-threads>
  env: SERIAL (adb serial), IMAGE (device path), CERA_DIR, LLAMA_DIR

Metrics (per run):
  ttft_ms   everything before the first decode step
            cera:      "Image prefill" (preprocess + vision tower + image prefill + text prefill)
            llama.cpp: sum of "mtmd batch encoding done" (vision tower) + "prompt eval time"
                       (image + text token decode); jpeg decode/resize (a few ms) is not included
  tower_ms  vision tower only
  decode    decode tokens per second as each engine reports it
  e2e_ms    ttft_ms + 63 / decode * 1000  (time to the 64th token)
"""
import json, os, re, statistics, subprocess, sys, time

SERIAL = os.environ.get("SERIAL") or sys.exit("set SERIAL to the adb serial of the device")
ADB = ["adb", "-s", SERIAL]
OUT = sys.argv[1]
MODE = sys.argv[2]  # sweep | matrix
IMAGE = os.environ.get("IMAGE", "/data/local/tmp/cmp-pug_512.jpg")
CERA_DIR = os.environ.get("CERA_DIR", "/data/local/tmp/cmp-cera")
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/data/local/tmp/cmp-llamacpp")
PROMPT = "Describe this image in detail."
MAXTOK = 64
MODEL_GGUF = "/data/local/tmp/LFM2.5-VL-450M-Q4_0.gguf"
MMPROJ = "/data/local/tmp/mmproj-LFM2.5-VL-450M-Q8_0.gguf"
MANIFEST = "/data/local/tmp/LFM2.5-VL-450M-Q4_0.json"
OOM_GUARD = "echo 0 > /proc/$$/oom_score_adj 2>/dev/null; "
CERA_ADSP = "ADSP_LIBRARY_PATH='" + CERA_DIR + "/skels;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/vendor/dsp' "
MASKS = {"all8": ("ff", 8), "mid6": ("3f", 6), "prime2": ("c0", 2)}
MIN_BATTERY = 30
os.makedirs(OUT, exist_ok=True)


def adb_shell(cmd, timeout=240):
    r = subprocess.run(ADB + ["shell", cmd], capture_output=True, text=True, errors="replace", timeout=timeout)
    return (r.stdout + r.stderr).replace("\r", ""), r.returncode


def battery():
    out, _ = adb_shell("dumpsys battery")
    lvl = re.search(r"^\s*level:\s*(\d+)", out, re.M)
    tmp = re.search(r"^\s*temperature:\s*(\d+)", out, re.M)
    return (int(lvl.group(1)) if lvl else -1, int(tmp.group(1)) / 10 if tmp else None)


def skin_temp():
    out, _ = adb_shell("dumpsys thermalservice 2>/dev/null | grep -m1 'mName=SKIN'")
    m = re.search(r"mValue=([\d.]+)", out)
    return float(m.group(1)) if m else None


def cmd_cera(dev, mask):
    pin = f"taskset {mask} " if mask else ""
    env = CERA_ADSP if dev == "hexagon" else ""
    return (f"{OOM_GUARD}cd {CERA_DIR} && {env}{pin}timeout 150 ./cera run -m {MANIFEST} "
            f"--image {IMAGE} -p '{PROMPT}' --max-tokens {MAXTOK} --temperature 0 --device {dev} 2>&1")


def cmd_llama(devargs, mask, threads):
    pin = f"taskset {mask} " if mask else ""
    return (f"{OOM_GUARD}cd {LLAMA_DIR} && export LD_LIBRARY_PATH=$PWD/lib "
            f"ADSP_LIBRARY_PATH=$PWD/lib && {pin}timeout 150 ./bin/llama-mtmd-cli -m {MODEL_GGUF} --mmproj {MMPROJ} "
            f"--image {IMAGE} -p '{PROMPT}' -n {MAXTOK} --temp 0 {devargs} -t {threads} --perf -lv 4 2>&1")


def parse_cera(text):
    m = re.search(r"Image prefill: (\d+) KV tokens, ([\d.]+) ms", text)
    d = re.search(r"Decode: ([\d.]+) tok/s", text)
    g = re.search(r"Generated tokens: (\d+)", text)
    if not (m and d):
        return None
    t = re.search(r"tower ([\d.]+) ms", text)
    body = [l for l in text.splitlines() if l.startswith("A ") or l.startswith("The ") or l.startswith("This ")]
    return dict(tokens=int(m.group(1)), ttft_ms=float(m.group(2)), tower_ms=float(t.group(1)) if t else None,
                decode=float(d.group(1)), gen=int(g.group(1)) if g else None, text=(body[0][:60] if body else ""))


def parse_llama(text):
    enc = [int(x) for x in re.findall(r"mtmd batch encoding done in (\d+) ms", text)]
    pe = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", text)
    ev = re.search(r"perf_context_print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second", text)
    if not (enc and pe and ev):
        return None
    body = [l for l in text.splitlines() if l.startswith("A ") or l.startswith("The ") or l.startswith("This ")]
    return dict(tokens=int(pe.group(2)), ttft_ms=sum(enc) + float(pe.group(1)), tower_ms=float(sum(enc)),
                decode=float(ev.group(3)), gen=int(ev.group(2)) + 1, text=(body[0][:60] if body else ""))


def run_cell(name, cmd, parser, tag):
    t0 = time.time()
    try:
        out, rc = adb_shell(cmd, timeout=200)
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", 124
    with open(f"{OUT}/{tag}-{name}.log", "w") as f:
        f.write(out)
    r = parser(out)
    if r is None:
        tail = " | ".join([l for l in out.strip().splitlines() if l.strip()][-2:])[:200]
        return dict(cell=name, ok=False, rc=rc, error=tail, wall=round(time.time() - t0, 1))
    r["e2e_ms"] = r["ttft_ms"] + (max(r["gen"] or MAXTOK, 2) - 1) / r["decode"] * 1000
    r.update(cell=name, ok=True, wall=round(time.time() - t0, 1))
    return r


def med(xs):
    return statistics.median(xs) if xs else None


def check_battery():
    lvl, _ = battery()
    if lvl < MIN_BATTERY:
        sys.exit(f"battery {lvl}% < {MIN_BATTERY}%: refusing to measure at a reduced power budget")
    return lvl


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["cell"], []).append(r)
    s = {}
    for c, rs in by.items():
        ok = [r for r in rs if r["ok"]]
        s[c] = dict(n_ok=len(ok), n=len(rs),
                    tokens=sorted({r["tokens"] for r in ok}),
                    ttft_ms=med([r["ttft_ms"] for r in ok]), tower_ms=med([r["tower_ms"] for r in ok if r["tower_ms"] is not None]),
                    decode=med([r["decode"] for r in ok]), e2e_ms=med([r["e2e_ms"] for r in ok]),
                    ttft_min=min([r["ttft_ms"] for r in ok], default=None), ttft_max=max([r["ttft_ms"] for r in ok], default=None),
                    dec_min=min([r["decode"] for r in ok], default=None), dec_max=max([r["decode"] for r in ok], default=None),
                    text=(ok[0]["text"] if ok else ""), errors=[r["error"] for r in rs if not r["ok"]][:2])
    return s


def sweep():
    reps, warm = 4, 1
    rows = []
    lvl = check_battery()
    print(f"battery {lvl}%  skin {skin_temp()}C", flush=True)
    for eng in ("cera", "llama"):
        for mname, (mask, nthr) in MASKS.items():
            cell = f"{eng}-cpu@{mname}"
            if eng == "cera":
                cmd, parser = cmd_cera("cpu", mask), parse_cera
            else:
                cmd, parser = cmd_llama("-dev none -mmdev none -ngl 0", mask, nthr), parse_llama
            for i in range(warm + reps):
                r = run_cell(cell, cmd, parser, f"sweep-{i}")
                if i >= warm:
                    rows.append(r)
            print(f"{cell}: " + (f"ttft {med([r['ttft_ms'] for r in rows if r['cell']==cell and r['ok']]):.0f} ms" if any(r['ok'] and r['cell']==cell for r in rows) else "FAILED"), flush=True)
    json.dump(dict(rows=rows, summary=summarize(rows)), open(f"{OUT}/sweep.json", "w"), indent=1)


def matrix(cera_mask, llama_mask, llama_thr):
    warm, passes = 2, 5
    cells = [
        ("cera-cpu", cmd_cera("cpu", cera_mask), parse_cera),
        ("llama-cpu", cmd_llama("-dev none -mmdev none -ngl 0", llama_mask, llama_thr), parse_llama),
        ("cera-gpu", cmd_cera("gpu", ""), parse_cera),
        ("llama-gpu", cmd_llama("-dev GPUOpenCL -mmdev GPUOpenCL -ngl 99", "", 8), parse_llama),
        ("cera-npu", cmd_cera("hexagon", ""), parse_cera),
        ("llama-npu", cmd_llama("-dev HTP0 -mmdev HTP0 -ngl 99", "", 8), parse_llama),
    ]
    rows, trace = [], []
    lvl = check_battery()
    trace.append(dict(at="start", battery=lvl, skin=skin_temp()))
    print(f"start battery {lvl}% skin {trace[0]['skin']}C", flush=True)
    status, error = "complete", None
    try:
        for p in range(warm + passes):
            for name, cmd, parser in cells:
                r = run_cell(name, cmd, parser, f"pass{p}")
                r["pass"] = p
                r["warm"] = p < warm
                rows.append(r)
                print(f"pass {p}{' (warm)' if p < warm else ''} {name}: " +
                      (f"ttft {r['ttft_ms']:.0f} ms decode {r['decode']:.1f} tok/s tokens {r['tokens']}" if r["ok"] else f"FAILED rc={r['rc']} {r['error']}"), flush=True)
            trace.append(dict(at=f"after-pass-{p}", battery=battery()[0], skin=skin_temp()))
            # Refuse to START another pass at a reduced power budget, but never throw away a finished run:
            # the check after the last pass would exit before the results below are written.
            if p + 1 < warm + passes:
                lvl = check_battery()
    except BaseException as e:  # recorded in the JSON, then re-raised (SystemExit from the battery guard too)
        status, error = "aborted", f"{type(e).__name__}: {e}"
        raise
    finally:
        # Written even when the battery guard (or a crash) ends the run early, so finished passes are kept;
        # `status` tells a reader whether they are looking at a complete run.
        meas = [r for r in rows if not r["warm"]]
        json.dump(dict(rows=rows, summary=summarize(meas), trace=trace,
                       status=status, error=error, passes_done=sum(1 for q in {r['pass'] for r in rows} if sum(r['pass'] == q for r in rows) == len(cells)),
                       config=dict(image=IMAGE, prompt=PROMPT, max_tokens=MAXTOK, cera_mask=cera_mask, llama_mask=llama_mask, llama_threads=llama_thr)),
                  open(f"{OUT}/matrix.json", "w"), indent=1)


if __name__ == "__main__":
    if MODE == "sweep":
        sweep()
    else:
        matrix(sys.argv[3], sys.argv[4], int(sys.argv[5]))

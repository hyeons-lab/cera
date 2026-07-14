#!/usr/bin/env bash
# Android benchmark harness: cera (CPU + wgpu/Vulkan) vs llama.cpp, on-device.
#
# Companion to scripts/bench_matrix.sh (which is the Mac/desktop equivalent).
# Emits one CSV row per config plus the raw stdout of every run, because the
# parsed medians hide the per-run variance that matters on a phone (scheduler
# migration and thermal drift both show up as a bimodal run distribution, not as
# a shifted median).
#
# Thread pinning is not a micro-optimization here: cera's RowPool and llama's
# threadpool land very differently on big.LITTLE, so a benchmark that doesn't
# pin is measuring the kernel scheduler as much as the engine. Each engine is
# therefore run across several taskset masks and the best config is what should
# be compared.
#
# Usage:
#   scripts/bench_android.sh --model <name.gguf> [--serial <adb-serial>]
#                            [--llama-bench <path-on-device>]
#                            [--prompt 512] [--decode 128] [--runs 5]
#
# The model must already be on the device at $DEVICE_DIR/<name.gguf>, and the
# cera binary is pushed from target/aarch64-linux-android/release/cera (build
# with: cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu).
set -euo pipefail

DEVICE_DIR="/data/local/tmp/cera-bench"
MODEL=""
SERIAL="${CERA_ANDROID_SERIAL:-}"
LLAMA_BENCH=""
PROMPT=512
DECODE=128
RUNS=5
WARMUP=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --serial)      SERIAL="$2"; shift 2 ;;
    --llama-bench) LLAMA_BENCH="$2"; shift 2 ;;
    --prompt)      PROMPT="$2"; shift 2 ;;
    --decode)      DECODE="$2"; shift 2 ;;
    --runs)        RUNS="$2"; shift 2 ;;
    --warmup)      WARMUP="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model <name.gguf> is required" >&2; exit 2; }

# More than one device (phone + emulator) is the common case on a dev box, and
# adb errors out rather than picking one — so require an explicit serial.
if [[ -z "$SERIAL" ]]; then
  n=$(adb devices | grep -cE "[[:space:]]device$" || true)
  if [[ "$n" -ne 1 ]]; then
    echo "error: $n adb devices attached — pass --serial (or set CERA_ANDROID_SERIAL)." >&2
    adb devices >&2
    exit 2
  fi
fi
ADB=(adb)
[[ -n "$SERIAL" ]] && ADB=(adb -s "$SERIAL")

BIN_LOCAL="target/aarch64-linux-android/release/cera"
[[ -f "$BIN_LOCAL" ]] || { echo "missing $BIN_LOCAL — build it first (see header)" >&2; exit 2; }

OUT="bench_android.csv"
LOG="bench_android_raw.log"
echo "engine,backend,config,prefill_p50,decode_p50,prefill_stddev,decode_stddev" > "$OUT"
: > "$LOG"

echo "==> pushing cera to $DEVICE_DIR"
"${ADB[@]}" shell "mkdir -p $DEVICE_DIR"
"${ADB[@]}" push "$BIN_LOCAL" "$DEVICE_DIR/cera" >/dev/null
"${ADB[@]}" shell "chmod +x $DEVICE_DIR/cera"

# taskset masks for this SoC class (Tensor G5 / typical big.LITTLE):
#   (none) = let the scheduler place threads (cera's RowPool sizes itself)
#   80     = prime core only (cpu7)
#   fc     = perf + prime (cpu2-7)
#   7c     = perf cluster only (cpu2-6)
run_cera() {
  local backend="$1" label="$2" mask="$3"
  local pin=""; [[ -n "$mask" ]] && pin="taskset $mask "
  local cmd="cd $DEVICE_DIR && ${pin}./cera bench -m $MODEL --device $backend \
--prompt-tokens $PROMPT --max-tokens $DECODE --runs $RUNS --warmup $WARMUP --no-cache --gpu-io"

  echo "=== cera $backend [$label] ===" | tee -a "$LOG"
  local out
  if ! out=$("${ADB[@]}" shell "$cmd" 2>&1); then
    echo "$out" >> "$LOG"
    echo "cera,$backend,$label,FAIL,FAIL,FAIL,FAIL" >> "$OUT"
    return
  fi
  echo "$out" >> "$LOG"

  local pre dec
  pre=$(grep -E "^prefill tok/s:" <<<"$out" | head -1)
  dec=$(grep -E "^decode tok/s:"  <<<"$out" | head -1)
  local p50 d50 psd dsd
  p50=$(sed -n 's/.*p50=\([0-9.]*\).*/\1/p' <<<"$pre")
  d50=$(sed -n 's/.*p50=\([0-9.]*\).*/\1/p' <<<"$dec")
  psd=$(sed -n 's/.*stddev=\([0-9.]*\).*/\1/p' <<<"$pre")
  dsd=$(sed -n 's/.*stddev=\([0-9.]*\).*/\1/p' <<<"$dec")
  echo "cera,$backend,$label,$p50,$d50,$psd,$dsd" >> "$OUT"
  echo "  -> prefill p50=$p50 decode p50=$d50" | tee -a "$LOG"
  # The --gpu-io line is the one that says whether a GPU change actually removed
  # round-trips; keep it in the raw log where it can't be lost to CSV flattening.
  grep -E "^gpu I/O" <<<"$out" | tee -a "$LOG" || true
}

run_cera cpu "default-rowpool" ""
run_cera cpu "pin-prime-80"    "80"
run_cera cpu "pin-perf-7c"     "7c"
run_cera gpu "wgpu-vulkan"     ""

# llama.cpp reference, if a llama-bench is present on the device. Uses the same
# model file so the comparison is same-quant, same-weights.
if [[ -n "$LLAMA_BENCH" ]]; then
  rt=$(dirname "$LLAMA_BENCH")
  for cfg in "1:80" "5:7c" "6:fc"; do
    t="${cfg%%:*}"; mask="${cfg##*:}"
    echo "=== llama-bench -t $t (taskset $mask) ===" | tee -a "$LOG"
    out=$("${ADB[@]}" shell "cd $rt && LD_LIBRARY_PATH=. taskset $mask ./$(basename "$LLAMA_BENCH") \
-m $DEVICE_DIR/$MODEL -t $t -p $PROMPT -n $DECODE -r $RUNS -o md" 2>&1) || true
    echo "$out" >> "$LOG"
    # llama-bench md rows: | model | size | params | backend | threads | test | t/s |
    pp=$(grep -E "\|[[:space:]]*pp$PROMPT[[:space:]]*\|" <<<"$out" | sed -n "s/.*|[[:space:]]*\\([0-9.]*\\) ±.*/\\1/p" | head -1)
    tg=$(grep -E "\|[[:space:]]*tg$DECODE[[:space:]]*\|" <<<"$out" | sed -n "s/.*|[[:space:]]*\\([0-9.]*\\) ±.*/\\1/p" | head -1)
    echo "llama.cpp,cpu,t$t-$mask,${pp:-NA},${tg:-NA},NA,NA" >> "$OUT"
    echo "  -> pp=$pp tg=$tg" | tee -a "$LOG"
  done
fi

echo
echo "==> $OUT"
column -t -s, "$OUT"
echo "==> raw runs: $LOG"

#!/usr/bin/env bash
# On-device CPU profile of `cera bench` via simpleperf.
#
# The point of this script is to make a perf claim falsifiable: after a kernel
# change, the top symbols must actually move. A throughput delta alone can come
# from thermal drift or scheduler luck; a hotspot that shifts off the function
# you optimized cannot.
#
# Pinned to a single big core by default: sampling across big.LITTLE mixes cores
# with a ~4x IPC difference into one profile, which smears the very hotspot
# you're trying to read.
#
# Usage:
#   scripts/profile_android_cpu.sh --model <name.gguf> [--serial <adb-serial>]
#                                  [--mask 80] [--device cpu]
#                                  [--prompt 512] [--decode 128]
#
# Requires simpleperf on the device. It ships inside the NDK at
# $ANDROID_NDK_HOME/simpleperf/bin/android/arm64/simpleperf — this script pushes
# it if it isn't already on the device.
set -euo pipefail

DEVICE_DIR="/data/local/tmp/cera-bench"
MODEL=""
SERIAL="${CERA_ANDROID_SERIAL:-}"
MASK="80"          # prime core (cpu7)
BACKEND="cpu"
PROMPT=512
DECODE=128
DURATION=20        # seconds of sampling

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL="$2"; shift 2 ;;
    --serial) SERIAL="$2"; shift 2 ;;
    --mask)   MASK="$2"; shift 2 ;;
    --device) BACKEND="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --decode) DECODE="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model <name.gguf> is required" >&2; exit 2; }

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

"${ADB[@]}" shell "mkdir -p $DEVICE_DIR"

# Push simpleperf from the NDK if the device doesn't already have it.
if ! "${ADB[@]}" shell "test -x $DEVICE_DIR/simpleperf" 2>/dev/null; then
  : "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME so simpleperf can be located}"
  SP="$ANDROID_NDK_HOME/simpleperf/bin/android/arm64/simpleperf"
  [[ -f "$SP" ]] || { echo "simpleperf not found at $SP" >&2; exit 2; }
  echo "==> pushing simpleperf"
  "${ADB[@]}" push "$SP" "$DEVICE_DIR/simpleperf" >/dev/null
  "${ADB[@]}" shell "chmod +x $DEVICE_DIR/simpleperf"
fi

# A release cera is stripped (see the release profile), which would leave the
# profile as a wall of hex addresses. Warn rather than silently produce one.
if ! "${ADB[@]}" shell "test -x $DEVICE_DIR/cera"; then
  echo "error: $DEVICE_DIR/cera missing — run scripts/bench_android.sh first." >&2
  exit 2
fi

echo "==> profiling: device=$BACKEND mask=$MASK prompt=$PROMPT decode=$DECODE"
echo "    (symbols require a non-stripped binary; build with"
echo "     CARGO_PROFILE_RELEASE_STRIP=false and RUSTFLAGS='-C force-frame-pointers=yes')"

# -g gives call graphs (needs frame pointers to be useful on aarch64).
# --duration bounds the sample window; bench keeps running past it, which is
# fine — we only want a representative window of steady-state decode.
"${ADB[@]}" shell "cd $DEVICE_DIR && ./simpleperf record -g --duration $DURATION -o perf.data \
  -- taskset $MASK ./cera bench -m $MODEL --device $BACKEND \
     --prompt-tokens $PROMPT --max-tokens $DECODE --runs 3 --warmup 1 --no-cache" 2>&1 | tail -5

echo
echo "==> top symbols (self time)"
"${ADB[@]}" shell "cd $DEVICE_DIR && ./simpleperf report -i perf.data --sort symbol -n" 2>&1 | head -35

echo
echo "==> pulling perf.data for offline inspection (simpleperf report -i perf.data)"
"${ADB[@]}" pull "$DEVICE_DIR/perf.data" ./perf.data >/dev/null 2>&1 && echo "    ./perf.data"

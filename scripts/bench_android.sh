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
#                            [--decode-prompt 128] [--passes 5] [--equil-warm 2]
#                            [--min-battery 30]
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
DECODE=512
# Prompt depth cera's decode run starts from. This is the one axis the two
# harnesses do NOT match: llama-bench's `tg` always starts from an empty context
# and this script does not pass its `-d/--n-depth`, so cera decodes at greater KV
# depth. That disfavours cera, so a cera decode win here is a lower bound; keep
# this small rather than inheriting $PROMPT, which made the gap much worse.
DECODE_PROMPT=128
RUNS=5
WARMUP=2
# Passes to discard while the SoC reaches thermal equilibrium, then measured
# passes. Do NOT add an idle between them: cooling drops the device back into the
# thermal transient, which is what made earlier numbers unrepeatable.
EQUIL_WARM=2
PASSES=5
# Minimum battery level to measure at. Android reduces peak clocks at low
# battery, and a run that drains across its own matrix compares its early cells
# against its late ones at different power budgets. A whole session was measured
# here from 93% down to 14% before this was noticed.
MIN_BATTERY=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --serial)      SERIAL="$2"; shift 2 ;;
    --llama-bench) LLAMA_BENCH="$2"; shift 2 ;;
    --prompt)      PROMPT="$2"; shift 2 ;;
    --decode)      DECODE="$2"; shift 2 ;;
    --runs)        RUNS="$2"; shift 2 ;;
    --warmup)      WARMUP="$2"; shift 2 ;;
    --decode-prompt) DECODE_PROMPT="$2"; shift 2 ;;
    --passes)      PASSES="$2"; shift 2 ;;
    --equil-warm)  EQUIL_WARM="$2"; shift 2 ;;
    --min-battery) MIN_BATTERY="$2"; shift 2 ;;
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
# One row per cell, aggregated across PASSES measured passes rather than a single
# invocation: the dominant variance on this device is between invocations, so the
# invocation has to be the sampling unit. `*_cov` is the coefficient of variation
# across passes (percent) and is the number that says whether a gap is real.
# soc_big_min/max bracket the BIG-cluster temperature seen around this cell, and
# exist to prove equilibrium held; if they span more than a few degrees the run
# was still in the thermal transient and the medians are not comparable.
echo "engine,config,prefill_med,prefill_cov,decode_med,decode_cov,n,soc_big_min,soc_big_max,cpu_mhz_min,cpu_mhz_max,batt_start,batt_end,power_state" > "$OUT"
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
#
# ---------------------------------------------------------------------------
# Why this measures at thermal equilibrium
#
# A short decode measurement on this device happens inside a violent thermal
# transient: the BIG cluster goes 26 C -> 74 C in about twelve seconds of load
# and falls back to 30 C within twenty seconds of stopping. Where a 2-second
# measurement lands inside that ramp decides its result, which is why repeating
# one identical pinned config gave 7.1 / 63.3 / 64.9 / 40.0 / 22.7 / 22.1, a
# 9.1x spread.
#
# Battery temperature does NOT show this. It moves ~0.5 C while the silicon
# swings 48 C, so it is useless as a gate; this script reads the live BIG/MID
# cluster temperatures out of `dumpsys thermalservice` instead.
#
# Two fixes were measured. Gating each invocation on a cold SoC still left 1.48x
# (the transient is the problem, not the start point). Driving the device to
# thermal equilibrium and measuring there gave 1.12x spread and ~4.3% CoV on
# both engines, and a cera/llama decode ratio significant at 3.9 sigma. So:
#
#   1. run EQUIL_WARM invocations first, discarded, to reach steady state;
#   2. then run PASSES measured passes BACK TO BACK with no cooldown, because
#      idling between measurements drops the device back into the transient;
#   3. interleave the engines, which both removes the ordering bias and keeps
#      the load continuous so equilibrium holds;
#   4. report median and CoV across passes, with the SoC temperature range as
#      evidence that equilibrium actually held.
#
# This measures SUSTAINED throughput, which is the reproducible quantity and the
# one that matters for comparing engines or commits. It is deliberately not the
# peak a cold phone can hit for two seconds.
# ---------------------------------------------------------------------------

# Live BIG-cluster temperature, integer C. Must come from "Current temperatures
# from HAL"; the "Cached temperatures" section earlier in the same dumpsys is
# stale and reads high long after the device has cooled.
soc_big() {
  "${ADB[@]}" shell "dumpsys thermalservice" 2>/dev/null \
    | awk '/Current temperatures from HAL/,/Current cooling devices/' \
    | sed -n 's/.*mValue=\([0-9.]*\), mType=0, mName=BIG, .*/\1/p' | head -1 | cut -d. -f1
}

# Mid-run CPU frequency on a perf core, MHz. This is the variable that actually
# predicts throughput: measured against llama-bench decode over 16 invocations,
# corr(tok/s, frequency) = +0.83, and filtering to samples taken at the maximum
# clock cut the coefficient of variation from 17.1% to 2.6% (cera: 13.2% to
# 6.0%). Temperature correlates with frequency at +0.95 in the SAME direction,
# so a hot device here is a fast one; thermal throttling is not the mechanism.
#
# The governor (`sched_pixel`, 24 operating points from 177 MHz to 3052 MHz)
# ramps on recent load, and a short measurement can finish before it reaches the
# top. That cannot be fixed from outside: pinning the governor needs root,
# pre-warming decays across the adb round trip, longer measurements trade ramp
# noise for thermal decline, and warmup iterations make no difference (all
# measured). So record it and let the reader see which samples were taken at
# speed.
cpu_mhz() {
  local f
  f=$("${ADB[@]}" shell "cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq" 2>/dev/null | tr -d '\r')
  [[ -n "$f" ]] && echo $((f / 1000)) || echo ""
}

batt_level() { "${ADB[@]}" shell "dumpsys battery | grep -i '^  level'" 2>/dev/null | grep -oE '[0-9]+' | head -1; }

# Three-way, not a boolean: "plugged in but not charging" is its own power
# envelope (the charger is current-limiting), and conflating it with charging
# hides a real difference in what the SoC is allowed to draw.
power_state() {
  local out; out=$("${ADB[@]}" shell "dumpsys battery" 2>/dev/null)
  local plugged=no
  grep -qE '^  (AC|USB|Wireless|Dock) powered: true' <<<"$out" && plugged=yes
  local status; status=$(grep -oE '^  status: [0-9]+' <<<"$out" | grep -oE '[0-9]+')
  if [[ "$plugged" == yes && "$status" == 2 ]]; then echo charging
  elif [[ "$plugged" == yes ]]; then echo plugged_not_charging
  else echo on_battery; fi
}

# Samples accumulate in one file per cell+phase. Deliberately not `declare -A`:
# macOS ships bash 3.2, which has no associative arrays, and this script is run
# from a Mac far more often than from the device.
SAMPLE_DIR=$(mktemp -d)
trap 'rm -rf "$SAMPLE_DIR"' EXIT

slug() { tr -c 'A-Za-z0-9_' '_' <<<"$1"; }

record() { # key value
  [[ -n "$2" ]] || return 0
  echo "$2" >> "$SAMPLE_DIR/$(slug "$1").vals"
}

note_soc() { # key
  local t; t=$(soc_big); [[ -n "$t" ]] && echo "$t" >> "$SAMPLE_DIR/$(slug "$1").soc"
  local m; m=$(cpu_mhz); [[ -n "$m" ]] && echo "$m" >> "$SAMPLE_DIR/$(slug "$1").mhz"
  return 0
}

sample_cera() { # backend label mask
  local backend="$1" mask="$3" key="cera|$2"
  local pin=""; [[ -n "$mask" ]] && pin="taskset $mask "
  local base="cd $DEVICE_DIR && ${pin}./cera bench -m $MODEL --device $backend \
--runs $RUNS --warmup $WARMUP --no-cache --gpu-io"
  note_soc "$key"
  local pre dec
  pre=$("${ADB[@]}" shell "$base --prompt-tokens $PROMPT --max-tokens 0" 2>&1 || true)
  echo "$pre" >> "$LOG"
  record "$key|prefill" "$(sed -n 's/.*prefill tok\/s: p50=\([0-9.]*\).*/\1/p' <<<"$pre" | head -1)"
  dec=$("${ADB[@]}" shell "$base --prompt-tokens $DECODE_PROMPT --max-tokens $DECODE" 2>&1 || true)
  echo "$dec" >> "$LOG"
  record "$key|decode" "$(sed -n 's/.*decode tok\/s: p50=\([0-9.]*\).*/\1/p' <<<"$dec" | head -1)"
  note_soc "$key"
  printf '%s\n%s\n' "$pre" "$dec" | grep -E "^gpu I/O" >> "$LOG" || true
}

sample_llama() { # threads mask
  [[ -n "$LLAMA_BENCH" ]] || return 0
  local t="$1" mask="$2" key="llama.cpp|t$1-$2" rt
  rt=$(dirname "$LLAMA_BENCH")
  note_soc "$key"
  local out
  out=$("${ADB[@]}" shell "cd $rt && LD_LIBRARY_PATH=. taskset $mask ./$(basename "$LLAMA_BENCH") \
-m $DEVICE_DIR/$MODEL -t $t -p $PROMPT -n $DECODE -r $RUNS -o md" 2>&1) || true
  echo "$out" >> "$LOG"
  record "$key|prefill" "$(grep -E "\|[[:space:]]*pp${PROMPT}[[:space:]]*\|" <<<"$out" | sed -n 's/.*|[[:space:]]*\([0-9.]*\) ±.*/\1/p' | head -1)"
  record "$key|decode"  "$(grep -E "\|[[:space:]]*tg${DECODE}[[:space:]]*\|"  <<<"$out" | sed -n 's/.*|[[:space:]]*\([0-9.]*\) ±.*/\1/p' | head -1)"
  note_soc "$key"
}

# One pass over every cell, engines interleaved.
one_pass() {
  sample_cera cpu "default-rowpool" ""
  sample_llama 5 "7c"
  sample_cera cpu "pin-prime-80" "80"
  sample_llama 1 "80"
  sample_cera cpu "pin-perf-7c" "7c"
  sample_llama 6 "fc"
  sample_cera gpu "wgpu-vulkan" ""
}

BATT_START=$(batt_level); POWER_STATE=$(power_state)
echo "==> battery ${BATT_START}% (${POWER_STATE}), min required ${MIN_BATTERY}%" | tee -a "$LOG"
if [[ -n "$BATT_START" && "$BATT_START" -lt "$MIN_BATTERY" ]]; then
  echo "error: battery ${BATT_START}% is below --min-battery ${MIN_BATTERY}%." >&2
  echo "       Android throttles at low battery, so these numbers would not be" >&2
  echo "       comparable to a run taken at a healthy level. Charge, or lower the" >&2
  echo "       gate deliberately and record that you did." >&2
  exit 3
fi

echo "==> driving to thermal equilibrium ($EQUIL_WARM warm-up passes, discarded)"
for ((w = 0; w < EQUIL_WARM; w++)); do
  one_pass >/dev/null 2>&1 || true
  echo "    warm-up pass $((w + 1))/$EQUIL_WARM done, BIG=$(soc_big) C" | tee -a "$LOG"
done
rm -f "$SAMPLE_DIR"/*.vals "$SAMPLE_DIR"/*.soc   # discard warm-up samples

for ((pass = 1; pass <= PASSES; pass++)); do
  echo "==> measured pass $pass/$PASSES (BIG=$(soc_big) C)" | tee -a "$LOG"
  one_pass
done

# median and CoV across passes
stats() { # file -> "median cov n"
  grep -hE '^[0-9.]+$' "$1" 2>/dev/null | sort -n | awk '
    {v[NR]=$1; s+=$1}
    END{
      if (NR==0) { print "NA NA 0"; exit }
      m = (NR%2) ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2
      mean = s/NR; ss=0
      for (i=1;i<=NR;i++) ss += (v[i]-mean)^2
      cov = (mean>0) ? sqrt(ss/NR)/mean*100 : 0
      printf "%.1f %.1f %d", m, cov, NR
    }'
}

BATT_END=$(batt_level)

range_of() { # key ext -> "min max"
  local f; f="$SAMPLE_DIR/$(slug "$1").$2"
  [[ -s "$f" ]] || { echo "NA NA"; return; }
  sort -n "$f" | awk 'NR==1{min=$1} {max=$1} END{printf "%s %s", min, max}'
}

{
  for cell in "cera|default-rowpool" "cera|pin-prime-80" "cera|pin-perf-7c" "cera|wgpu-vulkan" \
              "llama.cpp|t5-7c" "llama.cpp|t1-80" "llama.cpp|t6-fc"; do
    eng="${cell%%|*}"; cfg="${cell##*|}"
    read -r pmed pcov _ <<<"$(stats "$SAMPLE_DIR/$(slug "$cell|prefill").vals")"
    read -r dmed dcov dn <<<"$(stats "$SAMPLE_DIR/$(slug "$cell|decode").vals")"
    read -r smin smax <<<"$(range_of "$cell" soc)"
    read -r fmin fmax <<<"$(range_of "$cell" mhz)"
    echo "$eng,$cfg,$pmed,$pcov,$dmed,$dcov,$dn,$smin,$smax,$fmin,$fmax,${BATT_START:-NA},${BATT_END:-NA},${POWER_STATE:-NA}"
  done
} >> "$OUT"

echo
echo "==> $OUT"
column -t -s, "$OUT"
echo "==> raw runs: $LOG"

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
FORCE=no

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
    --force)       FORCE=yes; shift ;;
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

# ---------------------------------------------------------------------------
# CPU topology, detected rather than hardcoded.
#
# The taskset masks here used to be literals (80 / fc / 7c) picked for one SoC.
# Run against a different big.LITTLE layout the same literal selects different
# cores: on a Tensor G5 `7c` is the perf cluster, on a Tensor G4 it spans two
# LITTLE cores plus the mid cluster. That produces a wrong number that looks
# like a right one, so derive the masks from the device instead.
#
# Clusters are grouped by cpuinfo_max_freq: highest = prime, next = mid, the
# rest little. LITTLE cores are never a benchmark target on their own; they only
# appear inside the combined "big" mask if the SoC has no separate mid tier.
# ---------------------------------------------------------------------------
TOPO_RAW=""
detect_topology() {
  TOPO_RAW=$("${ADB[@]}" shell 'for c in /sys/devices/system/cpu/cpu[0-9]*; do
      n=${c##*/cpu}; f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null)
      [ -n "$f" ] && echo "$n $f"
    done' 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+ [0-9]+$' || true)
  [[ -n "$TOPO_RAW" ]] || { echo "error: could not read CPU topology from device" >&2; exit 2; }

  local freqs; freqs=$(awk '{print $2}' <<<"$TOPO_RAW" | sort -rnu)
  local prime_f mid_f
  prime_f=$(sed -n 1p <<<"$freqs")
  mid_f=$(sed -n 2p <<<"$freqs")

  read -r PRIME_MASK N_PRIME <<<"$(cluster_mask "$prime_f")"
  PRIME_CORE=$(awk -v w="$prime_f" '$2==w {print $1; exit}' <<<"$TOPO_RAW")

  if [[ -n "$mid_f" ]]; then
    read -r MID_MASK N_MID <<<"$(cluster_mask "$mid_f")"
    read -r BIG_MASK N_BIG <<<"$(cluster_mask "$prime_f" "$mid_f")"
  else
    # Single-frequency SoC: no separate mid tier to pin against.
    MID_MASK="$PRIME_MASK"; N_MID="$N_PRIME"
    BIG_MASK="$PRIME_MASK"; N_BIG="$N_PRIME"
  fi
}

# Hex taskset mask (and core count) for every core whose max freq is one of the
# given values.
cluster_mask() {
  local want="$1" want2="${2:-}"
  awk -v a="$want" -v b="$want2" '
    ($2 == a) || (b != "" && $2 == b) { m += 2 ^ $1; n++ }
    END { printf "%x %d", m, n }' <<<"$TOPO_RAW"
}

OUT="${CERA_BENCH_OUT:-bench_android.csv}"
LOG="${OUT%.csv}_raw.log"

# A full matrix costs hours of device time, and the previous version silently
# truncated both files on every invocation, so a two-minute smoke test run from
# the same directory destroyed a two-hour result set. Refuse to overwrite; set
# CERA_BENCH_OUT to write elsewhere, or --force when you mean it.
if [[ -e "$OUT" && "$FORCE" != "yes" ]]; then
  echo "error: $OUT already exists; refusing to overwrite a previous run." >&2
  echo "       Use --force to replace it, or set CERA_BENCH_OUT=<path.csv>." >&2
  exit 2
fi
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

# taskset masks, derived per device by detect_topology:
#   (none)     = let the scheduler place threads (cera's RowPool sizes itself)
#   PRIME_MASK = prime core(s) only
#   MID_MASK   = mid cluster only
#   BIG_MASK   = prime + mid
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
  # `|| true` is load-bearing: this is an *assignment*-form substitution, so
  # under `set -euo pipefail` a failing pipeline (offline core, missing cpufreq
  # node, adb hiccup) aborts the whole run rather than skipping one sample. The
  # argument-form parses elsewhere degrade to an empty value instead, which is
  # why they do not need it.
  f=$("${ADB[@]}" shell "cat /sys/devices/system/cpu/cpu${PRIME_CORE}/cpufreq/scaling_cur_freq" 2>/dev/null | tr -d '\r' || true)
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
# stop_samplers first: an interrupted run would otherwise leave a host-side adb
# shell and an on-device polling loop running after the script exits.
trap 'stop_samplers 2>/dev/null; rm -rf "$SAMPLE_DIR"' EXIT

slug() { tr -c 'A-Za-z0-9_' '_' <<<"$1"; }

record() { # key value
  [[ -n "$2" ]] || return 0
  echo "$2" >> "$SAMPLE_DIR/$(slug "$1").vals"
}

# Temperature and frequency must be sampled DURING an invocation, not around it.
#
# The previous version called a sampler before a cell's first run and after its
# last. Both are moments when the CPU has been idle for an adb round trip, and
# this silicon sheds heat and drops clocks in seconds, so those columns recorded
# 700 MHz and 45 C while the workload was actually running at 3015 MHz and 85 C.
# They could not see the effect they existed to detect, which is the same mistake
# as the earlier battery-temperature instrumentation.
#
# Frequency comes off sysfs on-device at 1 Hz, which costs one `cat` per second.
# Temperature has no unprivileged sysfs path (`thermal_zone*` is root-only), so
# it is polled from the host via dumpsys at 0.2 Hz. That is a real perturbation,
# but a small and uniform one, and measuring the right quantity approximately
# beats measuring the wrong one precisely.
MHZ_SAMPLER=""
SOC_SAMPLER=""
SAMPLE_BASE=""
SAMPLER_TAG="cera_bench_sampler_$$"

start_samplers() { # key
  SAMPLE_BASE="$SAMPLE_DIR/$(slug "$1")"
  # Deliberately NOT `adb ... | awk ... &`. For a backgrounded pipeline `$!` is
  # the LAST command (awk), so killing it leaves the adb shell running and the
  # subsequent `wait` blocks on the rest of the pipeline forever. Redirect the
  # raw stream to a file so `$!` is adb itself, and parse it in stop_samplers.
  # The tag exists so stop_samplers can pkill the DEVICE-side loop. Killing the
  # host adb client does not kill it: adbd keeps the pty open, so the loop never
  # takes SIGPIPE and survives until `timeout` reaps it. Left unfixed, a full
  # matrix accumulates one polling loop per cell per pass, all of them running
  # for fifteen minutes on the device whose performance is being measured.
  "${ADB[@]}" shell "timeout 900 sh -c ': $SAMPLER_TAG; while :; do
      cat /sys/devices/system/cpu/cpu${PRIME_CORE}/cpufreq/scaling_cur_freq 2>/dev/null; sleep 1
    done'" > "$SAMPLE_BASE.mhz.raw" 2>/dev/null &
  MHZ_SAMPLER=$!
  ( while :; do
      t=$(soc_big); [[ -n "$t" ]] && echo "$t" >> "$SAMPLE_BASE.soc"
      sleep 5
    done ) &
  SOC_SAMPLER=$!
}

stop_samplers() {
  [[ -n "$MHZ_SAMPLER" ]] && { kill "$MHZ_SAMPLER" 2>/dev/null || true; }
  [[ -n "$SOC_SAMPLER" ]] && { kill "$SOC_SAMPLER" 2>/dev/null || true; }
  [[ -n "$MHZ_SAMPLER" ]] && { "${ADB[@]}" shell "pkill -f $SAMPLER_TAG" >/dev/null 2>&1 || true; }
  wait "$MHZ_SAMPLER" "$SOC_SAMPLER" 2>/dev/null || true
  if [[ -n "$SAMPLE_BASE" && -f "$SAMPLE_BASE.mhz.raw" ]]; then
    awk '{ sub(/\r$/, ""); if ($0 ~ /^[0-9]+$/) print int($0 / 1000) }' \
      "$SAMPLE_BASE.mhz.raw" >> "$SAMPLE_BASE.mhz"
    rm -f "$SAMPLE_BASE.mhz.raw"
  fi
  MHZ_SAMPLER=""; SOC_SAMPLER=""; SAMPLE_BASE=""
  return 0
}

sample_cera() { # backend label mask
  local backend="$1" mask="$3" key="cera|$2"
  local pin=""; [[ -n "$mask" ]] && pin="taskset $mask "
  local base="cd \"$DEVICE_DIR\" && ${pin}./cera bench -m \"$MODEL\" --device $backend \
--runs $RUNS --warmup $WARMUP --no-cache --gpu-io"
  local pre dec
  start_samplers "$key"
  pre=$("${ADB[@]}" shell "$base --prompt-tokens $PROMPT --max-tokens 0" 2>&1 || true)
  dec=$("${ADB[@]}" shell "$base --prompt-tokens $DECODE_PROMPT --max-tokens $DECODE" 2>&1 || true)
  stop_samplers
  printf '%s\n' "$pre" >> "$LOG"
  printf '%s\n' "$dec" >> "$LOG"
  # cera prints its decode summary BEFORE its prefill summary, so each value must
  # be taken from its own invocation rather than by position in the log.
  record "$key|prefill" "$(sed -n 's/.*prefill tok\/s: p50=\([0-9.]*\).*/\1/p' <<<"$pre" | head -1)"
  record "$key|decode" "$(sed -n 's/.*decode tok\/s: p50=\([0-9.]*\).*/\1/p' <<<"$dec" | head -1)"
  printf '%s\n%s\n' "$pre" "$dec" | grep -E "^gpu I/O" >> "$LOG" || true
}

# Config label for a llama cell: threads, cluster name, and the mask it actually
# ran with, so a row stays interpretable on a device with a different layout.
llama_cfg() { # threads label mask
  if [[ -n "$3" ]]; then echo "t$1-$2-$3"; else echo "t$1-$2"; fi
}

sample_llama() { # threads label mask
  [[ -n "$LLAMA_BENCH" ]] || return 0
  # Declared separately from the substitution: `local x=$(...)` makes `local`
  # the command whose status is seen, masking a failure inside (SC2155).
  local t="$1" mask="$3" key rt
  key="llama.cpp|$(llama_cfg "$1" "$2" "$3")"
  rt=$(dirname "$LLAMA_BENCH")
  local out pin=""
  # An empty mask means "unpinned", the symmetric counterpart to cera's default
  # RowPool cell; taskset with no mask is a syntax error, so omit it entirely.
  [[ -n "$mask" ]] && pin="taskset $mask "
  start_samplers "$key"
  out=$("${ADB[@]}" shell "cd \"$rt\" && LD_LIBRARY_PATH=. ${pin}./\"$(basename "$LLAMA_BENCH")\" \
-m \"$DEVICE_DIR/$MODEL\" -t $t -p $PROMPT -n $DECODE -r $RUNS -o md" 2>&1) || true
  stop_samplers
  printf '%s\n' "$out" >> "$LOG"
  record "$key|prefill" "$(grep -E "\|[[:space:]]*pp${PROMPT}[[:space:]]*\|" <<<"$out" | sed -n 's/.*|[[:space:]]*\([0-9.]*\) ±.*/\1/p' | head -1)"
  record "$key|decode"  "$(grep -E "\|[[:space:]]*tg${DECODE}[[:space:]]*\|"  <<<"$out" | sed -n 's/.*|[[:space:]]*\([0-9.]*\) ±.*/\1/p' | head -1)"
}

# The cells, and the single pass over them. Labels carry the mask they actually
# ran with so a CSV row stays interpretable on a device with a different layout.
CELLS=()
build_cells() {
  N_ALL=$(wc -l <<<"$TOPO_RAW" | tr -d ' ')
  CERA_DEFAULT="default-rowpool"
  CERA_PRIME="pin-prime-$PRIME_MASK"
  CERA_MID="pin-mid-$MID_MASK"
  CERA_GPU="wgpu-vulkan"
  LLAMA_PRIME=$(llama_cfg 1 prime "$PRIME_MASK")
  LLAMA_MID=$(llama_cfg "$N_MID" mid "$MID_MASK")
  LLAMA_BIG=$(llama_cfg "$N_BIG" big "$BIG_MASK")
  # Unpinned, every core. Without this cell the matrix has no counterpart to
  # cera's default RowPool, which is also unpinned across every core, so a
  # best-vs-best prefill comparison silently pits cera on N cores against llama
  # on the largest cluster only. That asymmetry favours cera.
  LLAMA_ALL=$(llama_cfg "$N_ALL" all "")
  CELLS=(
    "cera|$CERA_DEFAULT" "cera|$CERA_PRIME" "cera|$CERA_MID" "cera|$CERA_GPU"
    "llama.cpp|$LLAMA_PRIME" "llama.cpp|$LLAMA_MID" "llama.cpp|$LLAMA_BIG"
    "llama.cpp|$LLAMA_ALL"
  )
}

# One pass over every cell, engines interleaved.
one_pass() {
  sample_cera cpu "$CERA_DEFAULT" ""
  sample_llama "$N_MID" mid "$MID_MASK"
  sample_cera cpu "$CERA_PRIME" "$PRIME_MASK"
  sample_llama 1 prime "$PRIME_MASK"
  sample_cera cpu "$CERA_MID" "$MID_MASK"
  sample_llama "$N_BIG" big "$BIG_MASK"
  sample_cera gpu "$CERA_GPU" ""
  sample_llama "$N_ALL" all ""
}

detect_topology
build_cells
echo "==> topology: prime=$PRIME_MASK (${N_PRIME}c, sampling cpu$PRIME_CORE), mid=$MID_MASK (${N_MID}c), big=$BIG_MASK (${N_BIG}c)" | tee -a "$LOG"

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
# Discard warm-up samples. This must cover every extension the samplers write:
# `.mhz` was missing here, so warm-up frequencies leaked into the measured range.
rm -f "$SAMPLE_DIR"/*.vals "$SAMPLE_DIR"/*.soc "$SAMPLE_DIR"/*.mhz

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

# A run that loses the device produces a header-only CSV and, before this
# check, exited 0. An empty result that reports success is worse than a failure:
# it reads downstream as "measured, nothing to see". Fail loudly instead.
if ! ls "$SAMPLE_DIR"/*.vals >/dev/null 2>&1; then
  echo "error: no samples collected across $PASSES passes; every cell failed." >&2
  if ! "${ADB[@]}" get-state >/dev/null 2>&1; then
    echo "       The device is no longer reachable over adb (it dropped mid-run)." >&2
  fi
  echo "       See $LOG for the raw output." >&2
  exit 4
fi

range_of() { # key ext -> "min max"
  local f; f="$SAMPLE_DIR/$(slug "$1").$2"
  [[ -s "$f" ]] || { echo "NA NA"; return; }
  sort -n "$f" | awk 'NR==1{min=$1} {max=$1} END{printf "%s %s", min, max}'
}

{
  for cell in "${CELLS[@]}"; do
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

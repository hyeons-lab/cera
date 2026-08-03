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
#                            [--decode-prompt 128] [--settle 30]
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
# Prompt depth cera's decode run starts from. This is the one axis the two
# harnesses do NOT match: llama-bench's `tg` always starts from an empty context
# and this script does not pass its `-d/--n-depth`, so cera decodes at greater KV
# depth. That disfavours cera, so a cera decode win here is a lower bound; keep
# this small rather than inheriting $PROMPT, which made the gap much worse.
DECODE_PROMPT=128
RUNS=5
WARMUP=2
# Seconds to idle between measurements. Thermal state dominates on a phone and
# back-to-back runs drift downward, so this defaults ON; pass `--settle 0` to opt
# out when you only want a quick smoke run rather than comparable numbers.
SETTLE=30

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
    --settle)      SETTLE="$2"; shift 2 ;;
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
# Prefill and decode are measured in SEPARATE invocations, and this is not
# cosmetic. Timing prefill in a run that also decodes reads it ~10% low with ~8x
# the variance, and llama-bench times its `pp` and `tg` runs separately, so a
# combined cera run silently biases every cross-engine prefill ratio against
# cera. Decode likewise wants its own run: llama-bench's `tg` starts from an
# empty context, so decoding after a 512-token prefill compares different KV
# depths. See "Known measurement traps" in benchmarks/BASELINE.md.
run_cera() {
  local backend="$1" label="$2" mask="$3"
  local pin=""; [[ -n "$mask" ]] && pin="taskset $mask "
  local base="cd $DEVICE_DIR && ${pin}./cera bench -m $MODEL --device $backend \
--runs $RUNS --warmup $WARMUP --no-cache --gpu-io"

  echo "=== cera $backend [$label] ===" | tee -a "$LOG"
  local pre_out dec_out
  if ! pre_out=$("${ADB[@]}" shell "$base --prompt-tokens $PROMPT --max-tokens 0" 2>&1); then
    echo "$pre_out" >> "$LOG"
    echo "cera,$backend,$label,FAIL,FAIL,FAIL,FAIL" >> "$OUT"
    return
  fi
  # Log the prefill run as soon as it succeeds: if the decode run below fails we
  # return early, and its output would otherwise be lost from the raw log.
  echo "$pre_out" >> "$LOG"
  # Decode measures right after a full prefill run; settle so it is not timed on
  # the heat that run just produced.
  settle
  if ! dec_out=$("${ADB[@]}" shell "$base --prompt-tokens $DECODE_PROMPT --max-tokens $DECODE" 2>&1); then
    echo "${dec_out:-}" >> "$LOG"
    echo "cera,$backend,$label,FAIL,FAIL,FAIL,FAIL" >> "$OUT"
    return
  fi
  echo "$dec_out" >> "$LOG"

  local pre dec p50 d50 psd dsd
  # `|| true` on every grep: under `set -euo pipefail` a non-matching grep in a
  # command substitution aborts the whole script, which would silently kill the
  # rest of the matrix instead of writing the FAIL row handled above.
  pre=$(grep -E "^prefill tok/s:" <<<"$pre_out" | head -1 || true)
  dec=$(grep -E "^decode tok/s:"  <<<"$dec_out" | head -1 || true)
  p50=$(sed -n 's/.*p50=\([0-9.]*\).*/\1/p' <<<"$pre")
  d50=$(sed -n 's/.*p50=\([0-9.]*\).*/\1/p' <<<"$dec")
  psd=$(sed -n 's/.*stddev=\([0-9.]*\).*/\1/p' <<<"$pre")
  dsd=$(sed -n 's/.*stddev=\([0-9.]*\).*/\1/p' <<<"$dec")
  echo "cera,$backend,$label,${p50:-NA},${d50:-NA},${psd:-NA},${dsd:-NA}" >> "$OUT"
  echo "  -> prefill p50=$p50 decode p50=$d50" | tee -a "$LOG"
  # The --gpu-io lines say whether a GPU change actually removed round-trips;
  # keep them in the raw log where they can't be lost to CSV flattening. Grep
  # BOTH runs: `cera bench` skips the report when it generates no tokens, so the
  # prefill-at-$PROMPT counters only ever appear in the decode run's output, and
  # the prefill run contributes its own submit counts when it does report.
  printf '%s\n%s\n' "$pre_out" "$dec_out" | grep -E "^gpu I/O" | tee -a "$LOG" || true
}

# llama.cpp reference, if a llama-bench is present on the device. Uses the same
# model file so the comparison is same-quant, same-weights. llama-bench already
# times `pp` and `tg` as separate runs internally, which is what run_cera above
# was changed to match.
run_llama() {
  [[ -n "$LLAMA_BENCH" ]] || return 0
  local t="$1" mask="$2"
  local rt; rt=$(dirname "$LLAMA_BENCH")
  echo "=== llama-bench -t $t (taskset $mask) ===" | tee -a "$LOG"
  local out
  out=$("${ADB[@]}" shell "cd $rt && LD_LIBRARY_PATH=. taskset $mask ./$(basename "$LLAMA_BENCH") \
-m $DEVICE_DIR/$MODEL -t $t -p $PROMPT -n $DECODE -r $RUNS -o md" 2>&1) || true
  echo "$out" >> "$LOG"
  # llama-bench md rows: | model | size | params | backend | threads | test | t/s |
  # Keep llama-bench's own `± stddev`: every claim in benchmarks/BASELINE.md is
  # argued against dispersion, and dropping it made the llama rows the only ones
  # that could not be tested for significance. `|| true` for the same reason as
  # in run_cera: a changed md format must not abort the rest of the matrix.
  local pprow tgrow pp tg ppsd tgsd
  pprow=$(grep -E "\|[[:space:]]*pp${PROMPT}[[:space:]]*\|" <<<"$out" | head -1 || true)
  tgrow=$(grep -E "\|[[:space:]]*tg${DECODE}[[:space:]]*\|" <<<"$out" | head -1 || true)
  pp=$(sed -n "s/.*|[[:space:]]*\\([0-9.]*\\) ±.*/\\1/p" <<<"$pprow")
  tg=$(sed -n "s/.*|[[:space:]]*\\([0-9.]*\\) ±.*/\\1/p" <<<"$tgrow")
  ppsd=$(sed -n "s/.*± *\\([0-9.]*\\).*/\\1/p" <<<"$pprow")
  tgsd=$(sed -n "s/.*± *\\([0-9.]*\\).*/\\1/p" <<<"$tgrow")
  echo "llama.cpp,cpu,t$t-$mask,${pp:-NA},${tg:-NA},${ppsd:-NA},${tgsd:-NA}" >> "$OUT"
  echo "  -> pp=$pp tg=$tg" | tee -a "$LOG"
}

settle() { [[ "$SETTLE" -gt 0 ]] && sleep "$SETTLE" || true; }

# Engines are INTERLEAVED. Running every cera config and then every llama config
# measures llama on a hotter device, which is a systematic bias in cera's favour
# whenever cooling is imperfect. Alternating spreads any residual thermal drift
# across both engines instead of concentrating it in the one that runs last.
run_cera cpu "default-rowpool" "";  settle
run_llama 5 "7c";                   settle
run_cera cpu "pin-prime-80"    "80"; settle
run_llama 1 "80";                   settle
run_cera cpu "pin-perf-7c"     "7c"; settle
run_llama 6 "fc";                   settle
run_cera gpu "wgpu-vulkan"     ""

echo
echo "==> $OUT"
column -t -s, "$OUT"
echo "==> raw runs: $LOG"

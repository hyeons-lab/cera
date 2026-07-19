#!/usr/bin/env bash
# Host CPU profile of `cera bench` via perf (Linux) or samply (Linux/macOS).
#
# The host-side counterpart to scripts/profile_android_cpu.sh, and it exists for
# the same reason: to make a perf claim falsifiable. After a kernel change the
# top symbols must actually move. A throughput delta alone can come from turbo
# residency, an unlucky scheduler, or a noisy neighbour; a hotspot that shifts
# off the function you optimized cannot.
#
# Profile prefill and decode *separately* (--mode). They have opposite
# bottlenecks — prefill is weight-bandwidth bound and should sit in the batched
# GEMM, decode is latency bound and should sit in the GEMV. A blended profile
# averages the two into something that describes neither.
#
# Pinned to a fixed core set by default: on a hybrid part (Intel P/E, Zen with
# dual CCX) an unpinned sample mixes cores with different clocks and cache
# topology into one profile, smearing the hotspot you are trying to read.
#
# Usage:
#   scripts/profile_cpu.sh --model <path.gguf> [--mode prefill|decode|both]
#                          [--cores 0-15] [--device cpu] [--tool auto|perf|samply]
#                          [--prompt 512] [--decode 128]
#
# Symbols require a non-stripped binary with frame pointers — cera's release
# profile strips (see Cargo.toml). Prefer `just profile-cpu <model>`, which
# builds exactly that binary and then runs this script.
#
# Building by hand takes one more flag than it looks: env RUSTFLAGS *replaces*
# `.cargo/config.toml`'s `[target.*] rustflags`, so passing only the frame
# -pointer flag drops this host's `target-cpu` and profiles a differently-tuned
# binary than `just release` ships. Re-state it:
#   CARGO_PROFILE_RELEASE_STRIP=false \
#   RUSTFLAGS='-C force-frame-pointers=yes -C target-cpu=x86-64-v3' \
#     cargo build --release -p cera-cli
#   # macOS: use `-C target-cpu=native`; aarch64-linux: omit target-cpu.
set -euo pipefail

MODEL=""
BACKEND="cpu"
MODE="both"
CORES=""
PROMPT=512
DECODE=128
TOOL="auto"
BIN="./target/release/cera"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="${2:?--model requires a value}"; shift 2 ;;
    --bin)      BIN="${2:?--bin requires a value}"; shift 2 ;;
    --device)   BACKEND="${2:?--device requires a value}"; shift 2 ;;
    --mode)     MODE="${2:?--mode requires a value}"; shift 2 ;;
    --cores)    CORES="${2:?--cores requires a value}"; shift 2 ;;
    --prompt)   PROMPT="${2:?--prompt requires a value}"; shift 2 ;;
    --decode)   DECODE="${2:?--decode requires a value}"; shift 2 ;;
    --tool)     TOOL="${2:?--tool requires a value}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model <path.gguf> is required" >&2; exit 2; }
# -e, not -f: the CLI takes a LeapBundle *directory* as a model source too.
[[ -e "$MODEL" ]] || { echo "error: model not found: $MODEL" >&2; exit 2; }
[[ -x "$BIN" ]]   || { echo "error: $BIN missing — cargo build --release -p cera-cli" >&2; exit 2; }
case "$MODE" in prefill|decode|both) ;; *) echo "--mode must be prefill|decode|both" >&2; exit 2 ;; esac
case "$TOOL" in auto|perf|samply) ;; *) echo "--tool must be auto|perf|samply" >&2; exit 2 ;; esac

OS="$(uname -s)"

# Default core pinning: the first half of the logical CPUs, which on an SMT part
# is one thread per physical core. Explicit --cores overrides.
#
# Floor the half at 1: on a single-core host or a 1-CPU container `NPROC / 2` is
# 0, and the naive `0-$((NPROC/2-1))` builds the string `0--1`, which taskset
# rejects.
if [[ -z "$CORES" && "$OS" == "Linux" ]]; then
  NPROC="$(nproc)"
  HALF=$(( NPROC / 2 ))
  (( HALF < 1 )) && HALF=1
  CORES="0-$(( HALF - 1 ))"
fi

# A stripped binary profiles as a wall of hex addresses. Say so up front rather
# than after a 20-second sample.
if command -v nm >/dev/null 2>&1 && ! nm "$BIN" >/dev/null 2>&1; then
  echo "warning: $BIN has no symbol table (stripped) — the report will be addresses only." >&2
  echo "         rebuild with 'just profile-cpu' (see the header for the raw cargo invocation)." >&2
fi

# Pick the profiler. `perf` is preferred because it reports top symbols to the
# terminal, which is what makes a before/after diff readable in a PR; samply is
# the portable fallback and opens a flamegraph UI instead.
if [[ "$TOOL" == "auto" ]]; then
  if [[ "$OS" == "Linux" ]] && command -v perf >/dev/null 2>&1; then TOOL="perf"
  elif command -v samply >/dev/null 2>&1; then TOOL="samply"
  else
    echo "error: no profiler found. Install one:" >&2
    echo "  perf:   sudo apt install linux-tools-common linux-tools-\$(uname -r)" >&2
    echo "  samply: cargo install samply" >&2
    exit 2
  fi
fi

# An explicit --tool skipped every check auto-select does, so a typo'd or
# absent profiler surfaced as a bare "command not found" from inside the run
# — after the build, and with no hint about which of the two is missing.
if [[ "$TOOL" == "perf" && "$OS" != "Linux" ]]; then
  echo "error: perf is Linux-only (this is $OS); use --tool samply." >&2
  exit 2
fi
command -v "$TOOL" >/dev/null 2>&1 || {
  echo "error: --tool $TOOL is not on PATH. Install it:" >&2
  echo "  perf:   sudo apt install linux-tools-common linux-tools-\$(uname -r)" >&2
  echo "  samply: cargo install samply" >&2
  exit 2
}

# Both perf and samply need perf_event_open, which is gated by this sysctl.
if [[ "$OS" == "Linux" && -r /proc/sys/kernel/perf_event_paranoid ]]; then
  PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
  if (( PARANOID > 1 )); then
    echo "error: /proc/sys/kernel/perf_event_paranoid is $PARANOID; user-space sampling needs <= 1." >&2
    echo "  sudo sysctl -w kernel.perf_event_paranoid=1" >&2
    echo "  (persist: echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf)" >&2
    exit 2
  fi
fi

PIN=()
if [[ -n "$CORES" && "$OS" == "Linux" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    PIN=(taskset -c "$CORES")
  else
    # Minimal containers often ship without util-linux. Unpinned sampling is
    # noisier across a big.LITTLE or boost-happy machine, but it still profiles
    # — refusing to run at all would be the worse trade.
    echo "note: taskset not found; running unpinned (hotspots may smear across cores)." >&2
    CORES=""
  fi
elif [[ "$OS" != "Linux" ]]; then
  # Only worth saying if pinning was actually asked for — on macOS the default
  # is unpinned anyway, and an unconditional note is just noise on every run.
  # Spelled as if/then, not `[[ … ]] && echo`: under `set -e` that idiom exits
  # the script when the test is false unless another command follows it, which
  # makes the line above load-bearing for reasons no reader would guess.
  if [[ -n "$CORES" ]]; then
    echo "note: core pinning is Linux-only (taskset); running unpinned on $OS." >&2
  fi
  CORES=""
fi

# Prefill and decode are separated by starving the other phase: --max-tokens 1
# leaves essentially only prefill, and a short prompt leaves essentially only
# decode. Neither is perfectly pure, but each is dominated by its phase.
run_one() {
  local label="$1" prompt="$2" maxtok="$3"
  local out="perf-cera-${label}"

  echo
  echo "======================================================================"
  echo "==> $label profile: device=$BACKEND cores=${CORES:-all} prompt=$prompt decode=$maxtok"
  echo "======================================================================"

  local bench=("$BIN" bench --model "$MODEL" --device "$BACKEND"
               --prompt-tokens "$prompt" --max-tokens "$maxtok"
               --runs 3 --warmup 1 --no-cache)

  # Delete any previous capture first. If the recorder then fails, we must not
  # fall through and report the *last* run's samples as if they were this run's
  # — a profiler that silently shows stale data defeats the entire point of the
  # script, which is to make a perf claim falsifiable.
  rm -f "${out}.data" "${out}.json.gz"

  if [[ "$TOOL" == "perf" ]]; then
    # -g call graphs (needs frame pointers); -- everything after is the workload.
    if ! perf record -g --freq 999 -o "${out}.data" -- \
         "${PIN[@]+"${PIN[@]}"}" "${bench[@]}" >"${out}.log" 2>&1; then
      echo "error: perf record failed — last lines of ${out}.log:" >&2
      tail -5 "${out}.log" >&2
      return 1
    fi

    echo
    echo "==> top symbols (self time)"
    # `|| true` because `head` closing the pipe SIGPIPEs its producers, and
    # under `set -o pipefail` that non-zero status would trip `set -e` and kill
    # the script — which in `--mode both` meant the decode profile never ran.
    { perf report -i "${out}.data" --stdio --sort symbol --percent-limit 0.5 2>/dev/null \
        | grep -v '^#' | grep -v '^$' | head -30; } || true
    echo
    echo "    full report: perf report -i ${out}.data"
  else
    if ! samply record --save-only -o "${out}.json.gz" -- \
         "${PIN[@]+"${PIN[@]}"}" "${bench[@]}" >"${out}.log" 2>&1; then
      echo "error: samply record failed — last lines of ${out}.log:" >&2
      tail -5 "${out}.log" >&2
      return 1
    fi
    echo "    saved: ${out}.json.gz  (view: samply load ${out}.json.gz)"
  fi
}

echo "==> cera CPU tier: $("$BIN" cpu 2>/dev/null || echo unknown)"
echo "==> profiler: $TOOL"

case "$MODE" in
  prefill) run_one prefill "$PROMPT" 1 ;;
  decode)  run_one decode 32 "$DECODE" ;;
  both)    run_one prefill "$PROMPT" 1; run_one decode 32 "$DECODE" ;;
esac

echo
echo "==> done. A kernel change should MOVE the top symbols above, not just the tok/s."

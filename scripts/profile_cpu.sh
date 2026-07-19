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
#                          [--prompt 512] [--decode 128] [--duration 20]
#
# Symbols require a non-stripped binary with frame pointers — cera's release
# profile strips (see Cargo.toml), so build the binary for profiling with:
#   CARGO_PROFILE_RELEASE_STRIP=false RUSTFLAGS='-C force-frame-pointers=yes' \
#     cargo build --release -p cera-cli
set -euo pipefail

MODEL=""
BACKEND="cpu"
MODE="both"
CORES=""
PROMPT=512
DECODE=128
DURATION=20
TOOL="auto"
BIN="./target/release/cera"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2"; shift 2 ;;
    --bin)      BIN="$2"; shift 2 ;;
    --device)   BACKEND="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --cores)    CORES="$2"; shift 2 ;;
    --prompt)   PROMPT="$2"; shift 2 ;;
    --decode)   DECODE="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --tool)     TOOL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model <path.gguf> is required" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL" >&2; exit 2; }
[[ -x "$BIN" ]]   || { echo "error: $BIN missing — cargo build --release -p cera-cli" >&2; exit 2; }
case "$MODE" in prefill|decode|both) ;; *) echo "--mode must be prefill|decode|both" >&2; exit 2 ;; esac

OS="$(uname -s)"

# Default core pinning: the first half of the logical CPUs, which on an SMT part
# is one thread per physical core. Explicit --cores overrides.
if [[ -z "$CORES" && "$OS" == "Linux" ]]; then
  NPROC="$(nproc)"
  CORES="0-$(( NPROC / 2 - 1 ))"
fi

# A stripped binary profiles as a wall of hex addresses. Say so up front rather
# than after a 20-second sample.
if command -v nm >/dev/null 2>&1 && ! nm "$BIN" >/dev/null 2>&1; then
  echo "warning: $BIN has no symbol table (stripped) — the report will be addresses only." >&2
  echo "         rebuild with CARGO_PROFILE_RELEASE_STRIP=false RUSTFLAGS='-C force-frame-pointers=yes'" >&2
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
  PIN=(taskset -c "$CORES")
elif [[ "$OS" != "Linux" ]]; then
  echo "note: core pinning is Linux-only (taskset); running unpinned on $OS." >&2
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

  if [[ "$TOOL" == "perf" ]]; then
    # -g call graphs (needs frame pointers); --  everything after is the workload.
    perf record -g --freq 999 -o "${out}.data" -- \
      "${PIN[@]}" "${bench[@]}" >/dev/null 2>&1 || true

    echo
    echo "==> top symbols (self time)"
    perf report -i "${out}.data" --stdio --sort symbol --percent-limit 0.5 2>/dev/null \
      | grep -v '^#' | grep -v '^$' | head -30
    echo
    echo "    full report: perf report -i ${out}.data"
  else
    samply record --save-only -o "${out}.json.gz" -- \
      "${PIN[@]}" "${bench[@]}" >/dev/null 2>&1 || true
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

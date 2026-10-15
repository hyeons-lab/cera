#!/usr/bin/env bash
# Phase 0 section 2 baselines: one fresh cera process per measurement, each
# wrapped in hwinfo_sm.py so package power is sampled over that process only.
#   baselines.sh matrix   two interleaved rounds of {Q4_0,Q8_0} x {gpu,cpu} x prompt {512,2048}
#   baselines.sh decode   decode-dominant runs (16-token prompt, 512 tokens) for decode tok/J
# Paths come from common.sh; override them through the environment.
set -eo pipefail
case "${1:-}" in matrix|decode) ;; *) echo "usage: baselines.sh matrix|decode" >&2; exit 2 ;; esac  # before common.sh touches anything
source "$(dirname "$0")/common.sh"
need_file "$CERA" "cera.exe built with --features gpu"
for q in Q4_0 Q8_0; do need_file "$(model $q)" "model"; done  # before the first measurement

measure() {  # measure <tag> <dev> <quant> <prompt_tokens> <max_tokens> <runs>
  local m
  m="$(model "$3")"
  echo "=== $1 $(date +%H:%M:%S)"
  sampled "$1" --match "CPU Package Power" --match "Total System Power" -- \
    "$CERA" bench --model "$m" --device "$2" --prompt-tokens "$4" --max-tokens "$5" --runs "$6"
  sleep 20  # let package power settle between measurements
}

case "${1:-}" in
  matrix)
    for round in 1 2; do for p in 512 2048; do for q in Q4_0 Q8_0; do for dev in gpu cpu; do
      measure "r${round}_${dev}_${q}_p$p" "$dev" "$q" "$p" 128 20
    done; done; done; done ;;
  decode)
    for q in Q4_0 Q8_0; do for dev in gpu cpu; do
      measure "dec_${dev}_${q}" "$dev" "$q" 16 512 10
    done; done ;;
esac

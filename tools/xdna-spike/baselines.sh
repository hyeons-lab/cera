#!/usr/bin/env bash
# Phase 0 section 2 baselines (run from WSL; every measured program runs natively
# on Windows). One fresh cera process per measurement, each wrapped in
# hwinfo_sm.py so package power is sampled over that process only.
#   baselines.sh matrix   two interleaved rounds of {Q4_0,Q8_0} x {gpu,cpu} x prompt {512,2048}
#   baselines.sh decode   decode-dominant runs (16-token prompt, 512 tokens) for decode tok/J
# Override the paths below through the environment.
set -eo pipefail
PY="${PY:-/mnt/c/Users/iamst/AppData/Local/Programs/Python/Python311/python.exe}"
SM="${SM:-C:\\Users\\iamst\\development-windows\\cera\\tools\\xdna-spike\\hwinfo_sm.py}"
CERA="${CERA:-C:\\Users\\iamst\\cera-target-gpu\\release\\cera.exe}"   # built with --features gpu
MD="${MD:-C:\\Users\\iamst\\.leap\\models}"
OUT_WIN="${OUT_WIN:-C:\\Users\\iamst\\xdna-spike\\baselines}"
OUT="${OUT:-$(wslpath -u "$OUT_WIN")}"
mkdir -p "$OUT"

measure() {  # measure <tag> <dev> <quant> <prompt_tokens> <max_tokens> <runs>
  local m="$MD\\Llama-3.2-1B-Instruct-$3\\Llama-3.2-1B-Instruct-$3.gguf"
  echo "=== $1 $(date +%H:%M:%S)"
  "$PY" "$SM" run --match "CPU Package Power" --match "Total System Power" --csv "$OUT_WIN\\$1.csv" -- \
    "$CERA" bench --model "$m" --device "$2" --prompt-tokens "$4" --max-tokens "$5" --runs "$6" 2>&1 \
    | tr -d '\r' | grep -E 'decode tok/s|prefill tok/s|wall time|Package Power|System Power|rror' | tee "$OUT/$1.log"
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
  *) echo "usage: baselines.sh matrix|decode" >&2; exit 2 ;;
esac

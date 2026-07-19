#!/usr/bin/env bash
# CI CPU benchmark: sweeps the LFM2 quants cera actually ships (Q4_0/Q4_K_M/Q8_0)
# and records prefill + decode tok/s in a stable, parseable form. Designed to run
# on the `ubuntu-24.04-arm` runner (Neoverse N2, FEAT_I8MM) — where the aarch64
# SMMLA GEMM kernels actually execute — and on `ubuntu-latest` (x86_64) as the
# desktop baseline. Absolute tok/s on shared CI runners drifts run-to-run; the
# durable signals are (a) the resolved CPU tier and (b) relative movement across
# commits for the same runner + quant.
#
# Inputs (env):
#   BIN            path to the release `cera` binary        (default ./target/release/cera)
#   ARCH_LABEL     label for this runner in output          (default: uname -m)
#   BUNDLE_ID      LeapBundles model id                     (default LFM2.5-350M)
#   QUANTS         space-separated quant labels             (default "Q4_0 Q4_K_M Q8_0")
#   PROMPT_TOKENS  prefill length                           (default 512)
#   MAX_TOKENS     decode length                            (default 32)
#   RUNS           measured runs                            (default 10)
#   WARMUP         warmup runs                              (default 2)
#   CONTEXT_SIZE   KV context window                        (default 8192)
#   CACHE_DIR      bundle download cache                    (default $HOME/.cache/cera)
#   OUT_JSON       results JSON path                        (default bench-results.json)
#
# Outputs:
#   $OUT_JSON                     github-action-benchmark "customBiggerIsBetter" array
#   $GITHUB_STEP_SUMMARY (if set) a markdown results table
#   stdout                        the raw bench logs + a plain results table
set -euo pipefail

BIN="${BIN:-./target/release/cera}"
ARCH_LABEL="${ARCH_LABEL:-$(uname -m)}"
BUNDLE_ID="${BUNDLE_ID:-LFM2.5-350M}"
QUANTS="${QUANTS:-Q4_0 Q4_K_M Q8_0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
MAX_TOKENS="${MAX_TOKENS:-32}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-2}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8192}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/cera}"
OUT_JSON="${OUT_JSON:-bench-results.json}"

# Resolved CPU tier — the single most important line for interpreting the numbers
# (e.g. `neon+i8mm` means the SMMLA GEMM path is live; `neon+dotprod` means it is not).
TIER="$("$BIN" cpu 2>/dev/null || echo 'cpu: tier=unknown')"
echo "runner arch: $ARCH_LABEL"
echo "$TIER"
echo "params: bundle=$BUNDLE_ID prompt_tokens=$PROMPT_TOKENS max_tokens=$MAX_TOKENS runs=$RUNS warmup=$WARMUP"
echo

# Markdown + JSON accumulators.
md_rows=""
json_entries=""
ok_count=0

emit_json() {
  # $1 metric ("prefill"/"decode"), $2 quant, $3 value(tok/s)
  local name="$1 ${BUNDLE_ID} $2 (${ARCH_LABEL})"
  local entry
  entry=$(printf '  {"name": "%s", "unit": "tok/s", "value": %s}' "$name" "$3")
  if [ -z "$json_entries" ]; then json_entries="$entry"; else json_entries="$json_entries,
$entry"; fi
}

for quant in $QUANTS; do
  echo "=== bench $BUNDLE_ID $quant (cpu) ==="
  out=""
  if ! out=$("$BIN" bench \
      --bundle-id "$BUNDLE_ID" --quant "$quant" \
      --cache-dir "$CACHE_DIR" \
      --device cpu \
      --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" \
      --runs "$RUNS" --warmup "$WARMUP" --no-cache \
      --context-size "$CONTEXT_SIZE" 2>&1); then
    echo "$out"
    echo "FAILED: $BUNDLE_ID $quant"
    md_rows="${md_rows}| $quant | FAIL | FAIL |
"
    continue
  fi
  echo "$out"

  pre=$(echo "$out" | grep -E "^prefill tok/s:" | head -1 || true)
  dec=$(echo "$out" | grep -E "^decode tok/s:"  | head -1 || true)
  p_p50=$(echo "$pre" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  d_p50=$(echo "$dec" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')

  # A missing/unparseable p50 is a broken measurement, NOT a real 0 tok/s point.
  # Record FAIL and skip the JSON entry rather than poisoning the trend with a
  # fake 0 that looks like a catastrophic regression. (A genuine 0 from the model
  # still parses as "0" here and is emitted — only an *absent* number is dropped.)
  if [ -z "$p_p50" ] || [ -z "$d_p50" ]; then
    echo "PARSE FAILED: $BUNDLE_ID $quant (bench ran but prefill/decode p50 not found)"
    echo
    md_rows="${md_rows}| $quant | FAIL | FAIL |
"
    continue
  fi

  echo "  -> prefill p50=$p_p50 tok/s | decode p50=$d_p50 tok/s"
  echo
  md_rows="${md_rows}| $quant | $p_p50 | $d_p50 |
"
  emit_json "prefill" "$quant" "$p_p50"
  emit_json "decode" "$quant" "$d_p50"
  ok_count=$((ok_count + 1))
done

# github-action-benchmark customBiggerIsBetter array.
printf '[\n%s\n]\n' "$json_entries" > "$OUT_JSON"
echo "wrote $OUT_JSON"

# Plain table to stdout.
echo
echo "results ($ARCH_LABEL, $TIER):"
printf '| quant | prefill p50 tok/s | decode p50 tok/s |\n'
printf '|---|---|---|\n'
printf '%s' "$md_rows"

# Markdown summary for the Actions UI.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### CPU bench — $BUNDLE_ID ($ARCH_LABEL)"
    echo
    echo "\`$TIER\` · prompt_tokens=$PROMPT_TOKENS · max_tokens=$MAX_TOKENS · runs=$RUNS"
    echo
    echo "| quant | prefill p50 tok/s | decode p50 tok/s |"
    echo "|---|---|---|"
    printf '%s' "$md_rows"
  } >> "$GITHUB_STEP_SUMMARY"
fi

# Fail the job if NOTHING was measured — otherwise a run where every quant failed
# (e.g. a model-download outage) would exit 0 and report green having measured
# nothing. JSON + summary are already written above, so the artifact still
# uploads for debugging. A partial run (≥1 quant OK) still succeeds.
if [ "$ok_count" -eq 0 ]; then
  echo "ERROR: no quant produced a valid measurement — failing the benchmark job." >&2
  exit 1
fi

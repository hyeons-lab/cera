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
#   LLAMA_BENCH    path to an upstream `llama-bench` binary  (default: unset → no
#                  comparison). When set and executable, each model is *also* run
#                  through llama.cpp on the same GGUF, and the summary table gains
#                  `llama` + `gap (llama/cera)` columns. Best-effort: a missing
#                  binary, an unloadable model, or a parse failure degrades that
#                  cell to `n/a` and never fails the job. The tracked JSON stays
#                  cera-only (llama is a human-readable reference overlay).
#   LLAMA_THREADS  thread count for llama-bench              (default: unset → llama-bench's default, the physical-core count)
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
LLAMA_BENCH="${LLAMA_BENCH:-}"
LLAMA_THREADS="${LLAMA_THREADS:-}"

# Whether the llama.cpp comparison is active for this run.
LLAMA_ON=0
if [ -n "$LLAMA_BENCH" ] && [ -x "$LLAMA_BENCH" ]; then
  LLAMA_ON=1
fi

# t/s of the llama-bench table row whose `test` column exactly equals $2, read
# from the multi-line table in $1. `| … | <test> | <t/s> ± <std> |` → the test
# label is the 2nd-to-last `|`-field and the t/s the last populated one. Anchored
# on the exact field (not a substring of the whole row) so `pp512` can't match a
# `pp512+tg32` row or a model-name column. Empty if absent.
llama_row_ts() {
  # Here-string, not `echo "$1" | awk`: awk `exit`s on the first match, which
  # would SIGPIPE the writer of a pipe — harmless for a tiny table but the same
  # class the `find … | head` call guards against. A here-string has no pipe.
  awk -F'|' -v want="$2" '
    NF < 3 { next }                                  # skip non-table lines (NF<3 ⇒ $(NF-2) invalid)
    { lbl=$(NF-2); gsub(/ /,"",lbl) }
    lbl==want { split($(NF-1),a,"±"); gsub(/ /,"",a[1]); print a[1]; exit }' <<<"$1"
}

# Run upstream llama-bench on one GGUF and echo "PREFILL DECODE" tok/s (both
# means over `-r RUNS` reps), or "n/a n/a" on any failure. Best-effort — never
# `exit`s (a build/load/timeout/parse failure degrades to n/a; the cera job goes
# on).
#
# `-p P` gives a standalone `ppP` row (prefill); `-pg P,G` gives a `ppP+tgG` row
# (prompt-then-generate, reported as *combined* throughput). Decode is derived
# from the two so it's measured at the SAME KV depth cera's is — cera decodes G
# tokens *after* the P-token prefill, whereas llama-bench's bare `-n G` generates
# from an EMPTY context and would understate cera's post-prefill decode:
#   gen_rate = G / (total_time − pp_time) = G / ((P+G)/combo − P/pp).
# (llama-bench also emits its default `tg128` row here; we ignore it by matching
# the exact `ppP` / `ppP+tgG` test labels.)
llama_bench_model() {
  local gguf="$1"
  local threads_arg=()
  [ -n "$LLAMA_THREADS" ] && threads_arg=(-t "$LLAMA_THREADS")
  local out
  # `-n 0` suppresses llama-bench's default standalone `tg128` test (we don't use
  # it), so it isn't generated and timed for nothing on the shared runner.
  if ! out=$(timeout 900 "$LLAMA_BENCH" -m "$gguf" \
      -p "$PROMPT_TOKENS" -n 0 -pg "${PROMPT_TOKENS},${MAX_TOKENS}" -ngl 0 -r "$RUNS" \
      "${threads_arg[@]}" 2>/dev/null); then
    echo "n/a n/a"
    return
  fi
  local pp combo dec
  pp=$(llama_row_ts "$out" "pp${PROMPT_TOKENS}")
  combo=$(llama_row_ts "$out" "pp${PROMPT_TOKENS}+tg${MAX_TOKENS}")
  dec=$(awk -v p="$PROMPT_TOKENS" -v g="$MAX_TOKENS" -v pp="$pp" -v combo="$combo" 'BEGIN{
    if (pp+0>0 && combo+0>0) {
      gt=(p+g)/combo - p/pp;                 # total_time − prompt_time
      # Sanity-bound the derived rate: `combo` and `pp` are independent samples,
      # so a near-zero-but-positive `gt` from run-to-run noise would produce an
      # absurd decode rate. Prompt processing is batched and always outpaces
      # single-token decode, so a derived decode ≥ the prefill rate is noise →
      # report n/a rather than a garbage number.
      if (gt>0) { r=g/gt; if (r < pp+0) { printf "%.2f", r; exit } }
    }
    printf "n/a"
  }')
  echo "${pp:-n/a} ${dec:-n/a}"
}

# "gap" = llama / cera, one decimal + `x`. Both sides are *mean* tok/s (see the
# call site — cera's mean, not p50, so the ratio is like-for-like against
# llama-bench's mean). `n/a` unless both are positive numbers. A non-numeric
# cell ("n/a") coerces to 0 under `+0`, so `>0` alone rejects it — simpler and
# more portable than an `x+0==x` numeric-string probe, and it also rejects a
# parsed-but-nonpositive 0/negative that a bare `==` check would let through.
gap() {
  awk -v c="$1" -v l="$2" 'BEGIN{
    if (c+0>0 && l+0>0) printf "%.1fx", l/c; else printf "n/a"
  }'
}

# Resolved CPU tier — the single most important line for interpreting the numbers
# (e.g. `neon+i8mm` means the SMMLA GEMM path is live; `neon+dotprod` means it is not).
TIER="$("$BIN" cpu 2>/dev/null || echo 'cpu: tier=unknown')"
echo "runner arch: $ARCH_LABEL"
echo "$TIER"
echo "params: bundle=$BUNDLE_ID prompt_tokens=$PROMPT_TOKENS max_tokens=$MAX_TOKENS runs=$RUNS warmup=$WARMUP"
if [ "$LLAMA_ON" -eq 1 ]; then
  echo "llama.cpp comparison: ON ($LLAMA_BENCH) — gap = llama/cera (higher ⇒ cera further behind)"
else
  echo "llama.cpp comparison: off (set LLAMA_BENCH to enable)"
fi
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

# A FAIL row with the right column count for the active table layout.
fail_row() {
  if [ "$LLAMA_ON" -eq 1 ]; then
    printf '| %s | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |\n' "$1"
  else
    printf '| %s | FAIL | FAIL |\n' "$1"
  fi
}

# Table header + separator for the active layout.
if [ "$LLAMA_ON" -eq 1 ]; then
  THEAD="| quant | cera prefill | llama prefill | gap (p) | cera decode | llama decode | gap (d) |"
  TSEP="|---|---|---|---|---|---|---|"
else
  THEAD="| quant | prefill p50 tok/s | decode p50 tok/s |"
  TSEP="|---|---|---|"
fi

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
    md_rows="${md_rows}$(fail_row "$quant")
"
    continue
  fi
  echo "$out"

  pre=$(echo "$out" | grep -E "^prefill tok/s:" | head -1 || true)
  dec=$(echo "$out" | grep -E "^decode tok/s:"  | head -1 || true)
  # p50 is the tracked/displayed metric; mean is used only for the llama gap, so
  # both sides of the ratio summarize their run distribution the same way
  # (llama-bench reports the mean).
  p_p50=$(echo "$pre" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  d_p50=$(echo "$dec" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  p_mean=$(echo "$pre" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
  d_mean=$(echo "$dec" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')

  # A missing/unparseable p50 is a broken measurement, NOT a real 0 tok/s point.
  # Record FAIL and skip the JSON entry rather than poisoning the trend with a
  # fake 0 that looks like a catastrophic regression. (A genuine 0 from the model
  # still parses as "0" here and is emitted — only an *absent* number is dropped.)
  if [ -z "$p_p50" ] || [ -z "$d_p50" ]; then
    echo "PARSE FAILED: $BUNDLE_ID $quant (bench ran but prefill/decode p50 not found)"
    echo
    md_rows="${md_rows}$(fail_row "$quant")
"
    continue
  fi

  echo "  -> cera prefill p50=$p_p50 tok/s | decode p50=$d_p50 tok/s"
  emit_json "prefill" "$quant" "$p_p50"
  emit_json "decode" "$quant" "$d_p50"
  ok_count=$((ok_count + 1))

  # llama.cpp on the same GGUF (best-effort; the cera measurement above is the
  # tracked one and already counted). Locate the file cera just downloaded.
  if [ "$LLAMA_ON" -eq 1 ]; then
    # `|| true`: on the (unexpected) >1 match, `head` closes the pipe and `find`
    # can take SIGPIPE → non-zero under pipefail, which would trip `set -e` and
    # kill the job — the one thing the llama overlay must never do.
    gguf=$(find "$CACHE_DIR" -name "${BUNDLE_ID}-${quant}.gguf" -type f 2>/dev/null | head -1 || true)
    if [ -n "$gguf" ]; then
      # Capture into a var + split with parameter expansion, not `read < <(…)`:
      # `read` returns non-zero at EOF, which under `set -e` would kill the job if
      # `llama_bench_model` ever produced no line. `llama_bench_model` always
      # echoes exactly "PREFILL DECODE" and exits 0, so this is set -e-safe.
      llama_out=$(llama_bench_model "$gguf")
      l_pre=${llama_out%% *}
      l_dec=${llama_out##* }
      # gap uses cera's *mean* (matching llama-bench's mean); the table still
      # displays cera p50 (the tracked metric).
      p_gap=$(gap "$p_mean" "$l_pre")
      d_gap=$(gap "$d_mean" "$l_dec")
      echo "  -> llama prefill=$l_pre | decode=$l_dec tok/s | gap (llama/cera, means) prefill=$p_gap decode=$d_gap"
      md_rows="${md_rows}| $quant | $p_p50 | $l_pre | $p_gap | $d_p50 | $l_dec | $d_gap |
"
    else
      echo "  -> llama: no GGUF matching ${BUNDLE_ID}-${quant}.gguf under $CACHE_DIR (skipping)"
      md_rows="${md_rows}| $quant | $p_p50 | n/a | n/a | $d_p50 | n/a | n/a |
"
    fi
  else
    md_rows="${md_rows}| $quant | $p_p50 | $d_p50 |
"
  fi
  echo
done

# github-action-benchmark customBiggerIsBetter array.
printf '[\n%s\n]\n' "$json_entries" > "$OUT_JSON"
echo "wrote $OUT_JSON"

# One caption, shared by the stdout table and the Actions summary, so the gap
# direction + methodology can't be read off one without the other.
LLAMA_CAPTION="vs upstream llama.cpp on the same GGUF · cera columns are p50; gap = llama_mean / cera_mean (higher ⇒ cera further behind), so it won't exactly equal llama÷(shown p50) · decode is depth-matched to cera's post-prefill decode (derived from llama-bench -pg)"

# Plain table to stdout.
echo
echo "results ($ARCH_LABEL, $TIER):"
[ "$LLAMA_ON" -eq 1 ] && echo "$LLAMA_CAPTION"
printf '%s\n%s\n' "$THEAD" "$TSEP"
printf '%s' "$md_rows"

# Markdown summary for the Actions UI.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### CPU bench — $BUNDLE_ID ($ARCH_LABEL)"
    echo
    echo "\`$TIER\` · prompt_tokens=$PROMPT_TOKENS · max_tokens=$MAX_TOKENS · runs=$RUNS"
    if [ "$LLAMA_ON" -eq 1 ]; then
      echo
      echo "$LLAMA_CAPTION"
    fi
    echo
    echo "$THEAD"
    echo "$TSEP"
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

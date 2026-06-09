#!/usr/bin/env bash
# Compare KV prefix cache modes — cold (no cache) vs warm (in-process memory)
# vs cold-tier (across-process disk hit, the mobile-restart scenario).
#
# Usage:
#   ./benchmarks/cache_compare.sh [<gguf-path>] [<device>]
#     device defaults to `metal`; pass `cpu` to benchmark the CPU backend.
#     CPU got prefix-cache integration in the same PR that added this
#     option; both backends now participate.

set -euo pipefail

CERA="${CERA:-$(pwd)/target/release/cera}"
MODEL="${1:-$HOME/.leap/models/LFM2.5-Audio-1.5B-Q4_0/LFM2.5-Audio-1.5B-Q4_0.gguf}"
DEVICE="${2:-metal}"
CACHE="${CACHE:-/tmp/cera-cache-compare}"

if [[ ! -x "$CERA" ]]; then
  echo "cera binary not found at $CERA — build with:" >&2
  echo "  cargo build -p cera-cli --release --features metal" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

# Long enough that the prefill is measurable; short enough to run quickly.
PROMPT=$(python3 -c '
import sys
text = "In computer science, a cache is a hardware or software component that stores data. " * 30
sys.stdout.write(text)')

rm -rf "$CACHE"

run_no_cache() {
  "$CERA" run -m "$MODEL" --no-cache --device "$DEVICE" \
    --prompt "$PROMPT" --max-tokens 1 2>&1 \
    | grep -E "Prefill|Prompt tokens" | tail -2
}

run_with_disk_cache() {
  "$CERA" run -m "$MODEL" --cache-dir "$CACHE" --device "$DEVICE" \
    --prompt "$PROMPT" --max-tokens 1 2>&1 \
    | grep -E "Prefill|Prompt tokens" | tail -2
}

echo "## Cross-process disk-cache benchmark (device=$DEVICE)"
echo
echo "### Run 1: --no-cache (cold baseline)"
run_no_cache
echo
echo "### Run 2: --no-cache again (cold sanity)"
run_no_cache
echo
echo "### Run 3: --cache-dir (cold + populates disk)"
run_with_disk_cache
echo
echo "### Run 4: --cache-dir (DISK HIT — fresh process, warm-cache empty)"
run_with_disk_cache
echo
echo "### Run 5: --cache-dir (disk hit sanity)"
run_with_disk_cache
echo
echo
echo "## In-process warm-cache benchmark (cera bench --runs 5, device=$DEVICE)"
echo
echo "### --no-cache (every iter cold)"
"$CERA" bench -m "$MODEL" --device "$DEVICE" --prompt-tokens 482 \
  --max-tokens 1 --runs 5 --warmup 0 --no-cache 2>&1 \
  | grep -E "prefill|decode"
echo
echo "### default (iter 1 cold, iters 2-5 warm hit)"
"$CERA" bench -m "$MODEL" --device "$DEVICE" --prompt-tokens 482 \
  --max-tokens 1 --runs 5 --warmup 0 2>&1 \
  | grep -E "prefill|decode"

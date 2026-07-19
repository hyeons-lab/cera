#!/usr/bin/env bash
# Fetch the small GGUF fixtures the batched-prefill parity tests need.
#
# These tests compare the batched-GEMM prefill against the sequential per-token
# path on a real model. That comparison cannot be faked with synthetic weights
# in-tree: the bugs it catches (layout, transpose, accumulation order, a gate
# that declines and leaves a stale buffer) only appear once real quantized
# tensors flow through the whole forward pass.
#
# The fixtures below are chosen to be CI-sized rather than representative —
# together ~310 MB and ~30 s of test time, against the multi-GB files the other
# parity tests use locally:
#
#   SmolLM-135M.Q4_0        88 MB   llama arch, GQA (9 heads / 3 kv), ctx 2048,
#                                   all projections Q4_0. Exercises the dense
#                                   transformer batched path on both the naive
#                                   and flash (>=256 token) branches.
#   LFM2.5-350M-Q4_K_M     219 MB   LFM2 hybrid, 82 Q4_K + 11 Q6_K tensors —
#                                   the K-quant GEMM kernels and their gates.
#
# Idempotent: a file whose checksum already matches is left alone, so this is
# safe to run on every CI invocation in front of a warm cache. Checksums are
# not optional — a truncated or silently-substituted fixture would otherwise
# surface as a parity *failure*, sending the reader after a kernel bug that
# does not exist. (Both were verified to download intact; some HF repos serve
# Xet-backed blobs that resolve to zero-filled files, which is exactly the
# failure mode this guards.)
#
# Usage:
#   scripts/fetch_test_models.sh [--dest <dir>]
#
# Default dest is `target/oracle/models`, where the parity tests look. Point
# `CERA_MODEL_ROOT` at the directory *containing* `target/` to use a checkout
# other than the current one (a git worktree, say).
set -euo pipefail

DEST="target/oracle/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="${2:?--dest requires a value}"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# name <TAB> sha256 <TAB> url
# Keep this table the single source of truth: the CI cache key is derived from
# a hash of this file, so editing a checksum or adding a row invalidates the
# cache automatically rather than relying on someone bumping a version suffix.
MANIFEST=$(cat <<'EOF'
SmolLM-135M.Q4_0.gguf	6429c98b87a4ee1ca12afe14a5d3e4658b4753c17192369485dcc51cbef9a898	https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q4_0.gguf
LFM2.5-350M-Q4_K_M.gguf	7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4	https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf
EOF
)

mkdir -p "$DEST"

# `shasum -a 256` on macOS, `sha256sum` on Linux — neither is present on both.
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "error: need sha256sum or shasum on PATH to verify fixtures" >&2
  exit 2
fi

fetched=0
cached=0

while IFS=$'\t' read -r name want url; do
  [[ -n "$name" ]] || continue
  path="$DEST/$name"

  if [[ -f "$path" ]]; then
    got="$(sha256_of "$path")"
    if [[ "$got" == "$want" ]]; then
      echo "==> $name: cached (checksum ok)"
      cached=$((cached + 1))
      continue
    fi
    echo "==> $name: checksum mismatch, refetching" >&2
    echo "    want $want" >&2
    echo "    got  $got" >&2
    rm -f "$path"
  fi

  echo "==> $name: downloading"
  # Download to a temp name and move only after the checksum passes, so an
  # interrupted run can never leave a half-file that a later run treats as
  # present-and-correct.
  tmp="$path.partial"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$tmp" "$url"

  got="$(sha256_of "$tmp")"
  if [[ "$got" != "$want" ]]; then
    rm -f "$tmp"
    echo "error: $name checksum mismatch after download" >&2
    echo "    want $want" >&2
    echo "    got  $got" >&2
    exit 1
  fi
  mv "$tmp" "$path"
  fetched=$((fetched + 1))
done <<< "$MANIFEST"

echo "==> fixtures ready in $DEST ($fetched downloaded, $cached cached)"

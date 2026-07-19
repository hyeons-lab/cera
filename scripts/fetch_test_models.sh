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
# together ~1.4 GB and ~55 s of test time, against the multi-GB files the other
# parity tests use locally:
#
# Chosen to cover one batched-GEMM weight dtype each — the dispatch in
# `gemm_preq` picks a different kernel per dtype, so a fixture set that misses
# one leaves that kernel untested:
#
#   TinyStories-LLaMA2-20M
#     -GQA.Q8_0             21 MB   llama arch, GQA (16 heads / 8 kv), ctx 2048,
#                                   vocab 32000, every projection Q8_0 —
#                                   `gemm_q8_0_q8_0`. Runs in ~1 s.
#   SmolLM-135M.Q4_0        88 MB   llama arch, GQA (9 heads / 3 kv), ctx 2048,
#                                   all projections Q4_0 — `gemm_q4_0_q8_0`.
#                                   ~8 s.
#   LFM2.5-230M-Q4_K_M     153 MB   LFM2 hybrid, 74 Q4_K + 9 Q6_K tensors —
#                                   the K-quant GEMM kernels and their gates.
#                                   ~16 s.
#
# Then one per remaining dense arch, since each has a distinct forward path the
# batched GEMM has to stay in step with. Larger, but the cache amortizes them:
#
#   qwen2-0_5b-instruct
#     -q8_0                531 MB   qwen2: Q/K/V projection biases. ~12 s.
#   Qwen3-0.6B-Q8_0        639 MB   qwen3: per-head Q/K RMSNorm (QK-norm) and a
#                                   decoupled head_dim. ~19 s.
#
# All Q8_0, and that is not incidental. The `q4_0` builds of these repos are
# not uniformly Q4_0 — Qwen2's carries Q4_1 `ffn_down`, Granite's carries 5
# Q4_1 plus a Q6_K. cera cannot dequantize Q4_1, so those files panic inside
# the kernel rather than declining cleanly; the same trap is why the
# Llama-3.2-1B fixture below is the Q8_0 build. Check the dtype census with
# `cera inspect` before adding any new fixture.
#
# Still missing: granite (scalar multipliers) and Llama-3 NORM rope with
# `rope_freqs` factors. Both exist only as ~1.3-2.7 GB Q8_0 builds.
#
# Each covers both the naive and the flash (>=256 token) branch, which is why
# every fixture needs a context length above 288.
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
TinyStories-LLaMA2-20M-GQA.Q8_0.gguf	86cd37850fa561d5ae2a368b9d85fcb87949c0468eda5a6da1cfab45469b1b9b	https://huggingface.co/mradermacher/TinyStories-LLaMA2-20M-256h-4l-GQA-GGUF/resolve/main/TinyStories-LLaMA2-20M-256h-4l-GQA.Q8_0.gguf
SmolLM-135M.Q4_0.gguf	6429c98b87a4ee1ca12afe14a5d3e4658b4753c17192369485dcc51cbef9a898	https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q4_0.gguf
LFM2.5-230M-Q4_K_M.gguf	7bbd90384d3deffe4c646ec9643b212802d32d4ce417c90a1ec9282100650062	https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_K_M.gguf
qwen2-0_5b-instruct-q8_0.gguf	834f4115ad5a836c9f17716b1577290fda96de3deb881ba45a4d5476fd202e96	https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q8_0.gguf
Qwen3-0.6B-Q8_0.gguf	9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031	https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
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

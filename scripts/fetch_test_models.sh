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
# together 6.3 GB and ~105 s of test time, against the multi-GB files the other
# parity tests use locally.
#
# Sizes here are decimal (MB = 10^6, GB = 10^9), matching what Hugging Face
# reports and what GitHub measures cache against. Worth stating because
# `du -sh` prints GiB while labelling it "G" — mixing the two is how an
# earlier revision of this header claimed 5.1 GB for a 5.5 GB set.
#
# Chosen to cover one batched-GEMM weight dtype each — the dispatch in
# `gemm_preq` picks a different kernel per dtype, so a fixture set that misses
# one leaves that kernel untested:
#
#   TinyStories-LLaMA2-20M
#     -GQA.Q8_0             21 MB   llama arch, GQA (16 heads / 8 kv), ctx 2048,
#                                   vocab 32000, every projection Q8_0 —
#                                   `gemm_q8_0_q8_0`. Runs in ~1 s.
#   SmolLM-135M.Q4_0        92 MB   llama arch, GQA (9 heads / 3 kv), ctx 2048,
#                                   all projections Q4_0 — `gemm_q4_0_q8_0`.
#                                   ~8 s.
#   LFM2.5-230M-Q4_K_M     153 MB   LFM2 hybrid, 74 Q4_K + 9 Q6_K tensors —
#                                   the K-quant GEMM kernels and their gates.
#                                   ~16 s.
#
# Then the dense-transformer fixtures. Mostly one per arch — each has a distinct
# forward path the batched GEMM has to stay in step with — plus a second Llama
# build, because arch coverage and *dtype* coverage are different axes: the Q8_0
# file exercises Llama-3 rope, the Q4_K_M one exercises the K-quant kernels on a
# dense model. Larger, but the cache amortizes them:
#
#   qwen2-0_5b-instruct
#     -q8_0                531 MB   qwen2: Q/K/V projection biases. ~12 s.
#   Qwen3-0.6B-Q8_0        639 MB   qwen3: per-head Q/K RMSNorm (QK-norm) and a
#                                   decoupled head_dim. ~19 s.
#   Llama-3.2-1B-Instruct
#     -Q8_0               1321 MB   llama: NORM rope with Llama-3 `rope_freqs`
#                                   frequency-scaling factors. ~14 s.
#   Llama-3.2-1B-Instruct
#     -Q4_K_M              808 MB   the dense-transformer K-quant path: 96 Q4_K
#                                   + 17 Q6_K. Needs a 256-divisible hidden size
#                                   (2048) — Q4_K_M on a 896-hidden model like
#                                   Qwen2-0.5B falls back to Q5_0, which cera
#                                   cannot load. ~16 s.
#   granite-3.1-2b-instruct
#     -Q8_0               2694 MB   granite: the four scalar multipliers
#                                   (embedding/residual/attention/logit), which
#                                   the batched path has to apply identically
#                                   to the per-token one. ~35 s.
#
# That is every architecture cera supports on the dense-transformer batched
# path, plus LFM2. The last two are large; they are here because the cache
# makes the steady-state cost a checksum pass, and because a scalar or a rope
# factor applied in one path and not the other is invisible without them.
#
# All Q8_0, and that is not incidental. The `q4_0` builds of these repos are
# not uniformly Q4_0 — Qwen2's carries a Q4_1 `ffn_down`, Granite's carries 5
# Q4_1 plus a Q6_K, and the Llama-3.2-1B one has Q4_1 in blocks 0/1. cera
# cannot dequantize Q4_1, and today the failure is a panic from inside a
# kernel rather than a clean rejection, so check the dtype census with
# `cera inspect` before adding any fixture:
#
#   cargo run --release -p cera-cli -- inspect --model <file> | grep -oE "\| Q[A-Z0-9_]+ \|" | sort | uniq -c
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
#   scripts/fetch_test_models.sh [--set core|all] [--dest <dir>]
#
# `--set core` (the default) fetches the three fixtures that cover both int8
# GEMM kernels and the K-quant path — 267 MB, what CI pulls on a PR.
# `--set all` adds the five dense-transformer fixtures for a total of 6.3 GB,
# which is what CI pulls on a main push.
#
# Default dest is `target/oracle/models`, where the parity tests look. Point
# `CERA_MODEL_ROOT` at the directory *containing* `target/` to use a checkout
# other than the current one (a git worktree, say) — the same variable the
# tests resolve fixtures through, so fetching and reading stay in agreement.
# `--dest` overrides both.
set -euo pipefail

DEST="${CERA_MODEL_ROOT:-.}/target/oracle/models"
SET="core"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="${2:?--dest requires a value}"; shift 2 ;;
    --set)  SET="${2:?--set requires a value}"; shift 2 ;;
    # Print the whole comment header rather than a hardcoded line range: the
    # range was '2,32p' and silently stopped mid-table as the header grew, so
    # --help documented a subset of the fixtures and looked complete.
    -h|--help) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# tier <TAB> name <TAB> sha256 <TAB> url
# Keep this table the single source of truth: the CI cache key is derived from
# a hash of this file, so editing a checksum or adding a row invalidates the
# cache automatically rather than relying on someone bumping a version suffix.
MANIFEST=$(cat <<'EOF'
core	TinyStories-LLaMA2-20M-GQA.Q8_0.gguf	86cd37850fa561d5ae2a368b9d85fcb87949c0468eda5a6da1cfab45469b1b9b	https://huggingface.co/mradermacher/TinyStories-LLaMA2-20M-256h-4l-GQA-GGUF/resolve/main/TinyStories-LLaMA2-20M-256h-4l-GQA.Q8_0.gguf
core	SmolLM-135M.Q4_0.gguf	6429c98b87a4ee1ca12afe14a5d3e4658b4753c17192369485dcc51cbef9a898	https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q4_0.gguf
core	LFM2.5-230M-Q4_K_M.gguf	7bbd90384d3deffe4c646ec9643b212802d32d4ce417c90a1ec9282100650062	https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_K_M.gguf
arch	qwen2-0_5b-instruct-q8_0.gguf	834f4115ad5a836c9f17716b1577290fda96de3deb881ba45a4d5476fd202e96	https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q8_0.gguf
arch	Qwen3-0.6B-Q8_0.gguf	9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031	https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
arch	Llama-3.2-1B-Instruct-Q8_0.gguf	432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3	https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
arch	Llama-3.2-1B-Instruct-Q4_K_M.gguf	6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83	https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
arch	granite-3.1-2b-instruct-Q8_0.gguf	883a1094022c40cb05f481fbe236b315cf2ca4f30ff84f8852aa3301e0edde72	https://huggingface.co/bartowski/granite-3.1-2b-instruct-GGUF/resolve/main/granite-3.1-2b-instruct-Q8_0.gguf
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

case "$SET" in core|all) ;; *) echo "--set must be core|all" >&2; exit 2 ;; esac

while IFS=$'\t' read -r tier name want url; do
  [[ -n "$name" ]] || continue
  # `core` is the per-PR set: both int8 GEMM kernels plus the K-quant path, at
  # 267 MB. `all` adds the five dense-transformer fixtures (6.3 GB), worth
  # caching but not worth pulling on every PR — GitHub allows 10 GB of cache
  # per repo in total, and a 6.3 GB entry would evict the rust and gradle caches
  # that every other job depends on.
  if [[ "$SET" == "core" && "$tier" != "core" ]]; then
    continue
  fi
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
  # Remove the fragment if curl dies mid-transfer. `set -e` aborts the script on
  # a failed download, which skipped the cleanup below and left the .partial on
  # disk — never mistaken for the real file (that was the point of the suffix),
  # but repeated failures accumulated multi-hundred-MB fragments. Seen for real:
  # a 2.5 GB fetch died at 126 MB with an SSL record error.
  if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$tmp" "$url"; then
    rm -f "$tmp"
    echo "error: $name download failed" >&2
    exit 1
  fi

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

echo "==> fixtures ready in $DEST [set=$SET] ($fetched downloaded, $cached cached)"

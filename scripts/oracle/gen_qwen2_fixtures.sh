#!/usr/bin/env bash
# Generate the committed Qwen2 oracle fixtures. Run vendor_llama_cpp.sh first.
# Re-run this whenever the prompt set or the pinned llama.cpp SHA changes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="${REPO_ROOT}/target/oracle/llama.cpp/build/bin"
MODEL="${REPO_ROOT}/target/oracle/models/qwen2-0_5b-instruct-q4_0.gguf"
OUT_DIR="${REPO_ROOT}/cera/tests/fixtures/oracle/qwen2-0_5b"
LLAMA_SHA="$(git -C "${REPO_ROOT}/target/oracle/llama.cpp" rev-parse HEAD)"

# Small fixed prompt set — chosen to exercise distinct code paths:
#   factual English, code, non-ASCII/multilingual (byte-level BPE), digits
#   (Qwen2 single-digit pretokenizer split).
PROMPTS=(
  "The capital of France is"
  $'def add(a, b):\n    return'
  "Hola, ¿cómo estás?"
  "1 + 2 + 3 ="
)

python3 "${REPO_ROOT}/scripts/oracle/gen_text_oracle.py" \
  --bin-dir "${BIN_DIR}" \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --llama-sha "${LLAMA_SHA}" \
  --n-predict 16 \
  "${PROMPTS[@]}"

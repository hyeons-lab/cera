#!/usr/bin/env bash
# Generate committed oracle fixtures for one text model. Run vendor_llama_cpp.sh
# first, and have the GGUF under target/oracle/models/.
#
# Usage: gen_fixtures.sh <model-gguf-basename> <fixture-subdir>
#   e.g. gen_fixtures.sh qwen2-0_5b-instruct-q8_0.gguf qwen2-0_5b
#        gen_fixtures.sh Qwen3-0.6B-Q8_0.gguf           qwen3-0_6b
#
# Re-run whenever the prompt set or the pinned llama.cpp SHA changes.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <model-gguf-basename> <fixture-subdir>" >&2
  exit 2
fi
MODEL_BASENAME="$1"
SUBDIR="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="${REPO_ROOT}/target/oracle/llama.cpp/build/bin"
MODEL="${REPO_ROOT}/target/oracle/models/${MODEL_BASENAME}"
OUT_DIR="${REPO_ROOT}/cera/tests/fixtures/oracle/${SUBDIR}"
LLAMA_SHA="$(git -C "${REPO_ROOT}/target/oracle/llama.cpp" rev-parse HEAD)"

# Small fixed prompt set — chosen to exercise distinct code paths:
#   factual English, code (indentation → whitespace pretokenization), non-ASCII /
#   multilingual (byte-level BPE), digits (single-digit pretokenizer split).
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

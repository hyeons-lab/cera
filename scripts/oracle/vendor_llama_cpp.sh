#!/usr/bin/env bash
# Vendor + build upstream llama.cpp at a pinned SHA, for use as a numeric
# reference oracle when validating cera's text-model forward pass.
#
# The build is a LOCAL tool only — it is NOT committed. Golden fixtures generated
# from it ARE committed (under cera/tests/fixtures/oracle/), and CI diffs against
# those, so CI needs neither llama.cpp nor this script.
#
# Scope: TEXT models (Qwen2/Qwen3) use upstream ggml-org/llama.cpp, pinned below.
# LFM2.5-Audio is NOT in upstream and is out of scope here — see
# devlog/plans/000150-01-llama-qwen2-support.md for the (audio-only) fork note.
set -euo pipefail

# Pinned upstream commit (ggml-org/llama.cpp, branch `master`). Bump deliberately.
LLAMA_CPP_SHA="d8a24ccee207a1ff24c513fe1c7d3222b3ccd837"
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp.git"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/target/oracle/llama.cpp"   # target/ is gitignored
JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc || echo 4)"

echo "[oracle] vendoring llama.cpp @ ${LLAMA_CPP_SHA}"
echo "[oracle] build root: ${BUILD_ROOT}"

if [ ! -d "${BUILD_ROOT}/.git" ]; then
  mkdir -p "${BUILD_ROOT}"
  git -C "${BUILD_ROOT}" init -q
  git -C "${BUILD_ROOT}" remote add origin "${LLAMA_CPP_URL}" 2>/dev/null || true
fi
git -C "${BUILD_ROOT}" fetch --depth 1 origin "${LLAMA_CPP_SHA}" -q
git -C "${BUILD_ROOT}" checkout -q "${LLAMA_CPP_SHA}"
echo "[oracle] checked out $(git -C "${BUILD_ROOT}" rev-parse --short HEAD)"

# Configure + build only the two binaries we need:
#   llama-completion    — raw greedy decode (--temp 0 --top-k 1) for the
#                         token-stream/text gate. NOTE: upstream split the old
#                         `llama-cli` raw-completion path into `llama-completion`.
#   llama-eval-callback — per-tensor dumps with full-tensor `sum` checksums,
#                         filterable via --tensor-filter (sub-step gates).
cmake -S "${BUILD_ROOT}" -B "${BUILD_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DGGML_METAL=ON \
  >/dev/null
cmake --build "${BUILD_ROOT}/build" \
  --target llama-completion llama-eval-callback \
  -j "${JOBS}"

echo "[oracle] built:"
ls -1 "${BUILD_ROOT}/build/bin/llama-completion" "${BUILD_ROOT}/build/bin/llama-eval-callback"
echo "[oracle] done."

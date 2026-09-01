#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# patch_llama_cpp.sh: Applies native Causal DSpark sidecar support to llama.cpp
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/training/patches/0001-feat-models-add-native-causal-dspark-draft-model-arc.patch"

TARGET_DIR=""
DO_BUILD=false

for arg in "$@"; do
    if [[ "$arg" == "--build" ]]; then
        DO_BUILD=true
    elif [[ -z "${TARGET_DIR}" ]]; then
        TARGET_DIR="$arg"
    fi
done

if [[ -z "${TARGET_DIR}" ]]; then
    TARGET_DIR="${REPO_ROOT}/scratch/llama.cpp"
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "❌ Error: Patch file not found at: ${PATCH_FILE}"
    exit 1
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
    if [[ -n "${LLAMA_CPP_DIR:-}" && -d "${LLAMA_CPP_DIR}" ]]; then
        TARGET_DIR="${LLAMA_CPP_DIR}"
    else
        echo "❌ Error: Target llama.cpp directory not found: ${TARGET_DIR}"
        echo "Usage: $0 [path/to/llama.cpp] [--build] (or set LLAMA_CPP_DIR)"
        exit 1
    fi
fi

echo "🔍 Target llama.cpp repo: ${TARGET_DIR}"

# Check if already patched
if grep -q "LLM_ARCH_DSPARK" "${TARGET_DIR}/src/llama-arch.h" 2>/dev/null; then
    echo "✅ llama.cpp at '${TARGET_DIR}' is already patched with DSpark support."
else
    echo "📦 Applying DSpark patch (${PATCH_FILE})..."
    cd "${TARGET_DIR}"
    
    if git apply --check "${PATCH_FILE}" 2>/dev/null; then
        git apply "${PATCH_FILE}"
        echo "✅ Successfully applied DSpark patch to ${TARGET_DIR}."
    else
        echo "⚠️ Standard git apply check failed, attempting 3-way merge / fallback..."
        git apply --3way "${PATCH_FILE}" || patch -p1 < "${PATCH_FILE}"
        if find . -maxdepth 3 -name "*.rej" 2>/dev/null | grep -q .; then
            echo "❌ Error: Patch application produced rejected hunks (*.rej)."
            exit 1
        fi
        echo "✅ Applied patch with 3-way merge fallback."
    fi
fi

if [ "$DO_BUILD" = true ]; then
    echo "🔨 Building llama.cpp with Metal..."
    cd "${TARGET_DIR}"
    cmake -B build -DGGML_METAL=ON
    cmake --build build -j 8
    echo "🚀 Build complete! Binaries available in ${TARGET_DIR}/build/bin"
fi

echo ""
echo "🎉 DSpark patch ready! You can run:"
echo "   llama-cli -m <base-model.gguf> -md <LFM2.5-VL-450M-DSpark-Q4_0.gguf> -ngl 99 -ngld 99 -p '...'"

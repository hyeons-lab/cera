#!/usr/bin/env bash
set -e

# Search for installed Hexagon SDK
POSSIBLE_SDK_DIRS=(
    "${HEXAGON_SDK_ROOT}"
    "/workspace/Hexagon_SDK"
    "/workspace/Qualcomm/Hexagon_SDK"
    "/opt/qcom/Hexagon_SDK"
)

FOUND_SDK=""
for dir in "${POSSIBLE_SDK_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/setup_sdk_env.source" ]; then
        FOUND_SDK="$dir"
        break
    fi
done

if [ -n "$FOUND_SDK" ]; then
    echo "[hexagon-docker] Found Hexagon SDK at: $FOUND_SDK"
    export HEXAGON_SDK_ROOT="$FOUND_SDK"
    # Source the Qualcomm SDK environment script if present
    # shellcheck disable=SC1091
    source "$FOUND_SDK/setup_sdk_env.source" > /dev/null 2>&1 || true

    # Locate Hexagon tools
    TOOLS_DIR=$(find "$FOUND_SDK/tools/HEXAGON_Tools" -maxdepth 1 -type d -name "*.*" 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$TOOLS_DIR" ]; then
        export HEXAGON_TOOLS_ROOT="$TOOLS_DIR"
        export PATH="$HEXAGON_TOOLS_ROOT/bin:$HEXAGON_SDK_ROOT/tools/scripts:$PATH"
        echo "[hexagon-docker] HEXAGON_TOOLS_ROOT set to: $HEXAGON_TOOLS_ROOT"
    fi
else
    echo "[hexagon-docker] Note: Hexagon SDK not detected yet."
    echo "[hexagon-docker] To install via QPM CLI, run:"
    echo "  1. qpm-cli --login <qualcomm-id-email>"
    echo "  2. qpm-cli --license-activate hexagonsdk6.0"
    echo "  3. qpm-cli --install hexagonsdk6.0 --path /opt/qcom/Hexagon_SDK"
fi

exec "$@"

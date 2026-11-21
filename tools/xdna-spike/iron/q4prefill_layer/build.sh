#!/usr/bin/env bash
set -eo pipefail
CHUNKS="${1:-0}"
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$(cd "$HERE/../.." && pwd)/out"
NAME="q4prefill_layer_c${CHUNKS}"
set --
source "$IRON_HOME/ironenv/bin/activate"
source "$IRON_HOME/mlir-aie/utils/env_setup.sh" >/dev/null
export PATH="$IRON_HOME/xrt-local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$IRON_HOME/xrt-local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/$NAME.prj"
cd "$OUT_DIR"
python3 "$HERE/q4prefill_layer.py" --dev npu2 --kv-chunks "$CHUNKS" \
  --xclbin-path="$OUT_DIR/$NAME.xclbin" \
  --insts-path="$OUT_DIR/insts_$NAME.bin"
readelf -S "$OUT_DIR/$NAME.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf" | grep -E "Name|\.text|\.data|\.bss" || true
ls -lh "$OUT_DIR/$NAME.xclbin" "$OUT_DIR/insts_$NAME.bin"

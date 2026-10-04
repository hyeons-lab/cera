#!/usr/bin/env bash
# Compile one design for npu2 (Strix / Strix Halo) into $OUT_DIR.
#   build.sh gemv   <M> <K> <n_cores> <n_cols>
#   build.sh memcpy <length_int32>
# Needs an IRON venv (see ../README.md); IRON_HOME points at it.
set -eo pipefail
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
OUT_DIR="${OUT_DIR:-$(pwd)/out}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$IRON_HOME/ironenv/bin/activate"
source "$IRON_HOME/mlir-aie/utils/env_setup.sh" >/dev/null 2>&1
set -u  # after the upstream env scripts, which reference unset variables
export PATH="$IRON_HOME/xrt-local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$IRON_HOME/xrt-local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT_DIR"
case "$1" in
  gemv)
    tag="${2}x${3}_c${4}"
    python3 "$HERE/gemv_multi.py" --dev npu2 -M "$2" -K "$3" --n_cores "$4" --n_cols "$5" \
      --xclbin-path="$OUT_DIR/gemv_$tag.xclbin" --insts-path="$OUT_DIR/insts_$tag.bin" ;;
  memcpy)
    python3 "$HERE/memcpy_bw.py" --dev npu2 -l "$2" \
      --xclbin-path="$OUT_DIR/memcpy_tp.xclbin" --insts-path="$OUT_DIR/insts_memcpy_tp.bin" ;;
  *) echo "usage: build.sh gemv <M> <K> <n_cores> <n_cols> | memcpy <length>" >&2; exit 2 ;;
esac

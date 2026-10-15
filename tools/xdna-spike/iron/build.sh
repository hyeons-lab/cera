#!/usr/bin/env bash
# Compile one design for npu2 (Strix / Strix Halo) into $OUT_DIR.
#   build.sh gemv   <M> <K> <n_cores> <n_cols>   -> gemv_<M>x<K>_c<n_cores>_cols<n_cols>.*
#   build.sh memcpy <length_int32>               -> memcpy_<length>.*
#   build.sh readbw <length_int32> [tile_int32]  -> readbw_<length>_t<tile>.*
#   build.sh q4gemv <M> [K]                      -> q4gemv_<M>x<K>.*  (M a multiple of 4096,
#                                                   at least 8192; K a multiple of 64, 2048 tested)
# Each design writes <name>.xclbin and insts_<name>.bin. Needs an IRON venv
# (see ../README.md); IRON_HOME points at it.
set -eo pipefail
usage() {
  echo "usage: build.sh gemv <M> <K> <n_cores> <n_cols> | memcpy <length> | readbw <length> [tile] | q4gemv <M> [K]" >&2
  exit 2
}
# Check arguments before sourcing anything, so a mistake prints the usage line.
case "${1:-}" in
  gemv) [ $# -eq 5 ] || usage ;;
  memcpy) [ $# -eq 2 ] || usage ;;
  readbw) [ $# -ge 2 ] && [ $# -le 3 ] || usage ;;
  q4gemv) [ $# -ge 2 ] && [ $# -le 3 ] || usage ;;
  *) usage ;;
esac
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$(cd "$HERE/.." && pwd)/out}"  # where common.sh looks (ARTIFACTS)
ARGS=("$@")
set --  # env_setup.sh reads $1 and $2 as install paths, so it must not see ours
source "$IRON_HOME/ironenv/bin/activate"
source "$IRON_HOME/mlir-aie/utils/env_setup.sh" >/dev/null  # stdout only: keep upstream errors visible
set -- "${ARGS[@]}"
set -u  # after the upstream env scripts, which reference unset variables
export PATH="$IRON_HOME/xrt-local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$IRON_HOME/xrt-local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT_DIR"
case "$1" in
  gemv)
    name="gemv_${2}x${3}_c${4}_cols${5}"
    python3 "$HERE/gemv_multi.py" --dev npu2 -M "$2" -K "$3" --n_cores "$4" --n_cols "$5" \
      --xclbin-path="$OUT_DIR/$name.xclbin" --insts-path="$OUT_DIR/insts_$name.bin" ;;
  memcpy)
    name="memcpy_${2}"
    python3 "$HERE/memcpy_bw.py" --dev npu2 -l "$2" \
      --xclbin-path="$OUT_DIR/$name.xclbin" --insts-path="$OUT_DIR/insts_$name.bin" ;;
  readbw)
    name="readbw_${2}_t${3:-1024}"
    python3 "$HERE/readbw.py" --dev npu2 -l "$2" -t "${3:-1024}" \
      --xclbin-path="$OUT_DIR/$name.xclbin" --insts-path="$OUT_DIR/insts_$name.bin" ;;
  q4gemv)
    name="q4gemv_${2}x${3:-2048}"
    python3 "$HERE/q4gemv/q4gemv.py" --dev npu2 -M "$2" -K "${3:-2048}" \
      --xclbin-path="$OUT_DIR/$name.xclbin" --insts-path="$OUT_DIR/insts_$name.bin" ;;
esac
echo "built $OUT_DIR/$name.xclbin and $OUT_DIR/insts_$name.bin"

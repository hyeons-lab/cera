#!/usr/bin/env bash
# Helper for building the spike IRON designs from a Windows checkout (whose
# files git may store with CRLF) inside WSL: copies tools/xdna-spike/iron into
# the WSL-native $IRON_HOME/step2, strips CR bytes on text sources, then runs
# build.sh for each "<M>[,<K>]" or "q4layer" target passed as arguments, e.g.:
#   fix_and_build.sh 2048,2048 512,2048 q4layer
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: fix_and_build.sh <M>[,<K>] | q4layer [...]" >&2
  exit 2
fi

SRC="$(cd "${SRC:-$(dirname "$0")}" && pwd)"
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
DST="${DST:-$IRON_HOME/step2}"
OUT_DIR="${OUT_DIR:-$SRC/out}"

mkdir -p "$DST" "$OUT_DIR"
DST="$(cd "$DST" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
export OUT_DIR

rm -rf "$DST/iron"
cp -r "$SRC/iron" "$DST/"
find "$DST/iron" -type d \( -name "__pycache__" -o -name "*.prj" \) -prune -exec rm -rf {} +

python3 - "$DST/iron" <<'PY'
import pathlib
import sys

TEXT_SUFFIXES = {
    ".py", ".cc", ".cpp", ".c", ".h", ".hpp", ".sh", ".txt", ".md", ".mlir", ".json",
}
root = pathlib.Path(sys.argv[1])
for p in root.rglob("*"):
    if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
        data = p.read_bytes()
        if b"\r\n" in data:
            p.write_bytes(data.replace(b"\r\n", b"\n"))
PY

cd "$DST"
for pair in "$@"; do
  if [ "$pair" = "q4layer" ]; then
    bash iron/build.sh q4layer
    continue
  fi
  if [[ "$pair" == *,* ]]; then
    M="${pair%%,*}"
    K="${pair#*,}"
  else
    M="$pair"
    K=2048
  fi
  if [[ ! "$M" =~ ^[1-9][0-9]*$ ]] || [[ ! "$K" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: invalid q4gemv shape '$pair'; expected <M>[,<K>] with positive integers or 'q4layer'" >&2
    exit 2
  fi
  bash iron/build.sh q4gemv "$M" "$K"
done

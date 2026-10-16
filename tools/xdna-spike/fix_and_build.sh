#!/usr/bin/env bash
# Helper for building the spike IRON designs from a Windows checkout (whose
# files git stores with CRLF) inside WSL: copies tools/xdna-spike/iron into
# the WSL-native ~/xdna-iron/step2, strips CR bytes, then runs build.sh for
# each "M K" pair passed as arguments, e.g. `fix_and_build.sh 2048,2048 512,2048`.
set -eo pipefail
SRC=/mnt/c/Users/iamst/development-windows/cera/tools/xdna-spike
DST=$HOME/xdna-iron/step2
mkdir -p "$DST"
cp -r "$SRC/iron" "$DST/"
python3 - <<'PY'
import pathlib
root = pathlib.Path.home() / "xdna-iron" / "step2"
for p in root.rglob("*"):
    if p.is_file():
        data = p.read_bytes()
        if b"\r\n" in data:
            p.write_bytes(data.replace(b"\r\n", b"\n"))
PY
export OUT_DIR="$SRC/out"
cd "$DST"
for pair in "$@"; do
  if [ "$pair" = "q4layer" ]; then
    bash iron/build.sh q4layer 2>&1 | tail -2
    continue
  fi
  M="${pair%%,*}"
  K="${pair##*,}"
  bash iron/build.sh q4gemv "$M" "$K" 2>&1 | tail -2
done

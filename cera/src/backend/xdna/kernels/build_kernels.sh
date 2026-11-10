#!/usr/bin/env bash
# Regenerate the committed XDNA kernel artifacts from the IRON sources in
# iron/. Run inside WSL2 on the target machine (building only; the NPU is
# driven from Windows). Needs the pinned toolchain: mlir-aie v1.4.3 and
# Peano 22.0.0.2026090701 under $IRON_HOME (default ~/xdna-iron), set up as
# in tools/xdna-spike/README.md.
#
#   bash build_kernels.sh            # rebuild all artifacts in place
#
# Artifacts: q4layer (fused Llama-3.2-1B layer) and q4head (131072x2048
# GEMV for the padded LM head). After rebuilding, update the SHA-256 table
# in README.md; CI compares the table against the committed files.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/xdna-iron/kernels-build"
mkdir -p "$DST"
cp -r "$HERE/iron" "$DST/"
python3 - "$DST" <<'PY'
import pathlib, sys
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file():
        data = p.read_bytes()
        if b"\r\n" in data:
            p.write_bytes(data.replace(b"\r\n", b"\n"))
PY
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
ARGS=()
set --  # env_setup.sh reads $1/$2 as install paths
source "$IRON_HOME/ironenv/bin/activate"
source "$IRON_HOME/mlir-aie/utils/env_setup.sh" >/dev/null
set -- "${ARGS[@]}"
set -u
export PATH="$IRON_HOME/xrt-local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$IRON_HOME/xrt-local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# The IRON JIT cache (.prj next to an artifact) does not notice kernel-source
# changes; never let a stale one survive a rebuild.
rm -rf "$HERE/q4layer.prj" "$HERE/q4head.prj"
python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
  --xclbin-path="$HERE/q4layer.xclbin" --insts-path="$HERE/insts_q4layer.bin"
rm -rf "$HERE/q4layer.prj" "$HERE/q4head.prj"
Q4LAYER_HEAD=1 python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
  --xclbin-path="$HERE/q4head.xclbin" --insts-path="$HERE/insts_q4head.bin"
cp "$HERE/q4layer.xclbin" "$HERE/q4head.xclbin"
rm -rf "$HERE/q4layer.prj" "$HERE/q4head.prj"
echo "rebuilt q4layer and q4head artifacts in $HERE"

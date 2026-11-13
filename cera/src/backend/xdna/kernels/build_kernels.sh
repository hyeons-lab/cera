#!/usr/bin/env bash
# Regenerate the committed XDNA kernel artifacts from the IRON sources in
# iron/. Run inside WSL2 on the target machine (building only; the NPU is
# driven from Windows). Needs the pinned toolchain: mlir-aie v1.4.3 and
# Peano 22.0.0.2026090701 under $IRON_HOME (default ~/xdna-iron), set up as
# in tools/xdna-spike/README.md.
#
#   bash build_kernels.sh            # rebuild all artifacts in place
#
# Artifacts: q4layer (fused Llama-3.2-1B layer: c1..c8 bucketed instruction
# streams plus default 9-chunk insts_q4layer.bin sharing q4layer.xclbin) and
# q4head
# (131072x2048 GEMV for the padded LM head, copied to q4head.xclbin alongside
# insts_q4head.bin). After rebuilding, update the SHA-256 table in README.md;
# CI compares the table against the committed files.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
DST="${DST:-$IRON_HOME/kernels-build}"
OUT="$DST/out"
rm -rf "$DST/iron" "$OUT"
mkdir -p "$OUT"
cp -r "$HERE/iron" "$DST/"
find "$DST/iron" -type d \( -name '__pycache__' -o -name '*.prj' \) -prune -exec rm -rf {} +
python3 - "$DST/iron" <<'PY'
import pathlib, sys
TEXT_EXTS = {".py", ".cc", ".h", ".sh", ".md", ".txt"}
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file() and p.suffix in TEXT_EXTS:
        data = p.read_bytes()
        if b"\r\n" in data:
            p.write_bytes(data.replace(b"\r\n", b"\n"))
PY
set --  # env_setup.sh reads $1/$2 as install paths
source "$IRON_HOME/ironenv/bin/activate"
source "$IRON_HOME/mlir-aie/utils/env_setup.sh" >/dev/null
set -u
export PATH="$IRON_HOME/xrt-local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$IRON_HOME/xrt-local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Build full 9-chunk q4layer xclbin + instruction stream on native ext4.
python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
  --xclbin-path="$OUT/q4layer.xclbin" --insts-path="$OUT/insts_q4layer.bin"

ELF="$(find "$OUT/q4layer.prj" -name 'core_*.elf' | head -n 1)"
if [[ -n "$ELF" ]]; then
  echo "=== Core ELF size ($ELF) ==="
  "$PEANO_INSTALL_DIR/bin/llvm-size" "$ELF"
fi
rm -rf "$OUT/q4layer.prj"

# Emit bucketed KV-chunk instruction streams (all share q4layer.xclbin).
for c in 1 2 3 4 5 6 7 8; do
  Q4LAYER_KV_CHUNKS="$c" python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
    --xclbin-path="$OUT/tmp_c${c}.xclbin" --insts-path="$OUT/insts_q4layer_c${c}.bin"
  rm -rf "$OUT/tmp_c${c}.prj" "$OUT/tmp_c${c}.xclbin"
done

# Emit q4head instruction stream (shares q4layer.xclbin).
Q4LAYER_HEAD=1 python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
  --xclbin-path="$OUT/tmp_head.xclbin" --insts-path="$OUT/insts_q4head.bin"
rm -rf "$OUT/tmp_head.prj" "$OUT/tmp_head.xclbin"

cp "$OUT/q4layer.xclbin" "$HERE/q4layer.xclbin"
cp "$OUT/q4layer.xclbin" "$HERE/q4head.xclbin"
cp "$OUT/insts_q4layer.bin" "$HERE/insts_q4layer.bin"
for c in 1 2 3 4 5 6 7 8; do
  cp "$OUT/insts_q4layer_c${c}.bin" "$HERE/insts_q4layer_c${c}.bin"
done
cp "$OUT/insts_q4head.bin" "$HERE/insts_q4head.bin"
ls -lh "$HERE"/*.bin "$HERE"/*.xclbin
echo "rebuilt q4layer (c1..c9) and q4head artifacts in $HERE"

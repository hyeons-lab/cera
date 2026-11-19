#!/usr/bin/env bash
set -eo pipefail
HERE="${KERNELS_DIR:-$(cd "$(dirname "$0")" && pwd)}"
IRON_HOME="${IRON_HOME:-$HOME/xdna-iron}"
DST="${DST:-$IRON_HOME/lfm2-build}"
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
set --
source "$IRON_HOME/ironenv/bin/activate"
source "$IRON_HOME/mlir-aie/utils/env_setup.sh" >/dev/null
set -u
export PATH="$IRON_HOME/xrt-local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$IRON_HOME/xrt-local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

LFM2_ATTN_BUCKETS="0 1 2 4 8 12 16 24 32 48 64 96 128 192 255"

if [ "${DEC_ONLY:-0}" != "1" ]; then
  echo "=== Building q4prefill_lfm2 (unified prefill) ==="
  python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
    --layer-mode=0 --kv-chunks=0 \
    --xclbin-path="$OUT/q4prefill_lfm2.xclbin" --insts-path="$OUT/insts_lfm2_pre_conv.bin"
  ELF=$(find "$OUT/q4prefill_lfm2.prj" -name 'core_*.elf' | head -n 1 || true)
  [ -n "$ELF" ] && "$PEANO_INSTALL_DIR/bin/llvm-size" "$ELF"
  rm -rf "$OUT/q4prefill_lfm2.prj"

  if [ "${ALL_BUCKETS:-0}" = "1" ]; then
    for c in $LFM2_ATTN_BUCKETS; do
      python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
        --layer-mode=1 --kv-chunks="$c" --xclbin-path="$OUT/tmp_lfm2_p_c${c}.xclbin" --insts-path="$OUT/insts_lfm2_pre_attn_c${c}.bin"
      rm -rf "$OUT/tmp_lfm2_p_c${c}.prj" "$OUT/tmp_lfm2_p_c${c}.xclbin"
      python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
        --layer-mode=5 --kv-chunks="$c" --xclbin-path="$OUT/tmp_lfm2_pf_c${c}.xclbin" --insts-path="$OUT/insts_lfm2_pre_fa_fin_c${c}.bin"
      rm -rf "$OUT/tmp_lfm2_pf_c${c}.prj" "$OUT/tmp_lfm2_pf_c${c}.xclbin"
    done
    python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
      --layer-mode=2 --kv-chunks=0 --xclbin-path="$OUT/tmp_lfm2_pa.xclbin" --insts-path="$OUT/insts_lfm2_pre_phase_a.bin"
    rm -rf "$OUT/tmp_lfm2_pa.prj" "$OUT/tmp_lfm2_pa.xclbin"
    python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
      --layer-mode=4 --kv-chunks=256 --xclbin-path="$OUT/tmp_lfm2_pp.xclbin" --insts-path="$OUT/insts_lfm2_pre_fa_part_c256.bin"
    rm -rf "$OUT/tmp_lfm2_pp.prj" "$OUT/tmp_lfm2_pp.xclbin"
  else
    python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
      --layer-mode=1 --kv-chunks=0 --xclbin-path="$OUT/tmp_lfm2_p_c0.xclbin" --insts-path="$OUT/insts_lfm2_pre_attn_c0.bin"
    rm -rf "$OUT/tmp_lfm2_p_c0.prj" "$OUT/tmp_lfm2_p_c0.xclbin"
  fi
fi

if [ "${PRE_ONLY:-0}" != "1" ]; then
  echo "=== Building q4layer_lfm2 (decode) ==="
  python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
    --layer-mode=0 --kv-chunks=0 \
    --xclbin-path="$OUT/q4layer_lfm2.xclbin" --insts-path="$OUT/insts_lfm2_dec_conv.bin"
  ELF=$(find "$OUT/q4layer_lfm2.prj" -name 'core_*.elf' | head -n 1 || true)
  [ -n "$ELF" ] && "$PEANO_INSTALL_DIR/bin/llvm-size" "$ELF"
  rm -rf "$OUT/q4layer_lfm2.prj"

  if [ "${ALL_BUCKETS:-0}" = "1" ]; then
    for c in $LFM2_ATTN_BUCKETS; do
      python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
        --layer-mode=1 --kv-chunks="$c" --xclbin-path="$OUT/tmp_lfm2_d_c${c}.xclbin" --insts-path="$OUT/insts_lfm2_dec_attn_c${c}.bin"
      rm -rf "$OUT/tmp_lfm2_d_c${c}.prj" "$OUT/tmp_lfm2_d_c${c}.xclbin"
      python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
        --layer-mode=4 --kv-chunks="$c" --xclbin-path="$OUT/tmp_lfm2_df_c${c}.xclbin" --insts-path="$OUT/insts_lfm2_dec_fa_fin_c${c}.bin"
      rm -rf "$OUT/tmp_lfm2_df_c${c}.prj" "$OUT/tmp_lfm2_df_c${c}.xclbin"
    done
    python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
      --layer-mode=2 --kv-chunks=0 --xclbin-path="$OUT/tmp_lfm2_da.xclbin" --insts-path="$OUT/insts_lfm2_dec_phase_a.bin"
    rm -rf "$OUT/tmp_lfm2_da.prj" "$OUT/tmp_lfm2_da.xclbin"
    python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
      --layer-mode=3 --kv-chunks=256 --xclbin-path="$OUT/tmp_lfm2_dp.xclbin" --insts-path="$OUT/insts_lfm2_dec_fa_part_c256.bin"
    rm -rf "$OUT/tmp_lfm2_dp.prj" "$OUT/tmp_lfm2_dp.xclbin"
  else
    # Fast initial smoke build: c=0 for dec_attn as well
    python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
      --layer-mode=1 --kv-chunks=0 --xclbin-path="$OUT/tmp_lfm2_d_c0.xclbin" --insts-path="$OUT/insts_lfm2_dec_attn_c0.bin"
    rm -rf "$OUT/tmp_lfm2_d_c0.prj" "$OUT/tmp_lfm2_d_c0.xclbin"
  fi
fi

ls -lh "$OUT"/q4*.xclbin "$OUT"/insts_*.bin
cp "$OUT"/q4*.xclbin "$HERE/"
cp "$OUT"/insts_*.bin "$HERE/"

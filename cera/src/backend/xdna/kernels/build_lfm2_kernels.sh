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

LFM2_PRE_BUCKETS="0 1 2 4 8 12 16 24 32 48 64 80 96 112 128 144 160 176 192 208 224 240 255 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 511 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1023"
LFM2_EXT_BUCKETS="0 1 2 4 8 12 16 24 32 48 64 96 128 192 255 320 384 448 511 512 576 640 768 896 1023"
LFM2_DEC_BUCKETS="0 1 2 4 8 12 16 18 24 32 34 48 64 66 96 128 130 192 255 258 320 384 448 511 512"
JOBS="${JOBS:-4}"
PIDS=()

spawn_job() {
  "$@" &
  PIDS+=("$!")
  if [ "${#PIDS[@]}" -ge "$JOBS" ]; then
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    PIDS=()
  fi
}

flush_jobs() {
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  PIDS=()
}

build_one() {
  local script="$1" mode="$2" c="$3" tag="$4" insts="$5"
  shift 5
  python3 "$script" --dev npu2 \
    --layer-mode="$mode" --kv-chunks="$c" "$@" \
    --xclbin-path="$OUT/tmp_${tag}.xclbin" --insts-path="$OUT/$insts"
  rm -rf "$OUT/tmp_${tag}.prj" "$OUT/tmp_${tag}.xclbin"
}

if [ "${DEC_ONLY:-0}" != "1" ]; then
  echo "=== Building q4prefill_lfm2 (unified prefill) ==="
  python3 "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" --dev npu2 \
    --layer-mode=0 --kv-chunks=0 \
    --xclbin-path="$OUT/q4prefill_lfm2.xclbin" --insts-path="$OUT/insts_lfm2_pre_conv.bin"
  ELF=$(find "$OUT/q4prefill_lfm2.prj" -name 'elfs_main_core_0_2.elf' -o -name 'core_*.elf' | head -n 1 || true)
  [ -n "$ELF" ] && "$PEANO_INSTALL_DIR/bin/llvm-size" -A "$ELF"
  rm -rf "$OUT/q4prefill_lfm2.prj"

  if [ "${ALL_BUCKETS:-0}" = "1" ]; then
    spawn_job build_one "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" 5 0 "lfm2_pnf" "insts_lfm2_pre_conv_no_ffn.bin"
    for c in $LFM2_PRE_BUCKETS; do
      spawn_job build_one "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" 1 "$c" "lfm2_p_c${c}" "insts_lfm2_pre_attn_c${c}.bin"
      spawn_job build_one "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" 6 "$c" "lfm2_pnf_c${c}" "insts_lfm2_pre_attn_no_ffn_c${c}.bin"
    done
    flush_jobs
    if [ "${BUILD_EXT:-0}" = "1" ]; then
      python3 "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" --dev npu2 \
        --layer-mode=2 --kv-chunks=0 --param-slot=0 \
        --xclbin-path="$OUT/q4prefill_lfm2_ext.xclbin" --insts-path="$OUT/insts_lfm2_pre_phase_a.bin"
      rm -rf "$OUT/q4prefill_lfm2_ext.prj"
      spawn_job build_one "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" 0 0 "lfm2_ec" "insts_lfm2_pre_ext_conv.bin" --param-slot=0
      spawn_job build_one "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" 7 0 "lfm2_ecnf" "insts_lfm2_pre_ext_conv_no_ffn.bin" --param-slot=0
      for c in $LFM2_EXT_BUCKETS; do
        spawn_job build_one "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" 5 "$c" "lfm2_pf_c${c}" "insts_lfm2_pre_fa_fin_c${c}.bin" --param-slot=3
        spawn_job build_one "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" 6 "$c" "lfm2_pfnf_c${c}" "insts_lfm2_pre_fa_fin_no_ffn_c${c}.bin" --param-slot=3
      done
      spawn_job build_one "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" 3 1024 "lfm2_pp0" "insts_lfm2_pre_fa_part0_c1024.bin" --param-slot=1
      spawn_job build_one "$DST/iron/q4prefill_lfm2_ext/q4prefill_lfm2_ext.py" 4 1024 "lfm2_pp" "insts_lfm2_pre_fa_part_c1024.bin" --param-slot=2
      flush_jobs
    fi
  else
    build_one "$DST/iron/q4prefill_lfm2/q4prefill_lfm2.py" 1 0 "lfm2_p_c0" "insts_lfm2_pre_attn_c0.bin"
  fi
fi

if [ "${PRE_ONLY:-0}" != "1" ]; then
  echo "=== Building q4layer_lfm2 (decode) ==="
  python3 "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" --dev npu2 \
    --layer-mode=0 --kv-chunks=0 \
    --xclbin-path="$OUT/q4layer_lfm2.xclbin" --insts-path="$OUT/insts_lfm2_dec_conv.bin"
  ELF=$(find "$OUT/q4layer_lfm2.prj" -name 'elfs_main_core_0_2.elf' -o -name 'core_*.elf' | head -n 1 || true)
  [ -n "$ELF" ] && "$PEANO_INSTALL_DIR/bin/llvm-size" -A "$ELF"
  rm -rf "$OUT/q4layer_lfm2.prj"

  if [ "${ALL_BUCKETS:-0}" = "1" ]; then
    spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 5 0 "lfm2_dnf" "insts_lfm2_dec_conv_no_ffn.bin"
    for c in $LFM2_DEC_BUCKETS; do
      spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 1 "$c" "lfm2_d_c${c}" "insts_lfm2_dec_attn_c${c}.bin"
      spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 6 "$c" "lfm2_dnf_c${c}" "insts_lfm2_dec_attn_no_ffn_c${c}.bin"
      spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 4 "$c" "lfm2_df_c${c}" "insts_lfm2_dec_fa_fin_c${c}.bin"
      spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 7 "$c" "lfm2_dfnf_c${c}" "insts_lfm2_dec_fa_fin_no_ffn_c${c}.bin"
    done
    spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 2 0 "lfm2_da" "insts_lfm2_dec_phase_a.bin"
    spawn_job build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 3 512 "lfm2_dp" "insts_lfm2_dec_fa_part_c512.bin"
    flush_jobs
  else
    # Fast initial smoke build: c=0 for dec_attn as well
    build_one "$DST/iron/q4layer_lfm2/q4layer_lfm2.py" 1 0 "lfm2_d_c0" "insts_lfm2_dec_attn_c0.bin"
  fi
fi

ls -lh "$OUT"/q4*.xclbin "$OUT"/insts_*.bin
cp "$OUT"/q4*.xclbin "$HERE/"
cp "$OUT"/insts_*.bin "$HERE/"

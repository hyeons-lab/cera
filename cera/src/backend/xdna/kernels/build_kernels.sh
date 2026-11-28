#!/usr/bin/env bash
# Regenerate the committed XDNA kernel artifacts from the IRON sources in
# iron/. Run inside WSL2 on the target machine (building only; the NPU is
# driven from Windows). Needs the pinned toolchain: mlir-aie v1.4.3 and
# Peano 22.0.0.2026090701 under $IRON_HOME (default ~/xdna-iron), set up as
# in tools/xdna-spike/README.md.
#
#   bash build_kernels.sh            # rebuild all artifacts in place
#
# Artifacts:
# - q4layer (fused Llama-3.2-1B decode layer: bucketed c1..c48 instruction
#   streams plus default 64-chunk insts_q4layer.bin sharing q4layer.xclbin)
# - q4head (131072x2048 GEMV for the padded LM head, copied to q4head.xclbin
#   alongside insts_q4head.bin, sharing q4layer.xclbin)
# - q4prefill_layer (fused B=32 Llama-3.2-1B prefill layer: c0..c63
#   instruction streams sharing q4prefill_layer.xclbin)
# - q4prefill_lfm3 / q4layer_lfm3 (LFM3 / d1
#   unified B=64 prefill ShortConv, causal/SWA/bidirectional Attention, and B=1 decode)
# After rebuilding, update the SHA-256 table in README.md; CI compares the
# table against the committed files.
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

# Build full 64-chunk q4layer xclbin + instruction stream on native ext4.
python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
  --xclbin-path="$OUT/q4layer.xclbin" --insts-path="$OUT/insts_q4layer.bin"

ELF="$(find "$OUT/q4layer.prj" -name 'core_*.elf' | head -n 1)"
if [[ -n "$ELF" ]]; then
  echo "=== Core ELF size ($ELF) ==="
  "$PEANO_INSTALL_DIR/bin/llvm-size" "$ELF"
fi
rm -rf "$OUT/q4layer.prj"

BUCKETS="1 2 4 8 12 16 24 32 48"

# Emit bucketed KV-chunk instruction streams (all share q4layer.xclbin).
for c in $BUCKETS; do
  Q4LAYER_KV_CHUNKS="$c" python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
    --xclbin-path="$OUT/tmp_c${c}.xclbin" --insts-path="$OUT/insts_q4layer_c${c}.bin"
  rm -rf "$OUT/tmp_c${c}.prj" "$OUT/tmp_c${c}.xclbin"
done

# Emit q4head instruction stream (shares q4layer.xclbin).
Q4LAYER_HEAD=1 python3 "$DST/iron/q4layer/q4layer.py" --dev npu2 \
  --xclbin-path="$OUT/tmp_head.xclbin" --insts-path="$OUT/insts_q4head.bin"
rm -rf "$OUT/tmp_head.prj" "$OUT/tmp_head.xclbin"

# Build q4prefill_layer shared xclbin + bucketed KV-chunk instruction streams.
PREFILL_BUCKETS="0 1 2 4 8 12 16 24 32 48 63"
python3 "$DST/iron/q4prefill_layer/q4prefill_layer.py" --dev npu2 --kv-chunks 0 \
  --xclbin-path="$OUT/q4prefill_layer.xclbin" --insts-path="$OUT/insts_q4prefill_layer_c0.bin"

P_ELF="$(find "$OUT/q4prefill_layer.prj" -name '*.elf' | head -n 1 || true)"
if [[ -n "$P_ELF" ]]; then
  echo "=== q4prefill_layer Core ELF size ($P_ELF) ==="
  "$PEANO_INSTALL_DIR/bin/llvm-size" "$P_ELF"
fi
rm -rf "$OUT/q4prefill_layer.prj"

for c in $PREFILL_BUCKETS; do
  if [[ "$c" == "0" ]]; then
    continue
  fi
  python3 "$DST/iron/q4prefill_layer/q4prefill_layer.py" --dev npu2 --kv-chunks "$c" \
    --xclbin-path="$OUT/tmp_pf_c${c}.xclbin" --insts-path="$OUT/insts_q4prefill_layer_c${c}.bin"
  rm -rf "$OUT/tmp_pf_c${c}.prj" "$OUT/tmp_pf_c${c}.xclbin"
done

rm -f "$HERE"/insts_q4layer_c*.bin "$HERE"/insts_q4prefill_layer_c*.bin \
  "$HERE"/insts_lfm3_pre_*.bin "$HERE"/insts_lfm3_dec_attn_c*.bin

# Build unified LFM3 B=64 prefill xclbin + Mode 0 (ShortConv), Mode 1 (causal/SWA),
# Mode 2 (Phase A), Mode 3 (bidir Phase B), Mode 4 (FA partial), and Mode 5 (FA final).
LFM3_ATTN_BUCKETS="0 1 2 4 8 12 16 24 32 48 64 96 128 192 255"
LFM3_DEC_ATTN_BUCKETS="0 1 2 4 8 12 16 24 32 48 64 96 128 136 160 192 255 256 264 288 320 384 448 511"
LFM3_BIDIR_B_BUCKETS="1 2 4 8 12 16 24 32 48 64 96 128 192 255 256"
python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
  --layer-mode=0 --kv-chunks=0 \
  --xclbin-path="$OUT/q4prefill_lfm3.xclbin" --insts-path="$OUT/insts_lfm3_pre_conv.bin"
rm -rf "$OUT/q4prefill_lfm3.prj"
for c in $LFM3_ATTN_BUCKETS; do
  python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
    --layer-mode=1 --kv-chunks="$c" \
    --xclbin-path="$OUT/tmp_lfm3_a_c${c}.xclbin" --insts-path="$OUT/insts_lfm3_pre_attn_c${c}.bin"
  rm -rf "$OUT/tmp_lfm3_a_c${c}.prj" "$OUT/tmp_lfm3_a_c${c}.xclbin"
  python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
    --layer-mode=5 --kv-chunks="$c" --param-slot=3 \
    --xclbin-path="$OUT/tmp_lfm3_f_c${c}.xclbin" --insts-path="$OUT/insts_lfm3_pre_fa_fin_c${c}.bin"
  rm -rf "$OUT/tmp_lfm3_f_c${c}.prj" "$OUT/tmp_lfm3_f_c${c}.xclbin"
done
python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
  --layer-mode=2 --kv-chunks=0 \
  --xclbin-path="$OUT/tmp_lfm3_phase_a.xclbin" --insts-path="$OUT/insts_lfm3_pre_phase_a.bin"
rm -rf "$OUT/tmp_lfm3_phase_a.prj" "$OUT/tmp_lfm3_phase_a.xclbin"
python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
  --layer-mode=6 --kv-chunks=256 --param-slot=1 \
  --xclbin-path="$OUT/tmp_lfm3_fa_part0.xclbin" --insts-path="$OUT/insts_lfm3_pre_fa_part0_c256.bin"
rm -rf "$OUT/tmp_lfm3_fa_part0.prj" "$OUT/tmp_lfm3_fa_part0.xclbin"
python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
  --layer-mode=4 --kv-chunks=256 --param-slot=2 \
  --xclbin-path="$OUT/tmp_lfm3_fa_part.xclbin" --insts-path="$OUT/insts_lfm3_pre_fa_part_c256.bin"
rm -rf "$OUT/tmp_lfm3_fa_part.prj" "$OUT/tmp_lfm3_fa_part.xclbin"
for c in $LFM3_BIDIR_B_BUCKETS; do
  python3 "$DST/iron/q4prefill_lfm3/q4prefill_lfm3.py" --dev npu2 \
    --layer-mode=3 --kv-chunks="$c" \
    --xclbin-path="$OUT/tmp_lfm3_bidir_b_c${c}.xclbin" --insts-path="$OUT/insts_lfm3_pre_bidir_b_c${c}.bin"
  rm -rf "$OUT/tmp_lfm3_bidir_b_c${c}.prj" "$OUT/tmp_lfm3_bidir_b_c${c}.xclbin"
done

# Build LFM3 B=1 decode xclbin + ShortConv (Mode 0), Attention <=8K (Mode 1), and >8K paged (Modes 2/3/4).
python3 "$DST/iron/q4layer_lfm3/q4layer_lfm3.py" --dev npu2 \
  --layer-mode=0 --kv-chunks=0 \
  --xclbin-path="$OUT/q4layer_lfm3.xclbin" --insts-path="$OUT/insts_lfm3_dec_conv.bin"
rm -rf "$OUT/q4layer_lfm3.prj"
python3 "$DST/iron/q4layer_lfm3/q4layer_lfm3.py" --dev npu2 \
  --layer-mode=2 --kv-chunks=0 \
  --xclbin-path="$OUT/tmp_lfm3_dec_m2.xclbin" --insts-path="$OUT/insts_lfm3_dec_phase_a.bin"
rm -rf "$OUT/tmp_lfm3_dec_m2.prj" "$OUT/tmp_lfm3_dec_m2.xclbin"
python3 "$DST/iron/q4layer_lfm3/q4layer_lfm3.py" --dev npu2 \
  --layer-mode=3 --kv-chunks=256 \
  --xclbin-path="$OUT/tmp_lfm3_dec_m3.xclbin" --insts-path="$OUT/insts_lfm3_dec_fa_part_c256.bin"
rm -rf "$OUT/tmp_lfm3_dec_m3.prj" "$OUT/tmp_lfm3_dec_m3.xclbin"
for c in $LFM3_DEC_ATTN_BUCKETS; do
  python3 "$DST/iron/q4layer_lfm3/q4layer_lfm3.py" --dev npu2 \
    --layer-mode=1 --kv-chunks="$c" \
    --xclbin-path="$OUT/tmp_lfm3_dec_c${c}.xclbin" --insts-path="$OUT/insts_lfm3_dec_attn_c${c}.bin"
  rm -rf "$OUT/tmp_lfm3_dec_c${c}.prj" "$OUT/tmp_lfm3_dec_c${c}.xclbin"
done
for c in $LFM3_ATTN_BUCKETS; do
  python3 "$DST/iron/q4layer_lfm3/q4layer_lfm3.py" --dev npu2 \
    --layer-mode=4 --kv-chunks="$c" \
    --xclbin-path="$OUT/tmp_lfm3_dec_f_c${c}.xclbin" --insts-path="$OUT/insts_lfm3_dec_fa_fin_c${c}.bin"
  rm -rf "$OUT/tmp_lfm3_dec_f_c${c}.prj" "$OUT/tmp_lfm3_dec_f_c${c}.xclbin"
done

cp "$OUT/q4layer.xclbin" "$HERE/q4layer.xclbin"
cp "$OUT/q4layer.xclbin" "$HERE/q4head.xclbin"
cp "$OUT/insts_q4layer.bin" "$HERE/insts_q4layer.bin"
for c in $BUCKETS; do
  cp "$OUT/insts_q4layer_c${c}.bin" "$HERE/insts_q4layer_c${c}.bin"
done
cp "$OUT/insts_q4head.bin" "$HERE/insts_q4head.bin"
cp "$OUT/q4prefill_layer.xclbin" "$HERE/q4prefill_layer.xclbin"
for c in $PREFILL_BUCKETS; do
  cp "$OUT/insts_q4prefill_layer_c${c}.bin" "$HERE/insts_q4prefill_layer_c${c}.bin"
done
cp "$OUT/q4prefill_lfm3.xclbin" "$HERE/q4prefill_lfm3.xclbin"
cp "$OUT/insts_lfm3_pre_conv.bin" "$HERE/insts_lfm3_pre_conv.bin"
for c in $LFM3_ATTN_BUCKETS; do
  cp "$OUT/insts_lfm3_pre_attn_c${c}.bin" "$HERE/insts_lfm3_pre_attn_c${c}.bin"
  cp "$OUT/insts_lfm3_pre_fa_fin_c${c}.bin" "$HERE/insts_lfm3_pre_fa_fin_c${c}.bin"
  cp "$OUT/insts_lfm3_dec_fa_fin_c${c}.bin" "$HERE/insts_lfm3_dec_fa_fin_c${c}.bin"
done
for c in $LFM3_DEC_ATTN_BUCKETS; do
  cp "$OUT/insts_lfm3_dec_attn_c${c}.bin" "$HERE/insts_lfm3_dec_attn_c${c}.bin"
done
cp "$OUT/insts_lfm3_pre_phase_a.bin" "$HERE/insts_lfm3_pre_phase_a.bin"
cp "$OUT/insts_lfm3_pre_fa_part0_c256.bin" "$HERE/insts_lfm3_pre_fa_part0_c256.bin"
cp "$OUT/insts_lfm3_pre_fa_part_c256.bin" "$HERE/insts_lfm3_pre_fa_part_c256.bin"
for c in $LFM3_BIDIR_B_BUCKETS; do
  cp "$OUT/insts_lfm3_pre_bidir_b_c${c}.bin" "$HERE/insts_lfm3_pre_bidir_b_c${c}.bin"
done
cp "$OUT/q4layer_lfm3.xclbin" "$HERE/q4layer_lfm3.xclbin"
cp "$OUT/insts_lfm3_dec_conv.bin" "$HERE/insts_lfm3_dec_conv.bin"
cp "$OUT/insts_lfm3_dec_phase_a.bin" "$HERE/insts_lfm3_dec_phase_a.bin"
cp "$OUT/insts_lfm3_dec_fa_part_c256.bin" "$HERE/insts_lfm3_dec_fa_part_c256.bin"
ls -lh "$HERE"/*.bin "$HERE"/*.xclbin
python3 "$HERE/iron/gemm_bf16/gemm_bf16.py" --dev npu2 --k-chunks=1 \
  --xclbin-path="$HERE/gemm_bf16.xclbin" --insts-path="$HERE/insts_gemm_bf16_k1.bin"
for kc in 2 4 8 12 16 18 24 32 48 64 68 96 128; do
  python3 "$HERE/iron/gemm_bf16/gemm_bf16.py" --dev npu2 --k-chunks="$kc" \
    --xclbin-path="/tmp/tmp_gemm_bf16_k${kc}.xclbin" --insts-path="$HERE/insts_gemm_bf16_k${kc}.bin"
  rm -f "/tmp/tmp_gemm_bf16_k${kc}.xclbin"
done

echo "rebuilt q4layer, q4head, q4prefill_layer, LFM3, and gemm_bf16 artifacts in $HERE"

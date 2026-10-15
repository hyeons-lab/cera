#!/usr/bin/env bash
# Device power for docs/XDNA_PHASE0_RESULTS.md: package power while one load
# runs back to back for about a minute. Loads alternate with 60 s idle floors
# (idle, load, idle, load, ...), so each load sits between the floor measured
# just before it and the one just after it. HWiNFO has no NPU or GPU rail, so a
# device's cost is package power, gross and above its adjacent floors.
# "steady" lines average only the samples where the device's usage reading is
# at least 90%, which excludes host-side setup. Floors are only valid when
# they show the NPU at 0% usage; check the logs.
#   npu_power.sh npu   read-only kernel, int16 GEMV, Q4 GEMV (16 runs in flight)
#   npu_power.sh gpu   the GPU reference: cera Q4_0 decode on the iGPU
# Paths come from common.sh; override them through the environment.
set -eo pipefail
case "${1:-}" in npu|gpu) ;; *) echo "usage: npu_power.sh npu|gpu" >&2; exit 2 ;; esac  # before common.sh touches anything
source "$(dirname "$0")/common.sh"
NPU_MATCH=(--match "CPU Package Power" --match "Total System Power" --match "NPU Clock" --steady "NPU Computing 0 Usage>=90")
GPU_MATCH=(--match "CPU Package Power" --match "Total System Power" --match "NPU Computing 0 Usage" --steady "GPU Utilization>=90")

idle() {  # idle <tag>
  echo "=== $1 $(date +%H:%M:%S)"
  "$PY" "$SM" idle 60 --match "CPU Package Power" --match "Total System Power" --match "NPU Computing 0 Usage" \
    --csv "$OUT_WIN\\$1.csv" 2>&1 | tr -d '\r' | tee "$OUT/$1.log"
  sleep 15
}
load() {  # load <tag> <match-array-name> <command...>
  local tag=$1 arr=$2; shift 2
  local -n matches=$arr
  echo "=== $tag $(date +%H:%M:%S)"
  sampled "$tag" "${matches[@]}" -- "$@"
  sleep 15
}

case "${1:-}" in
  npu)
    need_file "$SPIKE" "spike.exe (build.ps1)"
    # Resolve and check every artifact before the first floor, so a missing
    # file stops the script instead of surfacing mid-run.
    RB_X="$ARTIFACTS\\readbw_67108864_t4096.xclbin";  RB_I="$ARTIFACTS\\insts_readbw_67108864_t4096.bin"
    GV_X="$ARTIFACTS\\gemv_8192x2048_c8_cols8.xclbin"; GV_I="$ARTIFACTS\\insts_gemv_8192x2048_c8_cols8.bin"
    Q4_X="$ARTIFACTS\\q4gemv_65536x2048.xclbin";      Q4_I="$ARTIFACTS\\insts_q4gemv_65536x2048.bin"
    for f in "$RB_X" "$RB_I" "$GV_X" "$GV_I" "$Q4_X" "$Q4_I"; do need_file "$f" "artifact (build it with iron/build.sh)"; done
    idle idle_npu_0
    load readbw_256MB NPU_MATCH "$SPIKE" readbw "$RB_X" "$RB_I" 67108864 16 4096 12000
    idle idle_npu_1
    load gemv_8192x2048_c8 NPU_MATCH "$SPIKE" gemv "$GV_X" "$GV_I" 8192 2048 30000
    idle idle_npu_2
    SPIKE_Q4_STREAM=16 WSLENV="${WSLENV:+$WSLENV:}SPIKE_Q4_STREAM/w" \
      load q4gemv_65536_stream16 NPU_MATCH "$SPIKE" q4gemv "$Q4_X" "$Q4_I" 65536 2048 2600
    idle idle_npu_3 ;;
  gpu)
    need_file "$CERA" "cera.exe built with --features gpu"
    m="$(model Q4_0)"
    need_file "$m" "model"
    idle idle_gpu_before
    load gpu_Q4_0_decode GPU_MATCH "$CERA" bench --model "$m" --device gpu --prompt-tokens 16 --max-tokens 512 --runs 10
    idle idle_gpu_after ;;
esac
echo "=== done $(date +%H:%M:%S)"

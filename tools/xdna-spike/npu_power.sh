#!/usr/bin/env bash
# NPU power: package power while an NPU kernel runs back to back for about a
# minute, bracketed by idle floors. HWiNFO has no NPU rail, so NPU cost is read
# as package power above idle. Run from WSL; programs run natively on Windows.
set -eo pipefail
PY="${PY:-/mnt/c/Users/iamst/AppData/Local/Programs/Python/Python311/python.exe}"
SM="${SM:-C:\\Users\\iamst\\development-windows\\cera\\tools\\xdna-spike\\hwinfo_sm.py}"
SPIKE_DIR="${SPIKE_DIR:-C:\\Users\\iamst\\xdna-spike}"   # holds spike.exe and iron\ artifacts
OUT_WIN="${OUT_WIN:-C:\\Users\\iamst\\xdna-spike\\npu_power}"
OUT="${OUT:-$(wslpath -u "$OUT_WIN")}"
mkdir -p "$OUT"
MATCH=(--match "CPU Package Power" --match "Total System Power" --match "NPU Computing 0 Usage" --match "NPU Clock")
S="$SPIKE_DIR\\spike.exe"

idle() {
  echo "=== $1 $(date +%H:%M:%S)"
  "$PY" "$SM" idle 60 "${MATCH[@]}" --csv "$OUT_WIN\\$1.csv" 2>&1 | tr -d '\r' | tee "$OUT/$1.log"
  sleep 15
}
load() {  # load <tag> <spike args...>
  local tag=$1; shift
  echo "=== $tag $(date +%H:%M:%S)"
  "$PY" "$SM" run "${MATCH[@]}" --csv "$OUT_WIN\\$tag.csv" -- "$S" "$@" 2>&1 | tr -d '\r' | tee "$OUT/$tag.log"
  sleep 15
}

idle idle_before
load readbw_256MB readbw "$SPIKE_DIR\\iron\\readbw_67108864_t4096.xclbin" "$SPIKE_DIR\\iron\\insts_readbw_67108864_t4096.bin" 67108864 16 4096 12000
load gemv_8192x2048_c8 gemv "$SPIKE_DIR\\iron\\gemv_8192x2048_c8.xclbin" "$SPIKE_DIR\\iron\\insts_8192x2048_c8.bin" 8192 2048 30000
idle idle_after
echo "=== done $(date +%H:%M:%S)"

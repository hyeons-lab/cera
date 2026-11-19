#!/usr/bin/env bash
# Matrix bench: 3 devices × 2 models × 10 prompt sizes (128..32768), --warmup 3.
# Run order: metal first (fast), then cpu/gpu @ small/medium prompts, then long.
set -euo pipefail

BIN="./target/release/cera"
OUT="bench_results.csv"
LOG="bench_log.txt"

M450="$HOME/.leap/models/LFM2-VL-450M-Q4_0/LFM2-VL-450M-Q4_0.gguf"
M16B="$HOME/.leap/models/LFM2.5-VL-1.6B-Q4_0/LFM2.5-VL-1.6B-Q4_0.gguf"

echo "device,model,prompt_tokens,prefill_p50,decode_p50,prefill_mean,decode_mean,prefill_stddev,decode_stddev,n" > "$OUT"
: > "$LOG"

run_one() {
  local device="$1" model_label="$2" model_path="$3" ptok="$4" runs="$5"
  # Default --context-size is 4096; scale dynamically per prompt length so
  # prefill + decode always fit cleanly (minimum 8192 for short prompts).
  local ctx=$(( ((ptok + 128 + 2047) / 2048) * 2048 ))
  (( ctx < 8192 )) && ctx=8192
  local cmd="$BIN bench --model $model_path --device $device --prompt-tokens $ptok --max-tokens 128 --runs $runs --warmup 3 --no-cache --context-size $ctx"
  echo "=== $(date '+%H:%M:%S') $device $model_label p=$ptok runs=$runs ===" | tee -a "$LOG"
  local out
  if ! out=$($cmd 2>&1); then
    echo "FAILED: $cmd" | tee -a "$LOG"
    echo "$out" >> "$LOG"
    echo "$device,$model_label,$ptok,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,$runs" >> "$OUT"
    return
  fi
  echo "$out" >> "$LOG"
  local pre dec
  pre=$(echo "$out" | grep -E "^prefill tok/s:" | head -1)
  dec=$(echo "$out" | grep -E "^decode tok/s:"  | head -1)
  local p_p50 p_mean p_std p_n d_p50 d_mean d_std d_n
  p_p50=$(echo "$pre" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  p_mean=$(echo "$pre" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
  p_std=$(echo "$pre" | sed -n 's/.*stddev=\([0-9.]*\).*/\1/p')
  p_n=$(echo "$pre" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
  d_p50=$(echo "$dec" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
  d_mean=$(echo "$dec" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
  d_std=$(echo "$dec" | sed -n 's/.*stddev=\([0-9.]*\).*/\1/p')
  echo "$device,$model_label,$ptok,$p_p50,$d_p50,$p_mean,$d_mean,$p_std,$d_std,$p_n" >> "$OUT"
  echo "  -> prefill p50=$p_p50 decode p50=$d_p50" | tee -a "$LOG"
}

# Phase A: metal — fast, run small/medium at --runs 20, long at --runs 5.
for ptok in 128 1024 1100 2048 2100 3000 4096; do run_one metal 450M  "$M450" "$ptok" 20; done
for ptok in 8192 16384 32768; do run_one metal 450M  "$M450" "$ptok" 5; done
for ptok in 128 1024 1100 2048 2100 3000 4096; do run_one metal 1.6B  "$M16B" "$ptok" 20; done
for ptok in 8192 16384 32768; do run_one metal 1.6B  "$M16B" "$ptok" 5; done

# Phase B: cpu — small/medium prompts at full runs, p2048..p4096 at 10, >=8192 at 3 to keep wall-time manageable.
for ptok in 128 1024 1100; do run_one cpu 450M "$M450" "$ptok" 20; done
for ptok in 128 1024 1100; do run_one cpu 1.6B "$M16B" "$ptok" 20; done
for ptok in 2048 2100 3000 4096; do run_one cpu 450M "$M450" "$ptok" 10; done
for ptok in 2048 2100 3000 4096; do run_one cpu 1.6B "$M16B" "$ptok" 10; done
for ptok in 8192 16384 32768; do run_one cpu 450M "$M450" "$ptok" 3; done
for ptok in 8192 16384 32768; do run_one cpu 1.6B "$M16B" "$ptok" 3; done

# Phase C: gpu (wgpu) — slowest, especially at long contexts.
for ptok in 128 1024 1100; do run_one gpu 450M "$M450" "$ptok" 20; done
for ptok in 128 1024 1100; do run_one gpu 1.6B "$M16B" "$ptok" 20; done
for ptok in 2048 2100 3000 4096; do run_one gpu 450M "$M450" "$ptok" 10; done
for ptok in 2048 2100 3000 4096; do run_one gpu 1.6B "$M16B" "$ptok" 10; done
for ptok in 8192 16384 32768; do run_one gpu 450M "$M450" "$ptok" 3; done
for ptok in 8192 16384 32768; do run_one gpu 1.6B "$M16B" "$ptok" 3; done

echo "DONE: $(date '+%H:%M:%S')" | tee -a "$LOG"

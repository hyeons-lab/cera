#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo " Starting Dual DSpark Drafter Pipeline for LFM2.5-VL-450M"
echo " Target: Apple M1 Max (MPS) | 15 Epochs | Window k=9"
echo " Modes: [1] Standalone Drafter (Cera)  [2] Markov Drafter (llama.cpp)"
echo "=========================================================="

source .venv/bin/activate

# 1. Joint Training of both drafters simultaneously
python src/train.py --mode both --epochs 15 --batch-size 8 --grad-accum-steps 4

echo ""
echo "=== Converting Checkpoints to GGUF Sidecars ==="
# Option B Standalone (for Cera)
python src/convert_gguf.py checkpoints/best_dspark_standalone.pt checkpoints/lfm2.5-vl-450m-dspark-standalone.gguf
python src/convert_gguf_quant.py checkpoints/best_dspark_standalone.pt checkpoints/lfm2.5-vl-450m-dspark-standalone-Q4_0.gguf Q4_0

# Option B Markov (for llama.cpp)
python src/convert_gguf.py checkpoints/best_dspark_markov.pt checkpoints/lfm2.5-vl-450m-dspark-markov.gguf
python src/convert_gguf_quant.py checkpoints/best_dspark_markov.pt checkpoints/lfm2.5-vl-450m-dspark-markov-Q4_0.gguf Q4_0

echo ""
echo "=========================================================="
echo " Pipeline Complete!"
echo " Sidecar Artifacts (Option B):"
echo "   🚀 Cera:      $DIR/checkpoints/lfm2.5-vl-450m-dspark-standalone-Q4_0.gguf"
echo "   🦙 llama.cpp: $DIR/checkpoints/lfm2.5-vl-450m-dspark-markov-Q4_0.gguf"
echo "=========================================================="


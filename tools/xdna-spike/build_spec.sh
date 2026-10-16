#!/usr/bin/env bash
# Builds q4layer with Q4LAYER_SPEC read from spec_tmp.txt next to this script
# (avoids semicolon quoting problems across PowerShell/WSL boundaries).
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/spec_tmp.txt" ]; then
  export Q4LAYER_SPEC="$(cat "$HERE/spec_tmp.txt")"
  echo "Q4LAYER_SPEC=$Q4LAYER_SPEC"
fi
rm -rf "$HERE/out/q4layer.prj"
bash "$HERE/fix_and_build.sh" q4layer

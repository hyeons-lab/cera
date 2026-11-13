#!/usr/bin/env bash
# Builds q4layer with Q4LAYER_SPEC read from spec_tmp.txt next to this script
# (avoids semicolon quoting problems across PowerShell/WSL boundaries).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/spec_tmp.txt" ]; then
  spec="$(tr -d '\r\n ' < "$HERE/spec_tmp.txt" | sed 's/;*$//')"
  if [ -n "$spec" ]; then
    export Q4LAYER_SPEC="$spec"
    echo "Q4LAYER_SPEC=$Q4LAYER_SPEC"
  fi
fi
bash "$HERE/fix_and_build.sh" q4layer

#!/usr/bin/env bash
#
# Assert that a built cera-ffi native library carries the UniFFI ffi-buffer
# trampolines.
#
# Why this exists
# ---------------
# Three consumers share one native library. Kotlin and Swift call the standard
# `uniffi_cera_ffi_*` scaffolding; the Dart bindings call `uniffi_ffibuffer_*`
# trampolines instead, and UniFFI emits those only under the `ffi-buffer`
# feature (`uniffi/scaffolding-ffi-buffer-fns`).
#
# Omitting the feature produces a library that is correct in every way anyone
# checks: right architecture, right install name, all the symbols Kotlin and
# Swift need, embeds and signs and loads fine. Dart then dies at the first call
# with `dlsym: symbol not found`. It shipped that way to every platform at once
# and was caught only by running an app on a simulator.
#
# So: every build that produces a cera-ffi artifact runs this.
#
# Usage: scripts/assert-ffibuffer.sh <path-to-library> [...]
#
# Uses `grep -a` rather than `nm` on purpose: exported symbol names are stored
# as plain strings in ELF, Mach-O, and PE alike, so one check covers .so,
# .dylib, and .dll without needing a per-platform symbol tool (`nm` is not
# available on the Windows runner, and cross-inspecting Android .so files from
# macOS needs the NDK's llvm-nm).
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <library> [library ...]" >&2
    exit 2
fi

status=0
for lib in "$@"; do
    if [ ! -f "$lib" ]; then
        echo "assert-ffibuffer: $lib does not exist" >&2
        status=1
        continue
    fi
    if grep -aq "uniffi_ffibuffer" "$lib"; then
        echo "  ok: $lib carries the ffi-buffer trampolines"
    else
        echo "assert-ffibuffer: $lib has NO uniffi_ffibuffer_* trampolines" >&2
        echo "  The Dart/Flutter bindings cannot call into this build." >&2
        echo "  Build cera-ffi with --features ffi-buffer." >&2
        status=1
    fi
done

exit "$status"

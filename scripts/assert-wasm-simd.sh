#!/usr/bin/env bash
#
# Assert that a built .wasm actually contains SIMD instructions.
#
# Why this exists
# ---------------
# `.cargo/config.toml` gives wasm32 `-C target-feature=+simd128`. What that
# flag controls is not whether the `wasm_simd` kernels in
# `cera/src/backend/simd.rs` compile: they carry their own
# `#[target_feature(enable = "simd128")]` and emit real vector code either way.
# It controls the `#[cfg(all(target_arch = "wasm32", target_feature =
# "simd128"))]` dispatch in `vec_dot_*`, which is whether anything ever CALLS
# them. Without the flag every call goes to `crate::quant::vec_dot_*_scalar`,
# the kernels become unreachable and are stripped, and the shipped .wasm ends
# up with no vector instructions in it at all. That scalar path is the one the
# module header measures at 0.68 tok/s against 160 for native NEON.
#
# A `RUSTFLAGS` in the environment REPLACES that config rather than appending
# to it, so any workflow, recipe or shell that sets RUSTFLAGS for an unrelated
# reason silently drops the feature. Both CI workflows did exactly that with
# `RUSTFLAGS: ""`, and the published npm package shipped scalar for it.
#
# Little else catches it. The build succeeds. The package gets SMALLER (78 KB
# of kernels stripped), so a size budget reads the regression as an
# improvement. And the oracle suite still passes, because those tests call the
# kernels directly and so never touch the dispatch that broke.
#
# So: every build that produces a shipped .wasm runs this.
#
# Usage: scripts/assert-wasm-simd.sh <path-to-wasm> [...]
#
# `wasm-dis` comes from binaryen, which every job that calls this script
# already installs for `wasm-opt`. It counts lines carrying a `v128.` mnemonic
# in the finished artifact rather than checking the flag, because the flag is
# the input and what matters is whether vector code survived into the thing
# people download. That is a proxy, not a census: the `f32x4.*` and `i32x4.*`
# families are vector ops too and are not counted. It does not need to be a
# census. A real build gives 14100 of them against 0.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <wasm> [wasm ...]" >&2
    exit 2
fi

if ! command -v wasm-dis >/dev/null 2>&1; then
    echo "assert-wasm-simd: wasm-dis not found (install binaryen)" >&2
    exit 2
fi

# A real build has ~14000. The floor is not 1 because the kernels carry their
# own `#[target_feature]`: a zero count today depends on them being stripped
# once nothing calls them, and any future vector code that is reachable from
# the scalar path would let a mis-dispatched build through at a count of a
# handful. Anything in the thousands can only be the kernels.
MIN_V128=1000

status=0
for wasm in "$@"; do
    if [ ! -f "$wasm" ]; then
        echo "assert-wasm-simd: $wasm does not exist" >&2
        status=1
        continue
    fi
    # Disassemble first, so a wasm-dis failure is reported as itself rather
    # than reaching the counter as "no vector instructions" and being
    # misdiagnosed as a build-configuration problem.
    #
    # The feature flags are not optional, for the same reason `wasm-opt` needs
    # them in cera-wasm/Cargo.toml: binaryen refuses a non-MVP module as
    # invalid input unless told which proposals to accept. `--enable-simd`
    # covers the `v128` ops this script exists to count, and `--enable-threads`
    # the atomics in the threaded build. Recent binaryen accepts these modules
    # without either, which is exactly why they are easy to leave off and then
    # discover on whatever version the runner's apt happens to pin.
    if ! disassembly=$(wasm-dis --enable-simd --enable-threads "$wasm"); then
        echo "assert-wasm-simd: wasm-dis could not read $wasm" >&2
        status=1
        continue
    fi
    # `|| true` so a zero count reaches the comparison instead of tripping
    # `set -e` on grep's exit status 1.
    count=$(printf '%s\n' "$disassembly" | grep -c 'v128\.' || true)
    if [ "$count" -ge "$MIN_V128" ]; then
        echo "  ok: $wasm has $count lines with v128 ops"
    else
        echo "assert-wasm-simd: $wasm has $count lines with v128 ops, under the ${MIN_V128} floor" >&2
        echo "  Built without -C target-feature=+simd128, so the vec_dot" >&2
        echo "  dispatch routes to the scalar reference and the simd128" >&2
        echo "  kernels are stripped. Check whether a RUSTFLAGS in the" >&2
        echo "  environment is replacing .cargo/config.toml's wasm32 entry." >&2
        status=1
    fi
done

exit "$status"

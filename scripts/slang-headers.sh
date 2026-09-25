#!/usr/bin/env bash
# Shared `// slang-*` header reader for `just slang` and the CI drift check.
# Prints the space-separated values of one header; multi-line headers union
# (callers split on all whitespace, same as the `sed ... p` this replaces).
#
#   scripts/slang-headers.sh entries <file.slang>  # default: the basename
#   scripts/slang-headers.sh targets <file.slang>  # default: "wgsl metal"
#
# build.rs `slang_header_values` is the canonical Rust copy: same header
# match, same union-across-lines, same fail-fast on an empty targets list,
# same `wgsl`/`metal` allowlist. Pinned by `cera/tests/slang_headers.rs`
# (shell semantics); keep the two parsers (and the pinning test) in sync
# by hand.
set -euo pipefail

key="${1:-}"
file="${2:-}"

# A bad or missing key reaches usage before the readability guard, so a
# bare invocation names the contract instead of a shell-internal error.
case "$key" in
entries | targets) ;;
*)
    echo "usage: $0 entries|targets <file.slang>" >&2
    exit 1
    ;;
esac

# Fail closed on unreadable input like build.rs (`slang_header_values`
# panics): without this, `targets` mode mistakes grep's read-error exit
# for a missing header and silently defaults to "wgsl metal". The `-f`
# conjunct rejects directories, which are readable but neither parser
# accepts (the Rust side panics with "failed to read Slang source").
[ -f "$file" ] && [ -r "$file" ] || {
    echo "error: cannot read $file" >&2
    exit 1
}

case "$key" in
entries)
    values=$(sed -n 's|^[[:space:]]*//[[:space:]]*slang-entries:[[:space:]]*||p' "$file")
    if [ -z "$values" ]; then
        basename "$file" .slang
    else
        printf '%s\n' "$values"
    fi
    ;;
targets)
    if grep -q '^[[:space:]]*//[[:space:]]*slang-targets:' "$file"; then
        values=$(sed -n 's|^[[:space:]]*//[[:space:]]*slang-targets:[[:space:]]*||p' "$file")
        # An empty list fails fast like build.rs: silently defaulting to
        # both would regenerate a dead twin the build refuses to emit.
        [ -n "$values" ] || {
            echo "error: empty slang-targets list in $file" >&2
            exit 1
        }
        for t in $values; do
            case "$t" in
            wgsl | metal) ;;
            *)
                echo "error: unknown slang target '$t' in $file (want wgsl/metal)" >&2
                exit 1
                ;;
            esac
        done
        printf '%s\n' "$values"
    else
        echo "wgsl metal"
    fi
    ;;
esac

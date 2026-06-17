#!/usr/bin/env bash
#
# bump-version.sh — single source of truth for the cera workspace version.
#
# The repo version lives in the top-level VERSION file. This script propagates
# it to every place that must agree:
#   - Cargo.toml  [workspace.package].version  (inherited by all crates via
#     version.workspace = true)
#   - each dependent crate's internal `cera = { ..., version = "X.Y.Z", ... }`
#     path-dep pin (cera-cli, cera-ffi, cera-wasm, cera-parity)
#   - cera-ffi-flutter/pubspec.yaml  version:  (build name; any "+build" suffix
#     after the version is preserved)
#   - Cargo.lock  (refreshed via `cargo update --workspace`)
#
# The release pipeline (.github/workflows/publish.yml) reads the version from
# `cargo metadata` (the `cera` crate), so Cargo stays authoritative for the
# tag/assets; this script just keeps VERSION, every crate, and the Dart package
# in lockstep. The cera-ffi-kotlin Gradle modules publish a fixed `-SNAPSHOT`
# coordinate and are intentionally NOT touched here.
#
# Usage:
#   scripts/bump-version.sh <X.Y.Z>   Set a new version, then propagate it.
#   scripts/bump-version.sh           Re-sync all files to the current VERSION.
#   scripts/bump-version.sh --check   Verify every file matches VERSION; exit
#                                     non-zero on drift (no writes). For CI.
#   scripts/bump-version.sh --help    Show this help.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="$ROOT/VERSION"
CARGO_TOML="$ROOT/Cargo.toml"
PUBSPEC="$ROOT/cera-ffi-flutter/pubspec.yaml"
# Dependent crates carrying an internal `cera` path-dep pin.
PIN_CRATES=(cera-cli cera-ffi cera-wasm cera-parity)

SEMVER_RE='^[0-9]+\.[0-9]+\.[0-9]+$'

die() { echo "error: $*" >&2; exit 1; }

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"; }

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

[ -f "$VERSION_FILE" ] || die "VERSION file not found at $VERSION_FILE"
[ -f "$CARGO_TOML" ]   || die "Cargo.toml not found at $CARGO_TOML"
[ -f "$PUBSPEC" ]      || die "pubspec.yaml not found at $PUBSPEC"

CHECK_ONLY=0
NEW_VERSION=""
case "${1:-}" in
  --check) CHECK_ONLY=1 ;;
  "")      ;;
  *)       NEW_VERSION="$1" ;;
esac

current_version() { tr -d '[:space:]' < "$VERSION_FILE"; }

if [ -n "$NEW_VERSION" ]; then
  [[ "$NEW_VERSION" =~ $SEMVER_RE ]] || die "version '$NEW_VERSION' is not MAJOR.MINOR.PATCH"
  VERSION="$NEW_VERSION"
else
  VERSION="$(current_version)"
  [[ "$VERSION" =~ $SEMVER_RE ]] || die "VERSION file holds '$VERSION', not MAJOR.MINOR.PATCH"
fi

# --- read current values from each file (without mutating) -------------------

cargo_pkg_version() {
  perl -0777 -ne 'print "$1" if /\[workspace\.package\].*?^version = "([0-9]+\.[0-9]+\.[0-9]+)"/sm' "$CARGO_TOML"
}
crate_pin_version() {
  # versions on the internal `cera*` path-dep lines in crate $1, one per line.
  # A crate may pin more than one internal crate (e.g. cera-parity pins both
  # `cera` and `cera-ffi`), so this can emit multiple versions.
  perl -ne 'print "$1\n" if /^cera[a-z-]* = \{.*?\bversion = "([0-9]+\.[0-9]+\.[0-9]+)"/' "$ROOT/$1/Cargo.toml"
}
pubspec_build_name() {
  perl -ne 'print "$1" if /^version:\s*([0-9]+\.[0-9]+\.[0-9]+)/' "$PUBSPEC"
}
pubspec_build_suffix() {
  # the "+N" after the build name, if any (empty otherwise)
  perl -ne 'print "$1" if /^version:\s*[0-9]+\.[0-9]+\.[0-9]+(\+\S+)?/ && defined $1' "$PUBSPEC"
}

if [ "$CHECK_ONLY" -eq 1 ]; then
  want="$(current_version)"
  [[ "$want" =~ $SEMVER_RE ]] || die "VERSION file holds '$want', not MAJOR.MINOR.PATCH"
  drift=0
  cp="$(cargo_pkg_version)"
  [ "$cp" = "$want" ] || { echo "drift: Cargo.toml workspace.package version is '$cp', want '$want'" >&2; drift=1; }
  for c in "${PIN_CRATES[@]}"; do
    while IFS= read -r pin; do
      [ -n "$pin" ] || continue
      [ "$pin" = "$want" ] || { echo "drift: $c internal dep pin is '$pin', want '$want'" >&2; drift=1; }
    done < <(crate_pin_version "$c")
  done
  pp="$(pubspec_build_name)"
  [ "$pp" = "$want" ] || { echo "drift: pubspec.yaml build name is '$pp', want '$want'" >&2; drift=1; }
  if [ "$drift" -eq 0 ]; then echo "OK: all files match VERSION $want"; fi
  exit "$drift"
fi

# --- propagate ---------------------------------------------------------------

printf '%s\n' "$VERSION" > "$VERSION_FILE"

# [workspace.package].version — the lone top-level `version = "..."` line.
perl -i -pe 'BEGIN{$v=shift} s/^version = "[0-9]+\.[0-9]+\.[0-9]+"/version = "$v"/' "$VERSION" "$CARGO_TOML"

# Internal `cera*` path-dep pins in each dependent crate. Anchored on a line
# starting `cera* = {` so only internal dep requirements are touched (a crate
# may pin several, e.g. cera-parity pins both `cera` and `cera-ffi`).
for c in "${PIN_CRATES[@]}"; do
  perl -i -pe 'BEGIN{$v=shift} s/(^cera[a-z-]* = \{.*?\bversion = ")[0-9]+\.[0-9]+\.[0-9]+(")/${1}$v$2/' \
    "$VERSION" "$ROOT/$c/Cargo.toml"
done

# pubspec.yaml — replace the build name, preserve any "+build" suffix.
suffix="$(pubspec_build_suffix)"
perl -i -pe 'BEGIN{$v=shift; $s=shift} s/^version:\s*[0-9]+\.[0-9]+\.[0-9]+(\+\S+)?/version: $v$s/' "$VERSION" "$suffix" "$PUBSPEC"

# Refresh Cargo.lock so the workspace entries match.
if command -v cargo >/dev/null 2>&1; then
  ( cd "$ROOT" && cargo update --workspace >/dev/null 2>&1 ) \
    && echo "refreshed Cargo.lock via cargo update --workspace" \
    || echo "warning: 'cargo update --workspace' failed; run it manually" >&2
else
  echo "warning: cargo not found; run 'cargo update --workspace' to refresh Cargo.lock" >&2
fi

echo "version set to $VERSION"
echo "  VERSION"
echo "  Cargo.toml        (workspace.package + ${#PIN_CRATES[@]} internal cera pins)"
echo "  pubspec.yaml      ${VERSION}${suffix}"
echo "  Cargo.lock        (workspace entries)"
echo
echo "review the diff, then commit."

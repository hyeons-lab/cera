#!/usr/bin/env bash
#
# bump-version.sh — single source of truth for the cera workspace version.
#
# The repo version lives in the top-level VERSION file. This script propagates
# it to every place that must agree so that ALL published artifacts (crates.io,
# npm, Maven Central) carry the SAME version:
#   - Cargo.toml  [workspace.package].version  (inherited by all crates via
#     version.workspace = true)
#   - each dependent crate's internal `cera = { ..., version = "X.Y.Z", ... }`
#     path-dep pin (cera-cli, cera-ffi, cera-wasm, cera-parity)
#   - cera-ffi-flutter/pubspec.yaml  version:  (build name; any "+build" suffix
#     after the version is preserved)
#   - cera-ffi-kotlin/gradle.properties  VERSION_NAME  (the Maven Central
#     coordinate for the Kotlin/Android bindings; any "-QUALIFIER" suffix such
#     as "-SNAPSHOT" is preserved)
#
# The release pipeline (.github/workflows/publish.yml) reads the version from
# `cargo metadata` (the `cera` crate) for the git tag + npm/CLI assets, and
# passes it to Gradle for the Maven coordinate, so Cargo stays authoritative;
# this script keeps VERSION, every crate, the Dart package, and the Gradle
# coordinate in lockstep, and `--check` (run in CI and on the publish path)
# fails the build on any drift.
#
# This script edits files only — it does NOT touch Cargo.lock (which is
# gitignored here; cargo refreshes the workspace entries on the next build).
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
GRADLE_PROPS="$ROOT/cera-ffi-kotlin/gradle.properties"
# Dependent crates carrying an internal `cera` path-dep pin.
PIN_CRATES=(cera-cli cera-ffi cera-wasm cera-parity)

SEMVER_RE='^[0-9]+\.[0-9]+\.[0-9]+$'

die() { echo "error: $*" >&2; exit 1; }

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"; }

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

[ -f "$VERSION_FILE" ]  || die "VERSION file not found at $VERSION_FILE"
[ -f "$CARGO_TOML" ]    || die "Cargo.toml not found at $CARGO_TOML"
[ -f "$PUBSPEC" ]       || die "pubspec.yaml not found at $PUBSPEC"
[ -f "$GRADLE_PROPS" ]  || die "gradle.properties not found at $GRADLE_PROPS"
for c in "${PIN_CRATES[@]}"; do
  [ -f "$ROOT/$c/Cargo.toml" ] || die "$c/Cargo.toml not found at $ROOT/$c/Cargo.toml"
done

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

# --- readers: each prints the MAJOR.MINOR.PATCH core it finds ----------------
# These are the canonical "what version is in this file" functions, used by both
# --check and the post-write verification below. Because every write is verified
# by re-reading with these, a writer regex that ever drifts from its reader (or
# silently matches nothing) is caught immediately instead of passing silently.

cargo_pkg_version() {
  perl -0777 -ne 'print "$1" if /\[workspace\.package\].*?^\s*version = "([0-9]+\.[0-9]+\.[0-9]+)"/sm' "$CARGO_TOML"
}
crate_pin_version() {
  # versions on the internal `cera*` path-dep lines in crate $1, one per line.
  # A crate may pin more than one internal crate (e.g. cera-parity pins both
  # `cera` and `cera-ffi`), so this can emit multiple versions.
  perl -ne 'print "$1\n" if /^\s*cera[a-z-]* = \{.*?\bversion = "([0-9]+\.[0-9]+\.[0-9]+)"/' "$ROOT/$1/Cargo.toml"
}
pubspec_version() {
  perl -ne 'print "$1" if /^version:\s*([0-9]+\.[0-9]+\.[0-9]+)/' "$PUBSPEC"
}
pubspec_suffix() {
  # the "+build" suffix after the version, if any (empty otherwise)
  perl -ne 'print "$1" if /^version:\s*[0-9]+\.[0-9]+\.[0-9]+(\+\S+)?/ && defined $1' "$PUBSPEC"
}
gradle_version() {
  perl -ne 'print "$1" if /^VERSION_NAME\s*=\s*([0-9]+\.[0-9]+\.[0-9]+)/' "$GRADLE_PROPS"
}
gradle_suffix() {
  # the "-QUALIFIER" suffix after the version (e.g. -SNAPSHOT), if any
  perl -ne 'print "$1" if /^VERSION_NAME\s*=\s*[0-9]+\.[0-9]+\.[0-9]+(-\S+)?/ && defined $1' "$GRADLE_PROPS"
}

# ─── --check: report drift, never write ──────────────────────────────────────
if [ "$CHECK_ONLY" -eq 1 ]; then
  want="$VERSION"   # already read + validated from the VERSION file above
  drift=0
  check() { # <label> <got>
    [ "$2" = "$want" ] || { echo "drift: $1 is '$2', want '$want'" >&2; drift=1; }
  }

  check "Cargo.toml [workspace.package] version" "$(cargo_pkg_version)"
  for c in "${PIN_CRATES[@]}"; do
    seen_pin=0
    while IFS= read -r pin; do
      [ -n "$pin" ] || continue
      seen_pin=1
      check "$c internal dep pin" "$pin"
    done < <(crate_pin_version "$c")
    # Every PIN_CRATE is expected to carry at least one internal cera* pin; if
    # the reader finds none (a removed pin or a reformatting the regex misses),
    # treat it as drift rather than silently passing.
    [ "$seen_pin" -eq 1 ] || { echo "drift: $c has no internal cera* path-dep pin (expected at least one)" >&2; drift=1; }
  done
  check "pubspec.yaml build name" "$(pubspec_version)"
  check "gradle.properties VERSION_NAME" "$(gradle_version)"

  if [ "$drift" -eq 0 ]; then echo "OK: all files match VERSION $want"; fi
  exit "$drift"
fi

# ─── propagate: write, then verify every write landed ────────────────────────
# verify() re-reads the file with its canonical reader and aborts if the value
# isn't exactly $VERSION — so a no-op write (regex miss, reformatted file) fails
# loudly instead of reporting a bump that never happened.
verify() { # <label> <got>
  [ "$2" = "$VERSION" ] || die "write did not land: $1 reads '$2' after update, expected '$VERSION' (file format/regex drift?)"
}

printf '%s\n' "$VERSION" > "$VERSION_FILE"

# [workspace.package].version — the lone top-level `version = "..."` line.
perl -i -pe 'BEGIN{$v=shift} s/^(\s*)version = "[0-9]+\.[0-9]+\.[0-9]+"/${1}version = "$v"/' "$VERSION" "$CARGO_TOML"
verify "Cargo.toml [workspace.package] version" "$(cargo_pkg_version)"

# Internal `cera*` path-dep pins in each dependent crate. Anchored on a line
# starting `cera* = {` so only internal dep requirements are touched (a crate
# may pin several, e.g. cera-parity pins both `cera` and `cera-ffi`).
for c in "${PIN_CRATES[@]}"; do
  perl -i -pe 'BEGIN{$v=shift} s/(^\s*cera[a-z-]* = \{.*?\bversion = ")[0-9]+\.[0-9]+\.[0-9]+(")/${1}$v$2/' \
    "$VERSION" "$ROOT/$c/Cargo.toml"
  pins="$(crate_pin_version "$c")"
  [ -n "$pins" ] || die "$c: no internal cera* path-dep pin found to update"
  while IFS= read -r pin; do verify "$c internal dep pin" "$pin"; done <<< "$pins"
done

# pubspec.yaml — replace the build name, preserve any "+build" suffix.
psuffix="$(pubspec_suffix)"
perl -i -pe 'BEGIN{$v=shift; $s=shift} s/^version:\s*[0-9]+\.[0-9]+\.[0-9]+(\+\S+)?/version: $v$s/' \
  "$VERSION" "$psuffix" "$PUBSPEC"
verify "pubspec.yaml build name" "$(pubspec_version)"

# gradle.properties VERSION_NAME — the Maven coordinate. Replace the version,
# preserve any "-QUALIFIER" suffix (e.g. -SNAPSHOT) so the release channel
# encoded in the file (release vs snapshot) is kept.
gsuffix="$(gradle_suffix)"
perl -i -pe 'BEGIN{$v=shift; $s=shift} s/^(VERSION_NAME\s*=\s*)[0-9]+\.[0-9]+\.[0-9]+(-\S+)?/${1}$v$s/' \
  "$VERSION" "$gsuffix" "$GRADLE_PROPS"
verify "gradle.properties VERSION_NAME" "$(gradle_version)"

echo "version set to $VERSION"
echo "  VERSION"
echo "  Cargo.toml        (workspace.package + ${#PIN_CRATES[@]} internal cera pins)"
echo "  pubspec.yaml      ${VERSION}${psuffix}"
echo "  gradle.properties ${VERSION}${gsuffix}"
echo
echo "review the diff, then commit. (Cargo.lock is gitignored; cargo refreshes"
echo "it on the next build.)"

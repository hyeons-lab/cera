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
#   - cera_ffi_flutter/pubspec.yaml  version:  (build name; any "+build" suffix
#     after the version is preserved)
#   - cera-ffi-kotlin/gradle.properties  VERSION_NAME  (the Maven Central
#     coordinate for the Kotlin/Android bindings; any "-QUALIFIER" suffix such
#     as "-SNAPSHOT" is preserved)
#   - the cera_ffi_flutter platform manifests, which name the *published*
#     native artifact each platform resolves at build time: the AAR coordinate
#     in android/build.gradle and the release tag the linux/windows
#     CMakeLists download from, plus the two podspec versions (see
#     PLUGIN_SITES below)
#
# The release pipeline (.github/workflows/publish.yml) reads the version from
# `cargo metadata` (the `cera` crate) for the git tag + npm/CLI assets, and
# passes it to Gradle for the Maven coordinate, so Cargo stays authoritative;
# this script keeps VERSION, every crate, the Dart package, and the Gradle
# coordinate in lockstep, and `--check` (run in CI and on the publish path)
# fails the build on any drift.
#
# This script edits files only, and it does NOT touch Cargo.lock. Since #349 the
# lock is TRACKED, and it records the workspace crates' own versions, so a bump
# leaves it stale until cargo rewrites it. Run any cargo command afterwards
# (`cargo metadata` is enough) and commit the refreshed Cargo.lock with the
# bump, or CI's `--locked` builds fail on the mismatch. Verified: bumping to
# 0.4.1 without refreshing makes `cargo metadata --locked` exit non-zero.
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
PUBSPEC="$ROOT/cera_ffi_flutter/pubspec.yaml"
GRADLE_PROPS="$ROOT/cera-ffi-kotlin/gradle.properties"
# Dependent crates carrying an internal `cera` path-dep pin.
PIN_CRATES=(cera-cli cera-ffi cera-wasm cera-parity)

# Flutter plugin platform manifests, one entry per version *site*: a file with
# two references appears twice. Format is "<label>|<path from ROOT>|<regex>",
# where the regex names three captures — `pre`, `ver`, `post` — so the single
# pattern serves as both reader and writer and the two cannot drift apart.
#
# These matter more than the other files here, because getting one wrong does
# not fail a build. They name published release artifacts, so a site left
# behind on a bump silently resolves the *previous* release's native library
# (Android), or a git tag carrying no assets at all (Linux/Windows). Both
# surface as a runtime failure in a consumer's app, long after the release.
PLUGIN_SITES=(
  "android build.gradle version|cera_ffi_flutter/android/build.gradle|(?<pre>^version = ')(?<ver>[0-9]+\.[0-9]+\.[0-9]+)(?<post>')"
  "android cera-ffi-android dependency|cera_ffi_flutter/android/build.gradle|(?<pre>^\s*api 'com\.hyeons-lab:cera-ffi-android:)(?<ver>[0-9]+\.[0-9]+\.[0-9]+)(?<post>')"
  "linux CMakeLists CERA_VERSION|cera_ffi_flutter/linux/CMakeLists.txt|(?<pre>^set\(CERA_VERSION \")(?<ver>[0-9]+\.[0-9]+\.[0-9]+)(?<post>\"\))"
  "windows CMakeLists CERA_VERSION|cera_ffi_flutter/windows/CMakeLists.txt|(?<pre>^set\(CERA_VERSION \")(?<ver>[0-9]+\.[0-9]+\.[0-9]+)(?<post>\"\))"
  "ios podspec s.version|cera_ffi_flutter/ios/cera_ffi_flutter.podspec|(?<pre>^\s*s\.version\s*=\s*')(?<ver>[0-9]+\.[0-9]+\.[0-9]+)(?<post>')"
  "macos podspec s.version|cera_ffi_flutter/macos/cera_ffi_flutter.podspec|(?<pre>^\s*s\.version\s*=\s*')(?<ver>[0-9]+\.[0-9]+\.[0-9]+)(?<post>')"
)

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
for site in "${PLUGIN_SITES[@]}"; do
  IFS='|' read -r _label spath _re <<< "$site"
  [ -f "$ROOT/$spath" ] || die "plugin manifest not found at $ROOT/$spath"
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
plugin_site_versions() { # <path from ROOT> <regex> — one version per match
  perl -ne 'BEGIN{$re = shift} print "$+{ver}\n" if /$re/' "$2" "$ROOT/$1"
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
  for site in "${PLUGIN_SITES[@]}"; do
    IFS='|' read -r slabel spath sre <<< "$site"
    seen_site=0
    while IFS= read -r sver; do
      [ -n "$sver" ] || continue
      seen_site=1
      check "$slabel" "$sver"
    done < <(plugin_site_versions "$spath" "$sre")
    # A site whose regex now matches nothing is drift too: the file was
    # reformatted or the line moved, and staying quiet about it is how a
    # published artifact reference gets left a release behind.
    [ "$seen_site" -eq 1 ] || { echo "drift: $slabel matched nothing in $spath" >&2; drift=1; }
  done

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

# Flutter plugin platform manifests. The Apple SPM manifests are deliberately
# absent: their `RELEASE_VERSION` / `RELEASE_CHECKSUM` literals are rewritten by
# the release workflow, which is the only place the checksum is known.
for site in "${PLUGIN_SITES[@]}"; do
  IFS='|' read -r slabel spath sre <<< "$site"
  perl -i -pe 'BEGIN{$re = shift; $v = shift} s/$re/$+{pre}$v$+{post}/' \
    "$sre" "$VERSION" "$ROOT/$spath"
  svers="$(plugin_site_versions "$spath" "$sre")"
  [ -n "$svers" ] || die "$slabel: regex matched nothing in $spath"
  while IFS= read -r sver; do verify "$slabel" "$sver"; done <<< "$svers"
done

echo "version set to $VERSION"
echo "  VERSION"
echo "  Cargo.toml        (workspace.package + ${#PIN_CRATES[@]} internal cera pins)"
echo "  pubspec.yaml      ${VERSION}${psuffix}"
echo "  gradle.properties ${VERSION}${gsuffix}"
echo "  cera_ffi_flutter  ${#PLUGIN_SITES[@]} platform-manifest sites"
echo
echo "Cargo.lock is tracked and records these crate versions, so refresh it before"
echo "committing:  cargo metadata --offline >/dev/null  (then include it in the diff)."
echo
echo "review the diff, then commit."

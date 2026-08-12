#!/usr/bin/env bash
#
# Materialize the patched Dart bindings generator.
#
# The generator is a fork of nchapman/uniffi-bindgen-dart. Rather than commit
# the forked source, this repo keeps the upstream pin (`UPSTREAM`) and the
# divergence as a patch series (`patches/*.patch`), and rebuilds the working
# tree from the two. Every local change is therefore visible as a patch and is
# directly `git am`-able against upstream when it is time to send it there.
#
# Layout under third_party/uniffi-bindgen-dart/:
#
#   UPSTREAM        the pinned repo, tag and commit
#   patches/        the fork, one logical change per file, applied in order
#   .cache/         bare clone of upstream (gitignored)
#   build/          materialized crate: upstream + patches (gitignored)
#
# The clone is cached, so only the first run and a pin change need the network.
# Pass --offline to fail instead of fetching, and --force to rebuild build/
# from scratch even when the stamp says it is current.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="$root/third_party/uniffi-bindgen-dart"
cache="$vendor/.cache"
build="$vendor/build"
stamp="$build/.vendor-stamp"
# Parking spot for `build/target` while `build/` is rebuilt. Beside it rather
# than in /tmp so the move stays on one filesystem, i.e. a rename.
cargo_target="$vendor/.target"

offline=0
force=0
for arg in "$@"; do
  case "$arg" in
    --offline) offline=1 ;;
    --force) force=1 ;;
    *) echo "usage: $(basename "$0") [--offline] [--force]" >&2; exit 2 ;;
  esac
done

# Parse the pin. Only `key=value` lines; everything else is commentary.
repo=""; tag=""; rev=""; subdir=""
while IFS='=' read -r key value; do
  case "$key" in
    repo) repo="$value" ;;
    tag) tag="$value" ;;
    rev) rev="$value" ;;
    subdir) subdir="$value" ;;
  esac
done < <(grep -E '^[a-z]+=' "$vendor/UPSTREAM")

if [ -z "$repo" ] || [ -z "$rev" ] || [ -z "$subdir" ]; then
  echo "error: $vendor/UPSTREAM is missing repo, rev or subdir" >&2
  exit 1
fi

# The stamp covers the WHOLE pin and the patches, so editing either rebuilds
# without anyone having to remember --force. `repo` and `subdir` are in there
# alongside `rev`: keying on the rev alone meant repointing the pin at a
# different repo or a different crate left the stamp valid, and the script
# happily reported "up to date" over a build/ materialized from the old one.
want="$repo $rev $subdir $(cat "$vendor"/patches/*.patch | shasum -a 256 | cut -d' ' -f1)"

# Recover a parked cargo target directory before doing anything else. The
# rebuild below moves `build/target` aside and back, so an interrupted run (a
# Ctrl-C while iterating on a patch is the likely one) can leave it stranded at
# `$cargo_target` with `build/` already gone. Restoring here rather than only on
# the rebuild path means the stamp early-exit cannot strand it permanently,
# which it otherwise did: the script reported "up to date" while a full
# generator recompile was silently waiting.
if [ -d "$cargo_target" ] && [ ! -d "$build/target" ] && [ -d "$build" ]; then
  mv "$cargo_target" "$build/target"
fi

if [ "$force" -eq 0 ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
  echo "generator up to date ($tag, $(ls "$vendor"/patches/*.patch | wc -l | tr -d ' ') patches)"
  exit 0
fi

# A full bare clone, deliberately not `--filter=blob:none`. A partial clone is
# smaller but leaves the cache a promisor: `git archive` then lazily fetches the
# blobs it needs from origin, which quietly defeats --offline and makes the
# cache useless without a network. This repo is small enough that fetching it
# whole once is the better trade.
if [ ! -d "$cache" ]; then
  if [ "$offline" -eq 1 ]; then
    echo "error: no cached clone at $cache and --offline was given." >&2
    echo "       Run without --offline once to populate it." >&2
    exit 1
  fi
  echo "cloning $repo (first run)…"
  git clone --quiet --bare "$repo" "$cache"
fi

if ! git -C "$cache" cat-file -e "$rev^{commit}" 2>/dev/null; then
  if [ "$offline" -eq 1 ]; then
    echo "error: $rev is not in the cache at $cache and --offline was given." >&2
    exit 1
  fi
  echo "fetching $rev…"
  # The fallback needs an explicit refspec. `git clone --bare` writes no
  # `remote.origin.fetch`, so a bare `fetch --tags origin` would fetch tags and
  # nothing else, and the pin would stay unresolvable the moment it moves to a
  # commit that carries no tag. Fetching the branch heads too is what makes the
  # documented "change the rev in UPSTREAM and re-run" workflow actually work on
  # a machine whose cache predates the new rev.
  git -C "$cache" fetch --quiet origin "$rev" ||
    git -C "$cache" fetch --quiet --tags origin '+refs/heads/*:refs/remotes/origin/*'
fi

if ! git -C "$cache" cat-file -e "$rev^{commit}" 2>/dev/null; then
  echo "error: pinned rev $rev does not exist in $repo" >&2
  exit 1
fi

# `rev` is what we build; `tag` is only there to say which release it is. Warn
# when upstream has moved the tag off it, since the two then disagree about
# what "$tag" means and every later message here would name a release this is
# no longer built from. Not fatal: the rev is the pin, and a rev that resolves
# is a reproducible build whatever the tag now points at.
if [ -n "$tag" ]; then
  tagged="$(git -C "$cache" rev-parse --quiet --verify "refs/tags/$tag^{commit}" 2>/dev/null || true)"
  if [ -n "$tagged" ] && [ "$tagged" != "$rev" ]; then
    echo "warning: upstream tag $tag now points at $tagged, not the pinned $rev." >&2
    echo "         Building the pinned rev. Update UPSTREAM if the move was intended." >&2
  fi
fi

echo "materializing $tag ($rev) + $(ls "$vendor"/patches/*.patch | wc -l | tr -d ' ') patches…"
# Keep cargo's output across the rebuild. Editing a patch is the documented
# inner loop, and wiping `build/target` with the sources turns each iteration
# into a full recompile of the generator's dependency tree.
if [ -d "$build/target" ]; then
  rm -rf "$cargo_target"
  mv "$build/target" "$cargo_target"
fi
rm -rf "$build"
mkdir -p "$build"
if [ -d "$cargo_target" ]; then
  mv "$cargo_target" "$build/target"
fi
# Strip exactly as many leading components as `subdir` has, so the crate lands
# at the root of build/ whatever its depth upstream. Hardcoding 2 would silently
# produce a half-stripped tree the day `subdir` moves.
strip=$(awk -F/ '{print NF}' <<< "$subdir")
git -C "$cache" archive "$rev" "$subdir" | tar -x -C "$build" --strip-components="$strip"

# Applied with --directory from the repo root, NOT by cd-ing into build/.
# `build/` sits inside this repository's work tree, and `git apply` run from a
# subdirectory resolves patch paths against the repository root and silently
# SKIPS anything that lands outside the current prefix, reporting success
# while applying nothing. --directory prefixes the paths explicitly instead.
#
# --check before each apply, so a patch that does not fit is reported without
# being half-written into the file. Earlier patches in the series are already
# applied at that point; no stamp is written on the failure path, so the next
# run wipes build/ and starts over rather than layering onto the remains.
build_rel="${build#"$root"/}"
for patch in "$vendor"/patches/*.patch; do
  if ! git -C "$root" apply --directory="$build_rel" -p1 --check "$patch" 2>/dev/null; then
    echo "error: $(basename "$patch") does not apply to $tag." >&2
    echo "       Upstream moved under it, or an earlier patch changed shape." >&2
    echo "       Fix the patch (or the pin) rather than editing build/, which is regenerated." >&2
    echo >&2
    git -C "$root" apply --directory="$build_rel" -p1 --check "$patch" >&2 || true
    exit 1
  fi
  git -C "$root" apply --directory="$build_rel" -p1 "$patch"
  echo "  applied $(basename "$patch")"
done

# The skip-silently failure mode above is worth a guard rather than trust: if
# the flatten patch did not land, the crate still inherits from a workspace
# that is not there and the build fails much later with an unrelated message.
if ! grep -q '^\[workspace\]' "$build/Cargo.toml" 2>/dev/null; then
  echo "error: build/Cargo.toml has no [workspace] table; patches did not apply." >&2
  exit 1
fi

echo "$want" > "$stamp"
echo "generator ready at ${build#"$root"/}"

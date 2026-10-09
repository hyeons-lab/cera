//! Embedded Hexagon DSP skels + device probe.
//!
//! The `skels/` directory vendors the five `libcera-htp-vXX.so` DSP
//! libraries (see `skels/SOURCE.md` for provenance). They are embedded
//! in the binary so apps ship one artifact: at runtime the engine writes
//! the matching skel next to itself (or into an app-provided directory)
//! where FastRPC's loader can find it.

use std::sync::Arc;

use super::LockOrRecover;
use super::device::{HexagonArch, HexagonDevice};
use super::sys::FastRpcDriver;
use crate::session::CeraError;

/// Embedded skel bytes per architecture (see `skels/SOURCE.md`).
pub fn embedded_skel(arch: HexagonArch) -> &'static [u8] {
    match arch {
        HexagonArch::V73 => include_bytes!("skels/libcera-htp-v73.so"),
        HexagonArch::V75 => include_bytes!("skels/libcera-htp-v75.so"),
        HexagonArch::V79 => include_bytes!("skels/libcera-htp-v79.so"),
        HexagonArch::V81 => include_bytes!("skels/libcera-htp-v81.so"),
        HexagonArch::V85 => include_bytes!("skels/libcera-htp-v85.so"),
    }
}

/// All architectures we ship skels for, probe order (most common first).
pub const PROBE_ARCHS: [HexagonArch; 5] = [
    HexagonArch::V79,
    HexagonArch::V75,
    HexagonArch::V73,
    HexagonArch::V81,
    HexagonArch::V85,
];

/// Serializes `install_skels`: two concurrent installs must neither
/// interleave a `var`/`set_var` pair (lost prepend) nor share a `.so.tmp`
/// path mid-write.
static SKEL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Whether a skel directory owned by `uid` with `mode` cannot be trusted by a
/// process running as `euid`: owned by someone else, or writable by "other".
/// Group-write is tolerated on purpose: Android app-private dirs are
/// `rwxrwx--x` with the app's own uid as group.
fn skel_dir_unsafe(uid: u32, euid: u32, mode: u32) -> bool {
    uid != euid || mode & 0o002 != 0
}

/// Create `dir` (mode 0700 when newly created) and refuse one the loader
/// would execute code from but another local user could tamper with: an
/// existing directory owned by someone else, or writable by "other". App-private
/// Android dirs (owned by the app uid, at most `rwxrwx--x`) pass.
fn ensure_private_skel_dir(dir: &std::path::Path) -> Result<(), CeraError> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .map_err(|e| CeraError::Backend(format!("skel dir {}: {e}", dir.display())))?;
    let meta = std::fs::metadata(dir)
        .map_err(|e| CeraError::Backend(format!("skel dir {}: {e}", dir.display())))?;
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let euid = unsafe { libc::geteuid() };
    if skel_dir_unsafe(meta.uid(), euid, meta.mode()) {
        return Err(CeraError::Backend(format!(
            "skel dir {} must be owned by the current user and not world-writable \
             (uid {} vs {euid}, mode {:o})",
            dir.display(),
            meta.uid(),
            meta.mode() & 0o7777
        )));
    }
    Ok(())
}

/// Exclusively create `path` (mode 0644) and write `bytes`.
fn write_new_file(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o644)
        .open(path)?;
    f.write_all(bytes)?;
    // Flushed before the rename so a power loss cannot leave a zero-length
    // `.so` at the loader-visible path.
    f.sync_all()
}

/// Write the embedded skels into `dir` (created private, mode 0700, if
/// missing) and point FastRPC's loader at it via `ADSP_LIBRARY_PATH`.
/// Prepends to an existing value instead of replacing it (no-op when `dir` is already
/// listed). Returns the number of skels written; files are only rewritten
/// when their bytes differ, so repeated calls are cheap and idempotent.
///
/// Call once during single-threaded startup, before spawning threads that
/// read the environment: `set_var` is process-global and cannot be made
/// sound against foreign threads (JVM, loader) racing a `getenv` from
/// inside this process.
///
/// Why the env mutation cannot be made opt-in or skipped today: the JVM and
/// desktop flows reach this function through `hexagon_install_skels`, whose
/// contract is that it also points the loader at `dir`, so dropping the
/// `set_var` would silently break every caller that relies on it, and a flag
/// to skip it would change the UniFFI surface pinned by the api_contracts
/// snapshot. Kotlin `HexagonNpu.setup` therefore runs both mutations (this
/// one, then its `Os.setenv` that adds the vendor fallback paths) exactly
/// once, on the caller's thread, inside its `@Synchronized` setup at startup.
/// A cleaner split (a file-staging entry point that never touches the
/// environment, with the caller owning `setenv`) is a public API change and
/// belongs in a deliberate FFI revision.
///
/// The two staging flows (this function and Kotlin `HexagonNpu.setup`)
/// serialize internally (this side via `SKEL_ENV_LOCK`) but against
/// *different* monitors, so they may compose only sequentially, on one
/// thread, during single-threaded startup. Concurrent composition can
/// lost-update `ADSP_LIBRARY_PATH` and drop a skel dir.
///
/// Refuses an existing `dir` that another user owns or that is writable by
/// "other" (the loader executes what is in it), before touching anything.
pub fn install_skels(dir: &std::path::Path) -> Result<usize, CeraError> {
    // Held for the whole install: serializes the file writes (shared
    // `.so.tmp` names) as well as the `ADSP_LIBRARY_PATH` update below.
    let _env_guard = SKEL_ENV_LOCK.lock_or_recover();
    let dir_str = dir
        .to_str()
        .ok_or_else(|| CeraError::Backend("skel dir is not UTF-8".into()))?;
    if dir_str.contains(';') || dir_str.contains('\0') || dir_str.contains('=') {
        return Err(CeraError::Backend(format!(
            "skel dir {dir_str:?} contains ';', NUL bytes, or '=', which is invalid for loader search entries"
        )));
    }
    ensure_private_skel_dir(dir)?;
    let mut count = 0;
    for arch in PROBE_ARCHS {
        let bytes = embedded_skel(arch);
        let path = dir.join(arch.skel_filename());
        // Compare contents, not just size: a corrupt same-length skel must
        // still be repaired, or the DSP loads wrong code with no signal.
        // (Length stat first: no point reading a file already known stale.)
        let rewrite = match std::fs::metadata(&path) {
            Ok(m) if m.len() != bytes.len() as u64 => true,
            Ok(_) => match std::fs::read(&path) {
                Ok(existing) => existing.as_slice() != bytes,
                Err(_) => true,
            },
            Err(_) => true,
        };
        if rewrite {
            // Publish atomically: FastRPC reads the loader-visible path, so
            // a crash mid-write must never leave a truncated `.so` behind.
            // Unique per process: same-process installs are mutex-serialized,
            // but two app processes share filesDir and would otherwise share
            // one staging path (truncate race publishes a truncated `.so`).
            // Disjoint tmps make last-rename-wins correct (identical bytes).
            // A killed install may leave its `*.so.tmp.<pid>` behind; it is
            // never loader-visible and safe to delete.
            let tmp = path.with_extension(format!("so.tmp.{}", std::process::id()));
            let _ = std::fs::remove_file(&tmp);
            // Remove the staging file on any failure (ENOSPC, interrupted
            // write): its pid-suffixed name differs per launch, so leaks add up.
            // `create_new` (O_EXCL) never follows a pre-placed symlink.
            write_new_file(&tmp, bytes)
                .map_err(|e| CeraError::Backend(format!("write {}: {e}", tmp.display())))
                .and_then(|()| {
                    std::fs::rename(&tmp, &path)
                        .map_err(|e| CeraError::Backend(format!("rename {}: {e}", path.display())))
                })
                .inspect_err(|_| {
                    let _ = std::fs::remove_file(&tmp);
                })?;
            count += 1;
        }
    }
    remove_legacy_skels(dir, LEGACY_SKELS);
    // `;`-joined: the separator the FastRPC loader parses (verified with
    // the AAR flow on a retail S25U) and the one `HexagonNpu.setup` writes.
    // A `:`-joined value would reach the loader as one path containing a
    // literal `:`, failing skel resolution whenever the var is pre-set.
    // `var_os`, not `var(...).ok()`: a non-UTF8 current value must fail
    // loudly (same as a non-UTF8 dir above), not silently clobber someone
    // else's loader paths on the merge.
    let merged = merge_adsp_paths_os(dir_str, std::env::var_os("ADSP_LIBRARY_PATH").as_deref())?;
    // SAFETY: the mutex serializes every Rust-side `ADSP_LIBRARY_PATH`
    // read-modify-write; the caller upholds the documented startup-only
    // requirement, which is what keeps foreign-thread `getenv` out of the
    // race (no in-process primitive can enforce that half).
    unsafe { std::env::set_var("ADSP_LIBRARY_PATH", merged) };
    Ok(count)
}

/// A skel that earlier cera builds installed under the name llama.cpp's own skels use:
/// `(file name, length, FNV-1a 64 of the contents)`.
type LegacySkel = (&'static str, u64, u64);

/// The `libggml-htp-vXX.so` files cera shipped before the skels were renamed (the build before
/// the Q8 quantizer fix, and the one after it).
const LEGACY_SKELS: &[LegacySkel] = &[
    ("libggml-htp-v73.so", 890856, 0x9ff26275dfa72858),
    ("libggml-htp-v73.so", 890856, 0xd20698d252543138),
    ("libggml-htp-v75.so", 836680, 0x5086a2b100fe8ead),
    ("libggml-htp-v75.so", 836680, 0xab2f061fbf9e2e04),
    ("libggml-htp-v79.so", 853160, 0xd993ac714327ddab),
    ("libggml-htp-v79.so", 853160, 0x41da0f7f31f74734),
    ("libggml-htp-v81.so", 881960, 0x09fb2a033ba74d5e),
    ("libggml-htp-v81.so", 881928, 0x260b85a6333e67f0),
    ("libggml-htp-v85.so", 881960, 0x09fb2a033ba74d5e),
    ("libggml-htp-v85.so", 881928, 0x260b85a6333e67f0),
];

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

/// Delete the previous cera skels left in `dir` under llama.cpp's file names. The loader
/// searches `dir` before the other entries on `ADSP_LIBRARY_PATH`, so a stale copy there would
/// answer a coexisting llama.cpp runtime's request for its own skel with cera's old, ABI-skewed
/// one. Only a regular file whose length and contents match a known cera build goes: an
/// app-provided directory may legitimately hold llama.cpp's own skels under those names.
fn remove_legacy_skels(dir: &std::path::Path, known: &[LegacySkel]) {
    for &(name, len, hash) in known {
        let path = dir.join(name);
        let is_ours = std::fs::symlink_metadata(&path)
            .ok()
            .filter(|m| m.is_file() && m.len() == len)
            .and_then(|_| std::fs::read(&path).ok())
            .is_some_and(|b| fnv1a64(&b) == hash);
        if is_ours {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Merge a staged path list with the current `ADSP_LIBRARY_PATH` value:
/// staged entries first, `;`-joined, empty segments dropped, first
/// occurrence wins. This is the one merge contract both staging flows
/// implement (Kotlin `HexagonNpu.mergeAdspPaths` mirrors it case for
/// case, and the unit tests below mirror its truth table), so composing
/// the flows in either order (sequentially) yields the same dir set.
fn merge_adsp_paths(staged: &str, current: Option<&str>) -> String {
    let mut seen = std::collections::HashSet::<&str>::new();
    staged
        .split(';')
        .chain(current.unwrap_or("").split(';'))
        .filter(|p| !p.is_empty())
        .filter(|p| seen.insert(*p))
        .collect::<Vec<_>>()
        .join(";")
}

/// `merge_adsp_paths` over a possibly non-UTF8 current value: refuses to
/// merge (rather than clobber) when the conversion fails. Pure over
/// `&OsStr` so the refusal arm is unit-testable without touching the
/// process env (this file allows exactly one env writer).
fn merge_adsp_paths_os(
    staged: &str,
    current: Option<&std::ffi::OsStr>,
) -> Result<String, CeraError> {
    let current = match current {
        None => None,
        Some(v) => Some(v.to_str().ok_or_else(|| {
            CeraError::Backend(
                "ADSP_LIBRARY_PATH is not UTF-8; refusing to merge rather than clobber it".into(),
            )
        })?),
    };
    Ok(merge_adsp_paths(staged, current))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skel_dir_is_created_private_and_world_writable_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("a/b/skels");
        ensure_private_skel_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "newly created dir must be owner-only");
        // Android's app files dir is 0771: group access is fine.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o771)).unwrap();
        assert!(ensure_private_skel_dir(&dir).is_ok());
        // "Other" write access lets another local user swap a loaded skel.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(ensure_private_skel_dir(&dir).is_err());
    }

    #[test]
    fn install_skels_refuses_a_world_writable_dir_before_writing() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("skels");
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        // The public entry, not just the helper: swapping the directory check
        // for a plain `create_dir_all` must fail this test.
        assert!(install_skels(&dir).is_err());
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "nothing may be written into a refused dir"
        );
    }

    #[test]
    fn legacy_skels_are_removed_only_when_they_are_ours() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        let ours = b"a previous cera skel".as_slice();
        let known: &[LegacySkel] = &[
            ("libggml-htp-v73.so", ours.len() as u64, fnv1a64(ours)),
            ("libggml-htp-v75.so", ours.len() as u64, fnv1a64(ours)),
            ("libggml-htp-v79.so", ours.len() as u64, fnv1a64(ours)),
        ];
        std::fs::write(dir.join("libggml-htp-v79.so"), ours).unwrap();
        // Same name, same length, other contents: llama.cpp's own copy, kept.
        let theirs = b"A previous cera skel".as_slice();
        assert_eq!(theirs.len(), ours.len());
        std::fs::write(dir.join("libggml-htp-v75.so"), theirs).unwrap();
        // A link to a matching file is never followed or removed (the link is examined before
        // anything else is deleted, and its target outlives the call).
        std::fs::write(dir.join("elsewhere.so"), ours).unwrap();
        std::os::unix::fs::symlink(dir.join("elsewhere.so"), dir.join("libggml-htp-v73.so"))
            .unwrap();
        remove_legacy_skels(dir, known);
        assert!(!dir.join("libggml-htp-v79.so").exists(), "ours is removed");
        assert!(dir.join("elsewhere.so").exists());
        assert_eq!(
            std::fs::read(dir.join("libggml-htp-v75.so")).unwrap(),
            theirs
        );
        assert!(dir.join("libggml-htp-v73.so").symlink_metadata().is_ok());
    }

    /// The previous builds, as `(name, length, FNV-1a 64)` of the blobs at the commits that
    /// shipped them (before and after the Q8 quantizer fix). An edit here is deliberate: a wrong
    /// value silently leaves a stale skel installed.
    #[test]
    fn the_legacy_table_pins_the_shipped_builds() {
        let want: &[LegacySkel] = &[
            ("libggml-htp-v73.so", 890856, 0x9ff26275dfa72858),
            ("libggml-htp-v73.so", 890856, 0xd20698d252543138),
            ("libggml-htp-v75.so", 836680, 0x5086a2b100fe8ead),
            ("libggml-htp-v75.so", 836680, 0xab2f061fbf9e2e04),
            ("libggml-htp-v79.so", 853160, 0xd993ac714327ddab),
            ("libggml-htp-v79.so", 853160, 0x41da0f7f31f74734),
            ("libggml-htp-v81.so", 881960, 0x09fb2a033ba74d5e),
            ("libggml-htp-v81.so", 881928, 0x260b85a6333e67f0),
            ("libggml-htp-v85.so", 881960, 0x09fb2a033ba74d5e),
            ("libggml-htp-v85.so", 881928, 0x260b85a6333e67f0),
        ];
        assert_eq!(LEGACY_SKELS, want);
    }

    #[test]
    fn the_legacy_table_names_every_renamed_skel() {
        for arch in PROBE_ARCHS {
            let old = arch.skel_filename().replace("libcera-htp", "libggml-htp");
            assert!(
                LEGACY_SKELS.iter().any(|&(n, ..)| n == old),
                "{old} has no legacy entry"
            );
        }
    }

    #[test]
    fn write_new_file_refuses_an_existing_path_or_symlink() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("x.so");
        write_new_file(&file, b"one").unwrap();
        assert!(
            write_new_file(&file, b"two").is_err(),
            "O_EXCL: no overwrite"
        );
        let link = root.path().join("y.so");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert!(
            write_new_file(&link, b"three").is_err(),
            "never follows a link"
        );
        assert_eq!(std::fs::read(&file).unwrap(), b"one");
    }

    /// Restores `ADSP_LIBRARY_PATH` on drop, so a mid-test panic cannot
    /// leak a dirty loader var into the rest of the suite (the manual
    /// trailing restore it replaces could).
    struct RestoreAdspPath(Option<std::ffi::OsString>);

    impl Drop for RestoreAdspPath {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => unsafe { std::env::set_var("ADSP_LIBRARY_PATH", v) },
                None => unsafe { std::env::remove_var("ADSP_LIBRARY_PATH") },
            }
        }
    }

    /// Truth-table pins for [`merge_adsp_paths`], mirroring Kotlin
    /// `MergeAdspPathsTest` case for case: `;` separator (a `:`-joined
    /// value passes through as one opaque entry), staged-first order,
    /// first-wins dedup, empty segments dropped.
    #[test]
    fn merge_adsp_paths_truth_table() {
        assert_eq!(
            merge_adsp_paths("/data/app/lib;/odm/lib/rfsa/adsp", None),
            "/data/app/lib;/odm/lib/rfsa/adsp"
        );
        assert_eq!(
            merge_adsp_paths("/data/app/lib;/vendor/dsp", Some("/staged/by/rust")),
            "/data/app/lib;/vendor/dsp;/staged/by/rust"
        );
        assert_eq!(
            merge_adsp_paths(
                "/data/app/lib;/vendor/dsp",
                Some("/vendor/dsp;/data/app/lib")
            ),
            "/data/app/lib;/vendor/dsp"
        );
        assert_eq!(
            merge_adsp_paths("/data/app/lib;", Some(";/data/app/lib;")),
            "/data/app/lib"
        );
        assert_eq!(
            merge_adsp_paths("/data/app/lib", Some("/a:/b")),
            "/data/app/lib;/a:/b"
        );
        // Empty-string current contributes no entries, exactly like `None`
        // (mirrored by the Kotlin table's empty-string case).
        assert_eq!(merge_adsp_paths("/data/app/lib", Some("")), "/data/app/lib");
        // Idempotent: merging the merged value changes nothing.
        let once = merge_adsp_paths("/d", Some("/v;/o"));
        assert_eq!(merge_adsp_paths("/d", Some(&once)), once);
    }

    // Unix-only: fabricating a non-UTF8 `OsStr` needs `OsStringExt`
    // (`ADSP_LIBRARY_PATH` itself is an Android/Linux concept).
    #[cfg(unix)]
    #[test]
    fn merge_refuses_non_utf8_current() {
        use std::os::unix::ffi::OsStringExt;
        // A non-UTF8 current value fails loudly instead of being silently
        // clobbered on the merge (pure `&OsStr` helper: no env touched).
        let bad = std::ffi::OsString::from_vec(vec![0xff, 0xfe]);
        let err = merge_adsp_paths_os("/d", Some(&bad)).unwrap_err();
        assert!(
            matches!(err, CeraError::Backend(ref s) if s.contains("not UTF-8")),
            "{err:?}"
        );
        // UTF-8 and absent currents merge exactly like `merge_adsp_paths`.
        assert_eq!(
            merge_adsp_paths_os("/d", Some(std::ffi::OsStr::new("/v;/o"))).unwrap(),
            "/d;/v;/o"
        );
        assert_eq!(merge_adsp_paths_os("/d", None).unwrap(), "/d");
    }

    /// Pins the `install_skels` contract the FFI docs promise: count means
    /// files written (0 when fresh), same-length corruption is repaired,
    /// the loader path is not duplicated, and re-running is a no-op.
    #[test]
    fn install_is_idempotent_and_repairs_corruption() {
        // NOTE: process-global env assertion; safe only because no other
        // test in this binary may write `ADSP_LIBRARY_PATH`, not even via
        // a restore guard's `Drop`, which would race these exact-merge
        // assertions just the same. Save/restore so the suite leaves no
        // trace either way.
        let _restore = RestoreAdspPath(std::env::var_os("ADSP_LIBRARY_PATH"));
        let dir = std::env::temp_dir().join(format!("skel-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // First install writes every skel...
        assert_eq!(install_skels(&dir).unwrap(), PROBE_ARCHS.len());
        // ...second is a no-op returning 0, env path not duplicated.
        assert_eq!(install_skels(&dir).unwrap(), 0);
        let v = std::env::var("ADSP_LIBRARY_PATH").unwrap();
        assert_eq!(
            v.split(';').filter(|p| *p == dir.to_str().unwrap()).count(),
            1
        );
        // A pre-set `;`-joined value merges (same separator the FastRPC
        // loader and `HexagonNpu.setup` use), still without duplicating.
        unsafe {
            std::env::set_var(
                "ADSP_LIBRARY_PATH",
                "/vendor/lib/rfsa/adsp;/odm/lib/rfsa/adsp",
            )
        };
        assert_eq!(install_skels(&dir).unwrap(), 0);
        assert_eq!(
            std::env::var("ADSP_LIBRARY_PATH").unwrap(),
            format!("{};/vendor/lib/rfsa/adsp;/odm/lib/rfsa/adsp", dir.display())
        );
        assert_eq!(install_skels(&dir).unwrap(), 0);
        assert_eq!(
            std::env::var("ADSP_LIBRARY_PATH").unwrap(),
            format!("{};/vendor/lib/rfsa/adsp;/odm/lib/rfsa/adsp", dir.display())
        );
        // Same-length corruption is still repaired.
        let p = dir.join(PROBE_ARCHS[0].skel_filename());
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(install_skels(&dir).unwrap(), 1);
        assert_eq!(std::fs::read(&p).unwrap(), embedded_skel(PROBE_ARCHS[0]));

        let _ = std::fs::remove_dir_all(&dir);
        // Env restore is the `_restore` guard's `Drop` (panic-safe).
    }

    /// A staging dir containing `;`, `=`, or `\0` fails closed instead of
    /// corrupting the loader path list.
    #[test]
    fn install_rejects_semicolon_dir() {
        // No restore guard here on purpose: the rejection returns before
        // any env write, so there is nothing to restore, and a guard's
        // `Drop` would race the idempotent test's phased assertions above
        // (see its NOTE). A second env writer in this binary is forbidden.
        let err_semi = install_skels(std::path::Path::new("/tmp/skel-test-;")).unwrap_err();
        assert!(
            err_semi.to_string().contains("';'"),
            "unexpected error: {err_semi}"
        );

        let err_eq = install_skels(std::path::Path::new("/tmp/skel-test-=")).unwrap_err();
        assert!(
            err_eq.to_string().contains("'='"),
            "unexpected error: {err_eq}"
        );

        let err_nul = install_skels(std::path::Path::new("/tmp/skel-test-\0")).unwrap_err();
        assert!(
            err_nul.to_string().contains("NUL"),
            "unexpected error: {err_nul}"
        );
    }

    #[test]
    fn skel_dir_unsafe_checks_owner_and_other_write() {
        assert!(!skel_dir_unsafe(1000, 1000, 0o700));
        assert!(!skel_dir_unsafe(1000, 1000, 0o771));
        assert!(skel_dir_unsafe(0, 1000, 0o700), "foreign owner");
        assert!(skel_dir_unsafe(1000, 1000, 0o777), "world-writable");
        assert!(skel_dir_unsafe(1000, 1000, 0o702));
    }
}

/// Successful NPU probe result: the working arch plus DSP capabilities.
#[derive(Debug, Clone)]
pub struct HexagonProbe {
    pub arch: HexagonArch,
    pub n_threads: u32,
    pub n_hvx: u32,
    pub n_hmx: u32,
    pub vtcm_bytes: u64,
}

/// Open the FastRPC driver, try each bundled skel in [`PROBE_ARCHS`]
/// order, and return the first working device's capabilities. The probe
/// device is dropped before returning; use [`HexagonDevice`] for real
/// sessions. On failure the error aggregates every arch's message.
///
/// Honors the `CERA_DISABLE_HEXAGON` / `CERA_NO_HEXAGON` kill switches before
/// touching the driver: a probe that reported a working NPU (and took a power
/// vote) while every loader refuses would mislead callers.
pub fn probe() -> Result<HexagonProbe, CeraError> {
    probe_with(|k| std::env::var_os(k), FastRpcDriver::load)
}

/// [`probe`] with the env lookup and driver loader injected (tests).
pub(crate) fn probe_with(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
    load: impl FnOnce() -> Result<Arc<FastRpcDriver>, CeraError>,
) -> Result<HexagonProbe, CeraError> {
    super::ensure_not_disabled_with(get)?;
    let driver = load()?;
    let mut errors = Vec::new();
    for arch in PROBE_ARCHS {
        match HexagonDevice::new(Arc::clone(&driver), arch) {
            Ok(dev) => {
                let hw = dev.hw_info();
                return Ok(HexagonProbe {
                    arch: dev.arch(),
                    n_threads: hw.n_threads,
                    n_hvx: hw.n_hvx,
                    n_hmx: hw.n_hmx,
                    vtcm_bytes: hw.vtcm_size,
                });
            }
            Err(e) => errors.push(format!("{arch:?}: {e}")),
        }
    }
    Err(CeraError::Backend(format!(
        "no working Hexagon NPU ({}); {}",
        PROBE_ARCHS
            .iter()
            .map(|a| a.skel_filename())
            .collect::<Vec<_>>()
            .join(", "),
        errors.join("; ")
    )))
}

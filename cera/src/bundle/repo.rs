//! Native bundle store: HTTP download + on-disk cache.
//!
//! The URL-to-cache-entry addressing this uses lives in
//! [`super::cache_key`], shared with the browser store; everything here
//! is the part that needs a filesystem and reqwest.
//!
//! `BundleRepo` resolves a remote URL to a file in a caller-chosen
//! cache directory. This is Phase 1.6 PR A scope: it does *not* resolve
//! bundle IDs like `"LiquidAI/LFM2-1.2B-GGUF"` to a manifest URL (PR B);
//! it takes a direct URL (typically pulled from a manifest's `files`
//! entries) and returns a local path.
//!
//! ## Caching
//!
//! Files are stored under `<store_dir>/<host>/<url-path>`, mirroring the
//! URL structure so contents are trivially inspectable and swappable
//! with a CDN mirror that preserves paths.
//!
//! ## `store_dir`, not `cache_dir`
//!
//! The directory the caller supplies is named `store_dir` on purpose.
//! On Android the consumer is expected to pass `Context.getFilesDir()`
//! (persistent storage), **not** `Context.getCacheDir()` — the latter
//! is OS-purgeable under storage pressure and would cause silent,
//! expensive re-downloads. Desktop and server callers typically pass
//! something like `~/.cache/cera/` but the crate never hardcodes a
//! default location; it's always caller-supplied.
//!
//! ## Integrity
//!
//! Each download is SHA-256'd on the fly and compared against either a
//! caller-supplied hash (via the `expected_sha256` argument to
//! [`BundleRepo::resolve_url`]) or the server's `X-Linked-Etag`
//! header (HuggingFace sets this for LFS objects — content-addressed,
//! stable across revisions). The successful hash is persisted as
//! `<dest>.sha256` alongside the cached file; subsequent cache hits
//! read the sidecar and compare it against the etag in O(1) rather
//! than re-hashing multi-GB files on every resolve. A missing or
//! stale sidecar triggers a full rehash (which also repairs the
//! sidecar on success).
//!
//! A cached file is considered valid when:
//! 1. A caller-supplied hash matches the sidecar (or full rehash
//!    fallback), or
//! 2. HEAD provides `X-Linked-Etag` and it matches the sidecar (or
//!    full rehash fallback), or
//! 3. HEAD provides only `Content-Length` and the sizes match, or
//! 4. HEAD fails entirely — reuse whatever's cached so a transient
//!    upstream blip doesn't defeat a CI cache hit.
//!
//! See `download::head_info` for the HEAD probe.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use reqwest::blocking::Client;

use super::cache_key::{
    LEAP_BUNDLES_API_URL, LeapBundleEntry, cache_relative_segments, parse_leap_bundles,
};
use super::download;
use crate::session::CeraError;

/// Callback receiver for download progress events. Implementations
/// must be `Send + Sync` because downloads run on whatever thread
/// the caller drove them from (often a `spawn_blocking` worker).
///
/// Throttling: `download::download_to` calls `on_progress` at most
/// once per ~256 KB written + once at end-of-stream. Implementers
/// don't need to dedupe or rate-limit on their side.
pub trait DownloadProgress: Send + Sync + std::fmt::Debug {
    /// Called periodically during a download. `bytes_downloaded` is
    /// monotonic across the same call's stream; `total_bytes` is the
    /// `Content-Length` reported by the server (may be `None` for
    /// chunked-transfer responses or when HEAD didn't surface a
    /// length). Same `url` value across all calls for one download.
    fn on_progress(&self, url: &str, bytes_downloaded: u64, total_bytes: Option<u64>);
}

/// Repository for remote bundle files cached to a caller-chosen
/// directory. Construction is cheap — create one per `CeraEngine` at
/// most, or pass the same instance to multiple engines.
///
/// Holds two pooled `reqwest::blocking::Client`s:
/// - `http_client`: default redirect policy. Used for the `GET`
///   streaming-download path so HF's 302 to the CDN is followed
///   automatically.
/// - `head_client`: redirects **disabled**. Used for `HEAD` probes so
///   the code reads headers from HF's origin-hop 302 (which carries
///   `X-Linked-Etag`, the content SHA-256). A redirect-following
///   client would surface the CDN's unrelated `ETag` instead.
#[derive(Clone, Debug)]
pub struct BundleRepo {
    store_dir: PathBuf,
    http_client: Client,
    head_client: Client,
    /// Optional progress callback fired during cache-miss downloads.
    /// `None` for `BundleRepo::new`; populated by `with_progress`.
    /// Cache-hit resolves don't fire any callbacks since there's no
    /// streaming work to report on.
    progress: Option<Arc<dyn DownloadProgress>>,
}

impl BundleRepo {
    /// Create a new repo rooted at `store_dir`. The directory does not
    /// need to exist yet — it will be created on the first download.
    ///
    /// Constructs two clients (see [`BundleRepo`] docs for the
    /// redirect-policy split). Both use reqwest's defaults otherwise;
    /// per-request timeouts override at the call site (30s for HEAD,
    /// 10min for GET). `Client::builder().build()` panics only on
    /// severe OS resource failure (can't create a Tokio runtime) —
    /// documented failure mode, acceptable for a process-startup path.
    pub fn new(store_dir: impl Into<PathBuf>) -> Self {
        let head_client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            store_dir: store_dir.into(),
            http_client: Client::new(),
            head_client,
            progress: None,
        }
    }

    /// Variant of [`BundleRepo::new`] that attaches a progress
    /// callback. The callback fires during cache-miss downloads only
    /// (cache hits return without streaming I/O). Same callback gets
    /// called for every URL the repo downloads — implementers can
    /// branch on the `url` argument to drive a per-file UI.
    pub fn with_progress(
        store_dir: impl Into<PathBuf>,
        progress: Arc<dyn DownloadProgress>,
    ) -> Self {
        let mut repo = Self::new(store_dir);
        repo.progress = Some(progress);
        repo
    }

    /// Root directory backing this repo.
    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// Progress callback attached to this repo, if any.
    pub fn progress(&self) -> Option<Arc<dyn DownloadProgress>> {
        self.progress.clone()
    }

    /// Total bytes currently held in the cache. Recursively sums file
    /// sizes under `store_dir`; returns `Ok(0)` if the dir doesn't
    /// exist yet (no downloads have run). Skips files that fail to
    /// `metadata()` (deleted mid-walk, permission glitches) — partial
    /// totals beat hard-erroring on a transient I/O blip.
    ///
    /// O(n) over the cache contents; for a large cache (multiple GB
    /// across many shards) this is a real walk, not a constant-time
    /// query — the OS doesn't track per-directory totals. Callers
    /// surfacing the value in a UI should run it off the main thread.
    pub fn cache_size(&self) -> Result<u64, CeraError> {
        // No `exists()` pre-check — `walk_dir_size` already maps a
        // missing root to `Ok(())` via `read_dir`'s `NotFound` arm,
        // which leaves `total` at zero. Skipping the extra syscall
        // also closes the same TOCTOU we close in `clear_cache`.
        let mut total = 0u64;
        Self::walk_dir_size(&self.store_dir, &mut total)?;
        Ok(total)
    }

    fn walk_dir_size(dir: &Path, total: &mut u64) -> Result<(), CeraError> {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            // Directory disappeared mid-walk (concurrent clear, etc.) —
            // treat as empty rather than propagate.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for entry in entries.flatten() {
            // `entry.file_type()` is a syscall on POSIX, so use it
            // instead of `metadata()` for the directory test — cheaper
            // when we don't need the size yet.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                Self::walk_dir_size(&entry.path(), total)?;
            } else if file_type.is_file()
                && let Ok(meta) = entry.metadata()
            {
                *total = total.saturating_add(meta.len());
            }
        }
        Ok(())
    }

    /// Wipe every file the repo has cached, leaving `store_dir` itself
    /// in place (so subsequent downloads land in the same path).
    /// Idempotent — calling on an empty repo or non-existent
    /// `store_dir` is a no-op success.
    ///
    /// Mobile apps trigger this from a "clear downloaded models" UI
    /// action. Removes the tree under `store_dir` (sidecar `.sha256`
    /// files included) and recreates `store_dir` empty. In-flight
    /// downloads to the same repo will see the partial files vanish
    /// and may fail; callers should serialize the clear against
    /// any active `from_bundle_id*` calls themselves (typically
    /// trivial since the action is user-driven).
    pub fn clear_cache(&self) -> Result<(), CeraError> {
        // `remove_dir_all` + `create_dir_all` is simpler than walking
        // and `unlink`-ing each file. The dir-recreate keeps the
        // store_dir invariant (parent for future downloads).
        //
        // `NotFound` from `remove_dir_all` is treated as success: the
        // dir is already absent (lazy-creation invariant — no download
        // has run, or a concurrent clear got there first). In that
        // case we also skip `create_dir_all` to preserve the lazy
        // contract: nothing eagerly materializes `store_dir`. No
        // `exists()` pre-check — it would be a TOCTOU race against
        // the remove.
        match fs::remove_dir_all(&self.store_dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        fs::create_dir_all(&self.store_dir)?;
        Ok(())
    }

    /// Resolve a remote URL to a local path, downloading if not cached.
    ///
    /// `expected_sha256`: if provided, the cached entry's hash (or the
    /// freshly-downloaded bytes' hash) must match this exactly. If
    /// `None`, integrity verification falls back to the server's
    /// `X-Linked-Etag` (when present) or a size check (when only
    /// `Content-Length` is available).
    ///
    /// ### Verification policy
    ///
    /// On cache hit:
    /// - If a sidecar `<dest>.sha256` exists, compare it against the
    ///   expected hash in O(1). This is the fast path for multi-GB
    ///   cached GGUFs.
    /// - Else fall back to `sha256_file` (full rehash) and repair the
    ///   sidecar on success.
    /// - If no hash is available anywhere, fall back to a
    ///   `Content-Length` size check.
    /// - If HEAD also fails, reuse the cached file (transient outage
    ///   shouldn't defeat a CI cache hit).
    ///
    /// On cache miss: download, hashing on the fly, verifying against
    /// `expected_sha256` or `X-Linked-Etag`. A mismatch deletes the
    /// partial and returns `CeraError::Backend`. The sidecar is
    /// persisted on success.
    pub fn resolve_url(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
    ) -> Result<PathBuf, CeraError> {
        let dest = self.path_for_url(url)?;

        // Probe HEAD once up front (no-redirect client so HF's
        // `X-Linked-Etag` is captured from the first hop). Used both
        // to validate a cached file and — on cache miss — to hand the
        // server's advertised content hash to `download_to`, so the
        // first download is integrity-verified even when the caller
        // didn't pin a hash. Re-used across cache-hit and cache-miss
        // code paths so we issue at most one HEAD per resolve.
        let head = if expected_sha256.is_some() {
            // Caller pinned a hash — no need to consult the server.
            download::HeadInfo {
                content_length: None,
                linked_sha256: None,
            }
        } else {
            download::head_info(&self.head_client, url)
        };

        if dest.exists() && self.cache_hit_valid(&dest, url, expected_sha256, &head) {
            return Ok(dest);
        }

        // Cache miss: only now does the filesystem need to exist.
        // Deferred so cache-hit callers don't pay a `stat` on the
        // parent directory on every resolve.
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        // Prefer caller's hash; else use the server's linked-etag
        // captured pre-redirect.
        let download_hash = expected_sha256
            .map(|s| s.to_ascii_lowercase())
            .or_else(|| head.linked_sha256.clone());

        tracing::info!(
            target: "cera::bundle",
            url,
            dest = %dest.display(),
            hash_source = match (expected_sha256.is_some(), head.linked_sha256.is_some()) {
                (true, _) => "caller",
                (false, true) => "x-linked-etag",
                (false, false) => "unverified",
            },
            "downloading bundle file"
        );
        download::download_to(
            &self.http_client,
            url,
            &dest,
            download_hash.as_deref(),
            // HEAD probe captures `x-linked-size` from HF's no-redirect
            // response. The GET path's CDN response usually echoes
            // `Content-Length` too, but defensively prefer the HEAD-
            // probed value so the progress callback gets a reliable
            // total even when the CDN omits the header.
            head.content_length,
            self.progress.as_deref(),
        )?;
        Ok(dest)
    }

    /// Decide whether an existing cached entry at `dest` is still
    /// valid. Verification prefers caller-supplied hash, then etag via
    /// sidecar, then etag via full rehash, then size, then reuse-on-
    /// HEAD-failure. Any mismatch returns `false` → caller re-downloads.
    fn cache_hit_valid(
        &self,
        dest: &Path,
        url: &str,
        expected_sha256: Option<&str>,
        head: &download::HeadInfo,
    ) -> bool {
        // Caller hash takes precedence over whatever the server
        // advertised — lets manifest-level hashes override an etag
        // that's been rotated.
        let expected_hash = expected_sha256
            .map(|s| s.to_ascii_lowercase())
            .or_else(|| head.linked_sha256.clone());

        if let Some(exp_hash) = expected_hash {
            return hash_matches(dest, url, &exp_hash);
        }

        if let Some(exp_len) = head.content_length {
            let actual = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
            if actual == exp_len {
                return true;
            }
            tracing::info!(
                target: "cera::bundle",
                url,
                expected = exp_len,
                actual,
                "cached file size mismatch; re-downloading"
            );
            return false;
        }

        // HEAD failed entirely — best-effort reuse so a transient
        // outage doesn't defeat a CI cache hit.
        true
    }

    /// Compute the on-disk cache location for `url`, rooted at
    /// `store_dir`. Mirrors `<host>/<path>` so the cache is inspectable
    /// and safely swappable with a host-preserving mirror.
    ///
    /// The addressing itself, and the allowlist that keeps a hostile URL
    /// from escaping the store root, live in
    /// [`super::cache_key::cache_relative_segments`] so the browser store maps
    /// a URL to the same entry. All this adds is the join against
    /// `store_dir`.
    fn path_for_url(&self, url: &str) -> Result<PathBuf, CeraError> {
        let mut out = self.store_dir.clone();
        // `PathBuf::push` is not a validator; what keeps the result
        // inside `store_dir` is that every segment already passed
        // `validate_path_segment` on the way out of `cache_key`.
        out.extend(cache_relative_segments(url)?);
        Ok(out)
    }
}

/// Check whether `dest` hashes to `expected_hash` (case-insensitive
/// hex), preferring the sidecar fast path. Logs a tracing event when
/// a full rehash is performed or a mismatch is detected.
fn hash_matches(dest: &Path, url: &str, expected_hash: &str) -> bool {
    let expected = expected_hash.to_ascii_lowercase();

    // Fast path: trust the sidecar. We wrote it ourselves after the
    // last successful verification, so it's at least as trustworthy
    // as the cached file itself. `read_sidecar` returns lowercase.
    if let Some(cached) = download::read_sidecar(dest) {
        if cached == expected {
            return true;
        }
        tracing::info!(
            target: "cera::bundle",
            url,
            expected = %expected,
            actual = %cached,
            "cached file sidecar hash mismatch; re-downloading"
        );
        return false;
    }

    // Slow path: full rehash. Only hits when the sidecar is absent
    // (e.g. cached before the sidecar feature shipped) or unreadable.
    // `sha256_file` returns lowercase hex. On a match, persist the
    // sidecar so the next cache hit skips straight to the fast path.
    match download::sha256_file(dest) {
        Ok(actual) if actual == expected => {
            download::write_sidecar(dest, &actual);
            true
        }
        Ok(actual) => {
            tracing::info!(
                target: "cera::bundle",
                url,
                expected = %expected,
                actual = %actual,
                "cached file hash mismatch; re-downloading"
            );
            false
        }
        Err(e) => {
            tracing::warn!(
                target: "cera::bundle",
                url,
                error = %e,
                "failed to hash cached file; re-downloading"
            );
            false
        }
    }
}

/// HTTP timeout for `list_leap_bundles`. The HF model-info endpoint
/// returns a few KB of JSON — anything past 30 s is a stalled
/// connection (captive portal, network glitch) rather than a slow
/// response. Matches `HEAD_TIMEOUT` in `download.rs`.
const LIST_BUNDLES_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// List bundles published on `LiquidAI/LeapBundles`.
///
/// Single GET to the HuggingFace model-info endpoint; groups the
/// returned `siblings` array by directory to surface
/// `<bundle>/<quant>` pairs. Top-level files (`README.md`,
/// `.gitattributes`, the `*.bundle` blobs Liquid publishes for
/// their own packaging tool) are ignored — only entries shaped
/// as `<bundle>/<quant>.json` count, which is the schema the
/// rest of `cera` consumes via [`super::leap_bundles_manifest_url`].
///
/// Network: blocking GET with a 30 s timeout (see
/// `LIST_BUNDLES_TIMEOUT`) so a captive portal or stalled
/// connection surfaces as an error instead of hanging the CLI.
/// No retry. Caller errors surface as [`CeraError::Backend`] with
/// the underlying reqwest message.
pub fn list_leap_bundles() -> Result<Vec<LeapBundleEntry>, CeraError> {
    // Build a one-shot client per call. `list_leap_bundles` runs at
    // most once per CLI invocation, so the cost is well below the
    // network round-trip; sharing a `Client` with `BundleRepo`
    // would force callers (FFI consumers, tests) to thread it
    // through, and the API stays simpler this way.
    let client = Client::builder()
        .timeout(LIST_BUNDLES_TIMEOUT)
        .build()
        .map_err(|e| CeraError::Backend(format!("list-bundles client build failed: {e}")))?;
    let body = client
        .get(LEAP_BUNDLES_API_URL)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.text())
        .map_err(|e| CeraError::Backend(format!("list-bundles HTTP failed: {e}")))?;
    let mut entries = parse_leap_bundles(&body)?;
    if !entries.iter().any(|e| e.name == "LFM2.5-VL-3B-GGUF") {
        entries.push(LeapBundleEntry {
            name: "LFM2.5-VL-3B-GGUF".to_string(),
            quants: vec![
                "BF16".to_string(),
                "F16".to_string(),
                "Q4_0".to_string(),
                "Q4_K_M".to_string(),
                "Q5_K_M".to_string(),
                "Q6_K".to_string(),
                "Q8_0".to_string(),
            ],
        });
        entries.sort_by(|a, b| a.name.cmp(&b.name));
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pick a temp path scoped to this process + test name so parallel
    /// test binaries and prior runs don't collide. tempfile is already
    /// a workspace dev-dep (used by other bundle tests below).
    fn unique_test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cera-bundle-test-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// `cache_size` returns 0 when the repo's `store_dir` doesn't
    /// exist yet (no downloads have run). Lazy-creation invariant:
    /// constructing a `BundleRepo` doesn't touch the disk; querying
    /// size before any download returns 0 cleanly.
    #[test]
    fn cache_size_is_zero_when_store_dir_missing() {
        let dir = unique_test_dir("size-empty");
        let repo = BundleRepo::new(&dir);
        assert!(
            !dir.exists(),
            "BundleRepo::new must not eagerly create store_dir"
        );
        assert_eq!(repo.cache_size().unwrap(), 0);
    }

    /// `cache_size` walks nested directories and sums file sizes.
    /// Builds a small synthetic cache (3 files of known sizes spread
    /// across two subdirectories), then asserts the total matches the
    /// sum.
    #[test]
    fn cache_size_sums_nested_files() {
        let dir = unique_test_dir("size-sum");
        fs::create_dir_all(dir.join("huggingface.co/LiquidAI/A")).unwrap();
        fs::create_dir_all(dir.join("huggingface.co/LiquidAI/B")).unwrap();
        fs::write(dir.join("huggingface.co/LiquidAI/A/file1"), vec![0u8; 1024]).unwrap();
        fs::write(
            dir.join("huggingface.co/LiquidAI/A/file1.sha256"),
            b"deadbeef",
        )
        .unwrap();
        fs::write(dir.join("huggingface.co/LiquidAI/B/file2"), vec![0u8; 4096]).unwrap();

        let repo = BundleRepo::new(&dir);
        assert_eq!(repo.cache_size().unwrap(), 1024 + 8 + 4096);

        let _ = fs::remove_dir_all(&dir);
    }

    /// `clear_cache` is idempotent: calling on a non-existent
    /// `store_dir` is a no-op success. Mobile apps invoking it from
    /// a "clear cache" UI before any download has run shouldn't crash.
    #[test]
    fn clear_cache_is_idempotent_on_missing_store_dir() {
        let dir = unique_test_dir("clear-empty");
        let repo = BundleRepo::new(&dir);
        assert!(!dir.exists());
        repo.clear_cache().unwrap();
        // Still no eager creation.
        assert!(!dir.exists());
    }

    /// `clear_cache` removes all files but leaves `store_dir` itself
    /// in place (so subsequent downloads land in the same path).
    /// Asserts the dir still exists + is empty after.
    #[test]
    fn clear_cache_wipes_files_but_keeps_store_dir() {
        let dir = unique_test_dir("clear-wipe");
        fs::create_dir_all(dir.join("huggingface.co/LiquidAI/A")).unwrap();
        fs::write(dir.join("huggingface.co/LiquidAI/A/file"), vec![0u8; 100]).unwrap();
        let repo = BundleRepo::new(&dir);
        assert_eq!(repo.cache_size().unwrap(), 100);

        repo.clear_cache().unwrap();
        assert!(dir.exists(), "store_dir must survive clear_cache");
        assert_eq!(repo.cache_size().unwrap(), 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_matches_uses_sidecar_fast_path() {
        // Covers the first-class invariant behind the PR #37 review's
        // concern about rehashing large cached GGUFs: when a sidecar
        // exists and matches, `hash_matches` must return `true`
        // WITHOUT touching the file contents. We prove the latter by
        // never writing file contents at all — only a sidecar.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        std::fs::write(&dest, b"").unwrap();
        let hex = "0123456789abcdef".repeat(4);
        assert_eq!(hex.len(), 64);
        std::fs::write(download::sidecar_path(&dest), &hex).unwrap();

        assert!(hash_matches(&dest, "https://example.com/x", &hex));
        // Case-insensitive match.
        assert!(hash_matches(
            &dest,
            "https://example.com/x",
            &hex.to_uppercase()
        ));
        // Mismatch returns false without panicking.
        let wrong = "f".repeat(64);
        assert!(!hash_matches(&dest, "https://example.com/x", &wrong));
    }

    #[test]
    fn hash_matches_full_rehash_when_no_sidecar() {
        // When the sidecar is absent, `hash_matches` falls back to
        // hashing the file.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.bin");
        std::fs::write(&dest, b"hello").unwrap();
        let correct = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let wrong = "0".repeat(64);
        assert!(hash_matches(&dest, "https://example.com/x", correct));
        assert!(!hash_matches(&dest, "https://example.com/x", &wrong));
    }
}

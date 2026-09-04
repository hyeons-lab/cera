//! HTTP downloader with streaming SHA-256 hashing + atomic cache writes.
//!
//! Used by `BundleRepo` to fetch manifest / GGUF files from a remote URL
//! into a local cache. Gated behind the `remote` feature so wasm and
//! minimal-footprint builds don't pull `reqwest`.
//!
//! ## Integrity policy
//!
//! Downloaded bytes are hashed on the fly (SHA-256) and compared against
//! one of:
//!
//! 1. **Caller-supplied** `expected_sha256` on `BundleRepo::resolve_url`.
//!    Lets a caller (e.g. a manifest with per-file hashes) pin an exact
//!    value regardless of what the server advertises.
//! 2. **`X-Linked-Etag`**: HuggingFace serves LFS objects with an
//!    `X-Linked-Etag: sha256:<hex>` header. **Critical detail:** this
//!    header is only present on the first-hop response (the 302
//!    redirect from `huggingface.co`). The final CDN response after
//!    the redirect carries a different `ETag` that is the CAS storage
//!    key, NOT the file's content SHA-256. [`head_info`] therefore
//!    uses a no-redirect client so it reads headers from the origin's
//!    302. `BundleRepo::resolve_url` then threads the captured
//!    linked-etag into [`download_to`] as `expected_sha256`, so the
//!    cache-miss path is integrity-verified even when the caller
//!    didn't pin a hash.
//!
//! When neither a caller hash nor a server etag is available, the
//! download succeeds without hash verification (HTTPS still protects
//! transport; callers can tighten by plumbing an explicit hash).
//!
//! On mismatch the partial file is deleted and the caller receives
//! `CeraError::Backend(…)` describing expected vs. actual.
//!
//! ## Sidecar hash files
//!
//! After a successful download, the computed SHA-256 is persisted to
//! `<dest>.sha256` (just the hex digest, no trailing newline). On cache
//! hits, `BundleRepo::resolve_url` can read the sidecar and compare it
//! against the server's `X-Linked-Etag` in O(1) instead of re-hashing
//! the whole file (which is an I/O + CPU tax on every resolve for
//! multi-GB GGUFs). Missing or mismatched sidecars fall back to a full
//! `sha256_file` pass, which also repairs the sidecar on success.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

use crate::session::CeraError;

/// HTTP timeout for the GET request. Downloads run in a single shot;
/// if a 10-minute window isn't enough for a 10 GB+ shard on a slow
/// connection the caller should split the download itself.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// HTTP timeout for HEAD probes. A slow HEAD means the server is
/// struggling; we'd rather fall back to a best-effort cache reuse
/// than block the caller for minutes.
const HEAD_TIMEOUT: Duration = Duration::from_secs(30);

/// HEAD-probe result.
pub(crate) struct HeadInfo {
    /// Expected total bytes. `None` means the header was absent or the
    /// request failed.
    pub content_length: Option<u64>,
    /// Content-addressed hash from `X-Linked-Etag: sha256:<hex>`. HF
    /// and LFS-compatible CDNs set this; origin servers usually don't.
    pub linked_sha256: Option<String>,
}

/// Sidecar filename for a given cache entry. `<dest>.sha256` holds the
/// bare hex digest (64 chars, no newline). See module docs for the
/// rationale: it turns a multi-GB rehash into a single small read on
/// every cache hit.
pub(crate) fn sidecar_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".sha256");
    PathBuf::from(s)
}

/// Read a previously-persisted sidecar hex digest, if any. Returns
/// `None` on missing file, I/O error, or invalid content; callers
/// treat `None` as "full rehash required."
pub(crate) fn read_sidecar(dest: &Path) -> Option<String> {
    let text = fs::read_to_string(sidecar_path(dest)).ok()?;
    let hex = text.trim();
    if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(hex.to_ascii_lowercase())
    } else {
        None
    }
}

/// Persist `sha256_hex` alongside `dest` as a sidecar file. Best-effort:
/// a failure to write just means the next cache hit pays a rehash.
pub(crate) fn write_sidecar(dest: &Path, sha256_hex: &str) {
    let _ = fs::write(sidecar_path(dest), sha256_hex);
}

/// Issue a `HEAD` for `url` using `client` and extract size +
/// linked-etag. Swallows network errors into `None` fields: the caller
/// decides how strict to be.
///
/// **The `client` passed here MUST be configured with redirects
/// disabled.** HuggingFace serves `X-Linked-Etag` only on the first
/// response (the 302 redirect to the CDN); a redirect-following client
/// surfaces the CDN's unrelated `ETag` instead. `BundleRepo::new`
/// constructs a dedicated no-redirect client for this reason. On a
/// 3xx response from the non-following client we still read the
/// headers (that's the whole point); on any non-success, non-3xx
/// response we conservatively return `None` fields.
pub(crate) fn head_info(client: &Client, url: &str) -> HeadInfo {
    let none = HeadInfo {
        content_length: None,
        linked_sha256: None,
    };
    let token = get_hf_auth_token_for_url(url);
    let mut req = client.head(url).timeout(HEAD_TIMEOUT);
    if let Some(t) = &token {
        req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let Ok(resp) = req.send().and_then(|r| r.error_for_status()) else {
        return none;
    };
    // HF sets `x-linked-size` on the first hop; prefer that over
    // `Content-Length` because `Content-Length` on a 302 reflects the
    // redirect body (zero or a tiny HTML stub), not the file size.
    let headers = resp.headers();
    let content_length = headers
        .get("x-linked-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| {
            headers
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
        });
    let linked_sha256 = extract_linked_sha256(headers);
    HeadInfo {
        content_length,
        linked_sha256,
    }
}

/// Pull `X-Linked-Etag: "sha256:<hex>"` from a header map. Returns
/// `None` if the header is missing or not an `sha256:` scheme.
fn extract_linked_sha256(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"'))
        .and_then(|s| s.strip_prefix("sha256:"))
        .map(|h| h.to_ascii_lowercase())
}

/// SHA-256 a file that's already on disk. Used to verify a cached
/// entry when callers want stronger than size-only assurance and no
/// sidecar is available.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hex_encode(&hasher.finalize()))
}

/// Stream `url` into `dest` atomically + verify the content hash +
/// persist a sidecar hash file.
///
/// Writes to `<dest>.partial` first and renames on success. Interrupted
/// downloads can resume from existing bytes in `<dest>.partial`.
/// On integrity failure the partial is deleted.
/// `expected_sha256` overrides any server-provided `X-Linked-Etag`.
///
/// `progress`, when `Some`, is called periodically during the byte
/// stream: at most once per ~256 KB written, plus one final
/// callback at end-of-stream with the total bytes written (skipped
/// when the in-loop callback already reported the same value, which
/// happens for files whose size is an exact multiple of 256 KB).
/// `None` makes downloads silent. The writer is still wrapped in
/// the `ProgressingWriter` either way (one Option-check + a `u64`
/// add per `Write::write` call, well below noise floor of disk +
/// network I/O), but no callback dispatch happens when `None`.
///
/// `total_bytes_hint` lets the caller plumb a known total (e.g. from
/// a HEAD probe's `x-linked-size`) to the progress callback even
/// when the GET response omits `Content-Length` (some chunked-transfer
/// CDNs do). Falls back to `resp.content_length()` when `None`.
fn get_hf_auth_token_for_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    let after_scheme = lower
        .strip_prefix("https://")
        .or_else(|| lower.strip_prefix("http://"))?;
    let host = after_scheme
        .split(['/', ':', '?', '#'])
        .next()
        .unwrap_or("");

    if crate::bundle::hf::is_hf_or_endpoint_host(host) {
        crate::bundle::hf::get_hf_auth_token()
    } else {
        None
    }
}

/// Destination path for in-progress downloads: `<dest>.partial`.
/// A stable path per destination enables resuming interrupted downloads
/// via HTTP Range requests.
pub(crate) fn partial_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".partial");
    PathBuf::from(s)
}

static ACTIVE_DOWNLOADS: LazyLock<Mutex<HashSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn lock_active_downloads() -> std::sync::MutexGuard<'static, HashSet<PathBuf>> {
    ACTIVE_DOWNLOADS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// RAII guard ensuring at most one active download to a given destination in this process.
struct ActiveDownloadGuard {
    path: PathBuf,
}

impl ActiveDownloadGuard {
    fn acquire(path: &Path) -> Result<Self, CeraError> {
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        };
        let canon = if abs.exists() {
            abs.canonicalize().unwrap_or(abs)
        } else if let Some(parent) = abs.parent().filter(|p| p.exists()) {
            parent
                .canonicalize()
                .map(|p| p.join(abs.file_name().unwrap_or_default()))
                .unwrap_or(abs)
        } else {
            abs
        };
        let mut active = lock_active_downloads();
        if !active.insert(canon.clone()) {
            return Err(CeraError::Backend(format!(
                "download for {} is already in progress in this process",
                path.display()
            )));
        }
        Ok(Self { path: canon })
    }
}

impl Drop for ActiveDownloadGuard {
    fn drop(&mut self) {
        let mut active = lock_active_downloads();
        active.remove(&self.path);
    }
}

/// Parse `Content-Range: bytes <start>-<end>/<total>`.
/// Returns `Some((start, end, total))` where `total` is `None` for wildcard `*`.
pub(crate) fn parse_content_range(
    headers: &reqwest::header::HeaderMap,
) -> Option<(u64, u64, Option<u64>)> {
    let header_val = headers.get(reqwest::header::CONTENT_RANGE)?.to_str().ok()?;
    let trimmed = header_val.trim();
    let (unit, rest) = trimmed.split_once(char::is_whitespace)?;
    if !unit.eq_ignore_ascii_case("bytes") {
        return None;
    }
    let rest = rest.trim_start();
    let (range_part, total_part) = rest.split_once('/')?;
    let (start_str, end_str) = range_part.split_once('-')?;
    let start = start_str.trim().parse::<u64>().ok()?;
    let end = end_str.trim().parse::<u64>().ok()?;
    if start > end {
        return None;
    }
    let total = if total_part.trim() == "*" {
        None
    } else {
        let tot = total_part.trim().parse::<u64>().ok()?;
        if end >= tot {
            return None;
        }
        Some(tot)
    };
    Some((start, end, total))
}

pub(crate) fn download_to(
    client: &Client,
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    total_bytes_hint: Option<u64>,
    progress: Option<&dyn crate::bundle::DownloadProgress>,
) -> Result<(), CeraError> {
    const MAX_DOWNLOAD_RETRIES: u32 = 5;
    const BASE_RETRY_DELAY_MS: u64 = 1000;

    let _download_guard = ActiveDownloadGuard::acquire(dest)?;
    let partial = partial_path(dest);
    let token = get_hf_auth_token_for_url(url);

    let mut response_and_resuming = None;
    let mut existing_len = 0u64;
    let mut last_err = String::new();

    for attempt in 0..MAX_DOWNLOAD_RETRIES {
        if attempt > 0 {
            let delay_ms = BASE_RETRY_DELAY_MS * (1 << (attempt - 1));
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }

        // Check if a partial download file already exists on disk.
        existing_len = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

        let mut req = client.get(url).timeout(DOWNLOAD_TIMEOUT);
        if let Some(t) = &token {
            req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {t}"));
        }
        if existing_len > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={existing_len}-"));
        }

        match req.send() {
            Ok(r) => {
                let status = r.status();
                if status == reqwest::StatusCode::PARTIAL_CONTENT {
                    // 206 Partial Content: verify server range matches existing prefix offset.
                    if let Some((start, _end, _total)) = parse_content_range(r.headers()) {
                        if start == existing_len {
                            response_and_resuming = Some((r, true));
                            break;
                        } else {
                            let _ = fs::remove_file(&partial);
                            last_err = format!(
                                "HTTP 206 Content-Range start {start} != expected offset {existing_len}"
                            );
                            existing_len = 0;
                            continue;
                        }
                    } else {
                        let _ = fs::remove_file(&partial);
                        last_err = "HTTP 206 without valid Content-Range header".to_string();
                        existing_len = 0;
                        continue;
                    }
                } else if status.is_success() {
                    // 200 OK: server sent the full file from byte 0 (Range ignored or not requested).
                    response_and_resuming = Some((r, false));
                    break;
                } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                    // 416 Range Not Satisfiable: partial file reached/exceeded remote size or remote changed.
                    // If the existing partial file already matches expected sha256 (e.g. download finished
                    // but process exited before rename), finalize it immediately instead of re-downloading.
                    let expected_hash = expected_sha256
                        .map(|s| s.to_ascii_lowercase())
                        .or_else(|| extract_linked_sha256(r.headers()));
                    if existing_len > 0
                        && let Some(exp) = expected_hash.as_deref()
                        && let Ok(actual_hex) = sha256_file(&partial)
                        && actual_hex == exp
                    {
                        if let Some(p) = progress {
                            p.on_progress(url, existing_len, Some(existing_len));
                        }
                        let _ = fs::remove_file(dest);
                        if let Err(e) = fs::rename(&partial, dest) {
                            let _ = fs::remove_file(&partial);
                            return Err(e.into());
                        }
                        write_sidecar(dest, exp);
                        return Ok(());
                    }

                    // Reset partial file and retry from byte 0.
                    let _ = fs::remove_file(&partial);
                    existing_len = 0;
                    last_err = "HTTP 416 Range Not Satisfiable".to_string();
                    continue;
                } else if status == reqwest::StatusCode::REQUEST_TIMEOUT
                    || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error()
                {
                    last_err = format!("HTTP {status}");
                } else {
                    return Err(CeraError::Backend(format!("GET {url}: HTTP {status}")));
                }
            }
            Err(e) => {
                last_err = format!("{e}");
            }
        }
    }

    let (mut resp, is_resuming) = match response_and_resuming {
        Some(pair) => pair,
        None => {
            return Err(CeraError::Backend(format!(
                "GET {url} failed after {MAX_DOWNLOAD_RETRIES} attempts: {last_err}"
            )));
        }
    };

    // Calculate total expected bytes across existing prefix and incoming response.
    let total_bytes = if is_resuming {
        parse_content_range(resp.headers())
            .and_then(|(_, _, total)| total)
            .or_else(|| {
                resp.content_length()
                    .map(|rem| existing_len.saturating_add(rem))
            })
            .or(total_bytes_hint)
    } else {
        total_bytes_hint.or_else(|| resp.content_length())
    };

    // Server-side hash fallback when the caller did not pin one.
    let server_hash = extract_linked_sha256(resp.headers());
    let expected = expected_sha256
        .map(|s| s.to_ascii_lowercase())
        .or(server_hash);

    let copy_result: Result<(String, u64), CeraError> = {
        let (mut file, hasher, initial_bytes) = if is_resuming && existing_len > 0 {
            // Hash the existing on-disk prefix so running SHA-256 covers the whole file.
            let prefix_file = fs::File::open(&partial)?;
            let mut reader = io::BufReader::with_capacity(128 * 1024, prefix_file);
            let mut hasher = Sha256::new();
            let copied = io::copy(&mut reader, &mut hasher)?;
            if copied != existing_len {
                drop(reader);
                let _ = fs::remove_file(&partial);
                return Err(CeraError::Backend(format!(
                    "partial file size changed during resume for {url}"
                )));
            }
            drop(reader);
            let file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&partial)?;
            (file, hasher, existing_len)
        } else {
            let _ = fs::remove_file(&partial);
            let file = fs::File::create(&partial)?;
            let hasher = Sha256::new();
            (file, hasher, 0u64)
        };

        let mut hashing = HashingWriter {
            inner: &mut file,
            hasher,
        };

        let (final_bytes, last_in_loop_callback_at) = {
            let mut counting =
                ProgressingWriter::new(&mut hashing, progress, url, total_bytes, initial_bytes);
            io::copy(&mut resp, &mut counting)
                .map_err(|e| CeraError::Backend(format!("write {}: {e}", partial.display())))?;
            (counting.bytes_written, counting.last_callback_at)
        };

        if let Some(p) = progress
            && final_bytes != last_in_loop_callback_at
        {
            p.on_progress(url, final_bytes, total_bytes);
        }
        let digest = hashing.hasher.finalize();
        file.sync_all()?;
        Ok((hex_encode(&digest), final_bytes))
    };

    let (actual_hex, bytes_received) = match copy_result {
        Ok((h, b)) => (h, b),
        Err(e) => {
            // Do not delete partial file on mid-stream transport errors so future retries can resume.
            return Err(e);
        }
    };

    if let Some(exp_total) = total_bytes
        && bytes_received < exp_total
    {
        return Err(CeraError::Backend(format!(
            "download incomplete for {url}: expected {exp_total} bytes, received {bytes_received}"
        )));
    }

    if let Some(exp) = expected.as_deref()
        && exp != actual_hex
    {
        // Delete corrupt partial file on cryptographic integrity failure.
        let _ = fs::remove_file(&partial);
        return Err(CeraError::Backend(format!(
            "integrity check failed for {url}: expected sha256:{exp}, got sha256:{actual_hex}"
        )));
    }

    let _ = fs::remove_file(dest);
    if let Err(e) = fs::rename(&partial, dest) {
        let _ = fs::remove_file(&partial);
        return Err(e.into());
    }
    write_sidecar(dest, &actual_hex);
    Ok(())
}

/// `io::Write` adapter that fans bytes into both the wrapped writer
/// and a running SHA-256 digest. Avoids a second pass over the file
/// after writing.
struct HashingWriter<'a, W: io::Write> {
    inner: &'a mut W,
    hasher: Sha256,
}

impl<W: io::Write> io::Write for HashingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Counts bytes written + (optionally) reports them to a
/// `DownloadProgress` callback. Wraps another `io::Write` so the
/// hashing + counting + (maybe) callback layers stack cleanly:
/// `io::copy(resp, ProgressingWriter -> HashingWriter -> File)`.
///
/// Throttled at `PROGRESS_THROTTLE_BYTES` granularity to avoid
/// hammering a UI's main thread on multi-MB downloads; callers
/// targeting a progress bar typically can't repaint faster than
/// ~30 Hz anyway, and a 256 KB step at 10 MB/s is ~25 callbacks
/// per second, comfortably matched.
struct ProgressingWriter<'a, W: io::Write> {
    inner: &'a mut W,
    progress: Option<&'a dyn crate::bundle::DownloadProgress>,
    url: &'a str,
    total_bytes: Option<u64>,
    bytes_written: u64,
    last_callback_at: u64,
}

const PROGRESS_THROTTLE_BYTES: u64 = 256 * 1024;

impl<'a, W: io::Write> ProgressingWriter<'a, W> {
    fn new(
        inner: &'a mut W,
        progress: Option<&'a dyn crate::bundle::DownloadProgress>,
        url: &'a str,
        total_bytes: Option<u64>,
        initial_bytes: u64,
    ) -> Self {
        if let Some(p) = progress
            && initial_bytes > 0
        {
            p.on_progress(url, initial_bytes, total_bytes);
        }
        Self {
            inner,
            progress,
            url,
            total_bytes,
            bytes_written: initial_bytes,
            last_callback_at: initial_bytes,
        }
    }
}

impl<W: io::Write> io::Write for ProgressingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes_written += n as u64;
        if let Some(p) = self.progress
            && self.bytes_written - self.last_callback_at >= PROGRESS_THROTTLE_BYTES
        {
            p.on_progress(self.url, self.bytes_written, self.total_bytes);
            self.last_callback_at = self.bytes_written;
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_appends_extension() {
        assert_eq!(
            sidecar_path(Path::new("/cache/x.gguf")),
            PathBuf::from("/cache/x.gguf.sha256")
        );
    }

    /// `ProgressingWriter` throttles its callback to ~PROGRESS_THROTTLE_BYTES
    /// granularity. Driving it with three writes (small / large /
    /// small) verifies:
    /// - Writes that don't cross the threshold don't trigger a
    ///   callback (first 1 KB → no event).
    /// - A single write that crosses the threshold triggers exactly
    ///   one event with the cumulative byte count (300 KB → event at
    ///   301 KB total).
    /// - Subsequent small writes don't re-trigger until the next
    ///   threshold crossing (final 1 KB → no event).
    /// - The final emitted byte count matches the high-water mark.
    #[test]
    fn progressing_writer_throttles_callback() {
        use crate::bundle::DownloadProgress;
        use std::io::Write as _;
        use std::sync::Mutex;

        #[derive(Debug, Default)]
        struct Recorder {
            calls: Mutex<Vec<(String, u64, Option<u64>)>>,
        }
        impl DownloadProgress for Recorder {
            fn on_progress(&self, url: &str, bytes: u64, total: Option<u64>) {
                self.calls
                    .lock()
                    .unwrap()
                    .push((url.to_string(), bytes, total));
            }
        }

        let recorder = Recorder::default();
        let mut sink = std::io::sink();
        let mut writer = ProgressingWriter::new(
            &mut sink,
            Some(&recorder as &dyn DownloadProgress),
            "https://example.com/foo.gguf",
            Some(1024 * 1024),
            0,
        );

        // Below threshold: no callback.
        writer.write_all(&[0u8; 1024]).unwrap();
        assert_eq!(recorder.calls.lock().unwrap().len(), 0);

        // Crossing threshold (256 KB) inside one write: one callback,
        // bytes = 1 KB + 300 KB = 308224 = 301 KiB.
        writer.write_all(&[0u8; 300 * 1024]).unwrap();
        let calls_after_big = recorder.calls.lock().unwrap().clone();
        assert_eq!(calls_after_big.len(), 1);
        assert_eq!(calls_after_big[0].0, "https://example.com/foo.gguf");
        assert_eq!(calls_after_big[0].1, (1 + 300) * 1024);
        assert_eq!(calls_after_big[0].2, Some(1024 * 1024));

        // Below the next threshold: no new callback.
        writer.write_all(&[0u8; 1024]).unwrap();
        assert_eq!(recorder.calls.lock().unwrap().len(), 1);

        // bytes_written reflects everything written.
        assert_eq!(writer.bytes_written, (1 + 300 + 1) * 1024);
    }

    /// Verifies the end-of-stream de-dup logic in `download_to`:
    /// after the throttled in-loop callback fires, `last_callback_at`
    /// equals `bytes_written`. The download_to caller checks
    /// `final_bytes != last_in_loop_callback_at` before firing the
    /// end-of-stream callback. Without this guard, a file whose size
    /// is an exact multiple of `PROGRESS_THROTTLE_BYTES` would emit
    /// two callbacks at the same byte count.
    #[test]
    fn progressing_writer_last_callback_at_equals_bytes_at_threshold_boundary() {
        use crate::bundle::DownloadProgress;
        use std::io::Write as _;
        use std::sync::Mutex;

        #[derive(Debug, Default)]
        struct Recorder {
            calls: Mutex<Vec<u64>>,
        }
        impl DownloadProgress for Recorder {
            fn on_progress(&self, _: &str, b: u64, _: Option<u64>) {
                self.calls.lock().unwrap().push(b);
            }
        }

        let recorder = Recorder::default();
        let mut sink = std::io::sink();
        let mut writer = ProgressingWriter::new(
            &mut sink,
            Some(&recorder as &dyn DownloadProgress),
            "https://example.com/exact.bin",
            Some(PROGRESS_THROTTLE_BYTES),
            0,
        );
        // Single write that exactly hits the throttle threshold.
        writer
            .write_all(&vec![0u8; PROGRESS_THROTTLE_BYTES as usize])
            .unwrap();
        // In-loop callback fired; bytes_written == last_callback_at.
        assert_eq!(writer.bytes_written, PROGRESS_THROTTLE_BYTES);
        assert_eq!(writer.last_callback_at, PROGRESS_THROTTLE_BYTES);
        let calls = recorder.calls.lock().unwrap();
        assert_eq!(
            calls.len(),
            1,
            "exactly one in-loop callback at the boundary"
        );
        assert_eq!(calls[0], PROGRESS_THROTTLE_BYTES);
    }

    /// `progress = None` → ProgressingWriter is a thin forwarder, no
    /// allocation, no calls. Sanity-checks the `if let Some(p)` branch
    /// stays cold so the no-progress path doesn't pay for it.
    #[test]
    fn progressing_writer_with_none_progress_is_silent() {
        use std::io::Write as _;
        let mut sink = std::io::sink();
        let mut writer = ProgressingWriter::new(&mut sink, None, "https://example.com/x", None, 0);
        writer.write_all(&[0u8; 1024 * 1024]).unwrap();
        assert_eq!(writer.bytes_written, 1024 * 1024);
    }

    #[test]
    fn progressing_writer_reports_initial_resumed_position() {
        use crate::bundle::DownloadProgress;
        use std::sync::Mutex;

        #[derive(Debug, Default)]
        struct Recorder {
            calls: Mutex<Vec<(String, u64, Option<u64>)>>,
        }
        impl DownloadProgress for Recorder {
            fn on_progress(&self, url: &str, bytes: u64, total: Option<u64>) {
                self.calls
                    .lock()
                    .unwrap()
                    .push((url.to_string(), bytes, total));
            }
        }

        let recorder = Recorder::default();
        let mut sink = std::io::sink();
        let _writer = ProgressingWriter::new(
            &mut sink,
            Some(&recorder as &dyn DownloadProgress),
            "https://example.com/model.gguf",
            Some(1000),
            750,
        );

        let calls = recorder.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "https://example.com/model.gguf");
        assert_eq!(calls[0].1, 750);
        assert_eq!(calls[0].2, Some(1000));
    }

    #[test]
    fn partial_path_appends_extension() {
        assert_eq!(
            partial_path(Path::new("/cache/x.gguf")),
            PathBuf::from("/cache/x.gguf.partial")
        );
    }

    #[test]
    fn parse_content_range_parses_standard_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes 500-999/5000".parse().unwrap(),
        );
        assert_eq!(parse_content_range(&headers), Some((500, 999, Some(5000))));

        headers.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes 0-100/1000".parse().unwrap(),
        );
        assert_eq!(parse_content_range(&headers), Some((0, 100, Some(1000))));

        headers.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes 500-999/*".parse().unwrap(),
        );
        assert_eq!(parse_content_range(&headers), Some((500, 999, None)));
    }

    #[test]
    fn read_sidecar_accepts_valid_hex() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let expected = "a".repeat(64);
        fs::write(sidecar_path(&dest), &expected).unwrap();
        assert_eq!(read_sidecar(&dest).as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn read_sidecar_normalizes_case_and_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let hex = "AbCdEf".repeat(8) + "AbCdEfAbCdEfAbCd";
        assert_eq!(hex.len(), 64);
        fs::write(sidecar_path(&dest), format!("  {hex}  \n")).unwrap();
        assert_eq!(read_sidecar(&dest), Some(hex.to_ascii_lowercase()));
    }

    #[test]
    fn read_sidecar_rejects_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        fs::write(sidecar_path(&dest), "deadbeef").unwrap();
        assert!(read_sidecar(&dest).is_none());
    }

    #[test]
    fn read_sidecar_rejects_non_hex() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let garbage = "z".repeat(64);
        fs::write(sidecar_path(&dest), garbage).unwrap();
        assert!(read_sidecar(&dest).is_none());
    }

    #[test]
    fn read_sidecar_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("nonexistent.gguf");
        assert!(read_sidecar(&dest).is_none());
    }

    #[test]
    fn sha256_file_round_trips_known_input() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.bin");
        fs::write(&p, b"hello").unwrap();
        // Known SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert_eq!(
            sha256_file(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_download_to_resumes_partial_file() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let full_content = b"hello beautiful world of cera!";
        let mut hasher = Sha256::new();
        hasher.update(full_content);
        let expected_sha256 = hex_encode(&hasher.finalize());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut req_buf = vec![0u8; 1024];
            let n = stream.read(&mut req_buf).unwrap();
            let req_str = String::from_utf8_lossy(&req_buf[..n]);

            if req_str.to_ascii_lowercase().contains("range: bytes=5-") {
                let body = &full_content[5..];
                let response = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes 5-{}/{}\r\nConnection: close\r\n\r\n",
                    body.len(),
                    full_content.len() - 1,
                    full_content.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(body).unwrap();
            } else {
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    full_content.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(full_content).unwrap();
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        let partial = partial_path(&dest);

        // Pre-create partial file with first 5 bytes ("hello")
        fs::write(&partial, &full_content[..5]).unwrap();

        let client = Client::new();
        let url = format!("http://127.0.0.1:{}/model.gguf", addr.port());

        download_to(&client, &url, &dest, Some(&expected_sha256), None, None).unwrap();

        server_thread.join().unwrap();

        // Verify partial file was removed on completion and dest exists with full contents
        assert!(!partial.exists());
        assert!(dest.exists());
        let downloaded = fs::read(&dest).unwrap();
        assert_eq!(downloaded, full_content);
        assert_eq!(
            read_sidecar(&dest).as_deref(),
            Some(expected_sha256.as_str())
        );
    }

    #[test]
    fn test_download_to_handles_206_offset_mismatch_by_resetting() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let full_content = b"hello beautiful world of cera!";
        let mut hasher = Sha256::new();
        hasher.update(full_content);
        let expected_sha256 = hex_encode(&hasher.finalize());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            // First attempt: client sends Range: bytes=5-, server returns 206 with mismatched start=10
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req_buf = vec![0u8; 1024];
                let n = stream.read(&mut req_buf).unwrap();
                let req_str = String::from_utf8_lossy(&req_buf[..n]);
                let req_lower = req_str.to_ascii_lowercase();
                assert!(req_lower.contains("range: bytes=5-"));

                let response = "HTTP/1.1 206 Partial Content\r\nContent-Length: 10\r\nContent-Range: bytes 10-19/31\r\nConnection: close\r\n\r\n0123456789";
                stream.write_all(response.as_bytes()).unwrap();
            }

            // Second attempt: client has reset partial and requests from byte 0
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req_buf = vec![0u8; 1024];
                let n = stream.read(&mut req_buf).unwrap();
                let req_str = String::from_utf8_lossy(&req_buf[..n]);
                let req_lower = req_str.to_ascii_lowercase();
                assert!(!req_lower.contains("range:"));

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    full_content.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(full_content).unwrap();
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("mismatch_model.gguf");
        let partial = partial_path(&dest);

        // Pre-create partial file with 5 bytes
        fs::write(&partial, &full_content[..5]).unwrap();

        let client = Client::new();
        let url = format!("http://127.0.0.1:{}/model.gguf", addr.port());

        download_to(&client, &url, &dest, Some(&expected_sha256), None, None).unwrap();

        server_thread.join().unwrap();

        assert!(!partial.exists());
        assert!(dest.exists());
        let downloaded = fs::read(&dest).unwrap();
        assert_eq!(downloaded, full_content);
    }

    #[test]
    fn active_download_guard_mutual_exclusion() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model.bin");
        let guard1 = ActiveDownloadGuard::acquire(&target).unwrap();
        let guard2 = ActiveDownloadGuard::acquire(&target);
        assert!(guard2.is_err());
        drop(guard1);
        let guard3 = ActiveDownloadGuard::acquire(&target);
        assert!(guard3.is_ok());
    }

    #[test]
    fn test_parse_content_range_rfc_and_validation() {
        use reqwest::header::{CONTENT_RANGE, HeaderMap, HeaderValue};

        let mut headers = HeaderMap::new();

        // Standard bytes range
        headers.insert(CONTENT_RANGE, HeaderValue::from_static("bytes 0-499/1000"));
        assert_eq!(parse_content_range(&headers), Some((0, 499, Some(1000))));

        // Case-insensitive unit
        headers.insert(
            CONTENT_RANGE,
            HeaderValue::from_static("Bytes 500-999/1000"),
        );
        assert_eq!(parse_content_range(&headers), Some((500, 999, Some(1000))));

        // Wildcard total
        headers.insert(CONTENT_RANGE, HeaderValue::from_static("bytes 100-200/*"));
        assert_eq!(parse_content_range(&headers), Some((100, 200, None)));

        // Invalid: start > end
        headers.insert(
            CONTENT_RANGE,
            HeaderValue::from_static("bytes 500-100/1000"),
        );
        assert_eq!(parse_content_range(&headers), None);

        // Invalid: end >= total
        headers.insert(CONTENT_RANGE, HeaderValue::from_static("bytes 0-1000/1000"));
        assert_eq!(parse_content_range(&headers), None);

        // Invalid unit
        headers.insert(CONTENT_RANGE, HeaderValue::from_static("items 0-499/1000"));
        assert_eq!(parse_content_range(&headers), None);
    }

    #[test]
    fn test_download_to_handles_416_with_valid_completed_partial() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let full_content = b"hello beautiful world of cera!";
        let mut hasher = Sha256::new();
        hasher.update(full_content);
        let expected_sha256 = hex_encode(&hasher.finalize());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut req_buf = vec![0u8; 1024];
            let n = stream.read(&mut req_buf).unwrap();
            let req_str = String::from_utf8_lossy(&req_buf[..n]);
            let expected_range = format!("range: bytes={}-", full_content.len());
            assert!(req_str.to_ascii_lowercase().contains(&expected_range));

            let response = format!(
                "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{}\r\nConnection: close\r\n\r\n",
                full_content.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("full_partial_model.gguf");
        let partial = partial_path(&dest);

        // Pre-create partial file with full bytes
        fs::write(&partial, full_content).unwrap();

        let client = Client::new();
        let url = format!("http://127.0.0.1:{}/model.gguf", addr.port());

        download_to(&client, &url, &dest, Some(&expected_sha256), None, None).unwrap();

        server_thread.join().unwrap();

        assert!(!partial.exists());
        assert!(dest.exists());
        let downloaded = fs::read(&dest).unwrap();
        assert_eq!(downloaded, full_content);
        assert_eq!(
            read_sidecar(&dest).as_deref(),
            Some(expected_sha256.as_str())
        );
    }

    #[test]
    fn test_download_to_handles_416_with_corrupt_partial_by_resetting() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let full_content = b"hello beautiful world of cera!";
        let mut hasher = Sha256::new();
        hasher.update(full_content);
        let expected_sha256 = hex_encode(&hasher.finalize());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            // First attempt: client sends range for 31 bytes, server returns 416
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req_buf = vec![0u8; 1024];
                let n = stream.read(&mut req_buf).unwrap();
                let req_str = String::from_utf8_lossy(&req_buf[..n]);
                assert!(req_str.to_ascii_lowercase().contains("range: bytes=31-"));

                let response = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */31\r\nConnection: close\r\n\r\n";
                stream.write_all(response.as_bytes()).unwrap();
            }

            // Second attempt: partial was deleted because hash mismatched, requests from byte 0
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req_buf = vec![0u8; 1024];
                let n = stream.read(&mut req_buf).unwrap();
                let req_str = String::from_utf8_lossy(&req_buf[..n]);
                assert!(!req_str.to_ascii_lowercase().contains("range:"));

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    full_content.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(full_content).unwrap();
            }
        });

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("corrupt_partial_model.gguf");
        let partial = partial_path(&dest);

        // Pre-create corrupt partial file with 31 bogus bytes
        fs::write(&partial, b"corrupted partial content 12345").unwrap();

        let client = Client::new();
        let url = format!("http://127.0.0.1:{}/model.gguf", addr.port());

        download_to(&client, &url, &dest, Some(&expected_sha256), None, None).unwrap();

        server_thread.join().unwrap();

        assert!(!partial.exists());
        assert!(dest.exists());
        let downloaded = fs::read(&dest).unwrap();
        assert_eq!(downloaded, full_content);
    }
}

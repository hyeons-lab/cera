//! Browser bundle store, backed by the Origin Private File System.
//!
//! This is the web counterpart to `cera::bundle::BundleRepo`. It answers
//! the same question ("give me the bytes for this URL, downloading and
//! caching if needed") against the only browser storage API that gives
//! real file semantics.
//!
//! ## Why OPFS and not the alternatives
//!
//! `wasm32-unknown-unknown` has no filesystem at all: `std::fs` compiles
//! but every call returns `Unsupported`, because there are no syscalls
//! behind it. (WASI has a real one, but that's a different target and
//! not what wasm-bindgen builds.) So persistence has to come from a JS
//! API, and there are four:
//!
//! - **OPFS** (`navigator.storage.getDirectory()`), an origin-private
//!   directory tree with `getFileHandle` / `getDirectoryHandle` /
//!   `removeEntry`. Disk-backed, and the only one with directories, so
//!   the native store's `<host>/<url-path>` layout carries over exactly.
//!   In a worker, `createSyncAccessHandle()` also allows reading
//!   straight into wasm linear memory with no intermediate JS buffer.
//!   Chrome 86+, Safari 15.2+, Firefox 111+.
//! - **Cache Storage**: streams a `Response` to disk without buffering
//!   in the JS heap, which is genuinely nice for a multi-GB download,
//!   but `cache.put()` reports no progress, entries are flat and keyed
//!   by `Request`, and reads hand back a whole `Blob`.
//! - **IndexedDB**: universal and disk-backed, but flat key/value over
//!   `Blob`s: the directory layout, the size walk and the sidecar all
//!   become bookkeeping we'd have to invent.
//! - **`localStorage`**: strings only, ~5 MB. Not a candidate.
//!
//! OPFS wins on fit. We do not fall back to IndexedDB: every browser
//! that can run this engine at a useful speed has OPFS, and a silent
//! fallback with different eviction and different performance is worse
//! than a clear error. Under Node, where OPFS does not exist, there is
//! a real filesystem and the right answer is `node:fs` plus
//! `CeraEngine.fromGgufParts`; that's what the error says.
//!
//! ## Two things a browser cannot match
//!
//! **Integrity.** The native store reads HuggingFace's `X-Linked-Etag`
//! (the content SHA-256) from a redirect-less `HEAD`. A browser cannot:
//! `redirect: "manual"` yields an opaque response with no readable
//! headers, and cross-origin header access is limited to what the server
//! puts in `Access-Control-Expose-Headers`. So the policy here is: use a
//! caller-supplied hash when there is one, opportunistically use
//! `x-linked-etag` when CORS happens to expose it, and otherwise fall
//! back to the `Content-Length` size check (`Content-Length` is
//! CORS-safelisted, so that much always works). A cached entry is
//! verified against its `.sha256` sidecar exactly as on native.
//!
//! **Size.** wasm32 linear memory tops out at 4 GB, and the engine needs
//! the whole GGUF resident, so a browser model is bounded well below
//! what a native one is. That limit is the wasm target's, not this
//! module's; OPFS itself will happily store more than the engine can
//! load. Storage quota is a separate cap (Chrome allows a large
//! fraction of free disk, Safari far less) and eviction is real unless
//! the page calls `navigator.storage.persist()`, which the exported
//! `persistStorage()` requests. (A code span, not an intra-doc link:
//! `#[wasm_bindgen]` rewrites the item, so rustdoc can't resolve the
//! Rust name, and the JS name is what a `.d.ts` reader wants anyway.)

use std::future::Future;
use std::pin::Pin;

use cera::bundle::cache_key::{
    LEAP_BUNDLES_API_URL, cache_relative_segments, leap_bundles_manifest_url, parse_leap_bundles,
    validate_path_segment,
};
use js_sys::{Function, Reflect, Uint8Array};
use sha2::{Digest, Sha256};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AbortSignal, File, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemHandle,
    FileSystemHandleKind, FileSystemReadWriteOptions, FileSystemRemoveOptions,
    FileSystemSyncAccessHandle, FileSystemWritableFileStream, ReadableStreamDefaultReader,
    RequestInit, Response, StorageManager,
};

/// `fetch` off the global scope. Bound directly rather than through
/// `web_sys::Window::fetch_with_str` because this code runs in a
/// dedicated worker at least as often as on the main thread, and
/// `Window` doesn't exist there. Both scopes expose a global `fetch`.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = fetch)]
    fn global_fetch(input: &str) -> js_sys::Promise;

    #[wasm_bindgen(js_name = fetch)]
    fn global_fetch_with_init(input: &str, init: &RequestInit) -> js_sys::Promise;

    /// `AbortSignal.timeout(ms)`. Bound by hand because the pinned
    /// `web-sys` exposes the `AbortSignal` type without this static
    /// constructor.
    ///
    /// Bound infallibly, so a browser without it throws a `TypeError`
    /// out of every fetch here rather than degrading. That is a
    /// deliberate floor, not an oversight: it arrived in Chrome 103 /
    /// Safari 16, later than OPFS itself (Chrome 86 / Safari 15.2), so
    /// the two are not equivalent. It is however comfortably older than
    /// the WebGPU this crate's fast path needs (Chrome 113 / Safari
    /// 18), and a store that silently reverts to unbounded fetches is
    /// the hang this exists to prevent.
    #[wasm_bindgen(js_namespace = AbortSignal, js_name = timeout)]
    fn abort_after(ms: u32) -> AbortSignal;
}

/// Directory name used when the caller doesn't pick one. Everything
/// lives under this single top-level OPFS entry so `clearCache` can
/// remove the tree without touching anything else the origin stored.
const DEFAULT_STORE_DIR: &str = "cera-models";

/// Deadline for the small requests: the `HEAD` probe and the catalog
/// GET in `list_leap_bundles`. Matches the native store's.
///
/// `fetch` has no default timeout, and a stalled connection (captive
/// portal, a proxy that blackholes rather than refuses) leaves its
/// promise *pending* rather than rejecting it. Without a signal the
/// await never returns, so a load hangs with no error and no progress
/// instead of falling back. That is not hypothetical: it is what a
/// stalled HEAD against huggingface.co did here, and it defeats
/// `head_info`'s "never fails" contract, which only holds for a
/// rejected fetch.
const HEAD_TIMEOUT_MS: u32 = 30_000;

/// Deadline for a model download, covering the whole body rather than
/// just the response head. Generous because a multi-GB GGUF on a slow
/// link legitimately takes minutes; it is still a real ceiling, and a
/// transfer slower than roughly 2.5 MB/s sustained will not finish a
/// 1.5 GB bundle inside it. The store has no resume, so such a download
/// restarts. Matches the native store, deliberately: a shared number
/// that is occasionally too small beats two that drift.
const GET_TIMEOUT_MS: u32 = 600_000;

/// Suffix for the persisted per-file hash, matching the native store's
/// `<dest>.sha256` sidecar so the two layouts stay readable as one.
const SIDECAR_SUFFIX: &str = ".sha256";

/// Progress callbacks fire at most once per this many bytes, plus once
/// at end of stream. Matches `cera::bundle::download`'s cadence: a
/// callback that crosses into JS (and, under `cera_dart`, `postMessage`
/// to another thread) per 16 KB network chunk would cost more than the
/// download.
const PROGRESS_INTERVAL_BYTES: u64 = 256 * 1024;

/// Remote bundle store over the Origin Private File System.
///
/// Construct once and reuse: it holds only the store directory name, so
/// copies are cheap and concurrent downloads through the same instance
/// are fine (every method takes `&self`).
///
/// ```js
/// const repo = new BundleRepo();                 // "cera-models"
/// const engine = await CeraEngine.fromBundleId(
///     repo, "LFM2-1.2B-GGUF", "Q4_0", 4096,
///     (url, done, total) => console.log(url, done / total),
/// );
/// ```
#[wasm_bindgen]
pub struct BundleRepo {
    store_dir: String,
}

#[wasm_bindgen]
impl BundleRepo {
    /// Create a repo rooted at `storeDir` inside the origin's private
    /// filesystem, defaulting to `"cera-models"`. Nothing is created
    /// until the first download, so constructing this is free and
    /// cannot fail on a storage error.
    ///
    /// `storeDir` is a single directory name, not a path: it goes
    /// through the same allowlist as URL-derived cache segments, so a
    /// name containing `/` or `..` is rejected here rather than
    /// silently addressing something else.
    #[wasm_bindgen(constructor)]
    pub fn new(store_dir: Option<String>) -> Result<BundleRepo, JsError> {
        let store_dir = store_dir.unwrap_or_else(|| DEFAULT_STORE_DIR.to_string());
        // Reuse the engine's allowlist rather than writing a second
        // one: `getDirectoryHandle` would happily accept `..` as a
        // literal entry name, and a caller passing a path would get a
        // confusing "not found" instead of a real diagnosis.
        validate_path_segment("storeDir", &store_dir).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(BundleRepo { store_dir })
    }

    /// The OPFS directory this repo caches under. Matches what was
    /// passed to the constructor.
    #[wasm_bindgen(getter, js_name = storeDir)]
    pub fn store_dir(&self) -> String {
        self.store_dir.clone()
    }

    /// Total bytes currently cached, summed by walking the tree.
    ///
    /// Returns 0 when nothing has been downloaded yet (the directory
    /// doesn't exist). Like the native `cache_size`, this is a real
    /// O(n) walk rather than a constant-time query, and it counts the
    /// `.sha256` sidecars along with the payloads.
    ///
    /// This is deliberately not `navigator.storage.estimate()`: that
    /// reports the whole origin's usage, so a page that also stores
    /// user data would see it attributed to the model cache.
    #[wasm_bindgen(js_name = cacheSize)]
    pub async fn cache_size(&self) -> Result<f64, JsError> {
        match self.store_root(false).await? {
            Some(dir) => dir_size(dir).await,
            None => Ok(0.0),
        }
    }

    /// Delete everything this repo has cached. Idempotent: clearing an
    /// empty or never-created store succeeds.
    ///
    /// Unlike the native store this doesn't recreate the directory
    /// afterwards, because there's nothing to preserve: OPFS
    /// directories are created on demand by the next download.
    ///
    /// An in-flight download through the same repo will fail when its
    /// file vanishes. As on native, serializing a user-driven "clear
    /// downloads" action against active loads is the caller's job.
    #[wasm_bindgen(js_name = clearCache)]
    pub async fn clear_cache(&self) -> Result<(), JsError> {
        let root = root_directory().await?;
        let opts = FileSystemRemoveOptions::new();
        opts.set_recursive(true);
        match JsFuture::from(root.remove_entry_with_options(&self.store_dir, &opts)).await {
            Ok(_) => Ok(()),
            // Already absent; the lazy-creation invariant means this
            // is the normal state before any download.
            Err(e) if is_not_found(&e) => Ok(()),
            Err(e) => Err(js_err("clearCache", e)),
        }
    }

    /// Whether `url` is present in the cache. Existence only: this does
    /// not verify the hash or size, so a truthy answer means "a
    /// download landed here", not "the bytes are known good".
    #[wasm_bindgen(js_name = isCached)]
    pub async fn is_cached(&self, url: String) -> Result<bool, JsError> {
        Ok(self.file_handle(&url, false).await?.is_some())
    }

    /// Download `url` into the cache if it isn't already there.
    ///
    /// `expectedSha256` pins the content hash; when omitted, integrity
    /// falls back to `x-linked-etag` (if CORS exposes it) and then to a
    /// `Content-Length` size check. See the module docs for why a
    /// browser cannot do better.
    ///
    /// `onProgress(url, bytesDownloaded, totalBytes)` fires at most
    /// once per 256 KB and once at end of stream; `totalBytes` is
    /// `null` when the server sends no length. It is not called at all
    /// on a cache hit, since there is no streaming work to report.
    #[wasm_bindgen(js_name = download)]
    pub async fn download(
        &self,
        url: String,
        expected_sha256: Option<String>,
        on_progress: Option<Function>,
    ) -> Result<(), JsError> {
        self.ensure_cached(&url, expected_sha256.as_deref(), on_progress.as_ref())
            .await
    }

    /// Cached bytes for `url`, downloading first if needed.
    ///
    /// **Copies the whole file into a JS `Uint8Array`**, so peak memory
    /// is roughly twice the file. That's fine for a manifest and wrong
    /// for a model: to load a model, use `CeraEngine.fromBundleId` or
    /// `fromManifestUrl`, which keep the bytes inside wasm and hand
    /// them straight to the engine.
    #[wasm_bindgen(js_name = bytes)]
    pub async fn bytes(
        &self,
        url: String,
        expected_sha256: Option<String>,
        on_progress: Option<Function>,
    ) -> Result<Uint8Array, JsError> {
        let bytes = self
            .read_or_download(&url, expected_sha256.as_deref(), on_progress.as_ref())
            .await?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }

    /// Cached text for `url` (UTF-8), downloading first if needed.
    /// Intended for manifests; throws if the bytes aren't valid UTF-8.
    #[wasm_bindgen(js_name = text)]
    pub async fn text(&self, url: String) -> Result<String, JsError> {
        let bytes = self.read_or_download(&url, None, None).await?;
        String::from_utf8(bytes)
            .map_err(|e| JsError::new(&format!("`{url}` is not valid UTF-8: {e}")))
    }

    /// Drop the cache entry for one URL, leaving the rest intact.
    /// Returns whether anything was removed. Removes the `.sha256`
    /// sidecar alongside the payload, so a later re-download can't
    /// match a stale hash.
    #[wasm_bindgen(js_name = remove)]
    pub async fn remove(&self, url: String) -> Result<bool, JsError> {
        let segments = self.segments_for(&url)?;
        let Some(parent) = self.parent_dir(&segments, false).await? else {
            return Ok(false);
        };
        let Some(name) = segments.last() else {
            return Ok(false);
        };
        let removed = remove_entry(&parent, name).await?;
        // Best-effort: a missing sidecar is the normal case for an
        // entry cached without a known hash.
        remove_entry(&parent, &format!("{name}{SIDECAR_SUFFIX}")).await?;
        Ok(removed)
    }
}

// Internal helpers. Deliberately outside the `#[wasm_bindgen] impl` so
// they can take and return Rust types (`Vec<u8>`, `&str`) instead of
// being forced through the JS boundary.
impl BundleRepo {
    /// Cache-key segments for `url`, shared with the native store so
    /// both address an entry identically.
    fn segments_for(&self, url: &str) -> Result<Vec<String>, JsError> {
        cache_relative_segments(url).map_err(|e| JsError::new(&e.to_string()))
    }

    /// This repo's root directory, or `None` when it doesn't exist and
    /// `create` is false.
    async fn store_root(&self, create: bool) -> Result<Option<FileSystemDirectoryHandle>, JsError> {
        let root = root_directory().await?;
        get_directory(&root, &self.store_dir, create).await
    }

    /// Directory holding the entry for `segments`, i.e. everything but
    /// the final filename.
    async fn parent_dir(
        &self,
        segments: &[String],
        create: bool,
    ) -> Result<Option<FileSystemDirectoryHandle>, JsError> {
        let Some(mut dir) = self.store_root(create).await? else {
            return Ok(None);
        };
        let Some((_, parents)) = segments.split_last() else {
            return Ok(None);
        };
        for segment in parents {
            match get_directory(&dir, segment, create).await? {
                Some(next) => dir = next,
                None => return Ok(None),
            }
        }
        Ok(Some(dir))
    }

    /// File handle for `url`, or `None` when absent and `create` is
    /// false.
    async fn file_handle(
        &self,
        url: &str,
        create: bool,
    ) -> Result<Option<FileSystemFileHandle>, JsError> {
        let segments = self.segments_for(url)?;
        let Some(parent) = self.parent_dir(&segments, create).await? else {
            return Ok(None);
        };
        let Some(name) = segments.last() else {
            return Ok(None);
        };
        get_file(&parent, name, create).await
    }

    /// Ensure `url` is cached and valid, downloading if it is not.
    async fn ensure_cached(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
        on_progress: Option<&Function>,
    ) -> Result<(), JsError> {
        let expected = expected_sha256.map(|s| s.to_ascii_lowercase());

        // One network probe per resolve, same as the native store. It
        // supplies both the cache-validation inputs and, on a miss, the
        // content hash to verify the download against.
        let head = if expected.is_some() {
            // Caller pinned a hash; nothing the server says can improve
            // on that, so skip the round-trip entirely.
            HeadInfo::default()
        } else {
            head_info(url).await
        };
        let expected = expected.or_else(|| head.linked_sha256.clone());

        if self
            .cache_hit_valid(url, expected.as_deref(), &head)
            .await?
        {
            return Ok(());
        }

        self.download_to_cache(url, expected.as_deref(), head.content_length, on_progress)
            .await
    }

    /// Cached bytes for `url`, downloading first if needed.
    async fn read_or_download(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
        on_progress: Option<&Function>,
    ) -> Result<Vec<u8>, JsError> {
        self.ensure_cached(url, expected_sha256, on_progress)
            .await?;
        let Some(handle) = self.file_handle(url, false).await? else {
            return Err(JsError::new(&format!(
                "`{url}` vanished from the cache between download and read; \
                 another tab may have cleared the store"
            )));
        };
        read_file(&handle).await
    }

    /// Whether the cached entry for `url` can be reused.
    ///
    /// Mirrors the native policy, minus what CORS forbids: hash if one
    /// is known (sidecar first, full rehash as repair), else size, else
    /// reuse. The last arm is deliberate: a transient network failure
    /// should not force a multi-GB re-download of a file already on
    /// disk.
    async fn cache_hit_valid(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
        head: &HeadInfo,
    ) -> Result<bool, JsError> {
        let Some(handle) = self.file_handle(url, false).await? else {
            return Ok(false);
        };

        if let Some(expected) = expected_sha256 {
            // Fast path: trust the sidecar. We wrote it after the last
            // successful verification, so it is at least as trustworthy
            // as the cached file, and rehashing a multi-GB GGUF on
            // every load would dominate a cache hit.
            if let Some(cached) = self.read_sidecar(url).await? {
                return Ok(cached == expected);
            }
            // Slow path: no sidecar (cached before one was written, or
            // unreadable). Rehash, and repair the sidecar on success so
            // the next load takes the fast path.
            let actual = sha256_of(&handle).await?;
            if actual == expected {
                self.write_sidecar(url, &actual).await?;
                return Ok(true);
            }
            return Ok(false);
        }

        if let Some(expected_len) = head.content_length {
            return Ok(file_size(&handle).await? == expected_len);
        }

        Ok(true)
    }

    /// Drain the response body into `writer`, hashing and reporting progress
    /// as it goes. Returns the number of bytes written.
    ///
    /// Split out of `download_to_cache` so that every way this can fail (a
    /// rejected read, an abort, a malformed chunk, a failed write) leaves
    /// through one `Err` its caller can clean up after, rather than each error
    /// having to remember to undo the partial file itself.
    async fn stream_body(
        reader: &ReadableStreamDefaultReader,
        writer: &mut Writer,
        hasher: &mut Sha256,
        url: &str,
        total: Option<f64>,
        on_progress: Option<&Function>,
    ) -> Result<u64, JsError> {
        let mut written: u64 = 0;
        let mut reported: u64 = 0;
        loop {
            let next = JsFuture::from(reader.read())
                .await
                .map_err(|e| js_err(&format!("reading `{url}`"), e))?;
            if Reflect::get(&next, &JsValue::from_str("done"))
                .map(|v| v.is_truthy())
                .unwrap_or(true)
            {
                break;
            }
            let chunk: Uint8Array = Reflect::get(&next, &JsValue::from_str("value"))
                .map_err(|e| js_err(&format!("reading `{url}`"), e))?
                .unchecked_into();
            let chunk = chunk.to_vec();
            hasher.update(&chunk);
            writer.write(&chunk).await?;
            written = written.saturating_add(chunk.len() as u64);
            if written - reported >= PROGRESS_INTERVAL_BYTES {
                reported = written;
                report_progress(on_progress, url, written, total);
            }
        }
        Ok(written)
    }

    /// Stream `url` into the cache, hashing as it goes.
    async fn download_to_cache(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
        content_length: Option<f64>,
        on_progress: Option<&Function>,
    ) -> Result<(), JsError> {
        // The deadline covers the response head *and* the body stream:
        // an `AbortSignal` fires against the whole request, so a
        // transfer that stalls mid-body aborts too rather than leaving
        // the reader waiting on a chunk that never arrives.
        let init = RequestInit::new();
        init.set_signal(Some(&abort_after(GET_TIMEOUT_MS)));
        let response = fetch_ok(url, Some(&init)).await?;
        // Prefer the HEAD-probed length: the CDN response usually
        // echoes `Content-Length` too, but a chunked one won't, and a
        // progress bar that loses its total mid-download is worse than
        // one that never had it.
        let total = content_length.or_else(|| header_f64(&response, "content-length"));

        let body = response
            .body()
            .ok_or_else(|| JsError::new(&format!("`{url}` returned a response with no body")))?;
        let reader: ReadableStreamDefaultReader = body.get_reader().unchecked_into();

        // Create the file only now. A failed fetch shouldn't leave an
        // empty entry behind that a later size check would have to
        // reject.
        let handle = self
            .file_handle(url, true)
            .await?
            .ok_or_else(|| JsError::new(&format!("could not create a cache entry for `{url}`")))?;
        let mut writer = Writer::open(&handle).await?;

        let mut hasher = Sha256::new();
        // Every failure inside the stream gets the same cleanup, so the loop
        // runs in a helper and the recovery lives at one call site. It used to
        // be attached only to the write branch, which left the read branch
        // returning through `?` with a truncated file still in the cache and,
        // on the sync writer, its exclusive lock still held. Both matter more
        // now that `GET_TIMEOUT_MS` makes a mid-body abort a routine outcome
        // rather than a network freak: the truncated entry has no `.sha256`
        // sidecar, so a later resolve whose HEAD probe also fails accepts it as
        // a cache hit, and the held lock fails the immediate retry, which on
        // the worker's `auto` path is the CPU load right behind a GPU attempt.
        let streamed =
            Self::stream_body(&reader, &mut writer, &mut hasher, url, total, on_progress).await;
        let written = match streamed {
            Ok(written) => written,
            Err(e) => {
                writer.abandon();
                let _ = self.remove(url.to_string()).await;
                return Err(e);
            }
        };
        // `close` is part of the write, not a release: the `Writer::Stream`
        // variant only commits there, so this is where a multi-GB model meets
        // `QuotaExceededError`. A failure has to drop the entry for the same
        // reason a mid-stream one does, or it stays as a truncated file with no
        // sidecar for a later resolve to accept as a cache hit.
        if let Err(e) = writer.close().await {
            let _ = self.remove(url.to_string()).await;
            return Err(e);
        }
        report_progress(on_progress, url, written, total);

        let actual = hex(&hasher.finalize());
        if let Some(expected) = expected_sha256
            && actual != expected
        {
            let _ = self.remove(url.to_string()).await;
            return Err(JsError::new(&format!(
                "hash mismatch for `{url}`: expected {expected}, got {actual}"
            )));
        }
        // Persist even when unverified: the sidecar records what we
        // actually stored, which is what a later caller-supplied hash
        // gets compared against.
        self.write_sidecar(url, &actual).await
    }

    async fn read_sidecar(&self, url: &str) -> Result<Option<String>, JsError> {
        let Some(handle) = self.file_handle(&sidecar_url(url), false).await? else {
            return Ok(None);
        };
        let bytes = read_file(&handle).await?;
        Ok(String::from_utf8(bytes).ok().map(|s| {
            // Written by us, but a truncated write (quota exhaustion
            // mid-flush) could leave whitespace or a partial digest;
            // both simply fail the comparison rather than matching
            // something wrong.
            s.trim().to_ascii_lowercase()
        }))
    }

    async fn write_sidecar(&self, url: &str, hash: &str) -> Result<(), JsError> {
        let handle = self
            .file_handle(&sidecar_url(url), true)
            .await?
            .ok_or_else(|| JsError::new("could not create the .sha256 sidecar"))?;
        let mut writer = Writer::open(&handle).await?;
        // `abandon` on failure, for the same reason the payload write does it:
        // a `Writer::Sync` dropped without releasing its access handle holds
        // OPFS's exclusive lock for the life of the worker, so the immediate
        // retry (the CPU load right behind a GPU attempt) would fail with
        // `NoModificationAllowedError` on a sidecar that not even `remove` can
        // delete. Reachable here on the same quota-exhaustion path as the
        // payload, since this write is what follows a multi-GB flush.
        if let Err(e) = writer.write(hash.as_bytes()).await {
            writer.abandon();
            return Err(e);
        }
        writer.close().await
    }
}

/// The URL whose cache entry holds `url`'s hash sidecar.
///
/// Appending to the URL rather than to the resolved entry name keeps
/// every path in this module going through `cache_relative_segments`,
/// so the sidecar can't land somewhere the payload wouldn't.
fn sidecar_url(url: &str) -> String {
    // Split off query/fragment first: `cache_relative_segments` strips
    // them, so appending to the raw URL would produce
    // `model.gguf?x=1.sha256`, whose entry name is `model.gguf` again
    // and would overwrite the payload.
    let base = url.split(['?', '#']).next().unwrap_or(url);
    format!("{base}{SIDECAR_SUFFIX}")
}

/// What a HEAD probe managed to learn. Both fields are optional
/// because a browser is often told neither (see the module docs).
#[derive(Default)]
struct HeadInfo {
    content_length: Option<f64>,
    linked_sha256: Option<String>,
}

/// Probe `url` with a HEAD request. Never fails: a probe that can't
/// reach the server yields an empty `HeadInfo`, which downgrades the
/// caller to "reuse whatever is cached" rather than forcing a
/// re-download on a flaky connection.
async fn head_info(url: &str) -> HeadInfo {
    let init = RequestInit::new();
    init.set_method("HEAD");
    init.set_signal(Some(&abort_after(HEAD_TIMEOUT_MS)));
    let Ok(response) = fetch_ok(url, Some(&init)).await else {
        return HeadInfo::default();
    };
    HeadInfo {
        content_length: header_f64(&response, "content-length"),
        // Only readable when the origin lists it in
        // `Access-Control-Expose-Headers`; absent is the normal case.
        linked_sha256: response
            .headers()
            .get("x-linked-etag")
            .ok()
            .flatten()
            .map(|v| v.trim_matches('"').to_ascii_lowercase())
            .filter(|v| v.len() == 64 && v.chars().all(|c| c.is_ascii_hexdigit())),
    }
}

/// `fetch` + status check, returning the `Response` only for 2xx.
async fn fetch_ok(url: &str, init: Option<&RequestInit>) -> Result<Response, JsError> {
    let promise = match init {
        Some(init) => global_fetch_with_init(url, init),
        None => global_fetch(url),
    };
    let response: Response = JsFuture::from(promise)
        .await
        .map_err(|e| js_err(&format!("fetching `{url}`"), e))?
        .unchecked_into();
    if !response.ok() {
        return Err(JsError::new(&format!(
            "fetching `{url}` failed: HTTP {} {}",
            response.status(),
            response.status_text()
        )));
    }
    Ok(response)
}

fn header_f64(response: &Response, name: &str) -> Option<f64> {
    response
        .headers()
        .get(name)
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
}

fn report_progress(on_progress: Option<&Function>, url: &str, done: u64, total: Option<f64>) {
    let Some(f) = on_progress else { return };
    // A throwing callback is the caller's bug, not a download failure;
    // swallowing it here keeps a broken progress bar from destroying a
    // multi-GB transfer.
    let _ = f.call3(
        &JsValue::NULL,
        &JsValue::from_str(url),
        &JsValue::from_f64(done as f64),
        &total.map(JsValue::from_f64).unwrap_or(JsValue::NULL),
    );
}

/// Writes a file through whichever OPFS mechanism this scope allows.
///
/// `createSyncAccessHandle` is worker-only (and the only writer Safari
/// had before 17), while `createWritable` is available on the main
/// thread. Trying the sync handle first therefore takes the faster path
/// in the worker `cera_dart` actually runs in, and still works on a
/// main-thread page.
enum Writer {
    /// Sync handles don't track a cursor across calls, so the offset is
    /// ours to maintain.
    Sync {
        handle: FileSystemSyncAccessHandle,
        offset: f64,
    },
    Stream(FileSystemWritableFileStream),
}

impl Writer {
    async fn open(handle: &FileSystemFileHandle) -> Result<Self, JsError> {
        if has_sync_access(handle)
            && let Ok(sync) = JsFuture::from(handle.create_sync_access_handle()).await
        {
            let sync: FileSystemSyncAccessHandle = sync.unchecked_into();
            // Opening keeps any existing contents; truncate so a
            // re-download of a shorter file can't leave a tail of the
            // previous one.
            sync.truncate_with_u32(0)
                .map_err(|e| js_err("truncating the cache entry", e))?;
            return Ok(Writer::Sync {
                handle: sync,
                offset: 0.0,
            });
        }
        let stream = JsFuture::from(handle.create_writable())
            .await
            .map_err(|e| js_err("opening the cache entry for writing", e))?;
        Ok(Writer::Stream(stream.unchecked_into()))
    }

    async fn write(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        match self {
            Writer::Sync { handle, offset } => {
                let opts = FileSystemReadWriteOptions::new();
                opts.set_at(*offset);
                let n = handle
                    .write_with_u8_array_and_options(chunk, &opts)
                    .map_err(|e| js_err("writing the cache entry", e))?;
                // A short write means the quota ran out mid-file. It is
                // not an exception, so an unchecked caller would happily
                // finish and cache a truncated model.
                if n < chunk.len() as f64 {
                    return Err(JsError::new(&format!(
                        "short write to the cache ({n} of {} bytes); \
                         the origin's storage quota is likely exhausted",
                        chunk.len()
                    )));
                }
                *offset += n;
                Ok(())
            }
            Writer::Stream(stream) => {
                let promise = stream
                    .write_with_u8_array(chunk)
                    .map_err(|e| js_err("writing the cache entry", e))?;
                JsFuture::from(promise)
                    .await
                    .map(|_| ())
                    .map_err(|e| js_err("writing the cache entry", e))
            }
        }
    }

    async fn close(self) -> Result<(), JsError> {
        match self {
            Writer::Sync { handle, .. } => {
                // The handle is closed whether or not the flush succeeded.
                // Returning through `?` here would drop it still holding
                // OPFS's exclusive lock on the file, so the retry right behind
                // this one (the worker's CPU fallback after a GPU attempt)
                // would fail with `NoModificationAllowedError` for the
                // lifetime of the worker, which is the invariant `abandon`
                // exists to keep.
                let flushed = handle
                    .flush()
                    .map_err(|e| js_err("flushing the cache entry", e));
                handle.close();
                flushed
            }
            // A writable stream only commits its contents on `close`,
            // so this is the write, not just a release.
            Writer::Stream(stream) => JsFuture::from(stream.close())
                .await
                .map(|_| ())
                .map_err(|e| js_err("closing the cache entry", e)),
        }
    }

    /// Release the handle without committing. Used on the error path,
    /// where a `Writer::Sync` would otherwise hold an exclusive lock on
    /// the file and block the cleanup that follows.
    fn abandon(self) {
        match self {
            Writer::Sync { handle, .. } => handle.close(),
            // Dropping a writable stream without `close()` discards the
            // pending contents, which is exactly what's wanted.
            Writer::Stream(_) => {}
        }
    }
}

/// Whether this scope can open a sync access handle on `handle`.
///
/// `createSyncAccessHandle` is worker-only: on the main thread the
/// method is simply absent from `FileSystemFileHandle`. web-sys
/// generates its binding without `catch`, so calling it there throws a
/// synchronous `TypeError` that is neither a `Result` nor a rejected
/// promise, and the "try sync, fall back to a writable stream" shape
/// below would take down the page instead of falling back. Probing for
/// the method is what makes the fallback reachable.
fn has_sync_access(handle: &FileSystemFileHandle) -> bool {
    has_method(handle, "createSyncAccessHandle")
}

/// Whether `object` actually has `name` as a callable method.
///
/// Several of the APIs here are exposed on some global scopes and not
/// others (`createSyncAccessHandle` is worker-only, `persist` is
/// Window-only). web-sys types don't model that, so a plain call is a
/// `TypeError` rather than a feature check.
fn has_method(object: &JsValue, name: &str) -> bool {
    Reflect::get(object, &JsValue::from_str(name))
        .map(|f| f.is_function())
        .unwrap_or(false)
}

/// The origin's private root directory.
async fn root_directory() -> Result<FileSystemDirectoryHandle, JsError> {
    let storage = storage_manager()?;
    let root = JsFuture::from(storage.get_directory())
        .await
        .map_err(|e| js_err("navigator.storage.getDirectory()", e))?;
    Ok(root.unchecked_into())
}

/// `navigator.storage`, reached through the global scope so this works
/// in a worker (no `window`) as well as on a page.
fn storage_manager() -> Result<StorageManager, JsError> {
    let global = js_sys::global();
    let navigator = present(Reflect::get(&global, &JsValue::from_str("navigator")).ok())
        .ok_or_else(opfs_unavailable)?;
    let storage = present(Reflect::get(&navigator, &JsValue::from_str("storage")).ok())
        .ok_or_else(opfs_unavailable)?;
    let has_get_directory = Reflect::get(&storage, &JsValue::from_str("getDirectory"))
        .map(|f| f.is_function())
        .unwrap_or(false);
    if !has_get_directory {
        return Err(opfs_unavailable());
    }
    Ok(storage.unchecked_into())
}

fn present(value: Option<JsValue>) -> Option<JsValue> {
    value.filter(|v| !v.is_undefined() && !v.is_null())
}

fn opfs_unavailable() -> JsError {
    JsError::new(
        "BundleRepo needs the Origin Private File System, and \
         `navigator.storage.getDirectory` is not available here. It ships in \
         Chrome 86+, Safari 15.2+ and Firefox 111+. Under Node there is a real \
         filesystem instead: read the GGUF with `node:fs` and pass the bytes to \
         `CeraEngine.fromGgufBytes` or `fromGgufParts`.",
    )
}

/// Subdirectory `name` of `parent`, created when `create` is set.
/// `None` means absent, which is only reachable with `create: false`.
async fn get_directory(
    parent: &FileSystemDirectoryHandle,
    name: &str,
    create: bool,
) -> Result<Option<FileSystemDirectoryHandle>, JsError> {
    let opts = FileSystemGetDirectoryOptions::new();
    opts.set_create(create);
    match JsFuture::from(parent.get_directory_handle_with_options(name, &opts)).await {
        Ok(dir) => Ok(Some(dir.unchecked_into())),
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(js_err(&format!("opening cache directory `{name}`"), e)),
    }
}

/// File `name` in `parent`, created when `create` is set.
async fn get_file(
    parent: &FileSystemDirectoryHandle,
    name: &str,
    create: bool,
) -> Result<Option<FileSystemFileHandle>, JsError> {
    let opts = FileSystemGetFileOptions::new();
    opts.set_create(create);
    match JsFuture::from(parent.get_file_handle_with_options(name, &opts)).await {
        Ok(file) => Ok(Some(file.unchecked_into())),
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(js_err(&format!("opening cache entry `{name}`"), e)),
    }
}

/// Remove `name` from `parent`, reporting whether it existed.
async fn remove_entry(parent: &FileSystemDirectoryHandle, name: &str) -> Result<bool, JsError> {
    match JsFuture::from(parent.remove_entry(name)).await {
        Ok(_) => Ok(true),
        Err(e) if is_not_found(&e) => Ok(false),
        Err(e) => Err(js_err(&format!("removing cache entry `{name}`"), e)),
    }
}

/// Recursively sum the sizes of every file under `dir`.
///
/// Boxed because it recurses: an `async fn` that awaits itself has an
/// infinitely-sized future otherwise.
fn dir_size(dir: FileSystemDirectoryHandle) -> Pin<Box<dyn Future<Output = Result<f64, JsError>>>> {
    Box::pin(async move {
        let iter = dir.values();
        let mut total = 0.0;
        loop {
            let promise = iter
                .next()
                .map_err(|e| js_err("walking the cache directory", e))?;
            let next = JsFuture::from(promise)
                .await
                .map_err(|e| js_err("walking the cache directory", e))?;
            if Reflect::get(&next, &JsValue::from_str("done"))
                .map(|v| v.is_truthy())
                .unwrap_or(true)
            {
                break;
            }
            let value = Reflect::get(&next, &JsValue::from_str("value"))
                .map_err(|e| js_err("walking the cache directory", e))?;
            let handle: FileSystemHandle = value.unchecked_into();
            match handle.kind() {
                FileSystemHandleKind::File => {
                    let file: FileSystemFileHandle = handle.unchecked_into();
                    // An entry deleted mid-walk (another tab clearing
                    // the store) shouldn't fail the whole query; a
                    // partial total beats an error on a size read.
                    total += file_size(&file).await.unwrap_or(0.0);
                }
                _ => {
                    let sub: FileSystemDirectoryHandle = handle.unchecked_into();
                    total += dir_size(sub).await?;
                }
            }
        }
        Ok(total)
    })
}

async fn file_size(handle: &FileSystemFileHandle) -> Result<f64, JsError> {
    let file: File = JsFuture::from(handle.get_file())
        .await
        .map_err(|e| js_err("reading cache entry size", e))?
        .unchecked_into();
    Ok(file.size())
}

/// Read a cached file into wasm memory.
///
/// Uses a sync access handle when the scope allows one: `read` fills a
/// view over wasm linear memory directly, so a multi-GB model never
/// exists as a JS `ArrayBuffer` at the same time as its Rust `Vec`. The
/// main-thread fallback goes through `Blob.arrayBuffer()`, which does
/// pay that second copy.
async fn read_file(handle: &FileSystemFileHandle) -> Result<Vec<u8>, JsError> {
    if has_sync_access(handle)
        && let Ok(sync) = JsFuture::from(handle.create_sync_access_handle()).await
    {
        let sync: FileSystemSyncAccessHandle = sync.unchecked_into();
        let size = sync
            .get_size()
            .map_err(|e| js_err("sizing the cache entry", e))?;
        let mut buf = vec![0u8; size as usize];
        let read = sync.read_with_u8_array(&mut buf);
        // Release the exclusive lock before propagating any error, or
        // the next open of this file fails with NoModificationAllowed.
        sync.close();
        let read = read.map_err(|e| js_err("reading the cache entry", e))?;
        if read < size {
            return Err(JsError::new(&format!(
                "short read from the cache ({read} of {size} bytes)"
            )));
        }
        return Ok(buf);
    }

    let file: File = JsFuture::from(handle.get_file())
        .await
        .map_err(|e| js_err("opening the cache entry", e))?
        .unchecked_into();
    let buffer = JsFuture::from(file.array_buffer())
        .await
        .map_err(|e| js_err("reading the cache entry", e))?;
    Ok(Uint8Array::new(&buffer).to_vec())
}

/// Chunk size for the streaming rehash. Large enough that the per-call
/// JS boundary crossing is noise against the hashing itself, small
/// enough that a 4 GB file doesn't need a 4 GB buffer to verify.
const REHASH_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// SHA-256 of a cached file.
///
/// Reads in chunks through a sync access handle where one is available,
/// so verifying a multi-GB cached model costs a few MB rather than a
/// second copy of the model. The main-thread fallback has no
/// random-access read, so it pays the full buffer; that path only runs
/// on a page that isn't using a worker.
async fn sha256_of(handle: &FileSystemFileHandle) -> Result<String, JsError> {
    let mut hasher = Sha256::new();

    if has_sync_access(handle)
        && let Ok(sync) = JsFuture::from(handle.create_sync_access_handle()).await
    {
        let sync: FileSystemSyncAccessHandle = sync.unchecked_into();
        let result = hash_via_sync_handle(&sync, &mut hasher);
        // Release the exclusive lock whether or not the read worked, or
        // every later open of this file fails with
        // NoModificationAllowedError.
        sync.close();
        result?;
        return Ok(hex(&hasher.finalize()));
    }

    hasher.update(&read_file(handle).await?);
    Ok(hex(&hasher.finalize()))
}

/// Feed `hasher` from `sync` in `REHASH_CHUNK_BYTES` slices.
///
/// Split out so the caller can `close()` the handle on both the success
/// and the error path without duplicating it.
fn hash_via_sync_handle(
    sync: &FileSystemSyncAccessHandle,
    hasher: &mut Sha256,
) -> Result<(), JsError> {
    let size = sync
        .get_size()
        .map_err(|e| js_err("sizing the cache entry", e))?;
    let mut buf = vec![0u8; REHASH_CHUNK_BYTES];
    let mut offset = 0.0;
    while offset < size {
        let opts = FileSystemReadWriteOptions::new();
        opts.set_at(offset);
        let read = sync
            .read_with_u8_array_and_options(&mut buf, &opts)
            .map_err(|e| js_err("reading the cache entry", e))?;
        if read <= 0.0 {
            // Short of `size` with nothing left to read: the file
            // shrank under us. Returning the hash of a prefix would
            // silently fail the comparison and trigger a re-download,
            // which is right, but saying so beats a mystery.
            return Err(JsError::new(
                "cache entry ended early while rehashing; it may have been \
                 truncated by another tab",
            ));
        }
        hasher.update(&buf[..read as usize]);
        offset += read;
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether a rejected DOM promise is a `NotFoundError`, i.e. "this
/// entry doesn't exist" rather than a real failure.
fn is_not_found(value: &JsValue) -> bool {
    Reflect::get(value, &JsValue::from_str("name"))
        .ok()
        .and_then(|v| v.as_string())
        .is_some_and(|name| name == "NotFoundError")
}

/// Render a rejected JS value as something a developer can act on.
/// `JsValue`'s `Debug` prints `JsValue(DOMException)` for a
/// DOMException, dropping the message that says what went wrong.
fn describe(value: &JsValue) -> String {
    if let Some(s) = value.as_string() {
        return s;
    }
    let name = Reflect::get(value, &JsValue::from_str("name"))
        .ok()
        .and_then(|v| v.as_string());
    let message = Reflect::get(value, &JsValue::from_str("message"))
        .ok()
        .and_then(|v| v.as_string());
    match (name, message) {
        (Some(n), Some(m)) => format!("{n}: {m}"),
        (Some(n), None) => n,
        (None, Some(m)) => m,
        (None, None) => format!("{value:?}"),
    }
}

fn js_err(context: &str, value: JsValue) -> JsError {
    JsError::new(&format!("{context}: {}", describe(&value)))
}

/// Ask the browser to exempt this origin's storage from eviction under
/// disk pressure, resolving to whether persistence is now in effect.
///
/// Worth calling before a multi-GB download. Without it, a browser is
/// free to evict the cache: Chrome does so only under real pressure,
/// but Safari discards non-persisted storage after roughly a week of
/// the site going unused, which shows up as a surprise re-download.
/// Some browsers grant it silently based on engagement, others prompt,
/// and a `false` result is normal rather than an error.
///
/// **Requesting persistence is a Window-only capability.** `persist()`
/// is not exposed on a worker's `StorageManager`, and a worker is
/// exactly where an engine embedder tends to run, so calling it there
/// would throw for a completely ordinary caller. From a worker this
/// falls back to `persisted()`, which *is* exposed and reports whether
/// the page already obtained persistence. So the return value always
/// answers "is this origin's storage protected", and a worker that
/// wants to *change* the answer has to ask its page to call this.
#[wasm_bindgen(js_name = persistStorage)]
pub async fn persist_storage() -> Result<bool, JsError> {
    let storage = storage_manager()?;
    let request = if has_method(&storage, "persist") {
        storage
            .persist()
            .map_err(|e| js_err("navigator.storage.persist()", e))?
    } else if has_method(&storage, "persisted") {
        storage
            .persisted()
            .map_err(|e| js_err("navigator.storage.persisted()", e))?
    } else {
        return Ok(false);
    };
    let granted = JsFuture::from(request)
        .await
        .map_err(|e| js_err("navigator.storage persistence query", e))?;
    Ok(granted.is_truthy())
}

/// Bundles published on `LiquidAI/LeapBundles`, as
/// `[{ name, quants: [...] }]`.
///
/// One GET to the HuggingFace model-info endpoint, grouped by the same
/// parser the native CLI uses, so the browser and the CLI list the same
/// catalog. Entries whose names wouldn't survive
/// `CeraEngine.fromBundleId` are filtered out rather than offered.
#[wasm_bindgen(js_name = listLeapBundles)]
pub async fn list_leap_bundles() -> Result<JsValue, JsError> {
    // Same deadline as the HEAD probe: this is a small JSON response,
    // and a picker that never opens is worse than one that reports it
    // couldn't reach the catalog.
    let init = RequestInit::new();
    init.set_signal(Some(&abort_after(HEAD_TIMEOUT_MS)));
    let response = fetch_ok(LEAP_BUNDLES_API_URL, Some(&init)).await?;
    let body = JsFuture::from(
        response
            .text()
            .map_err(|e| js_err("reading the LeapBundles catalog", e))?,
    )
    .await
    .map_err(|e| js_err("reading the LeapBundles catalog", e))?
    .as_string()
    .ok_or_else(|| JsError::new("the LeapBundles catalog was not text"))?;

    let entries = parse_leap_bundles(&body).map_err(|e| JsError::new(&e.to_string()))?;
    let out = js_sys::Array::new();
    for entry in entries {
        let obj = js_sys::Object::new();
        set(&obj, "name", &JsValue::from_str(&entry.name))?;
        let quants = js_sys::Array::new();
        for quant in &entry.quants {
            quants.push(&JsValue::from_str(quant));
        }
        set(&obj, "quants", &quants)?;
        out.push(&obj);
    }
    Ok(out.into())
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsError> {
    Reflect::set(obj, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|e| js_err("building the result object", e))
}

/// Resolve a bundle id + quant to its manifest URL, then load every
/// file it names through `repo`.
///
/// Returns the pieces `CeraEngine::from_parts` needs. Split out from
/// the constructor so the bytes never cross into JS: a 1 GB model would
/// otherwise be copied into an `ArrayBuffer` and straight back.
pub(crate) async fn load_bundle(
    repo: &BundleRepo,
    bundle_id: &str,
    quant: &str,
    on_progress: Option<&Function>,
) -> Result<cera::ModelBytes, JsError> {
    let manifest_url =
        leap_bundles_manifest_url(bundle_id, quant).map_err(|e| JsError::new(&e.to_string()))?;
    load_manifest(repo, &manifest_url, on_progress).await
}

/// Load the bundle described by the manifest at `manifest_url`.
pub(crate) async fn load_manifest(
    repo: &BundleRepo,
    manifest_url: &str,
    on_progress: Option<&Function>,
) -> Result<cera::ModelBytes, JsError> {
    let manifest_bytes = repo
        .read_or_download(manifest_url, None, on_progress)
        .await?;
    let manifest = cera::manifest::Manifest::from_bytes(&manifest_bytes)
        .map_err(|e| JsError::new(&format!("parsing `{manifest_url}`: {e:#}")))?;

    let model_url = join_url(manifest_url, &manifest.files.model).map_err(|e| JsError::new(&e))?;
    let model = repo.read_or_download(&model_url, None, on_progress).await?;

    let mmproj = match manifest.files.multimodal_projector.as_deref() {
        Some(rel) => {
            let url = join_url(manifest_url, rel).map_err(|e| JsError::new(&e))?;
            Some(repo.read_or_download(&url, None, on_progress).await?)
        }
        None => None,
    };

    Ok(cera::ModelBytes {
        model: model.into(),
        multimodal_projector: mmproj.map(Into::into),
        // The manifest states the modality outright, so unlike
        // `fromGgufParts` this path never has to infer it from whether
        // an mmproj was supplied.
        inference_type: Some(manifest.inference_type),
        chat_template: manifest.chat_template,
    })
}

/// Resolve a manifest entry against the manifest's own URL.
///
/// Entries are usually bare filenames sitting next to the manifest, but
/// the schema also allows an absolute URL, which is passed through.
///
/// Returns the message rather than a `JsError` so the rules above stay
/// testable: a `JsError` is write-only from Rust, so a test could only
/// assert that an error occurred, not that it's the right one.
fn join_url(manifest_url: &str, entry: &str) -> Result<String, String> {
    let lower = entry.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return Ok(entry.to_string());
    }
    if entry.is_empty() {
        return Err("manifest names an empty file entry".to_string());
    }
    if entry.contains('/') {
        // The native loader resolves these against a local directory.
        // Here there is no directory, and quietly gluing a relative
        // path onto the manifest URL would fetch the wrong thing on a
        // nested layout instead of saying it can't.
        return Err(format!(
            "manifest entry `{entry}` is not a plain filename or absolute URL; \
             bundles with nested layouts aren't loadable from the browser yet"
        ));
    }
    let base = manifest_url
        .rsplit_once('/')
        .map(|(base, _)| base)
        .ok_or_else(|| format!("manifest URL `{manifest_url}` has no path"))?;
    Ok(format!("{base}/{entry}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    // `#[wasm_bindgen_test]`, not `#[test]`: this crate is
    // `cfg(target_arch = "wasm32")`, so a plain `#[test]` compiles and
    // is then never executed by anything, which reads as a pass. These
    // run under `just wasm-test`.
    //
    // Scope is the pure URL logic. The OPFS paths need a real browser
    // and belong in `tests/`, but the addressing is where a bug would
    // be silent rather than loud: a sidecar that resolves onto its own
    // payload overwrites the model with a hash and the failure surfaces
    // much later as a corrupt GGUF.

    #[wasm_bindgen_test]
    fn sidecar_sits_next_to_its_payload() {
        assert_eq!(
            sidecar_url("https://hf.co/a/model.gguf"),
            "https://hf.co/a/model.gguf.sha256"
        );
    }

    #[wasm_bindgen_test]
    fn sidecar_drops_query_before_appending() {
        // `cache_relative_segments` strips the query, so appending to
        // the raw URL would give the sidecar the same entry name as the
        // payload and overwrite the model with its own hash.
        assert_eq!(
            sidecar_url("https://hf.co/a/model.gguf?download=true"),
            "https://hf.co/a/model.gguf.sha256"
        );
        let payload = cache_relative_segments("https://hf.co/a/model.gguf?download=true").unwrap();
        let sidecar =
            cache_relative_segments(&sidecar_url("https://hf.co/a/model.gguf?download=true"))
                .unwrap();
        assert_ne!(payload, sidecar);
    }

    #[wasm_bindgen_test]
    fn manifest_entries_resolve_next_to_the_manifest() {
        let manifest = "https://huggingface.co/LiquidAI/LeapBundles/resolve/main/B/Q4_0.json";
        assert_eq!(
            join_url(manifest, "model.gguf").unwrap(),
            "https://huggingface.co/LiquidAI/LeapBundles/resolve/main/B/model.gguf"
        );
    }

    #[wasm_bindgen_test]
    fn absolute_manifest_entries_pass_through() {
        let manifest = "https://hf.co/a/Q4_0.json";
        assert_eq!(
            join_url(manifest, "https://cdn.example.com/m.gguf").unwrap(),
            "https://cdn.example.com/m.gguf"
        );
        assert_eq!(
            join_url(manifest, "HTTPS://cdn.example.com/m.gguf").unwrap(),
            "HTTPS://cdn.example.com/m.gguf"
        );
    }

    #[wasm_bindgen_test]
    fn nested_manifest_entries_are_refused_not_guessed() {
        let manifest = "https://hf.co/a/Q4_0.json";
        let err = join_url(manifest, "sub/model.gguf").unwrap_err();
        assert!(
            err.contains("nested layouts"),
            "a nested entry must be refused explicitly, not joined blindly; got: {err}"
        );
        assert!(
            join_url(manifest, "").is_err(),
            "an empty entry is not a filename"
        );
    }

    #[wasm_bindgen_test]
    fn hex_is_lowercase_and_zero_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }
}

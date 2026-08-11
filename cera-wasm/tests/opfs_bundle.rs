//! Browser test for the OPFS-backed bundle store.
//!
//! The unit tests in `src/bundle.rs` cover the pure addressing; this
//! covers the half that only exists inside a browser. Nothing in
//! `navigator.storage`, `createSyncAccessHandle` or `fetch` streaming
//! can be exercised on the host or under Node, so without this file the
//! entire storage layer would ship on the strength of it compiling.
//!
//! The URL under test is the harness page itself, served over HTTP by
//! `wasm-bindgen-test-runner`. That keeps the test hermetic (no network,
//! no fixture) while still going through a real `fetch`, a real
//! streaming body, and real OPFS writes.
//!
//! Run: `just wasm-test-opfs`. Requires a Chrome + matching chromedriver.

#![cfg(target_arch = "wasm32")]

use cera_wasm::bundle::BundleRepo;
use js_sys::{Function, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// A same-origin URL the test server will actually serve.
///
/// `cache_relative_segments` requires a path component, and the harness
/// page is usually served at `/`, so fall back to the document that a
/// directory URL resolves to.
fn page_url() -> String {
    let location = Reflect::get(&js_sys::global(), &JsValue::from_str("location"))
        .expect("no `location` in the test scope");
    let href = Reflect::get(&location, &JsValue::from_str("href"))
        .ok()
        .and_then(|v| v.as_string())
        .expect("`location.href` was not a string");
    // Strip query/fragment so the cache key and the request agree, then
    // give a bare-directory URL something to name.
    let href = href.split(['?', '#']).next().unwrap_or(&href).to_string();
    if href.ends_with('/') {
        // Named after a file the harness itself serves, not `index.html`.
        // `cache_relative_segments` rejects a URL with no path component, so a
        // bare directory URL needs *something* appended, and `index.html` is
        // the obvious guess. It is also wrong: `wasm-pack test` serves the
        // harness page at `/` without an `index.html` behind it, so every
        // download test here 404'd the moment this ran anywhere other than a
        // static-file server. This path is one the runner is known to serve,
        // because it is where the browser loaded the harness JS from.
        format!("{href}wasm-bindgen-test")
    } else {
        href
    }
}

/// A progress callback that counts its invocations.
///
/// Returned as a live `Closure` alongside the counter: dropping the
/// `Closure` invalidates the function, so the caller has to keep it.
fn counting_progress() -> (
    Closure<dyn FnMut(JsValue, f64, JsValue)>,
    std::rc::Rc<std::cell::Cell<u32>>,
) {
    let calls = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let sink = calls.clone();
    let closure = Closure::wrap(Box::new(move |_url: JsValue, _done: f64, _total: JsValue| {
        sink.set(sink.get() + 1);
    }) as Box<dyn FnMut(JsValue, f64, JsValue)>);
    (closure, calls)
}

fn as_function(closure: &Closure<dyn FnMut(JsValue, f64, JsValue)>) -> Function {
    closure.as_ref().unchecked_ref::<Function>().clone()
}

/// A store directory name is one entry, not a path. `getDirectoryHandle`
/// would take `..` as a literal name and quietly address a sibling of
/// the store, so this has to be refused at construction.
#[wasm_bindgen_test]
fn store_dir_must_be_a_single_safe_segment() {
    assert!(BundleRepo::new(Some("..".into())).is_err());
    assert!(BundleRepo::new(Some("a/b".into())).is_err());
    assert!(BundleRepo::new(Some(String::new())).is_err());
    assert!(BundleRepo::new(Some("cera-models".into())).is_ok());
    assert!(BundleRepo::new(None).is_ok());
}

/// Querying a store that has never been written must not create it, and
/// must report empty rather than failing on the missing directory.
#[wasm_bindgen_test]
async fn a_never_used_store_is_empty_and_clears_cleanly() {
    let repo = BundleRepo::new(Some("cera-test-untouched".into())).expect("construct repo");
    assert_eq!(repo.cache_size().await.expect("cache_size"), 0.0);
    // Idempotent on a directory that was never created.
    repo.clear_cache().await.expect("clear_cache");
    repo.clear_cache().await.expect("clear_cache twice");
    assert_eq!(repo.cache_size().await.expect("cache_size"), 0.0);
}

/// The full round trip: fetch over HTTP, stream into OPFS, serve the
/// second read from the cache, then remove.
#[wasm_bindgen_test]
async fn download_caches_to_opfs_and_reads_back() {
    let repo = BundleRepo::new(Some("cera-test-roundtrip".into())).expect("construct repo");
    // Independent of whatever a previous run left behind.
    repo.clear_cache().await.expect("clear_cache");

    let url = page_url();
    assert!(!repo.is_cached(url.clone()).await.expect("is_cached"));

    let (progress, calls) = counting_progress();
    repo.download(url.clone(), None, Some(as_function(&progress)))
        .await
        .expect("download");
    assert!(
        calls.get() >= 1,
        "the end-of-stream progress report must fire even for a body \
         smaller than the throttle interval"
    );

    assert!(repo.is_cached(url.clone()).await.expect("is_cached"));

    let bytes = repo.bytes(url.clone(), None, None).await.expect("bytes");
    assert!(bytes.length() > 0, "cached an empty body");

    // Payload plus a 64-hex-character `.sha256` sidecar, and nothing
    // else. Exact rather than `>`: if the sidecar resolved onto the
    // payload's own cache entry it would overwrite the file with its
    // hash, and every looser assertion here would still pass.
    assert_eq!(
        repo.cache_size().await.expect("cache_size"),
        f64::from(bytes.length()) + 64.0,
        "expected exactly the payload and its 64-byte sidecar"
    );

    // A second download is a cache hit: it must not stream again.
    let (progress2, calls2) = counting_progress();
    repo.download(url.clone(), None, Some(as_function(&progress2)))
        .await
        .expect("second download");
    assert_eq!(
        calls2.get(),
        0,
        "a cache hit must not re-download; progress fired {} times",
        calls2.get()
    );

    // Removing an entry takes its sidecar with it, or a later
    // re-download would be checked against a stale hash.
    assert!(repo.remove(url.clone()).await.expect("remove"));
    assert!(!repo.is_cached(url.clone()).await.expect("is_cached"));
    assert_eq!(
        repo.cache_size().await.expect("cache_size"),
        0.0,
        "remove left the sidecar behind"
    );

    repo.clear_cache().await.expect("clear_cache");
}

/// A hash the bytes don't match must fail the download *and* leave
/// nothing cached. A partial or wrong file left behind would be served
/// as a cache hit by the next size-checked load.
#[wasm_bindgen_test]
async fn a_hash_mismatch_fails_and_caches_nothing() {
    let repo = BundleRepo::new(Some("cera-test-mismatch".into())).expect("construct repo");
    repo.clear_cache().await.expect("clear_cache");

    let url = page_url();
    let wrong = "0".repeat(64);
    assert!(
        repo.download(url.clone(), Some(wrong), None).await.is_err(),
        "a mismatched sha256 must be rejected"
    );
    assert!(!repo.is_cached(url.clone()).await.expect("is_cached"));
    assert_eq!(
        repo.cache_size().await.expect("cache_size"),
        0.0,
        "a rejected download must not leave bytes in the cache"
    );

    repo.clear_cache().await.expect("clear_cache");
}

/// Two repos with different store directories must not see each other's
/// entries: `storeDir` is the isolation boundary an app relies on when
/// it caches more than one thing.
#[wasm_bindgen_test]
async fn stores_are_isolated_by_directory() {
    let a = BundleRepo::new(Some("cera-test-iso-a".into())).expect("repo a");
    let b = BundleRepo::new(Some("cera-test-iso-b".into())).expect("repo b");
    a.clear_cache().await.expect("clear a");
    b.clear_cache().await.expect("clear b");

    let url = page_url();
    a.download(url.clone(), None, None).await.expect("download");

    assert!(a.is_cached(url.clone()).await.expect("is_cached a"));
    assert!(!b.is_cached(url.clone()).await.expect("is_cached b"));
    assert_eq!(b.cache_size().await.expect("cache_size b"), 0.0);

    a.clear_cache().await.expect("clear a");
    assert!(!a.is_cached(url).await.expect("is_cached a after clear"));
}

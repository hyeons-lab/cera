//! Cache-key and catalog logic shared by every bundle store.
//!
//! Everything here is pure: no filesystem, no HTTP, no `remote` feature.
//! That is the point. There are two bundle stores in the workspace now —
//! the native [`super::BundleRepo`] (an on-disk tree under `store_dir`)
//! and the browser one in `cera-wasm` (the Origin Private File System) —
//! and they must agree on:
//!
//! - which cache entry a URL maps to ([`cache_relative_segments`]),
//! - which characters are safe in a path segment
//!   ([`validate_path_segment`]),
//! - how a bundle id + quant becomes a manifest URL
//!   ([`leap_bundles_manifest_url`]), and
//! - how the LeapBundles catalog JSON is grouped ([`parse_leap_bundles`]).
//!
//! Re-deriving any of those on the web side would be a second
//! implementation asserting it matches the first, which is exactly the
//! kind of drift that costs a debugging session when it stops being
//! true. So they live here and both stores call them.
//!
//! The *storage* differs (directories versus OPFS handles) and the
//! *integrity policy* differs (see `cera-wasm`'s bundle module on what
//! CORS makes impossible in a browser); the addressing does not.

use std::collections::{BTreeMap, BTreeSet};

use crate::session::CeraError;

/// HuggingFace model-info endpoint for the LeapBundles repo. The
/// response carries a `siblings` array listing every file in the
/// repo (one round-trip), which [`parse_leap_bundles`] walks to build
/// the bundle/quant catalog.
///
/// Public because the browser store issues this request itself
/// (`fetch`) rather than through reqwest, but must hit the same URL.
pub const LEAP_BUNDLES_API_URL: &str = "https://huggingface.co/api/models/LiquidAI/LeapBundles";

/// One bundle entry in the LeapBundles catalog: a directory at
/// `LiquidAI/LeapBundles/<name>/` plus the per-quant manifests
/// (`<quant>.json`) Liquid publishes inside it.
///
/// Returned by [`super::list_leap_bundles`]; both `name` and `quants`
/// are sorted ascending so output is stable across runs even if the
/// HF API reorders its `siblings` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeapBundleEntry {
    pub name: String,
    pub quants: Vec<String>,
}

/// Map a URL to the cache-relative path segments that address it,
/// `[host, ...path segments]`.
///
/// The **host** is lowercased before it becomes a directory name so
/// URLs that differ only in host casing (per RFC 3986 §3.2.2, hosts are
/// case-insensitive) share a cache entry on case-sensitive filesystems.
/// Path segments stay as-is: they're content-addressable and different
/// casings can legitimately resolve to different resources on the
/// origin.
///
/// Every segment goes through [`validate_path_segment`], so a URL whose
/// host or path could escape the cache root (`..`, null bytes, a bare
/// `/` component) is rejected here rather than at join time. Neither
/// `PathBuf::push` nor an OPFS `getDirectoryHandle` call is a validator,
/// and an attacker-controlled URL must not be able to address anything
/// outside the store.
///
/// Query and fragment are stripped: they're URL syntax that doesn't
/// belong in a cache key, and leaving them in would produce entries
/// named `model.gguf?foo=bar`. If a bundle URL ever genuinely needs
/// them to identify the resource, they'd have to be passed separately;
/// today every bundle URL is a clean path.
pub fn cache_relative_segments(url: &str) -> Result<Vec<String>, CeraError> {
    let (host, path) = split_url(url)?;
    let host_segment = encode_host(host);
    validate_path_segment("url host", &host_segment)?;

    let path_no_qs = path
        .trim_start_matches('/')
        .split(['?', '#'])
        .next()
        .unwrap_or("");
    if path_no_qs.is_empty() {
        return Err(CeraError::Backend(format!(
            "url `{url}` has no path component"
        )));
    }

    let mut segments = Vec::with_capacity(1 + path_no_qs.matches('/').count());
    segments.push(host_segment);
    for segment in path_no_qs.split('/') {
        validate_path_segment("url path segment", segment)?;
        segments.push(segment.to_string());
    }
    Ok(segments)
}

/// Build the canonical manifest URL for a LeapBundles entry.
///
/// The LeapBundles repo on HuggingFace is a flat catalog at
/// `LiquidAI/LeapBundles`; each bundle occupies a top-level directory
/// named after the model (e.g. `LFM2-1.2B-GGUF/`), and per-quant
/// manifests live inside as `<QUANT>.json` (e.g. `Q4_0.json`,
/// `F16.json`). There is no top-level index — bundle IDs are passed in
/// as opaque strings.
///
/// `bundle_id` / `quant` are interpolated directly into a URL path and
/// must be safe filesystem components, so they go through the same
/// strict [`validate_path_segment`] allowlist used for cache-dir
/// segments. URL-reserved characters (`?`, `#`, `%`) are rejected so
/// they can't alter URL semantics when interpolated.
pub fn leap_bundles_manifest_url(bundle_id: &str, quant: &str) -> Result<String, CeraError> {
    validate_path_segment("bundle_id", bundle_id)?;
    validate_path_segment("quant", quant)?;
    Ok(format!(
        "https://huggingface.co/LiquidAI/LeapBundles/resolve/main/{bundle_id}/{quant}.json"
    ))
}

/// Parse a HuggingFace model-info JSON body into the bundle catalog.
/// Split out from the HTTP call so the grouping/filtering logic is
/// unit-testable without a live round-trip, and so the browser store
/// can reuse it after its own `fetch`.
pub fn parse_leap_bundles(body: &str) -> Result<Vec<LeapBundleEntry>, CeraError> {
    #[derive(serde::Deserialize)]
    struct Sibling {
        rfilename: String,
    }
    #[derive(serde::Deserialize)]
    struct Resp {
        siblings: Vec<Sibling>,
    }
    let resp: Resp = serde_json::from_str(body)
        .map_err(|e| CeraError::Backend(format!("list-bundles JSON parse failed: {e}")))?;

    // BTreeMap/BTreeSet sort by key; the iter() walk that follows
    // produces a stable lexicographic ordering of bundles + quants.
    let mut by_bundle: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for sib in resp.siblings {
        // Only `<bundle>/<quant>.json` shapes count. Everything
        // else (top-level `*.bundle` blobs, `.gitattributes`,
        // `README.md`, nested-deeper paths) is silently dropped —
        // those aren't consumable by `from_bundle_id` and
        // surfacing them would confuse users.
        let Some((dir, file)) = sib.rfilename.split_once('/') else {
            continue;
        };
        let Some(quant) = file.strip_suffix(".json") else {
            continue;
        };
        if quant.contains('/') {
            // Nested deeper than `<bundle>/<quant>.json` — not
            // part of LeapBundles' schema today; skip rather than
            // mis-display.
            continue;
        }
        // Both segments must survive the same allowlist that
        // `leap_bundles_manifest_url` enforces — otherwise we'd
        // surface entries that `from_bundle_id` would reject at
        // resolve time. Today every entry passes this check, but
        // a future HF entry with non-ASCII or whitespace would
        // get filtered cleanly here instead of producing a
        // confusing post-list `--bundle-id` failure.
        if validate_path_segment("bundle_id", dir).is_err()
            || validate_path_segment("quant", quant).is_err()
        {
            continue;
        }
        by_bundle
            .entry(dir.to_string())
            .or_default()
            .insert(quant.to_string());
    }
    Ok(by_bundle
        .into_iter()
        .map(|(name, quants)| LeapBundleEntry {
            name,
            quants: quants.into_iter().collect(),
        })
        .collect())
}

/// Fold `host[:port]` into one directory-safe segment.
///
/// A port is common the moment a URL isn't a public CDN: a dev server on
/// `localhost:8080`, an internal mirror on `:8443`. Without this the
/// whole URL is rejected, because `:` is banned from a path segment
/// (Windows reads it as a drive letter or an NTFS alternate data
/// stream), so `localhost:8080` could never become a directory name.
///
/// The port is joined with `_`, and any `_` already in the host is
/// doubled, so the mapping can't collide: host `a_1` becomes `a__1`
/// while host `a` port `1` becomes `a_1`. Underscores are not legal in
/// hostnames anyway, but a cache key that can alias two origins onto one
/// entry would serve the wrong model, and the escape costs nothing.
///
/// Anything that isn't a plain numeric port is left alone, so the
/// original host reaches [`validate_path_segment`] and produces its
/// error rather than being quietly reshaped into something valid.
fn encode_host(host: &str) -> String {
    let lower = host.to_ascii_lowercase();
    match lower.split_once(':') {
        Some((name, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
            format!("{}_{port}", name.replace('_', "__"))
        }
        // No port, or one that isn't plain digits. The escape still
        // applies: doubling only in the port branch would leave host
        // `a_1` mapping to `a_1`, the same entry as host `a` port `1`,
        // which is the collision the escape exists to prevent. A
        // malformed port keeps its `:` and is rejected downstream.
        _ => lower.replace('_', "__"),
    }
}

/// Strict allowlist for path segments: ASCII alphanumerics and
/// `-`, `_`, `.`. Everything else is rejected:
/// - `/`, `\` (path separators on any OS)
/// - `:` (Windows drive letters / NTFS alternate data streams)
/// - `*`, `"`, `<`, `>`, `|` (Windows-reserved)
/// - whitespace, null bytes, control chars (confusing / truncating)
/// - URL-reserved `?`, `#`, `%` (semantics-altering under URL parsing)
/// - non-ASCII (keeps paths portable across codepages; real bundle IDs
///   and URL segments in `LiquidAI/LeapBundles` are all ASCII today)
///
/// Public because the browser store validates its own store-directory
/// name with it: `getDirectoryHandle` would accept `..` as a literal
/// entry name, so a caller passing a path needs the same up-front
/// rejection the native store gets from `PathBuf` hygiene.
///
/// Used for both URL-derived cache segments (in
/// [`cache_relative_segments`], where lax input could escape the store
/// root via `..`) and LeapBundles bundle-id / quant components (in
/// [`leap_bundles_manifest_url`], where lax input could 404 or
/// manipulate the cache path). One allowlist covers both because the
/// same filename-safe subset works for every real bundle identifier
/// shipped in `LiquidAI/LeapBundles` today.
pub fn validate_path_segment(kind: &str, segment: &str) -> Result<(), CeraError> {
    if segment.is_empty() {
        return Err(CeraError::Backend(format!("{kind} must not be empty")));
    }
    if segment == "." || segment == ".." {
        return Err(CeraError::Backend(format!(
            "{kind} `{segment}` is not a valid path component"
        )));
    }
    for ch in segment.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.');
        if !ok {
            return Err(CeraError::Backend(format!(
                "{kind} `{segment}` contains forbidden character {ch:?}"
            )));
        }
    }
    Ok(())
}

/// Minimal URL parser: extract `(host, path)` from `https://host/path`.
/// Scheme comparison is case-insensitive (RFC 3986 §3.1) to match the
/// case-insensitive check in `engine::is_remote_url` — otherwise a
/// `HTTPS://…` URL would be accepted by the remote-URL gate but
/// rejected here. We avoid pulling in a full `url` crate dep — this
/// is the only URL handling `cera` needs and the shape we accept is
/// narrow.
fn split_url(url: &str) -> Result<(&str, &str), CeraError> {
    // Case-insensitive scheme match: find the `://` and check the
    // preceding label against our supported schemes.
    let scheme_end = url.find("://").ok_or_else(|| {
        CeraError::Backend(format!("url `{url}` must start with https:// or http://"))
    })?;
    let scheme = &url[..scheme_end];
    let lower = scheme.to_ascii_lowercase();
    if lower != "http" && lower != "https" {
        return Err(CeraError::Backend(format!(
            "url `{url}` must start with https:// or http://"
        )));
    }
    let after_scheme = &url[scheme_end + 3..]; // skip "://"
    let (host, path) = after_scheme
        .split_once('/')
        .ok_or_else(|| CeraError::Backend(format!("url `{url}` has no path component")))?;
    if host.is_empty() {
        return Err(CeraError::Backend(format!(
            "url `{url}` has empty host component"
        )));
    }
    // Return the path slice preserving its leading `/` so the caller
    // can detect an empty path after trimming.
    let path_start = url.len() - path.len() - 1;
    Ok((host, &url[path_start..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_segments_mirror_host_and_path() {
        let segs = cache_relative_segments(
            "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/x.gguf",
        )
        .unwrap();
        assert_eq!(
            segs,
            [
                "huggingface.co",
                "LiquidAI",
                "LFM2-1.2B-GGUF",
                "resolve",
                "main",
                "x.gguf"
            ]
        );
    }

    #[test]
    fn split_url_rejects_missing_scheme() {
        assert!(split_url("huggingface.co/x").is_err());
    }

    #[test]
    fn split_url_rejects_missing_path() {
        assert!(split_url("https://huggingface.co").is_err());
    }

    #[test]
    fn split_url_accepts_http_and_https() {
        assert!(split_url("http://example.com/x").is_ok());
        assert!(split_url("https://example.com/x").is_ok());
    }

    #[test]
    fn split_url_scheme_is_case_insensitive() {
        // Mixed-case schemes must be accepted (RFC 3986 §3.1) so we
        // don't drift from `engine::is_remote_url`, which already
        // does case-insensitive matching. Otherwise an `HTTPS://…`
        // URL would pass the remote-URL gate but fail here.
        assert!(split_url("HTTPS://example.com/x").is_ok());
        assert!(split_url("Http://example.com/x").is_ok());
        assert!(split_url("HTTP://example.com/x").is_ok());
    }

    #[test]
    fn cache_segments_lowercase_host_for_cache_consistency() {
        // Two URLs that differ only in host casing must share a cache
        // entry — hosts are case-insensitive per RFC 3986 §3.2.2, but
        // case-sensitive filesystems would otherwise double-cache.
        let a = cache_relative_segments("https://HuggingFace.co/LiquidAI/M/x.gguf").unwrap();
        let b = cache_relative_segments("https://huggingface.co/LiquidAI/M/x.gguf").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, ["huggingface.co", "LiquidAI", "M", "x.gguf"]);
    }

    /// A port has to survive into the cache key: a model served from a
    /// dev server or an internal mirror is an ordinary case, and `:`
    /// can't be a directory name on Windows.
    #[test]
    fn cache_segments_fold_a_port_into_the_host() {
        let segs = cache_relative_segments("http://localhost:8731/model.gguf").unwrap();
        assert_eq!(segs, ["localhost_8731", "model.gguf"]);
        // Different ports are different origins and must not share an
        // entry.
        let other = cache_relative_segments("http://localhost:9000/model.gguf").unwrap();
        assert_ne!(segs, other);
    }

    /// The `_` join must not let two different origins address the same
    /// entry, or one would serve the other's bytes.
    #[test]
    fn cache_segments_keep_ports_distinct_from_underscored_hosts() {
        let with_port = cache_relative_segments("https://a:1/m.gguf").unwrap();
        let underscored = cache_relative_segments("https://a_1/m.gguf").unwrap();
        assert_eq!(with_port, ["a_1", "m.gguf"]);
        assert_eq!(underscored, ["a__1", "m.gguf"]);
        assert_ne!(with_port, underscored);
    }

    /// Only a plain numeric port is folded. Anything else keeps the
    /// original host so the allowlist reports what's actually wrong
    /// instead of the reshaped value silently passing.
    #[test]
    fn cache_segments_reject_a_malformed_port() {
        for bad in [
            "https://a:/m.gguf",
            "https://a:80x/m.gguf",
            "https://a:1:2/m.gguf",
        ] {
            let e = cache_relative_segments(bad).expect_err(&format!("{bad} must be rejected"));
            assert!(
                format!("{e}").contains("forbidden character ':'"),
                "unexpected error for {bad}: {e}"
            );
        }
    }

    #[test]
    fn cache_segments_reject_parent_dir_segment() {
        // Attacker-controlled URL with `..` would otherwise escape the
        // store root once the segments are joined.
        let e = cache_relative_segments("https://evil.example.com/a/../../etc/passwd")
            .expect_err("`..` segment must be rejected");
        assert!(format!("{e}").contains("not a valid path component"));
    }

    #[test]
    fn cache_segments_reject_windows_reserved_chars() {
        // Chars that appear as segment content and are Windows-reserved:
        // `*`, `"`, `<`, `>`, `|`. (`?` and `#` are separately stripped
        // as URL syntax — see `cache_segments_strip_query_and_fragment`.)
        // Catching these up front means a Windows consumer never sees a
        // cryptic filesystem error at join time.
        for bad in ["a*b", "a\"b", "a<b", "a>b", "a|b"] {
            let url = format!("https://example.com/{bad}");
            let e = cache_relative_segments(&url).expect_err(&format!("{bad:?} must be rejected"));
            let msg = format!("{e}");
            assert!(
                msg.contains("forbidden"),
                "unexpected error for {bad:?}: {msg}"
            );
        }
    }

    #[test]
    fn cache_segments_reject_empty_segment() {
        // Double slash produces an empty segment.
        let e = cache_relative_segments("https://example.com/a//b")
            .expect_err("empty path segment must be rejected");
        assert!(format!("{e}").contains("must not be empty"));
    }

    #[test]
    fn cache_segments_strip_query_and_fragment() {
        // Query / fragment are URL syntax; they must not appear in the
        // cache key or the entry becomes `model.gguf?x=1` etc.
        let segs = cache_relative_segments("https://example.com/model.gguf?foo=bar#frag").unwrap();
        assert_eq!(segs, ["example.com", "model.gguf"]);
    }

    #[test]
    fn cache_segments_reject_null_byte_in_path() {
        let e = cache_relative_segments("https://example.com/a\0b")
            .expect_err("null byte in path must be rejected");
        assert!(format!("{e}").contains("forbidden"));
    }

    #[test]
    fn leap_manifest_url_happy_path() {
        let url = leap_bundles_manifest_url("LFM2-1.2B-GGUF", "Q4_0").unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/LiquidAI/LeapBundles/resolve/main/LFM2-1.2B-GGUF/Q4_0.json"
        );
    }

    #[test]
    fn leap_manifest_url_rejects_empty() {
        assert!(leap_bundles_manifest_url("", "Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2-1.2B-GGUF", "").is_err());
    }

    #[test]
    fn leap_manifest_url_rejects_path_separators() {
        assert!(leap_bundles_manifest_url("LFM2/GGUF", "Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2-1.2B-GGUF", "sub/Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2\\GGUF", "Q4_0").is_err());
    }

    #[test]
    fn leap_manifest_url_rejects_parent_dir() {
        assert!(leap_bundles_manifest_url("..", "Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2-1.2B-GGUF", "..").is_err());
    }

    #[test]
    fn leap_manifest_url_rejects_whitespace_and_url_reserved() {
        assert!(leap_bundles_manifest_url("LFM2 GGUF", "Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2-1.2B-GGUF", "Q4 0").is_err());
        assert!(leap_bundles_manifest_url("LFM2-1.2B-GGUF", "Q4_0\n").is_err());
        // URL-reserved chars must be rejected so they can't alter URL
        // semantics when interpolated.
        assert!(leap_bundles_manifest_url("LFM2?x", "Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2#x", "Q4_0").is_err());
        assert!(leap_bundles_manifest_url("LFM2%2E", "Q4_0").is_err());
    }

    /// `parse_leap_bundles` groups `<bundle>/<quant>.json` siblings
    /// into bundle entries with sorted quants, drops top-level
    /// blobs / READMEs, and rejects deeper-nested paths so a future
    /// schema change is visible instead of silently misrendered.
    #[test]
    fn parse_leap_bundles_groups_siblings() {
        let body = r#"{
            "siblings": [
                {"rfilename": ".gitattributes"},
                {"rfilename": "README.md"},
                {"rfilename": "LFM2-1.2B-8da4w_output_8da8w-seq_4096.bundle"},
                {"rfilename": "LFM2-1.2B-GGUF/Q8_0.json"},
                {"rfilename": "LFM2-1.2B-GGUF/Q4_0.json"},
                {"rfilename": "LFM2-1.2B-GGUF/Q4_K_M.json"},
                {"rfilename": "LFM2-2.6B-GGUF/Q4_0.json"},
                {"rfilename": "LFM2-2.6B-GGUF/notes/extra.json"},
                {"rfilename": "LFM2-2.6B-GGUF/extras.txt"}
            ]
        }"#;
        let entries = parse_leap_bundles(body).unwrap();
        assert_eq!(entries.len(), 2, "expected 2 bundles, got {entries:?}");
        // BTreeMap → ascending bundle name order.
        assert_eq!(entries[0].name, "LFM2-1.2B-GGUF");
        assert_eq!(entries[0].quants, vec!["Q4_0", "Q4_K_M", "Q8_0"]);
        assert_eq!(entries[1].name, "LFM2-2.6B-GGUF");
        // `extras.txt` (wrong suffix) and `notes/extra.json`
        // (deeper-nested) both filtered out.
        assert_eq!(entries[1].quants, vec!["Q4_0"]);
    }

    #[test]
    fn parse_leap_bundles_rejects_malformed_json() {
        let err = parse_leap_bundles("not json at all").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("JSON parse failed"), "got: {msg}");
    }

    /// Empty `siblings` is a valid response shape (e.g. an empty
    /// repo) and should yield an empty catalog rather than an
    /// error.
    #[test]
    fn parse_leap_bundles_empty_siblings_is_ok() {
        let entries = parse_leap_bundles(r#"{"siblings": []}"#).unwrap();
        assert!(entries.is_empty());
    }

    /// Entries whose bundle / quant segment would fail
    /// `validate_path_segment` are dropped silently — surfacing
    /// them would mislead users since `from_bundle_id` rejects
    /// the same characters at resolve time. Today's catalog has
    /// none of these, but the filter is forward-compatible with
    /// any HF schema drift toward whitespace / non-ASCII names.
    #[test]
    fn parse_leap_bundles_drops_invalid_path_segments() {
        let body = r#"{
            "siblings": [
                {"rfilename": "Good-Bundle-GGUF/Q4_0.json"},
                {"rfilename": "Has Space-GGUF/Q4_0.json"},
                {"rfilename": "Good-Bundle-GGUF/Q 0.json"},
                {"rfilename": "Has?Reserved/Q4_0.json"},
                {"rfilename": "Café-GGUF/Q4_0.json"}
            ]
        }"#;
        let entries = parse_leap_bundles(body).unwrap();
        assert_eq!(entries.len(), 1, "expected only the valid entry");
        assert_eq!(entries[0].name, "Good-Bundle-GGUF");
        // The invalid quant `Q 0` got filtered, leaving the one
        // good `Q4_0` quant.
        assert_eq!(entries[0].quants, vec!["Q4_0"]);
    }
}

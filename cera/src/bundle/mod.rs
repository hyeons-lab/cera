//! Bundle fetching + caching.
//!
//! Two layers, split by what they need from the host:
//!
//! - [`cache_key`] is pure and always compiled: URL-to-cache-entry
//!   addressing, the path-segment allowlist, LeapBundles manifest URLs
//!   and catalog parsing. The browser store in `cera-wasm` builds on
//!   this, which is why it isn't behind the `remote` feature.
//! - [`BundleRepo`] (behind `remote`) is the native store: a
//!   `reqwest::blocking` downloader over an on-disk tree, with
//!   SHA-256 integrity and sidecar caching.
//!
//! A `wasm32` build gets the first and not the second — there is no
//! filesystem and no reqwest — and supplies its own OPFS-backed store
//! that shares the addressing above. See `cera-wasm`'s bundle module.

pub mod cache_key;

pub use cache_key::{LeapBundleEntry, leap_bundles_manifest_url};

#[cfg(feature = "remote")]
pub(crate) mod download;

#[cfg(feature = "remote")]
mod repo;

#[cfg(feature = "remote")]
pub use repo::{BundleRepo, DownloadProgress, list_leap_bundles};

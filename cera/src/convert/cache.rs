//! Local completion records for streaming conversion. These detect accidental
//! artifact changes; the request also binds the resolved upstream commit.
//! The remote endpoint is trusted to honor commit URLs; records do not authenticate files.

use super::pipeline::QuantizeOptions;
use crate::bundle::{HfSpec, download};
use crate::manifest::Manifest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub(super) const RECEIPT_NAME: &str = "model.gguf.receipt.json";

/// Include every caller option affecting output, in matching order for overrides.
/// Bump the version when conversion semantics invalidate previously built output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ConversionRequest {
    version: u32,
    owner: String,
    repo: String,
    revision: String,
    resolved_revision: String,
    quant: String,
    strategy: String,
    tensor_overrides: Vec<(String, String)>,
}

impl ConversionRequest {
    pub(super) fn new(spec: &HfSpec, opts: &QuantizeOptions, resolved_revision: &str) -> Self {
        Self {
            version: 2,
            owner: spec.owner.clone(),
            repo: spec.repo.clone(),
            revision: spec.revision.clone(),
            resolved_revision: resolved_revision.into(),
            quant: opts.target_quant.as_str().into(),
            strategy: opts.strategy.as_str().into(),
            tensor_overrides: opts
                .tensor_overrides
                .iter()
                .map(|(pattern, quant)| (pattern.clone(), quant.as_str().into()))
                .collect(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct ConversionReceipt {
    request: ConversionRequest,
    manifest_sha256: String,
    model_sha256: String,
}

impl ConversionReceipt {
    pub(super) fn new(
        request: ConversionRequest,
        manifest_bytes: &[u8],
        model_sha256: String,
    ) -> Self {
        Self {
            request,
            manifest_sha256: format!("{:x}", Sha256::digest(manifest_bytes)),
            model_sha256,
        }
    }

    /// Verify the exact bytes before reuse; a sidecar alone cannot prove that
    /// the model has remained intact since conversion. Model hashing streams.
    pub(super) fn verified_manifest(
        receipt_path: &Path,
        manifest_path: &Path,
        model_path: &Path,
        request: &ConversionRequest,
    ) -> Option<Manifest> {
        let receipt: Self = serde_json::from_slice(&fs::read(receipt_path).ok()?).ok()?;
        if &receipt.request != request {
            return None;
        }
        let manifest_bytes = fs::read(manifest_path).ok()?;
        if format!("{:x}", Sha256::digest(&manifest_bytes)) != receipt.manifest_sha256 {
            return None;
        }
        let mut manifest = Manifest::from_bytes(&manifest_bytes).ok()?;
        if download::sha256_file(model_path).ok()? != receipt.model_sha256 {
            return None;
        }
        // A missing/stale auxiliary sidecar is repairable using verified bytes.
        if download::read_sidecar(model_path).as_ref() != Some(&receipt.model_sha256) {
            download::write_sidecar(model_path, &receipt.model_sha256);
        }
        manifest.files.model = model_path.to_string_lossy().into();
        Some(manifest)
    }
}

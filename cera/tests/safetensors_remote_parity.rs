//! Integration test for on-the-fly streaming quantization parity against remote Hugging Face models.
//!
//! Gated behind `--features remote` and opt-in environment variable `CERA_REMOTE_PARITY=1`.
//!
//! Usage:
//! ```sh
//! CERA_REMOTE_PARITY=1 cargo test -p cera --features remote \
//!     --test safetensors_remote_parity -- --ignored
//! ```

#![cfg(all(feature = "remote", feature = "mmap", feature = "std-fs"))]

use std::path::{Path, PathBuf};

use cera::convert::parity::audit_gguf_parity;
use cera::convert::pipeline::{QuantizeOptions, stream_quantize_hf_repo};
use cera::convert::quantize::{QuantStrategy, TargetQuant};

#[test]
#[ignore = "requires network download: set CERA_REMOTE_PARITY=1 and pass --ignored"]
fn test_remote_hf_safetensors_streaming_parity() {
    if std::env::var("CERA_REMOTE_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_REMOTE_PARITY=1 not set");
        return;
    }

    let repo_id = std::env::var("CERA_REMOTE_HF_REPO")
        .unwrap_or_else(|_| "LiquidAI/LFM2-350M-Extract".into());
    let ref_gguf_path = std::env::var("CERA_REMOTE_REF_GGUF")
        .ok()
        .map(PathBuf::from);

    let cache_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/tmp/cera_remote_parity_cache");
    let _ = std::fs::create_dir_all(&cache_dir);

    let spec =
        cera::bundle::HfSpec::parse(&repo_id).expect("valid Hugging Face repo spec required");

    let opts = QuantizeOptions {
        target_quant: TargetQuant::Q4_0,
        strategy: QuantStrategy::Auto,
        tensor_overrides: Vec::new(),
        cache_dir: cache_dir.clone(),
        auth_token: std::env::var("HF_TOKEN").ok(),
        progress: None,
        cancel: None,
    };

    let manifest =
        stream_quantize_hf_repo(&spec, opts).expect("remote streaming quantization failed");

    let converted_gguf = PathBuf::from(manifest.files.model.trim_start_matches("file://"));

    println!(
        "Successfully converted {repo_id} to {}",
        converted_gguf.display()
    );

    if let Some(ref_path) = ref_gguf_path {
        let report = audit_gguf_parity(&converted_gguf, &ref_path, Some("Hello, world!"), None)
            .expect("remote parity audit failed");

        println!("{}", report.format_table());
        assert!(
            report.is_passing(0.98, 0.95),
            "remote parity audit did not meet thresholds"
        );
    }
}

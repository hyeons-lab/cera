use std::path::Path;
use std::sync::Arc;

use cera::gguf::GgufFile;
use cera::model::dspark::DSparkConfig;

#[test]
fn test_dspark_config_defaults_and_fallbacks() {
    let mock_gguf_bytes = {
        // Minimal GGUF header with no metadata
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Tensor count = 0
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count = 0
        bytes
    };

    let gguf = GgufFile::from_bytes(Arc::from(mock_gguf_bytes.into_boxed_slice()))
        .expect("Failed to parse minimal GGUF");
    let config =
        DSparkConfig::from_gguf(&gguf, 65536, 1024).expect("Failed to create DSparkConfig");

    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.vocab_size, 65536);
    assert_eq!(config.block_size, 9);
    assert_eq!(config.markov_rank, 256);
    assert!(config.rope_theta > 0.0);
    assert_eq!(
        config.head_dim % 2,
        0,
        "head_dim must be even for interleaved RoPE"
    );
}

#[test]
fn test_dspark_sidecar_checkpoint_if_present() {
    let candidate_paths = [
        "training/checkpoints/lfm2.5-vl-450m-dspark.gguf",
        "training/checkpoints/lfm2.5-vl-450m-dspark-Q4_0.gguf",
        "training/checkpoints/lfm2.5-vl-450m-dspark-standalone.gguf",
        "training/checkpoints/lfm2.5-vl-450m-dspark-markov.gguf",
    ];

    let found = candidate_paths.iter().find(|p| Path::new(p).exists());
    if let Some(&path) = found {
        eprintln!("[dspark-test] Testing existing checkpoint: {path}");
        let gguf = GgufFile::open(Path::new(path)).expect("Failed to open GGUF");
        let config =
            DSparkConfig::from_gguf(&gguf, 65536, 1024).expect("Failed to parse DSparkConfig");

        assert!(config.num_layers > 0, "num_layers must be > 0");
        assert!(config.block_size >= 2, "block_size must be >= 2");
        assert_eq!(config.head_dim, config.hidden_size / config.num_heads);

        // Verify key tensors exist if it's a Markov sidecar
        if gguf.tensors.contains_key("dspark.markov_w1.weight")
            || gguf.tensors.contains_key("dflash.markov_w1.weight")
        {
            assert!(
                gguf.tensors.contains_key("dspark.markov_w2.weight")
                    || gguf.tensors.contains_key("dflash.markov_w2.weight"),
                "Markov sidecar must have both markov_w1 and markov_w2"
            );
        }
    } else {
        eprintln!(
            "[dspark-test] No pre-trained GGUF found in local workspace; skipping disk test."
        );
    }
}

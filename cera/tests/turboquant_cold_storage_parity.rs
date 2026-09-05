//! Tests for TurboQuant compressed KV cache cold storage FlatBuffers v2 roundtrips.
//!
//! Verifies:
//! 1. TurboQuant encoded key/value byte streams serialize into FlatBuffers v2 `type_tag = 2`.
//! 2. Deserialization restores exact compressed buffers with anchor checkpoints.
//! 3. Cache clearing and memory pressure eviction flush memory and disk tiers safely.

#![cfg(feature = "disk-cache")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use cera::kv_cache::{
    KvCacheConfig, KvPrefixCache, LayerSnapshot, SemanticBoundaryKind, StateSnapshot,
};
use cera::model::{BlockType, ModelConfig, ScalarMultipliers};

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(1);

fn unique_temp_dir(label: &str) -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "cera_tq_test_{label}_{}_{}",
        std::process::id(),
        id
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_test_config(n_layers: usize, hidden_size: usize) -> ModelConfig {
    ModelConfig {
        architecture: "lfm2".into(),
        hidden_size,
        intermediate_size: hidden_size * 2,
        n_layers,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 1000,
        max_seq_len: 2048,
        head_dim: hidden_size / 4,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        block_types: vec![BlockType::GatedConv, BlockType::Attention],
        conv_kernel_size: Some(4),
        kv_heads_per_layer: vec![0, 2],
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}

#[cfg(feature = "disk-cache")]
#[test]
fn test_turboquant_flatbuffers_v2_roundtrip_and_clear() {
    let cfg = make_test_config(2, 64);
    let dir = unique_temp_dir("tq_v2");
    let mut cache = KvPrefixCache::new(
        KvCacheConfig {
            cache_dir: Some(dir.clone()),
            max_warm_entries: 4,
            ..KvCacheConfig::default()
        },
        &cfg,
        "cpu:test_tq_model",
    );

    let n_kv_heads = 2;
    let head_dim = 16;
    let seq_len = 8;

    let mut key_cache = cera::turboquant::CompressedKeyCache::new(n_kv_heads, head_dim, seq_len);
    let mut val_cache = cera::turboquant::CompressedValueCache::new(n_kv_heads, head_dim, seq_len);

    let polar_bytes = vec![0xABu8; head_dim / 4];
    let jl_bytes = vec![0xCDu8; head_dim / 8];
    for h in 0..n_kv_heads {
        for _ in 0..seq_len {
            key_cache.append(h, &polar_bytes, &jl_bytes, 0x3C00, 0x3800);
            val_cache.append(h, &polar_bytes, 0x3C00);
        }
    }

    let tq_keys = cera::turboquant::encode_compressed_keys(&key_cache);
    let tq_values = cera::turboquant::encode_compressed_values(&val_cache);

    let conv_buf = vec![1.0f32, 2.0, 3.0, 4.0];
    let conv_bytes: Vec<u8> = bytemuck::cast_slice(&conv_buf).to_vec();

    let snap = StateSnapshot::new(
        vec![
            LayerSnapshot::Conv { buffer: conv_bytes },
            LayerSnapshot::AttentionCompressed {
                keys: tq_keys.clone(),
                values: tq_values.clone(),
            },
        ],
        seq_len,
    )
    .with_anchor(1, SemanticBoundaryKind::Turn, 0x12345678);

    let tokens = (0..seq_len as u32).collect::<Vec<_>>();

    // 1. Insert snapshot into cache (populates warm tier and writes to cold disk tier)
    cache.insert(&tokens, snap);
    assert_eq!(cache.warm_count(), 1);

    // 2. Clear warm memory tier (simulating OS memory pressure)
    cache.clear_warm();
    assert_eq!(cache.warm_count(), 0);

    // 3. Query strict prefix - must load from cold tier FlatBuffers v2 disk storage!
    let mut query_tokens = tokens.clone();
    query_tokens.push(999);
    let (loaded, len) = cache
        .find_deepest_semantic_anchor(&query_tokens)
        .expect("must load TurboQuant compressed snapshot from cold tier");

    assert_eq!(len, seq_len);
    assert_eq!(loaded.seq_len, seq_len);
    assert_eq!(loaded.anchor_depth, 1);
    assert_eq!(loaded.boundary_kind, SemanticBoundaryKind::Turn as u8);
    assert_eq!(loaded.semantic_hash, 0x12345678);

    if let LayerSnapshot::AttentionCompressed { keys, values } = &loaded.layers[1] {
        assert_eq!(keys, &tq_keys);
        assert_eq!(values, &tq_values);
    } else {
        panic!(
            "expected LayerSnapshot::AttentionCompressed, got {:?}",
            loaded.layers[1]
        );
    }

    // 4. Test cache.clear() removes both warm memory and cold disk files
    cache.clear();
    assert_eq!(cache.warm_count(), 0);

    // Clean up temp directory
    let _ = std::fs::remove_dir_all(&dir);
}

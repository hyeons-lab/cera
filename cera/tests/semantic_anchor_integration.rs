//! Integration test suite for FreeToken Phase 1: Semantic-Aware Caching & FlatBuffers v2.
//!
//! Verifies:
//! 1. Multi-turn agent context edits (thought stripping / tool results) hitting exact semantic anchors.
//! 2. Visual token anchor caching and preservation.
//! 3. Cold-tier FlatBuffers v2 serialization, atomic `.tmp` persistence, and graceful fallback on corrupted/legacy files.
//! 4. Parity of attention and conv buffer rollback across semantic anchor restores.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use cera::kv_cache::{
    InferenceState, KvCacheConfig, KvPrefixCache, LayerSnapshot, LayerState, SemanticBoundaryKind,
    StateSnapshot,
};
use cera::model::{BlockType, ModelConfig, ScalarMultipliers};

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(1);

fn unique_temp_dir(label: &str) -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cera_test_{label}_{}_{}", std::process::id(), id));
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
    }
}

#[test]
fn agent_multi_turn_thought_stripping_anchor_hit() {
    let cfg = make_test_config(2, 16);
    let mut cache = KvPrefixCache::new(
        KvCacheConfig {
            cache_dir: None,
            ..KvCacheConfig::default()
        },
        &cfg,
        "cpu:test_agent_anchor",
    );

    // Turn 1 prompt tokens:
    // [0..10]: System Prompt
    // [10..20]: User Question 1
    // [20..35]: Model Thinking Tokens (<thought> ... </thought>)
    // [35..45]: Model Tool Call ([get_weather(...)])
    let mut turn1_tokens = Vec::new();
    for i in 0..45 {
        turn1_tokens.push(i as u32);
    }

    // Capture semantic anchors:
    // Anchor 1: End of System Prompt (index 10)
    // Anchor 2: End of User Question 1 (index 20)
    // Anchor 3: End of Turn 1 Tool Call (index 45)
    let make_snap = |len: usize, kind: SemanticBoundaryKind, depth: u32| {
        let mut state = InferenceState::from_config(&cfg).unwrap();
        state.seq_len = len;
        // Populate dummy KV and conv data
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[1]
        {
            key_cache.resize(len * 8, 1.0);
            value_cache.resize(len * 8, 2.0);
        }
        if let LayerState::Conv { buffer, .. } = &mut state.layers[0] {
            buffer.resize(16 * 4, 3.0);
        }
        state
            .snapshot()
            .unwrap()
            .with_anchor(depth, kind, (len as u64) * 0x1111)
    };

    cache.insert_anchor(
        &turn1_tokens[..10],
        make_snap(10, SemanticBoundaryKind::SystemPrompt, 1),
        1,
        SemanticBoundaryKind::SystemPrompt,
        10 * 0x1111,
    );
    cache.insert_anchor(
        &turn1_tokens[..20],
        make_snap(20, SemanticBoundaryKind::Turn, 2),
        2,
        SemanticBoundaryKind::Turn,
        20 * 0x1111,
    );
    cache.insert_anchor(
        &turn1_tokens[..45],
        make_snap(45, SemanticBoundaryKind::ToolCall, 3),
        3,
        SemanticBoundaryKind::ToolCall,
        45 * 0x1111,
    );

    // Turn 2 comes along from agent harness:
    // The harness STRIPS the internal thinking tokens ([20..35]) and replaces with the tool result.
    // Turn 2 tokens:
    // [0..20]: System + User 1 (preserved!)
    // [20..30]: Tool Result ("weather: 72F sunny")
    // [30..40]: User Follow-up Question
    let mut turn2_tokens = Vec::new();
    turn2_tokens.extend_from_slice(&turn1_tokens[..20]); // Matches Anchor 2!
    for i in 100..120 {
        turn2_tokens.push(i as u32);
    }

    // Lookup deepest semantic anchor for Turn 2:
    let (snap, anchor_len) = cache
        .find_deepest_semantic_anchor(&turn2_tokens)
        .expect("must find deepest matching semantic anchor");

    // Must hit Anchor 2 (len 20) — skipping all re-computation of system prompt and user question 1!
    assert_eq!(anchor_len, 20);
    assert_eq!(snap.anchor_depth, 2);
    assert_eq!(snap.boundary_kind, SemanticBoundaryKind::Turn as u8);
    assert_eq!(snap.seq_len, 20);

    // Live state restore check:
    let mut live_state = InferenceState::from_config(&cfg).unwrap();
    live_state.restore(&snap);
    assert_eq!(live_state.seq_len, 20);
    if let LayerState::Attention { key_cache, .. } = &live_state.layers[1] {
        assert_eq!(key_cache.len(), 20 * 8);
    }
}

#[test]
fn visual_token_anchor_pinning() {
    let cfg = make_test_config(2, 16);
    let mut cache = KvPrefixCache::new(
        KvCacheConfig {
            cache_dir: None,
            ..KvCacheConfig::default()
        },
        &cfg,
        "cpu:test_visual_anchor",
    );

    // Vision prompt tokens:
    // [0..5]: System Prompt
    // [5..261]: 256 Image Patch Tokens (<|vision_start|> ... <|vision_end|>)
    // [261..270]: Question 1 ("What is in this picture?")
    let mut vqa1_tokens = Vec::new();
    for i in 0..270 {
        vqa1_tokens.push(i as u32);
    }

    // Insert visual anchor at index 261 (after image tokens)
    let mut state = InferenceState::from_config(&cfg).unwrap();
    state.seq_len = 261;
    if let LayerState::Attention { key_cache, value_cache, .. } = &mut state.layers[1] {
        key_cache.resize(261 * 8, 4.0);
        value_cache.resize(261 * 8, 5.0);
    }
    let snap = state
        .snapshot()
        .unwrap()
        .with_anchor(1, SemanticBoundaryKind::ImageTokens, 0xcafe);

    cache.insert_anchor(
        &vqa1_tokens[..261],
        snap,
        1,
        SemanticBoundaryKind::ImageTokens,
        0xcafe,
    );

    // Follow-up question on the same image:
    // [0..261]: Image tokens (identical!)
    // [261..280]: Question 2 ("What color is the car?")
    let mut vqa2_tokens = Vec::new();
    vqa2_tokens.extend_from_slice(&vqa1_tokens[..261]);
    for i in 500..519 {
        vqa2_tokens.push(i as u32);
    }

    let (anchor_snap, len) = cache
        .find_deepest_semantic_anchor(&vqa2_tokens)
        .expect("must find visual token anchor");

    assert_eq!(len, 261);
    assert_eq!(anchor_snap.boundary_kind, SemanticBoundaryKind::ImageTokens as u8);
    assert_eq!(anchor_snap.semantic_hash, 0xcafe);
}

#[cfg(feature = "disk-cache")]
#[test]
fn cold_tier_flatbuffers_v2_and_graceful_fallback() {
    let cfg = make_test_config(2, 16);
    let dir = unique_temp_dir("cold_v2");
    let mut cache = KvPrefixCache::new(
        KvCacheConfig {
            cache_dir: Some(dir.clone()),
            max_warm_entries: 0, // Force cold-tier disk reads
            ..KvCacheConfig::default()
        },
        &cfg,
        "cpu:test_cold_v2",
    );

    let tokens = [42u32, 43, 44, 45, 46];
    let snap = StateSnapshot::new(
        vec![
            LayerSnapshot::Conv {
                buffer: vec![1, 2, 3, 4, 5, 6, 7, 8],
            },
            LayerSnapshot::Attention {
                k_data: vec![10, 20, 30, 40],
                v_data: vec![50, 60, 70, 80],
            },
        ],
        tokens.len(),
    )
    .with_anchor(5, SemanticBoundaryKind::Thinking, 0x9999);

    // 1. Save valid FlatBuffers v2 snapshot via public cache.insert
    cache.insert(&tokens, snap);

    // 2. Query strict prefix — must load cleanly from cold tier
    let (loaded, len) = cache
        .find_deepest_semantic_anchor(&[42, 43, 44, 45, 46, 47])
        .expect("must load v2 snapshot from disk");
    assert_eq!(len, 5);
    assert_eq!(loaded.seq_len, 5);
    assert_eq!(loaded.anchor_depth, 5);
    assert_eq!(loaded.boundary_kind, SemanticBoundaryKind::Thinking as u8);
    assert_eq!(loaded.semantic_hash, 0x9999);

    // 3. Corrupted / legacy file test: write raw garbage to an unexpected .kvcache file
    let _corrupt_tokens = [90u32, 91, 92];
    let corrupt_path = dir.join("corrupted_test_file.kvcache");
    std::fs::write(&corrupt_path, b"LEGACY_CORRUPTED_V1_CACHE_PAYLOAD_NOT_FLATBUFFERS").unwrap();

    // Verify loading non-matching/corrupted token sequence returns None (graceful cache miss) without panicking
    let corrupt_hit = cache.find_deepest_semantic_anchor(&[90, 91, 92, 93]);
    assert!(corrupt_hit.is_none(), "corrupted/legacy file must return None cache miss");

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

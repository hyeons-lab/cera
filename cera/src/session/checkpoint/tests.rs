use super::*;
use crate::kv_cache::LayerSnapshot;

#[test]
fn session_checkpoint_roundtrip_all_layer_variants() {
    let layers = vec![
        LayerSnapshot::Attention {
            k_data: vec![1, 2, 3, 4],
            v_data: vec![5, 6, 7, 8],
        },
        LayerSnapshot::Conv {
            buffer: vec![9, 10, 11, 12],
        },
        LayerSnapshot::AttentionCompressed {
            keys: vec![13, 14],
            values: vec![15, 16],
        },
        LayerSnapshot::AttentionF16 {
            k_data: vec![17, 18],
            v_data: vec![19, 20],
        },
        LayerSnapshot::Mamba2 {
            conv_state: vec![21, 22, 23, 24],
            ssm_state: vec![25, 26, 27, 28],
        },
        LayerSnapshot::DeltaNet {
            conv_state: vec![29, 30, 31, 32],
            ssm_state: vec![33, 34, 35, 36],
        },
        LayerSnapshot::ParallelAttentionMamba2 {
            snap: Box::new(LayerSnapshot::Attention {
                k_data: vec![37, 38],
                v_data: vec![39, 40],
            }),
            conv_state: vec![41, 42],
            ssm_state: vec![43, 44],
        },
    ];

    let kv_state = StateSnapshot {
        layers,
        seq_len: 42,
        anchor_depth: 1,
        boundary_kind: 2,
        semantic_hash: 123456789,
        shift_offset: 0,
    };

    let checkpoint = SessionCheckpoint {
        model_fingerprint: 0xDEADBEEFCAFE,
        position: 42,
        max_seq_len: 2048,
        prefill_tokens: 42,
        prefill_elapsed_ms: 120,
        last_logits: Some(vec![1.5, -2.0, 3.25, 0.0]),
        token_history: vec![1, 100, 200, 300],
        kv_state,
    };

    let bytes = checkpoint.to_bytes();
    let decoded = SessionCheckpoint::from_bytes(&bytes).expect("decoding should succeed");
    assert_eq!(checkpoint, decoded);
}

#[test]
fn session_checkpoint_roundtrip_without_logits() {
    let kv_state = StateSnapshot::new(Vec::new(), 0);
    let checkpoint = SessionCheckpoint {
        model_fingerprint: 0x1234,
        position: 0,
        max_seq_len: 512,
        prefill_tokens: 0,
        prefill_elapsed_ms: 0,
        last_logits: None,
        token_history: Vec::new(),
        kv_state,
    };

    let bytes = checkpoint.to_bytes();
    let decoded = SessionCheckpoint::from_bytes(&bytes).expect("decoding should succeed");
    assert_eq!(checkpoint, decoded);
}

#[test]
fn session_checkpoint_magic_and_version_rejection() {
    let kv_state = StateSnapshot::new(Vec::new(), 0);
    let checkpoint = SessionCheckpoint {
        model_fingerprint: 0x1234,
        position: 0,
        max_seq_len: 512,
        prefill_tokens: 0,
        prefill_elapsed_ms: 0,
        last_logits: None,
        token_history: Vec::new(),
        kv_state,
    };

    let mut bytes = checkpoint.to_bytes();
    // Corrupt magic
    bytes[0] = b'X';
    assert!(SessionCheckpoint::from_bytes(&bytes).is_err());

    // Restore magic and corrupt version
    bytes[0] = b'C';
    bytes[8] = 99; // Version 99
    assert!(SessionCheckpoint::from_bytes(&bytes).is_err());
}

#[test]
fn chat_checkpoint_roundtrip() {
    let kv_state = StateSnapshot::new(Vec::new(), 10);
    let session_checkpoint = SessionCheckpoint {
        model_fingerprint: 0xAABBCCDDEEFF,
        position: 10,
        max_seq_len: 1024,
        prefill_tokens: 10,
        prefill_elapsed_ms: 50,
        last_logits: Some(vec![0.5, -0.5]),
        token_history: vec![1, 2, 3],
        kv_state,
    };

    let tool = ToolDef {
        name: "test_calc".into(),
        description: Some("calculator tool".into()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "val": { "type": "number" } }
        }),
    };

    let chat_checkpoint = ChatCheckpoint {
        session_checkpoint,
        phase: SessionPhase::TurnComplete,
        tool_format: ToolFormat::Hermes,
        tools: vec![tool],
    };

    let bytes = chat_checkpoint
        .to_bytes()
        .expect("serialization should succeed");
    let decoded = ChatCheckpoint::from_bytes(&bytes).expect("deserialization should succeed");
    assert_eq!(chat_checkpoint, decoded);
}

#[test]
fn chat_checkpoint_magic_rejection() {
    let kv_state = StateSnapshot::new(Vec::new(), 0);
    let session_checkpoint = SessionCheckpoint {
        model_fingerprint: 0x1122,
        position: 0,
        max_seq_len: 256,
        prefill_tokens: 0,
        prefill_elapsed_ms: 0,
        last_logits: None,
        token_history: Vec::new(),
        kv_state,
    };

    let chat_checkpoint = ChatCheckpoint {
        session_checkpoint,
        phase: SessionPhase::Idle,
        tool_format: ToolFormat::Lfm2Pythonic,
        tools: Vec::new(),
    };

    let mut bytes = chat_checkpoint.to_bytes().expect("to_bytes");
    bytes[0] = b'Z';
    assert!(ChatCheckpoint::from_bytes(&bytes).is_err());
}

#[test]
fn checkpoint_file_persistence() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let file_path = temp_dir.path().join("session.cchk");

    let kv_state = StateSnapshot::new(Vec::new(), 5);
    let checkpoint = SessionCheckpoint {
        model_fingerprint: 0x42,
        position: 5,
        max_seq_len: 512,
        prefill_tokens: 5,
        prefill_elapsed_ms: 15,
        last_logits: None,
        token_history: vec![1, 2],
        kv_state,
    };

    checkpoint.save_to_file(&file_path).expect("save_to_file");
    let loaded = SessionCheckpoint::load_from_file(&file_path).expect("load_from_file");
    assert_eq!(checkpoint, loaded);

    // Verify no temporary files remain in directory
    let entries: Vec<_> = std::fs::read_dir(temp_dir.path())
        .expect("read_dir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].file_name(), "session.cchk");
}

#[test]
fn checkpoint_trailing_bytes_rejection() {
    let kv_state = StateSnapshot::new(Vec::new(), 0);
    let session_checkpoint = SessionCheckpoint {
        model_fingerprint: 0x1122,
        position: 0,
        max_seq_len: 256,
        prefill_tokens: 0,
        prefill_elapsed_ms: 0,
        last_logits: None,
        token_history: Vec::new(),
        kv_state,
    };

    let mut session_bytes = session_checkpoint.to_bytes();
    session_bytes.extend_from_slice(b"extra_trailing_bytes");
    let err = SessionCheckpoint::from_bytes(&session_bytes).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(msg.contains("unexpected trailing bytes in session checkpoint"));
        }
        other => panic!("expected format error for trailing bytes, got {other:?}"),
    }

    let chat_checkpoint = ChatCheckpoint {
        session_checkpoint,
        phase: SessionPhase::Idle,
        tool_format: ToolFormat::Lfm2Pythonic,
        tools: Vec::new(),
    };
    let mut chat_bytes = chat_checkpoint.to_bytes().expect("to_bytes");
    chat_bytes.extend_from_slice(b"extra_junk");
    let err = ChatCheckpoint::from_bytes(&chat_bytes).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(msg.contains("unexpected trailing bytes in chat checkpoint"));
        }
        other => panic!("expected format error for trailing bytes, got {other:?}"),
    }
}

#[test]
fn checkpoint_reader_overflow_length_rejection() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"CERASCHK");
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&0x1234u64.to_le_bytes()); // model_fingerprint
    buf.extend_from_slice(&0u64.to_le_bytes()); // position
    buf.extend_from_slice(&256u64.to_le_bytes()); // max_seq_len
    buf.extend_from_slice(&0u32.to_le_bytes()); // prefill_tokens
    buf.extend_from_slice(&0u64.to_le_bytes()); // prefill_elapsed_ms
    buf.push(1); // has_logits = 1
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // huge logits count to test bounds check

    let err = SessionCheckpoint::from_bytes(&buf).unwrap_err();
    match err {
        CeraError::Format(msg) => {
            assert!(
                msg.contains("unexpected EOF reading byte buffer in checkpoint")
                    || msg.contains("overflow")
            );
        }
        other => panic!("expected format error for huge count EOF, got {other:?}"),
    }
}

//! Behavioral regressions for session and chat audit findings.
use cera::gguf::{GgufFile, GgufValue};
use cera::kv_cache::{InferenceState, LayerSnapshot, LayerState};
use cera::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use cera::session::chat::{Message, Profile, TEMPLATE};
use cera::session::{GenerateOpts, ModalityCapabilities, Session, SessionConfig};
use cera::tokenizer::BpeTokenizer;
use std::sync::Arc;

fn mla_model(kda_head_dim: u32) -> Vec<u8> {
    use cera::convert::writer::{GGML_TYPE_F32, GgufWriter};
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "bailingmoe3");
    writer.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
    for (key, value) in [
        ("block_count", 1),
        ("embedding_length", 4),
        ("feed_forward_length", 4),
        ("attention.head_count", 1),
        ("context_length", 32),
        ("kda.head_dim", kda_head_dim),
        ("attention.kv_lora_rank", 4),
        ("rope.dimension_count", 2),
        ("attention.key_length_mla", 4),
        ("attention.value_length_mla", 2),
    ] {
        writer.add_u32(format!("bailingmoe3.{key}"), value);
    }
    writer.add_i32_array("bailingmoe3.attention.head_count_kv", vec![1]);
    let tensors = [
        ("token_embd.weight", vec![4, 2]),
        ("output_norm.weight", vec![4]),
        ("blk.0.attn_norm.weight", vec![4]),
        ("blk.0.ffn_norm.weight", vec![4]),
        ("blk.0.attn_q.weight", vec![4, 4]),
        ("blk.0.attn_kv_a_mqa.weight", vec![4, 6]),
        ("blk.0.attn_kv_a_norm.weight", vec![4]),
        ("blk.0.attn_k_b.weight", vec![8]),
        ("blk.0.attn_v_b.weight", vec![8]),
        ("blk.0.attn_gate.weight", vec![4, 1]),
        ("blk.0.attn_output.weight", vec![2, 4]),
        ("blk.0.ffn_gate.weight", vec![4, 4]),
        ("blk.0.ffn_up.weight", vec![4, 4]),
        ("blk.0.ffn_down.weight", vec![4, 4]),
    ];
    for (name, shape) in &tensors {
        writer.add_tensor(
            *name,
            shape.clone(),
            GGML_TYPE_F32,
            shape.iter().product::<u64>() as usize * 4,
        );
    }
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    for (_, shape) in &tensors {
        writer
            .write_tensor_data(
                &mut bytes,
                &vec![0; shape.iter().product::<u64>() as usize * 4],
            )
            .unwrap();
    }
    bytes
}

#[test]
fn asymmetric_mla_checkpoint_round_trips_and_rejects_malformed_rows() {
    use cera::kv_cache::KvCompression;
    use cera::{BackendPreference, CeraEngine, EngineConfig};
    for kda_dim in [2, 8] {
        let engine = CeraEngine::from_bytes(
            mla_model(kda_dim),
            EngineConfig {
                backend: BackendPreference::Cpu,
                context_size: 32,
                ..Default::default()
            },
        )
        .unwrap();
        for compression in [KvCompression::None, KvCompression::F16] {
            let mut session = engine
                .new_session(SessionConfig {
                    kv_compression: compression,
                    ..Default::default()
                })
                .unwrap();
            session.append_tokens(&[0]).unwrap();
            let checkpoint = session.checkpoint().unwrap();
            session.append_tokens(&[1]).unwrap();
            let reference = session.checkpoint().unwrap();
            session.restore(&checkpoint).unwrap();
            session.append_tokens(&[1]).unwrap();
            let restored = session.checkpoint().unwrap();
            assert_eq!(restored.kv_state, reference.kv_state);
            assert_eq!(restored.last_logits, reference.last_logits);
            let mut malformed = checkpoint.clone();
            match &mut malformed.kv_state.layers[0] {
                LayerSnapshot::Attention { v_data, .. }
                | LayerSnapshot::AttentionF16 { v_data, .. } => {
                    v_data.clear();
                }
                _ => panic!("expected MLA attention cache"),
            }
            assert!(session.restore(&malformed).is_err());
            assert_eq!(session.checkpoint().unwrap().kv_state, reference.kv_state);
        }
    }
}

fn tokenizer(gemma: bool) -> Arc<BpeTokenizer> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.resize(32, 0);
    let mut gguf = GgufFile::from_bytes(bytes.into()).unwrap();
    let mut vocab: Vec<String> = vec![
        "<pad>",
        "<|startoftext|>",
        "<|image_start|>",
        "<|image_end|>",
        "<image>",
        "<|reserved_4|>",
        "<|im_start|>",
        "<|im_end|>",
        "Ġ",
        "Ċ",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    vocab.extend((33u8..=126).map(|b| (b as char).to_string()));
    vocab.extend(
        [
            "<0xC3>",
            "<0xA9>",
            "<bos>",
            "<eos>",
            "<start_of_turn>",
            "<end_of_turn>",
        ]
        .into_iter()
        .map(String::from),
    );
    let types = (0..vocab.len())
        .map(|i| {
            GgufValue::I32(if !(8..106).contains(&i) {
                3
            } else if i >= 104 {
                6
            } else {
                1
            })
        })
        .collect();
    gguf.metadata.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(vocab.into_iter().map(GgufValue::String).collect()),
    );
    gguf.metadata
        .insert("tokenizer.ggml.token_type".into(), GgufValue::Array(types));
    gguf.metadata.insert(
        "tokenizer.ggml.bos_token_id".into(),
        GgufValue::U32(if gemma { 106 } else { 1 }),
    );
    gguf.metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        GgufValue::U32(if gemma { 107 } else { 7 }),
    );
    let template = if gemma {
        "{{ bos_token }}{% for message in messages %}{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    } else {
        TEMPLATE
    };
    gguf.metadata.insert(
        "tokenizer.chat_template".into(),
        GgufValue::String(template.into()),
    );
    Arc::new(BpeTokenizer::from_gguf(&gguf).unwrap())
}
fn config(vocab: usize) -> ModelConfig {
    ModelConfig {
        architecture: "audit".into(),
        n_layers: 1,
        hidden_size: 2,
        intermediate_size: 2,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 2,
        vocab_size: vocab,
        max_seq_len: 1024,
        rope_theta: 10000.,
        rms_norm_eps: 1e-5,
        block_types: vec![BlockType::Attention],
        conv_kernel_size: None,
        ssm: None,
        kv_heads_per_layer: vec![1],
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}
struct Script {
    cfg: ModelConfig,
    first: u32,
    second: u32,
    eos: u32,
}
impl Model for Script {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        if let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = &mut state.layers[0]
        {
            assert_eq!(key_cache.len(), pos * 2, "KV rows disagree with position");
            key_cache.extend(tokens.iter().flat_map(|x| [*x as f32, *x as f32]));
            value_cache.extend(tokens.iter().flat_map(|x| [*x as f32, *x as f32]));
        }
        state.seq_len += tokens.len();
        let last = tokens.last().copied();
        let next = if last == Some(self.first) {
            self.second
        } else if last == Some(self.second) {
            self.eos
        } else {
            self.first
        };
        let mut logits = vec![-100.; self.cfg.vocab_size];
        logits[next as usize] = 100.;
        logits
    }
}
fn session(tok: Arc<BpeTokenizer>, first: u32, second: u32) -> Session {
    let model = Arc::new(Script {
        cfg: config(tok.vocab_size()),
        first,
        second,
        eos: tok.eos_token().unwrap(),
    });
    Session::new(
        model,
        tok,
        ModalityCapabilities::text_only(),
        SessionConfig::default(),
    )
    .unwrap()
}
fn opts() -> GenerateOpts {
    GenerateOpts {
        temperature: 0.,
        max_tokens: 4,
        flush_every_tokens: 1,
        flush_every_ms: 0,
        ..Default::default()
    }
}
use cera::kv_cache::{KvCompression, StateSnapshot};
use cera::model::{ModelSessionGate, ModelSessionLease};
use cera::session::CeraError;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
struct OwnedModel {
    cfg: ModelConfig,
    gate: ModelSessionGate,
    device: Mutex<InferenceState>,
    snapshots: AtomicUsize,
    restores: AtomicUsize,
}
impl Model for OwnedModel {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.gate.try_acquire().map(Some)
    }
    fn forward(&self, tokens: &[u32], _pos: usize, host: &mut InferenceState) -> Vec<f32> {
        let mut device = self.device.lock().unwrap();
        for _ in tokens {
            device.append_kv(0, &[1., 2.], &[3., 4.]);
            device.seq_len += 1;
        }
        host.seq_len += tokens.len();
        let mut logits = vec![0.; self.cfg.vocab_size];
        logits[20] = 1.;
        logits
    }
    fn snapshot_state(&self) -> StateSnapshot {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        self.device.lock().unwrap().snapshot().unwrap()
    }
    fn restore_state(&self, snapshot: &StateSnapshot) {
        self.restores.fetch_add(1, Ordering::Relaxed);
        self.device.lock().unwrap().restore(snapshot);
    }
}
struct CompressedProbe {
    cfg: ModelConfig,
    observed: Mutex<Vec<Vec<f32>>>,
}
impl Model for CompressedProbe {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn forward(&self, tokens: &[u32], _: usize, state: &mut InferenceState) -> Vec<f32> {
        use cera::turboquant::{
            attn_values_turboquant_gqa, compress_and_append_keys, compress_and_append_values,
        };
        let rot = state.tq_rotations[0].as_ref().unwrap();
        let cfg = state.tq_config.as_ref().unwrap();
        let scratch = state.tq_encode_scratch.as_mut().unwrap();
        let LayerState::Attention {
            compressed_keys: Some(keys),
            compressed_values: Some(values),
            ..
        } = &mut state.layers[0]
        else {
            panic!("expected compressed cache")
        };
        if state.seq_len > 0 {
            let mut scores = vec![0.; state.seq_len];
            scores[0] = 1.;
            let mut out = vec![0.; 32];
            attn_values_turboquant_gqa(
                values,
                0,
                0,
                1,
                &scores,
                &mut out,
                32,
                state.seq_len,
                rot,
                cfg,
            );
            self.observed.lock().unwrap().push(out);
        }
        for _ in tokens {
            let v = (0..32).map(|i| i as f32 / 32. + 0.1).collect::<Vec<_>>();
            compress_and_append_keys(&v, 1, 32, rot, cfg, keys, scratch);
            compress_and_append_values(&v, 1, 32, rot, cfg, values, scratch);
            state.seq_len += 1;
        }
        let mut logits = vec![0.; self.cfg.vocab_size];
        logits[20] = 1.;
        logits
    }
}

#[test]
fn streamed_utf8_matches_completed_text() {
    for (first, second, expected) in [(104, 105, "é"), (105, 104, "��")] {
        let mut chat = session(tokenizer(false), first, second)
            .into_chat()
            .unwrap();
        chat.ingest(&Message::user("hi")).unwrap();
        let mut text = String::new();
        let turn = chat.stream_text(&opts(), |s| text.push_str(s)).unwrap();
        assert_eq!(text, expected);
        assert_eq!(text, turn.text);
    }
}

#[test]
fn profile_terminal_stops_and_allows_continuation() {
    let tok = tokenizer(true);
    let profile = Profile::discover(tok.clone()).unwrap();
    assert_ne!(Some(profile.eos()), tok.eos_token());
    let mut chat = session(tok, 109, 107).into_chat().unwrap();
    chat.ingest(&Message::user("hi")).unwrap();
    chat.complete(&opts()).unwrap();
    assert_eq!(
        chat.phase(),
        cera::session::chat::SessionPhase::TurnComplete
    );
    chat.ingest(&Message::user("again")).unwrap();
}

#[test]
fn profile_terminal_finishes_structured_generation() {
    for ignore_eos in [false, true] {
        let tok = tokenizer(true);
        let number = tok.encode("1")[0];
        let mut chat = session(tok, number, 109).into_chat().unwrap();
        chat.ingest(&Message::user("hi")).unwrap();
        let options = GenerateOpts {
            ignore_eos,
            ..opts()
        };
        let result = chat.complete_json(&options, r#"{"const":1}"#).unwrap();
        assert_eq!(result.text, "1");
        assert_eq!(
            chat.phase(),
            cera::session::chat::SessionPhase::TurnComplete
        );
        chat.ingest(&Message::user("again")).unwrap();
    }
}

#[test]
fn ignore_eos_preserves_unconstrained_token_budget() {
    let mut chat = session(tokenizer(true), 109, 107).into_chat().unwrap();
    chat.ingest(&Message::user("hi")).unwrap();
    let options = GenerateOpts {
        ignore_eos: true,
        ..opts()
    };
    let turn = chat.complete(&options).unwrap();
    assert_eq!(turn.summary.tokens_generated, options.max_tokens);
    assert_eq!(
        turn.summary.finish_reason,
        cera::session::FinishReason::MaxTokens
    );
}

#[test]
fn malformed_checkpoint_rejected_without_mutation() {
    let mut session = session(tokenizer(false), 104, 105);
    session.append_tokens(&[10, 11]).unwrap();
    let good = session.checkpoint().unwrap();
    for delta in [-8isize, 4] {
        let mut bad = good.clone();
        let LayerSnapshot::Attention { k_data, .. } = &mut bad.kv_state.layers[0] else {
            unreachable!()
        };
        k_data.resize(k_data.len().checked_add_signed(delta).unwrap(), 0);
        assert!(session.restore(&bad).is_err());
        assert_eq!(session.checkpoint().unwrap(), good);
    }
    session.append_tokens(&[12]).unwrap();
    session.restore(&good).unwrap();
    assert_eq!(session.checkpoint().unwrap(), good);
    session.append_tokens(&[13]).unwrap();
    assert_eq!(session.position(), 3);
}

#[test]
fn backend_owned_checkpoint_rejected_without_mutation() {
    let tok = tokenizer(false);
    let cfg = config(tok.vocab_size());
    let device = InferenceState::from_config(&cfg).unwrap();
    let model = Arc::new(OwnedModel {
        cfg,
        gate: ModelSessionGate::default(),
        device: Mutex::new(device),
        snapshots: AtomicUsize::new(0),
        restores: AtomicUsize::new(0),
    });
    let mut owned = Session::new(
        model.clone(),
        tok.clone(),
        ModalityCapabilities::text_only(),
        SessionConfig::default(),
    )
    .unwrap();
    owned.append_tokens(&[10, 11]).unwrap();
    assert!(
        owned
            .checkpoint()
            .unwrap_err()
            .to_string()
            .contains("model-owned")
    );
    let cpu = session(tok, 104, 105);
    assert!(owned.restore(&cpu.checkpoint().unwrap()).is_err());
    assert_eq!(owned.position(), 2);
    assert_eq!(model.device.lock().unwrap().seq_len, 2);
    assert_eq!(model.restores.load(Ordering::Relaxed), 0);
}

#[test]
fn compressed_checkpoint_requires_same_seed_and_valid_geometry() {
    let tok = tokenizer(false);
    let mut cfg = config(tok.vocab_size());
    cfg.head_dim = 32;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 32;
    let probe = Arc::new(CompressedProbe {
        cfg,
        observed: Mutex::new(Vec::new()),
    });
    let make = |seed| {
        Session::new(
            probe.clone(),
            tok.clone(),
            ModalityCapabilities::text_only(),
            SessionConfig {
                kv_compression: KvCompression::turboquant(seed),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let mut a = make(1);
    a.append_tokens(&[10]).unwrap();
    let cp = a.checkpoint().unwrap();
    let mut b = make(2);
    let before = b.checkpoint().unwrap();
    assert!(b.restore(&cp).is_err());
    assert_eq!(b.checkpoint().unwrap(), before);
    let mut same = make(1);
    same.restore(&cp).unwrap();
    assert_eq!(same.checkpoint().unwrap(), cp);
    same.append_tokens(&[11]).unwrap();
    a.append_tokens(&[11]).unwrap();
    let observed = probe.observed.lock().unwrap();
    assert_eq!(observed.len(), 2);
    assert_eq!(observed[0], observed[1]);
    let mut bad = cp.clone();
    bad.position = 2;
    bad.kv_state.seq_len = 2;
    let before = same.checkpoint().unwrap();
    assert!(same.restore(&bad).is_err());
    assert_eq!(same.checkpoint().unwrap(), before);
}

#[test]
fn replacement_counts_only_new_prefill_tokens() {
    let mut chat = session(tokenizer(false), 104, 105).into_chat().unwrap();
    chat.ingest(&Message::user("a long prompt in the previous history"))
        .unwrap();
    chat.complete(&opts()).unwrap();
    let old = chat.position();
    let summary = chat.replace_messages(&[Message::user("x")]).unwrap();
    assert_eq!(summary.position_before, old);
    assert!(summary.position_after < old);
    assert_eq!(summary.input_tokens, summary.position_after);
    assert!(summary.input_tokens > 0);
}

#[test]
fn f16_snapshot_requires_exact_rows() {
    let cfg = config(32);
    let mut state =
        InferenceState::from_config_with_compression(&cfg, &KvCompression::F16).unwrap();
    state.append_kv_f16(0, &[1., 2.], &[3., 4.]);
    state.seq_len = 1;
    let mut snapshot = state.snapshot().unwrap();
    assert!(snapshot.validate_for_model(&cfg).is_ok());
    let LayerSnapshot::AttentionF16 { v_data, .. } = &mut snapshot.layers[0] else {
        unreachable!()
    };
    v_data.truncate(2);
    assert!(snapshot.validate_for_model(&cfg).is_err());
}

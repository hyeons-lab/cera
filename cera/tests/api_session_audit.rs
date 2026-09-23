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
/// Append two f32 rows per token to layer 0's KV caches and advance
/// `seq_len`: the shared "pretend decode" bookkeeping behind `Script` and
/// `FlatModel`, so session position/KV assertions stay meaningful.
fn track_kv(tokens: &[u32], state: &mut InferenceState) {
    if let LayerState::Attention {
        key_cache,
        value_cache,
        ..
    } = &mut state.layers[0]
    {
        key_cache.extend(tokens.iter().flat_map(|x| [*x as f32, *x as f32]));
        value_cache.extend(tokens.iter().flat_map(|x| [*x as f32, *x as f32]));
    }
    state.seq_len += tokens.len();
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
        if let LayerState::Attention { key_cache, .. } = &state.layers[0] {
            assert_eq!(key_cache.len(), pos * 2, "KV rows disagree with position");
        }
        track_kv(tokens, state);
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
fn second_session_on_gated_model_is_busy_until_first_drops() {
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
    let open = || {
        Session::new(
            model.clone(),
            tok.clone(),
            ModalityCapabilities::text_only(),
            SessionConfig::default(),
        )
    };
    let first = open().expect("first session acquires the gate");
    assert!(
        matches!(open(), Err(CeraError::Busy)),
        "a second live session on a gated model must fail fast"
    );
    drop(first);
    assert!(open().is_ok(), "dropping the session releases the gate");
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

/// Flat logits over the full vocab: every token is equally likely, so the
/// generated stream is pure sampler-RNG output. Uses [`track_kv`] like
/// `Script` so session position/KV assertions stay meaningful.
struct FlatModel {
    cfg: ModelConfig,
}
impl Model for FlatModel {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn forward(&self, tokens: &[u32], _pos: usize, state: &mut InferenceState) -> Vec<f32> {
        track_kv(tokens, state);
        vec![0.; self.cfg.vocab_size]
    }
}

struct Collect(Vec<u32>);
impl cera::ModalitySink for Collect {
    fn on_text_tokens(&mut self, t: &[u32]) {
        self.0.extend_from_slice(t);
    }
    fn on_done(&mut self, _r: cera::FinishReason) {}
}

fn stochastic_opts(seed: Option<u64>) -> GenerateOpts {
    GenerateOpts {
        max_tokens: 8,
        seed,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        // Flat logits can sample EOS early; ignore it so every run emits a
        // full 8-token stream and seeded runs are comparable token-for-token.
        ignore_eos: true,
        ..Default::default()
    }
}

fn greedy_opts(seed: Option<u64>) -> GenerateOpts {
    GenerateOpts {
        max_tokens: 2,
        seed,
        temperature: 0.0,
        ignore_eos: true,
        ..Default::default()
    }
}

fn flat_session(tok: Arc<BpeTokenizer>, seed: Option<u64>) -> Session {
    let model = Arc::new(FlatModel {
        cfg: config(tok.vocab_size()),
    });
    Session::new(
        model,
        tok,
        ModalityCapabilities::text_only(),
        SessionConfig {
            seed,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn per_request_seed_reproduces_and_leaves_kv_undisturbed() {
    let tok = tokenizer(false);
    let mut a = flat_session(tok.clone(), None);
    a.append_tokens(&[0, 1]).unwrap();
    let pos_before = a.position();
    let logits_before = a.last_logits().unwrap().to_vec();

    // `set_seed` restarts only the RNG: position and logits are untouched.
    a.set_seed(Some(5));
    assert_eq!(a.position(), pos_before);
    assert_eq!(a.last_logits().unwrap(), logits_before.as_slice());

    let mut sink = Collect(Vec::new());
    a.generate(&stochastic_opts(Some(99)), &mut sink).unwrap();
    assert_eq!(sink.0.len(), 8);

    // Same prompt + same per-request seed on a fresh session: identical stream.
    let mut b = flat_session(tok.clone(), None);
    b.append_tokens(&[0, 1]).unwrap();
    let mut sink_b = Collect(Vec::new());
    b.generate(&stochastic_opts(Some(99)), &mut sink_b).unwrap();
    assert_eq!(sink_b.0, sink.0);

    // A different per-request seed diverges (110^8 stream space).
    let mut sink_c = Collect(Vec::new());
    b.generate(&stochastic_opts(Some(100)), &mut sink_c)
        .unwrap();
    assert_ne!(sink_c.0, sink.0);
}

#[test]
fn per_request_seed_does_not_clobber_the_session_default() {
    let tok = tokenizer(false);
    // Session default seed 3; one call overrides it per-request.
    let mut a = flat_session(tok.clone(), Some(3));
    a.append_tokens(&[0, 1]).unwrap();
    let mut sink = Collect(Vec::new());
    a.generate(&stochastic_opts(Some(99)), &mut sink).unwrap();
    // After reset the sampler rebuilds from the session default (3), not the
    // per-request override (99): identical to a fresh session on seed 3.
    a.reset().unwrap();
    a.append_tokens(&[0, 1]).unwrap();
    let mut after = Collect(Vec::new());
    a.generate(&stochastic_opts(None), &mut after).unwrap();

    let mut b = flat_session(tok.clone(), Some(3));
    b.append_tokens(&[0, 1]).unwrap();
    let mut expected = Collect(Vec::new());
    b.generate(&stochastic_opts(None), &mut expected).unwrap();
    assert_eq!(after.0, expected.0);
}

#[test]
fn greedy_call_honors_per_request_seed_for_later_calls() {
    // Greedy decode samples nothing, but its seed still selects the stream
    // the next stochastic call continues. Both runs share identical
    // context; only the final stochastic call differs (continuation vs
    // explicit reseed), so flat logits make the streams exactly equal.
    // (The append between generations follows the standard loop: greedy
    // clears `last_logits`, so chaining needs a re-prime first.)
    let run = |seed: Option<u64>| {
        let tok = tokenizer(false);
        let mut s = flat_session(tok, None);
        s.append_tokens(&[0, 1]).unwrap();
        let mut greedy_sink = Collect(Vec::new());
        s.generate(&greedy_opts(Some(99)), &mut greedy_sink)
            .unwrap();
        s.append_tokens(&[0]).unwrap();
        let mut sink = Collect(Vec::new());
        s.generate(&stochastic_opts(seed), &mut sink).unwrap();
        sink.0
    };
    assert_eq!(run(None), run(Some(99)));
    assert_ne!(run(None), run(Some(100)));
}

#[test]
fn greedy_draws_nothing_from_the_rng_stream() {
    // Greedy decode must not consume randomness: the same reseed followed
    // by a stochastic continuation yields byte-identical streams whether
    // or not a greedy call runs in between. Flat logits make the
    // continuation pure RNG output, so any draw inside the greedy path
    // would shift it. (Position differs between the runs; FlatModel
    // ignores position.)
    let run = |with_greedy: bool| {
        let tok = tokenizer(false);
        let mut s = flat_session(tok, None);
        s.append_tokens(&[0, 1]).unwrap();
        s.set_seed(Some(7));
        if with_greedy {
            let mut greedy_sink = Collect(Vec::new());
            s.generate(&greedy_opts(None), &mut greedy_sink).unwrap();
            // Greedy clears `last_logits`, so chaining needs a re-prime.
            s.append_tokens(&[0]).unwrap();
        }
        let mut sink = Collect(Vec::new());
        s.generate(&stochastic_opts(None), &mut sink).unwrap();
        sink.0
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn set_seed_persists_across_reset() {
    // `set_seed` replaces the session default (unlike a per-request seed),
    // so after `reset` the sampler rebuilds from it: identical to a fresh
    // session constructed with that seed.
    let tok = tokenizer(false);
    let mut a = flat_session(tok.clone(), Some(3));
    a.append_tokens(&[0, 1]).unwrap();
    a.set_seed(Some(7));
    // Burn RNG draws before reset: if reset kept the live stream
    // instead of rebuilding from the session default, output would diverge.
    let mut burn = Collect(Vec::new());
    a.generate(&stochastic_opts(None), &mut burn).unwrap();
    a.reset().unwrap();
    a.append_tokens(&[0, 1]).unwrap();
    let mut after = Collect(Vec::new());
    a.generate(&stochastic_opts(None), &mut after).unwrap();

    let mut b = flat_session(tok.clone(), Some(7));
    b.append_tokens(&[0, 1]).unwrap();
    let mut expect = Collect(Vec::new());
    b.generate(&stochastic_opts(None), &mut expect).unwrap();
    assert_eq!(after.0, expect.0);

    // Non-vacuity: a different default seed diverges.
    let mut c = flat_session(tok, Some(3));
    c.append_tokens(&[0, 1]).unwrap();
    let mut other = Collect(Vec::new());
    c.generate(&stochastic_opts(None), &mut other).unwrap();
    assert_ne!(other.0, expect.0);
}

use cera::lora::{LoraAdapterWeights, LoraTarget};
use cera::model::MoeConfig;

/// LoRA-capable mock: flat logits (pure sampler-RNG streams, like
/// `FlatModel`) plus hidden-states extraction that stages `state.lora`, so
/// install/override tests can tell base output from adapted output, and one
/// adapter from another via its AttnQ B-factor checksum.
struct LoraProbe {
    cfg: ModelConfig,
}
impl Model for LoraProbe {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn forward(&self, tokens: &[u32], _pos: usize, state: &mut InferenceState) -> Vec<f32> {
        track_kv(tokens, state);
        vec![0.; self.cfg.vocab_size]
    }
    fn supports_lora(&self) -> bool {
        true
    }
    fn supports_hidden_states(&self) -> bool {
        true
    }
    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        let bump = state
            .lora
            .as_ref()
            .and_then(|a| a.get(0, LoraTarget::AttnQ))
            .map(|t| t.b.iter().sum::<f32>() * t.scale)
            .unwrap_or(0.0);
        // `config().hidden_size` is 2: two channels per token.
        tokens
            .iter()
            .flat_map(|&t| [t as f32 + bump, t as f32 - bump])
            .collect()
    }
}

fn lora_session(tok: Arc<BpeTokenizer>, seed: Option<u64>) -> Session {
    lora_session_with_config(tok, seed, config)
}

fn lora_session_with_config(
    tok: Arc<BpeTokenizer>,
    seed: Option<u64>,
    cfg: fn(usize) -> ModelConfig,
) -> Session {
    let model = Arc::new(LoraProbe {
        cfg: cfg(tok.vocab_size()),
    });
    Session::new(
        model,
        tok,
        ModalityCapabilities::text_only(),
        SessionConfig {
            seed,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Minimal PEFT adapter buffer: one rank-1 `q_proj` pair on layer 0 with
/// the given `(k, d)` and B fill. Loads through the real
/// `from_safetensors_bytes` path, so install tests exercise what users ship.
fn q_adapter_bytes(k: usize, d: usize, b_fill: f32) -> Vec<u8> {
    let a_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
    let b_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
    let a_len = k * 4;
    let b_len = d * 4;
    let header = serde_json::json!({
        a_name: { "dtype": "F32", "shape": [1, k], "data_offsets": [0, a_len] },
        b_name: { "dtype": "F32", "shape": [d, 1], "data_offsets": [a_len, a_len + b_len] },
    });
    let hs = serde_json::to_vec(&header).unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(hs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&hs);
    for _ in 0..k {
        buf.extend_from_slice(&0.5f32.to_le_bytes());
    }
    for _ in 0..d {
        buf.extend_from_slice(&b_fill.to_le_bytes());
    }
    buf
}

fn load_q_adapter(k: usize, d: usize, b_fill: f32) -> Arc<LoraAdapterWeights> {
    LoraAdapterWeights::from_safetensors_bytes(&q_adapter_bytes(k, d, b_fill), None).unwrap()
}

/// GGUF adapter with one stacked per-expert `ffn_gate_exps` pair on layer 0
/// (rank 1, `n_expert` slices). PEFT safetensors cannot carry expert deltas,
/// so the routed-FFN refusal test goes through `from_gguf_bytes`.
fn moe_adapter_bytes(k: usize, d: usize, n_expert: usize) -> Vec<u8> {
    use cera::convert::writer::{GGML_TYPE_F32, GgufWriter};
    let mut writer = GgufWriter::new();
    // GGUF `ne` is fastest-varying first: `[k, rank, n_slices]`.
    writer.add_tensor(
        "blk.0.ffn_gate_exps.weight.lora_a",
        vec![k as u64, 1, n_expert as u64],
        GGML_TYPE_F32,
        k * n_expert * 4,
    );
    writer.add_tensor(
        "blk.0.ffn_gate_exps.weight.lora_b",
        vec![1, d as u64, n_expert as u64],
        GGML_TYPE_F32,
        d * n_expert * 4,
    );
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    writer
        .write_tensor_data(&mut bytes, &vec![0u8; k * n_expert * 4])
        .unwrap();
    writer
        .write_tensor_data(&mut bytes, &vec![0u8; d * n_expert * 4])
        .unwrap();
    bytes
}

fn moe_probe_config(vocab: usize) -> ModelConfig {
    let mut cfg = config(vocab);
    cfg.moe = Some(MoeConfig {
        n_expert: 2,
        n_expert_used: 1,
        expert_ff_len: 3,
        is_moe_layer: vec![true],
    });
    cfg
}

#[test]
fn set_lora_adapters_empty_list_detaches_and_zero_scale_installs_noop() {
    // The audit config wants AttnQ (k=2, d=2).
    let adapter = load_q_adapter(2, 2, 1.0);
    let mut s = lora_session(tokenizer(false), None);
    let tokens = [0u32, 1];
    let base = s.hidden_states_for_tokens(&tokens).unwrap();

    s.set_lora_adapters(&[(adapter.clone(), 1.0)]).unwrap();
    assert!(s.has_lora_adapters());
    let adapted = s.hidden_states_for_tokens(&tokens).unwrap();
    assert_ne!(adapted, base);

    // An empty list detaches.
    s.set_lora_adapters(&[]).unwrap();
    assert!(!s.has_lora_adapters());
    assert_eq!(s.hidden_states_for_tokens(&tokens).unwrap(), base);

    // An all-zero-scale stack installs a no-op instead: attached (so
    // `has_lora_adapters` stays true) but applying nothing.
    s.set_lora_adapters(&[(adapter, 0.0)]).unwrap();
    assert!(s.has_lora_adapters());
    assert_eq!(s.hidden_states_for_tokens(&tokens).unwrap(), base);
}

#[test]
fn set_lora_adapters_is_atomic() {
    let good = load_q_adapter(2, 2, 1.0);
    // k=3 fits no projection of the audit config (hidden 2).
    let bad = load_q_adapter(3, 2, 1.0);
    let mut s = lora_session(tokenizer(false), None);
    s.set_lora_adapters(&[(good, 1.0)]).unwrap();
    let tokens = [0u32, 1];
    let before = s.hidden_states_for_tokens(&tokens).unwrap();

    let err = s.set_lora_adapters(&[(bad, 1.0)]).unwrap_err();
    assert!(matches!(err, CeraError::LoraDimMismatch(_)), "{err:?}");

    // The previous set is untouched: still attached, and byte-identical
    // output (identity, not just attached-ness).
    assert!(s.has_lora_adapters());
    assert_eq!(s.hidden_states_for_tokens(&tokens).unwrap(), before);
}

#[test]
fn set_lora_adapters_refuses_backends_without_hooks() {
    let adapter = load_q_adapter(2, 2, 1.0);
    // `FlatModel` inherits `supports_lora() == false`.
    let mut s = flat_session(tokenizer(false), None);
    let err = s.set_lora_adapters(&[(adapter.clone(), 1.0)]).unwrap_err();
    assert!(
        matches!(err, CeraError::LoraUnsupportedByBackend(_)),
        "{err:?}"
    );
    assert!(!s.has_lora_adapters());
    // `attach_lora_adapters` shares the guard.
    let err = s.attach_lora_adapters(adapter).unwrap_err();
    assert!(
        matches!(err, CeraError::LoraUnsupportedByBackend(_)),
        "{err:?}"
    );
    assert!(!s.has_lora_adapters());
}

#[test]
fn set_lora_adapters_refuses_routed_ffn_deltas_without_moe_hooks() {
    // `LoraProbe` opens `supports_lora` but inherits
    // `supports_moe_lora() == false`: a fitting expert adapter (layer 0 is
    // routed, k=hidden 2, d=expert_ff_len 3, 2 experts) must be refused as
    // a whole rather than silently half-applied.
    let bytes = moe_adapter_bytes(2, 3, 2);
    let adapter = LoraAdapterWeights::from_gguf_bytes(bytes.into()).expect("expert adapter loads");
    assert!(adapter.has_moe_deltas());
    let mut s = lora_session_with_config(tokenizer(false), None, moe_probe_config);
    let err = s.set_lora_adapters(&[(adapter, 1.0)]).unwrap_err();
    match err {
        CeraError::LoraUnsupportedByBackend(detail) => {
            assert!(detail.contains("routed-FFN"), "{detail}");
        }
        other => panic!("expected MoE refusal, got: {other:?}"),
    }
    assert!(!s.has_lora_adapters());
}

#[test]
fn hidden_states_using_some_extracts_through_the_override() {
    let adapter = load_q_adapter(2, 2, 1.0);
    let other = load_q_adapter(2, 2, 2.0);
    let mut s = lora_session(tokenizer(false), None);
    let tokens = [0u32, 1];
    let base = s.hidden_states_for_tokens(&tokens).unwrap();

    // Per-call override with nothing attached: adapted output, and the
    // session set stays empty afterwards.
    let over = s
        .hidden_states_for_tokens_using(&tokens, Some(&adapter))
        .unwrap();
    assert_ne!(over, base);
    assert!(!s.has_lora_adapters());
    assert_eq!(s.hidden_states_for_tokens(&tokens).unwrap(), base);

    // With a set attached, the override wins for that call only, and the
    // attached set still applies afterwards (distinct B fills keep the two
    // adapters' outputs apart).
    s.set_lora_adapters(&[(adapter, 1.0)]).unwrap();
    let attached_out = s.hidden_states_for_tokens(&tokens).unwrap();
    let over2 = s
        .hidden_states_for_tokens_using(&tokens, Some(&other))
        .unwrap();
    assert_ne!(over2, attached_out);
    assert!(s.has_lora_adapters());
    assert_eq!(s.hidden_states_for_tokens(&tokens).unwrap(), attached_out);

    // `None` extracts from the base model despite the attached set.
    let bare = s.hidden_states_for_tokens_using(&tokens, None).unwrap();
    assert_eq!(bare, base);
    assert!(s.has_lora_adapters());
}

#[test]
fn hidden_states_using_validates_the_bypass() {
    // Overrides bypass installation but must not bypass its guards: a
    // dim-mismatched adapter is refused rather than mis-zipped into the
    // apply hooks.
    let bad = load_q_adapter(3, 2, 1.0);
    let mut s = lora_session(tokenizer(false), None);
    let err = s
        .hidden_states_for_tokens_using(&[0u32, 1], Some(&bad))
        .unwrap_err();
    assert!(matches!(err, CeraError::LoraDimMismatch(_)), "{err:?}");
    assert!(!s.has_lora_adapters());
}

#[test]
fn hidden_states_pooled_and_text_using_cover_the_wrappers() {
    // The pooled/text `_using` wrappers forward the adapter choice; one
    // adapted call each pins the forwarding (the core swap/restore is
    // covered above).
    let adapter = load_q_adapter(2, 2, 1.0);
    let mut s = lora_session(tokenizer(false), None);
    let tokens = [0u32, 1];
    let base_pooled = s.hidden_states_mean_pooled(&tokens).unwrap();
    let over_pooled = s
        .hidden_states_mean_pooled_using(&tokens, Some(&adapter))
        .unwrap();
    assert_ne!(over_pooled, base_pooled);

    let base_text = s.hidden_states_for_text("ab").unwrap();
    let over_text = s
        .hidden_states_for_text_using("ab", Some(&adapter))
        .unwrap();
    assert_ne!(over_text, base_text);
    assert!(!s.has_lora_adapters());
}

#[test]
fn failed_empty_call_does_not_reseed_rng() {
    // A seeded `generate` with no primed prompt fails with `EmptyInput`
    // before the reseed point, so the RNG stream must be identical to a
    // control session that never made the failed call.
    let run = |probe: bool| {
        let tok = tokenizer(false);
        let mut s = flat_session(tok, Some(5));
        if probe {
            let mut sink = Collect(Vec::new());
            let err = s.generate(&stochastic_opts(Some(99)), &mut sink);
            assert!(matches!(err, Err(CeraError::EmptyInput)));
        }
        s.append_tokens(&[0, 1]).unwrap();
        let mut sink = Collect(Vec::new());
        s.generate(&stochastic_opts(None), &mut sink).unwrap();
        sink.0
    };
    assert_eq!(run(true), run(false));
}

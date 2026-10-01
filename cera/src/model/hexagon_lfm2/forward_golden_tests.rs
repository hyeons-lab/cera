//! Host tests that drive the real decode and prefill forward passes through a
//! queue session on the fake FastRPC driver and pin the op sequence they emit.
//!
//! The models are built by hand (no GGUF plan) from tiny configs: dense
//! attention, gated short-conv, Qwen 3.5 DeltaNet, and two "extras" variants
//! that switch on the dense-path semantics (biases, residual multiplier,
//! sliding window, YaRN, ...). The recorded text of every flushed batch is
//! hashed and compared with a constant, so a refactor of the forward code that
//! changes even one op, operand or kernel parameter fails here. Numerics are
//! not checked (all weights are zero); on-device parity is separate.

use super::*;
use crate::backend::hexagon::op_capture;
use crate::backend::hexagon::sys::fake;
use crate::model::{ModelConfig, ScalarMultipliers, SsmConfig};
use crate::tensor::DType;

const HS: usize = 64;
const N_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const KV_HEADS: usize = 1;
const INTER: usize = 128;
const VOCAB: usize = 64;
const MAX_SEQ: usize = 64;
const PREFILL_M: usize = 32;
// DeltaNet dims: conv_dim = 2 * n_group * d_state + dt_rank * d_state.
const DN_GROUP: usize = 2;
const DN_STATE: usize = 16;
const DN_RANK: usize = 32;
const DN_CONV: usize = 2 * DN_GROUP * DN_STATE + DN_RANK * DN_STATE;
const DN_VALUE: usize = DN_RANK * DN_STATE;
const DN_D_CONV: usize = 4;

/// Which optional dense-path semantics a test layer carries.
#[derive(Clone, Copy, Default)]
struct Extras {
    biases: bool,
    qk_norm: bool,
    qk_norm_full: bool,
    swa: bool,
    yarn: bool,
    q_gate: bool,
}

struct Plan {
    weights: usize,
    kv: usize,
}

impl Plan {
    fn weight(&mut self, in_dim: usize, out_dim: usize) -> HexagonWeight {
        let (size, wire_dtype, block_bytes, tile_size) =
            wire_plan(DType::Q4_0, in_dim, out_dim, "w").unwrap();
        HexagonWeight {
            offset: plan_offset(&mut self.weights, size),
            in_dim,
            out_dim,
            wire_dtype,
            block_bytes,
            tile_size,
        }
    }

    fn vec(&mut self, len: usize) -> usize {
        plan_offset(&mut self.weights, len * 4)
    }

    fn kv(&mut self, bytes: usize) -> usize {
        plan_offset(&mut self.kv, bytes)
    }

    fn ffn(&mut self, extras: Extras) -> HexagonFfn {
        HexagonFfn::Dense(HexagonDenseFfn {
            gate: self.weight(HS, INTER),
            up: self.weight(HS, INTER),
            down: self.weight(INTER, HS),
            gate_bias: extras.biases.then(|| self.vec(INTER)),
            up_bias: extras.biases.then(|| self.vec(INTER)),
            down_bias: extras.biases.then(|| self.vec(HS)),
        })
    }

    fn attention(&mut self, extras: Extras) -> HexagonLayer {
        let q_dim = N_HEADS * HEAD_DIM;
        let kv_dim = KV_HEADS * HEAD_DIM;
        let attn_norm_offset = self.vec(HS);
        let attn_q = self.weight(HS, if extras.q_gate { 2 * q_dim } else { q_dim });
        let attn_k = self.weight(HS, kv_dim);
        let attn_v = self.weight(HS, kv_dim);
        let attn_output = self.weight(q_dim, HS);
        let (qn, kn) = if extras.qk_norm {
            let (qlen, klen) = if extras.qk_norm_full {
                (q_dim, kv_dim)
            } else {
                (HEAD_DIM, HEAD_DIM)
            };
            (Some(self.vec(qlen)), Some(self.vec(klen)))
        } else {
            (None, None)
        };
        let qkv_bias = extras
            .biases
            .then(|| [self.vec(q_dim), self.vec(kv_dim), self.vec(kv_dim)]);
        let out_bias = extras.biases.then(|| self.vec(HS));
        let ffn_norm_offset = self.vec(HS);
        let ffn = self.ffn(extras);
        let slab = (MAX_SEQ * kv_dim * 2 + 255) & !255;
        let k_offset = self.kv(slab);
        let v_offset = self.kv(slab);
        HexagonLayer::Attention(HexagonAttentionLayer {
            attn_norm_offset,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm_offset: qn,
            attn_k_norm_offset: kn,
            attn_post_norm_offset: None,
            ffn_norm_offset,
            ffn,
            ffn_post_norm_offset: None,
            k_offset,
            v_offset,
            q_dim,
            kv_dim,
            has_q_gate: extras.q_gate,
            qkv_bias,
            out_bias,
            qk_norm_full: extras.qk_norm_full,
            swa: extras.swa,
            yarn: extras.yarn.then(|| {
                crate::backend::cpu::YarnParams::new_with_log_mul(
                    0.25, 1.0, 1.0, 32.0, 1.0, 16, 0.1,
                )
            }),
        })
    }

    fn conv(&mut self) -> HexagonLayer {
        let attn_norm_offset = self.vec(HS);
        let in_proj = self.weight(HS, 3 * HS);
        let out_proj = self.weight(HS, HS);
        let conv_w0_offset = self.vec(HS);
        let conv_w1_offset = self.vec(HS);
        let conv_w2_offset = self.vec(HS);
        let conv_ssm_offset = self.vec(3 * HS);
        let ffn_norm_offset = self.vec(HS);
        let ffn = self.ffn(Extras::default());
        let state_offset = self.kv(2 * ((HS * 4 + 255) & !255));
        HexagonLayer::Conv(HexagonConvLayer {
            attn_norm_offset,
            in_proj,
            out_proj,
            conv_w0_offset,
            conv_w1_offset,
            conv_w2_offset,
            conv_ssm_offset,
            state_offset,
            ffn_norm_offset,
            ffn,
        })
    }

    fn deltanet(&mut self) -> HexagonLayer {
        let attn_norm_offset = self.vec(HS);
        let wqkv = self.weight(HS, DN_CONV);
        let wqkv_gate = self.weight(HS, DN_VALUE);
        let ssm_beta = self.weight(HS, DN_RANK);
        let ssm_alpha = self.weight(HS, DN_RANK);
        let ssm_conv1d_offset = self.vec(DN_CONV * DN_D_CONV);
        let ssm_dt_offset = self.vec(DN_RANK);
        let ssm_a_offset = self.vec(DN_RANK);
        let ssm_norm_offset = self.vec(DN_STATE);
        let ssm_out = self.weight(DN_VALUE, HS);
        let ffn_norm_offset = self.vec(HS);
        let ffn = self.ffn(Extras::default());
        let conv_state_offset = self.kv(DN_CONV * (DN_D_CONV - 1) * 4);
        let ssm_state_offset = self.kv(DN_RANK * DN_STATE * DN_STATE * 4);
        HexagonLayer::DeltaNet(HexagonDeltaNetLayer {
            attn_norm_offset,
            wqkv,
            wqkv_gate,
            ssm_beta,
            ssm_alpha,
            ssm_conv1d_offset,
            ssm_conv1d_bias_offset: None,
            ssm_dt_offset,
            ssm_a_offset,
            ssm_norm_offset,
            ssm_out,
            attn_post_norm_offset: None,
            ffn_norm_offset,
            ffn,
            ffn_post_norm_offset: None,
            conv_state_offset,
            ssm_state_offset,
            conv_dim: DN_CONV,
            d_conv: DN_D_CONV,
            d_state: DN_STATE,
            dt_rank: DN_RANK,
            n_group: DN_GROUP,
        })
    }
}

fn config(block_types: Vec<BlockType>, deltanet: bool) -> ModelConfig {
    let n_layers = block_types.len();
    ModelConfig {
        architecture: "test".into(),
        n_layers,
        hidden_size: HS,
        intermediate_size: INTER,
        n_heads: N_HEADS,
        n_kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        vocab_size: VOCAB,
        max_seq_len: MAX_SEQ,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types,
        conv_kernel_size: Some(3),
        ssm: deltanet.then_some(SsmConfig {
            d_conv: DN_D_CONV,
            d_inner: DN_VALUE,
            d_state: DN_STATE,
            dt_rank: DN_RANK,
            n_group: DN_GROUP,
        }),
        kv_heads_per_layer: vec![KV_HEADS; n_layers],
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}

/// Everything a test model needs beyond its layers.
struct Spec {
    layers: Vec<HexagonLayer>,
    plan: Plan,
    config: ModelConfig,
    dense: DenseSemantics,
    cpu_rope: bool,
    has_deltanet: bool,
    with_swa_mask: bool,
    final_logit_softcapping: Option<f32>,
}

fn embd_table() -> EmbeddingTable {
    let data: Vec<u8> = (0..HS * VOCAB)
        .flat_map(|i| (0.001 * (i % 97) as f32).to_le_bytes())
        .collect();
    let gguf = super::test_support::embd_gguf(0, HS, VOCAB, &data);
    EmbeddingTable::new(&gguf, VOCAB, HS, 1.0).unwrap()
}

fn build(mut spec: Spec) -> HexagonLfmModel {
    let (driver, device) = op_capture::fresh_device();
    let q_dim = N_HEADS * HEAD_DIM;
    let mut so = ScratchOffsets::new(
        HS,
        2 * q_dim,
        KV_HEADS * HEAD_DIM,
        INTER,
        VOCAB,
        MAX_SEQ,
        None,
        spec.has_deltanet.then_some(DN_CONV),
    );
    if spec.with_swa_mask {
        so = so.with_swa_mask(MAX_SEQ);
    }
    let alloc = |size: usize| RpcmemBuffer::alloc(Arc::clone(&driver), size.max(256), true);
    let output_norm_offset = spec.plan.vec(HS);
    let lm_head = spec.plan.weight(HS, VOCAB);
    let mask_size = (MAX_SEQ * 2).max(128);
    if spec.with_swa_mask {
        spec.dense.mask_swa = Some(alloc(mask_size).unwrap());
    }
    // The same assembly path the constructors use, with every knob at its
    // default, so new knob-derived fields are covered and exported
    // `CERA_HEXAGON_*` variables cannot change the goldens.
    let knobs = HexagonKnobs::from_lookup(|_| None);
    HexagonLfmModel::from_parts(
        device,
        &knobs,
        Mutex::new(None),
        ModelParts {
            config: spec.config,
            token_embd: embd_table(),
            weights_buf: alloc(spec.plan.weights).unwrap(),
            layers: spec.layers,
            output_norm_offset,
            lm_head,
            kv_state_buf: alloc(spec.plan.kv).unwrap(),
            scratch_buf: alloc(so.total_size).unwrap(),
            scratch_offsets: so,
            mask_buf: alloc(mask_size).unwrap(),
            rope_type: RopeType::Neox,
            cpu_rope: spec.cpu_rope,
            dense: spec.dense,
            activation: FfnActivation::Swiglu,
            attn_logit_softcapping: None,
            final_logit_softcapping: spec.final_logit_softcapping,
            has_deltanet: spec.has_deltanet,
            kv_dtype: HtpDataType::F16,
        },
    )
}

fn spec(layers: impl FnOnce(&mut Plan) -> Vec<HexagonLayer>, deltanet: bool) -> Spec {
    let mut plan = Plan { weights: 0, kv: 0 };
    let layers = layers(&mut plan);
    let block_types = layers
        .iter()
        .map(|l| match l {
            HexagonLayer::Attention(_) => BlockType::Attention,
            _ => BlockType::GatedConv,
        })
        .collect();
    Spec {
        layers,
        plan,
        config: config(block_types, deltanet),
        dense: DenseSemantics::default(),
        cpu_rope: false,
        has_deltanet: deltanet,
        with_swa_mask: false,
        final_logit_softcapping: None,
    }
}

fn dense_spec() -> Spec {
    spec(
        |p| {
            vec![
                p.attention(Extras::default()),
                p.attention(Extras {
                    qk_norm: true,
                    ..Default::default()
                }),
            ]
        },
        false,
    )
}

fn conv_spec() -> Spec {
    spec(|p| vec![p.conv(), p.attention(Extras::default())], false)
}

fn deltanet_spec() -> Spec {
    spec(
        |p| {
            vec![
                p.deltanet(),
                p.attention(Extras {
                    qk_norm: true,
                    q_gate: true,
                    ..Default::default()
                }),
            ]
        },
        true,
    )
}

/// Biases, residual multiplier, sliding window, full-vector QK norm, loop
/// norm, logit scale and attention scale, all on the DSP route.
fn extras_dsp_spec() -> Spec {
    let extras = Extras {
        biases: true,
        qk_norm: true,
        qk_norm_full: true,
        swa: true,
        ..Default::default()
    };
    let mut s = spec(|p| vec![p.attention(extras), p.attention(extras)], false);
    let residual = s.plan.vec(HS);
    s.dense = DenseSemantics {
        attn_scale: Some(0.2),
        residual_vec_offset: Some(residual),
        logit_scale: Some(0.5),
        loop_norm_interval: Some(1),
        swa_window: Some(8),
        ..Default::default()
    };
    s.with_swa_mask = true;
    s.final_logit_softcapping = Some(30.0);
    s
}

/// Host RoPE (YaRN + Llama-3 factors), attention temperature and the
/// post-norm ordering.
fn extras_host_spec() -> Spec {
    let extras = Extras {
        yarn: true,
        ..Default::default()
    };
    let mut s = spec(
        |p| vec![p.attention(extras), p.attention(Extras::default())],
        false,
    );
    s.dense = DenseSemantics {
        rope_freqs: Some((0..HEAD_DIM / 2).map(|i| 1.0 + i as f32).collect()),
        attn_temp: Some((0.1, 4)),
        post_norm: true,
        ..Default::default()
    };
    s.cpu_rope = true;
    s
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Everything flushed by one forward pass: the batches (one per flush,
/// queue-internal auto-flushes included) and their `(op count, digest)`.
struct Captured {
    batches: Vec<String>,
    key: (usize, u64),
}

fn capture() -> Captured {
    let batches = op_capture::take();
    let text = batches.join("--flush--\n");
    let key = (
        text.lines().filter(|l| !l.starts_with("--")).count(),
        fnv(&text),
    );
    Captured { batches, key }
}

fn fresh_state(model: &HexagonLfmModel) -> InferenceState {
    InferenceState::from_config_capped(&model.config, &KvCompression::None, MAX_SEQ).unwrap()
}

fn decode_capture(spec: Spec, output: DecodeOutput) -> Captured {
    let model = build(spec);
    let mut state = fresh_state(&model);
    model
        .try_forward_input(DecodeInput::Token(3), output, 0, &mut state)
        .unwrap();
    capture()
}

fn prefill_capture_rows(spec: Spec, rows: usize) -> Captured {
    let model = build(spec);
    let mut state = fresh_state(&model);
    let tokens: Vec<u32> = (0..rows as u32).collect();
    model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .unwrap();
    capture()
}

fn prefill_capture(spec: Spec) -> Captured {
    prefill_capture_rows(spec, PREFILL_M)
}

fn histogram(batches: &[String]) -> std::collections::BTreeMap<String, usize> {
    op_capture::histogram(batches)
}

fn decode_batches(spec: Spec) -> Vec<String> {
    decode_capture(spec, DecodeOutput::Logits).batches
}

/// Check one pass against its pinned `(op count, digest)`.
fn assert_pinned(label: &str, got: &Captured, expected: (usize, u64)) {
    op_capture::assert_golden(label, got.key, expected, &got.batches);
}

/// Golden `(op count, digest)` per model kind and pass, pinned; regenerate
/// with `CERA_UPDATE_GOLDEN=1`. A mismatch means an op, operand or kernel
/// parameter changed.
#[test]
fn forward_op_sequences_are_pinned() {
    type Case = (&'static str, fn() -> Spec, (usize, u64), (usize, u64));
    let cases: [Case; 5] = [
        (
            "dense",
            dense_spec,
            (32, 18119808870354662753),
            (32, 9260691779644291356),
        ),
        (
            "conv",
            conv_spec,
            (33, 5465884269152608048),
            (31, 4826624611626170526),
        ),
        (
            "deltanet",
            deltanet_spec,
            (32, 15798833645049647584),
            (94, 13774925117970349302),
        ),
        (
            "extras_dsp",
            extras_dsp_spec,
            (53, 10850756757047705024),
            (53, 11863474017519813384),
        ),
        (
            "extras_host",
            extras_host_spec,
            (22, 7614221555435809170),
            (22, 11600172258080515279),
        ),
    ];
    for (label, spec, decode, prefill) in cases {
        let d = decode_capture(spec(), DecodeOutput::Logits);
        assert_pinned(&format!("fwd_{label}_decode"), &d, decode);
        let p = prefill_capture(spec());
        assert_pinned(&format!("fwd_{label}_prefill"), &p, prefill);
    }
    let greedy = decode_capture(dense_spec(), DecodeOutput::Greedy);
    assert_pinned("fwd_dense_greedy", &greedy, (33, 17315577994383314263));
}

/// Small-M prefill (fewer than `SMALL_M_FLUSH_CAP_ROWS` rows) caps ops per
/// flush, so the queue itself flushes mid-batch every `MAX_OPS_PER_FLUSH`
/// ops. The capture sits inside `HexagonQueueSession::flush`, so those
/// auto-flushes are part of the pinned sequence.
#[test]
fn small_m_capped_prefill_is_pinned() {
    let p = prefill_capture_rows(dense_spec(), 8);
    assert!(
        p.batches.len() > 1,
        "capped prefill must flush mid-batch, got {} flushes",
        p.batches.len()
    );
    for (i, b) in p.batches.iter().enumerate() {
        let n = b.lines().count();
        assert!(
            n <= MAX_OPS_PER_FLUSH,
            "flush {i} carries {n} ops, above the {MAX_OPS_PER_FLUSH} cap"
        );
    }
    assert_pinned("fwd_dense_small_m_prefill", &p, (32, 15080154379553609450));
}

/// The dense-path semantics add exactly the ops the CPU reference performs:
/// per layer 3 QKV + 1 output + 3 FFN bias adds and 2 residual multiplies,
/// plus one loop norm (interval 1, two layers) and 2 extra full-vector QK
/// norms over the dense baseline.
#[test]
fn extras_emit_the_expected_dsp_ops() {
    let base = histogram(&decode_batches(dense_spec()));
    let dsp = histogram(&decode_batches(extras_dsp_spec()));
    // Baseline: 4 residual adds; extras: 7 bias adds per layer on top.
    assert_eq!(base["Add"], 4);
    assert_eq!(dsp["Add"], 4 + 2 * 7);
    // Residual multiplier: attention and FFN block of each layer.
    assert!(!base.contains_key("Mul"));
    assert_eq!(dsp["Mul"], 4);
    // dense_spec norms: 2 layers x 2 block norms + 1 QK norm pair + final;
    // extras: 2 x 2 block + 2 layers x 2 QK + 1 loop norm + final.
    assert_eq!(base["RmsNormMul"], 7);
    assert_eq!(dsp["RmsNormMul"], 10);
    // Bias vectors are weight-flagged row broadcasts.
    let text = decode_batches(extras_dsp_spec()).join("");
    let bias_adds = text
        .lines()
        .filter(|l| l.starts_with("Add ") && l.contains(&format!("fl={HTP_TENSOR_WEIGHT} ")))
        .count();
    assert_eq!(bias_adds, 14);
}

/// Host-route models flush the queue around the host RoPE and the
/// attention-temperature scaling, so they emit several batches.
#[test]
fn host_steps_split_the_batch() {
    let dense = decode_batches(dense_spec());
    assert_eq!(dense.len(), 1);
    let host = decode_batches(extras_host_spec());
    assert!(host.len() > 2, "host RoPE flushes once per attention layer");
    // No DSP Rope op: both layers rotate on the host.
    assert!(!histogram(&host).contains_key("Rope"));
}

fn state_at(model: &HexagonLfmModel, seq_len: usize) -> InferenceState {
    let mut state =
        InferenceState::from_config_capped(&model.config, &KvCompression::None, MAX_SEQ).unwrap();
    state.seq_len = seq_len;
    model.current_seq_len.store(seq_len, Ordering::SeqCst);
    state
}

/// Recurrent (short-conv) state has no history ring, so a partial rewind
/// must be refused; attention-only models rewind freely.
#[test]
fn partial_rewind_is_gated_on_recurrent_layers() {
    use crate::kv_cache::KvRewindError;
    let conv = build(conv_spec());
    assert!(conv.has_recurrent_layers());
    let state = state_at(&conv, 10);
    assert!(matches!(
        conv.check_kv_rewind(&state, 5),
        Err(KvRewindError::BackendUnsupported)
    ));
    // Same length and a full rewind are always servable.
    assert!(conv.check_kv_rewind(&state, 10).is_ok());
    assert!(conv.check_kv_rewind(&state, 0).is_ok());
    assert!(matches!(
        conv.check_kv_rewind(&state, 11),
        Err(KvRewindError::OutOfBounds { .. })
    ));

    let dense = build(dense_spec());
    assert!(!dense.has_recurrent_layers());
    let state = state_at(&dense, 10);
    assert!(dense.check_kv_rewind(&state, 5).is_ok());
}

/// `CERA_HEXAGON_STEP` flushes at op-group boundaries: with step mode on,
/// every dispatch is its own batch (so a DSP fault names one op), and the
/// forward still completes and emits the same ops as the unstepped run.
#[test]
fn step_mode_flushes_each_op_group_and_completes() {
    type Case = (&'static str, fn() -> Spec);
    let specs: [Case; 4] = [
        ("dense", dense_spec),
        ("conv", conv_spec),
        ("deltanet", deltanet_spec),
        ("extras_host", extras_host_spec),
    ];
    for (label, spec) in specs {
        let plain = decode_capture(spec(), DecodeOutput::Logits);
        let plain_ops = plain.key.0;

        let model = build(spec());
        model
            .device
            .lock()
            .unwrap()
            .queue_session_mut()
            .set_step_mode(true);
        let mut state = fresh_state(&model);
        decode_once(&model, &mut state).unwrap();
        let stepped = capture();
        let batches: Vec<&String> = stepped.batches.iter().filter(|b| !b.is_empty()).collect();
        assert_eq!(
            stepped.key.0, plain_ops,
            "{label}: step mode changed the ops"
        );
        assert_eq!(batches.len(), plain_ops, "{label}: one flush per op group");
        assert!(batches.iter().all(|b| b.lines().count() == 1), "{label}");

        let tokens: Vec<u32> = (0..PREFILL_M as u32).collect();
        model
            .try_forward_prefill_chunk(&tokens, 0, &mut state)
            .unwrap();
        assert_eq!(state.seq_len, PREFILL_M);
    }
}

/// Serialize a staged (template) batch the way `op_capture` serializes a
/// flushed one, by loading it into the session and recording that.
fn staged_text(model: &HexagonLfmModel, staged: &StagedBatch) -> String {
    use crate::backend::hexagon::{HtpBufDesc, HtpOpDesc, HtpTensor};
    fn read_all<T: Copy>(bytes: &[u8], n: usize) -> Vec<T> {
        (0..n)
            .map(|i| unsafe {
                std::ptr::read_unaligned(
                    bytes.as_ptr().add(i * std::mem::size_of::<T>()) as *const T
                )
            })
            .collect()
    }
    let raw = &staged.raw_bytes;
    let bufs: Vec<HtpBufDesc> = read_all(raw, staged.n_bufs as usize);
    let tens: Vec<HtpTensor> = read_all(&raw[staged.bufs_bytes..], staged.n_tensors as usize);
    let ops: Vec<HtpOpDesc> = read_all(
        &raw[staged.bufs_bytes + staged.tens_bytes..],
        staged.n_ops as usize,
    );
    let mut device = model.device.lock().unwrap();
    let session = device.queue_session_mut();
    let map = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| (b.fd as i32, i as u16))
        .collect();
    session.load_batch(&bufs, &map, &tens, &ops);
    op_capture::take();
    op_capture::record(session);
    session.drop_pending_batch();
    op_capture::take().pop().unwrap()
}

/// The templated decode replay (`flush_staged_resident`) bypasses
/// `flush()`, so the op goldens never see it. Pin it against the fresh
/// emission: a second decode step replayed from the patched template
/// carries exactly the ops a from-scratch emission at that position does.
#[test]
fn templated_decode_replay_matches_fresh_emission() {
    let pos_step = |model: &HexagonLfmModel, state: &mut InferenceState, pos: usize| {
        model
            .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, pos, state)
            .unwrap();
    };
    // Replay model: step 0 builds the template, step 1 replays it.
    let replay = build(dense_spec());
    let mut state = fresh_state(&replay);
    pos_step(&replay, &mut state, 0);
    let mut step0 = op_capture::take();
    assert_eq!(step0.len(), 1, "step 0 flushes normally");
    let step0 = step0.pop().unwrap();
    let attempts = replay
        .device
        .lock()
        .unwrap()
        .queue_session_mut()
        .dispatch_attempts();
    pos_step(&replay, &mut state, 1);
    assert!(
        op_capture::take().is_empty(),
        "step 1 must replay the template, not re-emit"
    );
    assert_eq!(
        replay
            .device
            .lock()
            .unwrap()
            .queue_session_mut()
            .dispatch_attempts(),
        attempts + 1
    );
    let replayed = {
        let guard = replay.decode_template.lock().unwrap();
        staged_text(&replay, &guard.as_ref().unwrap().staged)
    };

    // Fresh model: same step 0, then drop the template so step 1 re-emits.
    let fresh = build(dense_spec());
    let mut state = fresh_state(&fresh);
    pos_step(&fresh, &mut state, 0);
    op_capture::take();
    *fresh.decode_template.lock().unwrap() = None;
    pos_step(&fresh, &mut state, 1);
    let mut batches = op_capture::take();
    assert_eq!(batches.len(), 1);
    assert_eq!(replayed, batches.pop().unwrap());
    // The patch is doing work: position 1 differs from the step-0 text.
    assert_ne!(replayed, step0);
}

/// Make every DSP batch write fail (`true`) or succeed again (`false`).
fn fail_dsp_writes(fail: bool) {
    fake::with(|s| s.fail_write = fail);
}

fn decode_once(model: &HexagonLfmModel, state: &mut InferenceState) -> Result<(), CeraError> {
    model
        .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, 0, state)
        .map(|_| ())
}

fn is_torn_error(e: &CeraError) -> bool {
    matches!(e, CeraError::Backend(m) if m.contains("recurrent state torn"))
}

/// A decode that dies at a capped flush, mid-way through the layer stack,
/// leaves recurrent state advanced while `seq_len` is not: the model must
/// refuse every later forward and partial rewind until a full reset.
#[test]
fn mid_forward_failure_tears_recurrent_state_until_reset() {
    use crate::kv_cache::KvRewindError;
    let mut model = build(conv_spec());
    // Flush every 4 ops so the failing flush is mid-forward, not the last.
    model.decode_ops_cap = Some(4);
    let mut state = fresh_state(&model);
    assert!(!model.state_torn.load(Ordering::SeqCst));

    fail_dsp_writes(true);
    assert!(decode_once(&model, &mut state).is_err());
    fail_dsp_writes(false);
    assert!(model.state_torn.load(Ordering::SeqCst));
    assert_eq!(
        state.seq_len, 0,
        "a failed forward does not advance seq_len"
    );

    // Device healthy again, but the state is torn: decode, prefill and a
    // partial rewind all refuse with the typed error.
    let err = decode_once(&model, &mut state).unwrap_err();
    assert!(is_torn_error(&err), "{err}");
    let err = model
        .try_forward_prefill_chunk(&[1, 2, 3], 0, &mut state)
        .unwrap_err();
    assert!(is_torn_error(&err), "{err}");
    state.seq_len = 5;
    assert!(matches!(
        model.check_kv_rewind(&state, 3),
        Err(KvRewindError::BackendUnsupported)
    ));
    assert!(model.check_kv_rewind(&state, 0).is_ok());

    // A full reset zeroes the state and clears the flag.
    model.truncate_kv(&mut state, 0);
    assert!(!model.state_torn.load(Ordering::SeqCst));
    decode_once(&model, &mut state).unwrap();

    // `try_reset_kv` clears it as well.
    model.decode_ops_cap = Some(4);
    let mut state = fresh_state(&model);
    fail_dsp_writes(true);
    assert!(decode_once(&model, &mut state).is_err());
    fail_dsp_writes(false);
    assert!(model.state_torn.load(Ordering::SeqCst));
    model
        .try_reset_kv(&mut state, &KvCompression::None, MAX_SEQ)
        .unwrap();
    assert!(!model.state_torn.load(Ordering::SeqCst));
    decode_once(&model, &mut state).unwrap();
}

/// Smoke test for the check-then-lock race: a forward queued on the device
/// lock while another forward fails must see the flag that failure sets.
/// The ordering itself is a compile-time property, not pinned by this test
/// (which is interleaving-dependent): `ensure_state_intact` takes the device
/// guard as a witness, so checking before locking does not compile.
#[test]
fn torn_check_smoke_forward_queued_behind_a_failing_one() {
    let model = build(conv_spec());
    let guard = model.device.lock().unwrap();
    let err = std::thread::scope(|scope| {
        let waiter = scope.spawn(|| {
            let mut state = fresh_state(&model);
            decode_once(&model, &mut state).unwrap_err()
        });
        model.mark_state_torn();
        drop(guard);
        waiter.join().unwrap()
    });
    assert!(is_torn_error(&err), "{err}");
}

/// A panic unwinding through a forward poisons the device lock while
/// recurrent state may have advanced and `mark_state_torn` never ran.
/// `lock_device` marks the state torn on poison, so the next forward reports
/// the torn error, and a full reset recovers. Attention-only models have no
/// recurrent state and just continue.
#[test]
fn panic_in_forward_tears_state_and_reset_recovers() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut model = build(conv_spec());
    // Flush every 4 ops so the panic is mid-forward.
    model.decode_ops_cap = Some(4);
    let mut state = fresh_state(&model);
    decode_once(&model, &mut state).unwrap();
    model.truncate_kv(&mut state, 0);

    op_capture::panic_on_next_record();
    let unwound = catch_unwind(AssertUnwindSafe(|| decode_once(&model, &mut state)));
    assert!(unwound.is_err(), "the injected panic must unwind");
    assert!(model.device.is_poisoned(), "unwind poisons the device lock");
    assert!(
        !model.state_torn.load(Ordering::SeqCst),
        "nothing marked the state on the unwind itself"
    );

    let err = decode_once(&model, &mut state).unwrap_err();
    assert!(is_torn_error(&err), "{err}");
    assert!(!model.device.is_poisoned(), "poison is cleared, not sticky");
    let err = model
        .try_forward_prefill_chunk(&[1, 2, 3], 0, &mut state)
        .unwrap_err();
    assert!(is_torn_error(&err), "{err}");

    model.truncate_kv(&mut state, 0);
    assert!(!model.state_torn.load(Ordering::SeqCst));
    decode_once(&model, &mut state).unwrap();

    // Attention-only: no recurrent state to tear, the retry just works.
    let dense = build(dense_spec());
    let mut state = fresh_state(&dense);
    op_capture::panic_on_next_record();
    let unwound = catch_unwind(AssertUnwindSafe(|| decode_once(&dense, &mut state)));
    assert!(unwound.is_err());
    decode_once(&dense, &mut state).unwrap();
    assert!(!dense.state_torn.load(Ordering::SeqCst));
}

/// A reset must not zero recurrent state a timed-out batch may still be
/// writing: it quiesces the queue first and stays torn if the DSP does not
/// answer, then completes once the response arrives.
#[test]
fn reset_waits_for_a_timed_out_batch_before_zeroing() {
    let mut model = build(conv_spec());
    model.decode_ops_cap = Some(4);
    let mut state = fresh_state(&model);
    // The read times out (fake: fails) after the batch was written.
    fake::with(|s| s.fail_read = true);
    assert!(decode_once(&model, &mut state).is_err());
    assert!(model.state_torn.load(Ordering::SeqCst));
    unsafe {
        std::ptr::write_bytes(
            model.kv_state_buf.as_mut_ptr(),
            0xAB,
            model.kv_state_buf.size(),
        );
    }
    let untouched = |m: &HexagonLfmModel| unsafe {
        std::slice::from_raw_parts(m.kv_state_buf.as_ptr(), m.kv_state_buf.size())
            .iter()
            .all(|&b| b == 0xAB)
    };

    // DSP still silent: neither reset flavor may touch the buffer.
    state.seq_len = 3;
    model.truncate_kv(&mut state, 0);
    assert!(untouched(&model), "truncate_kv(0) zeroed a busy buffer");
    assert!(model.state_torn.load(Ordering::SeqCst), "stays torn");
    let err = model
        .try_reset_kv(&mut state, &KvCompression::None, MAX_SEQ)
        .unwrap_err();
    assert!(err.to_string().contains("outstanding"), "{err}");
    assert!(untouched(&model), "try_reset_kv zeroed a busy buffer");

    // The DSP answers: the same reset now completes and heals.
    fake::with(|s| s.fail_read = false);
    model.truncate_kv(&mut state, 0);
    assert!(!model.state_torn.load(Ordering::SeqCst));
    assert!(!untouched(&model));
    decode_once(&model, &mut state).unwrap();
}

/// Recovery's tail rewind (`try_truncate_kv(0)`) must not report success
/// when the reset could not quiesce the DSP: it errors before touching
/// `state`, leaves the buffer alone and the model torn, so recovery falls
/// through to a full reset. Once the DSP answers, the same call zeroes and
/// heals.
#[test]
fn try_truncate_kv_zero_fails_when_the_reset_cannot_quiesce() {
    use crate::kv_cache::KvRewindError;
    let mut model = build(conv_spec());
    let mut state = fresh_state(&model);
    // A healthy uncapped decode first, so a decode template exists to survive
    // the failed rewind below; then cap the ops so the next decode tears.
    decode_once(&model, &mut state).unwrap();
    let has_template = |m: &HexagonLfmModel| m.decode_template.lock().unwrap().is_some();
    assert!(has_template(&model), "precondition: template captured");
    model.decode_ops_cap = Some(4);
    fake::with(|s| s.fail_read = true);
    assert!(decode_once(&model, &mut state).is_err());
    assert!(model.state_torn.load(Ordering::SeqCst));
    unsafe {
        std::ptr::write_bytes(
            model.kv_state_buf.as_mut_ptr(),
            0xAB,
            model.kv_state_buf.size(),
        );
    }
    let untouched = |m: &HexagonLfmModel| unsafe {
        std::slice::from_raw_parts(m.kv_state_buf.as_ptr(), m.kv_state_buf.size())
            .iter()
            .all(|&b| b == 0xAB)
    };
    state.seq_len = 3;
    model.current_seq_len.store(3, Ordering::SeqCst);

    let err = model.try_truncate_kv(&mut state, 0).unwrap_err();
    assert!(matches!(err, KvRewindError::BackendUnsupported), "{err}");
    assert!(untouched(&model), "failed reset zeroed a busy buffer");
    assert_eq!(state.seq_len, 3, "state untouched on failure");
    assert_eq!(model.current_seq_len.load(Ordering::SeqCst), 3);
    assert!(model.state_torn.load(Ordering::SeqCst), "stays torn");
    assert!(has_template(&model), "a failed rewind changes nothing");

    // The DSP answers: the same call now zeroes, heals and moves the position.
    fake::with(|s| s.fail_read = false);
    model.try_truncate_kv(&mut state, 0).unwrap();
    assert!(!untouched(&model));
    assert_eq!(state.seq_len, 0);
    assert!(!model.state_torn.load(Ordering::SeqCst));
    decode_once(&model, &mut state).unwrap();
}

/// A failure before any batch reaches the DSP (here a tensor-table cap, the
/// way an over-full batch is rejected at registration) leaves the recurrent
/// state untouched: decode and prefill both stay usable. Pins the
/// `mark_state_torn_if_dispatched` call sites, which an unconditional
/// `mark_state_torn` would fail.
#[test]
fn pre_dispatch_failure_does_not_tear_state() {
    let model = build(conv_spec());
    let mut state = fresh_state(&model);
    let set_cap = |cap: Option<usize>| {
        model
            .device
            .lock()
            .unwrap()
            .queue_session_mut()
            .set_tensor_cap(cap);
    };
    let attempts = || {
        model
            .device
            .lock()
            .unwrap()
            .queue_session_mut()
            .dispatch_attempts()
    };
    let before = attempts();

    set_cap(Some(1));
    let err = decode_once(&model, &mut state).unwrap_err();
    assert!(err.to_string().contains("tensor cap"), "{err}");
    let tokens: Vec<u32> = (0..8).collect();
    let err = model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .unwrap_err();
    assert!(err.to_string().contains("tensor cap"), "{err}");
    assert_eq!(attempts(), before, "nothing reached the DSP");
    assert!(!model.state_torn.load(Ordering::SeqCst));

    set_cap(None);
    decode_once(&model, &mut state).unwrap();
    model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .unwrap();
    assert!(!model.state_torn.load(Ordering::SeqCst));
}

/// A failure that never reached the DSP (emit-time validation, batch
/// registration) leaves the recurrent state untouched, so the session
/// stays usable; one after a dispatch attempt tears it. The end-to-end
/// pre-dispatch case is `pre_dispatch_failure_does_not_tear_state` (tensor
/// cap, decode emit and prefill); the templated-replay site cannot fail
/// before dispatch in the tiny fixtures, so this also pins the decision at
/// its seam: the attempt counter every forward path hands to
/// `mark_state_torn_if_dispatched`.
#[test]
fn torn_only_when_a_dispatch_was_attempted() {
    let model = build(conv_spec());
    let mut state = fresh_state(&model);

    let before = model
        .device
        .lock()
        .unwrap()
        .queue_session_mut()
        .dispatch_attempts();
    {
        let mut device = model.device.lock().unwrap();
        model.mark_state_torn_if_dispatched(device.queue_session_mut(), before);
    }
    assert!(
        !model.state_torn.load(Ordering::SeqCst),
        "no dispatch attempted: state intact"
    );
    decode_once(&model, &mut state).unwrap();

    // The same decision after a real dispatch (a decode) since `before`.
    {
        let mut device = model.device.lock().unwrap();
        model.mark_state_torn_if_dispatched(device.queue_session_mut(), before);
    }
    assert!(model.state_torn.load(Ordering::SeqCst));

    // A dispatch that fails still counts as an attempt: the end-to-end
    // failure path tears (see the tests above) and a later reset heals.
    model.truncate_kv(&mut state, 0);
    assert!(!model.state_torn.load(Ordering::SeqCst));
    fail_dsp_writes(true);
    assert!(decode_once(&model, &mut state).is_err());
    fail_dsp_writes(false);
    assert!(model.state_torn.load(Ordering::SeqCst));
}

/// Prefill chunks (small-M chunks flush every `MAX_OPS_PER_FLUSH` ops)
/// tear the state the same way.
#[test]
fn mid_prefill_failure_tears_recurrent_state() {
    let model = build(conv_spec());
    let mut state = fresh_state(&model);
    let tokens: Vec<u32> = (0..8).collect();
    fail_dsp_writes(true);
    assert!(
        model
            .try_forward_prefill_chunk(&tokens, 0, &mut state)
            .is_err()
    );
    fail_dsp_writes(false);
    assert!(model.state_torn.load(Ordering::SeqCst));
    let err = model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .unwrap_err();
    assert!(is_torn_error(&err), "{err}");
    model.truncate_kv(&mut state, 0);
    model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .unwrap();
    assert_eq!(state.seq_len, 8);
}

/// Attention-only KV slots are rewritten in place on a retry, so a failed
/// forward never marks the model torn and the retry just works.
#[test]
fn attention_only_model_is_never_torn() {
    let mut model = build(dense_spec());
    model.decode_ops_cap = Some(4);
    let mut state = fresh_state(&model);
    fail_dsp_writes(true);
    assert!(decode_once(&model, &mut state).is_err());
    let tokens: Vec<u32> = (0..8).collect();
    assert!(
        model
            .try_forward_prefill_chunk(&tokens, 0, &mut state)
            .is_err()
    );
    fail_dsp_writes(false);
    assert!(!model.state_torn.load(Ordering::SeqCst));
    decode_once(&model, &mut state).unwrap();
    model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .unwrap();
}

/// A full reset zeroes the recurrent state buffer; an attention-only model
/// leaves the KV bytes alone (stale rows past `seq_len` are never read).
#[test]
fn truncate_to_zero_clears_recurrent_state_only() {
    let fill = |m: &HexagonLfmModel| unsafe {
        std::ptr::write_bytes(m.kv_state_buf.as_mut_ptr(), 0xAB, m.kv_state_buf.size());
    };
    let all = |m: &HexagonLfmModel, byte: u8| unsafe {
        std::slice::from_raw_parts(m.kv_state_buf.as_ptr(), m.kv_state_buf.size())
            .iter()
            .all(|&b| b == byte)
    };

    let conv = build(conv_spec());
    fill(&conv);
    let mut state = state_at(&conv, 10);
    conv.truncate_kv(&mut state, 0);
    assert!(all(&conv, 0));
    assert_eq!(state.seq_len, 0);
    assert_eq!(conv.current_seq_len.load(Ordering::SeqCst), 0);

    let dense = build(dense_spec());
    fill(&dense);
    let mut state = state_at(&dense, 10);
    dense.truncate_kv(&mut state, 4);
    assert!(all(&dense, 0xAB));
    assert_eq!(state.seq_len, 4);
    dense.truncate_kv(&mut state, 0);
    assert!(all(&dense, 0xAB));
}

/// `check_kv_rewind` reads the torn flag under the device lock: while another
/// thread holds it, the call blocks, and the flag set meanwhile is seen.
#[test]
fn check_kv_rewind_takes_the_device_lock() {
    use crate::kv_cache::KvRewindError;
    let model = build(conv_spec());
    let state = state_at(&model, 4);
    // Passes while the flag is clear (a non-zero target on a torn model errs).
    assert!(model.check_kv_rewind(&state, 4).is_ok());
    let guard = model.device.lock().unwrap();
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| model.check_kv_rewind(&state, 4));
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            !handle.is_finished(),
            "check_kv_rewind ran without the lock"
        );
        model.mark_state_torn();
        drop(guard);
        assert!(matches!(
            handle.join().unwrap(),
            Err(KvRewindError::BackendUnsupported)
        ));
    });
}

//! Host tests of the three real constructors on a fake FastRPC device, from
//! synthetic tiny GGUFs: dense (Qwen2-style biases, tied head), LFM2 with a
//! short-conv layer, and LFM2-MoE (routed FFN, stacked Q4_0 experts). Each
//! digest covers the planned layer offsets, the scratch layout, the semantics
//! and every byte of the repacked weights buffer, plus the op sequence of a
//! decode and a prefill pass over the built model, so a change to weight
//! planning, copying or the shared setup shows up here without a device.

use super::*;
use crate::backend::hexagon::op_capture;
use crate::gguf::{GgufBuilder, KvValue};

const HS: usize = 64;
const INTER: usize = 128;
const VOCAB: usize = 64;
const N_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const N_EXPERT: usize = 8;
const N_USED: usize = 2;
const EXPERT_FF: usize = 96;

/// One synthetic tensor.
struct Tensor {
    name: String,
    dims: Vec<usize>,
    /// GGML type id.
    ggml_type: u32,
    data: Vec<u8>,
}

struct Gen(u64);

impl Gen {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn f32(&mut self, name: &str, dims: &[usize], scale: f32, base: f32) -> Tensor {
        let n: usize = dims.iter().product();
        Tensor {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type: 0,
            data: (0..n)
                .flat_map(|_| (base + self.next() * 2.0 * scale).to_le_bytes())
                .collect(),
        }
    }

    /// Q4_0 blocks: a finite f16 scale followed by random nibbles.
    fn q4_0(&mut self, name: &str, dims: &[usize]) -> Tensor {
        let blocks: usize = dims.iter().product::<usize>() / 32;
        let mut data = Vec::with_capacity(blocks * 18);
        for _ in 0..blocks {
            data.extend_from_slice(
                &crate::quant::f32_to_f16(0.02 + self.next().abs() * 0.05).to_le_bytes(),
            );
            data.extend((0..16).map(|_| ((self.next() + 0.5) * 255.0) as u8));
        }
        Tensor {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type: 2,
            data,
        }
    }
}

impl Gen {
    /// Q8_0 blocks: a finite f16 scale followed by 32 random i8 quants.
    fn q8_0(&mut self, name: &str, dims: &[usize]) -> Tensor {
        let blocks: usize = dims.iter().product::<usize>() / 32;
        let mut data = Vec::with_capacity(blocks * 34);
        for _ in 0..blocks {
            data.extend_from_slice(
                &crate::quant::f32_to_f16(0.01 + self.next().abs() * 0.02).to_le_bytes(),
            );
            data.extend((0..32).map(|_| (self.next() * 254.0) as i8 as u8));
        }
        Tensor {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type: 8,
            data,
        }
    }
}

fn gguf(kv: Vec<(String, KvValue)>, tensors: Vec<Tensor>) -> GgufFile {
    let mut b = GgufBuilder::new();
    for (key, v) in kv {
        b = b.kv(key, v);
    }
    for t in tensors {
        b = b.tensor(t.name, &t.dims, t.ggml_type, t.data);
    }
    b.build()
}

/// A tensor with no dimensions (the GGUF reader accepts `n_dims == 0`) is
/// a load error naming the tensor, not an index panic.
#[test]
fn zero_dim_tensor_is_a_named_error() {
    let g = gguf(
        vec![],
        vec![Tensor {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![],
            ggml_type: 0,
            data: vec![0; 4],
        }],
    );
    let err = GgufSource { gguf: &g }
        .weight_shape("blk.0.attn_q.weight")
        .unwrap_err();
    assert!(err.to_string().contains("blk.0.attn_q.weight"), "{err}");
    assert!(err.to_string().contains("no dimensions"), "{err}");
}

fn backend() -> Backend {
    let (driver, device) = op_capture::fresh_device();
    // Every default but the decode tensor cap, which would split the decode the
    // goldens pin as one batch (the forward goldens say why).
    let knobs =
        HexagonKnobs::from_lookup(|k| (k == "CERA_HEXAGON_BATCH_TENSORS").then(|| "0".to_string()));
    Backend::with_device(driver, device, knobs)
}

fn u32kv(key: &str, prefix: &str, v: u32) -> (String, KvValue) {
    (format!("{prefix}.{key}"), KvValue::U32(v))
}

/// Qwen2-style dense model: QKV, output and FFN biases, a tied LM head, and
/// a residual multiplier-free config.
fn dense_gguf() -> GgufFile {
    dense_gguf_with(vec![])
}

/// [`dense_gguf`] plus extra metadata pairs (appended after the base ones).
fn dense_gguf_with(extra_kv: Vec<(String, KvValue)>) -> GgufFile {
    let mut g = Gen(0x1234_5678);
    let mut t = vec![
        g.f32("token_embd.weight", &[HS, VOCAB], 1.0, 0.0),
        g.f32("output_norm.weight", &[HS], 0.1, 1.0),
    ];
    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = HEAD_DIM;
    for l in 0..2 {
        let n = |s: &str| format!("blk.{l}.{s}");
        t.push(g.f32(&n("attn_norm.weight"), &[HS], 0.1, 1.0));
        t.push(g.f32(&n("ffn_norm.weight"), &[HS], 0.1, 1.0));
        t.push(g.f32(&n("attn_q.weight"), &[HS, q_dim], 0.3, 0.0));
        t.push(g.f32(&n("attn_k.weight"), &[HS, kv_dim], 0.3, 0.0));
        t.push(g.f32(&n("attn_v.weight"), &[HS, kv_dim], 0.3, 0.0));
        t.push(g.f32(&n("attn_output.weight"), &[q_dim, HS], 0.3, 0.0));
        t.push(g.f32(&n("attn_q.bias"), &[q_dim], 0.1, 0.0));
        t.push(g.f32(&n("attn_k.bias"), &[kv_dim], 0.1, 0.0));
        t.push(g.f32(&n("attn_v.bias"), &[kv_dim], 0.1, 0.0));
        t.push(g.f32(&n("attn_output.bias"), &[HS], 0.1, 0.0));
        t.push(g.f32(&n("ffn_gate.weight"), &[HS, INTER], 0.3, 0.0));
        t.push(g.f32(&n("ffn_up.weight"), &[HS, INTER], 0.3, 0.0));
        t.push(g.f32(&n("ffn_down.weight"), &[INTER, HS], 0.3, 0.0));
        t.push(g.f32(&n("ffn_gate.bias"), &[INTER], 0.1, 0.0));
        t.push(g.f32(&n("ffn_up.bias"), &[INTER], 0.1, 0.0));
        t.push(g.f32(&n("ffn_down.bias"), &[HS], 0.1, 0.0));
    }
    let p = "qwen2";
    let mut kv = vec![
        ("general.architecture".into(), KvValue::Str("qwen2".into())),
        u32kv("block_count", p, 2),
        u32kv("embedding_length", p, HS as u32),
        u32kv("feed_forward_length", p, INTER as u32),
        u32kv("attention.head_count", p, N_HEADS as u32),
        u32kv("attention.head_count_kv", p, 1),
        (
            format!("{p}.attention.layer_norm_rms_epsilon"),
            KvValue::F32(1e-6),
        ),
        (format!("{p}.rope.freq_base"), KvValue::F32(10_000.0)),
        u32kv("context_length", p, 64),
        u32kv("vocab_size", p, VOCAB as u32),
    ];
    kv.extend(extra_kv);
    gguf(kv, t)
}

/// Qwen 3.5 hybrid: layers 0 and 2 Gated DeltaNet, layers 1 and 3 full
/// attention (layer 1 with the doubled Q for the output gate), partial
/// rotary (16 of 32 dims) and a tied LM head.
fn qwen35_gguf() -> GgufFile {
    const DN_GROUP: usize = 2;
    const DN_STATE: usize = 16;
    const DN_RANK: usize = 4;
    const DN_CONV_K: usize = 4;
    const CONV_DIM: usize = 2 * DN_GROUP * DN_STATE + DN_RANK * DN_STATE;
    const VALUE_DIM: usize = DN_RANK * DN_STATE;
    let mut g = Gen(0x5157_4e33);
    let mut t = vec![
        g.f32("token_embd.weight", &[HS, VOCAB], 1.0, 0.0),
        g.f32("output_norm.weight", &[HS], 0.1, 1.0),
    ];
    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = HEAD_DIM;
    for l in 0..4 {
        let n = |s: &str| format!("blk.{l}.{s}");
        t.push(g.f32(&n("attn_norm.weight"), &[HS], 0.1, 1.0));
        t.push(g.f32(&n("attn_post_norm.weight"), &[HS], 0.1, 1.0));
        t.push(g.q4_0(&n("ffn_gate.weight"), &[HS, INTER]));
        t.push(g.q4_0(&n("ffn_up.weight"), &[HS, INTER]));
        t.push(g.q4_0(&n("ffn_down.weight"), &[INTER, HS]));
        if l % 2 == 0 {
            t.push(g.q4_0(&n("attn_qkv.weight"), &[HS, CONV_DIM]));
            t.push(g.q4_0(&n("attn_gate.weight"), &[HS, VALUE_DIM]));
            t.push(g.f32(&n("ssm_conv1d.weight"), &[DN_CONV_K, CONV_DIM], 0.2, 0.0));
            t.push(g.f32(&n("ssm_conv1d.bias"), &[CONV_DIM], 0.05, 0.0));
            t.push(g.f32(&n("ssm_dt.bias"), &[DN_RANK], 0.1, 0.5));
            t.push(g.f32(&n("ssm_a"), &[DN_RANK], 0.1, -0.5));
            t.push(g.q4_0(&n("ssm_beta.weight"), &[HS, DN_RANK]));
            t.push(g.q4_0(&n("ssm_alpha.weight"), &[HS, DN_RANK]));
            t.push(g.f32(&n("ssm_norm.weight"), &[DN_STATE], 0.1, 1.0));
            t.push(g.q4_0(&n("ssm_out.weight"), &[VALUE_DIM, HS]));
        } else {
            let q_out = if l == 1 { 2 * q_dim } else { q_dim };
            t.push(g.q4_0(&n("attn_q.weight"), &[HS, q_out]));
            t.push(g.q4_0(&n("attn_k.weight"), &[HS, kv_dim]));
            t.push(g.q4_0(&n("attn_v.weight"), &[HS, kv_dim]));
            t.push(g.q4_0(&n("attn_output.weight"), &[q_dim, HS]));
            t.push(g.f32(&n("attn_q_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
            t.push(g.f32(&n("attn_k_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
        }
    }
    let p = "qwen35";
    gguf(
        vec![
            ("general.architecture".into(), KvValue::Str("qwen35".into())),
            u32kv("block_count", p, 4),
            u32kv("embedding_length", p, HS as u32),
            u32kv("feed_forward_length", p, INTER as u32),
            u32kv("attention.head_count", p, N_HEADS as u32),
            u32kv("attention.head_count_kv", p, 1),
            u32kv("attention.key_length", p, HEAD_DIM as u32),
            (
                format!("{p}.attention.layer_norm_rms_epsilon"),
                KvValue::F32(1e-6),
            ),
            (format!("{p}.rope.freq_base"), KvValue::F32(10_000.0)),
            u32kv("rope.dimension_count", p, 16),
            u32kv("ssm.conv_kernel", p, DN_CONV_K as u32),
            u32kv("ssm.state_size", p, DN_STATE as u32),
            u32kv("ssm.time_step_rank", p, DN_RANK as u32),
            u32kv("ssm.group_count", p, DN_GROUP as u32),
            u32kv("ssm.inner_size", p, VALUE_DIM as u32),
            u32kv("full_attention_interval", p, 2),
            u32kv("context_length", p, 64),
            u32kv("vocab_size", p, VOCAB as u32),
        ],
        t,
    )
}

/// LFM2: layer 0 short conv, layer 1 attention with QK norm; `moe` swaps the
/// FFN of layers 1 and 2 for routed experts (layer 2 is a second conv).
fn lfm2_gguf(moe: bool) -> GgufFile {
    lfm2_gguf_layers(moe, if moe { 3 } else { 2 })
}

/// [`lfm2_gguf`] with `n_layers` layers: past layer 1 they alternate no
/// further, every extra layer is a short conv (routed FFN when `moe`).
fn lfm2_gguf_layers(moe: bool, n_layers: usize) -> GgufFile {
    let mut g = Gen(0x9e37_79b9);
    let prefix = if moe { "lfm2moe" } else { "lfm2" };
    let mut t = vec![
        g.f32("token_embd.weight", &[HS, VOCAB], 1.0, 0.0),
        g.f32("token_embd_norm.weight", &[HS], 0.1, 1.0),
    ];
    let q_dim = N_HEADS * HEAD_DIM;
    for l in 0..n_layers {
        let n = |s: &str| format!("blk.{l}.{s}");
        t.push(g.f32(&n("attn_norm.weight"), &[HS], 0.1, 1.0));
        t.push(g.f32(&n("ffn_norm.weight"), &[HS], 0.1, 1.0));
        let is_attn = l == 1;
        if is_attn {
            t.push(g.q4_0(&n("attn_q.weight"), &[HS, q_dim]));
            t.push(g.q4_0(&n("attn_k.weight"), &[HS, HEAD_DIM]));
            t.push(g.q4_0(&n("attn_v.weight"), &[HS, HEAD_DIM]));
            t.push(g.q4_0(&n("attn_output.weight"), &[q_dim, HS]));
            t.push(g.f32(&n("attn_q_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
            t.push(g.f32(&n("attn_k_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
        } else {
            t.push(g.q4_0(&n("shortconv.in_proj.weight"), &[HS, 3 * HS]));
            t.push(g.q4_0(&n("shortconv.out_proj.weight"), &[HS, HS]));
            t.push(g.f32(&n("shortconv.conv.weight"), &[3, HS], 0.3, 0.0));
        }
        if moe && l > 0 {
            t.push(g.f32(&n("ffn_gate_inp.weight"), &[HS, N_EXPERT], 0.3, 0.0));
            t.push(g.f32(&n("exp_probs_b.bias"), &[N_EXPERT], 0.05, 0.0));
            t.push(g.q4_0(&n("ffn_gate_exps.weight"), &[HS, EXPERT_FF, N_EXPERT]));
            t.push(g.q4_0(&n("ffn_up_exps.weight"), &[HS, EXPERT_FF, N_EXPERT]));
            // Q8_0 down experts: the fixture exercises both stacked repackers.
            t.push(g.q8_0(&n("ffn_down_exps.weight"), &[EXPERT_FF, HS, N_EXPERT]));
        } else {
            t.push(g.q4_0(&n("ffn_gate.weight"), &[HS, INTER]));
            t.push(g.q4_0(&n("ffn_up.weight"), &[HS, INTER]));
            t.push(g.q4_0(&n("ffn_down.weight"), &[INTER, HS]));
        }
    }
    let mut kv = vec![
        (
            "general.architecture".into(),
            KvValue::Str(if moe { "lfm2moe" } else { "lfm2" }.into()),
        ),
        u32kv("block_count", prefix, n_layers as u32),
        u32kv("embedding_length", prefix, HS as u32),
        u32kv("feed_forward_length", prefix, INTER as u32),
        u32kv("attention.head_count", prefix, N_HEADS as u32),
        (
            format!("{prefix}.attention.head_count_kv"),
            KvValue::I32Array((0..n_layers).map(|l| if l == 1 { 1 } else { 0 }).collect()),
        ),
        u32kv("shortconv.l_cache", prefix, 3),
        (
            format!("{prefix}.attention.layer_norm_rms_epsilon"),
            KvValue::F32(1e-5),
        ),
        (format!("{prefix}.rope.freq_base"), KvValue::F32(10_000.0)),
        u32kv("context_length", prefix, 64),
        u32kv("vocab_size", prefix, VOCAB as u32),
    ];
    if moe {
        kv.push(u32kv("expert_gating_func", prefix, 2));
        kv.push(u32kv("expert_count", prefix, N_EXPERT as u32));
        kv.push(u32kv("expert_used_count", prefix, N_USED as u32));
        kv.push(u32kv(
            "expert_feed_forward_length",
            prefix,
            EXPERT_FF as u32,
        ));
    }
    gguf(kv, t)
}

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Digest of the built model: layout, semantics, weights and the op
/// sequence of one decode and one prefill pass. Also returns the recorded
/// op batches so a mismatch can print their histogram.
fn model_digest(label: &str, model: &HexagonLfmModel) -> ((u64, u64), Vec<String>) {
    let mut plan = String::new();
    for layer in &model.layers {
        match layer {
            HexagonLayer::Attention(l) => plan.push_str(&format!("{l:?}\n")),
            HexagonLayer::Conv(l) => plan.push_str(&format!("{l:?}\n")),
            HexagonLayer::DeltaNet(l) => plan.push_str(&format!("{l:?}\n")),
        }
    }
    let d = &model.dense;
    plan.push_str(&format!(
            "{:?} {:?} lm={:?} onorm={} rope={:?} cpu_rope={} act={:?} sc={:?}/{:?} moe={} dn={} kv={:?} \
             vtcm={} attn_scale={:?} rope_dim={:?} res={:?} logit={:?} freqs={:?} post={} loop={:?} \
             temp={:?} swa={:?}/{}",
            model.config.architecture,
            model.scratch_offsets,
            model.lm_head,
            model.output_norm_offset,
            model.rope_type,
            model.cpu_rope,
            model.activation,
            model.attn_logit_softcapping,
            model.final_logit_softcapping,
            model.has_moe,
            model.has_deltanet,
            model.kv_dtype,
            model.vtcm_budget,
            d.attn_scale,
            d.rope_dim,
            d.residual_vec_offset,
            d.logit_scale,
            d.rope_freqs,
            d.post_norm,
            d.loop_norm_interval,
            d.attn_temp,
            d.swa_window,
            d.mask_swa.is_some(),
        ));
    op_capture::dump_if_requested(&format!("{label}_plan"), std::slice::from_ref(&plan));
    let mut h = fnv(0xcbf2_9ce4_8422_2325, plan.as_bytes());
    h = fnv(h, model.weights_buf.as_slice());
    h = fnv(
        h,
        &model.kv_state_buf.as_slice()[..model.kv_state_buf.size()],
    );

    let mut state = InferenceState::from_config_capped(
        &model.config,
        &KvCompression::None,
        model.config.max_seq_len,
    )
    .unwrap();
    model
        .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, 0, &mut state)
        .unwrap();
    let decode_batches = op_capture::take();
    let decode = decode_batches.join("|");
    let tokens: Vec<u32> = (0..32).collect();
    model
        .try_forward_prefill_chunk(&tokens, 1, &mut state)
        .unwrap();
    let prefill_batches = op_capture::take();
    let prefill = prefill_batches.join("|");
    let ops = fnv(
        fnv(0xcbf2_9ce4_8422_2325, decode.as_bytes()),
        prefill.as_bytes(),
    );
    let mut batches = decode_batches;
    batches.extend(prefill_batches);
    ((h, ops), batches)
}

/// Check a built model against its pinned `(weights digest, op digest)`.
fn assert_model(label: &str, model: &HexagonLfmModel, expected: (u64, u64)) {
    let (got, batches) = model_digest(label, model);
    op_capture::assert_golden(label, got, expected, &batches);
}

/// A fake device with every knob at its default, including the tensor cap that `backend()` zeroes.
fn default_backend() -> Backend {
    let (driver, device) = op_capture::fresh_device();
    Backend::with_device(driver, device, HexagonKnobs::from_lookup(|_| None))
}

/// The constructors refuse a model whose conv layers would share a staging slot (the layer index
/// wraps past `CONV_STAGE_SLOTS`), instead of building one whose conv state silently corrupts. The
/// helper's own test cannot see a constructor that stops calling it.
#[test]
fn constructor_refuses_conv_layers_that_share_a_stage_slot() {
    let err = HexagonLfmModel::from_gguf_on(
        default_backend(),
        lfm2_gguf_layers(false, CONV_STAGE_SLOTS + 2),
        64,
    )
    .err()
    .expect("a model with conv layers that share a stage slot must be refused");
    assert!(err.to_string().contains("conv stage slot"), "{err}");
}

/// No model runs with a tensor cap by default. Long batches were irreproducible until the conv state
/// writeback stopped reading a row every conv layer shares; the dense LFM2, the routed-expert LFM2,
/// a plain Llama and Qwen 3.5 (DeltaNet) were then verified bit-exact uncapped on a device. Runs
/// hermetically: `default_backend()` takes its knobs from `from_lookup`, not the environment.
#[test]
fn default_batch_cap_is_off_for_every_model() {
    let lfm2 = HexagonLfmModel::from_gguf_on(default_backend(), lfm2_gguf(false), 64).unwrap();
    let moe = HexagonLfmModel::from_gguf_on(default_backend(), lfm2_gguf(true), 64).unwrap();
    let dense = HexagonLfmModel::from_llama_on(default_backend(), dense_gguf(), None, 64).unwrap();
    let cpu = crate::model::qwen35::Qwen35Model::from_gguf(qwen35_gguf(), 64).unwrap();
    let qwen35 = HexagonLfmModel::from_qwen35_model_on(default_backend(), &cpu, 64).unwrap();
    for (name, m) in [
        ("lfm2", lfm2),
        ("moe", moe),
        ("dense", dense),
        ("qwen35", qwen35),
    ] {
        assert_eq!(m.batch_tensor_cap, None, "{name} runs uncapped");
    }
}

/// Constructor digests, pinned; regenerate with `CERA_UPDATE_GOLDEN=1`. A
/// mismatch means planning, copying or setup changed.
#[test]
fn constructors_build_pinned_models() {
    let dense = HexagonLfmModel::from_llama_on(backend(), dense_gguf(), None, 64).unwrap();
    assert_model(
        "ctor_dense",
        &dense,
        (5153141503434646990, 64189687555477636),
    );
    let lfm2 = HexagonLfmModel::from_gguf_on(backend(), lfm2_gguf(false), 64).unwrap();
    assert_model(
        "ctor_lfm2",
        &lfm2,
        (3932535791928950102, 16081593749882006606),
    );
    let moe = HexagonLfmModel::from_gguf_on(backend(), lfm2_gguf(true), 64).unwrap();
    assert_model(
        "ctor_moe",
        &moe,
        (17415058047840074287, 4431082039610235919),
    );
    let cpu = crate::model::qwen35::Qwen35Model::from_gguf(qwen35_gguf(), 64).unwrap();
    let qwen35 = HexagonLfmModel::from_qwen35_model_on(backend(), &cpu, 64).unwrap();
    assert_model(
        "ctor_qwen35",
        &qwen35,
        (3116775937788930880, 1277822656173788493),
    );
}

fn weights_f32(model: &HexagonLfmModel, offset: usize, len: usize) -> Vec<f32> {
    model.weights_buf.as_slice()[offset..offset + len * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn tensor_f32(gguf: &GgufFile, name: &str) -> Vec<f32> {
    gguf.get_tensor(name).unwrap().to_f32_vec()
}

/// Every bias the dense GGUF carries lands, value for value, at the offset
/// the forward pass adds it from.
#[test]
fn dense_biases_are_planned_and_copied_where_the_forward_reads_them() {
    let file = dense_gguf();
    let model = HexagonLfmModel::from_llama_on(backend(), file.clone(), None, 64).unwrap();
    assert!(model.dense.residual_vec_offset.is_none() && !model.dense.post_norm);
    for (l, layer) in model.layers.iter().enumerate() {
        let HexagonLayer::Attention(a) = layer else {
            panic!("dense model has only attention layers");
        };
        let [qo, ko, vo] = a.qkv_bias.expect("qwen2 carries QKV biases");
        let b = |s: &str| tensor_f32(&file, &format!("blk.{l}.{s}"));
        assert_eq!(weights_f32(&model, qo, a.q_dim), b("attn_q.bias"));
        assert_eq!(weights_f32(&model, ko, a.kv_dim), b("attn_k.bias"));
        assert_eq!(weights_f32(&model, vo, a.kv_dim), b("attn_v.bias"));
        let out = a.out_bias.expect("attn_output.bias");
        assert_eq!(weights_f32(&model, out, HS), b("attn_output.bias"));
        let HexagonFfn::Dense(ffn) = &a.ffn else {
            panic!("dense FFN expected");
        };
        assert_eq!(
            weights_f32(&model, ffn.gate_bias.unwrap(), INTER),
            b("ffn_gate.bias")
        );
        assert_eq!(
            weights_f32(&model, ffn.up_bias.unwrap(), INTER),
            b("ffn_up.bias")
        );
        assert_eq!(
            weights_f32(&model, ffn.down_bias.unwrap(), HS),
            b("ffn_down.bias")
        );
    }
    // 3 QKV + 1 output + 3 FFN bias adds per layer show up in the ops.
    let mut state =
        InferenceState::from_config_capped(&model.config, &KvCompression::None, 64).unwrap();
    model
        .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, 0, &mut state)
        .unwrap();
    let text = op_capture::take().join("");
    let bias_adds = text
        .lines()
        .filter(|l| l.starts_with("Add ") && l.contains(&format!("fl={HTP_TENSOR_WEIGHT} ")))
        .count();
    assert_eq!(bias_adds, 2 * 7);
}

/// A residual multiplier (Granite / MiniCPM) becomes a `[hidden]` constant
/// vector in `weights_buf`, at the offset the residual-scale op reads. The
/// only test with `scalars.residual != 1.0`: dropping the
/// `CopyOp::Constant` arm leaves the vector zero and fails here.
#[test]
fn residual_multiplier_is_copied_as_a_constant_vector() {
    let scale = 0.22f32;
    let file = dense_gguf_with(vec![("qwen2.residual_scale".into(), KvValue::F32(scale))]);
    let model = HexagonLfmModel::from_llama_on(backend(), file, None, 64).unwrap();
    assert_eq!(model.config.scalars.residual, scale);
    let off = model
        .dense
        .residual_vec_offset
        .expect("residual != 1.0 plans a constant vector");
    assert_eq!(weights_f32(&model, off, HS), vec![scale; HS]);
    // The default model plans none.
    let plain = HexagonLfmModel::from_llama_on(backend(), dense_gguf(), None, 64).unwrap();
    assert!(plain.dense.residual_vec_offset.is_none());
}

/// Routed layers build with stacked experts (the old caller swapped the
/// expert rows and columns) and the renormalization scratch is seeded.
#[test]
fn moe_model_carries_stacked_experts_and_renorm_scratch() {
    let model = HexagonLfmModel::from_gguf_on(backend(), lfm2_gguf(true), 64).unwrap();
    assert!(model.has_moe);
    let so = &model.scratch_offsets;
    assert_ne!(so.moe_renorm, 0);
    let floor = unsafe {
        *(model
            .scratch_buf
            .as_ptr()
            .add(so.moe_renorm + 7 * MOE_RENORM_SLOT_BYTES + 4) as *const f32)
    };
    assert_eq!(floor, MOE_DENOM_FLOOR);
    let moe_layers = model
        .layers
        .iter()
        .filter(|l| match l {
            HexagonLayer::Attention(a) => matches!(a.ffn, HexagonFfn::Moe(_)),
            HexagonLayer::Conv(c) => matches!(c.ffn, HexagonFfn::Moe(_)),
            HexagonLayer::DeltaNet(_) => false,
        })
        .count();
    assert_eq!(moe_layers, 2);
    // Stacked expert dims: gate/up map HS -> EXPERT_FF, down maps back. A
    // swapped rows/cols planning shows up as swapped in/out dims here.
    for layer in &model.layers {
        let ffn = match layer {
            HexagonLayer::Attention(a) => &a.ffn,
            HexagonLayer::Conv(c) => &c.ffn,
            HexagonLayer::DeltaNet(_) => continue,
        };
        let HexagonFfn::Moe(m) = ffn else { continue };
        for w in [&m.gate, &m.up] {
            assert_eq!((w.in_dim, w.out_dim, w.n_expert), (HS, EXPERT_FF, N_EXPERT));
        }
        assert_eq!(
            (m.down.in_dim, m.down.out_dim, m.down.n_expert),
            (EXPERT_FF, HS, N_EXPERT)
        );
        assert_eq!((m.n_expert, m.expert_ff_len), (N_EXPERT, EXPERT_FF));
    }
    // Each routed token's chain ends in a Div by the clamped sum.
    let mut state =
        InferenceState::from_config_capped(&model.config, &KvCompression::None, 64).unwrap();
    op_capture::take();
    model
        .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, 0, &mut state)
        .unwrap();
    let text = op_capture::take().join("");
    assert_eq!(text.lines().filter(|l| l.starts_with("Div ")).count(), 2);
    assert_eq!(
        text.lines().filter(|l| l.starts_with("UnaryRelu ")).count(),
        2
    );
}

/// The real Qwen 3.5 0.8B GGUF (llama.cpp tensor names such as
/// `post_attention_norm`, K-quants requantized for the DSP) plans and copies
/// through `from_qwen35_model_on` on the fake device. The synthetic goldens
/// use hand-picked names, so only this catches a naming drift from real files.
/// Skips without the fixture; fails under `CERA_REQUIRE_MODEL`.
#[cfg(feature = "mmap")]
#[test]
fn real_qwen35_gguf_builds_on_the_fake_device() {
    let home = std::env::var("HOME").expect("HOME unset");
    let path = std::path::PathBuf::from(home)
        .join(".leap/models/Qwen3.5-0.8B-Q4_K_M/Qwen3.5-0.8B-Q4_K_M.gguf");
    if !crate::model::transformer::require_model_or_skip(&path) {
        return;
    }
    let cpu =
        crate::model::qwen35::Qwen35Model::from_gguf(GgufFile::open(&path).unwrap(), 64).unwrap();
    let model = HexagonLfmModel::from_qwen35_model_on(backend(), &cpu, 64).unwrap();
    assert!(model.has_deltanet);
    assert_eq!(model.config.n_layers, cpu.config().n_layers);
}

fn moe_of(layer: &HexagonLayer) -> Option<&HexagonMoeFfn> {
    let ffn = match layer {
        HexagonLayer::Attention(a) => &a.ffn,
        HexagonLayer::Conv(c) => &c.ffn,
        HexagonLayer::DeltaNet(_) => return None,
    };
    match ffn {
        HexagonFfn::Moe(m) => Some(m),
        HexagonFfn::Dense(_) => None,
    }
}

/// Paging moves only the DSP mapping: every expert byte is where the
/// unpaged layout puts it, relative to the layer's own buffer, and a pass over
/// the model flushes once per window rotation instead of once per layer.
#[test]
fn paged_experts_hold_the_same_bytes_and_rotate_through_a_window() {
    use crate::backend::hexagon::sys::fake::{self, Event};
    let file = lfm2_gguf_layers(true, 8);
    let roomy = Paging {
        budget: 1 << 30,
        ..Paging::default()
    };
    let unpaged = HexagonLfmModel::from_gguf_on_with(backend(), file.clone(), 64, roomy).unwrap();
    assert!(unpaged.pager.is_none(), "a model that fits is not paged");

    // Forced paging with room for everything pins every layer: this reveals
    // the sizes the tight budget below is built from.
    let all_pinned = HexagonLfmModel::from_gguf_on_with(
        backend(),
        file.clone(),
        64,
        Paging {
            force: true,
            ..roomy
        },
    )
    .unwrap();
    let pager = all_pinned.pager.as_ref().expect("forced paging");
    let groups: Vec<usize> = (1..8)
        .map(|i| moe_of(&all_pinned.layers[i]).unwrap().gate.group.unwrap())
        .collect();
    assert_eq!(groups, (0..7).collect::<Vec<_>>());
    assert_eq!(pager.pinned(), 7);
    let layer_bytes = pager.buf(0).size();
    let resident = pager.mapped_bytes() - 7 * layer_bytes;

    // Room for the resident mappings and a two-layer window, nothing to pin.
    let tight = Paging {
        force: true,
        budget: resident + 2 * layer_bytes + layer_bytes / 2,
        ..Paging::default()
    };
    // Distinct fds let the event log tell the buffers apart (the goldens do
    // not hash this test's batches). A fresh backend resets the fake, so this
    // comes after it.
    let backend = backend();
    fake::with(|s| s.distinct_fds = true);
    let paged = HexagonLfmModel::from_gguf_on_with(backend, file, 64, tight).unwrap();
    let pager = paged.pager.as_ref().unwrap();
    assert_eq!(pager.pinned(), 0);

    // Nothing the paged model reads is mapped until it is paged in.
    let moe = paged.layers.iter().find_map(moe_of).unwrap();
    let err = paged
        .stacked_buf(&moe.gate)
        .err()
        .expect("an unmapped expert buffer is an error, not a fault on the DSP");
    assert!(err.to_string().contains("unmapped"), "{err}");

    for (u, p) in unpaged.layers.iter().zip(&paged.layers) {
        let (Some(u), Some(p)) = (moe_of(u), moe_of(p)) else {
            continue;
        };
        for (uw, pw) in [(&u.gate, &p.gate), (&u.up, &p.up), (&u.down, &p.down)] {
            assert_eq!((uw.size, uw.expert_stride), (pw.size, pw.expert_stride));
            let want = &unpaged.weights_buf.as_slice()[uw.offset..uw.offset + uw.size];
            let got = &pager.buf(pw.group.unwrap()).as_slice()[pw.offset..pw.offset + pw.size];
            assert!(want == got, "expert bytes differ in group {:?}", pw.group);
        }
    }

    let mut state = InferenceState::from_config_capped(
        &paged.config,
        &KvCompression::None,
        paged.config.max_seq_len,
    )
    .unwrap();
    let _ = op_capture::take();
    paged
        .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, 0, &mut state)
        .unwrap();
    let decode_flushes = op_capture::take().len();
    // The production adapter tells the DSP to let go of a buffer before the
    // host unmaps it, every time.
    let mut released = std::collections::HashSet::new();
    let mut unmaps = 0;
    for e in fake::events() {
        match e {
            Event::Release(fd) => {
                released.insert(fd);
            }
            Event::Unmap(fd) => {
                assert!(released.remove(&fd), "fd {fd} unmapped without a release");
                unmaps += 1;
            }
            _ => {}
        }
    }
    assert_eq!(unmaps, 6, "three rotations of a two-layer window");
    // 7 routed layers through a window of 2: layers 3, 5 and 7 each rotate.
    assert_eq!(pager.stats().page_ins, 7);
    assert_eq!(pager.stats().rotations, 3);
    assert_eq!(
        decode_flushes,
        3 + 1,
        "one flush per rotation plus the last"
    );

    let tokens: Vec<u32> = (0..16).collect();
    paged
        .try_forward_prefill_chunk(&tokens, 1, &mut state)
        .unwrap();
    // The pass starts with layers 6 and 7 still mapped; layer 1 rotates.
    assert!(pager.stats().rotations > 3);

    // Dropping the model lets the DSP let go of the pager's still-mapped window
    // before the host unmaps it, like every other buffer. The device's own
    // staging buffer (the first mapping) is released with the device.
    let staging = fake::events()
        .iter()
        .find_map(|e| match e {
            Event::Map(fd) => Some(*fd),
            _ => None,
        })
        .unwrap();
    let at_drop = fake::events().len();
    drop(paged);
    let mut released = std::collections::HashSet::new();
    let mut unmapped = 0;
    for e in &fake::events()[at_drop..] {
        match e {
            Event::Release(fd) => {
                released.insert(*fd);
            }
            Event::Unmap(fd) if *fd != staging => {
                assert!(released.contains(fd), "fd {fd} unmapped without a release");
                unmapped += 1;
            }
            _ => {}
        }
    }
    assert!(
        unmapped > 4,
        "the model's buffers and the window: {unmapped}"
    );
}

/// `--device auto` refuses a routed model that would have to be paged (the
/// CPU is faster) without ever opening the device, loads one that fits, and
/// pages once paging is asked for.
#[test]
fn auto_refuses_paging_unless_asked() {
    let file = lfm2_gguf_layers(true, 8);
    let roomy = Paging {
        budget: 1 << 30,
        ..Paging::default()
    };
    // Fits: auto loads it, unpaged.
    let m =
        HexagonLfmModel::from_gguf_auto_on(|| Ok(()), || Ok(backend()), file.clone(), 64, roomy)
            .unwrap();
    assert!(m.pager.is_none());

    // A budget the weights cannot meet needs paging: refused from the GGUF
    // alone, before the opener runs.
    let tight = Paging {
        budget: 1 << 20,
        ..Paging::default()
    };
    let opened = std::cell::Cell::new(false);
    let err = HexagonLfmModel::from_gguf_auto_on(
        || Ok(()),
        || {
            opened.set(true);
            Ok(backend())
        },
        file.clone(),
        64,
        tight,
    )
    .err()
    .expect("auto must not page");
    let msg = err.to_string();
    assert!(msg.contains("CERA_HEXAGON_PAGE_EXPERTS"), "{msg}");
    assert!(msg.contains("--device hexagon") && msg.contains("MiB DSP mapping budget"));
    assert!(!opened.get(), "refused before opening the device");

    // With no NPU (no driver, or the kill switch) the caller sees that error,
    // not a paging notice that would send the user after the wrong knob.
    let err = HexagonLfmModel::from_gguf_auto_on(
        || Err(CeraError::Backend("no NPU here".into())),
        || Ok(backend()),
        file.clone(),
        64,
        tight,
    )
    .err()
    .expect("no device");
    assert!(err.to_string().contains("no NPU here"), "{err}");

    // The late check still guards a borderline model that got that far.
    let err = HexagonLfmModel::from_gguf_on_with(backend(), file.clone(), 64, tight.for_auto())
        .err()
        .expect("late check");
    assert!(
        err.to_string().contains("CERA_HEXAGON_PAGE_EXPERTS"),
        "{err}"
    );

    // The opt-in pages it under auto; an explicit load pages it too.
    let forced = Paging {
        force: true,
        ..roomy
    };
    let m =
        HexagonLfmModel::from_gguf_auto_on(|| Ok(()), || Ok(backend()), file.clone(), 64, forced)
            .unwrap();
    assert!(m.pager.is_some());
    let m = HexagonLfmModel::from_gguf_on_with(backend(), file, 64, forced).unwrap();
    assert!(m.pager.is_some(), "an explicit NPU load pages");
}

/// A weight planned for an expert group on a model with no pager is an error,
/// not a read of the shared buffer.
#[test]
fn grouped_weight_without_a_pager_is_an_error() {
    let file = lfm2_gguf_layers(true, 4);
    let m = HexagonLfmModel::from_gguf_on(backend(), file, 64).unwrap();
    assert!(m.pager.is_none());
    let moe = m.layers.iter().find_map(moe_of).unwrap();
    let grouped = HexagonStackedWeight {
        group: Some(0),
        ..moe.gate
    };
    let err = m.stacked_buf(&grouped).err().expect("no pager for group 0");
    assert!(err.to_string().contains("no pager"), "{err}");
}

/// A prefill chunk whose routed rows would not fit one staging buffer is
/// flushed between rows instead of failing: with the staging buffer pretended
/// to be tiny, the chunk still runs, in several batches each within it.
#[test]
fn long_routed_prefill_chunk_flushes_between_rows() {
    let file = lfm2_gguf_layers(true, 4);
    let model = HexagonLfmModel::from_gguf_on(backend(), file, 64).unwrap();
    let cap = 128 * 1024;
    model
        .device
        .lock()
        .unwrap()
        .queue_session_mut()
        .set_staging_capacity_for_test(Some(cap));
    let mut state = InferenceState::from_config_capped(
        &model.config,
        &KvCompression::None,
        model.config.max_seq_len,
    )
    .unwrap();
    let tokens: Vec<u32> = (0..60).collect();
    let _ = op_capture::take();
    model
        .try_forward_prefill_chunk(&tokens, 0, &mut state)
        .expect("routed rows must flush between rows once the batch is half full");
    let batches = op_capture::take();
    assert!(
        batches.len() > 3,
        "expected several flushes, got {}",
        batches.len()
    );
}

/// A dropped model lets the DSP let go of each buffer a batch read before the host
/// unmaps it. Without that the unmap fails on a device and the mapping leaks
/// (four debug lines at every teardown).
#[test]
fn dropping_a_model_releases_every_mapped_buffer_before_unmapping_it() {
    use crate::backend::hexagon::sys::fake::{self, Event};
    let backend = backend();
    fake::with(|s| s.distinct_fds = true);
    // The device's own staging buffer is mapped before the model exists.
    let before = fake::events().len();
    let model = HexagonLfmModel::from_llama_on(backend, dense_gguf(), None, 64).unwrap();
    let mapped: std::collections::HashSet<i32> = fake::events()[before..]
        .iter()
        .filter_map(|e| match e {
            Event::Map(fd) => Some(*fd),
            _ => None,
        })
        .collect();
    assert!(
        mapped.len() >= 4,
        "weights, KV state, scratch and mask: {mapped:?}"
    );
    let at_drop = fake::events().len();
    drop(model);
    let mut released = std::collections::HashSet::new();
    let mut unmapped = std::collections::HashSet::new();
    for e in &fake::events()[at_drop..] {
        match e {
            Event::Release(fd) => {
                released.insert(*fd);
            }
            Event::Unmap(fd) if mapped.contains(fd) => {
                assert!(released.contains(fd), "fd {fd} unmapped without a release");
                unmapped.insert(*fd);
            }
            _ => {}
        }
    }
    assert_eq!(unmapped, mapped, "every mapping is unmapped at drop");
}

// ---- lfm3 -------------------------------------------------------------------

const L3_WINDOW: usize = 8;
const L3_TAPS: usize = 7;
/// conv, global attention, conv, sliding attention.
const L3_SLIDING: [bool; 4] = [false, false, false, true];

/// A GGUF `bool[]` value.
fn bool_array(key: &str, values: &[bool]) -> (String, KvValue) {
    let mut bytes = 7u32.to_le_bytes().to_vec(); // element type: BOOL
    bytes.extend((values.len() as u64).to_le_bytes());
    bytes.extend(values.iter().map(|&v| v as u8));
    (key.into(), KvValue::Raw(9, bytes)) // value type: ARRAY
}

/// A tiny `lfm3`: sandwich norms on every layer, a 7-tap short conv, a per-head
/// attention gate, a global layer (partial rotary through `rope_freqs.weight`) and
/// a sliding-window layer with its own RoPE base. Sized like the real 1.7B's
/// pattern, not its width.
fn lfm3_gguf() -> GgufFile {
    lfm3_gguf_with(L3_TAPS, &|_, _| {})
}

/// An edit applied to an lfm3 GGUF's metadata and tensors before it is built.
type Lfm3Edit = dyn Fn(&mut Vec<(String, KvValue)>, &mut Vec<Tensor>);

/// [`lfm3_gguf`] with a chosen conv width and a last chance to edit the metadata and
/// tensors before the file is built.
fn lfm3_gguf_with(taps: usize, edit: &Lfm3Edit) -> GgufFile {
    let mut g = Gen(0x1f3a_7c15);
    let q_dim = N_HEADS * HEAD_DIM;
    let mut t = vec![
        g.f32("token_embd.weight", &[HS, VOCAB], 1.0, 0.0),
        g.f32("output_norm.weight", &[HS], 0.1, 1.0),
    ];
    // global layers rotate the first half of each head's pairs; the rest carry 1e30
    let pairs = HEAD_DIM / 2;
    let mut factors = vec![1.0f32; pairs / 2];
    factors.resize(pairs, 1e30);
    t.push(Tensor {
        name: "rope_freqs.weight".into(),
        dims: vec![pairs],
        ggml_type: 0,
        data: factors.iter().flat_map(|f| f.to_le_bytes()).collect(),
    });
    for l in 0..4 {
        let n = |s: &str| format!("blk.{l}.{s}");
        for norm in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
        ] {
            t.push(g.f32(&n(norm), &[HS], 0.1, 1.0));
        }
        if l % 2 == 0 {
            t.push(g.q4_0(&n("shortconv.in_proj.weight"), &[HS, 3 * HS]));
            t.push(g.q4_0(&n("shortconv.out_proj.weight"), &[HS, HS]));
            t.push(g.f32(&n("shortconv.conv.weight"), &[taps, HS], 0.3, 0.0));
        } else {
            t.push(g.q4_0(&n("attn_q.weight"), &[HS, q_dim]));
            t.push(g.q4_0(&n("attn_k.weight"), &[HS, HEAD_DIM]));
            t.push(g.q4_0(&n("attn_v.weight"), &[HS, HEAD_DIM]));
            t.push(g.q4_0(&n("attn_output.weight"), &[q_dim, HS]));
            t.push(g.f32(&n("attn_q_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
            t.push(g.f32(&n("attn_k_norm.weight"), &[HEAD_DIM], 0.1, 1.0));
            t.push(g.q8_0(&n("attn_gate.weight"), &[HS, N_HEADS]));
        }
        t.push(g.q4_0(&n("ffn_gate.weight"), &[HS, INTER]));
        t.push(g.q4_0(&n("ffn_up.weight"), &[HS, INTER]));
        t.push(g.q4_0(&n("ffn_down.weight"), &[INTER, HS]));
    }
    let p = "lfm3";
    let mut kv = vec![
        ("general.architecture".into(), KvValue::Str(p.into())),
        u32kv("block_count", p, 4),
        u32kv("embedding_length", p, HS as u32),
        u32kv("feed_forward_length", p, INTER as u32),
        u32kv("attention.head_count", p, N_HEADS as u32),
        (
            format!("{p}.attention.head_count_kv"),
            KvValue::I32Array(vec![0, 1, 0, 1]),
        ),
        u32kv("attention.key_length", p, HEAD_DIM as u32),
        u32kv("attention.value_length", p, HEAD_DIM as u32),
        u32kv("rope.dimension_count", p, HEAD_DIM as u32),
        u32kv("shortconv.l_cache", p, taps as u32),
        (
            format!("{p}.attention.layer_norm_rms_epsilon"),
            KvValue::F32(1e-5),
        ),
        (format!("{p}.rope.freq_base"), KvValue::F32(1_000_000.0)),
        (format!("{p}.rope.freq_base_swa"), KvValue::F32(10_000.0)),
        u32kv("attention.sliding_window", p, L3_WINDOW as u32),
        bool_array(
            &format!("{p}.attention.sliding_window_pattern"),
            &L3_SLIDING,
        ),
        u32kv("context_length", p, 64),
        u32kv("vocab_size", p, VOCAB as u32),
    ];
    edit(&mut kv, &mut t);
    gguf(kv, t)
}

/// Set (or, with `None`, remove) the metadata `key`.
fn set_kv(kv: &mut Vec<(String, KvValue)>, key: &str, value: Option<KvValue>) {
    kv.retain(|(k, _)| k != key);
    kv.extend(value.map(|v| (key.to_string(), v)));
}

/// One malformed-tensor case: label, expected error text, the edit that makes the file.
type ShapeCase = (&'static str, &'static str, Box<Lfm3Edit>);

fn resize(t: &mut [Tensor], name: &str, dims: Vec<usize>) {
    let x = t.iter_mut().find(|x| x.name == name).unwrap();
    x.ggml_type = 0; // F32
    x.data = vec![0u8; dims.iter().product::<usize>() * 4];
    x.dims = dims;
}

/// Every lfm3 tensor the graph sizes by the config: a wide projection or a long norm used to
/// load and then panic in the first forward (`destination buffer underflow`), and a short
/// norm made `rmsnorm` return early (plausible, wrong logits). Both loaders refuse them.
fn lfm3_shape_cases() -> Vec<ShapeCase> {
    vec![
        (
            "an attn_v wider than the head width, with no value_length key",
            "blk.1.attn_v.weight is",
            Box::new(|kv, t| {
                set_kv(kv, "lfm3.attention.value_length", None);
                resize(t, "blk.1.attn_v.weight", vec![HS, HEAD_DIM * 2]);
            }),
        ),
        (
            "an attention gate with a head too many",
            "attn_gate.weight is",
            Box::new(|_, t| resize(t, "blk.1.attn_gate.weight", vec![HS, N_HEADS + 1])),
        ),
        (
            "an epsilon stored as an integer",
            "layer_norm_rms_epsilon must be stored as an f32",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.attention.layer_norm_rms_epsilon",
                    Some(KvValue::U32(1)),
                )
            }),
        ),
        (
            "a post norm in a dtype that cannot be read as f32",
            "blk.1.post_ffw_norm.weight cannot be read as f32",
            Box::new(|_, t| {
                let x = t
                    .iter_mut()
                    .find(|x| x.name == "blk.1.post_ffw_norm.weight")
                    .unwrap();
                x.ggml_type = 26; // I32: parses, but has no f32 reader
            }),
        ),
        (
            "a post norm of the wrong length",
            "entries, expected",
            Box::new(|_, t| resize(t, "blk.1.post_ffw_norm.weight", vec![HS + 1])),
        ),
        (
            "an untied output head of another vocabulary",
            "output.weight is",
            Box::new(|_, t| {
                t.push(Tensor {
                    name: "output.weight".into(),
                    dims: vec![HS, VOCAB + 1],
                    ggml_type: 0,
                    data: vec![0u8; HS * (VOCAB + 1) * 4],
                })
            }),
        ),
        (
            "a short q_norm",
            "blk.1.attn_q_norm.weight has",
            Box::new(|_, t| resize(t, "blk.1.attn_q_norm.weight", vec![HEAD_DIM - 1])),
        ),
        (
            "a short attention norm",
            "blk.0.attn_norm.weight has",
            Box::new(|_, t| resize(t, "blk.0.attn_norm.weight", vec![HS - 1])),
        ),
        (
            "a long ffn norm",
            "blk.0.ffn_norm.weight has",
            Box::new(|_, t| resize(t, "blk.0.ffn_norm.weight", vec![HS + 1])),
        ),
        (
            "a short output norm",
            "output_norm.weight has",
            Box::new(|_, t| resize(t, "output_norm.weight", vec![HS - 1])),
        ),
        (
            "an attn_q of the wrong height",
            "blk.1.attn_q.weight is",
            Box::new(|_, t| resize(t, "blk.1.attn_q.weight", vec![HS, N_HEADS * HEAD_DIM + 1])),
        ),
        (
            "an attn_k of the wrong height",
            "blk.1.attn_k.weight is",
            Box::new(|_, t| resize(t, "blk.1.attn_k.weight", vec![HS, HEAD_DIM * 2])),
        ),
        (
            "an attn_output of the wrong width",
            "blk.1.attn_output.weight is",
            Box::new(|_, t| {
                resize(
                    t,
                    "blk.1.attn_output.weight",
                    vec![N_HEADS * HEAD_DIM + 1, HS],
                )
            }),
        ),
        (
            "a short k_norm",
            "blk.1.attn_k_norm.weight has",
            Box::new(|_, t| resize(t, "blk.1.attn_k_norm.weight", vec![HEAD_DIM - 1])),
        ),
        (
            "a long post_attention_norm",
            "blk.0.post_attention_norm.weight has",
            Box::new(|_, t| resize(t, "blk.0.post_attention_norm.weight", vec![HS + 1])),
        ),
        (
            "an ffn_gate with too many rows",
            "blk.2.ffn_gate.weight is",
            Box::new(|_, t| resize(t, "blk.2.ffn_gate.weight", vec![HS, INTER + 1])),
        ),
        (
            "an ffn_up with too few rows",
            "blk.2.ffn_up.weight is",
            Box::new(|_, t| resize(t, "blk.2.ffn_up.weight", vec![HS, INTER - 1])),
        ),
        (
            "an ffn_down with too few columns",
            "blk.1.ffn_down.weight is",
            Box::new(|_, t| resize(t, "blk.1.ffn_down.weight", vec![INTER, HS / 2])),
        ),
        (
            "a conv in_proj of 2 hidden widths",
            "shortconv.in_proj.weight is",
            Box::new(|_, t| resize(t, "blk.0.shortconv.in_proj.weight", vec![HS, 2 * HS])),
        ),
        (
            "a conv out_proj with extra rows",
            "shortconv.out_proj.weight is",
            Box::new(|_, t| resize(t, "blk.0.shortconv.out_proj.weight", vec![HS, HS + 1])),
        ),
        (
            "an embedding with extra rows",
            "token_embd.weight is",
            Box::new(|_, t| resize(t, "token_embd.weight", vec![HS, VOCAB + 3])),
        ),
    ]
}

#[test]
fn the_cpu_loader_checks_lfm3_tensor_shapes() {
    for (label, expect, edit) in lfm3_shape_cases() {
        let err = match crate::model::lfm2::LfmModel::from_gguf(lfm3_gguf_with(L3_TAPS, &*edit), 64)
        {
            Ok(_) => panic!("{label}: the file was accepted"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(expect), "{label}: {err}");
    }
}

/// The NPU constructor plans its planes from "at least" lengths, so without the shared
/// validator it read the wrong weights on the device with no error.
#[test]
fn the_hexagon_loader_checks_lfm3_tensor_shapes() {
    for (label, expect, edit) in lfm3_shape_cases() {
        let err =
            match HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf_with(L3_TAPS, &*edit), 64) {
                Ok(_) => panic!("{label}: the file was accepted"),
                Err(e) => e.to_string(),
            };
        assert!(err.contains(expect), "{label}: {err}");
    }
}

/// `InferenceState.lora` is a public field: a direct caller can attach a classifier adapter
/// to an lfm3 model past every entry guard (`Session`, `CeraEngine::detect_pii_with_lora`).
/// A classifier adapter would switch an LFM2 model to its bidirectional branches, which are
/// written for three conv taps and know nothing of lfm3's sandwich norms, gate or windows
/// (hidden states differed by 4.7 on a quantized checkpoint), so lfm3 stays causal.
#[test]
fn an_lfm3_model_stays_causal_whatever_the_state_holds() {
    use crate::model::Model;
    let model = crate::model::lfm2::LfmModel::from_gguf(lfm3_gguf(), 64).unwrap();
    let tokens: Vec<u32> = (1..9).collect();
    let run = |lora: Option<Arc<crate::lora::LoraAdapterWeights>>| {
        let mut state = InferenceState::from_config(crate::model::Model::config(&model)).unwrap();
        state.lora = lora;
        model.hidden_states(&tokens, &mut state)
    };
    let classifier = crate::lora::LoraAdapterWeights::new_classifier_for_testing(
        vec![0.0; 2 * HS],
        None,
        vec!["O".into(), "X".into()],
    );
    assert_eq!(run(None), run(Some(classifier)));
}

/// The lfm3 loader refuses a file that would otherwise run with the wrong attention
/// width, window, rotary table or conv width. Each case is one inconsistency and the
/// text that names it; a guard that stopped firing would load a plausible model.
#[test]
fn lfm3_loader_refuses_inconsistent_files() {
    assert!(HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).is_ok());
    // Public LFM3 GGUFs omit `lfm3.vocab_size` and derive it from `token_embd.weight`.
    let no_vocab_meta = || lfm3_gguf_with(L3_TAPS, &|kv, _| set_kv(kv, "lfm3.vocab_size", None));
    assert_eq!(
        crate::model::lfm2::LfmModel::parse_config(&no_vocab_meta(), 64)
            .unwrap()
            .vocab_size,
        VOCAB
    );
    assert_eq!(
        HexagonLfmModel::from_gguf_on(backend(), no_vocab_meta(), 64)
            .unwrap()
            .config
            .vocab_size,
        VOCAB
    );
    // both ends of the accepted conv width range (the refusal side is the "a conv wider than the activation scratch" row below)
    for taps in [2, 9] {
        HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf_with(taps, &|_, _| {}), 64)
            .unwrap_or_else(|e| panic!("a {taps}-tap conv is refused: {e}"));
    }
    type Edit = Box<Lfm3Edit>;
    let pairs = HEAD_DIM / 2;
    let cases: Vec<(&str, usize, &str, Edit)> = vec![
        (
            "a vocab_size stored as an i32",
            L3_TAPS,
            "lfm3.vocab_size must be stored as a u32",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.vocab_size",
                    Some(KvValue::Raw(5, (VOCAB as i32).to_le_bytes().to_vec())),
                )
            }),
        ),
        (
            "a zero vocab_size",
            L3_TAPS,
            "lfm3.vocab_size must be > 0",
            Box::new(|kv, _| set_kv(kv, "lfm3.vocab_size", Some(KvValue::U32(0)))),
        ),
        (
            "head_count 0",
            L3_TAPS,
            "head_count must be positive",
            Box::new(|kv, _| set_kv(kv, "lfm3.attention.head_count", Some(KvValue::U32(0)))),
        ),
        (
            "a head gate wider than the hidden scratch",
            L3_TAPS,
            "do not fit the gate scratch",
            // 65 heads pad to 96 gate outputs per row, above the 64 values a row of the scratch
            // holds, while q_dim (2080) is far above both: the bound is the hidden size
            Box::new(|kv, t| {
                set_kv(kv, "lfm3.attention.head_count", Some(KvValue::U32(65)));
                for l in [1, 3] {
                    resize(
                        t,
                        &format!("blk.{l}.attn_q.weight"),
                        vec![HS, 65 * HEAD_DIM],
                    );
                    resize(
                        t,
                        &format!("blk.{l}.attn_output.weight"),
                        vec![65 * HEAD_DIM, HS],
                    );
                    resize(t, &format!("blk.{l}.attn_gate.weight"), vec![HS, 65]);
                }
            }),
        ),
        (
            "head_count over the cap",
            L3_TAPS,
            "exceeds the supported",
            Box::new(|kv, _| set_kv(kv, "lfm3.attention.head_count", Some(KvValue::U32(129)))),
        ),
        (
            "key_length * head_count not a multiple of 32",
            L3_TAPS,
            "must be a multiple of 32",
            Box::new(move |kv, _| set_kv(kv, "lfm3.attention.key_length", Some(KvValue::U32(6)))),
        ),
        (
            "an odd key_length",
            L3_TAPS,
            "must be even",
            Box::new(|kv, _| {
                // 64 heads of width 1: the multiple-of-32 check passes, evenness is what refuses
                set_kv(
                    kv,
                    "lfm3.attention.head_count",
                    Some(KvValue::U32(HS as u32)),
                );
                set_kv(kv, "lfm3.attention.key_length", Some(KvValue::U32(1)));
            }),
        ),
        (
            "a value_length that differs from key_length",
            L3_TAPS,
            "attention.value_length must equal",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.attention.value_length",
                    Some(KvValue::U32((HEAD_DIM * 2) as u32)),
                )
            }),
        ),
        (
            "mismatched post_attention_norm and post_ffw_norm",
            L3_TAPS,
            "must both be present or both absent",
            Box::new(|_, t| {
                t.retain(|x| x.name != "blk.0.post_attention_norm.weight");
            }),
        ),
        (
            "routed experts disagreeing with leading_dense_block_count",
            L3_TAPS,
            "leading_dense_block_count",
            Box::new(|kv, t| {
                for (key, v) in [
                    ("leading_dense_block_count", 2u32),
                    ("expert_gating_func", 2),
                    ("expert_count", 4),
                    ("expert_used_count", 2),
                    ("expert_feed_forward_length", 32),
                ] {
                    set_kv(kv, &format!("lfm3.{key}"), Some(KvValue::U32(v)));
                }
                t.push(Tensor {
                    name: "blk.0.ffn_gate_exps.weight".into(),
                    dims: vec![HS],
                    ggml_type: 0,
                    data: vec![0u8; HS * 4],
                })
            }),
        ),
        (
            "a classifier head",
            L3_TAPS,
            "causal language-model files only",
            Box::new(|_, t| {
                t.push(Tensor {
                    name: "classifier.weight".into(),
                    dims: vec![HS],
                    ggml_type: 0,
                    data: vec![0u8; HS * 4],
                })
            }),
        ),
        (
            "a classifier head next to an explicit causal flag",
            L3_TAPS,
            "causal language-model files only",
            // GGUF value type 7 is BOOL: an explicit `true` skips the classifier default
            Box::new(|kv, t| {
                set_kv(kv, "lfm3.attention.causal", Some(KvValue::Raw(7, vec![1])));
                t.push(Tensor {
                    name: "classifier.weight".into(),
                    dims: vec![HS],
                    ggml_type: 0,
                    data: vec![0u8; HS * 4],
                })
            }),
        ),
        (
            "is_causal and attention.causal that disagree",
            L3_TAPS,
            "disagree",
            Box::new(|kv, _| {
                // GGUF value type 7 is BOOL
                set_kv(kv, "lfm3.is_causal", Some(KvValue::Raw(7, vec![1])));
                set_kv(kv, "lfm3.attention.causal", Some(KvValue::Raw(7, vec![0])));
            }),
        ),
        (
            "a key_length stored as an i32",
            L3_TAPS,
            "must be stored as a u32",
            // GGUF value type 5 is INT32
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.attention.key_length",
                    Some(KvValue::Raw(5, (HEAD_DIM as i32).to_le_bytes().to_vec())),
                )
            }),
        ),
        (
            "a rope.dimension_count stored as an i32",
            L3_TAPS,
            "must be stored as a u32",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.rope.dimension_count",
                    Some(KvValue::Raw(
                        5,
                        (HEAD_DIM as i32 / 2).to_le_bytes().to_vec(),
                    )),
                )
            }),
        ),
        (
            "a rope_freqs.weight factor that is a subnormal",
            L3_TAPS,
            "non-finite factor or one below",
            Box::new(|_, t| {
                let ff = t
                    .iter_mut()
                    .find(|x| x.name == "rope_freqs.weight")
                    .unwrap();
                ff.data[..4].copy_from_slice(&f32::from_bits(1).to_le_bytes());
            }),
        ),
        (
            "a causal key stored as a non-bool",
            L3_TAPS,
            "must be stored as a bool",
            // GGUF value type 0 is UINT8: it reads as absent through get_bool
            Box::new(|kv, _| set_kv(kv, "lfm3.attention.causal", Some(KvValue::Raw(0, vec![1])))),
        ),
        (
            "a bidirectional file",
            L3_TAPS,
            "causal language-model files only",
            // GGUF value type 7 is BOOL
            Box::new(|kv, _| set_kv(kv, "lfm3.attention.causal", Some(KvValue::Raw(7, vec![0])))),
        ),
        (
            "partial rope.dimension_count",
            L3_TAPS,
            "rope.dimension_count (16) must equal attention.key_length (32)",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.rope.dimension_count",
                    Some(KvValue::U32((HEAD_DIM / 2) as u32)),
                )
            }),
        ),
        (
            "a window without a per-layer pattern",
            L3_TAPS,
            "sliding_window_pattern",
            Box::new(|kv, _| set_kv(kv, "lfm3.attention.sliding_window_pattern", None)),
        ),
        (
            "a pattern shorter than block_count",
            L3_TAPS,
            "sliding_window_pattern has",
            Box::new(|kv, _| {
                let (k, v) = bool_array("lfm3.attention.sliding_window_pattern", &[false; 2]);
                set_kv(kv, &k, Some(v));
            }),
        ),
        (
            "a global rope base of 1",
            L3_TAPS,
            "lfm3.rope.freq_base must be",
            Box::new(|kv, _| set_kv(kv, "lfm3.rope.freq_base", Some(KvValue::F32(1.0)))),
        ),
        (
            "an infinite global rope base",
            L3_TAPS,
            "lfm3.rope.freq_base must be",
            Box::new(|kv, _| set_kv(kv, "lfm3.rope.freq_base", Some(KvValue::F32(f32::INFINITY)))),
        ),
        (
            "an infinite sliding-layer rope base",
            L3_TAPS,
            "freq_base_swa",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.rope.freq_base_swa",
                    Some(KvValue::F32(f32::INFINITY)),
                )
            }),
        ),
        (
            "a zero sliding window",
            L3_TAPS,
            "sliding_window must be > 0",
            Box::new(|kv, _| set_kv(kv, "lfm3.attention.sliding_window", Some(KvValue::U32(0)))),
        ),
        (
            "a sliding window stored as i32",
            L3_TAPS,
            "must be stored as a u32",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.attention.sliding_window",
                    Some(KvValue::Raw(5, 64i32.to_le_bytes().to_vec())),
                )
            }),
        ),
        (
            "an i32 sliding window with no pattern",
            L3_TAPS,
            "must be stored as a u32",
            Box::new(|kv, _| {
                set_kv(kv, "lfm3.attention.sliding_window_pattern", None);
                set_kv(
                    kv,
                    "lfm3.attention.sliding_window",
                    Some(KvValue::Raw(5, 64i32.to_le_bytes().to_vec())),
                )
            }),
        ),
        (
            "a sliding rope base stored as f64",
            L3_TAPS,
            "must be stored as an f32",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.rope.freq_base_swa",
                    Some(KvValue::Raw(12, 500f64.to_le_bytes().to_vec())),
                )
            }),
        ),
        (
            "a global rope base stored as f64",
            L3_TAPS,
            "must be stored as an f32",
            Box::new(|kv, _| {
                set_kv(
                    kv,
                    "lfm3.rope.freq_base",
                    Some(KvValue::Raw(12, 1e6f64.to_le_bytes().to_vec())),
                )
            }),
        ),
        (
            "a rope factor so small that position / factor overflows",
            L3_TAPS,
            "non-finite factor or one below",
            Box::new(|_, t| {
                let ff = t
                    .iter_mut()
                    .find(|x| x.name == "rope_freqs.weight")
                    .unwrap();
                ff.data[..4].copy_from_slice(&1e-12f32.to_le_bytes());
            }),
        ),
        (
            "a NaN global rope base",
            L3_TAPS,
            "lfm3.rope.freq_base must be",
            Box::new(|kv, _| set_kv(kv, "lfm3.rope.freq_base", Some(KvValue::F32(f32::NAN)))),
        ),
        (
            "a sliding-layer rope base of 0.5",
            L3_TAPS,
            "freq_base_swa",
            Box::new(|kv, _| set_kv(kv, "lfm3.rope.freq_base_swa", Some(KvValue::F32(0.5)))),
        ),
        (
            "a conv wider than the activation scratch",
            10,
            "outside 2..=9",
            Box::new(|_, _| {}),
        ),
        (
            "a rope_freqs.weight of the wrong length",
            L3_TAPS,
            "rope_freqs.weight has",
            Box::new(move |_, t| {
                let ff = t
                    .iter_mut()
                    .find(|x| x.name == "rope_freqs.weight")
                    .unwrap();
                ff.dims = vec![pairs - 1];
                ff.data.truncate((pairs - 1) * 4);
            }),
        ),
        (
            "a rope_freqs.weight factor that is not finite",
            L3_TAPS,
            "non-finite factor or one below",
            Box::new(|_, t| {
                let ff = t
                    .iter_mut()
                    .find(|x| x.name == "rope_freqs.weight")
                    .unwrap();
                ff.data[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
            }),
        ),
    ];
    for (label, taps, expect, edit) in cases {
        let err = match HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf_with(taps, &*edit), 64) {
            Ok(_) => panic!("{label}: the file was accepted"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(expect), "{label}: {err}");
    }
}

/// The model builds, and every lfm3 extra is planned where the forward reads it.
#[test]
fn lfm3_constructor_plans_the_extras() {
    let src = lfm3_gguf();
    let model = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    assert_eq!(model.config.architecture, "lfm3");
    assert_eq!(model.layers.len(), 4);

    // a norm's planned offset holds the GGUF tensor, value for value
    let norm = |offset: usize, name: &str| {
        assert_eq!(
            weights_f32(&model, offset, HS),
            tensor_f32(&src, name),
            "{name}"
        );
    };
    let mut conv_states = Vec::new();
    for (l, layer) in model.layers.iter().enumerate() {
        match layer {
            HexagonLayer::Conv(c) => {
                assert_eq!(c.d_conv, L3_TAPS, "layer {l}");
                norm(
                    c.post_norm_offset.expect("conv post norm"),
                    &format!("blk.{l}.post_attention_norm.weight"),
                );
                norm(
                    c.ffn_post_norm_offset.expect("conv ffn post norm"),
                    &format!("blk.{l}.post_ffw_norm.weight"),
                );
                // the SsmConv taps are the GGUF `[7, C]` tensor verbatim, with no
                // per-tap planes (those exist only for the 3-tap manual chain)
                assert_eq!(
                    weights_f32(&model, c.conv_ssm_offset, L3_TAPS * HS),
                    tensor_f32(&src, &format!("blk.{l}.shortconv.conv.weight")),
                    "layer {l} taps"
                );
                assert_eq!(
                    [c.conv_w0_offset, c.conv_w1_offset, c.conv_w2_offset],
                    [0; 3]
                );
                conv_states.push(c.state_offset);
            }
            HexagonLayer::Attention(a) => {
                norm(
                    a.attn_post_norm_offset.expect("attn post norm"),
                    &format!("blk.{l}.post_attention_norm.weight"),
                );
                norm(
                    a.ffn_post_norm_offset.expect("attn ffn post norm"),
                    &format!("blk.{l}.post_ffw_norm.weight"),
                );
                // the gate is planned padded to a 32-row tile so no kernel writes past
                // a `n_heads`-wide output row
                let gate = a.head_gate.expect("per-head gate");
                assert_eq!((gate.in_dim, gate.out_dim), (HS, 32), "layer {l}");
                assert!(!a.has_q_gate, "Qwen's fused gate is a different thing");
                let sliding = L3_SLIDING[l];
                assert_eq!(a.swa, sliding, "layer {l} window");
                // sliding layers rotate the whole head with their own base and no mask;
                // global layers keep the model base and rotate on the DSP with the mask
                assert_eq!(a.rope_theta, sliding.then_some(10_000.0), "layer {l}");
                assert_eq!(a.rope_freqs_offset.is_some(), !sliding, "layer {l}");
            }
            HexagonLayer::DeltaNet(_) => panic!("lfm3 has no DeltaNet layer"),
        }
    }
    // each conv layer owns a `[C, 6]` state slab, not the `[C, 2]` of lfm2
    let slab = (L3_TAPS - 1) * align256(HS * 4);
    assert!(conv_states[1] - conv_states[0] >= slab, "{conv_states:?}");

    assert_eq!(model.dense.swa_window, Some(L3_WINDOW));
    assert!(model.dense.mask_swa.is_some());
    let freqs = model.dense.rope_freqs.as_ref().expect("rope_freqs");
    assert_eq!(freqs.len(), HEAD_DIM / 2);
    assert_eq!(freqs[0], 1.0);
    assert_eq!(freqs[HEAD_DIM / 2 - 1], 1e30);
    // every rotation is a DSP op, so the recorded decode template stays available
    assert!(!model.cpu_rope);
}

/// One tensor descriptor of a captured op: `[bN off=.. size=.. fl=.. ty=.. ne=[..] nb=[..] ..]`.
#[derive(Debug, PartialEq)]
struct Desc {
    off: usize,
    ne: [usize; 4],
    nb: [usize; 4],
}

fn field(chunk: &str, key: &str) -> Vec<usize> {
    let start = chunk
        .find(key)
        .unwrap_or_else(|| panic!("{key} in {chunk}"))
        + key.len();
    let rest = &chunk[start..];
    // a list (`ne=[..]`) runs to its closing bracket, a scalar (`off=..`) to the next space
    let end = rest
        .find(if key.ends_with('[') { ']' } else { ' ' })
        .unwrap_or(rest.len());
    rest[..end]
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

/// The source then destination descriptors of one captured op line.
fn descs(line: &str) -> (Vec<Desc>, Vec<Desc>) {
    let (srcs, dsts) = line.split_once(" dst: ").expect("dst section");
    let srcs = srcs.split_once(" src: ").expect("src section").1;
    let parse = |section: &str| -> Vec<Desc> {
        section
            .split("] [")
            .map(|chunk| {
                let ne = field(chunk, "ne=[");
                let nb = field(chunk, "nb=[");
                Desc {
                    off: field(chunk, "off=")[0],
                    ne: [ne[0], ne[1], ne[2], ne[3]],
                    nb: [nb[0], nb[1], nb[2], nb[3]],
                }
            })
            .collect()
    };
    (parse(srcs), parse(dsts))
}

fn kparams(line: &str) -> Vec<i64> {
    let start = line.find("kparams=[").expect("kparams") + "kparams=[".len();
    line[start..line[start..].find(']').unwrap() + start]
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

fn op_lines<'a>(batches: &'a [String], kind: &str) -> Vec<&'a str> {
    let prefix = format!("{kind} ");
    batches
        .iter()
        .flat_map(|b| b.lines())
        .filter(|l| l.starts_with(&prefix))
        .collect()
}

/// The ops of one lfm3 decode pass and one `m`-row prefill chunk.
fn lfm3_forward_ops(m: usize) -> (Vec<String>, Vec<String>) {
    lfm3_forward_ops_on(backend(), m)
}

fn lfm3_forward_ops_on(backend: Backend, m: usize) -> (Vec<String>, Vec<String>) {
    let model = HexagonLfmModel::from_gguf_on(backend, lfm3_gguf(), 64).unwrap();
    let mut state = InferenceState::from_config_capped(
        &model.config,
        &KvCompression::None,
        model.config.max_seq_len,
    )
    .unwrap();
    model
        .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, 0, &mut state)
        .unwrap();
    let decode = op_capture::take();
    let tokens: Vec<u32> = (0..m as u32).collect();
    model
        .try_forward_prefill_chunk(&tokens, 1, &mut state)
        .unwrap();
    (decode, op_capture::take())
}

fn count(batches: &[String], kind: &str) -> usize {
    op_lines(batches, kind).len()
}

/// `Session::new` treats a TurboQuant request on lfm3 as a warned downgrade to the plain
/// cache, so `Session::reset` must accept it too; the other architectures still refuse it
/// (the Hexagon backend has no compressed cache for them).
#[test]
fn lfm3_resets_on_a_turboquant_request_but_lfm2_refuses_it() {
    let tq = KvCompression::turboquant(1);
    let lfm3 = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    let mut state = InferenceState::from_config_capped(&lfm3.config, &tq, 64).unwrap();
    lfm3.try_reset_kv(&mut state, &tq, 64)
        .expect("lfm3 resolves a TurboQuant request to the plain cache");
    assert!(matches!(tq.resolved_for(&lfm3.config), KvCompression::None));

    let lfm2 = HexagonLfmModel::from_gguf_on(backend(), lfm2_gguf(false), 64).unwrap();
    let mut state =
        InferenceState::from_config_capped(&lfm2.config, &KvCompression::None, 64).unwrap();
    let err = lfm2.try_reset_kv(&mut state, &tq, 64).unwrap_err();
    assert!(err.to_string().contains("TurboQuant"), "{err}");
}

/// The host-RoPE route (`CERA_HEXAGON_CPU_ROPE=1`) rotates lfm3's global layer with the
/// `rope_freqs` partial-rotary mask and its sliding layer with every pair at its own base.
/// The expected rotations come from the GGUF's own mask, not from the model's copy of it.
#[test]
fn lfm3_host_rope_applies_the_mask_to_global_layers_only() {
    use crate::backend::cpu::rope_neox_freq_factors as rotate;
    let mask = lfm3_gguf()
        .get_tensor("rope_freqs.weight")
        .unwrap()
        .to_f32_vec();
    let model = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    let mut seen = [false; 2];
    for (idx, layer) in model.layers.iter().enumerate() {
        let HexagonLayer::Attention(a) = layer else {
            continue;
        };
        let global = a.rope_freqs_offset.is_some();
        let kv_heads = model.config.kv_heads_per_layer[idx];
        let theta = if global { 1.0e6 } else { 1.0e4 };
        let q0: Vec<f32> = (0..N_HEADS * HEAD_DIM)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let k0: Vec<f32> = (0..kv_heads * HEAD_DIM)
            .map(|i| (i as f32 * 0.21).cos())
            .collect();
        let heads = (N_HEADS, kv_heads);
        let (mut q, mut k) = (q0.clone(), k0.clone());
        let mut scratch = Default::default();
        model.host_rope_row(a, &mut scratch, &mut q, &mut k, 5, heads, HEAD_DIM, theta);
        let (mut eq, mut ek) = (q0.clone(), k0.clone());
        rotate(
            &mut eq,
            &mut ek,
            5,
            heads,
            HEAD_DIM,
            theta,
            global.then_some(&mask[..]),
        );
        assert_eq!((q.clone(), k.clone()), (eq.clone(), ek), "layer {idx}");
        // the mask must matter here, or the comparison above proves nothing
        let (mut uq, mut uk) = (q0, k0);
        rotate(&mut uq, &mut uk, 5, heads, HEAD_DIM, theta, None);
        assert_eq!(uq == eq, !global, "layer {idx}: the mask changed nothing");
        seen[usize::from(global)] = true;
    }
    assert_eq!(seen, [true, true], "one sliding and one global layer");
}

/// `CERA_HEXAGON_SSM_CONV=0` restores the manual 3-tap chain for LFM2; a 7-tap conv has no
/// such chain (its tap planes are not laid out for it), so the knob must not apply to lfm3.
#[test]
fn lfm3_ignores_the_ssm_conv_knob() {
    let (driver, device) = op_capture::fresh_device();
    let knobs = HexagonKnobs::from_lookup(|k| (k == "CERA_HEXAGON_SSM_CONV").then(|| "0".into()));
    let mut off = None;
    let warned = crate::audio_profile::tests::warnings_of(|| {
        off = Some(lfm3_forward_ops_on(
            Backend::with_device(driver, device, knobs),
            8,
        ));
    });
    let off = off.unwrap();
    // an operator who sets the knob on a phone is told it does nothing here
    assert_eq!(
        warned
            .iter()
            .filter(|w| w.contains("CERA_HEXAGON_SSM_CONV=0 is ignored"))
            .count(),
        1,
        "{warned:?}"
    );
    let default = lfm3_forward_ops(8);
    assert_eq!(count(&off.0, "SsmConv"), 2, "decode keeps the fused conv");
    assert_eq!(off, default, "the knob changed an lfm3 graph");
}

/// A decode pass issues the lfm3 graph: four sandwich norms per block, the gate, the
/// 7-tap conv, and DSP rotations: the global layer's carries the `rope_freqs` mask as a
/// third source, the sliding layer's runs at its own base without one. (Which layer reads the window mask is the dense path's
/// existing `swa` selection; a captured descriptor cannot tell the two mask
/// buffers apart, so it is checked through the planned layer flags instead.)
#[test]
fn lfm3_decode_emits_the_lfm3_graph() {
    let (decode, _) = lfm3_forward_ops(8);
    // every rotation is a DSP op, so the whole token is one batch
    assert_eq!(decode.len(), 1, "no host step inside the token");

    // norms: 4 per block (attn, post-attn, ffn, post-ffn) + 2 QK norms per attention
    // layer + the output norm
    assert_eq!(count(&decode, "RmsNormMul"), 4 * 4 + 2 * 2 + 1);
    assert_eq!(
        count(&decode, "UnarySigmoid"),
        2,
        "one gate per attention layer"
    );
    assert_eq!(count(&decode, "FlashAttnExt"), 2);
    // Q and K of both attention layers rotate on the DSP
    assert_eq!(count(&decode, "Rope"), 4);

    // the 7-tap conv: [6 state slots | bx] -> [7, C], SsmConv with 7 taps, and the
    // state advances by copying conv_x's last 6 columns back
    let concats = op_lines(&decode, "Concat");
    assert_eq!(concats.len(), 2);
    for line in concats {
        let (src, dst) = descs(line);
        assert_eq!(src[0].ne[..2], [6, HS], "state: 6 slots per channel");
        assert_eq!(src[0].nb[..2], [4, 6 * 4]);
        assert_eq!(src[1].ne[..2], [1, HS]);
        assert_eq!(dst[0].ne[..2], [7, HS]);
        assert_eq!(dst[0].nb[..2], [4, 7 * 4]);
    }
    let convs = op_lines(&decode, "SsmConv");
    assert_eq!(convs.len(), 2);
    for line in &convs {
        assert_eq!(kparams(line)[1], 7, "d_conv");
        let (src, dst) = descs(line);
        assert_eq!(src[0].ne[..2], [7, HS]);
        assert_eq!(src[1].ne[..2], [7, HS], "taps are [7, C]");
        assert_eq!(dst[0].ne[..2], [HS, 1]);
    }
    let concat_dst = descs(op_lines(&decode, "Concat")[0]).1[0].off;
    let writebacks: Vec<_> = op_lines(&decode, "Cpy")
        .into_iter()
        .map(descs)
        .filter(|(src, _)| src[0].ne[..2] == [6, HS])
        .collect();
    assert_eq!(writebacks.len(), 2, "one state copy per conv layer");
    let (src, dst) = &writebacks[0];
    assert_eq!(src[0].off, concat_dst + 4, "skip the oldest column");
    assert_eq!(src[0].nb[..2], [4, 7 * 4]);
    assert_eq!(dst[0].nb[..2], [4, 6 * 4], "dense [C, 6] state");

    // the per-head gate: a matmul to the padded 32 outputs, then a per-row scalar
    // multiply of the attention output viewed as one row per head
    let gate_mm: Vec<_> = op_lines(&decode, "MulMat")
        .into_iter()
        .map(descs)
        .filter(|(_, dst)| dst[0].ne[..2] == [32, 1])
        .collect();
    assert_eq!(gate_mm.len(), 2, "one gate projection per attention layer");
    // the gate reads the normed block input, the same activation the attention
    // layer's last fused projection reads (not the residual stream)
    let mut last_proj_in = None;
    let mut gate_in = Vec::new();
    let mut proj_in = Vec::new();
    for line in decode.iter().flat_map(|b| b.lines()) {
        if line.starts_with("MulMatNx ") {
            last_proj_in = descs(line).0.last().map(|d| d.off);
        } else if line.starts_with("MulMat ") {
            let (src, dst) = descs(line);
            if dst[0].ne[..2] == [32, 1] {
                gate_in.push(src[1].off);
                proj_in.push(last_proj_in.expect("projection before the gate"));
            }
        }
    }
    assert_eq!(gate_in.len(), 2);
    assert_eq!(gate_in, proj_in);
    let scalar_muls: Vec<_> = op_lines(&decode, "Mul")
        .into_iter()
        .filter(|l| {
            kparams(l)[0] == i64::from(crate::backend::hexagon::HTP_BINARY_KERNEL_SCALAR_DMA)
        })
        .collect();
    assert_eq!(
        scalar_muls.len(),
        2,
        "SCALAR_DMA multiply per attention layer"
    );
    for line in scalar_muls {
        let (src, dst) = descs(line);
        assert_eq!(src[0].ne[..2], [HEAD_DIM, N_HEADS]);
        assert_eq!(src[1].ne[..2], [1, N_HEADS]);
        assert_eq!(src[1].nb[..2], [4, 4], "contiguous scalar table");
        assert_eq!(dst[0].ne[..2], [HEAD_DIM, N_HEADS]);
    }

    // The sliding layer rotates at the sliding base with no mask; the global layer at
    // the model base with `rope_freqs` as the third source and its slot in the params.
    let rope_bits = |theta: f32| theta.to_bits() as i64;
    let (mut sliding, mut global) = (0, 0);
    for line in op_lines(&decode, "Rope") {
        let params: Vec<i64> = line[line.find("params=[").unwrap() + 8..]
            .split(']')
            .next()
            .unwrap()
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let (src, _) = descs(line);
        let kp = kparams(line);
        if params.contains(&rope_bits(10_000.0)) {
            assert!(!params.contains(&rope_bits(1_000_000.0)), "{params:?}");
            assert_eq!((src.len(), kp[8]), (2, 0), "sliding layer has no mask");
            sliding += 1;
        } else {
            assert!(params.contains(&rope_bits(1_000_000.0)), "{params:?}");
            assert_eq!(src.len(), 3, "global layer reads rope_freqs");
            assert_eq!(src[2].ne[..2], [HEAD_DIM / 2, 1]);
            assert_eq!(src[2].nb[0], 4);
            // the slot is 256-aligned, after the per-thread slabs, inside the VTCM total
            assert_eq!(kp[8], 256, "{kp:?}");
            assert_eq!(kp[3], kp[7] + kp[8], "{kp:?}");
            global += 1;
        }
    }
    assert_eq!((sliding, global), (2, 2), "Q and K of each attention layer");
    // Both global rotations read the planned `rope_freqs` vector itself: the offset
    // the layer carries, holding the GGUF tensor verbatim, sized for every pair.
    let model = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    let planned: Vec<usize> = model
        .layers
        .iter()
        .filter_map(|l| match l {
            HexagonLayer::Attention(a) => a.rope_freqs_offset,
            _ => None,
        })
        .collect();
    assert_eq!(planned.len(), 1, "one global layer");
    let ff_srcs: Vec<_> = op_lines(&decode, "Rope")
        .into_iter()
        .filter_map(|l| descs(l).0.get(2).map(|d| (d.off, d.nb[1])))
        .collect();
    assert_eq!(ff_srcs, vec![(planned[0], HEAD_DIM / 2 * 4); 2]);
    assert_eq!(
        weights_f32(&model, planned[0], HEAD_DIM / 2),
        tensor_f32(&lfm3_gguf(), "rope_freqs.weight")
    );
}

/// The recorded decode template holds the lfm3 graph (the DSP rotation is what lets
/// it): a replayed step carries exactly the ops a fresh emission does, at positions
/// before, at and past the sliding window.
#[test]
fn lfm3_decode_template_replays_like_fresh_emission() {
    use super::forward_golden_tests::staged_text;
    let step = |model: &HexagonLfmModel, state: &mut InferenceState, pos: usize| {
        model
            .try_forward_input(DecodeInput::Token(3), DecodeOutput::Logits, pos, state)
            .unwrap();
    };
    let new_state = |model: &HexagonLfmModel| {
        InferenceState::from_config_capped(
            &model.config,
            &KvCompression::None,
            model.config.max_seq_len,
        )
        .unwrap()
    };
    let replay = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    let mut rstate = new_state(&replay);
    let fresh = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    let mut fstate = new_state(&fresh);
    step(&replay, &mut rstate, 0);
    step(&fresh, &mut fstate, 0);
    op_capture::take();
    for pos in 1..=L3_WINDOW + 4 {
        step(&replay, &mut rstate, pos);
        assert!(
            op_capture::take().is_empty(),
            "pos {pos}: replayed, not re-emitted"
        );
        let replayed = {
            let guard = replay.decode_template.lock().unwrap();
            staged_text(&replay, &guard.as_ref().expect("template").staged)
        };
        *fresh.decode_template.lock().unwrap() = None;
        step(&fresh, &mut fstate, pos);
        let mut batches = op_capture::take();
        assert_eq!(batches.len(), 1, "pos {pos}");
        assert_eq!(replayed, batches.pop().unwrap(), "pos {pos}");
    }
}

/// A prefill chunk issues the same graph over `m` rows, with the conv window and the
/// state write-back sized to the chunk, and the gate packed per token.
#[test]
fn lfm3_prefill_emits_the_lfm3_graph() {
    const M: usize = 8;
    let (_, prefill) = lfm3_forward_ops(M);
    assert_eq!(count(&prefill, "RmsNormMul"), 4 * 4 + 2 * 2 + 1);
    assert_eq!(count(&prefill, "UnarySigmoid"), 2);
    assert_eq!(count(&prefill, "SsmConv"), 2);
    assert_eq!(count(&prefill, "FlashAttnExt"), 2);
    // the chunk rotates on the DSP too: Q and K of the global layer carry the mask
    let rope_srcs: Vec<_> = op_lines(&prefill, "Rope")
        .into_iter()
        .map(|l| (descs(l).0.len(), kparams(l)[8]))
        .collect();
    assert_eq!(
        rope_srcs
            .iter()
            .filter(|&&(n, ff)| n == 3 && ff == 256)
            .count(),
        2
    );
    assert_eq!(
        rope_srcs
            .iter()
            .filter(|&&(n, ff)| n == 2 && ff == 0)
            .count(),
        2
    );

    for line in op_lines(&prefill, "Concat") {
        let (src, dst) = descs(line);
        assert_eq!(src[0].ne[..2], [6, HS], "state slots");
        assert_eq!(src[1].ne[..2], [M, HS], "the chunk's bx rows");
        assert_eq!(dst[0].ne[..2], [6 + M, HS]);
    }
    for line in op_lines(&prefill, "SsmConv") {
        assert_eq!(kparams(line)[1], 7);
        let (src, dst) = descs(line);
        assert_eq!(src[0].ne[..2], [6 + M, HS]);
        assert_eq!(dst[0].ne[..2], [HS, M]);
    }
    // next state = the last 6 columns of every channel row of the window
    let concat_dst = descs(op_lines(&prefill, "Concat")[0]).1[0].off;
    let writebacks: Vec<_> = op_lines(&prefill, "Cpy")
        .into_iter()
        .map(descs)
        .filter(|(src, dst)| src[0].ne[..2] == [6, HS] && dst[0].nb[1] == 6 * 4)
        .collect();
    assert_eq!(writebacks.len(), 2);
    let (src, dst) = &writebacks[0];
    assert_eq!(
        src[0].off,
        concat_dst + M * 4,
        "skip the chunk's first M columns"
    );
    assert_eq!(src[0].nb[..2], [4, (6 + M) * 4]);
    assert_eq!(dst[0].nb[..2], [4, 6 * 4]);

    // gate: the padded [M, 32] projection is packed to [M, n_heads] for the scalar multiply
    let pack: Vec<_> = op_lines(&prefill, "Cpy")
        .into_iter()
        .map(descs)
        .filter(|(src, _)| src[0].ne[..2] == [N_HEADS, M] && src[0].nb[1] == 32 * 4)
        .collect();
    assert_eq!(pack.len(), 2, "one gate packing per attention layer");
    assert_eq!(pack[0].1[0].nb[..2], [4, N_HEADS * 4]);
    let prefill_gates: Vec<_> = op_lines(&prefill, "Mul")
        .into_iter()
        .filter(|l| {
            kparams(l)[0] == i64::from(crate::backend::hexagon::HTP_BINARY_KERNEL_SCALAR_DMA)
        })
        .collect();
    assert_eq!(
        prefill_gates.len(),
        2,
        "a SCALAR_DMA multiply per attention layer"
    );
    for line in prefill_gates {
        let (src, _) = descs(line);
        assert_eq!(
            src[0].ne[..2],
            [HEAD_DIM, M * N_HEADS],
            "a row per (token, head)"
        );
        assert_eq!(src[1].ne[..2], [1, M * N_HEADS]);
    }
}

/// The same lfm3 model, pinned: any change to its planning, copying or either pass shows
/// up here. Regenerate with `CERA_UPDATE_GOLDEN=1`.
#[test]
fn lfm3_constructor_is_pinned() {
    let model = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    assert_model(
        "ctor_lfm3",
        &model,
        (14294067538294377481, 13080567521429679373),
    );
}

#[test]
fn lfm3_without_sandwich_norms_loads_and_matches_prefill_to_decode() {
    use crate::model::Model;
    let make = || {
        lfm3_gguf_with(L3_TAPS, &|_, t| {
            t.retain(|x| {
                !x.name.ends_with(".post_attention_norm.weight")
                    && !x.name.ends_with(".post_ffw_norm.weight")
            });
        })
    };
    let hex = HexagonLfmModel::from_gguf_on(backend(), make(), 64).unwrap();
    for layer in &hex.layers {
        match layer {
            HexagonLayer::Conv(c) => {
                assert!(c.post_norm_offset.is_none());
                assert!(c.ffn_post_norm_offset.is_none());
            }
            HexagonLayer::Attention(a) => {
                assert!(a.attn_post_norm_offset.is_none());
                assert!(a.ffn_post_norm_offset.is_none());
            }
            HexagonLayer::DeltaNet(_) => unreachable!(),
        }
    }
    let cpu = crate::model::lfm2::LfmModel::from_gguf(make(), 64).unwrap();
    let tokens = [1u32, 3, 5, 7];
    let mut s_prefill = InferenceState::from_config(crate::model::Model::config(&cpu)).unwrap();
    let prefill_logits = cpu.forward_prefill(&tokens, 0, &mut s_prefill);
    let mut s_decode = InferenceState::from_config(crate::model::Model::config(&cpu)).unwrap();
    let mut decode_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        decode_logits = cpu.forward(&[tok], pos, &mut s_decode);
    }
    let max_diff = prefill_logits
        .iter()
        .zip(&decode_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "max_diff={max_diff}");
}

#[test]
fn lfm3_with_decoupled_head_dim_loads_and_matches_prefill_to_decode() {
    use crate::model::Model;
    let new_head_dim = HEAD_DIM * 2; // 32 -> q_dim = 4 * 32 = 128 != HS (64)
    let new_q_dim = N_HEADS * new_head_dim;
    let make = || {
        lfm3_gguf_with(L3_TAPS, &move |kv, t| {
            set_kv(
                kv,
                "lfm3.attention.key_length",
                Some(KvValue::U32(new_head_dim as u32)),
            );
            set_kv(
                kv,
                "lfm3.attention.value_length",
                Some(KvValue::U32(new_head_dim as u32)),
            );
            set_kv(
                kv,
                "lfm3.rope.dimension_count",
                Some(KvValue::U32(new_head_dim as u32)),
            );
            resize(t, "rope_freqs.weight", vec![new_head_dim / 2]);
            let ff = t
                .iter_mut()
                .find(|x| x.name == "rope_freqs.weight")
                .unwrap();
            for chunk in ff.data.as_chunks_mut::<4>().0 {
                chunk.copy_from_slice(&1.0f32.to_le_bytes());
            }
            for l in [1usize, 3] {
                resize(t, &format!("blk.{l}.attn_q.weight"), vec![HS, new_q_dim]);
                resize(t, &format!("blk.{l}.attn_k.weight"), vec![HS, new_head_dim]);
                resize(t, &format!("blk.{l}.attn_v.weight"), vec![HS, new_head_dim]);
                resize(
                    t,
                    &format!("blk.{l}.attn_output.weight"),
                    vec![new_q_dim, HS],
                );
                resize(
                    t,
                    &format!("blk.{l}.attn_q_norm.weight"),
                    vec![new_head_dim],
                );
                resize(
                    t,
                    &format!("blk.{l}.attn_k_norm.weight"),
                    vec![new_head_dim],
                );
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    let norm_t = t
                        .iter_mut()
                        .find(|x| x.name == format!("blk.{l}.{suffix}"))
                        .unwrap();
                    for chunk in norm_t.data.as_chunks_mut::<4>().0 {
                        chunk.copy_from_slice(&1.0f32.to_le_bytes());
                    }
                }
            }
        })
    };
    let hex = HexagonLfmModel::from_gguf_on(backend(), make(), 64).unwrap();
    assert_eq!(hex.config.head_dim, new_head_dim);
    let cpu = crate::model::lfm2::LfmModel::from_gguf(make(), 64).unwrap();
    assert_eq!(crate::model::Model::config(&cpu).head_dim, new_head_dim);
    let tokens = [1u32, 3, 5, 7];
    let mut s_prefill = InferenceState::from_config(crate::model::Model::config(&cpu)).unwrap();
    let prefill_logits = cpu.forward_prefill(&tokens, 0, &mut s_prefill);
    let mut s_decode = InferenceState::from_config(crate::model::Model::config(&cpu)).unwrap();
    let mut decode_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        decode_logits = cpu.forward(&[tok], pos, &mut s_decode);
    }
    let max_diff = prefill_logits
        .iter()
        .zip(&decode_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "max_diff={max_diff}");
}

/// The routed-expert edit of the lfm3 MoE file: three of four layers route over four experts.
fn lfm3_moe_edit(kv: &mut Vec<(String, KvValue)>, t: &mut Vec<Tensor>) {
    let n_expert = 4usize;
    let n_used = 2usize;
    let exp_ff = 64usize;
    let mut g = Gen(0x9988_7766);
    for (key, v) in [
        ("leading_dense_block_count", 1u32),
        ("expert_gating_func", 2),
        ("expert_count", n_expert as u32),
        ("expert_used_count", n_used as u32),
        ("expert_feed_forward_length", exp_ff as u32),
    ] {
        set_kv(kv, &format!("lfm3.{key}"), Some(KvValue::U32(v)));
    }
    for l in 1..4 {
        t.retain(|x| {
            !x.name.starts_with(&format!("blk.{l}.ffn_gate."))
                && !x.name.starts_with(&format!("blk.{l}.ffn_up."))
                && !x.name.starts_with(&format!("blk.{l}.ffn_down."))
        });
        t.push(g.f32(
            &format!("blk.{l}.ffn_gate_inp.weight"),
            &[HS, n_expert],
            0.1,
            0.0,
        ));
        t.push(g.f32(&format!("blk.{l}.exp_probs_b.bias"), &[n_expert], 0.05, 0.0));
        t.push(g.q4_0(
            &format!("blk.{l}.ffn_gate_exps.weight"),
            &[HS, exp_ff, n_expert],
        ));
        t.push(g.q4_0(
            &format!("blk.{l}.ffn_up_exps.weight"),
            &[HS, exp_ff, n_expert],
        ));
        t.push(g.q4_0(
            &format!("blk.{l}.ffn_down_exps.weight"),
            &[exp_ff, HS, n_expert],
        ));
    }
}

#[test]
fn lfm3_moe_loads_and_matches_prefill_to_decode() {
    use crate::model::Model;
    let make = || lfm3_gguf_with(L3_TAPS, &lfm3_moe_edit);
    let hex = HexagonLfmModel::from_gguf_on(backend(), make(), 64).unwrap();
    assert!(hex.config.moe.is_some());
    assert_eq!(
        hex.config.moe.as_ref().unwrap().is_moe_layer,
        vec![false, true, true, true]
    );
    let cpu = crate::model::lfm2::LfmModel::from_gguf(make(), 64).unwrap();
    let tokens = [1u32, 3, 5, 7];
    let mut s_prefill = InferenceState::from_config(crate::model::Model::config(&cpu)).unwrap();
    let prefill_logits = cpu.forward_prefill(&tokens, 0, &mut s_prefill);
    let mut s_decode = InferenceState::from_config(crate::model::Model::config(&cpu)).unwrap();
    let mut decode_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        decode_logits = cpu.forward(&[tok], pos, &mut s_decode);
    }
    let max_diff = prefill_logits
        .iter()
        .zip(&decode_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "max_diff={max_diff}");
}

/// The routed-expert flags are typed like every other lfm3 key: a wrong-typed
/// `expert_weights_norm` used to read as absent and load as normalised routing.
#[test]
fn lfm3_moe_routing_flags_must_be_typed() {
    let err = crate::model::lfm2::LfmModel::from_gguf(
        lfm3_gguf_with(L3_TAPS, &|kv, t| {
            lfm3_moe_edit(kv, t);
            set_kv(kv, "lfm3.expert_weights_norm", Some(KvValue::U32(0)));
        }),
        64,
    )
    .err()
    .expect("a mistyped expert_weights_norm was accepted")
    .to_string();
    assert!(
        err.contains("expert_weights_norm must be stored as a bool"),
        "{err}"
    );
}

/// A new sequence must start with no conv history at all: the 7-tap kernel keeps six planes
/// per layer, and clearing only two left most channels carrying the previous request's state.
#[test]
fn a_sequence_reset_clears_every_history_plane_of_every_conv_layer() {
    let model = HexagonLfmModel::from_gguf_on(backend(), lfm3_gguf(), 64).unwrap();
    unsafe {
        std::ptr::write_bytes(
            model.kv_state_buf.as_mut_ptr(),
            0xFF,
            model.kv_state_buf.size(),
        );
    }
    model.zero_recurrent_layers();
    let mut conv_layers = 0;
    for layer in &model.layers {
        let HexagonLayer::Conv(conv) = layer else {
            continue;
        };
        conv_layers += 1;
        assert_eq!(conv.d_conv, L3_TAPS);
        let len = (conv.d_conv - 1) * align256(HS * 4);
        let slab = unsafe {
            std::slice::from_raw_parts(model.kv_state_buf.as_mut_ptr().add(conv.state_offset), len)
        };
        let first_stale = slab.iter().position(|&b| b != 0);
        assert_eq!(first_stale, None, "a conv history byte survived the reset");
    }
    assert!(conv_layers > 0);
}

#[test]
fn lfm3_all_backends_match_cpu_prefill_and_decode() {
    use crate::model::Model;
    let tokens = [1u32, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
    let decode_tokens = [25u32, 27];

    let cpu = crate::model::lfm2::LfmModel::from_gguf(lfm3_gguf(), 64).unwrap();
    let mut cpu_prefill_state = InferenceState::from_config(Model::config(&cpu)).unwrap();
    let cpu_prefill = cpu.forward_prefill(&tokens, 0, &mut cpu_prefill_state);

    let mut cpu_decode_state = InferenceState::from_config(Model::config(&cpu)).unwrap();
    let mut cpu_decode = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        cpu_decode = cpu.forward(&[tok], pos, &mut cpu_decode_state);
    }
    let prefill_vs_decode = cpu_prefill
        .iter()
        .zip(&cpu_decode)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        prefill_vs_decode < 1e-4,
        "CPU lfm3 prefill vs decode max_diff={prefill_vs_decode}"
    );

    let cpu_step1 = cpu.forward(&[decode_tokens[0]], tokens.len(), &mut cpu_prefill_state);
    let cpu_step2 = cpu.forward(
        &[decode_tokens[1]],
        tokens.len() + 1,
        &mut cpu_prefill_state,
    );

    let check_backend = |name: &str, got_prefill: &[f32], got_s1: &[f32], got_s2: &[f32]| {
        for (stage, want, got) in [
            ("prefill", cpu_prefill.as_slice(), got_prefill),
            ("decode_1", cpu_step1.as_slice(), got_s1),
            ("decode_2", cpu_step2.as_slice(), got_s2),
        ] {
            let dot: f32 = want.iter().zip(got).map(|(a, b)| a * b).sum();
            let na: f32 = want.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            let max_diff = want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                cos > 0.998 && max_diff < 0.55,
                "{name} {stage}: cosine={cos} max_diff={max_diff}"
            );
        }
    };

    let moe_tokens = [1u32, 3, 5, 7];
    let cpu_moe =
        crate::model::lfm2::LfmModel::from_gguf(lfm3_gguf_with(L3_TAPS, &lfm3_moe_edit), 64)
            .unwrap();
    let mut cpu_moe_state = InferenceState::from_config(Model::config(&cpu_moe)).unwrap();
    let want_moe_prefill = cpu_moe.forward_prefill(&moe_tokens, 0, &mut cpu_moe_state);
    let want_moe_decode = cpu_moe.forward(&[9], moe_tokens.len(), &mut cpu_moe_state);

    #[cfg(feature = "gpu")]
    if let Ok(gpu) = crate::model::gpu_lfm2::GpuLfmModel::from_gguf(lfm3_gguf(), 64) {
        let mut s = InferenceState::from_config(Model::config(&gpu)).unwrap();
        let p = gpu.forward_prefill(&tokens, 0, &mut s);
        let d1 = gpu.forward(&[decode_tokens[0]], tokens.len(), &mut s);
        let d2 = gpu.forward(&[decode_tokens[1]], tokens.len() + 1, &mut s);
        check_backend("wgpu", &p, &d1, &d2);

        let gpu_moe = crate::model::gpu_lfm2::GpuLfmModel::from_gguf(
            lfm3_gguf_with(L3_TAPS, &lfm3_moe_edit),
            64,
        )
        .unwrap();
        let mut s = InferenceState::from_config(Model::config(&gpu_moe)).unwrap();
        let got_p = gpu_moe.forward_prefill(&moe_tokens, 0, &mut s);
        let got_d = gpu_moe.forward(&[9], moe_tokens.len(), &mut s);
        for (stage, want, got) in [
            (
                "wgpu moe prefill",
                want_moe_prefill.as_slice(),
                got_p.as_slice(),
            ),
            (
                "wgpu moe decode",
                want_moe_decode.as_slice(),
                got_d.as_slice(),
            ),
        ] {
            let dot: f32 = want.iter().zip(got).map(|(a, b)| a * b).sum();
            let na: f32 = want.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos > 0.998, "{stage}: cosine={cos}");
        }
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    if let Ok(metal) = crate::model::metal_lfm2::MetalLfmModel::from_gguf(lfm3_gguf(), None, 64) {
        let mut s = InferenceState::from_config(Model::config(&metal)).unwrap();
        let p = metal.forward_prefill(&tokens, 0, &mut s);
        let d1 = metal.forward(&[decode_tokens[0]], tokens.len(), &mut s);
        let d2 = metal.forward(&[decode_tokens[1]], tokens.len() + 1, &mut s);
        check_backend("metal", &p, &d1, &d2);

        let metal_moe = crate::model::metal_lfm2::MetalLfmModel::from_gguf(
            lfm3_gguf_with(L3_TAPS, &lfm3_moe_edit),
            None,
            64,
        )
        .unwrap();
        let mut s = InferenceState::from_config(Model::config(&metal_moe)).unwrap();
        let got_p = metal_moe.forward_prefill(&moe_tokens, 0, &mut s);
        let got_d = metal_moe.forward(&[9], moe_tokens.len(), &mut s);
        for (stage, want, got) in [
            (
                "metal moe prefill",
                want_moe_prefill.as_slice(),
                got_p.as_slice(),
            ),
            (
                "metal moe decode",
                want_moe_decode.as_slice(),
                got_d.as_slice(),
            ),
        ] {
            let dot: f32 = want.iter().zip(got).map(|(a, b)| a * b).sum();
            let na: f32 = want.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos > 0.998, "{stage}: cosine={cos}");
        }
    }
}

#[test]
fn lfm3_d1_all_backends_match_cpu() {
    use crate::model::Model;
    let make_d1 = || {
        lfm3_gguf_with(L3_TAPS, &|kv, _| {
            set_kv(kv, "lfm3.attention.causal", Some(KvValue::Raw(7, vec![0])));
            set_kv(
                kv,
                "lfm3.decision.type",
                Some(KvValue::Str("lfm3-d1".into())),
            );
        })
    };

    let tokens = [2u32, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
    let cpu = crate::model::lfm2::LfmModel::from_gguf(make_d1(), 64).unwrap();
    assert!(!Model::config(&cpu).is_causal);
    assert!(cpu.layer_is_non_causal(1));
    assert!(!cpu.layer_is_non_causal(3));

    let mut cpu_state = InferenceState::from_config(Model::config(&cpu)).unwrap();
    let cpu_logits = cpu.forward_prefill(&tokens, 0, &mut cpu_state);
    let mut cpu_hs_state = InferenceState::for_prefill(Model::config(&cpu), tokens.len()).unwrap();
    let cpu_hs = cpu.hidden_states(&tokens, &mut cpu_hs_state);

    let mut tokens_alt = tokens;
    tokens_alt[tokens.len() - 1] = 31;
    let mut cpu_alt_state = InferenceState::for_prefill(Model::config(&cpu), tokens.len()).unwrap();
    let cpu_hs_alt = cpu.hidden_states(&tokens_alt, &mut cpu_alt_state);
    let row0_drift = cpu_hs[..HS]
        .iter()
        .zip(&cpu_hs_alt[..HS])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        row0_drift > 1e-4,
        "lfm3-d1 global attention must let the last token move row 0 (drift={row0_drift})"
    );

    let hex = HexagonLfmModel::from_gguf_on(backend(), make_d1(), 64).unwrap();
    assert!(!hex.config.is_causal);
    assert_eq!(hex.dense.swa_window, Some(L3_WINDOW));

    let check_cosine = |label: &str, want: &[f32], got: &[f32]| {
        assert_eq!(want.len(), got.len(), "{label} len");
        let dot: f32 = want.iter().zip(got).map(|(a, b)| a * b).sum();
        let na: f32 = want.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(cos > 0.999, "{label}: cosine={cos}");
    };

    #[cfg(feature = "gpu")]
    if let Ok(gpu) = crate::model::gpu_lfm2::GpuLfmModel::from_gguf(make_d1(), 64) {
        let mut s = InferenceState::from_config(Model::config(&gpu)).unwrap();
        let got_logits = gpu.forward_prefill(&tokens, 0, &mut s);
        check_cosine("wgpu d1 prefill logits", &cpu_logits, &got_logits);
        let mut hs_s = InferenceState::for_prefill(Model::config(&gpu), tokens.len()).unwrap();
        let got_hs = gpu.hidden_states(&tokens, &mut hs_s);
        check_cosine("wgpu d1 hidden_states", &cpu_hs, &got_hs);
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    if let Ok(metal) = crate::model::metal_lfm2::MetalLfmModel::from_gguf(make_d1(), None, 64) {
        let mut s = InferenceState::from_config(Model::config(&metal)).unwrap();
        let got_logits = metal.forward_prefill(&tokens, 0, &mut s);
        check_cosine("metal d1 prefill logits", &cpu_logits, &got_logits);
        let mut hs_s = InferenceState::for_prefill(Model::config(&metal), tokens.len()).unwrap();
        let got_hs = metal.hidden_states(&tokens, &mut hs_s);
        check_cosine("metal d1 hidden_states", &cpu_hs, &got_hs);
    }
}

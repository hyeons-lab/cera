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
    Backend::with_device(driver, device, HexagonKnobs::from_lookup(|_| None))
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

/// Constructor digests, pinned; regenerate with `CERA_UPDATE_GOLDEN=1`. A
/// mismatch means planning, copying or setup changed.
#[test]
fn constructors_build_pinned_models() {
    let dense = HexagonLfmModel::from_llama_on(backend(), dense_gguf(), None, 64).unwrap();
    assert_model(
        "ctor_dense",
        &dense,
        (8419770052891033654, 4083050957091022688),
    );
    let lfm2 = HexagonLfmModel::from_gguf_on(backend(), lfm2_gguf(false), 64).unwrap();
    assert_model(
        "ctor_lfm2",
        &lfm2,
        (8055141403873995527, 1102571476056680706),
    );
    let moe = HexagonLfmModel::from_gguf_on(backend(), lfm2_gguf(true), 64).unwrap();
    assert_model(
        "ctor_moe",
        &moe,
        (10223184744877565750, 11398544486494475589),
    );
    let cpu = crate::model::qwen35::Qwen35Model::from_gguf(qwen35_gguf(), 64).unwrap();
    let qwen35 = HexagonLfmModel::from_qwen35_model_on(backend(), &cpu, 64).unwrap();
    assert_model(
        "ctor_qwen35",
        &qwen35,
        (2939268980605172228, 7392097742996743327),
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
    let file = lfm2_gguf_layers(true, 8);
    let roomy = Paging {
        force: false,
        budget: 1 << 30,
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
    };
    let paged = HexagonLfmModel::from_gguf_on_with(backend(), file, 64, tight).unwrap();
    let pager = paged.pager.as_ref().unwrap();
    assert_eq!(pager.pinned(), 0);

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
}

/// A prefill chunk whose routed rows would not fit one staging buffer is
/// flushed between rows instead of failing: with the staging buffer pretended
/// to be tiny, the chunk still runs, in several batches each within it.
#[test]
fn long_routed_prefill_chunk_flushes_between_rows() {
    let file = lfm2_gguf_layers(true, 4);
    let mut model = HexagonLfmModel::from_gguf_on(backend(), file, 64).unwrap();
    let cap = 128 * 1024;
    model
        .device
        .get_mut()
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

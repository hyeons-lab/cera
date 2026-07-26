//! Shader-only oracle tests for the Metal TurboQuant kernels — the MSL mirror of
//! `wgpu_turboquant_oracle.rs`. Every kernel is checked against the same CPU
//! reference in `cera::turboquant`, on the same fixed inputs, **without** loading
//! any model weights.
//!
//! Running both suites against one oracle is the point: the WGSL and MSL kernels
//! must produce a bit-identical packed cache, because the `TQK1`/`TQV1`
//! prefix-cache blobs are shared across CPU, wgpu, and Metal. The encode test
//! asserts exact byte equality with the CPU packing, so if the two GPU backends
//! ever diverge, one of these two files fails.
//!
//! See the wgpu suite's header for the determinism and tolerance rationale; both
//! use the same LCG inputs and the same `2e-4` float tolerance.
//!
//! Gating: needs a Metal device. Skips cleanly when none is available (a CI
//! runner without a GPU), which the sibling `metal_params_layout.rs` covers
//! device-free.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use cera::backend::metal::{MetalContext, MetalParams, TqAttnParams, TqParams, shaders};
use cera::turboquant::{
    CompressedKeyCache, CompressedValueCache, EncodeScratch, QueryRotationScratch, RotationState,
    TqLayout, TurboQuantConfig, attn_scores_turboquant_gqa, attn_values_turboquant_gqa,
    compress_and_append_keys, compress_and_append_values, rotate_queries,
};
use metal::MTLSize;

const SEED: u64 = 0xC0FFEE;
/// Cache capacity in timesteps, deliberately > `SEQ_LEN` so the per-head region
/// stride (`max_seq_len`, not `seq_len`) is actually exercised.
const MAX_SEQ: usize = 12;
const SEQ_LEN: usize = 7;
const N_KV_HEADS: usize = 2;
/// 4 query heads over 2 KV heads → GQA group_size 2.
const N_HEADS: usize = 4;
/// Non-zero so the params' `sign_off` is exercised.
const LAYER: usize = 3;
const TOL: f32 = 2e-4;
/// Threads per threadgroup for the encode / rotate kernels — must match `TQ_WG`
/// in `turboquant.metal`.
const TQ_WG: u64 = 128;
/// Threads per threadgroup for `flash_attention_tq` — must match its tile width.
const ATTN_TG: u64 = 256;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

struct Fixture {
    head_dim: usize,
    rotation: RotationState,
    config: TurboQuantConfig,
    layout: TqLayout,
    signs: Vec<f32>,
    sign_off: usize,
}

impl Fixture {
    fn new(head_dim: usize) -> Self {
        // The engine seeds each layer as `seed ^ layer_idx`; mirror that exactly.
        let rotation = RotationState::from_seed(SEED ^ LAYER as u64, head_dim);
        let mut signs = Vec::new();
        for layer in 0..=LAYER {
            let r = RotationState::from_seed(SEED ^ layer as u64, head_dim);
            signs.extend_from_slice(&r.polar_signs);
            signs.extend_from_slice(&r.jl_signs);
        }
        Self {
            head_dim,
            rotation,
            config: TurboQuantConfig::for_head_dim(head_dim),
            layout: TqLayout::new(head_dim),
            signs,
            sign_off: LAYER * 2 * head_dim,
        }
    }

    fn encode_params(&self, n_tokens: usize, n_heads: usize, src_stride: usize) -> TqParams {
        let c = &self.config.centroids;
        let b = &self.config.boundaries;
        TqParams {
            n_tokens: n_tokens as u32,
            n_heads: n_heads as u32,
            head_dim: self.head_dim as u32,
            src_stride: src_stride as u32,
            dst_pos: 0,
            max_seq_len: MAX_SEQ as u32,
            sign_off: self.sign_off as u32,
            q_cap: SEQ_LEN as u32,
            c0: c[0],
            c1: c[1],
            c2: c[2],
            c3: c[3],
            b0: b[0],
            b1: b[1],
            b2: b[2],
            _pad: 0,
        }
    }
}

/// Dispatch one `turboquant.metal` kernel and return the output buffer's words.
fn run_tq_kernel(
    ctx: &MetalContext,
    entry: &str,
    src: &[f32],
    out_words: usize,
    signs: &[f32],
    params: &TqParams,
    groups: usize,
) -> Vec<u32> {
    let pipeline = ctx
        .create_pipeline(shaders::TURBOQUANT, entry)
        .unwrap_or_else(|e| panic!("{entry} pipeline: {e}"));
    let src_buf = ctx.upload_f32(src);
    let dst_buf = ctx.create_buffer((out_words * 4) as u64);
    let signs_buf = ctx.upload_f32(signs);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&src_buf), 0);
    enc.set_buffer(1, Some(&dst_buf), 0);
    enc.set_buffer(2, Some(&signs_buf), 0);
    params.set(enc, 3);
    enc.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(TQ_WG, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    read_u32(&dst_buf, out_words)
}

/// Metal buffers are shared (unified memory), so a readback is a plain borrow of
/// the mapped contents.
fn read_u32(buf: &metal::Buffer, count: usize) -> Vec<u32> {
    let ptr = buf.contents() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
}

fn cpu_encode(
    fx: &Fixture,
    k: &[f32],
    v: &[f32],
    kv_dim: usize,
) -> (CompressedKeyCache, CompressedValueCache) {
    let mut keys = CompressedKeyCache::new(N_KV_HEADS, fx.head_dim, SEQ_LEN);
    let mut values = CompressedValueCache::new(N_KV_HEADS, fx.head_dim, SEQ_LEN);
    let mut scratch = EncodeScratch::new(fx.head_dim);
    for t in 0..SEQ_LEN {
        compress_and_append_keys(
            &k[t * kv_dim..(t + 1) * kv_dim],
            N_KV_HEADS,
            fx.head_dim,
            &fx.rotation,
            &fx.config,
            &mut keys,
            &mut scratch,
        );
        compress_and_append_values(
            &v[t * kv_dim..(t + 1) * kv_dim],
            N_KV_HEADS,
            fx.head_dim,
            &fx.rotation,
            &fx.config,
            &mut values,
            &mut scratch,
        );
    }
    (keys, values)
}

/// The packed cache the MSL encode kernels write must be byte-identical to the
/// CPU's — and therefore to the WGSL kernels', which the wgpu suite pins to the
/// same reference.
fn check_encode(ctx: &MetalContext, head_dim: usize) {
    let fx = Fixture::new(head_dim);
    let kv_dim = N_KV_HEADS * head_dim;
    let mut rng = Lcg::new(11);
    let k = rng.fill(SEQ_LEN * kv_dim);
    let v = rng.fill(SEQ_LEN * kv_dim);
    let (cpu_keys, cpu_values) = cpu_encode(&fx, &k, &v, kv_dim);

    let pw = fx.layout.polar_words;
    let jw = fx.layout.jl_words;
    let vecs = N_KV_HEADS * MAX_SEQ;
    let (jl_off, norm_off) = fx.layout.key_regions(vecs);
    let params = fx.encode_params(SEQ_LEN, N_KV_HEADS, kv_dim);

    let words = run_tq_kernel(
        ctx,
        "tq_encode_keys",
        &k,
        fx.layout.key_words(vecs),
        &fx.signs,
        &params,
        SEQ_LEN * N_KV_HEADS,
    );
    for h in 0..N_KV_HEADS {
        for t in 0..SEQ_LEN {
            let slot = h * MAX_SEQ + t;
            let gpu_polar: &[u8] = bytemuck::cast_slice(&words[slot * pw..slot * pw + pw]);
            assert_eq!(
                gpu_polar,
                &cpu_keys.polar_data[h][t * pw * 4..(t + 1) * pw * 4],
                "hd={head_dim} key polar mismatch at h={h} t={t}"
            );
            let gpu_jl: &[u8] =
                bytemuck::cast_slice(&words[jl_off + slot * jw..jl_off + slot * jw + jw]);
            assert_eq!(
                gpu_jl,
                &cpu_keys.jl_data[h][t * jw * 4..(t + 1) * jw * 4],
                "hd={head_dim} key JL mismatch at h={h} t={t}"
            );
            let nw = words[norm_off + slot];
            assert_eq!(
                (nw & 0xFFFF) as u16,
                cpu_keys.norms[h][t],
                "hd={head_dim} key norm mismatch at h={h} t={t}"
            );
            assert_eq!(
                (nw >> 16) as u16,
                cpu_keys.residual_norms[h][t],
                "hd={head_dim} key residual-norm mismatch at h={h} t={t}"
            );
        }
    }

    let v_norm_off = vecs * pw;
    let words = run_tq_kernel(
        ctx,
        "tq_encode_values",
        &v,
        fx.layout.value_words(vecs),
        &fx.signs,
        &params,
        SEQ_LEN * N_KV_HEADS,
    );
    for h in 0..N_KV_HEADS {
        for t in 0..SEQ_LEN {
            let slot = h * MAX_SEQ + t;
            let gpu_polar: &[u8] = bytemuck::cast_slice(&words[slot * pw..slot * pw + pw]);
            assert_eq!(
                gpu_polar,
                &cpu_values.polar_data[h][t * pw * 4..(t + 1) * pw * 4],
                "hd={head_dim} value polar mismatch at h={h} t={t}"
            );
            assert_eq!(
                (words[v_norm_off + slot] & 0xFFFF) as u16,
                cpu_values.norms[h][t],
                "hd={head_dim} value norm mismatch at h={h} t={t}"
            );
        }
    }
}

fn check_rotate_q(ctx: &MetalContext, head_dim: usize) {
    let fx = Fixture::new(head_dim);
    let q_dim = N_HEADS * head_dim;
    let mut rng = Lcg::new(23);
    let q = rng.fill(SEQ_LEN * q_dim);

    let params = fx.encode_params(SEQ_LEN, N_HEADS, q_dim);
    let region = SEQ_LEN * N_HEADS * head_dim;
    let words = run_tq_kernel(
        ctx,
        "tq_rotate_q",
        &q,
        2 * region + SEQ_LEN * N_HEADS,
        &fx.signs,
        &params,
        SEQ_LEN * N_HEADS,
    );
    let got: &[f32] = bytemuck::cast_slice(&words);

    let mut scratch = QueryRotationScratch::new(N_HEADS, head_dim);
    for t in 0..SEQ_LEN {
        rotate_queries(
            &q[t * q_dim..(t + 1) * q_dim],
            N_HEADS,
            head_dim,
            &fx.rotation,
            &mut scratch,
        );
        for h in 0..N_HEADS {
            let base = (t * N_HEADS + h) * head_dim;
            for d in 0..head_dim {
                assert_close(
                    got[base + d],
                    scratch.q_rot[h * head_dim + d],
                    &format!("hd={head_dim} q_rot t={t} h={h} d={d}"),
                );
                assert_close(
                    got[region + base + d],
                    scratch.q_jl[h * head_dim + d],
                    &format!("hd={head_dim} q_jl t={t} h={h} d={d}"),
                );
            }
            assert_close(
                got[2 * region + t * N_HEADS + h],
                scratch.q_jl_total_sums[h],
                &format!("hd={head_dim} q_jl sum t={t} h={h}"),
            );
        }
    }
}

fn check_attention(ctx: &MetalContext, head_dim: usize) {
    let fx = Fixture::new(head_dim);
    let kv_dim = N_KV_HEADS * head_dim;
    let q_dim = N_HEADS * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group_size = N_HEADS / N_KV_HEADS;

    let mut rng = Lcg::new(37);
    let k = rng.fill(SEQ_LEN * kv_dim);
    let v = rng.fill(SEQ_LEN * kv_dim);
    let q = rng.fill(SEQ_LEN * q_dim);
    let (cpu_keys, cpu_values) = cpu_encode(&fx, &k, &v, kv_dim);

    let vecs = N_KV_HEADS * MAX_SEQ;
    let enc_params = fx.encode_params(SEQ_LEN, N_KV_HEADS, kv_dim);
    let key_words = run_tq_kernel(
        ctx,
        "tq_encode_keys",
        &k,
        fx.layout.key_words(vecs),
        &fx.signs,
        &enc_params,
        SEQ_LEN * N_KV_HEADS,
    );
    let value_words = run_tq_kernel(
        ctx,
        "tq_encode_values",
        &v,
        fx.layout.value_words(vecs),
        &fx.signs,
        &enc_params,
        SEQ_LEN * N_KV_HEADS,
    );
    let rot_params = fx.encode_params(SEQ_LEN, N_HEADS, q_dim);
    let region = SEQ_LEN * N_HEADS * head_dim;
    let qrot_words = run_tq_kernel(
        ctx,
        "tq_rotate_q",
        &q,
        2 * region + SEQ_LEN * N_HEADS,
        &fx.signs,
        &rot_params,
        SEQ_LEN * N_HEADS,
    );

    let c = &fx.config.centroids;
    let attn_params = TqAttnParams {
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: head_dim as u32,
        max_seq: SEQ_LEN as u32,
        start_pos: 0,
        scale,
        q_cap: SEQ_LEN as u32,
        out_stride: q_dim as u32,
        qjl_scale: (std::f32::consts::PI / 2.0).sqrt() / head_dim as f32,
        sign_off: fx.sign_off as u32,
        c0: c[0],
        c1: c[1],
        c2: c[2],
        c3: c[3],
        q_base: 0,
        cache_cap: MAX_SEQ as u32,
    };
    let got = run_attention(
        ctx,
        &qrot_words,
        &key_words,
        &value_words,
        &fx.signs,
        &attn_params,
        SEQ_LEN * q_dim,
        N_HEADS,
        SEQ_LEN,
    );

    let mut scratch = QueryRotationScratch::new(N_HEADS, head_dim);
    for t_q in 0..SEQ_LEN {
        let live = t_q + 1;
        rotate_queries(
            &q[t_q * q_dim..(t_q + 1) * q_dim],
            N_HEADS,
            head_dim,
            &fx.rotation,
            &mut scratch,
        );
        let mut expected = vec![0.0f32; q_dim];
        for kv_head in 0..N_KV_HEADS {
            let group_start = kv_head * group_size;
            let mut scores = vec![0.0f32; group_size * live];
            attn_scores_turboquant_gqa(
                &cpu_keys,
                kv_head,
                group_start,
                group_size,
                &mut scores,
                head_dim,
                scale,
                live,
                &fx.config,
                &mut scratch,
            );
            for g in 0..group_size {
                let row = &mut scores[g * live..(g + 1) * live];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in row.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for s in row.iter_mut() {
                    *s /= sum;
                }
            }
            attn_values_turboquant_gqa(
                &cpu_values,
                kv_head,
                group_start,
                group_size,
                &scores,
                &mut expected,
                head_dim,
                live,
                &fx.rotation,
                &fx.config,
            );
        }
        for (i, &want) in expected.iter().enumerate() {
            assert_close(
                got[t_q * q_dim + i],
                want,
                &format!("hd={head_dim} attn out t_q={t_q} i={i}"),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_attention(
    ctx: &MetalContext,
    qrot: &[u32],
    keys: &[u32],
    values: &[u32],
    signs: &[f32],
    params: &TqAttnParams,
    out_floats: usize,
    n_heads: usize,
    n_queries: usize,
) -> Vec<f32> {
    let pipeline = ctx
        .create_pipeline(shaders::FLASH_ATTENTION_TQ, "flash_attention_tq")
        .expect("flash_attention_tq pipeline");
    let qrot_buf = ctx.upload_bytes(bytemuck::cast_slice(qrot));
    let k_buf = ctx.upload_bytes(bytemuck::cast_slice(keys));
    let v_buf = ctx.upload_bytes(bytemuck::cast_slice(values));
    let out_buf = ctx.create_buffer((out_floats * 4) as u64);
    let signs_buf = ctx.upload_f32(signs);

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&qrot_buf), 0);
    enc.set_buffer(1, Some(&k_buf), 0);
    enc.set_buffer(2, Some(&v_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    params.set(enc, 4);
    enc.set_buffer(5, Some(&signs_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(n_heads as u64, n_queries as u64, 1),
        MTLSize::new(ATTN_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    ctx.read_f32(&out_buf, out_floats)
}

fn assert_close(got: f32, want: f32, what: &str) {
    let diff = (got - want).abs();
    assert!(
        diff < TOL,
        "{what}: got={got} want={want} diff={diff} (tol={TOL})"
    );
}

fn context() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no Metal device ({e})");
            None
        }
    }
}

#[test]
fn encode_matches_cpu() {
    let Some(ctx) = context() else { return };
    check_encode(&ctx, 64);
    check_encode(&ctx, 128);
}

#[test]
fn rotate_q_matches_cpu() {
    let Some(ctx) = context() else { return };
    check_rotate_q(&ctx, 64);
    check_rotate_q(&ctx, 128);
}

#[test]
fn attention_matches_cpu() {
    let Some(ctx) = context() else { return };
    check_attention(&ctx, 64);
    check_attention(&ctx, 128);
}

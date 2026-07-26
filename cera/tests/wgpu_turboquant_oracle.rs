//! Shader-only oracle tests for the wgpu TurboQuant kernels. Every kernel is
//! checked against its CPU reference in `cera::turboquant` on synthetic data,
//! **without** loading any model weights.
//!
//! Four kernels, four tests:
//!
//! 1. `tq_encode_keys` / `tq_encode_values` vs `compress_and_append_keys` /
//!    `compress_and_append_values` — compares the packed bytes and the f16 norm
//!    words directly. This is the strongest available check on the bit layout: a
//!    wrong shift, a byte-order flip, or a mis-strided region shows up as a
//!    wholesale mismatch.
//! 2. `tq_rotate_q` vs `rotate_queries` — the randomized Hadamard transform and
//!    the per-head `sum(q_jl)` the score correction needs.
//! 3. `flash_attention_tq` vs `attn_scores_turboquant_gqa` + softmax +
//!    `attn_values_turboquant_gqa` — the whole compressed read path, including the
//!    causal mask, the online-softmax accumulation, and the single inverse RHT in
//!    the epilogue.
//! 4. `TqGpuCache::snapshot_layer` / `restore_layer` — the prefix-cache blobs a
//!    GPU cache produces must be byte-identical to the CPU backend's `TQK1`/`TQV1`
//!    encoding (so the two are mutually loadable), and must round-trip.
//!
//! ## Determinism
//!
//! Inputs come from a fixed LCG, never `rand`. The GPU norm is a tree reduction
//! while the CPU's is a sequential sum, so the two normalized vectors differ in
//! the last bits — a coordinate landing within that epsilon of a Lloyd-Max
//! decision boundary would quantize differently on the two sides. With fixed
//! inputs that either happens or it doesn't, so test 1 can assert exact byte
//! equality without being flaky. (Were it to trip on a future `head_dim`, the
//! honest fix is to compare dequantized vectors, not to loosen the byte compare
//! into a mismatch budget.)
//!
//! ## Tolerances
//!
//! Tests 2 and 3 compare floats. Both sides run the same butterfly in the same
//! order, so the drift is confined to the reductions (tree vs sequential) and, in
//! test 3, to online vs batched softmax accumulation. `2e-4` on attention outputs
//! whose magnitude is ~1 is far above that and still catches a sign error, a
//! swapped centroid, a dropped QJL correction, or a missing inverse rotation.
//!
//! Gating: needs only a GPU adapter (wgpu → Metal/Vulkan/DX); skips cleanly if
//! none is available. Excluded on wasm32 — these drive the kernels through the
//! blocking `GpuContext::new()`, which is native-only.

#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use cera::backend::wgpu::{GpuContext, shaders};
use cera::model::gpu_turboquant::{TqAttnParams, TqGpuCache, TqLayout, TqMode, TqParams};
use cera::model::{BlockType, ModelConfig, ScalarMultipliers};
use cera::turboquant::{
    CompressedKeyCache, CompressedValueCache, EncodeScratch, QueryRotationScratch, RotationState,
    TurboQuantConfig, attn_scores_turboquant_gqa, attn_values_turboquant_gqa,
    compress_and_append_keys, compress_and_append_values, encode_compressed_keys,
    encode_compressed_values, rotate_queries,
};

const SEED: u64 = 0xC0FFEE;
/// Cache capacity in timesteps. Deliberately larger than `SEQ_LEN` so the
/// per-head region stride (`max_seq_len`, not `seq_len`) is actually exercised —
/// with the two equal, a kernel that confused them would still pass.
const MAX_SEQ: usize = 12;
const SEQ_LEN: usize = 7;
const N_KV_HEADS: usize = 2;
/// 4 query heads over 2 KV heads → GQA group_size 2, so the kernel's
/// `kv_head = head / group_size` mapping is under test.
const N_HEADS: usize = 4;
/// Non-zero so the params' `sign_off` (`layer * 2 * head_dim`) is exercised
/// rather than silently defaulting to the start of the signs buffer.
const LAYER: usize = 3;
const TOL: f32 = 2e-4;

/// Deterministic input generator — a plain LCG, so the test data is identical on
/// every platform and run (see the module note on determinism).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    /// Roughly uniform on [-1, 1).
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Everything both sides need: the layer's rotation, the quantizer config, and
/// the signs buffer laid out the way the kernels expect.
struct Fixture {
    head_dim: usize,
    rotation: RotationState,
    config: TurboQuantConfig,
    layout: TqLayout,
    /// `[polar | jl]` per layer for `LAYER + 1` layers; only `LAYER`'s slice is
    /// non-trivial for this test, but the earlier layers must be present so
    /// `sign_off` addresses the right place.
    signs: Vec<f32>,
    sign_off: usize,
}

impl Fixture {
    fn new(head_dim: usize) -> Self {
        // The engine seeds each layer as `seed ^ layer_idx`; mirror that exactly,
        // otherwise the CPU and GPU would rotate in different bases.
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
        TqParams {
            n_tokens: n_tokens as u32,
            n_heads: n_heads as u32,
            head_dim: self.head_dim as u32,
            src_stride: src_stride as u32,
            dst_pos: 0,
            max_seq_len: MAX_SEQ as u32,
            sign_off: self.sign_off as u32,
            q_cap: SEQ_LEN as u32,
            ..Default::default()
        }
        .with_quant_config(&self.config)
    }
}

/// Dispatch one of the three `turboquant.wgsl` entry points and return the output
/// buffer's words. `groups` is the workgroup count (one per (row, head)).
fn run_tq_kernel(
    ctx: &GpuContext,
    entry: &str,
    src: &[f32],
    out_words: usize,
    signs: &[f32],
    params: &TqParams,
    groups: usize,
) -> Vec<u32> {
    let pipeline = ctx.create_pipeline(shaders::TURBOQUANT, entry, entry);
    let src_buf = ctx.upload_f32(src, "src");
    let dst_buf = ctx.create_storage_rw((out_words * 4) as u64, "dst");
    let signs_buf = ctx.upload_f32(signs, "signs");
    let params_buf = ctx.upload_storage(bytemuck::bytes_of(params), "params");

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: signs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups as u32, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.download_u32(&dst_buf, out_words)
}

/// CPU-side compressed caches for `k` / `v` (`SEQ_LEN × kv_dim` row-major).
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

/// Test 1: the packed cache the encode kernels write is byte-identical to the
/// CPU's, for both keys and values.
fn check_encode(ctx: &GpuContext, head_dim: usize) {
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
            let cpu_polar = &cpu_keys.polar_data[h][t * pw * 4..(t + 1) * pw * 4];
            assert_eq!(
                gpu_polar, cpu_polar,
                "hd={head_dim} key polar mismatch at h={h} t={t}"
            );
            let gpu_jl: &[u8] =
                bytemuck::cast_slice(&words[jl_off + slot * jw..jl_off + slot * jw + jw]);
            let cpu_jl = &cpu_keys.jl_data[h][t * jw * 4..(t + 1) * jw * 4];
            assert_eq!(
                gpu_jl, cpu_jl,
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
            let cpu_polar = &cpu_values.polar_data[h][t * pw * 4..(t + 1) * pw * 4];
            assert_eq!(
                gpu_polar, cpu_polar,
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

/// Test 2: `tq_rotate_q` reproduces `rotate_queries` — both rotations and the
/// per-head `sum(q_jl)`.
fn check_rotate_q(ctx: &GpuContext, head_dim: usize) {
    let fx = Fixture::new(head_dim);
    let q_dim = N_HEADS * head_dim;
    let mut rng = Lcg::new(23);
    let q = rng.fill(SEQ_LEN * q_dim);

    let mut params = fx.encode_params(SEQ_LEN, N_HEADS, q_dim);
    params.q_cap = SEQ_LEN as u32;
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

/// Test 3: `flash_attention_tq` reproduces the CPU compressed read path for every
/// causal query row.
fn check_attention(ctx: &GpuContext, head_dim: usize) {
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

    // GPU: encode into the packed caches, rotate the queries, run attention.
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

    let attn_params = TqAttnParams {
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        head_dim: head_dim as u32,
        max_seq: SEQ_LEN as u32,
        start_pos: 0,
        scale,
        q_cap: SEQ_LEN as u32,
        out_stride: q_dim as u32,
        qjl_scale: TqAttnParams::qjl_scale_for(head_dim),
        sign_off: fx.sign_off as u32,
        centroids: fx.config.centroids,
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

    // CPU reference, per causal query row.
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
            // Batched softmax per query head over the causal window.
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

/// Dispatch `flash_attention_tq` and return the output floats.
#[allow(clippy::too_many_arguments)]
fn run_attention(
    ctx: &GpuContext,
    qrot: &[u32],
    keys: &[u32],
    values: &[u32],
    signs: &[f32],
    params: &TqAttnParams,
    out_floats: usize,
    n_heads: usize,
    n_queries: usize,
) -> Vec<f32> {
    let pipeline = ctx.create_pipeline(
        shaders::FLASH_ATTENTION_TQ,
        "flash_attention_tq",
        "flash_attention_tq",
    );
    let qrot_buf = ctx.upload_storage(bytemuck::cast_slice(qrot), "qrot");
    let k_buf = ctx.upload_storage(bytemuck::cast_slice(keys), "keys");
    let v_buf = ctx.upload_storage(bytemuck::cast_slice(values), "values");
    let out_buf = ctx.create_storage_rw((out_floats * 4) as u64, "out");
    let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params.to_u32_array()), "params");
    let signs_buf = ctx.upload_f32(signs, "signs");

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flash_attention_tq"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: qrot_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: signs_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flash_attention_tq"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_heads as u32, n_queries as u32, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.download_f32(&out_buf, out_floats)
}

/// Test 4: the prefix-cache snapshot of a GPU-resident compressed cache is
/// byte-identical to what the CPU backend would write for the same KV, and
/// restoring it reproduces the cache exactly.
///
/// Byte equality against `encode_compressed_keys` is the whole point: it is what
/// makes a snapshot written by one backend loadable by the other, and it means
/// the packed GPU layout really is the CPU layout plus a stride.
fn check_snapshot_roundtrip(ctx: &GpuContext, head_dim: usize) {
    let fx = Fixture::new(head_dim);
    let kv_dim = N_KV_HEADS * head_dim;
    let mut rng = Lcg::new(59);
    let k = rng.fill(SEQ_LEN * kv_dim);
    let v = rng.fill(SEQ_LEN * kv_dim);
    let (cpu_keys, cpu_values) = cpu_encode(&fx, &k, &v, kv_dim);

    // A config whose every layer is attention, so `LAYER` is a valid index and
    // each layer gets its own buffers to move data between.
    let n_layers = LAYER + 1;
    let config = ModelConfig {
        architecture: "lfm2".into(),
        n_layers,
        hidden_size: N_HEADS * head_dim,
        intermediate_size: N_HEADS * head_dim,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim,
        vocab_size: 32,
        max_seq_len: MAX_SEQ,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types: vec![BlockType::Attention; n_layers],
        conv_kernel_size: Some(3),
        kv_heads_per_layer: vec![N_KV_HEADS; n_layers],
        scalars: ScalarMultipliers::default(),
    };
    let tq = TqGpuCache::new(ctx, &config, MAX_SEQ, SEQ_LEN, TqMode { seed: SEED })
        .expect("TqGpuCache allocation");

    tq.write_params(ctx, &config, SEQ_LEN, 0, 1.0 / (head_dim as f32).sqrt());
    let k_buf = ctx.upload_f32(&k, "k_src");
    let v_buf = ctx.upload_f32(&v, "v_src");
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    tq.encode_kv(ctx, &mut enc, LAYER, &k_buf, &v_buf, SEQ_LEN);
    ctx.queue.submit(Some(enc.finish()));

    let (keys_blob, values_blob) = tq.snapshot_layer(ctx, LAYER, SEQ_LEN);
    assert_eq!(
        keys_blob,
        encode_compressed_keys(&cpu_keys),
        "hd={head_dim} TQK1 snapshot differs from the CPU encoding"
    );
    assert_eq!(
        values_blob,
        encode_compressed_values(&cpu_values),
        "hd={head_dim} TQV1 snapshot differs from the CPU encoding"
    );

    // Restore into a DIFFERENT layer's buffers and re-snapshot: equality proves
    // the upload lands at the same strided offsets the readback gathers from.
    assert_eq!(
        tq.restore_layer(ctx, 0, &keys_blob, &values_blob),
        Some(SEQ_LEN),
        "hd={head_dim} restore rejected a blob it wrote itself"
    );
    let (keys_again, values_again) = tq.snapshot_layer(ctx, 0, SEQ_LEN);
    assert_eq!(
        keys_again, keys_blob,
        "hd={head_dim} key restore round-trip"
    );
    assert_eq!(
        values_again, values_blob,
        "hd={head_dim} value restore round-trip"
    );

    // A malformed blob must be rejected rather than partially applied — the
    // caller turns `None` into a prefix-cache miss.
    assert_eq!(
        tq.restore_layer(ctx, 0, &keys_blob[..keys_blob.len() - 1], &values_blob),
        None,
        "hd={head_dim} truncated TQK1 blob was accepted"
    );
    assert_eq!(
        tq.restore_layer(ctx, 0, &values_blob, &keys_blob),
        None,
        "hd={head_dim} swapped key/value blobs were accepted"
    );

    // An empty cache still round-trips through a header-only blob.
    let (empty_k, empty_v) = tq.snapshot_layer(ctx, 0, 0);
    assert_eq!(tq.restore_layer(ctx, 0, &empty_k, &empty_v), Some(0));
}

fn assert_close(got: f32, want: f32, what: &str) {
    let diff = (got - want).abs();
    assert!(
        diff < TOL,
        "{what}: got={got} want={want} diff={diff} (tol={TOL})"
    );
}

/// `None` when no adapter is available, so the tests skip instead of failing on
/// a headless CI runner.
fn context() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no wgpu adapter ({e})");
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

#[test]
fn snapshot_roundtrips_and_matches_cpu_blobs() {
    let Some(ctx) = context() else { return };
    check_snapshot_roundtrip(&ctx, 64);
    check_snapshot_roundtrip(&ctx, 128);
}
